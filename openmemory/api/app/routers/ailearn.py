"""AI Learning router for OpenMemory API.

Provides endpoints for:
- Starting/stopping background learning
- Querying learning status and progress
- Viewing detected patterns and skills
- Managing amendment proposals

Enhanced features:
- Turn-aware pattern detection
- LLM-powered skill extraction
- Cross-session analysis
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import List, Optional
from uuid import UUID, uuid4

from app.database import get_db
from app.models import Turn, User
from app.utils.db import get_db_session
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ailearn", tags=["ailearn"])

# ------ Global AI Learning Instance ------
_ailearn_instance = None
_background_task = None
_llm_client = None  # Will be initialized from server.py config


def get_llm_client():
    """Get LLM client from configuration."""
    global _llm_client
    if _llm_client is None:
        import os
        from openai import OpenAI

        # Use same configuration as server.py
        api_key = os.environ.get("OPENAI_API_KEY", "fai-2-977-cdc6435fbca2")
        base_url = os.environ.get(
            "OPENAI_BASE_URL",
            "https://trip-llm.alibaba-inc.com/api/openai/v1"
        )

        _llm_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    return _llm_client


async def fetch_recent_turns(
    db: Session,
    limit: int = 100,
) -> List[dict]:
    """Fetch recent turns from database."""
    try:
        turns = db.query(Turn).order_by(Turn.created_at.desc()).limit(limit).all()

        return [
            {
                "id": str(turn.id),
                "session_id": turn.session_id,
                "user_id": turn.user_id,
                "agent_id": turn.agent_id,
                "messages": turn.messages,
                "source": turn.source,
                "created_at": turn.created_at.isoformat() if turn.created_at else None,
                "message_count": turn.message_count,
                "tool_call_count": turn.tool_call_count,
                "total_tokens": turn.total_tokens,
            }
            for turn in turns
        ]
    except Exception as e:
        logger.error(f"Failed to fetch turns: {e}")
        return []

class AILearnStatus(BaseModel):
    """Current status of AI Learning system."""
    is_running: bool
    observations_count: int
    patterns_detected: int
    skills_extracted: int
    amendments_proposed: int
    health_status: str
    last_analysis: Optional[datetime] = None
    next_analysis: Optional[datetime] = None


class PatternInfo(BaseModel):
    """A detected pattern."""
    id: str
    pattern_type: str
    name: str
    description: str
    confidence: float
    frequency: int
    extracted_at: datetime


class SkillInfo(BaseModel):
    """An extracted skill."""
    id: str
    name: str
    description: str
    trigger_phrases: List[str]
    confidence: float
    extracted_at: datetime


class AmendmentInfo(BaseModel):
    """An amendment proposal."""
    id: str
    amendment_type: str
    memory_id: str
    reasoning: str
    confidence: float
    expected_impact: float
    created_at: datetime


class AILearnConfig(BaseModel):
    """AI Learning configuration."""
    enabled: bool = True
    auto_learn_interval_minutes: int = Field(default=5, ge=1, le=60)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_observations_per_batch: int = Field(default=1000, ge=100, le=10000)


class AILearnStartResponse(BaseModel):
    """Response when starting AI Learning."""
    message: str
    config: AILearnConfig


# ------ Background Task ------

async def background_learning_task(
    interval_minutes: int,
    max_observations: int,
    confidence_threshold: float,
):
    """Background task that runs learning analysis periodically."""
    global _ailearn_instance

    logger.info(f"Starting Enhanced AI Learning background task (interval: {interval_minutes}m)")

    while _ailearn_instance is not None:
        try:
            logger.info("Running Enhanced AI Learning analysis...")

            # Fetch recent turns from database
            with get_db_session() as db:
                turns = await fetch_recent_turns(db, limit=100)
                logger.info(f"Fetched {len(turns)} turns for analysis")

            # Import here to avoid circular imports
            from mem0.ailearn.enhanced import EnhancedAILearn

            if not isinstance(_ailearn_instance, EnhancedAILearn):
                logger.warning(f"Invalid ailearn instance type: {type(_ailearn_instance)}")
                break

            # Run analysis with turns
            results = await _ailearn_instance.analyze_and_learn(
                turns=turns,
                limit=max_observations,
            )

            logger.info(
                f"Enhanced AI Learning analysis complete: "
                f"{results['observations_analyzed']} observations, "
                f"{results['turns_analyzed']} turns, "
                f"{results['patterns_detected']} patterns, "
                f"{results['skills_extracted']} skills, "
                f"{results['amendments_proposed']} amendments"
            )

        except Exception as e:
            logger.error(f"Error in Enhanced AI Learning background task: {e}", exc_info=True)

        # Wait for next interval
        await asyncio.sleep(interval_minutes * 60)

    logger.info("Enhanced AI Learning background task stopped")


# ------ API Endpoints ------

@router.get("/status", response_model=AILearnStatus)
async def get_ailearn_status(
    db: Session = Depends(get_db),
):
    """Get current AI Learning status."""
    global _ailearn_instance

    if _ailearn_instance is None:
        return AILearnStatus(
            is_running=False,
            observations_count=0,
            patterns_detected=0,
            skills_extracted=0,
            amendments_proposed=0,
            health_status="disabled",
        )

    try:
        # Get health status
        health = await _ailearn_instance.get_health_status()

        # Get observation count
        observations = await _ailearn_instance.observation_store.get_by_project(
            _ailearn_instance.project_id,
            limit=1000000,  # Large number to get all
        )

        # Get patterns (simplified - in real implementation, query from pattern store)
        patterns_count = 0  # TODO: Implement pattern storage

        return AILearnStatus(
            is_running=True,
            observations_count=len(observations),
            patterns_detected=patterns_count,
            skills_extracted=0,  # TODO: Implement skill storage
            amendments_proposed=0,  # TODO: Implement amendment storage
            health_status=health.get("status", "unknown"),
            last_analysis=datetime.now(UTC),  # TODO: Track actual last analysis time
        )
    except Exception as e:
        logger.error(f"Error getting AI Learning status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=AILearnStartResponse)
async def start_ailearn(
    config: AILearnConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Start Enhanced AI Learning background task."""
    global _ailearn_instance, _background_task

    if _ailearn_instance is not None:
        raise HTTPException(status_code=400, detail="AI Learning is already running")

    try:
        # Import here to avoid circular imports
        from mem0.ailearn.enhanced import EnhancedAILearn

        # Get LLM client
        llm_client = get_llm_client()

        # Initialize Enhanced AI Learning
        _ailearn_instance = EnhancedAILearn(
            storage_path="~/.mem0/ailearn",
            project_id="openmemory",
            auto_learn=config.enabled,
            llm_client=llm_client,
            llm_model="gpt-4.1-nano-2025-04-14",
        )

        # Start background task
        _background_task = asyncio.create_task(
            background_learning_task(
                interval_minutes=config.auto_learn_interval_minutes,
                max_observations=config.max_observations_per_batch,
                confidence_threshold=config.confidence_threshold,
            )
        )

        logger.info("Enhanced AI Learning started successfully")

        return AILearnStartResponse(
            message="Enhanced AI Learning started successfully",
            config=config,
        )
    except Exception as e:
        logger.error(f"Error starting Enhanced AI Learning: {e}", exc_info=True)
        _ailearn_instance = None
        _background_task = None
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_ailearn():
    """Stop AI Learning background task."""
    global _ailearn_instance, _background_task

    if _ailearn_instance is None:
        raise HTTPException(status_code=400, detail="AI Learning is not running")

    try:
        # Shutdown ailearn
        await _ailearn_instance.shutdown()

        # Cancel background task
        if _background_task:
            _background_task.cancel()
            try:
                await _background_task
            except asyncio.CancelledError:
                pass

        _ailearn_instance = None
        _background_task = None

        logger.info("AI Learning stopped successfully")

        return {"message": "AI Learning stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping AI Learning: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def trigger_analysis(
    db: Session = Depends(get_db),
):
    """Manually trigger a learning analysis with recent turns."""
    global _ailearn_instance

    if _ailearn_instance is None:
        raise HTTPException(status_code=400, detail="AI Learning is not running")

    try:
        # Fetch recent turns
        turns = await fetch_recent_turns(db, limit=100)

        # Run analysis
        results = await _ailearn_instance.analyze_and_learn(
            turns=turns,
            limit=1000,
        )
        return results
    except Exception as e:
        logger.error(f"Error triggering analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns", response_model=List[PatternInfo])
async def get_patterns(
    limit: int = Query(10, ge=1, le=100),
):
    """Get detected patterns."""
    # TODO: Implement pattern storage and retrieval
    return []


@router.get("/skills", response_model=List[SkillInfo])
async def get_skills(
    limit: int = Query(10, ge=1, le=100),
):
    """Get extracted skills."""
    # TODO: Implement skill storage and retrieval
    return []


@router.get("/amendments", response_model=List[AmendmentInfo])
async def get_amendments(
    limit: int = Query(10, ge=1, le=100),
):
    """Get amendment proposals."""
    # TODO: Implement amendment storage and retrieval
    return []


@router.post("/amendments/{amendment_id}/apply")
async def apply_amendment(
    amendment_id: str,
):
    """Apply an approved amendment proposal."""
    global _ailearn_instance

    if _ailearn_instance is None:
        raise HTTPException(status_code=400, detail="AI Learning is not running")

    try:
        # TODO: Implement amendment application
        return {"message": f"Amendment {amendment_id} applied successfully"}
    except Exception as e:
        logger.error(f"Error applying amendment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/amendments/{amendment_id}")
async def reject_amendment(
    amendment_id: str,
):
    """Reject an amendment proposal."""
    # TODO: Implement amendment rejection
    return {"message": f"Amendment {amendment_id} rejected"}
