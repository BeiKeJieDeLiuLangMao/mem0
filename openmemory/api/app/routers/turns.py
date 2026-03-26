"""
Turn storage API router.

存储和查询原始对话 turn（用户-模型的完整交互历史）。
"""

import logging
from typing import List, Optional
from uuid import UUID

from app.database import get_db
from app.models import Turn
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/turns", tags=["turns"])


class TurnCreate(BaseModel):
    """创建 Turn 的请求模型"""
    session_id: str
    user_id: str
    agent_id: Optional[str] = None
    messages: List[dict]
    message_count: int = 0
    tool_call_count: int = 0
    total_tokens: int = 0
    source: str = "openclaw"  # 默认值


class TurnResponse(BaseModel):
    """Turn 响应模型"""
    id: str
    session_id: str
    user_id: str
    agent_id: Optional[str]
    messages: List[dict]
    source: str
    created_at: str
    message_count: int
    tool_call_count: int
    total_tokens: int


class TurnListResponse(BaseModel):
    """Turn 列表响应模型"""
    turns: List[TurnResponse]
    total: int


def _turn_to_response(turn: Turn) -> TurnResponse:
    """将 Turn 模型转换为响应模型"""
    return TurnResponse(
        id=str(turn.id),
        session_id=turn.session_id,
        user_id=turn.user_id,
        agent_id=turn.agent_id,
        messages=turn.messages,
        source=turn.source,
        created_at=turn.created_at.isoformat(),
        message_count=turn.message_count,
        tool_call_count=turn.tool_call_count,
        total_tokens=turn.total_tokens,
    )


@router.post("/", response_model=TurnResponse)
async def create_turn(
    request: TurnCreate,
    db: Session = Depends(get_db)
):
    """
    存储原始 turn 到数据库

    接收完整的用户-模型交互历史，包括所有消息和工具调用。
    """
    try:
        turn = Turn(**request.model_dump())
        db.add(turn)
        db.commit()
        db.refresh(turn)

        logger.info(f"Created turn {turn.id} for session {request.session_id}")
        return _turn_to_response(turn)
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create turn: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create turn: {e}")


@router.get("/{turn_id}", response_model=TurnResponse)
async def get_turn(
    turn_id: UUID,
    db: Session = Depends(get_db)
):
    """获取单个 turn 的完整消息"""
    turn = db.query(Turn).filter(Turn.id == turn_id).first()
    if not turn:
        raise HTTPException(status_code=404, detail="Turn not found")
    return _turn_to_response(turn)


@router.get("/", response_model=TurnListResponse)
async def list_turns(
    user_id: str = Query(..., description="User identifier"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    limit: int = Query(50, ge=1, le=200, description="Max turns to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """列出 turns（支持按 session/agent 过滤）"""
    query = db.query(Turn).filter(Turn.user_id == user_id)

    if session_id:
        query = query.filter(Turn.session_id == session_id)
    if agent_id:
        query = query.filter(Turn.agent_id == agent_id)

    # 获取总数
    total = query.count()

    # 分页查询
    turns = query.order_by(Turn.created_at.desc()).offset(offset).limit(limit).all()

    return TurnListResponse(
        turns=[_turn_to_response(turn) for turn in turns],
        total=total
    )


class TurnBatchRequest(BaseModel):
    """批量获取 Turn 的请求模型"""
    ids: List[str]


@router.post("/batch", response_model=List[TurnResponse])
async def get_turns_batch(
    request: TurnBatchRequest,
    db: Session = Depends(get_db)
):
    """批量获取 turns"""
    try:
        # 将字符串 ID 转换为 UUID
        uuids = [UUID(id) for id in request.ids]
        turns = db.query(Turn).filter(Turn.id.in_(uuids)).all()
        return [_turn_to_response(turn) for turn in turns]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid UUID format: {e}")
    except Exception as e:
        logger.error(f"Failed to batch get turns: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get turns: {e}")


@router.delete("/{turn_id}")
async def delete_turn(
    turn_id: UUID,
    db: Session = Depends(get_db)
):
    """删除 turn"""
    turn = db.query(Turn).filter(Turn.id == turn_id).first()
    if not turn:
        raise HTTPException(status_code=404, detail="Turn not found")

    try:
        db.delete(turn)
        db.commit()
        return {"status": "deleted", "id": str(turn_id)}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete turn: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete turn: {e}")
