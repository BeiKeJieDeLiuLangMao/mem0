import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from app.database import get_db
from app.models import (
    AccessControl,
    App,
    Category,
    Memory,
    MemoryAccessLog,
    MemoryState,
    MemoryStatusHistory,
    User,
)
from app.schemas import MemoryResponse
from app.utils.memory import get_memory_client
from app.utils.permissions import check_memory_access_permissions
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi_pagination import Page, Params
from fastapi_pagination.ext.sqlalchemy import paginate as sqlalchemy_paginate
from pydantic import BaseModel, field_validator, model_validator
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

router = APIRouter(prefix="/api/v1/memories", tags=["memories"])


def get_memory_or_404(db: Session, memory_id: UUID) -> Memory:
    memory = db.query(Memory).filter(Memory.id == memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


def update_memory_state(db: Session, memory_id: UUID, new_state: MemoryState, user_id: UUID):
    memory = get_memory_or_404(db, memory_id)
    old_state = memory.state

    # Update memory state
    memory.state = new_state
    if new_state == MemoryState.archived:
        memory.archived_at = datetime.now(UTC)
    elif new_state == MemoryState.deleted:
        memory.deleted_at = datetime.now(UTC)

    # Record state change
    history = MemoryStatusHistory(
        memory_id=memory_id,
        changed_by=user_id,
        old_state=old_state,
        new_state=new_state
    )
    db.add(history)
    db.commit()
    return memory


def get_accessible_memory_ids(db: Session, app_id: UUID) -> Set[UUID]:
    """
    Get the set of memory IDs that the app has access to based on app-level ACL rules.
    Returns all memory IDs if no specific restrictions are found.
    """
    # Get app-level access controls
    app_access = db.query(AccessControl).filter(
        AccessControl.subject_type == "app",
        AccessControl.subject_id == app_id,
        AccessControl.object_type == "memory"
    ).all()

    # If no app-level rules exist, return None to indicate all memories are accessible
    if not app_access:
        return None

    # Initialize sets for allowed and denied memory IDs
    allowed_memory_ids = set()
    denied_memory_ids = set()

    # Process app-level rules
    for rule in app_access:
        if rule.effect == "allow":
            if rule.object_id:  # Specific memory access
                allowed_memory_ids.add(rule.object_id)
            else:  # All memories access
                return None  # All memories allowed
        elif rule.effect == "deny":
            if rule.object_id:  # Specific memory denied
                denied_memory_ids.add(rule.object_id)
            else:  # All memories denied
                return set()  # No memories accessible

    # Remove denied memories from allowed set
    if allowed_memory_ids:
        allowed_memory_ids -= denied_memory_ids

    return allowed_memory_ids


class QdrantMemoryResponse(BaseModel):
    """Memory from Qdrant"""
    id: str
    content: str
    memory_type: str  # "summary" or "fact"
    turn_id: Optional[str] = None
    agent_id: Optional[str] = None  # Extracted from metadata
    source: Optional[str] = None  # Source from turn table
    score: Optional[float] = None
    metadata: dict = {}
    created_at: Optional[str] = None


class QdrantMemoryListResponse(BaseModel):
    """List of memories from Qdrant"""
    items: List[QdrantMemoryResponse]
    total: int
    page: int
    size: int


# List all memories with filtering (from Qdrant)
@router.get("/", response_model=QdrantMemoryListResponse)
async def list_memories(
    user_id: str,
    agent_id: Optional[str] = Query(None, description="Filter by agent_id"),
    memory_type: Optional[str] = Query(None, description="Filter by memory_type: summary or fact"),
    turn_id: Optional[str] = Query(None, description="Filter by turn_id"),
    limit: int = Query(200, ge=1, le=1000, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
):
    """
    List memories from Qdrant with optional filters.

    Returns both summary and fact memories, grouped by turn_id when available.
    """
    try:
        # Direct Qdrant query to avoid embedding API calls
        from qdrant_client import QdrantClient

        qdrant_client = QdrantClient(host="127.0.0.1", port=6333)

        # Scroll through Qdrant to get all memories (without filter for now)
        results = []
        current_offset = None
        scroll_limit = limit + offset  # Use local variable to avoid conflict
        while len(results) < scroll_limit:
            records, next_offset = qdrant_client.scroll(
                collection_name="memories",
                limit=scroll_limit,
                offset=current_offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break
            for record in records:
                payload = record.payload or {}
                # Filter by user_id in code (support both userId and user_id)
                record_user_id = payload.get("user_id") or payload.get("userId", "")
                if not record_user_id.startswith(user_id):
                    continue

                # Apply optional filters
                if memory_type and payload.get("metadata", {}).get("memory_type") != memory_type:
                    continue
                if turn_id and payload.get("metadata", {}).get("turn_id") != turn_id:
                    continue
                if agent_id:
                    # Extract agent_id from metadata (support both agentId and userId)
                    record_agent_id = payload.get("metadata", {}).get("agent_id") or \
                                     payload.get("agentId") or \
                                     payload.get("userId", "")
                    # Check if agent_id matches (support legacy format: yishu:agent:{agent_id})
                    if record_agent_id != agent_id and not record_agent_id.endswith(f":agent:{agent_id}"):
                        continue

                results.append({
                    "id": str(record.id),
                    "memory": payload.get("data", ""),
                    "metadata": payload,
                    "score": None,
                })
            if next_offset is None:
                break
            current_offset = next_offset

        # Apply pagination
        paginated_results = results[offset:offset + limit]

        # Transform to response format
        items = []
        for r in paginated_results:
            # Extract metadata
            metadata = r.get("metadata", {})
            memory_type = metadata.get("memory_type", "fact")
            turn_id = metadata.get("turn_id")

            # Extract agent_id from metadata (handle both agentId and userId formats)
            agent_id = metadata.get("agentId")
            if not agent_id:
                # Try to extract from userId (format: yishu:agent:xxx)
                user_id_val = metadata.get("userId", "")
                if ":agent:" in user_id_val:
                    agent_id = user_id_val.split(":agent:")[-1]

            # 查询 source 信息（如果有 turn_id）
            source = None
            if turn_id:
                try:
                    from app.models import Turn
                    from sqlalchemy import select
                    turn = db.execute(select(Turn.source).where(Turn.id == UUID(turn_id))).scalar()
                    source = turn
                except:
                    source = None
            else:
                source = "manual"  # 没有 turn_id 的记忆是手动添加的

            items.append(QdrantMemoryResponse(
                id=r.get("id", ""),
                content=r.get("memory", ""),
                memory_type=memory_type,
                turn_id=turn_id,
                agent_id=agent_id,  # Add extracted agent_id
                source=source,     # Add source from turn
                score=r.get("score"),
                metadata=metadata,
                created_at=metadata.get("createdAt"),
            ))

        return QdrantMemoryListResponse(
            items=items,
            total=len(results),  # Total count before pagination
            page=(offset // limit) + 1 if limit > 0 else 1,
            size=limit,
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to list memories from Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list memories: {str(e)}")


# Legacy SQLite-based list endpoint (deprecated, kept for compatibility)
@router.get("/legacy", response_model=Page[MemoryResponse], include_in_schema=False)
async def list_memories_legacy(
    user_id: str,
    app_id: Optional[UUID] = None,
    from_date: Optional[int] = Query(
        None,
        description="Filter memories created after this date (timestamp)",
        examples=[1718505600]
    ),
    to_date: Optional[int] = Query(
        None,
        description="Filter memories created before this date (timestamp)",
        examples=[1718505600]
    ),
    categories: Optional[str] = None,
    params: Params = Depends(),
    search_query: Optional[str] = None,
    sort_column: Optional[str] = Query(None, description="Column to sort by (memory, categories, app_name, created_at)"),
    sort_direction: Optional[str] = Query(None, description="Sort direction (asc or desc)"),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Build base query
    query = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted,
        Memory.state != MemoryState.archived,
        Memory.content.ilike(f"%{search_query}%") if search_query else True
    )

    # Apply filters
    if app_id:
        query = query.filter(Memory.app_id == app_id)

    if from_date:
        from_datetime = datetime.fromtimestamp(from_date, tz=UTC)
        query = query.filter(Memory.created_at >= from_datetime)

    if to_date:
        to_datetime = datetime.fromtimestamp(to_date, tz=UTC)
        query = query.filter(Memory.created_at <= to_datetime)

    # Add joins for app and categories after filtering
    query = query.outerjoin(App, Memory.app_id == App.id)
    query = query.outerjoin(Memory.categories)

    # Apply category filter if provided
    if categories:
        category_list = [c.strip() for c in categories.split(",")]
        query = query.filter(Category.name.in_(category_list))

    # Apply sorting if specified
    if sort_column:
        sort_field = getattr(Memory, sort_column, None)
        if sort_field:
            query = query.order_by(sort_field.desc()) if sort_direction == "desc" else query.order_by(sort_field.asc())

    # Add eager loading for app and categories
    query = query.options(
        joinedload(Memory.app),
        joinedload(Memory.categories)
    ).distinct(Memory.id)

    # Get paginated results with transformer
    return sqlalchemy_paginate(
        query,
        params,
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=memory.metadata_
            )
            for memory in items
            if check_memory_access_permissions(db, memory, app_id)
        ]
    )


# Get all categories
@router.get("/categories")
async def get_categories(
    user_id: str,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get unique categories associated with the user's memories
    # Get all memories
    memories = db.query(Memory).filter(Memory.user_id == user.id, Memory.state != MemoryState.deleted, Memory.state != MemoryState.archived).all()
    # Get all categories from memories
    categories = [category for memory in memories for category in memory.categories]
    # Get unique categories
    unique_categories = list(set(categories))

    return {
        "categories": unique_categories,
        "total": len(unique_categories)
    }


class CreateMemoryRequest(BaseModel):
    user_id: str
    text: Optional[str] = None  # Deprecated: use messages instead
    messages: Optional[List[dict]] = None  # New: support full conversation format
    metadata: dict = {}
    infer: bool = True
    app: str = "openmemory"
    memory_type: str = "fact"  # "summary" or "fact"
    turn_id: Optional[str] = None  # Link to original turn
    agent_id: Optional[str] = None  # Agent identifier

    @model_validator(mode='after')
    def validate_text_or_messages(self):
        """Ensure either text or messages is provided."""
        if not self.text and not self.messages:
            raise ValueError("Either 'text' or 'messages' must be provided")
        return self


# Create new memory
@router.post("/")
async def create_memory(
    request: CreateMemoryRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Get or create app
    app_obj = db.query(App).filter(App.name == request.app,
                                   App.owner_id == user.id).first()
    if not app_obj:
        app_obj = App(name=request.app, owner_id=user.id)
        db.add(app_obj)
        db.commit()
        db.refresh(app_obj)

    # Check if app is active
    if not app_obj.is_active:
        raise HTTPException(status_code=403, detail=f"App {request.app} is currently paused on OpenMemory. Cannot create new memories.")

    # Validate input: either text or messages must be provided
    if not request.text and not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Either 'text' or 'messages' must be provided"
        )

    # Convert text to messages format for backward compatibility
    if request.text and not request.messages:
        messages = [{"role": "user", "content": request.text}]
    else:
        messages = request.messages

    # Log what we're about to do
    logging.info(f"Creating memory for user_id: {request.user_id}, agent_id: {request.agent_id} with app: {request.app}")
    logging.info(f"Messages count: {len(messages)}, infer: {request.infer}")

    # Try to get memory client safely
    memory_client = None
    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise Exception("Memory client is not available")
    except Exception as client_error:
        logging.warning(f"Memory client unavailable: {client_error}. Creating memory in database only.")

    # Try to save to Qdrant via memory_client
    # If Qdrant fails, we still save to database as a fallback
    created_memories = []
    qdrant_memory_id = None

    if memory_client:
        try:
            # Build metadata for mem0
            mem0_metadata = {
                "source_app": "openmemory",
                "mcp_client": request.app,
                "memory_type": request.memory_type,
                "turn_id": request.turn_id,
            }
            if request.agent_id:
                mem0_metadata["agent_id"] = request.agent_id

            # Use mem0's add() method with proper messages format
            qdrant_response = memory_client.add(
                messages,  # Pass messages directly
                user_id=request.user_id,
                agent_id=request.agent_id,
                metadata=mem0_metadata,
                infer=request.infer
            )

            # Log the response for debugging
            logging.info(f"Qdrant response: {qdrant_response}")
            logging.info(f"Qdrant response type: {type(qdrant_response)}")

            # Process Qdrant response
            if isinstance(qdrant_response, dict) and 'results' in qdrant_response:
                for result in qdrant_response['results']:
                    logging.info(f"Processing result: {result}")
                    logging.info(f"Result keys: {result.keys()}")
                    logging.info(f"Result.get('memory'): {result.get('memory')}")
                    logging.info(f"Result.get('text'): {result.get('text')}")

                    if result.get('event') == 'ADD':
                        memory_id_str = result.get('id')
                        if not memory_id_str:
                            logging.warning(f"Skipping memory without id: {result}")
                            continue

                        qdrant_memory_id = UUID(memory_id_str)
                        memory_content = result.get('memory') or result.get('text')

                        if not memory_content:
                            logging.warning(f"Memory {memory_id_str} has no content, skipping")
                            continue

                        # Check if memory already exists
                        existing_memory = db.query(Memory).filter(Memory.id == qdrant_memory_id).first()

                        if existing_memory:
                            # Update existing memory
                            existing_memory.state = MemoryState.active
                            existing_memory.content = memory_content
                            memory = existing_memory
                        else:
                            # Create memory with the EXACT SAME ID from Qdrant
                            memory = Memory(
                                id=qdrant_memory_id,
                                user_id=user.id,
                                app_id=app_obj.id,
                                content=memory_content,
                                metadata_=request.metadata,
                                state=MemoryState.active
                            )
                            db.add(memory)

                        # Create history entry
                        history = MemoryStatusHistory(
                            memory_id=qdrant_memory_id,
                            changed_by=user.id,
                            old_state=MemoryState.deleted if existing_memory else MemoryState.deleted,
                            new_state=MemoryState.active
                        )
                        db.add(history)

                        created_memories.append(memory)
        except Exception as qdrant_error:
            import traceback
            logging.warning(f"Qdrant operation failed: {qdrant_error}. Falling back to database only.")
            logging.warning(f"Qdrant error traceback: {traceback.format_exc()}")

    # If Qdrant failed or memory_client is not available, save directly to database
    if not created_memories:
        logging.info("Saving memory directly to database (Qdrant unavailable or failed)")
        # Extract content: prefer text field, fallback to messages[0].content
        fallback_content = request.text
        if not fallback_content and request.messages:
            fallback_content = request.messages[0].get("content", "") if request.messages else ""
        if not fallback_content:
            raise HTTPException(
                status_code=400,
                detail="No content provided: text and messages are both empty"
            )
        # Create a simple memory without fact extraction
        memory = Memory(
            user_id=user.id,
            app_id=app_obj.id,
            content=fallback_content,
            metadata_=request.metadata,
            state=MemoryState.active
        )
        db.add(memory)
        created_memories.append(memory)

    # Commit all changes at once
    if created_memories:
        db.commit()
        for memory in created_memories:
            db.refresh(memory)

        # Return the first memory
        return created_memories[0]

    # Should not reach here, but return error if somehow we get here
    return {"error": "Failed to create memory"}


# Semantic vector search
class SemanticSearchResponse(BaseModel):
    items: List[QdrantMemoryResponse] = []
    total: int = 0


@router.get("/search", response_model=SemanticSearchResponse)
async def search_memories_semantic(
    user_id: str = Query(..., description="User identifier"),
    query: str = Query(..., min_length=1, description="Semantic search query"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    limit: int = Query(10, ge=1, le=100, description="Max results"),
    threshold: float = Query(0.0, ge=0.0, le=1.0, description="Min score threshold"),
    db: Session = Depends(get_db),
):
    """
    Semantic vector search over memories using the configured embedder + vector store.

    Embeds the query using the same model used for memory storage, then performs
    a cosine similarity search in Qdrant filtered by user_id (and optionally agent_id).
    Returns memories with similarity scores.
    """
    memory_client = get_memory_client()

    filters: Dict[str, Any] = {"user_id": user_id}
    if agent_id:
        filters["agent_id"] = agent_id

    try:
        embeddings = memory_client.embedding_model.embed(query, "search")
        hits = memory_client.vector_store.search(
            query=query,
            vectors=embeddings,
            limit=limit,
            filters=filters,
        )
    except Exception as e:
        logging.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    results = []
    for h in hits:
        if h.score < threshold:
            continue
        payload = h.payload or {}
        results.append(
            QdrantMemoryResponse(
                id=str(h.id) if h.id else None,
                content=payload.get("data") or payload.get("text") or "",
                memory_type=payload.get("memory_type"),
                agent_id=payload.get("agent_id"),
                source=payload.get("source_app"),
                score=h.score,
                created_at=payload.get("created_at"),
                metadata_=payload,
            )
        )

    return SemanticSearchResponse(items=results, total=len(results))


# Get memory by ID
@router.get("/{memory_id}")
async def get_memory(
    memory_id: UUID,
    db: Session = Depends(get_db)
):
    memory = get_memory_or_404(db, memory_id)
    return {
        "id": memory.id,
        "text": memory.content,
        "created_at": int(memory.created_at.timestamp()),
        "state": memory.state.value,
        "app_id": memory.app_id,
        "app_name": memory.app.name if memory.app else None,
        "categories": [category.name for category in memory.categories],
        "metadata_": memory.metadata_
    }


class DeleteMemoriesRequest(BaseModel):
    memory_ids: List[UUID]
    user_id: str

# Delete multiple memories
@router.delete("/")
async def delete_memories(
    request: DeleteMemoriesRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get memory client to delete from vector store
    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise HTTPException(
                status_code=503,
                detail="Memory client is not available"
            )
    except HTTPException:
        raise
    except Exception as client_error:
        logging.error(f"Memory client initialization failed: {client_error}")
        raise HTTPException(
            status_code=503,
            detail=f"Memory service unavailable: {str(client_error)}"
        )

    # Delete from vector store then mark as deleted in database
    for memory_id in request.memory_ids:
        try:
            memory_client.delete(str(memory_id))
        except Exception as delete_error:
            logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

        update_memory_state(db, memory_id, MemoryState.deleted, user.id)

    return {"message": f"Successfully deleted {len(request.memory_ids)} memories"}


# Archive memories
@router.post("/actions/archive")
async def archive_memories(
    memory_ids: List[UUID],
    user_id: UUID,
    db: Session = Depends(get_db)
):
    for memory_id in memory_ids:
        update_memory_state(db, memory_id, MemoryState.archived, user_id)
    return {"message": f"Successfully archived {len(memory_ids)} memories"}


class PauseMemoriesRequest(BaseModel):
    memory_ids: Optional[List[UUID]] = None
    category_ids: Optional[List[UUID]] = None
    app_id: Optional[UUID] = None
    all_for_app: bool = False
    global_pause: bool = False
    state: Optional[MemoryState] = None
    user_id: str

# Pause access to memories
@router.post("/actions/pause")
async def pause_memories(
    request: PauseMemoriesRequest,
    db: Session = Depends(get_db)
):
    
    global_pause = request.global_pause
    all_for_app = request.all_for_app
    app_id = request.app_id
    memory_ids = request.memory_ids
    category_ids = request.category_ids
    state = request.state or MemoryState.paused

    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_id = user.id
    
    if global_pause:
        # Pause all memories
        memories = db.query(Memory).filter(
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": "Successfully paused all memories"}

    if app_id:
        # Pause all memories for an app
        memories = db.query(Memory).filter(
            Memory.app_id == app_id,
            Memory.user_id == user.id,
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": f"Successfully paused all memories for app {app_id}"}
    
    if all_for_app and memory_ids:
        # Pause all memories for an app
        memories = db.query(Memory).filter(
            Memory.user_id == user.id,
            Memory.state != MemoryState.deleted,
            Memory.id.in_(memory_ids)
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": "Successfully paused all memories"}

    if memory_ids:
        # Pause specific memories
        for memory_id in memory_ids:
            update_memory_state(db, memory_id, state, user_id)
        return {"message": f"Successfully paused {len(memory_ids)} memories"}

    if category_ids:
        # Pause memories by category
        memories = db.query(Memory).join(Memory.categories).filter(
            Category.id.in_(category_ids),
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": f"Successfully paused memories in {len(category_ids)} categories"}

    raise HTTPException(status_code=400, detail="Invalid pause request parameters")


# Get memory access logs
@router.get("/{memory_id}/access-log")
async def get_memory_access_log(
    memory_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    query = db.query(MemoryAccessLog).filter(MemoryAccessLog.memory_id == memory_id)
    total = query.count()
    logs = query.order_by(MemoryAccessLog.accessed_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

    # Get app name
    for log in logs:
        app = db.query(App).filter(App.id == log.app_id).first()
        log.app_name = app.name if app else None

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "logs": logs
    }


class UpdateMemoryRequest(BaseModel):
    memory_content: str
    user_id: str

# Update a memory
@router.put("/{memory_id}")
async def update_memory(
    memory_id: UUID,
    request: UpdateMemoryRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    memory = get_memory_or_404(db, memory_id)
    memory.content = request.memory_content
    db.commit()
    db.refresh(memory)
    return memory

class FilterMemoriesRequest(BaseModel):
    user_id: str
    page: int = 1
    size: int = 10
    search_query: Optional[str] = None
    app_ids: Optional[List[UUID]] = None
    category_ids: Optional[List[UUID]] = None
    sort_column: Optional[str] = None
    sort_direction: Optional[str] = None
    from_date: Optional[int] = None
    to_date: Optional[int] = None
    show_archived: Optional[bool] = False

@router.post("/filter", response_model=Page[MemoryResponse])
async def filter_memories(
    request: FilterMemoriesRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Build base query
    query = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted,
    )

    # Filter archived memories based on show_archived parameter
    if not request.show_archived:
        query = query.filter(Memory.state != MemoryState.archived)

    # Apply search filter
    if request.search_query:
        query = query.filter(Memory.content.ilike(f"%{request.search_query}%"))

    # Apply app filter
    if request.app_ids:
        query = query.filter(Memory.app_id.in_(request.app_ids))

    # Add joins for app and categories
    query = query.outerjoin(App, Memory.app_id == App.id)

    # Apply category filter
    if request.category_ids:
        query = query.join(Memory.categories).filter(Category.id.in_(request.category_ids))
    else:
        query = query.outerjoin(Memory.categories)

    # Apply date filters
    if request.from_date:
        from_datetime = datetime.fromtimestamp(request.from_date, tz=UTC)
        query = query.filter(Memory.created_at >= from_datetime)

    if request.to_date:
        to_datetime = datetime.fromtimestamp(request.to_date, tz=UTC)
        query = query.filter(Memory.created_at <= to_datetime)

    # Apply sorting
    if request.sort_column and request.sort_direction:
        sort_direction = request.sort_direction.lower()
        if sort_direction not in ['asc', 'desc']:
            raise HTTPException(status_code=400, detail="Invalid sort direction")

        sort_mapping = {
            'memory': Memory.content,
            'app_name': App.name,
            'created_at': Memory.created_at
        }

        if request.sort_column not in sort_mapping:
            raise HTTPException(status_code=400, detail="Invalid sort column")

        sort_field = sort_mapping[request.sort_column]
        if sort_direction == 'desc':
            query = query.order_by(sort_field.desc())
        else:
            query = query.order_by(sort_field.asc())
    else:
        # Default sorting
        query = query.order_by(Memory.created_at.desc())

    # Add eager loading for categories and make the query distinct
    query = query.options(
        joinedload(Memory.categories)
    ).distinct(Memory.id)

    # Use fastapi-pagination's paginate function
    return sqlalchemy_paginate(
        query,
        Params(page=request.page, size=request.size),
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=memory.metadata_
            )
            for memory in items
        ]
    )


@router.get("/{memory_id}/related", response_model=Page[MemoryResponse])
async def get_related_memories(
    memory_id: UUID,
    user_id: str,
    params: Params = Depends(),
    db: Session = Depends(get_db)
):
    # Validate user
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get the source memory
    memory = get_memory_or_404(db, memory_id)
    
    # Extract category IDs from the source memory
    category_ids = [category.id for category in memory.categories]
    
    if not category_ids:
        return Page.create([], total=0, params=params)
    
    # Build query for related memories
    query = db.query(Memory).distinct(Memory.id).filter(
        Memory.user_id == user.id,
        Memory.id != memory_id,
        Memory.state != MemoryState.deleted
    ).join(Memory.categories).filter(
        Category.id.in_(category_ids)
    ).options(
        joinedload(Memory.categories),
        joinedload(Memory.app)
    ).order_by(
        func.count(Category.id).desc(),
        Memory.created_at.desc()
    ).group_by(Memory.id)
    
    # ⚡ Force page size to be 5
    params = Params(page=params.page, size=5)
    
    return sqlalchemy_paginate(
        query,
        params,
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=memory.metadata_
            )
            for memory in items
        ]
    )