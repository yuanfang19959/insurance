from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Response
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.router import get_agent
from app.agent.service import delete_message_history
from app.chat_thread.crud import ChatThreadService
from app.chat_thread.exceptions import ChatThreadNotFoundError
from app.chat_thread.schemas import ChatThreadResponse, ChatThreadCreate
from app.core.database import get_db
from app.core.logging import get_logger
from app.security.jwt import UserContext, get_current_user

logger = get_logger(__name__)

# 定义router
router = APIRouter(prefix="/chat-threads")

# 定义依赖注入函数，返回值是ChatThreadService，而这个函数又依赖get_db这个函数来获取session
def get_thread_service(db: AsyncSession = Depends(get_db)) -> ChatThreadService:
    return ChatThreadService(db)

# 定义接口
@router.post("", response_model=ChatThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_thread(
    body: ChatThreadCreate,
    chat_thread_service: ChatThreadService = Depends(get_thread_service),
    user: UserContext = Depends(get_current_user)
) -> ChatThreadResponse:
    """创建新会话"""
    user_id = user.id
    thread = await chat_thread_service.create(user_id, body.title)
    logger.info("会话创建成功~", thread_id=str(thread.id), user_id=user_id)
    return thread

@router.get("", response_model=list[ChatThreadResponse], status_code=status.HTTP_200_OK)
async def get_chat_list(
    chat_thread_service: ChatThreadService = Depends(get_thread_service),
    user: UserContext = Depends(get_current_user)
) -> list[ChatThreadResponse]:
    user_id = user.id
    # 查询当前用户的会话
    result = await chat_thread_service.get_chat_list(user_id)
    logger.info(f"查询用户的会话列表:{len(result)}")
    logger.info(result)
    return result


@router.patch("/{thread_id}", response_model=ChatThreadResponse, status_code=status.HTTP_200_OK)
async def rename_chat_thread(
    thread_id: UUID,
    body: ChatThreadCreate,
    chat_thread_service: ChatThreadService = Depends(get_thread_service),
    user: UserContext = Depends(get_current_user)
) -> ChatThreadResponse:
    """重命名会话"""
    user_id = user.id
    try:
        thread = await chat_thread_service.rename(thread_id, user_id, body.title)
        logger.info("chat_thread_renamed", thread_id=str(thread_id), user_id=user_id)
        return thread
    except ChatThreadNotFoundError as exc:
        raise _not_found() from exc

@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_thread(
    thread_id: UUID,
    agent: CompiledStateGraph=Depends(get_agent),
    chat_thread_service: ChatThreadService = Depends(get_thread_service),
    user: UserContext = Depends(get_current_user)
) -> None:
    """删除会话"""
    user_id = user.id
    try:
        await chat_thread_service.delete(thread_id, user_id)

        # 删除会话历史消息
        await delete_message_history(agent, thread_id)

        logger.info("chat_thread_deleted", thread_id=str(thread_id), user_id=user_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except ChatThreadNotFoundError as exc:
        raise _not_found() from exc

# 定义通用的404异常方法
def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="chat chat_thread not found")