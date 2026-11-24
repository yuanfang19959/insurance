from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.exc import InvalidRequestError

from app.agent.schemas import ChatRequest
from app.agent.service import execute_agent, get_message_history
from app.chat_thread.schemas import ChatHistoryResponse
from app.core.logging import get_logger
from app.security.jwt import UserContext, get_current_user

router = APIRouter(prefix="/chat")

logger = get_logger(__name__)

def get_agent(request: Request):
    """从 FastAPI 的 app.state 中安全地提取全局唯一的 Agent 实例"""
    return request.app.state.agent


@router.post("", response_class=EventSourceResponse)
async def chat(
    request: ChatRequest,
    agent: CompiledStateGraph = Depends(get_agent),
    user: UserContext = Depends(get_current_user)
):
    logger.info(f"User {user.username} is making a chat request")
    logger.info(f"user {user.id}")
    logger.info(f"user {user.token}")
    # 获取请求参数
    if not request.thread_id:
        raise InvalidRequestError("thread_id不能为空")
    if not request.message:
        raise InvalidRequestError("message不能为空！")

    # 调用agent
    async for chunk in execute_agent(agent, str(request.thread_id), request.message or '', request.decision, user):
        # 把结果封装为ServerSentEvent
        yield ServerSentEvent(event=chunk.get('event', 'done'), data=chunk.get('data', ''))


@router.get("-threads/{thread_id}/messages")
async def get_chat_history(
    thread_id: UUID,
    agent: CompiledStateGraph = Depends(get_agent),
):
    messages = await get_message_history(agent, str(thread_id))
    return ChatHistoryResponse(
        thread_id=thread_id,
        messages=messages,
    )