from typing import Any

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph

from app.agent.schemas import ChatRuntimeContext
from app.chat_thread.schemas import ChatHistoryMessage
from app.security.jwt import UserContext


async def execute_agent(
    agent: CompiledStateGraph[MessagesState],
    thread_id: str,
    message: str,
    decision: dict[str, Any],
    user: UserContext
):
    # 组织输入
    _input = {'messages': [HumanMessage(content=message)]}
    # 会话记忆id
    config = {"configurable": {"thread_id": thread_id}}
    # 调用模型
    response = agent.astream(_input, config, stream_mode='messages', version='v2', context=ChatRuntimeContext(user_id=user.id, token=user.token),)
    # 返回结果
    async for chunk in response:
        if chunk['type'] == 'messages':
            token, metadata = chunk['data']
            if isinstance(token, AIMessage) and token.content:
                yield {'event': 'message', 'data': token.content}
    # 消息结束，返回done
    yield {'event': 'done', 'data': {'thread_id': thread_id, 'finish': 'stop'}}


def _config_with_thread_id(thread_id):
    # LangGraph 用 configurable.thread_id 区分不同会话的 checkpoint（记忆）
    return {"configurable": {"thread_id": thread_id}}

async def get_message_history(
    agent: CompiledStateGraph,
    thread_id: str
):
    """把完整 LangGraph 消息投影为前端可展示的 user / assistant 文本。

    LangGraph 会把一次对话的完整状态（含 messages）按 thread_id 存进 checkpoint。
    这里做的事：按 thread_id 读出 state → 过滤出可展示的文本 → 转成前端 schema。
    """
    config = _config_with_thread_id(thread_id)

    # aget_state：异步读取该 thread 当前最新的图状态（StateSnapshot）
    # state.values 是真正的业务状态字典，常见键是 "messages"
    state = await agent.aget_state(config)
    print(state)
    # 新会话或尚未写入过任何内容时，可能没有 state / values
    if state is None or not state.values:
        return []

    # messages 是 LangChain 消息对象列表，不只是纯字符串：
    # HumanMessage / AIMessage / ToolMessage / SystemMessage 等可能混在一起
    # 注意：key 存在但值为 None 时，.get("messages", []) 仍会得到 None，所以用 or []
    messages = state.values.get("messages") or []

    histories: list[ChatHistoryMessage] = []
    for message in messages:
        # 用户输入 → 前端 role = "user"
        if isinstance(message, HumanMessage):
            role = "user"
        # 模型输出 → 前端 role = "assistant"
        elif isinstance(message, AIMessage):
            # Agent 调用工具时也会产生 AIMessage，但内容往往是 tool_calls，不是给人看的回复。
            # 这类消息跳过，避免前端把「工具调用意图」当成聊天气泡展示。
            if message.tool_calls or message.additional_kwargs.get("tool_calls"):
                continue
            role = "assistant"
        else:
            # ToolMessage、SystemMessage 等对前端聊天列表通常无意义，直接跳过
            continue

        content = message.content
        # content 可能是空字符串，或（少数情况下）多模态结构；这里只展示有文本的消息
        if isinstance(content, str) and content:
            histories.append(ChatHistoryMessage(role=role, content=content))
    return histories

async def delete_message_history(
        agent: CompiledStateGraph,
        thread_id: str):
    """会话历史"""
    await agent.checkpointer.adelete_thread(thread_id)
