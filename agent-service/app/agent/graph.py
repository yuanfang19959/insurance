from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolRuntime, tools_condition, ToolNode

from app.clients.biz_client import biz_client, BusinessServiceError
from app.core.config import settings
from app.agent.schemas import ChatRuntimeContext
from app.core.logging import get_logger

logger = get_logger(__name__)
# 1.定义工具
@tool
async def get_products():
    """查询所有的保险产品"""
    try:
        result = await biz_client.list_products()
        return result
    except Exception as e:
        logger.error(f"查询保险产品异常, e: {e}")
        return "未知异常，查询失败！"

@tool
async def get_current_insurance_plan(runtime: ToolRuntime[ChatRuntimeContext]):
    """查询当前用户的保险方案"""
    logger.info(runtime.context)
    try:
        result = await biz_client.get_current_insurance_plan(runtime.context.token)
        logger.info(f"查询保险方案成功, result: {result}")
        return result
    except BusinessServiceError as e:
        logger.error(f"查询保险方案异常, {e.status_code} e: {e.body}")
        if e.status_code == 404:
            return "没有查询到保险方案"
        return "未知异常，查询失败！"

async def create_agent(checkpointer):
    # 1.初始化对话模型
    # pydantic-settings 只把 .env 读进 settings，不会自动写入 os.environ；
    # ChatDeepSeek 默认读环境变量 DEEPSEEK_API_KEY，所以这里必须显式传 api_key
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置，无法初始化对话模型")
    llm = init_chat_model(
        settings.chat_model,
        model_provider="deepseek",
        api_key=settings.deepseek_api_key,
        extra_body={"thinking": {"type": "disabled"}},
    )
    tools = [get_products, get_current_insurance_plan]
    llm = llm.bind_tools(tools)

    # 2.定义Node
    async def llm_node(state: MessagesState):
        # 直接对话
        response = await llm.ainvoke(state['messages'])
        # 返回结果
        return {'messages': [response]}

    # 3.创建Graph
    builder = StateGraph(MessagesState, context_schema=ChatRuntimeContext)
    builder.add_node('llm', llm_node)
    builder.add_node('tools', ToolNode(tools))

    builder.add_edge(START, 'llm')
    builder.add_conditional_edges('llm', tools_condition)
    builder.add_edge('tools', 'llm')
    builder.add_edge('llm', END)

    graph = builder.compile(checkpointer=checkpointer)

    return graph