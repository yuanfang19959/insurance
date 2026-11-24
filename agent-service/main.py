import time
import sys
import uuid
import asyncio
from contextlib import asynccontextmanager
import structlog
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from app.agent.graph import create_agent
from app.agent.checkpointer import init_checkpointer, close_checkpointer
from app.core.config import settings
from app.core.database import init_db, close_db
from app.core.logging import configure_logging, get_logger
from app.clients.biz_client import biz_client


# 初始化日志配置
configure_logging(level=settings.log_level, log_format=settings.log_format)
logger = get_logger(__name__)

# 生命周期管理（预留）
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("agent_service_starting", app_version=settings.app_version)
        # 初始化数据库连接池和表
        await init_db()
        # 初始化Checkpointer
        checkpointer = await init_checkpointer()

        # 初始化agent,保存到app.state中
        app.state.agent = await create_agent(checkpointer)

        # 初始化biz_client
        await biz_client.init()
        yield
    finally:
        logger.info("agent_service_stopping")
        # 关闭数据库连接池
        await close_db()
        # 关闭Checkpointer
        await close_checkpointer()

        # 关闭bizclient客户端
        await biz_client.close()
        logger.info("agent_service_stopped")

# 初始化FastAPI
app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

# 日志输出中间件
@app.middleware("http")
async def add_log_uuid(request: Request, call_next):
    # 1. 清理并生成新的请求上下文
    structlog.contextvars.clear_contextvars()
    request_id = str(uuid.uuid4())

    # 2. 绑定当前请求的核心信息
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    # 3. 记录请求结束日志
    logger.info("request_processed", status_code=response.status_code, duration=f"{duration_ms:.4f}ms")
    return response

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins="http://127.0.0.1:5173",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入路由
from app.chat_thread.router import router as chat_thread_router
from app.agent.router import router as agent_router
app.include_router(chat_thread_router, prefix="/api/v1", tags=["会话管理相关接口"])
app.include_router(agent_router, prefix="/api/v1", tags=["聊天相关接口"])
@app.get("/")
async def healthy():
    return {"status": "running", "port": "8001"}

if __name__ == "__main__":
    import uvicorn
    # 启动命令：python main.py
    config = uvicorn.Config(
        "main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=True
    )
    server = uvicorn.Server(config)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())
    else:
        server.run()