from __future__ import annotations

from datetime import datetime
from typing import Literal, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ──────────── 会话相关模型 ────────────
# 1.新增、更新会话的请求模型
class ChatThreadCreate(BaseModel):
    title: str = Field(default="新会话", min_length=1, max_length=200)

    @field_validator('title')
    @classmethod
    def title_must_not_be_empty(cls, v: str) -> str:
        # strip() 可以同时把全是空格的字符串 "   " 也拦截掉
        if not v or not v.strip():
            raise ValueError('字段不能为空或全为空格')
        return v
# 2.会话响应模型，不返回user_id，避免敏感信息泄露
class ChatThreadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # 允许直接从 ORM 对象生成响应

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


# ──────────── 会话消息相关模型 ────────────
# 1.会话消息
class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    additional_info: Any | None = None # 拓展字段，暂时不用

# 2.会话消息响应模型，可能包含 messages或 interrupt
class ChatHistoryResponse(BaseModel):
    thread_id: UUID
    messages: list[ChatHistoryMessage]
    interrupt: dict[str, Any] | None = None