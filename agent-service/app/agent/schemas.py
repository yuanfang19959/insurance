from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    thread_id: Optional[UUID] = None
    message: Optional[str] = None # 用户输入
    decision: Optional[dict[str, Any]] = None # HITL人的回复

# 上下文信息
@dataclass
class ChatRuntimeContext:
    user_id: str | None
    token: str | None