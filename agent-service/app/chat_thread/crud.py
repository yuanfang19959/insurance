from uuid import UUID

from .exceptions import ChatThreadNotFoundError
from .models import ChatThreadTable

from sqlalchemy import select, desc, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

class ChatThreadService:
    """管理用户会话数据"""
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, user_id: int, title: str) -> ChatThreadTable:
        """新增会话"""
        async with self.session.begin():
            # 保存会话并返回结果
            chat_thread = ChatThreadTable(user_id=user_id, title=title)
            self.session.add(chat_thread)
        return chat_thread

    async def get_chat_list(self, user_id: int) -> list[ChatThreadTable]:
        # 查询列表（SQLAlchemy 2.x：select(Model)，不要传 list）
        result = await self.session.execute(
            select(ChatThreadTable).where(ChatThreadTable.user_id == user_id)
            .order_by(desc(ChatThreadTable.updated_at), desc(ChatThreadTable.created_at))
        )
        return list(result.scalars().all())

    async def rename(self, thread_id, user_id: int, title: str) -> ChatThreadTable | None:
        """重命名会话"""
        async with self.session.begin():
            # 这里是直接手动执行update ... returning，先更新再返回
            result = await self.session.execute(
                update(ChatThreadTable)
                .where(ChatThreadTable.id == thread_id, ChatThreadTable.user_id == user_id)
                .values(title=title)
                .returning(ChatThreadTable)
            )
            # 如果没找到要更新的数据，直接报错
            row = result.scalar_one_or_none()
            if row is None:
                raise ChatThreadNotFoundError(str(thread_id))
            return row

    async def delete(self, thread_id: UUID, user_id: int) -> bool:
        """删除会话。删到返回 True，不存在返回 False。"""
        async with self.session.begin():
            result = await self.session.execute(
                delete(ChatThreadTable)
                .where(
                    ChatThreadTable.id == thread_id,
                    ChatThreadTable.user_id == user_id,
                )
                .returning(ChatThreadTable.id)
            )
            deleted_id = result.scalar_one_or_none()
        return deleted_id is not None