from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy.orm import DeclarativeBase
from app.core.config import get_settings
from app.core.logging import logger
import traceback

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

def _create_engine():
    settings = get_settings()
    return create_async_engine(
        settings.DATABASE_URL,
        echo = False,  # log SQL queries in dev
        pool_size = 10,
        max_overflow = 20,
    )

engine = _create_engine()

AsyncSessionFactory = async_sessionmaker(
    bind = engine,
    class_ = AsyncSession,
    expire_on_commit = False,
)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency - yields an async DB session per request.
    """

    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            logger.error(f"Database session rollback due to error: {e}\n{traceback.format_exc()}")
            await session.rollback()
            raise