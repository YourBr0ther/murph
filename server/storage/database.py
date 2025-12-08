"""
Murph - Database Connection Management
Async SQLite database setup using SQLAlchemy 2.0 and aiosqlite.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

# Default database location
DEFAULT_DB_PATH = Path("~/.murph/memory.db").expanduser()


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""

    pass


class Database:
    """
    Async database manager for Murph's long-term memory.

    Usage:
        db = Database()
        await db.initialize()

        async with db.session() as session:
            # do work
            await session.commit()

        await db.close()
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file. Defaults to ~/.murph/memory.db
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def db_path(self) -> Path:
        """Get the database file path."""
        return self._db_path

    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create async engine
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{self._db_path}",
            echo=False,  # Set True for SQL debugging
        )

        # Create session factory
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Create all tables
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session as an async context manager.

        Usage:
            async with db.session() as session:
                result = await session.execute(query)
                await session.commit()

        Yields:
            AsyncSession for database operations
        """
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        async with self._session_factory() as session:
            yield session

    async def close(self) -> None:
        """Close the database connection."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized."""
        return self._engine is not None
