"""
SQLAlchemy table definitions.

Tables:
    conversations  → one row per chat session
    messages       → one row per message (linked to a conversation)
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Conversation(Base):
    """
    One row = one chat session.
    """
    __tablename__ = "conversations"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id      = Column(String, unique=True, nullable=False, index=True)
    session_name    = Column(String, nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    

    def __repr__(self):
        return f"<Conversation session_id={self.session_id}>"


