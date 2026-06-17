"""
All database operations.

This is the ONLY file the rest of your app should talk to for DB work.

Functions:
    save_session_name(session, session_id, name)      → Save Session name in the conversations table
    get_all_sessions(session)                         → lists all conversation sessions
"""

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from chatbot import logger
from chatbot.database.models import Conversation


# ─────────────────────────────────────────────
#  Save the session and session name 
# ─────────────────────────────────────────────


def save_session_name(session: Session, session_id: str, name: str):
    """
    Saves the first message as the session name.
    Only sets it once — if name already exists, skip.
    """
    try:

        conversation = session.query(Conversation).filter_by(session_id=session_id).first()

        if not conversation:
            # First time — create the row with session name
            conversation = Conversation(
                session_id=session_id,
                session_name=name[:40]  # truncate to 40 chars
            )
            session.add(conversation)
            logger.info(f"Created new conversation session: '{session_id}'")

        elif not conversation.session_name:
            conversation.session_name = name[:40]
            logger.info(f"Updated session name for session: '{session_id}'")
        else:
            # Name already set — skip silently
            logger.debug(f"Session name already set for '{session_id}', skipping.")
            return
        session.commit()
        logger.info(f"Session name saved for session_id='{session_id}'")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(
                f"Database error while saving session name "
                f"for session_id='{session_id}': {e}"
            )
        raise
    except Exception as e:                      # ✅ Catch any unexpected errors
        session.rollback()
        logger.error(
            f"Unexpected error while saving session name "
            f"for session_id='{session_id}': {e}"
        )
        raise

# ─────────────────────────────────────────────
#  List all sessions
# ─────────────────────────────────────────────

def get_all_sessions(session: Session) -> list[dict]:
    """
    Returns all conversation sessions, newest first.
    Useful for showing a chat history sidebar.

    Returns:
        List of dicts: [{"session_id": "abc123", "created_at": ..., "updated_at": ...}]
    """
    try:
        conversations = (
            session.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .all()
        )
        logger.info(f"Fetched {len(conversations)} conversation session(s).")
        return [
            {
                "session_id": c.session_id,
                "session_name": c.session_name or "New Chat",
                "created_at": c.created_at,
                "updated_at": c.updated_at,
            }
            for c in conversations
        ]
    except SQLAlchemyError as e:                
            logger.error(f"Database error while fetching all sessions: {e}")
            raise                                 
    except Exception as e:                      # ✅ Catch any unexpected errors
        logger.error(f"Unexpected error while fetching all sessions: {e}")
        raise

