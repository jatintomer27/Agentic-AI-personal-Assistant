"""
Handles PostgreSQL connection.
- Creates the database if it doesn't exist
- Creates all tables if they don't exist
- Returns a session factory for DB operations
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from chatbot.database.models import Base
from chatbot import logger

load_dotenv()

# ─────────────────────────────────────────────
#  Read config
# ─────────────────────────────────────────────

DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = os.getenv("DB_PORT",     "5432")
DB_NAME     = os.getenv("DB_NAME",     "chatbot_db")

# Connection URL to the TARGET database
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Connection URL to 'postgres' (default db) — used to CREATE our db if missing
POSTGRES_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres"


# ─────────────────────────────────────────────
#  Step 1 — Create DB if it doesn't exist
# ─────────────────────────────────────────────

def create_database_if_not_exists():
    """
    Connects to the default 'postgres' database and creates
    our target database if it doesn't already exist.
    """
    engine = None
    try:
        # autocommit=True is required for CREATE DATABASE statements
        engine = create_engine(POSTGRES_URL, isolation_level="AUTOCOMMIT")

        with engine.connect() as conn:
            # Check if the database already exists
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                {"db_name": DB_NAME}
            )
            exists = result.fetchone()

            if not exists:
                conn.execute(text(f'CREATE DATABASE "{DB_NAME}"'))
                logger.info(f"Database '{DB_NAME}' created.")
            else:
                logger.info(f"Database '{DB_NAME}' already exists.")
    except Exception as e:
        logger.error(
            f"Error occured while connecting with PostgreSQL "
            f"and creating the database '{DB_NAME}': {e}"
        )
        raise
    finally:
        if engine:
            engine.dispose()


# ─────────────────────────────────────────────
#  Step 2 — Create tables if they don't exist
# ─────────────────────────────────────────────

def create_tables(engine):
    """
    Creates all tables defined in models.py if they don't exist.
    Safe to call every time — won't drop existing tables.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Tables are ready.")
    except Exception as e:
        logger.error(f"Error occurred while creating tables: {e}")
        raise


# ─────────────────────────────────────────────
#  Step 3 — Build engine + session factory
# ─────────────────────────────────────────────

def init_db():
    """
    Full initialization:
      1. Create DB if missing
      2. Create tables if missing
      3. Return a session factory

    Call this ONCE at app startup.
    """
    
    # 1. Create database if needed
    create_database_if_not_exists()


    try:
        # 2. Create engine pointed at our database
        engine = create_engine(
            DATABASE_URL,
            pool_size=5,          # max persistent connections
            max_overflow=10,      # extra connections allowed under load
            pool_pre_ping=True,   # test connection before using it
        )
        logger.info("Database engine created successfully.")
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise

    # 3. Create tables if needed
    create_tables(engine)

    try:
        # 4. Return a session factory
        SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        logger.info("Session factory created successfully.")
        return SessionLocal
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise