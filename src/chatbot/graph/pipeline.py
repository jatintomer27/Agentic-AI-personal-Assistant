"""
Assembles the LangGraph StateGraph from individual nodes.

Current graph:
    START → preprocess → llm_call → END
"""


import psycopg
from box import ConfigBox
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode

from chatbot import logger
from chatbot.database import DATABASE_URL
from chatbot.graph.state import AgentState
from chatbot.graph.nodes import (
    preprocess,
    llm_call, 
    extract_ltm_info,
    summarize_conversation,
    should_summarize,
    _route_after_llm,
)
from chatbot.tools import ALL_TOOLS
from chatbot.utils.common import load_config_file



class GraphBuilder:
    """
    Builds and compiles a LangGraph StateGraph based on active feature flags.

    Usage:
        builder = GraphBuilder(config)
        chatbot = builder.build()
    """

    def __init__(self, config: ConfigBox):
        self.config = config
        self.builder = StateGraph(AgentState)

    # ── Public API ────────────────────────────────────────────────────

    def build(self):
        """
        Orchestrates node registration, edge wiring, and compilation.
        Returns a compiled graph ready for .invoke() or .stream().
        """
        try:
            self._register_nodes()
            self._wire_edges()
            checkpointer = self._create_checkpointer()
            ltm_store = self._create_ltm_store()
            graph = self.builder.compile(
                checkpointer=checkpointer,
                store=ltm_store
            )
            logger.info("LangGraph chatbot compiled successfully.")
            return graph
        except Exception as e:
            logger.error(f"Failed to build the LangGraph: {e}")
            raise



    # ── Private helpers ───────────────────────────────────────────────

    def _register_nodes(self):
        """Adds always-on and feature-gated nodes to the graph."""
        try:
            # Always-on nodes
            self.builder.add_node("preprocess", preprocess)
            self.builder.add_node("ltm_info", extract_ltm_info)
            self.builder.add_node("llm_call", llm_call)
            self.builder.add_node("summarize_conversation", summarize_conversation)

            # Optional nodes (added only when feature is enabled)
            
            if self.config.features.tools_enabled:
                self.builder.add_node("tools", ToolNode(ALL_TOOLS))

            # if self.config.features.hitl_enabled:
            #     self.builder.add_node("hitl_check", hitl_check)

            # if self.config.features.mcp_enabled:
            #     self.builder.add_node("mcp_node", mcp_node)

            logger.info("All graph nodes registered successfully.")
        except AttributeError as e:
            # Missing config key e.g. config.features.tools_enabled not found
            logger.error(f"Missing config key while registering nodes: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while registering nodes: {e}")
            raise

    def _wire_edges(self):
        """
        Connects nodes in the correct order based on which features are active.
        """
        try:
            # Entry point
            
            self.builder.add_edge(START, "preprocess")
            self.builder.add_edge("preprocess", "ltm_info")
            self.builder.add_edge("ltm_info", "llm_call")

            # Optionally route to tools
            if self.config.features.tools_enabled:
                self.builder.add_conditional_edges(
                    "llm_call",
                    _route_after_llm, # returns "tools" or "__end__" by default
                    {
                        "tools": "tools",
                        "summarize": "summarize_conversation",    # intercept "__end__" → go summarize first
                        END:END
                    }
                )
                self.builder.add_edge("tools", "llm_call")
            else:
                self.builder.add_conditional_edges(
                    "llm_call",
                    should_summarize,
                    {
                        True: "summarize_conversation",
                        False: END,
                    }
                )
            self.builder.add_edge("summarize_conversation", END)
            logger.info("Graph edges wired successfully.")
        except AttributeError as e:
            # Missing config key
            logger.error(f"Missing config key while wiring edges: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while wiring edges: {e}")
            raise
    
    @staticmethod
    def _create_checkpointer():
        """Creates the PostgreSQL checkpointer for the LangGraph."""
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            conn = psycopg.connect(DATABASE_URL, autocommit=True)
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()  # Creates checkpointer tables if not exist
            logger.info("PostgreSQL checkpointer created successfully.")
            return checkpointer
        except ImportError as e:
            logger.error(
                f"Missing package for PostgreSQL checkpointer: {e}\n"
                f"Fix: pip install psycopg langgraph-checkpoint-postgres"
            )
            raise
        except Exception as e:
            logger.error(
                f"Failed to create PostgreSQL checkpointer: {e}\n"
                f"Most likely cause: PostgreSQL is not running or DATABASE_URL is wrong."
            )
            raise

    @staticmethod
    def _create_ltm_store():
        """Creates the PostgreSQL LTM store for the LangGraph."""
        try:
            from langgraph.store.postgres import PostgresStore
            from psycopg_pool import ConnectionPool
            pool = ConnectionPool(
                conninfo=DATABASE_URL,
                max_size=10,
                kwargs={"autocommit": True},
            )
            store = PostgresStore(pool)

            # conn = psycopg.connect(DATABASE_URL, autocommit=True)  # keep connection alive yourself
            # store = PostgresStore(conn)
            store.setup() # Creates required tables
            logger.info("PostgreSQL Store for LTM created successfully.")
            return store
        except ImportError as e:
            logger.error(
                f"Missing package for PostgreSQL Store: {e}\n"
                f"Fix: pip install langgraph-store-postgres"
            )
            raise
        except Exception as e:
            logger.error(
                f"Failed to create PostgreSQL store: {e}\n"
                f"Most likely cause: PostgreSQL is not running or DATABASE_URL is wrong."
            )
            raise


# ── Module-level compiled graph instance ─────────────────────────────



def get_chatbot():
    """
    Loads config and builds the compiled LangGraph chatbot.
    Call this once at app startup via st.cache_resource.
    """
    try:
        _config = load_config_file(__file__)
    except Exception as e:
        raise

    try:
        chatbot = GraphBuilder(_config).build()
        logger.info(f"Chatbot is ready.")
        return chatbot
    except Exception as e:
        logger.error(
            f"Failed to initialize chatbot graph: {e}\n"
            f"Check database connection and config settings."
        )
        raise
