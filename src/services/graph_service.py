from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.config.settings import (
    POSTGRES_POOL_MAX_SIZE,
    POSTGRES_POOL_MIN_SIZE,
    SUPABASE_POSTGRES_URI,
)
from src.graph.graph_builder import graph_builder


class GraphService:
    """
    Compiles the graph with a Postgres (Supabase) checkpointer, so every
    session_id keeps its own conversation history between requests.
    """

    def __init__(self):
        # Use Supabase's "Direct connection" or "Session pooler" URI.
        self._pool = ConnectionPool(
            conninfo=SUPABASE_POSTGRES_URI,
            min_size=POSTGRES_POOL_MIN_SIZE,
            max_size=POSTGRES_POOL_MAX_SIZE,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
            open=True,
        )

        try:
            self._pool.wait(timeout=10)
            checkpointer = PostgresSaver(self._pool)
            checkpointer.setup()
        except Exception:
            self._pool.close()
            raise

        self.compiled_graph = graph_builder.compile(checkpointer=checkpointer)

    def get_graph_mermaid(self) -> str:
        # xray=True also draws the nodes inside the subgraphs
        return self.compiled_graph.get_graph(xray=True).draw_mermaid()

    def process_message(self, input_message: str, session_id: str, username: str) -> dict:
        response = self.compiled_graph.invoke(
            {
                "input_message": input_message,
                "session_id": session_id,
                "username": username,
            },
            config={
                "metadata": {
                    "session_id": session_id,
                    "user_id": username
                },
                "configurable": {
                    "thread_id": session_id
                }
            }
        )

        return response

    def close(self) -> None:
        self._pool.close()
