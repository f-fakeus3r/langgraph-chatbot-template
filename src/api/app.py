import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes.chatbot_route import router as chatbot_router
from src.config.settings import LOG_LEVEL, check_required_settings
from src.services.graph_service import GraphService

logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_required_settings()
    app.state.graph_service = GraphService()
    yield
    app.state.graph_service.close()


def create_app() -> FastAPI:

    app = FastAPI(
        title="LangGraph Example Template API",
        description="",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(chatbot_router)

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok"}

    return app


app = create_app()
