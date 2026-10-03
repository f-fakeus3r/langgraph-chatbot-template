from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse

from src.api.schemas import MessageResponse, UserData
from src.services.graph_service import GraphService

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


def get_graph_service(request: Request) -> GraphService:
    return request.app.state.graph_service


GraphServiceDep = Annotated[GraphService, Depends(get_graph_service)]


@router.get("/get-graph", response_class=PlainTextResponse)
def get_graph(graph_service: GraphServiceDep):
    return graph_service.get_graph_mermaid()


@router.post("/message", response_model=MessageResponse)
def send_message(input_query: UserData, graph_service: GraphServiceDep):
    response = graph_service.process_message(
        input_message=input_query.input_message,
        # You must remember to make an authentication middleware if you want to publish your code in production.
        session_id=input_query.session_id,
        username=input_query.username,
    )

    return MessageResponse(
        input_message=input_query.input_message,
        output_message=response["last_message"],
        route=response["route"],
        input_guardrail=response["input_guardrail_result"].get("result", ""),
        output_guardrail=response["output_guardrail_result"].get("result", ""),
    )
