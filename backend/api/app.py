"""Private FastAPI application for the MAGIFF fantasy agent."""

from __future__ import annotations

import json
import logging
import secrets
import uuid
from dataclasses import asdict

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.agent import AgentService

from .config import ApiSettings
from .schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    HealthResponse,
    ReadinessResponse,
)


LOGGER = logging.getLogger(__name__)
BEARER = HTTPBearer(auto_error=False)


def create_app(
    *,
    settings: ApiSettings | None = None,
    agent_service: AgentService | None = None,
) -> FastAPI:
    api_settings = settings or ApiSettings()
    api_settings.validate_runtime()
    service = agent_service or AgentService(model=api_settings.agent_model)

    app = FastAPI(
        title="MAGIFF Agent API",
        version="0.1.0",
        description=(
            "Private HTTP transport for the structured-data and report-backed "
            "fantasy-football agent."
        ),
    )
    app.state.settings = api_settings
    app.state.agent_service = service

    if api_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=api_settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    def require_api_key(
        credentials: HTTPAuthorizationCredentials | None = Depends(BEARER),
    ) -> None:
        expected = api_settings.api_key_value
        if expected is None:
            if api_settings.environment == "production":
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="API authentication is not configured",
                )
            return
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or not secrets.compare_digest(credentials.credentials, expected)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/", response_model=HealthResponse, tags=["system"])
    def root() -> HealthResponse:
        return HealthResponse(status="ok", service="magiff-agent-api")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        """Cheap liveness probe that does not spend tokens or query Supabase."""
        return HealthResponse(status="ok", service="magiff-agent-api")

    @app.get("/ready", response_model=ReadinessResponse, tags=["system"])
    def ready():
        dependencies = api_settings.dependency_status()
        response = ReadinessResponse(
            status="ready" if all(dependencies.values()) else "not_ready",
            dependencies=dependencies,
        )
        if response.status == "not_ready":
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response.model_dump(),
            )
        return response

    @app.post(
        "/v1/agent/query",
        response_model=AgentQueryResponse,
        dependencies=[Depends(require_api_key)],
        tags=["agent"],
    )
    def query_agent(
        payload: AgentQueryRequest,
        request: Request,
    ) -> AgentQueryResponse:
        try:
            result = service.run(payload.prompt)
        except Exception as error:
            LOGGER.exception(
                "Agent request failed request_id=%s",
                request.state.request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Agent request failed",
            ) from error

        LOGGER.info(
            json.dumps(
                {
                    "event": "agent_request_complete",
                    "request_id": request.state.request_id,
                    "model": result.model,
                    "latency_seconds": round(result.latency_seconds, 3),
                    "input_tokens": result.usage.input_tokens,
                    "cached_input_tokens": result.usage.cached_input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "tool_calls": len(result.tool_calls),
                    "router_fallback": result.route.fallback_used,
                },
                separators=(",", ":"),
            )
        )
        response = AgentQueryResponse.model_validate(
            {
                "request_id": request.state.request_id,
                **asdict(result),
            }
        )
        return response

    return app


app = create_app()
