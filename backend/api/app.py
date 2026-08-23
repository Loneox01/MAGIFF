"""Private FastAPI application for the MAGIFF fantasy agent."""

from __future__ import annotations

import json
import logging
import secrets
import uuid
from dataclasses import asdict
from typing import Any

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from integrations.discord import (
    APPLICATION_COMMAND_INTERACTION,
    CHANNEL_MESSAGE_RESPONSE,
    EPHEMERAL_MESSAGE_FLAG,
    PING_INTERACTION,
    PONG_RESPONSE,
    DiscordCompletion,
    DiscordInteractionRunner,
    DiscordRequestVerifier,
    DiscordWebhookClient,
    RecentInteractionIds,
    extract_ask_prompt,
    format_discord_thinking,
)
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
    discord_webhook_client: DiscordWebhookClient | None = None,
    discord_interaction_ids: RecentInteractionIds | None = None,
) -> FastAPI:
    api_settings = settings or ApiSettings()
    api_settings.validate_runtime()
    service = agent_service or AgentService(model=api_settings.agent_model)
    discord_verifier = None
    discord_runner = None
    interaction_ids = discord_interaction_ids or RecentInteractionIds()
    if api_settings.discord_configured:
        discord_verifier = DiscordRequestVerifier(
            api_settings.discord_public_key_value or ""
        )
        discord_runner = DiscordInteractionRunner(
            application_id=api_settings.discord_application_id or "",
            agent_service=service,
            webhook_client=discord_webhook_client,
        )

    app = FastAPI(
        title="MAGIFF Agent API",
        version="0.2.0",
        description=(
            "Private HTTP transport for the structured-data and report-backed "
            "fantasy-football agent."
        ),
    )
    app.state.settings = api_settings
    app.state.agent_service = service
    app.state.discord_interaction_ids = interaction_ids

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

    def discord_message(content: str) -> dict[str, Any]:
        return {
            "type": CHANNEL_MESSAGE_RESPONSE,
            "data": {
                "content": content,
                "flags": EPHEMERAL_MESSAGE_FLAG,
                "allowed_mentions": {"parse": []},
            },
        }

    @app.post("/v1/discord/interactions", tags=["discord"])
    async def discord_interactions(
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> JSONResponse:
        """Verify, acknowledge, and asynchronously complete a Discord command."""
        if discord_verifier is None or discord_runner is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Discord integration is not configured",
            )

        raw_body = await request.body()
        if not discord_verifier.verify(
            signature_hex=request.headers.get("X-Signature-Ed25519"),
            timestamp=request.headers.get("X-Signature-Timestamp"),
            body=raw_body,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Discord request signature",
            )

        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Discord interaction payload",
            ) from error
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Discord interaction payload",
            )

        if str(payload.get("application_id", "")) != (
            api_settings.discord_application_id
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Discord application ID does not match",
            )

        interaction_type = payload.get("type")
        if interaction_type == PING_INTERACTION:
            return JSONResponse({"type": PONG_RESPONSE})
        if interaction_type != APPLICATION_COMMAND_INTERACTION:
            return JSONResponse(
                discord_message("That Discord interaction is not supported.")
            )
        if str(payload.get("guild_id", "")) != api_settings.discord_guild_id:
            return JSONResponse(
                discord_message("MAGIFF is private to its configured server.")
            )

        prompt = extract_ask_prompt(payload)
        if prompt is None:
            return JSONResponse(
                discord_message("Use `/ask question:<your question>`.")
            )

        interaction_id = str(payload.get("id", "")).strip()
        interaction_token = str(payload.get("token", "")).strip()
        if not interaction_id or not interaction_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Discord interaction ID and token are required",
            )

        if interaction_ids.claim(interaction_id):
            background_tasks.add_task(
                discord_runner.complete,
                DiscordCompletion(
                    interaction_id=interaction_id,
                    interaction_token=interaction_token,
                    prompt=prompt,
                    request_id=request.state.request_id,
                ),
            )
        else:
            LOGGER.info(
                "Ignored duplicate Discord interaction interaction_id=%s",
                interaction_id,
            )

        return JSONResponse(
            {
                "type": CHANNEL_MESSAGE_RESPONSE,
                "data": {
                    "content": format_discord_thinking(prompt),
                    "allowed_mentions": {"parse": []},
                },
            }
        )

    return app


app = create_app()
