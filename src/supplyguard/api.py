"""HTTP API for the local SupplyGuard RAG service."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock
from typing import Any, Literal, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from supplyguard.config import DEFAULT_CONFIG, SupplyGuardConfig
from supplyguard.errors import (
    ConfigurationError,
    CorpusError,
    InputValidationError,
    ModelError,
    RetrievalIndexError,
    SupplyGuardError,
)
from supplyguard.llm.answer_generation import RAGPipeline
from supplyguard.runtime import load_runtime

type RetrievalMethod = Literal["bm25", "dense", "hybrid"]


def _package_version() -> str:
    try:
        return version("supplyguard")
    except PackageNotFoundError:
        return "0.1.0"


class AskRequest(BaseModel):
    """A grounded advisory question and retrieval strategy."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=DEFAULT_CONFIG.models.question_max_chars)
    retrieval_method: RetrievalMethod = "hybrid"


class AdvisorySourceResponse(BaseModel):
    """A trusted OSV source used to generate an answer."""

    advisory_id: str
    chunk_id: str
    url: str
    summary: str
    rank: int
    score: float


class AskResponse(BaseModel):
    """A grounded answer with retrieval and citation metadata."""

    question: str
    answer: str
    retrieval_method: RetrievalMethod
    sources: list[AdvisorySourceResponse]
    num_chunks: int
    num_sources: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready"] = "ready"
    corpus_advisories: int
    language_model: str
    embedding_model: str
    device: str


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
    validation_errors: list["ValidationIssue"] | None = None


class ValidationIssue(BaseModel):
    location: str
    message: str
    error_type: str


@dataclass(slots=True)
class SupplyGuardService:
    """Thread-safe application service around one initialized RAG pipeline."""

    pipeline: RAGPipeline
    config: SupplyGuardConfig
    corpus_size: int
    _inference_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @classmethod
    def load(
        cls,
        corpus_path: Path,
        config: SupplyGuardConfig,
        rebuild_index: bool,
    ) -> "SupplyGuardService":
        runtime = load_runtime(
            corpus_path,
            config=config,
            rebuild_index=rebuild_index,
        )
        pipeline = RAGPipeline(runtime.bm25, runtime.dense, runtime.llm, config=config)
        return cls(pipeline=pipeline, config=config, corpus_size=len(runtime.records))

    def answer(self, question: str, retrieval_method: RetrievalMethod) -> dict[str, Any]:
        # Local model runtimes do not guarantee thread-safe inference.
        with self._inference_lock:
            return self.pipeline.generate_answer(
                question,
                retrieval_method=retrieval_method,
            )


type ServiceFactory = Callable[[], SupplyGuardService]


def _status_code(error: SupplyGuardError) -> int:
    if isinstance(error, InputValidationError):
        return 422
    if isinstance(error, ConfigurationError):
        return 400
    if isinstance(error, (CorpusError, RetrievalIndexError, ModelError)):
        return 503
    return 500


def create_app(
    *,
    corpus_path: Path = DEFAULT_CONFIG.paths.corpus_path,
    config: SupplyGuardConfig = DEFAULT_CONFIG,
    rebuild_index: bool = False,
    service_factory: ServiceFactory | None = None,
) -> FastAPI:
    """Create an API whose expensive runtime is initialized during startup."""

    if service_factory is None:

        def default_service_factory() -> SupplyGuardService:
            return SupplyGuardService.load(corpus_path, config, rebuild_index)

        resolved_service_factory = default_service_factory
    else:
        resolved_service_factory = service_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.supplyguard_service = resolved_service_factory()
        yield

    app = FastAPI(
        title="SupplyGuard API",
        description="Grounded questions over a local OSV security advisory corpus.",
        version=_package_version(),
        lifespan=lifespan,
    )

    @app.exception_handler(SupplyGuardError)
    async def handle_supplyguard_error(
        _request: Request,
        error: SupplyGuardError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=_status_code(error),
            content={"detail": str(error), "error_type": type(error).__name__},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        issues = [
            {
                "location": ".".join(str(part) for part in issue["loc"]),
                "message": issue["msg"],
                "error_type": issue["type"],
            }
            for issue in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Request validation failed",
                "error_type": "RequestValidationError",
                "validation_errors": issues,
            },
        )

    def current_service(request: Request) -> SupplyGuardService:
        return cast(SupplyGuardService, request.app.state.supplyguard_service)

    @app.get("/healthz", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        return HealthResponse(version=app.version)

    @app.get("/readyz", response_model=ReadinessResponse, tags=["operations"])
    def readiness(request: Request) -> ReadinessResponse:
        service = current_service(request)
        return ReadinessResponse(
            corpus_advisories=service.corpus_size,
            language_model=service.config.models.language_model_name,
            embedding_model=service.config.models.embedding_model_name,
            device=service.config.models.device,
        )

    @app.post(
        "/v1/ask",
        response_model=AskResponse,
        responses={
            400: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["advisories"],
    )
    def ask(request_body: AskRequest, request: Request) -> AskResponse:
        result = current_service(request).answer(
            request_body.question,
            request_body.retrieval_method,
        )
        return AskResponse.model_validate(result)

    return app


def run_api_server(
    *,
    host: str,
    port: int,
    corpus_path: Path,
    config: SupplyGuardConfig,
    rebuild_index: bool,
) -> None:
    """Run the API with Uvicorn using the selected local models."""

    import uvicorn

    uvicorn.run(
        create_app(
            corpus_path=corpus_path,
            config=config,
            rebuild_index=rebuild_index,
        ),
        host=host,
        port=port,
    )
