from fastapi.testclient import TestClient

from supplyguard.api import SupplyGuardService, create_app
from supplyguard.config import DEFAULT_CONFIG
from supplyguard.errors import InputValidationError, ModelGenerationError


class FakePipeline:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def generate_answer(self, question, *, retrieval_method):
        self.calls.append((question, retrieval_method))
        if self.error is not None:
            raise self.error
        return {
            "question": question,
            "answer": "Upgrade to 1.2.3.",
            "retrieval_method": retrieval_method,
            "sources": [
                {
                    "advisory_id": "GHSA-test-0000-0001",
                    "chunk_id": "GHSA-test-0000-0001_chunk_0",
                    "url": "https://osv.dev/vulnerability/GHSA-test-0000-0001",
                    "summary": "Example vulnerability",
                    "rank": 1,
                    "score": 0.9,
                }
            ],
            "num_chunks": 1,
            "num_sources": 1,
        }


def build_service(pipeline=None):
    return SupplyGuardService(
        pipeline=pipeline or FakePipeline(),
        config=DEFAULT_CONFIG,
        corpus_size=1_000,
    )


def test_health_and_readiness_report_loaded_runtime():
    factory_calls = []
    service = build_service()
    app = create_app(service_factory=lambda: factory_calls.append(True) or service)

    with TestClient(app) as client:
        health = client.get("/healthz")
        readiness = client.get("/readyz")

    assert factory_calls == [True]
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "version": "0.1.0"}
    assert readiness.status_code == 200
    assert readiness.json() == {
        "status": "ready",
        "corpus_advisories": 1_000,
        "language_model": DEFAULT_CONFIG.models.language_model_name,
        "embedding_model": DEFAULT_CONFIG.models.embedding_model_name,
        "device": DEFAULT_CONFIG.models.device,
    }


def test_ask_defaults_to_hybrid_and_returns_trusted_sources():
    pipeline = FakePipeline()
    app = create_app(service_factory=lambda: build_service(pipeline))

    with TestClient(app) as client:
        response = client.post("/v1/ask", json={"question": "Which version is fixed?"})

    assert response.status_code == 200
    assert pipeline.calls == [("Which version is fixed?", "hybrid")]
    assert response.json() == {
        "question": "Which version is fixed?",
        "answer": "Upgrade to 1.2.3.",
        "retrieval_method": "hybrid",
        "sources": [
            {
                "advisory_id": "GHSA-test-0000-0001",
                "chunk_id": "GHSA-test-0000-0001_chunk_0",
                "url": "https://osv.dev/vulnerability/GHSA-test-0000-0001",
                "summary": "Example vulnerability",
                "rank": 1,
                "score": 0.9,
            }
        ],
        "num_chunks": 1,
        "num_sources": 1,
    }


def test_ask_accepts_an_explicit_retrieval_method():
    pipeline = FakePipeline()
    app = create_app(service_factory=lambda: build_service(pipeline))

    with TestClient(app) as client:
        response = client.post(
            "/v1/ask",
            json={"question": "Which version is fixed?", "retrieval_method": "bm25"},
        )

    assert response.status_code == 200
    assert pipeline.calls == [("Which version is fixed?", "bm25")]


def test_invalid_api_payload_is_rejected_before_inference():
    pipeline = FakePipeline()
    app = create_app(service_factory=lambda: build_service(pipeline))

    with TestClient(app) as client:
        response = client.post(
            "/v1/ask",
            json={
                "question": "Which version is fixed?",
                "retrieval_method": "unsupported",
                "unexpected": True,
            },
        )

    assert response.status_code == 422
    assert pipeline.calls == []
    response_body = response.json()
    assert response_body["detail"] == "Request validation failed"
    assert response_body["error_type"] == "RequestValidationError"
    assert {issue["location"] for issue in response_body["validation_errors"]} == {
        "body.retrieval_method",
        "body.unexpected",
    }


def test_domain_validation_errors_have_a_stable_api_shape():
    pipeline = FakePipeline(
        InputValidationError("Question contains unsupported control characters")
    )
    app = create_app(service_factory=lambda: build_service(pipeline))

    with TestClient(app) as client:
        response = client.post("/v1/ask", json={"question": "unsafe"})

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Question contains unsupported control characters",
        "error_type": "InputValidationError",
    }


def test_model_failures_are_reported_as_service_unavailable():
    pipeline = FakePipeline(ModelGenerationError("generation failed"))
    app = create_app(service_factory=lambda: build_service(pipeline))

    with TestClient(app) as client:
        response = client.post("/v1/ask", json={"question": "Which version is fixed?"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "generation failed",
        "error_type": "ModelGenerationError",
    }
