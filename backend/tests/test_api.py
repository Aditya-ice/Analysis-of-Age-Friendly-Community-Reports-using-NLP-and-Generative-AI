from elderhelp.config import Settings
from elderhelp.main import create_app
from fastapi.testclient import TestClient


def client() -> TestClient:
    settings = Settings(database_url="sqlite+aiosqlite://", google_cloud_project=None)
    from uuid import uuid4

    from elderhelp.v2.auth import Pilot, authenticate

    app = create_app(settings, ranker=object())
    app.dependency_overrides[authenticate] = lambda: Pilot(uuid4(), uuid4())
    return TestClient(app)


def test_health_is_independent_from_readiness():
    with client() as test_client:
        health = test_client.get("/healthz")
        readiness = test_client.get("/readyz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.headers["X-Request-ID"]
    assert readiness.status_code == 503


def test_answer_rejects_whitespace_question():
    with client() as test_client:
        response = test_client.post("/v1/answers/stream", json={"question": "   "})
    assert response.status_code == 422


def test_answer_reports_missing_provider_before_streaming():
    with client() as test_client:
        response = test_client.post("/v1/answers/stream", json={"question": "Housing?"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Verified answers are unavailable; use keyword search"


def test_request_logs_do_not_contain_question(caplog):
    private_question = "private question text"
    with caplog.at_level("INFO", logger="elderhelp.http"), client() as test_client:
        test_client.post("/v1/answers/stream", json={"question": private_question})
    assert private_question not in caplog.text
