from fastapi.testclient import TestClient

from app.main import app


def test_root():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["service"] == "idp-ocr"


def test_health():
    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model"] == "PP-OCRv6_medium"


def test_rejects_bad_type():
    with TestClient(app) as client:
        r = client.post(
            "/api/ocr",
            files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
        )
        assert r.status_code == 400
