import sys
import pytest
import clair
from tests.conftest import DummyResponse


def test_url_fetch(monkeypatch):
    def fake_get(url, timeout=None):
        return DummyResponse(text="webdata", headers={"Content-Type": "text/plain"})

    captured = {}

    def fake_send(base_url, payload, images_present, user_content, stream):
        captured['content'] = user_content
        return "ok"

    monkeypatch.setattr(clair.requests, "get", fake_get)
    monkeypatch.setattr(clair, "send_with_fallback", fake_send)
    monkeypatch.setattr(sys, "argv", ["clair.py", "-p", "hi", "--url", "http://example.com"])
    clair.main()
    assert "webdata" in captured['content']


def test_url_conflicts_with_file(tmp_path, monkeypatch):
    f = tmp_path / "a.txt"
    f.write_text("data", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["clair.py", "-p", "hi", "-f", str(f), "--url", "http://example.com"],
    )
    with pytest.raises(SystemExit):
        clair.main()


def test_url_fetch_image_auto(monkeypatch):
    bin_data = b"img"

    def fake_get(url, timeout=None):
        return DummyResponse(content=bin_data, headers={"Content-Type": "image/png"})

    captured = {}

    def fake_send(base_url, payload, images_present, user_content, stream):
        captured['images'] = payload['messages'][0].get('images', [])
        return "ok"

    monkeypatch.setattr(clair.requests, "get", fake_get)
    monkeypatch.setattr(clair, "send_with_fallback", fake_send)
    monkeypatch.setattr(sys, "argv", ["clair.py", "-p", "hi", "--url", "http://img"])
    clair.main()
    assert len(captured['images']) == 1


def test_url_fetch_manual_type(monkeypatch):
    bin_data = b"data"

    def fake_get(url, timeout=None):
        return DummyResponse(content=bin_data, headers={"Content-Type": "text/plain"})

    captured = {}

    def fake_send(base_url, payload, images_present, user_content, stream):
        captured['images'] = payload['messages'][0].get('images', [])
        return "ok"

    monkeypatch.setattr(clair.requests, "get", fake_get)
    monkeypatch.setattr(clair, "send_with_fallback", fake_send)
    monkeypatch.setattr(
        sys,
        "argv",
        ["clair.py", "-p", "hi", "--url", "http://example.com", "-t", "image"],
    )
    clair.main()
    assert len(captured['images']) == 1

