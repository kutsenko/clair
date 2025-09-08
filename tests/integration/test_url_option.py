import sys
import pytest
import ollama_send
from tests.conftest import DummyResponse


def test_url_fetch(monkeypatch):
    def fake_get(url, timeout=None):
        return DummyResponse(text="webdata", headers={"Content-Type": "text/plain"})

    captured = {}

    def fake_send(base_url, payload, images_present, user_content, stream):
        captured['content'] = user_content
        return "ok"

    monkeypatch.setattr(ollama_send.requests, "get", fake_get)
    monkeypatch.setattr(ollama_send, "send_with_fallback", fake_send)
    monkeypatch.setattr(sys, "argv", ["ollama_send.py", "-p", "hi", "--url", "http://example.com"])
    ollama_send.main()
    assert "webdata" in captured['content']


def test_url_conflicts_with_file(tmp_path, monkeypatch):
    f = tmp_path / "a.txt"
    f.write_text("data", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["ollama_send.py", "-p", "hi", "-f", str(f), "--url", "http://example.com"],
    )
    with pytest.raises(SystemExit):
        ollama_send.main()


def test_url_fetch_image_auto(monkeypatch):
    bin_data = b"img"

    def fake_get(url, timeout=None):
        return DummyResponse(content=bin_data, headers={"Content-Type": "image/png"})

    captured = {}

    def fake_send(base_url, payload, images_present, user_content, stream):
        captured['images'] = payload['messages'][0].get('images', [])
        return "ok"

    monkeypatch.setattr(ollama_send.requests, "get", fake_get)
    monkeypatch.setattr(ollama_send, "send_with_fallback", fake_send)
    monkeypatch.setattr(sys, "argv", ["ollama_send.py", "-p", "hi", "--url", "http://img"])
    ollama_send.main()
    assert len(captured['images']) == 1


def test_url_fetch_manual_type(monkeypatch):
    bin_data = b"data"

    def fake_get(url, timeout=None):
        return DummyResponse(content=bin_data, headers={"Content-Type": "text/plain"})

    captured = {}

    def fake_send(base_url, payload, images_present, user_content, stream):
        captured['images'] = payload['messages'][0].get('images', [])
        return "ok"

    monkeypatch.setattr(ollama_send.requests, "get", fake_get)
    monkeypatch.setattr(ollama_send, "send_with_fallback", fake_send)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ollama_send.py", "-p", "hi", "--url", "http://example.com", "-c", "image"],
    )
    ollama_send.main()
    assert len(captured['images']) == 1

