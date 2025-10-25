import sys
import pytest

from clair import main


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def set_argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["prog", *args])


def test_openai_backend_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    set_argv(monkeypatch, "-p", "hi", "-b", "openai")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err


def test_openai_models_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    set_argv(monkeypatch, "--list-models", "openai")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err


def test_openai_models_list(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "token")
    set_argv(monkeypatch, "--list-models", "openai")

    def fake_get(url, headers=None, timeout=None):
        assert url == "https://api.openai.com/v1/models"
        assert headers == {"Authorization": "Bearer token"}
        return DummyResponse({"data": [{"id": "gpt-test"}]})

    monkeypatch.setattr("clair.requests.get", fake_get)
    main()
    out = capsys.readouterr().out
    assert "gpt-test" in out


def test_huggingface_models_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    set_argv(monkeypatch, "--list-models", "huggingface")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "HUGGINGFACE_API_KEY" in err


def test_huggingface_models_list(monkeypatch, capsys):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "token")
    set_argv(monkeypatch, "--list-models", "huggingface")

    def fake_get(url, headers=None, timeout=None):
        assert url == "https://api-inference.huggingface.co/v1/models"
        assert headers == {"Authorization": "Bearer token"}
        return DummyResponse({"data": [{"id": "hf-test"}]})

    monkeypatch.setattr("clair.requests.get", fake_get)
    main()
    out = capsys.readouterr().out
    assert "hf-test" in out


def test_xai_models_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    set_argv(monkeypatch, "--list-models", "xai")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "XAI_API_KEY" in err


def test_xai_models_list(monkeypatch, capsys):
    monkeypatch.setenv("XAI_API_KEY", "token")
    set_argv(monkeypatch, "--list-models", "xai")

    def fake_get(url, headers=None, timeout=None):
        assert url == "https://api.x.ai/v1/models"
        assert headers == {"Authorization": "Bearer token"}
        return DummyResponse({"data": [{"id": "xai-test"}]})

    monkeypatch.setattr("clair.requests.get", fake_get)
    main()
    out = capsys.readouterr().out
    assert "xai-test" in out


def test_gemini_models_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    set_argv(monkeypatch, "--list-models", "gemini")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "GEMINI_API_KEY" in err


def test_gemini_models_list(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "token")
    set_argv(monkeypatch, "--list-models", "gemini")
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params))
        if len(calls) == 1:
            return DummyResponse(
                {"models": [{"name": "models/gemini-1"}], "nextPageToken": "next"}
            )
        assert params.get("pageToken") == "next"
        return DummyResponse({"models": [{"name": "models/gemini-pro"}]})

    monkeypatch.setattr("clair.requests.get", fake_get)
    main()
    out = capsys.readouterr().out
    assert "models/gemini-1" in out
    assert "models/gemini-pro" in out
    assert calls[0]["key"] == "token"


def test_ollama_models_list(monkeypatch, capsys):
    set_argv(monkeypatch, "--list-models", "ollama")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    def fake_get(url, timeout=None):
        assert url == "http://localhost:11434/api/tags"
        return DummyResponse({"models": [{"name": "ollama-test"}]})

    monkeypatch.setattr("clair.requests.get", fake_get)
    main()
    out = capsys.readouterr().out
    assert "ollama-test" in out


def test_huggingface_backend_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    set_argv(monkeypatch, "-p", "hi", "-b", "huggingface")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "HUGGINGFACE_API_KEY" in err


def test_xai_backend_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    set_argv(monkeypatch, "-p", "hi", "-b", "xai")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "XAI_API_KEY" in err


def test_gemini_backend_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    set_argv(monkeypatch, "-p", "hi", "-b", "gemini")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "GEMINI_API_KEY" in err


def test_claude_models_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    set_argv(monkeypatch, "--list-models", "claude")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err


def test_claude_models_list(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "token")
    set_argv(monkeypatch, "--list-models", "claude")

    def fake_get(url, headers=None, timeout=None):
        assert url == "https://api.anthropic.com/v1/models"
        assert headers == {"Authorization": "Bearer token"}
        return DummyResponse({"data": [{"id": "claude-3-opus"}]})

    monkeypatch.setattr("clair.requests.get", fake_get)
    main()
    out = capsys.readouterr().out
    assert "claude-3-opus" in out
