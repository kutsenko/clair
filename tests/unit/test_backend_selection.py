import sys
import pytest

from clair import main


def test_openai_backend_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog", "-p", "hi", "-b", "openai"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err


def test_openai_models_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog", "--openai-models"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err


def test_openai_models_exclusive(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(sys, "argv", ["prog", "--openai-models", "-b", "openai"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--openai-models cannot be combined" in err


def test_huggingface_backend_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog", "-p", "hi", "-b", "huggingface"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "HUGGINGFACE_API_KEY" in err


def test_xai_backend_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog", "-p", "hi", "-b", "xai"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "XAI_API_KEY" in err
