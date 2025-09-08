import sys
import pytest

from ollama_send import main


def test_openai_backend_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_AP_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog", "-p", "hi", "-b", "openai"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "OPENAI_AP_KEY" in err
