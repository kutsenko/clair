import sys
from pathlib import Path

import ollama_send


def test_output_to_named_file(tmp_path, monkeypatch):
    def fake_send(*args, **kwargs):
        print("answer")
        return "answer"

    monkeypatch.setattr(ollama_send, "send_with_fallback", fake_send)
    outfile = tmp_path / "result.txt"
    monkeypatch.setattr(sys, "argv", ["ollama_send.py", "-p", "hi", "-o", str(outfile)])
    monkeypatch.chdir(tmp_path)
    ollama_send.main()
    assert outfile.read_text(encoding="utf-8") == "answer"


def test_output_default_filename(tmp_path, monkeypatch):
    def fake_send(*args, **kwargs):
        print("hello")
        return "hello"

    monkeypatch.setattr(ollama_send, "send_with_fallback", fake_send)
    infile = tmp_path / "sample.bin"
    infile.write_text("data", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["ollama_send.py", "-p", "hi", "-f", str(infile), "-o"],
    )
    monkeypatch.chdir(tmp_path)
    ollama_send.main()
    out_file = Path(str(infile) + ".txt")
    assert out_file.read_text(encoding="utf-8") == "hello"
