import sys
from pathlib import Path

import clair


def test_output_to_named_file(tmp_path, monkeypatch):
    def fake_send(*args, **kwargs):
        print("answer")
        return "answer"

    monkeypatch.setattr(clair, "send_with_fallback", fake_send)
    outfile = tmp_path / "result.txt"
    monkeypatch.setattr(sys, "argv", ["clair.py", "-p", "hi", "-o", str(outfile)])
    monkeypatch.chdir(tmp_path)
    clair.main()
    assert outfile.read_text(encoding="utf-8") == "answer"


def test_output_default_filename(tmp_path, monkeypatch):
    def fake_send(*args, **kwargs):
        print("hello")
        return "hello"

    monkeypatch.setattr(clair, "send_with_fallback", fake_send)
    infile = tmp_path / "sample.bin"
    infile.write_text("data", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["clair.py", "-p", "hi", "-f", str(infile), "-o"],
    )
    monkeypatch.chdir(tmp_path)
    clair.main()
    out_file = Path(str(infile) + ".txt")
    assert out_file.read_text(encoding="utf-8") == "hello"
