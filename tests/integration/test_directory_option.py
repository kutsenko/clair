import sys

import ollama_send


def test_directory_processes_files(tmp_path, monkeypatch):
    responses = {"a.txt": "resp_a", "b.txt": "resp_b"}

    def fake_send(*args, **kwargs):
        user_content = kwargs.get("user_content", "")
        # determine which file is being processed by checking filename in prompt
        for name in responses:
            if name in user_content:
                print(responses[name])
                return responses[name]
        return ""

    monkeypatch.setattr(ollama_send, "send_with_fallback", fake_send)

    dir_path = tmp_path / "files"
    dir_path.mkdir()
    for name in responses:
        (dir_path / name).write_text("data", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["ollama_send.py", "-p", "hi", "-d", str(dir_path)])
    monkeypatch.chdir(tmp_path)
    ollama_send.main()

    for name, resp in responses.items():
        outfile = dir_path / f"{name}.txt"
        assert outfile.read_text(encoding="utf-8") == resp


def test_directory_recurses(tmp_path, monkeypatch):
    responses = {"a.txt": "resp_a", "b.txt": "resp_b"}

    def fake_send(*args, **kwargs):
        user_content = kwargs.get("user_content", "")
        for name, resp in responses.items():
            if name in user_content:
                print(resp)
                return resp
        return ""

    monkeypatch.setattr(ollama_send, "send_with_fallback", fake_send)

    root = tmp_path / "files"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("data", encoding="utf-8")
    (root / "sub" / "b.txt").write_text("data", encoding="utf-8")

    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        ["ollama_send.py", "-p", "hi", "-d", str(root), "-o", str(out_dir)],
    )
    monkeypatch.chdir(tmp_path)
    ollama_send.main()

    assert (out_dir / "a.txt.txt").read_text(encoding="utf-8") == "resp_a"
    assert (out_dir / "sub" / "b.txt.txt").read_text(encoding="utf-8") == "resp_b"
