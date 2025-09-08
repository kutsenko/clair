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
