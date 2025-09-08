import os
from pathlib import Path

from ollama_send import is_image, is_video, read_text_file, build_user_content


def test_is_image_and_video_detection():
    assert is_image("photo.jpg")
    assert not is_image("document.txt")
    assert is_video("clip.mp4")
    assert not is_video("image.png")


def test_read_text_file_truncation(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("1234567890ABC", encoding="utf-8")
    content, truncated = read_text_file(str(p), max_chars=10)
    assert truncated is True
    assert content.endswith("[... truncated ...]")


def test_build_user_content():
    prompt = "Hello"
    attachments = [("file.txt", "content")]
    notes = ["note"]
    result = build_user_content(prompt, attachments, notes)
    assert "Hello" in result
    assert "file.txt" in result
    assert "content" in result
    assert "Video Notes" in result
    assert "- note" in result
