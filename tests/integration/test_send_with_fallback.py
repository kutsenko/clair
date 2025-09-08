from unittest.mock import patch

from ollama_send import send_with_fallback
from tests.conftest import DummyResponse


def test_fallback_to_generate_on_404(capsys):
    base_url = "http://testserver"
    payload_chat = {
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    responses = [
        DummyResponse(status_code=404, text="not found"),
        DummyResponse(status_code=200, json_data={"response": "hi"}),
    ]

    def fake_post(url, json=None, stream=False, timeout=None):
        return responses.pop(0)

    with patch("ollama_send.requests.post", side_effect=fake_post):
        send_with_fallback(base_url, payload_chat, images_present=False, user_content="hello", stream=False)

    captured = capsys.readouterr()
    assert captured.out.strip() == "hi"


def test_chat_endpoint_success(capsys):
    base_url = "http://testserver"
    payload_chat = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }

    responses = [
        DummyResponse(status_code=200, json_data={"message": {"content": "chat"}})
    ]

    def fake_post(url, json=None, stream=False, timeout=None):
        return responses.pop(0)

    with patch("ollama_send.requests.post", side_effect=fake_post):
        send_with_fallback(base_url, payload_chat, images_present=False, user_content="hi", stream=False)

    captured = capsys.readouterr()
    assert captured.out.strip() == "chat"
