import sys
import types
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class DummyResponse:
    def __init__(self, status_code=200, text="", json_data=None, lines=None):
        self.status_code = status_code
        self.text = text
        self._json = json_data or {}
        self._lines = lines or []

    def json(self):
        return self._json

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP error {self.status_code}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def dummy_post(url, json=None, stream=False, timeout=None):
    return DummyResponse()


requests_stub = types.SimpleNamespace(post=dummy_post, Response=DummyResponse)
sys.modules.setdefault("requests", requests_stub)
