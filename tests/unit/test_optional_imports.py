import sys
import types

import pytest

import clair


def test_optional_import_returns_existing_module(monkeypatch):
    module_name = "_clair_dummy_module"
    dummy = types.ModuleType(module_name)
    sys.modules[module_name] = dummy
    try:
        # Ensure we would notice an unexpected spec lookup
        monkeypatch.setattr(
            clair.util,
            "find_spec",
            lambda *args, **kwargs: pytest.fail("find_spec should not be called"),
        )

        result = clair._optional_import(module_name)
        assert result is dummy
    finally:
        sys.modules.pop(module_name, None)


def test_optional_import_returns_none_when_missing(monkeypatch):
    module_name = "_clair_missing_module"
    sys.modules.pop(module_name, None)

    monkeypatch.setattr(clair.util, "find_spec", lambda name: None)

    assert clair._optional_import(module_name) is None


def test_optional_import_imports_when_available(monkeypatch):
    module_name = "_clair_available_module"
    sys.modules.pop(module_name, None)

    sentinel = object()

    class DummySpec:
        pass

    monkeypatch.setattr(clair.util, "find_spec", lambda name: DummySpec())
    monkeypatch.setattr(clair, "import_module", lambda name: sentinel)

    assert clair._optional_import(module_name) is sentinel
