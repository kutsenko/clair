import sys

from clair import main


def test_pdf_default_skips_local_extraction(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

    def fake_extract(path):
        raise AssertionError("local extraction should be disabled by default")

    captured = {}

    def fake_send(host, payload, images_present, user_content, stream):
        captured["user_content"] = user_content
        return ""

    monkeypatch.setattr("clair.try_extract_pdf_text", fake_extract)
    monkeypatch.setattr("clair.send_with_fallback", fake_send)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "-p", "hi", "-f", str(pdf_path)],
    )

    main()

    assert "hi" in captured["user_content"]
    assert "sample.pdf" in captured["user_content"]


def test_pdf_extract_text_flag_enables_local_tools(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

    calls = {"extract": False}

    def fake_extract(path):
        calls["extract"] = True
        return "EXTRACTED CONTENT"

    def fake_send(host, payload, images_present, user_content, stream):
        assert "EXTRACTED CONTENT" in user_content
        return ""

    monkeypatch.setattr("clair.try_extract_pdf_text", fake_extract)
    monkeypatch.setattr("clair.send_with_fallback", fake_send)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "-p", "hi", "-f", str(pdf_path), "--extract-text"],
    )

    main()

    assert calls["extract"] is True
