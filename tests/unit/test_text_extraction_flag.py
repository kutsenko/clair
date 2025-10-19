import sys

from clair import main


def test_pdf_default_skips_local_extraction(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

    def fake_extract(path):
        raise AssertionError("local extraction should be disabled by default")

    def fake_preview(blob):
        return ["cHJldmlldy1pbWFnZQ=="], 4

    captured = {}

    def fake_send(host, payload, images_present, user_content, stream):
        captured["user_content"] = user_content
        captured["payload"] = payload
        return ""

    monkeypatch.setattr("clair.try_extract_pdf_text", fake_extract)
    monkeypatch.setattr("clair.convert_pdf_blob_to_image_previews", fake_preview)
    monkeypatch.setattr("clair.send_with_fallback", fake_send)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "-p", "hi", "-f", str(pdf_path)],
    )

    main()

    assert "hi" in captured["user_content"]
    assert "sample.pdf" in captured["user_content"]
    message = captured["payload"]["messages"][0]
    images = message["images"]
    assert images == ["cHJldmlldy1pbWFnZQ=="]
    assert "documents" not in message


def test_pdf_extract_text_flag_enables_local_tools(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

    calls = {"extract": False}
    captured = {}

    def fake_extract(path):
        calls["extract"] = True
        return "EXTRACTED CONTENT"

    def fail_preview(blob):
        raise AssertionError("preview should not be generated when --extract-text is set")

    def fake_send(host, payload, images_present, user_content, stream):
        assert "EXTRACTED CONTENT" in user_content
        captured["payload"] = payload
        return ""

    monkeypatch.setattr("clair.try_extract_pdf_text", fake_extract)
    monkeypatch.setattr("clair.convert_pdf_blob_to_image_previews", fail_preview)
    monkeypatch.setattr("clair.send_with_fallback", fake_send)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "-p", "hi", "-f", str(pdf_path), "--extract-text"],
    )

    main()

    assert calls["extract"] is True
    message = captured["payload"]["messages"][0]
    assert "images" not in message
    assert "documents" not in message
