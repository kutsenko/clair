import sys

import clair


def test_frame_by_frame_mode(monkeypatch, tmp_path, capsys):
    # Prepare a dummy video file
    video = tmp_path / "video.mp4"
    video.write_bytes(b"data")

    # Mock frame extraction to return two frames
    monkeypatch.setattr(
        clair,
        "extract_video_frames_b64",
        lambda *args, **kwargs: ["f1", "f2"],
    )

    # Capture payloads sent to API
    calls = []

    def fake_send(host, payload, images_present, user_content, stream):
        calls.append(payload)
        result = f"resp{len(calls)}"
        print(result)
        return result

    monkeypatch.setattr(clair, "send_with_fallback", fake_send)
    monkeypatch.setattr(
        sys,
        "argv",
        ["clair.py", "-p", "hi", "-f", str(video), "--frame-by-frame"],
    )
    monkeypatch.chdir(tmp_path)

    clair.main()

    # Ensure one request per frame
    assert len(calls) == 2
    assert calls[0]["messages"][0]["images"] == ["f1"]
    assert calls[1]["messages"][0]["images"] == ["f2"]

    # Responses printed in order
    captured = capsys.readouterr()
    assert captured.out == "resp1\nresp2\n"
