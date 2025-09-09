import base64
import logging
import types
import sys

from clair import extract_video_frames_b64


def test_extract_video_frames_logging(monkeypatch, caplog):
    class DummyCapture:
        def __init__(self, path):
            self.path = path
        def isOpened(self):
            return True
        def get(self, prop):
            return 1
        def set(self, prop, idx):
            pass
        def read(self):
            return True, DummyFrame()
        def release(self):
            pass
    class DummyFrame:
        shape = (1, 1, 3)
    class DummyBuf:
        def tobytes(self):
            return b'data'
    def video_capture(path):
        return DummyCapture(path)
    def imencode(ext, frame):
        return True, DummyBuf()
    dummy_cv2 = types.SimpleNamespace(
        VideoCapture=video_capture,
        imencode=imencode,
        CAP_PROP_FRAME_COUNT=0,
        CAP_PROP_POS_FRAMES=0,
        INTER_AREA=0,
    )
    monkeypatch.setitem(sys.modules, "cv2", dummy_cv2)
    with caplog.at_level(logging.INFO, logger="clair"):
        frames = extract_video_frames_b64("video.mp4", max_frames=1)
    assert frames == [base64.b64encode(b"data").decode("ascii")]
    assert "Extracted 1 frames from 'video.mp4'" in caplog.text
