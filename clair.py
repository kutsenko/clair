#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Send a prompt plus arbitrary files (including MP4) to Ollama.
- Prefers /api/chat; falls back to /api/generate on 404.
- Images -> base64 in the "images" field (for vision models).
- Text files -> appended to the prompt as code blocks.
- PDF/DOCX -> optional text extraction with --extract-text (PyPDF2 / python-docx).
- MP4 -> extracts N frames (PNG, base64) and appends them to "images".

Compatible with Ollama 0.11.10 (REST).
Extended tracing via --verbose/--debug: endpoints, duration, status/body, error path.
"""

import argparse
import base64
import io
import json
import logging
import mimetypes
import os
import sys
import tempfile
import textwrap
import time
from importlib import import_module, util
from typing import Any, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests


# --------------------------- Logging / Tracing ---------------------------

LOG = logging.getLogger("clair")


def setup_logging(verbosity: int) -> None:
    """
    verbosity 0 = WARNING, 1 = INFO, 2+ = DEBUG
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    handler = logging.StreamHandler(sys.stderr)
    fmt = "[%(levelname)s] %(message)s"
    if level == logging.DEBUG:
        fmt = "[%(levelname)s] %(asctime)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    LOG.setLevel(level)
    LOG.handlers.clear()
    LOG.addHandler(handler)


# ---------------------- Optional parsers for PDF/DOCX --------------------


def _optional_import(module_name: str) -> Optional[Any]:
    if module_name in sys.modules:
        return sys.modules[module_name]
    if util.find_spec(module_name) is None:
        return None
    return import_module(module_name)


def try_extract_pdf_text(path: str) -> str:
    module = _optional_import("PyPDF2")
    if module is None:
        LOG.debug("PyPDF2 not installed – PDF will not be extracted.")
        return ""
    try:
        text = []
        with open(path, "rb") as f:
            reader = module.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        out = "\n".join(text).strip()
        LOG.info("Extracted PDF text: %s (%d chars)", os.path.basename(path), len(out))
        return out
    except Exception as e:
        LOG.warning("PDF extraction failed (%s): %s", path, e)
        return ""


def try_extract_docx_text(path: str) -> str:
    module = _optional_import("docx")
    if module is None:
        LOG.debug("python-docx not installed – DOCX will not be extracted.")
        return ""
    try:
        with open(path, "rb") as f:
            blob = f.read()  # ensure closing
        doc = module.Document(io.BytesIO(blob))  # no open file handle
        out = "\n".join(p.text for p in doc.paragraphs).strip()
        LOG.info("Extracted DOCX text: %s (%d chars)", os.path.basename(path), len(out))
        return out
    except Exception as e:
        LOG.warning("DOCX extraction failed (%s): %s", path, e)
        return ""


# -------------------- Video (MP4) -> Frames (base64 PNG) -----------------


def extract_video_frames_b64(
    path: str,
    max_frames: int = 8,
    target_width: Optional[int] = 640,
) -> List[str]:
    """
    Extracts up to max_frames evenly distributed frames and returns them
    as base64-encoded PNG strings. Requires opencv-python.
    """
    cv2_module = _optional_import("cv2")
    if cv2_module is None:
        LOG.error("opencv-python is required for video frames.")
        return []

    LOG.info("Video detected, extracting frames: %s", path)
    cap = cv2_module.VideoCapture(path)
    if not cap.isOpened():
        LOG.error("Could not open video: %s", path)
        return []

    try:
        frame_count = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            LOG.error("No valid frames in: %s", path)
            return []

        num = max(1, min(max_frames, frame_count))
        indices = (
            [int(i * (frame_count - 1) / (num - 1)) for i in range(num)]
            if num > 1
            else [0]
        )

        images_b64: List[str] = []
        for idx in indices:
            cap.set(cv2_module.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                LOG.debug("Frame %d could not be read.", idx)
                continue

            if target_width and frame.shape[1] > target_width:
                h, w = frame.shape[:2]
                scale = target_width / float(w)
                new_w = target_width
                new_h = int(h * scale)
                frame = cv2_module.resize(
                    frame,
                    (new_w, new_h),
                    interpolation=cv2_module.INTER_AREA,
                )

            ok, buf = cv2_module.imencode(".png", frame)
            if not ok:
                LOG.debug("Frame %d could not be encoded as PNG.", idx)
                continue
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            images_b64.append(b64)

        LOG.info(
            "Extracted %d frames from '%s' (target width: %s)",
            len(images_b64),
            os.path.basename(path),
            f"{target_width}px" if target_width else "original",
        )
        return images_b64
    finally:
        cap.release()
        del cap  # optional


# ------------------------------ Utilities --------------------------------

TEXT_LIKE_EXT = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".css",
    ".js",
    ".ts",
    ".py",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".ini",
    ".conf",
    ".log",
}

PDF_PREVIEW_MAX_PAGES = 3
DOCX_PREVIEW_MAX_IMAGES = 3
DOCX_PREVIEW_WRAP = 90
DOCX_PREVIEW_LINES_PER_IMAGE = 40


def render_text_block_to_image_b64(lines: Iterable[str]) -> Optional[str]:
    """Render wrapped text lines to a PNG image encoded as base64."""

    image_module = _optional_import("PIL.Image")
    draw_module = _optional_import("PIL.ImageDraw")
    font_module = _optional_import("PIL.ImageFont")

    if not image_module or not draw_module or not font_module:
        LOG.debug("Pillow is required to render text previews.")
        return None

    try:
        font = font_module.load_default()
    except Exception as exc:  # pragma: no cover - extremely unlikely
        LOG.warning("Failed to load Pillow default font: %s", exc)
        return None

    # Calculate approximate canvas size.
    lines_list = list(lines)
    if not lines_list:
        lines_list = [""]

    try:
        sample_bbox = font.getbbox("Mg")
        char_height = sample_bbox[3] - sample_bbox[1]
        char_width = sample_bbox[2] - sample_bbox[0]
    except Exception:  # pragma: no cover - defensive guard
        char_height = 12
        char_width = 7

    max_line_len = max(len(line) for line in lines_list)
    width = max(640, min(2048, 40 + char_width * max_line_len))
    line_height = char_height + 6
    height = max(80, 40 + line_height * len(lines_list))

    image = image_module.new("RGB", (width, height), "white")
    draw = draw_module.Draw(image)

    y = 20
    for line in lines_list:
        draw.text((20, y), line, fill="black", font=font)
        y += line_height

    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")


def wrap_text_for_preview(text: str, *, width: int) -> List[str]:
    wrapper = textwrap.TextWrapper(
        width=width,
        replace_whitespace=False,
        drop_whitespace=False,
    )
    lines: List[str] = []
    for paragraph in text.splitlines():
        if not paragraph:
            lines.append("")
            continue
        wrapped = wrapper.wrap(paragraph)
        if wrapped:
            lines.extend(wrapped)
        else:
            lines.append("")
    if not lines:
        lines.append("")
    return lines


def convert_docx_blob_to_image_previews(
    blob: bytes,
    *,
    max_chars: int,
    wrap: int = DOCX_PREVIEW_WRAP,
    max_images: int = DOCX_PREVIEW_MAX_IMAGES,
    lines_per_image: int = DOCX_PREVIEW_LINES_PER_IMAGE,
) -> Tuple[List[str], bool]:
    """
    Render a DOCX document to up to ``max_images`` PNG previews.

    Returns the list of base64 images and a boolean indicating whether the
    preview text had to be truncated due to ``max_chars``.
    """

    docx_module = _optional_import("docx")
    if docx_module is None:
        LOG.debug("python-docx is required for DOCX previews.")
        return [], False

    try:
        document = docx_module.Document(io.BytesIO(blob))
    except Exception as exc:
        LOG.warning("Could not open DOCX for preview: %s", exc)
        return [], False

    text = "\n".join(p.text for p in document.paragraphs).strip()
    if not text:
        return [], False

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n\n[... truncated for preview ...]"

    wrapped_lines = wrap_text_for_preview(text, width=wrap)
    chunks: List[List[str]] = []
    for idx in range(0, len(wrapped_lines), lines_per_image):
        if len(chunks) >= max_images:
            break
        chunks.append(wrapped_lines[idx : idx + lines_per_image])

    previews: List[str] = []
    for chunk in chunks:
        image_b64 = render_text_block_to_image_b64(chunk)
        if image_b64:
            previews.append(image_b64)

    if not previews:
        LOG.debug("DOCX preview rendering produced no images (Pillow missing?).")

    return previews, truncated


def convert_pdf_blob_to_image_previews(
    blob: bytes,
    *,
    max_pages: int = PDF_PREVIEW_MAX_PAGES,
    dpi: int = 200,
) -> Tuple[List[str], int]:
    """
    Render the first ``max_pages`` of a PDF as PNG previews.

    Returns a tuple of (images, total_pages_in_document).
    """

    pdfium = _optional_import("pypdfium2")
    if pdfium is None:
        LOG.debug("pypdfium2 is required for PDF previews.")
        return [], 0

    try:
        pdf = pdfium.PdfDocument(io.BytesIO(blob))
    except Exception as exc:
        LOG.warning("Could not open PDF for preview: %s", exc)
        return [], 0

    total_pages = len(pdf)
    previews: List[str] = []

    try:
        render_pages = min(total_pages, max_pages)
        for index in range(render_pages):
            page = pdf[index]
            try:
                bitmap = page.render(scale=dpi / 72)
                try:
                    image = bitmap.to_pil()
                except Exception as exc:
                    LOG.warning(
                        "PDF preview rendering requires Pillow (page %d): %s",
                        index + 1,
                        exc,
                    )
                    return [], total_pages
            except Exception as exc:
                LOG.warning("Failed to render PDF page %d: %s", index + 1, exc)
                continue
            finally:
                try:
                    bitmap.close()
                except Exception:
                    pass
                try:
                    page.close()
                except Exception:
                    pass

            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                previews.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    finally:
        try:
            pdf.close()
        except Exception:
            pass

    if not previews:
        LOG.debug("No PDF preview images were generated.")

    return previews, total_pages
VIDEO_EXT = {".mp4"}  # extend as needed: .mov, .mkv ...
IMAGE_MIME_PREFIXES = ("image/",)


def _ensure_image_mime(mime: Optional[str]) -> str:
    if mime and mime.startswith("image/"):
        return mime
    return "image/png"


def is_image(path: str) -> bool:
    mime, _ = mimetypes.guess_type(path)
    return bool(mime and mime.startswith(IMAGE_MIME_PREFIXES))


def is_video(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in VIDEO_EXT


def infer_content_type(header: str) -> str:
    header = header.split(";")[0].strip().lower()
    if header.startswith("image/"):
        return "image"
    if header.startswith("video/"):
        return "video"
    return "doc"


def read_text_file(
    path: str, encoding: str = "utf-8", max_chars: int = 200_000
) -> Tuple[str, bool]:
    try:
        with open(path, "r", encoding=encoding, errors="replace") as f:
            data = f.read()
        truncated = False
        if len(data) > max_chars:
            data = data[:max_chars] + "\n\n[... truncated ...]"
            truncated = True
        LOG.info(
            "Read text file: %s (%d chars%s)",
            os.path.basename(path),
            len(data),
            ", truncated" if truncated else "",
        )
        return data, truncated
    except Exception as e:
        LOG.warning("Failed to read as text (%s): %s", path, e)
        return f"[ERROR reading as text: {e}]", False


def to_base64(path: str) -> str:
    with open(path, "rb") as f:
        blob = f.read()
    LOG.debug("File base64-encoded: %s (%d bytes)", os.path.basename(path), len(blob))
    return base64.b64encode(blob).decode("ascii")


def build_user_content(
    prompt: str, text_attachments: List[Tuple[str, str]], video_notes: List[str]
) -> str:
    parts = [prompt]
    if text_attachments:
        parts.append("\n\n---\n### Attachments (Text)")
        for fname, txt in text_attachments:
            parts.append(f"\n**{fname}**:\n```\n{txt}\n```")
    if video_notes:
        parts.append("\n\n---\n### Video Notes")
        for note in video_notes:
            parts.append(f"- {note}")
    return "\n".join(parts)


def build_gemini_contents(text: str, images: List[Tuple[str, str]]) -> List[dict]:
    parts: List[dict] = []
    if text:
        parts.append({"text": text})
    for b64, mime in images:
        parts.append(
            {"inline_data": {"mime_type": _ensure_image_mime(mime), "data": b64}}
        )
    return [{"role": "user", "parts": parts or [{"text": ""}]}]


# ------------------------- HTTP + Endpoint-Fallback -----------------------


def _post_json(
    url: str, payload: dict, *, stream: bool = False, timeout: int = 1200
) -> requests.Response:
    LOG.debug("POST %s | stream=%s | timeout=%s", url, stream, timeout)
    LOG.debug("Payload keys: %s", list(payload.keys()))
    start = time.monotonic()
    resp = requests.post(url, json=payload, stream=stream, timeout=timeout)
    dur = time.monotonic() - start
    LOG.info("Response %s in %.3fs from %s", resp.status_code, dur, url)
    if resp.status_code >= 400:
        # Body (max 2k) for diagnostics
        try:
            body = resp.text
            if len(body) > 2048:
                body = body[:2048] + "\n...[truncated]..."
        except Exception:
            body = "<Body not readable>"
        LOG.warning("Error body (%s):\n---\n%s\n---", url, body)
    return resp


def _read_nonstream_json_fallback(resp: requests.Response) -> dict:
    """
    Robust against misconfigured streaming servers: tries resp.json() first,
    then falls back to NDJSON/line parsing (uses the last valid JSON block).
    """
    try:
        return resp.json()
    except ValueError:
        chunks = [ln for ln in resp.text.splitlines() if ln.strip()]
        for ln in reversed(chunks):
            try:
                return json.loads(ln)
            except Exception:
                continue
        raise


def send_with_fallback(
    base_url: str,
    payload_chat: dict,
    images_present: bool,
    user_content: str,
    stream: bool,
) -> str:
    """
    Tries /api/chat first. On 404:
      - if "model not found" -> clear message
      - otherwise fall back to /api/generate (prompt + images)
    Outputs the model response to stdout (streaming or final block) and
    returns it as a string.
    """
    base_url = base_url.rstrip("/")
    chat_url = base_url + "/api/chat"
    gen_url = base_url + "/api/generate"

    # 1) /api/chat
    try:
        if stream:
            buffer: List[str] = []
            with _post_json(chat_url, payload_chat, stream=True) as r:
                if r.status_code == 404:
                    body = r.text.lower()
                    if "model" in body and "not found" in body:
                        model = payload_chat.get("model", "<unknown>")
                        LOG.error(
                            "Model '%s' not found. Please run: ollama pull %s",
                            model,
                            model,
                        )
                        print(
                            f"[ERROR] Model '{model}' not found. Please run 'ollama pull {model}'.",
                            file=sys.stderr,
                        )
                        return ""
                    LOG.info(
                        "Endpoint /api/chat not available – falling back to /api/generate."
                    )
                else:
                    r.raise_for_status()
                    LOG.info("Streaming from /api/chat started ...")
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except Exception:
                            print(line)
                            continue
                        if "message" in ev and "content" in ev["message"]:
                            chunk = ev["message"]["content"]
                            print(chunk, end="", flush=True)
                            buffer.append(chunk)
                        if ev.get("done"):
                            print()
                            return "".join(buffer)
        else:
            with _post_json(chat_url, payload_chat, timeout=600) as r:
                if r.status_code == 404:
                    body = r.text.lower()
                    if "model" in body and "not found" in body:
                        model = payload_chat.get("model", "<unknown>")
                        LOG.error(
                            "Model '%s' not found. Please run: ollama pull %s",
                            model,
                            model,
                        )
                        print(
                            f"[ERROR] Model '{model}' not found. Please run 'ollama pull {model}'.",
                            file=sys.stderr,
                        )
                        return ""
                    LOG.info(
                        "Endpoint /api/chat not available – falling back to /api/generate."
                    )
                else:
                    r.raise_for_status()
                    data = _read_nonstream_json_fallback(r)
                    content = data.get("message", {}).get("content", "")
                    print(content)
                    return content
    except requests.RequestException as e:
        LOG.warning("/api/chat failed (%s). Trying fallback.", e)

    # 2) /api/generate (Fallback)
    payload_generate = {
        "model": payload_chat.get("model"),
        "prompt": user_content,
        "stream": True if stream else False,  # <<< IMPORTANT
    }
    if images_present:
        imgs = payload_chat["messages"][0].get("images", [])
        if imgs:
            payload_generate["images"] = imgs
    documents = payload_chat["messages"][0].get("documents")
    if documents:
        payload_generate["documents"] = documents

    try:
        if stream:
            buffer: List[str] = []
            with _post_json(gen_url, payload_generate, stream=True) as r:
                r.raise_for_status()
                LOG.info("Streaming from /api/generate started ...")
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        print(line)
                        continue
                    if "response" in ev:
                        chunk = ev["response"]
                        print(chunk, end="", flush=True)
                        buffer.append(chunk)
                    if ev.get("done"):
                        print()
                        return "".join(buffer)
        else:
            with _post_json(gen_url, payload_generate, timeout=600) as r:
                if r.status_code >= 400:
                    r.raise_for_status()
                data = _read_nonstream_json_fallback(r)
                content = data.get("response", "")
                print(content)
                return content
    except requests.RequestException as e:
        LOG.error("Fallback /api/generate failed: %s", e)
        print(
            f"[ERROR] Request to Ollama failed (fallback /api/generate): {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    return ""


# ------------------------------- Gemini ----------------------------------


def _extract_gemini_text(payload: dict) -> str:
    chunks: List[str] = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                chunks.append(text)
    return "".join(chunks)


def send_gemini(
    base_url: str, payload: dict, *, model: str, api_key: str, stream: bool
) -> str:
    endpoint = "streamGenerateContent" if stream else "generateContent"
    url = base_url.rstrip("/") + f"/v1beta/models/{model}:{endpoint}?key={api_key}"
    headers = {"Content-Type": "application/json"}
    try:
        if stream:
            buffer: List[str] = []
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=600
            ) as r:
                r.raise_for_status()
                LOG.info("Streaming from Gemini started ...")
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[len("data: ") :]
                    try:
                        event = json.loads(line)
                    except Exception:
                        print(line)
                        continue
                    chunk = _extract_gemini_text(event)
                    if chunk:
                        print(chunk, end="", flush=True)
                        buffer.append(chunk)
            if buffer:
                print()
            return "".join(buffer)
        else:
            with requests.post(url, headers=headers, json=payload, timeout=600) as r:
                r.raise_for_status()
                data = r.json()
                content = _extract_gemini_text(data)
                print(content)
                return content
    except requests.RequestException as e:
        LOG.error("Request to Gemini failed: %s", e)
        print(f"[ERROR] Request to Gemini failed: {e}", file=sys.stderr)
        sys.exit(1)

    return ""


# ------------------------------- OpenAI ----------------------------------


def _send_openai_style(
    base_url: str, payload: dict, api_key: str, stream: bool, provider: str
) -> str:
    """Send payload to an OpenAI-compatible Chat Completions API."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        if stream:
            buffer: List[str] = []
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=600
            ) as r:
                r.raise_for_status()
                LOG.info("Streaming from %s started ...", provider)
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[len("data: ") :]
                    if line.strip() == "[DONE]":
                        print()
                        return "".join(buffer)
                    try:
                        ev = json.loads(line)
                    except Exception:
                        print(line)
                        continue
                    chunk = (
                        ev.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    )
                    if chunk:
                        print(chunk, end="", flush=True)
                        buffer.append(chunk)
            return "".join(buffer)
        else:
            with requests.post(url, headers=headers, json=payload, timeout=600) as r:
                r.raise_for_status()
                data = r.json()
                content = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                print(content)
                return content
    except requests.RequestException as e:
        LOG.error("Request to %s failed: %s", provider, e)
        print(f"[ERROR] Request to {provider} failed: {e}", file=sys.stderr)
        sys.exit(1)

    return ""


def send_openai(base_url: str, payload: dict, api_key: str, stream: bool) -> str:
    """Send payload to OpenAI's Chat Completions API."""
    return _send_openai_style(base_url, payload, api_key, stream, "OpenAI")


def send_huggingface(base_url: str, payload: dict, api_key: str, stream: bool) -> str:
    """Send payload to Hugging Face's Chat Completions API."""
    return _send_openai_style(base_url, payload, api_key, stream, "Hugging Face")


def send_xai(base_url: str, payload: dict, api_key: str, stream: bool) -> str:
    """Send payload to xAI's Grok Chat Completions API."""
    return _send_openai_style(base_url, payload, api_key, stream, "xAI")


def list_openai_models(base_url: str, api_key: str, provider: str = "OpenAI") -> None:
    """Fetch and print available model IDs from an OpenAI-compatible API."""
    url = base_url.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("data", []):
            model_id = item.get("id")
            if model_id:
                print(model_id)
    except requests.RequestException as e:
        LOG.error("Request to %s models endpoint failed: %s", provider, e)
        print(
            f"[ERROR] Request to {provider} models endpoint failed: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def list_ollama_models(base_url: str) -> None:
    """Fetch and print models available on an Ollama server."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("models", []):
            name = item.get("name")
            if name:
                print(name)
    except requests.RequestException as e:
        LOG.error("Request to Ollama tags endpoint failed: %s", e)
        print(f"[ERROR] Request to Ollama tags endpoint failed: {e}", file=sys.stderr)
        sys.exit(1)


def list_gemini_models(base_url: str, api_key: str) -> None:
    """Fetch and print available models from the Gemini API."""
    url = base_url.rstrip("/") + "/v1beta/models"
    params = {"key": api_key}
    try:
        while True:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("models", []):
                model_name = item.get("name")
                if model_name:
                    print(model_name)
            token = data.get("nextPageToken")
            if not token:
                break
            params["pageToken"] = token
    except requests.RequestException as e:
        LOG.error("Request to Gemini models endpoint failed: %s", e)
        print(f"[ERROR] Request to Gemini models endpoint failed: {e}", file=sys.stderr)
        sys.exit(1)


# --------------------------------- CLI -----------------------------------


def process_single(args) -> None:
    images_b64: List[Tuple[str, str]] = []
    text_attachments: List[Tuple[str, str]] = []
    video_notes: List[str] = []
    document_attachments: List[dict] = []

    def record_document_attachment(
        name: str, data: bytes, mime_type: Optional[str]
    ) -> None:
        """Render previews for binary documents and fall back to attachments."""

        mime = mime_type or "application/octet-stream"

        if mime == "application/pdf":
            previews, total_pages = convert_pdf_blob_to_image_previews(data)
            if previews:
                plural = "s" if len(previews) != 1 else ""
                total_display = total_pages or len(previews)
                note = (
                    f"[PDF preview ready: {name} | {len(previews)} image{plural}"
                    f" | {len(data)} bytes | first {len(previews)} of {total_display} pages"
                    " | use --extract-text for local parsing]"
                )
                text_attachments.append((name, note))
                for image_b64 in previews:
                    images_b64.append((image_b64, "image/png"))
                LOG.info(
                    "Rendered %d preview image(s) for PDF '%s'.",
                    len(previews),
                    name,
                )
                return
            placeholder = (
                f"[PDF preview unavailable: {name} | {len(data)} bytes | install"
                " 'pypdfium2'+'Pillow' or use --extract-text]"
            )
        elif (
            mime
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            previews, truncated = convert_docx_blob_to_image_previews(
                data, max_chars=args.max_chars
            )
            if previews:
                plural = "s" if len(previews) != 1 else ""
                suffix = " (truncated)" if truncated else ""
                note = (
                    f"[DOCX preview ready: {name} | {len(previews)} image{plural}"
                    f" | {len(data)} bytes{suffix} | use --extract-text for text mode]"
                )
                text_attachments.append((name, note))
                for image_b64 in previews:
                    images_b64.append((image_b64, "image/png"))
                LOG.info(
                    "Rendered %d preview image(s) for DOCX '%s'.",
                    len(previews),
                    name,
                )
                return
            placeholder = (
                f"[DOCX preview unavailable: {name} | {len(data)} bytes | install"
                " 'python-docx'+'Pillow' or use --extract-text]"
            )
        else:
            placeholder = (
                f"[Document attached: {name} | MIME: {mime} | {len(data)} bytes]"
            )

        document_attachments.append(
            {
                "name": name,
                "data": base64.b64encode(data).decode("ascii"),
                "mime_type": mime,
            }
        )
        text_attachments.append((name, placeholder))
        LOG.info(
            "Stored binary attachment for '%s' (mime=%s, %d bytes)", name, mime, len(data)
        )

    # --- Gather URLs ---
    for url in args.urls:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            header_ct = resp.headers.get("Content-Type", "")
            ctype = args.type or infer_content_type(header_ct)
            name = os.path.basename(urlparse(url).path) or url

            if ctype == "image":
                mime = header_ct.split(";")[0].strip().lower()
                if not mime.startswith("image/"):
                    mime = None
                images_b64.append(
                    (
                        base64.b64encode(resp.content).decode("ascii"),
                        _ensure_image_mime(mime),
                    )
                )
                LOG.info("Fetched image URL: %s (%d bytes)", url, len(resp.content))
                continue

            if ctype == "video":
                suffix = os.path.splitext(name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                    tmp.write(resp.content)
                    tmp.flush()
                    frames = extract_video_frames_b64(
                        tmp.name,
                        max_frames=max(1, args.video_max_frames),
                        target_width=args.video_width if args.video_width > 0 else None,
                    )
                if frames:
                    images_b64.extend((frame, "image/png") for frame in frames)
                    note = (
                        f"Extracted {len(frames)} frames from '{name}' "
                        f"(width ~{args.video_width if args.video_width > 0 else 'original'} px)."
                    )
                    video_notes.append(note)
                else:
                    video_notes.append(
                        f"Video '{name}' could not be processed (maybe run 'pip install opencv-python')."
                    )
                LOG.info("Fetched video URL: %s (%d bytes)", url, len(resp.content))
                continue

            # default: treat as doc/text
            header_main = header_ct.split(";")[0].lower()
            if not header_main:
                _, guess_ext = os.path.splitext(name)
                if guess_ext.lower() == ".pdf":
                    header_main = "application/pdf"
                elif guess_ext.lower() == ".docx":
                    header_main = (
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            if header_main == "application/pdf":
                if args.extract_text:
                    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
                        tmp.write(resp.content)
                        tmp.flush()
                        extracted = try_extract_pdf_text(tmp.name)
                    content = extracted or resp.text
                else:
                    record_document_attachment(name, resp.content, header_main)
                    continue
            elif (
                header_main
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                if args.extract_text:
                    with tempfile.NamedTemporaryFile(suffix=".docx") as tmp:
                        tmp.write(resp.content)
                        tmp.flush()
                        extracted = try_extract_docx_text(tmp.name)
                    content = extracted or resp.text
                else:
                    record_document_attachment(name, resp.content, header_main)
                    continue
            else:
                content = resp.text

            truncated = False
            if len(content) > args.max_chars:
                content = content[: args.max_chars] + "\n\n[... truncated ...]"
                truncated = True
            text_attachments.append((name, content))
            LOG.info(
                "Fetched URL: %s (%d chars%s)",
                url,
                len(content),
                ", truncated" if truncated else "",
            )
        except Exception as e:
            LOG.warning("Failed to fetch URL %s: %s", url, e)

    # --- Gather files ---
    for path in args.files:
        if not os.path.isfile(path):
            LOG.warning("File not found: %s", path)
            continue

        if is_image(path):
            try:
                mime, _ = mimetypes.guess_type(path)
                images_b64.append((to_base64(path), _ensure_image_mime(mime)))
                LOG.info("Added image: %s", path)
            except Exception as e:
                LOG.error("Could not read image (%s): %s", path, e)
            continue

        if is_video(path):
            frames = extract_video_frames_b64(
                path,
                max_frames=max(1, args.video_max_frames),
                target_width=args.video_width if args.video_width > 0 else None,
            )
            if frames:
                images_b64.extend((frame, "image/png") for frame in frames)
                note = (
                    f"Extracted {len(frames)} frames from '{os.path.basename(path)}' "
                    f"(width ~{args.video_width if args.video_width > 0 else 'original'} px)."
                )
                video_notes.append(note)
            else:
                video_notes.append(
                    f"Video '{os.path.basename(path)}' could not be processed "
                    f"(maybe run 'pip install opencv-python')."
                )
            continue

        _, ext = os.path.splitext(path)
        ext = ext.lower()

        if ext in TEXT_LIKE_EXT:
            content, _ = read_text_file(path, max_chars=args.max_chars)
            text_attachments.append((os.path.basename(path), content))
            continue

        if ext == ".pdf":
            if args.extract_text:
                extracted = try_extract_pdf_text(path)
                if extracted:
                    if len(extracted) > args.max_chars:
                        extracted = (
                            extracted[: args.max_chars] + "\n\n[... truncated ...]"
                        )
                    text_attachments.append((os.path.basename(path), extracted))
                    continue
            else:
                try:
                    with open(path, "rb") as fh:
                        blob = fh.read()
                    record_document_attachment(
                        os.path.basename(path), blob, "application/pdf"
                    )
                    continue
                except Exception as e:
                    LOG.warning("Failed to read PDF as binary (%s): %s", path, e)

        if ext == ".docx":
            if args.extract_text:
                extracted = try_extract_docx_text(path)
                if extracted:
                    if len(extracted) > args.max_chars:
                        extracted = (
                            extracted[: args.max_chars] + "\n\n[... truncated ...]"
                        )
                    text_attachments.append((os.path.basename(path), extracted))
                    continue
            else:
                try:
                    with open(path, "rb") as fh:
                        blob = fh.read()
                    record_document_attachment(
                        os.path.basename(path), blob, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                    continue
                except Exception as e:
                    LOG.warning("Failed to read DOCX as binary (%s): %s", path, e)

        # Fallback: try as text (may be binary)
        content, _ = read_text_file(path, max_chars=args.max_chars)
        text_attachments.append((os.path.basename(path), content))

    user_content = build_user_content(args.prompt, text_attachments, video_notes)

    # Optional frame-by-frame mode: send each image separately
    if args.frame_by_frame and images_b64:
        LOG.info(
            "Frame-by-frame mode enabled: sending %d images individually",
            len(images_b64),
        )
        responses: List[str] = []
        for b64, mime in images_b64:
            if args.backend == "gemini":
                payload = {
                    "contents": build_gemini_contents(
                        user_content, [(b64, _ensure_image_mime(mime))]
                    )
                }
                resp = send_gemini(
                    args.host,
                    payload,
                    model=args.model,
                    api_key=args.api_key,
                    stream=args.stream,
                )
            elif args.backend in ("openai", "huggingface", "xai"):
                content_parts = [{"type": "text", "text": user_content}]
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{_ensure_image_mime(mime)};base64,{b64}"
                        },
                    }
                )
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": content_parts}],
                    "stream": True if args.stream else False,
                }
                if args.backend == "openai":
                    resp = send_openai(
                        args.host,
                        payload,
                        api_key=args.api_key,
                        stream=args.stream,
                    )
                elif args.backend == "huggingface":
                    resp = send_huggingface(
                        args.host,
                        payload,
                        api_key=args.api_key,
                        stream=args.stream,
                    )
                else:
                    resp = send_xai(
                        args.host,
                        payload,
                        api_key=args.api_key,
                        stream=args.stream,
                    )
            else:
                payload = {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": user_content,
                            "images": [b64],
                        }
                    ],
                    "stream": True if args.stream else False,  # <<< IMPORTANT
                }
                if document_attachments:
                    payload["messages"][0]["documents"] = document_attachments
                resp = send_with_fallback(
                    args.host,
                    payload,
                    images_present=True,
                    user_content=user_content,
                    stream=args.stream,
                )
            responses.append(resp)
        response = "\n".join(responses)
    else:
        # Payload for /api/chat – IMPORTANT: set stream flag correctly
        if args.backend == "gemini":
            payload = {
                "contents": build_gemini_contents(
                    user_content,
                    [(b64, _ensure_image_mime(mime)) for b64, mime in images_b64],
                )
            }
            response = send_gemini(
                args.host,
                payload,
                model=args.model,
                api_key=args.api_key,
                stream=args.stream,
            )
        elif args.backend in ("openai", "huggingface", "xai"):
            content_parts = [{"type": "text", "text": user_content}]
            for b64, mime in images_b64:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{_ensure_image_mime(mime)};base64,{b64}"
                        },
                    }
                )
            payload = {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": content_parts,
                    }
                ],
                "stream": True if args.stream else False,
            }
            if args.backend == "openai":
                response = send_openai(
                    args.host,
                    payload,
                    api_key=args.api_key,
                    stream=args.stream,
                )
            elif args.backend == "huggingface":
                response = send_huggingface(
                    args.host,
                    payload,
                    api_key=args.api_key,
                    stream=args.stream,
                )
            else:
                response = send_xai(
                    args.host,
                    payload,
                    api_key=args.api_key,
                    stream=args.stream,
                )
        else:
            payload = {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": user_content,
                    }
                ],
                "stream": True if args.stream else False,  # <<< IMPORTANT
            }
            if images_b64:
                payload["messages"][0]["images"] = [b64 for b64, _ in images_b64]
                LOG.info(
                    "Images in payload: %d (including possible video frames)",
                    len(images_b64),
                )
            if document_attachments:
                payload["messages"][0]["documents"] = document_attachments

            # Send (with fallback & tracing)
            response = send_with_fallback(
                args.host,
                payload,
                images_present=bool(images_b64),
                user_content=user_content,
                stream=args.stream,
            )

    if args.output:
        if args.output is True:
            if args.files:
                filename = args.files[0] + ".txt"
            else:
                filename = "response.txt"
        else:
            filename = args.output
        try:
            with open(filename, "w", encoding="utf-8") as fh:
                fh.write(response)
            LOG.info("Saved response to %s", filename)
        except Exception as e:
            LOG.error("Could not write output file %s: %s", filename, e)


def main():
    parser = argparse.ArgumentParser(
        description="Send a prompt plus files (including MP4) to AI backends."
    )
    parser.add_argument(
        "-m",
        "--model",
        default="llama3.2-vision",
        help="Model name (e.g. llama3.2-vision, gpt-4o, etc.)",
    )
    parser.add_argument("-p", "--prompt", help="Prompt/user instruction")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-f",
        "--file",
        action="append",
        dest="files",
        default=[],
        help="File path (repeatable)",
    )
    group.add_argument(
        "--url",
        action="append",
        dest="urls",
        default=[],
        help="Fetch URL and include response text (repeatable)",
    )
    model_list_group = parser.add_mutually_exclusive_group()
    model_list_group.add_argument(
        "--openai-models",
        action="store_true",
        help="List available OpenAI models and exit (no other args allowed)",
    )
    model_list_group.add_argument(
        "--huggingface-models",
        action="store_true",
        help="List available Hugging Face models and exit (no other args allowed)",
    )
    model_list_group.add_argument(
        "--xai-models",
        action="store_true",
        help="List available xAI models and exit (no other args allowed)",
    )
    model_list_group.add_argument(
        "--gemini-models",
        action="store_true",
        help="List available Gemini models and exit (no other args allowed)",
    )
    model_list_group.add_argument(
        "--ollama-models",
        action="store_true",
        help="List available Ollama models and exit (no other args allowed)",
    )
    parser.add_argument(
        "-d", "--directory", help="Process all files in DIRECTORY individually"
    )
    parser.add_argument(
        "--host", default=None, help="API host (default: depends on backend)"
    )
    parser.add_argument(
        "-b",
        "--backend",
        choices=["ollama", "openai", "huggingface", "xai", "gemini"],
        default="ollama",
        help="API backend to use",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=["image", "video", "doc"],
        dest="type",
        help="Override content type for fetched URLs",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=200_000,
        help="Max chars per text file (before truncation)",
    )
    parser.add_argument(
        "--extract-text",
        action="store_true",
        help="Use local text extraction tools for PDF/DOCX",
    )
    parser.add_argument(
        "--stream", action="store_true", help="Stream response as server-sent events"
    )
    parser.add_argument(
        "--video-max-frames", type=int, default=8, help="Max extracted frames per video"
    )
    parser.add_argument(
        "--video-width", type=int, default=640, help="Width to resize frames (0=off)"
    )
    parser.add_argument(
        "--frame-by-frame",
        action="store_true",
        help="Send images/video frames individually and join responses",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const=True,
        default=False,
        metavar="FILE",
        help="Save response to a file. Optional filename; defaults to '<first file>.txt'",
    )

    # Tracing / verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="More output (once = INFO, twice = DEBUG)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Alias for very verbose logging (DEBUG)"
    )

    args = parser.parse_args()
    if args.debug:
        args.verbose = max(args.verbose, 2)
    setup_logging(args.verbose)

    if (
        args.openai_models
        or args.huggingface_models
        or args.xai_models
        or args.gemini_models
        or args.ollama_models
    ):
        if len(sys.argv) > 2:
            parser.error("Model listing flags cannot be combined with other arguments")
        if args.openai_models:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print(
                    "[ERROR] OPENAI_API_KEY environment variable is required to list models.",
                    file=sys.stderr,
                )
                sys.exit(1)
            list_openai_models("https://api.openai.com", api_key, "OpenAI")
        elif args.huggingface_models:
            api_key = os.environ.get("HUGGINGFACE_API_KEY")
            if not api_key:
                print(
                    "[ERROR] HUGGINGFACE_API_KEY environment variable is required to list models.",
                    file=sys.stderr,
                )
                sys.exit(1)
            list_openai_models(
                "https://api-inference.huggingface.co", api_key, "Hugging Face"
            )
        elif args.xai_models:
            api_key = os.environ.get("XAI_API_KEY")
            if not api_key:
                print(
                    "[ERROR] XAI_API_KEY environment variable is required to list models.",
                    file=sys.stderr,
                )
                sys.exit(1)
            list_openai_models("https://api.x.ai", api_key, "xAI")
        elif args.gemini_models:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print(
                    "[ERROR] GEMINI_API_KEY environment variable is required to list models.",
                    file=sys.stderr,
                )
                sys.exit(1)
            list_gemini_models("https://generativelanguage.googleapis.com", api_key)
        else:
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            list_ollama_models(host)
        return

    if not args.prompt:
        parser.error("the following arguments are required: -p/--prompt")

    if not args.host:
        if args.backend == "openai":
            args.host = "https://api.openai.com"
        elif args.backend == "huggingface":
            args.host = "https://api-inference.huggingface.co"
        elif args.backend == "xai":
            args.host = "https://api.x.ai"
        elif args.backend == "gemini":
            args.host = "https://generativelanguage.googleapis.com"
        else:
            args.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    if args.backend == "openai":
        args.api_key = os.environ.get("OPENAI_API_KEY")
        if not args.api_key:
            LOG.error(
                "OPENAI_API_KEY environment variable is required for backend 'openai'."
            )
            print(
                "[ERROR] OPENAI_API_KEY environment variable is required for backend 'openai'.",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.backend == "huggingface":
        args.api_key = os.environ.get("HUGGINGFACE_API_KEY")
        if not args.api_key:
            LOG.error(
                "HUGGINGFACE_API_KEY environment variable is required for backend 'huggingface'."
            )
            print(
                "[ERROR] HUGGINGFACE_API_KEY environment variable is required for backend 'huggingface'.",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.backend == "xai":
        args.api_key = os.environ.get("XAI_API_KEY")
        if not args.api_key:
            LOG.error("XAI_API_KEY environment variable is required for backend 'xai'.")
            print(
                "[ERROR] XAI_API_KEY environment variable is required for backend 'xai'.",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.backend == "gemini":
        args.api_key = os.environ.get("GEMINI_API_KEY")
        if not args.api_key:
            LOG.error(
                "GEMINI_API_KEY environment variable is required for backend 'gemini'."
            )
            print(
                "[ERROR] GEMINI_API_KEY environment variable is required for backend 'gemini'.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        args.api_key = None

    LOG.info(
        "Starting clair | model=%s | host=%s | files=%d | urls=%d | stream=%s | backend=%s",
        args.model,
        args.host,
        len(args.files),
        len(args.urls),
        args.stream,
        args.backend,
    )

    if args.directory:
        if not os.path.isdir(args.directory):
            LOG.error("Directory not found: %s", args.directory)
            return

        file_list: List[str] = []
        for dirpath, _, filenames in os.walk(args.directory):
            for name in sorted(filenames):
                file_list.append(os.path.join(dirpath, name))

        out_dir = None
        if isinstance(args.output, str):
            out_dir = args.output
            os.makedirs(out_dir, exist_ok=True)

        for path in file_list:
            sub_args = argparse.Namespace(**vars(args))
            sub_args.directory = None
            sub_args.files = [path]
            if out_dir:
                rel_path = os.path.relpath(path, args.directory)
                out_path = os.path.join(out_dir, rel_path + ".txt")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                sub_args.output = out_path
            elif not args.output:
                sub_args.output = True
            process_single(sub_args)
        return

    process_single(args)


if __name__ == "__main__":
    main()
