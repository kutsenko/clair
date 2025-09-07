#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sende Prompt + beliebige Dateien (inkl. MP4) an Ollama.
- /api/chat wird bevorzugt; bei 404 Fallback auf /api/generate.
- Bilder -> base64 im 'images'-Feld (für Vision-Modelle).
- Textdateien -> als Anhang im Prompt (Codeblock).
- PDF/DOCX -> optionale Textextraktion (PyPDF2 / python-docx).
- MP4 -> extrahiert N Frames (PNG, base64) und hängt sie an 'images'.

Kompatibel mit Ollama 0.11.10 (REST).
Erweitertes Tracing via --verbose/--debug: Endpoints, Dauer, Status/Body, Fehlerpfad.
"""

import argparse
import base64
import io
import json
import logging
import mimetypes
import os
import sys
import time
from typing import List, Tuple, Optional

import requests


# --------------------------- Logging / Tracing ---------------------------

LOG = logging.getLogger("ollama_send")

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


# ---------------------- Optionale Parser für PDF/DOCX --------------------

def try_extract_pdf_text(path: str) -> str:
    try:
        import PyPDF2  # type: ignore
    except Exception:
        LOG.debug("PyPDF2 nicht installiert – PDF wird nicht extrahiert.")
        return ""
    try:
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        out = "\n".join(text).strip()
        LOG.info("PDF-Text extrahiert: %s (%d Zeichen)", os.path.basename(path), len(out))
        return out
    except Exception as e:
        LOG.warning("PDF-Extraktion fehlgeschlagen (%s): %s", path, e)
        return ""


def try_extract_docx_text(path: str) -> str:
    try:
        import docx  # python-docx
    except Exception:
        LOG.debug("python-docx nicht installiert – DOCX wird nicht extrahiert.")
        return ""
    try:
        with open(path, "rb") as f:
            blob = f.read()  # sicher schließen
        doc = docx.Document(io.BytesIO(blob))  # kein offener File-Handle
        out = "\n".join(p.text for p in doc.paragraphs).strip()
        LOG.info("DOCX-Text extrahiert: %s (%d Zeichen)", os.path.basename(path), len(out))
        return out
    except Exception as e:
        LOG.warning("DOCX-Extraktion fehlgeschlagen (%s): %s", path, e)
        return ""


# -------------------- Video (MP4) -> Frames (base64 PNG) -----------------

def extract_video_frames_b64(
    path: str,
    max_frames: int = 8,
    target_width: Optional[int] = 640,
) -> List[str]:
    """
    Extrahiert bis zu max_frames gleichmäßig verteilte Frames und liefert sie
    als base64-kodierte PNGs (Strings) zurück. Benötigt opencv-python.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOG.error("Für Video-Frames wird opencv-python benötigt: %s", e)
        return []

    LOG.info("Video erkannt, extrahiere Frames: %s", path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        LOG.error("Konnte Video nicht öffnen: %s", path)
        return []

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            LOG.error("Keine gültigen Frames in: %s", path)
            return []

        num = max(1, min(max_frames, frame_count))
        indices = [int(i * (frame_count - 1) / (num - 1)) for i in range(num)] if num > 1 else [0]

        images_b64: List[str] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                LOG.debug("Frame %d konnte nicht gelesen werden.", idx)
                continue

            if target_width and frame.shape[1] > target_width:
                import cv2  # ensure in scope
                h, w = frame.shape[:2]
                scale = target_width / float(w)
                new_w = target_width
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            ok, buf = cv2.imencode(".png", frame)
            if not ok:
                LOG.debug("Frame %d konnte nicht als PNG encodiert werden.", idx)
                continue
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            images_b64.append(b64)

        LOG.info("Aus '%s' wurden %d Frames extrahiert (Zielbreite: %s).",
                 os.path.basename(path), len(images_b64),
                 f"{target_width}px" if target_width else "Original")
        return images_b64
    finally:
        cap.release()
        del cap  # optional


# ------------------------------ Utilities --------------------------------

TEXT_LIKE_EXT = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml", ".xml",
    ".html", ".htm", ".css", ".js", ".ts", ".py", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".ini", ".conf", ".log"
}
VIDEO_EXT = {".mp4"}  # ggf. erweitern: .mov, .mkv ...
IMAGE_MIME_PREFIXES = ("image/",)

def is_image(path: str) -> bool:
    mime, _ = mimetypes.guess_type(path)
    return bool(mime and mime.startswith(IMAGE_MIME_PREFIXES))

def is_video(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in VIDEO_EXT

def read_text_file(path: str, encoding: str = "utf-8", max_chars: int = 200_000) -> Tuple[str, bool]:
    try:
        with open(path, "r", encoding=encoding, errors="replace") as f:
            data = f.read()
        truncated = False
        if len(data) > max_chars:
            data = data[:max_chars] + "\n\n[... abgeschnitten …]"
            truncated = True
        LOG.info("Textdatei gelesen: %s (%d Zeichen%s)",
                 os.path.basename(path), len(data), ", gekürzt" if truncated else "")
        return data, truncated
    except Exception as e:
        LOG.warning("Als Text lesen fehlgeschlagen (%s): %s", path, e)
        return f"[FEHLER beim Lesen als Text: {e}]", False

def to_base64(path: str) -> str:
    with open(path, "rb") as f:
        blob = f.read()
    LOG.debug("Datei base64-kodiert: %s (%d Bytes)", os.path.basename(path), len(blob))
    return base64.b64encode(blob).decode("ascii")

def build_user_content(prompt: str, text_attachments: List[Tuple[str, str]], video_notes: List[str]) -> str:
    parts = [prompt]
    if text_attachments:
        parts.append("\n\n---\n### Anhänge (Text)")
        for fname, txt in text_attachments:
            parts.append(f"\n**{fname}**:\n```\n{txt}\n```")
    if video_notes:
        parts.append("\n\n---\n### Video-Hinweise")
        for note in video_notes:
            parts.append(f"- {note}")
    return "\n".join(parts)


# ------------------------- HTTP + Endpoint-Fallback -----------------------

def _post_json(url: str, payload: dict, *, stream: bool = False, timeout: int = 600) -> requests.Response:
    LOG.debug("POST %s | stream=%s | timeout=%s", url, stream, timeout)
    LOG.debug("Payload keys: %s", list(payload.keys()))
    start = time.monotonic()
    resp = requests.post(url, json=payload, stream=stream, timeout=timeout)
    dur = time.monotonic() - start
    LOG.info("Response %s in %.3fs von %s", resp.status_code, dur, url)
    if resp.status_code >= 400:
        # Body (max 2k) zur Diagnose
        try:
            body = resp.text
            if len(body) > 2048:
                body = body[:2048] + "\n...[gekürzt]..."
        except Exception:
            body = "<Body nicht lesbar>"
        LOG.warning("Fehler-Body (%s):\n---\n%s\n---", url, body)
    return resp

def _read_nonstream_json_fallback(resp: requests.Response) -> dict:
    """
    Robust gegen falsch gesetzer Streaming-Server: Versucht zuerst resp.json(),
    fällt dann auf NDJSON-/Zeilen-Parsing zurück (nimmt den letzten validen JSON-Block).
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

def send_with_fallback(base_url: str, payload_chat: dict, images_present: bool, user_content: str, stream: bool) -> None:
    """
    Versucht zuerst /api/chat. Bei 404:
      - wenn "model not found" -> klare Meldung
      - sonst Fallback auf /api/generate (Prompt + images)
    Gibt die Modellantwort auf stdout aus (Streaming oder finaler Block).
    """
    base_url = base_url.rstrip("/")
    chat_url = base_url + "/api/chat"
    gen_url  = base_url + "/api/generate"

    # 1) /api/chat
    try:
        if stream:
            with _post_json(chat_url, payload_chat, stream=True) as r:
                if r.status_code == 404:
                    body = r.text.lower()
                    if "model" in body and "not found" in body:
                        model = payload_chat.get("model", "<unbekannt>")
                        LOG.error("Modell '%s' nicht vorhanden. Bitte ausführen: ollama pull %s", model, model)
                        print(f"[FEHLER] Modell '{model}' nicht vorhanden. Bitte 'ollama pull {model}' ausführen.", file=sys.stderr)
                        return
                    LOG.info("Endpoint /api/chat nicht verfügbar – Fallback auf /api/generate.")
                else:
                    r.raise_for_status()
                    LOG.info("Streaming von /api/chat gestartet …")
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except Exception:
                            print(line)
                            continue
                        if "message" in ev and "content" in ev["message"]:
                            print(ev["message"]["content"], end="", flush=True)
                        if ev.get("done"):
                            print()
                            return
        else:
            with _post_json(chat_url, payload_chat, timeout=600) as r:
                if r.status_code == 404:
                    body = r.text.lower()
                    if "model" in body and "not found" in body:
                        model = payload_chat.get("model", "<unbekannt>")
                        LOG.error("Modell '%s' nicht vorhanden. Bitte ausführen: ollama pull %s", model, model)
                        print(f"[FEHLER] Modell '{model}' nicht vorhanden. Bitte 'ollama pull {model}' ausführen.", file=sys.stderr)
                        return
                    LOG.info("Endpoint /api/chat nicht verfügbar – Fallback auf /api/generate.")
                else:
                    r.raise_for_status()
                    data = _read_nonstream_json_fallback(r)
                    content = data.get("message", {}).get("content", "")
                    print(content)
                    return
    except requests.RequestException as e:
        LOG.warning("/api/chat fehlgeschlagen (%s). Fallback wird versucht.", e)

    # 2) /api/generate (Fallback)
    payload_generate = {
        "model": payload_chat.get("model"),
        "prompt": user_content,
        "stream": True if stream else False,  # <<< WICHTIG
    }
    if images_present:
        imgs = payload_chat["messages"][0].get("images", [])
        if imgs:
            payload_generate["images"] = imgs

    try:
        if stream:
            with _post_json(gen_url, payload_generate, stream=True) as r:
                r.raise_for_status()
                LOG.info("Streaming von /api/generate gestartet …")
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        print(line)
                        continue
                    if "response" in ev:
                        print(ev["response"], end="", flush=True)
                    if ev.get("done"):
                        print()
                        return
        else:
            with _post_json(gen_url, payload_generate, timeout=600) as r:
                if r.status_code >= 400:
                    r.raise_for_status()
                data = _read_nonstream_json_fallback(r)
                content = data.get("response", "")
                print(content)
                return
    except requests.RequestException as e:
        LOG.error("Fallback /api/generate fehlgeschlagen: %s", e)
        print(f"[FEHLER] Anfrage an Ollama fehlgeschlagen (Fallback /api/generate): {e}", file=sys.stderr)
        sys.exit(1)


# --------------------------------- CLI -----------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prompt + Dateien (inkl. MP4) an Ollama senden.")
    parser.add_argument("-m", "--model", default="llama3.2-vision",
                        help="Ollama-Modell (z. B. llama3.2-vision, llama3.1, qwen2.5, usw.)")
    parser.add_argument("-p", "--prompt", required=True, help="Prompt/Nutzeranweisung")
    parser.add_argument("-f", "--file", action="append", dest="files", default=[], help="Dateipfad (mehrfach nutzbar)")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                        help="Ollama-Host (Standard: http://localhost:11434 oder Env OLLAMA_HOST)")
    parser.add_argument("--max-chars", type=int, default=200_000, help="Max Zeichen pro Textdatei (vor Truncation)")
    parser.add_argument("--no-extract", action="store_true", help="Keine Textextraktion für PDF/DOCX versuchen")
    parser.add_argument("--stream", action="store_true", help="Antwort als Server-Sent Events streamen")
    parser.add_argument("--video-max-frames", type=int, default=8, help="Max. extrahierte Frames pro Video")
    parser.add_argument("--video-width", type=int, default=640, help="Breite zum Resize der Frames (0=aus)")

    # Tracing / Verbosity
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Mehr Ausgaben (einmal = INFO, zweimal = DEBUG)")
    parser.add_argument("--debug", action="store_true",
                        help="Alias für sehr ausführliches Logging (DEBUG)")

    args = parser.parse_args()
    if args.debug:
        args.verbose = max(args.verbose, 2)
    setup_logging(args.verbose)

    LOG.info("Starte ollama_send | Modell=%s | Host=%s | Dateien=%d | Stream=%s",
             args.model, args.host, len(args.files), args.stream)

    images_b64: List[str] = []
    text_attachments: List[Tuple[str, str]] = []
    video_notes: List[str] = []

    # --- Dateien einsammeln ---
    for path in args.files:
        if not os.path.isfile(path):
            LOG.warning("Datei nicht gefunden: %s", path)
            continue

        if is_image(path):
            try:
                images_b64.append(to_base64(path))
                LOG.info("Bild hinzugefügt: %s", path)
            except Exception as e:
                LOG.error("Konnte Bild nicht einlesen (%s): %s", path, e)
            continue

        if is_video(path):
            frames = extract_video_frames_b64(
                path,
                max_frames=max(1, args.video_max_frames),
                target_width=args.video_width if args.video_width > 0 else None,
            )
            if frames:
                images_b64.extend(frames)
                note = (f"Aus '{os.path.basename(path)}' wurden {len(frames)} Frames extrahiert "
                        f"(Breite ~{args.video_width if args.video_width > 0 else 'Original'} px).")
                video_notes.append(note)
            else:
                video_notes.append(
                    f"Video '{os.path.basename(path)}' konnte nicht verarbeitet werden "
                    f"(ggf. 'pip install opencv-python')."
                )
            continue

        _, ext = os.path.splitext(path)
        ext = ext.lower()

        if ext in TEXT_LIKE_EXT:
            content, _ = read_text_file(path, max_chars=args.max_chars)
            text_attachments.append((os.path.basename(path), content))
            continue

        if not args.no_extract and ext == ".pdf":
            extracted = try_extract_pdf_text(path)
            if extracted:
                if len(extracted) > args.max_chars:
                    extracted = extracted[:args.max_chars] + "\n\n[... abgeschnitten …]"
                text_attachments.append((os.path.basename(path), extracted))
                continue

        if not args.no_extract and ext == ".docx":
            extracted = try_extract_docx_text(path)
            if extracted:
                if len(extracted) > args.max_chars:
                    extracted = extracted[:args.max_chars] + "\n\n[... abgeschnitten …]"
                text_attachments.append((os.path.basename(path), extracted))
                continue

        # Fallback: als Text versuchen (kann binär sein)
        content, _ = read_text_file(path, max_chars=args.max_chars)
        text_attachments.append((os.path.basename(path), content))

    user_content = build_user_content(args.prompt, text_attachments, video_notes)

    # Payload für /api/chat – WICHTIG: stream-Flag korrekt setzen
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": user_content,
            }
        ],
        "stream": True if args.stream else False,   # <<< WICHTIG
    }
    if images_b64:
        payload["messages"][0]["images"] = images_b64
        LOG.info("Images im Payload: %d (inkl. evtl. Video-Frames)", len(images_b64))

    # Senden (mit Fallback & Tracing)
    send_with_fallback(args.host, payload, images_present=bool(images_b64),
                       user_content=user_content, stream=args.stream)

if __name__ == "__main__":
    main()
