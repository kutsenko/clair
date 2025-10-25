# KI CLI-Tools

## Installation

### Voraussetzungen

- Python 3.11 oder höher
- [virtualenv](https://virtualenv.pypa.io/) oder das integrierte Modul `venv`

### Virtuelle Umgebung erstellen und aktivieren

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Unter Windows aktivierst du die Umgebung mit `.\\.venv\\Scripts\\activate`.

### Abhängigkeiten installieren

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Optionale Abhängigkeiten

Installiere diese Extras, wenn du eine erweiterte Dateiverarbeitung benötigst:

- `PyPDF2` – wird mit `--extract-text` verwendet, um PDF-Inhalte zu extrahieren und vor dem Senden an das ausgewählte Backend in die Prompts einzufügen.
- `python-docx` – wird mit `--extract-text` verwendet, um DOCX-Dateien zu parsen und den Absatztext inline einzubinden.
- `opencv-python` – extrahiert repräsentative PNG-Frames aus Videodateien bei Verwendung der Verarbeitung `--frame-by-frame`.
- `pypdfium2` + `Pillow` – rendert PDFs in PNG-Vorschauen, die an visionsfähige Backends hochgeladen werden, wenn `--extract-text` **nicht** angegeben wird.
- `Pillow` (alleine) – rendert DOCX-Schnappschüsse, damit Textverarbeitungsdateien als Bilder betrachtet werden können, wenn auf Textextraktion verzichtet wird.

Installiere die Vorschau-Toolchain (für PDFs und DOCX-Dateien) mit:

```bash
pip install pillow pypdfium2 python-docx
```

## Verwendung

Nachdem du die virtuelle Umgebung aktiviert hast, führe die CLI wie folgt aus:

```bash
python3 clair.py -p "Hello" [-f path/to/file | -d path/to/dir | --url https://example.com] [-o [output.txt]] [-b backend]
```

Verwende `--url`, um Inhalte von einer Webressource herunterzuladen und sie wie einen Dateianhang einzubinden. Der Inhaltstyp wird anhand der HTTP-Antwort automatisch erkannt; überschreibe ihn bei Bedarf mit `-t/--type` (`image`, `video`, `doc`). `--url` und `-f/--file` schließen sich gegenseitig aus.

Wähle das Backend mit `-b/--backend` (`ollama`, `openai`, `huggingface`, `xai`, `gemini` oder `claude`).
Bei Verwendung des OpenAI-Backends setzt du den API-Schlüssel über die Umgebungsvariable `OPENAI_API_KEY`. Für das Hugging-Face-Backend wird der Schlüssel aus der Umgebungsvariable `HUGGINGFACE_API_KEY` gelesen. Das xAI-Backend (Grok) nutzt die Umgebungsvariable `XAI_API_KEY`. Das Google-Gemini-Backend erwartet eine Umgebungsvariable `GEMINI_API_KEY` und verwendet standardmäßig `https://generativelanguage.googleapis.com`. Das Anthropic-Claude-Backend liest den Schlüssel aus der Umgebungsvariable `ANTHROPIC_API_KEY` und verwendet standardmäßig `https://api.anthropic.com`.
Nutze die backend-spezifischen Modellauflistungs-Flags (`--openai-models`, `--huggingface-models`, `--xai-models`, `--gemini-models`, `--claude-models`, `--ollama-models`), um die Modelle des jeweiligen Dienstes anzuzeigen. Diese Flags können nicht mit anderen Optionen kombiniert werden.

Verwende `-d`/`--directory`, um **rekursiv** jede Datei in einem Ordner zu senden. Jede Antwort wird unter `<filename>.txt` geschrieben oder, wenn `-o` auf ein Verzeichnis verweist, in diesem Verzeichnis unter Beibehaltung des relativen Pfads gespeichert.

Für Modelle, die nur einzelne Bilder akzeptieren, aktiviere `--frame-by-frame`, um extrahierte Videoframes einzeln zu senden und die Antworten zu verketten.

Nutze `-o`/`--output`, um die Modellantwort in einer Datei zu speichern. Wenn kein Dateiname angegeben wird, wird der Name der ersten angehängten Datei mit angehängtem `.txt` verwendet; sind keine Dateien angehängt, wird `response.txt` erstellt.

Standardmäßig rendert die CLI PDF- und DOCX-Dateien in PNG-Vorschau-Bilder und sendet diese an visionsfähige Backends (z. B. multimodale Ollama-Modelle). Wenn die Vorschau-Toolchain nicht verfügbar ist, wird das Binärdokument stattdessen mit einem Hinweis angehängt, wie Vorschauen aktiviert werden können. Verwende `--extract-text`, wenn du lokale Extraktionstools (PDF via PyPDF2, DOCX via python-docx) bevorzugst, bevor der Inhalt in den Prompt eingebettet wird.
