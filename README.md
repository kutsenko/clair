# CLAIR (AI CLI Tools)

German version: [README.de.md](README.de.md)

## Installation

### Prerequisites

- Python 3.11 or later
- [virtualenv](https://virtualenv.pypa.io/) or the built-in `venv` module

### Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with `.\.venv\Scripts\activate`.

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional dependencies

Install these extras when you need richer file handling:

- `PyPDF2` – used with `--extract-text` to pull PDF contents into prompts before sending them to the selected backend.
- `python-docx` – used with `--extract-text` to parse DOCX files and include paragraph text inline.
- `opencv-python` – extracts representative PNG frames from video files when using `--frame-by-frame` processing.
- `pypdfium2` + `Pillow` – render PDFs into PNG previews that are uploaded to vision-capable backends when `--extract-text` is **not** supplied.
- `Pillow` (alone) – renders DOCX snapshots so word-processing files can be viewed as images when skipping text extraction.

Install the preview toolchain (for PDFs and DOCX files) with:

```bash
pip install pillow pypdfium2 python-docx
```

## Usage

After activating the virtual environment, run the CLI as shown below:

```bash
python3 clair.py -p "Hello" [-f path/to/file | -d path/to/dir | --url https://example.com] [-o [output.txt]] [-b backend]
```

Use `--url` to download content from a web resource and include it as if it
were a file attachment. The content type is auto-detected from the HTTP
response; override with `-t/--type` (`image`, `video`, `doc`).
`--url` and `-f/--file` are mutually exclusive.

Select the backend with `-b/--backend` (`ollama`, `openai`, `huggingface`,
`xai`, `gemini`, or `claude`).
When using the OpenAI backend, set the API key via the `OPENAI_API_KEY`
environment variable. For the Hugging Face backend, the key is read from the
`HUGGINGFACE_API_KEY` environment variable. The xAI (Grok) backend reads the key
from the `XAI_API_KEY` environment variable. The Google Gemini backend expects a
`GEMINI_API_KEY` environment variable and targets
`https://generativelanguage.googleapis.com` by default. The Anthropic Claude
backend reads the key from the `ANTHROPIC_API_KEY` environment variable and targets
`https://api.anthropic.com` by default.
Use the backend-specific model listing flags (`--openai-models`,
`--huggingface-models`, `--xai-models`, `--gemini-models`, `--claude-models`,
`--ollama-models`) to print the models exposed by the respective service. These
flags cannot be combined with other options.

Use `-d`/`--directory` to send every file in a folder **recursively**.
Each response is written to `<filename>.txt` or, when `-o` points to a
directory, saved under that directory while preserving the relative path.

For models that only accept single images, enable `--frame-by-frame` to
send extracted video frames individually and concatenate the responses.

Use `-o`/`--output` to save the model response to a file. When no
filename is supplied, the first attached file name with `.txt` appended is
used; if no files are attached, `response.txt` is created.

By default the CLI renders PDF and DOCX files into PNG preview images and sends
those to vision-capable backends (such as Ollama multimodal models). When the
preview toolchain is unavailable, the binary document is attached instead with a
note explaining how to enable previews. Use `--extract-text` when you prefer to
run local extraction tools (PDF via PyPDF2, DOCX via python-docx) before
embedding the contents into the prompt.
