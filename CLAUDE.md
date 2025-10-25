# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CLAIR (AI CLI Tools)** is a unified command-line gateway for interacting with multiple AI/LLM backends (Ollama, OpenAI, Hugging Face, xAI/Grok, Google Gemini). It enables users to send prompts with attachments (text files, images, videos, PDFs, DOCX documents) to different AI models with intelligent content processing and format handling.

The entire application is contained in a single **1594-line `clair.py` file** organized into logical sections for maintainability.

## Common Commands

### Installation and Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install optional dependencies for document/video processing
pip install pillow pypdfium2 python-docx opencv-python PyPDF2
```

### Running the Application

```bash
# Basic usage: send prompt to default backend (Ollama)
python3 clair.py -p "Hello"

# With file attachment and specific backend
python3 clair.py -p "Analyze this" -f path/to/file -b openai

# Process entire directory recursively
python3 clair.py -p "Review all files" -d path/to/directory -o output_dir

# Fetch and process content from URL
python3 clair.py -p "Summarize" --url https://example.com -b gemini

# List available models for a backend
python3 clair.py --openai-models
python3 clair.py --ollama-models
```

### Testing

```bash
# Run the smoke test (mandatory before committing)
python3 clair.py -p "Hello"

# Run all automated tests
pytest

# Run specific test file
pytest tests/unit/test_backend_selection.py

# Run with verbose output
pytest -v

# Run only integration tests
pytest tests/integration/
```

### Code Quality

Format with `black` and lint with `ruff` when possible:

```bash
black clair.py
ruff check clair.py
```

## Architecture Overview

### Core Processing Pipeline

The main execution flow follows this pattern:

```
main() [CLI Entry Point]
  ↓
parse_args() [Parse command-line arguments]
  ↓
process_single(args) [Process single file/prompt]
  ├─ Fetch URLs (if --url provided)
  ├─ Detect content type (text, image, video, PDF, DOCX)
  ├─ Extract content/render previews/extract frames
  ├─ Encode to base64 if needed
  ├─ Build backend-specific payload
  └─ Send request via selected backend
     └─ Output response (stdout or file)
```

For directory processing, `main()` recursively walks the directory and spawns `process_single()` for each file while preserving the relative directory structure.

### Content Processing Pipeline

The application intelligently handles different file types:

1. **Text Files** (`read_text_file()`):
   - Reads and truncates to configurable length
   - Preserves encoding

2. **Images** (JPEG, PNG, WebP, GIF):
   - Detected via extension and MIME type
   - Base64-encoded for backend transmission

3. **Videos** (MP4):
   - Frame extraction via `extract_video_frames_b64()` (requires `opencv-python`)
   - For models that only accept images: `--frame-by-frame` mode sends frames individually

4. **PDFs**:
   - **Default**: Render to PNG previews via `convert_pdf_blob_to_image_previews()` (requires `pypdfium2` + `Pillow`)
   - **With `--extract-text`**: Extract text via `try_extract_pdf_text()` (requires `PyPDF2`)
   - **Fallback**: Attach raw binary with explanation

5. **DOCX Files**:
   - **Default**: Render snapshots to PNG via `convert_docx_blob_to_image_previews()` (requires `Pillow`)
   - **With `--extract-text`**: Extract paragraph text via `try_extract_docx_text()` (requires `python-docx`)
   - **Fallback**: Attach raw binary with explanation

### Backend Implementations

Each backend has consistent `send_<backend>()` and `list_<backend>_models()` functions:

1. **Ollama** (default, local):
   - Endpoint: `http://localhost:11434` (configurable via `OLLAMA_HOST`)
   - `send_with_fallback()`: Tries `/api/chat` then falls back to `/api/generate`
   - Supports streaming via Server-Sent Events (SSE)
   - No API key required

2. **OpenAI**:
   - API Key: `OPENAI_API_KEY` environment variable
   - Endpoint: `https://api.openai.com/v1/chat/completions`
   - Vision-capable models automatically selected

3. **Hugging Face**:
   - API Key: `HUGGINGFACE_API_KEY` environment variable
   - Inference API endpoint

4. **xAI (Grok)**:
   - API Key: `XAI_API_KEY` environment variable
   - OpenAI-compatible API format

5. **Google Gemini**:
   - API Key: `GEMINI_API_KEY` environment variable
   - Custom payload format handled by `build_gemini_contents()`
   - Supports pagination for model listing

All backends (except Ollama) use `_send_openai_style()` for common request handling.

### Optional Dependencies

The application gracefully handles missing optional modules via `_optional_import()`:

- **PyPDF2**: PDF text extraction (`--extract-text` mode)
- **python-docx**: DOCX text extraction (`--extract-text` mode)
- **Pillow**: Document/text-to-image rendering
- **pypdfium2**: PDF-to-image rendering (paired with Pillow)
- **opencv-python**: Video frame extraction

Missing dependencies trigger informative warnings; features degrade gracefully.

### Error Handling Strategy

1. **Missing API Keys**: Explicit error messages guide users to set environment variables
2. **Failed Requests**: HTTP errors logged with endpoint and status code
3. **Optional Dependencies**: Warnings logged; fallback to raw binary attachment
4. **Content Type Mismatches**: User override via `-t/--type` flag (image, video, doc)

### Key Design Patterns

1. **Graceful Degradation**: Missing optional dependencies don't break core functionality
2. **Backend Abstraction**: Consistent interface across all AI backends
3. **Content Type Detection**: MIME type inference from file extensions and HTTP headers
4. **Fallback Strategy**: Ollama tries `/api/chat`, falls back to `/api/generate` for broader compatibility
5. **Streaming Support**: Server-Sent Events for real-time response output
6. **Configuration via Environment**: All API keys from environment variables

## File Structure

```
clair/
├── clair.py                        # Main application (1594 lines)
├── requirements.txt                # Core dependencies
├── README.md                       # User documentation (English)
├── README.de.md                    # User documentation (German)
├── AGENTS.md                       # Developer guidelines
├── CLAUDE.md                       # This file
├── LICENSE                         # LGPL v3
├── data/
│   └── sample.pdf                  # Test data
├── tests/
│   ├── conftest.py                 # Shared test utilities and mocks
│   ├── unit/                       # Unit tests
│   │   ├── test_backend_selection.py
│   │   ├── test_optional_imports.py
│   │   ├── test_text_extraction_flag.py
│   │   ├── test_frame_by_frame.py
│   │   ├── test_utils.py
│   │   └── test_video_logging.py
│   └── integration/                # Integration tests
│       ├── test_directory_option.py
│       ├── test_output_option.py
│       ├── test_send_with_fallback.py
│       └── test_url_option.py
└── .github/workflows/
    └── ci.yml                      # GitHub Actions CI
```

## Internal Function Organization

The `clair.py` file is organized into these logical sections:

1. **Logging and Tracing**: `setup_logging()`
2. **Optional Dependencies**: `_optional_import()`
3. **CLI Argument Parsing**: `parse_args()`
4. **Content Type Detection**: `is_image()`, `is_video()`, `infer_content_type()`
5. **File Reading**: `read_text_file()`, `to_base64()`
6. **Text Extraction**: `try_extract_pdf_text()`, `try_extract_docx_text()`
7. **Video Processing**: `extract_video_frames_b64()`
8. **Document Preview Rendering**: `render_text_block_to_image_b64()`, `convert_pdf_blob_to_image_previews()`, `convert_docx_blob_to_image_previews()`
9. **Content Building**: `build_user_content()`, `build_gemini_contents()`
10. **HTTP Communication**: `_post_json()`, `_read_nonstream_json_fallback()`
11. **Backend Implementations**: `send_ollama()`, `send_with_fallback()`, `send_openai()`, `send_huggingface()`, `send_xai()`, `send_gemini()`
12. **Model Listing**: `list_ollama_models()`, `list_openai_models()`, `list_huggingface_models()`, `list_xai_models()`, `list_gemini_models()`
13. **Main Entry Point**: `main()`, `process_single()`

## Development Workflow

1. **Before Making Changes**:
   - Ensure you understand which content type or backend your change affects
   - Review the relevant test files in `tests/`

2. **When Adding a New Backend**:
   - Implement `send_<backend>()` function following the pattern of existing backends
   - Add `list_<backend>_models()` for model discovery
   - Add backend option to argument parser
   - Create corresponding tests in `tests/unit/test_backend_selection.py` and optionally in `tests/integration/`
   - Update `README.md` with the new backend and API key environment variable
   - Update `AGENTS.md` if applicable

3. **When Modifying Content Processing**:
   - Changes to text/image/video handling should be in their respective functions
   - Test with sample files from `data/` directory
   - Ensure graceful degradation for optional dependencies
   - Update both unit and integration tests

4. **Code Style**:
   - Follow PEP 8 conventions
   - Use `black` for formatting and `ruff` for linting
   - Write clear, focused methods with single responsibility
   - Document with English comments and docstrings

5. **Testing Requirements**:
   - Run smoke test: `python3 clair.py -p "Hello"`
   - Run full test suite: `pytest`
   - Ensure tests pass before committing
   - Add tests for new backends, content types, or features

## Environment Variables

**Backend API Keys:**
- `OPENAI_API_KEY` – OpenAI
- `HUGGINGFACE_API_KEY` – Hugging Face
- `XAI_API_KEY` – xAI/Grok
- `GEMINI_API_KEY` – Google Gemini

**Ollama Configuration:**
- `OLLAMA_HOST` – Ollama endpoint (default: `http://localhost:11434`)

## Testing Infrastructure

**Test Utilities** (`conftest.py`):
- Mock HTTP responses for backends
- Dummy request objects for testing
- Reusable test fixtures

**Unit Tests** cover:
- Backend initialization and API key validation
- Optional dependency handling
- PDF/DOCX text extraction modes
- Video frame processing
- Utility functions

**Integration Tests** cover:
- Directory recursion and output preservation
- File output handling
- URL fetching and content type detection
- Ollama endpoint fallback behavior

## Known Limitations and Edge Cases

1. **Large Files**: Text files are truncated at configurable lengths to avoid overwhelming model context windows
2. **Optional Dependencies**: PDF previews require both `pypdfium2` AND `Pillow`; fallback to text extraction or raw binary
3. **Video Frames**: Frame extraction quality depends on `opencv-python` version and codec support
4. **Streaming**: Currently only implemented for Ollama; other backends may buffer entire responses
5. **URL Content Type**: Auto-detection from HTTP headers; override with `-t/--type` if needed
6. **Directory Mode**: Output filenames derived from input filenames; `.txt` extension added

## Contributing

Before committing:
1. Run the smoke test
2. Run the full test suite
3. Format code with `black` and lint with `ruff`
4. Update documentation if behavior changes
5. Use concise, imperative commit messages
6. In PRs, describe the change and list tests executed
