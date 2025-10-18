# AGENTS.md

## Scope
These guidelines apply to the entire repository.

## Project Overview
This repository implements a command-line interface (CLI) that can interact with multiple AI backends. Consult [`README.md`](README.md) for a detailed description of the workflow, including environment setup and available backends.

## Development Guidelines
- Use Python 3.11 or later.
- Follow PEP 8 style conventions and the official best practices published on [python.org](https://www.python.org/dev/peps/). Format code with `black` and lint with `ruff` when possible.
- Organize modules by responsibility (e.g., CLI parsing, backend abstractions, utilities).
- Name methods concisely in line with Clean Code principles and keep each method focused on a single semantic level.
- Do not wrap import statements in `try`/`except` blocks.
- Write all documentation, comments, identifiers, and other artifacts in English.

## Testing
- Run `python clair.py -p "Hello"` as a smoke test after changes that affect the CLI.
- Execute the automated test suite with `pytest` before committing; this step is mandatory for every code change.
- Ensure unit and integration tests cover as many realistic use cases as possible to maintain strong coverage.
- When adding a new backend, add corresponding tests under `tests/` and document new CLI options.

## Documentation and PRs
- Update `README.md` or other documentation when behavior changes.
- Use concise, imperative commit messages.
- Summaries in pull requests should describe the change and list the tests that were executed.
