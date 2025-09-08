# ai-cli
AI Agents CLI Tools

## Usage

```bash
python ollama_send.py -p "Hello" [-f path/to/file | --url https://example.com] [-o [output.txt]]
```

Use `--url` to download text from a web resource and include it as if it
were a file attachment. `--url` and `-f/--file` are mutually exclusive.

Use `-o`/`--output` to save the model response to a file. When no
filename is supplied, the first attached file name with `.txt` appended is
used; if no files are attached, `response.txt` is created.
