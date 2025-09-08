# ai-cli
AI Agents CLI Tools

## Usage

```bash
python ollama_send.py -p "Hello" [-f path/to/file | -d path/to/dir | --url https://example.com] [-o [output.txt]]
```

Use `--url` to download content from a web resource and include it as if it
were a file attachment. The content type is auto-detected from the HTTP
response; override with `-t/--type` (`image`, `video`, `doc`).
`--url` and `-f/--file` are mutually exclusive.

Use `-d`/`--directory` to send every file in a folder **recursively**.
Each response is written to `<filename>.txt` or, when `-o` points to a
directory, saved under that directory while preserving the relative path.

Use `-o`/`--output` to save the model response to a file. When no
filename is supplied, the first attached file name with `.txt` appended is
used; if no files are attached, `response.txt` is created.
