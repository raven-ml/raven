# Quill

**A writing-first, interactive document environment for OCaml**

Quill blends the simplicity of distraction-free writing with the power of live OCaml code. It’s built for focus, whether you’re working solo or collaborating, reimagining the notebook experience to keep you in the flow.

## Features

- Minimal markdown-based rich text editing with live formatting
- OCaml cells fit naturally into your document, part of the story, not an interruption.
- Inline code execution with persistent session state
- Data visualization integration via Hugin
 
## Quick Start

Quill can be used in two main ways: as a web server for interactive documents or to execute code blocks in Markdown files.

### Starting the Web Server

```bash
# Start the Quill web server, you can serve a directory
quill serve quill/example/
# or a specific file
quill serve quill/example/demo.md

# Specify a port (default is 8080)
quill serve --port 3000 ~/documents/quill

# Open your browser to http://localhost:3000 (or your specified port)
```

### Executing Code in Markdown Files

```bash
# Execute code blocks in a Markdown file and update it with outputs
quill exec quill/example/demo.md

# Watch mode: automatically re-execute when the file changes
quill exec -w quill/example/demo.md
```
 
## Contributing
 
See the [Raven monorepo README](../README.md) for guidelines.
 
## License
 
ISC License. See [LICENSE](../LICENSE) for details.
