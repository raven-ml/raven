# Getting Started with Quill

Create your first interactive notebook in minutes.

## What is Quill?

Quill turns markdown files into interactive OCaml notebooks. Write your analysis as a document, mark code blocks as OCaml, and Quill executes them in sequence.

## Quick Start

### 1. Create a Notebook

Create a file `notebook.md` with markdown content and OCaml code blocks:

- Start with a title and introduction
- Add OCaml code blocks using triple backticks
- Write explanations between code blocks
- Structure with markdown headings

### 2. Run Your Notebook

Three ways to execute:

```bash
quill eval notebook.md           # Print results to terminal
quill eval --inplace notebook.md # Save outputs in the file
quill serve notebook.md          # Interactive web interface
```

### 3. View Results

After execution, outputs appear as HTML comments below each code block. These preserve your markdown's readability while storing results.

## Key Concepts

### Persistent State

Variables and functions defined in one code block are available in all subsequent blocks. This lets you build up your analysis step by step.

### Rich Output

- **Nx and Rune Tensors**: Display as formatted matrices
- **Hugin Figures**: Render as inline images
- **Errors**: Show with full context
- **Values**: Print with their types

### Markdown First

Your notebook remains a valid markdown file that renders beautifully on GitHub, in editors, or anywhere else markdown is supported.

## Execution Modes

### Command Line

For quick execution and automation:

```bash
quill eval notebook.md
```

### In-Place Updates

To save outputs directly in your notebook:

```bash
quill eval --inplace notebook.md
```

### Watch Mode

For interactive development:

```bash
quill eval --watch --inplace notebook.md
```

### Web Interface

For a full notebook experience:

```bash
quill serve notebook.md
# Open http://localhost:8080
```

## Next Steps

- Explore the [example notebooks](https://github.com/raven-ml/raven/tree/main/quill/example)
- Learn about [advanced features](advanced.md) (coming soon)
- Join the community and share your notebooks