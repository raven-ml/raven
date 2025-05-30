# quill Documentation

Quill reimagines notebooks for scientific computing in OCaml.

## What quill Does

Quill is our take on Jupyter, but different. Instead of JSON files with cells, quill uses markdown files where code and prose flow together naturally. Write your analysis in markdown, mark code blocks as OCaml, and quill executes them. Simple.

This design means notebooks are plain text, diff beautifully in git, and work with any editor. No more merge conflicts from notebook metadata.

## Current Status

Quill is in active development for the alpha release.

What works today:
- Markdown notebook format
- OCaml code execution
- Basic web interface
- Plot display from hugin

What's coming:
- Rich output (tables, images, LaTeX)
- Export to HTML/PDF
- Notebook templates

## Philosophy

Quill embraces simplicity. A notebook is just a markdown file with an OCaml toplevel running through it. Each code block builds on the previous one's context. No magic, no hidden state, just OCaml code executing in sequence.

The web interface feels more like Typora than Jupyter, seamless switching between writing and viewing, with code execution as a natural part of the flow.

## Learn More

- Getting Started - (coming soon)
- Notebook Format - (coming soon)