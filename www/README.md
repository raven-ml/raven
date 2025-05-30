# Raven Website

This directory contains the static website for the Raven ecosystem, hosted at [raven-ml.dev](https://raven-ml.dev).

## Overview

The website is built using [Soupault](https://www.soupault.app/), a static site generator that processes HTML templates and Markdown content to generate the final site.

## Directory Structure

```
www/
├── README.md              # This file
├── soupault.toml          # Soupault configuration
├── site/                  # Source content
│   ├── index.html         # Landing page
│   ├── docs/              # Documentation pages
│   │   ├── index.html     # Documentation index
│   │   ├── nx/            # Nx documentation
│   │   ├── hugin/         # Hugin documentation
│   │   ├── rune/          # Rune documentation
│   │   ├── quill/         # Quill documentation
│   │   ├── kaun/          # Kaun documentation
│   │   └── sowilo/        # Sowilo documentation
│   └── *.html             # Project landing pages
├── templates/             # HTML templates
│   ├── layout_docs*.html  # Documentation templates
├── plugins/               # Custom Soupault plugins
└── build/                 # Generated site output
```

## Building the Website

```bash
# Build the website
dune -w build build/
```

The generated website will be in the `build/` directory. Serve it with

```bash
python3 -m http.server 8000
```
