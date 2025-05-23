# Raven Website

This directory contains the static website for the Raven ecosystem, hosted at [raven-ml.dev](https://raven-ml.dev).

## Overview

The website is built using [Soupault](https://www.soupault.app/), a static site generator that processes HTML templates and Markdown content to generate the final site.

## Directory Structure

```
www/
├── README.md              # This file
├── soupault.toml         # Soupault configuration
├── site/                 # Source content
│   ├── index.html       # Landing page
│   ├── docs/            # Documentation pages
│   │   ├── index.html   # Documentation index
│   │   ├── ndarray/     # Ndarray documentation
│   │   ├── hugin/       # Hugin documentation
│   │   ├── rune/        # Rune documentation
│   │   ├── quill/       # Quill documentation
│   │   ├── kaun/        # Kaun documentation
│   │   └── sowilo/      # Sowilo documentation
│   └── *.html           # Project landing pages
├── templates/           # HTML templates
│   ├── main.html       # Base template
│   ├── layout_docs*.html # Documentation templates
│   └── partials/       # Reusable template components
├── plugins/             # Custom Soupault plugins
└── build/              # Generated site output
```

## Building the Website

### Prerequisites

- [Soupault](https://www.soupault.app/) static site generator
- [Dune](https://dune.build/) (for integration with the main project)

### Build Commands

```bash
# Build the website
dune build

# Clean build artifacts
dune clean
```

The generated website will be in the `build/` directory.

## Development

### Adding New Pages

1. Create HTML or Markdown files in the `site/` directory
2. Use appropriate templates from `templates/` for consistent styling
3. Update navigation in `templates/partials/project-nav.html` if needed

### Templates

- `main.html`: Base template for all pages
- `layout_docs.html`: General documentation template  
- `layout_docs_*.html`: Project-specific documentation templates
- `partials/`: Reusable components like navigation and footer

### Configuration

The website behavior is configured in `soupault.toml`:

- Template assignments for different sections
- Widget configurations for breadcrumbs, titles, etc.
- Plugin settings

### Styling

The website uses Tailwind CSS classes throughout the templates for styling.

## Content Guidelines

- Keep content focused and user-friendly
- Use consistent terminology across all documentation
- Include code examples where appropriate
- Maintain clear navigation between related topics

## Deployment

The website is automatically deployed when changes are pushed to the main branch. Build artifacts in `build/` should not be committed to the repository.
