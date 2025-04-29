# Raven Development Sandbox

This directory contains experimental prototypes, incubation projects, and development scratchbooks that support the Raven ecosystem.

## Purpose

The `dev/` directory serves as:

- An **incubation space** for early-stage ideas that may eventually graduate to full Raven projects
- A **sandbox** for exploring algorithms, techniques, and implementation approaches
- A **scratchbook** for kernel development and hardware-specific optimizations
- A **testing ground** for integrations with external systems and libraries


## Projects

| Name                | Description                                | Status      |
| ------------------- | ------------------------------------------ | ----------- |
| **`ndarray-metal`** | Metal backend for Ndarray                  | Incubation  |
| **`ndarray-cuda`**  | Cuda backend for Ndarray                   | Exploration |
| **`kaun`**          | Deep Learning library built on top of Rune | Exploration |

## Project Status

Projects in this directory exist in various stages of development:

| Status          | Description                                                        |
| --------------- | ------------------------------------------------------------------ |
| **Exploration** | Initial research and experimentation                               |
| **Prototyping** | Proof-of-concept implementation with limited functionality         |
| **Incubation**  | More structured development with potential for graduation          |
| **Graduating**  | Being prepared for promotion to a standalone Raven project         |
| **Archived**    | Experiments that provided valuable insights but won't be continued |

## Directory Structure

```
dev/
├── kernels/             # Low-level performance optimizations and hardware acceleration
├── prototypes/          # Early-stage project prototypes
├── notebooks/           # Example notebooks and experiments
```

## Development Guidelines

While this is an experimental space, we maintain certain standards to keep the development process productive:

1. **Document your experiments** - Even failed attempts provide valuable knowledge
2. **Use meaningful directory names** - Help others understand your work at a glance
3. **Keep a README in each project** - Explain the purpose, status, and how to run your code
4. **Tag abandoned experiments** - Mark projects you're no longer actively developing

## Graduating Projects

When a project in `dev/` shows promise for wider adoption, it may graduate to becoming a full Raven component. The graduation process involves:

1. Code review and architecture assessment
2. Documentation expansion
3. Test coverage improvement
4. Performance benchmarking
5. API stabilization

## Archiving Projects

If a project is deemed unfit for graduation, it may be archived. Archived projects are removed from the main development branch.

## Contributing

The `dev/` directory welcomes exploratory contributions:

- Feel free to create new subdirectories for your experiments
- Document your approach and findings
- Consider writing simple tests for core functionality
- Share interesting results with the community

See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for general contribution guidelines.

## License

All code in this directory is available under the [ISC License](../LICENSE), consistent with the rest of the Raven ecosystem.
