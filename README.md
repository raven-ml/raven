# Raven

**Modern scientific computing for OCaml.**

Raven is a comprehensive ecosystem that brings scientific computing capabilities to OCaml, designed for teams who need both rapid prototyping and production-ready systems.

We're prioritizing developer experience and competitive performance to give developers a real choice beyond Python for scientific computing.

> **Note**: Raven is currently in **pre-alpha**. We're actively seeking user feedback to shape the project's direction. Please [open issues](https://github.com/raven-ml/raven/issues) or reach out [by email](mailto:thibaut.mattio@gmail.com)!

## The Ecosystem

Raven is built from modular projects that form a cohesive ecosystem:

**Core Libraries:**
| **Raven Project**   | **Python Equivalent** | **Description**                                     |
| ------------------- | --------------------- | --------------------------------------------------- |
| [**Nx**](nx/)       | NumPy                 | N-dimensional arrays with pluggable backends        |
| [**Hugin**](hugin/) | Matplotlib            | Publication-quality data visualization and plotting |
| [**Quill**](quill/) | Jupyter               | A love letter to scientific writing                 |

**Rune Ecosystem:**
| **Raven Project**       | **Python Equivalent** | **Description**                                        |
| ----------------------- | --------------------- | ------------------------------------------------------ |
| [**Rune**](rune/)       | JAX                   | Autodiff with multi-device support and JIT compilation |
| [**Kaun** ᚲ](kaun/)     | PyTorch/Flax          | Deep learning framework built on Rune                  |
| [**Sowilo** ᛊ](sowilo/) | OpenCV                | Computer vision framework built on Rune                |

## Why Raven?

- **Ship reliable systems**: Strong typing catches bugs that would crash your ML pipeline in production
- **Stop rewriting code**: Prototype and deploy in the same language, no more "Python for research, X for production"
- **Match Python's performance**: JIT compilation designed to compete with NumPy and PyTorch, not just beat them by 20%
- **Built for developers**: Notebooks that feel like writing documents, intuitive APIs, and an ecosystem designed to work seamlessly together
- **Designed for extension**: Pluggable backends, modular architecture, and building blocks you can actually extend and customize

## Documentation

**[📖 Read the Introduction](docs/book/01-introduction.md)** - Learn about our vision, philosophy, and approach

_More comprehensive documentation and examples are coming soon as we are heading towards the release._

## Contributing

Raven is in active development and we welcome contributions from the community!

**Ways to Contribute:**
- **Share feedback** - [Open issues](https://github.com/raven-ml/raven/issues) or [email us](mailto:thibaut.mattio@gmail.com) with your thoughts on APIs, performance, or developer experience
- **Test the libraries** - Try Raven libraries with your workflows and report what works (or breaks)
- **Improve documentation** - Help us make the docs clearer and more comprehensive
- **Build new libraries** - Interested in creating any of the planned libraries below? Let's collaborate!

**Future Libraries (Open for Contributions):**
For our first release, we're focused on the foundation (Nx, Hugin, Quill) and the deep learning vertical (Rune, Kaun). These areas are planned for future development:

- **DataFrame library** (Pandas equivalent)
- **Statistical computing** (R-like statistical functions)  
- **Time series analysis** and **geospatial computing**
- **Distributed computing** (Dask equivalent)

Whether you're an OCaml expert or new to the language, we'd love your help building the future of scientific computing in OCaml!

See our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

Raven is available under the [ISC License](LICENSE), making it free for both personal and commercial use.
