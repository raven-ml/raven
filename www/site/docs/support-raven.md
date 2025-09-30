# Support Raven

## Raven in One Minute

Python's monopoly on scientific computing forces an impossible choice: ship everything in Python (endure runtime crashes, the GIL's multicore ceiling, and gigabyte containers), or prototype in Python then rewrite for production (doubling the work and creating siloed teams).

**We think there's a better way.** OCaml lets you prototype as quickly as Python and scale the same code to production. Same expressiveness, strong typing catches bugs before they crash your ML pipeline, while JIT compilation matches NumPy/PyTorch performance. One language from research to production—it just needs a production-grade ML stack.

**Raven brings that stack to OCaml:** Nx (NumPy), Rune (JAX with effects-based autodiff), Kaun (PyTorch/Flax), Hugin (Matplotlib), and Quill (notebooks done right). You can now ship ML models in 10 MB statically-linked binaries; Port to new hardware in days. We built Raven for teams that want both development speed and reliable systems.

_Learn more: [Introduction](/docs/introduction)_

_We're currently pre-alpha with working simple neural networks. Next milestone: An MNIST demo in Quill running on CUDA/Metal GPU and Rune's kernel-fusing JIT._

## Roadmap & Funding Goals

_See the [full roadmap](/docs/roadmap) for our complete vision and timeline._

### €50k - First Release (Q3 2025)
- MNIST training demo in Quill notebook
- CUDA and Metal GPU backends for Nx
- Basic JIT compilation in Rune with kernel fusion
- API stabilization and v1.0 release

### €150k - Performance Parity (Q1 2026)
- Full JIT compiler targeting LLVM, Metal and Cuda
- Performance matching NumPy/PyTorch on standard benchmarks
- Complete Kaun with modern architectures (transformers, etc.)

We're also open to discussing custom sponsorship packages based on your needs.

## Ways to Support

### For Developers
- **Try it out**: Test Raven with your workflows and [report issues](https://github.com/raven-ml/raven/issues)
- **Contribute code**: See our [contributing guide](https://github.com/raven-ml/raven/blob/main/CONTRIBUTING.md) for areas where we need help
- **Share feedback**: What would make you switch from Python? [Tell us](mailto:thibaut.mattio@gmail.com)
- **Spread the word**: Star the repo, share with your team, write about your experience

### For Companies
- **Use Raven**: Reach out if you're interested in using it—we're keen on prioritizing development based on real-world needs
- **Sponsor development**: Email [thibaut.mattio@gmail.com](mailto:thibaut.mattio@gmail.com) for sponsorship packages

### For Individuals
- **GitHub Sponsors**: [Support the project with monthly contributions](https://github.com/sponsors/tmattio)
- **One-time donations**: Every contribution helps us reach the next milestone
- **Write tutorials**: Help others learn Raven and grow the community

## Current Sponsors

We're grateful for the support of our sponsors:

### Corporate Sponsors

- [**Ahrefs**](https://ahrefs.com) - Building tools to help you grow your search traffic
- [**Tarides**](https://tarides.com) - Secure-by-design infrastructure and tooling for a better digital world

### Individual Sponsors

Thank you to all our individual sponsors for their support!

## Get in Touch

**For sponsorship inquiries**: [thibaut.mattio@gmail.com](mailto:thibaut.mattio@gmail.com)  
**For feature request or bug reports**: [GitHub Issues](https://github.com/raven-ml/raven/issues)

---

_Raven is built by [Thibaut Mattio](https://github.com/tmattio) and contributors. We believe OCaml deserves a world-class scientific computing ecosystem, and we're committed to building it._
