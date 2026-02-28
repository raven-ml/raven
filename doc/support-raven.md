# Support Raven

## Raven in One Minute

Python's monopoly on scientific computing forces an impossible choice: ship everything in Python (endure runtime crashes, the GIL's multicore ceiling, and gigabyte containers), or prototype in Python then rewrite for production (doubling the work and creating siloed teams).

**We think there's a better way.** OCaml lets you prototype as quickly as Python and scale the same code to production. Same expressiveness, strong typing catches bugs before they crash your ML pipeline, while JIT compilation matches NumPy/PyTorch performance. One language from research to production — it just needs a production-grade ML stack.

**Raven brings that stack to OCaml:** Nx (NumPy), Rune (JAX with effects-based autodiff), Kaun (Flax), Brot (tokenization), Hugin (Matplotlib), and Quill (notebooks done right). Train models with automatic differentiation and JIT compilation, then deploy as a MirageOS unikernel or a static binary — no Python, no CUDA dependency hell, no 5 GB Docker images. We built Raven for teams that want both development speed and reliable systems.

_Learn more: [Introduction](/docs/introduction)_

_We're in alpha with the full stack working end-to-end (we've trained GPT-2 on CPU). Next milestone: JIT compilation via tolk with performance close to PyTorch._

## Roadmap & Funding Goals

_See the [full roadmap](/docs/roadmap) for our complete vision and timeline._

### Beta — JIT Compilation & Performance
- Integrate tolk (tinygrad-based compiler) as a JIT transformation in Rune
- Target CPU, CUDA, Metal, OpenCL, and HIP
- Kernel fusion and optimization
- Performance within 2x of PyTorch on standard workloads

### V1 — Production-Ready Training & Deployment
- Production training: gradient accumulation, mixed precision, gradient checkpointing, flash attention
- ONNX import for PyTorch model portability
- AOT compilation to standalone binaries (CPU and GPU)
- Inference engine with KV cache, continuous batching, and PagedAttention
- MirageOS unikernel deployment
- Post-training quantization (INT8/INT4)

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
