# Roadmap

Raven is currently in alpha. We've focused on the core infrastructure to train large language models. We're successfully training GPT2 on CPU using the full Raven stack (Kaun → Rune → Nx).

## Alpha Releases ✅ 

**alpha1** has been released with three new libraries (Talon, Saga, Fehu) and major enhancements to Nx, Rune, and Kaun. This represents the complete scope for alpha.

**Future alpha releases** will focus exclusively on bug fixes and stability improvements. No new features are planned for the alpha cycle.

Key achievements:
- Complete Nx numerical computing capabilities (linear algebra, FFT, extended dtypes)
- Expanded Kaun with high-level training APIs inspired by PyTorch and Flax
- Successfully trained GPT2 using the full Raven stack on CPU
- Added DataFrame processing (Talon), NLP (Saga), and reinforcement learning (Fehu)

## Beta: JIT Compilation & Performance (Current Stage)

The beta cycle will have a single focus: **JIT compilation with performance close to PyTorch**.

- Complete LLVM-based JIT compiler for Rune
- Target CPU, CUDA, and Metal for hardware acceleration
- Optimize compilation pipeline and runtime performance
- Benchmark against PyTorch on standard workloads
- Achieve competitive performance on common deep learning tasks

## V1: Developer Experience

Once performance is competitive using JIT compilation, V1 will focus on **developer experience and documentation**:

**Developer Tooling**
- Complete Hugin (plotting library) with publication-quality visualizations
- Complete Quill (notebook environment) for interactive data science
- Integrated workflows for data scientists coming from Python

**Documentation**
- Comprehensive tutorials and getting-started guides
- Complete API reference documentation
- Migration guides for NumPy/PyTorch users
- Real-world examples and case studies

**API Stability**
- Finalize and stabilize all public APIs
- Ensure backward compatibility guarantees

## Post-V1: Production Scale

After V1, we'll focus on **scaling for real-world production constraints**:

**Distributed Training**
- Multi-GPU training on a single machine
- Distributed training across multiple machines
- Efficient data parallelism and model parallelism

**Deployment**
- Model serving infrastructure
- Optimization for inference workloads
- Integration with deployment platforms

**Production Readiness**
- Monitoring and observability tools
- Performance profiling and optimization
- Enterprise support and stability guarantees

The path is clear: alpha proves the concept, beta matches Python's performance, V1 delivers great developer experience, and post-V1 enables production ML at scale.
