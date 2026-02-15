# Introduction

Raven is a project to bring modern scientific computing to the OCaml programming language. We're building a comprehensive ecosystem, from low-level numerical libraries and automatic differentiation to high-level machine learning frameworks and interactive notebooks.

Our ambition is to make scientific computing in OCaml feel as natural as it does in Python. This means not just matching Python's capabilities, but delivering the same level of ergonomics, performance, and developer experience that has made Python the de facto standard for scientific computing.

If successful, Raven would establish OCaml as a genuine alternative in the scientific computing landscape. It's an ambitious undertaking, but one we believe is both necessary and achievable.

## Why Not Just Use Python?

Today, Python has an effective monopoly on scientific computing. Unlike web development, where we can choose between multiple mature ecosystems, numerical computing offers essentially one realistic option. This lack of choice is unfortunate.

What's more problematic is that Python, while excellent for quick experimentation, doesn't particularly shine for building robust production systems. Its interpreted nature, dynamic typing, and limited multicore support create real challenges when you need to deploy and maintain large-scale applications.

If you've worked in this space, you've likely experienced this firsthand: rapid prototypes that become production nightmares, debugging sessions where type errors only surface at runtime, or performance bottlenecks that force you to drop down to C extensions.

Often, this mismatch forces a wasteful pattern: researchers prototype in Python, then teams reimplement everything for production in other languages. This induces all kinds of second-order effects on organization structures, team dynamics, development velocity, and workload.

The scientific community deserves better options than being forced into one language, and we believe OCaml occupies a unique sweet spot between rapid experimentation and building production-grade systems. It just needs the scientific ecosystem to match its technical strengths. This is the gap that Raven aims to fill.

In the AI era, we believe OCaml has an important role to play. If you're generating 80% of your code with AI assistance, wouldn't you prefer a language that catches errors at compile time rather than runtime? The productivity gains from AI coding are amplified when you have a type system that gives you stronger guarantees about your generated code. Raven is our contribution to putting OCaml in the spotlight for scientific computing in this new era.

## What Does Success Look Like?

Our goal isn't just to build OCaml versions of Python libraries: it's to create a compelling alternative for busy developers who just want the best tool for the job.

Success means two things. First, **OCaml developers shouldn't have to switch to Python for numerical computing**. Whether you're analyzing data, training models, or building computational systems, you should be able to stay in the OCaml ecosystem with the same productivity you'd expect from Python.

Second, **Raven should break into the mainstream scientific computing conversation**. It shouldn't just serve existing OCaml developers: we're building for teams who need to ship reliable systems, not just an OCaml curiosity for language enthusiasts.

We measure success across five key dimensions:

- **Capability parity**: Everything you can do in Python, you should be able to do with Raven
- **Development productivity**: Getting from idea to working prototype is as fast as it would be in Python
- **Developer experience**: Developers get the kind of documentation, tooling, and APIs they dream every project had
- **Production performance**: Match or exceed NumPy/PyTorch performance on the fast path
- **Production readiness**: Teams can ship robust, maintainable Raven-built applications that perform well under real-world conditions

We believe this is achievable through focused execution and strategic choices. We're prioritizing the 80% that matter most, focusing on one blessed workflow per use-case, and building modular components that encourage ecosystem growth, rather than trying to match Python everywhere from day one.

## Why Not Just Use Owl?

Owl deserves credit for the amount of work and love that has been poured into it. It demonstrated that serious numerical computing in OCaml was possible, spanning everything from statistics and signal processing to basic linear algebra and neural networks, and more.

However, Owl can't compete with NumPy or PyTorch on performance, and performance parity isn't optional if we want teams to seriously consider OCaml over Python.

The reality is that we can't realistically match NumPy and PyTorch's performance through traditional optimization. These projects have hundreds of developers working on hand-optimized kernels. With our small team, JIT compilation is our only viable path to competitive performance.

This creates a fundamental constraint. Building for JIT-first changes everything about your design: API choices, memory layouts, operator fusion strategies, even how you structure the development experience. Rather than retrofitting these assumptions onto existing work, we decided a clean slate would be more effective.

There's also the ecosystem question. Despite Owl's technical achievements, it hasn't generated the kind of flourishing community we need. We suspect this is partly due to its lack of modularity: without libraries designed as composable building blocks, it's challenging to build a broader ecosystem around the foundation.

Raven is designed from the ground up to (1) compete with Python's scientific computing stack on performance and (2) build the flourishing ecosystem that OCaml's scientific computing community deserves.

## What We're Building

Raven is a comprehensive ecosystem that spans the entire scientific computing stack. Here's what we're building:

**Core Libraries**
- **Nx**: N-dimensional arrays with pluggable backends, our NumPy equivalent
- **Hugin**: Publication-quality data visualization and plotting, our Matplotlib equivalent  
- **Quill**: Interactive notebooks designed for scientific writing

**The Rune Ecosystem**
- **Rune**: Automatic differentiation with multi-device support and JIT compilation (our Flax equivalent)
- **Kaun**: Deep learning framework built on Rune (our PyTorch equivalent)
- **Sowilo**: Computer vision framework built on Rune (our OpenCV equivalent)

This gives us coverage from basic numerical operations to high-level machine learning, with everything designed to work together seamlessly.

**Key Innovations**
While we aim to feel familiar to Python users, Raven brings genuine innovations to scientific computing:

**Nx** uses pluggable backends inspired by Tinygrad's minimalist approach, giving us flexibility to optimize for different hardware without monolithic complexity.

**Rune** implements automatic differentiation using OCaml's effects system. As far as we know, it is the first project of this scale to use effects for autodiff, building on recent research, and implementing Jax's vision for functional numerical computation with a truly functional foundation.

**Quill** reimagines the notebook experience. Instead of Jupyter's cell-based approach, we're building something closer to Typora: a seamless writing experience where code and markdown flow together naturally, with plans for real-time collaboration.

**Current Focus**
We're currently focusing on machine learning workflows, with the goal of training and deploying models in OCaml with performance on par with leading Python frameworks. This means prioritizing the complete ML pipeline: from data manipulation with Nx, through model development and training with Rune and Kaun, to visualization with Hugin and interactive development with Quill.
