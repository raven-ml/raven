# Raven in Late 2025: Alpha Complete, Toward the JIT

*This update was originally shared with our sponsors in November 2025. We're publishing it retroactively as the first entry in our public update series.*

[Raven](https://github.com/raven-ml/raven) is an ecosystem of OCaml libraries for machine learning and scientific computing. This update covers the work since the zeroth alpha in July 2025.

## Community in numbers

- Discord: 80 members
- Contributors: 19 total, 4 regular contributors
- Users: 4 users we're working with closely, and many enthusiasts based on open issues

## Summary

We released alpha1 in October. It included 3 new libraries (Fehu for reinforcement learning, Saga for tokenization and NLP, Talon for DataFrame manipulation), and many important new features, including:

- Support for FFT in Nx
- A complete linear algebra suite
- Support for forward-mode differentiation
- Support for new data types, including boolean arrays and machine-learning-specific ones (FP16, BF16, etc.)

We hosted 2 workshops at FunOCaml in September, covering the training of LLMs and reinforcement learning, and there has since been demand from several companies for tailored workshops.

Raven has submitted two projects for Outreachy:

- Development of an OxCaml backend
- Development of a terminal-based monitoring dashboard equivalent to TensorBoard

Our scope for alpha is complete. We'll be releasing alpha2 with Outreachy contributions from October, and any future alpha release will focus on correctness and performance (with the exception of the Outreachy contributions and projects).

We're now focusing on releasing our first beta, which will come with an initial version of our JIT compilation pipeline. It will be largely inspired by tinygrad's design, and will provide the same benefits — notably a minimal JIT backend interface for fast backend development — and targets better performance than JAX/PyTorch.

We're excited to get there and what comes beyond. We believe this will position Raven with very real benefits over existing solutions, in particular enabling the deployment of optimized models as unikernels to save on infrastructure costs and improve performance.

## Refined Scope for Alpha

Back in July, our focus for the alpha was on building an end-to-end demo of a development workflow using Quill, Raven's Jupyter notebook. The target was to train an MNIST model within a Quill notebook, with integrated data visualizations and images.

Following the zeroth alpha, a few early users got in touch to start using Raven in their projects, in particular:

- SoundML, a digital signal processing (DSP) library for OCaml, was interested in porting their code from Owl to Nx, and needed support for FFT in Nx for it.
- A neuroscience lab at the University of Cambridge was interested in using Raven in place of PyTorch bindings for OCaml, and needed a complete linear algebra suite in Nx, as well as support for forward-mode differentiation.
- A researcher at The American University in Cairo reached out about the need for a DataFrame library for data exploration, which led to the creation of Talon, our Pandas equivalent.

None of them (and other users) were particularly interested in using Quill, so following that feedback, we refined our alpha scope to remove developer tooling and focus on the development of the libraries.

In parallel, we committed to hosting two Raven workshops at FunOCaml in September:

- A workshop on training Large Language Models (LLMs)
- A workshop on Reinforcement Learning (RL), hosted by Lukasz Stafiniak, the author of OCANNL, another ML framework for OCaml, with whom we've had close collaboration.

As part of the preparation for these workshops, we extended Kaun, Raven's deep learning library, to support transformers, and created two new libraries:

- Saga, a text processing and NLP library offering utilities necessary for LLMs such as tokenizers and samplers.
- Fehu, a reinforcement learning library, whose design borrows from Gymnasium and Stable Baselines3.

Overall, based on user feedback from alpha0 and in preparation for the FunOCaml workshops, we've extended the number of Raven libraries quite a bit, and have been working on all of them:

| Library | Description                                   | Equivalent                          |
| ------- | --------------------------------------------- | ----------------------------------- |
| Nx      | N-dimensional arrays with pluggable backends  | NumPy                               |
| Saga    | Text processing and tokenization for NLP      | HuggingFace tokenizers/transformers |
| Talon   | DataFrames with heterogeneous columns         | Pandas/Polars                       |
| Rune    | Autodiff with multi-device support and JIT    | JAX                                 |
| Kaun    | Deep learning framework on Rune               | Flax/PyTorch                        |
| Fehu    | RL environments and algorithms                | Gymnasium/SB3                       |
| Hugin   | Data visualization and plotting               | Matplotlib                          |
| Quill   | Scientific notebook experience                | Jupyter                             |
| Sowilo  | Differentiable computer vision on Rune        | OpenCV                              |

We released Raven alpha1 in October, shortly after FunOCaml. It comes with a first version of the new libraries mentioned above, FFT support, a complete linear algebra suite, and much more (see the release notes on GitHub).

With all of these, we believe the core scope of Raven is close to being complete (we're considering adding a library for distributed computing later on). The libraries themselves are of course far from complete, but we have laid down all the foundations we'll be building on from now on.

Sitting from this, we believe we've exceeded the scope of alpha, and won't be adding any new features as part of our alpha releases. Future alpha releases (with the exception of alpha2, see Outreachy Internships) will focus only on correctness and performance, and we're now ready to focus on our first beta (see Road to Beta).

## Outreachy Internships

We have submitted two projects for the current Outreachy cohort:

- Development of an OxCaml backend
- Development of a terminal-based monitoring dashboard equivalent to TensorBoard

We started accepting contributions as part of the application period, and have seen an impressive level of enthusiasm and quality of contributions — we're totalling 30 contributions from a dozen applicants.

The internships will start early December.

### Development of an OxCaml backend

The goal is to develop a new Nx backend, offering an alternative to our current C backend, using Jane Street's OxCaml compiler.

We hope to match our C backend performance on most operations.

Ultimately, the goal is to provide a real-world community use case for some of the extensions in OxCaml, which might aid in the upstreaming process. We're excited about the potential for OxCaml to make OCaml a great language for safe performance-engineering, and how this could benefit Raven.

### Development of a terminal-based monitoring dashboard

The goal of this project is to offer a way to monitor model training runs with Raven.

The Python community typically uses TensorBoard, and we think we can build a better developer experience by shipping a terminal-based UI which covers the same features.

We'll be using Mosaic, an upcoming TUI library for OCaml, which we're planning to release in the coming weeks.

If we see interest from the broader ML community, the tool could be shipped as a standalone executable to offer an alternative to TensorBoard outside the Raven ecosystem.

## Road to Beta

As mentioned previously, we have completed our scope for alpha, and, with the exception of the Outreachy contributions, future alphas will focus solely on correctness and performance for reported issues. Our main focus is now to release a first beta with an initial version of our JIT compilation pipeline.

We're still exploring the design space, but we're leaning more and more toward porting tinygrad to OCaml — their design and philosophy is completely aligned with Raven's, and they are on track to exceed every other solution on both performance and elegance. We're still considering the pros and cons, but a port rather than a new implementation would fit Raven's goal better and would allow us to focus on developer experience rather than going deep into building an ML compiler.

We're excited about future steps once we've shipped a JIT compilation pipeline: at this stage, Raven will have all the building blocks to build modern models, including LLMs (alpha) with performance matching PyTorch (beta). The natural next steps will be to enable large-scale training and deployment.

In particular, one line of exploration we're anticipating is the deployment of models as unikernels. We think ML infrastructure suits the benefits of unikernels particularly well, as a way to save cost and increase performance with horizontal scaling.

## Thanks

Thank you to our sponsors — your support directly enabled the completion of our scope for alpha, the hosting of workshops at FunOCaml, and the mentoring of applicants and future Outreachy interns, and it is securing the development of Raven toward a first beta. And thank you to everyone who contributed, filed issues, and joined the community.

We're seeing the build-up of an enthusiastic community of users and contributors, and are looking forward to continuing to make OCaml a great language for machine learning and scientific computing.
