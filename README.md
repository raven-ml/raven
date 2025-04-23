# Raven

**OCaml's Wings for Machine Learning**

Raven is a comprehensive ecosystem of libraries, frameworks, and tools that brings machine learning and data science capabilities to OCaml.

## Vision

Raven aims to make training models, running data science tasks, and building pipelines in OCaml as efficient and intuitive as Python, while leveraging OCaml's inherent type safety and performance advantages. We prioritize developer experience and seamless integration.

## Status

Raven is currently in **pre-alpha** and we're seeking user feedback:

- **[Ndarray](ndarray/)** and **[Hugin](hugin/)**: Scope is feature-complete for the first alpha release, though feedback may influence refinements.
- **[Rune](rune/)**: Proof-of-concept stage.
- **[Quill](quill/)**: Early prototyping phase.

## The Ecosystem

Raven is a constellation of sub-projects, each addressing a specific aspect of the machine learning and data science workflow:

- **[Ndarray](ndarray/)**: The core of Raven, providing high-performance numerical computation with multi-device support (CPU, GPU), similar to NumPy but with OCaml's type safety.
  - **[Ndarray-CV](ndarray-cv/)**: A collection of computer vision utilities built on top of Ndarray.
  - **[Ndarray-IO](ndarray-io/)**: A library for reading and writing Ndarray data in various formats.
  - **[Ndarray-Datasets](ndarray-datasets/)**: Easy access to popular machine learning and data.
science datasets as Ndarrays.
- **[Quill](quill/)**: An interactive notebook application for data exploration, prototyping, and knowledge sharing.
- **[Hugin](hugin/)**: A visualization library that produces publication-quality plots and charts.
- **[Rune](rune/)**: A library for automatic differentiation and JIT compilation, inspired by JAX.
- **(More to come!)**: Raven is an evolving ecosystem, and we have exciting plans for additional libraries and tools to make OCaml a premier choice for machine learning and data science.

## Python vs Raven: A Comparison

The table below compares Python's popular data science libraries with their Raven counterparts. For detailed code examples, see the linked documentation files.

| Task                      | Python Ecosystem    | Raven Ecosystem     | Comparison Guide                                   | Examples                     |
| ------------------------- | ------------------- | ------------------- | -------------------------------------------------- | ---------------------------- |
| Numerical Computing       | NumPy               | [Ndarray](ndarray/) | [Comparison Guide](docs/compare_python_ndarray.md) | [Examples](ndarray/example/) |
| Visualization             | Matplotlib, Seaborn | [Hugin](hugin/)     | [Comparison Guide](docs/compare_python_hugin.md)   | [Examples](hugin/example/)   |
| Notebooks                 | Jupyter             | [Quill](quill/)     | N/A                                                | N/A                          |
| Automatic Differentiation | JAX                 | [Rune](rune/)       | *In progress*                                      | *In progress*                |
| Dataframe Manipulation    | Pandas              | *Not yet*           | N/A                                                | N/A                          |
| Deep Learning             | Pytorch, Tensorflow | *Not yet*           | N/A                                                | N/A                          |

## Contributing

We welcome contributions from everyone—whether you're an OCaml expert, a data scientist, or simply curious about the project:

- Report issues for bugs or feature requests
- Submit pull requests for code improvements, documentation, or examples

See our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

Raven is available under the [ISC License](LICENSE), making it free for both personal and commercial use.
