# nx-oxcaml

nx-oxcaml is a high-performance nx backend using oxcaml's unboxed types and SIMD intrinsics.

it is part of the raven ecosystem. see the root [AGENTS.md](../AGENTS.md) for overall project philosophy and guidelines.

## project structure

- `lib/` - main library (`nx_oxcaml` / `nx-oxcaml`)
- `test/` - test suite (`test_nx_oxcaml`)
- `bench/` - benchmarks (`bench_nx_oxcaml`)
- `vendor/` - vendored dependencies

## build instructions

all dune commands MUST be run from the `nx-oxcaml/` directory with `--root .` to get an isolated build that does not conflict with the parent raven project.

```sh
# build
dune build --root .

# run tests
dune test --root .

# run benchmarks
dune exec --root . bench/bench_nx_oxcaml.exe

# watch mode
dune build --root . --watch
```

## important rules

- ALWAYS use `--root .` with every dune command
- ALWAYS run dune commands from the `nx-oxcaml/` directory
- NEVER run dune commands from the raven root — this will cause conflicts with the parent project
- NEVER stage or commit changes unless explicitly requested
- NEVER run `dune clean`
- NEVER use the `--force` argument
- NEVER try to remove the dune lock file or kill dune when running in watch mode
- NEVER hide warnings and NEVER hide unused variables by adding an underscore
