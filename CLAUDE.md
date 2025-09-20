# AGENTS.md

## Project Overview

Raven: OCaml ML ecosystem. Pre-alpha.

**Libraries:** nx (tensors), hugin (plotting), quill (notebooks), rune (autodiff+JIT), sowilo (vision)

- **Philosophy**: Unix-style - do one thing well, fail loudly, clarity over cleverness
- **Commit messages**: Use conventional commit prefixes: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`, `style:`, `perf:`
  - Include scope prefix if useful (e.g., `feat(rune):`)
  - Start the commit with a capital letter (e.g., `feat(rune): Add ...`)
- When creating commits, only commit what's in the staged area if there's anything there, the user would have prepared the changes to commit themselves.
- When you run a dune command, if there is an instance of dune running, it means we're running in watch mode, DO NOT kill dune, and do not remove the lock. You should be able to build while dune is running.
- You can read the documentation of installed dependencies in `_build/_private/default/.pkg/<library>`, located at the root of the project (make sure your current working directory is at the root of the project)
- NEVER delete the build directory _build, the lock directory dune.lock, or the dune lock when running in watch mode. NEVER kill dune when running in watch mode.
- NEVER hide warnings with `[@warning -69"]`, and NEVER hide unused variables by adding an underscore. Warnings is the compiler telling us that the code is wrong, the implementation is incomplete, so ALWAYS understand what the issue is, whether we need to fix the implementation to use the variable, or remove the variable.

## Workflow

### Task Management

1. **Start**: Read `TODO.md` for current priorities
2. **Plan**: Complex tasks → write plan in `.claude/plans/YYYY-MM-DD-description.md`  
3. **Update**: Keep TODO.md current

### Testing & Debugging
- Float comparisons need `~eps`: `check_t ~eps:1e-6 "name" shape values result`

**Dune watch mode** (the user typically runs dune build in watch mode outside Claude session):
- Check for `_build/.lock` to see if watch mode is active
- Build errors visible via `dune build <dir>` (works with watch mode)
- Run tests directly: `_build/default/nx/test/test_foo.exe` (dune exec doesn't work with watch mode)
- **Never** kill dune, remove lock file, or clean build when watch mode is active
- In watch mode, When you make changes, check if there are build errors with `dune build <dir>` first, otherwise you'll be running the old tests.
- Do not create new stanzas (e.g. new `executable`), instead reuse existing stanzas (e.g. `executables`), otherwise dune watch mode won't build the new targets.

When tests fail:
1. Create a minimal repro file in `<project>/test/failing/bug_<bug_name>.ml`
2. Make sure the test is correct before fixing the implementation
3. Understand NumPy conventions (exclusive slice ends, dimension squeezing)
4. Print intermediate values to debug
5. **Never hack to pass tests**: Fix root cause, maintain correct semantics
   - View operations must not create copies
   - Don't change tests unless they're genuinely wrong
   - Preserve NumPy compatibility over convenience

## Essential Commands

```bash
dune build                        # Build everything
dune build nx/                    # Build targets in a directory
dune build @runtest               # Run all tests
dune build @nx/runtest            # Run tests in a directory
dune exec nx/test/test_foo.exe    # Run single test file
dune build fmt                    # Format code
```

**With dune watch mode active:**
```bash
# Check for build errors and run
dune build nx/test/test_foo.exe &&  _build/default/nx/test/test_foo.exe
```

## Code Style

- **Naming**: `snake_case` for values/functions/types, `My_Module` for modules/variants
- **Philosophy**: Unix-style - do one thing well, fail loudly, clarity over cleverness
- **Interfaces**: One `.mli` per `.ml`, keep minimal
- **Docs**: Terse first line, document invariants not obvious behavior
- **Errors**: `function_name: what went wrong` format, fail fast
- **Tests**: Alcotest framework, test edge cases, group related tests
- **Type annotations**: Avoid explicit types unless required by type checker (dtype pattern matching)

## Architecture

**Key files:**
- `nx/lib/core/backend_intf.ml` - Tinygrad-inspired UOps (~50 ops)
- `nx/lib/core/frontend.ml` - NumPy-like API
- `rune/lib/autodiff.ml` - Effect-based autodiff

## Critical Knowledge

- Don't clean the world: Never clean the dune cache or delete `_build` directory. The problem is not the build system.

### OCaml Gotchas
- **GADTs**: Can't group pattern match branches
- **Circular deps**: Watch for functions calling each other (use backend ops directly)
- **Dtype pattern matching**: Need locally abstract types: `let f (type a b) (x : (a, b) t) = match dtype x with ...`

### Nx Conventions
- **Slicing**: `R (1, 4)` = indices 1,2,3 (exclusive end like Python)
- **Single index**: `get [0]` on `[2; 3]` → shape `[3]` (dimension squeezed)
- **Creation funcs**: Return contiguous arrays (use `init` not `broadcast_to`)
- **NumPy compatibility**: Follows NumPy API and conventions closely

### Backend
- Do not add new operations, unless instructed to
- Comparison ops return uint8 (because of lack of bool array support in OCaml bigarray)
- Cannot reshape broadcast views (zero strides)

### Performance
- Use `Bigarray.unsafe_get/set` in loops (validate indices outside)
- Batch operations to minimize backend dispatch

## Alcotest Commands

### Running Tests
- Run all tests: `test.exe`
- List available tests: `test.exe list`
- Run specific tests by name regex: `test.exe test <NAME_REGEX>`
- Run specific test cases: `test.exe test <NAME_REGEX> <TESTCASES>`

### Useful Options
- Stop on first failure: `test.exe test --bail`
- Compact output: `test.exe test -c`
- Show test errors: `test.exe test -e`
- Run only quick tests: `test.exe test -q`
- Verbose output: `test.exe test -v`
