# CLAUDE.md

## Project Overview

Raven: OCaml ML ecosystem. Pre-alpha.

**Libraries:** nx (tensors), hugin (plotting), quill (notebooks), rune (autodiff+JIT), sowilo (vision)

## Workflow

### Task Management

1. **Start**: Read `TODO.md` for current priorities
2. **Plan**: Complex tasks → write plan in `.claude/plans/YYYY-MM-DD-description.md`  
3. **Update**: Keep TODO.md current

### Testing & Debugging
- Debug with backtrace: `OCAMLRUNPARAM=b dune exec nx/test/test_foo.exe 2>&1`
- Float comparisons need `~eps`: `check_t ~eps:1e-6 "name" shape values result`

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
dune build @nx/runtest            # Run specific tests
dune exec nx/test/test_foo.exe    # Run single test file
dune build fmt                    # Format code
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
- **Slicing**: `R [1; 4]` = indices 1,2,3 (exclusive end like Python)
- **Single index**: `get [0]` on `[2; 3]` → shape `[3]` (dimension squeezed)
- **Creation funcs**: Return contiguous arrays (use `init` not `broadcast_to`)
- **NumPy compatibility**: Follows NumPy API and conventions closely

### Backend (Tinygrad UOps)
- Do not add new operations, unless instructed to
- Comparison ops return uint8 (because of lack of bool array support in OCaml bigarray)
- Cannot reshape broadcast views (zero strides)

### Performance
- Use `Bigarray.unsafe_get/set` in loops (validate indices outside)
- Batch operations to minimize backend dispatch
