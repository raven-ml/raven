# nx

n-dimensional arrays as values on device sets: a dtype, a shape and a
placement over the set the value's brand names, and operations that the
set's kernel library computes where their operands lie. Everything else in
raven exchanges `Nx.t` values and `Nx.Ptree` structures.

## Structure

- `nx` is `nx.mli` over private modules. `nx.array` holds dtypes, layouts
  and arrays over rig buffers, with the door of `nx_array.h` through which
  C kernels claim arrays. `nx.kernel` holds `Nx_kernel.S`, the contract a
  kernel library implements, its programs (`Prog`) and `nx_kinds.h`.
- `nx.cpu` is the only kernel library. It computes on memory the host
  addresses. There is no eager GPU kernel library: a GPU set computes
  under `Rune.jit`.
- `nx.quant`, `nx.ragged` and `nx.wide` build on `Nx` and `nx.array`
  alone, `Nx.Prim` included, with no C and no kernels. `nx.io` reads files
  into host values and writes values of any set.

## Boundaries

- nx defines no derivative and no compiler. `Nx.Prim`'s interpretations
  give operations other meanings; rune answers nx through them from
  outside. nx's docs name rune's functions only as examples.
- An operation computes where its operands lie and moves none of them.
  `Nx.place` moves a value; a mismatch raises naming the `place` to write.
  A fold over a cut axis moves only its partials.
- A set's kernels are fixed when `Nx.devices` mints it: nx.cpu for
  host-run devices, else none unless `~kernels` names a library. On a set
  without kernels a creation is made on the host and placed; every other
  operation that makes elements raises. After a kernel declines, nx
  raises. Nothing computes on the host in a set's place.
- rig holds bytes; a kernel library computes. A dependency edge that adds
  link weight to every user of a device blocks a design.
- A kernel library's `.mli` states what it computes on, the cases it
  declines, the associations, algorithms and bounds the contract leaves
  to it, and whether domains may call it concurrently, then includes
  `Nx_kernel.S`. Nothing is added to `Nx_kernel.S` without being asked.
- tolk is the only compiler. Kernel libraries never generate code at run
  time, and implement operation families: no kernel for a named formula
  (RoPE, AdamW, SwiGLU, a norm's tail).

## Semantics

- Floats narrower than `float32` compute in `float32` and round once;
  8- and 16-bit integers compute at 32 bits and wrap. Floats keep IEEE NaN
  and infinities.
- nx.cpu computes each kind by its function in `nx_kinds.h`: one
  polynomial, fma where written, `-ffp-contract=off`, no `-ffast-math`,
  subnormals kept, so its bits, NaNs aside, match on every target. The
  coefficients come from `test/kernel/gen/kinds.py`.
- On nx.cpu, layout and thread count never change a bit of a result.
- One Threefry generator, `Nx.Rng`. Its key is an `int32` array, so it can
  be a structure's leaf, a jit input or a vmap axis.
- A value made from numbers is a host value; `_at` and `_like` forms make
  it at a placement. Code for every set makes its constants at `Nx.beside`
  of its operands, under a locally abstract set type.
- `Nx.donate` lets one consumer write over the donor's memory, only where
  it reads that memory at the result's own index and the memory is
  C-contiguous, unshared, and of the result's shape and dtype. The result
  is the same either way.
- A fix that changes results or random draws takes the correct version.
  The commit body states the change.
- No public `Nx` renames. Departing from a name or behaviour array users
  expect needs a clear win. Linear algebra and FFT live at `Nx`'s top
  level.

## Kernels and performance

- Contraction runs at vendor parity (ratio at most 1.02) in every kernel
  library; every other kernel within 1.5x of the vendor. Vendor libraries
  are measured outside the repo, never linked.
- Close a gap from a permissive reference (BLIS, libxsmm, SLEEF, CUTLASS,
  CK), writing our own code. Check its licence first; never GPL. Never read
  the instruction stream of a proprietary library. Code that follows
  LAPACK, ReLAPACK or pocketfft keeps its `LICENSE-*` notice.
- On x86-64 every nx.cpu unit but `target.c` is built for x86-64-v3.
  Code that runs before `target.c` checks the host stays there.
- A kernel's external is never `[@@noalloc]`: the door may run OCaml.
- Benches are `bench/array`, `bench/kernel` and `bench/nx`. A baseline
  comes from thumper's bless proposal, never by hand.

## Tests

- `test/test_*.ml` for `nx`, `test/<library>/` for each sub-library.
  Generators under `gen/` run through `uv run` and make the goldens.
- `test/kernel/` runs its laws on the backends of `Nx_kernels_support`.
  Contraction and linear algebra, which differ by target table, run on
  every table the host runs (`cpus`), base first.
