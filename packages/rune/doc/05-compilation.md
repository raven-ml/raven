# Compilation

`Rune.jit` traces a function once, compiles the traced computation into fused kernels for a device, and replays them on later calls. This page covers what a compiled call does with its arguments, where it runs and keeps memory, how compiled programs are cached across processes, and how to tune and debug them. `Rune.jit`'s documentation in `rune.mli` states the contract; this page explains how to use it.

## Signatures

A compiled function is described by a signature with the shape of the function itself. Each argument is read (`@->`) or consumed (`consumes ... @@`), and `returns` gives the result's structure:

<!-- $MDX skip -->
```ocaml
let caches = Nx.Ptree.list (Nx.Ptree.instantiate (module Kaun.Attention.Cache))

let step =
  Rune.jit
    Nx.Ptree.(
      tensor @-> Kaun.Cache_index.ptree @-> consumes caches
      @@ returns (pair tensor caches))
    (fun tokens index caches -> decode params tokens index caches)
```

`step` has the type of the function it compiles. `params` is captured: it is a constant of the compiled function, bound once and read by every program. The caches are given up by each call, which may write its result over their storage, and every result is a value of its own, so the sampled tokens can be read at any later time.

A single tensor keeps its shorthand: `Rune.jit' f` is `Rune.jit Nx.Ptree.(tensor @-> returns tensor) f`.

Apply `jit` once and reuse the function it returns: the compiled programs live in that partial application. A call whose arguments have another key (another dtype or shape, another list length, another reported window) traces and compiles a program of its own, and later calls with that key replay it.

## Consumption

Before its first kernel, a call marks the storage of every consumed argument as consumed. Reading a value over that storage afterwards raises `Invalid_argument` with the path of the leaf that consumed it:

<!-- $MDX skip -->
```ocaml
let ids, caches' = step tokens index caches in
Nx.to_array (List.hd caches).keys
(* Invalid_argument "this value was consumed at 2.0.keys in a compiled call's
   arguments; use the value the call returned" *)
```

A consumed leaf must hold its storage alone. A slice, a transpose or a broadcast of a larger tensor raises before the call runs (pass `Nx.copy` of it), and so does a storage that two leaves of the call, or a leaf and a capture, reach. A host value has no storage to consume: it is uploaded, stays usable, and lends nothing.

A result takes the storage of a consumed leaf when writing it there cannot change what the program reads: an optimizer update, or a cache written at a few slots. A loop that consumes its state therefore holds one generation of it on the device.

## Devices and Memory

A call runs where its placed arguments and captures are (`Nx.place`, `Nx.placement`), and on `Rune.default_device ()` when none is placed. `~devices` names the devices instead: `[ Rune.device "METAL" ]`, or several devices of one backend. Host arguments are copied to the device on every call; a placed argument, such as the output of an earlier call, seeds the program with no transfer. Placing a model's weights once, as the kaun examples' importers do, means no compiled function uploads them.

Values placed on several devices run a program over those devices. A value split along an axis (`Nx.Placement.sharded ~axis`) is one slice on each device, and a copy (`Nx.Placement.replicated`) or a host argument is the whole value on each. The function sees whole values: an elementwise operation keeps its operands' split, and a reduction over a split axis becomes an allreduce, so the gradient of a loss over a batch split across devices is summed across them. Operands split differently raise as the function traces, as they do eagerly; `Nx.place` inside the function gathers a value to a copy on each device or splits one. A per-device computation is `vmap` over an axis split one slice per device.

Outputs are values on the device: shape and dtype never transfer, and a read copies the elements it reads and leaves the output where it is. An nx operation on a placed value outside a compiled function computes on the host and places its result. A view of part of a storage is read in place: the program reads the storage the view reaches and applies a strided view's layout itself. Views that differ only by an offset that is a multiple of 16 bytes share a program, and a C-order window at such an offset shares the program of a value that covers its storage. Only views whose windows overlap (`Nx.sliding_window`) are copied.

Device memory that backs an output is held until the output is garbage-collected or consumed. Past a budget of device allocations since the last major collection, 4 GiB by default and set in bytes by `RUNE_JIT_RESIDENT_BUDGET`, a collection runs before allocating more; an allocation that still fails raises `Nx.Device.Out_of_memory` before the call consumes anything. The intermediate values of a call live in scratch memory shared by every compiled function on the device, sized to the largest any of them needs, so the blocks of a deep model called in turn do not each hold their own.

## Numerics

A compiled program performs the operations the function performs. A sum over an axis (`Nx.sum`, `Nx.mean`, the contraction of `Nx.matmul`) is the sum of its terms in an unspecified association. The compiler may add the terms in another order than eager, split them across threads, and move a factor that does not vary along the summed axis out of the sum (`sum (0.125 * a * b)` becomes `0.125 * sum (a * b)`). Results then differ from eager's in rounding, and at overflow in whether a term overflows. A maximum over an axis is exact, except which zero it returns when -0 and +0 tie.

Beyond that, compiled float results can differ from eager's in the last bits where the kernel compiler fuses a multiply and an add, where a division by a constant becomes a multiplication by its rounded reciprocal, and in transcendental functions, which are approximations within a few units in the last place (`Nx.pow` about 70); Metal flushes float32 subnormals to zero, a `float16` program on the CPU is not rounded after each operation, and signed integer overflow is undefined in the generated C.

## The Persistent Cache

The first compilation of a trace writes its scheduled and compiled kernels to a disk cache under `$XDG_CACHE_HOME/tolk/rune_jit` (`XDG_CACHE_HOME` defaults to `~/.cache` on Linux and `~/Library/Caches` on macOS). A later process compiling the same trace loads them, so tracing is most of a warm start. Entries are invalidated when the executable, the device, its compiler or the code generation options change. `JITCACHE=0` disables the cache; programs over several devices are never persisted. Results are identical either way.

## Beam Search

By default each kernel is scheduled by fixed heuristics. `~beam:n` (or the `BEAM` environment variable) searches schedules instead: each round compiles and times candidates on the device and keeps the `n` best. Compilation is much slower and the kernels are usually faster. The tuned result lands in the persistent cache, so the search runs once per trace. `~parallel:k` (or `PARALLEL`) bounds domains compiling independent kernels and search candidates. The default uses available CPUs; `0` compiles sequentially. Candidates are still timed one at a time.

The first compilation using a positive `PARALLEL` value fixes the shared worker
limit for the process. Later positive values reuse that limit; `~parallel:0`
compiles in the caller. Candidate timing remains sequential.

## Debugging

`RUNE_JIT_DEBUG=1` prints, for each compiled function:

- each retrace, with the first difference from the previous call's key: `rune.jit: retrace: 1.window: int 3 here, int 2 in the previous key`;
- each replay, with the bytes it moved to and from the device;
- what became of each consumed leaf: `rune.jit: 2.0.keys -> result 1.0.keys reused`, or `consumed, storage released`.

`Rune.jit_stats` returns the same transfer counters for programmatic checks. A leaf is named by its path: the argument's position counted from 0, then its path inside that argument.
