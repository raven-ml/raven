# Compilation

`Rune.jit` traces a function once, compiles the traced computation into fused kernels for the devices its values are on, and replays them on later calls. This page covers what a compiled call does with its arguments, where it runs and keeps memory, how compiled programs are cached across processes, how a GPU computes eagerly, and how to tune and debug them. `Rune.jit`'s documentation in `rune.mli` states the contract; this page explains how to use it.

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

A consumed leaf must cover its whole storage. A slice, a transpose or a broadcast of a larger tensor raises before the call runs (pass `Nx.copy` of it), and so does a storage that two leaves of the call, or a leaf and a capture, reach.

A result takes the storage of a consumed leaf when writing it there cannot change what the program reads: an optimizer update, or a cache written at a few slots. A loop that consumes its state therefore holds one generation of it on the device.

## Devices and Memory

A call compiles for the memories of its placed arguments and captures, and for the host's when there are none: the backends their devices carry take no part, and the results land on the arguments' devices. A GPU's device comes from its vendor's library (`Nx_metal.device 0` from nx.metal, `Nx_cuda.device 1` from nx.cuda), and a value is put on it with `Nx.place`:

<!-- $MDX skip -->
```ocaml
let metal = Nx.Placement.on (Nx_metal.device 0)
let step = Rune.jit' (fun x -> Nx.tanh (Nx.matmul x x))
let y = step (Nx.place metal (Nx.rand Nx.float32 [| 64; 64 |]))
(* compiled for Metal *)
```

A host argument is uploaded on each call, as an eager operation would move it, and a placed argument, such as the output of an earlier call, is read where it is. A capture is bound once per compiled function, at the trace that meets it: a value placed where the program computes is read in place, and any other value is placed there once. Placing a model's weights once, as the kaun examples' importers do, means no compiled function uploads them.

Values placed on several devices run a program over those devices. A value split along an axis (`Nx.Placement.sharded ~axis`) is one slice on each device, and a copy (`Nx.Placement.replicated`) or a host argument is the whole value on each. The function sees whole values: an elementwise operation keeps its operands' split, and a reduction over a split axis becomes an allreduce, so the gradient of a loss over a batch split across devices is summed across them. Operands split differently raise as the function traces, as they do eagerly; `Nx.place` inside the function gathers a value to a copy on each device or splits one. A per-device computation is `vmap` over an axis split one slice per device.

Outputs are values on the device: shape and dtype never transfer, and a read copies the elements it reads and leaves the output where it is. A view of part of a storage is read in place, its strides expressed in the program, so a window of a cache costs no copy.

Device memory that backs an output is held until the output is garbage-collected or consumed. An allocation that fails raises `Nx_device.Out_of_memory` before the call consumes anything.

## Numerics

A compiled program performs the operations the function performs. A sum over an axis (`Nx.sum`, `Nx.mean`, the contraction of `Nx.matmul`) is the sum of its terms in an unspecified association. The compiler may add the terms in another order than eager, split them across threads, and move a factor that does not vary along the summed axis out of the sum (`sum (0.125 * a * b)` becomes `0.125 * sum (a * b)`). Results then differ from eager's in rounding, and at overflow in whether a term overflows. A maximum over an axis is exact.

Beyond that, compiled float results can differ from eager's in the last bits where the compiler fuses a multiply and an add or turns a division by a constant into a multiplication, and in transcendental functions, which are approximations within a few units in the last place; Metal flushes float32 subnormals to zero. A failed factorisation gives non-finite values where eager raises `Nx_backend.Linalg_error`.

## The Persistent Cache

rune keeps no cache of its own across processes. tolk's caches are the persistent cache: compiled binaries under `CCACHE` and schedules under `SCACHE=2`, keyed by the scheduled program. A warm start still traces the function, which produces the key, and then loads what follows.

## Beam Search

By default each kernel is scheduled by fixed heuristics. `jit ~beam:2` searches each kernel's schedules instead: each round compiles and times candidates on the device and keeps the `beam` fastest. The first call of a key compiles much longer and the kernels usually run faster. `~parallel` sets how many domains compile the candidates. Without these arguments, the `BEAM` and `PARALLEL` settings decide for every compiled function; an explicit `~beam`, `0` included, overrides them. The width is part of a call's key, so functions searched at different widths never share a program.

```ocaml
let square = Rune.jit' ~beam:2 ~parallel:8 (fun x -> Nx.mul x x)
```

## Transformations of a Compiled Function

`grad`, `jvp` and `vmap` of a compiled function compile, and so does a `Total.collect` around one. Each runs programs compiled for the function the transformation derives, kept in the compiled function by key as its own programs are, so a `grad (jit f)` in a loop compiles once. Under `grad`, the call splits at its residuals, the values `f` computes that the backward pass reads: the forward pass is a program that returns `f`'s results and the residuals, and the backward pass a program that reads the residuals and the cotangents and never runs `f`. `jit (grad f)` compiles both passes as one program, which keeps no residual between them, so it stays the fast form.

A compiled function that reads, through its closure, a value a transformation around it tracks raises `Invalid_argument`: pass the value as an argument.

## Eager Computation on a GPU

Outside a compiled function, an operation computes eagerly with the backend of its operands' devices: nx.cpu on the host and on test devices. A GPU computes eagerly with nothing until a backend is paired with it (`Nx.Device.with_backend`), so an eager operation on a GPU value raises `Invalid_argument` before any work, naming the remedies: compile the function with `Rune.jit`, pair the device with a backend, or `Nx.place` the value on the host. Constants, views, reads and `Nx.place` work on every device:

<!-- $MDX skip -->
```ocaml
let gpu = Nx.Placement.on (Nx_metal.device 0)
let x = Nx.place gpu (Nx.rand Nx.float32 [| 1024; 1024 |])
let row = Nx.slice [ I 0 ] x                 (* a view: works *)
let zeros = Nx.zeros_like x                  (* a constant: works *)
let y = Rune.jit' (fun x -> Nx.tanh (Nx.matmul x x)) x  (* compiled *)
let z = Nx.sum x                             (* raises *)
```

## Debugging

`RUNE_JIT_DEBUG=1` prints, for each compiled function:

- each retrace, with the first difference from the previous call's key: `rune.jit: retrace: 1.window: int 3 here, int 2 in the previous key`;
- each call's storage reuse: `rune.jit: 2.0.keys -> result 1.0.keys reused`, with the views a call copies and the storage it copies because it cannot lend.

A leaf is named by its path: the argument's position counted from 0, then its path inside that argument. The runtime counts what compiled code allocates and moves: `Nx_device.stats d` before and after, compared with `Nx_device.Stats.diff`, gives the bytes allocated on `d` and copied into and out of it. A profile taken with `Nx_device.Profile.start` holds one span per kernel on the device's clock, and host spans for a first call's phases (`rune.jit: trace`, `schedule`, `compile`, `link`); `DEBUG=2` prints one line per kernel.
