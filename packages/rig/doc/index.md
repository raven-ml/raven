# rig

rig runs compiled work on the devices of a machine, such as its GPUs, from
OCaml. A program allocates memory on a device, submits work that uses it,
and waits for the work. rig orders the work of all devices by the memory it
uses.

rig drives Apple GPUs through Metal, NVIDIA GPUs through CUDA or NVIDIA's
kernel driver, and AMD GPUs through Linux's `amdgpu`. On Linux it also drives
AMD and NVIDIA GPUs with no kernel driver, over PCI. Files are devices of
their own, and another machine's devices open through an agent running there.

## Install

rig needs OCaml 5.5 or later. Install the `rig` package and add `rig` to your
executable's `libraries`:

<!-- $MDX skip -->
```sh
$ opam install rig
```

```dune
(executable
 (name main)
 (libraries rig))
```

Each driver is a library of its own, such as `rig.metal`, `rig.cuda` or
`rig.amd`, which a program adds beside `rig`. Some open their devices through
a further library, such as `rig.amd.amdgpu` for AMD GPUs under Linux's
`amdgpu`.

## A first program

`submit` returns the *point* of the work on its device's *timeline*, and
`Point.wait` returns once the work up to that point has completed. The
preamble of [`rig.mli`](../lib/rig.mli) defines timelines, values and points.

This program copies one integer on a [memory device](../lib/rig.mli),
whose memory is the host's and whose work runs before `submit` returns. It
runs on any machine.

```ocaml
# open Rig;;
# let () =
    let device = memory_device "example" in
    let src = Buffer.create device 4 in
    let dst = Buffer.create device 4 in
    Buffer.write src Bigarray.int32 (fun a -> a.{0} <- 42l);
    let work =
      Submission.(make device [| copy ~src:(Buffer src) ~dst:(Buffer dst) |])
    in
    let run = Submission.Run.make () in
    let done_ = submit work ~run ~buffers:[||] in
    Point.wait done_;
    Buffer.read dst Bigarray.int32 (fun a ->
        Printf.printf "copied %ld\n" a.{0});;
copied 42
```

Each call, in order:

- `memory_device "example"` opens the memory device named `"example"`, or
  returns it if it is open already.
- `Buffer.create device 4` allocates 4 bytes of the device's memory.
- `Buffer.write src Bigarray.int32 f` calls `f` on `src`'s bytes as an array
  the host writes, since the host addresses a memory device's memory, once
  the work that used them is done.
- `copy ~src:(Buffer src) ~dst:(Buffer dst)` is one *part* of the work, a
  copy between two buffers. An end may also be a `Slot`, a buffer each
  submit passes ([`Submission.side`](../lib/rig.mli)). rig orders the parts
  of a submission by the memory they use. It runs a copy on the queue of the
  last earlier part the copy follows, where that queue runs copies, and
  otherwise on the device's copy engine, the queue its driver names in the
  `copy_engine` fact ([`Submission.make`](../lib/rig.mli)).
- `Submission.make device [| ... |]` prepares the parts once, as a
  *submission* that any number of submits run.
- `Submission.Run.make ()` makes a *run*: the storage that one submit's
  work reads its parameters from, such as a launch's grid
  ([On a GPU](#on-a-gpu)). A copy has none, but every submit takes a run.
- `submit work ~run ~buffers:[||]` runs the submission once.
  `buffers` lists the buffers of the slots that a launch's parameters and a
  copy's `Slot` ends name, and this copy names none. An optional `~waits`
  lists points of other devices the work must follow.
- `Point.wait done_` returns once the copy has completed. `Buffer.read`
  would wait for it too, since it follows the last write of `dst`.

## On a GPU

On a GPU, the work is usually a *launch*, which runs a function of an *image*,
a binary compiled for the device, over a grid of threads. The calls are the
same on every GPU, and the work may complete after `submit` returns.

This function doubles the `n` floats of the device buffer `x`, `n` a multiple
of 256, with the function `twice` of `binary`:

<!-- $MDX skip -->
```ocaml
let double d binary x n =
  let image = Rig.Image.load d binary |> Result.get_ok in
  let twice =
    Rig.Submission.(launch ~refs:[| write 0 ~at:0 |] image "twice" ~params:8)
  in
  let s = Rig.Submission.make d [| twice |] in
  let run = Rig.Submission.Run.make () in
  let b = Rig.Submission.block s 0 in
  Rig.Submission.Run.groups run b (n / 256) 1 1;
  Rig.Submission.Run.threads run b 256 1 1;
  Rig.Point.wait (Rig.submit s ~run ~buffers:[| x |])
```

The launch's 8 bytes of parameters hold one pointer. A *ref*
([`Submission.ref`](../lib/rig.mli)) makes 8 bytes of the parameters an
offset into one of the submit's buffers, its *slot*. This one,
`write 0 ~at:0`, makes bytes 0 to 7 an offset into `buffers.(0)`, here `x`,
which the work reads and writes, and the driver's
`submit` adds `x`'s address to it as it encodes the work
([Drivers](05-drivers.md#one-submission)). The run never stores the offset,
and bytes a run never stores read 0, so `twice` gets `x`'s address.

A ref states how the work uses its slot: `Submission.read`,
`Submission.write` or `Submission.unordered`. `Submission.make` derives the
slots from the refs, and each submit orders the work after earlier work on
each slot's memory by that use.

A *block* is the part of a run that holds one launch's grid, the threads of
each group, and its parameters ([`Submission.block`](../lib/rig.mli)).
`Submission.block s 0` names the block of part 0, and the two stores set its
grid to `n / 256` groups of 256 threads.

On a Mac, the binary is a metallib, compiled with
`xcrun -sdk macosx metal -c twice.metal -o twice.air` and
`xcrun -sdk macosx metallib twice.air -o twice.metallib` from:

<!-- $MDX skip -->
```c
struct args { device float *x; };
kernel void twice(constant args &a [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) { a.x[i] *= 2; }
```

With CUDA, it can be PTX, which CUDA compiles for the GPU as it loads,
made with `nvcc -ptx -o twice.ptx twice.cu` from:

<!-- $MDX skip -->
```c
extern "C" __global__ void twice(float *x) {
  x[blockIdx.x * blockDim.x + threadIdx.x] *= 2;
}
```

A program opens the GPU through its driver's opener, moves the floats to the
GPU's memory, runs `double`, and copies them back:

<!-- $MDX skip -->
```ocaml
let () =
  let d = Rig.open_ (Rig_metal.gpu 0) |> Result.get_ok in
  (* With CUDA: Rig.open_ (Rig_cuda.gpu 0) and "twice.ptx". *)
  let binary =
    In_channel.with_open_bin "twice.metallib" In_channel.input_all
  in
  let n = 65536 in
  let host = Rig.Buffer.create Rig.host (4 * n) in
  Rig.Buffer.write host Bigarray.float32 (fun a ->
      for i = 0 to n - 1 do a.{i} <- float_of_int i done);
  let x = Rig.Buffer.create d (4 * n) in
  Rig.Buffer.copy ~src:host ~dst:x;
  double d binary x n;
  Rig.Buffer.copy ~src:x ~dst:host;
  Rig.Buffer.read host Bigarray.float32 (fun a ->
      Printf.printf "%g\n" a.{n - 1})
```

`Buffer.copy` orders itself after the work that used its buffers, so the copy
back follows the launch (the Order section of [`rig.mli`](../lib/rig.mli),
and [Ordering](01-ordering.md)). A copy returns once the bytes are in its
destination, and `Buffer.read` follows the last
write of its buffer whatever came before it (the `Buffer` module's Host
access section).
[Memory](03-memory.md#stamps-through-one-program) traces each allocation's
stamps through this program, and through a loss of the GPU.

## Reading order

The interfaces state every rule, and read top to bottom:

1. [Rig](../lib/rig.mli): devices, timelines, the order of work, buffers,
   running work, claims, domains, loss and the return of memory.
2. [Rig_edge](../lib/edge/rig_edge.mli): the contract a driver implements.
3. One driver: [Metal](../lib/metal/rig_metal.mli),
   [CUDA](../lib/cuda/rig_cuda.mli),
   [NVIDIA](../lib/nv/nvidia/rig_nv_nvidia.mli), or
   [AMD through amdgpu](../lib/amd/amdgpu/rig_amd_amdgpu.mli).

These pages show the rules at work, with programs and traces:

1. [Ordering](01-ordering.md): how work on several devices is ordered by
   the memory it uses.
2. [Claims](02-claims.md): a function that writes over its argument.
3. [Memory](03-memory.md): a buffer's life, and its stamps through the GPU
   program above.
4. [Loss](04-loss.md): what a program sees when a device fails.
5. [Drivers](05-drivers.md): one submission from `submit` to the hardware
   and back.
6. [GPUs without a driver](06-gpus-without-a-driver.md): opening an AMD or
   NVIDIA GPU over PCI.
7. [Machines](07-machines.md): devices of other machines.

The numbered examples, from [01-add](../examples/01-add/), teach one idea
each and run on any machine. [x-pci](../examples/x-pci/) needs Linux.
[Hardware notes](hardware-notes.md) describe the hardware behaviour each
driver relies on.
