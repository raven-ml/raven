# Ordering

This page shows, with programs, how rig orders the work of several devices by
the memory it uses. The rules themselves are in [`rig.mli`](../lib/rig.mli):
values and points in its preamble, commits in its Timelines section, the six
rules and the stamps in its Order section, and what holds between domains in
its Domains section.

## The idea

Each allocation of memory records the point of the last work that wrote it
and, for each device, the point of the last work that used it. These are its
*stamps*. Each submit reads the stamps of the memory its work uses, makes the
work follow the points they name, and adds its own point to them. Two pieces
of work need an order only when they share memory, so the order lives with
the memory, and a program names no events or fences. A library passes a
point to `Rig.submit`'s `waits` only to order work after other work that
shares no memory with it.

A program that reaches memory only through rig's calls, from one domain, and
touches a buffer's bytes only through `Buffer.read` and `Buffer.write`, needs
nothing else from this page. The rest shows why, and where that stops.

## Devices and timelines

The pages write a point as `A:1`, for value 1 of device *A*. The preamble of
`rig.mli` defines timelines, values and points, and its Timelines section says
when a driver commits the work it holds.

The examples below use [memory devices](../lib/rig.mli) (`Rig.memory_device`),
whose memory is the host's and whose work runs before `Rig.submit` returns.
They number their work as a GPU does, so the examples run on any machine and
show the same stamps a GPU's work would leave.

## Correct ordering

Two memory devices stand for two GPUs here. The helpers make a copy part and
submit it:

```ocaml
open Rig

let copy src dst = Submission.(copy ~src:(Buffer src) ~dst:(Buffer dst))

let run = Submission.Run.make ()

let submit_parts d parts =
  submit (Submission.make d parts) ~run ~buffers:[||]
```

```ocaml
# let () =
    let a = memory_device "A" in
    let b = memory_device "B" in
    let src = Buffer.of_string "hello, world" in
    let x = Buffer.create a 12 and y = Buffer.create b 12 in
    ignore (submit_parts a [| copy (Option.get (Buffer.borrow a src)) x |]);
    ignore (submit_parts b [| copy (Option.get (Buffer.borrow b x)) y |]);
    let out = Bytes.create 12 in
    Buffer.blit_to_bytes y 0 out 0 12;
    print_endline (Bytes.to_string out);;
hello, world
```

This program prints `hello, world` on any devices that run these copies.
*A*'s copy writes `x` at `A:1`. *B*'s copy reads `x` through a borrow, a
buffer on *B* over `x`'s allocation, so by the Order section's rule 3 it
follows `A:1`. The host's read of `y` follows `y`'s last write, `B:1`, by
rule 5, and so, since follows is transitive, `A:1`, though the program names
no point. Memory devices borrow any host memory, so `Buffer.borrow` returns
`Some` here.

## Incorrect ordering

An array that `Buffer.export` made follows only the work done when it was
made:

```ocaml
let read_too_soon d src =
  let x = Buffer.create host 4 in
  let a = Buffer.export x Bigarray.int32 in
  ignore (submit_parts d [| copy src (Option.get (Buffer.borrow d x)) |]);
  a.{0}
```

On a memory device this returns the copied integer, because the copy ran
before `submit` returned. On a GPU, nothing guarantees that the read sees the
copy, which came after the export. `Buffer.read x Bigarray.int32 (fun a ->
a.{0})` in its place is correct: it follows `x`'s last write.

Each `Buffer.of_bigarray` call makes an allocation with stamps of its own,
even over bytes another call wraps:

```ocaml
let twice ba = (Buffer.of_bigarray ba, Buffer.of_bigarray ba)
```

Work through one of the two buffers follows no work through the other, beyond
device order. Wrap an array once and share the buffer through views and
borrows.

Polling memory that work writes through an exported array is incorrect:

```ocaml
let rec spin flag = if flag.{0} = 0l then spin flag
```

As before, the read follows nothing, so seeing the flag set does not mean the
work's other writes are visible. Nothing guarantees either that the loop
ends. The driver may hold the work that sets the flag until it is committed,
and the loop commits nothing. Neither does a loop on `Rig.signaled`. A wait
commits (the Timelines section of [`rig.mli`](../lib/rig.mli)).

## Several domains

The stamps order calls that the program has ordered. A wait orders the host
after the work submitted before it returns, and after nothing that another
domain submits later. If one domain waits for a buffer and reads it while
another submits work that writes it, nothing orders the read and that work.
Two domains that concurrently submit work on two devices over one
allocation, where either work writes it, are not ordered against each other
either. The program orders them, such as by starting the second submit after
the first has returned. The Order section states this, and the Domains
section lists the few waits that hold a lock of rig's.
