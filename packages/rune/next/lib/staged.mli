(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** rune's constructs in a compiled call's trace.

    A trace lowers every operation of its extent ({!Lower.op}) and answers
    rune's constructs ({!Construct.t}) outside that interception, so that the
    operations an answer issues are lowered in a scope of its choosing.

    {b Staged scans.} A scan stages as one loop of the program: its step runs
    once, traced as the body of one call that a range of as many trips as the
    scan has steps runs. The body's parameters stand for one trip's carry and
    rows; every part of the step's graph that does not depend on them is
    computed once, before the loop, and passed to the call. On trip [i], the
    step reads row [r] of each stacked input and writes row [r] of each stacked
    output, [r] being [i], or [n - 1 - i] for a reversed scan of [n] steps. Rows
    are 16 bytes of memory apart, as the body's vector accesses require: a
    stacked input whose rows are not is read from a padded copy, made once per
    call. Each carry is one buffer that each trip updates in place, once every
    kernel that reads it ran, as the schedule orders them; a next carry that
    reads the carry elsewhere than at its own index is computed into storage of
    its own first, and so is, each trip, the latest of carries whose next values
    read each other in a cycle, such as two carries that swap. A scan inside the
    step is written out inside the body.

    A scan stages when its leaves lie on one device whose work runs from command
    queues (Metal, CUDA, AMD, NV), host leaves joining it, and when its body
    runs there as one loop of one batch ({!Tolk_next.Hcq2.stages}). It is
    declined ({!Scan.Not_staged}), and folds where it is written:
    - before its step runs, when no leaf lies on such a device, as on the host;
    - after its step ran once, in a trace whose values nothing keeps, when the
      step's next carry differs from its carry in a shape or a placement, when
      an output lies elsewhere than the loop, when the step draws from a key the
      body does not vary (a key scope's draw: the step runs outside the scopes
      the function opened, and a traced draw would repeat on every trip), when
      the scan has one step, or when the body runs a host program, a copy the
      host makes, or calls devices of two kinds.

    So a staged scan runs its step once per trace, and a declined one once per
    row, plus once before on a device with queues. *)

(** The type for how a node reads a value: not at all, at each element's own
    index only (through elementwise operations, casts and bitcasts that keep the
    width of their elements, reshapes and contiguous markers), or otherwise. *)
type reach = Apart | Own | Other

val reach : from:Tolk_next.Ops.t -> Tolk_next.Ops.t -> reach
(** [reach ~from u] is how [u] reads [from]. The partial application
    [reach ~from] walks each node once. *)

val install : Lower.scope -> (unit -> 'a) -> 'a
(** [install s f] is [f ()] traced in [s]: every operation of its extent lowered
    by [Lower.op s], and the constructs it performs answered as follows.
    - [Scan r] is staged, or declined as above. An exception of the step
      propagates unchanged.
    - [Remat { recomputed = true; f; args; _ }] is [f args], each argument that
      the trace computes materialised: its storage is what the backward pass
      reads again.
    - [Barrier { values; after }] is [values], each one that the trace computes
      read through a copy of it stored once every value of [after] exists, so
      that the backward pass recomputes from it with nodes of its own. Storage
      the trace reads, constants, and values of other transformations are read
      as they are.
    - [Detach x] is [x].

    Inside a staged body, a remat and a barrier pass outward: the backward loop
    of a staged scan recomputes each step already. Every other construct passes
    outward.

    Raises as [Lower.op] does. *)
