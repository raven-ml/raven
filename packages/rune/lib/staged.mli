(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** rune's constructs in a compiled call's trace.

    A trace lowers every operation of its extent ({!Lower.op}) and answers
    rune's constructs ({!Construct.t}) at their call ({!Construct.here}): a
    function a construct carries runs under a trace of its own, a loop's step
    under a trace of its body, and the loop itself is lowered at the call.

    {b Staged loops.} A loop stages as one loop of the program: its step runs
    once, traced as the body of one call. A loop over rows, a scan, runs the
    call once per row, on a range of as many trips. A loop until a stop runs it
    while a flag holds, at most [max] trips: the stop at the initial carry is
    computed before the loop, and each trip computes the stop at the carry it
    stores, keeps the stop's value and stores the flag that it does not hold
    yet, which the engine reads before the next trip ({!Tolk.Ops.backedge}).
    Once the trips are done, a stop that still fails raises the loop's failure,
    as {!Nx.check} does, when the compiled call returns.

    The body's parameters stand for one trip's carry and rows; every part of the
    step's graph that does not depend on them is computed once, before the loop,
    and passed to the call. On trip [i], the step reads row [r] of each stacked
    input and writes row [r] of each stacked output, [r] being [i], or
    [n - 1 - i] for a reversed scan of [n] steps; a loop until a stop has [max]
    output rows, of which those past its last trip hold no value of the loop.
    Rows are 16 bytes of memory apart, as the body's vector accesses require: a
    stacked input whose rows are not is read from a padded copy, made once per
    call. Each carry is one buffer that each trip updates in place, once every
    kernel that reads it ran, as the schedule orders them; a next carry that
    reads the carry elsewhere than at its own index is computed into storage of
    its own first, and so is, each trip, the latest of carries whose next values
    read each other in a cycle, such as two carries that swap.

    A loop inside the step stages as a loop of the body, nested in the outer
    one: its step is traced once, and its values are those of the loop written
    out, each trip's carry stored before the next reads it. The engine runs an
    outer loop around a loop until a stop trip by trip.

    A loop stages when its leaves lie on one device, host leaves joining it, and
    its loop runs ({!Tolk.Hcq2.runs}): on a device whose work runs from command
    queues (Metal, CUDA, AMD, NV), as one batch of the body's calls, or one
    batch per trip for a loop until a stop and a loop around one; on the host,
    or a device without queues, its calls once per trip. A scan is otherwise
    declined ({!Trips.Not_staged}), and folds where it is written; a loop until
    a stop, which has no written-out form, raises {!Lower.Jit_error}:
    - before its step runs, when its leaves lie on several devices;
    - after its step ran once, in a trace whose values nothing keeps, when the
      step's next carry differs from its carry in a shape or a placement, when
      an output lies elsewhere than the loop's device or the host, when the step
      draws from a key no parameter of the body varies, a traced draw that would
      repeat on every trip, or when the body's calls run on a device with queues
      and on the host, or on devices of two kinds, which no loop runs. The step
      draws from a scope of its own, rooted at a constant, so that a draw from
      the scope around the loop is written out, each trip drawing where the loop
      is written.

    The step and the stop run at the loop's call, inside the handlers around it.
    Outside every body, an operation that reads a value a step computed raises
    [Invalid_argument]: such a value reaches the trace around the loop only
    through a handler, which runs outside the step.

    An output the step computes on the host, such as a constant, is written on
    the loop's device, and its rows are copied to the host once per call.

    So a staged loop runs its step once per trace, and a declined scan once per
    row, plus once before when it declined after its step ran. *)

(** The type for how a node reads a value: not at all, at each element's own
    index only (through elementwise operations, casts and bitcasts that keep the
    width of their elements, reshapes and contiguous markers), or otherwise. *)
type reach = Apart | Own | Other

val reach : from:Tolk.Ops.t -> Tolk.Ops.t -> reach
(** [reach ~from u] is how [u] reads [from]. The partial application
    [reach ~from] walks each node once. *)

val install : Lower.scope -> (unit -> 'a) -> 'a
(** [install s f] is [f ()] traced in [s]: every operation of its extent lowered
    by [Lower.op s], and the constructs it performs answered as follows.
    - [Loop r] is staged, or declined as above. An exception of the step
      propagates unchanged. A loop until a stop of no trip checks its stop at
      its initial carry.
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
