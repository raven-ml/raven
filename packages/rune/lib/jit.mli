(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Compiled calls.

    A compiled function keeps a program per {e key}: its arguments' visits
    ({!Nx.Ptree.visits}); for each leaf, its dtype, shape and placement, its
    strides but those of axes of one element (a view whose only strides out of C
    order are on such axes shares the C-order program), and the run of storage
    its parameter binds: where the run starts below the view's first element
    ({!Lower.span}) and where it starts within 16 bytes of memory
    ({!Lower.phase}), on every device, the host included; and the settings of
    tolk that a caller may change around a call ({!Tolk.Helpers.context}) and
    that change the program: [BEAM], [NOOPT] and whether batches are profiled
    ([DEBUG] at 2 or more). tolk's other settings are read when a key is first
    compiled, and overriding them around a compiled call is not supported.

    A key being compiled makes the other calls with that key wait. A compile
    that raises installs nothing: its waiters, and later calls, trace again. A
    call that needs storage another call holds raises [Invalid_argument] and
    never waits.

    The first call with a key traces the function ({!Staged.install}): each leaf
    is a parameter over its storage ({!Lower.param}), and each result leaf is
    stored into a parameter of its own, or into the storage of a consumed leaf
    it is lent. The stores are scheduled, compiled for the trace's devices and
    linked once, with the captures the trace bound; the call then runs as every
    later call with that key does:
    + it borrows the storage of every leaf and takes the consumed ones
      exclusively ({!Nx.Repr.Storage}), refusing a consumed leaf that does not
      cover its storage or whose storage another leaf or a capture reaches;
    + it allocates each result's storage, and copies each consumed storage that
      cannot lend on this call: borrowed memory, memory that does not span its
      buffer, or storage another program binds;
    + it consumes the consumed leaves, binds every parameter and queues the
      program ({!Tolk_engine.run});
    + it wraps each result's storage as a value at the result's placement.

    {b Lending.} A result takes the storage of a consumed leaf of equal dtype,
    size and placement whose storage starts on 16 bytes of memory, when writing
    the result there cannot change what the program still reads of the leaf: the
    result reads the leaf only at each element's own index (through elementwise
    operations, casts and bitcasts that keep the width of their elements,
    reshapes and contiguous markers), or does not read it, which the schedule
    then orders after every read of it. Pairs are chosen once per program, in
    three passes: the results of an indexed write ([Nx.Op.Update],
    [Nx.Op.Scatter]) that read the leaf at their own index; then the other
    results that do; then, in the order the function computed them, the results
    that do not read it, each taking the first free leaf in walk order. Each
    storage lends at most once. A consumed leaf that a result returns unchanged,
    in C order over its storage, is lent with no store: the result's storage is
    the leaf's memory.

    A program keeps the storage it binds as captures. Host storage records no
    such binding: a host capture that another call consumes is dead for the
    program too, whose next run raises [Invalid_argument] naming the path at
    which it was consumed.

    {b Reports.} With [RUNE_JIT_DEBUG=1] in the environment when the module is
    initialised, a call prints on standard error each retrace, with the first
    difference between its key and the previous call's, and each consumed leaf's
    fate, as in ["rune.jit: 2.0.keys -> result 1.0.keys reused"].

    While a profile is taken ({!Nx_device.Profile}), a first call's phases are
    host spans: ["rune.jit: trace"], ["rune.jit: schedule"],
    ["rune.jit: compile"] and ["rune.jit: link"]. *)

val jit :
  ?beam:int ->
  ?parallel:int ->
  string ->
  ('a -> 'b) Nx.Ptree.fn ->
  ('a -> 'b) ->
  'a ->
  'b
(** [jit ~beam ~parallel entry s f] is [f] compiled, as {!Rune.jit} documents
    it, for the entry point named [entry], which its messages start with. Each
    call performs {!Construct.Compiled}: a transformation around it passes on
    the call of the function it derives, whose programs the compiler that
    [derive] gives keeps, and with none the call runs the program of its key. A
    derived function consumes nothing.

    Raises [Invalid_argument] when applied to [s] if [s] has no argument; and at
    a call, before any work, for a consumed leaf that does not cover its whole
    storage or whose storage another leaf or a capture of its program reaches,
    naming both paths, for storage another call holds, and as the trace raises
    ({!Lower.op}, {!Staged.install}). *)
