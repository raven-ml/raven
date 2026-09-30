(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Sharded values.

    A sharded value is an {!Op.Unshard} of a shard: the part of the value that
    one device, or one thread, holds, with a range per sharded axis whose value
    says which part it is ({!Ops.sharding}). The shards of an axis are
    consecutive blocks of equal size. A device range ({!Ops.Axis_type.Device})
    shards a value across the devices of a placement; any other range, such as a
    thread's, shards it within a kernel.

    This module rewrites a graph so that operations compute on shards: each
    {!Op.Unshard} moves towards the graph's outputs, and each device's program
    names only its own part. *)

val multi_pm : (unit, Ops.t) Ops.Pattern_matcher.t
(** [multi_pm] rewrites an operation whose sources are sharded into the
    operation on their shards, sharded as the result is:

    - an arithmetic operation whose sources are sharded alike, or are scalars,
      or have the result's whole shape, computes on each shard, a whole source
      taking the shard's part of itself. A {!Op.Stack} whose sharded sources are
      sharded alike stacks their shards and its other sources. Sources sharded
      differently are resharded on the result's one sharded axis, each taking
      its part of a whole copy;
    - a reduction of sharded axes reduces each shard, then reduces the shards
      across the devices ({!Ops.allreduce}); with {!Helpers.allreduce_cast}, a
      value cast from a half or a bfloat16 crosses the devices in that type. A
      reduction of none of them reduces each shard;
    - movements move the sharded axes. A reshape keeps each sharded axis whole
      and divisible by its shard count; a pad and a flip leave sharded axes
      alone; a shrink keeps a sharded axis whole, or takes exactly the shard of
      its range, which removes that axis's sharding. On a single sharded axis
      across devices, a shrink to one shard is a copy of that shard to every
      device;
    - an {!Op.Index} of a sharded value takes, on each sharded axis, the index
      into the shard the range owns: [r * n + i] for a block of [n] elements, or
      [r + i * n] for every [n]th element from [r];
    - a copy to one device joins the shards along their axes; a copy to several
      devices places each shard at its offset in zeros and sums them across the
      devices;
    - an allreduce of a sharded value reduces its shards across the devices, and
      stays sharded;
    - a store into a sharded destination stores each shard into its own; a store
      of a sharded value into a whole destination stores each shard into its
      part of it;
    - a call to a function without its own compilation ({!Ops.is_inline_call})
      rewrites its body, and passes the shard of each sharded argument; any
      other call and casts, bitcasts, {!Op.Stage}, {!Op.After}, {!Op.Detach} and
      {!Op.Contiguous_backward} pass the shard through.

    It also rewrites copies and shard selections across devices:

    - a copy of a value on one device to several is a copy to each
      ({!Ops.mstack}), or the value itself on each if it has no device once
      simplified; a copy of a value on several devices to one is a copy of its
      first shard, or that shard if it is already there;
    - a shard selection of an {!Op.Mstack} is its source; one of a movement or
      of an arithmetic operation selects the shards of their sources, and the
      movement's arguments take the device's position for the device range;
    - a shrink of an {!Op.Mstack} shrinks each of its sources, substituting in
      each the device's position for the device range, and materialises them.

    With [LATE_ALLREDUCE] set to [0] in the environment when the library is
    initialised, each {!Op.Allreduce} of a value that is not sharded is also
    replaced by its value ({!Allreduce.handle_allreduce}).

    Raises [Invalid_argument] on what has no shard of its own: a reshape that
    moves elements between shards, a pad, flip or shrink of a sharded axis other
    than those above, a reduction of some sharded axes but not all, an index
    that crosses shards, or values on different devices resharded together. *)
