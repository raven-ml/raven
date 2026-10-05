(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Reductions across devices.

    An {!Op.Allreduce} combines, element by element, the shards that a value
    holds on each of its devices, and places the result on a device or on
    several. This module expresses it with what devices can run: copies between
    devices ({!Ops.copy_to_device}), elementwise operations and multi-device
    values ({!Ops.mstack}). *)

val handle_allreduce : Ops.t -> Ops.t option
(** [handle_allreduce red] is the value of the allreduce [red] of a source [x]
    across [n] devices with the operation [op], built from copies and [op], or
    [None] if [x] is not on several devices. [x] is padded to its greatest shape
    ({!Ops.max_shape}) and made contiguous first. The algorithm is the first
    that applies:

    - {e hierarchical}, when [x]'s shape is known and
      {!Setting.allreduce_node_ndevs} is some [h > 0] dividing [n]: the devices
      form nodes of [h] consecutive devices, and the flattened value [h] chunks.
      Each device reduces one chunk within its node, then with the devices of
      the same rank in the other nodes, and each node gathers the chunks;
    - {e naive}, unless all-to-all or ring applies: each shard is copied to
      [red]'s devices, the copies are combined in device order, and the result
      is shrunk back to [x]'s shape;
    - {e all-to-all}, when [x]'s shape is known and {!Setting.all2all} is [2],
      or [1] with more than two devices and more elements than
      [RING_ALLREDUCE_THRESHOLD] (default [256000]): the flattened value is cut
      into [n] chunks, of sizes multiples of the largest of [32], [16], [8], [4]
      and [2] that divides its size; device [i] gathers and reduces chunk [i]
      from every device, then sends it to every device;
    - {e ring}, when all-to-all does not apply and {!Setting.ring} is [2], or
      [1] under the same conditions: the same chunks each travel around the ring
      of devices, reduced at each step, and the reduced chunks travel around it
      again to reach every device.

    When [red] places its result on one device, each reduced chunk is copied
    there. With {!Setting.debug} at [2] or more, the algorithm and the size are
    printed on standard output. *)

val create_allreduce_function : Ops.t -> Ops.t
(** [create_allreduce_function red] is the value of the allreduce [red],
    computed by a call to a function of its own, compiled on its own and named
    ["allreduce"]. The call stores {!handle_allreduce}'s value into new storage
    ({!Op.Alloc}) of [red]'s type and greatest size on [red]'s devices; its
    arguments are that storage, then [red]'s source made contiguous
    ({!Ops.contiguous}). The value is the storage viewed at [red]'s shape,
    ordered after the call.

    Raises [Invalid_argument] if [red] is not an {!Op.Allreduce}, or its source
    is not on several devices. *)
