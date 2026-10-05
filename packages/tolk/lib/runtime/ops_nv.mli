(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA's command queues, as batches encode them.

    An NV device runs work from two channels, a compute channel and a copy
    channel, each fed by a ring of 64-bit entries (its GPFIFO), each entry
    naming a segment of 32-bit method words. A batch's queue ["COMPUTE:0"] is
    the compute channel and ["COPY:0"] the copy channel. Its host program
    appends the queue's command buffer to the channel: it writes the buffer's
    entry at [put mod entries] of the ring, stores [put + 1] into [put] and
    [(put + 1) mod entries] into [gp_put], then the channel's token into its
    doorbell. These words are volatile placeholders of the device, which the
    engine binds to the channel's ({!storage}).

    A launch on the compute channel runs from a launch descriptor (a QMD), which
    addresses the launch's constant buffer 0: the driver's parameters, then the
    addresses of the buffers and the variables. Both are regions ({!Ops.arg}'s
    [Region]) of 256-byte alignment, named ["qmd"] and ["cbuf"], which
    {!Hcq2.bufferize_cmdbuf} lays out in a buffer per name; a launch in a loop
    has a copy of each for every trip. Each program is its cubin, which the
    engine loads on the device ({!Program}), and its launch descriptor is made
    from the cubin's registers, shared and local memory and constant banks. The
    local memory each thread may use is the word of a placeholder ({!Local}),
    which the engine fills once the device's local memory holds what the batch's
    kernels need per thread.

    Values signalled are 64 bits. The compute channel writes them whole. The
    copy engine writes 32-bit words: its signal writes the low word of the
    value, then, when that low word is [0], the high word at the next four bytes
    of the signal word; when the low word is not [0], the second write goes to a
    word of its own, a volatile placeholder of the device tagged ["nv_sink"]. A
    word being written thus never reads above its old value, and a high word
    written late never takes the value back. *)

(** {1:queues Command queues} *)

type channel = {
  entries : int;  (** The number of entries of its ring. *)
  token : int;  (** The token its doorbell takes. *)
}
(** The type for the channels of a device, as its host programs address them. *)

type props = {
  compute_class : int;
      (** The class of the compute engine, which decides the layout of launch
          descriptors: version 5 from Blackwell's class on, version 3 before. *)
  sass_version : int;  (** The version of the machine code the GPU runs. *)
  shared_window : int;
      (** The address at which kernels' shared memory appears. *)
  local_window : int;  (** The address at which their local memory appears. *)
  compute : channel;  (** The compute channel. *)
  copy : channel;  (** The copy channel. *)
}
(** The type for the properties of an NV device that its commands depend on. *)

val queues : host:string -> reaches:(string -> bool) -> props -> Hcq2.queues
(** [queues ~host ~reaches props] is the command queues of an NV device of
    properties [props], whose batches are submitted by host programs of [host]
    and whose queues address the memory of the devices [reaches] holds for. It
    has a compute queue and one copy queue. The compute queue's commands are:
    - [exec call prg], which lays out [prg]'s launch descriptor and constant
      buffer 0 at the next slot of the queue's descriptors, and starts it after
      the queue's previous launch: chained to that launch's descriptor when no
      other command came between them, and scheduled by the channel otherwise;
    - [wait word v], a 64-bit semaphore acquire of [word], circular and at least
      [v];
    - [signal word v], a 64-bit release of [v] into [word] once the commands
      before it are complete: through one of the two releases of the previous
      launch's descriptor if a launch directly precedes it and one is free, and
      a semaphore release of the channel after waiting for it to idle, with a
      non-stalling interrupt, otherwise;
    - [timestamp slot], the same release of [0], which also writes the GPU's
      timer into the second word of [slot];
    - [memory_barrier ()], which invalidates the compute engine's instruction,
      data and constant caches, and ends the chain of launches;
    - [loop r body], which runs [body]'s commands once per trip of [r], each
      trip's launches chained on descriptors of the trip's own and scheduled by
      the channel once per trip;
    - [submit ()], which appends the queue's command buffer to its channel, one
      entry of its ring, and raises {!Hcq2.Over_capacity} if the buffer has more
      32-bit words than an entry's length field holds, [2{^21} - 1].

    The copy queue's commands are [copy dst src n], copies of at most 2 GiB each
    from [src] to [dst], [wait] as above, [signal], through the copy engine as
    the module's introduction says, [timestamp], a four-word release of the copy
    engine, [loop] and [submit].

    Raises [Invalid_argument] from [exec] if [prg] is no cubin, or launches more
    threads than its registers allow, more than 1024 threads per block, or a
    size above the hardware's limits, and from the copy queue's [exec] and the
    compute queue's [copy]. *)

(** {1:linking What the engine links} *)

(** The type for the storage that the placeholders of NV's commands name. The
    queues are named ["COMPUTE:0"] and ["COPY:0"], as {!Hcq2.Queue.name} names
    them. *)
type storage =
  | Program of { binary : string; name : string }
      (** The cubin [binary], loaded on the device, of which the commands launch
          the kernel [name]: its image ({!Nx_nv_cubin.image}), relocated.
          Launches address its code and constant banks at the offsets of that
          image. *)
  | Ring of string  (** The ring of entries of the queue's channel. *)
  | Gp_put of string
      (** The 32-bit index after the last entry written in the ring, which the
          GPU reads. *)
  | Put of string
      (** The 64-bit count of the entries ever written in the ring, where the
          next writer appends. *)
  | Doorbell of string  (** The doorbell of the queue's channel. *)
  | Local of int
      (** A 32-bit word holding the bytes of local memory per thread the
          device's local memory provides, once it provides at least that many
          bytes. The word is read when the host program runs, so it may grow
          after the batch is linked. *)

val storage : Ops.t -> storage option
(** [storage u] is the storage of the placeholder [u] of a batch if NV's
    commands name it, and [None] for the others, which the engine allocates. *)
