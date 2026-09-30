(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Batches of calls, encoded into command queues and submitted by host
    programs.

    A device that runs work from command queues gets a schedule's calls in
    batches. {!compile_linear} groups each run of consecutive calls on such
    devices into one batch, and orders the calls of a batch on the device's
    queues (a compute queue, and copy queues) with waits on the signals of the
    queues and devices whose results they read ({!sched_batches}). Each queue's
    commands are then encoded into command words by the device's vendor
    ({!commands}), and the batch becomes a {e host program}: a kernel for the
    host that writes the words known only when it runs into the command buffers,
    and submits them. The rest of the words are written once, when the batch is
    linked.

    Everything here is data. Nothing opens a device: the engine links each batch
    (it allocates the storage the batch names and writes the words known at
    link), and runs it, as {!section-running} says. *)

(** {1:queues Command queues} *)

(** The command queues of a batch, as a vendor encodes them. *)
module Queue : sig
  type t
  (** The type for a queue being encoded: the bytes of its commands so far, and
      the words among them computed when the batch is linked or run. *)

  val devices : t -> string list
  (** [devices q] is the devices [q]'s commands run on, the first submitting
      them. *)

  val name : t -> string
  (** [name q] is [q]'s name among its device's queues: ["COMPUTE:0"], or
      ["COPY:i"] for the [i]th copy queue. *)

  val q : t -> Ops.t list -> int
  (** [q q words] appends [words] to [q], and is [q]'s size in bytes after them.
      A constant (seen through casts) appends its value in its type's width,
      little-endian; an {!Op.Binary} its bytes; any other node leaves room of
      its type's width for its value, written when the batch is linked or run.
  *)

  val dword : int -> Ops.t
  (** [dword n] is the 32-bit command word [n]: its low 32 bits, as a [uint32]
      constant. *)

  val size : t -> int
  (** [size q] is the number of bytes [q] holds. *)

  val get_dword : t -> int -> int
  (** [get_dword q off] is the 32-bit word at byte [off] of [q]. *)

  val set_dword : t -> int -> int -> unit
  (** [set_dword q off n] makes [n] the 32-bit word at byte [off] of [q], a word
      a constant wrote. *)

  val reset : t -> unit
  (** [reset q] empties [q], to encode another stream of commands. *)
end

type commands = {
  exec : Ops.t -> Ops.t -> unit;
      (** [exec call prg] enqueues the compiled program [prg] ({!Op.Program}) on
          the arguments of [call]. *)
  copy : Ops.t -> Ops.t -> int -> unit;
      (** [copy dst src n] enqueues a copy of [n] bytes of the storage [src]
          into [dst]. *)
  wait : Ops.t -> Ops.t -> unit;
      (** [wait signal v] makes the commands after it wait until the 64-bit word
          [signal] holds at least [v]. *)
  signal : Ops.t -> Ops.t -> unit;
      (** [signal word v] stores [v] into the 64-bit [word] once the commands
          before it are complete. *)
  timestamp : Ops.t -> unit;
      (** [timestamp slot] writes the device's time into the second word of the
          16-byte [slot] once the commands before it are complete. *)
  memory_barrier : unit -> unit;
      (** [memory_barrier ()] makes the memory the queue's earlier work wrote
          visible to its later work. *)
  submit : Ops.t -> Ops.t;
      (** [submit cmdbuf] is the effect of the host program that submits the
          command buffer [cmdbuf], the queue's commands. *)
}
(** The type for the commands a vendor encodes on one queue. *)

val bufferize_cmdbuf : Queue.t -> string -> string -> Ops.t
(** [bufferize_cmdbuf q name device] is the command buffer of [q]'s commands: a
    placeholder of [device] tagged [to_name [name; Queue.name q]] with the words
    of [q] written into it, those known at link then and the others when the
    host program runs. Each word used at several offsets is written by one loop
    over them. The {!Op.Linear}s that words address ({!Op.Getaddr}), such as a
    program's arguments, are buffers of their own, one per {!Op.Linear} name
    (its argument), written before the command buffer. *)

(** {1:devices Devices} *)

type queues = {
  commands : Queue.t -> commands;
      (** [commands q] encodes the queue [q] of the vendor's devices. *)
  copy_queue : bool;
      (** Whether the device copies on queues of its own. Without them, a copy
          on the device is a kernel on its compute queue. *)
  host : string;
      (** The device that runs the host programs submitting to the device's
          queues, and whose memory holds the staging of copies. *)
  reaches : string -> bool;
      (** [reaches d] is [true] iff the device's queues address the memory of
          the device [d], another than itself: its queues always address its
          own. A copy from or to memory the device cannot address goes through
          staging memory of [host], copied in by the source's queues and out
          by the destination's. *)
}
(** The type for the command queues of a device, as a compiler sees them. *)

type device = {
  target : Helpers.Target.t;
      (** What the device's programs are compiled for. *)
  queues : queues option;
      (** Its command queues, or [None] for a device that runs no work from
          queues: its calls are run one by one, outside batches. *)
}
(** The type for devices, as a compiler sees them. The compiler never opens a
    device: the engine describes each device a schedule names with a [device].
*)

(** {1:words Words and storage} *)

val unwrap_view : Ops.t -> Ops.t * int
(** [unwrap_view v] is the storage [v] views and the byte offset of the view,
    through bitcasts, {!Op.After}s and one-dimensional shrinks.

    Raises [Invalid_argument] if a shrink is not one-dimensional or its start
    is not a constant. *)

val unwrap_lane : Ops.t -> Ops.t * int option * int
(** [unwrap_lane v] is {!unwrap_view}, through a shard selection too: the
    storage, the shard selected, if any, and the byte offset.

    Raises [Invalid_argument] as {!unwrap_view} does. *)

val lane_offset : Ops.t -> Ops.t * int option * Ops.sint
(** [lane_offset v] is {!unwrap_lane} for a view whose offset may read a
    variable, such as a range's value ({!range_value}): its byte offset is a
    symbolic integer.

    Raises [Invalid_argument] if a shrink of the view has more than one
    dimension. *)

val to_name : string list -> string
(** [to_name parts] is [parts] joined by ['_'], lowercased, with each [':']
    replaced by ['_']: [to_name ["cmdbuf"; "COMPUTE:0"]] is
    ["cmdbuf_compute_0"]. *)

val signal_word : string -> Ops.t
(** [signal_word d] is the 64-bit word into which [d]'s queues store the value
    of their work when it completes: a volatile placeholder of [d], tagged
    ["timeline"]. *)

val submitted : string -> Ops.t
(** [submitted d] is the value of the last work submitted to [d] before the
    batch: a variable of the batch's host program, which the engine binds. *)

val value : string -> Ops.t
(** [value d] is the value the batch's work on [d] signals once complete: a
    variable of the batch's host program, which the engine binds. *)

val range_value : Ops.t -> Ops.t
(** [range_value r] is the value of the range [r] in the calls it is around: a
    variable named ["range_"] and [r]'s identity ({!Ops.range_str}), of [r]'s
    type committed ({!Dtype.strong}), from [0] to the last value of [r], which
    the engine binds on each trip. *)

val patch : ?blob:string -> Ops.t -> (Ops.t * Ops.t) list -> Ops.t
(** [patch ~blob buf rows] is [buf] after [blob] (if any) is written at its
    start, then each [(off, w)] of [rows]: the word [w] at byte [off]. A row
    whose offset is a constant and whose word is known at link (it reads no
    variable, no load and no register, and addresses no input and no view that
    moves with a range) is written when the batch is linked; the others when its
    host program runs. Rows of one type, one alignment within it, one time of
    writing and one loop are written by one store of a stack. *)

(** {1:ffi Host functions and C structures} *)

val layout_args : ?offset:int -> Ops.t list -> (int * Ops.t) list
(** [layout_args ~offset args] is each of [args] with its byte offset when they
    are laid out in order from [offset] (default [0]), each aligned to its size
    ({!Device.Tiny_elf.iter_sig}). *)

val pack_args : (int * Ops.t) list -> int -> Ops.t list
(** [pack_args rows size] is the words of [size] bytes that hold each word of
    [rows] at its byte offset, zeros ({!Op.Binary}) elsewhere. *)

val ccall :
  host:string -> lib:string -> ?ret:Dtype.t -> string -> Ops.t list -> Ops.t
(** [ccall ~host ~lib ~ret f args] calls the C function [f] of the library [lib]
    on [args] from the host program, returning [ret] (default {!Dtype.Void}).
    Its address is read from a word of [host], a placeholder tagged
    [("cfunc", lib, f)] that the engine fills with the address at link. *)

type c_struct = {
  struct_name : string;  (** The struct's name. *)
  struct_size : int;  (** Its size in bytes. *)
  fields : (string * int * int) list;
      (** Each field's name, byte offset and size in bytes. *)
}
(** The type for the layouts of C structures. *)

val cstruct : host:string -> c_struct -> (string * Ops.t) list -> Ops.t
(** [cstruct ~host s fields] is a C structure of layout [s] in volatile memory
    of [host], zeroed, with each [(f, v)] of [fields] written into the field [f]
    as the unsigned integer of the field's size ({!patch}).

    Raises [Invalid_argument] naming a field [s] lacks. *)

val cfield : Ops.t -> c_struct -> string -> Ops.t
(** [cfield buf s f] is the field [f] of the structure of layout [s] at the
    start of [buf], as an unsigned integer of its size, ready to load or store.

    Raises [Invalid_argument] naming a field [s] lacks. *)

(** {1:deps Dependencies} *)

(** Dependencies between accesses to storage. *)
module Deps : sig
  type 'a t
  (** The type for the accesses recorded so far, each by an ['a]. *)

  val make : unit -> 'a t
  (** [make ()] has no access recorded. *)

  val access : 'a t -> Ops.t list -> writes:int list -> 'a -> 'a list
  (** [access t bufs ~writes x] records that [x] accesses the storage views
      [bufs], writing those at the positions [writes] and reading the others,
      and is the accesses [x] must follow, each once, in the order found: the
      last writes of the bytes [x] reads or writes, and, for the bytes it
      writes, the reads since. Views are compared by storage, shard and byte
      range ({!unwrap_lane}), so accesses to disjoint bytes do not depend on
      each other, and a write keeps the accesses to the bytes it does not write.
  *)
end

(** {1:batches Batches} *)

val sched_batches : devices:(string -> device) -> profile:bool -> Ops.t -> Ops.t
(** [sched_batches ~devices ~profile linear] is [linear] with each run of
    consecutive calls enqueued on devices with queues replaced by one batch per
    kind of device ({!Helpers.Target.t.device}), in the order the kinds first
    appear. A call is enqueued on the device of its buffers that has queues, the
    source's for a copy, and a program runs on its compute queue
    (["COMPUTE:0"]), a copy on a copy queue (["COPY:0"], or with
    {!Helpers.all2all}, one of up to eight between AMD devices, or the
    [HCQ_NUM_SDMA] of them). A call is not enqueued when a device of its buffers
    has no queues, and neither is a copy on Metal, whose memory the host copies.

    A range around calls ({!Op.End} of a call, or of an {!Op.Linear} of calls)
    stays a range, around its calls batched on their own, with each of its
    ranges [r] replaced in them by [range_value r]: the calls run once per trip,
    each trip one submission.

    A batch is a call, with an {!Ops.hcq_info} argument and, with [profile], the
    slots of its devices as arguments, of a sink of one submission per queue. A
    submission is a {!Op.Custom_function} named
    [to_name ["submit"; kind; queue kind]] of an {!Op.Linear} of the queue's
    commands, ordered after the submissions before it and after the batch's
    fence. The commands of a queue are, in order:
    - once, a memory barrier, and waits for the work submitted before the batch
      on its device and on the devices whose memory its calls touch
      ([signal_word d] at least [submitted d]);
    - for each call: waits for the calls on other queues it depends on
      ({!Deps}), each a wait for the queue's signal to reach the call's position
      plus one; with [profile], a timestamp before and after it; the call; and,
      when another queue waits for it, a store of its position plus one into its
      queue's signal;
    - on one queue of each device, the compute queue when the device has
      several, waits for the device's other queues and for the queues of other
      devices that touched its memory, then a store of [value d] into
      [signal_word d].

    On NV, a compute queue that waits for another queue also waits for its own
    previous call. The fence re-arms every queue signal (stores [0]).

    The slots of a device are a volatile placeholder tagged ["slots"] of 16-byte
    slots, each a signal then a timestamp: one per queue of the device, then,
    with [profile], two per call of the batch, its start and end timestamps. *)

val lower_call : devices:(string -> device) -> Ops.t -> Ops.t
(** [lower_call ~devices batch] is the batch [batch] as a call of its host
    program, a kernel of the host of its first device ({!queues.host}):
    - each submission is encoded by its device's {!commands}, and the fence
      becomes the stores it makes;
    - the address of each input (a storage view over an untagged {!Op.Param}) is
      loaded from the address table, a placeholder of the host tagged
      ["inputs"], where the engine writes it on each run. The addresses of other
      storage are written into the table at link;
    - the placeholders of one tag, device, type and volatility become views of
      one, each 128-byte aligned;
    - the words known at link are hoisted out of the kernel into the stores the
      result is ordered after ({!Op.After}), for the engine to write at link.

    The call's arguments are the placeholders its host program reads, in the
    order it reads them; its host program's variables ({!submitted}, {!value}
    and the schedule's) are bound by name. Its {!Ops.hcq_info} gives the
    arguments' number ([nargs]), the table's position ([table], or [-1]), each
    input ([inputs]: the storage, the byte offset and the device whose address
    the table holds, in table order), and the position of each device's slots
    ([slots]).

    Raises [Invalid_argument] if [batch] is not a batch, or is lowered already.
*)

(** {1:compiling Compiling} *)

val compile_linear :
  ?search:(int -> Postrange.Scheduler.t -> Postrange.Scheduler.t) ->
  ?profile:bool ->
  devices:(string -> device) ->
  Ops.t ->
  Ops.t
(** [compile_linear ~search ~profile ~devices linear] is the schedule [linear],
    whose devices [devices] describes, ready to link and run:
    + when the setting {!Helpers.beam} is [1] or more, each kernel that asks for
      no beam search asks for one of that width;
    + its kernels are compiled ({!Realize.lower_and_compile}, with [search]);
    + a call enqueued on devices with queues whose buffers are on several
      devices becomes one call per device, each with the [d]th shard of its
      buffers and ["_device_num"] bound to [d];
    + a copy between memory the queues cannot address ({!queues.reaches})
      becomes copies through the two halves of a 128 MiB staging placeholder of
      the host, tagged ["staging"], in turn; one on a device without copy queues
      becomes a kernel that copies bytes;
    + its enqueued calls are batched ({!sched_batches}), each batch is lowered
      ({!lower_call}) and its host program compiled, with no dtype emulated.

    [profile] defaults to {!Helpers.debug} at [2] or more. A linear that holds a
    lowered batch is returned as it is.

    Raises as {!Realize.lower_and_compile} does. *)

(** {1:running Linking and running}

    The engine runs [compile_linear]'s result. Each call of it is one of:
    - {b a kernel}, a call of a compiled {!Op.Program} without {!Ops.hcq_info},
      on devices without queues. Its arguments are storage views and bound
      variables; the program runs on each device of its first argument
      ({!Device.Tiny_elf.of_program}).
    - {b a copy}, a call of an {!Op.Store} of its second argument's storage into
      its first's.
    - {b a batch}, a call of the compiled host program with an {!Ops.hcq_info}
      [i], ordered after its link patches ({!Op.After}).
    - {b a range around calls}, an {!Op.End} of ranges around one of the above,
      or around an {!Op.Linear} of them. The engine runs them once for each
      combination of the ranges' values, the last range varying fastest, with
      [range_value r]'s variable bound to [r]'s value. A batch in a range is one
      submission per trip; views of storage in it move with the ranges' values,
      and the batch adds their moves to the addresses it reads from its table.

    {b Linking a batch}, once:
    + Each argument is a placeholder, which the engine allocates on its device:
      a volatile one in memory the host and the device both see coherently; a
      ["cmdbuf"] one uncached. The engine binds [signal_word d]'s placeholder to
      [d]'s signal word, fills a [("cfunc", lib, f)] one with [f]'s address, and
      allocates each tag its vendor names ({!commands}) as the vendor says. Each
      {!Op.Buffer} the patches address is allocated, unless bound.
    + Each link patch is then written, in order: a store of bytes ({!Op.Binary},
      possibly bitcast) into a view of an argument, or of a stack of words into
      a view at a stack of constant element indices. The address of a view
      ({!Op.Getaddr}) is its storage's address on the device the address is
      taken on, plus its byte offset; with addresses known, every word is a
      constant expression.

    {b Running a batch}, each time, inside one [Nx_device.submit] of the devices
    [i.device], touching every buffer the batch reaches:
    + {b the run fence}: for each device [d] of [i.device], it waits for the
      value [d] was signalled by the batch's previous run ([Submission.wait]).
      Runs of one linked batch share its arguments, so they are serialized;
    + it writes the address table: entry [k] is the address, on the device
      [dev], of the storage bound to [base], plus [off], for the [k]th
      [(base, off, dev)] of [i.inputs];
    + it calls the host program on its arguments, with [submitted d]'s variable
      bound to [d]'s last submitted value ([Nx_device.submitted]) and
      [value d]'s to [Submission.value], for each [d] of [i.device], and the
      schedule's variables to theirs. The host program waits on each device for
      the devices it names, so a pair of [Submission.waits] of a device outside
      [i.device] is waited for on the host;
    + with a profile, it records each kernel's span on each of its devices
      ([Submission.record]): the 32 bytes of the device's slots whose second and
      fourth words are the kernel's [stamps]. *)
