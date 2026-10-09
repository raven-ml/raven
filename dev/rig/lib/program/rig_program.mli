(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Programs as data: work for one machine's devices, described once, loaded,
    and run many times.

    A {e program} is memory and images on the devices of one machine, and the
    {e steps} of its {e run}: submissions and copies. Its {e description} ({!t})
    is plain data that names each of these by its index in the description, and
    links no vendor library. What only the loading process knows, such as an
    address or an image's function, is a {e hole} in the description's bytes,
    filled when it is loaded ({!leaf}). What changes from one run to the next is
    the run's {e frame}: the buffers it reads and writes ({!frame}).

    {!load} makes a program's memory, loads its images and prepares one
    submission per step, on the devices it is given. {!run} runs the steps once,
    in order, over a frame.

    {v
      t ──load devices──> loaded ──run frame──> points
      │                     │
      memory, images,       Rig.Buffer, Rig.Image,
      steps                 one Rig.Submission per step (two for Two)
    v}

    {1:order Order}

    rig orders a run's work by the memory it names, as it orders every submit
    ({!Rig.submit}): a step's submission names its memory as its slots, its
    parts' buffers and its fixed memory, and work that reads memory follows its
    last write, on any device. So steps on several devices need no other order
    than the description's: a step follows the earlier steps whose memory it
    names, and its device's earlier work. Work that reaches memory its step does
    not name, through an address in memory or a parameter, is not ordered by it:
    the description names it, as {!Rig.Submission.make}'s [fixed] states.

    {1:domains Domains}

    {!load} and {!run} may be called from any domain. The runs of one loaded
    program take turns: they share its memory, and a run starts once the
    previous one returned. *)

(** {1:descriptions Descriptions} *)

(** The type for what only a loaded program knows: a word of its memory or its
    images, filled once at {!load}. *)
type leaf =
  | Address of { memory : int; on : int }
      (** The address of memory [memory]'s first byte as device [on]'s work
          addresses it ({!Rig.Buffer.address}): through a borrow on [on] made at
          load where the memory is another device's. For memory with [Two]
          copies, the copy the run uses ({!type-copies}). *)
  | Handle of int
      (** The driver's object for memory [i] ({!Rig.Buffer.handle}), such as an
          [MTLBuffer]. For memory with [Two] copies, the copy the run uses. *)
  | Entry of { image : int; name : string }
      (** What compiled code names image [image]'s function [name] by
          ({!Rig.Image.entry}): a kernel descriptor's address, a [CUfunction],
          an executable graph. *)

(** The type for the widths of holes. *)
type width = W32 | W64

type 'a hole = {
  at : int;  (** The hole's first byte. *)
  width : width;
  leaf : 'a;
  add : int;
  shift : int;
}
(** The type for holes: the low [width] bits of [(v + add) lsr shift], [v] the
    leaf's value, ORed into the little-endian word of the bytes at [at]. The
    bytes hold the word's constant bits, and [0] where the value goes. For
    example, the high half of an address is
    [{ width = W32; add = 0; shift = 32 }], and a word that holds an address
    beside a length in its top bits holds the length in the bytes. [v + add] is
    an [int]: in a [W64] word, the value's bit 63 is its sign, so a constant bit
    63 lies in the bytes. [shift] is [0] to [62], and holes of one bytes share
    no byte. *)

type 'a data = { bytes : string; holes : 'a hole array }
(** The type for bytes with holes. *)

(** The type for what a run knows: a word read when the step that needs it runs.
*)
type value =
  | Fixed of int  (** The integer itself. *)
  | Input of { input : int; on : int }
      (** The address of the run's input [input] as device [on]'s work addresses
          it, through a borrow on [on] made for the run where it is another
          device's. *)
  | Leaf of leaf

(** The type for how many copies of a memory a program keeps. *)
type copies =
  | One  (** One, which every run uses. *)
  | Two
      (** Two: run [n] uses copy [n mod 2], and runs once the work of run
          [n - 2] on it is done. Work a run hands over may then write its copy
          while the previous run's work still reads the other. *)

(** The type for a program's memory. *)
type memory =
  | Alloc of {
      device : int;
      kind : Rig.Buffer.memory;
      bytes : int;
      init : leaf data;
          (** Its first bytes, written at load into each copy, each hole filled
              for that copy. *)
      copies : copies;
    }  (** Memory made at load ({!Rig.Buffer.create}). *)

type image = { device : int; binary : leaf data }
(** The type for images: [binary] in [device]'s format, loaded in the
    description's order ({!Rig.Image.load}). A hole may name an earlier image's
    entry, so an image can be a vendor object over loaded code, such as a CUDA
    graph over a module's functions, where the device's driver takes it. *)

type view = { memory : int; offset : int; length : int }
(** The type for [length] bytes of memory [memory] from its byte [offset]: of
    the run's copy for memory with [Two] copies. *)

(** The type for the buffers a step names. *)
type slot = Memory of view | Input of int  (** The run's input [i], whole. *)

(** The type for the work of a part, as {!Rig.Submission.work}. *)
type work =
  | Words of view
  | Fill of { fill : leaf; arg : view; ring_units : int; segment_bytes : int }
  | Copy of { src : view; dst : view }
  | Launch of {
      image : int;
      kernel : string;
      params : value data;
          (** Its parameters' bytes, a whole number of 32-bit words and at most
              4096 bytes, each hole written into the run's block before each
              submit. *)
      refs : Rig.Submission.ref array;
      groups : value * value * value;
      threads : value * value * value;
      shared : value;
    }

type part = { queue : string; after : int array; work : work }
(** The type for parts, as {!Rig.Submission.part}. *)

type submit = {
  device : int;
  parts : part array;
  reads : slot array;
  writes : slot array;
  fixed : (view * Rig.Buffer.access) array;
}
(** The type for submissions: [parts] on [device], each run reading [reads] and
    writing [writes], whose refs count [reads] then [writes]
    ({!Rig.Submission.ref}), with the fixed memory [fixed]. Memory of another
    device is named through its borrow on [device], made at load; an input of
    another device, through a borrow made for the run. *)

(** The type for the steps of a run. *)
type step =
  | Submit of submit  (** One {!Rig.submit}. *)
  | Move of { src : slot; dst : slot }  (** One {!Rig.Buffer.copy}. *)

type input = { device : int; bytes : int; access : Rig.Buffer.access }
(** The type for a run's inputs: a buffer of [device] of at least [bytes] bytes,
    whose memory admits [access]. *)

type t = {
  devices : string array;  (** Each device's {!Rig.arch}, checked at load. *)
  memory : memory array;
  images : image array;
  inputs : input array;
  steps : step array;
}
(** The type for descriptions. *)

(** {1:loading Loading} *)

type loaded
(** The type for loaded programs. A loaded program keeps its memory and images
    while it is reachable; once it is not, they return by rig's rules, after the
    work of its last run. *)

val load : t -> Rig.t array -> (loaded, string) result
(** [load t devices] is [t] loaded on [devices], device [i] of [t] being
    [devices.(i)], all of one machine ({!Rig.host_of}). It:
    + checks [t] against [devices];
    + makes each memory, twice for [Two] copies, and borrows it on the devices
      that name it;
    + loads each image in order, its holes filled;
    + writes each memory's [init], its holes filled;
    + makes each [Submit] step's submission, one per copy where it names memory
      of [Two] copies, and stores its launches' parameters and geometry that no
      run changes.

    [Error why] if [t] is not well formed for [devices]: an index out of its
    array, other than as many devices as [t]'s, a device whose arch differs, a
    negative size, a hole outside its bytes, sharing a byte with another or with
    a [shift] outside [0] to [62], a leaf naming memory of [Two] copies in an
    image or in memory of [One] copy, an [Entry] of a later image, an [init] or
    a view outside its memory, a launch's parameters of another length, a memory
    that a device naming it cannot borrow ({!Rig.Buffer.borrow}), an [Address]
    of memory named by its handle only, or a step whose submission
    {!Rig.Submission.make} refuses or whose launch geometry a run refuses
    ({!Rig.Submission.Run}), [why] naming the step; and with the reason,
    starting with the device's name, if a device refuses an image
    ({!Rig.Image.load}) or has no function an [Entry] names.

    Raises [Invalid_argument] if [devices] are of several machines,
    {!Rig.Out_of_memory} and {!Rig.Lost}. *)

(** {1:running Running} *)

type frame = { inputs : Rig.Buffer.t array }
(** The type for a run's frame: its inputs, by index. *)

val run : ?after:Rig.Point.t array -> loaded -> frame -> Rig.Point.t array
(** [run ~after p f] runs [p]'s steps once, in order, over [f], each device's
    first submission after the points of [after] (defaults to none) and after
    its device's earlier work. Before its first step it waits for the work of
    run [n - 2] on its copies of [Two] memory. It is the points the run's work
    ends at, one for each device that ran a submission, in the order of [p]'s
    devices: what it submitted last there. Its [Move] steps are done when it
    returns.

    Raises [Invalid_argument] if [f]'s inputs are not as many as [p]'s, an input
    is not on its device, holds fewer bytes, or admits less access than [p]
    declares, an input of another device cannot be borrowed, and as
    {!Rig.submit} and {!Rig.Buffer.copy} raise; {!Rig.Lost} as the devices raise
    it. *)
