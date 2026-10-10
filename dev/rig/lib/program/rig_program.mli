(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Programs as data: work for one machine's devices, described once, loaded,
    and run many times.

    A {e program} is memory, images and host code on the devices of one machine,
    and the {e steps} of its {e run}: submissions, copies, calls of host code
    and loops. Its {e description} ({!t}) is plain data that names each of these
    by its index in the description, so it crosses between processes as bytes
    ({!to_string}) and links no vendor library. What only the loading process
    knows, such as an address or an image's function, is a {e hole} in the
    description's bytes, filled when it is loaded ({!leaf}). What changes from
    one run to the next is the run's {e frame}: the buffers it reads and writes,
    and its ints ({!frame}).

    {!load} makes a program's memory, loads its images, links its host code and
    prepares one submission per step, on the devices it is given. {!run} runs
    the steps once, in order, over a frame.

    {v
      t ──load devices──> loaded ──run frame──> points
      │                     │
      memory, images,       Rig.Buffer, Rig.Image, Rig_host.t,
      code, steps           one Rig.Submission per step (two for Two)
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

    A [Host] step runs on this process's host once the work that its buffers
    wait for is done ({!Rig.Buffer.wait}), and {!run} calls the steps in order,
    so a step after it follows its writes. Host code that follows a device's
    work names that work's memory among its buffers. This is how a device that
    cannot call a function, such as a GPU, signals a TCP rail
    ({!Rig_remote_abi}): its step writes the rail's outbound area ([Rail]
    memory), and a [Host] step that names the area calls the rail's ready
    function ([Ready]) with the transfer's count.

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
  | Code of int  (** The address of host code [i]'s entry. *)
  | Ready of int
      (** The address of the ready function of this machine's end of the rail of
          id [i] ([ready_fn] of {!Rig_remote_abi.end_}). *)
  | Ready_arg of int  (** That function's argument ([ready_arg]). *)

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
    beside a length in its top bits holds the length in the bytes. A value that
    meets a set bit of the bytes under it would clobber a constant: {!load}
    answers [Error] for it, and {!run} raises for a launch's value read per run,
    so the bits outside each value keep the description's bytes. [v + add] is an
    [int]: in a [W64] word, the value's bit 63 is its sign, so a constant bit 63
    lies in the bytes. [shift] is [0] to [62], and holes of one bytes share no
    byte. *)

type 'a data = { bytes : string; holes : 'a hole iarray }
(** The type for bytes with holes. *)

(** The type for what a run knows: a word read when the step that needs it runs.
*)
type value =
  | Fixed of int  (** The integer itself. *)
  | Int of int
      (** The 64-bit word [i] of the run's ints ({!t.ints}), as it holds when
          read: after the frame's ints were written, the enclosing loops' trips
          stored and the earlier [Host] steps returned. *)
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

(** The type for the areas of a rail's end ({!Rig_remote_abi.end_}). *)
type area = Outbound | Inbound | Counts

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
  | Rail of { rail : int; area : area }
      (** This machine's end's [area] of the rail of id [rail]: host memory the
          rail made, of one copy, which the program borrows. *)

type image = { device : int; binary : leaf data }
(** The type for images: [binary] in [device]'s format, loaded in the
    description's order ({!Rig.Image.load}). A hole may name an earlier image's
    entry, so an image can be a vendor object over loaded code, such as a CUDA
    graph over a module's functions, where the device's driver takes it. *)

type view = { memory : int; offset : int; length : int }
(** The type for [length] bytes of memory [memory] from its byte [offset]: of
    the run's copy for memory with [Two] copies. *)

(** The type for the buffers a step names. *)
type slot =
  | Memory of view
  | Input of int  (** The run's input [i], whole. *)
  | Ints  (** The run's ints, whole: [8 * ints] bytes of host memory. *)

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
      refs : Rig.Submission.ref iarray;
      groups : value * value * value;
      threads : value * value * value;
      shared : value;
    }

type part = { queue : string; after : int iarray; work : work }
(** The type for parts, as {!Rig.Submission.part}. *)

type submit = {
  device : int;
  parts : part iarray;
  buffers : (slot * Rig.Buffer.access) iarray;
  fixed : (view * Rig.Buffer.access) iarray;
}
(** The type for submissions: [parts] on [device], each run using [buffers],
    each with its access, which the parts' refs index ({!Rig.Submission.ref}),
    with the fixed memory [fixed]. Memory of another device is named through its
    borrow on [device], made at load; an input of another device, through a
    borrow made for the run. *)

type code = { obj : string; entry : string }
(** The type for host code: the ELF object [obj], linked on the loading
    machine's host and called at its function [entry] ({!Rig_host.link}). *)

type split = { extent : value; blocks : int; lo : int; hi : int }
(** The type for splits of host code's iterations, as {!Rig_host.split} with an
    extent read per run. *)

(** The type for the steps of a run. *)
type step =
  | Submit of submit  (** One {!Rig.submit}. *)
  | Move of { src : slot; dst : slot }  (** One {!Rig.Buffer.copy}. *)
  | Host of {
      code : int;
      buffers : (slot * Rig.Buffer.access) iarray;
      values : value iarray;
      split : split option;
    }
      (** A call of host code [code] on the host addresses of [buffers] and on
          [values] ({!Rig_host.call}), once the work each buffer waits for with
          its access is done ({!Rig.Buffer.wait}). Its buffers are memory the
          host addresses. *)
  | Loop of {
      trips : value;
      trip : int option;
      flag : view option;
      body : step iarray;
    }
      (** [body], [trips] times, none where [trips <= 0], storing each trip's
          index from [0] into word [trip] of the run's ints. With [flag], before
          each trip it reads [flag]'s first byte, once the work that wrote it is
          done, and stops at [0]. *)

type input = { device : int; bytes : int }
(** The type for inputs: a buffer of [device] of at least [bytes] bytes. *)

type t = {
  devices : string iarray;  (** Each device's {!Rig.arch}, checked at load. *)
  memory : memory iarray;
  images : image iarray;
  code : code iarray;
  inputs : input iarray;
  ints : int;
      (** The number of 64-bit words of the run's ints: host memory of two
          copies that {!load} makes ({!type-copies}). The frame's ints are its
          first words; loops store their trips there, and [Host] steps compute
          others. *)
  steps : step iarray;
}
(** The type for descriptions. *)

(** {1:bytes Bytes} *)

val to_string : t -> string
(** [to_string t] is [t] as bytes, in a format of a version of its own. *)

val of_string : string -> (t, string) result
(** [of_string s] is the description {!to_string} wrote as [s]. [Error why] if
    [s] is malformed, [why] saying where, or is of another version of the
    format, [why] naming both versions. It decodes only: {!load} checks the
    description. *)

(** {1:loading Loading} *)

type loaded
(** The type for loaded programs. A loaded program keeps its memory, images and
    code while it is reachable; once it is not, they return by rig's rules,
    after the work of its last run. *)

val load :
  ?rails:(int -> Rig_remote_abi.end_ option) ->
  t ->
  Rig.t iarray ->
  (loaded, string) result
(** [load ~rails t devices] is [t] loaded on [devices], device [i] of [t] being
    [devices.(i)], all of one machine ({!Rig.host_of}). [rails id] is this
    machine's end of the rail [id] ({!Rig_remote_abi.rail}), for a program on
    this machine (defaults to none). It:
    + checks [t] against [devices];
    + makes each memory, twice for [Two] copies, and borrows it on the devices
      that name it;
    + loads each image in order, its holes filled;
    + writes each memory's [init], its holes filled;
    + links each host code;
    + makes each [Submit] step's submission, one per copy where it names memory
      of [Two] copies, and stores its launches' parameters and geometry that no
      run changes.

    On another machine, whose devices are an agent's ({!Rig_remote_abi}), it
    checks [t] and the architectures here and loads [t] on that machine's host
    ({!Rig.Image.load}), whose agent loads it there ({!load_share}).

    [Error why] if [t] is not well formed for [devices]: an index out of its
    array, other than as many devices as [t]'s, a device whose arch differs, a
    negative size, a hole outside its bytes, sharing a byte with another or with
    a [shift] outside [0] to [62], a hole whose value meets a set bit of the
    bytes under it, a leaf naming memory of [Two] copies in an image or in
    memory of [One] copy, an [Entry] of a later image, an [init] or a view
    outside its memory, a launch's parameters of another length, a memory that a
    device naming it cannot borrow ({!Rig.Buffer.borrow}), an [Address] of
    memory named by its handle only, an [Int] or a loop's [trip] past the ints,
    a flag of no byte, a [Host] step's memory the host does not address, a
    [split] whose [blocks], [lo] or [hi] {!Rig_host.split} refuses, a [Rail]
    memory, [Ready] or [Ready_arg] of a rail that [rails] does not give, or a
    step whose submission {!Rig.Submission.make} refuses or whose launch
    geometry a run refuses ({!Rig.Submission.Run}), [why] naming the step; and
    with the reason, starting with the device's name, if a device refuses an
    image ({!Rig.Image.load}) or has no function an [Entry] names, and with
    [Rig_host.link]'s reason if host code does not link; on another machine, if
    a device is no agent's, and as the agent's load answers.

    Raises [Invalid_argument] if [devices] are of several machines, or of
    another machine and [rails] is given, {!Rig.Out_of_memory} and {!Rig.Lost}.
*)

(** {1:running Running} *)

type frame = { inputs : Rig.Buffer.t array; ints : int array }
(** The type for a run's frame: its inputs, by index, and the first words of its
    ints. *)

val run : ?after:Rig.Point.t array -> loaded -> frame -> Rig.Point.t array
(** [run ~after p f] runs [p]'s steps once, in order, over [f], each device's
    first submission after the points of [after] (defaults to none) and after
    its device's earlier work. It first waits for the work of run [n - 2] on its
    copies of [Two] memory, its ints included, then writes [f]'s ints. It is the
    points the run's work ends at, one for each device that ran a submission, in
    the order of [p]'s devices: what it submitted last there. The work it does
    on the host ([Move] and [Host] steps, a loop's flag) is done when it
    returns. An [Int] value reads the ints once the run's earlier work that
    writes them is done, and a loop stores its trip once the run's earlier work
    that reads or writes them is done. A step reads each input it uses from [f]
    as it runs.

    On another machine it is one submission on that machine's host, and the
    host's point: the agent reaches it once the run's points there are reached.
    [run] reads each of [f]'s inputs once, before it submits.

    Raises [Invalid_argument] if [f]'s inputs are not as many as [p]'s, an input
    it reads is not on its device, holds fewer bytes, or is read-only memory
    ({!Rig.Buffer.val-access}) that a step writes (a [Submit]'s or a [Host]
    step's [Read_write] buffer, or a [Move]'s destination), an input of another
    device cannot be borrowed, an input a [Host] step names is memory the host
    does not address, [f]'s ints are more than [p]'s, a launch's geometry or
    shared memory, read from the ints, that the run's setters refuse
    ({!Rig.Submission.Run}), a launch's hole whose value meets a set bit of its
    bytes, and as {!Rig.submit}, {!Rig.Buffer.copy} and {!Rig_host.call} raise;
    {!Rig.Lost} as the devices raise it. *)

(** {1:machines Programs of another machine}

    {!load} on another machine's devices sends that machine's host the
    description with each device's id there, as the binary of {!Rig.Image.load},
    and asks for the image's function ["run"] ({!Rig.Image.entry}). Each {!run}
    is one submission on the host whose one part is {!Rig.Submission.Words}: the
    entry {!Rig.Image.entry} answered, then the frame, each input as its
    memory's id on the machine, its offset and its length. The submission names
    no memory: the agent orders the run after the work handed over before it,
    and the work handed over after it after the run ({!Rig_remote}). The agent
    of the machine answers with these two functions. *)

val load_share :
  ?rails:(int -> Rig_remote_abi.end_ option) ->
  string ->
  device:(int -> Rig.t option) ->
  (loaded, string) result
(** [load_share ~rails b ~device] is {!load} [~rails] of the description in [b],
    the binary another process's {!load} sent to this machine's host, on the
    devices [device] gives for the ids [b] names. [Error why] as {!load}, or if
    [b] is malformed or names an id that [device] does not give. *)

val run_share :
  ?after:Rig.Point.t array ->
  string ->
  program:(int -> loaded option) ->
  region:(int -> Rig.Buffer.t option) ->
  (Rig.Point.t array, string) result
(** [run_share ~after w ~program ~region] is {!run} [~after] of the program that
    [program] gives for the entry [w] starts with, over the frame [w] holds,
    each input the view of [region id] at its offset and length. [Error why] if
    [w] is malformed, or [program] or [region] gives nothing for an id it names;
    otherwise it raises as {!run}. *)
