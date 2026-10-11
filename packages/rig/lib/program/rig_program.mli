(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Programs of steps on one machine's devices, as data that loads once and
    runs many times.

    A {e program} is the memory and images a computation needs on the devices
    of one machine, and the {e steps} each run of it takes: submissions,
    copies and loops. Its {e description}
    ({!t}) is plain data that names each of these by its index in the
    description, so it travels between processes as bytes ({!to_string}). What
    only the loading process knows, such as an address or an image's function,
    stands in the description's bytes as a {e hole}, which {!load} fills, or
    each run for a value it reads ({!type-hole}). What changes from one run to
    the next is the run's {e frame} ({!type-frame}): its input buffers and the
    first words of its {e ints}, 64-bit integers in host memory that loops and
    the work of its steps also read and write.

    Memory the program makes may have two copies ({!type-copies}). Run [n]
    uses copy [n mod 2], so a run's work may write its copy while the previous
    run's work still reads the other. The run's ints always have two copies.

    The life of a program goes like this:
    + A caller builds its description, in this process or in another one that
      sends it as bytes.
    + {!load} checks the description against the devices of one machine,
      makes its memory, loads its images and prepares a {!Rig.Submission} for
      each [Submit] step, one per copy where the step names memory of two
      copies.
    + {!run} runs the steps once, in order, over a frame, and returns the
      points its work ends at. A loaded program runs any number of times.
    + Once the loaded program is unreachable, its memory and images return
      after the work of its last run.

    {v
      t ──load devices──> loaded ──run frame──> points
      │                     │
      memory, images,       Rig.Buffer, Rig.Image,
      steps                 a Rig.Submission per Submit step and copy
    v}

    A program for another machine's devices loads and runs through that
    machine's agent ([Rig_remote], {!section-machines}), and programs on two
    machines exchange bytes through rails ({!section-rails}). Its work is
    ordered as every submission's is ({!section-order}). *)

(** {1:order Order}

    A step's work follows other work by the memory the step names, as every
    {!Rig.submit} does: work that reads memory follows the work that last
    wrote it, and work that writes memory follows every earlier use of it, on
    any device. A [Submit] step names the memory of its [slots], of its
    parts' views and of its [fixed]. A step therefore follows the earlier
    steps whose memory it names, and its device's earlier work.

    Work that reaches memory through an address it was given escapes this
    order, so an address or handle may stand only where its step names the
    memory. A step names memory {e whole} if one of its views starts at the
    memory's first byte and covers all of it, and {e fixes} it whole if that
    view is in its [fixed]. {!load} returns [Error] unless:
    {ul
    {- an image holds no [Address] or [Handle];}
    {- an [Address] in a launch's or a fill's parameters or geometry is of
       memory its step fixes whole, on its step's device, and a [Handle]
       there is of memory its step names whole;}
    {- a loop's trips hold no [Address], and a [Handle] there is of memory
       the loop's flag names whole;}
    {- a [Submit] step that names memory whose [init] holds an [Address] or
       a [Handle] of other memory names that other memory whole too;}
    {- an [Address] on device [d] in an [init] is of memory that a [Submit]
       step on [d] fixes whole.}}

    Host code runs as a [Submit] step on the machine's host, whose launches
    are done when its submit returns ({!Rig.section-host_code}).

    {1:rails Rails}

    A program reaches its machine's end of a rail
    ({!type:Rig_remote_abi.end_}) as [Rail] memory, in three areas:
    [Outbound] holds the bytes it sends, [Inbound] the bytes it receives, and
    [Counts] the counts that number the transfers. [Outbound] and [Inbound]
    each hold two copies of their bytes, independent of the program's
    {!type-copies}, and [Rail] memory covers both, so a step's views select
    the copy a transfer uses.

    No stamp records the rail's own reads and writes, so rig does not order
    them. A program orders them by the rail's counts, as
    {!Rig_remote_abi.section-rails} states. To send, a [Submit] step on the
    host after the work that writes [Outbound] names [Outbound], so that it
    follows that work, and [Counts], and its launch calls the rail's ready
    function, given as a [Ready] leaf, with the address of [Counts] and the
    count.

    {1:domains Domains}

    {!load} and {!run} may be called from any domain. Runs of one loaded
    program share its memory, so a run waits for the previous one to return
    before its first step. *)

(** {1:descriptions Descriptions} *)

(** The type for what only a loaded program knows: a word of its memory or
    its images, filled once at {!load}. *)
type leaf =
  | Address of { memory : int; on : int }
      (** The address of memory [memory]'s first byte as device [on]'s work
          addresses it, as {!Rig.Submission.address} gives it for a step on
          [on] that fixes the memory whole. For memory of two copies
          ({!type-copies}), the address of the copy the run uses. *)
  | Handle of int
      (** The driver's object for memory [i] ({!Rig.Buffer.handle}), such as
          an [MTLBuffer]. For memory of two copies, the object of the copy the
          run uses. *)
  | Entry of { image : int; name : string }
      (** What compiled code names image [image]'s function [name] by
          ({!Rig.Image.entry}): a kernel descriptor's address, a
          [CUfunction], an executable graph, the address of a function of
          the host's code. The loaded program keeps the image until the work
          of each of its runs is done. *)
  | Ready of int
      (** The address of the ready function of this machine's end of the rail
          of id [i] ([ready_fn] of {!type:Rig_remote_abi.end_}). *)

(** The type for the widths of holes. *)
type width = W32 | W64

type 'a hole = {
  at : int;  (** The hole's first byte. *)
  width : width;
  terms : ('a * int) iarray;  (** Each value with its factor. *)
  add : int;
  shift : int;
}
(** The type for holes: places in a description's bytes that receive a value
    when the program loads or runs. A hole ORs the low [width] bits of
    [s + add], as a 64-bit word shifted right by [shift] bits, into the
    little-endian word at byte [at], where [s] is the sum, over [terms], of
    each value times its factor. The bytes hold the word's constant bits, and
    [0] where the value goes. For example:
    {ul
    {- the high half of an address [a] is [{ width = W32; terms = [: (a, 1)
       :]; add = 0; shift = 32 }];}
    {- a word with an address in its low bits and a length in its high bits
       holds the length in the bytes;}
    {- at a ref's [at], whose bytes hold the offset the ref adds to its
       slot's address ({!Rig.Submission.type-ref}), the terms
       [[: (Int i, n) :]] reach row [r] of rows of [n] bytes, where word [i]
       of the run's ints holds [r], such as the trip a loop stores there.}}
    The rules:
    {ul
    {- [shift] is [0] to [62], and no two holes of one bytes share a byte.}
    {- A value that meets a set bit of the bytes under it would overwrite a
       constant. {!load} returns [Error] for it, and {!run} raises for a
       parameter's value read per run.}
    {- The sum is computed on OCaml [int]s, which wrap modulo [2{^63}]. The
       64-bit word is the sum's two's complement, and the shift brings in
       zeros at the top, so the high half of [-8] is [0xFFFFFFFF]. The value
       sets bit 63 of a [W64] word iff it is negative and [shift] is [0]. A
       word whose bit 63 is a constant holds it in the bytes.}} *)

type 'a data = { bytes : string; holes : 'a hole iarray }
(** The type for bytes with holes. *)

(** The type for what a run knows: a word read when the step that needs it
    runs. *)
type value =
  | Fixed of int  (** The integer itself. *)
  | Int of int
      (** Word [i] of the run's ints ({!t.ints}), as it holds when read:
          after the frame's ints were written, the enclosing loops' trips
          stored and the run's earlier work that writes the ints is done. *)
  | Leaf of leaf

(** The type for how many copies of its memory a program keeps. *)
type copies =
  | One  (** One, which every run uses. *)
  | Two
      (** Two: run [n] uses copy [n mod 2], once the work of run [n - 2] on
          that copy is done. *)

(** The type for the areas of a rail's end ({!section-rails}). *)
type area = Outbound | Inbound | Counts

(** The type for a program's memory. *)
type memory =
  | Alloc of {
      device : int;
      kind : Rig.Buffer.kind;
      bytes : int;
      init : leaf data;
          (** Its first bytes, written at load into each copy, each hole
              filled for that copy. *)
      copies : copies;
    }  (** Memory made at load ({!Rig.Buffer.create}). *)
  | Rail of { rail : int; area : area }
      (** The area [area] of this machine's end of the rail of id [rail]:
          host memory the rail made, which the program borrows and every run
          uses ({!section-rails}). *)

type image = { device : int; binary : leaf data }
(** The type for images: [binary] in [device]'s format, loaded in the
    description's order ({!Rig.Image.load}). A hole may name an earlier
    image's entry, so an image can be a vendor object over loaded code, such
    as a CUDA graph over a module's functions, on a driver that loads such
    objects. *)

type view = { memory : int; offset : int; length : int }
(** The type for [length] bytes of memory [memory] from its byte [offset].
    For memory of two copies, the bytes are those of the copy the run uses. *)

(** The type for the buffers a step names. *)
type slot =
  | Memory of view
  | Input of int  (** The run's input [i], whole. *)
  | Ints  (** The run's ints, whole: [8 * ints] bytes of host memory. *)

(** The type for how a ref's work uses its slot, as {!Rig.Submission.read},
    {!Rig.Submission.write} and {!Rig.Submission.unordered}. *)
type use = Read | Write | Unordered

type ref = { slot : int; at : int; use : use }
(** The type for refs, as {!Rig.Submission.type-ref}. The work uses the
    submit's slot [slot] as [use] says, and the 8 parameter bytes at [at]
    hold an offset into that slot's buffer. *)

(** The type for parts, as {!Rig.Submission.type-part}. *)
type part =
  | Words of view
  | Fill of {
      fill : leaf;
          (** The function the fill calls: an [Entry] or a [Ready]. *)
      params : value data;
          (** Its parameters' bytes, a whole number of 32-bit words and at
              most 4096 bytes, each hole written into the run's block before
              each submit. *)
      refs : ref iarray;
    }
  | Copy of { src : int; dst : int }
      (** A copy from the submit's slot [src] into its slot [dst], as
          {!Rig.Submission.copy}. An end whose slot is [Memory] is that
          memory's buffer, set once at load. An end whose slot is an [Input]
          or the [Ints] is a [Slot] end ({!Rig.Submission.type-side}), whose
          buffer each run passes. *)
  | Launch of {
      image : int;
      name : string;
      params : value data;
          (** Its parameters' bytes, a whole number of 32-bit words and at
              most 4096 bytes, each hole written into the run's block before
              each submit. *)
      refs : ref iarray;
      groups : value * value * value;
      threads : value * value * value;
      shared : value;
    }

type submit = {
  device : int;
  parts : part iarray;
  slots : slot iarray;
  fixed : (view * Rig.access) iarray;
}
(** The type for submissions: [parts] on device [device]. Each run passes
    [slots], which the parts' refs and copies index, and every run uses the
    fixed memory [fixed] ({!Rig.Submission.make}). Memory of another device
    is named through its borrow on [device], made at load, and an input of
    another device through a borrow made for each run. *)

(** The type for the steps of a run. *)
type step =
  | Submit of submit  (** One {!Rig.submit}. *)
  | Move of { src : slot; dst : slot }  (** One {!Rig.Buffer.copy}. *)
  | Loop of {
      trips : value;
      trip : int option;
      flag : slot option;
      body : step iarray;
    }
      (** [body], [trips] times, none where [trips <= 0], storing each trip's
          index from [0] into word [trip] of the run's ints. With [flag], it
          reads the first byte of [flag]'s buffer before each trip, once the
          work that wrote it is done, and stops at [0]. *)

type input = { device : int; bytes : int }
(** The type for inputs: a buffer of [device] of at least [bytes] bytes. *)

type t = {
  devices : string iarray;  (** Each device's {!Rig.arch}, checked at load. *)
  memory : memory iarray;
  images : image iarray;
  inputs : input iarray;
  ints : int;
      (** The number of 64-bit words of the run's ints, host memory that
          {!load} makes in two copies ({!type-copies}). A frame writes the
          first words, loops store their trips there, and the work of its
          steps may write others. *)
  steps : step iarray;
}
(** The type for descriptions. *)

(** {1:bytes Bytes} *)

val to_string : t -> string
(** [to_string t] is [t] as bytes, in a format with a version of its own. *)

val of_string : string -> (t, string) result
(** [of_string s] is the description {!to_string} wrote as [s]. It only
    decodes, and {!load} checks the description.

    [Error why] if [s] is malformed, [why] saying where, or is of another
    version of the format, [why] naming both versions. *)

(** {1:loading Loading} *)

type loaded
(** The type for loaded programs. A loaded program keeps its memory, images
    and the rail ends [rails] gave it while it is reachable. Once it
    is unreachable, they return as any memory does, after the work of its
    last run ({!Rig.section-reclaim}). *)

val load :
  ?rails:(int -> Rig_remote_abi.end_ option) ->
  t ->
  Rig.t iarray ->
  (loaded, string) result
(** [load ~rails t devices] is [t] loaded on [devices], all of one machine
    ({!Rig.host_of}). Device [i] of [t] is [devices.(i)]. [rails id] is this
    machine's end of the rail [id] ({!Rig_remote_abi.type-rail}), and
    defaults to none. It loads the images in order and writes each [Alloc]
    memory's [init] last, their holes filled. A [Submit] step's submission
    borrows the memory the step names on its device, and stores the
    parameters and geometry that no run changes. On another machine, [load]
    checks [t] here and loads it through the agent ({!section-machines}).

    [Error why] if [t] breaks a rule its types or {!section-order} state, or
    if:
    {ul
    {- [devices] are not as many as [t]'s, or a device's {!Rig.arch} differs
       from [t]'s;}
    {- an index is outside its array, a size is negative, or an [init] or a
       view lies outside its memory;}
    {- a leaf in an image or in memory of one copy names memory of two
       copies, an [Entry] names a later image, or an [Address] is of memory
       its driver names by handle only;}
    {- a device that names memory cannot borrow it ({!Rig.Buffer.borrow});}
    {- an [Int] or a loop's [trip] is past the ints, or a loop's flag has no
       byte, as an input of no [bytes] or ints of no word;}
    {- a ref or a copy names no slot of its submit, a slot is named by no
       ref or copy, or a slot is named by both an [Unordered] ref and a
       [Read] or [Write] ref or a copy;}
    {- a copy's ends differ in length where load knows the lengths, which
       are a [Memory] view's [length], [8 * ints] for the ints, and at least
       [bytes] for an input;}
    {- a copy's two ends are both the ints, or both one input;}
    {- [Rail] memory or a [Ready] names a rail that [rails] does not give;}
    {- {!Rig.Submission.make} or a run ({!Rig.Submission.Run}) refuses a
       step, [why] naming the step;}
    {- a device refuses an image ({!Rig.Image.load}) or has no function an
       [Entry] names, [why] starting with the device's name;}
    {- on another machine, a device is not an agent's, or the agent's load
       is [Error why].}}

    Raises [Invalid_argument] if [devices] are of several machines, or are of
    another machine and [rails] is given; {!Rig.Out_of_memory} if memory
    cannot be made; and {!Rig.Lost} if a device is lost. *)

(** {1:running Running}

    A run {e writes} an input if a [Submit] step passes it in a slot that a
    [Write] or an [Unordered] ref or a copy's [dst] names, or a [Move] has it
    as its destination. *)

type frame = { inputs : Rig.Buffer.t array; ints : int array }
(** The type for a run's frame: its inputs, by index, and the first words of
    its ints. *)

val run : ?waits:Rig.Point.t array -> loaded -> frame -> Rig.Point.t array
(** [run ~waits p f] runs [p]'s steps once, in order, over the frame [f]
    ({!section-order}). It is the points the run's work ends at: for each
    device that ran a submission, in the order of [p]'s devices, the point of
    its last one. Each device's first submission follows the points of
    [waits] (defaults to none) and the device's earlier work.
    {ul
    {- It first waits for the work of run [n - 2] on the copies this run
       uses, the run's ints included, then writes [f]'s ints.}
    {- The work it does on the host, [Move] steps, [Submit] steps on the
       host and loops' flags, is done when it returns.}
    {- An [Int] value is read once the run's earlier work that writes the
       ints is done, and a loop stores its trip once the run's earlier work
       that uses the ints is done.}
    {- It reads [f] once, before its first step, so a change to [f] during
       the run changes nothing. It keeps [f]'s inputs, and their borrows on
       the other devices that name them, until it returns.}}
    On another machine, the raises at a step fail the job there instead
    ({!section-machines}).

    Raises [Invalid_argument]:
    {ul
    {- before any step runs, if [f] has another number of inputs than [p]
       or more ints, or if an input is not on its device, holds fewer bytes,
       is read-only memory ({!Rig.Buffer.val-access}) that the run writes,
       or cannot be borrowed by a device that names it;}
    {- at a step, after the earlier steps ran, if the run refuses a launch's
       geometry, shared memory or a parameter's value read from the ints
       ({!Rig.Submission.Run}), and as {!Rig.submit} and {!Rig.Buffer.copy}
       raise.}}
    Raises {!Rig.Lost} as the devices raise it. *)

(** {1:machines Programs of another machine}

    {!load} on another machine's devices sends that machine's host the
    description, with each device's id on that machine, as the binary of an
    image ({!Rig.Image.load}), and asks for the image's function ["run"]
    ({!Rig.Image.entry}). Each {!run} is then one submission on that host,
    whose one part is a {!Rig.Submission.words}: the function's value, the
    frame's inputs, each as its memory's id on the machine, its offset and its
    length, and the frame's ints. The submission names only the memory of
    this process that holds these words, and its point is reached once the
    run's work on that machine is done. The machine's agent runs it after the
    work submitted to the machine before it, and runs the work submitted
    after it once the run is done ([Rig_remote]'s order section).

    {!run} raises here, before the submission, for each refusal that comes
    before any step. It also raises [Invalid_argument] here for a launch
    whose threads per group exceed the most its function may have
    ({!Rig.Image.threads}), where each of its three sizes is a constant or a word of
    the frame's ints that no step may change, and at least one is such a
    word. A step may change the ints where a [Submit] passes [Ints] in a
    slot it writes or a [Move] writes [Ints], and a loop changes the word
    [trip] it stores its trips in. The agent loads and runs the program with
    the two functions below, and a run they refuse or raise on at a step,
    such as any other launch with too many threads, fails the job
    ([Rig_remote]'s failure section). *)

val load_received :
  ?rails:(int -> Rig_remote_abi.end_ option) ->
  string ->
  device:(int -> Rig.t option) ->
  (loaded, string) result
(** [load_received ~rails b ~device] is {!load}[ ~rails] of the description in
    [b], the binary another process's {!load} sent to this machine's host, on
    the devices that [device] gives for the ids [b] names.

    [Error why] as {!load}, or if [b] is malformed or names an id that
    [device] does not give. *)

val run_received :
  ?waits:Rig.Point.t array ->
  string ->
  program:(int -> loaded option) ->
  region:(int -> Rig.Buffer.t option) ->
  (Rig.Point.t array, string) result
(** [run_received ~waits w ~program ~region] is {!run}[ ~waits] of the program
    that [program] gives for the value [w] starts with, over the frame [w]
    holds. Each input is the view of [region id] at its offset and length.

    [Error why] if [w] is malformed, or [program] or [region] gives nothing
    for an id it names. Otherwise it raises as {!run}. *)

(**/**)

val threads : loaded -> int option array
(** [threads p] is, for each [Launch] part of [p]'s description in the
    order of its steps, with a loop's body in place, the most threads a
    group of its function may have, or [None] if its driver does not know it
    ({!Rig.Image.threads}). On another machine, it is what the machine's
    agent stated when it loaded [p] ({!section-machines}). *)
