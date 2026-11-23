(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Programs of steps on one machine's devices, as data that loads once and
    runs many times.

    A {e program} is the memory and images a computation needs on the devices
    of one machine, and the {e steps} each run of it takes: submissions,
    copies, loops and sends. Its {e description} ({!t}) is plain data that
    names each of these by its index in the description, so it travels
    between processes as bytes ({!to_string}). What changes from one run to
    the next is the run's {e frame} ({!type-frame}): its input buffers and the
    first words of its {e ints}, 64-bit integers in host memory that loops and
    the work of its steps also read and write. An integer of a step that a
    run may change, such as a launch's parameter or grid or a copy's offset,
    is a {e value} ({!type-value}): a sum of words of the ints, which the run
    computes when the step runs.

    The run's ints have two copies. Run [n] uses copy [n mod 2], so a run may
    write its ints while the previous run's work still reads the other copy.

    The life of a program goes like this:
    + A caller builds its description, in this process or in another one that
      sends it as bytes.
    + {!load} checks the description against the devices of one machine,
      makes its memory, loads its images and prepares a {!Rig.Submission} for
      each [Submit] step, one per copy of the ints where the step names them.
    + {!run} runs the steps once, in order, over a frame, and returns its
      ends, the points its work ends at. A loaded program runs any number of
      times.
    + Once the loaded program is unreachable, its memory and images return
      after the work of its last run.

    {v
      t ──load devices──> loaded ──run frame──> ends
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
    {!Rig.submit}'s does ({!Rig.section-order}). A [Submit] step names the
    memory of its [slots] and of its parts' views, and a [Move] the memory of
    its two slots. A step therefore follows the earlier steps whose memory it
    names, and its device's earlier work.

    Host code runs as a [Submit] step on the machine's host, whose launches
    are done when its submit returns ({!Rig.section-host_launches}).

    {1:rails Rails}

    A program reaches its machine's end of a rail
    ({!type:Rig_remote_abi.end_}) as [Rail] memory, in three areas:
    [Outbound] holds the bytes it sends, [Inbound] the bytes it receives, and
    [Counts] the counts that number the transfers. [Outbound] and [Inbound]
    each hold two copies of their bytes, and [Rail] memory covers both, so a
    step's views select the copy a transfer uses.

    No stamp records the rail's own reads and writes, so rig does not order
    them. A program orders them by the rail's counts, as
    {!Rig_remote_abi.section-rails} states. Its steps write a transfer's
    bytes into [Outbound], then a [Send] step tells the rail that the
    transfer is there, once the work that wrote them is done.

    {1:domains Domains}

    {!load}, {!run} and {!time} may be called from any domain. Runs of one
    loaded program share its memory, so a run waits until the previous one
    has run its steps before its first step. *)

(** {1:descriptions Descriptions} *)

type value = { add : int; terms : (int * int) iarray }
(** The type for integers a run computes from its ints: [add] plus the sum,
    over [terms], of each word [w] of the run's ints ({!t.ints}) times its
    [factor], for [(w, factor)]. The run reads each word when the step that
    needs the value runs: after the frame's ints were written, the enclosing
    loops' trips stored and the run's earlier work that writes the ints is
    done. A value is {e constant} if [terms] is empty: it is [add], which
    {!load} knows. The sum is exact. It starts from [add] and adds the terms
    in order, and the run refuses a product or a partial sum past an OCaml
    [int]: [{ add = 1; terms = [: (0, max_int); (1, -1) :] }] with both words
    [1] is refused, since [1 + max_int] is past an [int], though the sum is
    [max_int]. For example, the offset of a copy that a loop moves along rows
    of [n] bytes, the loop storing its trip in word [i], is
    [{ add = 0; terms = [: (i, n) :] }]. *)

(** The type for the widths of scalars. *)
type width = W32 | W64

type scalar = { at : int; width : width; value : value }
(** The type for scalars, parameters a run writes: [value] as the
    little-endian word of [width] bytes at byte [at] of a launch's
    parameters, over their constant bytes. A [W32] scalar takes a value that
    32 bits hold as a signed or an unsigned integer, from [-2{^31}] to
    [2{^32} - 1], and writes its low 32 bits. A [W64] scalar takes any value,
    in two's complement. No two scalars of a launch share a byte. *)

(** The type for the areas of a rail's end ({!section-rails}). *)
type area = Outbound | Inbound | Counts

(** The type for a program's memory. *)
type memory =
  | Alloc of {
      device : int;
      kind : Rig.Buffer.kind;
      bytes : int;
      init : string;  (** Its first bytes, written at load. *)
    }  (** Memory made at load ({!Rig.Buffer.create}). *)
  | Rail of { rail : int; area : area }
      (** The area [area] of this machine's end of the rail of id [rail]:
          host memory the rail made, which the program borrows and every run
          uses ({!section-rails}). *)

type image = { device : int; binary : string }
(** The type for images: [binary] in [device]'s format ({!Rig.Image.load}). *)

type view = { memory : int; offset : int; length : int }
(** The type for [length] bytes of memory [memory] from its byte [offset]. *)

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
    hold an offset into that slot's buffer. A [W64] scalar at [at] makes the
    offset the run's: with [{ at; width = W64; value = { add = 0; terms = [:
    (i, n) :] } }], the ref reaches row [r] of rows of [n] bytes, where word
    [i] of the run's ints holds [r], such as the trip a loop stores there. *)

(** The type for parts, as {!Rig.Submission.type-part}. *)
type part =
  | Words of view
  | Copy of {
      src : int;
      src_offset : value;
      dst : int;
      dst_offset : value;
      length : int;
    }
      (** A copy of [length] bytes from byte [src_offset] of the submit's slot
          [src] into byte [dst_offset] of its slot [dst], as
          {!Rig.Submission.copy}. The offsets are computed when the step runs,
          so a copy in a loop may move with the loop's trips. An end whose
          slot is [Memory] and whose offset is constant is those bytes of the
          memory, set once at load. Any other end is a [Slot] end
          ({!Rig.Submission.type-side}), whose buffer each run passes. *)
  | Launch of {
      image : int;
      name : string;  (** The function of image [image] it runs. *)
      label : string;
          (** The name of its work in profiles and refusals
              ({!Rig.Submission.launch}), which tells apart launches of one
              function. *)
      params : string;
          (** Its parameters' constant bytes, a whole number of 32-bit
              words. *)
      scalars : scalar iarray;
          (** Written over [params] into the launch's values: at load for
              those whose value is constant, and before each submit for the
              others. *)
      refs : ref iarray;
      groups : value * value * value;
      threads : value * value * value;
      shared : value;
    }

type submit = { device : int; parts : part iarray; slots : slot iarray }
(** The type for submissions: [parts] on device [device]. Each run passes
    [slots], which the parts' refs and copies index. Memory of another device
    is named through its borrow on [device], made at load, and an input of
    another device through a borrow made for each run. *)

(** The type for the steps of a run. *)
type step =
  | Submit of submit  (** One {!Rig.submit}. *)
  | Move of {
      src : slot;
      src_offset : value;
      dst : slot;
      dst_offset : value;
      length : int;
    }
      (** One {!Rig.Buffer.submit_copy} of [length] bytes from byte
          [src_offset] of [src]'s buffer into byte [dst_offset] of [dst]'s.
          The offsets are computed as a [Copy] part's, so a move in a loop
          may move with the loop's trips. Between the program's host memory,
          the ints included, and memory of another of its devices, the move
          names the host memory's borrow on that device, made at the first
          such move and kept with the program, so that the device copies it
          directly and the host does not wait for the bytes
          ({!Rig.Buffer.section-copying}). Host memory a move names is made so
          that a device can borrow it ({!Rig.Buffer.create}). Where the device
          does not borrow it, or has no room to map it, the program's moves
          through it pass through staging memory. *)
  | Loop of {
      trips : value;
      trip : int option;
      flag : (slot * int) option;
      body : step iarray;
    }
      (** [body], [trips] times, none where [trips <= 0], storing each trip's
          index from [0] into word [trip] of the run's ints. With
          [flag = Some (s, at)], it reads byte [at] of [s]'s buffer before
          each trip, once the work that wrote it is done, and stops at [0]. *)
  | Send of { rail : int; count : value }
      (** Tells this machine's end of the rail of id [rail] that the transfer
          of count [count], at least [1], lies in its [Outbound]
          ({!Rig_remote_abi.section-rails}). Once the points of the run's
          [waits] ({!run}) are reached and the work that writes the
          program's [Rail] memory of that [Outbound] is done, it calls the
          end's [ready] with [count] on the host. *)

type input = { device : int; bytes : int }
(** The type for inputs: a buffer of [device] of at least [bytes] bytes. *)

type t = {
  devices : string iarray;
      (** The {!Rig.arch} each device's code is compiled for, checked at
          load: a device runs code of its own arch, a host of an x86-64
          level also runs code of each level below it, and an arm64 host
          code of each arm64 arch whose features are among its own. *)
  memory : memory iarray;
  images : image iarray;
  inputs : input iarray;
  ints : int;
      (** The number of 64-bit words of the run's ints, host memory that
          {!load} makes in two copies. A frame writes the first words, loops
          store their trips there, and the work of its steps may write
          others. *)
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

val footprint : t -> Rig.t iarray -> (Rig.t * int) list
(** [footprint t devices] is, for each of [devices] and the host, the bytes
    {!load}[ t devices] on this machine makes there and counts in each device's
    budget, as {!Rig.allocation} counts them, making nothing: each [Alloc]
    memory, host memory a move names taking {!Rig.Buffer.paged_from} bytes at
    least so that a device can borrow it, and the run's ints, kept twice on the
    host, as much each at least where a step names them, each in the budget
    {!Rig.charged} names for its kind. [Mapped] memory that falls back at load
    ({!Rig.charged}) is counted as [Mapped]. Loaded images' code and a rail's
    memory are not counted. *)

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
    defaults to none. It loads the images and writes each [Alloc] memory's
    [init] last. A [Submit] step's submission borrows the memory the step
    names on its device, and its launches' values hold the parameters and
    geometry whose values are constant. On another machine, [load] checks
    [t] here and loads it through the agent ({!section-machines}).

    [Error why] only for what a device refuses:
    {ul
    {- a device refuses an image ({!Rig.Image.load}), or has no function a
       [Launch] names, or cannot run it ({!Rig.Image.resolve}), [why]
       starting with the device's name;}
    {- a device that names memory cannot borrow it ({!Rig.Buffer.borrow});}
    {- {!Rig.Submission.make} or a part's constructor refuses a step, or a
       launch's grid, threads or shared memory is, from its constant sizes
       alone, one that every {!Rig.submit} refuses ({!Rig.section-valid}),
       [why] naming the step;}
    {- on another machine, the agent's load is [Error why].}}

    Raises [Invalid_argument] if [t] breaks a rule its types state, which
    [load] checks before it makes anything. Among them:
    {ul
    {- [devices] are not as many as [t]'s, or a device does not run code of
       [t]'s arch for it: its {!Rig.arch} differs, and is neither an x86-64
       level at or above [t]'s nor an arm64 arch with each of [t]'s
       features;}
    {- an index is outside its array, a size is negative, or an [init] or a
       view lies outside its memory;}
    {- a launch's parameters are not a whole number of 32-bit words, a
       scalar lies outside them, two scalars share a byte, or the width of a
       scalar whose value is constant does not hold it;}
    {- a value reads a word past the ints, a loop's [trip] is past them, or
       a loop's flag lies at a negative byte or past its slot where load
       knows the slot's length: a [Memory] view's [length] and [8 * ints] for
       the ints;}
    {- a ref or a copy names no slot of its submit, a slot is named by no
       ref or copy, or a slot is named by both an [Unordered] ref and a
       [Read] or [Write] ref or a copy;}
    {- a copy's or a move's [length] is negative, or an end's constant
       offset is negative or puts the end past its slot where load knows the
       slot's length: a [Memory] view's [length] and [8 * ints] for the ints;}
    {- a copy's or a move's two ends share a byte of one memory, of the ints
       or of one input, at constant offsets;}
    {- a [Send]'s count is constant and below [1];}
    {- [Rail] memory or a [Send] names a rail that [rails] does not give;}
    {- on another machine, a device is not an agent's.}}
    Raises [Invalid_argument] also if [devices] are of several machines, or
    are of another machine and [rails] is given, or, having done nothing,
    from a callback inside a lock ({!Rig.section-domains});
    {!Rig.Out_of_memory} if memory cannot be made; and {!Rig.Lost} if a
    device is lost. *)

(** {1:running Running}

    A run {e writes} an input if a [Submit] step passes it in a slot that a
    [Write] or an [Unordered] ref or a copy's [dst] names, or a [Move] has it
    as its destination. *)

type frame = { inputs : Rig.Buffer.t array; ints : int array }
(** The type for a run's frame: its inputs, by index, and the first words of
    its ints. *)

val run : ?waits:Rig.Point.t array -> loaded -> frame -> Rig.Point.t array
(** [run ~waits p f] runs [p]'s steps once, in order, over the frame [f]
    ({!section-order}). It is the run's {e ends}, the points its work ends
    at: for each device that ran a submission or a move, in the order of
    [p]'s devices, the point of its last one. A move runs on the device of
    the point {!Rig.Buffer.submit_copy} returns for it. Each device's first
    submission, and each move, follows the points of [waits] (defaults to
    none) and the device's earlier work, and each [Send] calls its end once
    they are reached.
    {ul
    {- It first waits for the work of run [n - 2] on its copy of the ints,
       then writes [f]'s ints.}
    {- The work it does on the host, [Submit] steps on the host, moves the
       host makes, loops' flags and [Send] steps, is done when it returns. A
       move whose point is of a device outside [p]'s is done at its step,
       which waits for it.}
    {- A value's words are read once the run's earlier work that writes the
       ints is done, and a loop stores its trip once the run's earlier work
       that uses the ints is done.}
    {- It reads [f]'s inputs and ints once, before its first step, so a
       change to them during the run changes nothing. It keeps [f]'s inputs,
       and their borrows on the other devices that name them, until it
       returns.}}
    On another machine, the raises at a step fail the job there instead
    ({!section-machines}).

    Raises [Invalid_argument]:
    {ul
    {- before any step runs, if [f] has another number of inputs than [p] or
       more ints, or if an input is not on its device, holds fewer bytes, is
       read-only memory ({!Rig.Buffer.val-access}) that the run writes, or
       cannot be borrowed by a device that names it, if it is called on the
       thread that runs [p], as from a callback inside that run, or from a
       callback inside a lock ({!Rig.section-domains});}
    {- at a step, after the earlier steps ran, if a value's sum is past an
       [int], if a scalar's width does not hold its value, if a copy's or a
       move's offset puts its end outside its slot's buffer, if a [Send]'s
       count is below [1], if a loop's flag lies past its input's bytes, and
       as {!Rig.submit} and {!Rig.Buffer.submit_copy} raise, for a launch's
       geometry that the run sets among others.}}
    Raises {!Rig.Lost} as the devices raise it. *)

val time : ?waits:Rig.Point.t array -> loaded -> frame -> int array
(** [time ~waits p f] runs [p] over [f] as {!run} does, with the work of its
    steps timed, and returns once that work is done. It is, for each of [p]'s
    steps in step order, the time in nanoseconds its device took for the
    step's work: from when the device started the work of the step's
    submission to when it ended it, on the host's clock ({!Rig.Profile.now}),
    as exact as the device's driver correlates the device's clock with the
    host's, which its interface states. A step whose device did not time its
    work, as its driver's interface states, has [-1]. A step with no work to
    time, a [Move], a [Send] or a [Submit] whose submission has no parts, has
    [0], and a timed step may have [0] too.

    Only the submits of the run's steps are timed, so other runs and domains
    pay nothing for it. Timing costs each device what its driver's interface
    states. On a device whose driver measures its launches' counters, a
    timed launch is measured too.

    On another machine, the times are read there and come back
    ({!section-machines}).

    Raises [Invalid_argument], before any step runs, if [p] has a [Loop]
    step. Raises as {!run} does, and {!Rig.Lost} if a device is lost, or
    closed ({!Rig.close}), before [time] has read its times. *)

(** {1:machines Programs of another machine}

    {!load} on another machine's devices sends that machine's host the
    description, with each device's id on that machine, as the binary of an
    image ({!Rig.Image.load}), and asks for the image's function ["run"]
    ({!Rig.Image.resolve}). Each {!run} is then one submission on that host,
    whose one part is a {!Rig.Submission.words}: the function's value, the
    {e times word}, [-1] for a run, the frame's inputs, each as its memory's
    id on the machine, its offset and its length, and the frame's ints. The
    submission names only the memory of this process that holds these words,
    and its point is reached once the run's work on that machine is done.
    The machine's agent runs it after the work submitted to the machine
    before it, and runs the work submitted after it once the run is done
    ([Rig_remote]'s order section).

    For a program without a loop that has a [Submit] step, {!load} also
    makes 8 bytes per step on that host. A {!time} makes as many in this
    process's host memory, and submits the same words, the times word the id
    of the memory on the machine, and a copy of it into its own. The agent
    times the run, writes each step's time there as a little-endian 64-bit
    integer, and the copy brings them back. Times share the memory on the
    machine, since the agent runs a later run's words once the copy is done.
    A program without a [Submit] step is timed as it runs, and has only
    [0]s.

    {!run} raises here, before the submission, for each refusal that comes
    before any step. It also raises [Invalid_argument] here for a launch
    whose threads per group exceed the most its function may have
    ({!Rig.Image.threads}), where its three sizes read only words of the
    frame's ints that no step may change and at least one size reads one,
    as every {!Rig.submit} refuses them: [x] by [y] past the most whatever
    [z] is, or the three. A size whose sum does not fit an [int] is left to
    the agent. A step may change the ints where a [Submit] passes [Ints] in
    a slot it writes or a [Move] writes [Ints], and a loop changes the word
    [trip] it stores its trips in. The agent loads
    and runs the program with the two functions below, and a run they refuse
    or raise on at a step, such as any other launch with too many threads,
    fails the job ([Rig_remote]'s failure section). *)

val load_received :
  ?rails:(int -> Rig_remote_abi.end_ option) ->
  string ->
  device:(int -> Rig.t option) ->
  (loaded, string) result
(** [load_received ~rails b ~device] is {!load}[ ~rails] of the description in
    [b], the binary another process's {!load} sent to this machine's host, on
    the devices that [device] gives for the ids [b] names.

    [Error why] as {!load}, and where {!load} raises [Invalid_argument] for
    a rule the description breaks, since another process made [b]; or if [b]
    is malformed or names an id that [device] does not give. *)

val run_received :
  ?waits:Rig.Point.t array ->
  string ->
  program:(int -> loaded option) ->
  region:(int -> Rig.Buffer.t option) ->
  (Rig.Point.t array, string) result
(** [run_received ~waits w ~program ~region] is {!run}[ ~waits] of the program
    that [program] gives for the value [w] starts with, over the frame [w]
    holds. Each input is the view of [region id] at its offset and length.
    Where [w]'s times word is an id, it is {!time}[ ~waits] instead, which
    writes each step's time into [region id] as a little-endian 64-bit
    integer, and is the run's ends ({!section-machines}).

    [Error why] if [w] is malformed, or [program] or [region] gives nothing
    for an id it names, or the times' memory is not 8 bytes per step.
    Otherwise it raises as {!run} or {!time}. *)

(**/**)

val threads : loaded -> int option array
(** [threads p] is, for each [Launch] part of [p]'s description in the
    order of its steps, with a loop's body in place, the most threads a
    group of its function may have, or [None] if its driver does not know it
    ({!Rig.Image.threads}). On another machine, it is what the machine's
    agent stated when it loaded [p] ({!section-machines}). *)
