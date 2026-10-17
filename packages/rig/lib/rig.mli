(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices such as GPUs, and the order of the work they run.

    A {e device} holds memory and, unless it is an io device, runs work on
    it. The {!host} is this process's device. A driver's device, such as a
    GPU, runs the work its driver places on the device's queues, and a
    program opens it with the opener the driver supplies ({!open_}). An
    {e io device}, such as a file, holds memory that the host reads and
    writes through its library's calls ({!open_io}). A program uses the
    devices of another machine through that machine's host
    ({!section-machines}).

    Most programs need only {!open_}, {!Buffer} and {!Buffer.copy}. {!Image},
    {!Submission}, {!submit}, {!Local} and {!Claim} are for libraries that
    run compiled code, keep values per device or write results over their
    inputs.

    Work on a GPU runs asynchronously. The call that submits it returns
    before the work has run. So that the host can tell what has run, rig
    numbers the work submitted to a device with consecutive integers, in the
    order the device accepts it. These numbers are the {e values} of the
    device's {e timeline}. A value [v] is {e reached} once the work of every value up
    to [v] has completed. A {e point} is a device together with one of its
    values ({!Point}). Work or host access {e follows} other work when it
    starts after that work has completed, and so sees its writes. It follows
    a point when it starts after the point is reached.

    An {e allocation} is the memory that one call, such as {!Buffer.create},
    makes on a device. A {e buffer} is a range of bytes in an allocation, and
    a {e borrow} is a buffer on another device over the same allocation,
    without a copy ({!Buffer.borrow}). Each allocation records, for each
    device, the point of its last write and the point of its last use.
    These points are its {e stamps}.

    rig orders work and host access by the stamps of the memory they use
    ({!section-order}). Two pieces of work need an order only when they
    share memory, so the order lives with the memory and no call names an
    event or a fence. {!submit} returns the point of the work it runs, and a
    library passes points to later work only to order it after work that
    shares no memory with it. An allocation returns to its device some time
    after no buffer over it, and no array made of it, is reachable
    ({!section-reclaim}).

    A function may write its result over the memory of an argument it no
    longer needs, which is called {e donation}. A {e claim} records that
    something reads an allocation, so that a donation writes only memory
    that nothing else reads ({!section-claims}). A donor writes over memory
    that it holds {e exclusive}, by the only claim on it. It first
    {e consumes} that memory, so every buffer made over the memory until
    then is {e dead} from then on ({!Claim.consume}).

    A device is {e lost} when it can run no more work, for good, such as
    after a fault of its driver ({!section-loss}). Unless a function's doc
    says otherwise:
    {ul
    {- a call that takes a buffer raises [Invalid_argument] if the buffer is
       dead;}
    {- a call that uses a device or a buffer's memory raises {!Lost} where
       {!section-loss_memory} says;}
    {- a call that allocates raises {!Out_of_memory} if its device cannot
       provide the bytes ({!section-reclaim});}
    {- a call may be made from any domain, concurrently with any other,
       and one that may take a lock raises [Invalid_argument], having done
       nothing, when it is made inside a lock ({!section-domains}).}}

    This copies five bytes to a device and back. The blit back reads [dev],
    so it follows the blit that wrote it.
    {[
      let d = Result.get_ok (Rig.open_ opener) in
      let dev = Rig.Buffer.create d 5 in
      Rig.Buffer.blit_from_string "hello" 0 dev 0 5;
      let b = Bytes.create 5 in
      Rig.Buffer.blit_to_bytes dev 0 b 0 5
    ]}

    The sections below state every rule. {!section-timeline},
    {!section-order} and {!section-domains} say how work is ordered,
    {!section-loss} what a lost device leaves, and {!section-reclaim} when
    memory returns. The module reference comes after them: {!Buffer} for
    memory and host access, {!section-submitting} for compiled code,
    {!Local} for values a library keeps per device, and {!Claim} for writing
    over an argument. A driver implements {!Rig_edge}. *)

(** {1:devices Devices}

    A machine has at most one open device per name. *)

type t
(** The type for devices. *)

val host : t
(** [host] is this process's device, named ["CPU"]. Its memory is the
    process's memory, and its work is the process's own code: host access,
    and launches of code it links, which run on its cores
    ({!section-host_code}). *)

(** {2:opening Opening and closing}

    An open returns [Error why] when the device cannot be opened, and raises
    [Invalid_argument] when it is misused. An open of a lost device's name
    first runs the device's owed stop ({!section-stops}). Every open is
    [Error] after the process has failed ({!fail}), while the name's
    device is lost and its driver's stop ({!Rig_edge.Driver.val-stop}) has
    not returned, when the machine's host ({!section-machines}) is lost or
    closed, and while 65,535 devices hold every device slot. For a lost or
    closed host, [why] is what that host's {!Lost} prints.

    A device holds a slot of the process's 65,535, a bound that comes from a
    point fitting an OCaml int ({!Point}). It holds its slot while it is
    open, while memory of it or a value its submissions keep remains to
    return, and while it is lost and has not stopped ({!section-stops}). A
    device that never stops, the process's first {!failure}, and a device
    whose timeline passed 2{^ 46} hold theirs for the rest of the process. A
    device that opens takes a free slot, and its values go on past every
    value the slot's earlier devices were given, so the number of opens a
    process makes has no bound. Each open still costs what its driver keeps
    for the process, as the driver's documentation states. The GPU drivers
    keep a few dozen bytes of C state per open. A device whose queue waits
    on another device's word through a mapping keeps one such mapping for
    each slot it waits on, of the slot's device it waited on last. One of an
    earlier device of that slot goes back at the first of three calls after
    the device's queue has passed the waits that read it. These are the
    device's next submit whose queue waits on a later device of a slot it
    waited on, a call that moves its collected memory ({!section-reclaim}),
    and its stop. A timeline word of a device's own goes back at the first
    call that moves collected memory after the device has left its slot and
    no other device maps it. A device lost before it reached every value it
    was given leaves its slot too, and keeps its record for the rest of the
    process, which its dropped points raise {!Lost} with.

    An exception raised inside an open, by the opener's [make] or
    asynchronously ({!section-errors}), is raised again. The name is then
    unopened, or open with the device the open made, which the next open
    returns. A handle that [make] returned and that no device holds is
    stopped once ({!Rig_edge.Driver.val-stop}). *)

val open_ : ?machine:t -> 'a Rig_edge.opener -> (t, string) result
(** [open_ ~machine o] is the open device named [o.name] on [machine]'s
    machine (defaults to this machine). If none is open there, [o.make ()]
    opens one. It is [Error why] if [o.make ()] is [Error why]. It is
    [Error ("NAME: " ^ reason)] if reading the new device's facts
    ({!Rig_edge.type-facts}) raises its driver's [Fault reason].

    Raises [Invalid_argument] if the device open under that name is another
    driver's, if the facts break a rule of {!Rig_edge.type-facts}, or if
    called on the thread that runs an open of that name, from the opener's
    [make] or a callback inside that open ({!section-domains}). *)

val open_io : ?machine:t -> 'a Rig_edge.io_opener -> (t, string) result
(** [open_io ~machine o] is {!open_} for an io device ({!Rig_edge.Io}). *)

val memory_device : string -> (t, string) result
(** [memory_device name] is {!open_} of the {e memory device} named [name] on
    this machine. A memory device's memory is its own, in host memory that
    this process addresses. Its work is this process's, and it has a
    timeline of its own. It runs fills and copies, which run before
    {!submit} returns, loads no code and times no work. Tests use it to
    stand for a GPU in code that orders work across devices. It is
    [Error why] where {!open_} is.

    Raises as {!open_} does. *)

val close : t -> unit
(** [close d] waits for the work submitted on [d] before the call, then loses
    [d] with the reason ["closed"] and returns once its driver's stop
    ({!Rig_edge.Driver.val-stop}) has returned, which may be before [d] has
    stopped ({!section-stops}). From then on, uses of [d] raise {!Lost}.
    Once [d] has stopped, buffers over its memory return that memory once
    collected, and [d]'s mappings of other memory ({!Buffer.borrow}) go back:
    at the stop if [d] has stopped by then, and otherwise at
    {!free_cache}[ d], a {!Buffer.create} or {!Buffer.borrow} on [d], or
    through reclamation ({!section-reclaim}). Work that another domain
    submits on [d] during the close may be lost with [d].
    {ul
    {- On another machine's host ({!host_of}), lost or not, [close] first
       closes every device of that machine, including those still opening,
       and opens on that machine are [Error] from then on.}
    {- On any other lost [d], [close] only runs its owed stop, or waits for
       the call that runs it ({!section-stops}). An io device ends
       immediately.}
    {- A close that [Sys.Break] interrupted finishes when [close] is called
       again.}}
    A close is never the process's {!failure}.

    Raises [Invalid_argument], leaving [d] as it was, if [d] is {!host}, or
    if [d] is a machine's host and the call runs on the thread of an open on
    that machine, from its opener's [make] or a callback inside it
    ({!section-domains}). *)

(** {2:properties Properties}

    A device's properties are set when it opens, and keep their values once
    it is lost. A device's properties, timeline and reason live as long as
    the program holds the device. Its points name it until a later device
    takes its slot ({!Point.val-device}), and answer {!Point.wait} and
    {!Point.reached} for good. Its driver's stop releases what the device
    needs only to run work, such as its queues, and once its work has ended
    rig gives back its mappings of other devices' words and memory. *)

val name : t -> string
(** [name d] is [d]'s name for people: its opener's [name], with
    ["@MACHINE"] appended for a device of another machine. *)

val equal : t -> t -> bool
(** [equal d d'] is [d == d']: each device is one value for the life of the
    process. A reopened name is another device ({!section-loss}). *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a device's {!val-name}. *)

val arch : t -> string
(** [arch d] is the architecture of [d]'s processor as its driver names it,
    such as ["gfx1100"], ["sm_89"] or ["Apple7"]; the machine's instruction
    set for a host, ["arm64"] or ["x86_64"]; and [""] for an io device. *)

val computes : t -> bool
(** [computes d] is [true] iff [d] runs work, as every device but an io
    device does. *)

val runs_on_host : t -> bool
(** [runs_on_host d] is [true] iff [d]'s work is this process's code, which
    holds for {!host} and a {!memory_device}. The host computes on such a
    device's memory as on its own. *)

val alignment : t -> int
(** [alignment d] is the power of two that the first byte of every non-empty
    buffer {!Buffer.create} makes on [d] starts on, whatever its kind and
    size, as [d]'s driver states it from its allocator
    ({!Rig_edge.type-facts}): [16] for {!host}, [64] for a {!memory_device},
    and [1] for an io device. {!Buffer.create} starts a host buffer of 128
    bytes or more on 128 bytes or a page, multiples of this alignment that
    hold for those sizes only. For a device of another machine, it is that
    machine's device's. *)

val most_parts : t -> int
(** [most_parts d] is the most parts that an idle [d] always runs in one
    submission, given memory for its queues to grow ({!submit}), as [d]'s
    driver states it ({!Rig_edge.type-facts}): [max_int] where its driver
    has no such bound, and [0] for an io device. For a device of another
    machine, it is that machine's device's, but [max_int] for an io device
    there, since its agent runs each part handed to it as a copy of its own.

    The count holds for launches and fills up to the argument size, and
    copies up to the length, that [d]'s driver's [.mli] states. Words parts,
    and larger launches and copies, take room by their sizes, which the
    caller sets. A submission of more parts may still run: {!submit} decides
    by the room [d]'s queues have. *)

val capability : t -> 'a Type.Id.t -> 'a option
(** [capability d k] is the record that [d]'s driver declares under [k]
    ({!Rig_edge.type-capability}), filled when [d] opened, or [None] if it
    declares none under [k]. *)

(** {2:machines Machines}

    A {e machine} is a host and the devices it drives. A library that uses
    another computer opens that machine's host with {!open_host}, then the
    machine's devices with {!open_}'s [machine]. A library must give each
    machine it connects to a name of its own for the life of the process,
    and treat a second connection to one address as another machine.
    [Rig_remote] runs jobs over the devices of several machines, and states
    how a job fails. *)

val open_host : machine:string -> 'a Rig_edge.opener -> (t, string) result
(** [open_host ~machine o] is {!open_} of the host of the machine named
    [machine], a driver's device of that machine that runs submissions and
    loads the code its driver takes. A machine has one host for its life, so
    [open_host] of a machine whose host is lost or closed is [Error].

    Raises [Invalid_argument] as {!open_}, or if [machine] has a host of
    another name that is still opening, or open and not lost. *)

val host_of : t -> t
(** [host_of d] is the host of [d]'s machine: {!host} for this machine's
    devices, and the device {!open_host} opened for another machine's. A
    host is its own host. The loss of another machine's host loses none of
    its devices. *)

val reaches : t -> t -> bool
(** [reaches d d'] is [true] iff [d]'s work can address [d']'s own memory
    ({!Buffer.Device}) through a borrow on [d] ({!Buffer.borrow}). A
    compiler that places copies asks it before any buffer exists.
    {ul
    {- A device reaches its own memory.}
    {- {!host} reaches the memory of a {!memory_device}, and of a driver's
       device whose driver declares [host_addresses]
       ({!Rig_edge.type-facts}).}
    {- Every other device whose driver declares [maps_host], as a memory
       device's does, reaches {!host}'s memory and every memory device's
       memory.}
    {- Every other device reaches the memory of another device of its own
       driver that it maps ({!Rig_edge.Driver.peer}).}
    {- Devices of different machines never reach each other. An io device
       and any other device never reach each other.}}
    Another machine's host is a driver's device of that machine, and these
    rules hold for it there.

    A borrow on [d] can map the memory of every device that [d] reaches. A
    device that reaches {!host}'s memory also maps other memory of its
    machine that the host addresses and that starts on a page, including an
    io device's memory through its pages ({!Rig_edge.Io.pages}). {!host} and
    memory devices also map host memory that starts elsewhere, such as a host
    buffer under 64 KiB. *)

(** {1:timeline Timelines}

    {!submit} assigns the values of a device's timeline, and so do
    {!Buffer.copy} and {!Image.load} for the copies the device runs, in the
    order the device accepts the work, whichever domain submits it. The
    device's {e submitted} value is the last value assigned on its timeline
    ({!submitted}). A device reports the values it has reached
    ({!signaled}), and a lost device may report more ({!section-loss}).

    A driver may hold submitted work until it is {e committed}, released by
    the driver to the device, and a device runs only committed work. Every
    wait for a point, on the host or in another device's queue, first
    commits the work it waits for. A driver also commits on its own at least
    once every [k] values, where [k] is a constant of the driver's that its
    interface states, such as 256 for Metal ({!Rig_edge.section-calls}). rig
    does not read [k]. A device so reports each value at most [k] values
    late while work is submitted, and a program never commits by hand.
    {!signaled} commits nothing, so the last values submitted may stay
    unreported until a wait or {!Point.reached} commits them. *)

(** Points on timelines.

    A point is an immediate value, so storing one in an array or a record
    allocates nothing. *)
module Point : sig
  type device := t

  type t [@@immediate]
  (** The type for points. *)

  val device : t -> device option
  (** [device p] is [Some d], [p]'s device [d], until [d] has stopped and a
      later device took its slot ({!section-opening}), and [None] from
      then on. *)

  val value : t -> int
  (** [value p] is [p]'s value, above the value its device's timeline
      started after ({!submitted}). *)

  val wait : t -> unit
  (** [wait p] returns once [p]'s device has reached [p]'s value. It commits
      the device's work up to that value, then blocks however long that work
      runs, unless the device has a hang bound ({!section-hangs}). While it
      blocks it releases the domain lock ({!section-domains}). It returns at
      once for a point its device reached before a loss, whatever device its
      slot holds now.

      Raises {!Lost} if the device was lost before it reached [p]'s value,
      or is lost during the wait before it does. *)

  val reached : t -> bool
  (** [reached p] is [true] iff [p]'s device has reached [p]'s value, as
      {!wait} would find it, also once a later device took [p]'s slot.
      Otherwise it commits the device's work as {!wait} does, so a loop on
      it ends once that work runs. The commit waits for no work. It waits up
      to 200 ms for another submit or commit on the device, and blocks while
      a driver that declares [may_block] commits.

      Raises {!Lost} if the device was lost before it reached [p]'s value. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a point as [NAME:v], or [ended:v] once {!val-device} is
      [None]. *)
end

val submitted : t -> int
(** [submitted d] is the last value assigned on [d]'s timeline
    ({!section-timeline}), or, before any, the value [d]'s timeline started
    after: [0] for a device that took a slot no device took before, and the
    last value the slot's earlier device was given otherwise. It never
    decreases. *)

val signaled : t -> int
(** [signaled d] is the last value [d] has reported. While [d] is live,
    [d] has reached it, and the writes of the work up to it are visible to
    the host. On a lost [d] it may move on to dropped values
    ({!section-loss}). It commits no work and waits for no work
    ({!section-timeline}).

    Raises {!Lost} if reading the value through [d]'s driver loses [d], as
    for a device of another machine ({!Rig_edge.Driver.signaled}), or if
    [d] is lost while that read starts. *)

(** {1:order Order}

    Follows is transitive. These rules say which work and host code follow
    which:
    {ol
    {- {e Device order.} On one device, the work of value [v] follows the
       work of every value below [v], on every queue.}
    {- {e Parts.} Within a submission, a part follows each earlier part that
       may share memory with it, where one of the two writes.
       {!Submission.make} says which parts may share memory and which queue
       each runs on. A {!Submission.words} part, whose effect rig cannot see,
       also follows every earlier part, and a later part on its queue follows
       it. Parts that need no order may run concurrently.}
    {- {e Memory.} Work that reads an allocation follows every device's last
       write of it. Work that writes it follows every use of it, by any
       device.}
    {- {e Waits.} The work of {!submit} follows the points of its [waits].}
    {- {e Host access.} Host code after {!Point.wait}[ p], or after
       {!Point.reached}[ p] answers [true], follows [p]'s work.
       {!Buffer.read}'s function follows every device's last write of the
       buffer's memory, and {!Buffer.write}'s every use of it, as host code
       after {!Buffer.wait}[ b Read] or [b Read_write] does. {!Buffer.copy},
       {!Buffer.blit_to_bytes}, {!Buffer.blit_from_string} and
       {!Buffer.fill} order themselves the same way, as work that reads the
       buffer they copy from and writes the buffer they write.}
    {- {e Host writes.} Work submitted after a host write to memory that its
       device addresses sees the write. For another GPU's memory that the
       device addresses through a borrow, this rests on peer access between
       GPUs, whose status each driver's interface states.}}

    Rule 3 works through each allocation's stamps. Once {!submit} has
    assigned its value [v] on [d], it hands the work to [d]'s driver and adds
    [(d, v)] to the stamps of every allocation the work uses, also when the
    driver fails to take the work and [d] is lost. A copy that a device runs
    does the same. Work on {!host} is done when its submit returns, so it
    adds no point. A later access reads the stamps of the memory it uses and
    follows the points they name. Where the device's queues cannot wait for
    another device's point, {!submit} waits for it on the host, so a submit
    may block.

    Work through an {!Submission.unordered} ref is the one exception to
    rule 3. It follows no other work on the allocation beyond its device's
    order, and no later work or host access follows it. The program orders
    it against other work on the memory, such as with atomics on flags that
    the work of two devices spins on.

    An access follows the points that the stamps hold when it reads them, so
    the stamps order two calls only where the program orders the calls
    themselves. If two domains concurrently submit work on two devices that
    uses one allocation, and either work writes it, nothing makes either
    follow the other. Work and host access that start after both submits
    have returned follow both. Bytes that both works write hold either
    work's values or a mix of them. A program that needs one work to follow
    the other orders the two, such as by starting the second submit after
    the first has returned. In the same way, a wait orders the host after
    the work submitted before it returns, and after nothing that another
    domain submits later. On one device, device order orders all work. *)

(** {1:domains Domains}

    Calls on several domains run concurrently, and the stamps order them only
    as far as {!section-order} states. Host access to one buffer from several
    domains needs synchronization of the program's own, as for any shared
    bigarray.

    While a function waits it holds no lock of rig's, with three
    exceptions:
    {ul
    {- {!open_} holds its name's lock while the opener runs, so opens of one
       name run one at a time while opens of other names go on.}
    {- A {!Buffer.copy} through staging memory holds one of that staging
       memory's two slots until its pieces are done, and a copy between two
       devices that stage through memory of their own holds one slot of
       each, taken together. At most two copies run through one staging
       memory at a time, and a third waits until one is done, or raises
       [Invalid_argument] on the thread that holds both.}
    {- While a driver's [room], [submit] or [commit] blocks, where its
       driver declares [may_block], other submits and commits on that device
       wait for it.}}

    A wait releases the domain lock while it blocks, and returns to OCaml at
    least every 200 ms, where a pending [Sys.Break] from Ctrl-C raises. An
    interrupted wait loses no device and assigns no value.

    rig also holds locks of its own, briefly, inside most calls, and a driver
    or an io library holds its own inside its calls and inside the functions
    of its record that compiled code calls ({!capability}). A signal handler,
    finaliser or [Gc.Memprof] callback runs where the code it interrupts
    allocates or polls, which may be inside such a lock on the same thread.
    A call is made {e inside a lock} when such a callback makes it while its
    thread holds a lock of rig's, of a driver's or of an io library's.

    A call to rig that may take a lock, made inside a lock, raises
    [Invalid_argument] as it starts, having done nothing, but for {!submit},
    which frees its run. It raises even where it needs none of the locks its
    thread holds, since the thread's count of its locks does not say which
    they are. A function that takes no lock may be called from a callback at
    any point, and never raises it: {!name}, {!equal}, {!pp}, {!arch},
    {!computes}, {!runs_on_host}, {!alignment}, {!most_parts}, {!capability},
    {!host_of}, {!reaches}, {!submitted}, {!signaled}, {!lost}, {!failure},
    {!budget}, {!allocated}, {!Point.device}, {!Point.value},
    {!Point.reached}, {!Point.pp}, the properties and low-level values of
    {!Buffer}, {!Buffer.view}, {!Buffer.of_bigarray}, {!Image.device},
    {!Image.name}, {!Image.image}, {!Image.params}, {!Image.threads},
    {!Image.entry}, {!Submission}'s refs and parts, {!Submission.block},
    {!Submission.address}, {!Submission.Run}, {!Claim}, {!Profile} but
    {!Profile.take}, {!Local.new_key}, and {!Local.get} of a value already
    made for the device. Nor does {!fail}, which inside a lock leaves the
    owed stops to later calls. Outside a lock, a call from a callback runs as
    any call does.
    If the callback lets the exception escape, it ends the interrupted call as
    an asynchronous exception ({!section-errors}). The callback must not block
    either, in a wait of rig's or for another thread: other calls, and every
    round of reclamation, wait meanwhile for the lock its thread holds. A
    program that calls rig from such a callback handles that exception, or
    has the callback record the work for the program's own code to do, such
    as in an [Atomic.t]. *)

(** {1:loss Loss}

    A device becomes lost when:
    {ul
    {- its driver reports a fault, or fails to take or commit work;}
    {- another exception escapes its driver while the driver changes the
       device's memory or code ({!section-errors});}
    {- its progress stands still past its hang bound ({!section-hangs});}
    {- its queue waits for a value that a lost device never reached;}
    {- {!close} ends it;}
    {- {!fail} fails the process;}
    {- in a child of [fork], it is a driver's device the child inherited
       ({!section-stops}).}}
    A loss is final. Opening the device's name again makes a new device,
    unequal to the lost one, with a timeline of its own. Other devices go
    on, and a lost device's properties keep their values.

    rig treats as completed the values a device had reached when its loss
    was recorded. The work of later values may have run in full, in part or
    not at all, and no program can tell which. Values the lost device
    reports past those are {e dropped}. They are never reached, and they
    count only for the return of memory ({!section-reclaim}). The loss spreads
    along every chain of queue waits for values the lost device never
    reached. *)

(** {2:loss_memory Memory after a loss}

    A use of memory {e depends on} the points whose work must have
    completed for the use to be valid.
    {ul
    {- A read depends on the points of every device's last write of the
       memory. This holds for {!Buffer.wait}[ b Read], {!Buffer.read},
       {!Buffer.blit_to_bytes}, the source of a {!Buffer.copy}, a submit's
       read, and a claim that [rig_buffer_claim] takes for reading
       ({!Buffer.section-low}).}
    {- A use through an {!Submission.unordered} ref depends on no point.}
    {- Every other use depends on every point of the stamps, such as a
       write, a borrow, {!Buffer.wait}[ b Read_write], {!Buffer.write},
       {!Buffer.export}, and a claim that {!Claim.with_} takes. A
       {!Claim.with_} claim names no access, so it depends on what a write
       would.}}

    After a loss, these raise {!Lost}:
    {ul
    {- every use of the lost device, of its memory through any buffer, and
       of a borrow on it;}
    {- every use of other memory that depends on a point the lost device
       never reached, since that point's work may never have run.}}
    A point the lost device never reached stays so. Memory with a last write
    at such a point raises {!Lost} at every use through rig's calls but an
    {!Submission.unordered} one, for the rest of the process. Memory that
    the lost device only read raises it only for the uses that depend on
    every point. An array {!Buffer.export} made over such memory before the
    loss reads and writes it with no check of loss, and its bytes may be the
    old ones, the lost work's, or a mix. *)

(** {2:hangs Hangs}

    A driver whose work no kernel or vendor timeout bounds declares a
    {e hang bound}, [hang_ms] in {!Rig_edge.type-facts}. A wait checks the
    bound's clock while it watches the device. The clock starts when a wait
    first finds committed work of the device unreached. It starts again
    when a wait finds that the device has reached a later value, and when a
    wait finds the device idle, with all its committed work reached or its
    queue waiting for another device's unreached point. A device whose clock
    passes the bound is lost with the reason ["no progress for n ms"]. Only
    a wait checks the clock, so no device is lost to a hang while nothing
    waits for it. *)

(** {2:stops Stops and forks}

    A lost device has {e stopped} once its driver's stop
    ({!Rig_edge.Driver.val-stop}) has returned and the device reports its last
    submitted value. Only then do its memory and the values its submissions
    keep ({!Submission.make}) return. A device that never reports that value
    keeps them for the life of the process.

    From its loss on, a device's stop is {e owed} until a call runs it,
    which happens once none of the device's driver calls is in flight.
    {!close} runs it, after waiting for those calls. An open of the
    device's name runs it, and is [Error] while those calls are in flight.
    {!fail}, every round of reclamation and an open that finds every device
    slot held run each owed stop whose device has no driver call in
    flight. Another call that uses the device may run it sooner, except
    while it waits ({!section-domains}). A lost device that no such call
    reaches keeps its stop owed, and with it its memory, its slot and the
    values its submissions keep. So a program that has no more use for a
    lost device closes it, and the device's memory returns once it has
    stopped.

    A call that runs a stop takes the time of the driver's stop, and raises
    again any exception other than a fault that escapes it. An
    interrupt during a wait leaves no stop half run, since no call runs one
    while it waits.

    In a child of [fork], every driver's device inherited from the parent is
    lost, with the reason ["forked"] unless it was lost before, and its
    points count as never reached. The child never calls that device's
    driver, and a submission on it made before the fork keeps nothing: the
    child's next call on the device lets go of the value it keeps, without
    waiting for its work. A value kept by a submission on {!host} returns as
    in any process, so one kept after work a thread of the parent was
    submitting on {!host} at the fork stays until the child's next submit
    on {!host}. rig leaves a live io device as it was, and its library
    decides what a fork does to it, as a child of [fork] keeps the open
    files of its parent. An io device whose stop has not returned at the
    fork, owed or under way, stays lost in the child, which runs no stop of
    it and frees none of its memory, since that stop is the parent's. A
    device the child opens is its own. *)

(** {2:errors Errors and failure}

    An exception raised inside a call to rig leaves rig's own state
    consistent: its devices, names, timelines, stamps and memory. Every later
    call behaves as its doc states as far as that state decides it. No later
    call waits on the call the exception ended, and no name stays opening.

    An {e asynchronous} exception is one the runtime raises where the call
    allocates OCaml values or polls, such as [Sys.Break],
    [Stdlib.Out_of_memory], or an exception from a signal handler, a
    finaliser or a [Gc.Memprof] callback, or where a function of the call
    starts, as [Stack_overflow] is. It may leak the memory of the call it
    interrupts. One that interrupts a driver's stop ({!section-stops}) may
    leave that device never stopped, holding its memory and its slot for
    the rest of the process.

    A driver changes the memory and code that a device's work uses when it
    allocates, maps or gives back memory, and when it loads, looks up or
    unloads code. An exception other than a fault that escapes the driver
    during such a change may leave it half done. rig then loses the device,
    with the exception's text as the reason ({!Lost}), and raises the
    exception again. Calls that allocate, borrow, load or ask for a
    function ({!Image.resolve}) make such changes, and so do reclamation
    ({!section-reclaim}), on whichever device's memory returns, and a
    submit that gives back its device's mappings of other devices' words
    ({!section-opening}). An
    asynchronous exception from any of these calls may therefore lose a
    device. Every other exception leaves the device serving later calls,
    such as one that interrupts a wait ({!section-domains}), one from an io
    device's call, the [Invalid_argument] of a function the device cannot
    run ({!Image.resolve}), and that of a call made inside a lock
    ({!section-domains}). *)

exception Lost of t * string
(** [Lost (d, why)] is raised by a use that the loss of [d] makes invalid
    ({!section-loss_memory}). [why] is one of:
    {ul
    {- the driver's message for a fault or a failed [submit] or [commit];}
    {- for another exception that escaped the driver ({!section-errors}),
       its constructor's name and its integer and string arguments, such as
       ["Stdlib.Sys.Break"], ["Out_of_memory"] or ["Failure(\"no queue\")"];}
    {- ["no progress for n ms"] for a hang ({!section-hangs});}
    {- ["NAME lost"] for a device lost because its queue waited for the lost
       device named [NAME];}
    {- ["closed"] for a device {!close} ended;}
    {- the reason given to {!fail};}
    {- ["forked"] in a child of [fork];}
    {- ["out of memory"] when the C heap had no room for the reason.}}
    It prints as ["NAME lost: why"] with [d]'s name. *)

val lost : t -> string option
(** [lost d] is [Some why] if [d] is lost, where [why] is the reason its
    {!Lost} carries, and [None] otherwise. It raises nothing and waits for
    nothing. *)

val fail : string -> unit
(** [fail why] fails the process. Every device but {!host} that is not lost
    is lost with [why], as a fault loses a device, and devices already lost
    keep their reason. Every {!open_} and {!open_io} that starts afterwards
    returns [Error why] without calling its opener, and one that races
    [fail] makes a lost device. [fail] runs each owed stop it can
    ({!section-stops}), and returns without waiting for any device to have
    stopped. Only the first call acts. Called inside a lock
    ({!section-domains}), it fails the process and leaves the owed stops to
    later calls. *)

val failure : unit -> string option
(** [failure ()] is the process's first failure, if any: the first loss of a
    device that no {!close} or fork ended, as {!Lost} prints it, or [why] of
    {!fail} if that came first. It raises nothing and waits for nothing. *)

(** {1:reclaim Reclamation}

    No function frees a buffer. After its use, the life of an allocation
    goes like this:
    {ol
    {- {e Collection.} Once every buffer over the allocation, and every
       array {!Buffer.read}, {!Buffer.write} or {!Buffer.export} made of it,
       is unreachable, the OCaml collector collects it, on whichever domain
       the collector runs. A prepared submission keeps the buffers it uses
       reachable ({!Submission}).}
    {- {e Waiting.} It waits until every point of its stamps, and of the
       work that used it through {!Submission.unordered} refs, which rig
       records beside the stamps, is reached or dropped.}
    {- {e Reuse.} It then enters its device's {e cache}, the device's
       returned memory, which {!Buffer.create} reuses with empty stamps. A
       new buffer over it waits for no work of its earlier use.}
    {- {e Release.} Cached memory returns to the driver when the cache is
       freed ({!free_cache}), when the budget is lowered ({!set_budget}), or
       when an allocation needs the room. An allocation that finishes
       waiting while its device holds more than its budget skips the cache,
       as does an io device's memory, and returns to the driver. A lost
       device's memory returns as {!section-stops} states.}}
    Memory that an asynchronous exception leaked ({!section-errors}) may
    never take these steps.

    Collected memory moves to the cache or the driver only inside rig's
    calls, on any domain. {!Buffer.create}, {!Buffer.of_string},
    {!Buffer.of_io} and {!Buffer.copy} first move the collected memory whose
    points are reached or dropped on the devices they use, and let go of the
    values that collected submissions on those devices keep, once the work
    submitted on the device before the first such call after the collection
    is done, also when they then raise {!Lost}. {!free_cache} moves memory on its device, reclamation on every
    device, and a lost device's stop on its own. Memory collected while a
    domain blocks returns through other domains' calls. A device reports its
    values up to [k] values late ({!section-timeline}), and memory and kept
    values return by what it reports.

    A device's {e budget} is the most bytes it holds at one time of the
    memory that counts in it: its live buffers, its loaded images' code and
    its cache ({!budget}). [Pinned] memory counts in the host's budget,
    except on a device whose memory the host addresses, such as Metal's,
    where it counts in the device's. A host buffer counts the bytes it
    takes of the C library's memory: its own, and up to 112 more where it
    starts on 128 bytes ({!Buffer.create}). An allocation's bytes also pace the
    OCaml collector, so a device nearing its budget collects unreachable
    buffers more often. Live buffers are never freed, so a program that
    keeps every buffer reachable runs out of memory however large its
    budget.
    {ul
    {- A dropped host buffer of 64 KiB or more holds its memory until a
       major cycle of the collector frees it. A cycle comes once the bytes
       of such buffers that reached the major heap since the last one reach
       [custom_major_ratio] / 150 ({!Gc.control}) of the program's memory,
       the allocating domain's major heap and the host buffers live at the
       last cycle, or half the host's last-level cache if that is more, and
       at least 4 MiB.}
    {- The host keeps collected buffers of 64 KiB or more for reuse. It keeps
       up to 32 MiB of them, and beyond that only while the bytes of its kept
       and live buffers together stay within the peak of its live buffers
       over the collector's current and previous major cycles.}
    {- On a driver's or an io device, an allocation larger than the budget it
       counts in raises {!Out_of_memory} immediately, without freeing the
       cache.}
    {- An allocation that its budget or its driver refuses first reclaims.
       rig frees the cached memory charged to that budget, waits for the work
       that holds back other collected memory, from the second round on runs
       the collector, and tries again. The allocation raises {!Out_of_memory}
       after three such rounds.}
    {- A growth of a device's queues ({!submit}) that its driver has no
       memory for reclaims as an allocation does, both the device's budget
       and the host's, as the driver may grow them in memory that counts in
       either.}} *)

val budget : t -> int
(** [budget d] is [d]'s budget ({!section-reclaim}). It starts at [max_int]
    for {!host}, at its driver's [budget] ({!Rig_edge.type-facts}) for a
    driver's device, and at {!Rig_edge.Io.budget} for an io device. *)

val allocated : t -> int
(** [allocated d] is the bytes that [d] holds now of the memory that counts
    in its budget ({!section-reclaim}). *)

val set_budget : t -> int -> unit
(** [set_budget d n] sets [d]'s budget to [n], then frees cached memory
    ({!free_cache}) until [d] holds at most [n] bytes or its cache is empty.
    It never frees a live buffer.

    Raises [Invalid_argument] if [n < 0]. *)

val free_cache : t -> unit
(** [free_cache d] first moves [d]'s collected memory ({!section-reclaim}),
    then returns [d]'s cached memory to its driver once the work using it is
    done. On a lost [d] that has stopped ({!section-stops}), that is all of
    its memory that is collected. On {!host}, it frees the collected buffers
    the host keeps for reuse. *)

exception Out_of_memory of t * int
(** [Out_of_memory (d, n)] is raised when [d] cannot allocate [n] bytes,
    immediately or after the rounds of reclamation. [d] is the device the
    allocation was for, also when another device's budget refused it, as the
    host's does for [Pinned] memory ({!Buffer.Pinned}). *)

(** {1:buffers Buffers} *)

(** The type for accesses to memory: what memory admits, what a host access
    waits for, and how fixed memory is used ({!Submission.make}). *)
type access =
  | Read  (** Reading. *)
  | Read_write
      (** Reading and writing, which the memory must admit
          ({!Buffer.val-access}). *)

(** Buffers of device memory.

    A buffer's {e memory} is the whole allocation it lies in. A {e view} is
    a buffer over part of another buffer's bytes, in the same allocation
    ({!Buffer.view}). Views and borrows share their memory's stamps and
    claims, so work through any of them is work on that allocation. rig does
    not interpret the bytes.

    A device's work addresses only buffers on that device. To use another
    device's memory, a submission names a borrow of it, which exists where
    the device can map that memory ({!Rig.reaches}). A borrow may map the
    allocation into the borrowing device, and every borrow on that device
    shares the mapping. When the allocation returns, rig releases the
    mapping once the borrowing device has run the work submitted until then,
    and gives the allocation back after that. A lost device releases its
    mappings once it has stopped, so memory it maps returns only then.

    The life of an allocation goes like this:
    {ol
    {- {!Buffer.create}, {!Buffer.of_string}, {!Buffer.of_bigarray} or
       {!Buffer.of_io} makes it, with empty stamps.}
    {- {!Buffer.view} and {!Buffer.borrow} make further buffers over it.}
    {- Copies, submissions and host access use it, each after the points its
       stamps name. Submissions on a device other than {!Rig.host}, and
       copies that such a device runs, add a point of their own.}
    {- It returns to its device some time after no buffer over it, and no
       array made of it, is reachable ({!Rig.section-reclaim}).}} *)
module Buffer : sig
  type device := t

  type t
  (** The type for buffers. Compare buffers with [==]. Polymorphic
      comparison does not look at their bytes: [compare] terminates, with an
      order that says nothing about contents, and [(=)] raises
      [Invalid_argument] when it reaches two buffers of one device, a buffer
      and itself included, and finds buffers of different devices unequal.
      [Hashtbl.hash] depends on a buffer's range and on its memory's
      addresses and state, which can change while the buffer lives, as when
      a device first borrows host memory. *)

  (** {1:making Making buffers} *)

  (** The type for the kinds of a device's memory that {!create} allocates. *)
  type kind = Rig_edge.memory =
    | Device
        (** The device's own memory, which counts in the device's {!budget}. *)
    | Pinned
        (** Host memory that both the device's work and the host address,
            page-locked where the host pages memory out, with no flush
            needed between them. On a device whose memory the host addresses,
            it is the device's own memory. Which {!budget} it counts in is
            stated in {!Rig.section-reclaim}. *)
    | Mapped
        (** The device's own memory, which the host also addresses through a
            write-combined window, and which counts in the device's {!budget}.
            Work submitted after a host write sees it, and host reads are
            slow. Where the device has no such window, or the window or the
            device's budget has no room for the buffer, it is [Pinned]
            memory. The host addresses it, and work submitted after a host
            write sees it, but it counts in the budget that [Pinned] memory
            counts in ({!Rig.section-reclaim}). rig falls back to [Pinned]
            without first freeing the device's cache. *)

  val create : ?memory:kind -> device -> int -> t
  (** [create d n] is a buffer of [n] new bytes of [d]'s memory of kind
      [memory] (defaults to [Device]), with unspecified contents. An empty
      buffer allocates nothing. A non-empty one starts on [d]'s {!alignment}.
      A host buffer of at least 64 KiB and at least four pages starts on a
      page, so devices can {!borrow} it; a smaller one of at least 128 bytes
      starts on 128 bytes, so threads that write it in pieces of whole
      128-byte blocks share no cache line, on Apple silicon's 128-byte lines
      or x86's 64-byte ones. On an io device the memory is what
      {!Rig_edge.Io.alloc} makes, whatever [memory].

      Raises [Invalid_argument] if [n < 0], or if [n > 0], [d] is an io
      device and its {!Rig_edge.Io.alloc} raises it, and {!Lost} if [d] is
      lost or its driver faults during the allocation. *)

  val of_string : string -> t
  (** [of_string s] is a buffer on {!host} holding [s]'s bytes, placed as
      {!create} places a host buffer. *)

  val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> t
  (** [of_bigarray ba] is a buffer on {!host} over [ba]'s bytes, without a
      copy. It keeps [ba] reachable, and its memory is outside the claims
      ({!Claim}). Each call makes a new allocation with stamps of its own,
      even over bytes another call wraps, so rig does not order work through
      one such buffer with work through the other. Wrap an array once and
      share the buffer through {!view} and {!borrow}. *)

  val of_io : device -> 'r Type.Id.t -> 'r -> access:access -> int -> t
  (** [of_io d k r ~access n] is a buffer over the [n] bytes of [r], a
      {e region} of the io device [d]. A region is memory as [d]'s library
      represents it, and the library declares its type under [k]
      ({!Rig_edge.Io.region_key}). Its memory admits [access], which is
      [Read] for a region [d] holds for reading, such as a file opened
      read-only. Its memory is outside the claims ({!Claim}), and [r] returns
      to {!Rig_edge.Io.free} once unreachable.

      Raises [Invalid_argument] if [d] is not an io device, its library's key
      is not [k], or [n < 0], and {!Lost} if [d] is lost or closed. *)

  val view : t -> first:int -> length:int -> t
  (** [view b ~first ~length] is the [length] bytes of [b] from its byte
      [first] on, in [b]'s memory.

      Raises [Invalid_argument] if [b] is dead, [first] or [length] is
      negative, or the bytes do not lie inside [b]'s. *)

  val borrow : device -> t -> t option
  (** [borrow d b] is a buffer on [d] over [b]'s bytes, without a copy, or
      [None] if [d] cannot map [b]'s memory ({!Rig.reaches}) or its driver
      refuses the mapping. It is [Some b] for [b] on [d]. Memory of no
      bytes, which has none to map, borrows on every device of its machine
      but an io device, whatever {!Rig.reaches} says, and an io device's on
      every other device. The borrow keeps [b] reachable and shares its
      memory, so work through it is ordered as work on [b]. A borrow of a
      borrow maps the memory the first one maps. Every borrow on [d] of one
      allocation shares one mapping, which lasts until the allocation
      returns or [d] has stopped.

      Raises [Invalid_argument] if [b] is dead; {!Lost} if [d] is lost or its
      driver faults during the borrow; and [Sys_error] if asking an io device
      for [b]'s pages fails transiently ({!Rig_edge.Io.pages}). *)

  (** {1:properties Properties}

      These return a value for every buffer, a dead one included, and raise
      nothing. *)

  val device : t -> device
  (** [device b] is the device [b] is on: [d] for a buffer that {!create},
      {!of_io} or {!borrow} made on [d], {!host} for one that {!of_string} or
      {!of_bigarray} made, and its buffer's device for a {!view} and for
      {!Claim.consume}'s result. *)

  val length : t -> int
  (** [length b] is the number of [b]'s bytes. *)

  val is_borrowed : t -> bool
  (** [is_borrowed b] is [true] iff [b] lies over memory that something other
      than [b]'s device holds, either a bigarray ({!of_bigarray}) or another
      device's memory ({!borrow}). *)

  val spans : t -> bool
  (** [spans b] is [true] iff [b]'s bytes are all of its memory, as for a
      buffer {!create}, {!of_string}, {!of_bigarray} or {!of_io} made, a
      borrow of one, or a view of all of one. *)

  val overlaps : t -> t -> bool
  (** [overlaps b b'] is [true] iff [b] and [b'] share a byte of memory,
      through views of one allocation, a borrow and the memory it maps, or two
      bigarrays over the same bytes. An empty buffer overlaps nothing. *)

  val access : t -> access
  (** [access b] is the access [b]'s memory admits, the same through each of
      its views and borrows. It is [Read_write] for memory {!create},
      {!of_string} and {!of_bigarray} make, and what {!of_io} was given.

      rig never writes [Read] memory, and refuses a copy or work that would
      ({!copy}, {!Submission.make}, {!submit}), and {!write} of it raises. A
      write through an array {!export} made breaks this, and reads of the
      memory then see unspecified bytes. A library that writes its caller's
      buffer checks this first, so that it reports the error itself. *)

  val dead : t -> string option
  (** [dead b] is [Some why] if [b] is dead, where [why] is the reason its
      memory was consumed with ({!Claim.consume}), and [None] otherwise. A
      library that receives a buffer checks it first, so that it reports the
      error itself. *)

  (** {1:host_access Host access}

      The host reads and writes a buffer's bytes in place where it
      {e addresses} the buffer's memory: memory on {!host}, a borrow on
      {!host} included, and memory of a device of this machine that has a
      host address, such as a {!memory_device}'s. {!read} and {!write} take a
      read claim on the memory, so that no donation writes it meanwhile
      ({!Rig.section-claims}), wait for the work the access follows, call
      their function on the bytes, then release the claim when it returns or
      raises. The bytes are an array of [length b / Bigarray.kind_size_in_bytes
      k] elements of kind [k], in the host's byte order, without a copy, and
      are ordered only during the call. An empty buffer is an empty array,
      whatever its memory. {!export} waits itself, and {!wait} orders other
      host code. Host access takes no value and adds no point to the
      stamps. The blits and {!fill} ({!section-copying}) copy bytes
      to and from any memory, the host's or not.

      A wait covers only the work submitted before it returns
      ({!Rig.section-order}). Host access from several domains needs its own
      synchronization.

      {!read}, {!write} and {!export} raise [Invalid_argument] if [b] is
      dead, its bytes are not a whole number of elements of [k], [b] has
      bytes that the host does not address or that do not start at a
      multiple of the elements' size (of one component's size for complex
      kinds), or a {!Claim.with_} holds the memory exclusive. *)

  val wait : t -> access -> unit
  (** [wait b access] returns once the work that a host access [access] of
      [b] must follow is done. For [Read] it waits for every device's last
      write of [b]'s memory, and for [Read_write] for every use of it by any
      device. It allocates nothing when that work is already done.
      {ul
      {- Host memory and an io device's memory have work to wait for only
         where a device borrowed them.}
      {- It does not wait for work through {!Submission.unordered} refs,
         which the caller orders.}
      {- It takes no claim, so a donation on another domain may write the
         memory once it returns, which {!read}, {!write}, the copies, the
         blits and {!fill} prevent.}}

      Raises [Invalid_argument] if [b] is dead, and {!Lost} where
      {!Rig.section-loss_memory} says, before it waits on any device if the
      loss came first. *)

  val read :
    t ->
    ('a, 'b) Bigarray.kind ->
    (('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> 'c) ->
    'c
  (** [read b k f] is [f a], where [a] is [b]'s bytes as elements of [k],
      once every device's last write of [b]'s memory is done. [a] is the
      memory itself, so the host must address it where [b] has bytes;
      {!blit_to_bytes} copies out of any memory. [f] must not write through
      [a], since work that reads the memory may still run, and must not keep
      [a] after it returns, since nothing orders a later access through it. *)

  val write :
    t ->
    ('a, 'b) Bigarray.kind ->
    (('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> 'c) ->
    'c
  (** [write b k f] is [f a], where [a] is [b]'s bytes as elements of [k],
      once every use of [b]'s memory is done. Work submitted after [write]
      returns sees [f]'s writes. [f] must not keep [a] after it returns,
      since nothing orders a later access through it.

      Raises [Invalid_argument] also if [b]'s memory admits only [Read]
      ({!val-access}). *)

  val export :
    t -> ('a, 'b) Bigarray.kind -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
  (** [export b k] is [b]'s bytes as elements of [k] for good, once every use
      of [b]'s memory is done, for code that keeps them, such as a library's
      array. The array keeps the memory alive while it is reachable. From
      then on the memory is outside the claims ({!Claim}), and the caller
      orders access through the array with later work, as for
      {!of_bigarray}. The array reads without any check of loss
      ({!Rig.section-loss_memory}).

      Raises [Invalid_argument] also if [b]'s memory is not {!host}'s, which
      {!create} made on {!host}, or {!of_string} or {!of_bigarray} made.
      Another device's memory is refused even where the host addresses it,
      since every use of a lost device's memory raises {!Lost}
      ({!Rig.section-loss_memory}) and reads through an array cannot. *)

  (** {1:copying Copying}

      A copy is ordered as work that reads its source and writes its
      destination. Where the host addresses both sides, or one side is an io
      device's memory and the host addresses the other, the host or the io
      device copies, and no timeline gets a value. Otherwise a device copies
      on its {e copy engine}, the queue that its driver names in
      [copy_engine] ({!Rig_edge.type-facts}), and the copy takes the next
      value of that device's timeline ({!Rig.submitted}). On another machine,
      a device of that machine copies.

      A copy that no device can run directly passes through {e staging
      memory}, host memory that rig makes in allocations of 32 MiB, each at
      the first copy whose bytes need it, and keeps for the life of the
      process. A device that maps no host memory ([maps_host] in
      {!Rig_edge.type-facts}) stages through [Pinned] memory of its own,
      which it keeps until it is lost. Between two such devices, the host
      copies from one device's staging memory into the other's.

      The blits copy between a buffer and an OCaml string or bytes, and
      {!fill} sets a buffer's bytes. They are ordered as copies. Where the
      host addresses the buffer ({!section-host_access}), the host copies
      once the work the copy follows is done. Otherwise the bytes pass
      through staging memory, and a device copies them to or from the buffer
      as {!copy} does. Each returns once its bytes are where they go.

      A copy, blit or fill holds a read claim on its buffers' memory until it
      returns, so that no donation writes the memory meanwhile
      ({!Rig.section-claims}). *)

  val copy : src:t -> dst:t -> unit
  (** [copy ~src ~dst] copies [src]'s bytes into [dst] and returns once they
      are there.

      Raises [Invalid_argument] if [src] or [dst] is dead, they differ in
      length or overlap ({!overlaps}), [dst]'s memory admits only [Read], one
      side is an io device's memory and the other a borrow of it, a side's
      memory is held exclusive, the copy has bytes and no device can copy
      between the two sides, or it goes through staging memory whose two
      slots the calling thread holds ({!section-domains}). Raises {!Lost} if
      a device that runs the copy is lost or the copy loses it,
      {!Out_of_memory} if staging memory cannot be allocated or mapped, and
      what an io device's read or write raises. *)

  val blit_from_string : string -> int -> t -> int -> int -> unit
  (** [blit_from_string s i b j n] copies the [n] bytes of [s] from [i] into
      [b] from its byte [j].

      Raises [Invalid_argument] if the ranges are not valid, [b] is dead or
      its memory admits only [Read], and otherwise as {!copy}[ ~dst:b]
      raises. *)

  val blit_to_bytes : t -> int -> bytes -> int -> int -> unit
  (** [blit_to_bytes b i s j n] copies the [n] bytes of [b] from its byte [i]
      into [s] from [j].

      Raises [Invalid_argument] if the ranges are not valid or [b] is dead,
      and otherwise as {!copy}[ ~src:b] raises. *)

  val fill : t -> char -> unit
  (** [fill b c] sets every byte of [b] to [c].

      Raises [Invalid_argument] if [b] is dead or its memory admits only
      [Read], and otherwise as {!copy}[ ~dst:b] raises. *)

  (** {1:low Low level}

      For the libraries that submit work. C code reads a buffer's host
      address, its length, where its bytes lie and the reason it is dead with
      [rig_buffer_host], [rig_buffer_bytes], [rig_buffer_span] and
      [rig_buffer_why], claims its memory with [rig_buffer_claim] and
      [rig_buffer_release], and waits under the claim with [rig_buffer_wait],
      which runs {!wait}. All are declared in [rig.h].

      Memory is {e handle-only} when its driver names it by a handle and
      gives it no address ({!Rig_edge.location}). *)

  val io : t -> 'r Type.Id.t -> 'r option
  (** [io b k] is the region [b]'s memory lies in, if it is an io device's
      memory of a library whose key is [k], and [None] otherwise.

      Raises [Invalid_argument] if [b] is dead. *)

  val aligned : t -> at:int -> int -> bool
  (** [aligned b ~at n] is [true] iff the address [at] bytes past [b]'s first
      byte, in the addresses of [b]'s device's work, is a multiple of [n], for
      [n] a power of two. It is arithmetic on the address, defined for every
      [at], in [b]'s bytes or outside them.

      Raises [Invalid_argument] if [b] is dead, an io device's memory or
      handle-only, or if [n] is not a positive power of two. *)

  val offset : t -> int
  (** [offset b] is the number of bytes of [b]'s memory before [b]'s first
      byte. *)

  val handle : t -> nativeint
  (** [handle b] is the driver's object for [b]'s memory
      ({!Rig_edge.location}), such as an [MTLBuffer], which [b] starts
      {!offset} bytes into. It is [0n] for an empty buffer that {!create}
      made on a driver's device.

      {b Warning.} The object lives only while [b] is reachable. Work that
      uses it must keep [b] reachable until the work is done, for example
      with the [keep] of its submission ({!Submission.make}).

      Raises [Invalid_argument] if [b] is dead, or is {!host}'s or an io
      device's memory, which no driver object names. *)
end

(** {1:submitting Running work}

    A device runs code from an {!Image} loaded on it. A {!Submission}
    describes work for one device once, and {!submit} runs it with one
    submit's buffers and returns the point of that work on the device's
    timeline.

    A library runs work like this:
    {ol
    {- It loads an image ({!Image.load}), resolves its functions
       ({!Image.resolve}), and makes a submission of parts that launch them
       ({!Submission.make}).}
    {- It makes a run of the submission ({!Submission.Run.make}), which
       holds the launches' grids and parameters, and stores them.}
    {- It calls {!submit} with the run and this submit's buffers. For each
       later submit it takes the run ({!Submission.Run.take}), stores new
       parameters and submits again.}
    {- Once the submission is unreachable, its memory returns after the
       work of its last submit is done.}}
    {!section-host_objects} has an example. *)

(** {2:images Images} *)

(** Code loaded on a device. *)
module Image : sig
  type device := t

  type t
  (** The type for loaded images: a binary in a device's format, whose
      functions that device's work runs. *)

  val load : device -> string -> (t, string) result
  (** [load d binary] loads [binary] on [d], such as a code object for AMD, a
      cubin for NV, a CUDA module for CUDA, a metallib for Metal or a
      relocatable ELF object for {!host} ({!section-host_linking}). Each call
      loads [binary] again, so that each image's module variables are its
      own. Where [d]'s memory holds the code, it counts in [d]'s {!budget},
      and [load] places it there after the work submitted on [d] before it.
      The image stays loaded while it is reachable. A call on [d] that finds
      it collected ({!Rig.section-reclaim}) unloads it once the work
      submitted on [d] before that call is done.

      It is [Error why] if [d]'s driver rejects [binary], where [why] is
      [d]'s {!val-name}, then the driver's reason. [d] stays usable.

      Raises [Invalid_argument] if [d] is an io device, which loads no code,
      or its driver places the code in [d]'s memory ([Place] in
      {!Rig_edge.type-code}) where [d] has no copy engine and the host does
      not address that memory; and {!Lost} if [d] is lost or its driver
      faults during the load. *)

  val device : t -> device
  (** [device i] is the device [i] is loaded on. *)

  type func
  (** The type for a function of a loaded image. A function keeps its image
      reachable. *)

  val resolve : t -> string -> func option
  (** [resolve i f] is [i]'s function [f], made ready to launch, or [None] if
      [i] has none. The first resolve of [f] makes what its launches need,
      such as its scratch.

      Raises [Invalid_argument] if [i]'s device cannot run [f]
      ({!Rig_edge.Driver.entry}), {!Out_of_memory} if the device has no
      memory for what [f]'s launches need after the rounds of reclamation
      ({!section-reclaim}), and {!Lost} if [i]'s device is lost. *)

  val name : func -> string
  (** [name fn] is [fn]'s name in its image. *)

  val image : func -> t
  (** [image fn] is the image [fn] belongs to. *)

  val params : func -> int option
  (** [params fn] is the number of parameter bytes, from byte [0], that [fn]
      reads, or [None] if its driver does not know it. A launch of [fn] must
      give at least that many ({!Submission.launch}). *)

  val threads : func -> int option
  (** [threads fn] is the most threads a group of [fn] may have, or [None]
      if its driver does not know it. {!Submission.Run.threads} refuses
      more. *)

  (** {1:raw Raw function values}

      This value serves code that refers to a function outside a launch's
      parameters, such as an indirect command buffer, a CUDA graph over a
      module's functions, or host code that calls another image's function.
      rig cannot see where such code keeps a value, so it cannot keep the
      image loaded for it. Except for these uses, a launch is better given a
      function's value by an entry ref ({!Submission.entry}), which keeps
      the image. *)

  val entry : func -> int
  (** [entry fn] is the value by which compiled code refers to [fn] (the
      [code] of {!Rig_edge.Driver.entry}), such as a kernel descriptor's
      address, a [CUfunction], an [MTLComputePipelineState] or the address
      of a host function.

      {b Warning.} The value is valid only while [fn] is reachable. Work that
      uses it must keep [fn] reachable until the work is done, for example
      with the [keep] of its submission ({!Submission.make}). *)
end

(** {2:host_code Host code}

    {!host} runs code that the process links from relocatable ELF objects,
    on its own cores. {!Image.load} on {!host} links an object into
    executable memory once ({!section-host_linking}), and the image's
    functions are the named symbols of its executable sections. The host's
    queue runs only launches. *)

(** {3:host_launches Launches}

    A launch runs on the thread that submits it and on the workers of the
    host's pool, which the host's other parallel code, such as nx's CPU
    kernels, shares, and it is done when {!submit} returns.

    A host function's C type is
    {v void f(const void *params, int64_t first, int64_t end); v}
    [params] points to the launch's parameters, in which each ref holds the
    host address it names ({!Submission.type-ref}). The launch's {e threads}
    are numbered [g * t + i], where [g] is a group's index in the grid with x
    fastest, [t] the threads of a group, and [i] a thread's index in its group
    with x fastest. Each call of [f] runs the threads [first] to [end - 1],
    which are whole groups, and the calls together run each thread once. The
    calls run in any order and in parallel, on at most as many threads as the
    host has {e performance cores} ({!section-host_cores}), so a call must
    not write what another reads. A launch with shared memory, with an axis
    of zero groups or threads, or with more than 2{^62} threads never runs,
    and {!submit} raises. *)

(** {3:host_cores Performance cores}

    The performance cores are the host's cores that run compute-bound work
    at more than half the speed of its fastest, bounded by the cores the
    process may occupy. Their count is taken at its first use and stays the
    same for the life of the process. They are:
    {ul
    {- On Linux, the CPUs in the affinity mask of the first thread to ask
       whose reported capacity is more than half the largest, or all of them
       where no capacity is reported, and at most the process's cgroup v2
       quotas allow.}
    {- On macOS, the performance cores where the system reports them.}
    {- Otherwise, the cores the process may occupy.}} *)

(** {3:host_objects Objects}

    A C compiler makes such an object, for instance with
    {v clang -c -O2 -fPIC --target=x86_64-none-unknown-elf f.c v}
    for an x86_64 host. For an arm64 host the flags are
    [--target=aarch64-none-unknown-elf -ffixed-x18], since macOS and Windows
    reserve the register [x18].

    For instance, this function adds the floats of two arrays into a third:
    {v
    #include <stdint.h>
    struct add { const float *x, *y; float *z; int64_t n; };
    void add(const struct add *p, int64_t first, int64_t end) {
      for (int64_t i = first; i < end && i < p->n; i++)
        p->z[i] = p->x[i] + p->y[i];
    }
    v}
    Loaded once, it runs over [n] floats in 64 groups:
    {[
    let image = Result.get_ok (Rig.Image.load Rig.host obj) in
    let add = Option.get (Rig.Image.resolve image "add") in
    let refs = Submission.[| read 0 ~at:0; read 1 ~at:8; write 2 ~at:16 |] in
    let s = Submission.(make host [| launch ~refs add ~params:32 |]) in
    let run = Submission.Run.make s and k = Submission.block s 0 in
    Submission.Run.groups run k 64 1 1;
    Submission.Run.threads run k ((n + 63) / 64) 1 1;
    Submission.Run.int64 run k 24 (Int64.of_int n);
    ignore (submit run ~buffers:[| x; y; z |])
    ]} *)

(** {3:host_linking Linking}

    {!Image.load} on {!host} links an object that meets these rules, and is
    [Error] for any other:
    {ul
    {- The host is x86_64 or arm64, and the object is a well-formed 64-bit,
       little-endian, relocatable ELF object ([ET_REL]) for the host's
       machine ([EM_X86_64] or [EM_AARCH64]).}
    {- No allocated section is writable and not empty, as a [.data] or
       [.bss] in use would be. Code that needs a variable gets it through
       its parameters.}
    {- Each relocation holds its addend in its entry ([SHT_RELA]), is of a
       type listed under {!section-host_relocations}, uses no symbol defined
       outside the image, such as a common or thread-local variable, patches
       inside the image, and has a value that fits its field.}
    {- Each symbol the object refers to and does not define is defined by
       the process ({!section-host_symbols}).}
    {- The image and its stubs fit in [max_int] bytes.}}

    The link lays out the object's allocated sections ([SHF_ALLOC]), its code
    and read-only data, in one image, each at an address that is a multiple
    of its alignment ([sh_addralign]). It writes the image once, applies its
    relocations, then makes the memory executable. The memory is never
    writable and executable at the same time ({!section-host_platforms}). *)

(** {3:host_relocations Relocations}

    The relocations are those a compiler emits for position-independent code
    of the small code model, with [-fPIC], that calls functions and reads its
    own data. For a relocation at the place [P] of the image, of the symbol
    whose address is [S], with the addend [A], the link writes:
    {ul
    {- [R_X86_64_PC32]: [S + A - P].}
    {- [R_X86_64_PLT32]: [S + A - P], for a call or jump, through a stub
       when [S] is out of its reach.}
    {- [R_AARCH64_ADR_PREL_PG_HI21]: [Page(S + A) - Page(P)].}
    {- [R_AARCH64_ADD_ABS_LO12_NC], and [R_AARCH64_LDST8_ABS_LO12_NC] to
       [R_AARCH64_LDST128_ABS_LO12_NC] for loads and stores of 8 to 128
       bits: the low 12 bits of [S + A].}
    {- [R_AARCH64_CALL26] and [R_AARCH64_JUMP26]: [S + A - P], through a
       stub when [S] is out of reach.}}

    A stub lies after the image for each symbol the object refers to and does
    not define, with a word that holds the symbol's address. An x86_64 stub
    jumps through the word and uses no register. An arm64 stub loads the word
    into [x17], which a veneer may use.

    {b References.}
    {ul
    {- The {{:https://gitlab.com/x86-psABIs/x86-64-ABI}x86-64 psABI}, chapter
       4.4: relocation types and their formulas.}
    {- The
       {{:https://github.com/ARM-software/abi-aa/blob/main/aaelf64/aaelf64.rst}
        ELF for the Arm 64-bit Architecture}, section 5.7: relocation types,
       their formulas and fields, and the registers a veneer may use.}} *)

(** {3:host_symbols Symbols}

    A symbol the object refers to is its definition in the object, or else
    its definition in the libraries the process loaded into its global
    scope, such as the C and math libraries. A function of another image is
    called through its address, which an entry ref places in the launch's
    parameters ({!Submission.entry}).

    A compiler also calls functions the source does not name, such as
    [__extendhfsf2] where the machine has no instruction for a 16-bit float
    conversion, or [memcpy] for a large copy. The object defines these, or
    they come from the process and follow the process's convention. On x86_64
    Windows that convention differs from the object's, so code for Windows
    must define every function the compiler calls. *)

(** {3:host_abi Calling conventions}

    Linked code follows its object's convention. A compiler that targets ELF
    without an operating system uses System V AMD64 on x86_64 and AAPCS64 on
    arm64. x86_64 Windows has its own convention, but the host still calls a
    function by System V. A function of the process's libraries follows the
    platform's convention, so on x86_64 Windows linked code declares those it
    calls [__attribute__((ms_abi))]. Code for an ELF target probes no stack,
    and Windows commits a thread's stack one guard page at a time, so on
    Windows a thread's stack is committed, but for its last 64 KiB, before
    the thread first runs linked code. A stack frame of any size within it
    then runs, as elsewhere. *)

(** {3:host_platforms Platform support}

    The host links code on x86_64 and arm64 hosts running Linux, macOS or
    Windows. On arm64 macOS the code's memory is mapped with [MAP_JIT],
    whose write protection only the linking thread lifts while it writes.
    Elsewhere the memory is mapped writable, then made executable and
    read-only once written. *)

(** {3:host_macos Executable memory on macOS}

    On macOS, a program built with the hardened runtime maps executable memory
    only with the [com.apple.security.cs.allow-jit] entitlement (arm64) or
    [com.apple.security.cs.allow-unsigned-executable-memory] (x86_64). Without
    it a load is an [Error]. With [com.apple.security.cs.single-jit], a
    process maps executable memory once, so its loads after the first are
    [Error]s.

    With [com.apple.security.cs.jit-write-allowlist], the system writes code
    only through functions that the executable, or a library it loaded at
    start-up, lists. rig's C stubs list theirs, so loads work where the stubs
    are linked into the executable: in native programs, and in bytecode ones
    built with [-custom] or [-output-complete-exe]. In a bytecode program
    that loads the stubs at run time, as [ocamlrun] does, the system ends the
    process at its first load. Link the stubs into its executable, or give it
    the [com.apple.security.cs.jit-write-allowlist-freeze-late] entitlement
    and call [pthread_jit_write_freeze_callbacks_np] once the stubs are
    loaded, before its first load. *)

(** {2:prepared Submissions} *)

(** Prepared submissions.

    A submission is work for one device, made of {e parts}, such as copies
    and launches. It is made once and run many times. The buffers and the
    points to wait for change from one run to the next, so they are
    arguments of {!submit}, and so are the launch parameters, which a
    {!Submission.Run} holds. A submission holds nothing of a run, so any
    number of domains or threads submit it concurrently, each with its own
    run.

    A submission's {e fixed memory} is the buffers that every run uses and
    that neither a part nor a submit names, such as an argument buffer and
    the memory its addresses name. Each is fixed with an access, and each
    submit orders it by that access. A buffer may be fixed in any number of
    submissions.

    Rule 2 of {!Rig.section-order} orders a submission's parts by the memory
    they use. A copy reads its [src] and writes its [dst], words read their
    buffer, a buffer ref uses its slot as its kind says, and every part uses
    the fixed memory with its access. Two uses may share memory where their
    buffers overlap ({!Buffer.overlaps}), or where either is a slot, whose
    buffer only a run gives. A {!Submission.unordered} ref orders nothing.

    A launch or a fill runs on the first of the device's queues that runs
    its kind ({!Rig_edge.type-facts}). Words run on the first queue that
    runs both launches and words, and otherwise on the first that runs
    words. A copy runs on the queue of the last earlier part it follows,
    where that queue runs copies, and otherwise on the device's copy engine,
    the queue that [copy_engine] names in its facts.

    A submission keeps every buffer and image its parts name, its fixed
    memory and its [keep] reachable while it is reachable, so that memory is
    never freed between {!Submission.make} and a submit. A launch needs no
    [keep] for its image, since an image unloads only once the work
    submitted before the image became unreachable is done ({!Image.load}).
    {!Rig.section-claims} says when it claims that memory. *)
module Submission : sig
  type device := t

  type t
  (** The type for prepared submissions. *)

  (** {1:refs Refs} *)

  type ref
  (** The type for a launch's or a fill's references to a submit's buffers
      and to functions. A ref places a value in the 8 parameter bytes at
      [at] when the work runs. [at] must be a multiple of 8 and its 8 bytes
      must lie among the part's parameters, and two refs of a part must not
      share an [at].

      A {e buffer ref} names a {e slot}, an index of {!submit}'s [buffers].
      Its 8 bytes hold a byte offset into the slot's buffer, and when the
      work runs they hold that offset plus the address of the buffer's first
      byte, as the device's work addresses it. Through a buffer ref the work
      may address only that buffer's bytes, and write them only through a
      {!write} or an {!unordered} ref. rig orders the work by its buffer refs
      and checks neither rule.

      {!read}, {!write} and {!unordered} raise [Invalid_argument] if [slot]
      or [at] is negative or [at] is no multiple of 8, and {!entry} if [at]
      is. *)

  val read : int -> at:int -> ref
  (** [read slot ~at] is a ref through which the work reads the buffer
      [slot]. *)

  val write : int -> at:int -> ref
  (** [write slot ~at] is a ref through which the work reads and writes the
      buffer [slot], whose memory must admit writes ({!Buffer.val-access}). *)

  val unordered : int -> at:int -> ref
  (** [unordered slot ~at] is a ref through which the work reads and writes
      the buffer [slot] at any time, ordered only by device order, the
      exception to rule 3 of {!Rig.section-order}. The buffer's memory must
      admit writes. *)

  val entry : Image.func -> at:int -> ref
  (** [entry fn ~at] is a ref whose 8 bytes hold, when the work runs, the
      value by which the device's work refers to [fn] ({!Image.entry}). rig
      writes it over the run's bytes there. The submission keeps [fn]
      reachable, so the work may call [fn] through it. [fn]'s image must be
      loaded on the device. *)

  (** {1:parts Parts} *)

  type part
  (** The type for parts: a piece of a submission's work. *)

  (** The type for the ends of a copy. *)
  type side =
    | Slot of int  (** The buffer a submit passes as this slot ({!submit}). *)
    | Buffer of Buffer.t  (** This buffer, in every run. *)

  val copy : src:side -> dst:side -> part
  (** [copy ~src ~dst] copies [src]'s bytes into [dst]. The two buffers must
      have one length and share no byte ({!Buffer.overlaps}), and [dst]'s
      memory must admit writes. Each buffer must be on the device
      ({!Buffer.borrow} puts another device's memory there). On a driver's
      device of another machine, one of the two may instead be memory this
      process's host addresses, whose bytes the device's driver carries. A
      slot's buffer may be such memory only if copies alone name the slot.
      {!make} checks where the ends that are buffers are, and each {!submit}
      checks the ends that are slots, with its buffers.

      Raises [Invalid_argument] if [src] and [dst] are one slot or a slot is
      negative, a buffer end is dead, [dst] is a buffer whose memory admits
      only reads, or both ends are buffers and they differ in length or
      share a byte. *)

  val launch : ?refs:ref array -> Image.func -> params:int -> part
  (** [launch ~refs fn ~params] runs [fn] once, over the grid of groups of
      threads that its block in the run sets ({!val-block}), with the
      [params] bytes of parameters of that block and each of [refs]
      (defaults to [[||]]) placed ({!type-ref}), which [launch] reads once.
      [fn]'s image must be loaded on the device.

      Raises [Invalid_argument] if [params] is negative or above
      [2{^31} - 32], the most one block of a run holds ({!make}), or is
      fewer than the bytes [fn] reads where its driver knows them
      ({!Image.params}); or if two refs share an [at], or a ref's 8 bytes
      lie past [params] or past the bytes [fn] reads. *)

  val fill : ?refs:ref array -> nativeint -> params:int -> part
  (** [fill ~refs f ~params] calls the C function at [f] from the driver's
      [submit] ({!Rig_edge.section-calls}) with the queue's context, the host
      address of its argument and the work's value. The argument is a copy of
      the [params] bytes of parameters of the fill's block in the run
      ({!val-block}), with each of [refs] (defaults to [[||]]) placed
      ({!type-ref}), and is valid only during the call. The function must
      call no function of the OCaml runtime, since the driver's [submit] may
      call it with the runtime released. It returns [0], or another value for
      a failure, which loses the device.

      Raises [Invalid_argument] if [params] is negative or above
      [2{^31} - 32], or if two refs share an [at] or a ref's 8 bytes lie past
      [params]. *)

  val words : Buffer.t -> part
  (** [words b] places the 32-bit words of [b] on the device's queue, such as
      indirect-buffer or GPFIFO entries. Words may do what rig cannot see,
      such as start a thread trace or invalidate caches before a launch, so
      rule 2 of {!Rig.section-order} orders them after every earlier part,
      and {!Rig.Submission}'s preamble says which queue runs them. Memory the
      words address is ordered against other work only if it is among the
      submission's fixed memory. A driver may bound the words of one
      submission ({!Rig.section-valid}).

      Raises [Invalid_argument] if [b] is dead, the host does not address its
      memory, or its length is no multiple of 4. *)

  (** {1:making Making} *)

  val make :
    ?keep:'a -> ?fixed:(Buffer.t * access) list -> device -> part array -> t
  (** [make ~keep ~fixed d parts] is a submission of [parts] on [d]. Its
      slots are [0] to [n - 1], where [n] is one more than the highest slot
      its buffer refs and copies name, and each run passes one buffer per
      slot ({!submit}). One buffer may fill several slots. [fixed] (defaults
      to [[]]) is the submission's fixed memory. [make] reads [parts] once,
      and no bytes of the buffers, so a caller may write {!address}es into
      them afterwards.

      [keep] is a value the work uses beyond memory, such as the driver
      objects that a graph or an indirect command buffer made. The
      submission keeps it reachable while it is reachable. A call on [d]
      that finds the submission collected ({!Rig.section-reclaim}) lets it go
      once the work submitted on [d] before that call is done, or, on a lost
      [d], once [d] has stopped ({!Rig.section-stops}). In a child of [fork],
      on a driver's device the child inherited, the child's next call on [d]
      lets it go without waiting for the work. [keep] orders no memory.
      {!Rig.Submission}'s preamble states how the parts are ordered and
      which queue runs each.

      Raises [Invalid_argument] if the submission would not be valid
      ({!Rig.section-valid}), and {!Lost} if [d] is lost. *)

  type block
  (** The type for a launch's or a fill's block in a run of one submission
      ({!Run}). *)

  val block : t -> int -> block
  (** [block s i] is the block of [s]'s part [i] in a run ({!Run}). A fill
      reads nothing of its block but the parameters. The blocks of [s]'s
      launches and fills lie one after the other, in the order of its
      parts.

      Raises [Invalid_argument] if [s] has no part [i], or it is a {!words}
      or a {!copy} part. *)

  val address : t -> Buffer.t -> int
  (** [address s b] is the address of [b]'s first byte as the work of [s]'s
      device addresses it, where [b]'s bytes lie within a buffer of [s]'s
      fixed memory. A caller writes it into [s]'s blocks or into memory [s]
      fixes, such as an argument buffer or a replay's commands, before the
      submit that reads it. An empty buffer that {!Buffer.create} made on a
      driver's device has the address [0]. An address fits in the 62 bits of
      an [int]'s non-negative range.

      Raises [Invalid_argument] if [b] is dead or handle-only memory
      ({!Buffer.section-low}), or no buffer of [s]'s fixed memory holds [b]'s
      bytes. *)

  (** Runs. *)
  module Run : sig
    type submission := t

    type t
    (** The type for runs: the storage of one submission's launch and fill
        blocks, for one submit at a time. A library that submits from
        several domains keeps a few runs beside each submission it caches: a
        call takes the first run whose {!take} is [true], and otherwise makes
        one and adds it, and releases the run ({!release}) if the call
        raises before its submit. *)

    val make : submission -> t
    (** [make s] is a run of [s] holding every block of [s], outside the
        OCaml heap, taken by the calling thread ({!take}). Its bytes read [0]
        until stored, and keep their last store across submits.

        Raises [Stdlib.Out_of_memory] if the host has no memory for it. *)

    val submission : t -> submission
    (** [submission r] is the submission whose blocks [r] holds. *)

    val take : t -> bool
    (** [take r] reserves [r] for the calling thread's stores and its next
        submit of [r], and is [true]; or is [false], reserving nothing, if
        [r] is reserved or a submit of it is under way. It is one atomic step
        and allocates nothing. A run of a submission with no launch or fill
        holds no bytes: [take] is [true] and reserves nothing, and any number
        of submits of it run concurrently.

        A run stays taken until the thread that took it releases or submits
        it. A run taken by a thread that ends without doing either stays
        taken, as does, in a child of [fork], one that another thread of the
        parent had taken. *)

    val release : t -> unit
    (** [release r] ends the calling thread's reservation of [r], keeping
        its bytes. It does nothing if the thread does not hold [r]. *)

    (** {1:blocks Blocks}

        A launch's block holds its grid (its count of groups), the threads
        of each group, its shared memory and its parameters. A fill's holds
        its parameters. A run never grows. A setter stores only into a run
        the calling thread has taken, and raises [Invalid_argument], storing
        nothing, otherwise. It raises in particular while a submit of the
        run is under way, such as a signal handler's store during the
        submit's wait. A store reads the run's state once and takes no lock,
        and no setter allocates. A setter raises [Invalid_argument] for a
        block of another submission. *)

    external groups :
      t ->
      block ->
      (int[@untagged]) ->
      (int[@untagged]) ->
      (int[@untagged]) ->
      unit = "caml_rig_run_groups_byte" "caml_rig_run_groups"
    (** [groups r b x y z] sets the grid of [b] to [x] by [y] by [z] groups.

        Raises [Invalid_argument] if one is negative or above [2{^32} - 1]. *)

    external threads :
      t ->
      block ->
      (int[@untagged]) ->
      (int[@untagged]) ->
      (int[@untagged]) ->
      unit = "caml_rig_run_threads_byte" "caml_rig_run_threads"
    (** [threads r b x y z] sets each group of [b] to [x] by [y] by [z]
        threads.

        Raises [Invalid_argument] if one is negative or above [2{^32} - 1], or
        if [x * y * z] exceeds the most threads a group of [b]'s function may
        have, where its driver knows it ({!Image.threads}). *)

    external shared : t -> block -> (int[@untagged]) -> unit
      = "caml_rig_run_shared_byte" "caml_rig_run_shared"
    (** [shared r b n] sets the dynamic shared memory of each group of [b] to
        [n] bytes.

        Raises [Invalid_argument] if [n] is negative or above [2{^32} - 1]. *)

    (** {2:params Parameters}

        [int32 r b i v] stores [v] at byte [i] of [b]'s parameters, in the
        host's byte order, and the others store likewise for their sizes:
        [int32] the 32 low bits of [v], [int64] [v], [float32] [v] rounded to
        single precision, [float64] [v]. At a buffer ref's [at], the 8 bytes
        are the offset into its buffer ({!type-ref}). At an entry ref's, each
        submit writes the function's value ({!Submission.entry}).

        Each raises [Invalid_argument] if [i] is negative or the bytes it
        stores end past [b]'s parameters. *)

    external int32 :
      t -> block -> (int[@untagged]) -> (int[@untagged]) -> unit
      = "caml_rig_run_int32_byte" "caml_rig_run_int32"

    external int64 :
      t -> block -> (int[@untagged]) -> (int64[@unboxed]) -> unit
      = "caml_rig_run_int64_byte" "caml_rig_run_int64"

    external float32 :
      t -> block -> (int[@untagged]) -> (float[@unboxed]) -> unit
      = "caml_rig_run_float32_byte" "caml_rig_run_float32"

    external float64 :
      t -> block -> (int[@untagged]) -> (float[@unboxed]) -> unit
      = "caml_rig_run_float64_byte" "caml_rig_run_float64"
  end
end

(** {3:valid Valid submissions}

    A submission is {e valid} when its device runs submitted work and has a
    queue for each of its parts, as {!Rig.Submission} places them; its
    images, its parts' buffers and its fixed memory are live and on the
    device, but as {!Submission.copy} admits, and none of its fixed memory
    written is [Read]; every slot below the highest is named, and none by
    both an {!Submission.unordered} ref and an ordered use; and its launches
    and fills fit a run: [2{^31} - 1] bytes, each counting 16 bytes plus its
    parameters rounded up to a multiple of 16. {!Submission.make} raises for
    each.

    A submit is valid when it passes one buffer per slot; no buffer its work
    uses is dead or held exclusive ({!Claim}); each buffer is on the device
    ({!Buffer.borrow}), but as {!Submission.copy} admits; it writes no
    [Read] memory ({!Buffer.val-access}); each copy's two buffers have one
    length and share no byte; no ref names handle-only memory
    ({!Buffer.section-low}); and its driver can run the work
    ({!Rig_edge.section-calls}). A driver may bound the words one submission
    places, as its interface states. {!submit} raises for each. *)

(** {2:submits Submitting}

    A submit's work is ordered by the rules of {!section-order}. The
    device's queue waits for another device's point where its driver
    supports that ({!Rig_edge.type-waits}). Otherwise {!submit} waits for
    the point on the host. A submit for which the driver's [room] finds no
    room on the device's queues waits until the device has reached one more
    value, and asks again, holding no lock while it waits. A driver may
    instead block in its [submit] until the work fits, as Metal's does, and
    other submits and commits on the device then wait for it
    ({!section-domains}). *)

val submit :
  ?waits:Point.t array -> Submission.Run.t -> buffers:Buffer.t array -> Point.t
(** [submit ~waits r ~buffers] runs the work of [r]'s submission [s]
    ({!Submission.Run.val-submission}) once and returns its point [(d, v)],
    where [d] is [s]'s device and [v] is one more than {!submitted}[ d]. The
    work uses [buffers.(i)] as its slot [i] ({!Submission.make}), takes its
    blocks from [r], and follows the points of [waits] (defaults to [[||]])
    besides those its memory orders. [submit] reads each element of
    [buffers] once, and a store into [r] after it returns changes nothing of
    [v]'s work.

    [submit] uses [r] until it returns or raises, and frees it then, a
    submit made inside a lock ({!section-domains}) included: a run the
    calling thread has taken ({!Submission.Run.take}), or a free one, which
    [submit] takes itself. A run that holds no bytes it uses without taking
    it.

    [submit] blocks while [d]'s queues have no room for the work, while it
    waits for a point on the host, and where its driver's interface says it
    blocks, as some do while a profile is being taken. Where the work
    exceeds [d]'s queues and its driver grows them
    ({!Rig_edge.Driver.grow}), [submit] grows them first. It allocates
    nothing unless it waits, a profile is being taken or [d]'s queues grow,
    and, in a build without cross-module inlining, a [~waits] given as an
    array allocates its option. Once it has assigned [v], [d]'s driver
    receives [v]'s work or [d] is lost, and the stamps of every allocation
    the work uses name [(d, v)] either way. On {!host}, [v]'s work is done
    when [submit] returns, and the stamps name no point of it.

    Raises [Invalid_argument] if another thread has taken [r], another
    submit of [r] is under way, or the submit would not be valid
    ({!section-valid}). Raises {!Out_of_memory} if [d]'s queues must grow
    for the work and [d] has no memory for the growth after the rounds of
    reclamation ({!section-reclaim}). Raises {!Lost} if [d] is lost before
    [submit] returns, if its driver fails to take the work, or if a point of
    [waits] is a lost device's and that device never reached it. A submit
    after [d]'s value [2]{^ 47}[ - 1], the last a point holds, loses [d]
    with that reason and assigns nothing. *)

(** {1:per_device Values per device}

    A library keeps values per device under a {e key}, such as the image it
    loaded on each device and the workspace its code uses there. A key's
    value on a device is what the key's [init] makes at the first
    {!Local.get} on that device ({!Local.new_key}), and it lives as long as
    the device. Once the device's driver's stop ({!Rig_edge.Driver.val-stop})
    has returned, rig drops the value, so the memory it holds can return,
    and {!close} returns after the drop. A reopened name is a new device,
    with values of its own. *)

module Local : sig
  type device := t

  type 'a key
  (** The type for keys of values of type ['a] kept per device. *)

  val new_key : (device -> 'a) -> 'a key
  (** [new_key init] is a key whose value on a device [d] is [init d]. A key
      is never freed, so a library must make its keys once, such as at a
      module's top level. *)

  val get : 'a key -> device -> 'a
  (** [get k d] is [k]'s value on [d]. The first get of [k] on [d] runs
      [init d] and stores its result, which every later get answers. A get
      that finds its value takes no lock and allocates nothing.
      {ul
      {- [init d] runs on the calling domain holding no lock of rig's, so it
         may call any of rig's functions, [get] of any key included.}
      {- Gets that race on a device with no value may each run [init d]. The
         first value stored is kept and every get answers it, so [init] makes
         values of which a spare is harmless.}
      {- An [init] that gets [k] on [d] runs a nested [init], whose value is
         stored first, so both gets answer it.}
      {- If [init d] raises, [get] raises it and stores nothing.}}

      Raises {!Lost} if [d] is lost. *)
end

(** {1:claims Claims} *)

(** Claims on memory.

    A function may donate an input that dies with the call, such as an array
    operation's. A donation is safe only if nothing else reads the memory, and
    claims record what does. A claim lasts as long as its holder keeps it, and
    every view and borrow of an allocation shares the allocation's claims.
    {ul
    {- A {e read claim} marks an allocation as being read. The host holds
       one during {!Buffer.read} and {!Buffer.write}, and {!Claim.with_} on
       every buffer it is given. {!submit} holds one on every allocation
       the work uses until it returns, and a copy, blit or fill
       ({!Buffer.section-copying}) on its buffers until it returns. Work on
       a device holds none, since the stamps order a write in place after
       that work.}
    {- Only {!Claim.with_} holds memory exclusive, for an input its call
       donates, until it consumes the memory. Every other claim is a read
       claim.}
    {- Memory that something outside rig can read or write is
       {e outside the claims}
       and never exclusive. This holds for memory {!Buffer.of_bigarray} and
       {!Buffer.of_io} make, memory {!Buffer.export} exported, and memory
       {!Claim.share} marked. [Read] memory ({!Buffer.val-access}) is never
       exclusive either.}
    {- {!Claim.consume} consumes an allocation. A call that uses a dead
       buffer's bytes raises [Invalid_argument] with the reason given to the
       consumption. Consuming frees nothing, and the memory lives on in the
       buffer that {!Claim.consume} returns. Where the memory was held
       exclusive, that buffer's holder may write it in place, and no holder
       of a buffer made before the consumption sees the write. Code that the
       holder hands the new buffer sees the write, so the holder must order
       that code's accesses as for a buffer it made.}}

    Taking a claim never waits. A read claim on memory held exclusive raises
    [Invalid_argument], and a donated input whose memory another claim holds
    gets a read claim only.

    Which inputs a function donates is part of that function's contract, and
    the function decides alone. A caller that passes a buffer to such a
    function and still needs it afterwards must hold a read claim on it
    across the call. The donation then stays a read claim, and the function
    writes its result elsewhere. A host reader of memory that another
    domain may donate uses {!Buffer.read}, which claims before it waits,
    since a claim taken after {!Buffer.wait} returns leaves a moment in which
    the donation writes the memory. Host code over several buffers waits for
    them ({!Buffer.wait}) inside a {!Claim.with_} that reads them. Where no
    other domain may donate the memory, the reader may use {!Buffer.wait}.

    A prepared submission claims the memory it uses only while a submit
    runs. A donation between two submits can consume that memory, and the
    next submit then raises [Invalid_argument] for the dead buffer. A
    library that keeps a submission over memory it does not own holds a read
    claim on that memory for as long as it keeps the submission, or passes
    the memory to {!Claim.share}.

    A donor takes its inputs with {!Claim.with_}. It consumes an input it
    holds exclusive before it uses it, since every call that touches the
    bytes of memory held exclusive raises [Invalid_argument]. While its
    function runs, the donor uses the buffer that {!Claim.consume} returns as
    a buffer it made. It may write it from the host, copy into it, submit
    work on it, export it or share it. Every buffer made over the memory
    before the consumption is dead. A donor of one input may take it with
    {!Claim.take}, which returns the consumed buffer where it held the
    memory exclusive. *)
module Claim : sig
  val share : Buffer.t -> unit
  (** [share b] puts [b]'s memory outside the claims for good, through every
      view and borrow. A library calls it before it hands out a value over
      the memory that its own claims do not track, so that no donation writes
      what that value reads. {!Buffer.export} does the same to the memory it
      exports.

      Raises [Invalid_argument] if [b] is dead, or a {!with_} holds its
      memory exclusive. *)

  type t
  (** The type for the claims of a {!with_}. *)

  val with_ : read:Buffer.t list -> donate:Buffer.t list -> (t -> 'a) -> 'a
  (** [with_ ~read ~donate f] takes claims [c] and is [f c]. It takes a read
      claim on the memory of every buffer in [read] and [donate]. The claim
      of a buffer of [donate] becomes exclusive if the buffer {!Buffer.spans}
      its memory and no other claim holds that memory. An input whose buffers
      lie on several devices, such as the shards of an array split across
      GPUs, is donated when each of them is {!exclusive}.

      Every claim is released when [f] returns or raises, and [c] then holds
      nothing. A {!consume} on another domain that overlaps the return
      completes before the release or raises [Invalid_argument]. Any other
      use of [c] on another domain must finish before [f] returns. rig does
      not detect one that overlaps the return, which races the memory's next
      holder.

      Raises [Invalid_argument] before calling [f], releasing what it took,
      if a buffer is dead, an allocation is held exclusive, or a buffer of
      [donate] overlaps another of [read] or [donate]. *)

  val exclusive : t -> Buffer.t -> bool
  (** [exclusive c b] is [true] iff [c] holds [b]'s memory exclusive, so
      that the caller may consume it and write the buffer {!consume} returns
      in place ({!Rig.section-claims}). It is [false] once [c] has consumed
      the memory. *)

  val consume : t -> why:string -> Buffer.t -> Buffer.t
  (** [consume c ~why b] consumes [b]'s memory with the reason [why], so [b]
      and every buffer made over that memory until then are dead. It is a live
      buffer over the same memory, the only one. The caller must not write
      it in place unless [exclusive c b] was [true] before the call. [c] no longer
      holds the memory exclusive, and the buffer is claimed as any other.

      Raises [Invalid_argument] if [c]'s {!with_} has returned or raised, [c]
      does not claim [b]'s memory, or [b] is dead or does not
      {!Buffer.spans} its memory. *)

  val take : why:string -> Buffer.t -> Buffer.t option
  (** [take ~why b] donates [b] alone. It is [Some b'] if a {!with_} that
      donates [b] holds its memory exclusive, where [b'] is
      {!consume}[ c ~why b] under that claim, and [None] otherwise, which
      leaves [b] unchanged. The caller writes [b'] in place, and [b] is then
      dead. A function with several inputs takes them with {!with_}, which
      claims every input it reads.

      Raises as {!with_} does. *)
end

(** {1:profiles Profiles} *)

(** Profiles of devices' work.

    A profile of every device is taken while a function runs
    ({!Profile.take}). Profiles nest and overlap, in any domains, and each
    holds the events recorded while it is taken. Every time is on the host
    clock ({!Profile.now}). When no profile is being taken, recording costs
    one atomic read and allocates nothing. Taking a profile may slow the
    submits of a device whose driver times its work, as the driver's
    interface states. *)
module Profile : sig
  type device := t

  (** The type for profile events. Times are nanoseconds of the host clock.
      rig records a submission's work, spans, allocations, loads and copies.
      A library whose compiled code measures its work, such as with a
      device's counters or thread traces, records [Counters], [Trace] and
      [Overwritten] events ({!record}). *)
  type event =
    | Work of {
        point : Point.t;  (** The submit's point ({!submit}). *)
        device : device;
        lane : string;  (** Its track within the device. *)
        name : string;
        start : int;
        stop : int;
      }
        (** A submission's work on a lane of its device, from when the
            device started it to when the device ended all its parts' work,
            on the lane of the queue its first part ran on, such as
            ["COMPUTE:0"] ({!Rig_edge.type-queue}). Each submission with
            parts records it. It is named after the functions its launches
            run, in the order of its parts, joined by [", "], or, with no
            launch, after its first part's kind, ["copies"], ["fills"] or
            ["words"]. A device whose driver cannot time a submission's work
            records none of it.

            Its driver converts the device's clock to the host's through
            measures of one against the other. A time's error is the
            deviation of the measures it was converted with, as each driver
            states, plus the error of extrapolating since the latest. *)
    | Span of {
        device : device;
        lane : string;  (** Its track within the device. *)
        name : string;
        start : int;
        stop : int;
      }
        (** Host code that {!span} timed, on the calling domain's lane of
            {!host}: ["domain 0"], ["domain 1"], …. *)
    | Allocation of { device : device; time : int; allocated : int }
        (** The bytes of memory [device] allocated from [time] on. *)
    | Load of { image : Image.t; binary : string; time : int }
        (** An image loaded. *)
    | Counters of {
        device : device;
        name : string;
        start : int;
        stop : int;
        counters : (string * int array) list;
            (** Each counter's name and its count during the run, one per unit
                of the hardware that counts it. *)
      }  (** The counters of a run of a function. *)
    | Trace of {
        device : device;
        name : string;
        start : int;
        stop : int;
        part : int;  (** The index of the device's unit that wrote it. *)
        data : string;  (** The trace as the device wrote it. *)
      }  (** The thread trace that a unit of a device wrote during a run. *)
    | Overwritten of { device : device; time : int; runs : int }
        (** Runs whose counters and traces the device overwrote unread. *)
    | Copy of {
        src : device;
        dst : device;
        bytes : int;
        point : Point.t option;
            (** [Some p] for a copy a device runs, [p] its submission's
                point, and [None] for a copy the host or an io device
                runs. *)
        start : int;
        stop : int;
      }
        (** A transfer of [bytes] from [src]'s memory to [dst]'s: a copy part
            of a submission ({!Submission.copy}), rig's own and those of a
            {!Buffer.copy} among them, or a copy the host or an io device
            runs. [src] and [dst] are the devices in whose {!allocated} count
            the memory the bytes leave and reach lies ({!section-reclaim}),
            or the device whose memory it is where it counts in none. So a
            {!Buffer.borrow} names the device whose memory it borrows, and
            [Pinned] memory is the host's except on a device whose memory
            the host addresses.

            Each copy part of a submission records an event with the times
            of the submission's [Work] where its device times the work.
            Otherwise the event starts when the copy was asked, at the
            submit. It ends when the copy ended where the submit runs it, as
            for a copy the host, an io device or a {!memory_device} runs,
            and otherwise when {!take} reads that it is done, so the event
            then lasts at least as long as the copy. A copy through staging
            memory records each piece into and out of it, with {!host} as
            one side, and no event of its own. *)

  val take : (unit -> 'a) -> 'a * event list
  (** [take f] is [f ()] and the events recorded while it ran, in time order
      and, at equal times, longest first, then in the order they were
      recorded. Before it returns it waits for the work whose events are
      still to be read, such as the submissions made while it ran, and leaves
      out those of a device lost meanwhile. If [f] raises, [take] discards
      the events and raises the exception again with its backtrace. *)

  val enabled : unit -> bool
  (** [enabled ()] is [true] iff some profile is being taken. *)

  val now : unit -> int
  (** [now ()] is the host clock, in nanoseconds of the system's monotonic
      clock from an unspecified start. *)

  val record : event -> unit
  (** [record e] adds [e] to each profile being taken when it is called, so
      {!take} returns it among rig's own events, in time order. It records
      nothing while no profile is being taken, so counters a library reads
      after [take] has returned join no profile, even one of the work they
      count. *)

  val span : string -> (unit -> 'a) -> 'a
  (** [span name f] is [f ()], recorded as a span named [name] on the calling
      domain's lane of {!host}, from the call until [f] returns or raises, in
      each profile taken when the call starts. *)

  val output_chrome_trace : out_channel -> event list -> unit
  (** [output_chrome_trace oc events] writes [events] to [oc] in Chrome's
      trace event format, JSON, which Perfetto and [chrome://tracing] load.
      It writes a process per device, a thread per lane, a complete event per
      work and span, per copy (on its [src]'s process) and per run's counters
      (each summed over its units), a counter [memory] per allocation change,
      and instant events for loads, traces and overwritten runs. Times are
      microseconds from the earliest event. [oc] is neither flushed nor
      closed. *)
end
