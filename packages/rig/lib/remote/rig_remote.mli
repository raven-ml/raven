(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Jobs, in which one process uses the devices of other machines through an
    agent on each.

    A {e machine} is a host and the devices it drives
    ({!Rig.section-machines}). A {e job} is one process, the {e controller},
    and one {e agent} on each other machine of the job. An agent is a process
    that opens its machine's devices for the controller and runs the work the
    controller submits to them ({!serve}). For each other machine, the
    controller gets the machine's host ({!hosts}), and a {e proxy} of each
    device the agent opens ({!devices}): a device of the controller whose
    memory is that device's. A {e rail} carries a fixed set of transfers
    between host memory on two machines, again and again, without relaying
    them through the controller ({!Rig_remote_abi.section-rails}).
    {!section-order} follows a job's work from the controller's call to the
    agent's devices and back.

    A job's life goes like this:
    + Each other machine runs an agent with the job's key: [rig agent], or a
      process that calls {!listen} and then {!serve}.
    + The controller starts the job with {!connect}, or with {!launched} when
      [rig run] has started it and its agents.
    + The controller opens the job's devices ({!hosts}, {!devices}) and uses
      them through {!Rig} as it uses its own, loading rig.program programs on
      their machine ({!Rig_program.load}).
    + The controller ends the job with {!close}, or by exiting. A failure of
      any process ends the whole job ({!section-failure}).

    The functions of this module are safe to call from several domains in
    parallel. They are also safe to call from a signal handler, finaliser or
    [Gc.Memprof] callback that interrupts one of them on the same thread:
    such a call waits for no lock its thread holds, and raises
    [Invalid_argument] where a call to rig that it makes does
    ({!Rig.section-domains}). *)

(** {1:keys Keys and trust}

    A job assumes a trusted network. Every pair of the job's processes proves,
    on connecting, that both hold the job's key
    ({!Rig_remote_proxy.Wire.section-handshake}), and the key never crosses
    the job's connections.

    Nothing after the proofs is authenticated or encrypted. Whoever reads the
    network reads the bytes of copies and runs, and whoever writes it can
    change them. An agent runs the code the controller sends it, host code
    included. Listen only on a network that only the job's machines reach,
    such as a cluster's own. TCP's checksum is the only check on the bytes
    of copies, runs and rails, so a program that must know its data arrived
    intact checks it end to end.

    The dialing end proves the key first, so a process that replies in a
    listener's place gets a proof against which it can test guesses of the
    key. Make the key random, with
    [(umask 077 && head -c 32 /dev/urandom > FILE)], and read it with
    {!read_key}. *)

type key
(** The type for a job's keys: 16 to 4096 bytes. *)

val key : string -> (key, string) result
(** [key s] is [s] as a job's key, or [Error why] if it has fewer than 16 or
    more than 4096 bytes; [why] gives its length, never its bytes. *)

val read_key : string -> (key, string) result
(** [read_key file] is the key in [file]: its bytes, 16 to 4096 of them, read
    through one open of it. [file] must be a regular file. On POSIX systems it
    must also belong to this process's user and grant its group and others no
    access, so that only this user knows the key.

    [Error why] naming [file] if it cannot be opened or read, is not a regular
    file (a FIFO is refused without waiting for a writer), belongs to another
    user, grants its group or others any access, or holds too few or too many
    bytes. *)

(** {1:jobs Jobs}

    Every process of a job names each other machine by one name: its agent's
    ["HOST:PORT"] as given to {!connect}, or the [NAME] a launcher gives it
    ({!section-launchers}). The process's second connection to one address
    names that machine ["HOST:PORT#2"], the third ["HOST:PORT#3"], and so on
    ({!Rig.open_host}). A machine's host is named ["CPU@NAME"], and a proxy
    after the agent's device and its machine, as ["CUDA:3@h100-b"].

    {2:order Order and costs}

    Work on a device of another machine, from {!Rig.submit} or
    {!Rig.Buffer.copy}, goes to that machine's agent and back in four steps:
    + The proxy's driver sends the work as one message on the controller's
      connection to the agent. A copy from this process's memory carries its
      bytes in the message.
    + The messages to one agent leave in the order their calls made them,
      from every domain. The agent applies them in that order through its
      own rig, and finishes each submission's work before it applies the
      next message.
    + A copy into this process's memory comes back first, in a message of
      its own. Once the work of a value of the proxy's timeline is done, the
      agent reports the value.
    + The controller writes each copy's bytes into this process's memory as
      they come, then the value into the proxy's timeline word with release
      order. A wait on the proxy returns once the word has reached the
      value.

    So work submitted to a machine follows, on that machine, all the work
    submitted to it before, whichever of its devices runs it. rig orders
    work and host access by the memory they use across machines as on one
    ({!Rig}), and a proxy reports a copy into this process's memory reached
    once its bytes are there. A wait between two devices of one machine
    costs the controller nothing unless the submission copies from this
    process's memory.

    A submit returns once its message is sent. It blocks while the
    connection sends an earlier message, while its device has more than
    64 MiB of work not yet reported reached, and, before a copy from this
    process's memory, until the work that copy follows is done. Making a
    buffer, borrowing one and opening devices each take a round trip to the
    agent.

    A rail carries its bytes over TCP, on the job's connection between its
    two machines' processes. {!Rig.Buffer.copy} between devices of two other
    machines raises [Invalid_argument], unless it copies no bytes. Two copies
    through a buffer on {!Rig.host} carry the bytes through this process, and
    a rail between the two machines carries repeated transfers directly. *)

type t
(** The type for jobs, seen from their controller. *)

val connect : key:key -> (string * int) list -> (t, string) result
(** [connect ~key agents] starts a job with an agent at each host and port of
    [agents], and opens each machine's host ({!hosts}). Every pair of the
    job's processes connects and proves [key] ({!section-keys}). Each
    connection, handshake reply and connection between agents comes within
    10 seconds, or the job fails. The process closes the job at exit
    ({!close}), and fails it with the reason where that close raises
    [Invalid_argument].

    [Error why] if an agent cannot be reached, serves another job, speaks
    another version of the protocol or holds another key, or if two agents
    cannot connect to each other. [why] then starts with a machine's name and
    [": "], or names both machines that cannot connect, and the agents
    reached are told the job has failed. [Error why], [why] starting with the
    first machine's name and [": "], if the process has failed ({!Rig.fail}).

    Raises [Invalid_argument] if [agents] is empty or lists an address twice, or
    if a job of the process is open. It first fails an open job that a loss on
    this process fails ({!section-failure}), without the second the process
    may take to notice the loss. *)

val hosts : t -> Rig.t list
(** [hosts j] is the host of each of [j]'s machines, in the order of their
    agents in {!connect}'s list or in [RIG_REMOTE_AGENTS]
    ({!section-launchers}). A host's {!Rig.arch} is the agent's instruction
    set. It runs copies between its machine's memory and this process's, and
    loads rig.program programs for devices of its machine: {!Rig.Image.load}
    on it is the agent's {!Rig_program.load_received}, and a
    {!Rig.Submission.words} part on it is a run ({!Rig_program.run_received}).
    Its capability record (under {!Rig_remote_abi.key}) is a
    {!Rig_remote_abi.Host}, which holds the function that makes rails. *)

val devices : Rig.t -> string -> (Rig.t list, string) result
(** [devices h kind] is a proxy of each device of [kind], such as ["CUDA"],
    that the agent of [h]'s machine serves ({!serve}), in the agent's index
    order. The first call for [kind] opens them, and later calls give the
    same devices. A proxy's {!Rig.arch} and {!Rig.budget} are those of the
    agent's device when it opened, and {!Rig.reaches} between devices of one
    machine is as the agent's rig has it. A proxy runs copies, and
    {!Rig.Image.load} on it is [Error]. Its capability record is a
    {!Rig_remote_abi.Device}, which holds only the device's id, so code
    compiled here cannot build a vendor object for it, such as a CUDA graph.

    [Error why] if the agent serves no devices of [kind] or its opener fails,
    [why] starting with the machine's name, or if the job has failed or
    closed, [why] its root cause or that it has closed.

    Raises [Invalid_argument] if [h] is not a host that {!hosts} gave for the
    process's last job, or as {!Rig.open_} does for a device of [kind], such
    as from a callback inside that device's open. *)

(** {1:launchers Launchers}

    A {e launcher}, such as [rig run], starts a job's processes and learns how
    the job ends from each process's {e report}: a descriptor open for
    writing, whose decimal number it passes in [RIG_REMOTE_REPORT], on
    Windows that of a C runtime descriptor. A process writes on its report at
    most one ["started"] line and one end line, each in one write. Newlines
    in a reason are written as spaces.
    - ["started"], from the controller, once the job has started;
    - ["failed WHY"] once the job has failed, [WHY] its root cause, and on an
      agent before its process fails ({!Rig.fail});
    - ["closed"] once the process has ended the job in order, the controller
      by {!close} or at exit, an agent once {!serve} has dropped what the
      closed job held.

    A process takes [RIG_REMOTE_REPORT] out of its environment and sets the
    descriptor close-on-exec, and a child of [fork] writes nothing. The
    launcher gives the controller two more variables:
    - [RIG_REMOTE_AGENTS]: the job's agents, comma-separated, each as
      [NAME=ADDRESS:PORT], an IPv6 [ADDRESS] in brackets, as
      ["h100-b=10.0.0.2:41234,[fd00::2]=[fd00::2]:41234"]. An IPv6 [ADDRESS]
      without brackets is malformed, since in ["fd00::2:4123"] the ["4123"]
      may be its last group or the [PORT];
    - [RIG_REMOTE_KEY]: the job's key, 64 hexadecimal characters whose bytes
      are the key without decoding. *)

val launched : unit -> (t, string) result option
(** [launched ()] starts the job that a launcher set up for this process, or
    is [None] if [RIG_REMOTE_REPORT] is not in the environment. It takes the
    launcher's three variables out of the environment, starts the job as
    {!connect} does, with an agent at each [ADDRESS] and [PORT] named [NAME],
    and reports on the descriptor.

    [Some (Error why)], after it has written ["failed WHY"], if
    [RIG_REMOTE_AGENTS] or [RIG_REMOTE_KEY] is missing or malformed, or the
    job does not start, as {!connect} says; the process should then exit.
    [Some (Error why)], with nothing written, if [RIG_REMOTE_REPORT] names no
    open descriptor.

    Raises [Invalid_argument] if the process called [launched] before, or if
    a job of the process is open, which it checks as {!connect} does, once
    the descriptor and the variables are valid. *)

(** {1:failure Failure}

    A job {e fails} once, as a whole. Each of its processes records the
    first failure it learns of as its {e root cause} ({!failed}), which is
    one of:
    - a connection between two of its processes that ends without a close;
    - no byte on a connection for 10 seconds, though each end sends at least
      once a second whatever its process runs, so host code that never
      returns keeps its job going;
    - a send on a connection that makes no progress for 10 seconds;
    - a malformed message;
    - work or a release that names memory, a program or an object the job
      does not hold on its machine;
    - a copy or a run that the agent's rig refuses or raises
      [Invalid_argument] or {!Rig.Out_of_memory} for
      ({!Rig_program.run_received});
    - a device of an agent lost other than by a close ({!Rig.failure}),
      which the agent notices within a second;
    - on the controller, the loss of one of the job's hosts or proxies other
      than by a close, as {!Rig.fail} loses them too, which the controller
      notices within a second. A loss of another of the controller's
      devices reaches nothing of the job, which goes on;
    - an exception an agent does not expect, such as one an opener of
      {!serve} raises, whose text ([Printexc.to_string]) is the root cause;
    - an abort from another process of the job, which carries that
      process's root cause [WHY], recorded as ["NAME aborted: WHY"].

    In a root cause, [NAME] is the process at the other end of a connection:
    its machine's name, or ["controller"] for the controller. A connection's
    end names it too, as in ["NAME: closed its connection"]. So a root cause
    that starts ["NAME aborted: "] came from another process, and every other
    is the process's own.

    Once the job has failed, each of its processes still running raises
    every count of every rail ({!Rig_remote_abi.section-rails}). It sends the
    other processes an abort with its root cause, after the message each
    connection is sending, and ends its side of each connection. It then
    reads and discards what arrives on each connection until the other end
    has ended its side too, until 10 seconds after the failure or after the
    abort, if it left later behind a message, and no longer than 10
    seconds after the other end's last byte, and closes the connection only
    then, since a connection closed over unread bytes is reset, which may
    lose the abort. {!serve} returns, and {!connect} and {!launched} return
    their [Error], once every connection of the job is closed, and the
    controller's process waits for its last job's connections at exit. A
    process misses an abort only when the network fails, when it reads
    nothing of the connection for 10 seconds, or when the aborting process
    ends otherwise, such as by a signal. It then records its own cause, such
    as the end of the connection.

    An agent fails its process with the root cause ({!Rig.fail}) before it
    returns from {!serve}. On the controller, each host and proxy of the job
    faults with the root cause, so its next use loses it and raises
    {!Rig.Lost}, as every use after does. The controller's own devices go
    on, and it may start another job.

    A child of [fork] never uses its parent's connections. In the child the
    job is failed from the start, its rails' counts raised as on any
    failure, and the parent's job goes on. *)

val failed : t -> string option
(** [failed j] is [Some why] if [j] has failed, [why] the root cause this
    process recorded, and [None] otherwise. It raises nothing and waits for
    nothing. *)

val close : t -> unit
(** [close j] ends [j] in order. It closes each machine's host ({!Rig.close}),
    which waits for the work submitted on each of the machine's devices and
    closes them first. Each agent then receives a close. It closes the
    devices it opened, which frees their memory, and lets go of the job's
    objects on its host, whose memory returns once they are collected. It
    then stops listening and returns from {!serve}. [close] returns once
    every agent has stopped listening, or immediately if [j] has closed. On
    a failed [j] it only closes each machine's host, which stops the
    machine's lost devices, so their memory returns ({!Rig.section-stops}),
    and returns without waiting for their work. After a close, {!failed} of
    [j] is [None], and the process may start another job. A job that fails
    during [close] ends as a failed job does ({!section-failure}): [close]
    returns once every connection has closed, up to about 11 seconds after
    the failure, or after an abort that left later behind a message, and
    does not return to OCaml meanwhile, so a [Sys.Break] from Ctrl-C raises
    only then. The process's exit waits the same way for its last job.

    Raises [Invalid_argument] as {!Rig.close} does, such as from a callback
    inside a call of rig's on the same thread. *)

(** {1:agents Agents} *)

type agent
(** The type for agents of this machine. *)

val listen : key:key -> string -> int -> (agent, string) result
(** [listen ~key host port] listens at [host] and [port] for the processes of
    one job that prove [key]. Port [0] lets the system choose ({!port}). It
    opens no device and joins no job until a controller proves [key].
    Each connection proves the key on a thread of its own, and one that
    proves nothing within 10 seconds is dropped. The agent greets every
    connection, and refuses one before its proof only when the system gives
    it no thread. When the system lacks a descriptor or memory to accept a
    connection, the agent accepts again at most a second later. Linux keeps
    that connection waiting; macOS closes it.

    [Error why] if [host] does not resolve or the process cannot listen
    there. *)

val port : agent -> int
(** [port a] is the port [a] listens at: the one {!listen} was given, or the
    one the system chose for [0]. *)

val serve :
  agent ->
  (string * (unit -> (Rig.t list, string) result)) list ->
  (unit, string) result
(** [serve a kinds] makes the process the agent of one job on its machine, and
    returns when the job ends. It waits for a controller that proves [a]'s
    key, then for the job's other agents, and tells a second controller that
    it serves another job. It opens the devices of a kind of [kinds] with the
    kind's opener at the controller's first {!devices} call for it, and runs
    the work the controller submits from the calling domain
    ({!section-order}). That work, and the openers of [kinds], run as the
    program's own code: rig's calls they make block signals as {!Rig}'s
    Errors section states, and a signal's handler may run between them, its
    exception ending [serve] as an asynchronous one. [serve]'s own steps
    block signals but in their waits, except on Windows, where nothing
    blocks them ({!Rig}'s Errors section).

    [Ok ()] once the controller has closed the job, as {!close} states.
    [Error why] once the job has failed, [why] its root cause
    ({!section-failure}). Work of a GPU that no kernel driver resets may then
    still reach memory the process holds, so the process must exit. [Error
    why] also once accepting a connection fails with an error other than a
    lack of descriptors, buffers or memory, or a failure of the connection
    it took: [why] names it, and a job [a] serves fails with it. [a] stops
    listening when [serve] returns.

    An agent that a launcher started reports its job's end
    ({!section-launchers}), and returns [Error why] immediately if
    [RIG_REMOTE_REPORT] names no open descriptor.

    Raises [Invalid_argument] if [a] served already, or [kinds] names a kind
    twice. Raises again an exception that ends one of [serve]'s own steps,
    such as [Sys.Break] from a Ctrl-C in its wait for a controller, once [a]
    has stopped listening. A job [a] serves then, if it has not ended, fails,
    and as for [Error why], the process's devices are lost ({!Rig.fail}). *)
