(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices of other machines, through an agent on each.

    A {e job} is one process, the {e controller}, and one {e agent} on each
    other machine: a process that opens its machine's devices for the controller
    and runs the work the controller hands them ({!serve}). The controller
    {!connect}s to every agent of the job at once, and gets, for each other
    machine, devices of two kinds:
    - the machine's {e host} ({!hosts}): a device whose memory is the agent's;
    - a {e proxy} of each device the agent opens ({!devices}): its memory is
      that device's.

    Both copy between their machine's memory and this process's. A program uses
    them through {!Rig} as it uses this machine's devices. {!Rig.host_of} of
    each is the machine's host: two devices are of one machine iff their hosts
    are equal.

    {v
       controller                                   agent of machine B
      ───────────────────────────────────          ─────────────────────────
       hosts j, devices h ─ alloc, free ────────>  Rig.Buffer.create
                          ── hand-overs ────────>  Rig.Buffer.copy
                         <── words, copies' bytes  copies done
    v}

    {1:order Order}

    Every call on a job's device that reaches its agent is a {e frame} on the
    connection, and a device's hand-over of one value is one frame. Frames leave
    in the order their calls made them, from every domain, and the agent applies
    them in that order through its own rig, each hand-over's copies to their end
    before the next frame. So work the controller hands over follows, on that
    machine, all the work handed over before it, whichever device's it was.

    A device's word, the host's included, advances in value order, and only once
    the agent's devices reached the work of the value: for a copy into this
    process's memory, once the copy's bytes are in that memory. Values are
    assigned here, under each device's turn, as for any device. A wait between
    two devices of one machine travels in the hand-over, and the agent's order
    keeps it.

    {1:costs Costs}

    A hand-over costs a copy of its parts into the connection's queue, which a
    thread of the connection sends: it returns once queued. It waits while the
    queue is full, while its device has more than 64 MiB of work handed over and
    not done, and, for a copy from this process's memory, until the work that
    copy follows is done. An allocation, a mapping and opening devices each wait
    for the agent's answer, a round trip. A copy between this process's memory
    and a machine's carries its bytes across the connection once; a copy between
    two devices of one machine never crosses it.

    {1:failure Failure}

    The job {e fails} once, as a whole, with its first failure as the
    {e root cause} ({!failure}), when:
    - a connection between two of its processes ends without a close;
    - no byte comes on a connection for 10 seconds, though each end sends at
      least once a second, or a send on one makes no progress for 10 seconds;
    - a frame is malformed;
    - a hand-over names memory the job does not hold on its machine, or carries
      words, which agents do not run, or a release names an object the job does
      not hold there;
    - a device of any of its processes is lost other than by a close, as
      {!Rig.failure} reports it. This process notices its own within a second.

    Then, on each of its processes that still answer: every connection is shut
    down, every count of a rail is raised so that no wait for one blocks, and
    every device of the process but {!Rig.host} is lost with the root cause
    ({!Rig.fail}). Agents then return from {!serve}. Here the call in progress,
    and every later use of any device of the process but {!Rig.host}, raises
    {!Rig.Lost} with the root cause. The process stays failed: it starts no
    other job ({!connect}).

    In a process in no job, a lost device fails alone, as {!Rig} states; the
    loss is still the process's failure, so it starts no job afterwards.

    A child of [fork] never uses its parent's connections, whose streams it
    would interleave with the parent's: there the job is failed from the start,
    and the parent's job goes on.

    {1:security Security}

    Each pair of the job's processes proves, on connecting, that both hold the
    job's key, by HMAC over BLAKE2b-256 of fresh random nonces of both; the key
    never crosses the network. A process that does not hold the key is refused,
    so an agent of another job, or one left from an earlier job, is refused when
    each job has a key of its own. Nothing after the proofs is authenticated or
    encrypted: whoever reads the network reads the bytes of copies and of runs,
    and whoever writes it can change them. Listen only on a network that only
    the job's machines read and write, such as the cluster's own. The dialing
    end proves the key first, so whoever answers in a listener's place learns a
    proof against which it can test guesses: make the key random, with
    [head -c 32 /dev/urandom > FILE && chmod 600 FILE], and read it with
    {!read_key}.

    {1:references References}

    - H. Krawczyk, M. Bellare and R. Canetti.
      {{:https://www.rfc-editor.org/rfc/rfc2104}RFC 2104},
      {e HMAC: Keyed-Hashing for Message Authentication}: the proof, over a hash
      of 128-byte blocks.
    - M-J. Saarinen and J-P. Aumasson.
      {{:https://www.rfc-editor.org/rfc/rfc7693}RFC 7693},
      {e The BLAKE2 Cryptographic Hash and Message Authentication Code}:
      BLAKE2b, its block and digest sizes. *)

(** {1:jobs Jobs} *)

type t
(** The type for jobs, seen from their controller. Every function may be called
    from any domain at once. *)

val connect : key:string -> (string * int) list -> (t, string) result
(** [connect ~key agents] starts a job with an agent at each host and port of
    [agents]. It connects to each agent, proves [key] to it and checks that it
    proves [key] back, has the agents connect to each other likewise, and opens
    each machine's host ({!hosts}). Each connection, and each answer of a
    handshake, comes within 10 seconds, or the job fails. Once it returns, the
    job's processes watch each other, and the process closes the job at exit
    ({!close}).

    Each machine's name is ["HOST:PORT"] as [agents] gives them, followed by
    ["#n"] for the process's [n]th connection to that address from the second
    on: a name names one machine for the life of the process, so a device opened
    on one job's machine is never another's.

    [Error why] if an agent cannot be reached, serves another job, speaks
    another version of the protocol, or does not know [key], if the controller
    does not, or if two agents cannot connect to each other, [why] starting with
    ["HOST:PORT: "]; and if the process failed ({!Rig.failure}), [why] its
    failure: a process whose job failed, or that lost a device before, starts no
    job. The agents reached are told the job failed.

    Raises [Invalid_argument] if [key] has fewer than 16 or more than 4096
    bytes, [agents] is empty or lists an address twice, or a job of the process
    is open. *)

val hosts : t -> Rig.t list
(** [hosts j] is the host of each of [j]'s machines, in the order {!connect} was
    given their agents. A host is named ["CPU@NAME"], [NAME] its machine's name,
    and is {!Rig.host_of} of every device of its machine. Its {!Rig.arch} is the
    agent's instruction set, and its memory is the agent's. Its queues are
    ["COMPUTE:0"] and ["COPY:0"]; it runs copies between its memory and this
    process's ({!Rig.Buffer.copy}). It loads no code: {!Rig.Image.load} on it is
    [Error].

    Its capability record ({!Rig.capability}, {!Rig_remote_abi.key}) is a
    {!Rig_remote_abi.Host}, whose function makes rails between its machine and
    another of the job, or this process's. *)

val devices : Rig.t -> string -> (Rig.t list, string) result
(** [devices h kind] is a proxy of each device of [kind] that the agent of [h]'s
    machine serves ({!serve}), [h] a host of {!hosts}, in the agent's index
    order, opened at the first call for [kind] and the same devices at the next
    ones. A proxy's name is the agent's name for the device followed by
    ["@NAME"], [NAME] the machine's name, as ["CUDA:3@h100-b:7000"]. Its
    {!Rig.arch} and {!Rig.budget} are the agent's device's when it opened, and
    {!Rig.reaches} answers between devices of the machine as the agent's rig
    does. It loads no code: {!Rig.Image.load} on it is [Error]. Its capability
    record is a {!Rig_remote_abi.Device} with the device's id on its machine.

    A proxy copies between memory of its machine and this process's memory, as
    one {!Rig.Submission.Copy} the agent runs ({!Rig.Buffer.copy}). This process
    writes the bytes a copy brings back only into the memory its copy named.

    [Error why] if the agent serves no devices of [kind] or its opener fails,
    [why] starting with the machine's name, or if the job failed or was closed,
    [why] its root cause or that it was closed.

    Raises [Invalid_argument] if [h] is no host that {!hosts} gave for the
    process's last job. *)

val failure : t -> string option
(** [failure j] is [Some why] if [j] failed, [why] its root cause, and [None]
    otherwise. It raises nothing and waits for nothing. *)

val close : t -> unit
(** [close j] ends [j] in order. It closes each machine's host ({!Rig.close}),
    which waits for the work submitted on each of the machine's devices and
    closes them first; then each agent receives a close, releases what the job
    held on its machine and returns from {!serve}. It returns once every agent
    did, or at once if [j] failed or was closed. A close is no failure of the
    job or of the process. *)

(** {1:agents Agents} *)

type agent
(** The type for agents of this machine. *)

val listen : key:string -> string -> int -> (agent, string) result
(** [listen ~key host port] listens at [host] and [port] for the processes of
    one job that prove [key]. Port [0] lets the system choose ({!port}). It
    holds nothing until a controller proves [key]; connections beyond 64 waiting
    for their proofs are told it has too many, and a connection that proves
    nothing within 10 seconds is dropped.

    [Error why] if [host] does not resolve or the process cannot listen there.

    Raises [Invalid_argument] if [key] has fewer than 16 or more than 4096
    bytes. *)

val port : agent -> int
(** [port a] is the port [a] listens at: the one {!listen} was given, or the one
    the system chose for [0]. *)

val serve :
  agent ->
  (string * (unit -> (Rig.t list, string) result)) list ->
  (unit, string) result
(** [serve a kinds] makes the process the agent of one job on its machine, and
    returns when the job ends. It waits for a controller that proves [a]'s key,
    then for the job's other agents; a second controller is told the agent
    serves another job. It opens the devices of a kind of [kinds] at the
    controller's first {!devices} call for it, with the kind's opener, runs the
    work the controller hands over, and advances each device's word as
    {{!order}Order} says. It applies the controller's frames from one domain, in
    the order they were made.

    The result is [Ok ()] once the controller closed the job ({!close}), every
    device's work handed over is done and the job's memory is released, and
    [Error why] once the job failed, [why] its root cause, every device of the
    process lost. After [Error], a GPU that no kernel driver resets may still
    run into memory the process holds: exit the process. [a] listens no more
    once [serve] returns.

    Raises [Invalid_argument] if [a] served already, or [kinds] names a kind
    twice. *)

(** {1:keys Keys} *)

val read_key : string -> (string, string) result
(** [read_key file] is the key in [file]: its bytes, 16 to 4096 of them, read
    through one open of it. [file] must be a regular file; on POSIX systems it
    must also belong to this process's user and grant its group and others no
    access, so that only this user knows the key.

    [Error why] naming [file] if it cannot be opened or read, is no regular
    file, belongs to another user, grants its group or others any access, or
    holds too few or too many bytes. *)
