(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Links: this process's ends of its job's connections, in C.

    A {e link} is this process's end of its connection to one other process of
    its job, once the handshake admitted it ({!Wire}). It owns the socket and
    two threads of C, which never hold the OCaml runtime:
    - the {e sending thread} sends the link's queue in order. Hand-overs from
      the proxies' C and the frames of this module's functions enter the queue;
      a full queue makes its writer wait. The thread also sends the transfers of
      the link's rails ({!rail}), and a beat after a second in which it sent
      nothing. It waits on nothing but its queue and its socket.
    - the {e receiving thread} reads each frame. It places the transfers of the
      link's rails, and on the controller, writes the bytes of copies into this
      process's memory, advances the proxies' words and answers {!request}s. It
      queues every other frame for {!next}, a hand-over with its bytes.

    {1:failure Jobs and failure}

    The links of a process belong to its {e job} ({!job}), which fails or closes
    as a whole. The job {e fails} once, with the first of these as its
    {e root cause}:
    - a link's stream breaks or ends without a close:
      ["NAME: closed its connection"], or the system's error after ["NAME: "];
    - no byte comes on a link for 10 seconds: ["NAME: silent for 10 s"];
    - a frame is malformed: ["NAME: a malformed frame"];
    - a peer aborts: its reason, unchanged;
    - {!fail}.

    [NAME] is the link's {!name}. Then every link of the job sends its peer an
    abort with the root cause if its stream takes it at once, and shuts its
    socket down, so that no thread waits on it. Its queue drops what it is
    given, {!request} and {!next} answer [Error], the proxies' sleeps return,
    and every count of every rail of the job reads [Int64.max_int]. Nothing
    raises from a C thread, and nothing calls OCaml: the process learns of the
    failure from {!failure}, {!wait} or a function's [Error].

    A child of [fork] never uses its parent's links, whose streams it would
    interleave with the parent's: every function compares the process id with
    the one that started the job before anything else, and in a child the job is
    failed with ["a child of fork does not use its parent's connections"], in
    the child's copy alone. The child sends nothing and closes nothing.

    {1:domains Domains}

    Every function may be called from any domain at once. A function that waits,
    for room in a queue, an answer, a frame or the job, releases the runtime
    while it waits.

    A job and its links are never freed: proxies' C state holds their addresses.
    A link that ended keeps its name and reason. *)

(** {1:jobs Jobs} *)

type job
(** The type for the jobs of this process. A job is {e open} from {!job} until
    it fails or closes. *)

val job : unit -> job
(** [job ()] is a new open job, with no link.

    Raises [Invalid_argument] if a job of the process is open. *)

(** The type for the states of a job. *)
type state =
  | Open  (** Neither failed nor closed. *)
  | Closed  (** Closed in order ({!close}). *)
  | Failed of string  (** Failed, with its root cause. *)

val wait : job -> ms:int -> state
(** [wait j ~ms] is [j]'s state once it is no longer [Open], or after [ms]
    milliseconds. *)

val failure : job -> string option
(** [failure j] is [Some why] iff [wait j ~ms:0] is [Failed why]. *)

val fail : job -> string -> unit
(** [fail j why] fails [j] with the root cause [why], unless it failed or
    closed. *)

val close : job -> unit
(** [close j] ends [j] in order: each of its links sends a close after the
    frames queued before, and [close] returns once every peer's close came and
    the links' threads ended, or once [j] failed. A link's peer sends its close
    when it ends the job itself, so [close] waits for every peer to end it. On a
    job that failed or closed it returns at once. *)

(** {1:links Links} *)

type t
(** The type for links. *)

val make : job -> Unix.file_descr -> name:string -> peer:Wire.process -> t
(** [make j fd ~name ~peer] is a link of [j] over the connected socket [fd],
    which it takes, whose handshake admitted it, to the process [peer] named
    [name]: its machine's name, which starts its reasons. Its threads start at
    once. On a failed job the link is failed and [fd] closed.

    Raises [Invalid_argument] if [j] is closed. *)

val name : t -> string
(** [name l] is [l]'s name. *)

val job_of : t -> job
(** [job_of l] is [l]'s job. *)

val fresh : unit -> int
(** [fresh ()] is an id the process gave no object before: the controller names
    each object it makes on an agent by one ({!Wire.request}). *)

(** {1:controller The controller's end} *)

val request :
  t ->
  'a Wire.request ->
  ('a, [ `Refused of string | `Failed of string ]) result
(** [request l r] sends [r] after every frame queued before it and is the
    agent's answer: [Error (`Refused why)] if the agent refused [r], the job
    going on, and [Error (`Failed why)] if the job failed, before or meanwhile,
    [why] its root cause. An answer that does not decode as [r]'s fails the job.
*)

val drop : t -> int -> unit
(** [drop l id] sends the release of the agent's object [id] after every frame
    queued before it. *)

(** {1:agent The agent's end} *)

val next : t -> (Wire.command, string) result
(** [next l] is the next command [l]'s peer sent, waiting for it. [Error why]
    once the job failed, [why] its root cause. A frame that does not decode as a
    command fails the job. *)

val answer : t -> 'a Wire.request -> ('a, string) result -> unit
(** [answer l r a] sends [a] as the answer to [r]. [Error why] refuses [r] with
    [why].

    Raises [Invalid_argument] if [r] is not, physically, the oldest request
    {!next} gave and none answered. *)

val word : t -> device:int -> int -> unit
(** [word l ~device v] sends that the work of [device] up to [v] is done. *)

val bytes : t -> device:int -> value:int -> Rig_remote_abi.area -> unit
(** [bytes l ~device ~value b] sends [b]'s bytes as those of the next copy into
    the controller's memory of the work of [device] at [value]. It copies [b]
    before it returns. *)

(** {1:rails Rails} *)

val rail :
  t ->
  id:int ->
  send:Rig_remote_abi.transfer array ->
  receive:Rig_remote_abi.transfer array ->
  Rig_remote_abi.end_
(** [rail l ~id ~send ~receive] is this machine's end of the rail [id] to [l]'s
    peer, which carries [send] to it and [receive] from it, with the rail's
    areas, as [Rig_remote_abi] states them, zeroed, and its [ready] function.
    From then on the sending thread, which the [ready] function wakes, sends
    transfer [j]'s bytes once [ready] reaches its count, and stores [sent] once
    they are sent; the receiving thread places arriving transfers and stores
    [arrived].

    Raises [Invalid_argument] if [l] has a rail [id], [send] and [receive] are
    both empty, or a transfer's [length] is not positive or its [src] or [dst]
    is negative. *)

val release_rail : t -> int -> unit
(** [release_rail l id] ends [l]'s rail [id] here: once it returns, neither
    thread reads or writes its end, whose memory lives while it is reachable. A
    transfer of it that arrives later fails the job. It does nothing if [l] has
    no rail [id]. *)
