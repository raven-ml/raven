(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Links: this process's ends of its job's connections, in C.

    A {e link} is this process's end of its connection to one other process of
    its job, once the handshake admitted it ({!Wire}). It owns the socket and
    two threads of C, which never hold the OCaml runtime. A frame, a hand-over
    from a proxy's C or one of this module's functions, goes from its writer's
    thread: the writer waits for the frame being sent, sends its own whole, and
    returns once it is sent. The socket is the only back-pressure.
    - the {e sending thread} sends what the [ready] functions of the link's
      rails ({!rail}) leave, and a beat after a second in which nothing was
      sent, between writers' frames. It waits on nothing but the writers and
      its socket.
    - the {e receiving thread} reads each frame. It places the transfers of the
      link's rails, and on the controller, writes the bytes of copies into this
      process's memory, advances the proxies' words and delivers the answers to
      {!request}s. It queues every other frame for {!next}, a hand-over with its
      bytes.

    Frames carry no integrity check after the handshake, as {!Wire} states.

    {1:failure Jobs and failure}

    The links of a process belong to its {e job} ({!job}), which fails or closes
    as a whole. The job {e fails} once, with the first of these as its
    {e root cause}:
    - a link's stream breaks or ends without a close:
      ["NAME: closed its connection"], or the system's error after ["NAME: "];
    - no byte comes on a link for 10 seconds: ["NAME: silent for 10 s"];
    - a link's peer takes no byte of a send for 10 seconds:
      ["NAME: read nothing for 10 s"];
    - a frame is malformed: ["NAME: a malformed frame"];
    - a frame is larger than this process can hold:
      ["NAME: a frame larger than this process can hold"];
    - a peer aborts: its reason, unchanged;
    - {!fail}.

    [NAME] is the link's {!name}. A reason is any bytes, at most 4096 of them: a
    longer one is cut there, here and in an agent's refusal ({!answer}).

    Then every link of the job that sent no close sends its peer an abort with
    the root cause, after the frame it is sending if any, and ends its stream.
    It reads and discards what the peer still sends until the peer ends its
    stream or is silent for 10 seconds, so the abort reaches a peer still
    reading earlier frames. It sends no frame it is given, {!request} and
    {!next} answer [Error], the proxies' sleeps return, and every count of every
    rail of the job reads [Int64.max_int]. Nothing raises from a C thread, and
    nothing calls OCaml: the process learns of the failure from {!failure},
    {!wait} or a function's [Error].

    A child of [fork] never uses its parent's links, whose streams it would
    interleave with the parent's: every function compares the process id with
    the one that started the job, and in a child the job is failed with
    ["a child of fork does not use its parent's connections"], in the child's
    copy alone. The child sends nothing on its parent's links and closes none of
    them; a link it makes is failed, its socket closed.

    {1:domains Domains}

    Every function may be called from any domain at once. A function that waits,
    for the socket, an answer, a frame or the job, releases the runtime while it
    waits.

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
    frames sent before, and [close] returns once every peer's close came and the
    links' threads ended, or once [j] failed. A link's peer sends its close when
    it ends the job itself, so [close] waits for every peer to end it. Once a
    link took its close, it sends no frame it is given. On a job that failed or
    closed it returns at once. *)

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
(** [request l r] sends [r] after every frame sent before it and is the agent's
    answer: [Error (`Refused why)] if the agent refused [r], the job going on,
    and [Error (`Failed why)] if the job failed, before or meanwhile, [why] its
    root cause, or if its close began ({!close}), [why] being
    ["the job is closed"]. An answer that does not decode as [r]'s fails the
    job.

    Raises [Invalid_argument] before sending anything if an id of [r] is
    negative. *)

val drop : t -> int -> unit
(** [drop l id] sends the release of the agent's object [id] after every frame
    sent before it, and returns once it is sent.

    Raises [Invalid_argument] if [id] is negative. *)

(** {1:agent The agent's end} *)

val next : t -> (Wire.command, string) result
(** [next l] is the next command [l]'s peer sent, waiting for it. [Error why]
    once the job failed, [why] its root cause. A frame that does not decode as a
    command fails the job.

    A hand-over's areas hold its bytes until the next call of [next] on [l].
    From then on the link reuses their memory: a later hand-over whose bytes fit
    lands in it. The link keeps one such memory, of at most 128 MiB, so a stream
    of large hand-overs lands in memory already mapped. *)

val answer : t -> 'a Wire.request -> ('a, string) result -> unit
(** [answer l r a] sends [a] as the answer to [r]. [Error why] refuses [r] with
    [why].

    Raises [Invalid_argument] if [r] is not, physically, the oldest request
    {!next} gave and none answered. *)

val word : t -> device:int -> int -> unit
(** [word l ~device v] sends that the work of [device] up to [v] is done. *)

val bytes : t -> device:int -> value:int -> Rig_remote_abi.area -> unit
(** [bytes l ~device ~value b] sends [b]'s bytes as those of the next copy into
    the controller's memory of the work of [device] at [value]. It reads [b] in
    place and returns once its bytes are sent, or once the job failed: the
    caller must not write [b] until it returns. *)

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
    From then on, once [ready] reaches transfer [j]'s count, its bytes are
    sent, as many as the socket takes at once by the [ready] function if the
    link sends nothing then, the rest by the sending thread, and [sent] is
    stored once they all are. The receiving thread places arriving transfers
    and stores [arrived]. The end's memory lives until {!release_rail},
    reachable or not.

    Raises [Invalid_argument] if [l] has a rail [id], [send] and [receive] are
    both empty, or a transfer's [length] is not positive or its [src] or [dst]
    is negative. *)

val release_rail : t -> int -> unit
(** [release_rail l id] ends [l]'s rail [id] here: once it returns, neither
    thread reads or writes its end, whose memory lives while it is reachable,
    and its [ready] function must not be called again. A transfer of it that
    arrives later fails the job. It does nothing if [l] has no rail [id]. *)
