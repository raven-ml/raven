(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Links: this process's ends of its job's connections, in C.

    A {e link} is this process's end of its connection to one other process
    of its job, once the handshake has admitted it ({!Wire}). Whoever writes a
    message sends it from its own thread, a proxy's C sending a [handover] or
    a function of this module. The writer waits while an earlier message is
    sent, sends its own whole, and returns once it is sent, so the socket is
    the only back-pressure. A link owns the socket and two threads of C,
    which never hold the OCaml runtime:
    - the {e sending thread} sends what the [ready] functions of the link's
      rails ({!rail}) leave, and a beat after a second in which nothing was
      sent, between writers' messages. It waits on nothing but the writers
      and its socket.
    - the {e receiving thread} reads each message. It places the transfers of
      the link's rails, and on the controller, writes the bytes of copies into
      this process's memory, advances the proxies' timeline words and
      delivers the answers to {!request}s. It queues every other message for
      {!next}, a [handover] with its bytes.

    Messages carry no integrity check after the handshake, as {!Wire} states.

    {1:failure Jobs and failure}

    The links of a process belong to its {e job} ({!type-job}), which fails or
    closes as a whole. The job {e fails} once, with the first of these as its
    {e root cause}:
    - a link's stream breaks or ends without a close:
      ["NAME: closed its connection"], or the system's error after ["NAME: "];
    - no byte comes on a link for 10 seconds: ["NAME: silent for 10 s"];
    - a link's peer takes no byte of a send for 10 seconds:
      ["NAME: read nothing for 10 s"];
    - a message is malformed, as is one with a u64 above [2]{^ 62}[ - 1]
      ({!Wire}): ["NAME: a malformed frame"];
    - a message is larger than this process can hold:
      ["NAME: a frame larger than this process can hold"];
    - a peer aborts: its reason, unchanged;
    - {!fail}.

    [NAME] is the link's {!name}. A reason is any bytes, at most 4096 of them.
    A longer one is cut there, here and in an agent's refusal ({!answer}).

    Once the job fails, every link of the job that has sent no close sends
    its peer an abort with the root cause, after the message it is sending
    if any, and ends its stream; a link whose send failed sends none, its
    stream cut. It reads and discards what the peer still sends until the
    peer ends its stream, aborts or closes, or 10 seconds after the last
    byte it received before the failure, so the abort reaches a peer still
    reading earlier messages. It sends no message it is given.

    The process learns of the failure only once every abort left, so a
    process that ends on its failure ends after its peers were sent the root
    cause. Then {!request} and {!next} return [Error], the proxies' sleeps
    return, and every count of every rail of the job reads [Int64.max_int],
    except the rails {!release_rail} is releasing. Nothing raises from a C
    thread, and nothing calls OCaml. The process learns of the failure from
    {!failed}, {!wait} or a function's [Error].

    A child of [fork] never uses its parent's links, whose streams it would
    interleave with the parent's. Every function compares the process id with
    the one that started the job, and in a child the job is failed with
    ["a child of fork does not use its parent's connections"], in the child's
    copy alone. There every count of every rail reads [Int64.max_int], as
    after any failure. The child sends nothing on its parent's links and
    closes none of them, and a link it makes is failed, its socket closed.

    {1:domains Domains}

    Every function may be called from several domains in parallel. A function
    that waits, for the socket, an answer, a message or the job, releases the
    runtime while it waits.

    A job and its links are never freed, because proxies' C state holds their
    addresses. A link that has ended keeps its name and reason. *)

(** {1:jobs Jobs} *)

type job
(** The type for the jobs of this process. A job is {e open} from {!val-job}
    until it has failed or closed. *)

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

val failed : job -> string option
(** [failed j] is [Some why] iff [wait j ~ms:0] is [Failed why]. *)

val fail : job -> string -> unit
(** [fail j why] fails [j] with the root cause [why], unless it has failed or
    closed, and returns once [j]'s failure is known ({!failed}), whatever
    began it. Each link's abort follows the message the link is sending and
    waits up to 10 seconds for a peer that reads nothing. [fail] releases the
    runtime while it waits. *)

val close : job -> unit
(** [close j] ends [j] in order. Each of its links sends a close after the
    messages sent before, and [close] returns once every peer's close has come
    and the links' threads have ended, or once [j] has failed. A link's peer
    sends its close when it ends the job itself, so [close] waits for every
    peer to end it. Once a link has taken its close, it sends no message it is
    given. On a job that has failed or closed, [close] returns immediately. *)

(** {1:links Links} *)

type t
(** The type for links. *)

val make : job -> Unix.file_descr -> name:string -> peer:Wire.process -> t
(** [make j fd ~name ~peer] is a link of [j] over the connected socket [fd],
    whose handshake has admitted it, to the process [peer] named [name]: its
    machine's name, which starts its reasons. [make] takes [fd], and its
    threads start immediately. On a failed job the link is failed and [fd]
    closed.

    Raises [Invalid_argument] if [j] is closed, and [Out_of_memory] if the
    link cannot be allocated. [fd] is closed in either case. *)

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
(** [request l r] sends [r] after every message sent before it and is the
    agent's answer. It is [Error (`Refused why)] if the agent refused [r], and
    the job goes on. It is [Error (`Failed why)] if the job has failed, before
    or meanwhile, [why] its root cause, or if its close has begun ({!close}),
    [why] then ["the job is closed"]. An answer that does not decode as [r]'s
    fails the job.

    Raises [Invalid_argument] before sending anything if an id of [r] is
    negative. *)

val drop : t -> int -> unit
(** [drop l id] sends the release of the agent's object [id] after every
    message sent before it, and returns once it is sent, or immediately if the
    job has failed or its close has begun.

    Raises [Invalid_argument] if [id] is negative. *)

(** {1:agent The agent's end} *)

val next : t -> (Wire.command, string) result
(** [next l] is the next command [l]'s peer sent, waiting for it. [Error why]
    once the job has failed, [why] its root cause. A message that does not
    decode as a command fails the job.

    A [Handover]'s areas hold its bytes until the next call of [next] on [l],
    from any thread, so a thread must finish with a [Handover]'s areas before
    any thread calls [next]. The link may then keep their memory, up to
    128 MiB of it, for later commands to land in until [next] gives the close,
    so a stream of large [Handover] commands lands in memory already
    mapped. *)

val answer : t -> 'a Wire.request -> ('a, string) result -> unit
(** [answer l r a] sends [a] as the answer to [r]. [Error why] refuses [r] with
    [why]. From several threads, the check and the send are one step, so answers
    reach the peer in the order of their requests; a call for a request that is
    not the oldest unanswered raises, as below.

    Raises [Invalid_argument] if [r] is not, physically, the oldest request
    {!next} gave and none answered. *)

val word : t -> device:int -> int -> unit
(** [word l ~device v] sends that the work of [device] up to [v] is done. *)

val bytes : t -> device:int -> value:int -> Rig_remote_abi.area -> unit
(** [bytes l ~device ~value b] sends [b]'s bytes as those of the next copy into
    the controller's memory of the work of [device] at [value]. It reads [b] in
    place and returns once its bytes are sent, or once the job has failed or
    its close has begun. The caller must not write [b] until it returns. *)

(** {1:rails Rails} *)

val rail :
  t ->
  id:int ->
  send:Rig_remote_abi.transfer array ->
  receive:Rig_remote_abi.transfer array ->
  Rig_remote_abi.end_
(** [rail l ~id ~send ~receive] is this machine's end of the rail [id] to [l]'s
    peer, which carries [send] to it and [receive] from it, with the rail's
    areas zeroed ({!Rig_remote_abi.section-rails}) and its [ready] function.
    Once [ready] reaches transfer [j]'s count, its bytes are sent. If the link
    is sending nothing then, the [ready] function sends as many as the socket
    takes, and the sending thread sends the rest. [sent] is stored once they
    all are. The receiving thread places arriving transfers and stores
    [arrived]. The end's memory lives until {!release_rail}, reachable or
    not.

    Raises [Invalid_argument] if [l] has a rail [id] or
    {!Rig_remote_abi.check_transfers} is [Error], and [Out_of_memory] if the
    host cannot allocate the areas. *)

val release_rail : t -> int -> unit
(** [release_rail l id] ends [l]'s rail [id] here: once it returns, neither
    thread reads or writes its end, whose memory lives while it is reachable,
    and its [ready] function must not be called again. Its [ready_fn] may still
    store to its counts, as {!type:Rig_remote_abi.end_} states. Concurrent
    releases of the same rail all wait for its active transfers to finish. A
    transfer of it that arrives later is discarded, as is one of a rail [l]
    never had. It does nothing if [l] has no rail [id]. It takes no lock an
    OCaml thread holds, so a finaliser may call it. *)
