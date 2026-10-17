(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Proxies: devices of another machine, driven through its agent.

    A {e proxy} is a device of this process that stands for a device an
    {e agent}, a process on another machine, opened there ([Rig_remote]).
    The machine's host is one of them. The proxy's driver turns each call
    into a message on the machine's {!Link.t}: a request, which has an
    answer, the release of one of the agent's objects, or a [handover], the
    work of one value of the proxy's timeline. The agent runs the work and
    reports each value done, and the link's receiving thread advances the
    proxy's timeline word, which lies in this process's memory, as the
    reports come.

    rig.remote opens proxies with [Rig.open_], one per device the agent
    opened, and uses this library's {!Wire} and {!Link} for the job's
    connections. The library links [rig.edge], [rig.remote.abi] and [unix],
    and this module matches {!Rig_edge.Driver}.

    Every value may be called from several domains in parallel, except the
    driver's [room] and [submit] for one proxy, which rig calls one at a time
    ({!Rig_edge}). A call that waits for the agent or the link releases the
    runtime while it waits. *)

module Wire = Wire
(** The protocol between the processes of a job. *)

module Link = Link
(** This process's ends of its job's connections. *)

(** {1:proxies Proxies} *)

type t
(** The type for proxies. *)

val device : Link.t -> Wire.account -> Rig_remote_abi.t -> t Rig_edge.opener
(** [device l a c] describes the proxy on [l] of the agent's device [a], whose
    record is [c]. Its name is [a.name]. Constructing it registers nothing and
    sends nothing: [a] is the agent's answer for a device it opened.

    Its [make ~word] registers the proxy on [l], with the 8 bytes at [word]
    as its timeline word, and returns it. It sends nothing. The proxy's
    values start after the value those bytes hold.

    [make] raises [Invalid_argument] if [c] is [Host _] and [a]'s id is not
    [0], if [c] is [Device { id }] and [id] is not [a]'s, or if [l] has a proxy
    of [a]'s device. Closing a proxy does not permit another proxy of that
    account on the same link. *)

include Rig_edge.Driver with type t := t

val capability : t -> Rig_remote_abi.t
(** [capability d] is the record {!device} was given, which its facts declare
    ({!Rig_remote_abi.declare}). *)

(** {1:facts Facts}

    {t
      | Fact             | Value                                             |
      |------------------|---------------------------------------------------|
      | [arch]           | The agent's device's [Rig.arch], from its account |
      | [budget]         | The agent's device's [Rig.budget] when it opened  |
      | [queues]         | ["COMPUTE:0"] and ["COPY:0"]                      |
      | [copy_engine]    | [Some 1], ["COPY:0"]                              |
      | [completion]     | [Host]                                            |
      | [waits]          | [hosts]; [most] [max_int]                         |
      | [may_block]      | [true]                                            |
      | [hang_ms]        | [None]                                            |
      | [maps_host]      | [false]                                           |
      | [host_addresses] | [false]                                           |
      | [capability]     | {!capability} ({!Rig_remote_abi.declare})         |
      | [word]           | The word [make] was given                         |
    }

    Each queue runs [Copy], and on the machine's host [Words] too. The
    link's receiving thread writes the timeline word as the agent reports
    ({!section-progress}). A wait on a proxy of the same link goes to the
    agent in the [handover]. [submit] sends its message itself, so it may
    block. A copy names this process's memory by its host address, and the
    proxy's own memory is another machine's. *)

(** {1:work Work}

    A proxy runs copies, and the machine's host also runs
    [Rig.Submission.words] parts, each a run of the rig.program code loaded on
    that host. A copy from this process's memory carries its bytes in its
    [handover], and a copy into it in a [bytes] message of the agent's, which
    the receiving thread writes where the copy names before the timeline word
    reaches the copy's value. Either crosses the link once. The agent runs the
    parts of both queues in one order, that of the [handover], after the
    proxy's earlier values.

    The driver's [room] returns:
    - [RIG_NEVER] for a part other than words and copies, for words on a
      proxy other than the machine's host, and for a submission with a copy
      from this process's memory after a copy into it, because [submit] reads
      the source bytes before the agent runs the earlier part;
    - [RIG_LATER] while the proxy has more than 64 MiB of copies and words
      submitted and not reported done;
    - [RIG_FITS] otherwise, so that a proxy with nothing in flight takes any
      submission it can run.

    The driver's [submit] sends a {!Wire.handover} message on the link: the
    waits, each as the device [at] and its value, and the parts, copies with
    their sides' handles and offsets, words with their words. It sends after
    the message the link is sending, and returns once its message is sent,
    so it also waits for the peer to take its bytes. Before a submission with
    a copy from this process's memory, it waits until the work the submission
    follows is done here: the timeline word of each wait's proxy reaches the
    wait's value, and [d]'s own word the last of [d]'s values with a copy
    into this process's memory. It then sends the copies' bytes from that
    memory in place, so the bytes they copy are final, and holds the
    submitting thread while they cross the link.

    [submit] returns [RIG_COMMITTED] once its message is sent. It returns
    [RIG_FAILED] with the job's root cause if the job has failed, with
    ["the job is closed"] if the job closes, and with
    ["out of memory for a hand-over"] if memory runs out. The driver's
    [commit] does nothing, since each [submit] sends its message. *)

(** {1:regions Memory}

    A region is an object of the agent, named by an id of the job
    ({!Link.fresh}), or a proxy's timeline word. {!alloc}[ d m n] asks the
    agent for [n] bytes of [m] on its device. It is [None] if the agent lacks
    the room or refuses, and raises {!Fault} if the job has failed. {!free}
    sends the release of a region's object, after every message sent before,
    and the agent releases it once the work submitted before no longer needs
    it. It does nothing for a word.

    {!locate} of memory is the object's id as its [handle], which [submit]
    sends as a copy's side, with no [address] and no [host]. Of a word it is
    [Some i] as its [address], [i] the id of the word's device on the agent,
    which a wait's [at] carries and [submit] sends as the device to wait on;
    the word's host address as its [host]; and [0n] as its [handle].

    {!peer}[ d d'] is [true] iff [d] and [d'] are proxies of one link and the
    agent's device of [d] reaches the own memory of [d']'s, as the account
    with the larger id describes ({!Wire.account}). {!map_peer}[ d d' r] is
    [None] for proxies of two links. A word maps as itself. For memory, it
    asks the agent to map [r]'s object on [d]'s device as a new object, and is
    [None] if the agent cannot or refuses; it raises {!Fault} if it asks and
    the job has failed. {!map_host} is [None]. *)

(** {1:images Code}

    {!val-image}[ d b] loads [b] on the agent's host if [d] is the machine's
    host. It is [Ok (Loaded i)], or [Error why] with the agent's reason,
    rig.program's for a binary it does not load. On any other proxy it is
    [Error why], [why] saying the proxy loads no code. It raises {!Fault} if
    [d] is the machine's host and the job has failed. {!entry} is the agent's
    answer for the function, [None] if the image has none or the agent
    refuses; it raises {!Fault} if the job has failed. {!unload} sends the
    release of the image, after every message sent before, and the agent
    releases it once the work submitted before no longer needs it. *)

(** {1:progress Timeline, failure and stopping}

    The proxy's timeline word is the eight bytes of this process's memory
    that [make] was given. They hold, as an unsigned 64-bit integer in the
    host's byte order, the last value [v] the agent has reported done. The
    receiving thread writes [v] with release order, after it has written the
    bytes of the copies into this process's memory of every value up to
    [v]. After {!stop} the word holds the last value submitted. It never
    decreases, and the edge's state holds its address. {!signaled} reads it with acquire order.
    {!sleep} returns once the word differs from [seen], immediately if it
    does already, or after [still_ms] milliseconds; it raises {!Fault} if the
    job has failed, before or meanwhile.

    Proxies fail with their link's job, as {!Link} states. {!Fault} reports
    that the job has failed, its message the job's root cause, or that the
    job's close has begun ({!Link.close}), its message
    ["the job is closed"]. Once the job has failed, the calls that ask the
    agent ({!alloc}, {!map_peer} of memory, {!val-image}, {!entry}) and
    {!sleep} raise {!Fault}, and [submit] returns [RIG_FAILED] with the root
    cause. {!free} and {!unload} then send nothing, and the other calls
    behave as before. Once the close has begun, the calls that ask the agent
    raise {!Fault}.

    {!stop} writes the last value submitted into [d]'s timeline word with
    release order, unless the word holds it already, once no bytes of a copy
    into this process's memory may still land: immediately if none is
    pending or the link's receiving thread has ended, and otherwise as the
    last pending copy's bytes land or the thread ends. It waits for nothing,
    and ignores [fault]. Once the word holds that value, nothing writes it:
    the receiving thread writes it under the link's lock, as the stop does,
    and drops a report of a stopped proxy that the word has passed. *)

(**/**)

val threads : int -> int option array
(** [threads id] is what the agent stated when it loaded the program [id]
    ({!Wire.type-request}), until {!unload} unloads it, or [[||]] for none.
    It fills the [threads] of {!Rig_remote_abi.type-host}. *)
