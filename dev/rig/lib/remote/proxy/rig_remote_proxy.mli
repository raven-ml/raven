(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Proxies: devices of another machine, driven through its agent.

    A {e proxy} is a device of this process for a device that an {e agent}, a
    process on another machine, opened there; the machine's host is one of them.
    The proxy's driver turns each call into a frame on the machine's {!Link.t}:
    a request that the agent answers, the release of one of the agent's objects,
    or a {e hand-over}, the work of one value of the proxy's timeline. The agent
    runs the work and reports each value done. The proxy's timeline word is a
    {e shadow} in this process's memory, which the link's receiving thread
    advances as the reports come.

    rig.remote opens proxies with [Rig.open_], one per device the agent opened,
    and uses this library's {!Wire} and {!Link} for the job's connections. This
    module links only [rig.edge] and matches {!Rig_edge.Driver}.

    {b Work.} A proxy runs copies, and the machine's host also runs
    [Rig.Submission.Words] parts: each is a run of the code loaded on that host,
    which rig.remote's agent runs as rig.program's. A copy moves bytes between
    memory of the agent's machine, or between that memory and this process's. A
    copy from this process's memory carries its bytes across the link in its
    hand-over; a copy into it, in a frame of the agent's, which the receiving
    thread writes where the copy names before the shadow reaches the copy's
    value. Either crosses the link once. The agent runs a hand-over's parts in
    their order, after its waits, and after the proxy's earlier values.

    {b Failure.} Proxies fail with their link's job, as {!Link} states. Once the
    job failed, the calls that ask the agent ({!alloc}, {!map_peer} of memory,
    {!val-image}, {!entry}) and {!sleep} raise {!Fault} with the job's root
    cause, and the hand-over answers [RIG_FAILED] with it. {!free} and {!unload}
    then send nothing, and the other calls answer as before.

    {b Domains.} Every value may be called from any domain at once, except the
    room check and the hand-over of one proxy, which run one at a time as
    [rig_edge.h] states. A call that waits for the agent or the link releases
    the runtime while it waits. *)

module Wire = Wire
(** The protocol between the processes of a job. *)

module Link = Link
(** This process's ends of its job's connections. *)

(** {1:proxies Proxies} *)

type t
(** The type for proxies. *)

val make : Link.t -> Wire.account -> Rig_remote_abi.t -> t
(** [make l a c] is the proxy on [l] of the agent's device [a], whose record is
    [c]. It sends nothing: [a] is the agent's answer for a device it opened.

    Raises [Invalid_argument] if [c] is [Host _] and [a]'s id is not [0], if [c]
    is [Device { id }] and [id] is not [a]'s, or if [l] has a proxy of [a]'s
    device. *)

include Rig_edge.Driver with type t := t

val capability : t -> Rig_remote_abi.t
(** [capability d] is the record {!make} was given, which its facts carry
    under {!Rig_remote_abi.key}. *)

(** {1:this This driver}

    {t
      | Fact | Value |
      |------|-------|
      | [arch] | The [Rig.arch] of the agent's device, as its account states. |
      | [budget] | The agent's device's [Rig.budget] when it opened. |
      | [queues] | ["COMPUTE:0"], ["COPY:0"]; each runs [Copy], and on the machine's host [Words] too. |
      | [completion] | [Host]: the link's receiving thread writes the shadow as the agent reports. |
      | [waits] | [hosts] only, with no bound: a wait on a proxy of the same link goes to the agent in the hand-over. |
      | [may_block] | [true]: the hand-over sends its frame itself. |
      | [maps_host] | [false]: a copy names this process's memory by its host address instead. |
      | [host_addresses] | [false]: the memory is another machine's. |
      | [capability] | {!capability}, under {!Rig_remote_abi.key}. |
      | [word] | The shadow. |
    }

    The agent runs the parts of both queues in one order, that of the
    hand-over. The hand-over returns once its frame is sent, so it waits for
    the frame the link is sending and for the peer to take its bytes. A copy
    from this process's memory holds the submitting thread while its bytes
    cross the link, and before such a copy the hand-over also waits for the
    work the copy follows.

    {!Fault} reports that the proxy's job failed, its message the job's root
    cause.

    {b Memory.} A region is an object of the agent, named by an id of the job
    ({!Link.fresh}), or a proxy's word. {!alloc}[ d m n] asks the agent for [n]
    bytes of [m] on its device; it is [None] if the agent has not the room or
    refuses, and raises {!Fault} if the job failed. {!free} sends the release
    of a region's object, after every frame sent before; the agent releases it
    once the work handed over before no longer needs it. It does nothing for a
    word.

    {!locate} of memory is the object's id as its [handle], which the
    hand-over sends as a copy's side, with no [address] and no [host]. Of a
    word it is [Some i] as its [address], [i] the id of the word's device on
    the agent, which a wait's [at] carries and the hand-over sends as the
    device to wait on; the shadow's host address as its [host]; and [0n] as
    its [handle].

    {!peer}[ d d'] is [true] iff [d] and [d'] are proxies of one link and the
    agent's device of [d] reaches the own memory of [d']'s ([reaches] of
    {!Wire.account}). {!map_peer}[ d d' r] is [None] for proxies of two links.
    A word maps as itself. For memory, it asks the agent to map [r]'s object on
    [d]'s device as a new object, and is [None] if the agent cannot or refuses;
    it raises {!Fault} if it asks and the job failed. {!map_host} is [None].

    {b Code.} {!val-image}[ d b] loads [b] on the agent's host if [d] is the
    machine's host: it is [Ok (Loaded i)], or [Error why] with the agent's
    reason, rig.program's for a binary it does not load. On any other proxy it
    is [Error why], [why] saying the proxy loads no code. It raises {!Fault} if
    [d] is the machine's host and the job failed. {!entry} is the agent's answer
    for the function, [None] if the image has none or the agent refuses; it
    raises {!Fault} if the job failed. {!unload} sends the release of the image,
    after every frame sent before; the agent releases it once the work handed
    over before no longer needs it.

    {b Timeline.} The shadow is eight bytes of this process's memory that hold,
    as an unsigned 64-bit integer in the host's byte order, the last value [v]
    the agent reported done. The receiving thread writes [v] with release
    order, after it wrote the bytes of the copies into this process's memory of
    every value up to [v]. After {!stop} it takes the last value handed over.
    It never decreases and is never freed. {!signaled} reads it with acquire
    order. {!sleep} returns once the shadow differs from [seen], at once if it
    does already, or after [still_ms] milliseconds; it raises {!Fault} if the
    job failed, before or meanwhile.

    {!stop} writes the last value handed over into [d]'s shadow with release
    order, unless the shadow holds it already, once no bytes of a copy into this
    process's memory may still land: at once if none is pending or the link's
    receiving thread ended, and otherwise as the last pending copy's bytes land
    or the thread ends. It waits for nothing, and ignores [fault].

    {b Work.} The edge's state holds the shadow, which is never freed. The room
    check answers:
    - [RIG_NEVER] for a part other than words and copies, for words on a proxy
      other than the machine's host, and for a submission with a copy from this
      process's memory after a copy into it: the hand-over reads the source
      bytes before the agent runs the earlier part.
    - [RIG_LATER] while the proxy has more than 64 MiB of copies and words
      handed over and not reported done.
    - [RIG_FITS] otherwise, so that a proxy with nothing in flight takes any
      submission it can run.

    The hand-over sends the hand-over's frame on the link
    ({!Wire.handover}): the waits, each as the device [at] and its value, and
    the parts, copies with their sides' handles and offsets, words with their
    words. [handles] is ignored.

    Before a submission with a copy from this process's memory, it waits until
    the work the submission follows is done here: the shadow of each wait's
    proxy reaches the wait's value, and [d]'s own shadow the last of [d]'s
    values with a copy into this process's memory. Then it sends the copies'
    bytes from that memory, in place, so the bytes they copy are final. It sends
    after the frame the link is sending.

    It answers [RIG_COMMITTED] once the frame is sent. It answers [RIG_FAILED]
    with the job's root cause if the job failed, with ["the job is closed"] if
    it closes, and with ["out of memory for a hand-over"] if memory ran out.

    The commit does nothing: each value's hand-over sends its message and
    answers [RIG_COMMITTED]. *)
