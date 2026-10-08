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
    module matches [Rig.Driver] without linking rig.

    {b Work.} A proxy runs copies, and the machine's host also runs
    [Rig.Submission.Words] parts: each is a run of the code loaded on that host.
    A copy moves bytes between memory of the agent's machine, or between that
    memory and this process's. A copy from this process's memory carries its
    bytes across the link in its hand-over; a copy into it, in a frame of the
    agent's, which the receiving thread writes where the copy names before the
    shadow reaches the copy's value. Either crosses the link once. The agent
    runs a hand-over's parts in their order, after its waits, and after the
    proxy's earlier values.

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

type capability = Rig_remote_abi.t
(** The type for what compiled code needs from a proxy. *)

val make : Link.t -> Wire.account -> capability -> t
(** [make l a c] is the proxy on [l] of the agent's device [a], whose record is
    [c]. It sends nothing: [a] is the agent's answer for a device it opened.

    Raises [Invalid_argument] if [c] is [Host _] and [a]'s id is not [0], if [c]
    is [Device { id }] and [id] is not [a]'s, or if [l] has a proxy of [a]'s
    device. *)

exception Fault of string
(** [Fault why] reports that the proxy's job failed, [why] its root cause. *)

(** {1:facts Facts} *)

val key : t Type.Id.t
(** [key] tells proxies apart from other drivers' devices. Proxies of one link
    map each other's words ({!map_peer}); proxies of two links map nothing of
    each other's. *)

val arch : t -> string
(** [arch d] is the [Rig.arch] of the agent's device, as its account states. *)

val budget : t -> int
(** [budget d] is the [Rig.budget] of the agent's device when it opened, as its
    account states. *)

val queues : t -> string list
(** [queues d] is [["COMPUTE:0"; "COPY:0"]]. The agent runs the parts of both in
    one order, that of the hand-over. *)

val completion : t -> [ `Store | `Object of nativeint | `Host ]
(** [completion d] is [`Host]: the link's receiving thread writes the shadow as
    the agent reports. *)

val waits_on : t -> [ `Store | `Object | `Host ] -> bool
(** [waits_on d c] is [true] iff [c] is [`Host]: a wait on a proxy of the same
    link goes to the agent in the hand-over. *)

val max_waits : t -> int
(** [max_waits d] is [max_int]: the hand-over carries every wait. *)

val blocks : t -> [ `Returns | `May_block ]
(** [blocks d] is [`May_block]: the hand-over waits while the link's queue is
    full, and before a copy from this process's memory, for the work it follows
    ({!submit_entry}). *)

val maps_host : t -> bool
(** [maps_host d] is [false]: a copy names this process's memory by its host
    address instead. *)

val capability : t -> capability
(** [capability d] is the record {!make} was given. *)

val capability_key : capability Type.Id.t
(** [capability_key] is {!Rig_remote_abi.key}. *)

(** {1:memory Memory} *)

type region
(** The type for memory of a proxy: an object of the agent, named by an id of
    the job ({!Link.fresh}), or a proxy's {!word}. *)

val alloc : t -> [ `Device | `Pinned | `Mapped ] -> int -> region option
(** [alloc d kind n] asks the agent for [n] bytes of [kind] on its device. It is
    [None] if the agent has not the room or refuses.

    Raises {!Fault} if the job failed. *)

val free : t -> region -> unit
(** [free d r] sends the release of [r]'s object, after every frame queued
    before. The agent releases it once the work handed over before no longer
    needs it. It does nothing for a word. *)

val address : region -> int option
(** [address r] is [None] for memory, which the hand-over names by {!handle}.
    For a word, it is [Some i], [i] the id of the word's device on the agent: a
    wait's [at] carries it, and the hand-over sends it as the device to wait on.
*)

val handle : region -> nativeint
(** [handle r] is the id of [r]'s object on the agent, which the hand-over sends
    as a copy's side. It is [0n] for a word. *)

val host : region -> int option
(** [host r] is [None] for memory, and for a word [Some a], [a] the host address
    of its shadow. *)

val peer : t -> t -> bool
(** [peer d d'] is [true] iff [d] and [d'] are proxies of one link and the
    agent's device of [d] reaches the own memory of [d']'s ([reaches] of
    {!Wire.account}). *)

val map_peer : t -> t -> region -> region option
(** [map_peer d d' r] is a region of [d] over [r], a region of [d'], if [d] and
    [d'] are proxies of one link, and [None] otherwise. A word maps as itself.
    For memory, [map_peer] asks the agent to map [r]'s object on [d]'s device as
    a new object, and is [None] if the agent cannot or refuses.

    Raises {!Fault} if it asks the agent and the job failed. *)

val map_host : t -> int -> int -> region option
(** [map_host d p n] is [None]: proxies map no host memory ({!maps_host}). *)

(** {1:code Code} *)

type image
(** The type for code loaded on the machine's host. *)

val image :
  t ->
  string ->
  ( [ `Loaded of image | `Place of int * (region -> image * string) ],
    string )
  result
(** [image d b] loads [b] on the agent's host if [d] is the machine's host: it
    is [Ok (`Loaded i)], or [Error why] with the agent's reason. On any other
    proxy it is [Error why], [why] saying that the machine's host loads its
    code.

    Raises {!Fault} if [d] is the machine's host and the job failed. *)

val entry : image -> string -> int option
(** [entry i f] is the agent's answer for the function [f] of [i]: [Some e], or
    [None] if [i] has no function [f] or the agent refuses.

    Raises {!Fault} if the job failed. *)

val unload : t -> image -> unit
(** [unload d i] sends the release of [i], after every frame queued before. The
    agent releases it once the work handed over before no longer needs it. *)

(** {1:timeline Timeline} *)

val word : t -> region
(** [word d] is [d]'s word: its shadow, eight bytes of this process's memory
    that hold, as an unsigned 64-bit integer in the host's byte order, the last
    value [v] the agent reported done, or that {!stop} wrote. The receiving
    thread writes it with release order, after it wrote the bytes of the copies
    into this process's memory of every value up to [v]. It never decreases and
    is never freed. *)

val signaled : t -> int
(** [signaled d] is the value in [d]'s shadow, read with acquire order. *)

val sleep : t -> seen:int -> still_ms:int -> unit
(** [sleep d ~seen ~still_ms] returns once [d]'s shadow differs from [seen], at
    once if it does already, or after [still_ms] milliseconds.

    Raises {!Fault} if the job failed, before or meanwhile. *)

(** {1:work Work} *)

val room_entry : nativeint
(** [room_entry] is the address of the proxy's room check, in the shape
    [rig_room_fn] of [rig_edge.h]. It answers:
    - [RIG_NEVER] for a fill, for words on a proxy other than the machine's
      host, and for a submission with a copy from this process's memory after a
      copy into it: the hand-over reads the source bytes before the agent runs
      the earlier part.
    - [RIG_LATER] while the proxy has more than 64 MiB of copies and words
      handed over and not reported done.
    - [RIG_FITS] otherwise, so that a proxy with nothing in flight takes any
      submission it can run. *)

val submit_entry : nativeint
(** [submit_entry] is the address of the proxy's hand-over, in the shape
    [rig_submit_fn] of [rig_edge.h]. It queues the hand-over's frame on the link
    ({!Wire.handover}): the waits, each as the device [at] and its value, and
    the parts, copies with their sides' handles and offsets, words with their
    words. [handles] is ignored.

    Before a submission with a copy from this process's memory, it waits until
    the work the submission follows is done here: the shadow of each wait's
    proxy reaches the wait's value, and [d]'s own shadow the last of [d]'s
    values with a copy into this process's memory. Then it reads the copies'
    bytes into the frame, so the bytes they copy are final. It also waits while
    the link's queue is full.

    It answers [RIG_OK] once the frame is queued. It answers [RIG_FAILED] with
    the job's root cause if the job failed, with ["the job is closed"] if it
    closes, and with ["out of memory for a hand-over"] if memory ran out. *)

val self : t -> nativeint
(** [self d] is the address of [d]'s C state, the [self] argument of the room
    check and the hand-over. It is valid while the process runs: the state holds
    the shadow, which is never freed. *)

(** {1:stopping Stopping} *)

val stop : t -> unit
(** [stop d] writes the last value handed over into [d]'s shadow with release
    order, unless the shadow holds it already, without waiting for the agent. *)
