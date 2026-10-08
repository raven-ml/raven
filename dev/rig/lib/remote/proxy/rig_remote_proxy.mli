(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Proxies: devices of another machine, driven through its agent.

    A {e proxy} is a device of this process for a device that an agent opened on
    another machine, the machine's host among them. Its driver sends each call
    to the agent as a frame on the machine's {!Link.t}, and its word is a
    {e shadow} in this process's memory, which the link's receiving thread
    advances as the agent reports the work done. It matches [Rig.Driver] without
    linking rig; rig.remote opens proxies with [Rig.open_] and links this
    library for its {!Wire} and {!Link} too.

    {1:facts Facts}

    All proxies are one driver with one {!key}. Proxies of one link map each
    other's words, and map each other's memory as the agent's devices reach each
    other's ([reaches] of {!Wire.account}); proxies of two links never do.
    {!arch} and {!budget} are the agent's device's when it opened. {!queues} are
    ["COMPUTE:0"] and ["COPY:0"]. {!completion} is [`Host], since the agent's
    reports write the word; {!waits_on} is [true] for [`Host] alone, so that
    waits between proxies of one link reach the agent, and {!max_waits} is
    [max_int]. {!blocks} is [`May_block] and {!maps_host} is [false].
    {!capability} is the record {!make} was given.

    {1:work Work}

    A proxy runs copies, and the machine's host also runs words: a
    [Rig.Submission.Words] part is a run of the code loaded on the host. Its
    room check answers [RIG_NEVER] for any other part, and for a submission
    whose part copies from this process's memory after a part of it copies into
    that memory; [RIG_LATER] while its device has more than 64 MiB of copies and
    words handed over and not reached; and [RIG_FITS] otherwise, so that a
    device with nothing in flight always takes one submission.

    Its hand-over encodes the submission's waits on proxies of its link and its
    parts into the link's queue, and copies the words of each part. If a part
    copies from this process's memory, it first waits until the work the
    submission follows is done there: the shadows of its waits reach their
    values, and its own device's shadow the last of its values that copies into
    this process's memory. So the bytes it copies are final. It also waits while
    the link's queue is full. The agent runs the parts in their order, after the
    waits.

    A copy between this process's memory and the machine's carries its bytes
    across the link once. The sending thread reads a copy's bytes from this
    process's memory when it sends them. The receiving thread writes the bytes
    of a copy into this process's memory only where the copy's part named,
    before the shadow reaches its value.

    {1:failure Failure}

    Once the link's job failed, every counted call and {!sleep} raise {!Fault}
    with the root cause, the hand-over answers [RIG_FAILED] with it, and {!stop}
    writes the last value handed over into the shadow. *)

module Wire = Wire
module Link = Link

type t
(** The type for proxies. *)

type region
(** The type for memory of a proxy: an object of the agent, named by an id of
    the job. *)

type image
(** The type for code loaded on the machine's host. *)

type capability = Rig_remote_abi.t
(** The type for proxies' records. *)

exception Fault of string
(** [Fault why] reports that the proxy's job failed, [why] its root cause. *)

val make : Link.t -> Wire.account -> capability -> t
(** [make l a c] is the proxy of the agent's device of account [a] on [l], whose
    record is [c]. It sends nothing: [a] is the agent's answer for a device it
    opened.

    Raises [Invalid_argument] if [c] is [Host _] and [a]'s id is not [0], or
    [Device { id }] and [id] is not [a]'s. *)

(** {1:driver Driver}

    The values of [Rig.Driver]. A {!region} of memory has no address and no host
    address: {!handle} is its id. A {!word} is the shadow: its host address is
    this process's. {!map_peer} of a word, for a proxy of the same link, is a
    region whose {!address} is the producer's id on the agent: rig carries it in
    a wait's [at], and the hand-over sends it as the wait's device. *)

val key : t Type.Id.t
val arch : t -> string
val budget : t -> int
val queues : t -> string list
val completion : t -> [ `Store | `Object of nativeint | `Host ]
val waits_on : t -> [ `Store | `Object | `Host ] -> bool
val max_waits : t -> int
val blocks : t -> [ `Returns | `May_block ]
val maps_host : t -> bool
val capability : t -> capability
val capability_key : capability Type.Id.t
val alloc : t -> [ `Device | `Pinned | `Mapped ] -> int -> region option
val free : t -> region -> unit
val address : region -> int option
val handle : region -> nativeint
val host : region -> int option
val peer : t -> t -> bool
val map_peer : t -> t -> region -> region option
val map_host : t -> int -> int -> region option

val image :
  t ->
  string ->
  ( [ `Loaded of image | `Place of int * (region -> image * string) ],
    string )
  result
(** [image d b] loads [b] on the agent's host if [d] is the machine's host:
    [`Loaded i], or [Error why] with the agent's reason. On any other proxy it
    is [Error] naming the machine's host, which loads the machine's code. *)

val entry : image -> string -> int option
val unload : t -> image -> unit
val word : t -> region
val signaled : t -> int
val sleep : t -> seen:int -> still_ms:int -> unit
val room_entry : nativeint
val submit_entry : nativeint
val self : t -> nativeint
val stop : t -> unit
