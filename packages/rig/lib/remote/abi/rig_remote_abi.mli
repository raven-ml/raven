(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from another machine's devices.

    A {e proxy} is a device of this process that stands for a device an
    {e agent}, a process on another machine, opened there ([Rig_remote]).
    Code for a proxy is compiled here, then loaded and run on that machine,
    so what the code needs from the device is plain data: the device's id on
    its machine, and from the machine's host, a function that makes rails.

    Every proxy declares one record ({!declare}): the machine's host a
    {!Host}, every other device a {!Device}. rig.remote fills the records as
    each proxy opens, and compiled code finds them under {!key}. This
    library depends on neither.

    {1:rails Rails}

    A {e rail} carries a fixed set of {e transfers} between host memory on
    two machines, again and again, without relaying them through the
    controller. In each direction it has a fixed array of them, which it
    carries once per {e run}. Each machine has one {e end} of the rail
    ({!type:end_}), in host memory of that machine that its devices may
    borrow, each area starting on a page. An end holds two landing areas,
    [outbound] for what its machine sends and [inbound] for what it receives,
    each in two copies, and three counts: [ready] and [sent] for what it
    sends, [arrived] for what it receives.

    In a direction of [n] transfers, the count [1 <= c < Int64.max_int] names
    run [r = (c - 1) / n] and transfer [j = (c - 1) mod n]; the run uses copy
    [k = r mod 2] of each landing area. For each [c] in order:
    + the sending machine's work places transfer [j]'s bytes at its [src] in
      copy [k] of the sender's [outbound], then advances [ready] to [c] through
      its end's [ready] function, which stores it with release order and wakes
      the rail;
    + the rail sees [ready >= c], moves the bytes to [dst] in copy [k] of the
      receiver's [inbound], then stores [arrived := c] at the receiver with
      release order, so that work that reads [arrived] with acquire order and
      finds [c] sees the bytes;
    + the rail stores [sent := c] at the sender with release order once the
      source bytes may be written again.

    [ready] must advance only through that function. The rail reads the count
    only when the function wakes it, so a store to the count alone may never
    be seen. Work of a device that cannot call the function, such as a GPU's,
    is followed by host code that waits for its point and calls it. Counts
    only grow, and only the rail writes [sent] and [arrived]. Work that waits
    for [arrived >= c] then reads the transfer's bytes, and work that writes
    copy [k] of [outbound] again must first wait for [sent] to reach the
    count of the last transfer that read it. A rail moves bytes only between
    landing areas.

    A transfer may land in an [inbound] copy that work still reads. The
    caller must order it, for example by waiting for the receiving
    machine's reads of run [r] before the sender starts run [r + 2], which
    lands in the same copy.

    [Int64.max_int] is reserved for failure. If the job fails, every count of
    every end still in use on a machine still running is raised to it and
    stays there, so that no wait for one blocks. What the landing areas then
    hold is unspecified. *)

(** {1:records Records} *)

type area =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for the host memory of an end. *)

type transfer = {
  src : int;  (** Its first byte in a copy of the sender's [outbound]. *)
  dst : int;  (** Its first byte in a copy of the receiver's [inbound]. *)
  length : int;  (** Its bytes, at least [1]. *)
}
(** The type for transfers: the same bytes at the same places in every run. *)

val check_transfers :
  send:transfer array -> receive:transfer array -> (unit, string) result
(** [check_transfers ~send ~receive] is [Ok ()] iff a rail can carry [send] and
    [receive]: they are not both empty, and each transfer's [length] is positive
    and its [src] and [dst] are non-negative, with [src + length] and
    [dst + length] at most [2]{^ 60}. [Error why] otherwise, [why] naming the
    transfer. *)

type end_ = private {
  outbound : area;
      (** Two copies, one after the other, of the bytes this machine sends: each
          the largest [src + length] of its transfers, rounded up to a multiple
          of 256 bytes. Empty if it sends nothing. *)
  inbound : area;
      (** Two copies of the bytes this machine receives: each the largest
          [dst + length] of its transfers, rounded up likewise. Empty if it
          receives nothing. *)
  counts : area;
      (** [ready], [sent] and [arrived], each a 64-bit unsigned integer in the
          host's byte order, at bytes [0], [128] and [256]: each alone in its
          cache line. They start at [0]. *)
  ready : int -> unit;
      (** [ready c] advances [ready] to at least [c] with release order and
          wakes the rail. A failed job's count stays at [Int64.max_int]. It may
          be called from any domain, until the rail ends. *)
  ready_fn : nativeint;
      (** The address of the C function
          {v void ready(void *counts, uint64_t c); v}
          that does what [ready] does when called with the host address of this
          end's [counts], for compiled host code, such as a launch on the
          host whose parameters name the counts. Once the rail has ended, it
          still advances [ready] and wakes nothing. It calls nothing of the
          OCaml runtime and blocks only on locks protecting live rails and the
          connection. *)
}
(** The type for one machine's end of a rail. *)

type rail = private {
  id : int;
      (** The rail's id among the job's rails, by which a machine's compiled
          code finds its end there. *)
  local : end_ option;
      (** The end on this process's machine, if the rail has one. *)
}
(** The type for rails. A rail's value or local end keeps both ends alive. Once
    neither is reachable, new programs cannot name the rail; each end lives
    while a program already loaded there keeps it ([Rig_program.load]).
    Closing or failing the job stops transfers; an end's memory lives while
    it is reachable.

    {b Warning.} A submission whose work uses an end must keep the rail or
    the end ([Rig.Submission.make ~keep:rail]), or the end may be freed
    while its work runs. *)

(** {1:providers Providers} *)

val end_ :
  outbound:area ->
  inbound:area ->
  counts:area ->
  ready:(int -> unit) ->
  ready_fn:nativeint ->
  end_
(** [end_ ~outbound ~inbound ~counts ~ready ~ready_fn] is a provider's new rail
    end. The provider owns the areas and transport and arranges their lifetime
    as {!type:end_} and {!type:rail} require. This call registers no cleanup.
    A provider must not make an end from the fields of an existing one. *)

val rail : id:int -> local:end_ option -> rail
(** [rail ~id ~local] is a provider's new rail. The provider owns its
    registration on both machines and arranges its lifetime as {!type:rail}
    requires. This call registers no cleanup. A provider must not make a rail
    from the fields of an existing one. *)

(** {1:proxies Proxies} *)

type host = {
  machine : string;  (** The machine's name, as its devices' names end. *)
  rail :
    host option ->
    send:transfer array ->
    receive:transfer array ->
    (rail, string) result;
      (** [rail peer ~send ~receive] makes a rail between this host's machine
          and [peer]'s, or this process's machine for [None], that carries
          [send] from this host's machine and [receive] to it. [Error why] if
          the job has failed or [peer] is of another job, [why] naming the
          cause. It may be called from any domain.

          Raises [Invalid_argument] if [peer] is this host or {!check_transfers}
          is [Error]. *)
  threads : int -> int option array;
      (** [threads p] is what the machine's agent stated when it loaded the
          program whose function ["run"] the machine's host names [p]
          ([Rig.Image.entry]): the most threads a group of each launch of the
          program may have, in the order of its steps with a loop's body in
          place ([Rig.Image.threads]). It is [[||]] for no such program. Any
          domain may call it. *)
}
(** The type for the record of another machine's host. *)

type device = {
  id : int;  (** The device's id on its machine, which code names it by. *)
}
(** The type for the record of another machine's device. *)

(** The type for the records of proxies. *)
type t =
  | Host of host  (** The machine's host. *)
  | Device of device  (** Any other device of the machine. *)

val key : t Type.Id.t
(** [key] is the key of a proxy's record. Compiled code finds the record
    of a device [d] with [Rig.capability d key], which is [None] if [d] is
    not a proxy. *)

val declare : t -> Rig_edge.capability
(** [declare c] is [c] under {!key}, as the facts of a proxy carry it
    ({!Rig_edge.type-facts}). *)
