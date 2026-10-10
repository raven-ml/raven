(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from another machine's devices.

    A device of another machine is a {e proxy}: a device of this process whose
    driver sends its calls to an {e agent}, a process on that machine that
    opened the device. Code for it is compiled here, then loaded and run on that
    machine, so what the code needs from the device is plain data: the device's
    id on its machine, and from the machine's host, a function that makes
    {e rails} between machines.

    Every proxy declares one record under {!key}: the machine's host a {!Host},
    every other device a {!Device}. The library that opens proxies fills them as
    each opens; compiled code finds them with [Rig.capability]. Neither links
    the other, and this library links neither.

    {1:rails Rails}

    A {e rail} carries {e transfers} between two machines: in each direction a
    fixed array of them, which it carries again and again, in {e runs}. Each
    machine has one {e end} of the rail ({!end_}): host memory of that machine,
    each area starting on a page, which its devices may borrow. An end holds two
    landing areas, [outbound] for what its machine sends and [inbound] for what
    it receives, each in two copies, and three counts: [ready] and [sent] for
    what it sends, [arrived] for what it receives.

    In a direction of [n] transfers, the count [c >= 1] names run
    [r = (c - 1) / n] and transfer [j = (c - 1) mod n]; the run uses copy
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

    [ready] advances only through that function: the rail reads the count when
    something wakes it, so a store to it alone may go unseen for up to a second.
    Work of a device that cannot call the function, such as a GPU's, is followed
    by host code that waits for its point and calls it. Counts only grow, and
    nothing else writes [sent] and [arrived]. Work that waits for [arrived >= c]
    then reads the transfer's bytes; work that writes copy [k] of [outbound]
    again first waits for [sent] to reach the count of the last transfer that
    read it. A rail moves no byte outside the landing areas.

    If the job fails, every count of every end on a machine that still answers
    is raised to [Int64.max_int], so that no wait for one blocks; what the
    landing areas then hold is unspecified. *)

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

type end_ = {
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
      (** [ready c] stores [ready := c] with release order and wakes the rail.
          It may be called from any domain, until the rail is released. *)
  ready_fn : nativeint;
      (** The address of the C function
          {v void ready(void *arg, uint64_t c); v}
          that does what [ready] does, called with [ready_arg], for compiled
          host code, until the rail is released. It calls nothing of the OCaml
          runtime and blocks only on a lock of the rail's connection. *)
  ready_arg : nativeint;  (** The [arg] of [ready_fn] for this end. *)
}
(** The type for one machine's end of a rail. *)

type rail = {
  id : int;
      (** The rail's id among the job's rails, by which a machine's compiled
          code finds its end there. *)
  local : end_ option;
      (** The end on this process's machine, if the rail has one. *)
  release : unit -> unit;
      (** Ends the rail on both machines, and frees both ends. Call it once no
          work that uses either end can run on any machine, such as from the
          release of the hold that every submission using the rail names
          ([Rig.Hold.make]). Calling it again does nothing. It may be called
          from any domain. *)
}
(** The type for rails. *)

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
          the job failed or [peer] is of another job, [why] naming the cause. It
          may be called from any domain.

          Raises [Invalid_argument] if [peer] is this host or {!check_transfers}
          answers [Error]. *)
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
(** [key] is the key every proxy declares its record under. *)
