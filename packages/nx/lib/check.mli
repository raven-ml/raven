(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pending checks: the failures of checks off the host that waits raise,
    which [Nx]'s waiting section states.

    A {e check} is a [Check] operation: boolean {e flags} [ok], {e data} of
    their shape, and [fail], which makes the exception of the first false
    flag in C order from its index and the data's elements there. On another
    set than the host, {!Eval} reads a check at its issue where reading its
    flags and data waits for nothing ({!written}): a passing one leaves no
    trace and a failing one is kept as the builder of its exception, its
    position and elements read on the host ({!add_read}). Otherwise it
    computes the check there without reading it, copies whether every flag
    holds into memory the host addresses, and the first false position and
    each datum's element there into memory of its own, and hands this
    module the {e pending} check ({!add}). A pending check of one flag on a
    device whose memory the host addresses holds that flag and those
    elements in place instead, under a keep ({!Rig.Buffer.val-keep}) that
    ends once the check is read, since a copy there would make the host wait
    for the work that writes them.

    The pending checks and failures are the process's, by set: every domain
    and systhread reaches them, and a set with none has no entry. Each set
    has a {e claim}, which one reader at a time holds while it reads the
    set's checks and takes them off; adding needs no claim. A check is read
    once: whether its flags hold, and for a failing one its position and
    elements, after which it holds nothing of a device's and its keep ends.
    A read that raises, such as {!Rig.Lost}, takes its check off before the
    exception passes on.

    A {!wait} on the set takes its claim, waiting for another reader to let
    it go, reads its checks oldest first, waiting for each, takes off the
    passing ones up to the first failure, and takes off that failure: one
    failure per wait, the oldest first, the later ones left for later waits.
    It lets the claim go, then [fail] builds the exception, on the waiting
    domain, and may read the values it receives, which waits on a set. A wait
    inside that build raises no failure, which would replace the one being
    built.

    Once a pending check's reduction, and its copy, are submitted, it records
    the value its device has submitted ({!Rig.submitted}). Adding a check to a
    set then reads that set's earlier unread checks whose reading waits for
    nothing, where no other reader holds the claim, so that an issue never
    waits for one: on a device the new check computes on, once the device has
    reported that value ({!Rig.signaled}), which commits no work and waits
    for none; on another, which may get no more work, as
    {!Rig.Buffer.reached} finds it, which commits that device's work; and for
    a check read in place, once its kept buffers are {!written}. It reads
    them oldest first and stops at the first still running. A passing check
    goes, and a failing one stays as its failure until a wait raises it. A
    device reports its work at most its driver's batch of values late while
    work is submitted on it, so the unread checks no wait follows are those
    of the work in flight and of that batch; on a domain whose submits stop,
    they stay until a wait.

    This module lies above {!Exec} and below {!Eval}, [Rng] and [Nx]. [Nx]'s
    reads of a value on the host wait on its set first, and [Nx] and [Rng]
    read values on the host through {!on_host}. *)

val written : own:bool -> Rig.Buffer.t list -> bool
(** [written ~own bs] is whether a host read of the buffers [bs] waits for
    nothing, [own] being whether the devices whose work writes them have
    done it, which the caller asks of their timelines without committing
    ({!Rig.signaled}). Where [own] holds it asks each buffer
    ({!Rig.Buffer.reached}), which commits only the work of another device
    that writes one; it asks nothing otherwise. *)

type pending = {
  holds : (bool, Nx_array.Dtype.bool_elt) Nx_array.t;
      (** Whether every flag holds, in memory the host addresses. *)
  on : Rig.t;  (** The device that wrote [holds]. *)
  mark : int;  (** [on]'s value once [holds] was written. *)
  kept : (Rig.Buffer.keep * Rig.Buffer.t list) option;
      (** The keep of the caller's buffers the check reads in place, and
          those buffers. *)
  found : unit -> unit -> exn;
      (** Reads the failure's position and elements on the host, and gives
          the builder of its exception, which calls [fail]. *)
}
(** The type for pending checks, as {!Eval} computes them. *)

val add : devices:Rig.t list -> int -> pending -> unit
(** [add ~devices n c] adds [c] to the pending checks of the set numbered [n]
    ({!Devices.number}), then reads that set's earlier checks as stated
    above, [devices] being those [c] computed on.

    Raises what reading an earlier check raises, such as {!Rig.Lost}, with
    [c] added. *)

val add_read : devices:Rig.t list -> int -> (unit -> exn) option -> unit
(** [add_read ~devices n f] is {!add} for a check read at its issue: it keeps
    the builder of its exception for a failing one, [Some], and nothing for a
    passing one. *)

val traced : by:string -> Value.interpretation -> 'a
(** [traced ~by owner] refuses a read of a value [owner] traces, which has no
    elements. It raises [Invalid_argument] as [by: the value is traced by name]
    while [owner] runs, and as [by: a value of name, used after name returned]
    once it has returned, where [name] is [owner]'s. *)

val wait : by:string -> ('v, 's, 'd) Value.t -> unit
(** [wait ~by x] waits for the work [x]'s arrays depend on, then raises the
    oldest failure pending on [x]'s set as stated above, unless it runs
    inside the build of a failure's exception.

    Raises that failure: what its [fail] returns or raises, or what reading
    it raises, such as {!Rig.Lost}; and [Invalid_argument] naming [by] for a
    traced [x]. *)

val on_host : by:string -> ('v, 's, 'd) Value.t -> ('v, 's) Nx_array.t
(** [on_host ~by x] is an array on the host holding [x]'s elements, [x]
    placed there ({!Exec.value}), which is [x]'s own array where [x] lies
    on the host. No interpretation receives it, and it settles no check.

    Raises [Invalid_argument] naming [by] for a traced [x], and what
    {!Exec.value} raises. *)
