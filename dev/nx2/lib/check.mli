(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Checks: conditions on data whose failure raises the caller's exception,
    at the call on the host and at the next wait on another set. This module
    holds the pending checks that [Nx]'s waiting section states.

    A {e check} is a [Check] operation: boolean {e flags} [ok] of one shape,
    {e data} that broadcast to it, and [fail], which makes the exception of
    the first false flag in C order from its index and the data's elements
    there. A check computes on the set where its flags lie. An empty [ok]
    passes.

    A check reduces its flags where they lie, by windows or by partials, to
    whether all hold and the first false position in C order. One flag is
    its own reduction: its element, on the device of the window that holds
    it, as a 0-d view, and position [0]. On the host, {!issue} reads the
    reduction and raises at once. On another set it leaves the reduction
    unread and adds the check to the calling domain's {e pending} checks on
    the set, which the domain's systhreads share. A pending check holds its
    data's memory and its reduction's, a flag's own memory for one flag
    ({!Rig.Claim.hold}), so that no donation writes what its failure
    reports. A {!wait} on the set drains them: it reads them oldest first
    and raises the first failure. A failing check's exception is built by
    its [fail], which may read the values it receives and so wait on the
    set. A wait inside that build reads only the checks issued before the
    one it builds, and a wait on another systhread meanwhile builds the
    exception itself and raises it. The failure raises at one wait.

    Once a pending check's reduction is submitted, the check records the
    value its device has submitted ({!Rig.submitted}). Issuing a check on a
    set first settles that set's earlier pending checks whose reduction is
    written: on a device the new check computes on, once the device has
    reported that value ({!Rig.signaled}), which commits no work and waits
    for none; on another, which may get no more work, as
    {!Rig.Buffer.reached} finds it, which commits that device's work. A
    passing check goes and a failing one keeps only its exception; the
    checks after a failure go, since no wait can raise them. A device
    reports its work at most its driver's batch of values late while work
    is submitted on it, so the checks no wait follows hold the data of the
    work in flight and of that batch; on a domain whose submits stop, they
    hold it until a wait. The new check is computed and kept only if no
    failure of the set is known then, and a raise while settling leaves it
    unissued.

    The host reads a reduction in place where it addresses its device's
    memory. On a device of this machine whose memory the host does not
    address, whether all flags hold is copied in the device's order into
    [Pinned] memory ({!Rig.Buffer.kind}), which both address: one more copy
    submitted per check, which nothing waits for. On another machine's
    device the host reads it through a copy, which follows that device's
    later work, so settling such a check waits for the work submitted
    there. A failing check's elements are read on the host, through a copy
    where the host does not address them.

    This module lies above {!Exec} and {!Partial}, which compute what it
    reads, and below {!Eval}, [Rng] and [Nx]. {!Eval.run} issues each check,
    and waits on a value's set before it places the value onto the host from
    another set. [Nx]'s reads of a value on the host wait on its set first.
    [Nx] and [Rng] read values on the host through {!on_host}. *)

val issue :
  by:string ->
  (bool, Nx_array.Dtype.bool_elt, 'd) Value.t ->
  ('d, 'e) Value.data ->
  (int array -> 'e -> exn) ->
  unit
(** [issue ~by ok data fail] issues the check of the flags [ok] on [data],
    as stated above. [ok] and [data] must be alive, and [data] must
    broadcast to [ok]'s shape ({!Prim.layouts} of the [Check]).

    Raises [fail i d] on the host, where flag [i] is the first false one and
    [d] the data's elements at [i]; what reading an earlier check raises
    while it settles, such as {!Rig.Lost}; and what a refusal or decline of
    the flags' reduction raises, naming [by]. *)

val wait : by:string -> ('v, 's, 'd) Value.t -> unit
(** [wait ~by x] waits for the work [x]'s arrays depend on, then drains the
    calling domain's pending checks on [x]'s set as stated above. Each
    drained check's hold on its data ends then ({!Rig.Claim.release}).

    Raises the first failure among them, and [Invalid_argument] naming [by]
    for a traced [x]. *)

val on_host : by:string -> ('v, 's, 'd) Value.t -> ('v, 's) Nx_array.t
(** [on_host ~by x] is an array on the host holding [x]'s elements, [x]
    placed there ({!Place.value}), which is [x]'s own array where [x] lies
    on the host. No interpretation receives it, and it settles no check.

    Raises [Invalid_argument] naming [by] for a traced [x], and what
    {!Place.value} raises. *)
