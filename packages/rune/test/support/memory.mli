(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The bytes a call leaves allocated on a device of its own. *)

val device : Rig.t
(** [device] is the memory device the counts read. Its bytes are counted as
    differences across a call, so values other suites place there do not count.
*)

module On : Nx.Devices
(** [On] is the set of {!device} alone. *)

val settled : unit -> int
(** [settled ()] is the bytes {!device} holds after four rounds of collection,
    each returning to it what it collected: a chain of finalisers frees its
    memory one round late. *)

val held : (unit -> unit) -> int
(** [held f] runs [f] once uncounted, to warm the device, then is the bytes
    {!device} holds once [f ()] returns again beyond those it held before. *)
