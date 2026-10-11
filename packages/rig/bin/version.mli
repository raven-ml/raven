(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** rig's version, the [rig] package's. *)

val v : string
(** [v] is the version, as ["1.0.0~alpha3"], which [rig --version] prints. *)

val session : string
(** [session] is what a session's first line carries: [v] and the version of
    the protocol between a job's processes, as ["1.0.0~alpha3 protocol 5"]. A
    machine's half and [rig run] speak to each other only at one [session]:
    builds of one release can speak different protocols. *)
