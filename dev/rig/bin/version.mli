(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** rig's version, the [rig] package's. *)

val v : string
(** [v] is the version, as ["1.0.0~alpha3"], which [rig --version] prints and a
    session's first line carries: a machine's half and [rig run] speak to each
    other only at one version. *)
