(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Selected by dune on platforms without the Metal runtime library. *)

let opener : (string -> Tolk.Device.t) option = None
