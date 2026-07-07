(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Selected by dune when the Metal runtime library is available. *)

let opener : (string -> Tolk.Device.t) option = Some Tolk_metal.create
