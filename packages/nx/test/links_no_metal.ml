(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx alone, linked: the build refuses it if Metal's runtime reaches it. *)

let () = ignore (Nx.Device.name Nx.Device.host)
