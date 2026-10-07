(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Structures as Qmd builds them. Structure exports these types private, so that
   every structure comes from Qmd. *)

type 'v hole = { at : int; bits : int; value : 'v Packet.term }
type 'v structure = { bytes : string; holes : 'v hole list }
