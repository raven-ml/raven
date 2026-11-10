(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external now : unit -> (int[@untagged])
  = "caml_rig_edge_now_byte" "caml_rig_edge_now"
[@@noalloc]

external words : unit -> int * int = "caml_rig_edge_words"

external holds : unit -> (int[@untagged])
  = "caml_rig_edge_holds_byte" "caml_rig_edge_holds"
[@@noalloc]

external hold : unit -> unit = "caml_rig_edge_hold" [@@noalloc]
external unhold : unit -> unit = "caml_rig_edge_unhold" [@@noalloc]
external mask : unit -> unit = "caml_rig_edge_mask" [@@noalloc]
external unmask : unit -> unit = "caml_rig_edge_unmask"
external shield : unit -> unit = "caml_rig_edge_shield" [@@noalloc]
external unshield : unit -> unit = "caml_rig_edge_unshield"
external unblock : unit -> unit = "caml_rig_edge_unblock" [@@noalloc]
external block : unit -> unit = "caml_rig_edge_block"
