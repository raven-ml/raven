(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Boxes: the implementation of {!Mynd.Box}. *)

val of_xywh : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val to_xywh : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val of_cxcywh : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val to_cxcywh : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val area : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val iou : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val of_mask : (float, 's) Nx.dtype -> 'd Nx.bool_t -> (float, 's, 'd) Nx.t

val nms :
  ?classes:'d Nx.int64_t ->
  iou:float ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  'd Nx.bool_t

val nms_rounds :
  ?classes:'d Nx.int64_t ->
  iou:float ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  'd Nx.bool_t * 'd Nx.int32_t
