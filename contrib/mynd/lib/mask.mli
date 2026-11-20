(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Masks and their components: the implementation of {!Mynd.Mask}. *)

type connectivity = [ `Four | `Eight ]

val iou :
  (float, 's) Nx.dtype -> 'd Nx.bool_t -> 'd Nx.bool_t -> (float, 's, 'd) Nx.t

val components : connectivity -> 'd Nx.bool_t -> 'd Nx.int64_t

val components_rounds :
  connectivity -> 'd Nx.bool_t -> 'd Nx.int64_t * 'd Nx.int32_t

val reconstruct :
  connectivity -> marker:'d Nx.bool_t -> 'd Nx.bool_t -> 'd Nx.bool_t
