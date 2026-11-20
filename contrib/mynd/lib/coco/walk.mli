(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pixel counts of two masks over their runs, for {!Mynd_coco.evaluate_masks}.
    Counts are {!Mynd.Rle.t}'s; the two masks have the same size. *)

val columns : height:int -> int array -> int * int
(** [columns ~height c] is the first and last column holding a [true] pixel of
    the mask [c], and [(1, 0)] when it has none. *)

val iou : int array -> int array -> crowd:bool -> float
(** [iou d g ~crowd] is the IoU of the masks [d] and [g], [0] when they share no
    pixel, with [d]'s area for divisor when [crowd]. *)
