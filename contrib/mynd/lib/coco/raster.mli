(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** COCO's polygon rule and the union of runs: the implementation of
    {!Mynd_coco.runs}. Counts are {!Mynd.Rle.t}'s, column-major from a run of
    [false] pixels. Each function takes [fn], the public name of its caller, for
    its messages. *)

val polygon : string -> height:int -> width:int -> float array -> int array
(** [polygon fn ~height ~width xy] is the counts of the polygon [xy]'s mask by
    {!Mynd_coco.runs}' rule, with no zero count but the first.

    Raises [Invalid_argument] as {!Mynd_coco.runs} does on a polygon. [height]
    and [width] must be non-negative. *)

val union : int array -> int array -> int array
(** [union a b] is the counts of the union of the masks [a] and [b], with no
    zero count but the first. Their counts must sum to the same total. *)
