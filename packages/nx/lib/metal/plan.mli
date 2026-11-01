(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A contraction's launches on Apple GPUs.

    A call's launch is chosen from the contraction's dtypes, its view's
    extents and strides, and its operands' alignment, never from the GPU's
    size or clocks. Each output's association depends on [k], the dtypes
    and the order alone, the order chosen from the rows, so that a window of
    a result computed alone in that order has the whole's bits. *)

type launch
(** The type for launches: a call's operands, kernel and geometry, as
    {!choose} decided them. *)

(** The type for what a plan does with a call that does work or none. *)
type verdict =
  | Nothing  (** The result has no element: no work. *)
  | Launch of launch  (** {!parts} and {!write} launch it. *)

val choose :
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  acc:Nx_array.Dtype.any ->
  dst:('v, 's) Nx_array.t ->
  init:('v, 's) Nx_array.t option ->
  ('a, 'b) Nx_array.t ->
  ('c, 'e) Nx_array.t ->
  (verdict, Nx_array.answer) result
(** [choose ~batch ~contracting ~acc ~dst ~init a b] plans the contraction
    of [a] with [b], computed in [acc], into [dst], as
    {!Nx_kernel.S.contract} takes it, unscaled and unsegmented: in lane order
    at 1 row, in chain order otherwise. A float product in chain order runs
    on the matrix units, and one in lane order on the skinny kernels, a
    threadgroup summing 1 row. It is [Error] of what {!View.v} answers where
    the axes do not group,
    and [Error Declined] where its kernels do not compute the call: a and b
    of different dtypes, dtypes outside the float and integer families, or a
    k above [2{^31} - 1]. The operands' buffers are live. *)

val pieces : launch -> int
(** [pieces l] is the number of launches [l] runs: one, unless its batch or
    rows pass [2{^30}] or its columns [2{^26}]. *)

val sequences : int
(** [sequences] bounds {!sequence}. *)

val sequence : launch -> int
(** [sequence l] is [l]'s kernel and slots as a key below {!sequences}. *)

val parts : int -> Rig.Image.t -> Rig.Submission.part array
(** [parts key i] is the launch of the sequence [key] of [i]'s kernels, its
    refs into the slots {!buffers} fills.

    Raises [Invalid_argument] if [key] is not below {!sequences}. *)

val write : launch -> piece:int -> Rig.Submission.Run.t -> Rig.Submission.t -> unit
(** [write l ~piece r s] stores the geometry and parameters of [l]'s piece
    [piece], below {!pieces}, into [r]'s block of [s], a submission of
    [parts (sequence l)]. *)

val buffers : launch -> Rig.Buffer.t array
(** [buffers l] is the buffers of a submit of [l], in slot order. *)
