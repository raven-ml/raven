(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A contraction's launches on gfx1201.

    A call's launch is chosen from the contraction's dtypes, its view's
    extents and strides, and its operands' alignment, never from the GPU's
    size or clocks, so that a result's bits depend on its shape alone. *)

type launch
(** The type for launches: a call's operands, kernels, geometry and
    workspace, as {!choose} decided them. *)

(** The type for what a plan does with a call. *)
type verdict =
  | Declined  (** Its kernels do not compute the call. *)
  | Nothing  (** The result has no element: no work. *)
  | Launch of launch  (** {!parts} and {!write} launch it. *)

val choose :
  Nx_kernel.Contract_view.t ->
  acc:Nx_array.Dtype.any ->
  dst:('v, 's) Nx_array.t ->
  init:('v, 's) Nx_array.t option ->
  ('a, 'b) Nx_array.t ->
  ('c, 'e) Nx_array.t ->
  verdict
(** [choose v ~acc ~dst ~init a b] plans the contraction of [a] with [b],
    computed in [acc], into [dst], whose axes [v] groups. The operands'
    buffers are live. *)

val workspace : launch -> int
(** [workspace l] is the bytes of workspace [l] addresses. *)

val sequences : int
(** [sequences] bounds {!sequence}. *)

val sequence : launch -> int
(** [sequence l] is [l]'s kernels and slots as a key below {!sequences}. *)

val parts : int -> Rig.Image.t -> Rig.Submission.part array
(** [parts key i] is the launches of the sequence [key] of [i]'s kernels,
    their refs into the slots {!buffers} fills.

    Raises [Invalid_argument] if [key] is not below {!sequences}. *)

val write : launch -> Rig.Submission.Run.t -> Rig.Submission.t -> unit
(** [write l r s] stores the geometry and parameters of [l] into [r]'s blocks
    of [s], a submission of [parts (sequence l)]. *)

val buffers : launch -> workspace:Rig.Buffer.t -> Rig.Buffer.t array
(** [buffers l ~workspace] is the buffers of a submit of [l], in slot order,
    [workspace] among them where [workspace l > 0]. [workspace] must hold
    [workspace l] bytes. *)
