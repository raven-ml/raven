(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A contraction's launches on Apple GPUs.

    A plan is chosen from the contraction's dtypes, its view's extents and
    strides, and its operands' alignment, never from the GPU's size or clocks,
    so that a result's bits depend on its shape alone. *)

type t
(** The type for plans: a call's kernels, geometry and workspace pieces,
    rewritten by {!choose}. *)

val make : unit -> t
(** [make ()] is a plan to {!choose}. *)

(** The type for what a plan does with a call. *)
type verdict =
  | Declined  (** Its kernels do not compute the call. *)
  | Nothing  (** The result has no element: no work. *)
  | Launches  (** {!parts} and {!write} launch it. *)

val choose :
  t ->
  Nx_kernel.Contract_view.t ->
  acc:Nx_array.Dtype.any ->
  dst:('v, 's) Nx_array.t ->
  init:('v, 's) Nx_array.t option ->
  ('a, 'b) Nx_array.t ->
  ('c, 'e) Nx_array.t ->
  verdict
(** [choose p v ~acc ~dst ~init a b] plans the contraction of [a] with [b],
    computed in [acc], into [dst], whose axes [v] groups. The operands'
    buffers are live. It allocates nothing. *)

val workspace : t -> int
(** [workspace p] is the bytes of workspace [p]'s launches address. *)

val sequences : int
(** [sequences] bounds {!sequence}. *)

val sequence : t -> int
(** [sequence p] is [p]'s kernels and slots as a key below {!sequences}: two
    plans of one key make the same {!parts}. *)

val access : t -> Rig.access array
(** [access p] is the access of each buffer of a run of [p], in slot order: [a],
    [b], the result, then [init] where given, then the workspace where
    [workspace p > 0]. *)

val parts : t -> Rig.Image.t -> Rig.Submission.part array
(** [parts p i] is the launches of [p]'s sequence of [i]'s kernels, their
    refs into the slots {!access} lists, each with its slot's access. *)

val write : Rig.Submission.Run.t -> Rig.Submission.t -> t -> unit
(** [write r s p] stores the geometry and parameters of [p]'s launches into
    [r]'s blocks of [s], a submission of [parts p]. *)
