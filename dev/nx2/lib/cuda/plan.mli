(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A contraction's launches on sm_89.

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
  Nx_kernel.Spec.Contract_view.t ->
  Nx_kernel.Spec.contract Nx_kernel.Spec.t ->
  dst:Nx_array.any ->
  Nx_array.any array ->
  verdict
(** [choose p v s ~dst ops] plans the contraction [s] of [ops] into [dst], whose
    axes [v] groups. The operands' buffers are live. It allocates nothing. *)

val workspace : t -> int
(** [workspace p] is the bytes of workspace [p]'s launches address. *)

val sequences : int
(** [sequences] bounds {!sequence}. *)

val sequence : t -> int
(** [sequence p] is [p]'s kernels and slots as a key below {!sequences}: two
    plans of one key make the same {!parts}. *)

val reads : t -> int
(** [reads p] is the count of buffers [p]'s launches read: [a], [b], then [init]
    where given. *)

val writes : t -> int
(** [writes p] is the count of buffers they write: the result, then the
    workspace where [workspace p > 0], then the tickets where
    [tickets p > 0]. *)

val tickets : t -> int
(** [tickets p] is the bytes of tickets [p]'s split sum takes, from the start
    of a buffer whose words are zero, which [p]'s launches leave zero; [0]
    for no split. *)

val parts : t -> Rig.Image.t -> queue:string -> Rig.Submission.part array
(** [parts p i ~queue] is the launches of [p]'s sequence of [i]'s kernels on
    [queue], their refs into the slots {!reads} and {!writes} count. *)

val write : Rig.Submission.Run.t -> Rig.Submission.t -> t -> unit
(** [write r s p] stores the geometry and parameters of [p]'s launches into
    [r]'s blocks of [s], a submission of [parts p]. *)
