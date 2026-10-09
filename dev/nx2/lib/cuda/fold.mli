(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A reduction's or a scan's launches on sm_89.

    A plan is chosen from the operand's dtype and layout and the reduced
    axes, never from the GPU's size or clocks; its ranges and lanes follow
    fold.cu's association whatever it chooses, so a result's bits depend on
    the values and the shape alone. *)

type t
(** The type for plans, rewritten by {!choose}. *)

val make : unit -> t
(** [make ()] is a plan to {!choose}. *)

(** The type for what a plan does with a call. *)
type verdict =
  | Declined  (** Its kernels do not compute the call. *)
  | Refused of Nx_array.answer  (** The call is wrong: this refusal. *)
  | Nothing  (** The result has no element: no work. *)
  | Launches  (** {!parts} and {!write} launch it. *)

val choose :
  t ->
  [ `Reduce | `Scan ] ->
  [< `Reduce | `Scan ] Nx_kernel.Spec.t ->
  dst:Nx_array.any ->
  Nx_array.any ->
  verdict
(** [choose p family s ~dst x] plans [s], a reduction or a scan as [family]
    says, of the operand [x] into [dst]. It computes one [Sum], [Prod], [Max]
    or [Min] of a program's one operand into its own dtype, read plain, at
    [float32], [float64] and the 8- to 64-bit integers, and [Max] and [Min]
    at [bool]. It refuses [Wrong_dtype] for an operand or destination of
    another dtype than [s]'s, and [Shape_mismatch] for a destination of
    another shape than the result's, or a [Max] or [Min] of outputs with no
    term. It allocates nothing. *)

val workspace : t -> int
(** [workspace p] is the bytes of workspace [p]'s launches address. *)

val sequences : int
(** [sequences] bounds {!sequence}. *)

val sequence : t -> int
(** [sequence p] is [p]'s kernels as a key below {!sequences}: two plans of
    one key make the same {!parts}. *)

val writes : t -> int
(** [writes p] is the count of buffers [p]'s launches write: the result,
    then the workspace where [workspace p > 0]. They read one, the
    operand. *)

val parts : t -> Rig.Image.t -> queue:string -> Rig.Submission.part array
(** [parts p i ~queue] is the launches of [p]'s sequence of [i]'s kernels on
    [queue], their refs into the operand, then the slots {!writes} counts. *)

val write : Rig.Submission.Run.t -> Rig.Submission.t -> t -> unit
(** [write r s p] stores the geometry and parameters of [p]'s launches into
    [r]'s blocks of [s], a submission of [parts p]. *)
