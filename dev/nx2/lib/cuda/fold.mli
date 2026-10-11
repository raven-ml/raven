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
(** The type for plans, rewritten by {!reduce} and {!scan}. *)

val make : unit -> t
(** [make ()] is a plan to {!reduce} or {!scan}. *)

(** The type for what a plan does with a call. *)
type verdict =
  | Declined  (** Its kernels do not compute the call. *)
  | Refused of Nx_array.answer  (** The call is wrong: this refusal. *)
  | Nothing  (** The result has no element: no work. *)
  | Launches  (** {!parts} and {!write} launch it. *)

val reduce :
  t ->
  axes:int array ->
  Nx_kernel.monoid ->
  Nx_kernel.Prog.t ->
  dst:Nx_array.any ->
  ('v, 's) Nx_array.t ->
  verdict
(** [reduce p ~axes m prog ~dst x] plans the reduction by [m] along [axes] of
    [prog]'s output read plain from [x], into [dst]. It computes a [Sum],
    [Prod], [Max] or [Min] of a program that is its one operand, into that
    operand's dtype, at [float32], [float64] and the 8- to 64-bit integers,
    and [Max] and [Min] at [bool]. It refuses [Wrong_dtype] for an [x] of
    another dtype than [prog]'s operand, and [Shape_mismatch] where
    {!Nx_kernel.Shape.reduce} answers [Error] or [dst]'s shape is not the one
    it gives. It allocates nothing but where the reduction has outputs and no
    term. *)

val scan :
  t ->
  axis:int ->
  Nx_kernel.monoid ->
  Nx_kernel.Prog.t ->
  dst:Nx_array.any ->
  ('v, 's) Nx_array.t ->
  verdict
(** [scan p ~axis m prog ~dst x] plans the scan by [m] along [axis], as
    {!reduce} plans a reduction, with {!Nx_kernel.Shape.scan}'s shape. It
    allocates nothing. *)

val workspace : t -> int
(** [workspace p] is the bytes of workspace [p]'s launches address. *)

val sequences : int
(** [sequences] bounds {!sequence}. *)

val sequence : t -> int
(** [sequence p] is [p]'s kernels as a key below {!sequences}: two plans of
    one key make the same {!parts}. *)

val access : t -> Rig.access array
(** [access p] is the access of each buffer of a run of [p], in slot order:
    the operand, the result, then the workspace where [workspace p > 0]. *)

val parts : t -> Rig.Image.t -> Rig.Submission.part array
(** [parts p i] is the launches of [p]'s sequence of [i]'s kernels, their
    refs into the slots {!access} lists, each with its slot's access. *)

val write : Rig.Submission.Run.t -> Rig.Submission.t -> t -> unit
(** [write r s p] stores the geometry and parameters of [p]'s launches into
    [r]'s blocks of [s], a submission of [parts p]. *)
