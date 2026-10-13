(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A reduction's or a scan's launches on sm_89.

    A call's launch is chosen from the operand's dtype and layout and the
    reduced axes, never from the GPU's size or clocks; its ranges and lanes
    follow reduce.cu's association whatever it chooses, so a result's bits
    depend on the values and the shape alone. *)

type launch
(** The type for launches: a call's operand, result, kernels and geometry,
    as {!reduce} or {!scan} decided them. *)

(** The type for what a plan does with a call that does work or none. *)
type verdict =
  | Nothing  (** The result has no element: no work. *)
  | Launch of launch  (** {!parts} and {!write} launch it. *)

(** {1:plans Plans}

    A plan takes a call its caller has checked: [x] of [prog]'s operand's
    dtype, and [dst] of the shape {!Nx_kernel.Shape.reduce} or
    {!Nx_kernel.Shape.scan} gives. *)

val reduce :
  axes:int array ->
  Nx_kernel.monoid ->
  Nx_kernel.Prog.t ->
  dst:Nx_array.any ->
  ('v, 's) Nx_array.t ->
  (verdict, Nx_array.answer) result
(** [reduce ~axes m prog ~dst x] plans the reduction by [m] along [axes] of
    [prog]'s output read plain from [x], into [dst]. It computes a [Sum],
    [Prod], [Max] or [Min] of a program that is its one operand, into that
    operand's dtype, at [float32], [float64] and the 8- to 64-bit integers,
    and [Max] and [Min] at [bool]; it is [Error Declined] for the others. *)

val scan :
  axis:int ->
  Nx_kernel.monoid ->
  Nx_kernel.Prog.t ->
  dst:Nx_array.any ->
  ('v, 's) Nx_array.t ->
  (verdict, Nx_array.answer) result
(** [scan ~axis m prog ~dst x] plans the scan by [m] along [axis], as
    {!reduce} plans a reduction. *)

(** {1:launches Launches} *)

val workspace : launch -> int
(** [workspace l] is the bytes of workspace [l] addresses. *)

val sequences : int
(** [sequences] bounds {!sequence}. *)

val sequence : launch -> int
(** [sequence l] is [l]'s kernels as a key below {!sequences}. *)

val parts : int -> Rig.Image.t -> Rig.Submission.part array
(** [parts key i] is the launches of the sequence [key] of [i]'s kernels,
    their refs into the slots {!buffers} fills.

    Raises [Invalid_argument] if [key] is not below {!sequences}. *)

val write : launch -> Rig.Submission.Run.t -> Rig.Submission.t -> unit
(** [write l r s] stores the geometry and parameters of [l] into [r]'s blocks
    of [s], a submission of [parts (sequence l)]. *)

val buffers : launch -> workspace:Rig.Buffer.t -> Rig.Buffer.t array
(** [buffers l ~workspace] is the buffers of a submit of [l], in slot order:
    the operand, the result, then [workspace] where [workspace l > 0], which
    must hold that many bytes. *)
