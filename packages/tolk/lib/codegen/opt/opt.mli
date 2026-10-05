(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Kernel optimisations.

    An optimisation reshapes the loops of a kernel. It names the axis it acts on
    by its position among the kernel's axes. *)

(** The type for the roles a split gives the axis it makes: an axis unrolled
    into vector lanes, a reduction unrolled within a thread, or an axis across
    the threads of a workgroup. *)
type target = Upcast | Unroll | Local

(** The type for kernel optimisations. *)
type t =
  | Tc of { axis : int; tc_select : int; tc_opt : int; use_tc : int }
      (** Use a tensor core on the reduction at [axis]: [tc_select] picks the
          core ([-1] the first that fits), [tc_opt] how far to relax its
          requirements, from 0 to 2, and [use_tc] how to use it, 1 or 2.
          [tc_opt] and [use_tc] carry the levels of the settings
          {!Setting.tc_opt} or {!Setting.beam_tc_opt}, and {!Setting.use_tc}. *)
  | Split of { axis : int; amount : int; target : target; top : bool }
      (** Split [amount] out of [axis] into a new axis for [target], taken from
          the outer end if [top]. An [amount] of [0] takes the whole axis. *)
  | Padto of { axis : int; amount : int }
      (** Pad [axis] to a multiple of [amount]. *)
  | Swap of { axis : int; with_axis : int }  (** Swap two global axes. *)

val axis : t -> int
(** [axis o] is the axis [o] acts on. *)

val equal : t -> t -> bool
(** [equal o0 o1] is [true] iff [o0] and [o1] are the same optimisation. *)

val compare : t -> t -> int
(** [compare o0 o1] orders optimisations by kind ({!Tc}, {!Split}, {!Padto},
    {!Swap}), then axis, then arguments. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats [Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST))]. *)
