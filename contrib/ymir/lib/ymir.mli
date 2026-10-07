(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Astronomy over {!Nx} tensors.

    Ymir gives astronomical meaning to tensors: the units astronomy measures in,
    and the models built on them. Every function is a formula of nx operations,
    so it runs batched, compiled and differentiated under rune's transformations
    with no rule of its own.

    [open Ymir] brings the unit algebra of [ymir.units] ({!Unit}, {!Quantity},
    {!Constant}, {!Codata}, {!Vocabulary}) and astronomy's units ({!Units}) into
    scope. *)

(** {1:units Units} *)

module Unit = Ymir_units.Unit
(** Units as exact values. *)

module Quantity = Ymir_units.Quantity
(** Values in a unit. *)

module Constant = Ymir_units.Constant
(** Measured constants. *)

module Codata = Ymir_units.Codata
(** CODATA releases of the measured constants. *)

module Vocabulary = Ymir_units.Vocabulary
(** Names for units. *)

(** Astronomy's units.

    Each is an exact {!Unit.t}, built from the SI's units and the IAU's
    definitions. *)
module Units : sig
  val parsec : Unit.t
  (** [parsec] is 648000/π {!Unit.astronomical_unit}, the distance at which one
      astronomical unit subtends one arcsecond. *)

  val julian_year : Unit.t
  (** [julian_year] is 365.25 {!Unit.day}, 31557600 s. *)
end
