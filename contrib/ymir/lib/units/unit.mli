(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Units, as {!Ymir_units.Unit} documents them. *)

type t

val one : t
val int : int -> t
val decimal : string -> t
val pi : t
val symbol : string -> t
val scoped : scope:string -> string -> t
val ( * ) : t -> t -> t
val ( / ) : t -> t -> t
val ( ** ) : t -> int -> t
val root : int -> t -> t
val convertible : t -> t -> bool
val ratio : ('a, 'b) Nx.dtype -> t -> t -> 'a

type term =
  | Prime of int
  | Pi
  | Symbol of { name : string; scope : string option }

val terms : t -> (term * int * int) list
val to_string : t -> string
val of_string : string -> (t, string) result
val pp : Format.formatter -> t -> unit
val equal : t -> t -> bool
val compare : t -> t -> int
val ptree : t Nx.Ptree.t
val metre : t
val kilogram : t
val second : t
val ampere : t
val kelvin : t
val mole : t
val candela : t
val radian : t
val steradian : t
val hertz : t
val newton : t
val pascal : t
val joule : t
val watt : t
val coulomb : t
val volt : t
val farad : t
val ohm : t
val siemens : t
val weber : t
val tesla : t
val henry : t
val lumen : t
val lux : t
val becquerel : t
val gray : t
val sievert : t
val katal : t
val gram : t
val tonne : t
val minute : t
val hour : t
val day : t
val litre : t
val hectare : t
val astronomical_unit : t
val degree : t
val arcminute : t
val arcsecond : t
val electronvolt : t
val quecto : t -> t
val ronto : t -> t
val yocto : t -> t
val zepto : t -> t
val atto : t -> t
val femto : t -> t
val pico : t -> t
val nano : t -> t
val micro : t -> t
val milli : t -> t
val centi : t -> t
val deci : t -> t
val deca : t -> t
val hecto : t -> t
val kilo : t -> t
val mega : t -> t
val giga : t -> t
val tera : t -> t
val peta : t -> t
val exa : t -> t
val zetta : t -> t
val yotta : t -> t
val ronna : t -> t
val quetta : t -> t
val speed_of_light : t
val planck : t
val hbar : t
val elementary_charge : t
val boltzmann : t
val avogadro : t
val caesium_frequency : t
val luminous_efficacy : t
val gas_constant : t
val faraday : t
val stefan_boltzmann : t
