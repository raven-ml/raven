(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Celestial frames: the implementation of {!Ymir.Frame}.

    A frame is a constructor of {!t}, whose index is the frame's type, so
    matching on a frame refines its type. *)

(** The two families. No value of these types is ever built: their
    constructors make the two types distinct, so that a match on a frame of
    one family needs no case of the other. *)
type 'a fixed = Fixed_frame of 'a
type icrs = [ `Icrs ] fixed
type fk5_j2000 = [ `Fk5_j2000 ] fixed
type galactic = [ `Galactic ] fixed
type ecliptic_j2000 = [ `Ecliptic_j2000 ] fixed
type supergalactic = [ `Supergalactic ] fixed

type 'a geocentric = Geocentric_frame of 'a
type gcrs = [ `Gcrs ] geocentric
type cirs = [ `Cirs ] geocentric
type true_of_date = [ `True_of_date ] geocentric
type tirs = [ `Tirs ] geocentric

(** {!Ymir.Frame.t}, one constructor per frame. *)
type _ t =
  | Icrs : icrs t
  | Fk5_j2000 : fk5_j2000 t
  | Galactic : galactic t
  | Ecliptic_j2000 : ecliptic_j2000 t
  | Supergalactic : supergalactic t
  | Gcrs : gcrs t
  | Cirs : cirs t
  | True_of_date : true_of_date t
  | Tirs : tirs t

(** Which family a frame is in. *)
type _ kind = Fixed : 'a fixed kind | Geocentric : 'a geocentric kind

val kind : 'f t -> 'f kind
(** [kind f] is [f]'s family; matching on it refines [f]'s type. *)

val icrs : icrs t
val fk5_j2000 : fk5_j2000 t
val galactic : galactic t
val ecliptic_j2000 : ecliptic_j2000 t
val supergalactic : supergalactic t
val gcrs : gcrs t
val cirs : cirs t
val true_of_date : true_of_date t
val tirs : tirs t
val matrix : 'a fixed t -> 'b fixed t -> (float, Nx.float64_elt, Nx.host) Nx.t

val entries : 'a fixed t -> 'b fixed t -> float array
(** [entries a b] is [matrix a b]'s nine entries in row-major order. The
    array may be one of {!Orientation}'s or one shared by every call, so
    callers must not change it. *)

val matrix_at : 'd Earth.t -> 'a t -> 'b t -> 'd Earth.tensor

val name : 'f t -> string
val pp : Format.formatter -> 'f t -> unit
val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'f t -> 'f t
