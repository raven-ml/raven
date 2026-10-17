(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Celestial frames: the implementation of {!Ymir.Frame}.

    A frame is a constructor of {!t}, whose index is the frame's type, so
    matching on a frame refines its type. *)

type !'a fixed
type icrs = [ `Icrs ] fixed
type fk5_j2000 = [ `Fk5_j2000 ] fixed
type galactic = [ `Galactic ] fixed
type ecliptic_j2000 = [ `Ecliptic_j2000 ] fixed
type supergalactic = [ `Supergalactic ] fixed

(** {!Ymir.Frame.t}, one constructor per frame. *)
type _ t =
  | Icrs : icrs t
  | Fk5_j2000 : fk5_j2000 t
  | Galactic : galactic t
  | Ecliptic_j2000 : ecliptic_j2000 t
  | Supergalactic : supergalactic t

val icrs : icrs t
val fk5_j2000 : fk5_j2000 t
val galactic : galactic t
val ecliptic_j2000 : ecliptic_j2000 t
val supergalactic : supergalactic t
val matrix : 'a fixed t -> 'b fixed t -> (float, Nx.float64_elt) Nx.t

val entries : 'a fixed t -> 'b fixed t -> float array
(** [entries a b] is [matrix a b]'s nine entries in row-major order. The
    array may be one of {!Orientation}'s or one shared by every call, so
    callers must not change it. *)

val name : 'f t -> string
val pp : Format.formatter -> 'f t -> unit
val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'f t -> 'f t
