(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Instants on time scales: the implementation of {!Ymir.Time}.

    A scale is a constructor of {!scale}, whose index is the scale's type,
    so matching on a scale refines its type. *)

type !'a uniform
type utc
type tai = [ `Tai ] uniform
type tt = [ `Tt ] uniform
type tdb = [ `Tdb ] uniform

(** {!Ymir.Time.scale}, one constructor per scale. *)
type _ scale =
  | Utc : utc scale
  | Tai : tai scale
  | Tt : tt scale
  | Tdb : tdb scale

val utc : utc scale
val tai : tai scale
val tt : tt scale
val tdb : tdb scale
val name : 's scale -> string

type ('s, 'd) t = {
  scale : 's scale;
  day : (int64, Nx.int64_elt, 'd) Nx.t;
  second : (float, Nx.float64_elt, 'd) Nx.t;
}
(** {!Ymir.Time.t}: [day] is the modified Julian date and [second] the
    seconds since its 0h, in \[0, the day's length), or NaN. *)

val v :
  's scale ->
  day:(int64, Nx.int64_elt, 'd) Nx.t ->
  second:(float, Nx.float64_elt, 'd) Nx.t Ymir_units.Quantity.t ->
  ('s, 'd) t

val of_jd :
  ('a uniform as 's) scale ->
  (float, Nx.float64_elt, 'd) Nx.t ->
  (float, Nx.float64_elt, 'd) Nx.t ->
  ('s, 'd) t

val of_date :
  's scale -> int * int * int -> int * int * float -> ('s, Nx.host) t
val scale : ('s, 'd) t -> 's scale
val day : ('s, 'd) t -> (int64, Nx.int64_elt, 'd) Nx.t
val second :
  ('s, 'd) t -> (float, Nx.float64_elt, 'd) Nx.t Ymir_units.Quantity.t

val jd :
  ('a uniform, 'd) t ->
  (float, Nx.float64_elt, 'd) Nx.t * (float, Nx.float64_elt, 'd) Nx.t

val date : ('s, Nx.host) t -> (int * int * int) * (int * int * float)
val convert : 'b scale -> ('a, 'd) t -> ('b, 'd) t

val diff :
  ('a uniform as 's, 'd) t ->
  ('s, 'd) t ->
  (float, Nx.float64_elt, 'd) Nx.t Ymir_units.Quantity.t

val add :
  ('a uniform as 's, 'd) t ->
  (float, Nx.float64_elt, 'd) Nx.t Ymir_units.Quantity.t ->
  ('s, 'd) t

val leap_seconds_known_until : (utc, Nx.host) t
val carry :
  (int64, Nx.int64_elt, 'd) Nx.t ->
  (float, Nx.float64_elt, 'd) Nx.t ->
  (float, Nx.float64_elt, 'd) Nx.t ->
  (int64, Nx.int64_elt, 'd) Nx.t * (float, Nx.float64_elt, 'd) Nx.t
(** [carry d a b] is the instant [a + b] seconds after 0h of day [d], as a
    day and the seconds in \[0, 86400): the sum and the whole days taken
    off it are exact, so the seconds round once. A NaN sum keeps day [d]. *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('s, 'd) t -> ('s, 'e) t
val ptree : unit -> (('s, 'd) t, 'd) Nx.Ptree.t
