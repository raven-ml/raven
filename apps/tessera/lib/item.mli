(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** STAC items as TESSERA's recipes read them: accessors jord's STAC items
    lack. *)

val fail : Jord_stac.item -> ('a, unit, string, 'b) format4 -> 'a
(** [fail it fmt] raises [Failure] with the message, after [it]'s id. *)

val group :
  (Jord_stac.item -> 'k) ->
  Jord_stac.item list ->
  ('k * Jord_stac.item list) list
(** [group key items] is [items] grouped by [key], each group in the order of
    [items], the groups in the order of their first items. *)

val string : string -> Jord_stac.item -> string
(** [string name it] is [it]'s property [name], a string.

    Raises [Failure] naming [it] if it has none. *)

val compare : Jord_stac.item -> Jord_stac.item -> int
(** [compare a b] orders items by {!Stac.time}, then id. *)

val has : string list -> Jord_stac.item -> bool
(** [has keys it] is [true] if [it] has an asset of each of [keys]. *)

val window :
  Stac.session ->
  centres:(Jord.Crs.t -> Nx.host Nx.float64_t) ->
  Jord.Warp.rule ->
  Jord_stac.item ->
  string ->
  Jord_stac.asset * Stac.window
(** [window s ~centres rule it key] is [it]'s asset [key] and the window
    {!Stac.window} reads of it at [centres crs], [crs] the system of the asset's
    grid as the item states it.

    Raises [Failure] naming [it] and [key] if the asset or its grid is missing,
    and as {!Stac.window} does. *)
