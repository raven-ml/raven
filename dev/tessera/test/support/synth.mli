(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Synthetic pixels for tessera's tests. *)

type spec = { s2 : int list; s1 : int list; seed : int }
(** The type for pixels by their observation counts: pixel [i] has
    [List.nth s2 i] Sentinel-2 and [List.nth s1 i] Sentinel-1 observations,
    whose values [seed] draws. [s2] and [s1] have one length. *)

val spec : ?pixels:int -> ?long:bool -> unit -> spec Windtrap.Gen.t
(** [spec ~pixels ~long ()] draws [1] to [pixels] pixels (default [6]): one
    in seven without observations, two in seven of one sensor, the others of
    both; counts of [1] or [2], near a multiple of 8, up to 20, and with
    [long] above 256. *)

val pixels : spec -> Nx.host Tessera.Pixels.t
(** [pixels s] is the pixels [s] describes: Sentinel-2 bands in [[1, 12000]],
    Sentinel-1 bands in [[1, 9000]] with an orbit each, days in [[1, 366]]. *)

val rebuild :
  Nx.host Tessera.Pixels.t ->
  Tessera.Batch.counts ->
  (int -> int array -> int array) ->
  Nx.host Tessera.Pixels.t * Tessera.Batch.counts
(** [rebuild px c f] is [px] and [c] with each row of each sensor replaced: for
    pixel [p] whose row has counts [cs], the new row is the observations
    [f p cs] indexes in the old one, with their counts. *)

val counts : Tessera.Batch.counts -> int array * int array
(** [counts c] is [c]'s counts of each sensor as OCaml arrays. *)
