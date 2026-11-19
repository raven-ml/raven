(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** TESSERA's Sentinel-1 recipe: the code it stores of radar backscatter, and
    the frame it makes of a day's products of one orbit direction on a tile's
    cells.

    TESSERA reads Microsoft's Planetary Computer's RTC γ⁰, linear power
    ({!Jord.Sentinel1}), by the nearest cell, as its preprocessing does
    (github.com/ucam-eo/tessera, [tessera_preprocessing/s1_fast_processor.py]):
    - a product's polarisation is kept when at least {!min_coverage} percent of
      the region of interest has a γ⁰ above 0, and a product when one of its two
      is;
    - each kept polarisation is stored as {!code}, and 0 outside the region or
      where it was not kept;
    - a day's cell takes, in each polarisation, the first nonzero code of its
      products in time order. *)

(** {1:code The code} *)

val code : Nx.host Nx.float64_t -> Nx.host Nx.int16_t
(** [code g] is TESSERA's int16 code of each γ⁰ [g], linear power: for a finite
    [g > 0], ⌊400 · dB + 10000⌋ clipped to \[[0], [32767]\], dB = 10 · log10 g,
    computed as TESSERA computes it, in float64, ⌊min (max ((20 · log10 g + 50)
    · 200, 0), 32767)⌋, with the C library's [log10]; and [0] for any other [g].
    So γ⁰ below −24.9975 dB (a code below 1) codes 0, which reads as no
    observation where both polarisations do ({!Pixels}). *)

val min_coverage : float
(** [min_coverage] is [0.01], in percent of the region of interest's cells. *)

val covered : roi:Nx.host Nx.bool_t -> Nx.host Nx.float64_t -> bool
(** [covered ~roi g] is [true] when at least {!min_coverage} percent of [roi]'s
    true cells have a finite γ⁰ above 0 in [g], [[h; w]], computed as TESSERA
    does: [100 · cells' / cells] in float64 against [0.01]. It is [false] for an
    empty region. *)

val merge : Nx.host Nx.int16_t list -> Nx.host Nx.int16_t
(** [merge codes] is, at each element, the first nonzero of [codes], all of one
    shape, or 0.

    Raises [Invalid_argument] if [codes] is empty or of different shapes. *)

(** {1:frames Frames} *)

val day :
  Stac.session ->
  roi:Nx.host Nx.bool_t ->
  centres:(Jord.Crs.t -> Nx.host Nx.float64_t) ->
  Jord_stac.item list ->
  Nx.host Nx.int16_t option
(** [day s ~roi ~centres items] is the frame, [[h; w; 2]], VV then VH, of MPC's
    RTC products [items], of one day and orbit direction, in time order, on the
    cells of a grid whose centres in a system are [centres crs]
    ({!Jord.Warp.centres}), and [roi] its region of interest: [None] if no
    product is kept. It reads each product's [vv] and [vh] by {!Stac.window}; a
    value its asset's band states as nodata has no γ⁰.

    Raises [Failure] naming the item if it lacks an asset or its grid, and as
    {!Stac.window} does. *)

val days :
  Jord_stac.item list ->
  ((Stac.date * Jord_stac.orbit_state) * Jord_stac.item list) list
  * Jord_stac.item list
(** [days items] is [items] grouped by the UTC date of their time
    ({!Stac.time}) and their orbit state, [Ascending] or [Descending], each
    group in time order, then by id, and the groups in the order of their
    first items; and the items left out for lacking [vv] or [vh].

    Raises [Failure] naming an item that states no orbit state,
    [Geostationary] or [Crossing]. *)
