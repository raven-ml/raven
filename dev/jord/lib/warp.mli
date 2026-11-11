(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Arrays moved from one grid to another.

    To {e warp} an array on a grid [src] onto a grid [dst] gives each cell of
    [dst] the array's value at the cell's centre: the centre is taken back into
    [src]'s plane and read there, a point sample by inverse mapping. Each step
    is exact to the transforms' stated error ({!Crs}), with no approximation of
    the map between the grids.

    A cell of [src] holds its value over its whole square, GeoTIFF's
    PixelIsArea (OGC 19-008r4, Annex B.2.2), so a point has data when the cell
    containing it does. ISO 19123 (§8.11.1) puts an interpolated grid's domain
    on the hull of the cells' centres, half a cell inside; the cells' area is
    kept instead, as the sample space its §8.2.2 describes, so that nearest and
    bilinear share one footprint and source tiles warped one at a time leave no
    seam between them. A raster of postings that [Jord_geotiff.grid] reads as
    PixelIsPoint is read the same way: a posting's value holds over the cell
    centred on it, half a cell past the outermost postings.

    The work has three steps. {!centres} takes [dst]'s centres into [src]'s
    system, the costly step, which depends only on [dst] and [src.crs];
    {!pixel} takes them to [src]'s pixel points; a sampler reads the array
    there and gives the values with a mask of the cells that have data:

    {[
      let p = Warp.pixel src (Warp.centres ~at:Nx.Host.on src.crs dst) in
      let labels, ok = Warp.nearest ~valid p labels in
      let bands, ok' = Warp.bilinear ~valid p bands
    ]}

    One [centres] serves every grid in [src.crs], as products whose grid moves
    from date to date; one [p] serves every array on [src]. A cell of [dst]
    coarser than [src]'s reads the source at its centre, not the mean or the
    mode of the cells it covers. Each cell's result depends on that cell alone,
    so warping windows of [dst] ({!Grid.window}) to bound memory gives the same
    values. *)

val centres : at:'d Nx.Placement.t -> Crs.t -> Grid.t -> 'd Nx.float64_t
(** [centres ~at crs dst] is each cell centre of [dst] as a point of [crs],
    [[dst.rows; dst.cols; 2]], made at [at]. It is {!Grid.centres} when [crs] is
    [dst.crs], with no transform, so equal systems keep the affine maps' exact
    lattice. Otherwise the centres go to [Lonlat], a centre outside [crs]'s
    domain ({!Crs.in_domain}) becomes NaN, a point with no data, and the others
    go on to [crs] ({!Crs.transform}).

    Raises as {!Crs.transform} does where a centre lies outside [dst.crs]'s own
    domain. *)

val pixel : Grid.t -> 'd Nx.float64_t -> 'd Nx.float64_t
(** [pixel src x] is {!Grid.pixel}[ src x], with each longitude of a [Lonlat]
    [src] first taken into the 360° from its westernmost corner: λ becomes
    w + ((λ - w) mod 360), where w is the least longitude of [src]'s four
    corners. A longitude λ ≥ w whose λ - w rounds below 360 is kept exactly.
    So a meridian finds its cells whichever 360° [src] uses, 0..360 or across
    the antimeridian, and a grid narrower than 360° that crosses the
    antimeridian reads on both sides.

    Raises [Invalid_argument] if [x]'s last axis is not 2. *)

(** {1:samplers Samplers}

    A sampler reads an array [x] at pixel points [p], [[...; 2]], of the grid
    of [x]'s first two axes. Its result has [p]'s batch shape followed by
    [x]'s other axes, and a mask of [p]'s batch shape that holds where the
    cell containing the point, [(⌊row⌋, ⌊col⌋)], lies in the grid and [valid]
    holds there. A NaN point is in no cell. [valid] has [x]'s first two axes,
    and every cell has data when it is absent: a nodata value, sentinels or
    NaN become a mask by {!Nx.not_equal_s}, {!Nx.logical_and} and
    {!Nx.isnan}. Where the mask is false the value is 0.

    Raises [Invalid_argument] if [p]'s last axis is not 2, [x] has fewer than
    two axes, or [valid]'s shape is not [x]'s first two. *)

val nearest :
  ?valid:'d Nx.bool_t ->
  'd Nx.float64_t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t * 'd Nx.bool_t
(** [nearest ?valid p x] is [x]'s cell containing each point, in [x]'s dtype.
    A point on a border between two cells reads the one after it. This is the
    method for classes and other nominal values (ISO 19123, §C.2). *)

val bilinear :
  ?valid:'d Nx.bool_t ->
  'd Nx.float64_t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t * 'd Nx.bool_t
(** [bilinear ?valid p x] interpolates between the centres of the four cells
    around each point, weighted as ISO 19123 §C.6 gives in pixel coordinates.
    A cell outside the grid or not [valid] is left out, by selection, and so
    is a cell of weight 0 whose value is NaN or infinite, so that value never
    reaches the result; the other weights are divided by their sum. A cell of
    weight 0 whose value is finite is kept and adds 0, whose sign can make a
    mean of 0 read as -0. The value is computed in float64. Its mask is
    {!nearest}'s: the containing cell has weight at least 1/4, so the value is
    a mean of valid values around the point, and at the grid's edge it is the
    edge's values interpolated along it. A NaN of a valid cell of positive
    weight is data, and gives NaN.

    The value is stored in [x]'s dtype. A float rounds once to it. On integers
    the value v becomes ⌊v + ½⌋ in float64 arithmetic, GDAL's rule, on every
    sign (-2.5 gives -2); it lies between valid values, so it never
    saturates. Integers past 2⁵³ in magnitude are rounded to float64 first.

    {b Gradient.} The value is nx arithmetic in the points, so rune
    differentiates it in [p] for a float [x]; an integer value is rounded, and
    has none, so a caller decodes integers to floats first. The gradient is
    the derivative of the mean over the cells kept at the point. A cell left
    out (outside the grid, not valid, or of weight 0 and not finite) adds
    nothing to it, so along that cell's axis the derivative is 0, as at the
    grid's edge: a NaN neighbour of weight 0 gives 0, as a hole does, and its
    NaN reaches the value once the point moves toward it. The four cells come
    from ⌊row - ½⌋ and ⌊col - ½⌋, so a point on the line through a row or
    column of centres belongs to the span after it, whose far cells have
    weight 0. Where they are kept, that is the right-hand derivative, toward
    them. So descent in the points moves from a start on the centres, as the
    centres of a grid in the source's own system are.

    Raises [Invalid_argument] as the samplers do, and if [x] holds booleans
    or complex numbers. *)

(** {1:windows Windows} *)

(** The type for the samplers, as {!window} names them. *)
type rule = Nearest | Bilinear

val window : rule -> Grid.t -> 'd Nx.float64_t -> int * int * int * int
(** [window rule src p] is [(row, col, rows, cols)], the block of [src]'s cells
    that [rule] reads at the pixel points [p] of [src] whose containing cell
    lies in [src]: those cells for [Nearest], and for [Bilinear] the four
    around each point, clipped to [src]. It is [(0, 0, 0, 0)] where no point
    has a cell. A ranged reader reads only that block [x'] of [src]'s array:
    [rule] on [x'] at [p] less [(row, col)] gives the values and mask it gives
    on the whole array, bit for bit, since subtracting those integers from the
    points is exact. It reads five numbers back to the host.

    Raises [Invalid_argument] if [p]'s last axis is not 2. *)
