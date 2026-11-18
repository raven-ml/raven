(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Arrays moved from one grid to another.

    To {e warp} an array on a grid [src] onto a grid [dst] gives each cell of
    [dst] the array's value at the cell's centre. The centre is taken back into
    [src]'s plane and read there, a point sample by inverse mapping. Each step
    is exact to the transforms' stated error ({!Crs}), with no approximation of
    the map between the grids.

    A cell of [src] holds its value over its whole square ({!Grid}), so a point
    has data when the cell containing it does. ISO 19123 (§8.11.1) puts an
    interpolated grid's domain on the hull of the cells' centres, half a cell
    inside; the cells' area is kept instead, as the sample space its §8.2.2
    describes, so that nearest and bilinear share one domain and source tiles
    warped one at a time leave no gap between them. A raster of
    {e postings}, values sampled at points of the plane, is read the same way,
    each posting the centre of a cell, as [Jord_geotiff.grid] reads
    PixelIsPoint. A posting's value holds over its cell, so the raster extends
    half a cell past the outermost postings.

    The work has three steps. {!centres} takes [dst]'s centres into [src]'s
    system, the costly step, which depends only on [dst] and [src.crs];
    {!pixel} takes them to [src]'s pixel points; a sampler reads the array
    there and gives the values with a mask of the cells that have data:

    {[
      let p = Warp.pixel src (Warp.centres ~at:Nx.Host.on src.crs dst) in
      let labels, ok = Warp.nearest ~valid src p labels in
      let bands, ok' = Warp.bilinear ~valid src p bands
    ]}

    One [centres] serves every grid in [src.crs], as products whose grid moves
    from date to date; one [p] serves every array on [src]. Each cell's result
    depends on that cell alone, so warping windows of [dst] ({!Grid.window})
    to bound memory gives the same values. A rule reads [src]'s array whole
    or a {e block} of it, the array of a rectangle of [src]'s cells from
    cell [(row, col)], as a program reads it from a file's tiles, at the
    whole grid's [src] ({!section-windows}).

    Each rule gives a cell the source's field over the cell's {e footprint},
    the part of [src]'s pixel plane it reads: {!nearest} reads the centre,
    {!bilinear} the square of one source cell centred there, and the
    {{!section-area}area rules} {!average} and {!mode} the cell itself, the
    quadrilateral of its four corners in [src]'s pixel plane, joined by
    straight edges. A sampler on a [dst] coarser than [src] aliases, in the
    sense that detail finer than a cell of [dst] folds into false patterns.
    So a coarser grid takes {!average}, or {!mode} for classes, and an equal
    or finer one a sampler. *)

val centres : at:'d Nx.Placement.t -> Crs.t -> Grid.t -> 'd Nx.float64_t
(** [centres ~at crs dst] is each cell centre of [dst] as a point of [crs],
    [[dst.rows; dst.cols; 2]], made at [at]. It is {!Grid.centres} when [crs] is
    [dst.crs], with no transform, so equal systems keep the affine maps' exact
    lattice. Otherwise the centres go to the longitude and latitude of
    [dst.crs]'s datum, a centre outside the domain of the transforms from
    there to [crs] ({!Crs.in_domain}) becomes NaN, a point with no data, and
    the others go on to [crs] ({!Crs.transform}).

    Raises as {!Crs.transform} does where a centre lies outside [dst.crs]'s own
    domain. *)

val corners : at:'d Nx.Placement.t -> Crs.t -> Grid.t -> 'd Nx.float64_t
(** [corners ~at crs dst] is each pixel point [(i, j)] of [dst], for
    [i ≤ dst.rows] and [j ≤ dst.cols], as a point of [crs],
    [[dst.rows + 1; dst.cols + 1; 2]], made at [at]: the corners of [dst]'s
    cells, which the {{!section-area}area rules} read, by {!centres}'s laws
    applied to {!Grid.world} of the pixel points.

    The area rules cannot place a cell with a corner 180° or more of
    longitude from its first ({!section-antimeridian}), by [dst]'s own
    longitudes, on a projection λ - λ₀ in \[-180°, 180°\] on the easting's
    side of λ₀. [corners] checks the cells of every [dst] but a projected
    one in [crs] itself, whose cells are their footprints, and one of
    Albers' conic, whose cut a valid cell can cross.

    Raises [Invalid_argument] if a cell it checks has such a corner, or,
    where [crs] is not [dst.crs], if a corner lies outside [dst.crs]'s
    domain, at its index (each checked by {!Nx.check}). *)

val pixel : Grid.t -> 'd Nx.float64_t -> 'd Nx.float64_t
(** [pixel src x] is {!Grid.pixel}[ src x], with each longitude of a
    geographic [src] first taken into the 360° from its westernmost corner: λ
    becomes w + ((λ - w) mod 360), where w is the least longitude of [src]'s
    four corners. A longitude λ ≥ w whose λ - w rounds below 360 is kept
    exactly. So a meridian finds its cells whichever 360° [src] uses, 0..360
    or across the antimeridian, and a grid narrower than 360° that crosses the
    antimeridian reads on both sides.

    Raises [Invalid_argument] if [x]'s last axis is not 2. *)

(** {1:samplers Samplers}

    A sampler reads an array [x] on a grid [src] at pixel points [p] of
    [src], [[...; 2]]. It gives values of [p]'s batch shape followed by
    [x]'s other axes, and a mask of the values' shape, which keeps [x]'s
    other axes and is the same along them, that holds where the cell
    containing the point, [(⌊row⌋, ⌊col⌋)], lies in [src] and [valid] holds
    there. Where the mask is false the value is 0, and a NaN point's mask is
    false. [valid] holds one value per cell ({!section-valid}), and without
    it every cell has data. [x] and [valid] may be the block of [src]'s
    cells from cell [(row, col)], 0 by default, such as {!window} gives,
    while [p] stays the whole grid's ({!section-windows}).

    A program must join masks of one shape, since a [[r; c]] mask and a
    [[r; c; 1]] one broadcast to another shape when joined, without an error
    where [r = c], [r = 1] or [c = 1]. So the mask of bands stacked
    [[h; w; 4]] is reduced over its band axis before it joins the mask of a
    layer of [[h; w]], such as a scene classification:

    {[
      let bands, ok = Warp.bilinear ~valid src p bands in
      let scl, seen = Warp.nearest src' p' scl in
      let ok = Nx.logical_and (Nx.all ~axes:[ 2 ] ok) seen
    ]}

    A geographic [src]'s {e turn} is the step of its pixel points that adds
    360° to their longitude, found by the inverse of [src]'s map. [src]
    {e spans 360°} where its turn's row is 0, as where its columns run along
    a parallel, and its column is [±cols] to within 2{^-50}·cols, east or
    west. {!bilinear} and the {{!section-area}area rules} read such a
    [src]'s columns modulo their number, as {!pixel} wraps its longitudes.
    [src]'s {e seam} is its west edge where [src] also holds, a turn east,
    what lies west of the edge, as the whole west edge of a [src] that
    spans 360° is.

    Raises [Invalid_argument] if [p]'s last axis is not 2, [x] has fewer than
    two axes or its first two, from cell [(row, col)], are not a block of
    [src]'s cells, if a point whose containing cell lies in [src] reads a
    cell of [src] outside that block (checked by {!Nx.check}), or if [valid]
    holds other than one value per cell. *)

val nearest :
  ?row:int ->
  ?col:int ->
  ?valid:'d Nx.bool_t ->
  Grid.t ->
  'd Nx.float64_t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t * 'd Nx.bool_t
(** [nearest ?row ?col ?valid src p x] is [x]'s cell containing each point,
    in [x]'s dtype.
    A point on a border between two cells reads the one after it. This is the
    method for classes and other nominal values (ISO 19123, §C.2). *)

val bilinear :
  ?row:int ->
  ?col:int ->
  ?valid:'d Nx.bool_t ->
  Grid.t ->
  'd Nx.float64_t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t * 'd Nx.bool_t
(** [bilinear ?row ?col ?valid src p x] interpolates between the centres of
    the four cells around each point, weighted as ISO 19123 §C.6 gives in
    pixel coordinates, over the cells it keeps, its {e taps}, the weights
    divided by their sum ({!section-taps}). Its mask is {!nearest}'s. The
    containing cell has weight at least 1/4, so the value is a mean of
    valid values around the point, and at [src]'s edge it is the edge's
    values interpolated along it. On a [src] that spans 360° the taps'
    columns are read modulo [src.cols], so a point by its west or east edge
    reads both sides. The value is stored in [x]'s dtype
    ({!section-storage}), and rune differentiates it in [p]
    ({!section-gradients_bilinear}).

    Raises [Invalid_argument] as the samplers do, and if [x] holds booleans
    or complex numbers. *)

(** {2:valid The mask [valid]}

    [valid] holds one value per cell, in the shape of [x]'s first two axes
    followed by any axes of size 1, such as {!Band.decode}'s mask of an
    image of one channel that [Nx_io.load_tiff] reads, [[h; w; 1]].
    {!Nx.all} over the channels makes one from a mask of several. *)

(** {2:taps Bilinear's taps}

    {!bilinear} leaves out, by selection, a cell outside [src] or not
    [valid] and a cell of weight 0 whose value is NaN or infinite. It keeps
    a cell of weight 0 whose value is finite, which adds 0, whose sign can
    make a mean of 0 read as -0. A NaN of a valid cell of positive
    weight is data, and gives NaN. *)

(** {2:storage Storage}

    {!bilinear} and {!average} compute their value in float64 and store it in
    [x]'s dtype. A float rounds once to it. On integers the value v becomes
    ⌊v + ½⌋ in float64 arithmetic, GDAL's rule, on every sign (-2.5 gives
    -2); it lies between valid values, so it never saturates. Integers past
    2⁵³ in magnitude are rounded to float64 first. *)

(** {1:area Area rules}

    The area rules, {!average}, {!mode}, {!val-span}, {!area_window} and
    {!corners}, are provisional. Their names, [k] among them, and the
    fraction may change with a first consumer.

    An area rule reads an array [x] on a grid [src] over each cell's
    footprint in [src]'s pixel plane, whose corners are [dst]'s corners
    ({!corners}) as pixel points of [src]:

    {[
      let q = Warp.pixel src (Warp.corners ~at:Nx.Host.on src.crs dst) in
      let k = Warp.span src q in
      let bands, fraction = Warp.average ~k ~valid src q bands
    ]}

    So corner [(i, j)] of [q], [[dst.rows + 1; dst.cols + 1; 2]], is pixel
    point [(i, j)] of [dst]. Equal systems keep their lattice, and a corner
    outside [src.crs]'s domain is NaN. *)

(** {2:area_values Average and mode}

    Each rule weighs a valid cell of [src] by the area of its square in the
    footprint ({!section-weights}), counts only the cells the footprint
    {e reaches}, those whose area in it passes rounding's bound
    ({!section-reach}), takes the footprint's edges as straight
    ({!section-edges}) and reads [k × k] cells, which hold its {e span},
    the cells from the one holding its least row and column to the one
    holding its greatest ({!section-span}). A footprint {e reads cells}
    where its corners are finite and its span meets [src]. On a
    geographic [src], each rule moves a footprint by whole turns
    ({!section-samplers}) so that its easternmost corner lies in the 360°
    east of [src]'s west corner ({!section-antimeridian}), and on one that
    spans 360° it reads the columns modulo their number. Each edge of a
    footprint must span less than 180° of longitude
    ({!section-antimeridian}). Each rule gives its value and its fraction
    in [q]'s cells, [[r; c]], followed by [x]'s other axes, the fraction the
    same along them.

    An area rule raises [Invalid_argument] if [k] is below 1, if a footprint
    that reads cells spans more than [k] (checked by {!Nx.check}), if [q]'s
    shape is not [[r + 1; c + 1; 2]], if [x] has fewer than two axes or its
    first two, from cell [(row, col)], are not a block of [src]'s cells
    ({!section-windows}), if a footprint that reads cells reads one outside
    that block (checked by {!Nx.check}), or if [valid] holds other than one
    value per cell ({!section-valid}). On a geographic [src] it also raises
    if a footprint whose rows meet [src] has an edge of 180° of longitude
    or more once placed, as one around a pole has, or if a footprint that
    reads cells lies across the seam of a [src] that does not span 360°
    (each checked by {!Nx.check}, {!section-antimeridian}). *)

val average :
  k:int ->
  ?row:int ->
  ?col:int ->
  ?valid:'d Nx.bool_t ->
  Grid.t ->
  'd Nx.float64_t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t * 'd Nx.float64_t
(** [average ~k ?row ?col ?valid src q x] is, for each footprint of the
    corners [q], the mean of [x] over it, [Σ aⱼ xⱼ / Σ aⱼ] over the cells
    with data, and the footprint's fraction with data. A footprint on a
    geographic [src] is placed by turns, and one across the west edge of a
    [src] spanning 360° reads both sides ({!section-antimeridian}). [x] and
    [valid] may be the block of [src]'s cells from cell [(row, col)], 0 by
    default, such as {!area_window} gives, while [src] and [q] stay the
    whole grid's ({!section-windows}).

    A NaN of a valid cell it reaches is data, and gives NaN ({!section-reach}).
    The value is stored in [x]'s dtype ({!section-storage}), and rune
    differentiates it ({!section-gradients_average}). Two laws tie it to
    {!bilinear} and to the integral of [x] ({!section-average_laws}).

    Raises [Invalid_argument] as the area rules do, and if [x] holds
    booleans or complex numbers. *)

val mode :
  k:int ->
  ?row:int ->
  ?col:int ->
  ?valid:'d Nx.bool_t ->
  Grid.t ->
  'd Nx.float64_t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t * 'd Nx.float64_t
(** [mode ~k ?row ?col ?valid src q x] is, for each footprint of the corners
    [q], the class of [x] that covers the most of it, the class [v] of
    greatest [Σ_{j : xⱼ = v} aⱼ] over the cells with data, and {!average}'s
    fraction. A tie, sums equal to rounding, goes to the least class. A class
    that covers less of a footprint than another is missing from its value,
    so a program must mask a class, such as cloud, through [valid]. On a
    finer grid it is {!nearest}'s value but across a source cell's edge,
    where it is the larger share. Each of [x]'s other axes is taken alone. A
    footprint is placed as {!average}'s ({!section-antimeridian}), and [x]
    may be a block as for {!average} ({!section-windows}). It has no
    derivative ({!section-gradients}).

    Raises [Invalid_argument] as the area rules do, and if [x] does not hold
    integers. *)

(** {3:weights Weights and fraction}

    A source cell [j] weighs [aⱼ], the area of its square inside the footprint
    in [src]'s pixel plane, computed exactly. That is GDAL's measure, the
    fraction of a source cell the destination cell covers ([gdalwarp]'s
    [average]). The rule is renormalised over the cells that lie in [src]
    and are [valid], as ESMF's fractional-area normalisation divides by the
    destination fraction (ESMF Reference Manual, §25.2.8). Its second
    result is that fraction, the share of the footprint over such cells, in
    [\[0, 1\]]: 1 where the footprint lies in the data, below 1 at [src]'s
    edges and holes, and 0 where it meets no data, where the value is 0. A
    caller keeps the cells covered well enough by thresholding it.

    On a projected [src] the measure is the ground's area up to the square
    of the map's scale. On a geographic [src] it is an extent in degrees²,
    so a cell's part nearer a pole weighs as much as its part farther from
    it, where the earth's area is larger. ESMF and CDO weigh by the earth's
    area instead (ESMF Reference Manual, §25.2.7). A footprint of height Δφ
    radians at latitude φ then has its centroid in degrees² about
    Δφ tan φ / 12 of its height poleward of its centroid on the earth:
    1.5·10{^-3} of it for a 1° cell at 45°, and about a sixth for a cell
    in the row next to a pole, where the earth's area grows as the distance
    from it. The mean over the earth's area composes from the rule: average
    the two channels [[x·A; A]], [A] each source cell's area on the earth,
    on the sphere a fixed multiple of its width in longitude times
    sin φ₁ - sin φ₀ between its latitudes φ₀ and φ₁, and divide the first
    channel by the second. That is exact where the footprint holds whole
    source cells. *)

(** {3:reach Reach}

    Rounding leaves areas in cells a footprint does not meet, at most 24 u M W,
    with u = 2{^-53}, W the sum of the widths its edges span within the cell's
    columns and M the largest magnitude of its corners' coordinates from the
    cell's corner, at least 1. Only a valid cell a footprint reaches gives it
    data. As the area is about W times the footprint's thickness in the cell
    over 2, a footprint reaches a cell where it is thicker there than about
    48 u M, some fifty ulps of its coordinates. As {!bilinear} leaves out a
    cell of weight 0, {!average} leaves out a cell the footprint does not
    reach where its value is NaN or infinite. *)

(** {3:edges Edges}

    The edges are straight in [src]'s pixel plane. They are chords of the
    cell's curved image, as for any straight-edged footprint (ESMF Reference
    Manual, §25.2.9). On a geographic [src] an edge of length L at latitude
    φ lies up to about L² tan φ / (8R) from its chord, the most along a
    parallel, R the earth's radius: 18 µm for a 30 m edge at 45° and a
    millimetre at 89°. Between neighbouring zones of UTM it is micrometres
    on such cells. Near a pole that distance is L² / (8ρ), ρ the distance
    from the pole, an eighth of a cell as wide as it is far from the pole.
    MODIS's 926 m cells next to the pole are such cells of the sinusoid, on
    which the mean of a smooth field errs by up to 9% of the field's range
    over the cell. A cell of Albers' conic whose edge cuts across the arc
    that is the pole's image errs the same way. A program that needs such
    cells closer warps onto a finer [dst] and averages the fine cells of
    each cell, weighted by fraction × footprint area, which by the
    conservation law is the mean over their union
    ({!section-average_laws}). A footprint is taken to be a simple
    quadrilateral; one with a NaN or infinite corner has no data, however
    much of it covers [src]. So, on a source in a zone of UTM, whose domain
    reaches 40° of longitude from the zone's meridian
    ({!Crs.section-domains_tm}), a cell with a corner past that has
    fraction 0, as does every cell wider than 80° of longitude, such as a
    cell of a cap around a pole. jord has no polar system
    ({!Crs.section-systems}). *)

(** {3:span The span [k]}

    Clipped to [src], but for a 360° [src]'s columns
    ({!section-antimeridian}), a footprint's span must fit in the [k × k]
    cells from the one holding its least row and column, which the rule
    reads. A [k] too small raises and never gives a wrong value, and
    {!val-span} gives the least [k]. A footprint that reads no cell has no
    data whatever [k] is.

    A cell is half-open ({!Grid}), so a footprint whose greatest corner lies
    on a line of cell edges, as one on [src]'s lattice does, holds in its
    span the row or column of cells after that line, in which its area is
    0. The rule and {!area_window} read that row and column too: a
    footprint of 2 × 2 cells of [src] on its lattice spans 3 × 3, and its
    [k] is 3.

    [k] is a static size, so an area rule traces under rune's [jit] as the
    samplers do, and a program that compiles one rule for several sets of
    corners passes the largest of their spans. Eagerly, the rule holds
    about 300 bytes per candidate cell, [n k²] of them for [n] footprints,
    for an [x] of one band, and more for more bands: about 350 bytes at 8
    and 670 at 16. A program bounds the memory by warping windows of [dst]
    ({!Grid.window}). *)

val span : Grid.t -> 'd Nx.float64_t -> int
(** [span src q] is the least [k] the area rules take at the corners [q] of
    [src]: the most rows or columns of cells that the span of a footprint
    that reads cells holds, after its placement by turns
    ({!section-antimeridian}), and 1 where none reads cells. It reads one
    number back to the host, so it raises inside rune's [jit] or [vmap]; a
    program calls it before.

    Raises [Invalid_argument] if [q]'s shape is not [[r + 1; c + 1; 2]], and,
    on a geographic [src], as the area rules do for a footprint around a
    pole or across a seam of a [src] that does not span 360° (checked by
    {!Nx.check}, {!section-antimeridian}). *)

(** {3:antimeridian The antimeridian}

    An area rule moves each corner of a footprint by whole turns to within
    180° of longitude of its first corner, then the footprint by whole
    turns so that its easternmost corner lies more than 0° and at most 360°
    east of [src]'s west corner ({!pixel}). So a footprint across the
    antimeridian reads both sides of a [src] that spans it, and one across
    [src]'s west edge reads the part east of the edge, or, where [src]
    spans 360°, its first and last columns.

    A footprint's {e intended} edges are the arcs of longitude between its
    corners that the program meant, which the corners cannot show, and its
    {e placed} edges those between its placed corners, which the rule
    reads. Each intended edge must therefore span less than 180° of
    longitude. The placed edges are then the intended ones exactly where
    each of them spans less than 180° too, since a placed edge is an
    intended one moved by a whole number of turns. A rule raises where a
    placed edge spans 180° or more, as for a footprint around a pole, whose
    edges wind once around the circle, or one with a corner 180° or more
    from its first. A footprint whose rows miss [src] reads nothing and
    passes. A footprint with an intended edge of 180° or more and placed
    edges under 180° is read as the other side of the circle along that
    edge: the intended edge east from 100° W to 100° E, of 200°, is placed
    as the 160° east from 100° E to 100° W. Pixel points of a geographic
    [src] hold longitudes only modulo 360°, so the corners cannot tell the
    two apart, and no rule raises, which is the limit ESMF states for its
    own edges (ESMF Reference Manual, §25.2.9).

    {!corners} refuses a cell with a corner 180° or more from its first,
    but on the grids its doc exempts. An Albers cell around a pole winds,
    and the rules raise. A program that makes its corners otherwise must
    keep to the limit.

    A footprint across the west edge of a [src] that does not span 360°
    reads the side east of the edge alone. A rule raises where such a
    footprint reads cells and lies across [src]'s seam
    ({!section-samplers}), in the sense that its westernmost corner, moved
    a turn east, lies west of [src]'s easternmost corner. Such a [src] is
    wider than a turn, as a gridline-registered global source of 361 or
    3601 columns is, or narrower by less than the footprint's part west of
    the edge, as one whose step is printed short of 360° / cols is. The
    message names [src]'s turn and the remedy that fits it. Where the turn
    is a whole number n of columns along a row, a program cuts a [src]
    wider than a turn to its first n columns ({!Grid.window} and a slice
    of [x]), or pads one narrower than a turn to n columns with cells that
    are not [valid], and [src] then spans 360°. Where it is not, and the
    column step is a rounding of (360° / n, 0) for a whole n, or of
    (-360° / n, 0) for columns that run west, as 1° printed 0.9999999° is,
    the program states that step. Otherwise no whole number of [src]'s
    cells makes a turn, and no rule reads across its seam: the program cuts
    [src] to columns narrower than a turn that hold every footprint whole,
    or makes [dst] with no cell across [src]'s west edge. *)

(** {3:average_laws The average's laws}

    Two laws tie {!average} to the others. Where a footprint is the square
    of one source cell centred at a point, the value is {!bilinear}'s at
    the point, to rounding. Where the footprints tile the plane and cover
    [src], the sum of value × fraction × footprint area over [dst] is the
    sum of [x] over its valid cells, to rounding, so the integral is
    conserved. On [src] itself, whose corners are integers when its numbers
    are exact, the value is [x]. *)

(** {1:gradients Gradients}

    The values of {!bilinear} and {!average} are nx arithmetic in the points and
    corners, so rune differentiates them for a float [x]. An integer value is
    rounded and has none, so a caller decodes integers to floats first;
    {!mode}'s classes have none either. On a line of the source grid, a line
    through a row or column of centres for {!bilinear} and a line of cell edges
    for {!average}, the derivative is the right-hand one, since a cell's edge
    belongs to the cell after it, and a point on a line of centres to the span
    after it. So descent in the points or corners moves from a start on the
    lattice, as the centres and corners of a grid in the source's own system
    are. *)

(** {2:gradients_bilinear Bilinear's gradient}

    {!bilinear}'s gradient in [p] is the derivative of the mean over the
    cells kept at the point. A cell left out (outside the grid, not valid,
    or of weight 0 and not finite) adds 0 to it, so along that cell's
    axis the derivative is 0, as at the grid's edge. A NaN neighbour of
    weight 0 gives 0, as a hole does, and its NaN enters the value once the
    point moves toward it. The four cells come from ⌊row - ½⌋ and
    ⌊col - ½⌋, so on a line of centres the far cells have weight 0, and
    where they are kept the derivative is toward them. *)

(** {2:gradients_average The average's gradient}

    {!average} is differentiable in [q] and in a float [x], and its fraction
    in [q]. Each area is continuous in the corners, and continuously
    differentiable but on a line of cell edges, where a cell the footprint
    does not reach adds its area's derivative. The value is continuous but
    where the fraction falls to 0. There a footprint leaving the data reads
    the last sliver's mean, then 0, and the gradient grows as the inverse of
    the area with data as it gets there. *)

(** {1:windows Windows}

    A program that reads [src]'s array from a file in blocks, such as a
    cloud-optimised GeoTIFF's tiles, reads the cells a rule needs.
    {!window} gives the block a sampler reads, and {!area_window} the block
    the area rules read. A rule runs on its block with [~row ~col], the
    block's first cell, at the whole grid's [src] and points or corners, and
    gives the values and the mask or fraction it gives on the whole array,
    bit for bit.

    Each rule computes its weights and areas on [src] itself, so that only
    the index of a cell in the block depends on it, and its law holds for
    every [src], rotated, west-running or 360°. A point or footprint that
    reads a cell outside the block raises, so a block too small never gives
    a wrong value. On a [src] that spans 360° a block's
    columns run modulo [src.cols], as the rules read them: across [src]'s
    west edge it is the columns [col] to [src.cols - 1] followed by [0] on,
    which a program reads in two parts, since a file's window lies in the
    raster ([Nx_io.Tiff.window]), [read] here:

    {[
      let row, col, rows, cols = Warp.area_window src q in
      let part col cols = read ~row ~col ~rows ~cols in
      let x =
        if col + cols <= src.cols then part col cols
        else
          Nx.concatenate ~axis:1
            [ part col (src.cols - col); part 0 (col + cols - src.cols) ]
      in
      let bands, fraction = Warp.average ~k ~row ~col src q x
    ]} *)

(** The type for the samplers, as {!window} names them. *)
type rule = Nearest | Bilinear

val window : rule -> Grid.t -> 'd Nx.float64_t -> int * int * int * int
(** [window rule src p] is [(row, col, rows, cols)], the block of [src]'s cells
    that [rule] reads at the pixel points [p] of [src] whose containing cell
    lies in [src]. For [Nearest] these are the containing cells, and for
    [Bilinear] the four around each point, clipped to [src]. On a [src] that
    spans 360°, [Bilinear]'s columns run [col], [col + 1], … modulo
    [src.cols], with [col < src.cols], so [col + cols] may pass [src.cols],
    and it is the least such run, as {!area_window}'s. It is [(0, 0, 0, 0)]
    where no point has a cell. It reads five numbers back to the host, and
    for [Bilinear] on a [src] that spans 360°, [3n + 3] at [n] points.
    [rule]'s sampler runs on that block with [~row ~col]
    ({!section-windows}), and [Nx_io.Tiff.window] reads it from a TIFF file
    by the byte ranges it names, which the program fetches.

    Raises [Invalid_argument] if [p]'s last axis is not 2. *)

val area_window : Grid.t -> 'd Nx.float64_t -> int * int * int * int
(** [area_window src q] is [(row, col, rows, cols)], the least block of
    [src]'s cells that the area rules read at the corners [q]: the spans
    ({!section-span}) of the footprints that read cells, after their
    placement by turns. On a [src] that spans 360° its columns run [col],
    [col + 1], … modulo [src.cols], with [col < src.cols], so [col + cols]
    may pass [src.cols], and it is the least such run. It is [(0, 0, 0, 0)]
    where no footprint reads cells. It reads [5n] numbers back to the host
    for [n] footprints, so it raises inside rune's [jit] or [vmap]; a
    program calls it before.

    Raises [Invalid_argument] if [q]'s shape is not [[r + 1; c + 1; 2]], and,
    on a geographic [src], as the area rules do for a footprint around a
    pole or across a seam of a [src] that does not span 360° (checked by
    {!Nx.check}, {!section-antimeridian}). *)
