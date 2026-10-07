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
    scope. {!Frame} names the celestial frames and {!Direction} holds directions
    in them. {!Transform} maps pixels to the sky, {!Grid} is a shape of cells
    seen through a transform, {!Region} is a shape placed on a grid's world, and
    {!Observation} holds data on a grid and integrates them over a region. *)

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

(** {1:frames Frames and directions} *)

(** Celestial frames.

    A {e frame} is a celestial coordinate system: an orientation of three axes.
    Its {e orientation} [R_f] is the matrix taking ICRS components to frame
    [f]'s, [v_f = R_f · v_icrs]. A {e fixed} frame is one whose orientation from
    ICRS is constant.

    Each frame type has exactly one value, so two values of one type are one
    frame and a frame mismatch is a type error. A fixed frame's type is an
    instance of {!fixed}: {!matrix} and {!Direction.rotate} take any two fixed
    frames, and no other. The users of a frame write its type, as in
    [Frame.galactic Direction.t].

    Each fixed frame's orientation is its standard's own numbers, rounded once
    to float64. A conversion between two fixed frames is computed from their two
    orientations alone, so it depends only on the two frames, never on a path
    through others. *)
module Frame : sig
  type 'f t
  (** The type for celestial frames. Each frame type has exactly one value. *)

  type !'a fixed
  (** The type of a frame at a constant orientation from ICRS. *)

  type icrs = [ `Icrs ] fixed
  type fk5_j2000 = [ `Fk5_j2000 ] fixed
  type galactic = [ `Galactic ] fixed
  type ecliptic_j2000 = [ `Ecliptic_j2000 ] fixed
  type supergalactic = [ `Supergalactic ] fixed

  val icrs : icrs t
  (** [icrs] is the International Celestial Reference System. *)

  val fk5_j2000 : fk5_j2000 t
  (** [fk5_j2000] is FK5, mean equator and equinox of J2000.0, oriented at epoch
      J2000.0: the rotation from ICRS by the vector (−19.9, −9.1, +22.9) mas of
      Mignard and Froeschlé (2000), 31.7 mas in all. *)

  val galactic : galactic t
  (** [galactic] is Galactic coordinates as the Hipparcos Catalogue defines them
      on ICRS: the north Galactic pole at α = 192.85948°, δ = 27.12825°, and the
      north celestial pole at l = 122.93192°. *)

  val ecliptic_j2000 : ecliptic_j2000 t
  (** [ecliptic_j2000] is the IAU 2006 mean ecliptic and equinox of J2000.0, on
      ICRS: the obliquity ε₀ = 84381.406″ applied to the IAU 2006 frame bias at
      J2000.0. *)

  val supergalactic : supergalactic t
  (** [supergalactic] is supergalactic coordinates, defined on {!galactic}: the
      north supergalactic pole at Galactic (47.37°, +6.32°), the origin at l =
      137.37°. *)

  val matrix : 'a fixed t -> 'b fixed t -> (float, Nx.float64_elt) Nx.t
  (** [matrix a b] is [M], [[3; 3]] on the host, with [v_b = M · v_a]: for
      quantities that are not directions, such as a velocity ([M v]) or a
      covariance ([M Σ Mᵀ]).

      [M] is [R_b · R_aᵀ]. [matrix a a] is the identity, [matrix icrs b] is
      [R_b] and [matrix a icrs] is [R_aᵀ]; another pair's entries are each
      computed with one {!Float.fma} chain, so [matrix b a] is the exact
      transpose of [matrix a b]. *)

  val name : 'f t -> string
  (** [name f] is [f]'s canonical name: ["icrs"], ["fk5_j2000"], ["galactic"],
      ["ecliptic_j2000"], ["supergalactic"]. It is the frame's identity in keys,
      text and files. *)

  val pp : Format.formatter -> 'f t -> unit
  (** [pp] prints {!name}. *)

  val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'f t -> 'f t
  (** [walk c f] reports {!name} as one case at [c]'s path and is [f]. A frame
      holds no tensors. *)
end

(** Directions in a frame.

    A direction is a frame and float64 vectors [[...; 3]], each read as the ray
    it spans. A unit norm cannot be a property of the type: [jvp] tangents,
    [grad] cotangents and {!Nx.Ptree.map} build directions of any norm. So the
    contract has two halves.

    - {e Constructors return unit vectors.} {!lonlat} and {!of_xyz} give
      [|xyz| = 1] within 3 ulp, and {!rotate} preserves the norm within 3 ulp.
    - {e Readers read the ray.} {!lon}, {!lat}, {!separation} and
      {!position_angle} read every finite row as the direction it points in. A
      positive multiple of a row reads the same up to the rounding of the
      multiple, and bit for bit when the multiple is a power of two that keeps
      the components normal. A NaN component gives NaN, and an infinite one
      raises [Invalid_argument]. {!rotate} and {!xyz} are linear.

    So a sum of two directions read as a direction is their chordal midpoint.

    {b Derivatives.} Every function has its analytic derivative wherever it is
    differentiable. Where it has none it gives a stated value with derivative 0,
    never NaN:
    - {!lon} at a pole ([x = y = 0]): 0.
    - {!lat} at a pole: ±π/2; at a zero row: 0.
    - {!separation} where [a × (b − a) = 0], that is coincident or antipodal
      rows or a zero row: π if [a · b < 0], else 0.
    - {!position_angle} at the same points: 0. With respect to [a] at a pole,
      its derivative is 0.

    At [lat]'s poles and [separation]'s 0 and π the function behaves like [|x|],
    and 0 is a subgradient; the other points are discontinuities. Near a pole
    the derivative of [lon] grows as 1/ρ, ρ the distance from the axis.

    Every angle in or out is a float64 quantity. Indexing and reshaping go
    through {!ptree}:

    {[
    let near =
      Nx.Ptree.map (Direction.ptree ())
        (fun _ x -> Nx.take ~axis:0 ~indices x)
        stars
    ]} *)
module Direction : sig
  type 'f t
  (** The type for directions in frame ['f]: float64 vectors [[...; 3]], each
      read as the ray it spans. *)

  val lonlat :
    'f Frame.t ->
    lon:(float, Nx.float64_elt) Nx.t Quantity.t ->
    lat:(float, Nx.float64_elt) Nx.t Quantity.t ->
    'f t
  (** [lonlat f ~lon ~lat] is the unit vectors
      [(cos b cos l, cos b sin l, sin b)] at longitude [l] and latitude [b],
      their batch axes broadcast. Every finite angle names a direction: latitude
      100° at longitude 0° is latitude 80° at longitude 180°. A NaN angle gives
      a NaN row.

      Raises [Invalid_argument] if [lon] or [lat] is not an angle, or where an
      angle is infinite, as in ["Direction.lonlat: lat at [3] is infinite"]. *)

  val of_xyz : 'f Frame.t -> (float, Nx.float64_elt) Nx.t -> 'f t
  (** [of_xyz f v] is [v / |v|] along [v]'s last axis. A NaN row stays NaN. Its
      derivative is the projection onto the tangent plane.

      Raises [Invalid_argument] if the last axis does not have 3 elements, or
      where a vector is infinite or zero, as in
      ["Direction.of_xyz: the vector at [17] is zero and names no direction"].
  *)

  val frame : 'f t -> 'f Frame.t
  (** [frame d] is [d]'s frame. *)

  val xyz : 'f t -> (float, Nx.float64_elt) Nx.t
  (** [xyz d] is [d]'s vectors, [[...; 3]]: unit vectors for values the
      constructors build. *)

  val lon : 'f t -> (float, Nx.float64_elt) Nx.t Quantity.t
  (** [lon d] is the longitude in \[+0, 2π) radians, [atan2 y x]. It is 0 at a
      pole, so a function that needs a meridian at a pole takes the meridian of
      longitude 0. *)

  val lat : 'f t -> (float, Nx.float64_elt) Nx.t Quantity.t
  (** [lat d] is the latitude in \[−π/2, π/2\] radians, [atan2 z ρ] with
      [ρ = √(x² + y²)]. *)

  val rotate : ('b Frame.fixed as 'g) Frame.t -> 'a Frame.fixed t -> 'g t
  (** [rotate g d] is [d] in frame [g]: [Frame.matrix (frame d) g] times each
      vector. It is linear in [d], within [1.5 · 2⁻⁵² · |v|] of the exact
      product per component, and rounds the same eagerly and compiled. *)

  val separation : 'f t -> 'f t -> (float, Nx.float64_elt) Nx.t Quantity.t
  (** [separation a b] is the angle between the rays, in \[0, π\] radians.
      Leading axes broadcast. It is [atan2 |a × d| (a · b)] with [d = b − a],
      which keeps relative accuracy at any separation: 3e-16 at 1 mas. *)

  val position_angle : 'f t -> 'f t -> (float, Nx.float64_elt) Nx.t Quantity.t
  (** [position_angle a b] is [b]'s bearing from [a], east of north, in \[+0,
      2π) radians. Leading axes broadcast. At a pole, north is the limit along
      longitude 0: at the north pole it points toward longitude 180° and east
      toward 90°. *)

  val ptree : unit -> 'f t Nx.Ptree.t
  (** [ptree ()] is the structure of directions: the frame as {!Frame.walk}
      reports it, then the vectors as a tensor of a fixed type, at the root
      path. {!Nx.Ptree.cast} keeps them float64, and {!Nx.Ptree.map}, placement
      and rune's transformations treat them as any tensor. *)
end

(** {1:grids Transforms, grids and observations} *)

(** Maps between planes and the sky.

    A transform is a list of {e stages}, each one family's parameters and a
    sense, forward or inverse. Its endpoint types say what it maps:
    [Nx.float32_elt plane] points, [Frame.icrs Direction.t] directions.
    Composing appends stages and inverting reverses them; no stage is ever
    fused, so what a file stated is what a writer prints.

    A FITS celestial image stored [[NAXIS2; NAXIS1]] reads as

    {[
    axes [| 1; 0 |] ~origin:1
    >> shift crpix >> linear cd
    >> celestial Tan Frame.icrs Nx.float64 ~pv ~native ~crval ~lonpole ~latpole
    ]}

    {!axes} holds raven's pixel convention (0-based centres in the data tensor's
    axis order) and FITS's (1-based, [NAXIS1] first) in one stage.

    {b Units.} Each planar family reads its input in the unit it states: {!axes}
    with an origin and {!linear} and {!scale} in {!Unit.one}, {!shift} in its
    offset's unit, {!celestial} in an angle. A list in the wrong order raises
    [Invalid_argument] where a stage reads its input.

    {b Domains.} A stage is a bijection on its domain. TAN's projection, from
    directions to the plane, needs native θ > 0; ARC's deprojection needs a
    distance from the reference point of at most 180°. {!apply} at a finite
    point outside a domain raises [Invalid_argument] through {!Nx.check}, naming
    the stage, the point and its distance, as in
    ["Transform.apply: point 12 is 93.1 deg from the TAN reference (110.8375,
     -73.4537) deg, outside the projection's domain; Transform.covers gives the
     mask"]. NaN maps to NaN. {!covers} gives the mask for callers that expect
    points outside.

    {b Precision.} Planar stages hold their parameters at the plane's dtype and
    compute at it. Celestial stages hold float64 parameters and compute in
    float64: a float32 plane point is cast up exactly. A float32 planar map
    resolves positions to about 2⁻²⁴ of their distance from its origin.

    Batch axes of points and parameters broadcast. *)
module Transform : sig
  type ('a, 'b) t
  (** The type for maps from points of type ['a] to points of type ['b]. *)

  type 'e plane = (float, 'e) Nx.t Quantity.t
  (** The type for points [[...; n]] in a plane. Pixel coordinates are in
      {!Unit.one}, intermediate and tangent-plane coordinates in an angle. *)

  (** The type for projections, by their FITS codes. *)
  type code =
    | Tan  (** Gnomonic: great circles map to lines. *)
    | Arc
        (** Zenithal equidistant: distance from the reference is preserved. *)

  (** {1:constructors Constructors} *)

  val id : ('a, 'a) t
  (** [id] maps every point to itself. *)

  val axes : int array -> origin:int -> ('e plane, 'e plane) t
  (** [axes p ~origin] maps [x] to [x.(p.(k)) + origin] in component [k]. With
      [origin = 0] it permutes plane points of any unit; otherwise they are in
      {!Unit.one}.

      Raises [Invalid_argument] if [p] is not a permutation of [0], ...,
      [n - 1]. *)

  val shift : 'e plane -> ('e plane, 'e plane) t
  (** [shift r] maps [x] to [x - r]: the plane whose origin is [r]. Points are
      read in [r]'s unit.

      Raises [Invalid_argument] if [r] is a scalar. *)

  val linear : 'e plane -> ('e plane, 'e plane) t
  (** [linear m] maps [x], in {!Unit.one}, to [m · x] in [m]'s unit: CD, or PC
      in {!Unit.one}. [m] is [[...; n; n]].

      Raises [Invalid_argument] if [m] is not square on its last two axes. *)

  val scale : 'e plane -> ('e plane, 'e plane) t
  (** [scale d] maps [x], in {!Unit.one}, to [x.(k) · d.(k)] in component [k],
      in [d]'s unit: CDELT.

      Raises [Invalid_argument] if [d] is a scalar. *)

  val celestial :
    ?stated:int array ->
    code ->
    'f Frame.t ->
    (float, 'e) Nx.dtype ->
    pv:(float, Nx.float64_elt) Nx.t ->
    native:Nx.float64_elt plane ->
    crval:Nx.float64_elt plane ->
    lonpole:Nx.float64_elt plane ->
    latpole:Nx.float64_elt plane ->
    ('e plane, 'f Direction.t) t
  (** [celestial code f dtype ~pv ~native ~crval ~lonpole ~latpole] is the
      projection [code] with parameters [pv], [[...; m]], followed by the
      rotation taking the native point [native] (φ₀, θ₀) to [crval] (lon, lat)
      in [f], with the celestial pole at native longitude [lonpole]; [latpole]
      picks between two poles where FITS defines two. Angles are [[...; 2]] for
      [native] and [crval] and scalars for the poles, in any angle unit, as the
      file gives them. Plane points are angles at [dtype]; the stage computes in
      float64, and its inverse returns points at [dtype]. [stated] lists the
      indices of [pv] the file gave, all by default.

      TAN and ARC take no parameter ([m = 0]) and native θ₀ = 90°: another θ₀
      raises [Invalid_argument] where the stage is applied.

      Raises [Invalid_argument] if [pv]'s last axis is not [m], if [stated] is
      not ascending indices below [m], or if an angle is not [[...; 2]] or not
      in an angle unit. *)

  val about : 'f Direction.t -> ('f Direction.t, Nx.float64_elt plane) t
  (** [about c] maps directions to angular offsets about [c], in radians, x east
      and y north: the inverse of [celestial Arc] at [c] with LONPOLE 180°. The
      offset's norm is {!Direction.separation} from [c], and its bearing from +y
      toward +x is {!Direction.position_angle}. A disc of radius ρ about the
      origin is exactly the cap of angular radius ρ about [c]. At a pole the
      meridian is longitude 0, as {!Direction.lon} takes it. *)

  val gnomonic : 'f Direction.t -> ('f Direction.t, Nx.float64_elt plane) t
  (** [gnomonic c] is the same with [Tan]: great circles map to lines. *)

  (** {1:ops Operations} *)

  val ( >> ) : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t
  (** [t >> u] applies [t], then [u]. *)

  val inverse : ('a, 'b) t -> ('b, 'a) t
  (** [inverse t] reverses [t]'s stages and flips each one's sense.
      [inverse (inverse t)] has [t]'s stages. *)

  val apply : ('a, 'b) t -> 'a -> 'b
  (** [apply t x] maps [x] through each stage. Directions are read as the rays
      they span and returned as unit vectors.

      Raises [Invalid_argument] where a point is outside a stage's domain, or
      where a stage reads points of the wrong unit or size. *)

  val covers : ('a, 'b) t -> 'a -> Nx.bool_t
  (** [covers t x] is where [apply t x] is defined: [false] outside a stage's
      domain and at NaN. It has [x]'s batch shape, and is a scalar [true] for a
      transform with no stage. It does not raise at any point. *)

  val pp : Format.formatter -> ('a, 'b) t -> unit
  (** [pp] formats a transform's stages. *)

  val ptree : unit -> ('a, 'b) t Nx.Ptree.t
  (** [ptree ()] is the structure of transforms: the number of stages, then each
      stage at its index, reporting its family, its sense and its static data
      (permutation, origin, code, frame, dtype, stated terms) and walking its
      parameters. No float is static. *)
end

(** Cells seen through a transform.

    A {e cell} is one sample's footprint, a pixel's square. A pixel grid is a
    shape and a transform from its pixel coordinates to its {e world}, ['w]:
    directions on the sky or points in a plane. Cell [(i, j)] has its centre at
    pixel coordinates [(i, j)], 0-based in the data tensor's axis order, and its
    corners at [(i ± ½, j ± ½)].

    A {e window} is a block of a grid's cells of static shape at a traced start,
    [[...; 2]], possibly batched: the grid of a stamp, or of a million stamps. A
    window keeps its {e base}, the whole image's shape, and its transform over
    the whole image's pixel coordinates, so its cells are the base's cells.
    Cells beyond the base exist in a window, mapped through the same transform.
    The {e border} of a window is its outer ring of cells.

    A cell's {e measure} is its solid angle on the sky, or its area in a plane:
    the measure of the quadrilateral of its corners mapped to the world, with
    great-circle edges on the sky. Measures are computed in float64. *)
module Grid : sig
  type ('w, 'e) t
  (** The type for grids of cells whose world is ['w], with measures at dtype
      ['e]. *)

  val cell : Unit.t
  (** [cell] is the unscoped symbol counting cells. Data per cell carry
      [cell⁻¹]: electrons per second in a cell are
      [Unit.(symbol "electron" / second / Grid.cell)]. It never converts to an
      angle; the grid's measure does that. *)

  val pixels :
    shape:int array ->
    (float, 'e) Nx.dtype ->
    ('e Transform.plane, 'w) Transform.t ->
    ('w, 'e) t
  (** [pixels ~shape dtype t] is the image of [shape] cells,
      [[|rows; columns|]], seen through [t]: the whole image, its own base, at
      [dtype].

      Raises [Invalid_argument] unless [shape] is two non-negative sizes. *)

  (** The type for a grid's kind. *)
  type ('w, 'e) kind =
    | Pixels : {
        base : int array;  (** The whole image's shape. *)
        shape : int array;  (** The window's shape. *)
        start : Nx.int64_t;  (** The window's first cell, [[...; 2]]. *)
        transform : ('e Transform.plane, 'w) Transform.t;
            (** From the whole image's pixel coordinates. *)
      }
        -> ('w, 'e) kind

  val kind : ('w, 'e) t -> ('w, 'e) kind
  (** [kind g] is [g]'s kind and parts. *)

  val shape : ('w, 'e) t -> int array
  (** [shape g] is the shape of [g]'s cells: the data's last axes. *)

  val centres : ('w, 'e) t -> 'w
  (** [centres g] is each cell's centre in the world, [batch @ shape g] points.
      A window's centres are its base's at the same cells, bit for bit.

      Raises [Invalid_argument] as {!Transform.apply} does. *)

  val corners : ('w, 'e) t -> 'w
  (** [corners g] is each cell's four corners in the world,
      [batch @ shape g @ [4]] points, counter-clockwise in the pixel plane. *)

  val measure : ('w, 'e) t -> (float, 'e) Nx.t Quantity.t
  (** [measure g] is each cell's measure, [batch @ shape g]: in steradians on
      the sky, by Van Oosterom and Strackee's formula on corner differences,
      within about 1e-9 of a 0.031″ TAN pixel's solid angle; in the plane's unit
      squared in a plane, by the shoelace formula. *)

  val window : start:Nx.int64_t -> shape:int array -> ('w, 'e) t -> ('w, 'e) t
  (** [window ~start ~shape g] is the block of [shape] cells of [g] from
      [start], [[...; 2]] along [g]'s own axes. Its batch axes are [start]'s.

      Raises [Invalid_argument] unless [shape] is two non-negative sizes and
      [start]'s last axis is 2. *)

  val around : 'w -> shape:int array -> ('w, 'e) t -> ('w, 'e) t
  (** [around x ~shape g] is the [shape] block of [g]'s base centred on the cell
      holding each point [x], located through the inverse transform, with the
      extra cell of an even size on the high side. Its batch axes are [x]'s. It
      carries no derivative and does not raise at any point: a point the inverse
      does not cover gets a window wholly beyond the base, and every start is
      clamped to \[[-shape], [base]\].

      Raises [Invalid_argument] unless [shape] is two non-negative sizes. *)

  val ptree : unit -> ('w, 'e) t Nx.Ptree.t
  (** [ptree ()] is the structure of grids: the kind, the base and window shapes
      and the dtype as static data, then the start and the transform. *)
end

(** Planar shapes placed on a grid's world.

    A region is a planar shape centred on the origin of a plane, and a
    {e placement}: the transform from a world into that plane. On the sky,
    [Transform.about c] is the plane of angular offsets about [c], where a disc
    of radius ρ is exactly the cap of angular radius ρ; [about c >> shift v] is
    a cap to |v|²/6 relative. In pixels, [Transform.shift p] places a shape at
    [p]. The shape's centre is a parameter of the placement, so a fitted
    offset's gradient flows through the placement into the overlap.

    A cell's {e weight} is the fraction of its area inside the shape, both taken
    in the shape's plane: the quadrilateral of the cell's corners mapped there,
    edges straight. The overlap is exact, computed with no data-dependent branch
    and differentiable in the shape's centre and size. A cell weighs exactly 1
    when its corners are all inside a disc, and exactly 0 when its nearest point
    is at least the radius from the centre and it does not hold the centre. An
    annulus weighs the outer disc's weight minus the inner's, with the disc's
    predicates on each.

    A zero size weighs 0 with zero gradient. Regions have no set algebra: the
    fraction of a cell inside a union is no function of the two fractions. *)
module Region : sig
  type ('w, 'e) t
  (** The type for regions placed from world ['w], sized at dtype ['e]. *)

  val circle :
    ('w, 'p Transform.plane) Transform.t ->
    radius:(float, 'e) Nx.t Quantity.t ->
    ('w, 'e) t
  (** [circle p ~radius] is the disc of [radius] about the origin of [p]'s
      plane, in a unit of that plane. *)

  val annulus :
    ('w, 'p Transform.plane) Transform.t ->
    inner:(float, 'e) Nx.t Quantity.t ->
    outer:(float, 'e) Nx.t Quantity.t ->
    ('w, 'e) t
  (** [annulus p ~inner ~outer] is the ring between the discs of radii [inner]
      and [outer] about the origin of [p]'s plane. *)

  val weights : ('w, 'e) t -> ('w, 'e) Grid.t -> (float, 'e) Nx.t
  (** [weights r g] is each cell's weight in \[0, 1\], [batch @ Grid.shape g],
      the region's and the grid's batch axes broadcast. Corners map in float64
      and their offsets in the shape's plane are cast to ['e].

      Raises [Invalid_argument] through {!Nx.check} where a size is negative or
      NaN, an annulus's [inner] is above its [outer], or a cell maps to a
      quadrilateral that is not convex or not oriented as its window's first
      cell. *)

  val ptree : unit -> ('w, 'e) t Nx.Ptree.t
  (** [ptree ()] is the structure of regions: the shape's kind, its sizes and
      its placement. *)
end

(** Measured samples on a grid.

    An observation is data of shape [batch @ Grid.shape g] on a grid [g], with
    optional variance, validity and area.

    {b Validity.} [valid] is true where a sample was measured. Data and variance
    under an invalid sample are replaced by 0 wherever one arises, so a NaN or a
    sentinel there reaches no result under any derivative. Validity only
    narrows: {!restrict} conjoins, and only {!v} sets a mask anew. Windows
    always have a mask, false beyond the data. Masks carry no gradient.

    {b Area.} A pipeline that defined its surface brightness against a nominal
    area, such as JWST's [PIXAR_SR], or a per-pixel area map, states it as
    [area]. The grid's {!Grid.measure} stays each cell's own.

    {b The unit rule.} Data whose unit has {!Grid.cell}⁻¹ are values per cell,
    and a cell counts as one cell. Every other unit is a field over the cells,
    and a cell counts by the observation's [area] where it has one, its measure
    otherwise. *)
module Observation : sig
  type ('w, 'e) t
  (** The type for observations on a grid whose world is ['w], at dtype ['e]. *)

  val v :
    ?variance:(float, 'e) Nx.t Quantity.t ->
    ?valid:Nx.bit_t ->
    ?area:(float, 'e) Nx.t Quantity.t ->
    ('w, 'e) Grid.t ->
    (float, 'e) Nx.t Quantity.t ->
    ('w, 'e) t
  (** [v ?variance ?valid ?area g data] is [data], [batch @ Grid.shape g], on
      [g]. [variance] is in the data's unit squared. [valid] is true where a
      sample was measured. [area] is the measure each cell's data are per, when
      it is not the cell's own: one value per cell, its last axes [g]'s shape,
      or one for all cells.

      Raises [Invalid_argument] if [data] does not end with [g]'s shape, the
      variance's unit is not the data's squared, or [area] is not a measure of
      [g]'s cells or has another shape. *)

  val grid : ('w, 'e) t -> ('w, 'e) Grid.t
  (** [grid o] is [o]'s grid. *)

  val data : ('w, 'e) t -> (float, 'e) Nx.t Quantity.t
  (** [data o] is [o]'s data, 0 where invalid. *)

  val variance : ('w, 'e) t -> (float, 'e) Nx.t Quantity.t option
  (** [variance o] is [o]'s variance, 0 where invalid. *)

  val valid : ('w, 'e) t -> Nx.bit_t option
  (** [valid o] is where [o]'s samples were measured, if known. *)

  val area : ('w, 'e) t -> (float, 'e) Nx.t Quantity.t option
  (** [area o] is the measure [o]'s data are per, when not the cells' own. *)

  val restrict : Nx.bit_t -> ('w, 'e) t -> ('w, 'e) t
  (** [restrict m o] keeps a sample valid where it was valid and [m] holds. *)

  val window : start:Nx.int64_t -> shape:int array -> ('w, 'e) t -> ('w, 'e) t
  (** [window ~start ~shape o] is [o] on [Grid.window ~start ~shape (grid o)]:
      data, variance, validity and a per-cell area sliced with the grid, invalid
      beyond [o]'s cells. *)

  val around : 'w -> shape:int array -> ('w, 'e) t -> ('w, 'e) t
  (** [around x ~shape o] is [o] on [Grid.around x ~shape (grid o)], sliced as
      {!window} slices. *)

  type 'e integral = {
    value : (float, 'e) Nx.t Quantity.t;
        (** Σ data × w × a over valid cells, [a] a cell's area for a field and
            one cell for values per cell. *)
    variance : (float, 'e) Nx.t Quantity.t option;
        (** Σ variance × w² × a², the variance of [value] for independent
            samples, when the observation has a variance. *)
    area : (float, 'e) Nx.t Quantity.t;
        (** Σ w × a over valid cells: [value / area] is the mean. *)
    coverage : (float, 'e) Nx.t;
        (** The valid cells' share of the region's overlap with the window, in
            the region's plane: below 1 where the region crosses an edge of the
            image or invalid samples, 0 where it does not overlap the window. *)
  }
  (** The type for integrals of an observation over a region, each of the
      region's and window's batch axes broadcast. *)

  val integrate : ('w, 'e) Region.t -> ('w, 'e) t -> 'e integral
  (** [integrate r o] weighs [o]'s cells with {!Region.weights} and sums each
      term once in double-word arithmetic at ['e], with [w] the weight.

      Raises [Invalid_argument] through {!Nx.check} where a cell of the window's
      border is inside the base, the base continues beyond it, and the region is
      not exactly outside it: the window clipped the region, as in
      ["Observation.integrate: the region reaches the border of its 104x104
       window at sample (0, 51); take a larger ~shape"]. Raises as
      {!Region.weights} does. *)

  val ptree : unit -> ('w, 'e) t Nx.Ptree.t
  (** [ptree ()] is the structure of observations: the grid, then the data,
      variance, validity and area, each present or not. *)
end
