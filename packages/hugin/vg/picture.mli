(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pictures.

    A picture is an immutable description of a drawing. Its leaves paint:
    {!fill} paints the area of a path, {!stroke} its outline, {!glyphs} a glyph
    run and {!image} an image. Its nodes compose: {!group} paints pictures one
    over another, {!clip}, {!transform} and {!opacity} cut, map and fade a
    picture, {!stamp} paints one at many points, and {!tag} records what a
    picture draws for the tools that read the output.
    {!Hugin_vg.Renderable.v} puts a picture on a page of a physical size,
    and renderers draw renderables.

    Pictures follow the
    {{!Hugin_gg.section-conventions}geometry conventions}: the plane is
    y-down, and functions take the picture they build on last, so
    [Picture.group [ a; b ] |> Picture.opacity 0.5] chains. The unit of the
    plane is the point, 1/72 inch.

    {1:semantics Semantics}

    {2:compositing Colours and compositing}

    A picture [p] denotes a map [⟦p⟧] from the points of the plane to colours:
    [⟦p⟧ pt] is the colour [p] paints at [pt] on a transparent backdrop, and
    [∅], {!Hugin_gg.Color.transparent}, where it paints nothing. Colours
    are encoded sRGB with straight alpha, and they composite with the
    {e source-over} operator applied to the encoded components, as browsers and
    PDF viewers composite. [(c, a) over (c', a')] is [(c'', a'')], where [c],
    [c'] and [c''] stand for any one of the three components:
    - [a'' = a + a' (1 - a)];
    - [c'' = (a c + a' (1 - a) c') / a''], or [0] if [a'' = 0].

    Source-over is associative, so a picture [p] painted over a backdrop [b]
    shows [⟦p⟧ pt over b pt] at every point [pt].

    {2:areas Areas}

    A path [q] and a fill {!type-rule} [r] define an area [A r q] of the plane.
    The subpaths of [q] are taken as {!Hugin_gg.Path.fold} visits them,
    {{!Hugin_gg.Path.section-gaps}gaps} applied, each closed by a line back
    to its start if it is open. For a point [pt] that no subpath passes through,
    cast a ray from [pt] to infinity that crosses the subpaths transversally:
    - [`Nonzero]: [pt] is in [A `Nonzero q] iff the {e winding number} of [q]
      around [pt] is non-zero: the number of segments crossing the ray from its
      right to its left minus the number crossing it from its left to its right.
    - [`Even_odd]: [pt] is in [A `Even_odd q] iff the ray crosses the subpaths
      an odd number of times.

    The points of the subpaths bound the area and have no area themselves: a
    renderer may count them in or out.

    {2:outlines Outlines}

    A path [q] stroked with a style [s] covers its {e outline} [O s q], the
    union of the outlines of its subpaths as {!Hugin_gg.Path.fold} visits
    them. If [s] is dashed, each subpath of positive length is first cut into
    dashes: following the subpath from its start, and the pattern from
    [Stroke.dash_offset s], the lengths of the pattern alternately keep and drop
    the subpath, and each piece kept is an open subpath, a dash. The outline of
    a subpath is the union of:
    - the points at most [Stroke.width s /. 2.] away from a point of one of its
      segments, along the segment's normal there;
    - where two of its segments meet, and where a closed subpath meets its
      start, the join that {!Hugin_gg.Stroke.type-join} describes for
      [Stroke.join s];
    - at the two ends of an open subpath, the caps that
      {!Hugin_gg.Stroke.type-cap} describes for [Stroke.cap s].

    A subpath of zero length has no direction: dashed or not, it covers a disc
    of diameter [Stroke.width s] with [`Round] caps and nothing with the others.
    A dash of zero length takes the direction of the subpath it was cut from, so
    it covers a disc with [`Round] caps, a square of side [Stroke.width s] with
    [`Square] caps and nothing with [`Butt] caps.

    {2:denotation Denotation}

    Writing [inv m] for the inverse of an affine map [m]:
    - [⟦empty⟧ pt = ∅].
    - [⟦fill ~rule c q⟧ pt = c] if [pt] is in [A rule q], and [∅] otherwise.
    - [⟦stroke s c q⟧ pt = c] if [pt] is in [O s q], and [∅] otherwise.
    - [⟦glyphs c at r⟧] is [⟦group [ g0; …; gk ]⟧] for the glyphs [0] to [k] of
      the run [r], where [gi] fills with [c], under the nonzero rule, the
      outline of glyph [i] placed as {!glyphs} says.
    - [⟦image b px⟧ pt] is the colour of the pixel of [px] whose cell of [b]
      holds [pt], as {!image} says, if [pt] is in [b], and [∅] otherwise.
    - [⟦group [ p1; …; pn ]⟧ pt = ⟦pn⟧ pt over … over ⟦p1⟧ pt].
    - [⟦clip ~rule q p⟧ pt = ⟦p⟧ pt] if [pt] is in [A rule q], and [∅]
      otherwise.
    - [⟦transform m p⟧ pt = ⟦p⟧ (inv m pt)].
    - [⟦opacity a p⟧ pt] is [⟦p⟧ pt] with its alpha multiplied by [a].
    - [⟦stamp ~fills ~strokes ~scales xs ys p⟧] is the group of copies of [p]
      that {!stamp} describes. Its copies replace colours and pens leaf by leaf,
      so it depends on the structure of [p], not only on [⟦p⟧].
    - [⟦tag t p⟧ = ⟦p⟧].

    Each node composites its result as a whole. [opacity 0.5 (group [ a; b ])]
    fades [a] and [b] together, so [a] does not show through [b] where they
    overlap, whereas [group [ opacity 0.5 a; opacity 0.5 b ]] fades each, and
    [a] shows through.

    {2:rendering Rendering}

    A renderer approximates [⟦p⟧] on its device: a pixel of raster output shows
    the average of [⟦p⟧] over the pixel's square, to the accuracy the raster
    renderer states, and SVG and PDF viewers do the same at the resolution they
    display. *)

open Hugin_gg
open Hugin_font

(** {1:types Types} *)

type rule = [ `Nonzero | `Even_odd ]
(** The type for fill rules, which say what {{!section-areas}area} a path
    bounds. *)

(** The type for the data a tagged picture draws. *)
type rows =
  | Rows of int array
      (** [Rows a]: the picture draws the data rows [a]. If the picture is a
          {!stamp}, tagged directly, its instance [i] draws row [a.(i)];
          otherwise the picture draws the rows of [a] together. *)
  | Cells of { box : Box2.t; width : int; height : int }
      (** [Cells { box; width; height }]: the picture draws a grid of [width] by
          [height] equal cells dividing [box], in the picture's coordinates, the
          cell of index [k] being column [k mod width] of row [k / width],
          counted from the top left of [box]. A picture that reduces data names
          cells, as a heatmap drawn as one image does, and whoever drew it maps
          cells to data rows. *)

type tag = { id : Nx.Ptree.Path.t; rows : rows }
(** The type for tags. [id] names what drew the picture, in Hugin a node of a
    figure, and [rows] the data the picture draws. *)

(** The type for pictures. The cases are exposed so that renderers can match on
    them, and later versions may add cases. Each is built, validated and
    normalised by the function of the same name, as its documentation says, so a
    matched value holds the invariants stated there. A picture is a value: the
    arrays a match exposes must not be mutated, and the behaviour of a picture
    whose arrays were mutated is undefined. *)
type t = private
  | Empty  (** See {!empty}. *)
  | Fill of { rule : rule; color : Color.t; path : Path.t }  (** See {!fill}. *)
  | Stroke of { stroke : Stroke.t; color : Color.t; path : Path.t }
      (** See {!val-stroke}. *)
  | Glyphs of { color : Color.t; at : P2.t; run : Run.t }  (** See {!glyphs}. *)
  | Image of { box : Box2.t; pixels : Nx.uint8_t }  (** See {!image}. *)
  | Group of t list  (** See {!group}. *)
  | Clip of { rule : rule; path : Path.t; picture : t }  (** See {!clip}. *)
  | Transform of { m : Affine.t; picture : t }  (** See {!transform}. *)
  | Opacity of { opacity : float; picture : t }  (** See {!opacity}. *)
  | Stamp of {
      picture : t;
      xs : float array;
      ys : float array;
      scales : float array option;
      fills : Color.t array option;
      strokes : Color.t array option;
    }  (** See {!stamp}. *)
  | Tag of { tag : tag; picture : t }  (** See {!val-tag}. *)

(** {1:leaves Leaves} *)

val empty : t
(** [empty] paints nothing. *)

val fill : ?rule:rule -> Color.t -> Path.t -> t
(** [fill ~rule c q] paints the {{!section-areas}area} of [q] under [rule] with
    [c]. [rule] defaults to [`Nonzero]. Open subpaths are closed for filling. It
    is {!empty} if [q] is {!Hugin_gg.Path.empty}. *)

val stroke : Stroke.t -> Color.t -> Path.t -> t
(** [stroke s c q] paints the {{!section-outlines}outline} of [q] under [s] with
    [c]. Open subpaths get caps, and closed ones a join where they meet their
    start. It is {!empty} if [q] is {!Hugin_gg.Path.empty} or the width of
    [s] is [0.]. *)

val glyphs : Color.t -> P2.t -> Run.t -> t
(** [glyphs c at r] paints the glyphs of the run [r] with [c], the origin of [r]
    at [at]. Glyph [i] of [r] is its outline in the font of [r]
    ({!Hugin_font.Font.outline}) scaled by [Run.size r], with its origin at
    [at] moved by [(Run.x r i, Run.y r i)]. Glyphs are painted one by one, in
    order, so a translucent [c] composites twice where two glyphs overlap. SVG
    and PDF output keep the text of [r] with its glyphs, so that the text can be
    searched and copied. It is {!empty} if [r] has no glyph, if [Run.size r] is
    [0.] or if a coordinate of [at] is not finite. *)

val image : Box2.t -> Nx.uint8_t -> t
(** [image b px] paints the pixels of [px] over the box [b]. [px] has shape
    [[|h; w; c|]]: [h] rows of [w] pixels, from the top row and its left pixel,
    of [c] channels each, with [c] being [1] for grey, [3] for RGB or [4] for
    RGBA with straight alpha, and each value being an encoded component times
    [255]. The box is divided into [h] rows of [w] equal cells, and every point
    of the cell of row [i] and column [j] is painted with pixel [(i, j)], with
    no interpolation between pixels. A cell holds its top and left edges, and
    the cells of the last row and column their bottom and right edges too.

    The tensor is kept as given, on its device, and read when the picture is
    rendered. It is {!empty} if [px] has no pixel or [b] has width or height
    [0.].

    Raises [Invalid_argument] if [px] does not have shape [[|h; w; c|]] with [c]
    in [1], [3] or [4]. *)

(** {1:composing Composing} *)

val group : t list -> t
(** [group ps] paints the pictures of [ps] in order, each over those before it.
    The elements of [ps] that are {!empty} are dropped: [group []] is {!empty}
    and [group [ p ]] is [p]. *)

val clip : ?rule:rule -> Path.t -> t -> t
(** [clip ~rule q p] paints [p] only inside the {{!section-areas}area} of [q]
    under [rule], which defaults to [`Nonzero]. Clips nest by intersection. It
    is {!empty} if [p] is, or if [q] is {!Hugin_gg.Path.empty}. *)

val transform : Affine.t -> t -> t
(** [transform m p] paints [p] with the plane mapped through [m]: what [p]
    paints at [pt], [transform m p] paints at [P2.transform m pt]. Stroke
    widths, dashes, glyphs and images are mapped like everything else. It is
    {!empty} if [p] is, or if {!Hugin_gg.Affine.invert} gives [m] no
    inverse, since such a map paints no area. *)

val opacity : float -> t -> t
(** [opacity a p] paints [p] faded to the opacity [a]: [p] is painted on its
    own, over a transparent backdrop, and the result is composited with its
    alpha multiplied by [a]. This is {e group opacity}: where pictures of [p]
    overlap, the earlier do not show through the later. [opacity 1. p] is [p].
    [opacity 0. p] paints nothing and keeps the {!bounds} and tags of [p].

    Raises [Invalid_argument] if [a] is not in \[[0];[1]\]. *)

val stamp :
  ?fills:Color.t array ->
  ?strokes:Color.t array ->
  ?scales:float array ->
  float array ->
  float array ->
  t ->
  t
(** [stamp ~fills ~strokes ~scales xs ys p] paints a copy of [p], an
    {e instance}, at each point [(xs.(i), ys.(i))], in the order of [i]. It is
    [group [ q0; q1; … ]], one [qi] per index [i] of [xs], with
    [qi = transform Affine.(translate xs.(i) ys.(i) * scale si si) pi], where
    [si] is [scales.(i)] and [pi] is [p] with:
    - the colour of each {!fill} and {!glyphs} in it replaced by [fills.(i)], if
      [fills] is given;
    - the colour of each {!stroke} in it replaced by [strokes.(i)], if [strokes]
      is given;
    - the width, dash lengths and dash offset of each {!stroke} in it divided by
      [si], so that scaling an instance changes its geometry and keeps the pens
      of [p].

    [scales] defaults to ones. The colours of [p] that [fills] or [strokes]
    replace do not matter. The replacement reaches the leaves of stamps nested
    in [p], but not their own [fills] and [strokes], which replace the colours
    of their instances in turn. It acts on leaves, so it tells apart pictures
    that paint alike: [fill Color.transparent q] takes the colours of [fills]
    and {!empty} does not. An instance whose position or scale is not finite is
    skipped, and one of scale [0.] paints nothing.

    A stamp is one node however many instances it has, and renderers reuse the
    work of drawing [p] where they can, so it is the way to paint one marker at
    many points. The arrays are copied. It is {!empty} if [p] is or [xs] is
    empty.

    Raises [Invalid_argument] if [ys], [fills], [strokes] or [scales] differs in
    length from [xs], or if a finite scale is negative. *)

(** {1:tags Tags}

    A tag records what drew a picture and which data it draws, for the tools
    that read output, such as hit testing. It changes nothing painted. Raster
    and PDF output ignore tags, and SVG output writes them as [data-] attributes
    of the picture's element, so that a static SVG document carries its
    provenance. *)

val tag : tag -> t -> t
(** [tag t p] paints [p] tagged with [t]. The rows of [t] are copied. A tag
    gives the rows of a {!stamp} instance by instance only when it is put on the
    stamp itself, as [tag t (stamp xs ys p)]: a tag over a transform, a clip or
    an opacity of a stamp gives its rows together. It is {!empty} if [p] is.

    Raises [Invalid_argument] if:
    - [t.rows] is [Rows a], [p] is a {!stamp} and [a] differs in length from its
      positions;
    - [t.rows] is [Cells { width; height; _ }] and [width] or [height] is less
      than [1]. *)

(** {1:bounds Bounds} *)

val bounds : t -> Box2.t option
(** [bounds p] is a box containing every point [p] paints, or [None] if [p] has
    no extent: no leaf of [p] has a path that visits a point, a glyph with ink
    or an image. Colours and opacities do not count, so a transparent fill has
    bounds.

    The box of a leaf is the box of its path ({!Hugin_gg.Path.bounds}) for
    a fill, the box of its path grown on every side by the reach of its pen for
    a stroke, the box of its ink for a glyph run, and its box for an image. The
    reach of a pen is {!Hugin_gg.Stroke.reach}, divided by the scales of
    the stamps above it. The box of [p] is the union of the boxes of its leaves,
    in every instance of its stamps, each mapped with
    {!Hugin_gg.Box2.transform} through the composition of the transforms
    above it, which enlarges it under rotations, and cut by the boxes of the
    paths of the clips above it, mapped likewise. A clip whose box does not meet
    that of its picture has no extent.

    Raises [Invalid_argument] if a corner of the box is not finite, which takes
    coordinates near [max_float]. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal p p'] is [true] iff [p] and [p'] have the same structure: the same
    cases, with fields compared by the [equal] function of their type
    ({!Hugin_gg.Path.equal}, {!Hugin_font.Run.equal}, which compares
    fonts by their bytes, {!Hugin_gg.Color.equal}, [Nx.Ptree.Path.equal]
    for tag ids and so on), numbers by [Float.equal], and image tensors by shape
    and elements, which reads them, so that [equal] raises what reading a tensor
    raises. Equal pictures render to the same bytes in every renderer. One
    drawing has many structures: [group [ group [ a; b ]; c ]] paints what
    [group [ a; b; c ]] paints and is not equal to it. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf p] formats [p] for debugging and tests as nested s-expressions, one
    per case: paths as SVG path data, colours as {!Hugin_gg.Color.pp}
    prints them, images by the shape of their tensor and arrays in full. The
    output may change between releases. *)
