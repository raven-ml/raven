(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Figures.

    A figure is an immutable description of a chart. Its {e marks} draw rows of
    data: each binds {e channels} to its {e roles}, such as [x], [y] and [fill].
    A channel lifts an nx tensor by what it means ({!num} for quantities, {!cat}
    for coded categories, {!strings}), reads the index along an axis as a
    category ({!dim}) or a quantity ({!index}), or holds a constant ({!const}),
    and reads a {e scale} found by name. Composition ({!layer}, {!grid},
    {!share}) decides which channels share a scale, and axes and legends follow
    from the scales each part of a figure holds. A mark is already a figure:

    {[
    open Hugin_next

    let () = save "loss.svg" (line ~y:(num losses) ())
    ]}

    A figure renders in three pure stages, each able to reuse its previous
    output: {!resolve} fits the scales to the data, {!layout} measures text,
    chooses ticks and places panels on a grid that keeps data areas aligned, and
    {!draw} reduces large marks where their data lives and paints a tagged
    picture. {!render} runs the three, {!save} writes PNG, SVG or PDF, and {!pp}
    displays a figure in a Quill notebook. {!Mark.v} defines marks as the
    built-in ones are defined.

    {1:model The figure model}

    A figure denotes a pair: the values it contributes to its scales, and a
    function from the fitted scales to what a page shows. A mark contributes the
    rows of its channels and draws them in the panels its facet channels select.
    {!layer} is the union of its children over their arrangements, {!grid} and
    {!span} arrange, and the wrappers {!title}, {!coord}, {!name} and {!share}
    each change one aspect of the figure they wrap. A combinator's meaning is a
    function of its arguments' meanings: there is no precedence between
    settings, no child is dropped, and two settings that cannot both hold raise
    in {!resolve}.

    {1:conventions Conventions}

    - {b Units.} Lengths are in points, 1/72 inch, or in {e em}, multiples of
      the theme's base size ({!Theme.size}). Pixels appear only through a
      {e density}, in device pixels per point, which {!draw} reduces data for
      and raster output is drawn at.
    - {b Planes.} Pictures, boxes and projected points are in the y-down plane
      of {{!Hugin_next_gg.section-conventions}[hugin.next.gg]}, in points.
      Positions within a panel are {e normalised}: [0.] at its left or bottom
      edge and [1.] at its right or top edge, whatever its coordinate system,
      which maps them into the panel's box ({!Coord}).
    - {b Argument order.} Functions that build on a figure take it last, so
      [f |> title t |> coord c] chains.
    - {b Equality.} [equal] functions compare structure, floats by
      [Float.equal]. The leaves of figures, tensors and functions, compare
      physically ({!equal}), and images by their elements ({!Picture.equal}).
    - {b Printing.} The [pp] functions of stage outputs format them for
      debugging and baselines; their output may change between releases. {!pp}
      on figures displays them.
    - {b Purity.} There is no global state: a stage reads its arguments and
      nothing else, no clock, no current figure, no hash table order, and equal
      arguments give equal outputs, byte for byte once rendered. Functions given
      to a figure ({!map_range}, {!bind}, the draw and swatch functions of
      {!Mark.v}) must not read mutable state; results are unspecified otherwise.
    - {b Errors.} Programming errors raise [Invalid_argument] as early as they
      can be detected: shapes, dtypes, axes and labels when a lift or a mark is
      made; composition errors and conflicting specifications in {!resolve};
      sizes too small and figure text that no face draws in {!layout}. Messages
      name the nodes at fault by their {{!section-ids}ids}. Problems with data
      values are {e warnings}: nothing is substituted for the values at fault,
      the stage goes on, and each stage output lists its warnings and those of
      the stages before it ({!Drawing.warnings}). {!resolve} and {!draw} also
      raise the [Invalid_argument] that reading a tensor raises, for a traced
      tensor or one whose storage a compiled call consumed, with its message
      prefixed by the id of the mark.

    {1:data Data and rows}

    The channels of a mark broadcast together under nx's broadcasting rule into
    the mark's {e shape}. Each element of that shape is a {e datum}, which the
    mark draws as one {e row}: one tensor of five loss curves of [T] steps each
    is 5 × [T] rows of one line mark. A constant, an {!index} and a {!dim}
    without a mask take no part in broadcasting, and a mark whose channels are
    all constants has the shape [[||]] and one row. The facet channels of a mark
    ({!section-facets}) put each row in one panel.

    {2:missing Missing values}

    A value is {e missing} if it is NaN or infinite, if the [valid] mask of its
    channel is [false] there, or if its scale cannot place it
    ({{!Scale.section-missing}missing values}), such as a value that is not
    positive on a log scale or a code outside its labels. A row with a missing
    value is {e dropped}: it is drawn nowhere and no scale is fitted to it.
    Colours are the exception: a missing colour paints its scale's unknown
    colour ({!Scale.unknown}), by default no paint, and leaves its row drawn and
    fitted. Lines break where rows are dropped.

    Positions are clipped to their domains, and ink is not. A position outside
    its scale's domain is drawn and cut at the domain's edge: a line or a
    segment ends there, an extent stops there, and a symbol or a text whose
    point lies outside is not drawn. What a row inside the domain draws, its
    symbol, the half of its line's width beyond the edge, its text, is drawn
    whole, over the panel's edges. Domains do not grow to make room for ink, so
    axes keep their ends.

    {1:ids Ids}

    Every node of a figure has an {e id}, a path that derives from the figure as
    written. The root's is [Nx.Ptree.Path.root]. A child of {!layer} and a cell
    of {!grid} add [Index i], [i] counting in reading order from [0]; {!name}
    [s] replaces its node's index by [Field s]; a {!layer} whose children
    include a grid is a grid of the shape their arrangements broadcast to, and
    its cell [k] in reading order adds [Field "cell"], then [Index k], so a 1 ×
    3 grid layered with a 2 × 1 grid makes six cells; a facet panel adds
    [Field "panel"], then [Field c] for the name ({!Scale.type-categories}) of
    its [fy] category, then for that of its [fx] category, those it has; a
    generated axis adds [Field "axis"], then [Field] of the name of its scale; a
    generated legend adds [Field "legend"], then [Field] of the name of its
    scale, then [Field] of its kind, ["num"] or ["cat"], since a scale is
    identified by its name and kind. Wrappers, {!span} and {!bind} add nothing.
    {!name} takes none of ["axis"], ["legend"], ["panel"] and ["cell"], so a
    generated node never has the id of a written one. Since ids depend only on
    the structure of the figure, a figure rebuilt by the same code has the same
    ids, and {!name} pins a subtree whose position varies. Errors, warnings,
    {!View.zoom}, {!Resolved.scale} and the tags of drawn pictures
    ({!Picture.tag}) name nodes by id. *)

(** {1:lower Lower libraries}

    The modules of the lower libraries that figures and marks are described
    with. *)

module P2 = Hugin_next_gg.P2
(** Points. *)

module Box2 = Hugin_next_gg.Box2
(** Boxes. *)

module Affine = Hugin_next_gg.Affine
(** Affine maps. *)

module Path = Hugin_next_gg.Path
(** Paths. *)

module Stroke = Hugin_next_gg.Stroke
(** Stroke styles. *)

module Color = Hugin_next_gg.Color
(** Colours. *)

module Font = Hugin_next_font.Font
(** Fonts. *)

module Text = Hugin_next_text.Text
(** Rich text and its layout. *)

module Picture = Hugin_next_vg.Picture
(** Pictures. *)

module Renderable = Hugin_next_vg.Renderable
(** Pictures on a page. *)

module Locale = Hugin_next_kit.Locale
(** Locales. *)

module Scale = Hugin_next_kit.Scale
(** Scale specifications and fitted scales. *)

module Scheme = Hugin_next_kit.Scheme
(** Colour schemes. *)

module Symbol = Hugin_next_kit.Symbol
(** Marker symbols. *)

module Curve = Hugin_next_kit.Curve
(** Curves through points. *)

(** {1:figures Figures} *)

type t
(** The type for figures. *)

type id = Nx.Ptree.Path.t
(** The type for the {{!section-ids}ids} of a figure's nodes. *)

type warning = id * string
(** The type for warnings: the node whose data a problem concerns, and a message
    for users. *)

val pp_warning : Format.formatter -> warning -> unit
(** [pp_warning ppf w] formats [w] for users: the id, then the message. *)

val equal : t -> t -> bool
(** [equal f f'] is [true] iff [f] and [f'] are made by the same combinators
    from equal arguments: tensors and functions compared physically, string
    arrays element by element, and the values of the lower libraries by their
    [equal]. A figure rebuilt from the same tensors by the same code is equal to
    the first unless it holds a new function, as a {!map_range} or {!bind} of a
    fresh closure does. [layer [ layer [ a; b ]; c ]] draws what
    [layer [ a; b; c ]] draws and is not equal to it. *)

(** {1:channels Channels} *)

type ('d, 'r) channel
(** The type for channels for roles of range ['r]: data whose values, of the
    domain type ['d], a scale normalises and the role maps into ['r], or a
    constant of ['r]. The domain types are those of the kinds of scales
    ({!Scale.type-kind}): [float] for quantities and [string] for categories. A
    channel bound with [let] serves roles of the one range its first use fixes;
    a channel for another range is lifted again. *)

(** {2:roles Roles}

    A role is a mark's slot for a channel ({!Role}). Its range fixes what a
    constant means, and it reads the scale named after it unless noted:

    {t
      | Role             | Range      | Scale          | A constant is                  |
      |------------------|------------|----------------|--------------------------------|
      | [x], [y]         | [float]    | ["x"], ["y"]   | a normalised position          |
      | [x2], [y2]       | [float]    | ["x"], ["y"]   | a normalised position          |
      | [fill], [stroke] | [Color.t]  | ["color"]      | a colour                       |
      | [opacity]        | [float]    | ["opacity"]    | an opacity in \[[0];[1]\]      |
      | [size]           | [float]    | ["size"]       | an area in square points       |
      | [width]          | [float]    | ["width"]      | a line width in points         |
      | [symbol]         | [Symbol.t] | ["symbol"]     | a symbol                       |
      | [text]           | [Text.t]   | none           | a text                         |
      | [fx], [fy]       | [string]   | ["fx"], ["fy"] | the panel the mark is drawn in |
    }

    A role takes channels of any kind, except that [size] takes quantities and
    [symbol], [fx] and [fy] take categories, as their types say: neither
    [~fx:(num x)] nor [num ~scale:(Scale.band ()) x] type-checks. The ranges
    that normalised values map into are those the scale's specification sets, or
    else the theme's ({!Theme.section-ranges}).

    The [text] role reads no scale, so a channel given to it has neither a
    [scale] nor a [title] ({!Mark.v}). A category is written as its label, or as
    the text its scale shows it by. A quantity is written in plain notation in
    the theme's locale, each value of the mark with the decimals that the value
    needing the most needs to reproduce itself in its source dtype
    ({!Hugin_next_kit.Number.decimals}): a float32 [0.9234] reads [0.9234], a
    column of numbers shares its decimals, and an integer count reads without a
    decimal separator. *)

val num :
  ?scale:float Scale.t ->
  ?valid:Nx.bool_t ->
  ?title:Text.t ->
  ('a, 'b) Nx.t ->
  (float, 'r) channel
(** [num ~scale ~valid ~title x] is the quantities of [x], one datum per
    element, where:
    - [scale] specifies the scale the channel reads ({!section-names}). Defaults
      to an unnamed specification that sets nothing.
    - [valid] keeps the elements where it is [true] and makes the others
      {{!section-missing}missing}. It broadcasts to the shape of [x] without
      growing it. Defaults to keeping every element.
    - [title] titles the guide of the scale ({!section-guides}). Defaults to
      none.

    Elements are read as floats, so an integer beyond 2{^ 53} is rounded.
    Nothing is read when the channel is made: [x] stays where it lives until
    {!resolve} summarises it and {!draw} reduces and reads it.

    Raises [Invalid_argument] if [x] has a complex or boolean dtype, or if
    [valid] does not broadcast to the shape of [x] without growing it. *)

val cat :
  ?scale:string Scale.t ->
  ?valid:Nx.bool_t ->
  ?title:Text.t ->
  ?labels:string array ->
  ('a, 'b) Nx.t ->
  (string, 'r) channel
(** [cat ~scale ~valid ~title ~labels codes] is the categories of the integer
    [codes], one datum per element, with [scale], [valid] and [title] as for
    {!num}, and:
    - with [labels], {e labelled} categories: code [i] is the category named
      [labels.(i)], in the order of [labels], and the channel contributes every
      label to its scale's domain, whether or not a code uses it, so that
      colours stay put across filtered data. A code outside
      \[[0];[Array.length labels - 1]\] is missing, with a warning.
    - without [labels], {e indexed} categories: code [i] is the category
      identified by the integer [i] ({!Scale.type-categories}), shown by the
      text another channel of its scale gives it ({!dim}) or else in decimal. A
      code beyond the range of [int] is missing, with a warning.

    [labels] is copied.

    Raises [Invalid_argument] if [codes] does not have an integer dtype, if
    [labels] holds a label twice, or if [valid] does not broadcast to the shape
    of [codes] without growing it. *)

val strings :
  ?scale:string Scale.t -> ?title:Text.t -> string array -> (string, 'r) channel
(** [strings ~scale ~title a] is the labelled categories named by the elements
    of [a], one datum per element, with [scale] and [title] as for {!num}: its
    shape is [[|Array.length a|]], and it contributes its distinct strings to
    its scale's domain in order of first appearance. [a] is copied. *)

val dim :
  ?scale:string Scale.t ->
  ?valid:Nx.bool_t ->
  ?title:Text.t ->
  ?labels:string array ->
  int ->
  (string, 'r) channel
(** [dim ~scale ~valid ~title ~labels k] is the index along axis [k] of the
    shape of the mark it is bound in, as indexed categories: the datum
    [(i0, …, in)] is the category identified by [ik]. A negative [k] counts from
    the last axis, [-1] being the last. With [labels], category [i] is shown as
    [labels.(i)], and labels may repeat since the integer identifies the
    category; without, it is shown by the text another channel of its scale
    gives it, or else in decimal. The channel contributes every index of its
    axis to its scale's domain, whether or not a row uses it. [scale] and
    [title] are as for {!num}. Unlike the masks of other lifts, [valid] joins
    the mark's shape: it broadcasts with the mark's channels, so
    [rect ~fx:(dim ~valid:m 0) ()] has the shape of [m]. [labels] is copied.

    The mark it is bound in raises [Invalid_argument] when it is made if its
    shape has no axis [k] or if [labels] differs in length from that axis. *)

val index : ?scale:float Scale.t -> ?title:Text.t -> int -> (float, 'r) channel
(** [index ~scale ~title k] is the index along axis [k] of the shape of the mark
    it is bound in, as quantities: the datum [(i0, …, in)] has the value
    [float ik]. A negative [k] counts from the last axis, [-1] being the last.
    [scale] and [title] are as for {!num}. It holds no tensor and takes no part
    in broadcasting: [line ~x:(index ~title:(Text.v "step") (-1)) ~y:(num l) ()]
    titles the x that {!line} gives by default.

    The mark it is bound in raises [Invalid_argument] when it is made if its
    shape has no axis [k]. *)

val const : 'r -> ('d, 'r) channel
(** [const v] is [v] for every row, in the range of the role it is bound to
    ({!section-roles}): [~x:(const 0.5)] is the middle of the panel, and
    [~fx:(const "a")] draws its mark in the panel of the category ["a"] only
    ({!section-facets}). A constant reads no scale, contributes to no domain and
    yields no guide. *)

val map_range : ('r -> 'r) -> ('d, 'r) channel -> ('d, 'r) channel
(** [map_range f c] is [c] with [f] applied to each of its values after the role
    maps it into the range, and to the value of a constant:
    [~fill:(map_range Color.contrast recall)] paints text that reads on the
    colour [recall] gives its cell. Missing values are not given to [f]: they
    paint their scale's unknown colour. [c] still contributes to its scale's
    domain, and a scale read only through [map_range] yields no legend. *)

(** {1:marks Marks}

    A mark is a figure that draws its rows in each panel its facet channels
    select ({!section-facets}). The built-in marks below are made as {!Mark.v}
    makes marks, from bindings and a draw function, and take their channels as
    labelled arguments named after their roles ({!section-roles}). {!dot},
    {!rect}, {!rule} and {!contour} use only what {!Mark} offers. {!line},
    {!text} and {!image} also keep parameters in roles a user cannot bind, a
    curve, text offsets and pixels, and {!image} gathers its pixels at the
    density {!draw} is given. Every mark takes the facet channels [fx] and [fy],
    and every mark but {!image} takes [opacity], [1.] by default.

    A mark without a colour channel paints with the theme's accent
    ({!Theme.accent}), except {!rule} and {!text}, which paint with its ink
    ({!Theme.ink}). Default lengths are the theme's ({!Theme.section-lengths}).

    A mark raises [Invalid_argument] when it is made if its channels do not
    broadcast together, or as {!dim}, {!index} and {!Mark.v} state. *)

val dot :
  ?fill:('f, Color.t) channel ->
  ?stroke:('s, Color.t) channel ->
  ?opacity:('o, float) channel ->
  ?size:(float, float) channel ->
  ?symbol:(string, Symbol.t) channel ->
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  x:('x, float) channel ->
  y:('y, float) channel ->
  unit ->
  t
(** [dot ~x ~y ()] draws a symbol at the point of each row, in row order, filled
    with [fill] and outlined with [stroke] at the theme's outline width. Without
    either it is filled with the accent; with [stroke] alone it is outlined
    only. [size] is the symbol's area in square points ({!Symbol}), the theme's
    dot size by default. [symbol] defaults to {!Symbol.circle}; categories read
    through it take the symbols their scale sets ({!Scale.symbols}), or else
    those of {!Symbol.filled} if the dot is filled and of {!Symbol.stroked} if
    it is outlined only.

    A dot mark with more than 20,000 rows in a panel, or more rows than the
    panel has device pixels, is drawn there as one image of its rows painted in
    row order ({!Mark.raster}). *)

val line :
  ?x:('x, float) channel ->
  ?stroke:('s, Color.t) channel ->
  ?fill:('f, Color.t) channel ->
  ?width:('w, float) channel ->
  ?opacity:('o, float) channel ->
  ?curve:Curve.t ->
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  y:('y, float) channel ->
  unit ->
  t
(** [line ~y ()] draws a curve through the points of each {e series} of its
    rows, in order along the last axis of the mark's shape. A series
    ({!Mark.series}) holds the rows that share their index along every other
    axis and their category in every channel other than a position that reads a
    band scale: [~stroke:(cat c)] splits the rows by [c], and a categorical [x]
    does not. Where:
    - [x] defaults to [index (-1)] ({!index}).
    - [curve] says how the curve passes through the points, {!Curve.linear} by
      default. A curve breaks where a row is dropped ({!Curve.section-runs}).
    - Without [fill], each series is stroked with [stroke] at [width], the
      theme's line width by default.
    - With [fill], each series is closed and filled under the even-odd rule, and
      outlined only if [stroke] is given.

    A series takes its colours, width and opacity from its first row that is not
    dropped, with a warning if one of them varies along it.

    A line with more than four rows per device-pixel column is reduced
    ({!Mark.m4}). *)

val rect :
  ?x:('x, float) channel ->
  ?x2:('x, float) channel ->
  ?y:('y, float) channel ->
  ?y2:('y, float) channel ->
  ?fill:('f, Color.t) channel ->
  ?stroke:('s, Color.t) channel ->
  ?opacity:('o, float) channel ->
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  unit ->
  t
(** [rect ()] draws, for each row, the rectangle that its extents along x and y
    span ({!Mark.extent}). Its extent along x is:
    - with [x] and [x2], the hull of the two, an end on a band scale standing
      for its band;
    - with [x] alone on a band scale, its band; if [y] reads a continuous scale,
      the rect is a {e bar}, and [rect] implies padding [0.2] on the band scale
      ({!Theme.section-ranges}), so that bars stand apart, while a cell of a
      heatmap, whose [y] reads a band scale too, implies none;
    - with [x] alone on a continuous scale, a {e length}: from zero to [x], zero
      being clamped into the domain, so that on a log scale a length starts at
      the domain's lower end. [rect] implies [zero] on the scale of a length
      ({!section-merging});
    - without [x], the whole panel.

    Likewise along y. So [rect] draws bars, cells, heatmaps, spans and, with
    [stroke] and no [fill], frames. It is filled with [fill], or with the accent
    unless [stroke] alone is given, and outlined with [stroke] at the theme's
    outline width.

    A rect whose [x] and [y] read band scales without padding, with neither
    [x2], [y2] nor [stroke], and no two of whose rows in a panel share a cell,
    is drawn under an affine projection as one image per panel whose pixels are
    its cells ({!Mark.cells}). Otherwise each row is drawn as a rectangle.

    Raises [Invalid_argument] if [x2] is given without [x] or [y2] without [y].
*)

val rule :
  ?x:('x, float) channel ->
  ?x2:('x, float) channel ->
  ?y:('y, float) channel ->
  ?y2:('y, float) channel ->
  ?stroke:('s, Color.t) channel ->
  ?width:('w, float) channel ->
  ?opacity:('o, float) channel ->
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  unit ->
  t
(** [rule ()] draws a straight segment for each row:
    - with [x] and no [x2], a vertical segment at [x] over the extent along y
      that {!rect} gives [y] and [y2], so [rule ~x ()] crosses the panel and
      [rule ~x ~y ()] is a stem from zero to [y];
    - otherwise, with [y] and no [y2], a horizontal segment at [y] over the
      extent along x that {!rect} gives [x] and [x2];
    - otherwise, with [x], [x2], [y] and [y2], the segment from [(x, y)] to
      [(x2, y2)].

    A position on a band scale is the centre of its band. [rule] implies [zero]
    on the scale of a length, as {!rect} does, so stems are proportional to
    their values. The segment is stroked with [stroke], the ink by default, at
    [width], the theme's line width by default.

    Raises [Invalid_argument] if [x2] is given without [x] or [y2] without [y],
    or if the channels given match none of these cases. *)

val text :
  ?fill:('f, Color.t) channel ->
  ?opacity:('o, float) channel ->
  ?dx:float ->
  ?dy:float ->
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  x:('x, float) channel ->
  y:('y, float) channel ->
  text:('t, Text.t) channel ->
  unit ->
  t
(** [text ~x ~y ~text ()] sets each row's text in the theme's faces at its base
    size, centred on the row's point moved [dx] points right and [dy] points up,
    both [0.] by default ({!Mark.text}). Characters without a colour of their
    own ({!Text.color}) are painted with [fill], the ink by default. Data are
    written as the [text] role writes them ({!section-roles}). A character that
    no face of the theme has is drawn as its face's [.notdef] glyph, with a
    warning. *)

val image :
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  ('a, 'b) Nx.t ->
  t
(** [image px] draws the images of [px]. A rank-2 [px] is one grey image of
    shape [[|h; w|]]. Otherwise the trailing axes of [px] are [[|h; w; c|]], [c]
    being [1] for grey, [3] for RGB and [4] for RGBA with straight alpha, and
    its leading axes are datum axes, one image per datum, which [fx] and [fy] (a
    {!dim} of a leading axis) put in panels of their own; the images of one
    panel are drawn in row order. Values range over [0] to [255] for [uint8] and
    over \[[0];[1]\], clamped, for a floating-point dtype; a pixel with a NaN
    component paints nothing.

    Each image covers \[[0];[w]\] on the scale ["x"] and \[[0];[h]\] on the
    scale ["y"], its row [0] at the top, so that marks layered over it draw in
    pixel coordinates: [image] implies [nice] off and no axis on both scales,
    and [reverse] on ["y"] ({!Mark.bind}). Pixels are square: [image] implies
    [Coord.cartesian ~aspect:1. ()] ({!Mark.v}).

    Raises [Invalid_argument] if [px] has a rank below [2], a rank above [2]
    with a last axis other than [1], [3] or [4], or a dtype other than [uint8]
    and the floating-point dtypes. *)

val contour :
  ?x:('x, float) channel ->
  ?y:('y, float) channel ->
  ?opacity:('o, float) channel ->
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  fill:(float, Color.t) channel ->
  unit ->
  t
(** [contour ~x ~y ~fill ()] draws the filled contours of fields sampled on
    grids. The last two axes of the mark's shape are a grid's rows and columns,
    and its other axes are datum axes, one field each. The sample of row [i] and
    column [j] lies at the [x] of column [j] and the [y] of row [i], so [x] must
    not vary along the second-to-last axis nor [y] along the last: their tensors
    have size [1] there, or their {!dim} or {!index} names another axis.
    [~x:(num alphas)] with [alphas] of shape [[|m|]] and [~y:(num betas)] with
    [betas] of shape [[|n; 1|]] place a field of shape [[|n; m|]]. [x] defaults
    to [index (-1)] and [y] to [index (-2)] ({!index}). A field's extent is its
    grid, as an image's is its pixels: [contour] implies [nice] off on the
    continuous scales of [x] and [y] ({!Mark.bind}), so the field fills its
    panel, and marks layered over it fit its domain unless they reach outside
    it.

    The levels are [0.] and [1.], the normalised ends of the fill scale's
    domain, and the normalised ticks {!layout} froze for it ({!Mark.ticks}), in
    increasing order without repeats, so the bands are the intervals of its
    colour bar. Between two consecutive levels [lo] and [hi], the isoband
    ({!Hugin_next_gg_kit.Field2.isoband}) of the field's normalised values is
    filled with [f ((lo +. hi) /. 2.)], where [Mark.range rows Role.fill] is
    [Some f] ({!Mark.range}). Missing samples cut holes. A field whose columns'
    x or rows' y are not strictly monotone once normalised draws nothing, with a
    warning.

    Raises [Invalid_argument] if the mark's shape has fewer than two axes, if
    [fill] is a constant, if [x] or [y] can vary along the axis it must not vary
    along, or if [fx] or [fy] can vary along either axis of the grid, which
    would put one field in several panels. *)

(** {1:scales Scales}

    {2:names Names}

    A channel reads the scale its specification names ({!Scale.name}), or else
    its role's default scale ({!section-roles}). A specification without a name
    specifies the default scale of the role it is given to, and the names of
    roles are reserved: [num ~scale:(Scale.log ()) loss] given to [~y] makes
    every channel on ["y"] in its scope logarithmic, and
    [Scale.linear ~name:"y" ()] is that scale. A name a user gives denotes one
    scale, whose kind ({!Scale.type-kind}) its readers fix, so a sweep can
    colour loss curves in one panel and place final accuracies in another by the
    same rates. The default scale of a role other than a position or a facet is
    identified by its name and its kind, so categorical and quantitative
    ["color"] channels read two scales, each with its legend. In a panel, the
    channels on [x] read one scale, and likewise [y], [fx] and [fy]: {!resolve}
    raises [Invalid_argument], naming both marks, if they read two, as a
    [dot ~x:(num ~scale:lr rates)] layered with a mark on the default ["x"]
    does.

    {2:scopes Scopes}

    A scale is fitted once per name, kind and {e scope}. The names ["x"], ["y"],
    ["fx"] and ["fy"] are scoped by the innermost grid cell that holds the
    channel, a layer's broadcast cells included ({!section-ids}), or by the
    whole figure outside any grid, and every other name by the whole figure. So
    the children of a {!layer} and the panels of a facet share every scale, and
    the cells of a grid keep their own positions and facets and share the rest:
    a figure has one colour legend however it is arranged. {!share} regroups one
    name at one node.

    A node {e lies in} the cells it draws in: a grid in each of its cells, a
    node that a layer repeats over a grid in each cell it is repeated in, and
    any other node in one. The scope of a name that {e holds} a node is the one
    that a channel of that name would read in every cell the node lies in, if
    there is one. So a cell names its own scopes, and a grid, or a reference
    line layered over it, names the scales its cells share, such as the colour,
    and no position or facet scale unless a {!share} gives its cells one. A mark
    made [`Independent] per panel holds its own scale of that name if it draws
    in one panel, and a facet panel holds the one that a mark has there if no
    other mark has one.

    {2:merging Merging}

    A property of a scale's specification ({!Scale.type-property}) is
    {e explicit} if a lift's [scale] sets it, {e implied} if a mark implies it
    ({!Mark.bind}), and {e defaulted} otherwise; explicit beats implied, which
    beats defaulted. A length implies [zero] ({!rect}, {!rule}), the band
    position of a bar implies padding ({!rect}, {!Theme.section-ranges}), and a
    band scale read by [y] implies [reverse], so that its first category is at
    the top. Whether a scale has a guide is a property of the same levels: an
    explicit {!axis} or {!legend} beats what marks imply ({!Mark.bind}), which
    beats the defaults of {!section-guides}. Once every {!share} is known,
    {!resolve} merges the explicit specifications of each scale's channels, then
    their implied ones ({!Scale.merge}, {!Scale.imply}), and raises
    [Invalid_argument], naming the scale and both marks, on two values of one
    property at the same level, readers of two kinds under a user's, position or
    facet name, labelled and indexed categories on one scale, or two texts for
    one indexed category.

    {2:fitting Fitting}

    {!resolve} fits every scale with {!Scale.fit}, categorical scales first. A
    continuous domain is the hull of its channels' values over the rows that are
    not dropped, widened by [zero] and rounded outward by [nice]; {!layout}
    chooses ticks inside it and never widens it. A categorical domain holds
    labelled or indexed categories ({!Scale.type-categories}):
    - Its labelled categories are the union, in the order the figure is written,
      before {!layer} broadcasts, of every label of each channel [cat ~labels],
      used or not, and of the distinct strings of each {!strings} channel in
      order of first appearance, those of dropped rows included.
    - Its indexed categories are the increasing union of every index of the axis
      of each {!dim} channel and of the codes in the rows of each {!cat} channel
      without labels. A category is shown by the labels of the [dim] channels
      that give it a text, or else in decimal.

    An explicit domain replaces the fitted one, and a zoom ({!View.zoom})
    replaces both.

    {1:guides Guides}

    The coordinate system of a panel draws an axis for each of its position
    scales, unless the scope holds an explicit {!axis} for it, and the axes of
    the facet scales are the panels' headers, titled beside them once in each
    cell. An axis shared by the panels of a column, for x, or of a row, for y,
    is labelled on the outer panel only. Every scale that a role other than a
    position or a facet reads, other than only through {!map_range}, yields one
    legend at its scope unless the scope holds an explicit {!legend} for it: a
    colour bar for a continuous colour scale, and otherwise one entry per guide
    value, drawn by the swatch ({!Mark.v}) of every mark that reads the scale. A
    guide is titled by the distinct titles of the channels that read its scale,
    in the order the figure is written, separated by commas, and is untitled if
    they have none. These rules are defaults: marks can imply that a scale has a
    guide or none ({!Mark.bind}), and an explicit {!axis} or {!legend} decides
    over both ({!section-merging}).

    A scale read by several roles has a guide for each: a scale named by a user
    and read by [x] in one panel and by [fill] in another has an axis there and
    a legend. Its ticks are one set, which every guide of the scale shows: those
    of a facet scale and of a categorical scale with a legend are its
    categories, and the others are chosen so that their labels overlap on none
    of its guides ({!layout}).

    {1:facets Facets}

    The facet channels [fx] and [fy] read band scales whose categories are
    panels: a mark with facet channels draws each row in the panel of its
    categories, one column per category of its scope's ["fx"] scale and one row
    per category of its ["fy"] scale, in domain order from the left and from the
    top. A mark without facet channels draws all its rows in every panel of its
    layer, so a reference line layered over facets is drawn in each. A facet
    constant draws its mark in the panel of its category only, and nowhere, with
    a warning, if there is no such panel; its rows still fit the other scales
    its channels read. [Scale.band ~wrap] on the ["fx"] scale wraps its panels
    into rows of that many panels; a [wrap] with an ["fy"] scale in the same
    scope raises [Invalid_argument] in {!resolve}. Facet panels share every
    scale of their scope, so 144 attention panels have one colour bar. *)

(** {1:coordinates Coordinate systems} *)

(** Coordinate systems.

    A coordinate system projects the normalised positions of a panel into its
    box, inverts that projection where it can, and generates the panel's axes.
    Marks never see the form of the projection: they give normalised geometry to
    {!Mark.points}, {!Mark.project} and {!Coord.point}. {!coord} gives a
    figure's panels a coordinate system. *)
module Coord : sig
  (** {1:systems Coordinate systems} *)

  type t
  (** The type for coordinate systems. *)

  val cartesian : ?aspect:float -> unit -> t
  (** [cartesian ~aspect ()] maps normalised positions affinely onto the box of
      the panel, x from its left edge to its right and y from its bottom edge to
      its top, and draws x axes below the panel and y axes to its left. Without
      [aspect], the panel takes the box the layout gives it. With [aspect], one
      unit of y is [aspect] times as long on the page as one unit of x, a unit
      being one of the transform of a continuous scale
      ({!Scale.section-continuous}), such as a power of the base on a log scale,
      and one step of a band scale ({!Scale.length}); a scale whose domain spans
      no unit, or more than the floats hold, counts as spanning one. So [1.]
      gives equal data units on both axes, and square cells to two band scales
      of equal paddings. The aspect sizes the panel's grid track, so the spines
      of its neighbours stay aligned.

      Raises [Invalid_argument] if [aspect] is not finite and positive. *)

  val equal : t -> t -> bool
  (** [equal c c'] is [true] iff [c] and [c'] are the same coordinate system
      with equal parameters. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf c] formats [c] for debugging. *)

  (** {1:projections Projections} *)

  type projection
  (** The type for projections: a coordinate system fitted to a panel's box. *)

  val point : projection -> float -> float -> P2.t
  (** [point p x y] is the point of the page, in points, at which [p] puts the
      normalised position [(x, y)]. Positions outside the unit square
      extrapolate. *)

  val invert : projection -> P2.t -> (float * float) option
  (** [invert p pt] is [Some (x, y)], the normalised position that [p] puts at
      the point [pt] of the page, possibly outside the unit square, or [None] if
      [p] puts no position there. Under {!cartesian} it is [Some _] for every
      finite [pt] if the panel's box has a positive width and height, and [None]
      otherwise. *)
end

(** {1:views Views} *)

(** View values.

    A view holds the values a viewer sets on a figure: the values of keys the
    figure reads through {!bind}, and the zooms of its continuous scales. Views
    live outside figures, so a figure rebuilt when a notebook cell runs again
    reads the values the viewer set on the previous one. A key is identified by
    its name and its sort, never by the value that holds it: code that runs
    again builds equal keys. *)
module View : sig
  (** {1:keys Keys} *)

  type 'a key
  (** The type for keys of values of type ['a]. A key has a name, a sort
      (number, choice, interval, or zoom of a kind of scale) and an initial
      value. *)

  val number : string -> init:float -> float key
  (** [number name ~init] is the key [name] of a number, initially [init]. *)

  val choice : string -> init:string -> string key
  (** [choice name ~init] is the key [name] of a choice among strings, initially
      [init]. *)

  val interval :
    string -> init:(float * float) option -> (float * float) option key
  (** [interval name ~init] is the key [name] of an interval or none, such as a
      brushed selection, initially [init].

      Raises [Invalid_argument] if [init] is [Some (lo, hi)] with [lo] or [hi]
      not finite or [lo > hi]. *)

  val zoom : ?at:id -> 'd Scale.t -> ('d * 'd) option key
  (** [zoom ~at s] is the key of the zoom of the continuous scale named like
      [s], of the kind of [s], in the scope that holds the node [at]
      ({!Hugin_next.section-scopes}), the root by default. Its value
      [Some (a, b)] sets that scale's domain to \[[a];[b]\]
      ({!Scale.with_domain}) in place of its fitted and explicit domains, and
      its initial value [None] leaves it. Every continuous scale has such a key
      without declaring one, and user keys cannot name it. In {!resolve}, after
      the figure's structure changes, a zoom whose node lies in another scope
      with a scale of its name and kind applies there; one whose node, name or
      kind is gone, whose node no scope of its name holds, or whose domain the
      scale cannot take, is ignored with a warning, and so are the zooms of a
      scale that several set.

      Raises [Invalid_argument] if [s] is unnamed or categorical. *)

  (** {1:views Views} *)

  type t
  (** The type for views: keys bound to values. *)

  val empty : t
  (** [empty] binds no key, so every key has its initial value. *)

  val set : 'a key -> 'a -> t -> t
  (** [set k v view] is [view] with [k] bound to [v], replacing any value bound
      to a key of the name of [k], whatever its sort.

      Raises [Invalid_argument] if [k] is an interval key and [v] is
      [Some (lo, hi)] with [lo] or [hi] not finite or [lo > hi]. *)

  val get : 'a key -> t -> 'a
  (** [get k view] is the value bound to [k] in [view], or the initial value of
      [k] if [view] binds none or binds the name of [k] with another sort. *)

  val equal : t -> t -> bool
  (** [equal view view'] is [true] iff [view] and [view'] bind the same keys to
      equal values, floats compared by [Float.equal]. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf view] formats the keys of [view] and their values for debugging.
  *)
end

(** {1:composing Composing} *)

val layer : t list -> t
(** [layer fs] draws the figures of [fs] over one another, the first at the
    bottom, as one figure whose children share every scale ({!section-scopes}).
    [layer []] is an empty panel.
    - {b Broadcasting.} A figure's {e arrangement} is the grid of cells it draws
      in: a {!grid}'s rows and columns, or one cell for any other figure.
      Arrangements broadcast under nx's rule on the shapes [[|rows; columns|]]:
      a figure that is not a grid and a grid of one row or one column repeat
      along the other dimension, and [layer] is the cell-by-cell layer of its
      children. A grid with {!span}s broadcasts only with a figure that is not a
      grid and with a grid of the same cells.
    - {b Wrappers.} [layer] lifts its children's wrappers: [layer [ w a; b ]] is
      [w (layer [ a; b ])] for a {!title} or a {!coord} [w], so
      [layer [ a |> title x ] |> title y] draws [y] above [x]. Wrappers lift one
      at a time, outermost first, equal ones as one:
      [layer [ title x (title y a); title x b ]] is
      [title x (title y (layer [ a; b ]))]. Two different wrappers of one kind
      that would lift at once raise in {!resolve}.
    - {b Ids.} A nested [layer] stays one child with one index, so
      [layer [ layer [ a; b ]; c ]] draws what [layer [ a; b; c ]] draws and
      gives [a] another id.

    {!resolve} raises [Invalid_argument] if the children's arrangements do not
    broadcast, or broadcast to no cells while a child has some, since that child
    would be drawn nowhere: [layer [ a; grid [] ]] raises. *)

val grid : ?widths:float list -> ?heights:float list -> t list list -> t
(** [grid rows] arranges figures in a grid of rows, from the top, of cells, from
    the left: [grid [ [ a; b ] ]] puts [a] left of [b], and
    [grid [ [ a ]; [ b ] ]] puts [a] above [b]. A cell covers one row and one
    column unless it is a {!span}, and is placed in the first column of its row
    that no cell of a row above covers. Data areas align: the data areas of a
    column share their left and right edges and those of a row their top and
    bottom edges, the gaps between cells absorbing what each cell's ticks and
    labels protrude ({!layout}). The position and facet scales of each cell are
    its own unless a {!share} says otherwise.

    [widths] and [heights] weigh the flexible columns and rows, which share the
    room that fixed tracks, such as a legend's, and the tracks of panels with an
    aspect ({!Coord.cartesian}) leave, in proportion to their weights; every
    weight defaults to [1.]. [grid []] draws nothing.

    Raises [Invalid_argument] if a weight is not finite and positive. {!resolve}
    raises [Invalid_argument] if the rows cover different numbers of columns, if
    a span reaches past the last row or covers a column of a row that a span of
    a row above covers, or if [widths] or [heights] differs in length from the
    number of columns or rows. *)

val span : ?rows:int -> ?cols:int -> t -> t
(** [span ~rows ~cols f] is [f] as a grid cell that covers [rows] rows and
    [cols] columns, both [1] by default.

    Raises [Invalid_argument] if [rows] or [cols] is less than [1]. {!resolve}
    raises [Invalid_argument] if it is not a cell of a grid, possibly under
    wrappers. *)

type sharing =
  [ `Shared  (** One scale for the whole node. *)
  | `Independent
    (** One scale per child of the node: per cell of a grid, per child of a
        layer, or per panel of a mark's facets. *) ]
(** The type for the ways a node shares a scale. *)

val share : (string * sharing) list -> t -> t
(** [share pairs f] is [f] with the scales named [n] regrouped by [s] for each
    pair [(n, s)] of [pairs], overriding {!section-scopes} and looking through
    wrappers to find the node's children:
    [grid [ [ a ]; [ b ] ] |> share [ ("x", `Shared) ]] puts [a] and [b] on one
    x scale, labelled below [b], and [share [ ("color", `Independent) ]] on a
    grid gives each cell its own colour scale and legend.

    A pair that leaves a scale's scope as it is changes nothing, so
    [share p (share p f)] means [share p f] and [share [ ("x", `Shared) ]] on a
    grid of one cell is that grid. [share [ ("x", `Independent) ]] on a grid
    whose cells hold grids gives each of its cells one x, which the cells of the
    grid it holds share, where each innermost cell has its own by default.

    Raises [Invalid_argument] if [pairs] names a scale twice. {!resolve} raises
    [Invalid_argument] for a pair whose name no channel in [f] reads; for
    [`Independent] at a layer on a name that a position or facet role in it
    reads, since the children of a layer draw in the same panels and a panel has
    one scale per position and facet; and for [`Independent] at a mark on a name
    that its [fx] or [fy] reads, since that scale makes the panels. *)

val title : ?align:Text.Layout.halign -> Text.t -> t -> t
(** [title ~align s f] is [f] with the title [s] above it, set in the theme's
    faces at its base size. [align] defaults to [`Center], which centres the
    title on the data areas of [f], moved as little as keeps it within [f] with
    its protrusions and legends; [`Left] and [`Right] align it with the left or
    right edge of [f] with its protrusions, where panel labels such as (a) go.
    Titles nest: [title a (title b f)] draws [a] above [b]. *)

val coord : Coord.t -> t -> t
(** [coord c f] draws the panels of [f] in the coordinate system [c]. A panel
    under no [coord] is drawn in the system its marks imply ({!Mark.v}), or else
    in [Coord.cartesian ()].

    {!resolve} raises [Invalid_argument] if a panel lies under two [coord] with
    unequal systems, or if two of its marks imply unequal systems and no [coord]
    is above it. *)

val name : string -> t -> t
(** [name s f] is [f] with the id segment [Field s] in place of its index
    ({!section-ids}), so that the ids in [f] stay the same when the siblings
    before it change.

    Raises [Invalid_argument] if [s] is ["axis"], ["legend"], ["panel"] or
    ["cell"], the segments of generated nodes. {!resolve} raises
    [Invalid_argument] if two siblings have one name or a node has two. *)

val bind : 'a View.key -> ('a -> t) -> t
(** [bind k f] is the figure [f v], where [v] is the value of [k] in the view
    {!resolve} is given ({!View.get}). {!resolve} applies [f] once per call.

    {!resolve} raises [Invalid_argument] if the figure reads two keys of one
    name and different sorts, and warns about a value of the view that no key of
    its sort reads. *)

(** {1:guide_figures Guides as figures} *)

type side = [ `Left | `Right | `Top | `Bottom ]
(** The type for the sides of a panel or of a scope's figure. *)

val axis : ?side:side -> ?grid:bool -> ?show:bool -> string -> t
(** [axis ~side ~grid ~show name] is the axis of the position or facet scale
    [name] in each panel of the figures it is layered with, in place of the one
    the coordinate system generates ({!section-guides}). It draws nothing
    itself. Where:
    - [side] is the side of the panel it is drawn on. Defaults to the side the
      coordinate system gives the role reading the scale: under
      {!Coord.cartesian}, [`Bottom] for x and [`Left] for y; [`Top] for fx and
      [`Right] for fy.
    - [grid] says whether lines cross the panel at its ticks. Defaults to
      [false].
    - [show] says whether it is drawn at all. Defaults to [true].

    {!resolve} raises [Invalid_argument] if [name] names no position or facet
    scale of the panels it is layered with, if [side] is [`Left] or [`Right] for
    a scale that [x] reads or [`Top] or [`Bottom] for one that [y] reads, or if
    a panel holds two different axes for one scale. *)

val legend : ?side:side -> ?show:bool -> string -> t
(** [legend ~side ~show name] is the legend of the scales named [name] in the
    scope of the figures it is layered with, in place of the one generated for
    each ({!section-guides}), placed on [side] of the scope's figure, [`Right]
    by default, and drawn iff [show], [true] by default. It draws nothing
    itself.

    {!resolve} raises [Invalid_argument] if [name] names no scale with a legend
    in that scope, or if the scope holds two different legends for one scale. *)

(** {1:presentation Sizes and themes} *)

(** Figure sizes.

    A size says which length the caller fixes: the whole figure's, or each data
    area's, the figure growing to hold what protrudes. Lengths are in points;
    {!mm} and {!dpi} convert other units. *)
module Size : sig
  type t
  (** The type for sizes. *)

  val figure : float -> float -> t
  (** [figure w h] is a figure [w] points wide and [h] points high. The flexible
      tracks of its grid share what fixed and aspect tracks, gaps and
      protrusions leave, and {!layout} raises if that is negative.

      Raises [Invalid_argument] if [w] or [h] is not finite and positive. *)

  val panels : float -> float -> t
  (** [panels w h] gives each flexible column of weight [k] ({!grid}) a data
      area [k *. w] points wide and each flexible row of weight [k] one [k *. h]
      points high, the figure being as large as its tracks, gaps and protrusions
      need. A track grows past that to hold a title, header or legend longer
      than it ({!layout}).

      Raises [Invalid_argument] if [w] or [h] is not finite and positive. *)

  val mm : float -> float
  (** [mm l] is [l] millimetres in points, [l *. 72. /. 25.4]. *)

  val dpi : float -> float
  (** [dpi d] is the density of [d] dots per inch, [d /. 72.] device pixels per
      point. *)

  val equal : t -> t -> bool
  (** [equal s s'] is [true] iff [s] and [s'] fix the same length to equal
      values. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf s] formats [s] for debugging. *)
end

(** Themes.

    A theme is the presentation a figure does not state: the colours of ink,
    paper and marks, the faces text is set in, the base size every length
    derives from, the default colour schemes, and the locale numbers are written
    in. {!layout} reads it, and {!draw} reads it through the layout; {!resolve}
    does not, so fitted scales never depend on it. *)
module Theme : sig
  type t
  (** The type for themes. *)

  val v :
    ?ink:Color.t ->
    ?paper:Color.t ->
    ?accent:Color.t ->
    ?size:float ->
    ?fonts:Font.t list ->
    ?palette:Scheme.t ->
    ?scheme:Scheme.t ->
    ?locale:Locale.t ->
    unit ->
    t
  (** [v ~ink ~paper ~accent ~size ~fonts ~palette ~scheme ~locale ()] is the
      theme with:
      - [ink], the colour of text, axes and rules. Defaults to [Color.gray 0.1].
      - [paper], the colour {!draw} paints under the figure;
        {!Color.transparent} paints nothing. Defaults to {!Color.white}.
      - [accent], the colour of marks without a colour channel. Defaults to the
        first colour of [palette], so that a lone series and the first category
        agree.
      - [size], the base size in points, one em: the size of plain text and the
        unit of every other length ({!section-lengths}). Defaults to [10.].
      - [fonts], the faces text is set in, as {!Text.Layout.v} sets it: the
        first is the plain face, weight and slant select among them, and the
        others are fallbacks for the characters it lacks
        ({!Text.Layout.section-faces}). Defaults to
        [[ Font.regular; Font.bold ]].
      - [palette], the scheme of band scales that set none. Defaults to
        {!Scheme.tableau10}.
      - [scheme], the scheme of continuous scales that set none. Defaults to
        {!Scheme.viridis}.
      - [locale], the strings tick labels and the [text] role write numbers
        with. Defaults to {!Locale.default}.

      Raises [Invalid_argument] if [size] is not finite and positive or [fonts]
      is empty. *)

  val default : t
  (** [default] is [v ()]. *)

  (** {1:accessors Accessors} *)

  val ink : t -> Color.t
  (** [ink th] is the ink of [th]. *)

  val paper : t -> Color.t
  (** [paper th] is the paper of [th]. *)

  val accent : t -> Color.t
  (** [accent th] is the accent of [th]. *)

  val size : t -> float
  (** [size th] is the base size of [th], in points. *)

  val fonts : t -> Font.t list
  (** [fonts th] is the faces of [th]. *)

  val palette : t -> Scheme.t
  (** [palette th] is the scheme of band scales of [th]. *)

  val scheme : t -> Scheme.t
  (** [scheme th] is the scheme of continuous scales of [th]. *)

  val locale : t -> Locale.t
  (** [locale th] is the locale of [th]. *)

  (** {1:lengths Derived lengths}

      Every other length a figure needs is a multiple of the base size, so a
      theme of size [8.] sets a figure for a paper column: tick labels, legend
      entries and facet headers at [0.9] em, and the titles of channels and
      figures at [1] em; ticks [0.35] em long and [0.25] em from their labels,
      the labels of one axis or legend at least [0.5] em apart, and titles
      [0.25] em from what they title; ticks aiming to lie [5] em apart on x
      axes and colour bars and [3.5] em apart on y axes, or twice the mean
      extent of their labels if that is more
      ({!Hugin_next_kit.Ticks.choose}); legend swatches [1] em square and [0.25]
      em from their labels, and colour bars [1] em wide; axis lines, ticks and
      the outlines of marks [0.08] em wide, and grid lines [0.06] em wide in the
      ink at a fifth of its opacity; lines and rules [0.15] em wide; dots of the
      area of a circle [0.5] em across; and [1] em added to the protrusions that
      meet a gap between grid cells.

      {1:ranges Ranges}

      The ranges a role maps normalised values into are those its scale's
      specification sets, and otherwise: colours by the theme's [scheme] on a
      continuous scale and its [palette] on a band scale; symbols by
      {!Symbol.filled} or {!Symbol.stroked}, as {!dot} says; sizes from area
      [0.] to the area of a circle [1.5] em across, the [size] role implying
      [zero] on its scale since an area is a magnitude; widths from [0.05] to
      [0.5] em; opacities from [0.] to [1.], the normalised value clamped into
      \[[0];[1]\]; positions by the coordinate system, a bar's band position
      implying padding [0.2] of a step on its band scale ({!rect}), so that bars
      stand apart. *)

  (** {1:comparing Comparing and formatting} *)

  val equal : t -> t -> bool
  (** [equal th th'] is [true] iff [th] and [th'] have equal colours, sizes,
      locales, schemes ({!Scheme.equal}) and lists of faces ({!Font.equal}). *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf th] formats [th] for debugging. *)
end

(** {1:extending Extending}

    A mark is a list of channel bindings and a function that draws one panel's
    rows. The built-in marks are written with these modules. *)

(** Roles.

    A role is a mark's slot for a channel ({!section-roles}). It is identified
    by its name. *)
module Role : sig
  type ('d, 'r) t
  (** The type for roles taking channels of domain values ['d] and range ['r].
  *)

  (** {1:positions Positions} *)

  val x : ('d, float) t
  (** [x] is the horizontal position. *)

  val x2 : ('d, float) t
  (** [x2] is the other end of a horizontal extent, on the scale of {!x}. *)

  val y : ('d, float) t
  (** [y] is the vertical position. *)

  val y2 : ('d, float) t
  (** [y2] is the other end of a vertical extent, on the scale of {!y}. *)

  (** {1:appearance Appearance} *)

  val fill : ('d, Color.t) t
  (** [fill] is the colour that fills shapes and glyphs. *)

  val stroke : ('d, Color.t) t
  (** [stroke] is the colour that draws lines and outlines. *)

  val opacity : ('d, float) t
  (** [opacity] is an opacity in \[[0];[1]\]. *)

  val size : (float, float) t
  (** [size] is an area, in square points. *)

  val width : ('d, float) t
  (** [width] is a line width, in points. *)

  val symbol : (string, Symbol.t) t
  (** [symbol] is a marker shape. *)

  val text : ('d, Text.t) t
  (** [text] is a text. It reads no scale ({!Hugin_next.section-roles}). *)

  (** {1:facets Facets} *)

  val fx : (string, string) t
  (** [fx] is the column of facet panels. *)

  val fy : (string, string) t
  (** [fy] is the row of facet panels. *)

  (** {1:values Values} *)

  val value : name:string -> (float, float) t
  (** [value ~name] is the role [name] whose values are the quantities of its
      channel as they are, [nan] where missing: it reads no scale, contributes
      to no domain and yields no guide, so a channel given to it has neither a
      [scale] nor a [title] ({!Mark.v}). It gives a draw function values that
      are neither positions nor colours, such as a cart's angle.

      Raises [Invalid_argument] if [name] is empty or is the name of a role
      above. *)
end

(** Marks.

    {!v} makes a mark from channel bindings and a draw function. Nothing is read
    when a mark is made: {!draw} reads each binding's data, reduced where it
    lives, and calls the draw function once for each panel the mark has a row
    in, with that panel's rows. *)
module Mark : sig
  (** {1:bindings Bindings} *)

  type binding
  (** The type for channels bound to roles. *)

  val bind :
    ?imply:'d Scale.t ->
    ?guide:bool ->
    ('d, 'r) Role.t ->
    ('d, 'r) channel ->
    binding
  (** [bind ~imply ~guide r c] binds the channel [c] to the role [r], where:
      - [imply] is the specification whose properties the mark implies on the
        scale [c] reads ({!Hugin_next.section-merging}), of the kind of [c]'s
        data; its name and transform are ignored, and so is [imply] if [c] is a
        constant. {!rect} and {!rule} bind a length with
        [~imply:(Scale.linear ~zero:true ())], and {!rect} the band position of
        a bar with [~imply:(Scale.band ~padding:0.2 ())].
      - [guide] implies whether the scale [c] reads has a guide, an axis or a
        legend: [true] that it has one, [false] that it has none. It merges as
        the scale's properties do ({!Hugin_next.section-merging}): two marks
        implying different values raise in {!resolve}, and an explicit {!axis}
        or {!legend} overrides both. Unset, it implies nothing. *)

  (** {1:rows Rows}

      The rows of a mark in one panel are those its facet channels put there, in
      the order of their data. A {e dropped} row ({!Hugin_next.section-missing})
      keeps its place. {!positions}, {!points} and {!extent} put it at [nan], so
      that a draw function skips it or breaks a line there, as {!Picture.stamp},
      {!Curve.path} and the gaps of {!Path} do with non-finite coordinates.

      The domain of a panel is the unit square of normalised positions, its
      edges included. A draw function clips positions to it and leaves ink whole
      ({!Hugin_next.section-missing}): it joins {!positions} into paths and
      gives them to {!project}, which cuts them at the domain's edges, and
      places symbols and texts at {!points}, which are [nan] outside it.
      {!index} gives its datum, and {!get} and {!normalized} its own values,
      missing only where they are missing, so that a contour knows the x of a
      column one of whose samples is dropped. Rows are valid only during the
      call of the function given them. *)

  type rows
  (** The type for the rows of a mark in one panel. *)

  val id : rows -> id
  (** [id rows] is the id of the mark, or of the legend in a swatch ({!v}). *)

  val length : rows -> int
  (** [length rows] is the number of rows of [rows]. *)

  val shape : rows -> int array
  (** [shape rows] is the shape of the mark ({!Hugin_next.section-data}). *)

  val index : rows -> int array
  (** [index rows] is the datum of each row, as its index in the mark's shape
      flattened in row-major order. Rows keep their data's indices through
      reductions. *)

  val get : rows -> ('d, 'r) Role.t -> 'r array option
  (** [get rows r] is [Some vs], the value of each row for the role [r] in its
      range, positions normalised and a constant repeated, or [None] if the mark
      binds no channel to [r]. A missing value is the unknown colour of its
      scale for a colour, [nan] for a float, the first symbol of its scale for a
      symbol and the empty text for a text. The array is fresh. *)

  val normalized : rows -> ('d, 'r) Role.t -> float array option
  (** [normalized rows r] is [Some us], the value of each row for the role [r]
      normalised by its scale, before the role's range and {!map_range}, or
      [None] if [r] is unbound, bound to a constant or reads no scale. The array
      is fresh. *)

  val range : rows -> ('d, 'r) Role.t -> (float -> 'r) option
  (** [range rows r] is [Some f] if [r] is bound to a channel that reads a
      scale, and [None] if it is unbound, bound to a constant or reads no scale.
      [f u] is the value of [r] at the normalised value [u], mapped by the
      channel's {!map_range}: on a continuous scale, the value the range of [r]
      gives [u] ({!Hugin_next.section-roles}), as the scale's specification or
      else the theme sets it, and [u] itself for a position; on a band scale,
      the value of the category whose step holds [u] ({!Scale.invert}). For
      [nan], and on a band scale for a [u] that no step holds, it is the value
      {!get} gives a missing value. With [range rows r = Some f], [get rows r]
      is [Option.map (Array.map f) (normalized rows r)], and a contour fills the
      band between two levels with [f] at their midpoint. *)

  val ticks : rows -> ('d, 'r) Role.t -> float array option
  (** [ticks rows r] is [Some us], the normalised positions of the major ticks
      that {!layout} froze for the scale [r] reads, in increasing order, whether
      or not a guide shows them, or [None] if [r] is unbound, bound to a
      constant or reads no scale. On a reversed scale the ticks' values decrease
      along [us]. The array is fresh. *)

  val positions : rows -> float array * float array
  (** [positions rows] is [(us, vs)], row [i] being at the normalised position
      [(us.(i), vs.(i))]: its normalised [x] and [y], [0.5] for a position the
      mark does not bind and the centre of its band on a band scale. A dropped
      row is at [(nan, nan)], and a position outside the domain is kept. The
      arrays are fresh. *)

  val points : rows -> float array * float array
  (** [points rows] is [(xs, ys)], row [i] being at [(xs.(i), ys.(i))] in points
      on the page: its {!positions} mapped by the panel's projection. A dropped
      row and a row whose position lies outside the domain are at [(nan, nan)].
      The arrays are fresh. *)

  val extent : rows -> [ `X | `Y ] -> float array * float array
  (** [extent rows `X] is [(a, b)], the normalised interval each row covers
      along x, from [a.(i)] to [b.(i)] in either order:
      - if the mark binds [x2], the hull of the extents of [x] and [x2], each
        its band on a band scale and its value otherwise;
      - otherwise, if [x] reads a band scale, its band;
      - otherwise, if [x] reads a continuous scale, a {e length}: from the
        normalised value of [0.] clamped into the domain, which is the domain's
        lower end on a log scale, to [x];
      - for a constant [x], from [x] to [x];
      - if the mark binds no [x], from [0.] to [1.].

      Ends outside the domain are clamped into \[[0];[1]\], and an interval
      wholly below [0.] or wholly above [1.] covers [(nan, nan)], as does a
      dropped row. [extent rows `Y] is likewise along y, with [y] and [y2]. The
      arrays are fresh. *)

  val projection : rows -> Coord.projection
  (** [projection rows] is the panel's projection, which puts the normalised
      position [(x, y)] at [Coord.point (projection rows) x y] on the page. *)

  val project : rows -> Path.t -> Path.t
  (** [project rows p] is [p], given in the panel's normalised positions, cut at
      the domain's edges ({!Path.crop}) and mapped into points on the page by
      the panel's projection: an open subpath keeps its pieces within the
      domain, and a closed one the part of its region there, so fill a path
      closed. Where the projection is not affine, segments are subdivided so
      that the path follows it. Under {!Coord.cartesian} the cut path is mapped
      by one affine map ({!Path.transform}). *)

  val series : rows -> rows list
  (** [series rows] is [rows] split into {e series}: two rows are in one series
      iff they have the same index along every axis of the mark's shape but the
      last and the same category in every channel that reads a band scale and is
      bound to a role other than [x], [x2], [y] and [y2]. A missing category
      counts as one category of its own. So a categorical [stroke] splits a
      line's rows and a categorical [x] does not. Series are in the order of
      their first rows and keep their rows in order. *)

  val theme : rows -> Theme.t
  (** [theme rows] is the theme of the layout being drawn. *)

  val text :
    ?halign:Text.Layout.halign ->
    ?valign:Text.Layout.valign ->
    rows ->
    Color.t ->
    P2.t ->
    Text.t ->
    Picture.t
  (** [text ~halign ~valign rows c pt s] draws [s] set in the faces of the theme
      at its base size, aligned about the point [pt] of the page by [halign] and
      [valign] ({!Text.Layout.section-alignment}), [`Center] and [`Middle] by
      default, its characters without a colour of their own painted with [c]. A
      character that no face has is drawn as its face's [.notdef] glyph, with a
      warning ({!warn}). *)

  val warn : rows -> string -> unit
  (** [warn rows msg] adds the warning [msg], about the mark's data, to the
      warnings of the drawing ({!Drawing.warnings}), under the mark's id. *)

  (** {1:reducers Reducers}

      A reducer draws a mark's rows in its stead when they are many for the
      density {!draw} is given. {!m4} and {!cells} choose rows where the data
      lives, before they are read; {!raster} reads the rows and paints them
      once. It applies only to the marks that name it, which so state that they
      draw what it draws, and every reduction draws what the mark would draw
      from every row, as each states. *)

  type reducer
  (** The type for reducers. *)

  val m4 : reducer
  (** [m4] keeps, of each series ({!series}) with more than four rows per
      device-pixel column, each column's first, last, lowest and highest rows,
      when the series' [x] is monotone and its other channels constant, and the
      projection is affine. A mark naming it draws each series as a path through
      its points: the reduced drawing keeps each column's end points and
      vertical extent, and differs from the whole one only in antialiased edge
      pixels, by the coverage of the dropped segments. *)

  val cells : reducer
  (** [cells] draws a panel's rows as one image whose pixels are their cells,
      when the projection is affine, [x] and [y] read band scales without
      padding, the mark binds no [x2], [y2] or [stroke], and no two rows share a
      cell; a cell no row covers paints nothing. A padded band scale leaves gaps
      between cells, which one image of steps cannot show. Past 4 cells per
      device pixel along either axis, only the cells that raster output samples
      are gathered and read. A mark naming it fills the rectangle of each row's
      extents ({!extent}) with its [fill] at its [opacity] and draws nothing
      else, so the image is exactly its drawing. *)

  val raster : reducer
  (** [raster] draws a panel's picture of the mark as one image painted by the
      raster renderer at the density, over the device pixels the picture reaches
      ({!Picture.bounds}), when the mark has more than 20,000 rows in the panel
      or more rows than the panel has device pixels, whatever the output. Rows
      are painted in their order, so the image is what the picture paints, each
      component rounded to a level. Composited onto a raster page, which rounds
      again, it is within one level of the picture drawn there directly. *)

  (** {1:making Making marks} *)

  val v :
    name:string ->
    ?reduce:reducer ->
    ?coord:Coord.t ->
    ?swatch:(rows -> Picture.t) ->
    binding list ->
    (rows -> Picture.t) ->
    t
  (** [v ~name ~reduce ~coord ~swatch bindings draw] is the mark [name] of the
      channels [bindings], where [draw rows] is its picture in a panel for that
      panel's rows, in points on the page. [draw] applies every role it binds,
      [opacity] included, and clips positions to the domain ({!section-rows}).
      {!draw} only selects each panel's rows and tags the picture with the
      mark's id and rows ({!Picture.tag}), instance by instance if it is a stamp
      of one instance per row: it clips nothing, so ink drawn from a row inside
      the domain shows whole. Other arguments are:
      - [name], the kind of mark in messages and printed forms, such as ["dot"]
        or ["fehu.cart"].
      - [reduce], the reducer that may draw the mark's rows. Unset, rows are
        never reduced.
      - [coord], the coordinate system the mark implies for the panels it is
        drawn in, which a {!val-coord} above it overrides.
      - [swatch], the picture of a legend entry. It is called with one row in a
        box one em square, onto which the projection maps the unit square, and
        in which the roles that read the legend's scale take the entry's value,
        constants keep theirs and other roles are unbound, so that its point is
        the box's centre ({!points}) and its extents span the box ({!extent}).
        For the entry [k] of a legend of [n] entries, the rows have the shape
        [[|n|]], the index [[|k|]] and the legend's id
        ({!Hugin_next.section-ids}), with which {!draw} tags the swatch. Unset,
        it is [draw].

      [draw] and [swatch] must not read mutable state.

      Raises [Invalid_argument] if:
      - two bindings bind one role or roles of one name;
      - [x2] is bound without [x] or [y2] without [y];
      - [x] and [x2], or [y] and [y2], are bound to one channel of quantities
        and one of categories;
      - a channel with a [scale] or a [title] is bound to a role that reads no
        scale, [text] or a {!Role.value};
      - the channels do not broadcast, or a {!Hugin_next.dim} or a
        {!Hugin_next.index} does not fit their shape
        ({!Hugin_next.section-data}). *)
end

(** {1:stages Stages} *)

(** Resolved figures.

    A resolved figure is a figure with its view applied, its {!bind}s evaluated,
    its ids assigned, its scopes formed and its composition checked, and every
    scale fitted to its data, which was summarised where it lives. It holds no
    size and no theme. *)
module Resolved : sig
  type t
  (** The type for resolved figures. *)

  val scale : ?at:id -> t -> 'd Scale.t -> 'd Scale.t
  (** [scale ~at r s] is the scale named like [s], of the kind of [s], in the
      scope that holds the node [at] ({!Hugin_next.section-scopes}), the root by
      default, fitted: its domain set and its [nice] and [zero] unset
      ({!Scale.fit}), and its other properties those its channels merged,
      implied ones included. Given to another figure, it normalises as it does
      in [r] and carries the ranges it states. So a zoomed scale has its zoomed
      domain ({!View.zoom}), and a band scale read by [y] has the [reverse] that
      [y] implies, which reverses x where the scale is given to [x]. Only the
      name and kind of [s] are read: [scale r (Scale.linear ~name:"color" ())]
      is the fitted quantitative ["color"].

      Raises [Invalid_argument] if [s] is unnamed, if no node of [r] has the id
      [at], if no scope of the name of [s] holds [at], or if its scope has no
      scale of that name and kind. *)

  val warnings : t -> warning list
  (** [warnings r] is the warnings of resolving, in the order of the nodes they
      are about in the figure, a generated node in the place of the node it lies
      under, then those about values of the view that no key reads. *)

  val equal : t -> t -> bool
  (** [equal r r'] is [true] iff [r] and [r'] resolve {!Hugin_next.equal}
      figures under {!View.equal} views to the same scopes, {!Scale.equal}
      fitted scales and equal warnings. The figures that {!bind} returns are not
      compared: they are functions of the figure and the view, and comparing a
      figure that a {!bind} builds afresh on each call would make a resolved
      figure unequal to itself resolved again. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf r] formats the scopes of [r], their fitted scales and the
      warnings, for debugging and baselines. *)
end

(** Laid-out figures.

    A laid-out figure is a resolved figure on a page: its text measured with the
    theme's faces, its ticks chosen and frozen, its guides built, its grid
    solved and each panel's projection fixed. *)
module Layout : sig
  type t
  (** The type for laid-out figures. *)

  type panel = {
    id : id;  (** The id of the panel ({!Hugin_next.section-ids}). *)
    box : Box2.t;  (** The data area, in points on the page. *)
    projection : Coord.projection;  (** The panel's projection. *)
  }
  (** The type for panels. *)

  val size : t -> float * float
  (** [size l] is the width and height of the figure, in points. *)

  val panels : t -> panel list
  (** [panels l] is the panels of [l], in the order of the figure. *)

  val warnings : t -> warning list
  (** [warnings l] is the warnings of resolving and laying out, in the order of
      the figure. *)

  val equal : t -> t -> bool
  (** [equal l l'] is [true] iff [l] and [l'] lay out {!Resolved.equal} figures
      in {!Theme.equal} themes and have equal sizes, panels of equal ids, boxes
      and coordinate systems, equal frozen ticks, guides, headers and titles of
      equal contents in equal boxes, and equal warnings. Equal layouts draw
      {!Drawing.equal} drawings at one density. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf l] formats the page size, panels, ticks and guides of [l] and the
      warnings, for debugging and baselines. *)
end

(** Drawings.

    A drawing is a laid-out figure drawn at a density: its marks reduced, read
    and drawn, painted over the theme's paper with every mark and panel tagged
    with its id. *)
module Drawing : sig
  type t
  (** The type for drawings. *)

  val renderable : t -> Renderable.t
  (** [renderable d] is the picture of [d] on a page of the figure's size: the
      paper, then the panels' marks, the guides and the titles. *)

  val warnings : t -> warning list
  (** [warnings d] is the warnings of resolving, laying out and drawing, in the
      order of the figure. *)

  val equal : t -> t -> bool
  (** [equal d d'] is [true] iff [d] and [d'] have equal warnings and
      {!Renderable.equal} renderables. A drawing drawn again is equal to the
      first. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf d] formats [d] for debugging. *)
end

val resolve : ?prev:Resolved.t -> ?view:View.t -> t -> Resolved.t
(** [resolve ~prev ~view f] is [f] resolved under [view], {!View.empty} by
    default: its {!bind}s evaluated, its ids assigned, its scopes formed and its
    composition checked, then every scale's specification merged and fitted to
    summaries of its marks' data computed where their tensors live, categorical
    scales first. It reads neither size, theme nor density.

    With [prev], a mark's summary is reused if its tensors and functions are
    physically, and its specifications and the transforms and explicit
    categorical domains that merging gives the scales it reads are structurally,
    those of a mark of [prev]; every scope is fitted again. The result is
    {!Resolved.equal} to [resolve ~view f]. A summary is a mark's, and depends
    on those scales, because a row is dropped by all its channels together and a
    value is missing where its scale cannot place it ({!section-missing}): a
    mark layered with a channel that makes ["y"] logarithmic loses its rows with
    non-positive [y] from every hull.

    Raises [Invalid_argument], naming the nodes at fault, on the composition
    errors and conflicts that {!layer}, {!grid}, {!span}, {!share}, {!coord},
    {!name}, {!bind}, {!axis}, {!legend}, {!section-names} and
    {!section-merging} state, and as reading a tensor raises
    ({!section-conventions}). *)

val layout :
  ?prev:Layout.t -> ?theme:Theme.t -> Size.t -> Resolved.t -> Layout.t
(** [layout ~prev ~theme size r] lays [r] out at [size] in [theme],
    {!Theme.default} by default. It measures text in the theme's faces; chooses
    the ticks of each guide by their measured labels
    ({!Hugin_next_kit.Ticks.choose}) at the lengths a solve with empty
    protrusions gives, then again at those that a solve with the first choice's
    protrusions gives, and freezes the second choice; wraps the entries of
    legends above or below their panels into rows at the lengths a solve with
    the frozen ticks gives; builds axes, legends, headers and titles; solves the
    grid a last time; and builds each panel's projection. An axis or colour bar
    too short for its labels drops alternate ones, which never widens a
    protrusion.

    The grid sizes fixed and aspect tracks first, makes each gap the largest
    protrusions that meet it from either side plus the theme's gap
    ({!Theme.section-lengths}), and shares what remains among flexible tracks by
    weight: each takes its weight's share, or what it needs if that is more, the
    others sharing the rest. A panel protrudes by its guides, and by half its
    longest tick label past the ends of each labelled axis, where a label
    centred on an end tick reaches; the titles and headers of its axes need its
    data area to be as long as they are, which for a panel with an aspect sizes
    both its column and its row. A legend stands beside the panels of its scope,
    as long as their data areas, and a title above the figure it titles, each a
    gap beyond the protrusions of those panels wherever they lie in their cells.
    A panel with an aspect that its cell cannot hold is drawn in the largest box
    of its aspect, centred in its cell.

    With [prev], the measurements of labels of equal text in an equal theme are
    reused and the grid is solved again: the result is {!Layout.equal} to
    [layout ~theme size r].

    Raises [Invalid_argument] if fixed and aspect tracks with their gaps and
    protrusions exceed a {!Size.figure}, naming the size the figure needs, or if
    a title, a channel's title or a tick label of a continuous scale holds a
    character that no face of the theme has, naming it. Category labels with
    such characters are data: they are drawn with [.notdef] glyphs, with a
    warning. *)

val draw : ?prev:Drawing.t -> density:float -> Layout.t -> Drawing.t
(** [draw ~prev ~density l] paints [l] at [density] device pixels per point. It
    reduces marks where their data lives by their reducers ({!Mark.reducer}),
    reads each channel's reduced data to the host once, as [float64], keeping
    its source dtype for writing numbers, then calls each mark's draw function
    on each panel's rows and paints the paper, the panels and the guides. Vector
    output of the drawing is reduced at [density] too.

    With [prev], the picture of each panel whose id, box, coordinate system,
    scales with their frozen ticks, marks with their ids, theme and density
    equal those of a panel of [prev] is reused, with the warnings its draw
    functions gave: the result is {!Drawing.equal} to [draw ~density l]. Draw
    functions read the theme and the ticks ({!Mark.theme}, {!Mark.ticks}), so a
    change of either draws the panel again. Marks compare as {!equal} compares
    them, tensors physically, so a tensor changed in place between two draws is
    not seen, and the panel is reused with its old picture.

    Raises [Invalid_argument] if [density] is not finite and positive, what draw
    functions raise, and as reading a tensor raises ({!section-conventions}). *)

val render :
  ?view:View.t -> ?theme:Theme.t -> ?density:float -> Size.t -> t -> Drawing.t
(** [render ~view ~theme ~density size f] is
    [draw ~density (layout ~theme size (resolve ~view f))], with [density] [2.]
    by default. *)

(** {1:output Output} *)

val save :
  ?warn:(warning -> unit) ->
  ?view:View.t ->
  ?theme:Theme.t ->
  ?size:Size.t ->
  ?density:float ->
  string ->
  t ->
  unit
(** [save ~warn ~view ~theme ~size ~density file f] writes the renderable of
    [render ~view ~theme ~density size f] to [file] in the format its extension
    names, ignoring case: PNG for [.png], drawn at [density] with the density
    and the sRGB colour space recorded ({!Hugin_next_vg_raster.png}); SVG for
    [.svg]; PDF for [.pdf]. [size] defaults to [Size.figure 360. 240.] and
    [density] to [2.]. [warn] is called on each of the drawing's warnings, in
    order, before the file is written; by default it formats the warning with
    {!pp_warning} on [Format.err_formatter].

    Raises [Invalid_argument] for another extension, what {!render} raises, and
    [Sys_error] if [file] cannot be written. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf f] displays [f] in a Quill notebook: it renders [f] as SVG as
    {!save} does by default, and prints on [ppf] a one-line summary of [f]
    inside a [Format.String_tag]. The tag's string is the line [quill.display],
    the line [image/svg+xml], an empty display id line, then the SVG document: a
    display tag of the display protocol documented in Quill's [Quill.Cell]
    module. Hugin depends on no Quill library.

    Formatters ignore the tag by default and print the summary alone. It counts
    the drawing's warnings, such as [hugin figure (2 warnings)], so that a user
    without the display sees that there are some.

    Raises what {!render} raises. *)
