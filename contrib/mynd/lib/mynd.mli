(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Images, positions in them and their annotations, as nx values.

    An {e image} is an [Nx.t] whose last three axes are [[h; w; c]]: [h] rows
    from the top, [w] columns from the left and [c] channels, one included. Its
    other axes index independent images: no function mixes them, and a
    function's result for one image is the same in any batch. A {e mask} is a
    boolean image of one channel, [[…; h; w; 1]], so that every function on
    images takes masks too.

    {e Values.} An integer of [n] bits holds a {e code} [k], which stands for a
    fraction of full scale as Vulkan's normalized fixed-point formats read it:
    [k / (2ⁿ - 1)] for an unsigned integer (UNORM), in [[0, 1]], and
    [max (k / (2ⁿ⁻¹ - 1)) (-1)] for a signed one (SNORM), in [[-1, 1]]. A float
    holds the fraction itself. {!convert} changes the dtype and keeps the
    fraction. Nothing else clamps.

    {e Encoding.} An image holds the values its caller gives it. Decoders return
    a file's encoded values, sRGB for most photographs, with alpha, where
    present, straight: colours not multiplied by it. Every function computes on
    those numbers, so resizing or blending sRGB values averages encoded numbers,
    as the models trained on them did. To average light, apply {!srgb_to_linear}
    first and {!linear_to_srgb} after. To interpolate an image with alpha,
    {!premultiply} first and {!unpremultiply} after, or the colour of
    transparent pixels bleeds into visible ones.

    {e Coordinates} are continuous, [x] to the right and [y] down, in pixels.
    Pixel [(i, j)], row [i] and column [j], is the {e cell}
    [\[j, j + 1) × \[i, i + 1)], centred at [(j + ½, i + ½)]; an [h] by [w]
    image covers [\[0, w) × \[0, h)], and scaling an image by [s] scales its
    coordinates by [s]. A {e position} is a point [(x, y)] held along a last
    axis of extent [2], [x] first. Positions and boxes use these coordinates.

    Unless its doc says otherwise, a function computes where its arguments lie,
    is differentiable where its result is a float that depends continuously on a
    float argument, and raises [Invalid_argument] at the call on a shape it
    cannot read. *)

(** {1:values Values} *)

val convert : ('a, 'b) Nx.dtype -> ('v, 'r, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [convert dt x] is [x] in [dt], each value keeping its fraction; [x] itself
    where [dt] is [x]'s dtype. With [m] the largest code:
    - integer to float: [k / m] rounded once into [dt] for codes of up to 16
      bits, else at float64 then into [dt]; a signed [-m - 1] gives [-1];
    - float to integer: [x] clamped to [[0, 1]] ([[-1, 1]] signed) times [m], in
      float32, or float64 for a float64 [x] or an [m] past 24 bits, rounded to
      even; NaN gives [0], [1] the largest code [m] and, signed, [-1] gives
      [-m];
    - integer to integer: the nearest code, computed exactly; a negative
      fraction into an unsigned [dt] gives [0];
    - float to float: {!Nx.cast}, the only pair with a derivative.

    A code survives a round trip through float8_e4m3fn and every float more
    precise (not float8_e5m2) at 4 bits, through float16 and bfloat16 and wider
    at 8 bits, through float32 at 16 bits and float64 at 32 bits, a signed
    [-m - 1] coming back as [-m].

    Raises [Invalid_argument] if either dtype is boolean or complex. *)

val srgb_to_linear : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [srgb_to_linear x] is the linear light of sRGB-encoded values, by CSS Color
    4's extended transfer function: [x / 12.92] where [|x| <= 0.04045], else
    [sign x ((|x| + 0.055) / 1.055){^2.4}]. Every channel is converted: split
    off an alpha channel first, since alpha is linear. *)

val linear_to_srgb : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [linear_to_srgb x] inverts {!srgb_to_linear}: [12.92 x] where
    [|x| <= 0.0031308], else [sign x (1.055 |x|{^1/2.4} - 0.055)]. *)

val premultiply : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [premultiply x] is [x] with every channel but the last multiplied by the
    last, its alpha. *)

val unpremultiply : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [unpremultiply x] divides every channel but the last by the last; where
    alpha is [0] the colour is [0] and its derivative [0]. *)

(** {1:colour Colour}

    A {e colour} is a last axis of extent [3] holding red, green and blue, in
    that order. The functions of this section read the colours along the last
    axis of an array of any rank, each on its own, and return a result per
    colour. They take floats: {!convert} an integer image first. Any channel
    beyond the three, such as alpha, is split off with {!Nx.split_at} first and
    joined back with {!Nx.concatenate} after. A last axis other than [3] raises
    [Invalid_argument].

    {e Luma} is the weighted sum of encoded values that ITU-R BT.601 §2.5.1
    defines, [0.299 r + 0.587 g + 0.114 b]: the Y of JPEG's YCbCr, and the grey
    that video and image files carry. {e Luminance} is CIE's Y of linear sRGB
    ({!srgb_to_linear}), the light a colour emits relative to white. The two
    differ on the same colour, so each function names the values it reads.

    {e Linear sRGB} has ITU-R BT.709's primaries and white (CIE 1931 xy: red
    [(0.640, 0.330)], green [(0.300, 0.600)], blue [(0.150, 0.060)], white D65
    [(0.3127, 0.3290)]). Its matrix [M] to CIE XYZ is derived from them, as CSS
    Color Module Level 4 gives it in rationals, so white [(1, 1, 1)] has
    [Y = 1].

    The {e HSV} form of a colour is Smith's hexcone (Color Gamut Transform
    Pairs, 1978). With [v] the greatest channel and [m] the least, its value is
    [v], its saturation [(v - m) / v], and its hue the colour's place on the
    hexagon of hues in turns, in [\[0, 1)]: red [0], yellow [1/6], green [1/3],
    cyan [1/2], blue [2/3], magenta [5/6]. A {e grey}, [v = m], has hue and
    saturation [0]. An HSV array holds hue, saturation and value as three
    fractions, so {!convert} takes it to codes as any image.

    These functions clamp nothing: values outside [[0, 1]] take the same
    formulas. A NaN channel gives NaN in every channel of the result. Where
    channels tie, [v] and [m] take the tangents {!Rune} gives [Nx.max] and
    [Nx.min] ({!Rune.section-kinks}). *)

val luma : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [luma x] is the luma of the colours [x], [[…; 3]] to [[…; 1]], computed as
    [g + 0.299 (r - g) + 0.114 (b - g)] in [x]'s dtype, so that a grey
    [(v, v, v)] gives [v], equal as a number. Where that form is not finite, as
    at a grey of infinities or where [r - g] overflows, it is the sum
    [0.299 r + 0.587 g + 0.114 b]. Linear values take {!luminance}. *)

val luminance : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [luminance x] is the luminance of the linear sRGB colours [x], [[…; 3]] to
    [[…; 1]]: the middle row of [M], [87098/409605], [175762/245763],
    [12673/175545], in the form and with the greys of {!luma}. On encoded values
    it is a sum without meaning: apply {!srgb_to_linear} first. *)

val rgb_to_hsv : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [rgb_to_hsv x] is the HSV form of the colours [x], [[…; 3]] to [[…; 3]]. The
    hue is [(g - b) / (6 (r - m))] where [r] is greatest,
    [1/3 + (b - r) / (6 (g - m))] where [g] is and [r] is not, and
    [2/3 + (r - g) / (6 (b - m))] elsewhere, taken modulo [1]; two channels tied
    for the greatest give one hue by either rule. The saturation is infinite
    where [v = 0 > m]. A colour with an infinite channel has NaN hue and
    saturation, and value [v].

    Away from greys the hue is continuously differentiable, across ties and
    across red. At a grey the hue's and the saturation's derivatives are [0]. *)

val hsv_to_rgb : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [hsv_to_rgb y] is the colours whose HSV forms are [y], [[…; 3]] to [[…; 3]]:
    channel [n], [5] for red, [3] for green and [1] for blue, is [v - v s c]
    with [c = min (max (min k (4 - k)) 0) 1] and [k] the remainder of [n + 6 h]
    by [6] in [\[0, 6)]. Any finite hue is read modulo [1], so adding [d] to the
    hue turns the colour by [d] turns; a NaN or infinite component gives NaN in
    every channel.

    The derivative in the hue is that of the side of a sextant's edge the hue
    lies on. Where [6 h], or its remainder by 6 (from 0 up to 6), rounds to an
    integer, at an edge, it is the mean of its two one-sided derivatives, as
    [Nx.minimum]'s is.

    [hsv_to_rgb (rgb_to_hsv x)] is [x] to rounding for every finite colour that
    is grey or has [v ≠ 0]. Its derivative is the identity there, at ties of
    channels too, but at greys and at a colour within rounding of a tie whose
    hue rounds onto an edge or across it. *)

val linear_to_lab : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [linear_to_lab x] is the CIE 1976 L*a*b* coordinates of the linear sRGB
    colours [x], [[…; 3]] to [[…; 3]], relative to sRGB's own white D65, with no
    chromatic adaptation. With [(X, Y, Z)] the product of [M] and the colour,
    each divided by white's, [L* = 116 f(Y) - 16], [a* = 500 (f(X) - f(Y))] and
    [b* = 200 (f(Y) - f(Z))], where [f t] is [t{^1/3}] above [ε = 216/24389] and
    [(κ t + 16) / 116] elsewhere, [κ = 24389/27]: CIE's exact constants, with
    which the two pieces of [f] meet with equal slopes. White is [(100, 0, 0)]
    and black [(0, 0, 0)]. A colour measured relative to D50, as ICC profiles
    and many image editors report it, must be adapted to D65 first. *)

val lab_to_linear : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [lab_to_linear y] inverts {!linear_to_lab}: [f]'s inverse is [u{^3}] above
    [6/29], else [(116 u - 16) / κ], then [M]'s inverse. Colours outside sRGB's
    gamut come back with channels below [0] or above [1]. *)

(** {1:sampling Sampling at positions}

    A {e kernel} [k] of half-width [h] weighs a pixel by [k(τ)], [τ] the
    position less the pixel's centre in pixels, and is [0] outside [\[-h, h)];
    [2h] is a whole number. To {e sample} an image [x] at [(u, v)] is to read
    the weighted mean

    [Σ k(u - ½ - j) k(v - ½ - i) x̃(i, j) / Σ k(u - ½ - j) k(v - ½ - i)]

    over the [2h] rows [i] and [2h] columns [j] whose arguments lie in
    [\[-h, h)], where [x̃] reads rows and columns outside the image by an
    {e edge} rule. [`Box] reads the pixel whose cell holds the position,
    [`Triangle] interpolates bilinearly and [`Cubic a] is Keys' bicubic.
    Sampling filters nothing: positions more than a pixel apart alias, so shrink
    with {!resize}.

    Images and positions pair batch axes: the positions' first axes, as many as
    the image's batch axes, broadcast with them, and their other axes but the
    last are the {e points}, so the result is [[b…; points…; c]]. Images of
    several sizes are a list; a padded batch reads as its images only under
    [`Zero] with zeros as padding. *)

type kernel = [ `Box | `Triangle | `Cubic of float | `Lanczos of int ]
(** The type for kernels:
    - [`Box] is [1] on [\[-½, ½)].
    - [`Triangle] is [1 - |τ|] on [\[-1, 1)].
    - [`Cubic a] is Keys' cubic convolution on [\[-2, 2)]:
      [(a + 2)|τ|³ - (a + 3)|τ|² + 1] to [|τ| = 1], then
      [a|τ|³ - 5a|τ|² + 8a|τ| - 4a]. [`Cubic (-0.5)] is the Catmull-Rom spline;
      [-0.75] is sharper.
    - [`Lanczos n] is [sinc τ sinc (τ / n)] on [\[-n, n)] ({!Nx.sinc}), for [n]
      lobes.

    A function taking a kernel raises [Invalid_argument] for [`Cubic a] with [a]
    not finite or [`Lanczos n] with [n < 1]. *)

type extension =
  [ `Zero | `Replicate | `Symmetric_whole | `Symmetric_half | `Periodic ]
(** The type for extension rules: {!Nx.extension}'s rules as {!Nx.extend} reads
    a signal, but [`Antisymmetric_whole], whose reads combine two pixels. *)

type edge = [ extension | `Renormalise ]
(** The type for edge rules: an extension, or [`Renormalise], which reads only
    the pixels inside the image and takes the mean over their weights. Where
    those weights sum to [0], or there is no pixel inside, the read is [0]. *)

val sample :
  edge:[< edge ] ->
  kernel ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t
(** [sample ~edge k x p] is the images [x], [[b…; h; w; c]], sampled under [k]
    at the positions [p], [[b…; q…; 2]]. Each argument is the exact difference
    rounded once into [x]'s dtype, held below [h] where it rounds up to it. Each
    axis's [2h] kernel values are divided by their sum, added in order; a
    pixel's weight is the product of its row's and its column's, times the
    pixel, and the products add in row-major order from the top left. A position
    with a NaN or infinite part reads NaN, and one past [±2⁶²] reads as there.
    The result is linear in [x]; its derivative in [p] is the weights', from the
    right where a part of a position crosses a centre, and [0] under [`Box].

    Raises [Invalid_argument] if [x] has fewer than three axes, [p]'s last axis
    is not [2], [p]'s leading axes do not broadcast with [x]'s, or [x] has no
    row or no column and [edge] is not [`Zero] or [`Renormalise]. *)

val nearest :
  edge:[< extension ] ->
  ('v, 'r, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  ('v, 'r, 'd) Nx.t
(** [nearest ~edge x p] is, at each position [(u, v)] of [p], the pixel
    [x̃(⌊v⌋, ⌊u⌋)] whose cell holds it, for every dtype: a gather, so labels and
    masks keep their values. At a finite position it is [sample ~edge `Box x p]
    on floats. A position with a NaN or infinite part reads the dtype's zero,
    [false] in a mask. Its derivative in [x] scatters back; in [p] it is [0].

    Raises [Invalid_argument] as {!sample} does. *)

val centres : (float, 's) Nx.dtype -> int * int -> (float, 's, Nx.host) Nx.t
(** [centres dt (h, w)] is the [[h; w; 2]] host positions of an [h] by [w]
    image's centres, [(j + ½, i + ½)] at [(i, j)], exact while [2 max h w] is an
    integer [dt] holds.

    Raises [Invalid_argument] if [h] or [w] is negative. *)

(** {1:transforms Transforms}

    A {e transform} is a [[…; 3; 3]] array [m] mapping a position [(x, y)] to
    [((m₀₀ x + m₀₁ y + m₀₂) / d, (m₁₀ x + m₁₁ y + m₁₂) / d)] with
    [d = m₂₀ x + m₂₁ y + m₂₂], each sum added left to right; an {e affine} one
    has the last row [0 0 1], so [d] is [1] at a finite position.
    [Nx.matmul a b] applies [b] first; {!Nx.inv} inverts. A warp samples at the
    transform of its own centres, so its transform maps output positions to
    input ones: [sample ~edge k x (Transform.apply (Nx.inv m) c)] for [m]
    mapping input to output. Each constructor takes parameters [[…]] or [[…; 2]]
    and gives the transforms [[…; 3; 3]], differentiable in them. *)
module Transform : sig
  val apply :
    (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [apply m p] is the positions [p], [[b…; q…; 2]], each mapped by its
      transform in [m], [[b…; 3; 3]], batch axes paired as {!sample} pairs them.

      Raises [Invalid_argument] if [m]'s last two axes are not [3] and [3],
      [p]'s last axis is not [2] or the batch axes do not broadcast. *)

  val translation : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [translation t] adds [t], [[…; 2]], to positions. *)

  val scaling : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [scaling s] multiplies [x] by [s] at [0] and [y] by [s] at [1] along [s]'s
      last axis. *)

  val rotation : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [rotation a] turns positions by [a] radians about the origin, from [+x]
      toward [+y]: clockwise on the image, whose [y] points down. *)

  val shear : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [shear s] adds [s] at [0] times [y] to [x] and [s] at [1] times [x] to
      [y]. *)
end

(** {1:resizing Resizing}

    To {e resize} an axis of [n] pixels to [n'] is to read it at [n'] new
    centres through a kernel stretched to the new spacing. New pixel [j]'s
    centre lies at [c = (j + ½) n / n'] in old pixels; with
    [s = max 1 (n / n')], old pixel [i] is a {e tap} of [j] when
    [τ = (c - i - ½) / s] lies in [\[-h, h)]. Shrinking stretches the kernel by
    [s], which band-limits the result to the new spacing; enlarging reads as
    {!sample} at the new centres. A tap's {e weight} is [k(τ)] divided by the
    sum over [j]'s taps, so a flat image resizes to itself. A tap outside the
    image reads as the edge says; [`Renormalise] drops it before the sum. A tap
    of weight [0] is not read.

    Each [τ] is a quotient of two integers rounded once at float64, each weight
    and each sum computed at float64 and the weight rounded once into the dtype
    the pass computes in: [x]'s, or float32 for the formats narrower than 32
    bits, which round into [x]'s dtype once at the end. *)

val resize_weights :
  edge:[< edge ] ->
  kernel ->
  int ->
  int ->
  Nx.host Nx.int64_t * Nx.host Nx.float64_t
(** [resize_weights ~edge k n n'] is the table {!resize} applies to an axis of
    [n] pixels resized to [n']: for each new pixel a row of [t] taps, [t] the
    most any row has, as the old pixel each reads, folded into [\[0, n)] by
    {!Nx.extension_index}, and its weight. A row lists its taps in increasing
    [i] before folding, then [-1] with weight [0] up to [t]; a tap that reads
    zero under [`Zero], is dropped by [`Renormalise] or weighs [0] is left out.

    Raises [Invalid_argument] if [n'] is below [1], [n] is below [1], or [k] is
    invalid ({!type-kernel}). *)

val resize :
  edge:[< edge ] ->
  kernel ->
  int * int ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t
(** [resize ~edge k (h', w') x] is [x] resized to [w'] columns, then to [h']
    rows: each new pixel the sum of its taps' pixels times their weights from
    {!resize_weights}, added in the row's order. An axis whose extent is
    unchanged is left as it is. A pixel reaches only the new pixels that weigh
    it. The result is linear in [x], with its derivative; the sizes have none.

    Raises [Invalid_argument] if [x] has fewer than three axes, no row or no
    column, if [h'] or [w'] is below [1], or [k] is invalid. *)

(** {1:boxes Boxes}

    A {e box} is the rectangle [\[x₀, x₁) × \[y₀, y₁)], held as the floats
    [x₀ y₀ x₁ y₁] along a last axis of extent [4]. It is {e empty} when
    [x₁ <= x₀] or [y₁ <= y₀]. The box of the pixels in columns [j₀] to [j₁] and
    rows [i₀] to [i₁] is [j₀ i₀ (j₁ + 1) (i₁ + 1)]. A box is clipped to an image
    by {!Nx.minimum} with [[w; h; w; h]] then {!Nx.maximum_s} of [0]. Each
    function rounds each operation it states once, in the boxes' dtype. *)
module Box : sig
  val of_xywh : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [of_xywh b] is the boxes of corner [(x, y)], width [w] and height [h]:
      [x y (x + w) (y + h)]. *)

  val to_xywh : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [to_xywh b] is [x₀ y₀ (x₁ - x₀) (y₁ - y₀)]. *)

  val of_cxcywh : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [of_cxcywh b] is the boxes of centre [(cx, cy)] and size [(w, h)]:
      [(cx - ½w) (cy - ½h) (cx + ½w) (cy + ½h)]. *)

  val to_cxcywh : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [to_cxcywh b] is [(½(x₀ + x₁)) (½(y₀ + y₁)) (x₁ - x₀) (y₁ - y₀)]. *)

  val area : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [area b] is [(x₁ - x₀)(y₁ - y₀)], and [0] for an empty box. *)

  val iou : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [iou a b] is the intersection over union of every box of [a], [[…; n; 4]],
      with every box of [b], [[…; m; 4]], leading axes broadcast: [[…; n; m]],
      [i / ((area a + area b) - i)] for [i] the intersection's area, the product
      of its sides each clamped at [0], and [0] where the union is [0]. *)

  val of_mask : (float, 's) Nx.dtype -> 'd Nx.bool_t -> (float, 's, 'd) Nx.t
  (** [of_mask dt m] is the least box holding every pixel of each mask of [m],
      [[…; h; w; 1]] to [[…; 4]], and [0 0 0 0] for a mask with none. *)
end

(** {1:masks Masks} *)
module Mask : sig
  val iou :
    (float, 's) Nx.dtype -> 'd Nx.bool_t -> 'd Nx.bool_t -> (float, 's, 'd) Nx.t
  (** [iou dt a b] is the intersection over union of every mask of [a],
      [[…; n; h; w; 1]], with every mask of [b], [[…; m; h; w; 1]]: the pixels
      in both and in either counted exactly as integers, their quotient rounded
      once into [dt] for masks of fewer than 2²⁸ pixels, and [0] where the union
      is empty. *)
end

(** {1:rle Run-length encoding}

    COCO's run-length encoding of a mask [[h; w; 1]]: the lengths of the runs of
    equal pixels in column-major order from pixel [(0, 0)], the first a run of
    [false] pixels, possibly empty. Runs are host values, since their number
    depends on the mask. *)
module Rle : sig
  type t = private { height : int; width : int; counts : int array }
  (** The type for encodings. *)

  val v : height:int -> width:int -> int array -> t
  (** [v ~height ~width counts] is the encoding of those runs.

      Raises [Invalid_argument] unless [height] and [width] are non-negative and
      the counts are non-negative and sum to [height * width]. *)

  val encode : Nx.host Nx.bool_t -> t
  (** [encode m] is the encoding of the mask [m].

      Raises [Invalid_argument] unless [m]'s axes are [[h; w; 1]]. *)

  val decode : t -> Nx.host Nx.bool_t
  (** [decode r] is the mask [r] encodes. *)

  val area : t -> int
  (** [area r] is the number of [true] pixels: the sum of the odd-indexed
      counts. *)

  val to_string : t -> string
  (** [to_string r] is COCO's compressed counts: each count after the third as
      its signed difference from the count two before, the others as they are,
      each in groups of five bits from the low end with a continuation bit, a
      character each from ASCII 48. *)

  val of_string : height:int -> width:int -> string -> t
  (** [of_string ~height ~width s] reads {!to_string}'s form.

      Raises [Invalid_argument] if [s] is not such a string, or its counts are
      negative or do not sum to [height * width]. *)
end

(** {1:selection Selection}

    These decide which elements to keep or how to group them. Their results are
    booleans or integers and have no derivative; the values they select keep
    theirs. Each repeats a step under {!Rune.iterate} until nothing changes, so
    a compiled call runs as many steps as its input needs. *)

val nms :
  ?classes:'d Nx.int64_t ->
  iou:float ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  'd Nx.bool_t
(** [nms ~classes ~iou b s] is which boxes of [b], [[…; n; 4]], scored [s],
    [[…; n]], greedy suppression keeps, in [b]'s order. By descending score,
    equal scores ([-0.] and [0.] among them) lowest position first and a NaN
    first, a box is kept unless a kept box before it has a {!Box.iou} with it,
    in the boxes' dtype, greater than the real number [iou]. With [classes],
    [[…; n]], only boxes of one class suppress each other. Each set of leading
    axes selects on its own.

    It repeats a step over the [n × n] pairs, deciding every box from those kept
    so far, until none changes: at most [2L + 1] and [n + 1] steps, [L] the
    longest chain of boxes each deciding the next; tens on dense detections, [n]
    on a chain.

    Raises [Invalid_argument] if [b]'s last axis is not [4] or the leading axes
    of [b], [s] and [classes] differ. *)

val components : [ `Four | `Eight ] -> 'd Nx.bool_t -> 'd Nx.int64_t
(** [components c m] labels the connected components of each mask of [m],
    [[…; h; w; 1]], at the same shape. Pixels are {e next} to each other when
    they share an edge, or under [`Eight] a corner; a component is a largest set
    of true pixels joined through next ones. A true pixel's label is the least
    row-major index [i w + j] in its component, a false pixel's [-1]; a batch
    padded with [false] keeps its components, labelled in the padded width.
    Areas are {!Nx.reduce_segments}[ Add ~segments:(h * w)] of ones over the
    flat labels, read back by {!Nx.take}.

    It runs Liu and Tarjan's algorithm R with two shortcuts a round (Simple
    Concurrent Connected Components Algorithms), at most
    [6 + ⌊3 log₂ (3 h w / 2)⌋] rounds over every pixel. Under {!Rune.vmap} each
    mask stops on its own.

    Raises [Invalid_argument] unless [m]'s axes are [[…; h; w; 1]]. *)

(**/**)

(** For the tests: the number of rounds each selection ran. *)

val components_rounds :
  [ `Four | `Eight ] -> 'd Nx.bool_t -> 'd Nx.int64_t * 'd Nx.int32_t

val nms_rounds :
  ?classes:'d Nx.int64_t ->
  iou:float ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  'd Nx.bool_t * 'd Nx.int32_t
