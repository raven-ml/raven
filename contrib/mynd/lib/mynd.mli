(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Images, positions in them and their annotations, as nx arrays.

    An {e image} is an [Nx.t] whose last three axes [[h; w; c]] hold [h] rows
    from the top, [w] columns from the left and [c] channels. The other axes
    are {e batch axes}, each index of which is an independent image
    ({!section-batches}). Images of different sizes go in a list. A {e mask}
    is a boolean image of one channel, [[…; h; w; 1]]. The functions on
    images of every dtype, such as {!Filter.median}, take masks.

    An integer of [n] bits holds a {e code} [k], which stands for a fraction of
    {e full scale}, the range [[0, 1]] for an unsigned integer and [[-1, 1]] for
    a signed one, as Vulkan's normalized fixed-point formats read it. The
    fraction is [k / (2ⁿ - 1)] unsigned (UNORM) and [max (k / (2ⁿ⁻¹ - 1)) (-1)]
    signed (SNORM). A float holds the fraction itself. {!convert} changes the
    dtype and keeps the fraction. No other function clamps a value to these
    ranges.

    An image holds the values its caller gives it. Values are {e linear} when
    they are proportional to light, and {e encoded} when a
    {e transfer function}, such as sRGB's, has mapped them to spend more codes
    on dark tones. Decoders return a file's encoded values, which for most
    photographs are sRGB. Alpha, where a file has it, is the last channel and
    {e straight}, in the sense that the colours are not multiplied by it.
    Every function computes on those numbers, so resizing or blending sRGB
    values averages encoded numbers. To average light, apply
    {!srgb_to_linear} first and {!linear_to_srgb} after. To interpolate an
    image with alpha, apply {!premultiply} first and {!unpremultiply} after, or
    the colour of transparent pixels bleeds into visible ones.

    Coordinates are continuous, [x] to the right and [y] down, in pixels.
    Pixel [(i, j)], row [i] and column [j], is the {e cell}
    [\[j, j + 1) × \[i, i + 1)], centred at [(j + ½, i + ½)]. An [h] by [w]
    image covers [\[0, w) × \[0, h)], and scaling an image by [s] scales its
    coordinates by [s]. A {e position} is a point [(x, y)] held along a last
    axis of extent [2], [x] first. Positions and boxes use these coordinates.

    {!Colour} reads the colours of pixels. {!Sample} reads images at positions
    and resizes them, and {!Transform} maps positions, which is how an image is
    warped. {!Filter} computes each pixel from the pixels about it. {!Box} and
    {!Mask} hold annotations and select among them, and {!Rle} encodes masks
    as COCO (Common Objects in Context) does. An array whose type names
    [Nx.host] lies on the host. {!Rune} differentiates, batches and compiles
    every function that takes an array, with two exceptions. {!Rle}'s
    functions read and build host arrays in OCaml. {!Filter.erode} and
    {!Filter.dilate} read the boolean array that selects their pixels on the
    host, so it cannot be traced or batched; they differentiate in the
    image.

    Unless a function's doc says otherwise:
    {ul
    {- it computes on the device set that holds its arguments
       ({!Nx.section-devices}), in their dtype;}
    {- it is differentiable where its result is a float that depends
       continuously on a float argument;}
    {- it raises [Invalid_argument] at the call where an argument's shape is
       not the one its doc gives, such as an image of fewer than three
       axes;}
    {- if it repeats a {e round}, a step over all of its elements, until
       nothing changes, it does so under {!Rune.iterate}, so a compiled call
       runs as many rounds as its input needs.}}

    A function {e computes wide} if it computes on values of a float format
    narrower than 32 bits in float32 throughout, intermediate results
    included, and rounds its result once into that format at the end. Its
    derivative computes wide too.

    This averages two photographs of 8-bit sRGB codes, three channels each,
    in linear light.
    {[
      let mean a b =
        let linear x = Mynd.srgb_to_linear (Mynd.convert Nx.float32 x) in
        let m = Nx.mul_s (Nx.add (linear a) (linear b)) 0.5 in
        Mynd.convert Nx.uint8 (Mynd.linear_to_srgb m)
    ]} *)

(** {1:values Conversions}

    These change the dtype of an image's values, their transfer function and
    the form of their alpha. *)

val convert : ('a, 'b) Nx.dtype -> ('v, 'r, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [convert dt x] is [x] in [dt], each value keeping its fraction, and [x]
    itself where [dt] is [x]'s dtype. With [m] the largest code:
    - Integer to float is [k / m], rounded once into [dt] for codes of up to
      16 bits, and else at float64, then into [dt]. A signed
      [-m - 1] gives [-1].
    - Float to integer is [x] clamped to [[0, 1]] ([[-1, 1]] signed), times
      [m] in float32, or in float64 for a float64 [x] or an [m] past 24 bits,
      rounded to even. NaN gives [0], [1] gives [m] and, signed, [-1] gives
      [-m].
    - Integer to integer is the nearest code, computed exactly, never a
      tie. A negative fraction into an unsigned [dt] gives [0].
    - Float to float is {!Nx.cast}, the only pair with a derivative.

    A code of [Nx.uint4] or [Nx.int4] survives a round trip through
    float8_e4m3fn and every float with more significand bits, one of 8 bits
    through float16, bfloat16 and wider, one of 16 bits through float32 and
    float64, and one of 32 bits through float64.

    Raises [Invalid_argument] if either dtype is boolean or complex. *)

val srgb_to_linear : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [srgb_to_linear x] is the linear light of the sRGB-encoded values [x], by
    CSS Color 4's extended transfer function, [x / 12.92] where
    [|x| <= 0.04045] and [sign x ((|x| + 0.055) / 1.055){^2.4}] elsewhere. It
    converts every channel, so an alpha channel, which is linear, must be split
    off first. *)

val linear_to_srgb : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [linear_to_srgb x] inverts {!srgb_to_linear}. It is [12.92 x] where
    [|x| <= 0.0031308] and [sign x (1.055 |x|{^1/2.4} - 0.055)] elsewhere. *)

val premultiply : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [premultiply x] is [x] with every channel but the last multiplied by the
    last, its alpha. To work in linear light, a program applies
    {!srgb_to_linear} before [premultiply] and {!linear_to_srgb} after
    {!unpremultiply}. *)

val unpremultiply : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [unpremultiply x] is [x] with every channel but the last divided by the
    last. Where alpha is [0], the colour and its derivative are [0]. *)

(** {1:batches Batches}

    A function computes each image's result from that image and the arrays
    paired with it, so the result is the same in any batch. Only {!Mask.iou}
    compares images across a batch axis. An array is {e paired} with images
    when its batch axes broadcast with theirs, and the result has the
    broadcast batch axes. Where they do not broadcast, the call raises
    [Invalid_argument]. Its doc names one of two alignments:
    {ul
    {- {e Leading}: the array's first axes, exactly as many as the batch axes
       it pairs with, broadcast with them one by one, and its other axes are
       its own, as for positions [[b…; q…; 2]].}
    {- {e Trailing}: the array's batch axes broadcast with the images' as nx
       broadcasts, aligned at the right, so either may have more, as for
       weights [[b'…; kh; kw]]. Deviations [[n]] pair with the last of two
       batch axes, and [[n; 1]] with the first.}}

    A paired array has the images' device set ({!Nx.section-devices}) by
    type, so one from another set, such as the host, is moved first
    ({!Nx.place}). *)

(** {1:padding Padding}

    A batch is {e padded} when smaller images sit at its images' top left,
    with a {e fill}, a constant value, below and to their right. A function
    {e respects} padding by a fill when each image's result on its own pixels,
    or at each position, or as a box or a number, is its result alone. Those
    of {!section-values} and {!Colour} respect every fill, and the others
    only the fills their docs name. *)

(** {1:outside Reading outside the image}

    A function that takes a rule [~edge] reads the pixels outside an image by
    that rule. Of the five {e extension rules} ({!type-extension}), [`Zero]
    reads zero and [`Replicate] the nearest pixel inside. [`Symmetric_whole]
    mirrors the image about its border pixels, [`Symmetric_half] about the
    border itself, so the border pixel repeats, and [`Periodic] repeats the
    image. {!Filter.correlate}, {!Filter.gaussian} and {!Filter.gradient} also
    take [`Antisymmetric_whole] ({!Nx.extension}), which reads
    [2 x(0) - x(i)] at [-i], so a linear ramp continues as the same line.
    {!Sample.at} and resizing also take [`Renormalise] ({!Sample.type-edge}).
    Where a read lands outside again, as past a small image, the rule applies
    again ({!Nx.extension_index}), and an axis of one pixel reads that pixel
    under every rule but [`Zero].

    Under every rule but [`Zero] and [`Renormalise], a function raises
    [Invalid_argument] for an image with no row or no column. Under those two
    it reads [0] there, so a filter of such an image is empty, unless its doc
    says it raises, as {!Sample.resize}'s does. *)

type extension =
  [ `Zero | `Replicate | `Symmetric_whole | `Symmetric_half | `Periodic ]
(** The type for extension rules. They are {!Nx.extension}'s rules except
    [`Antisymmetric_whole], which reads each pixel outside from two pixels
    inside. *)

(** {1:modules Modules} *)

module Colour : sig
  (** Colours, their luma and luminance, and their HSV and CIE Lab forms.

      A {e colour} is a last axis of extent [3] holding red, green and blue,
      in that order. The functions of this module read the colours along the
      last axis of an array of any rank, each on its own, and return a result
      for each colour. They take floats, so an integer image is
      {!Mynd.convert}ed first. A channel beyond the three, such as alpha, is
      split off with {!Nx.split_at} first and joined back with
      {!Nx.concatenate} after. A last axis other than [3] raises
      [Invalid_argument].

      {e Linear sRGB} has ITU-R BT.709's primaries and white, in CIE 1931 xy
      red [(0.640, 0.330)], green [(0.300, 0.600)], blue [(0.150, 0.060)] and
      white D65 [(0.3127, 0.3290)]. [M] is its matrix to CIE XYZ, derived from
      them in rationals as CSS Color Module Level 4 gives it, so white
      [(1, 1, 1)] has [Y = 1].

      A {e weighted sum} [wr r + wg g + wb b], with weights that sum to [1], is
      computed as [g + wr (r - g) + wb (b - g)] in the colours' dtype, so a
      grey [(v, v, v)] gives [v] exactly, except that [-0.] gives [+0.]. Where
      that form is not finite, as at a grey of infinities or where [r - g]
      overflows, it is the plain sum. {e Luma} is the weighted sum of encoded
      values that ITU-R BT.601 §2.5.1 defines, the Y of JPEG's YCbCr and the
      grey that video and image files carry. {e Luminance} is CIE's Y of linear
      sRGB, the light a colour emits relative to white.

      The {e HSV} form of a colour is Smith's hexcone (Color Gamut Transform
      Pairs, 1978). With [v] the greatest channel and [m] the least, its value
      is [v] and its saturation [(v - m) / v]. Its hue is the colour's place
      on the hexagon of hues in turns, in [\[0, 1)], red at [0], yellow at
      [1/6], green at [1/3], cyan at [1/2], blue at [2/3] and magenta at
      [5/6]. These six hues bound the hexagon's six {e sextants}. A {e grey},
      [v = m], has hue and saturation [0], except a grey of infinities
      ({!to_hsv}). An HSV array holds hue, saturation and value as three
      fractions, so {!Mynd.convert} takes it to codes as any image. {e Lab} is
      CIE 1976 L*a*b* relative to D65 ({!linear_to_lab}).

      Values outside [[0, 1]] take the same formulas ({!Mynd}). A NaN channel
      gives NaN in every channel of the result. Where channels tie, [v] and [m]
      take the derivatives rune gives [Nx.max] and [Nx.min] at a tie
      ({!Rune.section-kinks}).

      This is CIE 1976's colour difference ΔE*ab between two images of
      sRGB-encoded colours, the distance between their Lab forms. Lab reads
      linear values, so each image is converted to linear light first.
      {[
        let delta_e a b =
          let lab x = Mynd.Colour.linear_to_lab (Mynd.srgb_to_linear x) in
          let d = Nx.sub (lab a) (lab b) in
          Nx.sqrt (Nx.sum ~axes:[ -1 ] ~keepdims:true (Nx.square d))
      ]} *)

  val luma : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [luma x] is the luma of the colours [x], [[…; 3]] to [[…; 1]], their
      weighted sum by [0.299], [0.587] and [0.114]. Linear values take
      {!luminance}. *)

  val luminance : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [luminance x] is the luminance of the linear sRGB colours [x], [[…; 3]]
      to [[…; 1]], their weighted sum by the middle row of [M],
      [87098/409605], [175762/245763] and [12673/175545]. Encoded values must
      be converted by {!Mynd.srgb_to_linear} first. {!luma} weighs encoded
      values. *)

  val to_hsv : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [to_hsv x] is the HSV form of the colours [x], [[…; 3]] to [[…; 3]]. The
      hue is [(g - b) / (6 (r - m))] where [r] is greatest,
      [1/3 + (b - r) / (6 (g - m))] where [g] is and [r] is not, and
      [2/3 + (r - g) / (6 (b - m))] elsewhere, modulo [1]. Two channels tied
      for the greatest give one hue by either rule. The saturation is infinite
      where [v = 0 > m]. A colour with an infinite channel has NaN hue and
      saturation, and value [v].

      Away from greys the hue's derivative is continuous, across ties and
      across red, where the hue itself wraps from near [1] to [0]. At a grey
      the hue's and the saturation's derivatives are [0]. *)

  val of_hsv : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [of_hsv y] is the colours whose HSV forms are [y], [[…; 3]] to [[…; 3]].
      Channel [n], [5] for red, [3] for green and [1] for blue, is [v - v s c]
      with [c = min (max (min k (4 - k)) 0) 1] and [k] the remainder of
      [n + 6 h] by [6] in [\[0, 6)]. It reads any finite hue modulo [1], so
      adding [d] to the hue turns the colour by [d] turns. A NaN or infinite
      component gives NaN in every channel.

      The derivative in the hue is that of the side of a sextant boundary the
      hue lies on. On a boundary, where the computed [k] is an integer, it is
      the mean of the two one-sided derivatives, as [Nx.minimum]'s is.

      [of_hsv (to_hsv x)] is [x] to rounding for every finite colour that is
      grey or has [v ≠ 0]. Its derivative is the identity at those colours but
      greys, ties included, except where rounding moves a tied colour's hue
      onto a sextant boundary or across one. *)

  val linear_to_lab : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [linear_to_lab x] is the CIE 1976 L*a*b* coordinates of the linear sRGB
      colours [x], [[…; 3]] to [[…; 3]], relative to sRGB's own white D65, with
      no chromatic adaptation (no change of the white a viewer adapts to).
      With [(X, Y, Z)] the product of [M] and the colour, each divided by
      white's, [L* = 116 f(Y) - 16], [a* = 500 (f(X) - f(Y))] and
      [b* = 200 (f(Y) - f(Z))]. [f t] is
      [t{^1/3}] above [ε = 216/24389] and [(κ t + 16) / 116] elsewhere, with
      [κ = 24389/27], CIE's exact constants. White is [(100, 0, 0)] and black
      [(0, 0, 0)]. Colour management's ICC profiles and many image editors
      report Lab relative to D50, which differs, so a colour must be adapted
      between the two whites before Lab values relative to D50 and these are
      compared. *)

  val lab_to_linear : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [lab_to_linear y] inverts {!linear_to_lab}. [f]'s inverse is [u{^3}]
      above [6/29] and [(116 u - 16) / κ] elsewhere, followed by [M]'s inverse.
      Colours outside sRGB's gamut, the colours its primaries mix with weights
      in [[0, 1]], come back with channels below [0] or above [1]. *)
end

module Sample : sig
  (** Images read at positions through a kernel, and resized.

      A {e kernel} [k] of half-width [r] weighs a pixel by [k(τ)], where [τ]
      is the position less the pixel's centre, in pixels, and is [0] outside
      [\[-r, r)]. [2r] is a whole number. To {e sample} an image [x] at
      [(u, v)] is to read the weighted mean

      [Σ k(u - ½ - j) k(v - ½ - i) x̃(i, j) / Σ k(u - ½ - j) k(v - ½ - i)]

      over its {e taps}, the pixels whose arguments lie in [\[-r, r)]. When
      sampling these are [2r] rows [i] and [2r] columns [j], and a shrinking
      resize has more ({!section-resizing}). [x̃] reads outside the image by
      an {e edge rule} ({!type-edge}, {!Mynd.section-outside}). Where an
      axis's kernel values sum to [0], the read is [0]. Sampling reads
      each position on its own, so positions more than a pixel apart
      {e alias}, in the sense that detail finer than their spacing reads as
      coarser detail. {!resize} removes that detail as it shrinks.

      Positions [[b…; q…; 2]] are paired with images [[b…; h; w; c]] at the
      leading axes ({!Mynd.section-batches}). The axes [q…] index the
      positions read in each image, and the result is [[b…; q…; c]]. A
      {e warp} samples at positions mapped by a transform ({!Transform}). *)

  type kernel = [ `Box | `Triangle | `Cubic of float | `Lanczos of int ]
  (** The type for kernels:
      - [`Box] is [1] on [\[-½, ½)]. It reads the pixel whose cell holds the
        position.
      - [`Triangle] is [1 - |τ|] on [\[-1, 1)]. It interpolates bilinearly.
      - [`Cubic a] is Keys' cubic convolution on [\[-2, 2)],
        [(a + 2)|τ|³ - (a + 3)|τ|² + 1] up to [|τ| = 1], then
        [a|τ|³ - 5a|τ|² + 8a|τ| - 4a]. [`Cubic (-0.5)] is the Catmull-Rom
        spline, and [-0.75] is sharper.
      - [`Lanczos n] is [sinc τ sinc (τ / n)] on [\[-n, n)] ({!Nx.sinc}), for
        [n] lobes.

      A function that takes a kernel raises [Invalid_argument] for [`Cubic a]
      with [a] not finite and for [`Lanczos n] with [n < 1]. *)

  type edge = [ extension | `Renormalise ]
  (** The type for edge rules, the extension rules ({!Mynd.section-outside})
      and [`Renormalise]. [`Renormalise] reads only the pixels inside the image
      and takes the mean over their weights, along each axis. Where those
      weights sum to [0], or no pixel lies inside, the read is [0]. *)

  val at :
    edge:[< edge ] ->
    kernel ->
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t
  (** [at ~edge k x p] is the images [x], [[b…; h; w; c]], sampled under [k] at
      the positions [p], [[b…; q…; 2]]. Each kernel argument [τ], such as
      [u - ½ - j], is the exact difference rounded once into [x]'s dtype, and
      is held below [r] where it rounds up to [r]. Each axis's [2r] kernel
      values are divided by their sum, added in order, with the values of
      taps outside made [0] first under [`Renormalise]. A pixel's weight is the
      product of its row's and its column's, and the products of weights and
      pixels add in row-major order from the top left. A position with a NaN
      or infinite part reads NaN, and a part past [±2⁶²] reads as [±2⁶²]. The
      result is linear in [x] and differentiable in it. Its derivative in [p]
      is the weights' derivative, taken from the right where a part of a
      position crosses a centre, and [0] under [`Box]. It respects padding by
      zeros under [`Zero]. An invalid [k] raises as {!type-kernel} states. *)

  val nearest :
    edge:[< extension ] ->
    ('v, 'r, 'd) Nx.t ->
    (float, 's, 'd) Nx.t ->
    ('v, 'r, 'd) Nx.t
  (** [nearest ~edge x p] is, at each position [(u, v)] of [p], the pixel
      [x̃(⌊v⌋, ⌊u⌋)] whose cell holds it, for images of every dtype. It
      gathers, so labels and masks keep their values. At a finite position it
      is [at ~edge `Box x p] on floats. A position with a NaN or infinite part
      reads the dtype's zero, [false] in a mask. Its derivative in [x] scatters
      back, and in [p] it is [0]. It respects padding by zeros under
      [`Zero]. *)

  val centres : (float, 's) Nx.dtype -> int * int -> (float, 's, Nx.host) Nx.t
  (** [centres dt (h, w)] is the [[h; w; 2]] positions of the centres of an [h]
      by [w] image, [(j + ½, i + ½)] at [(i, j)]. They are exact while every
      integer up to [2 max h w] is one [dt] holds, up to 1024 pixels a side
      in float16.

      Raises [Invalid_argument] if [h] or [w] is negative. *)

  (** {1:resizing Resizing}

      To {e resize} an axis of [n] pixels to [n'] is to read it at [n'] new
      centres through a kernel stretched to the new spacing. New pixel [j]'s
      centre lies at [c = (j + ½) n / n'] in old pixels. With
      [s = max 1 (n / n')], old pixel [i] is a tap of [j] when
      [τ = (c - i - ½) / s] lies in [\[-r, r)], and its weight is [k(τ)]
      divided by the sum over [j]'s taps, those outside included, except under
      [`Renormalise], which drops them first. So a flat image resizes to
      itself under every rule but [`Zero]. Shrinking {e band-limits} the
      result, in the sense that it removes detail finer than the new spacing.
      Enlarging reads as {!at} at the new centres, to rounding, at finite
      pixels. *)

  val resize_weights :
    edge:[< edge ] ->
    kernel ->
    int ->
    int ->
    Nx.host Nx.int64_t * Nx.host Nx.float64_t
  (** [resize_weights ~edge k n n'] is the table {!resize} applies to an axis
      of [n] pixels resized to [n'], as two [[n'; t]] arrays, [t] the most taps
      any new pixel has. Each row holds a new pixel's taps, as the old pixel
      each reads, folded into [\[0, n)] by {!Nx.extension_index}, and its
      weight. A row lists its taps in increasing [i] before folding, then [-1]
      with weight [0] up to [t]. A tap that reads zero under [`Zero], is
      dropped by [`Renormalise] or weighs [0] is left out. Each [τ] is a
      quotient of two integers rounded once at float64, and each weight and
      each sum is computed at float64.

      Raises [Invalid_argument] if [n] or [n'] is below [1], or for an invalid
      [k] ({!type-kernel}). *)

  val resize :
    edge:[< edge ] ->
    kernel ->
    int * int ->
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t
  (** [resize ~edge k (h', w') x] is [x] resized to [w'] columns, then to [h']
      rows. It computes wide. Each new pixel is the sum of its taps' pixels
      times their weights from {!resize_weights}, each weight rounded once
      into the dtype [resize] computes in, added in the row's order. An axis
      whose extent is unchanged is left as it is. A tap of weight [0] is not
      read, so a pixel changes only the new pixels in which its weight is not
      [0], and a NaN pixel makes only those NaN. The result is linear in [x]
      and differentiable in it.

      Raises [Invalid_argument] if [x] has no row or no column, under every
      edge rule, if [h'] or [w'] is below [1], or for an invalid [k]
      ({!type-kernel}). *)
end

module Transform : sig
  (** Projective maps of positions, and warps.

      A {e transform} is a [[…; 3; 3]] array [m] that maps a position [(x, y)]
      to [((m₀₀ x + m₀₁ y + m₀₂) / d, (m₁₀ x + m₁₁ y + m₁₂) / d)], where
      [d = m₂₀ x + m₂₁ y + m₂₂] and each sum adds left to right. An
      {e affine} transform has the last row [0 0 1], so [d] is [1] at a finite
      position. [Nx.matmul a b] applies [b] first, and {!Nx.inv} inverts. Each
      constructor takes parameters [[…]] or [[…; 2]] and gives transforms
      [[…; 3; 3]], differentiable in them.

      A warp ({!Sample}) samples the input image at the output image's
      centres, each mapped by a transform to the input position it reads. So
      the transform a warp applies maps output positions to input ones, and a
      program that holds the map from input to output applies its inverse. A
      warp that shrinks reads positions more than a pixel apart, which alias
      unless the image is blurred first ({!Filter.gaussian}).

      This warps an image [x], [[h₀; w₀; c]], by a transform [m], [[3; 3]],
      that maps its positions to those of an image of [h] rows and [w]
      columns, so it applies [m]'s inverse. {!Nx.beside} is the placement of
      [x], whole on each of its devices, and {!Nx.place} moves the host
      centres there. For images
      [[n; h₀; w₀; c]], the centres take a leading axis of [1]
      ({!Nx.unsqueeze}), and [m] of [[n; 3; 3]] warps each image by its own
      transform.
      {[
        let warp ~edge k (h, w) m x =
          let c = Mynd.Sample.centres (Nx.dtype x) (h, w) in
          let c = Nx.place (Nx.beside x) c in
          Mynd.Sample.at ~edge k x (Mynd.Transform.apply (Nx.inv m) c)
      ]} *)

  val apply :
    (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [apply m p] is the positions [p], [[b…; q…; 2]], each mapped by its
      transform in [m], [[b…; 3; 3]], paired with [p] at the leading axes
      ({!Mynd.section-batches}). *)

  val translation : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [translation t] is the transforms that add [t], [[…; 2]], to
      positions. *)

  val scaling : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [scaling s] is the transforms that multiply [x] by the first element of
      [s]'s last axis and [y] by the second. *)

  val rotation : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [rotation a] is the transforms that turn positions by [a] radians about
      the origin, from [+x] toward [+y], which is clockwise on the image, whose
      [y] points down. *)

  val shear : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [shear s] is the transforms that add the first element of [s]'s last
      axis times [y] to [x], and the second times [x] to [y]. *)
end

module Filter : sig
  (** Filters, which compute each pixel from the pixels about it, such as
      correlation, the Gaussian blur, the median, gradients and morphology.

      The {e window} of extents [(kh, kw)] of pixel [(i, j)] is the [kh] by
      [kw] block of pixels about it, in its channel. Its pixel at {e offset}
      [(p, q)] is the one at row [i - kh / 2 + p] and column [j - kw / 2 + q],
      read by the rule [~edge] where it lies outside the image
      ({!Mynd.section-outside}). The {e anchor}, offset [(kh / 2, kw / 2)], is
      the pixel itself. So along an axis of odd extent the window is centred
      on the pixel's centre, [j + ½] or [i + ½], and along one of even extent
      on the pixel's left or top edge, [j] or [i]. A window extent below [1]
      raises [Invalid_argument].

      Weights [[b'…; kh; kw]] and the deviations [[b'…]] of {!gaussian} are
      paired with the images [[b…; h; w; c]] at the trailing axes
      ({!Mynd.section-batches}). Every function but {!windows} returns images
      of the images' last three axes. {!correlate}, {!gaussian} and
      {!gradient} also take [`Antisymmetric_whole]
      ({!Mynd.section-outside}). The functions that take floats take an
      integer image once it is {!Mynd.convert}ed. *)

  val windows :
    edge:[< extension ] -> int * int -> ('v, 'r, 'd) Nx.t -> ('v, 'r, 'd) Nx.t
  (** [windows ~edge (kh, kw) x] is the windows of extents [(kh, kw)] of the
      images [x], [[b…; h; w; c]], of any dtype, as [[b…; h; w; c; kh; kw]].
      Element [(…, i, j, l, p, q)] is the pixel at offset [(p, q)] of pixel
      [(i, j)]'s window in channel [l], with [`Zero] reading the dtype's zero,
      [false] in a mask. It is a view of one copy of [x] extended by [kh - 1]
      rows and [kw - 1] columns. An operation that stores the windows, such as
      {!Nx.sort} or a reshape of the last two axes into one, holds [kh kw]
      elements a pixel. Its derivative adds each element's derivative back to
      the pixel it reads. It respects padding by zeros under [`Zero]. *)

  val correlate :
    edge:[< Nx.extension ] ->
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t
  (** [correlate ~edge k x] is, at each pixel and channel of the images [x],
      [[b…; h; w; c]], the sum over its window's offsets [(p, q)] under [edge]
      of the pixel there times the weight [k(p, q)], for the weights [k],
      [[b'…; kh; kw]]. The weights apply unflipped to every channel. Each sum
      adds its [kh kw] products in the order and the accumulator of
      {!Nx.conv} for one channel, so it computes wide, except that under
      [`Antisymmetric_whole] a pixel read outside first rounds into [x]'s
      dtype, as {!Nx.extend} computes it. A weight of [0]
      still reads its pixel, so a NaN or infinite pixel makes NaN or infinite
      every result whose window reads it. The result is linear in [k] and in
      [x] and differentiable in both. It respects padding by zeros under
      [`Zero]. *)

  val gaussian :
    edge:[< Nx.extension ] ->
    radius:int ->
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t
  (** [gaussian ~edge ~radius sigma x] is [x] correlated ({!correlate}) with
      the [2 radius + 1] weights [g(n) = e{^-n²/2σ²} / Σ{_m} e{^-m²/2σ²}], [n]
      and [m] from [-radius] to [radius] and [σ] the deviations [sigma], as a
      row [[b'…; 1; 2 radius + 1]], then as a column [[b'…; 2 radius + 1; 1]],
      at float32 throughout for a narrower [x], weights and reads outside
      included, rounded once into [x]'s dtype. These are the Gaussian of
      deviation [σ] at the pixels, scaled to sum to [1], so a
      constant image stays constant up to rounding under every rule but
      [`Zero]. They depend on [σ²], an infinite [σ] makes them equal, and a
      NaN [σ] gives NaN. Where [σ] is so small that every weight but the
      centre's is [0], [σ = 0] included, the weights are the impulse, with
      derivative [0] in [σ]. It respects padding by zeros under [`Zero].

      Each pass differs from the untruncated correlation with the samples at
      every integer by at most [2 erfc (radius / (σ √2))] times the largest
      magnitude it reads (Getreuer, "A Survey of Gaussian Convolution
      Algorithms", 2013, Theorem 1). The samples' gain at a frequency of [ω]
      radians a pixel is the continuous Gaussian's [e{^-σ²ω²/2}] to within
      [e{^-π²σ²/2}], so below [σ] near [1] it blurs less than [σ] states.

      Raises [Invalid_argument] if [radius] is negative. *)

  val median :
    edge:[< extension ] -> int * int -> ('v, 'r, 'd) Nx.t -> ('v, 'r, 'd) Nx.t
  (** [median ~edge (kh, kw) x] is, at each pixel and channel of the images
      [x], the lower median of the values at its window's [n = kh kw] offsets
      under [edge], in {!Nx.sort}'s order, which orders complex values by real
      part, then imaginary part. That is the value with [(n - 1) / 2]
      values before it, equal values ordered row-major by offset. It takes
      every dtype. The result is that value, and its derivative goes to the
      pixel it was read from. On a mask it is [true] where more than half the
      window is [true]. NaN sorts above every number, so the result is NaN only
      where more than half the window's values are NaN. It respects padding by
      zeros under [`Zero]. For images of at least [kh kw] pixels it holds one
      extended copy of [x], the result, about twice as many further elements
      as [x] has, and one [int64] position for each element of [x]. *)

  (** {1:gradients Gradients and edges}

      The {e gradient} of an image at a pixel and channel is the pair of its
      derivatives along [x] and along [y], in values per pixel. [gx] is
      positive where values grow to the right, and [gy] where they grow
      downward. Its {e strength} is [Nx.hypot gx gy], and its direction
      [Nx.atan2 gy gx] turns from [+x] toward [+y], clockwise on the image.

      A Gaussian derivative of deviation [σ] is the [`Central] gradient of
      {!gaussian}'s result (Lindeberg, "Discrete approximations of Gaussian
      smoothing and Gaussian derivatives", 2024, §6.1). Canny's edges ("A
      Computational Approach to Edge Detection", 1986) are the
      {!gradient_maxima} of such a gradient whose strength is above a low
      threshold, kept where they join one above a high threshold through
      pixels that share a side or a corner ({!Mask.reconstruct} with
      [`Eight]). *)

  type stencil = [ `Central | `Sobel | `Scharr ]
  (** The type for gradient stencils. Each takes the difference [-½ 0 ½] along
      one axis, which reads a linear ramp's slope, after weights along the
      other axis that sum to [1]:
      - [`Central] takes the difference alone.
      - [`Sobel] smooths by [¼ ½ ¼] first.
      - [`Scharr] smooths by [3/16 10/16 3/16] first.

      Smoothing makes the gradient's direction more accurate on fine detail.
      [`Scharr]'s weights make it the most accurate. *)

  val gradient :
    edge:[< Nx.extension ] ->
    [< stencil ] ->
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t * (float, 's, 'd) Nx.t
  (** [gradient ~edge s x] is the gradient [(gx, gy)] of each channel of the
      images [x], [[b…; h; w; c]], under the stencil [s], each of [x]'s shape.
      [gx] is [x] smoothed down each column by [s]'s weights, then half the
      difference of the pixels right and left of each pixel. [gy] is [x]
      smoothed along each row, then half the difference of the pixels below
      and above. Each pass reads outside the image under [edge] as
      {!correlate} does, and each outer weight multiplies the sum of its two
      pixels. Under [`Antisymmetric_whole] the difference at the border is
      one-sided, [x(1) - x(0)] at column [0]. Flipping [x] along an axis flips
      [(gx, gy)] and negates the component along that axis, and transposing
      [x] transposes and swaps them, both exactly but for the sign of a zero
      or a NaN. It computes wide, and respects padding by zeros under [`Zero].
      The result is linear in [x] and differentiable in it. *)

  val gradient_maxima :
    (float, 's, 'd) Nx.t * (float, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [gradient_maxima (gx, gy)] is, for each pixel and channel of the
      gradients [(gx, gy)], [[b…; h; w; c]], [true] where their strength [m]
      is a maximum along the gradient. It is a mask where [c] is [1]. At
      pixel [p] of gradient [g], let [u] be [g] divided by its larger absolute
      component, which moves [p] onto the ring of the eight pixels around it.
      The strengths at [p - u] and [p + u] are interpolated linearly between
      the two ring pixels each lies between, and a position outside the image
      is not compared. [p] is kept where [m(p)] is above the strength at
      [p - u] and at least the one at [p + u]. This is Canny's interpolation
      (Grompone von Gioi and Randall, "A Sub-Pixel Edge Detector: an
      Implementation of the Canny/Devernay Algorithm", 2017, §2). So where two
      pixels that share a side or a corner have equal strengths, and the
      gradient of one points away from the other along the line through them,
      that one is not kept. A pixel is not kept where its [m] is [0] or a
      strength it reads is NaN. Where [m] is constant and not [0], as on a
      linear ramp, a pixel whose [p - u] lies outside the image is kept, and
      rounding decides which others are. Formats narrower
      than 32 bits form [m] at float32.

      Raises [Invalid_argument] if [gx] and [gy] differ in shape. *)

  (** {1:morphology Morphology}

      A {e footprint} [fp] is a host boolean array [[kh; kw]] that selects the
      offsets [(p, q)] of a pixel's window where it is [true]. Its extents may
      be even, and it may leave out the anchor. A selected offset outside the
      image is left out, and where none lies inside, as for a footprint of
      extent [0], {!erode} gives the dtype's greatest value
      ({!Nx.Dtype.max_value}, [+∞] for a float with infinities) and {!dilate}
      its least. An image with no row or no column gives an empty result.

      For images without NaN, [erode fp] and [dilate fp] are {e adjoint}, in
      the sense that [dilate fp y <= x] at every pixel exactly when
      [y <= erode fp x] at every pixel. An edge rule would break this at the
      border, so these functions take none. *)

  val erode : Nx.host Nx.bool_t -> ('v, 'r, 'd) Nx.t -> ('v, 'r, 'd) Nx.t
  (** [erode fp x] is, at each pixel and channel of the images [x],
      [[b…; h; w; c]], the least of the pixels at the offsets [fp] selects that
      lie inside the image, in {!Nx.minimum}'s order, and NaN where a pixel
      read is NaN. A mask erodes to [true] where every such pixel is [true].
      The result is a value of [x] or the dtype's greatest, so its bits, but
      for a NaN's sign and payload, do not depend on the order in which the
      pixels are compared. Its derivative comes from the pixels read that
      equal the result, in equal shares as {!Nx.min}'s does, and is [0] where
      the result is NaN. The derivative computes wide. It respects padding by
      the dtype's greatest value, [true] in a mask.

      Raises [Invalid_argument] if [x] is complex, a dtype with no greatest
      value ({!Nx.Dtype.max_value}). *)

  val dilate : Nx.host Nx.bool_t -> ('v, 'r, 'd) Nx.t -> ('v, 'r, 'd) Nx.t
  (** [dilate fp x] is, at each pixel [(i, j)] and channel of the images [x],
      the greatest of the pixels inside the image at row [i + kh / 2 - p] and
      column [j + kw / 2 - q] for the offsets [(p, q)] [fp] selects, which is
      the window reflected about the pixel. For a footprint symmetric about
      its anchor, [dilate] reads the pixels {!erode} reads. A footprint [true]
      in the first row or column of an even axis, a full rectangle of an even
      extent included, is asymmetric about its anchor. For every footprint,
      the greatest of the pixels {!erode} reads is [dilate] by [fp] flipped
      ({!Nx.flip}) with a row or column of [false] put before each even axis.
      Order, bits and derivative are {!erode}'s, with greatest for least. It
      respects padding by the dtype's least value, [false] in a mask.

      Raises [Invalid_argument] as {!erode} does. *)
end

module Box : sig
  (** Boxes, their conversions and overlaps, and non-maximum suppression.

      A {e box} is the rectangle [\[x₀, x₁) × \[y₀, y₁)], held as the floats
      [x₀ y₀ x₁ y₁] along a last axis of extent [4]. It is {e empty} when
      [x₁ <= x₀] or [y₁ <= y₀]. The box of the pixels in columns [j₀] to [j₁]
      and rows [i₀] to [i₁] is [j₀ i₀ (j₁ + 1) (i₁ + 1)]. A box is clipped to
      an image by {!Nx.minimum} with [[w; h; w; h]] then {!Nx.maximum_s} of
      [0]. Each operation a function's doc states is rounded once, in the
      boxes' dtype.

      {e Non-maximum suppression} takes the boxes in descending order of score
      and keeps
      each box unless a box kept before it has an intersection over union
      ({!iou}) with it above a threshold. *)

  val of_xywh : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [of_xywh b] is the boxes of corner [(x, y)], width [w] and height [h],
      [x y (x + w) (y + h)]. *)

  val to_xywh : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [to_xywh b] is [x₀ y₀ (x₁ - x₀) (y₁ - y₀)]. *)

  val of_cxcywh : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [of_cxcywh b] is the boxes of centre [(cx, cy)] and size [(w, h)],
      [(cx - ½w) (cy - ½h) (cx + ½w) (cy + ½h)]. *)

  val to_cxcywh : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [to_cxcywh b] is [(½(x₀ + x₁)) (½(y₀ + y₁)) (x₁ - x₀) (y₁ - y₀)]. *)

  val area : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [area b] is [(x₁ - x₀)(y₁ - y₀)], and [0] for an empty box. *)

  val iou : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [iou a b] is the intersection over union of every box of [a],
      [[…; n; 4]], with every box of [b], [[…; m; 4]], as [[…; n; m]], the
      leading axes broadcast as nx broadcasts. It is
      [i / ((area a + area b) - i)], with [i] the intersection's area, the
      product of its sides each clamped at [0], and [0] where the union is
      [0]. *)

  val of_mask : (float, 's) Nx.dtype -> 'd Nx.bool_t -> (float, 's, 'd) Nx.t
  (** [of_mask dt m] is the least box holding every [true] pixel of each mask
      of [m], [[…; h; w; 1]] to [[…; 4]], and [0 0 0 0] for a mask with none.
      It respects padding by [false]. *)

  val transform :
    (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [transform m b] is each box of [b], [[b…; q…; 4]], moved by its
      transform in [m], [[b…; 3; 3]], paired with [b] at the leading axes
      ({!Mynd.section-batches}). It spans the least to the greatest
      coordinates of the box's corners as {!Transform.apply} maps them. Where
      the corners' [d] ({!Transform}) are all positive or all negative, these
      are the bounds of the closed box's image, to the corners' rounding.

      An empty box gives [0 0 0 0], whatever [m]. A box whose corners' [d]
      differ in sign or include [0] meets the line [d = 0], which [m] sends
      to infinity, so its image is unbounded and it gives NaN in all four.
      So does a box that is not empty and has a NaN or infinite coordinate.

      Moving by [a], then by [c], can give a larger box than moving once by
      [Nx.matmul c a], as two turns by 45° double a square's side. *)

  val nms :
    ?classes:'d Nx.int64_t ->
    iou:float ->
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t ->
    'd Nx.bool_t
  (** [nms ~classes ~iou b s] is, for each box of [b], [[…; n; 4]], whether
      non-maximum suppression by the scores [s], [[…; n]], keeps it, as
      [[…; n]]. Equal scores, [-0.] and [0.] among them, go lowest position
      first, and a NaN score goes first. A kept box suppresses a later box
      when their {!iou}, computed in the boxes' dtype, is above the threshold
      [iou], compared exactly with the real number [iou] and never rounded. A
      NaN IoU suppresses nothing. With [classes], [[…; n]], only boxes of one
      class suppress each other. Each index of the leading axes selects among
      its own [n] boxes.

      A box {e conflicts} with a later one when it would suppress it if kept.
      With [L] the longest chain of boxes each conflicting with the next,
      [nms] takes at most [min (2L + 1) (n + 1)] rounds.

      Raises [Invalid_argument] if the leading axes of [b], [s] and [classes]
      differ. *)

  (**/**)

  (** For the tests: the number of rounds {!nms} ran. *)

  val nms_rounds :
    ?classes:'d Nx.int64_t ->
    iou:float ->
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t ->
    'd Nx.bool_t * 'd Nx.int32_t
end

module Mask : sig
  (** Masks ({!Mynd}), their overlaps, components and reconstruction.

      Pixels are {e next} to each other when they share a side, or under
      [`Eight] a corner ({!type-connectivity}). A {e component} of a mask is a
      largest set of its [true] pixels joined through next ones. The
      {e labels} of masks [[…; h; w; 1]] are an int64 array of their shape,
      where a [true] pixel's label is the least row-major index [i w + j] in
      its component and a [false] pixel's is [-1]. A label indexes its own
      mask's pixels flattened to [[…; h w]], so each mask counts in its own
      labels. {!Nx.cast} makes a mask a float image of [0] and [1]. *)

  type connectivity = [ `Four | `Eight ]
  (** The type for connectivities. Under [`Four] pixels are next when they
      share a side, and under [`Eight] also when they share a corner. *)

  val iou :
    (float, 's) Nx.dtype -> 'd Nx.bool_t -> 'd Nx.bool_t -> (float, 's, 'd) Nx.t
  (** [iou dt a b] is the intersection over union of every mask of [a],
      [[…; n; h; w; 1]], with every mask of [b], [[…; m; h; w; 1]], as
      [[…; n; m]], the leading axes broadcast as nx broadcasts. It counts the
      pixels in both and in either exactly, as integers, and rounds their
      quotient once into [dt] for masks of fewer than 2²⁸ pixels. For larger
      masks the quotient is rounded at float64, then into [dt]. It is [0]
      where the union is empty, and respects padding by [false]. *)

  val components : connectivity -> 'd Nx.bool_t -> 'd Nx.int64_t
  (** [components c m] is the labels of the components of each mask of [m],
      [[…; h; w; 1]], with pixels next under [c]. A batch padded with [false]
      keeps each mask's components, labelled in the padded width [w']. A label
      [l] there is [(l / w') w + l mod w'] in the mask's own width [w].

      It repeats a round over every pixel until no label changes, at most
      [6 + ⌊3 log₂ (3 h w / 2)⌋] rounds (Liu and Tarjan, "Simple Concurrent
      Connected Components Algorithms", Lemma 23). Under {!Rune.vmap} each
      mask stops on its own. *)

  val reconstruct :
    connectivity -> marker:'d Nx.bool_t -> 'd Nx.bool_t -> 'd Nx.bool_t
  (** [reconstruct c ~marker m] is the masks [m], [[…; h; w; 1]], keeping only
      the components under [c] that hold a [true] pixel of [marker] in the
      same mask. This is the reconstruction of [marker] within [m] (Heijmans,
      "Introduction to Connected Operators", 1999, §6.2), [marker] grown within
      [m] through next pixels until nothing changes. A marker pixel outside [m]
      is ignored. It takes at most {!components}' bound of rounds, however far
      the growth reaches, and respects padding by [false].

      Raises [Invalid_argument] unless [marker] has [m]'s shape. *)

  (**/**)

  (** For the tests: the number of rounds {!components} ran. *)

  val components_rounds :
    connectivity -> 'd Nx.bool_t -> 'd Nx.int64_t * 'd Nx.int32_t
end

module Rle : sig
  (** COCO's run-length encoding of masks.

      The encoding of a mask [[h; w; 1]] is the lengths of the runs of equal
      pixels in column-major order from pixel [(0, 0)], the first a run of
      [false] pixels, possibly empty. Runs are host values, since their number
      depends on the mask, so a batch or a mask on a device is encoded one
      host mask at a time. *)

  type t = private { height : int; width : int; counts : int array }
  (** The type for encodings. *)

  val v : height:int -> width:int -> int array -> t
  (** [v ~height ~width counts] is the encoding of those runs.

      Raises [Invalid_argument] unless [height] and [width] are non-negative
      and the counts are non-negative and sum to [height * width]. *)

  val encode : Nx.host Nx.bool_t -> t
  (** [encode m] is the encoding of the mask [m], [[h; w; 1]]. *)

  val decode : t -> Nx.host Nx.bool_t
  (** [decode r] is the mask [r] encodes. *)

  val area : t -> int
  (** [area r] is the number of [true] pixels, the sum of the odd-indexed
      counts. *)

  val to_string : t -> string
  (** [to_string r] is COCO's compressed counts. Each count after the third is
      written as its signed difference from the count two before, and the
      others as they are. Each is written in groups of five bits from the low
      end, and each group, with a continuation bit above it, as the character
      whose code is [48] plus the six bits. *)

  val of_string : height:int -> width:int -> string -> t
  (** [of_string ~height ~width s] reads {!to_string}'s form.

      Raises [Invalid_argument] if [s] is not such a string, or its counts are
      negative or do not sum to [height * width]. *)
end
