(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Raster rendering.

    Draws {{!Hugin_next_vg.Renderable}renderables} as pixels: an RGBA tensor
    with {!render}, a PNG file with {!png}. A {e density}, in device pixels per
    point, fixes the size of a pixel on the page: at density [2.], a page of 360
    by 240 points is 720 by 480 pixels, each pixel half a point wide.

    {1:accuracy Accuracy}

    Each pixel approximates the average over its square of what the picture
    paints, as {!Hugin_next_vg.Picture.section-semantics} defines it:
    - Fills, glyphs and clips are antialiased by exact coverage: a pixel takes
      the fraction of the colour equal to the fraction of its square that the
      area covers. Curves are first flattened to within a tenth of a pixel, and
      glyphs are drawn from their outlines, without hinting.
    - A stroke's outline is covered as the union of the pieces it is made of,
      one per segment, join and cap, with round ones flattened likewise. A pixel
      that the edges of two overlapping pieces cross takes the sum of their
      coverages, up to the whole pixel. A point of a subpath within a twentieth
      of a pixel, along both axes, of the point kept before it is dropped, so an
      open subpath or a dash shorter than that is drawn as its caps alone. A
      closed subpath that would keep fewer than three points keeps all of them
      but exact repeats instead, so that it has the extent of its pen at any
      size. A dash pattern whose period is below a thousandth of a pixel is
      drawn solid.
    - Primitives are composited one after another, each against what the
      previous left. Two shapes that abut along an edge thus each cover part of
      the pixels on the edge, and let a little of the backdrop through there, as
      in browsers and PDF viewers.
    - Compositing is source-over on the encoded components, premultiplied by
      alpha, in single-precision floating point. Components are rounded to 8
      bits once, when the image is read out, so that translucent primitives
      drawn over one another reach the colour they composite to: up to a
      thousand fills at any opacity from [0.02] land within a level of it.
    - An image paints a pixel with the image pixel under the pixel's centre, so
      its edges are not antialiased. Where an image is shown smaller than its
      pixels, a pixel averages the image pixels under [k] by [k] points spread
      evenly over its square, [k] being the ratio of image pixels to device
      pixels rounded up, at most [4].
    - An {{!Hugin_next_vg.Picture.opacity}opacity} is drawn into a layer of its
      own, then composited as one primitive.
    - The instances of a {{!Hugin_next_vg.Picture.stamp}stamp} are placed at
      their positions rounded to the nearest quarter of a pixel in each
      direction, an error of at most an eighth of a pixel.
    - A picture under transforms whose composition, with the density, has no
      inverse or overflows the range of floats paints nothing.
    - Tags are ignored. *)

(** {1:render Rendering} *)

val render : density:float -> Hugin_next_vg.Renderable.t -> Nx.uint8_t
(** [render ~density r] is [r] drawn as a [[|h; w; 4|]] tensor of RGBA pixels
    with straight alpha, on the host, where [w] and [h] are the width and height
    of the page of [r] multiplied by [density] and rounded to the nearest
    integer. Pixel [(i, j)] shows the square of the page with top left corner
    [(float j /. density, float i /. density)] and side [1. /. density]. Each
    component is the level nearest its composited value. Pixels the picture
    paints nothing on, and those whose alpha rounds to [0], are [(0, 0, 0, 0)].

    Raises [Invalid_argument] if [density] is not finite and positive, or if [w]
    or [h] is less than [1] or greater than [2{^31} - 1], and what reading the
    tensor of an image of [r] raises, such as for a tensor whose storage a
    compiled call consumed. *)

val png : density:float -> Hugin_next_vg.Renderable.t -> string
(** [png ~density r] is the PNG file of [render ~density r]: 8-bit RGBA, with a
    [pHYs] chunk of [Float.round (density *. 72. /. 0.0254)] pixels per metre in
    both directions, so that viewers and printers show it at the size of the
    page of [r], and an [sRGB] chunk with the perceptual rendering intent.

    Raises [Invalid_argument] as {!render} does, and if that number of pixels
    per metre is not in \[[1];[2{^31} - 1]\], the range of a PNG integer, which
    takes a density below about [0.00018] or above about [758,000]. *)
