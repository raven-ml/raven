(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Raster rendering.

    Draws {{!Hugin_vg.Renderable}renderables} as pixels: an RGBA tensor
    with {!val-render}, a PNG file with {!png}. A {e density}, in device pixels
    per point, fixes the size of a pixel on the page: at density [2.], a page of
    360 by 240 points is 720 by 480 pixels, each pixel half a point wide.

    {1:accuracy Accuracy}

    Each pixel approximates the average over its square of what the picture
    paints, as {!Hugin_vg.Picture.section-semantics} defines it:
    - Fills, glyphs and clips are antialiased by exact coverage: a pixel takes
      the fraction of the colour equal to the fraction of its square that the
      area covers. Curves are first flattened to within a tenth of a pixel
      wherever they meet the page, at any size, and glyphs are drawn from their
      outlines, without hinting.
    - A clipped primitive takes in each pixel the product of its coverage and
      the clip's. One whose box, as {!Hugin_vg.Picture.val-bounds} takes it,
      does not meet the boxes of the paths of the clips above it, or only
      touches them, to within rounding, paints nothing.
    - A stroke's outline is covered as the union of the pieces it is made of,
      one per segment, join and cap, with round ones flattened likewise. A
      stroke's curves are flattened in its own coordinates, to a tenth of a
      pixel under the largest stretch of the transforms and the density: a
      solid stroke's wherever they reach the page, at any size, when its pen
      reaches no farther from its path than [65536²] tenths of a pixel, about
      [4e8] pixels, where its curves near the page take no more chords than a
      whole cubic cut whole; a dashed one's, whose dashes are measured along
      the chords, and a wider pen's, each into at most 65536 chords, which
      keeps a curve within that while its control points lie within [1e8]
      pixels of each other. Dashes are placed along a subpath by
      their distance from its start, to within the spacing of floats at that
      distance, which is under a pixel below [2{^53}] pixels of path. A pixel
      that the edges of two overlapping pieces cross takes the sum of their
      coverages, up to the whole pixel. A point of a subpath within a twentieth
      of a pixel, along both axes, of the point kept before it is dropped, so an
      open subpath or a dash shorter than that is drawn as its caps alone.
      Dashes are cut from the points a subpath keeps, so their lengths leave out
      the arc through dropped points, a loss of second order on a smooth curve,
      and a dash's end within that distance of a kept point moves to it. A
      solid closed subpath that would keep fewer than three points keeps all of
      them but exact repeats instead, so that it has the extent of its pen at
      any size. A dash pattern whose period, along a segment, is below a
      thousandth of a pixel is drawn solid there.
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
      its edges are not antialiased. A centre on an edge of the image, to within
      the rounding of its position, is under the image, and one on the boundary
      between two of its pixels takes the right or lower one. A centre off such
      a line by less than about [2{^-46} r] along the image's axis, in the
      image's units, may fall on either side of it, and which side may change
      with the picture's place on the page and the page's size: an image pixel
      shown thinner than that may show the colour of another within that
      distance. For the map [m] from the image's coordinates to device pixels
      and its inverse [n], [r] is [(|n.xx| + |n.xy|) (W + T) + c X] along x and
      [(|n.yx| + |n.yy|) (W + T) + c Y] along y. [W] is twice the page's larger
      side in pixels, plus, for each stamp the image is in, [2{^18}] and twice
      the farthest the stamp's picture reaches from an instance's position along
      either axis, in pixels, since a stamp draws its instances on tiles of up
      to [2{^16}] pixels around them. [T] is the largest magnitude, in pixels,
      of the terms that the translations of [m] and of the maps it is composed
      from sum, from the density down through the transforms and stamps above
      the image: a map that moves far and back rounds at the far point. [X] and
      [Y] are the largest magnitudes of the box's coordinates along each axis,
      and [c = (|m.xx m.yy| + |m.xy m.yx|) / |m.xx m.yy - m.xy m.yx|] grows as
      [m] nears having no inverse. Where an image is shown smaller than its
      pixels, a pixel averages the image pixels under [k] by [k] points spread
      evenly over its square, [k] being the ratio of image pixels to device
      pixels rounded up, at most {!image_samples}.
    - An {{!Hugin_vg.Picture.opacity}opacity} is drawn into a layer of its
      own, then composited as one primitive.
    - The instances of a {{!Hugin_vg.Picture.stamp}stamp} are placed at
      their positions rounded to the nearest quarter of a pixel in each
      direction, an error of at most an eighth of a pixel.
    - Points are mapped to the device and edges cut to the page in double
      precision, so that an edge lies within about [2{^-50} d] pixels of where
      it should, [d] being the largest magnitude, in pixels, of the terms that
      map the nearer of its ends to the device (each coordinate times its
      factor, and the translation), for an edge of a stroke's outline, of its
      pen's half width and of the point of its path it is drawn about, and,
      for an edge of a dash, of the dash's distance from the start of its
      subpath. A segment is drawn along its own
      line however far the transforms and the density map its points, beyond
      the range of floats included: it is cut in the coordinates of its path,
      about [2{^1000}] pixels beyond the page. An open subpath that comes back
      from there starts its dash pattern afresh, and a closed one carries it
      across a run along the cut about [2{^1000}] pixels long.
    - A picture under transforms whose composition, with the density, has no
      inverse or overflows the range of floats paints nothing.
    - Tags are ignored. *)

(** {1:render Rendering} *)

val image_samples : int
(** [image_samples] is [4], the most points per axis that a pixel averages an
    image shown smaller than its pixels at: the image pixels under
    [(p + (i + 0.5) / 4)], for device pixel [p] and [i] in \[[0];[3]\], along
    each axis ({!section-accuracy}). *)

val render : density:float -> Hugin_vg.Renderable.t -> Nx.host Nx.uint8_t
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

val png : density:float -> Hugin_vg.Renderable.t -> string
(** [png ~density r] is the PNG file of [render ~density r]: 8-bit RGBA, with a
    [pHYs] chunk of [Float.round (density *. 72. /. 0.0254)] pixels per metre in
    both directions, so that viewers and printers show it at the size of the
    page of [r], and an [sRGB] chunk with the perceptual rendering intent.

    Raises [Invalid_argument] as {!val-render} does, and if that number of
    pixels per metre is not in \[[1];[2{^31} - 1]\], the range of a PNG integer,
    which takes a density below about [0.00018] or above about [758,000]. *)
