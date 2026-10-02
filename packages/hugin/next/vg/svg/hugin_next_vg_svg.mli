(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** SVG rendering.

    Writes {{!Hugin_next_vg.Renderable}renderables} as self-contained SVG 2
    documents: text stays text, fonts and images are embedded, and every picture
    maps to the elements below.

    {1:mapping Pictures as SVG}

    - The document is [w] by [h] points ([pt] units) with a [viewBox] of one
      user unit per point, for a renderable of [w] by [h].
    - Fills and strokes are [path] elements, with the fill rule, the stroke
      style and the colour as attributes. A colour is written as [#rrggbb] and
      its alpha, if below [1.], as [fill-opacity] or [stroke-opacity]. A subpath
      of zero length stroked with square caps is left out, since SVG draws it as
      a square.
    - A glyph run is a [text] element in the run's font at the run's size, each
      character placed at the position of its glyph by [x] and [y] lists, styled
      to turn off the viewer's ligatures, kerning and other shaping features, so
      that the viewer draws the run's glyphs where the run puts them and the
      text can be searched and copied. Each font is embedded once, by an
      [@font-face] rule whose source is a data URI of the font file. A run is
      written as text only if each of its glyphs renders one character of the
      Basic Multilingual Plane, which XML can hold and which the font maps to
      that glyph, other than glyph [0], which stands for the characters the font
      lacks. Any other run is written as the filled outlines of its glyphs, in a
      group whose [aria-label] is the run's text, its characters that XML cannot
      hold replaced by U+FFFD.
    - An image is an [image] element holding a PNG data URI of its pixels,
      stretched over its box and drawn with [image-rendering: pixelated].
    - A clip is a [clipPath] element and a group that refers to it; a transform
      writes no element of its own; an opacity is a group with an [opacity].
    - A stamp writes its picture once, as a definition, and draws it with a
      [use] element per instance, which sets what the instance varies: its
      translation and scale, the colours and alphas of the leaves it replaces,
      and their pen when it scales. It writes each instance in full instead when
      one definition cannot carry these: when it scales its instances and either
      the strokes of its picture differ in width or dashes or its picture holds
      a stamp that scales.
    - A tag is a group carrying [data-] attributes, which {!section-tags}
      describes.

    Styles are attributes of each element, never selectors of a style sheet, so
    that documents inlined in one HTML page do not restyle each other.

    {1:accuracy Accuracy}

    The document places every point the picture paints on the page within 0.001
    point of where the picture puts it, horizontally and vertically, whatever
    the transforms above it, except the ends of dashes past the first thousand
    lengths of their pattern along a subpath:
    - Coordinates are mapped to the page through the transforms above them, in
      double precision, and written in points with three decimals, so that the
      numbers of the document are of the order of the page's size.
    - Geometry is cut at a margin around the page that keeps all that reaches
      it: a path is cropped there, keeping the exact part of its curves within,
      except that a dashed stroke is cut there with its curves that cross the
      margin first flattened to within 0.0005 point; an image is cropped to the
      pixels that meet it; and a glyph run or a stamp instance that does not
      meet it is left out.
    - A matrix is written only for a stroke whose pen the transforms above it
      stretch unevenly, for a glyph run they do more than scale evenly, and for
      an image they do more than scale along the axes. The numbers under a
      matrix get the decimals that keep the accuracy, and so do those of a
      stamp's picture, which is written once, relative to its instances.
    - Stroke widths and font sizes are written to the same accuracy, and dash
      lengths, whose errors add up along a subpath, to a thousandth of it.
      Colours are written to 8 bits per component and alphas to three decimals.
    - A picture under transforms whose composition overflows the range of floats
      or has no inverse is left out, and a number beyond [1e15] in magnitude is
      written as [±1e15].

    A viewer that computes in single precision, as browsers do, rounds each
    number it reads to 24 significant bits: less than 0.001 point for a
    coordinate of a page under 16,384 points, and more under a matrix, by up to
    the ratio of the matrix's largest stretch to its smallest.

    {1:viewers Viewers}

    The document shows the picture's meaning in viewers that support:
    - CSS [@font-face] rules with data URIs. A viewer without them draws text in
      a font of its own, at the positions of the run's glyphs.
    - The CSS properties [font-variant-ligatures], [font-kerning] and
      [font-feature-settings]. A viewer without them may substitute or move the
      run's glyphs.
    - The CSS property [image-rendering] with the value [pixelated]. A viewer
      without it smooths images shown larger than their pixels.
    - SVG 2's [href] without a namespace, [data-] attributes and [aria-label].

    Current browsers support all of them.

    {1:tags Tags}

    The group of a tag carries:
    - [data-id], the segments of the tag's id as a JSON array, an index as a
      number and a field as a string, such as [[0,"axis","x"]], so that a field
      holding [.] stays unambiguous;
    - for [Rows a] on a stamp, nothing more: the element of each instance [i]
      written carries [data-row], [a.(i)] in decimal;
    - for [Rows a] on any other picture, if [a] is not empty, [data-rows], the
      elements of [a] in decimal separated by spaces;
    - for [Cells { box; width; height }], [data-cells], the corner, width and
      height of [box] and the grid's [width] and [height], separated by spaces,
      and, if transforms lie above the tag, [data-matrix], the composition of
      those transforms as [xx yx xy yy x0 y0], which maps the coordinates of
      [box] to the group's: the page's, or, in a stamp's picture written once,
      those of the instance, which its [use] element places on the page.

    {1:determinism Determinism}

    Equal renderables give equal documents, byte for byte, and a number that
    rounds to zero is written [0], whatever its sign. The ids of clip paths,
    stamp definitions and font families derive from the content they name, so
    that documents inlined in one HTML page share an id only for equal content,
    and never conflict. *)

val render : Hugin_next_vg.Renderable.t -> string
(** [render r] is the SVG document of [r], encoded in UTF-8.

    Raises what reading the tensor of an image of [r] raises, such as for a
    tensor whose storage a compiled call consumed. *)
