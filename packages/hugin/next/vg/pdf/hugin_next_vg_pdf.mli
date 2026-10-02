(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PDF rendering.

    Writes {{!Hugin_next_vg.Renderable}renderables} as one-page PDF 1.7
    documents: text stays text, fonts and images are embedded, every stream is
    deflated, and every picture maps to the operators below.

    {1:mapping Pictures as PDF}

    - The page's media box is [w] by [h] points, for a renderable of [w] by [h].
      Where the picture paints nothing, viewers show their page colour, usually
      white.
    - Fills and strokes are path painting operators. Colours are in the
      [DeviceRGB] colour space, which viewers display as sRGB, and an alpha
      below [1.] is set by a graphics state's [ca] for fills and glyphs and [CA]
      for strokes.
    - A glyph run is text shown in the run's font at the run's size, each glyph
      by its glyph id at its position in the run. Each font is embedded once, as
      a [CIDFontType2] font with the [Identity-H] encoding. A [ToUnicode] map
      gives each glyph the text it renders, and a run whose text that map cannot
      reproduce, such as one where a glyph renders several characters or two
      characters share a glyph, carries its text in an [ActualText] span, so
      that copying text from the page gives each run's text.
    - An image is an image XObject painted over its box, its samples deflated
      losslessly and marked to be shown without interpolation. An RGBA image has
      its alpha as a soft mask.
    - A clip is a clipping path; a transform writes no operator of its own; an
      opacity is a transparency group XObject painted with the opacity as its
      alpha.
    - A stamp writes its picture once, as a form XObject, and paints it at each
      instance after setting what the instance varies: its translation and
      scale, the colours and alphas of the leaves it replaces, and their pen
      when it scales. It writes each instance in full instead when one form
      cannot carry these: when it scales its instances and the strokes of its
      picture differ in width or dashes, or its picture holds a stamp that
      scales or a stroke within an opacity, whose transparency group some
      viewers do not pass the pen on to; and when a replaced colour has an alpha
      below [1.] and the picture holds an opacity, which resets alphas. It also
      writes in full an instance whose scale, or the width of the pen it sets,
      the form's numbers cannot write to the accuracy.
    - Tags are ignored.

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
      pixels that meet it; and a glyph or a stamp instance that does not meet it
      is left out.
    - A matrix is written for each glyph run and image, which PDF places by one,
      and for a stroke whose pen the transforms above it stretch unevenly. The
      numbers under a matrix get the decimals that keep the accuracy, and so do
      those of a stamp's picture, which is written once, relative to its
      instances.
    - Stroke widths, font sizes and glyph widths are written to the same
      accuracy, and dash lengths, whose errors add up along a subpath, to a
      thousandth of it. Colour components and alphas are written to three
      decimals. What paints too thin a shape to be written at that accuracy is
      left out: a stroke whose width rounds to zero, a glyph run whose size
      does, and a stroke or an image whose matrix rounds to one that flattens
      it. A dash pattern whose lengths all round to zero is written solid.
    - A picture under transforms whose composition overflows the range of floats
      or has no inverse is left out, and a number beyond [1e15] in magnitude is
      written as [±1e15].

    A viewer that computes in single precision rounds each number it reads to 24
    significant bits: less than 0.001 point for a coordinate of a page under
    16,384 points, and more under a matrix, by up to the ratio of the matrix's
    largest stretch to its smallest.

    {1:viewers Viewers}

    The document shows the picture's meaning in viewers that support
    transparency groups and soft masks (PDF 1.4), honour an image's request not
    to be interpolated, and draw a dash of zero length with square caps as a
    square. A viewer that interpolates anyway smooths images shown larger than
    their pixels, and one that draws no such square leaves out the dots of a
    dotted line with square caps.

    {1:determinism Determinism}

    Equal renderables give equal documents, byte for byte: the document holds no
    creation date and no identifier, objects are numbered in the order the
    picture first uses them, and a number that rounds to zero is written [0],
    whatever its sign. *)

val render : Hugin_next_vg.Renderable.t -> string
(** [render r] is the PDF document of [r].

    Raises what reading the tensor of an image of [r] raises, such as for a
    tensor whose storage a compiled call consumed. *)
