(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Planar geometry.

    The kit computes shapes of the plane, below rendering. {!Ring2} holds closed
    polygonal curves and {!Pgon2} the polygons they bound. {!Field2} samples a
    scalar field on a grid, read from an nx tensor, and gives its isolines as
    paths and its isobands as polygons with holes. Rings and polygons convert to
    {!Hugin_gg.Path.t} for drawing.

    {1:conventions Conventions}

    - {b The plane} is that of
      {{!Hugin_gg.section-conventions}[hugin.gg]}: the x axis points
      right and the y axis down. Coordinates are in the caller's units.
    - {b Orientation.} A closed curve is {e positively oriented} if it turns
      like positive angles, from the positive x axis towards the positive y
      axis: clockwise on screen, counterclockwise in a frame whose y axis points
      up. Walking along a positively oriented curve that does not cross itself,
      the region it encloses is on the right on screen. The {e winding number}
      of a closed curve around a point off it is the number of times the curve
      turns around the point, counted positively in that direction. The
      {e signed area} of closed curves is the integral of their winding number
      over the plane: for one curve that does not cross itself, the area it
      encloses, positive or negative with its orientation.
    - {b Surfaces.} The surface of a polygon is the set of points around which
      its rings wind a non-zero number of times: the nonzero fill rule of the
      renderers. Outer boundaries are positively oriented and holes negatively,
      so a hole's winding cancels its outer boundary's. A polygon's path filled
      with the nonzero rule paints its surface.
    - {b Argument order, equality and printing} follow
      {{!Hugin_gg.section-conventions}[hugin.gg]}: the value a
      function reads or transforms comes last, except that indexed reads take
      the value first, as [Array.get] does; [equal] compares structure with
      floats compared by [Float.equal]; and [pp] output is for debugging and may
      change between releases.
    - {b Errors.} Functions raise [Invalid_argument] only on arguments that are
      programming errors, as each states. *)

module Ring2 = Ring2
module Pgon2 = Pgon2
module Field2 = Field2
