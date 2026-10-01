(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Geometry and colour.

    These are the values every other Hugin library speaks in: points {!P2},
    boxes {!Box2}, affine maps {!Affine}, paths {!Path} and the pen that strokes
    them {!Stroke}, and colours {!Color}. The library depends on nothing.

    {1:conventions Conventions}

    - {b The plane.} The x axis points right and the y axis points down, as on a
      page or a screen. Units are the caller's; Hugin uses points, 1/72 inch.
      Angles are in radians, and a positive angle turns from the positive x axis
      towards the positive y axis, clockwise on screen.
    - {b Argument order.} Functions that build or transform a value take it
      last, so they chain with [|>]:
      [Path.empty |> Path.move_to p |> Path.line_to q]. {!Path.append} follows
      the rule, which makes it the opposite of [List.append].
    - {b Equality.} [equal] functions compare structure, with floats compared by
      [Float.equal], so [nan] equals [nan] and [0.] equals [-0.]. One shape can
      have several structures: a rectangle from {!Path.rect} and the same
      rectangle drawn from another corner are not equal. [compare] functions are
      total orders compatible with [equal].
    - {b Printing.} [pp] functions format values for debugging and tests. Their
      output may change between releases unless the function says otherwise.
    - {b Non-finite numbers.} Points and affine maps hold any floats, and
      functions on them follow float arithmetic: a non-finite coordinate or
      coefficient makes the points it reaches non-finite. In a path such points
      are gaps ({!Path.section-gaps}). Boxes, stroke styles and colours never
      hold a non-finite number: their constructors raise [Invalid_argument], as
      each states. *)

module Affine = Affine
module P2 = P2
module Box2 = Box2
module Path = Path
module Stroke = Stroke
module Color = Color
