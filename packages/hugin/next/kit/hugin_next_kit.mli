(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Visual encoding.

    The kit turns data values into the numbers, strings, colours and paths a
    figure draws, and draws nothing itself. {!Scale} normalises data into
    \[[0];[1]\] and gives its guide values, {!Ticks} chooses and labels the
    ticks of an axis, {!Number} writes numbers, {!Time} puts instants on the
    calendar and {!Locale} holds the strings numbers and dates are written with.
    {!Scheme} colours normalised values and categories, {!Symbol} draws marker
    shapes, {!Curve} draws lines through points, {!Stack} lays lengths end to
    end and {!Stats} summarises data, as histograms. The kit measures no text:
    choosing ticks takes a function that measures labels.

    {1:conventions Conventions}

    - {b Normalised values.} A scale maps its domain onto \[[0];[1]\], every
      position a scale or ticks return is such a fraction, and schemes colour
      such fractions. Points, pixels and projections belong to the figure:
      curves are drawn in the units of the points they are given, symbols in the
      units whose square their size is, and stacks in the units of their
      lengths.
    - {b Missing values.} A value a scale cannot place is
      {{!Scale.section-missing}missing}: {!Scale.normalize} returns [nan] for it
      and {!Scale.invert} returns [None] where no value exists; neither raises.
      Downstream, {!Scheme.color} paints a normalised [nan] with the colour
      given for unknown values, curves break at points with a non-finite
      coordinate, stacks skip non-finite lengths and histograms count no
      non-finite value. Functions raise [Invalid_argument] only on arguments
      that are programming errors, as each states.
    - {b Argument order, equality and printing} follow
      {{!Hugin_next_gg.section-conventions}[hugin.next.gg]}: the value a
      function reads or transforms comes last, [equal] compares structure with
      floats compared by [Float.equal], and [pp] output may change between
      releases unless a printer says otherwise. Functions held by a value, those
      of a {!Scale.custom} scale, compare physically.
    - {b Strings.} Every string the kit writes is UTF-8 and depends only on the
      arguments of the call that writes it: there is no global locale, every
      writer takes a {!Locale.t} that defaults to {!Locale.default}. Digits are
      the kit's own and the same on every platform ({!Number.section-writing}).
    - {b Typography.} Labels use the characters typesetting asks for: U+2212
      MINUS SIGN by default, U+00D7 MULTIPLICATION SIGN, the superscript digits
      and U+207B SUPERSCRIPT MINUS, and U+00B5 MICRO SIGN. *)

module Locale = Locale
module Number = Number
module Time = Time
module Scale = Scale
module Ticks = Ticks
module Scheme = Scheme
module Symbol = Symbol
module Curve = Curve
module Stack = Stack
module Stats = Stats
