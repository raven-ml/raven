(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Rich text and its layout.

    {!Text} is the rich text that labels, titles and legend entries are made of,
    and {!Text.Layout} sets a text in a list of faces as the glyph runs
    ({!Hugin_next_font.Run}) that pictures draw, measuring it without drawing.
    The library depends on fonts, not on pictures: a caller draws a layout's
    runs in whatever it draws with.

    Lengths follow the
    {{!Hugin_next_gg.section-conventions}geometry conventions}: the plane is
    y-down. *)

module Text = Text
