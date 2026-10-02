(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Fonts and glyph runs.

    {!Font} decodes OpenType fonts: character maps, advances, kerning, vertical
    metrics and glyph outlines. {!Run} is text set in one font, glyphs placed by
    text layout, which pictures draw and renderers embed. Neither lays out text;
    that is the text library's work.

    Lengths follow the
    {{!Hugin_gg.section-conventions}geometry conventions}: a y-down plane,
    angles in radians. *)

module Font = Font
module Run = Run
