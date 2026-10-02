(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Color = Hugin_next_gg.Color
module Font = Hugin_next_font.Font
module Scheme = Hugin_next_kit.Scheme
module Locale = Hugin_next_kit.Locale
open Common

type t = {
  ink : Color.t;
  paper : Color.t;
  accent : Color.t;
  size : float;
  fonts : Font.t list;
  palette : Scheme.t;
  scheme : Scheme.t;
  locale : Locale.t;
}

let v ?(ink = Color.gray 0.1) ?(paper = Color.white) ?accent ?(size = 10.)
    ?(fonts = [ Font.regular; Font.bold ]) ?(palette = Scheme.tableau10)
    ?(scheme = Scheme.viridis) ?(locale = Locale.default) () =
  if not (is_pos size) then
    err "Theme.v" "size %g is not finite and positive" size;
  (match fonts with [] -> err "Theme.v" "fonts is empty" | _ :: _ -> ());
  let accent =
    match accent with Some c -> c | None -> (Scheme.colors 1 palette).(0)
  in
  { ink; paper; accent; size; fonts; palette; scheme; locale }

let default = v ()
let ink th = th.ink
let paper th = th.paper
let accent th = th.accent
let size th = th.size
let fonts th = th.fonts
let palette th = th.palette
let scheme th = th.scheme
let locale th = th.locale

let equal th th' =
  Color.equal th.ink th'.ink
  && Color.equal th.paper th'.paper
  && Color.equal th.accent th'.accent
  && Float.equal th.size th'.size
  && List.equal Font.equal th.fonts th'.fonts
  && Scheme.equal th.palette th'.palette
  && Scheme.equal th.scheme th'.scheme
  && Locale.equal th.locale th'.locale

let pp ppf th =
  Format.fprintf ppf
    "@[<v 1>(theme@ ink %a@ paper %a@ accent %a@ size %g@ fonts %a@ palette \
     %a@ scheme %a@ locale %a)@]"
    Color.pp th.ink Color.pp th.paper Color.pp th.accent th.size
    (Format.pp_print_list ~pp_sep:Format.pp_print_space Font.pp)
    th.fonts Scheme.pp th.palette Scheme.pp th.scheme Locale.pp th.locale
