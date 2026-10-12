(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Common

type t = Figure of float * float | Panels of float * float

let check fn w h =
  if not (is_pos w && is_pos h) then
    err fn "%a × %a is not finite and positive" Float_text.pp w Float_text.pp h

let figure w h =
  check "Size.figure" w h;
  Figure (w, h)

let panels w h =
  check "Size.panels" w h;
  Panels (w, h)

let mm l = l *. 72. /. 25.4
let dpi d = d /. 72.

let equal s s' =
  match (s, s') with
  | Figure (w, h), Figure (w', h') | Panels (w, h), Panels (w', h') ->
      Float.equal w w' && Float.equal h h'
  | Figure _, Panels _ | Panels _, Figure _ -> false

let pp ppf = function
  | Figure (w, h) -> Format.fprintf ppf "figure %a × %a pt" Float_text.pp w Float_text.pp h
  | Panels (w, h) -> Format.fprintf ppf "panels %a × %a pt" Float_text.pp w Float_text.pp h
