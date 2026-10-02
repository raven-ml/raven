(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The built-in marks. *)

module Text := Hugin_next_text.Text
module Symbol := Hugin_next_kit.Symbol
module Curve := Hugin_next_kit.Curve
module Color := Hugin_next_gg.Color

val dot :
  ?fill:('f, Color.t) Channel.t ->
  ?stroke:('s, Color.t) Channel.t ->
  ?opacity:('o, float) Channel.t ->
  ?size:(float, float) Channel.t ->
  ?symbol:(string, Symbol.t) Channel.t ->
  ?fx:(string, string) Channel.t ->
  ?fy:(string, string) Channel.t ->
  x:('x, float) Channel.t ->
  y:('y, float) Channel.t ->
  unit ->
  Figure.t

val line :
  ?x:('x, float) Channel.t ->
  ?stroke:('s, Color.t) Channel.t ->
  ?fill:('f, Color.t) Channel.t ->
  ?width:('w, float) Channel.t ->
  ?opacity:('o, float) Channel.t ->
  ?curve:Curve.t ->
  ?fx:(string, string) Channel.t ->
  ?fy:(string, string) Channel.t ->
  y:('y, float) Channel.t ->
  unit ->
  Figure.t

val rect :
  ?x:('x, float) Channel.t ->
  ?x2:('x, float) Channel.t ->
  ?y:('y, float) Channel.t ->
  ?y2:('y, float) Channel.t ->
  ?fill:('f, Color.t) Channel.t ->
  ?stroke:('s, Color.t) Channel.t ->
  ?opacity:('o, float) Channel.t ->
  ?fx:(string, string) Channel.t ->
  ?fy:(string, string) Channel.t ->
  unit ->
  Figure.t

val rule :
  ?x:('x, float) Channel.t ->
  ?x2:('x, float) Channel.t ->
  ?y:('y, float) Channel.t ->
  ?y2:('y, float) Channel.t ->
  ?stroke:('s, Color.t) Channel.t ->
  ?width:('w, float) Channel.t ->
  ?opacity:('o, float) Channel.t ->
  ?fx:(string, string) Channel.t ->
  ?fy:(string, string) Channel.t ->
  unit ->
  Figure.t

val text :
  ?fill:('f, Color.t) Channel.t ->
  ?opacity:('o, float) Channel.t ->
  ?dx:float ->
  ?dy:float ->
  ?fx:(string, string) Channel.t ->
  ?fy:(string, string) Channel.t ->
  x:('x, float) Channel.t ->
  y:('y, float) Channel.t ->
  text:('t, Text.t) Channel.t ->
  unit ->
  Figure.t

val image :
  ?fx:(string, string) Channel.t ->
  ?fy:(string, string) Channel.t ->
  ('a, 'b) Nx.t ->
  Figure.t

val contour :
  ?x:('x, float) Channel.t ->
  ?y:('y, float) Channel.t ->
  ?opacity:('o, float) Channel.t ->
  ?fx:(string, string) Channel.t ->
  ?fy:(string, string) Channel.t ->
  fill:(float, Color.t) Channel.t ->
  unit ->
  Figure.t
