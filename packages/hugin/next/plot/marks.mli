(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The built-in marks. *)

open Api

val dot :
  ?fill:('f, Color.t) channel ->
  ?stroke:('s, Color.t) channel ->
  ?opacity:('o, float) channel ->
  ?size:(float, float) channel ->
  ?symbol:(string, Symbol.t) channel ->
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  x:('x, float) channel ->
  y:('y, float) channel ->
  unit ->
  t

val line :
  ?x:('x, float) channel ->
  ?stroke:('s, Color.t) channel ->
  ?fill:('f, Color.t) channel ->
  ?width:('w, float) channel ->
  ?opacity:('o, float) channel ->
  ?curve:Curve.t ->
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  y:('y, float) channel ->
  unit ->
  t

val rect :
  ?x:('x, float) channel ->
  ?x2:('x, float) channel ->
  ?y:('y, float) channel ->
  ?y2:('y, float) channel ->
  ?fill:('f, Color.t) channel ->
  ?stroke:('s, Color.t) channel ->
  ?opacity:('o, float) channel ->
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  unit ->
  t

val rule :
  ?x:('x, float) channel ->
  ?x2:('x, float) channel ->
  ?y:('y, float) channel ->
  ?y2:('y, float) channel ->
  ?stroke:('s, Color.t) channel ->
  ?width:('w, float) channel ->
  ?opacity:('o, float) channel ->
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  unit ->
  t

val text :
  ?fill:('f, Color.t) channel ->
  ?opacity:('o, float) channel ->
  ?dx:float ->
  ?dy:float ->
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  x:('x, float) channel ->
  y:('y, float) channel ->
  text:('t, Text.t) channel ->
  unit ->
  t

val image :
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  ('a, 'b) Nx.t ->
  t

val contour :
  ?x:('x, float) channel ->
  ?y:('y, float) channel ->
  ?opacity:('o, float) channel ->
  ?fx:(string, string) channel ->
  ?fy:(string, string) channel ->
  fill:(float, Color.t) channel ->
  unit ->
  t
