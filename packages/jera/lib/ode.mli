(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ordinary differential equations: the implementation of {!Jera.Ode}. *)

type (-'k, 'y, 't) t

val euler : ([ `Formula ], 'y, 't) t
val rk4 : ([ `Formula ], 'y, 't) t
val ssprk3 : ([ `Formula ], 'y, 't) t
val bs3 : ([ `Formula | `Embedded ], 'y, 't) t
val tsit5 : ([ `Formula | `Embedded ], 'y, 't) t
val dopri5 : ([ `Formula | `Embedded ], 'y, 't) t

val tableau :
  a:float array array ->
  b:float array ->
  c:float array ->
  ([ `Formula ], 'y, 't) t

val kvaerno5 :
  ?mass:('y -> 'y) ->
  linear:'y Linear.t ->
  ((float, 't) Nx.t -> 'y -> 'y -> 'y) ->
  ([ `Embedded ], 'y, 't) t

type 't time = (float, 't) Nx.t
type ('y, 't) field = 't time -> 'y -> 'y

val march :
  'y Nx.Ptree.t ->
  ([> `Formula ], 'y, 't) t ->
  steps:int ->
  ('y, 't) field ->
  at:'t time ->
  'y ->
  'y

val solve :
  'y Nx.Ptree.t ->
  ([> `Embedded ], 'y, 't) t ->
  tol:Tol.t ->
  budget:int ->
  ('y, 't) field ->
  t0:'t time ->
  t1:'t time ->
  'y ->
  'y Solution.t

val sample :
  'y Nx.Ptree.t ->
  ([> `Embedded ], 'y, 't) t ->
  tol:Tol.t ->
  budget:int ->
  ('y, 't) field ->
  at:'t time ->
  'y ->
  'y Solution.t

val path :
  'y Nx.Ptree.t ->
  ([> `Embedded ], 'y, 't) t ->
  tol:Tol.t ->
  budget:int ->
  ('y, 't) field ->
  t0:'t time ->
  t1:'t time ->
  'y ->
  ('y, 't) Piecewise.t Solution.t

val event :
  'y Nx.Ptree.t ->
  ([> `Embedded ], 'y, 't) t ->
  tol:Tol.t ->
  budget:int ->
  ('y, 't) field ->
  event:('t time -> 'y -> (float, 'e) Nx.t) ->
  t0:'t time ->
  t1:'t time ->
  'y ->
  ('t time * 'y * (int32, Nx.int32_elt) Nx.t) Solution.t

val delay :
  'y Nx.Ptree.t ->
  ([> `Formula | `Embedded ], 'y, 't) t ->
  tol:Tol.t ->
  budget:int ->
  pieces:int ->
  ('t time -> 'y -> 'y -> 'y) ->
  lags:'t time ->
  history:('t time -> 'y) ->
  at:'t time ->
  'y ->
  'y Solution.t
