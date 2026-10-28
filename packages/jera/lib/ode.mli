(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ordinary differential equations: the implementation of {!Jera.Ode}. *)

type (-'k, 'y, 't, 'd) t

val euler : ([ `Formula ], 'y, 't, 'd) t
val rk4 : ([ `Formula ], 'y, 't, 'd) t
val ssprk3 : ([ `Formula ], 'y, 't, 'd) t
val bs3 : ([ `Formula | `Embedded ], 'y, 't, 'd) t
val tsit5 : ([ `Formula | `Embedded ], 'y, 't, 'd) t
val dopri5 : ([ `Formula | `Embedded ], 'y, 't, 'd) t

val tableau :
  a:float array array ->
  b:float array ->
  c:float array ->
  ([ `Formula ], 'y, 't, 'd) t

val kvaerno5 :
  ?mass:('y -> 'y) ->
  linear:'y Linear.t ->
  ((float, 't, 'd) Nx.t -> 'y -> 'y -> 'y) ->
  ([ `Embedded ], 'y, 't, 'd) t

type ('t, 'd) time = (float, 't, 'd) Nx.t
type ('y, 't, 'd) field = ('t, 'd) time -> 'y -> 'y

val march :
  ('y, 'd) Nx.Ptree.t ->
  ([> `Formula ], 'y, 't, 'd) t ->
  steps:int ->
  ('y, 't, 'd) field ->
  at:('t, 'd) time ->
  'y ->
  'y

val solve :
  ('y, 'd) Nx.Ptree.t ->
  ([> `Embedded ], 'y, 't, 'd) t ->
  tol:Tol.t ->
  budget:int ->
  ('y, 't, 'd) field ->
  t0:('t, 'd) time ->
  t1:('t, 'd) time ->
  'y ->
  ('y, 'd) Solution.t

val sample :
  ('y, 'd) Nx.Ptree.t ->
  ([> `Embedded ], 'y, 't, 'd) t ->
  tol:Tol.t ->
  budget:int ->
  ('y, 't, 'd) field ->
  at:('t, 'd) time ->
  'y ->
  ('y, 'd) Solution.t

val path :
  ('y, 'd) Nx.Ptree.t ->
  ([> `Embedded ], 'y, 't, 'd) t ->
  tol:Tol.t ->
  budget:int ->
  ('y, 't, 'd) field ->
  t0:('t, 'd) time ->
  t1:('t, 'd) time ->
  'y ->
  (('y, 't, 'd) Piecewise.t, 'd) Solution.t

val event :
  ('y, 'd) Nx.Ptree.t ->
  ([> `Embedded ], 'y, 't, 'd) t ->
  tol:Tol.t ->
  budget:int ->
  ('y, 't, 'd) field ->
  event:(('t, 'd) time -> 'y -> (float, 'e, 'd) Nx.t) ->
  t0:('t, 'd) time ->
  t1:('t, 'd) time ->
  'y ->
  (('t, 'd) time * 'y * 'd Nx.int32_t, 'd) Solution.t

val delay :
  ('y, 'd) Nx.Ptree.t ->
  ([> `Formula | `Embedded ], 'y, 't, 'd) t ->
  tol:Tol.t ->
  budget:int ->
  pieces:int ->
  (('t, 'd) time -> 'y -> 'y -> 'y) ->
  lags:('t, 'd) time ->
  history:(('t, 'd) time -> 'y) ->
  at:('t, 'd) time ->
  'y ->
  ('y, 'd) Solution.t
