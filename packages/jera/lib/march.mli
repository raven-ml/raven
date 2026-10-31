(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The loop every march runs: one interval of [at] per trip of a scan, each
    interval a {!Rune.remat} region of equal steps, the states stacked at the
    times of [at]. *)

val check : string -> steps:int -> (float, 'b, 'd) Nx.t -> unit
(** [check fn ~steps at] raises [Invalid_argument] naming [fn] if [steps < 1],
    if [at] is not 1-D or is empty, and, through {!Nx.check}, if [at] is not
    strictly monotone. *)

val run :
  ('c, 'd) Nx.Ptree.t ->
  ('y, 'd) Nx.Ptree.t ->
  at:(float, 't, 'd) Nx.t ->
  interval:((float, 't, 'd) Nx.t -> (float, 't, 'd) Nx.t -> 'c -> 'c) ->
  state:('c -> 'y) ->
  'c ->
  'y
(** [run c y ~at ~interval ~state init] is [state init] followed by
    [state (interval t_i t_(i+1) carry)] for each interval of [at], stacked on a
    new leading axis. *)

val grid :
  steps:int ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t * (float, 'b, 'd) Nx.t
(** [grid ~steps t0 t1] is [(h, starts)] for [steps] steps from the scalar
    [t0] to the scalar [t1]: [h = (t1 − t0) / steps], and a vector of [steps]
    starts, step [j]'s [t0 + j h] or [t1] where that rounds past it. Every
    step but the last has size [h]; the last has size [t1] minus its start,
    never past [t1]. [t1] may lie below [t0]. *)

val steps :
  ('c, 'd) Nx.Ptree.t ->
  ('x, 'd) Nx.Ptree.t ->
  int ->
  ('x -> 'c -> 'c) ->
  'c ->
  'x ->
  'c
(** [steps c x n f init xs] applies [f] to row [j] of [xs] for
    [j = 0, ..., n - 1], each leaf of [xs] a vector of at least [n] elements,
    by a scan when [n > 1]; [n = 0] is [init]. *)
