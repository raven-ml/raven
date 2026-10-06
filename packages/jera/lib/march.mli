(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The loop every march runs: one interval of [at] per trip of a scan, each
    interval a {!Rune.remat} region of equal steps, the states stacked at the
    times of [at]. *)

val check : string -> steps:int -> (float, 'b) Nx.t -> unit
(** [check fn ~steps at] raises [Invalid_argument] naming [fn] if [steps < 1],
    if [at] is not 1-D or is empty, and, through {!Nx.check}, if [at] is not
    strictly monotone. *)

val run :
  'c Nx.Ptree.t ->
  'y Nx.Ptree.t ->
  at:(float, 't) Nx.t ->
  interval:((float, 't) Nx.t -> (float, 't) Nx.t -> 'c -> 'c) ->
  state:('c -> 'y) ->
  'c ->
  'y
(** [run c y ~at ~interval ~state init] is [state init] followed by
    [state (interval t_i t_(i+1) carry)] for each interval of [at], stacked on a
    new leading axis. *)

val steps :
  'c Nx.Ptree.t ->
  (float, 't) Nx.dtype ->
  int ->
  ((float, 't) Nx.t -> 'c -> 'c) ->
  'c ->
  'c
(** [steps c dtype n f init] applies [f j] for [j = 0, ..., n - 1], [j] a scalar
    of [dtype], by a scan when [n > 1]; [n = 0] is [init]. *)
