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

val interval :
  ('c, 'd) Nx.Ptree.t ->
  steps:int ->
  ((float, 't, 'd) Nx.t ->
  (float, 't, 'd) Nx.t ->
  (float, 't, 'd) Nx.t ->
  'c ->
  'c) ->
  (float, 't, 'd) Nx.t ->
  (float, 't, 'd) Nx.t ->
  'c ->
  'c
(** [interval c ~steps step t0 t1 init] takes [steps] steps from the scalar [t0]
    to the scalar [t1], [step t t_end h carry] each, by a scan when there are
    more than two. Step [j] starts at [t0 + j h] with [h = (t1 − t0) / steps],
    or at [t1] where that rounds past it, and ends where step [j + 1] starts;
    the last ends at [t1] itself. Every step but the last has size [h]; the last
    has size [t1] minus its start, never past [t1]. A step that reads the time
    at its end takes [t_end], which its start plus its size can round apart
    from. [t1] may lie below [t0]. *)
