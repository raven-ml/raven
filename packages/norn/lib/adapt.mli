(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Warmup: window schedules, dual averaging and Fisher fits of geometry. *)

(** {1:schedule Schedule} *)

val schedule : int -> (int * bool) list
(** [schedule n] is Stan's windows for [n] warmup steps scaled to [n]: each
    window's length and whether the geometry is refitted at its end. *)

val windows : (int * bool) list -> Nx.int32_t * Nx.int32_t
(** [windows s] is [s]'s lengths and refits ([1] or [0]) as tensors. *)

(** {1:averaging Dual averaging} *)

type 'f averaging = {
  mu : (float, 'f) Nx.t;
  s_bar : (float, 'f) Nx.t;
  x_bar : (float, 'f) Nx.t;
  count : (float, 'f) Nx.t;
}
(** The state of dual averaging (Nesterov 2009; Hoffman and Gelman 2014) of log
    step sizes, of any shape. *)

val averaging_ptree : unit -> 'f averaging Nx.Ptree.t

val restart : (float, 'f) Nx.t -> 'f averaging
(** [restart eps] averages anew from the step sizes [eps]. *)

val average :
  'f averaging ->
  target:(float, 'f) Nx.t ->
  (float, 'f) Nx.t ->
  'f averaging * (float, 'f) Nx.t
(** [average a ~target stat] is the averaging after an acceptance [stat], and
    the next step sizes. *)

val final : 'f averaging -> (float, 'f) Nx.t
(** [final a] is the averaged step sizes. *)

(** {1:windows Windows} *)

type 'u window = {
  n : Nx.int32_t;
  shift : 'u;
  sx : 'u;
  sxx : 'u;
  sg : 'u;
  sgg : 'u;
  xs : 'u;
  gs : 'u;
}
(** A window's draws and scores of every chain: their count, sums shifted by the
    window's first position, and a buffer of them per chain. *)

val window_ptree : 'u Nx.Ptree.t -> 'u window Nx.Ptree.t

val empty : 'u Nx.Ptree.t -> buffer:int -> 'u -> 'u window
(** [empty u ~buffer x] is a window opened at the position [x] that buffers
    [buffer] draws per chain, none if [buffer = 0]. *)

val reopen : 'u Nx.Ptree.t -> 'u window -> 'u -> 'u window
(** [reopen u w x] is [w] emptied and opened at [x], keeping its buffers. *)

val record : 'u Nx.Ptree.t -> 'u window -> 'u -> 'u -> 'u window
(** [record u w x g] is [w] with the draw [x] and its score [g]. *)

(** {1:fits Fisher fits} *)

val clip : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [clip v] is the variances [v] clipped to [[1e-20, 1e20]]. *)

val per_chain :
  'u Nx.Ptree.t ->
  (float, 'f) Nx.dtype ->
  rank:int ->
  'u window ->
  ('u, 'f) Gaussian.t
(** [per_chain u dt ~rank w] is each chain's Gaussian fitted to its draws and
    scores in [w] by the Fisher divergence (Seyboldt, Carlson and Carpenter
    2026), stacked on the chain axis: a diagonal scale [sqrt (sd x / sd score)]
    per element, then [rank] directions, the window buffering at least [rank]
    draws. *)

val pooled :
  'u Nx.Ptree.t ->
  (float, 'f) Nx.dtype ->
  rank:int ->
  'u window ->
  ('u, 'f) Gaussian.t
(** [pooled u dt ~rank w] is one Gaussian fitted as {!per_chain} to every
    chain's draws together. *)

val population :
  'u Nx.Ptree.t ->
  (float, 'f) Nx.dtype ->
  (float, 'f) Nx.t ->
  'u ->
  ('u, 'f) Gaussian.t
(** [population u dt lw x] is the Gaussian of the draws [x], on a leading axis,
    weighted by [exp lw]: their weighted mean and covariance, the covariance as
    a diagonal and every direction of its whitened form. A direction outside the
    draws' span keeps the diagonal's variance, and a factorisation that fails
    leaves the diagonal. *)
