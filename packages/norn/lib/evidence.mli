(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Evidence}, with the constructor {!Nested}
    and {!Smc} build their estimates with. *)

type ('u, 'f, 'd) t

val log_evidence : ('u, 'f, 'd) t -> (float, 'f, 'd) Nx.t

val error : ('u, 'f, 'd) t -> (float, 'f, 'd) Nx.t

val information : ('u, 'f, 'd) t -> (float, 'f, 'd) Nx.t

val sample : ('u, 'f, 'd) t -> ('u, 'f, 'd) Weighted.t

val expectation :
  ('u, 'd) Nx.Ptree.t ->
  ('u -> (float, 'f, 'd) Nx.t) ->
  ('u, 'f, 'd) t ->
  (float, 'f, 'd) Nx.t * (float, 'f, 'd) Nx.t

(** {1:stop Why a run stopped} *)

type stop =
  | Converged
  | Remaining of float
  | Temperature of float

val stop : ('u, 'f, 'd) t -> stop

(** {1:structure Structure and formatting} *)

val ptree : ('u, 'd) Nx.Ptree.t -> (('u, 'f, 'd) t, 'd) Nx.Ptree.t

val pp : Format.formatter -> ('u, 'f, 'd) t -> unit

(** {1:construct Construction} *)

val converged : int32
(** [converged] is the code of {!Converged} in [v]'s [stop]. *)

val remaining : int32
(** [remaining] is the code of {!Remaining}. *)

val temperature : int32
(** [temperature] is the code of {!Temperature}. *)

val v :
  sample:('u, 'f, 'd) Weighted.t ->
  log_evidence:(float, 'f, 'd) Nx.t ->
  error:(float, 'f, 'd) Nx.t ->
  information:(float, 'f, 'd) Nx.t ->
  stop:'d Nx.int32_t ->
  reached:(float, 'f, 'd) Nx.t ->
  replicates:(float, 'f, 'd) Nx.t ->
  groups:'d Nx.int32_t ->
  group_count:int ->
  ('u, 'f, 'd) t
(** [v ~sample ~log_evidence ~error ~information ~stop ~reached ~replicates
    ~groups ~group_count] is the estimate that stopped as the scalar code
    [stop] says, {!converged}, {!remaining} or {!temperature}, the latter two
    with the payload [reached], a scalar. [sample]'s log weights must
    be normalised. [replicates], of shape [[r; n]] for [n] draws, holds the
    sample's normalised log weights under each of [r] simulated shrinkage
    sequences, or [sample]'s own log weights as one row. [groups] gives each
    draw its group, from [0] to [group_count - 1]: draws of one group may be
    correlated, draws of two are independent. {!expectation} reads
    [replicates] and [groups]. *)
