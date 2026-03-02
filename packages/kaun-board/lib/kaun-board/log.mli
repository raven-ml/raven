(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Training logger.

    A logger creates a {!Run} directory and appends scalar {!Event}s as JSONL.
    All write operations are thread-safe.

    {[
      let logger = Log.create ~experiment:"mnist" () in
      (* in training loop *)
      Log.log_scalar logger ~step ~tag:"train/loss" loss;
      Log.close logger
    ]} *)

(** {1:types Types} *)

type t
(** The type for logging sessions. *)

(** {1:constructors Constructors} *)

val create :
  ?base_dir:string ->
  ?experiment:string ->
  ?tags:string list ->
  ?config:(string * Jsont.json) list ->
  unit ->
  t
(** [create ()] is a new logging session backed by a fresh run directory.

    [base_dir] defaults to the value of [RAVEN_RUNS_DIR], or
    [XDG_CACHE_HOME/raven/runs] when unset. [experiment] names the experiment
    and is appended to the run ID. [tags] defaults to [[]]. [config] defaults to
    [[]] and is stored in the run manifest.

    Raises [Sys_error] if the run directory cannot be created. *)

(** {1:logging Logging} *)

val log_scalar :
  t ->
  step:int ->
  ?epoch:int ->
  ?direction:Event.direction ->
  tag:string ->
  float ->
  unit
(** [log_scalar t ~step ~tag v] appends a scalar event with metric name [tag]
    and value [v].

    [epoch] defaults to [None]. [direction] indicates whether lower or higher
    is better for best-value tracking; when omitted, the dashboard uses a
    heuristic (e.g. tags containing "loss" prefer lower). *)

val log_scalars :
  t ->
  step:int ->
  ?epoch:int ->
  ?directions:(string * Event.direction) list ->
  (string * float) list ->
  unit
(** [log_scalars t ~step pairs] appends one scalar event per [(tag, value)]
    pair.

    [epoch] defaults to [None]. [directions] maps tag names to [direction] for
    best-value tracking; tags not listed fall back to the heuristic. *)

(** {1:accessors Accessors} *)

val run_id : t -> string
(** [run_id t] is the unique run identifier. *)

val run_dir : t -> string
(** [run_dir t] is the path to the run directory. *)

(** {1:lifecycle Lifecycle} *)

val close : t -> unit
(** [close t] marks the session as closed. Subsequent writes are silently
    ignored. Safe to call multiple times. *)
