(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Training run logging for monitoring dashboards.

    This module provides a file-based logging system for training metrics.
    Multiple output backends are supported (JSONL for kaun-console, TensorBoard
    format, or both).

    {2 Usage}

    The primary API is direct logging with explicit step values:

    {[
      let logger = Log.create ~experiment:"mnist" () in

      (* In training loop *)
      let state, loss = train_step ... in
      if state.step mod 100 = 0 then
        Log.log_scalar logger ~step:state.step ~tag:"train/loss" loss;

      Log.close logger
    ]}

    For users of {!Training.fit}, use {!Training.Callbacks.logging}:

    {[
      let logger = Log.create ~experiment:"mnist" () in
      let _, _ =
        Training.fit ~callbacks:[ Training.Callbacks.logging logger ] ...
      in
      Log.close logger
    ]}

    {2 Multiple Backends}

    To log to both kaun-console and TensorBoard:

    {[
      let logger =
        Log.create ~backend:(Log.multi [ Log.jsonl; Log.tensorboard ])
          ~experiment:"mnist" ()
      in
      ...
    ]} *)

(** {1 Output Backends} *)

type backend
(** Output backend for log events. *)

val jsonl : backend
(** JSONL backend (default). Writes events to [events.jsonl] in the run
    directory. Compatible with kaun-console. *)

val tensorboard : backend
(** TensorBoard backend. Writes event files compatible with TensorBoard.

    {b Note}: Writes a simple CSV format that can be imported into TensorBoard.
    For full TensorBoard integration, consider using the TensorBoard callback
    or a dedicated TensorBoard library. *)

val multi : backend list -> backend
(** [multi backends] combines multiple backends. Events are written to all. *)

(** {1 Logger Sessions} *)

type t
(** A logging session for a single training run. Each session writes to its
    own run directory containing a manifest and event log. Thread-safe. *)

val create :
  ?backend:backend ->
  ?base_dir:string ->
  ?experiment:string ->
  ?tags:string list ->
  ?config:(string * Yojson.Safe.t) list ->
  unit ->
  t
(** [create ?backend ?base_dir ?experiment ?tags ?config ()] creates a new
    logging session.

    @param backend Output backend (default: {!jsonl})
    @param base_dir Directory for all runs. Defaults to [RAVEN_RUNS_DIR] if set,
    otherwise [XDG_CACHE_HOME/raven/runs].
    @param experiment Name to identify this experiment
    @param tags List of tags for filtering runs
    @param config Hyperparameters/configuration stored in the run manifest *)

val run_id : t -> string
(** [run_id logger] returns the unique run identifier (timestamp-based). *)

val run_dir : t -> string
(** [run_dir logger] returns the filesystem path to the run directory. *)

val close : t -> unit
(** [close logger] flushes buffers and closes the log file. *)

(** {1 Scalar Logging} *)

val log_scalar : t -> step:int -> epoch:int -> tag:string -> float -> unit
(** [log_scalar logger ~step ~epoch ~tag value] logs a single scalar metric.

    The [step] should be the global training step (e.g., [state.step] from
    {!Train_state.t}). The [epoch] allows dashboards to display "Epoch 5"
    alongside step numbers.

    {4 Example}
    {[
      log_scalar logger ~step:state.step ~epoch:5 ~tag:"train/loss" 0.1234
    ]} *)

val log_scalars : t -> step:int -> epoch:int -> (string * float) list -> unit
(** [log_scalars logger ~step ~epoch metrics] logs multiple scalars at once.

    {4 Example}
    {[
      log_scalars logger ~step:state.step ~epoch:5
        [ ("train/loss", 0.1234); ("train/accuracy", 0.95) ]
    ]} *)

(** {1 Metrics Integration} *)

val log_metrics :
  t -> step:int -> epoch:int -> prefix:string -> Metrics.Collection.t -> unit
(** [log_metrics logger ~step ~epoch ~prefix collection] logs all metrics from
    a Collection.

    Metric names are prefixed: ["train"] -> ["train/accuracy"],
    ["train/loss"], etc.

    {4 Example}
    {[
      log_metrics logger ~step:state.step ~epoch:5 ~prefix:"train" collection
    ]} *)

