(** Orbax-style checkpoint management for Kaun/Rune models.

    This module provides checkpoint functionality for saving and loading model
    parameters, optimizer states, and training metadata. *)

(** {1 Types} *)

type ('layout, 'dev) params =
  | Tensor of (float, 'layout, 'dev) Rune.t
  | List of ('layout, 'dev) params list
  | Record of (string * ('layout, 'dev) params) list
      (** Parameter tree structure matching Kaun's params type *)

type metadata = (string * string) list
(** Metadata as key-value pairs *)

type checkpoint_info = {
  step : int;  (** Training step number *)
  timestamp : float;  (** Unix timestamp when checkpoint was created *)
  metadata : metadata;  (** User-defined metadata *)
}
(** Information about a checkpoint *)

(** {1 Core Checkpointing} *)

module Checkpointer : sig
  type t
  (** Checkpointer handles saving and restoring of parameters *)

  val create : unit -> t
  (** Create a new checkpointer *)

  val save :
    t ->
    path:string ->
    params:('layout, 'dev) params ->
    ?metadata:metadata ->
    unit ->
    unit
  (** [save checkpointer ~path ~params ?metadata ()] saves parameters to disk.

      @param path Directory path where checkpoint will be saved
      @param params Parameter tree to save
      @param metadata Optional metadata to store with checkpoint *)

  val restore :
    t ->
    path:string ->
    device:'dev Rune.device ->
    dtype:(float, 'layout) Rune.dtype ->
    ('layout, 'dev) params
  (** [restore checkpointer ~path ~device ~dtype] loads parameters from disk.

      @param path Directory path to load checkpoint from
      @param device Device to load tensors onto
      @param dtype Data type for tensors
      @return Restored parameter tree *)

  val save_file :
    t ->
    path:string ->
    params:('layout, 'dev) params ->
    ?metadata:metadata ->
    unit ->
    unit
  (** [save_file checkpointer ~path ~params ?metadata ()] saves to a single
      file.

      Like [save] but uses a single file format instead of directory structure.
  *)

  val restore_file :
    t ->
    path:string ->
    device:'dev Rune.device ->
    dtype:(float, 'layout) Rune.dtype ->
    ('layout, 'dev) params
  (** [restore_file checkpointer ~path ~device ~dtype] loads from a single file.
  *)
end

(** {1 Checkpoint Management} *)

module CheckpointManager : sig
  type t
  (** Manages multiple checkpoints with automatic versioning and cleanup *)

  type options = {
    max_to_keep : int option;
        (** Maximum number of checkpoints to keep (None = keep all) *)
    keep_checkpoint_every_n_steps : int option;
        (** Keep checkpoint every N steps regardless of max_to_keep *)
    best_fn : (checkpoint_info -> float) option;
        (** Function to compute metric for best checkpoint selection *)
    best_mode : [ `min | `max ];
        (** Whether to minimize or maximize the best_fn metric *)
  }
  (** Configuration options for checkpoint manager *)

  val default_options : options
  (** Default options: keep 5 checkpoints, no periodic keeping, no best tracking
  *)

  val create :
    directory:string ->
    ?options:options ->
    ?checkpointer:Checkpointer.t ->
    unit ->
    t
  (** [create ~directory ?options ?checkpointer ()] creates a checkpoint
      manager.

      @param directory Base directory for all checkpoints
      @param options Configuration options
      @param checkpointer Custom checkpointer (default: new one created) *)

  val save :
    t ->
    step:int ->
    params:('layout, 'dev) params ->
    ?metadata:metadata ->
    ?metrics:(string * float) list ->
    unit ->
    unit
  (** [save manager ~step ~params ?metadata ?metrics ()] saves a checkpoint.

      The manager handles versioning, cleanup of old checkpoints, and tracking
      of best checkpoints based on metrics.

      @param step Current training step
      @param params Parameters to save
      @param metadata Optional metadata
      @param metrics Optional metrics for best checkpoint selection *)

  val restore :
    t ->
    device:'dev Rune.device ->
    dtype:(float, 'layout) Rune.dtype ->
    ?step:int ->
    unit ->
    ('layout, 'dev) params * checkpoint_info
  (** [restore manager ?step ~device ~dtype] restores a checkpoint.

      @param step Specific step to restore (default: latest)
      @param device Device to load tensors onto
      @param dtype Data type for tensors
      @return Restored parameters and checkpoint info *)

  val restore_best :
    t ->
    device:'dev Rune.device ->
    dtype:(float, 'layout) Rune.dtype ->
    ('layout, 'dev) params * checkpoint_info
  (** [restore_best manager ~device ~dtype] restores the best checkpoint.

      Requires [best_fn] to be set in options.
      @raise Invalid_argument if no best_fn is configured *)

  val latest_step : t -> int option
  (** Get the latest checkpoint step number, if any *)

  val best_step : t -> int option
  (** Get the best checkpoint step number, if any *)

  val all_steps : t -> int list
  (** Get all available checkpoint steps, sorted in ascending order *)

  val checkpoint_exists : t -> step:int -> bool
  (** Check if a checkpoint exists for a given step *)

  val delete : t -> step:int -> unit
  (** Delete a specific checkpoint *)

  val cleanup : t -> unit
  (** Manually trigger cleanup based on retention policy *)
end

(** {1 Utilities} *)

val save_params :
  path:string ->
  params:('layout, 'dev) params ->
  ?metadata:metadata ->
  unit ->
  unit
(** Convenience function to save parameters without a manager.

    @param path File or directory path for checkpoint
    @param params Parameters to save
    @param metadata Optional metadata *)

val load_params :
  path:string ->
  device:'dev Rune.device ->
  dtype:(float, 'layout) Rune.dtype ->
  ('layout, 'dev) params
(** Convenience function to load parameters without a manager.

    @param path File or directory path to load from
    @param device Device to load tensors onto
    @param dtype Data type for tensors *)

val params_to_json : ('layout, 'dev) params -> Yojson.Basic.t
(** Convert parameters to JSON representation for serialization *)

val params_from_json :
  Yojson.Basic.t ->
  device:'dev Rune.device ->
  dtype:(float, 'layout) Rune.dtype ->
  ('layout, 'dev) params
(** Convert JSON back to parameters *)

(** {1 Async Operations} *)

module Async : sig
  (** Asynchronous checkpoint operations for non-blocking saves *)

  type save_future
  (** Future representing an ongoing save operation *)

  val save :
    path:string ->
    params:('layout, 'dev) params ->
    ?metadata:metadata ->
    unit ->
    save_future
  (** Start an asynchronous save operation *)

  val wait : save_future -> unit
  (** Wait for save operation to complete *)

  val is_ready : save_future -> bool
  (** Check if save operation has completed *)
end
