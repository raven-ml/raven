type t

val custom : ?close:(unit -> unit) -> (Fehu.Render.t -> unit) -> t
(** Create a sink from a push callback and optional close action. *)

val noop : t
(** Sink that drops all frames. *)

val ffmpeg : ?fps:int -> path:string -> unit -> t
(** Stream frames to an ffmpeg process producing a video file. *)

val gif : ?fps:int -> path:string -> unit -> t
(** Stream frames to ffmpeg configured for GIF output. *)

val wandb : ?fps:int -> project:string -> run:string -> unit -> t
(** Capture frames, encode to MP4, and log to Weights & Biases via Python. *)

val close : t -> unit
(** Flush and finalize the sink. *)

val push : t -> Fehu.Render.t -> unit
(** Push a single frame to the sink. *)

val with_sink : (unit -> t) -> (t -> 'a) -> 'a
(** [with_sink create f] creates a sink via [create], runs [f], and guarantees
    the sink is closed afterwards. *)

val with_ffmpeg : ?fps:int -> path:string -> (t -> 'a) -> 'a
(** Convenience wrapper around {!ffmpeg} that automatically closes the sink. *)

val with_gif : ?fps:int -> path:string -> (t -> 'a) -> 'a
(** Convenience wrapper around {!gif} that automatically closes the sink. *)

val with_wandb : ?fps:int -> project:string -> run:string -> (t -> 'a) -> 'a
(** Convenience wrapper around {!wandb} that automatically closes the sink. *)
