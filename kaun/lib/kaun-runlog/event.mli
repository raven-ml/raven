(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Training run events.

    Events represent discrete observations during training, such as scalar
    metrics. They are stored as JSON lines in the run's [events.jsonl] file. *)

type t =
  | Scalar of {
      step : int;  (** Training step (iteration count). *)
      epoch : int option;  (** Training epoch, if applicable. *)
      tag : string;  (** Metric name (e.g., ["loss"], ["accuracy"]). *)
      value : float;  (** Metric value. *)
      wall_time : float;  (** Unix timestamp when the event was recorded. *)
    }  (** A scalar metric observation. *)

(** {1 Serialization} *)

val of_json : Yojson.Safe.t -> (t, string) result
(** [of_json json] parses a JSON object into an event.

    Expects a ["type"] field to determine the event variant. For ["scalar"],
    requires ["step"], ["tag"], and ["value"] fields; ["epoch"] is optional.

    Returns [Error msg] if the JSON structure is invalid or the type is unknown.
*)

val to_json : t -> Yojson.Safe.t
(** [to_json event] serializes an event to a JSON object.

    The output includes a ["type"] field indicating the event variant. *)
