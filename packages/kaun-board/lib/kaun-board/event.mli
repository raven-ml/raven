(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Training events.

    Events are discrete observations recorded during training. They are stored
    as JSON lines in a run's [events.jsonl] file.

    See {!Kaun_board.Log} for writing events and {!Kaun_board.Run} for reading
    them. *)

(** {1:types Types} *)

type direction = [ `Min | `Max ]
(** Whether the metric is better when minimized ([`Min]) or maximized ([`Max]).
    Used for best-value tracking; when omitted, the store uses a tag heuristic
    (e.g. tags containing "loss" or "error" prefer lower). *)

type t =
  | Scalar of {
      step : int;  (** Training step (iteration count). *)
      epoch : int option;  (** Training epoch, if any. *)
      tag : string;  (** Metric name (e.g. ["train/loss"]). *)
      value : float;  (** Metric value. *)
      wall_time : float;  (** Unix timestamp of the observation. *)
      direction : direction option;  (** Minimize or maximize for best value. *)
    }  (** A scalar metric observation. *)

(** {1:converting Converting} *)

val of_json : Jsont.json -> (t, string) result
(** [of_json json] is the event represented by [json].

    Errors if [json] is not a valid event object or has an unknown [type] field.
*)

val to_json : t -> Jsont.json
(** [to_json ev] is the JSON representation of [ev]. *)
