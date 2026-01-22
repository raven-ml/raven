(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Domain types and JSON parsing for Kaun event logs.

    Kaun logs events as JSON Lines (JSONL), one JSON object per line.

    This module owns:
    - The event domain model
    - Parsing from Yojson / strings
    - A helper to parse chunks of JSONL safely (only complete lines)
*)

(** Scalar metric event. *)
type scalar = {
  step : int;
  (** Training step. *)

  epoch : int option;
  (** Optional epoch number. Some loggers may not emit epochs. *)

  tag : string;
  (** Metric name, e.g. ["train/loss"]. *)

  value : float;
  (** Metric value. *)
}

(** Event type. *)
type t =
  | Scalar of scalar
  (** A recognized scalar metric event. *)

  | Unknown of Yojson.Basic.t
  (** Valid JSON, but not a recognized event schema. *)

  | Malformed of { line : string; error : string }
  (** The line was not valid JSON. Useful for debugging log corruption or
      writer/reader schema mismatches. *)

val of_yojson : Yojson.Basic.t -> t
(** [of_yojson json] parses a JSON value into an event. Unrecognized but valid
    schemas become {!Unknown}. *)

val of_json_string : string -> t
(** [of_json_string line] parses a single JSON object line into an event.
    Parse errors become {!Malformed}. *)

val parse_jsonl_chunk : string -> t list * string
(** [parse_jsonl_chunk chunk] parses a chunk of JSONL text.

    - Only lines terminated by '\n' are parsed into events.
    - The returned [pending] string is the trailing unterminated fragment
      (no '\n') which must be prepended to the next read.

    Empty/whitespace-only lines are ignored.

    CRLF is supported (a trailing '\r' before '\n' is stripped). *)

val pp : Format.formatter -> t -> unit
(** Pretty-printer for debugging. *)
