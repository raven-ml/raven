(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Incremental JSONL event reader with position tracking.

    This module provides efficient incremental reading of event log files.
    Instead of re-reading the entire file on each update, it tracks the last
    read position and only reads new lines that were appended since the last
    read.

    {2 Usage}

    {[
      let reader = Event_reader.create ~file_path:"runs/xyz/events.jsonl" in
      
      (* Initial read - loads all events *)
      let events = Event_reader.read_new reader in
      
      (* Later reads - only new events *)
      let new_events = Event_reader.read_new reader in
      
      (* Cleanup *)
      Event_reader.close reader
    ]}

    {2 Performance}

    For long-running training with thousands of events, incremental reading
    provides 100-1000x speedup compared to re-reading the entire file.

    Complexity:
    - Initial read: O(n) where n is total events
    - Subsequent reads: O(m) where m is new events since last read *)

(** {1 Event Types} *)

type scalar = {
  step : int;
  epoch : int;
  tag : string;
  value : float;
}
(** Scalar metric event. *)

type event = Scalar of scalar | Unknown of Yojson.Basic.t
(** Event type - either a recognized scalar or unknown event. *)

(** {1 Reader} *)

type t
(** Incremental event reader with position tracking. *)

val create : file_path:string -> t
(** [create ~file_path] creates a new event reader for the given file.

    The reader starts at position 0. The file doesn't need to exist yet;
    reads will return empty lists until the file is created. *)

val read_new : t -> event list
(** [read_new reader] reads only new events since the last read.

    Returns empty list if:
    - File doesn't exist
    - File hasn't been modified since last read
    - No new complete lines available

    The reader tracks byte position and modification time to efficiently
    detect new data. Channel is kept open across reads for performance.

    Handles partial last lines gracefully - incomplete lines are skipped
    and will be read when complete on the next call. *)

val close : t -> unit
(** [close reader] closes the reader and releases file handles.

    Safe to call multiple times. *)

val reset : t -> unit
(** [reset reader] resets the reader to start from the beginning.

    The next [read_new] call will re-read all events from the file.
    Useful if the file was truncated or recreated. *)

val file_exists : t -> bool
(** [file_exists reader] checks if the reader's file currently exists. *)
