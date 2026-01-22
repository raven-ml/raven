(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Incremental JSONL event reader with position tracking and rotation safety.

    This module provides efficient incremental reading of Kaun event log files.
    It tracks:
    - byte position (64-bit safe)
    - file identity (dev+inode) to detect rotation/replacement
    - a pending unterminated line fragment to avoid dropping partial JSON

    Reads return parsed {!Event.t} values.
*)

type t
(** Incremental reader state. *)

val create : file_path:string -> t
(** [create ~file_path] creates a reader for [file_path]. The file may not
    exist yet; reads will return [[]] until it appears. *)

val read_new : t -> Event.t list
(** [read_new r] reads only newly appended events since the last call.

    - Returns complete events in file order.
    - Incomplete trailing lines (missing '\n') are buffered and parsed when
      the line is completed later.
    - If the file is truncated or replaced (rotation), the reader resets to
      the beginning automatically.

    This function never raises for normal IO problems; it returns [[]] and
    keeps the reader usable. *)

val close : t -> unit
(** [close r] closes any open file handle. Safe to call multiple times. *)

val reset : t -> unit
(** [reset r] resets the reader to start from the beginning, dropping any
    buffered partial line. *)

val file_exists : t -> bool
(** [file_exists r] returns [true] if the path exists at call time. *)
