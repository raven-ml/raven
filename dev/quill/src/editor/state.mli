(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Int_set : Set.S with type elt = int

type load_state =
  | Idle
  | Loading of { path : string }
  | Load_failed of { path : string; error : string }

type caret = { block_id : int; inline_id : int option; offset : int }

type selection =
  | No_selection
  | Caret of caret
  | Range of { anchor : caret; focus : caret }

type config = { history_limit : int; auto_normalize : bool }
type snapshot = { document : Document.t; selection : selection }
type history = { past : snapshot list; future : snapshot list; capacity : int }

type t = {
  document : Document.t;
  selection : selection;
  load_state : load_state;
  running_blocks : Int_set.t;
  history : history;
  config : config;
}

val default_config : config
val init : t

val create :
  ?config:config -> ?document:Document.t -> ?selection:selection -> unit -> t

val with_document : ?config:config -> ?selection:selection -> Document.t -> t
val set_document : t -> Document.t -> t
val set_load_state : t -> load_state -> t
val set_selection : t -> selection -> t
val clear_selection : t -> t
val mark_block_running : t -> int -> t
val mark_block_idle : t -> int -> t
val is_block_running : t -> int -> bool
val snapshot : t -> snapshot
val restore : t -> config:config -> history:history -> t
val record_document_change : ?selection:selection -> t -> Document.t -> t
val has_undo : t -> bool
val has_redo : t -> bool
val undo : t -> t option
val redo : t -> t option
val selection_blocks : t -> int list
