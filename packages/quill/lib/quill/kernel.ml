(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type status = Starting | Idle | Busy | Shutting_down

type event =
  | Output of { cell_id : Cell.id; output : Cell.output }
  | Finished of { cell_id : Cell.id; success : bool }
  | Status_changed of status

type completion_kind =
  | Value
  | Type
  | Module
  | Module_type
  | Constructor
  | Label

type completion_item = {
  label : string;
  kind : completion_kind;
  detail : string;
}

type severity = Error | Warning

type diagnostic = {
  from_pos : int;
  to_pos : int;
  severity : severity;
  message : string;
}

type type_info = {
  typ : string;
  doc : string option;
  from_pos : int;
  to_pos : int;
}

type t = {
  execute : cell_id:Cell.id -> code:string -> unit;
  interrupt : unit -> unit;
  complete : code:string -> pos:int -> completion_item list;
  type_at : (code:string -> pos:int -> type_info option) option;
  diagnostics : (code:string -> diagnostic list) option;
  status : unit -> status;
  shutdown : unit -> unit;
}
