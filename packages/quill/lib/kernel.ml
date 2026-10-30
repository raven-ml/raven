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
  language : string;
  on_event : event -> unit;
  execute : cell_id:Cell.id -> code:string -> unit;
  interrupt : unit -> unit;
  complete : code:string -> pos:int -> completion_item list;
  type_at : code:string -> pos:int -> type_info option;
  diagnostics : code:string -> diagnostic list;
  is_complete : string -> bool;
  status : unit -> status;
  shutdown : unit -> unit;
}

let make ~language ~on_event ~execute ~interrupt ~complete
    ?(type_at = fun ~code:_ ~pos:_ -> None) ?(diagnostics = fun ~code:_ -> [])
    ?(is_complete = fun _ -> true) ~status ~shutdown () =
  if not (Cell.is_language language) then
    invalid_arg
      (Printf.sprintf "Quill.Kernel.make: %S is not a language" language);
  {
    language;
    on_event;
    execute;
    interrupt;
    complete;
    type_at;
    diagnostics;
    is_complete;
    status;
    shutdown;
  }

let language k = k.language

(* [source k c] is the source of [c] if [k] runs it. *)
let source k = function
  | Cell.Code { language; source; _ } when String.equal language k.language ->
      Some source
  | Cell.Code _ | Cell.Text _ -> None

let refusal ~language k =
  Printf.sprintf "no kernel runs %s code: this notebook's kernel runs %s"
    language k.language

let execute k = function
  | Cell.Text _ -> invalid_arg "Quill.Kernel.execute: a text cell"
  | Cell.Code { id; language; source; _ } ->
      if String.equal language k.language then
        k.execute ~cell_id:id ~code:source
      else (
        k.on_event
          (Output { cell_id = id; output = Error (refusal ~language k) });
        k.on_event (Finished { cell_id = id; success = false }))

let complete k c ~pos =
  match source k c with Some code -> k.complete ~code ~pos | None -> []

let type_at k c ~pos =
  match source k c with Some code -> k.type_at ~code ~pos | None -> None

let diagnostics k c =
  match source k c with Some code -> k.diagnostics ~code | None -> []

let is_complete k c =
  match source k c with Some code -> k.is_complete code | None -> true

let interrupt k = k.interrupt ()
let status k = k.status ()
let shutdown k = k.shutdown ()
