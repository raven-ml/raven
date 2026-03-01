(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Quill

let ( let* ) = Result.bind

(* ───── JSON helpers ───── *)

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let json_mem name = function
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem name mems with
      | Some (_, v) -> v
      | None -> Jsont.Null ((), Jsont.Meta.none))
  | _ -> Jsont.Null ((), Jsont.Meta.none)

let json_of_string s =
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok v -> Ok v
  | Error e -> Error e

let json_to_string j =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json j with
  | Ok s -> s
  | Error e -> failwith e

(* ───── Field extraction ───── *)

let get_string name json =
  match json_mem name json with
  | Jsont.String (s, _) -> Ok s
  | _ -> Error (Printf.sprintf "missing or invalid field '%s'" name)

let get_int name json =
  match json_mem name json with
  | Jsont.Number (n, _) -> Ok (int_of_float n)
  | _ -> Error (Printf.sprintf "missing or invalid field '%s'" name)

let get_bool name json =
  match json_mem name json with Jsont.Bool (b, _) -> Ok b | _ -> Ok false

let get_string_list name json =
  match json_mem name json with
  | Jsont.Array (items, _) ->
      let rec collect acc = function
        | [] -> Ok (List.rev acc)
        | Jsont.String (s, _) :: rest -> collect (s :: acc) rest
        | _ :: _ -> Error (Printf.sprintf "invalid item in '%s'" name)
      in
      collect [] items
  | _ -> Error (Printf.sprintf "missing or invalid field '%s'" name)

(* ───── Client message parsing ───── *)

type client_msg =
  | Update_source of { cell_id : string; source : string }
  | Checkpoint
  | Execute_cell of { cell_id : string }
  | Execute_cells of { cell_ids : string list }
  | Execute_all
  | Interrupt
  | Insert_cell of { pos : int; kind : [ `Code | `Text ] }
  | Delete_cell of { cell_id : string }
  | Move_cell of { cell_id : string; pos : int }
  | Set_cell_kind of { cell_id : string; kind : [ `Code | `Text ] }
  | Set_cell_attrs of { cell_id : string; attrs : Cell.attrs }
  | Clear_outputs of { cell_id : string }
  | Clear_all_outputs
  | Save
  | Undo
  | Redo
  | Complete of { request_id : string; code : string; pos : int }
  | Type_at of { request_id : string; code : string; pos : int }
  | Diagnostics of { request_id : string; code : string }

let parse_kind json =
  match get_string "kind" json with
  | Ok "code" -> Ok `Code
  | Ok "text" -> Ok `Text
  | Ok k -> Error (Printf.sprintf "unknown cell kind '%s'" k)
  | Error e -> Error e

let client_msg_of_json s =
  match json_of_string s with
  | Error e -> Error e
  | Ok json -> (
      match get_string "type" json with
      | Ok "update_source" ->
          let* cell_id = get_string "cell_id" json in
          let* source = get_string "source" json in
          Ok (Update_source { cell_id; source })
      | Ok "checkpoint" -> Ok Checkpoint
      | Ok "execute_cell" ->
          let* cell_id = get_string "cell_id" json in
          Ok (Execute_cell { cell_id })
      | Ok "execute_cells" ->
          let* cell_ids = get_string_list "cell_ids" json in
          Ok (Execute_cells { cell_ids })
      | Ok "execute_all" -> Ok Execute_all
      | Ok "interrupt" -> Ok Interrupt
      | Ok "insert_cell" ->
          let* pos = get_int "pos" json in
          let* kind = parse_kind json in
          Ok (Insert_cell { pos; kind })
      | Ok "delete_cell" ->
          let* cell_id = get_string "cell_id" json in
          Ok (Delete_cell { cell_id })
      | Ok "move_cell" ->
          let* cell_id = get_string "cell_id" json in
          let* pos = get_int "pos" json in
          Ok (Move_cell { cell_id; pos })
      | Ok "set_cell_kind" ->
          let* cell_id = get_string "cell_id" json in
          let* kind = parse_kind json in
          Ok (Set_cell_kind { cell_id; kind })
      | Ok "set_cell_attrs" ->
          let* cell_id = get_string "cell_id" json in
          let* collapsed = get_bool "collapsed" json in
          let* hide_source = get_bool "hide_source" json in
          Ok
            (Set_cell_attrs { cell_id; attrs = { Cell.collapsed; hide_source } })
      | Ok "clear_outputs" ->
          let* cell_id = get_string "cell_id" json in
          Ok (Clear_outputs { cell_id })
      | Ok "clear_all_outputs" -> Ok Clear_all_outputs
      | Ok "save" -> Ok Save
      | Ok "undo" -> Ok Undo
      | Ok "redo" -> Ok Redo
      | Ok "complete" ->
          let* request_id = get_string "request_id" json in
          let* code = get_string "code" json in
          let* pos = get_int "pos" json in
          Ok (Complete { request_id; code; pos })
      | Ok "type_at" ->
          let* request_id = get_string "request_id" json in
          let* code = get_string "code" json in
          let* pos = get_int "pos" json in
          Ok (Type_at { request_id; code; pos })
      | Ok "diagnostics" ->
          let* request_id = get_string "request_id" json in
          let* code = get_string "code" json in
          Ok (Diagnostics { request_id; code })
      | Ok t -> Error (Printf.sprintf "unknown message type '%s'" t)
      | Error e -> Error e)

(* ───── Server message encoding ───── *)

let status_string = function
  | Session.Idle -> "idle"
  | Session.Queued -> "queued"
  | Session.Running -> "running"

let output_to_json (o : Cell.output) =
  match o with
  | Stdout text ->
      json_obj
        [
          ("kind", Jsont.Json.string "stdout"); ("text", Jsont.Json.string text);
        ]
  | Stderr text ->
      json_obj
        [
          ("kind", Jsont.Json.string "stderr"); ("text", Jsont.Json.string text);
        ]
  | Error text ->
      json_obj
        [
          ("kind", Jsont.Json.string "error"); ("text", Jsont.Json.string text);
        ]
  | Display { mime; data } ->
      json_obj
        [
          ("kind", Jsont.Json.string "display");
          ("mime", Jsont.Json.string mime);
          ("data", Jsont.Json.string data);
        ]

let attrs_to_json (a : Cell.attrs) =
  let pairs = ref [] in
  if a.hide_source then pairs := ("hide_source", Jsont.Json.bool true) :: !pairs;
  if a.collapsed then pairs := ("collapsed", Jsont.Json.bool true) :: !pairs;
  json_obj !pairs

let cell_to_json (cell : Cell.t) (status : Session.cell_status) =
  match cell with
  | Code { id; source; language; outputs; execution_count; attrs } ->
      json_obj
        [
          ("id", Jsont.Json.string id);
          ("kind", Jsont.Json.string "code");
          ("source", Jsont.Json.string source);
          ("language", Jsont.Json.string language);
          ("outputs", Jsont.Json.list (List.map output_to_json outputs));
          ("execution_count", Jsont.Json.int execution_count);
          ("status", Jsont.Json.string (status_string status));
          ("attrs", attrs_to_json attrs);
        ]
  | Text { id; source; attrs } ->
      let html = Quill_markdown.Edit.to_html source in
      json_obj
        [
          ("id", Jsont.Json.string id);
          ("kind", Jsont.Json.string "text");
          ("source", Jsont.Json.string source);
          ("rendered_html", Jsont.Json.string html);
          ("status", Jsont.Json.string (status_string status));
          ("attrs", attrs_to_json attrs);
        ]

let notebook_to_json ~cells ~can_undo ~can_redo =
  json_to_string
    (json_obj
       [
         ("type", Jsont.Json.string "notebook");
         ( "cells",
           Jsont.Json.list (List.map (fun (c, s) -> cell_to_json c s) cells) );
         ("can_undo", Jsont.Json.bool can_undo);
         ("can_redo", Jsont.Json.bool can_redo);
       ])

let cell_output_to_json ~cell_id output =
  json_to_string
    (json_obj
       [
         ("type", Jsont.Json.string "cell_output");
         ("cell_id", Jsont.Json.string cell_id);
         ("output", output_to_json output);
       ])

let cell_status_to_json ~cell_id status =
  json_to_string
    (json_obj
       [
         ("type", Jsont.Json.string "cell_status");
         ("cell_id", Jsont.Json.string cell_id);
         ("status", Jsont.Json.string (status_string status));
       ])

let cell_inserted_to_json ~pos cell status =
  json_to_string
    (json_obj
       [
         ("type", Jsont.Json.string "cell_inserted");
         ("pos", Jsont.Json.int pos);
         ("cell", cell_to_json cell status);
       ])

let cell_deleted_to_json ~cell_id =
  json_to_string
    (json_obj
       [
         ("type", Jsont.Json.string "cell_deleted");
         ("cell_id", Jsont.Json.string cell_id);
       ])

let cell_moved_to_json ~cell_id ~pos =
  json_to_string
    (json_obj
       [
         ("type", Jsont.Json.string "cell_moved");
         ("cell_id", Jsont.Json.string cell_id);
         ("pos", Jsont.Json.int pos);
       ])

let cell_updated_to_json cell status =
  json_to_string
    (json_obj
       [
         ("type", Jsont.Json.string "cell_updated");
         ("cell_id", Jsont.Json.string (Cell.id cell));
         ("cell", cell_to_json cell status);
       ])

let completion_kind_to_string = function
  | Kernel.Value -> "value"
  | Type -> "type"
  | Module -> "module"
  | Module_type -> "module_type"
  | Constructor -> "constructor"
  | Label -> "label"

let completion_item_to_json (item : Kernel.completion_item) =
  json_obj
    [
      ("label", Jsont.Json.string item.label);
      ("kind", Jsont.Json.string (completion_kind_to_string item.kind));
      ("detail", Jsont.Json.string item.detail);
    ]

let completions_to_json ~request_id items =
  json_to_string
    (json_obj
       [
         ("type", Jsont.Json.string "completions");
         ("request_id", Jsont.Json.string request_id);
         ("items", Jsont.Json.list (List.map completion_item_to_json items));
       ])

let type_at_to_json ~request_id info =
  let info_json =
    match info with
    | None -> Jsont.Json.null ()
    | Some (ti : Kernel.type_info) ->
        let doc_json =
          match ti.doc with
          | Some d -> Jsont.Json.string d
          | None -> Jsont.Json.null ()
        in
        json_obj
          [
            ("type", Jsont.Json.string ti.typ);
            ("doc", doc_json);
            ("from", Jsont.Json.int ti.from_pos);
            ("to", Jsont.Json.int ti.to_pos);
          ]
  in
  json_to_string
    (json_obj
       [
         ("type", Jsont.Json.string "type_at");
         ("request_id", Jsont.Json.string request_id);
         ("info", info_json);
       ])

let severity_to_string = function
  | Kernel.Error -> "error"
  | Warning -> "warning"

let diagnostic_to_json (d : Kernel.diagnostic) =
  json_obj
    [
      ("from", Jsont.Json.int d.from_pos);
      ("to", Jsont.Json.int d.to_pos);
      ("severity", Jsont.Json.string (severity_to_string d.severity));
      ("message", Jsont.Json.string d.message);
    ]

let diagnostics_to_json ~request_id items =
  json_to_string
    (json_obj
       [
         ("type", Jsont.Json.string "diagnostics");
         ("request_id", Jsont.Json.string request_id);
         ("items", Jsont.Json.list (List.map diagnostic_to_json items));
       ])

let saved_to_json () =
  json_to_string (json_obj [ ("type", Jsont.Json.string "saved") ])

let undo_redo_to_json ~can_undo ~can_redo =
  json_to_string
    (json_obj
       [
         ("type", Jsont.Json.string "undo_redo");
         ("can_undo", Jsont.Json.bool can_undo);
         ("can_redo", Jsont.Json.bool can_redo);
       ])

let error_to_json msg =
  json_to_string
    (json_obj
       [
         ("type", Jsont.Json.string "error"); ("message", Jsont.Json.string msg);
       ])
