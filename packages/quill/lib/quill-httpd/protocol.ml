(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Quill

let err_expected_char : _ format = "expected '%c' at %d"
let err_invalid_token = "invalid token"
let err_unterminated_string = "unterminated string"
let err_unterminated_escape = "unterminated escape"
let err_short_unicode = "short unicode escape"
let err_bad_unicode = "bad unicode escape"
let err_bad_escape : _ format = "bad escape '\\%c'"
let err_unexpected_char : _ format = "unexpected char '%c' at %d"
let err_missing_field : _ format = "missing or invalid field '%s'"
let err_invalid_item : _ format = "invalid item in '%s'"
let err_unknown_kind : _ format = "unknown cell kind '%s'"
let err_unknown_msg_type : _ format = "unknown message type '%s'"
let err_expected_object = "expected JSON object"
let ( let* ) = Result.bind

(* ───── JSON helpers ───── *)

let json_escape s =
  let buf = Buffer.create (String.length s) in
  for i = 0 to String.length s - 1 do
    match String.unsafe_get s i with
    | '"' -> Buffer.add_string buf {|\"|}
    | '\\' -> Buffer.add_string buf {|\\|}
    | '\n' -> Buffer.add_string buf {|\n|}
    | '\r' -> Buffer.add_string buf {|\r|}
    | '\t' -> Buffer.add_string buf {|\t|}
    | c when Char.code c < 0x20 ->
        Buffer.add_string buf (Printf.sprintf "\\u%04x" (Char.code c))
    | c -> Buffer.add_char buf c
  done;
  Buffer.contents buf

let jstr s = Printf.sprintf {|"%s"|} (json_escape s)

(* ───── Minimal JSON parser ───── *)

type json =
  | Jstring of string
  | Jnumber of float
  | Jbool of bool
  | Jnull
  | Jarray of json list
  | Jobject of (string * json) list

exception Parse_error of string

let parse_json (s : string) : json =
  let len = String.length s in
  let pos = ref 0 in
  let peek () = if !pos < len then String.unsafe_get s !pos else '\x00' in
  let advance () = incr pos in
  let skip_ws () =
    while
      !pos < len
      &&
      let c = String.unsafe_get s !pos in
      c = ' ' || c = '\t' || c = '\n' || c = '\r'
    do
      incr pos
    done
  in
  let expect c =
    skip_ws ();
    if !pos < len && String.unsafe_get s !pos = c then incr pos
    else raise (Parse_error (Printf.sprintf err_expected_char c !pos))
  in
  let rec parse_value () =
    skip_ws ();
    match peek () with
    | '"' -> Jstring (parse_string ())
    | '{' -> parse_object ()
    | '[' -> parse_array ()
    | 't' ->
        if !pos + 4 <= len && String.sub s !pos 4 = "true" then (
          pos := !pos + 4;
          Jbool true)
        else raise (Parse_error err_invalid_token)
    | 'f' ->
        if !pos + 5 <= len && String.sub s !pos 5 = "false" then (
          pos := !pos + 5;
          Jbool false)
        else raise (Parse_error err_invalid_token)
    | 'n' ->
        if !pos + 4 <= len && String.sub s !pos 4 = "null" then (
          pos := !pos + 4;
          Jnull)
        else raise (Parse_error err_invalid_token)
    | c when c = '-' || (c >= '0' && c <= '9') -> parse_number ()
    | c -> raise (Parse_error (Printf.sprintf err_unexpected_char c !pos))
  and parse_string () =
    expect '"';
    let buf = Buffer.create 32 in
    let rec loop () =
      if !pos >= len then raise (Parse_error err_unterminated_string);
      match String.unsafe_get s !pos with
      | '"' ->
          advance ();
          Buffer.contents buf
      | '\\' ->
          advance ();
          if !pos >= len then raise (Parse_error err_unterminated_escape);
          (match String.unsafe_get s !pos with
          | '"' ->
              Buffer.add_char buf '"';
              advance ()
          | '\\' ->
              Buffer.add_char buf '\\';
              advance ()
          | '/' ->
              Buffer.add_char buf '/';
              advance ()
          | 'n' ->
              Buffer.add_char buf '\n';
              advance ()
          | 'r' ->
              Buffer.add_char buf '\r';
              advance ()
          | 't' ->
              Buffer.add_char buf '\t';
              advance ()
          | 'b' ->
              Buffer.add_char buf '\b';
              advance ()
          | 'u' ->
              advance ();
              if !pos + 4 > len then raise (Parse_error err_short_unicode);
              let hex =
                try int_of_string ("0x" ^ String.sub s !pos 4)
                with _ -> raise (Parse_error err_bad_unicode)
              in
              pos := !pos + 4;
              if hex < 0x80 then Buffer.add_char buf (Char.chr hex)
              else if hex < 0x800 then (
                Buffer.add_char buf (Char.chr (0xC0 lor (hex lsr 6)));
                Buffer.add_char buf (Char.chr (0x80 lor (hex land 0x3F))))
              else (
                Buffer.add_char buf (Char.chr (0xE0 lor (hex lsr 12)));
                Buffer.add_char buf
                  (Char.chr (0x80 lor ((hex lsr 6) land 0x3F)));
                Buffer.add_char buf (Char.chr (0x80 lor (hex land 0x3F))))
          | c -> raise (Parse_error (Printf.sprintf err_bad_escape c)));
          loop ()
      | c ->
          Buffer.add_char buf c;
          advance ();
          loop ()
    in
    loop ()
  and parse_number () =
    let start = !pos in
    if peek () = '-' then advance ();
    while
      !pos < len
      && String.unsafe_get s !pos >= '0'
      && String.unsafe_get s !pos <= '9'
    do
      advance ()
    done;
    if !pos < len && String.unsafe_get s !pos = '.' then (
      advance ();
      while
        !pos < len
        && String.unsafe_get s !pos >= '0'
        && String.unsafe_get s !pos <= '9'
      do
        advance ()
      done);
    if
      !pos < len
      && (String.unsafe_get s !pos = 'e' || String.unsafe_get s !pos = 'E')
    then (
      advance ();
      if
        !pos < len
        && (String.unsafe_get s !pos = '+' || String.unsafe_get s !pos = '-')
      then advance ();
      while
        !pos < len
        && String.unsafe_get s !pos >= '0'
        && String.unsafe_get s !pos <= '9'
      do
        advance ()
      done);
    Jnumber (float_of_string (String.sub s start (!pos - start)))
  and parse_object () =
    expect '{';
    skip_ws ();
    if peek () = '}' then (
      advance ();
      Jobject [])
    else
      let rec loop acc =
        skip_ws ();
        let key = parse_string () in
        expect ':';
        let value = parse_value () in
        let acc = (key, value) :: acc in
        skip_ws ();
        if peek () = ',' then (
          advance ();
          loop acc)
        else (
          expect '}';
          Jobject (List.rev acc))
      in
      loop []
  and parse_array () =
    expect '[';
    skip_ws ();
    if peek () = ']' then (
      advance ();
      Jarray [])
    else
      let rec loop acc =
        let value = parse_value () in
        let acc = value :: acc in
        skip_ws ();
        if peek () = ',' then (
          advance ();
          loop acc)
        else (
          expect ']';
          Jarray (List.rev acc))
      in
      loop []
  in
  parse_value ()

let get_string key fields =
  match List.assoc_opt key fields with
  | Some (Jstring s) -> Ok s
  | _ -> Error (Printf.sprintf err_missing_field key)

let get_int key fields =
  match List.assoc_opt key fields with
  | Some (Jnumber n) -> Ok (int_of_float n)
  | _ -> Error (Printf.sprintf err_missing_field key)

let get_string_list key fields =
  match List.assoc_opt key fields with
  | Some (Jarray items) ->
      let rec collect acc = function
        | [] -> Ok (List.rev acc)
        | Jstring s :: rest -> collect (s :: acc) rest
        | _ -> Error (Printf.sprintf err_invalid_item key)
      in
      collect [] items
  | _ -> Error (Printf.sprintf err_missing_field key)

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
  | Clear_outputs of { cell_id : string }
  | Clear_all_outputs
  | Save
  | Undo
  | Redo
  | Complete of { request_id : string; code : string; pos : int }

let parse_kind fields =
  match get_string "kind" fields with
  | Ok "code" -> Ok `Code
  | Ok "text" -> Ok `Text
  | Ok k -> Error (Printf.sprintf err_unknown_kind k)
  | Error e -> Error e

let client_msg_of_json s =
  match parse_json s with
  | Jobject fields -> (
      match get_string "type" fields with
      | Ok "update_source" ->
          let* cell_id = get_string "cell_id" fields in
          let* source = get_string "source" fields in
          Ok (Update_source { cell_id; source })
      | Ok "checkpoint" -> Ok Checkpoint
      | Ok "execute_cell" ->
          let* cell_id = get_string "cell_id" fields in
          Ok (Execute_cell { cell_id })
      | Ok "execute_cells" ->
          let* cell_ids = get_string_list "cell_ids" fields in
          Ok (Execute_cells { cell_ids })
      | Ok "execute_all" -> Ok Execute_all
      | Ok "interrupt" -> Ok Interrupt
      | Ok "insert_cell" ->
          let* pos = get_int "pos" fields in
          let* kind = parse_kind fields in
          Ok (Insert_cell { pos; kind })
      | Ok "delete_cell" ->
          let* cell_id = get_string "cell_id" fields in
          Ok (Delete_cell { cell_id })
      | Ok "move_cell" ->
          let* cell_id = get_string "cell_id" fields in
          let* pos = get_int "pos" fields in
          Ok (Move_cell { cell_id; pos })
      | Ok "set_cell_kind" ->
          let* cell_id = get_string "cell_id" fields in
          let* kind = parse_kind fields in
          Ok (Set_cell_kind { cell_id; kind })
      | Ok "clear_outputs" ->
          let* cell_id = get_string "cell_id" fields in
          Ok (Clear_outputs { cell_id })
      | Ok "clear_all_outputs" -> Ok Clear_all_outputs
      | Ok "save" -> Ok Save
      | Ok "undo" -> Ok Undo
      | Ok "redo" -> Ok Redo
      | Ok "complete" ->
          let* request_id = get_string "request_id" fields in
          let* code = get_string "code" fields in
          let* pos = get_int "pos" fields in
          Ok (Complete { request_id; code; pos })
      | Ok t -> Error (Printf.sprintf err_unknown_msg_type t)
      | Error e -> Error e)
  | _ -> Error err_expected_object
  | exception Parse_error msg -> Error msg

(* ───── Server message encoding ───── *)

let status_string = function
  | Session.Idle -> "idle"
  | Session.Queued -> "queued"
  | Session.Running -> "running"

let output_to_json (o : Cell.output) =
  match o with
  | Stdout text -> Printf.sprintf {|{"kind":"stdout","text":%s}|} (jstr text)
  | Stderr text -> Printf.sprintf {|{"kind":"stderr","text":%s}|} (jstr text)
  | Error text -> Printf.sprintf {|{"kind":"error","text":%s}|} (jstr text)
  | Display { mime; data } ->
      Printf.sprintf {|{"kind":"display","mime":%s,"data":%s}|} (jstr mime)
        (jstr data)

let cell_to_json (cell : Cell.t) (status : Session.cell_status) =
  match cell with
  | Code { id; source; language; outputs; execution_count } ->
      let outputs_json =
        "[" ^ String.concat "," (List.map output_to_json outputs) ^ "]"
      in
      Printf.sprintf
        {|{"id":%s,"kind":"code","source":%s,"language":%s,"outputs":%s,"execution_count":%d,"status":%s}|}
        (jstr id) (jstr source) (jstr language) outputs_json execution_count
        (jstr (status_string status))
  | Text { id; source } ->
      let html = Quill_markdown.Edit.to_html source in
      Printf.sprintf
        {|{"id":%s,"kind":"text","source":%s,"rendered_html":%s,"status":%s}|}
        (jstr id) (jstr source) (jstr html)
        (jstr (status_string status))

let notebook_to_json ~cells ~can_undo ~can_redo =
  let cells_json =
    "["
    ^ String.concat "," (List.map (fun (c, s) -> cell_to_json c s) cells)
    ^ "]"
  in
  Printf.sprintf {|{"type":"notebook","cells":%s,"can_undo":%b,"can_redo":%b}|}
    cells_json can_undo can_redo

let cell_output_to_json ~cell_id output =
  Printf.sprintf {|{"type":"cell_output","cell_id":%s,"output":%s}|}
    (jstr cell_id) (output_to_json output)

let cell_status_to_json ~cell_id status =
  Printf.sprintf {|{"type":"cell_status","cell_id":%s,"status":%s}|}
    (jstr cell_id)
    (jstr (status_string status))

let cell_inserted_to_json ~pos cell status =
  Printf.sprintf {|{"type":"cell_inserted","pos":%d,"cell":%s}|} pos
    (cell_to_json cell status)

let cell_deleted_to_json ~cell_id =
  Printf.sprintf {|{"type":"cell_deleted","cell_id":%s}|} (jstr cell_id)

let cell_moved_to_json ~cell_id ~pos =
  Printf.sprintf {|{"type":"cell_moved","cell_id":%s,"pos":%d}|} (jstr cell_id)
    pos

let cell_updated_to_json cell status =
  Printf.sprintf {|{"type":"cell_updated","cell_id":%s,"cell":%s}|}
    (jstr (Cell.id cell))
    (cell_to_json cell status)

let completion_kind_to_string = function
  | Kernel.Value -> "value"
  | Type -> "type"
  | Module -> "module"
  | Module_type -> "module_type"
  | Constructor -> "constructor"
  | Label -> "label"

let completion_item_to_json (item : Kernel.completion_item) =
  Printf.sprintf {|{"label":%s,"kind":%s,"detail":%s}|} (jstr item.label)
    (jstr (completion_kind_to_string item.kind))
    (jstr item.detail)

let completions_to_json ~request_id items =
  let items_json =
    "[" ^ String.concat "," (List.map completion_item_to_json items) ^ "]"
  in
  Printf.sprintf {|{"type":"completions","request_id":%s,"items":%s}|}
    (jstr request_id) items_json

let saved_to_json () = {|{"type":"saved"}|}

let undo_redo_to_json ~can_undo ~can_redo =
  Printf.sprintf {|{"type":"undo_redo","can_undo":%b,"can_redo":%b}|} can_undo
    can_redo

let error_to_json msg =
  Printf.sprintf {|{"type":"error","message":%s}|} (jstr msg)
