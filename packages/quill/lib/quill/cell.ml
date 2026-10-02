(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Identifiers ───── *)

type id = string

let () = Random.self_init ()

let fresh_id () =
  let n = 12 in
  let chars = "abcdefghijklmnopqrstuvwxyz0123456789" in
  let b = Bytes.create (n + 2) in
  Bytes.unsafe_set b 0 'c';
  Bytes.unsafe_set b 1 '_';
  for i = 0 to n - 1 do
    Bytes.unsafe_set b (i + 2) chars.[Random.int 36]
  done;
  Bytes.unsafe_to_string b

(* ───── Outputs ───── *)

type output =
  | Stdout of string
  | Stderr of string
  | Error of string
  | Display of { mime : string; id : string option; data : string }

(* ───── Display protocol ───── *)

let display_prefix = "quill.display\n"

let base64_alphabet =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

let base64_encode s =
  let len = String.length s in
  let out = Bytes.create ((len + 2) / 3 * 4) in
  let digit n = String.unsafe_get base64_alphabet (n land 0x3f) in
  let byte i = if i < len then Char.code (String.unsafe_get s i) else 0 in
  for g = 0 to ((len + 2) / 3) - 1 do
    let i = 3 * g and j = 4 * g in
    let n = (byte i lsl 16) lor (byte (i + 1) lsl 8) lor byte (i + 2) in
    Bytes.unsafe_set out j (digit (n lsr 18));
    Bytes.unsafe_set out (j + 1) (digit (n lsr 12));
    Bytes.unsafe_set out (j + 2) (if i + 1 < len then digit (n lsr 6) else '=');
    Bytes.unsafe_set out (j + 3) (if i + 2 < len then digit n else '=')
  done;
  Bytes.unsafe_to_string out

let malformed what = Some (Error ("malformed display tag: " ^ what))

let output_of_tag tag =
  if not (String.starts_with ~prefix:display_prefix tag) then None
  else
    let mime_start = String.length display_prefix in
    match String.index_from_opt tag mime_start '\n' with
    | None -> malformed "no line ends the MIME type"
    | Some mime_end -> (
        match String.index_from_opt tag (mime_end + 1) '\n' with
        | None -> malformed "no line ends the display id"
        | Some _ when mime_end = mime_start ->
            malformed "the MIME type is empty"
        | Some id_end ->
            let mime = String.sub tag mime_start (mime_end - mime_start) in
            let id =
              if id_end = mime_end + 1 then None
              else Some (String.sub tag (mime_end + 1) (id_end - mime_end - 1))
            in
            let start = id_end + 1 in
            let content = String.sub tag start (String.length tag - start) in
            let data =
              if String.starts_with ~prefix:"image/" mime then
                base64_encode content
              else content
            in
            Some (Display { mime; id; data }))

(* ───── Attributes ───── *)

type attrs = { collapsed : bool; hide_source : bool }

let default_attrs = { collapsed = false; hide_source = false }

(* ───── Cells ───── *)

type t =
  | Code of {
      id : id;
      source : string;
      language : string;
      outputs : output list;
      execution_count : int;
      attrs : attrs;
    }
  | Text of { id : id; source : string; attrs : attrs }

let code ?id ?(language = "ocaml") ?(attrs = default_attrs) source =
  let id = match id with Some id -> id | None -> fresh_id () in
  Code { id; source; language; outputs = []; execution_count = 0; attrs }

let text ?id ?(attrs = default_attrs) source =
  let id = match id with Some id -> id | None -> fresh_id () in
  Text { id; source; attrs }

let id = function Code c -> c.id | Text t -> t.id
let source = function Code c -> c.source | Text t -> t.source
let attrs = function Code c -> c.attrs | Text t -> t.attrs

let set_source s = function
  | Code c -> Code { c with source = s }
  | Text t -> Text { t with source = s }

let set_attrs a = function
  | Code c -> Code { c with attrs = a }
  | Text t -> Text { t with attrs = a }

let set_outputs os = function
  | Code c -> Code { c with outputs = os }
  | Text _ as t -> t

let apply_cr s =
  let lines = String.split_on_char '\n' s in
  let apply_line line =
    match String.rindex_opt line '\r' with
    | None -> line
    | Some i -> String.sub line (i + 1) (String.length line - i - 1)
  in
  String.concat "\n" (List.map apply_line lines)

let rec append_or_coalesce o acc = function
  | [] -> List.rev (o :: acc)
  | [ Stdout prev ] ->
      begin match o with
      | Stdout next -> List.rev (Stdout (apply_cr (prev ^ next)) :: acc)
      | _ -> List.rev (o :: Stdout prev :: acc)
      end
  | out :: rest -> append_or_coalesce o (out :: acc) rest

let same_display d = function
  | Display { id = Some d'; _ } -> String.equal d d'
  | _ -> false

let append_output o = function
  | Text _ as t -> t
  | Code c -> (
      match o with
      | Display { id = Some d; _ } when List.exists (same_display d) c.outputs
        ->
          let replace out = if same_display d out then o else out in
          Code { c with outputs = List.map replace c.outputs }
      | _ -> Code { c with outputs = append_or_coalesce o [] c.outputs })

let clear_outputs = function
  | Code c -> Code { c with outputs = [] }
  | Text _ as t -> t

let increment_execution_count = function
  | Code c -> Code { c with execution_count = c.execution_count + 1 }
  | Text _ as t -> t
