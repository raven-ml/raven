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
  | Display of { mime : string; data : string }

type Format.stag += Display_tag of { mime : string; data : string }

(* ───── Cells ───── *)

type t =
  | Code of {
      id : id;
      source : string;
      language : string;
      outputs : output list;
      execution_count : int;
    }
  | Text of { id : id; source : string }

let code ?id ?(language = "ocaml") source =
  let id = match id with Some id -> id | None -> fresh_id () in
  Code { id; source; language; outputs = []; execution_count = 0 }

let text ?id source =
  let id = match id with Some id -> id | None -> fresh_id () in
  Text { id; source }

let id = function Code c -> c.id | Text t -> t.id
let source = function Code c -> c.source | Text t -> t.source

let set_source s = function
  | Code c -> Code { c with source = s }
  | Text t -> Text { t with source = s }

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
  | [ Stdout prev ] -> begin
      match o with
      | Stdout next -> List.rev (Stdout (apply_cr (prev ^ next)) :: acc)
      | _ -> List.rev (o :: Stdout prev :: acc)
    end
  | out :: rest -> append_or_coalesce o (out :: acc) rest

let append_output o = function
  | Code c -> Code { c with outputs = append_or_coalesce o [] c.outputs }
  | Text _ as t -> t

let clear_outputs = function
  | Code c -> Code { c with outputs = [] }
  | Text _ as t -> t

let increment_execution_count = function
  | Code c -> Code { c with execution_count = c.execution_count + 1 }
  | Text _ as t -> t
