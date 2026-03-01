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
  let buf = Buffer.create (n + 2) in
  Buffer.add_string buf "c_";
  for _ = 1 to n do
    Buffer.add_char buf chars.[Random.int (String.length chars)]
  done;
  Buffer.contents buf

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

let append_output o = function
  | Code c ->
      let outputs =
        match (o, List.rev c.outputs) with
        | Stdout new_text, Stdout prev_text :: rest ->
            List.rev (Stdout (apply_cr (prev_text ^ new_text)) :: rest)
        | _ -> c.outputs @ [ o ]
      in
      Code { c with outputs }
  | Text _ as t -> t

let clear_outputs = function
  | Code c -> Code { c with outputs = [] }
  | Text _ as t -> t

let increment_execution_count = function
  | Code c -> Code { c with execution_count = c.execution_count + 1 }
  | Text _ as t -> t
