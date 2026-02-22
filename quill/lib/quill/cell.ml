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

let append_output o = function
  | Code c -> Code { c with outputs = c.outputs @ [ o ] }
  | Text _ as t -> t

let clear_outputs = function
  | Code c -> Code { c with outputs = [] }
  | Text _ as t -> t

let increment_execution_count = function
  | Code c -> Code { c with execution_count = c.execution_count + 1 }
  | Text _ as t -> t
