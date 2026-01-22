(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type scalar = {
  step : int;
  epoch : int option;
  tag : string;
  value : float;
}

type t =
  | Scalar of scalar
  | Unknown of Yojson.Basic.t
  | Malformed of { line : string; error : string }

(* ───── Small helpers ───── *)

let is_whitespace (c : char) =
  match c with
  | ' ' | '\t' | '\r' | '\n' -> true
  | _ -> false

let is_blank (s : string) =
  let len = String.length s in
  let rec loop i =
    if i >= len then true else if is_whitespace s.[i] then loop (i + 1) else false
  in
  loop 0

let get_string key (fields : (string * Yojson.Basic.t) list) =
  match List.assoc_opt key fields with
  | Some (`String s) -> Some s
  | _ -> None

let as_int (j : Yojson.Basic.t) : int option =
  match j with
  | `Int i -> Some i
  | _ -> None

let as_float (j : Yojson.Basic.t) : float option =
  match j with
  | `Float f -> Some f
  | `Int i -> Some (float_of_int i)
  | _ -> None

let get_int key fields =
  match List.assoc_opt key fields with
  | Some j -> as_int j
  | None -> None

let get_float key fields =
  match List.assoc_opt key fields with
  | Some j -> as_float j
  | None -> None

(* ───── Parsing ───── *)

let of_yojson (json : Yojson.Basic.t) : t =
  match json with
  | `Assoc fields -> (
      match get_string "type" fields with
      | Some "scalar" -> (
          match (get_int "step" fields, get_string "tag" fields, get_float "value" fields) with
          | Some step, Some tag, Some value ->
              let epoch = get_int "epoch" fields in
              Scalar { step; epoch; tag; value }
          | _ ->
              (* Schema mismatch despite "type":"scalar" *)
              Unknown json
        )
      | _ -> Unknown json)
  | _ -> Unknown json

let of_json_string (line : string) : t =
  try
    let json = Yojson.Basic.from_string line in
    of_yojson json
  with
  | exn ->
      Malformed { line; error = Printexc.to_string exn }

let parse_jsonl_chunk (chunk : string) : t list * string =
  let len = String.length chunk in
  let rec scan i line_start acc =
    if i >= len then
      (* trailing fragment without '\n' *)
      let pending =
        if line_start >= len then "" else String.sub chunk line_start (len - line_start)
      in
      (List.rev acc, pending)
    else if chunk.[i] = '\n' then
      let line_len = i - line_start in
      let line =
        if line_len <= 0 then ""
        else
          (* Strip CR if CRLF *)
          let effective_len =
            if line_len > 0 && chunk.[i - 1] = '\r' then line_len - 1 else line_len
          in
          if effective_len <= 0 then "" else String.sub chunk line_start effective_len
      in
      let acc =
        if line = "" || is_blank line then acc
        else
          let ev = of_json_string line in
          ev :: acc
      in
      scan (i + 1) (i + 1) acc
    else
      scan (i + 1) line_start acc
  in
  scan 0 0 []

let pp fmt (ev : t) =
  match ev with
  | Scalar s ->
      Format.fprintf fmt
        "Scalar{tag=%S; step=%d; epoch=%s; value=%g}"
        s.tag s.step
        (match s.epoch with None -> "None" | Some e -> "Some " ^ string_of_int e)
        s.value
  | Unknown _ ->
      Format.fprintf fmt "Unknown{...}"
  | Malformed { error; _ } ->
      Format.fprintf fmt "Malformed{error=%S}" error
