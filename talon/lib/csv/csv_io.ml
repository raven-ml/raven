(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Minimal RFC 4180 CSV parser and writer. *)

let parse_row separator line =
  let len = String.length line in
  let fields = ref [] in
  let buf = Buffer.create 64 in
  let i = ref 0 in
  while !i < len do
    if line.[!i] = '"' then (
      incr i;
      let in_quotes = ref true in
      while !i < len && !in_quotes do
        if line.[!i] = '"' then
          if !i + 1 < len && line.[!i + 1] = '"' then (
            Buffer.add_char buf '"';
            i := !i + 2)
          else (
            in_quotes := false;
            incr i)
        else (
          Buffer.add_char buf line.[!i];
          incr i)
      done;
      if !i < len && line.[!i] = separator then incr i;
      fields := Buffer.contents buf :: !fields;
      Buffer.clear buf)
    else if line.[!i] = separator then (
      fields := Buffer.contents buf :: !fields;
      Buffer.clear buf;
      incr i)
    else (
      Buffer.add_char buf line.[!i];
      incr i)
  done;
  fields := Buffer.contents buf :: !fields;
  List.rev !fields

let strip_cr line =
  let len = String.length line in
  if len > 0 && line.[len - 1] = '\r' then String.sub line 0 (len - 1) else line

let parse ?(separator = ',') content =
  let lines = String.split_on_char '\n' content in
  let lines = List.map strip_cr lines in
  let lines = List.filter (fun l -> l <> "") lines in
  List.map (parse_row separator) lines

let needs_quoting separator field =
  let len = String.length field in
  let rec check i =
    if i >= len then false
    else
      let c = field.[i] in
      c = separator || c = '"' || c = '\n' || c = '\r' || check (i + 1)
  in
  check 0

let quote_field separator field =
  if needs_quoting separator field then (
    let buf = Buffer.create (String.length field + 4) in
    Buffer.add_char buf '"';
    String.iter
      (fun c ->
        if c = '"' then Buffer.add_string buf "\"\"" else Buffer.add_char buf c)
      field;
    Buffer.add_char buf '"';
    Buffer.contents buf)
  else field

let write_row buf separator fields =
  List.iteri
    (fun i field ->
      if i > 0 then Buffer.add_char buf separator;
      Buffer.add_string buf (quote_field separator field))
    fields;
  Buffer.add_char buf '\n'

let serialize ?(separator = ',') rows =
  let buf = Buffer.create 1024 in
  List.iter (write_row buf separator) rows;
  Buffer.contents buf

let write_row_to_channel oc separator fields =
  let buf = Buffer.create 256 in
  write_row buf separator fields;
  output_string oc (Buffer.contents buf)

