(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  file : string option;
  line : int option;
  column : int option;
  row_group : int option;
  bytes : (int * int) option;
  text : string option;
  msg : string;
}

let err fmt = Format.kasprintf invalid_arg ("Error.v: " ^^ fmt)

let v ?file ?line ?column ?row_group ?bytes ?text msg =
  (match line with
  | Some l when l < 1 -> err "line %d is not positive" l
  | _ -> ());
  (match (line, column) with
  | _, Some c when c < 1 -> err "column %d is not positive" c
  | None, Some _ -> err "a column needs a line"
  | _ -> ());
  (match row_group with
  | Some g when g < 0 -> err "row group %d is negative" g
  | _ -> ());
  (match bytes with
  | Some (first, last) when first < 0 || last < first ->
      err "invalid byte range (%d, %d)" first last
  | _ -> ());
  { file; line; column; row_group; bytes; text; msg }

let text_limit = 64

(* [is_control u] is [true] iff [u] is a C0 or C1 control, or a bidirectional
   formatting control, which could reorder the text around it. *)
let is_control u =
  u < 0x20
  || (0x7f <= u && u < 0xa0)
  || (0x202a <= u && u <= 0x202e)
  || (0x2066 <= u && u <= 0x2069)

(* [pp_text ppf s] quotes [s], escaping it and cutting it at [text_limit] bytes,
   since it is untrusted input. *)
let pp_text ppf s =
  let b = Buffer.create text_limit in
  let hex c = Printf.bprintf b "\\x%02x" (Char.code c) in
  let rec loop i =
    if i >= String.length s then i
    else
      let d = String.get_utf_8_uchar s i in
      let n = Uchar.utf_decode_length d in
      if not (Uchar.utf_decode_is_valid d) then
        if i < text_limit then begin
          hex s.[i];
          loop (i + 1)
        end
        else i
      else if i + n > text_limit then i
      else begin
        (match s.[i] with
        | ('"' | '\\') as c ->
            Buffer.add_char b '\\';
            Buffer.add_char b c
        | _ when is_control (Uchar.to_int (Uchar.utf_decode_uchar d)) ->
            String.iter hex (String.sub s i n)
        | _ -> Buffer.add_string b (String.sub s i n));
        loop (i + n)
      end
  in
  let cut = loop 0 in
  Format.fprintf ppf "\"%s\"%s" (Buffer.contents b)
    (if cut < String.length s then "…" else "")

let pp_bytes ppf = function
  | first, last when first = last -> Format.fprintf ppf "byte %d" first
  | first, last -> Format.fprintf ppf "bytes %d-%d" first last

let pp ppf e =
  let place pp v = Format.fprintf ppf "%a: " pp v in
  (match (e.file, e.line, e.column) with
  | Some f, None, _ -> Format.fprintf ppf "%s: " f
  | Some f, Some l, None -> Format.fprintf ppf "%s:%d: " f l
  | Some f, Some l, Some c -> Format.fprintf ppf "%s:%d:%d: " f l c
  | None, Some l, None -> Format.fprintf ppf "line %d: " l
  | None, Some l, Some c -> Format.fprintf ppf "line %d, column %d: " l c
  | None, None, _ -> ());
  Option.iter (Format.fprintf ppf "row group %d: ") e.row_group;
  Option.iter (place pp_bytes) e.bytes;
  Option.iter (place pp_text) e.text;
  Format.pp_print_string ppf e.msg

let get_ok = function
  | Ok v -> v
  | Error e -> failwith (Format.asprintf "%a" pp e)
