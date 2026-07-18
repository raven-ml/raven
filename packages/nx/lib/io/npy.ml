(*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  NumPy's NPY container is deliberately implemented here from the published
  format rather than through Python-literal evaluation. The parser accepts the
  three specified header versions and only the scalar literal forms used by the
  format's required dictionary.
  --------------------------------------------------------------------------*)

open Bigarray

exception Read_error of string

let read_error fmt =
  Printf.ksprintf (fun message -> raise (Read_error message)) fmt

type bytes = (int, int8_unsigned_elt, c_layout) Array1.t
type packed_kind = K : ('a, 'b) Nx_buffer.kind -> packed_kind

type header = {
  kind : packed_kind;
  shape : int array;
  fortran_order : bool;
  swap_endian : bool;
  elements : int;
  element_size : int;
  data_offset : int;
  data_size : int;
}

type packed = P : ('a, 'b) Nx_buffer.t * int array -> packed

type encoded =
  | E : {
      header : string;
      data : ('a, 'b, c_layout) Array1.t;
      data_size : int;
    }
      -> encoded

let magic = "\x93NUMPY"
let max_header_size = 1 lsl 20

let checked_add context a b =
  if a < 0 || b < 0 || a > max_int - b then read_error "%s is too large" context;
  a + b

let checked_mul context a b =
  if a < 0 || b < 0 || (a <> 0 && b > max_int / a) then
    read_error "%s is too large" context;
  a * b

let byte src off limit index =
  let pos = checked_add "byte offset" off index in
  if pos >= limit then read_error "unexpected end of NPY header";
  Array1.unsafe_get src pos

let little_u16 src off limit =
  byte src off limit 0 lor (byte src off limit 1 lsl 8)

let little_u32 src off limit =
  let value =
    Int64.logor
      (Int64.of_int (byte src off limit 0))
      (Int64.logor
         (Int64.shift_left (Int64.of_int (byte src off limit 1)) 8)
         (Int64.logor
            (Int64.shift_left (Int64.of_int (byte src off limit 2)) 16)
            (Int64.shift_left (Int64.of_int (byte src off limit 3)) 24)))
  in
  if value > Int64.of_int max_int then read_error "NPY header is too large";
  Int64.to_int value

let substring src off len =
  String.init len (fun i -> Char.chr (Array1.unsafe_get src (off + i)))

type token =
  | Left_brace
  | Right_brace
  | Left_paren
  | Right_paren
  | Colon
  | Comma
  | String of string
  | Ident of string
  | Integer of int
  | End

type lexer = { text : string; mutable pos : int }

let is_space = function ' ' | '\t' | '\r' | '\n' -> true | _ -> false
let is_ident = function 'a' .. 'z' | 'A' .. 'Z' | '_' -> true | _ -> false

let rec skip_space lexer =
  if lexer.pos < String.length lexer.text && is_space lexer.text.[lexer.pos]
  then (
    lexer.pos <- lexer.pos + 1;
    skip_space lexer)

let lex_string lexer quote =
  let buffer = Buffer.create 16 in
  let rec loop () =
    if lexer.pos >= String.length lexer.text then
      read_error "unterminated string in NPY header";
    let c = lexer.text.[lexer.pos] in
    lexer.pos <- lexer.pos + 1;
    if c = quote then Buffer.contents buffer
    else if c = '\\' then (
      if lexer.pos >= String.length lexer.text then
        read_error "unterminated escape in NPY header";
      let escaped = lexer.text.[lexer.pos] in
      lexer.pos <- lexer.pos + 1;
      let decoded =
        match escaped with
        | '\\' -> '\\'
        | '\'' -> '\''
        | '"' -> '"'
        | 'n' -> '\n'
        | 'r' -> '\r'
        | 't' -> '\t'
        | c -> read_error "unsupported escape \\%c in NPY header" c
      in
      Buffer.add_char buffer decoded;
      loop ())
    else (
      Buffer.add_char buffer c;
      loop ())
  in
  loop ()

let lex_while lexer predicate =
  let start = lexer.pos in
  while
    lexer.pos < String.length lexer.text && predicate lexer.text.[lexer.pos]
  do
    lexer.pos <- lexer.pos + 1
  done;
  String.sub lexer.text start (lexer.pos - start)

let next lexer =
  skip_space lexer;
  if lexer.pos = String.length lexer.text then End
  else
    let c = lexer.text.[lexer.pos] in
    lexer.pos <- lexer.pos + 1;
    match c with
    | '{' -> Left_brace
    | '}' -> Right_brace
    | '(' -> Left_paren
    | ')' -> Right_paren
    | ':' -> Colon
    | ',' -> Comma
    | '\'' | '"' -> String (lex_string lexer c)
    | '-' | '0' .. '9' ->
        lexer.pos <- lexer.pos - 1;
        let literal =
          lex_while lexer (function '-' | '0' .. '9' -> true | _ -> false)
        in
        let value =
          match int_of_string_opt literal with
          | Some value -> value
          | None -> read_error "invalid integer %S in NPY header" literal
        in
        Integer value
    | c when is_ident c ->
        lexer.pos <- lexer.pos - 1;
        Ident (lex_while lexer is_ident)
    | c -> read_error "unexpected character %C in NPY header" c

let expect lexer expected =
  let actual = next lexer in
  if actual <> expected then read_error "malformed NPY header"

let parse_shape lexer =
  expect lexer Left_paren;
  let dimensions = ref [] in
  let rec loop () =
    match next lexer with
    | Right_paren -> Array.of_list (List.rev !dimensions)
    | Integer dim -> (
        if dim < 0 then read_error "negative NPY dimension %d" dim;
        dimensions := dim :: !dimensions;
        match next lexer with
        | Comma -> loop ()
        | Right_paren -> Array.of_list (List.rev !dimensions)
        | _ -> read_error "malformed NPY shape")
    | _ -> read_error "malformed NPY shape"
  in
  loop ()

let parse_dictionary text =
  let lexer = { text; pos = 0 } in
  expect lexer Left_brace;
  let descr = ref None in
  let fortran_order = ref None in
  let shape = ref None in
  let set_once name slot value =
    match !slot with
    | None -> slot := Some value
    | Some _ -> read_error "duplicate NPY header field %S" name
  in
  let rec fields () =
    match next lexer with
    | Right_brace -> ()
    | String name -> (
        expect lexer Colon;
        (match name with
        | "descr" -> (
            match next lexer with
            | String value -> set_once name descr value
            | _ -> read_error "NPY descr must be a string")
        | "fortran_order" -> (
            match next lexer with
            | Ident "True" -> set_once name fortran_order true
            | Ident "False" -> set_once name fortran_order false
            | _ -> read_error "NPY fortran_order must be True or False")
        | "shape" -> set_once name shape (parse_shape lexer)
        | _ -> read_error "unsupported NPY header field %S" name);
        match next lexer with
        | Comma -> fields ()
        | Right_brace -> ()
        | _ -> read_error "malformed NPY dictionary")
    | _ -> read_error "malformed NPY dictionary"
  in
  fields ();
  (match next lexer with
  | End -> ()
  | _ -> read_error "trailing NPY header data");
  let required name = function
    | Some value -> value
    | None -> read_error "missing NPY header field %S" name
  in
  ( required "descr" !descr,
    required "fortran_order" !fortran_order,
    required "shape" !shape )

let kind_of_code = function
  | "f2" -> K Nx_buffer.Float16
  | "f4" -> K Nx_buffer.Float32
  | "f8" -> K Nx_buffer.Float64
  | "i1" -> K Nx_buffer.Int8
  | "i2" -> K Nx_buffer.Int16
  | "i4" -> K Nx_buffer.Int32
  | "i8" -> K Nx_buffer.Int64
  | "u1" -> K Nx_buffer.UInt8
  | "u2" -> K Nx_buffer.UInt16
  | "u4" -> K Nx_buffer.UInt32
  | "u8" -> K Nx_buffer.UInt64
  | "c8" -> K Nx_buffer.Complex64
  | "c16" -> K Nx_buffer.Complex128
  | "b1" -> K Nx_buffer.Bool
  | code -> read_error "unsupported NPY dtype %S" code

let decode_descr descr =
  if String.length descr < 2 then read_error "invalid NPY dtype %S" descr;
  let endian = descr.[0] in
  let kind = kind_of_code (String.sub descr 1 (String.length descr - 1)) in
  let (K concrete) = kind in
  let element_size = Nx_buffer.kind_size_in_bytes concrete in
  let swap_endian =
    match endian with
    | '|' ->
        if element_size <> 1 then
          read_error "byte-order-independent dtype %S has multi-byte elements"
            descr;
        false
    | '=' -> false
    | '<' -> Sys.big_endian && element_size > 1
    | '>' -> (not Sys.big_endian) && element_size > 1
    | marker -> read_error "invalid NPY byte-order marker %C" marker
  in
  (kind, element_size, swap_endian)

let parse_header src ~off ~len =
  if off < 0 || len < 0 || off > Array1.dim src || len > Array1.dim src - off
  then invalid_arg "Npy.parse_header: byte span out of bounds";
  let limit = off + len in
  if len < 10 then read_error "truncated NPY preamble";
  for i = 0 to String.length magic - 1 do
    if byte src off limit i <> Char.code magic.[i] then
      read_error "not an NPY stream (bad magic)"
  done;
  let major = byte src off limit 6 in
  let minor = byte src off limit 7 in
  if minor <> 0 then read_error "unsupported NPY version %d.%d" major minor;
  let length_size, header_len =
    match major with
    | 1 -> (2, little_u16 src (off + 8) limit)
    | 2 | 3 -> (4, little_u32 src (off + 8) limit)
    | _ -> read_error "unsupported NPY version %d.%d" major minor
  in
  if header_len > max_header_size then
    read_error "NPY header exceeds the %d-byte safety limit" max_header_size;
  let preamble = 8 + length_size in
  let data_offset = checked_add "NPY header" preamble header_len in
  if data_offset > len then read_error "truncated NPY header";
  let text = substring src (off + preamble) header_len in
  let descr, fortran_order, shape = parse_dictionary text in
  let kind, element_size, swap_endian = decode_descr descr in
  let elements =
    Array.fold_left
      (fun product dim -> checked_mul "NPY shape" product dim)
      1 shape
  in
  let data_size = checked_mul "NPY payload" elements element_size in
  {
    kind;
    shape;
    fortran_order;
    swap_endian;
    elements;
    element_size;
    data_offset;
    data_size;
  }

type payload =
  | Stored of { src : bytes; off : int }
  | Deflated of { src : bytes; src_off : int; src_len : int; skip : int }

let flat_buffer buffer elements =
  Bigarray.reshape_1 (Nx_buffer.to_genarray buffer [| elements |]) elements

let materialize header payload =
  let (K kind) = header.kind in
  let buffer = Nx_buffer.create kind header.elements in
  let destination = flat_buffer buffer header.elements in
  let fill target =
    match payload with
    | Stored { src; off } ->
        Nx_io_codec.blit_bytes ~src ~src_off:off ~dst:target ~dst_off:0
          ~len:header.data_size;
        None
    | Deflated { src; src_off; src_len; skip } ->
        Some
          (Nx_io_codec.inflate_raw_into src ~src_off ~src_len ~skip ~dst:target
             ~dst_off:0 ~dst_len:header.data_size)
  in
  let crc =
    if header.fortran_order && Array.length header.shape > 1 then (
      let temporary = Array1.create int8_unsigned c_layout header.data_size in
      let checksum = fill temporary in
      Nx_io_codec.reorder_fortran_to_c ~src:temporary ~src_off:0
        ~dst:destination ~shape:header.shape ~element_size:header.element_size;
      checksum)
    else fill destination
  in
  if header.swap_endian then
    Nx_io_codec.byteswap destination ~element_size:header.element_size
      ~elements:header.elements;
  (P (buffer, Array.copy header.shape), crc)

let map_file fd size =
  if size = 0 then Array1.create int8_unsigned c_layout 0
  else
    Unix.map_file fd int8_unsigned c_layout false [| size |]
    |> Bigarray.array1_of_genarray

let with_fd path flags permissions f =
  let fd = Unix.openfile path flags permissions in
  Fun.protect ~finally:(fun () -> Unix.close fd) (fun () -> f fd)

let read_copy path =
  with_fd path [ Unix.O_RDONLY ] 0 @@ fun fd ->
  let size = (Unix.fstat fd).st_size in
  let src = map_file fd size in
  let header = parse_header src ~off:0 ~len:size in
  if header.data_offset > size || header.data_size <> size - header.data_offset
  then read_error "NPY payload size does not match its shape and dtype";
  materialize header (Stored { src; off = header.data_offset }) |> fst

let code_of_kind : type a b. (a, b) Nx_buffer.kind -> string = function
  | Float16 -> "f2"
  | Float32 -> "f4"
  | Float64 -> "f8"
  | Int8 -> "i1"
  | UInt8 -> "u1"
  | Int16 -> "i2"
  | UInt16 -> "u2"
  | Int32 -> "i4"
  | Int64 -> "i8"
  | UInt32 -> "u4"
  | UInt64 -> "u8"
  | Complex64 -> "c8"
  | Complex128 -> "c16"
  | Bool -> "b1"
  | BFloat16 | Float8_e4m3 | Float8_e5m2 | Int4 | UInt4 ->
      invalid_arg "dtype has no standard NPY representation"

let descr_of_kind kind =
  let size = Nx_buffer.kind_size_in_bytes kind in
  let endian = if size = 1 then '|' else if Sys.big_endian then '>' else '<' in
  Printf.sprintf "%c%s" endian (code_of_kind kind)

let shape_literal shape =
  match Array.to_list shape with
  | [] -> ""
  | [ dim ] -> Printf.sprintf "%d," dim
  | dimensions -> String.concat ", " (List.map string_of_int dimensions)

let padded_header ~version body =
  let length_size = if version = 1 then 2 else 4 in
  let preamble = 8 + length_size in
  let base = String.length body + 1 in
  let padding = (64 - ((preamble + base) mod 64)) mod 64 in
  body ^ String.make padding ' ' ^ "\n"

let encode_header kind shape =
  let body =
    Printf.sprintf "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }"
      (descr_of_kind kind) (shape_literal shape)
  in
  let header_v1 = padded_header ~version:1 body in
  let version, header =
    if String.length header_v1 <= 0xffff then (1, header_v1)
    else (2, padded_header ~version:2 body)
  in
  let length = String.length header in
  let length_bytes =
    if version = 1 then
      String.init 2 (fun i -> Char.chr ((length lsr (8 * i)) land 0xff))
    else String.init 4 (fun i -> Char.chr ((length lsr (8 * i)) land 0xff))
  in
  magic ^ String.make 1 (Char.chr version) ^ "\x00" ^ length_bytes ^ header

let encode (P (buffer, shape)) =
  let kind = Nx_buffer.kind buffer in
  let data_size =
    checked_mul "NPY payload" (Nx_buffer.length buffer)
      (Nx_buffer.kind_size_in_bytes kind)
  in
  E
    {
      header = encode_header kind shape;
      data = flat_buffer buffer (Nx_buffer.length buffer);
      data_size;
    }

let really_write_string fd text =
  let rec loop off =
    if off < String.length text then (
      let written =
        Unix.write_substring fd text off (String.length text - off)
      in
      if written = 0 then raise (Unix.Unix_error (Unix.EIO, "write", ""));
      loop (off + written))
  in
  loop 0

let write ?(exclusive = false) packed path =
  let flags =
    if exclusive then [ Unix.O_CREAT; Unix.O_EXCL; Unix.O_WRONLY ]
    else [ Unix.O_CREAT; Unix.O_TRUNC; Unix.O_WRONLY ]
  in
  let fd = Unix.openfile path flags 0o640 in
  match
    Fun.protect
      ~finally:(fun () -> Unix.close fd)
      (fun () ->
        let (E encoded) = encode packed in
        really_write_string fd encoded.header;
        Nx_io_codec.write_all fd encoded.data ~off:0 ~len:encoded.data_size)
  with
  | () -> ()
  | exception exn ->
      (if exclusive then try Sys.remove path with Sys_error _ -> ());
      raise exn
