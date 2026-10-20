(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  NumPy's NPY format, implemented from its published description (NEP 1) rather
  than by evaluating Python literals: the parser accepts the three header
  versions and only the literal forms the format's dictionary uses.
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module L = Nx_array.Layout

let magic = "\x93NUMPY"

(* NumPy reads headers of at most this many bytes. *)
let max_header = 1 lsl 20

(* Dtypes *)

let code (type v s) (dt : (v, s) D.t) =
  match dt with
  | D.Float16 -> Some "f2"
  | D.Float32 -> Some "f4"
  | D.Float64 -> Some "f8"
  | D.Int8 -> Some "i1"
  | D.Int16 -> Some "i2"
  | D.Int32 -> Some "i4"
  | D.Int64 -> Some "i8"
  | D.Uint8 -> Some "u1"
  | D.Uint16 -> Some "u2"
  | D.Uint32 -> Some "u4"
  | D.Uint64 -> Some "u8"
  | D.Complex64 -> Some "c8"
  | D.Complex128 -> Some "c16"
  | D.Bool -> Some "b1"
  | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2 | D.Float4_e2m1fn | D.Int4
  | D.Uint4 | D.Bit ->
      None

let of_code c = List.find_opt (fun (D.Any dt) -> code dt = Some c) D.all

let refusal ~by dt =
  match dt with
  | D.Any D.Bit ->
      Fail.invalid_argf
        "%s: npy has no bit; save Nx.cast Nx.bool of it, a byte per element" by
  | D.Any dt -> Fail.invalid_argf "%s: npy has no %s" by (D.name dt)

let check ~by dt = if code dt = None then refusal ~by (D.Any dt)

(* Headers *)

type header = {
  dtype : D.any;
  shape : int array;
  fortran : bool;
  swap : bool;
  data : int;
  size : int;
}

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
let is_digit = function '-' | '0' .. '9' -> true | _ -> false

let lex_while l p =
  let start = l.pos in
  while l.pos < String.length l.text && p l.text.[l.pos] do
    l.pos <- l.pos + 1
  done;
  String.sub l.text start (l.pos - start)

let lex_string l quote =
  let b = Buffer.create 16 in
  let rec loop () =
    if l.pos >= String.length l.text then Fail.failf "unterminated string";
    let c = l.text.[l.pos] in
    l.pos <- l.pos + 1;
    if c = quote then Buffer.contents b
    else if c <> '\\' then (
      Buffer.add_char b c;
      loop ())
    else begin
      if l.pos >= String.length l.text then Fail.failf "unterminated escape";
      let e = l.text.[l.pos] in
      l.pos <- l.pos + 1;
      Buffer.add_char b
        (match e with
        | '\\' | '\'' | '"' -> e
        | 'n' -> '\n'
        | 'r' -> '\r'
        | 't' -> '\t'
        | c -> Fail.failf "the escape \\%c" c);
      loop ()
    end
  in
  loop ()

let next l =
  ignore (lex_while l is_space);
  if l.pos = String.length l.text then End
  else
    let c = l.text.[l.pos] in
    l.pos <- l.pos + 1;
    match c with
    | '{' -> Left_brace
    | '}' -> Right_brace
    | '(' -> Left_paren
    | ')' -> Right_paren
    | ':' -> Colon
    | ',' -> Comma
    | '\'' | '"' -> String (lex_string l c)
    | c when is_digit c -> (
        l.pos <- l.pos - 1;
        let s = lex_while l is_digit in
        match int_of_string_opt s with
        | Some n -> Integer n
        | None -> Fail.failf "the integer %S in the header" s)
    | c when is_ident c ->
        l.pos <- l.pos - 1;
        Ident (lex_while l is_ident)
    | c -> Fail.failf "the character %C in the header" c

let malformed () = Fail.failf "a malformed header"
let expect l t = if next l <> t then malformed ()

(* A shape, after its left parenthesis. *)
let shape l =
  let rec loop dims =
    match next l with
    | Right_paren -> Array.of_list (List.rev dims)
    | Integer d -> (
        if d < 0 then Fail.failf "the negative extent %d" d;
        match next l with
        | Comma -> loop (d :: dims)
        | Right_paren -> Array.of_list (List.rev (d :: dims))
        | _ -> malformed ())
    | _ -> malformed ()
  in
  loop []

let dictionary text =
  let l = { text; pos = 0 } in
  expect l Left_brace;
  let descr = ref None and fortran = ref None and dims = ref None in
  let once name slot v =
    if !slot <> None then Fail.failf "the field %S is given twice" name;
    slot := Some v
  in
  let rec fields () =
    match next l with
    | Right_brace -> ()
    | String name -> (
        expect l Colon;
        (match (name, next l) with
        | "descr", String v -> once name descr v
        | "fortran_order", Ident "True" -> once name fortran true
        | "fortran_order", Ident "False" -> once name fortran false
        | "shape", Left_paren -> once name dims (shape l)
        | ("descr" | "fortran_order" | "shape"), _ -> malformed ()
        | _ -> Fail.failf "the field %S" name);
        match next l with
        | Comma -> fields ()
        | Right_brace -> ()
        | _ -> malformed ())
    | _ -> malformed ()
  in
  fields ();
  if next l <> End then malformed ();
  let required name = function
    | Some v -> v
    | None -> Fail.failf "no field %S" name
  in
  ( required "descr" !descr,
    required "fortran_order" !fortran,
    required "shape" !dims )

(* A descr is a byte-order mark and a dtype's code, as ["<f4"]. Its elements are
   swapped where their order differs from the host's. *)
let descr d =
  let n = String.length d in
  if n < 2 then Fail.failf "the dtype %S" d;
  match of_code (String.sub d 1 (n - 1)) with
  | None -> Fail.failf "the dtype %S" d
  | Some (D.Any dt as any) ->
      let wide = D.bits dt > 8 in
      let swap =
        match d.[0] with
        | '=' -> false
        | '<' -> wide && Sys.big_endian
        | '>' -> wide && not Sys.big_endian
        | '|' when not wide -> false
        | _ -> Fail.failf "the dtype %S" d
      in
      (any, swap)

let le s i n =
  let v = ref 0 in
  for j = n - 1 downto 0 do
    v := (!v lsl 8) lor Char.code s.[i + j]
  done;
  !v

let header read ~length =
  let preamble = read 0 (Int.min length 12) in
  if
    String.length preamble < 10
    || not (String.starts_with ~prefix:magic preamble)
  then Fail.failf "no npy magic";
  let major = Char.code preamble.[6] and minor = Char.code preamble.[7] in
  let width =
    match (major, minor) with
    | 1, 0 -> 2
    | (2 | 3), 0 when String.length preamble = 12 -> 4
    | (2 | 3), 0 -> Fail.failf "the file ends inside its header"
    | _ -> Fail.failf "the npy version %d.%d" major minor
  in
  let n = le preamble 8 width in
  if n > max_header then Fail.failf "a header of more than %d bytes" max_header;
  let data = 8 + width + n in
  if data > length then Fail.failf "the file ends inside its header";
  let d, fortran, shape = dictionary (read (8 + width) n) in
  let dtype, swap = descr d in
  Storage.check_shape shape;
  let (D.Any dt) = dtype in
  let numel = Array.fold_left ( * ) 1 shape in
  let size = D.bytes dt numel in
  if size <> length - data then
    Fail.failf "%d bytes of data, %d for its dtype and shape" (length - data)
      size;
  { dtype; shape; fortran; swap; data; size }

(* Fortran order lays out the first axis fastest. *)
let layout h =
  if not h.fortran then L.contiguous h.shape
  else
    let n = Array.length h.shape in
    let strides = Array.make n 1 in
    for i = 1 to n - 1 do
      strides.(i) <- strides.(i - 1) * h.shape.(i - 1)
    done;
    L.v ~strides h.shape

(* The stream [b]'s array over its data's bytes, read by [read]. *)
let read_data
    (read :
      'v 's.
      Rig.Buffer.t ->
      ('v, 's) D.t ->
      Nx_array.Layout.t ->
      ('v, 's, Nx.host) Nx.t) h b =
  let (D.Any dt) = h.dtype in
  let x = read (Rig.Buffer.view b ~first:h.data ~length:h.size) dt (layout h) in
  Nx.P (if h.swap then Storage.byteswap x else x)

let value h b = read_data Storage.value h b
let copied h b = read_data Storage.copied h b

(* Writing *)

let shape_literal s =
  match Array.to_list s with
  | [] -> ""
  | [ d ] -> Printf.sprintf "%d," d
  | ds -> String.concat ", " (List.map string_of_int ds)

(* The header is padded with spaces so that the data starts on a multiple of 64
   bytes, and ends with a newline. *)
let padded ~width body =
  let used = 8 + width + String.length body + 1 in
  body ^ String.make ((64 - (used mod 64)) mod 64) ' ' ^ "\n"

let encode dt shape =
  let body =
    Printf.sprintf "{'descr': '%c%s', 'fortran_order': False, 'shape': (%s), }"
      (if D.bits dt > 8 then '<' else '|')
      (Option.get (code dt))
      (shape_literal shape)
  in
  let v1 = padded ~width:2 body in
  let version, text, width =
    if String.length v1 <= 0xffff then (1, v1, 2)
    else (2, padded ~width:4 body, 4)
  in
  let n = String.length text in
  magic
  ^ String.make 1 (Char.chr version)
  ^ "\x00"
  ^ String.init width (fun i -> Char.chr ((n lsr (8 * i)) land 0xff))
  ^ text
