(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  The SafeTensors format, as huggingface/safetensors's README specifies it: an
  8-byte little-endian header length, a JSON header, then the arrays' bytes.
  ---------------------------------------------------------------------------*)

module B = Rig.Buffer
module D = Nx_array.Dtype

let strf = Printf.sprintf

(* JSON, the subset a header uses: objects, arrays, strings and integers. *)

module Json = struct
  type t =
    [ `Assoc of (string * t) list
    | `String of string
    | `Int of int
    | `List of t list ]

  exception Parse_error of string

  (* Serialization *)

  let quote s =
    let buf = Buffer.create (String.length s + 2) in
    Buffer.add_char buf '"';
    String.iter
      (function
        | '"' -> Buffer.add_string buf "\\\""
        | '\\' -> Buffer.add_string buf "\\\\"
        | '\b' -> Buffer.add_string buf "\\b"
        | '\012' -> Buffer.add_string buf "\\f"
        | '\n' -> Buffer.add_string buf "\\n"
        | '\r' -> Buffer.add_string buf "\\r"
        | '\t' -> Buffer.add_string buf "\\t"
        | c when Char.code c < 0x20 ->
            Buffer.add_string buf (strf "\\u%04x" (Char.code c))
        | c -> Buffer.add_char buf c)
      s;
    Buffer.add_char buf '"';
    Buffer.contents buf

  let rec to_string : t -> string = function
    | `String s -> quote s
    | `Int i -> string_of_int i
    | `List l -> "[" ^ String.concat ", " (List.map to_string l) ^ "]"
    | `Assoc kv ->
        let pair (k, v) = strf "%s: %s" (quote k) (to_string v) in
        "{" ^ String.concat ", " (List.map pair kv) ^ "}"

  (* Parsing *)

  type parser = { input : string; mutable pos : int }

  let peek p =
    if p.pos < String.length p.input then Some p.input.[p.pos] else None

  let advance p = p.pos <- p.pos + 1

  let skip_ws p =
    while
      p.pos < String.length p.input
      &&
      match p.input.[p.pos] with
      | ' ' | '\t' | '\n' | '\r' -> true
      | _ -> false
    do
      advance p
    done

  let expect p c =
    skip_ws p;
    match peek p with
    | Some ch when ch = c -> advance p
    | Some ch -> raise (Parse_error (strf "expected '%c' got '%c'" c ch))
    | None -> raise (Parse_error (strf "expected '%c' got EOF" c))

  let take p c =
    match peek p with
    | Some ch when ch = c -> advance p
    | _ -> raise (Parse_error (strf "expected '%c'" c))

  let hex4 p =
    let value = ref 0 in
    for i = 0 to 3 do
      let digit =
        match peek p with
        | Some ('0' .. '9' as c) -> Char.code c - Char.code '0'
        | Some ('a' .. 'f' as c) -> Char.code c - Char.code 'a' + 10
        | Some ('A' .. 'F' as c) -> Char.code c - Char.code 'A' + 10
        | _ ->
            raise (Parse_error (strf "invalid Unicode escape digit %d" (i + 1)))
      in
      value := (!value lsl 4) lor digit;
      advance p
    done;
    !value

  let unicode p =
    let first = hex4 p in
    if first >= 0xd800 && first <= 0xdbff then begin
      take p '\\';
      take p 'u';
      let second = hex4 p in
      if second < 0xdc00 || second > 0xdfff then
        raise (Parse_error "invalid surrogate pair");
      Uchar.of_int (0x10000 + ((first - 0xd800) lsl 10) + second - 0xdc00)
    end
    else if first >= 0xdc00 && first <= 0xdfff then
      raise (Parse_error "unpaired low surrogate")
    else Uchar.of_int first

  let parse_string p =
    expect p '"';
    let buf = Buffer.create 16 in
    let rec loop () =
      match peek p with
      | None -> raise (Parse_error "unterminated string")
      | Some '"' ->
          advance p;
          Buffer.contents buf
      | Some '\\' ->
          advance p;
          (match peek p with
          | None -> raise (Parse_error "unterminated escape")
          | Some c -> (
              advance p;
              match c with
              | '"' | '\\' | '/' -> Buffer.add_char buf c
              | 'b' -> Buffer.add_char buf '\b'
              | 'f' -> Buffer.add_char buf '\012'
              | 'n' -> Buffer.add_char buf '\n'
              | 'r' -> Buffer.add_char buf '\r'
              | 't' -> Buffer.add_char buf '\t'
              | 'u' -> Buffer.add_utf_8_uchar buf (unicode p)
              | _ -> raise (Parse_error "invalid escape")));
          loop ()
      | Some c ->
          if Char.code c < 0x20 then
            raise (Parse_error "unescaped control character");
          Buffer.add_char buf c;
          advance p;
          loop ()
    in
    loop ()

  let parse_int p =
    skip_ws p;
    let start = p.pos in
    (match peek p with Some '-' -> advance p | _ -> ());
    while
      p.pos < String.length p.input
      && match p.input.[p.pos] with '0' .. '9' -> true | _ -> false
    do
      advance p
    done;
    let s = String.sub p.input start (p.pos - start) in
    try int_of_string s with _ -> raise (Parse_error ("invalid number: " ^ s))

  (* A header nests at most three deep: the header, an array's description and
     its shape or offsets. Deeper values are refused before they are read, so
     that no header can exhaust the stack. *)
  let max_depth = 3

  let rec parse_value depth p =
    if depth > max_depth then
      raise (Parse_error (strf "values nested more than %d deep" max_depth));
    skip_ws p;
    match peek p with
    | None -> raise (Parse_error "unexpected EOF")
    | Some '"' -> `String (parse_string p)
    | Some '{' -> parse_object (depth + 1) p
    | Some '[' -> parse_list (depth + 1) p
    | Some ('-' | '0' .. '9') -> `Int (parse_int p)
    | Some c -> raise (Parse_error (strf "unexpected char: '%c'" c))

  and parse_list depth p =
    expect p '[';
    skip_ws p;
    if peek p = Some ']' then (
      advance p;
      `List [])
    else
      let rec loop acc =
        let v = parse_value depth p in
        skip_ws p;
        match peek p with
        | Some ',' ->
            advance p;
            loop (v :: acc)
        | Some ']' ->
            advance p;
            `List (List.rev (v :: acc))
        | _ -> raise (Parse_error "expected ',' or ']'")
      in
      loop []

  and parse_object depth p =
    expect p '{';
    skip_ws p;
    if peek p = Some '}' then (
      advance p;
      `Assoc [])
    else
      let rec loop acc =
        skip_ws p;
        let key = parse_string p in
        skip_ws p;
        expect p ':';
        let value = parse_value depth p in
        skip_ws p;
        match peek p with
        | Some ',' ->
            advance p;
            loop ((key, value) :: acc)
        | Some '}' ->
            advance p;
            `Assoc (List.rev ((key, value) :: acc))
        | _ -> raise (Parse_error "expected ',' or '}'")
      in
      loop []

  let from_string s : t =
    let p = { input = s; pos = 0 } in
    try
      let v = parse_value 0 p in
      skip_ws p;
      if p.pos < String.length s then
        raise (Parse_error "trailing characters after JSON");
      v
    with Parse_error msg ->
      raise (Parse_error (strf "at position %d: %s" p.pos msg))
end

(* Dtypes *)

(* How an entry's dtype loads: as an nx dtype, or as its bytes, keeping the
   entry's shape where its elements are whole bytes. *)
type kind = Dtype of D.any | Bytes of { whole : bool }

(* The format's dtypes, by name, with their width in bits, in the order of the
   reference implementation's [Dtype] enum (safetensors/src/tensor.rs). *)
let dtypes =
  [
    ("BOOL", 8, Dtype (D.Any D.Bool));
    ("F4", 4, Bytes { whole = false });
    ("F6_E2M3", 6, Bytes { whole = false });
    ("F6_E3M2", 6, Bytes { whole = false });
    ("U8", 8, Dtype (D.Any D.Uint8));
    ("I8", 8, Dtype (D.Any D.Int8));
    ("F8_E5M2", 8, Dtype (D.Any D.Float8_e5m2));
    ("F8_E4M3", 8, Dtype (D.Any D.Float8_e4m3fn));
    ("F8_E8M0", 8, Bytes { whole = true });
    ("F8_E4M3FNUZ", 8, Bytes { whole = true });
    ("F8_E5M2FNUZ", 8, Bytes { whole = true });
    ("I16", 16, Dtype (D.Any D.Int16));
    ("U16", 16, Dtype (D.Any D.Uint16));
    ("F16", 16, Dtype (D.Any D.Float16));
    ("BF16", 16, Dtype (D.Any D.Bfloat16));
    ("I32", 32, Dtype (D.Any D.Int32));
    ("U32", 32, Dtype (D.Any D.Uint32));
    ("F32", 32, Dtype (D.Any D.Float32));
    ("C64", 64, Dtype (D.Any D.Complex64));
    ("F64", 64, Dtype (D.Any D.Float64));
    ("I64", 64, Dtype (D.Any D.Int64));
    ("U64", 64, Dtype (D.Any D.Uint64));
  ]

(* [code dt] is [dt]'s name in the format and its rank in [dtypes], by which a
   file orders its arrays, as the reference implementation does. *)
let code (type v s) (dt : (v, s) D.t) =
  let rec find i = function
    | [] -> None
    | (name, _, Dtype (D.Any d)) :: _ when D.equal d dt -> Some (name, i)
    | _ :: rest -> find (i + 1) rest
  in
  find 0 dtypes

(* Headers *)

let max_header = 100_000_000

type entry = {
  name : string;
  kind : kind;
  shape : int array;
  first : int; (* From the data's start. *)
  length : int;
}

let is_valid_utf8 s =
  let rec go i =
    i >= String.length s
    ||
    let d = String.get_utf_8_uchar s i in
    Uchar.utf_decode_is_valid d && go (i + Uchar.utf_decode_length d)
  in
  go 0

(* The header's key for the file's metadata, which names no array. *)
let metadata = "__metadata__"

let check_name ~by name =
  if not (is_valid_utf8 name) then
    Fail.invalid_argf "%s: %S is not UTF-8" by name;
  if name = metadata then
    Fail.invalid_argf "%s: %S names the file's metadata" by name

let int = function `Int n -> n | _ -> failwith "not an integer"

let entry (name, j) =
  let field f =
    match j with
    | `Assoc kv -> (
        match List.assoc_opt f kv with
        | Some v -> v
        | None -> Fail.failf "%s: no %s" name f)
    | _ -> Fail.failf "%s: not an object" name
  in
  if name = "" then failwith "an array has an empty name";
  let _, bits, kind =
    match field "dtype" with
    | `String c -> (
        match List.find_opt (fun (c', _, _) -> c = c') dtypes with
        | Some d -> d
        | None -> Fail.failf "%s: the dtype %S" name c)
    | _ -> Fail.failf "%s: a dtype that is not a string" name
  in
  let ints f =
    match field f with
    | `List l -> (
        try List.map int l
        with Failure _ -> Fail.failf "%s: %s holds a non-integer" name f)
    | _ -> Fail.failf "%s: %s is not an array" name f
  in
  let shape = Array.of_list (ints "shape") in
  let first, stop =
    match ints "data_offsets" with
    | [ s; e ] when 0 <= s && s <= e -> (s, e)
    | _ -> Fail.failf "%s: data_offsets is not a range" name
  in
  (try Storage.check_shape shape with Failure m -> Fail.failf "%s: %s" name m);
  let numel = Array.fold_left ( * ) 1 shape in
  if numel * bits mod 8 <> 0 then
    Fail.failf "%s: its elements do not fill whole bytes" name;
  if stop - first <> numel * bits / 8 then
    Fail.failf "%s: %d bytes, %d for its dtype and shape" name (stop - first)
      (numel * bits / 8);
  { name; kind; shape; first; length = stop - first }

(* The header's entries, ordered by offset, and the length of the data they
   cover, which they cover without a gap. *)
let parse text =
  if not (is_valid_utf8 text) then failwith "the header is not UTF-8";
  let fields =
    match Json.from_string text with
    | `Assoc kv -> kv
    | _ -> failwith "the header is not an object"
    | exception Json.Parse_error e -> Fail.failf "the header's JSON %s" e
  in
  let names = Hashtbl.create (List.length fields) in
  let entries =
    List.filter_map
      (fun (name, v) ->
        if Hashtbl.mem names name then Fail.failf "%s is named twice" name;
        Hashtbl.add names name ();
        if name <> metadata then Some (entry (name, v))
        else
          match v with
          | `Assoc kv ->
              List.iter
                (function
                  | _, `String _ -> ()
                  | k, _ -> Fail.failf "the metadata %s is not a string" k)
                kv;
              None
          | _ -> failwith "the metadata is not an object")
      fields
  in
  let entries = List.sort (fun a b -> compare a.first b.first) entries in
  let stop =
    List.fold_left
      (fun pos e ->
        if e.first <> pos then
          Fail.failf "%s: data_offsets leave a gap or overlap" e.name;
        e.first + e.length)
      0 entries
  in
  (entries, stop)

let load file =
  let length = B.length file in
  if length < 8 then Fail.failf "%d bytes, too short for SafeTensors" length;
  let n = String.get_int64_le (Storage.string file ~pos:0 8) 0 in
  if Int64.compare n 0L < 0 || Int64.compare n (Int64.of_int max_header) > 0
  then Fail.failf "a header length of %Lu bytes, above %d" n max_header;
  let n = Int64.to_int n in
  if 8 + n > length then
    Fail.failf "a header of %d bytes in a file of %d" n length;
  let entries, data = parse (Storage.string file ~pos:8 n) in
  if 8 + n + data <> length then
    Fail.failf "the header describes a file of %d bytes, the file has %d"
      (8 + n + data)
      length;
  List.fold_left
    (fun archive e ->
      let b = B.view file ~first:(8 + n + e.first) ~length:e.length in
      let value =
        match e.kind with
        | Dtype (D.Any dt) ->
            Nx.P
              (Storage.little b dt
                 (Nx_array.Layout.contiguous (Iarray.of_array e.shape)))
        | Bytes { whole } ->
            let shape = if whole then e.shape else [| e.length |] in
            Nx.P
              (Storage.value b D.Uint8
                 (Nx_array.Layout.contiguous (Iarray.of_array shape)))
      in
      Archive.add e.name value archive)
    Archive.empty entries

(* Writing *)

(* [header entries] is the header of a file of [entries], each a name, a dtype
   code, a shape and a length in bytes, whose data follows in that order. The
   JSON is padded with spaces to a multiple of 8 bytes. *)
let header entries =
  let field (pos, fields) (name, code, shape, length) =
    let shape = List.map (fun d -> `Int d) (Array.to_list shape) in
    let info =
      `Assoc
        [
          ("dtype", `String code);
          ("shape", `List shape);
          ("data_offsets", `List [ `Int pos; `Int (pos + length) ]);
        ]
    in
    (pos + length, (name, info) :: fields)
  in
  let _, fields = List.fold_left field (0, []) entries in
  let json = Json.to_string (`Assoc (List.rev fields)) in
  let n = (String.length json + 7) / 8 * 8 in
  let b = Bytes.make (8 + n) ' ' in
  Bytes.set_int64_le b 0 (Int64.of_int n);
  Bytes.blit_string json 0 b 8 (String.length json);
  Bytes.unsafe_to_string b
