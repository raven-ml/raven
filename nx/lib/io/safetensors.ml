(*---------------------------------------------------------------------------
  Safetensors format reader/writer.

  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Result monad *)

let ( let* ) x f = match x with Ok v -> f v | Error _ as e -> e

(*---------------------------------------------------------------------------
  Minimal JSON codec (subset needed for safetensors headers)
  ---------------------------------------------------------------------------*)

module Json = struct
  type t =
    [ `Assoc of (string * t) list
    | `String of string
    | `Int of int
    | `List of t list ]

  exception Parse_error of string

  (* Serialization *)

  let rec to_string = function
    | `String s -> strf "\"%s\"" (String.escaped s)
    | `Int i -> string_of_int i
    | `List l -> "[" ^ String.concat ", " (List.map to_string l) ^ "]"
    | `Assoc kv ->
        let pair (k, v) = strf "\"%s\": %s" (String.escaped k) (to_string v) in
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
          | Some 'n' -> Buffer.add_char buf '\n'
          | Some 'r' -> Buffer.add_char buf '\r'
          | Some 't' -> Buffer.add_char buf '\t'
          | Some (('"' | '\\') as c) -> Buffer.add_char buf c
          | Some c -> Buffer.add_char buf c);
          advance p;
          loop ()
      | Some c ->
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

  let rec parse_value p =
    skip_ws p;
    match peek p with
    | None -> raise (Parse_error "unexpected EOF")
    | Some '"' -> `String (parse_string p)
    | Some '{' -> parse_object p
    | Some '[' -> parse_list p
    | Some ('-' | '0' .. '9') -> `Int (parse_int p)
    | Some c -> raise (Parse_error (strf "unexpected char: '%c'" c))

  and parse_list p =
    expect p '[';
    skip_ws p;
    if peek p = Some ']' then (
      advance p;
      `List [])
    else
      let rec loop acc =
        let v = parse_value p in
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

  and parse_object p =
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
        let value = parse_value p in
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

  let from_string s =
    let p = { input = s; pos = 0 } in
    try
      let v = parse_value p in
      skip_ws p;
      if p.pos < String.length s then
        raise (Parse_error "trailing characters after JSON");
      v
    with Parse_error msg ->
      raise (Parse_error (strf "at position %d: %s" p.pos msg))

  (* Accessors *)

  let to_assoc = function
    | `Assoc kv -> kv
    | _ -> raise (Parse_error "expected object")

  let to_string_val = function
    | `String s -> s
    | _ -> raise (Parse_error "expected string")

  let to_int_val = function
    | `Int i -> i
    | _ -> raise (Parse_error "expected integer")

  let to_list_val = function
    | `List l -> l
    | _ -> raise (Parse_error "expected array")

  let member key = function
    | `Assoc kv -> List.assoc key kv
    | _ -> raise (Parse_error "expected object")
end

(*---------------------------------------------------------------------------
  Safetensors format
  ---------------------------------------------------------------------------*)

(* Errors *)

type error =
  | Invalid_header of string
  | Invalid_header_deserialization of string
  | Header_too_large
  | Header_too_small
  | Invalid_header_length
  | Tensor_not_found of string
  | Tensor_invalid_info
  | Invalid_offset of string
  | Io_error of string
  | Invalid_tensor_view of string * int list * int
  | Metadata_incomplete_buffer
  | Validation_overflow
  | Misaligned_slice

let string_of_error = function
  | Invalid_header e -> "invalid UTF-8 in header: " ^ e
  | Invalid_header_deserialization e -> "invalid JSON in header: " ^ e
  | Header_too_large -> "header too large"
  | Header_too_small -> "header too small"
  | Invalid_header_length -> "invalid header length"
  | Tensor_not_found n -> strf "tensor '%s' not found" n
  | Tensor_invalid_info -> "invalid shape, dtype, or offset for tensor"
  | Invalid_offset n -> strf "invalid offset for tensor '%s'" n
  | Io_error e -> "I/O error: " ^ e
  | Invalid_tensor_view (dt, shape, n) ->
      let dims = List.map string_of_int shape |> String.concat ", " in
      strf "tensor of type %s and shape (%s) can't be created from %d bytes" dt
        dims n
  | Metadata_incomplete_buffer -> "incomplete metadata, file not fully covered"
  | Validation_overflow -> "overflow computing buffer size"
  | Misaligned_slice -> "slice does not end at a byte boundary"

(* Dtype *)

type dtype =
  | BOOL
  | F4
  | F6_E2M3
  | F6_E3M2
  | U8
  | I8
  | F8_E5M2
  | F8_E4M3
  | F8_E8M0
  | I16
  | U16
  | F16
  | BF16
  | I32
  | U32
  | F32
  | F64
  | I64
  | U64

let dtype_to_string = function
  | BOOL -> "BOOL"
  | F4 -> "F4"
  | F6_E2M3 -> "F6_E2M3"
  | F6_E3M2 -> "F6_E3M2"
  | U8 -> "U8"
  | I8 -> "I8"
  | F8_E5M2 -> "F8_E5M2"
  | F8_E4M3 -> "F8_E4M3"
  | F8_E8M0 -> "F8_E8M0"
  | I16 -> "I16"
  | U16 -> "U16"
  | F16 -> "F16"
  | BF16 -> "BF16"
  | I32 -> "I32"
  | U32 -> "U32"
  | F32 -> "F32"
  | F64 -> "F64"
  | I64 -> "I64"
  | U64 -> "U64"

let dtype_of_string = function
  | "BOOL" -> Some BOOL
  | "F4" -> Some F4
  | "F6_E2M3" -> Some F6_E2M3
  | "F6_E3M2" -> Some F6_E3M2
  | "U8" -> Some U8
  | "I8" -> Some I8
  | "F8_E5M2" -> Some F8_E5M2
  | "F8_E4M3" -> Some F8_E4M3
  | "F8_E8M0" -> Some F8_E8M0
  | "I16" -> Some I16
  | "U16" -> Some U16
  | "F16" -> Some F16
  | "BF16" -> Some BF16
  | "I32" -> Some I32
  | "U32" -> Some U32
  | "F32" -> Some F32
  | "F64" -> Some F64
  | "I64" -> Some I64
  | "U64" -> Some U64
  | _ -> None

let bitsize = function
  | F4 -> 4
  | F6_E3M2 | F6_E2M3 -> 6
  | BOOL | U8 | I8 | F8_E5M2 | F8_E4M3 | F8_E8M0 -> 8
  | I16 | U16 | F16 | BF16 -> 16
  | I32 | U32 | F32 -> 32
  | I64 | U64 | F64 -> 64

(* Alignment rank for serialization ordering (ascending alignment) *)
let dtype_rank = function
  | BOOL -> 0
  | F4 -> 1
  | F6_E2M3 -> 2
  | F6_E3M2 -> 3
  | U8 -> 4
  | I8 -> 5
  | F8_E5M2 -> 6
  | F8_E4M3 -> 7
  | F8_E8M0 -> 8
  | I16 -> 9
  | U16 -> 10
  | F16 -> 11
  | BF16 -> 12
  | I32 -> 13
  | U32 -> 14
  | F32 -> 15
  | F64 -> 16
  | I64 -> 17
  | U64 -> 18

(* Tensor model *)

type tensor_info = { dtype : dtype; shape : int list; data_offsets : int * int }

type metadata = {
  metadata_kv : (string * string) list option;
  tensors : tensor_info array;
  index_map : (string, int) Hashtbl.t;
}

(* UTF-8 validation *)

let is_valid_utf8 s =
  let n = String.length s in
  let i = ref 0 in
  let ok = ref true in
  while !ok && !i < n do
    let c = Char.code s.[!i] in
    if c land 0x80 = 0 then incr i
    else if c land 0xE0 = 0xC0 && !i + 1 < n then begin
      let c1 = Char.code s.[!i + 1] in
      if c1 land 0xC0 <> 0x80 || c < 0xC2 then ok := false else i := !i + 2
    end
    else if c land 0xF0 = 0xE0 && !i + 2 < n then begin
      let c1 = Char.code s.[!i + 1] in
      let c2 = Char.code s.[!i + 2] in
      if c1 land 0xC0 <> 0x80 || c2 land 0xC0 <> 0x80 then ok := false
      else if c = 0xE0 && c1 < 0xA0 then ok := false
      else if c = 0xED && c1 >= 0xA0 then ok := false
      else i := !i + 3
    end
    else if c land 0xF8 = 0xF0 && !i + 3 < n then begin
      let c1 = Char.code s.[!i + 1] in
      let c2 = Char.code s.[!i + 2] in
      let c3 = Char.code s.[!i + 3] in
      if c1 land 0xC0 <> 0x80 || c2 land 0xC0 <> 0x80 || c3 land 0xC0 <> 0x80
      then ok := false
      else if c = 0xF0 && c1 < 0x90 then ok := false
      else if c = 0xF4 && c1 >= 0x90 then ok := false
      else if c > 0xF4 then ok := false
      else i := !i + 4
    end
    else ok := false
  done;
  !ok

(* Arithmetic with overflow checking *)

let int64_mul_checked a b =
  if a = 0L || b = 0L then Ok 0L
  else if a > Int64.div Int64.max_int b then Error ()
  else Ok (Int64.mul a b)

(* Validation *)

exception Validate_error of error

let validate m =
  let start = ref 0 in
  let buffer_end = ref 0 in
  try
    Array.iteri
      (fun i info ->
        let s, e = info.data_offsets in
        if s <> !start || e < s then begin
          let name = ref "unknown" in
          Hashtbl.iter (fun k idx -> if idx = i then name := k) m.index_map;
          raise_notrace (Validate_error (Invalid_offset !name))
        end;
        start := e;
        let ne =
          List.fold_left
            (fun acc d ->
              if d < 0 then raise_notrace (Validate_error Validation_overflow);
              match int64_mul_checked acc (Int64.of_int d) with
              | Ok v -> v
              | Error () -> raise_notrace (Validate_error Validation_overflow))
            1L info.shape
        in
        let nbits =
          match int64_mul_checked ne (Int64.of_int (bitsize info.dtype)) with
          | Ok v -> v
          | Error () -> raise_notrace (Validate_error Validation_overflow)
        in
        if Int64.rem nbits 8L <> 0L then
          raise_notrace (Validate_error Misaligned_slice);
        let size = Int64.to_int (Int64.div nbits 8L) in
        if e - s <> size then raise_notrace (Validate_error Tensor_invalid_info);
        buffer_end := e)
      m.tensors;
    Ok !buffer_end
  with Validate_error e -> Error e

(* Little-endian I/O *)

let read_u64_le s off =
  let get i = Int64.of_int (Char.code s.[off + i]) in
  Int64.(
    logor (get 0)
      (logor
         (shift_left (get 1) 8)
         (logor
            (shift_left (get 2) 16)
            (logor
               (shift_left (get 3) 24)
               (logor
                  (shift_left (get 4) 32)
                  (logor
                     (shift_left (get 5) 40)
                     (logor (shift_left (get 6) 48) (shift_left (get 7) 56))))))))

let write_u64_le b off v =
  for i = 0 to 7 do
    Bytes.set b (off + i)
      (Char.chr
         (Int64.to_int (Int64.logand (Int64.shift_right v (8 * i)) 0xFFL)))
  done

(* JSON ↔ metadata *)

let metadata_to_json m =
  let names = Array.make (Array.length m.tensors) "" in
  Hashtbl.iter (fun name idx -> names.(idx) <- name) m.index_map;
  let base =
    Array.to_list
      (Array.mapi
         (fun i ti ->
           let shape = `List (List.map (fun d -> `Int d) ti.shape) in
           let s, e = ti.data_offsets in
           let offs = `List [ `Int s; `Int e ] in
           ( names.(i),
             `Assoc
               [
                 ("dtype", `String (dtype_to_string ti.dtype));
                 ("shape", shape);
                 ("data_offsets", offs);
               ] ))
         m.tensors)
  in
  let kv =
    match m.metadata_kv with
    | None -> base
    | Some md ->
        let obj = `Assoc (List.map (fun (k, v) -> (k, `String v)) md) in
        ("__metadata__", obj) :: base
  in
  Json.to_string (`Assoc kv)

let parse_tensor_info (name, j) =
  try
    let dt_str = j |> Json.member "dtype" |> Json.to_string_val in
    let dt =
      match dtype_of_string dt_str with
      | Some d -> d
      | None -> failwith "bad dtype"
    in
    let shape =
      j |> Json.member "shape" |> Json.to_list_val |> List.map Json.to_int_val
    in
    let s, e =
      match
        j |> Json.member "data_offsets" |> Json.to_list_val
        |> List.map Json.to_int_val
      with
      | [ s; e ] -> (s, e)
      | _ -> failwith "bad offsets"
    in
    Ok (name, { dtype = dt; shape; data_offsets = (s, e) })
  with e -> Error (Printexc.to_string e)

let json_to_metadata j : (metadata, error) result =
  let parse () =
    let obj = Json.to_assoc j in
    let md =
      match List.assoc_opt "__metadata__" obj with
      | None -> Ok None
      | Some jmd ->
          let kv = Json.to_assoc jmd in
          Ok (Some (List.map (fun (k, v) -> (k, Json.to_string_val v)) kv))
    in
    let kv_no_md = List.filter (fun (k, _) -> k <> "__metadata__") obj in
    let rec parse_tensors acc = function
      | [] -> Ok (List.rev acc)
      | entry :: rest ->
          let* ti = parse_tensor_info entry in
          parse_tensors (ti :: acc) rest
    in
    let* md = md in
    let* ts = parse_tensors [] kv_no_md in
    let ts =
      List.sort (fun (_, a) (_, b) -> compare a.data_offsets b.data_offsets) ts
    in
    let index_map = Hashtbl.create (List.length ts) in
    let tensors =
      Array.of_list
        (List.mapi
           (fun i (name, t) ->
             Hashtbl.add index_map name i;
             t)
           ts)
    in
    Ok { metadata_kv = md; tensors; index_map }
  in
  match parse () with
  | Error e -> Error (Invalid_header_deserialization e)
  | Ok m ->
      let* _ = validate m in
      Ok m

(* Tensor views *)

type tensor_view = {
  dtype : dtype;
  shape : int list;
  data : string;
  offset : int;
  length : int;
}

let tensor_view_new ~dtype ~shape ~data =
  let nbits =
    let ne =
      List.fold_left
        (fun acc d ->
          match acc with
          | Error _ as e -> e
          | Ok a -> (
              if d < 0 then Error Validation_overflow
              else
                match int64_mul_checked a (Int64.of_int d) with
                | Ok v -> Ok v
                | Error () -> Error Validation_overflow))
        (Ok 1L) shape
    in
    match ne with
    | Error e -> Error e
    | Ok ne -> (
        match int64_mul_checked ne (Int64.of_int (bitsize dtype)) with
        | Ok v -> Ok v
        | Error () -> Error Validation_overflow)
  in
  match nbits with
  | Error e -> Error e
  | Ok nb ->
      if Int64.rem nb 8L <> 0L then Error Misaligned_slice
      else
        let size = Int64.to_int (Int64.div nb 8L) in
        if String.length data <> size then
          Error
            (Invalid_tensor_view
               (dtype_to_string dtype, shape, String.length data))
        else Ok { dtype; shape; data; offset = 0; length = size }

(* Container *)

type t = { metadata : metadata; data : string }

let max_header_size = 100_000_000
let header_len_bytes = 8

let next_multiple_of x k =
  if k <= 0 || x mod k = 0 then x else x + (k - (x mod k))

(* Deserialization *)

let deserialize buffer =
  let len = String.length buffer in
  if len < header_len_bytes then Error Header_too_small
  else
    let n = read_u64_le buffer 0 in
    if n > Int64.of_int max_header_size then Error Header_too_large
    else
      let n_int = Int64.to_int n in
      let stop =
        match Int64.to_int (Int64.add n (Int64.of_int header_len_bytes)) with
        | exception _ -> -1
        | v -> v
      in
      if stop < 0 || stop > len then Error Invalid_header_length
      else
        let header = String.sub buffer header_len_bytes n_int in
        if not (is_valid_utf8 header) then Error (Invalid_header "bad utf8")
        else
          try
            let j = Json.from_string header in
            let* m = json_to_metadata j in
            let* buffer_end = validate m in
            if buffer_end + header_len_bytes + n_int <> len then
              Error Metadata_incomplete_buffer
            else
              let data =
                String.sub buffer (header_len_bytes + n_int)
                  (len - (header_len_bytes + n_int))
              in
              Ok { metadata = m; data }
          with Json.Parse_error e -> Error (Invalid_header_deserialization e)

let tensors st =
  let names = ref [] in
  Hashtbl.iter (fun name _ -> names := name :: !names) st.metadata.index_map;
  List.map
    (fun name ->
      let idx = Hashtbl.find st.metadata.index_map name in
      let info = st.metadata.tensors.(idx) in
      let s, e = info.data_offsets in
      ( name,
        {
          dtype = info.dtype;
          shape = info.shape;
          data = st.data;
          offset = s;
          length = e - s;
        } ))
    !names

(* Serialization *)

let prepare data data_info =
  let sorted =
    List.sort
      (fun (ln, lt) (rn, rt) ->
        let cmp = compare (dtype_rank rt.dtype) (dtype_rank lt.dtype) in
        if cmp <> 0 then cmp else compare ln rn)
      data
  in
  let offset = ref 0 in
  let hmetadata = ref [] in
  let tensors = ref [] in
  List.iter
    (fun (name, t) ->
      let n = t.length in
      let ti =
        {
          dtype = t.dtype;
          shape = t.shape;
          data_offsets = (!offset, !offset + n);
        }
      in
      offset := !offset + n;
      hmetadata := (name, ti) :: !hmetadata;
      tensors := t :: !tensors)
    sorted;
  let hmetadata = List.rev !hmetadata in
  let index_map = Hashtbl.create (List.length hmetadata) in
  let tensors_arr =
    Array.of_list
      (List.mapi
         (fun i (name, ti) ->
           Hashtbl.add index_map name i;
           ti)
         hmetadata)
  in
  let meta = { metadata_kv = data_info; tensors = tensors_arr; index_map } in
  let* _ = validate meta in
  let json = metadata_to_json meta in
  let n_aligned = next_multiple_of (String.length json) header_len_bytes in
  let header_bytes =
    if n_aligned = String.length json then json
    else
      let b = Bytes.make n_aligned ' ' in
      Bytes.blit_string json 0 b 0 (String.length json);
      Bytes.to_string b
  in
  Ok (n_aligned, header_bytes, !offset, List.rev !tensors)

let serialize_to_file data data_info filename =
  let* n_aligned, header_bytes, total_data_len, tensors =
    prepare data data_info
  in
  let total = header_len_bytes + n_aligned + total_data_len in
  let b = Bytes.create total in
  write_u64_le b 0 (Int64.of_int n_aligned);
  Bytes.blit_string header_bytes 0 b header_len_bytes n_aligned;
  let pos = ref (header_len_bytes + n_aligned) in
  List.iter
    (fun (tv : tensor_view) ->
      Bytes.blit_string tv.data tv.offset b !pos tv.length;
      pos := !pos + tv.length)
    tensors;
  try
    let oc = open_out_bin filename in
    Fun.protect ~finally:(fun () -> close_out oc) (fun () -> output_bytes oc b);
    Ok ()
  with e -> Error (Io_error (Printexc.to_string e))
