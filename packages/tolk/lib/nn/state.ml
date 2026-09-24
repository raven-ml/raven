(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_frontend
module D = Tolk_uop.Dtype

let bad_header msg = invalid_arg ("State.safe_load: " ^ msg)

(* Safetensors headers need lossless number tokens: dimensions and offsets
   must not pass through a floating-point JSON representation. *)
module Json = struct
  type t = Obj of (string * t) list | Arr of t list | Str of string
         | Number of string | Bool of bool | Null

  type parser = { input : string; mutable pos : int }

  let error p msg =
    bad_header (Printf.sprintf "%s at byte %d" msg p.pos)

  let peek p =
    if p.pos < String.length p.input then Some p.input.[p.pos] else None

  let skip_ws p =
    while
      p.pos < String.length p.input
      && match p.input.[p.pos] with ' ' | '\t' | '\n' | '\r' -> true | _ -> false
    do p.pos <- p.pos + 1 done

  let take p c =
    match peek p with
    | Some ch when ch = c -> p.pos <- p.pos + 1
    | _ -> error p (Printf.sprintf "expected %C" c)

  let expect p c = skip_ws p; take p c

  let hex4 p =
    let value = ref 0 in
    for i = 0 to 3 do
      let digit = match peek p with
        | Some ('0' .. '9' as c) -> Char.code c - Char.code '0'
        | Some ('a' .. 'f' as c) -> Char.code c - Char.code 'a' + 10
        | Some ('A' .. 'F' as c) -> Char.code c - Char.code 'A' + 10
        | _ -> error p (Printf.sprintf "invalid Unicode escape digit %d" (i + 1))
      in
      value := (!value lsl 4) lor digit;
      p.pos <- p.pos + 1
    done;
    !value

  let unicode p =
    let first = hex4 p in
    if first >= 0xd800 && first <= 0xdbff then begin
      take p '\\'; take p 'u';
      let second = hex4 p in
      if second < 0xdc00 || second > 0xdfff then error p "invalid surrogate pair";
      Uchar.of_int (0x10000 + ((first - 0xd800) lsl 10) + second - 0xdc00)
    end else if first >= 0xdc00 && first <= 0xdfff then
      error p "unpaired low surrogate"
    else Uchar.of_int first

  let string_ p =
    expect p '"';
    let buf = Buffer.create 16 in
    let rec loop () =
      match peek p with
      | None -> error p "unterminated string"
      | Some '"' -> p.pos <- p.pos + 1; Buffer.contents buf
      | Some '\\' ->
          p.pos <- p.pos + 1;
          (match peek p with
           | None -> error p "unterminated escape"
           | Some c ->
               p.pos <- p.pos + 1;
               match c with
               | '"' | '\\' | '/' -> Buffer.add_char buf c
               | 'b' -> Buffer.add_char buf '\b'
               | 'f' -> Buffer.add_char buf '\012'
               | 'n' -> Buffer.add_char buf '\n'
               | 'r' -> Buffer.add_char buf '\r'
               | 't' -> Buffer.add_char buf '\t'
               | 'u' -> Buffer.add_utf_8_uchar buf (unicode p)
               | _ -> error p "invalid escape");
          loop ()
      | Some c ->
          if Char.code c < 0x20 then error p "unescaped control character";
          Buffer.add_char buf c;
          p.pos <- p.pos + 1;
          loop ()
    in
    loop ()

  let digits p =
    let start = p.pos in
    while match peek p with
      | Some ('0' .. '9') -> p.pos <- p.pos + 1; true
      | _ -> false
    do () done;
    if start = p.pos then error p "expected a digit"

  let number p =
    let start = p.pos in
    (match peek p with Some '-' -> p.pos <- p.pos + 1 | _ -> ());
    (match peek p with
     | Some '0' -> p.pos <- p.pos + 1
     | _ -> digits p);
    (match peek p with
     | Some '.' -> p.pos <- p.pos + 1; digits p
     | _ -> ());
    (match peek p with
     | Some ('e' | 'E') ->
         p.pos <- p.pos + 1;
         (match peek p with Some ('+' | '-') -> p.pos <- p.pos + 1 | _ -> ());
         digits p
     | _ -> ());
    Number (String.sub p.input start (p.pos - start))

  let literal p text value =
    String.iter (take p) text;
    value

  let rec value p =
    skip_ws p;
    match peek p with
    | Some '{' -> obj p
    | Some '[' -> arr p
    | Some '"' -> Str (string_ p)
    | Some ('-' | '0' .. '9') -> number p
    | Some 't' -> literal p "true" (Bool true)
    | Some 'f' -> literal p "false" (Bool false)
    | Some 'n' -> literal p "null" Null
    | _ -> error p "expected a JSON value"

  and obj p =
    expect p '{';
    skip_ws p;
    if peek p = Some '}' then (p.pos <- p.pos + 1; Obj [])
    else
      let rec fields acc =
        let key = string_ p in
        expect p ':';
        let v = value p in
        skip_ws p;
        match peek p with
        | Some ',' -> p.pos <- p.pos + 1; fields ((key, v) :: acc)
        | Some '}' -> p.pos <- p.pos + 1; Obj (List.rev ((key, v) :: acc))
        | _ -> error p "expected ',' or '}'"
      in
      fields []

  and arr p =
    expect p '[';
    skip_ws p;
    if peek p = Some ']' then (p.pos <- p.pos + 1; Arr [])
    else
      let rec items acc =
        let v = value p in
        skip_ws p;
        match peek p with
        | Some ',' -> p.pos <- p.pos + 1; items (v :: acc)
        | Some ']' -> p.pos <- p.pos + 1; Arr (List.rev (v :: acc))
        | _ -> error p "expected ',' or ']'"
      in
      items []

  let parse input =
    if not (String.is_valid_utf_8 input) then bad_header "invalid UTF-8";
    let p = { input; pos = 0 } in
    let result = value p in
    skip_ws p;
    if p.pos <> String.length input then error p "trailing data";
    result
end

let object_fields = function
  | Json.Obj members ->
      let seen = Hashtbl.create (List.length members) in
      List.map (fun (name, value) ->
          if Hashtbl.mem seen name then bad_header ("duplicate field " ^ name);
          Hashtbl.add seen name ();
          (name, value)) members
  | _ -> bad_header "expected an object"

let json_string = function
  | Json.Str s -> s
  | _ -> bad_header "expected a string"

let json_int = function
  | Json.Number token ->
      (match int_of_string_opt token with
       | Some n when n >= 0 -> n
       | _ -> bad_header "expected a representable non-negative integer")
  | _ -> bad_header "expected an integer"

let dtype_of_string = function
  | "BOOL" -> D.bool
  | "I8" -> D.int8
  | "U8" -> D.uint8
  | "I16" -> D.int16
  | "U16" -> D.uint16
  | "I32" -> D.int32
  | "U32" -> D.uint32
  | "I64" -> D.int64
  | "U64" -> D.uint64
  | "F8_E4M3" -> D.fp8e4m3
  | "F8_E5M2" -> D.fp8e5m2
  | "F16" -> D.float16
  | "BF16" -> D.bfloat16
  | "F32" -> D.float32
  | "F64" -> D.float64
  | s -> invalid_arg (Printf.sprintf "State.safe_load: unknown dtype %S" s)

let safe_load fn =
  In_channel.with_open_bin fn (fun ic ->
      let file_size = In_channel.length ic in
      let head = Bytes.create 8 in
      (match In_channel.really_input ic head 0 8 with
      | Some () -> ()
      | None -> bad_header "truncated file");
      let header_len = Bytes.get_int64_le head 0 in
      if header_len < 0L || header_len > Int64.sub file_size 8L
         || header_len > Int64.of_int Sys.max_string_length
      then bad_header "invalid header length";
      let header =
        match In_channel.really_input_string ic (Int64.to_int header_len) with
        | Some s -> s
        | None -> bad_header "truncated header"
      in
      let entries =
        object_fields (Json.parse header)
      in
      let data_start = Int64.add 8L header_len in
      let data_size = Int64.sub file_size data_start in
      let descriptors = List.filter_map (fun (name, entry) ->
          let fields = object_fields entry in
          if name = "__metadata__" then begin
            List.iter (fun (_, v) -> ignore (json_string v)) fields;
            None
          end else begin
            let field key = match List.assoc_opt key fields with
              | Some v -> v | None -> bad_header ("missing field " ^ key)
            in
            let ints key = match field key with
              | Json.Arr vs -> List.map json_int vs
              | _ -> bad_header ("expected array " ^ key)
            in
            let dtype = dtype_of_string (json_string (field "dtype")) in
            let shape = ints "shape" in
            let off0, off1 = match ints "data_offsets" with
              | [ a; b ] when a <= b && Int64.of_int b <= data_size -> (a, b)
              | _ -> bad_header ("invalid offsets for " ^ name)
            in
            let nbytes =
              if List.mem 0 shape then 0
              else List.fold_left (fun size dim ->
                  if size > Sys.max_string_length / dim then
                    bad_header ("tensor too large: " ^ name);
                  size * dim) (D.itemsize dtype) shape
            in
            if off1 - off0 <> nbytes then bad_header ("shape/size mismatch for " ^ name);
            Some (name, dtype, shape, off0, off1)
          end) entries
      in
      let sorted = List.sort (fun (_, _, _, a, b) (_, _, _, c, d) ->
          let order = Int.compare a c in if order = 0 then Int.compare b d else order) descriptors in
      let end_offset = List.fold_left (fun previous (_, _, _, start, stop) ->
          if previous <> start then bad_header "overlapping or non-contiguous tensor data";
          stop) 0 sorted in
      if Int64.of_int end_offset <> data_size then bad_header "unclaimed tensor data";
      List.map (fun (name, dtype, shape, off0, off1) ->
          let data = Bytes.create (off1 - off0) in
          In_channel.seek ic (Int64.add data_start (Int64.of_int off0));
          (match In_channel.really_input ic data 0 (Bytes.length data) with
          | Some () -> ()
          | None -> bad_header "truncated tensor data");
          (name, Run.of_bytes ~dtype ~shape data)) descriptors)

let load_state_dict ?(strict = true) ?(realize = true) model state_dict =
  let loaded =
    List.filter_map
      (fun (k, v) ->
        match List.assoc_opt k state_dict with
        | None ->
            if strict then
              invalid_arg
                (Printf.sprintf "State.load_state_dict: missing key %S" k)
            else None
        | Some s ->
            let sv = Tensor.shape v and ss = Tensor.shape s in
            let s =
              if sv = ss then s
              else if (sv = [] && ss = [ 1 ]) || (sv = [ 1 ] && ss = []) then
                Movement.reshape s sv
              else
                invalid_arg
                  (Printf.sprintf
                     "State.load_state_dict: shape mismatch for %S" k)
            in
            Tensor.set_uop v (Tensor.uop s);
            Some v)
      model
  in
  if realize then Run.realize_many loaded
