(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_frontend
module D = Tolk_uop.Dtype

(* Minimal JSON reader for safetensors headers: objects, arrays, strings, and
   non-negative integers are all the format uses. *)
module Json = struct
  type t =
    | Obj of (string * t) list
    | Arr of t list
    | Str of string
    | Int of int

  type parser = { input : string; mutable pos : int }

  let error p msg =
    invalid_arg (Printf.sprintf "safetensors header: %s at %d" msg p.pos)

  let peek p =
    if p.pos < String.length p.input then Some p.input.[p.pos] else None

  let skip_ws p =
    while
      p.pos < String.length p.input
      && match p.input.[p.pos] with ' ' | '\t' | '\n' | '\r' -> true | _ -> false
    do
      p.pos <- p.pos + 1
    done

  let expect p c =
    skip_ws p;
    match peek p with
    | Some ch when ch = c -> p.pos <- p.pos + 1
    | _ -> error p (Printf.sprintf "expected %C" c)

  let string_ p =
    expect p '"';
    let buf = Buffer.create 16 in
    let rec loop () =
      match peek p with
      | None -> error p "unterminated string"
      | Some '"' -> p.pos <- p.pos + 1
      | Some '\\' -> (
          p.pos <- p.pos + 1;
          match peek p with
          | Some (('"' | '\\' | '/') as c) ->
              Buffer.add_char buf c;
              p.pos <- p.pos + 1;
              loop ()
          | Some 'n' -> Buffer.add_char buf '\n'; p.pos <- p.pos + 1; loop ()
          | Some 't' -> Buffer.add_char buf '\t'; p.pos <- p.pos + 1; loop ()
          | Some 'u' ->
              (* Header names are ASCII in practice; keep the escape verbatim. *)
              Buffer.add_string buf "\\u";
              p.pos <- p.pos + 1;
              loop ()
          | _ -> error p "bad escape")
      | Some c ->
          Buffer.add_char buf c;
          p.pos <- p.pos + 1;
          loop ()
    in
    loop ();
    Buffer.contents buf

  let int_ p =
    skip_ws p;
    let start = p.pos in
    while
      p.pos < String.length p.input
      && match p.input.[p.pos] with '0' .. '9' -> true | _ -> false
    do
      p.pos <- p.pos + 1
    done;
    if p.pos = start then error p "expected integer";
    int_of_string (String.sub p.input start (p.pos - start))

  let rec value p =
    skip_ws p;
    match peek p with
    | Some '{' -> obj p
    | Some '[' -> arr p
    | Some '"' -> Str (string_ p)
    | Some '0' .. '9' -> Int (int_ p)
    | _ -> error p "expected value"

  and obj p =
    expect p '{';
    skip_ws p;
    if peek p = Some '}' then (p.pos <- p.pos + 1; Obj [])
    else
      let rec fields acc =
        let k = (skip_ws p; string_ p) in
        expect p ':';
        let v = value p in
        skip_ws p;
        match peek p with
        | Some ',' -> p.pos <- p.pos + 1; fields ((k, v) :: acc)
        | Some '}' -> p.pos <- p.pos + 1; Obj (List.rev ((k, v) :: acc))
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

  let parse s =
    let p = { input = s; pos = 0 } in
    let v = value p in
    skip_ws p;
    if p.pos <> String.length s then error p "trailing data";
    v
end

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
  | "F16" -> D.float16
  | "BF16" -> D.bfloat16
  | "F32" -> D.float32
  | "F64" -> D.float64
  | s -> invalid_arg (Printf.sprintf "State.safe_load: unknown dtype %S" s)

let safe_load fn =
  let ic = In_channel.open_bin fn in
  Fun.protect
    ~finally:(fun () -> In_channel.close ic)
    (fun () ->
      let head = Bytes.create 8 in
      (match In_channel.really_input ic head 0 8 with
      | Some () -> ()
      | None -> invalid_arg "State.safe_load: truncated file");
      let header_len = Int64.to_int (Bytes.get_int64_le head 0) in
      let header =
        match In_channel.really_input_string ic header_len with
        | Some s -> s
        | None -> invalid_arg "State.safe_load: truncated header"
      in
      let data_start = 8 + header_len in
      let entries =
        match Json.parse header with
        | Json.Obj kvs -> kvs
        | _ -> invalid_arg "State.safe_load: header is not an object"
      in
      List.filter_map
        (fun (name, entry) ->
          if String.equal name "__metadata__" then None
          else
            match entry with
            | Json.Obj fields ->
                let str k =
                  match List.assoc_opt k fields with
                  | Some (Json.Str s) -> s
                  | _ -> invalid_arg ("State.safe_load: bad field " ^ k)
                in
                let ints k =
                  match List.assoc_opt k fields with
                  | Some (Json.Arr vs) ->
                      List.map
                        (function
                          | Json.Int i -> i
                          | _ -> invalid_arg ("State.safe_load: bad field " ^ k))
                        vs
                  | _ -> invalid_arg ("State.safe_load: bad field " ^ k)
                in
                let dtype = dtype_of_string (str "dtype") in
                let shape = ints "shape" in
                let off0, off1 =
                  match ints "data_offsets" with
                  | [ a; b ] -> (a, b)
                  | _ -> invalid_arg "State.safe_load: bad data_offsets"
                in
                let nbytes = off1 - off0 in
                let data = Bytes.create nbytes in
                In_channel.seek ic (Int64.of_int (data_start + off0));
                (match In_channel.really_input ic data 0 nbytes with
                | Some () -> ()
                | None -> invalid_arg "State.safe_load: truncated tensor data");
                Some (name, Run.of_bytes ~dtype ~shape data)
            | _ -> invalid_arg "State.safe_load: bad header entry")
        entries)

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
            let s =
              if Tensor.shape v = Tensor.shape s then s
              else if Tensor.numel v = 1 && Tensor.numel s = 1 then
                Movement.reshape s (Tensor.shape v)
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
