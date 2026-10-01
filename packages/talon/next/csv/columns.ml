(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next
module A1 = Bigarray.Array1

type t =
  | Fixed of { valid : Nx.bool_t option; values : Nx.packed }
  | Varsize of {
      valid : Nx.bool_t option;
      offsets : Nx.int64_t;
      data : Nx.uint8_t;
    }

exception Invalid of { row : int; reason : string }

let reads (Type.Any t) =
  match t with
  | Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64
  | Float16 | Float32 | Float64 | Decimal _ | String | Binary | Categorical _
  | Date | Datetime _ ->
      true
  | Clock _ | Duration _ | List _ | Record _ | Tensor _ | Ext _ -> false

type reader = {
  ty : Type.any;
  quote : char;
  nulls : string list;
  codes : (string, int) Hashtbl.t;
}

let reader ~quote ~nulls ty =
  let codes = Hashtbl.create 16 in
  (match ty with
  | Type.Any (Categorical d) ->
      Iarray.iteri (fun i s -> Hashtbl.add codes s i) d
  | _ -> ());
  { ty; quote; nulls; codes }

let rec same_from b pos s i =
  i = String.length s
  || Bytes.unsafe_get b (pos + i) = String.unsafe_get s i
     && same_from b pos s (i + 1)

let rec is_token b pos len = function
  | [] -> false
  | t :: ts ->
      (String.length t = len && same_from b pos t 0) || is_token b pos len ts

let is_null nulls s r j =
  (not (Scan.quoted s r j))
  &&
  let len = Scan.len s r j in
  len = 0 || is_token (Scan.bytes s) (Scan.pos s r j) len nulls

let tensor a = Nx.of_bigarray (Bigarray.genarray_of_array1 a)

(* [null valid rows r] marks row [r] null, making the validity bytes of a column
   of [rows] rows at its first null. *)
let null valid rows r =
  match !valid with
  | Some v -> A1.unsafe_set v r 0
  | None ->
      let v = A1.create Bigarray.int8_unsigned Bigarray.c_layout rows in
      A1.fill v 1;
      A1.unsafe_set v r 0;
      valid := Some v

let mask valid = Option.map (fun v -> Nx.cast Nx.bool (tensor v)) !valid

let fixed c s j ~first ~rows kind zero parse =
  let a = A1.create kind Bigarray.c_layout rows in
  A1.fill a zero;
  let valid = ref None and r = ref 0 in
  (try
     while !r < rows do
       let row = first + !r in
       if is_null c.nulls s row j then null valid rows !r
       else parse s row j a !r;
       incr r
     done
   with Text.Invalid reason -> raise (Invalid { row = first + !r; reason }));
  (tensor a, mask valid)

let varsize c s j ~first ~rows ~utf_8 =
  let b = Scan.bytes s and q = c.quote in
  let total = ref 0 in
  for r = first to first + rows - 1 do
    total := !total + Scan.len s r j
  done;
  let data = A1.create Bigarray.int8_unsigned Bigarray.c_layout !total in
  let offsets = A1.create Bigarray.int64 Bigarray.c_layout (rows + 1) in
  A1.unsafe_set offsets 0 0L;
  let valid = ref None and o = ref 0 and r = ref 0 in
  (try
     while !r < rows do
       let row = first + !r in
       if is_null c.nulls s row j then null valid rows !r
       else begin
         let pos = Scan.pos s row j and len = Scan.len s row j in
         let quoted = Scan.quoted s row j in
         if utf_8 then Text.utf_8 b pos len;
         let k = ref pos in
         while !k < pos + len do
           let ch = Bytes.unsafe_get b !k in
           A1.unsafe_set data !o (Char.code ch);
           incr o;
           k := if quoted && ch = q then !k + 2 else !k + 1
         done
       end;
       A1.unsafe_set offsets (!r + 1) (Int64.of_int !o);
       incr r
     done
   with Text.Invalid reason -> raise (Invalid { row = first + !r; reason }));
  Varsize
    {
      valid = mask valid;
      offsets = tensor offsets;
      data = tensor (A1.sub data 0 !o);
    }

let code c s row j =
  match Hashtbl.find c.codes (Scan.text s row j) with
  | k -> k
  | exception Not_found -> raise (Text.Invalid "not in the dictionary")

let read c s j ~first ~rows =
  let text f s row j = f (Scan.bytes s) (Scan.pos s row j) (Scan.len s row j) in
  let ints f = fixed c s j ~first ~rows Bigarray.int64 0L (text f) in
  let floats f = fixed c s j ~first ~rows Bigarray.float64 0. (text f) in
  let of_field f =
    fixed c s j ~first ~rows Bigarray.int64 0L (fun s row j a i ->
        A1.unsafe_set a i (Int64.of_int (f s row j)))
  in
  let of_int f = of_field (text f) in
  let keep (values, valid) = Fixed { valid; values = Nx.P values } in
  let cast dt (values, valid) =
    Fixed { valid; values = Nx.P (Nx.cast dt values) }
  in
  let small ~min ~max = of_int (Text.int ~min ~max) in
  let (Type.Any ty) = c.ty in
  match ty with
  | Bool ->
      cast Nx.bool (of_int (fun b pos len -> Bool.to_int (Text.bool b pos len)))
  | Int8 -> cast Nx.int8 (small ~min:(-0x80) ~max:0x7F)
  | Int16 -> cast Nx.int16 (small ~min:(-0x8000) ~max:0x7FFF)
  | Int32 -> cast Nx.int32 (small ~min:(-0x8000_0000) ~max:0x7FFF_FFFF)
  | Uint8 -> cast Nx.uint8 (small ~min:0 ~max:0xFF)
  | Uint16 -> cast Nx.uint16 (small ~min:0 ~max:0xFFFF)
  | Uint32 -> cast Nx.uint32 (small ~min:0 ~max:0xFFFF_FFFF)
  | Int64 -> keep (ints Text.int64)
  | Uint64 ->
      let values, valid = ints Text.uint64 in
      Fixed { valid; values = Nx.P (Nx.bitcast Nx.uint64 values) }
  | Float16 -> cast Nx.float16 (floats Text.float16)
  | Float32 -> cast Nx.float32 (floats Text.float32)
  | Float64 -> keep (floats Text.float)
  | Decimal { precision; scale } ->
      keep (of_int (Text.decimal ~precision ~scale))
  | Date -> cast Nx.int32 (of_int Text.date)
  | Datetime { unit_; zone } ->
      keep (ints (Text.datetime unit_ ~zoned:(zone <> None)))
  | Categorical _ -> cast Nx.int32 (of_field (code c))
  | String -> varsize c s j ~first ~rows ~utf_8:true
  | Binary -> varsize c s j ~first ~rows ~utf_8:false
  | Clock _ | Duration _ | List _ | Record _ | Tensor _ | Ext _ ->
      invalid_arg "Columns.read: a type CSV does not read"
