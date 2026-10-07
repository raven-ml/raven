(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The layout of a binary table (FITS 4.0 §7.3): rows of fields given by
   TFORMn, then a heap of the arrays that P and Q descriptors point to. *)

open Err

let strf = Printf.sprintf

type elt = L | X | B | I | J | K | A | E | D | C | M

(* Where a field's elements are: in the row, or in the heap through a P
   (32-bit) or Q (64-bit) descriptor. *)
type heap = Row | P | Q
type form = { repeat : int; elt : elt; heap : heap }

let elt_of_char = function
  | 'L' -> Some L
  | 'X' -> Some X
  | 'B' -> Some B
  | 'I' -> Some I
  | 'J' -> Some J
  | 'K' -> Some K
  | 'A' -> Some A
  | 'E' -> Some E
  | 'D' -> Some D
  | 'C' -> Some C
  | 'M' -> Some M
  | _ -> None

let char_of_elt = function
  | L -> 'L'
  | X -> 'X'
  | B -> 'B'
  | I -> 'I'
  | J -> 'J'
  | K -> 'K'
  | A -> 'A'
  | E -> 'E'
  | D -> 'D'
  | C -> 'C'
  | M -> 'M'

(* Bytes per element; X holds 8 elements per byte. *)
let size = function
  | L | B | A | X -> 1
  | I -> 2
  | J | E -> 4
  | K | D | C -> 8
  | M -> 16

(* [tform s] reads [rTa]: a repeat (1 when absent), a type code, and for P
   and Q the code of the heap's elements and an optional (max), which sizes
   nothing. *)
let tform s =
  let s = String.trim s in
  let n = String.length s in
  let i = ref 0 in
  while !i < n && s.[!i] >= '0' && s.[!i] <= '9' do
    incr i
  done;
  let repeat =
    if !i = 0 then Some 1 else int_of_string_opt (String.sub s 0 !i)
  in
  match repeat with
  | None -> None
  | Some repeat -> (
      if !i >= n then None
      else
        match s.[!i] with
        | ('P' | 'Q') as h ->
            if !i + 1 >= n then None
            else
              Option.map
                (fun elt -> { repeat; elt; heap = (if h = 'P' then P else Q) })
                (elt_of_char s.[!i + 1])
        | c ->
            Option.map (fun elt -> { repeat; elt; heap = Row }) (elt_of_char c))

(* The bytes a field takes in a row. *)
let width f =
  match f.heap with
  | P -> if f.repeat = 0 then 0 else 8
  | Q -> if f.repeat = 0 then 0 else 16
  | Row -> (
      match f.elt with
      | X -> (f.repeat + 7) / 8
      | e -> (
          match Err.mul f.repeat (size e) with
          | Some w -> w
          | None -> fail "a field's width overflows"))

type column = {
  number : int;  (** from 1 *)
  name : string;  (** TTYPEn, [""] when absent *)
  form : form;
  start : int;  (** byte offset in the row *)
}

type t = {
  place : Err.place;
  store : Hdu.store;
  rows : int;
  row_bytes : int;
  columns : column array;
  heap_start : int;  (** THEAP, from the data unit's start *)
  heap_size : int;
}

let column_place t c =
  Err.sub t.place
    (if c.name = "" then strf "column %d" c.number
     else strf "column %d (%s)" c.number c.name)

let layout h (store : Hdu.store) =
  let place = Header.place h in
  let get v k = Hdu.get_struct h v k in
  let row_bytes = get Value.int "NAXIS1" and rows = get Value.int "NAXIS2" in
  if row_bytes < 0 || rows < 0 then fail_at place "NAXIS1 or NAXIS2 is negative";
  let n = get Value.int "TFIELDS" in
  if n < 0 || n > 999 then fail_at place "TFIELDS %d is outside 0-999" n;
  let start = ref 0 in
  let columns =
    Array.init n (fun i ->
        let k = strf "TFORM%d" (i + 1) in
        let text = get Value.string k in
        let form =
          match tform text with
          | Some f -> f
          | None ->
              Hdu.card_fail h k (strf "%S is not a binary table format" text)
        in
        let name =
          match Header.find_struct Value.string (strf "TTYPE%d" (i + 1)) h with
          | Ok (Some s) -> s
          | Ok None -> ""
          | Error e -> fail "%s" e
        in
        let c = { number = i + 1; name; form; start = !start } in
        (match Err.add !start (width form) with
        | Some s -> start := s
        | None -> fail_at place "the row's width overflows");
        c)
  in
  if !start <> row_bytes then
    fail_at place "NAXIS1 is %d, the fields' widths sum to %d" row_bytes !start;
  let table =
    match Err.mul rows row_bytes with
    | Some b -> b
    | None -> fail_at place "the table's size overflows"
  in
  let heap_start =
    match Header.find_struct Value.int "THEAP" h with
    | Ok (Some v) -> v
    | Ok None -> table
    | Error e -> fail "%s" e
  in
  if heap_start < table || heap_start > store.size then
    Hdu.card_fail h "THEAP"
      (strf "%d is outside the rows' end %d and the data unit's end %d"
         heap_start table store.size);
  {
    place;
    store;
    rows;
    row_bytes;
    columns;
    heap_start;
    heap_size = store.size - heap_start;
  }

let find t name =
  match List.filter (fun c -> c.name = name) (Array.to_list t.columns) with
  | [ c ] -> Some c
  | _ -> None

(* [read_rows t a b] is a host copy of rows [a] to [b - 1]. *)
let read_rows t a b =
  let n = (b - a) * t.row_bytes in
  let host = Hdu.host_bytes n in
  Hdu.copy_in t.store.buffer
    ~offset:(t.store.offset + (a * t.row_bytes))
    host ~at:0 n;
  Hdu.bigbytes host

let be_int (a : Checksum.bigbytes) off n =
  let v = ref 0 in
  for i = 0 to n - 1 do
    v := (!v lsl 8) lor Bigarray.Array1.get a (off + i)
  done;
  !v

let be_int32 a off =
  let v = be_int a off 4 in
  if v land 0x80000000 <> 0 then v - (1 lsl 32) else v

let be_int64 (a : Checksum.bigbytes) off =
  let v = ref 0L in
  for i = 0 to 7 do
    v :=
      Int64.logor (Int64.shift_left !v 8)
        (Int64.of_int (Bigarray.Array1.get a (off + i)))
  done;
  !v

(* [descriptor t c rows r] is the element count and heap offset of column
   [c] in row [r] of the host rows [rows] starting at row 0 of [rows]. A
   descriptor of no elements names no bytes; any other one's bytes lie in
   the heap. *)
let descriptor t c rows r =
  let off = (r * t.row_bytes) + c.start in
  let count, offset =
    match c.form.heap with
    | P -> (be_int32 rows off, be_int32 rows (off + 4))
    | Q ->
        let n = be_int64 rows off and o = be_int64 rows (off + 8) in
        let fits v =
          Int64.compare v 0L >= 0 && Int64.compare v (Int64.of_int max_int) <= 0
        in
        if not (fits n && fits o) then (-1, -1)
        else (Int64.to_int n, Int64.to_int o)
    | Row -> invalid_arg "Bintable.descriptor: not a heap column"
  in
  if count < 0 || offset < 0 then
    fail_at (column_place t c)
      "a descriptor's count %d or offset %d is negative" count offset;
  if count = 0 then (0, 0)
  else
    let bytes =
      match c.form.elt with
      | X -> (count + 7) / 8
      | e -> (
          match Err.mul count (size e) with Some b -> b | None -> max_int)
    in
    if bytes > t.heap_size || offset > t.heap_size - bytes then
      fail_at (column_place t c)
        "the descriptor's %d elements at heap byte %d end past the heap's %d \
         bytes"
        count offset t.heap_size;
    (count, offset)

(* [heap t off n] is a host copy of [n] bytes of the heap from [off]. *)
let heap t off n =
  let host = Hdu.host_bytes n in
  Hdu.copy_in t.store.buffer
    ~offset:(t.store.offset + t.heap_start + off)
    host ~at:0 n;
  Hdu.bigbytes host

let float_at (a : Checksum.bigbytes) off = function
  | E -> Int32.float_of_bits (Int32.of_int (be_int a off 4))
  | D -> Int64.float_of_bits (be_int64 a off)
  | _ -> invalid_arg "Bintable.float_at"
