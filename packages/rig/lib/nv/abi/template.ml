(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type operand = Int of int | Slot of int
type patch = { at : int; bits : int; slot : int; add : int64; shift : int }
type t = { bytes : string; patches : patch list }

let concat ts =
  let n = List.fold_left (fun n t -> n + String.length t.bytes) 0 ts in
  let b = Bytes.create n in
  let at = ref 0 and patches = ref [] in
  let add t =
    let off = !at in
    Bytes.blit_string t.bytes 0 b off (String.length t.bytes);
    List.iter
      (fun p -> patches := { p with at = p.at + off } :: !patches)
      t.patches;
    at := off + String.length t.bytes
  in
  List.iter add ts;
  { bytes = Bytes.unsafe_to_string b; patches = List.rev !patches }

(* A patch's word: the narrowest of 1, 2, 4 and 8 bytes that holds [bits]. *)
let word_bytes bits =
  if bits <= 8 then 1 else if bits <= 16 then 2 else if bits <= 32 then 4 else 8

let mask bits = if bits = 64 then -1L else Int64.(pred (shift_left 1L bits))

(* The little-endian word of [n] bytes at [at] of [b], and its update. Only the
   word's own bits are read back, so a sign-extended read is harmless. *)
let get b at = function
  | 1 -> Int64.of_int (Bytes.get_uint8 b at)
  | 2 -> Int64.of_int (Bytes.get_uint16_le b at)
  | 4 -> Int64.of_int32 (Bytes.get_int32_le b at)
  | _ -> Bytes.get_int64_le b at

let set b at n v =
  match n with
  | 1 -> Bytes.set_uint8 b at (Int64.to_int v land 0xff)
  | 2 -> Bytes.set_uint16_le b at (Int64.to_int v land 0xffff)
  | 4 -> Bytes.set_int32_le b at (Int64.to_int32 v)
  | _ -> Bytes.set_int64_le b at v

let fill value t =
  let b = Bytes.of_string t.bytes in
  let patch p =
    let n = word_bytes p.bits and m = mask p.bits in
    let v =
      Int64.shift_right_logical (Int64.add (value p.slot) p.add) p.shift
    in
    set b p.at n
      (Int64.logor
         (Int64.logand (get b p.at n) (Int64.lognot m))
         (Int64.logand v m))
  in
  List.iter patch t.patches;
  Bytes.unsafe_to_string b

(* Words *)

let unsigned fn n =
  if n < 0 then
    Printf.ksprintf invalid_arg "Template.%s: %d, expected an unsigned integer"
      fn n

let u32 s v =
  match v with
  | Int n ->
      unsigned "u32" n;
      let b = Bytes.create 4 in
      Bytes.set_int32_le b 0 (Int32.of_int (n lsr s));
      { bytes = Bytes.unsafe_to_string b; patches = [] }
  | Slot slot ->
      {
        bytes = "\000\000\000\000";
        patches = [ { at = 0; bits = 32; slot; add = 0L; shift = s } ];
      }

let u64 n v =
  match v with
  | Int x ->
      unsigned "u64" x;
      let b = Bytes.create 8 in
      Bytes.set_int64_le b 0 (Int64.add (Int64.of_int x) n);
      { bytes = Bytes.unsafe_to_string b; patches = [] }
  | Slot slot ->
      {
        bytes = String.make 8 '\000';
        patches = [ { at = 0; bits = 64; slot; add = n; shift = 0 } ];
      }
