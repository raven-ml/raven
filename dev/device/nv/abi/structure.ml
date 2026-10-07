(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Packet

type 'v hole = 'v Repr.hole = private {
  at : int;
  bits : int;
  value : 'v Packet.term;
}

type 'v t = 'v Repr.structure = private { bytes : string; holes : 'v hole list }

let rec eval value = function
  | Value v -> value v
  | Add (t, n) -> Int64.add (eval value t) n
  | Shift (t, n) ->
      let v = eval value t in
      if n < 0 || n > 63 then
        invalid_arg
          (Printf.sprintf "Structure.encode: shift by %d, expected 0 to 63" n);
      Int64.shift_right_logical v n

(* A hole's word: the narrowest of 1, 2, 4 and 8 bytes that holds [bits]. *)
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

let encode value s =
  let b = Bytes.of_string s.bytes in
  let fill h =
    let n = word_bytes h.bits and m = mask h.bits in
    let kept = Int64.logand (get b h.at n) (Int64.lognot m) in
    set b h.at n (Int64.logor kept (Int64.logand (eval value h.value) m))
  in
  List.iter fill s.holes;
  Bytes.unsafe_to_string b
