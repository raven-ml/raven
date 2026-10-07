(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The ones' complement sum of big-endian 32-bit words (FITS 4.0 §4.4.2.7,
   Appendix J): a 32-bit sum whose carries wrap around to bit 0. *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

let mask = 0xFFFFFFFF

let add a b =
  let s = a + b in
  (s land mask) + (s lsr 32)

(* [bigbytes s b off len] is [s] plus the words of the [len] bytes of [b] from
   [off], [len] a multiple of 4. The 16-bit halves accumulate apart and fold
   every 2^15 words, before an [int] could overflow. *)
let bigbytes s (b : bigbytes) off len =
  let acc = ref s and i = ref off and stop = off + len in
  while !i < stop do
    let chunk = Int.min stop (!i + (4 lsl 15)) in
    let hi = ref 0 and lo = ref 0 in
    while !i < chunk do
      let u j = Bigarray.Array1.unsafe_get b j in
      hi := !hi + ((u !i lsl 8) lor u (!i + 1));
      lo := !lo + ((u (!i + 2) lsl 8) lor u (!i + 3));
      i := !i + 4
    done;
    acc := add !acc (add ((!hi lsl 16) land mask) (!hi lsr 16));
    acc := add !acc (add !lo 0)
  done;
  add !acc 0

let string s str =
  let n = String.length str in
  let acc = ref s in
  let i = ref 0 in
  while !i < n do
    let u j = Char.code (String.unsafe_get str j) in
    let w =
      (u !i lsl 24) lor (u (!i + 1) lsl 16) lor (u (!i + 2) lsl 8) lor u (!i + 3)
    in
    acc := add !acc w;
    i := !i + 4
  done;
  !acc

(* Appendix J: the 16 characters whose words sum to [x], each byte of [x] spread
   over four characters from '0' up, stepped off the punctuation between the
   digits and the letters, then rotated one place right. *)
let exclude =
  [|
    0x3a; 0x3b; 0x3c; 0x3d; 0x3e; 0x3f; 0x40; 0x5b; 0x5c; 0x5d; 0x5e; 0x5f; 0x60;
  |]

let encode x =
  let asc = Bytes.create 16 in
  for i = 0 to 3 do
    let byte = (x lsr (24 - (8 * i))) land 0xFF in
    let q = (byte / 4) + 0x30 and r = byte mod 4 in
    let ch = [| q + r; q; q; q |] in
    let rec settle () =
      let check = ref false in
      Array.iter
        (fun e ->
          let j = ref 0 in
          while !j < 4 do
            if ch.(!j) = e || ch.(!j + 1) = e then begin
              ch.(!j) <- ch.(!j) + 1;
              ch.(!j + 1) <- ch.(!j + 1) - 1;
              check := true
            end;
            j := !j + 2
          done)
        exclude;
      if !check then settle ()
    in
    settle ();
    for j = 0 to 3 do
      Bytes.set asc ((4 * j) + i) (Char.chr ch.(j))
    done
  done;
  String.init 16 (fun i -> Bytes.get asc ((i + 15) mod 16))

(* The CHECKSUM text that makes an HDU whose other bytes sum to [s] sum to -0,
   given that its CHECKSUM field held '0000000000000000' in that sum. *)
let checksum s = encode (mask - s)
