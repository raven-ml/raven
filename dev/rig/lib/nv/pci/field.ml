(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Structures *)

let set b (off, n) x =
  match n with
  | 1 -> Bytes.set_uint8 b off x
  | 2 -> Bytes.set_uint16_le b off x
  | 4 -> Bytes.set_int32_le b off (Int32.of_int x)
  | _ -> Bytes.set_int64_le b off (Int64.of_int x)

let get s (off, n) =
  match n with
  | 1 -> String.get_uint8 s off
  | 2 -> String.get_uint16_le s off
  | 4 -> Int32.to_int (String.get_int32_le s off) land 0xffff_ffff
  | _ -> Int64.to_int (String.get_int64_le s off)

let at (base, _) (off, n) = (base + off, n)

let record size fill =
  let b = Bytes.make size '\000' in
  fill b;
  Bytes.unsafe_to_string b

(* Registers *)

let mask (lo, n) = ((1 lsl n) - 1) lsl lo
let put (lo, n) x = (x land ((1 lsl n) - 1)) lsl lo
