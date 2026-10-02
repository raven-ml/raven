(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [escaped d] is [true] iff the decode [d] is malformed, a control character, a
   double quote or a backslash. *)
let escaped d =
  let u = Uchar.to_int (Uchar.utf_decode_uchar d) in
  (not (Uchar.utf_decode_is_valid d))
  || u < 0x20
  || (0x7F <= u && u <= 0x9F)
  || u = 0x22 || u = 0x5C

let pp ppf s =
  let b = Buffer.create (String.length s + 2) in
  let rec chars i width =
    if i >= String.length s then width
    else
      let d = String.get_utf_8_uchar s i in
      let n = Uchar.utf_decode_length d in
      let c = String.sub s i n in
      if escaped d then begin
        let e = String.escaped c in
        Buffer.add_string b e;
        chars (i + n) (width + String.length e)
      end
      else begin
        Buffer.add_string b c;
        chars (i + n) (width + 1)
      end
  in
  Buffer.add_char b '"';
  let width = chars 0 2 in
  Buffer.add_char b '"';
  Format.pp_print_as ppf width (Buffer.contents b)
