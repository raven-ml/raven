(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* CPython's _PyUnicode_IsWhitespace: the characters of bidirectional class WS,
   B or S, or of category Zs. *)
let is_space u =
  let c = Uchar.to_int u in
  (c >= 0x09 && c <= 0x0D)
  || (c >= 0x1C && c <= 0x20)
  || c = 0x85 || c = 0xA0 || c = 0x1680
  || (c >= 0x2000 && c <= 0x200A)
  || c = 0x2028 || c = 0x2029 || c = 0x202F || c = 0x205F || c = 0x3000

let strip s =
  let n = String.length s in
  (* The start of the first character that is no space, and the end of the
     last. *)
  let rec scan i first last =
    if i >= n then (first, last)
    else
      let d = String.get_utf_8_uchar s i in
      let next = i + Uchar.utf_decode_length d in
      if Uchar.utf_decode_is_valid d && is_space (Uchar.utf_decode_uchar d) then
        scan next first last
      else scan next (if first < 0 then i else first) next
  in
  let first, last = scan 0 (-1) 0 in
  if first < 0 then "" else String.sub s first (last - first)
