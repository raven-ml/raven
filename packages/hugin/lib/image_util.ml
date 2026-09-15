(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shared image encoding utilities *)

(* Base64 *)

let base64_alphabet =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

let base64_encode input =
  let len = String.length input in
  let out_len = (len + 2) / 3 * 4 in
  let out = Bytes.create out_len in
  let rec loop i j =
    if i < len then begin
      let b0 = Char.code (String.unsafe_get input i) in
      let b1 =
        if i + 1 < len then Char.code (String.unsafe_get input (i + 1)) else 0
      in
      let b2 =
        if i + 2 < len then Char.code (String.unsafe_get input (i + 2)) else 0
      in
      Bytes.unsafe_set out j (String.unsafe_get base64_alphabet (b0 lsr 2));
      Bytes.unsafe_set out (j + 1)
        (String.unsafe_get base64_alphabet (((b0 land 3) lsl 4) lor (b1 lsr 4)));
      Bytes.unsafe_set out (j + 2)
        (if i + 1 < len then
           String.unsafe_get base64_alphabet
             (((b1 land 0xf) lsl 2) lor (b2 lsr 6))
         else '=');
      Bytes.unsafe_set out (j + 3)
        (if i + 2 < len then String.unsafe_get base64_alphabet (b2 land 0x3f)
         else '=');
      loop (i + 3) (j + 4)
    end
  in
  loop 0 0;
  Bytes.unsafe_to_string out
