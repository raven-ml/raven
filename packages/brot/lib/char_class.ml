(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let other = 0
let whitespace = 1
let letter = 2
let numeric = 3
let punct = 4
let word = 8
let known = 128

let ascii =
  let t = Bytes.make 128 (Char.unsafe_chr (other lor known)) in
  let set i v = Bytes.set t i (Char.unsafe_chr (v lor known)) in
  for i = 33 to 126 do
    set i (other lor punct)
  done;
  for i = 9 to 13 do
    set i whitespace
  done;
  set 32 whitespace;
  for i = 48 to 57 do
    set i (numeric lor word)
  done;
  for i = 65 to 90 do
    set i (letter lor word)
  done;
  for i = 97 to 122 do
    set i (letter lor word)
  done;
  set 95 (other lor punct lor word);
  t

let unicode = ref Bytes.empty

let classify t cp =
  let v =
    if not (Uchar.is_valid cp) then other lor known
    else
      let u = Uchar.unsafe_of_int cp in
      let gc = Uucp.Gc.general_category u in
      let category =
        if Uucp.White.is_white_space u then whitespace
        else
          match gc with
          | `Ll | `Lm | `Lo | `Lt | `Lu -> letter
          | `Nd | `Nl | `No -> numeric
          | _ -> other
      in
      let punct =
        match gc with
        | `Pc | `Pd | `Pe | `Pf | `Pi | `Po | `Ps -> punct
        | _ -> 0
      in
      let word =
        if
          Uucp.Alpha.is_alphabetic u
          || (match gc with `Mc | `Me | `Mn | `Nd | `Pc -> true | _ -> false)
          || Uucp.Func.is_join_control u
        then word
        else 0
      in
      category lor punct lor word lor known
  in
  Bytes.unsafe_set t (cp - 128) (Char.unsafe_chr v);
  v

let[@inline never] fill cp =
  if cp > 0x10FFFF then other lor known
  else
    let t =
      if Bytes.length !unicode <> 0 then !unicode
      else begin
        let t = Bytes.make (0x110000 - 128) '\000' in
        unicode := t;
        t
      end
    in
    classify t cp

let[@inline] props cp =
  if cp < 128 then Char.code (Bytes.unsafe_get ascii cp)
  else
    let t = !unicode in
    let i = cp - 128 in
    if i < Bytes.length t then
      let v = Char.code (Bytes.unsafe_get t i) in
      if v <> 0 then v else fill cp
    else fill cp

let[@inline] category cp = props cp land 3
let[@inline] at_len d = d land 7
let[@inline] at_category d = (d lsr 3) land 3
let[@inline] at_is_punctuation d = (d lsr 3) land punct <> 0
let[@inline] at_is_word d = (d lsr 3) land word <> 0
let[@inline] pack cp len = (props cp lsl 3) lor len
let stray = ((other lor known) lsl 3) lor 1

let[@inline never] decode s i stop =
  let c = Char.code (String.unsafe_get s i) in
  if c < 0xC2 then stray
  else if c < 0xE0 then
    if i + 1 >= stop then stray
    else
      let b1 = Char.code (String.unsafe_get s (i + 1)) in
      if b1 land 0xC0 <> 0x80 then stray
      else pack (((c land 0x1F) lsl 6) lor (b1 land 0x3F)) 2
  else if c < 0xF0 then
    if i + 2 >= stop then stray
    else
      let b1 = Char.code (String.unsafe_get s (i + 1)) in
      let b2 = Char.code (String.unsafe_get s (i + 2)) in
      if b1 land 0xC0 <> 0x80 || b2 land 0xC0 <> 0x80 then stray
      else
        pack
          (((c land 0x0F) lsl 12) lor ((b1 land 0x3F) lsl 6) lor (b2 land 0x3F))
          3
  else if c < 0xF8 then
    if i + 3 >= stop then stray
    else
      let b1 = Char.code (String.unsafe_get s (i + 1)) in
      let b2 = Char.code (String.unsafe_get s (i + 2)) in
      let b3 = Char.code (String.unsafe_get s (i + 3)) in
      if b1 land 0xC0 <> 0x80 || b2 land 0xC0 <> 0x80 || b3 land 0xC0 <> 0x80
      then stray
      else
        pack
          (((c land 0x07) lsl 18)
          lor ((b1 land 0x3F) lsl 12)
          lor ((b2 land 0x3F) lsl 6)
          lor (b3 land 0x3F))
          4
  else stray

let[@inline] at s i ~stop =
  let c = Char.code (String.unsafe_get s i) in
  if c < 0x80 then (Char.code (Bytes.unsafe_get ascii c) lsl 3) lor 1
  else decode s i stop
