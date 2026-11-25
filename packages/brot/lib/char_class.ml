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
let mark = 16
let upper = 32
let lower = 64
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
    set i (letter lor word lor upper)
  done;
  for i = 97 to 122 do
    set i (letter lor word lor lower)
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
      let mark = match gc with `Mc | `Me | `Mn -> mark | _ -> 0 in
      let case =
        match gc with `Lu | `Lt -> upper | `Ll -> lower | _ -> 0
      in
      category lor punct lor word lor mark lor case lor known
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
let[@inline] at_is_mark d = (d lsr 3) land mark <> 0
let[@inline] at_is_upper d = (d lsr 3) land upper <> 0
let[@inline] at_is_lower d = (d lsr 3) land lower <> 0
let[@inline] pack cp len = (props cp lsl 3) lor len
let stray = ((other lor known) lsl 3) lor 1

(* RFC 3629's well-formed sequences, as Unicode's Table 3-7 lists them: the
   second byte's range depends on the lead, which rules out overlong forms (E0
   80..9F, F0 80..8F), surrogates (ED A0..BF) and values above U+10FFFF (F4
   90..BF, F5..FF). *)
let[@inline] byte s i = Char.code (String.unsafe_get s i)
let[@inline] in_range b lo hi = b >= lo && b <= hi
let[@inline] continuation b = b land 0xC0 = 0x80

let decode s i ~stop =
  let c = byte s i in
  if c < 0x80 then (c lsl 3) lor 1
  else if c < 0xC2 then -1
  else if c < 0xE0 then
    if i + 1 >= stop then -1
    else
      let b1 = byte s (i + 1) in
      if not (continuation b1) then -1
      else ((((c land 0x1F) lsl 6) lor (b1 land 0x3F)) lsl 3) lor 2
  else if c < 0xF0 then
    if i + 2 >= stop then -1
    else
      let b1 = byte s (i + 1) and b2 = byte s (i + 2) in
      let lo = if c = 0xE0 then 0xA0 else 0x80 in
      let hi = if c = 0xED then 0x9F else 0xBF in
      if not (in_range b1 lo hi && continuation b2) then -1
      else
        (((c land 0x0F) lsl 12) lor ((b1 land 0x3F) lsl 6) lor (b2 land 0x3F))
        lsl 3
        lor 3
  else if c < 0xF5 then
    if i + 3 >= stop then -1
    else
      let b1 = byte s (i + 1) and b2 = byte s (i + 2) and b3 = byte s (i + 3) in
      let lo = if c = 0xF0 then 0x90 else 0x80 in
      let hi = if c = 0xF4 then 0x8F else 0xBF in
      if not (in_range b1 lo hi && continuation b2 && continuation b3) then -1
      else
        (((c land 0x07) lsl 18)
        lor ((b1 land 0x3F) lsl 12)
        lor ((b2 land 0x3F) lsl 6)
        lor (b3 land 0x3F))
        lsl 3
        lor 4
  else -1

let length s i ~stop =
  let d = decode s i ~stop in
  if d < 0 then 1 else d land 7

let is_one_character s =
  let stop = String.length s in
  stop > 0
  &&
  let d = decode s 0 ~stop in
  d >= 0 && d land 7 = stop

let[@inline never] at_beyond_ascii s i stop =
  let d = decode s i ~stop in
  if d < 0 then stray else pack (d lsr 3) (d land 7)

let[@inline] at s i ~stop =
  let c = Char.code (String.unsafe_get s i) in
  if c < 0x80 then (Char.code (Bytes.unsafe_get ascii c) lsl 3) lor 1
  else at_beyond_ascii s i stop

(* Internals *)

let unicode_table () = !unicode
