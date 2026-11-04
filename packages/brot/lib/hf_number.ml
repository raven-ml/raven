(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Powers of ten. serde_json's table holds the double nearest each 10^k up to
   10^308. 10^k is 5^k × 2^k, so the table is 5^k, held exactly as an integer in
   limbs of 24 bits, least significant first, rounded to 53 bits half to even,
   then scaled by 2^k, which is exact. Every integer here stays under 2^27, and
   the 53 bits gather in a double, which holds them exactly, so the table is
   the same where [int] has 31 bits. *)

let limb = 24
let bit n i = (n.(i / limb) lsr (i mod limb)) land 1

let times5 n =
  let carry = ref 0 in
  let n =
    Array.map
      (fun x ->
        let y = (5 * x) + !carry in
        carry := y lsr limb;
        y land ((1 lsl limb) - 1))
      n
  in
  if !carry = 0 then n else Array.append n [| !carry |]

let rec width x = if x = 0 then 0 else 1 + width (x lsr 1)

let nearest_double n =
  let top = Array.length n - 1 in
  let bits = (top * limb) + width n.(top) in
  let shift = max 0 (bits - 53) in
  let m = ref 0.0 in
  for i = bits - 1 downto shift do
    m := (2.0 *. !m) +. Float.of_int (bit n i)
  done;
  (* Whether any of the bits under the half bit, [0 .. shift - 2], is set:
     whole limbs, then the low bits of the one the half bit is in. *)
  let under = max 0 (shift - 1) in
  let sticky = ref false in
  for j = 0 to (under / limb) - 1 do
    if n.(j) <> 0 then sticky := true
  done;
  let low = under mod limb in
  if low > 0 && n.(under / limb) land ((1 lsl low) - 1) <> 0 then
    sticky := true;
  let half = shift > 0 && bit n (shift - 1) = 1 in
  let odd = bit n shift = 1 in
  let m = if half && (!sticky || odd) then !m +. 1.0 else !m in
  Float.ldexp m shift

(* Built when the module is, before any domain can read it. *)
let pow10 =
  let table = Array.make 309 1.0 in
  let five = ref [| 1 |] in
  for k = 1 to 308 do
    five := times5 !five;
    table.(k) <- Float.ldexp (nearest_double !five) k
  done;
  table

(* serde_json's arithmetic *)

let err_range = "number out of range"

(* An unsigned 64-bit integer held in an [Int64.t], rounded to the nearest
   double, ties to even. Its two halves of 32 bits are exact doubles and so is
   the high one scaled by 2^32, so their sum rounds once, as the conversion
   does. *)
let u64_to_float x =
  let high = Int64.to_float (Int64.shift_right_logical x 32) in
  let low = Int64.to_float (Int64.logand x 0xffff_ffffL) in
  Float.ldexp high 32 +. low

(* serde_json's [overflow!(s * 10 + d, u64::MAX)]: u64::MAX is
   1844674407370955161 tens and 5. *)
let u64_max_tens = Int64.unsigned_div (-1L) 10L

let overflows s d =
  let c = Int64.unsigned_compare s u64_max_tens in
  c > 0 || (c = 0 && d > 5)

(* The same for an exponent against i32::MAX, 214748364 tens and 7. The
   exponent's arithmetic needs an [int] of 32 bits or more, as OCaml and
   js_of_ocaml have. *)
let i32_max = 0x7fff_ffff
let i32_min = -i32_max - 1
let exp_overflows e d = e > i32_max / 10 || (e = i32_max / 10 && d > 7)

(* [a + b] held to i32's range, as serde_json's [saturating_add], without
   leaving that range on the way. *)
let saturating_add a b =
  if b > 0 && a > i32_max - b then i32_max
  else if b < 0 && a < i32_min - b then i32_min
  else a + b

(* [f] × 10^[e]: one rounded multiplication or division, after dividing by
   10^308 for as long as [e] is below -308. *)
let rec scale f e =
  if e >= -308 && e <= 308 then
    if e >= 0 then
      let f = f *. pow10.(e) in
      if Float.is_finite f then f else failwith err_range
    else f /. pow10.(-e)
  else if f = 0.0 then f
  else if e >= 0 then failwith err_range
  else scale (f /. pow10.(308)) (e + 308)

let digit text i stop =
  if i >= stop then -1
  else
    let c = text.[i] in
    if c >= '0' && c <= '9' then Char.code c - Char.code '0' else -1

let read text ~first ~last =
  let stop = last + 1 in
  let at = ref first in
  let negative = !at < stop && text.[!at] = '-' in
  if negative then incr at;
  let significand = ref 0L and exponent = ref 0 in
  let push d =
    significand := Int64.add (Int64.mul !significand 10L) (Int64.of_int d)
  in
  (* The integer part goes into the significand until a digit would overflow
     it; that digit and the ones after it only count tens. *)
  let full = ref false in
  while digit text !at stop >= 0 do
    let d = digit text !at stop in
    if (not !full) && overflows !significand d then full := true;
    if !full then incr exponent else push d;
    incr at
  done;
  (* The fraction goes in until a digit would overflow, and the rest is
     dropped. *)
  if !at < stop && text.[!at] = '.' then begin
    incr at;
    let full = ref false in
    while digit text !at stop >= 0 do
      let d = digit text !at stop in
      if (not !full) && overflows !significand d then full := true;
      if not !full then begin
        push d;
        decr exponent
      end;
      incr at
    done
  end;
  let f = u64_to_float !significand in
  let value =
    if !at < stop && (text.[!at] = 'e' || text.[!at] = 'E') then begin
      incr at;
      let up = not (!at < stop && text.[!at] = '-') in
      if !at < stop && (text.[!at] = '-' || text.[!at] = '+') then incr at;
      (* An exponent past i32::MAX makes the number zero, or out of range
         when it is positive and the digits are not all zeros. *)
      let e = ref 0 and huge = ref false in
      while digit text !at stop >= 0 do
        let d = digit text !at stop in
        if (not !huge) && exp_overflows !e d then huge := true;
        if not !huge then e := (10 * !e) + d;
        incr at
      done;
      if not !huge then
        scale f (saturating_add !exponent (if up then !e else - !e))
      else if up && !significand <> 0L then failwith err_range
      else 0.0
    end
    else scale f !exponent
  in
  if negative then -.value else value
