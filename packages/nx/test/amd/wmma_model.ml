(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type fmt = F16 | Bf16 | E4m3 | E5m2

(* Fraction bits, bias, exponent bits. *)
let frac = function F16 -> 10 | Bf16 -> 7 | E4m3 -> 3 | E5m2 -> 2
let bias = function F16 -> 15 | Bf16 -> 127 | E4m3 -> 7 | E5m2 -> 15
let ebits = function F16 -> 5 | Bf16 -> 8 | E4m3 -> 4 | E5m2 -> 5
let bits = function F16 | Bf16 -> 16 | E4m3 | E5m2 -> 8
let is_f8 = function E4m3 | E5m2 -> true | F16 | Bf16 -> false

(* A code's sign, NaN, infinity, significand and exponent: the value is sg
   2^(exp - frac), exp = max field 1 - bias. *)
type factor = { neg : bool; nan : bool; inf : bool; sg : int; exp : int }

let decode f code =
  let nb = bits f and fr = frac f in
  let mag = code land ((1 lsl (nb - 1)) - 1) in
  let field = mag lsr fr and m = mag land ((1 lsl fr) - 1) in
  let emax = (1 lsl ebits f) - 1 in
  let nan, inf =
    match f with
    | E4m3 -> (mag = 0x7F, false)
    | _ -> (field = emax && m <> 0, field = emax && m = 0)
  in
  {
    neg = (code lsr (nb - 1)) land 1 = 1;
    nan;
    inf;
    sg = (if field = 0 then m else m lor (1 lsl fr));
    exp = Int.max field 1 - bias f;
  }

let value f code =
  let x = decode f code in
  if x.nan then Float.nan
  else
    let v =
      if x.inf then Float.infinity
      else Float.ldexp (float_of_int x.sg) (x.exp - frac f)
    in
    if x.neg then -.v else v

(* float32 *)

let factor_nan = 0xFFC00A3D
let invalid_nan = 0xFFC00000
let pos_inf = 0x7F800000
let neg_inf = 0xFF800000

(* The bit length of [n] >= 0. *)
let length n =
  let rec go n l = if n = 0 then l else go (n lsr 1) (l + 1) in
  go n 0

(* The float32 bits of [n 2^grid] rounded to nearest even, +0 where that is
   zero: the R9700 gives +0 for a negative [n] that rounds to zero. *)
let rne n grid =
  if n = 0 then 0
  else
    let sign = if n < 0 then 1 lsl 31 else 0 in
    let m = abs n in
    let e = length m - 1 + grid in
    let q = Int.max e (-126) - 23 in
    let n =
      if q <= grid then m lsl (grid - q)
      else
        let s = q - grid in
        if s > 62 then 0
        else
          let r = m land ((1 lsl s) - 1) and half = 1 lsl (s - 1) in
          let n = m lsr s in
          if r > half || (r = half && n land 1 = 1) then n + 1 else n
    in
    let n, q = if n >= 1 lsl 24 then (n lsr 1, q + 1) else (n, q) in
    if n = 0 then 0
    else if n < 1 lsl 23 then sign lor n
    else
      let biased = q + 23 + 127 in
      if biased >= 255 then sign lor pos_inf
      else sign lor (biased lsl 23) lor (n land 0x7FFFFF)

(* [sg 2^exp] (negated where [neg]) on the grid [2^grid], truncated toward minus
   infinity; an accumulator ([acc]) shifted out entirely is +0. *)
let align ?(acc = false) sg exp neg grid =
  let shift = exp - grid in
  let mag =
    if sg = 0 || shift <= -63 then 0
    else if shift < 0 then sg lsr -shift
    else sg lsl shift
  in
  let tail =
    sg <> 0 && shift < 0 && (shift <= -63 || sg land ((1 lsl -shift) - 1) <> 0)
  in
  if neg then -mag - if tail && ((not acc) || mag <> 0) then 1 else 0 else mag

let none = -1024
let lead sg exp = if sg = 0 then none else exp + length sg - 1

(* Steps *)

let dot fa fb ~group a b c =
  let n = Array.length a in
  let invalid_or_nan = if is_f8 fa then factor_nan else invalid_nan in
  let sgs = Array.make n 0 and exps = Array.make n 0 in
  let negs = Array.make n false and grids = Array.make n none in
  let nan = ref false and invalid = ref false in
  let pinf = ref false and ninf = ref false in
  for k = 0 to n - 1 do
    let l = decode fa a.(k) and r = decode fb b.(k) in
    if l.nan || r.nan then nan := true;
    if (l.inf && r.sg = 0) || (r.inf && l.sg = 0) then invalid := true;
    if l.inf || r.inf then if l.neg <> r.neg then ninf := true else pinf := true;
    sgs.(k) <- l.sg * r.sg;
    exps.(k) <- l.exp + r.exp - frac fa - frac fb;
    negs.(k) <- l.neg <> r.neg;
    if sgs.(k) <> 0 then grids.(k) <- l.exp + r.exp - 24
  done;
  if !nan then factor_nan
  else if !invalid || (!pinf && !ninf) then invalid_or_nan
  else if c land 0x7FFFFFFF > pos_inf then c lor 0x400000
  else
    let pinf = !pinf || c = pos_inf and ninf = !ninf || c = neg_inf in
    if pinf && ninf then invalid_or_nan
    else if ninf then neg_inf
    else if pinf then pos_inf
    else
      let field = (c lsr 23) land 255 in
      let csg =
        if field = 0 then c land 0x7FFFFF else c land 0x7FFFFF lor 0x800000
      in
      let cexp = Int.max field 1 - 127 - 23 in
      let pgrid = Array.fold_left Int.max none grids in
      let grid = Int.max pgrid (Int.max (-126) (lead csg cexp) - 26) in
      let total = ref (align ~acc:(grid > -126) csg cexp (c lsr 31 = 1) grid) in
      let i = ref 0 in
      while !i < n do
        let last = Int.min n (!i + group) in
        let gg = ref none in
        for k = !i to last - 1 do
          gg := Int.max !gg grids.(k)
        done;
        let s = ref 0 in
        for k = !i to last - 1 do
          s := !s + align sgs.(k) exps.(k) negs.(k) !gg
        done;
        total := !total + align (abs !s) !gg (!s < 0) grid;
        i := last
      done;
      rne !total grid

let wmma fa fb a b c =
  let step c first count group =
    dot fa fb ~group (Array.sub a first count) (Array.sub b first count) c
  in
  if is_f8 fa then step (step c 0 8 4) 8 8 4
  else List.fold_left (fun c first -> step c first 4 2) c [ 0; 8; 4; 12 ]

(* Layout slot s of 16-bit elements holds k [order.(s)]: the steps, over the
   slots 0-3, 8-11, 4-7 then 12-15, take k in increasing order. *)
let order = [| 0; 1; 2; 3; 8; 9; 10; 11; 4; 5; 6; 7; 12; 13; 14; 15 |]

let ordered fa fb a b c =
  if is_f8 fa then wmma fa fb a b c
  else
    let slots x = Array.map (fun k -> x.(k)) order in
    wmma fa fb (slots a) (slots b) c

let iwmma ~bits ~sa ~sb a b c =
  let ext s v = if s && v lsr (bits - 1) = 1 then v - (1 lsl bits) else v in
  let acc = ref c in
  Array.iteri (fun k x -> acc := !acc + (ext sa x * ext sb b.(k))) a;
  !acc land 0xFFFF_FFFF

(* Register images *)

let slot ~k r kk =
  if k = 32 then (r + (16 * (kk / 8 mod 2)), (kk mod 8) + (8 * (kk / 16)))
  else (r + (16 * (kk / 8)), kk mod 8)

let image ~bits ~k m =
  let img = Array.make 128 0 in
  let per = 32 / bits in
  Array.iteri
    (fun r row ->
      Array.iteri
        (fun kk v ->
          let lane, e = slot ~k r kk in
          let w = (4 * lane) + (e / per) and s = bits * (e mod per) in
          let mask = (1 lsl bits) - 1 in
          img.(w) <- img.(w) land lnot (mask lsl s) lor ((v land mask) lsl s))
        row)
    m;
  img

let c_image c =
  let img = Array.make 256 0 in
  for i = 0 to 15 do
    for j = 0 to 15 do
      img.((8 * (j + (16 * (i / 8)))) + (i mod 8)) <- c.(i).(j)
    done
  done;
  img

let c_matrix img =
  Array.init 16 (fun i ->
      Array.init 16 (fun j -> img.((8 * (j + (16 * (i / 8)))) + (i mod 8))))
