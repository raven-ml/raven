(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* A format's codes as unsigned integers: [floor r] is the greatest
   non-negative code whose value is at most [r], for [r >= 0.] finite, and
   [value c] the value of a non-negative code, [2^emax] at the code of
   [+inf], so that the codes about the greatest finite one have a midpoint. *)
type format = {
  sign : int;
  inf : int;
  floor : float -> int;
  value : int -> float;
}

let nan f c = c land (f.sign - 1) > f.inf
let rank f c = if c land f.sign <> 0 then -(c land (f.sign - 1)) - 1 else c

let f32 =
  let value c =
    if c = 0x7F800000 then Float.ldexp 1. 128
    else Int32.float_of_bits (Int32.of_int c)
  in
  let floor r =
    let c = Int32.to_int (Int32.bits_of_float r) land 0xFFFFFFFF in
    if value c > r then c - 1 else c
  in
  { sign = 0x80000000; inf = 0x7F800000; floor; value }

(* A 16-bit format from the float64 values of its codes, which widen
   exactly. *)
let f16 (type s) (dt : (float, s) Nx.dtype) =
  let inf, emax =
    match dt with
    | Nx.Dtype.Float16 -> (0x7C00, 16)
    | Nx.Dtype.Bfloat16 -> (0x7F80, 128)
    | _ -> invalid_arg "Exps.law16: not a 16-bit float"
  in
  let codes = Nx.create Nx.uint16 [| inf |] (Array.init inf Fun.id) in
  let values = Nx.to_array (Nx.cast Nx.float64 (Nx.bitcast dt codes)) in
  let value c = if c = inf then Float.ldexp 1. emax else values.(c) in
  (* The greatest code in [lo, hi] at most [r], [value lo <= r]. *)
  let rec search lo hi r =
    if lo = hi then lo
    else
      let mid = (lo + hi + 1) / 2 in
      if value mid <= r then search mid hi r else search lo (mid - 1) r
  in
  { sign = 0x8000; inf; floor = (fun r -> search 0 inf r); value }

(* Within this share of an ulp of a midpoint, libm's float64 value may round
   apart from the exact one. *)
let margin = 0x1p-16

(* The codes that count as the correctly rounded value of [r], libm's
   float64 result, [r >= 0.]. *)
let references f r =
  if r = Float.infinity then [ f.inf ]
  else
    let lo = f.floor r in
    let vl = f.value lo in
    if lo = f.inf || vl = r then [ lo ]
    else
      let frac = (r -. vl) /. (f.value (lo + 1) -. vl) in
      if Float.abs (frac -. 0.5) <= margin then [ lo; lo + 1 ]
      else if frac < 0.5 then [ lo ]
      else [ lo + 1 ]

let distance f got refs =
  List.fold_left (fun d c -> min d (abs (rank f got - rank f c))) max_int refs

(* What an operand's result must be: a NaN, a code, within the bound of one of
   the reference's codes, or anything where it is not checked. *)
type expect = Nan | Exactly of int | Near of int list | Unchecked

let holds f ~ulps g = function
  | Nan -> nan f g
  | Exactly c -> g = c
  | Near refs -> (not (nan f g)) && distance f g refs <= ulps
  | Unchecked -> true

let show = function
  | Nan -> "a NaN"
  | Exactly c -> Printf.sprintf "%#x exactly" c
  | Near refs -> String.concat " or " (List.map (Printf.sprintf "%#x") refs)
  | Unchecked -> "unchecked"

(* The operands at which [got] breaks the law, at most eight. *)
let broken f ~ulps xs expects got =
  let out = ref [] in
  Array.iteri
    (fun i x ->
      if (not (holds f ~ulps got.(i) expects.(i))) && List.length !out < 8 then
        out :=
          Printf.sprintf "exp(%h) = %#x, reference %s, beyond %d ulps" x got.(i)
            (show expects.(i)) ulps
          :: !out)
    xs;
  List.rev !out

let compiled (type s) (module S : Nx.Devices) (x : (float, s, Nx.host) Nx.t) =
  let before = Rune.evaluations () in
  let y =
    Rune.jit Nx.Ptree.(tensor @-> returns tensor) Nx.exp (Nx.place S.on x)
  in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  Nx.place Nx.Host.on y

let least_normal = Float.ldexp 1. (-126)

(* Annex F's value where the operand is a zero, an infinity or a NaN, and the
   reference elsewhere. *)
let expect f ~flushes x =
  if Float.is_nan x then Nan
  else if x = 0. then Exactly (f.floor 1.)
  else if x = Float.infinity then Exactly f.inf
  else if x = Float.neg_infinity then Exactly 0
  else
    let r = Float.exp x in
    if flushes && r < least_normal then Unchecked else Near (references f r)

(* The operands where Annex F states the value. *)
let specials =
  [|
    ("an operand of +0.", 0.);
    ("an operand of -0.", -0.);
    ("an operand of +inf", Float.infinity);
    ("an operand of -inf", Float.neg_infinity);
  |]

(* Every case's first codes: Annex F's operands, NaNs quiet and signalling
   with payloads, and the least subnormals and greatest finite floats. *)
let special_codes =
  [
    0l; Int32.min_int; 0x7F800000l; 0xFF800000l; 0x7FC00000l; 0xFFC00001l;
    0x7F800001l; 1l; 0x80000001l; 0x7F7FFFFFl; 0xFF7FFFFFl;
  ]

let law (module S : Nx.Devices) ~ulps ~flushes codes =
  let n = Array.length codes in
  let x = Nx.bitcast Nx.float32 (Nx.create Nx.int32 [| n |] codes) in
  let got =
    Array.map
      (fun c -> Int32.to_int c land 0xFFFFFFFF)
      (Nx.to_array (Nx.bitcast Nx.int32 (compiled (module S) x)))
  in
  let xs = Array.map Int32.float_of_bits codes in
  let expects = Array.map (expect f32 ~flushes) xs in
  let has p =
    Array.exists (function Near refs -> p refs | _ -> false) expects
  in
  (* A device that flushes leaves the results below the least normal
     unchecked. *)
  if not flushes then begin
    cover "a subnormal result"
      (has (List.for_all (fun c -> c > 0 && c < 0x00800000)));
    cover "a result that rounds to +0." (has (( = ) [ 0 ]))
  end;
  cover "a result that rounds to +inf" (has (( = ) [ f32.inf ]));
  Array.iter
    (fun (label, x) ->
      let same y = Int64.bits_of_float y = Int64.bits_of_float x in
      cover label (Array.exists same xs))
    specials;
  cover "a result of 1. from a non-zero operand"
    (has (( = ) [ f32.floor 1. ]));
  equal (list string) [] (broken f32 ~ulps xs expects got)

(* Generators *)

let bits32 v = Int32.bits_of_float v
let block g = Gen.array ~size:(Gen.constant 256) g

let gen =
  let open Gen in
  let range lo hi = map bits32 (float_range lo hi) in
  (* Operands below one in magnitude down to float32's least subnormal, whose
     results round to [1.] below [2^-25]. *)
  let small =
    let+ e = int_range (-149) (-1)
    and+ m = float_range 1. 2.
    and+ neg = bool in
    bits32 (Float.ldexp (if neg then -.m else m) e)
  in
  let+ a = block int32
  and+ b = block (range (-104.) 89.)
  (* Overflow above about 88.72, subnormal results below about -87.34 and
     zero below about -103.97. *)
  and+ c = block (range 88.5 89.)
  and+ d = block (range (-87.6) (-87.1))
  and+ e = block (range (-104.2) (-103.7))
  and+ f = block small in
  Array.concat [ Array.of_list special_codes; a; b; c; d; e; f ]

(* A case prints as its size: a failure names each operand that breaks the
   law. *)
let gen =
  Gen.with_pp
    (fun ppf a -> Format.fprintf ppf "%d float32 codes" (Array.length a))
    gen

let law16 (type s) (module S : Nx.Devices) ~flushes
    (dt : (float, s) Nx.dtype) =
  let f = f16 dt in
  let codes = Array.init 0x10000 Fun.id in
  let x = Nx.bitcast dt (Nx.create Nx.uint16 [| 0x10000 |] codes) in
  let got = Nx.to_array (Nx.bitcast Nx.uint16 (compiled (module S) x)) in
  let xs = Nx.to_array (Nx.cast Nx.float64 x) in
  let expects = Array.map (expect f ~flushes) xs in
  equal (list string) [] (broken f ~ulps:1 xs expects got)
