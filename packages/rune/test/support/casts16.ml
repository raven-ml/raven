(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* Elements are compared as their bits, unsigned in an [int]: an OCaml float
   would quiet a signalling NaN. *)

(* A 16-bit format: its quiet bit and the code of its positive infinity. *)
type format = { quiet : int; inf : int }

let format (type s) (dt : (float, s) Nx.dtype) =
  match dt with
  | Nx.Dtype.Float16 -> { quiet = 0x200; inf = 0x7C00 }
  | Nx.Dtype.Bfloat16 -> { quiet = 0x40; inf = 0x7F80 }
  | _ -> invalid_arg "Casts16.law: not a 16-bit float"

let nan16 f c = c land 0x7FFF > f.inf
let quiet16 f c = c land f.quiet <> 0
let nan32 b = b land 0x7FFFFFFF > 0x7F800000
let quiet32 b = b land 0x400000 <> 0
let sign32 = 0x80000000

(* float32's infinities, and NaNs quiet and signalling of both signs. *)
let specials32 =
  [
    0x7F800000; 0xFF800000; 0x7FC00000; 0xFFC12345; 0x7FFFFFFF; 0x7F800001;
    0xFFA00002; 0x7FBFFFFF;
  ]

let host x = Nx.place Nx.Host.on x
let unsigned b = Int32.to_int b land 0xFFFFFFFF
let bits16 x = Nx.to_array (Nx.bitcast Nx.uint16 (host x))
let bits32 x = Array.map unsigned (Nx.to_array (Nx.bitcast Nx.int32 (host x)))

let of_bits32 bs =
  let n = Array.length bs in
  Nx.bitcast Nx.float32 (Nx.create Nx.int32 [| n |] (Array.map Int32.of_int bs))

(* The float32 operands of a narrowing to [f], from [widened], every code's
   float32 bits: each finite code's value, the midpoint to the next code and
   its neighbours, of both signs. The midpoint past the largest finite code is
   a half step above it, as if the exponent were unbounded. Every midpoint is
   a float32: it needs one bit more than the format holds. *)
let narrowed f widened =
  let v c = Int32.float_of_bits (Int32.of_int widened.(c)) in
  let mid c =
    if c + 1 < f.inf then (v c +. v (c + 1)) /. 2.
    else v c +. ((v c -. v (c - 1)) /. 2.)
  in
  let around c =
    let m = unsigned (Int32.bits_of_float (mid c)) in
    [ widened.(c); m - 1; m; m + 1 ]
  in
  let pos = List.concat_map around (List.init f.inf Fun.id) in
  Array.of_list (pos @ List.map (fun b -> b lor sign32) pos @ specials32)

(* The operands at which [got] breaks the law against [want], at most eight:
   eager's bits, or a NaN where eager's is one, quiet where the operand is. *)
let broken ~label ~nan_in ~quiet_in ~nan_out ~quiet_out xs want got =
  let out = ref [] in
  Array.iteri
    (fun i x ->
      let e = want.(i) and j = got.(i) in
      let ok =
        if not (nan_out e) then e = j
        else nan_out j && ((not (nan_in x && quiet_in x)) || quiet_out j)
      in
      if (not ok) && List.length !out < 8 then
        out :=
          Printf.sprintf "%s %#x: eager %#x, compiled %#x" label x e j :: !out)
    xs;
  List.rev !out

let law (type s) (module S : Nx.Devices) (dt : (float, s) Nx.dtype) =
  let f = format dt in
  let compiled g x =
    let before = Rune.evaluations () in
    let y = Rune.jit Nx.Ptree.(tensor @-> returns tensor) g (Nx.place S.on x) in
    equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
    y
  in
  let codes = Array.init 0x10000 Fun.id in
  let x16 = Nx.bitcast dt (Nx.create Nx.uint16 [| 0x10000 |] codes) in
  let widen x = Nx.cast Nx.float32 x and narrow x = Nx.cast dt x in
  let widened = bits32 (widen x16) in
  let inputs = narrowed f widened in
  let x32 = of_bits32 inputs in
  let round x = narrow (widen x) in
  let from16 = (nan16 f, quiet16 f) and from32 = (nan32, quiet32) in
  let check label (nan_in, quiet_in) (nan_out, quiet_out) xs want got =
    equal ~msg:label (list string) []
      (broken ~label ~nan_in ~quiet_in ~nan_out ~quiet_out xs want got)
  in
  check "widened" from16 from32 codes widened (bits32 (compiled widen x16));
  check "narrowed" from32 from16 inputs (bits16 (narrow x32))
    (bits16 (compiled narrow x32));
  check "round trip" from16 from16 codes (bits16 (round x16))
    (bits16 (compiled round x16));
  let across (type r) (o : (float, r) Nx.dtype) =
    let cross x = Nx.cast o x and g = format o in
    check "across" from16 (nan16 g, quiet16 g) codes (bits16 (cross x16))
      (bits16 (compiled cross x16))
  in
  match dt with
  | Nx.Dtype.Float16 -> across Nx.bfloat16
  | _ -> across Nx.float16
