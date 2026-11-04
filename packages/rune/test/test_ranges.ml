(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A range gather ({!Nx.Prim.Gather_ranges}) through rune: its transpose
   satisfies the adjoint law, so it reads the positions the kernel reads,
   at starts and ends near int64's opposite extremes too; its gradient is
   its finite differences; its tangent is the range gather of the tangent;
   a map over any of its operands is its lanes one by one; and a compiled
   call has eager's bits. *)

open Windtrap
module T = Nx.Ptree

(* Cases *)

(* Runs over [lead] rows of a source of [n] positions and [cells], packed
   into [size] positions: ends that sum lengths, or ends near int64's least
   before starts near its greatest. *)
type case = {
  lead : int array;
  n : int;
  cells : int array;
  size : int;
  k : int;
  starts : int64 array;
  ends : int64 array;
}

let pp_int64s ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map Int64.to_string a)))

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let pp_case ppf c =
  Format.fprintf ppf "lead %a, n %d, cells %a, size %d: starts %a ends %a"
    pp_ints c.lead c.n pp_ints c.cells c.size pp_int64s c.starts pp_int64s
    c.ends

let case_gen =
  let open Gen in
  let* lead = of_list [ [||]; [| 2 |] ] in
  let* n = int_range 0 6 in
  let* cells = of_list [ [||]; [| 2 |] ] in
  let* k = int_range 0 4 in
  let* size = int_range 0 10 in
  let* wrap = bool in
  let rows = Array.fold_left ( * ) 1 lead in
  let start =
    frequency
      [
        (6, map Int64.of_int (int_range (-2) (n + 1)));
        (1, of_list [ Int64.min_int; Int64.max_int; Int64.pred Int64.max_int ]);
      ]
  in
  let* starts = array ~size:(constant (rows * k)) start in
  let* lengths = array ~size:(constant (rows * k)) (int_range (-1) 4) in
  let+ cs = array ~size:(constant (rows * k)) (int_range 0 2) in
  (* In [wrap], odd runs start near int64's greatest after an end near its
     least, and read the axis from position [p - 1 - c] on, modulo 2^64. *)
  let odd i = wrap && i mod k mod 2 = 1 in
  let ends = Array.make (rows * k) 0L in
  for i = 0 to (rows * k) - 1 do
    let before = if i mod k = 0 then 0L else ends.(i - 1) in
    ends.(i) <-
      (if not wrap then Int64.add before (Int64.of_int lengths.(i))
       else if odd i then Int64.of_int (3 + lengths.(i) + 1)
       else Int64.add Int64.min_int (Int64.of_int cs.(i)))
  done;
  let starts =
    Array.mapi
      (fun i s ->
        if odd i then Int64.sub Int64.max_int (Int64.of_int cs.(i)) else s)
      starts
  in
  { lead; n; cells; size; k; starts; ends }

let cases = Gen.with_pp pp_case case_gen

let runs c a = Nx.create Nx.int64 (Array.append c.lead [| c.k |]) a
let source c = Array.concat [ c.lead; [| c.n |]; c.cells ]

let result c =
  let s = source c in
  s.(Array.length c.lead) <- c.size;
  s

let numel s = Array.fold_left ( * ) 1 s

(* The range gather of [x] by [c]'s runs. *)
let ranges ?(starts_of = fun c -> runs c c.starts)
    ?(ends_of = fun c -> runs c c.ends) c x =
  Nx.Prim.eval ~by:"test"
    (Gather_ranges { size = c.size; starts = starts_of c; ends = ends_of c; x })

(* Small integers, whose sums of products every float holds exactly. *)
let ints dt s seed =
  Nx.create dt s
    (Array.init (numel s) (fun i ->
         Float.of_int ((((i * 7) + seed) mod 11) - 5)))

let float_of_scalar (type s) (x : (float, s, Nx.host) Nx.t) = Nx.item [] x

(* Whether a run reads a position inside the axis that the exact sum puts
   outside it: a start at least 0 whose sum with the position's offset wraps
   past 2^64 to below the start. *)
let wraps c =
  let rows = numel c.lead and found = ref false in
  for r = 0 to rows - 1 do
    for p = 0 to c.size - 1 do
      let p64 = Int64.of_int p and at j = c.ends.((r * c.k) + j) in
      let j = ref 0 in
      while !j < c.k && Int64.compare (at !j) p64 <= 0 do incr j done;
      if !j < c.k then begin
        let s = c.starts.((r * c.k) + !j) in
        let before = if !j = 0 then 0L else at (!j - 1) in
        let w = Int64.add s (Int64.sub p64 before) in
        if
          Int64.compare s 0L >= 0 && Int64.compare w s < 0
          && Int64.compare w 0L >= 0
          && Int64.compare w (Int64.of_int c.n) < 0
        then found := true
      end
    done
  done;
  !found

(* Laws *)

(* <w, L v> = <L^T w, v>, in float64 and as Re Σ unconjugated in complex64:
   integer elements make both sides exact, so the transpose's positions are
   the kernel's. *)
let law_adjoint c =
  cover "a position inside the axis only modulo 2^64" (wraps c);
  cover "no run" (c.k = 0);
  cover "an empty axis" (c.n = 0);
  cover "cells" (c.cells <> [||]);
  cover "lead axes" (c.lead <> [||]);
  let v = ints Nx.float64 (source c) 1 and w = ints Nx.float64 (result c) 2 in
  let lt = Rune.linear_transpose T.tensor T.tensor (ranges c) v in
  let dot a b = float_of_scalar (Nx.sum (Nx.mul a b)) in
  equal ~msg:"float64" float_exact (dot w (ranges c v)) (dot (lt w) v);
  let cx a b =
    Nx.complex Nx.complex64 ~re:(Nx.cast Nx.float32 a)
      ~im:(Nx.cast Nx.float32 b)
  in
  let vc = cx v (ints Nx.float64 (source c) 3)
  and wc = cx w (ints Nx.float64 (result c) 4) in
  let ltc = Rune.linear_transpose T.tensor T.tensor (ranges c) vc in
  let re a b = Nx.item [] (Nx.real Nx.float32 (Nx.sum (Nx.mul a b))) in
  equal ~msg:"complex64" float_exact (re wc (ranges c vc)) (re (ltc wc) vc)

(* The gradient of Σ w · L x is its finite differences, exact over small
   integers since L is linear, at float32 and float64. *)
let law_grad c =
  cover "a position inside the axis only modulo 2^64" (wraps c);
  let check (type s) name (dt : (float, s) Nx.dtype) =
    let x = ints dt (source c) 1 and w = ints dt (result c) 2 in
    let loss x = Nx.sum (Nx.mul w (ranges c x)) in
    let g = Rune.grad' loss x in
    let fd =
      Array.init (numel (source c)) (fun i ->
          let m = numel (source c) in
          let e =
            Nx.create dt (source c)
              (Array.init m (fun j -> if j = i then 1. else 0.))
          in
          let at d = float_of_scalar (loss (Nx.add x (Nx.mul_s e d))) in
          (at 1. -. at (-1.)) /. 2.)
    in
    equal ~msg:name (array float_exact) fd (Nx.to_array (Nx.flatten g))
  in
  check "float32" Nx.float32;
  check "float64" Nx.float64

(* The tangent is the range gather of the tangent. *)
let law_jvp c =
  let x = ints Nx.float64 (source c) 1 and t = ints Nx.float64 (source c) 5 in
  let _, dt = Rune.jvp' (ranges c) x t in
  equal (array float_exact) (Nx.to_array (ranges c t)) (Nx.to_array dt)

(* A map over three lanes, each operand a lane's or one every lane shares,
   is the lanes one by one, stacked. *)
let law_vmap (c, (bs, be, bx)) =
  cover "starts of each lane" bs;
  cover "ends of each lane" be;
  cover "a source of each lane" bx;
  cover "shared starts and ends, a source of each lane"
    ((not bs) && (not be) && bx);
  let lanes = 3 in
  let lane_runs a l =
    Array.mapi
      (fun i v ->
        if l = 0 then v else Int64.add v (Int64.of_int (((i + l) mod 3) - 1)))
      a
  in
  let s = Array.init lanes (fun l -> runs c (lane_runs c.starts l)) in
  let e = Array.init lanes (fun l -> runs c (lane_runs c.ends l)) in
  let x = Array.init lanes (fun l -> ints Nx.float64 (source c) l) in
  let one l =
    ranges
      ~starts_of:(fun _ -> if bs then s.(l) else s.(0))
      ~ends_of:(fun _ -> if be then e.(l) else e.(0))
      c
      (if bx then x.(l) else x.(0))
  in
  let want = Nx.stack (List.init lanes one) in
  let f sl el xl =
    ranges
      ~starts_of:(fun _ -> if bs then sl else s.(0))
      ~ends_of:(fun _ -> if be then el else e.(0))
      c
      (if bx then xl else x.(0))
  in
  let got =
    Rune.vmap
      T.(tensor @-> tensor @-> tensor @-> returns tensor)
      f
      (Nx.stack (Array.to_list s))
      (Nx.stack (Array.to_list e))
      (Nx.stack (Array.to_list x))
  in
  equal (array float_exact) (Nx.to_array want) (Nx.to_array got)

(* Values of every dtype: a sub-byte one's and a boolean's cast from small
   integers, any other's bytes, every third element a special one of its
   width, -0 then NaNs of payloads, and the rest from a cycle of bytes. *)
let specials = function
  | 2 -> [| 0x8000L; 0x7E01L; 0x7C01L |]
  | 4 -> [| 0x80000000L; 0x7FC00001L; 0x7F800001L |]
  | 8 -> [| 0x8000000000000000L; 0x7FF8000000000001L; 0x7FF0000000000002L |]
  | _ -> [| 0x80L; 0x7FL; 0xFFL |]

let any_of (type v s) (dt : (v, s) Nx.dtype) shape : (v, s, Nx.host) Nx.t =
  let n = numel shape and w = Nx.Dtype.bits dt / 8 in
  if Nx.Dtype.bits dt < 8 || Nx.Dtype.is Nx.Dtype.Boolean dt then
    Nx.cast dt
      (Nx.create Nx.int32 shape
         (Array.init n (fun i -> Int32.of_int (((i * 5) mod 7) - 3))))
  else
    let sp = specials (min w 8) in
    let byte i k =
      let e = i / w and b = i mod w in
      if e mod 3 = 0 && b < 8 && b < k then
        Int64.to_int
          (Int64.logand
             (Int64.shift_right_logical sp.(e / 3 mod 3) (8 * b))
             0xFFL)
      else ((i * 37) + 11) land 0xFF
    in
    let k = min w 8 in
    Nx.bitcast dt
      (Nx.create Nx.uint8 (Array.append shape [| w |])
         (Array.init (n * w) (fun i -> byte i k)))

(* An array's bits, or a sub-byte or boolean element's value. *)
let bits_of (type v s) (y : (v, s, Nx.host) Nx.t) =
  let dt = Nx.dtype y in
  if Nx.Dtype.bits dt < 8 || Nx.Dtype.is Nx.Dtype.Boolean dt then
    Nx.to_array (Nx.cast Nx.int32 y)
  else Array.map Int32.of_int (Nx.to_array (Nx.bitcast Nx.uint8 y))

let dtyped =
  Gen.with_pp
    (fun ppf (c, Nx.Dtype.Any dt) ->
      Format.fprintf ppf "%a, %s" pp_case c (Nx.Dtype.name dt))
    (Gen.pair case_gen (Gen.of_list Nx.Dtype.all))

(* A compiled call has eager's bits, computed by the compiler: a program
   holds the range gather as a gather at its positions, which it lowers. *)
let law_jit (c, Nx.Dtype.Any dt) =
  let float = Nx.Dtype.is Nx.Dtype.Float dt in
  cover "a position inside the axis only modulo 2^64" (wraps c);
  cover "int4" (Nx.Dtype.equal dt Nx.int4);
  cover "a boolean" (Nx.Dtype.equal dt Nx.bool);
  cover "bfloat16" (Nx.Dtype.equal dt Nx.bfloat16);
  cover "a float's -0 and NaN payloads" (float && numel (source c) >= 4);
  let x = any_of dt (source c) in
  let before = Rune.evaluations () in
  let y = Rune.jit' (ranges c) x in
  equal ~msg:"compiled" int 0 (Rune.evaluations () - before);
  equal (array int32) (bits_of (ranges c x)) (bits_of y)

(* A compiled gradient, which runs the range gather's transpose, is
   eager's, at each float dtype it takes: small integers, whose sums each
   holds exactly. *)
let grad_dtypes =
  Nx.Dtype.[ Any Nx.float16; Any Nx.bfloat16; Any Nx.float32; Any Nx.float64 ]

let law_jit_grad (c, Nx.Dtype.Any dt) =
  match Nx.Dtype.kind dt with
  | Float ->
      cover "a position inside the axis only modulo 2^64" (wraps c);
      cover "a 16-bit float" (Nx.Dtype.bits dt = 16);
      let x = ints dt (source c) 1 and w = ints dt (result c) 2 in
      let loss x = Nx.sum (Nx.mul w (ranges c x)) in
      let before = Rune.evaluations () in
      let g = Rune.jit' (Rune.grad' loss) x in
      equal ~msg:"compiled" int 0 (Rune.evaluations () - before);
      equal (array int32) (bits_of (Rune.grad' loss x)) (bits_of g)
  | _ -> fail "a gradient's dtype that is not a float"

let float_typed =
  Gen.with_pp
    (fun ppf (c, Nx.Dtype.Any dt) ->
      Format.fprintf ppf "%a, %s" pp_case c (Nx.Dtype.name dt))
    (Gen.pair case_gen (Gen.of_list grad_dtypes))

let flags = Gen.(triple bool bool bool)

let () =
  exit
    (run "rune ranges"
       [
         prop ~count:300 "a transpose satisfies the adjoint law" cases
           law_adjoint;
         prop ~count:300 "the gradient is its finite differences" cases
           law_grad;
         prop "the tangent is the range gather of the tangent" cases law_jvp;
         prop "a map is its lanes one by one"
           (Gen.with_pp
              (fun ppf (c, (a, b, d)) ->
                Format.fprintf ppf "%a, lanes %b %b %b" pp_case c a b d)
              (Gen.pair case_gen flags))
           law_vmap;
         prop ~count:300 "a compiled call has eager's bits, every dtype"
           dtyped law_jit;
         prop "a compiled gradient has eager's bits, every float dtype"
           float_typed law_jit_grad;
       ])
