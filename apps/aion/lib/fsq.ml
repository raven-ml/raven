(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { levels : int list; [@ptree.int] } [@@deriving ptree]

(* The bound's [ε], upstream's FiniteScalarQuantizer default, which both of
   AION-1's codecs keep. *)
let eps = 1e-3
let invalid fmt = Printf.ksprintf invalid_arg fmt

let v levels =
  if levels = [] then invalid "Aion.Fsq.v: no level";
  let step b l =
    if l < 2 then invalid "Aion.Fsq.v: level %d, at least 2 expected" l;
    if b > max_int / l then invalid "Aion.Fsq.v: more codes than max_int";
    b * l
  in
  ignore (List.fold_left step 1 levels);
  { levels }

let levels q = q.levels
let size q = List.fold_left ( * ) 1 q.levels
let offset l = if l mod 2 = 0 then 0.5 else 0.
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* [h] and [s] as upstream's FiniteScalarQuantizer computes them in PyTorch's
   default dtype: [(L - 1) * (1 + eps) / 2] multiplies the integer by [1 + eps]
   rounded to that dtype, rounds, then halves, and [s] is [atanh] of [o / h]
   rounded. In float32 [h] for [L = 7] is 0x1.806250p+1, two floats above 3.003
   rounded. Float64 holds AION-1's float64 reference run's; a narrower dtype
   takes the float32 constants, rounded. *)
let constants (type s) (dt : (float, s) Nx.dtype) l =
  let h, s =
    match dt with
    | Nx.Dtype.Float64 ->
        let h = Float.of_int (l - 1) *. (1. +. eps) /. 2. in
        (h, Float.atanh (offset l /. h))
    | _ ->
        let h = f32 (Float.of_int (l - 1) *. f32 (1. +. eps)) /. 2. in
        (h, f32 (Float.atanh (f32 (offset l /. h))))
  in
  (h, s)

(* [B(i)], the product of the levels before [i], on [x]'s set. *)
let basis x q =
  let step (b, bs) l = (b * l, Int64.of_int b :: bs) in
  let bs = List.rev (snd (List.fold_left step (1, []) q.levels)) in
  let bs = Array.of_list bs in
  Nx.place (Nx.beside x) (Nx.create Nx.int64 [| Array.length bs |] bs)

(* The vector of [f] of each level, on [x]'s set. *)
let per_level dt x q f =
  let a = Array.of_list (List.map f q.levels) in
  Nx.place (Nx.beside x) (Nx.create dt [| Array.length a |] a)

let check_last fn q x =
  let s = Nx.shape x and d = List.length q.levels in
  let r = Array.length s in
  if r = 0 || s.(r - 1) <> d then
    invalid "Aion.%s: last axis of %d entries, %d expected" fn
      (if r = 0 then 0 else s.(r - 1))
      d

let bound_in fn q z =
  check_last fn q z;
  let dt = Nx.dtype z in
  let h = per_level dt z q (fun l -> fst (constants dt l)) in
  let o = per_level dt z q offset in
  let s = per_level dt z q (fun l -> snd (constants dt l)) in
  Nx.sub (Nx.mul (Nx.tanh (Nx.add z s)) h) o

let bound q z = bound_in "Fsq.bound" q z

let encode q z =
  let y = bound_in "Fsq.encode" q z in
  Nx.check Nx.Ptree.unit
    (Nx.logical_not (Nx.isnan y))
    ()
    (fun i () ->
      Invalid_argument
        (Format.asprintf "Aion.Fsq.encode: NaN at %a" Nx.pp_shape i));
  (* Rounded to integers, the bounded numbers convert to int64 exactly. *)
  let digit =
    Nx.add
      (Nx.cast Nx.int64 (Nx.roundeven y))
      (per_level Nx.int64 y q (fun l -> Int64.of_int (l / 2)))
  in
  Nx.sum ~axes:[ -1 ] (Nx.mul digit (basis y q))

let decode dt q k =
  let size = size q in
  let inside =
    Nx.logical_and (Nx.greater_equal_s k 0L) (Nx.less_s k (Int64.of_int size))
  in
  Nx.check Nx.Ptree.tensor inside k (fun i k ->
      Invalid_argument
        (Format.asprintf "Aion.Fsq.decode: code %Ld at %a outside [0, %d)"
           (Nx.item [] k) Nx.pp_shape i size));
  let k = Nx.unsqueeze ~axes:[ Nx.ndim k ] k in
  (* Codes and their quotients are not negative, so [div] and [mod_], which
     truncate, are floor division and its remainder. *)
  let digit =
    Nx.mod_ (Nx.div k (basis k q))
      (per_level Nx.int64 k q (fun l -> Int64.of_int l))
  in
  let w = per_level dt k q (fun l -> Float.of_int (l / 2)) in
  Nx.div (Nx.sub (Nx.cast dt digit) w) w
