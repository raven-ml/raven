(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let eps (type b) (dtype : (float, b) Nx.dtype) =
  match dtype with
  | Nx.Float64 -> epsilon_float
  | Nx.Float32 -> Float.ldexp 1. (-23)
  | Nx.Float16 -> Float.ldexp 1. (-10)
  | Nx.BFloat16 -> Float.ldexp 1. (-7)
  | Nx.Float8_e4m3 -> Float.ldexp 1. (-3)
  | Nx.Float8_e5m2 -> Float.ldexp 1. (-2)

let tiny (type b) (dtype : (float, b) Nx.dtype) =
  match dtype with
  | Nx.Float64 -> Float.min_float
  | Nx.Float32 | Nx.BFloat16 -> Float.ldexp 1. (-126)
  | Nx.Float16 -> Float.ldexp 1. (-14)
  | Nx.Float8_e4m3 -> Float.ldexp 1. (-6)
  | Nx.Float8_e5m2 -> Float.ldexp 1. (-14)

let huge (type b) (dtype : (float, b) Nx.dtype) =
  match dtype with
  | Nx.Float64 -> Float.max_float
  | Nx.Float32 -> Float.ldexp (2. -. Float.ldexp 1. (-23)) 127
  | Nx.BFloat16 -> Float.ldexp (2. -. Float.ldexp 1. (-7)) 127
  | Nx.Float16 -> 65504.
  | Nx.Float8_e4m3 -> 448.
  | Nx.Float8_e5m2 -> 57344.

let precision dtype = 1 - Float.to_int (Float.log2 (eps dtype))

let bits (type b) (dtype : (float, b) Nx.dtype) =
  match dtype with
  | Nx.Float64 -> 64
  | Nx.Float32 -> 32
  | Nx.Float16 | Nx.BFloat16 -> 16
  | Nx.Float8_e4m3 | Nx.Float8_e5m2 -> 8

let constant dtype a = Nx.create dtype [| Array.length a |] a

type map = { f : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t }

let on_float (type a c) fn m (x : (a, c) Nx.t) : (a, c) Nx.t =
  match Nx.dtype x with
  | Nx.Float64 -> m.f x
  | Nx.Float32 -> m.f x
  | Nx.Float16 -> m.f x
  | Nx.BFloat16 -> m.f x
  | Nx.Float8_e4m3 -> m.f x
  | Nx.Float8_e5m2 -> m.f x
  | dtype ->
      invalid_arg
        (Printf.sprintf "%s: a %s leaf; the leaves must be float tensors" fn
           (Nx_dtype.to_string dtype))

let shape s =
  "[" ^ String.concat "," (Array.to_list (Array.map string_of_int s)) ^ "]"

let check_increasing fn what x =
  let n = Nx.dim 0 x in
  if n > 1 then
    let lo = Nx.shrink [| (0, n - 1) |] x and hi = Nx.shrink [| (1, n) |] x in
    Nx.check
      Nx.Ptree.(pair tensor tensor)
      (Nx.less lo hi) (lo, hi)
      (fun i (lo, hi) ->
        Invalid_argument
          (Printf.sprintf
             "%s: %s are not strictly increasing at [%d]: %g after %g" fn what
             (i.(0) + 1)
             (Nx.item [] hi) (Nx.item [] lo)))

(* A float's ordered-integer image: its bits read as a signed integer of its
   width, with negative floats' magnitudes negated, so that the images of [a <
   b] are [ka < kb] and of [−0.] and [0.] both [0]. *)
type ('i, 'j) ordered = { bits : ('i, 'j) Nx.dtype; min : 'i; one : 'i }

let key o x =
  let b = Nx.bitcast o.bits x in
  Nx.where
    (Nx.greater_equal b (Nx.zeros_like b))
    b
    (Nx.sub (Nx.full_like b o.min) b)

let of_key o dtype k =
  let b =
    Nx.where
      (Nx.greater_equal k (Nx.zeros_like k))
      k
      (Nx.sub (Nx.full_like k o.min) k)
  in
  Nx.bitcast dtype b

let mid o dtype a b =
  let ka = key o a and kb = key o b in
  (* ⌊(ka + kb) / 2⌋ without overflow. *)
  let m =
    Nx.add
      (Nx.add (Nx.rshift ka 1) (Nx.rshift kb 1))
      (Nx.bitwise_and (Nx.bitwise_and ka kb) (Nx.full_like ka o.one))
  in
  of_key o dtype m

let next o a b =
  let ka = key o a and kb = key o b in
  Nx.less_equal kb (Nx.add ka (Nx.full_like ka o.one))

let i64 = { bits = Nx.int64; min = Int64.min_int; one = 1L }
let i32 = { bits = Nx.int32; min = Int32.min_int; one = 1l }
let i16 = { bits = Nx.int16; min = -32768; one = 1 }
let i8 = { bits = Nx.int8; min = -128; one = 1 }

let ordered_midpoint (type b) (a : (float, b) Nx.t) (b : (float, b) Nx.t) :
    (float, b) Nx.t =
  match Nx.dtype a with
  | Nx.Float64 -> mid i64 Nx.float64 a b
  | Nx.Float32 -> mid i32 Nx.float32 a b
  | Nx.Float16 -> mid i16 Nx.float16 a b
  | Nx.BFloat16 -> mid i16 Nx.bfloat16 a b
  | Nx.Float8_e4m3 -> mid i8 Nx.float8_e4m3 a b
  | Nx.Float8_e5m2 -> mid i8 Nx.float8_e5m2 a b

let adjacent (type b) (a : (float, b) Nx.t) (b : (float, b) Nx.t) =
  match Nx.dtype a with
  | Nx.Float64 -> next i64 a b
  | Nx.Float32 -> next i32 a b
  | Nx.Float16 | Nx.BFloat16 -> next i16 a b
  | Nx.Float8_e4m3 | Nx.Float8_e5m2 -> next i8 a b

let rms_rows r =
  let k = Nx.dim 0 r and n = Nx.dim 1 r in
  if n = 0 then Nx.zeros (Nx.dtype r) [| k |]
  else
    let rec pow2 m = if m >= n then m else pow2 (2 * m) in
    let m = pow2 1 in
    let sq = Nx.square r in
    let sq =
      if m = n then sq
      else Nx.concatenate ~axis:1 [ sq; Nx.zeros (Nx.dtype r) [| k; m - n |] ]
    in
    (* Neighbours added by sums over axes of two: a sum of two terms has one
       association, so each level rounds alike eagerly and compiled, and each
       level is a reduction, which a compiled program computes once rather than
       inlining its producers into every term. *)
    let rec halve v w =
      if w = 1 then Nx.reshape [| k |] v
      else halve (Nx.sum ~axes:[ 2 ] (Nx.reshape [| k; w / 2; 2 |] v)) (w / 2)
    in
    Nx.sqrt (Nx.div_s (halve sq m) (float n))
