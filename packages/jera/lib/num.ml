(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let eps dtype = (Nx.Dtype.float_format dtype).epsilon
let tiny dtype = (Nx.Dtype.float_format dtype).min_normal
let huge dtype = (Nx.Dtype.float_format dtype).max_finite
let precision dtype = (Nx.Dtype.float_format dtype).fraction_bits + 1
let bits dtype = Nx.Dtype.bits dtype
let constant at dtype a = Nx.place at (Nx.create dtype [| Array.length a |] a)

let on_float (type a c) fn (f : (float, c, 'd) Nx.t -> (float, c, 'd) Nx.t)
    (x : (a, c, 'd) Nx.t) : (a, c, 'd) Nx.t =
  match Nx.Dtype.kind (Nx.dtype x) with
  | Float -> f x
  | _ ->
      invalid_arg
        (Printf.sprintf "%s: a %s leaf; the leaves must be float tensors" fn
           (Nx.Dtype.name (Nx.dtype x)))

let shape s =
  "[" ^ String.concat "," (Array.to_list (Array.map string_of_int s)) ^ "]"

let check_increasing fn what x =
  let n = Nx.dim 0 x in
  if n > 1 then
    let lo = Nx.slice [ Nx.R (0, n - 1) ] x
    and hi = Nx.slice [ Nx.R (1, n) ] x in
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
let i4 = { bits = Nx.int4; min = -8; one = 1 }

let ordered_midpoint (type b) (a : (float, b, 'd) Nx.t)
    (b : (float, b, 'd) Nx.t) : (float, b, 'd) Nx.t =
  match Nx.dtype a with
  | Nx.Dtype.Float64 -> mid i64 Nx.float64 a b
  | Nx.Dtype.Float32 -> mid i32 Nx.float32 a b
  | Nx.Dtype.Float16 -> mid i16 Nx.float16 a b
  | Nx.Dtype.Bfloat16 -> mid i16 Nx.bfloat16 a b
  | Nx.Dtype.Float8_e4m3fn -> mid i8 Nx.float8_e4m3fn a b
  | Nx.Dtype.Float8_e5m2 -> mid i8 Nx.float8_e5m2 a b
  | Nx.Dtype.Float4_e2m1fn -> mid i4 Nx.float4_e2m1fn a b

let adjacent (type b) (a : (float, b, 'd) Nx.t) (b : (float, b, 'd) Nx.t) =
  match Nx.dtype a with
  | Nx.Dtype.Float64 -> next i64 a b
  | Nx.Dtype.Float32 -> next i32 a b
  | Nx.Dtype.Float16 | Nx.Dtype.Bfloat16 -> next i16 a b
  | Nx.Dtype.Float8_e4m3fn | Nx.Dtype.Float8_e5m2 -> next i8 a b
  | Nx.Dtype.Float4_e2m1fn -> next i4 a b

let sum_rows r =
  let k = Nx.dim 0 r and n = Nx.dim 1 r in
  if n = 0 then Nx.sum ~axes:[ 1 ] r
  else
    let rec pow2 m = if m >= n then m else pow2 (2 * m) in
    let m = pow2 1 in
    let r =
      if m = n then r
      else
        Nx.concatenate ~axis:1
          [ r; Nx.zeros_at (Nx.beside r) (Nx.dtype r) [| k; m - n |] ]
    in
    (* Neighbours added by sums over axes of two: a sum of two terms has one
       association, so each level rounds alike eagerly and compiled, and each
       level is a reduction, which a compiled program computes once rather than
       inlining its producers into every term. *)
    let rec halve v w =
      if w = 1 then Nx.reshape [| k |] v
      else halve (Nx.sum ~axes:[ 2 ] (Nx.reshape [| k; w / 2; 2 |] v)) (w / 2)
    in
    halve r m

let rms_rows r =
  let n = Nx.dim 1 r in
  if n = 0 then Nx.sum ~axes:[ 1 ] r
  else Nx.sqrt (Nx.div_s (sum_rows (Nx.square r)) (float n))

