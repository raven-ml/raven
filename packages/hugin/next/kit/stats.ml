(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Histograms *)

type bins = {
  x : Nx.float64_t;
  x2 : Nx.float64_t;
  count : Nx.float64_t;
  density : Nx.float64_t;
}

(* Sturges' rule: 1 + ceil (log2 n), in integers. *)
let sturges n =
  let rec log2_ceil k p = if p >= n then k else log2_ceil (k + 1) (2 * p) in
  if n <= 1 then 1 else 1 + log2_ceil 0 1

let f64 = Nx.scalar Nx.float64

(* The least and greatest finite values of [v], as device scalars. All equal
   values [a] widen to a +/- h so that the bins have a width, and no finite
   value to [0.] and [1.]. *)
let span v =
  if Nx.numel v = 0 then (f64 0., f64 1.)
  else
    let finite = Nx.isfinite v in
    let lo = Nx.min (Nx.where finite v (f64 Float.infinity)) in
    let hi = Nx.max (Nx.where finite v (f64 Float.neg_infinity)) in
    let none = Nx.greater lo hi and same = Nx.equal lo hi in
    let h = Nx.maximum (f64 0.5) (Nx.mul_s (Nx.abs lo) Float.epsilon) in
    let lo' = Nx.maximum (Nx.sub lo h) (f64 (-.Float.max_float)) in
    let hi' = Nx.minimum (Nx.add hi h) (f64 Float.max_float) in
    ( Nx.where none (f64 0.) (Nx.where same lo' lo),
      Nx.where none (f64 1.) (Nx.where same hi' hi) )

(* The [n + 1] edges from [lo] to [hi], as [lo (1 - t) + hi t]: the ends are
   exactly [lo] and [hi], and no term overflows where [hi - lo] would. *)
let edges n (lo, hi) =
  let t = Nx.div_s (Nx.arange Nx.float64 0 (n + 1) 1) (Float.of_int n) in
  Nx.add (Nx.mul lo (Nx.sub (f64 1.) t)) (Nx.mul hi t)

let histogram (type a b) ?bins (v : (a, b) Nx.t) =
  (match Nx.dtype v with
  | Nx.Complex64 | Nx.Complex128 | Nx.Bool ->
      invalid_arg
        (Format.asprintf "Stats.histogram: values of dtype %a" Nx.pp_dtype
           (Nx.dtype v))
  | _ -> ());
  let shape = Nx.shape v in
  let rank = Array.length shape in
  if rank = 0 then invalid_arg "Stats.histogram: a scalar has no axis";
  let len = shape.(rank - 1) in
  let n = match bins with None -> sturges len | Some n -> n in
  if n < 1 then invalid_arg (strf "Stats.histogram: %d bins" n);
  let v = Nx.cast Nx.float64 v in
  let e = edges n (span v) in
  let x = Nx.slice [ Nx.R (0, n) ] e and x2 = Nx.slice [ Nx.R (1, n + 1) ] e in
  let groups = Array.fold_left ( * ) 1 (Array.sub shape 0 (rank - 1)) in
  let out = Array.copy shape in
  out.(rank - 1) <- n;
  if groups = 0 then
    let zeros = Nx.zeros Nx.float64 out in
    { x; x2; count = zeros; density = zeros }
  else
    (* One histogram in two dimensions: the group of a value, as a float
       coordinate binned by the edges 0, 1, ..., groups, and the value. *)
    let v = Nx.reshape [| groups; len |] v in
    let group =
      Nx.broadcast_to [| groups; len |]
        (Nx.reshape [| groups; 1 |] (Nx.arange Nx.float64 0 groups 1))
    in
    let count =
      Nx.histogram [ (Nx.arange Nx.float64 0 (groups + 1) 1, group); (e, v) ]
    in
    let total = Nx.sum ~axes:[ 1 ] ~keepdims:true count in
    let density = Nx.div (Nx.div count total) (Nx.sub x2 x) in
    { x; x2; count = Nx.reshape out count; density = Nx.reshape out density }
