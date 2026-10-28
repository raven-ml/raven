(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Histograms *)

type 'd bins = {
  x : 'd Nx.float64_t;
  x2 : 'd Nx.float64_t;
  count : 'd Nx.float64_t;
  density : 'd Nx.float64_t;
}

(* Sturges' rule: 1 + ceil (log2 n), in integers. *)
let sturges n =
  let rec log2_ceil k p = if p >= n then k else log2_ceil (k + 1) (2 * p) in
  if n <= 1 then 1 else 1 + log2_ceil 0 1

(* Each of [n] bins spans at least this many float spacings at the span's ends.
   An edge, computed as in [edges], is within 1.5 spacings of its exact value,
   so the edges of bins this wide stay distinct. *)
let bin_spacings = 4.

(* [lo] and [hi] moved apart by the same amount on either side to the width [w]
   of [n] bins, and shifted to stay within the finite floats. Equal ends widen
   to at least [1.]. *)
let widen n lo hi =
  let c = Nx.scalar_like lo in
  let m = Nx.maximum (Nx.abs lo) (Nx.abs hi) in
  let w = Nx.mul_s m (bin_spacings *. Float.of_int n *. Float.epsilon) in
  let w = Nx.where (Nx.equal lo hi) (Nx.maximum w (c 1.)) w in
  let d = Nx.div_s (Nx.maximum (c 0.) (Nx.sub w (Nx.sub hi lo))) 2. in
  let past room = Nx.maximum (c 0.) (Nx.sub d room) in
  let past_lo = past (Nx.add_s lo Float.max_float)
  and past_hi = past (Nx.sub (c Float.max_float) hi) in
  ( Nx.sub lo (Nx.sub (Nx.add d past_hi) past_lo),
    Nx.add hi (Nx.sub (Nx.add d past_lo) past_hi) )

(* The least and greatest finite values of [v], or [0.] and [1.] if there are
   none, widened for [n] bins, as 0-d values beside [v]. *)
let span n v =
  let c = Nx.scalar_like v in
  if Nx.numel v = 0 then widen n (c 0.) (c 1.)
  else
    let finite = Nx.isfinite v in
    let lo = Nx.min (Nx.where finite v (c Float.infinity)) in
    let hi = Nx.max (Nx.where finite v (c Float.neg_infinity)) in
    let none = Nx.greater lo hi in
    widen n (Nx.where none (c 0.) lo) (Nx.where none (c 1.) hi)

(* The float64 values [0.], [1.], ..., [float (n - 1)] beside [v]. *)
let range v n =
  Nx.arange_at (Nx.beside v) Nx.float64 0 n 1

(* The [n + 1] edges from [lo] to [hi], as [lo (1 - t) + hi t]: the ends are
   exactly [lo] and [hi], and no term overflows where [hi - lo] would. *)
let edges n (lo, hi) =
  let t = Nx.div_s (range lo (n + 1)) (Float.of_int n) in
  Nx.add (Nx.mul lo (Nx.sub (Nx.scalar_like lo 1.) t)) (Nx.mul hi t)

let histogram (type a b d) ?bins (v : (a, b, d) Nx.t) =
  (match Nx.Dtype.kind (Nx.dtype v) with
  | Complex | Boolean ->
      invalid_arg
        (Format.asprintf "Stats.histogram: values of dtype %a" Nx.pp_dtype
           (Nx.dtype v))
  | Float | Signed | Unsigned -> ());
  let shape = Nx.shape v in
  let rank = Array.length shape in
  if rank = 0 then invalid_arg "Stats.histogram: a scalar has no axis";
  let len = shape.(rank - 1) in
  let n = match bins with None -> sturges len | Some n -> n in
  if n < 1 then invalid_arg (strf "Stats.histogram: %d bins" n);
  let v = Nx.cast Nx.float64 v in
  let e = edges n (span n v) in
  let x = Nx.slice [ Nx.R (0, n) ] e and x2 = Nx.slice [ Nx.R (1, n + 1) ] e in
  let groups = Array.fold_left ( * ) 1 (Array.sub shape 0 (rank - 1)) in
  let out = Array.copy shape in
  out.(rank - 1) <- n;
  if groups = 0 then
    let zeros = Nx.broadcast_to out (Nx.scalar_like v 0.) in
    { x; x2; count = zeros; density = zeros }
  else
    (* One histogram in two dimensions: the group of a value, as a float
       coordinate binned by the edges 0, 1, ..., groups, and the value. *)
    let v = Nx.reshape [| groups; len |] v in
    let group =
      Nx.broadcast_to [| groups; len |]
        (Nx.reshape [| groups; 1 |] (range v groups))
    in
    let count = Nx.histogram [ (range v (groups + 1), group); (e, v) ] in
    let total = Nx.sum ~axes:[ 1 ] ~keepdims:true count in
    let density = Nx.div (Nx.div count total) (Nx.sub x2 x) in
    { x; x2; count = Nx.reshape out count; density = Nx.reshape out density }
