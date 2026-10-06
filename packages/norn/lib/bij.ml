(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'f t = {
  name : string;
  forward : (float, 'f) Nx.t -> (float, 'f) Nx.t * (float, 'f) Nx.t;
  inverse : (float, 'f) Nx.t -> (float, 'f) Nx.t;
  shape : int array -> int array;
}

let forward b u = b.forward u
let inverse b x = b.inverse x
let shape b s = b.shape s
let pp ppf b = Format.pp_print_string ppf b.name

(* Helpers *)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let const x v = Nx.scalar (Nx.dtype x) v
let last x = Nx.ndim x - 1

(* [nudge x] is one or two units in the last place of [x], at least the smallest
   normal: [up x] and [down x] are the nearest values of an open interval with
   bound [x]. *)
let nudge x =
  let dt = Nx.dtype x in
  Nx.maximum (Nx.mul_s (Nx.abs x) (Prec.eps dt)) (const x (Prec.tiny dt))

let up x = Nx.add x (nudge x)
let down x = Nx.sub x (nudge x)

(* [saturate ok ld] is [ld] where [ok] holds and [-inf] elsewhere. *)
let saturate ok ld = Nx.where ok ld (const ld Float.neg_infinity)

let finite_within x =
  Nx.clamp ~min:(-.Prec.huge (Nx.dtype x)) ~max:(Prec.huge (Nx.dtype x)) x

let range axis (a, b) x =
  let s = Nx.shape x in
  Nx.shrink (Array.mapi (fun d n -> if d = axis then (a, b) else (0, n)) s) x

let sum_last k x =
  if k <= 0 then x
  else
    let n = Nx.ndim x in
    Nx.sum ~axes:(List.init k (fun i -> n - 1 - i)) x

let shape_with_last k f fn s =
  let n = Array.length s in
  if n < k then
    invalid_argf
      "Norn.Bij.shape: %s: a value of shape [%s] has fewer than %d axes" fn
      (String.concat "; " (Array.to_list (Array.map string_of_int s)))
      k;
  f (Array.sub s 0 (n - k)) (Array.sub s (n - k) k)

(* Elementwise *)

let identity =
  {
    name = "identity";
    forward = (fun u -> (u, Nx.zeros_like u));
    inverse = Fun.id;
    shape = Fun.id;
  }

let exp =
  let forward u =
    let dt = Nx.dtype u in
    let x = Nx.exp u in
    let ok =
      Nx.logical_and
        (Nx.greater_equal x (const x (Prec.tiny dt)))
        (Nx.less_equal x (const x (Prec.huge dt)))
    in
    (Nx.clamp ~min:(Prec.tiny dt) ~max:(Prec.huge dt) x, saturate ok u)
  in
  { name = "exp"; forward; inverse = Nx.log; shape = Fun.id }

let greater ~low =
  let forward u =
    let x = Nx.add low (Nx.exp u) in
    let above = Nx.greater x low in
    let ok = Nx.logical_and above (Nx.isfinite x) in
    let x = finite_within (Nx.where above x (up low)) in
    (x, saturate ok u)
  in
  let inverse x = Nx.log (Nx.sub x low) in
  { name = "greater"; forward; inverse; shape = Fun.id }

(* [softplus x] is [log (1 + exp x)], exact in both tails. *)
let softplus x =
  Nx.add (Nx.maximum x (const x 0.)) (Nx.log1p (Nx.exp (Nx.neg (Nx.abs x))))

let interval ~low ~high =
  let forward u =
    let width = Nx.sub high low in
    let x = Nx.add low (Nx.mul width (Nx.sigmoid u)) in
    let above = Nx.greater x low and below = Nx.less x high in
    let x = Nx.where above x (up low) in
    let x = Nx.where below x (down high) in
    (* log (high - low) + log sigmoid u + log sigmoid (-u) *)
    let ld =
      Nx.sub (Nx.log width) (Nx.add (softplus u) (softplus (Nx.neg u)))
    in
    (x, saturate (Nx.logical_and above below) ld)
  in
  let inverse x = Nx.sub (Nx.log (Nx.sub x low)) (Nx.log (Nx.sub high x)) in
  { name = "interval"; forward; inverse; shape = Fun.id }

let affine ~loc ~scale =
  let forward u =
    let x = Nx.add loc (Nx.mul scale u) in
    let ld = Nx.broadcast_to (Nx.shape x) (Nx.log (Nx.abs scale)) in
    (finite_within x, saturate (Nx.isfinite x) ld)
  in
  let inverse x = Nx.div (Nx.sub x loc) scale in
  { name = "affine"; forward; inverse; shape = Fun.id }

(* Sums to zero *)

(* The Helmert basis of the vectors of [n + 1] components summing to zero: basis
   vector [i], for [i] from 1 to [n], is [i] ones then [-i], over [sqrt (i (i +
   1))]. Coordinates [u] map to [sum_i u_i v_i], whose component [k] is the
   suffix sum [S (k + 1)] of [w_i = u_i / sqrt (i (i + 1))] minus [k w_k], so
   the map costs two cumulative sums. *)

let helmert_scales x n =
  let i = Nx.arange_f (Nx.dtype x) 1. (float_of_int (n + 1)) 1. in
  (i, Nx.rsqrt (Nx.mul i (Nx.add_s i 1.)))

let helmert u =
  let ax = last u in
  let n = (Nx.shape u).(ax) in
  let i, c = helmert_scales u n in
  let w = Nx.mul u c in
  let suffix =
    Nx.flip ~axes:[ ax ] (Nx.cumsum ~axis:ax (Nx.flip ~axes:[ ax ] w))
  in
  let pad before after x =
    Nx.pad
      (Array.init (Nx.ndim x) (fun d ->
           if d = ax then (before, after) else (0, 0)))
      0. x
  in
  Nx.sub (pad 0 1 suffix) (pad 1 0 (Nx.mul w i))

let helmert_inverse x =
  let ax = last x in
  let n = (Nx.shape x).(ax) - 1 in
  let i, c = helmert_scales x n in
  let prefix = range ax (0, n) (Nx.cumsum ~axis:ax x) in
  Nx.mul (Nx.sub prefix (Nx.mul i (range ax (1, n + 1) x))) c

(* [bounded u] is [u] clamped to [huge / 2K] in magnitude, [K] its last axis'
   length plus one, and whether each vector was within: so bounded, every
   component of [helmert u] is finite. *)
let bounded u =
  let ax = last u in
  let bound =
    Prec.huge (Nx.dtype u) /. (2. *. float_of_int ((Nx.shape u).(ax) + 1))
  in
  let ok = Nx.all ~axes:[ ax ] (Nx.less_equal (Nx.abs u) (const u bound)) in
  (Nx.clamp ~min:(-.bound) ~max:bound u, ok)

let one_shorter fn =
  shape_with_last 1
    (fun batch ev ->
      if ev.(0) = 0 then
        invalid_argf "Norn.Bij.shape: %s: a vector of no component" fn;
      Array.append batch [| ev.(0) - 1 |])
    fn

let sum_to_zero =
  let forward u =
    let u, ok = bounded u in
    let k = float_of_int ((Nx.shape u).(last u) + 1) in
    let batch = Array.sub (Nx.shape u) 0 (last u) in
    (* The map onto the first [K - 1] components has determinant [1 / sqrt
       K]. *)
    (helmert u, saturate ok (Nx.full (Nx.dtype u) batch (-0.5 *. Float.log k)))
  in
  {
    name = "sum_to_zero";
    forward;
    inverse = helmert_inverse;
    shape = (fun s -> one_shorter "sum_to_zero" s);
  }

let simplex =
  let forward u =
    let dt = Nx.dtype u in
    let ax = last u in
    let k = (Nx.shape u).(ax) + 1 in
    let u, within = bounded u in
    let lx = Nx.log_softmax ~axes:[ ax ] (helmert u) in
    let log_tiny = Float.log (Prec.tiny dt) in
    let ok =
      Nx.logical_and within
        (Nx.all ~axes:[ ax ] (Nx.greater_equal lx (const lx log_tiny)))
    in
    let x = Nx.exp (Nx.maximum lx (const lx log_tiny)) in
    (* The Jacobian of the first [K - 1] components is [prod x] times the
       determinant [sqrt K] of the basis read in log-ratios to the last. *)
    let ld =
      Nx.add_s (Nx.sum ~axes:[ ax ] lx) (0.5 *. Float.log (float_of_int k))
    in
    (x, saturate ok ld)
  in
  {
    name = "simplex";
    forward;
    inverse = (fun x -> helmert_inverse (Nx.log x));
    shape = (fun s -> one_shorter "simplex" s);
  }

(* Ordered *)

let ordered =
  (* Each increment is capped at [huge / 2K] and the first component at [huge /
     2], so no sum overflows; each component is at least one unit in the last
     place above the previous one, so the vector strictly increases. *)
  let forward u =
    let dt = Nx.dtype u in
    let ax = last u in
    let k = (Nx.shape u).(ax) in
    let huge = Prec.huge dt in
    let at j = range ax (j, j + 1) u in
    let cap = huge /. (2. *. float_of_int (max k 1)) in
    let first = at 0 in
    let ok = Nx.less_equal (Nx.abs first) (const first (huge /. 2.)) in
    let first = Nx.clamp ~min:(-.huge /. 2.) ~max:(huge /. 2.) first in
    let rec go j prev acc ok =
      if j = k then (List.rev acc, ok)
      else
        let e = Nx.exp (at j) in
        let next = Nx.add prev (Nx.minimum e (const e cap)) in
        let ok =
          Nx.logical_and ok
            (Nx.logical_and (Nx.greater next prev)
               (Nx.less_equal e (const e cap)))
        in
        let next = Nx.maximum next (up prev) in
        go (j + 1) next (next :: acc) ok
    in
    let xs, ok = go 1 first [ first ] ok in
    let x = Nx.concatenate ~axis:ax xs in
    let ld = Nx.sum ~axes:[ ax ] (range ax (1, k) u) in
    (x, saturate (Nx.squeeze ~axes:[ ax ] ok) ld)
  in
  let inverse x =
    let ax = last x in
    let k = (Nx.shape x).(ax) in
    let gaps = Nx.log (Nx.sub (range ax (1, k) x) (range ax (0, k - 1) x)) in
    Nx.concatenate ~axis:ax [ range ax (0, 1) x; gaps ]
  in
  let shape s =
    shape_with_last 1
      (fun batch ev ->
        if ev.(0) = 0 then
          invalid_arg "Norn.Bij.shape: ordered: a vector of no component";
        Array.append batch ev)
      "ordered" s
  in
  { name = "ordered"; forward; inverse; shape }

(* Cholesky factors of correlations *)

(* Row [i] of the factor is a stick-breaking of the unit sphere: with [z] the
   row's canonical partial correlations, [L(i,j) = z(i,j) sqrt (prod_(k<j) (1 -
   z(i,k)^2))] below the diagonal and [L(i,i) = sqrt (prod_(k<i) (1 -
   z(i,k)^2))]. The products are cumulative sums of logarithms. *)

let corr_dim m =
  let n = (1 + int_of_float (Float.sqrt (float_of_int (1 + (8 * m))))) / 2 in
  if n * (n - 1) / 2 <> m then
    invalid_argf
      "Norn.Bij.forward: cholesky_corr: %d coordinates is no n (n - 1) / 2" m;
  n

(* [lower_indices n] is, for each element of an [n × n] matrix in C order, its
   index in the strict lower triangle read row by row, or [n (n - 1) / 2], which
   [Nx.take] reads as zero, off it. *)
let lower_indices n =
  let m = n * (n - 1) / 2 in
  let idx = Array.make (n * n) (Int64.of_int m) in
  let k = ref 0 in
  for i = 0 to n - 1 do
    for j = 0 to i - 1 do
      idx.((i * n) + j) <- Int64.of_int !k;
      incr k
    done
  done;
  Nx.create Nx.int64 [| n * n |] idx

(* [lower_positions n] is the C-order position of each element of the strict
   lower triangle, read row by row. *)
let lower_positions n =
  let pos = ref [] in
  for i = 0 to n - 1 do
    for j = 0 to i - 1 do
      pos := Int64.of_int ((i * n) + j) :: !pos
    done
  done;
  let pos = Array.of_list (List.rev !pos) in
  Nx.create Nx.int64 [| Array.length pos |] pos

let cholesky_corr =
  let forward u =
    let dt = Nx.dtype u in
    let ax = last u in
    let shape = Nx.shape u in
    let n = corr_dim shape.(ax) in
    let batch = Array.sub shape 0 ax in
    let edge = 1. -. Prec.eps dt in
    let z = Nx.tanh u in
    let ok = Nx.all ~axes:[ ax ] (Nx.less_equal (Nx.abs z) (const z edge)) in
    let z = Nx.clamp ~min:(-.edge) ~max:edge z in
    let z =
      Nx.reshape
        (Array.append batch [| n; n |])
        (Nx.take ~axis:ax ~indices:(lower_indices n) z)
    in
    let q = Nx.log1p (Nx.neg (Nx.square z)) in
    let s = Nx.cumsum ~axis:(ax + 1) q in
    let excl = Nx.sub s q in
    let diag = Nx.exp (Nx.mul_s s 0.5) in
    let l =
      Nx.add (Nx.mul z (Nx.exp (Nx.mul_s excl 0.5))) (Nx.mul (Nx.eye dt n) diag)
    in
    let strict = Nx.tril ~k:(-1) (Nx.ones dt [| n; n |]) in
    let ld =
      Nx.sum ~axes:[ ax; ax + 1 ] (Nx.mul strict (Nx.add q (Nx.mul_s excl 0.5)))
    in
    (* Along a row the cumulative products only shrink, so every element of
       [diag] is at least its row's diagonal. *)
    let diag_ok =
      Nx.all
        ~axes:[ ax; ax + 1 ]
        (Nx.greater_equal diag (const diag (Prec.tiny dt)))
    in
    (l, saturate (Nx.logical_and ok diag_ok) ld)
  in
  let inverse l =
    let ax = last l in
    let shape = Nx.shape l in
    let n = shape.(ax) in
    let batch = Array.sub shape 0 (ax - 1) in
    let sq = Nx.square l in
    let excl = Nx.sub (Nx.cumsum ~axis:ax sq) sq in
    let z = Nx.div l (Nx.sqrt (Nx.sub (const l 1.) excl)) in
    let z = Nx.reshape (Array.append batch [| n * n |]) z in
    Nx.atanh (Nx.take ~axis:(ax - 1) ~indices:(lower_positions n) z)
  in
  let shape s =
    shape_with_last 2
      (fun batch ev ->
        if ev.(0) <> ev.(1) then
          invalid_argf
            "Norn.Bij.shape: cholesky_corr: a %d x %d matrix is not square"
            ev.(0) ev.(1);
        Array.append batch [| ev.(0) * (ev.(0) - 1) / 2 |])
      "cholesky_corr" s
  in
  { name = "cholesky_corr"; forward; inverse; shape }

(* Composition *)

let compose b c =
  (* The log-determinant with the finer units is summed to the other's. *)
  let forward u =
    let v, ldc = c.forward u in
    let x, ldb = b.forward v in
    let k = Nx.ndim ldc - Nx.ndim ldb in
    (x, Nx.add (sum_last k ldc) (sum_last (-k) ldb))
  in
  {
    name = Printf.sprintf "compose(%s, %s)" b.name c.name;
    forward;
    inverse = (fun x -> c.inverse (b.inverse x));
    shape = (fun s -> c.shape (b.shape s));
  }
