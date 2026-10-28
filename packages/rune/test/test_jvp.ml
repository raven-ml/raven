(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jvp and its forms: tangents and second derivatives against central
   differences at float64 for every operation, complex functions as maps of
   their parts; the values the kinks' rules and the derivatives' definitions
   give; the gauges factorisations fix; the interface and its errors; jvp of a
   compiled function at every dtype; a map's tangent in one kernel. *)

open Windtrap
open Rune_support
open Cases
open Witness

let numel = Operand.numel
let finite a = Array.for_all Float.is_finite a

(* A direction in [-1, 1] for [n] elements, spread from [seed]. *)
let spread seed n =
  Array.init n (fun k -> Float.sin ((Float.of_int k *. 2.3) +. seed))

(* Whether a derivative is a value of its own: C order from the start of its
   memory. *)
let fresh ~msg x =
  let layout = (Nx.Prim.form x).layout in
  equal ~msg (pair bool int) (true, 0)
    (L.is_contiguous layout, fst (L.span layout))

(* Central differences *)

let h = 1e-6

(* [d], the derivative of [g] at [x] along [t], is within [1e-5 (1 + |fd|)] of
   the central difference [fd] of step [h], wherever [g] is finite at both
   steps. *)
let differences ~msg g x t d =
  let s = Nx.shape x and xs = Nx.to_array x and ts = Nx.to_array t in
  let at e =
    Nx.to_array
      (g (Nx.create f64 s (Array.mapi (fun k x -> x +. (e *. ts.(k))) xs)))
  in
  let plus = at h and minus = at (-.h) in
  let d = Nx.to_array d in
  equal ~msg:(msg ^ "s' count") int (Array.length plus) (Array.length d);
  Array.iteri
    (fun k d ->
      let fd = (plus.(k) -. minus.(k)) /. (2. *. h) in
      if finite [| plus.(k); minus.(k); fd |] then
        less
          ~msg:(Printf.sprintf "%s %d: %h, differences %h" msg k d fd)
          float_exact
          ~than:(1e-5 *. (1. +. Float.abs fd))
          (Float.abs (d -. fd)))
    d

let law c (s, xs, ts) =
  if c.empty then cover "a zero-size shape" (numel s = 0);
  cover "a shape of several elements" (numel s > 1);
  let x = Nx.create f64 s xs and t = Nx.create f64 s ts in
  let y, dy = Rune.jvp' c.f x t in
  equal ~msg:"primal" bits_t (bits (c.f x)) (bits y);
  equal ~msg:"tangent's shape" (array int) (Nx.shape y) (Nx.shape dy);
  fresh ~msg:"the tangent is fresh" dy;
  differences ~msg:"tangent" c.f x t dy

(* The tangent along [t] of the tangent along [u] is the central difference
   along [t] of the tangent along [u]: mixed second derivatives, so that a
   derivative's own derivative is the function's. *)
let second_law c (s, xs, ts) =
  if c.empty then cover "a zero-size shape" (numel s = 0);
  cover "a shape of several elements" (numel s > 1);
  let x = Nx.create f64 s xs and t = Nx.create f64 s ts in
  let u = Nx.create f64 s (spread (Array.fold_left ( +. ) 0.5 ts) (numel s)) in
  let along_u x = snd (Rune.jvp' c.f x u) in
  let _, d2 = Rune.jvp' along_u x t in
  differences ~msg:"second derivative" along_u x t d2

let every =
  real_cases @ contract_cases @ factor_cases @ complex_cases
  @ complex_factor_cases

let tangent_props = List.map (fun c -> prop c.name (gen c) (law c)) every
let second_props = List.map (fun c -> prop c.name (gen c) (second_law c)) every

(* Values the kinks' rules give *)

let two () = T.(pair tensor tensor)
let along f x t = Nx.to_array (Nx.cast f64 (snd (Rune.jvp' f x t)))

let along2 f (x, y) (tx, ty) =
  Nx.to_array (snd (Rune.jvp (two ()) T.tensor f (x, y) (tx, ty)))

type row = {
  name : string;
  rel : float; (* The relative tolerance. *)
  expected : float array;
  tangent : unit -> float array;
}

let row ?(rel = 1e-12) name expected tangent = { name; rel; expected; tangent }

let check_row r =
  equal (array (float_rel ~rel:r.rel ~abs:0.)) r.expected (r.tangent ())

let nan = Float.nan
let inf = Float.infinity
let float16 xs = Nx.cast Nx.float16 (v xs)
let one_of x = Nx.reshape [| 1 |] x

(* [x] reduced along axis 0 into the dtype [reduction] names, as no function of
   nx's reduces. *)
let into reduction x =
  fst
    (Nx.Prim.eval ~by:"test"
       (Reduce
          {
            layout = contiguous (Nx.shape x);
            axes = [| 0 |];
            prog = identity;
            reductions = [ reduction ];
            loads = [| Plain x |];
          }))

let kinks =
  let ones = Nx.ones Nx.float16 [| 65536 |] in
  let pow (a, b) = Nx.pow a b in
  let base b a = snd (Rune.jvp' (fun a -> Nx.pow a b) a (Nx.ones_like a)) in
  let a = v [| 0.1; 0.5; 2.; 1e-310; inf |] in
  let zeros = Nx.zeros_like a and units = Nx.ones_like a in
  let r = Float.sqrt 0.5 in
  [
    row "abs at both zeros" [| 1.; 1.; 1.; -1. |] (fun () ->
        along Nx.abs (v [| 0.; -0.; 2.; -2. |]) (v [| 1.; 1.; 1.; 1. |]));
    row "maximum's ties and NaNs" [| 2.; 1.; 3.; 0.; 0. |] (fun () ->
        along2
          (fun (a, b) -> Nx.maximum a b)
          (v [| 1.; 2.; 0.; nan; 1. |], v [| 1.; 0.; 2.; 1.; nan |])
          (v [| 1.; 1.; 1.; 1.; 1. |], v [| 3.; 3.; 3.; 3.; 3. |]));
    row "minimum's ties and NaNs" [| 2.; 3.; 1.; 0.; 0. |] (fun () ->
        along2
          (fun (a, b) -> Nx.minimum a b)
          (v [| 1.; 2.; 0.; 1.; nan |], v [| 1.; 0.; 2.; nan; 1. |])
          (v [| 1.; 1.; 1.; 1.; 1. |], v [| 3.; 3.; 3.; 3.; 3. |]));
    row "max's ties" [| 3. |] (fun () ->
        along (Nx.max ~keepdims:true) (v [| 1.; 3.; 3. |]) (v [| 1.; 2.; 4. |]));
    row "min's ties" [| 3. |] (fun () ->
        along (Nx.min ~keepdims:true) (v [| 2.; 1.; 1. |]) (v [| 5.; 2.; 4. |]));
    (* More ties than float16 counts: their tangents' mean is still 1. *)
    row "max's 65536 ties, float16" [| 1. |] (fun () ->
        along (fun x -> one_of (Nx.max x)) ones ones);
    row "a scatter's 65536 tied updates, float16" [| 1. |] (fun () ->
        along
          (fun u ->
            Nx.scatter ~combine:Max ~axis:0
              (Nx.zeros Nx.int64 [| 65536 |])
              u
              (Nx.zeros Nx.float16 [| 1 |]))
          ones ones);
    row "a scatter's target and the update tying it" [| 1.; 3. |] (fun () ->
        along2
          (fun (into, u) ->
            Nx.scatter ~combine:Max ~axis:0 (idx [| 1; 0 |]) u into)
          (v [| 1.; 5. |], v [| 5.; 0.5 |])
          (v [| 1.; 2. |], v [| 4.; 8. |]));
    row "an arg reduction's tied extreme" [| 3. |] (fun () ->
        along
          (fun x -> one_of (fst (fst (reduce1 [ Arg (Max, 0, f64) ] x))))
          (Nx.reshape [| 1; 3 |] (v [| 1.; 3.; 3. |]))
          (Nx.reshape [| 1; 3 |] (v [| 1.; 2.; 4. |])));
    row "cummax's first element to reach it" [| 1.; 2.; 2.; 2. |] (fun () ->
        along (Nx.cummax ~axis:0)
          (v [| 1.; 3.; 3.; 2. |])
          (v [| 1.; 2.; 4.; 8. |]));
    row "cummax from a NaN" [| 1.; 10.; 10.; 10. |] (fun () ->
        along (Nx.cummax ~axis:0)
          (v [| 1.; nan; 5.; 0. |])
          (v [| 1.; 10.; 100.; 1000. |]));
    row "cummin from a NaN" [| 1.; 10.; 10.; 10. |] (fun () ->
        along (Nx.cummin ~axis:0)
          (v [| 1.; nan; 5.; 0. |])
          (v [| 1.; 10.; 100.; 1000. |]));
    row "a running max from the carry tying it" [| 9.; 9. |] (fun () ->
        along
          (fun x ->
            let c, r = carried x in
            Nx.reshape [| -1 |] (scan1 ~carry:c (Monoid (Max, 0, f64)) r))
          (Nx.reshape [| 1; 3 |] (v [| 3.; 3.; 1. |]))
          (Nx.reshape [| 1; 3 |] (v [| 9.; 5.; 7. |])));
    row "an infinite logsumexp" [| 2. |] (fun () ->
        along
          (Nx.logsumexp ~keepdims:true)
          (v [| inf; inf; 1. |])
          (v [| 1.; 3.; 5. |]));
    row "logsumexp's 65536 infinite terms, float16" [| 1. |] (fun () ->
        along
          (fun x -> one_of (Nx.logsumexp x))
          (Nx.full Nx.float16 [| 65536 |] inf)
          ones);
    row "logcumsumexp at and after -inf" [| 1.; 2.; 5. |] (fun () ->
        along Nx.logcumsumexp
          (v [| Float.neg_infinity; Float.neg_infinity; 0. |])
          (v [| 1.; 3.; 5. |]));
    row "hypot where its operands' magnitudes tie"
      [| 3. *. r; r |]
      (fun () ->
        along2
          (fun (a, b) -> Nx.hypot a b)
          (v [| 1.; -2. |], v [| 1.; 2. |])
          (v [| 1.; 1. |], v [| 2.; 2. |]));
    (* At their jumps and between them. *)
    row "sign, floor, ceil, round and trunc" (Array.make 25 0.) (fun () ->
        let x = v [| 0.; -0.; 1.; -1.5; 2.5 |] in
        let t = v [| 1.; 1.; 1.; 1.; 1. |] in
        Array.concat
          (List.map
             (fun f -> along f x t)
             [ Nx.sign; Nx.floor; Nx.ceil; Nx.round; Nx.trunc ]));
    row "pow along a zero base" [| 0.; 0.; 0. |] (fun () ->
        along
          (Nx.pow (v [| 0.; 0.; 0. |]))
          (v [| -1.; 2.; 0.5 |])
          (v [| 1.; 1.; 1. |]));
    row "pow along a zero exponent" [| 0.; 0.; 0. |] (fun () ->
        along2 pow
          (v [| 0.; 1e-310; 2. |], v [| 0.; 0.; 0. |])
          (v [| 1.; 1.; 1. |], v [| 0.; 0.; 0. |]));
    row "pow at a zero base and exponent, along both" [| 0. |] (fun () ->
        along2 pow (v [| 0. |], v [| 0. |]) (v [| 1. |], v [| 1. |]));
    (* The tangent along the base is [b a^(b - 1)] at [b = 0] too: its
       derivative along the exponent there is [1 / a], and along the base 0. *)
    row "pow's mixed second derivative at a zero exponent"
      [| 10.; 2.; 0.5; inf; 0. |] (fun () ->
        along (fun b -> base b a) zeros units);
    row "pow's second derivative along the base at a zero exponent"
      [| 0.; 0.; 0.; 0.; 0. |] (fun () ->
        along (base zeros) (v [| 1e-200; 0.5; 2.; 1e-310; inf |]) units);
  ]

(* Values the derivatives' definitions give where rounding or an infinity could
   reach a tangent: products at zeros, infinities and underflow, an operand
   without a tangent beside an infinite coefficient, narrow dtypes whose primal
   rounds, reductions into another dtype. *)
let edges =
  [
    row "prod at a zero" [| 6. |] (fun () ->
        along (Nx.prod ~keepdims:true) (v [| 0.; 2.; 3. |]) (v [| 1.; 0.; 0. |]));
    row "cumprod at a zero" [| 1.; 2.; 6. |] (fun () ->
        along Nx.cumprod (v [| 0.; 2.; 3. |]) (v [| 1.; 0.; 0. |]));
    row "cumprod's first at an infinity" [| 1. |] (fun () ->
        [| (along Nx.cumprod (v [| inf; 1. |]) (v [| 1.; 0. |])).(0) |]);
    (* Each element's derivative is the product of the others. *)
    row "prod that underflows" [| 5.33e-9; 4.94e-324 |] (fun () ->
        Nx.to_array
          (Rune.jacfwd' (Nx.prod ~keepdims:true) (v [| 4.94e-324; 5.33e-9 |])));
    row "a constant factor of an infinity" [| 2.; 0.5 |] (fun () ->
        let x = v [| inf |] and t = v [| 1. |] in
        Array.append
          (along (fun x -> Nx.mul_s x 2.) x t)
          (along (fun x -> Nx.div_s x 2.) x t));
    row "a constant exponent of a zero" [| 0.; 0. |] (fun () ->
        let x = v [| 0. |] and t = v [| 1. |] in
        Array.append
          (along (fun x -> Nx.pow_s x 2.) x t)
          (along (fun x -> Nx.pow_s x 3.) x t));
    row "logsumexp shifted, bfloat16" [| 1. |] (fun () ->
        along
          (Nx.logsumexp ~keepdims:false)
          (Nx.cast Nx.bfloat16 (v [| 100.; 100. |]))
          (Nx.cast Nx.bfloat16 (v [| 1.; 1. |])));
    row "var shifted, float16" [| 0. |] (fun () ->
        along
          (fun x -> one_of (Nx.var x))
          (float16 [| 2000.; 2001. |])
          (float16 [| 1.; 1. |]));
    row "var along one, float16" [| 0.5 |] (fun () ->
        along
          (fun x -> one_of (Nx.var x))
          (float16 [| 2000.; 2001. |])
          (float16 [| 0.; 1. |]));
    row ~rel:(1. /. 300.) "prod in its carrier, float16" [| 300. |] (fun () ->
        along
          (fun x -> one_of (Nx.prod x))
          (float16 [| 300.; 300.; 1. /. 300.; 1. /. 300. |])
          (float16 [| 0.; 0.; 1.; 0. |]));
    row "max into float32" [| 2. |] (fun () ->
        along
          (fun x -> into (Monoid (Max, 0, Nx.float32)) x)
          (v [| 0.1; 0.2 |])
          (v [| 1.; 2. |]));
    row ~rel:1e-6 "prod into float32, in the output's carrier" [| 1e30 |]
      (fun () ->
        along
          (fun x -> into (Monoid (Prod, 0, Nx.float32)) x)
          (v [| 1e30; 1e30; 1e-30; 1e-30 |])
          (v [| 0.; 0.; 1.; 0. |]));
    row "max into int32" [| 0. |] (fun () ->
        along
          (fun x -> into (Monoid (Max, 0, Nx.int32)) x)
          (v [| 1.5; 2.5 |])
          (v [| 1.; 2. |]));
  ]

(* A tangent is the directional derivative: [f'(z) t] for a holomorphic [f],
   [conj t] for the conjugate, [Re (conj z t) / |z|] for the modulus. *)
let test_complex () =
  let z =
    Nx.create c128 [| 2 |] [| { re = 0.5; im = -1. }; { re = -2.; im = 0.25 } |]
  in
  let t =
    Nx.create c128 [| 2 |] [| { re = 1.; im = 2. }; { re = -0.5; im = 1. } |]
  in
  let close = array (pair (float 1e-12) (float 1e-12)) in
  let parts z =
    Array.map (fun (c : Complex.t) -> (c.re, c.im)) (Nx.to_array z)
  in
  equal ~msg:"exp" close
    (parts (Nx.mul (Nx.exp z) t))
    (parts (snd (Rune.jvp' Nx.exp z t)));
  equal ~msg:"mul by a constant" close
    (parts (Nx.mul z t))
    (parts (snd (Rune.jvp' (fun w -> Nx.mul w z) z t)));
  equal ~msg:"conjugate" close
    (parts (Nx.conjugate t))
    (parts (snd (Rune.jvp' Nx.conjugate z t)));
  let expected =
    Array.map2
      (fun (z : Complex.t) (t : Complex.t) ->
        ((z.re *. t.re) +. (z.im *. t.im)) /. Complex.norm z)
      (Nx.to_array z) (Nx.to_array t)
  in
  equal ~msg:"magnitude"
    (array (float 1e-12))
    expected
    (Nx.to_array (snd (Rune.jvp' (Nx.magnitude f64) z t)))

(* Factorisations *)

type factorisation = Eigh | Eig | Svd

let pp_factorisation ppf f =
  Format.pp_print_string ppf
    (match f with Eigh -> "eigh" | Eig -> "eig" | Svd -> "svd")

(* A factorisation, real or complex, of a square matrix, or for svd of any
   shape, with or without a batch axis, and a matrix and a tangent's elements in
   [-1, 1], the imaginary parts after the real ones. *)
let factor_draws =
  let open Gen in
  (let* f = of_list [ Eigh; Eig; Svd ] in
   let* complex = bool and+ batch = bool in
   let* m = int_range 1 3 and+ n = int_range 1 4 in
   let s =
     Array.append
       (if batch then [| 2 |] else [||])
       [| m; (if f = Svd then n else m) |]
   in
   let k = numel s * if complex then 2 else 1 in
   let unit = array ~size:(constant k) (float_range (-1.) 1.) in
   let+ a = unit and+ t = unit in
   (f, complex, s, a, t))
  |> with_pp (fun ppf (f, complex, s, _, _) ->
      Format.fprintf ppf "%a, %s %a" pp_factorisation f
        (if complex then "complex" else "real")
        Operand.pp_shape s)

let made s xs ~part =
  let n = numel s in
  part (fun k -> Nx.create f64 s (Array.sub xs (k * n) n))

let real part = part 0
let complex part = Nx.complex c128 ~re:(part 0) ~im:(part 1)

let diagonal_zero ~msg m =
  let parts =
    Array.map
      (fun (c : Complex.t) -> (c.re, c.im))
      (Nx.to_array (Nx.cast c128 (Nx.diagonal m)))
  in
  equal ~msg
    (array (pair (float 1e-10) (float 1e-10)))
    (Array.map (fun _ -> (0., 0.)) parts)
    parts

let columns s = Nx.unsqueeze ~axes:[ Nx.ndim s - 1 ] s

(* The tangent of each vector of eigh and eig is orthogonal to it, as is each
   right singular vector's of svd, whose left ones keep [a = u diag(s) vh] to
   first order. *)
let gauges : type v s.
    factorisation -> (v, s, Nx.host) Nx.t -> (v, s, Nx.host) Nx.t -> unit =
 fun f a t ->
  match f with
  | Eigh ->
      let _, (_, dv) = Rune.jvp T.tensor (two ()) (Nx.eigh f64) a t in
      let _, v = Nx.eigh f64 a in
      diagonal_zero ~msg:"eigh: vᴴ dv" (Nx.matmul (adjoint v) dv)
  | Eig ->
      let (_, v), (_, dv) = Rune.jvp T.tensor (two ()) (Nx.eig c128) a t in
      diagonal_zero ~msg:"eig: vᴴ dv" (Nx.matmul (adjoint v) dv)
  | Svd ->
      let svd a =
        let u, s, vh = Nx.svd f64 a in
        (u, (s, vh))
      in
      let (u, (s, vh)), (du, (ds, dvh)) =
        Rune.jvp T.tensor T.(pair tensor (pair tensor tensor)) svd a t
      in
      diagonal_zero ~msg:"svd: vh dvhᴴ" (Nx.matmul vh (adjoint dvh));
      let dt = Nx.dtype a in
      let scaled u s = Nx.mul u (columns (Nx.cast dt s)) in
      let first_order =
        Nx.add
          (Nx.add (Nx.matmul (scaled du s) vh) (Nx.matmul (scaled u ds) vh))
          (Nx.matmul (scaled u s) dvh)
      in
      let parts x =
        Array.map
          (fun (c : Complex.t) -> (c.re, c.im))
          (Nx.to_array (Nx.cast c128 x))
      in
      equal ~msg:"svd: da = d(u diag(s) vh)"
        (array (pair (float 1e-10) (float 1e-10)))
        (parts t) (parts first_order)

(* A factorisation's values alone take their tangents from the vectors the full
   routine computes: the full routine's values' tangents, bit for bit. *)
let alone : type v s.
    factorisation -> (v, s, Nx.host) Nx.t -> (v, s, Nx.host) Nx.t -> unit =
 fun f a t ->
  let tangent g = codes (snd (Rune.jvp' g a t)) in
  let same alone full = equal codes_t (tangent full) (tangent alone) in
  match f with
  | Eigh ->
      same
        (fun a -> fst (Nx.eigh ~vectors:false f64 a))
        (fun a -> fst (Nx.eigh f64 a))
  | Eig ->
      same
        (fun a -> fst (Nx.eig ~vectors:false c128 a))
        (fun a -> fst (Nx.eig c128 a))
  | Svd ->
      let s ~vectors a =
        let _, s, _ = Nx.svd ~vectors f64 a in
        s
      in
      same (s ~vectors:false) (s ~vectors:true)

(* A law over a draw's matrix, spaced so that its values are distinct, and its
   tangent. *)
type law = {
  law : 'v 's. ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t -> unit;
}

let on_draw f (fn, complex_dtype, s, xs, ts) =
  cover "complex" complex_dtype;
  cover "a batch" (Array.length s = 3);
  let r = Array.length s in
  cover "a non-square svd" (fn = Svd && s.(r - 1) <> s.(r - 2));
  let square a =
    match fn with Eigh -> symmetric a | Eig -> half (half a) | Svd -> a
  in
  if complex_dtype then
    f.law (zspaced (square (made s xs ~part:complex))) (made s ts ~part:complex)
  else f.law (spaced (square (made s xs ~part:real))) (made s ts ~part:real)

let gauge_law ((fn, _, _, _, _) as d) =
  on_draw { law = (fun a t -> gauges fn a t) } d

let alone_law ((fn, _, _, _, _) as d) =
  on_draw { law = (fun a t -> alone fn a t) } d

let test_repeated () =
  let i = Nx.eye f64 2 and t = Nx.create f64 [| 2; 2 |] [| 0.; 1.; 1.; 0. |] in
  let _, (_, dv) = Rune.jvp T.tensor (two ()) (Nx.eigh f64) i t in
  let _, (du, _) =
    Rune.jvp T.tensor (two ())
      (fun a ->
        let u, s, _ = Nx.svd f64 a in
        (u, s))
      i t
  in
  (* Both values are tied, so each vector's tangent holds a non-finite
     element. *)
  let finite_columns x =
    let finite j =
      Array.for_all Float.is_finite (Nx.to_array (Nx.slice [ A; I j ] x))
    in
    List.length (List.filter finite [ 0; 1 ])
  in
  equal ~msg:"eigh's finite vectors" int 0 (finite_columns dv);
  equal ~msg:"svd's finite vectors" int 0 (finite_columns du)

(* Interface *)

(* An integer or boolean result has a zero tangent, as one on no parameter does,
   and an integer or boolean parameter's tangent is ignored. *)
let test_zero_tangents () =
  let i = Nx.create Nx.int32 [| 2 |] [| 3l; 4l |] in
  let b = Nx.create Nx.bool [| 2 |] [| true; false |] in
  let _, (ti, (tn, tb)) =
    Rune.jvp
      T.(pair tensor (pair tensor tensor))
      T.(pair tensor (pair tensor tensor))
      (fun (x, (n, b)) -> (Nx.cast Nx.int32 x, (Nx.add n n, Nx.logical_not b)))
      (v [| 1.5; 2. |], (i, b))
      (v [| 1.; 1. |], (i, b))
  in
  equal ~msg:"integer result" (array int32) [| 0l; 0l |] (Nx.to_array ti);
  equal ~msg:"integer parameter" (array int32) [| 0l; 0l |] (Nx.to_array tn);
  equal ~msg:"boolean parameter" (array bool) [| false; false |]
    (Nx.to_array tb);
  let _, t = Rune.jvp' (fun _ -> v [| 7. |]) (v [| 1. |]) (v [| 1. |]) in
  equal ~msg:"a constant" (array float_exact) [| 0. |] (Nx.to_array t)

(* A tangent is a value of its own, of the identity and after a transpose too:
   it can be donated, and its seed lives on. *)
let test_fresh () =
  let x = v [| 1.; 2. |] and dx = v [| 3.; 4. |] in
  let _, t = Rune.jvp' Fun.id x dx in
  fresh ~msg:"the identity's" t;
  ignore (Nx.add (Nx.donate t) x);
  equal ~msg:"its seed" (array float_exact) [| 3.; 4. |] (Nx.to_array dx);
  let m = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  fresh ~msg:"a transpose's" (snd (Rune.jvp' Nx.transpose m m))

(* One array at two leaves is two parameters, each with its tangent. *)
let test_two_leaves () =
  let xs = [| 0.5; -2. |] in
  let x = v xs in
  let expected =
    Array.map (fun x -> Float.sin x +. (2. *. x *. Float.cos x)) xs
  in
  equal
    (array (float 1e-12))
    expected
    (along2
       (fun (a, b) -> Nx.mul a (Nx.sin b))
       (x, x)
       (v [| 1.; 1. |], v [| 2.; 2. |]))

(* A parameter of zeros has its tangent, through a scan too. *)
let test_zero_primal () =
  let zero = Nx.zeros f64 [| 2 |] and t = v [| 1.; 2. |] in
  let y, dy = Rune.jvp' (fun x -> Nx.add (Nx.mul x x) x) zero t in
  equal ~msg:"primal" (array float_exact) [| 0.; 0. |] (Nx.to_array y);
  equal ~msg:"tangent" (array float_exact) [| 1.; 2. |] (Nx.to_array dy);
  let xs = Nx.create f64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let f c =
    fst (Rune.scan' ~f:(fun c x -> (Nx.add c (Nx.mul c x), c)) ~init:c xs)
  in
  (* c_3 = c_0 (1 + x_0) (1 + x_1) (1 + x_2), linear in c_0. *)
  equal ~msg:"through a scan" (array float_exact)
    [| 1. *. 2. *. 4. *. 6.; 2. *. 3. *. 5. *. 7. |]
    (along f zero t)

let test_detach () =
  let x = v [| 2.; -1. |] and t = v [| 1.; 3. |] in
  equal ~msg:"detach" (array float_exact) [| 2.; -3. |]
    (along (fun x -> Nx.mul x (Rune.detach x)) x t);
  (* Every differentiation around the call detaches it. *)
  let one = v [| 1. |] in
  let inner x = fst (Rune.jvp' (fun y -> Rune.detach (Nx.mul x y)) one one) in
  equal ~msg:"nested" (array float_exact) [| 0. |]
    (along inner (v [| 2. |]) one)

(* An OCaml branch on a value reads its primal. *)
let test_read () =
  let f x = if Nx.item [ 0 ] x > 0. then Nx.sin x else Nx.cos x in
  equal
    (array (float 1e-12))
    [| Float.cos 1. |]
    (along f (v [| 1. |]) (v [| 1. |]))

(* Nested forward modes: each tangent belongs to its own. *)
let test_nesting () =
  let one = v [| 1. |] in
  let f x = Nx.mul x (snd (Rune.jvp' (fun y -> Nx.add x y) one one)) in
  equal ~msg:"perturbations apart" (array float_exact) [| 1. |]
    (along f one one)

let test_jacfwd () =
  let floats = array (float 1e-12) in
  let x = v [| 1.; 2.; -3. |] in
  let j = Rune.jacfwd' (fun x -> Nx.mul x x) x in
  equal ~msg:"shape" (array int) [| 3; 3 |] (Nx.shape j);
  equal floats [| 2.; 0.; 0.; 0.; 4.; 0.; 0.; 0.; -6. |] (Nx.to_array j);
  (* A linear map's Jacobian is its matrix, rows the result's elements. *)
  let a = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let j = Rune.jacfwd' (fun x -> Nx.matmul a x) x in
  equal ~msg:"a matrix's shape" (array int) [| 2; 3 |] (Nx.shape j);
  equal ~msg:"a matrix" floats (Nx.to_array a) (Nx.to_array j);
  (* A transpose's: [J i j k l] is 1 where [(k, l) = (j, i)], in C order. *)
  let m = Nx.create f64 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let j = Rune.jacfwd' Nx.transpose m in
  equal ~msg:"a transpose's shape" (array int) [| 2; 2; 2; 2 |] (Nx.shape j);
  let expected =
    Array.init 16 (fun n ->
        let i = n / 8
        and jj = n / 4 mod 2
        and k = n / 2 mod 2
        and l = n mod 2 in
        if k = jj && l = i then 1. else 0.)
  in
  equal ~msg:"a transpose" floats expected (Nx.to_array j);
  (* A complex argument's columns are the tangents along its real units. *)
  let z =
    Nx.create c128 [| 2 |] [| { re = 1.; im = 2. }; { re = -1.; im = 0.5 } |]
  in
  let j = Rune.jacfwd' (fun z -> Nx.mul z z) z in
  let two_z =
    Array.map (fun c -> Complex.mul { re = 2.; im = 0. } c) (Nx.to_array z)
  in
  equal ~msg:"complex"
    (array (pair (float 1e-12) (float 1e-12)))
    (Array.map
       (fun (c : Complex.t) -> (c.re, c.im))
       [| two_z.(0); Complex.zero; Complex.zero; two_z.(1) |])
    (Array.map (fun (c : Complex.t) -> (c.re, c.im)) (Nx.to_array j));
  let j = Rune.jacfwd' (fun x -> Nx.sum x) (Nx.zeros f64 [| 0 |]) in
  equal ~msg:"no element" (array int) [| 0 |] (Nx.shape j)

(* Column [k] of the Jacobian, its elements at index [k] of the argument's axes
   in C order, is the tangent along element [k], bit for bit; where jvp' raises,
   jacfwd' raises its message under its own name. *)
let jac_cases =
  let open Gen in
  (let* s = Operand.shape in
   let* e = Expr.gen ~rank:(Array.length s) in
   let* k = int_range 0 (Int.max 0 (numel s - 1)) in
   let+ x = Operand.draw s in
   (s, x, k, e))
  |> with_pp (fun ppf (s, (x : Operand.t), k, e) ->
      Format.fprintf ppf "%a, x %a, column %d: %a" Operand.pp_shape s
        Operand.pp_view x.view k Expr.pp e)

(* A maximum over an axis a 0-d argument lacks, which nx refuses under jvp'
   and jacfwd' alike. No drawn case reaches it. *)
let jac_examples =
  [
    ( [||],
      { Operand.view = Plain; values = [| Operand.F 1. |] },
      0,
      Expr.(Un (Max 0, X)) );
  ]

let law_jacobian (s, xo, k, e) =
  let x = Operand.make f64 s xo in
  let n = numel s in
  cover "a zero-size shape" (n = 0);
  cover "several elements" (n > 1);
  let f x = Expr.eval x x e in
  let jacobian = outcome (fun () -> Rune.jacfwd' f x) in
  cover "a refused function"
    (match jacobian with Raised _ -> true | _ -> false);
  match jacobian with
  | Raised m ->
      let renamed m =
        let jvp = "Rune.jvp'" and n = String.length m in
        let k = String.length jvp in
        if n >= k && String.sub m 0 k = jvp then
          "Rune.jacfwd'" ^ String.sub m k (n - k)
        else m
      in
      let along =
        match Rune.jvp' f x (Nx.zeros_like x) with
        | _ -> "no raise"
        | exception Invalid_argument m -> renamed m
      in
      equal ~msg:"raises as jvp'" string along m
  | Value j ->
      let sy = Array.sub (Nx.shape j) 0 (Nx.ndim j - Array.length s) in
      equal ~msg:"shape" (array int) (Array.append sy s) (Nx.shape j);
      if n > 0 then begin
        let column =
          Nx.reshape sy (Nx.slice [ A; I k ] (Nx.reshape [| numel sy; n |] j))
        in
        let unit =
          Nx.reshape s
            (Nx.create f64 [| n |]
               (Array.init n (fun i -> if i = k then 1. else 0.)))
        in
        equal ~msg:"column" bits_t
          (bits (snd (Rune.jvp' f x unit)))
          (bits column)
      end

(* An addition under forward mode adds its primal. *)
let test_total () =
  let total = Rune.Total.make () in
  let _, sum =
    Rune.Total.collect total ~init:(v [| 0. |]) (fun () ->
        Rune.jvp'
          (fun x ->
            Rune.Total.add total x;
            Nx.sin x)
          (v [| 2. |]) (v [| 1. |]))
  in
  equal (array float_exact) [| 2. |] (Nx.to_array sum)

(* Errors *)

type message = Is of string | Starts of string

(* A float array of any width, so that a tangent's dtype may differ from its
   parameter's. *)
type real = Real : (float, 's, Nx.host) Nx.t -> real

let real = T.v (fun c (Real x) -> Real (T.Walk.tensor c x))
let tall = Nx.create f64 [| 3; 2 |] [| 1.; 0.; 0.; 1.; 1.; 1. |]

(* [raised f] is the message [f] raises; [as_jacfwd m] is a message of jvp'
   under jacfwd''s name. *)
let raised f =
  match f () with
  | _ -> "no raise"
  | exception Invalid_argument m -> m

let as_jacfwd m =
  let jvp = "Rune.jvp'" in
  if not (String.starts_with ~prefix:jvp m) then m
  else
    let k = String.length jvp in
    "Rune.jacfwd'" ^ String.sub m k (String.length m - k)

let complete_qr a = fst (Nx.qr ~factors:Complete a)

let errors =
  let vector = v [| 1.; 2. |] and ints = Nx.zeros Nx.int32 [| 2 |] in
  let scaled s =
    Nx.Prim.eval ~by:"test"
      (Contract
         {
           batch = [||];
           contracting = [| (1, 0) |];
           acc = any;
           out = f64;
           a = Nx.reshape [| 1; 2 |] vector;
           b = Nx.reshape [| 2; 1 |] (v [| 3.; 4. |]);
           init = None;
           a_scale = Some (Value s);
           b_scale = None;
           segments = None;
         })
  in
  [
    ( "jvp of integer parameters",
      Is "Rune.jvp: the parameters hold no real or complex tensor",
      fun () -> ignore (Rune.jvp T.tensor T.tensor Fun.id ints ints) );
    ( "jvp' of integer parameters",
      Is "Rune.jvp': the parameters hold no real or complex tensor",
      fun () -> ignore (Rune.jvp' Fun.id ints ints) );
    ( "jacfwd' of integer parameters",
      Is "Rune.jacfwd': the parameters hold no real or complex tensor",
      fun () -> ignore (Rune.jacfwd' Fun.id ints) );
    ( "a tangent of another shape",
      Is "Rune.jvp: 1: shape [2] in the parameters, [3] in the tangents",
      fun () ->
        ignore
          (Rune.jvp (two ()) T.tensor fst (vector, vector)
             (vector, v [| 1.; 2.; 3. |])) );
    ( "a tangent of another dtype",
      Starts "Rune.jvp: 1: ",
      fun () ->
        ignore
          (Rune.jvp
             T.(pair tensor real)
             T.tensor fst (vector, Real vector)
             (vector, Real (Nx.ones Nx.float32 [| 2 |]))) );
    ( "tangents of other visits",
      Starts "Rune.jvp: 1: ",
      fun () ->
        ignore
          (Rune.jvp
             T.(pair tensor (list tensor))
             T.tensor fst (vector, [ vector ]) (vector, [])) );
    ( "a complete QR of a tall matrix",
      Starts "Rune.jvp': ",
      fun () ->
        ignore (Rune.jvp' (fun a -> fst (Nx.qr ~factors:Complete a)) tall tall)
    );
    ( "a complete QR of a tall matrix under jacfwd'",
      Is (as_jacfwd (raised (fun () -> Rune.jvp' complete_qr tall tall))),
      fun () ->
        ignore (Rune.jacfwd' complete_qr tall)
    );
    ( "a complete SVD of a non-square matrix",
      Starts "Rune.jvp': ",
      fun () ->
        ignore
          (Rune.jvp'
             (fun a ->
               let _, s, _ = Nx.svd ~factors:Complete f64 a in
               s)
             tall tall) );
    ( "a contraction's scale on the parameters",
      Starts "Rune.jvp': ",
      fun () ->
        let s = Nx.reshape [| 1; 2 |] (v [| 1.; 1. |]) in
        ignore (Rune.jvp' scaled s s) );
  ]

let test_error (_, message, f) =
  match f () with
  | () -> failf "returned"
  | exception Invalid_argument m -> (
      match message with
      | Is e -> equal string e m
      | Starts p -> starts_with ~affix:p m)

(* jvp of a compiled function *)

type jit_case =
  | Jit_case :
      ('v, 's) Nx.dtype
      * int array
      * Operand.t
      * Operand.t
      * Operand.t
      * Operand.t
      * Expr.t
      -> jit_case

(* float32 and float64 in nearly half the cases, every dtype in the rest,
   complex and narrow floats often enough that every run reaches them. *)
let dtypes =
  let floats = [ D.Any f64; D.Any Nx.float32 ] in
  let complex = List.filter (fun (D.Any d) -> D.is Complex d) D.all in
  let narrow =
    List.filter
      (fun (D.Any d) ->
        D.is Float d && not (D.equal d f64 || D.equal d Nx.float32))
      D.all
  in
  Gen.frequency
    [
      (3, Gen.of_list floats);
      (1, Gen.of_list complex);
      (1, Gen.of_list narrow);
      (2, Gen.of_list D.all);
    ]

(* An expression of the functions a compiled call computes exactly at the case's
   dtype, whose derivatives sum no terms. *)
let jit_cases =
  let open Gen in
  (let* s = Operand.shape in
   let* e = Expr.gen ~rank:(Array.length s) in
   let* x = Operand.draw s
   and+ y = Operand.draw s
   and+ tx = Operand.draw s
   and+ ty = Operand.draw s in
   let+ (D.Any dt) = dtypes in
   let e =
     match D.kind dt with
     | Float -> Expr.exact e
     | _ -> Expr.exact_at dt (Expr.generic e)
   in
   Jit_case (dt, s, x, y, tx, ty, Expr.unsummed e))
  |> with_pp (fun ppf (Jit_case (dt, s, x, y, _, _, e)) ->
      Format.fprintf ppf "%a %a, x %a, y %a: %a" Nx.pp_dtype dt Operand.pp_shape
        s Operand.pp_view x.view Operand.pp_view y.view Expr.pp e)

let offset (o : Operand.t) = match o.view with Offset _ -> true | _ -> false

let eval (type v s) (dt : (v, s) Nx.dtype) e :
    (v, s, 'd) Nx.t -> (v, s, 'd) Nx.t -> (v, s, 'd) Nx.t =
  match D.kind dt with
  | Float -> fun x y -> Expr.eval x y e
  | _ -> fun x y -> Expr.eval_any x y e

(* jvp of a compiled function and a compiled jvp give jvp's bits, every dtype's:
   integers and booleans refused alike. *)
let law_jit (Jit_case (dt, s, xo, yo, txo, tyo, e)) =
  let make = Operand.make dt s in
  let x = make xo and y = make yo and tx = make txo and ty = make tyo in
  cover "a zero-size shape" (numel s = 0);
  cover "one element" (numel s = 1 && Array.length s > 0);
  cover "a 0-d shape" (Array.length s = 0);
  cover "a view operand" (xo.view <> Plain || yo.view <> Plain);
  cover "an offset operand" (offset xo || offset yo);
  cover "a composition of over 6 functions" (Expr.size e > 6);
  cover "a complex dtype" (D.is Complex dt);
  cover "a narrow float dtype"
    (D.is Float dt && not (D.equal dt f64 || D.equal dt Nx.float32));
  cover "an integer dtype" (D.is Signed dt || D.is Unsigned dt);
  let f = eval dt e in
  let both (p, t) = (codes p, codes t) in
  let jvp f =
    outcome (fun () ->
        both (Rune.jvp (two ()) T.tensor (fun (x, y) -> f x y) (x, y) (tx, ty)))
  in
  let expected = jvp f in
  cover "a refused function"
    (match expected with Raised _ -> true | _ -> false);
  let t = outcome_t (Testable.pair codes_t codes_t) in
  let g = Rune.jit T.(tensor @-> tensor @-> returns tensor) f in
  equal ~msg:"jvp of jit" t expected (jvp g);
  let staged =
    Rune.jit
      T.(
        tensor @-> tensor @-> tensor @-> tensor @-> returns (pair tensor tensor))
      (fun x y tx ty ->
        Rune.jvp (two ()) T.tensor (fun (x, y) -> f x y) (x, y) (tx, ty))
  in
  equal ~msg:"jit of jvp" t expected
    (outcome (fun () -> both (staged x y tx ty)))

(* One kernel per map *)

let test_one_kernel () =
  let x = Nx.place Count.on (v [| 0.5; -1.; 2. |]) in
  let t = Nx.place Count.on (v [| 1.; 1.; 1. |]) in
  let calls x t f =
    C.reset ();
    ignore (Rune.jvp' f x t);
    (C.calls (), C.positions ())
  in
  equal ~msg:"sin" (pair int int) (1, 3) (calls x t Nx.sin);
  equal ~msg:"a chain of three" (pair int int) (3, 9)
    (calls x t (fun x -> Nx.exp (Nx.mul (Nx.sin x) x)));
  equal ~msg:"a sum and its tangent" (pair int int) (1, 3)
    (calls x t (fun x -> Nx.sum x));
  (* A derivative wider than a block of nx.cpu's holds is one loop too. *)
  let w = Nx.place Count.on (Nx.full f64 [| wide; 3 |] 0.5) in
  let tw = Nx.place Count.on (Nx.full f64 [| wide; 3 |] 1.) in
  equal ~msg:"a map of twenty operands" (pair int int) (1, 3)
    (calls w tw wide_map);
  equal ~msg:"a sum over twenty operands" (pair int int) (1, 3)
    (calls w tw (wide_reduce (Monoid (Sum, 0, f64))))

let () =
  exit
    (run "rune jvp"
       [
         group "tangents" tangent_props;
         group "second derivatives" second_props;
         cases ~name:(fun r -> r.name) "kinks" kinks check_row;
         cases ~name:(fun r -> r.name) "edges" edges check_row;
         group "conventions"
           [
             test "complex tangents are directional derivatives" test_complex;
             prop "factorisations fix their vectors' gauges" factor_draws
               gauge_law;
             prop "values alone take the full routine's tangents" factor_draws
               alone_law;
             test "vectors at repeated values are not finite" test_repeated;
           ];
         group "interface"
           [
             test "integer, boolean and constant tangents" test_zero_tangents;
             test "a tangent is a value of its own" test_fresh;
             test "one array at two leaves" test_two_leaves;
             test "a parameter of zeros" test_zero_primal;
             test "detach" test_detach;
             test "a read reads the primal" test_read;
             test "nesting" test_nesting;
             test "jacfwd'" test_jacfwd;
             prop ~examples:jac_examples
               "column k of jacfwd' is jvp' along element k" jac_cases
               law_jacobian;
             test "totals" test_total;
           ];
         cases ~name:(fun (n, _, _) -> n) "errors" errors test_error;
         group "jit"
           [
             prop "jvp of jit and jit of jvp of exact functions" jit_cases
               law_jit;
           ];
         group "kernels"
           [ test "a map's tangent is one kernel" test_one_kernel ];
       ])
