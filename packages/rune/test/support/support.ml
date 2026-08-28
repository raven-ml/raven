(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shared fixtures and checks for the rune test suite. *)

open Windtrap

let f32 = Nx.float32
let f64 = Nx.float64
let vec32 xs = Nx.create f32 [| Array.length xs |] xs
let vec64 xs = Nx.create f64 [| Array.length xs |] xs
let mat64 r c xs = Nx.create f64 [| r; c |] xs
let to_arr t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))
let scalar t = (to_arr t).(0)

(* The sliding-window movement is internal — it backs [Nx.stft] and is not part
   of Nx's public surface — but the transformation rules are written against its
   effect, so exercise it there rather than through a caller that would also
   drag in [rfft], which has no rule of its own. *)
let sliding_window ~axis ~window ~step x =
  Nx_effect.sliding_window x ~axis ~window ~step

let check_arr ?(eps = 1e-5) ~msg expected actual =
  let t = if eps = 0. then float_exact else float eps in
  let actual = to_arr actual in
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i e -> equal ~msg:(Printf.sprintf "%s[%d]" msg i) t e actual.(i))
    expected

let scalar_like (type a b) (t : (a, b) Nx.t) (v : float) : (a, b) Nx.t =
  let dt = Nx.dtype t in
  Nx.full dt [||] (Nx_core.Dtype.of_float dt v)

let as_f32 (type a b) (x : (a, b) Nx.t) : Nx.float32_t =
  match Nx_core.Dtype.equal_witness (Nx.dtype x) f32 with
  | Some Type.Equal -> x
  | None -> failwith "expected a float32 leaf"

(* A statically-typed parameter record with mixed dtypes: the canonical Ptree.S
   instance used across suites. *)

type params = { w : Nx.float32_t; b : Nx.float32_t; scale : Nx.float64_t }

module Params = struct
  type t = params

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w; b; scale } =
    { w = f w; b = f b; scale = f scale }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { w = f p.w q.w; b = f p.b q.b; scale = f p.scale q.scale }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w; b; scale } =
    f w;
    f b;
    f scale
end

let params () =
  {
    w = vec32 [| 1.0; -2.0; 3.0 |];
    b = vec32 [| 0.5 |];
    scale = vec64 [| 2.0 |];
  }

(* A pair of float64 tensors, for differentiating binary operations. *)

type pair = { fst : Nx.float64_t; snd : Nx.float64_t }

module Pair = struct
  type t = pair

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { fst; snd } =
    { fst = f fst; snd = f snd }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { fst = f p.fst q.fst; snd = f p.snd q.snd }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { fst; snd } =
    f fst;
    f snd
end

(* Finite-difference oracle.

   Gradient rules are validated against central differences of a scalar float64
   loss. The loss weights the operation's output with a fixed non-uniform
   tensor: a uniform cotangent would let transposition and permutation mistakes
   cancel out in the comparison. *)

(* [weighted y] is [sum (w * y)] with deterministic non-uniform weights. *)
let weighted y =
  let n = Nx.numel y in
  let w =
    Nx.create f64 (Nx.shape y)
      (Array.init n (fun i -> float_of_int ((i mod 5) + 1) /. 2.0))
  in
  Nx.sum (Nx.mul y w)

let central_diff ~h (eval : float array -> float) (xs : float array) :
    float array =
  Array.init (Array.length xs) (fun i ->
      let at d =
        let ys = Array.copy xs in
        ys.(i) <- ys.(i) +. d;
        eval ys
      in
      (at h -. at (-.h)) /. (2.0 *. h))

let check_close ~tol ~msg expected actual =
  Array.iteri
    (fun i e ->
      equal ~msg:(Printf.sprintf "%s[%d]" msg i) (float tol) e actual.(i))
    expected

(* [check_grad ~msg f x] compares [grad' (weighted . f)] at [x] against central
   differences. [f] maps a float64 tensor to a float64 tensor. *)
let check_grad ?(h = 1e-5) ?(tol = 1e-3) ~msg (f : Nx.float64_t -> Nx.float64_t)
    (x : Nx.float64_t) =
  let shape = Nx.shape x in
  let loss x = weighted (f x) in
  let analytic = to_arr (Rune.grad' loss x) in
  let numeric =
    central_diff ~h
      (fun ys -> scalar (loss (Nx.create f64 shape ys)))
      (to_arr x)
  in
  check_close ~tol ~msg numeric analytic

(* [check_grad2 ~msg f a b] is {!check_grad} for a binary operation,
   differentiating with respect to both arguments through a {!Pair}. *)
let check_grad2 ?(h = 1e-5) ?(tol = 1e-3) ~msg
    (f : Nx.float64_t -> Nx.float64_t -> Nx.float64_t) (a : Nx.float64_t)
    (b : Nx.float64_t) =
  let loss p = weighted (f p.fst p.snd) in
  let g = Rune.grad (module Pair) loss { fst = a; snd = b } in
  let shape_a = Nx.shape a and shape_b = Nx.shape b in
  let arr_a = to_arr a and arr_b = to_arr b in
  let num_a =
    central_diff ~h
      (fun ys -> scalar (loss { fst = Nx.create f64 shape_a ys; snd = b }))
      arr_a
  in
  let num_b =
    central_diff ~h
      (fun ys -> scalar (loss { fst = a; snd = Nx.create f64 shape_b ys }))
      arr_b
  in
  check_close ~tol ~msg:(msg ^ ".fst") num_a (to_arr g.fst);
  check_close ~tol ~msg:(msg ^ ".snd") num_b (to_arr g.snd)

(* [tangent_like t] is a deterministic, non-uniform tangent for [t]: zero or
   uniform tangents would mask permutation and scaling mistakes. *)
let tangent_like t =
  let n = Nx.numel t in
  Nx.create f64 (Nx.shape t)
    (Array.init n (fun i -> float_of_int ((i * 7 mod 11) - 5) /. 4.0))

(* [check_jvp ~msg f x] compares the forward-mode tangent of [f] at [x] along
   [tangent_like x] against the central difference of [f] along the same
   direction, elementwise on the output. *)
let check_jvp ?(h = 1e-5) ?(tol = 1e-3) ~msg (f : Nx.float64_t -> Nx.float64_t)
    (x : Nx.float64_t) =
  let v = tangent_like x in
  let _, dy = Rune.jvp' f x v in
  let shape = Nx.shape x in
  let xs = to_arr x and vs = to_arr v in
  let eval d =
    to_arr
      (f
         (Nx.create f64 shape (Array.mapi (fun i xi -> xi +. (d *. vs.(i))) xs)))
  in
  let fp = eval h and fm = eval (-.h) in
  let numeric =
    Array.init (Array.length fp) (fun i -> (fp.(i) -. fm.(i)) /. (2.0 *. h))
  in
  check_close ~tol ~msg numeric (to_arr dy)

(* [check_jvp2 ~msg f a b] is {!check_jvp} for a binary operation, feeding
   tangents to both arguments through a {!Pair}. *)
let check_jvp2 ?(h = 1e-5) ?(tol = 1e-3) ~msg
    (f : Nx.float64_t -> Nx.float64_t -> Nx.float64_t) (a : Nx.float64_t)
    (b : Nx.float64_t) =
  let va = tangent_like a and vb = tangent_like b in
  let _, dy =
    Rune.jvp
      (module Pair)
      (fun p -> f p.fst p.snd)
      { fst = a; snd = b } { fst = va; snd = vb }
  in
  let shape_a = Nx.shape a and shape_b = Nx.shape b in
  let arr_a = to_arr a and arr_b = to_arr b in
  let arr_va = to_arr va and arr_vb = to_arr vb in
  let eval d =
    let bump xs vs = Array.mapi (fun i xi -> xi +. (d *. vs.(i))) xs in
    to_arr
      (f
         (Nx.create f64 shape_a (bump arr_a arr_va))
         (Nx.create f64 shape_b (bump arr_b arr_vb)))
  in
  let fp = eval h and fm = eval (-.h) in
  let numeric =
    Array.init (Array.length fp) (fun i -> (fp.(i) -. fm.(i)) /. (2.0 *. h))
  in
  check_close ~tol ~msg numeric (to_arr dy)

(* Complex finite-difference oracle.

   A complex tensor is a pair of real components, and every rule is a real
   linear map on them. These checks differentiate both components separately,
   assemble the real Jacobian, and compare against it under rune's packing: a
   tangent carries [dre + i*dim], a cotangent carries [dL/dre - i*dL/dim]. That
   makes them independent of the convention being right — they measure the
   operation, not another rule. *)

let c128 = Nx.complex128
let cx re im = Complex.{ re; im }

let cvec xs =
  Nx.create c128 [| Array.length xs |] (Array.map (fun (re, im) -> cx re im) xs)

let cmat r c xs =
  Nx.create c128 [| r; c |] (Array.map (fun (re, im) -> cx re im) xs)

let to_carr t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

(* Deterministic complex weights. Both components vary and neither is ever zero:
   a cotangent that is real everywhere is exactly what lets a missing
   conjugation agree with the oracle by coincidence. The two generators differ
   so that a binary rule reading the wrong argument cannot cancel out. *)

let cotangent_like t =
  Nx.create c128 (Nx.shape t)
    (Array.init (Nx.numel t) (fun i ->
         cx
           (float_of_int ((i mod 5) + 1) /. 2.0)
           (float_of_int (((i + 1) mod 4) - 5) /. 4.0)))

let ctangent_like t =
  Nx.create c128 (Nx.shape t)
    (Array.init (Nx.numel t) (fun i ->
         cx
           (float_of_int ((i * 3 mod 5) - 6) /. 4.0)
           (float_of_int ((i mod 4) + 2) /. 3.0)))

let check_cclose ~tol ~msg expected actual =
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i (e : Complex.t) ->
      let a : Complex.t = actual.(i) in
      equal
        ~msg:(Printf.sprintf "%s[%d].re" msg i)
        (float tol) e.Complex.re a.Complex.re;
      equal
        ~msg:(Printf.sprintf "%s[%d].im" msg i)
        (float tol) e.Complex.im a.Complex.im)
    expected

(* [check_carr ~msg expected t] is {!check_cclose} on the flattened complex
   tensor [t]. *)
let check_carr ?(eps = 1e-10) ~msg expected actual =
  check_cclose ~tol:eps ~msg expected (to_carr actual)

(* [cvjp_numeric ~h f z w] is the cotangent [w] pulled back through the real
   Jacobian of [f] at [z], measured by central differences: perturb each
   component of each input, read how each component of each output responds, and
   contract with [w] unpacked into [(dL/dre, dL/dim)]. *)
let cvjp_numeric ~h f (z : Nx.complex128_t) (w : Complex.t array) =
  let shape = Nx.shape z in
  let zs = to_carr z in
  let eval a = to_carr (f (Nx.create c128 shape a)) in
  Array.init (Array.length zs) (fun j ->
      let at dre dim =
        let ys = Array.copy zs in
        ys.(j) <- cx (ys.(j).Complex.re +. dre) (ys.(j).Complex.im +. dim);
        eval ys
      in
      let fpre = at h 0.0 and fmre = at (-.h) 0.0 in
      let fpim = at 0.0 h and fmim = at 0.0 (-.h) in
      let dre = ref 0.0 and dim = ref 0.0 in
      Array.iteri
        (fun i (wi : Complex.t) ->
          let slope p m component =
            (component p.(i) -. component m.(i)) /. (2.0 *. h)
          in
          let re (c : Complex.t) = c.Complex.re
          and im (c : Complex.t) = c.Complex.im in
          let contract p m =
            (slope p m re *. wi.Complex.re) -. (slope p m im *. wi.Complex.im)
          in
          dre := !dre +. contract fpre fmre;
          dim := !dim +. contract fpim fmim)
        w;
      cx !dre (-. !dim))

(* [cjvp_numeric ~h f z v] is the central difference of [f] at [z] along [v],
   taken on both components at once — the directional derivative that a
   pushforward must reproduce. *)
let cjvp_numeric ~h f (z : Nx.complex128_t) (v : Nx.complex128_t) =
  let shape = Nx.shape z in
  let zs = to_carr z and vs = to_carr v in
  let eval d =
    to_carr
      (f
         (Nx.create c128 shape
            (Array.mapi
               (fun i (zi : Complex.t) ->
                 cx
                   (zi.Complex.re +. (d *. vs.(i).Complex.re))
                   (zi.Complex.im +. (d *. vs.(i).Complex.im)))
               zs)))
  in
  let fp = eval h and fm = eval (-.h) in
  Array.init (Array.length fp) (fun i ->
      cx
        ((fp.(i).Complex.re -. fm.(i).Complex.re) /. (2.0 *. h))
        ((fp.(i).Complex.im -. fm.(i).Complex.im) /. (2.0 *. h)))

(* [check_cgrad ~msg f z] compares the reverse-mode pullback of a deterministic
   complex cotangent against the transposed finite-difference Jacobian. *)
let check_cgrad ?(h = 1e-5) ?(tol = 1e-5) ~msg
    (f : Nx.complex128_t -> Nx.complex128_t) (z : Nx.complex128_t) =
  let w = cotangent_like (f z) in
  let _, g = Rune.vjp' f z w in
  check_cclose ~tol ~msg (cvjp_numeric ~h f z (to_carr w)) (to_carr g)

(* [check_cjvp ~msg f z] compares the forward-mode pushforward of a
   deterministic complex tangent against the same central difference. *)
let check_cjvp ?(h = 1e-5) ?(tol = 1e-5) ~msg
    (f : Nx.complex128_t -> Nx.complex128_t) (z : Nx.complex128_t) =
  let v = ctangent_like z in
  let _, dy = Rune.jvp' f z v in
  check_cclose ~tol ~msg (cjvp_numeric ~h f z v) (to_carr dy)

(* A pair of complex128 tensors, for differentiating binary operations. *)

type cpair = { cfst : Nx.complex128_t; csnd : Nx.complex128_t }

module Cpair = struct
  type t = cpair

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { cfst; csnd } =
    { cfst = f cfst; csnd = f csnd }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { cfst = f p.cfst q.cfst; csnd = f p.csnd q.csnd }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { cfst; csnd } =
    f cfst;
    f csnd
end

(* [check_cgrad2 ~msg f a b] is {!check_cgrad} for a binary operation. Each
   argument is checked against the Jacobian taken with the other one held fixed,
   so a contribution routed to the wrong argument shows up. *)
let check_cgrad2 ?(h = 1e-5) ?(tol = 1e-5) ~msg
    (f : Nx.complex128_t -> Nx.complex128_t -> Nx.complex128_t)
    (a : Nx.complex128_t) (b : Nx.complex128_t) =
  let w = cotangent_like (f a b) in
  let _, g =
    Rune.vjp (module Cpair) (fun p -> f p.cfst p.csnd) { cfst = a; csnd = b } w
  in
  let warr = to_carr w in
  check_cclose ~tol ~msg:(msg ^ ".fst")
    (cvjp_numeric ~h (fun x -> f x b) a warr)
    (to_carr g.cfst);
  check_cclose ~tol ~msg:(msg ^ ".snd")
    (cvjp_numeric ~h (fun y -> f a y) b warr)
    (to_carr g.csnd)

(* [check_cjvp2 ~msg f a b] is {!check_cjvp} for a binary operation. The
   pushforward is linear in the tangents, so the oracle is the sum of the two
   partial directional derivatives. *)
let check_cjvp2 ?(h = 1e-5) ?(tol = 1e-5) ~msg
    (f : Nx.complex128_t -> Nx.complex128_t -> Nx.complex128_t)
    (a : Nx.complex128_t) (b : Nx.complex128_t) =
  let va = ctangent_like a and vb = cotangent_like b in
  let _, dy =
    Rune.jvp
      (module Cpair)
      (fun p -> f p.cfst p.csnd)
      { cfst = a; csnd = b } { cfst = va; csnd = vb }
  in
  let from_a = cjvp_numeric ~h (fun x -> f x b) a va in
  let from_b = cjvp_numeric ~h (fun y -> f a y) b vb in
  check_cclose ~tol ~msg
    (Array.mapi (fun i d -> Complex.add d from_b.(i)) from_a)
    (to_carr dy)
