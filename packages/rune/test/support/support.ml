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

let check_arr ?(eps = 1e-5) ~msg expected actual =
  let actual = to_arr actual in
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i e ->
      equal ~msg:(Printf.sprintf "%s[%d]" msg i) (float eps) e actual.(i))
    expected

let scalar_like (type a b) (t : (a, b) Nx.t) (v : float) : (a, b) Nx.t =
  let dt = Nx.dtype t in
  Nx.full dt [||] (Nx_core.Dtype.of_float dt v)

let as_f32 (type a b) (x : (a, b) Nx.t) : Nx.float32_t =
  match Nx_core.Dtype.equal_witness (Nx.dtype x) f32 with
  | Some Type.Equal -> x
  | None -> failwith "expected a float32 leaf"

let raises_invalid_arg f =
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    f

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
