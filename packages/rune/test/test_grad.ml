(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Transformation semantics over structures: grad, value_and_grad,
   value_and_grad_aux, vjp, vjp_fun and remat on user-defined records. *)

open Windtrap
open Rune_test_support.Support

(* loss p = sum (w * w) + 3 * sum b, ignoring scale. d/dw = 2w, d/db = 3,
   d/dscale = 0. *)
let quadratic p = Nx.add (Nx.sum (Nx.mul p.w p.w)) (Nx.mul_s (Nx.sum p.b) 3.0)

let test_grad_record_analytic () =
  let g = Rune.grad params_ptree quadratic (params ()) in
  check_arr ~msg:"dw" [| 2.0; -4.0; 6.0 |] g.w;
  check_arr ~msg:"db" [| 3.0 |] g.b

let test_grad_unused_leaf_zero () =
  let g = Rune.grad params_ptree quadratic (params ()) in
  check_arr ~msg:"dscale" [| 0.0 |] g.scale

let test_grad_preserves_structure () =
  let p = params () in
  let g = Rune.grad params_ptree quadratic p in
  equal ~msg:"w shape" (array int) (Nx.shape p.w) (Nx.shape g.w);
  equal ~msg:"b shape" (array int) (Nx.shape p.b) (Nx.shape g.b);
  equal ~msg:"scale shape" (array int) (Nx.shape p.scale) (Nx.shape g.scale)

let test_value_and_grad_value () =
  let v, _ = Rune.value_and_grad params_ptree quadratic (params ()) in
  (* 1 + 4 + 9 + 3 * 0.5 = 15.5 *)
  check_arr ~msg:"value" [| 15.5 |] v

let test_value_and_grad_aux () =
  let f p = (quadratic p, "aux") in
  let v, g, aux = Rune.value_and_grad_aux params_ptree f (params ()) in
  check_arr ~msg:"value" [| 15.5 |] v;
  check_arr ~msg:"dw" [| 2.0; -4.0; 6.0 |] g.w;
  equal ~msg:"aux" string "aux" aux

let test_vjp_cotangent_scales () =
  let _, g =
    Rune.vjp params_ptree Nx.Ptree.tensor quadratic (params ())
      (Nx.scalar f32 2.0)
  in
  check_arr ~msg:"dw scaled" [| 4.0; -8.0; 12.0 |] g.w

let test_vjp_non_scalar_output () =
  (* vjp accepts non-scalar outputs: for f(w) = w*w and cotangent ct, the
     pulled-back cotangent is 2*w*ct. *)
  let p = params () in
  let f p = Nx.mul p.w p.w in
  let _, g =
    Rune.vjp params_ptree Nx.Ptree.tensor f p (vec32 [| 1.0; 2.0; 3.0 |])
  in
  check_arr ~msg:"dw" [| 2.0; -8.0; 18.0 |] g.w

let test_mixed_dtype_single_pass () =
  (* loss p = sum (w * w) + sum (scale * scale), via a cast to float64. d/dw =
     2w (float32), d/dscale = 2 scale (float64). *)
  let f p =
    Nx.add
      (Nx.cast f64 (Nx.sum (Nx.mul p.w p.w)))
      (Nx.sum (Nx.mul p.scale p.scale))
  in
  let g = Rune.grad params_ptree f (params ()) in
  check_arr ~msg:"dw" [| 2.0; -4.0; 6.0 |] g.w;
  check_arr ~msg:"dscale" [| 4.0 |] g.scale

let test_gradient_descent_converges () =
  (* Minimize sum (w * w) + sum (scale * scale): both go to zero. *)
  let f p =
    Nx.add
      (Nx.cast f64 (Nx.sum (Nx.mul p.w p.w)))
      (Nx.sum (Nx.mul p.scale p.scale))
  in
  let step p =
    let g = Rune.grad params_ptree f p in
    Nx.Ptree.map2 params_ptree
      (fun _ p g -> Nx.sub p (Nx.mul g (scalar_like g 0.1)))
      p g
  in
  let p = ref (params ()) in
  for _ = 1 to 100 do
    p := step !p
  done;
  check_arr ~msg:"w" [| 0.0; 0.0; 0.0 |] !p.w;
  check_arr ~msg:"scale" [| 0.0 |] !p.scale

let test_grad_single_tensor () =
  let f x = Nx.sum (Nx.mul x x) in
  let g = Rune.grad' f (vec32 [| 1.0; -2.0; 3.0 |]) in
  check_arr ~msg:"dx" [| 2.0; -4.0; 6.0 |] g

(* A bitcast has zero derivative, as the bitwise operations do: the gradient
   flows only through the other use of [x]. *)
let test_bitcast_has_zero_derivative () =
  let f x = Nx.add x (Nx.cast f32 (Nx.bitcast Nx.int32 x)) in
  let x = vec32 [| 1.0; -2.0; 3.0 |] in
  check_arr ~msg:"reverse" [| 1.0; 1.0; 1.0 |]
    (Rune.grad' (fun x -> Nx.sum (f x)) x);
  check_arr ~msg:"forward" [| 0.5; 0.5; 0.5 |]
    (snd (Rune.jvp' f x (vec32 [| 0.5; 0.5; 0.5 |])))

let test_vjp_single_tensor () =
  let f x = Nx.mul x x in
  let _, g = Rune.vjp' f (vec32 [| 1.0; 2.0 |]) (vec32 [| 10.0; 1.0 |]) in
  check_arr ~msg:"dx" [| 20.0; 4.0 |] g

let test_vjp_structured_output () =
  (* vjp against cotangents equals the gradient of <cotangents, f>. *)
  let a = vec64 [| 0.7; -1.3; 2.1 |] and b = vec64 [| 1.9; 0.8; -0.6 |] in
  let ca = vec64 [| 1.0; -2.0; 0.5 |] and cb = vec64 [| 0.3; 1.1; -0.7 |] in
  let f p = { fst = Nx.mul p.fst p.snd; snd = Nx.add p.fst p.snd } in
  let _, g =
    Rune.vjp pair_ptree pair_ptree f { fst = a; snd = b } { fst = ca; snd = cb }
  in
  let dotted p =
    let y = f p in
    Nx.add (Nx.sum (Nx.mul y.fst ca)) (Nx.sum (Nx.mul y.snd cb))
  in
  let expected = Rune.grad pair_ptree dotted { fst = a; snd = b } in
  check_arr ~msg:"da" (to_arr expected.fst) g.fst;
  check_arr ~msg:"db" (to_arr expected.snd) g.snd

let test_vjp_cotangent_shape_mismatch () =
  raises
    (Invalid_argument
       "Rune.vjp: fst: cotangent shape [1] does not match result shape [2]")
    (fun () ->
      ignore
        (Rune.vjp pair_ptree pair_ptree
           (fun p -> p)
           { fst = vec64 [| 1.0; 2.0 |]; snd = vec64 [| 3.0 |] }
           { fst = vec64 [| 1.0 |]; snd = vec64 [| 1.0 |] }))

(* Cotangents of another structure than the result name the first path where
   they differ. *)
let test_vjp_cotangent_structure_mismatch () =
  let result = Nx.Ptree.(list tensor) in
  raises
    (Invalid_argument
       "Rune.vjp: the root: length 2 in the result, length 1 in the cotangents")
    (fun () ->
      ignore
        (Rune.vjp Nx.Ptree.tensor result
           (fun x -> [ x; Nx.mul x x ])
           (vec64 [| 1.0 |])
           [ vec64 [| 1.0 |] ]))

let test_vjp_fun_structured () =
  let a = vec64 [| 0.7; -1.3 |] and b = vec64 [| 1.9; 0.8 |] in
  let f p = (Nx.mul p.fst p.snd, Nx.sum p.fst) in
  let result = Nx.Ptree.(pair tensor tensor) in
  let (y, s), pullback =
    Rune.vjp_fun pair_ptree result f { fst = a; snd = b }
  in
  check_arr ~msg:"y" [| 0.7 *. 1.9; -1.3 *. 0.8 |] y;
  check_arr ~msg:"s" [| 0.7 -. 1.3 |] s;
  let g = pullback (vec64 [| 1.0; 0.0 |], Nx.scalar f64 2.0) in
  check_arr ~msg:"d fst" [| 1.9 +. 2.0; 2.0 |] g.fst;
  check_arr ~msg:"d snd" [| 0.7; 0.0 |] g.snd

let test_remat_same_gradient () =
  (* remat changes memory behavior, never values or gradients. *)
  let f p = Nx.sum (Nx.mul (Nx.exp p.fst) (Nx.sin p.snd)) in
  let params =
    { fst = vec64 [| 0.7; -1.3; 2.1 |]; snd = vec64 [| 1.9; 0.8; -0.6 |] }
  in
  let g = Rune.grad pair_ptree f params in
  let g' =
    Rune.grad pair_ptree
      (fun p -> Rune.remat Nx.Ptree.(pair_ptree @-> returns tensor) f p)
      params
  in
  check_arr ~msg:"d fst" (to_arr g.fst) g'.fst;
  check_arr ~msg:"d snd" (to_arr g.snd) g'.snd

let test_remat_value () =
  let f p = Nx.sum (Nx.mul p.fst p.snd) in
  let params =
    { fst = vec64 [| 0.7; -1.3; 2.1 |]; snd = vec64 [| 1.9; 0.8; -0.6 |] }
  in
  check_arr ~msg:"value"
    (to_arr (f params))
    (Rune.remat Nx.Ptree.(pair_ptree @-> returns tensor) f params)

(* remat takes a curried function of several arguments and a structured result,
   as its signature describes. *)
let test_remat_signature () =
  let block x y = (Nx.mul (Nx.exp x) y, Nx.sin y) in
  let s = Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor)) in
  let loss block p =
    let u, v = block p.fst p.snd in
    Nx.add (Nx.sum u) (Nx.sum (Nx.mul v v))
  in
  let params = { fst = vec64 [| 0.7; -1.3 |]; snd = vec64 [| 1.9; 0.8 |] } in
  let g = Rune.grad pair_ptree (loss block) params in
  let g' = Rune.grad pair_ptree (loss (Rune.remat s block)) params in
  check_arr ~msg:"d fst" (to_arr g.fst) g'.fst;
  check_arr ~msg:"d snd" (to_arr g.snd) g'.snd

(* A result that is one of the arguments has the result's cotangent alone. *)
let test_remat_returns_an_argument () =
  let x = vec64 [| 0.5; -1.0 |] in
  let s = Nx.Ptree.(tensor @-> returns (pair tensor tensor)) in
  let g =
    Rune.grad'
      (fun x ->
        let y, z = Rune.remat s (fun x -> (x, Nx.mul x x)) x in
        Nx.add (Nx.sum y) (Nx.sum z))
      x
  in
  check_arr ~msg:"1 + 2x" [| 2.0; -1.0 |] g

let test_remat_rejects_consumes () =
  raises
    (Invalid_argument
       "Rune.remat: the argument at 1 is consumed; only a compiled call \
        consumes its arguments") (fun () ->
      let (_ : Nx.float64_t -> Nx.float64_t -> Nx.float64_t) =
        Rune.remat
          Nx.Ptree.(tensor @-> consumes tensor @@ returns tensor)
          Nx.add
      in
      ())

(* A block whose backward pass reads its intermediates, and a point and a
   direction to differentiate it at. *)
let remat_block x = Nx.sum (Nx.mul (Nx.tanh (Nx.mul x x)) (Nx.exp x))
let rematted = Rune.remat Nx.Ptree.(tensor @-> returns tensor) remat_block
let at () = vec64 [| 0.7; -1.3; 2.1 |]
let along () = vec64 [| 0.5; 1.0; -2.0 |]

let test_remat_under_jvp () =
  let y, dy = Rune.jvp' remat_block (at ()) (along ()) in
  let y', dy' = Rune.jvp' rematted (at ()) (along ()) in
  check_arr ~msg:"value" (to_arr y) y';
  check_arr ~msg:"tangent" (to_arr dy) dy'

(* Rows of [xs] are the lanes. *)
let test_remat_under_vmap () =
  let xs () = Nx.create f64 [| 2; 3 |] [| 0.7; -1.3; 2.1; 0.2; 0.9; -0.4 |] in
  let lanes f x = Nx.sum (Rune.vmap' f x) in
  check_arr ~msg:"values"
    (to_arr (Rune.vmap' remat_block (xs ())))
    (Rune.vmap' rematted (xs ()));
  check_arr ~msg:"grad of vmap"
    (to_arr (Rune.grad' (lanes remat_block) (xs ())))
    (Rune.grad' (lanes rematted) (xs ()));
  check_arr ~msg:"vmap of grad"
    (to_arr (Rune.vmap' (Rune.grad' remat_block) (xs ())))
    (Rune.vmap' (Rune.grad' rematted) (xs ()));
  check_arr ~msg:"compiled grad of vmap"
    (to_arr (Rune.grad' (lanes remat_block) (xs ())))
    (Rune.jit' (Rune.grad' (lanes rematted)) (xs ()));
  check_arr ~msg:"compiled vmap of grad"
    (to_arr (Rune.vmap' (Rune.grad' remat_block) (xs ())))
    (Rune.jit' (Rune.vmap' (Rune.grad' rematted)) (xs ()))

let test_remat_under_jit () =
  let f p = Nx.sum (Nx.mul (Nx.exp p.fst) (Nx.sin (Nx.mul p.snd p.fst))) in
  let params =
    { fst = vec64 [| 0.7; -1.3; 2.1 |]; snd = vec64 [| 1.9; 0.8; -0.6 |] }
  in
  let g = Rune.grad pair_ptree f params in
  let g' =
    Rune.jit
      Nx.Ptree.(pair_ptree @-> returns pair_ptree)
      (Rune.grad pair_ptree
         (Rune.remat Nx.Ptree.(pair_ptree @-> returns tensor) f))
      params
  in
  check_arr ~msg:"d fst" (to_arr g.fst) g'.fst;
  check_arr ~msg:"d snd" (to_arr g.snd) g'.snd

(* Second derivatives differentiate the recomputation, and under jit its
   barrier. *)
let test_remat_second_order_under_jit () =
  let hvp f x =
    Rune.grad' (fun x -> Nx.sum (Nx.mul (Rune.grad' f x) (along ()))) x
  in
  check_arr ~msg:"reverse over reverse"
    (to_arr (hvp remat_block (at ())))
    (Rune.jit' (hvp rematted) (at ()));
  let fwd f x = snd (Rune.jvp' (Rune.grad' f) x (along ())) in
  check_arr ~msg:"forward over reverse"
    (to_arr (fwd remat_block (at ())))
    (Rune.jit' (fwd rematted) (at ()))

(* A layer that closes over its weight [w]: every transformation reaches the
   capture through the remat, eagerly and compiled. *)
let layer w x = Nx.mul (Nx.exp x) w
let rematted_layer = Rune.remat Nx.Ptree.(tensor @-> returns tensor)

let check_eager_and_jit ~msg f g x =
  check_arr ~msg:(msg ^ ", eager") (to_arr (f x)) (g x);
  check_arr ~msg:(msg ^ ", compiled") (to_arr (f x)) (Rune.jit' g x)

let test_remat_grad_of_capture () =
  let loss r w = Nx.sum (Nx.sin (r (layer w) (at ()))) in
  check_eager_and_jit ~msg:"d w"
    (Rune.grad' (loss (fun f -> f)))
    (Rune.grad' (loss rematted_layer))
    (along ())

let test_remat_jvp_of_capture () =
  let fn r w = r (layer w) (at ()) in
  let tangent r w = snd (Rune.jvp' (fn r) w (along ())) in
  check_eager_and_jit ~msg:"tangent"
    (tangent (fun f -> f))
    (tangent rematted_layer) (at ())

(* [w] is captured and feeds the argument: both shares reach it. *)
let test_remat_grad_of_capture_and_argument () =
  let loss r w = Nx.sum (r (layer w) (Nx.mul w w)) in
  check_eager_and_jit ~msg:"d w"
    (Rune.grad' (loss (fun f -> f)))
    (Rune.grad' (loss rematted_layer))
    (along ())

(* Second derivatives with respect to weights that are inputs of the compiled
   function, passed to the remat or captured by it. *)
let test_remat_hvp_of_weights () =
  let hvp loss w =
    Rune.grad' (fun w -> Nx.sum (Nx.mul (Rune.grad' loss w) (along ()))) w
  in
  let passed r w =
    Nx.sum (Nx.sin (r (fun w x -> Nx.tanh (Nx.mul x w)) w (Nx.cos (at ()))))
  in
  check_eager_and_jit ~msg:"passed"
    (hvp (passed (fun f -> f)))
    (hvp (passed (Rune.remat Nx.Ptree.(tensor @-> tensor @-> returns tensor))))
    (along ());
  let captured r w =
    Nx.sum (Nx.sin (r (fun x -> Nx.mul (Nx.exp x) (Nx.mul w w)) (at ())))
  in
  check_eager_and_jit ~msg:"captured"
    (hvp (captured (fun f -> f)))
    (hvp (captured rematted_layer))
    (along ())

(* The argument is the tensor the function captures. *)
let test_remat_argument_also_captured () =
  let loss r w = Nx.sum (r (fun x -> Nx.mul (Nx.sin x) w) w) in
  check_eager_and_jit ~msg:"d w"
    (Rune.grad' (loss (fun f -> f)))
    (Rune.grad' (loss rematted_layer))
    (along ())

(* The lane's row is captured and the argument is a constant of the map. *)
let test_remat_batched_capture () =
  let xs () = Nx.create f64 [| 2; 3 |] [| 0.7; -1.3; 2.1; 0.2; 0.9; -0.4 |] in
  let lane r x = r (fun c -> Nx.mul (Nx.sin c) x) (along ()) in
  check_eager_and_jit ~msg:"values"
    (Rune.vmap' (lane (fun f -> f)))
    (Rune.vmap' (lane rematted_layer))
    (xs ());
  let lanes r xs = Nx.sum (Rune.vmap' (lane r) xs) in
  check_eager_and_jit ~msg:"grad of vmap"
    (Rune.grad' (lanes (fun f -> f)))
    (Rune.grad' (lanes rematted_layer))
    (xs ())

(* A single tensor has nowhere to carry a non-differentiable value, so an
   integer argument there is simply the wrong dtype. *)
let test_grad_rejects_integer_leaves () =
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (Rune.grad'
           (fun x -> Nx.sum x)
           (Nx.create Nx.int32 [| 2 |] [| 1l; 2l |])))

(* A structure does have somewhere. The canonical case is an RNG key, which must
   sit in the structure to reach a compiled step as an input, but is not
   something to differentiate. Such a leaf is carried: zero in the gradient,
   untouched by the optimizer, while the real parameters update. Before this,
   grad refused the whole structure and the key had to be captured in a closure
   and the parameters kept in a second structure. *)
module Stepper = struct
  type stepper = { w : Nx.float32_t; key : Nx.Rng.t }
  type _ t = stepper

  let walk c { w; key } =
    let open Nx.Ptree.Walk in
    let w = field c "w" tensor w in
    let key = field c "key" (structure Nx.Rng.ptree) key in
    { w; key }
end

let stepper_ptree = Nx.Ptree.instantiate (module Stepper)

let stepper () =
  Stepper.{ w = Nx.create f32 [| 3 |] [| 1.0; -2.0; 3.0 |]; key = Nx.Rng.key 7 }

(* The draw is a constant of the tape, so the loss below has the same gradient
   it would without the mask, scaled by it. *)
let test_grad_carries_a_key_leaf () =
  let p = stepper () in
  let g =
    Rune.grad stepper_ptree (fun p -> Nx.sum (Nx.mul p.Stepper.w p.Stepper.w)) p
  in
  check_arr ~msg:"the float leaf differentiates" [| 2.0; -4.0; 6.0 |]
    g.Stepper.w;
  equal ~msg:"the key's gradient is zero" (array int32) [| 0l; 0l |]
    (Nx.to_array (g.Stepper.key :> Nx.int32_t))

(* One tensor behind both leaves. A structure is positional: each leaf is its
   own parameter with its own gradient (and, in forward mode, its own tangent),
   not one tied parameter reported twice. *)
let test_aliased_leaves_are_separate_parameters () =
  let x = vec64 [| 1.0; 2.0 |] in
  let f p = Nx.add (Nx.sum p.fst) (Nx.mul_s (Nx.sum p.snd) 3.0) in
  let g = Rune.grad pair_ptree f { fst = x; snd = x } in
  check_arr ~msg:"d/dfst" [| 1.0; 1.0 |] g.fst;
  check_arr ~msg:"d/dsnd" [| 3.0; 3.0 |] g.snd;
  let _, dy =
    Rune.jvp pair_ptree Nx.Ptree.tensor f { fst = x; snd = x }
      { fst = vec64 [| 1.0; 0.0 |]; snd = vec64 [| 0.0; 1.0 |] }
  in
  (* 1 * 1 + 3 * 1: each leaf contributes along its own tangent. *)
  check_arr ~msg:"jvp" [| 4.0 |] dy

(* A capture that is the same value as an argument is a constant of the
   transformation: each argument tensor is replaced by a fresh alias before it
   is tracked. *)
let test_capture_of_the_argument_is_constant () =
  let w = vec64 [| 2.0; -3.0 |] in
  check_arr ~msg:"grad'" [| 2.0; -3.0 |]
    (Rune.grad' (fun x -> Nx.sum (Nx.mul x w)) w);
  check_arr ~msg:"vjp'" [| 2.0; -3.0 |]
    (snd (Rune.vjp' (fun x -> Nx.mul x w) w (vec64 [| 1.0; 1.0 |])));
  let g =
    Rune.grad pair_ptree
      (fun p -> Nx.sum (Nx.mul (Nx.mul p.fst p.snd) w))
      { fst = w; snd = vec64 [| 1.0; 1.0 |] }
  in
  check_arr ~msg:"grad over a record" [| 2.0; -3.0 |] g.fst;
  let _, dy = Rune.jvp' (fun x -> Nx.mul x w) w (vec64 [| 1.0; 1.0 |]) in
  check_arr ~msg:"jvp'" [| 2.0; -3.0 |] dy

(* The functional update differentiates through both operands: the window
   shadows the template, and the value receives the window of the cotangent,
   summed over the axes it was broadcast along. *)
let test_set_grad_both_operands () =
  let c = vec32 [| 1.0; 2.0; 3.0; 4.0 |] in
  let t = vec32 [| 0.0; 0.0; 0.0; 0.0 |] and v = vec32 [| 5.0; 6.0 |] in
  let by_t t = Nx.sum (Nx.mul c (Nx.set [ Nx.R (1, 3) ] v t)) in
  check_arr ~msg:"dt: the window is shadowed" [| 1.0; 0.0; 0.0; 4.0 |]
    (Rune.grad' by_t t);
  let by_v v = Nx.sum (Nx.mul c (Nx.set [ Nx.R (1, 3) ] v t)) in
  check_arr ~msg:"dv: the window of the cotangent" [| 2.0; 3.0 |]
    (Rune.grad' by_v v);
  let by_s s = Nx.sum (Nx.mul c (Nx.set [ Nx.R (1, 3) ] s t)) in
  check_arr ~msg:"a broadcast value sums its window" [| 5.0 |]
    (Rune.grad' by_s (Nx.scalar f32 1.0));
  let pos = Nx.scalar Nx.int32 2l in
  let by_v_at v = Nx.sum (Nx.mul c (Nx.set [ Nx.D (pos, 2) ] v t)) in
  check_arr ~msg:"dv through a run-time start" [| 3.0; 4.0 |]
    (Rune.grad' by_v_at v);
  let by_t_at t = Nx.sum (Nx.mul c (Nx.set [ Nx.D (pos, 2) ] v t)) in
  check_arr ~msg:"dt through a run-time start" [| 1.0; 2.0; 0.0; 0.0 |]
    (Rune.grad' by_t_at t)

let tests =
  [
    group "grad over records"
      [
        test "aliased leaves are separate parameters"
          test_aliased_leaves_are_separate_parameters;
        test "a capture of the argument is a constant"
          test_capture_of_the_argument_is_constant;
        test "matches the analytic gradient" test_grad_record_analytic;
        test "unused leaf has zero gradient" test_grad_unused_leaf_zero;
        test "preserves structure and shapes" test_grad_preserves_structure;
        test "value_and_grad returns the value" test_value_and_grad_value;
        test "value_and_grad_aux returns auxiliary data" test_value_and_grad_aux;
        test "mixed dtypes differentiate in one pass"
          test_mixed_dtype_single_pass;
        test "gradient descent converges" test_gradient_descent_converges;
        test "rejects an integer single-tensor argument"
          test_grad_rejects_integer_leaves;
        test "carries a non-differentiable leaf" test_grad_carries_a_key_leaf;
      ];
    group "vjp"
      [
        test "scales by the cotangent" test_vjp_cotangent_scales;
        test "accepts non-scalar outputs" test_vjp_non_scalar_output;
        test "pulls back structured cotangents" test_vjp_structured_output;
        test "rejects a cotangent shape mismatch"
          test_vjp_cotangent_shape_mismatch;
        test "rejects cotangents of another structure"
          test_vjp_cotangent_structure_mismatch;
        test "vjp_fun pulls back a structured result" test_vjp_fun_structured;
      ];
    group "remat"
      [
        test "gradients are unchanged" test_remat_same_gradient;
        test "values are unchanged" test_remat_value;
        test "takes a signature" test_remat_signature;
        test "a returned argument is not counted twice"
          test_remat_returns_an_argument;
        test "rejects a consumed argument" test_remat_rejects_consumes;
        test "is its function under jvp" test_remat_under_jvp;
        test "composes with vmap" test_remat_under_vmap;
        test "gradients are unchanged under jit" test_remat_under_jit;
        test "second derivatives are unchanged under jit"
          test_remat_second_order_under_jit;
        test "differentiates a captured tensor" test_remat_grad_of_capture;
        test "pushes forward a captured tensor's tangent"
          test_remat_jvp_of_capture;
        test "differentiates a tensor both captured and passed"
          test_remat_grad_of_capture_and_argument;
        test "maps a batched capture" test_remat_batched_capture;
        test "second derivatives with respect to weights under jit"
          test_remat_hvp_of_weights;
        test "differentiates an argument it also captures"
          test_remat_argument_also_captured;
      ];
    group "set"
      [ test "differentiates both operands" test_set_grad_both_operands ];
    group "single-tensor variants"
      [
        test "grad' matches the analytic gradient" test_grad_single_tensor;
        test "vjp' pulls back the cotangent" test_vjp_single_tensor;
        test "a bitcast has zero derivative" test_bitcast_has_zero_derivative;
      ];
  ]

let () = run "rune grad" tests
