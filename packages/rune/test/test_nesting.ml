(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Differentiations nested in one another. Each differentiates only the values
   of its own function, whatever the modes: a confused nesting gives a different
   number on the programs below. A custom rule holds at every order of
   differentiation around it. Transformations on two domains at once do not see
   each other. *)

open Windtrap

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let exact () = Oracle.tensor ()
let close () = Oracle.tensor ~rel:1e-12 ()
let one () = vec [| 1. |]

(* The derivative of [f] at [x], by mode: [grad'], [jvp'] along one, or [vjp']'s
   pullback of one. *)
type mode = Grad | Jvp | Pullback

let mode_name = function
  | Grad -> "grad"
  | Jvp -> "jvp"
  | Pullback -> "vjp's pullback"

let derivative mode f x =
  match mode with
  | Grad -> Rune.grad' (fun x -> Nx.sum (f x)) x
  | Jvp -> snd (Rune.jvp' f x (Nx.ones_like x))
  | Pullback -> snd (Rune.vjp' f x) (Nx.ones_like (f x))

let modes = [ Grad; Jvp; Pullback ]

let mode_pairs =
  List.concat_map (fun o -> List.map (fun i -> (o, i)) modes) modes

let pair_name (o, i) = mode_name o ^ " of " ^ mode_name i

(* Perturbations *)

let perturbation_tests =
  [
    cases ~name:pair_name "d/dx [x · d/dy (x + y)] at y = 1 is 1" mode_pairs
      (fun (outer, inner) ->
        let f x = Nx.mul x (derivative inner (fun y -> Nx.add x y) (one ())) in
        equal (exact ()) (one ()) (derivative outer f (one ())));
    cases ~name:pair_name "d/dx [d/dy (x · y)] at y = 1 is 1" mode_pairs
      (fun (outer, inner) ->
        let f x = derivative inner (fun y -> Nx.mul x y) (one ()) in
        equal (exact ()) (one ()) (derivative outer f (vec [| 3. |])));
    test "one function differentiated at two levels at once" (fun () ->
        (* f x = Σ x³; the objective Σᵢ 3xᵢ² · f x has the gradient 6xₖ · f x +
           3xₖ² · Σᵢ 3xᵢ². *)
        let f x = Nx.sum (Nx.mul x (Nx.mul x x)) in
        let x = vec [| 0.5; -1.2 |] in
        let s = (0.5 ** 3.) +. (-1.2 ** 3.)
        and q = (0.5 ** 2.) +. (-1.2 ** 2.) in
        let expected =
          vec
            (Array.map
               (fun x -> (6. *. x *. s) +. (9. *. (x ** 2.) *. q))
               [| 0.5; -1.2 |])
        in
        equal (close ()) expected
          (Rune.grad' (fun x -> Nx.sum (Nx.mul (Rune.grad' f x) (f x))) x));
    test "the second derivative of a cube" (fun () ->
        let cube x = Nx.sum (Nx.mul x (Nx.mul x x)) in
        equal (close ())
          (vec [| 6.; -12.; 18. |])
          (Rune.grad'
             (fun x -> Nx.sum (Rune.grad' cube x))
             (vec [| 1.; -2.; 3. |])));
    test "the third derivative of a fourth power" (fun () ->
        let quart x = Nx.sum (Nx.mul (Nx.mul x x) (Nx.mul x x)) in
        let d1 x = Nx.sum (Rune.grad' quart x) in
        let d2 x = Nx.sum (Rune.grad' d1 x) in
        equal (close ()) (vec [| 48. |]) (Rune.grad' d2 (vec [| 2. |])));
    test "a Hessian-vector product, forward over reverse" (fun () ->
        let cube x = Nx.sum (Nx.mul x (Nx.mul x x)) in
        equal (close ())
          (vec [| 6.; -6.; -18. |])
          (snd
             (Rune.jvp' (Rune.grad' cube)
                (vec [| 1.; -2.; 3. |])
                (vec [| 1.; 0.5; -1. |]))));
    test "a gradient of a tangent, reverse over forward" (fun () ->
        let v = vec [| 1.; 0.5; -1. |] in
        equal (close ())
          (vec [| 2.; 1.; -2. |])
          (Rune.grad'
             (fun x -> snd (Rune.jvp' (fun x -> Nx.sum (Nx.mul x x)) x v))
             (vec [| 1.; -2.; 3. |])));
    test "a tangent of a tangent" (fun () ->
        (* d²(Σ x³)[v, v] = 6 Σ x v². *)
        let cube x = Nx.sum (Nx.mul x (Nx.mul x x)) in
        let v = vec [| 1.; 0.5; -1. |] in
        let inner x = snd (Rune.jvp' cube x v) in
        equal (close ()) (scalar 21.)
          (snd (Rune.jvp' inner (vec [| 1.; -2.; 3. |]) v)));
  ]

(* Custom rules at every order *)

let sigmoid0 = 0.5
and sigmoid'0 = 0.25

(* The softplus Hessian at 0: d²/dx² (softplus x)² = 2σ² + 2 softplus · σ'. *)
let softplus_hessian_at_0 =
  (2. *. sigmoid0 *. sigmoid0) +. (2. *. Float.log 2. *. sigmoid'0)

let stable x =
  Nx.add (Nx.relu x) (Nx.log (Nx.add_s (Nx.exp (Nx.neg (Nx.abs x))) 1.))

let softplus =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
      (stable x, fun dx -> Nx.mul (Nx.sigmoid x) dx))

let squared x = Nx.sum (Nx.square (softplus x))

(* A rule whose tangent map is twice the function's derivative, so that an order
   that applies the rule differs from one that differentiates the function's
   code. *)
let doubled =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
      (Nx.sin x, fun dx -> Nx.mul (Nx.mul_s (Nx.cos x) 2.) dx))

let total : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

(* A rule with no tensor in its result, whose tangent map adds the sum of the
   tangents to [total]. *)
let mark =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.unit (fun _ ->
      ((), fun dy -> Rune.Total.add total (Nx.sum dy)))

let custom_order_tests =
  let x0 = scalar 0. and x = 0.4 in
  [
    test "the softplus Hessian at 0, forward over reverse" (fun () ->
        equal (close ())
          (scalar softplus_hessian_at_0)
          (Rune.jacfwd' (Rune.grad' squared) x0));
    test "the softplus Hessian at 0, reverse over reverse" (fun () ->
        equal (close ())
          (scalar softplus_hessian_at_0)
          (Rune.grad' (fun x -> Nx.sum (Rune.grad' squared x)) x0));
    test "the softplus Hessian at 0, jvp over reverse" (fun () ->
        equal (close ())
          (scalar softplus_hessian_at_0)
          (snd (Rune.jvp' (Rune.grad' squared) x0 (scalar 1.))));
    test "the softplus Hessian at 0, forward over forward" (fun () ->
        equal (close ())
          (scalar softplus_hessian_at_0)
          (snd
             (Rune.jvp'
                (fun x -> snd (Rune.jvp' squared x (scalar 1.)))
                x0 (scalar 1.))));
    test "the first derivative is the rule's" (fun () ->
        equal ~msg:"grad" (close ())
          (scalar (2. *. Float.cos x))
          (Rune.grad' (fun x -> Nx.sum (doubled x)) (scalar x));
        equal ~msg:"jvp" (close ())
          (scalar (2. *. Float.cos x))
          (snd (Rune.jvp' doubled (scalar x) (scalar 1.))));
    test "an enclosing differentiation applies the rule to the inner primal"
      (fun () ->
        equal (close ())
          (scalar (2. *. Float.cos x *. 3.))
          (snd
             (Rune.jvp'
                (fun x -> fst (Rune.jvp' doubled x (scalar 1.)))
                (scalar x) (scalar 3.))));
    test "the second derivative differentiates the tangent map's code"
      (fun () ->
        equal ~msg:"forward over reverse" (close ())
          (scalar (-2. *. Float.sin x))
          (Rune.jacfwd' (Rune.grad' (fun x -> Nx.sum (doubled x))) (scalar x));
        equal ~msg:"reverse over reverse" (close ())
          (scalar (-2. *. Float.sin x))
          (Rune.grad'
             (fun x -> Nx.sum (Rune.grad' (fun x -> Nx.sum (doubled x)) x))
             (scalar x)));
    test "a rule with no tensor result is not applied under grad" (fun () ->
        let g, t =
          Rune.Total.collect total ~zero:(scalar 0.) (fun () ->
              Rune.grad'
                (fun x ->
                  mark x;
                  Nx.sum (Nx.mul x x))
                (vec [| 1.; 2. |]))
        in
        equal ~msg:"gradient" (exact ()) (vec [| 2.; 4. |]) g;
        equal ~msg:"total" (exact ()) (scalar 0.) t);
    test "a rule with no tensor result observes the tangents under jvp"
      (fun () ->
        let _, t =
          Rune.Total.collect total ~zero:(scalar 0.) (fun () ->
              Rune.jvp'
                (fun x ->
                  mark x;
                  Nx.sum (Nx.mul x x))
                (vec [| 1.; 2. |])
                (vec [| 3.; -1. |]))
        in
        equal (exact ()) (scalar 2.) t);
  ]

(* Independence *)

(* A curvature mark: a rule with no tensor result whose tangent map gathers the
   tangents of every direction of a map and adds their Gram matrix to a
   total. *)
let directions = Rune.axis ()
let curvature : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

let curvature_mark =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.unit (fun _ ->
      ( (),
        fun dy ->
          let g = Rune.lanes directions dy in
          Rune.Total.add curvature (Nx.matmul g (Nx.matrix_transpose g)) ))

(* y = c ⊙ θ, marked, and a loss of it. *)
let c () = vec [| 0.5; -2.; 1.5 |]

let marked_loss theta =
  let y = Nx.mul (c ()) theta in
  curvature_mark y;
  Nx.sum (Nx.square y)

let theta () = vec [| 0.3; 1.1; -0.7 |]
let dirs () = Nx.create f64 [| 2; 3 |] [| 1.; 0.; 2.; -1.; 3.; 0.5 |]

let mark_tests =
  [
    test "a curvature mark sums the directions' Gram matrix in every lane"
      (fun () ->
        let zero = Nx.zeros f64 [| 2; 2 |] in
        let _, totals =
          Rune.vmap ~axis:directions
            Nx.Ptree.(tensor @-> returns (pair (pair tensor tensor) tensor))
            (fun dir ->
              Rune.Total.collect curvature ~zero (fun () ->
                  Rune.jvp' marked_loss (theta ()) dir))
            (dirs ())
        in
        (* G stacks the directions' tangents of y, c ⊙ dir. *)
        let g = Nx.mul (dirs ()) (c ()) in
        let gram = Nx.matmul g (Nx.matrix_transpose g) in
        equal (close ()) (Nx.stack [ gram; gram ]) totals);
    test "a curvature mark is inert under value_and_grad" (fun () ->
        let (_, g), total =
          Rune.Total.collect curvature
            ~zero:(Nx.zeros f64 [| 2; 2 |])
            (fun () -> Rune.value_and_grad' marked_loss (theta ()))
        in
        equal ~msg:"gradient" (close ())
          (Nx.mul_s (Nx.mul (Nx.square (c ())) (theta ())) 2.)
          g;
        equal ~msg:"total" (exact ()) (Nx.zeros f64 [| 2; 2 |]) total);
  ]

let independence_tests =
  [
    test "differentiations on two domains at once" (fun () ->
        let f x = Nx.sum (Nx.mul (Nx.sin x) x) in
        let x = vec [| 0.3; -1.1; 2. |] and v = vec [| 1.; 0.5; -2. |] in
        let g = Rune.grad' f x and dy = snd (Rune.jvp' f x v) in
        let reverse =
          Domain.spawn (fun () -> List.init 50 (fun _ -> Rune.grad' f x))
        and forward =
          Domain.spawn (fun () -> List.init 50 (fun _ -> snd (Rune.jvp' f x v)))
        in
        List.iter (equal ~msg:"grad" (exact ()) g) (Domain.join reverse);
        List.iter (equal ~msg:"jvp" (exact ()) dy) (Domain.join forward));
    test "a domain started inside grad computes without it" (fun () ->
        let inside = ref (vec [||]) in
        ignore
          (Rune.grad'
             (fun x ->
               inside :=
                 Domain.join
                   (Domain.spawn (fun () -> Nx.mul_s (vec [| 2. |]) 3.));
               Nx.sum x)
             (vec [| 1. |]));
        equal (exact ()) (vec [| 6. |]) !inside);
    test "two totals are distinct" (fun () ->
        let other : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make () in
        let (_, inner), outer =
          Rune.Total.collect total ~zero:(scalar 0.) (fun () ->
              Rune.Total.collect other ~zero:(scalar 0.) (fun () ->
                  Rune.Total.add total (scalar 1.);
                  Rune.Total.add other (scalar 10.)))
        in
        equal ~msg:"other" (exact ()) (scalar 10.) inner;
        equal ~msg:"total" (exact ()) (scalar 1.) outer);
    test "two axes are distinct" (fun () ->
        let b = Rune.axis () and c = Rune.axis () in
        let y =
          Rune.vmap' ~axis:b
            (fun x -> Nx.sum ~axes:[ 0 ] (Rune.lanes c x))
            (vec [| 1.; 2.; 3. |])
        in
        (* No map named c: one lane, so each lane sees its own x. *)
        equal (exact ()) (vec [| 1.; 2.; 3. |]) y);
  ]

let () =
  exit
    (run "Rune nesting"
       [
         group "perturbation" perturbation_tests;
         group "custom rules at every order" custom_order_tests;
         group "a curvature mark" mark_tests;
         group "independence" independence_tests;
       ])
