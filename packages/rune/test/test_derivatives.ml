(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The differentiation entry points: what each returns, the structures it takes,
   the values that may and may not leave a differentiated function, pullbacks
   applied anywhere, numerics at the edges, the forms of one tensor, and the
   errors a tangent map raises. Expected values are analytic, or come from a
   central difference or the other mode on generated programs. *)

open Windtrap

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let exact () = Oracle.tensor ()
let close () = Oracle.tensor ~rel:1e-12 ()
let near rel = Oracle.tensor ~rel ()

let escape =
  "a traced tensor has no bytes; it was used outside the trace that made it"

let starts fn m = starts_with ~affix:(fn ^ ": ") m

(* A record of parameters: two float leaves and a carried integer one. *)
type 'a params = { w : 'a; b : 'a; steps : Nx.int32_t }

module Params = struct
  type 'a t = 'a params

  let walk c { w; b; steps } =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let b = field c "b" leaf b in
    let steps = field c "steps" tensor steps in
    { w; b; steps }
end

let params_s : Nx.float64_t params Nx.Ptree.t =
  Nx.Ptree.instantiate (module Params)

let params () =
  {
    w = vec [| 1.; -2.; 3. |];
    b = vec [| 0.5 |];
    steps = Nx.create Nx.int32 [| 2 |] [| 7l; 9l |];
  }

(* sum (w·w) + 3 sum b, whose gradient is (2w, 3, 0). *)
let quadratic p = Nx.add (Nx.sum (Nx.mul p.w p.w)) (Nx.mul_s (Nx.sum p.b) 3.)
let pair = Nx.Ptree.(pair tensor tensor)

(* One tensor of any dtype: the structure through which two values of one
   structure can differ in dtype. *)
module Packed = struct
  type _ t = Nx.packed

  let walk c (Nx.P x) = Nx.P (Nx.Ptree.Walk.tensor c x)
end

let packed = Nx.Ptree.instantiate (module Packed)

(* grad *)

let grad_tests =
  [
    test "the gradient of a record is its analytic gradient" (fun () ->
        let g = Rune.grad params_s quadratic (params ()) in
        equal ~msg:"w" (close ()) (vec [| 2.; -4.; 6. |]) g.w;
        equal ~msg:"b" (close ()) (vec [| 3. |]) g.b);
    test "leaves of two dtypes differentiate in one pass" (fun () ->
        let w = Nx.create Nx.float32 [| 2 |] [| 1.; -2. |]
        and s = vec [| 3. |] in
        let gw, gs =
          Rune.grad
            Nx.Ptree.(pair tensor tensor)
            (fun (w, s) ->
              Nx.add (Nx.cast f64 (Nx.sum (Nx.mul w w))) (Nx.sum (Nx.mul s s)))
            (w, s)
        in
        equal ~msg:"float32" (exact ())
          (Nx.create Nx.float32 [| 2 |] [| 2.; -4. |])
          gw;
        equal ~msg:"float64" (exact ()) (vec [| 6. |]) gs);
    test "a carried integer leaf has a zero gradient of its dtype and shape"
      (fun () ->
        let g = Rune.grad params_s quadratic (params ()) in
        equal (exact ()) (Nx.zeros Nx.int32 [| 2 |]) g.steps);
    test "a leaf the objective does not use has a gradient of +0." (fun () ->
        let g =
          Rune.grad pair
            (fun (a, _) -> Nx.sum (Nx.mul a a))
            (vec [| 1.; 2. |], vec [| -0.; 3. |])
        in
        equal (exact ()) (vec [| 0.; 0. |]) (snd g));
    test "a tensor behind two leaves is two parameters" (fun () ->
        let x = vec [| 1.; 2. |] in
        let ga, gb =
          Rune.grad pair
            (fun (a, b) -> Nx.add (Nx.sum a) (Nx.mul_s (Nx.sum b) 3.))
            (x, x)
        in
        equal ~msg:"first" (exact ()) (vec [| 1.; 1. |]) ga;
        equal ~msg:"second" (exact ()) (vec [| 3.; 3. |]) gb);
    test "a capture that is also the argument is a constant" (fun () ->
        let w = vec [| 2.; -3. |] in
        equal (exact ()) w (Rune.grad' (fun x -> Nx.sum (Nx.mul x w)) w));
    test "the objective runs once" (fun () ->
        let runs = ref 0 in
        ignore
          (Rune.grad'
             (fun x ->
               incr runs;
               Nx.sum (Nx.mul x x))
             (vec [| 1. |]));
        equal int 1 !runs);
    test "two differentiations of one function give one gradient" (fun () ->
        let f x = Nx.sum (Nx.mul (Nx.sin x) x) in
        let x = vec [| 0.3; -1.2 |] in
        equal (exact ()) (Rune.grad' f x) (Rune.grad' f x));
    test "a value read inside the objective is its primal" (fun () ->
        let seen = ref nan in
        let g =
          Rune.grad'
            (fun x ->
              seen := Nx.item [ 1 ] (Nx.tanh x);
              Nx.sum (Nx.mul x x))
            (vec [| 0.5; -2. |])
        in
        equal ~msg:"value" float_exact (Float.tanh (-2.)) !seen;
        equal ~msg:"gradient" (exact ()) (vec [| 1.; -4. |]) g);
    test
      "a compaction reads its length from the primal and differentiates the \
       elements it keeps" (fun () ->
        let f x =
          Nx.sum (Nx.mul_s (Nx.compress ~condition:(Nx.greater_s x 0.) x) 3.)
        in
        equal (exact ())
          (vec [| 3.; 0.; 3.; 0. |])
          (Rune.grad' f (vec [| 0.5; -1.; 2.; 0. |])));
    test "a range reduction differentiates each range's rows" (fun () ->
        let lo = Nx.create Nx.int64 [| 3 |] [| -1L; 1L; 3L |]
        and hi = Nx.create Nx.int64 [| 3 |] [| 2L; 4L; 3L |] in
        let x = vec [| 0.5; -1.; 2.; 0. |] in
        equal ~msg:"Add" (exact ())
          (vec [| 1.; 2.; 1.; 1. |])
          (Rune.grad' (fun x -> Nx.sum (Nx.reduce_ranges `Add ~lo ~hi x)) x);
        equal ~msg:"Max" (exact ())
          (vec [| 1.; 0.; 1.; 0. |])
          (Rune.grad' (fun x -> Nx.sum (Nx.reduce_ranges `Max ~lo ~hi x)) x));
    test
      "a range's extreme gives its whole derivative to one of its tied rows, \
       infinite ones too" (fun () ->
        let lo = Nx.create Nx.int64 [| 2 |] [| 0L; 3L |]
        and hi = Nx.create Nx.int64 [| 2 |] [| 3L; 6L |] in
        List.iter
          (fun (op, msg, xs) ->
            let g =
              Nx.to_array
                (Rune.grad'
                   (fun x -> Nx.sum (Nx.reduce_ranges op ~lo ~hi x))
                   (vec xs))
            in
            let total a b = Array.fold_left ( +. ) 0. (Array.sub g a (b - a)) in
            equal ~msg float_exact 1. (total 0 3);
            equal ~msg float_exact 1. (total 4 6);
            equal ~msg (array float_exact) [| 0.; 0. |] [| g.(3); g.(6) |];
            is_true ~msg (Array.for_all (fun v -> v = 0. || v = 1.) g))
          [
            ( `Max,
              "Max",
              [| neg_infinity; neg_infinity; neg_infinity; 1.; 2.; 2.; 0. |] );
            (`Min, "Min", [| infinity; infinity; infinity; 1.; 0.; 0.; 2. |]);
          ]);
    test "a branch on a value differentiates the branch taken" (fun () ->
        let f x =
          if Nx.item [] (Nx.sum x) > 0. then Nx.sum (Nx.mul x x) else Nx.sum x
        in
        equal ~msg:"then" (exact ())
          (vec [| 1.; 4. |])
          (Rune.grad' f (vec [| 0.5; 2. |]));
        equal ~msg:"else" (exact ())
          (vec [| 1.; 1. |])
          (Rune.grad' f (vec [| -0.5; -2. |])));
    test "a recursion on a value differentiates the iterations taken" (fun () ->
        (* Each element doubles until the sum passes 10: three times from (1,
           0.5), so the derivative is 2³. *)
        let rec double c =
          if Nx.item [] (Nx.sum c) < 10. then double (Nx.mul_s c 2.) else c
        in
        equal (exact ())
          (vec [| 8.; 8. |])
          (Rune.grad' (fun x -> Nx.sum (double x)) (vec [| 1.; 0.5 |])));
    test "gradient descent on a square shrinks it by the step each time"
      (fun () ->
        (* x - 0.1 · 2x is 0.8 x. *)
        let p = ref (vec [| 1.; -2.; 3. |]) in
        for _ = 1 to 100 do
          let g = Rune.grad' (fun x -> Nx.sum (Nx.mul x x)) !p in
          p := Nx.sub !p (Nx.mul_s g 0.1)
        done;
        let k = 0.8 ** 100. in
        equal (near 1e-12) (vec [| k; -2. *. k; 3. *. k |]) !p);
  ]

(* value_and_grad, value_and_grad_aux *)

let value_tests =
  [
    test "the value is the objective's, bit for bit" (fun () ->
        let f x = Nx.sum (Nx.mul (Nx.exp x) (Nx.sin x)) in
        let x = vec [| 0.7; -1.3; 2.1 |] in
        let v, g = Rune.value_and_grad Nx.Ptree.tensor f x in
        equal ~msg:"value" (exact ()) (f x) v;
        equal ~msg:"gradient" (exact ()) (Rune.grad' f x) g);
    test "an auxiliary result leaves through its structure as values" (fun () ->
        let x = vec [| 0.5; -1. |] in
        let _, _, aux =
          Rune.value_and_grad_aux Nx.Ptree.tensor Nx.Ptree.tensor
            (fun x -> (Nx.sum (Nx.mul x x), Nx.tanh x))
            x
        in
        equal (exact ()) (Nx.tanh x) aux);
    test "an auxiliary result does not contribute to the gradient" (fun () ->
        let x = vec [| 0.5; -1. |] in
        let y, g, _ =
          Rune.value_and_grad_aux Nx.Ptree.tensor Nx.Ptree.tensor
            (fun x -> (Nx.sum (Nx.mul x x), Nx.mul_s (Nx.exp x) 100.))
            x
        in
        equal ~msg:"value" (exact ()) (scalar 1.25) y;
        equal ~msg:"gradient" (exact ()) (vec [| 1.; -2. |]) g);
    test "an auxiliary result of unit" (fun () ->
        let _, g, () =
          Rune.value_and_grad_aux Nx.Ptree.tensor Nx.Ptree.unit
            (fun x -> (Nx.sum x, ()))
            (vec [| 1.; 2. |])
        in
        equal (exact ()) (vec [| 1.; 1. |]) g);
  ]

(* vjp *)

let vjp_tests =
  [
    test "the pullback scales by the cotangent" (fun () ->
        let _, pb = Rune.vjp' (fun x -> Nx.mul x x) (vec [| 1.; 2. |]) in
        equal (exact ()) (vec [| 20.; 4. |]) (pb (vec [| 10.; 1. |])));
    test "a structured result's pullback is the gradient of its pairing"
      (fun () ->
        let a = vec [| 0.7; -1.3; 2.1 |] and b = vec [| 1.9; 0.8; -0.6 |] in
        let ca = vec [| 1.; -2.; 0.5 |] and cb = vec [| 0.3; 1.1; -0.7 |] in
        let f (a, b) = (Nx.mul a b, Nx.add a b) in
        let _, pb = Rune.vjp pair pair f (a, b) in
        let paired p =
          let y, z = f p in
          Nx.add (Nx.sum (Nx.mul y ca)) (Nx.sum (Nx.mul z cb))
        in
        equal
          (Oracle.structure ~rel:1e-12 pair)
          (Rune.grad pair paired (a, b))
          (pb (ca, cb)));
    test "a leaf at two result positions receives their sum" (fun () ->
        let _, pb =
          Rune.vjp Nx.Ptree.tensor pair (fun x -> (x, x)) (vec [| 1.; 2. |])
        in
        equal (exact ())
          (vec [| 4.; 6. |])
          (pb (vec [| 1.; 2. |], vec [| 3.; 4. |])));
    test "the identity's pullback is the cotangent" (fun () ->
        let _, pb = Rune.vjp' Fun.id (vec [| 1.; 2. |]) in
        equal (exact ()) (vec [| 5.; -6. |]) (pb (vec [| 5.; -6. |])));
    test "a constant result pulls back to zeros" (fun () ->
        let c = vec [| 9.; 9. |] in
        let _, pb = Rune.vjp' (fun _ -> c) (vec [| 1.; 2. |]) in
        equal (exact ()) (vec [| 0.; 0. |]) (pb (vec [| 1.; 1. |])));
    test "a unit result pulls back to zeros" (fun () ->
        let _, pb =
          Rune.vjp Nx.Ptree.tensor Nx.Ptree.unit
            (fun _ -> ())
            (vec [| 1.; 2. |])
        in
        equal (exact ()) (vec [| 0.; 0. |]) (pb ()));
    test "a parameter the result does not use pulls back to zeros" (fun () ->
        let _, pb =
          Rune.vjp pair Nx.Ptree.tensor
            (fun (a, _) -> Nx.mul a a)
            (vec [| 1. |], vec [| 2.; 3. |])
        in
        equal (exact ()) (vec [| 0.; 0. |]) (snd (pb (vec [| 1. |]))));
    test "applying the pullback runs no part of the function" (fun () ->
        let runs = ref 0 in
        let _, pb =
          Rune.vjp'
            (fun x ->
              incr runs;
              Nx.sin x)
            (vec [| 1. |])
        in
        ignore (pb (vec [| 1. |]));
        ignore (pb (vec [| 2. |]));
        equal int 1 !runs);
    test "a pullback applied twice equals two fresh pullbacks" (fun () ->
        let f (a, b) = Nx.mul (Nx.sin a) b in
        let x = (vec [| 0.7; -1.3 |], vec [| 1.9; 0.8 |]) in
        let c1 = vec [| 1.; 0. |] and c2 = vec [| 0.; 2. |] in
        let _, pb = Rune.vjp pair Nx.Ptree.tensor f x in
        let g1 = pb c1 in
        let g2 = pb c2 in
        let fresh c = snd (Rune.vjp pair Nx.Ptree.tensor f x) c in
        equal ~msg:"first" (Oracle.structure pair) (fresh c1) g1;
        equal ~msg:"second" (Oracle.structure pair) (fresh c2) g2);
    test "an integer result leaf's cotangent is taken and ignored" (fun () ->
        let q = Nx.Ptree.(pair tensor tensor) in
        let _, pb =
          Rune.vjp Nx.Ptree.tensor q
            (fun x -> (Nx.mul x x, Nx.create Nx.int32 [| 2 |] [| 1l; 2l |]))
            (vec [| 1.; 2. |])
        in
        equal (exact ())
          (vec [| 2.; 4. |])
          (pb (vec [| 1.; 1. |], Nx.create Nx.int32 [| 2 |] [| 50l; 60l |])));
    test "cotangents of another structure are refused at the pullback"
      (fun () ->
        let _, pb =
          Rune.vjp Nx.Ptree.tensor
            Nx.Ptree.(list tensor)
            (fun x -> [ x; Nx.mul x x ])
            (vec [| 1. |])
        in
        raises
          (Invalid_argument
             "Rune.vjp: the root: length 2 in the result, length 1 in the \
              cotangents") (fun () -> pb [ vec [| 1. |] ]));
    test "a cotangent of another shape is refused at the pullback" (fun () ->
        let _, pb = Rune.vjp' Nx.sin (vec [| 1.; 2. |]) in
        raises
          (Invalid_argument
             "Rune.vjp': the root: shape [2] in the result, [1] in the \
              cotangents") (fun () -> pb (vec [| 1. |])));
    test "a cotangent of another dtype is refused at the pullback" (fun () ->
        let _, pb =
          Rune.vjp Nx.Ptree.tensor packed
            (fun x -> Nx.P (Nx.sin x))
            (vec [| 1. |])
        in
        raises
          (Invalid_argument
             "Rune.vjp: the root: float64 in the result, float32 in the \
              cotangents") (fun () ->
            pb (Nx.P (Nx.create Nx.float32 [| 1 |] [| 1. |]))));
    test "an integer result leaf's cotangent must have its shape" (fun () ->
        let q = Nx.Ptree.(pair tensor tensor) in
        let _, pb =
          Rune.vjp Nx.Ptree.tensor q
            (fun x -> (x, Nx.create Nx.int32 [| 1 |] [| 1l |]))
            (vec [| 1. |])
        in
        raises
          (Invalid_argument
             "Rune.vjp: 1: shape [1] in the result, [2] in the cotangents")
          (fun () -> pb (vec [| 1. |], Nx.zeros Nx.int32 [| 2 |])));
  ]

(* jvp *)

let jvp_tests =
  [
    test "the tangent of a record is its analytic tangent" (fun () ->
        let p = params () in
        let t =
          {
            w = vec [| 1.; 1.; 1. |];
            b = vec [| 2. |];
            steps = Nx.zeros Nx.int32 [| 2 |];
          }
        in
        let _, dy = Rune.jvp params_s Nx.Ptree.tensor quadratic p t in
        (* 2 (1 - 2 + 3) + 3·2 *)
        equal (close ()) (scalar 10.) dy);
    test "leaves of two dtypes push their tangents forward in one pass"
      (fun () ->
        let w = Nx.create Nx.float32 [| 2 |] [| 1.; -2. |]
        and s = vec [| 3. |] in
        let _, dy =
          Rune.jvp
            Nx.Ptree.(pair tensor tensor)
            Nx.Ptree.tensor
            (fun (w, s) ->
              Nx.add (Nx.cast f64 (Nx.sum (Nx.mul w w))) (Nx.sum (Nx.mul s s)))
            (w, s)
            (Nx.create Nx.float32 [| 2 |] [| 1.; 0. |], vec [| 1. |])
        in
        (* 2 w₀ + 2 s *)
        equal (exact ()) (scalar 8.) dy);
    test "the primal is the function's value, bit for bit" (fun () ->
        let f x = Nx.mul (Nx.exp x) (Nx.tanh x) in
        let x = vec [| 0.3; -0.8 |] in
        equal (exact ()) (f x) (fst (Rune.jvp' f x (vec [| 1.; 1. |]))));
    test "a function of nothing it differentiates has a zero tangent" (fun () ->
        let y, dy =
          Rune.jvp' (fun _ -> scalar 42.) (vec [| 1. |]) (vec [| 1. |])
        in
        equal ~msg:"value" (exact ()) (scalar 42.) y;
        equal ~msg:"tangent" (exact ()) (scalar 0.) dy);
    test "each result leaf has its own tangent" (fun () ->
        let f (a, b) = (Nx.mul a b, Nx.add a b) in
        let x = (vec [| 0.7; -1.3 |], vec [| 1.9; 0.8 |]) in
        let t = (vec [| 1.; 0.5 |], vec [| -1.; 2. |]) in
        let _, (dp, ds) = Rune.jvp pair pair f x t in
        let _, dp' = Rune.jvp pair Nx.Ptree.tensor (fun x -> fst (f x)) x t in
        let _, ds' = Rune.jvp pair Nx.Ptree.tensor (fun x -> snd (f x)) x t in
        equal ~msg:"product" (exact ()) dp' dp;
        equal ~msg:"sum" (exact ()) ds' ds);
    test "a tensor behind two leaves has two tangents" (fun () ->
        let x = vec [| 1.; 2. |] in
        let _, dy =
          Rune.jvp pair Nx.Ptree.tensor
            (fun (a, b) -> Nx.add (Nx.sum a) (Nx.mul_s (Nx.sum b) 3.))
            (x, x)
            (vec [| 1.; 0. |], vec [| 0.; 1. |])
        in
        equal (exact ()) (scalar 4.) dy);
    test "an integer result leaf has a zero tangent of its dtype" (fun () ->
        let q = Nx.Ptree.(pair tensor tensor) in
        let _, (_, dk) =
          Rune.jvp Nx.Ptree.tensor q
            (fun x -> (x, Nx.create Nx.int32 [| 2 |] [| 4l; 5l |]))
            (vec [| 1. |]) (vec [| 1. |])
        in
        equal (exact ()) (Nx.zeros Nx.int32 [| 2 |]) dk);
    test "the function runs once" (fun () ->
        let runs = ref 0 in
        ignore
          (Rune.jvp'
             (fun x ->
               incr runs;
               Nx.sin x)
             (vec [| 1. |]) (vec [| 1. |]));
        equal int 1 !runs);
    test "tangents of another structure are refused" (fun () ->
        raises
          (Invalid_argument
             "Rune.jvp: the root: length 2 in the parameters, length 1 in the \
              tangents") (fun () ->
            Rune.jvp
              Nx.Ptree.(list tensor)
              Nx.Ptree.tensor
              (fun l -> Nx.sum (List.hd l))
              [ vec [| 1. |]; vec [| 2. |] ]
              [ vec [| 1. |] ]));
    test "a tangent of another shape is refused" (fun () ->
        raises
          (Invalid_argument
             "Rune.jvp': the root: shape [2] in the parameters, [1] in the \
              tangents") (fun () ->
            Rune.jvp' Nx.sum (vec [| 1.; 2. |]) (vec [| 1. |])));
    test "a tangent of another dtype is refused" (fun () ->
        raises
          (Invalid_argument
             "Rune.jvp: the root: float64 in the parameters, float32 in the \
              tangents") (fun () ->
            Rune.jvp packed Nx.Ptree.unit
              (fun _ -> ())
              (Nx.P (vec [| 1. |]))
              (Nx.P (Nx.create Nx.float32 [| 1 |] [| 1. |]))));
  ]

(* jacfwd', jacrev' *)

(* R³ → R²: (x₀x₁, sin x₂ · x₀). *)
let r3_to_r2 x =
  let at i = Nx.slice [ Nx.R (i, i + 1) ] x in
  Nx.concatenate ~axis:0 [ Nx.mul (at 0) (at 1); Nx.mul (Nx.sin (at 2)) (at 0) ]

let jacobian_tests =
  [
    test "the Jacobian is its analytic matrix" (fun () ->
        let x = vec [| 0.7; -1.3; 2.1 |] in
        let expected =
          Nx.create f64 [| 2; 3 |]
            [| -1.3; 0.7; 0.; Float.sin 2.1; 0.; 0.7 *. Float.cos 2.1 |]
        in
        equal ~msg:"jacfwd'" (close ()) expected (Rune.jacfwd' r3_to_r2 x);
        equal ~msg:"jacrev'" (close ()) expected (Rune.jacrev' r3_to_r2 x));
    test "the Jacobian's shape is the result's then the argument's" (fun () ->
        let x = Nx.reshape [| 1; 3 |] (vec [| 0.7; -1.3; 2.1 |]) in
        let f x = Nx.reshape [| 2; 1 |] (r3_to_r2 (Nx.reshape [| 3 |] x)) in
        equal ~msg:"jacfwd'" (array int) [| 2; 1; 1; 3 |]
          (Nx.shape (Rune.jacfwd' f x));
        equal ~msg:"jacrev'" (array int) [| 2; 1; 1; 3 |]
          (Nx.shape (Rune.jacrev' f x)));
    test "the Jacobian of a scalar function of a scalar is a scalar" (fun () ->
        let f x = Nx.mul x (Nx.sin x) in
        let x = scalar 0.4 in
        let d = Float.sin 0.4 +. (0.4 *. Float.cos 0.4) in
        equal ~msg:"jacfwd'" (close ()) (scalar d) (Rune.jacfwd' f x);
        equal ~msg:"jacrev'" (close ()) (scalar d) (Rune.jacrev' f x));
    test "an empty argument gives a Jacobian with an empty axis" (fun () ->
        let f x =
          Nx.add_s
            (Nx.reshape [| 2 |]
               (Nx.sum x ~keepdims:false |> Nx.broadcast_to [| 2 |]))
            1.
        in
        equal ~msg:"jacfwd'" (array int) [| 2; 0 |]
          (Nx.shape (Rune.jacfwd' f (vec [||])));
        equal ~msg:"jacrev'" (array int) [| 2; 0 |]
          (Nx.shape (Rune.jacrev' f (vec [||]))));
    test "jacfwd' has the result's dtype and jacrev' the argument's" (fun () ->
        let x = Nx.create Nx.float32 [| 3 |] [| 0.7; -1.3; 2.1 |] in
        let f x = Nx.cast f64 (Nx.mul x x) in
        let expected =
          Nx.create f64 [| 3; 3 |] [| 1.4; 0.; 0.; 0.; -2.6; 0.; 0.; 0.; 4.2 |]
        in
        equal ~msg:"jacfwd'" (near 1e-6) expected (Rune.jacfwd' f x);
        equal ~msg:"jacrev'" (near 1e-6)
          (Nx.cast Nx.float32 expected)
          (Rune.jacrev' f x));
    test "each runs the function once" (fun () ->
        let runs = ref 0 in
        let f x =
          incr runs;
          r3_to_r2 x
        in
        ignore (Rune.jacfwd' f (vec [| 1.; 2.; 3. |]));
        equal ~msg:"jacfwd'" int 1 !runs;
        runs := 0;
        ignore (Rune.jacrev' f (vec [| 1.; 2.; 3. |]));
        equal ~msg:"jacrev'" int 1 !runs);
    test "a Hessian is jacfwd' of grad'" (fun () ->
        let cube x = Nx.sum (Nx.mul x (Nx.mul x x)) in
        let h = Rune.jacfwd' (Rune.grad' cube) (vec [| 0.7; -1.3; 2.1 |]) in
        equal (close ())
          (Nx.create f64 [| 3; 3 |]
             [| 4.2; 0.; 0.; 0.; -7.8; 0.; 0.; 0.; 12.6 |])
          h);
    test "a Hessian-vector product is jvp of grad" (fun () ->
        (* f (a, b) = Σ a² + Σ a·b, whose Hessian maps (va, vb) to (2 va + vb,
           va). *)
        let f (a, b) = Nx.add (Nx.sum (Nx.mul a a)) (Nx.sum (Nx.mul a b)) in
        let x = (vec [| 0.7; -1.3; 2.1 |], vec [| 1.9; 0.8; -0.6 |]) in
        let v = (vec [| 1.; 0.; 2. |], vec [| 0.5; -1.; 0. |]) in
        let _, hv = Rune.jvp pair pair (Rune.grad pair f) x v in
        equal
          (Oracle.structure ~rel:1e-12 pair)
          (vec [| 2.5; -1.; 4. |], vec [| 1.; 0.; 2. |])
          hv);
  ]

(* check_grads *)

let sin_with_pullback scale =
  Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
      (Nx.sin x, fun g -> Nx.mul g (Nx.mul_s (Nx.cos x) scale)))

let check_grads_tests =
  [
    test "a correct gradient is accepted" (fun () ->
        let f (a, b) = Nx.sum (Nx.mul (Nx.exp a) (Nx.sin b)) in
        is_ok
          (Rune.check_grads pair f
             (vec [| 0.7; -1.3; 2.1 |], vec [| 1.9; 0.8; -0.6 |])));
    test "a pullback twice the true one is caught" (fun () ->
        is_error
          (Rune.check_grads Nx.Ptree.tensor
             (fun x -> Nx.sum (sin_with_pullback 2. x))
             (vec [| 0.7; -1.3; 2.1 |])));
    test "an error in one element of one leaf is caught" (fun () ->
        let bump = vec [| 0.; 0.; 0.5 |] in
        let rule =
          Rune.custom_vjp pair Nx.Ptree.tensor (fun (a, b) ->
              ( Nx.add (Nx.sin a) (Nx.sin b),
                fun g ->
                  ( Nx.mul g (Nx.cos a),
                    Nx.add (Nx.mul g (Nx.cos b)) (Nx.mul g bump) ) ))
        in
        is_error
          (Rune.check_grads pair
             (fun p -> Nx.sum (rule p))
             (vec [| 0.7; -1.3; 2.1 |], vec [| 1.9; 0.8; -0.6 |])));
    test "the tolerance decides" (fun () ->
        let f x = Nx.sum (sin_with_pullback 1.005 x) in
        let x = vec [| 0.7; -1.3; 2.1 |] in
        is_ok ~msg:"1e-2" (Rune.check_grads ~tol:1e-2 Nx.Ptree.tensor f x);
        is_error ~msg:"1e-3" (Rune.check_grads ~tol:1e-3 Nx.Ptree.tensor f x));
    test "a gradient equal to its difference passes a tolerance of zero"
      (fun () ->
        (* At 0 with a step of 1, the difference of a one-element sum is exact,
           so agreement within 0 is equality. *)
        is_ok
          (Rune.check_grads ~eps:1. ~tol:0. Nx.Ptree.tensor Nx.sum
             (vec [| 0. |])));
    test "a broken precondition raises" (fun () ->
        raises
          (Invalid_argument
             "Rune.check_grads: the objective must return a real or complex \
              scalar, got float64 [2]") (fun () ->
            Rune.check_grads Nx.Ptree.tensor
              (fun x -> Nx.mul x x)
              (vec [| 1.; 2. |])));
  ]

(* detach *)

let detach_tests =
  [
    test "a detached mean centres a value without its derivative" (fun () ->
        (* Σ (x - m)² with m held constant has the gradient 2 (x - m): at [1; 2;
           3; 6], m = 3. *)
        equal (exact ())
          (vec [| -4.; -2.; 0.; 6. |])
          (Rune.grad'
             (fun x -> Nx.sum (Nx.square (Nx.sub x (Rune.detach (Nx.mean x)))))
             (vec [| 1.; 2.; 3.; 6. |])));
    test "outside every transformation detach is its argument" (fun () ->
        let x = vec [| 1.; 2. |] in
        is_true (Rune.detach x == x));
    test "under grad a detached value has no derivative" (fun () ->
        equal (exact ()) (vec [| 3. |])
          (Rune.grad'
             (fun x -> Nx.sum (Nx.mul x (Rune.detach x)))
             (vec [| 3. |])));
    test "under jvp a detached value has no tangent" (fun () ->
        let y, dy =
          Rune.jvp'
            (fun x -> Nx.mul x (Rune.detach x))
            (vec [| 3. |]) (vec [| 1. |])
        in
        equal ~msg:"value" (exact ()) (vec [| 9. |]) y;
        equal ~msg:"tangent" (exact ()) (vec [| 3. |]) dy);
    test "detach holds a value constant under every differentiation around it"
      (fun () ->
        let inner x =
          Rune.grad'
            (fun y -> Nx.sum (Nx.mul y (Rune.detach (Nx.mul x x))))
            (vec [| 1. |])
        in
        equal ~msg:"grad" (exact ()) (vec [| 0. |])
          (Rune.grad' (fun x -> Nx.sum (inner x)) (vec [| 2. |]));
        equal ~msg:"jvp" (exact ()) (vec [| 0. |])
          (snd (Rune.jvp' inner (vec [| 2. |]) (vec [| 1. |]))));
    test "a value detached under grad may leave the function" (fun () ->
        let seen = ref None in
        ignore
          (Rune.grad'
             (fun x ->
               seen := Some (Rune.detach (Nx.tanh x));
               Nx.sum x)
             (vec [| 0.5 |]));
        equal (exact ()) (vec [| Float.tanh 0.5 |]) (Option.get !seen));
  ]

(* Values stay inside *)

exception Carried of Nx.float64_t

let escapes_tests =
  [
    test "an intermediate kept in a reference cannot be read after grad"
      (fun () ->
        let seen = ref None in
        ignore
          (Rune.grad'
             (fun x ->
               let h = Nx.tanh x in
               seen := Some h;
               Nx.sum h)
             (vec [| 0.5 |]));
        let h = Option.get !seen in
        raises ~msg:"to_array" (Invalid_argument escape) (fun () ->
            Nx.to_array h);
        raises ~msg:"item" (Invalid_argument escape) (fun () -> Nx.item [ 0 ] h);
        raises ~msg:"operand" (Invalid_argument escape) (fun () -> Nx.add h h));
    test "an escaped value keeps its shape and dtype" (fun () ->
        let seen = ref None in
        ignore
          (Rune.grad'
             (fun x ->
               seen := Some (Nx.sin x);
               Nx.sum x)
             (vec [| 1.; 2. |]));
        let h = Option.get !seen in
        equal ~msg:"shape" (array int) [| 2 |] (Nx.shape h);
        equal ~msg:"dtype" string "float64" (Nx_dtype.to_string (Nx.dtype h)));
    test "an escaped value raises inside a later grad" (fun () ->
        let seen = ref None in
        ignore
          (Rune.grad'
             (fun x ->
               seen := Some (Nx.sin x);
               Nx.sum x)
             (vec [| 1. |]));
        let h = Option.get !seen in
        raises (Invalid_argument escape) (fun () ->
            Rune.grad' (fun y -> Nx.sum (Nx.mul y h)) (vec [| 1. |])));
    test "a closure run after its grad cannot use the values it captured"
      (fun () ->
        let k = ref (fun () -> ()) in
        ignore
          (Rune.grad'
             (fun x ->
               let h = Nx.exp x in
               (k := fun () -> ignore (Nx.to_array (Nx.add h h)));
               Nx.sum h)
             (vec [| 1. |]));
        raises (Invalid_argument escape) !k);
    test "a differentiated value cannot be used on another domain" (fun () ->
        let outcome = ref "" in
        ignore
          (Rune.grad'
             (fun x ->
               let h = Nx.sin x in
               (outcome :=
                  match
                    Domain.join
                      (Domain.spawn (fun () -> Nx.to_array (Nx.add h h)))
                  with
                  | _ -> "computed"
                  | exception Invalid_argument m -> m);
               Nx.sum h)
             (vec [| 1. |]));
        equal string escape !outcome);
    test "a value carried out by an exception cannot be read" (fun () ->
        match
          Rune.grad' (fun x -> raise (Carried (Nx.sin x))) (vec [| 1. |])
        with
        | _ -> fail "grad returned"
        | exception Carried h ->
            raises (Invalid_argument escape) (fun () -> Nx.to_array h));
    test "a custom_jvp rule cannot use its own differentiation's value"
      (fun () ->
        raises
          (Invalid_argument
             "Rune.custom_jvp: the rule uses a value its own differentiation \
              tracks; pass it as an argument") (fun () ->
            Rune.grad'
              (fun x ->
                let w = Nx.mul x x in
                let g =
                  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun y ->
                      (Nx.mul y w, fun dy -> Nx.mul dy w))
                in
                Nx.sum (g x))
              (vec [| 1.; 2. |])));
    test "a custom_vjp rule cannot use its own differentiation's value"
      (fun () ->
        raises
          (Invalid_argument
             "Rune.custom_vjp: the rule uses a value its own differentiation \
              tracks; pass it as an argument") (fun () ->
            Rune.grad'
              (fun x ->
                let w = Nx.mul x x in
                let g =
                  Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun y ->
                      (Nx.mul y w, fun ct -> Nx.mul ct w))
                in
                Nx.sum (g x))
              (vec [| 1.; 2. |])));
    test "a rule may use an enclosing differentiation's value" (fun () ->
        (* The inner gradient of Σ x·w is w, so the outer one of its sum is 1
           per element: the inner rule captures the outer w, which the outer
           grad differentiates through the rule's code. *)
        let outer w =
          let g =
            Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
                (Nx.mul x w, fun dx -> Nx.mul dx w))
          in
          Nx.sum (Rune.grad' (fun x -> Nx.sum (g x)) (vec [| 1.; 1. |]))
        in
        equal (exact ())
          (vec [| 1.; 1. |])
          (Rune.grad' outer (vec [| 3.; 4. |])));
    test
      "a pullback that holds an outer differentiation's value cannot run after \
       it" (fun () ->
        let kept = ref None in
        ignore
          (Rune.grad'
             (fun w ->
               let _, pb = Rune.vjp' (fun x -> Nx.mul x w) (vec [| 1. |]) in
               kept := Some pb;
               Nx.sum w)
             (vec [| 2. |]));
        raises (Invalid_argument escape) (fun () ->
            Nx.to_array (Option.get !kept (vec [| 1. |]))));
  ]

(* Pullbacks anywhere *)

let pullback_tests =
  [
    test "a pullback applied under vmap is its loop" (fun () ->
        let _, pb =
          Rune.vjp' (fun x -> Nx.mul (Nx.sin x) x) (vec [| 0.5; -1. |])
        in
        let cts = Nx.create f64 [| 3; 2 |] [| 1.; 0.; 0.; 1.; 2.; -3. |] in
        let loop = Nx.stack (List.init 3 (fun i -> pb (Nx.get [ i ] cts))) in
        equal (close ()) loop (Rune.vmap' pb cts));
    test "a pullback applied under jit is its eager value" (fun () ->
        let _, pb =
          Rune.vjp' (fun x -> Nx.mul (Nx.sin x) x) (vec [| 0.5; -1. |])
        in
        let ct = vec [| 2.; -3. |] in
        equal (close ()) (pb ct) (Rune.jit' pb ct));
    test "a pullback may run on two domains at once" (fun () ->
        let _, pb =
          Rune.vjp' (fun x -> Nx.mul (Nx.exp x) x) (vec [| 0.5; -1.; 2. |])
        in
        let cts =
          List.init 100 (fun i ->
              vec [| Float.of_int i; 1.; -0.5 *. Float.of_int i |])
        in
        let sequential = List.map pb cts in
        let run () = List.map pb cts in
        let a = Domain.spawn run and b = Domain.spawn run in
        let a = Domain.join a and b = Domain.join b in
        List.iteri
          (fun i expected ->
            equal
              ~msg:(Printf.sprintf "first domain, %d" i)
              (exact ()) expected (List.nth a i);
            equal
              ~msg:(Printf.sprintf "second domain, %d" i)
              (exact ()) expected (List.nth b i))
          sequential);
    test "a derivative through a pullback" (fun () ->
        (* f x = x³ pulls c back to 3x²c: its derivative in x is 6xc, in c
           3x². *)
        let f x = Nx.mul x (Nx.mul x x) in
        let c = vec [| 1.; -2. |] and x = vec [| 0.5; 1.5 |] in
        let in_x = Rune.grad' (fun x -> Nx.sum (snd (Rune.vjp' f x) c)) x in
        let _, pb = Rune.vjp' f x in
        let in_c = Rune.grad' (fun c -> Nx.sum (pb c)) c in
        equal ~msg:"in x" (close ()) (vec [| 3.; -18. |]) in_x;
        equal ~msg:"in c" (close ()) (vec [| 0.75; 6.75 |]) in_c);
  ]

(* Numerics at the edges *)

let inf = Float.infinity

(* [flat x] is [x] flattened, which raises unless [x] is C-contiguous. *)
let flat x = Nx.reshape [| -1 |] x

let mat r c =
  Nx.reshape [| r; c |] (Nx.arange_f f64 1. (Float.of_int (1 + (r * c))) 1.)

let fresh_tests =
  [
    test "a gradient through a matmul's transposed operand flattens" (fun () ->
        let q = mat 2 3 and k = mat 4 3 in
        let _, gk =
          Rune.grad
            Nx.Ptree.(pair tensor tensor)
            (fun (q, k) -> Nx.sum (Nx.matmul q (Nx.matrix_transpose k)))
            (q, k)
        in
        is_true (Nx.is_c_contiguous gk);
        equal (exact ())
          (flat
             (Nx.contiguous
                (Nx.broadcast_to [| 4; 3 |]
                   (Nx.sum ~axes:[ 0 ] ~keepdims:true q))))
          (flat gk));
    test "a sum's gradient, a broadcast, is a value of its own" (fun () ->
        is_true (Nx.is_c_contiguous (Rune.grad' Nx.sum (mat 2 3))));
    test "a pullback's transposed cotangent flattens" (fun () ->
        let _, pullback = Rune.vjp' Nx.matrix_transpose (mat 2 3) in
        let ct = mat 3 2 in
        equal (exact ())
          (flat (Nx.contiguous (Nx.matrix_transpose ct)))
          (flat (pullback ct)));
    test "a transposed tangent flattens" (fun () ->
        let dx = mat 2 3 in
        equal (exact ())
          (flat (Nx.contiguous (Nx.matrix_transpose dx)))
          (flat (snd (Rune.jvp' Nx.matrix_transpose (mat 2 3) dx))));
    test "a transposed primal flattens" (fun () ->
        let y, dy =
          Rune.jvp'
            (fun x -> Nx.flatten (Nx.matrix_transpose x))
            (mat 2 3) (mat 2 3)
        in
        equal ~msg:"primal" (exact ()) (vec [| 1.; 4.; 2.; 5.; 3.; 6. |]) y;
        equal ~msg:"tangent" (exact ()) (vec [| 1.; 4.; 2.; 5.; 3.; 6. |]) dy);
    test "a gradient through a flattened transpose" (fun () ->
        let w = vec [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        equal (exact ())
          (Nx.create f64 [| 2; 3 |] [| 1.; 3.; 5.; 2.; 4.; 6. |])
          (Rune.grad'
             (fun x -> Nx.sum (Nx.mul (Nx.flatten (Nx.matrix_transpose x)) w))
             (mat 2 3)));
    test "a read under jvp sees a transposed primal in C order" (fun () ->
        let second x = (Nx.to_array (Nx.matrix_transpose x)).(1) in
        let y, _ =
          Rune.jvp' (fun x -> Nx.mul_s x (second x)) (mat 2 3) (mat 2 3)
        in
        equal (exact ()) (Nx.mul_s (mat 2 3) 4.) y);
    test "a reshape that a strided value's view cannot take raises as eagerly"
      (fun () ->
        let tr = Nx.matrix_transpose in
        let refused f x shape =
          let e =
            Invalid_argument
              (Printf.sprintf
                 "reshape: cannot reshape %s, call contiguous() first" shape)
          in
          raises ~msg:"eagerly" e (fun () -> ignore (f x));
          raises ~msg:"under jvp" e (fun () -> ignore (Rune.jvp' f x x));
          raises ~msg:"under grad" e (fun () ->
              ignore (Rune.grad' (fun x -> Nx.sum (f x)) x))
        in
        refused
          (fun x -> Nx.reshape [| 24 |] (tr (Nx.reshape [| 3; 8 |] (tr x))))
          (mat 4 6) "[6,4] to [3,8], strides [1,6] cannot view it";
        refused
          (fun x ->
            Nx.reshape [| 4; 6 |]
              (Nx.transpose ~axes:[ 1; 0; 2 ] (Nx.reshape [| 2; 2; 6 |] x)))
          (mat 4 6) "[2,2,6] to [4,6], strides [6,12,1] cannot view it");
    test "a transposed dual is not C-contiguous, as eagerly" (fun () ->
        let seen = ref [] in
        let f x =
          seen := Nx.is_c_contiguous (Nx.matrix_transpose x) :: !seen;
          x
        in
        ignore (f (mat 2 3));
        ignore (Rune.jvp' f (mat 2 3) (mat 2 3));
        ignore (Rune.grad' (fun x -> Nx.sum (f x)) (mat 2 3));
        equal (list bool) [ false; false; false ] !seen);
  ]

let edge_tests =
  [
    test "a constant operand adds no term at an infinite argument" (fun () ->
        let one = vec [| 1. |] and x = vec [| inf |] in
        equal ~msg:"jvp of x·2" (exact ()) (vec [| 2. |])
          (snd (Rune.jvp' (fun x -> Nx.mul_s x 2.) x one));
        equal ~msg:"grad of x·2" (exact ()) (vec [| 2. |])
          (Rune.grad' (fun x -> Nx.sum (Nx.mul_s x 2.)) x);
        equal ~msg:"jvp of x/2" (exact ()) (vec [| 0.5 |])
          (snd (Rune.jvp' (fun x -> Nx.div_s x 2.) x one));
        equal ~msg:"grad of x/2" (exact ()) (vec [| 0.5 |])
          (Rune.grad' (fun x -> Nx.sum (Nx.div_s x 2.)) x));
    test "a gradient that one contribution makes keeps the sign of -0."
      (fun () ->
        let x = vec [| -0. |] in
        equal ~msg:"grad" (exact ()) (vec [| -0. |])
          (Rune.grad' (fun x -> Nx.sum (Nx.mul x x)) x);
        equal ~msg:"jvp" (exact ()) (vec [| -0. |])
          (snd (Rune.jvp' (fun x -> Nx.mul x x) x (vec [| 1. |]))));
    test "a subnormal argument's derivative is not flushed" (fun () ->
        let x = vec [| 1e-310 |] in
        equal ~msg:"grad" (exact ()) (vec [| 2e-310 |])
          (Rune.grad' (fun x -> Nx.sum (Nx.mul x x)) x);
        equal ~msg:"jvp" (exact ()) (vec [| 2e-310 |])
          (snd (Rune.jvp' (fun x -> Nx.mul x x) x (vec [| 1. |]))));
    test "a NaN argument gives a NaN derivative where the derivative reads it"
      (fun () ->
        let g = Rune.grad' (fun x -> Nx.sum (Nx.mul x x)) (vec [| nan; 1. |]) in
        is_true ~msg:"at NaN" (Float.is_nan (Nx.item [ 0 ] g));
        equal ~msg:"elsewhere" float_exact 2. (Nx.item [ 1 ] g));
    test "an infinite argument of a sum has derivative one" (fun () ->
        equal (exact ())
          (vec [| 1.; 1. |])
          (Rune.grad' Nx.sum (vec [| inf; neg_infinity |])));
    test "an empty leaf has an empty gradient" (fun () ->
        let ga, gb =
          Rune.grad pair
            (fun (a, b) -> Nx.add (Nx.sum a) (Nx.sum (Nx.mul b b)))
            (vec [||], vec [| 1.; 2. |])
        in
        equal ~msg:"empty" (exact ()) (vec [||]) ga;
        equal ~msg:"other" (exact ()) (vec [| 2.; 4. |]) gb);
  ]

(* The error table *)

let tangent_map map =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x -> (Nx.sin x, map))

let under_grad map x = Rune.grad' (fun x -> Nx.sum (tangent_map map x)) x
let total : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

let nonlinear entry op =
  Printf.sprintf
    "%s: a custom_jvp tangent map applies %s to a tangent; a tangent map must \
     be linear in its tangents"
    entry op

let error_cases =
  [
    ( "a tangent map applying exp",
      nonlinear "Rune.grad'" "exp",
      fun () -> under_grad Nx.exp (vec [| 1. |]) );
    ( "a tangent map adding a value",
      nonlinear "Rune.grad'" "add",
      fun () -> under_grad (fun dx -> Nx.add_s dx 1.) (vec [| 1. |]) );
    ( "a tangent map padding with a nonzero fill",
      nonlinear "Rune.grad'" "pad",
      fun () ->
        Rune.grad'
          (fun x ->
            Nx.sum
              (Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor
                 (fun x -> (Nx.pad [| (1, 1) |] 0. x, Nx.pad [| (1, 1) |] 1.))
                 x))
          (vec [| 1. |]) );
    ( "a tangent map reading a tangent",
      "Rune.grad': a custom_jvp tangent map reads a tangent's value with \
       Nx.item; under reverse mode a tangent has none",
      fun () ->
        under_grad
          (fun dx ->
            ignore (Nx.item [ 0 ] dx);
            dx)
          (vec [| 1. |]) );
    ( "a tangent map adding a tangent to a total",
      "Rune.Total.add: a custom_jvp tangent map adds a tangent under reverse \
       mode; a total takes values",
      fun () ->
        fst
          (Rune.Total.collect total ~zero:(Nx.zeros f64 [| 1 |]) (fun () ->
               under_grad
                 (fun dx ->
                   Rune.Total.add total dx;
                   dx)
                 (vec [| 1. |]))) );
    ( "the nonlinear map under value_and_grad",
      nonlinear "Rune.value_and_grad" "exp",
      fun () ->
        snd
          (Rune.value_and_grad Nx.Ptree.tensor
             (fun x -> Nx.sum (tangent_map Nx.exp x))
             (vec [| 1. |])) );
    ( "the nonlinear map under vjp",
      nonlinear "Rune.vjp" "exp",
      fun () ->
        fst
          (Rune.vjp Nx.Ptree.tensor Nx.Ptree.tensor (tangent_map Nx.exp)
             (vec [| 1. |])) );
    ( "jvp of a custom_vjp whose result holds a tensor",
      "Rune.jvp': a custom_vjp rule has no forward derivative; give the \
       function a custom_jvp rule",
      fun () ->
        snd (Rune.jvp' (sin_with_pullback 1.) (vec [| 1. |]) (vec [| 1. |])) );
    ( "a lane read inside vmap",
      "Nx.item: cannot read the value of a batched tensor inside vmap; return \
       it from the mapped function instead",
      fun () ->
        Rune.vmap'
          (fun x ->
            ignore (Nx.item [] x);
            x)
          (vec [| 1.; 2. |]) );
  ]

let error_tests =
  [
    group "the messages"
      (List.map
         (fun (name, expected, f) ->
           test name (fun () -> equal string expected (Oracle.message f)))
         error_cases);
    test "no message names rune's internals" (fun () ->
        List.iter
          (fun (name, _, f) ->
            let m = String.lowercase_ascii (Oracle.message f) in
            List.iter
              (fun word -> not_contains ~msg:name ~sub:word m)
              [ "slot"; "dual"; "installation"; "tape"; "recorder" ])
          error_cases);
    test "an error raises at the operation that causes it" (fun () ->
        (* The objective catches the refused tangent map and continues: the
           gradient is the fallback's. *)
        let g =
          Rune.grad'
            (fun x ->
              match tangent_map Nx.exp x with
              | y -> Nx.sum y
              | exception Invalid_argument _ -> Nx.sum (Nx.mul x x))
            (vec [| 3. |])
        in
        equal (exact ()) (vec [| 6. |]) g);
    test "an affine tangent map is accepted under jvp" (fun () ->
        let _, dy =
          Rune.jvp'
            (tangent_map (fun dx -> Nx.add_s dx 1.))
            (vec [| 0. |]) (vec [| 1. |])
        in
        equal (exact ()) (vec [| 2. |]) dy);
  ]

(* Complex parameters *)

let cvec a =
  Nx.create Nx.complex128
    [| Array.length a |]
    (Array.map (fun (re, im) -> { Complex.re; im }) a)

let z3 () = cvec [| (1.1, 0.5); (-0.7, 1.3); (0.4, -0.9) |]
let c3 () = cvec [| (0.6, -1.1); (1.4, 0.3); (-0.8, 0.7) |]

let cscale x s =
  Nx.mul x (Nx.full Nx.complex128 (Nx.shape x) { Complex.re = s; im = 0. })

let modulus2 z = Nx.sum (Nx.square (Nx.magnitude f64 z))

let complex_tests =
  [
    test "the gradient of |z - c|² is 2 (z - c)" (fun () ->
        let c = c3 () in
        equal (close ())
          (cscale (Nx.sub (z3 ()) c) 2.)
          (Rune.grad' (fun z -> modulus2 (Nx.sub z c)) (z3 ())));
    test "the gradient of Re (c z) is conj c" (fun () ->
        let c = c3 () in
        equal (close ()) (Nx.conjugate c)
          (Rune.grad' (fun z -> Nx.sum (Nx.real f64 (Nx.mul c z))) (z3 ())));
    test "a complex objective is differentiated through its real part"
      (fun () ->
        let z = z3 () in
        equal (close ())
          (Nx.conjugate (cscale z 2.))
          (Rune.grad' (fun z -> Nx.sum (Nx.mul z z)) z));
    test "a step against the gradient descends by lr |g|² to first order"
      (fun () ->
        let loss z = modulus2 (Nx.sub (Nx.mul z (c3 ())) z) in
        let z = z3 () in
        let g = Rune.grad' loss z in
        let before = Nx.item [] (loss z)
        and after = Nx.item [] (loss (Nx.sub z (cscale g 1e-5))) in
        equal (float 1e-7) (-1e-5 *. Oracle.dot g g) (after -. before));
    test "vjp's pullback is the adjoint of jvp on complex tensors" (fun () ->
        let f z = Nx.mul (Nx.abs z) (Nx.add z (Nx.conjugate (Nx.mul z z))) in
        let z = z3 ()
        and v = c3 ()
        and w = cvec [| (0.3, 0.2); (-1., 0.5); (0.7, -0.4) |] in
        let _, dy = Rune.jvp' f z v in
        let _, pb = Rune.vjp' f z in
        equal (float 1e-10) (Oracle.dot w dy) (Oracle.dot (pb w) v));
    test "a complex custom_vjp equals its function's pullback" (fun () ->
        let sin =
          Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun z ->
              (Nx.sin z, fun ct -> Nx.mul ct (Nx.conjugate (Nx.cos z))))
        in
        let w = c3 () in
        equal (close ())
          (snd (Rune.vjp' Nx.sin (z3 ())) w)
          (snd (Rune.vjp' sin (z3 ())) w));
    test "jacrev' is jacfwd' on a complex-differentiable function" (fun () ->
        let f z = Nx.mul (Nx.exp z) (Nx.flip ~axes:[ 0 ] z) in
        equal (close ()) (Rune.jacfwd' f (z3 ())) (Rune.jacrev' f (z3 ())));
    test "the Hessian-vector product of |z|² is 2v" (fun () ->
        let v = c3 () in
        equal (close ()) (cscale v 2.)
          (snd (Rune.jvp' (Rune.grad' modulus2) (z3 ()) v)));
    test "check_grads accepts a complex parameter" (fun () ->
        is_ok (Rune.check_grads Nx.Ptree.tensor modulus2 (z3 ())));
  ]

(* The operations of a gradient *)

(* The names of the operations [f ()] issues, sorted, and its result. *)
let operations f =
  let names = ref [] in
  let y =
    Nx.Op.intercept
      {
        run =
          (fun op ->
            names := Nx.Op.name op :: !names;
            Nx.Op.eval op);
        claims = (fun _ -> true);
      }
      f
  in
  (y, List.sort compare !names)

let operation_tests =
  [
    test
      "a pair of cancelling transposes adds no copy and no arithmetic to a \
       gradient" (fun () ->
        let q = Nx.ones f64 [| 2; 2; 2; 3; 4 |] in
        let attention k =
          let scores =
            Nx.matmul q (Nx.swapaxes 3 4 (Nx.unsqueeze ~axes:[ 2 ] k))
          in
          Nx.sum (Nx.mul scores scores)
        in
        let heads x = Nx.swapaxes 1 2 (Nx.reshape [| 2; 3; 2; 4 |] x) in
        let paired t = Nx.swapaxes 1 2 (Nx.swapaxes 1 2 t) in
        let x =
          Nx.init f64 [| 2; 3; 8 |] (fun i ->
              Float.of_int (((i.(1) * 8) + i.(2)) mod 7))
        in
        let g, plain =
          operations (fun () -> Rune.grad' (fun x -> attention (heads x)) x)
        in
        let g', with_pair =
          operations (fun () ->
              Rune.grad' (fun x -> attention (paired (heads x))) x)
        in
        equal ~msg:"gradient" (close ()) g g';
        let count p names = List.length (List.filter p names) in
        let copy n = n = "contiguous" in
        let movement = function
          | "reshape" | "expand" | "permute" | "shrink" | "flip"
          | "sliding_window" ->
              true
          | _ -> false
        in
        let arithmetic n = not (copy n || movement n) in
        equal ~msg:"copies" int (count copy plain) (count copy with_pair);
        equal ~msg:"arithmetic" int (count arithmetic plain)
          (count arithmetic with_pair));
    test "a ragged take's values differentiate as the rows they repeat"
      (fun () ->
        let offsets = Nx.create Nx.int64 [| 4 |] [| 0L; 2L; 3L; 6L |] in
        let indices = Nx.create Nx.int64 [| 3 |] [| 2L; 0L; 2L |] in
        let f x =
          Nx.sum
            (Nx_ragged.values
               (Nx_ragged.take ~indices (Nx_ragged.v ~offsets x)))
        in
        equal (exact ())
          (vec [| 1.; 1.; 0.; 2.; 2.; 2. |])
          (Rune.grad' f (vec [| 1.; 2.; 3.; 4.; 5.; 6. |])));
    test "rows grouped by ids differentiate as the rows they permute" (fun () ->
        let ids = Nx.create Nx.int64 [| 4 |] [| 1L; 0L; 1L; 5L |] in
        let f x =
          Nx.sum
            (Nx.mul
               (vec [| 1.; 2.; 3.; 4. |])
               (Nx_ragged.values (Nx_ragged.of_ids ~segments:2 ids x)))
        in
        equal (exact ())
          (vec [| 2.; 1.; 3.; 4. |])
          (Rune.grad' f (vec [| 1.; -2.; 3.; 0.5 |])));
  ]

(* Laws over generated programs *)

let point = Expr.point

let laws =
  [
    prop "jvp and vjp's pullback are adjoint"
      Gen.(quad Expr.gen point point point)
      (fun (p, x, v, w) ->
        let f = Expr.eval p in
        let _, dy = Rune.jvp' f x v in
        let _, pb = Rune.vjp' f x in
        equal
          (float_rel ~rel:1e-10 ~abs:1e-12)
          (Oracle.dot w dy)
          (Oracle.dot (pb w) v));
    prop "jvp agrees with a central difference"
      Gen.(triple Expr.smooth point point)
      (fun (p, x, v) ->
        let f = Expr.eval p in
        equal
          (Oracle.tensor ~rel:1e-5 ~abs:1e-7 ())
          (Oracle.central ~eps:1e-6 f x v)
          (snd (Rune.jvp' f x v)));
    prop "jvp along v is the gradient paired with v"
      Gen.(triple Expr.gen point point)
      (fun (p, x, v) ->
        let f = Expr.objective p in
        equal
          (float_rel ~rel:1e-10 ~abs:1e-12)
          (Oracle.dot (Rune.grad' f x) v)
          (Nx.item [] (snd (Rune.jvp' f x v))));
    prop "jacfwd' equals jacrev'"
      Gen.(pair Expr.gen point)
      (fun (p, x) ->
        let f = Expr.eval p in
        equal
          (Oracle.tensor ~rel:1e-10 ~abs:1e-12 ())
          (Rune.jacfwd' f x) (Rune.jacrev' f x));
    prop "a Hessian is symmetric"
      Gen.(pair Expr.gen point)
      (fun (p, x) ->
        let h =
          Nx.reshape [| 6; 6 |] (Rune.jacfwd' (Rune.grad' (Expr.objective p)) x)
        in
        equal (Oracle.tensor ~rel:1e-10 ~abs:1e-12 ()) (Nx.transpose h) h);
    prop "value_and_grad is the value and the gradient"
      Gen.(pair Expr.gen point)
      (fun (p, x) ->
        let f = Expr.objective p in
        let v, g = Rune.value_and_grad' f x in
        equal ~msg:"value" (exact ()) (f x) v;
        equal ~msg:"gradient" (exact ()) (Rune.grad' f x) g);
    prop "a pullback is linear"
      Gen.(quad Expr.gen point point point)
      (fun (p, x, c1, c2) ->
        let _, pb = Rune.vjp' (Expr.eval p) x in
        equal
          (Oracle.tensor ~rel:1e-10 ~abs:1e-12 ())
          (Nx.add (Nx.mul_s (pb c1) 2.) (Nx.mul_s (pb c2) (-3.)))
          (pb (Nx.add (Nx.mul_s c1 2.) (Nx.mul_s c2 (-3.)))));
  ]

(* The forms of one tensor *)

let shorthand_tests =
  [
    prop "grad' is grad at one tensor"
      Gen.(pair Expr.gen point)
      (fun (p, x) ->
        let f = Expr.objective p in
        equal (exact ()) (Rune.grad Nx.Ptree.tensor f x) (Rune.grad' f x));
    prop "value_and_grad' is value_and_grad at one tensor"
      Gen.(pair Expr.gen point)
      (fun (p, x) ->
        let f = Expr.objective p in
        let v, g = Rune.value_and_grad Nx.Ptree.tensor f x in
        let v', g' = Rune.value_and_grad' f x in
        equal ~msg:"value" (exact ()) v v';
        equal ~msg:"gradient" (exact ()) g g');
    prop "vjp' is vjp at one tensor"
      Gen.(triple Expr.gen point point)
      (fun (p, x, c) ->
        let f = Expr.eval p in
        let y, pb = Rune.vjp Nx.Ptree.tensor Nx.Ptree.tensor f x in
        let y', pb' = Rune.vjp' f x in
        equal ~msg:"value" (exact ()) y y';
        equal ~msg:"pullback" (exact ()) (pb c) (pb' c));
    prop "jvp' is jvp at one tensor"
      Gen.(triple Expr.gen point point)
      (fun (p, x, v) ->
        let f = Expr.eval p in
        let y, dy = Rune.jvp Nx.Ptree.tensor Nx.Ptree.tensor f x v in
        let y', dy' = Rune.jvp' f x v in
        equal ~msg:"value" (exact ()) y y';
        equal ~msg:"tangent" (exact ()) dy dy');
    test "each form names itself in its errors" (fun () ->
        let not_scalar x = Nx.mul x x and x = vec [| 1.; 2. |] in
        starts "Rune.grad'" (Oracle.message (fun () -> Rune.grad' not_scalar x));
        starts "Rune.grad"
          (Oracle.message (fun () -> Rune.grad Nx.Ptree.tensor not_scalar x));
        starts "Rune.value_and_grad'"
          (Oracle.message (fun () -> Rune.value_and_grad' not_scalar x));
        starts "Rune.value_and_grad"
          (Oracle.message (fun () ->
               Rune.value_and_grad Nx.Ptree.tensor not_scalar x)));
  ]

let () =
  exit
    (run "Rune derivatives"
       [
         group "grad" grad_tests;
         group "value" value_tests;
         group "vjp" vjp_tests;
         group "jvp" jvp_tests;
         group "jacobians" jacobian_tests;
         group "check_grads" check_grads_tests;
         group "detach" detach_tests;
         group "escapes" escapes_tests;
         group "pullbacks" pullback_tests;
         group "fresh derivatives" fresh_tests;
         group "edges" edge_tests;
         group "errors" error_tests;
         group "complex" complex_tests;
         group "operations" operation_tests;
         group "laws" laws;
         group "shorthands" shorthand_tests;
       ])
