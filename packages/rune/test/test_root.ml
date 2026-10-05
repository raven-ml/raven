(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.root: a value stated to solve an equation, whose derivative is the
   implicit function theorem's at the value. The trusted side is a closed-form
   root and its finite differences, or the theorem's derivative computed by
   hand. *)

open Windtrap

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let exact () = Oracle.tensor ()
let close () = Oracle.tensor ~rel:1e-7 ~abs:1e-9 ()
let one = Nx.Ptree.tensor
let lane i x = Nx.get [ i ] x
let stack n f = Nx.stack (List.init n f)

(* Square roots *)

(* [x ↦ x² − a] vanishes at √a, which Newton's method from [a + 1] finds. *)
let newton a =
  Rune.iterate' ~max:100
    ~until:(fun x -> Nx.less_s (Nx.max (Nx.abs (Nx.sub (Nx.mul x x) a))) 1e-13)
    ~f:(fun x -> Nx.mul_s (Nx.add x (Nx.div a x)) 0.5)
    (Nx.add_s a 1.)

let sqrt_root a =
  Rune.root one ~residual:(fun x -> Nx.sub (Nx.mul x x) a) (fun () -> newton a)

let loss a = Nx.sum (Nx.sin (sqrt_root a))
let closed a = Nx.sum (Nx.sin (Nx.sqrt a))

let positive =
  Gen.(
    map
      (fun l -> vec (Array.of_list l))
      (list ~size:(int_range 1 4) (float_range 0.5 3.)))

let direction a = Nx.sin (Nx.mul_s a 7.)

let derivative_tests =
  [
    test "the value is the solve's" (fun () ->
        let a = vec [| 0.5; 2.; 3. |] in
        equal (exact ()) (newton a) (sqrt_root a));
    prop "jvp is the finite difference of the closed-form root" positive
      (fun a ->
        let v = direction a in
        equal (close ())
          (Oracle.central ~eps:1e-6 closed a v)
          (snd (Rune.jvp' loss a v)));
    prop "grad is the finite difference of the closed-form root" positive
      (fun a ->
        let v = direction a in
        equal (close ())
          (Oracle.central ~eps:1e-6 closed a v)
          (scalar (Oracle.dot (Rune.grad' loss a) v)));
    prop "second order: jvp of grad is the finite difference of the gradient"
      positive (fun a ->
        let v = direction a in
        equal (close ())
          (Oracle.central ~eps:1e-5 (Rune.grad' closed) a v)
          (snd (Rune.jvp' (Rune.grad' loss) a v)));
    prop "second order: grad of grad is the finite difference of the gradient"
      positive (fun a ->
        let v = direction a in
        let hv = Oracle.central ~eps:1e-5 (Rune.grad' closed) a v in
        equal (close ()) hv
          (Rune.grad' (fun a -> Nx.sum (Nx.mul (Rune.grad' loss a) v)) a));
    test "vmap solves each lane's own system" (fun () ->
        let a = Nx.create f64 [| 3; 2 |] [| 0.5; 2.; 3.; 1.5; 0.8; 2.5 |] in
        equal (close ()) (Nx.sqrt a) (Rune.vmap' sqrt_root a));
    test "vmap of grad is each lane's gradient" (fun () ->
        let a = Nx.create f64 [| 3; 2 |] [| 0.5; 2.; 3.; 1.5; 0.8; 2.5 |] in
        equal (close ())
          (stack 3 (fun i -> Rune.grad' closed (lane i a)))
          (Rune.vmap' (Rune.grad' loss) a));
    test "grad of vmap is each lane's gradient" (fun () ->
        let a = Nx.create f64 [| 3; 2 |] [| 0.5; 2.; 3.; 1.5; 0.8; 2.5 |] in
        equal (close ())
          (Rune.grad' (fun a -> Nx.sum (Nx.sin (Nx.sqrt a))) a)
          (Rune.grad' (fun a -> Nx.sum (Nx.sin (Rune.vmap' sqrt_root a))) a));
    test "vmap of jvp is each lane's tangent" (fun () ->
        let a = Nx.create f64 [| 3; 2 |] [| 0.5; 2.; 3.; 1.5; 0.8; 2.5 |] in
        let v = direction a in
        equal (close ())
          (stack 3 (fun i -> snd (Rune.jvp' closed (lane i a) (lane i v))))
          (Rune.vmap
             Nx.Ptree.(tensor @-> tensor @-> returns tensor)
             (fun a v -> snd (Rune.jvp' loss a v))
             a v));
  ]

(* What a derivative reads *)

(* [x ↦ x² − a] at the [x̂] two Newton steps from [a + 1] reach: the derivative
   of [x̂] along [a] is the theorem's at [x̂], [1 / 2x̂], and not the derivative of
   the two steps. *)
let early a =
  let step x = Nx.mul_s (Nx.add x (Nx.div a x)) 0.5 in
  step (step (Nx.add_s a 1.))

let reading_tests =
  [
    test "a solve may return a tracked value as it is" (fun () ->
        (* x² − c² = 0 at x = c: the root's derivative along c is 1, whatever
           solve returns it from. *)
        let root c =
          Rune.root one
            ~residual:(fun x -> Nx.sub (Nx.mul x x) (Nx.mul c c))
            (fun () -> c)
        in
        let c = vec [| 0.5; 2.; 3. |] in
        equal ~msg:"grad" (close ()) (Nx.ones_like c)
          (Rune.grad' (fun c -> Nx.sum (root c)) c);
        equal ~msg:"jvp" (close ()) (Nx.ones_like c)
          (snd (Rune.jvp' root c (Nx.ones_like c))));
    test "the derivative is taken at the returned point" (fun () ->
        let a = scalar 2. in
        let x = early a in
        let early_root a =
          Rune.root one
            ~residual:(fun x -> Nx.sub (Nx.mul x x) a)
            (fun () -> early a)
        in
        equal (close ())
          (Nx.recip (Nx.mul_s x 2.))
          (snd (Rune.jvp' early_root a (scalar 1.)));
        equal (close ()) (Nx.recip (Nx.mul_s x 2.)) (Rune.grad' early_root a));
    test "a value the residual captures contributes to the derivative"
      (fun () ->
        (* x³ + c x − 1 = 0 in x, with c captured: dx/dc = −x / (3x² + c). *)
        let c = scalar 0.7 in
        let solve_for c =
          Rune.iterate' ~max:100
            ~until:(fun x ->
              Nx.less_s
                (Nx.abs
                   (Nx.sub_s (Nx.add (Nx.mul (Nx.mul x x) x) (Nx.mul c x)) 1.))
                1e-14)
            ~f:(fun x ->
              let r =
                Nx.sub_s (Nx.add (Nx.mul (Nx.mul x x) x) (Nx.mul c x)) 1.
              in
              Nx.sub x (Nx.div r (Nx.add (Nx.mul_s (Nx.mul x x) 3.) c)))
            (scalar 1.)
        in
        let cubic c =
          Rune.root one
            ~residual:(fun x ->
              Nx.sub_s (Nx.add (Nx.mul (Nx.mul x x) x) (Nx.mul c x)) 1.)
            (fun () -> solve_for c)
        in
        let x = solve_for c in
        let expected =
          Nx.neg (Nx.div x (Nx.add (Nx.mul_s (Nx.mul x x) 3.) c))
        in
        equal (close ()) expected (Rune.grad' cubic c);
        equal (close ()) expected (snd (Rune.jvp' cubic c (scalar 1.))));
    test "solve's branches on the values it reads are not differentiated"
      (fun () ->
        (* The solve picks a start by reading [a]: differentiating it would need
           a derivative of the branch. *)
        let a = scalar 2. in
        let branching a =
          Rune.root one
            ~residual:(fun x -> Nx.sub (Nx.mul x x) a)
            (fun () -> if Nx.item [] a > 1. then newton a else Nx.ones f64 [||])
        in
        equal (close ())
          (Nx.recip (Nx.mul_s (Nx.sqrt a) 2.))
          (Rune.grad' branching a));
  ]

(* Structures and linear solves *)

(* u + v = s and u − v = d: u = (s + d) / 2, v = (s − d) / 2. *)
let pair = Nx.Ptree.(pair tensor tensor)

let halves (s, d) =
  Rune.root pair
    ~residual:(fun (u, v) -> (Nx.sub (Nx.add u v) s, Nx.sub (Nx.sub u v) d))
    (fun () -> (Nx.mul_s (Nx.add s d) 0.5, Nx.mul_s (Nx.sub s d) 0.5))

let halves_loss sd =
  let u, v = halves sd in
  Nx.sum (Nx.add (Nx.mul u u) (Nx.mul_s v 3.))

(* Conjugate gradients for a symmetric positive definite [op], from zero. *)
let cg op b =
  let dot x y = Nx.sum (Nx.mul x y) in
  let x, _ =
    Rune.iterate
      Nx.Ptree.(pair tensor (pair tensor tensor))
      ~max:50
      ~until:(fun (_, (r, _)) -> Nx.less_s (dot r r) 1e-28)
      ~f:(fun (x, (r, p)) ->
        let ap = op p in
        let alpha = Nx.div (dot r r) (dot p ap) in
        let x = Nx.add x (Nx.mul alpha p) in
        let r' = Nx.sub r (Nx.mul alpha ap) in
        let beta = Nx.div (dot r' r') (dot r r) in
        (x, (r', Nx.add r' (Nx.mul beta p))))
      (Nx.zeros_like b, (b, b))
  in
  x

let spd () =
  Nx.create f64 [| 3; 3 |] [| 4.; 1.; 0.5; 1.; 3.; 0.2; 0.5; 0.2; 2. |]

let rhs () = vec [| 1.; -2.; 0.5 |]

let solved ?linear_solve (a, b) =
  Rune.root ?linear_solve one
    ~residual:(fun x -> Nx.sub (Nx.matmul a x) b)
    (fun () -> cg (Nx.matmul a) b)

let solved_loss ?linear_solve ab = Nx.sum (Nx.sin (solved ?linear_solve ab))
let closed_solve (a, b) = Nx.sum (Nx.sin (Nx.solve a (Nx.reshape [| 3; 1 |] b)))

let structure_tests =
  [
    test "a structure's root differentiates every leaf" (fun () ->
        let s = vec [| 1.; 2. |] and d = vec [| 0.5; -1. |] in
        let gs, gd = Rune.grad pair halves_loss (s, d) in
        (* d/ds (u² + 3v) = u + 3/2, d/dd = u − 3/2. *)
        let u = Nx.mul_s (Nx.add s d) 0.5 in
        equal ~msg:"s" (close ()) (Nx.add_s u 1.5) gs;
        equal ~msg:"d" (close ()) (Nx.sub_s u 1.5) gd);
    test "a linear solve's gradient is the closed form's" (fun () ->
        let ga, gb = Rune.grad pair solved_loss (spd (), rhs ()) in
        let ga', gb' = Rune.grad pair closed_solve (spd (), rhs ()) in
        equal ~msg:"a" (close ()) ga' ga;
        equal ~msg:"b" (close ()) gb' gb);
    test "a given linear solve serves both modes" (fun () ->
        let f = solved_loss ~linear_solve:cg in
        let ga, gb = Rune.grad pair f (spd (), rhs ()) in
        let ga', gb' = Rune.grad pair closed_solve (spd (), rhs ()) in
        equal ~msg:"a" (close ()) ga' ga;
        equal ~msg:"b" (close ()) gb' gb;
        let v = (Nx.mul_s (spd ()) 0.1, vec [| 0.3; 0.1; -0.2 |]) in
        equal ~msg:"jvp" (close ())
          (snd (Rune.jvp pair one closed_solve (spd (), rhs ()) v))
          (snd (Rune.jvp pair one f (spd (), rhs ()) v)));
    test "the residual's result must have the solution's structure" (fun () ->
        raises
          (Invalid_argument
             "Rune.root: 0: shape [3] in the residual's result, [2] in the \
              solution") (fun () ->
            Rune.grad'
              (fun s ->
                let u, _ =
                  Rune.root pair
                    ~residual:(fun (u, v) ->
                      (Nx.concatenate ~axis:0 [ u; Nx.slice [ R (0, 1) ] v ], v))
                    (fun () -> (s, s))
                in
                Nx.sum u)
              (vec [| 1.; 2. |])));
  ]

(* Linear solves under maps *)

(* Lane [θ]'s system [A(θ) x = b(θ)]: [A] symmetric with its diagonal 2 ± 0.5
   and its rows' other entries summing below 0.5, so its eigenvalues lie in [1,
   3]. *)
let system th =
  let s = Nx.sin th and c = Nx.cos th in
  let k v = Nx.full_like th v in
  let row l = Nx.stack l in
  let a01 = Nx.mul_s c 0.25 and a12 = Nx.mul_s s 0.1 in
  let a =
    Nx.stack
      [
        row [ Nx.add_s (Nx.mul_s s 0.5) 2.; a01; k 0.2 ];
        row [ a01; Nx.rsub_s 2. (Nx.mul_s s 0.5); a12 ];
        row [ k 0.2; a12; Nx.add_s (Nx.mul_s c 0.25) 2. ];
      ]
  in
  (a, Nx.stack [ c; th; k 1. ])

let solve_dense a b = Nx.reshape [| 3 |] (Nx.solve a (Nx.reshape [| 3; 1 |] b))

let system_root ?linear_solve th =
  let a, b = system th in
  Rune.root ?linear_solve one
    ~residual:(fun x -> Nx.sub (Nx.matmul a x) b)
    (fun () -> solve_dense a b)

let system_loss ?linear_solve th =
  Nx.sum (Nx.sin (system_root ?linear_solve th))

let system_closed th =
  let a, b = system th in
  Nx.sum (Nx.sin (solve_dense a b))

(* The operator's matrix, its columns mapped over the basis. *)
let columns op n = Nx.matrix_transpose (Rune.vmap' op (Nx.eye f64 n))
let mapped_dense op b = solve_dense (columns op 3) b

(* The columns mapped twice, and two corrections that vanish when the operator
   gives each lane its own product: one applied to a constant no map holds, one
   to a value shared across a map the solve opens. *)
let mapped_twice op b =
  let basis = Nx.reshape [| 3; 1; 3 |] (Nx.eye f64 3) in
  let j =
    Nx.matrix_transpose
      (Nx.reshape [| 3; 3 |] (Rune.vmap' (Rune.vmap' op) basis))
  in
  let ones = Nx.ones f64 [| 3 |] in
  let v = Nx.add ones (solve_dense j (Nx.sub b (op ones))) in
  let shared =
    Nx.get [ 0 ] (Rune.vmap' (fun _ -> op v) (Nx.zeros f64 [| 2 |]))
  in
  Nx.add v (solve_dense j (Nx.sub b shared))

(* Richardson's iteration, which takes more trips on worse-conditioned lanes. *)
let richardson op b =
  Rune.iterate' ~max:400
    ~until:(fun v -> Nx.less_s (Nx.max (Nx.abs (Nx.sub b (op v)))) 1e-13)
    ~f:(fun v -> Nx.add v (Nx.mul_s (Nx.sub b (op v)) 0.3))
    (Nx.zeros_like b)

(* Richardson's iterations for [b] and [8 b] under a map the solve opens: the
   second takes more trips. *)
let richardson_mapped op b =
  let bs = Nx.stack [ b; Nx.mul_s b 8. ] in
  Nx.get [ 0 ] (Rune.vmap' (richardson op) bs)

let mapped_solves =
  [
    ("a linear solve that maps over a basis", Some mapped_dense);
    ("the default linear solve", None);
    ( "a linear solve that maps twice and applies its operator to shared values",
      Some mapped_twice );
    ("a linear solve whose iteration stops lanes apart", Some richardson);
    ( "a linear solve that maps iterations stopping apart",
      Some richardson_mapped );
  ]

let thetas n =
  Gen.(
    map
      (fun l -> vec (Array.of_list l))
      (list ~size:(int_range n n) (float_range (-1.5) 1.5)))

let some_thetas =
  Gen.(
    let* n = int_range 1 4 in
    thetas n)

let per_lane f ths = stack (Nx.dim 0 ths) (fun i -> f (lane i ths))

let mapped_props =
  List.concat_map
    (fun (name, linear_solve) ->
      let loss = system_loss ?linear_solve in
      [
        prop ~count:20 ("grad of vmap: " ^ name) some_thetas (fun ths ->
            cover "one lane" (Nx.dim 0 ths = 1);
            let g = Rune.grad' (fun ths -> Nx.sum (Rune.vmap' loss ths)) ths in
            equal ~msg:"closed form" (close ())
              (per_lane (Rune.grad' system_closed) ths)
              g;
            equal ~msg:"finite differences" (close ())
              (per_lane
                 (fun th ->
                   Oracle.central ~eps:1e-6 system_closed th (scalar 1.))
                 ths)
              g);
        prop ~count:20 ("jvp of vmap: " ^ name) some_thetas (fun ths ->
            let v = Nx.cos ths in
            equal (close ())
              (Nx.mul (per_lane (Rune.grad' system_closed) ths) v)
              (snd (Rune.jvp' (Rune.vmap' loss) ths v)));
        prop ~count:10
          ("grad of vmap of vmap: " ^ name)
          Gen.(map (Nx.reshape [| 2; 2 |]) (thetas 4))
          (fun ths ->
            equal (close ())
              (Nx.reshape [| 2; 2 |]
                 (per_lane (Rune.grad' system_closed) (Nx.reshape [| 4 |] ths)))
              (Rune.grad'
                 (fun ths -> Nx.sum (Rune.vmap' (Rune.vmap' loss) ths))
                 ths));
        prop ~count:10
          ("vmap of grad of vmap: " ^ name)
          Gen.(map (Nx.reshape [| 2; 2 |]) (thetas 4))
          (fun ths ->
            equal (close ())
              (Nx.reshape [| 2; 2 |]
                 (per_lane (Rune.grad' system_closed) (Nx.reshape [| 4 |] ths)))
              (Rune.vmap'
                 (Rune.grad' (fun ths -> Nx.sum (Rune.vmap' loss ths)))
                 ths));
      ])
    mapped_solves

let second_order_props =
  List.map
    (fun (name, linear_solve) ->
      prop ~count:10 ("grad of grad of vmap: " ^ name) some_thetas (fun ths ->
          let v = Nx.sin (Nx.mul_s ths 3.) in
          let grad_of loss ths =
            Rune.grad' (fun ths -> Nx.sum (Rune.vmap' loss ths)) ths
          in
          equal
            (Oracle.tensor ~rel:1e-5 ~abs:1e-7 ())
            (Oracle.central ~eps:1e-5 (grad_of system_closed) ths v)
            (Rune.grad'
               (fun ths ->
                 Nx.sum (Nx.mul (grad_of (system_loss ?linear_solve) ths) v))
               ths)))
    [ List.nth mapped_solves 0; List.nth mapped_solves 1 ]

let mapped_tests =
  [
    test "jit of grad of vmap is eager" (fun () ->
        let ths = vec [| 0.3; -1.1; 0.7 |] in
        let f ths = Nx.sum (Rune.vmap' system_loss ths) in
        equal (close ()) (Rune.grad' f ths) (Rune.jit' (Rune.grad' f) ths));
    test "a map of no lanes" (fun () ->
        let ths = Nx.zeros f64 [| 0 |] in
        equal (exact ()) (Nx.zeros f64 [| 0 |])
          (Rune.grad'
             (fun ths ->
               Nx.sum (Rune.vmap' (system_loss ~linear_solve:mapped_dense) ths))
             ths));
    test "an inner map of one lane" (fun () ->
        let one_lane op b =
          Nx.get [ 0 ] (Rune.vmap' (mapped_dense op) (Nx.reshape [| 1; 3 |] b))
        in
        let ths = vec [| 0.3; -1.1 |] in
        equal (close ())
          (per_lane (Rune.grad' system_closed) ths)
          (Rune.grad'
             (fun ths ->
               Nx.sum (Rune.vmap' (system_loss ~linear_solve:one_lane) ths))
             ths));
    test "a scalar root under vmap" (fun () ->
        let root th =
          Rune.root one
            ~residual:(fun x ->
              Nx.sub (Nx.mul (Nx.add_s (Nx.sin th) 2.) x) (Nx.cos th))
            (fun () -> Nx.div (Nx.cos th) (Nx.add_s (Nx.sin th) 2.))
        in
        let closed th = Nx.div (Nx.cos th) (Nx.add_s (Nx.sin th) 2.) in
        let ths = vec [| 0.3; -1.1; 0.9 |] in
        let solves =
          [
            None;
            Some
              (fun op b ->
                Nx.div b (Nx.get [ 0 ] (Rune.vmap' op (Nx.ones f64 [| 1 |]))));
          ]
        in
        List.iter
          (fun linear_solve ->
            let root th =
              match linear_solve with
              | None -> root th
              | Some linear_solve ->
                  Rune.root ~linear_solve one
                    ~residual:(fun x ->
                      Nx.sub (Nx.mul (Nx.add_s (Nx.sin th) 2.) x) (Nx.cos th))
                    (fun () -> closed th)
            in
            equal (close ())
              (per_lane (Rune.grad' closed) ths)
              (Rune.grad' (fun ths -> Nx.sum (Rune.vmap' root ths)) ths))
          solves);
    test "a root of two leaves under vmap" (fun () ->
        (* u + v = c, u − 2v = s, in u of shape [2] and a scalar v: u is the
           mean-weighted pair, v the rest. *)
        let pair = Nx.Ptree.(pair tensor tensor) in
        let root th =
          let c = Nx.stack [ Nx.cos th; th ] and s = Nx.sin th in
          let u, v =
            Rune.root pair
              ~residual:(fun (u, v) ->
                ( Nx.sub (Nx.add u (Nx.broadcast_to [| 2 |] v)) c,
                  Nx.sub (Nx.sub (Nx.sum u) (Nx.mul_s v 2.)) s ))
              (fun () ->
                (* 2 + 2·... solved in closed form: Σu = Σc − 2v, so Σc − 4v =
                   s. *)
                let v = Nx.mul_s (Nx.sub (Nx.sum c) s) 0.25 in
                (Nx.sub c (Nx.broadcast_to [| 2 |] v), v))
          in
          Nx.add (Nx.sum (Nx.sin u)) (Nx.mul_s v 3.)
        in
        let ths = vec [| 0.3; -1.1; 0.9 |] in
        equal (close ())
          (per_lane
             (fun th -> Oracle.central ~eps:1e-6 root th (scalar 1.))
             ths)
          (Rune.grad' (fun ths -> Nx.sum (Rune.vmap' root ths)) ths));
  ]

(* Refusals *)

let refused message linear_solve =
  raises (Invalid_argument message) (fun () ->
      Rune.grad'
        (fun ths -> Nx.sum (Rune.vmap' (system_loss ~linear_solve) ths))
        (vec [| 0.3; -1.1 |]))

let stash = ref None

let refusal_cases =
  [
    ( "the operator called after linear_solve returned",
      fun () ->
        let ths = vec [| 0.3; -1.1 |] in
        ignore
          (Rune.grad'
             (fun ths ->
               Nx.sum
                 (Rune.vmap'
                    (system_loss ~linear_solve:(fun op b ->
                         stash := Some op;
                         mapped_dense op b))
                    ths))
             ths);
        raises
          (Invalid_argument
             "Rune.root: linear_solve's operator was applied after \
              linear_solve returned, or inside a Rune.jit it called") (fun () ->
            (Option.get !stash) (vec [| 1.; 2.; 3. |])) );
    ( "the operator under jvp inside linear_solve",
      fun () ->
        refused
          "Rune.root: linear_solve's operator cannot be differentiated inside \
           linear_solve" (fun op b ->
            ignore (Rune.jvp' op b b);
            mapped_dense op b) );
    ( "the operator inside a Rune.jit linear_solve calls",
      fun () ->
        refused
          "Rune.root: linear_solve's operator was applied after linear_solve \
           returned, or inside a Rune.jit it called" (fun op b ->
            mapped_dense (Rune.jit' op) b) );
    ( "a residual that reads other lanes of the root's map",
      fun () ->
        let a = Rune.axis () in
        raises
          (Invalid_argument
             "Rune.root: the residual reads other lanes of the map, so the \
              lanes' systems are not separate") (fun () ->
            Rune.grad'
              (fun ths ->
                Nx.sum
                  (Rune.vmap' ~axis:a
                     (fun th ->
                       Rune.root one
                         ~residual:(fun x ->
                           Nx.sub (Nx.mul_s x 2.) (Nx.sum (Rune.lanes a th)))
                         (fun () -> th))
                     ths))
              (vec [| 0.3; -1.1 |])) );
  ]

let refusal_tests =
  List.map (fun (name, f) -> test (name ^ " is refused") f) refusal_cases

(* Totals *)

let total : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

let counted a =
  Rune.root one
    ~residual:(fun x ->
      Rune.Total.add total (scalar 100.);
      Nx.sub (Nx.mul x x) a)
    (fun () ->
      Rune.Total.add total (scalar 1.);
      newton a)

let total_tests =
  [
    test "solve's additions count once, the residual's none, under grad"
      (fun () ->
        let _, n =
          Rune.Total.collect total ~zero:(scalar 0.) (fun () ->
              Rune.grad' (fun a -> Nx.sum (counted a)) (vec [| 2.; 3. |]))
        in
        equal (exact ()) (scalar 1.) n);
    test "a scope inside grad counts solve's additions once" (fun () ->
        (* With [n] one, the gradient is that of the sum of the roots. *)
        let a = vec [| 2.; 3. |] in
        let g =
          Rune.grad'
            (fun a ->
              let x, n =
                Rune.Total.collect total ~zero:(scalar 0.) (fun () -> counted a)
              in
              Nx.mul n (Nx.sum x))
            a
        in
        equal (close ()) (Nx.recip (Nx.mul_s (Nx.sqrt a) 2.)) g);
    test "vmap's lanes each count solve's additions" (fun () ->
        let _, n =
          Rune.Total.collect total ~zero:(scalar 0.) (fun () ->
              Rune.vmap' counted (vec [| 2.; 3.; 5. |]))
        in
        equal (exact ()) (scalar 3.) n);
  ]

(* Compilation *)

let compiled_tests =
  [
    test "jit of grad is eager grad" (fun () ->
        let root a =
          Rune.root one
            ~residual:(fun x -> Nx.sub (Nx.mul x x) a)
            (fun () -> Nx.sqrt a)
        in
        let f a = Nx.sum (Nx.sin (root a)) in
        let a = vec [| 0.5; 2.; 3. |] in
        equal (close ()) (Rune.grad' f a) (Rune.jit' (Rune.grad' f) a));
  ]

let () =
  exit
    (run "Rune.root"
       [
         group "derivatives" derivative_tests;
         group "reading" reading_tests;
         group "structures" structure_tests;
         group "mapped linear solves"
           (mapped_props @ second_order_props @ mapped_tests);
         group "refusals" refusal_tests;
         group "totals" total_tests;
         group "compiled" compiled_tests;
       ])
