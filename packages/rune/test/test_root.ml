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
let lane i x = Nx.slice [ Nx.I i ] x
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
    Nx.slice [ Nx.I 0 ] (Rune.vmap' (fun _ -> op v) (Nx.zeros f64 [| 2 |]))
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
  Nx.slice [ Nx.I 0 ] (Rune.vmap' (richardson op) bs)

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
          Nx.slice [ Nx.I 0 ]
            (Rune.vmap' (mapped_dense op) (Nx.reshape [| 1; 3 |] b))
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
                Nx.div b
                  (Nx.slice [ Nx.I 0 ] (Rune.vmap' op (Nx.ones f64 [| 1 |]))));
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

(* The operator's contract *)

(* Each context a root's derivative can run in, as each lane's gradient of the
   system's loss. *)
let contexts =
  let ths = vec [| 0.3; -1.1 |] in
  [
    ( "grad",
      fun linear_solve -> per_lane (Rune.grad' (system_loss ~linear_solve)) ths
    );
    ( "jvp",
      fun linear_solve ->
        per_lane
          (fun th -> snd (Rune.jvp' (system_loss ~linear_solve) th (scalar 1.)))
          ths );
    ( "grad of vmap",
      fun linear_solve ->
        Rune.grad'
          (fun ths -> Nx.sum (Rune.vmap' (system_loss ~linear_solve) ths))
          ths );
    ( "vmap of grad",
      fun linear_solve ->
        Rune.vmap' (Rune.grad' (system_loss ~linear_solve)) ths );
  ]

(* Solvers that differentiate their operator: its matrix by forward mode, and by
   reverse mode. *)
let by_jacfwd op b = solve_dense (Rune.jacfwd' op b) b
let by_jacrev op b = solve_dense (Rune.jacrev' op b) b

let contract_tests =
  let expected = per_lane (Rune.grad' system_closed) (vec [| 0.3; -1.1 |]) in
  List.concat_map
    (fun (context, run) ->
      [
        test
          (context ^ ": a linear solve may differentiate its operator forward")
          (fun () -> equal (close ()) expected (run by_jacfwd));
        test
          (context ^ ": a linear solve may differentiate its operator backward")
          (fun () -> equal (close ()) expected (run by_jacrev));
      ])
    contexts

(* Refusals *)

let escaped =
  "Rune.root: linear_solve's operator was applied after linear_solve returned, \
   or inside a Rune.jit it called"

let stash = ref None

let refusal_cases =
  List.concat_map
    (fun (context, run) ->
      [
        ( context ^ ": the operator called after linear_solve returned",
          fun () ->
            ignore
              (run (fun op b ->
                   stash := Some op;
                   mapped_dense op b));
            raises (Invalid_argument escaped) (fun () ->
                (Option.get !stash) (vec [| 1.; 2.; 3. |])) );
        ( context ^ ": the operator inside a Rune.jit linear_solve calls",
          fun () ->
            raises (Invalid_argument escaped) (fun () ->
                run (fun op b -> mapped_dense (Rune.jit' op) b)) );
      ])
    contexts
  @ [
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
  @
  let counted_solve op b =
    Rune.Total.add total (scalar 10000.);
    Nx.div b (op (Nx.ones_like b))
  in
  let solved a =
    Rune.root ~linear_solve:counted_solve one
      ~residual:(fun x ->
        Rune.Total.add total (scalar 100.);
        Nx.sub (Nx.mul x x) a)
      (fun () ->
        Rune.Total.add total (scalar 1.);
        newton a)
  in
  let a = vec [| 2.; 3.; 5. |] in
  let lanes = Nx.reshape [| 3; 1 |] a in
  List.map
    (fun (name, n, f) ->
      test (name ^ ": solve's additions count once, linear_solve's none")
        (fun () ->
          let _, got =
            Rune.Total.collect total ~zero:(scalar 0.) (fun () -> f ())
          in
          equal (exact ()) (scalar n) got))
    [
      ("grad", 1., fun () -> ignore (Rune.grad' (fun a -> Nx.sum (solved a)) a));
      ("jvp", 1., fun () -> ignore (Rune.jvp' solved a a));
      ( "vmap of grad",
        3.,
        fun () ->
          ignore (Rune.vmap' (Rune.grad' (fun a -> Nx.sum (solved a))) lanes) );
      ( "grad of vmap",
        3.,
        fun () ->
          ignore (Rune.grad' (fun a -> Nx.sum (Rune.vmap' solved a)) lanes) );
      ( "grad of grad",
        1.,
        fun () ->
          ignore
            (Rune.grad'
               (fun a -> Nx.sum (Rune.grad' (fun a -> Nx.sum (solved a)) a))
               a) );
    ]

(* Second order, forward over forward and reverse over forward *)

let second_order_tests =
  [
    prop "second order: jvp of jvp is the finite difference of jvp" positive
      (fun a ->
        let v = direction a in
        equal (close ())
          (Oracle.central ~eps:1e-5 (fun a -> snd (Rune.jvp' closed a v)) a v)
          (snd (Rune.jvp' (fun a -> snd (Rune.jvp' loss a v)) a v)));
    prop "second order: grad of jvp is the finite difference of the gradient"
      positive (fun a ->
        let v = direction a in
        equal (close ())
          (Oracle.central ~eps:1e-5 (Rune.grad' closed) a v)
          (Rune.grad' (fun a -> snd (Rune.jvp' loss a v)) a));
  ]

(* Where the theorem's derivative is stated *)

(* [a x − b] in [x], at a returned [x̂ = 5] that is not its zero: the tangent [u]
   has [a u + (da x̂ − db) = 0], so [∂x/∂a = −x̂ / a] and [∂x/∂b = 1 / a]. *)
let off_root (a, b) =
  Rune.root one ~residual:(fun x -> Nx.sub (Nx.mul a x) b) (fun () -> scalar 5.)

(* [x² − a] at [a = 0]: [J = 2x̂ = 0], so no tangent [u] has [J u + r = 0]. *)
let at_zero a =
  Rune.root one
    ~residual:(fun x -> Nx.sub (Nx.mul x x) a)
    (fun () -> Nx.zeros_like a)

let stated_tests =
  [
    test "the derivative is taken at a returned point that is not a zero"
      (fun () ->
        let ab = (scalar 2., scalar 3.) in
        let ga, gb = Rune.grad pair (fun ab -> off_root ab) ab in
        equal ~msg:"grad a" (close ()) (scalar (-2.5)) ga;
        equal ~msg:"grad b" (close ()) (scalar 0.5) gb;
        equal ~msg:"jvp a" (close ()) (scalar (-2.5))
          (snd (Rune.jvp pair one off_root ab (scalar 1., scalar 0.)));
        equal ~msg:"jvp b" (close ()) (scalar 0.5)
          (snd (Rune.jvp pair one off_root ab (scalar 0., scalar 1.))));
    test "a tracked value only solve reads has no derivative" (fun () ->
        (* The residual [x² − 2] reads nothing tracked: the result's tangent is
           zero, whatever solve computed it from. *)
        let r a =
          Rune.root one
            ~residual:(fun x -> Nx.sub_s (Nx.mul x x) 2.)
            (fun () -> Nx.mul_s (Nx.sqrt (Nx.div_s a 2.)) (Float.sqrt 2.))
        in
        let a = vec [| 2.; 2. |] in
        equal ~msg:"grad" (exact ()) (Nx.zeros_like a)
          (Rune.grad' (fun a -> Nx.sum (r a)) a);
        equal ~msg:"jvp" (exact ()) (Nx.zeros_like a)
          (snd (Rune.jvp' r a (Nx.ones_like a)));
        equal ~msg:"vmap of grad" (exact ()) (Nx.zeros_like a)
          (Rune.vmap' (Rune.grad' (fun a -> Nx.sum (r a))) a));
    test "grad at a singular derivative is NaN" (fun () ->
        equal (close ()) (scalar Float.nan)
          (Rune.grad' (fun a -> Nx.sum (at_zero a)) (scalar 0.)));
    test "jvp at a singular derivative is NaN" (fun () ->
        equal (close ()) (scalar Float.nan)
          (snd (Rune.jvp' at_zero (scalar 0.) (scalar 1.))));
    test "a scalar root's gradient" (fun () ->
        let a = scalar 3. in
        equal (close ())
          (Nx.recip (Nx.mul_s (Nx.sqrt a) 2.))
          (Rune.grad'
             (fun a ->
               Rune.root one
                 ~residual:(fun x -> Nx.sub (Nx.mul x x) a)
                 (fun () -> newton a))
             a));
    test "a root with a zero-size leaf" (fun () ->
        let r (e, u) =
          Rune.root pair
            ~residual:(fun (x, y) -> (Nx.sub x e, Nx.sub (Nx.mul y y) u))
            (fun () -> (e, Nx.sqrt u))
        in
        let loss eu =
          let x, y = r eu in
          Nx.add (Nx.sum x) (Nx.sum y)
        in
        let e = Nx.zeros f64 [| 0 |] and u = vec [| 2.; 3. |] in
        let ge, gu = Rune.grad pair loss (e, u) in
        equal ~msg:"empty" (exact ()) e ge;
        equal ~msg:"rest" (close ()) (Nx.recip (Nx.mul_s (Nx.sqrt u) 2.)) gu);
    test "a root of no elements" (fun () ->
        let e = Nx.zeros f64 [| 0 |] in
        equal (exact ()) e
          (Rune.grad'
             (fun a ->
               Nx.sum
                 (Rune.root one ~residual:(fun x -> Nx.sub x a) (fun () -> a)))
             e));
    test "the default solve refuses leaves of two dtypes" (fun () ->
        raises_match (Exn.invalid_arg ~substring:"Rune.root") (fun () ->
            Rune.grad'
              (fun a ->
                let x, y =
                  Rune.root
                    Nx.Ptree.(pair tensor tensor)
                    ~residual:(fun (x, y) ->
                      (Nx.sub x (Nx.cast Nx.float32 a), Nx.sub y a))
                    (fun () -> (Nx.cast Nx.float32 a, a))
                in
                Nx.add (Nx.sum (Nx.cast f64 x)) (Nx.sum y))
              (vec [| 1.; 2. |])));
  ]

(* Complex residuals *)

let c128 = Nx.complex128

(* [z² − a] in complex [z], at the principal square root: [∂z/∂a = 1 / 2z],
   holomorphic, so a tangent [v] gives [v / 2z]. *)
let za () =
  Nx.create c128 [| 3 |]
    Complex.
      [| { re = 2.; im = 1. }; { re = -1.; im = 0.5 }; { re = 0.3; im = -2. } |]

let cw () =
  Nx.create c128 [| 3 |]
    Complex.
      [|
        { re = 0.5; im = -1. }; { re = 1.; im = 2. }; { re = -0.7; im = 0.1 };
      |]

(* The default solve, and one for a diagonal [J]: [op] of ones is its
   diagonal. *)
let diagonal op b = Nx.div b (op (Nx.ones_like b))

let csqrt ?linear_solve a =
  Rune.root ?linear_solve one
    ~residual:(fun z -> Nx.sub (Nx.mul z z) a)
    (fun () -> Nx.sqrt a)

let closs ?linear_solve a =
  Nx.sum (Nx.real f64 (Nx.mul (csqrt ?linear_solve a) (cw ())))

let closs_closed a = Nx.sum (Nx.real f64 (Nx.mul (Nx.sqrt a) (cw ())))

(* [cdiff f a v] is the central difference of the real [f] at [a] along [v]. *)
let cdiff f a v =
  let eps = 1e-6 in
  let at s =
    Nx.item [] (f (Nx.add a (Nx.mul_s v { Complex.re = s; im = 0. })))
  in
  (at eps -. at (-.eps)) /. (2. *. eps)

let complex_tests =
  List.concat_map
    (fun (name, linear_solve) ->
      [
        test ("jvp of a complex root is v / 2z, " ^ name) (fun () ->
            let a = za () and v = cw () in
            equal (close ())
              (Nx.div v (Nx.mul_s (Nx.sqrt a) { Complex.re = 2.; im = 0. }))
              (snd (Rune.jvp' (csqrt ?linear_solve) a v)));
        test ("grad of a complex root is the directional derivative's, " ^ name)
          (fun () ->
            let a = za () in
            let g = Rune.grad' (closs ?linear_solve) a in
            List.iter
              (fun (msg, v) ->
                equal ~msg (close ())
                  (scalar (cdiff closs_closed a v))
                  (scalar (Oracle.dot g v)))
              [
                ("real direction", Nx.ones c128 [| 3 |]);
                ( "imaginary direction",
                  Nx.full c128 [| 3 |] { Complex.re = 0.; im = 1. } );
                ("mixed direction", cw ());
              ]);
        test ("vmap of jvp of a complex root, " ^ name) (fun () ->
            let a = Nx.reshape [| 3; 1 |] (za ())
            and v = Nx.reshape [| 3; 1 |] (cw ()) in
            equal (close ())
              (Nx.div v (Nx.mul_s (Nx.sqrt a) { Complex.re = 2.; im = 0. }))
              (Rune.vmap
                 Nx.Ptree.(tensor @-> tensor @-> returns tensor)
                 (fun a v -> snd (Rune.jvp' (csqrt ?linear_solve) a v))
                 a v));
      ])
    [ ("default solve", None); ("given solve", Some diagonal) ]

(* Roots in loops *)

(* An implicit step of [x' = −w x]: the [y] with [y + h w y = x], found by the
   fixed-point iteration [y ← x − h w y], which contracts by [h w]. *)
let h = 0.5

let implicit w x =
  let r y = Nx.sub (Nx.add y (Nx.mul_s (Nx.mul w y) h)) x in
  Rune.root one ~residual:r (fun () ->
      Rune.iterate' ~max:200
        ~until:(fun y -> Nx.less_s (Nx.max (Nx.abs (r y))) 1e-13)
        ~f:(fun y -> Nx.sub x (Nx.mul_s (Nx.mul w y) h))
        x)

let explicit w x = Nx.div x (Nx.add_s (Nx.mul_s w h) 1.)
let settled x = Nx.less_s (Nx.max (Nx.abs x)) 0.05

(* The implicit march until the state is small, and the same march with the
   step's closed form, written out as an OCaml loop. *)
let march w x = Rune.iterate' ~max:80 ~until:settled ~f:(implicit w) x

let written_march w x =
  let rec go k x =
    if Nx.item [] (settled x) then x
    else if k = 80 then invalid_arg "written_march: too many steps"
    else go (k + 1) (explicit w x)
  in
  go 0 x

(* A root in a scan's step, whose solve is itself a scan of Newton steps. *)
let scan_newton a =
  fst
    (Rune.scan'
       ~f:(fun x _ -> (Nx.mul_s (Nx.add x (Nx.div a x)) 0.5, x))
       ~init:(Nx.add_s a 1.) (Nx.zeros f64 [| 60 |]))

let scan_rows = vec [| 0.5; -1.2; 0.8 |]

let root_in_scan root a =
  let step c r =
    let c = root (Nx.add c (Nx.mul r r)) in
    (c, c)
  in
  let c, ys = Rune.scan' ~f:step ~init:a scan_rows in
  Nx.add (Nx.sum c) (Nx.sum (Nx.sin ys))

let scanned_root a =
  Rune.root one
    ~residual:(fun x -> Nx.sub (Nx.mul x x) a)
    (fun () -> scan_newton a)

let loop_tests =
  let w = scalar 0.7 in
  let xs = vec [| 1.9; -0.3; 0.02; 1.2; -1.7 |] in
  let loss march w x = Nx.sum (Nx.sin (march w x)) in
  [
    test "an implicit march's lanes stop apart" (fun () ->
        let trips x =
          let rec go k x =
            if Nx.item [] (settled x) then k else go (k + 1) (explicit w x)
          in
          go 0 x
        in
        let ks = List.init 5 (fun i -> trips (lane i xs)) in
        equal int 0 (List.fold_left min max_int ks);
        at_least int ~than:3 (List.length (List.sort_uniq compare ks)));
    test "vmap of a root in an iterate's step" (fun () ->
        equal (close ())
          (per_lane (written_march w) xs)
          (Rune.vmap' (march w) xs));
    test "vmap of grad of a root in an iterate's step" (fun () ->
        equal (close ())
          (per_lane (Rune.grad' (loss written_march w)) xs)
          (Rune.vmap' (Rune.grad' (loss march w)) xs));
    test "grad of a captured parameter of a root in an iterate's step"
      (fun () ->
        equal (close ())
          (Rune.grad'
             (fun w ->
               Nx.sum (stack 5 (fun i -> loss written_march w (lane i xs))))
             w)
          (Rune.grad' (fun w -> Nx.sum (Rune.vmap' (loss march w) xs)) w));
    test "jvp of vmap of a root in an iterate's step" (fun () ->
        let vs = Nx.cos xs in
        equal (close ())
          (Nx.mul (per_lane (Rune.grad' (loss written_march w)) xs) vs)
          (snd (Rune.jvp' (Rune.vmap' (loss march w)) xs vs)));
    test "grad of a root in a scan's step whose solve is a scan" (fun () ->
        let a = scalar 1.3 in
        equal (close ())
          (Rune.grad' (root_in_scan Nx.sqrt) a)
          (Rune.grad' (root_in_scan scanned_root) a));
    test "vmap of grad of a root in a scan's step" (fun () ->
        let a = vec [| 1.3; 0.4; 2.2 |] in
        equal (close ())
          (per_lane (Rune.grad' (root_in_scan Nx.sqrt)) a)
          (Rune.vmap' (Rune.grad' (root_in_scan scanned_root)) a));
    test "grad of vmap of a root in a scan's step" (fun () ->
        let a = vec [| 1.3; 0.4; 2.2 |] in
        equal (close ())
          (per_lane (Rune.grad' (root_in_scan Nx.sqrt)) a)
          (Rune.grad'
             (fun a -> Nx.sum (Rune.vmap' (root_in_scan scanned_root) a))
             a));
    test "a solve that scans two steps is not differentiated" (fun () ->
        (* Two Newton steps from [a + 1]: the derivative is the theorem's at
           their result [x̂], [1 / 2x̂]. *)
        let a = scalar 2. in
        let two a =
          fst
            (Rune.scan'
               ~f:(fun x _ -> (Nx.mul_s (Nx.add x (Nx.div a x)) 0.5, x))
               ~init:(Nx.add_s a 1.) (Nx.zeros f64 [| 2 |]))
        in
        let r a =
          Rune.root one
            ~residual:(fun x -> Nx.sub (Nx.mul x x) a)
            (fun () -> two a)
        in
        equal (close ()) (Nx.recip (Nx.mul_s (two a) 2.)) (Rune.grad' r a);
        equal (close ())
          (Nx.recip (Nx.mul_s (two a) 2.))
          (Rune.vmap' (Rune.grad' r) (vec [| 2. |]) |> Nx.reshape [||]));
  ]

(* Compilation *)

let pair_to_pair =
  Nx.Ptree.(pair tensor tensor @-> returns (pair tensor tensor))

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
    test "jit of jvp is eager jvp" (fun () ->
        let root a =
          Rune.root one
            ~residual:(fun x -> Nx.sub (Nx.mul x x) a)
            (fun () -> Nx.sqrt a)
        in
        let f a = snd (Rune.jvp' root a (Nx.cos a)) in
        let a = vec [| 0.5; 2.; 3. |] in
        equal (close ()) (f a) (Rune.jit' f a));
    test "jit of grad of a mapped linear solve is eager" (fun () ->
        let ths = vec [| 0.3; -1.1; 0.7 |] in
        let f ths =
          Nx.sum (Rune.vmap' (system_loss ~linear_solve:mapped_dense) ths)
        in
        equal (close ()) (Rune.grad' f ths) (Rune.jit' (Rune.grad' f) ths));
    test "jit of a root whose solve iterates is eager's value" (fun () ->
        let a = vec [| 0.5; 2.; 3. |] in
        equal (close ()) (sqrt_root a) (Rune.jit' sqrt_root a));
    prop ~count:10
      "jit of jvp of a root whose solve iterates is the finite difference of \
       the closed-form root"
      positive (fun a ->
        let v = direction a in
        equal (close ())
          (Oracle.central ~eps:1e-6 closed a v)
          (Rune.jit
             Nx.Ptree.(tensor @-> tensor @-> returns tensor)
             (fun a v -> snd (Rune.jvp' loss a v))
             a v));
    prop ~count:10
      "jit of grad of a root whose solve iterates is the finite difference of \
       the closed-form root"
      positive (fun a ->
        let v = direction a in
        equal (close ())
          (Oracle.central ~eps:1e-6 closed a v)
          (scalar (Oracle.dot (Rune.jit' (Rune.grad' loss) a) v)));
    test "jit of vmap of a root whose solve iterates solves each lane"
      (fun () ->
        let a = vec [| 0.5; 2.; 3. |] in
        equal (close ()) (Nx.sqrt a) (Rune.jit' (Rune.vmap' sqrt_root) a));
    test "jit of grad of a linear solve on iterate is the closed form's"
      (fun () ->
        let f = solved_loss ~linear_solve:cg in
        let ga, gb =
          Rune.jit pair_to_pair (Rune.grad pair f) (spd (), rhs ())
        in
        let ga', gb' = Rune.grad pair closed_solve (spd (), rhs ()) in
        equal ~msg:"a" (close ()) ga' ga;
        equal ~msg:"b" (close ()) gb' gb);
  ]

let () =
  exit
    (run "Rune.root"
       [
         group "derivatives" (derivative_tests @ second_order_tests);
         group "where the derivative is stated" stated_tests;
         group "complex residuals" complex_tests;
         group "roots in loops" loop_tests;
         group "reading" reading_tests;
         group "structures" structure_tests;
         group "mapped linear solves"
           (mapped_props @ second_order_props @ mapped_tests);
         group "the operator's contract" contract_tests;
         group "refusals" refusal_tests;
         group "totals" total_tests;
         group "compiled" compiled_tests;
       ])
