(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The transformations' laws over generated programs of every differentiable
   family, at arguments of any shape: a gradient is the limit of central
   differences, a pullback is the transpose of a tangent map, a tangent map is
   linear, a mapped function is the function on each row, and a compiled
   function differentiates as the function does. Where an operation's interface
   defines a derivative at a kink, cases state it. *)

open Windtrap

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let exact () = Oracle.tensor ()

(* The sum of the magnitudes of [t]'s elements. *)
let magnitude t =
  Array.fold_left (fun s v -> s +. Float.abs v) 0. (Nx.to_array t)

(* The products a pairing [<a, b>] adds up, by magnitude. *)
let pairing_scale a b = magnitude (Nx.mul a b)

(* Drawn cases *)

type case = {
  p : Formula.t;
  xs : Nx.float64_t list;  (** Arguments and directions, of [p.input]. *)
  ys : Nx.float64_t list;  (** Cotangents, of [p.output]. *)
}

let pp_case ppf c =
  Format.fprintf ppf "@[<v>%a@,%a@]" Formula.pp c.p
    (Format.pp_print_list Nx.pp)
    (c.xs @ c.ys)

(* [drawn ~inputs ~outputs programs] draws a program and [inputs] tensors of its
   argument's shape and [outputs] of its result's. *)
let drawn ?(outputs = 0) ~inputs programs =
  let open Gen in
  let tensors n s = list ~size:(constant n) (Formula.point s) in
  with_pp pp_case
    (let* p = programs in
     let+ xs = tensors inputs p.Formula.input
     and+ ys = tensors outputs p.Formula.output in
     { p; xs; ys })

(* Every family, and the shapes with a zero or a one-element axis, appear in
   some case of each law. *)
let covered (p : Formula.t) =
  let used = Formula.families p in
  List.iter (fun f -> cover f (List.mem f used)) Formula.all_families;
  let shapes = [ p.input; p.output ] in
  cover "an axis of zero elements" (List.exists (Array.mem 0) shapes);
  cover "an axis of one element" (List.exists (Array.mem 1) shapes);
  cover "a scalar argument" (p.input = [||])

(* Central differences *)

(* A central difference [(F (x + h v) - F (x - h v)) / 2h] differs from the
   directional derivative by [h² F''' / 6], its truncation, and by about [ε S /
   h], its rounding, [ε] the float64 epsilon and [S] the magnitude of the terms
   [F] adds up. With [h = 2^-17] both factors are below [6e-11]. The tolerance
   allows third derivatives, and rounding, up to [10^4] times the scale [1 + S +
   |d|], [d] the derivative. *)
let step = Float.ldexp 1. (-17)

let difference_tolerance scale =
  1e4 *. ((step *. step) +. (epsilon_float /. step)) *. scale

let central_difference =
  prop "a gradient paired with v is the central difference along v"
    (drawn ~inputs:2 Formula.smooth) (fun c ->
      covered c.p;
      let x, v = match c.xs with [ x; v ] -> (x, v) | _ -> assert false in
      let f = Formula.objective c.p in
      let d = Oracle.dot (Rune.grad' f x) v in
      let fd = Nx.item [] (Oracle.central ~eps:step f x v) in
      let scale = 1. +. magnitude (Formula.eval c.p x) +. Float.abs d in
      equal (float (difference_tolerance scale)) d fd)

(* Adjoints and linearity *)

(* Two orders of the same float64 sums differ by a few units of [ε] in each term
   they add: the tolerance is [10^3 ε] of the terms' magnitudes. *)
let rounding scale = 1e3 *. epsilon_float *. (1. +. scale)

let adjoint =
  prop "a pullback is the transpose of the tangent map: <u, J v> = <Jᵀ u, v>"
    (drawn ~inputs:2 ~outputs:1 Formula.gen) (fun c ->
      covered c.p;
      let x, v = match c.xs with [ x; v ] -> (x, v) | _ -> assert false in
      let u = List.hd c.ys in
      let f = Formula.eval c.p in
      let _, jv = Rune.jvp' f x v in
      let _, pb = Rune.vjp' f x in
      let jtu = pb u in
      let scale = pairing_scale u jv +. pairing_scale jtu v in
      equal (float (rounding scale)) (Oracle.dot u jv) (Oracle.dot jtu v))

let linear =
  prop "a tangent map is linear in the tangent"
    Gen.(
      pair
        (drawn ~inputs:3 Formula.gen)
        (pair (float_range (-2.) 2.) (float_range (-2.) 2.)))
    (fun (c, (a, b)) ->
      covered c.p;
      let x, v, w =
        match c.xs with [ x; v; w ] -> (x, v, w) | _ -> assert false
      in
      let jvp t = snd (Rune.jvp' (Formula.eval c.p) x t) in
      let combined = jvp (Nx.add (Nx.mul_s v a) (Nx.mul_s w b)) in
      let expected = Nx.add (Nx.mul_s (jvp v) a) (Nx.mul_s (jvp w) b) in
      let scale =
        (Float.abs a *. magnitude (jvp v)) +. (Float.abs b *. magnitude (jvp w))
      in
      equal (Oracle.tensor ~rel:0. ~abs:(rounding scale) ()) expected combined)

(* Maps *)

(* A batch of [n] arguments of a program's argument shape. *)
let batched =
  let open Gen in
  with_pp
    (fun ppf (p, xs) ->
      Format.fprintf ppf "@[<v>%a@,%a@]" Formula.pp p Nx.pp xs)
    (let* p = Formula.gen in
     let* n = int_range 1 3 in
     let+ xs = Formula.points n p.Formula.input in
     (p, xs))

let rows f xs =
  Nx.stack (List.init (Nx.shape xs).(0) (fun i -> f (Nx.slice [ I i ] xs)))

let mapped =
  prop "a mapped program is the program on each row, bit for bit" batched
    (fun (p, xs) ->
      covered p;
      let f = Formula.eval p in
      equal (exact ()) (rows f xs) (Rune.vmap' f xs))

let mapped_gradient =
  prop "a mapped gradient is the gradient at each row, bit for bit" batched
    (fun (p, xs) ->
      covered p;
      let g = Rune.grad' (Formula.objective p) in
      equal (exact ()) (rows g xs) (Rune.vmap' g xs))

(* Compilation *)

(* A compiled function rounds its sums and its transcendental functions in its
   own order, a few units of [ε] in each term: the tolerance is relative,
   [10^-9], and absolute, [10^3 ε] of the gradient's magnitude. *)
let compiled_tolerance g =
  Oracle.tensor ~rel:1e-9 ~abs:(rounding (magnitude g)) ()

let compiled_gradient =
  prop "a gradient through a compiled function is the gradient"
    (drawn ~inputs:1 Formula.gen) (fun c ->
      covered c.p;
      let x = List.hd c.xs in
      let f = Formula.objective c.p in
      let g = Rune.grad' f x in
      equal ~msg:"grad (jit f)" (compiled_tolerance g) g
        (Rune.grad' (Rune.jit' f) x);
      equal ~msg:"jit (grad f)" (compiled_tolerance g) g
        (Rune.jit' (Rune.grad' f) x))

(* Kinks the interfaces define *)

let kinks =
  [
    test "a scatter maximum's tie gives the whole derivative to the element"
      (fun () ->
        (* The element 2. ties the update 2.: the element's bits are the
           result's, so the element takes the derivative and the update none. *)
        let indices = Nx.create Nx.int64 [| 1 |] [| 1L |] in
        let f (t, u) =
          Nx.sum (Nx.scatter ~mode:`Max ~axis:0 ~indices ~values:u t)
        in
        let s = Nx.Ptree.(pair tensor tensor) in
        let gt, gu = Rune.grad s f (vec [| 1.; 2.; 3. |], vec [| 2. |]) in
        equal ~msg:"element" (exact ()) (vec [| 1.; 1.; 1. |]) gt;
        equal ~msg:"update" (exact ()) (vec [| 0. |]) gu);
    test "a scatter maximum's tie among updates gives it to the first update"
      (fun () ->
        let indices = Nx.create Nx.int64 [| 3 |] [| 0L; 0L; 0L |] in
        let f u =
          Nx.sum
            (Nx.scatter ~mode:`Max ~axis:0 ~indices ~values:u (vec [| -1. |]))
        in
        equal (exact ())
          (vec [| 0.; 1.; 0. |])
          (Rune.grad' f (vec [| 0.5; 2.; 2. |])));
    test "a segment maximum's tie gives the whole derivative to its first row"
      (fun () ->
        let ids = Nx.create Nx.int64 [| 4 |] [| 0L; 0L; 1L; 0L |] in
        let f x = Nx.sum (Nx.reduce_segments `Max ~segments:2 ids x) in
        equal (exact ())
          (vec [| 1.; 0.; 1.; 0. |])
          (Rune.grad' f (vec [| 3.; 3.; -1.; 3. |])));
    test "a maximum along an axis shares its derivative among the tied elements"
      (fun () ->
        let g = Rune.grad' (fun x -> Nx.max x) (vec [| 5.; 1.; 5.; 2. |]) in
        let g = Nx.to_array g in
        equal ~msg:"the untied elements" (list float_exact) [ 0.; 0. ]
          [ g.(1); g.(3) ];
        greater ~msg:"the first tied element" float_exact ~than:0. g.(0);
        greater ~msg:"the second tied element" float_exact ~than:0. g.(2);
        equal ~msg:"the shares" (float 1e-15) 1. (g.(0) +. g.(2)));
  ]

let () =
  exit
    (run "Rune laws"
       [
         group "derivatives" [ central_difference; adjoint; linear ];
         group "maps" [ mapped; mapped_gradient ];
         group ~tags:[ "slow" ] "compilation" [ compiled_gradient ];
         group "kinks" kinks;
       ])
