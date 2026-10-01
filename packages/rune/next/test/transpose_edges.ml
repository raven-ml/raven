(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The pullbacks' edges: what a duplicated, shadowed, padded or unread element
   receives, the conjugate transpose of a triangular solve, where the two modes
   part at a non-finite coefficient, and the affine tangent maps the recorder
   refuses or reads as linear. *)

open Windtrap
module Rune = Rune_next.Rune
module Op = Nx.Op

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let floats = array float_exact
let gradient f x = Nx.to_array (Rune.grad' (fun x -> Nx.sum (f x)) x)

let indices a =
  Nx.create Nx.int64 [| Array.length a |] (Array.map Int64.of_int a)

let indexed =
  [
    test "under Set a shadowed update receives nothing" (fun () ->
        let scatter u =
          Op.eval
            (Scatter
               {
                 mode = `Set;
                 unique = false;
                 axis = 0;
                 indices = indices [| 1; 1 |];
                 updates = u;
                 into = Nx.zeros f64 [| 3 |];
               })
        in
        equal floats [| 0.; 1. |] (gradient scatter (vec [| 5.; 6. |])));
    test "under Set the overwritten target receives nothing" (fun () ->
        let scatter into =
          Op.eval
            (Scatter
               {
                 mode = `Set;
                 unique = false;
                 axis = 0;
                 indices = indices [| 1; 1 |];
                 updates = vec [| 5.; 6. |];
                 into;
               })
        in
        equal floats [| 1.; 0.; 1. |] (gradient scatter (vec [| 1.; 2.; 3. |])));
    test "under Add every duplicate update receives the cotangent" (fun () ->
        let scatter u =
          Op.eval
            (Scatter
               {
                 mode = `Add;
                 unique = false;
                 axis = 0;
                 indices = indices [| 1; 1 |];
                 updates = u;
                 into = Nx.zeros f64 [| 3 |];
               })
        in
        equal floats [| 1.; 1. |] (gradient scatter (vec [| 5.; 6. |])));
    test
      "a gathered element receives the sum of its reads, an index outside the \
       axis nothing" (fun () ->
        let gather x = Op.eval (Gather (0, indices [| 2; 0; 2; -1; 3 |], x)) in
        equal floats [| 1.; 0.; 2. |] (gradient gather (vec [| 1.; 2.; 3. |])));
    test "a pad's fill receives nothing" (fun () ->
        equal floats [| 1.; 1. |]
          (gradient
             (fun x -> Op.eval (Pad ([| (2, 1) |], 5., x)))
             (vec [| 1.; 2. |])));
    test "under unit_diag the diagonal receives nothing" (fun () ->
        let solve a =
          Op.eval
            (Solve_triangular
               {
                 upper = false;
                 transpose = false;
                 unit_diag = true;
                 a;
                 b = vec [| 1.; -2. |];
               })
        in
        let g =
          Rune.grad'
            (fun a -> Nx.sum (solve a))
            (Nx.create f64 [| 2; 2 |] [| 9.; 9.; 0.5; 9. |])
        in
        equal floats [| 0.; 0. |] (Nx.to_array (Nx.diagonal g)));
    test "an element no window reads receives zero" (fun () ->
        let windows x =
          Op.eval (Move (x, Window { axis = 0; size = 2; step = 3 }))
        in
        equal floats [| 1.; 1.; 0.; 1.; 1. |]
          (gradient windows (vec [| 1.; 2.; 3.; 4.; 5. |])));
    test
      "under Max the cotangent goes to the operand whose bits the result \
       holds, the element first at a tie" (fun () ->
        let scatter u =
          Op.eval
            (Scatter
               {
                 mode = `Max;
                 unique = false;
                 axis = 0;
                 indices = indices [| 1; 1; 0; 2 |];
                 updates = Nx.slice [ R (3, 7) ] u;
                 into = Nx.slice [ R (0, 3) ] u;
               })
        in
        equal floats
          [| 1.; 0.; 1.; 0.; 1.; 0.; 0. |]
          (gradient scatter (vec [| 0.; 5.; 4.; 7.; 9.; -1.; 4. |])));
  ]

let recordings =
  [
    test "a cotangent laid out unlike its operand reshapes as the operand does"
      (fun () ->
        (* The pullback of a transpose is a transposed view. *)
        let f x =
          Op.eval
            (Move (Op.eval (Move (x, Reshape [| 3; 2 |])), Permute [| 1; 0 |]))
        in
        let w = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let _, pullback = Rune.vjp' f (vec [| 0.; 0.; 0.; 0.; 0.; 0. |]) in
        equal floats [| 1.; 4.; 2.; 5.; 3.; 6. |] (Nx.to_array (pullback w)));
    cases
      ~name:(fun n -> Printf.sprintf "%d operations" n)
      "a recording of any length transposes"
      [ 61; 62; 63; 64; 65; 127; 128; 129 ]
      (fun n ->
        let rec negate k x =
          if k = 0 then x else negate (k - 1) (Op.eval (Unary (Neg, x)))
        in
        let expected = if n mod 2 = 0 then 1. else -1. in
        equal floats [| expected; expected |]
          (gradient (negate n) (vec [| 1.; 2. |])));
  ]

(* A triangular solve's pullback in its right-hand side is the solve by the
   conjugate transpose: [b ↦ A⁻¹ b] has the adjoint [A⁻ᴴ] under [Re ⟨u, v⟩]. *)
let solve_adjoint =
  let a =
    Nx.create Nx.complex128 [| 3; 3 |]
      (Array.map
         (fun (re, im) -> { Complex.re; im })
         [|
           (2.1, 0.5);
           (-0.7, 1.3);
           (0.4, -0.9);
           (0.8, 0.2);
           (1.9, -0.6);
           (0.3, 0.7);
           (-0.5, 0.4);
           (0.6, -0.3);
           (2.4, 0.8);
         |])
  in
  let b = Nx.cast Nx.complex128 (vec [| 1.; -2.; 0.5 |]) in
  let flags =
    List.concat_map
      (fun upper ->
        List.concat_map
          (fun transpose ->
            List.map
              (fun unit_diag -> (upper, transpose, unit_diag))
              [ false; true ])
          [ false; true ])
      [ false; true ]
  in
  cases
    ~name:(fun (u, t, d) ->
      Printf.sprintf "%s%s%s"
        (if u then "upper" else "lower")
        (if t then ", transposed" else "")
        (if d then ", unit diagonal" else ""))
    "a solve's pullback in b is the solve by the conjugate transpose" flags
    (fun (upper, transpose, unit_diag) ->
      let solve transpose b =
        Op.eval (Solve_triangular { upper; transpose; unit_diag; a; b })
      in
      let w =
        Nx.create Nx.complex128 [| 3 |]
          Complex.[| { re = 0.3; im = -1.1 }; one; { re = -0.7; im = 0.9 } |]
      in
      let _, pullback = Rune.vjp' (solve transpose) b in
      equal
        (Reference.close ~rel:1e-12 ())
        [ Reference.complexes (solve (not transpose) w) ]
        [ Reference.complexes (pullback w) ])

(* [where (x > 0) (sqrt x) 0] at [0]: forward mode selects the zero branch's
   tangent, and reverse mode multiplies the unselected branch's zero cotangent
   by [sqrt]'s infinite coefficient. *)
let where_at_an_infinite_coefficient =
  test "where at 0 over sqrt: jvp gives 0 and grad gives NaN" (fun () ->
      let f x =
        Op.eval
          (Where
             ( Op.eval (Compare (Less, Nx.zeros_like x, x)),
               Op.eval (Unary (Sqrt, x)),
               Nx.zeros_like x ))
      in
      let x = vec [| 0. |] in
      equal ~msg:"jvp" floats [| 0. |]
        (Nx.to_array (snd (Rune.jvp' f x (vec [| 1. |]))));
      equal ~msg:"grad" floats [| Float.nan |] (gradient f x))

(* The recorder reads a tangent map built from the rows. *)

let affine name f x =
  test name (fun () ->
      raises
        (Invalid_argument
           (Printf.sprintf
              "Rune.grad': a custom_jvp tangent map applies %s to a tangent; a \
               tangent map must be linear in its tangents"
              name))
        (fun () ->
          Rune.grad'
            (fun x ->
              Nx.sum
                (Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor
                   (fun x -> (x, f))
                   x))
            x))

(* A plain operand beside a tangent of [where], [cat], [scatter] or [update] is
   read as zero: the gradient of the map with it equals the gradient of the map
   with zeros in its place. *)
let taken_as_zero name with_constant with_zeros x =
  test name (fun () ->
      let through map x =
        Rune.grad'
          (fun x ->
            Nx.sum
              (Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor
                 (fun x -> (x, map))
                 x))
          x
      in
      equal floats
        (Nx.to_array (through with_zeros x))
        (Nx.to_array (through with_constant x)))

let recorder =
  let c = vec [| 7.; 8. |] and z = Nx.zeros f64 [| 2 |] in
  let x = vec [| 1.; 2. |] in
  let mask = Nx.create Nx.bool [| 2 |] [| true; false |] in
  let at = indices [| 1; 0 |] in
  let through map x =
    Rune.grad'
      (fun x ->
        Nx.sum
          (Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor
             (fun x -> (x, map))
             x))
      x
  in
  [
    test "a detached tangent in a tangent map is the tangent" (fun () ->
        equal floats [| 1.; 1. |] (Nx.to_array (through Rune.detach x)));
    test "a loop in a tangent map is written out and transposed" (fun () ->
        (* The carry starts as a tangent: a plain zero added to one is
           affine. *)
        let cumsum dx =
          snd
            (Rune.scan'
               ~f:(fun c r -> (Nx.add c r, Nx.add c r))
               ~init:(Nx.mul_s (Nx.get [ 0 ] dx) 0.)
               dx)
        in
        equal floats [| 3.; 2.; 1. |]
          (Nx.to_array (through cumsum (vec [| 1.; 2.; 3. |]))));
    affine "add" (fun dx -> Op.eval (Binary (Add, dx, c))) x;
    affine "sub" (fun dx -> Op.eval (Binary (Sub, c, dx))) x;
    affine "pad"
      (fun dx ->
        Op.eval
          (Move (Op.eval (Pad ([| (1, 1) |], 5., dx)), Shrink [| (1, 3) |])))
      x;
    affine "cast"
      (fun dx ->
        Op.eval (Convert (Cast, f64, Op.eval (Convert (Cast, Nx.int32, dx)))))
      x;
    test "a pad with a zero fill is linear" (fun () ->
        let map dx =
          Op.eval
            (Move (Op.eval (Pad ([| (1, 1) |], 0., dx)), Shrink [| (1, 3) |]))
        in
        equal floats [| 1.; 1. |]
          (Nx.to_array
             (Rune.grad'
                (fun x ->
                  Nx.sum
                    (Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor
                       (fun x -> (x, map))
                       x))
                x)));
    taken_as_zero "a constant branch of where is read as zero"
      (fun dx -> Op.eval (Where (mask, dx, c)))
      (fun dx -> Op.eval (Where (mask, dx, z)))
      x;
    taken_as_zero "a constant piece of cat is read as zero"
      (fun dx ->
        Op.eval (Move (Op.eval (Cat (0, [ dx; c ])), Shrink [| (1, 3) |])))
      (fun dx ->
        Op.eval (Move (Op.eval (Cat (0, [ dx; z ])), Shrink [| (1, 3) |])))
      x;
    taken_as_zero "a constant target of scatter is read as zero"
      (fun dx ->
        Op.eval
          (Scatter
             {
               mode = `Add;
               unique = false;
               axis = 0;
               indices = at;
               updates = dx;
               into = c;
             }))
      (fun dx ->
        Op.eval
          (Scatter
             {
               mode = `Add;
               unique = false;
               axis = 0;
               indices = at;
               updates = dx;
               into = z;
             }))
      x;
    taken_as_zero "a constant target of update is read as zero"
      (fun dx ->
        Op.eval
          (Move
             ( Op.eval (Update (vec [| 9.; 9.; 9. |], indices [| 1 |], dx)),
               Shrink [| (1, 3) |] )))
      (fun dx ->
        Op.eval
          (Move
             ( Op.eval (Update (Nx.zeros f64 [| 3 |], indices [| 1 |], dx)),
               Shrink [| (1, 3) |] )))
      x;
  ]

(* [lanes a] inside the map named [a]: a linear call whose transpose sums the
   lanes' cotangents and gives each lane its row. *)
(* A draw inside a differentiated function is a constant of it: the same
   values flow forward and back, and nothing flows into the key. *)
let draws =
  [
    test "a drawn mask is a constant of the differentiation" (fun () ->
        let key = Nx.Rng.key 7 in
        let mask = ref None in
        let g =
          Rune.grad'
            (fun x ->
              let m =
                Nx.cast f64 (Nx.Rng.bernoulli key (Nx.full f64 [| 8 |] 0.5))
              in
              mask := Some m;
              Nx.sum (Nx.mul x m))
            (vec (Array.init 8 float_of_int))
        in
        equal (Reference.exact ()) (Option.get !mask) g);
  ]

let lanes =
  let n = 3 and k = 2 in
  let xs = Nx.create f64 [| n; k |] [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |] in
  let ws =
    Nx.create f64 [| n; n; k |]
      (Array.init (n * n * k) (fun i -> float_of_int ((i * 5 mod 7) - 3)))
  in
  [
    test "a lane's gradient is its row of every lane's cotangent" (fun () ->
        let a = Rune.axis () in
        let grads =
          Rune.vmap ~axis:a
            Nx.Ptree.(tensor @-> tensor @-> returns tensor)
            (fun x w ->
              Rune.grad' (fun x -> Nx.sum (Nx.mul (Rune.lanes a x) w)) x)
            xs ws
        in
        (* Lane j's objective reads lane i's x through w_j[i]. *)
        let expected = Nx.sum ~axes:[ 0 ] ws in
        equal floats (Nx.to_array expected) (Nx.to_array grads));
    test
      "a lane's gradient through shared weights is the lane count times its row"
      (fun () ->
        (* Four lanes of matrices, one captured weight: every lane's objective
           reads lane i's x through w[i], so lane i receives 4 w[i]. *)
        let a = Rune.axis () in
        let xs =
          Nx.create f64 [| 4; 2; 2 |]
            (Array.init 16 (fun i -> Float.sin (float_of_int i)))
        in
        let w =
          Nx.create f64 [| 4; 2; 2 |]
            (Array.init 16 (fun i -> float_of_int ((i * 3 mod 5) - 2)))
        in
        let grads =
          Rune.vmap' ~axis:a
            (fun x ->
              Rune.grad' (fun x -> Nx.sum (Nx.mul (Rune.lanes a x) w)) x)
            xs
        in
        equal floats (Nx.to_array (Nx.mul_s w 4.)) (Nx.to_array grads));
    test "a map of lanes has its tangent's adjoint for a pullback" (fun () ->
        let a = Rune.axis () in
        let g xs =
          Rune.vmap' ~axis:a (fun x -> Nx.mul_s (Rune.lanes a (Nx.sin x)) 2.) xs
        in
        let r = Random.State.make [| 3 |] in
        let v = Reference.direction r Nx.Ptree.tensor xs in
        let y, jv = Rune.jvp' g xs v in
        let w = Reference.direction r Nx.Ptree.tensor y in
        let _, pullback = Rune.vjp' g xs in
        let t = Nx.Ptree.tensor in
        let l = Reference.dot t w jv and pb = pullback w in
        let rhs = Reference.dot t pb v in
        let bound =
          1e-12 *. (Reference.magnitude t w jv +. Reference.magnitude t pb v)
        in
        equal (float bound) l rhs);
  ]

let tests =
  [
    group "indexed" indexed;
    group "recordings" recordings;
    solve_adjoint;
    where_at_an_infinite_coefficient;
    group "the recorder" recorder;
    group "draws" draws;
    group "lanes" lanes;
  ]
