(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Maps over the operands a row's draws never batch: conditions, indices, starts
   and keys; a read inside a map; and nx's functions made of several rows, each
   against its loop. *)

open Windtrap
module Rune = Rune_next.Rune
module Op = Nx.Op

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let mat r c a = Nx.create f64 [| r; c |] a
let int32s shape a = Nx.create Nx.int32 shape (Array.map Int32.of_int a)

(* [loop f xs] is [f] of each row of [xs], stacked. *)
let loop f xs =
  Nx.stack (List.init (Nx.dim 0 xs) (fun k -> f (Nx.get [ k ] xs)))

let is_the_loop name f xs =
  test name (fun () -> equal (Reference.exact ()) (loop f xs) (Rune.vmap' f xs))

let close_to_the_loop name f xs =
  test name (fun () ->
      equal
        (Reference.close ~rel:1e-12 ~floor:1e-14 ())
        [ Reference.complexes (loop f xs) ]
        [ Reference.complexes (Rune.vmap' f xs) ])

let data = vec [| 0.5; -1.2; 2.1; 1.7 |]

let integer_operands =
  [
    is_the_loop "a batched condition alone selects per lane"
      (fun c -> Op.eval (Where (c, data, Nx.neg data)))
      (Nx.create Nx.bool [| 3; 4 |]
         [|
           true;
           false;
           true;
           false;
           false;
           false;
           true;
           true;
           true;
           true;
           false;
           false;
         |]);
    is_the_loop "batched indices alone gather per lane"
      (fun i -> Op.eval (Gather (0, i, data)))
      (int32s [| 3; 2 |] [| 0; 3; 2; 2; -1; 1 |]);
    is_the_loop "batched indices alone scatter per lane"
      (fun i ->
        Op.eval
          (Scatter
             {
               mode = `Add;
               unique = false;
               axis = 0;
               indices = i;
               updates = vec [| 1.; 10. |];
               into = data;
             }))
      (int32s [| 3; 2 |] [| 0; 3; 2; 2; 1; 0 |]);
    is_the_loop "batched starts alone write each lane's window"
      (fun s -> Op.eval (Update (data, s, vec [| 7.; 8. |])))
      (int32s [| 4; 1 |] [| 0; 1; 2; 1 |]);
    is_the_loop "batched starts alone write each lane's window of a matrix"
      (fun s ->
        Op.eval
          (Update
             ( mat 3 3 (Array.init 9 float_of_int),
               s,
               mat 2 2 [| -1.; -2.; -3.; -4. |] )))
      (int32s [| 4; 2 |] [| 0; 0; 1; 0; 0; 1; 1; 1 |]);
    test "batched starts and values write each lane's window" (fun () ->
        let starts = int32s [| 3; 1 |] [| 2; 0; 1 |] in
        let values = mat 3 2 [| 7.; 8.; 9.; 10.; 11.; 12. |] in
        let write s v = Op.eval (Update (data, s, v)) in
        let expected =
          Nx.stack
            (List.init 3 (fun k ->
                 write (Nx.get [ k ] starts) (Nx.get [ k ] values)))
        in
        equal (Reference.exact ()) expected
          (Rune.vmap
             Nx.Ptree.(tensor @-> tensor @-> returns tensor)
             write starts values));
    is_the_loop "a batched key draws each lane's words"
      (fun key -> Op.eval (Threefry (key, int32s [| 2 |] [| 5; 6 |])))
      (int32s [| 3; 2 |] [| 1; 2; 3; 4; 5; 6 |]);
  ]

let reads =
  [
    test "reading a lane raises" (fun () ->
        raises
          (Invalid_argument
             "Rune: cannot read the value of a batched tensor inside vmap; \
              return it from the mapped function instead") (fun () ->
            Rune.vmap' (fun x -> Nx.mul_s x (Nx.item [] x)) data));
    test "reading a constant inside a map computes" (fun () ->
        let c = Nx.scalar f64 3. in
        equal (Reference.exact ()) (Nx.mul_s data 3.)
          (Rune.vmap' (fun x -> Nx.mul_s x (Nx.item [] c)) data));
  ]

let xs () = mat 3 3 [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2; 1.3; -0.7 |]

let compositions =
  [
    is_the_loop "an elementwise chain"
      (fun r -> Nx.add (Nx.exp (Nx.sin r)) (Nx.mul r r))
      (xs ());
    is_the_loop "captured constants broadcast"
      (fun r -> Nx.mul r (vec [| 2.; -1.; 0.5 |]))
      (xs ());
    is_the_loop "a scalar captured constant" (fun r -> Nx.add_s r 3.) (xs ());
    is_the_loop "a constant result is broadcast"
      (fun _ -> Nx.scalar f64 7.)
      (xs ());
    close_to_the_loop
      "a product with a captured constant of batch axes of its own"
      (fun r ->
        Nx.matmul r
          (Nx.create f64 [| 2; 3; 2 |]
             (Array.init 12 (fun i -> float_of_int (i - 5) /. 4.))))
      (xs ());
    test "a product of two batched operands with batch axes of their own"
      (fun () ->
        let a =
          Nx.create f64 [| 2; 2; 2; 3 |]
            (Array.init 24 (fun i -> Float.sin (float_of_int i)))
        in
        let b =
          Nx.create f64 [| 2; 3; 2 |]
            (Array.init 12 (fun i -> Float.cos (float_of_int i)))
        in
        let expected =
          Nx.stack
            (List.init 2 (fun k -> Nx.matmul (Nx.get [ k ] a) (Nx.get [ k ] b)))
        in
        equal
          (Reference.close ~rel:1e-12 ~floor:1e-14 ())
          [ Reference.complexes expected ]
          [
            Reference.complexes
              (Rune.vmap
                 Nx.Ptree.(tensor @-> tensor @-> returns tensor)
                 Nx.matmul a b);
          ]);
    close_to_the_loop "softmax"
      (fun r ->
        let e = Nx.exp r in
        Nx.div e (Nx.sum e ~keepdims:true))
      (xs ());
    close_to_the_loop "centering uses each lane's mean"
      (fun r -> Nx.sub r (Nx.mean r ~keepdims:true))
      (xs ());
    close_to_the_loop "per-sample gradients of a gather"
      (Rune.grad' (fun x ->
           Nx.sum
             (Nx.take ~axis:0
                ~indices:(int32s [| 3 |] [| 2; 0; 2 |])
                (Nx.mul x x))))
      (xs ());
    close_to_the_loop "per-sample gradients of a sliding window"
      (Rune.grad' (fun x ->
           let w = Nx.sliding_window ~window:2 ~step:1 x in
           Nx.sum (Nx.mul w w)))
      (xs ());
    close_to_the_loop "per-sample gradients of a spectral round trip"
      (Rune.grad' (fun x ->
           let y = Nx.irfft f64 ~n:5 (Nx.rfft Nx.complex128 x) in
           Nx.sum (Nx.mul y y)))
      (mat 3 5
         (Array.init 15 (fun i -> float_of_int ((i * 7 mod 13) - 6) /. 4.)));
  ]

let tests =
  [
    group "integer operands" integer_operands;
    group "reads" reads;
    group "compositions" compositions;
  ]
