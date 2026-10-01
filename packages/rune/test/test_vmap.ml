(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.vmap as a whole: the structures it maps, what it leaves captured, the
   arguments it refuses, the randomness its lanes draw, and the lanes a map
   names. Each operation's batching is the rule suites'. *)

open Windtrap

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let floats = array float_exact
let values x = Nx.to_array x

let xs () =
  Nx.create f64 [| 4; 3 |]
    [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2; 1.3; -0.7; 0.8; -1.6; 0.4 |]

let ms () =
  Nx.create f64 [| 2; 2; 3 |]
    [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2; 1.3; -0.7; 0.8; -1.6; 0.4 |]

let lane i x = Nx.get [ i ] x
let stack n f = Nx.stack (List.init n f)
let close = array (float_rel ~rel:1e-12 ~abs:1e-14)

(* Structures *)

let structures =
  let a = ms () in
  let b =
    Nx.create f64 [| 2; 3; 2 |]
      [| 1.1; 0.3; -0.8; 0.6; 0.4; -1.5; 0.9; -0.2; 0.7; 1.4; -0.3; 0.5 |]
  in
  let products a b = stack 2 (fun i -> Nx.matmul (lane i a) (lane i b)) in
  [
    test "every leaf of a structure is mapped" (fun () ->
        equal close
          (values (products a b))
          (values
             (Rune.vmap
                Nx.Ptree.(pair tensor tensor @-> returns tensor)
                (fun (a, b) -> Nx.matmul a b)
                (a, b))));
    test "every argument of a curried function is mapped" (fun () ->
        equal close
          (values (products a b))
          (values
             (Rune.vmap
                Nx.Ptree.(tensor @-> tensor @-> returns tensor)
                Nx.matmul a b)));
    test "leaves of different ranks are mapped along their own axis 0"
      (fun () ->
        (* The lanes' sizes coincide, so a misplaced batch axis would pair the
           wrong matrices with no error. *)
        let a =
          Nx.create f64 [| 2; 2; 2; 3 |]
            (Array.init 24 (fun i -> Float.sin (float_of_int i)))
        in
        equal close
          (values (products a b))
          (values
             (Rune.vmap
                Nx.Ptree.(pair tensor tensor @-> returns tensor)
                (fun (a, b) -> Nx.matmul a b)
                (a, b))));
    test "every leaf of a structured result gains the batch axis" (fun () ->
        let c = vec [| 9. |] in
        let y, k =
          Rune.vmap
            Nx.Ptree.(tensor @-> returns (pair tensor tensor))
            (fun x -> (Nx.mul x x, c))
            (xs ())
        in
        equal ~msg:"the mapped leaf" floats
          (values (Nx.mul (xs ()) (xs ())))
          (values y);
        equal ~msg:"the constant leaf's shape" (array int) [| 4; 1 |]
          (Nx.shape k);
        equal ~msg:"the constant leaf" floats [| 9.; 9.; 9.; 9. |] (values k));
  ]

(* Captures *)

let captures =
  [
    test "a captured value is a constant of the map" (fun () ->
        let w = Nx.create f64 [| 3; 2 |] [| 1.1; 0.3; -0.8; 0.6; 0.4; -1.5 |] in
        equal close
          (values (stack 2 (fun i -> Nx.matmul (lane i (ms ())) w)))
          (values (Rune.vmap' (fun m -> Nx.matmul m w) (ms ()))));
    test "a detached captured value is the value" (fun () ->
        let c = vec [| 1.; 2.; 3. |] in
        equal floats
          (values (Nx.mul (xs ()) (Nx.broadcast_to [| 4; 3 |] c)))
          (values (Rune.vmap' (fun x -> Nx.mul x (Rune.detach c)) (xs ()))));
    test "a capture that is also the argument is a constant" (fun () ->
        let w = vec [| 1.; 2.; 3. |] in
        let y = Rune.vmap' (fun x -> Nx.add x w) w in
        equal ~msg:"shape" (array int) [| 3; 3 |] (Nx.shape y);
        equal floats [| 2.; 3.; 4.; 3.; 4.; 5.; 4.; 5.; 6. |] (values y));
  ]

(* Refusals *)

let refusals =
  let pair = Nx.Ptree.(pair tensor tensor @-> returns tensor) in
  let add (a, b) = Nx.add a b in
  [
    test "leaves of two leading lengths are refused, naming both" (fun () ->
        raises (Invalid_argument "Rune.vmap: 0.1: 3 rows along axis 0, 0.0: 2")
          (fun () ->
            Rune.vmap pair add (vec [| 1.; 2. |], vec [| 1.; 2.; 3. |])));
    test "a scalar leaf is refused, naming it" (fun () ->
        starts_with ~affix:"Rune.vmap: 0.1: a scalar"
          (match Rune.vmap pair add (vec [| 1.; 2. |], Nx.scalar f64 1.) with
          | _ -> "no exception"
          | exception Invalid_argument m -> m));
  ]

(* Checks *)

let below_one x =
  Nx.check (Nx.less_s x 1.) (fun i ->
      Printf.sprintf "element %s"
        (String.concat "," (Array.to_list (Array.map string_of_int i))));
  x

let checks =
  [
    test "a check names the first false element of the first lane that has one"
      (fun () ->
        let rows =
          Nx.create f64 [| 3; 3 |] [| 0.; 0.; 0.; 0.; 0.; 5.; 7.; 0.; 0. |]
        in
        raises (Invalid_argument "element 2") (fun () ->
            Rune.vmap' below_one rows));
    test "a check that holds in every lane passes" (fun () ->
        equal floats [| 0.; 0.5 |]
          (values (Rune.vmap' below_one (vec [| 0.; 0.5 |]))));
    test "a check of mapped matrices names a matrix's index" (fun () ->
        let ms =
          Nx.create f64 [| 2; 2; 2 |] [| 0.; 0.; 0.; 0.; 0.; 0.; 3.; 0. |]
        in
        raises (Invalid_argument "element 1,0") (fun () ->
            Rune.vmap' below_one ms));
    test "a check of mapped values compiles" (fun () ->
        let rows = Nx.create f64 [| 2; 2 |] [| 0.; 0.; 0.; 4. |] in
        raises (Invalid_argument "element 1") (fun () ->
            Rune.jit' (Rune.vmap' below_one) rows));
  ]

(* Randomness *)

let randomness =
  [
    test "an implicit draw is a constant of the map: every lane draws the same"
      (fun () ->
        let y =
          Nx.Rng.with_key (Nx.Rng.key 42) (fun () ->
              Rune.vmap' (fun r -> Nx.add r (Nx.rand f64 [| 3 |])) (xs ()))
        in
        let draws = Nx.sub y (xs ()) in
        for i = 1 to 3 do
          equal
            ~msg:(Printf.sprintf "lane %d" i)
            close
            (values (lane 0 draws))
            (values (lane i draws))
        done);
    test "a key folded with the lane index draws per lane" (fun () ->
        let key = Nx.Rng.key 7 in
        let y =
          Rune.vmap'
            (fun r ->
              Nx.add (Nx.mul_s r 0.)
                (Nx.Rng.uniform
                   (Nx.Rng.fold_in_tensor key (Rune.lane_index ()))
                   f64 [| 3 |]))
            (xs ())
        in
        equal floats
          (values
             (stack 4 (fun i ->
                  Nx.Rng.uniform (Nx.Rng.fold_in key i) f64 [| 3 |])))
          (values y));
    test "outside a map the lane index is 0" (fun () ->
        equal (array int32) [| 0l |]
          (Nx.to_array (Nx.reshape [| 1 |] (Rune.lane_index ()))));
    test "a map over a batch of keys draws each key's values" (fun () ->
        let keys = Nx.Rng.split ~n:4 (Nx.Rng.key 42) in
        let y =
          Rune.vmap
            Nx.Ptree.(Nx.Rng.ptree @-> returns tensor)
            (fun key -> Nx.Rng.uniform key f64 [| 8 |])
            (Nx.Rng.split_batch ~n:4 (Nx.Rng.key 42))
        in
        equal floats
          (values (stack 4 (fun i -> Nx.Rng.uniform keys.(i) f64 [| 8 |])))
          (values y));
    test "a scope rooted at a mapped key draws each key's values" (fun () ->
        let keys = Nx.Rng.split ~n:4 (Nx.Rng.key 42) in
        let draw () = Nx.rand f64 [| 8 |] in
        let y =
          Rune.vmap
            Nx.Ptree.(Nx.Rng.ptree @-> returns tensor)
            (fun key -> Nx.Rng.with_key key draw)
            (Nx.Rng.split_batch ~n:4 (Nx.Rng.key 42))
        in
        equal floats
          (values (stack 4 (fun i -> Nx.Rng.with_key keys.(i) draw)))
          (values y));
    test "a named map passes the lane index of the anonymous map around it on"
      (fun () ->
        let key = Nx.Rng.key 7 and a = Rune.axis () in
        let draw () =
          Nx.Rng.uniform
            (Nx.Rng.fold_in_tensor key (Rune.lane_index ()))
            f64 [| 3 |]
        in
        let y =
          Rune.vmap'
            (fun t ->
              Rune.vmap' ~axis:a
                (fun r -> Nx.add (Nx.mul_s (Nx.add r t) 0.) (draw ()))
                (xs ()))
            (Nx.zeros f64 [| 2; 3 |])
        in
        let trial i = Nx.Rng.uniform (Nx.Rng.fold_in key i) f64 [| 3 |] in
        equal floats
          (values (stack 2 (fun i -> stack 4 (fun _ -> trial i))))
          (values y));
  ]

(* Lanes *)

let lanes =
  [
    test "the named map answers with every lane's value" (fun () ->
        let a = Rune.axis () and x = xs () in
        equal ~msg:"x xᵀ" close
          (values (Nx.matmul x (Nx.transpose x)))
          (values
             (Rune.vmap' ~axis:a (fun r -> Nx.matmul (Rune.lanes a r) r) x));
        let y = Rune.vmap' ~axis:a (fun r -> Rune.lanes a r) x in
        equal ~msg:"shape" (array int) [| 4; 4; 3 |] (Nx.shape y);
        equal ~msg:"every lane holds every row" floats
          (values (stack 4 (fun _ -> x)))
          (values y));
    test "a value every lane shares is gathered as its copies" (fun () ->
        let a = Rune.axis () and c = vec [| 1.; -2.; 0.5 |] in
        equal floats
          (values
             (stack 4 (fun i ->
                  Nx.add (Nx.broadcast_to [| 4; 3 |] c) (lane i (xs ())))))
          (values
             (Rune.vmap' ~axis:a (fun r -> Nx.add (Rune.lanes a c) r) (xs ()))));
    test "a map between the gather and its named map keeps its own lanes"
      (fun () ->
        let a = Rune.axis () and b = Rune.axis () in
        let x = Nx.create f64 [| 2; 3; 4 |] (Array.init 24 Float.of_int) in
        let expected =
          stack 2 (fun _ ->
              stack 3 (fun j -> stack 2 (fun i -> lane j (lane i x))))
        in
        let y = Rune.vmap' ~axis:a (Rune.vmap' (fun r -> Rune.lanes a r)) x in
        equal ~msg:"shape" (array int) [| 2; 3; 2; 4 |] (Nx.shape y);
        equal ~msg:"an anonymous map" floats (values expected) (values y);
        equal ~msg:"a map of another name" floats (values expected)
          (values
             (Rune.vmap' ~axis:a
                (Rune.vmap' ~axis:b (fun r -> Rune.lanes a r))
                x));
        equal ~msg:"an operand the inner map does not batch" floats
          (values (stack 2 (fun _ -> stack 3 (fun _ -> x))))
          (values
             (Rune.vmap' ~axis:a
                (fun r ->
                  Rune.vmap' (fun _ -> Rune.lanes a r) (Nx.zeros f64 [| 3; 1 |]))
                x)));
    test "with no map of its name around it a gather is one lane" (fun () ->
        let a = Rune.axis () and b = Rune.axis () and x = xs () in
        let one = Rune.lanes a x in
        equal ~msg:"no map, shape" (array int) [| 1; 4; 3 |] (Nx.shape one);
        equal ~msg:"no map" floats (values x) (values one);
        List.iter
          (fun (name, y) ->
            equal ~msg:(name ^ ", shape") (array int) [| 4; 1; 3 |] (Nx.shape y);
            equal ~msg:name floats (values x) (values y))
          [
            ("an anonymous map", Rune.vmap' (fun r -> Rune.lanes a r) x);
            ( "a map of another name",
              Rune.vmap' ~axis:b (fun r -> Rune.lanes a r) x );
          ]);
    test "a gather's tangent is the gather of its tangent" (fun () ->
        let a = Rune.axis () and x = xs () and dx = Nx.mul_s (xs ()) 0.5 in
        let gathered x =
          Rune.vmap' ~axis:a (fun r -> Nx.sum ~axes:[ 0 ] (Rune.lanes a r)) x
        in
        equal ~msg:"jvp of the map" close
          (values (gathered dx))
          (values (snd (Rune.jvp' gathered x dx)));
        let y, dy = Rune.jvp' (Rune.lanes a) x dx in
        equal ~msg:"no map, primal" floats (values x) (values y);
        equal ~msg:"no map, tangent" floats (values dx) (values dy);
        let c = vec [| 0.3; -0.7; 1.1 |] in
        let inside =
          Rune.vmap' ~axis:a
            (fun d ->
              snd
                (Rune.jvp'
                   (fun c -> Nx.sum ~axes:[ 0 ] (Rune.lanes a (Nx.mul c c)))
                   c d))
            dx
        in
        let total = Nx.mul (Nx.mul_s c 2.) (Nx.sum ~axes:[ 0 ] dx) in
        equal ~msg:"jvp inside the map" close
          (values (stack 4 (fun _ -> total)))
          (values inside));
    test "a lane index names its map through a map of another name" (fun () ->
        let a = Rune.axis () and b = Rune.axis () in
        let index ?axis () = Nx.cast f64 (Rune.lane_index ?axis ()) in
        let grid = Nx.zeros f64 [| 2; 3 |] in
        let rows = stack 2 (fun i -> Nx.full f64 [| 3 |] (Float.of_int i)) in
        equal ~msg:"a named outer map, an anonymous inner one" floats
          (values rows)
          (values
             (Rune.vmap' ~axis:a
                (Rune.vmap' (fun z -> Nx.add z (index ~axis:a ())))
                grid));
        equal ~msg:"an anonymous outer map, a named inner one" floats
          (values rows)
          (values
             (Rune.vmap'
                (Rune.vmap' ~axis:b (fun z -> Nx.add z (index ())))
                grid)));
    test "reverse mode inside a map counts the lanes of the named map around it"
      (fun () ->
        (* Each lane's z enters every lane's Σₗ zₗ yₗ, so its gradient sums the
           four: 4 y. *)
        let a = Rune.axis () and x = xs () in
        equal close
          (values (Nx.mul_s x 4.))
          (values
             (Rune.vmap' ~axis:a
                (Rune.vmap' (fun y ->
                     Rune.grad' (fun z -> Nx.sum (Rune.lanes a (Nx.mul z y))) y))
                x)));
    test "a remat's barrier on values no lane holds passes through the map"
      (fun () ->
        let c = vec [| 0.3; -0.7; 1.1 |] and x = xs () in
        let rematted = Rune.remat Nx.Ptree.(tensor @-> returns tensor) Nx.sin in
        let g () = Rune.grad' (fun w -> Nx.sum (rematted w)) c in
        equal close
          (values (Nx.add x (Nx.broadcast_to [| 4; 3 |] (Nx.cos c))))
          (values (Rune.vmap' (fun r -> Nx.add r (g ())) x)));
    test "outside its named map, reverse mode differentiates through the gather"
      (fun () ->
        (* Σᵢ (Σⱼ xⱼ) · xᵢ = |Σⱼ xⱼ|² has the gradient 2 Σⱼ xⱼ in every row. *)
        let a = Rune.axis () and x = xs () in
        let f x =
          Nx.sum
            (Rune.vmap' ~axis:a
               (fun r ->
                 Nx.sum (Nx.mul (Nx.sum ~axes:[ 0 ] (Rune.lanes a r)) r))
               x)
        in
        equal close
          (values
             (Nx.broadcast_to [| 4; 3 |] (Nx.mul_s (Nx.sum ~axes:[ 0 ] x) 2.)))
          (values (Rune.grad' f x)));
  ]

let () =
  exit
    (run "Rune vmap"
       [
         group "structures" structures;
         group "captures" captures;
         group "refusals" refusals;
         group "checks" checks;
         group "randomness" randomness;
         group "lanes" lanes;
       ])
