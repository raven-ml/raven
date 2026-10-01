(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The running products, maxima and minima at every order, against their
   definitions in OCaml floats: a running extremum is the element it took, the
   first of equal ones, and a running product's sum is multilinear. *)

open Windtrap
module Op = Nx.Op

let vec a = Nx.create Nx.float64 [| Array.length a |] a
let scan k x = Op.eval (Scan (k, 0, x))
let total k x = Nx.sum (scan k x)
let floats = array float_exact

(* [near scale] compares to rounding against [scale], the sum of the absolute
   values of the terms, so a value that cancels is compared at its terms'
   size. *)
let near scale = array (float (Float.max 1e-300 (1e-12 *. scale)))
let pp_floats ppf a = Nx.pp ppf (vec a)

let vector g =
  Gen.with_pp pp_floats
    (let open Gen in
     let* n = int_range 1 7 in
     array ~size:(constant n) g)

(* Running extrema *)

let max_better a b = if Jvp_edges.running_takes_first then a > b else a >= b
let min_better a b = if Jvp_edges.running_takes_first then a < b else a <= b

(* The gradient of the sum of a running extremum: each position counts the
   running extrema it is. *)
let extremum_gradient better xs =
  let g = Array.make (Array.length xs) 0. in
  Array.iter (fun k -> g.(k) <- g.(k) +. 1.) (Reference.running_arg better xs);
  g

let ties = Gen.of_list ~pp:Format.pp_print_float [ -1.; 0.; -0.; 1.; 1.; 2. ]

let extremum_laws (k : Nx_backend.reduce) better =
  let name = match k with Max -> "cummax" | _ -> "cummin" in
  [
    prop
      ~examples:[ [| 3.; 1.; 2. |]; [| 1.; 3.; 3.; 2. |]; [| 3.; 1.; 2.; 1. |] ]
      (name ^ ": the gradient of the sum is its definition's")
      (vector ties)
      (fun xs ->
        cover "a tie"
          (Array.length
             (Array.of_list (List.sort_uniq compare (Array.to_list xs)))
          < Array.length xs);
        equal floats
          (extremum_gradient better xs)
          (Nx.to_array (Rune.grad' (total k) (vec xs))));
    prop
      (name ^ ": the tangent is each running extremum's element's, bit for bit")
      (Gen.pair (vector ties) (Gen.int_range 0 1_000_000))
      (fun (xs, seed) ->
        let r = Random.State.make [| seed |] in
        let v = Array.map (fun _ -> Random.State.float r 4. -. 2.) xs in
        let args = Reference.running_arg better xs in
        equal floats
          (Array.map (fun k -> v.(k)) args)
          (Nx.to_array (snd (Rune.jvp' (scan k) (vec xs) (vec v)))));
    prop (name ^ ": the sum has a zero second derivative, ties included")
      (vector ties) (fun xs ->
        equal floats
          (Array.make (Array.length xs) 0.)
          (Nx.to_array
             (Rune.grad' (fun x -> Nx.sum (Rune.grad' (total k) x)) (vec xs))));
  ]

let extremum_cases =
  [
    test "cummax at [3; 1; 2] has gradient [3; 0; 0]" (fun () ->
        equal floats [| 3.; 0.; 0. |]
          (Nx.to_array (Rune.grad' (total Max) (vec [| 3.; 1.; 2. |]))));
    test "of equal elements a running extremum takes the convention's"
      (fun () ->
        List.iter
          (fun (k, better, xs) ->
            equal
              ~msg:(Format.asprintf "%a" pp_floats xs)
              floats
              (extremum_gradient better xs)
              (Nx.to_array (Rune.grad' (total k) (vec xs))))
          [
            (Nx_backend.Max, max_better, [| 1.; 3.; 3.; 2. |]);
            (Min, min_better, [| 3.; 1.; 2.; 1. |]);
          ];
        if Jvp_edges.running_takes_first then
          equal ~msg:"jvp" floats [| 1.; 1.; 1.; 10. |]
            (Nx.to_array
               (snd
                  (Rune.jvp' (scan Max)
                     (vec [| 3.; 1.; 3.; 4. |])
                     (vec [| 1.; 10.; 100.; 10. |])))));
    test "a NaN running extremum has the NaN element's tangent" (fun () ->
        let v = vec [| 1.; 10.; 100.; 1000. |] in
        List.iter
          (fun k ->
            equal floats [| 1.; 10.; 10.; 10. |]
              (Nx.to_array
                 (snd (Rune.jvp' (scan k) (vec [| 1.; Float.nan; 5.; 0. |]) v))))
          [ Nx_backend.Max; Min ]);
    test "a scan of length 0 or 1" (fun () ->
        equal floats [||]
          (Nx.to_array (snd (Rune.jvp' (scan Max) (vec [||]) (vec [||]))));
        equal floats [| 7. |]
          (Nx.to_array
             (snd (Rune.jvp' (scan Max) (vec [| 2. |]) (vec [| 7. |])))));
  ]

(* Running products *)

let factors =
  Gen.frequency [ (1, Gen.constant 0.); (3, Gen.float_range (-2.) 2.) ]

let indices n = List.init n Fun.id

(* [contract xs dirs] is the derivative of [Σ cumprod] along each of [dirs], one
   per order: [Σ_{i j ...} D_{i j ...} u_i v_j ...]. *)
let contract xs dirs =
  let n = Array.length xs in
  let rec go chosen = function
    | [] -> Reference.cumprod_derivative xs chosen
    | d :: rest ->
        List.fold_left
          (fun acc i -> acc +. (d.(i) *. go (i :: chosen) rest))
          0. (indices n)
  in
  go [] dirs

let magnitude xs dirs =
  contract (Array.map Float.abs xs) (List.map (Array.map Float.abs) dirs)

(* [gradient_of xs dirs] is the gradient of [contract xs dirs] in [x]. *)
let gradient_of xs dirs =
  Array.init (Array.length xs) (fun i ->
      let e =
        Array.init (Array.length xs) (fun j -> if i = j then 1. else 0.)
      in
      contract xs (e :: dirs))

let directions n seed k =
  let r = Random.State.make [| seed |] in
  List.init k (fun _ -> Array.init n (fun _ -> Random.State.float r 4. -. 2.))

let points =
  Gen.with_pp
    (fun ppf (xs, seed) -> Format.fprintf ppf "%a@ seed %d" pp_floats xs seed)
    (Gen.pair (vector factors) (Gen.int_range 0 1_000_000))

let gradient_scale xs dirs =
  Array.fold_left Float.max 0.
    (Array.init (Array.length xs) (fun i ->
         let e =
           Array.init (Array.length xs) (fun j -> if i = j then 1. else 0.)
         in
         magnitude xs (e :: dirs)))

let jvp_along f dir x = snd (Rune.jvp' f x (vec dir))
let sum_with f dir x = Nx.sum (Nx.mul (f x) (vec dir))
let total_prod = total Prod

let product_laws =
  let examples =
    [
      ([| 2.; 3.; 0. |], 0); ([| 0.; 2.; 3. |], 1); ([| 2.; 0.; 3.; 0.; 5. |], 2);
    ]
  in
  [
    prop ~examples "cumprod: the gradient of the sum is its definition's" points
      (fun (xs, _) ->
        cover "a zero" (Array.exists (fun x -> x = 0.) xs);
        cover "two zeros"
          (Array.length
             (Array.of_list (List.filter (fun x -> x = 0.) (Array.to_list xs)))
          >= 2);
        equal
          (near (gradient_scale xs []))
          (gradient_of xs [])
          (Nx.to_array (Rune.grad' total_prod (vec xs))));
    prop ~examples
      "cumprod: every second derivative of the sum is its definition's, in \
       each mode pairing"
      points (fun (xs, seed) ->
        let u, v =
          match directions (Array.length xs) seed 2 with
          | [ u; v ] -> (u, v)
          | _ -> assert false
        in
        let x = vec xs in
        let hu = gradient_of xs [ u ] in
        let close = near (gradient_scale xs [ u ]) in
        equal ~msg:"grad of grad" close hu
          (Nx.to_array (Rune.grad' (sum_with (Rune.grad' total_prod) u) x));
        equal ~msg:"jvp of grad" close hu
          (Nx.to_array (jvp_along (Rune.grad' total_prod) u x));
        equal ~msg:"jvp of jvp"
          (near (magnitude xs [ u; v ]))
          [| contract xs [ u; v ] |]
          (Nx.to_array
             (Nx.reshape [| 1 |] (jvp_along (jvp_along total_prod u) v x)));
        equal ~msg:"grad of jvp" close hu
          (Nx.to_array (Rune.grad' (jvp_along total_prod u) x)));
    prop ~examples
      "cumprod: every third derivative of the sum is its definition's" points
      (fun (xs, seed) ->
        let u, v, t =
          match directions (Array.length xs) seed 3 with
          | [ u; v; t ] -> (u, v, t)
          | _ -> assert false
        in
        let x = vec xs in
        equal ~msg:"jvp of jvp of jvp"
          (near (magnitude xs [ u; v; t ]))
          [| contract xs [ u; v; t ] |]
          (Nx.to_array
             (Nx.reshape [| 1 |]
                (jvp_along (jvp_along (jvp_along total_prod u) v) t x)));
        equal ~msg:"grad of jvp of jvp"
          (near (gradient_scale xs [ u; v ]))
          (gradient_of xs [ u; v ])
          (Nx.to_array (Rune.grad' (jvp_along (jvp_along total_prod u) v) x)));
  ]

let product_cases =
  let gradient xs = Nx.to_array (Rune.grad' total_prod (vec xs)) in
  [
    test "cumprod is exact at zeros" (fun () ->
        equal ~msg:"[2; 3; 0]" floats [| 4.; 2.; 6. |]
          (gradient [| 2.; 3.; 0. |]);
        equal ~msg:"a leading zero" floats [| 9.; 0.; 0. |]
          (gradient [| 0.; 2.; 3. |]);
        equal ~msg:"two zeros" floats [| 1.; 8.; 0.; 0.; 0. |]
          (gradient [| 2.; 0.; 3.; 0.; 5. |]);
        equal ~msg:"jvp" floats [| 1.; 5.; 6. |]
          (Nx.to_array
             (snd
                (Rune.jvp' (scan Prod)
                   (vec [| 2.; 3.; 0. |])
                   (vec [| 1.; 1.; 1. |])))));
    test
      "the Hessian of cumprod at [2; 3; 0], summed over its rows, is [4; 3; 5]"
      (fun () ->
        equal floats [| 4.; 3.; 5. |]
          (Nx.to_array
             (Rune.grad'
                (fun x -> Nx.sum (Rune.grad' total_prod x))
                (vec [| 2.; 3.; 0. |]))));
    cases
      ~name:(fun xs ->
        Format.asprintf "second derivatives at %a are not lost" pp_floats xs)
      "subnormal factors"
      [ [| 1e-310; 2.; 3. |]; [| 1e-310; 1e-310; 4. |] ]
      (fun xs ->
        let n = Array.length xs in
        for i = 0 to n - 1 do
          let e = Array.init n (fun j -> if i = j then 1. else 0.) in
          equal
            ~msg:(Printf.sprintf "Hessian row %d" i)
            floats (gradient_of xs [ e ])
            (Nx.to_array
               (Rune.grad' (sum_with (Rune.grad' total_prod) e) (vec xs)))
        done);
  ]

let tests =
  [
    group "running extrema"
      (extremum_laws Max max_better
      @ extremum_laws Min min_better
      @ extremum_cases);
    group "running products" (product_laws @ product_cases);
  ]
