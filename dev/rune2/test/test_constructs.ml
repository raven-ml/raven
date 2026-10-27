(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Constructs without programs: detach, lanes, lane_index and totals, eagerly,
   inside a traced step and under jit. *)

open Windtrap
open Compose
module Rune = Rune2
module Total = Rune.Total

let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let f32 = Nx.float32
let v xs = Nx.create f32 [| Array.length xs |] xs

let test_detach () =
  let x = v [| 1.; -0. |] in
  let one_value = Testable.make ~pp:Nx.pp ~equal:( == ) in
  equal ~msg:"x itself" one_value x (Rune.detach x);
  equal bits_t
    (bits (Nx.sin x))
    (bits (Rune.jit' (fun x -> Nx.sin (Rune.detach x)) x))

(* With no map around, lanes and lane_index raise eagerly and under jit. *)
let test_no_map () =
  let a = Rune.axis () in
  let no_lanes = "Rune.lanes: no map named by the axis is around the call" in
  let no_index =
    "Rune.lane_index: no map named by the axis is around the call"
  in
  invalid ~sub:no_lanes (fun () -> Rune.lanes a (v [| 1. |]));
  invalid ~sub:no_index (fun () -> Rune.lane_index a ());
  invalid ~sub:no_lanes (fun () ->
      Rune.jit' (fun x -> Rune.lanes a x) (v [| 1. |]));
  invalid ~sub:no_index (fun () ->
      Rune.jit'
        (fun x -> Nx.add x (Nx.cast f32 (Rune.lane_index a ())))
        (v [| 1. |]))

(* Totals *)

let zero () = Nx.scalar f32 (-0.)

let test_collect () =
  let t = Total.make () in
  let a = Nx.scalar f32 1.5 and b = Nx.scalar f32 (-0.25) in
  let r, sum =
    Total.collect t ~zero:(zero ()) (fun () ->
        Total.add t a;
        Total.add t b;
        "done")
  in
  equal string "done" r;
  equal bits_t (bits Nx.(add (add (zero ()) a) b)) (bits sum);
  (* No addition keeps the zero's bits. *)
  let _, sum = Total.collect t ~zero:(zero ()) (fun () -> ()) in
  equal bits_t (bits (zero ())) (bits sum);
  (* With no scope open, an addition does nothing. *)
  Total.add t a;
  (* A scope of another total lets the additions pass. *)
  let u = Total.make () in
  let ((), inner), outer =
    Total.collect t ~zero:(zero ()) (fun () ->
        Total.collect u ~zero:(zero ()) (fun () ->
            Total.add t a;
            Total.add u b))
  in
  equal bits_t (bits b) (bits inner);
  equal bits_t (bits a) (bits outer)

(* Two open scopes of one total: the innermost takes the additions. *)
let test_same_total () =
  let t = Total.make () in
  let a = Nx.scalar f32 1.5 and b = Nx.scalar f32 (-0.25) in
  let ((), inner), outer =
    Total.collect t ~zero:(zero ()) (fun () ->
        Total.add t a;
        Total.collect t ~zero:(zero ()) (fun () -> Total.add t b))
  in
  equal bits_t (bits Nx.(add (zero ()) b)) (bits inner);
  equal bits_t (bits Nx.(add (zero ()) a)) (bits outer)

(* A scope inside a jit sums the function's additions there. *)
let test_scope_in_jit () =
  let t = Total.make () in
  let f x =
    let y, s =
      Total.collect t ~zero:(zero ()) (fun () ->
          Total.add t (Nx.sum x);
          Nx.sin x)
    in
    Nx.add y s
  in
  let x = v [| 0.5; 2. |] in
  equal bits_t (bits (f x)) (bits (Rune.jit' f x))

(* An addition counts where it is made, and a traced function's as its construct
   returns: a jit whose function raises adds nothing. *)
let test_raising_jit () =
  let t = Total.make () in
  let f x =
    Total.add t x;
    raise Exit
  in
  let x = Nx.scalar f32 2. in
  let caught g = try ignore (g x) with Exit -> () in
  let (), eager = Total.collect t ~zero:(zero ()) (fun () -> caught f) in
  let (), jitted =
    Total.collect t ~zero:(zero ()) (fun () -> caught (Rune.jit' f))
  in
  equal bits_t (bits Nx.(add (zero ()) x)) (bits eager);
  equal bits_t (bits (zero ())) (bits jitted)

(* The additions a function makes, collected around it and around its jit, give
   the same total, its zero's sign kept. *)
let additions =
  let open Gen in
  let value = map (fun i -> pool.(i)) (int_range 0 (Array.length pool - 1)) in
  let zero = frequency [ (1, constant (-0.)); (1, constant 0.); (2, value) ] in
  pair zero (list ~size:(int_range 0 4) value)
  |> with_pp (fun ppf (z, cs) ->
      Format.fprintf ppf "zero %h, adds %s" z
        (String.concat " " (List.map (Printf.sprintf "%h") cs)))

let law_additions (z, cs) =
  cover "no addition" (cs = []);
  cover "a -0 zero" (z = 0. && 1. /. z < 0.);
  let t = Total.make () in
  let f x =
    List.iter (fun c -> Total.add t (Nx.mul x (Nx.scalar f32 c))) cs;
    Nx.neg x
  in
  let x = Nx.scalar f32 1. and zero () = Nx.scalar f32 z in
  let y, s = Total.collect t ~zero:(zero ()) (fun () -> f x) in
  let y', s' = Total.collect t ~zero:(zero ()) (fun () -> Rune.jit' f x) in
  equal bits_t (bits y) (bits y');
  equal bits_t (bits s) (bits s')

let test_shape_refused () =
  let t = Total.make () in
  invalid ~sub:"Rune.Total.add: shape [2], where the total's zero has []"
    (fun () ->
      Total.collect t ~zero:(zero ()) (fun () -> Total.add t (v [| 1.; 2. |])))

let test_raises () =
  let t = Total.make () in
  raises Exit (fun () ->
      Total.collect t ~zero:(zero ()) (fun () ->
          Total.add t (Nx.scalar f32 1.);
          raise Exit))

(* A scope around a jit receives the function's additions once per call. *)
let test_jit_scope () =
  let t = Total.make () in
  let f x =
    Total.add t x;
    Nx.sin x
  in
  let x = v [| 0.5; 2. |] in
  let y, sum =
    Total.collect t ~zero:(v [| 0.; 0. |]) (fun () -> Rune.jit' f x)
  in
  equal bits_t (bits (Nx.sin x)) (bits y);
  equal bits_t (bits (Nx.add (v [| 0.; 0. |]) x)) (bits sum)

let () =
  exit
    (run "rune2 constructs"
       [
         test "detach is its operand" test_detach;
         test "lanes and lane_index raise with no map" test_no_map;
         group "totals"
           [
             test "collect sums each addition once" test_collect;
             test "the innermost scope of a total" test_same_total;
             test "a scope inside a jit" test_scope_in_jit;
             prop "jit adds what eager adds" additions law_additions;
             test "a jit that raises adds nothing" test_raising_jit;
             test "an addition of another shape raises" test_shape_refused;
             test "an exception passes through collect" test_raises;
             test "a jit adds once per call" test_jit_scope;
           ];
       ])
