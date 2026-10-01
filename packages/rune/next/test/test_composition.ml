(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The composition matrix. One function, F x = Σ t(x) with t x = tanh(x²) eˣ,
   written plainly and through each of rune's constructs, goes through every
   ordered pair of grad, jvp, vmap and jit. A pair's expected value is the
   closed form of t, t' and t'' computed in OCaml floats, or the error the
   construct documents for that composition: no cell is left out. Ordered
   triples compare each construct with the plain function through the same
   composition. *)

open Windtrap
module Rune = Rune_next.Rune

let f64 = Nx.float64
let mat r c a = Nx.create f64 [| r; c |] a
let x () = Nx.create f64 [| 3 |] [| 0.7; -1.3; 2.1 |]
let xs () = mat 2 3 [| 0.7; -1.3; 2.1; 0.2; 0.9; -0.4 |]
let xss () = Nx.stack [ xs (); Nx.mul_s (xs ()) 0.5 ]
let v () = Nx.create f64 [| 3 |] [| 0.5; 1.; -2. |]
let vs () = mat 2 3 [| 0.5; 1.; -2.; 1.5; -0.3; 0.8 |]

(* The direction a jvp takes at an argument of shape [shape]. *)
let direction x =
  match Nx.shape x with
  | [| 3 |] -> v ()
  | [| 2; 3 |] -> vs ()
  | _ -> Nx.stack [ vs (); Nx.mul_s (vs ()) 2. ]

(* The function through each construct *)

let a = Rune.axis ()
let tot : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()
let term x = Nx.mul (Nx.tanh (Nx.mul x x)) (Nx.exp x)

(* t', with nx's operations, so that a differentiation around a rule
   differentiates it. *)
let term' x =
  let t = Nx.tanh (Nx.mul x x) in
  let s = Nx.sub (Nx.ones_like t) (Nx.mul t t) in
  Nx.mul (Nx.exp x) (Nx.add t (Nx.mul (Nx.mul_s x 2.) s))

let plain x = Nx.sum (term x)

type construct =
  | Plain
  | Scan
  | Remat
  | Custom_jvp
  | Custom_vjp
  | Lanes
  | Total_inside
  | Detach

let constructs =
  [ Plain; Scan; Remat; Custom_jvp; Custom_vjp; Lanes; Total_inside; Detach ]

let construct_name = function
  | Plain -> "plain"
  | Scan -> "scan"
  | Remat -> "remat"
  | Custom_jvp -> "custom_jvp"
  | Custom_vjp -> "custom_vjp"
  | Lanes -> "lanes"
  | Total_inside -> "a total collected inside"
  | Detach -> "detach"

let through = function
  | Plain -> plain
  | Scan ->
      fun x ->
        fst
          (Rune.scan'
             ~f:(fun acc xi -> (Nx.add acc (term xi), acc))
             ~init:(Nx.scalar f64 0.) x)
  | Remat -> Rune.remat Nx.Ptree.(tensor @-> returns tensor) plain
  | Custom_jvp ->
      Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
          (plain x, fun dx -> Nx.sum (Nx.mul (term' x) dx)))
  | Custom_vjp ->
      Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
          (plain x, fun ct -> Nx.mul ct (term' x)))
  | Lanes -> fun x -> Nx.sum (Nx.sum ~axes:[ 0 ] (Rune.lanes a (term x)))
  | Total_inside ->
      fun x ->
        snd
          (Rune.Total.collect tot ~zero:(Nx.scalar f64 0.) (fun () ->
               Rune.Total.add tot (plain x)))
  | Detach -> fun x -> Nx.sum (Nx.add (term x) (Rune.detach (term x)))

(* Compositions *)

type transformation = Grad | Jvp | Vmap | Jit

let transformation_name = function
  | Grad -> "grad"
  | Jvp -> "jvp"
  | Vmap -> "vmap"
  | Jit -> "jit"

(* [compose ts f] is [f] through [ts], the first outermost. The innermost map is
   named [a], so that [lanes a] in [f] addresses it. *)
let compose ts f =
  let rec go named = function
    | [] -> f
    | t :: rest -> (
        let inner_map = List.mem Vmap rest in
        let g = go (named || not inner_map) rest in
        match t with
        | Grad -> fun x -> Rune.grad' (fun x -> Nx.sum (g x)) x
        | Jvp -> fun x -> snd (Rune.jvp' g x (direction x))
        | Vmap -> if inner_map then Rune.vmap' g else Rune.vmap' ~axis:a g
        | Jit -> Rune.jit' g)
  in
  go false ts

let argument ts =
  match List.length (List.filter (( = ) Vmap) ts) with
  | 0 -> x ()
  | 1 -> xs ()
  | _ -> xss ()

let name ts = String.concat " ∘ " (List.map transformation_name ts)

(* What a cell waits for, when rune does not compute it yet: the cell runs and
   is expected to fail, so it turns red the day it passes. *)
let waits_for ts = if List.mem Jit ts then Some "jit" else None

let cell ?tags ts name f =
  let t = test ?tags name f in
  match waits_for ts with
  | Some why -> xfail ~reason:("pending: " ^ why) t
  | None -> t

(* Expected values *)

type outcome = Value of Nx.float64_t | Raises of string

let t_ x = Oracle.map Oracle.term x
let d1 x = Oracle.map Oracle.term' x
let d2 x = Oracle.map Oracle.term'' x
let rows m = Nx.sum ~axes:[ 1 ] (t_ m)
let lanes_of m = Nx.broadcast_to [| (Nx.shape m).(0) |] (Nx.sum (t_ m))

let no_forward =
  "Rune.jvp': a custom_vjp rule has no forward derivative; give the function a \
   custom_jvp rule"

(* The outcome of the pair [outer ∘ inner] of [c], from t, t' and t''. *)
let expected outer inner c =
  let x = x () and xs = xs () and v = v () and vs = vs () in
  let lanes = c = Lanes and twice = if c = Detach then 2. else 1. in
  let forward_inside =
    (* The innermost differentiation is forward mode. *)
    match (outer, inner) with
    | _, Jvp -> true
    | Jvp, (Vmap | Jit) -> true
    | _ -> false
  in
  if c = Custom_vjp && forward_inside then Raises no_forward
  else
    match (outer, inner) with
    | Grad, Grad -> Value (d2 x)
    | Grad, Jvp | Jvp, Grad -> Value (Nx.mul (d2 x) v)
    | Grad, Vmap -> Value (if lanes then Nx.mul_s (d1 xs) 2. else d1 xs)
    | Grad, Jit | Jit, Grad -> Value (d1 x)
    | Jvp, Jvp -> Value (Nx.sum (Nx.mul (d2 x) (Nx.mul v v)))
    | Jvp, Vmap ->
        let per_row = Nx.mul (d1 xs) vs in
        Value
          (if lanes then Nx.broadcast_to [| 2 |] (Nx.sum per_row)
           else Nx.sum ~axes:[ 1 ] per_row)
    | Jvp, Jit | Jit, Jvp -> Value (Nx.sum (Nx.mul (d1 x) v))
    | Vmap, Grad -> Value (if lanes then Nx.mul_s (d1 xs) 2. else d1 xs)
    | Vmap, Jvp ->
        let per_row = Nx.mul (d1 xs) (Nx.broadcast_to [| 2; 3 |] v) in
        Value
          (if lanes then Nx.broadcast_to [| 2 |] (Nx.sum per_row)
           else Nx.sum ~axes:[ 1 ] per_row)
    | Vmap, Vmap ->
        let per = Nx.sum ~axes:[ 2 ] (t_ (xss ())) in
        let per =
          if lanes then
            Nx.broadcast_to [| 2; 2 |] (Nx.sum ~axes:[ 1 ] ~keepdims:true per)
          else per
        in
        Value (Nx.mul_s per twice)
    | Vmap, Jit | Jit, Vmap ->
        Value (Nx.mul_s (if lanes then lanes_of xs else rows xs) twice)
    | Jit, Jit -> Value (Nx.mul_s (Nx.sum (t_ x)) twice)

let check outcome run =
  match outcome with
  | Value e -> equal (Oracle.tensor ~rel:1e-9 ()) e (run ())
  | Raises m -> raises (Invalid_argument m) (fun () -> ignore (run ()))

let transformations = [ Grad; Jvp; Vmap; Jit ]

let pairs =
  List.concat_map
    (fun o -> List.map (fun i -> [ o; i ]) transformations)
    transformations

let pair_group ts =
  let outer, inner = match ts with [ o; i ] -> (o, i) | _ -> assert false in
  let tags = if List.mem Jit ts then [ "slow" ] else [] in
  group ~tags (name ts)
    (List.map
       (fun c ->
         cell ts (construct_name c) (fun () ->
             check (expected outer inner c) (fun () ->
                 compose ts (through c) (argument ts))))
       constructs)

(* Triples: each construct against the plain function through the same
   composition, or its documented error. [lanes], whose value inside its map is
   not the plain function's, is checked by the pairs only. *)

let triples =
  List.concat_map
    (fun t1 ->
      List.concat_map
        (fun t2 ->
          List.filter_map
            (fun t3 ->
              if t1 = t2 || t2 = t3 || t1 = t3 then None
              else Some [ t1; t2; t3 ])
            transformations)
        transformations)
    transformations

(* The innermost differentiation of [ts], from the outside in. *)
let innermost_derivative ts =
  List.fold_left
    (fun acc t -> match t with Grad | Jvp -> Some t | Vmap | Jit -> acc)
    None ts

let triple_group ts =
  let tags = if List.mem Jit ts then [ "slow" ] else [] in
  group ~tags (name ts)
    (List.map
       (fun c ->
         cell ts (construct_name c) (fun () ->
             let run () = compose ts (through c) (argument ts) in
             let outcome =
               if c = Custom_vjp && innermost_derivative ts = Some Jvp then
                 Raises no_forward
               else Value (compose ts plain (argument ts))
             in
             check outcome run))
       (List.filter (( <> ) Lanes) constructs))

(* A total collected around each composition *)

let added x =
  Rune.Total.add tot (plain x);
  plain x

let total_group =
  group "a total collected around a composition counts each evaluation once"
    (List.map
       (fun ts ->
         let tags = if List.mem Jit ts then [ "slow" ] else [] in
         cell ~tags ts (name ts) (fun () ->
             let arg = argument ts in
             let _, total =
               Rune.Total.collect tot ~zero:(Nx.scalar f64 0.) (fun () ->
                   compose ts added arg)
             in
             equal (Oracle.tensor ~rel:1e-9 ()) (Nx.sum (t_ arg)) total))
       pairs)

(* Code that runs again (a remat's rerun, a rule an outer differentiation
   applies again) adds to a total only the first time. *)
let added_in_remat = Rune.remat Nx.Ptree.(tensor @-> returns tensor) added

let added_by_rule =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
      Rune.Total.add tot (plain x);
      (plain x, fun dx -> Nx.sum (Nx.mul (term' x) dx)))

let again_group =
  group "code that runs again adds to a total once"
    (List.map
       (fun (name, ts, f) ->
         test name (fun () ->
             let arg = argument ts in
             let _, total =
               Rune.Total.collect tot ~zero:(Nx.scalar f64 0.) (fun () ->
                   compose ts f arg)
             in
             equal (Oracle.tensor ~rel:1e-9 ()) (Nx.sum (t_ arg)) total))
       [
         ("a remat under grad", [ Grad ], added_in_remat);
         ("a remat under grad ∘ grad", [ Grad; Grad ], added_in_remat);
         ("a custom_jvp rule under jvp ∘ jvp", [ Jvp; Jvp ], added_by_rule);
         ("a custom_jvp rule under grad ∘ jvp", [ Grad; Jvp ], added_by_rule);
         ("a custom_jvp rule under jvp ∘ grad", [ Jvp; Grad ], added_by_rule);
       ])

(* An argument every lane shares *)

(* x · w with its pullback, w captured outside the map: the gradient of w is the
   sum over the lanes of each lane's. *)
let shared_tests =
  [
    test "a custom_vjp's cotangent of an argument the lanes share is their sum"
      (fun () ->
        let rule =
          Rune.custom_vjp
            Nx.Ptree.(pair tensor tensor)
            Nx.Ptree.tensor
            (fun (x, w) ->
              (Nx.sum (Nx.mul x w), fun g -> (Nx.mul g w, Nx.mul g x)))
        in
        let w = v () in
        let g =
          Rune.grad'
            (fun w -> Nx.sum (Rune.vmap' (fun x -> rule (x, w)) (xs ())))
            w
        in
        equal (Oracle.tensor ~rel:1e-12 ()) (Nx.sum ~axes:[ 0 ] (xs ())) g);
  ]

(* vjp's pullback inside the matrix *)

let pullback_group =
  group "a pullback of each construct"
    (List.concat_map
       (fun c ->
         [
           cell [ Vmap ]
             (construct_name c ^ " under vmap is its loop")
             (fun () ->
               let _, pb = Rune.vjp' (through c) (x ()) in
               let cts = Nx.create f64 [| 3 |] [| 1.; -2.; 0.5 |] in
               let loop =
                 Nx.stack (List.init 3 (fun i -> pb (Nx.get [ i ] cts)))
               in
               equal (Oracle.tensor ~rel:1e-12 ()) loop (Rune.vmap' pb cts));
           cell [ Grad ]
             (construct_name c ^ " from two domains")
             (fun () ->
               let _, pb = Rune.vjp' (through c) (x ()) in
               let ct = Nx.scalar f64 1.5 in
               let expected = pb ct in
               let d1 = Domain.spawn (fun () -> pb ct)
               and d2 = Domain.spawn (fun () -> pb ct) in
               equal ~msg:"first" (Oracle.tensor ()) expected (Domain.join d1);
               equal ~msg:"second" (Oracle.tensor ()) expected (Domain.join d2));
         ])
       constructs)

let () =
  exit
    (run "Rune compositions"
       [
         group "pairs" (List.map pair_group pairs);
         group "triples" (List.map triple_group triples);
         total_group;
         again_group;
         group "an argument the lanes share" shared_tests;
         pullback_group;
       ])
