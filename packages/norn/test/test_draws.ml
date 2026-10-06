(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module D = Norn.Draws

type 'a pos = { a : 'a; b : 'a }

module Pos = struct
  type 'a t = 'a pos

  let walk c { a; b } =
    let open Nx.Ptree.Walk in
    let a = field c "a" leaf a in
    let b = field c "b" leaf b in
    { a; b }
end

let pos : Nx.float64_t pos Nx.Ptree.t = Nx.Ptree.instantiate (module Pos)
let floats x = Nx.to_array x

(* Draws of 2 chains of 3: [a] scalar per draw, [b] a vector of 2. Element [(i,
   j)] of [a] is [10 i + j]. *)
let draws =
  D.v pos
    {
      a = Nx.create Nx.float64 [| 2; 3 |] [| 0.; 1.; 2.; 10.; 11.; 12. |];
      b = Nx.reshape [| 2; 3; 2 |] (Nx.arange_f Nx.float64 0. 12. 1.);
    }

let ( !! ) (type u) (d : u D.t) : u = (d :> u)

let validation =
  group "v"
    [
      test "a tensor needs a chain and a draw axis" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"a: shape [3]") (fun () ->
              D.v pos
                {
                  a = Nx.zeros Nx.float64 [| 3 |];
                  b = Nx.zeros Nx.float64 [| 2; 3; 2 |];
                }));
      test "every tensor has the same chains and draws" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"b: 2 chains of 4 draws")
            (fun () ->
              D.v pos
                {
                  a = Nx.zeros Nx.float64 [| 2; 3 |];
                  b = Nx.zeros Nx.float64 [| 2; 4; 2 |];
                }));
    ]

let maps =
  group "map"
    [
      test "a function of one draw applies to every draw" (fun () ->
          let d =
            D.map pos Nx.Ptree.tensor (fun p -> Nx.add p.a (Nx.sum p.b)) draws
          in
          let d = !!d in
          equal (array int) [| 2; 3 |] (Nx.shape d);
          let expected =
            Array.init 6 (fun k ->
                let i = k / 3 and j = k mod 3 in
                float_of_int ((10 * i) + j) +. float_of_int ((4 * k) + 1))
          in
          equal (array (float 1e-12)) expected (floats d));
      test "draw j of chain i has the key fold_in k (i n + j)" (fun () ->
          let k = Nx.Rng.key 7 in
          let d =
            D.simulate pos Nx.Ptree.tensor
              (fun k _ -> Nx.Rng.uniform k Nx.float64 [||])
              k draws
          in
          let d = !!d in
          for i = 0 to 1 do
            for j = 0 to 2 do
              let expected =
                Nx.item []
                  (Nx.Rng.uniform
                     (Nx.Rng.fold_in k ((i * 3) + j))
                     Nx.float64 [||])
              in
              equal
                ~msg:(Printf.sprintf "(%d, %d)" i j)
                float_exact expected
                (Nx.item [ i; j ] d)
            done
          done);
    ]

let joins =
  group "append and thin"
    [
      test "append puts the second draws after the first in each chain"
        (fun () ->
          let d = !!(D.append pos draws draws) in
          equal (array int) [| 2; 6 |] (Nx.shape d.a);
          equal
            (array (float 0.1))
            [| 10.; 11.; 12.; 10.; 11.; 12. |]
            (floats (Nx.slice [ Nx.I 1 ] d.a)));
      test "thin keeps every n-th draw from the first" (fun () ->
          let d = !!(D.thin pos ~every:2 draws) in
          equal (array (float 0.1)) [| 0.; 2.; 10.; 12. |] (floats d.a);
          equal (array int) [| 2; 2; 2 |] (Nx.shape d.b));
      test "thinning every draw keeps them all" (fun () ->
          equal
            (array (float 0.1))
            (floats !!draws.a)
            (floats !!(D.thin pos ~every:1 draws).a));
      test "thin refuses a step below one" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"every = 0") (fun () ->
              D.thin pos ~every:0 draws));
    ]

let structures =
  group "structures"
    [
      test "draws are walked as their structure" (fun () ->
          let paths =
            List.filter_map
              (function
                | Nx.Ptree.Leaf p -> Some (Nx.Ptree.Path.to_string p)
                | Report _ -> None)
              (Nx.Ptree.visits (D.ptree pos) draws)
          in
          equal (list string) [ "a"; "b" ] paths);
      test "statistics are walked field by field" (fun () ->
          let st =
            Norn.Stats.
              {
                lp = Nx.zeros Nx.float32 [| 2 |];
                acceptance = Nx.zeros Nx.float32 [| 2 |];
                step_size = Nx.zeros Nx.float32 [| 2 |];
                n_steps = Nx.zeros Nx.int32 [| 2 |];
                diverging = Nx.zeros Nx.bool [| 2 |];
                saturated = Nx.zeros Nx.bool [| 2 |];
                energy = Nx.zeros Nx.float32 [| 2 |];
              }
          in
          let paths =
            List.filter_map
              (function
                | Nx.Ptree.Leaf p -> Some (Nx.Ptree.Path.to_string p)
                | Report _ -> None)
              (Nx.Ptree.visits (Norn.Stats.ptree Nx.float32) st)
          in
          equal (list string)
            [
              "lp";
              "acceptance";
              "step_size";
              "n_steps";
              "diverging";
              "saturated";
              "energy";
            ]
            paths);
    ]

let () = exit (run "Norn.Draws" [ validation; maps; joins; structures ])
