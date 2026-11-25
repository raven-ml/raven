(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module D = Norn.Draws

type 'a pos = { a : 'a; b : 'a }

module Pos = struct
  type ('a, _) t = 'a pos

  let walk c { a; b } =
    let open Nx.Ptree.Walk in
    let a = field c "a" leaf a in
    let b = field c "b" leaf b in
    { a; b }
end

let pos : (Nx.host Nx.float64_t pos, Nx.host) Nx.Ptree.t =
  Nx.Ptree.instantiate (module Pos)

let t = Nx.Ptree.tensor
let floats x = Nx.to_array x
let ( !! ) (type u) (d : u D.t) : u = (d :> u)

let paths s x =
  List.filter_map
    (function
      | Nx.Ptree.Leaf p -> Some (Nx.Ptree.Path.to_string p) | Report _ -> None)
    (Nx.Ptree.visits s x)

(* 2 chains of 3 draws: [a] a scalar, element [(i, j)] equal to [10 i + j], and
   [b] a vector of 2. *)
let draws () =
  D.v pos
    {
      a = Nx.create Nx.float64 [| 2; 3 |] [| 0.; 1.; 2.; 10.; 11.; 12. |];
      b = Nx.reshape [| 2; 3; 2 |] (Nx.arange_f Nx.float64 0. 12. 1.);
    }

let stats =
  test "Stats.ptree walks its fields in order" (fun () ->
      let z = Nx.zeros Nx.float32 [| 2 |] and no = Nx.zeros Nx.bool [| 2 |] in
      let st =
        Norn.Stats.
          {
            lp = z;
            acceptance = z;
            step_size = z;
            n_steps = Nx.zeros Nx.int32 [| 2 |];
            diverging = no;
            saturated = no;
            energy = z;
          }
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
        (paths (Norn.Stats.ptree Nx.float32) st))

let v =
  group "v"
    [
      test "raises naming a tensor with fewer than two axes" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Draws.v: a: ") (fun () ->
              D.v pos
                {
                  a = Nx.zeros Nx.float64 [| 3 |];
                  b = Nx.zeros Nx.float64 [| 2; 3; 2 |];
                }));
      test "raises naming a tensor of other chains or draws" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Draws.v: b: ") (fun () ->
              D.v pos
                {
                  a = Nx.zeros Nx.float64 [| 2; 3 |];
                  b = Nx.zeros Nx.float64 [| 2; 4; 2 |];
                }));
    ]

let ptree =
  test "ptree has the structure's paths" (fun () ->
      equal (list string) [ "a"; "b" ] (paths (D.ptree pos) (draws ())))

let map =
  group "map"
    [
      test "applies f to every draw" (fun () ->
          let d =
            !!(D.map pos t (fun p -> Nx.add p.a (Nx.sum p.b)) (draws ()))
          in
          (* Draw [k], chain-major, has [a] of [10 (k / 3) + k mod 3] and [b]
             summing to [4 k + 1]. *)
          let expected =
            Array.init 6 (fun k ->
                float_of_int ((10 * (k / 3)) + (k mod 3) + (4 * k) + 1))
          in
          equal (array int) [| 2; 3 |] (Nx.shape d);
          equal (array float_exact) expected (floats d));
      test "runs f once for all draws" (fun () ->
          let calls = ref 0 in
          let count p =
            incr calls;
            p.a
          in
          ignore (D.map pos t count (draws ()));
          equal int 1 !calls);
    ]

let simulate =
  test "simulate gives draw j of chain i the key fold_in k (i n + j)" (fun () ->
      let k = Nx.Rng.key 7 in
      let uniform k = Nx.Rng.uniform ~key:k Nx.float64 [||] in
      let d = !!(D.simulate pos t (fun k _ -> uniform k) k (draws ())) in
      let expected =
        Array.init 6 (fun ij -> Nx.item [] (uniform (Nx.Rng.fold_in k ij)))
      in
      equal (array float_exact) expected (floats d))

let simulate_nothing =
  test "simulate raises for draws with no tensor" (fun () ->
      let u = Nx.Ptree.unit in
      raises_match (Exn.invalid_arg ~sub:"Norn.Draws.simulate") (fun () ->
          D.simulate u u (fun _ () -> ()) (Nx.Rng.key 0) (D.v u ())))

let append =
  let l = Nx.Ptree.list t in
  let x = Nx.zeros Nx.float64 [| 2; 3 |] in
  group "append"
    [
      test "puts the second draws after the first in every chain" (fun () ->
          let d =
            !!(D.append pos (draws ()) (D.thin pos ~every:2 (draws ())))
          in
          equal (array float_exact)
            [| 0.; 1.; 2.; 0.; 2.; 10.; 11.; 12.; 10.; 12. |]
            (floats d.a);
          equal (array int) [| 2; 5; 2 |] (Nx.shape d.b));
      test "raises for another number of chains" (fun () ->
          let y = Nx.zeros Nx.float64 [| 3; 3 |] in
          raises_match (Exn.invalid_arg ~sub:"Norn.Draws.append") (fun () ->
              D.append l (D.v l [ x ]) (D.v l [ y ])));
      test "raises for other visits" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Draws.append") (fun () ->
              D.append l (D.v l [ x ]) (D.v l [ x; x ])));
      test "raises for a tensor of another element shape" (fun () ->
          let d = D.v t (Nx.zeros Nx.float64 [| 2; 3; 2 |]) in
          let d' = D.v t (Nx.zeros Nx.float64 [| 2; 3; 3 |]) in
          raises_match (Exn.invalid_arg ~sub:"Norn.Draws.append") (fun () ->
              D.append t d d'));
    ]

let thin =
  group "thin"
    [
      prop "keeps every every-th draw from the first" (Gen.int_range 1 5)
        (fun every ->
          let kept =
            List.filter
              (fun a -> a mod 10 mod every = 0)
              [ 0; 1; 2; 10; 11; 12 ]
          in
          equal (list float_exact)
            (List.map float_of_int kept)
            (Array.to_list (floats !!(D.thin pos ~every (draws ())).a)));
      test "raises for every below 1" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Draws.thin") (fun () ->
              D.thin pos ~every:0 (draws ())));
    ]

let () =
  exit
    (run "Norn.Draws"
       [ stats; v; ptree; map; simulate; simulate_nothing; append; thin ])
