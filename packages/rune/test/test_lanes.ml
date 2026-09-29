(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Named maps and lane gathers: the map that owns a name answers [lanes a] with
   the whole lane batch, every other map passes the gather on while keeping its
   own lanes, jvp gathers the tangent, grad refuses the transpose, and outside
   any map named [a] the gather is one lane. *)

open Windtrap
open Rune_test_support.Support

let xs () = Nx.create f64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
let unit_dirs () = Nx.create f64 [| 2; 2 |] [| 1.; 0.; 0.; 1. |]

(* Semantics *)

let test_no_map_is_one_lane () =
  let a = Rune.axis () in
  let x = vec64 [| 1.; 2.; 3. |] in
  check_arr ~msg:"one lane" [| 1.; 2.; 3. |]
    (Nx.reshape [| 3 |] (Rune.lanes a x))

let test_names_are_generative () =
  let a = Rune.axis () and b = Rune.axis () in
  let x = vec64 [| 1.; 2.; 3. |] in
  (* [b] names no map here, so [lanes b] is one lane, not a's gather. *)
  let ys =
    Rune.vmap ~axis:a
      Nx.Ptree.(tensor @-> returns tensor)
      (fun _ -> Rune.lanes b x)
      (unit_dirs ())
  in
  check_arr ~msg:"b is not a" [| 1.; 2.; 3.; 1.; 2.; 3. |] ys

let test_named_map_gathers_the_lanes () =
  let a = Rune.axis () in
  let xs = xs () in
  (* Inside the map a row is a lane's argument; the gather is the whole [3;2]
     batch, a constant of the map, so every lane of the result holds it. *)
  let ys = Rune.vmap' ~axis:a (fun row -> Rune.lanes a row) xs in
  check_arr ~msg:"lane 0" (to_arr xs) (Nx.slice [ Nx.I 0 ] ys);
  check_arr ~msg:"every lane" (to_arr xs) (Nx.slice [ Nx.I 2 ] ys)

let test_named_map_broadcasts_a_constant () =
  let a = Rune.axis () in
  let c = vec64 [| 7.; 8. |] in
  let ys = Rune.vmap' ~axis:a (fun _ -> Rune.lanes a c) (xs ()) in
  (* One [3;2] gather per lane: the same constant stacked over the lanes. *)
  check_arr ~msg:"constant lane 0"
    [| 7.; 8.; 7.; 8.; 7.; 8. |]
    (Nx.slice [ Nx.I 0 ] ys);
  check_arr ~msg:"constant every lane"
    [| 7.; 8.; 7.; 8.; 7.; 8. |]
    (Nx.slice [ Nx.I 2 ] ys)

let test_anonymous_map_passes_the_gather_on () =
  let a = Rune.axis () in
  let es = vec64 [| 10.; 20.; 30. |] in
  (* Every direction lane meets every example lane: the inner map keeps its own
     axis in front and the outer map supplies the gather. *)
  let ys =
    Rune.vmap ~axis:a
      Nx.Ptree.(tensor @-> returns tensor)
      (fun _ -> Rune.vmap' (fun e -> Rune.lanes a e) es)
      (unit_dirs ())
  in
  check_arr ~msg:"all directions per example"
    [| 10.; 10.; 20.; 20.; 30.; 30. |]
    (Nx.slice [ Nx.I 0 ] ys)

let test_another_named_map_answers_no_collective () =
  let a = Rune.axis () and b = Rune.axis () in
  let es = vec64 [| 10.; 20.; 30. |] in
  (* The map named [b] is an "other" map for [lanes a]: it passes the gather on
     and swaps the answer's leading axes, so the body still addresses [a]. *)
  let ys =
    Rune.vmap ~axis:a
      Nx.Ptree.(tensor @-> returns tensor)
      (fun _ -> Rune.vmap' ~axis:b (fun e -> Rune.lanes a e) es)
      (unit_dirs ())
  in
  check_arr ~msg:"b passes a on"
    [| 10.; 10.; 20.; 20.; 30.; 30. |]
    (Nx.slice [ Nx.I 0 ] ys)

let test_jvp_gathers_the_tangent () =
  let a = Rune.axis () in
  let x = vec64 [| 1.; 2.; 3. |] in
  let v = vec64 [| 0.5; -0.5; 1.0 |] in
  (* No map named [a]: one lane, and its tangent gathers too. *)
  let y, dy = Rune.jvp' (fun x -> Rune.lanes a x) x v in
  check_arr ~msg:"primal" [| 1.; 2.; 3. |] (Nx.reshape [| 3 |] y);
  check_arr ~msg:"tangent" [| 0.5; -0.5; 1.0 |] (Nx.reshape [| 3 |] dy);
  (* Under the map named [a], the gather of the map's input is linear. *)
  let dirs =
    Nx.create f64 [| 3; 3 |] [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]
  in
  let ys =
    Rune.vmap ~axis:a
      Nx.Ptree.(tensor @-> returns (pair tensor tensor))
      (fun d -> Rune.jvp' (fun x -> Rune.lanes a x) x d)
      dirs
  in
  let y, dy = ys in
  check_arr ~msg:"batched primal"
    [| 1.; 2.; 3.; 1.; 2.; 3.; 1.; 2.; 3. |]
    (Nx.slice [ Nx.I 0 ] y);
  check_arr ~msg:"batched tangent"
    [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]
    (Nx.slice [ Nx.I 0 ] dy)

let test_a_named_map_passes_axis_index_on () =
  (* A named map answers no collective but its gather: `Nx.Rng.fold_in_axis` in
     code the map encloses folds the index of an inner map, or one lane's index
     when there is none — never the direction map's. *)
  let a = Rune.axis () in
  let key = Nx.Rng.key 7 in
  let keys =
    Rune.vmap ~axis:a
      Nx.Ptree.(tensor @-> returns tensor)
      (fun _ -> (Nx.Rng.fold_in_axis key :> Nx.int32_t))
      (unit_dirs ())
  in
  let lane i = Nx.to_array (Nx.slice [ Nx.I i ] keys) in
  if lane 0 <> lane 1 then fail "a named map answered axis_index itself"

let test_grad_refuses_the_transpose () =
  let a = Rune.axis () in
  raises_match Exn.invalid_arg (fun () ->
      ignore (Rune.grad' (fun x -> Nx.sum (Rune.lanes a x)) (vec64 [| 1. |])))

let test_compiled_gather_is_one_lane () =
  let a = Rune.axis () in
  let f = Rune.jit' ~devices:[ Rune.device "CPU" ] (fun x -> Rune.lanes a x) in
  check_arr ~msg:"compiled one lane" [| 1.; 2.; 3. |]
    (Nx.reshape [| 3 |] (f (vec64 [| 1.; 2.; 3. |])))

let tests =
  [
    group "semantics"
      [
        test "no map is one lane" test_no_map_is_one_lane;
        test "names are generative" test_names_are_generative;
        test "the named map gathers its lanes" test_named_map_gathers_the_lanes;
        test "the named map broadcasts a constant"
          test_named_map_broadcasts_a_constant;
        test "an anonymous map passes the gather on"
          test_anonymous_map_passes_the_gather_on;
        test "another named map answers no collective"
          test_another_named_map_answers_no_collective;
        test "a named map passes axis_index on"
          test_a_named_map_passes_axis_index_on;
      ];
    group "transformations"
      [
        test "jvp gathers the tangent" test_jvp_gathers_the_tangent;
        test "grad refuses the transpose" test_grad_refuses_the_transpose;
        test "compiled gathers are one lane" test_compiled_gather_is_one_lane;
      ];
  ]

let () = exit (run "rune lanes" tests)
