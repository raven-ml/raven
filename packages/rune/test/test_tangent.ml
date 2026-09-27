(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reading tangents: the query is answered by the innermost forward mode and is
   silent outside one, under no_grad, and for constants of the differentiation.
   It is the mechanism a consumer of tangents — sofo's curvature collector —
   uses to see a forward pass's directional derivatives, including the whole
   [k]-lane batch a [vmap] around [jvp] produces. *)

open Windtrap
open Rune_test_support.Support

let v3 () = vec64 [| 0.7; -1.3; 2.1 |]

(* A deterministic, non-uniform lane batch for the batched cases. *)
let lane_batch ~k t =
  let s = Nx.shape t in
  let n = Nx.numel t in
  let data =
    Array.init (k * n) (fun i ->
        let lane = i / n and j = i mod n in
        float_of_int ((((j * 7) + (3 * lane)) mod 11) - 5) /. 4.0)
  in
  Nx.create f64 (Array.append [| k |] s) data

(* The single-tangent answer: inside [jvp] the query returns the tangent the
   seed put there, for a value in the middle of the graph. *)

let test_single_tangent_under_jvp () =
  let x = v3 () in
  let v = tangent_like x in
  let f x = Nx.mul (Nx.sin x) x in
  let seen = ref None in
  let g x =
    let y = f x in
    seen := Some (Rune.tangent y);
    y
  in
  ignore (Rune.jvp' g x v);
  let _, expected = Rune.jvp' f x v in
  match !seen with
  | Some (Some dy) -> check_arr ~msg:"query under jvp" (to_arr expected) dy
  | Some None -> fail "the query reported no tangent for a tracked tensor"
  | None -> fail "the query was not made"

(* Degradations: no forward mode, a disabled gate, and constants. *)

let test_silent_outside_a_forward_mode () =
  match Rune.tangent (v3 ()) with
  | None -> ()
  | Some _ -> fail "a tangent was reported with no forward mode installed"

let test_silent_under_no_grad () =
  let x = v3 () in
  let seen = ref None in
  let g x =
    let y = Nx.mul x x in
    Rune.no_grad (fun () -> seen := Some (Rune.tangent y));
    y
  in
  ignore (Rune.jvp' g x (tangent_like x));
  match !seen with
  | Some None -> ()
  | _ -> fail "a tangent was reported inside no_grad"

let test_constant_intermediate_has_no_tangent () =
  let x = v3 () in
  let seen = ref None in
  let g x =
    let c = Nx.ones_like x in
    seen := Some (Rune.tangent c);
    Nx.mul x c
  in
  ignore (Rune.jvp' g x (tangent_like x));
  match !seen with
  | Some None -> ()
  | _ -> fail "a constant intermediate reported a tangent"

let test_detached_tensor_has_no_tangent () =
  let x = v3 () in
  let seen = ref None in
  let g x =
    let y = Nx.mul x x in
    let d = Rune.detach y in
    seen := Some (Rune.tangent d);
    Nx.add y d
  in
  ignore (Rune.jvp' g x (tangent_like x));
  match !seen with
  | Some None -> ()
  | _ -> fail "a detached tensor reported a tangent"

(* Nesting: the innermost forward mode owns the answer; an enclosing store is
   never consulted for a tensor the inner mode does not track. *)

let test_innermost_forward_mode_owns_the_answer () =
  let x = v3 () in
  let v = tangent_like x in
  let inner_seed = ref None in
  let inner_own = ref None in
  let inner_outer = ref None in
  let outer x =
    let seeded = Nx.mul x x in
    let outer_only = Nx.sin x in
    ignore
      (Rune.jvp'
         (fun w ->
           inner_seed := Some (Rune.tangent w);
           inner_outer := Some (Rune.tangent outer_only);
           let y = Nx.mul w w in
           inner_own := Some (Rune.tangent y);
           y)
         seeded v
      |> fst);
    Nx.sum outer_only
  in
  ignore (Rune.jvp' outer x v);
  (match !inner_seed with
  | Some (Some dy) ->
      check_arr ~msg:"the inner mode answers the tensor it seeds" (to_arr v) dy
  | _ -> fail "the inner forward mode did not answer the tensor it seeds");
  (match !inner_own with
  | Some (Some dy) ->
      check_arr ~msg:"the inner mode answers its own interior"
        (to_arr (Nx.mul_s (Nx.mul v (Nx.mul x x)) 2.0))
        dy
  | _ -> fail "the inner forward mode did not answer its own tensor");
  match !inner_outer with
  | Some None -> ()
  | _ -> fail "the inner mode leaked an enclosing store's tangent"

(* Reads are inert, and a fold's steps answer as they run. *)

let test_querying_does_not_perturb () =
  let x = v3 () in
  let v = tangent_like x in
  let f x = Nx.sum (Nx.mul (Nx.sin x) (Nx.exp x)) in
  let g x =
    let y = f x in
    ignore (Rune.tangent y : Nx.float64_t option);
    y
  in
  let y_plain, dy_plain = Rune.jvp' f x v in
  let y_query, dy_query = Rune.jvp' g x v in
  check_arr ~msg:"value" (to_arr y_plain) y_query;
  check_arr ~msg:"tangent" (to_arr dy_plain) dy_query

let test_queries_inside_a_scan_body () =
  let x = vec64 [| 0.5; -1.0 |] in
  let v = vec64 [| 0.3; -0.7 |] in
  let xs =
    Nx.reshape [| 4; 1 |]
      (Nx.create f64 [| 4 |] [| 0.2; -0.4; 0.6; 0.1 |])
  in
  let answers = ref [] in
  let step c xi =
    let c' = Nx.tanh (Nx.add c (Nx.mul_s xi 0.5)) in
    answers := Rune.tangent c' :: !answers;
    (c', c')
  in
  let f x =
    let c, _ys = Rune.scan' ~f:step ~init:x xs in
    Nx.sum c
  in
  let _, dy = Rune.jvp' f x v in
  equal ~msg:"one answer per step" int 4 (List.length !answers);
  List.iter
    (function
      | Some dc ->
          equal ~msg:"a step's tangent has the carry's shape" int 1
            (Array.length (Nx.shape dc))
      | None -> fail "a step's tangent was not answered")
    !answers;
  equal ~msg:"the scalar loss has a scalar tangent" int 0
    (Array.length (Nx.shape dy))

(* The batch of a vmap around jvp: the composition that replaces a dedicated
   batched forward mode. The mapped function runs once for all lanes, and the
   query inside it answers the whole [k]-lane batch. *)

let test_vmap_over_jvp_answers_the_whole_batch () =
  let x = v3 () in
  let k = 3 in
  let thetas = lane_batch ~k x in
  let f x = Nx.mul (Nx.sin x) x in
  let seen = ref None in
  let fq x =
    let y = f x in
    seen := Some (Rune.tangent y);
    y
  in
  let dy = Rune.vmap' (fun v -> snd (Rune.jvp' fq x v)) thetas in
  let expected =
    Nx.stack
      (List.init k (fun j -> snd (Rune.jvp' f x (Nx.slice [ Nx.I j ] thetas))))
  in
  check_arr ~msg:"vmap over jvp is the stacked per-lane JVPs" (to_arr expected)
    dy;
  match !seen with
  | Some (Some dl) ->
      equal ~msg:"the query leads with the lane axis" int k (Nx.shape dl).(0);
      check_arr ~msg:"the query is the whole tangent batch" (to_arr expected) dl
  | _ -> fail "the query did not answer the batch inside vmap over jvp"

let test_the_batch_is_read_outside_the_map () =
  (* Inside the map's extent [Nx.shape] reports the map's virtual, lane-less
     shape: the answer physically carries the batch, but an in-scope consumer
     cannot see the lanes — only reading it after the map returns gives the
     batch. *)
  let x = v3 () in
  let k = 2 in
  let thetas = lane_batch ~k x in
  let f x = Nx.sum (Nx.mul (Nx.sin x) x) in
  let in_scope_shape = ref None in
  let saved = ref None in
  let fq x =
    let l = f x in
    (match Rune.tangent l with
    | Some dl ->
        in_scope_shape := Some (Nx.shape dl);
        saved := Some dl
    | None -> ());
    l
  in
  let dy = Rune.vmap' (fun v -> snd (Rune.jvp' fq x v)) thetas in
  (match !in_scope_shape with
  | Some s ->
      equal ~msg:"the in-scope shape is the map's virtual shape" int 0
        (Array.length s)
  | None -> fail "the query was not answered in scope");
  match !saved with
  | Some dl ->
      equal ~msg:"out of scope the lane axis is visible" int k
        (Nx.shape dl).(0);
      check_arr ~msg:"the scalar loss's tangent batch" (to_arr dy) dl
  | None -> fail "the query was not answered"

(* sofo's structure: a parameter record walked by [Nx.Ptree], a mapped
   signature, and queries of a prediction and of the loss inside. *)

type tang_params = { w : Nx.float64_t; b : Nx.float64_t }

module Tang_params = struct
  type _ t = tang_params

  let walk c { w; b } =
    let open Nx.Ptree.Walk in
    { w = field c "w" tensor w; b = field c "b" tensor b }
end

let tang_params_ptree : tang_params Nx.Ptree.t =
  Nx.Ptree.instantiate (module Tang_params)

let test_structure_query_under_vmap_over_jvp () =
  let p = { w = vec64 [| 0.7; -1.3; 2.1 |]; b = vec64 [| -0.4; 0.9 |] } in
  let k = 2 in
  let thetas = { w = lane_batch ~k p.w; b = lane_batch ~k p.b } in
  let loss p =
    let y = Nx.mul (Nx.sin p.w) p.w in
    Nx.add (Nx.sum y) (Nx.sum (Nx.mul p.b p.b))
  in
  let seen_y = ref None in
  let seen_l = ref None in
  let calls = ref 0 in
  let f p =
    incr calls;
    let y = Nx.mul (Nx.sin p.w) p.w in
    let l = Nx.add (Nx.sum y) (Nx.sum (Nx.mul p.b p.b)) in
    seen_y := Some (Rune.tangent y);
    seen_l := Some (Rune.tangent l);
    l
  in
  let dy =
    Rune.vmap
      Nx.Ptree.(tang_params_ptree @-> returns tensor)
      (fun th -> snd (Rune.jvp tang_params_ptree Nx.Ptree.tensor f p th))
      thetas
  in
  equal ~msg:"the mapped function runs once for the whole batch" int 1 !calls;
  let expected =
    Nx.stack
      (List.init k (fun j ->
           let lane leaf = Nx.slice [ Nx.I j ] leaf in
           snd
             (Rune.jvp tang_params_ptree Nx.Ptree.tensor loss p
                { w = lane thetas.w; b = lane thetas.b })))
  in
  check_arr ~msg:"the mapped loss tangent" (to_arr expected) dy;
  match (!seen_y, !seen_l) with
  | Some (Some dy'), Some (Some dl') ->
      equal ~msg:"the prediction tangent leads with the lane axis" int k
        (Nx.shape dy').(0);
      check_arr ~msg:"the loss tangent is the whole batch" (to_arr expected) dl'
  | _ -> fail "a structure query was not answered under vmap over jvp"

let test_scan_steps_under_vmap_over_jvp () =
  let x = vec64 [| 0.5; -1.0 |] in
  let k = 3 in
  let thetas = lane_batch ~k x in
  let xs =
    Nx.reshape [| 4; 1 |]
      (Nx.create f64 [| 4 |] [| 0.2; -0.4; 0.6; 0.1 |])
  in
  let answers = ref [] in
  let step c xi =
    let c' = Nx.tanh (Nx.add c (Nx.mul_s xi 0.5)) in
    (c', c')
  in
  let g x =
    let c, _ys =
      Rune.scan'
        ~f:(fun c xi ->
          let c', _ = step c xi in
          answers := Rune.tangent c' :: !answers;
          (c', c'))
        ~init:x xs
    in
    Nx.sum c
  in
  let g_ref x =
    let c, _ys = Rune.scan' ~f:step ~init:x xs in
    Nx.sum c
  in
  let dy = Rune.vmap' (fun v -> snd (Rune.jvp' g x v)) thetas in
  equal ~msg:"one answer per step" int 4 (List.length !answers);
  List.iter
    (function
      | Some dc ->
          equal ~msg:"a step's answer leads with the lane axis" int k
            (Nx.shape dc).(0);
          equal ~msg:"and keeps the carry's rank" int 2
            (Array.length (Nx.shape dc))
      | None -> fail "a step's tangent was not answered")
    !answers;
  let expected =
    Nx.stack
      (List.init k (fun j ->
           snd (Rune.jvp' g_ref x (Nx.slice [ Nx.I j ] thetas))))
  in
  check_arr ~msg:"the fold's endpoint batch" (to_arr expected) dy

let tests =
  [
    group "the installed forward mode answers"
      [ test "single tangent under jvp" test_single_tangent_under_jvp ];
    group "degradations"
      [
        test "silent outside a forward mode" test_silent_outside_a_forward_mode;
        test "silent under no_grad" test_silent_under_no_grad;
        test "a constant intermediate has no tangent"
          test_constant_intermediate_has_no_tangent;
        test "a detached tensor has no tangent"
          test_detached_tensor_has_no_tangent;
      ];
    group "nesting"
      [
        test "the innermost mode owns the answer"
          test_innermost_forward_mode_owns_the_answer;
      ];
    group "reads are inert"
      [
        test "querying does not perturb the run" test_querying_does_not_perturb;
        test "queries inside a scan body" test_queries_inside_a_scan_body;
      ];
    group "a vmap around jvp"
      [
        test "the query is the whole tangent batch"
          test_vmap_over_jvp_answers_the_whole_batch;
        test "the batch is read outside the map"
          test_the_batch_is_read_outside_the_map;
        test "a structure query is the whole batch"
          test_structure_query_under_vmap_over_jvp;
        test "a fold's steps carry the batch"
          test_scan_steps_under_vmap_over_jvp;
      ];
  ]

let () = exit (run "rune tangent" tests)
