(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reading tangents: the query is answered by the innermost forward mode — the
   single tangent under jvp, the k-lane batch under jvp_k — and is silent
   outside one, under no_grad, and for constants of the differentiation. *)

open Windtrap
open Rune_test_support.Support

let v3 () = vec64 [| 0.7; -1.3; 2.1 |]

(* A deterministic lane batch for the jvp_k cases. *)
let lane_batch ~k t =
  let s = Nx.shape t in
  let n = Nx.numel t in
  let data =
    Array.init (k * n) (fun i ->
        let lane = i / n and j = i mod n in
        float_of_int (((j * 7) + (3 * lane)) mod 11 - 5) /. 4.0)
  in
  Nx.create f64 (Array.append [| k |] s) data

(* The single-tangent answer: inside jvp the query returns the tangent the seed
   put there, for a value in the middle of the graph. *)

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

let test_lane_batch_under_jvp_k () =
  let x = v3 () in
  let k = 3 in
  let thetas = lane_batch ~k x in
  let f x = Nx.mul (Nx.sin x) x in
  let seen = ref None in
  let g x =
    let y = f x in
    seen := Some (Rune.tangent y);
    y
  in
  ignore (Rune.jvp_k' g x thetas);
  let _, expected = Rune.jvp_k' f x thetas in
  match !seen with
  | Some (Some dy) ->
      equal ~msg:"the lane axis leads" int k (Nx.shape dy).(0);
      check_arr ~msg:"query under jvp_k" (to_arr expected) dy
  | _ -> fail "the query did not return the tangent batch inside jvp_k"

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

(* Nesting: the innermost mode owns the answer and its shape convention. *)

let test_innermost_mode_owns_the_convention () =
  (* jvp_k outside, jvp inside. Within the inner scope, a tensor the inner jvp
     differentiates is answered with its single tangent — not the enclosing
     mode's batch — and a tensor only the outer mode tracks is answered [None]
     rather than by leaking the outer store. *)
  let x = v3 () in
  let k = 3 in
  let thetas = lane_batch ~k x in
  let v = tangent_like x in
  let inner_seed = ref None in
  let inner_own = ref None in
  let inner_outer = ref None in
  let outer x =
    let seeded = Nx.mul x x in
    let outer_only = Nx.sin x in
    let _ =
      Rune.jvp'
        (fun w ->
          inner_seed := Some (Rune.tangent w);
          inner_outer := Some (Rune.tangent outer_only);
          let y = Nx.mul w w in
          inner_own := Some (Rune.tangent y);
          y)
        seeded v
      |> fst
    in
    Nx.sum outer_only
  in
  ignore (Rune.jvp_k' outer x thetas);
  (match !inner_seed with
  | Some (Some dy) ->
      check_arr ~msg:"the inner mode answers the tensor it differentiates" (to_arr v)
        dy
  | _ -> fail "the inner forward mode did not answer the tensor it seeds");
  (match !inner_own with
  | Some (Some dy) ->
      equal ~msg:"single-tangent convention" int (Array.length (Nx.shape x))
        (Array.length (Nx.shape dy))
  | _ -> fail "the inner forward mode did not answer its own tensor");
  match !inner_outer with
  | Some None -> ()
  | _ -> fail "the inner mode leaked an enclosing store's tangent"

(* Reads are inert, and a fold's steps are read as they run. *)

let test_querying_does_not_perturb () =
  let x = v3 () in
  let k = 2 in
  let thetas = lane_batch ~k x in
  let f x = Nx.sum (Nx.mul (Nx.sin x) (Nx.exp x)) in
  let g x =
    let y = f x in
    ignore (Rune.tangent y : Nx.float64_t option);
    y
  in
  let y_plain, dy_plain = Rune.jvp_k' f x thetas in
  let y_query, dy_query = Rune.jvp_k' g x thetas in
  check_arr ~msg:"value" (to_arr y_plain) y_query;
  check_arr ~msg:"tangent" (to_arr dy_plain) dy_query

module Single = struct
  type t = Nx.float64_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

let test_queries_inside_a_scan_body () =
  (* Every step of a fold runs under the same handler, so a per-step read sees
     that step's tangent batch. *)
  let x = vec64 [| 0.5; -1.0 |] in
  let k = 2 in
  let thetas = lane_batch ~k x in
  let xs =
    Nx.reshape [| 4; 1 |] (Nx.create f64 [| 4 |] [| 0.2; -0.4; 0.6; 0.1 |])
  in
  let answers = ref [] in
  let f x =
    let c, _ys =
      Rune.scan (module Single)
        ~f:(fun c xi ->
          let c' = Nx.tanh (Nx.add c (Nx.mul_s xi 0.5)) in
          answers := Rune.tangent c' :: !answers;
          (c', c'))
        ~init:x xs
    in
    Nx.sum c
  in
  let _, dy = Rune.jvp_k' f x thetas in
  equal ~msg:"one answer per step" int 4 (List.length !answers);
  List.iter
    (function
      | Some dy ->
          (* The carry is a vector: one dimension plus the lane axis. *)
          equal ~msg:"step tangent carries the lane axis" int 2
            (Array.length (Nx.shape dy))
      | None -> fail "a step's tangent was not answered")
    !answers;
  (* The scalar loss's tangent is the k-vector of lane derivatives. *)
  equal ~msg:"endpoint tangent lanes" int k (Nx.numel dy)

let tests =
  [
    group "the installed forward mode answers"
      [
        test "single tangent under jvp" test_single_tangent_under_jvp;
        test "lane batch under jvp_k" test_lane_batch_under_jvp_k;
        test "innermost mode owns the convention"
          test_innermost_mode_owns_the_convention;
      ];
    group "degradations"
      [
        test "silent outside a forward mode" test_silent_outside_a_forward_mode;
        test "silent under no_grad" test_silent_under_no_grad;
        test "a constant intermediate has no tangent"
          test_constant_intermediate_has_no_tangent;
        test "a detached tensor has no tangent"
          test_detached_tensor_has_no_tangent;
      ];
    group "reads are inert"
      [
        test "querying does not perturb the run" test_querying_does_not_perturb;
        test "queries inside a scan body" test_queries_inside_a_scan_body;
      ];
  ]

let () = run "rune tangent" tests
