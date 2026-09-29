(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Control-flow combinators: eager semantics, and composition with grad and
   vmap. *)

open Windtrap
open Rune_test_support.Support

let v4 () = vec64 [| 0.5; -1.2; 2.1; 0.8 |]

(* scan with a running-sum carry is a cumulative sum. *)
let cumsum_scan xs =
  snd
    (Rune.scan'
       ~f:(fun c x ->
         let c = Nx.add c x in
         (c, c))
       ~init:(Nx.scalar f64 0.0) xs)

let test_scan_is_cumsum () =
  check_arr ~msg:"scan cumsum"
    (to_arr (Nx.cumsum (v4 ())))
    (cumsum_scan (v4 ()))

let test_scan_final_carry () =
  let carry, _ =
    Rune.scan' ~f:(fun c x -> (Nx.add c x, c)) ~init:(Nx.scalar f64 0.0) (v4 ())
  in
  check_arr ~msg:"carry" [| 0.5 -. 1.2 +. 2.1 +. 0.8 |] carry

let test_grad_through_scan () =
  (* Gradients through the scan equal gradients through the primitive. *)
  let f xs = Nx.sum (Nx.mul (cumsum_scan xs) (cumsum_scan xs)) in
  let g xs = Nx.sum (Nx.mul (Nx.cumsum xs) (Nx.cumsum xs)) in
  check_arr ~msg:"d scan" (to_arr (Rune.grad' g (v4 ()))) (Rune.grad' f (v4 ()))

let test_vmap_of_scan () =
  let x =
    Nx.create f64 [| 2; 4 |] [| 0.5; -1.2; 2.1; 0.8; 1.7; -0.4; 0.9; 0.2 |]
  in
  let y = Rune.vmap' cumsum_scan x in
  check_arr ~msg:"vmap scan" (to_arr (Nx.cumsum ~axis:1 x)) y

let test_scan_rejects_scalar () =
  raises_match Exn.invalid_arg (fun () ->
      ignore (cumsum_scan (Nx.scalar f64 1.0)))

(* Structured scans: a carry of two tensors, rows of a pair, outputs that are a
   list, and a fold that emits nothing. *)
let test_scan_structures () =
  let xs = (v4 (), Nx.mul_s (v4 ()) 2.0) in
  let (sum, count), ys =
    Rune.scan
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.(list tensor)
      ~f:(fun (sum, count) (a, b) ->
        let sum = Nx.add sum (Nx.add a b) in
        ((sum, Nx.add_s count 1.0), [ sum; a ]))
      ~init:(Nx.scalar f64 0.0, Nx.scalar f64 0.0)
      xs
  in
  check_arr ~msg:"sum" [| 3.0 *. (0.5 -. 1.2 +. 2.1 +. 0.8) |] sum;
  check_arr ~msg:"count" [| 4.0 |] count;
  (match ys with
  | [ sums; rows ] ->
      check_arr ~msg:"sums" (to_arr (Nx.cumsum (Nx.mul_s (v4 ()) 3.0))) sums;
      check_arr ~msg:"rows" (to_arr (v4 ())) rows
  | _ -> fail "the outputs are a list of two tensors");
  let total, () =
    Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
      ~f:(fun c x -> (Nx.add c x, ()))
      ~init:(Nx.scalar f64 0.0) (v4 ())
  in
  check_arr ~msg:"nothing to emit" [| 0.5 -. 1.2 +. 2.1 +. 0.8 |] total

(* A body that returns a carry of another structure than it received raises,
   naming the path where they differ. *)
let grow c x = (c @ [ x ], ())
let carry = Nx.Ptree.(list tensor)

let test_scan_rejects_a_changed_carry () =
  raises
    (Invalid_argument
       "Rune.scan: the root: length 2 in the carry the body returned, length 1 \
        in the carry it received") (fun () ->
      ignore
        (Rune.scan carry Nx.Ptree.tensor Nx.Ptree.unit ~f:grow
           ~init:[ Nx.scalar f64 0.0 ]
           (v4 ())))

let test_staged_scan_rejects_a_changed_carry () =
  let f xs =
    let c, () =
      Rune.scan carry Nx.Ptree.tensor Nx.Ptree.unit ~f:grow
        ~init:[ Nx.scalar f64 0.0 ]
        xs
    in
    List.hd c
  in
  raises
    (Invalid_argument
       "Rune.scan: the root: length 2 in the carry the body returned, length 1 \
        in the carry it received") (fun () -> ignore (Rune.jit' f (v4 ())))

(* Under vmap and jvp, the staged scan's step runs the body inside their rules:
   its error still reaches the scan. *)
let test_staged_scan_under_vmap_and_jvp_rejects_a_changed_carry () =
  let f xs =
    let c, () =
      Rune.scan carry Nx.Ptree.tensor Nx.Ptree.unit ~f:grow
        ~init:[ Nx.scalar f64 0.0 ]
        xs
    in
    List.hd c
  in
  let g xs =
    Rune.vmap'
      (fun dx -> snd (Rune.jvp' f xs dx))
      (Nx.stack ~axis:0 [ v4 (); v4 () ])
  in
  raises
    (Invalid_argument
       "Rune.scan: the root: length 2 in the carry the body returned, length 1 \
        in the carry it received") (fun () -> ignore (Rune.jit' g (v4 ())))

(* Every step's outputs have the first step's visits. *)
(* A carry that changes dtype is named by the scan. The carry is a tensor of
   any dtype, so that the body can change it. *)
module Packed = struct
  type _ t = Nx.packed

  let walk c (Nx.P x) = Nx.P (Nx.Ptree.Walk.tensor c x)
end

let test_scan_rejects_a_changed_dtype () =
  raises
    (Invalid_argument
       "Rune.scan: the root: float32 in the carry the body returned, float64 \
        in the carry it received") (fun () ->
      ignore
        (Rune.scan
           (Nx.Ptree.instantiate (module Packed))
           Nx.Ptree.tensor Nx.Ptree.unit
           ~f:(fun (Nx.P c) _ -> (Nx.P (Nx.cast Nx.float32 c), ()))
           ~init:(Nx.P (Nx.scalar f64 0.0))
           (v4 ())))

let test_scan_rejects_changed_outputs () =
  let f c x =
    let c = Nx.add_s c 1.0 in
    (c, if Nx.item [] c > 1.5 then Some x else None)
  in
  raises
    (Invalid_argument
       "Rune.scan: the root: Some in a step's outputs, None in the first \
        step's outputs") (fun () ->
      ignore
        (Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor
           Nx.Ptree.(option tensor)
           ~f ~init:(Nx.scalar f64 0.0) (v4 ())))

(* Compositions where another transformation claims the scan below any stager:
   reverse-mode must fold eagerly so every step lands on its tape. Each case
   compares against the identical computation over the cumsum primitive. *)

let quad_scan xs = Nx.sum (Nx.mul (cumsum_scan xs) (cumsum_scan xs))
let quad_prim xs = Nx.sum (Nx.mul (Nx.cumsum xs) (Nx.cumsum xs))

let test_vmap_of_grad_through_scan () =
  let x =
    Nx.create f64 [| 2; 4 |] [| 0.5; -1.2; 2.1; 0.8; 1.7; -0.4; 0.9; 0.2 |]
  in
  check_arr ~msg:"per-sample grads"
    (to_arr (Rune.vmap' (Rune.grad' quad_prim) x))
    (Rune.vmap' (Rune.grad' quad_scan) x)

let test_grad_of_grad_through_scan () =
  let second f xs = Rune.grad' (fun xs -> Nx.sum (Rune.grad' f xs)) xs in
  check_arr ~msg:"second order"
    (to_arr (second quad_prim (v4 ())))
    (second quad_scan (v4 ()))

let test_hvp_through_scan () =
  (* Forward-over-reverse. *)
  let v = vec64 [| 1.0; 0.0; -1.0; 0.5 |] in
  check_arr ~msg:"hvp"
    (to_arr (Rune.hvp' quad_prim (v4 ()) v))
    (Rune.hvp' quad_scan (v4 ()) v)

let test_cond_branches () =
  let branch x =
    Rune.cond
      (Nx.greater (Nx.sum x) (Nx.scalar f64 0.0))
      ~then_:(fun () -> Nx.sum (Nx.mul x x))
      ~else_:(fun () -> Nx.sum x)
  in
  check_arr ~msg:"then" [| 0.25 +. 4.0 |] (branch (vec64 [| 0.5; 2.0 |]));
  check_arr ~msg:"else" [| -2.5 |] (branch (vec64 [| -0.5; -2.0 |]))

let test_grad_through_cond () =
  (* The taken branch is what gets differentiated. *)
  let f x =
    Rune.cond
      (Nx.greater (Nx.sum x) (Nx.scalar f64 0.0))
      ~then_:(fun () -> Nx.sum (Nx.mul x x))
      ~else_:(fun () -> Nx.sum x)
  in
  check_arr ~msg:"then grad" [| 1.0; 4.0 |]
    (Rune.grad' f (vec64 [| 0.5; 2.0 |]));
  check_arr ~msg:"else grad" [| 1.0; 1.0 |]
    (Rune.grad' f (vec64 [| -0.5; -2.0 |]))

let test_while_loop () =
  (* Double until the sum exceeds 10: 1.5 -> 3 -> 6 -> 12. *)
  let y =
    Rune.while_loop
      ~cond:(fun c -> Nx.less (Nx.sum c) (Nx.scalar f64 10.0))
      ~body:(fun c -> Nx.mul_s c 2.0)
      (vec64 [| 1.0; 0.5 |])
  in
  check_arr ~msg:"final" [| 8.0; 4.0 |] y

let test_grad_through_while_loop () =
  (* Each x doubles k times before the loop exits; d/dx sum = 2^k. *)
  let f x =
    Nx.sum
      (Rune.while_loop
         ~cond:(fun c -> Nx.less (Nx.sum c) (Nx.scalar f64 10.0))
         ~body:(fun c -> Nx.mul_s c 2.0)
         x)
  in
  check_arr ~msg:"d while" [| 8.0; 8.0 |] (Rune.grad' f (vec64 [| 1.0; 0.5 |]))

(* Exceptions. A handler answers the operation that asked: an exception raised
   while a transformation handles a call reaches the call, where a [try] around
   it catches it and a [Fun.protect] around it runs its finaliser, eagerly and
   compiled. *)

exception Boom

let xs = mat64 2 3 [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |]
let vs = mat64 2 3 [| 1.0; 0.5; -2.0; 0.3; 1.5; -0.7 |]
let x0 = vec64 [| 0.5; -1.2; 2.1 |]
let v0 = vec64 [| 1.0; 0.5; -2.0 |]
let rows = mat64 2 3 [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |]
let seen : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()
let boom = function Boom -> true | _ -> false
let invalid = function Invalid_argument _ -> true | _ -> false
let refused = function Rune.Jit_error _ -> true | _ -> false

(* [guarded ~raises finalised call x] is [call x], or [3 x] when [call] raises
   an exception [raises] accepts; it counts its finaliser's runs in
   [finalised]. *)
let guarded ~raises finalised call x =
  match Fun.protect ~finally:(fun () -> incr finalised) (fun () -> call x) with
  | y -> y
  | exception e when raises e -> Nx.mul_s x 3.0

(* Calls whose code a handler runs, raising [Boom]. *)
let calls =
  [
    ( "remat",
      boom,
      fun x ->
        Rune.remat Nx.Ptree.(tensor @-> returns tensor) (fun _ -> raise Boom) x
    );
    ( "custom_jvp",
      boom,
      fun x ->
        Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor
          ~f:(fun _ -> raise Boom)
          ~jvp:(fun _ _ -> raise Boom)
          x );
    ( "custom_vjp",
      boom,
      fun x ->
        Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor
          ~fwd:(fun _ -> raise Boom)
          ~bwd:(fun () g -> g)
          x );
    ( "scan",
      boom,
      fun x -> fst (Rune.scan' ~f:(fun _ _ -> raise Boom) ~init:x rows) );
  ]

(* The transformations' own errors: an operation with no rule, a value read
   inside a map, a refused operation, a custom rule's tangent of another shape,
   an addition of another shape. *)
let no_rule = ("an operation with no rule", invalid, fun x -> Nx.mod_ x x)

let read_in_a_map =
  ( "a value read in a map",
    invalid,
    fun x ->
      ignore (Nx.to_array x);
      x )

let refused_op =
  ( "an operation jit refuses",
    refused,
    fun x ->
      ignore (Nx.rfft Nx.complex128 x);
      x )

let tangent_shape =
  ( "a custom rule's tangent of another shape",
    invalid,
    fun x ->
      Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor
        ~f:(fun x -> Nx.mul_s x 2.0)
        ~jvp:(fun x dx -> (Nx.mul_s x 2.0, Nx.sum dx))
        x )

let addition_shape =
  ( "an addition of another shape",
    invalid,
    fun x ->
      Rune.Total.add seen (Nx.sum x);
      x )

(* Each transformation of a function [g] of an input, the input, the
   transformation of [guarded]'s [3 x], and the calls to try. Rerun code is the
   recomputation of a remat, under [no_grad] there. *)
let transformations =
  [
    ("eager", (fun g x -> g x), x0, Nx.mul_s x0 3.0, calls);
    ("jit", (fun g x -> Rune.jit' g x), x0, Nx.mul_s x0 3.0, [ refused_op ]);
    ( "grad",
      (fun g x -> Rune.grad' (fun x -> Nx.sum (g x)) x),
      x0,
      Nx.full f64 [| 3 |] 3.0,
      calls @ [ no_rule ] );
    ( "grad of rerun code",
      (fun g x ->
        Rune.grad'
          (fun x ->
            Nx.sum
              (Rune.remat
                 Nx.Ptree.(tensor @-> returns tensor)
                 (fun x -> Nx.mul x (Rune.no_grad (fun () -> g x)))
                 x))
          x),
      x0,
      Nx.mul_s x0 3.0,
      calls );
    ( "jvp",
      (fun g x -> snd (Rune.jvp' g x v0)),
      x0,
      Nx.mul_s v0 3.0,
      calls @ [ no_rule; tangent_shape ] );
    ( "vmap",
      (fun g x -> Rune.vmap' g x),
      xs,
      Nx.mul_s xs 3.0,
      calls @ [ read_in_a_map ] );
    ( "jvp of a map",
      (fun g x -> snd (Rune.jvp' (Rune.vmap' g) x vs)),
      xs,
      Nx.mul_s vs 3.0,
      calls @ [ no_rule; read_in_a_map; tangent_shape ] );
    ( "a total's scope",
      (fun g x ->
        fst
          (Rune.Total.collect seen ~zero:(Nx.zeros f64 [| 3 |]) (fun () -> g x))),
      x0,
      Nx.mul_s x0 3.0,
      calls @ [ addition_shape ] );
  ]

(* An operation a call's code leaves unhandled is that code's error: the call's
   fallback for a missing handler of its own does not run the code again. *)
type _ Effect.t += Unanswered : unit Effect.t

let test_unhandled_inside_a_call () =
  let calls runs =
    let f x =
      incr runs;
      Effect.perform Unanswered;
      x
    in
    [
      ("remat", fun x -> Rune.remat Nx.Ptree.(tensor @-> returns tensor) f x);
      ( "custom_jvp",
        fun x ->
          Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor ~f
            ~jvp:(fun x dx -> (f x, dx))
            x );
      ( "custom_vjp",
        fun x ->
          Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor
            ~fwd:(fun x -> (f x, ()))
            ~bwd:(fun () g -> g)
            x );
      ("scan", fun x -> fst (Rune.scan' ~f:(fun c _ -> (f c, c)) ~init:x rows));
    ]
  in
  let runs = ref 0 in
  List.iter
    (fun (name, call) ->
      runs := 0;
      let g x =
        match call x with
        | y -> y
        | exception Effect.Unhandled Unanswered -> Nx.mul_s x 3.0
      in
      check_arr ~msg:name [| 3.0; 3.0; 3.0 |]
        (Rune.grad' (fun x -> Nx.sum (g x)) x0);
      equal ~msg:(name ^ ": runs") int 1 !runs)
    (calls runs)

let exception_tests =
  List.map
    (fun (name, transform, x, expected, calls) ->
      let check ~compiled (_, raises, call) () =
        let finalised = ref 0 in
        let run = transform (guarded ~raises finalised call) in
        let y = if compiled then Rune.jit' run x else run x in
        check_arr ~msg:"caught at the call" (to_arr expected) y;
        is_true ~msg:"finalised" (!finalised > 0)
      in
      group name
        (List.concat_map
           (fun ((call_name, _, _) as c) ->
             [
               test call_name (check ~compiled:false c);
               test (call_name ^ ", compiled") (check ~compiled:true c);
             ])
           calls))
    transformations

let tests =
  [
    group "scan"
      [
        test "running-sum scan is cumsum" test_scan_is_cumsum;
        test "returns the final carry" test_scan_final_carry;
        test "differentiates like the primitive" test_grad_through_scan;
        test "vectorizes over the batch" test_vmap_of_scan;
        test "rejects a scalar input" test_scan_rejects_scalar;
        test "folds structures" test_scan_structures;
        test "rejects a changed carry" test_scan_rejects_a_changed_carry;
        test "rejects a changed carry under jit"
          test_staged_scan_rejects_a_changed_carry;
        test "a staged scan under vmap and jvp rejects a changed carry"
          test_staged_scan_under_vmap_and_jvp_rejects_a_changed_carry;
        test "rejects changed outputs" test_scan_rejects_changed_outputs;
        test "rejects a carry of another dtype"
          test_scan_rejects_a_changed_dtype;
        test "per-sample gradients (vmap of grad)"
          test_vmap_of_grad_through_scan;
        test "second-order gradients (grad of grad)"
          test_grad_of_grad_through_scan;
        test "hessian-vector product (jvp of grad)" test_hvp_through_scan;
      ];
    group "cond"
      [
        test "selects the branch by predicate" test_cond_branches;
        test "differentiates the taken branch" test_grad_through_cond;
      ];
    group "while_loop"
      [
        test "iterates until the predicate fails" test_while_loop;
        test "differentiates the taken iterations" test_grad_through_while_loop;
      ];
    group "exceptions"
      (exception_tests
      @ [
          test "an operation left unhandled inside a call"
            test_unhandled_inside_a_call;
        ]);
  ]

let () = exit (run "rune control" tests)
