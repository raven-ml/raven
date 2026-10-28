(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.linearize and Rune.linear_transpose: a linearization applied at several
   tangents is jvp at each, and a transpose satisfies the adjoint law <w, L v> =
   <L^T w, v>, Re Σ unconjugated. *)

open Windtrap
open Compose
module Rune = Rune2
module T = Nx.Ptree
module D = Nx.Dtype

let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let pair () = T.(pair tensor tensor)

(* linearize *)

type case =
  | Case :
      (float, 's) Nx.dtype * int array * operand array * expr * expr
      -> case

let cases =
  let open Gen in
  (let* s = shape in
   let rank = Array.length s in
   let* g = expr ~rank and+ h = expr ~rank in
   let* ops = array ~size:(constant 8) (operand s) in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any Nx.float64 ] in
   match D.kind dt with Float -> Case (dt, s, ops, g, h) | _ -> assert false)
  |> with_pp (fun ppf (Case (dt, s, _, g, h)) ->
      Format.fprintf ppf "%a %a: %a, %a" Nx.pp_dtype dt pp_shape s pp g pp h)

(* Three tangents at one point: each is jvp's tangent there, bit for bit, and
   the linearization's primal is the function's. *)
let law (Case (dt, s, ops, g, h)) =
  let mk k = make dt s ops.(k) in
  let x = (mk 0, mk 1) in
  let f (a, b) = (eval a b g, eval a b h) in
  let both (a, b) = (bits a, bits b) in
  let t2 = outcome_t (Testable.pair bits_t bits_t) in
  let lin = outcome (fun () -> Rune.linearize (pair ()) (pair ()) f x) in
  equal ~msg:"primal" t2
    (outcome (fun () ->
         both (fst (Rune.jvp (pair ()) (pair ()) f x (mk 2, mk 3)))))
    (match lin with Value (y, _) -> Value (both y) | Raised m -> Raised m);
  List.iter
    (fun (k, k') ->
      let t = (mk k, mk k') in
      equal ~msg:"tangent" t2
        (outcome (fun () -> both (snd (Rune.jvp (pair ()) (pair ()) f x t))))
        (match lin with
        | Value (_, jvp_at) -> outcome (fun () -> both (jvp_at t))
        | Raised m -> Raised m))
    [ (2, 3); (4, 5); (6, 7) ]

let f64 = Nx.float64
let vec xs = Nx.create f64 [| Array.length xs |] xs

let test_reuse () =
  let x = vec [| 0.5; -1.; 2. |] in
  let calls = ref 0 in
  let f x =
    incr calls;
    Nx.mul (Nx.sin x) x
  in
  let _, jvp_at = Rune.linearize T.tensor T.tensor f x in
  let ts =
    [ vec [| 1.; 0.; 0. |]; vec [| 0.; 2.; 0. |]; vec [| 1.; 1.; 1. |] ]
  in
  List.iter
    (fun t -> equal bits_t (bits (snd (Rune.jvp' f x t))) (bits (jvp_at t)))
    ts;
  (* The record is evaluated, [f] runs once per linearize and once per jvp. *)
  equal ~msg:"runs" int 4 !calls;
  (* Under a map, one lane per tangent. *)
  equal ~msg:"lanes" bits_t
    (bits (Nx.stack (List.map jvp_at ts)))
    (bits (Rune.vmap' jvp_at (Nx.stack ts)));
  (* A tangent is fresh: the identity's is not its argument. *)
  let _, id_at = Rune.linearize T.tensor T.tensor Fun.id x in
  let t = vec [| 3.; 4.; 5. |] in
  ignore (Nx.add (Nx.donate (id_at t)) x);
  equal bits_t (bits (vec [| 3.; 4.; 5. |])) (bits t);
  invalid
    ~sub:
      "Rune.linearize: the root: shape [3] in the parameters, [2] in the \
       tangents" (fun () -> jvp_at (vec [| 1.; 2. |]))

(* linear_transpose *)

(* Linear functions of [v], composed of linear nx functions with constant
   coefficients. *)
type lin =
  | V
  | Scale of int * lin
  | Cumsum of lin
  | Flip of lin
  | Neg of lin
  | Conj of lin
  | Matvec of int * lin
  | Sum of lin
  | Add of lin * lin
  | Loop of lin

let rec pp_lin ppf = function
  | V -> Format.pp_print_string ppf "v"
  | Scale (k, e) -> Format.fprintf ppf "scale%d(%a)" k pp_lin e
  | Cumsum e -> Format.fprintf ppf "cumsum(%a)" pp_lin e
  | Flip e -> Format.fprintf ppf "flip(%a)" pp_lin e
  | Neg e -> Format.fprintf ppf "neg(%a)" pp_lin e
  | Conj e -> Format.fprintf ppf "conj(%a)" pp_lin e
  | Matvec (k, e) -> Format.fprintf ppf "matvec%d(%a)" k pp_lin e
  | Sum e -> Format.fprintf ppf "sum(%a)" pp_lin e
  | Add (a, b) -> Format.fprintf ppf "add(%a, %a)" pp_lin a pp_lin b
  | Loop e -> Format.fprintf ppf "loop(%a)" pp_lin e

let lin_expr =
  let open Gen in
  let rec go d =
    if d = 0 then constant V
    else
      let sub = go (d - 1) in
      let one f = map f sub in
      let indexed f =
        let+ k = int_range 0 7 and+ e = sub in
        f k e
      in
      frequency
        [
          (2, constant V);
          (2, indexed (fun k e -> Scale (k, e)));
          (1, one (fun e -> Cumsum e));
          (1, one (fun e -> Flip e));
          (1, one (fun e -> Neg e));
          (1, one (fun e -> Conj e));
          (1, indexed (fun k e -> Matvec (k, e)));
          (1, one (fun e -> Sum e));
          ( 1,
            let+ a = sub and+ b = sub in
            Add (a, b) );
          (1, one (fun e -> Loop e));
        ]
  in
  go 4

let n = 4

(* Coefficients drawn from fixed keys, at [dt]: a complex one's parts drawn
   apart. *)
let coefficient (type v s) (dt : (v, s) Nx.dtype) k shape : (v, s, Nx.host) Nx.t
    =
  let re = Nx.Rng.normal ~key:(Nx.Rng.key (2 * k)) f64 shape in
  match D.kind dt with
  | Complex ->
      let im = Nx.Rng.normal ~key:(Nx.Rng.key ((2 * k) + 1)) f64 shape in
      Nx.complex dt ~re ~im
  | _ -> Nx.cast dt re

let rec apply : type v s.
    (v, s) Nx.dtype -> lin -> (v, s, Nx.host) Nx.t -> (v, s, Nx.host) Nx.t =
 fun dt e v ->
  match e with
  | V -> v
  | Scale (k, e) -> Nx.mul (coefficient dt k [| n |]) (apply dt e v)
  | Cumsum e -> Nx.cumsum ~axis:0 (apply dt e v)
  | Flip e -> Nx.flip (apply dt e v)
  | Neg e -> Nx.neg (apply dt e v)
  | Conj e -> Nx.conjugate (apply dt e v)
  | Matvec (k, e) ->
      Nx.matmul (coefficient dt (k + 8) [| n; n |]) (apply dt e v)
  | Sum e -> Nx.broadcast_to [| n |] (Nx.sum ~keepdims:true (apply dt e v))
  | Add (a, b) -> Nx.add (apply dt a v) (apply dt b v)
  | Loop e ->
      (* A scan whose step applies [e] to the carry and adds a row times a
         coefficient: linear in [v], the carry's init. *)
      let w = coefficient dt 20 [| n |] in
      fst
        (Rune.scan T.tensor T.tensor T.unit
           ~f:(fun c r -> (Nx.add (apply dt e c) (Nx.mul w r), ()))
           ~init:v
           (Nx.zeros dt [| 2; n |]))

(* [Re Σ a b], unconjugated, and a tolerance of the dtype. *)
let pairing (type v s) (dt : (v, s) Nx.dtype) (a : (v, s, _) Nx.t) b =
  let s = Nx.sum (Nx.mul a b) in
  match D.kind dt with
  | Complex -> (Nx.item [] (Nx.cast Nx.complex128 s)).Complex.re
  | _ -> Nx.item [] (Nx.cast f64 s)

let adjoint (e, D.Any dt) =
  match D.kind dt with
  | Float | Complex ->
      let v = coefficient dt 30 [| n |] and w = coefficient dt 31 [| n |] in
      let l = apply dt e in
      let lt = Rune.linear_transpose T.tensor T.tensor l v in
      let lv = pairing dt w (l v) and ltw = pairing dt (lt w) v in
      let eps =
        if D.equal dt Nx.float32 || D.equal dt Nx.complex64 then 1e-3 else 1e-9
      in
      let scale = Float.max 1. (Float.abs lv) in
      less ~msg:"adjoint" float_exact ~than:(eps *. scale)
        (Float.abs (lv -. ltw))
  | _ -> ()

let lin_cases =
  Gen.pair lin_expr
    (Gen.of_list
       [ D.Any Nx.float32; Any Nx.float64; Any Nx.complex64; Any Nx.complex128 ])
  |> Gen.with_pp (fun ppf (e, D.Any dt) ->
      Format.fprintf ppf "%a %a" Nx.pp_dtype dt pp_lin e)

(* A real function: the transpose of a matrix, against its transpose. *)
let test_real () =
  let a = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let lt =
    Rune.linear_transpose T.tensor T.tensor (Nx.matmul a) (Nx.zeros f64 [| 3 |])
  in
  let w = vec [| 1.; -1. |] in
  equal bits_t (bits (Nx.matmul (Nx.matrix_transpose a) w)) (bits (lt w));
  (* An affine term is dropped, and an integer argument has a zero cotangent. *)
  let p = T.(pair tensor tensor) in
  let f (x, _) = Nx.add (Nx.mul_s x 3.) (Nx.ones_like x) in
  let lt =
    Rune.linear_transpose p T.tensor f
      (vec [| 0.; 0. |], Nx.create Nx.int32 [| 2 |] [| 3l; -2l |])
  in
  let gx, gk = lt (vec [| 1.; 10. |]) in
  equal bits_t (bits (vec [| 3.; 30. |])) (bits gx);
  equal (Testable.array Testable.int32) [| 0l; 0l |] (Nx.to_array gk)

let test_refused () =
  let lt = Rune.linear_transpose T.tensor T.tensor Nx.sin (vec [| 0.; 1. |]) in
  invalid
    ~sub:"Rune.linear_transpose: Sin is not linear in the function's argument"
    (fun () -> lt (vec [| 1.; 1. |]));
  let lt = Rune.linear_transpose T.tensor T.tensor Fun.id (vec [| 0.; 1. |]) in
  invalid
    ~sub:
      "Rune.linear_transpose: the root: shape [2] in the result, [3] in the \
       cotangents" (fun () -> lt (vec [| 1.; 1.; 1. |]))

let () =
  exit
    (run "rune2 linear"
       [
         group "linearize"
           [
             prop "a linearization is jvp at each tangent" cases law;
             test "applied again, under a map, fresh" test_reuse;
           ];
         group "linear_transpose"
           [
             prop "the adjoint law" lin_cases adjoint;
             test "a matrix, an affine term, an integer" test_real;
             test "a non-linear function raises" test_refused;
           ];
       ])
