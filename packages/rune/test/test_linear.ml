(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.linearize and Rune.linear_transpose: a linearization applied at any
   tangent is jvp there, without running the function again, from either domain;
   a transpose satisfies the adjoint law <w, L v> = <L^T w, v>, Re Σ
   unconjugated, over every construct a linear function may hold, reads no
   element of its parameters, and sums its contributions as reverse mode does;
   the errors of both. *)

open Windtrap
open Rune2_support
open Witness
module T = Nx.Ptree
module D = Nx.Dtype

let numel = Operand.numel
let f64 = Nx.float64
let vec xs = Nx.create f64 [| Array.length xs |] xs
let pair () = T.(pair tensor tensor)

(* linearize *)

type case =
  | Case :
      (float, 's) Nx.dtype * int array * Operand.t array * Expr.t * Expr.t
      -> case

let draws =
  let open Gen in
  (let* s = Operand.shape in
   let rank = Array.length s in
   let* g = Expr.gen ~rank and+ h = Expr.gen ~rank in
   let* ops = array ~size:(constant 8) (Operand.draw s) in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with Float -> Case (dt, s, ops, g, h) | _ -> assert false)
  |> with_pp (fun ppf (Case (dt, s, ops, g, h)) ->
      Format.fprintf ppf "%a %a, x %a, y %a: %a, %a" Nx.pp_dtype dt
        Operand.pp_shape s Operand.pp_view ops.(0).view Operand.pp_view
        ops.(1).view Expr.pp g Expr.pp h)

(* Three tangents at one point: each is jvp's tangent there, bit for bit, and
   the linearization's primal is the function's. *)
let law (Case (dt, s, ops, g, h)) =
  cover "a zero-size shape" (numel s = 0);
  cover "one element" (numel s = 1);
  cover "a view operand"
    (Array.exists (fun (o : Operand.t) -> o.view <> Plain) ops);
  let mk k = Operand.make dt s ops.(k) in
  let x = (mk 0, mk 1) in
  let f (a, b) = (Expr.eval a b g, Expr.eval a b h) in
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

let at = vec [| 0.5; -1.; 2. |]
let product x = Nx.mul (Nx.sin x) x
let direction k = vec [| k; 1.; -2. |]

let test_once () =
  let calls = ref 0 in
  let f x =
    incr calls;
    product x
  in
  let _, jvp_at = Rune.linearize T.tensor T.tensor f at in
  equal ~msg:"at the call" int 1 !calls;
  ignore (jvp_at (direction 1.));
  ignore (jvp_at (direction 2.));
  equal ~msg:"applied twice" int 1 !calls

let test_ignored () =
  let p = T.(pair tensor tensor) in
  let f (x, k) = Nx.mul x (Nx.cast f64 k) in
  let ints = Nx.create Nx.int32 [| 3 |] [| 3l; -2l; 7l |] in
  let _, jvp_at = Rune.linearize p T.tensor f (at, ints) in
  let t = direction 1. in
  let other = Nx.create Nx.int32 [| 3 |] [| 100l; Int32.min_int; 1l |] in
  equal bits_t (bits (Nx.mul t (Nx.cast f64 ints))) (bits (jvp_at (t, other)))

let test_lanes () =
  let _, jvp_at = Rune.linearize T.tensor T.tensor product at in
  let ts = List.map direction [ 1.; -3.; 0. ] in
  equal bits_t
    (bits (Nx.stack (List.map jvp_at ts)))
    (bits (Rune.vmap' jvp_at (Nx.stack ts)))

(* The identity's tangent is fresh: donating it leaves the argument. *)
let test_fresh () =
  let _, id_at = Rune.linearize T.tensor T.tensor Fun.id at in
  let t = direction 3. in
  ignore (Nx.add (Nx.donate (id_at t)) at);
  equal bits_t (bits (direction 3.)) (bits t)

(* One tangent map applied from two domains gives what it gives on one. *)
let domain_commands =
  let start () = snd (Rune.linearize T.tensor T.tensor product at) in
  let map = abstract "jvp_at" in
  let apply jvp_at k = bits (jvp_at (direction k)) in
  let k = Gen.float_range (-2.) 2. in
  [
    command "start" (Gen.unit @-> makes map) start start;
    command "apply" (map ^-> k @-> returns bits_t) apply apply;
  ]

(* linear_transpose *)

(* Linear functions of [v], composed of linear nx functions with constant
   coefficients and of the constructs a linear function may hold. *)
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
  | Map of lin
  | Remat of lin
  | Call of lin (* A compiled call of [reversed_sums] on [lin]. *)

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
  | Map e -> Format.fprintf ppf "map(%a)" pp_lin e
  | Remat e -> Format.fprintf ppf "remat(%a)" pp_lin e
  | Call e -> Format.fprintf ppf "call(%a)" pp_lin e

let rec holds p e =
  p e
  ||
  match e with
  | V -> false
  | Scale (_, e)
  | Cumsum e
  | Flip e
  | Neg e
  | Conj e
  | Matvec (_, e)
  | Sum e
  | Loop e
  | Map e
  | Remat e
  | Call e ->
      holds p e
  | Add (a, b) -> holds p a || holds p b

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
          (1, one (fun e -> Map e));
          (1, one (fun e -> Remat e));
          (1, one (fun e -> Call e));
        ]
  in
  go 4

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

(* A compiled linear call, one per dtype, so that each shape compiles once
   across cases. *)
let reversed_sums x = Nx.flip (Nx.cumsum ~axis:0 x)

type ('v, 's) host = ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t

let called_f32 : (float, D.float32_elt) host = Rune.jit' reversed_sums
let called_f64 : (float, D.float64_elt) host = Rune.jit' reversed_sums
let called_c64 : (Complex.t, D.complex32_elt) host = Rune.jit' reversed_sums
let called_c128 : (Complex.t, D.complex64_elt) host = Rune.jit' reversed_sums

let called : type v s.
    (v, s) Nx.dtype -> (v, s, Nx.host) Nx.t -> (v, s, Nx.host) Nx.t =
 fun dt x ->
  match dt with
  | Float32 -> called_f32 x
  | Float64 -> called_f64 x
  | Complex64 -> called_c64 x
  | Complex128 -> called_c128 x
  | _ -> invalid_arg "called: not a drawn dtype"

let rec apply : type v s.
    (v, s) Nx.dtype ->
    int ->
    lin ->
    (v, s, Nx.host) Nx.t ->
    (v, s, Nx.host) Nx.t =
 fun dt n e v ->
  let sub e = apply dt n e in
  match e with
  | V -> v
  | Scale (k, e) -> Nx.mul (coefficient dt k [| n |]) (sub e v)
  | Cumsum e -> Nx.cumsum ~axis:0 (sub e v)
  | Flip e -> Nx.flip (sub e v)
  | Neg e -> Nx.neg (sub e v)
  | Conj e -> Nx.conjugate (sub e v)
  | Matvec (k, e) -> Nx.matmul (coefficient dt (k + 8) [| n; n |]) (sub e v)
  | Sum e -> Nx.broadcast_to [| n |] (Nx.sum ~keepdims:true (sub e v))
  | Add (a, b) -> Nx.add (sub a v) (sub b v)
  | Loop e ->
      (* A scan whose step applies [e] to the carry and adds a row times a
         coefficient: linear in [v], the carry's init. *)
      let w = coefficient dt 20 [| n |] in
      fst
        (Rune.scan T.tensor T.tensor T.unit
           ~f:(fun c r -> (Nx.add (sub e c) (Nx.mul w r), ()))
           ~init:v
           (Nx.zeros dt [| 2; n |]))
  | Map e ->
      (* [e] of [v] and of [v] reversed, one lane each, summed. *)
      Nx.sum ~axes:[ 0 ] (Rune.vmap' (sub e) (Nx.stack [ v; Nx.flip v ]))
  | Remat e -> Rune.remat T.(tensor @-> returns tensor) (sub e) v
  | Call e -> called dt (sub e v)

(* [Re Σ a b], unconjugated, and [Σ |a b|]. *)
let pairing (a : ('v, 's, _) Nx.t) b =
  let ab = Nx.cast Nx.complex128 (Nx.mul a b) in
  ( (Nx.item [] (Nx.sum ab)).Complex.re,
    Nx.item [] (Nx.sum (Nx.magnitude f64 ab)) )

type lin_case =
  | Lin : lin * ('v, 's) Nx.dtype * float array * float array -> lin_case

let length (type v s) (dt : (v, s) Nx.dtype) xs =
  match D.kind dt with Complex -> Array.length xs / 2 | _ -> Array.length xs

(* A linear function, a dtype, a length [n] from 0 to 5, and [v]'s and [w]'s
   elements, a complex one's imaginary parts after its real ones. *)
let lin_cases =
  let open Gen in
  (let* e = lin_expr and+ n = int_range 0 5 in
   let* (D.Any dt) =
     of_list [ D.Any Nx.float32; Any f64; Any Nx.complex64; Any Nx.complex128 ]
   in
   let k = match D.kind dt with Complex -> 2 * n | _ -> n in
   let elements = array ~size:(constant k) (float_range (-2.) 2.) in
   let+ v = elements and+ w = elements in
   Lin (e, dt, v, w))
  |> with_pp (fun ppf (Lin (e, dt, v, _)) ->
      Format.fprintf ppf "%a [%d] %a" Nx.pp_dtype dt (length dt v) pp_lin e)

let elements (type v s) (dt : (v, s) Nx.dtype) xs : (v, s, Nx.host) Nx.t =
  match D.kind dt with
  | Complex ->
      let n = Array.length xs / 2 in
      Nx.complex dt ~re:(vec (Array.sub xs 0 n)) ~im:(vec (Array.sub xs n n))
  | _ -> Nx.cast dt (vec xs)

(* The two pairings agree within the magnitude of their terms times [2^-13] at
   32 bits or [2^-40] at 64: the rounding of four nested compositions, each of
   at most five-term sums. *)
let adjoint (Lin (e, dt, vs, ws)) =
  let v = elements dt vs and w = elements dt ws in
  let n = length dt vs in
  let has p = holds p e in
  let complex = match D.kind dt with Complex -> true | _ -> false in
  cover "n = 0" (n = 0);
  cover "n = 1" (n = 1);
  cover "a conjugate at a complex dtype"
    (complex && has (function Conj _ -> true | _ -> false));
  cover "a loop" (has (function Loop _ -> true | _ -> false));
  cover "a map" (has (function Map _ -> true | _ -> false));
  cover "a remat" (has (function Remat _ -> true | _ -> false));
  cover "a compiled call" (has (function Call _ -> true | _ -> false));
  let l = apply dt n e in
  let lt = Rune.linear_transpose T.tensor T.tensor l v in
  let lv, s1 = pairing w (l v) and ltw, s2 = pairing (lt w) v in
  let eps =
    if D.equal dt Nx.float32 || D.equal dt Nx.complex64 then 0x1p-13
    else 0x1p-40
  in
  at_most ~msg:"adjoint" float_exact
    ~than:(eps *. (1. +. s1 +. s2))
    (Float.abs (lv -. ltw))

let test_matrix () =
  let a = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let lt =
    Rune.linear_transpose T.tensor T.tensor (Nx.matmul a) (Nx.zeros f64 [| 3 |])
  in
  equal bits_t (bits (vec [| 9.; 12.; 15. |])) (bits (lt (vec [| 1.; 2. |])))

(* NaN parameters give the zero parameters' transpose: no element is read. *)
let test_unread () =
  let f x = Nx.mul x (vec [| 2.; -1.; 0.5 |]) in
  let w = vec [| 1.; 2.; 3. |] in
  let lt x = Rune.linear_transpose T.tensor T.tensor f x w in
  equal bits_t
    (bits (lt (Nx.zeros f64 [| 3 |])))
    (bits (lt (Nx.full f64 [| 3 |] Float.nan)))

let test_once_at_call () =
  let calls = ref 0 in
  let f x =
    incr calls;
    Nx.mul_s x 2.
  in
  let lt = Rune.linear_transpose T.tensor T.tensor f at in
  equal ~msg:"at the call" int 1 !calls;
  ignore (lt at);
  ignore (lt at);
  equal ~msg:"applied twice" int 1 !calls

let test_affine () =
  let f x = Nx.add (Nx.mul_s x 3.) (Nx.ones_like x) in
  let lt = Rune.linear_transpose T.tensor T.tensor f (vec [| 0.; 0. |]) in
  equal bits_t (bits (vec [| 3.; 30. |])) (bits (lt (vec [| 1.; 10. |])))

let test_integer () =
  let p = T.(pair tensor tensor) in
  let lt =
    Rune.linear_transpose p T.tensor
      (fun (x, _) -> Nx.mul_s x 3.)
      (vec [| 0.; 0. |], Nx.create Nx.int32 [| 2 |] [| 3l; -2l |])
  in
  let _, gk = lt (vec [| 1.; 10. |]) in
  equal (Testable.array Testable.int32) [| 0l; 0l |] (Nx.to_array gk)

(* The identity's transpose is fresh: donating it leaves the cotangent. *)
let test_transpose_fresh () =
  let lt = Rune.linear_transpose T.tensor T.tensor Fun.id at in
  let w = direction 3. in
  ignore (Nx.add (Nx.donate (lt w)) at);
  equal bits_t (bits (direction 3.)) (bits w)

(* Sums of zeros: a single contribution keeps its sign; a broadcast's copies sum
   from [+0.]; an element with no contribution is [+0.]. *)
let signs =
  [
    ("one contribution", Fun.id, vec [| -0.; -0. |], [| -0.; -0. |]);
    ( "a broadcast's copies",
      (fun x -> Nx.broadcast_to [| 2; 2 |] x),
      Nx.full f64 [| 2; 2 |] (-0.),
      [| 0.; 0. |] );
    ("no contribution", Nx.slice [ Nx.R (0, 1) ], vec [| -0. |], [| -0.; 0. |]);
  ]

let test_sign (_, f, w, expected) =
  let lt = Rune.linear_transpose T.tensor T.tensor f (vec [| 1.; 1. |]) in
  equal bits_t (bits (vec expected)) (bits (lt w))

(* Errors *)

let errors =
  (* Each transpose is made here, so that a raise at the call fails the row. *)
  let lt f = Rune.linear_transpose T.tensor T.tensor f (vec [| 0.; 1. |]) in
  let ints = Nx.zeros Nx.int32 [| 3 |] in
  let _, jvp_at = Rune.linearize T.tensor T.tensor product at in
  let halved b =
    Rune.root T.tensor
      ~residual:(fun x -> Nx.sub (Nx.mul_s x 2.) b)
      (fun () -> Nx.div_s b 2.)
  in
  let sin_t = lt Nx.sin and id_t = lt Fun.id and root_t = lt halved in
  [
    ( "linearize of integer parameters",
      "Rune.linearize: the parameters hold no real or complex tensor",
      fun () -> ignore (Rune.linearize T.tensor T.tensor (Nx.cast f64) ints) );
    ( "a tangent of another shape",
      "Rune.linearize: the root: shape [3] in the parameters, [2] in the \
       tangents",
      fun () -> ignore (jvp_at (vec [| 1.; 2. |])) );
    ( "a non-linear function",
      "Rune.linear_transpose: Sin is not linear in the function's argument",
      fun () -> ignore (sin_t (vec [| 1.; 1. |])) );
    ( "a cotangent of another shape",
      "Rune.linear_transpose: the root: shape [2] in the result, [3] in the \
       cotangents",
      fun () -> ignore (id_t (vec [| 1.; 1.; 1. |])) );
    ( "a root",
      "Rune.linear_transpose: a loop has no transpose yet",
      fun () -> ignore (root_t (vec [| 1.; 1. |])) );
  ]

let test_error (_, message, f) =
  match f () with
  | () -> failf "returned"
  | exception Invalid_argument m -> equal string message m

let () =
  exit
    (run "rune2 linear"
       [
         group "linearize"
           [
             prop "a linearization is jvp at each tangent" draws law;
             test "the function runs once, at the call" test_once;
             test "an integer parameter's tangent is ignored" test_ignored;
             test "under a map, one lane per tangent" test_lanes;
             test "a tangent is fresh" test_fresh;
             stateful ~domains:2 ~count:10 ~steps:8
               "a tangent map on two domains is one domain's" domain_commands;
           ];
         group "linear_transpose"
           [
             prop "the adjoint law" lin_cases adjoint;
             test "a matrix's transpose" test_matrix;
             test "no element of the parameters is read" test_unread;
             test "the function runs once, at the call" test_once_at_call;
             test "an affine term is dropped" test_affine;
             test "an integer's cotangent is zero" test_integer;
             test "a transpose is fresh" test_transpose_fresh;
             cases ~name:(fun (n, _, _, _) -> n) "signs of zero" signs test_sign;
           ];
         cases ~name:(fun (n, _, _) -> n) "errors" errors test_error;
       ])
