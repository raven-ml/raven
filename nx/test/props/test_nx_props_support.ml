(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generator infrastructure and helpers for Nx property tests. *)

open Windtrap
module Gen = Windtrap.Gen

(* ── Shape Generators ── *)

let gen_shape ~max_ndim ~max_dim =
  let open Gen in
  let* ndim = int_range 1 max_ndim in
  let+ dims = list_size (pure ndim) (int_range 1 max_dim) in
  Array.of_list dims

let gen_shape_2d ~max_dim =
  let open Gen in
  let+ r = int_range 1 max_dim and+ c = int_range 1 max_dim in
  [| r; c |]

(* ── Scalar Value Generators ── *)

let gen_float_safe = Gen.float_range (-10.) 10.
let gen_float_positive = Gen.float_range 0.01 10.
let gen_float_unit = Gen.float_range (-0.999) 0.999
let gen_float_trig = Gen.float_range (-.Float.pi) Float.pi
let gen_float_small = Gen.float_range (-5.) 5.

let gen_int32_safe =
  let open Gen in
  let+ n = int_range (-100) 100 in
  Int32.of_int n

let gen_int32_nonzero =
  Gen.oneof
    [
      Gen.map Int32.of_int (Gen.int_range (-100) (-1));
      Gen.map Int32.of_int (Gen.int_range 1 100);
    ]

(* ── Tensor Generators ── *)

let gen_tensor_with_values (type a b) (dtype : (a, b) Nx.dtype)
    (gen_val : a Gen.t) (shape : int array) =
  let size = Array.fold_left ( * ) 1 shape in
  let open Gen in
  let+ data = list_size (pure size) gen_val in
  Nx.create dtype shape (Array.of_list data)

let gen_f32 shape = gen_tensor_with_values Nx.float32 gen_float_safe shape
let gen_f32_positive shape = gen_tensor_with_values Nx.float32 gen_float_positive shape
let gen_f32_unit shape = gen_tensor_with_values Nx.float32 gen_float_unit shape
let gen_f32_trig shape = gen_tensor_with_values Nx.float32 gen_float_trig shape
let gen_f32_small shape = gen_tensor_with_values Nx.float32 gen_float_small shape
let gen_i32 shape = gen_tensor_with_values Nx.int32 gen_int32_safe shape
let gen_i32_nonzero shape = gen_tensor_with_values Nx.int32 gen_int32_nonzero shape

(* Tensor with random shape *)
let gen_f32_any =
  let open Gen in
  let* shape = gen_shape ~max_ndim:3 ~max_dim:5 in
  gen_f32 shape

let gen_i32_any =
  let open Gen in
  let* shape = gen_shape ~max_ndim:3 ~max_dim:5 in
  gen_i32 shape

(* 2D tensor *)
let gen_f32_2d =
  let open Gen in
  let* shape = gen_shape_2d ~max_dim:5 in
  gen_f32 shape

(* Same-shape pairs *)
let gen_f32_pair =
  let open Gen in
  let* shape = gen_shape ~max_ndim:3 ~max_dim:4 in
  let+ a = gen_f32 shape and+ b = gen_f32 shape in
  (a, b)

let gen_i32_pair =
  let open Gen in
  let* shape = gen_shape ~max_ndim:3 ~max_dim:4 in
  let+ a = gen_i32 shape and+ b = gen_i32 shape in
  (a, b)

let gen_i32_pair_b_nonzero =
  let open Gen in
  let* shape = gen_shape ~max_ndim:3 ~max_dim:4 in
  let+ a = gen_i32 shape and+ b = gen_i32_nonzero shape in
  (a, b)

(* Same-shape triples *)
let gen_f32_triple =
  let open Gen in
  let* shape = gen_shape ~max_ndim:3 ~max_dim:3 in
  let+ a = gen_f32 shape and+ b = gen_f32 shape and+ c = gen_f32 shape in
  (a, b, c)

let gen_i32_triple =
  let open Gen in
  let* shape = gen_shape ~max_ndim:3 ~max_dim:3 in
  let+ a = gen_i32 shape and+ b = gen_i32 shape and+ c = gen_i32 shape in
  (a, b, c)

(* Square matrices (float64 for linalg stability) *)
let gen_square_f64 ~max_n =
  let open Gen in
  let* n = int_range 2 max_n in
  gen_tensor_with_values Nx.float64 (Gen.float_range (-5.) 5.) [| n; n |]

(* Positive definite matrix via A^T A + εI *)
let gen_posdef_f64 ~max_n =
  let open Gen in
  let+ a = gen_square_f64 ~max_n in
  let n = (Nx.shape a).(0) in
  let at = Nx.transpose a in
  let ata = Nx.matmul at a in
  let eps_i = Nx.mul_s (Nx.identity Nx.float64 n) 0.1 in
  Nx.add ata eps_i

(* 1D float32 for sorting *)
let gen_f32_1d =
  let open Gen in
  let* len = int_range 1 20 in
  gen_f32 [| len |]

let gen_i32_1d =
  let open Gen in
  let* len = int_range 1 20 in
  gen_i32 [| len |]

(* ── Testable Wrappers ── *)

let pp_tensor fmt t = Format.fprintf fmt "%s" (Nx.to_string t)

let approx_equal (type b) ?(epsilon = 1e-5) (a : (float, b) Nx.t)
    (b : (float, b) Nx.t) =
  if Nx.shape a <> Nx.shape b then false
  else
    let diff = Nx.sub a b in
    let abs_diff = Nx.abs diff in
    let max_diff = Nx.item [] (Nx.max abs_diff) in
    max_diff < epsilon

let exact_equal (type a b) (x : (a, b) Nx.t) (y : (a, b) Nx.t) =
  Nx.shape x = Nx.shape y && Nx.item [] (Nx.array_equal x y)

let mk_testable_f32 gen =
  Testable.make ~pp:pp_tensor
    ~equal:(fun a b -> approx_equal ~epsilon:1e-5 a b)
    ~gen ()

let mk_testable_f32_tol ~epsilon gen =
  Testable.make ~pp:pp_tensor
    ~equal:(fun a b -> approx_equal ~epsilon a b)
    ~gen ()

let mk_testable_i32 gen =
  Testable.make ~pp:pp_tensor ~equal:exact_equal ~gen ()

let mk_testable_f64 gen =
  Testable.make ~pp:pp_tensor
    ~equal:(fun a b -> approx_equal ~epsilon:1e-10 a b)
    ~gen ()

(* Single tensor testables *)
let f32_any = mk_testable_f32 gen_f32_any
let i32_any = mk_testable_i32 gen_i32_any
let f32_2d = mk_testable_f32 gen_f32_2d

(* Pair testables *)
let pp_pair pp1 pp2 fmt (a, b) =
  Format.fprintf fmt "(%a, %a)" pp1 a pp2 b

let pp_triple pp1 pp2 pp3 fmt (a, b, c) =
  Format.fprintf fmt "(%a, %a, %a)" pp1 a pp2 b pp3 c

let f32_pair =
  Testable.make
    ~pp:(pp_pair pp_tensor pp_tensor)
    ~equal:(fun (a1, b1) (a2, b2) ->
      approx_equal a1 a2 && approx_equal b1 b2)
    ~gen:gen_f32_pair ()

let i32_pair =
  Testable.make
    ~pp:(pp_pair pp_tensor pp_tensor)
    ~equal:(fun (a1, b1) (a2, b2) -> exact_equal a1 a2 && exact_equal b1 b2)
    ~gen:gen_i32_pair ()

let i32_pair_b_nonzero =
  Testable.make
    ~pp:(pp_pair pp_tensor pp_tensor)
    ~equal:(fun (a1, b1) (a2, b2) -> exact_equal a1 a2 && exact_equal b1 b2)
    ~gen:gen_i32_pair_b_nonzero ()

let i32_triple =
  Testable.make
    ~pp:(pp_triple pp_tensor pp_tensor pp_tensor)
    ~equal:(fun (a1, b1, c1) (a2, b2, c2) ->
      exact_equal a1 a2 && exact_equal b1 b2 && exact_equal c1 c2)
    ~gen:gen_i32_triple ()

let f32_1d = mk_testable_f32 gen_f32_1d
let i32_1d = mk_testable_i32 gen_i32_1d

let square_f64 = mk_testable_f64 (gen_square_f64 ~max_n:4)
let posdef_f64 = mk_testable_f64 (gen_posdef_f64 ~max_n:4)

(* ── Indexing Generators ── *)

(* Tensor + valid item indices (one per dimension) *)
let gen_f32_with_index =
  let open Gen in
  let* shape = gen_shape ~max_ndim:3 ~max_dim:5 in
  let* t = gen_f32 shape in
  let ndim = Array.length shape in
  let rec gen_indices i acc =
    if i >= ndim then pure (List.rev acc)
    else
      let* idx = int_range 0 (shape.(i) - 1) in
      gen_indices (i + 1) (idx :: acc)
  in
  let+ indices = gen_indices 0 [] in
  (t, indices)

(* 1D f32 tensor + i32 index tensor with valid indices *)
let gen_f32_1d_with_take_indices =
  let open Gen in
  let* len = int_range 1 10 in
  let* t = gen_f32 [| len |] in
  let* num_indices = int_range 1 8 in
  let+ idx_list =
    list_size (pure num_indices) (map Int32.of_int (int_range 0 (len - 1)))
  in
  let indices = Nx.create Nx.int32 [| num_indices |] (Array.of_list idx_list) in
  (t, indices)

(* Tensor + boolean mask of same shape *)
let gen_f32_with_mask =
  let open Gen in
  let* shape = gen_shape ~max_ndim:3 ~max_dim:4 in
  let size = Array.fold_left ( * ) 1 shape in
  let* t = gen_f32 shape in
  let+ bools = list_size (pure size) bool in
  let mask = Nx.create Nx.bool shape (Array.of_list bools) in
  (t, mask)

(* ── Broadcasting Generators ── *)

(* Generate a broadcastable shape pair.
   Strategy: generate a "result" shape, then for each dim, choose whether
   it comes from a (b gets 1), from b (a gets 1), or both (same value). *)
let gen_broadcastable_shapes =
  let open Gen in
  let* ndim = int_range 1 3 in
  let* dims = list_size (pure ndim) (int_range 1 5) in
  let result_shape = Array.of_list dims in
  let+ choices = list_size (pure ndim) (int_range 0 2) in
  let shape_a = Array.copy result_shape in
  let shape_b = Array.copy result_shape in
  List.iteri
    (fun i choice ->
      match choice with
      | 0 -> shape_b.(i) <- 1 (* a has the dim, b broadcasts *)
      | 1 -> shape_a.(i) <- 1 (* b has the dim, a broadcasts *)
      | _ -> () (* both have the dim *))
    choices;
  (shape_a, shape_b)

(* Two tensors with broadcastable shapes *)
let gen_f32_broadcastable_pair =
  let open Gen in
  let* shape_a, shape_b = gen_broadcastable_shapes in
  let+ a = gen_f32 shape_a and+ b = gen_f32 shape_b in
  (a, b)

(* Tensor with some dims of size 1 + valid broadcast target shape *)
let gen_f32_with_broadcast_shape =
  let open Gen in
  let* ndim = int_range 1 3 in
  let* dims = list_size (pure ndim) (int_range 1 5) in
  let target = Array.of_list dims in
  (* Build source shape: randomly set some dims to 1 *)
  let* which_ones = list_size (pure ndim) bool in
  let source =
    Array.mapi (fun i d -> if List.nth which_ones i then 1 else d) target
  in
  let+ t = gen_f32 source in
  (t, target)

(* ── Indexing & Broadcasting Testables ── *)

let pp_int_list fmt l =
  Format.fprintf fmt "[%s]" (String.concat "; " (List.map string_of_int l))

let pp_int_array fmt a =
  Format.fprintf fmt "[|%s|]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let f32_with_index =
  Testable.make
    ~pp:(pp_pair pp_tensor pp_int_list)
    ~equal:(fun (t1, i1) (t2, i2) -> approx_equal t1 t2 && i1 = i2)
    ~gen:gen_f32_with_index ()

let f32_1d_with_take_indices =
  Testable.make
    ~pp:(pp_pair pp_tensor pp_tensor)
    ~equal:(fun (t1, i1) (t2, i2) -> approx_equal t1 t2 && exact_equal i1 i2)
    ~gen:gen_f32_1d_with_take_indices ()

let f32_with_mask =
  Testable.make
    ~pp:(pp_pair pp_tensor pp_tensor)
    ~equal:(fun (t1, m1) (t2, m2) ->
      approx_equal t1 t2 && Nx.shape m1 = Nx.shape m2)
    ~gen:gen_f32_with_mask ()

let f32_broadcastable_pair =
  Testable.make
    ~pp:(pp_pair pp_tensor pp_tensor)
    ~equal:(fun (a1, b1) (a2, b2) -> approx_equal a1 a2 && approx_equal b1 b2)
    ~gen:gen_f32_broadcastable_pair ()

let f32_with_broadcast_shape =
  Testable.make
    ~pp:(pp_pair pp_tensor pp_int_array)
    ~equal:(fun (t1, s1) (t2, s2) -> approx_equal t1 t2 && s1 = s2)
    ~gen:gen_f32_with_broadcast_shape ()

(* ── Stress-Test Generators ── *)

(* Higher-rank tensors with larger dims for stress testing *)
let gen_f32_stress =
  let open Gen in
  let* shape = gen_shape ~max_ndim:5 ~max_dim:8 in
  gen_f32 shape

let f32_stress = mk_testable_f32 gen_f32_stress

let gen_f32_stress_pair =
  let open Gen in
  let* shape = gen_shape ~max_ndim:4 ~max_dim:6 in
  let+ a = gen_f32 shape and+ b = gen_f32 shape in
  (a, b)

let f32_stress_pair =
  Testable.make
    ~pp:(pp_pair pp_tensor pp_tensor)
    ~equal:(fun (a1, b1) (a2, b2) -> approx_equal a1 a2 && approx_equal b1 b2)
    ~gen:gen_f32_stress_pair ()

(* 2D+ tensor for transpose+slice combos *)
let gen_f32_2d_plus =
  let open Gen in
  let* ndim = int_range 2 4 in
  let* dims = list_size (pure ndim) (int_range 2 6) in
  gen_f32 (Array.of_list dims)

let f32_2d_plus = mk_testable_f32 gen_f32_2d_plus

(* Broadcastable pair with higher ranks *)
let gen_broadcastable_shapes_stress =
  let open Gen in
  let* ndim = int_range 2 5 in
  let* dims = list_size (pure ndim) (int_range 1 6) in
  let result_shape = Array.of_list dims in
  let+ choices = list_size (pure ndim) (int_range 0 2) in
  let shape_a = Array.copy result_shape in
  let shape_b = Array.copy result_shape in
  List.iteri
    (fun i choice ->
      match choice with
      | 0 -> shape_b.(i) <- 1
      | 1 -> shape_a.(i) <- 1
      | _ -> ())
    choices;
  (shape_a, shape_b)

let gen_f32_broadcastable_stress =
  let open Gen in
  let* shape_a, shape_b = gen_broadcastable_shapes_stress in
  let+ a = gen_f32 shape_a and+ b = gen_f32 shape_b in
  (a, b)

let f32_broadcastable_stress =
  Testable.make
    ~pp:(pp_pair pp_tensor pp_tensor)
    ~equal:(fun (a1, b1) (a2, b2) -> approx_equal a1 a2 && approx_equal b1 b2)
    ~gen:gen_f32_broadcastable_stress ()

(* ── Helper Predicates ── *)

let no_nan (type b) (t : (float, b) Nx.t) =
  not (Nx.item [] (Nx.any (Nx.isnan t)))

let all_finite (type b) (t : (float, b) Nx.t) =
  Nx.item [] (Nx.all (Nx.isfinite t))

let all_nonzero_f32 (type b) (t : (float, b) Nx.t) =
  let zeros = Nx.zeros_like t in
  not (Nx.item [] (Nx.any (Nx.equal t zeros)))

let allclose (type b) ?(atol = 1e-5) ?(rtol = 1e-5) (a : (float, b) Nx.t)
    (b : (float, b) Nx.t) =
  if Nx.shape a <> Nx.shape b then false
  else
    let diff = Nx.abs (Nx.sub a b) in
    let tol = Nx.add_s (Nx.mul_s (Nx.abs b) rtol) atol in
    Nx.item [] (Nx.all (Nx.less_equal diff tol))

let all_true (type b) (t : (bool, b) Nx.t) = Nx.item [] (Nx.all t)
