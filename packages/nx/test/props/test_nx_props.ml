(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Property-based tests for Nx operations.

   Each property verifies an algebraic law or invariant over randomly generated
   tensors. These complement the unit tests which cover edge cases, error
   conditions, and NaN/Inf behavior. *)

open Windtrap
open Test_nx_props_support

(* ── Arithmetic Properties ── *)

(* ── Shape Manipulation Properties ── *)

(* ── Comparison Properties ── *)

(* ── Logical & Bitwise Properties ── *)

(* ── Rounding Properties ── *)

(* ── Sorting Properties ── *)

let sorting_props =
  [
    prop "sort is sorted (f32 1d)" f32_1d (fun x ->
        assume (no_nan x);
        let sorted, _indices = Nx.sort x in
        let values = Nx.to_array sorted in
        equal ~msg:"sorted" (array float_exact)
          (Array.of_list (List.sort Float.compare (Array.to_list values)))
          values);
    prop "sort idempotent (f32 1d)" f32_1d (fun x ->
        assume (no_nan x);
        let s1, _ = Nx.sort x in
        let s2, _ = Nx.sort s1 in
        equal (approx ()) s2 s1);
    prop "sort preserves shape (f32 1d)" f32_1d (fun x ->
        let sorted, _ = Nx.sort x in
        equal (array int) (Nx.shape x) (Nx.shape sorted));
    prop "argsort valid indices (f32 1d)" f32_1d (fun x ->
        let _, indices = Nx.sort x in
        let n = Nx.numel x in
        Array.iter
          (fun i ->
            satisfies ~msg:"index in range" int32
              (fun i -> Int32.to_int i >= 0 && Int32.to_int i < n)
              i)
          (Nx.to_array indices));
    prop "sort preserves elements (i32 1d)" i32_1d (fun x ->
        let sorted, _ = Nx.sort x in
        let a = Array.copy (Nx.to_array x) in
        let b = Array.copy (Nx.to_array sorted) in
        Array.sort Int32.compare a;
        Array.sort Int32.compare b;
        equal (array int32) a b);
  ]

(* ── Math Function Properties ── *)

(* ── Reduction Properties ── *)

let reduction_props =
  [
    prop "sum of ones = numel (f32)" f32_any (fun t ->
        let ones = Nx.ones_like t in
        equal (float 1e-5)
          (Float.of_int (Nx.numel t))
          (Nx.item [] (Nx.sum ones)));
    prop "prod of ones = 1 (f32)" f32_any (fun t ->
        let ones = Nx.ones_like t in
        equal (float 1e-5) 1.0 (Nx.item [] (Nx.prod ones)));
    prop "mean = sum / numel (f32)" f32_any (fun t ->
        assume (Nx.numel t > 0);
        let m = Nx.item [] (Nx.mean t) in
        let s = Nx.item [] (Nx.sum t) in
        let n = Float.of_int (Nx.numel t) in
        equal (float 1e-4) (s /. n) m);
    prop "max >= all elements (f32)" f32_any (fun t ->
        assume (no_nan t && Nx.numel t > 0);
        let mx = Nx.max t in
        is_true @@ all_true (Nx.less_equal t (Nx.broadcast_to (Nx.shape t) mx)));
    prop "min <= all elements (f32)" f32_any (fun t ->
        assume (no_nan t && Nx.numel t > 0);
        let mn = Nx.min t in
        is_true
        @@ all_true (Nx.greater_equal t (Nx.broadcast_to (Nx.shape t) mn)));
    prop "var >= 0 (f32)" f32_any (fun t ->
        assume (Nx.numel t > 0);
        satisfies ~msg:"non-negative" float_exact
          (fun v -> v >= 0.0)
          (Nx.item [] (Nx.var t)));
    prop "sum linearity (f32)" f32_pair (fun (a, b) ->
        let lhs = Nx.item [] (Nx.sum (Nx.add a b)) in
        let rhs = Nx.item [] (Nx.sum a) +. Nx.item [] (Nx.sum b) in
        equal (float 1e-2) rhs lhs);
    prop "cumsum last = sum (f32 1d)" f32_1d (fun t ->
        assume (all_finite t && Nx.numel t > 0);
        let cs = Nx.cumsum t in
        let last = Nx.item [ Nx.numel t - 1 ] cs in
        equal (float 1e-3) (Nx.item [] (Nx.sum t)) last);
  ]

(* ── Linear Algebra Properties ── *)

let linalg_props =
  [
    prop "matmul identity (f64)" square_f64 (fun a ->
        let n = (Nx.shape a).(0) in
        let eye = Nx.eye Nx.float64 n in
        equal (approx ~epsilon:1e-10 ()) a (Nx.matmul a eye));
    prop "transpose matmul (f64)"
      (let gen =
         let open Gen in
         let* a = gen_square_f64 ~max_n:4 in
         let n = (Nx.shape a).(0) in
         let+ b =
           gen_tensor_with_values Nx.float64 (Gen.float_range (-5.) 5.)
             [| n; n |]
         in
         (a, b)
       in
       Gen.with_pp (pp_pair pp_tensor pp_tensor) gen)
      (fun (a, b) ->
        let lhs = Nx.transpose (Nx.matmul a b) in
        let rhs = Nx.matmul (Nx.transpose b) (Nx.transpose a) in
        equal (approx ~epsilon:1e-8 ()) rhs lhs);
    prop "trace = sum diagonal (f64)" square_f64 (fun a ->
        let tr = Nx.item [] (Nx.trace a) in
        equal (float 1e-10) (Nx.item [] (Nx.sum (Nx.diagonal a))) tr);
    prop "inv roundtrip (f64 posdef)" posdef_f64 (fun a ->
        let n = (Nx.shape a).(0) in
        let eye = Nx.eye Nx.float64 n in
        let inv_a = Nx.inv a in
        equal (close ~atol:1e-6 ~rtol:1e-6 ()) eye (Nx.matmul inv_a a));
    prop "qr reconstruction (f64)" square_f64 (fun a ->
        let q, r = Nx.qr a in
        equal (close ~atol:1e-6 ~rtol:1e-6 ()) a (Nx.matmul q r));
    prop "svd reconstruction (f64)" square_f64 (fun a ->
        let u, s, vh = Nx.svd a in
        let n = (Nx.shape a).(0) in
        let s_diag = Nx.mul (Nx.eye Nx.float64 n) (Nx.reshape [| 1; n |] s) in
        let reconstructed = Nx.matmul (Nx.matmul u s_diag) vh in
        equal (close ~atol:1e-6 ~rtol:1e-6 ()) a reconstructed);
    prop "cholesky reconstruction (f64 posdef)" posdef_f64 (fun a ->
        let l = Nx.cholesky a in
        let reconstructed = Nx.matmul l (Nx.transpose l) in
        equal (close ~atol:1e-6 ~rtol:1e-6 ()) a reconstructed);
    prop "det of identity = 1"
      (Gen.with_pp Format.pp_print_int (Gen.int_range 1 6))
      (fun n ->
        let eye = Nx.eye Nx.float64 n in
        equal (float 1e-10) 1.0 (Nx.item [] (Nx.det eye)));
  ]

(* ── Concatenation Properties ── *)

(* ── Indexing Properties ── *)

(* ── Broadcasting Properties ── *)

(* ── Einsum Equivalence Properties ── *)

let einsum_props =
  let mk_f32_matmul_pair =
    let gen =
      let open Gen in
      let* m = int_range 1 6 in
      let* n = int_range 1 6 in
      let* k = int_range 1 6 in
      let+ a = gen_f32 [| m; n |] and+ b = gen_f32 [| n; k |] in
      (a, b)
    in
    Gen.with_pp (pp_pair pp_tensor pp_tensor) gen
  in
  let mk_f32_1d_pair =
    let gen =
      let open Gen in
      let* n = int_range 1 10 in
      let+ a = gen_f32 [| n |] and+ b = gen_f32 [| n |] in
      (a, b)
    in
    Gen.with_pp (pp_pair pp_tensor pp_tensor) gen
  in
  let mk_f32_outer_pair =
    let gen =
      let open Gen in
      let* m = int_range 1 8 in
      let* n = int_range 1 8 in
      let+ a = gen_f32 [| m |] and+ b = gen_f32 [| n |] in
      (a, b)
    in
    Gen.with_pp (pp_pair pp_tensor pp_tensor) gen
  in
  [
    (* einsum matmul = Nx.matmul *)
    prop "einsum ij,jk->ik = matmul" mk_f32_matmul_pair (fun (a, b) ->
        let via_einsum = Nx.einsum "ij,jk->ik" [| a; b |] in
        let via_matmul = Nx.matmul a b in
        equal (close ~atol:1e-4 ~rtol:1e-4 ()) via_matmul via_einsum);
    (* einsum transpose = Nx.transpose *)
    prop "einsum ij->ji = transpose" f32_2d (fun a ->
        let via_einsum = Nx.einsum "ij->ji" [| a |] in
        let via_transpose = Nx.transpose a in
        equal (approx ()) via_transpose via_einsum);
    (* einsum trace = Nx.trace *)
    prop "einsum ii-> = trace" square_f64 (fun a ->
        let via_einsum = Nx.item [] (Nx.einsum "ii->" [| a |]) in
        equal (float 1e-10) (Nx.item [] (Nx.trace a)) via_einsum);
    (* einsum diagonal = Nx.diagonal *)
    prop "einsum ii->i = diagonal" square_f64 (fun a ->
        let via_einsum = Nx.einsum "ii->i" [| a |] in
        let via_diagonal = Nx.diagonal a in
        equal (approx ~epsilon:1e-10 ()) via_diagonal via_einsum);
    (* einsum dot product = sum of elementwise mul *)
    prop "einsum i,i-> = dot" mk_f32_1d_pair (fun (a, b) ->
        let via_einsum = Nx.item [] (Nx.einsum "i,i->" [| a; b |]) in
        equal (float 1e-3) (Nx.item [] (Nx.sum (Nx.mul a b))) via_einsum);
    (* einsum outer product *)
    prop "einsum i,j->ij = outer" mk_f32_outer_pair (fun (a, b) ->
        let via_einsum = Nx.einsum "i,j->ij" [| a; b |] in
        let via_outer =
          Nx.mul
            (Nx.reshape [| Nx.numel a; 1 |] a)
            (Nx.reshape [| 1; Nx.numel b |] b)
        in
        equal (close ~atol:1e-4 ~rtol:1e-4 ()) via_outer via_einsum);
    (* einsum total sum = Nx.sum *)
    prop "einsum ij-> = sum" f32_2d (fun a ->
        let via_einsum = Nx.item [] (Nx.einsum "ij->" [| a |]) in
        equal (float 1e-3) (Nx.item [] (Nx.sum a)) via_einsum);
    (* einsum row sum = sum axis 1 *)
    prop "einsum ij->i = sum axis 1" f32_2d (fun a ->
        let via_einsum = Nx.einsum "ij->i" [| a |] in
        let via_sum = Nx.sum ~axes:[ 1 ] a in
        equal (close ~atol:1e-4 ~rtol:1e-4 ()) via_sum via_einsum);
    (* einsum col sum = sum axis 0 *)
    prop "einsum ij->j = sum axis 0" f32_2d (fun a ->
        let via_einsum = Nx.einsum "ij->j" [| a |] in
        let via_sum = Nx.sum ~axes:[ 0 ] a in
        equal (close ~atol:1e-4 ~rtol:1e-4 ()) via_sum via_einsum);
    (* einsum hadamard = elementwise mul *)
    prop "einsum i,i->i = mul" mk_f32_1d_pair (fun (a, b) ->
        let via_einsum = Nx.einsum "i,i->i" [| a; b |] in
        let via_mul = Nx.mul a b in
        equal (approx ()) via_mul via_einsum);
    (* einsum Frobenius inner product *)
    prop "einsum ij,ij-> = sum(mul)"
      (let gen =
         let open Gen in
         let* shape = gen_shape_2d ~max_dim:5 in
         let+ a = gen_f32 shape and+ b = gen_f32 shape in
         (a, b)
       in
       Gen.with_pp (pp_pair pp_tensor pp_tensor) gen)
      (fun (a, b) ->
        let via_einsum = Nx.item [] (Nx.einsum "ij,ij->" [| a; b |]) in
        equal (float 1e-3) (Nx.item [] (Nx.sum (Nx.mul a b))) via_einsum);
    (* einsum matvec = matmul with reshaped vector *)
    prop "einsum ij,j->i = matvec"
      (let gen =
         let open Gen in
         let* m = int_range 1 6 in
         let* n = int_range 1 6 in
         let+ a = gen_f32 [| m; n |] and+ b = gen_f32 [| n |] in
         (a, b)
       in
       Gen.with_pp (pp_pair pp_tensor pp_tensor) gen)
      (fun (a, b) ->
        let via_einsum = Nx.einsum "ij,j->i" [| a; b |] in
        let via_matmul =
          Nx.reshape
            [| (Nx.shape a).(0) |]
            (Nx.matmul a (Nx.reshape [| Nx.numel b; 1 |] b))
        in
        equal (close ~atol:1e-4 ~rtol:1e-4 ()) via_matmul via_einsum);
  ]

(* ── Stress Tests: Strided Views, Non-Contiguous Ops, High Rank ── *)

let stress_props =
  [
    (* Transpose then slice, verify data integrity *)
    prop ~count:500 "transpose+slice preserves data (f32)" f32_2d_plus (fun t ->
        let tr = Nx.transpose t in
        let spec = List.init (Nx.ndim tr) (fun _ -> Nx.A) in
        let sliced = Nx.slice spec tr in
        equal (approx ()) (Nx.contiguous tr) (Nx.contiguous sliced));
    (* Transpose+slice then flatten vs direct flatten of transpose *)
    prop ~count:500 "transpose+contiguous = contiguous+transpose data (f32)"
      f32_2d_plus (fun t ->
        let a = Nx.to_array (Nx.contiguous (Nx.transpose t)) in
        equal (array float_exact) a
          (Nx.to_array (Nx.transpose t |> Nx.contiguous)));
    (* Slice a non-trivial range after transpose, check item access *)
    prop ~count:500 "item on transposed view (f32)" f32_2d_plus (fun t ->
        let s = Nx.shape t in
        let tr = Nx.transpose t in
        let ts = Nx.shape tr in
        (* item [0, ..., 0] of transpose should equal item [0, ..., 0] of
           original since both index the same element *)
        let zeros_orig = List.init (Array.length s) (fun _ -> 0) in
        let zeros_tr = List.init (Array.length ts) (fun _ -> 0) in
        equal float_exact (Nx.item zeros_orig t) (Nx.item zeros_tr tr));
    (* Flip + slice: flip is a strided view, slicing it compounds strides *)
    prop ~count:500 "flip+slice data integrity (f32)" f32_2d_plus (fun t ->
        let flipped = Nx.flip t in
        let spec = [ Nx.R (0, (Nx.shape flipped).(0)) ] in
        let sliced = Nx.slice spec flipped in
        equal (approx ()) (Nx.contiguous flipped) (Nx.contiguous sliced));
    (* Double transpose on high-rank tensor *)
    prop ~count:500 "double transpose high rank (f32)" f32_stress (fun t ->
        assume (Nx.ndim t >= 2);
        equal (approx ()) t (Nx.transpose (Nx.transpose t)));
    (* Contiguous on strided views: transpose then contiguous should equal copy
       of transpose *)
    prop ~count:500 "contiguous of strided view (f32)" f32_2d_plus (fun t ->
        let tr = Nx.transpose t in
        let c = Nx.contiguous tr in
        is_true ~msg:"contiguous" (Nx.is_c_contiguous c);
        equal (approx ()) tr c);
    (* Arithmetic on non-contiguous views *)
    prop ~count:500 "add on transposed views (f32)" f32_stress_pair
      (fun (a, b) ->
        assume (Nx.ndim a >= 2);
        let at = Nx.transpose a in
        let bt = Nx.transpose b in
        let sum_then_transpose = Nx.transpose (Nx.add a b) in
        let transpose_then_sum = Nx.add at bt in
        equal (approx ()) transpose_then_sum sum_then_transpose);
    (* Reduction on transposed view *)
    prop ~count:500 "sum of transpose = sum of original (f32)" f32_stress
      (fun t ->
        assume (all_finite t);
        let s1 = Nx.item [] (Nx.sum t) in
        equal (float 1e-2) s1 (Nx.item [] (Nx.sum (Nx.transpose t))));
    (* Broadcasting + arithmetic on high-rank tensors *)
    prop ~count:500 "mul broadcast high rank (f32)" f32_broadcastable_stress
      (fun (a, b) ->
        let result = Nx.mul a b in
        let a', b' = Nx.broadcasted a b in
        equal (approx ()) (Nx.mul a' b') result);
    (* Slice with step on high-rank tensor *)
    prop ~count:500 "slice with step roundtrip (f32)" f32_stress (fun t ->
        assume (Nx.ndim t >= 1 && (Nx.shape t).(0) >= 2);
        let dim0 = (Nx.shape t).(0) in
        let sliced = Nx.slice [ Nx.Rs (0, dim0, 2) ] t in
        let expected_len = (dim0 + 1) / 2 in
        equal ~msg:"length" int expected_len (Nx.shape sliced).(0);
        equal ~msg:"rank" int (Nx.ndim t) (Nx.ndim sliced));
    (* Copy of a strided view preserves data *)
    prop ~count:500 "copy strided view (f32)" f32_2d_plus (fun t ->
        let tr = Nx.transpose t in
        let c = Nx.copy tr in
        equal (approx ()) tr c;
        is_true ~msg:"copy is contiguous" (Nx.is_c_contiguous c));
    (* Reshape after contiguous on strided view *)
    prop ~count:500 "reshape contiguous strided (f32)" f32_2d_plus (fun t ->
        let tr = Nx.contiguous (Nx.transpose t) in
        let flat = Nx.reshape [| Nx.numel t |] tr in
        equal ~msg:"numel" int (Nx.numel t) (Nx.numel flat);
        equal (array float_exact) (Nx.to_array tr) (Nx.to_array flat));
  ]

(* ── Suite ── *)

let () =
  exit (run "Nx Properties"
    [
      group "Sorting" sorting_props;
      group "Reductions" reduction_props;
      group "Linear Algebra" linalg_props;
      group "Einsum" einsum_props;
      group "Stress Tests" stress_props;
    ])
