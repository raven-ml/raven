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

(* ── Math Function Properties ── *)

(* ── Reduction Properties ── *)

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

(* ── Suite ── *)

let () =
  exit (run "Nx Properties"
    [
      group "Linear Algebra" linalg_props;
      group "Einsum" einsum_props;
    ])
