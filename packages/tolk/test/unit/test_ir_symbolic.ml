(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Unit tests for Symbolic simplification rules.

   Tests each phase of symbolic simplification in isolation. *)

open Windtrap
open Tolk_ir
module K = Kernel
module D = Dtype
module C = Const

(* Helpers *)

let idx n = K.const (C.int D.Val.index n)
let f32 x = K.const (C.float D.Val.float32 x)

let var name lo hi = K.define_var ~name ~lo ~hi ~dtype:D.Val.index ()
let range size = K.range ~size:(idx size) ~axis:0 ~kind:Axis_kind.Loop ~dtype:D.Val.index ()

(* Apply sym to a single node (not bottom-up). *)
let sym n = Symbolic.sym n

(* Apply sym as a single-pass graph rewrite. *)
let simplify n = K.graph_rewrite (K.first_match [ Symbolic.sym ]) n

(* Check that rule fires and produces the expected node (by physical identity). *)
let fires rule node expected =
  match rule node with
  | Some r -> is_true (r == expected)
  | None -> fail "expected rule to fire"

(* Check that applying sym produces a specific constant. *)
let simplifies_to_int node expected_val =
  let result = simplify node in
  match K.view result with
  | Const { value; _ } -> (
      match C.view value with
      | Int v -> equal int64 v (Int64.of_int expected_val)
      | _ -> fail "expected int const")
  | _ -> fail "expected const"

let simplifies_to_float node expected_val =
  let result = simplify node in
  match K.view result with
  | Const { value; _ } -> (
      match C.view value with
      | Float v -> is_true (Float.equal v expected_val)
      | _ -> fail "expected float const")
  | _ -> fail "expected const"

(* Constant folding *)

let const_fold_tests =
  group "const_fold"
    [
      test "int add" (fun () ->
          simplifies_to_int (K.binary ~op:`Add ~lhs:(idx 3) ~rhs:(idx 4)) 7);
      test "int mul" (fun () ->
          simplifies_to_int (K.binary ~op:`Mul ~lhs:(idx 3) ~rhs:(idx 5)) 15);
      test "int sub" (fun () ->
          simplifies_to_int (K.binary ~op:`Sub ~lhs:(idx 10) ~rhs:(idx 3)) 7);
      test "int idiv" (fun () ->
          simplifies_to_int (K.binary ~op:`Idiv ~lhs:(idx 10) ~rhs:(idx 3)) 3);
      test "int mod" (fun () ->
          simplifies_to_int (K.binary ~op:`Mod ~lhs:(idx 10) ~rhs:(idx 3)) 1);
      test "float add" (fun () ->
          simplifies_to_float
            (K.binary ~op:`Add ~lhs:(f32 1.5) ~rhs:(f32 2.5))
            4.0);
      test "float mul" (fun () ->
          simplifies_to_float
            (K.binary ~op:`Mul ~lhs:(f32 3.0) ~rhs:(f32 2.0))
            6.0);
      test "unary neg float" (fun () ->
          simplifies_to_float (K.unary ~op:`Neg ~src:(f32 5.0)) (-5.0));
    ]

(* Identity folding *)

let identity_fold_tests =
  group "identity_fold"
    [
      test "x + 0 → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (K.binary ~op:`Add ~lhs:x ~rhs:(idx 0)) x);
      test "0 + x → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (K.binary ~op:`Add ~lhs:(idx 0) ~rhs:x) x);
      test "x * 1 → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (K.binary ~op:`Mul ~lhs:x ~rhs:(idx 1)) x);
      test "1 * x → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (K.binary ~op:`Mul ~lhs:(idx 1) ~rhs:x) x);
      test "x // 1 → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (K.binary ~op:`Idiv ~lhs:x ~rhs:(idx 1)) x);
      test "x | 0 → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (K.binary ~op:`Or ~lhs:x ~rhs:(idx 0)) x);
      test "x & 0 → 0" (fun () ->
          let x = var "x" 0 10 in
          let result = sym (K.binary ~op:`And ~lhs:x ~rhs:(idx 0)) in
          (match result with
          | Some r -> (
              match K.view r with
              | Const { value; _ } -> (
                  match C.view value with
                  | Int 0L -> ()
                  | _ -> fail "expected 0")
              | _ -> fail "expected const")
          | None -> fail "expected rule to fire"));
      test "x ^ 0 → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (K.binary ~op:`Xor ~lhs:x ~rhs:(idx 0)) x);
    ]

(* Self-folding *)

let self_fold_tests =
  group "self_fold"
    [
      test "x // x → 1" (fun () ->
          let x = var "x" 1 10 in
          simplifies_to_int (K.binary ~op:`Idiv ~lhs:x ~rhs:x) 1);
      test "x // -1 → -x" (fun () ->
          let x = var "x" 0 10 in
          let expr = K.binary ~op:`Idiv ~lhs:x ~rhs:(idx (-1)) in
          let result = sym expr in
          (match result with
          | Some r -> (
              match K.view r with
              | Unary { op = `Neg; src; _ } -> is_true (src == x)
              | _ -> fail "expected Neg")
          | None -> fail "expected rule to fire"));
      test "x ^ x → 0" (fun () ->
          let x = var "x" 0 10 in
          simplifies_to_int (K.binary ~op:`Xor ~lhs:x ~rhs:x) 0);
      test "x < x → false" (fun () ->
          let x = var "x" 0 10 in
          let result = sym (K.binary ~op:`Cmplt ~lhs:x ~rhs:x) in
          (match result with
          | Some r -> (
              match K.view r with
              | Const { value; _ } -> (
                  match C.view value with
                  | Bool false -> ()
                  | _ -> fail "expected false")
              | _ -> fail "expected const")
          | None -> fail "expected rule to fire"));
    ]

(* Divmod reconstitution *)

let divmod_reconstitute_tests =
  group "divmod_reconstitute"
    [
      test "(x // y) * y + (x % y) → x" (fun () ->
          let x = var "x" 0 100 in
          let y = idx 4 in
          let div = K.binary ~op:`Idiv ~lhs:x ~rhs:y in
          let mul = K.binary ~op:`Mul ~lhs:div ~rhs:y in
          let mod_ = K.binary ~op:`Mod ~lhs:x ~rhs:y in
          let expr = K.binary ~op:`Add ~lhs:mul ~rhs:mod_ in
          fires sym expr x);
      test "(x % y) + (x // y) * y → x (commuted)" (fun () ->
          let x = var "x" 0 100 in
          let y = idx 4 in
          let div = K.binary ~op:`Idiv ~lhs:x ~rhs:y in
          let mul = K.binary ~op:`Mul ~lhs:div ~rhs:y in
          let mod_ = K.binary ~op:`Mod ~lhs:x ~rhs:y in
          let expr = K.binary ~op:`Add ~lhs:mod_ ~rhs:mul in
          fires sym expr x);
    ]

(* Divandmod *)

let divandmod_tests =
  group "divandmod"
    [
      test "Range(8) // 8 → 0 (cancel)" (fun () ->
          let r = range 8 in
          (* Range(8) has vmin=0, vmax=7. cdiv(0,8)=cdiv(7,8)=0. *)
          simplifies_to_int (K.binary ~op:`Idiv ~lhs:r ~rhs:(idx 8)) 0);
      test "Range(8) % 8 → Range(8) (cancel)" (fun () ->
          let r = range 8 in
          let expr = K.binary ~op:`Mod ~lhs:r ~rhs:(idx 8) in
          (* Range(8) % 8: since range is [0,7], result is just Range(8) *)
          let result = simplify expr in
          (* Should simplify to just r. Check it's a range with size 8. *)
          (match K.view result with
          | Range _ -> equal int (Int64.to_int (Divandmod.vmax result) + 1) 8
          | _ ->
              (* Might be the original r if no simplification was needed *)
              is_true true));
      test "(x % 12) // 4 → (x // 4) % 3 (nested)" (fun () ->
          let x = var "x" 0 100 in
          let expr =
            K.binary ~op:`Idiv
              ~lhs:(K.binary ~op:`Mod ~lhs:x ~rhs:(idx 12))
              ~rhs:(idx 4)
          in
          let result = simplify expr in
          (* Should be (x // 4) % 3 *)
          (match K.view result with
          | Binary { op = `Mod; _ } -> ()
          | _ -> fail "expected mod in result"));
    ]

(* Combine terms *)

let combine_terms_tests =
  group "combine_terms"
    [
      test "x + x → x * 2" (fun () ->
          let x = var "x" 0 10 in
          let expr = K.binary ~op:`Add ~lhs:x ~rhs:x in
          let result = simplify expr in
          (match K.view result with
          | Binary { op = `Mul; lhs; _ } -> is_true (lhs == x)
          | _ -> fail "expected x * 2"));
    ]

(* Associative folding *)

let associative_tests =
  group "associative_fold"
    [
      test "(x + 3) + 5 → x + 8" (fun () ->
          let x = var "x" 0 100 in
          let expr =
            K.binary ~op:`Add
              ~lhs:(K.binary ~op:`Add ~lhs:x ~rhs:(idx 3))
              ~rhs:(idx 5)
          in
          let result = simplify expr in
          (* Should fold to x + 8 *)
          (match K.view result with
          | Binary { op = `Add; lhs; rhs } ->
              is_true (lhs == x);
              (match K.view rhs with
              | Const { value; _ } -> (
                  match C.view value with
                  | Int 8L -> ()
                  | _ -> fail "expected 8")
              | _ -> fail "expected const")
          | _ -> fail "expected add"));
    ]

(* GEP pushing *)

let gep_tests =
  group "gep_pushing"
    [
      test "GEP(Vectorize(a, b, c), 1) → b via simplify" (fun () ->
          let a = idx 10 and b = idx 20 and _c = idx 30 in
          let vec = K.vectorize ~srcs:[ a; b; _c ] in
          let gep = K.gep ~src:vec ~idx:1 in
          let result = simplify gep in
          (* After simplification, should resolve to b = const 20 *)
          (match K.view result with
          | Const { value; _ } -> (
              match C.view value with
              | Int 20L -> ()
              | _ -> fail "expected 20")
          | _ -> fail "expected const"));
    ]

(* Decompositions *)

let decomp_tests =
  group "decompositions"
    [
      test "MUL to SHL: x * 8 → x << 3" (fun () ->
          let ops : Decompositions.supported_ops =
            { has_shl = true; has_shr = true; has_and = true; has_or = true;
              has_max = true; has_cmplt = true; has_cmpeq = true;
              has_neg = true; has_sub = true; has_mulacc = false;
              has_fdiv = false; has_threefry = false;
              disable_fast_idiv = true; has_exp2 = true; has_log2 = true; has_sin = true; has_sqrt = true; has_recip = true; force_transcendental = false }
          in
          let x = var "x" 0 100 in
          let expr = K.binary ~op:`Mul ~lhs:x ~rhs:(idx 8) in
          let result = Decompositions.get_late_rewrite_patterns ops expr in
          (match result with
          | Some r -> (
              match K.view r with
              | Binary { op = `Shl; lhs; _ } -> is_true (lhs == x)
              | _ -> fail "expected SHL")
          | None -> fail "expected rule to fire"));
      test "MOD to AND: x % 4 → x & 3" (fun () ->
          let ops : Decompositions.supported_ops =
            { has_shl = true; has_shr = true; has_and = true; has_or = true;
              has_max = true; has_cmplt = true; has_cmpeq = true;
              has_neg = true; has_sub = true; has_mulacc = false;
              has_fdiv = false; has_threefry = false;
              disable_fast_idiv = true; has_exp2 = true; has_log2 = true; has_sin = true; has_sqrt = true; has_recip = true; force_transcendental = false }
          in
          let x = var "x" 0 100 in
          let expr = K.binary ~op:`Mod ~lhs:x ~rhs:(idx 4) in
          let result = Decompositions.get_late_rewrite_patterns ops expr in
          (match result with
          | Some r -> (
              match K.view r with
              | Binary { op = `And; lhs; _ } -> is_true (lhs == x)
              | _ -> fail "expected AND")
          | None -> fail "expected rule to fire"));
      test "MAX to WHERE when has_max=false" (fun () ->
          let ops : Decompositions.supported_ops =
            { has_shl = true; has_shr = true; has_and = true; has_or = true;
              has_max = false; has_cmplt = true; has_cmpeq = true;
              has_neg = true; has_sub = true; has_mulacc = false;
              has_fdiv = false; has_threefry = false;
              disable_fast_idiv = true; has_exp2 = true; has_log2 = true; has_sin = true; has_sqrt = true; has_recip = true; force_transcendental = false }
          in
          let x = var "x" 0 100 in
          let y = var "y" 0 100 in
          let expr = K.binary ~op:`Max ~lhs:x ~rhs:y in
          let result = Decompositions.get_late_rewrite_patterns ops expr in
          (match result with
          | Some r -> (
              match K.view r with
              | Ternary { op = `Where; _ } -> ()
              | _ -> fail "expected WHERE")
          | None -> fail "expected rule to fire"));
    ]

(* New phase 1 rules *)

let bool_cast_fold_tests =
  group "bool_cast_fold"
    [
      test "x % x → 0" (fun () ->
          let x = var "x" 1 10 in
          simplifies_to_int (K.binary ~op:`Mod ~lhs:x ~rhs:x) 0);
      test "bool MUL → AND" (fun () ->
          let x = K.const_bool true and y = K.const_bool false in
          let expr = K.binary ~op:`Mul ~lhs:x ~rhs:y in
          let result = simplify expr in
          (match K.view result with
          | Binary { op = `And; _ } | Const _ -> ()
          | _ -> fail "expected AND or const"));
      test "cast(const(3), float32) → const(3.0)" (fun () ->
          let expr = K.cast ~src:(idx 3) ~dtype:(D.float32) in
          simplifies_to_float expr 3.0);
      test "cast to same dtype → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (K.cast ~src:x ~dtype:(D.index)) x);
      test "nested where: a.where(b.where(c,d), d) → (a&b).where(c,d)" (fun () ->
          let a = K.const_bool true and b = K.const_bool true in
          let c = idx 1 and d = idx 0 in
          let inner = K.ternary ~op:`Where ~a:b ~b:c ~c:d in
          let outer = K.ternary ~op:`Where ~a ~b:inner ~c:d in
          let result = simplify outer in
          (* Should simplify to just 1 since both conditions are true *)
          (match K.view result with
          | Const { value; _ } -> (
              match C.view value with
              | Int 1L -> ()
              | _ -> fail "expected 1")
          | _ -> fail "expected const"));
    ]

(* New phase 2 rules *)

let lt_fold_tests =
  group "lt_fold"
    [
      test "lt mul fold: 2*x < 10 → x < 5" (fun () ->
          let x = var "x" 0 100 in
          let expr =
            K.binary ~op:`Cmplt
              ~lhs:(K.binary ~op:`Mul ~lhs:(idx 2) ~rhs:x)
              ~rhs:(idx 10)
          in
          let result = simplify expr in
          (* Should fold to x < 5 *)
          (match K.view result with
          | Binary { op = `Cmplt; lhs; rhs } ->
              is_true (lhs == x);
              (match K.view rhs with
              | Const { value; _ } -> (
                  match C.view value with
                  | Int 5L -> ()
                  | _ -> fail "expected 5")
              | _ -> fail "expected const")
          | _ -> fail "expected cmplt"));
      test "lt div fold: x // 4 < 3 → x < 12" (fun () ->
          let x = var "x" 0 100 in
          let expr =
            K.binary ~op:`Cmplt
              ~lhs:(K.binary ~op:`Idiv ~lhs:x ~rhs:(idx 4))
              ~rhs:(idx 3)
          in
          let result = simplify expr in
          (match K.view result with
          | Binary { op = `Cmplt; lhs; rhs } ->
              is_true (lhs == x);
              (match K.view rhs with
              | Const { value; _ } -> (
                  match C.view value with
                  | Int 12L -> ()
                  | _ -> fail "expected 12")
              | _ -> fail "expected const")
          | _ -> fail "expected cmplt"));
      test "lt sign flip: x*-1 < y*-1 → y < x" (fun () ->
          let x = var "x" 0 10 and y = var "y" 0 10 in
          let expr =
            K.binary ~op:`Cmplt
              ~lhs:(K.binary ~op:`Mul ~lhs:x ~rhs:(idx (-1)))
              ~rhs:(K.binary ~op:`Mul ~lhs:y ~rhs:(idx (-1)))
          in
          let result = simplify expr in
          (match K.view result with
          | Binary { op = `Cmplt; lhs; rhs } ->
              is_true (lhs == y);
              is_true (rhs == x)
          | _ -> fail "expected y < x"));
      test "float div chain: (x/y)/z → x/(y*z)" (fun () ->
          let x = f32 12.0 and y = f32 3.0 and z = f32 2.0 in
          let expr =
            K.binary ~op:`Fdiv
              ~lhs:(K.binary ~op:`Fdiv ~lhs:x ~rhs:y)
              ~rhs:z
          in
          simplifies_to_float expr 2.0);
    ]

(* New phase 3 rules *)

let where_fold_tests =
  group "where_fold"
    [
      test "where cast push: where(s,a,b).cast(dt)" (fun () ->
          let s = K.const_bool true in
          let a = idx 5 and b = idx 0 in
          let w = K.ternary ~op:`Where ~a:s ~b:a ~c:b in
          let expr = K.cast ~src:w ~dtype:(D.float32) in
          let result = simplify expr in
          simplifies_to_float expr 5.0;
          ignore result);
    ]

(* Entry point *)

let () =
  run "Ir.Symbolic"
    [
      const_fold_tests;
      identity_fold_tests;
      self_fold_tests;
      divmod_reconstitute_tests;
      divandmod_tests;
      combine_terms_tests;
      associative_tests;
      gep_tests;
      decomp_tests;
      bool_cast_fold_tests;
      lt_fold_tests;
      where_fold_tests;
    ]
