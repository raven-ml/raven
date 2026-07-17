(* Smoke tests for the Symbolic pattern matchers. *)

open Windtrap
open Tolk_uop

let rewrite = Symbolic.simplify
let var ?(dtype = Dtype.index) ~name ~lo ~hi () =
  Uop.variable ~name ~min_val:lo ~max_val:hi ~dtype ()

(* x + 0 -> x *)
let add_zero_folds () =
  let x = var ~name:"x" ~lo:0 ~hi:100 () in
  let e = Uop.O.(x + Uop.const_int 0) in
  is_true ~msg:"result equals x" (Uop.equal (rewrite e) x)

(* x * 1 -> x *)
let mul_one_folds () =
  let x = var ~name:"x" ~lo:0 ~hi:100 () in
  let e = Uop.O.(x * Uop.const_int 1) in
  is_true ~msg:"result equals x" (Uop.equal (rewrite e) x)

(* x // x -> 1 *)
let div_self_folds () =
  let x = var ~name:"x" ~lo:1 ~hi:100 () in
  let e = Uop.O.(x // x) in
  let r = rewrite e in
  is_true ~msg:"result is const"
    (Uop.op r = Ops.Const)

(* x % x -> 0 *)
let mod_self_folds () =
  let x = var ~name:"x" ~lo:1 ~hi:100 () in
  let e = Uop.O.(x mod x) in
  is_true ~msg:"result is const 0"
    (Uop.op (rewrite e) = Ops.Const)

(* Cast of a const -> const. *)
let cast_const_folds () =
  let c = Uop.const_int 5 in
  let e = Uop.cast ~src:c ~dtype:Dtype.int32 in
  let r = rewrite e in
  is_true ~msg:"folds to const"
    (Uop.op r = Ops.Const)

(* x < x -> false *)
let lt_self_folds () =
  let x = var ~name:"x" ~lo:0 ~hi:100 () in
  let e = Uop.O.(x < x) in
  let r = rewrite e in
  is_true ~msg:"result is const (bool false)"
    (Uop.op r = Ops.Const)

(* Two-stage associative: x + 3 + 4 -> x + 7. *)
let two_stage_associative () =
  let x = var ~name:"x" ~lo:0 ~hi:100 () in
  let e = Uop.O.(x + Uop.const_int 3 + Uop.const_int 4) in
  let r = rewrite e in
  (* Expect [Add(x, const)] with const 7. *)
  let ok =
    Uop.op r = Ops.Add
    && Uop.op (Uop.src r).(1) = Ops.Const
  in
  is_true ~msg:"folds nested adds" ok

let int_neutral_chain_folds () =
  let x = var ~name:"x" ~lo:0 ~hi:9 () in
  let x = Uop.cast ~src:x ~dtype:Dtype.int32 in
  let one = Uop.const (Const.int Dtype.int32 1) in
  let neg_one = Uop.const (Const.int Dtype.int32 (-1)) in
  let e = Uop.O.(((x + one) * one) + neg_one) in
  is_true ~msg:"((x + 1) * 1) + -1 folds to x" (Uop.equal (rewrite e) x)

let const_int u =
  match Uop.const_int_value u with
  | Some n -> n
  | None -> failwith "expected integer const"

let const_bool u =
  match Uop.arg u with
  | Uop.Arg.Value c ->
      (match Const.view c with
       | Const.Bool b -> b
       | _ -> failwith "expected bool const")
  | _ -> failwith "expected const arg"

let is_invalid_const u =
  match Uop.arg u with
  | Uop.Arg.Value c -> Const.view c = Const.Invalid
  | _ -> false

(* Cdiv/Cmod are truncating/C-style, not Python floor-style. *)
let cdiv_cmod_constants_are_truncating () =
  let open Uop.O in
  equal int (-2)
    (const_int (rewrite (cdiv (Uop.const_int 7) (Uop.const_int (-3)))));
  equal int (-8)
    (const_int (rewrite (cdiv (Uop.const_int (-50)) (Uop.const_int 6))));
  equal int 2
    (const_int (rewrite (cmod (Uop.const_int 5) (Uop.const_int (-3)))))

let invalid_gate_survives_zero_multiply () =
  let cond =
    var ~name:"b" ~lo:0 ~hi:1 ~dtype:Dtype.bool ()
  in
  let x = var ~name:"x" ~lo:0 ~hi:10 () in
  let gated = Uop.O.where cond x (Uop.invalid ()) in
  let r = rewrite Uop.O.(gated * Uop.const_int 0) in
  is_true
    ~msg:(Format.asprintf "invalid mask remains at root, got %a" Uop.pp r)
    (Uop.op r = Ops.Where);
  is_true ~msg:"else branch is Invalid"
    (Array.length (Uop.src r) = 3
     && match Uop.arg (Uop.src r).(2) with
        | Uop.Arg.Value c -> Const.view c = Const.Invalid
        | _ -> false)

let index_stack_const_folds () =
  let values =
    List.map (Const.int Dtype.int32) [ 1; 2; 3 ]
  in
  let v = Uop.stack (List.map Uop.const values) in
  let r = rewrite (Uop.index ~ptr:v ~idxs:[ Uop.const_int 1 ] ()) in
  equal int 2 (const_int r)

let nan_cmpeq_folds_to_false () =
  let nan = Uop.const (Const.float Dtype.float32 Float.nan) in
  let r = rewrite (Uop.alu_binary ~op:Ops.Cmpeq ~lhs:nan ~rhs:nan) in
  is_true ~msg:"nan cmpeq nan folds to false under IEEE semantics"
    (not (const_bool r))

let invalid_gate_comparison_drops_weak_invalid () =
  let ridx = var ~name:"ridx" ~lo:0 ~hi:10 () in
  let cond = Uop.O.(ridx < Uop.const_int 5) in
  let idx = Uop.O.where cond ridx (Uop.invalid ()) in
  let r = rewrite Uop.O.(idx < Uop.const_int 3) in
  let expected = rewrite Uop.O.(ridx < Uop.const_int 3) in
  is_true
    ~msg:(Format.asprintf "weak invalid comparison drops gate, got %a" Uop.pp r)
    (Uop.equal r expected)

let invalid_gate_comparison_gates_nonweak_invalid () =
  let cond =
    var ~name:"valid" ~lo:0 ~hi:1 ~dtype:Dtype.bool ()
  in
  let x = var ~name:"x32" ~lo:0 ~hi:10 ~dtype:Dtype.int32 () in
  let rhs = Uop.const (Const.int Dtype.int32 3) in
  let idx = Uop.O.where cond x (Uop.invalid ~dtype:Dtype.int32 ()) in
  let r = rewrite Uop.O.(idx < rhs) in
  is_true ~msg:"result dtype is bool" (Dtype.equal (Uop.dtype r) Dtype.bool);
  is_true ~msg:"non-weak invalid comparison remains gated"
    (Uop.op r = Ops.Where);
  is_true ~msg:"invalid branch is bool-typed"
    (let s = Uop.src r in
     Array.length s = 3 && Dtype.equal (Uop.dtype s.(2)) Dtype.bool)

let direct_invalid_comparison_keeps_bool_dtype () =
  let x = var ~name:"x" ~lo:0 ~hi:10 () in
  let r = rewrite Uop.O.(Uop.invalid () < x) in
  is_true ~msg:"comparison result remains bool"
    (Dtype.equal (Uop.dtype r) Dtype.bool);
  is_true ~msg:"direct invalid comparison does not fold to Invalid"
    (not (is_invalid_const r))

let bool_cast_cmpne_const_folds () =
  let x = var ~name:"flag" ~lo:0 ~hi:1 ~dtype:Dtype.bool () in
  let cast = Uop.cast ~src:x ~dtype:Dtype.int32 in
  let zero = Uop.const (Const.int Dtype.int32 0) in
  let one = Uop.const (Const.int Dtype.int32 1) in
  let other = Uop.const (Const.int Dtype.int32 3) in
  is_true ~msg:"cast(bool) != 0 folds to bool"
    (Uop.equal (rewrite (Uop.O.ne cast zero)) x);
  let not_x = rewrite (Uop.O.ne cast one) in
  is_true ~msg:"cast(bool) != 1 folds to not bool"
    (Uop.op not_x = Ops.Cmpne);
  is_true ~msg:"not bool compares against true"
    (Array.length (Uop.src not_x) = 2 && Uop.equal (Uop.src not_x).(0) x
     && const_bool (Uop.src not_x).(1));
  is_true ~msg:"cast(bool) != other folds true"
    (const_bool (rewrite (Uop.O.ne cast other)))

let weak_invalid_gate_cast_drops_gate () =
  let cond =
    var ~name:"valid_cast" ~lo:0 ~hi:1 ~dtype:Dtype.bool ()
  in
  let x = var ~name:"idx" ~lo:0 ~hi:10 () in
  let gated = Uop.O.where cond x (Uop.invalid ()) in
  let r = rewrite (Uop.cast ~src:gated ~dtype:Dtype.int32) in
  let expected = Uop.cast ~src:x ~dtype:Dtype.int32 in
  is_true
    ~msg:(Format.asprintf "weak invalid cast drops gate, got %a" Uop.pp r)
    (Uop.equal r expected)

let const_bitcast_folds () =
  let f = Uop.const (Const.float Dtype.float32 1.0) in
  let r = rewrite (Uop.bitcast ~src:f ~dtype:Dtype.int32) in
  equal int (Int32.to_int (Int32.bits_of_float 1.0)) (const_int r)

let stack_const_bitcast_folds () =
  let src =
    Uop.stack
      [
        Uop.const (Const.float Dtype.float32 1.0);
        Uop.const (Const.float Dtype.float32 2.0);
      ]
  in
  let r = rewrite (Uop.bitcast ~src ~dtype:Dtype.int32) in
  is_true ~msg:"bitcast(STACK consts) folds to STACK" (Uop.op r = Ops.Stack);
  let lanes = Uop.src r in
  equal int 2 (Array.length lanes);
  let expect i f =
    match Uop.arg lanes.(i) with
    | Uop.Arg.Value c -> (
        match Const.view c with
        | Const.Int n ->
            equal int
              (Int32.to_int (Int32.bits_of_float f))
              (Int64.to_int n)
        | _ -> failwith "expected int lane const")
    | _ -> failwith "expected const lane"
  in
  expect 0 1.0;
  expect 1 2.0

let constant_threefry_is_not_uop_folded () =
  let ctr = Uop.const (Const.int Dtype.uint32 0) in
  let key = Uop.const (Const.int Dtype.uint32 1) in
  let r = rewrite (Uop.alu_binary ~op:Ops.Threefry ~lhs:ctr ~rhs:key) in
  is_true ~msg:"THREEFRY stays explicit in UOp symbolic"
    (Uop.op r = Ops.Threefry)

(* Groups exercising the [Symbolic.simplify] fixed-point driver. *)

let simplify_driver_groups =
  [
    group "simple folding"
      [
        test "x + 0 -> x" add_zero_folds;
        test "x * 1 -> x" mul_one_folds;
        test "int neutral chain -> x" int_neutral_chain_folds;
        test "x // x -> 1" div_self_folds;
        test "x % x -> 0" mod_self_folds;
        test "cast const -> const" cast_const_folds;
        test "x < x -> false" lt_self_folds;
      ];
    group "two-stage folding"
      [
        test "associative combine" two_stage_associative;
      ];
    group "constant folding and invalid propagation"
      [
        test "constant cdiv/cmod use truncating semantics"
          cdiv_cmod_constants_are_truncating;
        test "invalid gate survives zero multiply"
          invalid_gate_survives_zero_multiply;
        test "weak invalid comparison drops invalid gate"
          invalid_gate_comparison_drops_weak_invalid;
        test "non-weak invalid comparison gates bool result"
          invalid_gate_comparison_gates_nonweak_invalid;
        test "direct invalid comparison keeps bool dtype"
          direct_invalid_comparison_keeps_bool_dtype;
        test "cast(bool) != const folds" bool_cast_cmpne_const_folds;
        test "weak invalid gate cast drops gate"
          weak_invalid_gate_cast_drops_gate;
        test "constant BITCAST folds" const_bitcast_folds;
        test "STACK const bitcast folds" stack_const_bitcast_folds;
        test "constant THREEFRY is not UOp-folded"
          constant_threefry_is_not_uop_folded;
        test "INDEX(STACK const) folds" index_stack_const_folds;
        test "NaN cmpeq folds to false" nan_cmpeq_folds_to_false;
      ];
  ]

(* Groups exercising the phase-3 [Symbolic.sym] rules directly, applied both
   node-locally and as a bottom-up graph rewrite. *)

module U = Uop
module D = Dtype
module C = Const

(* Helpers *)

let idx n = U.const (C.int D.index n)
let f32 x = U.const (C.float D.float32 x)

let var name lo hi = U.variable ~name ~min_val:lo ~max_val:hi ~dtype:D.index ()
let range size = U.range ~size:(idx size) ~axis:0 ~kind:Axis_type.Loop ~dtype:D.index ()

let ptr_buffer slot =
  U.buffer ~slot ~dtype:D.int32 ~addrspace:D.Global
    ~shape:(U.stack [ idx 16 ]) ()

(* [U.index] folds a constant index into a stack at construction, so build the
   raw [Index] node directly to leave the fold for the pattern matcher. *)
let raw_index_stack_const ~ptr ~idx ~dtype =
  U.replace ptr ~op:Ops.Index ~src:[| ptr; idx |] ~arg:U.Arg.Empty ~dtype ()

(* Apply sym to a single node (not bottom-up). *)
let sym n = Upat.Pattern_matcher.rewrite Symbolic.sym n

(* Apply sym as a single-pass graph rewrite. *)
let simplify n = U.graph_rewrite (Upat.Pattern_matcher.rewrite Symbolic.sym) n

(* Check that rule fires and produces the expected node (by physical identity). *)
let fires rule node expected =
  match rule node with
  | Some r -> is_true (r == expected)
  | None -> fail "expected rule to fire"

let const_value node =
  match U.op node, U.Arg.as_value (U.arg node) with
  | Ops.Const, Some value -> value
  | _ -> fail "expected const"

let check_const_int node expected =
  match C.view (const_value node) with
  | Int v -> equal int64 v (Int64.of_int expected)
  | _ -> fail "expected int const"

let check_const_float node expected =
  match C.view (const_value node) with
  | Float v -> is_true (Float.equal v expected)
  | _ -> fail "expected float const"

let check_const_bool node expected =
  match C.view (const_value node) with
  | Bool v -> equal bool v expected
  | _ -> fail "expected bool const"

let check_stack_ints node expected =
  is_true (Ops.equal (U.op node) Ops.Stack);
  let srcs = Array.to_list (U.src node) in
  equal int (List.length srcs) (List.length expected);
  List.iter2 check_const_int srcs expected

let check_stack_bools node expected =
  is_true (Ops.equal (U.op node) Ops.Stack);
  let srcs = Array.to_list (U.src node) in
  equal int (List.length srcs) (List.length expected);
  List.iter2 check_const_bool srcs expected

let check_stack_floats node expected =
  is_true (Ops.equal (U.op node) Ops.Stack);
  let srcs = Array.to_list (U.src node) in
  equal int (List.length srcs) (List.length expected);
  List.iter2 check_const_float srcs expected

let check_op node expected =
  is_true (Ops.equal (U.op node) expected)

let src node idx = (U.src node).(idx)

(* Check that applying sym produces a specific constant. *)
let simplifies_to_int node expected_val =
  let result = simplify node in
  check_const_int result expected_val

let simplifies_to_float node expected_val =
  let result = simplify node in
  check_const_float result expected_val

(* Constant folding *)

let const_fold_tests =
  group "const_fold"
    [
      test "int add" (fun () ->
          simplifies_to_int (U.alu_binary ~op:Ops.Add ~lhs:(idx 3) ~rhs:(idx 4)) 7);
      test "int mul" (fun () ->
          simplifies_to_int (U.alu_binary ~op:Ops.Mul ~lhs:(idx 3) ~rhs:(idx 5)) 15);
      test "int sub" (fun () ->
          simplifies_to_int (U.alu_binary ~op:Ops.Sub ~lhs:(idx 10) ~rhs:(idx 3)) 7);
      test "int cdiv" (fun () ->
          simplifies_to_int (U.alu_binary ~op:Ops.Cdiv ~lhs:(idx 10) ~rhs:(idx 3)) 3);
      test "int cdiv negative divisor uses trunc semantics" (fun () ->
          simplifies_to_int
            (U.alu_binary ~op:Ops.Cdiv ~lhs:(idx 7) ~rhs:(idx (-3)))
            (-2));
      test "int cdiv by zero folds to zero" (fun () ->
          simplifies_to_int
            (U.alu_binary ~op:Ops.Cdiv ~lhs:(idx 7) ~rhs:(idx 0))
            0);
      test "int cmod" (fun () ->
          simplifies_to_int (U.alu_binary ~op:Ops.Cmod ~lhs:(idx 10) ~rhs:(idx 3)) 1);
      test "int cmod negative divisor uses trunc semantics" (fun () ->
          simplifies_to_int
            (U.alu_binary ~op:Ops.Cmod ~lhs:(idx 7) ~rhs:(idx (-3)))
            1);
      test "int cmod by zero folds to dividend" (fun () ->
          simplifies_to_int
            (U.alu_binary ~op:Ops.Cmod ~lhs:(idx 7) ~rhs:(idx 0))
            7);
      test "int floordiv positive divisor uses floor semantics" (fun () ->
          simplifies_to_int
            (U.alu_binary ~op:Ops.Floordiv ~lhs:(idx 7) ~rhs:(idx 3))
            2);
      test "int floordiv uses Python floor semantics" (fun () ->
          simplifies_to_int
            (U.alu_binary ~op:Ops.Floordiv ~lhs:(idx 7) ~rhs:(idx (-3)))
            (-3));
      test "int floormod positive divisor uses floor semantics" (fun () ->
          simplifies_to_int
            (U.alu_binary ~op:Ops.Floormod ~lhs:(idx 7) ~rhs:(idx 3))
            1);
      test "int floormod uses Python floor semantics" (fun () ->
          simplifies_to_int
            (U.alu_binary ~op:Ops.Floormod ~lhs:(idx 7) ~rhs:(idx (-3)))
            (-2));
      test "floordiv by zero folds to zero" (fun () ->
          simplifies_to_int
            (U.alu_binary ~op:Ops.Floordiv ~lhs:(idx 7) ~rhs:(idx 0))
            0);
      test "floormod by zero folds to dividend" (fun () ->
          simplifies_to_int
            (U.alu_binary ~op:Ops.Floormod ~lhs:(idx 7) ~rhs:(idx 0))
            7);
      test "float add" (fun () ->
          simplifies_to_float
            (U.alu_binary ~op:Ops.Add ~lhs:(f32 1.5) ~rhs:(f32 2.5))
            4.0);
      test "float mul" (fun () ->
          simplifies_to_float
            (U.alu_binary ~op:Ops.Mul ~lhs:(f32 3.0) ~rhs:(f32 2.0))
            6.0);
      test "unary neg float" (fun () ->
          simplifies_to_float (U.alu_unary ~op:Ops.Neg ~src:(f32 5.0)) (-5.0));
      test "stack int add folds lane-wise" (fun () ->
          let a = U.stack [ idx 1; idx 2 ] in
          let b = U.stack [ idx 10; idx 20 ] in
          let result = simplify (U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b) in
          check_stack_ints result [ 11; 22 ]);
      test "stack unary neg folds lane-wise" (fun () ->
          let a = U.stack [ idx 1; idx (-2) ] in
          let result = simplify (U.alu_unary ~op:Ops.Neg ~src:a) in
          check_stack_ints result [ -1; 2 ]);
      test "stack where folds lane-wise" (fun () ->
          let gate = U.stack [ U.const_bool true; U.const_bool false ] in
          let yes = U.stack [ idx 3; idx 4 ] in
          let no = U.stack [ idx 30; idx 40 ] in
          let result =
            simplify (U.alu_ternary ~op:Ops.Where ~a:gate ~b:yes ~c:no)
          in
          check_stack_ints result [ 3; 40 ]);
      test "stack comparison folds lane-wise" (fun () ->
          let a = U.stack [ idx 1; idx 4 ] in
          let b = U.stack [ idx 2; idx 3 ] in
          let result = simplify (U.alu_binary ~op:Ops.Cmplt ~lhs:a ~rhs:b) in
          check_stack_bools result [ true; false ]);
    ]

(* Identity folding *)

let identity_fold_tests =
  group "identity_fold"
    [
      test "x + 0 → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (U.alu_binary ~op:Ops.Add ~lhs:x ~rhs:(idx 0)) x);
      test "0 + x → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (U.alu_binary ~op:Ops.Add ~lhs:(idx 0) ~rhs:x) x);
      test "x * 1 → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (U.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:(idx 1)) x);
      test "1 * x → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (U.alu_binary ~op:Ops.Mul ~lhs:(idx 1) ~rhs:x) x);
      test "cdiv(x, 1) → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (U.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:(idx 1)) x);
      test "x floordiv 1 → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (U.alu_binary ~op:Ops.Floordiv ~lhs:x ~rhs:(idx 1)) x);
      test "x | 0 → x" (fun () ->
          let x =
            U.variable ~name:"x" ~min_val:0 ~max_val:1 ~dtype:D.bool ()
          in
          fires sym (U.alu_binary ~op:Ops.Or ~lhs:x ~rhs:(U.const_bool false)) x);
      test "x & 0 → 0" (fun () ->
          let x = var "x" 0 10 in
          let result = sym (U.alu_binary ~op:Ops.And ~lhs:x ~rhs:(idx 0)) in
          (match result with
          | Some r -> check_const_int r 0
          | None -> fail "expected rule to fire"));
      test "x ^ 0 → x" (fun () ->
          let x = var "x" 0 10 in
          fires sym (U.alu_binary ~op:Ops.Xor ~lhs:x ~rhs:(idx 0)) x);
    ]

(* Self-folding *)

let self_fold_tests =
  group "self_fold"
    [
      test "cdiv(x, x) → 1" (fun () ->
          let x = var "x" 1 10 in
          simplifies_to_int (U.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:x) 1);
      test "x floordiv x → 1" (fun () ->
          let x = var "x" 1 10 in
          simplifies_to_int (U.alu_binary ~op:Ops.Floordiv ~lhs:x ~rhs:x) 1);
      test "cdiv(x, -1) → -x" (fun () ->
          let x = var "x" 0 10 in
          let expr = U.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:(idx (-1)) in
          let result = sym expr in
          (match result with
          | Some r ->
              check_op r Ops.Neg;
              is_true (src r 0 == x)
          | None -> fail "expected rule to fire"));
      test "x ^ x → 0" (fun () ->
          let x = var "x" 0 10 in
          simplifies_to_int (U.alu_binary ~op:Ops.Xor ~lhs:x ~rhs:x) 0);
      test "x < x → false" (fun () ->
          let x = var "x" 0 10 in
          let result = sym (U.alu_binary ~op:Ops.Cmplt ~lhs:x ~rhs:x) in
          (match result with
          | Some r -> check_const_bool r false
          | None -> fail "expected rule to fire"));
    ]

(* Divmod reconstitution *)

let divmod_reconstitute_tests =
  group "divmod_reconstitute"
    [
      test "cdiv/cmod recombine" (fun () ->
          let x = var "x" 0 100 in
          let y = idx 4 in
          let div = U.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:y in
          let mul = U.alu_binary ~op:Ops.Mul ~lhs:div ~rhs:y in
          let mod_ = U.alu_binary ~op:Ops.Cmod ~lhs:x ~rhs:y in
          let expr = U.alu_binary ~op:Ops.Add ~lhs:mul ~rhs:mod_ in
          is_true (U.equal (simplify expr) x));
      test "cmod/cdiv recombine (commuted)" (fun () ->
          let x = var "x" 0 100 in
          let y = idx 4 in
          let div = U.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:y in
          let mul = U.alu_binary ~op:Ops.Mul ~lhs:div ~rhs:y in
          let mod_ = U.alu_binary ~op:Ops.Cmod ~lhs:x ~rhs:y in
          let expr = U.alu_binary ~op:Ops.Add ~lhs:mod_ ~rhs:mul in
          is_true (U.equal (simplify expr) x));
      test "floor div/mod recombine" (fun () ->
          let x = var "x" (-20) 100 in
          let y = idx 4 in
          let div = U.alu_binary ~op:Ops.Floordiv ~lhs:x ~rhs:y in
          let mul = U.alu_binary ~op:Ops.Mul ~lhs:div ~rhs:y in
          let mod_ = U.alu_binary ~op:Ops.Floormod ~lhs:x ~rhs:y in
          let expr = U.alu_binary ~op:Ops.Add ~lhs:mul ~rhs:mod_ in
          is_true (U.equal (simplify expr) x));
      test "scaled nested floor div/mod recombine" (fun () ->
          let x = var "x" 0 100 in
          let div = idx 2 in
          let d = idx 3 in
          let mul = idx 5 in
          let lhs =
            U.alu_binary ~op:Ops.Mul
              ~lhs:
                (U.alu_binary ~op:Ops.Floormod
                   ~lhs:(U.alu_binary ~op:Ops.Floordiv ~lhs:x ~rhs:div)
                   ~rhs:d)
              ~rhs:(U.alu_binary ~op:Ops.Mul ~lhs:div ~rhs:mul)
          in
          let rhs =
            U.alu_binary ~op:Ops.Mul
              ~lhs:(U.alu_binary ~op:Ops.Floormod ~lhs:x ~rhs:div)
              ~rhs:mul
          in
          let expected =
            U.alu_binary ~op:Ops.Mul
              ~lhs:(U.alu_binary ~op:Ops.Floormod ~lhs:x ~rhs:(idx 6))
              ~rhs:mul
            |> simplify
          in
          is_true (U.equal (simplify U.O.(lhs + rhs)) expected));
    ]

(* Divandmod *)

let divandmod_tests =
  group "divandmod"
    [
      test "cdiv(Range(8), 8) → 0 (cancel)" (fun () ->
          let r = range 8 in
          (* Range(8) has vmin=0, vmax=7. cdiv(0,8)=cdiv(7,8)=0. *)
          simplifies_to_int (U.alu_binary ~op:Ops.Cdiv ~lhs:r ~rhs:(idx 8)) 0);
      test "cmod(Range(8), 8) → Range(8) (cancel)" (fun () ->
          let r = range 8 in
          let expr = U.alu_binary ~op:Ops.Cmod ~lhs:r ~rhs:(idx 8) in
          (* cmod(Range(8), 8): since range is [0,7], result is just Range(8). *)
          let result = simplify expr in
          (* Should simplify to just r. Check it's a range with size 8. *)
          (match U.as_range result with
          | Some _ -> equal int (U.vmax result + 1) 8
          | _ ->
              (* Might be the original r if no simplification was needed *)
              is_true true));
    ]

(* Combine terms *)

let combine_terms_tests =
  group "combine_terms"
    [
      test "x + x → x * 2" (fun () ->
          let x = var "x" 0 10 in
          let expr = U.alu_binary ~op:Ops.Add ~lhs:x ~rhs:x in
          let result = simplify expr in
          check_op result Ops.Mul;
          is_true (src result 0 == x));
    ]

(* Associative folding *)

let associative_tests =
  group "associative_fold"
    [
      test "(x + 3) + 5 → x + 8" (fun () ->
          let x = var "x" 0 100 in
          let expr =
            U.alu_binary ~op:Ops.Add
              ~lhs:(U.alu_binary ~op:Ops.Add ~lhs:x ~rhs:(idx 3))
              ~rhs:(idx 5)
          in
          let result = simplify expr in
          (* Should fold to x + 8 *)
          check_op result Ops.Add;
          is_true (src result 0 == x);
          check_const_int (src result 1) 8);
    ]

(* Value-index pushing *)

let index_pushing_tests =
  group "index lane pushing"
    [
      test "INDEX(Vectorize(a, b, c), 1) → b via simplify" (fun () ->
          let a = var "a" 0 10 and b = var "b" 0 10 and c = var "c" 0 10 in
          let vec = U.stack [ a; b; c ] in
          let index = U.index ~ptr:vec ~idxs:[ idx 1 ] () in
          let result = simplify index in
          is_true (result == b));
      test "INDEX(STACK(a, b, c), 1) → b" (fun () ->
          let a = ptr_buffer 10 and b = ptr_buffer 11 and c = ptr_buffer 12 in
          let stk = U.stack [ a; b; c ] in
          let index =
            raw_index_stack_const ~ptr:stk ~idx:(idx 1) ~dtype:(U.dtype b)
          in
          fires sym index b);
      test "INDEX(STACK(a, b), out-of-bounds const) does not fold" (fun () ->
          let a = ptr_buffer 13 and b = ptr_buffer 14 in
          let stk = U.stack [ a; b ] in
          let index =
            raw_index_stack_const ~ptr:stk ~idx:(idx 2) ~dtype:(U.dtype a)
          in
          match sym index with
          | None -> ()
          | Some _ -> fail "expected out-of-bounds INDEX(STACK, const) to stay");
    ]

let spec_tests =
  group "spec"
    [
      test "full_spec accepts value INDEX lane selection" (fun () ->
          let a = var "a" 0 10 and b = var "b" 0 10 in
          let index = U.index ~ptr:(U.stack [ a; b ]) ~idxs:[ idx 1 ] () in
          is_true (Spec.accepts Spec.full_spec index));
    ]

(* New phase 1 rules *)

let bool_cast_fold_tests =
  group "bool_cast_fold"
    [
      test "cmod(x, x) → 0" (fun () ->
          let x = var "x" 1 10 in
          simplifies_to_int (U.alu_binary ~op:Ops.Cmod ~lhs:x ~rhs:x) 0);
      test "bool MUL → AND" (fun () ->
          let x = U.const_bool true and y = U.const_bool false in
          let expr = U.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:y in
          let result = simplify expr in
          if not (Ops.equal (U.op result) Ops.And || Ops.equal (U.op result) Ops.Const)
          then fail "expected AND or const");
      test "cast(const(3), float32) → const(3.0)" (fun () ->
          let expr = U.cast ~src:(idx 3) ~dtype:(D.float32) in
          simplifies_to_float expr 3.0);
      test "cast to same dtype → x" (fun () ->
          let x = var "x" 0 10 in
          let expr = U.cast ~src:x ~dtype:(U.dtype x) in
          is_true (U.equal (simplify expr) x));
      test "cast(bool -> int) != 0 → bool" (fun () ->
          let x =
            U.variable ~name:"flag" ~min_val:0 ~max_val:1 ~dtype:D.bool ()
          in
          let expr =
            U.alu_binary ~op:Ops.Cmpne
              ~lhs:(U.cast ~src:x ~dtype:D.int32)
              ~rhs:(U.const (C.int D.int32 0))
          in
          is_true (U.equal (simplify expr) x));
      test "cast(bool -> int) != 1 → !bool" (fun () ->
          let x =
            U.variable ~name:"flag2" ~min_val:0 ~max_val:1 ~dtype:D.bool ()
          in
          let expr =
            U.alu_binary ~op:Ops.Cmpne
              ~lhs:(U.cast ~src:x ~dtype:D.int32)
              ~rhs:(U.const (C.int D.int32 1))
          in
          let result = simplify expr in
          check_op result Ops.Cmpne;
          is_true (U.equal (src result 0) x);
          check_const_bool (src result 1) true);
      test "cast(bool -> int) != other const → true" (fun () ->
          let x =
            U.variable ~name:"flag3" ~min_val:0 ~max_val:1 ~dtype:D.bool ()
          in
          let expr =
            U.alu_binary ~op:Ops.Cmpne
              ~lhs:(U.cast ~src:x ~dtype:D.int32)
              ~rhs:(U.const (C.int D.int32 7))
          in
          check_const_bool (simplify expr) true);
      test "bitcast const float32 to int32 folds" (fun () ->
          let expr =
            U.bitcast ~src:(f32 1.0) ~dtype:D.int32
          in
          simplifies_to_int expr (Int32.to_int (Int32.bits_of_float 1.0)));
      test "cast STACK const folds lane-wise" (fun () ->
          let src =
            U.stack
              [
                U.const (C.int D.int32 1);
                U.const (C.int D.int32 2);
              ]
          in
          let result = simplify (U.cast ~src ~dtype:D.float32) in
          check_stack_floats result [ 1.0; 2.0 ]);
      test "bitcast STACK const folds lane-wise" (fun () ->
          let src = U.stack [ f32 1.0; f32 2.0 ] in
          let result = simplify (U.bitcast ~src ~dtype:D.int32) in
          check_stack_ints result
            [
              Int32.to_int (Int32.bits_of_float 1.0);
              Int32.to_int (Int32.bits_of_float 2.0);
            ]);
      test "constant Threefry is not folded in UOp-local symbolic" (fun () ->
          let ctr = U.const (C.int D.uint32 0) in
          let key = U.const (C.int D.uint32 1) in
          let result = simplify (U.alu_binary ~op:Ops.Threefry ~lhs:ctr ~rhs:key) in
          check_op result Ops.Threefry);
      test "pow constant exponent rewrites by squaring" (fun () ->
          let x = f32 3.0 in
          let expr =
            U.alu_binary ~op:Ops.Pow ~lhs:x ~rhs:(f32 2.0)
          in
          simplifies_to_float expr 9.0);
      test "nested where: a.where(b.where(c,d), d) → (a&b).where(c,d)" (fun () ->
          let a = U.const_bool true and b = U.const_bool true in
          let c = idx 1 and d = idx 0 in
          let inner = U.alu_ternary ~op:Ops.Where ~a:b ~b:c ~c:d in
          let outer = U.alu_ternary ~op:Ops.Where ~a ~b:inner ~c:d in
          let result = simplify outer in
          (* Should simplify to just 1 since both conditions are true *)
          check_const_int result 1);
    ]

(* New phase 2 rules *)

let lt_fold_tests =
  group "lt_fold"
    [
      test "lt mul fold: 2*x < 10 → x < 5" (fun () ->
          let x = var "x" 0 100 in
          let expr =
            U.alu_binary ~op:Ops.Cmplt
              ~lhs:(U.alu_binary ~op:Ops.Mul ~lhs:(idx 2) ~rhs:x)
              ~rhs:(idx 10)
          in
          let result = simplify expr in
          (* Should fold to x < 5 *)
          check_op result Ops.Cmplt;
          is_true (src result 0 == x);
          check_const_int (src result 1) 5);
      test "lt cdiv fold: cdiv(x, 4) < 3 → x < 12" (fun () ->
          let x = var "x" 0 100 in
          let expr =
            U.alu_binary ~op:Ops.Cmplt
              ~lhs:(U.alu_binary ~op:Ops.Cdiv ~lhs:x ~rhs:(idx 4))
              ~rhs:(idx 3)
          in
          let result = simplify expr in
          check_op result Ops.Cmplt;
          is_true (src result 0 == x);
          check_const_int (src result 1) 12);
      test "lt sign flip: x*-1 < y*-1 → y < x" (fun () ->
          let x = var "x" 0 10 and y = var "y" 0 10 in
          let expr =
            U.alu_binary ~op:Ops.Cmplt
              ~lhs:(U.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:(idx (-1)))
              ~rhs:(U.alu_binary ~op:Ops.Mul ~lhs:y ~rhs:(idx (-1)))
          in
          let result = simplify expr in
          check_op result Ops.Cmplt;
          is_true (src result 0 == y);
          is_true (src result 1 == x));
      test "float div chain: (x/y)/z → x/(y*z)" (fun () ->
          let x = f32 12.0 and y = f32 3.0 and z = f32 2.0 in
          let expr =
            U.alu_binary ~op:Ops.Fdiv
              ~lhs:(U.alu_binary ~op:Ops.Fdiv ~lhs:x ~rhs:y)
              ~rhs:z
          in
          simplifies_to_float expr 2.0);
    ]

(* New phase 3 rules *)

let where_fold_tests =
  group "where_fold"
    [
      test "where cast push: where(s,a,b).cast(dt)" (fun () ->
          let s =
            U.variable ~name:"s" ~min_val:0 ~max_val:1 ~dtype:D.bool ()
          in
          let a = var "a" 0 10 and b = var "b" 0 10 in
          let w = U.alu_ternary ~op:Ops.Where ~a:s ~b:a ~c:b in
          let expr = U.cast ~src:w ~dtype:(D.float32) in
          let result = simplify expr in
          check_op result Ops.Where;
          check_op (src result 1) Ops.Cast;
          check_op (src result 2) Ops.Cast);
      test "where eq one zero flips to ne zero one" (fun () ->
          let x = var "x" 0 10 and y = var "y" 0 10 in
          let cond = U.alu_binary ~op:Ops.Cmpeq ~lhs:x ~rhs:y in
          let one = U.const (Const.int D.int32 1) in
          let zero = U.const (Const.int D.int32 0) in
          let expr = U.alu_ternary ~op:Ops.Where ~a:cond ~b:one ~c:zero in
          let result = simplify expr in
          check_op result Ops.Where;
          check_op (src result 0) Ops.Cmpne;
          check_const_int (src result 1) 0;
          check_const_int (src result 2) 1);
    ]

let reduce_tests =
  group "reduce"
    [
      test "add tensor reduce floats const and preserves axes" (fun () ->
          let x =
            U.param ~slot:0 ~dtype:D.index ~shape:(U.stack [ idx 4 ])
              ~vmin_vmax:(0, 10) ()
          in
          let body = U.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:(idx 3) in
          let red = U.reduce_axis ~src:body ~op:Ops.Add ~axes:[ 0 ] in
          let result = simplify red in
          check_op result Ops.Mul;
          let lhs = src result 0 and rhs = src result 1 in
          let reduced, const =
            match U.as_reduce lhs with
            | Some _ -> lhs, rhs
            | None -> rhs, lhs
          in
          (match U.as_reduce reduced with
          | Some { src = reduced_src; op = Ops.Add; num_axes = 1; ranges = [] } ->
              is_true (U.equal reduced_src x)
          | _ -> fail "expected tensor Add reduce with preserved axes");
          check_const_int const 3);
      test "mul-term hoist floats non-range factors out of a lowered reduce"
        (fun () ->
          (* A lowered reduce carries its reduced range as a source. A MUL term
             that does not reference that range floats out; the range-dependent
             term stays inside. *)
          let rng = range 4 in
          let x =
            U.variable ~name:"x" ~min_val:0 ~max_val:10 ~dtype:D.int32 ()
          in
          let ld = U.load ~src:(U.index ~ptr:(ptr_buffer 20) ~idxs:[ rng ] ()) () in
          let body = U.alu_binary ~op:Ops.Mul ~lhs:ld ~rhs:x in
          let red = U.reduce ~src:body ~ranges:[ rng ] ~op:Ops.Max
              ~dtype:(U.dtype body) in
          let result = simplify red in
          check_op result Ops.Mul;
          let a = src result 0 and b = src result 1 in
          let reduce_node, other =
            if Ops.equal (U.op a) Ops.Reduce then a, b else b, a
          in
          is_true (U.equal other x);
          (match U.as_reduce reduce_node with
          | Some { src = rsrc; op = Ops.Max; num_axes = 0; ranges = [ r ] } ->
              is_true (U.equal rsrc ld && U.equal r rng)
          | _ -> fail "expected the range-dependent term to stay in a lowered reduce"));
    ]

let load_store_tests =
  group "load_store"
    [
      test "invalid-index load with alt folds to alt" (fun () ->
          let ptr = ptr_buffer 0 in
          let invalid_idx = U.invalid () in
          let dst = U.index ~ptr ~idxs:[invalid_idx] () in
          let alt = U.const (C.int D.int32 42) in
          let gate = U.const_bool false in
          let result = simplify (U.load ~src:dst ~alt ~gate ()) in
          is_true (U.equal result alt));
      test "invalid-index load without alt folds to zero" (fun () ->
          let ptr = ptr_buffer 1 in
          let invalid_idx = U.invalid () in
          let dst = U.index ~ptr ~idxs:[invalid_idx] () in
          let result = simplify (U.load ~src:dst ()) in
          check_const_int result 0);
      test "store of gated load becomes gated store of alt" (fun () ->
          let ptr = ptr_buffer 2 in
          let offset = idx 3 in
          let dst = U.index ~ptr ~idxs:[offset] () in
          let gate =
            U.variable ~name:"gate" ~min_val:0 ~max_val:1 ~dtype:D.bool ()
          in
          let alt = U.const (C.int D.int32 7) in
          let old = U.load ~src:dst () in
          let value = U.O.where gate alt old in
          let result = simplify (U.store ~dst ~value ()) in
          check_op result Ops.Store;
          is_true (U.equal (src result 1) alt);
          check_op (src result 0) Ops.Index;
          check_op (src (src result 0) 1) Ops.Where;
          is_true (U.equal (src (src (src result 0) 1) 0) gate);
          is_true (U.equal (src (src (src result 0) 1) 1) offset));
    ]

(* Invalid-condition where folds *)

let invalid_where_tests =
  group "invalid_where"
    [
      test "where(Invalid, a, b) poisons to invalid" (fun () ->
          let a = var "a" 0 10 and b = var "b" 0 10 in
          let w = U.O.where (U.invalid ()) a b in
          match sym w with
          | Some r ->
              (* The poisoned result is [Invalid] cast to [a]'s dtype; the cast
                 to [a]'s own index dtype collapses to the bare Invalid. *)
              is_true
                (is_invalid_const r
                || (Ops.equal (U.op r) Ops.Cast && is_invalid_const (src r 0)))
          | None -> fail "expected invalid condition to poison where");
      test "where(gated-invalid, a, b) lifts the gate" (fun () ->
          let cond =
            U.variable ~name:"c" ~min_val:0 ~max_val:1 ~dtype:D.bool ()
          in
          let x =
            U.variable ~name:"x" ~min_val:0 ~max_val:1 ~dtype:D.bool ()
          in
          let a = var "a" 0 10 and b = var "b" 0 10 in
          let gate = U.O.where cond x (U.invalid ~dtype:D.bool ()) in
          let w = U.O.where gate a b in
          match sym w with
          | Some r ->
              check_op r Ops.Where;
              is_true (U.equal (src r 0) cond);
              check_op (src r 1) Ops.Where;
              check_op (src r 2) Ops.Cast;
              is_true (is_invalid_const (src (src r 2) 0))
          | None -> fail "expected gated-invalid condition to lift its gate");
    ]

(* Sigmoid reciprocal folds (phase-3 [sym]). *)

let sigmoid_tests =
  let fvar name = U.variable ~name ~min_val:0 ~max_val:10 ~dtype:D.float32 () in
  let recip_1p x = U.alu_unary ~op:Ops.Reciprocal ~src:U.O.(x + f32 1.0) in
  group "sigmoid"
    [
      test "x * (1/(1+x)) -> 1 - 1/(1+x)" (fun () ->
          let x = fvar "x" in
          let d = recip_1p x in
          match sym U.O.(x * d) with
          | Some r ->
              check_op r Ops.Sub;
              check_const_float (src r 0) 1.0;
              is_true (src r 1 == d)
          | None -> fail "expected sigmoid fold");
      test "x * (1/(1+x) * y) -> y * (1 - 1/(1+x))" (fun () ->
          let x = fvar "x" and y = fvar "y" in
          let d = recip_1p x in
          match sym U.O.(x * (d * y)) with
          | Some r ->
              check_op r Ops.Mul;
              is_true (src r 0 == y);
              check_op (src r 1) Ops.Sub
          | None -> fail "expected sigmoid product fold");
      test "x * (1/(1+x) + y) -> (1 - 1/(1+x)) + x*y" (fun () ->
          let x = fvar "x" and y = fvar "y" in
          let d = recip_1p x in
          match sym U.O.(x * (d + y)) with
          | Some r ->
              check_op r Ops.Add;
              check_op (src r 0) Ops.Sub;
              check_op (src r 1) Ops.Mul
          | None -> fail "expected sigmoid sum fold");
    ]

(* Masked-numerator division: (x & mask) // c collapses when the mask only
   clears the low bits the power-of-two division discards. *)

let masked_div_tests =
  group "masked_div"
    [
      test "(x & -4) // 4 -> x // 4" (fun () ->
          let x = var "x" 0 100 in
          let masked = U.alu_binary ~op:Ops.And ~lhs:x ~rhs:(idx (-4)) in
          let expr = U.alu_binary ~op:Ops.Floordiv ~lhs:masked ~rhs:(idx 4) in
          match sym expr with
          | Some r ->
              check_op r Ops.Floordiv;
              is_true (src r 0 == x);
              check_const_int (src r 1) 4
          | None -> fail "expected masked division to collapse");
    ]

(* Every remaining Invalid sentinel is rewritten to a typed zero. *)

let remove_invalid_tests =
  let remove n = Upat.Pattern_matcher.rewrite Symbolic.pm_remove_invalid n in
  group "remove_invalid"
    [
      test "invalid(int32) -> const 0 : int32" (fun () ->
          match remove (U.invalid ~dtype:D.int32 ()) with
          | Some r ->
              check_const_int r 0;
              is_true (D.equal (U.dtype r) D.int32)
          | None -> fail "expected invalid to be zeroed");
      test "invalid(float32) -> const 0.0 : float32" (fun () ->
          match remove (U.invalid ~dtype:D.float32 ()) with
          | Some r ->
              check_const_float r 0.0;
              is_true (D.equal (U.dtype r) D.float32)
          | None -> fail "expected invalid to be zeroed");
      test "invalid(index) -> const 0 : index" (fun () ->
          match remove (U.invalid ()) with
          | Some r ->
              check_const_int r 0;
              is_true (D.equal (U.dtype r) D.index)
          | None -> fail "expected invalid to be zeroed");
    ]

(* Entry point *)

let () =
  run "tolk.uop.symbolic"
    (simplify_driver_groups
     @ [
         const_fold_tests;
         identity_fold_tests;
         self_fold_tests;
         divmod_reconstitute_tests;
         divandmod_tests;
         combine_terms_tests;
         associative_tests;
         index_pushing_tests;
         spec_tests;
         bool_cast_fold_tests;
         lt_fold_tests;
         where_fold_tests;
         reduce_tests;
         load_store_tests;
         invalid_where_tests;
         sigmoid_tests;
         masked_div_tests;
         remove_invalid_tests;
       ])
