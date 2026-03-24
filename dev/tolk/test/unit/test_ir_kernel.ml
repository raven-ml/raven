(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module D = Tolk_ir.Dtype
module C = Tolk_ir.Const
module K = Tolk_ir.Kernel
module Ak = Tolk_ir.Axis_kind
module Sd = Tolk_ir.Special_dim

(* Helpers *)

let global_ptr dt = D.ptr_of dt ~addrspace:Global ~size:(-1)
let local_ptr dt = D.ptr_of dt ~addrspace:Local ~size:(-1)
let reg_ptr dt = D.ptr_of dt ~addrspace:Reg ~size:(-1)
let mk_idx () = K.const (C.int D.index 0)
let mk_f32 () = K.const (C.float D.float32 1.0)
let mk_i32 () = K.const (C.int D.int32 0)
let mk_param () = K.param ~idx:0 ~dtype:(global_ptr D.float32)

let mk_load () =
  let idx = K.index ~ptr:(mk_param ()) ~idxs:[ mk_idx () ] () in
  K.load ~src:idx ()

let contains haystack needle =
  let hlen = String.length haystack in
  let nlen = String.length needle in
  let rec loop i =
    if i + nlen > hlen then false
    else if String.sub haystack i nlen = needle then true
    else loop (i + 1)
  in
  loop 0

let validate_ok node = K.validate (K.sink [ node ])

let raises_validate substring fn =
  raises_match (function Failure msg -> contains msg substring | _ -> false) fn

let raises_invalid fn =
  raises_match (function Invalid_argument _ -> true | _ -> false) fn

let dtype_eq expected node =
  match K.dtype node with
  | Some dt -> is_true (D.equal dt expected)
  | None -> fail "expected a dtype but got None"

let has_const_int n nodes =
  List.exists
    (fun node ->
      match K.view node with
      | Const { value; _ } ->
          (match C.view value with Int v -> Int64.to_int v = n | _ -> false)
      | _ -> false)
    nodes

let const_int_value node =
  match K.view node with
  | Const { value; _ } ->
      (match C.view value with Int n -> Some (Int64.to_int n) | _ -> None)
  | _ -> None

let () =
  run "Ir_next.Kernel"
    [
      group "Smart constructor dtype inference"
        [
          test "binary cmplt produces bool" (fun () ->
            dtype_eq D.bool
              (K.binary ~op:`Cmplt ~lhs:(mk_f32 ()) ~rhs:(mk_f32 ())));
          test "binary cmpeq preserves lanes" (fun () ->
            let s1 = mk_f32 () and s2 = mk_f32 () in
            let s3 = mk_f32 () and s4 = mk_f32 () in
            let v1 = K.vectorize ~srcs:[ s1; s2; s3; s4 ] in
            let v2 = K.vectorize ~srcs:[ s4; s3; s2; s1 ] in
            dtype_eq (D.vec D.bool 4) (K.binary ~op:`Cmpeq ~lhs:v1 ~rhs:v2));
          test "binary cmpne produces bool" (fun () ->
            dtype_eq D.bool
              (K.binary ~op:`Cmpne ~lhs:(mk_i32 ()) ~rhs:(mk_i32 ())));
          test "binary add inherits lhs" (fun () ->
            dtype_eq D.float32
              (K.binary ~op:`Add ~lhs:(mk_f32 ()) ~rhs:(mk_f32 ())));
          test "binary shl inherits lhs" (fun () ->
            dtype_eq D.int32
              (K.binary ~op:`Shl ~lhs:(mk_i32 ()) ~rhs:(mk_i32 ())));
          test "ternary where inherits b" (fun () ->
            let b = mk_f32 () and c = mk_f32 () in
            dtype_eq D.float32
              (K.ternary ~op:`Where ~a:(K.const_bool true) ~b ~c));
          test "ternary mulacc inherits a" (fun () ->
            let a = mk_i32 () and b = mk_i32 () and c = mk_i32 () in
            dtype_eq D.int32 (K.ternary ~op:`Mulacc ~a ~b ~c));
          test "unary sqrt inherits src" (fun () ->
            dtype_eq D.float32 (K.unary ~op:`Sqrt ~src:(mk_f32 ())));
          test "index derives ptr dtype" (fun () ->
            let index = K.index ~ptr:(mk_param ()) ~idxs:[ mk_idx () ] () in
            is_true (K.is_ptr index));
          test "load derives base dtype" (fun () ->
            dtype_eq D.float32 (mk_load ()));
          test "vectorize dtype" (fun () ->
            let s1 = mk_f32 () and s2 = mk_f32 () and s3 = mk_f32 () in
            dtype_eq (D.vec D.float32 3) (K.vectorize ~srcs:[ s1; s2; s3 ]));
          test "cat sums counts" (fun () ->
            let a = mk_f32 () and b = mk_f32 () in
            let c = mk_f32 () and d = mk_f32 () and e = mk_f32 () in
            let v2 = K.vectorize ~srcs:[ a; b ] in
            let v3 = K.vectorize ~srcs:[ c; d; e ] in
            dtype_eq (D.vec D.float32 5) (K.cat ~srcs:[ v2; v3 ]));
          test "gep gives scalar" (fun () ->
            let s1 = mk_f32 () and s2 = mk_f32 () in
            let s3 = mk_f32 () and s4 = mk_f32 () in
            let v4 = K.vectorize ~srcs:[ s1; s2; s3; s4 ] in
            dtype_eq D.float32 (K.gep ~src:v4 ~idx:2));
          test "const_int is index" (fun () ->
            dtype_eq D.index (K.const_int 42));
          test "const_float is float32" (fun () ->
            dtype_eq D.float32 (K.const_float 3.14));
          test "const_bool is bool" (fun () ->
            dtype_eq D.bool (K.const_bool false));
        ];
      group "Smart constructor edge cases"
        [
          test "vectorize empty raises" (fun () ->
            raises_invalid (fun () -> ignore (K.vectorize ~srcs:[])));
          test "cat empty raises" (fun () ->
            raises_invalid (fun () -> ignore (K.cat ~srcs:[])));
          test "cat mixed scalar raises" (fun () ->
            let f = mk_f32 () and i = mk_i32 () in
            let vf = K.vectorize ~srcs:[ f; f ] in
            let vi = K.vectorize ~srcs:[ i; i ] in
            raises_invalid (fun () -> ignore (K.cat ~srcs:[ vf; vi ])));
          test "gep_multi empty raises" (fun () ->
            let v = K.vectorize ~srcs:[ mk_f32 (); mk_f32 () ] in
            raises_invalid (fun () -> ignore (K.gep_multi ~src:v ~idxs:[])));
          test "gep_multi identity on scalar idx 0" (fun () ->
            let s = mk_f32 () in
            is_true (K.gep_multi ~src:s ~idxs:[ 0 ] == s));
          test "gep_multi single gives gep" (fun () ->
            let add = K.binary ~op:`Add ~lhs:(mk_f32 ()) ~rhs:(mk_f32 ()) in
            let v = K.vectorize ~srcs:[ add; add ] in
            is_true (K.gep_multi ~src:v ~idxs:[ 0 ] == add);
            let result = K.gep_multi ~src:v ~idxs:[ 0; 1 ] in
            (match K.view result with
             | Gep { idxs = [0; 1]; _ } -> ()
             | _ -> fail "expected multi-element Gep"));
          test "gep_multi multi gives gep" (fun () ->
            let v =
              K.vectorize
                ~srcs:[ mk_f32 (); mk_f32 (); mk_f32 (); mk_f32 () ]
            in
            let result = K.gep_multi ~src:v ~idxs:[ 0; 2 ] in
            (match K.view result with
             | Gep { idxs = [0; 2]; _ } -> ()
             | _ -> fail "expected multi-element Gep"));
          test "broadcast scalar to n" (fun () ->
            let s = mk_f32 () in
            let b = K.broadcast s 4 in
            dtype_eq (D.vec D.float32 4) b;
            (match K.view b with
             | Vectorize { srcs; _ } -> equal int 4 (List.length srcs)
             | _ -> fail "expected Vectorize"));
          test "broadcast pointer creates vectorize" (fun () ->
            let p = mk_param () in
            let b = K.broadcast p 4 in
            is_true (match K.view b with K.Vectorize { srcs; _ } -> List.length srcs = 4 | _ -> false));
          test "broadcast n <= 1 identity" (fun () ->
            let s = mk_f32 () in
            is_true (K.broadcast s 1 == s);
            is_true (K.broadcast s 0 == s));
          test "zero_like float int bool" (fun () ->
            dtype_eq D.float32 (K.zero_like (mk_f32 ()));
            dtype_eq D.int32 (K.zero_like (mk_i32 ()));
            dtype_eq D.bool (K.zero_like (K.const_bool true)));
          test "zero_like no dtype raises" (fun () ->
            raises_invalid (fun () -> ignore (K.zero_like K.barrier)));
        ];
      group "Validation acceptance"
        [
          test "param global" (fun () ->
            validate_ok (K.param ~idx:0 ~dtype:(global_ptr D.float32)));
          test "define_local local" (fun () ->
            validate_ok (K.define_local ~size:8 ~dtype:(local_ptr D.float32)));
          test "define_reg reg" (fun () ->
            validate_ok (K.define_reg ~size:4 ~dtype:(reg_ptr D.float32) ~slot:0));
          test "define_var" (fun () ->
            validate_ok (K.define_var ~name:"n" ~lo:0 ~hi:10 ()));
          test "const all types" (fun () ->
            validate_ok (K.const_bool true);
            validate_ok (K.const_int 42);
            validate_ok (K.const_float 3.14));
          test "binary shift uint32 rhs" (fun () ->
            validate_ok
              (K.binary ~op:`Shl ~lhs:(mk_i32 ())
                 ~rhs:(K.const (C.int D.uint32 2))));
          test "full load/store chain" (fun () ->
            let ptr = K.param ~idx:0 ~dtype:(global_ptr D.float32) in
            let idx = mk_idx () in
            let index = K.index ~ptr ~idxs:[ idx ] () in
            let loaded = K.load ~src:index () in
            let added = K.binary ~op:`Add ~lhs:loaded ~rhs:(mk_f32 ()) in
            let dst_idx = K.index ~ptr ~idxs:[ idx ] () in
            K.validate
              (K.sink [ K.store ~dst:dst_idx ~value:added ~ranges:[] ]));
          test "store with ranges" (fun () ->
            let ptr = K.param ~idx:0 ~dtype:(global_ptr D.float32) in
            let index = K.index ~ptr ~idxs:[ mk_idx () ] () in
            let size = K.const (C.int D.index 10) in
            let range = K.range ~size ~axis:0 ~kind:Ak.Loop () in
            K.validate
              (K.sink
                 [ K.store ~dst:index ~value:(mk_f32 ()) ~ranges:[ range ] ]));
          test "contract axes" (fun () ->
            validate_ok
              (K.contract ~src:(mk_f32 ()) ~axes:[ (0, 3); (1, 2) ]
                 ~dtype:(D.vec D.float32 6)));
          test "unroll axes" (fun () ->
            let s = mk_f32 () and s2 = mk_f32 () and s3 = mk_f32 () in
            let s4 = mk_f32 () and s5 = mk_f32 () and s6 = mk_f32 () in
            let src = K.vectorize ~srcs:[ s; s2; s3; s4; s5; s6 ] in
            validate_ok
              (K.unroll ~src ~axes:[ (0, 3); (1, 2) ] ~dtype:D.float32));
          test "vectorized local index operand" (fun () ->
            let ptr = local_ptr D.float32 in
            let zero = K.const (C.int D.index 0) in
            let one = K.const (C.int D.index 1) in
            let idxs = K.vectorize ~srcs:[ zero; one ] in
            let local = K.define_local ~size:8 ~dtype:ptr in
            K.validate (K.sink [ K.index ~ptr:local ~idxs:[ idxs ] () ]));
          test "horizontal reduce src" (fun () ->
            let one = K.const (C.float D.float32 1.0) in
            let two = K.const (C.float D.float32 2.0) in
            let src = K.vectorize ~srcs:[ one; two ] in
            let size = K.const (C.int D.index 2) in
            let range = K.range ~size ~axis:0 ~kind:Ak.Reduce () in
            K.validate
              (K.sink
                 [ K.reduce ~op:`Add ~src ~ranges:[ range ] ~dtype:D.float32 ]));
        ];
      group "Validation rejection"
        [
          test "reject param local addrspace" (fun () ->
            raises_validate "Global addrspace" (fun () ->
                validate_ok (K.param ~idx:0 ~dtype:(local_ptr D.float32))));
          test "reject define_local global addrspace" (fun () ->
            raises_validate "Local addrspace" (fun () ->
                validate_ok
                  (K.define_local ~size:8 ~dtype:(global_ptr D.float32))));
          test "reject define_reg local addrspace" (fun () ->
            raises_validate "Reg addrspace" (fun () ->
                validate_ok
                  (K.define_reg ~size:4 ~dtype:(local_ptr D.float32) ~slot:0)));
          test "reject define_var vector" (fun () ->
            raises_validate "must be scalar" (fun () ->
                validate_ok
                  (K.define_var ~name:"v" ~lo:0 ~hi:4
                     ~dtype:(D.vec D.int32 4) ())));
          test "reject define_var float" (fun () ->
            raises_validate "must be int/index" (fun () ->
                validate_ok
                  (K.define_var ~name:"f" ~lo:0 ~hi:4 ~dtype:D.float32 ())));
          test "reject define_var lo > hi" (fun () ->
            raises_validate "lo > hi" (fun () ->
                validate_ok (K.define_var ~name:"x" ~lo:5 ~hi:3 ())));
          test "reject range float" (fun () ->
            raises_validate "Range must have int" (fun () ->
                validate_ok
                  (K.range ~size:(mk_f32 ()) ~axis:0 ~kind:Ak.Loop
                     ~dtype:D.float32 ())));
          test "reject range vector" (fun () ->
            raises_validate "Range must be scalar" (fun () ->
                validate_ok
                  (K.range ~size:(mk_i32 ()) ~axis:0 ~kind:Ak.Loop
                     ~dtype:(D.vec D.int32 4) ())));
          test "reject special vector" (fun () ->
            raises_validate "must be scalar" (fun () ->
                validate_ok
                  (K.special ~dim:(Sd.Group_id 0) ~size:(mk_i32 ())
                     ~dtype:(D.vec D.int32 2) ())));
          test "reject special float" (fun () ->
            raises_validate "must be index or int32" (fun () ->
                validate_ok
                  (K.special ~dim:(Sd.Group_id 0) ~size:(mk_f32 ())
                     ~dtype:D.float32 ())));
          test "reject index empty idxs" (fun () ->
            raises_validate "at least one index" (fun () ->
                validate_ok (K.index ~ptr:(mk_param ()) ~idxs:[] ())));
          test "reject index non-buffer base" (fun () ->
            raises_invalid (fun () ->
                ignore (K.index ~ptr:(mk_f32 ()) ~idxs:[ mk_idx () ] ())));
          test "reject index non-index operand" (fun () ->
            raises_validate "must be index-like" (fun () ->
                validate_ok
                  (K.index ~ptr:(mk_param ()) ~idxs:[ mk_f32 () ] ())));
          test "reject index non-bool gate" (fun () ->
            raises_validate "must be bool scalar" (fun () ->
                validate_ok
                  (K.index ~ptr:(mk_param ()) ~idxs:[ mk_idx () ]
                     ~gate:(mk_i32 ()) ())));
          test "reject cmp dtype mismatch" (fun () ->
            raises_validate "don't match" (fun () ->
                validate_ok
                  (K.binary ~op:`Cmplt ~lhs:(mk_f32 ()) ~rhs:(mk_i32 ()))));
          test "reject shift non-int" (fun () ->
            raises_validate "Shift must have int" (fun () ->
                validate_ok
                  (K.binary ~op:`Shl ~lhs:(mk_f32 ()) ~rhs:(mk_f32 ()))));
          test "reject idiv non-int" (fun () ->
            raises_validate "Idiv/Mod must have int" (fun () ->
                validate_ok
                  (K.binary ~op:`Idiv ~lhs:(mk_f32 ()) ~rhs:(mk_f32 ()))));
          test "reject where non-bool cond" (fun () ->
            raises_validate "must be bool" (fun () ->
                validate_ok
                  (K.ternary ~op:`Where ~a:(mk_i32 ()) ~b:(mk_f32 ())
                     ~c:(mk_f32 ()))));
          test "reject where mismatched arms" (fun () ->
            raises_validate "arms" (fun () ->
                validate_ok
                  (K.ternary ~op:`Where ~a:(K.const_bool true)
                     ~b:(mk_f32 ()) ~c:(mk_i32 ()))));
          test "reject gep out of bounds" (fun () ->
            let v =
              K.vectorize
                ~srcs:[ mk_f32 (); mk_f32 (); mk_f32 (); mk_f32 () ]
            in
            raises_validate "out of bounds" (fun () ->
                validate_ok (K.gep ~src:v ~idx:5)));
          test "accept gep scalar source" (fun () ->
            validate_ok (K.gep ~src:(mk_f32 ()) ~idx:0));
          test "reject store value dtype mismatch" (fun () ->
            let ptr = K.param ~idx:0 ~dtype:(global_ptr D.float32) in
            let index = K.index ~ptr ~idxs:[ mk_idx () ] () in
            raises_validate "Store value" (fun () ->
                K.validate
                  (K.sink
                     [ K.store ~dst:index ~value:(mk_i32 ()) ~ranges:[] ])));
          test "reject load alt without gate" (fun () ->
            let ptr = K.param ~idx:0 ~dtype:(global_ptr D.float32) in
            let index = K.index ~ptr ~idxs:[ mk_idx () ] () in
            raises_validate "alt requires gated" (fun () ->
                K.validate
                  (K.sink [ K.load ~src:index ~alt:(mk_f32 ()) () ])));
          test "reject ptrcat empty" (fun () ->
            raises_validate "at least one source" (fun () ->
                validate_ok
                  (K.ptrcat ~srcs:[] ~dtype:(global_ptr D.float32))));
          test "reject unroll count mismatch" (fun () ->
            let s = mk_f32 () and s2 = mk_f32 () in
            let s3 = mk_f32 () and s4 = mk_f32 () in
            let src = K.vectorize ~srcs:[ s; s2; s3; s4 ] in
            raises_validate "count mismatch" (fun () ->
                validate_ok
                  (K.unroll ~src ~axes:[ (0, 3); (1, 2) ] ~dtype:D.float32)));
          test "reject contract count mismatch" (fun () ->
            raises_validate "count mismatch" (fun () ->
                validate_ok
                  (K.contract ~src:(mk_f32 ()) ~axes:[ (0, 3) ]
                     ~dtype:(D.vec D.float32 2))));
          test "reject bufferize addrspace mismatch" (fun () ->
            let opts : K.bufferize_opts =
              { device = None; addrspace = D.Global; removable = false }
            in
            raises_validate "addrspace mismatch" (fun () ->
                validate_ok
                  (K.bufferize ~src:(mk_f32 ()) ~ranges:[]
                     ~dtype:(local_ptr D.float32) ~opts)));
        ];
      group "Graph infrastructure"
        [
          test "toposort leaf to root" (fun () ->
            let a = mk_f32 () in
            let b = K.unary ~op:`Neg ~src:a in
            let c = K.unary ~op:`Neg ~src:b in
            let root = K.sink [ c ] in
            let order = K.toposort root in
            equal int 4 (List.length order);
            is_true (List.hd order == a);
            is_true (List.nth order 3 == root));
          test "toposort diamond" (fun () ->
            let a = mk_f32 () in
            let b = K.unary ~op:`Neg ~src:a in
            let c = K.unary ~op:`Sqrt ~src:a in
            let d = K.binary ~op:`Add ~lhs:b ~rhs:c in
            let root = K.sink [ d ] in
            let order = K.toposort root in
            equal int 1
              (List.length (List.filter (fun n -> n == a) order));
            is_true (List.nth order (List.length order - 1) == root));
          test "intern dedup" (fun () ->
            let b1 = K.unary ~op:`Neg ~src:(K.const_int 42) in
            let b2 = K.unary ~op:`Neg ~src:(K.const_int 42) in
            let interned =
              K.intern (K.binary ~op:`Add ~lhs:b1 ~rhs:b2)
            in
            (match K.children interned with
             | [ lhs; rhs ] -> is_true (lhs == rhs)
             | _ -> fail "expected two children"));
          test "intern preserves validity" (fun () ->
            let ptr = K.param ~idx:0 ~dtype:(global_ptr D.float32) in
            let index = K.index ~ptr ~idxs:[ mk_idx () ] () in
            let root = K.sink [ K.load ~src:index () ] in
            K.validate root;
            K.validate (K.intern root));
          test "sort pointer variants" (fun () ->
            is_true (K.sort (mk_param ()) = K.Pointer);
            is_true
              (K.sort
                 (K.define_local ~size:8 ~dtype:(local_ptr D.float32))
               = K.Pointer);
            is_true
              (K.sort (K.define_reg ~size:4 ~dtype:(reg_ptr D.float32) ~slot:0)
               = K.Pointer);
            is_true
              (K.sort
                 (K.index ~ptr:(mk_param ()) ~idxs:[ mk_idx () ] ())
               = K.Pointer));
          test "sort effect variants" (fun () ->
            is_true (K.sort (K.sink []) = K.Effect);
            is_true (K.sort K.barrier = K.Effect);
            let index =
              K.index ~ptr:(mk_param ()) ~idxs:[ mk_idx () ] ()
            in
            is_true
              (K.sort (K.store ~dst:index ~value:(mk_f32 ()) ~ranges:[])
               = K.Effect));
          test "sort index variants" (fun () ->
            let size = K.const (C.int D.index 10) in
            is_true
              (K.sort (K.range ~size ~axis:0 ~kind:Ak.Loop ()) = K.Index);
            is_true
              (K.sort (K.special ~dim:(Sd.Group_id 0) ~size ()) = K.Index);
            is_true
              (K.sort (K.define_var ~name:"n" ~lo:0 ~hi:10 ()) = K.Index));
          test "sort value vs index" (fun () ->
            is_true
              (K.sort
                 (K.binary ~op:`Add ~lhs:(mk_f32 ()) ~rhs:(mk_f32 ()))
               = K.Value);
            is_true
              (K.sort
                 (K.binary ~op:`Add ~lhs:(mk_idx ()) ~rhs:(mk_idx ()))
               = K.Index));
          test "is_alu and is_ptr" (fun () ->
            is_true (K.is_alu (K.unary ~op:`Neg ~src:(mk_f32 ())));
            is_true
              (K.is_alu
                 (K.binary ~op:`Add ~lhs:(mk_f32 ()) ~rhs:(mk_f32 ())));
            is_true
              (K.is_alu
                 (K.ternary ~op:`Where ~a:(K.const_bool true)
                    ~b:(mk_f32 ()) ~c:(mk_f32 ())));
            is_false (K.is_alu (mk_f32 ()));
            is_true (K.is_ptr (mk_param ()));
            is_true
              (K.is_ptr
                 (K.index ~ptr:(mk_param ()) ~idxs:[ mk_idx () ] ()));
            is_false (K.is_ptr (mk_f32 ())));
        ];
      group "Rewriting"
        [
          test "rebuild replaces const" (fun () ->
            let four = K.const (C.int D.index 4) in
            let neg = K.unary ~op:`Neg ~src:(K.const (C.int D.index 3)) in
            let root = K.sink [ neg ] in
            let rewrite node =
              match K.view node with
              | Const { value; _ } ->
                  (match C.view value with
                   | Int n when Int64.to_int n = 3 -> Some four
                   | _ -> None)
              | _ -> None
            in
            let nodes = K.toposort (K.graph_rewrite rewrite root) in
            is_false (has_const_int 3 nodes);
            is_true (has_const_int 4 nodes));
          test "graph_rewrite no match identity" (fun () ->
            let root = K.sink [ K.unary ~op:`Neg ~src:(mk_f32 ()) ] in
            let result = K.graph_rewrite (fun _ -> None) root in
            equal int
              (List.length (K.toposort root))
              (List.length (K.toposort result)));
          test "graph_rewrite simplifies" (fun () ->
            let x = mk_f32 () in
            let zero = K.const (C.float D.float32 0.0) in
            let root =
              K.sink [ K.binary ~op:`Add ~lhs:x ~rhs:zero ]
            in
            let rewrite node =
              match K.view node with
              | Binary { op = `Add; rhs; _ } ->
                  (match K.view rhs with
                   | Const { value; _ } ->
                       (match C.view value with
                        | Float f when f = 0.0 -> Some x
                        | _ -> None)
                   | _ -> None)
              | _ -> None
            in
            is_true
              (List.length (K.toposort (K.graph_rewrite rewrite root))
               < List.length (K.toposort root)));
          test "first_match returns first" (fun () ->
            let r1 _ = Some (K.const_int 1) in
            let r2 _ = Some (K.const_int 2) in
            (match K.first_match [ r1; r2 ] (mk_f32 ()) with
             | Some result -> equal (option int) (Some 1) (const_int_value result)
             | None -> fail "expected Some"));
          test "first_match skips none" (fun () ->
            let r1 _ = None in
            let r2 _ = Some (K.const_int 2) in
            (match K.first_match [ r1; r2 ] (mk_f32 ()) with
             | Some result -> equal (option int) (Some 2) (const_int_value result)
             | None -> fail "expected Some"));
          test "replace binary children" (fun () ->
            let a = mk_f32 () and b = mk_f32 () in
            let add = K.binary ~op:`Add ~lhs:a ~rhs:b in
            let c = mk_f32 () and d = mk_f32 () in
            (match K.children (K.replace add ~children:[ c; d ] ()) with
             | [ lhs; rhs ] ->
                 is_true (lhs == c);
                 is_true (rhs == d)
             | _ -> fail "expected two children"));
        ];
      group "Formatting and Opt"
        [
          test "pp diamond bounded size" (fun () ->
            let rec build depth node =
              if depth = 0 then node
              else
                let left = K.unary ~op:`Neg ~src:node in
                let right = K.unary ~op:`Sqrt ~src:node in
                build (depth - 1) (K.binary ~op:`Add ~lhs:left ~rhs:right)
            in
            let root = K.sink [ build 20 (mk_f32 ()) ] in
            is_true
              (String.length (Format.asprintf "%a" K.pp root) < 10_000));
          test "pp includes ops and dtypes" (fun () ->
            let root =
              K.sink
                [ K.binary ~op:`Add ~lhs:(mk_f32 ()) ~rhs:(mk_f32 ()) ]
            in
            let output = Format.asprintf "%a" K.pp root in
            is_true (contains output "add");
            is_true (contains output "f32"));
          test "opt to_string all variants" (fun () ->
            equal string "LOCAL:0:4"
              (K.Opt.to_string (Local { axis = 0; amount = 4 }));
            equal string "UPCAST:1:8"
              (K.Opt.to_string (Upcast { axis = 1; amount = 8 }));
            equal string "UNROLL:2:3"
              (K.Opt.to_string (Unroll { axis = 2; amount = 3 }));
            equal string "GROUP:0:16"
              (K.Opt.to_string (Group { axis = 0; amount = 16 }));
            equal string "GROUPTOP:1:32"
              (K.Opt.to_string (Grouptop { axis = 1; amount = 32 }));
            equal string "THREAD:0:2"
              (K.Opt.to_string (Thread { axis = 0; amount = 2 }));
            equal string "NOLOCALS" (K.Opt.to_string Nolocals);
            equal string "TC:0:1:2:3"
              (K.Opt.to_string
                 (Tc { axis = 0; tc_select = 1; tc_opt = 2; use_tc = 3 }));
            equal string "PADTO:3:64"
              (K.Opt.to_string (Padto { axis = 3; amount = 64 }));
            equal string "SWAP:0:1"
              (K.Opt.to_string (Swap { axis = 0; with_axis = 1 })));
          test "kernel_info in sink survives validate" (fun () ->
            let ki : K.kernel_info =
              {
                name = "test_kernel";
                axis_kinds = [ Ak.Global; Ak.Loop; Ak.Reduce ];
                dont_use_locals = false;
                applied_opts = [ K.Opt.Local { axis = 0; amount = 4 } ];
                opts_to_apply = None;
                estimates = None;
              }
            in
            K.validate (K.sink ~kernel_info:ki []));
        ];
      group "Constructor short-circuits"
        [
          test "group singleton returns child" (fun () ->
            let x = mk_f32 () in
            is_true (K.group [ x ] == x));
          test "group empty creates Group" (fun () ->
            (match K.view (K.group []) with
             | Group _ -> ()
             | _ -> fail "expected Group"));
          test "group multi creates Group" (fun () ->
            let a = mk_f32 () and b = mk_f32 () in
            (match K.view (K.group [ a; b ]) with
             | Group { srcs } -> equal int 2 (List.length srcs)
             | _ -> fail "expected Group"));
          test "after empty deps returns src" (fun () ->
            let x = mk_f32 () in
            is_true (K.after ~src:x ~deps:[] == x));
          test "after with deps creates After" (fun () ->
            let x = mk_f32 () and d = mk_f32 () in
            (match K.view (K.after ~src:x ~deps:[ d ]) with
             | After { src; deps } ->
                 is_true (src == x);
                 equal int 1 (List.length deps)
             | _ -> fail "expected After"));
        ];
      group "Range analysis"
        [
          test "ended_ranges for End" (fun () ->
            let size = K.const (C.int D.index 4) in
            let r0 = K.range ~size ~axis:0 ~kind:Ak.Loop () in
            let ended = K.end_ ~value:(mk_f32 ()) ~ranges:[ r0 ] () in
            let ers = K.ended_ranges ended in
            equal int 1 (List.length ers);
            is_true (List.hd ers == r0));
          test "ended_ranges for Reduce" (fun () ->
            let size = K.const (C.int D.index 4) in
            let r0 = K.range ~size ~axis:0 ~kind:Ak.Reduce () in
            let red = K.reduce ~op:`Add ~src:(mk_f32 ()) ~ranges:[ r0 ]
                ~dtype:D.float32 in
            let ers = K.ended_ranges red in
            equal int 1 (List.length ers);
            is_true (List.hd ers == r0));
          test "ended_ranges for Store" (fun () ->
            let size = K.const (C.int D.index 4) in
            let r0 = K.range ~size ~axis:0 ~kind:Ak.Loop () in
            let idx = K.index ~ptr:(mk_param ()) ~idxs:[ mk_idx () ] () in
            let st = K.store ~dst:idx ~value:(mk_f32 ()) ~ranges:[ r0 ] in
            let ers = K.ended_ranges st in
            equal int 1 (List.length ers);
            is_true (List.hd ers == r0));
          test "ended_ranges for After delegates to deps" (fun () ->
            let size = K.const (C.int D.index 4) in
            let r0 = K.range ~size ~axis:0 ~kind:Ak.Loop () in
            let ended = K.end_ ~value:(mk_f32 ()) ~ranges:[ r0 ] () in
            let aft = K.after ~src:(mk_f32 ()) ~deps:[ ended ] in
            let ers = K.ended_ranges aft in
            equal int 1 (List.length ers);
            is_true (List.hd ers == r0));
          test "ended_ranges for leaf is empty" (fun () ->
            equal int 0 (List.length (K.ended_ranges (mk_f32 ()))));
          test "ended_ranges for Contract with live" (fun () ->
            let size = K.const (C.int D.index 4) in
            let r0 = K.range ~size ~axis:0 ~kind:Ak.Upcast () in
            let r1 = K.range ~size ~axis:1 ~kind:Ak.Upcast () in
            let r2 = K.range ~size ~axis:2 ~kind:Ak.Loop () in
            let live _ = [ r0; r1; r2 ] in
            let contract =
              K.contract ~src:(mk_f32 ()) ~axes:[ (0, 4); (1, 4) ]
                ~dtype:(D.vec D.float32 16)
            in
            let ers = K.ended_ranges ~live contract in
            equal int 2 (List.length ers);
            is_true (List.exists (fun r -> r == r0) ers);
            is_true (List.exists (fun r -> r == r1) ers);
            is_false (List.exists (fun r -> r == r2) ers));
          test "ended_ranges for Contract without live is empty" (fun () ->
            let contract =
              K.contract ~src:(mk_f32 ()) ~axes:[ (0, 4) ]
                ~dtype:(D.vec D.float32 4)
            in
            equal int 0 (List.length (K.ended_ranges contract)));
          test "live_ranges_tbl simple reduce" (fun () ->
            let size = K.const (C.int D.index 4) in
            let r0 = K.range ~size ~axis:0 ~kind:Ak.Reduce () in
            let red = K.reduce ~op:`Add ~src:(mk_f32 ()) ~ranges:[ r0 ]
                ~dtype:D.float32 in
            let root = K.sink [ red ] in
            let tbl = K.live_ranges_tbl root in
            (* r0 is live at itself *)
            let r0_live =
              match K.Ref_tbl.find_opt tbl r0 with Some r -> r | None -> []
            in
            is_true (List.exists (fun r -> r == r0) r0_live);
            (* r0 is NOT live at reduce (it's ended there) *)
            let red_live =
              match K.Ref_tbl.find_opt tbl red with Some r -> r | None -> []
            in
            is_false (List.exists (fun r -> r == r0) red_live));
          test "live_ranges_tbl nested ranges" (fun () ->
            let size = K.const (C.int D.index 4) in
            let r0 = K.range ~size ~axis:0 ~kind:Ak.Loop () in
            let r1 = K.range ~size ~axis:1 ~kind:Ak.Reduce () in
            (* src depends on both ranges so both are in its backward slice *)
            let src = K.binary ~op:`Add ~lhs:r0 ~rhs:r1 in
            let red = K.reduce ~op:`Add ~src ~ranges:[ r1 ]
                ~dtype:D.index in
            let ended = K.end_ ~value:red ~ranges:[ r0 ] () in
            let root = K.sink [ ended ] in
            let tbl = K.live_ranges_tbl root in
            (* r0 and r1 are both live at src (it depends on both ranges) *)
            let src_live =
              match K.Ref_tbl.find_opt tbl src with Some r -> r | None -> []
            in
            is_true (List.exists (fun r -> r == r0) src_live);
            is_true (List.exists (fun r -> r == r1) src_live);
            (* r1 is ended at reduce, but r0 is still live *)
            let red_live =
              match K.Ref_tbl.find_opt tbl red with Some r -> r | None -> []
            in
            is_true (List.exists (fun r -> r == r0) red_live);
            is_false (List.exists (fun r -> r == r1) red_live));
        ];
      group "Substitute"
        [
          test "substitute replaces by identity" (fun () ->
            let a = K.const (C.float D.float32 1.0) in
            let b = K.const (C.float D.float32 2.0) in
            let add = K.binary ~op:`Add ~lhs:a ~rhs:b in
            let root = K.sink [ add ] in
            let c = K.const (C.float D.float32 3.0) in
            let result = K.substitute [ (a, c) ] root in
            (* The old a should be replaced with c *)
            let nodes = K.toposort result in
            is_true (List.exists (fun n -> n == c) nodes);
            is_false (List.exists (fun n -> n == a) nodes));
          test "substitute no match identity" (fun () ->
            let a = mk_f32 () and b = mk_f32 () in
            let add = K.binary ~op:`Add ~lhs:a ~rhs:b in
            let root = K.sink [ add ] in
            let result = K.substitute [] root in
            is_true (result == root));
          test "substitute propagates tags" (fun () ->
            let tags = K.Ref_tbl.create 4 in
            let a = mk_f32 () in
            K.Ref_tbl.replace tags a 42;
            let b = mk_f32 () in
            let add = K.binary ~op:`Add ~lhs:a ~rhs:b in
            let root = K.sink [ add ] in
            let c = mk_f32 () in
            let _ = K.substitute ~tags [ (a, c) ] root in
            (* Tag should be copied from a to c *)
            equal (option int) (Some 42) (K.Ref_tbl.find_opt tags c));
        ];
    ]
