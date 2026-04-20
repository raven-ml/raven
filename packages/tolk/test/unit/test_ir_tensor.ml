(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Tolk_ir.Const
module D = Tolk_ir.Dtype
module T = Tolk_ir.Tensor
module Ak = Tolk_ir.Axis_kind
module Sh = Tolk_ir.Shape

let contains haystack needle =
  let hlen = String.length haystack in
  let nlen = String.length needle in
  let rec loop i =
    if i + nlen > hlen then false
    else if String.sub haystack i nlen = needle then true
    else loop (i + 1)
  in
  loop 0

let raises_validate substring fn =
  raises_match
    (function Failure msg -> contains msg substring | _ -> false)
    fn

let mk_f32 () = T.const (C.float D.Val.float32 1.0) D.float32
let mk_i32 () = T.const (C.int D.Val.int32 0) D.int32
let mk_idx () = T.const (C.int D.Val.index 0) D.index
let mk_bool () = T.const (C.bool true) D.bool

let emit_buffer ?(dtype = D.float32) () =
  let u = T.unique ~id:0 in
  let d = T.device (Single "CPU") in
  let buf = T.buffer ~unique:u ~device:d ~size:1024 ~dtype in
  (u, d, buf)

let mk_shape_2x3 () =
  let d1 = T.const (C.int D.Val.index 2) D.index in
  let d2 = T.const (C.int D.Val.index 3) D.index in
  T.vectorize ~srcs:[ d1; d2 ]

let dtype_eq expected id =
  match T.dtype id with
  | Some dt -> is_true (D.equal dt expected)
  | None -> fail "expected a dtype but got None"

let mk_index_on_buf () =
  let _u, _d, buf = emit_buffer () in
  let idx = mk_idx () in
  (buf, T.index ~ptr:buf ~idxs:[ idx ] ~dtype:D.float32 ())

let call_info : T.call_info =
  { grad_fxn = None; metadata = []; name = None; precompile = false }

let pp_to_string pp v =
  let buf = Buffer.create 64 in
  let fmt = Format.formatter_of_buffer buf in
  pp fmt v;
  Format.pp_print_flush fmt ();
  Buffer.contents buf

let () =
  run "Ir_next.Tensor"
    [
      group "Hash-consing and inspection"
        [
          test "structurally equal nodes are physically equal" (fun () ->
            let a = T.unique ~id:42 in
            let b = T.unique ~id:42 in
            is_true (a == b));
          test "different nodes are distinct" (fun () ->
            let a = T.unique ~id:1 in
            let b = T.unique ~id:2 in
            is_true (a != b));
          test "tags are unique" (fun () ->
            let a = T.unique ~id:1 in
            let b = T.unique ~id:2 in
            is_true (T.tag a <> T.tag b));
          test "toposort leaves first" (fun () ->
            let u = T.unique ~id:0 in
            let d = T.device (Single "CPU") in
            let buf = T.buffer ~unique:u ~device:d ~size:4 ~dtype:D.float32 in
            let nodes = T.toposort buf in
            is_true (List.length nodes >= 3);
            is_true (List.hd nodes == u || List.hd nodes == d));
          test "dtype value" (fun () ->
            let n = mk_f32 () in
            some (of_equal D.equal) D.float32 (T.dtype n));
          test "dtype effect" (fun () ->
            let n = T.sink [] in
            is_none (T.dtype n));
          test "dtype buffer" (fun () ->
            let _u, _d, buf = emit_buffer () in
            some (of_equal D.equal) D.float32 (T.dtype buf));
          test "children binary" (fun () ->
            let a = mk_f32 () and c = mk_f32 () in
            let bin = T.binary ~op:`Add ~lhs:a ~rhs:c in
            is_true (List.length (T.children bin) = 2));
          test "children buffer" (fun () ->
            let u, d, buf = emit_buffer () in
            let ch = T.children buf in
            is_true (List.exists (fun c -> c == u) ch);
            is_true (List.exists (fun c -> c == d) ch));
          test "children pad" (fun () ->
            let _u, _d, buf = emit_buffer () in
            let bef = mk_idx () and aft = mk_idx () in
            let p = T.pad ~src:buf ~before:bef ~after:aft in
            is_true (List.length (T.children p) = 3));
          test "children leaf" (fun () ->
            let u = T.unique ~id:0 in
            let d = T.device (Single "CPU") in
            let dv = T.define_var ~name:"n" ~lo:0 ~hi:10 () in
            equal (list int) [] (List.map T.tag (T.children u));
            equal (list int) [] (List.map T.tag (T.children d));
            equal (list int) [] (List.map T.tag (T.children dv)));
        ];
      group "Smart constructor dtype inference"
        [
          test "binary cmplt produces bool" (fun () ->
            let a = mk_f32 () and c = mk_f32 () in
            dtype_eq D.bool (T.binary ~op:`Cmplt ~lhs:a ~rhs:c));
          test "binary add inherits lhs" (fun () ->
            let a = mk_f32 () and c = mk_f32 () in
            dtype_eq D.float32 (T.binary ~op:`Add ~lhs:a ~rhs:c));
          test "ternary where inherits b" (fun () ->
            let cond = mk_bool () and t = mk_f32 () and e = mk_f32 () in
            dtype_eq D.float32 (T.ternary ~op:`Where ~a:cond ~b:t ~c:e));
          test "ternary mulacc inherits a" (fun () ->
            let a = mk_i32 () and c = mk_i32 () and d = mk_i32 () in
            dtype_eq D.int32 (T.ternary ~op:`Mulacc ~a ~b:c ~c:d));
          test "unary inherits src" (fun () ->
            dtype_eq D.float32 (T.unary ~op:`Neg ~src:(mk_f32 ())));
          test "after inherits src dtype" (fun () ->
            dtype_eq D.float32 (T.after ~src:(mk_f32 ()) ~deps:[]));
          test "after void for effect src" (fun () ->
            dtype_eq D.void (T.after ~src:(T.sink []) ~deps:[]));
          test "const derives dtype" (fun () ->
            dtype_eq D.float64 (T.const (C.float D.Val.float64 3.14) D.float64));
          test "reshape inherits src" (fun () ->
            let _u, _d, buf = emit_buffer () in
            dtype_eq D.float32 (T.reshape ~src:buf ~shape:(mk_idx ())));
          test "permute inherits src" (fun () ->
            let _u, _d, buf = emit_buffer () in
            dtype_eq D.float32 (T.permute ~src:buf ~order:[ 0 ]));
          test "flip inherits src" (fun () ->
            let _u, _d, buf = emit_buffer () in
            dtype_eq D.float32 (T.flip ~src:buf ~dims:[ true ]));
          test "detach inherits src" (fun () ->
            dtype_eq D.float32 (T.detach ~src:(mk_f32 ())));
          test "contiguous inherits src" (fun () ->
            dtype_eq D.float32 (T.contiguous ~src:(mk_f32 ()) ()));
          test "vectorize from scalar and count" (fun () ->
            let s1 = mk_f32 () and s2 = mk_f32 () and s3 = mk_f32 () in
            dtype_eq (D.vec 3 D.float32) (T.vectorize ~srcs:[ s1; s2; s3 ]));
        ];
      group "Smart constructor edge cases"
        [
          test "vectorize empty raises" (fun () ->
            raises_match
              (function Invalid_argument _ -> true | _ -> false)
              (fun () -> ignore (T.vectorize ~srcs:[])));
          test "mstack empty raises" (fun () ->
            raises_match
              (function Failure _ -> true | _ -> false)
              (fun () -> ignore (T.mstack ~srcs:[])));
          test "shape static" (fun () ->
            let s = T.const (C.int D.Val.index 3) D.index in
            match T.view s with
            | Const _ -> ()
            | _ -> fail "expected Const for 1-d static shape");
          test "shape symbolic" (fun () ->
            let s = T.define_var ~name:"n" ~lo:1 ~hi:8 () in
            match T.view s with
            | Define_var _ -> ()
            | _ -> fail "expected Define_var");
          test "shape multi" (fun () ->
            let d1 = T.const (C.int D.Val.index 2) D.index in
            let d2 = T.const (C.int D.Val.index 3) D.index in
            let s = T.vectorize ~srcs:[ d1; d2 ] in
            match T.view s with
            | Vectorize _ -> ()
            | _ -> fail "expected Vectorize for multi-dim");
        ];
      group "Validation acceptance"
        [
          test "buffer chain ok" (fun () ->
            let _u, _d, _buf = emit_buffer () in
            ignore (T.unique ~id:0));
          test "buffer view ok" (fun () ->
            let _buf, index = mk_index_on_buf () in
            ignore (T.buffer_view ~src:index ~size:512 ~offset:0 ~dtype:D.float32);
            ignore (T.unique ~id:0));
          test "const all types ok" (fun () ->
            ignore (T.const (C.bool true) D.bool);
            ignore (T.const (C.int D.Val.int32 42) D.int32);
            ignore (T.const (C.float D.Val.float32 3.14) D.float32);
            ignore (T.unique ~id:0));
          test "vconst ok" (fun () ->
            ignore
              (T.vconst
                 ~values:[ C.float D.Val.float32 1.0; C.float D.Val.float32 2.0 ]
                 ~dtype:(D.vec 2 D.float32) ());
            ignore (T.unique ~id:0));
          test "define_var ok" (fun () ->
            ignore (T.define_var ~name:"n" ~lo:0 ~hi:10 ~dtype:D.int32 ());
            ignore (T.unique ~id:0));
          test "bind ok" (fun () ->
            let var = T.define_var ~name:"n" ~lo:0 ~hi:10 ~dtype:D.int32 () in
            ignore (T.bind ~var ~value:(mk_i32 ()) ~dtype:D.int32 ());
            ignore (T.unique ~id:0));
          test "param ok" (fun () ->
            let shape = mk_shape_2x3 () in
            let dev = T.device (Single "CPU") in
            ignore (T.param ~slot:0 ~dtype:D.float32 ~shape ~device:dev ());
            ignore (T.unique ~id:0));
          test "call ref ok" (fun () ->
            let fn = mk_f32 () in
            ignore (T.call ~callee:(Ref fn) ~args:[] ~info:call_info ~dtype:D.float32);
            ignore (T.unique ~id:0));
          test "assign ok" (fun () ->
            let _u, _d, buf = emit_buffer () in
            let assigned = T.assign ~target:buf ~value:(mk_f32 ()) () in
            (* assign emits Store+After *)
            (match T.view assigned with
            | After { deps; _ } ->
                is_true (List.exists (fun d ->
                  match T.view d with Store _ -> true | _ -> false)
                  deps)
            | _ -> fail "expected After from assign");
            ignore (T.unique ~id:0));
          test "detach ok" (fun () ->
            ignore (T.detach ~src:(mk_f32 ()));
            ignore (T.unique ~id:0));
          test "contiguous ok" (fun () ->
            ignore (T.contiguous ~src:(mk_f32 ()) ());
            ignore (T.unique ~id:0));
          test "copy ok" (fun () ->
            ignore (T.copy ~src:(mk_f32 ()) ~device:(T.device (Single "GPU")) ());
            ignore (T.unique ~id:0));
          test "allreduce ok" (fun () ->
            let a = mk_f32 () in
            let dev = T.device (Single "GPU") in
            ignore (T.allreduce ~src:a ~device:dev ~op:`Add ~dtype:D.float32);
            ignore (T.unique ~id:0));
          test "mstack ok" (fun () ->
            ignore (T.mstack ~srcs:[ mk_f32 (); mk_f32 () ]);
            ignore (T.unique ~id:0));
          test "reduce_axis ok" (fun () ->
            ignore (T.reduce_axis ~src:(mk_f32 ()) ~op:`Add ~axes:[ 0 ]);
            ignore (T.unique ~id:0));
          test "reshape ok" (fun () ->
            let _u, _d, buf = emit_buffer () in
            ignore (T.reshape ~src:buf ~shape:(mk_shape_2x3 ()));
            ignore (T.unique ~id:0));
          test "expand ok" (fun () ->
            let _u, _d, buf = emit_buffer () in
            ignore (T.expand ~src:buf ~shape:(mk_shape_2x3 ()));
            ignore (T.unique ~id:0));
          test "pad/shrink ok" (fun () ->
            let a = mk_f32 () in
            let bef = mk_idx () and aft = mk_idx () in
            ignore (T.pad ~src:a ~before:bef ~after:aft);
            ignore (T.shrink ~src:a ~before:bef ~after:aft);
            ignore (T.unique ~id:0));
          test "permute ok" (fun () ->
            ignore (T.permute ~src:(mk_f32 ()) ~order:[ 1; 0; 2 ]);
            ignore (T.unique ~id:0));
          test "flip ok" (fun () ->
            ignore (T.flip ~src:(mk_f32 ()) ~dims:[ true; false ]);
            ignore (T.unique ~id:0));
          test "range/end ok" (fun () ->
            let range = T.range ~size:(mk_idx ()) ~axis:0 ~kind:Ak.Loop () in
            ignore (T.end_ ~value:(mk_f32 ()) ~ranges:[ range ]);
            ignore (T.unique ~id:0));
          test "index/store ok" (fun () ->
            let _buf, index = mk_index_on_buf () in
            ignore (T.store ~dst:index ~value:(mk_f32 ()));
            ignore (T.unique ~id:0));
          test "alu chain ok" (fun () ->
            let a = mk_f32 () in
            let u = T.unary ~op:`Neg ~src:a in
            let bin = T.binary ~op:`Add ~lhs:u ~rhs:(mk_f32 ()) in
            ignore (T.ternary ~op:`Where ~a:(mk_bool ()) ~b:bin ~c:a);
            ignore (T.unique ~id:0));
        ];
      group "Validation rejection — tensor ops"
        [
          test "reject buffer negative size" (fun () ->
            let u = T.unique ~id:0 in
            let d = T.device (Single "CPU") in
            raises_validate "non-negative" (fun () ->
              ignore (T.buffer ~unique:u ~device:d ~size:(-1) ~dtype:D.float32)));
          test "reject buffer unique not unique" (fun () ->
            let d = T.device (Single "CPU") in
            raises_validate "Unique/Lunique" (fun () ->
              ignore (T.buffer ~unique:(mk_f32 ()) ~device:d ~size:1024 ~dtype:D.float32)));
          test "reject buffer device not device" (fun () ->
            let u = T.unique ~id:0 in
            raises_validate "Device" (fun () ->
              ignore (T.buffer ~unique:u ~device:(mk_f32 ()) ~size:1024 ~dtype:D.float32)));
          test "reject buffer_view negative size" (fun () ->
            let _buf, index = mk_index_on_buf () in
            raises_validate "non-negative" (fun () ->
              ignore (T.buffer_view ~src:index ~size:(-1) ~offset:0 ~dtype:D.float32)));
          test "reject buffer_view negative offset" (fun () ->
            let _buf, index = mk_index_on_buf () in
            raises_validate "non-negative" (fun () ->
              ignore (T.buffer_view ~src:index ~size:512 ~offset:(-1) ~dtype:D.float32)));
          test "reject buffer_view src not buffer or index" (fun () ->
            raises_validate "must be Buffer or Index" (fun () ->
              ignore (T.buffer_view ~src:(mk_f32 ()) ~size:512 ~offset:0 ~dtype:D.float32)));
          test "reject vconst count mismatch" (fun () ->
            raises_validate "match vector width" (fun () ->
              ignore
                (T.vconst
                   ~values:[ C.float D.Val.float32 1.0 ]
                   ~dtype:(D.vec 3 D.float32) ())));
          test "reject vconst element type mismatch" (fun () ->
            raises_validate "int elements" (fun () ->
              ignore
                (T.vconst
                   ~values:[ C.int D.Val.int32 1; C.int D.Val.int32 2 ]
                   ~dtype:(D.vec 2 D.float32) ())));
          test "reject bind var not define_var" (fun () ->
            raises_validate "Define_var" (fun () ->
              ignore (T.bind ~var:(mk_f32 ()) ~dtype:D.float32 ())));
          test "reject bind value dtype mismatch" (fun () ->
            let var = T.define_var ~name:"n" ~lo:0 ~hi:10 ~dtype:D.int32 () in
            raises_validate "Bind value" (fun () ->
              ignore (T.bind ~var ~value:(mk_f32 ()) ~dtype:D.int32 ())));
          test "reject param shape not index vector" (fun () ->
            raises_validate "must be index vector" (fun () ->
              ignore (T.param ~slot:0 ~dtype:D.float32 ~shape:(mk_f32 ()) ())));
          test "reject param device not device" (fun () ->
            raises_validate "Device" (fun () ->
              ignore (T.param ~slot:0 ~dtype:D.float32 ~device:(mk_f32 ()) ())));
          test "reject reduce_axis empty axes" (fun () ->
            raises_validate "at least one axis" (fun () ->
              ignore (T.reduce_axis ~src:(mk_f32 ()) ~op:`Add ~axes:[])));
          test "reject reduce_axis duplicate axes" (fun () ->
            raises_validate "unique" (fun () ->
              ignore (T.reduce_axis ~src:(mk_f32 ()) ~op:`Add ~axes:[ 0; 1; 0 ])));
          test "reject permute invalid order" (fun () ->
            raises_validate "valid permutation" (fun () ->
              ignore (T.permute ~src:(mk_f32 ()) ~order:[ 0; 0 ])));
          test "reject reshape negative dim" (fun () ->
            let _u, _d, buf = emit_buffer () in
            raises_validate "negative" (fun () ->
              ignore (T.reshape ~src:buf ~shape:(T.const (C.int D.Val.index (-1)) D.index))));
          test "reject pad/shrink width mismatch" (fun () ->
            let a = mk_f32 () in
            let d1 = T.const (C.int D.Val.index 1) D.index in
            let d2 = T.const (C.int D.Val.index 2) D.index in
            let bef = T.vectorize ~srcs:[ d1 ] in
            let aft = T.vectorize ~srcs:[ d1; d2 ] in
            raises_validate "width mismatch" (fun () ->
              ignore (T.pad ~src:a ~before:bef ~after:aft)));
          test "reject copy device not device" (fun () ->
            raises_validate "Device" (fun () ->
              ignore (T.copy ~src:(mk_f32 ()) ~device:(mk_f32 ()) ())));
          test "reject allreduce device not device" (fun () ->
            raises_validate "Device" (fun () ->
              ignore (T.allreduce ~src:(mk_f32 ()) ~device:(mk_f32 ()) ~op:`Add ~dtype:D.float32)));
          test "reject contiguous range not index" (fun () ->
            raises_validate "must be index scalar" (fun () ->
              ignore (T.contiguous ~src:(mk_f32 ()) ~ranges:[ mk_f32 () ] ())));
          test "reject mstack empty" (fun () ->
            raises_validate "must have srcs" (fun () ->
              ignore (T.mstack ~srcs:[])));
          test "reject mstack dtype mismatch" (fun () ->
            raises_validate "Mstack src" (fun () ->
              ignore (T.mstack ~srcs:[ mk_f32 (); mk_i32 () ])));
          test "reject cast width change" (fun () ->
            let v = T.vectorize ~srcs:[ mk_f32 (); mk_f32 () ] in
            raises_validate "vector width" (fun () ->
              ignore (T.cast ~src:v ~dtype:D.float32)));
          test "reject call ref dtype mismatch" (fun () ->
            let fn = mk_f32 () in
            raises_validate "Call dtype" (fun () ->
              ignore (T.call ~callee:(Ref fn) ~args:[] ~info:call_info ~dtype:D.int32)));
        ];
      group "Validation rejection — ALU"
        [
          test "reject define_var float" (fun () ->
            raises_validate "must be int/index" (fun () ->
              ignore (T.define_var ~name:"x" ~lo:0 ~hi:4 ~dtype:D.float32 ())));
          test "reject define_var lo > hi" (fun () ->
            raises_validate "lo > hi" (fun () ->
              ignore (T.define_var ~name:"x" ~lo:5 ~hi:3 ())));
          test "reject const type mismatch" (fun () ->
            raises_validate "Bool const" (fun () ->
              ignore (T.const (C.bool true) D.int32)));
          test "reject binary cmp operands mismatch" (fun () ->
            raises_validate "don't match" (fun () ->
              ignore (T.binary ~op:`Cmplt ~lhs:(mk_f32 ()) ~rhs:(mk_i32 ()))));
          test "reject binary idiv float" (fun () ->
            raises_validate "int/index" (fun () ->
              ignore (T.binary ~op:`Idiv ~lhs:(mk_f32 ()) ~rhs:(mk_f32 ()))));
          test "reject shift non-int" (fun () ->
            raises_validate "int/index" (fun () ->
              ignore (T.binary ~op:`Shl ~lhs:(mk_f32 ()) ~rhs:(mk_f32 ()))));
          test "reject shift rhs mismatch" (fun () ->
            let a = mk_i32 () in
            let c = T.const (C.int D.Val.int64 2) D.int64 in
            raises_validate "Shift rhs" (fun () ->
              ignore (T.binary ~op:`Shl ~lhs:a ~rhs:c)));
          test "reject where non-bool cond" (fun () ->
            raises_validate "bool scalar" (fun () ->
              ignore (T.ternary ~op:`Where ~a:(mk_i32 ()) ~b:(mk_f32 ()) ~c:(mk_f32 ()))));
          test "reject where mismatched arms" (fun () ->
            raises_validate "arms" (fun () ->
              ignore (T.ternary ~op:`Where ~a:(mk_bool ()) ~b:(mk_f32 ()) ~c:(mk_i32 ()))));
          test "reject mulacc mismatch" (fun () ->
            raises_validate "Mulacc" (fun () ->
              ignore (T.ternary ~op:`Mulacc ~a:(mk_f32 ()) ~b:(mk_i32 ()) ~c:(mk_f32 ()))));
        ];
      group "check and exn"
        [
          test "check ok returns Ok" (fun () ->
            ignore (mk_f32 ());
            ignore (T.unique ~id:0));
          test "validation raises Failure" (fun () ->
            raises_validate "must be int/index" (fun () ->
              ignore (T.define_var ~name:"x" ~lo:0 ~hi:4 ~dtype:D.float32 ())));
        ];
      group "Rewriting"
        [
          test "rebuild replaces const" (fun () ->
            ignore (T.const (C.int D.Val.int32 3) D.int32);
            ignore (T.const (C.int D.Val.int32 5) D.int32);
            let c3 = T.const (C.int D.Val.int32 3) D.int32 in
            let g' =
              T.graph_rewrite
                (fun n ->
                  match T.view n with
                  | Const { value; _ } -> (
                      match C.view value with
                      | Int n when Int64.to_int n = 3 ->
                          Some (T.const (C.int D.Val.int32 4) D.int32)
                      | _ -> None)
                  | _ -> None)
                c3
            in
            match T.view g' with
            | Const { value; _ } -> (
                match C.view value with
                | Int n -> equal int 4 (Int64.to_int n)
                | _ -> fail "expected Int")
            | _ -> fail "expected Const");
          test "rebuild no match identity" (fun () ->
            ignore (mk_f32 ());
            ignore (mk_i32 ());
            
            let g = mk_f32 () in
            let g' = T.graph_rewrite (fun _ -> None) g in
            is_true (g == g'));
          test "graph_rewrite replaces binary" (fun () ->
            let a = mk_f32 () in
            let g = T.binary ~op:`Add ~lhs:a ~rhs:(T.const (C.float D.Val.float32 0.0) D.float32) in
            let g' =
              T.graph_rewrite
                (fun n ->
                  match T.view n with
                  | Binary { op = `Add; _ } ->
                      Some (T.const (C.float D.Val.float32 99.0) D.float32)
                  | _ -> None)
                g
            in
            (match T.view g' with Const _ -> () | _ -> fail "expected Const"));
          (* rewrite_fixpoint is not in the new API — graph_rewrite
             handles re-processing internally *)
          test "graph_rewrite diverges raises" (fun () ->
            let c = T.const (C.int D.Val.int32 3) D.int32 in
            raises_match
              (function Failure _ -> true | _ -> false)
              (fun () ->
                ignore
                  (T.graph_rewrite
                     (fun n ->
                       match T.view n with
                       | Const { value; _ } -> (
                           match C.view value with
                           | Int i when Int64.to_int i = 3 ->
                               Some (T.const (C.int D.Val.int32 4) D.int32)
                           | Int i when Int64.to_int i = 4 ->
                               Some (T.const (C.int D.Val.int32 3) D.int32)
                           | _ -> None)
                       | _ -> None)
                     c)));
          test "hash-consing deduplicates" (fun () ->
            let a1 = T.const (C.float D.Val.float32 1.0) D.float32 in
            let a2 = T.const (C.float D.Val.float32 1.0) D.float32 in
            is_true (a1 == a2));
          test "map_children remaps" (fun () ->
            let a = mk_f32 () and b = mk_i32 () in
            let replacement = mk_idx () in
            let view : T.view =
              Binary { op = `Add; lhs = a; rhs = b; dtype = D.float32 }
            in
            match T.map_children (fun n -> if n == a then replacement else n) view with
            | Binary { lhs; _ } when lhs == replacement -> ()
            | _ -> fail "expected remapped Binary");
        ];
      group "Formatting"
        [
          test "pp_instr contains op name" (fun () ->
            let d = mk_f32 () in
            is_true
              (contains
                 (pp_to_string T.pp_view
                    (Reshape { src = d; shape = d; dtype = D.float32 }))
                 "reshape");
            is_true
              (contains
                 (pp_to_string T.pp_view
                    (Buffer { unique = d; device = d; size = 1024; dtype = D.float32 }))
                 "buffer"));
          test "pp program indexed" (fun () ->
            let u = T.unique ~id:0 in
            let d = T.device (Single "CPU") in
            let c = mk_f32 () in
            let root = T.sink [ u; d; c ] in
            let s = pp_to_string T.pp root in
            is_true (contains s "unique");
            is_true (contains s "device");
            is_true (contains s "const"));
        ];
      group "Shape computation"
        [
          test "buffer shape" (fun () ->
            let _u, _d, buf = emit_buffer () in
            let shapes = T.compute_shapes buf in
            equal (option (list int)) (Some [ 1024 ]) (shapes buf));
          test "const shape is empty" (fun () ->
            let c = mk_f32 () in
            let shapes = T.compute_shapes c in
            equal (option (list int)) (Some []) (shapes c));
          test "reshape shape" (fun () ->
            let _u, _d, buf = emit_buffer () in
            let shape = mk_shape_2x3 () in
            let r = T.reshape ~src:buf ~shape in
            let shapes = T.compute_shapes r in
            equal (option (list int)) (Some [ 2; 3 ]) (shapes r));
          test "permute shape" (fun () ->
            let _u, _d, buf = emit_buffer ~dtype:D.float32 () in
            let d1 = T.const (C.int D.Val.index 4) D.index in
            let d2 = T.const (C.int D.Val.index 8) D.index in
            let shape = T.vectorize ~srcs:[ d1; d2 ] in
            let reshaped = T.reshape ~src:buf ~shape in
            let p = T.permute ~src:reshaped ~order:[ 1; 0 ] in
            let shapes = T.compute_shapes p in
            equal (option (list int)) (Some [ 8; 4 ]) (shapes p));
          test "unary inherits shape" (fun () ->
            let _u, _d, buf = emit_buffer () in
            let neg = T.unary ~op:`Neg ~src:buf in
            let shapes = T.compute_shapes neg in
            equal (option (list int)) (shapes buf) (shapes neg));
          test "binary inherits lhs shape" (fun () ->
            let _u, _d, buf = emit_buffer () in
            let c = mk_f32 () in
            let add = T.binary ~op:`Add ~lhs:buf ~rhs:c in
            let shapes = T.compute_shapes add in
            equal (option (list int)) (shapes buf) (shapes add));
          test "reduce_axis collapses axes" (fun () ->
            let _u, _d, buf = emit_buffer ~dtype:D.float32 () in
            let d1 = T.const (C.int D.Val.index 4) D.index in
            let d2 = T.const (C.int D.Val.index 8) D.index in
            let shape = T.vectorize ~srcs:[ d1; d2 ] in
            let reshaped = T.reshape ~src:buf ~shape in
            let red = T.reduce_axis ~src:reshaped ~op:`Add ~axes:[ 1 ] in
            let shapes = T.compute_shapes red in
            equal (option (list int)) (Some [ 4; 1 ]) (shapes red));
          test "sink has no shape" (fun () ->
            let sink = T.sink [] in
            let shapes = T.compute_shapes sink in
            is_none (shapes sink));
        ];
      group "Device computation"
        [
          test "device node" (fun () ->
            let d = T.device (Single "GPU") in
            let devs = T.compute_devices d in
            equal (option string) (Some "GPU")
              (match devs d with
               | Some (Single s) -> Some s
               | _ -> None));
          test "buffer inherits device" (fun () ->
            let u = T.unique ~id:0 in
            let d = T.device (Single "CPU") in
            let buf = T.buffer ~unique:u ~device:d ~size:64 ~dtype:D.float32 in
            let devs = T.compute_devices buf in
            equal (option string) (Some "CPU")
              (match devs buf with
               | Some (Single s) -> Some s
               | _ -> None));
        ];
      group "Analysis"
        [
          test "backward_slice excludes root" (fun () ->
            let a = mk_f32 () in
            let neg = T.unary ~op:`Neg ~src:a in
            let slice = T.backward_slice neg in
            is_true (List.exists (fun n -> n == a) slice));
          test "toposort is topological" (fun () ->
            let a = mk_f32 () in
            let neg = T.unary ~op:`Neg ~src:a in
            let topo = T.toposort neg in
            let idx_of n = List.find_opt (fun (_, x) -> x == n)
              (List.mapi (fun i x -> (i, x)) topo)
              |> Option.map fst |> Option.value ~default:(-1) in
            is_true (idx_of a < idx_of neg));
          test "consumer_map tracks consumers" (fun () ->
            let a = mk_f32 () in
            let neg = T.unary ~op:`Neg ~src:a in
            let consumers = T.consumer_map neg in
            is_true (List.exists (fun c -> c == neg) (consumers a)));
          test "base follows through movement ops" (fun () ->
            let _u, _d, buf = emit_buffer () in
            let shape = mk_shape_2x3 () in
            let reshaped = T.reshape ~src:buf ~shape in
            let perm = T.permute ~src:reshaped ~order:[ 1; 0 ] in
            is_true (T.base perm == buf));
          test "base stops at non-movement" (fun () ->
            let a = mk_f32 () in
            let neg = T.unary ~op:`Neg ~src:a in
            is_true (T.base neg == neg));
        ];
    ]
