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

let mk_f32 b = T.const b (C.float D.float32 1.0)
let mk_i32 b = T.const b (C.int D.int32 0)
let mk_idx b = T.const b (C.int D.index 0)
let mk_bool b = T.const b (C.bool true)

let emit_buffer ?(dtype = D.float32) b =
  let u = T.unique b ~id:0 in
  let d = T.device b (Single "CPU") in
  let buf = T.buffer b ~unique:u ~device:d ~size:1024 ~dtype in
  (u, d, buf)

let mk_shape_2x3 b =
  let d1 = T.const b (C.int D.index 2) in
  let d2 = T.const b (C.int D.index 3) in
  T.emit b (Vectorize { srcs = [ d1; d2 ]; dtype = D.vec D.index 2 })

let dtype_eq expected b id =
  match T.dtype (T.finish b) id with
  | Some dt -> is_true (D.equal dt expected)
  | None -> fail "expected a dtype but got None"

let mk_index_on_buf b =
  let _u, _d, buf = emit_buffer b in
  let idx = mk_idx b in
  (buf, T.index b ~ptr:buf ~idxs:[ idx ] ~dtype:D.float32 ())

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
      group "Builder and inspection"
        [
          test "empty program" (fun () ->
            equal int 0 (T.length (T.finish (T.create ()))));
          test "emit sequential ids" (fun () ->
            let b = T.create () in
            let ids = List.init 5 (fun _ -> T.emit b (T.Unique { id = 0 })) in
            equal (list int) [ 0; 1; 2; 3; 4 ] ids);
          test "finish preserves order" (fun () ->
            let b = T.create () in
            ignore (T.unique b ~id:0);
            ignore (T.device b (Single "CPU"));
            ignore (mk_f32 b);
            let g = T.finish b in
            (match T.view g 0 with Unique _ -> () | _ -> fail "expected Unique");
            (match T.view g 1 with Device _ -> () | _ -> fail "expected Device");
            (match T.view g 2 with Const _ -> () | _ -> fail "expected Const"));
          test "reallocation" (fun () ->
            let b = T.create () in
            for i = 0 to 31 do ignore (T.unique b ~id:i) done;
            equal int 32 (T.length (T.finish b)));
          test "finish is snapshot" (fun () ->
            let b = T.create () in
            ignore (T.unique b ~id:0);
            ignore (T.unique b ~id:1);
            let g1 = T.finish b in
            ignore (T.unique b ~id:2);
            let g2 = T.finish b in
            equal int 2 (T.length g1);
            equal int 3 (T.length g2));
          test "dtype value" (fun () ->
            let b = T.create () in
            ignore (mk_f32 b);
            some (of_equal D.equal) D.float32 (T.dtype (T.finish b) 0));
          test "dtype effect" (fun () ->
            let b = T.create () in
            ignore (T.sink b []);
            is_none (T.dtype (T.finish b) 0));
          test "dtype buffer" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            some (of_equal D.equal) D.float32 (T.dtype (T.finish b) buf));
          test "children binary" (fun () ->
            let b = T.create () in
            let a = mk_f32 b and c = mk_f32 b in
            let bin = T.binary b ~op:`Add ~lhs:a ~rhs:c in
            equal (list int) [ a; c ] (T.children (T.finish b) bin));
          test "children buffer" (fun () ->
            let b = T.create () in
            let u, d, buf = emit_buffer b in
            equal (list int) [ u; d ] (T.children (T.finish b) buf));
          test "children pad" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            let bef = mk_idx b and aft = mk_idx b in
            let p = T.pad b ~src:buf ~before:bef ~after:aft in
            equal (list int) [ buf; bef; aft ] (T.children (T.finish b) p));
          test "children leaf" (fun () ->
            let b = T.create () in
            ignore (T.unique b ~id:0);
            ignore (T.device b (Single "CPU"));
            ignore (T.define_var b ~name:"n" ~lo:0 ~hi:10 ());
            let g = T.finish b in
            equal (list int) [] (T.children g 0);
            equal (list int) [] (T.children g 1);
            equal (list int) [] (T.children g 2));
        ];
      group "Smart constructor dtype inference"
        [
          test "binary cmplt produces bool" (fun () ->
            let b = T.create () in
            let a = mk_f32 b and c = mk_f32 b in
            dtype_eq D.bool b (T.binary b ~op:`Cmplt ~lhs:a ~rhs:c));
          test "binary add inherits lhs" (fun () ->
            let b = T.create () in
            let a = mk_f32 b and c = mk_f32 b in
            dtype_eq D.float32 b (T.binary b ~op:`Add ~lhs:a ~rhs:c));
          test "ternary where inherits b" (fun () ->
            let b = T.create () in
            let cond = mk_bool b and t = mk_f32 b and e = mk_f32 b in
            dtype_eq D.float32 b (T.ternary b ~op:`Where ~a:cond ~b:t ~c:e));
          test "ternary mulacc inherits a" (fun () ->
            let b = T.create () in
            let a = mk_i32 b and c = mk_i32 b and d = mk_i32 b in
            dtype_eq D.int32 b (T.ternary b ~op:`Mulacc ~a ~b:c ~c:d));
          test "unary inherits src" (fun () ->
            let b = T.create () in
            dtype_eq D.float32 b (T.unary b ~op:`Neg ~src:(mk_f32 b)));
          test "after inherits src dtype" (fun () ->
            let b = T.create () in
            dtype_eq D.float32 b (T.after b ~src:(mk_f32 b) ~deps:[]));
          test "after void for effect src" (fun () ->
            let b = T.create () in
            dtype_eq D.void b (T.after b ~src:(T.sink b []) ~deps:[]));
          test "const derives dtype" (fun () ->
            let b = T.create () in
            dtype_eq D.float64 b (T.const b (C.float D.float64 3.14)));
          test "reshape inherits src" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            dtype_eq D.float32 b (T.reshape b ~src:buf ~shape:(mk_idx b)));
          test "permute inherits src" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            dtype_eq D.float32 b (T.permute b ~src:buf ~order:[ 0 ]));
          test "flip inherits src" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            dtype_eq D.float32 b (T.flip b ~src:buf ~dims:[ true ]));
          test "detach inherits src" (fun () ->
            let b = T.create () in
            dtype_eq D.float32 b (T.detach b ~src:(mk_f32 b)));
          test "contiguous inherits src" (fun () ->
            let b = T.create () in
            dtype_eq D.float32 b (T.contiguous b ~src:(mk_f32 b) ()));
          test "vectorize from scalar and count" (fun () ->
            let b = T.create () in
            let s1 = mk_f32 b and s2 = mk_f32 b and s3 = mk_f32 b in
            dtype_eq (D.vec D.float32 3) b (T.vectorize b ~srcs:[ s1; s2; s3 ]));
        ];
      group "Smart constructor edge cases"
        [
          test "vectorize empty raises" (fun () ->
            let b = T.create () in
            raises_match
              (function Invalid_argument _ -> true | _ -> false)
              (fun () -> ignore (T.vectorize b ~srcs:[])));
          test "mstack empty raises" (fun () ->
            let b = T.create () in
            raises_match
              (function Invalid_argument _ -> true | _ -> false)
              (fun () -> ignore (T.mstack b ~srcs:[])));
          test "shape static" (fun () ->
            let b = T.create () in
            let s = T.shape b (Sh.of_dims [ 3 ]) in
            match T.view (T.finish b) s with
            | Const _ -> ()
            | _ -> fail "expected Const for 1-d static shape");
          test "shape symbolic" (fun () ->
            let b = T.create () in
            let s =
              T.shape b (Sh.of_dim_list [ Sh.Symbol { name = "n"; lo = 1; hi = 8 } ])
            in
            match T.view (T.finish b) s with
            | Define_var _ -> ()
            | _ -> fail "expected Define_var");
          test "shape multi" (fun () ->
            let b = T.create () in
            let s = T.shape b (Sh.of_dims [ 2; 3 ]) in
            match T.view (T.finish b) s with
            | Vectorize _ -> ()
            | _ -> fail "expected Vectorize for multi-dim");
        ];
      group "Validation acceptance"
        [
          test "buffer chain ok" (fun () ->
            let b = T.create () in
            let _u, _d, _buf = emit_buffer b in
            T.validate (T.finish b));
          test "buffer view ok" (fun () ->
            let b = T.create () in
            let _buf, index = mk_index_on_buf b in
            ignore (T.buffer_view b ~src:index ~size:512 ~offset:0 ~dtype:D.float32);
            T.validate (T.finish b));
          test "const all types ok" (fun () ->
            let b = T.create () in
            ignore (T.const b (C.bool true));
            ignore (T.const b (C.int D.int32 42));
            ignore (T.const b (C.float D.float32 3.14));
            T.validate (T.finish b));
          test "vconst ok" (fun () ->
            let b = T.create () in
            ignore
              (T.vconst b
                 ~values:[ C.float D.float32 1.0; C.float D.float32 2.0 ]
                 ~dtype:(D.vec D.float32 2) ());
            T.validate (T.finish b));
          test "define_var ok" (fun () ->
            let b = T.create () in
            ignore (T.define_var b ~name:"n" ~lo:0 ~hi:10 ~dtype:D.int32 ());
            T.validate (T.finish b));
          test "bind ok" (fun () ->
            let b = T.create () in
            let var = T.define_var b ~name:"n" ~lo:0 ~hi:10 ~dtype:D.int32 () in
            ignore (T.bind b ~var ~value:(mk_i32 b) ());
            T.validate (T.finish b));
          test "param ok" (fun () ->
            let b = T.create () in
            let shape = mk_shape_2x3 b in
            let dev = T.device b (Single "CPU") in
            ignore (T.param b ~slot:0 ~dtype:D.float32 ~shape ~device:dev ());
            T.validate (T.finish b));
          test "call ref ok" (fun () ->
            let b = T.create () in
            let fn = mk_f32 b in
            ignore (T.call b ~callee:(Ref fn) ~args:[] ~info:call_info ~dtype:D.float32);
            T.validate (T.finish b));
          test "assign ok" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            let assigned = T.assign b ~target:buf ~value:(mk_f32 b) () in
            (* assign emits Store+After *)
            (match T.view (T.finish b) assigned with
            | After { deps; _ } ->
                is_true (List.exists (fun d ->
                  match T.view (T.finish b) d with Store _ -> true | _ -> false)
                  deps)
            | _ -> fail "expected After from assign");
            T.validate (T.finish b));
          test "detach ok" (fun () ->
            let b = T.create () in
            ignore (T.detach b ~src:(mk_f32 b));
            T.validate (T.finish b));
          test "contiguous ok" (fun () ->
            let b = T.create () in
            ignore (T.contiguous b ~src:(mk_f32 b) ());
            T.validate (T.finish b));
          test "copy ok" (fun () ->
            let b = T.create () in
            ignore (T.copy b ~src:(mk_f32 b) ~device:(T.device b (Single "GPU")) ());
            T.validate (T.finish b));
          test "allreduce ok" (fun () ->
            let b = T.create () in
            let a = mk_f32 b in
            let dev = T.device b (Single "GPU") in
            ignore (T.allreduce b ~src:a ~device:dev ~op:`Add);
            T.validate (T.finish b));
          test "mstack ok" (fun () ->
            let b = T.create () in
            ignore (T.mstack b ~srcs:[ mk_f32 b; mk_f32 b ]);
            T.validate (T.finish b));
          test "reduce_axis ok" (fun () ->
            let b = T.create () in
            ignore (T.reduce_axis b ~src:(mk_f32 b) ~op:`Add ~axes:[ 0 ]);
            T.validate (T.finish b));
          test "reshape ok" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            ignore (T.reshape b ~src:buf ~shape:(mk_shape_2x3 b));
            T.validate (T.finish b));
          test "expand ok" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            ignore (T.expand b ~src:buf ~shape:(mk_shape_2x3 b));
            T.validate (T.finish b));
          test "pad/shrink ok" (fun () ->
            let b = T.create () in
            let a = mk_f32 b in
            let bef = mk_idx b and aft = mk_idx b in
            ignore (T.pad b ~src:a ~before:bef ~after:aft);
            ignore (T.shrink b ~src:a ~before:bef ~after:aft);
            T.validate (T.finish b));
          test "permute ok" (fun () ->
            let b = T.create () in
            ignore (T.permute b ~src:(mk_f32 b) ~order:[ 1; 0; 2 ]);
            T.validate (T.finish b));
          test "flip ok" (fun () ->
            let b = T.create () in
            ignore (T.flip b ~src:(mk_f32 b) ~dims:[ true; false ]);
            T.validate (T.finish b));
          test "range/end ok" (fun () ->
            let b = T.create () in
            let range = T.range b ~size:(mk_idx b) ~axis:0 ~kind:Ak.Loop () in
            ignore (T.end_ b ~value:(mk_f32 b) ~ranges:[ range ]);
            T.validate (T.finish b));
          test "index/store ok" (fun () ->
            let b = T.create () in
            let _buf, index = mk_index_on_buf b in
            ignore (T.store b ~dst:index ~value:(mk_f32 b));
            T.validate (T.finish b));
          test "alu chain ok" (fun () ->
            let b = T.create () in
            let a = mk_f32 b in
            let u = T.unary b ~op:`Neg ~src:a in
            let bin = T.binary b ~op:`Add ~lhs:u ~rhs:(mk_f32 b) in
            ignore (T.ternary b ~op:`Where ~a:(mk_bool b) ~b:bin ~c:a);
            T.validate (T.finish b));
        ];
      group "Validation rejection — tensor ops"
        [
          test "reject buffer negative size" (fun () ->
            let b = T.create () in
            let u = T.unique b ~id:0 in
            let d = T.device b (Single "CPU") in
            ignore (T.buffer b ~unique:u ~device:d ~size:(-1) ~dtype:D.float32);
            raises_validate "non-negative" (fun () -> T.validate (T.finish b)));
          test "reject buffer unique not unique" (fun () ->
            let b = T.create () in
            let d = T.device b (Single "CPU") in
            ignore (T.buffer b ~unique:(mk_f32 b) ~device:d ~size:1024 ~dtype:D.float32);
            raises_validate "Unique/Lunique" (fun () -> T.validate (T.finish b)));
          test "reject buffer device not device" (fun () ->
            let b = T.create () in
            let u = T.unique b ~id:0 in
            ignore (T.buffer b ~unique:u ~device:(mk_f32 b) ~size:1024 ~dtype:D.float32);
            raises_validate "Device" (fun () -> T.validate (T.finish b)));
          test "reject buffer_view negative size" (fun () ->
            let b = T.create () in
            let _buf, index = mk_index_on_buf b in
            ignore (T.buffer_view b ~src:index ~size:(-1) ~offset:0 ~dtype:D.float32);
            raises_validate "non-negative" (fun () -> T.validate (T.finish b)));
          test "reject buffer_view negative offset" (fun () ->
            let b = T.create () in
            let _buf, index = mk_index_on_buf b in
            ignore (T.buffer_view b ~src:index ~size:512 ~offset:(-1) ~dtype:D.float32);
            raises_validate "non-negative" (fun () -> T.validate (T.finish b)));
          test "reject buffer_view src not buffer or index" (fun () ->
            let b = T.create () in
            ignore (T.buffer_view b ~src:(mk_f32 b) ~size:512 ~offset:0 ~dtype:D.float32);
            raises_validate "must be Buffer or Index" (fun () -> T.validate (T.finish b)));
          test "reject vconst count mismatch" (fun () ->
            let b = T.create () in
            ignore
              (T.vconst b
                 ~values:[ C.float D.float32 1.0 ]
                 ~dtype:(D.vec D.float32 3) ());
            raises_validate "match vector width" (fun () -> T.validate (T.finish b)));
          test "reject vconst element type mismatch" (fun () ->
            let b = T.create () in
            ignore
              (T.vconst b
                 ~values:[ C.int D.int32 1; C.int D.int32 2 ]
                 ~dtype:(D.vec D.float32 2) ());
            raises_validate "int elements" (fun () -> T.validate (T.finish b)));
          test "reject bind var not define_var" (fun () ->
            let b = T.create () in
            ignore (T.emit b (Bind { var = mk_f32 b; value = None; dtype = D.float32 }));
            raises_validate "Define_var" (fun () -> T.validate (T.finish b)));
          test "reject bind value dtype mismatch" (fun () ->
            let b = T.create () in
            let var = T.define_var b ~name:"n" ~lo:0 ~hi:10 ~dtype:D.int32 () in
            ignore (T.emit b (Bind { var; value = Some (mk_f32 b); dtype = D.int32 }));
            raises_validate "Bind value" (fun () -> T.validate (T.finish b)));
          test "reject param shape not index vector" (fun () ->
            let b = T.create () in
            ignore (T.param b ~slot:0 ~dtype:D.float32 ~shape:(mk_f32 b) ());
            raises_validate "must be index vector" (fun () -> T.validate (T.finish b)));
          test "reject param device not device" (fun () ->
            let b = T.create () in
            ignore (T.param b ~slot:0 ~dtype:D.float32 ~device:(mk_f32 b) ());
            raises_validate "Device" (fun () -> T.validate (T.finish b)));
          test "reject reduce_axis empty axes" (fun () ->
            let b = T.create () in
            ignore (T.reduce_axis b ~src:(mk_f32 b) ~op:`Add ~axes:[]);
            raises_validate "at least one axis" (fun () -> T.validate (T.finish b)));
          test "reject reduce_axis duplicate axes" (fun () ->
            let b = T.create () in
            ignore
              (T.reduce_axis b ~src:(mk_f32 b) ~op:`Add ~axes:[ 0; 1; 0 ]);
            raises_validate "unique" (fun () -> T.validate (T.finish b)));
          test "reject permute invalid order" (fun () ->
            let b = T.create () in
            ignore (T.permute b ~src:(mk_f32 b) ~order:[ 0; 0 ]);
            raises_validate "valid permutation" (fun () -> T.validate (T.finish b)));
          test "reject reshape negative dim" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            ignore (T.reshape b ~src:buf ~shape:(T.const b (C.int D.index (-1))));
            raises_validate "negative" (fun () -> T.validate (T.finish b)));
          test "reject pad/shrink width mismatch" (fun () ->
            let b = T.create () in
            let a = mk_f32 b in
            let d1 = T.const b (C.int D.index 1) in
            let d2 = T.const b (C.int D.index 2) in
            let bef =
              T.emit b (Vectorize { srcs = [ d1 ]; dtype = D.vec D.index 1 })
            in
            let aft =
              T.emit b (Vectorize { srcs = [ d1; d2 ]; dtype = D.vec D.index 2 })
            in
            ignore (T.pad b ~src:a ~before:bef ~after:aft);
            raises_validate "width mismatch" (fun () -> T.validate (T.finish b)));
          test "reject copy device not device" (fun () ->
            let b = T.create () in
            ignore (T.copy b ~src:(mk_f32 b) ~device:(mk_f32 b) ());
            raises_validate "Device" (fun () -> T.validate (T.finish b)));
          test "reject allreduce device not device" (fun () ->
            let b = T.create () in
            let a = mk_f32 b in
            ignore (T.allreduce b ~src:a ~device:(mk_f32 b) ~op:`Add);
            raises_validate "Device" (fun () -> T.validate (T.finish b)));
          test "reject contiguous range not index" (fun () ->
            let b = T.create () in
            ignore (T.contiguous b ~src:(mk_f32 b) ~ranges:[ mk_f32 b ] ());
            raises_validate "must be index scalar" (fun () -> T.validate (T.finish b)));
          test "reject mstack empty" (fun () ->
            let b = T.create () in
            ignore (T.emit b (Mstack { srcs = []; dtype = D.float32 }));
            raises_validate "must have srcs" (fun () -> T.validate (T.finish b)));
          test "reject mstack dtype mismatch" (fun () ->
            let b = T.create () in
            ignore
              (T.emit b (Mstack { srcs = [ mk_f32 b; mk_i32 b ]; dtype = D.float32 }));
            raises_validate "Mstack src" (fun () -> T.validate (T.finish b)));
          test "reject cast width change" (fun () ->
            let b = T.create () in
            let v = T.vectorize b ~srcs:[ mk_f32 b; mk_f32 b ] in
            ignore (T.cast b ~src:v ~dtype:D.float32);
            raises_validate "vector width" (fun () -> T.validate (T.finish b)));
          test "reject after dtype mismatch" (fun () ->
            let b = T.create () in
            ignore (T.emit b (After { src = mk_f32 b; deps = []; dtype = D.int32 }));
            raises_validate "After dtype" (fun () -> T.validate (T.finish b)));
          test "reject detach dtype mismatch" (fun () ->
            let b = T.create () in
            ignore (T.emit b (Detach { src = mk_f32 b; dtype = D.int32 }));
            raises_validate "dtype" (fun () -> T.validate (T.finish b)));
          test "reject store dtype mismatch" (fun () ->
            let b = T.create () in
            ignore (T.store b ~dst:(mk_f32 b) ~value:(mk_i32 b));
            raises_validate "Store value" (fun () -> T.validate (T.finish b)));
          test "reject call ref dtype mismatch" (fun () ->
            let b = T.create () in
            let fn = mk_f32 b in
            ignore (T.call b ~callee:(Ref fn) ~args:[] ~info:call_info ~dtype:D.int32);
            raises_validate "Call dtype" (fun () -> T.validate (T.finish b)));
        ];
      group "Validation rejection — ALU"
        [
          test "reject forward ref" (fun () ->
            let b = T.create () in
            ignore (T.emit b (Unary { op = `Neg; src = 1; dtype = D.float32 }));
            ignore (mk_f32 b);
            raises_validate "out of bounds or forward" (fun () ->
              T.validate (T.finish b)));
          test "reject define_var float" (fun () ->
            let b = T.create () in
            ignore (T.define_var b ~name:"x" ~lo:0 ~hi:4 ~dtype:D.float32 ());
            raises_validate "must be int/index" (fun () -> T.validate (T.finish b)));
          test "reject define_var lo > hi" (fun () ->
            let b = T.create () in
            ignore (T.define_var b ~name:"x" ~lo:5 ~hi:3 ());
            raises_validate "lo > hi" (fun () -> T.validate (T.finish b)));
          test "reject const type mismatch" (fun () ->
            let b = T.create () in
            ignore (T.emit b (Const { value = C.bool true; dtype = D.int32; srcs = [] }));
            raises_validate "Bool const" (fun () -> T.validate (T.finish b)));
          test "reject binary cmp operands mismatch" (fun () ->
            let b = T.create () in
            let a = mk_f32 b and c = mk_i32 b in
            ignore (T.emit b (Binary { op = `Cmplt; lhs = a; rhs = c; dtype = D.bool }));
            raises_validate "don't match" (fun () -> T.validate (T.finish b)));
          test "reject binary cmp non-bool result" (fun () ->
            let b = T.create () in
            let a = mk_f32 b and c = mk_f32 b in
            ignore (T.emit b (Binary { op = `Cmplt; lhs = a; rhs = c; dtype = D.int32 }));
            raises_validate "bool" (fun () -> T.validate (T.finish b)));
          test "reject binary idiv float" (fun () ->
            let b = T.create () in
            let a = mk_f32 b and c = mk_f32 b in
            ignore (T.emit b (Binary { op = `Idiv; lhs = a; rhs = c; dtype = D.float32 }));
            raises_validate "int/index" (fun () -> T.validate (T.finish b)));
          test "reject shift non-int" (fun () ->
            let b = T.create () in
            let a = mk_f32 b and c = mk_f32 b in
            ignore (T.emit b (Binary { op = `Shl; lhs = a; rhs = c; dtype = D.float32 }));
            raises_validate "int/index" (fun () -> T.validate (T.finish b)));
          test "reject shift rhs mismatch" (fun () ->
            let b = T.create () in
            let a = mk_i32 b in
            let c = T.const b (C.int D.int64 2) in
            ignore (T.emit b (Binary { op = `Shl; lhs = a; rhs = c; dtype = D.int32 }));
            raises_validate "shift rhs" (fun () -> T.validate (T.finish b)));
          test "reject unary dtype mismatch" (fun () ->
            let b = T.create () in
            ignore (T.emit b (Unary { op = `Neg; src = mk_f32 b; dtype = D.int32 }));
            raises_validate "Unary operand" (fun () -> T.validate (T.finish b)));
          test "reject where non-bool cond" (fun () ->
            let b = T.create () in
            let a = mk_i32 b and t = mk_f32 b and e = mk_f32 b in
            ignore
              (T.emit b (Ternary { op = `Where; a; b = t; c = e; dtype = D.float32 }));
            raises_validate "bool scalar" (fun () -> T.validate (T.finish b)));
          test "reject where mismatched arms" (fun () ->
            let b = T.create () in
            let cond = mk_bool b and t = mk_f32 b and e = mk_i32 b in
            ignore
              (T.emit b
                 (Ternary { op = `Where; a = cond; b = t; c = e; dtype = D.float32 }));
            raises_validate "arms" (fun () -> T.validate (T.finish b)));
          test "reject mulacc mismatch" (fun () ->
            let b = T.create () in
            let a = mk_f32 b and c = mk_i32 b and d = mk_f32 b in
            ignore
              (T.emit b (Ternary { op = `Mulacc; a; b = c; c = d; dtype = D.float32 }));
            raises_validate "Mulacc" (fun () -> T.validate (T.finish b)));
          test "reject vectorize count mismatch" (fun () ->
            let b = T.create () in
            let a = mk_f32 b and c = mk_f32 b in
            ignore (T.emit b (Vectorize { srcs = [ a; c ]; dtype = D.vec D.float32 3 }));
            raises_validate "count" (fun () -> T.validate (T.finish b)));
          test "reject index empty idxs" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            ignore (T.index b ~ptr:buf ~idxs:[] ~dtype:D.float32 ());
            raises_validate "at least one index" (fun () -> T.validate (T.finish b)));
        ];
      group "check and exn"
        [
          test "check ok returns Ok" (fun () ->
            let b = T.create () in
            ignore (mk_f32 b);
            T.validate (T.finish b));
          test "check error returns Error" (fun () ->
            let b = T.create () in
            ignore (T.emit b (Unary { op = `Neg; src = 1; dtype = D.float32 }));
            ignore (mk_f32 b);
            raises_validate "out of bounds or forward" (fun () ->
              T.validate (T.finish b)));
          test "exn raises Failure" (fun () ->
            let b = T.create () in
            ignore (T.emit b (Unary { op = `Neg; src = 1; dtype = D.float32 }));
            ignore (mk_f32 b);
            raises_validate "out of bounds or forward" (fun () ->
              T.validate (T.finish b)));
        ];
      group "Rewriting"
        [
          test "rebuild replaces const" (fun () ->
            let b = T.create () in
            ignore (T.const b (C.int D.int32 3));
            ignore (T.const b (C.int D.int32 5));
            let g' =
              T.rebuild
                (fun _id v ->
                  match v with
                  | Const { value; dtype; srcs } -> (
                      match C.view value with
                      | Int n when Int64.to_int n = 3 ->
                          Some (T.Const { value = C.int D.int32 4; dtype; srcs })
                      | _ -> None)
                  | _ -> None)
                (T.finish b)
            in
            match T.view g' 0 with
            | Const { value; _ } -> (
                match C.view value with
                | Int n -> equal int 4 (Int64.to_int n)
                | _ -> fail "expected Int")
            | _ -> fail "expected Const");
          test "rebuild no match identity" (fun () ->
            let b = T.create () in
            ignore (mk_f32 b);
            ignore (mk_i32 b);
            let g = T.finish b in
            equal int (T.length g) (T.length (T.rebuild (fun _id _v -> None) g)));
          test "rewrite_fixpoint converges" (fun () ->
            let b = T.create () in
            let a = mk_f32 b in
            ignore (T.binary b ~op:`Add ~lhs:a ~rhs:(T.const b (C.float D.float32 0.0)));
            let g = T.finish b in
            let g' =
              T.rewrite_fixpoint
                (fun _id v ->
                  match v with
                  | Binary { op = `Add; _ } ->
                      Some
                        (T.Const
                           { value = C.float D.float32 99.0;
                             dtype = D.float32;
                             srcs = [] })
                  | _ -> None)
                g
            in
            is_true (T.length g' <= T.length g));
          test "rewrite_fixpoint diverges raises" (fun () ->
            let b = T.create () in
            ignore (T.const b (C.int D.int32 3));
            let g = T.finish b in
            raises_match
              (function Failure msg -> contains msg "fixpoint not reached" | _ -> false)
              (fun () ->
                ignore
                  (T.rewrite_fixpoint ~max_iters:4
                     (fun _id v ->
                       match v with
                       | Const { value; dtype; srcs } -> (
                           match C.view value with
                           | Int n when Int64.to_int n = 3 ->
                               Some (T.Const { value = C.int D.int32 4; dtype; srcs })
                           | Int n when Int64.to_int n = 4 ->
                               Some (T.Const { value = C.int D.int32 3; dtype; srcs })
                           | _ -> None)
                       | _ -> None)
                     g)));
          test "intern shrinks length" (fun () ->
            let b = T.create () in
            let a1 = T.const b (C.float D.float32 1.0) in
            let a2 = T.const b (C.float D.float32 1.0) in
            ignore (T.unary b ~op:`Neg ~src:a1);
            ignore (T.unary b ~op:`Neg ~src:a2);
            let g = T.finish b in
            is_true (T.length (T.rebuild (fun _id _v -> None) g) <= T.length g));
          test "map_children remaps" (fun () ->
            let view : T.view =
              Binary { op = `Add; lhs = 2; rhs = 5; dtype = D.float32 }
            in
            match T.map_children (fun id -> id + 10) view with
            | Binary { lhs = 12; rhs = 15; _ } -> ()
            | _ -> fail "expected remapped Binary");
        ];
      group "Formatting"
        [
          test "pp_instr contains op name" (fun () ->
            is_true
              (contains
                 (pp_to_string T.pp_view
                    (Reshape { src = 0; shape = 1; dtype = D.float32 }))
                 "reshape");
            is_true
              (contains
                 (pp_to_string T.pp_view
                    (Buffer { unique = 0; device = 1; size = 1024; dtype = D.float32 }))
                 "buffer"));
          test "pp program indexed" (fun () ->
            let b = T.create () in
            ignore (T.unique b ~id:0);
            ignore (T.device b (Single "CPU"));
            ignore (mk_f32 b);
            let s = pp_to_string T.pp (T.finish b) in
            is_true (contains s "  0:");
            is_true (contains s "  1:");
            is_true (contains s "  2:"));
        ];
      group "Shape computation"
        [
          test "buffer shape" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            let shapes = T.compute_shapes (T.finish b) in
            equal (option (list int)) (Some [ 1024 ]) shapes.(buf));
          test "const shape is empty" (fun () ->
            let b = T.create () in
            let c = mk_f32 b in
            let shapes = T.compute_shapes (T.finish b) in
            equal (option (list int)) (Some []) shapes.(c));
          test "reshape shape" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            let shape = mk_shape_2x3 b in
            let r = T.reshape b ~src:buf ~shape in
            let shapes = T.compute_shapes (T.finish b) in
            equal (option (list int)) (Some [ 2; 3 ]) shapes.(r));
          test "permute shape" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer ~dtype:D.float32 b in
            let d1 = T.const b (C.int D.index 4) in
            let d2 = T.const b (C.int D.index 8) in
            let shape =
              T.emit b (Vectorize { srcs = [ d1; d2 ]; dtype = D.vec D.index 2 })
            in
            let reshaped = T.reshape b ~src:buf ~shape in
            let p = T.permute b ~src:reshaped ~order:[ 1; 0 ] in
            let shapes = T.compute_shapes (T.finish b) in
            equal (option (list int)) (Some [ 8; 4 ]) shapes.(p));
          test "unary inherits shape" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            let neg = T.unary b ~op:`Neg ~src:buf in
            let shapes = T.compute_shapes (T.finish b) in
            equal (option (list int)) shapes.(buf) shapes.(neg));
          test "binary inherits lhs shape" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            let c = mk_f32 b in
            let add = T.binary b ~op:`Add ~lhs:buf ~rhs:c in
            let shapes = T.compute_shapes (T.finish b) in
            equal (option (list int)) shapes.(buf) shapes.(add));
          test "reduce_axis collapses axes" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer ~dtype:D.float32 b in
            let d1 = T.const b (C.int D.index 4) in
            let d2 = T.const b (C.int D.index 8) in
            let shape =
              T.emit b (Vectorize { srcs = [ d1; d2 ]; dtype = D.vec D.index 2 })
            in
            let reshaped = T.reshape b ~src:buf ~shape in
            let red = T.reduce_axis b ~src:reshaped ~op:`Add ~axes:[ 1 ] in
            let shapes = T.compute_shapes (T.finish b) in
            equal (option (list int)) (Some [ 4; 1 ]) shapes.(red));
          test "sink has no shape" (fun () ->
            let b = T.create () in
            let sink = T.sink b [] in
            let shapes = T.compute_shapes (T.finish b) in
            is_none shapes.(sink));
        ];
      group "Device computation"
        [
          test "device node" (fun () ->
            let b = T.create () in
            let d = T.device b (Single "GPU") in
            let devs = T.compute_devices (T.finish b) in
            equal (option string) (Some "GPU")
              (match devs.(d) with
               | Some (Single s) -> Some s
               | _ -> None));
          test "buffer inherits device" (fun () ->
            let b = T.create () in
            let u = T.unique b ~id:0 in
            let d = T.device b (Single "CPU") in
            let buf = T.buffer b ~unique:u ~device:d ~size:64 ~dtype:D.float32 in
            let devs = T.compute_devices (T.finish b) in
            equal (option string) (Some "CPU")
              (match devs.(buf) with
               | Some (Single s) -> Some s
               | _ -> None));
        ];
      group "Analysis"
        [
          test "backward_slice includes root" (fun () ->
            let b = T.create () in
            let a = mk_f32 b in
            let neg = T.unary b ~op:`Neg ~src:a in
            let slice = T.backward_slice (T.finish b) neg in
            is_true (List.mem neg slice);
            is_true (List.mem a slice));
          test "backward_slice is topological" (fun () ->
            let b = T.create () in
            let a = mk_f32 b in
            let neg = T.unary b ~op:`Neg ~src:a in
            let slice = T.backward_slice (T.finish b) neg in
            let idx_a =
              let rec find i = function
                | [] -> -1 | x :: _ when x = a -> i | _ :: rest -> find (i + 1) rest
              in find 0 slice
            in
            let idx_neg =
              let rec find i = function
                | [] -> -1 | x :: _ when x = neg -> i | _ :: rest -> find (i + 1) rest
              in find 0 slice
            in
            is_true (idx_a < idx_neg));
          test "consumer_map tracks consumers" (fun () ->
            let b = T.create () in
            let a = mk_f32 b in
            let neg = T.unary b ~op:`Neg ~src:a in
            let consumers = T.consumer_map (T.finish b) in
            is_true (List.mem neg consumers.(a)));
          test "base follows through movement ops" (fun () ->
            let b = T.create () in
            let _u, _d, buf = emit_buffer b in
            let shape = mk_shape_2x3 b in
            let reshaped = T.reshape b ~src:buf ~shape in
            let perm = T.permute b ~src:reshaped ~order:[ 1; 0 ] in
            equal int buf (T.base (T.finish b) perm));
          test "base stops at non-movement" (fun () ->
            let b = T.create () in
            let a = mk_f32 b in
            let neg = T.unary b ~op:`Neg ~src:a in
            equal int neg (T.base (T.finish b) neg));
          test "merge_builder appends nodes" (fun () ->
            let b1 = T.create () in
            ignore (mk_f32 b1);
            ignore (mk_i32 b1);
            let prog = T.finish b1 in
            let b2 = T.create () in
            ignore (T.unique b2 ~id:42);
            let prog', shift = T.merge_builder prog b2 in
            equal int (T.length prog + 1) (T.length prog');
            equal int (T.length prog) (shift 0));
        ];
    ]
