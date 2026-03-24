(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Tolk_ir.Const
module D = Tolk_ir.Dtype
module P = Tolk_ir.Program
module Ak = Tolk_ir.Axis_kind
module Sd = Tolk_ir.Special_dim

(* Helpers *)

let global_ptr dt = D.ptr_of dt ~addrspace:Global ~size:(-1)
let local_ptr dt = D.ptr_of dt ~addrspace:Local ~size:(-1)
let reg_ptr dt = D.ptr_of dt ~addrspace:Reg ~size:(-1)
let dt = D.float32
let gptr = global_ptr dt

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

let raises_invalid fn =
  raises_match (function Invalid_argument _ -> true | _ -> false) fn

let emit_i32 b n =
  P.emit b (Const { value = C.int D.int32 n; dtype = D.int32 })

let emit_f32 b x =
  P.emit b (Const { value = C.float dt x; dtype = dt })

let emit_bool b v =
  P.emit b (Const { value = C.bool v; dtype = D.bool })

(* Emit Param(global f32) -> Const(0:i32) -> Index -> Load.
   Returns (ptr_id, idx_id, addr_id, value_id). *)
let emit_load_chain ?(dtype = dt) b =
  let ptr_dt = global_ptr dtype in
  let ptr = P.emit b (Param { idx = 0; dtype = ptr_dt }) in
  let idx = emit_i32 b 0 in
  let addr =
    P.emit b (Index { ptr; idxs = [ idx ]; gate = None; dtype = ptr_dt })
  in
  let value = P.emit b (Load { src = addr; alt = None; dtype }) in
  (ptr, idx, addr, value)

(* Emit a gated load chain with gate and alt.
   Returns (ptr_id, idx_id, gate_id, addr_id, alt_id, value_id). *)
let emit_gated_load_chain ?(dtype = dt) b =
  let ptr_dt = global_ptr dtype in
  let ptr = P.emit b (Param { idx = 0; dtype = ptr_dt }) in
  let idx = emit_i32 b 0 in
  let gate = emit_bool b true in
  let addr =
    P.emit b
      (Index { ptr; idxs = [ idx ]; gate = Some gate; dtype = ptr_dt })
  in
  let alt = P.emit b (Const { value = C.float dtype 0.0; dtype }) in
  let value = P.emit b (Load { src = addr; alt = Some alt; dtype }) in
  (ptr, idx, gate, addr, alt, value)

(* Emit Param(global f32) -> Const(0:i32) -> Index(ungated).
   Returns addr_id. *)
let emit_index_chain b =
  ignore (P.emit b (Param { idx = 0; dtype = gptr }));
  let idx = emit_i32 b 0 in
  P.emit b (Index { ptr = 0; idxs = [ idx ]; gate = None; dtype = gptr })

(* Build, finish, validate. *)
let validates fn =
  let b = P.create () in
  fn b;
  P.validate (P.finish b)

let rejects substring fn =
  let b = P.create () in
  fn b;
  raises_validate substring (fun () -> P.validate (P.finish b))

(* Wmma with default fields, overridable. *)
let wmma_fields ?(dims = (16, 16, 16)) ?(dtype_in = D.Float32)
    ?(dtype_out = D.Float32) ~a ~b ~c () : P.view =
  Wmma
    {
      name = "test";
      a;
      b;
      c;
      dtype = dt;
      dims;
      dtype_in;
      dtype_out;
      device = "METAL";
      threads = 32;
      upcast_axes = ([], [], []);
      reduce_axes = [];
    }

let () =
  run "Ir_next.Program"
    [
      group "Builder"
        [
          test "empty program" (fun () ->
            equal int 0 (P.length (P.finish (P.create ()))));
          test "emit sequential ids" (fun () ->
            let b = P.create () in
            let id0 = P.emit b Barrier in
            let id1 = P.emit b Barrier in
            let id2 = P.emit b Barrier in
            let id3 = P.emit b Barrier in
            let id4 = P.emit b Barrier in
            equal int 0 id0;
            equal int 1 id1;
            equal int 2 id2;
            equal int 3 id3;
            equal int 4 id4);
          test "finish preserves order" (fun () ->
            let b = P.create () in
            ignore (P.emit b (Param { idx = 0; dtype = gptr }));
            ignore (emit_i32 b 0);
            ignore
              (P.emit b
                 (Index
                    { ptr = 0; idxs = [ 1 ]; gate = None; dtype = gptr }));
            let p = P.finish b in
            (match P.view p 0 with
            | Param { idx = 0; _ } -> ()
            | _ -> fail "expected Param");
            (match P.view p 1 with Const _ -> () | _ -> fail "expected Const");
            (match P.view p 2 with
            | Index _ -> ()
            | _ -> fail "expected Index"));
          test "reallocation" (fun () ->
            let b = P.create () in
            for _ = 0 to 63 do
              ignore (P.emit b Barrier)
            done;
            let p = P.finish b in
            equal int 64 (P.length p);
            match P.view p 63 with
            | Barrier -> ()
            | _ -> fail "expected Barrier");
          test "finish is snapshot" (fun () ->
            let b = P.create () in
            ignore (P.emit b Barrier);
            ignore (P.emit b Barrier);
            let p1 = P.finish b in
            ignore (P.emit b Barrier);
            let p2 = P.finish b in
            equal int 2 (P.length p1);
            equal int 3 (P.length p2));
          test "barrier has no dtype" (fun () ->
            let b = P.create () in
            ignore (P.emit b Barrier);
            is_none (P.dtype (P.finish b) 0));
          test "view roundtrip" (fun () ->
            let b = P.create () in
            ignore (P.emit b (Param { idx = 3; dtype = global_ptr D.int32 }));
            match P.view (P.finish b) 0 with
            | Param { idx = 3; dtype } ->
                is_true (D.addrspace dtype = D.Global);
                is_true (D.equal (D.base dtype) D.int32)
            | _ -> fail "expected Param with idx=3");
        ];
      group "Inspection"
        [
          test "dtype value" (fun () ->
            let b = P.create () in
            ignore (emit_f32 b 1.0);
            some (of_equal D.equal) dt (P.dtype (P.finish b) 0));
          test "dtype pointer" (fun () ->
            let b = P.create () in
            ignore (P.emit b (Param { idx = 0; dtype = gptr }));
            some (of_equal D.equal) dt (P.dtype (P.finish b) 0));
          test "dtype effect" (fun () ->
            let b = P.create () in
            let _ptr, _idx, addr, value = emit_load_chain b in
            ignore (P.emit b (Store { dst = addr; value }));
            is_none (P.dtype (P.finish b) 4));
          test "dtype end_range" (fun () ->
            let b = P.create () in
            let size = emit_i32 b 10 in
            let range =
              P.emit b
                (Range { size; dtype = D.int32; axis = 0; sub = []; kind = Ak.Loop })
            in
            ignore (P.emit b (End_range { dep = range; range }));
            is_none (P.dtype (P.finish b) 2));
          test "sort pointer" (fun () ->
            let b = P.create () in
            ignore (P.emit b (Param { idx = 0; dtype = gptr }));
            ignore
              (P.emit b
                 (Define_local { size = 8; dtype = local_ptr dt }));
            ignore (P.emit b (Define_reg { size = 4; dtype = reg_ptr dt }));
            let idx = emit_i32 b 0 in
            ignore
              (P.emit b
                 (Index
                    { ptr = 0; idxs = [ idx ]; gate = None; dtype = gptr }));
            let p = P.finish b in
            is_true (P.sort p 0 = P.Pointer);
            is_true (P.sort p 1 = P.Pointer);
            is_true (P.sort p 2 = P.Pointer);
            is_true (P.sort p 4 = P.Pointer));
          test "sort index" (fun () ->
            let b = P.create () in
            ignore
              (P.emit b
                 (Define_var
                    { name = "n"; lo = 0; hi = 10; dtype = D.int32 }));
            let size = emit_i32 b 10 in
            ignore
              (P.emit b
                 (Range { size; dtype = D.int32; axis = 0; sub = []; kind = Ak.Loop }));
            ignore
              (P.emit b
                 (Special { dim = Sd.Group_id 0; size; dtype = D.int32 }));
            let p = P.finish b in
            is_true (P.sort p 0 = P.Index);
            is_true (P.sort p 2 = P.Index);
            is_true (P.sort p 3 = P.Index));
          test "sort effect" (fun () ->
            let b = P.create () in
            let _ptr, _idx, addr, value = emit_load_chain b in
            ignore (P.emit b (Store { dst = addr; value }));
            ignore (P.emit b Barrier);
            let p = P.finish b in
            is_true (P.sort p 4 = P.Effect);
            is_true (P.sort p 5 = P.Effect));
          test "sort value" (fun () ->
            let b = P.create () in
            let _ptr, _idx, _addr, value = emit_load_chain b in
            ignore (P.emit b (Unary { op = `Neg; src = value; dtype = dt }));
            ignore
              (P.emit b
                 (Binary
                    { op = `Add; lhs = value; rhs = value; dtype = dt }));
            let p = P.finish b in
            is_true (P.sort p 1 = P.Value);
            is_true (P.sort p 3 = P.Value);
            is_true (P.sort p 4 = P.Value);
            is_true (P.sort p 5 = P.Value));
          test "sort after void is effect" (fun () ->
            let b = P.create () in
            let barrier = P.emit b Barrier in
            ignore
              (P.emit b (After { src = barrier; deps = []; dtype = D.void }));
            is_true (P.sort (P.finish b) 1 = P.Effect));
          test "children binary" (fun () ->
            let b = P.create () in
            let a = emit_f32 b 1.0 in
            let c = emit_f32 b 2.0 in
            ignore (P.emit b (Binary { op = `Add; lhs = a; rhs = c; dtype = dt }));
            equal (list int) [ a; c ] (P.children (P.finish b) 2));
        ];
      group "Predicates"
        [
          test "is_alu true" (fun () ->
            is_true (P.is_alu (Unary { op = `Neg; src = 0; dtype = dt }));
            is_true
              (P.is_alu (Binary { op = `Add; lhs = 0; rhs = 1; dtype = dt }));
            is_true
              (P.is_alu
                 (Ternary
                    { op = `Where; a = 0; b = 1; c = 2; dtype = dt })));
          test "is_alu false" (fun () ->
            is_false
              (P.is_alu (Const { value = C.float dt 1.0; dtype = dt }));
            is_false (P.is_alu Barrier);
            is_false (P.is_alu (Store { dst = 0; value = 1 }));
            is_false (P.is_alu (Cast { src = 0; dtype = dt })));
          test "index_gate direct" (fun () ->
            let b = P.create () in
            ignore (P.emit b (Param { idx = 0; dtype = gptr }));
            let idx = emit_i32 b 0 in
            let gate = emit_bool b true in
            let addr =
              P.emit b
                (Index
                   { ptr = 0; idxs = [ idx ]; gate = Some gate; dtype = gptr })
            in
            some int gate (P.index_gate (P.finish b) addr));
          test "index_gate through chain" (fun () ->
            let b = P.create () in
            ignore (P.emit b (Param { idx = 0; dtype = gptr }));
            let idx = emit_i32 b 0 in
            let gate = emit_bool b true in
            let addr =
              P.emit b
                (Index
                   { ptr = 0; idxs = [ idx ]; gate = Some gate; dtype = gptr })
            in
            let cast = P.emit b (Cast { src = addr; dtype = dt }) in
            some int gate (P.index_gate (P.finish b) cast));
          test "index_gate none" (fun () ->
            let b = P.create () in
            ignore (P.emit b (Param { idx = 0; dtype = gptr }));
            let idx = emit_i32 b 0 in
            let addr =
              P.emit b
                (Index
                   { ptr = 0; idxs = [ idx ]; gate = None; dtype = gptr })
            in
            is_none (P.index_gate (P.finish b) addr));
        ];
      group "Validation general"
        [
          test "forward ref rejected" (fun () ->
            rejects "out of bounds or forward" (fun b ->
                ignore
                  (P.emit b (Unary { op = `Neg; src = 1; dtype = dt }));
                ignore (emit_f32 b 1.0)));
          test "self ref rejected" (fun () ->
            rejects "out of bounds or forward" (fun b ->
                ignore (P.emit b (Unary { op = `Neg; src = 0; dtype = dt }))));
          test "index dtype rejected" (fun () ->
            rejects "Index dtype not allowed" (fun b ->
                ignore
                  (P.emit b
                     (Const { value = C.int D.index 0; dtype = D.index }))));
          test "empty program accepted" (fun () ->
            validates (fun _b -> ()));
        ];
      group "Validation per-instruction"
        [
          (* Addrspace *)
          test "param global ok" (fun () ->
            validates (fun b ->
                ignore (P.emit b (Param { idx = 0; dtype = gptr }))));
          test "param local rejected" (fun () ->
            rejects "Global addrspace" (fun b ->
                ignore
                  (P.emit b (Param { idx = 0; dtype = local_ptr dt }))));
          test "define_local ok" (fun () ->
            validates (fun b ->
                ignore
                  (P.emit b
                     (Define_local { size = 8; dtype = local_ptr dt }))));
          test "define_local global rejected" (fun () ->
            rejects "Local addrspace" (fun b ->
                ignore
                  (P.emit b (Define_local { size = 8; dtype = gptr }))));
          test "define_reg ok" (fun () ->
            validates (fun b ->
                ignore
                  (P.emit b (Define_reg { size = 4; dtype = reg_ptr dt }))));
          test "define_reg local rejected" (fun () ->
            rejects "Reg addrspace" (fun b ->
                ignore
                  (P.emit b
                     (Define_reg { size = 4; dtype = local_ptr dt }))));
          (* Define_var *)
          test "define_var ok" (fun () ->
            validates (fun b ->
                ignore
                  (P.emit b
                     (Define_var
                        { name = "n"; lo = 0; hi = 10; dtype = D.int32 }))));
          test "define_var float rejected" (fun () ->
            rejects "must be int/index" (fun b ->
                ignore
                  (P.emit b
                     (Define_var
                        { name = "x"; lo = 0; hi = 4; dtype = dt }))));
          test "define_var vector rejected" (fun () ->
            rejects "must be scalar" (fun b ->
                ignore
                  (P.emit b
                     (Define_var
                        {
                          name = "v";
                          lo = 0;
                          hi = 4;
                          dtype = D.vec D.int32 4;
                        }))));
          test "define_var lo > hi rejected" (fun () ->
            rejects "lo > hi" (fun b ->
                ignore
                  (P.emit b
                     (Define_var
                        { name = "x"; lo = 5; hi = 3; dtype = D.int32 }))));
          (* Range / End_range *)
          test "range int ok" (fun () ->
            validates (fun b ->
                let size = emit_i32 b 10 in
                let range =
                  P.emit b
                    (Range { size; dtype = D.int32; axis = 0; sub = []; kind = Ak.Loop })
                in
                ignore (P.emit b (End_range { dep = range; range }))));
          test "range float rejected" (fun () ->
            rejects "int dtype" (fun b ->
                let size = emit_f32 b 10.0 in
                let range =
                  P.emit b
                    (Range { size; dtype = dt; axis = 0; sub = []; kind = Ak.Loop })
                in
                ignore (P.emit b (End_range { dep = range; range }))));
          test "range vector rejected" (fun () ->
            rejects "scalar" (fun b ->
                let size = emit_i32 b 10 in
                let range =
                  P.emit b
                    (Range
                       {
                         size;
                         dtype = D.vec D.int32 4;
                         axis = 0;
                         sub = [];
                         kind = Ak.Loop;
                       })
                in
                ignore (P.emit b (End_range { dep = range; range }))));
          test "range size mismatch rejected" (fun () ->
            rejects "Range size" (fun b ->
                let size =
                  P.emit b
                    (Const { value = C.int D.int64 10; dtype = D.int64 })
                in
                let range =
                  P.emit b
                    (Range { size; dtype = D.int32; axis = 0; sub = []; kind = Ak.Loop })
                in
                ignore (P.emit b (End_range { dep = range; range }))));
          test "end_range not range rejected" (fun () ->
            rejects "must reference a Range" (fun b ->
                let c = emit_i32 b 0 in
                ignore (P.emit b (End_range { dep = c; range = c }))));
          test "unclosed range rejected" (fun () ->
            rejects "unclosed Range" (fun b ->
                let size = emit_i32 b 10 in
                ignore
                  (P.emit b
                     (Range
                        { size; dtype = D.int32; axis = 0; sub = []; kind = Ak.Loop }))));
          test "end_range unbalanced rejected" (fun () ->
            rejects "unbalanced End_range" (fun b ->
                let size = emit_i32 b 10 in
                let outer =
                  P.emit b
                    (Range { size; dtype = D.int32; axis = 0; sub = []; kind = Ak.Loop })
                in
                ignore
                  (P.emit b
                     (Range
                        { size; dtype = D.int32; axis = 1; sub = []; kind = Ak.Loop }));
                ignore (P.emit b (End_range { dep = outer; range = outer }))));
          (* If / Endif *)
          test "if/endif ok" (fun () ->
            validates (fun b ->
                let addr = emit_index_chain b in
                let cond = emit_bool b true in
                let if_ = P.emit b (If { cond; idx_for_dedup = addr }) in
                ignore (P.emit b (Endif { if_ }))));
          test "if non-bool cond rejected" (fun () ->
            rejects "must be bool" (fun b ->
                let addr = emit_index_chain b in
                let if_ = P.emit b (If { cond = 1; idx_for_dedup = addr }) in
                ignore (P.emit b (Endif { if_ }))));
          test "if idx not index rejected" (fun () ->
            rejects "must reference Index" (fun b ->
                let cond = emit_bool b true in
                let not_index = emit_i32 b 0 in
                let if_ =
                  P.emit b (If { cond; idx_for_dedup = not_index })
                in
                ignore (P.emit b (Endif { if_ }))));
          test "if idx through cast ok" (fun () ->
            validates (fun b ->
                let addr = emit_index_chain b in
                let cast = P.emit b (Cast { src = addr; dtype = D.int32 }) in
                let cond = emit_bool b true in
                let if_ = P.emit b (If { cond; idx_for_dedup = cast }) in
                ignore (P.emit b (Endif { if_ }))));
          test "endif not if rejected" (fun () ->
            rejects "must reference an If" (fun b ->
                let c = emit_i32 b 0 in
                ignore (P.emit b (Endif { if_ = c }))));
          test "unclosed if rejected" (fun () ->
            rejects "unclosed If" (fun b ->
                let addr = emit_index_chain b in
                let cond = emit_bool b true in
                ignore (P.emit b (If { cond; idx_for_dedup = addr }))));
          (* Special *)
          test "special int32 ok" (fun () ->
            validates (fun b ->
                let size = emit_i32 b 32 in
                ignore
                  (P.emit b
                     (Special
                        { dim = Sd.Group_id 0; size; dtype = D.int32 }))));
          test "special float rejected" (fun () ->
            rejects "must be int32 scalar" (fun b ->
                let size = emit_f32 b 32.0 in
                ignore
                  (P.emit b
                     (Special { dim = Sd.Group_id 0; size; dtype = dt }))));
          test "special duplicate rejected" (fun () ->
            rejects "duplicate Special" (fun b ->
                let size = emit_i32 b 32 in
                ignore
                  (P.emit b
                     (Special
                        { dim = Sd.Group_id 0; size; dtype = D.int32 }));
                ignore
                  (P.emit b
                     (Special
                        { dim = Sd.Group_id 0; size; dtype = D.int32 }))));
          test "special different dims ok" (fun () ->
            validates (fun b ->
                let size = emit_i32 b 32 in
                ignore
                  (P.emit b
                     (Special
                        { dim = Sd.Group_id 0; size; dtype = D.int32 }));
                ignore
                  (P.emit b
                     (Special
                        { dim = Sd.Local_id 1; size; dtype = D.int32 }))));
          (* Index *)
          test "index ok" (fun () ->
            validates (fun b -> ignore (emit_index_chain b)));
          test "index bad base rejected" (fun () ->
            rejects "must be a Param" (fun b ->
                let c = emit_i32 b 0 in
                ignore
                  (P.emit b
                     (Index
                        {
                          ptr = c;
                          idxs = [ c ];
                          gate = None;
                          dtype = gptr;
                        }))));
          test "index empty idxs rejected" (fun () ->
            rejects "exactly one index" (fun b ->
                ignore (P.emit b (Param { idx = 0; dtype = gptr }));
                ignore
                  (P.emit b
                     (Index
                        { ptr = 0; idxs = []; gate = None; dtype = gptr }))));
          test "index multi-element idxs rejected" (fun () ->
            rejects "exactly one index" (fun b ->
                ignore (P.emit b (Param { idx = 0; dtype = gptr }));
                let i0 = emit_i32 b 0 in
                let i1 = emit_i32 b 1 in
                ignore
                  (P.emit b
                     (Index
                        {
                          ptr = 0;
                          idxs = [ i0; i1 ];
                          gate = None;
                          dtype = gptr;
                        }))));
          test "index float operand rejected" (fun () ->
            rejects "must be int" (fun b ->
                ignore (P.emit b (Param { idx = 0; dtype = gptr }));
                let fidx = emit_f32 b 0.0 in
                ignore
                  (P.emit b
                     (Index
                        {
                          ptr = 0;
                          idxs = [ fidx ];
                          gate = None;
                          dtype = gptr;
                        }))));
          test "index non-bool gate rejected" (fun () ->
            rejects "must be bool" (fun b ->
                ignore (P.emit b (Param { idx = 0; dtype = gptr }));
                let idx = emit_i32 b 0 in
                ignore
                  (P.emit b
                     (Index
                        {
                          ptr = 0;
                          idxs = [ idx ];
                          gate = Some idx;
                          dtype = gptr;
                        }))));
          (* Load *)
          test "load ok" (fun () ->
            validates (fun b -> ignore (emit_load_chain b)));
          test "load not index rejected" (fun () ->
            rejects "must reference Index" (fun b ->
                let c = emit_i32 b 0 in
                ignore
                  (P.emit b
                     (Load { src = c; alt = None; dtype = D.int32 }))));
          test "load alt gated ok" (fun () ->
            validates (fun b -> ignore (emit_gated_load_chain b)));
          test "load alt without gate rejected" (fun () ->
            rejects "alt requires gated" (fun b ->
                ignore (P.emit b (Param { idx = 0; dtype = gptr }));
                let idx = emit_i32 b 0 in
                let addr =
                  P.emit b
                    (Index
                       {
                         ptr = 0;
                         idxs = [ idx ];
                         gate = None;
                         dtype = gptr;
                       })
                in
                let alt = emit_f32 b 0.0 in
                ignore
                  (P.emit b
                     (Load { src = addr; alt = Some alt; dtype = dt }))));
          (* After *)
          test "after barrier void ok" (fun () ->
            validates (fun b ->
                let barrier = P.emit b Barrier in
                ignore
                  (P.emit b
                     (After { src = barrier; deps = []; dtype = D.void }))));
          test "after barrier non-void rejected" (fun () ->
            rejects "void dtype" (fun b ->
                let barrier = P.emit b Barrier in
                ignore
                  (P.emit b
                     (After { src = barrier; deps = []; dtype = dt }))));
          test "after value mismatch rejected" (fun () ->
            rejects "After src" (fun b ->
                let _ptr, _idx, _addr, value = emit_load_chain b in
                ignore
                  (P.emit b
                     (After { src = value; deps = []; dtype = D.int32 }))));
          (* ALU: Where *)
          test "where ok" (fun () ->
            validates (fun b ->
                let cond = emit_bool b true in
                let t = emit_f32 b 1.0 in
                let e = emit_f32 b 0.0 in
                ignore
                  (P.emit b
                     (Ternary
                        {
                          op = `Where;
                          a = cond;
                          b = t;
                          c = e;
                          dtype = dt;
                        }))));
          test "where non-bool rejected" (fun () ->
            rejects "must be bool" (fun b ->
                let cond = emit_i32 b 1 in
                let t = emit_f32 b 1.0 in
                let e = emit_f32 b 0.0 in
                ignore
                  (P.emit b
                     (Ternary
                        {
                          op = `Where;
                          a = cond;
                          b = t;
                          c = e;
                          dtype = dt;
                        }))));
          test "where mismatched arms rejected" (fun () ->
            rejects "Where branch" (fun b ->
                let cond = emit_bool b true in
                let t = emit_f32 b 1.0 in
                let e = emit_i32 b 0 in
                ignore
                  (P.emit b
                     (Ternary
                        {
                          op = `Where;
                          a = cond;
                          b = t;
                          c = e;
                          dtype = dt;
                        }))));
          (* ALU: Cmp *)
          test "cmp ok" (fun () ->
            validates (fun b ->
                let a = emit_f32 b 1.0 in
                let c = emit_f32 b 2.0 in
                ignore
                  (P.emit b
                     (Binary
                        { op = `Cmplt; lhs = a; rhs = c; dtype = D.bool }))));
          test "cmp non-bool result rejected" (fun () ->
            rejects "comparison result must be bool" (fun b ->
                let a = emit_f32 b 1.0 in
                let c = emit_f32 b 2.0 in
                ignore
                  (P.emit b
                     (Binary
                        { op = `Cmplt; lhs = a; rhs = c; dtype = D.int32 }))));
          test "cmp operands mismatch rejected" (fun () ->
            rejects "don't match" (fun b ->
                let a = emit_f32 b 1.0 in
                let c = emit_i32 b 2 in
                ignore
                  (P.emit b
                     (Binary
                        { op = `Cmpeq; lhs = a; rhs = c; dtype = D.bool }))));
          (* ALU: Idiv/Mod *)
          test "idiv int ok" (fun () ->
            validates (fun b ->
                let a = emit_i32 b 10 in
                let c = emit_i32 b 3 in
                ignore
                  (P.emit b
                     (Binary
                        { op = `Idiv; lhs = a; rhs = c; dtype = D.int32 }))));
          test "idiv float rejected" (fun () ->
            rejects "int dtype" (fun b ->
                let a = emit_f32 b 1.0 in
                let c = emit_f32 b 2.0 in
                ignore
                  (P.emit b
                     (Binary { op = `Idiv; lhs = a; rhs = c; dtype = dt }))));
          (* ALU: Shift *)
          test "shift ok" (fun () ->
            validates (fun b ->
                let a = emit_i32 b 8 in
                let c = emit_i32 b 2 in
                ignore
                  (P.emit b
                     (Binary
                        { op = `Shl; lhs = a; rhs = c; dtype = D.int32 }))));
          test "shift rhs mismatch rejected" (fun () ->
            rejects "shift rhs must match" (fun b ->
                let a = emit_i32 b 8 in
                let c =
                  P.emit b
                    (Const { value = C.int D.int64 2; dtype = D.int64 })
                in
                ignore
                  (P.emit b
                     (Binary
                        { op = `Shl; lhs = a; rhs = c; dtype = D.int32 }))));
          (* ALU: Unary *)
          test "unary ok" (fun () ->
            validates (fun b ->
                let a = emit_f32 b 1.0 in
                ignore
                  (P.emit b (Unary { op = `Neg; src = a; dtype = dt }))));
          test "unary mismatch rejected" (fun () ->
            rejects "unary ALU" (fun b ->
                let a = emit_f32 b 1.0 in
                ignore
                  (P.emit b
                     (Unary { op = `Neg; src = a; dtype = D.int32 }))));
          (* ALU: Binary general *)
          test "binary alu ok" (fun () ->
            validates (fun b ->
                let a = emit_f32 b 1.0 in
                let c = emit_f32 b 2.0 in
                ignore
                  (P.emit b
                     (Binary { op = `Add; lhs = a; rhs = c; dtype = dt }))));
          test "binary alu lhs mismatch rejected" (fun () ->
            rejects "binary ALU lhs" (fun b ->
                let a = emit_i32 b 1 in
                let c = emit_f32 b 2.0 in
                ignore
                  (P.emit b
                     (Binary { op = `Add; lhs = a; rhs = c; dtype = dt }))));
          (* ALU: Mulacc *)
          test "mulacc ok" (fun () ->
            validates (fun b ->
                let a = emit_f32 b 1.0 in
                let c = emit_f32 b 2.0 in
                let d = emit_f32 b 3.0 in
                ignore
                  (P.emit b
                     (Ternary
                        { op = `Mulacc; a; b = c; c = d; dtype = dt }))));
          test "mulacc mismatch rejected" (fun () ->
            rejects "Mulacc" (fun b ->
                let a = emit_f32 b 1.0 in
                let c = emit_i32 b 2 in
                let d = emit_f32 b 3.0 in
                ignore
                  (P.emit b
                     (Ternary
                        { op = `Mulacc; a; b = c; c = d; dtype = dt }))));
          (* Vectorize *)
          test "vectorize ok" (fun () ->
            validates (fun b ->
                let a = emit_f32 b 1.0 in
                let c = emit_f32 b 2.0 in
                let d = emit_f32 b 3.0 in
                ignore
                  (P.emit b
                     (Vectorize
                        { srcs = [ a; c; d ]; dtype = D.vec dt 3 }))));
          test "vectorize one source rejected" (fun () ->
            rejects "more than one source" (fun b ->
                let a = emit_f32 b 1.0 in
                ignore
                  (P.emit b
                     (Vectorize { srcs = [ a ]; dtype = D.vec dt 1 }))));
          (* Gep *)
          test "gep ok" (fun () ->
            validates (fun b ->
                let a = emit_f32 b 1.0 in
                let c = emit_f32 b 2.0 in
                let v =
                  P.emit b
                    (Vectorize { srcs = [ a; c ]; dtype = D.vec dt 2 })
                in
                ignore (P.emit b (Gep { src = v; idxs = [1]; dtype = dt }))));
          test "gep out of bounds rejected" (fun () ->
            rejects "out of bounds" (fun b ->
                let a = emit_f32 b 1.0 in
                let c = emit_f32 b 2.0 in
                let v =
                  P.emit b
                    (Vectorize { srcs = [ a; c ]; dtype = D.vec dt 2 })
                in
                ignore (P.emit b (Gep { src = v; idxs = [5]; dtype = dt }))));
          (* Store *)
          test "store ok" (fun () ->
            validates (fun b ->
                let _ptr, _idx, addr, _value = emit_load_chain b in
                let new_val = emit_f32 b 42.0 in
                ignore (P.emit b (Store { dst = addr; value = new_val }))));
          test "store not index rejected" (fun () ->
            rejects "must reference Index" (fun b ->
                let c = emit_i32 b 0 in
                let v = emit_f32 b 1.0 in
                ignore (P.emit b (Store { dst = c; value = v }))));
          test "store dtype mismatch rejected" (fun () ->
            rejects "Store value" (fun b ->
                let _ptr, _idx, addr, _value = emit_load_chain b in
                let wrong = emit_i32 b 7 in
                ignore (P.emit b (Store { dst = addr; value = wrong }))));
          (* Wmma *)
          test "wmma ok" (fun () ->
            validates (fun b ->
                let a = emit_f32 b 1.0 in
                let c = emit_f32 b 2.0 in
                let d = emit_f32 b 3.0 in
                ignore (P.emit b (wmma_fields ~a ~b:c ~c:d ()))));
          test "wmma zero dim rejected" (fun () ->
            rejects "dims must be positive" (fun b ->
                let a = emit_f32 b 1.0 in
                let c = emit_f32 b 2.0 in
                let d = emit_f32 b 3.0 in
                ignore
                  (P.emit b
                     (wmma_fields ~a ~b:c ~c:d ~dims:(0, 16, 16) ()))));
          test "wmma dtype mismatch rejected" (fun () ->
            rejects "must match dtype_out" (fun b ->
                let a = emit_f32 b 1.0 in
                let c = emit_f32 b 2.0 in
                let d = emit_f32 b 3.0 in
                ignore
                  (P.emit b
                     (wmma_fields ~a ~b:c ~c:d ~dtype_in:D.Float16
                        ~dtype_out:D.Float16 ()))));
        ];
      group "Control flow balancing"
        [
          test "nested ranges balanced" (fun () ->
            validates (fun b ->
                let size = emit_i32 b 10 in
                let outer =
                  P.emit b
                    (Range { size; dtype = D.int32; axis = 0; sub = []; kind = Ak.Loop })
                in
                let inner =
                  P.emit b
                    (Range { size; dtype = D.int32; axis = 1; sub = []; kind = Ak.Loop })
                in
                ignore (P.emit b (End_range { dep = inner; range = inner }));
                ignore (P.emit b (End_range { dep = outer; range = outer }))));
          test "nested ifs balanced" (fun () ->
            validates (fun b ->
                let addr = emit_index_chain b in
                let cond = emit_bool b true in
                let outer_if =
                  P.emit b (If { cond; idx_for_dedup = addr })
                in
                let inner_if =
                  P.emit b (If { cond; idx_for_dedup = addr })
                in
                ignore (P.emit b (Endif { if_ = inner_if }));
                ignore (P.emit b (Endif { if_ = outer_if }))));
          test "interleaved range/if" (fun () ->
            validates (fun b ->
                let addr = emit_index_chain b in
                let size = emit_i32 b 10 in
                let range =
                  P.emit b
                    (Range { size; dtype = D.int32; axis = 0; sub = []; kind = Ak.Loop })
                in
                let cond = emit_bool b true in
                let if_ = P.emit b (If { cond; idx_for_dedup = addr }) in
                ignore (P.emit b (Endif { if_ }));
                ignore (P.emit b (End_range { dep = range; range }))));
          test "sequential ranges" (fun () ->
            validates (fun b ->
                let size = emit_i32 b 10 in
                let r1 =
                  P.emit b
                    (Range { size; dtype = D.int32; axis = 0; sub = []; kind = Ak.Loop })
                in
                ignore (P.emit b (End_range { dep = r1; range = r1 }));
                let r2 =
                  P.emit b
                    (Range { size; dtype = D.int32; axis = 1; sub = []; kind = Ak.Loop })
                in
                ignore (P.emit b (End_range { dep = r2; range = r2 }))));
        ];
      group "Rewriting"
        [
          test "map_children binary" (fun () ->
            let view : P.view =
              Binary { op = `Add; lhs = 2; rhs = 5; dtype = dt }
            in
            match P.map_children (fun id -> id + 10) view with
            | Binary { lhs = 12; rhs = 15; _ } -> ()
            | _ -> fail "expected remapped Binary");
          test "map_children leaf identity" (fun () ->
            let view : P.view =
              Param { idx = 0; dtype = global_ptr D.int32 }
            in
            match P.map_children (fun id -> id + 10) view with
            | Param { idx = 0; _ } -> ()
            | _ -> fail "expected unchanged Param");
          test "map_alu remaps" (fun () ->
            let view : P.view =
              Unary { op = `Neg; src = 5; dtype = dt }
            in
            match P.map_alu ~map_ref:(fun r -> r + 1) ~dtype:D.int32 view with
            | Unary { op = `Neg; src = 6; dtype } when D.equal dtype D.int32 ->
                ()
            | _ -> fail "expected remapped Unary with int32 dtype");
          test "map_alu non-alu raises" (fun () ->
            raises_invalid (fun () ->
                ignore
                  (P.map_alu ~map_ref:Fun.id ~dtype:dt
                     (Const { value = C.float dt 1.0; dtype = dt }))));
          test "rebuild identity" (fun () ->
            let b = P.create () in
            let _ptr, _idx, addr, value = emit_load_chain b in
            ignore (P.emit b (Store { dst = addr; value }));
            let p = P.finish b in
            let p' = P.rebuild (fun ~emit:_ ~map_ref:_ _ -> None) p in
            equal int (P.length p) (P.length p'));
        ];
      group "Formatting"
        [
          test "pp_view param" (fun () ->
            let s =
              Format.asprintf "%a" P.pp_view
                (Param { idx = 0; dtype = gptr })
            in
            is_true (contains s "param");
            is_true (contains s "global"));
          test "pp program indexed" (fun () ->
            let b = P.create () in
            ignore (P.emit b (Param { idx = 0; dtype = gptr }));
            ignore (emit_i32 b 0);
            ignore (P.emit b Barrier);
            let s = Format.asprintf "%a" P.pp (P.finish b) in
            is_true (contains s "  0:");
            is_true (contains s "  1:");
            is_true (contains s "  2:"));
        ];
    ]
