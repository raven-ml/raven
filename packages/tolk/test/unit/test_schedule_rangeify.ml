(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for the schedule rangeify pipeline.

   Covers indexing.ml (core rangeify algorithm) and rangeify.ml (pipeline
   orchestrator). Tests are organized by responsibility:
   - is_always_contiguous: op classification
   - new_range: range creation with size-1 folding
   - apply_movement_op: range transforms for all 6 movement ops
   - run_rangeify: backward walk producing range_map/realize_map
   - get_kernel_graph: full pipeline kernel count tests

   Covers core rangeify correctness and schedule-level fusion decisions. *)

open Windtrap
open Tolk
module C = Tolk_ir.Const
module D = Tolk_ir.Dtype
module T = Tolk_ir.Tensor
module K = Tolk_ir.Kernel
module Ak = Tolk_ir.Axis_kind

(* Extract an int from a Const value, assuming it's Int. *)
let const_to_int (v : C.t) : int =
  match C.view v with Int n -> Int64.to_int n | _ -> failwith "not Int"

(* Helpers *)

(* Emit a shape-encoding node from a concrete int list.
   For 1-D: emits a single Const index.
   For N-D: emits a Vectorize of Const index nodes. *)
let mk_shape (dims : int list) : T.t =
  let ids = List.map (fun s -> T.const (C.int D.Val.index s) D.index) dims in
  match ids with
  | [ d ] -> d
  | ds ->
      T.vectorize ~srcs:ds

(* Emit a PARAM with a known shape and CPU device. *)
let mk_param ~slot (shape : int list) : T.t =
  let shape_id = if shape = [] then None else Some (mk_shape shape) in
  let dev = T.device (Single "CPU") in
  T.param ~slot ~dtype:D.float32 ?shape:shape_id ~device:dev ()

(* Count CALL nodes in a program. *)
let count_calls (root : T.t) : int =
  let n = ref 0 in
  List.iter (fun node ->
    match T.view node with Call _ -> incr n | _ -> ())
    (T.toposort root);
  !n

(* Wrap an expression in CONTIGUOUS -> SINK for get_kernel_graph. *)
let wrap_sink b (src : T.t) : T.t =
  let c = T.contiguous ~src () in
  T.sink [ c ]

(* Build a program from a builder function and run get_kernel_graph.
   Returns the kernel graph and CALL count. *)
let run_pipeline (build_fn : unit -> T.t) : T.t * int =
  let _sink = build_fn () in
  let result = Rangeify.get_kernel_graph (build_fn ()) in
  (result, count_calls result)

(* is_always_contiguous tests *)

let dummy = T.const (C.int D.Val.index 0) D.index
let dummy2 = T.const (C.int D.Val.index 1) D.index

let is_always_contiguous_tests =
  group "is_always_contiguous"
    [
      test "contiguous" (fun () ->
          let dummy = T.const (C.int D.Val.index 0) D.index in
          is_true
            (Indexing.is_always_contiguous
               (T.Contiguous { src = dummy; ranges = []; opts = []; dtype = D.float32 })));
      test "after with store (assign pattern)" (fun () ->
          let dummy = T.const (C.int D.Val.index 0) D.index in
          let dummy2 = T.const (C.int D.Val.index 1) D.index in
          (* AFTER is a buffer identity — always contiguous *)
          is_true
            (Indexing.is_always_contiguous
               (T.After { src = dummy; deps = [ dummy2 ]; dtype = D.float32 })));
      test "copy" (fun () ->
          let dummy = T.const (C.int D.Val.index 0) D.index in
          let dummy2 = T.const (C.int D.Val.index 1) D.index in
          is_true
            (Indexing.is_always_contiguous
               (T.Copy { src = dummy; device = dummy2; dtype = D.float32 })));
      test "buffer" (fun () ->
          let dummy = T.const (C.int D.Val.index 0) D.index in
          let dummy2 = T.const (C.int D.Val.index 1) D.index in
          is_true
            (Indexing.is_always_contiguous
               (T.Buffer
                  { unique = dummy; device = dummy2; size = 4; dtype = D.float32 })));
      test "const" (fun () ->
          is_true
            (Indexing.is_always_contiguous
               (T.Const
                  { value = C.int D.Val.int32 0; dtype = D.int32; srcs = [] })));
      test "param" (fun () ->
          is_true
            (Indexing.is_always_contiguous
               (T.Param
                  {
                    slot = 0;
                    dtype = D.float32;
                    shape = None;
                    device = None;
                  })));
      test "call" (fun () ->
          is_true
            (Indexing.is_always_contiguous
               (T.Call
                  {
                    callee = Ast (K.const (C.int D.Val.int32 0));
                    args = [];
                    info =
                      {
                        grad_fxn = None;
                        metadata = [];
                        name = None;
                        precompile = false;
                      };
                    dtype = D.float32;
                  })));
      test "reshape not contiguous" (fun () ->
          is_false
            (Indexing.is_always_contiguous
               (T.Reshape { src = dummy; shape = dummy2; dtype = D.float32 })));
      test "expand not contiguous" (fun () ->
          is_false
            (Indexing.is_always_contiguous
               (T.Expand { src = dummy; shape = dummy2; dtype = D.float32 })));
      test "reduce_axis not contiguous" (fun () ->
          is_false
            (Indexing.is_always_contiguous
               (T.Reduce_axis
                  { src = dummy; op = `Add; axes = [ 0 ]; dtype = D.float32 })));
      test "unary not contiguous" (fun () ->
          is_false
            (Indexing.is_always_contiguous
               (T.Unary { op = `Neg; src = dummy; dtype = D.float32 })));
      test "binary not contiguous" (fun () ->
          is_false
            (Indexing.is_always_contiguous
               (T.Binary { op = `Add; lhs = dummy; rhs = dummy2; dtype = D.float32 })));
    ]

(* new_range tests *)

let new_range_tests =
  group "new_range"
    [
      test "size 1 gives const 0" (fun () ->
          
          let ctx = Indexing.create_context () in
          let id = Indexing.new_range ctx 1 ~kind:Ak.Loop () in
          (match T.view id with
          | Const { value; _ } ->
              equal int 0 (const_to_int value)
          | _ -> fail "expected Const for size 1"));
      test "size 0 gives Range (resolve(s!=1) is true)" (fun () ->
          
          let ctx = Indexing.create_context () in
          let id = Indexing.new_range ctx 0 ~kind:Ak.Loop () in
          (match T.view id with
          | Range { size; axis; kind; _ } ->
              equal int 0 axis;
              is_true (kind = Ak.Loop);
              (match T.view size with
              | Const { value; _ } ->
                  equal int 0 (const_to_int value)
              | _ -> fail "expected Const for range size")
          | _ -> fail "expected Range for size 0"));
      test "size > 1 gives Range" (fun () ->
          
          let ctx = Indexing.create_context () in
          let id = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
          (match T.view id with
          | Range { size; axis; kind; _ } ->
              equal int 0 axis;
              is_true (kind = Ak.Loop);
              (match T.view size with
              | Const { value; _ } ->
                  equal int 4 (const_to_int value)
              | _ -> fail "expected Const for range size")
          | _ -> fail "expected Range for size > 1"));
      test "axis increments" (fun () ->
          
          let ctx = Indexing.create_context () in
          let id1 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
          let id2 = Indexing.new_range ctx 8 ~kind:Ak.Loop () in
          let axis1 =
            match T.view id1 with
            | Range { axis; _ } -> axis
            | _ -> fail "expected Range"
          in
          let axis2 =
            match T.view id2 with
            | Range { axis; _ } -> axis
            | _ -> fail "expected Range"
          in
          equal int 0 axis1;
          equal int 1 axis2);
      test "kind propagates" (fun () ->
          
          let ctx = Indexing.create_context () in
          let id = Indexing.new_range ctx 8 ~kind:Ak.Reduce () in
          (match T.view id with
          | Range { kind; _ } -> is_true (kind = Ak.Reduce)
          | _ -> fail "expected Range"));
    ]

(* apply_movement_op tests *)

let apply_movement_op_tests =
  group "apply_movement_op"
    [
      (* SHRINK *)
      group "shrink"
        [
          test "zero offset passthrough" (fun () ->
              let param = mk_param ~slot:0 [ 4; 4 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let rng1 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let before = mk_shape [ 0; 0 ] in
              let after = mk_shape [ 4; 4 ] in
              let shapes = T.compute_shapes param in
              let v =
                T.Shrink { src = param; before; after; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v
                  [ rng0; rng1 ]
              in
              (* zero offsets: output ranges should be same ids as input *)
              equal int (T.tag rng0) (T.tag (List.nth result 0));
              equal int (T.tag rng1) (T.tag (List.nth result 1)));
          test "nonzero offset adds" (fun () ->
              let param = mk_param ~slot:0 [ 4; 4 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let rng1 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let before = mk_shape [ 1; 2 ] in
              let after = mk_shape [ 3; 4 ] in
              let shapes = T.compute_shapes param in
              let v =
                T.Shrink { src = param; before; after; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v
                  [ rng0; rng1 ]
              in
              equal int 2 (List.length result);
              (match T.view (List.nth result 0) with
              | Binary { op = `Add; _ } -> ()
              | _ -> fail "expected Add for shrink offset 1");
              (match T.view (List.nth result 1) with
              | Binary { op = `Add; _ } -> ()
              | _ -> fail "expected Add for shrink offset 2"));
        ];
      (* PERMUTE *)
      group "permute"
        [
          test "swap [1;0]" (fun () ->
              let param = mk_param ~slot:0 [ 4; 8 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let rng1 = Indexing.new_range ctx 8 ~kind:Ak.Loop () in
              let shapes = T.compute_shapes param in
              let v =
                T.Permute { src = dummy; order = [ 1; 0 ]; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v
                  [ rng0; rng1 ]
              in
              (* permute [1;0]: argsort = [1;0] → result = [rng1; rng0] *)
              equal int (T.tag rng1) (T.tag (List.nth result 0));
              equal int (T.tag rng0) (T.tag (List.nth result 1)));
          test "identity [0;1;2]" (fun () ->
              let param = mk_param ~slot:0 [ 2; 3; 4 ] in
              ignore param;
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 2 ~kind:Ak.Loop () in
              let rng1 = Indexing.new_range ctx 3 ~kind:Ak.Loop () in
              let rng2 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let shapes = T.compute_shapes param in
              let v =
                T.Permute
                  { src = dummy; order = [ 0; 1; 2 ]; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v
                  [ rng0; rng1; rng2 ]
              in
              equal int (T.tag rng0) (T.tag (List.nth result 0));
              equal int (T.tag rng1) (T.tag (List.nth result 1));
              equal int (T.tag rng2) (T.tag (List.nth result 2)));
        ];
      (* FLIP *)
      group "flip"
        [
          test "flip true reverses" (fun () ->
              let param = mk_param ~slot:0 [ 4 ] in
              let ctx = Indexing.create_context () in
              let rng = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let shapes = T.compute_shapes param in
              let v =
                T.Flip { src = param; dims = [ true ]; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v [ rng ]
              in
              (match T.view (List.nth result 0) with
              | Binary { op = `Sub; _ } -> ()
              | _ -> fail "expected Sub for flip"));
          test "flip false passthrough" (fun () ->
              let param = mk_param ~slot:0 [ 4 ] in
              let ctx = Indexing.create_context () in
              let rng = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let shapes = T.compute_shapes param in
              let v =
                T.Flip { src = param; dims = [ false ]; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v [ rng ]
              in
              equal int (T.tag rng) (T.tag (List.nth result 0)));
        ];
      (* EXPAND *)
      group "expand"
        [
          test "same shape passthrough" (fun () ->
              let param = mk_param ~slot:0 [ 4; 4 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let rng1 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let out_shape = mk_shape [ 4; 4 ] in
              let shapes = T.compute_shapes param in
              let v =
                T.Expand
                  { src = param; shape = out_shape; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v
                  [ rng0; rng1 ]
              in
              equal int (T.tag rng0) (T.tag (List.nth result 0));
              equal int (T.tag rng1) (T.tag (List.nth result 1)));
          test "broadcast 1->N gives const 0" (fun () ->
              let param = mk_param ~slot:0 [ 1; 4 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let rng1 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let out_shape = mk_shape [ 4; 4 ] in
              let shapes = T.compute_shapes param in
              let v =
                T.Expand
                  { src = param; shape = out_shape; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v
                  [ rng0; rng1 ]
              in
              (* axis 0: in_shape=1, out_shape=4 -> const 0 *)
              (match T.view (List.nth result 0) with
              | Const { value; _ } ->
                  equal int 0 (const_to_int value)
              | _ -> fail "expected Const 0 for expanded dim");
              (* axis 1: in_shape=4, out_shape=4 -> passthrough *)
              equal int (T.tag rng1) (T.tag (List.nth result 1)));
        ];
      (* PAD *)
      group "pad"
        [
          test "zero pad passthrough" (fun () ->
              let param = mk_param ~slot:0 [ 4; 4 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let rng1 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let before = mk_shape [ 0; 0 ] in
              let after = mk_shape [ 0; 0 ] in
              let shapes = T.compute_shapes param in
              let v =
                T.Pad { src = param; before; after; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v
                  [ rng0; rng1 ]
              in
              equal int (T.tag rng0) (T.tag (List.nth result 0));
              equal int (T.tag rng1) (T.tag (List.nth result 1)));
          test "nonzero pad creates WHERE" (fun () ->
              let param = mk_param ~slot:0 [ 4; 4 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 6 ~kind:Ak.Loop () in
              let rng1 = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let before = mk_shape [ 2; 0 ] in
              let after = mk_shape [ 0; 0 ] in
              let shapes = T.compute_shapes param in
              let v =
                T.Pad { src = param; before; after; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v
                  [ rng0; rng1 ]
              in
              (* axis 0: pad_before=2 -> WHERE(valid, offset, invalid) *)
              (match T.view (List.nth result 0) with
              | Ternary { op = `Where; _ } -> ()
              | _ -> fail "expected WHERE for padded dim");
              (* axis 1: pad_before=0 -> passthrough *)
              equal int (T.tag rng1) (T.tag (List.nth result 1)));
          test "end-only pad creates WHERE (F2)" (fun () ->
              (* PAD with start=0, end=2 on a dim of size 4:
                 output range goes 0..5 but valid indices are only 0..3.
                 Must emit WHERE(r < 4, r, invalid). *)
              let param = mk_param ~slot:0 [ 4 ] in
              let ctx = Indexing.create_context () in
              let rng = Indexing.new_range ctx 6 ~kind:Ak.Loop () in
              let before = mk_shape [ 0 ] in
              let after = mk_shape [ 2 ] in
              let shapes = T.compute_shapes param in
              let v =
                T.Pad { src = param; before; after; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v [ rng ]
              in
              (* end padding nonzero -> WHERE must be generated *)
              (match T.view (List.nth result 0) with
              | Ternary { op = `Where; _ } -> ()
              | _ -> fail "expected WHERE for end-only padded dim"));
        ];
      (* RESHAPE *)
      group "reshape"
        [
          test "flatten [2;3] to [6]" (fun () ->
              (* apply_movement_op receives output ranges and returns input ranges.
                 Reshape [2;3] -> [6]: output shape [6], input shape [2;3].
                 Pass 1 output range, get back 2 input ranges. *)
              let param = mk_param ~slot:0 [ 2; 3 ] in
              let ctx = Indexing.create_context () in
              let rng_out = Indexing.new_range ctx 6 ~kind:Ak.Loop () in
              let new_shape = mk_shape [ 6 ] in
              let shapes = T.compute_shapes param in
              let v =
                T.Reshape
                  { src = param; shape = new_shape; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v [ rng_out ]
              in
              equal int 2 (List.length result));
          test "unflatten [6] to [2;3]" (fun () ->
              (* Reshape [6] -> [2;3]: output shape [2;3], input shape [6].
                 Pass 2 output ranges, get back 1 input range. *)
              let param = mk_param ~slot:0 [ 6 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 2 ~kind:Ak.Loop () in
              let rng1 = Indexing.new_range ctx 3 ~kind:Ak.Loop () in
              let new_shape = mk_shape [ 2; 3 ] in
              let shapes = T.compute_shapes param in
              let v =
                T.Reshape
                  { src = param; shape = new_shape; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v
                  [ rng0; rng1 ]
              in
              equal int 1 (List.length result));
          test "identity [4] to [4]" (fun () ->
              let param = mk_param ~slot:0 [ 4 ] in
              let ctx = Indexing.create_context () in
              let rng = Indexing.new_range ctx 4 ~kind:Ak.Loop () in
              let new_shape = mk_shape [ 4 ] in
              let shapes = T.compute_shapes param in
              let v =
                T.Reshape
                  { src = param; shape = new_shape; dtype = D.float32 }
              in
              let result =
                Indexing.apply_movement_op ~shapes v [ rng ]
              in
              equal int 1 (List.length result));
        ];
    ]

(* run_rangeify tests *)

let run_rangeify_tests =
  group "run_rangeify"
    [
      test "realized node creates Realized" (fun () ->
          
          let param = mk_param ~slot:0 [ 4 ] in
          let contig = T.contiguous ~src:param () in
          let _sink = T.sink [ contig ] in
          let shapes = T.compute_shapes _sink in
          let ctx = Indexing.run_rangeify _sink ~shapes in
          (match Hashtbl.find_opt ctx.realize_map (T.tag contig) with
          | Some (Indexing.Realized axes) ->
              equal (list int) [ 0 ] axes
          | Some Indexing.Marked -> fail "expected Realized, got Marked"
          | None -> fail "expected Realized, got None"));
      test "realized node has range_map entry" (fun () ->
          let param = mk_param ~slot:0 [ 4 ] in
          let contig = T.contiguous ~src:param () in
          let sink = T.sink [ contig ] in
          let shapes = T.compute_shapes sink in
          let ctx = Indexing.run_rangeify sink ~shapes in
          is_true (Hashtbl.mem ctx.range_map (T.tag contig)));
      test "elementwise inherits consumer ranges" (fun () ->
          let param = mk_param ~slot:0 [ 4 ] in
          let neg = T.unary ~op:`Neg ~src:param in
          let contig = T.contiguous ~src:neg () in
          let sink = T.sink [ contig ] in
          let shapes = T.compute_shapes sink in
          let ctx = Indexing.run_rangeify sink ~shapes in
          is_true (Hashtbl.mem ctx.range_map (T.tag neg)));
      test "reduce creates reduce-kind ranges" (fun () ->
          let param = mk_param ~slot:0 [ 4; 4 ] in
          let red = T.reduce_axis ~src:param ~op:`Add ~axes:[ 1 ] in
          let contig = T.contiguous ~src:red () in
          let sink = T.sink [ contig ] in
          let shapes = T.compute_shapes sink in
          let ctx = Indexing.run_rangeify sink ~shapes in
          (match Hashtbl.find_opt ctx.range_map (T.tag red) with
          | Some (in_rngs, _out_rngs) ->
              equal int 2 (List.length in_rngs);
              (match T.view (List.nth in_rngs 1) with
              | Range { kind; _ } -> is_true (kind = Ak.Reduce)
              | Const _ -> fail "expected Range for reduce axis, got Const"
              | _ -> fail "expected Range for reduce axis")
          | None -> fail "expected range_map entry for reduce"));
      test "movement op has different in and out ranges" (fun () ->
          let param = mk_param ~slot:0 [ 4; 8 ] in
          let perm = T.permute ~src:param ~order:[ 1; 0 ] in
          let contig = T.contiguous ~src:perm () in
          let sink = T.sink [ contig ] in
          let shapes = T.compute_shapes sink in
          let ctx = Indexing.run_rangeify sink ~shapes in
          (match Hashtbl.find_opt ctx.range_map (T.tag perm) with
          | Some (in_rngs, out_rngs) ->
              equal int 2 (List.length in_rngs);
              equal int 2 (List.length out_rngs);
              is_true (List.nth out_rngs 1 == List.nth in_rngs 0);
              is_true (List.nth out_rngs 0 == List.nth in_rngs 1)
          | None -> fail "expected range_map entry for permute"));
      test "2D realized node has all axes" (fun () ->
          
          let param = mk_param ~slot:0 [ 4; 8 ] in
          let contig = T.contiguous ~src:param () in
          let _sink = T.sink [ contig ] in
          let shapes = T.compute_shapes _sink in
          let ctx = Indexing.run_rangeify _sink ~shapes in
          (match Hashtbl.find_opt ctx.realize_map (T.tag contig) with
          | Some (Indexing.Realized axes) ->
              equal (list int) [ 0; 1 ] axes
          | _ -> fail "expected Realized with [0;1]"));
    ]

(* get_kernel_graph integration tests *)

(* Helper to build a pipeline test: build graph, run get_kernel_graph,
   assert CALL count. *)
let pipeline_test name ~expected_calls build_fn =
  test name (fun () ->
      let _, calls = run_pipeline build_fn in
      equal int expected_calls calls)

let get_kernel_graph_tests =
  group "get_kernel_graph"
    [
      (* test_basic_binop_fusion *)
      pipeline_test "elementwise fusion" ~expected_calls:1 (fun b ->
          let a = mk_param ~slot:0 [ 10 ] in
          let bp = mk_param ~slot:1 [ 10 ] in
          let c = mk_param ~slot:2 [ 10 ] in
          let ab = T.binary ~op:`Add ~lhs:a ~rhs:bp in
          let abc = T.binary ~op:`Add ~lhs:ab ~rhs:c in
          wrap_sink b abc);
      (* test_mulacc_fusion *)
      pipeline_test "mulacc fusion" ~expected_calls:1 (fun b ->
          let a = mk_param ~slot:0 [ 10 ] in
          let bp = mk_param ~slot:1 [ 10 ] in
          let mul = T.binary ~op:`Mul ~lhs:a ~rhs:bp in
          let red =
            T.reduce_axis ~src:mul ~op:`Add ~axes:[ 0 ]
          in
          wrap_sink b red);
      (* test_binop_reshape_fusion *)
      pipeline_test "binop reshape fusion" ~expected_calls:1 (fun b ->
          let a = mk_param ~slot:0 [ 10 ] in
          let bp = mk_param ~slot:1 [ 10 ] in
          let c = mk_param ~slot:2 [ 5; 2 ] in
          let ab = T.binary ~op:`Add ~lhs:a ~rhs:bp in
          let new_shape = mk_shape [ 5; 2 ] in
          let reshaped = T.reshape ~src:ab ~shape:new_shape in
          let result = T.binary ~op:`Add ~lhs:reshaped ~rhs:c in
          wrap_sink b result);
      (* test_binop_permute_fusion *)
      pipeline_test "binop permute fusion" ~expected_calls:1 (fun b ->
          let a = mk_param ~slot:0 [ 2; 5 ] in
          let bp = mk_param ~slot:1 [ 2; 5 ] in
          let c = mk_param ~slot:2 [ 5; 2 ] in
          let ab = T.binary ~op:`Add ~lhs:a ~rhs:bp in
          let permed = T.permute ~src:ab ~order:[ 1; 0 ] in
          let result = T.binary ~op:`Add ~lhs:permed ~rhs:c in
          wrap_sink b result);
      (* test_diamond_folded *)
      pipeline_test "diamond folded" ~expected_calls:1 (fun b ->
          let a = mk_param ~slot:0 [ 10 ] in
          let bp = mk_param ~slot:1 [ 10 ] in
          let c = mk_param ~slot:2 [ 10 ] in
          let d = mk_param ~slot:3 [ 10 ] in
          let ab = T.binary ~op:`Add ~lhs:a ~rhs:bp in
          let abc = T.binary ~op:`Add ~lhs:ab ~rhs:c in
          let abd = T.binary ~op:`Add ~lhs:ab ~rhs:d in
          let result = T.binary ~op:`Add ~lhs:abc ~rhs:abd in
          wrap_sink b result);
      (* test_fold_double_unary *)
      pipeline_test "fold double unary" ~expected_calls:1 (fun b ->
          let param = mk_param ~slot:0 [ 2 ] in
          let red =
            T.reduce_axis ~src:param ~op:`Add ~axes:[ 0 ]
          in
          let sq = T.unary ~op:`Sqrt ~src:red in
          let neg = T.unary ~op:`Neg ~src:sq in
          wrap_sink b neg);
      (* test_reduce_reshape_binop_fusion *)
      pipeline_test "reduce reshape binop fusion" ~expected_calls:1
        (fun b ->
          let a = mk_param ~slot:0 [ 10; 10 ] in
          let bp = mk_param ~slot:1 [ 10 ] in
          let red =
            T.reduce_axis ~src:a ~op:`Add ~axes:[ 0 ]
          in
          let new_shape = mk_shape [ 10 ] in
          let reshaped = T.reshape ~src:red ~shape:new_shape in
          let result = T.binary ~op:`Add ~lhs:reshaped ~rhs:bp in
          wrap_sink b result);
      (* test_reduce_permute_binop_fusion *)
      pipeline_test "reduce permute binop fusion" ~expected_calls:1
        (fun b ->
          let a = mk_param ~slot:0 [ 10; 10; 10 ] in
          let bp = mk_param ~slot:1 [ 10; 10; 1 ] in
          let red =
            T.reduce_axis ~src:a ~op:`Add ~axes:[ 0 ]
          in
          let permed = T.permute ~src:red ~order:[ 2; 1; 0 ] in
          let result = T.binary ~op:`Add ~lhs:permed ~rhs:bp in
          wrap_sink b result);
      (* test_push_permute_through_reshape *)
      pipeline_test "push permute through reshape" ~expected_calls:1
        (fun b ->
          let a = mk_param ~slot:0 [ 16; 16 ] in
          let bp = mk_param ~slot:1 [ 16; 16 ] in
          let ab = T.binary ~op:`Add ~lhs:a ~rhs:bp in
          let s4 = mk_shape [ 4; 4; 4; 4 ] in
          let reshaped = T.reshape ~src:ab ~shape:s4 in
          let permed =
            T.permute ~src:reshaped ~order:[ 2; 3; 0; 1 ]
          in
          wrap_sink b permed);
      (* test_multistage_reduce *)
      pipeline_test "multistage reduce" ~expected_calls:1 (fun b ->
          let a = mk_param ~slot:0 [ 32; 32; 32 ] in
          let red1 =
            T.reduce_axis ~src:a ~op:`Add ~axes:[ 2 ]
          in
          let relu =
            T.binary ~op:`Max ~lhs:red1
              ~rhs:(T.const (C.float D.Val.float32 0.0) D.float32)
          in
          let new_shape = mk_shape [ 32; 32 ] in
          let reshaped = T.reshape ~src:relu ~shape:new_shape in
          let red2 =
            T.reduce_axis ~src:reshaped ~op:`Add ~axes:[ 1 ]
          in
          wrap_sink b red2);
      (* test_children_dont_push:
         TODO: should be 1 kernel. remove_bufferize correctly identifies the
         removable bufferize but the substitution (inlining ranges into source)
         is not yet implemented. *)
      pipeline_test "children dont push" ~expected_calls:2 (fun b ->
          let a = mk_param ~slot:0 [ 10; 10; 1 ] in
          let bp = mk_param ~slot:1 [ 10; 10; 1 ] in
          let ab = T.binary ~op:`Add ~lhs:a ~rhs:bp in
          let exp_shape = mk_shape [ 10; 10; 10 ] in
          let expanded = T.expand ~src:ab ~shape:exp_shape in
          let permed = T.permute ~src:ab ~order:[ 2; 1; 0 ] in
          let result = T.binary ~op:`Add ~lhs:expanded ~rhs:permed in
          wrap_sink b result);
      (* test_reduce_permute_nofuse *)
      pipeline_test "reduce permute nofuse" ~expected_calls:1 (fun b ->
          let x = mk_param ~slot:0 [ 32; 32; 32 ] in
          let y = mk_param ~slot:1 [ 32; 32 ] in
          let red =
            T.reduce_axis ~src:x ~op:`Add ~axes:[ 2 ]
          in
          let new_shape = mk_shape [ 32; 32 ] in
          let reshaped = T.reshape ~src:red ~shape:new_shape in
          let permed = T.permute ~src:reshaped ~order:[ 1; 0 ] in
          let result = T.binary ~op:`Add ~lhs:permed ~rhs:y in
          wrap_sink b result);
    ]

(* Reshape merge (tested through get_kernel_graph pipeline) *)

let reshape_merge_tests =
  group "reshape merge"
    [
      (* Adjacent reshapes should be merged by earliest_rewrites. Verified
         indirectly: if they weren't merged, the graph might produce
         incorrect kernel structure. We test that the pipeline handles
         Reshape(Reshape(x, s1), s2) without error. *)
      pipeline_test "reshape chain produces 1 kernel" ~expected_calls:1
        (fun b ->
          let param = mk_param ~slot:0 [ 4; 4 ] in
          let s1 = mk_shape [ 16 ] in
          let r1 = T.reshape ~src:param ~shape:s1 in
          let s2 = mk_shape [ 2; 8 ] in
          let r2 = T.reshape ~src:r1 ~shape:s2 in
          let other = mk_param ~slot:1 [ 2; 8 ] in
          let result = T.binary ~op:`Add ~lhs:r2 ~rhs:other in
          wrap_sink b result);
    ]

(* Main *)

let () =
  run "Schedule.Rangeify"
    [
      is_always_contiguous_tests;
      new_range_tests;
      apply_movement_op_tests;
      run_rangeify_tests;
      get_kernel_graph_tests;
      reshape_merge_tests;
    ]
