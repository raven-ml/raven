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
module Bound = Tolk_uop.Bound
open Tolk
module C = Tolk_uop.Const
module D = Tolk_uop.Dtype
module Ops = Tolk_uop.Ops
module U = Tolk_uop.Uop
module Ak = Tolk_uop.Axis_type

(* Extract an int from a Const value, assuming it's Int. *)
let const_to_int (v : C.t) : int =
  match C.view v with C.Int n -> Z.to_int n | _ -> failwith "not Int"

let const_to_bool (v : C.t) : bool =
  match C.view v with C.Bool b -> b | _ -> failwith "not Bool"

(* Helpers *)

let op_is op u = Ops.equal (U.op u) op

let const_value u =
  if op_is Ops.Const u then U.Arg.as_value (U.arg u) else None

let const_int_uop u =
  match const_value u with
  | Some value -> const_to_int value
  | None -> fail "expected Const"

let const_bool_uop u =
  match const_value u with
  | Some value -> const_to_bool value
  | None -> fail "expected Const"

let first_src u = (U.src u).(0)

let shape_of u =
  let rec concrete = function
    | [] -> Some []
    | dim :: dims -> Option.bind (U.const_int_value dim) (fun dim ->
        Option.map (fun dims -> dim :: dims) (concrete dims)) in
  Option.bind (U.shape_opt u) concrete

let is_always_contiguous u = Indexing.always_contiguous (U.op u)

(* Emit a shape-encoding node from a concrete int list.
   For 1-D: emits a single Const index.
   For N-D: emits a Vectorize of Const index nodes. *)
let mk_shape (dims : int list) : U.t =
  let ids = List.map (fun s -> U.const (C.int D.weakint s)) dims in
  match ids with
  | [ d ] -> d
  | ds ->
      U.stack ds

(* Emit a PARAM with a known shape and CPU device. *)
let mk_param ~idx (shape : int list) : U.t =
  let shape_id = if shape = [] then None else Some (mk_shape shape) in
  let dev = U.Single "CPU" in
  U.param ~slot:idx ~dtype:D.float32 ?shape:shape_id ~device:dev ()

let index_of x l =
  let rec go i = function
    | [] -> raise Not_found
    | y :: _ when y = x -> i
    | _ :: t -> go (i + 1) t
  in
  go 0 l

(* Broadcast [src] (concrete shape [from_shape]) to [to_shape] using the
   primitive expand, which only prepends dims: squeeze the size-1 axes that
   grow, EXPAND them onto the front, then permute back into position. *)
let broadcast_to src ~from_shape ~to_shape =
  if from_shape = to_shape then src
  else begin
    let nd = List.length from_shape in
    let n_left = List.length to_shape - nd in
    let all = List.init nd Fun.id in
    let expand_at =
      List.filter
        (fun i ->
          List.nth from_shape i = 1 && List.nth to_shape (n_left + i) <> 1)
        all
    in
    let kept = List.filter (fun i -> not (List.mem i expand_at)) all in
    let squeezed_shape = List.map (List.nth from_shape) kept in
    let squeezed =
      if squeezed_shape = from_shape then src
      else U.reshape ~src ~shape:(mk_shape squeezed_shape)
    in
    let expand_dims =
      List.filteri (fun i _ -> i < n_left) to_shape
      @ List.map (fun i -> List.nth to_shape (n_left + i)) expand_at
    in
    let expanded =
      if expand_dims = [] then squeezed
      else U.expand ~src:squeezed ~dims:(mk_shape expand_dims)
    in
    let ne = List.length expand_at in
    let perm =
      List.init n_left Fun.id
      @ List.init nd (fun i ->
            n_left
            + (if List.mem i expand_at then index_of i expand_at
               else ne + index_of i kept))
    in
    if perm = List.init (List.length to_shape) Fun.id then expanded
    else U.permute ~src:expanded ~order:perm
  end

(* Count CALL nodes in a program. *)
let count_calls (root : U.t) : int =
  let n = ref 0 in
  List.iter (fun node ->
    if op_is Ops.Call node then incr n)
    (U.toposort root);
  !n

let find_node pred root =
  match List.find_opt pred (U.toposort root) with
  | Some u -> u
  | None -> fail "expected node"

let rec has_unindexed_param u =
  match U.op u with
  | Ops.Index -> false
  | Ops.Param -> Option.is_some (shape_of u)
  | _ -> Array.exists has_unindexed_param (U.src u)

(* Wrap an expression in CONTIGUOUS -> SINK for get_kernel_graph. *)
let wrap_sink (src : U.t) : U.t =
  let c = U.contiguous ~src () in
  U.sink [ c ]

(* Build a program from a builder function and run get_kernel_graph.
   Returns the kernel graph and CALL count. *)
let run_pipeline (build_fn : unit -> U.t) : U.t * int =
  let _sink = build_fn () in
  let result = Rangeify.get_kernel_graph (build_fn ()) in
  (result, count_calls result)

(* is_always_contiguous tests *)

let dummy = U.const (C.int D.weakint 0)
let dummy2 = U.const (C.int D.weakint 1)
let weak_int n = U.const (C.int D.weakint n)

let is_always_contiguous_tests =
  group "is_always_contiguous"
    [
      test "contiguous" (fun () ->
          let dummy = U.const (C.int D.weakint 0) in
          is_true (is_always_contiguous (U.contiguous ~src:dummy ())));
      test "after with store (assign pattern)" (fun () ->
          let dummy = U.const (C.int D.weakint 0) in
          let dummy2 = U.const (C.int D.weakint 1) in
          (* AFTER is a buffer identity — always contiguous *)
          is_true (is_always_contiguous (U.after ~src:dummy ~deps:[ dummy2 ])));
      (* A COPY is turned into a store before rangeify, so it is no buffer
         identity of its own. *)
      test "copy" (fun () ->
          let dummy = U.const (C.int D.int32 0) in
          is_false
            (is_always_contiguous
               (U.copy ~src:dummy ~device:(U.Single "CPU") ())));
      test "buffer" (fun () ->
          is_true
            (is_always_contiguous
               (U.buffer ~slot:0 ~device:(U.Single "CPU")
                  ~shape:(mk_shape [ 4 ]) ~dtype:D.float32 ())));
      test "const" (fun () ->
          is_true (is_always_contiguous (U.const_int 0)));
      test "param" (fun () ->
          is_true (is_always_contiguous (U.param ~slot:0 ~dtype:D.float32 ())));
      test "call" (fun () ->
          is_true
            (is_always_contiguous
               (U.call
                  ~body:(U.sink [])
                  ~args:[]
                  ~info:
                    {
                      aux = None;
                      grad_fxn = None;
                      name = None;
                      precompile = false;
                      precompile_backward = false;
                      dtype = Tolk_uop.Dtype.void;
                    })));
      test "reshape not contiguous" (fun () ->
          is_false (is_always_contiguous (U.reshape ~src:dummy ~shape:dummy2)));
      test "expand not contiguous" (fun () ->
          is_false (is_always_contiguous (U.expand ~src:dummy ~dims:dummy2)));
      test "reduce_axis not contiguous" (fun () ->
          is_false
            (is_always_contiguous
               (U.reduce_axis ~src:(mk_param ~idx:0 [ 4 ]) ~op:Ops.Add
                  ~axes:[ 0 ])));
      test "unary not contiguous" (fun () ->
          is_false (is_always_contiguous (U.alu_unary ~op:Ops.Neg ~src:dummy)));
      test "binary not contiguous" (fun () ->
          is_false
            (is_always_contiguous
               (U.alu_binary ~op:Ops.Add ~lhs:dummy ~rhs:dummy2)));
    ]

(* new_range tests *)

let new_range_tests =
  group "new_range"
    [
      test "size 1 gives const 0" (fun () ->
          
          let ctx = Indexing.create_context () in
          let id = Indexing.new_range ctx 1 ~kind:Ak.Weak () in
          equal int 0 (const_int_uop id));
      test "symbolic size resolving to 1 gives const 0" (fun () ->
          let ctx = Indexing.create_context () in
          let two = U.const (C.int D.weakint 2) in
          let one = U.const (C.int D.weakint 1) in
          let size = U.alu_binary ~op:Ops.Sub ~lhs:two ~rhs:one in
          let id = Indexing.new_range_expr ctx size ~kind:Ak.Weak () in
          equal int 0 (const_int_uop id));
      test "size 0 gives Range (resolve(s!=1) is true)" (fun () ->
          
          let ctx = Indexing.create_context () in
          let id = Indexing.new_range ctx 0 ~kind:Ak.Weak () in
          (match U.as_range id with
          | Some { size; axis; kind; _ } ->
              equal int 0 axis;
              is_true (kind = Ak.Weak);
              equal int 0 (const_int_uop size)
          | _ -> fail "expected Range for size 0"));
      test "size > 1 gives Range" (fun () ->
          
          let ctx = Indexing.create_context () in
          let id = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
          (match U.as_range id with
          | Some { size; axis; kind; _ } ->
              equal int 0 axis;
              is_true (kind = Ak.Weak);
              equal int 4 (const_int_uop size)
          | _ -> fail "expected Range for size > 1"));
      test "axis increments" (fun () ->
          
          let ctx = Indexing.create_context () in
          let id1 = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
          let id2 = Indexing.new_range ctx 8 ~kind:Ak.Weak () in
          let axis1 =
            match U.as_range id1 with
            | Some { axis; _ } -> axis
            | _ -> fail "expected Range"
          in
          let axis2 =
            match U.as_range id2 with
            | Some { axis; _ } -> axis
            | _ -> fail "expected Range"
          in
          equal int 0 axis1;
          equal int 1 axis2);
      test "kind propagates" (fun () ->
          
          let ctx = Indexing.create_context () in
          let id = Indexing.new_range ctx 8 ~kind:Ak.Reduce () in
          (match U.as_range id with
          | Some { kind; _ } -> is_true (kind = Ak.Reduce)
          | _ -> fail "expected Range"));
      test "range size returns existing range" (fun () ->
          let ctx = Indexing.create_context () in
          let existing =
            U.range ~size:(U.const (C.int D.weakint 4)) ~axis:7
              ~kind:Ak.Weak ()
          in
          let result = Indexing.new_range_expr ctx existing ~kind:Ak.Reduce () in
          is_true (result == existing);
          let next = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
          match U.as_range next with
          | Some { axis; _ } -> equal int 0 axis
          | None -> fail "expected Range");
    ]

let range_helper_tests =
  group "range helpers"
    [
      test "get_idx recurses through stack" (fun () ->
          let ctx = Indexing.create_context () in
          let rng0 = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
          let rng1 = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
          let gate = U.const_bool true in
          let gated =
            U.alu_ternary ~op:Ops.Where ~a:gate ~b:rng0 ~c:(U.invalid ())
          in
          let stacked = U.stack [ gated; rng1 ] in
          let result = U.get_idx stacked in
          is_true (op_is Ops.Stack result);
          equal int 2 (List.length (U.children result));
          is_true (List.nth (U.children result) 0 == rng0);
          is_true (List.nth (U.children result) 1 == rng1));
      test "get_valid recurses through stack" (fun () ->
          let ctx = Indexing.create_context () in
          let rng = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
          let gate = U.const_bool true in
          let gated =
            U.alu_ternary ~op:Ops.Where ~a:gate ~b:rng ~c:(U.invalid ())
          in
          let stacked = U.stack [ gated; U.invalid (); rng ] in
          let result = U.get_valid stacked in
          is_true (op_is Ops.Stack result);
          let lanes = U.children result in
          equal int 3 (List.length lanes);
          is_true (List.nth lanes 0 == gate);
          equal bool false (const_bool_uop (List.nth lanes 1));
          equal bool true (const_bool_uop (List.nth lanes 2)));
    ]

(* apply_movement_op tests *)

let apply_movement_op_tests =
  group "apply_movement_op"
    [
      (* SHRINK *)
      group "shrink"
        [
          test "zero offset passthrough" (fun () ->
              let param = mk_param ~idx:0 [ 4; 4 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
              let rng1 = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
              let before = mk_shape [ 0; 0 ] in
              let size = mk_shape [ 4; 4 ] in
              let v = U.shrink ~src:param ~offset:before ~size in
              let result =
                Indexing.apply_movement_op v
                  [ rng0; rng1 ]
              in
              (* zero offsets: output ranges should be same ids as input *)
              equal int (U.tag rng0) (U.tag (List.nth result 0));
              equal int (U.tag rng1) (U.tag (List.nth result 1)));
          test "nonzero offset adds" (fun () ->
              let param = mk_param ~idx:0 [ 4; 4 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
              let rng1 = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
              let before = mk_shape [ 1; 2 ] in
              let size = mk_shape [ 2; 2 ] in
              let v = U.shrink ~src:param ~offset:before ~size in
              let result =
                Indexing.apply_movement_op v
                  [ rng0; rng1 ]
              in
              equal int 2 (List.length result);
              is_true (op_is Ops.Add (List.nth result 0));
              is_true (op_is Ops.Add (List.nth result 1)));
        ];
      (* PERMUTE *)
      group "permute"
        [
          test "swap [1;0]" (fun () ->
              let param = mk_param ~idx:0 [ 4; 8 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
              let rng1 = Indexing.new_range ctx 8 ~kind:Ak.Weak () in
              let v = U.permute ~src:param ~order:[ 1; 0 ] in
              let result =
                Indexing.apply_movement_op v
                  [ rng0; rng1 ]
              in
              (* permute [1;0]: argsort = [1;0] → result = [rng1; rng0] *)
              equal int (U.tag rng1) (U.tag (List.nth result 0));
              equal int (U.tag rng0) (U.tag (List.nth result 1)));
          test "identity elides at construction" (fun () ->
              (* An identity permute returns its source, so no Permute node
                 ever reaches apply_movement_op — the range pass sees the ranges
                 straight through. *)
              let param = mk_param ~idx:0 [ 2; 3; 4 ] in
              is_true (U.permute ~src:param ~order:[ 0; 1; 2 ] == param));
        ];
      (* FLIP *)
      group "flip"
        [
          test "flip true reverses" (fun () ->
              let param = mk_param ~idx:0 [ 4 ] in
              let ctx = Indexing.create_context () in
              let rng = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
              let v = U.flip ~src:param ~dims:[ true ] in
              let result =
                Indexing.apply_movement_op v [ rng ]
              in
              (* (size-1) - r in the reference's a + b*(-1) form *)
              is_true (op_is Ops.Add (List.nth result 0)));
          test "flip false passthrough" (fun () ->
              let param = mk_param ~idx:0 [ 4 ] in
              let ctx = Indexing.create_context () in
              let rng = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
              let v = U.flip ~src:param ~dims:[ false ] in
              let result =
                Indexing.apply_movement_op v [ rng ]
              in
              equal int (U.tag rng) (U.tag (List.nth result 0)));
        ];
      (* EXPAND: prepends dims, so input ranges drop the leading dims *)
      group "expand"
        [
          test "one prepended dim drops leading range" (fun () ->
              let param = mk_param ~idx:0 [ 4 ] in
              let ctx = Indexing.create_context () in
              let r0 = Indexing.new_range ctx 3 ~kind:Ak.Weak () in
              let r1 = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
              let v = U.expand ~src:param ~dims:(mk_shape [ 3 ]) in
              let result =
                Indexing.apply_movement_op v [ r0; r1 ]
              in
              equal int 1 (List.length result);
              is_true (List.nth result 0 == r1));
          test "two prepended dims drop two leading ranges" (fun () ->
              let param = mk_param ~idx:0 [ 4 ] in
              let ctx = Indexing.create_context () in
              let r0 = Indexing.new_range ctx 2 ~kind:Ak.Weak () in
              let r1 = Indexing.new_range ctx 3 ~kind:Ak.Weak () in
              let r2 = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
              let v = U.expand ~src:param ~dims:(mk_shape [ 2; 3 ]) in
              let result =
                Indexing.apply_movement_op v [ r0; r1; r2 ]
              in
              equal int 1 (List.length result);
              is_true (List.nth result 0 == r2));
          test "symbolic prepended dim drops leading range" (fun () ->
              let n = U.variable ~name:"n" ~min_val:1 ~max_val:1024 () in
              let m = U.variable ~name:"m" ~min_val:1 ~max_val:1024 () in
              let param =
                U.param ~slot:0 ~dtype:D.float32 ~shape:n
                  ~device:(U.Single "CPU") ()
              in
              let expanded = U.expand ~src:param ~dims:m in
              let rng0 = U.range ~size:m ~axis:0 ~kind:Ak.Weak () in
              let rng1 = U.range ~size:n ~axis:1 ~kind:Ak.Weak () in
              let result =
                Indexing.apply_movement_op
                  expanded [ rng0; rng1 ]
              in
              equal int 1 (List.length result);
              is_true (List.nth result 0 == rng1));
        ];
      (* PAD *)
      group "pad"
        [
          test "zero pad passthrough" (fun () ->
              let param = mk_param ~idx:0 [ 4; 4 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
              let rng1 = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
              let before = mk_shape [ 0; 0 ] in
              let size = mk_shape [ 4; 4 ] in
              let v = U.pad ~src:param ~offset:before ~size in
              let result =
                Indexing.apply_movement_op v
                  [ rng0; rng1 ]
              in
              equal int (U.tag rng0) (U.tag (List.nth result 0));
              equal int (U.tag rng1) (U.tag (List.nth result 1)));
          test "nonzero pad creates WHERE" (fun () ->
              let param = mk_param ~idx:0 [ 4; 4 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 6 ~kind:Ak.Weak () in
              let rng1 = Indexing.new_range ctx 4 ~kind:Ak.Weak () in
              let before = mk_shape [ 2; 0 ] in
              let size = mk_shape [ 6; 4 ] in
              let v = U.pad ~src:param ~offset:before ~size in
              let result =
                Indexing.apply_movement_op v
                  [ rng0; rng1 ]
              in
              (* axis 0: pad_before=2 -> WHERE(valid, offset, invalid) *)
              is_true (op_is Ops.Where (List.nth result 0));
              (* axis 1: pad_before=0 -> passthrough *)
              equal int (U.tag rng1) (U.tag (List.nth result 1)));
          test "end-only pad creates WHERE (F2)" (fun () ->
              (* PAD with start=0, end=2 on a dim of size 4:
                 output range goes 0..5 but valid indices are only 0..3.
                 Must emit WHERE(r < 4, r, invalid). *)
              let param = mk_param ~idx:0 [ 4 ] in
              let ctx = Indexing.create_context () in
              let rng = Indexing.new_range ctx 6 ~kind:Ak.Weak () in
              let before = mk_shape [ 0 ] in
              let size = mk_shape [ 6 ] in
              let v = U.pad ~src:param ~offset:before ~size in
              let result =
                Indexing.apply_movement_op v [ rng ]
              in
              (* end padding nonzero -> WHERE must be generated *)
              is_true (op_is Ops.Where (List.nth result 0)));
          test "symbolic start pad preserves validity mask" (fun () ->
              let n = U.variable ~name:"n" ~min_val:1 ~max_val:1024 () in
              let size = U.alu_binary ~op:Ops.Add ~lhs:(weak_int 1) ~rhs:n in
              let param =
                U.param ~slot:0 ~dtype:D.float32 ~shape:n
                  ~device:(U.Single "CPU") ()
              in
              let padded =
                U.pad ~src:param ~offset:(weak_int 1) ~size
              in
              let rng = U.range ~size ~axis:0 ~kind:Ak.Weak () in
              let result =
                Indexing.apply_movement_op
                  padded [ rng ]
              in
              let gated = List.hd result in
              is_true (op_is Ops.Where gated);
              (* r - offset in the reference's a + b*(-1) form *)
              is_true (op_is Ops.Add (U.get_idx gated));
              is_false (op_is Ops.Const (U.get_valid gated)));
        ];
      (* RESHAPE *)
      group "reshape"
        [
          test "flatten [2;3] to [6]" (fun () ->
              (* apply_movement_op receives output ranges and returns input ranges.
                 Reshape [2;3] -> [6]: output shape [6], input shape [2;3].
                 Pass 1 output range, get back 2 input ranges. *)
              let param = mk_param ~idx:0 [ 2; 3 ] in
              let ctx = Indexing.create_context () in
              let rng_out = Indexing.new_range ctx 6 ~kind:Ak.Weak () in
              let new_shape = mk_shape [ 6 ] in
              let v = U.reshape ~src:param ~shape:new_shape in
              let result =
                Indexing.apply_movement_op v [ rng_out ]
              in
              equal int 2 (List.length result));
          test "unflatten [6] to [2;3]" (fun () ->
              (* Reshape [6] -> [2;3]: output shape [2;3], input shape [6].
                 Pass 2 output ranges, get back 1 input range. *)
              let param = mk_param ~idx:0 [ 6 ] in
              let ctx = Indexing.create_context () in
              let rng0 = Indexing.new_range ctx 2 ~kind:Ak.Weak () in
              let rng1 = Indexing.new_range ctx 3 ~kind:Ak.Weak () in
              let new_shape = mk_shape [ 2; 3 ] in
              let v = U.reshape ~src:param ~shape:new_shape in
              let result =
                Indexing.apply_movement_op v
                  [ rng0; rng1 ]
              in
              equal int 1 (List.length result));
          (* An identity reshape is never built, so its index is never
             re-derived. *)
          test "identity [4] to [4] is its source" (fun () ->
              let param = mk_param ~idx:0 [ 4 ] in
              is_true (U.reshape ~src:param ~shape:(mk_shape [ 4 ]) == param));
          test "symbolic flatten decomposes by symbolic shape" (fun () ->
              let n = U.variable ~name:"n" ~min_val:1 ~max_val:1024 () in
              let four = weak_int 4 in
              let flat = U.alu_binary ~op:Ops.Mul ~lhs:n ~rhs:four in
              let param =
                U.param ~slot:0 ~dtype:D.float32
                  ~shape:(U.stack [ n; four ])
                  ~device:(U.Single "CPU") ()
              in
              let reshaped = U.reshape ~src:param ~shape:flat in
              let rng = U.range ~size:flat ~axis:0 ~kind:Ak.Weak () in
              let result =
                Indexing.apply_movement_op
                  reshaped [ rng ]
              in
              equal int 2 (List.length result);
              is_true (op_is Ops.Floormod (List.nth result 1)));
        ];
    ]

(* run_rangeify tests *)

let materialize src =
  let dst = U.alloc ~slot:9000 ~dtype:(U.dtype src)
      ~shape:(U.stack (U.shape src)) () in
  U.store ~dst ~value:src ()

let run_rangeify_tests =
  group "run_rangeify"
    [
      test "realized node creates Realized" (fun () ->
          let param = mk_param ~idx:0 [ 4 ] in
          let store = materialize param in
          let sink = U.sink [ store ] in
          let ctx = Indexing.run_rangeify sink in
          (match Hashtbl.find_opt ctx.realize_map (U.tag store) with
          | Some (Indexing.Realized axes) ->
              equal (list int) [ 0 ] axes
          | Some Indexing.Marked -> fail "expected Realized, got Marked"
          | None -> fail "expected Realized, got None"));
      test "realized node has range_map entry" (fun () ->
          let param = mk_param ~idx:0 [ 4 ] in
          let store = materialize param in
          let sink = U.sink [ store ] in
          let ctx = Indexing.run_rangeify sink in
          is_true (Hashtbl.mem ctx.range_map (U.tag store)));
      (* The maps are keyed on the graph rangeify walked. A node the apply
         pass builds must not acquire an entry: hash-consing collapses nodes
         that differed only in movement ops, so an inherited entry can carry
         another node's rank and produce an INDEX with too few indices for
         the STAGE it reads. *)
      test "the apply pass adds no map entries" (fun () ->
          let x = mk_param ~idx:0 [ 4; 8 ] in
          let sc = U.reshape ~src:(mk_param ~idx:1 [ 1 ]) ~shape:(mk_shape []) in
          let recip = U.alu_unary ~op:Ops.Reciprocal ~src:sc in
          let loss = U.reduce_axis ~src:x ~op:Ops.Add ~axes:[ 0; 1 ] in
          let scalar_use = U.alu_binary ~op:Ops.Mul ~lhs:loss ~rhs:recip in
          let wide_use =
            U.alu_binary ~op:Ops.Mul ~lhs:x
              ~rhs:(broadcast_to recip ~from_shape:[] ~to_shape:[ 4; 8 ])
          in
          let sink =
            U.sink
              [ materialize scalar_use; materialize wide_use ]
          in
          let ctx = Indexing.run_rangeify sink in
          let keys tbl = Hashtbl.fold (fun k _ acc -> k :: acc) tbl [] in
          let ranged = keys ctx.range_map and realized = keys ctx.realize_map in
          ignore (Indexing.apply_rangeify_pass ctx sink);
          let added prev tbl =
            List.filter (fun k -> not (List.mem k prev)) (keys tbl)
          in
          equal (list int) [] (added ranged ctx.range_map);
          equal (list int) [] (added realized ctx.realize_map));
      test "elementwise inherits consumer ranges" (fun () ->
          let param = mk_param ~idx:0 [ 4 ] in
          let neg = U.alu_unary ~op:Ops.Neg ~src:param in
          let store = materialize neg in
          let sink = U.sink [ store ] in
          let ctx = Indexing.run_rangeify sink in
          is_true (Hashtbl.mem ctx.range_map (U.tag neg)));
      test "reduce creates reduce-kind ranges" (fun () ->
          let param = mk_param ~idx:0 [ 4; 4 ] in
          let red = U.reduce_axis ~src:param ~op:Ops.Add ~axes:[ 1 ] in
          let store = materialize red in
          let sink = U.sink [ store ] in
          let ctx = Indexing.run_rangeify sink in
          (match Hashtbl.find_opt ctx.range_map (U.tag red) with
          | Some (in_rngs, _out_rngs) ->
              equal int 2 (List.length in_rngs);
              (* The reduced axis is permuted to the front. *)
              (match U.as_range (List.nth in_rngs 0) with
              | Some { kind; _ } -> is_true (kind = Ak.Reduce)
              | None when op_is Ops.Const (List.nth in_rngs 0) ->
                  fail "expected Range for reduce axis, got Const"
              | _ -> fail "expected Range for reduce axis")
          | None -> fail "expected range_map entry for reduce"));
      test "movement op has different in and out ranges" (fun () ->
          let param = mk_param ~idx:0 [ 4; 8 ] in
          let perm = U.permute ~src:param ~order:[ 1; 0 ] in
          let store = materialize perm in
          let sink = U.sink [ store ] in
          let ctx = Indexing.run_rangeify sink in
          (match Hashtbl.find_opt ctx.range_map (U.tag perm) with
          | Some (in_rngs, out_rngs) ->
              equal int 2 (List.length in_rngs);
              equal int 2 (List.length out_rngs);
              is_true (List.nth out_rngs 1 == List.nth in_rngs 0);
              is_true (List.nth out_rngs 0 == List.nth in_rngs 1)
          | None -> fail "expected range_map entry for permute"));
      test "2D realized node has all axes" (fun () ->
          let param = mk_param ~idx:0 [ 4; 8 ] in
          let store = materialize param in
          let sink = U.sink [ store ] in
          let ctx = Indexing.run_rangeify sink in
          (match Hashtbl.find_opt ctx.realize_map (U.tag store) with
          | Some (Indexing.Realized axes) ->
              equal (list int) [ 0; 1 ] axes
          | _ -> fail "expected Realized with [0;1]"));
      test "symbolic param shape creates symbolic range size" (fun () ->
          let n = U.variable ~name:"n" ~min_val:1 ~max_val:1024 () in
          let param =
            U.param ~slot:0 ~dtype:D.float32 ~shape:(U.stack [ n ])
              ~device:(U.Single "CPU") ()
          in
          let store = materialize param in
          let sink = U.sink [ store ] in
          let ctx =
            Indexing.run_rangeify sink
          in
          match Hashtbl.find_opt ctx.range_map (U.tag store) with
          | Some ([ rng ], _) ->
              (match U.as_range rng with
              | Some { size; _ } -> is_true (size == n)
              | None -> fail "expected symbolic Range")
          | Some _ -> fail "expected one symbolic range"
          | None -> fail "expected range_map entry for symbolic store");
    ]

(* apply_rangeify_pass tests *)

let apply_rangeify_pass_tests =
  group "apply_rangeify_pass"
    [
      test "reduce indexes direct source before lowering" (fun () ->
          let param = mk_param ~idx:0 [ 4; 4 ] in
          let red = U.reduce_axis ~src:param ~op:Ops.Add ~axes:[ 1 ] in
          let root = U.sink [materialize red] in
          let ctx = Indexing.run_rangeify root in
          let lowered = Indexing.apply_rangeify_pass ctx root in
          let red =
            find_node
              (fun u ->
                match U.as_reduce u with
                | Some { num_axes = 0; ranges; _ } -> ranges <> []
                | _ -> false)
              lowered
          in
          match U.as_reduce red with
          | Some { src; _ } ->
              (match U.as_index src with
              | Some { ptr; _ } -> is_true (U.buf_uop ptr == U.buf_uop param)
              | None -> fail "expected indexed reduce source")
          | None -> fail "expected lowered reduce");
      test "pad where uses indexed child" (fun () ->
          let param = mk_param ~idx:0 [ 4 ] in
          let pad =
            U.pad ~src:param ~offset:(mk_shape [ 1 ]) ~size:(mk_shape [ 6 ])
          in
          let root = U.sink [materialize pad] in
          let ctx = Indexing.run_rangeify root in
          let lowered = Indexing.apply_rangeify_pass ctx root in
          let where =
            find_node
              (fun u ->
                op_is Ops.Where u
                &&
                match U.as_index (U.src u).(1) with
                | Some { ptr; _ } -> ptr == param
                | None -> false)
              lowered
          in
          let src = (U.src where).(1) in
          (match U.as_index src with
          | Some { ptr; _ } -> is_true (U.buf_uop ptr == U.buf_uop param)
          | None -> fail "expected indexed pad source"));
      (* Two consumers that disagree on their ranges stage the sum. *)
      test "staged elementwise indexes raw params" (fun () ->
          let a = mk_param ~idx:0 [ 2; 3 ] in
          let bp = mk_param ~idx:1 [ 2; 3 ] in
          let sum = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
          let root =
            U.sink
              [
                materialize sum;
                materialize (U.permute ~src:sum ~order:[ 1; 0 ]);
              ]
          in
          let ctx = Indexing.run_rangeify root in
          let lowered = Indexing.apply_rangeify_pass ctx root in
          let stage =
            find_node (fun u -> Option.is_some (U.as_stage u)) lowered
          in
          match U.as_stage stage with
          | Some { src; _ } -> is_false (has_unindexed_param src)
          | None -> fail "expected stage");
    ]

let early_movement_tests =
  group "early_movement_pass"
    [
      test "partial reshape index maps to source prefix" (fun () ->
          let param = mk_param ~idx:0 [ 2; 3; 4 ] in
          let reshaped = U.reshape ~src:param ~shape:(mk_shape [ 6; 4 ]) in
          let rng =
            U.range ~size:(U.const (C.int D.weakint 6)) ~axis:0
              ~kind:Ak.Weak ()
          in
          let indexed = U.index ~ptr:reshaped ~idxs:[rng] () in
          let result = Rangeify.early_movement_pass indexed in
          is_true (op_is Ops.Index result);
          is_true ((U.src result).(0) == param);
          equal int 3 (Array.length (U.src result)));
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
      test "rejects distinct written states of one buffer in a kernel" (fun () ->
          let buffer = mk_param ~idx:0 [ 4 ] in
          let written value =
            U.after ~src:buffer ~deps:[U.store ~dst:buffer ~value ()] in
          let first = written (mk_param ~idx:1 [ 4 ]) in
          let second = written (mk_param ~idx:2 [ 4 ]) in
          let sum = U.alu_binary ~op:Ops.Add ~lhs:first ~rhs:second in
          raises (Failure "cycle detected while indexing buffer") (fun () ->
              ignore (Rangeify.get_kernel_graph (wrap_sink sum))));
      (* test_basic_binop_fusion *)
      pipeline_test "elementwise fusion" ~expected_calls:1 (fun () ->
          let a = mk_param ~idx:0 [ 10 ] in
          let bp = mk_param ~idx:1 [ 10 ] in
          let c = mk_param ~idx:2 [ 10 ] in
          let ab = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
          let abc = U.alu_binary ~op:Ops.Add ~lhs:ab ~rhs:c in
          wrap_sink abc);
      (* test_mulacc_fusion *)
      pipeline_test "mulacc fusion" ~expected_calls:1 (fun () ->
          let a = mk_param ~idx:0 [ 10 ] in
          let bp = mk_param ~idx:1 [ 10 ] in
          let mul = U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:bp in
          let red =
            U.reduce_axis ~src:mul ~op:Ops.Add ~axes:[ 0 ]
          in
          wrap_sink red);
      (* test_binop_reshape_fusion *)
      pipeline_test "binop reshape fusion" ~expected_calls:1 (fun () ->
          let a = mk_param ~idx:0 [ 10 ] in
          let bp = mk_param ~idx:1 [ 10 ] in
          let c = mk_param ~idx:2 [ 5; 2 ] in
          let ab = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
          let new_shape = mk_shape [ 5; 2 ] in
          let reshaped = U.reshape ~src:ab ~shape:new_shape in
          let result = U.alu_binary ~op:Ops.Add ~lhs:reshaped ~rhs:c in
          wrap_sink result);
      (* test_binop_permute_fusion *)
      pipeline_test "binop permute fusion" ~expected_calls:1 (fun () ->
          let a = mk_param ~idx:0 [ 2; 5 ] in
          let bp = mk_param ~idx:1 [ 2; 5 ] in
          let c = mk_param ~idx:2 [ 5; 2 ] in
          let ab = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
          let permed = U.permute ~src:ab ~order:[ 1; 0 ] in
          let result = U.alu_binary ~op:Ops.Add ~lhs:permed ~rhs:c in
          wrap_sink result);
      (* test_diamond_folded *)
      pipeline_test "diamond folded" ~expected_calls:1 (fun () ->
          let a = mk_param ~idx:0 [ 10 ] in
          let bp = mk_param ~idx:1 [ 10 ] in
          let c = mk_param ~idx:2 [ 10 ] in
          let d = mk_param ~idx:3 [ 10 ] in
          let ab = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
          let abc = U.alu_binary ~op:Ops.Add ~lhs:ab ~rhs:c in
          let abd = U.alu_binary ~op:Ops.Add ~lhs:ab ~rhs:d in
          let result = U.alu_binary ~op:Ops.Add ~lhs:abc ~rhs:abd in
          wrap_sink result);
      (* test_fold_double_unary *)
      pipeline_test "fold double unary" ~expected_calls:1 (fun () ->
          let param = mk_param ~idx:0 [ 2 ] in
          let red =
            U.reduce_axis ~src:param ~op:Ops.Add ~axes:[ 0 ]
          in
          let sq = U.alu_unary ~op:Ops.Sqrt ~src:red in
          let neg = U.alu_unary ~op:Ops.Neg ~src:sq in
          wrap_sink neg);
      (* test_reduce_reshape_binop_fusion *)
      pipeline_test "reduce reshape binop fusion" ~expected_calls:1
        (fun () ->
          let a = mk_param ~idx:0 [ 10; 10 ] in
          let bp = mk_param ~idx:1 [ 10 ] in
          let red =
            U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 0 ]
          in
          let new_shape = mk_shape [ 10 ] in
          let reshaped = U.reshape ~src:red ~shape:new_shape in
          let result = U.alu_binary ~op:Ops.Add ~lhs:reshaped ~rhs:bp in
          wrap_sink result);
      pipeline_test "explicit contiguous materializes producer and consumer"
        ~expected_calls:2
        (fun () ->
          let x = mk_param ~idx:0 [ 32 ] in
          let y = mk_param ~idx:1 [ 32 ] in
          let z = mk_param ~idx:2 [ 32 ] in
          let add = U.alu_binary ~op:Ops.Add ~lhs:x ~rhs:y in
          let contig = U.contiguous ~src:add () in
          let result = U.alu_binary ~op:Ops.Add ~lhs:contig ~rhs:z in
          wrap_sink result);
      (* test_reduce_permute_binop_fusion *)
      pipeline_test "reduce permute binop fusion" ~expected_calls:1
        (fun () ->
          let a = mk_param ~idx:0 [ 10; 10; 10 ] in
          let bp = mk_param ~idx:1 [ 10; 10; 1 ] in
          (* [sum ~axis:0 ~keepdim:true]: the reduce drops the axis, the
             reshape puts it back at size one so the permute has rank 3. *)
          let red =
            U.reshape
              ~src:(U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 0 ])
              ~shape:(mk_shape [ 1; 10; 10 ])
          in
          let permed = U.permute ~src:red ~order:[ 2; 1; 0 ] in
          let result = U.alu_binary ~op:Ops.Add ~lhs:permed ~rhs:bp in
          wrap_sink result);
      (* test_push_permute_through_reshape *)
      pipeline_test "push permute through reshape" ~expected_calls:1
        (fun () ->
          let a = mk_param ~idx:0 [ 16; 16 ] in
          let bp = mk_param ~idx:1 [ 16; 16 ] in
          let ab = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
          let s4 = mk_shape [ 4; 4; 4; 4 ] in
          let reshaped = U.reshape ~src:ab ~shape:s4 in
          let permed =
            U.permute ~src:reshaped ~order:[ 2; 3; 0; 1 ]
          in
          wrap_sink permed);
      (* test_multistage_reduce *)
      pipeline_test "multistage reduce" ~expected_calls:1 (fun () ->
          let a = mk_param ~idx:0 [ 32; 32; 32 ] in
          let red1 =
            U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 2 ]
          in
          let relu =
            U.alu_binary ~op:Ops.Max ~lhs:red1
              ~rhs:(U.const (C.float D.float32 0.0))
          in
          let new_shape = mk_shape [ 32; 32 ] in
          let reshaped = U.reshape ~src:relu ~shape:new_shape in
          let red2 =
            U.reduce_axis ~src:reshaped ~op:Ops.Add ~axes:[ 1 ]
          in
          wrap_sink red2);
      (* test_children_dont_push *)
      pipeline_test "children dont push" ~expected_calls:1 (fun () ->
          let a = mk_param ~idx:0 [ 10; 10; 1 ] in
          let bp = mk_param ~idx:1 [ 10; 10; 1 ] in
          let ab = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
          let expanded =
            broadcast_to ab ~from_shape:[ 10; 10; 1 ] ~to_shape:[ 10; 10; 10 ]
          in
          let permed = U.permute ~src:ab ~order:[ 2; 1; 0 ] in
          let result = U.alu_binary ~op:Ops.Add ~lhs:expanded ~rhs:permed in
          wrap_sink result);
      (* test_reduce_permute_nofuse *)
      pipeline_test "reduce permute nofuse" ~expected_calls:1 (fun () ->
          let x = mk_param ~idx:0 [ 32; 32; 32 ] in
          let y = mk_param ~idx:1 [ 32; 32 ] in
          let red =
            U.reduce_axis ~src:x ~op:Ops.Add ~axes:[ 2 ]
          in
          let new_shape = mk_shape [ 32; 32 ] in
          let reshaped = U.reshape ~src:red ~shape:new_shape in
          let permed = U.permute ~src:reshaped ~order:[ 1; 0 ] in
          let result = U.alu_binary ~op:Ops.Add ~lhs:permed ~rhs:y in
          wrap_sink result);
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
        (fun () ->
          let param = mk_param ~idx:0 [ 4; 4 ] in
          let s1 = mk_shape [ 16 ] in
          let r1 = U.reshape ~src:param ~shape:s1 in
          let s2 = mk_shape [ 2; 8 ] in
          let r2 = U.reshape ~src:r1 ~shape:s2 in
          let other = mk_param ~idx:1 [ 2; 8 ] in
          let result = U.alu_binary ~op:Ops.Add ~lhs:r2 ~rhs:other in
          wrap_sink result);
    ]

let split_reduce_tests =
  group "split_reduce"
    [
      test "large reduce split rewrites reduce shape" (fun () ->
          let graph, calls =
            run_pipeline (fun () ->
                let param = mk_param ~idx:0 [ 65_536 ] in
                let red = U.reduce_axis ~src:param ~op:Ops.Add ~axes:[ 0 ] in
                wrap_sink red)
          in
          equal int 2 calls;
          let reduces =
            List.filter (fun u -> op_is Ops.Reduce u) (U.toposort graph)
          in
          is_true (List.length reduces >= 2);
          let has_split_range =
            List.exists
              (fun u ->
                match U.as_range u with
                | Some { size; _ } -> U.const_int_value size = Some 256
                | None -> false)
              (U.toposort graph)
          in
          is_true has_split_range);
    ]

(* Symbolic variables through the kernel split *)

let symbolic_variable_tests =
  (* The tensor graph a bound variable produces after callify: a named,
     ranged scalar PARAM sizing a SHRINK. *)
  let named_param () =
    U.param ~slot:1 ~dtype:D.weakint ~shape:(U.stack [])
      ~vmin_vmax:(Bound.int (1), Bound.int (7)) ~name:"start_pos" ~addrspace:D.Alu ()
  in
  let symbolic_shrink_sink () =
    let buf = mk_param ~idx:0 [ 8 ] in
    let size =
      U.alu_binary ~op:Ops.Add ~lhs:(named_param ()) ~rhs:(weak_int 1)
    in
    let shr = U.shrink ~src:buf ~offset:(weak_int 0) ~size in
    let red = U.reduce_axis ~src:shr ~op:Ops.Add ~axes:[ 0 ] in
    wrap_sink red
  in
  let kernel_body graph =
    match
      List.find_opt (fun u -> op_is Ops.Call u) (U.toposort graph)
    with
    | Some call ->
        (match U.as_call call with
         | Some { body; _ } -> body
         | None -> fail "expected CALL view")
    | None -> fail "expected a kernel CALL"
  in
  let canonical_variable body =
    List.find_opt
      (fun u ->
        match U.as_param u with
        | Some
            { param =
                { slot = -1; addrspace = D.Alu; name = Some "start_pos";
                  vmin_vmax = Some (lo, hi); _ };
              _ } ->
            Bound.equal lo (Bound.int 1) && Bound.equal hi (Bound.int 7)
        | _ -> false)
      (U.toposort ~enter_calls:true body)
  in
  group "symbolic variables"
    [
      test "STAGE shape reserves the maximum closed range extents" (fun () ->
          let n = U.variable ~name:"stage_extent" ~min_val:1 ~max_val:8 ~param:true () in
          let row = U.range ~size:n ~axis:0 ~kind:Ak.Weak () in
          let col = U.range ~size:(weak_int 3) ~axis:1 ~kind:Ak.Weak () in
          let value = U.O.(row * int_ 10 + col) in
          let stage = U.stage ~src:value ~ranges:[row; col]
              ~opts:{device = Some (U.Single "CPU"); addrspace = D.Global;
                     removable = false} in
          equal (option (list int)) (Some [8; 3]) (shape_of stage);
          match U.as_range row with
          | Some range -> is_true ~msg:"the loop retains its active bound" (range.size == n)
          | None -> fail "expected the symbolic loop");
      test "named PARAM normalises to the canonical variable" (fun () ->
          let graph = Rangeify.get_kernel_graph (symbolic_shrink_sink ()) in
          let body = kernel_body graph in
          is_true ~msg:"kernel body contains the canonical variable"
            (Option.is_some (canonical_variable body)));
      test "program_info_from_sink collects the variable" (fun () ->
          let graph = Rangeify.get_kernel_graph (symbolic_shrink_sink ()) in
          let body = kernel_body graph in
          let info = U.program_info_from_sink body in
          let var_names =
            List.filter_map
              (fun v ->
                match U.as_param v with
                | Some { param = { name; _ }; _ } -> name
                | None -> None)
              info.vars
          in
          equal (list string) [ "start_pos" ] var_names);
    ]

let shape_queries_release_graphs () =
  let weak = Stdlib.Weak.create 1 in
  let[@inline never] populate () =
    let p = U.param ~slot:898127 ~dtype:D.float32
        ~shape:(U.stack [ U.const_int 4; U.const_int 8 ]) () in
    let view = U.permute ~src:p ~order:[ 1; 0 ] in
    ignore (Prepare.detect_expanded view);
    Stdlib.Weak.set weak 0 (Some view)
  in
  populate ();
  Gc.full_major ();
  Gc.full_major ();
  is_false ~msg:"shape queries do not retain the tensor graph"
    (Stdlib.Weak.check weak 0)

let stack_selection_tests =
  group "stack selection"
    (List.map (fun n ->
         test (Printf.sprintf "%d sources preserve selection and bound depth" n)
           (fun () ->
             let index = U.variable ~name:"table_index" ~min_val:(-1) ~max_val:n () in
             let stack = U.stack (List.init n (fun i -> U.const_int (i * 3 + 7))) in
             let ctx = Indexing.create_context () in
             Hashtbl.add ctx.range_map (U.tag stack) ([], [ index ]);
             let selected = Indexing.apply_rangeify_pass ctx stack in
             let indices = if n <= 17 then List.init (n + 2) (fun i -> i - 1)
               else [ -1; 0; 7; 8; 15; 16; 255; 256; 511; 512; 1015; 1016; 1023; n ] in
             List.iter (fun i ->
                 let expected = (if i < 0 || i >= n then n - 1 else i) * 3 + 7 in
                 equal int expected (U.sym_infer selected [ "table_index", Int64.of_int i ]))
               indices;
             let depths = Hashtbl.create (n * 2) in
             List.iter (fun node ->
                 let depth = Array.fold_left (fun acc src ->
                     max acc (Hashtbl.find depths (U.tag src))) 0 (U.src node) in
                 Hashtbl.add depths (U.tag node)
                   (depth + if op_is Ops.Where node then 1 else 0))
               (U.toposort selected);
             is_true ~msg:"large selections have logarithmic conditional depth"
               (Hashtbl.find depths (U.tag selected) <= 16)))
       [ 1; 8; 9; 17; 1024 ])

let stage_capacity_tests =
  let stage src ranges =
    U.stage ~src ~ranges
      ~opts:{device = Some (U.Single "CPU"); addrspace = D.Global; removable = false} in
  let capacity name expected build = test name (fun () ->
      let staged = build () in
      equal int expected (U.max_numel staged);
      let graph = Rangeify.get_kernel_graph (U.sink [staged]) in
      let allocated = U.toposort ~enter_calls:true graph
          |> List.filter (op_is Ops.Alloc) |> List.map U.max_numel in
      equal ~msg:"one allocation reserves the complete canonical STAGE shape"
        (list int) [expected] allocated) in
  let axes () =
    U.range ~size:(weak_int 4) ~axis:0 ~kind:Ak.Weak (),
    U.range ~size:(weak_int 3) ~axis:1 ~kind:Ak.Weak () in
  let scalar r s = U.cast ~src:U.O.(r * weak_int 10 + s) ~dtype:D.int32 in
  let vector r = U.stack [U.cast ~src:r ~dtype:D.int32;
                         U.cast ~src:U.O.(r + weak_int 1) ~dtype:D.int32] in
  group "stage capacity"
    [
      test "zero-coordinate stages reuse materialized inputs" (fun () ->
          let src = mk_param ~idx:9010 [4] in
          let prepared staged =
            Prepare.prepare_rangeify (U.sink [staged]) |> first_src in
          is_true ~msg:"attributed stages preserve the existing storage identity"
            (prepared (stage src []) == src);
          is_true ~msg:"bare stages have the same materialization contract"
            (prepared (U.contiguous ~src ~force:true ()) == src));
      test "preparation leaves indexed stages to rangeify" (fun () ->
          let r = fst (axes ()) in
          let staged = stage (U.cast ~src:r ~dtype:D.int32) [r] in
          let prepared = Prepare.prepare_rangeify (U.sink [staged]) |> first_src in
          is_true ~msg:"a coordinate-bearing stage is not a bare materialization"
            (prepared == staged));
      test "zero-coordinate stages reuse explicit output storage" (fun () ->
          let src = mk_param ~idx:9011 [4] and dst = mk_param ~idx:9012 [4] in
          let value = stage U.O.(src + weak_int 1) [] in
          let graph = Prepare.prepare_rangeify (U.sink [U.store ~dst ~value ()]) in
          equal int 0 (List.length (List.filter (op_is Ops.Alloc) (U.toposort graph)));
          is_true (List.exists (fun n -> match U.as_store n with
              | Some {dst = target; _} -> target == dst | None -> false) (U.toposort graph)));
      test "coordinate stages are not forwarded as scalar output values" (fun () ->
          let r = fst (axes ()) and dst = mk_param ~idx:9013 [4] in
          let value = stage (U.cast ~src:r ~dtype:D.float32) [r] in
          let graph = Prepare.prepare_rangeify (U.sink [U.store ~dst ~value ()]) in
          is_true (List.exists (op_is Ops.Stage) (U.toposort graph)));
      test "zero-coordinate stages preserve DISK view-copy normalization" (fun () ->
          let src = U.param ~slot:9014 ~dtype:D.int32 ~shape:(weak_int 8)
              ~device:(U.Single "DISK:stage-arity") () in
          let view = U.shrink ~src ~offset:(weak_int 2) ~size:(weak_int 3) in
          let copy = U.copy ~src:(stage view []) ~device:(U.Single "CPU") () in
          let graph = Prepare.prepare_rangeify (U.sink [copy]) in
          equal (list int) [8] (U.toposort graph |> List.filter (op_is Ops.Alloc)
              |> List.map U.max_numel));
      test "zero-coordinate stages expose the same contiguous call view" (fun () ->
          let src = U.buffer ~slot:9015 ~dtype:D.int32 ~shape:(weak_int 8)
              ~device:(U.Single "CPU") () in
          let dst = U.buffer ~slot:9016 ~dtype:D.int32 ~shape:(weak_int 3)
              ~device:(U.Single "CPU") () in
          let view = U.shrink ~src ~offset:(weak_int 2) ~size:(weak_int 3) in
          List.iter (fun value ->
              let sink = U.sink [U.after ~src:dst ~deps:[U.store ~dst ~value ()]] in
              match U.as_call (Callify.transform_to_call sink) with
              | Some {args; _} -> equal (list int) [3;3] (List.map U.max_numel args)
              | None -> fail "callification must produce a call")
            [stage view []; U.contiguous ~src:view ~force:true ()]);
      capacity "zero coordinates reserve one scalar" 1 (fun () ->
          let src = U.variable ~name:"stage_scalar" ~min_val:0 ~max_val:100
              ~dtype:D.int32 ~param:true () in
          stage U.O.(src + weak_int 1) []);
      capacity "one coordinate" 4 (fun () ->
          let r = fst (axes ()) in stage (U.cast ~src:r ~dtype:D.int32) [r]);
      capacity "multiple coordinates" 12 (fun () ->
          let r, s = axes () in stage (scalar r s) [r;s]);
      capacity "expression coordinates" 15 (fun () ->
          let r, s = axes () in stage (scalar r s) [U.O.(r + weak_int 1);s]);
      capacity "vector source with one coordinate" 8 (fun () ->
          let r = fst (axes ()) in stage (vector r) [r]);
      capacity "vector source with multiple coordinates" 24 (fun () ->
          let r, s = axes () in stage (vector r) [r;s]);
      test "mixed symbolic and expression coordinates require an explicit active shape" (fun () ->
          let n = U.variable ~name:"mixed_stage_size" ~min_val:1 ~max_val:4 ~param:true () in
          let r = U.range ~size:n ~axis:0 ~kind:Ak.Weak () in
          let s = U.range ~size:(weak_int 3) ~axis:1 ~kind:Ak.Weak () in
          let staged = stage (scalar r s) [r; U.O.(s + weak_int 1)] in
          match Rangeify.get_kernel_graph (U.sink [staged]) with
          | _ -> fail "an expression coordinate must not invent an active extent of one"
          | exception Invalid_argument message ->
              equal string "Rangeify.flatten_stage: symbolic stage coordinates must be ranges or constants" message);
      test "zero storage capacity needs no allocation" (fun () ->
          let r = U.range ~size:(weak_int 0) ~axis:0 ~kind:Ak.Weak () in
          let src = U.variable ~name:"empty_stage_value" ~min_val:0 ~max_val:100
              ~dtype:D.int32 ~param:true () in
          let staged = stage src [r] in
          equal int 0 (U.max_numel staged);
          let graph = Rangeify.get_kernel_graph (U.sink [staged]) in
          let storage = U.toposort ~enter_calls:true graph
              |> List.filter (fun n -> List.mem (U.op n) [Ops.Stage; Ops.Alloc; Ops.Call]) in
          equal ~msg:"preparation removes empty storage before allocation lowering"
            int 0 (List.length storage));
    ]

let symbolic_empty_shape_tests =
  let n = U.variable ~name:"empty_symbolic_extent" ~min_val:1 ~max_val:8
      ~param:true () in
  let source dims = U.expand ~src:(U.const (C.int D.int32 7))
      ~dims:(U.stack dims) in
  let prepared value = Prepare.prepare_rangeify (U.sink [value]) |> first_src in
  group "symbolic empty shapes"
    [
      test "a literal zero dimension clears a symbolic tensor" (fun () ->
          let empty = U.range ~size:(weak_int 0) ~axis:0 ~kind:Ak.Weak () in
          let value = U.stage ~src:(source [n]) ~ranges:[empty]
              ~opts:{device = Some (U.Single "CPU"); addrspace = D.Global; removable = false} in
          equal int 0 (U.sym_infer (U.base (prepared value)) []));
      test "an empty sum keeps the identity with a symbolic output" (fun () ->
          let value = U.reduce_axis ~src:(source [weak_int 0; n])
              ~op:Ops.Add ~axes:[0] in
          equal int 0 (U.sym_infer (U.base (prepared value)) []));
      test "an empty product keeps the identity with a symbolic output" (fun () ->
          let value = U.reduce_axis ~src:(source [weak_int 0; n])
              ~op:Ops.Mul ~axes:[0] in
          equal int 1 (U.sym_infer (U.base (prepared value)) []));
    ]

let kernel_split_keeps_independent_ranges () =
  let n = U.variable ~name:"store_count" ~min_val:1 ~max_val:8 ~param:true () in
  let m = U.variable ~name:"value_count" ~min_val:1 ~max_val:8 ~param:true () in
  let dst_range = U.range ~size:n ~axis:0 ~kind:Ak.Loop () in
  let value_range = U.range ~size:m ~axis:1 ~kind:Ak.Loop () in
  let dst = mk_param ~idx:93740 [8] in
  let store = U.store ~dst:(U.index ~ptr:dst ~idxs:[dst_range] ())
      ~value:(U.cast ~src:value_range ~dtype:D.float32) () in
  let graph = Rangeify.get_kernel_graph
      (U.sink [U.end_ ~value:store ~ranges:[dst_range; value_range]]) in
  let bounds = U.toposort ~enter_calls:true graph
      |> List.filter_map (fun node -> Option.map
          (fun (range : U.range_view) -> U.sym_infer range.size
             ["store_count", 2L; "value_count", 5L]) (U.as_range node))
      |> List.sort Int.compare in
  equal ~msg:"equal maximum extents do not identify independent store loops"
    (list int) [2; 5] bounds

let moved_after_symbolic_tests =
  let add_one src = U.alu_binary ~op:Ops.Add ~lhs:src
      ~rhs:(U.const (C.float D.float32 1.)) in
  let stored value =
    let dst = U.param_like value ~slot:93801 in
    U.after ~src:dst ~deps:[U.store ~dst ~value ()] in
  let consumer materialized source =
    Helpers.Context_var.with_context [B (Helpers.openpilot_hacks, 1)] (fun () ->
        let result = Prepare.prepare_rangeify
            (U.sink [materialized; add_one source]) in
        (U.src result).(1)) in
  let uses_materialization value =
    is_true ~msg:"consumer reads the existing materialization"
      (List.exists (op_is Ops.After) (U.toposort value)) in
  group "moved materializations"
    [
      test "a symbolic reshape reuses its materialization" (fun () ->
          let n = U.variable ~name:"moved_extent" ~min_val:1 ~max_val:8
              ~param:true () in
          let source = add_one (U.param ~slot:93800 ~dtype:D.float32
              ~shape:(U.stack [n; weak_int 2]) ~device:(U.Single "CPU") ()) in
          let moved = U.reshape ~src:source ~shape:(U.stack [weak_int 2; n]) in
          let result = consumer (U.contiguous ~src:moved ()) source in
          uses_materialization result;
          List.iter (fun size ->
              equal (list int) [size; 2]
                (List.map (fun dim -> U.sym_infer dim
                    ["moved_extent", Int64.of_int size]) (U.shape result))) [3; 5]);
      test "a padded selection shrinks the reused materialization" (fun () ->
          let source = mk_param ~idx:93802 [2] in
          let padded = U.pad ~src:source ~offset:(weak_int 1) ~size:(weak_int 4) in
          let mask = U.param ~slot:93803 ~dtype:D.bool ~shape:(weak_int 4) () in
          let moved = U.alu_ternary ~op:Ops.Where ~a:mask ~b:padded ~c:(U.invalid ()) in
          let result = consumer (stored moved) source in
          uses_materialization result;
          equal (list int) [2] (U.max_shape result);
          let reused = List.find (fun value ->
              List.exists (op_is Ops.After) (U.toposort value))
              (U.children result) in
          (match Indexing.storage_window reused with
           | Some (_, offset) -> equal ~msg:"inverse pad byte offset" int 4 offset
           | None -> fail "expected a contiguous window of the stored result"));
      test "a staged permutation reuses its materialization" (fun () ->
          let source = add_one (mk_param ~idx:93804 [2; 3]) in
          let moved = U.permute ~src:source ~order:[1; 0] in
          let result = consumer (U.contiguous ~src:moved ()) source in
          uses_materialization result;
          equal (list int) [2; 3] (U.max_shape result));
    ]

let rangeify_symbolic_storage_tests =
  group "symbolic storage views"
    [
      test "stacked symbolic reshapes keep their logical shape" (fun () ->
          let n = U.variable ~name:"stacked_view_size" ~min_val:1 ~max_val:4 ~param:true () in
          let lane slot device =
            let p = U.param ~slot ~dtype:D.float32 ~shape:(weak_int 4)
                ~device:(U.Single device) () in
            U.reshape ~src:(U.shrink ~src:p ~offset:(weak_int 0) ~size:n)
              ~shape:(U.stack [n; weak_int 1]) in
          let stacked = U.mstack [lane 94001 "CPU:0"; lane 94002 "CPU:1"] in
          let graph = Rangeify.get_kernel_graph (U.sink [stacked]) in
          let result = first_src graph in
          equal ~msg:"moving reshapes through a stack preserves symbolic dimensions"
            (list int) [2;1]
            (List.map (fun d -> U.sym_infer d ["stacked_view_size", 2L]) (U.shape result)));
      test "stage removal rejects ambiguous partial indexing" (fun () ->
          let n = U.variable ~name:"partial_row_size" ~min_val:1 ~max_val:4 ~param:true () in
          let row = U.range ~size:(weak_int 2) ~axis:0 ~kind:Ak.Weak () in
          let column = U.range ~size:n ~axis:1 ~kind:Ak.Weak () in
          let staged = U.stage ~src:(U.cast ~src:U.Promoting.(row * weak_int 10 + column)
              ~dtype:D.int32) ~ranges:[row; column]
              ~opts:{device = Some (U.Single "CPU"); addrspace = D.Global; removable = true} in
          let selected = U.index ~ptr:staged ~idxs:[weak_int 1] () in
          let value = U.shrink ~src:selected ~offset:(weak_int 0) ~size:n in
          let dst = U.param ~slot:94003 ~dtype:D.int32 ~shape:(weak_int 4)
              ~device:(U.Single "CPU") () in
          let dst = U.shrink ~src:dst ~offset:(weak_int 0) ~size:n in
          raises (Invalid_argument
              "Rangeify: index on wrong stage, 2 ranges vs 1 index sources")
            (fun () -> ignore
                (Rangeify.get_kernel_graph (U.sink [U.store ~dst ~value ()]))));
    ]

(* Main *)

let () =
  run "Schedule.Rangeify"
    [
      test "kernel splitting preserves independent symbolic ranges"
        kernel_split_keeps_independent_ranges;
      symbolic_empty_shape_tests;
      rangeify_symbolic_storage_tests;
      moved_after_symbolic_tests;
      stage_capacity_tests;
      stack_selection_tests;
      test "Shape queries release graphs" shape_queries_release_graphs;
      is_always_contiguous_tests;
      new_range_tests;
      range_helper_tests;
      apply_movement_op_tests;
      run_rangeify_tests;
      apply_rangeify_pass_tests;
      early_movement_tests;
      get_kernel_graph_tests;
      reshape_merge_tests;
      split_reduce_tests;
      symbolic_variable_tests;
    ]
