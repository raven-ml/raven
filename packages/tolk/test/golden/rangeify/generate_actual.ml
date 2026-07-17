(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generates .actual files for rangeify pipeline golden tests. Each file
   contains tolk's rendered output for a specific backend + test case after
   running the full pipeline: Tensor.t -> Rangeify.get_kernel_graph ->
   Kernel.t -> Codegen.full_rewrite_to_sink -> Linearizer.linearize ->
   Renderer.render. Dune diff rules compare .actual against .expected. *)

open Tolk
open Tolk_uop
module U = Uop
module C = Const
module D = Dtype

(* Helpers *)

(* Emit a shape-encoding node from a concrete int list. *)
let mk_shape b (dims : int list) : U.t =
  let ids = List.map (fun s -> U.const (C.int D.weakint s)) dims in
  match ids with
  | [ d ] -> d
  | ds ->
      U.stack ds

(* Emit a PARAM with a known shape and CPU device. *)
let mk_param ?(dtype = D.float32) b ~slot (shape : int list) : U.t =
  let shape_id = if shape = [] then None else Some (mk_shape b shape) in
  let dev = U.Single "CPU" in
  U.param ~slot ~dtype ?shape:shape_id ~device:dev ()

let mk_ptr_param b ~slot size : U.t =
  U.param ~slot ~dtype:D.float32 ~shape:(mk_shape b [ size ])
    ~device:(U.Single "CPU") ()

(* Wrap source(s) in CONTIGUOUS -> SINK. *)
let wrap_sink b (srcs : U.t list) : U.t =
  let contigs =
    List.map (fun src -> U.contiguous ~src ()) srcs
  in
  U.sink contigs

let scheduled_kernel ?(name = "") ?(optimize = true) args body =
  let kernel_info : U.kernel_info =
    {
      name;
      axis_types = [];
      dont_use_locals = false;
      applied_opts = [];
      opts_to_apply = None;
      estimates = None;
      beam = 0;
    }
  in
  let info : U.call_info =
    {
      grad_fxn = None;
      name = None;
      precompile = false;
      precompile_backward = false;
      aux = None;
    }
  in
  let body = U.sink ~kernel_info [ body ] in
  let body = if optimize then body else U.with_tag "1" body in
  U.sink [ U.call ~body ~args ~info ]

(* Extract kernel ASTs from CALL nodes in topological (id) order. *)
let extract_kernels (root : U.t) : U.t list =
  let kernels = ref [] in
  List.iter (fun node ->
    match U.as_call node with
    | Some { body; _ } -> kernels := body :: !kernels
    | _ -> ())
    (U.toposort root);
  List.rev !kernels

(* Extract kernel name from a pipeline-processed Sink. *)
let name_of_sink sink =
  match U.as_kernel_info sink with Some ki -> ki.name | None -> "kernel"

(* Run the full pipeline: Tensor.t -> rendered source string. *)
let tensor_to_source renderer (build_fn : unit -> U.t) : string =
  let program = build_fn () in
  let kernel_graph = Rangeify.get_kernel_graph program in
  let kernels = extract_kernels kernel_graph in
  let sources =
    List.map
      (fun k ->
        let processed =
          Codegen.full_rewrite_to_sink ~optimize:true renderer k
        in
        let name = name_of_sink processed in
        let prog = Linearizer.linearize processed in
        String.trim (Renderer.render renderer ~name prog))
      kernels
  in
  String.concat "\n---\n" sources

(* Tensor graph builders *)

(* Each builder constructs a Tensor.t graph matching the corresponding
   builder in generate_expected.py. *)

let build_elementwise_add b =
  let a = mk_param b ~slot:0 [ 256 ] in
  let bp = mk_param b ~slot:1 [ 256 ] in
  let add = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
  wrap_sink b [ add ]

let build_elementwise_3way b =
  let a = mk_param b ~slot:0 [ 256 ] in
  let bp = mk_param b ~slot:1 [ 256 ] in
  let c = mk_param b ~slot:2 [ 256 ] in
  let ab = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
  let abc = U.alu_binary ~op:Ops.Add ~lhs:ab ~rhs:c in
  wrap_sink b [ abc ]

let build_mulacc b =
  let a = mk_param b ~slot:0 [ 256 ] in
  let bp = mk_param b ~slot:1 [ 256 ] in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:bp in
  let red =
    U.reduce_axis ~src:mul ~op:Ops.Add ~axes:[ 0 ] in
  wrap_sink b [ red ]

let build_binop_reshape b =
  let a = mk_param b ~slot:0 [ 10 ] in
  let bp = mk_param b ~slot:1 [ 10 ] in
  let c = mk_param b ~slot:2 [ 5; 2 ] in
  let add = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
  let reshaped = U.reshape ~src:add ~shape:(mk_shape b [ 5; 2 ]) in
  let result = U.alu_binary ~op:Ops.Add ~lhs:reshaped ~rhs:c in
  wrap_sink b [ result ]

let build_binop_permute b =
  let a = mk_param b ~slot:0 [ 2; 5 ] in
  let bp = mk_param b ~slot:1 [ 2; 5 ] in
  let c = mk_param b ~slot:2 [ 5; 2 ] in
  let add = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
  let permed = U.permute ~src:add ~order:[ 1; 0 ] in
  let result = U.alu_binary ~op:Ops.Add ~lhs:permed ~rhs:c in
  wrap_sink b [ result ]

let build_diamond b =
  let a = mk_param b ~slot:0 [ 10 ] in
  let bp = mk_param b ~slot:1 [ 10 ] in
  let c = mk_param b ~slot:2 [ 10 ] in
  let d = mk_param b ~slot:3 [ 10 ] in
  let ab = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
  let abc = U.alu_binary ~op:Ops.Add ~lhs:ab ~rhs:c in
  let abcab = U.alu_binary ~op:Ops.Add ~lhs:abc ~rhs:ab in
  let result = U.alu_binary ~op:Ops.Add ~lhs:abcab ~rhs:d in
  wrap_sink b [ result ]

let build_reduce_unary b =
  let a = mk_param b ~slot:0 [ 16 ] in
  let red =
    U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 0 ] in
  let sq = U.alu_unary ~op:Ops.Sqrt ~src:red in
  let neg = U.alu_unary ~op:Ops.Neg ~src:sq in
  wrap_sink b [ neg ]

let build_reduce_reshape_binop b =
  let a = mk_param b ~slot:0 [ 10; 10 ] in
  let bp = mk_param b ~slot:1 [ 10 ] in
  let red =
    U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 0 ] in
  let reshaped = U.reshape ~src:red ~shape:(mk_shape b [ 10 ]) in
  let result = U.alu_binary ~op:Ops.Add ~lhs:reshaped ~rhs:bp in
  wrap_sink b [ result ]

let build_reduce_permute_binop b =
  let a = mk_param b ~slot:0 [ 10; 10; 10 ] in
  let bp = mk_param b ~slot:1 [ 10; 10 ] in
  let red =
    U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 0 ] in
  let permed = U.permute ~src:red ~order:[ 1; 0 ] in
  let result = U.alu_binary ~op:Ops.Add ~lhs:permed ~rhs:bp in
  wrap_sink b [ result ]

let build_permute_through_reshape b =
  let a = mk_param b ~slot:0 [ 16; 16 ] in
  let bp = mk_param b ~slot:1 [ 16; 16 ] in
  let add = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
  let reshaped =
    U.reshape ~src:add ~shape:(mk_shape b [ 4; 4; 4; 4 ])
  in
  let permed = U.permute ~src:reshaped ~order:[ 2; 3; 0; 1 ] in
  wrap_sink b [ permed ]

let build_expand_permute b =
  let a = mk_param b ~slot:0 [ 10; 10; 1 ] in
  let bp = mk_param b ~slot:1 [ 10; 10; 1 ] in
  let ab = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bp in
  let expanded =
    U.broadcast_to ~src:ab ~shape:(mk_shape b [ 10; 10; 10 ])
  in
  let permed = U.permute ~src:ab ~order:[ 2; 1; 0 ] in
  let permed_expanded =
    U.broadcast_to ~src:permed ~shape:(mk_shape b [ 10; 10; 10 ])
  in
  let result = U.alu_binary ~op:Ops.Add ~lhs:expanded ~rhs:permed_expanded in
  wrap_sink b [ result ]

let build_shrink_fuse b =
  let a = mk_param b ~slot:0 [ 8192; 16 ] in
  let bp = mk_param b ~slot:1 [ 8192; 16 ] in
  let d = mk_param b ~slot:2 [ 1; 16 ] in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:bp in
  let before = mk_shape b [ 0; 0 ] in
  let size = mk_shape b [ 1; 16 ] in
  let shrunk = U.shrink ~src:mul ~offset:before ~size in
  let result = U.alu_binary ~op:Ops.Mul ~lhs:shrunk ~rhs:d in
  wrap_sink b [ result ]

let build_multistage_reduce b =
  let a = mk_param b ~slot:0 [ 32; 32; 32 ] in
  let red1 =
    U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 2 ] in
  let zero = U.const (C.float D.float32 0.0) in
  let relu = U.alu_binary ~op:Ops.Max ~lhs:red1 ~rhs:zero in
  let reshaped =
    U.reshape ~src:relu ~shape:(mk_shape b [ 32; 32 ])
  in
  let red2 =
    U.reduce_axis ~src:reshaped ~op:Ops.Add ~axes:[ 1 ] in
  wrap_sink b [ red2 ]

let build_two_sum b =
  let a = mk_param b ~slot:0 [ 64; 64 ] in
  let red0 =
    U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 0 ] in
  let red1 =
    U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 1 ] in
  let reshaped0 = U.reshape ~src:red0 ~shape:(mk_shape b [ 64 ]) in
  let reshaped1 = U.reshape ~src:red1 ~shape:(mk_shape b [ 64 ]) in
  let result = U.alu_binary ~op:Ops.Add ~lhs:reshaped0 ~rhs:reshaped1 in
  wrap_sink b [ result ]

let build_reduce_shrink b =
  let a = mk_param b ~slot:0 [ 32; 32 ] in
  let bp = mk_param b ~slot:1 [ 16 ] in
  let red =
    U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 1 ] in
  let reshaped = U.reshape ~src:red ~shape:(mk_shape b [ 32 ]) in
  let before = mk_shape b [ 0 ] in
  let size = mk_shape b [ 16 ] in
  let shrunk = U.shrink ~src:reshaped ~offset:before ~size in
  let result = U.alu_binary ~op:Ops.Add ~lhs:shrunk ~rhs:bp in
  wrap_sink b [ result ]

let build_contiguous_add b =
  let x = mk_param b ~slot:0 [ 32 ] in
  let y = mk_param b ~slot:1 [ 32 ] in
  let z = mk_param b ~slot:2 [ 32 ] in
  let add = U.alu_binary ~op:Ops.Add ~lhs:x ~rhs:y in
  let contig = U.contiguous ~src:add () in
  let result = U.alu_binary ~op:Ops.Add ~lhs:contig ~rhs:z in
  wrap_sink b [ result ]

let build_reshape_chain b =
  let a = mk_param b ~slot:0 [ 4; 4 ] in
  let bp = mk_param b ~slot:1 [ 2; 8 ] in
  let r1 = U.reshape ~src:a ~shape:(mk_shape b [ 16 ]) in
  let r2 = U.reshape ~src:r1 ~shape:(mk_shape b [ 2; 8 ]) in
  let result = U.alu_binary ~op:Ops.Add ~lhs:r2 ~rhs:bp in
  wrap_sink b [ result ]

let build_llama_rmsnorm b =
  let x = mk_param b ~slot:0 [ 2; 8 ] in
  let sq = U.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:x in
  let sum = U.reduce_axis ~src:sq ~op:Ops.Add ~axes:[ 1 ] in
  let mean =
    U.alu_binary ~op:Ops.Mul ~lhs:sum
      ~rhs:(U.const (C.float D.float32 0.125))
  in
  let eps =
    U.alu_binary ~op:Ops.Add ~lhs:mean
      ~rhs:(U.const (C.float D.float32 0.00001))
  in
  let sqrt = U.alu_unary ~op:Ops.Sqrt ~src:eps in
  let rsqrt = U.alu_unary ~op:Ops.Reciprocal ~src:sqrt in
  let result = U.reshape ~src:rsqrt ~shape:(mk_shape b [ 2 ]) in
  wrap_sink b [ result ]

let build_llama_ffn_gate b =
  let x = mk_param b ~slot:0 [ 2; 8 ] in
  let norm = mk_param b ~slot:1 [ 2 ] in
  let weight = mk_param b ~slot:2 [ 8 ] in
  let matrix = mk_param b ~slot:3 [ 8; 8 ] in
  let x3 = U.reshape ~src:x ~shape:(mk_shape b [ 2; 1; 8 ]) in
  let norm3 = U.reshape ~src:norm ~shape:(mk_shape b [ 2; 1; 1 ]) in
  let weight3 = U.reshape ~src:weight ~shape:(mk_shape b [ 1; 1; 8 ]) in
  let matrix3 = U.reshape ~src:matrix ~shape:(mk_shape b [ 1; 8; 8 ]) in
  let x3 = U.broadcast_to ~src:x3 ~shape:(mk_shape b [ 2; 8; 8 ]) in
  let norm3 = U.broadcast_to ~src:norm3 ~shape:(mk_shape b [ 2; 8; 8 ]) in
  let weight3 = U.broadcast_to ~src:weight3 ~shape:(mk_shape b [ 2; 8; 8 ]) in
  let matrix3 = U.broadcast_to ~src:matrix3 ~shape:(mk_shape b [ 2; 8; 8 ]) in
  let lhs = U.alu_binary ~op:Ops.Mul ~lhs:x3 ~rhs:norm3 in
  let lhs = U.alu_binary ~op:Ops.Mul ~lhs ~rhs:weight3 in
  let lhs = U.alu_binary ~op:Ops.Mul ~lhs ~rhs:matrix3 in
  let red = U.reduce_axis ~src:lhs ~op:Ops.Add ~axes:[ 2 ] in
  let result = U.reshape ~src:red ~shape:(mk_shape b [ 2; 8 ]) in
  wrap_sink b [ result ]

let build_llama_vector_scale b =
  let x = mk_param b ~slot:0 [ 2; 8 ] in
  let scale = mk_param b ~slot:1 [ 2 ] in
  let weight = mk_param b ~slot:2 [ 8 ] in
  let scale2 = U.reshape ~src:scale ~shape:(mk_shape b [ 2; 1 ]) in
  let weight2 = U.reshape ~src:weight ~shape:(mk_shape b [ 1; 8 ]) in
  let scale2 = U.broadcast_to ~src:scale2 ~shape:(mk_shape b [ 2; 8 ]) in
  let weight2 = U.broadcast_to ~src:weight2 ~shape:(mk_shape b [ 2; 8 ]) in
  let value = U.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:scale2 in
  let value = U.alu_binary ~op:Ops.Mul ~lhs:value ~rhs:weight2 in
  wrap_sink b [ value ]

let build_llama_output_projection b =
  let x = mk_param b ~slot:0 [ 2; 8 ] in
  let weight = mk_param b ~slot:1 [ 32; 8 ] in
  let x3 = U.reshape ~src:x ~shape:(mk_shape b [ 2; 1; 8 ]) in
  let weight3 = U.reshape ~src:weight ~shape:(mk_shape b [ 1; 32; 8 ]) in
  let x3 = U.broadcast_to ~src:x3 ~shape:(mk_shape b [ 2; 32; 8 ]) in
  let weight3 = U.broadcast_to ~src:weight3 ~shape:(mk_shape b [ 2; 32; 8 ]) in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:x3 ~rhs:weight3 in
  let red = U.reduce_axis ~src:mul ~op:Ops.Add ~axes:[ 2 ] in
  let result = U.reshape ~src:red ~shape:(mk_shape b [ 2; 32 ]) in
  wrap_sink b [ result ]

let f32 x = U.const (C.float D.float32 x)
let wi x = U.const (C.int D.weakint x)

let linear b ~x ~weight ~out_dim ~in_dim =
  let x3 = U.reshape ~src:x ~shape:(mk_shape b [ 2; 1; in_dim ]) in
  let w3 = U.reshape ~src:weight ~shape:(mk_shape b [ 1; out_dim; in_dim ]) in
  let x3 = U.broadcast_to ~src:x3 ~shape:(mk_shape b [ 2; out_dim; in_dim ]) in
  let w3 = U.broadcast_to ~src:w3 ~shape:(mk_shape b [ 2; out_dim; in_dim ]) in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:x3 ~rhs:w3 in
  let red = U.reduce_axis ~src:mul ~op:Ops.Add ~axes:[ 2 ] in
  U.reshape ~src:red ~shape:(mk_shape b [ 2; out_dim ])

let rms_norm_from_inv b x inv weight =
  let inv = U.reshape ~src:inv ~shape:(mk_shape b [ 2; 1 ]) in
  let inv = U.broadcast_to ~src:inv ~shape:(mk_shape b [ 2; 8 ]) in
  let weight = U.reshape ~src:weight ~shape:(mk_shape b [ 1; 8 ]) in
  let weight = U.broadcast_to ~src:weight ~shape:(mk_shape b [ 2; 8 ]) in
  U.alu_binary ~op:Ops.Mul
    ~lhs:(U.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:inv)
    ~rhs:weight

let silu b x =
  let scaled =
    U.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:(f32 (-1.4426950408889634))
  in
  let exp = U.alu_unary ~op:Ops.Exp2 ~src:scaled in
  let denom = U.alu_binary ~op:Ops.Add ~lhs:(f32 1.0) ~rhs:exp in
  U.alu_binary ~op:Ops.Mul ~lhs:x
    ~rhs:(U.alu_unary ~op:Ops.Reciprocal ~src:denom)

let build_llama_norm_linear ~out_dim b =
  let x = mk_param b ~slot:0 [ 2; 8 ] in
  let inv = mk_param b ~slot:1 [ 2 ] in
  let norm = mk_param b ~slot:2 [ 8 ] in
  let weight = mk_param b ~slot:3 [ out_dim; 8 ] in
  rms_norm_from_inv b x inv norm
  |> fun x -> linear b ~x ~weight ~out_dim ~in_dim:8
  |> fun u -> wrap_sink b [ u ]

let build_llama_attention_scores ?(kernel_name = "") b =
  let out = mk_ptr_param b ~slot:0 8 in
  let q = mk_ptr_param b ~slot:1 16 in
  let freqs = mk_ptr_param b ~slot:2 64 in
  let k = mk_ptr_param b ~slot:3 8 in
  let r = U.range ~size:(wi 2) ~axis:1 ~kind:Axis_type.Loop () in
  let open U.O in
  let load ptr idx = U.index ~ptr ~idxs:[ idx ] () in
  let rope4 ptr base freq_base =
    let x0 = load ptr base in
    let x1 = load ptr (base + wi 1) in
    let x2 = load ptr (base + wi 2) in
    let x3 = load ptr (base + wi 3) in
    let f0 = load freqs freq_base in
    let f1 = load freqs (freq_base + wi 1) in
    let f2 = load freqs (freq_base + wi 2) in
    let f3 = load freqs (freq_base + wi 3) in
    let ro0 = (x0 * f0) - (x1 * f1) in
    let co0 = (x0 * f1) + (x1 * f0) in
    let ro1 = (x2 * f2) - (x3 * f3) in
    let co1 = (x2 * f3) + (x3 * f2) in
    (ro0, co0, ro1, co1)
  in
  let k00, k01, k02, k03 = rope4 k (wi 0) (wi 0) in
  let k10, k11, k12, k13 = rope4 k (wi 4) (wi 4) in
  let scale v = v * f32 0.5 in
  let q_scores base freq_base =
    let x0 = load q base in
    let x1 = load q (base + wi 1) in
    let x2 = load q (base + wi 2) in
    let x3 = load q (base + wi 3) in
    let f0 = load freqs freq_base in
    let f1 = load freqs (freq_base + wi 1) in
    let f2 = load freqs (freq_base + wi 2) in
    let f3 = load freqs (freq_base + wi 3) in
    let co0 = (x0 * f1) + (x1 * f0) in
    let ro0 = (x0 * f0) - (x1 * f1) in
    let ro1 = (x2 * f2) - (x3 * f3) in
    let first0 = (ro0 * k00) + (ro1 * k02) + (co0 * k01) in
    let first1 = (ro0 * k10) + (ro1 * k12) + (co0 * k11) in
    let co1 = (x2 * f3) + (x3 * f2) in
    (scale (first0 + (co1 * k03)), scale (first1 + (co1 * k13)))
  in
  let q0_base = r * wi 4 in
  let q1_base = q0_base + wi 8 in
  let score10, score11 = q_scores q1_base (wi 4) in
  let score00, score01_base = q_scores q0_base (wi 0) in
  let score01 =
    score01_base + f32 neg_infinity
  in
  let base = r * wi 4 in
  let dst = U.index ~ptr:out ~idxs:[ base ] () in
  let value = U.stack [ score00; score01; score10; score11 ] in
  U.end_ ~value:(U.store ~dst ~value ()) ~ranges:[ r ]
  |> scheduled_kernel ~name:kernel_name [ out; q; freqs; k ]

let build_llama_attention_max b =
  let score = mk_param b ~slot:0 [ 4; 2 ] in
  U.reduce_axis ~src:score ~op:Ops.Max ~axes:[ 1 ]
  |> fun u -> wrap_sink b [ u ]

let softmax_exp2 b ~score ~maxv =
  let maxv = U.reshape ~src:maxv ~shape:(mk_shape b [ 4; 1 ]) in
  let maxv = U.broadcast_to ~src:maxv ~shape:(mk_shape b [ 4; 2 ]) in
  let diff = U.alu_binary ~op:Ops.Sub ~lhs:score ~rhs:maxv in
  U.alu_binary ~op:Ops.Mul ~lhs:diff ~rhs:(f32 1.4426950408889634)
  |> fun u -> U.alu_unary ~op:Ops.Exp2 ~src:u

let build_llama_attention_inv_sum b =
  let score = mk_param b ~slot:0 [ 4; 2 ] in
  let maxv = mk_param b ~slot:1 [ 4 ] in
  softmax_exp2 b ~score ~maxv
  |> fun exp -> U.reduce_axis ~src:exp ~op:Ops.Add ~axes:[ 1 ]
  |> fun sum -> U.alu_unary ~op:Ops.Reciprocal ~src:sum
  |> fun u -> wrap_sink b [ u ]

let build_llama_attention_context b =
  let out = mk_ptr_param b ~slot:0 16 in
  let score = mk_ptr_param b ~slot:1 8 in
  let maxv = mk_ptr_param b ~slot:2 4 in
  let inv = mk_ptr_param b ~slot:3 4 in
  let v = mk_ptr_param b ~slot:4 8 in
  let r1 = U.range ~size:(wi 4) ~axis:1 ~kind:Axis_type.Loop () in
  let r2 = U.range ~size:(wi 2) ~axis:2 ~kind:Axis_type.Reduce () in
  let r3 = U.range ~size:(wi 4) ~axis:3 ~kind:Axis_type.Loop () in
  let open U.O in
  let score_v = U.index ~ptr:score ~idxs:[ (r1 * wi 2) + r2 ] () in
  let max_v = U.index ~ptr:maxv ~idxs:[ r1 ] () in
  let inv_v = U.index ~ptr:inv ~idxs:[ r1 ] () in
  let scaled = (score_v - max_v) * f32 1.4426950408889634 in
  let exp = U.alu_unary ~op:Ops.Exp2 ~src:scaled in
  let value =
    U.alu_binary ~op:Ops.Mul
      ~lhs:(U.alu_binary ~op:Ops.Mul ~lhs:exp ~rhs:inv_v)
      ~rhs:(U.index ~ptr:v ~idxs:[ (r2 * wi 4) + r3 ] ())
  in
  let value =
    U.reduce ~src:value ~ranges:[ r2 ] ~op:Ops.Add ~dtype:D.float32
  in
  let dst = U.index ~ptr:out ~idxs:[ (r1 * wi 4) + r3 ] () in
  U.end_ ~value:(U.store ~dst ~value ()) ~ranges:[ r1; r2; r3 ]
  |> scheduled_kernel [ out; score; maxv; inv; v ]

let build_llama_attention_output b =
  let out = mk_ptr_param b ~slot:0 16 in
  let residual = mk_ptr_param b ~slot:1 16 in
  let ctx = mk_ptr_param b ~slot:2 16 in
  let weight = mk_ptr_param b ~slot:3 64 in
  let rred =
    U.range ~size:(wi 2) ~axis:0 ~sub:[ 0 ] ~kind:Axis_type.Reduce ()
  in
  let rseq = U.range ~size:(wi 2) ~axis:1 ~kind:Axis_type.Loop () in
  let rout = U.range ~size:(wi 8) ~axis:2 ~kind:Axis_type.Loop () in
  let rdim =
    U.range ~size:(wi 4) ~axis:0 ~sub:[ 0 ] ~kind:Axis_type.Reduce ()
  in
  let open U.O in
  let load ptr idx = U.index ~ptr ~idxs:[ idx ] () in
  let ctx_idx = (rred * wi 8) + (rseq * wi 4) + rdim in
  let weight_idx = (rred * wi 4) + (rout * wi 8) + rdim in
  let value = load ctx ctx_idx * load weight weight_idx in
  let value =
    U.reduce ~src:value ~ranges:[ rred; rdim ] ~op:Ops.Add
      ~dtype:D.float32
  in
  let idx = (rseq * wi 8) + rout in
  let value = load residual idx + value in
  let dst = U.index ~ptr:out ~idxs:[ idx ] () in
  U.end_ ~value:(U.store ~dst ~value ()) ~ranges:[ rseq; rout ]
  |> scheduled_kernel [ out; residual; ctx; weight ]

let build_llama_ffn_hidden b =
  let x = mk_param b ~slot:0 [ 2; 8 ] in
  let inv = mk_param b ~slot:1 [ 2 ] in
  let norm = mk_param b ~slot:2 [ 8 ] in
  let w1 = mk_param b ~slot:3 [ 16; 8 ] in
  let w3 = mk_param b ~slot:4 [ 16; 8 ] in
  let norm_linear weight =
    let x = U.reshape ~src:x ~shape:(mk_shape b [ 2; 1; 8 ]) in
    let inv = U.reshape ~src:inv ~shape:(mk_shape b [ 2; 1; 1 ]) in
    let norm = U.reshape ~src:norm ~shape:(mk_shape b [ 1; 1; 8 ]) in
    let weight = U.reshape ~src:weight ~shape:(mk_shape b [ 1; 16; 8 ]) in
    let x = U.broadcast_to ~src:x ~shape:(mk_shape b [ 2; 16; 8 ]) in
    let inv = U.broadcast_to ~src:inv ~shape:(mk_shape b [ 2; 16; 8 ]) in
    let norm = U.broadcast_to ~src:norm ~shape:(mk_shape b [ 2; 16; 8 ]) in
    let weight = U.broadcast_to ~src:weight ~shape:(mk_shape b [ 2; 16; 8 ]) in
    let mul = U.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:inv in
    let mul = U.alu_binary ~op:Ops.Mul ~lhs:mul ~rhs:norm in
    let mul = U.alu_binary ~op:Ops.Mul ~lhs:mul ~rhs:weight in
    let red = U.reduce_axis ~src:mul ~op:Ops.Add ~axes:[ 2 ] in
    U.reshape ~src:red ~shape:(mk_shape b [ 2; 16 ])
  in
  let gate = norm_linear w1 |> silu b in
  let up = norm_linear w3 in
  U.alu_binary ~op:Ops.Mul ~lhs:gate ~rhs:up |> fun u -> wrap_sink b [ u ]

let build_llama_ffn_output b =
  let residual = mk_param b ~slot:0 [ 2; 8 ] in
  let hidden = mk_param b ~slot:1 [ 2; 16 ] in
  let weight = mk_param b ~slot:2 [ 8; 16 ] in
  let ff = linear b ~x:hidden ~weight ~out_dim:8 ~in_dim:16 in
  wrap_sink b [ U.alu_binary ~op:Ops.Add ~lhs:residual ~rhs:ff ]

let llama_attention_scores_name renderer =
  match Renderer.name renderer with
  | "clang" -> "r_2_2_2_2_2"
  | "cuda" -> "r_2_2_2_2_2n1"
  | "metal" -> "r_2_2_2_2_2n2"
  | "opencl" -> "r_2_2_2_2_2n3"
  | _ -> "r_2_2_2_2_2"

let llama_forward_from_embedding_source renderer =
  [
    ("rmsnorm", build_llama_rmsnorm);
    ("norm_linear_8", build_llama_norm_linear ~out_dim:8);
    ("norm_linear_4_q", build_llama_norm_linear ~out_dim:4);
    ("norm_linear_4_k", build_llama_norm_linear ~out_dim:4);
    ( "attention_scores",
      build_llama_attention_scores
        ~kernel_name:(llama_attention_scores_name renderer) );
    ("attention_max", build_llama_attention_max);
    ("attention_inv_sum", build_llama_attention_inv_sum);
    ("attention_context", build_llama_attention_context);
    ("attention_output", build_llama_attention_output);
    ("ffn_hidden", build_llama_ffn_hidden);
    ("ffn_output", build_llama_ffn_output);
    ("vector_scale", build_llama_vector_scale);
    ("output_projection", build_llama_output_projection);
  ]
  |> (fun steps ->
       match Sys.getenv_opt "LLAMA_STEP" with
       | None -> steps
       | Some want ->
           List.filter (fun (name, _) -> String.equal name want) steps)
  |> List.map (fun (_, build) -> tensor_to_source renderer build)
  |> String.concat "\n---\n"

(* Test case type *)

type test_case = {
  name : string;
  build : unit -> U.t;
  backends : (string * Renderer.t) list;
}

let all_renderers =
  [
    ("clang", Cstyle.clang_no_abi Gpu_target.X86_64);
    ("cuda", Cstyle.cuda Gpu_target.SM80);
    ("metal", Cstyle.metal (Gpu_target.Apple 7));
    ("opencl", Cstyle.opencl "");
  ]

let test_cases =
  [
    (* Tier 1: Core fusion *)
    { name = "elementwise_add"; build = build_elementwise_add;
      backends = all_renderers };
    { name = "elementwise_3way"; build = build_elementwise_3way;
      backends = all_renderers };
    { name = "mulacc"; build = build_mulacc;
      backends = all_renderers };
    { name = "binop_reshape"; build = build_binop_reshape;
      backends = all_renderers };
    { name = "binop_permute"; build = build_binop_permute;
      backends = all_renderers };
    { name = "diamond"; build = build_diamond;
      backends = all_renderers };
    { name = "reduce_unary"; build = build_reduce_unary;
      backends = all_renderers };
    { name = "reduce_reshape_binop"; build = build_reduce_reshape_binop;
      backends = all_renderers };
    (* Tier 2: Movement ops *)
    { name = "reduce_permute_binop"; build = build_reduce_permute_binop;
      backends = all_renderers };
    { name = "permute_through_reshape"; build = build_permute_through_reshape;
      backends = all_renderers };
    { name = "expand_permute"; build = build_expand_permute;
      backends = all_renderers };
    { name = "shrink_fuse"; build = build_shrink_fuse;
      backends = all_renderers };
    (* Tier 3: Multi-reduce / multi-kernel *)
    { name = "multistage_reduce"; build = build_multistage_reduce;
      backends = all_renderers };
    { name = "two_sum"; build = build_two_sum;
      backends = all_renderers };
    { name = "reduce_shrink"; build = build_reduce_shrink;
      backends = all_renderers };
    (* Tier 4: Edge cases *)
    { name = "contiguous_add"; build = build_contiguous_add;
      backends = all_renderers };
    { name = "reshape_chain"; build = build_reshape_chain;
      backends = all_renderers };
    (* Tier 5: LLaMA model-derived rangeify/codegen kernels *)
    { name = "llama_rmsnorm"; build = build_llama_rmsnorm;
      backends = all_renderers };
    { name = "llama_ffn_gate"; build = build_llama_ffn_gate;
      backends = all_renderers };
    { name = "llama_vector_scale"; build = build_llama_vector_scale;
      backends = all_renderers };
    { name = "llama_output_projection"; build = build_llama_output_projection;
      backends = all_renderers };
    { name = "llama_forward_from_embedding";
      build = build_llama_rmsnorm; backends = all_renderers };
  ]

(* Main *)

let () =
  Printexc.record_backtrace true;
  let dir = Sys.argv.(1) in
  let test_cases =
    match Sys.getenv_opt "ONLY" with
    | None -> test_cases
    | Some only ->
        List.filter (fun { name; _ } -> String.equal name only) test_cases
  in
  List.iter
    (fun { name; build; backends } ->
      List.iter
        (fun (backend_name, renderer) ->
          let snap = Printf.sprintf "%s_%s" backend_name name in
          let out =
            match
              if String.equal name "llama_forward_from_embedding" then
                llama_forward_from_embedding_source renderer
              else tensor_to_source renderer build
            with
            | out -> out
            | exception exn ->
                Printf.eprintf "FAIL %s: %s\n%!" snap
                  (Printexc.to_string exn);
                Printf.eprintf "%s%!" (Printexc.get_backtrace ());
                Printf.sprintf "ERROR: %s" (Printexc.to_string exn)
          in
          let filename = Filename.concat dir (snap ^ ".actual") in
          let oc = open_out filename in
          output_string oc out;
          output_char oc '\n';
          close_out oc)
        backends)
    test_cases
