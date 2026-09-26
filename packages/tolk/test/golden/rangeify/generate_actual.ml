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
  let ids = List.map U.const_int dims in
  match ids with
  | [ d ] -> d
  | ds ->
      U.stack ds

(* Emit a PARAM with a known shape and CPU device. *)
let mk_param ?(dtype = D.float32) b ~slot (shape : int list) : U.t =
  let shape_id = if shape = [] then None else Some (mk_shape b shape) in
  let dev = U.Single "CPU" in
  U.param ~slot ~dtype ?shape:shape_id ~device:dev ()

(* Wrap source(s) in CONTIGUOUS -> SINK. *)
let wrap_sink b (srcs : U.t list) : U.t =
  let contigs =
    List.map (fun src -> U.contiguous ~src ()) srcs
  in
  U.sink contigs

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

let kernels_to_source renderer kernels =
  List.map
    (fun k ->
      let processed = Codegen.full_rewrite_to_sink ~optimize:true renderer k in
      let name = name_of_sink processed in
      let prog = Linearizer.linearize processed in
      String.trim (Renderer.render renderer ~name prog))
    kernels
  |> String.concat "\n---\n"

(* Run the full pipeline: Tensor.t -> rendered source string. *)
let tensor_to_source renderer (build_fn : unit -> U.t) : string =
  let kernel_graph = Rangeify.get_kernel_graph (build_fn ()) in
  kernels_to_source renderer (extract_kernels kernel_graph)

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
  (* The kernel boundary sits between the root and its reciprocal: the
     buffer holds [sqrt(mean + eps)] and each consumer divides by it. *)
  let sqrt = U.alu_unary ~op:Ops.Sqrt ~src:eps in
  let result = U.reshape ~src:sqrt ~shape:(mk_shape b [ 2 ]) in
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
  let norm3 = U.alu_unary ~op:Ops.Reciprocal ~src:norm3 in
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
  let scale2 = U.alu_unary ~op:Ops.Reciprocal ~src:scale2 in
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

let llama_forward_from_embedding_source renderer =
  let module T = Tolk_frontend.Tensor in
  let input _ shape =
    Tolk_frontend.Creation.empty ~dtype:D.float32 ~device:(U.Single "CPU") shape
  in
  let logits, h, parameters = Llama_fixture.build input in
  let sink = U.sink (List.map T.uop (logits :: h :: parameters)) in
  let sink, _ = Bufferize.run sink in
  let call = Callify.transform_to_call sink in
  let linear, _ =
    Schedule.create_linear_with_vars ~get_kernel_graph:Rangeify.get_kernel_graph
      call
  in
  let seen = U.Tbl.create 16 in
  let kernels =
    List.filter_map
      (fun call ->
        match U.as_call call with
        | Some { body; _ }
          when U.op body = Ops.Sink && not (U.Tbl.mem seen body) ->
            U.Tbl.add seen body ();
            Some body
        | _ -> None)
      (U.children linear)
  in
  kernels_to_source renderer kernels

(* Test case type *)

type test_case = {
  name : string;
  build : unit -> U.t;
  backends : (string * Renderer.t) list;
}

let all_renderers =
  [
    ("clang", Cstyle.clang_no_abi);
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
