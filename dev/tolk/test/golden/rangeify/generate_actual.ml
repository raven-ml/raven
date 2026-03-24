(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generates .actual files for rangeify pipeline golden tests. Each file
   contains tolk's rendered output for a specific backend + test case after
   running the full pipeline: Tensor.t -> Rangeify.get_kernel_graph ->
   Kernel.t -> Pipeline.full_rewrite_to_sink -> Linearizer.linearize ->
   Renderer.render. Dune diff rules compare .actual against .expected. *)

open Tolk
open Tolk_ir
module T = Tensor
module K = Kernel
module C = Const
module D = Dtype

(* Helpers *)

(* Emit a shape-encoding node from a concrete int list. *)
let mk_shape b (dims : int list) : T.id =
  let ids = List.map (fun s -> T.const b (C.int D.index s)) dims in
  match ids with
  | [ d ] -> d
  | ds ->
      T.emit b
        (Vectorize { srcs = ds; dtype = D.vec D.index (List.length ds) })

(* Emit a PARAM with a known shape and CPU device. *)
let mk_param b ~slot (shape : int list) : T.id =
  let shape_id = if shape = [] then None else Some (mk_shape b shape) in
  let dev = T.device b (Single "CPU") in
  T.param b ~slot ~dtype:D.float32 ?shape:shape_id ~device:dev ()

(* Wrap source(s) in CONTIGUOUS -> SINK. *)
let wrap_sink b (srcs : T.id list) : T.id =
  let contigs =
    List.map (fun src -> T.contiguous b ~src ()) srcs
  in
  T.sink b contigs

(* Extract Kernel.t ASTs from CALL nodes in topological (id) order. *)
let extract_kernels (program : T.t) : K.t list =
  let kernels = ref [] in
  for i = 0 to T.length program - 1 do
    match T.view program i with
    | Call { callee = Ast k; _ } -> kernels := k :: !kernels
    | _ -> ()
  done;
  List.rev !kernels

(* Extract kernel name from a pipeline-processed Sink. *)
let name_of_sink sink =
  match K.view sink with
  | K.Sink { kernel_info = Some ki; _ } -> ki.name
  | _ -> "kernel"

(* Run the full pipeline: Tensor.t -> rendered source string. *)
let tensor_to_source renderer (build_fn : T.builder -> T.id) : string =
  let b = T.create () in
  let _sink = build_fn b in
  let program = T.finish b in
  let kernel_graph = Rangeify.get_kernel_graph program in
  let kernels = extract_kernels kernel_graph in
  let sources =
    List.map
      (fun k ->
        let processed =
          Pipeline.full_rewrite_to_sink ~optimize:true renderer k
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
  let add = T.binary b ~op:`Add ~lhs:a ~rhs:bp in
  wrap_sink b [ add ]

let build_elementwise_3way b =
  let a = mk_param b ~slot:0 [ 256 ] in
  let bp = mk_param b ~slot:1 [ 256 ] in
  let c = mk_param b ~slot:2 [ 256 ] in
  let ab = T.binary b ~op:`Add ~lhs:a ~rhs:bp in
  let abc = T.binary b ~op:`Add ~lhs:ab ~rhs:c in
  wrap_sink b [ abc ]

let build_mulacc b =
  let a = mk_param b ~slot:0 [ 256 ] in
  let bp = mk_param b ~slot:1 [ 256 ] in
  let mul = T.binary b ~op:`Mul ~lhs:a ~rhs:bp in
  let red =
    T.reduce_axis b ~src:mul ~op:`Add ~axes:[ 0 ] in
  wrap_sink b [ red ]

let build_binop_reshape b =
  let a = mk_param b ~slot:0 [ 10 ] in
  let bp = mk_param b ~slot:1 [ 10 ] in
  let c = mk_param b ~slot:2 [ 5; 2 ] in
  let add = T.binary b ~op:`Add ~lhs:a ~rhs:bp in
  let reshaped = T.reshape b ~src:add ~shape:(mk_shape b [ 5; 2 ]) in
  let result = T.binary b ~op:`Add ~lhs:reshaped ~rhs:c in
  wrap_sink b [ result ]

let build_binop_permute b =
  let a = mk_param b ~slot:0 [ 2; 5 ] in
  let bp = mk_param b ~slot:1 [ 2; 5 ] in
  let c = mk_param b ~slot:2 [ 5; 2 ] in
  let add = T.binary b ~op:`Add ~lhs:a ~rhs:bp in
  let permed = T.permute b ~src:add ~order:[ 1; 0 ] in
  let result = T.binary b ~op:`Add ~lhs:permed ~rhs:c in
  wrap_sink b [ result ]

let build_diamond b =
  let a = mk_param b ~slot:0 [ 10 ] in
  let bp = mk_param b ~slot:1 [ 10 ] in
  let c = mk_param b ~slot:2 [ 10 ] in
  let d = mk_param b ~slot:3 [ 10 ] in
  let ab = T.binary b ~op:`Add ~lhs:a ~rhs:bp in
  let abc = T.binary b ~op:`Add ~lhs:ab ~rhs:c in
  let abd = T.binary b ~op:`Add ~lhs:ab ~rhs:d in
  let result = T.binary b ~op:`Add ~lhs:abc ~rhs:abd in
  wrap_sink b [ result ]

let build_reduce_unary b =
  let a = mk_param b ~slot:0 [ 16 ] in
  let red =
    T.reduce_axis b ~src:a ~op:`Add ~axes:[ 0 ] in
  let sq = T.unary b ~op:`Sqrt ~src:red in
  let neg = T.unary b ~op:`Neg ~src:sq in
  wrap_sink b [ neg ]

let build_reduce_reshape_binop b =
  let a = mk_param b ~slot:0 [ 10; 10 ] in
  let bp = mk_param b ~slot:1 [ 10 ] in
  let red =
    T.reduce_axis b ~src:a ~op:`Add ~axes:[ 0 ] in
  let reshaped = T.reshape b ~src:red ~shape:(mk_shape b [ 10 ]) in
  let result = T.binary b ~op:`Add ~lhs:reshaped ~rhs:bp in
  wrap_sink b [ result ]

let build_reduce_permute_binop b =
  let a = mk_param b ~slot:0 [ 10; 10; 10 ] in
  let bp = mk_param b ~slot:1 [ 10; 10; 1 ] in
  let red =
    T.reduce_axis b ~src:a ~op:`Add ~axes:[ 0 ] in
  let permed = T.permute b ~src:red ~order:[ 2; 1; 0 ] in
  let result = T.binary b ~op:`Add ~lhs:permed ~rhs:bp in
  wrap_sink b [ result ]

let build_permute_through_reshape b =
  let a = mk_param b ~slot:0 [ 16; 16 ] in
  let bp = mk_param b ~slot:1 [ 16; 16 ] in
  let add = T.binary b ~op:`Add ~lhs:a ~rhs:bp in
  let reshaped =
    T.reshape b ~src:add ~shape:(mk_shape b [ 4; 4; 4; 4 ])
  in
  let permed = T.permute b ~src:reshaped ~order:[ 2; 3; 0; 1 ] in
  wrap_sink b [ permed ]

let build_expand_permute b =
  let a = mk_param b ~slot:0 [ 10; 10; 1 ] in
  let bp = mk_param b ~slot:1 [ 10; 10; 1 ] in
  let ab = T.binary b ~op:`Add ~lhs:a ~rhs:bp in
  let expanded =
    T.expand b ~src:ab ~shape:(mk_shape b [ 10; 10; 10 ])
  in
  let permed = T.permute b ~src:ab ~order:[ 2; 1; 0 ] in
  let permed_expanded =
    T.expand b ~src:permed ~shape:(mk_shape b [ 10; 10; 10 ])
  in
  let result = T.binary b ~op:`Add ~lhs:expanded ~rhs:permed_expanded in
  wrap_sink b [ result ]

let build_shrink_fuse b =
  let a = mk_param b ~slot:0 [ 8192; 16 ] in
  let bp = mk_param b ~slot:1 [ 8192; 16 ] in
  let d = mk_param b ~slot:2 [ 1; 16 ] in
  let mul = T.binary b ~op:`Mul ~lhs:a ~rhs:bp in
  let before = mk_shape b [ 0; 0 ] in
  let after = mk_shape b [ 1; 16 ] in
  let shrunk = T.shrink b ~src:mul ~before ~after in
  let result = T.binary b ~op:`Mul ~lhs:shrunk ~rhs:d in
  wrap_sink b [ result ]

let build_multistage_reduce b =
  let a = mk_param b ~slot:0 [ 32; 32; 32 ] in
  let red1 =
    T.reduce_axis b ~src:a ~op:`Add ~axes:[ 2 ] in
  (* relu: max(red1, 0) — zero must be broadcast to [32,32,1] *)
  let zero = T.const b (C.float D.float32 0.0) in
  let zero_reshaped =
    T.reshape b ~src:zero ~shape:(mk_shape b [ 1; 1; 1 ])
  in
  let zero_expanded =
    T.expand b ~src:zero_reshaped ~shape:(mk_shape b [ 32; 32; 1 ])
  in
  let relu = T.binary b ~op:`Max ~lhs:red1 ~rhs:zero_expanded in
  let reshaped =
    T.reshape b ~src:relu ~shape:(mk_shape b [ 32; 32 ])
  in
  let red2 =
    T.reduce_axis b ~src:reshaped ~op:`Add ~axes:[ 1 ] in
  wrap_sink b [ red2 ]

let build_two_sum b =
  let a = mk_param b ~slot:0 [ 64; 64 ] in
  let red0 =
    T.reduce_axis b ~src:a ~op:`Add ~axes:[ 0 ] in
  let red1 =
    T.reduce_axis b ~src:a ~op:`Add ~axes:[ 1 ] in
  let reshaped0 = T.reshape b ~src:red0 ~shape:(mk_shape b [ 64 ]) in
  let reshaped1 = T.reshape b ~src:red1 ~shape:(mk_shape b [ 64 ]) in
  let result = T.binary b ~op:`Add ~lhs:reshaped0 ~rhs:reshaped1 in
  wrap_sink b [ result ]

let build_reduce_shrink b =
  let a = mk_param b ~slot:0 [ 32; 32 ] in
  let bp = mk_param b ~slot:1 [ 16 ] in
  let red =
    T.reduce_axis b ~src:a ~op:`Add ~axes:[ 1 ] in
  let reshaped = T.reshape b ~src:red ~shape:(mk_shape b [ 32 ]) in
  let before = mk_shape b [ 0 ] in
  let after = mk_shape b [ 16 ] in
  let shrunk = T.shrink b ~src:reshaped ~before ~after in
  let result = T.binary b ~op:`Add ~lhs:shrunk ~rhs:bp in
  wrap_sink b [ result ]

let build_contiguous_add b =
  let x = mk_param b ~slot:0 [ 32 ] in
  let y = mk_param b ~slot:1 [ 32 ] in
  let z = mk_param b ~slot:2 [ 32 ] in
  let add = T.binary b ~op:`Add ~lhs:x ~rhs:y in
  let contig = T.contiguous b ~src:add () in
  let result = T.binary b ~op:`Add ~lhs:contig ~rhs:z in
  wrap_sink b [ result ]

let build_reshape_chain b =
  let a = mk_param b ~slot:0 [ 4; 4 ] in
  let bp = mk_param b ~slot:1 [ 2; 8 ] in
  let r1 = T.reshape b ~src:a ~shape:(mk_shape b [ 16 ]) in
  let r2 = T.reshape b ~src:r1 ~shape:(mk_shape b [ 2; 8 ]) in
  let result = T.binary b ~op:`Add ~lhs:r2 ~rhs:bp in
  wrap_sink b [ result ]

(* Test case type *)

type test_case = {
  name : string;
  build : T.builder -> T.id;
  backends : (string * Renderer.t) list;
}

let all_renderers =
  [
    ("clang", Cstyle.clang_no_abi);
    ("cuda", Cstyle.cuda Gpu_target.SM80);
    ("metal", Cstyle.metal);
    ("opencl", Cstyle.opencl);
  ]

(* GPU renderers that don't need local memory (pm_add_buffers_local).
   Tests with REDUCE_AXIS on GPU require local bufferize (Step 10 DEFERRED
   in lowering.ml). Until pm_add_buffers_local is implemented, GPU reduce
   tests are excluded. Elementwise-only and multi-kernel elementwise tests
   work on all backends. *)
let gpu_renderers =
  List.filter (fun (name, _) -> name <> "clang") all_renderers

let test_cases =
  [
    (* Tier 1: Core fusion *)
    { name = "elementwise_add"; build = build_elementwise_add;
      backends = all_renderers };
    { name = "elementwise_3way"; build = build_elementwise_3way;
      backends = all_renderers };
    { name = "mulacc"; build = build_mulacc;
      (* GPU reduce needs pm_add_buffers_local (Step 10 DEFERRED) *)
      backends = all_renderers };
    { name = "binop_reshape"; build = build_binop_reshape;
      backends = all_renderers };
    { name = "binop_permute"; build = build_binop_permute;
      backends = all_renderers };
    { name = "diamond"; build = build_diamond;
      backends = all_renderers };
    { name = "reduce_unary"; build = build_reduce_unary;
      (* GPU reduce needs pm_add_buffers_local (Step 10 DEFERRED) *)
      backends = all_renderers };
    { name = "reduce_reshape_binop"; build = build_reduce_reshape_binop;
      (* GPU reduce needs pm_add_buffers_local (Step 10 DEFERRED) *)
      backends = all_renderers };
    (* Tier 2: Movement ops *)
    { name = "reduce_permute_binop"; build = build_reduce_permute_binop;
      (* GPU reduce needs pm_add_buffers_local (Step 10 DEFERRED) *)
      backends = all_renderers };
    { name = "permute_through_reshape"; build = build_permute_through_reshape;
      backends = all_renderers };
    { name = "expand_permute"; build = build_expand_permute;
      backends = all_renderers };
    { name = "shrink_fuse"; build = build_shrink_fuse;
      backends = all_renderers };
    (* Tier 3: Multi-reduce / multi-kernel *)
    { name = "multistage_reduce"; build = build_multistage_reduce;
      (* GPU reduce needs pm_add_buffers_local (Step 10 DEFERRED) *)
      backends = all_renderers };
    { name = "two_sum"; build = build_two_sum;
      backends = all_renderers };
    { name = "reduce_shrink"; build = build_reduce_shrink;
      (* GPU reduce needs pm_add_buffers_local (Step 10 DEFERRED) *)
      backends = all_renderers };
    (* Tier 4: Edge cases *)
    { name = "contiguous_add"; build = build_contiguous_add;
      backends = all_renderers };
    { name = "reshape_chain"; build = build_reshape_chain;
      backends = all_renderers };
  ]

(* Main *)

let () =
  let dir = Sys.argv.(1) in
  List.iter
    (fun { name; build; backends } ->
      List.iter
        (fun (backend_name, renderer) ->
          let snap = Printf.sprintf "%s_%s" backend_name name in
          let out =
            match tensor_to_source renderer build with
            | out -> out
            | exception exn ->
                Printf.eprintf "FAIL %s: %s\n%!" snap
                  (Printexc.to_string exn);
                Printf.sprintf "ERROR: %s" (Printexc.to_string exn)
          in
          let filename = Filename.concat dir (snap ^ ".actual") in
          let oc = open_out filename in
          output_string oc out;
          output_char oc '\n';
          close_out oc)
        backends)
    test_cases
