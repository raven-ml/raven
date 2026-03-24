(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* JIT compilation via effect handler.

   Ported from tinygrad/engine/jit.py (TinyJit). Operations do NOT execute
   during capture — the handler returns placeholder tensors, building a lazy
   graph. The graph is realized (rangeify → compile → execute) when the
   function returns. Replay re-executes the compiled schedule.

   State machine (matching tinygrad's cnt):
   - Warmup (cnt=0): execute eagerly via C backend (shape gathering)
   - Capture (cnt=1): intercept effects, build lazy graph, realize at end
   - Replay (cnt>=2): validate inputs, substitute buffers, execute *)

open Nx_effect
module Tensor = Tolk_ir.Tensor

(* --- Dtype mapping ------------------------------------------------------- *)

let tolk_dtype (type a b) (dt : (a, b) Nx.dtype) : Tolk_ir.Dtype.t =
  let open Tolk_ir.Dtype in
  match Nx_core.Dtype.to_string dt with
  | "float32" -> float32 | "float64" -> float64 | "float16" -> float16
  | "int32" -> int32 | "int64" -> int64 | "int8" -> int8
  | "int16" -> int16 | "uint8" -> uint8 | "uint16" -> uint16
  | "uint32" -> uint32 | "uint64" -> uint64 | "bool" -> bool
  | s -> failwith (Printf.sprintf "Jit: unsupported dtype %s" s)

(* --- Buffer transfer ----------------------------------------------------- *)

(* Nx tensor → device buffer *)
let nx_to_device_buffer (type a b) dev (t : (a, b) Nx_effect.t) : Tolk.Device.Buffer.t =
  let host = Nx_effect.to_host t in
  let shape = Nx_effect.view t |> Nx_core.View.shape in
  let num_elements = Int.max 1 (Array.fold_left ( * ) 1 shape) in
  let dt = tolk_dtype (Nx_effect.dtype t) in
  let buf = Tolk.Device.create_buffer ~size:num_elements ~dtype:dt dev in
  Tolk.Device.Buffer.ensure_allocated buf;
  let nbytes = num_elements * Tolk_ir.Dtype.itemsize dt in
  let src_bytes = Bytes.create nbytes in
  Nx_buffer.blit_to_bytes host src_bytes;
  Tolk.Device.Buffer.copyin buf src_bytes;
  buf

(* Device buffer → Nx tensor *)
let device_buffer_to_nx (type a b) (dt : (a, b) Nx.dtype) (shape : int array)
    (buf : Tolk.Device.Buffer.t) : (a, b) Nx_effect.t =
  let num_elements = Int.max 1 (Array.fold_left ( * ) 1 shape) in
  let tdt = tolk_dtype dt in
  let nbytes = num_elements * Tolk_ir.Dtype.itemsize tdt in
  let dst_bytes = Bytes.create nbytes in
  Tolk.Device.Buffer.copyout buf dst_bytes;
  let ctx = Nx_effect.create_context () in
  let nx_buf = Nx_effect.buffer ctx dt shape in
  let host = Nx_effect.to_host nx_buf in
  Nx_buffer.blit_from_bytes dst_bytes host;
  nx_buf

(* --- Identity-keyed hash table ------------------------------------------- *)

(* The default polymorphic [Hashtbl] uses structural equality for [Obj.t] keys.
   Two distinct [Nx_effect.t] values with the same dtype, shape, and data (e.g.
   a zero-filled placeholder and a zero-filled input) would collide. We need
   physical identity ([==]) to track tensor values through the effect handler. *)
module Phys_tbl = Hashtbl.Make (struct
  type t = Obj.t
  let equal = ( == )
  let hash x = Hashtbl.hash (Obj.obj x : int)
end)

(* --- Capture context ----------------------------------------------------- *)

type capture_ctx = {
  builder : Tensor.builder;
  tensor_to_id : Tensor.id Phys_tbl.t;
  (* slot → (original Nx tensor as Obj.t, tolk dtype, shape) *)
  slot_tensors : (int, Obj.t * Tolk_ir.Dtype.t * int array) Hashtbl.t;
  mutable param_count : int;
  device_id : Tensor.id;
}

let create_capture_ctx builder device_id =
  { builder; tensor_to_id = Phys_tbl.create 64;
    slot_tensors = Hashtbl.create 16;
    param_count = 0; device_id }

(* Placeholder: uninitialized Nx tensor with correct shape/dtype *)
let make_placeholder (type a b) (dt : (a, b) Nx.dtype) (shape : int array) : (a, b) Nx_effect.t =
  let ctx = Nx_effect.create_context () in
  Nx_effect.buffer ctx dt shape

let register ctx (t : _ Nx_effect.t) (id : Tensor.id) : unit =
  Phys_tbl.replace ctx.tensor_to_id (Obj.repr t) id

let lookup_or_param ctx (t : _ Nx_effect.t) : Tensor.id =
  let key = Obj.repr t in
  match Phys_tbl.find_opt ctx.tensor_to_id key with
  | Some id -> id
  | None ->
      let slot = ctx.param_count in
      ctx.param_count <- slot + 1;
      let dt = tolk_dtype (Nx_effect.dtype t) in
      let shape = Nx_effect.view t |> Nx_core.View.shape in
      let shape_id = Tensor.shape ctx.builder (Tolk_ir.Shape.of_dims (Array.to_list shape)) in
      let id = Tensor.param ctx.builder ~slot ~dtype:dt ~shape:shape_id
                 ~device:ctx.device_id () in
      Phys_tbl.replace ctx.tensor_to_id key id;
      Hashtbl.replace ctx.slot_tensors slot (key, dt, shape);
      id

(* --- Graph building ------------------------------------------------------ *)

let emit_binary ctx op (a : _ Nx_effect.t) (b : _ Nx_effect.t) ~dtype ~shape =
  let a_id = lookup_or_param ctx a in
  let b_id = lookup_or_param ctx b in
  let dt = tolk_dtype dtype in
  let out = make_placeholder dtype shape in
  let id = Tensor.emit ctx.builder (Binary { op; lhs = a_id; rhs = b_id; dtype = dt }) in
  register ctx out id; out

let emit_unary ctx op (src : _ Nx_effect.t) ~dtype ~shape =
  let src_id = lookup_or_param ctx src in
  let dt = tolk_dtype dtype in
  let out = make_placeholder dtype shape in
  let id = Tensor.emit ctx.builder (Unary { op; src = src_id; dtype = dt }) in
  register ctx out id; out

let emit_reduce ctx op ~axes (src : _ Nx_effect.t) ~dtype ~shape =
  let src_id = lookup_or_param ctx src in
  let dt = tolk_dtype dtype in
  let out = make_placeholder dtype shape in
  let id = Tensor.emit ctx.builder (Reduce_axis { src = src_id; op; axes; dtype = dt }) in
  register ctx out id; out

let infer_shape (t : _ Nx_effect.t) = Nx_effect.view t |> Nx_core.View.shape

let reduce_shape in_shape axes_list =
  let out = List.filteri (fun i _ -> not (List.mem i axes_list)) (Array.to_list in_shape) in
  if out = [] then [||] else Array.of_list out

(* --- Effect handler ------------------------------------------------------ *)

let make_capture_handler ctx =
  let open Effect.Deep in
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    match eff with
    | E_add { a; b } -> Some (fun k ->
        continue k (emit_binary ctx `Add a b ~dtype:(Nx_effect.dtype a) ~shape:(infer_shape a)))
    | E_sub { a; b } -> Some (fun k ->
        continue k (emit_binary ctx `Sub a b ~dtype:(Nx_effect.dtype a) ~shape:(infer_shape a)))
    | E_mul { a; b } -> Some (fun k ->
        continue k (emit_binary ctx `Mul a b ~dtype:(Nx_effect.dtype a) ~shape:(infer_shape a)))
    | E_max { a; b } -> Some (fun k ->
        continue k (emit_binary ctx `Max a b ~dtype:(Nx_effect.dtype a) ~shape:(infer_shape a)))
    | E_cmpeq { a; b } -> Some (fun k ->
        continue k (emit_binary ctx `Cmpeq a b ~dtype:Nx.bool ~shape:(infer_shape a)))
    | E_cmplt { a; b } -> Some (fun k ->
        continue k (emit_binary ctx `Cmplt a b ~dtype:Nx.bool ~shape:(infer_shape a)))
    | E_neg { t_in } -> Some (fun k ->
        continue k (emit_unary ctx `Neg t_in ~dtype:(Nx_effect.dtype t_in) ~shape:(infer_shape t_in)))
    | E_sqrt { t_in } -> Some (fun k ->
        continue k (emit_unary ctx `Sqrt t_in ~dtype:(Nx_effect.dtype t_in) ~shape:(infer_shape t_in)))
    | E_sin { t_in } -> Some (fun k ->
        continue k (emit_unary ctx `Sin t_in ~dtype:(Nx_effect.dtype t_in) ~shape:(infer_shape t_in)))
    | E_recip { t_in } -> Some (fun k ->
        continue k (emit_unary ctx `Recip t_in ~dtype:(Nx_effect.dtype t_in) ~shape:(infer_shape t_in)))
    | E_reduce_sum { t_in; axes; keepdims = _ } -> Some (fun k ->
        let axes_l = Array.to_list axes in
        continue k (emit_reduce ctx `Add ~axes:axes_l t_in
          ~dtype:(Nx_effect.dtype t_in) ~shape:(reduce_shape (infer_shape t_in) axes_l)))
    | E_reduce_max { t_in; axes; keepdims = _ } -> Some (fun k ->
        let axes_l = Array.to_list axes in
        continue k (emit_reduce ctx `Max ~axes:axes_l t_in
          ~dtype:(Nx_effect.dtype t_in) ~shape:(reduce_shape (infer_shape t_in) axes_l)))
    | E_reshape { t_in; new_shape } -> Some (fun k ->
        let src_id = lookup_or_param ctx t_in in
        let dt = tolk_dtype (Nx_effect.dtype t_in) in
        let out = make_placeholder (Nx_effect.dtype t_in) new_shape in
        let shape_id = Tensor.shape ctx.builder (Tolk_ir.Shape.of_dims (Array.to_list new_shape)) in
        let id = Tensor.emit ctx.builder (Reshape { src = src_id; shape = shape_id; dtype = dt }) in
        register ctx out id; continue k out)
    | E_permute { t_in; axes } -> Some (fun k ->
        let src_id = lookup_or_param ctx t_in in
        let dt = tolk_dtype (Nx_effect.dtype t_in) in
        let in_shape = infer_shape t_in in
        let out_shape = Array.map (fun ax -> in_shape.(ax)) axes in
        let out = make_placeholder (Nx_effect.dtype t_in) out_shape in
        let id = Tensor.emit ctx.builder (Permute { src = src_id; order = Array.to_list axes; dtype = dt }) in
        register ctx out id; continue k out)
    | E_expand { t_in; new_target_shape } -> Some (fun k ->
        let src_id = lookup_or_param ctx t_in in
        let dt = tolk_dtype (Nx_effect.dtype t_in) in
        let out = make_placeholder (Nx_effect.dtype t_in) new_target_shape in
        let shape_id = Tensor.shape ctx.builder (Tolk_ir.Shape.of_dims (Array.to_list new_target_shape)) in
        let id = Tensor.emit ctx.builder (Expand { src = src_id; shape = shape_id; dtype = dt }) in
        register ctx out id; continue k out)
    | E_cast { t_in; target_dtype } -> Some (fun k ->
        let src_id = lookup_or_param ctx t_in in
        let dt = tolk_dtype target_dtype in
        let out = make_placeholder target_dtype (infer_shape t_in) in
        let id = Tensor.emit ctx.builder (Cast { src = src_id; dtype = dt }) in
        register ctx out id; continue k out)
    | E_where { condition; if_true; if_false } -> Some (fun k ->
        let c_id = lookup_or_param ctx condition in
        let t_id = lookup_or_param ctx if_true in
        let f_id = lookup_or_param ctx if_false in
        let dt = tolk_dtype (Nx_effect.dtype if_true) in
        let out = make_placeholder (Nx_effect.dtype if_true) (infer_shape if_true) in
        let id = Tensor.emit ctx.builder
          (Ternary { op = `Where; a = c_id; b = t_id; c = f_id; dtype = dt }) in
        register ctx out id; continue k out)
    | E_const_scalar { context = _; value; dtype = dt } -> Some (fun k ->
        let out = make_placeholder dt [||] in
        let tdt = tolk_dtype dt in
        let cv =
          if Tolk_ir.Dtype.is_float tdt then Tolk_ir.Const.float tdt (Obj.magic value : float)
          else if Tolk_ir.Dtype.equal tdt Tolk_ir.Dtype.bool then Tolk_ir.Const.bool (Obj.magic value : bool)
          else Tolk_ir.Const.int tdt (Obj.magic value : int)
        in
        let id = Tensor.const ctx.builder cv in
        register ctx out id; continue k out)
    | _ -> None  (* Unhandled: will raise Effect.Unhandled *)
  in
  { retc = (fun x -> x); exnc = raise; effc }

(* --- Graph capture ------------------------------------------------------- *)

let capture_graph (type a b c d)
    ?(device_name = "CPU")
    (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t)
    : Tensor.t * capture_ctx * (c, d) Nx_effect.t =
  let builder = Tensor.create () in
  let device_id = Tensor.device builder (Single device_name) in
  let ctx = create_capture_ctx builder device_id in
  let handler = make_capture_handler ctx in
  let result = Effect.Deep.match_with f x handler in
  let result_id = lookup_or_param ctx result in
  let contiguous_id = Tensor.contiguous ctx.builder ~src:result_id () in
  ignore (Tensor.sink ctx.builder [contiguous_id]);
  let graph = Tensor.finish ctx.builder in
  (graph, ctx, result)

(* --- Compile + execute --------------------------------------------------- *)

type exec_item = {
  program : Tolk.Device.Program.t;
  arg_node_ids : Tensor.id array;
}

type input_info = {
  shape : int array;
  tolk_dtype : Tolk_ir.Dtype.t;
}

type captured_jit = {
  schedule : exec_item array;
  kernel_graph : Tensor.t;
  (* Buffers for allocated Buffer nodes (intermediates + output) *)
  alloc_bufs : (Tensor.id, Tolk.Device.Buffer.t) Hashtbl.t;
  (* Info for creating param device buffers on replay *)
  slot_tensors : (int, Obj.t * Tolk_ir.Dtype.t * int array) Hashtbl.t;
  (* Param tensor node id → slot *)
  param_node_slots : (Tensor.id, int) Hashtbl.t;
  expected_input_shape : int array;
  output_shape : int array;
  output_buf_id : Tensor.id;
  device : Tolk.Device.t;
}

(* Extract execution items from the kernel graph. Each CALL(Ast kernel) node
   becomes an exec_item with the compiled program and its arg node ids. *)
let extract_schedule (dev : Tolk.Device.t) (graph : Tensor.t) : exec_item list =
  let ren = Tolk.Device.renderer dev in
  let items = ref [] in
  for id = 0 to Tensor.length graph - 1 do
    match Tensor.view graph id with
    | Call { callee = Ast kernel; args; _ } ->
        let program = Tolk.Lowering.compile dev ren kernel in
        items := { program; arg_node_ids = Array.of_list args } :: !items
    | _ -> ()
  done;
  List.rev !items

(* Allocate device buffers for all Buffer nodes in the kernel graph. *)
let allocate_buffers (dev : Tolk.Device.t) (graph : Tensor.t)
    : (Tensor.id, Tolk.Device.Buffer.t) Hashtbl.t =
  let tbl = Hashtbl.create 16 in
  for id = 0 to Tensor.length graph - 1 do
    match Tensor.view graph id with
    | Buffer { size; dtype; _ } ->
        let buf = Tolk.Device.create_buffer ~size ~dtype dev in
        Tolk.Device.Buffer.ensure_allocated buf;
        Hashtbl.replace tbl id buf
    | _ -> ()
  done;
  tbl

(* Find the output Buffer node — the highest-id Buffer in the graph.
   This corresponds to the buffer created from the Contiguous→Bufferize chain,
   which is emitted last by pm_add_buffers. *)
let find_output_buffer_id (graph : Tensor.t) : Tensor.id =
  let last_buf = ref (-1) in
  for id = 0 to Tensor.length graph - 1 do
    match Tensor.view graph id with
    | Buffer _ -> last_buf := id
    | _ -> ()
  done;
  if !last_buf < 0 then failwith "Jit: no output buffer found in kernel graph";
  !last_buf

(* Build the buffer argument list for a kernel execution.
   Kernel params are numbered by tensor_subtree_to_kernel:
   - Param nodes keep their original slot
   - Buffer nodes get sequential indices starting from max(Param.slot)+1,
     assigned in backward_slice (topological) order
   We sort CALL args by their kernel param index to match. *)
let build_buf_args (graph : Tensor.t) (alloc_bufs : (Tensor.id, Tolk.Device.Buffer.t) Hashtbl.t)
    (param_bufs : (Tensor.id, Tolk.Device.Buffer.t) Hashtbl.t) (arg_ids : Tensor.id array)
    : Tolk.Device.Buffer.t list =
  (* Compute the kernel param index for each arg.
     Params: slot. Buffers: max_param_slot + 1 + counter in id order. *)
  let max_param_slot = Array.fold_left (fun acc id ->
    match Tensor.view graph id with
    | Param { slot; _ } -> Int.max acc slot
    | _ -> acc) (-1) arg_ids in
  let buf_counter = ref (max_param_slot + 1) in
  (* Walk args sorted by node id to match backward_slice order *)
  let sorted_ids = Array.copy arg_ids in
  Array.sort compare sorted_ids;
  let idx_map : (Tensor.id, int) Hashtbl.t = Hashtbl.create 8 in
  Array.iter (fun node_id ->
    match Tensor.view graph node_id with
    | Param { slot; _ } -> Hashtbl.replace idx_map node_id slot
    | Buffer _ ->
        Hashtbl.replace idx_map node_id !buf_counter;
        incr buf_counter
    | _ -> ()) sorted_ids;
  (* Build buffer list sorted by kernel param index *)
  let indexed = Array.map (fun node_id ->
    let kernel_idx = Hashtbl.find idx_map node_id in
    let buf = match Hashtbl.find_opt alloc_bufs node_id with
      | Some b -> b
      | None ->
          match Hashtbl.find_opt param_bufs node_id with
          | Some b -> b
          | None -> failwith (Printf.sprintf "Jit: no buffer for node %d" node_id)
    in
    (kernel_idx, buf)) arg_ids in
  Array.sort (fun (a, _) (b, _) -> compare a b) indexed;
  Array.to_list (Array.map snd indexed)

(* --- State machine ------------------------------------------------------- *)

type state =
  | Warmup
  | Capture
  | Compiled of captured_jit

(* --- Public API ---------------------------------------------------------- *)

let trace (type a b c d)
    ?(device : Tolk.Device.t option)
    (f : (a, b) Nx.t -> (c, d) Nx.t) : (a, b) Nx.t -> (c, d) Nx.t =
  let state = ref Warmup in
  fun (x : (a, b) Nx.t) ->
    match !state with
    | Warmup ->
        state := Capture;
        f x

    | Capture ->
        let dev = match device with
          | Some d -> d
          | None -> failwith "Jit.trace: device is required for JIT compilation"
        in
        (* Build lazy tensor graph under effect handler *)
        let (graph, ctx, result) =
          capture_graph ~device_name:(Tolk.Device.name dev) f x
        in
        ignore graph;

        (* Schedule: rangeify → kernel ASTs *)
        let kernel_graph = Tolk.Rangeify.get_kernel_graph graph in

        (* Compile kernels and allocate output/intermediate buffers *)
        let schedule = Array.of_list (extract_schedule dev kernel_graph) in
        let alloc_bufs = allocate_buffers dev kernel_graph in
        let output_buf_id = find_output_buffer_id kernel_graph in

        (* Build param node id → slot mapping for the kernel graph.
           After rangeify, Param nodes preserve their structure. *)
        let param_node_slots = Hashtbl.create 16 in
        for id = 0 to Tensor.length kernel_graph - 1 do
          match Tensor.view kernel_graph id with
          | Param { slot; _ } -> Hashtbl.replace param_node_slots id slot
          | _ -> ()
        done;

        (* Create device buffers for ALL params (function args + captured constants).
           Param node id → device buffer. *)
        let param_bufs : (Tensor.id, Tolk.Device.Buffer.t) Hashtbl.t = Hashtbl.create 16 in
        Hashtbl.iter (fun node_id slot ->
          let (_repr, dt, shape) = Hashtbl.find ctx.slot_tensors slot in
          let num_elements = Int.max 1 (Array.fold_left ( * ) 1 shape) in
          let buf = Tolk.Device.create_buffer ~size:num_elements ~dtype:dt dev in
          Tolk.Device.Buffer.ensure_allocated buf;
          Hashtbl.replace param_bufs node_id buf)
          param_node_slots;

        (* Copy input data to param device buffers.
           Slot 0 = function argument x; other slots = captured constants. *)
        Hashtbl.iter (fun node_id slot ->
          let buf = Hashtbl.find param_bufs node_id in
          if slot = 0 then begin
            let host = Nx_effect.to_host x in
            let nbytes = Tolk.Device.Buffer.nbytes buf in
            let src = Bytes.create nbytes in
            Nx_buffer.blit_to_bytes host src;
            Tolk.Device.Buffer.copyin buf src
          end else begin
            let (repr, _dt, _shape) = Hashtbl.find ctx.slot_tensors slot in
            let nbytes = Tolk.Device.Buffer.nbytes buf in
            let src = Bytes.create nbytes in
            (* Use Obj.magic to access the Nx_effect.t for the captured tensor *)
            let host = Nx_effect.to_host (Obj.obj repr : (a, b) Nx_effect.t) in
            Nx_buffer.blit_to_bytes host src;
            Tolk.Device.Buffer.copyin buf src
          end)
          param_node_slots;

        (* Execute schedule *)
        let q = Tolk.Device.queue dev in
        Array.iter (fun item ->
          let bufs = build_buf_args kernel_graph alloc_bufs param_bufs item.arg_node_ids in
          Tolk.Device.Queue.exec q item.program bufs [])
          schedule;
        Tolk.Device.Queue.synchronize q;

        (* Record for replay *)
        let input_shape = infer_shape x in
        let out_shape = infer_shape result in
        state := Compiled {
          schedule; kernel_graph; alloc_bufs;
          slot_tensors = ctx.slot_tensors;
          param_node_slots;
          expected_input_shape = input_shape;
          output_shape = out_shape;
          output_buf_id; device = dev;
        };

        (* Copy output back to Nx tensor *)
        let output_buf = Hashtbl.find alloc_bufs output_buf_id in
        device_buffer_to_nx (Nx_effect.dtype result) out_shape output_buf

    | Compiled captured ->
        (* Validate inputs *)
        let input_shape = infer_shape x in
        if input_shape <> captured.expected_input_shape then
          invalid_arg (Printf.sprintf "Jit: input shape changed: expected [%s], got [%s]"
            (String.concat ";" (List.map string_of_int (Array.to_list captured.expected_input_shape)))
            (String.concat ";" (List.map string_of_int (Array.to_list input_shape))));

        (* Create fresh param buffers and copy data *)
        let dev = captured.device in
        let param_bufs : (Tensor.id, Tolk.Device.Buffer.t) Hashtbl.t = Hashtbl.create 16 in
        Hashtbl.iter (fun node_id slot ->
          let (_repr, dt, shape) = Hashtbl.find captured.slot_tensors slot in
          let num_elements = Int.max 1 (Array.fold_left ( * ) 1 shape) in
          let buf = Tolk.Device.create_buffer ~size:num_elements ~dtype:dt dev in
          Tolk.Device.Buffer.ensure_allocated buf;
          if slot = 0 then begin
            let host = Nx_effect.to_host x in
            let nbytes = Tolk.Device.Buffer.nbytes buf in
            let src = Bytes.create nbytes in
            Nx_buffer.blit_to_bytes host src;
            Tolk.Device.Buffer.copyin buf src
          end else begin
            let (repr, _dt, _shape) = Hashtbl.find captured.slot_tensors slot in
            let nbytes = Tolk.Device.Buffer.nbytes buf in
            let src = Bytes.create nbytes in
            let host = Nx_effect.to_host (Obj.obj repr : (a, b) Nx_effect.t) in
            Nx_buffer.blit_to_bytes host src;
            Tolk.Device.Buffer.copyin buf src
          end;
          Hashtbl.replace param_bufs node_id buf)
          captured.param_node_slots;

        (* Execute schedule *)
        let q = Tolk.Device.queue dev in
        Array.iter (fun item ->
          let bufs = build_buf_args captured.kernel_graph captured.alloc_bufs
                       param_bufs item.arg_node_ids in
          Tolk.Device.Queue.exec q item.program bufs [])
          captured.schedule;
        Tolk.Device.Queue.synchronize q;

        (* Copy output back to Nx tensor *)
        let output_buf = Hashtbl.find captured.alloc_bufs captured.output_buf_id in
        device_buffer_to_nx (Obj.magic (Nx_effect.dtype x) : (c, d) Nx.dtype)
          captured.output_shape output_buf

(* --- Trace graph (debug/inspection) ------------------------------------- *)

type traced = {
  tensor_graph : Tensor.t;
  kernel_graph : Tensor.t;
  rendered_source : string list;
}

let extract_rendered_sources ren kernel_graph =
  let sources = ref [] in
  for id = 0 to Tensor.length kernel_graph - 1 do
    match Tensor.view kernel_graph id with
    | Call { callee = Ast kernel; _ } ->
        let processed = Tolk.Pipeline.full_rewrite_to_sink ~optimize:true ren kernel in
        let name = match Tolk_ir.Kernel.view processed with
          | Tolk_ir.Kernel.Sink { kernel_info = Some ki; _ } -> ki.name
          | _ -> "kernel" in
        let prog = Tolk.Linearizer.linearize processed in
        sources := String.trim (Tolk.Renderer.render ren ~name prog) :: !sources
    | _ -> ()
  done;
  List.rev !sources

let trace_graph (type a b c d)
    ?(device : Tolk.Device.t option)
    (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t) : traced =
  let device_name = match device with
    | Some dev -> Tolk.Device.name dev
    | None -> "CPU" in
  let (tensor_graph, _ctx, _result) = capture_graph ~device_name f x in
  let kernel_graph = Tolk.Rangeify.get_kernel_graph tensor_graph in
  let ren = match device with
    | Some dev -> Tolk.Device.renderer dev
    | None -> Tolk.Cstyle.clang_no_abi in
  let rendered_source = extract_rendered_sources ren kernel_graph in
  { tensor_graph; kernel_graph; rendered_source }

let reset () = ()
