(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* JIT compilation via effect handler.

   Intercepts Nx tensor operations to build a computation graph, then
   delegates scheduling, compilation, memory planning, and replay to
   the tolk JIT engine (Tolk.Jit.Tiny_jit).

   Three phases (managed by Tiny_jit):
   - Warmup (cnt=0): execute eagerly via C backend
   - Capture (cnt=1): intercept effects, build lazy graph, schedule,
     compile, and execute
   - Replay (cnt>=2): validate inputs, substitute buffers, execute *)

open Nx_effect
module T = Tolk_ir.Tensor
module B = Tolk.Device.Buffer

(* Dtype mapping *)

let tolk_dtype (type a b) (dt : (a, b) Nx.dtype) : Tolk_ir.Dtype.t =
  let open Tolk_ir.Dtype in
  match Nx_core.Dtype.to_string dt with
  | "float32" -> float32
  | "float64" -> float64
  | "float16" -> float16
  | "int32" -> int32
  | "int64" -> int64
  | "int8" -> int8
  | "int16" -> int16
  | "uint8" -> uint8
  | "uint16" -> uint16
  | "uint32" -> uint32
  | "uint64" -> uint64
  | "bool" -> bool
  | s -> failwith (Printf.sprintf "Jit: unsupported dtype %s" s)

(* Buffer transfer *)

let nx_to_device_buffer (type a b) dev (t : (a, b) Nx_effect.t) : B.t =
  let host = Nx_effect.to_host t in
  let shape = Nx_effect.view t |> Nx_core.View.shape in
  let num_elements = Int.max 1 (Array.fold_left ( * ) 1 shape) in
  let dt = tolk_dtype (Nx_effect.dtype t) in
  let buf = Tolk.Device.create_buffer ~size:num_elements ~dtype:dt dev in
  B.ensure_allocated buf;
  let nbytes = num_elements * Tolk_ir.Dtype.itemsize dt in
  let src_bytes = Bytes.create nbytes in
  Nx_buffer.blit_to_bytes host src_bytes;
  B.copyin buf src_bytes;
  buf

let device_buffer_to_nx (type a b) (dt : (a, b) Nx.dtype) (shape : int array)
    (buf : B.t) : (a, b) Nx_effect.t =
  let num_elements = Int.max 1 (Array.fold_left ( * ) 1 shape) in
  let tdt = tolk_dtype dt in
  let nbytes = num_elements * Tolk_ir.Dtype.itemsize tdt in
  let dst_bytes = Bytes.create nbytes in
  B.copyout buf dst_bytes;
  let ctx = Nx_effect.create_context () in
  let nx_buf = Nx_effect.buffer ctx dt shape in
  let host = Nx_effect.to_host nx_buf in
  Nx_buffer.blit_from_bytes dst_bytes host;
  nx_buf

(* Identity-keyed hash table *)

(* Physical identity ([==]) tracks tensor values through the effect
   handler — structural equality would collide on placeholder tensors. *)
module Phys_tbl = Hashtbl.Make (struct
  type t = Obj.t
  let equal = ( == )
  let hash x = Hashtbl.hash (Obj.obj x : int)
end)

(* Capture context *)

type capture_ctx = {
  tensor_to_node : T.t Phys_tbl.t;
  slot_tensors : (int, Obj.t * Tolk_ir.Dtype.t * int array) Hashtbl.t;
  mutable param_count : int;
  device_node : T.t;
}

let create_capture_ctx device_node = {
  tensor_to_node = Phys_tbl.create 64;
  slot_tensors = Hashtbl.create 16;
  param_count = 0;
  device_node;
}

let make_placeholder (type a b) (dt : (a, b) Nx.dtype) (shape : int array)
    : (a, b) Nx_effect.t =
  let ctx = Nx_effect.create_context () in
  Nx_effect.buffer ctx dt shape

let register ctx (t : _ Nx_effect.t) (node : T.t) : unit =
  Phys_tbl.replace ctx.tensor_to_node (Obj.repr t) node

let shape_node dims =
  let int_ n =
    T.const (Tolk_ir.Const.int Tolk_ir.Dtype.Val.index n) Tolk_ir.Dtype.index in
  match dims with
  | [] -> T.const (Tolk_ir.Const.int Tolk_ir.Dtype.Val.index 1) Tolk_ir.Dtype.index
  | _ ->
      match List.map int_ dims with [d] -> d | ds -> T.vectorize ~srcs:ds

(* Read a scalar value from an Nx tensor and construct a Const.t.
   Used to fold scalar constants into the kernel IR. *)
let read_scalar_const (type a b) (t : (a, b) Nx_effect.t)
    (dt : Tolk_ir.Dtype.t) : Tolk_ir.Const.t =
  let vdt = Tolk_ir.Dtype.val_of dt in
  let nbytes = Tolk_ir.Dtype.itemsize dt in
  let buf = Bytes.create nbytes in
  Nx_buffer.blit_to_bytes (Nx_effect.to_host t) buf;
  if Tolk_ir.Dtype.is_float dt then
    let v = if nbytes = 4 then
      Int32.float_of_bits (Bytes.get_int32_le buf 0)
    else Int64.float_of_bits (Bytes.get_int64_le buf 0) in
    Tolk_ir.Const.float vdt v
  else if Tolk_ir.Dtype.equal dt Tolk_ir.Dtype.bool then
    Tolk_ir.Const.bool (Bytes.get_uint8 buf 0 <> 0)
  else
    let v = if nbytes <= 4 then
      Int32.to_int (Bytes.get_int32_le buf 0)
    else Int64.to_int (Bytes.get_int64_le buf 0) in
    Tolk_ir.Const.int vdt v

let lookup_or_param ctx (t : _ Nx_effect.t) : T.t =
  let key = Obj.repr t in
  match Phys_tbl.find_opt ctx.tensor_to_node key with
  | Some node -> node
  | None ->
      let dt = tolk_dtype (Nx_effect.dtype t) in
      let shape = Nx_effect.view t |> Nx_core.View.shape in
      if shape = [||] then begin
        (* Scalar constant: fold into the IR directly. *)
        let cv = read_scalar_const t dt in
        let node = T.const cv dt in
        Phys_tbl.replace ctx.tensor_to_node key node;
        node
      end else begin
        let slot = ctx.param_count in
        ctx.param_count <- slot + 1;
        let sh = shape_node (Array.to_list shape) in
        let node =
          T.param ~slot ~dtype:dt ~shape:sh ~device:ctx.device_node ()
        in
        Phys_tbl.replace ctx.tensor_to_node key node;
        Hashtbl.replace ctx.slot_tensors slot (key, dt, shape);
        node
      end

(* Graph building *)

let emit_binary ctx op (a : _ Nx_effect.t) (b : _ Nx_effect.t) ~dtype ~shape =
  let a_node = lookup_or_param ctx a in
  let b_node = lookup_or_param ctx b in
  let out = make_placeholder dtype shape in
  let node = T.binary ~op ~lhs:a_node ~rhs:b_node in
  register ctx out node;
  out

let emit_unary ctx op (src : _ Nx_effect.t) ~dtype ~shape =
  let src_node = lookup_or_param ctx src in
  let out = make_placeholder dtype shape in
  let node = T.unary ~op ~src:src_node in
  register ctx out node;
  out

let emit_reduce ctx op ~axes (src : _ Nx_effect.t) ~dtype ~shape =
  let src_node = lookup_or_param ctx src in
  let out = make_placeholder dtype shape in
  let node = T.reduce_axis ~src:src_node ~op ~axes in
  register ctx out node;
  out

let infer_shape (t : _ Nx_effect.t) = Nx_effect.view t |> Nx_core.View.shape

let reduce_shape in_shape axes_list =
  let out =
    List.filteri (fun i _ -> not (List.mem i axes_list))
      (Array.to_list in_shape)
  in
  if out = [] then [||] else Array.of_list out

(* Effect handler *)

let make_capture_handler ctx =
  let open Effect.Deep in
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    match eff with
    | E_add { a; b } ->
        Some (fun k -> continue k
          (emit_binary ctx `Add a b ~dtype:(Nx_effect.dtype a)
             ~shape:(infer_shape a)))
    | E_sub { a; b } ->
        Some (fun k -> continue k
          (emit_binary ctx `Sub a b ~dtype:(Nx_effect.dtype a)
             ~shape:(infer_shape a)))
    | E_mul { a; b } ->
        Some (fun k -> continue k
          (emit_binary ctx `Mul a b ~dtype:(Nx_effect.dtype a)
             ~shape:(infer_shape a)))
    | E_max { a; b } ->
        Some (fun k -> continue k
          (emit_binary ctx `Max a b ~dtype:(Nx_effect.dtype a)
             ~shape:(infer_shape a)))
    | E_cmpeq { a; b } ->
        Some (fun k -> continue k
          (emit_binary ctx `Cmpeq a b ~dtype:Nx.bool
             ~shape:(infer_shape a)))
    | E_cmplt { a; b } ->
        Some (fun k -> continue k
          (emit_binary ctx `Cmplt a b ~dtype:Nx.bool
             ~shape:(infer_shape a)))
    | E_neg { t_in } ->
        Some (fun k -> continue k
          (emit_unary ctx `Neg t_in ~dtype:(Nx_effect.dtype t_in)
             ~shape:(infer_shape t_in)))
    | E_sqrt { t_in } ->
        Some (fun k -> continue k
          (emit_unary ctx `Sqrt t_in ~dtype:(Nx_effect.dtype t_in)
             ~shape:(infer_shape t_in)))
    | E_sin { t_in } ->
        Some (fun k -> continue k
          (emit_unary ctx `Sin t_in ~dtype:(Nx_effect.dtype t_in)
             ~shape:(infer_shape t_in)))
    | E_recip { t_in } ->
        Some (fun k -> continue k
          (emit_unary ctx `Recip t_in ~dtype:(Nx_effect.dtype t_in)
             ~shape:(infer_shape t_in)))
    | E_reduce_sum { t_in; axes; keepdims = _ } ->
        Some (fun k ->
          let axes_l = Array.to_list axes in
          continue k
            (emit_reduce ctx `Add ~axes:axes_l t_in
               ~dtype:(Nx_effect.dtype t_in)
               ~shape:(reduce_shape (infer_shape t_in) axes_l)))
    | E_reduce_max { t_in; axes; keepdims = _ } ->
        Some (fun k ->
          let axes_l = Array.to_list axes in
          continue k
            (emit_reduce ctx `Max ~axes:axes_l t_in
               ~dtype:(Nx_effect.dtype t_in)
               ~shape:(reduce_shape (infer_shape t_in) axes_l)))
    | E_reshape { t_in; new_shape } ->
        Some (fun k ->
          let src_node = lookup_or_param ctx t_in in
          let out = make_placeholder (Nx_effect.dtype t_in) new_shape in
          let sh = shape_node (Array.to_list new_shape) in
          let node = T.reshape ~src:src_node ~shape:sh in
          register ctx out node;
          continue k out)
    | E_permute { t_in; axes } ->
        Some (fun k ->
          let src_node = lookup_or_param ctx t_in in
          let in_shape = infer_shape t_in in
          let out_shape = Array.map (fun ax -> in_shape.(ax)) axes in
          let out = make_placeholder (Nx_effect.dtype t_in) out_shape in
          let node = T.permute ~src:src_node ~order:(Array.to_list axes) in
          register ctx out node;
          continue k out)
    | E_expand { t_in; new_target_shape } ->
        Some (fun k ->
          let src_node = lookup_or_param ctx t_in in
          let out = make_placeholder (Nx_effect.dtype t_in) new_target_shape in
          let sh = shape_node (Array.to_list new_target_shape) in
          let node = T.expand ~src:src_node ~shape:sh in
          register ctx out node;
          continue k out)
    | E_cast { t_in; target_dtype } ->
        Some (fun k ->
          let src_node = lookup_or_param ctx t_in in
          let dt = tolk_dtype target_dtype in
          let out = make_placeholder target_dtype (infer_shape t_in) in
          let node = T.cast ~src:src_node ~dtype:dt in
          register ctx out node;
          continue k out)
    | E_where { condition; if_true; if_false } ->
        Some (fun k ->
          let c_node = lookup_or_param ctx condition in
          let t_node = lookup_or_param ctx if_true in
          let f_node = lookup_or_param ctx if_false in
          let out =
            make_placeholder (Nx_effect.dtype if_true) (infer_shape if_true)
          in
          let node = T.ternary ~op:`Where ~a:c_node ~b:t_node ~c:f_node in
          register ctx out node;
          continue k out)
    | E_const_scalar { context = _; value; dtype = dt } ->
        Some (fun k ->
          let out = make_placeholder dt [||] in
          let tdt = tolk_dtype dt in
          let vdt = Tolk_ir.Dtype.val_of tdt in
          let cv =
            if Tolk_ir.Dtype.is_float tdt then
              Tolk_ir.Const.float vdt (Obj.magic value : float)
            else if Tolk_ir.Dtype.equal tdt Tolk_ir.Dtype.bool then
              Tolk_ir.Const.bool (Obj.magic value : bool)
            else Tolk_ir.Const.int vdt (Obj.magic value : int)
          in
          let node = T.const cv tdt in
          register ctx out node;
          continue k out)
    | _ -> None
  in
  { retc = (fun x -> x); exnc = raise; effc }

(* Graph capture *)

let capture_graph (type a b c d) ?(device_name = "CPU")
    (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t)
    : T.t * capture_ctx * (c, d) Nx_effect.t =
  let device_node = T.device (Single device_name) in
  let ctx = create_capture_ctx device_node in
  let handler = make_capture_handler ctx in
  let result = Effect.Deep.match_with f x handler in
  let result_node = lookup_or_param ctx result in
  let contig = T.contiguous ~src:result_node () in
  let graph = T.sink [ contig ] in
  (graph, ctx, result)

(* Scheduling bridge *)

(* Build the buffers callback for Schedule.linear_to_schedule.
   Maps PARAM tensor nodes to device buffers: slot 0 is the function
   input, other slots are captured constants. *)
let make_buffers_cb ctx dev input_buf =
  let cache : (int, B.t) Hashtbl.t = Hashtbl.create 16 in
  fun (node : T.t) ->
    match T.view node with
    | Param { slot; _ } ->
        (match Hashtbl.find_opt cache slot with
         | Some buf -> Some buf
         | None ->
             let buf =
               if slot = 0 then input_buf
               else
                 let repr, dt, shape = Hashtbl.find ctx.slot_tensors slot in
                 let num = Int.max 1 (Array.fold_left ( * ) 1 shape) in
                 let buf =
                   Tolk.Device.create_buffer ~size:num ~dtype:dt dev in
                 B.ensure_allocated buf;
                 let nbytes = num * Tolk_ir.Dtype.itemsize dt in
                 let src = Bytes.create nbytes in
                 let host =
                   Nx_effect.to_host (Obj.obj repr : (_, _) Nx_effect.t) in
                 Nx_buffer.blit_to_bytes host src;
                 B.copyin buf src;
                 buf
             in
             Hashtbl.replace cache slot buf;
             Some buf)
    | _ -> None

(* Find the output buffer in the captured schedule — first non-None
   buffer of the last exec item. *)
let find_output_buf cache =
  let n = Array.length cache in
  let rec loop i =
    if i < 0 then failwith "Jit: no output buffer in schedule";
    match (cache.(i)).Tolk.Jit.bufs.(0) with
    | Some buf -> buf
    | None -> loop (i - 1)
  in
  loop (n - 1)

(* Public API *)

let trace (type a b c d) ?(device : Tolk.Device.t option)
    (f : (a, b) Nx.t -> (c, d) Nx.t) : (a, b) Nx.t -> (c, d) Nx.t =
  (* The Tiny_jit is created lazily on the second call (capture phase),
     because warmup runs eagerly and doesn't need a device. *)
  let tjit_ref : (unit -> (c, d) Nx.t) Tolk.Jit.tiny_jit option ref =
    ref None in
  let input_nx_dtype : Obj.t option ref = ref None in
  let input_shape : int array ref = ref [||] in
  let output_nx_dtype : Obj.t option ref = ref None in
  let output_shape : int array ref = ref [||] in
  let buffers_ref : (T.t -> B.t option) ref = ref (fun _ -> None) in
  let warmup_done = ref false in
  let ensure_tjit () =
    match !tjit_ref with
    | Some t -> t
    | None ->
        let dev = match device with
          | Some d -> d
          | None -> failwith "Jit.trace: device is required for JIT"
        in
        let ren = Tolk.Device.renderer dev in
        let get_program = Tolk.Codegen.get_program dev ren in
        let device_name = Tolk.Device.name dev in
        let fxn (input_bufs : B.t array) _var_vals
            : unit -> (c, d) Nx.t =
          if Tolk.Jit.is_capturing () then begin
            (* Capture: build tensor graph under effect handler,
               schedule, and register the linear. *)
            let x = make_placeholder
              (Obj.obj (Option.get !input_nx_dtype) : (a, b) Nx.dtype)
              !input_shape in
            let graph, ctx, result =
              capture_graph ~device_name f x in
            output_shape := infer_shape result;
            output_nx_dtype :=
              Some (Obj.repr (Nx_effect.dtype result));
            buffers_ref := make_buffers_cb ctx dev input_bufs.(0);
            let linear =
              match Tolk.Schedule.lower_sink_to_linear
                      ~get_kernel_graph:Tolk.Rangeify.get_kernel_graph
                      graph with
              | Some l -> l
              | None -> failwith "Jit: scheduling failed"
            in
            Tolk.Jit.add_linear linear;
            let out_dt : (c, d) Nx.dtype =
              Obj.obj (Option.get !output_nx_dtype) in
            let out_shape = !output_shape in
            fun () ->
              let c = Option.get
                (Tolk.Jit.captured (Option.get !tjit_ref)) in
              let buf = find_output_buf (Tolk.Jit.jit_cache c) in
              device_buffer_to_nx out_dt out_shape buf
          end else begin
            (* Warmup inside Tiny_jit (cnt=0). *)
            let x = device_buffer_to_nx
              (Obj.obj (Option.get !input_nx_dtype) : (a, b) Nx.dtype)
              !input_shape input_bufs.(0) in
            let result = f x in
            output_shape := infer_shape result;
            output_nx_dtype :=
              Some (Obj.repr (Nx_effect.dtype result));
            fun () -> result
          end
        in
        let tjit =
          Tolk.Jit.create ~device:dev ~get_program ~fxn () in
        tjit_ref := Some tjit;
        tjit
  in
  fun (x : (a, b) Nx.t) ->
    if not !warmup_done then begin
      (* Warmup: run eagerly on the C backend, no device needed. *)
      warmup_done := true;
      f x
    end else begin
      let tjit = ensure_tjit () in
      input_nx_dtype := Some (Obj.repr (Nx_effect.dtype x));
      input_shape := infer_shape x;
      let dev = match device with
        | Some d -> d
        | None -> failwith "Jit.trace: device is required for JIT"
      in
      let buf = nx_to_device_buffer dev x in
      let thunk = Tolk.Jit.call tjit [| buf |] []
        ~buffers:(fun node -> !buffers_ref node) in
      thunk ()
    end

(* Trace graph (debug/inspection) *)

type traced = {
  tensor_graph : T.t;
  kernel_graph : T.t;
  rendered_source : string list;
}

let extract_rendered_sources dev ren kernel_graph =
  let sources = ref [] in
  List.iter (fun node ->
    match T.view node with
    | Call { callee = Ast kernel; _ } ->
        let p = Tolk.Codegen.get_program dev ren kernel in
        sources := String.trim (Tolk.Program_spec.src p) :: !sources
    | _ -> ())
    (T.toposort kernel_graph);
  List.rev !sources

let trace_graph (type a b c d) ~(device : Tolk.Device.t)
    (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t) : traced =
  let device_name = Tolk.Device.name device in
  let tensor_graph, _ctx, _result = capture_graph ~device_name f x in
  let kernel_graph = Tolk.Rangeify.get_kernel_graph tensor_graph in
  let ren = Tolk.Device.renderer device in
  let rendered_source = extract_rendered_sources device ren kernel_graph in
  { tensor_graph; kernel_graph; rendered_source }

let reset () = ()
