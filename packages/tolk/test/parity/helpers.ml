open Tolk
open Tolk_uop
module U = Uop

let global_fptr =
  Dtype.Ptr
    (Dtype.Ptr.create Dtype.Val.float32 ~addrspace:Global ~size:(-1))

let idx n = U.const (Const.int Dtype.Val.weakint n)

let all_backends =
  [
    ("cpu", Cstyle.clang_no_abi Gpu_target.X86_64);
    ("cuda", Cstyle.cuda Gpu_target.SM80);
    ("metal", Cstyle.metal (Gpu_target.Apple 7));
    ("opencl", Cstyle.opencl "");
  ]

let gpu_backends = List.filter (fun (name, _) -> name <> "cpu") all_backends

let write_file path contents =
  let oc = open_out path in
  output_string oc contents;
  output_char oc '\n';
  close_out oc

(* Drop a single trailing newline — [uops_to_string] and [render] both emit
   a terminator we add back when writing the file. *)
let rstrip_newline s =
  let n = String.length s in
  if n > 0 && s.[n - 1] = '\n' then String.sub s 0 (n - 1) else s

(* Stage 5: kernel AST after the full codegen rewrite pipeline. Per-backend
   because opt passes and pre/extra matchers are renderer-specific. *)
let stage5 ?(optimize = true) ren sink =
  let processed = Codegen.full_rewrite_to_sink ~optimize ren sink in
  rstrip_newline (Render.uops_to_string processed)

(* Stage 7: rendered backend source. *)
let stage7 ?(optimize = true) ren sink =
  let processed = Codegen.full_rewrite_to_sink ~optimize ren sink in
  let name =
    match U.as_kernel_info processed with Some ki -> ki.name | None -> "kernel"
  in
  String.trim (Renderer.render ren ~name (Linearizer.linearize processed))

type stage = Stage5 | Stage7

let stage_name = function Stage5 -> "stage5" | Stage7 -> "stage7"

let dump
    ?(backends = all_backends)
    ?(optimize = true)
    ~stages
    ~out_dir
    sink =
  List.iter
    (fun (backend, ren) ->
      List.iter
        (fun stage ->
          let path =
            Filename.concat out_dir
              (Printf.sprintf "%s_%s.actual" (stage_name stage) backend)
          in
          let out =
            match stage with
            | Stage5 -> stage5 ~optimize ren sink
            | Stage7 -> stage7 ~optimize ren sink
          in
          write_file path out)
        stages)
    backends

let dump_stage7 ?backends ?optimize ~out_dir sink =
  dump ?backends ?optimize ~stages:[ Stage7 ] ~out_dir sink

(* Tensor-graph builders, shared by rangeify parity cases. *)

let mk_shape dims =
  let ids =
    List.map (fun s -> U.const (Const.int Dtype.Val.weakint s)) dims
  in
  match ids with [ d ] -> d | ds -> U.stack ds

let mk_param ~idx ?(dtype = Dtype.float32) ?(device = "CPU") shape =
  let shape_id = if shape = [] then None else Some (mk_shape shape) in
  U.param ~slot:idx ~dtype ?shape:shape_id ~device:(Single device) ()

(* Multi-device PARAM. [shape] is the per-shard shape; when [axis] is given
   the stored shape multiplies that axis by the device count, mirroring the
   reference [UOp.param]. *)
let mk_param_multi ~idx ?(dtype = Dtype.float32) ~devices ?axis shape =
  let shape =
    match axis with
    | Some a ->
        List.mapi
          (fun i s -> if i = a then s * List.length devices else s)
          shape
    | None -> shape
  in
  let shape_id = if shape = [] then None else Some (mk_shape shape) in
  U.param ~slot:idx ~dtype ?shape:shape_id ~device:(Multi devices) ?axis ()

let wrap_sink srcs =
  let contigs = List.map (fun src -> U.contiguous ~src ()) srcs in
  U.sink contigs

(* Tensor-graph entry points: run rangeify to split into kernels, then per
   kernel run the codegen/render pipeline. *)

(* Inline kernel AST roots: CALL srcs whose body carries kernel metadata.
   Skips precompiled functions (e.g. allreduce), matching the reference. *)
let extract_kernels root =
  List.filter_map
    (fun node ->
      match U.op node, U.as_call node with
      | Ops.Call, Some { body; _ } when U.as_kernel_info body <> None ->
          Some body
      | _ -> None)
    (U.toposort root)

let stage5_tensor ?(optimize = true) ren sink =
  let kg = Rangeify.get_kernel_graph sink in
  let kernels = extract_kernels kg in
  let parts =
    List.mapi
      (fun i k ->
        let processed = Codegen.full_rewrite_to_sink ~optimize ren k in
        let body = rstrip_newline (Render.uops_to_string processed) in
        match kernels with
        | [ _ ] -> body
        | _ -> Printf.sprintf "=== kernel %d ===\n%s" i body)
      kernels
  in
  String.concat "\n" parts

let stage7_tensor ?(optimize = true) ren sink =
  let kg = Rangeify.get_kernel_graph sink in
  let kernels = extract_kernels kg in
  let sources =
    List.map
      (fun k ->
        let processed = Codegen.full_rewrite_to_sink ~optimize ren k in
        let name =
          match U.as_kernel_info processed with
          | Some ki -> ki.name
          | None -> "kernel"
        in
        String.trim (Renderer.render ren ~name (Linearizer.linearize processed)))
      kernels
  in
  String.concat "\n---\n" sources

let dump_tensor
    ?(backends = all_backends)
    ?(optimize = true)
    ~stages
    ~out_dir
    sink =
  List.iter
    (fun (backend, ren) ->
      List.iter
        (fun stage ->
          let path =
            Filename.concat out_dir
              (Printf.sprintf "%s_%s.actual" (stage_name stage) backend)
          in
          let out =
            match stage with
            | Stage5 -> stage5_tensor ~optimize ren sink
            | Stage7 -> stage7_tensor ~optimize ren sink
          in
          write_file path out)
        stages)
    backends

(* Stage 7 from a pre-linearized flat [Program_spec.program]. Cstyle-style: renders
   directly with no codegen rewrite or linearize pass. *)
let stage7_program ?(name = "test") ren program =
  String.trim (Renderer.render ren ~name program)

let dump_stage7_program
    ?(backends = all_backends)
    ?(name = "test")
    ~out_dir
    program =
  List.iter
    (fun (backend, ren) ->
      let path =
        Filename.concat out_dir (Printf.sprintf "stage7_%s.actual" backend)
      in
      write_file path (stage7_program ~name ren program))
    backends
