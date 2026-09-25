(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Codegen entry point — optimization dispatch + lowering. Ported from
   tinygrad/codegen/__init__.py onto the Tolk_uop IR. *)

open Tolk_uop
module U = Uop

(* Environment *)

let debug () = Helpers.getenv "DEBUG" 0
let beam_estimate () = Helpers.getenv "BEAM_ESTIMATE" 1
let noopt () = Helpers.Context_var.get Helpers.noopt

let prod = List.fold_left ( * ) 1

(* Allocate raw buffers for beam search from the kernel's global buffer Param
   nodes. Address space distinguishes buffer params from symbolic ALU params;
   the element dtype is the node dtype and the element count is the shape
   product. *)
let buffer_params ast =
  U.backward_slice ast
  |> List.filter_map (fun u ->
       match U.as_param u with
       | Some { param; _ }
         when param.slot >= 0 && param.addrspace = Dtype.Global ->
           Some (param.slot, U.dtype u, prod (U.max_shape u))
       | _ -> None)
  |> List.sort (fun (a, _, _) (b, _, _) -> Int.compare a b)

let has_tag u = match U.node_tag u with Some _ -> true | None -> false

let sym = Symbolic.sym

let kernel_info_exn stage sink =
  match U.as_kernel_info sink with
  | Some ki -> ki
  | None ->
      invalid_arg
        (Printf.sprintf "Codegen.%s: Sink is missing KernelInfo" stage)

let rec make_beam_search device beam_width =
  Option.map
    (fun dev k ->
      (* Timing buffers for this kernel's candidates. [nolru] so their
         release returns the memory to the driver: the LRU cache matches by
         exact size, so buffers of distinct per-kernel shapes would otherwise
         pile up in it for the rest of the compile and can OOM the device on
         large graphs. Freed explicitly below — [beam_search] only times on
         them — rather than waiting for a GC that may run much later. *)
      let spec = { Device.Buffer_spec.default with nolru = true } in
      let var_vals = U.symbolic_vars (Postrange.ast k)
          |> List.map (fun (_, name, lo, hi) ->
              let midpoint = Bound.integer (Bound.floordiv (Bound.add lo hi) (Bound.int 2)) in
              if not (Z.fits_int64 midpoint) then
                invalid_arg "Codegen: timing midpoint does not fit int64";
              name, Z.to_int64 midpoint) in
      let rawbufs =
        List.map
          (fun (_, dtype, size) -> Device.create_buffer ~size ~dtype ~spec dev)
          (buffer_params (Postrange.ast k))
      in
      Fun.protect
        ~finally:(fun () -> List.iter Device.Buffer.deallocate rawbufs)
        (fun () ->
          Search.beam_search
            ~to_program:(fun dev -> to_program ~optimize:false dev (Device.renderer dev))
            ~allow_test_size:(beam_estimate () <> 0)
            k rawbufs ~var_vals beam_width dev))
    device

(* Optimize and lower a kernel AST to a form ready for linearization.
   When [optimize] is true, runs load collapse, range splitting, symbolic
   simplification, range tightening, and dispatches to beam search or
   hand-coded optimizations via Postrange. *)
and full_rewrite_to_sink ?(optimize = true) ?beam_device ren sink =
  let beam = if optimize && not (has_tag sink) then
      (kernel_info_exn "full_rewrite_to_sink" sink).beam else 0 in
  if debug () >= 5 then Format.eprintf "=== ast ===@.%a@." U.pp sink;
  let sink = U.graph_rewrite ~bottom_up:true ~name:"early movement ops"
      Prepare.movement_ops sink in
  let sink =
    if optimize && not (has_tag sink) then
      let sink = Simplify.load_collapse_all sink in
      let sink = Simplify.split_ranges sink in
      let sink =
        U.graph_rewrite ~name:"initial symbolic"
          (U.first_match
             [
               Upat.Pattern_matcher.rewrite sym;
               Simplify.flatten_range;
            ])
          sink
      in
      let sink = Simplify.simplify_ranges sink in
      let beam_search =
        if beam >= 1 then make_beam_search beam_device beam else None
      in
      let hand_coded_optimizations =
        if noopt () = 0 then Some Heuristic.hand_coded_optimizations
        else None
      in
      let sink = Postrange.apply_opts ?beam_search ?hand_coded_optimizations sink ren in
      sink
    else sink
  in
  let sink = Codegen_lower.lower ren sink in
  sink

(* Build an on-graph PROGRAM node for kernel [sink]: optimize + lower, derive
   program metadata, linearize, render, and compile. The result is
   [PROGRAM(SINK, LINEAR, SOURCE, BINARY)] carrying the launch/argument
   metadata as its arg, mirroring the compiled-kernel representation the
   engine dispatches on. *)
and to_program ?(optimize = true) ?beam_device dev ren sink =
  ignore (kernel_info_exn "to_program" sink : U.kernel_info);
  let optimize = optimize && not (has_tag sink) in
  let beam_device = Option.value beam_device ~default:dev in
  let full_sink = full_rewrite_to_sink ~optimize ~beam_device ren sink in
  let ki = kernel_info_exn "to_program" full_sink in
  let program = Linearizer.linearize full_sink in
  let full_sink = List.hd (List.rev program) in
  let src = Renderer.render ren ~name:ki.name program in
  let comp =
    match Renderer.compiler ren with
    | Some c -> c
    | None -> invalid_arg "Codegen.to_program: device renderer has no compiler"
  in
  if debug () >= 3 && ki.applied_opts <> [] then
    Printf.eprintf "%-25s opts: %s\n%!" ki.name
      (String.concat ", " (List.map U.Opt.to_string ki.applied_opts));
  if debug () >= 4 then Printf.eprintf "%s\n%!" src;
  let lib = Compiler.compile_cached comp src in
  let info = U.program_info_from_sink ~target:(Renderer.target ren) full_sink in
  let full_sink =
    match ki.estimates with
    | Some _ -> full_sink
    | None ->
        let estimates =
          Program_spec.Estimates.(to_uop (of_program program))
        in
        U.replace full_sink
          ~arg:(U.Arg.Kernel_info { ki with estimates = Some estimates })
          ()
  in
  U.program ~sink:full_sink ~linear:(U.linear program) ~source:(U.source src)
    ~binary:(U.binary (Bytes.to_string lib)) ~info ()

(* Copy submission needs lowering, so install it at the existing compiler
   boundary rather than introduce another execution path in storage. *)
let () =
  Device.Buffer.install_copy_runner (fun ~dst ~src ->
      Device.Buffer.ensure_allocated dst;
      Device.Buffer.ensure_allocated src;
      let device = Device.get (Device.Buffer.device dst) in
      let call = U.store_call ~dst:(U.from_buffer dst) ~src:(U.from_buffer src) in
      Realize.run_linear ~device
        ~to_program:(fun dev -> to_program dev (Device.renderer dev))
        ~update_stats:false (U.linear [call]))
