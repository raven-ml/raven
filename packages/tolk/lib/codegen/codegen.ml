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
let beam () = Helpers.getenv "BEAM" 0
let beam_estimate () = Helpers.getenv "BEAM_ESTIMATE" 1
let noopt () = Helpers.getenv "NOOPT" 0

(* Allocate raw buffers for beam search from the kernel's Param nodes. *)
let buffer_params ast =
  U.backward_slice ast
  |> List.filter_map (fun u ->
       match U.as_param u, U.dtype u with
       | Some { param; _ }, Dtype.Ptr pty when param.slot >= 0 ->
           Some (param.slot, pty)
       | _ -> None)
  |> List.sort (fun (a, _) (b, _) -> Int.compare a b)

let make_beam_search device beam_width =
  Option.map
    (fun dev k ->
      let rawbufs =
        List.map
          (fun (_, pty) ->
            Device.create_buffer
              ~size:(Dtype.Ptr.size pty)
              ~dtype:(Dtype.Val (Dtype.Ptr.base pty))
              dev)
          (buffer_params (Postrange.ast k))
      in
      Search.beam_search
        ~allow_test_size:(beam_estimate () <> 0)
        k rawbufs beam_width dev)
    device

let beam_width sink =
  match U.as_kernel_info sink with
  | Some { beam = kernel_beam; _ } when kernel_beam >= 1 -> kernel_beam
  | _ -> beam ()

let has_tag u = match U.node_tag u with Some _ -> true | None -> false

let sym = Symbolic.sym

(* Optimize and lower a kernel AST to a form ready for linearization.
   When [optimize] is true, runs load collapse, range splitting, symbolic
   simplification, range tightening, and dispatches to beam search or
   hand-coded optimizations via Postrange. *)
let full_rewrite_to_sink ?(optimize = true) ?beam_device ren sink =
  if debug () >= 5 then Format.eprintf "=== ast ===@.%a@." U.pp sink;
  let sink = Rangeify.rewrite_movement_ops sink in
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
      let beam = beam_width sink in
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

let kernel_info_exn stage sink =
  match U.as_kernel_info sink with
  | Some ki -> ki
  | None ->
      invalid_arg
        (Printf.sprintf "Codegen.%s: Sink is missing KernelInfo" stage)

(* Build an on-graph PROGRAM node for kernel [sink]: optimize + lower, derive
   program metadata, linearize, render, and compile. The result is
   [PROGRAM(SINK, LINEAR, SOURCE, BINARY)] carrying the launch/argument
   metadata as its arg, mirroring the compiled-kernel representation the
   engine dispatches on. *)
let to_program ?(optimize = true) ?beam_device dev ren sink =
  ignore (kernel_info_exn "to_program" sink : U.kernel_info);
  let optimize = optimize && not (has_tag sink) in
  let beam_device = Option.value beam_device ~default:dev in
  let full_sink = full_rewrite_to_sink ~optimize ~beam_device ren sink in
  let ki = kernel_info_exn "to_program" full_sink in
  let program = Linearizer.linearize full_sink in
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
  let aux = Renderer.aux ren program in
  let info = U.program_info_from_sink ~aux full_sink in
  U.program ~sink:full_sink ~linear:(U.linear program) ~source:(U.source src)
    ~binary:(U.binary (Bytes.to_string lib)) ~info ()
