(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module K = Kernel

let debug_var = Device.Context.int ~name:"DEBUG" ~default:0
let beam_var = Device.Context.int ~name:"BEAM" ~default:0
let beam_estimate = Device.Context.int ~name:"BEAM_ESTIMATE" ~default:1
let noopt_var = Device.Context.int ~name:"NOOPT" ~default:0

let make_beam_search device beam =
  let allow_test_size = Device.Context.get beam_estimate <> 0 in
  Option.map
    (fun dev k ->
      let rawbufs =
        List.map
          (fun p ->
            match K.view p with
            | Param { dtype = pty; _ } ->
                Device.create_buffer ~size:(Dtype.ptr_size pty) ~dtype:(Dtype.base pty) dev
            | _ -> assert false)
          (Postrange.bufs_from_ast (Postrange.ast k))
      in
      Search.beam_search ~allow_test_size k rawbufs beam dev)
    device

(* Main codegen pipeline: applies simplification (load collapse, range
   splitting/flattening, symbolic rewrites, range tightening), then optional
   beam search and hand-coded optimizations via Postrange, and finally lowers
   the rewritten kernel DAG to the target Program IR. *)
let full_rewrite_to_sink ?(optimize = true) ?device ren sink =
  let dbg = Device.Context.get debug_var in
  if dbg >= 5 then K.print_uops ~label:"early movement ops" sink;
  let sink =
    if optimize then begin
      let sink = Simplify.pm_load_collapse sink in
      let sink = Simplify.pm_split_ranges sink in
      let sink = Simplify.pm_flatten_range sink in
      let sink = K.graph_rewrite ~name:"initial symbolic" (K.first_match [ Symbolic.sym ]) sink in
      let sink = Simplify.pm_flatten_range sink in
      let sink = Simplify.pm_simplify_ranges sink in
      let beam = Device.Context.get beam_var in
      let beam_search =
        if beam >= 1 then make_beam_search device beam else None
      in
      let hand_coded_optimizations =
        if Device.Context.get noopt_var = 0 then
          Some Heuristic.hand_coded_optimizations
        else None
      in
      Postrange.apply_opts ?beam_search ?hand_coded_optimizations sink ren
    end
    else sink
  in
  Lowering.lower ren sink

let get_program ?(optimize = true) ?device dev ren sink =
  let dbg = Device.Context.get debug_var in
  let sink = full_rewrite_to_sink ~optimize ?device ren sink in
  let ki = match K.view sink with
    | Sink { kernel_info = Some ki; _ } -> ki
    | _ -> { K.name = "kernel"; axis_kinds = []; dont_use_locals = false;
             applied_opts = []; opts_to_apply = None; estimates = None }
  in
  let program = Linearizer.linearize sink in
  if dbg >= 6 then begin
    Printf.eprintf "=== linearized ===\n%!";
    Format.eprintf "%a@." Program.pp program
  end;
  let estimates = match ki.estimates with
    | Some e -> Program_spec.Estimates.of_kernel e
    | None -> Program_spec.Estimates.of_program program
  in
  let compiled =
    Device.compile_program dev ~name:ki.name ~applied_opts:ki.applied_opts
      ~estimates program
  in
  if dbg >= 3 && ki.applied_opts <> [] then
    Printf.eprintf "%s\n%!"
      (String.concat ", " (List.map K.Opt.to_string ki.applied_opts));
  if dbg >= 4 then
    Printf.eprintf "%s\n%!" (Device.Program.src compiled);
  compiled
