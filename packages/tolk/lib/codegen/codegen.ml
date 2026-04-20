(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Codegen entry point — optimization dispatch + lowering. *)

open Tolk_ir
module K = Kernel

(* Environment *)

let debug = Helpers.getenv "DEBUG" 0
let beam = Helpers.getenv "BEAM" 0
let beam_estimate = Helpers.getenv "BEAM_ESTIMATE" 1
let noopt = Helpers.getenv "NOOPT" 0

(* Allocate raw buffers for beam search from the kernel's Param nodes. *)
let make_beam_search device beam_width =
  Option.map (fun dev k ->
    let rawbufs = List.map (fun p -> match K.view p with
      | Param { dtype = pty; _ } ->
          Device.create_buffer ~size:(Dtype.Ptr.size pty)
            ~dtype:(Dtype.Val (Dtype.Ptr.base pty)) dev
      | _ -> assert false)
      (Postrange.bufs_from_ast (Postrange.ast k)) in
    Search.beam_search ~allow_test_size:(beam_estimate <> 0)
      k rawbufs beam_width dev)
    device

(* Optimize and lower a kernel AST to a form ready for linearization.
   When [optimize] is true, runs load collapse, range splitting,
   symbolic simplification, range tightening, and dispatches to
   beam search or hand-coded optimizations via Postrange. *)
let full_rewrite_to_sink ?(optimize = true) ?device ren sink =
  let sink =
    if optimize then begin
      let sink = Simplify.pm_load_collapse sink in
      let sink = Simplify.pm_split_ranges sink in
      let sink = K.graph_rewrite ~name:"initial symbolic"
        (K.first_match [Symbolic.sym; Simplify.flatten_range]) sink in
      let sink = Simplify.pm_simplify_ranges sink in
      let beam_search =
        if beam >= 1 then make_beam_search device beam else None in
      let hand_coded_optimizations =
        if noopt = 0 then Some Heuristic.hand_coded_optimizations
        else None in
      Postrange.apply_opts ?beam_search ?hand_coded_optimizations sink ren
    end else sink
  in
  Codegen_lower.lower ren sink

(* Full pipeline: optimize + lower + linearize + render + compile. *)
let get_program ?(optimize = true) ?device dev ren sink =
  let sink = full_rewrite_to_sink ~optimize ?device ren sink in
  let ki = match K.view sink with
    | Sink { kernel_info = Some ki; _ } -> ki
    | _ -> { K.name = "kernel"; axis_kinds = []; dont_use_locals = false;
             applied_opts = []; opts_to_apply = None; estimates = None } in
  let program = Linearizer.linearize sink in
  let estimates = match ki.estimates with
    | Some e -> Program_spec.Estimates.of_kernel e
    | None -> Program_spec.Estimates.of_program program in
  let compiled =
    Device.compile_program dev ~name:ki.name ~applied_opts:ki.applied_opts
      ~estimates program in
  if debug >= 3 && ki.applied_opts <> [] then
    Printf.eprintf "%s\n%!"
      (String.concat ", " (List.map K.Opt.to_string ki.applied_opts));
  if debug >= 4 then
    Printf.eprintf "%s\n%!" (Program_spec.src compiled);
  compiled
