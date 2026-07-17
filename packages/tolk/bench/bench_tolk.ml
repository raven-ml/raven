(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tolk compile-pipeline microbenchmarks. Each workload graph (see
   {!Tolk_bench_graphs}) is timed one pipeline stage at a time: the stage input
   is built once in [setup] and the timed closure runs the single pass, so a
   regression or a superlinear cost localizes to one stage.

   Stages, in pipeline order:
     rangeify   tensor SINK      -> kernel graph
     schedule   kernel graph     -> planned LINEAR
     codegen    per-kernel AST   -> lowered sink
     linearize  lowered sink     -> program
     render     program          -> backend source

   The CPU (clang) renderer is used throughout — deterministic, present on
   every machine, and the exact renderer the parity "cpu" goldens bless
   through. Stage 7 (device compile) is out of scope: it shells out to the
   toolchain and does not belong in the tight lab gate.

   Caches that would serve a repeated pass are disabled by running with
   [SCACHE=0] (read at module init in the schedule engine); the timed stages
   themselves call the seam functions directly and hold no cross-call cache. *)

open Tolk
module U = Tolk_uop.Uop
module Graphs = Tolk_bench_graphs.Graphs

let ren = Cstyle.clang_no_abi Gpu_target.X86_64
let optimize = true
let keep x = ignore (Sys.opaque_identity x)

let rangeify_of w = Rangeify.get_kernel_graph (Graphs.sink w)
let kernels_of w = Graphs.kernels (rangeify_of w)

let codegen_of w =
  List.map (fun k -> Codegen.full_rewrite_to_sink ~optimize ren k) (kernels_of w)

let programs_of w =
  List.map
    (fun k ->
      let processed = Codegen.full_rewrite_to_sink ~optimize ren k in
      let name =
        match U.as_kernel_info processed with
        | Some ki -> ki.name
        | None -> "kernel"
      in
      (name, Linearizer.linearize processed))
    (kernels_of w)

(* One group per workload; the five stage cases give full paths of the form
   [<workload>/<stage>]. The lab tag is set per case: it drives the [--tag lab]
   gate filter, which group-level tags do not. *)
let workload_benches w =
  let bench ~setup name f =
    Thumper.bench_with_setup ~tags:[ "lab" ] ~setup name f
  in
  Thumper.group (Graphs.name w)
    [
      bench ~setup:(fun () -> Graphs.sink w) "rangeify" (fun sink ->
          keep (Rangeify.get_kernel_graph sink));
      bench ~setup:(fun () -> rangeify_of w) "schedule" (fun kg ->
          let linear = Schedule.create_schedule kg in
          keep (Schedule.memory_plan_rewrite linear []));
      bench ~setup:(fun () -> kernels_of w) "codegen" (fun ks ->
          List.iter
            (fun k -> keep (Codegen.full_rewrite_to_sink ~optimize ren k))
            ks);
      bench ~setup:(fun () -> codegen_of w) "linearize" (fun processed ->
          List.iter (fun p -> keep (Linearizer.linearize p)) processed);
      bench ~setup:(fun () -> programs_of w) "render" (fun programs ->
          List.iter
            (fun (name, prog) -> keep (Renderer.render ren ~name prog))
            programs);
    ]

let () =
  Thumper.run "tolk"
    ~budgets:
      [
        Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
        Thumper.Budget.no_more_alloc_than 0.01;
      ]
    (List.map workload_benches Graphs.all)
