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

(* Device renderer + host clang compiler for the stage-7 compile group. The
   render-parity renderer above emits device source that is not a compilable
   host translation unit; the CPU device's own renderer is the real compile
   path. *)
let device_ren = Device.renderer (Tolk_cpu.create "CPU:bench_tolk")

let clang_compiler =
  match Renderer.compiler device_ren with
  | Some c -> c
  | None -> failwith "CPU device renderer has no compiler"

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

let device_srcs_of w =
  List.map
    (fun k ->
      let processed = Codegen.full_rewrite_to_sink ~optimize device_ren k in
      let name =
        match U.as_kernel_info processed with
        | Some ki -> ki.name
        | None -> "kernel"
      in
      Renderer.render device_ren ~name (Linearizer.linearize processed))
    (kernels_of w)

(* One group per workload; the stage cases give full paths of the form
   [<workload>/<stage>]. The lab tag is set per case: it drives the [--tag lab]
   gate filter, which group-level tags do not. The graph-transform stages are
   tagged [lab], except codegen on the multi-kernel [attention] workload, which
   is the single priciest pass in the suite and would strain the tight gate;
   there it is untagged (run on demand), mirroring how codegen stays untagged
   on the scaling ladder. The stage-7 [compile] case shells out to clang, is
   wall-time-only, and stays untagged so the tight gate skips it — it is run on
   demand (with [CCACHE=0] to defeat the compiler's disk cache). *)
let workload_benches w =
  let bench ?(tags = [ "lab" ]) ~setup name f =
    Thumper.bench_with_setup ~tags ~setup name f
  in
  let codegen_tags = if Graphs.name w = "attention" then [] else [ "lab" ] in
  Thumper.group (Graphs.name w)
    [
      bench ~setup:(fun () -> Graphs.sink w) "rangeify" (fun sink ->
          keep (Rangeify.get_kernel_graph sink));
      bench ~setup:(fun () -> rangeify_of w) "schedule" (fun kg ->
          let linear = Schedule.create_schedule kg in
          keep (Schedule.memory_plan_rewrite linear []));
      bench ~tags:codegen_tags ~setup:(fun () -> kernels_of w) "codegen"
        (fun ks ->
          List.iter
            (fun k -> keep (Codegen.full_rewrite_to_sink ~optimize ren k))
            ks);
      bench ~setup:(fun () -> codegen_of w) "linearize" (fun processed ->
          List.iter (fun p -> keep (Linearizer.linearize p)) processed);
      bench ~setup:(fun () -> programs_of w) "render" (fun programs ->
          List.iter
            (fun (name, prog) -> keep (Renderer.render ren ~name prog))
            programs);
      Thumper.bench_with_setup ~metrics:[ Thumper.Metric.wall_time ]
        ~setup:(fun () -> device_srcs_of w) "compile" (fun srcs ->
          List.iter
            (fun s -> keep (Compiler.compile_cached clang_compiler s))
            srcs);
    ]

(* Scaling gate over each headline workload's size ladder. [alloc_words] across
   sizes is the superlinearity detector — an O(n^2) pass shows up as super-linear
   allocation growth before wall-time noise matters — and rangeify is where that
   risk lives, so it is the tagged tripwire, across three ladder points per
   workload. schedule is microsecond-scale (gated by the fixed workloads) and
   codegen is the priciest pass; both stay present but untagged so the tight gate
   holds near the ~2 min budget. Every stage/size is in the suite for on-demand
   runs; the tags only pick the tight subset. *)
let rangeify_lab_max = function "lorenz" -> 50 | _ -> 10

(* Trailing integer of a size descriptor ("n50" -> 50, "h10" -> 10). *)
let size_int s =
  let i = ref (String.length s) in
  while !i > 0 && s.[!i - 1] >= '0' && s.[!i - 1] <= '9' do
    decr i
  done;
  int_of_string (String.sub s !i (String.length s - !i))

let scaling_group w =
  let bench ~lab ~setup name f =
    Thumper.bench_with_setup ~tags:(if lab then [ "lab" ] else []) ~setup name f
  in
  let n = size_int (Graphs.size w) in
  let name = Graphs.name w in
  Thumper.group
    (Printf.sprintf "%s/%s" name (Graphs.size w))
    [
      bench ~lab:(n <= rangeify_lab_max name)
        ~setup:(fun () -> Graphs.sink w) "rangeify" (fun sink ->
          keep (Rangeify.get_kernel_graph sink));
      bench ~lab:false ~setup:(fun () -> rangeify_of w) "schedule" (fun kg ->
          let linear = Schedule.create_schedule kg in
          keep (Schedule.memory_plan_rewrite linear []));
      bench ~lab:false ~setup:(fun () -> kernels_of w) "codegen" (fun ks ->
          List.iter
            (fun k -> keep (Codegen.full_rewrite_to_sink ~optimize ren k))
            ks);
    ]

let scaling_benches =
  List.map (fun n -> scaling_group (Graphs.lorenz n)) Graphs.lorenz_ladder
  @ List.map (fun h -> scaling_group (Graphs.rnn h)) Graphs.rnn_ladder

let () =
  Thumper.run "tolk"
    ~budgets:
      [
        Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
        Thumper.Budget.no_more_alloc_than 0.01;
      ]
    (List.map workload_benches Graphs.all @ scaling_benches)
