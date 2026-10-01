(* Stage benchmarks of tolk's compiler. Each program, a graph recorded from
   tinygrad as prepare_rangeify receives it, is timed one stage at a time: a
   case's setup runs the stages before it, and the case runs its stage alone, so
   a regression shows in the stage that has it. The stages, in order:

   prepare tensor graph -> prepared graph kernel_graph prepared graph -> kernel
   graph schedule kernel graph -> linear of calls codegen each kernel -> lowered
   sink linearize each lowered sink -> instructions render each kernel's program
   -> C source

   Kernels are lowered and rendered for the CPU's C renderer, on a fixed
   architecture, so every machine times the same work. Nothing is compiled. *)

open Tolk_next

let target =
  {
    Helpers.Target.device = "CPU";
    renderer = "CLANG";
    arch = "x86_64,x86-64";
    interface = "";
    indices = "";
  }

let renderer =
  Renderer.with_compiler (Renderer.Compiler.v Fun.id) (Cstyle.clang target)

let graph name = Graph.of_string (List.assoc name Programs.all)
let prepared name = Prepare.prepare_rangeify (graph name)
let kernel_graph name = Rangeify.get_kernel_graph (prepared name)
let schedule name = Schedule.create_schedule (kernel_graph name)

(* The kernels of a schedule: the bodies of its calls, in loops or not, that are
   kernels rather than copies. *)
let kernels name =
  List.filter_map
    (fun entry ->
      let call = if Ops.op entry = End then Ops.nth entry 0 else entry in
      let body = Ops.body call in
      match Ops.arg body with Kernel _ -> Some body | _ -> None)
    (Ops.src (schedule name))

let lower k =
  Codegen.full_rewrite_to_sink ~optimize:(Option.is_none (Ops.tag k)) k renderer

let linearize sink =
  Codegen.line_rewrite
    (Linearizer.linearize sink)
    Codegen.pm_linearize_cleanups ()

(* The instructions each kernel's program renders. *)
let instructions name =
  List.map
    (fun k ->
      let program = Ops.v Op.Program ~src:[ lower k ] in
      Ops.src (Ops.nth (Codegen.to_program program renderer) 1))
    (kernels name)

let program name =
  let bench = Thumper.bench_with_setup ~tags:[ "lab" ] in
  Thumper.group name
    [
      bench ~setup:(fun () -> graph name) "prepare" Prepare.prepare_rangeify;
      bench
        ~setup:(fun () -> prepared name)
        "kernel_graph" Rangeify.get_kernel_graph;
      bench
        ~setup:(fun () -> kernel_graph name)
        "schedule" Schedule.create_schedule;
      bench ~setup:(fun () -> kernels name) "codegen" (List.map lower);
      bench
        ~setup:(fun () -> List.map lower (kernels name))
        "linearize" (List.map linearize);
      bench
        ~setup:(fun () -> instructions name)
        "render" (List.map renderer.render);
    ]

let () =
  Thumper.run "tolk"
    ~budgets:
      [
        Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
        Thumper.Budget.no_more_alloc_than 0.01;
      ]
    (List.map (fun (name, _) -> program name) Programs.all)
