(* Stage benchmarks of tolk's compiler. Each program, a graph recorded from
   tinygrad as prepare_rangeify receives it, is timed one stage at a time: a
   case's setup runs the stages before it, and the case runs its stage alone, so
   a regression shows in the stage that has it. The stages, in order: [prepare]
   makes the tensor graph a prepared graph, [kernel_graph] a kernel graph,
   [schedule] a linear of calls, [codegen] each kernel a lowered sink,
   [linearize] each lowered sink instructions, and [render] each kernel's
   program C source. [warm] is what a later process runs instead of them all: it
   reads the schedule and each kernel's program back from the disk cache.

   Kernels are lowered and rendered for the CPU's C renderer, on a fixed
   architecture, so every machine times the same work. Nothing is compiled.

   [search/linearize] times a beam search's lowering of a candidate up to its
   instructions, which every candidate pays, for the kernels of [lorenz] on a
   CUDA renderer. What a search pays per candidate beyond it, and what it costs
   whole, rune's compile suite times through [Rune.jit ~beam].

   [lorenz] runs the kernels of lorenz_simple's step (sofo-raven's tangent step)
   on the host where the hand-coded optimisations lost most to a beam search,
   each compiled with the optimisations the search chose for it ([searched]) and
   with the hand-coded ones ([heuristic]). No public call applies chosen
   optimisations to a kernel, so these run through tolk's engine. *)

open Tolk

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
    (fun k -> Ops.src (Ops.nth (Codegen.linearize k renderer) 1))
    (kernels name)

(* The keys under which [name]'s schedule and programs are put in the disk
   cache's table "bench", for [warm] to read back. *)
let kept name =
  let linear = schedule name in
  let graphs =
    linear :: List.map (fun k -> Codegen.to_program k renderer) (kernels name)
  in
  List.mapi
    (fun i g ->
      let key = Printf.sprintf "%s %d" name i in
      Helpers.Diskcache.put ~table:"bench" key (Graph.to_string g);
      key)
    graphs

let warm keys =
  List.map
    (fun key ->
      Graph.cached ~table:"bench" ~key
        ~valid:(fun _ -> true)
        (fun () -> failwith ("bench: no entry " ^ key)))
    keys

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
      bench ~setup:(fun () -> kept name) "warm" warm;
    ]

(* Lorenz kernels *)

(* An element of a buffer a timed kernel reads: a small float, a small integer.
   Uninitialised memory would time NaN and subnormal arithmetic. *)
let element dt i : Dtype.value =
  if Dtype.is_float dt then `Float (Float.of_int ((i mod 7) - 3) *. 0.25)
  else if Dtype.is_bool dt then `Bool (i mod 2 = 0)
  else `Int (Bigint.of_int (i mod 7))

(* [sink] compiled for the host, and a run of it on buffers of small values
   ({!element}). *)
let compiled sink =
  let d = Nx_device.host in
  let prg = Codegen.to_program sink (Tolk_engine.renderer d) in
  let elf = Device.Tiny_elf.of_program prg in
  let info = match Ops.arg prg with Program i -> i | _ -> assert false in
  let params =
    List.filteri (fun i _ -> i < List.length info.globals) elf.signature
  in
  let scratch (p : Device.Tiny_elf.param) =
    let n = List.fold_left ( * ) 1 p.shape in
    Run.buffer d p.dtype (Array.init n (element p.dtype))
  in
  let p = Tolk_engine.Program.load d prg
  and buffers = List.map scratch params in
  fun () -> Tolk_engine.Program.run p buffers

(* The kernel graph [text] with the optimisations it asks for, or with the
   hand-coded ones when [heuristic]. *)
let lorenz_kernel ~heuristic text =
  let k = Graph.of_string text in
  match Ops.arg k with
  | Ops.Kernel info when heuristic ->
      Ops.replace k ~arg:(Kernel { info with opts_to_apply = None })
  | _ -> k

let lorenz =
  let case text heuristic name =
    Thumper.bench_with_setup
      ~setup:(fun () -> compiled (lorenz_kernel ~heuristic text))
      name
      (fun run -> run ())
  in
  Thumper.group "lorenz"
    (List.map
       (fun (kernel, text) ->
         Thumper.group kernel
           [ case text false "searched"; case text true "heuristic" ])
       Lorenz.all)

(* Searches *)

(* A search's candidates of a lorenz kernel, lowered for a CUDA renderer that
   compiles nothing: the kernel as the search is handed it, with the
   optimisations the search chose dropped, and the first [linearized] of the
   kernels one action makes of it, each optimised as a candidate is. *)
let linearized = 8

let cuda_renderer =
  Renderer.with_compiler
    (Renderer.Compiler.v Fun.id)
    (Cstyle.cuda
       { target with device = "CUDA"; renderer = "CUDA"; arch = "sm_89" })

let candidates text =
  let k = Graph.of_string text in
  let k =
    match Ops.arg k with
    | Ops.Kernel info ->
        Ops.replace k ~arg:(Kernel { info with opts_to_apply = None; beam = 1 })
    | _ -> k
  in
  let handed = ref None in
  let search _ s =
    handed := Some (Postrange.Scheduler.copy s);
    s
  in
  ignore (Codegen.full_rewrite_to_sink ~beam:search k cuda_renderer);
  Search.get_kernel_actions (Option.get !handed)
  |> List.filteri (fun i _ -> i < linearized)
  |> List.map (fun (_, c) ->
      Postrange.Scheduler.get_optimized_ast ~name_override:"test" c)

(* Only the time is measured. What a lowering allocates depends on which nodes
   of earlier calls the collector has not yet reclaimed, since building a node
   that still exists allocates nothing: from one call to the next it moves by
   about 2%, and no count is constant for thumper to prove exact. *)
let linearize_candidates =
  Thumper.group "linearize"
    ~metrics:[ Thumper.Metric.wall_time ]
    (List.map
       (fun (kernel, text) ->
         Thumper.bench_with_setup
           ~setup:(fun () -> candidates text)
           kernel
           (List.map (fun ast -> Codegen.linearize ast cuda_renderer)))
       Lorenz.all)

let suite () =
  List.map (fun (name, _) -> program name) Programs.all
  @ [ lorenz; Thumper.group "search" [ linearize_candidates ] ]

let () =
  match Array.to_list Sys.argv with
  | [ _; "--warm" ] ->
      (* Each case that compiles, once, in as few calls as a trial takes: the
         kernels its setup compiles land in tolk's disk cache, which a
         measurement then reads. *)
      ignore
        (Thumper.measure
           ~config:Thumper.Config.(default |> samples 3 |> warmup 0.)
           ~filter:(`Id "lorenz/") (suite ()))
  | _ ->
      Thumper.run "tolk"
        ~config:Thumper.Config.(default |> deadline 30.)
        ~budgets:
          [
            Thumper.Budget.no_slower_than 0.05;
            Thumper.Budget.no_more_alloc_than 0.01;
          ]
        (suite ())
      |> exit
