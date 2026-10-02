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

   [decode] runs kernels: gpt-oss-20b's decode products, recorded by the
   Heuristic suite, compiled with the hand-coded optimisations for the host and
   for a CUDA device, and timed one run at a time, synchronized. [qkv] is the
   bfloat16 projection of a normalised activation; [gate_up] and [down] are
   four experts' MXFP4 products, [down] summing them. The CUDA device is opened
   in the measuring worker, which is forked without an exec, and CUDA's driver
   must not be initialized before the fork: a fresh process of this executable
   ([--cuda]) says whether a CUDA device opens.

   [lorenz] runs lorenz_simple's two costliest kernels on the host (sofo-raven's
   tangent step), each compiled with the optimisations a beam search chose for
   it ([searched]) and with the hand-coded ones ([heuristic]). *)

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
    (fun k ->
      let program = Ops.v Op.Program ~src:[ lower k ] in
      Ops.src (Ops.nth (Codegen.to_program program renderer) 1))
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

(* Decode products *)

(* An element of a buffer a timed kernel reads: a small float, a small
   integer, and 127 in a byte, which is 1 as an E8M0 scale and +-6 as two FP4
   codes. Uninitialised memory would time NaN and subnormal arithmetic. *)
let element dt i : Dtype.value =
  if Dtype.is_float dt then `Float (Float.of_int ((i mod 7) - 3) *. 0.25)
  else if Dtype.is_bool dt then `Bool (i mod 2 = 0)
  else if Dtype.equal dt Uint8 then `Int (Bigint.of_int 127)
  else `Int (Bigint.of_int (i mod 7))

(* [sink] compiled for [d], named [name], and a run of it on buffers of small
   values ({!element}) that waits for the device. *)
let compiled name d sink =
  let devices = Tolk_engine.device [ (name, d) ] in
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
  match (devices name).compiler.queues with
  | None ->
      let p = Tolk_engine.Program.load d prg and buffers = List.map scratch params in
      fun () -> Tolk_engine.Program.run p buffers
  | Some _ ->
      let slots = 1 + List.fold_left max 0 info.globals in
      let param slot =
        List.nth params
          (Option.get (List.find_index (Int.equal slot) info.globals))
      in
      let call =
        Ops.call prg
          (List.init slots (fun slot ->
               let p = param slot in
               Ops.param
                 ~shape:(List.map (fun n -> Ops.Int n) p.shape)
                 ~device:(Single name) slot p.dtype))
      in
      let linear =
        Hcq2.compile_linear
          ~devices:(fun n -> (devices n).compiler)
          (Ops.v Op.Linear ~src:[ call ])
      in
      let s = Tolk_engine.link ~devices linear in
      let buffers = Array.init slots (fun slot -> [ scratch (param slot) ]) in
      fun () ->
        Tolk_engine.run s buffers;
        Nx_device.synchronize d

let decode name open_device =
  Thumper.group ~id:name name
    (List.map
       (fun (kernel, text) ->
         Thumper.bench_with_setup
           ~setup:(fun () ->
             compiled (String.uppercase_ascii name) (open_device ())
               (Graph.of_string text))
           kernel
           (fun run -> run ()))
       Kernels.all)

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
      ~setup:(fun () ->
        compiled "CPU" Nx_device.host (lorenz_kernel ~heuristic text))
      name
      (fun run -> run ())
  in
  Thumper.group "lorenz"
    (List.map
       (fun (kernel, text) ->
         Thumper.group kernel
           [ case text false "searched"; case text true "heuristic" ])
       Lorenz.all)

let run_self flag =
  Sys.command (Filename.quote_command Sys.executable_name [ flag ])

let cuda () =
  if run_self "--cuda" <> 0 then []
  else
    [
      decode "cuda" (fun () ->
          match Nx_cuda_device.get 0 with Ok d -> d | Error e -> failwith e);
    ]

let () =
  (match Array.to_list Sys.argv with
  | [ _; "--cuda" ] ->
      exit (if Result.is_ok (Nx_cuda_device.get 0) then 0 else 1)
  | _ -> ());
  Thumper.run "tolk"
    ~budgets:
      [
        Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
        Thumper.Budget.no_more_alloc_than 0.01;
      ]
    (List.map (fun (name, _) -> program name) Programs.all
    @ (decode "cpu" (fun () -> Nx_device.host) :: lorenz :: cuda ()))
