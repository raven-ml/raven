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
   CUDA renderer, and [search/actions] its making of the candidates one action
   makes of a kernel, which a round pays for each kernel of its beam; without
   NVRTC, which the renderer's compiler needs, they are left out.
   [search/beam] times a whole search on the host, its candidates compiled,
   over rounds a timing of their optimisations alone fixes.

   [lorenz] runs the kernels of lorenz_simple's step (sofo-raven's tangent step)
   on the host where the hand-coded optimisations lost most to a beam search,
   each compiled with the optimisations the search chose for it ([searched]) and
   with the hand-coded ones ([heuristic]). No public call applies chosen
   optimisations to a kernel, so these run through tolk's engine.

   [outputs] builds and schedules a program of 2000 outputs of one element,
   each reading an earlier one: its cost is linear in the outputs.

   [host] runs, on the host, kernels whose lanes the host's register budget and
   vector width decide, each with the hand-coded optimisations.

   [gpt_oss] runs, on the first CUDA GPU, kernels of gpt-oss-20b's step at
   bfloat16 as rune compiles them, with the hand-coded optimisations: a decode
   step's attention values, 64 by 64 outputs summed over 1024 cache slots, and
   a prefill's router, 512 tokens by 32 experts summed over 2880. Each takes a
   tensor core that leaves few workgroups. Without a CUDA GPU the group is
   empty.

   [contract] runs, on the first CUDA GPU, dense contractions as rune compiles
   them, with the hand-coded optimisations, each summed in float32: products
   of bfloat16 on tensor cores, and of float32 in registers. Without a CUDA GPU
   the group is empty.

   [gated] runs, on the first CUDA GPU, a 2048-cubed bfloat16 product stored
   only where its block of 128 rows is flagged, with 5 of its 16 blocks off and
   with every block on: a block flagged off skips its loop. Without a CUDA GPU
   the group is empty. *)

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
  let optimize =
    match Ops.arg k with Kernel { optimized; _ } -> not optimized | _ -> true
  in
  Codegen.full_rewrite_to_sink ~optimize k renderer

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
  let bench = Thumper.bench_with_setup in
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

(* An element of a buffer a timed kernel reads, at an index of a buffer of a
   length: a small float, a small integer. Uninitialised memory would time NaN
   and subnormal arithmetic. *)
let element dt (_ : int) i : Dtype.value =
  if Dtype.is_float dt then `Float (Float.of_int ((i mod 7) - 3) *. 0.25)
  else if Dtype.is_bool dt then `Bool (i mod 2 = 0)
  else `Int (Bigint.of_int (i mod 7))

(* The buffer of [d] of [n] values of [dt] that [element] gives, from the
   type, the length and the index. *)
let values element d dt n = Run.buffer d dt (Array.init n (element dt n))

(* [sink] compiled for [d], and a run of it on buffers [scratch d dt n] makes
   for each parameter of [n] values of [dt], which returns once the run's
   writes are done. *)
let compiled ~scratch d sink =
  let device = (Tolk_engine.target d).device in
  let prg = Codegen.to_program sink (Tolk_engine.renderer d) in
  let params =
    List.filter
      (fun u -> Ops.addrspace u <> Some Dtype.Alu)
      (Shape.kernel_params prg)
  in
  let scratch u = Some (scratch d (Ops.dtype u) (Shape.max_numel u)) in
  let arg u =
    let slot = match Ops.arg u with Param p -> p.slot | _ -> assert false in
    Shape.param
      ~shape:[ Int (Shape.max_numel u) ]
      ~device slot (Ops.dtype u)
  in
  let s =
    Tolk_engine.link_call
      ~devices:(fun _ -> d)
      (Ops.call prg (List.map arg params))
  and buffers = Array.of_list (List.map scratch params) in
  fun () ->
    Tolk_engine.run s buffers;
    for i = 0 to Array.length buffers - 1 do
      match buffers.(i) with Some b -> Rig.Buffer.wait b Read | None -> ()
    done

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
        compiled ~scratch:(values element) Rig.host (lorenz_kernel ~heuristic text))
      name
      (fun run -> run ())
  in
  Thumper.group "lorenz"
    (List.map
       (fun (kernel, text) ->
         Thumper.group kernel
           [ case text false "searched"; case text true "heuristic" ])
       Lorenz.all)

(* Host kernels *)

let host =
  Thumper.group "host"
    (List.map
       (fun (kernel, text) ->
         Thumper.group kernel
           [
             Thumper.bench_with_setup
               ~setup:(fun () ->
                 compiled ~scratch:(values element) Rig.host
                   (lorenz_kernel ~heuristic:true text))
               "heuristic"
               (fun run -> run ());
           ])
       Host_kernels.all)

(* gpt-oss kernels *)

(* An element of a buffer a gpt-oss kernel reads: a small float, the position
   of a decode step in the middle of a cache of 1024 slots for a one-element
   integer buffer, and the slot itself for the table of a cache's slots, whose
   reads the position masks past it. *)
let position = 512

let cache_element dt n i : Dtype.value =
  if Dtype.is_int dt then `Int (Bigint.of_int (if n = 1 then position else i))
  else element dt n i

(* Whether CUDA's first GPU opens, asked of a fresh process: a GPU's driver must
   not be initialized before the fork that isolates a case. *)
let cuda_opens () =
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe [| exe; "--cuda" |] Unix.stdin Unix.stdout
      Unix.stderr
  in
  match Unix.waitpid [] pid with _, Unix.WEXITED 0 -> true | _ -> false

let cuda () =
  match Rig.open_ (Rig_cuda.gpu 0) with Ok d -> d | Error e -> failwith e

(* Only the time is measured: a GPU buffer's release allocates when the
   collector finalises it, so a call's count depends on the collection's
   phase. *)
let gpt_oss () =
  let case (kernel, text) =
    Thumper.bench_with_setup
      ~setup:(fun () ->
        compiled ~scratch:(values cache_element) (cuda ()) (Graph.of_string text))
      kernel
      (fun run -> run ())
  in
  Thumper.group "gpt_oss"
    ~metrics:[ Thumper.Metric.wall_time ]
    (if cuda_opens () then List.map case Gpt_oss_kernels.all else [])

(* Streamed contractions *)

(* A large buffer is filled with one byte, a small normal number in every
   float type and a code of every quantised format: filling hundreds of
   megabytes element by element would take longer than the run. An integer
   buffer, an index or a position, is small and takes [cache_element]'s
   values. *)
let large = 1 lsl 20

let filled d dt n =
  if n < large then Run.buffer d dt (Array.init n (cache_element dt n))
  else
    let b = Rig.Buffer.create d (n * Dtype.itemsize dt) in
    Rig.Buffer.fill b '\x3e';
    b

(* The rows of a group of recorded kernels: each a call's kernels, recorded as
   [row.i], run in the order of [i]. *)
let rows_of all =
  let row (name, _) = String.sub name 0 (String.index name '.') in
  let kernels = List.sort compare all in
  List.map
    (fun r -> (r, List.map snd (List.filter (fun k -> row k = r) kernels)))
    (List.sort_uniq String.compare (List.map row kernels))

let gpu_rows name all =
  let case (row, kernels) =
    Thumper.bench_with_setup
      ~setup:(fun () ->
        let d = cuda () in
        List.map
          (fun text -> compiled ~scratch:filled d (Graph.of_string text))
          kernels)
      row
      (fun runs -> List.iter (fun run -> run ()) runs)
  in
  Thumper.group name
    ~metrics:[ Thumper.Metric.wall_time ]
    (if cuda_opens () then List.map case (rows_of all) else [])

let stream () = gpu_rows "stream" Stream_kernels.all

(* Contractions on tensor cores and in registers *)

let contract () = gpu_rows "contract" Contract_kernels.all

(* Gated contractions *)

(* [out[i, j] = Σ_k a[i, k] b[k, j]] of [n]-square bfloat16 operands summed in
   float32, stored only where [i]'s block of 128 rows is flagged, as a routed
   product's blocks of padding hold no value: a block a staged core's workgroup
   takes whole. *)
let gated_product n =
  let open Ops.O in
  let param slot size dt = Shape.param ~shape:[ Int size ] slot dt in
  let out = param 0 Stdlib.(n * n) Float32
  and a = param 1 Stdlib.(n * n) Bfloat16
  and b = param 2 Stdlib.(n * n) Bfloat16
  and fl = param 3 Stdlib.(n / 128) Int32 in
  let i = Ops.range (Int n) [ 1 ] and j = Ops.range (Int n) [ 2 ] in
  let k = Ops.range (Int n) [ 0 ] in
  let ld x at = Ops.cast (Ops.index x [ at ]) Float32 in
  let sum =
    Ops.reduce (ld a ((i * int n) + k) * ld b ((k * int n) + j)) Add [ k ]
  in
  let flag = Ops.index fl [ i // int 128 ] in
  let v =
    Ops.where
      (flag <> Ops.int ~dtype:Int32 0)
      sum
      (Shape.const_like sum `Invalid)
  in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (Ops.index out [ (i * int n) + j ]) v) [ i; j ] ]

(* The gated product of 2048 with [off] of its 16 blocks flagged off, spread
   over the rows, with the hand-coded optimisations. *)
let gated () =
  let n = 2048 in
  let blocks = n / 128 in
  let flags off =
    Array.init blocks (fun b ->
        `Int (Bigint.of_int (if b * off mod blocks < off then 0 else 1)))
  in
  let case (row, off) =
    Thumper.bench_with_setup
      ~setup:(fun () ->
        let scratch d dt size =
          if Dtype.equal dt Int32 then Run.buffer d dt (flags off)
          else filled d dt size
        in
        compiled ~scratch (cuda ()) (gated_product n))
      row
      (fun run -> run ())
  in
  Thumper.group "gated"
    ~metrics:[ Thumper.Metric.wall_time ]
    (if cuda_opens () then
       List.map case
         [ ("bf16-2048-5-of-16-blocks-off", 5); ("bf16-2048-all-on", 0) ]
     else [])

(* Searches *)

(* A search's candidates of a lorenz kernel, lowered for a CUDA renderer that
   compiles nothing: the kernel as the search is handed it, with the
   optimisations the search chose dropped, and the first [linearized] of the
   kernels one action makes of it, each optimised as a candidate is. *)
let linearized = 8

(* The renderer is made with NVRTC's compiler, which needs NVRTC: the search
   rows run where it is found, as the gpt_oss rows run where a CUDA GPU
   opens. *)
let cuda_renderer =
  match
    Cstyle.cuda
      { target with device = "CUDA"; renderer = "CUDA"; arch = "sm_89" }
  with
  | r -> Some (Renderer.with_compiler (Renderer.Compiler.v Fun.id) r)
  | exception Failure _ -> None

(* The scheduler of a lorenz kernel that a search is handed. A kernel that
   reaches no search never calls the hook, and fails here by name. *)
let handed cuda_renderer kernel text =
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
  match !handed with
  | Some s -> s
  | None -> failwith ("bench: no search is handed " ^ kernel)

let candidates cuda_renderer kernel text =
  Search.get_kernel_actions (handed cuda_renderer kernel text)
  |> List.filteri (fun i _ -> i < linearized)
  |> List.map (fun (_, c) ->
      Postrange.Scheduler.get_optimized_ast ~name_override:"test" c)

(* Only the time is measured. What a lowering allocates depends on which nodes
   of earlier calls the collector has not yet reclaimed, since building a node
   that still exists allocates nothing: from one call to the next it moves by
   about 2%, and no count is constant for thumper to prove exact. *)
let linearize_candidates cuda_renderer =
  Thumper.group "linearize"
    ~metrics:[ Thumper.Metric.wall_time ]
    (List.map
       (fun (kernel, text) ->
         Thumper.bench_with_setup
           ~setup:(fun () -> candidates cuda_renderer kernel text)
           kernel
           (List.map (fun ast -> Codegen.linearize ast cuda_renderer)))
       Lorenz.all)

let kernel_actions cuda_renderer =
  Thumper.group "actions"
    ~metrics:[ Thumper.Metric.wall_time ]
    (List.map
       (fun (kernel, text) ->
         Thumper.bench_with_setup
           ~setup:(fun () -> handed cuda_renderer kernel text)
           kernel
           (fun s -> Search.get_kernel_actions s))
       Lorenz.all)

(* A whole search of width 1 of lorenz's r_128_16_3_4_4_3_3_400_400 on the
   host, its candidates compiled by Clang with no compile cache, and timed by
   a function of their optimisations alone, so that every run compiles the
   same candidates over the same rounds, whose last ones drop candidates past
   BEAM_UOPS_MAX. *)
let host_renderer = Cstyle.clang target

let applied prg =
  match Ops.arg (Ops.nth prg 0) with
  | Ops.Kernel info -> info.applied_opts
  | _ -> []

(* A program of more optimisations is faster, ties broken by a hash of them:
   each round goes on to a kernel of one more, until a round's candidates are
   all dropped. *)
let opts_time ~vars:_ prg =
  let opts = applied prg in
  let tie = Float.of_int (Hashtbl.hash opts mod 1000) /. 1e4 in
  1e-3 /. (Float.of_int (1 + List.length opts) +. tie)

let searched = "r_128_16_3_4_4_3_3_400_400"

let searches =
  Thumper.group "beam"
    ~metrics:[ Thumper.Metric.wall_time ]
    [
      Thumper.bench_with_setup
        ~setup:(fun () ->
          handed host_renderer searched (List.assoc searched Lorenz.all))
        searched
        (fun s ->
          Setting.context
            [ B (Setting.ccache, false); B (Setting.ignore_beam_cache, true) ]
            (fun () ->
              Search.beam_search ~link:Ops.body ~time:opts_time 1
                (Postrange.Scheduler.copy s)));
    ]

(* [outputs] builds and schedules a program of many outputs of one element in
   two chains, each output one more than the output two before it, as a chain
   of scalars makes: the whole of [Schedule.create_linear_with_vars], over
   storage and nodes no earlier call made. *)
let chained n =
  let element () = Shape.reshape (Ops.new_buffer "CPU" 1 Float32) [] in
  let rec go j prev prev' acc =
    if j = n then Ops.sink (List.rev acc)
    else
      let o = element () in
      let o = Ops.after o [ Ops.store o Ops.O.(prev + float 1.) ] in
      go (j + 1) prev' o (o :: acc)
  in
  let x = element () in
  go 0 x x []

let outputs =
  Thumper.group "outputs"
    [
      Thumper.bench "chain-2000" (fun () ->
          Schedule.create_linear_with_vars ~capturing:true (chained 2000));
    ]

let suite () =
  List.map (fun (name, _) -> program name) Programs.all
  @ [
      outputs;
      lorenz;
      host;
      gpt_oss ();
      stream ();
      contract ();
      gated ();
      Thumper.group "search"
        (searches
        ::
        (match cuda_renderer with
        | Some r -> [ linearize_candidates r; kernel_actions r ]
        | None -> []));
    ]

let () =
  match Array.to_list Sys.argv with
  | [ _; "--cuda" ] ->
      exit (if Result.is_ok (Rig.open_ (Rig_cuda.gpu 0)) then 0 else 1)
  | [ _; "--warm" ] ->
      (* Each case that compiles, once, in as few calls as a trial takes: the
         kernels its setup compiles land in tolk's disk cache, which a
         measurement then reads. *)
      ignore
        (Thumper.measure
           ~config:Thumper.Config.(default |> samples 3 |> warmup 0.)
           ~filter:
             (`Or
               [
                 `Id "lorenz/";
                 `Id "host/";
                 `Id "gpt_oss/";
                 `Id "stream/";
                 `Id "gated/";
               ])
           (suite ()))
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
