(* Tests of Tolk.Search: the actions, the candidates a kernel makes, and beam
   searches under made-up measurements: what a search links, compiles and times,
   how its rounds decide, and what its cache keeps. *)

open Windtrap
open Tolk
module K = Postrange.Scheduler

(* Kernels and renderers *)

(* No program is compiled: a program's binary is its source's bytes. *)
let renderer_of_row cell =
  Cstyle.without_compiler
    {
      Helpers.Target.device = cell "device";
      renderer = cell "renderer";
      arch = cell "arch";
      interface = "";
      indices = "";
    }

let renderers =
  lazy
    (List.map
       (fun cell -> (cell "target", renderer_of_row cell))
       (Golden.rows "targets.golden"))

let renderer target = List.assoc target (Lazy.force renderers)

let renderer_of target =
  Cstyle.without_compiler (Result.get_ok (Helpers.Target.of_string target))

let kernels = Hashtbl.create 16

let kernel name =
  match Hashtbl.find_opt kernels name with
  | Some k -> k
  | None ->
      let k = Golden.sink (name ^ ".golden") in
      Hashtbl.add kernels name k;
      k

(* A kernel as the search is handed it: its weak output axes made global. *)
let scheduled_ast ast ren =
  let k = K.v ast ren in
  K.convert_loop_to_global k;
  k

let scheduled_for ren name = scheduled_ast (kernel name) ren
let scheduled name target = scheduled_for (renderer target) name

(* Colour off, the cache off and one domain: a search on one domain times its
   candidates in their order, which the goldens record. *)
let quietly ?(settings = []) f =
  Setting.context
    (B (Setting.no_color, true)
    :: B (Setting.cachelevel, 0)
    :: B (Setting.parallel, 0)
    :: settings)
    f

(* The settings searches_environment.golden and actions_padto.golden are
   recorded under. *)
let environment f =
  Setting.context
    [
      B (Setting.beam_padto, true);
      B (Setting.use_tc, 2);
      B (Setting.beam_strict_mode, true);
      B (Setting.beam_uops_max, 50);
    ]
    f

let opts = list Kernel_opts.opt
let pp_opts = Testable.pp opts
let opts_text o = Format.asprintf "%a" pp_opts o

(* [o] on one line. *)
let opts_line o = String.concat "; " (List.map (Format.asprintf "%a" Opt.pp) o)

let bool_of_cell = function
  | "True" -> true
  | "False" -> false
  | c -> failf "no boolean %s" c

let position o =
  match List.find_index (Opt.equal o) (Search.actions ()) with
  | Some i -> i
  | None -> failf "%a is no action" Opt.pp o

let kernel_info prg =
  match Ops.arg (Ops.nth prg 0) with
  | Kernel k -> k
  | _ -> failf "the program %a has no kernel" Ops.pp prg

let applied p = (kernel_info p).applied_opts
let depth p = List.length (applied p)

let binary prg =
  match Ops.arg (Ops.nth prg 3) with
  | Bytes b -> b
  | _ -> failf "the program %a has no binary" Ops.pp prg

let program_info prg =
  match Ops.arg prg with
  | Program p -> p
  | _ -> failf "%a is no program" Ops.pp prg

let instructions prg = List.length (Ops.src (Ops.nth prg 1))

(* A host program that runs its output loop in blocks counts the loop's
   iterations as its first global size; they are no workgroups of its launch. *)
let workgroups ~vars prg =
  let p = program_info prg in
  let n =
    Helpers.prod (List.map (fun s -> Shape.sym_infer s vars) p.global_size)
  in
  match (kernel_info prg).split with
  | Some { iterations; _ } -> n / Shape.sym_infer iterations vars
  | None -> n

(* The product of [k]'s sizes on the axes of [types], each at its bound. *)
let size_of k types =
  List.fold_left
    (fun n a ->
      match List.nth (K.full_shape k) a with
      | Ops.Int m -> n * m
      | Sym u -> n * Dtype.Value.to_int (Ops.vmax u))
    1 (K.axes_of k types)

let lanes k = size_of k [ Upcast ]
let threads k = size_of k [ Warp; Local ]

(* Compilations counted per source, safe from several domains: the compiler
   [counting c] counts a source in [c], then compiles it to itself. *)
type counts = { lock : Mutex.t; table : (string, int) Hashtbl.t }

let counts () = { lock = Mutex.create (); table = Hashtbl.create 64 }

let counting ?cachekey c =
  Renderer.Compiler.v ?cachekey (fun src ->
      Mutex.protect c.lock (fun () ->
          Hashtbl.replace c.table src
            (1 + Option.value ~default:0 (Hashtbl.find_opt c.table src)));
      src)

let compiled c =
  Mutex.protect c.lock (fun () ->
      Hashtbl.fold (fun src n l -> (src, n) :: l) c.table [])

(* A name unique to this attempt of the test and this run, for what a disk cache
   keeps: a name of the test alone would read what an earlier run kept. The
   nonce comes from the system, since windtrap seeds the global generator for
   each test. *)
let unique name =
  Printf.sprintf "%s %d" name
    (Random.State.bits (Random.State.make_self_init ()))

(* Timings

   A timing is the search's [link] and [time], as a pair. *)

type call = { prg : Ops.t; vars : (string * int) list; time : float }

(* What a timing was given: the programs linked and the samples that returned,
   in order. *)
type record = { linked : Ops.t list; calls : call list }

(* The timing whose samples of a program take [sample ~vars prg i], [i] the
   number of samples of it before, and its record. *)
let sampling sample =
  let linked = ref [] and calls = ref [] in
  let link call =
    let prg = Ops.body call in
    linked := prg :: !linked;
    (prg, ref 0)
  in
  let time ~vars (prg, i) =
    let t = sample ~vars prg !i in
    incr i;
    calls := { prg; vars; time = t } :: !calls;
    t
  in
  let record () = { linked = List.rev !linked; calls = List.rev !calls } in
  ((link, time), record)

(* The timing whose samples of a program all take [time ~vars prg], and the
   samples that returned. *)
let recording time =
  let timing, record = sampling (fun ~vars prg _ -> time ~vars prg) in
  (timing, fun () -> (record ()).calls)

(* The timing whose samples take no time, so that a search ends after its first
   round, and the programs it linked. *)
let instant () =
  let timing, record = sampling (fun ~vars:_ _ _ -> 0.) in
  (timing, fun () -> (record ()).linked)

(* The timing whose samples all take a second. *)
let constant = (Fun.id, fun ~vars:_ _ -> 1.)

(* The timing that fails the test when the search links or times. *)
let untimed =
  ( (fun _ -> fail "the search linked a program"),
    fun ~vars:_ _ -> fail "the search timed a program" )

(* The timing that raises Exit at its first link: a search that measures. *)
let measuring = ((fun _ -> raise Exit), fun ~vars:_ _ -> raise Exit)

(* The measurement of searches.golden: an optimum at two optimisations, each
   weighed by its action's position, scaled by the launch's workgroups. When
   [failing], a program whose actions' positions sum to a multiple of 3
   fails. *)
let golden_time ~failing ~vars prg =
  let positions = List.map position (applied prg) in
  if failing && positions <> [] && List.fold_left ( + ) 0 positions mod 3 = 0
  then failwith "failing measurement";
  let tm = 1e-3 *. Float.of_int (1 + (3 * abs (List.length positions - 2))) in
  let weigh tm p = tm *. (0.8 +. (Float.of_int (37 * p mod 41) /. 100.)) in
  let tm = List.fold_left weigh tm positions in
  tm *. (1. +. (Float.of_int (workgroups ~vars prg mod 5) /. 10.))

(* The part of a device that the made-up timings stand for. *)
let part = "made-up"

let search ?settings ?(part = part) ?allow_test_size ~timing:(link, time) amt k
    =
  quietly ?settings (fun () ->
      Search.beam_search ~part ~link ~time ?allow_test_size amt k)

let least = List.fold_left Float.min infinity

(* A search's samples as it takes them: each program's first timing, its samples
   from its first until another program's, in order; and after each round that
   compares two kernels timed again, the beam's first and the fastest, timed in
   turn three times each, as their programs and least samples. *)
type timings = {
  first : (Ops.t * float list) list;
  again : ((Ops.t * float) * (Ops.t * float)) list;
}

let timings calls =
  let same c (p, _) = binary p = binary c.prg in
  (* [open_]: the last sample was of the first timing at the head of [first]. *)
  let step (first, again, open_) c =
    match first with
    | (p, ts) :: older when open_ && same c (p, ts) ->
        ((p, ts @ [ c.time ]) :: older, again, true)
    | _ when List.exists (same c) first -> (first, c :: again, false)
    | _ -> ((c.prg, [ c.time ]) :: first, again, true)
  in
  let first, again, _ = List.fold_left step ([], [], true) calls in
  let rec pairs = function
    | [] -> []
    | a1 :: b1 :: a2 :: b2 :: a3 :: b3 :: rest
      when List.for_all (fun c -> binary c.prg = binary a1.prg) [ a2; a3 ]
           && List.for_all (fun c -> binary c.prg = binary b1.prg) [ b2; b3 ]
           && binary a1.prg <> binary b1.prg ->
        ( (a1.prg, least [ a1.time; a2.time; a3.time ]),
          (b1.prg, least [ b1.time; b2.time; b3.time ]) )
        :: pairs rest
    | c :: _ ->
        failf "%a is timed again, but not in turn with another three times"
          pp_opts (applied c.prg)
  in
  { first = List.rev first; again = pairs (List.rev again) }

(* The first timings of each program linked, in order. *)
let samples_of calls = (timings calls).first

(* How many samples the first timings took: the measurements of a search that
   times nothing again. *)
let first_samples calls =
  List.fold_left (fun n (_, ts) -> n + List.length ts) 0 (samples_of calls)

(* A deterministic measurement drawn from [seed]: a program's time depends on
   its kernel's optimisations alone. *)
let drawn seed opts =
  1e-3
  +. Float.of_int (Hashtbl.seeded_hash seed (List.map position opts) mod 1000)
     *. 1e-6

let drawn_time seed ~vars:_ prg = drawn seed (applied prg)

(* A measurement by the number of optimisations [n]: the [n]th of [times], or
   its last for more. *)
let by_depth times ~vars:_ prg =
  List.nth times (min (depth prg) (List.length times - 1))

(* Samples drawn from [seed]: a program's base time, from a microsecond to two
   milliseconds, times up to one and a half, so that the samples of programs of
   near base times overlap. *)
let spread seed ~vars:_ prg i =
  let positions = List.map position (applied prg) in
  let base = 1 + (Hashtbl.seeded_hash seed positions mod 2000) in
  let noise = 100 + (Hashtbl.seeded_hash seed (positions, i) mod 51) in
  1e-8 *. Float.of_int (base * noise)

(* actions *)

let recorded_actions file =
  List.map (fun cell -> Kernel_opts.opt_of_cell (cell "opt")) (Golden.rows file)

let tc_levels () =
  List.filter_map
    (function Opt.Tc t -> Some t.tc_opt | _ -> None)
    (Search.actions ())

let actions =
  group "actions"
    [
      test "lists the optimisations its golden states, in order" (fun () ->
          equal opts (recorded_actions "actions.golden") (Search.actions ()));
      test
        "under BEAM_PADTO and TC=2, lists the optimisations its golden states, \
         pads included" (fun () ->
          environment (fun () ->
              equal opts
                (recorded_actions "actions_padto.golden")
                (Search.actions ())));
      test "asks for the tensor cores' levels whatever TC_OPT" (fun () ->
          let levels o = Setting.context [ B (Setting.tc_opt, o) ] tc_levels in
          equal (list int) (levels 0) (levels 2));
    ]

(* get_kernel_actions *)

(* The golden numbers a candidate by its action's position from 1, after the
   kernel itself at 0. *)
let candidates =
  Golden.cases ~key:[ "kernel"; "target"; "max_up" ] "candidates.golden"
    (fun cell ->
      let max_up = int_of_string_opt (cell "max_up") in
      let k = scheduled (cell "kernel") (cell "target") in
      let acted = quietly (fun () -> Search.get_kernel_actions ?max_up k) in
      equal (list int)
        (List.map int_of_string (String.split_on_char ' ' (cell "actions")))
        (0 :: List.map (fun (a, _) -> position a + 1) acted))

let small =
  [
    ("add_small", "clang");
    ("sum_rows", "metal");
    ("pad_7x7", "cuda");
    ("symbolic", "hip");
  ]

let name_of (kernel, target) = kernel ^ " on " ^ target

let acted ?settings ?max_up k =
  quietly ?settings (fun () -> Search.get_kernel_actions ?max_up k)

(* The actions of the candidates [acted] whose kernels satisfy [within]. *)
let actions_of ?(within = fun _ -> true) acted =
  List.filter_map (fun (a, k') -> if within k' then Some a else None) acted

let candidate_laws =
  [
    cases "leaves its kernel as it was" ~id:name_of small
      (fun (kernel, target) ->
        let k = scheduled kernel target in
        let ast = K.ast k in
        ignore (acted k);
        equal Uops.uop ast (K.ast k);
        equal opts [] (K.applied_opts k));
    cases "makes each candidate by applying its action to a copy of its kernel"
      ~id:name_of small (fun (kernel, target) ->
        let k = scheduled kernel target in
        let applied_to (a, k') =
          let msg = Format.asprintf "%a" Opt.pp a in
          let copy = K.copy k in
          (match K.apply_opt copy a with
          | Ok _ -> ()
          | Error e -> failf "%s does not apply: %s" msg e);
          equal ~msg opts [ a ] (K.applied_opts k');
          equal ~msg Uops.uop (K.ast copy) (K.ast k')
        in
        List.iter applied_to (acted k));
    cases "leaves out the candidates of more upcast lanes than max_up"
      ~id:name_of small (fun (kernel, target) ->
        let k = scheduled kernel target in
        let all = acted k in
        List.iter
          (fun m ->
            equal
              ~msg:(Printf.sprintf "max_up %d" m)
              opts
              (actions_of ~within:(fun k' -> lanes k' <= m) all)
              (actions_of (acted ~max_up:m k)))
          [ 1; 2; 4; 8 ]);
    cases
      "leaves out the candidates of more warp and local threads than \
       BEAM_LOCAL_MAX"
      ~id:name_of small (fun (kernel, target) ->
        let k = scheduled kernel target in
        let all = acted k in
        List.iter
          (fun m ->
            equal
              ~msg:(Printf.sprintf "BEAM_LOCAL_MAX %d" m)
              opts
              (actions_of ~within:(fun k' -> threads k' <= m) all)
              (actions_of (acted ~settings:[ B (Setting.beam_local_max, m) ] k)))
          [ 1; 4; 16 ]);
  ]

(* The most threads a workgroup holds: a CUDA block's 1024 (the CUDA guide's
   "Maximum number of threads per block") and an Apple threadgroup's 512 before
   family 4 (Metal's feature set tables). *)
let block_targets = [ ("CUDA:CUDA:sm_89", 1024); ("METAL:METAL:Apple3", 512) ]

(* [out[i * 4096 + j] = x[i * 4096 + j]] over 4096 by 4096. *)
let copy_4096 () =
  let i = Ops.range ~axis_type:Weak (Int 4096) [ 0 ]
  and j = Ops.range ~axis_type:Weak (Int 4096) [ 1 ] in
  let at = Ops.O.((i * Ops.int 4096) + j) in
  let x = Shape.param ~shape:[ Int (4096 * 4096) ] 1 Float32
  and out = Shape.param ~shape:[ Int (4096 * 4096) ] 0 Float32 in
  let st =
    Ops.store (Ops.index out [ at ]) (Ops.load (Ops.index x [ at ]) [])
  in
  Ops.sink ~kernel:(Ops.kernel_info ()) [ Ops.end_ st [ i; j ] ]

(* [out[0] = sum x[j]] over 4096 float32. *)
let sum_4096 () =
  let j = Ops.range ~axis_type:Weak (Int 4096) [ 0 ] in
  let x = Shape.param ~shape:[ Int 4096 ] 1 Float32
  and out = Shape.param ~shape:[ Int 1 ] 0 Float32 in
  let r = Ops.reduce (Ops.load (Ops.index x [ j ]) []) Op.Add [ j ] in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.store (Ops.index out [ Ops.int 0 ]) r ]

(* [ast]'s scheduler for [ren], its weak output axes made global and [opts]
   applied, each of the kernel it is given. *)
let opted ren ast opts =
  let k = scheduled_ast ast ren in
  List.iter
    (fun opt ->
      let o = opt k in
      match K.apply_opt k o with
      | Ok _ -> ()
      | Error e -> failf "%a does not apply: %s" Opt.pp o e)
    opts;
  k

let local_split ~reduce n k =
  let axis = List.hd (K.axes_of ~reduce k [ Weak; Global ]) in
  Opt.Split { axis; amount = n; target = Local; top = false }

(* The copy, its first axis split into 256 local threads. *)
let local_256 ren = opted ren (copy_4096 ()) [ local_split ~reduce:false 256 ]

(* The copy in a full workgroup, from 16 threads that a pad to 32 doubles. *)
let full_workgroup ren most =
  opted ren (copy_4096 ())
    [ local_split ~reduce:false 16; local_split ~reduce:false (most / 16) ]

(* The sum grouped by 16 twice: 256 partials. *)
let grouped_sum ren =
  opted ren (sum_4096 ())
    [ local_split ~reduce:true 16; local_split ~reduce:true 16 ]

(* The most partials a float sum of [k] adds: the product of the warp and local
   axes it closes. *)
let partials k =
  let joined r =
    List.fold_left
      (fun n x ->
        match (Ops.axis_type x, Ops.vmax x) with
        | (Warp | Local), `Int m -> n * (Bigint.to_int m + 1)
        | _ -> n)
      1
      (List.tl (Ops.src r))
  in
  List.fold_left (fun m r -> max m (joined r)) 1 (K.reduceops k)

(* Past BEAM_LOCAL_MAX, with pads of local axes among the actions. *)
let unbounded k =
  acted
    ~settings:
      [ B (Setting.beam_padto, true); B (Setting.beam_local_max, 1 lsl 20) ]
    k

let most_of f acted = List.fold_left (fun m (_, k) -> max m (f k)) 0 acted

(* The bounds a candidate keeps are Postrange.Scheduler.apply_opt's: a workgroup
   of at most the renderer's threads, and a float sum of at most Chains.chain
   threads' partials. A search reaches them past BEAM_LOCAL_MAX, and through
   pads of local axes. *)
let candidate_bounds =
  [
    cases
      "a candidate's workgroup holds at most its renderer's threads, past \
       BEAM_LOCAL_MAX"
      ~id:fst block_targets (fun (t, most) ->
        let acted =
          acted
            ~settings:[ B (Setting.beam_local_max, 1 lsl 20) ]
            (local_256 (renderer_of t))
        in
        equal int most (most_of threads acted));
    cases
      "a candidate that pads a local axis holds at most its renderer's threads"
      ~id:fst block_targets (fun (t, most) ->
        equal int most
          (most_of threads (unbounded (full_workgroup (renderer_of t) most))));
    cases "a candidate that pads a local axis adds at most 256 partials" ~id:fst
      block_targets (fun (t, _) ->
        equal int 256
          (most_of partials (unbounded (grouped_sum (renderer_of t)))));
    (let pp ppf picks =
       Format.fprintf ppf "picks %s"
         (String.concat " " (List.map string_of_int picks))
     in
     prop ~count:10
       "every kernel reached through candidates holds its renderer's threads \
        and 256 partials a float sum"
       Gen.(with_pp pp (list ~size:(int_range 1 2) nat))
       (fun picks ->
         let bounded t most (_, k) =
           let msg = Format.asprintf "%s, %a" t pp_opts (K.applied_opts k) in
           at_most ~msg int ~than:most (threads k);
           at_most ~msg int ~than:256 (partials k)
         in
         let rec walk t most k = function
           | [] -> ()
           | pick :: picks -> (
               match unbounded k with
               | [] -> ()
               | acted ->
                   List.iter (bounded t most) acted;
                   let k = snd (List.nth acted (pick mod List.length acted)) in
                   walk t most k picks)
         in
         List.iter
           (fun (t, most) ->
             let ren = renderer_of t in
             walk t most (full_workgroup ren most) picks;
             walk t most (grouped_sum ren) picks)
           block_targets));
  ]

(* The interpreter cannot evaluate a matrix multiply-accumulate, which a tensor
   core used in full makes. *)
let evaluable (_, k) =
  List.for_all
    (function Opt.Tc { use_tc = 1; _ } -> false | _ -> true)
    (K.applied_opts k)

let evaluable_candidates k = List.filter evaluable (acted k)

(* The kernel that the choices [picks] among its candidates make of [k]. *)
let rec picked k = function
  | [] -> k
  | pick :: picks -> (
      match evaluable_candidates k with
      | [] -> k
      | acted ->
          picked (snd (List.nth acted (pick mod List.length acted))) picks)

let writes_of k = Kernel_opts.writes (quietly (fun () -> K.get_optimized_ast k))

(* That optimisations keep a kernel's writes is Postrange.Scheduler's promise;
   these walk it over the search's actions. *)
let candidate_writes =
  [
    (let kernels =
       [ "add_small"; "sum_rows"; "variable_rows"; "symbolic"; "pad_7x7" ]
     in
     let pp ppf (kernel, picks) =
       Format.fprintf ppf "%s, picks %s" kernel
         (String.concat " " (List.map string_of_int picks))
     in
     prop ~count:10
       "every kernel reached through candidates writes what its kernel writes"
       Gen.(
         with_pp pp (pair (of_list kernels) (list ~size:(int_range 1 3) nat)))
       (fun (kernel_name, picks) ->
         List.iter
           (fun (target, _) ->
             equal ~msg:target (list Kernel_opts.write)
               (Kernel_opts.writes (kernel kernel_name))
               (writes_of (picked (scheduled kernel_name target) picks)))
           (Lazy.force renderers)));
    cases "every kernel two candidates make writes what its kernel writes"
      ~id:name_of
      (List.concat_map
         (fun kernel -> [ (kernel, "clang"); (kernel, "metal") ])
         [ "add_small"; "sum_rows"; "symbolic"; "pad_7x7" ])
      (fun (kernel_name, target) ->
        let expected = Kernel_opts.writes (kernel kernel_name) in
        let keeps (_, k) =
          equal
            ~msg:(opts_text (K.applied_opts k))
            (list Kernel_opts.write) expected (writes_of k)
        in
        List.iter
          (fun c -> List.iter keeps (evaluable_candidates (snd c)))
          (evaluable_candidates (scheduled kernel_name target)));
  ]

let candidates_reported =
  test "reports the candidates it leaves out under BEAM_LOG_SURPASS_MAX"
    (fun () ->
      ignore (output ());
      ignore
        (acted
           ~settings:[ B (Setting.beam_log_surpass_max, true) ]
           ~max_up:1
           (scheduled "sum_rows" "metal"));
      not_equal string "" (output ()))

(* [out[i * 16 + u] = u] over 64 [i] and an upcast axis of [n] lanes, [n] in [1,
   16]: whether a candidate has more than 4 lanes is [n]'s to decide. *)
let variable_lanes () =
  let z n = `Int (Bigint.of_int n) in
  let n = Ops.variable "n" (z 1) (z 16) in
  let u = Ops.range ~axis_type:Upcast (Sym n) [ 1 ]
  and i = Ops.range ~axis_type:Weak (Int 64) [ 0 ] in
  let out = Shape.param ~shape:[ Ops.Int (64 * 16) ] 0 Float32 in
  let at = Ops.O.((i * Ops.int 16) + u) in
  scheduled_ast
    (Ops.sink ~kernel:(Ops.kernel_info ())
       [
         Ops.end_
           (Ops.store (Ops.index out [ at ]) (Ops.cast u Float32))
           [ i; u ];
       ])
    (renderer "clang")

let lanes_by_variable =
  test
    "raises Invalid_argument when a variable decides whether a candidate has \
     too many lanes" (fun () ->
      raises_match (Exn.invalid_arg ?sub:None) (fun () ->
          acted ~max_up:4 (variable_lanes ())))

let get_kernel_actions =
  group "get_kernel_actions"
    ([ group "makes the candidates its golden states" [ candidates ] ]
    @ candidate_laws @ candidate_bounds @ candidate_writes
    @ [ candidates_reported; lanes_by_variable ])

(* beam_search links and times its candidates *)

let storage_placed =
  cases "a candidate's call names its renderer's device on its storage" ~id:fst
    [ ("clang", "CPU"); ("metal", "METAL") ]
    (fun (target, device) ->
      let named = ref [] in
      let link call =
        List.iter
          (fun u ->
            match Ops.arg u with
            | Param p when Ops.op u = Op.Param && p.addrspace <> Dtype.Alu ->
                named := p.device :: !named
            | _ -> ())
          (List.tl (Ops.src call))
      in
      ignore
        (search
           ~timing:(link, fun ~vars:_ () -> 0.)
           1
           (scheduled "add_small" target));
      greater int ~than:0 (List.length !named);
      List.iter (equal (option string) (Some device)) !named)

(* [out[r] = n + s] over 4 [r], for the named variable [n] and the positional
   scalar [s] of slot 2, both in [lo, hi]: slot 1 is no parameter's. *)
let scalars (lo, hi) =
  let z n = `Int (Bigint.of_int n) in
  let named = Ops.variable ~dtype:Int32 "n" (z lo) (z hi) in
  let positional = Shape.param ~addrspace:Alu ~vmin_vmax:(z lo, z hi) 2 Int32 in
  let output = Shape.param ~shape:[ Ops.Int 4 ] 0 Float32 in
  let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
  let value = Ops.cast (Ops.add named positional) Float32 in
  scheduled_ast
    (Ops.sink ~kernel:(Ops.kernel_info ())
       [ Ops.end_ (Ops.store (Ops.index output [ r ]) value) [ r ] ])
    (renderer "clang")

(* The calls a search of [k] links, and the variables it times them with. *)
let calls_and_vars k =
  let calls = ref [] and vars = ref [] in
  let link call = calls := call :: !calls in
  let time ~vars:v () =
    vars := v :: !vars;
    0.
  in
  ignore (search ~timing:(link, time) 1 k);
  (List.rev !calls, List.rev !vars)

let argument call slot = List.nth (Ops.src_without_body call) slot

let samples =
  cases "samples each scalar at its bounds' middle, rounded down"
    ~id:(fun ((lo, hi), _) -> Printf.sprintf "[%d, %d]" lo hi)
    [ ((1, 16), 8); ((-7, -2), -5); ((-7, -1), -4) ]
    (fun (bounds, middle) ->
      let calls, vars = calls_and_vars (scalars bounds) in
      greater int ~msg:"linked" ~than:0 (List.length calls);
      List.iter
        (fun call ->
          equal ~msg:"the positional scalar" Dtypes.const
            (`Int (Bigint.of_int middle))
            (Ops.value (argument call 2)))
        calls;
      List.iter
        (equal ~msg:"the variables" (list (pair string int)) [ ("n", middle) ])
        vars)

let unused_slot =
  test "passes a byte in a slot no parameter takes, before the last" (fun () ->
      let calls, _ = calls_and_vars (scalars (1, 16)) in
      greater int ~msg:"linked" ~than:0 (List.length calls);
      List.iter
        (fun call ->
          equal Uops.uop
            (Shape.param ~shape:[ Int 1 ] ~device:"CPU" 1 Uint8)
            (argument call 1))
        calls)

let positional =
  cases "binds a positional size in its call and launch" ~id:string_of_bool
    [ false; true ] (fun allow_test_size ->
      let ast = kernel "symbolic" in
      let v = List.hd (Shape.variables ast) in
      let p = Shape.param_like v 4 in
      let ast = Ops.substitute ~calls:Skip ~pass:Once ast [ (v, p) ] in
      let k = scheduled_ast ast (renderer "hip") in
      let linked = ref 0 in
      let link call =
        incr linked;
        equal Dtypes.const
          (`Int (Bigint.of_int 4))
          (Ops.value (argument call 4));
        let info = program_info (Ops.body call) in
        List.iter
          (fun s -> ignore (Shape.sym_infer s []))
          (info.global_size @ info.local_size)
      in
      let time ~vars () =
        equal (list (pair string int)) [] vars;
        0.
      in
      ignore (search ~allow_test_size ~timing:(link, time) 1 k);
      greater int ~than:0 !linked)

let small_searches =
  Gen.(
    with_pp
      (fun ppf (kernel, target, amt) ->
        Format.fprintf ppf "%s on %s, width %d" kernel target amt)
      (triple
         (of_list [ "add_small"; "sum_rows"; "variable_rows"; "symbolic" ])
         (of_list [ "clang"; "metal" ])
         (int_range 1 3)))

let links_once =
  prop ~count:10 "links each binary it times once, before its samples"
    small_searches (fun (kernel, target, amt) ->
      let timing, record =
        sampling (fun ~vars prg _ -> golden_time ~failing:false ~vars prg)
      in
      ignore (search ~timing amt (scheduled kernel target));
      let r = record () in
      let sampled =
        List.fold_left
          (fun ps c -> if List.memq c.prg ps then ps else c.prg :: ps)
          [] r.calls
      in
      equal ~msg:"linked before sampled" (list Uops.uop) r.linked
        (List.rev sampled);
      equal ~msg:"binaries" (list string)
        (List.sort_uniq String.compare (List.map binary r.linked))
        (List.sort String.compare (List.map binary r.linked)))

(* A timing of constant samples that raises [e] when it links a program, or when
   it samples one. *)
let raising_at stage e =
  ( (fun call -> if stage = `Link then raise e else call),
    fun ~vars:_ _ -> if stage = `Time then raise e else 1e-3 )

let stages = [ ("a link", `Link); ("a sample", `Time) ]

let raising =
  [
    cases "raises what its timing raises but Failure, from" ~id:fst stages
      (fun (_, stage) ->
        raises Exit (fun () ->
            search ~timing:(raising_at stage Exit) 1
              (scheduled "sum_rows" "clang")));
    cases
      "drops a program whose timing raises Failure, and keeps its kernel when \
       it times none, from"
      ~id:fst stages (fun (_, stage) ->
        let k = scheduled "sum_rows" "clang" in
        let chosen =
          search ~timing:(raising_at stage (Failure "no device")) 1 k
        in
        equal Uops.uop (K.ast k) (K.ast chosen);
        equal opts [] (K.applied_opts chosen));
  ]

let named_test =
  test "compiles each candidate as a kernel named test" (fun () ->
      let timing, linked = instant () in
      ignore (search ~timing 1 (scheduled "add_small" "clang"));
      let linked = linked () in
      greater int ~msg:"linked" ~than:0 (List.length linked);
      List.iter
        (fun p ->
          equal ~msg:(opts_text (applied p)) string "test" (kernel_info p).name)
        linked)

let links_group =
  group "beam_search links"
    ([
       named_test; storage_placed; samples; unused_slot; positional; links_once;
     ]
    @ raising)

(* beam_search compiles its candidates

   A search measures a program to choose the one that runs, so the program it
   times for a kernel's optimisations is the program a compile of the kernel
   with those optimisations makes. They can differ only where a pass after the
   optimisations reads what differs between a candidate and the kernel chosen:
   the host's blocks and decoded panels, and the staging of a tensor core's
   operands, on each target that stages. *)

let with_info f k =
  match Ops.arg k with
  | Kernel info -> Ops.replace k ~arg:(Kernel (f info))
  | _ -> failf "%a is no kernel" Ops.pp k

(* [c[i, j] = Σ_r a[i, r] * (float codes[j, r] * scales[j, r / 32])] over 2 by 2
   and 32 [r], in float32: a product whose right operand is decoded from byte
   codes, as a quantised matrix's is. *)
let decoded_product () =
  let m = 2 and n = 2 and k = 32 in
  let param slot size dt = Shape.param ~shape:[ Ops.Int size ] slot dt in
  let c = param 0 (m * n) Float32
  and a = param 1 (m * k) Float32
  and codes = param 2 (n * k) Uint8
  and scales = param 3 (n * k / 32) Float32 in
  let open Ops.O in
  let i = Ops.range ~axis_type:Weak (Int m) [ 0 ]
  and j = Ops.range ~axis_type:Weak (Int n) [ 1 ]
  and r = Ops.range ~axis_type:Weak (Int k) [ 2 ] in
  let value =
    Ops.cast (Ops.index codes [ (j * int k) + r ]) Float32
    * Ops.index scales [ (j * int Stdlib.(k / 32)) + (r // int 32) ]
  in
  let sum = Ops.reduce (Ops.index a [ (i * int k) + r ] * value) Add [ r ] in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (Ops.index c [ (i * int n) + j ]) sum) [ i; j ] ]

(* The program [ast] compiles to on [ren] with the optimisations [opts]. *)
let compiled_with ?(settings = []) ast ren opts =
  quietly ~settings (fun () ->
      Codegen.to_program
        (with_info (fun i -> { i with opts_to_apply = Some opts }) ast)
        ren)

(* Whether [prg]'s kernel copies a tensor core's operands into workgroup
   memory. *)
let staged prg =
  let nodes = Ops.toposort ~calls:Skip (Ops.nth prg 0) in
  List.exists (fun u -> Ops.op u = Op.Wmma) nodes
  && List.exists (fun u -> Ops.addrspace u = Some Dtype.Local) nodes

(* Each pass, the kernel and target that reach it, and whether a candidate's
   program went through it. *)
let paths =
  [
    ( "the host's blocks",
      "sum_rows",
      "clang",
      fun _ _ p -> Option.is_some (kernel_info p).split );
    ( "a decoded panel",
      "decoded",
      "clang",
      fun ast ren p ->
        binary p
        <> binary
             (compiled_with
                ~settings:[ B (Setting.stage, false) ]
                ast ren (applied p)) );
    ("a staged tensor core", "matmul_small", "metal", fun _ _ p -> staged p);
    ("a staged tensor core", "matmul_half", "cuda", fun _ _ p -> staged p);
    ("a staged tensor core", "matmul_half", "hip", fun _ _ p -> staged p);
    ( "a variable in a launch",
      "variable_rows",
      "metal",
      (* The search times a launch of constant sizes, the variable's sample. *)
      fun ast ren p ->
        List.exists
          (function Ops.Sym _ -> true | Int _ -> false)
          (program_info (compiled_with ast ren (applied p))).global_size );
  ]

let times_what_runs =
  cases "times the program each candidate compiles to, and returns it"
    ~id:(fun (path, name, target, _) ->
      Printf.sprintf "%s: %s on %s" path name target)
    paths
    (fun (path, name, target, reaches) ->
      let ren = renderer target in
      let ast = if name = "decoded" then decoded_product () else kernel name in
      let (link, time), record =
        sampling (fun ~vars prg _ -> drawn_time 0 ~vars prg)
      in
      let chosen =
        quietly (fun () ->
            Codegen.to_program
              ~beam:(Search.beam_search ~part ~link ~time)
              (with_info (fun i -> { i with beam = 1 }) ast)
              ren)
      in
      let linked = (record ()).linked in
      List.iter
        (fun p ->
          equal
            ~msg:(opts_text (applied p))
            string
            (binary (compiled_with ast ren (applied p)))
            (binary p))
        linked;
      let same p = List.equal Opt.equal (applied p) (applied chosen) in
      (match List.find_opt same linked with
      | Some p ->
          equal ~msg:"the chosen program" string (binary p) (binary chosen)
      | None -> failf "the chosen optimisations were never timed");
      satisfies
        ~claim:("a candidate through " ^ path)
        (list (pair opts bool))
        (List.exists snd)
        (List.map (fun p -> (applied p, reaches ast ren p)) linked))

(* [out[i] = a[i] * 2 + 1] over 4096 floats and [out[i] = a[i] * a[i]] over
   2048, on CUDA's buffers, the second asking for a search: the members of a
   launch. *)
let launch_members () =
  let floats n slot = Shape.param ~shape:[ Ops.Int n ] slot Float32 in
  let buf n = Ops.new_buffer "CUDA" n Float32 in
  let a = buf 4096 in
  let i = Ops.range ~axis_type:Weak (Int 4096) [ 0 ] in
  let scaled =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"scaled" ())
      [
        Ops.end_
          (Ops.store
             (Ops.index (floats 4096 0) [ i ])
             Ops.O.((Ops.index (floats 4096 1) [ i ] * float 2.) + float 1.))
          [ i ];
      ]
  in
  let k = Ops.range ~axis_type:Weak (Int 2048) [ 0 ] in
  let squared =
    let x = Ops.index (floats 4096 1) [ k ] in
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"squared" ~beam:1 ())
      [
        Ops.end_
          (Ops.store (Ops.index (floats 2048 0) [ k ]) Ops.O.(x * x))
          [ k ];
      ]
  in
  Ops.body
    (Shape.launch
       [ Ops.call scaled [ buf 4096; a ]; Ops.call squared [ buf 2048; a ] ])

let member_alone =
  test "times a launch's member as a kernel of its own" (fun () ->
      let (link, time), linked = instant () in
      ignore
        (quietly (fun () ->
             Codegen.to_program
               ~beam:(Search.beam_search ~part ~link ~time)
               (launch_members ()) (renderer "cuda")));
      let linked = linked () in
      greater int ~msg:"linked" ~than:0 (List.length linked);
      (* The launch holds three buffers, its member two. *)
      List.iter
        (fun p ->
          equal
            ~msg:(opts_text (applied p))
            int 2
            (List.length (program_info p).globals))
        linked)

(* The programs a search of [name] on the host links under the bound [m] of
   instructions, and the sources it compiles. *)
let linked_under m name =
  let c = counts () in
  let ren = Renderer.with_compiler (counting c) (renderer "clang") in
  let timing, linked = instant () in
  ignore
    (search
       ~settings:[ B (Setting.beam_uops_max, m) ]
       ~timing 1 (scheduled_for ren name));
  (linked (), List.map fst (compiled c))

let uops_bound =
  test "compiles and times no candidate of BEAM_UOPS_MAX instructions or more"
    (fun () ->
      let all, _ = linked_under max_int "add_small" in
      let m = List.fold_left (fun m p -> max m (instructions p)) 0 all in
      let kept, sources = linked_under m "add_small" in
      equal ~msg:"timed" (list opts)
        (List.filter_map
           (fun p -> if instructions p < m then Some (applied p) else None)
           all)
        (List.map applied kept);
      equal ~msg:"compiled" (list string)
        (List.sort_uniq String.compare (List.map binary kept))
        (List.sort_uniq String.compare sources))

let uops_unbounded =
  cases "compiles every candidate when BEAM_UOPS_MAX is 0 or less"
    ~id:string_of_int [ 0; -1 ] (fun m ->
      let all, _ = linked_under max_int "add_small" in
      let kept, _ = linked_under m "add_small" in
      equal (list opts) (List.map applied all) (List.map applied kept))

(* A Clang renderer whose compiler rejects each source whose length [every]
   divides (default 3), with [reject ()], and the count of rejections. *)
let rejecting ?(every = 3) reject =
  let rejected = ref 0 in
  let compile src =
    if String.length src mod every = 0 then (
      incr rejected;
      reject ())
    else src
  in
  ( Renderer.with_compiler (Renderer.Compiler.v compile) (renderer "clang"),
    rejected )

let rejected_by_compiler () = raise (Renderer.Compiler.Compile_error "rejected")

let dropped ?(settings = []) reject () =
  let ren, rejected = rejecting reject in
  let timing, linked = instant () in
  ignore (search ~settings ~timing 1 (scheduled_for ren "sum_rows"));
  greater int ~msg:"rejected" ~than:0 !rejected;
  greater int ~msg:"measured" ~than:0 (List.length (linked ()))

let strict = [ Setting.B (Setting.beam_strict_mode, true) ]

let uncompilable =
  [
    test "drops a candidate its compiler rejects" (dropped rejected_by_compiler);
    test "drops a candidate whose compilation raises Failure"
      (dropped (fun () -> failwith "rejected"));
    test "under BEAM_STRICT_MODE, raises what a compilation raises but Failure"
      (fun () ->
        let ren, _ = rejecting ~every:1 rejected_by_compiler in
        raises_match
          (function Renderer.Compiler.Compile_error _ -> true | _ -> false)
          (fun () ->
            search ~settings:strict ~timing:constant 1
              (scheduled_for ren "add_small")));
    test
      "under BEAM_STRICT_MODE, drops a candidate whose compilation raises \
       Failure"
      (dropped ~settings:strict (fun () -> failwith "rejected"));
  ]

let binaries_kept =
  test
    "keeps each candidate's binary in its compiler's table, and reads it back"
    (fun () ->
      let c = counts () in
      let table = unique "search test binaries" in
      let ren =
        Renderer.with_compiler
          (counting ~cachekey:(fun () -> table) c)
          (renderer "metal")
      in
      let part = unique "binaries" in
      let k = with_info (fun i -> { i with beam = 2 }) (kernel "sum_rows") in
      let (link, time), record =
        sampling (fun ~vars prg _ -> golden_time ~failing:false ~vars prg)
      in
      let prg =
        quietly
          ~settings:[ B (Setting.cachelevel, 1) ]
          (fun () ->
            Codegen.to_program
              ~beam:(Search.beam_search ~part ~link ~time)
              k ren)
      in
      (* The compiler returns the text it compiles, which is the key. *)
      let kept p =
        quietly
          ~settings:[ B (Setting.cachelevel, 1) ]
          (fun () -> Helpers.Diskcache.get ~table (binary p))
      in
      let candidates = (record ()).linked in
      greater int ~than:1 (List.length candidates);
      equal ~msg:"candidates' binaries kept"
        (list (option string))
        (List.map (fun p -> Some (binary p)) candidates)
        (List.map kept candidates);
      equal ~msg:"the chosen binary kept" (option string)
        (Some (binary prg))
        (kept prg);
      Hashtbl.reset c.table;
      let timing, linked = instant () in
      ignore
        (search ~part
           ~settings:
             [ B (Setting.cachelevel, 1); B (Setting.ignore_beam_cache, true) ]
           ~timing 2
           (scheduled_for ren "sum_rows"));
      greater int ~msg:"searched again" ~than:0 (List.length (linked ()));
      equal ~msg:"compiled again" (list (pair string int)) [] (compiled c))

let programs_not_kept =
  test "keeps no candidate's program" (fun () ->
      let before = Codegen.programs () in
      let timing, linked = instant () in
      ignore (search ~timing 2 (scheduled "sum_rows" "clang"));
      greater int ~msg:"linked" ~than:0 (List.length (linked ()));
      equal int before (Codegen.programs ()))

let compiles_group =
  group "beam_search compiles"
    ([ times_what_runs; member_alone; uops_bound; uops_unbounded ]
    @ uncompilable
    @ [ binaries_kept; programs_not_kept ])

(* beam_search times its candidates in rounds *)

(* How many samples the first timing of each program of [n] optimisations
   took. *)
let measured_times n calls =
  List.filter_map
    (fun (p, ts) -> if depth p = n then Some (List.length ts) else None)
    (samples_of calls)

let min_progress = 0.01 /. 1e6

let searched_by_depth times =
  let timing, calls = recording (by_depth times) in
  let k = search ~timing 1 (scheduled "sum_rows" "clang") in
  (k, calls ())

let of_depth d calls =
  List.filter_map
    (fun c -> if depth c.prg = d then Some (applied c.prg) else None)
    calls

let first_timing =
  [
    test
      "samples a candidate three times when its least is three times the \
       beam's first's" (fun () ->
        let _, calls = searched_by_depth [ 0.; 0.25; 0.75 ] in
        let n = measured_times 2 calls in
        greater int ~msg:"measured" ~than:0 (List.length n);
        equal (list int) (List.map (fun _ -> 3) n) n);
    test
      "samples a candidate once when its first sample exceeds three times the \
       beam's first's" (fun () ->
        let _, calls = searched_by_depth [ 0.; 0.25; 0.875 ] in
        let n = measured_times 2 calls in
        greater int ~msg:"measured" ~than:0 (List.length n);
        equal (list int) (List.map (fun _ -> 1) n) n);
    test "never times the kernel it starts from" (fun () ->
        let _, calls = searched_by_depth [ 1.; 1e-3; 1e-3 ] in
        equal (list opts) [] (of_depth 0 calls));
  ]

(* The least sample of a program of [d] optimisations in its first timing: the
   incumbent of the round that times programs of one more. *)
let incumbent calls d =
  least
    (List.concat_map
       (fun (p, ts) -> if depth p = d then ts else [])
       (samples_of calls))

(* A search of [kernel] on the host, under BEAM_PADTO if [padto], of width
   [amt], sampled by [spread seed]: its choice and samples. The covers of the
   candidates a round drops before timing them read its compilations. *)
let spread_search (kernel, padto) amt seed =
  let c = counts () in
  let ren = Renderer.with_compiler (counting c) (renderer "clang") in
  let timing, record = sampling (spread seed) in
  let chosen =
    K.applied_opts
      (search
         ~settings:[ B (Setting.beam_padto, padto) ]
         ~allow_test_size:false ~timing amt (scheduled_for ren kernel))
  in
  let r = record () in
  let linked = List.map binary r.linked and compiled = compiled c in
  cover "a candidate whose binary was timed before"
    (List.exists (fun (_, n) -> n > 1) compiled);
  cover "a candidate of too many operations"
    (List.exists (fun (src, _) -> not (List.mem src linked)) compiled);
  (chosen, r.calls)

(* The kernels the timing laws search: the cheapest whose searches go past their
   first round and time a binary again. *)
let timing_kernels =
  [ ("add_small", false); ("variable_rows", false); ("symbolic", false) ]

let pp_kernel ppf (kernel, padto) =
  Format.fprintf ppf "%s%s" kernel (if padto then " under BEAM_PADTO" else "")

let search_draws =
  Gen.(
    with_pp
      (fun ppf (kernel, amt, seed) ->
        Format.fprintf ppf "%a, width %d, seed %d" pp_kernel kernel amt seed)
      (triple (of_list timing_kernels) (int_range 1 3) nat))

(* transpose_33's pads on the host make 1000 times the operations of its splits:
   rounds that drop candidates for their operations. Of the covers of the laws
   below, this example alone holds "a candidate of too many operations" on every
   run, which no drawn case reaches, and "an early stop", which a drawn case
   reaches about half the time: its second round, which always runs, times
   candidates of more than three times its first's least. *)
let padded_transpose = (("transpose_33", true), 2, 0)

let early_stops =
  prop ~count:20 ~examples:[ padded_transpose ]
    "samples a candidate's first timing until its least exceeds three times \
     the beam's first's, three times at most"
    search_draws (fun (kernel, amt, seed) ->
      let _, calls = spread_search kernel amt seed in
      List.iter
        (fun (p, ts) ->
          let d = depth p in
          let stop =
            if d = 0 then infinity else 3. *. incumbent calls (d - 1)
          in
          let n = List.length ts in
          let msg = opts_text (applied p) in
          at_least int ~msg ~than:1 n;
          at_most int ~msg ~than:3 n;
          List.iteri
            (fun i _ ->
              let before = least (List.filteri (fun j _ -> j <= i) ts) in
              if i < n - 1 then at_most float_exact ~msg ~than:stop before
              else if n < 3 then greater float_exact ~msg ~than:stop before)
            ts;
          cover "an early stop" (n < 3);
          cover "three samples" (n = 3))
        (samples_of calls))

let rounds =
  [
    test "goes on while the fastest gains BEAM_MIN_PROGRESS or more" (fun () ->
        let _, calls =
          searched_by_depth [ 0.; 2. *. min_progress; min_progress; 1. ]
        in
        greater int ~msg:"a third round's samples" ~than:0
          (List.length (of_depth 3 calls)));
    test "keeps the beam's first when the fastest gains nothing" (fun () ->
        let k, calls = searched_by_depth [ 0.; 1e-3; 1e-3 ] in
        greater int ~msg:"a second round's samples" ~than:0
          (List.length (of_depth 2 calls));
        equal int 1 (List.length (K.applied_opts k)));
    test "ends when the fastest takes less than BEAM_MIN_PROGRESS" (fun () ->
        let k, calls = searched_by_depth [ 0.; min_progress /. 2.; 0. ] in
        equal (list opts) [] (of_depth 2 calls);
        equal int 1 (List.length (K.applied_opts k)));
    test "keeps the fastest alone when it gains less than BEAM_MIN_PROGRESS"
      (fun () ->
        let k, _ =
          searched_by_depth [ 0.; 1e-3; 1e-3 -. (min_progress /. 2.) ]
        in
        equal int 2 (List.length (K.applied_opts k)));
  ]

(* The fastest program of [d] optimisations, the first timed of ties, if any. *)
let fastest calls d =
  List.fold_left
    (fun best (p, ts) ->
      if depth p <> d then best
      else
        match best with
        | Some (_, bs) when least bs <= least ts -> best
        | _ -> Some (p, ts))
    None (samples_of calls)

let rounds_law =
  prop ~count:30 ~examples:[ padded_transpose ]
    "after a round, times the beam's first and its fastest again in turn, goes \
     on while the fastest beats the beam's first by BEAM_MIN_PROGRESS or more, \
     and keeps the last round's fastest if it beat it"
    search_draws (fun (kernel, amt, seed) ->
      let chosen, calls = spread_search kernel amt seed in
      let again = (timings calls).again in
      (* The last round that timed a candidate, and its fastest. *)
      let last =
        List.fold_left (fun n (p, _) -> max n (depth p)) 0 (samples_of calls)
      in
      let fastest d =
        match fastest calls d with
        | Some (p, ts) -> (p, least ts)
        | None -> failf "round %d timed no candidate" d
      in
      equal int ~msg:"rounds timed again" (max 0 (last - 1)) (List.length again);
      (* What round [d] compares: the beam's first's time and the fastest's. The
         first round's beam is the kernel, untimed. *)
      let compared d =
        if d = 1 then (infinity, snd (fastest 1))
        else
          let (first, t_first), (fast, t_fast) = List.nth again (d - 2) in
          let msg = Printf.sprintf "round %d" d in
          equal opts ~msg (applied (fst (fastest (d - 1)))) (applied first);
          equal opts ~msg (applied (fst (fastest d))) (applied fast);
          (t_first, t_fast)
      in
      let goes_on (t_first, t_fast) =
        t_fast >= min_progress && t_first -. t_fast >= min_progress
      in
      for d = 1 to last - 1 do
        let msg = Printf.sprintf "round %d" d in
        let t_first, t_fast = compared d in
        at_least float_exact ~msg ~than:min_progress t_fast;
        at_least float_exact ~msg ~than:min_progress (t_first -. t_fast)
      done;
      let expected =
        if last = 0 then []
        else
          let t_first, t_fast = compared last in
          if goes_on (t_first, t_fast) || t_fast < t_first then
            applied (fst (fastest last))
          else if last = 1 then []
          else applied (fst (fastest (last - 1)))
      in
      equal opts expected chosen;
      cover "a round timed again" (again <> []);
      cover "a last round timed again that kept the beam's first"
        (last >= 2 && List.length chosen < last))

let chooses_the_fastest =
  prop ~count:20 "chooses the fastest program it measured, at steady times"
    search_draws (fun ((kernel, _), amt, seed) ->
      let timing, calls = recording (drawn_time seed) in
      let k = scheduled kernel "clang" in
      let chosen = search ~timing amt k in
      match calls () with
      | [] -> equal Uops.uop (K.ast k) (K.ast chosen)
      | calls ->
          let fastest =
            List.fold_left (fun t c -> Float.min t c.time) infinity calls
          in
          equal float_exact fastest (drawn seed (K.applied_opts chosen)))

(* A timing by [by_depth times] whose samples of a program of [d] optimisations,
   any if [None], raise [e] from its fourth on: the times a round takes
   again. *)
let raising_again ?depth:d e times =
  fst
    (sampling (fun ~vars prg i ->
         if i >= 3 && Option.fold ~none:true ~some:(Int.equal (depth prg)) d
         then raise e
         else by_depth times ~vars prg))

let raising_again_cases =
  let chosen ?depth times =
    let timing = raising_again ?depth (Failure "no device") times in
    List.length
      (K.applied_opts (search ~timing 1 (scheduled "sum_rows" "clang")))
  in
  group "a time taken again that raises"
    [
      test "Failure counts as infinite for the fastest, which is not kept"
        (fun () -> equal int 1 (chosen ~depth:2 [ 0.; 1e-3; 0.9e-3 ]));
      test
        "Failure counts as infinite for the beam's first, which the fastest \
         beats" (fun () -> equal int 2 (chosen ~depth:1 [ 0.; 1e-3; 1e-3 ]));
      test "Failure for both ends the search on the beam's first" (fun () ->
          equal int 1 (chosen [ 0.; 1e-3; 0.9e-3 ]));
      test "Exit is raised by the search" (fun () ->
          raises Exit (fun () ->
              search
                ~timing:(raising_again Exit [ 0.; 1e-3; 0.9e-3 ])
                1
                (scheduled "sum_rows" "clang")));
    ]

let too_much_compute =
  test
    "times no candidate of more than 1000 times the fewest operations of its \
     round so far" (fun () ->
      Setting.context [ B (Setting.beam_padto, true) ] @@ fun () ->
      let c = counts () in
      let ren = Renderer.with_compiler (counting c) (renderer "metal") in
      let timing, record =
        sampling (fun ~vars prg _ -> golden_time ~failing:false ~vars prg)
      in
      ignore (search ~timing 1 (scheduled_for ren "transpose_33"));
      let calls () = (record ()).calls in
      let linked = List.map binary (record ()).linked in
      greater int ~msg:"measured" ~than:0 (List.length linked);
      (* A candidate compiled and never linked was dropped for its
         operations. *)
      greater int ~msg:"compiled, never linked" ~than:0
        (List.length
           (List.filter
              (fun (src, _) -> not (List.mem src linked))
              (compiled c)));
      (* [k] applies no optimisation, and a candidate of round [r] applies [r].
         A candidate dropped for its operations lowers no round's fewest, so the
         fewest so far are those measured. *)
      let fewest = Hashtbl.create 4 in
      List.iter
        (fun c ->
          let info = kernel_info c.prg in
          let round = List.length info.applied_opts in
          let ops =
            match info.estimates with
            | Some e -> Shape.sym_infer e.ops c.vars
            | None -> 0
          in
          let least =
            Option.fold ~none:ops ~some:(Int.min ops)
              (Hashtbl.find_opt fewest round)
          in
          at_most int
            ~msg:(opts_text info.applied_opts)
            ~than:(1000 * least) ops;
          Hashtbl.replace fewest round least)
        (calls ()))

(* [out[i] = j^2] for each [j < x[0]] in turn, over 16 [i], [x] of int64: its
   estimated operations count [j] up to 2^63 - 1 times on each [i], which fits
   no int. *)
let counts_to_loaded () =
  let x = Shape.param ~shape:[ Int 16 ] 1 Int64
  and out = Shape.param ~shape:[ Int 16 ] 0 Int64 in
  let i = Ops.range ~axis_type:Weak (Int 16) [ 0 ] in
  let count = Ops.load (Ops.index x [ Ops.int 0 ]) [] in
  let bound =
    Ops.cast (Ops.alu count Op.Max [ Ops.int ~dtype:Int64 0 ]) Weak_int
  in
  let j = Ops.range ~axis_type:Loop (Sym bound) [ 1 ] in
  let jl = Ops.cast j Int64 in
  let st = Ops.store (Ops.index out [ i ]) Ops.O.(jl * jl) in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"counts_to_loaded" ())
    [ Ops.end_ (Ops.end_ st [ j ]) [ i ] ]

let estimates_past_an_int =
  test "times candidates whose estimated operations fit no int" (fun () ->
      let timing, linked = instant () in
      ignore
        (search ~timing 1
           (scheduled_ast (counts_to_loaded ()) (renderer "clang")));
      greater int ~msg:"candidates measured" ~than:0 (List.length (linked ())))

let searched cell =
  let timing, calls =
    recording (golden_time ~failing:(bool_of_cell (cell "failing")))
  in
  let k =
    search ~timing
      ~allow_test_size:(bool_of_cell (cell "allow_test_size"))
      (int_of_string (cell "amt"))
      (scheduled (cell "kernel") (cell "target"))
  in
  equal opts (Kernel_opts.opts_of_cell (cell "opts")) (K.applied_opts k);
  equal ~msg:"measurements" int
    (int_of_string (cell "measurements"))
    (first_samples (calls ()))

let recorded_searches =
  group "chooses as its golden states, measuring as often"
    [
      Golden.cases
        ~key:[ "kernel"; "target"; "amt"; "failing"; "allow_test_size" ]
        "searches.golden" searched;
      Golden.cases ~key:[ "kernel"; "target"; "amt" ]
        "searches_environment.golden" (fun cell ->
          environment (fun () ->
              searched (fun c ->
                  match c with
                  | "failing" -> "False"
                  | "allow_test_size" -> "True"
                  | c -> cell c)));
    ]

(* A device whose clock changes between rounds: a sample of a program takes
   [time] of it times the [r]th of [clocks], or their last for more, where [r],
   the round the search is in, is the most optimisations of a program it linked
   so far. *)
let between_rounds clocks time =
  let round = ref 0 in
  let link call =
    let prg = Ops.body call in
    round := max !round (depth prg);
    prg
  in
  let time ~vars prg =
    let clock = List.nth clocks (min !round (List.length clocks - 1)) in
    time ~vars prg *. clock
  in
  (link, time)

let slower_round =
  test "goes on past a round whose clock is half the round before's" (fun () ->
      let timing =
        between_rounds [ 1.; 1.; 2. ] (by_depth [ 0.; 1e-3; 0.9e-3 ])
      in
      let k = search ~timing 1 (scheduled "sum_rows" "clang") in
      equal int 2 (List.length (K.applied_opts k)))

(* [drawn_time seed] with an optimum at two optimisations, so that most searches
   reach their second round. *)
let two_deep seed ~vars prg =
  drawn_time seed ~vars prg
  *. (1. +. (0.5 *. Float.of_int (abs (depth prg - 2))))

let clock_between_rounds =
  let pp ppf (kernel, amt, seed, (slower, later)) =
    Format.fprintf ppf "%s, width %d, seed %d, second round %s, then %a" kernel
      amt seed
      (if slower then "slower" else "faster")
      Format.(pp_print_list ~pp_sep:pp_print_space pp_print_float)
      later
  in
  let steady_clocks = List.init 6 (Fun.const 1.) in
  (* add_small's steady search of seed 0 keeps two optimisations. *)
  prop ~count:30
    ~examples:
      [
        ("add_small", 1, 0, (true, steady_clocks));
        ("add_small", 1, 0, (false, steady_clocks));
      ]
    "chooses as at a steady clock when its device's clock changes between \
     rounds"
    Gen.(
      with_pp pp
        (quad
           (of_list (List.map fst timing_kernels))
           (int_range 1 3) nat
           (pair bool (list ~size:(constant 6) (of_list [ 0.5; 1.; 1.5; 2. ])))))
    (fun (kernel, amt, seed, (slower, later)) ->
      let chosen clocks =
        K.applied_opts
          (search
             ~timing:(between_rounds clocks (two_deep seed))
             amt (scheduled kernel "clang"))
      in
      let steady = chosen [ 1. ] in
      let second = if slower then 2. else 0.5 in
      equal opts steady (chosen (1. :: 1. :: second :: later));
      cover "a slower second round, where a steady search goes past the first"
        (slower && List.length steady >= 2);
      cover "a faster second round" (not slower))

let times_group =
  group "beam_search times"
    (first_timing @ [ early_stops ] @ rounds
    @ [
        rounds_law;
        chooses_the_fastest;
        raising_again_cases;
        too_much_compute;
        estimates_past_an_int;
        recorded_searches;
        slower_round;
        clock_between_rounds;
      ])

(* beam_search measures on fewer workgroups *)

(* The launches the first round of a search of [kernel] on Metal measures, by
   the candidate's optimisations. *)
let launches kernel =
  let timing, calls = recording (fun ~vars:_ _ -> 0.) in
  ignore (search ~timing 1 (scheduled kernel "metal"));
  List.map
    (fun c ->
      ( applied c.prg,
        List.map
          (fun s -> Shape.sym_infer s c.vars)
          (program_info c.prg).global_size ))
    (calls ())

let launch_of launches opt =
  match List.assoc_opt [ opt ] launches with
  | Some sizes -> sizes
  | None -> failf "%a was not measured" Opt.pp opt

(* A time proportional to the workgroups launched, so that a program measured on
   fewer and scaled up takes the time of all of them. *)
let per_workgroup ~vars prg =
  Float.of_int (workgroups ~vars prg)
  *. 1e-9
  *. golden_time ~failing:false ~vars prg

let test_size =
  group "beam_search measures on fewer workgroups"
    [
      (* add_3d launches 2^19 workgroups, [1024; 32; 16] split by 2 on one axis:
         [512; 32; 16] halves its last size above 16 to [512; 16; 16], then its
         16s no more, to [512; 16; 8] at 2^16; [1024; 16; 16] halves to [512;
         16; 16], then [512; 8; 16]. *)
      test "halves the last size above 16 down to 65536 workgroups" (fun () ->
          let split axis amount =
            Opt.Split { axis; amount; target = Upcast; top = false }
          in
          let launches = launches "add_3d" in
          equal (list int) [ 512; 16; 8 ] (launch_of launches (split 0 2));
          equal (list int) [ 512; 8; 16 ] (launch_of launches (split 1 2)));
      test "keeps a launch of exactly 65536 workgroups whole" (fun () ->
          equal (list int) [ 65536; 1; 1 ]
            (launch_of (launches "add_large")
               (Opt.Split { axis = 0; amount = 16; target = Local; top = true })));
      test "measures no program on more than 65536 workgroups" (fun () ->
          let timing, calls = recording (golden_time ~failing:false) in
          ignore (search ~timing 1 (scheduled "add_large" "metal"));
          List.iter
            (fun c ->
              at_most
                ~msg:(opts_text (applied c.prg))
                int ~than:65536
                (workgroups ~vars:c.vars c.prg))
            (calls ()));
      test "measures a host program's blocks over its whole loop" (fun () ->
          let timing, calls = recording (golden_time ~failing:false) in
          ignore (search ~timing 1 (scheduled "add_large" "clang"));
          let split c =
            match (kernel_info c.prg).split with
            | Some { iterations; _ } ->
                Some
                  ( Shape.sym_infer iterations c.vars,
                    Helpers.prod
                      (List.map
                         (fun s -> Shape.sym_infer s c.vars)
                         (program_info c.prg).global_size) )
            | None -> None
          in
          let splits = List.filter_map split (calls ()) in
          greater int ~msg:"programs in blocks" ~than:0 (List.length splits);
          List.iter (fun (n, size) -> equal int n size) splits);
      test "takes BEAM_ESTIMATE's value when not given allow_test_size"
        (fun () ->
          let timing, calls = recording (golden_time ~failing:false) in
          ignore
            (search
               ~settings:[ B (Setting.beam_estimate, false) ]
               ~timing 1
               (scheduled "add_large" "metal"));
          greater int ~than:65536
            (List.fold_left
               (fun m c -> max m (workgroups ~vars:c.vars c.prg))
               0 (calls ())));
      test "measures every workgroup without allow_test_size" (fun () ->
          let timing, calls = recording (golden_time ~failing:false) in
          ignore
            (search ~allow_test_size:false ~timing 1
               (scheduled "add_large" "metal"));
          greater int ~than:65536
            (List.fold_left
               (fun m c -> max m (workgroups ~vars:c.vars c.prg))
               0 (calls ())));
      test "chooses as measuring every workgroup does, for a time per workgroup"
        (fun () ->
          let choose allow_test_size =
            let timing, _ = recording per_workgroup in
            K.applied_opts
              (search ~allow_test_size ~timing 1
                 (scheduled "add_large" "metal"))
          in
          equal opts (choose false) (choose true));
    ]

(* beam_search keeps its choice *)

(* A search under the cache, on the part [part], and its optimisations. *)
let cached ?(settings = []) ?allow_test_size ~part ~timing amt k =
  K.applied_opts
    (search ~part
       ~settings:(B (Setting.cachelevel, 1) :: settings)
       ?allow_test_size ~timing amt k)

let kept =
  test "a search kept compiles, links and times nothing" (fun () ->
      let c = counts () in
      let ren = Renderer.with_compiler (counting c) (renderer "clang") in
      let part = unique "kept" in
      let first =
        cached ~part
          ~timing:(fst (instant ()))
          1
          (scheduled_for ren "add_small")
      in
      greater int ~msg:"compiled first" ~than:0 (List.length (compiled c));
      Hashtbl.reset c.table;
      equal opts first
        (cached ~part ~timing:untimed 1 (scheduled_for ren "add_small"));
      equal ~msg:"compiled again" (list (pair string int)) [] (compiled c))

let ignored =
  test "a search kept is measured again under IGNORE_BEAM_CACHE" (fun () ->
      let part = unique "ignored" in
      ignore
        (cached ~part
           ~timing:(fst (instant ()))
           1
           (scheduled "add_small" "clang"));
      raises Exit (fun () ->
          cached
            ~settings:[ B (Setting.ignore_beam_cache, true) ]
            ~part ~timing:measuring 1
            (scheduled "add_small" "clang")))

(* What a search is keyed on. *)
type keyed = {
  kernel : string;
  amt : int;
  allow_test_size : bool;
  ren : Renderer.t;
  part : string;
  settings : Setting.binding list;
}

(* A setting its caller declares to reach output: the search knows nothing of
   it, yet keys on it. *)
let declared = Setting.int ~reach:Output "TOLK_TEST_SEARCH" 0
let changed setting s = { s with settings = setting :: s.settings }

let another_arch () =
  Cstyle.without_compiler
    {
      Helpers.Target.device = "CPU";
      renderer = "CLANG";
      arch = "x86_64,znver2";
      interface = "";
      indices = "";
    }

(* [r] under another name, a renderer of its target alike. *)
let renamed (r : Renderer.t) =
  Renderer.v ~name:(r.name ^ " another") ~suffix:r.suffix
    ~supports_float4:r.supports_float4 ~half8:r.half8 ~registers:r.registers
    ~vector_lanes:r.vector_lanes ~has_local:r.has_local ~has_shared:r.has_shared
    ~global_max:r.global_max ~local_max:r.local_max
    ?global_prod_max:r.global_prod_max ~shared_max:r.shared_max
    ~async_copy:r.async_copy ~tensor_cores:r.tensor_cores
    ~per_byte_first:r.per_byte_first ~units:r.units ~in_flight:r.in_flight
    ~extra_matcher:r.extra_matcher ~packed_args:r.packed_args
    ~code_for_op:r.code_for_op ~native:r.native ~render:r.render
    ~compiler:r.compiler r.target

let keys =
  [
    ("the kernel", fun s -> { s with kernel = "sum_rows" });
    ("the width", fun s -> { s with amt = 2 });
    ("allow_test_size", fun s -> { s with allow_test_size = false });
    ("the renderer's name", fun s -> { s with ren = renamed s.ren });
    ("the renderer's target", fun s -> { s with ren = another_arch () });
    ( "the compiler's table",
      fun s ->
        let table =
          Renderer.Compiler.v ~cachekey:(fun () -> "another") Fun.id
        in
        { s with ren = Renderer.with_compiler table s.ren } );
    ("the part", fun s -> { s with part = s.part ^ " another" });
    ("TC", changed (B (Setting.use_tc, 2)));
    ("BEAM_PADTO", changed (B (Setting.beam_padto, true)));
    ("BEAM_UOPS_MAX", changed (B (Setting.beam_uops_max, 2999)));
    ("BEAM_UPCAST_MAX", changed (B (Setting.beam_upcast_max, 255)));
    ("BEAM_LOCAL_MAX", changed (B (Setting.beam_local_max, 1023)));
    ("BEAM_MIN_PROGRESS", changed (B (Setting.beam_min_progress, 0.02)));
    ("BEAM_ESTIMATE", changed (B (Setting.beam_estimate, false)));
    ("TRANSCENDENTAL", changed (B (Setting.transcendental, 2)));
    ("a setting its caller declares", changed (B (declared, 1)));
  ]

let keyed_on =
  cases "a search kept under one value is measured again under another, of"
    ~id:fst keys (fun (_, change) ->
      let base =
        {
          kernel = "add_small";
          amt = 1;
          allow_test_size = true;
          ren = renderer "clang";
          part = unique "keyed";
          settings = [];
        }
      in
      let searched timing s =
        cached ~settings:s.settings ~allow_test_size:s.allow_test_size
          ~part:s.part ~timing s.amt
          (scheduled_for s.ren s.kernel)
      in
      let other = change base in
      (* Of the kernel, the row changes only what it names. *)
      equal ~msg:"the kernel on the row's renderer" Uops.uop
        (K.ast (scheduled_for base.ren base.kernel))
        (K.ast (scheduled_for other.ren base.kernel));
      ignore (searched (fst (instant ())) base);
      ignore (searched untimed base);
      raises Exit (fun () -> searched measuring other))

(* One expression stored on two devices of one target is one kernel, so the
   search of the second device's finds the first's in the cache. *)
let cache_devices =
  test "a search kept for one device serves another of its target" (fun () ->
      let on slot d =
        let x = Ops.new_buffer ~slot d 64 Float32 in
        let out = Ops.new_buffer ~slot:(slot + 2) d 64 Float32 in
        Ops.after out [ Ops.store out Ops.O.((x * x) + float 1.) ]
      in
      let linear, _ =
        Schedule.create_linear_with_vars ~cores:Cores.host
          (Ops.sink [ on 0 "CPU:0"; on 1 "CPU:1" ])
      in
      let part = unique "devices" in
      let searched timing call =
        cached ~part ~timing 2
          (scheduled_ast (Ops.body call) (renderer "clang"))
      in
      match Ops.src linear with
      | [ a; b ] ->
          let timing, _ = recording (golden_time ~failing:false) in
          let first = searched timing a in
          equal opts first (searched untimed b)
      | calls -> failf "%d calls, not 2" (List.length calls))

let beyond_applied =
  test
    "a search kept for a kernel with optimisations applies those it found \
     beyond them" (fun () ->
      let part = unique "beyond" in
      let upcast _ =
        Opt.Split { axis = 0; amount = 2; target = Upcast; top = false }
      in
      let k () = opted (renderer "clang") (kernel "sum_rows") [ upcast ] in
      let timing, _ = recording (golden_time ~failing:false) in
      let first = cached ~part ~timing 1 (k ()) in
      greater int ~msg:"found beyond the applied" ~than:1 (List.length first);
      equal opts first (cached ~part ~timing:untimed 1 (k ())))

(* A measurement that favours the optimisations [favoured] picks, with an
   optimum at one optimisation. *)
let favouring favoured ~vars:_ prg =
  let weight o = if favoured o then 0.5 else 1.2 in
  let tm = 1e-3 *. Float.of_int (1 + (3 * max 0 (depth prg - 1))) in
  List.fold_left (fun t o -> t *. weight o) tm (applied prg)

let kinds =
  [
    ( "a tensor core",
      "matmul_small",
      [],
      function Opt.Tc _ -> true | _ -> false );
    ("a swap", "matmul_small", [], function Opt.Swap _ -> true | _ -> false);
    ( "a pad",
      "add_small",
      [ Setting.B (Setting.beam_padto, true) ],
      function Opt.Padto _ -> true | _ -> false );
  ]

let kept_kinds =
  cases "a search kept keeps an optimisation of each kind, as"
    ~id:(fun (kind, _, _, _) -> kind)
    kinds
    (fun (_, name, settings, favoured) ->
      let part = unique "kinds" in
      let timing, _ = recording (favouring favoured) in
      let first = cached ~settings ~part ~timing 1 (scheduled name "metal") in
      satisfies ~claim:"holds the kind" opts (List.exists favoured) first;
      equal opts first
        (cached ~settings ~part ~timing:untimed 1 (scheduled name "metal")))

(* Damage reaches every entry of the table, so the searches that damage it run
   in a process of their own, on a cache of their own. The part prints the first
   search's choice, the second's, whether it measured, the third's choice and
   how many programs it linked; or "Failure". *)
let searched_again damage () =
  let k () = scheduled "sum_rows" "clang" in
  let timing, _ = recording (golden_time ~failing:false) in
  let first = cached ~part ~timing 2 (k ()) in
  Disk_cache.damage ~table:"beam_search" Helpers.cachedb damage;
  let timing, calls = recording (golden_time ~failing:false) in
  match cached ~part ~timing 2 (k ()) with
  | again ->
      let linked = ref 0 in
      let counted =
        ( (fun c ->
            incr linked;
            c),
          snd constant )
      in
      let kept = cached ~part ~timing:counted 2 (k ()) in
      Printf.printf "%s\n%s\n%b\n%s\n%d\n" (opts_line first) (opts_line again)
        (calls () <> [])
        (opts_line kept) !linked
  | exception Failure _ -> print_string "Failure"

let damages =
  [
    ("truncated", Disk_cache.truncated);
    ("under another key", Disk_cache.of_another_build);
  ]

let parts = List.map (fun (name, d) -> (name, searched_again d)) damages

let damaged =
  cases "a search measures again, and keeps what it finds, when its entry is"
    ~id:fst damages (fun (name, _) ->
      match Disk_cache.child ~cachedb:(Disk_cache.fresh ()) name with
      | Error e -> failf "the child failed: %s" e
      | Ok out -> (
          match String.split_on_char '\n' out with
          | [ first; again; measured; kept; linked; "" ] ->
              equal ~msg:"found again" string first again;
              equal ~msg:"measured again" string "true" measured;
              equal ~msg:"kept again" string first kept;
              equal ~msg:"linked when kept" string "0" linked
          | _ -> failf "the child wrote %S" out))

let cache =
  group "beam_search keeps its choice"
    [
      kept;
      ignored;
      keyed_on;
      cache_devices;
      beyond_applied;
      kept_kinds;
      damaged;
    ]

(* beam_search prints *)

let printing =
  group "beam_search prints"
    [
      test "its progress under DEBUG=2" (fun () ->
          ignore (output ());
          ignore
            (search
               ~settings:[ B (Setting.debug, 2) ]
               ~timing:(fst (instant ()))
               1
               (scheduled "add_small" "clang"));
          not_equal string "" (output ()));
      test "more under BEAM_DEBUG" (fun () ->
          ignore (output ());
          ignore
            (search
               ~settings:[ B (Setting.beam_debug, 1) ]
               ~timing:(fst (instant ()))
               1
               (scheduled "add_small" "clang"));
          not_equal string "" (output ()));
      test "a compilation's failure under DEBUG=4" (fun () ->
          let ren, _ = rejecting (fun () -> failwith "rejected") in
          ignore (output ());
          ignore
            (search
               ~settings:[ B (Setting.debug, 4) ]
               ~timing:(fst (instant ()))
               1
               (scheduled_for ren "sum_rows"));
          contains ~sub:"Failure(\"rejected\")" (output ()));
      test "nothing by default" (fun () ->
          ignore (output ());
          let timing, _ = recording (golden_time ~failing:true) in
          ignore (search ~timing 2 (scheduled "add_small" "clang"));
          ignore (acted ~max_up:1 (scheduled "sum_rows" "metal"));
          equal string "" (output ()));
    ]

(* beam_search raises *)

let refusals =
  group "beam_search raises Invalid_argument"
    [
      cases "for a width that is not positive" ~id:string_of_int [ 0; -1 ]
        (fun amt ->
          raises_match (Exn.invalid_arg ?sub:None) (fun () ->
              search ~timing:constant amt (scheduled "sum_rows" "clang")));
      test "as get_kernel_actions does" (fun () ->
          raises_match (Exn.invalid_arg ?sub:None) (fun () ->
              search
                ~settings:[ B (Setting.beam_upcast_max, 4) ]
                ~timing:constant 1 (variable_lanes ())));
    ]

let () =
  Disk_cache.play parts;
  exit
    (Windtrap.run "Tolk.Search"
       [
         actions;
         get_kernel_actions;
         links_group;
         compiles_group;
         times_group;
         test_size;
         cache;
         printing;
         refusals;
       ])
