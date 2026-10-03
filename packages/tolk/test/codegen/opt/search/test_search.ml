(* Tests of Tolk.Search: the actions and the candidates of real kernels are
   tinygrad's, a search chooses what tinygrad's chooses under the same
   measurement, it chooses the fastest program it measured, and every kernel it
   can choose writes what the kernel it started from writes. *)

open Windtrap
open Tolk
module K = Postrange.Scheduler

(* Goldens *)

(* No program is compiled: a program's binary is its source's bytes. *)
let uncompiled = Renderer.Compiler.v Fun.id

let renderer_of_row cell =
  let t =
    {
      Helpers.Target.device = cell "device";
      renderer = cell "renderer";
      arch = cell "arch";
      interface = "";
      indices = "";
    }
  in
  let r =
    match t.renderer with
    | "CLANG" -> Cstyle.clang t
    | "METAL" -> Cstyle.metal t
    | "CUDA" -> Cstyle.cuda t
    | "HIP" -> Cstyle.hip t
    | r -> failf "no renderer %s" r
  in
  Renderer.with_compiler uncompiled r

let renderers =
  List.map
    (fun cell -> (cell "target", renderer_of_row cell))
    (Golden.rows "targets.golden")

let renderer target = List.assoc target renderers
let kernels = Hashtbl.create 16

let kernel name =
  match Hashtbl.find_opt kernels name with
  | Some k -> k
  | None ->
      let k = Golden.sink (name ^ ".golden") in
      Hashtbl.add kernels name k;
      k

(* A kernel as the search is handed it: its weak output axes made global. *)
let scheduled name target =
  let k = K.v (kernel name) (renderer target) in
  K.convert_loop_to_global k;
  k

(* tinygrad's goldens name kernels without colour, and keep no search. *)
let quietly ?(settings = []) f =
  Helpers.context
    (B (Helpers.no_color, true) :: B (Helpers.cachelevel, 0) :: settings)
    f

let opts = list Kernel_opts.opt

let bool_of_cell = function
  | "True" -> true
  | "False" -> false
  | c -> failf "no boolean %s" c

let position o =
  match List.find_index (Opt.equal o) Search.actions with
  | Some i -> i
  | None -> failf "%a is no action" Opt.pp o

let kernel_info prg =
  match Ops.arg (Ops.nth prg 0) with
  | Kernel k -> k
  | _ -> failf "the program %a has no kernel" Ops.pp prg

let binary prg =
  match Ops.arg (Ops.nth prg 3) with
  | Bytes b -> b
  | _ -> failf "the program %a has no binary" Ops.pp prg

let workgroups ~vars prg =
  match Ops.arg prg with
  | Program p ->
      Helpers.prod (List.map (fun s -> Ops.sym_infer s vars) p.global_size)
  | _ -> failf "%a is no program" Ops.pp prg

(* Measurements *)

type call = {
  prg : Ops.t;
  cold : bool;
  vars : (string * int) list;
  time : float;
}

(* The measurement [time], recording each sample that returns. *)
let recording time =
  let calls = ref [] in
  let prepare ~cold ~vars prg =
    let t = time ~vars prg in
    fun () ->
      calls := { prg; cold; vars; time = t } :: !calls;
      t
  in
  (prepare, fun () -> List.rev !calls)

(* The measurement of searches.golden: an optimum at two optimisations, each
   weighed by its action's position, scaled by the launch's workgroups. When
   [failing], a program whose actions' positions sum to a multiple of 3
   fails. *)
let golden_time ~failing ~vars prg =
  let positions = List.map position (kernel_info prg).applied_opts in
  if failing && positions <> [] && List.fold_left ( + ) 0 positions mod 3 = 0
  then failwith "failing measurement";
  let tm = 1e-3 *. Float.of_int (1 + (3 * abs (List.length positions - 2))) in
  let weigh tm p = tm *. (0.8 +. (Float.of_int (37 * p mod 41) /. 100.)) in
  let tm = List.fold_left weigh tm positions in
  tm *. (1. +. (Float.of_int (workgroups ~vars prg mod 5) /. 10.))

let search ?settings ?allow_test_size ~prepare amt k =
  quietly ?settings (fun () ->
      Search.beam_search ~prepare ?allow_test_size amt k)

(* The actions *)

let recorded_actions file =
  List.map (fun cell -> Kernel_opts.opt_of_cell (cell "opt")) (Golden.rows file)

let actions =
  test "the actions are tinygrad's, in order" (fun () ->
      equal opts (recorded_actions "actions.golden") Search.actions)

(* Candidates *)

let candidates =
  Golden.cases ~key:[ "kernel"; "target"; "max_up" ] "candidates.golden"
    (fun cell ->
      let max_up = int_of_string_opt (cell "max_up") in
      let k = scheduled (cell "kernel") (cell "target") in
      let acted = quietly (fun () -> Search.get_kernel_actions ?max_up k) in
      equal (list int)
        (List.map int_of_string (String.split_on_char ' ' (cell "actions")))
        (List.map fst acted))

(* Searches *)

let searched cell =
  let prepare, calls =
    recording (golden_time ~failing:(bool_of_cell (cell "failing")))
  in
  let k =
    search ~prepare
      ~allow_test_size:(bool_of_cell (cell "allow_test_size"))
      (int_of_string (cell "amt"))
      (scheduled (cell "kernel") (cell "target"))
  in
  equal opts (Kernel_opts.opts_of_cell (cell "opts")) (K.applied_opts k);
  equal ~msg:"measurements" int
    (int_of_string (cell "measurements"))
    (List.length (calls ()))

let searches =
  let rows = Golden.rows "searches.golden" in
  let name cell =
    String.concat " "
      (List.map
         (fun c -> c ^ "=" ^ cell c)
         [ "kernel"; "target"; "amt"; "failing"; "allow_test_size" ])
  in
  let quick cell = cell "target" = "clang" in
  group "a search chooses what tinygrad's chooses"
    [
      cases ~name "on the host" (List.filter quick rows) searched;
      cases ~tags:[ "slow" ] ~name "on GPUs"
        (List.filter (fun c -> not (quick c)) rows)
        searched;
    ]

(* Laws of the candidates *)

let small =
  [
    ("add_small", "clang");
    ("sum_rows", "metal");
    ("pad_7x7", "cuda");
    ("symbolic", "hip");
  ]

let name_of (kernel, target) = kernel ^ " on " ^ target

let candidate_laws =
  group "get_kernel_actions"
    [
      cases ~name:name_of "leaves its scheduler as it was" small
        (fun (kernel, target) ->
          let k = scheduled kernel target in
          let ast = K.ast k in
          ignore (Search.get_kernel_actions k);
          equal Uops.uop ast (K.ast k);
          equal opts [] (K.applied_opts k));
      cases ~name:name_of "makes each candidate by applying its action" small
        (fun (kernel, target) ->
          let k = scheduled kernel target in
          let applied (i, k') =
            if i = 0 then equal ~msg:"0" Uops.uop (K.ast k) (K.ast k')
            else
              equal ~msg:(string_of_int i) opts
                [ List.nth Search.actions (i - 1) ]
                (K.applied_opts k')
          in
          List.iter applied (Search.get_kernel_actions k));
      cases ~name:name_of "leaves out the scheduler itself without include_0"
        small (fun (kernel, target) ->
          let k = scheduled kernel target in
          equal (list int)
            (List.tl (List.map fst (Search.get_kernel_actions k)))
            (List.map fst (Search.get_kernel_actions ~include_0:false k)));
      cases ~name:name_of "leaves out more candidates under a smaller max_up"
        small (fun (kernel, target) ->
          let k = scheduled kernel target in
          let fewer = List.map fst (Search.get_kernel_actions ~max_up:2 k) in
          let all = List.map fst (Search.get_kernel_actions k) in
          List.iter
            (fun i -> is_true ~msg:(string_of_int i) (List.mem i all))
            fewer);
    ]

(* Opt correctness *)

(* The interpreter cannot evaluate a matrix multiply-accumulate, which a tensor
   core used in full makes. *)
let evaluable (_, k) =
  List.for_all
    (function Opt.Tc { use_tc = 1; _ } -> false | _ -> true)
    (K.applied_opts k)

(* The kernel that the choices [picks] among its candidates make of [k]. *)
let rec picked k = function
  | [] -> k
  | pick :: picks -> (
      match
        List.filter evaluable (Search.get_kernel_actions ~include_0:false k)
      with
      | [] -> k
      | acted ->
          picked (snd (List.nth acted (pick mod List.length acted))) picks)

let keeps_writes =
  let kernels =
    [ "add_small"; "sum_rows"; "variable_rows"; "symbolic"; "pad_7x7" ]
  in
  let targets = List.map fst renderers in
  let pp ppf (kernel, target, picks) =
    Format.fprintf ppf "%s on %s, picks %s" kernel target
      (String.concat " " (List.map string_of_int picks))
  in
  prop ~count:30
    "every kernel a search can choose writes what its kernel writes"
    Gen.(
      with_pp pp
        (triple (of_list kernels) (of_list targets)
           (list ~size:(int_range 1 3) nat)))
    (fun (kernel_name, target, picks) ->
      let k = picked (scheduled kernel_name target) picks in
      equal (list Kernel_opts.write)
        (Kernel_opts.writes (kernel kernel_name))
        (Kernel_opts.writes (quietly (fun () -> K.get_optimized_ast k))))

(* Every kernel two actions make, as the old optimisation fuzzer checked. *)
let keeps_writes_exhaustively =
  let pairs =
    List.concat_map
      (fun kernel -> [ (kernel, "clang"); (kernel, "metal") ])
      [ "add_small"; "sum_rows"; "symbolic"; "pad_7x7" ]
  in
  let candidates k =
    List.filter evaluable (Search.get_kernel_actions ~include_0:false k)
  in
  cases ~tags:[ "slow" ] ~name:name_of
    "every kernel two actions make writes what its kernel writes" pairs
    (fun (kernel_name, target) ->
      let expected = Kernel_opts.writes (kernel kernel_name) in
      let keeps (_, k) =
        equal
          ~msg:
            (Format.asprintf "%a"
               (Format.pp_print_list Opt.pp)
               (K.applied_opts k))
          (list Kernel_opts.write) expected
          (Kernel_opts.writes (quietly (fun () -> K.get_optimized_ast k)))
      in
      List.iter
        (fun c -> List.iter keeps (candidates (snd c)))
        (candidates (scheduled kernel_name target)))

(* Laws of the search *)

(* A deterministic measurement drawn from [seed]: a program's time depends on
   its kernel's optimisations alone. *)
let drawn seed opts =
  1e-3
  +. Float.of_int (Hashtbl.seeded_hash seed (List.map position opts) mod 1000)
     *. 1e-6

let drawn_time seed ~vars:_ prg = drawn seed (kernel_info prg).applied_opts

(* A measurement by the number of optimisations [n]: the [n]th of [times], or
   its last for more. *)
let by_depth times ~vars:_ prg =
  let n = List.length (kernel_info prg).applied_opts in
  List.nth times (min n (List.length times - 1))

let depth c = List.length (kernel_info c.prg).applied_opts

(* How many times each program of [n] optimisations was measured. *)
let measured_times n calls =
  let counts = Hashtbl.create 16 in
  List.iter
    (fun c ->
      if depth c = n then
        let b = binary c.prg in
        Hashtbl.replace counts b
          (1 + Option.value ~default:0 (Hashtbl.find_opt counts b)))
    calls;
  Hashtbl.fold (fun _ n l -> n :: l) counts []

let rounds =
  let min_progress = 0.01 /. 1e6 in
  let searched times =
    let prepare, calls = recording (by_depth times) in
    let k = search ~prepare 1 (scheduled "sum_rows" "clang") in
    (k, calls ())
  in
  group "rounds"
    [
      test
        "a candidate is measured three times unless slower than three times \
         the best" (fun () ->
          let _, calls = searched [ 0.; 0.25; 0.75 ] in
          is_true ~msg:"measured" (measured_times 2 calls <> []);
          List.iter (equal int 3) (measured_times 2 calls);
          let _, calls = searched [ 0.; 0.25; 0.875 ] in
          List.iter (equal int 1) (measured_times 2 calls));
      test "a search goes on while it gains BEAM_MIN_PROGRESS or more"
        (fun () ->
          let _, calls =
            searched [ 0.; 2. *. min_progress; min_progress; 1. ]
          in
          is_true ~msg:"a third round"
            (List.exists (fun c -> depth c = 3) calls));
      test "a search that gains nothing keeps its best kernel" (fun () ->
          let k, _ = searched [ 0.; 1e-3; 1e-3 ] in
          equal int 1 (List.length (K.applied_opts k)));
    ]

let chooses_the_fastest =
  let pp ppf (kernel, amt, seed) =
    Format.fprintf ppf "%s, width %d, seed %d" kernel amt seed
  in
  prop ~tags:[ "slow" ] ~count:20
    "a search chooses the fastest program it measured"
    Gen.(
      with_pp pp
        (triple
           (of_list [ "add_small"; "sum_rows"; "variable_rows" ])
           (int_range 1 3) nat))
    (fun (kernel, amt, seed) ->
      let prepare, calls = recording (drawn_time seed) in
      let k = scheduled kernel "clang" in
      let chosen = search ~prepare amt k in
      match calls () with
      | [] -> equal Uops.uop (K.ast k) (K.ast chosen)
      | calls ->
          let fastest =
            List.fold_left (fun t c -> Float.min t c.time) infinity calls
          in
          equal float_exact fastest (drawn seed (K.applied_opts chosen)))

let binaries calls = List.map (fun c -> binary c.prg) calls

let is_deterministic =
  test "a search measures and chooses alike on one domain and on several"
    (fun () ->
      let run parallel =
        let prepare, calls = recording (golden_time ~failing:false) in
        let k =
          search
            ~settings:[ B (Helpers.parallel, parallel) ]
            ~prepare 2
            (scheduled "sum_rows" "clang")
        in
        (K.applied_opts k, binaries (calls ()))
      in
      let serial_opts, serial = run 0 and parallel_opts, parallel = run 4 in
      equal opts serial_opts parallel_opts;
      equal (list string) serial parallel)

let asks_cold_midpoints =
  test "a search asks for cold runs with each variable at its bounds' middle"
    (fun () ->
      let prepare, calls = recording (golden_time ~failing:false) in
      ignore (search ~prepare 1 (scheduled "variable_rows" "metal"));
      let calls = calls () in
      is_true ~msg:"measured" (calls <> []);
      List.iter
        (fun c ->
          is_true ~msg:"cold" c.cold;
          equal (list (pair string int)) [ ("n", 8) ] c.vars)
        calls)

let storage_placed =
  test "a candidate's storage is placed on its renderer's device" (fun () ->
      let prepare, calls = recording (golden_time ~failing:false) in
      ignore (search ~prepare 1 (scheduled "sum_rows" "metal"));
      let placed prg =
        List.iter
          (fun u ->
            match (Ops.op u, Ops.arg u) with
            | Op.Param, Param p when p.addrspace <> Some Dtype.Alu ->
                is_true ~msg:"on METAL" (p.device = Some (Ops.Single "METAL"))
            | _ -> ())
          (Ops.toposort (Ops.nth prg 0))
      in
      List.iter (fun c -> placed c.prg) (calls ()))

(* The launch each candidate of [kernel] is measured with, under a measurement
   that times them alike, so that the search ends after its first round. *)
let launches ?allow_test_size kernel =
  let prepare, calls = recording (fun ~vars:_ _ -> 1e-3) in
  ignore (search ?allow_test_size ~prepare 1 (scheduled kernel "metal"));
  List.map
    (fun c ->
      ( (kernel_info c.prg).applied_opts,
        match Ops.arg c.prg with
        | Program p -> List.map (fun s -> Ops.sym_infer s c.vars) p.global_size
        | _ -> failf "%a is no program" Ops.pp c.prg ))
    (calls ())

let launch_of kernel opt =
  match List.assoc_opt [ opt ] (launches kernel) with
  | Some sizes -> sizes
  | None -> failf "%a was not measured" Opt.pp opt

(* A time proportional to the workgroups launched, so that a program measured on
   fewer and scaled up takes the time of all of them. *)
let per_workgroup ~vars prg =
  Float.of_int (workgroups ~vars prg)
  *. 1e-9
  *. golden_time ~failing:false ~vars prg

let test_size =
  group "measuring on fewer workgroups"
    [
      test "halves the last size above 16 down to 65536 workgroups" (fun () ->
          let split axis amount =
            Opt.Split { axis; amount; target = Upcast; top = false }
          in
          equal (list int) [ 512; 16; 8 ] (launch_of "add_3d" (split 0 2));
          equal (list int) [ 512; 8; 16 ] (launch_of "add_3d" (split 1 2)));
      test "keeps a launch of 65536 workgroups" (fun () ->
          equal (list int) [ 65536; 1; 1 ]
            (launch_of "add_large"
               (Opt.Split { axis = 0; amount = 16; target = Local; top = true })));
      test "launches no measured program on more than 65536 workgroups"
        (fun () ->
          let prepare, calls = recording (golden_time ~failing:false) in
          ignore (search ~prepare 1 (scheduled "add_large" "metal"));
          List.iter
            (fun c ->
              satisfies ~claim:"at most 65536" int
                (fun n -> n <= 65536)
                (workgroups ~vars:c.vars c.prg))
            (calls ()));
      test "launches them all without allow_test_size" (fun () ->
          let prepare, calls = recording (golden_time ~failing:false) in
          ignore
            (search ~allow_test_size:false ~prepare 1
               (scheduled "add_large" "metal"));
          is_true
            (List.exists
               (fun c -> workgroups ~vars:c.vars c.prg > 65536)
               (calls ())));
      slow "chooses as measuring them all does, for a time per workgroup"
        (fun () ->
          let choose allow_test_size =
            let prepare, _ = recording per_workgroup in
            K.applied_opts
              (search ~allow_test_size ~prepare 1
                 (scheduled "add_large" "metal"))
          in
          equal opts (choose false) (choose true));
    ]

let failures =
  group "a failing measurement"
    [
      test
        "drops its candidate, and a search of no timed candidate keeps its \
         kernel" (fun () ->
          let k = scheduled "sum_rows" "clang" in
          let chosen =
            search ~prepare:(fun ~cold:_ ~vars:_ _ -> failwith "no device") 1 k
          in
          equal Uops.uop (K.ast k) (K.ast chosen);
          equal opts [] (K.applied_opts chosen));
      test "raising anything but Failure is raised by the search" (fun () ->
          raises Exit (fun () ->
              search
                ~prepare:(fun ~cold:_ ~vars:_ _ -> raise Exit)
                1
                (scheduled "sum_rows" "clang")));
    ]

let width =
  test "a search of no width is refused" (fun () ->
      raises_match (Exn.invalid_arg ?substring:None) (fun () ->
          search
            ~prepare:(fun ~cold:_ ~vars:_ _ -> fun () -> 1.)
            0
            (scheduled "sum_rows" "clang")))

(* Compilation *)

(* A Clang renderer whose compiler rejects each source whose length [every]
   divides (default 3), with [reject ()], and the count of rejections. It
   targets an architecture of its own, so that no other test's programs are
   reused. *)
let rejecting ?(every = 3) reject =
  let rejected = ref 0 in
  let compile src =
    if String.length src mod every = 0 then (
      incr rejected;
      reject ())
    else src
  in
  let t =
    {
      Helpers.Target.device = "CPU";
      renderer = "CLANG";
      arch = "x86_64,znver2";
      interface = "";
      indices = "";
    }
  in
  ( Renderer.with_compiler (Renderer.Compiler.v compile) (Cstyle.clang t),
    rejected )

let rejected_by_compiler () = raise (Renderer.Compiler.Compile_error "rejected")

let scheduled_for ren name =
  let k = K.v (kernel name) ren in
  K.convert_loop_to_global k;
  k

let uncompilable =
  let dropped reject () =
    let ren, rejected = rejecting reject in
    let prepare, calls = recording (golden_time ~failing:false) in
    ignore (search ~prepare 1 (scheduled_for ren "sum_rows"));
    is_true ~msg:"rejected some" (!rejected > 0);
    is_true ~msg:"measured others" (calls () <> [])
  in
  group "a candidate that does not compile is dropped"
    [
      test "when its compiler rejects it" (dropped rejected_by_compiler);
      test "when its compilation fails"
        (dropped (fun () -> failwith "rejected"));
    ]

(* Compilation is reusable: a kernel a round already compiled is not compiled
   again when a later round's action path makes it anew. The measurement favours
   the swap of the first two global axes, then its double: the search goes [k]
   -> [k + s] -> [k + s + s] (= [k]) -> [k + s], and the last candidate is the
   first round's kernel. The compiler counts every source it is handed: 58
   compilations for 50 distinct kernels, where a search that compiled every
   round's candidates anew takes 87. Its target is its own, so that no other
   test's programs are reused. *)
let compiled_once =
  let swap = Opt.Swap { axis = 0; with_axis = 1 } in
  let time ~vars:_ prg =
    match (kernel_info prg).applied_opts with
    | [ o ] when Opt.equal o swap -> 1e-3
    | [ o; o' ] when Opt.equal o swap && Opt.equal o' swap -> 5e-4
    | _ -> 1.
  in
  test "a kernel compiled in one round is not compiled again in another"
    (fun () ->
      let counts = Hashtbl.create 64 in
      let compile src =
        Hashtbl.replace counts src
          (1 + Option.value ~default:0 (Hashtbl.find_opt counts src));
        src
      in
      let t =
        {
          Helpers.Target.device = "METAL";
          renderer = "METAL";
          arch = "Apple10";
          interface = "";
          indices = "";
        }
      in
      let ren =
        Renderer.with_compiler (Renderer.Compiler.v compile) (Cstyle.metal t)
      in
      let prepare, _ = recording time in
      ignore (search ~prepare 1 (scheduled_for ren "add_3d"));
      let counts = Hashtbl.fold (fun _ n acc -> n :: acc) counts [] in
      equal ~msg:"distinct kernels" int 50 (List.length counts);
      equal ~msg:"compilations" int 58 (List.fold_left ( + ) 0 counts))

(* CACHEDB is a directory of the sandbox (see dune). *)
let cache =
  let cached ?(ignore = false) prepare =
    search
      ~settings:
        [ B (Helpers.cachelevel, 1); B (Helpers.ignore_beam_cache, ignore) ]
      ~prepare 2
      (scheduled "sum_rows" "metal")
  in
  let unmeasured ~cold:_ ~vars:_ _ = fail "the search measured" in
  test "a search kept in the cache measures nothing, unless it is ignored"
    (fun () ->
      let prepare, _ = recording (golden_time ~failing:false) in
      let first = K.applied_opts (cached prepare) in
      equal opts first (K.applied_opts (cached unmeasured));
      let prepare, calls = recording (golden_time ~failing:false) in
      equal opts first (K.applied_opts (cached ~ignore:true prepare));
      is_true ~msg:"measured again" (calls () <> []))

(* A measurement that favours tensor cores, then swaps, with an optimum at two
   optimisations. *)
let kinds_time ~vars:_ prg =
  let opts = (kernel_info prg).applied_opts in
  let weight = function Opt.Tc _ -> 0.5 | Opt.Swap _ -> 0.6 | _ -> 1.2 in
  let depth = 1 + (3 * max 0 (List.length opts - 2)) in
  List.fold_left (fun t o -> t *. weight o) (1e-3 *. Float.of_int depth) opts

let cache_kinds =
  slow "the cache keeps tensor cores and swaps" (fun () ->
      let cached prepare =
        K.applied_opts
          (search
             ~settings:[ B (Helpers.cachelevel, 1) ]
             ~prepare 1
             (scheduled "matmul_half" "metal"))
      in
      let prepare, _ = recording kinds_time in
      let first = cached prepare in
      is_true ~msg:"a tensor core"
        (List.exists (function Opt.Tc _ -> true | _ -> false) first);
      is_true ~msg:"a swap"
        (List.exists (function Opt.Swap _ -> true | _ -> false) first);
      equal opts first
        (cached (fun ~cold:_ ~vars:_ _ -> fail "the search measured")))

(* Printing *)

let progress =
  test "a search prints its progress under DEBUG=2" (fun () ->
      ignore (output ());
      let prepare, _ = recording (golden_time ~failing:false) in
      ignore
        (search
           ~settings:[ B (Helpers.debug, 2) ]
           ~prepare 1
           (scheduled "add_small" "clang"));
      in_order
        ~subs:[ "from   1 ->   1 actions"; "   1/"; "actions\027[K" ]
        (output ()))

let quiet =
  test "a search prints nothing by default" (fun () ->
      ignore (output ());
      let prepare, _ = recording (golden_time ~failing:true) in
      ignore (search ~prepare 2 (scheduled "add_small" "clang"));
      ignore
        (quietly (fun () ->
             Search.get_kernel_actions ~max_up:1 (scheduled "sum_rows" "metal")));
      equal string "" (output ()))

let failure_printed =
  test "a compilation's failure is printed under DEBUG=4" (fun () ->
      let ren, _ = rejecting (fun () -> failwith "rejected") in
      ignore (output ());
      let prepare, _ = recording (golden_time ~failing:false) in
      ignore
        (search
           ~settings:[ B (Helpers.debug, 4) ]
           ~prepare 1
           (scheduled_for ren "sum_rows"));
      contains ~sub:"Failure(\"rejected\")" (output ()))

(* The environment *)

let environment =
  [
    ("BEAM_PADTO", "1");
    ("TC", "2");
    ("TC_OPT", "0");
    ("BEAM_STRICT_MODE", "1");
    ("BEAM_UOPS_MAX", "43");
    ("BEAM_LOG_SURPASS_MAX", "1");
    ("BEAM_DEBUG", "2");
  ]

let instructions prg = List.length (Ops.src (Ops.nth prg 1))

(* The environment is read once per process: the suite runs this group in a
   process of its own (see dune). *)
let under_environment =
  group ~tags:[ "environment" ]
    (String.concat " " (List.map (fun (n, v) -> n ^ "=" ^ v) environment))
    [
      test "the suite runs this group with them" (fun () ->
          List.iter
            (fun (name, value) ->
              equal ~msg:name (option string) (Some value) (Sys.getenv_opt name))
            environment);
      test "the actions are tinygrad's, pads included" (fun () ->
          equal opts (recorded_actions "actions_padto.golden") Search.actions);
      test "a candidate of BEAM_UOPS_MAX instructions or more is dropped"
        (fun () ->
          ignore (output ());
          let prepare, calls = recording (golden_time ~failing:false) in
          ignore (search ~prepare 1 (scheduled "sum_rows" "clang"));
          List.iter
            (fun c ->
              satisfies ~claim:"fewer than 43" int
                (fun n -> n < 43)
                (instructions c.prg))
            (calls ());
          contains ~sub:"too many uops" (output ()));
      test "a compilation that raises is raised under BEAM_STRICT_MODE"
        (fun () ->
          let ren, _ = rejecting ~every:1 rejected_by_compiler in
          raises_match
            (function Renderer.Compiler.Compile_error _ -> true | _ -> false)
            (fun () ->
              search
                ~prepare:(fun ~cold:_ ~vars:_ _ -> fun () -> 1.)
                1
                (scheduled_for ren "sum_rows")));
      test
        "a search prints the kernel, failures and its choice under BEAM_DEBUG"
        (fun () ->
          ignore (output ());
          let prepare, _ = recording (golden_time ~failing:true) in
          ignore (search ~prepare 2 (scheduled "add_small" "clang"));
          in_order
            ~subs:
              [
                "BEAM_SEARCH:"; "BEAM failed for opts"; "BEAM_SEARCH: final tm=";
              ]
            (output ()));
      Golden.cases ~key:[ "kernel"; "target"; "amt" ]
        "searches_environment.golden" (fun cell ->
          let prepare, calls = recording (golden_time ~failing:false) in
          let k =
            search ~prepare
              (int_of_string (cell "amt"))
              (scheduled (cell "kernel") (cell "target"))
          in
          equal opts (Kernel_opts.opts_of_cell (cell "opts")) (K.applied_opts k);
          equal ~msg:"measurements" int
            (int_of_string (cell "measurements"))
            (List.length (calls ())));
      test
        "a candidate of more than 1000 times the fewest operations is not \
         measured" (fun () ->
          let is_pad = function Opt.Padto _ -> true | _ -> false in
          let k = scheduled "transpose_33" "metal" in
          let pads =
            List.filter
              (fun (_, k') -> List.exists is_pad (K.applied_opts k'))
              (quietly (fun () -> Search.get_kernel_actions k))
          in
          is_true ~msg:"a pad is a candidate" (pads <> []);
          ignore (output ());
          let prepare, calls = recording (golden_time ~failing:false) in
          ignore (search ~prepare 1 k);
          contains ~sub:"too much compute" (output ());
          is_true ~msg:"measured" (calls () <> []);
          List.iter
            (fun c ->
              is_true ~msg:"a pad was measured"
                (not (List.exists is_pad (kernel_info c.prg).applied_opts)))
            (calls ()));
      test "kernels of too many lanes are reported under BEAM_LOG_SURPASS_MAX"
        (fun () ->
          ignore (output ());
          ignore
            (quietly (fun () ->
                 Search.get_kernel_actions ~max_up:1
                   (scheduled "sum_rows" "metal")));
          contains ~sub:"too many upcast/local" (output ()));
    ]

let () =
  exit
    (Windtrap.run "Tolk.Search"
       [
         actions;
         under_environment;
         candidates;
         searches;
         candidate_laws;
         keeps_writes;
         keeps_writes_exhaustively;
         chooses_the_fastest;
         is_deterministic;
         asks_cold_midpoints;
         test_size;
         failures;
         width;
         cache;
         cache_kinds;
         uncompilable;
         compiled_once;
         progress;
         quiet;
         failure_printed;
         storage_placed;
         rounds;
       ])
