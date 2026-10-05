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
  Setting.context
    (B (Setting.no_color, true) :: B (Setting.cachelevel, 0) :: settings)
    f

let opts = list Kernel_opts.opt

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

let binary prg =
  match Ops.arg (Ops.nth prg 3) with
  | Bytes b -> b
  | _ -> failf "the program %a has no binary" Ops.pp prg

let workgroups ~vars prg =
  match Ops.arg prg with
  | Program p ->
      Helpers.prod (List.map (fun s -> Ops.sym_infer s vars) p.global_size)
  | _ -> failf "%a is no program" Ops.pp prg

(* Timings *)

type call = { prg : Ops.t; vars : (string * int) list; time : float }

(* What a timing was applied to: the kernels, the programs prepared and the
   samples that returned, in order. *)
type record = { kernels : Ops.t list; prepared : Ops.t list; calls : call list }

(* The timing whose samples of a program take [sample ~vars prg i], [i] the
   number of samples of it before, and its record. *)
let sampling sample =
  let kernels = ref [] and prepared = ref [] and calls = ref [] in
  let time ~vars kernel =
    kernels := kernel :: !kernels;
    fun prg ->
      prepared := prg :: !prepared;
      let i = ref 0 in
      fun () ->
        let t = sample ~vars prg !i in
        incr i;
        calls := { prg; vars; time = t } :: !calls;
        t
  in
  let record () =
    {
      kernels = List.rev !kernels;
      prepared = List.rev !prepared;
      calls = List.rev !calls;
    }
  in
  (time, record)

(* The timing whose samples of a program all take [time ~vars prg], and the
   samples that returned. *)
let recording time =
  let time, record = sampling (fun ~vars prg _ -> time ~vars prg) in
  (time, fun () -> (record ()).calls)

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

let search ?settings ?allow_test_size ~time amt k =
  quietly ?settings (fun () -> Search.beam_search ~time ?allow_test_size amt k)

(* The actions *)

let recorded_actions file =
  List.map (fun cell -> Kernel_opts.opt_of_cell (cell "opt")) (Golden.rows file)

let actions =
  test "the actions are tinygrad's, in order" (fun () ->
      equal opts (recorded_actions "actions.golden") (Search.actions ()))

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
  let time, calls =
    recording (golden_time ~failing:(bool_of_cell (cell "failing")))
  in
  let k =
    search ~time
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
                [ List.nth (Search.actions ()) (i - 1) ]
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
    let time, calls = recording (by_depth times) in
    let k = search ~time 1 (scheduled "sum_rows" "clang") in
    (k, calls ())
  in
  group "rounds"
    [
      test
        "a candidate is measured three times unless slower than three times \
         the incumbent" (fun () ->
          let _, calls = searched [ 1.; 0.25; 0.75 ] in
          is_true ~msg:"measured" (measured_times 2 calls <> []);
          List.iter (equal int 3) (measured_times 2 calls);
          let _, calls = searched [ 1.; 0.25; 0.875 ] in
          List.iter (equal int 1) (measured_times 2 calls));
      test "a search goes on while it gains more than BEAM_MIN_PROGRESS"
        (fun () ->
          let _, calls =
            searched [ 1.; 3. *. min_progress; min_progress; 1. ]
          in
          is_true ~msg:"a third round"
            (List.exists (fun c -> depth c = 3) calls));
      test "a search that gains BEAM_MIN_PROGRESS exactly stops" (fun () ->
          let k, calls = searched [ 1.; 2. *. min_progress; min_progress ] in
          is_true ~msg:"a second round"
            (List.exists (fun c -> depth c = 2) calls);
          equal int 1 (List.length (K.applied_opts k)));
      test "a search that gains nothing answers its kernel" (fun () ->
          let k, _ = searched [ 1e-3; 1e-3 ] in
          equal opts [] (K.applied_opts k));
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
      let time, calls = recording (drawn_time seed) in
      let k = scheduled kernel "clang" in
      let chosen = search ~time amt k in
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
        let time, calls = recording (golden_time ~failing:false) in
        let k =
          search
            ~settings:[ B (Setting.parallel, parallel) ]
            ~time 2
            (scheduled "sum_rows" "clang")
        in
        (K.applied_opts k, binaries (calls ()))
      in
      let serial_opts, serial = run 0 and parallel_opts, parallel = run 4 in
      equal opts serial_opts parallel_opts;
      equal (list string) serial parallel)

let midpoints =
  test "a search times with each variable at its bounds' middle" (fun () ->
      let time, calls = recording (golden_time ~failing:false) in
      ignore (search ~time 1 (scheduled "variable_rows" "metal"));
      let calls = calls () in
      is_true ~msg:"measured" (calls <> []);
      List.iter
        (fun c -> equal (list (pair string int)) [ ("n", 8) ] c.vars)
        calls)

(* The timing's stages *)

let small_searches =
  Gen.(
    with_pp
      (fun ppf (kernel, target, amt) ->
        Format.fprintf ppf "%s on %s, width %d" kernel target amt)
      (triple
         (of_list [ "add_small"; "sum_rows"; "variable_rows"; "symbolic" ])
         (of_list [ "clang"; "metal" ])
         (int_range 1 3)))

let applies_the_kernel_once =
  prop ~count:10 "a search applies its timing once, to its kernel"
    small_searches (fun (kernel, target, amt) ->
      let time, record =
        sampling (fun ~vars prg _ -> golden_time ~failing:false ~vars prg)
      in
      let k = scheduled kernel target in
      ignore (search ~time amt k);
      equal (list Uops.uop) [ K.ast k ] (record ()).kernels)

(* Samples drawn from [seed]: a program's base time, from a microsecond to two
   milliseconds, times up to one and a half, so that the samples of programs of
   near base times overlap. *)
let spread seed ~vars:_ prg i =
  let positions = List.map position (kernel_info prg).applied_opts in
  let base = 1 + (Hashtbl.seeded_hash seed positions mod 2000) in
  let noise = 100 + (Hashtbl.seeded_hash seed (positions, i) mod 51) in
  1e-8 *. Float.of_int (base * noise)

let least = List.fold_left Float.min infinity

(* The samples of each program prepared, by binary, in order. *)
let samples_of r =
  List.map
    (fun p ->
      ( p,
        List.filter_map
          (fun c -> if binary c.prg = binary p then Some c.time else None)
          r.calls ))
    r.prepared

(* The least sample of a program of [depth] optimisations: the incumbent of the
   round that times programs of one more. *)
let incumbent r depth =
  least
    (List.concat_map
       (fun (p, ts) ->
         if List.length (kernel_info p).applied_opts = depth then ts else [])
       (samples_of r))

let early_stops =
  let pp ppf (kernel, amt, seed) =
    Format.fprintf ppf "%s, width %d, seed %d" kernel amt seed
  in
  prop ~count:20
    "each program is prepared once and sampled until its least exceeds three \
     times the incumbent's, three times at most"
    Gen.(
      with_pp pp
        (triple
           (of_list [ "add_small"; "sum_rows"; "variable_rows" ])
           (int_range 1 3) nat))
    (fun (kernel, amt, seed) ->
      let time, record = sampling (spread seed) in
      ignore
        (search ~allow_test_size:false ~time amt (scheduled kernel "clang"));
      let r = record () in
      equal ~msg:"prepared once" (list string)
        (List.sort_uniq String.compare (List.map binary r.prepared))
        (List.sort String.compare (List.map binary r.prepared));
      List.iter
        (fun (p, ts) ->
          let depth = List.length (kernel_info p).applied_opts in
          let stop =
            if depth = 0 then infinity else 3. *. incumbent r (depth - 1)
          in
          let n = List.length ts in
          let msg =
            Format.asprintf "%a"
              (Format.pp_print_list Opt.pp)
              (kernel_info p).applied_opts
          in
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
        (samples_of r))

(* A timing of constant samples that raises [e] when applied to the kernel, when
   preparing a program, or when sampling one. *)
let raising_at stage e ~vars:_ _ =
  if stage = `Kernel then raise e;
  fun _ ->
    if stage = `Prepare then raise e;
    fun () -> if stage = `Sample then raise e else 1e-3

let stages =
  [
    ("the kernel", `Kernel); ("a preparation", `Prepare); ("a sample", `Sample);
  ]

let exceptions =
  group "a timing that raises"
    [
      cases ~name:fst "Exit is raised by the search, from" stages
        (fun (_, stage) ->
          raises Exit (fun () ->
              search ~time:(raising_at stage Exit) 1
                (scheduled "sum_rows" "clang")));
      test "Failure from the kernel is raised by the search" (fun () ->
          raises (Failure "no device") (fun () ->
              search
                ~time:(raising_at `Kernel (Failure "no device"))
                1
                (scheduled "sum_rows" "clang")));
      cases ~name:fst
        "Failure drops its program, and a search of no timed candidate keeps \
         its kernel, from"
        (List.tl stages) (fun (_, stage) ->
          let k = scheduled "sum_rows" "clang" in
          let chosen =
            search ~time:(raising_at stage (Failure "no device")) 1 k
          in
          equal Uops.uop (K.ast k) (K.ast chosen);
          equal opts [] (K.applied_opts chosen));
    ]

(* The incumbent *)

let incumbent_first =
  test "the kernel is sampled three times before any candidate" (fun () ->
      let time, record =
        sampling (fun ~vars prg _ -> by_depth [ 1e-6; 1e-3 ] ~vars prg)
      in
      let k = scheduled "sum_rows" "clang" in
      let chosen = search ~time 2 k in
      let depths = List.map depth (record ()).calls in
      equal (list int) [ 0; 0; 0 ] (List.filteri (fun i _ -> i < 3) depths);
      equal ~msg:"its samples" int 3
        (List.length (List.filter (Int.equal 0) depths));
      equal opts [] (K.applied_opts chosen))

(* A Clang renderer whose compiler rejects its first source, the kernel's in a
   search, for the architecture [arch] of its own. *)
let rejecting_first arch =
  let first = Atomic.make true in
  let compile src =
    if Atomic.exchange first false then
      raise (Renderer.Compiler.Compile_error "rejected")
    else src
  in
  let t =
    {
      Helpers.Target.device = "CPU";
      renderer = "CLANG";
      arch;
      interface = "";
      indices = "";
    }
  in
  Renderer.with_compiler (Renderer.Compiler.v compile) (Cstyle.clang t)

let uncompiled_kernel =
  test "a search whose kernel does not compile progresses on any candidate"
    (fun () ->
      let time, record = sampling (fun ~vars:_ _ _ -> 1.) in
      let k = K.v (kernel "sum_rows") (rejecting_first "x86_64,znver4") in
      K.convert_loop_to_global k;
      let chosen = search ~time 1 k in
      let r = record () in
      equal ~msg:"the kernel's samples" int 0
        (List.length (List.filter (fun c -> depth c = 0) r.calls));
      equal int 1 (List.length (K.applied_opts chosen)))

(* The fastest program of [depth] optimisations and its samples, the first of
   ties, if any. *)
let fastest r depth =
  List.fold_left
    (fun best (p, ts) ->
      if List.length (kernel_info p).applied_opts <> depth then best
      else
        match best with
        | Some (_, bs) when least bs <= least ts -> best
        | _ -> Some (p, ts))
    None (samples_of r)

let most = List.fold_left Float.max neg_infinity

let progress_law =
  let min_progress = 0.01 /. 1e6 in
  let pp ppf (kernel, amt, seed) =
    Format.fprintf ppf "%s, width %d, seed %d" kernel amt seed
  in
  prop ~count:30
    "a round goes on iff each sample of its fastest beats each of the \
     incumbent's by more than BEAM_MIN_PROGRESS"
    Gen.(
      with_pp pp
        (triple
           (of_list [ "add_small"; "sum_rows"; "variable_rows" ])
           (int_range 1 3) nat))
    (fun (kernel, amt, seed) ->
      let time, record = sampling (spread seed) in
      let chosen =
        search ~allow_test_size:false ~time amt (scheduled kernel "clang")
      in
      let r = record () in
      let rounds = List.length (K.applied_opts chosen) in
      let progresses d =
        match fastest r d with
        | None -> false
        | Some (_, ts) -> most ts +. min_progress < incumbent r (d - 1)
      in
      for d = 1 to rounds do
        is_true ~msg:(Printf.sprintf "round %d progressed" d) (progresses d)
      done;
      is_true
        ~msg:(Printf.sprintf "round %d stopped" (rounds + 1))
        (not (progresses (rounds + 1)));
      (match fastest r rounds with
      | Some (p, _) ->
          equal opts (kernel_info p).applied_opts (K.applied_opts chosen)
      | None -> equal int 0 rounds);
      cover "a round that progressed" (rounds > 0);
      cover "a timed round that stopped" (fastest r (rounds + 1) <> None))

let storage_placed =
  test "a candidate's storage is placed on its renderer's device" (fun () ->
      let time, calls = recording (golden_time ~failing:false) in
      ignore (search ~time 1 (scheduled "sum_rows" "metal"));
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
  let time, calls = recording (fun ~vars:_ _ -> 1e-3) in
  ignore (search ?allow_test_size ~time 1 (scheduled kernel "metal"));
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
          let time, calls = recording (golden_time ~failing:false) in
          ignore (search ~time 1 (scheduled "add_large" "metal"));
          List.iter
            (fun c ->
              satisfies ~claim:"at most 65536" int
                (fun n -> n <= 65536)
                (workgroups ~vars:c.vars c.prg))
            (calls ()));
      test "launches them all without allow_test_size" (fun () ->
          let time, calls = recording (golden_time ~failing:false) in
          ignore
            (search ~allow_test_size:false ~time 1
               (scheduled "add_large" "metal"));
          is_true
            (List.exists
               (fun c -> workgroups ~vars:c.vars c.prg > 65536)
               (calls ())));
      slow "chooses as measuring them all does, for a time per workgroup"
        (fun () ->
          let choose allow_test_size =
            let time, _ = recording per_workgroup in
            K.applied_opts
              (search ~allow_test_size ~time 1 (scheduled "add_large" "metal"))
          in
          equal opts (choose false) (choose true));
    ]

let width =
  test "a search of no width is refused" (fun () ->
      raises_match (Exn.invalid_arg ?substring:None) (fun () ->
          search
            ~time:(fun ~vars:_ _ _ () -> 1.)
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

(* A Metal renderer whose compiler counts the times it compiles each source, for
   the family [arch] of its own, so that no other test's programs are reused. *)
let counting arch =
  let compiled = Hashtbl.create 64 and lock = Mutex.create () in
  let compile src =
    Mutex.protect lock (fun () ->
        Hashtbl.replace compiled src
          (1 + Option.value ~default:0 (Hashtbl.find_opt compiled src)));
    src
  in
  let t =
    {
      Helpers.Target.device = "METAL";
      renderer = "METAL";
      arch;
      interface = "";
      indices = "";
    }
  in
  ( Renderer.with_compiler (Renderer.Compiler.v compile) (Cstyle.metal t),
    fun () -> Hashtbl.fold (fun src n l -> (src, n) :: l) compiled [] )

(* A search of width 2 whose beam after its first round is an upcast and a swap,
   and whose second round gains nothing, compiles its kernel and the candidates
   of two rounds. Its second round reaches kernels by both orders of the two, as
   [[upcast; swap]] and [[swap; upcast']], which are equal kernels. *)
let compiles_once =
  let upcast = Opt.Split { axis = 1; amount = 2; target = Upcast; top = false }
  and swap = Opt.Swap { axis = 0; with_axis = 1 } in
  let favoured ~vars:_ prg _ =
    match (kernel_info prg).applied_opts with
    | [] -> 1.
    | [ o ] when Opt.equal o upcast || Opt.equal o swap -> 1e-4
    | _ -> 1e-3
  in
  test "a search compiles each kernel once" (fun () ->
      let ren, compiled = counting "Apple7" in
      let k = scheduled_for ren "matmul_small" in
      let time, record = sampling favoured in
      ignore (search ~time 2 k);
      let applied =
        List.map (fun p -> (kernel_info p).applied_opts) (record ()).prepared
      in
      is_true ~msg:"the beam was timed"
        (List.mem [ upcast ] applied && List.mem [ swap ] applied);
      let acted o =
        let k' = K.copy k in
        ignore (K.apply_opt k' o);
        k'
      in
      let actions k =
        List.map snd
          (quietly (fun () -> Search.get_kernel_actions ~include_0:false k))
      in
      let distinct = Ops.Tbl.create 64 in
      List.iter
        (fun k -> Ops.Tbl.replace distinct (K.ast k) ())
        ((k :: actions k) @ actions (acted upcast) @ actions (acted swap));
      equal int (Ops.Tbl.length distinct)
        (List.fold_left (fun n (_, c) -> n + c) 0 (compiled ())))

let uncompilable =
  let dropped reject () =
    let ren, rejected = rejecting reject in
    let time, calls = recording (golden_time ~failing:false) in
    ignore (search ~time 1 (scheduled_for ren "sum_rows"));
    is_true ~msg:"rejected some" (!rejected > 0);
    is_true ~msg:"measured others" (calls () <> [])
  in
  group "a candidate that does not compile is dropped"
    [
      test "when its compiler rejects it" (dropped rejected_by_compiler);
      test "when its compilation fails"
        (dropped (fun () -> failwith "rejected"));
    ]

(* The cache *)

(* CACHEDB is a directory of the sandbox (see dune). *)
let cache =
  let cached ?(ignore = false) time =
    search
      ~settings:
        [ B (Setting.cachelevel, 1); B (Setting.ignore_beam_cache, ignore) ]
      ~time 2
      (scheduled "sum_rows" "metal")
  in
  let untimed ~vars:_ _ = fail "the search applied its timing" in
  test
    "a search kept in the cache applies nothing of its timing, unless it is \
     ignored" (fun () ->
      let time, _ = recording (golden_time ~failing:false) in
      let first = K.applied_opts (cached time) in
      equal opts first (K.applied_opts (cached untimed));
      let time, calls = recording (golden_time ~failing:false) in
      equal opts first (K.applied_opts (cached ~ignore:true time));
      is_true ~msg:"measured again" (calls () <> []))

(* A search kept under one value of a setting that shapes compilation is not
   used under another. *)
let cache_settings =
  test "a search kept under one setting measures again under another" (fun () ->
      let cached ?(settings = []) time =
        ignore
          (search
             ~settings:(B (Setting.cachelevel, 1) :: settings)
             ~time 2
             (scheduled "sum_rows" "metal"))
      in
      cached (fst (recording (golden_time ~failing:false)));
      let time, calls = recording (golden_time ~failing:false) in
      cached ~settings:[ B (Setting.transcendental, 2) ] time;
      greater int ~than:0 (List.length (calls ())))

(* A setting that its caller declares to reach output: the search knows nothing
   of it, yet keys on it. *)
let declared = Setting.int ~reach:Output "TOLK_TEST_SEARCH" 0

let cache_declared =
  test
    "a search kept under one value of a setting declared by its caller \
     measures again under another" (fun () ->
      let cached ?(settings = []) time =
        ignore
          (search
             ~settings:(B (Setting.cachelevel, 1) :: settings)
             ~time 2
             (scheduled "sum_rows" "metal"))
      in
      cached (fst (recording (golden_time ~failing:false)));
      let time, calls = recording (golden_time ~failing:false) in
      cached ~settings:[ B (declared, 1) ] time;
      greater int ~than:0 (List.length (calls ())))

(* The settings that pick candidates and how a search measures and stops shape
   compilation, at their defaults here. *)
let cache_variables =
  cases ~name:fst "the search's settings shape compilation"
    [
      ("BEAM_PADTO", "false");
      ("BEAM_UOPS_MAX", "3000");
      ("BEAM_UPCAST_MAX", "256");
      ("BEAM_LOCAL_MAX", "1024");
      ("BEAM_MIN_PROGRESS", "0x1.47ae147ae147bp-7");
      ("BEAM_ESTIMATE", "true");
    ]
    (fun (name, value) ->
      equal (option string) (Some value)
        (List.assoc_opt name (Setting.shaping ())))

(* A measurement that favours tensor cores, then swaps, with an optimum at two
   optimisations. *)
let kinds_time ~vars:_ prg =
  let opts = (kernel_info prg).applied_opts in
  let weight = function Opt.Tc _ -> 0.5 | Opt.Swap _ -> 0.6 | _ -> 1.2 in
  let depth = 1 + (3 * max 0 (List.length opts - 2)) in
  List.fold_left (fun t o -> t *. weight o) (1e-3 *. Float.of_int depth) opts

let cache_kinds =
  slow "the cache keeps tensor cores and swaps" (fun () ->
      let cached time =
        K.applied_opts
          (search
             ~settings:[ B (Setting.cachelevel, 1) ]
             ~time 1
             (scheduled "matmul_half" "metal"))
      in
      let time, _ = recording kinds_time in
      let first = cached time in
      is_true ~msg:"a tensor core"
        (List.exists (function Opt.Tc _ -> true | _ -> false) first);
      is_true ~msg:"a swap"
        (List.exists (function Opt.Swap _ -> true | _ -> false) first);
      equal opts first
        (cached (fun ~vars:_ _ -> fail "the search applied its timing")))

(* Printing *)

let progress =
  test "a search prints its progress under DEBUG=2" (fun () ->
      ignore (output ());
      let time, _ = recording (golden_time ~failing:false) in
      ignore
        (search
           ~settings:[ B (Setting.debug, 2) ]
           ~time 1
           (scheduled "add_small" "clang"));
      in_order
        ~subs:[ "from   1 ->   1 actions"; "   1/"; "actions\027[K" ]
        (output ()))

let quiet =
  test "a search prints nothing by default" (fun () ->
      ignore (output ());
      let time, _ = recording (golden_time ~failing:true) in
      ignore (search ~time 2 (scheduled "add_small" "clang"));
      ignore
        (quietly (fun () ->
             Search.get_kernel_actions ~max_up:1 (scheduled "sum_rows" "metal")));
      equal string "" (output ()))

let failure_printed =
  test "a compilation's failure is printed under DEBUG=4" (fun () ->
      let ren, _ = rejecting (fun () -> failwith "rejected") in
      ignore (output ());
      let time, _ = recording (golden_time ~failing:false) in
      ignore
        (search
           ~settings:[ B (Setting.debug, 4) ]
           ~time 1
           (scheduled_for ren "sum_rows"));
      contains ~sub:"Failure(\"rejected\")" (output ()))

(* The environment *)

let environment =
  [
    ("BEAM_PADTO", "1");
    ("TC", "2");
    ("BEAM_TC_OPT", "0");
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
          equal opts
            (recorded_actions "actions_padto.golden")
            (Search.actions ()));
      test "a candidate of BEAM_UOPS_MAX instructions or more is dropped"
        (fun () ->
          ignore (output ());
          let time, calls = recording (golden_time ~failing:false) in
          ignore (search ~time 1 (scheduled "sum_rows" "clang"));
          List.iter
            (fun c ->
              satisfies ~claim:"fewer than 43" int
                (fun n -> n < 43)
                (instructions c.prg))
            (calls ());
          contains ~sub:"too many uops" (output ()));
      test "a candidate of BEAM_UOPS_MAX instructions or more is not compiled"
        (fun () ->
          let ren, compiled = counting "Apple8" in
          let time, record =
            sampling (fun ~vars prg _ -> golden_time ~failing:false ~vars prg)
          in
          ignore (search ~time 1 (scheduled_for ren "sum_rows"));
          let prepared = List.map binary (record ()).prepared in
          is_true ~msg:"prepared" (prepared <> []);
          List.iter
            (fun (src, _) ->
              satisfies ~claim:"a prepared program's" string
                (fun src -> List.mem src prepared)
                src)
            (compiled ()));
      test "a compilation that raises is raised under BEAM_STRICT_MODE"
        (fun () ->
          let ren, _ = rejecting ~every:1 rejected_by_compiler in
          raises_match
            (function Renderer.Compiler.Compile_error _ -> true | _ -> false)
            (fun () ->
              search
                ~time:(fun ~vars:_ _ _ () -> 1.)
                1
                (scheduled_for ren "sum_rows")));
      test
        "a search prints the kernel, failures and its choice under BEAM_DEBUG"
        (fun () ->
          ignore (output ());
          let time, _ = recording (golden_time ~failing:true) in
          ignore (search ~time 2 (scheduled "add_small" "clang"));
          in_order
            ~subs:
              [
                "BEAM_SEARCH:"; "BEAM failed for opts"; "BEAM_SEARCH: final tm=";
              ]
            (output ()));
      Golden.cases ~key:[ "kernel"; "target"; "amt" ]
        "searches_environment.golden" (fun cell ->
          let time, calls = recording (golden_time ~failing:false) in
          let k =
            search ~time
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
          let time, calls = recording (golden_time ~failing:false) in
          ignore (search ~time 1 k);
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

(* The tensor cores' actions *)

let tc_levels () =
  List.filter_map
    (function Opt.Tc t -> Some (t.tc_opt, t.use_tc) | _ -> None)
    (Search.actions ())

let tc_actions =
  group "the tensor cores' actions"
    [
      test "ask for BEAM_TC_OPT's level on every axis but the first" (fun () ->
          let levels o =
            Setting.context
              [ B (Setting.beam_tc_opt, o) ]
              (fun () -> List.map fst (tc_levels ()))
          in
          equal (list int) ~msg:"default"
            (0 :: List.init 9 (Fun.const 2))
            (List.map fst (tc_levels ()));
          equal (list int) ~msg:"BEAM_TC_OPT=1"
            (0 :: List.init 9 (Fun.const 1))
            (levels 1));
      test "leave TC_OPT to hand-coded optimizations" (fun () ->
          let levels o =
            Setting.context
              [ B (Setting.tc_opt, o) ]
              (fun () -> List.map fst (tc_levels ()))
          in
          equal (list int) (levels 0) (levels 2));
      test "ask for TC's level" (fun () ->
          let uses =
            Setting.context
              [ B (Setting.use_tc, 2) ]
              (fun () -> List.map snd (tc_levels ()))
          in
          equal (list int) (List.init 10 (Fun.const 2)) uses);
    ]

let () =
  exit
    (Windtrap.run "Tolk.Search"
       [
         actions;
         under_environment;
         tc_actions;
         candidates;
         searches;
         candidate_laws;
         keeps_writes;
         keeps_writes_exhaustively;
         chooses_the_fastest;
         is_deterministic;
         midpoints;
         applies_the_kernel_once;
         early_stops;
         exceptions;
         test_size;
         width;
         cache;
         cache_settings;
         cache_declared;
         cache_variables;
         cache_kinds;
         uncompilable;
         compiles_once;
         progress;
         quiet;
         failure_printed;
         storage_placed;
         rounds;
         incumbent_first;
         uncompiled_kernel;
         progress_law;
       ])
