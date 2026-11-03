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

(* A GPU's renderer is made with its compiler, which finds its vendor's library
   or raises Failure: [renderer] skips the test on a machine without it, and
   [made] tells whether the machine makes a target's. *)
let renderers =
  List.map
    (fun cell ->
      ( cell "target",
        match renderer_of_row cell with
        | r -> Ok r
        | exception Failure why -> Error why ))
    (Golden.rows "targets.golden")

let renderer target =
  match List.assoc target renderers with
  | Ok r -> r
  | Error why -> skip ~reason:why ()

let made target = Result.is_ok (List.assoc target renderers)
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
(* On one domain, a search times its candidates in their order, which the
   goldens record. *)
let quietly ?(settings = []) f =
  Setting.context
    (B (Setting.no_color, true)
    :: B (Setting.cachelevel, 0)
    :: B (Setting.parallel, 0)
    :: settings)
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
      Helpers.prod (List.map (fun s -> Shape.sym_infer s vars) p.global_size)
  | _ -> failf "%a is no program" Ops.pp prg

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

(* The timing whose samples all take a second. *)
let constant = (Fun.id, fun ~vars:_ _ -> 1.)

(* The timing that fails the test when the search links or times. *)
let untimed =
  ( (fun _ -> fail "the search linked a program"),
    fun ~vars:_ _ -> fail "the search timed a program" )

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

let search ?settings ?allow_test_size ~timing:(link, time) amt k =
  quietly ?settings (fun () ->
      Search.beam_search ~link ~time ?allow_test_size amt k)

(* The actions *)

let recorded_actions file =
  List.map (fun cell -> Kernel_opts.opt_of_cell (cell "opt")) (Golden.rows file)

let actions =
  test "the actions are tinygrad's, in order" (fun () ->
      equal opts (recorded_actions "actions.golden") (Search.actions ()))

(* Candidates

   tinygrad numbers a candidate by its action's position from 1, after the
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

(* Searches *)

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
      cases "on the host" ~id:name (List.filter quick rows) searched;
      slow
      @@ cases "on GPUs" ~id:name
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
      cases "leaves its scheduler as it was" ~id:name_of small
        (fun (kernel, target) ->
          let k = scheduled kernel target in
          let ast = K.ast k in
          ignore (Search.get_kernel_actions k);
          equal Uops.uop ast (K.ast k);
          equal opts [] (K.applied_opts k));
      cases "makes each candidate by applying its action" ~id:name_of small
        (fun (kernel, target) ->
          let k = scheduled kernel target in
          let applied (a, k') =
            equal
              ~msg:(Format.asprintf "%a" Opt.pp a)
              opts [ a ] (K.applied_opts k')
          in
          List.iter applied (Search.get_kernel_actions k));
      cases "leaves out more candidates under a smaller max_up" ~id:name_of
        small (fun (kernel, target) ->
          let k = scheduled kernel target in
          let fewer = List.map fst (Search.get_kernel_actions ~max_up:2 k) in
          let all = List.map fst (Search.get_kernel_actions k) in
          List.iter
            (fun a ->
              is_true
                ~msg:(Format.asprintf "%a" Opt.pp a)
                (List.exists (Opt.equal a) all))
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
      match List.filter evaluable (Search.get_kernel_actions k) with
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
  let candidates k = List.filter evaluable (Search.get_kernel_actions k) in
  slow
  @@ cases "every kernel two actions make writes what its kernel writes"
       ~id:name_of pairs (fun (kernel_name, target) ->
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
    let timing, calls = recording (by_depth times) in
    let k = search ~timing 1 (scheduled "sum_rows" "clang") in
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
          let k, calls = searched [ 0.; 1e-3; 1e-3 ] in
          is_true ~msg:"a second round"
            (List.exists (fun c -> depth c = 2) calls);
          equal int 1 (List.length (K.applied_opts k)));
      test "a search whose fastest takes less than BEAM_MIN_PROGRESS stops"
        (fun () ->
          let k, calls = searched [ 0.; min_progress /. 2.; 0. ] in
          equal int 0 (List.length (List.filter (fun c -> depth c = 2) calls));
          equal int 1 (List.length (K.applied_opts k)));
      test
        "a search keeps the fastest alone when it gains less than \
         BEAM_MIN_PROGRESS" (fun () ->
          let k, _ = searched [ 0.; 1e-3; 1e-3 -. (min_progress /. 2.) ] in
          equal int 2 (List.length (K.applied_opts k)));
      test "the kernel is never timed" (fun () ->
          let _, calls = searched [ 1.; 1e-3; 1e-3 ] in
          equal int 0 (List.length (List.filter (fun c -> depth c = 0) calls)));
    ]

let chooses_the_fastest =
  let pp ppf (kernel, amt, seed) =
    Format.fprintf ppf "%s, width %d, seed %d" kernel amt seed
  in
  slow
  @@ prop ~count:20 "a search chooses the fastest program it measured"
       Gen.(
         with_pp pp
           (triple
              (of_list [ "add_small"; "sum_rows"; "variable_rows" ])
              (int_range 1 3) nat))
       (fun (kernel, amt, seed) ->
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


let midpoints =
  test "a search times with each variable at its bounds' middle" (fun () ->
      let timing, calls = recording (golden_time ~failing:false) in
      ignore (search ~timing 1 (scheduled "variable_rows" "metal"));
      let calls = calls () in
      is_true ~msg:"measured" (calls <> []);
      List.iter
        (fun c -> equal (list (pair string int)) [ ("n", 8) ] c.vars)
        calls)

let positional =
  cases "a search binds a positional size in its call and launch"
    ~id:string_of_bool [ false; true ] (fun allow_test_size ->
      let ast = kernel "symbolic" in
      let v = List.hd (Shape.variables ast) in
      let p = Shape.param_like v 4 in
      let ast = Ops.substitute ~calls:Skip ~pass:Once ast [ (v, p) ] in
      let k = K.v ast (renderer "hip") in
      K.convert_loop_to_global k;
      let linked = ref 0 in
      let link call =
        incr linked;
        equal Dtypes.const
          (`Int (Bigint.of_int 4))
          (Ops.value (List.nth (Ops.src_without_body call) 4));
        match Ops.arg (Ops.body call) with
        | Program info ->
            List.iter
              (fun s -> ignore (Shape.sym_infer s []))
              (info.global_size @ info.local_size)
        | _ -> fail "the timed call has no program"
      in
      let time ~vars () =
        equal (list (pair string int)) [] vars;
        1.
      in
      ignore (search ~allow_test_size ~timing:(link, time) 2 k);
      greater int ~than:0 !linked)

(* The timing's stages *)

let small_searches =
  Gen.(
    with_pp
      (fun ppf (kernel, target, amt) ->
        Format.fprintf ppf "%s on %s, width %d" kernel target amt)
      (triple
         (of_list [ "add_small"; "sum_rows"; "variable_rows"; "symbolic" ])
         (of_list (List.filter made [ "clang"; "metal" ]))
         (int_range 1 3)))

let links_once =
  prop ~count:10 "a search links each program it times once, before its samples"
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
      equal (list Uops.uop) r.linked (List.rev sampled))

(* Samples drawn from [seed]: a program's base time, from a microsecond to two
   milliseconds, times up to one and a half, so that the samples of programs of
   near base times overlap. *)
let spread seed ~vars:_ prg i =
  let positions = List.map position (kernel_info prg).applied_opts in
  let base = 1 + (Hashtbl.seeded_hash seed positions mod 2000) in
  let noise = 100 + (Hashtbl.seeded_hash seed (positions, i) mod 51) in
  1e-8 *. Float.of_int (base * noise)

let least = List.fold_left Float.min infinity

(* The samples of each program linked, by binary, in order. *)
let samples_of r =
  List.map
    (fun p ->
      ( p,
        List.filter_map
          (fun c -> if binary c.prg = binary p then Some c.time else None)
          r.calls ))
    r.linked

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
    "each program is linked once and sampled until its least exceeds three \
     times the incumbent's, three times at most"
    Gen.(
      with_pp pp
        (triple
           (of_list [ "add_small"; "sum_rows"; "variable_rows" ])
           (int_range 1 3) nat))
    (fun (kernel, amt, seed) ->
      let timing, record = sampling (spread seed) in
      ignore
        (search ~allow_test_size:false ~timing amt (scheduled kernel "clang"));
      let r = record () in
      equal ~msg:"linked once" (list string)
        (List.sort_uniq String.compare (List.map binary r.linked))
        (List.sort String.compare (List.map binary r.linked));
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

(* A timing of constant samples that raises [e] when it links a program, or when
   it samples one. *)
let raising_at stage e =
  ( (fun call -> if stage = `Link then raise e else call),
    fun ~vars:_ _ -> if stage = `Time then raise e else 1e-3 )

let stages = [ ("a link", `Link); ("a sample", `Time) ]

let exceptions =
  group "a timing that raises"
    [
      cases "Exit is raised by the search, from" ~id:fst stages
        (fun (_, stage) ->
          raises Exit (fun () ->
              search ~timing:(raising_at stage Exit) 1
                (scheduled "sum_rows" "clang")));
      cases
        "Failure drops its program, and a search of no timed candidate keeps \
         its kernel, from"
        ~id:fst stages (fun (_, stage) ->
          let k = scheduled "sum_rows" "clang" in
          let chosen =
            search ~timing:(raising_at stage (Failure "no device")) 1 k
          in
          equal Uops.uop (K.ast k) (K.ast chosen);
          equal opts [] (K.applied_opts chosen));
    ]

(* [out[i] = j^2] for each [j < x[0]] in turn, over 16 [i], [x] of
   int64: its estimated operations count [j] up to 2^63 - 1 times on each [i],
   which fits no int. *)
let counts_to_loaded () =
  let x = Shape.param ~shape:[ Int 16 ] 1 Int64
  and out = Shape.param ~shape:[ Int 16 ] 0 Int64 in
  let i = Ops.range (Int 16) [ 0 ] in
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
  test "a search measures candidates whose estimates fit no int" (fun () ->
      let timing, record = sampling (fun ~vars:_ _ _ -> 1.) in
      let k = K.v (counts_to_loaded ()) (renderer "clang") in
      K.convert_loop_to_global k;
      ignore (search ~timing 1 k);
      greater int ~msg:"candidates measured" ~than:0
        (List.length (List.filter (fun c -> depth c > 0) (record ()).calls)))

(* The fastest program of [depth] optimisations, the first timed of ties, if
   any. *)
let fastest r depth =
  List.fold_left
    (fun best (p, ts) ->
      if List.length (kernel_info p).applied_opts <> depth then best
      else
        match best with
        | Some (_, bs) when least bs <= least ts -> best
        | _ -> Some (p, ts))
    None (samples_of r)

let rounds_law =
  let min_progress = 0.01 /. 1e6 in
  let pp ppf (kernel, amt, seed) =
    Format.fprintf ppf "%s, width %d, seed %d" kernel amt seed
  in
  prop ~count:30
    "a round goes on while its fastest beats the beam's first by \
     BEAM_MIN_PROGRESS or more, and the last keeps its fastest if it beat it"
    Gen.(
      with_pp pp
        (triple
           (of_list [ "add_small"; "sum_rows"; "variable_rows" ])
           (int_range 1 3) nat))
    (fun (kernel, amt, seed) ->
      let timing, record = sampling (spread seed) in
      let chosen =
        search ~allow_test_size:false ~timing amt (scheduled kernel "clang")
      in
      let r = record () in
      let rounds = List.length (K.applied_opts chosen) in
      for d = 1 to rounds do
        let msg = Printf.sprintf "round %d" d in
        less float_exact ~msg ~than:(incumbent r (d - 1)) (incumbent r d);
        if d < rounds then begin
          at_least float_exact ~msg ~than:min_progress (incumbent r d);
          at_least float_exact ~msg ~than:min_progress
            (incumbent r (d - 1) -. incumbent r d)
        end
      done;
      (match fastest r (rounds + 1) with
      | Some _ ->
          at_least float_exact
            ~msg:(Printf.sprintf "round %d" (rounds + 1))
            ~than:(incumbent r rounds)
            (incumbent r (rounds + 1))
      | None -> ());
      (match fastest r rounds with
      | Some (p, _) ->
          equal opts (kernel_info p).applied_opts (K.applied_opts chosen)
      | None -> equal int 0 rounds);
      cover "a round that progressed" (rounds > 0);
      cover "a timed round that stopped" (fastest r (rounds + 1) <> None))

(* A candidate's program names no device: its call's storage names the
   renderer's device, which [link] places. *)
let storage_placed =
  test "a candidate's call names its renderer's device on its storage"
    (fun () ->
      let named = ref [] in
      let link call =
        List.iter
          (fun u ->
            match Ops.arg u with
            | Param p when Ops.op u = Op.Param && p.addrspace <> Dtype.Alu ->
                named := p.device :: !named
            | _ -> ())
          (List.tl (Ops.src call));
        Ops.body call
      in
      let timing = (link, golden_time ~failing:false) in
      ignore (search ~timing 1 (scheduled "sum_rows" "clang"));
      greater int ~than:0 (List.length !named);
      List.iter (equal (option string) (Some "CPU")) !named)

(* The launch each candidate of [kernel] is measured with, under a measurement
   that times them alike, so that the search ends after its first round. *)
let launches ?allow_test_size kernel =
  let timing, calls = recording (fun ~vars:_ _ -> 1e-3) in
  ignore (search ?allow_test_size ~timing 1 (scheduled kernel "metal"));
  List.map
    (fun c ->
      ( (kernel_info c.prg).applied_opts,
        match Ops.arg c.prg with
        | Program p -> List.map (fun s -> Shape.sym_infer s c.vars) p.global_size
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
          let timing, calls = recording (golden_time ~failing:false) in
          ignore (search ~timing 1 (scheduled "add_large" "metal"));
          List.iter
            (fun c ->
              satisfies ~claim:"at most 65536" int
                (fun n -> n <= 65536)
                (workgroups ~vars:c.vars c.prg))
            (calls ()));
      test "launches them all without allow_test_size" (fun () ->
          let timing, calls = recording (golden_time ~failing:false) in
          ignore
            (search ~allow_test_size:false ~timing 1
               (scheduled "add_large" "metal"));
          is_true
            (List.exists
               (fun c -> workgroups ~vars:c.vars c.prg > 65536)
               (calls ())));
      slow
      @@ test "chooses as measuring them all does, for a time per workgroup"
           (fun () ->
             let choose allow_test_size =
               let timing, _ = recording per_workgroup in
               K.applied_opts
                 (search ~allow_test_size ~timing 1
                    (scheduled "add_large" "metal"))
             in
             equal opts (choose false) (choose true));
    ]

let width =
  test "a search of no width is refused" (fun () ->
      raises_match (Exn.invalid_arg ?sub:None) (fun () ->
          search ~timing:constant 0 (scheduled "sum_rows" "clang")))

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

(* Counts per source, safe from several domains. *)
type counts = { lock : Mutex.t; table : (string, int) Hashtbl.t }

let counts () = { lock = Mutex.create (); table = Hashtbl.create 64 }

let count c src =
  Mutex.protect c.lock (fun () ->
      Hashtbl.replace c.table src
        (1 + Option.value ~default:0 (Hashtbl.find_opt c.table src)))

let to_list c =
  Mutex.protect c.lock (fun () ->
      Hashtbl.fold (fun src n l -> (src, n) :: l) c.table [])

(* A Metal renderer for the family [arch] of its own, so that no other test's
   programs are reused, whose compilations are counted per source. Its
   compiler counts a source, then compiles it to itself. *)
let counting arch =
  let compiled = counts () in
  let t =
    {
      Helpers.Target.device = "METAL";
      renderer = "METAL";
      arch;
      interface = "";
      indices = "";
    }
  in
  let m =
    match Cstyle.metal t with
    | m -> m
    | exception Failure why -> skip ~reason:why ()
  in
  let compile src =
    count compiled src;
    src
  in
  let ren =
    Renderer.v ~name:m.name ~suffix:m.suffix ~supports_float4:m.supports_float4
      ~has_local:m.has_local ~has_shared:m.has_shared ~global_max:m.global_max
      ~local_max:m.local_max ?global_prod_max:m.global_prod_max
      ~shared_max:m.shared_max ~tensor_cores:m.tensor_cores
      ~extra_matcher:m.extra_matcher ~code_for_op:m.code_for_op ~native:m.native
      ~render:m.render
      ~compiler:(Renderer.Compiler.v compile)
      t
  in
  (ren, compiled)

let uncompilable =
  let dropped reject () =
    let ren, rejected = rejecting reject in
    let timing, calls = recording (golden_time ~failing:false) in
    ignore (search ~timing 1 (scheduled_for ren "sum_rows"));
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
  let cached ?(ignore = false) timing =
    search
      ~settings:
        [ B (Setting.cachelevel, 1); B (Setting.ignore_beam_cache, ignore) ]
      ~timing 2
      (scheduled "sum_rows" "metal")
  in
  test
    "a search kept in the cache links and times nothing, unless it is ignored"
    (fun () ->
      let timing, _ = recording (golden_time ~failing:false) in
      let first = K.applied_opts (cached timing) in
      equal opts first (K.applied_opts (cached untimed));
      let timing, calls = recording (golden_time ~failing:false) in
      equal opts first (K.applied_opts (cached ~ignore:true timing));
      is_true ~msg:"measured again" (calls () <> []))

(* One expression stored on two devices of one target is one kernel, so the
   search of the second device's finds the first's in the cache. *)
let cache_devices =
  test "a search kept for one device serves another of its target" (fun () ->
      let on slot d =
        let x = Ops.new_buffer ~slot d 64 Float32 in
        let out = Ops.new_buffer ~slot:(slot + 2) d 64 Float32 in
        Ops.after out [ Ops.store out Ops.O.(x * x + float 1.) ]
      in
      let linear, _ =
        Schedule.create_linear_with_vars
          (Ops.sink [ on 0 "CPU:0"; on 1 "CPU:1" ])
      in
      let searched timing call =
        let k = K.v (Ops.body call) (renderer "clang") in
        K.convert_loop_to_global k;
        K.applied_opts
          (search ~settings:[ B (Setting.cachelevel, 1) ] ~timing 2 k)
      in
      match Ops.src linear with
      | [ a; b ] ->
          let timing, _ = recording (golden_time ~failing:false) in
          let first = searched timing a in
          equal opts first (searched untimed b)
      | calls -> failf "%d calls, not 2" (List.length calls))

(* A search whose entry is damaged on disk measures again, as one never kept,
   and keeps what it finds. *)
let cache_damaged =
  test "a search whose entry is damaged measures again" (fun () ->
      let cached timing =
        K.applied_opts
          (search
             ~settings:[ B (Setting.cachelevel, 1) ]
             ~timing 2
             (scheduled "sum_rows" "clang"))
      in
      let first = cached (fst (recording (golden_time ~failing:false))) in
      Disk_cache.damage ~table:"beam_search" Helpers.cachedb
        Disk_cache.truncated;
      let timing, calls = recording (golden_time ~failing:false) in
      equal opts ~msg:"found again" first (cached timing);
      greater int ~msg:"measured again" ~than:0 (List.length (calls ()));
      equal opts ~msg:"kept again" first (cached untimed))

(* A search kept under one value of a setting that shapes compilation is not
   used under another. *)
let cache_settings =
  test "a search kept under one setting measures again under another" (fun () ->
      let cached ?(settings = []) timing =
        ignore
          (search
             ~settings:(B (Setting.cachelevel, 1) :: settings)
             ~timing 2
             (scheduled "sum_rows" "metal"))
      in
      cached (fst (recording (golden_time ~failing:false)));
      let timing, calls = recording (golden_time ~failing:false) in
      cached ~settings:[ B (Setting.transcendental, 2) ] timing;
      greater int ~than:0 (List.length (calls ())))

(* A setting that its caller declares to reach output: the search knows nothing
   of it, yet keys on it. *)
let declared = Setting.int ~reach:Output "TOLK_TEST_SEARCH" 0

let cache_declared =
  test
    "a search kept under one value of a setting declared by its caller \
     measures again under another" (fun () ->
      let cached ?(settings = []) timing =
        ignore
          (search
             ~settings:(B (Setting.cachelevel, 1) :: settings)
             ~timing 2
             (scheduled "sum_rows" "metal"))
      in
      cached (fst (recording (golden_time ~failing:false)));
      let timing, calls = recording (golden_time ~failing:false) in
      cached ~settings:[ B (declared, 1) ] timing;
      greater int ~than:0 (List.length (calls ())))

(* The settings that pick candidates and how a search measures and stops shape
   compilation, at their defaults here. *)
let cache_variables =
  cases "the search's settings shape compilation" ~id:fst
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
  slow
  @@ test "the cache keeps tensor cores and swaps" (fun () ->
      let cached timing =
        K.applied_opts
          (search
             ~settings:[ B (Setting.cachelevel, 1) ]
             ~timing 1
             (scheduled "matmul_half" "metal"))
      in
      let timing, _ = recording kinds_time in
      let first = cached timing in
      is_true ~msg:"a tensor core"
        (List.exists (function Opt.Tc _ -> true | _ -> false) first);
      is_true ~msg:"a swap"
        (List.exists (function Opt.Swap _ -> true | _ -> false) first);
      equal opts first (cached untimed))

let source prg =
  match Ops.arg (Ops.nth prg 2) with
  | String s -> s
  | _ -> failf "the program %a has no source" Ops.pp prg

(* A compiler whose binaries are kept, in the table [binaries]. *)
let binaries = "search test binaries"

let cache_candidates =
  test
    "a search keeps each candidate's binary in its compiler's table, as a \
     kernel compiled with it keeps its own" (fun () ->
      let ren =
        Renderer.with_compiler
          (Renderer.Compiler.v ~cachekey:(fun () -> binaries) Fun.id)
          (renderer "metal")
      in
      let k = kernel "sum_rows" in
      let k = Ops.replace k ~arg:(Kernel (Ops.kernel_info ~beam:2 ())) in
      let (link, time), record =
        sampling (fun ~vars prg _ -> golden_time ~failing:false ~vars prg)
      in
      quietly ~settings:[ B (Setting.cachelevel, 1) ] @@ fun () ->
      Helpers.Diskcache.clear ();
      let prg =
        Codegen.to_program
          ~beam:(Search.beam_search ~link ~time)
          k ren
      in
      let kept p = Helpers.Diskcache.get ~table:binaries (source p) in
      let candidates = (record ()).linked in
      greater int ~than:1 (List.length candidates);
      equal ~msg:"candidates' binaries kept" (list (option string))
        (List.map (fun p -> Some (binary p)) candidates)
        (List.map kept candidates);
      equal (option string) (Some (binary prg)) (kept prg))

(* Printing *)

let progress =
  test "a search prints its progress under DEBUG=2" (fun () ->
      ignore (output ());
      let timing, _ = recording (golden_time ~failing:false) in
      ignore
        (search
           ~settings:[ B (Setting.debug, 2) ]
           ~timing 1
           (scheduled "add_small" "clang"));
      in_order
        ~subs:[ "from   1 ->   1 actions"; "   1/"; "actions\027[K" ]
        (output ()))

let quiet =
  test "a search prints nothing by default" (fun () ->
      ignore (output ());
      let timing, _ = recording (golden_time ~failing:true) in
      ignore (search ~timing 2 (scheduled "add_small" "clang"));
      ignore
        (quietly (fun () ->
             Search.get_kernel_actions ~max_up:1 (scheduled "sum_rows" "metal")));
      equal string "" (output ()))

let failure_printed =
  test "a compilation's failure is printed under DEBUG=4" (fun () ->
      let ren, _ = rejecting (fun () -> failwith "rejected") in
      ignore (output ());
      let timing, _ = recording (golden_time ~failing:false) in
      ignore
        (search
           ~settings:[ B (Setting.debug, 4) ]
           ~timing 1
           (scheduled_for ren "sum_rows"));
      contains ~sub:"Failure(\"rejected\")" (output ()))

(* The environment *)

let environment =
  [
    ("BEAM_PADTO", "1");
    ("TC", "2");
    ("BEAM_STRICT_MODE", "1");
    ("BEAM_UOPS_MAX", "50");
    ("BEAM_LOG_SURPASS_MAX", "1");
    ("BEAM_DEBUG", "2");
  ]

let instructions prg = List.length (Ops.src (Ops.nth prg 1))

(* The environment is read once per process: the suite runs this group in a
   process of its own (see dune). *)
let under_environment =
  tag "environment"
  @@ group
       (String.concat " " (List.map (fun (n, v) -> n ^ "=" ^ v) environment))
       [
         test "the suite runs this group with them" (fun () ->
             List.iter
               (fun (name, value) ->
                 equal ~msg:name (option string) (Some value)
                   (Sys.getenv_opt name))
               environment);
         test "the actions are tinygrad's, pads included" (fun () ->
             equal opts
               (recorded_actions "actions_padto.golden")
               (Search.actions ()));
         test "a candidate of BEAM_UOPS_MAX instructions or more is dropped"
           (fun () ->
             ignore (output ());
             let timing, calls = recording (golden_time ~failing:false) in
             ignore (search ~timing 1 (scheduled "sum_rows" "clang"));
             List.iter
               (fun c ->
                 satisfies ~claim:"fewer than 50" int
                   (fun n -> n < 50)
                   (instructions c.prg))
               (calls ());
             contains ~sub:"too many uops" (output ()));
         test
           "a candidate of BEAM_UOPS_MAX instructions or more is not compiled"
           (fun () ->
             let ren, compiled = counting "Apple8" in
             let timing, record =
               sampling (fun ~vars prg _ ->
                   golden_time ~failing:false ~vars prg)
             in
             ignore (search ~timing 1 (scheduled_for ren "sum_rows"));
             let linked = List.map binary (record ()).linked in
             is_true ~msg:"linked" (linked <> []);
             List.iter
               (fun (src, _) ->
                 satisfies ~claim:"a linked program's" string
                   (fun src -> List.mem src linked)
                   src)
               (to_list compiled));
         test "a compilation that raises is raised under BEAM_STRICT_MODE"
           (fun () ->
             let ren, _ = rejecting ~every:1 rejected_by_compiler in
             raises_match
               (function
                 | Renderer.Compiler.Compile_error _ -> true | _ -> false)
               (fun () ->
                 search ~timing:constant 1 (scheduled_for ren "sum_rows")));
         test
           "a search prints the kernel, failures and its choice under \
            BEAM_DEBUG" (fun () ->
             ignore (output ());
             let timing, _ = recording (golden_time ~failing:true) in
             ignore (search ~timing 2 (scheduled "add_small" "clang"));
             in_order
               ~subs:
                 [
                   "BEAM_SEARCH:";
                   "BEAM failed for opts";
                   "BEAM_SEARCH: final tm=";
                 ]
               (output ()));
         Golden.cases ~key:[ "kernel"; "target"; "amt" ]
           "searches_environment.golden" (fun cell ->
             let timing, calls = recording (golden_time ~failing:false) in
             let k =
               search ~timing
                 (int_of_string (cell "amt"))
                 (scheduled (cell "kernel") (cell "target"))
             in
             equal opts
               (Kernel_opts.opts_of_cell (cell "opts"))
               (K.applied_opts k);
             equal ~msg:"measurements" int
               (int_of_string (cell "measurements"))
               (List.length (calls ())));
         test
           "a candidate of more than 1000 times the fewest operations of its \
            round so far is not measured" (fun () ->
             let is_pad = function Opt.Padto _ -> true | _ -> false in
             let k = scheduled "transpose_33" "metal" in
             let pads =
               List.filter
                 (fun (_, k') -> List.exists is_pad (K.applied_opts k'))
                 (quietly (fun () -> Search.get_kernel_actions k))
             in
             is_true ~msg:"a pad is a candidate" (pads <> []);
             ignore (output ());
             let timing, calls = recording (golden_time ~failing:false) in
             ignore (search ~timing 1 k);
             contains ~sub:"too much compute" (output ());
             is_true ~msg:"measured" (calls () <> []);
             (* [k] applies no optimisation, and a candidate of round [r]
                applies [r]. A candidate dropped for its operations lowers no
                round's fewest, so the fewest so far are those measured. *)
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
                 let msg =
                   Format.asprintf "%a" (Testable.pp opts) info.applied_opts
                 in
                 at_most int ~msg ~than:(1000 * least) ops;
                 Hashtbl.replace fewest round least)
               (calls ()));
         test
           "kernels of too many lanes are reported under BEAM_LOG_SURPASS_MAX"
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
      test "ask for level 2 on every axis but the first" (fun () ->
          equal (list int)
            (0 :: List.init 9 (Fun.const 2))
            (List.map fst (tc_levels ())));
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
         midpoints;
         positional;
         links_once;
         early_stops;
         exceptions;
         test_size;
         width;
         cache;
         cache_devices;
         cache_damaged;
         cache_settings;
         cache_declared;
         cache_variables;
         cache_kinds;
         cache_candidates;
         uncompilable;
         progress;
         quiet;
         failure_printed;
         storage_placed;
         rounds;
         rounds_law;
         estimates_past_an_int;
       ])
