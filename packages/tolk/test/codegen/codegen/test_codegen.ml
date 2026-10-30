open Windtrap
open Tolk

(* The host's target, as the engine gives it. *)
let host_target = Tolk_engine.target Rig.host
let rejects f = raises_match (Exn.invalid_arg ?sub:None) f

(* Two graphs are the same when their texts are, and a failure is the diff of
   the texts. *)
let same_graph ?msg expected actual =
  equal ?msg text (Graph.to_string expected) (Graph.to_string actual)

let kernel_info u =
  match Ops.arg u with
  | Kernel k -> k
  | _ -> invalid_arg "a kernel's sink holds its kernel information"

let first n u = Ops.replace u ~src:(List.filteri (fun i _ -> i < n) (Ops.src u))

let source prg =
  match Ops.arg (Ops.nth prg 2) with
  | String src -> src
  | _ -> invalid_arg "a program's third source is its source"

(* Targets *)

let target device renderer arch =
  { Helpers.Target.device; renderer; arch; interface = ""; indices = "" }

(* The goldens record no binary: a program's binary is its source's bytes. *)
let uncompiled = Renderer.Compiler.v Fun.id

let renderer_for ?(compiler = uncompiled) (t : Helpers.Target.t) =
  let r =
    match t.renderer with
    | "CLANG" -> Cstyle.clang t
    | "METAL" -> Cstyle.metal t
    | "CUDA" -> Cstyle.cuda t
    | "HIP" -> Cstyle.hip t
    | r -> invalid_arg ("no renderer " ^ r)
  in
  Renderer.with_compiler compiler r

let clang_target = target "CPU" "CLANG" "x86_64,x86-64"
let clang = renderer_for clang_target
let metal = renderer_for (target "METAL" "METAL" "Apple9")

(* Cases *)

(* The cases of cases.golden compile the kernels of kernels.golden, each for a
   target and under settings. *)
let kernels = lazy (Array.of_list (Ops.src (Golden.sink "kernels.golden")))
let rows = Golden.rows "cases.golden"
let row_named name = List.find (fun row -> row "name" = name) rows
let kernel row = (Lazy.force kernels).(int_of_string (row "kernel"))
let kernel_of name = kernel (row_named name)

let renderer_of_row row =
  renderer_for (target (row "device") (row "renderer") (row "arch"))

(* tinygrad's goldens name kernels without colour. *)
let under row f =
  Setting.context
    (B (Setting.no_color, true) :: Kernel_opts.settings_of_cell (row "setting"))
    f

let optimize k = Ops.tag k = None

let lowered row =
  let k = kernel row in
  under row (fun () ->
      Codegen.full_rewrite_to_sink ~optimize:(optimize k) k
        (renderer_of_row row))

let program row =
  under row (fun () -> Codegen.to_program (kernel row) (renderer_of_row row))

(* The golden of a case is a sink of the lowered kernel and the program, less
   its binary. *)
let recorded row = Golden.sink (row "name" ^ ".golden")
let recorded_lowering row = Ops.nth (recorded row) 0
let recorded_program row = Ops.nth (recorded row) 1
let compiles row = row "outcome" = "ok"

(* The default run compiles each case for one of its targets, taken in turn, so
   that it compiles for every target; the slow run compiles each case for each
   target. Kernels that take long to compile run with the slow tests only, and a
   refused optimisation, which costs nothing, runs by default. *)
let heavy =
  [
    "add_big";
    "big_sum";
    "matmul_big";
    "matmul_half_big";
    "matvec";
    "conv_big";
    "matmul_half_tc_shaped";
  ]

let by_default =
  let chosen = Hashtbl.create 128 in
  let cases =
    List.fold_left
      (fun seen row ->
        if List.mem (row "case") seen then seen else row "case" :: seen)
      [] rows
    |> List.rev
  in
  List.iteri
    (fun i case ->
      let targets =
        List.filter_map
          (fun row ->
            if row "case" = case && compiles row then Some (row "target")
            else None)
          rows
      in
      if targets <> [] then
        Hashtbl.replace chosen case
          (List.nth targets (i mod List.length targets)))
    cases;
  fun row ->
    (not (compiles row))
    || (not (List.mem (row "case") heavy))
       && Hashtbl.find_opt chosen (row "case") = Some (row "target")

let per_row ?(only = fun _ -> true) ?(default = by_default) check =
  List.filter_map
    (fun row ->
      if not (only row) then None
      else
        let mark = if default row then Fun.id else slow in
        Some (mark @@ test (row "name") (fun () -> check row)))
    rows

(* A law that lowers a kernel again runs by default on a few kernels only: an
   elementwise one, a reduction, a gated store, a symbolic shape and an
   unroll. *)
let sample row =
  List.mem (row "case")
    [
      "elementwise_add"; "reduce_rows"; "gated_store"; "symbolic"; "sum_unroll";
    ]

(* An emulated float8 converts as IEEE does, where tinygrad saturates and
   flushes, so the graphs of these programs differ from tinygrad's in their
   conversions; `values on the host` pins what they compute. *)
let emulated_fp8 = [ "fp8_clang"; "fp8_metal"; "fp8_hip" ]
let d9 row = List.mem (row "name") emulated_fp8

(* Stages *)

let lowers_as_tinygrad row = same_graph (recorded_lowering row) (lowered row)

(* A program is compared without its binary; its source is compared apart. *)
let programs_as_tinygrad row =
  same_graph (first 2 (recorded_program row)) (first 2 (program row))

let program_info u =
  match Ops.arg u with
  | Program p -> Format.asprintf "%a" Ops.pp_program_info p
  | _ -> invalid_arg "a program holds its program information"

let launches_as_tinygrad row =
  equal string
    (program_info (recorded_program row))
    (program_info (program row))

(* CUDA keeps a float8 infinity special; the source compares with tinygrad's
   once the guard is written back as tinygrad writes it. A narrowed program's
   source is tinygrad's for its instructions with the casts that narrow, its
   golden <name>_narrowed. *)
let writes_as_tinygrad row =
  let src = source (program row) in
  let src =
    if row "renderer" = "CUDA" then Cuda_fp8.tinygrad_of_d16 src else src
  in
  equal text (source (recorded_program row)) src

let sources =
  List.filter_map
    (fun row ->
      if (not (compiles row)) || d9 row then None
      else
        let t =
          if row "narrowed" = "True" then
            Golden.text
              (row "name" ^ "_narrowed.golden")
              (fun () -> source (program row))
          else test (row "name") (fun () -> writes_as_tinygrad row)
        in
        Some (if by_default row then t else slow @@ group (row "name") [ t ]))
    rows

let refuses_as_tinygrad row =
  let message =
    String.sub (row "outcome") 16 (String.length (row "outcome") - 16)
  in
  raises_match
    (function Invalid_argument m -> String.equal m message | _ -> false)
    (fun () -> program row)

let stages =
  group "stages"
    [
      group "full_rewrite_to_sink lowers each kernel as tinygrad does"
        (per_row
           ~only:(fun row -> compiles row && not (d9 row))
           lowers_as_tinygrad);
      group "to_program makes each program as tinygrad does"
        (per_row
           ~only:(fun row -> compiles row && not (d9 row))
           programs_as_tinygrad);
      group "each program is rendered as tinygrad renders it" sources;
      group "an emulated float8 program launches as tinygrad's"
        (per_row ~only:d9 launches_as_tinygrad);
      group "an optimisation that does not apply raises tinygrad's error"
        (per_row ~only:(fun row -> not (compiles row)) refuses_as_tinygrad);
    ]

(* Values *)

(* A kernel is interpretable when it writes through stores alone: no memory of
   its own, no hardware index, no code the interpreter cannot read. *)
let opaque =
  Op.
    [
      Alloc;
      Buffer;
      After;
      Special;
      Wmma;
      Custom;
      Customi;
      Barrier;
      If;
      Endif;
      Backedge;
    ]

let interpretable u =
  List.for_all
    (fun v -> not (List.mem (Ops.op v) opaque))
    (Ops.toposort ~calls:Enter u)

let bound k =
  List.map
    (fun (name, v) -> (name, `Int (Bigint.of_int v)))
    (Kernel_opts.variables k)

(* The variable of [slot] among [uops]. *)
let variable_of uops slot =
  List.find
    (fun u ->
      Shape.is_variable u
      && match Ops.arg u with Param p -> p.slot = slot | _ -> false)
    uops

(* The interpreter's writes of the lowered sink, from the storage and variables
   that [Kernel_opts.writes] gives the kernel; a split kernel runs the block of
   its whole loop. *)
let keeps_writes row =
  let k = kernel row and u = lowered row in
  let whole =
    match (kernel_info u).split with
    | None -> []
    | Some s ->
        let name slot =
          Ops.expr (variable_of (Ops.toposort ~calls:Enter u) slot)
        in
        let hi = Shape.sym_infer s.iterations (Kernel_opts.variables k) in
        [ (name s.lo, `Int Bigint.zero); (name s.hi, `Int (Bigint.of_int hi)) ]
  in
  equal (list Kernel_opts.write) (Kernel_opts.writes k)
    (Interpreter.writes
       ~vars:(whole @ bound k)
       ~buffers:(Kernel_opts.inputs k) u)

(* The kernels that lower to stores alone, but that the interpreter does not
   read, each with its reason. *)
let not_interpreted =
  let fdiv = "divides with Ops.FDIV, which the interpreter does not compute" in
  [
    ("softmax", fdiv);
    ("exp_log", fdiv);
    ("exp_log_transcendental", fdiv);
    ("pow", fdiv);
    ("rmsnorm", fdiv);
    ("layernorm", fdiv);
    ("attention", fdiv);
    ("llama_ffn_gate", fdiv);
    ("llama_vector_scale", fdiv);
    ("threefry", "computes Ops.THREEFRY, which the interpreter does not compute");
    ("shard_sum", "stores a loaded vector, which the interpreter does not split");
    ("custom_eye", "stores through two indices");
    ("flip_contract", "stores through two indices");
    ("fp8", "reads its float8 storage as bytes");
    ("comparison_extrema", "reads a scalar parameter");
    ( "where_fold",
      "writes the element it reads, and the lowered kernel skips the store of \
       the value the element holds" );
    ("long_emulated", "reads its 64-bit storage as pairs of 32-bit words");
  ]
  @ List.map (fun case -> (case, "takes long to interpret")) heavy

let clang_rows_lowered_to_stores =
  List.filter
    (fun row ->
      row "target" = "clang"
      && compiles row
      && interpretable (kernel row)
      && interpretable (recorded_lowering row)
      && not (List.mem_assoc (row "case") not_interpreted))
    rows

(* The host's program is linearized for the host, then compiled by
   [Run.program]. *)
let host = lazy (Cstyle.clang host_target)
let host_uncompiled = lazy (renderer_for host_target)

(* The elements of each buffer that the run wrote: those it changed, and those
   the kernel writes ([expected]), whose values may equal what they held. *)
let written_by_run k ~expected outputs =
  let inputs = Kernel_opts.inputs k in
  let kernel_writes = Hashtbl.create (List.length expected) in
  List.iter (fun (s, j, _) -> Hashtbl.replace kernel_writes (s, j) ()) expected;
  List.concat_map
    (fun (slot, after) ->
      let before = List.assoc slot inputs in
      List.filteri
        (fun i _ ->
          (not (Testable.equal Dtypes.value before.(i) after.(i)))
          || Hashtbl.mem kernel_writes (slot, i))
        (Array.to_list (Array.mapi (fun i v -> (slot, i, v)) after)))
    outputs

(* The data type of each buffer, by slot. *)
let buffer_dtypes uops =
  List.filter_map
    (fun u ->
      match (Ops.op u, Ops.arg u) with
      | Op.Param, Param { slot; size = Some _; dtype; _ } when slot >= 0 ->
          Some (slot, dtype)
      | _ -> None)
    uops

(* A buffer of a type the target emulates is stored as the bits of its values,
   little-endian, in the type that emulates it: a float8 as a byte, a long as
   two ints. *)
let unsigned dt =
  match Dtype.itemsize dt with
  | 1 -> Dtype.Uint8
  | 2 -> Uint16
  | 4 -> Uint32
  | _ -> Uint64

let bits dt v =
  match Dtype.bitcast dt (unsigned dt) (Dtype.truncate dt v) with
  | `Int z -> z
  | _ -> invalid_arg "bits are an integer"

let words ~dt ~st v =
  let width = Dtype.bitsize st and z = bits dt v in
  List.init
    (Dtype.itemsize dt / Dtype.itemsize st)
    (fun i ->
      let word = Bigint.extract z (i * width) width in
      Dtype.bitcast (unsigned st) st (`Int word))

let of_words ~dt ~st ws =
  let width = Dtype.bitsize st in
  let z =
    List.fold_left
      (fun (acc, i) w ->
        (Bigint.logor acc (Bigint.shift_left (bits st w) (i * width)), i + 1))
      (Bigint.zero, 0) ws
    |> fst
  in
  Dtype.bitcast (unsigned dt) dt (`Int z)

(* A buffer the lowering left unread, as a load that folded away, is not
   bound. *)
let stored ~kernel ~program buffers =
  List.filter_map
    (fun (slot, values) ->
      let dt = List.assoc slot kernel in
      Option.map
        (fun st ->
          if Dtype.equal dt st then (slot, values)
          else
            ( slot,
              Array.of_list
                (List.concat_map (words ~dt ~st) (Array.to_list values)) ))
        (List.assoc_opt slot program))
    buffers

let read_back ~kernel ~program buffers =
  List.map
    (fun (slot, values) ->
      let dt = List.assoc slot kernel and st = List.assoc slot program in
      if Dtype.equal dt st then (slot, values)
      else
        let n = Dtype.itemsize dt / Dtype.itemsize st in
        ( slot,
          Array.init
            (Array.length values / n)
            (fun i ->
              of_words ~dt ~st (Array.to_list (Array.sub values (i * n) n))) ))
    buffers

let runs_as_interpreted name () =
  let row = row_named (name ^ "_clang") in
  let k = kernel row in
  let prg =
    under row (fun () -> Codegen.to_program k (Lazy.force host_uncompiled))
  in
  let uops = Ops.src (Ops.nth prg 1) in
  let kernel = buffer_dtypes (Ops.toposort ~calls:Enter k)
  and program = buffer_dtypes uops in
  let outputs =
    Run.on_host ~vars:(Kernel_opts.variables k)
      (Run.program (Lazy.force host) uops)
      (stored ~kernel ~program (Kernel_opts.inputs k))
  in
  let expected = Kernel_opts.writes k in
  equal (list Kernel_opts.write) expected
    (written_by_run k ~expected (read_back ~kernel ~program outputs))

(* Each kernel is compiled by the host's C compiler, so the default run runs a
   few of them. *)
let run_by_default = [ "add"; "gated_store"; "fp8"; "padto_idiv_upcast" ]

let run_on_host =
  [
    "add";
    "sum";
    "sum_all";
    "max";
    "matmul";
    "matmul_opts";
    "matmul_noopt";
    "sum_unroll";
    "pad";
    "where";
    "reduce_expand";
    "outer";
    "transpose";
    "flip";
    "cumsum";
    "conv";
    "idiv";
    "idiv_fast";
    "padto_idiv_upcast";
    "int8";
    "long";
    "long_emulated";
    "uint64";
    "bool_chain";
    "cast_half";
    "bf16";
    "fp8";
    "gated_store";
    "elementwise_add";
    "elementwise_int32";
    "elementwise_cast_f16";
    "sum_reduce";
    "max_reduce";
    "dot_product";
    "reduce_rows";
    "parallel_reduce";
    "two_outputs";
    "lorenz_fold";
    "no_optimize";
    "symbolic";
    "symbolic_sum";
    "zero_fold_upcast";
    "load_dedup_upcast";
    "reduce_upcast_unroll";
    "rewrite_reduction_opts";
    "linearizer_fail_1";
    "sum_acc_short";
    "sum_acc_half";
    "argmax";
    "triu";
    "triu_noopt";
    "where_fold";
  ]

let values =
  group "values"
    [
      group "the lowered sink writes what the kernel writes"
        (per_row
           ~only:(fun row -> List.memq row clang_rows_lowered_to_stores)
           keeps_writes);
      test
        "the interpreter reads elementwise, gated, symbolic and unrolled \
         kernels" (fun () ->
          let read =
            List.map (fun row -> row "case") clang_rows_lowered_to_stores
          in
          equal (list string) []
            (List.filter
               (fun case -> not (List.mem case read))
               [
                 "add";
                 "gated_store";
                 "symbolic";
                 "idiv_fast";
                 "zero_fold_upcast";
               ]));
      group "on the host, a program writes what its kernel writes"
        (List.map
           (fun name ->
             (if List.mem name run_by_default then Fun.id else slow)
             @@ test name (runs_as_interpreted name))
           run_on_host);
    ]

(* Beam search *)

let upcast4 = Opt.Split { axis = 0; amount = 4; target = Upcast; top = false }
let add_kernel = lazy (kernel_of "add_clang")

let asking ?opts_to_apply ~beam k =
  Ops.replace k ~arg:(Kernel { (kernel_info k) with beam; opts_to_apply })

(* A beam search that records the widths it is asked for and upcasts. *)
let recording () =
  let asked = ref [] in
  let beam w s =
    asked := w :: !asked;
    ignore (Result.get_ok (Postrange.Scheduler.apply_opt s upcast4));
    s
  in
  (asked, beam)

let searches_with_its_width () =
  let asked, beam = recording () in
  let searched =
    Codegen.full_rewrite_to_sink ~beam
      (asking ~beam:3 (Lazy.force add_kernel))
      clang
  in
  let listed =
    Codegen.full_rewrite_to_sink
      (asking ~opts_to_apply:[ upcast4 ] ~beam:0 (Lazy.force add_kernel))
      clang
  in
  equal (list int) ~msg:"the widths asked for" [ 3 ] !asked;
  equal (list Kernel_opts.opt) (kernel_info listed).applied_opts
    (kernel_info searched).applied_opts;
  same_graph (Ops.sink (Ops.src listed)) (Ops.sink (Ops.src searched))

let beam_search =
  group "beam search"
    [
      test "a kernel that asks for a beam of width w is optimised by beam w"
        searches_with_its_width;
      test
        "raises Invalid_argument when a kernel asks for a beam and none is \
         given" (fun () ->
          rejects (fun () ->
              Codegen.full_rewrite_to_sink
                (asking ~beam:3 (Lazy.force add_kernel))
                clang));
      test "to_program raises Invalid_argument too" (fun () ->
          rejects (fun () ->
              Codegen.to_program (asking ~beam:5 (Lazy.force add_kernel)) clang));
      test "a beam of width 1 is a search" (fun () ->
          let asked, beam = recording () in
          ignore
            (Codegen.full_rewrite_to_sink ~beam
               (asking ~beam:1 (Lazy.force add_kernel))
               clang);
          equal (list int) [ 1 ] !asked);
      test "a kernel that asks for no beam never searches" (fun () ->
          let asked, beam = recording () in
          ignore
            (Codegen.full_rewrite_to_sink ~beam
               (asking ~beam:0 (Lazy.force add_kernel))
               clang);
          equal (list int) [] !asked);
      test "the optimisations a kernel lists come before its beam" (fun () ->
          let asked, beam = recording () in
          ignore
            (Codegen.full_rewrite_to_sink ~beam
               (asking ~opts_to_apply:[] ~beam:3 (Lazy.force add_kernel))
               clang);
          equal (list int) [] !asked);
      test "an unoptimised kernel never searches" (fun () ->
          let asked, beam = recording () in
          ignore
            (Codegen.full_rewrite_to_sink ~optimize:false ~beam
               (asking ~beam:3 (Lazy.force add_kernel))
               clang);
          equal (list int) [] !asked);
    ]

(* Programs are kept *)

(* A kernel no other call compiles, even in a rerun of the suite: it stores a
   value of its own into 16 floats. *)
let fresh = Atomic.make 1000

let fresh_kernel ?opts_to_apply () =
  let value = Float.of_int (Atomic.fetch_and_add fresh 1) in
  let out = Call.param ~shape:[ Int 16 ] 0 Float32 in
  let r = Ops.range (Int 16) [ 0 ] in
  let store =
    Ops.store (Ops.index out [ r ]) (Ops.float ~dtype:Float32 value)
  in
  Ops.sink ~kernel:(Ops.kernel_info ?opts_to_apply ()) [ Ops.end_ store [ r ] ]

(* A compiler that counts its compilations, each taking long enough that domains
   calling at once all reach it before the first returns. *)
let counting () =
  let compiled = Atomic.make 0 in
  let compiler =
    Renderer.Compiler.v (fun src ->
        Atomic.incr compiled;
        Unix.sleepf 0.02;
        src)
  in
  (compiled, renderer_for ~compiler clang_target)

let keeps_its_programs () =
  let compiled, r = counting () in
  let k = fresh_kernel () in
  let p = Codegen.to_program k r in
  let again = Codegen.to_program (Graph.of_string (Graph.to_string k)) r in
  same_graph p again;
  equal int ~msg:"compilations" 1 (Atomic.get compiled)

let compiles_once_across_domains () =
  let compiled, r = counting () in
  let k = fresh_kernel () in
  let programs =
    List.init 4 (fun _ -> Domain.spawn (fun () -> Codegen.to_program k r))
    |> List.map Domain.join
  in
  List.iteri
    (fun i p ->
      same_graph ~msg:(Printf.sprintf "domain %d" i) (List.hd programs) p)
    programs;
  equal int ~msg:"compilations" 1 (Atomic.get compiled)

(* The heuristic unrolls a sum of 64 rows of 64 by 4, and NOOPT keeps it from
   it; the kernel is renamed so that no other test compiles it. *)
let separates_settings () =
  let sum = kernel_of "sum_clang" in
  let k = Ops.replace sum ~arg:(Kernel (Ops.kernel_info ~name:"settings" ())) in
  let optimised = Codegen.to_program k clang in
  let unoptimised =
    Setting.context
      [ B (Setting.noopt, true) ]
      (fun () -> Codegen.to_program k clang)
  in
  let applied p = (kernel_info (Ops.nth p 0)).applied_opts in
  let unroll4 =
    Opt.Split { axis = 1; amount = 4; target = Unroll; top = false }
  in
  equal (list Kernel_opts.opt) ~msg:"with the heuristic" [ unroll4 ]
    (applied optimised);
  equal (list Kernel_opts.opt) ~msg:"under NOOPT" [] (applied unoptimised)

(* Every setting a program depends on keeps a program made under another value
   from being returned: the program is made, and compiled, again. Among them, a
   setting its caller declares to reach output, of which codegen knows
   nothing. *)
let declared = Setting.int ~reach:Output "TOLK_TEST_CODEGEN" 0

let separates setting () =
  let compiled, r = counting () in
  let k = fresh_kernel () in
  ignore (Codegen.to_program k r);
  ignore (Setting.context [ setting ] (fun () -> Codegen.to_program k r));
  equal int ~msg:"compilations" 2 (Atomic.get compiled)

let target_of p =
  match Ops.arg p with
  | Program { target; _ } -> target.renderer
  | _ -> invalid_arg "a program holds its target"

let separates_targets () =
  let k = fresh_kernel () in
  equal string "CLANG" (target_of (Codegen.to_program k clang));
  equal string "METAL" (target_of (Codegen.to_program k metal))

(* Two compilers that differ only in their options, which each names in its key,
   made and used with CCACHE=0: a program compiled by one is not returned for
   the other. *)
let separates_compilers_without_ccache () =
  let compiled = Atomic.make 0 in
  let compiler options =
    Renderer.Compiler.v
      ~cachekey:(fun () -> "test_codegen_options_" ^ options)
      (fun src ->
        Atomic.incr compiled;
        options ^ src)
  in
  let k = fresh_kernel () in
  Setting.context
    [ B (Setting.ccache, false) ]
    (fun () ->
      List.iter
        (fun options ->
          ignore
            (Codegen.to_program k
               (renderer_for ~compiler:(compiler options) clang_target)))
        [ "-O1"; "-O2" ]);
  equal int ~msg:"compilations" 2 (Atomic.get compiled)

(* The compiler rejects its first source and accepts the next. *)
let makes_anew_after_a_failure () =
  let calls = Atomic.make 0 in
  let compiler =
    Renderer.Compiler.v (fun src ->
        if Atomic.fetch_and_add calls 1 = 0 then
          raise (Renderer.Compiler.Compile_error "once")
        else src)
  in
  let r = renderer_for ~compiler clang_target and k = fresh_kernel () in
  raises (Renderer.Compiler.Compile_error "once") (fun () ->
      Codegen.to_program k r);
  ignore (Codegen.to_program k r);
  equal int ~msg:"compilations" 2 (Atomic.get calls)

(* [made] waits for no program being made: a compiler that starts compiling,
   then holds the compilation, and with it the program's entry, until [made]
   has answered. *)
let made_waits_for_none () =
  let started = Atomic.make false and answered = Atomic.make false in
  let compiler =
    Renderer.Compiler.v (fun src ->
        Atomic.set started true;
        while not (Atomic.get answered) do
          Domain.cpu_relax ()
        done;
        src)
  in
  let r = renderer_for ~compiler clang_target and k = fresh_kernel () in
  let making = Domain.spawn (fun () -> Codegen.to_program k r) in
  while not (Atomic.get started) do
    Domain.cpu_relax ()
  done;
  is_none ~msg:"while it is made" (Codegen.made k r);
  Atomic.set answered true;
  let p = Domain.join making in
  match Codegen.made k r with
  | Some q -> same_graph p q
  | None -> fail "the made program is not returned"

let caching =
  group "programs are kept"
    [
      test "made is no program before to_program, then its program" (fun () ->
          let _, r = counting () in
          let k = fresh_kernel () in
          is_none ~msg:"before" (Codegen.made k r);
          let p = Codegen.to_program k r in
          match Codegen.made k r with
          | Some q -> same_graph p q
          | None -> fail "the made program is not returned");
      test "made waits for no program being made" made_waits_for_none;
      test "a second call with an equal kernel compiles nothing"
        keeps_its_programs;
      test "calls from several domains at once make one program, compiled once"
        compiles_once_across_domains;
      test "a program made under one setting is not returned under another"
        separates_settings;
      group "a program is made again under another value of"
        (List.map
           (fun (name, setting) -> test name (separates setting))
           Setting.
             [
               ("NOOPT", B (noopt, true));
               ("TC", B (use_tc, 0));
               ("TC_SELECT", B (tc_select, 0));
               ("TC_OPT", B (tc_opt, 1));
               ("BEAM", B (beam, 1));
               ("JITBEAM", B (jitbeam, Some 1));
               ("TC_MIN_GLOBALS", B (tc_min_globals, 1));
               ("TRANSCENDENTAL", B (transcendental, 2));
               ("DISABLE_FAST_IDIV", B (disable_fast_idiv, false));
               ("ALLOW_TF32", B (allow_tf32, true));
               ("DEFAULT_FLOAT", B (default_float, "half"));
               ("DEFAULT_INT", B (default_int, "long"));
               ("EMULATED_DTYPES", B (emulated_dtypes, [ "long" ]));
               ("TUPLE_ORDER", B (tuple_order, false));
               ("a setting its caller declares", B (declared, 1));
             ]);
      test
        "a program is not returned for a compiler of other options, with \
         CCACHE=0"
        separates_compilers_without_ccache;
      test "a program for one target is not returned for another"
        separates_targets;
      test "a failed compilation is not kept: the next call compiles anew"
        makes_anew_after_a_failure;
    ]

(* Linearizing *)

let completes_linearize =
  let rows = List.filter (fun row -> compiles row && by_default row) rows in
  prop ~count:20 "compile completes linearize into the program to_program makes"
    Gen.(
      with_pp
        (fun ppf row -> Format.pp_print_string ppf (row "name"))
        (of_list rows))
    (fun row ->
      under row (fun () ->
          let k = kernel row and ren = renderer_of_row row in
          let lin = Codegen.linearize k ren in
          let prg = Codegen.to_program k ren in
          same_graph ~msg:"its first two" (first 2 prg) lin;
          same_graph prg (Codegen.compile lin ren)))

let linearizing =
  group "linearize and compile"
    [
      completes_linearize;
      test "linearize compiles nothing" (fun () ->
          let compiled, r = counting () in
          ignore (Codegen.linearize (fresh_kernel ()) r);
          equal int 0 (Atomic.get compiled));
      test "compile keeps nothing: each call compiles" (fun () ->
          let compiled, r = counting () in
          let lin = Codegen.linearize (fresh_kernel ()) r in
          ignore (Codegen.compile lin r);
          ignore (Codegen.compile lin r);
          equal int 2 (Atomic.get compiled));
      test "compile refuses what linearize does not make" (fun () ->
          let k = fresh_kernel () in
          rejects (fun () -> Codegen.compile k clang);
          rejects (fun () -> Codegen.compile (Codegen.to_program k clang) clang));
      test "to_program refuses a program" (fun () ->
          let lin = Codegen.linearize (fresh_kernel ()) clang in
          rejects (fun () -> Codegen.to_program lin clang));
    ]

(* Programs on disk

   A child makes the program of add_clang's kernel and prints it. With
   ASSERT_COMPILE set, a program it would compile fails it instead, so a child
   that succeeds under ASSERT_COMPILE read its program from the disk. Programs
   are kept on disk only for a compiler that caches its binaries: [kept]'s
   binaries are their sources' bytes, kept in a table of their own. *)

let bytes_in table = Renderer.Compiler.v ~cachekey:(fun () -> table) Fun.id
let kept = renderer_for ~compiler:(bytes_in "test_codegen_bytes") clang_target

let printed ren () =
  print_string
    (Graph.to_string (Codegen.to_program (Lazy.force add_kernel) ren))

let parts =
  [
    ("program", printed kept);
    ("uncached", printed clang);
    ( "other compiler",
      printed
        (renderer_for ~compiler:(bytes_in "test_codegen_other") clang_target) );
  ]

let made ?env ?(part = "program") db = Disk_cache.child ?env ~cachedb:db part

let read_from_disk ?(env = []) ?part db =
  made ~env:(("ASSERT_COMPILE", "1") :: env) ?part db

let compiled = Error "tried to compile with ASSERT_COMPILE set"
let outcome = Disk_cache.outcome

let the_program () =
  Ok (Graph.to_string (Codegen.to_program (Lazy.force add_kernel) kept))

let reads_back () =
  let db = Disk_cache.fresh () in
  equal outcome ~msg:"made" (the_program ()) (made db);
  equal outcome ~msg:"read back" (the_program ()) (read_from_disk db)

(* A program made anew is kept beside the first; the binary of its source may be
   read back. *)
let misses_on env () =
  let db = Disk_cache.fresh () in
  equal outcome ~msg:"made" (the_program ()) (made db);
  (match made ~env db with
  | Ok _ -> ()
  | Error err -> failf "the child failed: %s" err);
  equal int ~msg:"programs kept" 2
    (List.length (Disk_cache.entries ~table:"to_program" db))

let recovers damage () =
  let db = Disk_cache.fresh () in
  equal outcome ~msg:"made" (the_program ()) (made db);
  Disk_cache.damage ~table:"to_program" db damage;
  equal outcome ~msg:"made again" (the_program ()) (made db);
  equal outcome ~msg:"then read back" (the_program ()) (read_from_disk db)

(* A program read back takes its binary from its compiler's table: with the
   binaries removed, it is compiled again. *)
let takes_its_binary () =
  let db = Disk_cache.fresh () in
  equal outcome ~msg:"made" (the_program ()) (made db);
  List.iter Sys.remove (Disk_cache.entries ~table:"test_codegen_bytes" db);
  equal outcome ~msg:"binaries removed" compiled (read_from_disk db);
  equal outcome ~msg:"made again" (the_program ()) (made db);
  equal outcome ~msg:"then read back" (the_program ()) (read_from_disk db)

let shows_its_source () =
  let db = Disk_cache.fresh () in
  equal outcome ~msg:"made" (the_program ()) (made db);
  let prg = Codegen.to_program (Lazy.force add_kernel) kept in
  match read_from_disk ~env:[ ("DEBUG", "4") ] db with
  | Ok out -> contains ~sub:(source prg) out
  | Error err -> failf "the child failed: %s" err

(* An entry of another build is made anew, and replaced by this build's. *)
let ignores_other_builds () =
  let db = Disk_cache.fresh () in
  equal outcome ~msg:"made" (the_program ()) (made db);
  let programs () =
    List.map
      (fun e -> In_channel.with_open_bin e In_channel.input_all)
      (Disk_cache.entries ~table:"to_program" db)
  in
  let kept = programs () in
  Disk_cache.damage ~table:"to_program" db Disk_cache.of_another_build;
  equal outcome ~msg:"made again" (the_program ()) (made db);
  equal (list string) ~msg:"replaced" kept (programs ())

let keeps_no_uncached () =
  let db = Disk_cache.fresh () in
  equal outcome ~msg:"made" (the_program ()) (made ~part:"uncached" db);
  equal outcome compiled (read_from_disk ~part:"uncached" db)

let other_compiler () =
  let db = Disk_cache.fresh () in
  equal outcome ~msg:"made" (the_program ()) (made db);
  equal outcome compiled (read_from_disk ~part:"other compiler" db)

let races () =
  let db = Disk_cache.fresh () in
  let children =
    List.init 4 (fun _ -> Disk_cache.start ~cachedb:db "program")
  in
  List.iter
    (fun c -> equal outcome (the_program ()) (Disk_cache.finish c))
    children;
  equal outcome ~msg:"read back" (the_program ()) (read_from_disk db)

(* Another value of each setting and variable that shapes what compilation
   makes, as the environment holds it: a flag flipped, a number plus one, and a
   string from this table, which a string declared later must join.
   DEFAULT_FLOAT retypes the recorded kernel the child reads, so its key is
   tested in memory, where the key is the same. *)
let other_values () =
  let strings =
    [
      ("DEFAULT_INT", "long");
      ("EMULATED_DTYPES", "long");
      ("HCQ_NUM_SDMA", "2");
      ("JITBEAM", "1");
      ("SUM_DTYPE", "half");
    ]
  in
  List.map
    (fun (name, value) ->
      match (value, int_of_string_opt value, float_of_string_opt value) with
      | "true", _, _ -> (name, "0")
      | "false", _, _ -> (name, "1")
      | _, Some n, _ -> (name, string_of_int (n + 1))
      | _, None, Some x -> (name, string_of_float (x +. 1.))
      | _, None, None -> (
          match List.assoc_opt name strings with
          | Some other -> (name, other)
          | None -> failf "%s holds a string: give it another value" name))
    (List.remove_assoc "DEFAULT_FLOAT" (Setting.shaping ()))

let on_disk =
  group "programs are kept on disk"
    [
      test "a program made by one process is read back by the next" reads_back;
      group "a program made under one setting is not read back under another"
        (List.map
           (fun (name, value) -> test name (misses_on [ (name, value) ]))
           (other_values ()));
      test "a program read back shows its source at DEBUG 4" shows_its_source;
      test "a program read back takes its binary from its compiler's table"
        takes_its_binary;
      group "a damaged entry is made anew, and replaced"
        [
          test "truncated" (recovers Disk_cache.truncated);
          test "holding no program" (recovers Disk_cache.not_a_graph);
        ];
      test "an entry of another build of the library is not read back"
        ignores_other_builds;
      test "a program of a compiler that caches no binaries is not kept"
        keeps_no_uncached;
      test "a program is not read back for a compiler of another table"
        other_compiler;
      test "processes making one program at once all get it" races;
    ]

(* Programs *)

let on_clang row = row "target" = "clang" && compiles row

let estimates row =
  let p = program row in
  match (kernel_info (Ops.nth p 0)).estimates with
  | Some e -> Format.asprintf "%a" Ops.pp_estimates e
  | None -> "none"

(* A split program's instructions count one block; its estimates count the block
   of the whole loop. *)
let whole_loop p (e : Ops.estimates) : Ops.estimates =
  match (kernel_info (Ops.nth p 0)).split with
  | None -> e
  | Some s ->
      let lo = variable_of (Ops.src (Ops.nth p 1)) s.lo
      and hi = variable_of (Ops.src (Ops.nth p 1)) s.hi in
      let bounds =
        [
          (lo, Ops.int ~dtype:(Ops.dtype lo) 0);
          (hi, Ops.sint_to_uop ~dtype:(Ops.dtype hi) s.iterations);
        ]
      in
      let fill : Ops.sint -> Ops.sint = function
        | Int _ as i -> i
        | Sym u ->
            Shape.ssimplify
              (Ops.substitute ~calls:Skip ~pass:Fixed_point u bounds)
      in
      { ops = fill e.ops; lds = fill e.lds; mem = fill e.mem }

let counts_its_instructions row =
  let p = program row in
  equal string
    (Format.asprintf "%a" Ops.pp_estimates
       (whole_loop p
          (Renderer.Estimates.of_uops ~ignore_indexing:true
             (Ops.src (Ops.nth p 1)))))
    (estimates row)

let lowers_the_same_twice row = same_graph (lowered row) (lowered row)

let binary_of p =
  match Ops.arg (Ops.nth p 3) with
  | Bytes b -> b
  | _ -> invalid_arg "a program's fourth source is its binary"

(* A kernel that stores the sum of ten int variables: on arm64, the ones past
   the eighth argument register go on the stack at 4 bytes each. *)
let ten_scalars () =
  let vars =
    List.init 10 (fun i ->
        Ops.variable ~dtype:Int32 (Printf.sprintf "v%d" i) (`Int Bigint.zero)
          (`Int (Bigint.of_int 100)))
  in
  let out = Call.param ~shape:[ Int 1 ] 0 Int32 in
  let sum = List.fold_left Ops.add (List.hd vars) (List.tl vars) in
  let k =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"ten" ())
      [ Ops.store (Ops.index out [ Ops.int ~dtype:Int32 0 ]) sum ]
  in
  let p = Codegen.to_program k (Lazy.force host) in
  let elf = Device.Tiny_elf.of_program p in
  equal ~msg:"the entry" string "ten" elf.name;
  let value v =
    List.assoc (Ops.expr v) (List.mapi (fun i v -> (Ops.expr v, i + 1)) vars)
  in
  let info = match Ops.arg p with Program info -> info | _ -> assert false in
  let result = Bigarray.(Array1.create int32 c_layout 1) in
  (* The parameters are the pointer to [out], then each variable, a 32-bit
     integer. *)
  (match Rig.Image.load Rig.host elf.lib with
  | Ok image ->
      let fn = require_some (Rig.Image.resolve image elf.name) in
      let s =
        Rig.Submission.(
          make Rig.host
            [|
              launch
                ~refs:[| write 0 ~at:0 |]
                fn
                ~params:(8 + (4 * List.length info.vars));
            |])
      in
      let run = Rig.Submission.Run.make s in
      let b = Rig.Submission.block s 0 in
      Rig.Submission.Run.groups run b 1 1 1;
      Rig.Submission.Run.threads run b 1 1 1;
      List.iteri
        (fun i v -> Rig.Submission.Run.int32 run b (8 + (4 * i)) (value v))
        info.vars;
      ignore
        (Rig.submit run ~buffers:[| Rig.Buffer.of_bigarray result |]
          : Rig.Point.t)
  | Error why -> fail why);
  equal ~msg:"1 + 2 + ... + 10" int32 55l result.{0}

let programs =
  group "programs"
    [
      group "the estimates count the instructions, leaving out index arithmetic"
        (per_row ~only:compiles counts_its_instructions);
      group "full_rewrite_to_sink lowers a kernel the same each time"
        (per_row ~only:on_clang ~default:sample lowers_the_same_twice);
      test
        "a device program's binary is its source compiled by the renderer's \
         compiler" (fun () ->
          let p = Codegen.to_program (Lazy.force add_kernel) metal in
          equal string (source p) (binary_of p));
      test "a host program's binary holds rig's host entry after the kernel"
        (fun () ->
          let p = Codegen.to_program (Lazy.force add_kernel) clang in
          equal string
            (String.concat "\n"
               [
                 "#define E_16_16 E_16_16_";
                 source p;
                 "#undef E_16_16";
                 "struct E_16_16_args { void *b0; void *b1; void *b2; long \
                  long n; };";
                 "void E_16_16(const struct E_16_16_args *p, long long first, \
                  long long end) { if (end > p->n) end = p->n; if (first < \
                  end) E_16_16_(p->b0, p->b1, p->b2, first, end); }";
               ])
            (binary_of p));
      test
        "a host program runs as a launch of rig's host, with ten scalars past \
         the argument registers"
        ten_scalars;
      test "the argument of a program is its program information for the target"
        (fun () ->
          let p = Codegen.to_program (Lazy.force add_kernel) clang in
          equal string
            (Format.asprintf "%a" Ops.pp_program_info
               (Call.program_info_of_sink ~target:clang_target (Ops.nth p 0)))
            (program_info p));
    ]

(* tinygrad's claims on programs *)

let instructions row = Ops.src (Ops.nth (program row) 1)
let is op u = Op.equal (Ops.op u) op
let count op uops = List.length (List.filter (is op) uops)
let alu uops = List.filter (fun u -> Op.Set.mem (Ops.op u) Op.Set.alu) uops
let in_space space u = Ops.addrspace u = Some space

let claim case name check =
  group
    (case ^ ": " ^ name)
    (per_row ~only:(fun row -> row "case" = case && compiles row) check)

let at_most bound what n =
  satisfies
    ~claim:(Printf.sprintf "%s at most %d" what bound)
    int
    (fun n -> n <= bound)
    n

(* The first memory an unoptimised program allocates for itself holds its
   accumulator. *)
let accumulates_in dt row =
  let own u = is Buffer u && (in_space Local u || in_space Reg u) in
  equal Dtypes.dtype dt (Ops.dtype (List.find own (instructions row)))

(* The first two stores outside registers: to local memory, then to the
   output. *)
let stores_outside_registers row =
  List.filter
    (fun u -> is Store u && not (in_space Reg (Ops.nth u 0)))
    (instructions row)

let stages_through_locals row =
  match stores_outside_registers row with
  | first :: last :: _ ->
      equal int ~msg:"lanes of the local store" 4
        (Shape.max_numel (Ops.nth first 1));
      is_true ~msg:"the first store is to local memory"
        (List.exists (in_space Local) (Ops.toposort ~calls:Enter first));
      equal int ~msg:"lanes of the global store" 1
        (Shape.max_numel (Ops.nth last 1));
      equal Dtypes.dtype Float32 (Ops.dtype (Ops.nth last 1));
      is_true ~msg:"the last store is to a parameter"
        (List.exists (is Param) (Ops.toposort ~calls:Enter last))
  | stores ->
      failf "a local then a global store, not %d stores" (List.length stores)

let loops_as_written row =
  let order =
    List.filter_map
      (fun u ->
        if is Range u || is Store u || is End u then Some (Ops.op u) else None)
      (instructions row)
  in
  equal
    (list (Testable.make ~pp:Op.pp ~equal:Op.equal))
    Op.[ Range; Range; Store; End; Store; End ]
    order;
  let ranges = List.filter (is Range) (instructions row) in
  let ended =
    List.map (fun e -> Ops.nth e 1) (List.filter (is End) (instructions row))
  in
  equal (list Uops.uop) (List.rev ranges) ended

let launches_reversed row =
  let specials =
    List.filter (is Special) (instructions row)
    |> List.map (fun u ->
        match (Ops.arg u, Ops.vmax u) with
        | String name, `Int z -> (name, Bigint.to_int z + 1)
        | _ -> invalid_arg "a hardware index is named and bounded")
    |> List.sort_uniq compare
  in
  equal
    (list (pair string int))
    [ ("gidx0", 6); ("gidx1", 5); ("gidx2", 4) ]
    specials

let arange_without_phis row =
  let uops = instructions row in
  let uops =
    match List.find_index (is If) uops with
    | Some i -> List.filteri (fun j _ -> j < i) uops
    | None -> uops
  in
  is_false ~msg:"both hardware indices and loops"
    (List.exists (is Special) uops && List.exists (is Range) uops);
  equal int ~msg:"stores to registers" 0
    (List.length
       (List.filter (fun u -> is Store u && in_space Reg (Ops.nth u 0)) uops));
  at_most 2 "maximums" (count Max uops)

let wide u = Dtype.equal (Ops.dtype u) Int64 || Dtype.equal (Ops.dtype u) Uint64

let wide_alu row =
  List.length
    (List.filter wide
       (alu (Ops.toposort ~calls:Enter (Ops.nth (program row) 0))))

let estimated_ops row =
  match (kernel_info (Ops.nth (program row) 0)).estimates with
  | Some { ops = Int n; _ } -> n
  | _ -> invalid_arg "a program's estimates count its operations"

let claims =
  group "tinygrad's claims on programs"
    [
      claim "load_dedup_upcast" "loads each element at most once" (fun row ->
          let loads = count Load (instructions row) in
          at_most 4 "loads" loads;
          is_true ~msg:"a load" (loads >= 1));
      claim "zero_fold_upcast" "stacks two values without arithmetic"
        (fun row -> equal (list Uops.uop) [] (alu (instructions row)));
      claim "reduce_upcast_unroll" "keeps no accumulator and stores once"
        (fun row ->
          let uops = instructions row in
          equal int ~msg:"register buffers" 0
            (List.length
               (List.filter (fun u -> is Buffer u && in_space Reg u) uops));
          equal int ~msg:"stores" 1 (count Store uops));
      claim "sum_acc_bool" "sums booleans in an int" (accumulates_in Int32);
      claim "sum_acc_short" "sums shorts in an int" (accumulates_in Int32);
      claim "sum_acc_half" "sums halves in a float" (accumulates_in Float32);
      claim "sum_acc_bf16" "sums bfloat16s in a float" (accumulates_in Float32);
      claim "matmul_acc_half" "accumulates in the half it asks for"
        (accumulates_in Float16);
      claim "upcast_with_locals_opts"
        "stores four lanes to local memory, then one float to the output"
        stages_through_locals;
      claim "dependent_loop_bound"
        "closes each loop after its stores, inner first" loops_as_written;
      claim "two_nested_range" "collapses the broadcast sum to one loop"
        (fun row -> equal int 1 (count Range (instructions row)));
      claim "three_nested_range" "collapses the broadcast sum to one loop"
        (fun row -> equal int 1 (count Range (instructions row)));
      claim "range_outer_op_before_phi_nested_range" "collapses to one loop"
        (fun row -> equal int 1 (count Range (instructions row)));
      claim "default_global_reversed" "launches the last axis first"
        launches_reversed;
      claim "where_fold" "folds the select of an assignment away" (fun row ->
          equal int 0 (count Where (instructions row)));
      claim "phi_arange_float" "computes an arange without accumulators"
        arange_without_phis;
      claim "phi_arange_negative" "computes an arange without accumulators"
        arange_without_phis;
      claim "phi_arange_255" "computes an arange without accumulators"
        arange_without_phis;
      claim "two_grouped_stores_local" "puts a barrier after each local store"
        (fun row ->
          let uops = instructions row in
          let local_store u = is Store u && in_space Local (Ops.nth u 0) in
          equal int
            (List.length (List.filter local_store uops))
            (count Barrier uops));
      claim "reduce_shapeless_const_unroll"
        "folds the sum of a constant over an unroll" (fun row ->
          let nodes = Ops.toposort ~calls:Enter (lowered row) in
          equal int ~msg:"reductions" 0 (count Reduce nodes);
          mem Dtypes.const (`Float 12.)
            (List.filter_map
               (fun u -> match Ops.arg u with Const c -> Some c | _ -> None)
               nodes));
      claim "threefry" "computes random bits without 64-bit values" (fun row ->
          if row "target" = "clang" then
            equal (list Uops.uop) [] (List.filter wide (instructions row)));
      claim "long" "computes a long in 64 bits" (fun row ->
          is_true ~msg:"a 64-bit operation" (wide_alu row > 0));
      claim "fancy_index" "indexes without 64-bit arithmetic" (fun row ->
          equal int 0 (wide_alu row));
      claim "cat" "concatenates in 20 operations an element" (fun row ->
          at_most (((2 * 1024) + 1) * 20) "operations" (estimated_ops row));
      claim "triu_noopt" "masks in 4 operations an element" (fun row ->
          at_most (4 * 256 * 256) "operations" (estimated_ops row));
      test "a kernel that lists no optimisation is compiled without any"
        (fun () ->
          let k = kernel_of "arange_clang" in
          let k =
            Ops.replace k ~arg:(Kernel (Ops.kernel_info ~opts_to_apply:[] ()))
          in
          let p = Codegen.to_program k clang in
          equal (list Kernel_opts.opt) []
            (kernel_info (Ops.nth p 0)).applied_opts);
      test "a kernel named by its argument keeps its name" (fun () ->
          let k = kernel_of "arange_clang" in
          let k =
            Ops.replace k ~arg:(Kernel (Ops.kernel_info ~name:"custom" ()))
          in
          let p = Codegen.to_program k clang in
          equal string "custom" (kernel_info (Ops.nth p 0)).name);
    ]

(* tinygrad's claims on lowered kernels *)

let lowered_nodes row = Ops.toposort ~calls:Enter (lowered row)

let one_ending ending row =
  match List.filter (is ending) (lowered_nodes row) with
  | [ e ] -> e
  | es -> failf "one %a, not %d" Op.pp ending (List.length es)

let ends_with_a_barrier ending row =
  let e = one_ending ending row in
  let barrier = Ops.nth e 0 in
  equal (Testable.make ~pp:Op.pp ~equal:Op.equal) Op.Barrier (Ops.op barrier);
  equal int ~msg:"sources of the barrier" 1 (List.length (Ops.src barrier));
  is_false ~msg:"a barrier around a barrier" (is Barrier (Ops.nth barrier 0))

let slots_of op nodes =
  List.filter_map
    (fun u ->
      match Ops.arg u with
      | Param { slot; _ } when is op u -> Some slot
      | _ -> None)
    nodes
  |> List.sort compare

let weak u = Dtype.equal (Ops.dtype u) Weak_int

(* The constants a kernel stores, each committed to its type by a cast. *)
let stored_constants row =
  List.filter_map
    (fun u ->
      if is Store u then
        let v = Ops.nth u 1 in
        let v = if is Cast v then Ops.nth v 0 else v in
        match Ops.arg v with Const c -> Some c | _ -> None
      else None)
    (lowered_nodes row)

let lowering_claims =
  group "tinygrad's claims on lowered kernels"
    [
      claim "shared_loop_barrier"
        "a loop that reads the local memory it writes ends with a barrier"
        (ends_with_a_barrier End);
      claim "shared_backedge_barrier"
        "so does an unbounded loop, which keeps its condition" (fun row ->
          ends_with_a_barrier Backedge row;
          let e = one_ending Backedge row in
          equal Uops.uop ~msg:"the condition" (Call.param 2 Bool) (Ops.nth e 2));
      claim "shared_backedge_two_buffers"
        "a loop that reads another local buffer needs no barrier" (fun row ->
          is_false (is Barrier (Ops.nth (one_ending Backedge row) 0)));
      claim "shared_backedge_barrier_kept"
        "a barrier already there is kept alone"
        (ends_with_a_barrier Backedge);
      claim "explicit_local_slots"
        "an accumulator and a staged buffer take the slots after the kernel's"
        (fun row ->
          equal (list int) [ 17; 18; 19 ] (slots_of Alloc (lowered_nodes row)));
      claim "anonymous_local"
        "a local buffer is allocated until the program declares it" (fun row ->
          equal (list int) ~msg:"lowered" [ 17 ]
            (slots_of Alloc (lowered_nodes row));
          equal (list int) ~msg:"linearized, allocated" []
            (slots_of Alloc (instructions row));
          equal (list int) ~msg:"linearized, declared" [ 17 ]
            (slots_of Buffer (instructions row)));
      claim "weak_index_store" "leaves a weak type only to constants"
        (fun row ->
          equal (list Uops.uop) []
            (List.filter
               (fun u -> weak u && not (is Const u))
               (lowered_nodes row)));
      claim "negated_index" "computes on loads, never on addresses" (fun row ->
          let computes_on_an_address u =
            Op.Set.mem (Ops.op u) Op.Set.alu
            && List.exists (is Index) (Ops.src u)
          in
          equal (list Uops.uop) []
            (List.filter computes_on_an_address (lowered_nodes row)));
      claim "numbered_variables"
        "numbers variables after the buffers, a slot each" (fun row ->
          let variables =
            List.filter_map
              (fun u ->
                match Ops.arg u with
                | Param { slot; name = Some name; size = None; _ }
                  when is Param u ->
                    Some (name, slot)
                | _ -> None)
              (lowered_nodes row)
          in
          equal (slist int compare) [ 1; 2 ] (List.map snd variables));
      claim "comparison_extrema" "folds comparisons with the bounds of a long"
        (fun row ->
          equal (list Dtypes.const)
            [ `Bool false; `Bool true; `Bool true; `Bool false ]
            (stored_constants row));
    ]

(* Whole graphs *)

(* tinygrad's tests lower a sink of values with a renderer that writes nothing,
   without checking the specification. *)
let plain = Renderer.v (target "" "" "")

let full_rewrite ?(ren = plain) uops =
  Setting.context
    [ B (Setting.spec, 0) ]
    (fun () ->
      Codegen.full_rewrite_to_sink
        (Ops.sink ~kernel:(Ops.kernel_info ()) uops)
        ren)

let apply_rewrite u = Ops.nth (full_rewrite [ u ]) 0

let const_value u =
  let u = if is Cast u then Ops.nth u 0 else u in
  match Ops.arg u with
  | Const c -> c
  | _ -> failf "a constant, not %a" Op.pp (Ops.op u)

let floats xs = Shape.stack (List.map (Ops.float ~dtype:Float32) xs)
let ints ?(dtype = Dtype.Int32) xs = Shape.stack (List.map (Ops.int ~dtype) xs)

let whole_graphs =
  let v = floats [ 1.; 2.; 3.; 4. ] in
  group "full_rewrite_to_sink folds whole graphs"
    [
      test "log2 of -1 is NaN and the reciprocal of 0 is infinity" (fun () ->
          match
            Ops.src
              (full_rewrite
                 [ Ops.log2 (Ops.float (-1.)); Ops.reciprocal (Ops.float 0.) ])
          with
          | [ log; recip ] ->
              equal Dtypes.const ~msg:"log2 (-1)" (`Float Float.nan)
                (const_value log);
              equal Dtypes.const ~msg:"1 / 0" (`Float Float.infinity)
                (const_value recip)
          | _ -> fail "two values");
      test "a lane of a vector is the lane's value" (fun () ->
          equal Uops.uop
            (apply_rewrite (Ops.nth v 2))
            (apply_rewrite (Ops.index v [ Ops.int 2 ])));
      test "a vector of lanes is the vector of their values" (fun () ->
          equal Uops.uop
            (apply_rewrite (Shape.stack [ Ops.nth v 2; Ops.nth v 3 ]))
            (apply_rewrite
               (Shape.stack
                  [ Ops.index v [ Ops.int 2 ]; Ops.index v [ Ops.int 3 ] ])));
      test "a vector of every lane of a vector is the vector" (fun () ->
          equal Uops.uop (apply_rewrite v)
            (apply_rewrite
               (Shape.stack (List.init 4 (fun i -> Ops.index v [ Ops.int i ])))));
      test "a comparison of lanes keeps the lanes apart" (fun () ->
          let lidx = Ops.special (Int 32) "lidx0" in
          let lhs =
            Ops.add
              (Shape.stack [ lidx; lidx; lidx; lidx ])
              (ints ~dtype:Weak_int [ 0; 256; 512; 768 ])
          in
          let folded =
            apply_rewrite (Ops.lt lhs (ints ~dtype:Weak_int [ 2; 2; 2; 2 ]))
          in
          if is Stack folded then
            match Ops.src folded with
            | first :: rest ->
                is_false ~msg:"every lane is the first"
                  (List.for_all (Ops.equal first) rest)
            | [] -> fail "a stack of four lanes");
      test "a bitcast of constants folds lane by lane" (fun () ->
          let bits =
            full_rewrite [ Ops.bitcast (ints [ -1; -0x80000000; 75 ]) Uint32 ]
          in
          let expected =
            full_rewrite [ ints ~dtype:Uint32 [ 0xffffffff; 0x80000000; 75 ] ]
          in
          equal (list Uops.uop) (Ops.src expected) (Ops.src bits));
      test "a cast of constants folds lane by lane" (fun () ->
          equal (list Uops.uop)
            (Ops.src (full_rewrite [ floats [ 1.; 2. ] ]))
            (Ops.src (full_rewrite [ Ops.cast (ints [ 1; 2 ]) Float32 ])));
    ]

(* Instructions of values *)

(* tinygrad's test helper: the instructions of the kernel that computes [uops],
   each of their ranges ended, lowered for [ren] without optimising. *)
let instructions_of ?(ren = clang) uops =
  let sink = Ops.sink uops in
  let sink = Ops.end_ sink (Ops.Nodes.to_list (Ops.ranges sink)) in
  let kernel =
    Ops.sink ~kernel:(Ops.kernel_info ~opts_to_apply:[] ()) [ sink ]
  in
  Codegen.line_rewrite
    (Linearizer.linearize (Codegen.full_rewrite_to_sink kernel ren))
    Codegen.pm_linearize_cleanups ()

let ops uops = List.map Ops.op uops
let op = Testable.make ~pp:Op.pp ~equal:Op.equal
let has o uops = mem op o (ops uops)

let lacks o uops =
  equal int ~msg:(Format.asprintf "%a" Op.pp o) 0 (count o uops)

let between_if uops =
  let if_ = List.find (is If) uops and endif = List.find (is Endif) uops in
  equal Uops.uop ~msg:"the endif closes the if" if_ (Ops.nth endif 0);
  let rec after_if = function
    | u :: rest when u == if_ -> rest
    | _ :: rest -> after_if rest
    | [] -> []
  in
  let rec until_endif = function
    | u :: _ when u == endif -> []
    | u :: rest -> u :: until_endif rest
    | [] -> []
  in
  until_endif (after_if uops)

let gated_store_inside_its_if stores =
  match between_if (instructions_of ~ren:metal stores) with
  | [ st ] ->
      is_true ~msg:"a store" (is Store st);
      equal int ~msg:"sources of the store" 2 (List.length (Ops.src st))
  | inside ->
      failf "one store inside the if, not %d instructions" (List.length inside)

let gidx0 = Ops.special (Int 4) "gidx0"
let forty_two = Ops.cast (Ops.float 42.) Float32

let gated_index slot =
  Ops.index
    (Call.param ~shape:[ Int 8 ] slot Float32)
    [ Shape.valid (Ops.mul gidx0 (Ops.int 2)) (Ops.lt gidx0 (Ops.int 1)) ]

let loaded ?(slot = 0) ?(size = 3) dt i =
  Ops.index (Call.param ~shape:[ Int size ] slot dt) [ Ops.int i ]

let fast_idiv = [ Setting.B (Setting.disable_fast_idiv, false) ]

(* Clang's operations without a maximum, as tinygrad's base C renderer. *)
let without_max =
  Renderer.v
    ~code_for_op:
      (List.filter (fun (o, _) -> not (Op.equal o Max)) clang.code_for_op)
    (target "" "" "")

let gated_stores =
  group "gated stores"
    [
      test "a store gated by its index is the one store inside an if" (fun () ->
          gated_store_inside_its_if [ Ops.store (gated_index 0) forty_two ]);
      test "an ungated store beside it stays out of the if" (fun () ->
          let ungated =
            Ops.index
              (Call.param ~shape:[ Int 8 ] 1 Float32)
              [ Ops.mul gidx0 (Ops.int 2) ]
          in
          gated_store_inside_its_if
            [ Ops.store (gated_index 0) forty_two; Ops.store ungated forty_two ]);
    ]

let divisions =
  let power_of_two dt =
    let uops = instructions_of [ Ops.alu (loaded dt 2) Cdiv [ Ops.int 2 ] ] in
    has Shr uops;
    lacks Cdiv uops
  in
  let dtype_cases name dts f =
    cases name ~id:(Format.asprintf "%a" Dtype.pp) dts f
  in
  group "divisions"
    [
      dtype_cases "a division by a power of two is a shift" [ Int32; Uint32 ]
        power_of_two;
      dtype_cases "a floor remainder by a power of two is a mask"
        [ Int32; Uint32 ] (fun dt ->
          let uops =
            instructions_of
              [ Ops.alu (loaded ~size:9 dt 8) Floormod [ Ops.int 8 ] ]
          in
          has And uops;
          lacks Cmod uops;
          lacks Floormod uops);
      test "a maximum proves its dividend positive before it is decomposed"
        (fun () ->
          let uops =
            instructions_of ~ren:without_max
              [ Ops.O.(Ops.maximum (loaded Int32 2) (int 0) // int 3) ]
          in
          lacks Max uops;
          lacks Cmod uops);
      test "a dividend that can wrap keeps the correction of a floor division"
        (fun () ->
          (* tinygrad's test divides max x 0 + 1, which is negative at the
             greatest int, where tolk's int32 wraps. *)
          let x =
            Ops.add (Ops.maximum (loaded Int32 2) (Ops.int 0)) (Ops.int 1)
          in
          has Cmod (instructions_of ~ren:without_max [ Ops.O.(x // int 3) ]));
      dtype_cases "a floor division by a power of two is a shift"
        [ Int32; Uint32; Int64; Uint64 ] (fun dt ->
          let uops =
            instructions_of [ Ops.alu (loaded dt 2) Floordiv [ Ops.int 2 ] ]
          in
          has Shr uops;
          lacks Cdiv uops;
          lacks Cmod uops;
          lacks Floordiv uops);
      cases "an unsigned floor division is a truncating one"
        ~id:(Format.asprintf "%a" Op.pp) [ Op.Floordiv; Floormod ] (fun o ->
          let uops =
            instructions_of
              [
                Ops.alu (loaded ~slot:0 Uint32 2) o [ loaded ~slot:1 Uint32 2 ];
              ]
          in
          lacks Cmplt uops;
          equal int 1 (count Cdiv uops + count Cmod uops));
      test
        "with DISABLE_FAST_IDIV=0, a division and a remainder by 3 multiply \
         and shift" (fun () ->
          Setting.context fast_idiv (fun () ->
              let div =
                instructions_of
                  [ Ops.alu (loaded ~size:4 Uint32 3) Cdiv [ Ops.int 3 ] ]
              in
              has Shr div;
              lacks Cdiv div;
              let rem =
                instructions_of
                  [ Ops.alu (loaded ~size:4 Uint32 3) Cmod [ Ops.int 3 ] ]
              in
              has Shr rem;
              lacks Cmod rem));
      cases "a division by a divisor that is not positive is no shift"
        ~id:(fun (o, d) -> Format.asprintf "%a by %d" Op.pp o d)
        [ (Op.Cdiv, -3); (Cdiv, 0); (Cmod, -3); (Cmod, 0) ]
        (fun (o, d) ->
          Setting.context fast_idiv (fun () ->
              lacks Shr
                (instructions_of
                   [ Ops.alu (Ops.range (Int 20) [ 0 ]) o [ Ops.int d ] ])));
      test "a remainder the fast division declines stays a remainder" (fun () ->
          Setting.context fast_idiv (fun () ->
              has Cmod
                (instructions_of
                   [
                     Ops.alu (Ops.range (Int 30) [ 0 ]) Cmod
                       [ loaded ~size:4 Int32 0 ];
                   ]);
              has Cmod
                (instructions_of
                   [
                     Ops.alu (loaded ~size:4 Uint64 0) Cmod
                       [ Ops.int ~dtype:Uint64 3 ];
                   ])));
      test "a division of 0 or 1 by 3 is 0" (fun () ->
          let x =
            Ops.variable ~dtype:Int32 "x" (`Int Bigint.zero) (`Int Bigint.one)
          in
          let lowered =
            Setting.context fast_idiv (fun () ->
                full_rewrite [ Ops.alu x Cdiv [ Ops.int 3 ] ])
          in
          List.iter
            (fun v ->
              equal Dtypes.const ~msg:(string_of_int v) (`Int Bigint.zero)
                (Interpreter.eval
                   ~vars:[ ("x", `Int (Bigint.of_int v)) ]
                   (Ops.nth lowered 0)))
            [ 0; 1 ]);
      test "a division by 7 times 64 shifts out the 64 first, in 32 bits"
        (fun () ->
          Setting.context fast_idiv (fun () ->
              let r = Ops.range (Int (1 lsl 20)) [ 0 ] in
              let uops =
                instructions_of
                  [ Ops.div ~rounding:`Floor r (Ops.int (7 * 64)) ]
              in
              equal (list Uops.uop) [] (List.filter wide uops);
              lacks Cdiv uops));
      test "DISABLE_FAST_IDIV=1 keeps a division by 3" (fun () ->
          Setting.context
            [ B (Setting.disable_fast_idiv, true) ]
            (fun () ->
              let uops =
                instructions_of
                  [ Ops.alu (loaded ~size:4 Uint32 3) Cdiv [ Ops.int 3 ] ]
              in
              lacks Shr uops;
              has Cdiv uops));
    ]

(* Range shrinking *)

(* tinygrad's test: the ranges left after lowering [uops] without optimising,
   and the constant a range of [n] iterations ends at once lowered. *)
let ranges_left uops =
  Setting.context
    [ B (Setting.noopt, true) ]
    (fun () ->
      List.filter (is Range) (Ops.toposort ~calls:Enter (full_rewrite uops)))

let lowered_end n =
  Setting.context
    [ B (Setting.noopt, true) ]
    (fun () -> Ops.nth (full_rewrite [ Ops.int ~dtype:Int32 n ]) 0)

let ends_at n uops =
  match ranges_left uops with
  | [ r ] -> equal Uops.uop (lowered_end n) (Ops.nth r 0)
  | rs -> failf "one range, not %d" (List.length rs)

let r = Ops.range (Int 204) [ 0 ]
let table = Call.param ~shape:[ Int 1024 ] 0 Float32

let gated_load bound =
  Ops.load (Ops.index table [ Shape.valid r (Ops.lt r (Ops.int bound)) ]) []

let out = Call.param ~shape:[ Int 204 ] 0 Float32

let range_shrinking =
  let x = Ops.where (Ops.lt r (Ops.int 4)) (Ops.float 1.) Ops.invalid in
  group "range shrinking"
    [
      test "a range guarded by r < 4 everywhere ends at 4" (fun () ->
          ends_at 4 [ gated_load 4 ]);
      test "guards of 4 and 8 end it at 8" (fun () ->
          ends_at 8 [ gated_load 4; gated_load 8 ]);
      test "a guard past its end leaves it" (fun () ->
          ends_at 204 [ gated_load 300 ]);
      test "a read without a guard leaves it" (fun () ->
          let unguarded =
            Ops.load
              (Ops.index (Call.param ~shape:[ Int 204 ] 1 Float32) [ r ])
              []
          in
          ends_at 204 [ gated_load 4; unguarded ]);
      test "a reduction over it leaves it" (fun () ->
          ends_at 204
            [
              Ops.reduce (Ops.add (Ops.cast r Float32) (gated_load 4)) Add [ r ];
            ]);
      test "a guard of 1 removes it" (fun () ->
          equal (list Uops.uop) [] (ranges_left [ gated_load 1 ]));
      test "a store of a value valid where r < 4 ends it at 4" (fun () ->
          ends_at 4
            [
              Ops.store (Ops.index out [ r ])
                (Ops.where (Ops.lt r (Ops.int 4)) x Ops.invalid);
            ]);
      test "so does its flipped selection" (fun () ->
          ends_at 4
            [
              Ops.store (Ops.index out [ r ])
                (Ops.where (Ops.ge r (Ops.int 4)) Ops.invalid x);
            ]);
    ]

(* The specification of the lowered graph *)

(* A renderer whose last rewrite makes a node no program holds. *)
let breaking =
  Renderer.v
    ~extra_matcher:
      (Ops.Pattern_matcher.v (fun () ->
           [
             Ops.Pattern_matcher.rule (Ops.Upat.op Customi) (fun _ ->
                 Some (Ops.v Source ~arg:(String "x")));
           ]))
    (target "" "" "")

let marker_kernel =
  lazy
    (let marker =
       Ops.v Customi ~arg:(Code { code = "make_movement()"; dtype = Int32 })
     in
     Ops.sink ~kernel:(Ops.kernel_info ())
       [
         Ops.store
           (Ops.index (Call.param ~shape:[ Int 1 ] 0 Int32) [ Ops.int 0 ])
           marker;
       ])

let checks_what_it_lowers () =
  Setting.context
    [ B (Setting.spec, 1) ]
    (fun () ->
      raises_match (Exn.invalid_arg ~sub:"UOp verification failed") (fun () ->
          Codegen.full_rewrite_to_sink ~optimize:false
            (Lazy.force marker_kernel) breaking))

(* Tensor-core accumulators *)

(* A Metal tensor core's product added, from the constant accumulator [c], to
   two floats the kernel loads: [out[i] = (wmma a b c)[i] + acc[i]], where
   [acc], of slot 2, stands for the running sum of a tensor-core loop. *)
let lane slot dt i =
  Ops.index (Call.param ~shape:[ Int 2 ] slot dt) [ Ops.int ~dtype:Int32 i ]

let lanes slot dt =
  Shape.stack (List.init 2 (fun i -> Ops.load (lane slot dt i) []))

let accumulated c =
  let c = Ops.float ~dtype:Float32 c in
  let product =
    Ops.wmma (lanes 0 Float16) (lanes 1 Float16)
      ~acc:(Shape.stack [ c; c ])
      ~dims:(8, 8, 8) ~threads:32
  in
  let sum = Ops.add product (lanes 2 Float32) in
  Ops.sink ~kernel:(Ops.kernel_info ())
    (List.init 2 (fun i ->
         Ops.store (lane 3 Float32 i) (Ops.index sum [ Ops.int ~dtype:Int32 i ])))

let lowered_on_metal k =
  Setting.context
    [ B (Setting.spec, 0) ]
    (fun () -> Codegen.full_rewrite_to_sink ~optimize:false k metal)

(* A tensor core computes its accumulator plus a product that does not depend on
   it: each tensor core of [u] stands here for its accumulator plus the two
   floats of slot 4. *)
let with_products u =
  let products = lanes 4 Float32 in
  Ops.substitute ~calls:Skip ~pass:Fixed_point u
    (List.filter_map
       (fun w ->
         if is Wmma w then Some (w, Ops.add (Ops.nth w 2) products) else None)
       (Ops.toposort ~calls:Enter u))

(* The kernel of [c] and its lowering, each tensor core standing for its sum,
   lowered once for every case of a property. *)
let standing_in =
  let made = Hashtbl.create 4 in
  fun c ->
    (* -0. and 0. are two kernels. *)
    let key = Int64.bits_of_float c in
    match Hashtbl.find_opt made key with
    | Some pair -> pair
    | None ->
        let k = accumulated c in
        let pair = (with_products k, with_products (lowered_on_metal k)) in
        Hashtbl.replace made key pair;
        pair

let zero_sign_apart v0 v1 =
  match (v0, v1) with
  | `Float f0, `Float f1 when f0 = 0. && f1 = 0. -> true
  | v0, v1 -> Testable.equal Dtypes.value v0 v1

let written =
  Testable.make ~pp:(Testable.pp Kernel_opts.write)
    ~equal:(fun (s0, i0, v0) (s1, i1, v1) ->
      s0 = s1 && i0 = i1 && zero_sign_apart v0 v1)

(* The sum of [c], the accumulator and the products, whose rounding is fixed for
   a zero accumulator and left open otherwise, as tinygrad reassociates it. *)
let keeps_the_value (c, (acc, products)) =
  let buffers =
    [
      (0, Array.make 2 (`Float 1.));
      (1, Array.make 2 (`Float 1.));
      (2, Array.of_list acc);
      (3, Array.make 2 (`Float 0.));
      (4, Array.of_list products);
    ]
  in
  let k, lowered = standing_in c in
  equal (list written)
    (Interpreter.writes ~buffers k)
    (Interpreter.writes ~buffers lowered)

let keeps_a_zero_accumulator_value (c, (acc, products)) =
  cover "an accumulator at -0."
    (List.exists (Testable.equal Dtypes.value (`Float (-0.))) acc);
  cover "an accumulator that starts at -0." (1. /. c < 0.);
  keeps_the_value (c, (acc, products))

(* Two floats, signed zeros often among them. *)
let float32s =
  Gen.list ~size:(Gen.constant 2)
    (Gen.frequency
       [
         (3, Dtypes.value_of Float32);
         ( 1,
           Gen.of_list ~pp:(Testable.pp Dtypes.value)
             [ `Float (-0.); `Float 0. ] );
       ])

let zeros = Gen.of_list ~pp:Format.pp_print_float [ 0.; -0. ]
let float_constant u = is Const u && Dtype.is_float (Ops.dtype u)

let replaces_a_zero c =
  List.iter
    (fun w ->
      equal (list Uops.uop) []
        (List.filter float_constant (Ops.toposort ~calls:Enter (Ops.nth w 2))))
    (List.filter (is Wmma)
       (Ops.toposort ~calls:Enter (lowered_on_metal (accumulated c))))

let accumulators =
  group "tensor-core accumulators"
    [
      cases "the running sum replaces a zero accumulator" ~id:string_of_float
        [ 0.; -0. ] replaces_a_zero;
      test
        "the running sum is added to an accumulator that is not zero, which \
         keeps a sum computed exactly" (fun () ->
          keeps_the_value
            (1.5, ([ `Float 2.; `Float (-3.) ], [ `Float 4.; `Float 0.5 ])));
      prop "the lowering keeps a tensor core's value, apart from a zero's sign"
        (Gen.pair zeros (Gen.pair float32s float32s))
        keeps_a_zero_accumulator_value;
    ]

(* Reductions over a variable size

   A kernel reduces [x], of [hi] floats, over a range whose size is a variable
   [n] from 0 to [hi]: over [x.(k)] for each [k] below [n], or over [x.(k / 2)],
   whose index the range's bounds fold to 0 when [hi] is 1, so that the reduced
   value then ignores the range. [x.(0)] is an infinity. *)

type index = Each | Halved

let reduction_by_size rop index hi =
  let out = Call.param ~shape:[ Int 1 ] 0 Float32 in
  let x = Call.param ~shape:[ Int hi ] 1 Float32 in
  let n = Ops.variable "n" (`Int Bigint.zero) (`Int (Bigint.of_int hi)) in
  let r = Ops.range ~axis_type:Reduce (Sym n) [ 0 ] in
  let at = match index with Each -> r | Halved -> Ops.O.(r // int 2) in
  let reduced = Ops.reduce (Ops.index x [ at ]) rop [ r ] in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.store (Ops.index out [ Ops.int 0 ]) reduced ]

(* The reduction from its identity, as Tolk.Op.Reduce defines it. *)
let reduced_by_size rop index x n =
  let identity, f =
    match rop with
    | Op.Add -> (0., ( +. ))
    | _ -> (Float.neg_infinity, Float.max)
  in
  let at k = match index with Each -> k | Halved -> k / 2 in
  List.fold_left (fun acc k -> f acc x.(at k)) identity (List.init n Fun.id)

let reduces_by_size (rop, index, hi) =
  let x =
    Array.init hi (fun j ->
        if j = 0 then Float.infinity else 1. +. float_of_int j)
  in
  let prg =
    Codegen.to_program (reduction_by_size rop index hi) (Lazy.force host)
  in
  let run n =
    let inputs = [ (1, Array.map (fun f -> `Float f) x) ] in
    List.assoc 0 (Run.on_host ~vars:[ ("n", n) ] prg inputs)
  in
  List.iter
    (fun n ->
      equal
        ~msg:(Printf.sprintf "at n = %d" n)
        (array Dtypes.value)
        [| `Float (reduced_by_size rop index x n) |]
        (run n))
    (List.init (hi + 1) Fun.id)

(* The int32 maximum of [2 k] for [k] below a variable [n] of at most [hi]: a
   factor that ignores the range moves out of the maximum. *)
let max_of_a_product hi =
  let out = Call.param ~shape:[ Int 1 ] 0 Int32 in
  let n = Ops.variable "n" (`Int Bigint.zero) (`Int (Bigint.of_int hi)) in
  let r = Ops.range ~axis_type:Reduce (Sym n) [ 0 ] in
  let doubled = Ops.O.(Ops.cast r Int32 * Ops.int ~dtype:Int32 2) in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [
      Ops.store (Ops.index out [ Ops.int 0 ]) (Ops.reduce doubled Op.Max [ r ]);
    ]

let maximum_of_a_product hi =
  let prg = Codegen.to_program (max_of_a_product hi) (Lazy.force host) in
  let at n =
    if n = 0 then `Int (Bigint.of_int32 Int32.min_int)
    else `Int (Bigint.of_int (2 * (n - 1)))
  in
  List.iter
    (fun n ->
      equal
        ~msg:(Printf.sprintf "at n = %d" n)
        (array Dtypes.value)
        [| at n |]
        (List.assoc 0 (Run.on_host ~vars:[ ("n", n) ] prg [])))
    (List.init (hi + 1) Fun.id)

let variable_sizes =
  let name (rop, index, hi) =
    Format.asprintf "%a of x.(%s) below n <= %d" Op.pp rop
      (match index with Each -> "k" | Halved -> "k / 2")
      hi
  in
  group "reductions over a variable size"
    [
      cases "are their identity where the size is 0, and reduce elsewhere"
        ~id:name
        (List.concat_map
           (fun rop ->
             List.concat_map
               (fun index -> List.map (fun hi -> (rop, index, hi)) [ 1; 3 ])
               [ Each; Halved ])
           [ Op.Add; Op.Max ])
        reduces_by_size;
      cases "a maximum of a product is the identity where the size is 0"
        ~id:(Printf.sprintf "n <= %d") [ 1; 3 ] maximum_of_a_product;
    ]

(* Multiply-adds

   A sum adds each product of its source into its running sum as one
   multiply-add, rounded once, in the order it adds them unfused; every other
   product and sum rounds as written. The two products below are -(1 + 2^-11)
   and (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24, added in that order: the second,
   fused, keeps the 2^-24 that rounding the product alone would lose. *)

let dot n =
  let out = Call.param ~shape:[ Int 1 ] 0 Float32 in
  let a = Call.param ~shape:[ Int n ] 1 Float32 in
  let b = Call.param ~shape:[ Int n ] 2 Float32 in
  let r = Ops.range ~axis_type:Reduce (Int n) [ 0 ] in
  let sum =
    Ops.reduce (Ops.mul (Ops.index a [ r ]) (Ops.index b [ r ])) Op.Add [ r ]
  in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.store (Ops.index out [ Ops.int 0 ]) sum ]

(* [out[i] = sum_k a[i, k] * b[k]] for [i] below [rows] and [k] below [n], with
   [i] upcast by 4 and [k] unrolled by 4: the reduce sums a view of its
   products, its unrolled lanes first. *)
let matvec ~rows n =
  let out = Call.param ~shape:[ Int rows ] 0 Float32 in
  let a = Call.param ~shape:[ Int (rows * n) ] 1 Float32 in
  let b = Call.param ~shape:[ Int n ] 2 Float32 in
  let i = Ops.range (Int rows) [ 0 ] in
  let k = Ops.range ~axis_type:Reduce (Int n) [ 1 ] in
  let a_ik = Ops.index a [ Ops.O.((i * Ops.int n) + k) ] in
  let sum = Ops.reduce (Ops.mul a_ik (Ops.index b [ k ])) Op.Add [ k ] in
  let split axis target = Opt.Split { axis; amount = 4; target; top = false } in
  Ops.sink
    ~kernel:
      (Ops.kernel_info ~opts_to_apply:[ split 0 Upcast; split 2 Unroll ] ())
    [ Ops.end_ (Ops.store (Ops.index out [ i ]) sum) [ i ] ]

let muladd n =
  let out = Call.param ~shape:[ Int n ] 0 Float32 in
  let x k = Ops.index (Call.param ~shape:[ Int n ] k Float32) in
  let i = Ops.range (Int n) [ 0 ] in
  let value = Ops.add (Ops.mul (x 1 [ i ]) (x 2 [ i ])) (x 3 [ i ]) in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (Ops.index out [ i ]) value) [ i ] ]

let mentions sub s =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
  in
  go 0

let occurrences sub s =
  let n = String.length sub in
  let rec go i count =
    if i + n > String.length s then count
    else go (i + 1) (if String.sub s i n = sub then count + 1 else count)
  in
  go 0 0

let multiply_adds =
  group "multiply-adds"
    [
      cases "a sum of products adds each into its running sum rounded once, of"
        ~id:string_of_int [ 2; 64 ] (fun n ->
          let prg = Codegen.to_program (dot n) (Lazy.force host) in
          is_true ~msg:"__builtin_fmaf"
            (mentions "__builtin_fmaf(" (source prg));
          let x = 1. +. 0x1p-12 in
          let terms first second =
            Array.init n (fun i ->
                `Float (match i with 0 -> first | 1 -> second | _ -> 0.))
          in
          let out =
            Run.on_host prg
              [ (1, terms (-.(1. +. 0x1p-11)) x); (2, terms 1. x) ]
          in
          equal (array Dtypes.value) [| `Float 0x1p-24 |] (List.assoc 0 out));
      test
        "a sum of products unrolled beside upcast lanes adds each into its \
         running sum rounded once" (fun () ->
          let rows = 8 and n = 16 in
          let prg = Codegen.to_program (matvec ~rows n) (Lazy.force host) in
          (* each of the 4 running sums adds a product and 3 multiply-adds *)
          equal int ~msg:"multiply-adds in the source" 12
            (occurrences "__builtin_fmaf(" (source prg));
          let x = 1. +. 0x1p-12 in
          let row first second =
            List.init n (fun k ->
                `Float (match k with 0 -> first | 1 -> second | _ -> 0.))
          in
          let a =
            List.init rows (fun _ -> row (-.(1. +. 0x1p-11)) x)
            |> List.concat |> Array.of_list
          in
          let out = Run.on_host prg [ (1, a); (2, Array.of_list (row 1. x)) ] in
          equal (array Dtypes.value)
            (Array.make rows (`Float 0x1p-24))
            (List.assoc 0 out));
      test "a product and a sum outside a reduction are not fused" (fun () ->
          let prg = Codegen.to_program (muladd 4) (Lazy.force host) in
          is_false ~msg:"fma" (mentions "fma" (source prg)));
    ]

(* Reductions that share a range

   [out[0] = max_{i,k} (a[k] + i) + sum_k a[k]] for [i] below 3 and [k] below
   5: the two reductions close the range [k], the maximum within [i]'s loop and
   the sum alone, as a sum over [i] of a source that does not read [i] leaves
   it. No optimisation applies, so both stay loops. With [a[k] = k + 1], it is
   [(5 + 2) + 15]. *)

let max_and_sum () =
  let out = Call.param ~shape:[ Int 1 ] 0 Float32 in
  let a = Call.param ~shape:[ Int 5 ] 1 Float32 in
  let i = Ops.range ~axis_type:Reduce (Int 3) [ 0 ] in
  let k = Ops.range ~axis_type:Reduce (Int 5) [ 1 ] in
  let a_k = Ops.index a [ k ] in
  let max = Ops.reduce (Ops.add a_k (Ops.cast i Float32)) Op.Max [ i; k ] in
  let sum = Ops.reduce a_k Op.Add [ k ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~opts_to_apply:[] ())
    [ Ops.store (Ops.index out [ Ops.int 0 ]) (Ops.add max sum) ]

let shared_ranges =
  group "reductions that share a range"
    [
      test "a range two reductions close at two depths runs as two loops"
        (fun () ->
          let prg = Codegen.to_program (max_and_sum ()) (Lazy.force host) in
          let a = Array.init 5 (fun k -> `Float (float_of_int (k + 1))) in
          let out = Run.on_host prg [ (1, a) ] in
          equal (array Dtypes.value) [| `Float 22. |] (List.assoc 0 out));
    ]

(* Lanes of an unrolled reduce

   When every lane of an unrolled reduce reads an Invalid index, the lanes stay
   a stack, so each lane's gated load folds to its own zero and the reduce sums
   lanes of zeros. *)

let sums_zeros name x zero () =
  let row = row_named (name ^ "_clang") in
  let prg =
    under row (fun () -> Codegen.to_program (kernel row) (Lazy.force host))
  in
  equal (array Dtypes.value) [| zero |]
    (List.assoc 0 (Run.on_host prg [ (1, x) ]))

let lanes =
  group "lanes of an unrolled reduce"
    [
      test
        "a reduce whose lanes all read an Invalid index is rendered as \
         tinygrad renders it" (fun () ->
          writes_as_tinygrad (row_named "invalid_lanes_metal"));
      test "a reduce whose lanes all read an Invalid index sums to zero"
        (sums_zeros "invalid_lanes" [| `Float 1.; `Float 2. |] (`Float 0.));
      test "a sum of int8 lanes that all read an Invalid index is zero"
        (sums_zeros "invalid_lanes_int8"
           [| `Int Bigint.one; `Int Bigint.one |]
           (`Int Bigint.zero));
    ]

(* Vectors in programs *)

let on_a_vector u =
  Op.Set.mem (Ops.op u) Op.Set.elementwise
  && match Shape.shape_opt u with Some (_ :: _) -> true | _ -> false

let applies_no_elementwise_operation_to_a_vector row =
  equal (list Uops.uop) [] (List.filter on_a_vector (instructions row))

(* A renderer that computes on vectors takes an operation on one axis, each
   source a vector of its lanes or a scalar every lane reads. *)
let on_a_vector_of_lanes u =
  let lanes s =
    match Shape.shape_opt s with
    | Some [] -> true
    | Some l -> l = Shape.shape u
    | None -> false
  in
  on_a_vector u
  && (List.compare_length_with (Shape.shape u) 1 <> 0
     || not
          (List.for_all
             (fun s -> Ops.is_invalid (Ops.base s) || lanes s)
             (Ops.src u)))

let applies_operations_to_vectors_of_lanes row =
  equal (list Uops.uop) [] (List.filter on_a_vector_of_lanes (instructions row))

let computes_on_vectors row = (renderer_of_row row).vector_alu

let lanes_of u =
  let lanes = function Ops.Int n -> n | Ops.Sym _ -> 0 in
  List.fold_left (fun n d -> n * lanes d) 1 (Shape.shape u)

(* The elementwise operations on vectors among [uops]: each operation, its
   lanes, and how many vectors of those lanes compute it. *)
let vector_operations uops =
  let tally =
    List.filter_map
      (fun u -> if on_a_vector u then Some (Ops.op u, lanes_of u) else None)
      uops
  in
  List.map
    (fun k -> (fst k, snd k, List.length (List.filter (( = ) k) tally)))
    (List.sort_uniq compare tally)

let op_w = Testable.make ~pp:Op.pp ~equal:Op.equal

(* The lanes of the vector types a host source declares, each once. *)
let vector_types src =
  let key = "ext_vector_type(" in
  let rec go acc i =
    match String.index_from_opt src i 'e' with
    | None -> List.sort_uniq compare acc
    | Some j
      when j + String.length key <= String.length src
           && String.sub src j (String.length key) = key ->
        let k = j + String.length key in
        let close = String.index_from src k ')' in
        go (int_of_string (String.sub src k (close - k)) :: acc) close
    | Some j -> go acc (j + 1)
  in
  go [] 0

(* [prg] computes each of its elementwise operations on vectors once, as one
   vector of [n] lanes, and its source declares that vector. *)
let one_vector n prg =
  let ops = vector_operations (Ops.src (Ops.nth prg 1)) in
  greater ~msg:"operations on vectors" int ~than:0 (List.length ops);
  equal
    (list (triple op_w int int))
    (List.map (fun (o, _, _) -> (o, n, 1)) ops)
    ops;
  equal (list int) [ n ] (List.filter (( = ) n) (vector_types (source prg)))

(* A sum of the two high nibbles of each pair of bytes, its output axis of 3
   upcast whole and its reduce unrolled: a body of 6 lanes. *)
let six_lanes_kernel () =
  let open Ops.O in
  let out = Call.param ~shape:[ Int 3 ] 0 Uint8 in
  let x = Call.param ~shape:[ Int 6 ] 1 Uint8 in
  let l = Ops.range (Int 3) [ 0 ] in
  let r = Ops.range ~axis_type:Reduce (Int 2) [ 1 ] in
  let byte = Ops.index x [ (l * int 2) + r ] in
  let nibble =
    (byte lsr Ops.int ~dtype:Uint8 4) land Ops.int ~dtype:Uint8 15
  in
  let sum = Ops.reduce nibble Op.Add [ r ] in
  let whole axis target = Opt.Split { axis; amount = 0; target; top = false } in
  let opts = [ whole 0 Upcast; whole 1 Unroll ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~opts_to_apply:opts ())
    [ Ops.end_ (Ops.store (Ops.index out [ l ]) sum) [ l ] ]

(* A selection of [y - x] where [x] is below [y] and of [x] elsewhere, [x] of
   [rows * cols] float32 elements and [y] of [cols] that each row reads, upcast
   whole along both axes: a body of [rows * cols] lanes, as the host lorenz
   kernels upcast an axis of 3 whole by 16 or 8 lanes. *)
let rows_kernel rows cols =
  let n = rows * cols in
  let open Ops.O in
  let out = Call.param ~shape:[ Int n ] 0 Float32 in
  let x = Call.param ~shape:[ Int n ] 1 Float32 in
  let y = Call.param ~shape:[ Int cols ] 2 Float32 in
  let i = Ops.range (Int rows) [ 0 ] in
  let j = Ops.range (Int cols) [ 1 ] in
  let at = (i * int cols) + j in
  let a = Ops.index x [ at ] and b = Ops.index y [ j ] in
  let whole axis =
    Opt.Split { axis; amount = 0; target = Upcast; top = false }
  in
  Ops.sink
    ~kernel:(Ops.kernel_info ~opts_to_apply:[ whole 1; whole 0 ] ())
    [
      Ops.end_
        (Ops.store (Ops.index out [ at ]) (Ops.where (a < b) (b - a) a))
        [ i; j ];
    ]

let selects_rows rows cols () =
  let prg = Codegen.to_program (rows_kernel rows cols) (Lazy.force host) in
  one_vector (rows * cols) prg;
  let x = Array.init (rows * cols) (fun k -> Float.of_int ((k * 7 mod 11) - 5))
  and y = Array.init cols (fun k -> Float.of_int ((k * 5 mod 9) - 4)) in
  let floats = Array.map (fun f -> `Float f) in
  let expected =
    Array.mapi
      (fun k a -> if a < y.(k mod cols) then y.(k mod cols) -. a else a)
      x
  in
  equal (array Dtypes.value) (floats expected)
    (List.assoc 0 (Run.on_host prg [ (1, floats x); (2, floats y) ]))

(* A selection of [y - x] where [x] is below [y] and of [x] elsewhere, over [n]
   float32 elements upcast whole: a body of [n] lanes. *)
let lanes_kernel n =
  let open Ops.O in
  let param slot = Call.param ~shape:[ Int n ] slot Float32 in
  let out = param 0 and x = param 1 and y = param 2 in
  let i = Ops.range (Int n) [ 0 ] in
  let a = Ops.index x [ i ] and b = Ops.index y [ i ] in
  let whole = Opt.Split { axis = 0; amount = 0; target = Upcast; top = false } in
  Ops.sink
    ~kernel:(Ops.kernel_info ~opts_to_apply:[ whole ] ())
    [
      Ops.end_
        (Ops.store (Ops.index out [ i ]) (Ops.where (a < b) (b - a) a))
        [ i ];
    ]

(* Each operation of the selection computes one vector of [n] lanes, and no
   lane of the selection apart. *)
let selects_lanes n () =
  let prg = Codegen.to_program (lanes_kernel n) (Lazy.force host) in
  let uops = Ops.src (Ops.nth prg 1) in
  equal
    (list (triple op_w int int))
    (List.sort compare (List.map (fun o -> (o, n, 1)) Op.[ Cmplt; Sub; Where ]))
    (vector_operations uops);
  equal (list int) [ n ] (List.filter (( = ) n) (vector_types (source prg)));
  let apart u =
    Op.Set.mem (Ops.op u) Op.Set.elementwise
    && Shape.shape_opt u = Some []
    && List.mem (Ops.dtype u) Dtype.[ Float32; Bool ]
  in
  equal (list Uops.uop) [] (List.filter apart uops);
  let x = Array.init n (fun k -> Float.of_int ((k * 7 mod 11) - 5))
  and y = Array.init n (fun k -> Float.of_int ((k * 5 mod 9) - 4)) in
  let floats = Array.map (fun f -> `Float f) in
  let expected = Array.map2 (fun a b -> if a < b then b -. a else a) x y in
  equal (array Dtypes.value) (floats expected)
    (List.assoc 0 (Run.on_host prg [ (1, floats x); (2, floats y) ]))

(* A dot product of 2 rows of [n] float64 pairs, the rows' axis upcast whole
   and the reduce unrolled: a sum of each row's [n] lanes of products. *)
let dot_kernel n =
  let size = 2 * n in
  let open Ops.O in
  let out = Call.param ~shape:[ Int 2 ] 0 Float64 in
  let x = Call.param ~shape:[ Int size ] 1 Float64 in
  let y = Call.param ~shape:[ Int size ] 2 Float64 in
  let l = Ops.range (Int 2) [ 0 ] in
  let r = Ops.range ~axis_type:Reduce (Int n) [ 1 ] in
  let at p = Ops.index p [ (l * int n) + r ] in
  let sum = Ops.reduce (at x * at y) Op.Add [ r ] in
  let whole axis target = Opt.Split { axis; amount = 0; target; top = false } in
  Ops.sink
    ~kernel:(Ops.kernel_info ~opts_to_apply:[ whole 0 Upcast; whole 1 Unroll ] ())
    [ Ops.end_ (Ops.store (Ops.index out [ l ]) sum) [ l ] ]

(* Each lane's product past the first fuses into a multiply-add and the first
   is a product of scalars: no product of vectors is left, and each row adds
   its products in order, each rounded once with its sum. The inputs are drawn
   so that rounding each product apart gives another sum in every row. *)
let sums_products n () =
  let prg = Codegen.to_program (dot_kernel n) (Lazy.force host) in
  let uops = Ops.src (Ops.nth prg 1) in
  equal
    (list (triple op_w int int))
    []
    (List.filter
       (fun (o, _, _) -> Op.equal o Op.Mul)
       (vector_operations uops));
  let f = Float.of_int in
  let x = Array.init (2 * n) (fun k -> Float.sin (f (k + 1)) *. f (k + 1))
  and y = Array.init (2 * n) (fun k -> Float.cos (f ((2 * k) + 1))) in
  let row i =
    let s = ref (x.(i * n) *. y.(i * n)) in
    for k = 1 to n - 1 do
      s := Float.fma x.((i * n) + k) y.((i * n) + k) !s
    done;
    `Float !s
  in
  let floats = Array.map (fun f -> `Float f) in
  equal (array Dtypes.value) [| row 0; row 1 |]
    (List.assoc 0 (Run.on_host prg [ (1, floats x); (2, floats y) ]))

(* The kernel that stores into slot 0, of [dt], element [i] of slot 3 where
   element [i] of slot 1 is below that of slot 2, both of [cmp], and of slot 4
   elsewhere, over [n] elements upcast whole, by their largest power of two
   first, at most 16: a selection of [n] lanes of [dt] by a comparison of [n]
   lanes of [cmp]. *)
let compare_select cmp dt n =
  let upcast amount =
    Opt.Split { axis = 0; amount; target = Upcast; top = false }
  in
  let w = min 16 (n land -n) in
  let opts = if w = n || w = 1 then [ upcast 0 ] else [ upcast w; upcast 0 ] in
  let open Ops.O in
  let param slot dt = Call.param ~shape:[ Int n ] slot dt in
  let out = param 0 dt and a = param 1 cmp and b = param 2 cmp in
  let c = param 3 dt and d = param 4 dt in
  let i = Ops.range (Int n) [ 0 ] in
  let at p = Ops.index p [ i ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~opts_to_apply:opts ())
    [
      Ops.end_
        (Ops.store (at out) (Ops.where (at a < at b) (at c) (at d)))
        [ i ];
    ]

(* Each lane compares the bounds of [cmp], their neighbours and 0, and selects
   between the bounds of [dt] and lanes counted from them. *)
let selects_by_comparison (cmp, dt, n) =
  let big = function `Int i -> i | _ -> invalid_arg "an integer bound" in
  let lo = big (Dtype.min cmp) and hi = big (Dtype.max cmp) in
  let edges =
    [|
      lo;
      Bigint.add lo Bigint.one;
      Bigint.minus_one;
      Bigint.zero;
      Bigint.one;
      Bigint.sub hi Bigint.one;
      hi;
    |]
  in
  let edges =
    Array.of_list (List.filter (fun e -> Bigint.leq lo e) (Array.to_list edges))
  in
  let m = Array.length edges in
  let a = Array.init n (fun k -> edges.(k mod m))
  and b = Array.init n (fun k -> edges.(k * 3 / 2 mod m)) in
  let vlo = big (Dtype.min dt) and vhi = big (Dtype.max dt) in
  let c = Array.init n (fun k -> Bigint.add vlo (Bigint.of_int k))
  and d = Array.init n (fun k -> Bigint.sub vhi (Bigint.of_int k)) in
  let ints = Array.map (fun i -> `Int i) in
  let expected =
    Array.init n (fun k -> if Bigint.lt a.(k) b.(k) then c.(k) else d.(k))
  in
  let prg = Codegen.to_program (compare_select cmp dt n) (Lazy.force host) in
  equal (array Dtypes.value) (ints expected)
    (List.assoc 0
       (Run.on_host prg [ (1, ints a); (2, ints b); (3, ints c); (4, ints d) ]))

let comparisons cmps dts lanes =
  List.concat_map
    (fun cmp ->
      List.concat_map (fun dt -> List.map (fun n -> (cmp, dt, n)) lanes) dts)
    cmps

let widths = Dtype.[ Int8; Int16; Int32; Int64; Uint8; Uint16; Uint32; Uint64 ]
let signed = Dtype.[ Int8; Int16; Int32; Int64 ]
let unsigned = Dtype.[ Uint8; Uint16; Uint32; Uint64 ]

(* A fold of an int8 [2; 2] to [2; 1], kernel [1; 2], dilation [1; 2] and
   padding [(0, 0); (1, 1)], every window in the padding. In devectorize, the
   gated load of each upcast lane folds to its own 0, so the select around it is
   a select per lane. *)
let vector_select_kernel () =
  let open Ops.O in
  let out = Call.param ~shape:[ Int 2 ] 0 Int8 in
  let x = Call.param ~shape:[ Int 4 ] 1 Int8 in
  let l = Ops.range ~axis_type:Weak (Int 2) [ 2 ] in
  let r0 = Ops.range ~axis_type:Reduce (Int 2) [ 0 ] in
  let r1 = Ops.range ~axis_type:Reduce (Int 4) [ 1 ] in
  let j = (r1 * int 3) + int 1 in
  let gate =
    ((r0 * int 2) + l < int 3)
    land (r1 < int 3)
    land ((r0 < int 1) land (j % int 5 < int 1))
  in
  let read =
    Ops.index x [ Shape.valid ((j // int 5 * int 2) + l) (r1 < int 3) ]
  in
  let value = Ops.where gate read (Ops.int ~dtype:Int8 0) in
  let sum = Ops.reduce (Ops.cast value Uint32) Op.Add [ r0; r1 ] in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (Ops.index out [ l ]) (Ops.cast sum Int8)) [ l ] ]

(* An unfold of a float16 whose windows all read padding, as rune lowered it
   before it lowered such an unfold to zeros: a weak 0. stored into each
   element. Upcast whole, the store is one store of four lanes, and the weak
   lowering casts the float32 stack of the constant's lanes to half: a
   conversion of a vector, which a program refuses on a renderer that computes
   lane by lane, and the host converts. *)
let half_zeros_kernel () =
  let open Ops.O in
  let out = Call.param ~shape:[ Int 4 ] 0 Float16 in
  let r0 = Ops.range ~axis_type:Weak (Int 2) [ 0 ] in
  let r1 = Ops.range ~axis_type:Weak (Int 2) [ 1 ] in
  let store = Ops.store (Ops.index out [ (r0 * int 2) + r1 ]) (float 0.) in
  let whole =
    Opt.Split { axis = 0; amount = 0; target = Upcast; top = false }
  in
  Ops.sink
    ~kernel:(Ops.kernel_info ~opts_to_apply:[ whole ] ())
    [ Ops.end_ store [ r0; r1 ] ]

let refused_on_a_cast kernel renderer () =
  Setting.context
    [ B (Setting.spec, 1) ]
    (fun () ->
      raises_match (Exn.invalid_arg ~sub:"on Ops.CAST") (fun () ->
          Codegen.to_program kernel renderer))

let vectors =
  group "vectors in programs"
    [
      test "a select of lanes whose loads all fold is devectorized" (fun () ->
          Setting.context
            [ B (Setting.spec, 1) ]
            (fun () ->
              ignore (Codegen.to_program (vector_select_kernel ()) clang)));
      test "a weak constant stored into four lanes of half is refused"
        (refused_on_a_cast (half_zeros_kernel ()) metal);
      test
        "a weak constant stored into four lanes of half on the host converts \
         them" (fun () ->
          let prg =
            Codegen.to_program (half_zeros_kernel ()) (Lazy.force host)
          in
          equal (array Dtypes.value)
            (Array.make 4 (`Float 0.))
            (List.assoc 0 (Run.on_host prg [])));
      group "no program applies an elementwise operation to a vector"
        (per_row
           ~only:(fun row -> compiles row && not (computes_on_vectors row))
           applies_no_elementwise_operation_to_a_vector);
      group "a host program computes on vectors of its lanes"
        (per_row
           ~only:(fun row -> compiles row && computes_on_vectors row)
           applies_operations_to_vectors_of_lanes);
      test "a host program of 6 lanes computes one vector" (fun () ->
          let prg =
            Codegen.to_program (six_lanes_kernel ()) (Lazy.force host)
          in
          one_vector 6 prg;
          let bytes = Array.map (fun b -> `Int (Bigint.of_int b)) in
          equal (array Dtypes.value)
            (bytes [| 0x1 + 0x2; 0xf + 0x0; 0x7 + 0x8 |])
            (List.assoc 0
               (Run.on_host prg
                  [ (1, bytes [| 0x10; 0x2f; 0xf3; 0x04; 0x7a; 0x80 |]) ])));
      test "a host program of 48 lanes computes one vector" (selects_rows 3 16);
      test "a host program of 24 lanes computes one vector" (selects_rows 3 8);
      cases "a host program of n lanes computes one vector"
        ~id:(Printf.sprintf "%d lanes") [ 3; 5; 7; 9; 12; 15 ] (fun n ->
          selects_lanes n ());
      cases "a host sum of a product's unrolled lanes multiplies no vectors"
        ~id:(Printf.sprintf "%d lanes") [ 3; 4; 9 ] (fun n ->
          sums_products n ());
      cases
        "a host selection of 8, 16 or 32 lanes by a comparison of integers \
         takes each lane's arm"
        ~id:(fun (cmp, dt, n) ->
          Format.asprintf "%a by %a, %d lanes" Dtype.pp dt Dtype.pp cmp n)
        (comparisons widths widths [ 8; 16; 32 ])
        selects_by_comparison;
      cases
        "a host selection of 3, 6, 12, 24 or 48 lanes by a comparison of \
         integers takes each lane's arm"
        ~id:(fun (cmp, dt, n) ->
          Format.asprintf "%a by %a, %d lanes" Dtype.pp dt Dtype.pp cmp n)
        (comparisons signed unsigned [ 3; 6; 12; 24; 48 ])
        selects_by_comparison;
    ]

(* Signed zeros *)

(* [padded ~flip fill before after xs] stores into a new buffer the float32
   values [xs] padded with [before] and [after] elements of [fill], as a pad by
   a value other than +0. is lowered: [fill] is selected off a mask of [xs]'
   elements, the pad's own zeros only where the mask holds. With [flip], the
   selection is by the mask's negation, [fill] first. *)
let padded ?(flip = false) fill before after xs =
  let n = Array.length xs in
  let x = Ops.new_buffer ~slot:1 (Single "CPU") n Float32 in
  let padding = [ Some (Ops.Int before, Ops.Int after) ] in
  let mask = Shape.pad (Shape.const_like ~dtype:Bool x (`Bool true)) padding in
  let fill = Ops.float ~dtype:Float32 fill in
  let value =
    if flip then Ops.where (Ops.logical_not mask) fill (Shape.pad x padding)
    else Ops.where mask (Shape.pad x padding) fill
  in
  let out =
    Ops.new_buffer ~slot:0 (Single "CPU") (n + before + after) Float32
  in
  (Ops.sink [ Ops.after out [ Ops.store out value ] ], value)

let floats xs = Array.map (fun x -> `Float x) xs

(* [scheduled sink] is the one kernel of [sink]'s schedule, with the slot of the
   buffer each of its parameters stands for. *)
let scheduled sink =
  let slot a =
    match Ops.arg a with
    | Param { slot; _ } -> slot
    | _ -> invalid_arg "an argument that is not storage"
  in
  match Ops.src (fst (Schedule.create_linear_with_vars sink)) with
  | [ call ] -> (Ops.body call, List.map slot (Ops.src_without_body call))
  | calls -> Format.kasprintf invalid_arg "%d kernels" (List.length calls)

(* [output slots values] is the contents of buffer 0 among the contents [values]
   of a kernel's parameters, and [input slots xs] binds buffer 1's parameter to
   [xs]. *)
let parameter slots b = Option.get (List.find_index (Int.equal b) slots)
let input slots xs = [ (parameter slots 1, floats xs) ]

let padding_computed ?flip fill before after xs =
  let sink, value = padded ?flip fill before after xs in
  let kernel, slots = scheduled sink in
  let lowered =
    Codegen.full_rewrite_to_sink ~optimize:false kernel
      (Lazy.force host_uncompiled)
  in
  let out = parameter slots 0 in
  let written =
    List.filter_map
      (fun (s, _, v) -> if s = out then Some v else None)
      (Interpreter.writes ~buffers:(input slots xs) lowered)
  in
  let expected =
    match Tensors.eval ~buffers:[ (1, floats xs) ] value with
    | [ values ] ->
        Array.to_list
          (Array.map
             (function
               | #Dtype.value as v -> v | `Invalid -> invalid_arg "Invalid")
             values)
    | _ -> invalid_arg "a value on one device"
  in
  equal (list Dtypes.value) expected written

let signed_zero_pads =
  Gen.map
    (fun ((flip, fill), (before, (after, xs))) ->
      (flip, fill, before, after, Array.of_list xs))
    (Gen.pair (Gen.pair Gen.bool zeros)
       (Gen.pair (Gen.int_range 0 3)
          (Gen.pair (Gen.int_range 0 3)
             (Gen.list ~size:(Gen.int_range 1 4)
                (Gen.of_list ~pp:Format.pp_print_float
                   [ 0.; -0.; 1.; -2.5; Float.nan ])))))

let signed_zeros =
  group "signed zeros"
    [
      test "a pad with -0. fill renders and computes -0." (fun () ->
          let sink, _ = padded (-0.) 1 1 [| 1.; 2. |] in
          let kernel, slots = scheduled sink in
          let prg = Codegen.to_program kernel (Lazy.force host) in
          contains ~sub:"-0.0f" (source prg);
          let results = Run.on_host prg (input slots [| 1.; 2. |]) in
          equal (array Dtypes.value)
            (floats [| -0.; 1.; 2.; -0. |])
            (List.assoc (parameter slots 0) results));
      test "Shape.pad by -0. computes -0. on the padding" (fun () ->
          let x = Ops.new_buffer ~slot:1 (Single "CPU") 2 Float32 in
          let value =
            Shape.pad ~value:(`Float (-0.)) x [ Some (Ops.Int 1, Ops.Int 1) ]
          in
          let out = Ops.new_buffer ~slot:0 (Single "CPU") 4 Float32 in
          let kernel, slots =
            scheduled (Ops.sink [ Ops.after out [ Ops.store out value ] ])
          in
          let prg = Codegen.to_program kernel (Lazy.force host) in
          let results = Run.on_host prg (input slots [| 1.; 2. |]) in
          equal (array Dtypes.value)
            (floats [| -0.; 1.; 2.; -0. |])
            (List.assoc (parameter slots 0) results));
      prop "a pad and a selection of signed zeros keep the interpreter's bits"
        signed_zero_pads (fun (flip, fill, before, after, xs) ->
          padding_computed ~flip fill before after xs);
    ]

(* NaN constants

   A NaN constant is stored with its bits: its sign and its payload, a
   signalling NaN staying one. *)

let word w = `Int (Bigint.of_int64_unsigned w)

let unsigned dt =
  List.find (fun u -> Dtype.itemsize u = Dtype.itemsize dt) Dtype.uints

(* The bits [dt] stores into each element of a kernel that stores the NaN of
   each word of [words]. *)
let stored_nans dt words =
  let out = Call.param ~shape:[ Int (List.length words) ] 0 dt in
  let store i w =
    let nan = Dtype.bitcast (unsigned dt) dt (word w) in
    Ops.store
      (Ops.index out [ Ops.int i ])
      (Ops.const ~dtype:dt (nan :> Dtype.const))
  in
  let sink = Ops.sink ~kernel:(Ops.kernel_info ()) (List.mapi store words) in
  let prg = Codegen.to_program sink (Lazy.force host) in
  Array.map (Dtype.bitcast dt (unsigned dt)) (List.assoc 0 (Run.on_host prg []))

let keeps_nan_bits (dt, words) =
  equal (array Dtypes.value)
    (Array.of_list (List.map word words))
    (stored_nans dt words)

let nan_constants =
  cases "a NaN constant is stored with its sign and payload"
    ~id:(fun (dt, _) -> Dtype.name dt)
    [
      (Dtype.Float16, [ 0x7e00L; 0xfe05L; 0x7d01L; 0xfc01L ]);
      (Float32, [ 0x7fc00000L; 0xffc00005L; 0x7f800001L; 0xff800003L ]);
      ( Float64,
        [
          0x7ff8000000000000L;
          0x7ff0000020000000L;
          0xfff8000000000001L;
          0xfff0000000000003L;
        ] );
    ]
    keeps_nan_bits

(* Casts to bfloat16

   A cast to bfloat16 from a type more precise than a float32 rounds once, from
   the exact value, where a cast through a float32 rounds twice: 2^40 + 2^32 + 1
   is 2^40 + 2^33 once, and 2^40 through the float32 2^40 + 2^32, a tie that
   rounds to even. *)

(* The kernel that stores into slot 0, of bfloat16, the cast of each of the [n]
   elements of slot 1, of [from]. *)
let cast_to_bfloat16 from n =
  let out = Call.param ~shape:[ Int n ] 0 Bfloat16 in
  let x = Call.param ~shape:[ Int n ] 1 from in
  let i = Ops.range (Int n) [ 0 ] in
  let store =
    Ops.store (Ops.index out [ i ]) (Ops.cast (Ops.index x [ i ]) Bfloat16)
  in
  Ops.sink ~kernel:(Ops.kernel_info ()) [ Ops.end_ store [ i ] ]

let rounds_once (from, values) =
  let n = Array.length values in
  let prg = Codegen.to_program (cast_to_bfloat16 from n) (Lazy.force host) in
  equal (array Dtypes.value)
    (Array.map
       (fun v ->
         match Dtype.const Bfloat16 v with
         | #Dtype.value as r -> r
         | `Invalid -> failf "%a is no bfloat16" (Testable.pp Dtypes.value) v)
       values)
    (List.assoc 0 (Run.on_host prg [ (1, values) ]))

let bfloat16_casts =
  let integers l =
    Array.of_list (List.map (fun s -> `Int (Bigint.of_string s)) l)
  in
  group "casts to bfloat16"
    [
      cases "an integer or a double rounds once on the host"
        ~id:(fun (dt, _) -> Dtype.name dt)
        [
          ( Dtype.Int64,
            integers
              [
                "1099511627777";
                "1103806595073";
                "-1103806595073";
                "16842753";
                "4629700416936869889";
              ] );
          (Dtype.Int32, integers [ "16842753"; "1077936129"; "-16842753" ]);
          (Dtype.Uint32, integers [ "2155872257" ]);
          (Dtype.Uint64, integers [ "9259400833873739777" ]);
          ( Dtype.Float64,
            [| `Float (1. +. 0x1p-8 +. 0x1p-40); `Float 0x1.0101p40 |] );
        ]
        rounds_once;
    ]

(* Selections between bfloat16 vectors *)

(* The kernel that stores into slot 0, of bfloat16, element [i] of slot 2 where
   element [i] of slot 1 is above 0, and of slot 3 elsewhere, over [n] elements,
   upcast by [n]: one selection between vectors of [n] lanes. *)
let select_bfloat16 n =
  let open Ops.O in
  let out = Call.param ~shape:[ Int n ] 0 Bfloat16 in
  let c = Call.param ~shape:[ Int n ] 1 Float32 in
  let a = Call.param ~shape:[ Int n ] 2 Bfloat16 in
  let b = Call.param ~shape:[ Int n ] 3 Bfloat16 in
  let i = Ops.range (Int n) [ 0 ] in
  let pick =
    Ops.where
      (Ops.index c [ i ] > float 0.)
      (Ops.index a [ i ]) (Ops.index b [ i ])
  in
  let k =
    Ops.sink ~kernel:(Ops.kernel_info ())
      [ Ops.end_ (Ops.store (Ops.index out [ i ]) pick) [ i ] ]
  in
  let upcast =
    Opt.Split { axis = 0; amount = n; target = Upcast; top = false }
  in
  Ops.replace k
    ~arg:(Kernel { (kernel_info k) with opts_to_apply = Some [ upcast ] })

let selects_bfloat16_lanes () =
  let floats l = Array.of_list (List.map (fun f -> `Float f) l) in
  let c = floats [ 1.; -1.; 0.; 2.; -3.; 4.; -0.5; 0.5 ]
  and a = floats [ 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. ]
  and b = floats [ -1.; -2.; -3.; -4.; -5.; -6.; -7.; -8. ] in
  let prg = Codegen.to_program (select_bfloat16 8) (Lazy.force host) in
  equal (array Dtypes.value)
    (floats [ 1.; -2.; -3.; 4.; -5.; 6.; -7.; 8. ])
    (List.assoc 0 (Run.on_host prg [ (1, c); (2, a); (3, b) ]))

(* Gated bfloat16 loads *)

(* The kernel that stores into slot 0, of bfloat16, element [k] of slot 1 where
   [k < n] and 0 elsewhere, [k] being element [i] of slot 2, over [n] elements:
   a load gated by data, whose alternative is a bfloat16 zero. *)
let take_bfloat16 n =
  let open Ops.O in
  let out = Call.param ~shape:[ Int n ] 0 Bfloat16 in
  let x = Call.param ~shape:[ Int n ] 1 Bfloat16 in
  let ids = Call.param ~shape:[ Int n ] 2 Int32 in
  let i = Ops.range (Int n) [ 0 ] in
  let k = Ops.index ids [ i ] in
  let inside = k < int n in
  let read = Ops.index x [ Shape.valid k inside ] in
  let taken = Ops.where inside read (Ops.float ~dtype:Bfloat16 0.) in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (Ops.index out [ i ]) taken) [ i ] ]

let takes_bfloat16 () =
  let floats l = Array.of_list (List.map (fun f -> `Float f) l) in
  let ints l = Array.of_list (List.map (fun i -> `Int (Bigint.of_int i)) l) in
  let prg = Codegen.to_program (take_bfloat16 4) (Lazy.force host) in
  equal (array Dtypes.value)
    (floats [ 3.; 0.; 1.; 0. ])
    (List.assoc 0
       (Run.on_host prg
          [ (1, floats [ 1.; 2.; 3.; 4. ]); (2, ints [ 2; 7; 0; 9 ]) ]))

(* Lanes all Invalid *)

(* rune's fold of a float32 [4; 1] to [3; 1], kernel [2; 2], dilation [2; 2] and
   padding [(0, 0); (1, 1)], every window of the second axis in the padding.
   Upcast, every lane of its gated index is Invalid; the lanes stay a stack, so
   the reshape and the permute that arranged them keep their width. *)
let invalid_lanes_kernel () =
  let open Ops.O in
  let out = Call.param ~shape:[ Int 3 ] 0 Float32 in
  let x = Call.param ~shape:[ Int 4 ] 1 Float32 in
  let o = Ops.range ~axis_type:Weak (Int 3) [ 2 ] in
  let r0 = Ops.range ~axis_type:Reduce (Int 4) [ 0 ] in
  let r1 = Ops.range ~axis_type:Reduce (Int 4) [ 1 ] in
  let j = (r0 * int 3) + o and k = (r1 * int 3) + int 1 in
  let inside = (r1 < int 3) land (j < int 10) in
  let gate =
    (j < int 10)
    land (r1 < int 3)
    land ((j % int 5 < int 1) land (k % int 5 < int 1))
  in
  let at = (j // int 5 * int 2) + (k // int 5) in
  let read = Ops.index x [ Shape.valid at inside ] in
  let zero = Ops.float ~dtype:Float32 0. in
  let sum = Ops.reduce (Ops.where gate read zero) Op.Add [ r0; r1 ] + zero in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (Ops.index out [ o ]) sum) [ o ] ]

let invalid_lanes =
  group "lanes all Invalid"
    [
      test "a kernel whose upcast lanes all read an Invalid index sums to zero"
        (fun () ->
          let prg =
            Codegen.to_program (invalid_lanes_kernel ()) (Lazy.force host)
          in
          let x = Array.init 4 (fun j -> `Float (float_of_int (j + 1))) in
          equal (array Dtypes.value)
            (Array.make 3 (`Float 0.))
            (List.assoc 0 (Run.on_host prg [ (1, x) ])));
    ]

(* Squares that wrap *)

(* The kernel that stores into slot 0 [(x * c) * (x * c)] of element [i] of slot
   1, over 8 elements of [dtype], with [opts]. *)
let square_kernel ~opts dtype c =
  let open Ops.O in
  let out = Call.param ~shape:[ Int 8 ] 0 dtype in
  let x = Call.param ~shape:[ Int 8 ] 1 dtype in
  let i = Ops.range (Int 8) [ 0 ] in
  let c = Ops.const ~dtype (`Int c) in
  let p = Ops.index x [ i ] * c in
  Ops.sink ~kernel:(Ops.kernel_info ~opts_to_apply:opts ())
    [ Ops.end_ (Ops.store (Ops.index out [ i ]) (p * p)) [ i ] ]

(* [c * c] is a multiple of [2^bits] where [c] is [2^(bits / 2)] or the
   greatest power of two, so each square wraps to 0, a lane at a time or 8
   lanes at once. *)
let square_wraps () =
  let upcast =
    Opt.Split { axis = 0; amount = 0; target = Upcast; top = false }
  in
  let power k = Bigint.shift_left Bigint.one k in
  List.iter
    (fun dtype ->
      let bits = Dtype.bitsize dtype in
      let top =
        match Dtype.min dtype with
        | `Int z when Bigint.sign z < 0 -> z
        | _ -> power (bits - 1)
      in
      List.iter
        (fun (c, opts) ->
          let k = square_kernel ~opts dtype c in
          let prg = Codegen.to_program k (Lazy.force host) in
          let x = Array.init 8 (fun j -> `Int (Bigint.of_int j)) in
          let msg =
            Format.asprintf "%a times %s, %d opts" Dtype.pp dtype
              (Bigint.to_string c) (List.length opts)
          in
          equal ~msg (array Dtypes.value)
            (Array.make 8 (`Int Bigint.zero))
            (List.assoc 0 (Run.on_host prg [ (1, x) ])))
        (List.concat_map
           (fun c -> [ (c, []); (c, [ upcast ]) ])
           [ power (bits / 2); top ]))
    Dtype.[ Int8; Int16; Int32; Int64; Uint8; Uint16; Uint32; Uint64 ]

(* The kernel that stores into slot 0 [x * c0 + x * c1] of element [i] of slot
   1, over 8 elements of [dtype], [c0] and [c1] weak. *)
let like_terms_kernel dtype c0 c1 =
  let open Ops.O in
  let out = Call.param ~shape:[ Int 8 ] 0 dtype in
  let x = Call.param ~shape:[ Int 8 ] 1 dtype in
  let i = Ops.range (Int 8) [ 0 ] in
  let v = Ops.index x [ i ] in
  let s = (v * Ops.const c0) + (v * Ops.const c1) in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (Ops.index out [ i ]) s) [ i ] ]

(* Like terms whose weak factors sum past the width, the greatest value and 3
   or 2^(bits - 1) + 1 and 2^(bits - 1) + 3: the sum of the products, which
   wraps, is [x * (c0 + c1)] at the width. *)
let like_terms_wrap () =
  List.iter
    (fun dtype ->
      let bits = Dtype.bitsize dtype in
      let _, hi = Dtypes.int_bounds dtype in
      let half = Bigint.shift_left Bigint.one (bits - 1) in
      List.iter
        (fun (c0, c1) ->
          let k = like_terms_kernel dtype (`Int c0) (`Int c1) in
          let prg = Codegen.to_program k (Lazy.force host) in
          let x = Array.init 8 (fun j -> `Int (Bigint.of_int j)) in
          let expected =
            Array.init 8 (fun j ->
                Dtype.truncate dtype (`Int Bigint.(of_int j * (c0 + c1))))
          in
          let msg =
            Format.asprintf "%a x * %s + x * %s" Dtype.pp dtype
              (Bigint.to_string c0) (Bigint.to_string c1)
          in
          equal ~msg (array Dtypes.value) expected
            (List.assoc 0 (Run.on_host prg [ (1, x) ])))
        Bigint.[ (hi, of_int 3); (succ half, half + of_int 3) ])
    Dtype.[ Int8; Int16; Int32; Int64; Uint8; Uint16; Uint32; Uint64 ]

(* The kernel that stores into slot [k] the [k]th of [ops] of [x] and [c] at
   element [i] of [x], slot 4, over 8 elements of [dtype], [c] weak, with
   [opts]. *)
let divide_kernel ~opts dtype ops c =
  let x = Call.param ~shape:[ Int 8 ] 4 dtype in
  let i = Ops.range (Int 8) [ 0 ] in
  let store k op =
    let out = Call.param ~shape:[ Int 8 ] k dtype in
    let q = Ops.alu (Ops.index x [ i ]) op [ Ops.const (`Int c) ] in
    Ops.store (Ops.index out [ i ]) q
  in
  Ops.sink ~kernel:(Ops.kernel_info ~opts_to_apply:opts ())
    [ Ops.end_ (Ops.group (List.mapi store ops)) [ i ] ]

(* A weak divisor past the width, above or below it, divides as its value in
   the dividend's type, a lane at a time or 8 lanes at once: on int8, 128 is
   -128, and on uint8, -3 is 253. *)
let divisors_wrap () =
  let upcast =
    Opt.Split { axis = 0; amount = 0; target = Upcast; top = false }
  in
  let ops =
    [
      (Op.Floordiv, Bigint.fdiv);
      (Op.Floormod, fun n d -> Bigint.(n - (d * fdiv n d)));
      (Op.Cdiv, Bigint.div);
      (Op.Cmod, Bigint.rem);
    ]
  in
  List.iter
    (fun dtype ->
      let bits = Dtype.bitsize dtype in
      let lo, hi = Dtypes.int_bounds dtype in
      let half = Bigint.shift_left Bigint.one (bits - 1) in
      let wide = Bigint.shift_left Bigint.one bits in
      let xs =
        if Dtype.is_unsigned dtype then
          Bigint.[ zero; one; of_int 2; of_int 7; of_int 100; half; pred hi; hi ]
        else Bigint.[ lo; succ lo; minus_one; zero; one; of_int 7; pred hi; hi ]
      in
      let x = Array.of_list (List.map (fun n -> `Int n) xs) in
      List.iter
        (fun c ->
          let d =
            match Dtype.truncate dtype (`Int c) with
            | `Int d -> d
            | _ -> failwith "an integer type truncates to an integer"
          in
          List.iter
            (fun opts ->
              let k = divide_kernel ~opts dtype (List.map fst ops) c in
              let prg = Codegen.to_program k (Lazy.force host) in
              let outs = Run.on_host prg [ (4, x) ] in
              List.iteri
                (fun slot (op, f) ->
                  let msg =
                    Format.asprintf "%a x %a %s, %d opts" Dtype.pp dtype Op.pp
                      op (Bigint.to_string c) (List.length opts)
                  in
                  equal ~msg (array Dtypes.value)
                    (Array.of_list
                       (List.map
                          (fun n -> Dtype.truncate dtype (`Int (f n d)))
                          xs))
                    (List.assoc slot outs))
                ops)
            [ []; [ upcast ] ])
        Bigint.
          [
            half; half + of_int 3; wide - of_int 3; wide + of_int 7;
            neg (succ half);
          ])
    Dtype.[ Int8; Int16; Int32; Int64; Uint8; Uint16; Uint32; Uint64 ]

let square_wrapping =
  group "constants past the width"
    [
      test "a square of a product whose constant squares to 0 is 0"
        square_wraps;
      test "like terms whose factors sum past the width wrap" like_terms_wrap;
      test "a weak divisor past the width divides as its value there"
        divisors_wrap;
    ]

(* An accumulator no lane adds to *)

(* [x] summed into each element [l] of three, where [l² mod 3 = 2]. No [l]
   satisfies it, but only a lane of [l] upcast whole, of one [l], shows it:
   each lane's addition to the accumulator folds to adding 0. *)
let unselected_sum_kernel () =
  let open Ops.O in
  let upcast =
    Opt.Split { axis = 0; amount = 0; target = Upcast; top = false }
  in
  let out = Call.param ~shape:[ Int 3 ] 0 Int32 in
  let x = Call.param ~shape:[ Int 64 ] 1 Int32 in
  let l = Ops.range ~axis_type:Weak (Int 3) [ 1 ] in
  let r = Ops.range ~axis_type:Reduce (Int 64) [ 0 ] in
  let gate = (l * l) % int 3 > int 1 in
  let value = Ops.where gate (Ops.index x [ r ]) (Ops.int ~dtype:Int32 0) in
  let sum = Ops.reduce value Op.Add [ r ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~opts_to_apply:[ upcast ] ())
    [ Ops.end_ (Ops.store (Ops.index out [ l ]) sum) [ l ] ]

(* The register elements [prg]'s instructions load before any stores them, by
   constant index; an element at a computed index is left out. *)
let unwritten_reads prg =
  let element u =
    let at = Ops.nth u 0 in
    let b = Ops.buf_uop at and i = Ops.nth at 1 in
    if Ops.addrspace b = Some Dtype.Reg && Ops.vmin i = Ops.vmax i then
      Some (b, Ops.vmin i)
    else None
  in
  let accessed u =
    if (is Op.Load u || is Op.Store u) && is Op.Index (Ops.nth u 0) then
      element u
    else None
  in
  let written = ref [] in
  let unwritten (b, i) =
    not (List.exists (fun (b', i') -> b' == b && i' = i) !written)
  in
  List.filter_map
    (fun u ->
      match accessed u with
      | Some e when is Op.Store u ->
          written := e :: !written;
          None
      | Some (b, i) when unwritten (b, i) -> Some i
      | _ -> None)
    (Ops.src (Ops.nth prg 1))

(* An unwritten register can hold 0, so only the first test fails whenever the
   initial store is lost; the second states the value. *)
let unselected_sum =
  group "an accumulator no lane adds to"
    [
      test "is stored before it is read" (fun () ->
          let prg =
            Codegen.to_program (unselected_sum_kernel ()) (Lazy.force host)
          in
          equal (list Dtypes.value) [] (unwritten_reads prg));
      test "sums to zero" (fun () ->
          let prg =
            Codegen.to_program (unselected_sum_kernel ()) (Lazy.force host)
          in
          let x = Array.init 64 (fun j -> `Int (Bigint.of_int (j + 1))) in
          equal (array Dtypes.value)
            (Array.make 3 (`Int Bigint.zero))
            (List.assoc 0 (Run.on_host prg [ (1, x) ])));
    ]

(* A maximum of weak integers

   A pad's fill is a bare constant: where the value it pads folds away, a
   maximum of the padding is one of weak integers, as in rune's Fold of a padded
   window. Its identity is the least value of its accumulator's width. *)

(* The greatest over [r1 < 5] of, where [r1 < 4], the greatest of a weak [0]
   over [r0 < 2], and [0] elsewhere: [0]. *)
let weak_max_kernel () =
  let open Ops.O in
  let out = Call.param ~shape:[ Int 1 ] 0 Int32 in
  let r1 = Ops.range ~axis_type:Reduce (Int 5) [ 1 ] in
  let r0 = Ops.range ~axis_type:Reduce (Int 3) [ 0 ] in
  let inner = Ops.where (r0 < int 2) (int 0) (Ops.const `Invalid) in
  let inner = Ops.reduce inner Op.Max [ r0 ] in
  let outer = Ops.reduce (Ops.where (r1 < int 4) inner (int 0)) Op.Max [ r1 ] in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.store (Ops.index out [ int 0 ]) (Ops.cast outer Int32) ]

let weak_max =
  group "a maximum of weak integers"
    [
      test "is the greatest value it reduces" (fun () ->
          let prg = Codegen.to_program (weak_max_kernel ()) (Lazy.force host) in
          equal (array Dtypes.value)
            [| `Int Bigint.zero |]
            (List.assoc 0 (Run.on_host prg [])));
    ]

(* Errors *)

(* A kernel holds no conditional: only a program's instructions do. *)
let spec_breaking =
  let at = Ops.index (Call.param ~shape:[ Int 1 ] 0 Int32) [ Ops.int 0 ] in
  let gate = Ops.lt (Ops.special (Int 4) "lidx0") (Ops.int 1) in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.store at (Ops.int ~dtype:Int32 1); Ops.v If ~src:[ gate; at ] ]

let errors =
  group "errors"
    [
      test
        "to_program raises Invalid_argument on a node that is no sink or \
         program" (fun () ->
          rejects (fun () -> Codegen.to_program (Ops.int 1) clang));
      test
        "to_program raises Invalid_argument on a sink without kernel \
         information" (fun () ->
          let k = Lazy.force add_kernel in
          rejects (fun () -> Codegen.to_program (Ops.sink (Ops.src k)) clang));
      test
        "full_rewrite_to_sink raises Invalid_argument when optimising a sink \
         without kernel information" (fun () ->
          let k = Lazy.force add_kernel in
          rejects (fun () ->
              Codegen.full_rewrite_to_sink (Ops.sink (Ops.src k)) clang));
      test
        "a lowered graph that breaks the specification raises Invalid_argument"
        checks_what_it_lowers;
      test
        "with SPEC=0, a lowered graph that breaks the specification is returned"
        (fun () ->
          let lowered =
            Setting.context
              [ B (Setting.spec, 0) ]
              (fun () ->
                Codegen.full_rewrite_to_sink ~optimize:false
                  (Lazy.force marker_kernel) breaking)
          in
          mem op Source (List.map Ops.op (Ops.toposort ~calls:Enter lowered)));
      test "with SPEC=0, a kernel that breaks the specification is lowered"
        (fun () ->
          let lowered =
            Setting.context
              [ B (Setting.spec, 0) ]
              (fun () ->
                Codegen.full_rewrite_to_sink ~optimize:false spec_breaking clang)
          in
          mem op If (List.map Ops.op (Ops.toposort ~calls:Enter lowered)));
      test "a kernel that breaks the specification raises Invalid_argument"
        (fun () ->
          Setting.context
            [ B (Setting.spec, 1) ]
            (fun () ->
              raises_match (Exn.invalid_arg ~sub:"UOp verification failed")
                (fun () -> Codegen.full_rewrite_to_sink spec_breaking clang)));
      test "a compiler that rejects the source raises Compile_error" (fun () ->
          let compiler =
            Renderer.Compiler.v (fun _ ->
                raise (Renderer.Compiler.Compile_error "no"))
          in
          raises (Renderer.Compiler.Compile_error "no") (fun () ->
              Codegen.to_program (fresh_kernel ())
                (renderer_for ~compiler clang_target)));
    ]

(* Diagnostics *)

let at_debug level f = Setting.context [ B (Setting.debug, level) ] f

(* tinygrad prints a tuple of optimisations: [(o,)], or [(o0, o1)]. *)
let prints_the_optimisations opts =
  let p =
    at_debug 3 (fun () ->
        Codegen.to_program (fresh_kernel ~opts_to_apply:opts ()) clang)
  in
  let tuple =
    match opts with
    | [ o ] -> Format.asprintf "(%a,)" Opt.pp o
    | opts ->
        "("
        ^ String.concat ", " (List.map (Format.asprintf "%a" Opt.pp) opts)
        ^ ")"
  in
  let name = Ops.function_name (kernel_info (Ops.nth p 0)) in
  contains ~sub:(Printf.sprintf "%-25s opts: %s" name tuple) (output ())

(* A child lowers the breaking graph, which must fail its check. DBGTV is read
   when the program starts, so the child runs with it set. *)
let lowers_a_breaking_graph () =
  Setting.context
    [ B (Setting.spec, 1) ]
    (fun () ->
      match
        Codegen.full_rewrite_to_sink ~optimize:false (Lazy.force marker_kernel)
          breaking
      with
      | exception Invalid_argument _ -> ()
      | _ -> failwith "the breaking graph passed its check")

let prints_a_breaking_graph () =
  match
    Disk_cache.child
      ~env:[ ("DBGTV", "1") ]
      ~cachedb:(Disk_cache.fresh ()) "breaking graph"
  with
  | Ok out -> contains ~sub:"Ops.SOURCE" out
  | Error err -> failf "the child failed: %s" err

let prints_the_source () =
  let p = at_debug 4 (fun () -> Codegen.to_program (fresh_kernel ()) clang) in
  contains ~sub:(source p) (output ())

let disassembles () =
  let compiler =
    Renderer.Compiler.v
      ~disassemble:(fun _ -> print_string "<disassembled>")
      Fun.id
  in
  ignore
    (at_debug 7 (fun () ->
         Codegen.to_program (fresh_kernel ())
           (renderer_for ~compiler clang_target)));
  contains ~sub:"<disassembled>" (output ())

let diagnostics =
  group "diagnostics"
    [
      cases "DEBUG=3 prints the optimisations applied"
        ~id:(fun opts -> string_of_int (List.length opts) ^ " optimisations")
        [
          [ upcast4 ];
          [
            upcast4;
            Split { axis = 0; amount = 2; target = Upcast; top = false };
          ];
        ]
        prints_the_optimisations;
      test
        "with DBGTV set, a lowered graph that breaks the specification is \
         printed"
        prints_a_breaking_graph;
      test "DEBUG=4 prints the source" prints_the_source;
      test "DEBUG=2 prints no optimisations" (fun () ->
          ignore
            (at_debug 2 (fun () ->
                 Codegen.to_program
                   (fresh_kernel ~opts_to_apply:[ upcast4 ] ())
                   clang));
          not_contains ~sub:"opts:" (output ()));
      test "DEBUG=3 prints nothing of a kernel without optimisations" (fun () ->
          ignore
            (at_debug 3 (fun () ->
                 Codegen.to_program (fresh_kernel ~opts_to_apply:[] ()) clang));
          not_contains ~sub:"opts:" (output ()));
      test "DEBUG=7 disassembles the binary" disassembles;
      test "DEBUG=2 prints no source" (fun () ->
          let p =
            at_debug 2 (fun () -> Codegen.to_program (fresh_kernel ()) clang)
          in
          not_contains ~sub:(source p) (output ()));
    ]

(* Instruction lists *)

let rule pat f = Ops.Pattern_matcher.rule pat f
let matcher rules = Ops.Pattern_matcher.fold (fun () -> rules)
let nothing = matcher []
let lidx = Ops.special (Int 4) "lidx0"
let gidx = Ops.special (Int 8) "gidx0"

(* A special node becomes another; users are rebuilt on it. *)
let to_gidx =
  matcher [ rule (Ops.Upat.op Special) (fun _ -> Some (gidx, [ gidx ])) ]

let line_rewrites =
  let sum = Ops.add lidx lidx in
  group "line_rewrite"
    [
      group "a matcher that matches nothing leaves the list as it is"
        (per_row ~only:on_clang (fun row ->
             let l = Ops.src (Ops.nth (recorded_program row) 1) in
             equal (list Uops.uop) l (Codegen.line_rewrite l nothing ())));
      test "a node's users see the node its rule gives" (fun () ->
          equal (list Uops.uop)
            [ gidx; Ops.add gidx gidx ]
            (Codegen.line_rewrite [ lidx; sum ] to_gidx ()));
      test "a rule's instructions replace its node, in order" (fun () ->
          let twice =
            matcher
              [
                rule (Ops.Upat.op Add ~name:"x") (fun m ->
                    Some (m "x", [ Ops.neg (m "x"); m "x" ]));
              ]
          in
          equal (list Uops.uop)
            [ lidx; Ops.neg sum; sum ]
            (Codegen.line_rewrite [ lidx; sum ] twice ()));
      test "a rule matches the node rebuilt on its sources' results" (fun () ->
          let rebuilt =
            Ops.Pattern_matcher.append to_gidx
              (matcher
                 [
                   rule
                     (Ops.Upat.op Add
                        ~src:
                          [
                            Ops.Upat.op Special ~arg:(String "gidx0");
                            Ops.Upat.wild;
                          ]
                        ~name:"x")
                     (fun m -> Some (m "x", [ Ops.neg (m "x") ]));
                 ])
          in
          equal (list Uops.uop)
            [ gidx; Ops.neg (Ops.add gidx gidx) ]
            (Codegen.line_rewrite [ lidx; sum ] rebuilt ()));
    ]

(* A store of 1 into the int buffer of slot 0, at the thread's index. *)
let out = Call.param ~shape:[ Int 4 ] 0 Int32
let at = Ops.index out [ lidx ]
let one = Ops.int ~dtype:Int32 1
let gate = Ops.ne lidx (Ops.int 0)
let store dst gate = Ops.v Store ~src:[ dst; one; gate ]
let cleaned uops = Codegen.line_rewrite uops Codegen.pm_linearize_cleanups ()

let made_conditional dst =
  let gated = store dst gate in
  let ungated = Ops.v Store ~src:[ dst; one ] in
  let if_ = Ops.v If ~src:[ gate; dst ] in
  equal (list Uops.uop)
    [ gate; dst; if_; ungated; Ops.v Endif ~src:[ if_ ] ]
    (cleaned [ gate; dst; gated ])

let pointer_cast dt = Ops.v Cast ~src:[ at ] ~arg:(Dtype dt)

let cleanups =
  group "pm_linearize_cleanups"
    [
      test "a gated store becomes a store inside an if on its gate" (fun () ->
          made_conditional at);
      test "so does a gated store through a cast of its index" (fun () ->
          made_conditional (pointer_cast Float32));
      cases "leaves alone a store it does not match" ~id:fst
        [
          ( "through a bitcast of its index",
            store (Ops.v Bitcast ~src:[ at ] ~arg:(Dtype Float32)) gate );
          ( "through a cast of a cast of its index",
            store
              (Ops.v Cast ~src:[ pointer_cast Float32 ] ~arg:(Dtype Int32))
              gate );
          ("gated by a value that is no boolean", store at lidx);
          ("without a gate", Ops.v Store ~src:[ at; one ]);
        ]
        (fun (_, st) -> equal (list Uops.uop) [ st ] (cleaned [ st ]));
      cases "raises Invalid_argument on an if already in the list" ~id:fst
        [
          ("if", Ops.v If ~src:[ gate; at ]);
          ("endif", Ops.v Endif ~src:[ Ops.v If ~src:[ gate; at ] ]);
        ]
        (fun (_, u) -> rejects (fun () -> cleaned [ u ]));
    ]

(* Host programs in blocks

   A host program splits the largest loop that every store to memory reads in
   its address into blocks, which the host's cores run at once. *)

let split_of prg =
  Option.map
    (fun (s : Ops.split) -> s.iterations)
    (kernel_info (Ops.nth prg 0)).split

let sint = Testable.make ~pp:Shape.Sint.pp ~equal:Shape.Sint.equal

let in_blocks uops =
  Codegen.to_program (Ops.sink ~kernel:(Ops.kernel_info ()) uops) clang

let floats n slot = Call.param ~shape:[ Int n ] slot Float32

let splits_the_output_loop () =
  let i = Ops.range (Int 1024) [ 0 ] in
  let a = Ops.index (floats 1024 1) [ i ] in
  let prg =
    in_blocks [ Ops.end_ (Ops.store (Ops.index (floats 1024 0) [ i ]) a) [ i ] ]
  in
  is_true ~msg:"split" (Option.is_some (split_of prg));
  let names =
    match Ops.arg prg with
    | Program p -> List.map Ops.expr p.vars
    | _ -> invalid_arg "a program holds its program information"
  in
  equal (list string) [ "block_lo"; "block_hi" ] names

(* A kernel whose buffers take the slots 0 and 2, and which reads a variable:
   the block's bounds take the first two slots past the buffers', and the
   variable the first past theirs, so that a launch finds each by its slot. *)
let numbers_past_a_skipped_slot () =
  let i = Ops.range (Int 1022) [ 0 ] in
  let n =
    Ops.variable ~dtype:Int32 "n" (Dtype.Value.of_int 0)
      (Dtype.Value.of_int 100)
  in
  let value = Ops.add (Ops.index (floats 1022 2) [ i ]) (Ops.cast n Float32) in
  let prg =
    in_blocks
      [ Ops.end_ (Ops.store (Ops.index (floats 1022 0) [ i ]) value) [ i ] ]
  in
  let slots =
    match Ops.arg prg with
    | Program p ->
        List.map
          (fun v ->
            match Ops.arg v with
            | Ops.Param q -> (Ops.expr v, q.slot)
            | _ -> invalid_arg "a variable of a program is a parameter")
          p.vars
    | _ -> invalid_arg "a program holds its program information"
  in
  equal
    (list (pair string int))
    [ ("block_lo", 3); ("block_hi", 4); ("n", 5) ]
    slots

(* Two outputs, each written in a loop of its own: no loop is read by every
   store, and splitting either would run the other's whole loop in every
   block. *)
let leaves_separate_loops () =
  let i = Ops.range (Int 1024) [ 0 ] and j = Ops.range (Int 512) [ 1 ] in
  let copy slot r n =
    Ops.end_
      (Ops.store
         (Ops.index (floats n slot) [ r ])
         (Ops.index (floats n 2) [ r ]))
      [ r ]
  in
  equal (option sint) None
    (split_of (in_blocks [ copy 0 i 1024; copy 1 j 512 ]))

let leaves_a_store_of_no_loop () =
  let i = Ops.range (Int 1024) [ 0 ] in
  let a = Ops.index (floats 1024 2) [ i ] in
  equal (option sint) None
    (split_of
       (in_blocks
          [
            Ops.end_ (Ops.store (Ops.index (floats 1024 0) [ i ]) a) [ i ];
            Ops.store (Ops.index (floats 1 1) [ Ops.int 0 ]) (Ops.float 1.);
          ]))

(* A serial loop's iterations may read what earlier ones wrote. *)
let leaves_a_serial_loop () =
  let i = Ops.range ~axis_type:Loop (Int 1024) [ 0 ] in
  let a = Ops.index (floats 1024 1) [ i ] in
  equal (option sint) None
    (split_of
       (in_blocks
          [ Ops.end_ (Ops.store (Ops.index (floats 1024 0) [ i ]) a) [ i ] ]))

let leaves_a_reduction () =
  let j = Ops.range ~axis_type:Reduce (Int 4096) [ 0 ] in
  let sum = Ops.reduce (Ops.index (floats 4096 1) [ j ]) Op.Add [ j ] in
  equal (option sint) None
    (split_of
       (in_blocks [ Ops.store (Ops.index (floats 1 0) [ Ops.int 0 ]) sum ]))

(* A store of a value valid where r < 4 shrinks its loop to 4 iterations and
   drops the guard; the split keeps both. *)
let keeps_a_shrunk_loop () =
  let r = Ops.range (Int 204) [ 0 ] in
  let x = Ops.where (Ops.lt r (Ops.int 4)) (Ops.float 1.) Ops.invalid in
  let prg =
    Setting.context
      [ B (Setting.noopt, true) ]
      (fun () ->
        in_blocks
          [ Ops.end_ (Ops.store (Ops.index (floats 204 0) [ r ]) x) [ r ] ])
  in
  equal (option sint) (Some (Int 4)) (split_of prg);
  equal int ~msg:"selections" 0 (count Where (Ops.src (Ops.nth prg 1)))

(* An index of a buffer of 2^25 elements, past the 2^31 / 127 a block count of
   up to 127 would multiply, stays 32 bits wide. *)
let keeps_narrow_indices () =
  let n = 1 lsl 25 in
  let i = Ops.range (Int n) [ 0 ] in
  let a = Ops.index (floats n 1) [ i ] in
  let prg =
    in_blocks [ Ops.end_ (Ops.store (Ops.index (floats n 0) [ i ]) a) [ i ] ]
  in
  equal (list Uops.uop) []
    (List.filter wide (Ops.toposort ~calls:Enter (Ops.nth prg 0)))

(* 2^16 sums of 2^16 products count at least a product and a sum each, 2^33
   operations: past the 32-bit index type a block's count is kept in. *)
let counts_a_large_loop () =
  let n = 1 lsl 16 in
  let i = Ops.range (Int n) [ 0 ] in
  let j = Ops.range ~axis_type:Reduce (Int n) [ 1 ] in
  let x =
    Ops.mul (Ops.index (floats n 1) [ i ]) (Ops.index (floats n 2) [ j ])
  in
  let sum = Ops.reduce x Op.Add [ j ] in
  let prg =
    in_blocks [ Ops.end_ (Ops.store (Ops.index (floats n 0) [ i ]) sum) [ i ] ]
  in
  match (kernel_info (Ops.nth prg 0)).estimates with
  | Some { ops = Int ops; _ } -> at_least int ~than:(1 lsl 33) ops
  | Some { ops = Sym u; _ } -> failf "the operations count %a" Ops.pp u
  | None -> failf "a program has estimates"

let blocks =
  group "host programs in blocks"
    [
      test "a host program splits the loop each store's address reads"
        splits_the_output_loop;
      test "stores of separate loops split none" leaves_separate_loops;
      test "a store of no loop splits none" leaves_a_store_of_no_loop;
      test "a reduction's loop is never split" leaves_a_reduction;
      test "a serial loop is never split" leaves_a_serial_loop;
      test "a loop shrunk by its guard splits at its shrunk end, unguarded"
        keeps_a_shrunk_loop;
      test "a split loop of 2^25 iterations keeps 32-bit indices"
        keeps_narrow_indices;
      test "a split loop's estimates count its whole loop past 2^31"
        counts_a_large_loop;
      test
        "a split's bounds and a variable take slots of their own past buffers \
         that skip one"
        numbers_past_a_skipped_slot;
    ]

let () =
  Disk_cache.play (("breaking graph", lowers_a_breaking_graph) :: parts);
  exit
    (run "Tolk.Codegen"
       [
         stages;
         values;
         claims;
         lowering_claims;
         whole_graphs;
         accumulators;
         variable_sizes;
         multiply_adds;
         shared_ranges;
         lanes;
         bfloat16_casts;
         test "a selection between bfloat16 vectors takes each lane's arm"
           selects_bfloat16_lanes;
         test
           "a gated bfloat16 load takes its alternative where its gate is false"
           takes_bfloat16;
         vectors;
         signed_zeros;
         nan_constants;
         invalid_lanes;
         square_wrapping;
         unselected_sum;
         weak_max;
         gated_stores;
         divisions;
         range_shrinking;
         blocks;
         beam_search;
         caching;
         linearizing;
         on_disk;
         programs;
         errors;
         diagnostics;
         line_rewrites;
         cleanups;
       ])
