open Windtrap
open Tolk_next

(* The host's target, as the engine gives it. *)
let host_target = Tolk_next_engine.target Nx_device.host
let rejects f = raises_match (Exn.invalid_arg ?substring:None) f

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
  Helpers.context
    (B (Helpers.no_color, true) :: Kernel_opts.settings_of_cell (row "setting"))
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
        let make = if default row then test else slow in
        Some (make (row "name") (fun () -> check row)))
    rows

(* A law that lowers a kernel again runs by default on a few kernels only: an
   elementwise one, a reduction, a gated store, a symbolic shape and an
   unroll. *)
let sample row =
  List.mem (row "case")
    [
      "elementwise_add"; "reduce_rows"; "gated_store"; "symbolic"; "sum_unroll";
    ]

(* D9: an emulated float8 converts as IEEE does, where tinygrad saturates and
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

(* D16: CUDA keeps a float8 infinity special; the source compares with
   tinygrad's once the guard is written back as tinygrad writes it. D17: a
   narrowed program's source is tinygrad's for its instructions with the casts
   that narrow, its golden <name>_narrowed. *)
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
        Some
          (if by_default row then t
           else group ~tags:[ "slow" ] (row "name") [ t ]))
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
      group "an emulated float8 program launches as tinygrad's (D9)"
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
  List.for_all (fun v -> not (List.mem (Ops.op v) opaque)) (Ops.toposort u)

let bound k =
  List.map
    (fun (name, v) -> (name, `Int (Z.of_int v)))
    (Kernel_opts.variables k)

(* The writes the interpreter gives [u] from the storage and variables that
   [Kernel_opts.writes] gives the kernel [k]. *)
let writes_of k u =
  Interpreter.writes ~vars:(bound k) ~buffers:(Kernel_opts.inputs k) u

let keeps_writes row =
  let k = kernel row in
  equal (list Kernel_opts.write) (Kernel_opts.writes k)
    (writes_of k (lowered row))

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
   the kernel writes, whose values may equal what they held. *)
let written_by_run k outputs =
  let inputs = Kernel_opts.inputs k in
  let expected = Kernel_opts.writes k in
  List.concat_map
    (fun (slot, after) ->
      let before = List.assoc slot inputs in
      List.filteri
        (fun i _ ->
          (not (Testable.equal Dtypes.value before.(i) after.(i)))
          || List.exists (fun (s, j, _) -> s = slot && j = i) expected)
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
      let word = Z.extract z (i * width) width in
      Dtype.bitcast (unsigned st) st (`Int word))

let of_words ~dt ~st ws =
  let width = Dtype.bitsize st in
  let z =
    List.fold_left
      (fun (acc, i) w ->
        (Z.logor acc (Z.shift_left (bits st w) (i * width)), i + 1))
      (Z.zero, 0) ws
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
  let kernel = buffer_dtypes (Ops.toposort k)
  and program = buffer_dtypes uops in
  let outputs =
    Run.on_host ~vars:(Kernel_opts.variables k)
      (Run.program (Lazy.force host) uops)
      (stored ~kernel ~program (Kernel_opts.inputs k))
  in
  equal (list Kernel_opts.write) (Kernel_opts.writes k)
    (written_by_run k (read_back ~kernel ~program outputs))

(* Each kernel is compiled by the host's C compiler, so the default run runs a
   few of them. *)
let run_by_default = [ "add"; "gated_store"; "fp8" ]

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
             (if List.mem name run_by_default then test else slow)
               name (runs_as_interpreted name))
           run_on_host);
    ]

(* Beam search (D4) *)

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
  group "beam search (D4)"
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

(* Programs are kept (D5) *)

(* A kernel no other call compiles, even in a rerun of the suite: it stores a
   value of its own into 16 floats. *)
let fresh = Atomic.make 1000

let fresh_kernel ?opts_to_apply () =
  let value = Float.of_int (Atomic.fetch_and_add fresh 1) in
  let out = Ops.param ~shape:[ Int 16 ] 0 Float32 in
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

(* The heuristic upcasts a 16 by 16 addition, and NOOPT keeps it from it; the
   kernel is renamed so that no other test compiles it. *)
let separates_settings () =
  let add = kernel_of "add_clang" in
  let k = Ops.replace add ~arg:(Kernel (Ops.kernel_info ~name:"settings" ())) in
  let optimised = Codegen.to_program k clang in
  let unoptimised =
    Helpers.context
      [ B (Helpers.noopt, true) ]
      (fun () -> Codegen.to_program k clang)
  in
  let applied p = (kernel_info (Ops.nth p 0)).applied_opts in
  equal (list Kernel_opts.opt) ~msg:"with the heuristic" [ upcast4 ]
    (applied optimised);
  equal (list Kernel_opts.opt) ~msg:"under NOOPT" [] (applied unoptimised)

let target_of p =
  match Ops.arg p with
  | Program { target; _ } -> target.renderer
  | _ -> invalid_arg "a program holds its target"

let separates_targets () =
  let k = fresh_kernel () in
  equal string "CLANG" (target_of (Codegen.to_program k clang));
  equal string "METAL" (target_of (Codegen.to_program k metal))

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

let caching =
  group "programs are kept"
    [
      test "a second call with an equal kernel compiles nothing"
        keeps_its_programs;
      test
        "calls from several domains at once make one program, compiled once \
         (D5)"
        compiles_once_across_domains;
      test "a program made under one setting is not returned under another"
        separates_settings;
      test "a program for one target is not returned for another"
        separates_targets;
      test "a failed compilation is not kept: the next call compiles anew"
        makes_anew_after_a_failure;
    ]

(* Programs *)

let on_clang row = row "target" = "clang" && compiles row

let resumes_from_its_sink row =
  let p = program row in
  let r = renderer_of_row row in
  under row (fun () ->
      same_graph ~msg:"from the lowered sink" p
        (Codegen.to_program (Ops.v Program ~src:[ lowered row ]) r);
      same_graph ~msg:"from the sink and the instructions, with its argument" p
        (Codegen.to_program
           (Ops.v Program ~src:[ Ops.nth p 0; Ops.nth p 1 ] ~arg:(Ops.arg p))
           r))

let estimates row =
  let p = program row in
  match (kernel_info (Ops.nth p 0)).estimates with
  | Some e -> Format.asprintf "%a" Ops.pp_estimates e
  | None -> "none"

let counts_its_instructions row =
  let p = program row in
  equal string
    (Format.asprintf "%a" Ops.pp_estimates
       (Renderer.Estimates.of_uops ~ignore_indexing:true
          (Ops.src (Ops.nth p 1))))
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
        Ops.variable ~dtype:Int32 (Printf.sprintf "v%d" i) (`Int Z.zero)
          (`Int (Z.of_int 100)))
  in
  let out = Ops.param ~shape:[ Int 1 ] 0 Int32 in
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
  (match
     Nx_device.Program.load Nx_device.host ~binary:elf.lib ~name:elf.name
   with
  | Ok prg ->
      Nx_device.Program.call prg
        [| Nx_device.Buffer.of_bigarray result |]
        (Array.of_list (List.map value info.vars))
  | Error why -> fail why);
  equal ~msg:"1 + 2 + ... + 10" int32 55l result.{0}

let programs =
  group "programs"
    [
      group "to_program resumes a program from its lowered sink"
        (per_row ~only:on_clang ~default:sample resumes_from_its_sink);
      group "the estimates count the instructions, leaving out index arithmetic"
        (per_row ~only:compiles counts_its_instructions);
      group "full_rewrite_to_sink lowers a kernel the same each time"
        (per_row ~only:on_clang ~default:sample lowers_the_same_twice);
      test
        "a device program's binary is its source compiled by the renderer's \
         compiler" (fun () ->
          let p = Codegen.to_program (Lazy.force add_kernel) metal in
          equal string (source p) (binary_of p));
      test "a host program's binary holds nx.device's entry after the kernel"
        (fun () ->
          let p = Codegen.to_program (Lazy.force add_kernel) clang in
          equal string
            (String.concat "\n"
               [
                 "#define E_64_4 E_64_4_";
                 source p;
                 "#undef E_64_4";
                 "void E_64_4(void **b, const long long *v) { E_64_4_(b[0], \
                  b[1], b[2]); }";
               ])
            (binary_of p));
      test
        "a host program runs through Program.call, with ten scalars past the \
         argument registers"
        ten_scalars;
      test "the argument of a program is its program information for the target"
        (fun () ->
          let p = Codegen.to_program (Lazy.force add_kernel) clang in
          equal string
            (Format.asprintf "%a" Ops.pp_program_info
               (Ops.program_info_of_sink ~target:clang_target (Ops.nth p 0)))
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
        (Ops.max_numel (Ops.nth first 1));
      is_true ~msg:"the first store is to local memory"
        (List.exists (in_space Local) (Ops.toposort first));
      equal int ~msg:"lanes of the global store" 1
        (Ops.max_numel (Ops.nth last 1));
      equal Dtypes.dtype Float32 (Ops.dtype (Ops.nth last 1));
      is_true ~msg:"the last store is to a parameter"
        (List.exists (is Param) (Ops.toposort last))
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
        | String name, `Int z -> (name, Z.to_int z + 1)
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
  List.length (List.filter wide (alu (Ops.toposort (Ops.nth (program row) 0))))

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
        (fun row -> equal int 2 (count Barrier (instructions row)));
      claim "reduce_shapeless_const_unroll"
        "folds the sum of a constant over an unroll" (fun row ->
          let nodes = Ops.toposort (lowered row) in
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

let lowered_nodes row = Ops.toposort (lowered row)

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
          equal Uops.uop ~msg:"the condition" (Ops.param 2 Bool) (Ops.nth e 2));
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
  Helpers.context
    [ B (Helpers.spec, 0) ]
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

let floats xs = Ops.stack (List.map (Ops.float ~dtype:Float32) xs)
let ints ?(dtype = Dtype.Int32) xs = Ops.stack (List.map (Ops.int ~dtype) xs)

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
            (apply_rewrite (Ops.stack [ Ops.nth v 2; Ops.nth v 3 ]))
            (apply_rewrite
               (Ops.stack
                  [ Ops.index v [ Ops.int 2 ]; Ops.index v [ Ops.int 3 ] ])));
      test "a vector of every lane of a vector is the vector" (fun () ->
          equal Uops.uop (apply_rewrite v)
            (apply_rewrite
               (Ops.stack (List.init 4 (fun i -> Ops.index v [ Ops.int i ])))));
      test "a comparison of lanes keeps the lanes apart" (fun () ->
          let lidx = Ops.special (Int 32) "lidx0" in
          let lhs =
            Ops.add
              (Ops.stack [ lidx; lidx; lidx; lidx ])
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
    (Ops.param ~shape:[ Int 8 ] slot Float32)
    [ Ops.valid (Ops.mul gidx0 (Ops.int 2)) (Ops.lt gidx0 (Ops.int 1)) ]

let loaded ?(slot = 0) ?(size = 3) dt i =
  Ops.index (Ops.param ~shape:[ Int size ] slot dt) [ Ops.int i ]

let fast_idiv = [ Helpers.B (Helpers.disable_fast_idiv, false) ]

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
              (Ops.param ~shape:[ Int 8 ] 1 Float32)
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
    cases ~name:(Format.asprintf "%a" Dtype.pp) name dts f
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
      test
        "a dividend that can wrap keeps the correction of a floor division \
         (D24)" (fun () ->
          (* tinygrad's test divides max x 0 + 1, which is negative at the
             greatest int, where tolk.next's int32 wraps. *)
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
      cases ~name:(Format.asprintf "%a" Op.pp)
        "an unsigned floor division is a truncating one"
        [ Op.Floordiv; Floormod ] (fun o ->
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
          Helpers.context fast_idiv (fun () ->
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
      cases
        ~name:(fun (o, d) -> Format.asprintf "%a by %d" Op.pp o d)
        "a division by a divisor that is not positive is no shift"
        [ (Op.Cdiv, -3); (Cdiv, 0); (Cmod, -3); (Cmod, 0) ]
        (fun (o, d) ->
          Helpers.context fast_idiv (fun () ->
              lacks Shr
                (instructions_of
                   [ Ops.alu (Ops.range (Int 20) [ 0 ]) o [ Ops.int d ] ])));
      test "a remainder the fast division declines stays a remainder" (fun () ->
          Helpers.context fast_idiv (fun () ->
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
          let x = Ops.variable ~dtype:Int32 "x" (`Int Z.zero) (`Int Z.one) in
          let lowered =
            Helpers.context fast_idiv (fun () ->
                full_rewrite [ Ops.alu x Cdiv [ Ops.int 3 ] ])
          in
          List.iter
            (fun v ->
              equal Dtypes.const ~msg:(string_of_int v) (`Int Z.zero)
                (Interpreter.eval
                   ~vars:[ ("x", `Int (Z.of_int v)) ]
                   (Ops.nth lowered 0)))
            [ 0; 1 ]);
      test "a division by 7 times 64 shifts out the 64 first, in 32 bits"
        (fun () ->
          Helpers.context fast_idiv (fun () ->
              let r = Ops.range (Int (1 lsl 20)) [ 0 ] in
              let uops =
                instructions_of
                  [ Ops.div ~rounding:`Floor r (Ops.int (7 * 64)) ]
              in
              equal (list Uops.uop) [] (List.filter wide uops);
              lacks Cdiv uops));
      test "DISABLE_FAST_IDIV=1 keeps a division by 3" (fun () ->
          Helpers.context
            [ B (Helpers.disable_fast_idiv, true) ]
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
  Helpers.context
    [ B (Helpers.noopt, true) ]
    (fun () -> List.filter (is Range) (Ops.toposort (full_rewrite uops)))

let lowered_end n =
  Helpers.context
    [ B (Helpers.noopt, true) ]
    (fun () -> Ops.nth (full_rewrite [ Ops.int ~dtype:Int32 n ]) 0)

let ends_at n uops =
  match ranges_left uops with
  | [ r ] -> equal Uops.uop (lowered_end n) (Ops.nth r 0)
  | rs -> failf "one range, not %d" (List.length rs)

let r = Ops.range (Int 204) [ 0 ]
let table = Ops.param ~shape:[ Int 1024 ] 0 Float32

let gated_load bound =
  Ops.load (Ops.index table [ Ops.valid r (Ops.lt r (Ops.int bound)) ]) []

let out = Ops.param ~shape:[ Int 204 ] 0 Float32

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
              (Ops.index (Ops.param ~shape:[ Int 204 ] 1 Float32) [ r ])
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
      (Ops.Pattern_matcher.v
         [
           Ops.Pattern_matcher.rule (Ops.Upat.op Customi) (fun _ ->
               Some (Ops.v Source ~arg:(String "x")));
         ])
    (target "" "" "")

let marker_kernel =
  lazy
    (let marker =
       Ops.v Customi ~arg:(Code { code = "make_movement()"; dtype = Int32 })
     in
     Ops.sink ~kernel:(Ops.kernel_info ())
       [
         Ops.store
           (Ops.index (Ops.param ~shape:[ Int 1 ] 0 Int32) [ Ops.int 0 ])
           marker;
       ])

let checks_what_it_lowers () =
  Helpers.context
    [ B (Helpers.spec, 1) ]
    (fun () ->
      raises_match (Exn.invalid_arg ~substring:"UOp verification failed")
        (fun () ->
          Codegen.full_rewrite_to_sink ~optimize:false
            (Lazy.force marker_kernel) breaking))

(* Tensor-core accumulators (D24) *)

(* A Metal tensor core's product added, from the constant accumulator [c], to
   two floats the kernel loads: [out[i] = (wmma a b c)[i] + acc[i]], where
   [acc], of slot 2, stands for the running sum of a tensor-core loop. *)
let lane slot dt i =
  Ops.index (Ops.param ~shape:[ Int 2 ] slot dt) [ Ops.int ~dtype:Int32 i ]

let lanes slot dt =
  Ops.stack (List.init 2 (fun i -> Ops.load (lane slot dt i) []))

let accumulated c =
  let c = Ops.float ~dtype:Float32 c in
  let product =
    Ops.wmma (lanes 0 Float16) (lanes 1 Float16)
      ~acc:(Ops.stack [ c; c ])
      ~dims:(8, 8, 8) ~threads:32
  in
  let sum = Ops.add product (lanes 2 Float32) in
  Ops.sink ~kernel:(Ops.kernel_info ())
    (List.init 2 (fun i ->
         Ops.store (lane 3 Float32 i) (Ops.index sum [ Ops.int ~dtype:Int32 i ])))

let lowered_on_metal k =
  Helpers.context
    [ B (Helpers.spec, 0) ]
    (fun () -> Codegen.full_rewrite_to_sink ~optimize:false k metal)

(* A tensor core computes its accumulator plus a product that does not depend on
   it: each tensor core of [u] stands here for its accumulator plus the two
   floats of slot 4. *)
let with_products u =
  let products = lanes 4 Float32 in
  Ops.substitute u
    (List.filter_map
       (fun w ->
         if is Wmma w then Some (w, Ops.add (Ops.nth w 2) products) else None)
       (Ops.toposort u))

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
        (List.filter float_constant (Ops.toposort (Ops.nth w 2))))
    (List.filter (is Wmma) (Ops.toposort (lowered_on_metal (accumulated c))))

let accumulators =
  group "tensor-core accumulators (D24)"
    [
      cases ~name:string_of_float "the running sum replaces a zero accumulator"
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

(* Lanes of a scalar (D53) *)

(* A read of a lane by a constant, from a value without lanes: a scalar, or a
   constant. *)
let reads_a_lane_of_a_scalar u =
  let constant c = is Const c || (is Cast c && is Const (Ops.nth c 0)) in
  is Index u
  &&
  match Ops.src u with
  | [ x; c ] ->
      constant c
      && Ops.shape_opt x = Some []
      && Ops.addrspace x = Some Dtype.Alu
  | _ -> false

let reads_no_lane_of_a_scalar row =
  equal (list Uops.uop) []
    (List.filter reads_a_lane_of_a_scalar (instructions row))

let lanes =
  group "lanes of a scalar (D53)"
    [
      test "a vector folded to a scalar is rendered as that scalar on Metal"
        (fun () -> writes_as_tinygrad (row_named "invalid_lanes_metal"));
      test
        "on the host, the program of a vector folded to a scalar writes what \
         its kernel writes"
        (runs_as_interpreted "invalid_lanes");
      test "a sum folded to a constant is rendered as that constant on Metal"
        (fun () -> writes_as_tinygrad (row_named "invalid_lanes_int8_metal"));
      test
        "on the host, the program of a sum folded to a constant writes what \
         its kernel writes"
        (runs_as_interpreted "invalid_lanes_int8");
      group "no program reads a lane of a scalar or a constant"
        (per_row ~only:compiles reads_no_lane_of_a_scalar);
    ]

(* Vectors in programs (D58) *)

let on_a_vector u =
  Op.Set.mem (Ops.op u) Op.Set.elementwise
  && match Ops.shape_opt u with Some (_ :: _) -> true | _ -> false

let applies_no_elementwise_operation_to_a_vector row =
  equal (list Uops.uop) [] (List.filter on_a_vector (instructions row))

(* A fold of an int8 [2; 2] to [2; 1], kernel [1; 2], dilation [1; 2] and
   padding [(0, 0); (1, 1)], every window in the padding. In devectorize, the
   gated load of each upcast lane folds to a scalar 0, and the select around it
   stays a select of two lanes, as tinygrad's does, which the weak lowering
   makes a cast of a stack of constants. *)
let vector_select_kernel () =
  let open Ops.O in
  let out = Ops.param ~shape:[ Int 2 ] 0 Int8 in
  let x = Ops.param ~shape:[ Int 4 ] 1 Int8 in
  let l = Ops.range ~axis_type:Weak (Int 2) [ 2 ] in
  let r0 = Ops.range ~axis_type:Reduce (Int 2) [ 0 ] in
  let r1 = Ops.range ~axis_type:Reduce (Int 4) [ 1 ] in
  let j = (r1 * int 3) + int 1 in
  let gate =
    (((r0 * int 2) + l < int 3) land (r1 < int 3))
    land ((r0 < int 1) land (j % int 5 < int 1))
  in
  let read = Ops.index x [ Ops.valid ((j // int 5 * int 2) + l) (r1 < int 3) ] in
  let value = Ops.where gate read (Ops.int ~dtype:Int8 0) in
  let sum = Ops.reduce (Ops.cast value Uint32) Op.Add [ r0; r1 ] in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (Ops.index out [ l ]) (Ops.cast sum Int8)) [ l ] ]

let vectors =
  group "vectors in programs (D58)"
    [
      test "a cast left on two lanes after devectorize is refused" (fun () ->
          Helpers.context
            [ B (Helpers.spec, 1) ]
            (fun () ->
              raises_match
                (Exn.invalid_arg ~substring:"on Ops.CAST")
                (fun () -> Codegen.to_program (vector_select_kernel ()) clang)));
      group "no program applies an elementwise operation to a vector"
        (per_row ~only:compiles applies_no_elementwise_operation_to_a_vector);
    ]

(* Signed zeros (D52) *)

(* [padded ~flip fill before after xs] stores into a new buffer the float32
   values [xs] padded with [before] and [after] elements of [fill], as a pad by
   a value other than +0. is lowered: [fill] is selected off a mask of [xs]'
   elements, the pad's own zeros only where the mask holds. With [flip], the
   selection is by the mask's negation, [fill] first. *)
let padded ?(flip = false) fill before after xs =
  let n = Array.length xs in
  let x = Ops.new_buffer ~slot:1 (Single "CPU") n Float32 in
  let padding = [ Some (Ops.Int before, Ops.Int after) ] in
  let mask = Ops.pad (Ops.const_like ~dtype:Bool x (`Bool true)) padding in
  let fill = Ops.float ~dtype:Float32 fill in
  let value =
    if flip then Ops.where (Ops.logical_not mask) fill (Ops.pad x padding)
    else Ops.where mask (Ops.pad x padding) fill
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
  group "signed zeros (D52)"
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
      prop "a pad and a selection of signed zeros keep the interpreter's bits"
        signed_zero_pads (fun (flip, fill, before, after, xs) ->
          padding_computed ~flip fill before after xs);
    ]

(* Lanes all Invalid *)

(* rune's fold of a float32 [4; 1] to [3; 1], kernel [2; 2], dilation [2; 2]
   and padding [(0, 0); (1, 1)], every window of the second axis in the
   padding. Upcast, every lane of its gated index is Invalid, and the stack of
   lanes folds to one Invalid without its width, as tinygrad's does; the
   reshape and the permute of the lanes are then left over a scalar. *)
let invalid_lanes_kernel () =
  let open Ops.O in
  let out = Ops.param ~shape:[ Int 3 ] 0 Float32 in
  let x = Ops.param ~shape:[ Int 4 ] 1 Float32 in
  let o = Ops.range ~axis_type:Weak (Int 3) [ 2 ] in
  let r0 = Ops.range ~axis_type:Reduce (Int 4) [ 0 ] in
  let r1 = Ops.range ~axis_type:Reduce (Int 4) [ 1 ] in
  let j = (r0 * int 3) + o and k = (r1 * int 3) + int 1 in
  let inside = (r1 < int 3) land (j < int 10) in
  let gate =
    (j < int 10) land (r1 < int 3)
    land ((j % int 5 < int 1) land (k % int 5 < int 1))
  in
  let at = (j // int 5 * int 2) + (k // int 5) in
  let read = Ops.index x [ Ops.valid at inside ] in
  let zero = Ops.float ~dtype:Float32 0. in
  let sum = Ops.reduce (Ops.where gate read zero) Op.Add [ r0; r1 ] + zero in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (Ops.index out [ o ]) sum) [ o ] ]

let invalid_lanes =
  group "lanes all Invalid"
    [
      xfail
        ~reason:
          "a stack of Invalid lanes folds to one Invalid without its width, \
           as tinygrad's does"
        (slow "a kernel whose upcast lanes all read an Invalid index compiles"
           (fun () ->
             ignore (Codegen.to_program (invalid_lanes_kernel ()) clang)));
    ]

(* Errors *)

(* A kernel holds no conditional: only a program's instructions do. *)
let spec_breaking =
  let at = Ops.index (Ops.param ~shape:[ Int 1 ] 0 Int32) [ Ops.int 0 ] in
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
            Helpers.context
              [ B (Helpers.spec, 0) ]
              (fun () ->
                Codegen.full_rewrite_to_sink ~optimize:false
                  (Lazy.force marker_kernel) breaking)
          in
          mem op Source (List.map Ops.op (Ops.toposort lowered)));
      test "with SPEC=0, a kernel that breaks the specification is lowered"
        (fun () ->
          let lowered =
            Helpers.context
              [ B (Helpers.spec, 0) ]
              (fun () ->
                Codegen.full_rewrite_to_sink ~optimize:false spec_breaking clang)
          in
          mem op If (List.map Ops.op (Ops.toposort lowered)));
      test "a kernel that breaks the specification raises Invalid_argument"
        (fun () ->
          Helpers.context
            [ B (Helpers.spec, 1) ]
            (fun () ->
              raises_match
                (Exn.invalid_arg ~substring:"UOp verification failed")
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

let at_debug level f = Helpers.context [ B (Helpers.debug, level) ] f

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

let prints_a_breaking_graph () =
  setenv "DBGTV" (Some "1");
  Helpers.context
    [ B (Helpers.spec, 1) ]
    (fun () ->
      rejects (fun () ->
          Codegen.full_rewrite_to_sink ~optimize:false
            (Lazy.force marker_kernel) breaking));
  contains ~sub:"Ops.SOURCE" (output ())

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
      cases
        ~name:(fun opts -> string_of_int (List.length opts) ^ " optimisations")
        "DEBUG=3 prints the optimisations applied"
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
let matcher rules = Ops.Pattern_matcher.fold rules
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
let out = Ops.param ~shape:[ Int 4 ] 0 Int32
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
      cases ~name:fst "leaves alone a store it does not match"
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
      cases ~name:fst "raises Invalid_argument on an if already in the list"
        [
          ("if", Ops.v If ~src:[ gate; at ]);
          ("endif", Ops.v Endif ~src:[ Ops.v If ~src:[ gate; at ] ]);
        ]
        (fun (_, u) -> rejects (fun () -> cleaned [ u ]));
    ]

let () =
  exit
    (run "Tolk_next.Codegen"
       [
         stages;
         values;
         claims;
         lowering_claims;
         whole_graphs;
         accumulators;
         lanes;
         vectors;
         signed_zeros;
         invalid_lanes;
         gated_stores;
         divisions;
         range_shrinking;
         beam_search;
         caching;
         programs;
         errors;
         diagnostics;
         line_rewrites;
         cleanups;
       ])
