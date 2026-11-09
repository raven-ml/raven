open Windtrap
open Tolk

(* The host's target, as the engine gives it. *)
let host_target = Tolk_engine.target Rig.host
let rejects f = raises_match (Exn.invalid_arg ?sub:None) f
let tensor_core = Testable.make ~pp:Tc.pp ~equal:Tc.equal

let contains s sub =
  let n = String.length sub in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = sub || at (i + 1))
  in
  at 0

(* Targets *)

let target device renderer arch =
  { Helpers.Target.device; renderer; arch; interface = ""; indices = "" }

(* A target's renderer, made without its toolchain: it renders on any
   machine. *)
let renderer_for = Cstyle.without_compiler

(* A GPU's renderer with its compiler, which finds its vendor's library or
   raises Failure: [compiling] skips the test on a machine without it. *)
let compiling (t : Helpers.Target.t) =
  let make =
    match t.renderer with
    | "CLANG" -> Cstyle.clang
    | "METAL" -> Cstyle.metal
    | "CUDA" -> Cstyle.cuda
    | "HIP" -> Cstyle.hip
    | r -> invalid_arg ("no renderer " ^ r)
  in
  match make t with r -> r | exception Failure why -> skip ~reason:why ()

(* The target of a golden row, written in its columns device, renderer and
   arch. *)
let target_of_row row = target (row "device") (row "renderer") (row "arch")
let renderer_of_row row = renderer_for (target_of_row row)
let clang_target = target "CPU" "CLANG" "x86_64,x86-64"
let metal_target = target "METAL" "METAL" "Apple9"
let cuda_target = target "CUDA" "CUDA" "sm_89"
let hip_target = target "AMD" "HIP" "gfx1100"
let clang = Cstyle.clang clang_target
let metal () = renderer_for metal_target
let cuda () = renderer_for cuda_target
let hip () = renderer_for hip_target

let renderers =
  [ ("clang", fun () -> clang); ("metal", metal); ("cuda", cuda); ("hip", hip) ]

let available =
  List.map renderer_for [ clang_target; metal_target; cuda_target; hip_target ]
let render (r : Renderer.t) uops = r.render uops

(* Sources *)

(* The cases of kernels.golden are its sink's sources, each a linear program
   named after its case, whose sources are the kernel's nodes in order. *)
let kernels = lazy (Array.of_list (Ops.src (Golden.sink "kernels.golden")))
let kernel row = Ops.src (Lazy.force kernels).(int_of_string (row "kernel"))
let case_rows = Golden.rows "cases.golden"
let find_case name = List.find (fun row -> row "case" = name) case_rows

let source_of_case name =
  render (renderer_of_row (find_case name)) (kernel (find_case name))

let sources_under setting =
  List.filter_map
    (fun row ->
      if row "setting" <> setting then None
      else
        let source () = render (renderer_of_row row) (kernel row) in
        Some (Golden.text (row "case" ^ ".golden") source))
    case_rows

(* A setting is read once per process, so the cases rendered under one run in a
   process of their own, which the stanza starts with the setting's variable set
   and the setting's tag selected. *)
let sources =
  group "sources"
    [
      group "by default" (sources_under "-");
      tag "expand-ssa"
      @@ group "with EXPAND_SSA=1" (sources_under "EXPAND_SSA=1");
    ]

(* C computes an operation on a signed char, a short or Clang's __fp16 in a
   wider type, so the source casts each such operation back to its type: an operation on a
   scalar of one of these types, inlined into its one user, which does not store
   it. The source of a kernel is then tinygrad's for the kernel with those
   casts, which [with_d17_casts row uops] builds and cases.golden's column
   narrowed names. *)

let narrow_scalars row =
  Dtype.[ Int8; Int16 ]
  @ if row "renderer" = "CLANG" then [ Dtype.Float16 ] else []

let with_d17_casts row uops =
  let children = Ops.Tbl.create 256 and user = Ops.Tbl.create 256 in
  let count v = Option.value ~default:0 (Ops.Tbl.find_opt children v) in
  List.iter
    (fun u ->
      List.iter
        (fun v ->
          Ops.Tbl.replace children v (count v + 1);
          Ops.Tbl.replace user v u)
        (Ops.src u))
    uops;
  let stores u =
    let s = Ops.Tbl.find user u in
    Op.equal (Ops.op s) Store && Ops.equal (Ops.nth s 1) u
  in
  let narrowed u =
    Op.Set.mem (Ops.op u) Op.Set.alu
    && (not (Op.equal (Ops.op u) Where))
    && count u = 1
    && Shape.max_numel u = 1
    && List.mem (Ops.dtype u) (narrow_scalars row)
    && not (stores u)
  in
  let rebuilt = Ops.Tbl.create 256 in
  let find v = Option.value ~default:v (Ops.Tbl.find_opt rebuilt v) in
  List.concat_map
    (fun u ->
      let u' = Ops.replace ~src:(List.map find (Ops.src u)) u in
      if narrowed u then begin
        let cast = Ops.v Cast ~src:[ u' ] ~arg:(Dtype (Ops.dtype u')) in
        Ops.Tbl.replace rebuilt u cast;
        [ u'; cast ]
      end
      else begin
        Ops.Tbl.replace rebuilt u u';
        [ u' ]
      end)
    uops

let written_kernel row =
  match row "narrowed" with
  | "-" -> kernel row
  | i -> Ops.src (Lazy.force kernels).(int_of_string i)

let narrowing =
  group "narrowing"
    [
      Golden.cases "cases.golden" (fun row ->
          if row "setting" = "-" then
            equal (list Uops.uop) (written_kernel row)
              (with_d17_casts row (kernel row)));
    ]

(* Rewrites *)

let rewrite_inputs =
  lazy (Array.of_list (Ops.src (Golden.sink "rewrite_inputs.golden")))

let rewritten = lazy (Array.of_list (Ops.src (Golden.sink "rewritten.golden")))
let nth_of file row column = (Lazy.force file).(int_of_string (row column))

let rewrite row =
  let r = renderer_of_row row in
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
    (nth_of rewrite_inputs row "input")
    (After_sources r.extra_matcher)

(* Under [-u], each row's rewrite is put in its place among upstream's outputs,
   and the whole is recorded as [rewritten.tolk] when it differs. *)
let recorded_rewrites () =
  let outputs = Array.of_list (Ops.src (Golden.upstream "rewritten.golden")) in
  List.iter
    (fun row -> outputs.(int_of_string (row "output")) <- rewrite row)
    (Golden.rows "rewrites.golden");
  Ops.sink (Array.to_list outputs)

let rewrites =
  group "extra_matcher"
    [
      (if Golden.updating then Golden.graph "rewritten.golden" recorded_rewrites
       else
         Golden.cases "rewrites.golden" (fun row ->
             equal Uops.uop (nth_of rewritten row "output") (rewrite row)));
    ]

(* Declarations *)

let sizes_of_cell = function
  | "-" -> None
  | s -> Some (List.map int_of_string (String.split_on_char ' ' s))

let bool_of_cell = function
  | "True" -> true
  | "False" -> false
  | s -> invalid_arg ("not a boolean: " ^ s)

let joined sep pp xs = String.concat sep (List.map (Format.asprintf "%a" pp) xs)
let or_dash = function "" -> "-" | s -> s

let op_names (r : Renderer.t) =
  List.sort String.compare (List.map (fun (op, _) -> Op.name op) r.code_for_op)
  |> String.concat " "

let declared column check =
  group column
    [
      Golden.cases "declarations.golden" ~key:[ "device"; "arch" ] (fun row ->
          check (renderer_of_row row) (row column));
    ]

let declarations =
  group "declarations"
    [
      declared "supports_float4" (fun r cell ->
          equal bool (bool_of_cell cell) r.supports_float4);
      declared "has_local" (fun r cell ->
          equal bool (bool_of_cell cell) r.has_local);
      declared "has_shared" (fun r cell ->
          equal bool (bool_of_cell cell) r.has_shared);
      declared "global_max" (fun r cell ->
          equal (option (list int)) (sizes_of_cell cell) (Some r.global_max));
      declared "local_max" (fun r cell ->
          equal (option (list int)) (sizes_of_cell cell) (Some r.local_max));
      declared "global_prod_max" (fun r cell ->
          equal (option (list int)) (sizes_of_cell cell) r.global_prod_max);
      declared "shared_max" (fun r cell ->
          equal int (int_of_string cell) r.shared_max);
      declared "tensor_cores" (fun r cell ->
          equal string cell (or_dash (joined " | " Tc.pp r.tensor_cores)));
      declared "code_for_op" (fun r cell -> equal string cell (op_names r));
      declared "supported" (fun r cell ->
          equal string cell (joined " " Dtype.pp (Renderer.supported_dtypes r)));
      declared "cachekey" (fun r cell ->
          (* tinygrad's table, then the digest of the toolchain and its
             options. *)
          let r = compiling r.target in
          let table = Option.get (Renderer.Compiler.cachekey r.compiler) in
          let prefix = cell ^ "_" in
          let n = Int.min (String.length prefix) (String.length table) in
          equal string ~msg:"the name" prefix (String.sub table 0 n);
          equal int ~msg:"the digest's length" 32 (String.length table - n));
    ]

(* Operations *)

let written =
  group "code_for_op"
    [
      Golden.cases "written.golden" ~key:[ "renderer"; "op"; "dtype" ]
        (fun row ->
          let r = renderer_of_row row in
          let name = row "op" in
          let op =
            Result.get_ok
              (Op.of_string (String.sub name 4 (String.length name - 4)))
          in
          let operands =
            if Op.Set.mem op Op.Set.unary then [ "a" ]
            else if Op.Set.mem op Op.Set.ternary then [ "a"; "b"; "c" ]
            else [ "a"; "b" ]
          in
          let write = List.assoc op r.code_for_op in
          equal string (row "written")
            (write operands (Dtypes.dtype_of_cell (row "dtype"))));
    ]

(* Errors *)

let errors =
  group "architectures"
    [
      test "Clang rejects an architecture of one field" (fun () ->
          rejects (fun () -> Cstyle.clang (target "CPU" "CLANG" "x86_64")));
      test "Clang rejects a machine other than x86_64, arm64 and riscv64"
        (fun () ->
          rejects (fun () -> Cstyle.clang (target "CPU" "CLANG" "mips,generic")));
      test "Metal rejects an Apple family that is no integer" (fun () ->
          rejects (fun () -> Cstyle.metal (target "METAL" "METAL" "AppleX")));
      test "Metal has no tensor cores on a family that is not Apple's"
        (fun () ->
          equal (list tensor_core) []
            (renderer_for (target "METAL" "METAL" "Mac2")).tensor_cores);
      test "CUDA rejects an architecture without a compute capability"
        (fun () -> rejects (fun () -> Cstyle.cuda (target "CUDA" "CUDA" "sm_")));
    ]

(* Parentheses *)

(* Five operations in a row, each on the last's result: an associative operation
   is written without the parentheses of its operands, and a subtraction keeps
   them. *)
(* [chain n] stores [r n] for [r 0 = data1[0]] and
   [r (i+1) = (r i << data2[0]) % data3[0]], over int32. Each step is read once,
   by the next, so written into its reader the chain would nest two brackets a
   step. *)
let chain n =
  let zero = Ops.int ~dtype:Int32 0 in
  let at slot = Ops.index (Shape.param ~shape:[ Int 1 ] slot Int32) [ zero ] in
  let shift = Ops.load (at 2) [] and modulus = Ops.load (at 3) [] in
  let step r _ = Ops.alu (Ops.alu r Shl [ shift ]) Truncmod [ modulus ] in
  let r = List.fold_left step (Ops.load (at 1) []) (List.init n Fun.id) in
  Linearizer.linearize
    (Ops.sink ~kernel:(Ops.kernel_info ~name:"chain" ()) [ Ops.store (at 0) r ])

(* The deepest nesting of brackets, ( [ and {, in [s]. *)
let nesting s =
  let depth = ref 0 and deepest = ref 0 in
  String.iter
    (function
      | '(' | '[' | '{' ->
          incr depth;
          deepest := max !deepest !depth
      | ')' | ']' | '}' -> decr depth
      | _ -> ())
    s;
  !deepest

(* An expression written into its reader nests at most 64 brackets. The
   statement that declares it adds its operation's brackets, and the function
   its brace. *)
let max_nesting = 64 + 4

let renderers_and_lengths =
  let pp ppf (r : Renderer.t) = Format.pp_print_string ppf r.name in
  Gen.pair
    (Gen.of_list ~pp available)
    (Gen.int_range 1 300)

let nests_a_bounded_depth ((r : Renderer.t), n) =
  cover "a chain that would nest past 256 brackets" (n > 128);
  cover "a chain short enough to be one expression" (n < 30);
  at_most int ~than:max_nesting (nesting (render r (chain n)))

(* [sum n] stores [data1[0] + data1[1] + ... + data1[n-1]], over float32, each
   addition read once, by the next. *)
let sum n =
  let x i =
    Ops.load
      (Ops.index
         (Shape.param ~shape:[ Int n ] 1 Float32)
         [ Ops.int ~dtype:Int32 i ])
      []
  in
  let add r i = Ops.alu r Add [ x i ] in
  let r = List.fold_left add (x 0) (List.init (n - 1) succ) in
  let out =
    Ops.index
      (Shape.param ~shape:[ Int 1 ] 0 Float32)
      [ Ops.int ~dtype:Int32 0 ]
  in
  Linearizer.linearize
    (Ops.sink ~kernel:(Ops.kernel_info ~name:"sum" ()) [ Ops.store out r ])

(* The operands of the sum the kernel [src] stores, as written. *)
let stored_sum src =
  let prefix = "*(data0_1+0) = (" in
  let line =
    List.find (String.starts_with ~prefix)
      (List.map String.trim (String.split_on_char '\n' src))
  in
  let start = String.length prefix in
  let sum = String.sub line start (String.length line - start - 2) in
  String.split_on_char '+' sum

(* The element of [data1] each value of [src] loads, by its name, from the
   statements that declare them: [float valK = ...] loading [data1_N] at
   [+I]. *)
let loads src =
  List.filter_map
    (fun line ->
      match String.split_on_char ' ' (String.trim line) with
      | [ "float"; name; "="; load ] when String.starts_with ~prefix:"val" name
        -> (
          match String.split_on_char '+' load with
          | [ _; i ] ->
              Some (name, int_of_string (List.hd (String.split_on_char ')' i)))
          | _ -> None)
      | _ -> None)
    (String.split_on_char '\n' src)

let writes_a_flat_sum n =
  cover "a sum longer than the chains that nest" (n > 300);
  let src = render clang (sum n) in
  let element = loads src in
  equal (list int) (List.init n Fun.id)
    (List.map (fun o -> List.assoc o element) (stored_sum src))

let parentheses =
  group "parentheses"
    [
      prop "a chain of any length nests a bounded number of brackets"
        renderers_and_lengths nests_a_bounded_depth;
      prop "a sum of any length is one flat expression of its loaded operands"
        (Gen.int_range 2 1000) writes_a_flat_sum;
      cases "an associative chain is written flat" ~id:fst
        [
          ("add", true);
          ("mul", true);
          ("xor", true);
          ("or", true);
          ("and", true);
          ("sub", false);
        ]
        (fun (op, flat) ->
          let src = source_of_case ("clang_chain_" ^ op) in
          equal bool (not flat) (contains src "((((("));
    ]

(* Rendering *)

let arity op =
  if Op.Set.mem op Op.Set.unary then 1
  else if Op.Set.mem op Op.Set.ternary then 3
  else 2

(* A kernel that stores one operation of loaded operands: the loads are named
   val0, val1 and val2 in the order of the operands. *)
let one_operation op dt =
  let operand i = if Op.equal op Where && i = 0 then Dtype.Bool else dt in
  let at slot dt =
    Ops.index (Shape.param ~shape:[ Int 1 ] slot dt) [ Ops.int ~dtype:Int32 0 ]
  in
  let loads =
    List.init (arity op) (fun i -> Ops.load (at (i + 1) (operand i)) [])
  in
  let value = Ops.v op ~src:loads in
  Ops.toposort ~calls:Enter
    (Ops.sink [ Ops.store (at 0 (Ops.dtype value)) value ])

let floats_only = Op.[ Exp2; Log2; Sin; Sqrt; Recip; Trunc; Fdiv ]
let ints_only = Op.[ Shl; Shr; And; Or; Xor; Truncmod; Truncdiv ]

let operations =
  let named = List.map (fun (r : Renderer.t) -> (r.name, r)) available in
  List.concat_map
    (fun (name, (r : Renderer.t)) ->
      List.concat_map
        (fun (op, _) ->
          List.filter_map
            (fun dt ->
              let fits =
                if List.mem op floats_only then Dtype.is_float dt
                else if List.mem op ints_only then Dtype.is_int dt
                else not (Dtype.is_bool dt)
              in
              if fits then Some (name, r, op, dt) else None)
            (Renderer.supported_dtypes r))
        r.code_for_op)
    named
  |> Gen.of_list
  |> Gen.with_pp (fun ppf (name, _, op, dt) ->
      Format.fprintf ppf "%s %a %a" name Op.pp op Dtype.pp dt)

let writes_its_operation (_, (r : Renderer.t), op, dt) =
  let operands = List.init (arity op) (Printf.sprintf "val%d") in
  let expr = (List.assoc op r.code_for_op) operands dt in
  let src = render r (one_operation op dt) in
  satisfies
    ~claim:("a source that stores " ^ expr)
    string
    (fun src -> contains src (" = " ^ expr ^ ";"))
    src

let named_cases = Gen.of_list (List.map (fun row -> row "case") case_rows)

let keeps_no_state (a, b) =
  let first = source_of_case a in
  ignore (source_of_case b);
  equal string first (source_of_case a)

(* Nodes are shared, so the kernel stores to a slot no other kernel of the suite
   has, and none of its nodes is alive elsewhere. *)
let keeps_no_reference () =
  let weak = Stdlib.Weak.create 1 in
  let[@inline never] render_once () =
    let at =
      Ops.index
        (Shape.param ~shape:[ Int 1 ] 7139 Float32)
        [ Ops.int ~dtype:Int32 0 ]
    in
    let sink = Ops.sink [ Ops.store at (Ops.float ~dtype:Float32 7139.) ] in
    Stdlib.Weak.set weak 0 (Some sink);
    ignore (render clang (Ops.toposort ~calls:Enter sink))
  in
  render_once ();
  Gc.full_major ();
  is_false ~msg:"the kernel is still reachable" (Stdlib.Weak.check weak 0)

let cuda_on_nv () =
  let cuda = cuda () and nv = renderer_for (target "NV" "CUDA" "sm_89") in
  List.iter
    (fun row ->
      if row "renderer" = "CUDA" && row "setting" = "-" then
        equal string ~msg:(row "case")
          (render cuda (kernel row))
          (render nv (kernel row)))
    case_rows

(* Custom code is a format string of its operands, read as Python's str.format
   reads one. *)
let custom code =
  let at slot =
    Ops.index
      (Shape.param ~shape:[ Int 1 ] slot Float32)
      [ Ops.int ~dtype:Int32 0 ]
  in
  let operands = [ Ops.load (at 1) []; Ops.load (at 2) [] ] in
  let value =
    Ops.call ~inline:true
      (Ops.custom_function ~dtype:Float32 ~fmt:true code [])
      operands
  in
  render clang (Ops.toposort ~calls:Enter (Ops.sink [ Ops.store (at 0) value ]))

let formats_custom_code (code, expr) =
  satisfies
    ~claim:("a source that stores " ^ expr)
    string
    (fun src -> contains src (" = " ^ expr ^ ";"))
    (custom code)

let rendering =
  group "rendering"
    [
      prop "writes each native operation as code_for_op does" operations
        writes_its_operation;
      prop "keeps no state from one kernel to the next"
        (Gen.pair named_cases named_cases)
        keeps_no_state;
      test "keeps no reference to the kernel it rendered" keeps_no_reference;
      cases "formats custom code as str.format does" ~id:fst
        [
          ("{0}+{1}", "val0+val1");
          ("{1}-{0}", "val1-val0");
          ("{}*{}", "val0*val1");
          ("{{{0}}}", "{val0}");
          ("f({0}) }}", "f(val0) }");
        ]
        formats_custom_code;
      cases "refuses custom code str.format refuses, naming it" ~id:Fun.id
        [ "f({0"; "{0}}"; "{0} } {1}"; "{2}"; "{x}" ] (fun code ->
          raises_match (Exn.invalid_arg ~sub:code) (fun () -> custom code));
      test "refuses custom code holding a NUL byte, which C cannot" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"NUL")
            (fun () -> custom "f({0}, \"\000\")"));
      test "refuses a parameter named with a NUL byte" (fun () ->
          let at =
            Ops.index
              (Shape.param ~shape:[ Int 1 ] ~name:"x\000" 0 Float32)
              [ Ops.int ~dtype:Int32 0 ]
          in
          raises_match
            (Exn.invalid_arg ~sub:"NUL")
            (fun () ->
              render clang
                (Ops.toposort ~calls:Enter
                   (Ops.sink [ Ops.store at (Ops.float ~dtype:Float32 1.) ]))));
      test "writes the same CUDA source for the CUDA and NV devices" cuda_on_nv;
      cases "raises Invalid_argument on an operation the target lacks"
        ~id:(fun (name, _) -> name)
        renderers
        (fun (_, r) ->
          let r = r () in
          rejects (fun () -> render r (one_operation Max Float32)));
      cases
        "raises Invalid_argument on a kernel of another target it cannot write"
        ~id:fst
        [
          ( "HIP gfx1100 converts to a float8 it lacks",
            (hip, "hip_cdna4_dtype_fp8e4m3") );
          ( "CUDA has no core for chars",
            (cuda, "hip_tc_i8_i32_16_16_16") );
        ]
        (fun (_, (r, case)) ->
          let r = r () in
          rejects (fun () -> render r (kernel (find_case case))));
    ]

(* Compilation *)

(* A machine without a target's toolchain refuses every source with the reason
   it cannot load the library. *)
let lacks_toolchain why =
  String.starts_with ~prefix:"failed to load library" why
  || String.starts_with ~prefix:"comgr not available" why

(* The sources the toolchain rejects as tinygrad writes them. CUDA's vectors are
   structs, without casts and with their lanes as members. *)
let rejected =
  [
    ("cuda_dynamic_lane", "reads a lane by a variable, which NVRTC rejects");
    ("cuda_vector_cast", "casts a vector, which NVRTC rejects");
  ]

let compiles_with_its_toolchain row =
  let t =
    test (row "case") (fun () ->
        let r = compiling (target_of_row row) in
        match Renderer.Compiler.compile r.compiler (render r (kernel row)) with
        | _ -> ()
        | exception Renderer.Compiler.Compile_error why when lacks_toolchain why
          ->
            skip ~reason:why ())
  in
  match List.assoc_opt (row "case") rejected with
  | Some reason -> xfail ~reason t
  | None -> t

let gpu_rows =
  List.filter
    (fun row -> row "renderer" <> "CLANG" && row "setting" = "-")
    case_rows

let compilation =
  slow
  @@ group "every GPU kernel compiles with its target's toolchain"
       (List.map compiles_with_its_toolchain gpu_rows)

(* CUDA's binaries *)

(* [check] of the binary [ren] compiles cuda_add to. *)
let compiled ren check =
  match
    Renderer.Compiler.compile ren.Renderer.compiler (source_of_case "cuda_add")
  with
  | binary -> check binary
  | exception Renderer.Compiler.Compile_error why when lacks_toolchain why ->
      skip ~reason:why ()

let cuda_binaries =
  group "CUDA's binaries"
    [
      slow
      @@ test "the CUDA device's kernels compile to a cubin" (fun () ->
          compiled (compiling cuda_target) (starts_with ~prefix:"\x7fELF"));
      slow
      @@ test "the NV device's kernels compile to a cubin" (fun () ->
          compiled
            (compiling (target "NV" "CUDA" "sm_89"))
            (starts_with ~prefix:"\x7fELF"));
    ]

(* bfloat16 truncation on Metal *)

(* Metal has no trunc of a bfloat: it truncates a float, so the extra matcher
   truncates a bfloat16 in float32. *)
let truncated_bf16 =
  let zero = Ops.int ~dtype:Int32 0 in
  let at slot = Ops.index (Shape.param ~shape:[ Int 1 ] slot Bfloat16) [ zero ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"trunc_bf16" ())
    [ Ops.store (at 0) (Ops.trunc (Ops.load (at 1) [])) ]

let through_metal_matcher sink =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() sink
    (After_sources (metal ()).extra_matcher)

let truncs sink =
  List.filter
    (fun u -> Op.equal (Ops.op u) Trunc)
    (Ops.toposort ~calls:Enter sink)

let truncates_in_float () =
  let rewritten = truncs (through_metal_matcher truncated_bf16) in
  equal (list Dtypes.dtype) [ Dtype.Float32 ] (List.map Ops.dtype rewritten)

(* The graph a kernel reaches the renderer as, after the rewrites that commit
   its constants. *)
let compiles_its_truncation () =
  let sink =
    List.fold_left
      (fun sink m ->
        Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() sink
          (After_sources m))
      (through_metal_matcher truncated_bf16)
      [
        Uop_weak.pm_commit_weak; Uop_weak.pm_lower_weak; Uop_weak.pm_cast_const;
      ]
  in
  let metal = compiling metal_target in
  match
    Renderer.Compiler.compile metal.compiler
      (render metal (Linearizer.linearize sink))
  with
  | _ -> ()
  | exception Renderer.Compiler.Compile_error why when lacks_toolchain why ->
      skip ~reason:why ()

let bf16_truncation =
  group "bfloat16 truncation on Metal"
    [
      test "truncates a bfloat16 in float32" truncates_in_float;
      test "leaves it to CUDA, which truncates a bfloat16 with htrunc"
        (fun () ->
          let rewritten =
            truncs
              (Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
                 truncated_bf16 (After_sources (cuda ()).extra_matcher))
          in
          equal (list Dtypes.dtype) [ Dtype.Bfloat16 ]
            (List.map Ops.dtype rewritten));
      slow
      @@ test "compiles a kernel that truncates a bfloat16"
           compiles_its_truncation;
    ]

(* bfloat16 on AMD

   LLVM folds a float32 converted from an integer and narrowed to bfloat16 into
   one conversion from the integer, which AMD's backend cannot select for an
   [i1], and the compiler aborts. These kernels hold no conversion to or from
   bfloat16 for LLVM to fold, and their values are IEEE's. *)

let hip_for arch = renderer_for (target "AMD" "HIP" arch)

let through_hip ?(arch = "gfx1201") sink =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() sink
    (After_sources (hip_for arch).extra_matcher)

(* The kernel storing in slot 0, of [to_], [f] of each element of slot 1, of
   [from]. *)
let converting from to_ n f =
  let r = Ops.range ~axis_type:Weak (Int n) [ 0 ] in
  let at slot dt = Ops.index (Shape.param ~shape:[ Int n ] slot dt) [ r ] in
  let v = f (Ops.load (at 1 from) []) in
  Ops.sink [ Ops.end_ (Ops.store (at 0 to_) v) [ r ] ]

let directly x = Ops.cast x Bfloat16
let through_float32 x = directly (Ops.cast x Float32)
let through_narrow x = directly (Decomp_dtype.narrow x Float32)
let as_float32 x = Ops.cast x Float32
let as_int32 x = Ops.cast x Int32

let bf16_conversions sink =
  let is_bf16 u = Dtype.equal (Ops.dtype u) Bfloat16 in
  List.filter
    (fun u -> Op.equal (Ops.op u) Cast && (is_bf16 u || is_bf16 (Ops.nth u 0)))
    (Ops.toposort ~calls:Enter sink)

let conversions =
  [
    ("a bool directly", Dtype.Bool, Dtype.Bfloat16, directly);
    ("a bool through a float32", Bool, Bfloat16, through_float32);
    ("a uint8 through a float32", Uint8, Bfloat16, through_float32);
    ("an int32 through narrow", Int32, Bfloat16, through_narrow);
    ("a float32", Float32, Bfloat16, directly);
    ("a bfloat16 to float32", Bfloat16, Float32, as_float32);
    ("a bfloat16 to int32", Bfloat16, Int32, as_int32);
  ]

let on_bits =
  List.concat_map
    (fun arch -> List.map (fun c -> (arch, c)) conversions)
    [ "gfx1100"; "gfx1201"; "gfx942"; "gfx950" ]

let leaves_no_conversion (arch, (_, from, to_, f)) =
  let rewritten = through_hip ~arch (converting from to_ 2 f) in
  equal ~msg:"conversions of bfloat16" int 0
    (List.length (bf16_conversions rewritten))

let pow2 k = Bigint.shift_left Bigint.one k
let bf16_precision = 8

(* The significant bits of the positive [m]. *)
let bit_length m =
  let rec at n = if Bigint.compare (pow2 n) m > 0 then n else at (n + 1) in
  at 0

(* The integer [v] rounded once to bfloat16: to [bf16_precision] significant
   bits, to nearest with ties to even. No integer reaches its greatest finite
   value. *)
let exact_bf16 v =
  let m = Bigint.abs v in
  let s = max 0 (bit_length m - bf16_precision) in
  let unit = pow2 s in
  let q = Bigint.div m unit and r = Bigint.rem m unit in
  let twice = Bigint.compare (Bigint.add r r) unit in
  let odd = not (Bigint.divisible q (Bigint.of_int 2)) in
  let q = if twice > 0 || (twice = 0 && odd) then Bigint.succ q else q in
  let x = Float.ldexp (Bigint.to_float q) s in
  if Bigint.sign v < 0 then -.x else x

(* An integer type's bounds and their neighbours, and at each exponent
   bfloat16's ties of both signs, with their neighbours: past 24 bits a float32
   rounds a neighbour to the tie, which a cast that rounds twice gets wrong. *)
let bf16_inputs src =
  let w = Dtype.bitsize src and signed = Dtype.mem src Dtype.sints in
  let top = if signed then w - 1 else w in
  let lo = if signed then Bigint.neg (pow2 top) else Bigint.zero in
  let hi = pow2 top in
  let around v = [ Bigint.pred v; v; Bigint.succ v ] in
  let ties e =
    let tie = Bigint.add (pow2 e) (pow2 (e - bf16_precision)) in
    around tie @ around (Bigint.add tie (pow2 (e - bf16_precision + 1)))
  in
  let positive =
    List.concat_map ties
      (List.init (max 0 (top - bf16_precision)) (( + ) bf16_precision))
  in
  around lo @ around (Bigint.pred hi)
  @ [ Bigint.zero; Bigint.one; Bigint.neg Bigint.one ]
  @ positive @ List.map Bigint.neg positive
  |> List.filter (fun v -> Bigint.compare v lo >= 0 && Bigint.compare v hi < 0)

(* What slot 0 of [k] holds, given slot 1's [inputs]. *)
let written_by k inputs =
  Interpreter.writes ~buffers:[ (1, Array.of_list inputs) ] k
  |> List.filter_map (fun (slot, _, v) -> if slot = 0 then Some v else None)

let integers_round_once src =
  let inputs = bf16_inputs src in
  let elements = List.map (fun v -> `Int v) inputs in
  let want = List.map (fun v -> `Float (exact_bf16 v)) inputs in
  List.iter
    (fun (name, route) ->
      let k = converting src Bfloat16 (List.length inputs) route in
      equal ~msg:name (list Dtypes.value) want
        (written_by (through_hip k) elements))
    [ ("directly", directly); ("through narrow", through_narrow) ]

let bools_are_exact () =
  List.iter
    (fun (name, route) ->
      let k = converting Bool Bfloat16 2 route in
      equal ~msg:name (list Dtypes.value)
        [ `Float 0.; `Float 1. ]
        (written_by (through_hip k) [ `Bool false; `Bool true ]))
    [ ("directly", directly); ("through a float32", through_float32) ]

let f32_of_bits b = Dtype.bitcast Uint32 Float32 (`Int (Bigint.of_int b))
let bf16_of_bits b = Dtype.bitcast Uint16 Bfloat16 (`Int (Bigint.of_int b))

let is_nan_value = function `Float x -> Float.is_nan x | _ -> false

let nan_or_value v =
  if is_nan_value v then "NaN"
  else Format.asprintf "%a" (Testable.pp Dtypes.value) v

(* Two values agree bit for bit, or are both NaN, whose bits nothing fixes. *)
let same_float dt want got =
  if is_nan_value want then equal string "NaN" (nan_or_value got)
  else
    let bits = Dtype.bitcast dt (if dt = Float32 then Uint32 else Uint16) in
    equal Dtypes.value (bits want) (bits got)

(* A float32 with bits [b] narrowed: bfloat16 is float32's top 16 bits, so the
   low 16 bits round away, to nearest with ties to even. A NaN stays a NaN. *)
let narrowed_bits b =
  let q = b lsr 16 and low = b land 0xffff in
  if low > 0x8000 || (low = 0x8000 && q land 1 = 1) then q + 1 else q

let float32_codes =
  [
    0; 0x80000000; 1; 0x7fff; 0x8000; 0x8001; 0x18000; 0x28000; 0x3f808000;
    0x3f818000; 0x3f808001; 0xbf808000; 0xbf818000; 0x7f7f7fff; 0x7f7f8000;
    0x7f7fffff; 0x7f800000; 0xff800000; 0x7f800001; 0x7fc00000; 0xffc00001;
  ]

let narrows_float32_codes () =
  let n = List.length float32_codes in
  let k = through_hip (converting Float32 Bfloat16 n directly) in
  let elements = List.map f32_of_bits float32_codes in
  List.iter2
    (fun b got ->
      let exp_ones = b land 0x7f800000 = 0x7f800000 in
      let nan = exp_ones && b land 0x7fffff <> 0 in
      let want =
        if nan then `Float Float.nan else bf16_of_bits (narrowed_bits b)
      in
      same_float Bfloat16 want got)
    float32_codes (written_by k elements)

let bf16_codes =
  [ 0; 0x8000; 1; 0x7f; 0x80; 0x3f80; 0xbf81; 0x7f7f; 0x7f80; 0xff80; 0x7fc0 ]

(* Widening puts a bfloat16's bits at the top of a float32's, which is the same
   value. *)
let widens_bf16_codes () =
  let n = List.length bf16_codes in
  let k = through_hip (converting Bfloat16 Float32 n as_float32) in
  let elements = List.map bf16_of_bits bf16_codes in
  List.iter2
    (fun c got -> same_float Float32 (f32_of_bits (c lsl 16)) got)
    bf16_codes (written_by k elements)

let bfloat16_on_amd =
  group "bfloat16 on AMD"
    [
      cases "leave LLVM no conversion to fold"
        ~id:(fun (arch, (name, _, _, _)) -> arch ^ " " ^ name)
        on_bits leaves_no_conversion;
      cases "an integer rounds once"
        ~id:(Format.asprintf "%a" Dtype.pp)
        Dtype.[ Int8; Uint8; Int16; Uint16; Int32; Uint32; Int64; Uint64 ]
        integers_round_once;
      test "a bool is exact" bools_are_exact;
      test "a float32 narrows to its nearest code" narrows_float32_codes;
      test "a bfloat16 widens to its value" widens_bf16_codes;
    ]

(* Execution *)

(* The host's renderer, which [in_pieces] replaces for its test. *)
let host = ref (lazy (Cstyle.clang host_target))
let loaded name = Run.program (Lazy.force !host) (kernel (find_case name))
let floats xs = Array.map (fun x -> `Float x) xs
let ints xs = Array.map (fun n -> `Int (Bigint.of_int n)) xs
let slot s outputs = List.assoc s outputs
let values = array Dtypes.value

let adds_two_buffers () =
  let a = Array.init 64 Float.of_int
  and b = Array.init 64 (fun i -> Float.of_int (100 - i)) in
  let out = Run.on_host (loaded "clang_add") [ (1, floats a); (2, floats b) ] in
  equal values (floats (Array.make 64 100.)) (slot 0 out)

let sums_a_buffer () =
  let out =
    Run.on_host (loaded "clang_sum")
      [ (1, floats (Array.init 256 Float.of_int)) ]
  in
  equal values (floats [| 32640. |]) (slot 0 out)

(* The operand of the maximum is the int one above the least, which the source
   writes as a literal. *)
let takes_the_maximum_of_a_literal () =
  let k = loaded "clang_inline_const_alu" in
  let max_of x = slot 0 (Run.on_host k [ (1, ints [| x |]) ]) in
  equal values (ints [| 1 |]) (max_of 1);
  equal values (ints [| -0x7fffffff |]) (max_of (-0x80000000))

let stores_where_its_gate_holds () =
  let out =
    Run.on_host
      (loaded "clang_gated_store_in_loop")
      [ (0, ints (Array.make 16 (-1))) ]
  in
  equal values
    (ints (Array.init 16 (fun i -> if i < 8 then i else -1)))
    (slot 0 out)

let loads_zero_outside_its_padding () =
  let x = Array.init 14 (fun i -> Float.of_int (i + 1)) in
  let out = Run.on_host (loaded "clang_padded") [ (1, floats x) ] in
  let expected =
    Array.init 16 (fun i -> if i = 0 || i = 15 then 1. else x.(i - 1) +. 1.)
  in
  equal values (floats expected) (slot 0 out)

let loops_until_its_test_fails () =
  let k = loaded "clang_unbounded_loop" in
  let after start = slot 0 (Run.on_host k [ (0, ints [| start |]) ]) in
  equal values ~msg:"from 0" (ints [| 10 |]) (after 0);
  equal values ~msg:"from 12, the body runs once" (ints [| 13 |]) (after 12)

let reads_a_constant_table () =
  let out = Run.on_host (loaded "clang_table") [] in
  equal values (ints [| 0; 127; 128; 255 |]) (slot 0 out)

let passes_variables_of_32_and_64_bits () =
  let out =
    Run.on_host
      ~vars:[ ("start", 2); ("offset", 1 lsl 40) ]
      (loaded "clang_scalar_params")
      []
  in
  equal values (ints (Array.init 4 (fun i -> i + 2 + (1 lsl 40)))) (slot 0 out)

(* 300 steps of [chain], 600 brackets deep if written as one expression. The
   operands keep every step below 2^27, so int32's [<<] and [%] are OCaml's. *)
let computes_a_long_chain () =
  let n = 300 and x = 123457 and shift = 7 and modulus = 1000003 in
  let step r _ = (r lsl shift) mod modulus in
  let expected = List.fold_left step x (List.init n Fun.id) in
  let k = Run.program (Lazy.force !host) (chain n) in
  let out =
    Run.on_host k
      [ (1, ints [| x |]); (2, ints [| shift |]); (3, ints [| modulus |]) ]
  in
  equal values (ints [| expected |]) (slot 0 out)

(* Long kernels *)

(* A loop of [n] operations on each element, an addition of a value loaded
   before the loop and an exclusive or with it in turn, so that a piece of the
   loop reads that value from the frame on every trip. Each operation's value
   is read once: the source writes the chain as one statement per 60 or so
   operations. *)
let looped_chain n =
  let zero = Ops.int ~dtype:Int32 0 in
  let c =
    Ops.load (Ops.index (Shape.param ~shape:[ Int 1 ] 2 Int32) [ zero ]) []
  in
  let trips = Ops.v Cast ~src:[ Ops.int ~dtype:Int32 4 ] ~arg:(Dtype Int32) in
  let r = Ops.range ~axis_type:Loop ~dtype:Int32 (Sym trips) [ 0 ] in
  let at slot = Ops.index (Shape.param ~shape:[ Int 4 ] slot Int32) [ r ] in
  let step x i = Ops.alu x (if i mod 2 = 0 then Add else Xor) [ c ] in
  let out = List.fold_left step (Ops.load (at 1) []) (List.init n Fun.id) in
  Linearizer.linearize
    (Ops.sink
       ~kernel:(Ops.kernel_info ~name:"looped_chain" ())
       [ Ops.end_ (Ops.store (at 0) out) [ r ] ])

(* [f ()] with every host kernel written as functions of one statement each, so
   that every value crosses to the statements that read it. *)
let in_pieces f () =
  let whole = !host in
  host := lazy (Cstyle.clang_in_pieces ~piece:1 host_target);
  Fun.protect ~finally:(fun () -> host := whole) f

(* What [looped_chain n] writes for [x] and [c]. *)
let chain_of n c x =
  List.fold_left
    (fun x i -> if i mod 2 = 0 then x + c else x lxor c)
    x (List.init n Fun.id)

let computes_its_chain n program =
  let c = 3 and xs = [| 10; -2; 0; 7 |] in
  let k = Run.program (Lazy.force !host) program in
  let out = Run.on_host k [ (1, ints xs); (2, ints [| c |]) ] in
  equal values (ints (Array.map (chain_of n c) xs)) (slot 0 out)

(* Past the length from which a kernel is cut, linearized once for the tests
   that read it. *)
let long_chain = 20_500
let long_program = lazy (looped_chain long_chain)

let computes_a_long_chain_of_values_read_once () =
  computes_its_chain long_chain (Lazy.force long_program)

let reads_a_value_of_the_outer_scope () =
  computes_its_chain 200 (looped_chain 200)

(* Two loops one after the other whose indices render as one name, each
   scoped to its loop: the first writes [2 * (3 * i + 100)] through a value it
   reads twice, so that the value is declared, the second [i + 1]. *)
let sibling_loops () =
  let trips n = Ops.v Cast ~src:[ Ops.int ~dtype:Int32 n ] ~arg:(Dtype Int32) in
  let loop n slot value =
    let r = Ops.range ~axis_type:Loop ~dtype:Int32 (Sym (trips n)) [ 0 ] in
    let at = Ops.index (Shape.param ~shape:[ Int 4 ] slot Int32) [ r ] in
    Ops.end_ (Ops.store at (value r)) [ r ]
  in
  let first r =
    let v =
      Ops.alu (Ops.alu r Mul [ Ops.int ~dtype:Int32 3 ]) Add
        [ Ops.int ~dtype:Int32 100 ]
    in
    Ops.alu v Add [ v ]
  in
  let second r = Ops.alu r Add [ Ops.int ~dtype:Int32 1 ] in
  Linearizer.linearize
    (Ops.sink
       ~kernel:(Ops.kernel_info ~name:"sibling_loops" ())
       [ loop 4 0 first; loop 3 1 second ])

let keeps_two_loops_of_one_index_name =
  in_pieces @@ fun () ->
  let k = Run.program (Lazy.force !host) (sibling_loops ()) in
  let out =
    Run.on_host k [ (0, ints [| 0; 0; 0; 0 |]); (1, ints [| 9; 9; 9; 9 |]) ]
  in
  equal values ~msg:"first loop" (ints [| 200; 206; 212; 218 |]) (slot 0 out);
  equal values ~msg:"second loop" (ints [| 1; 2; 3; 9 |]) (slot 1 out)

(* A loop that reads a value loaded before it only at its start, then declares
   values of its type: [2 * (2 * (x + c))^2], each value read twice so that it
   is declared. The value read at the start is read again on the next trip. *)
let early_read () =
  let zero = Ops.int ~dtype:Int32 0 in
  let c =
    Ops.load (Ops.index (Shape.param ~shape:[ Int 1 ] 2 Int32) [ zero ]) []
  in
  let trips = Ops.v Cast ~src:[ Ops.int ~dtype:Int32 4 ] ~arg:(Dtype Int32) in
  let r = Ops.range ~axis_type:Loop ~dtype:Int32 (Sym trips) [ 0 ] in
  let at slot = Ops.index (Shape.param ~shape:[ Int 4 ] slot Int32) [ r ] in
  let twice op x = Ops.alu x op [ x ] in
  let v = Ops.alu (Ops.load (at 1) []) Add [ c ] in
  let out = twice Add (twice Mul (twice Add v)) in
  Linearizer.linearize
    (Ops.sink
       ~kernel:(Ops.kernel_info ~name:"early_read" ())
       [ Ops.end_ (Ops.store (at 0) out) [ r ] ])

let keeps_a_value_read_early_in_a_loop () =
  let xs = [| 10; -2; 0; 7 |] and c = 3 in
  let k = Run.program (Lazy.force !host) (early_read ()) in
  let out = Run.on_host k [ (1, ints xs); (2, ints [| c |]) ] in
  let expected x = 2 * (2 * (x + c)) * (2 * (x + c)) in
  equal values (ints (Array.map expected xs)) (slot 0 out)

let cut source = contains source "looped_chain_frame"

let a_long_kernel_is_written_in_pieces () =
  let source program = render (renderer_for clang_target) program in
  equal ~msg:"past the length" Testable.bool true
    (cut (source (Lazy.force long_program)));
  equal ~msg:"below it" Testable.bool false (cut (source (looped_chain 200)))

let a_gpu_kernel_stays_one_function () =
  List.iter
    (fun (r, renderer) ->
      equal ~msg:r Testable.bool false
        (cut (render (renderer ()) (Lazy.force long_program))))
    (List.filter (fun (n, _) -> n <> "clang") renderers)

let runs_custom_code () =
  let x = [| -2.5; 0.; 3.; -0. |] in
  let out = Run.on_host (loaded "clang_custom") [ (1, floats x) ] in
  equal values (floats (Array.map Float.abs x)) (slot 0 out)

let accesses_volatile_buffers () =
  let out =
    Run.on_host (loaded "clang_volatile") [ (1, ints [| 1; 2; 3; -4 |]) ]
  in
  equal values (ints [| 2; 3; 4; -3 |]) (slot 0 out)

(* Each constant is the value its type holds for the literal: -1 is the greatest
   unsigned integer, and 3.14 is rounded once to its float. *)
let stores_each_constant_as_its_type_holds_it () =
  let literals : (Dtype.t * Dtype.value) list =
    [
      (Bool, `Bool true);
      (Bool, `Bool false);
      (Int8, `Int (Bigint.of_int (-3)));
      (Uint8, `Int (Bigint.of_int 200));
      (Int16, `Int (Bigint.of_int (-300)));
      (Uint16, `Int (Bigint.of_int 60000));
      (Int32, `Int (Bigint.of_int 42));
      (Uint32, `Int (Bigint.of_int 42));
      (Uint32, `Int Bigint.minus_one);
      (Int64, `Int (Bigint.of_int 12345));
      (Uint64, `Int (Bigint.of_int 42));
      (Uint64, `Int Bigint.minus_one);
      (Float16, `Float 1.5);
      (Bfloat16, `Float 1.5);
      (Float32, `Float 3.14);
      (Float64, `Float 3.14);
    ]
  in
  let out = Run.on_host (loaded "clang_constants") [] in
  List.iteri
    (fun slot (dt, v) ->
      equal values
        ~msg:(Format.asprintf "slot %d, %a" slot Dtype.pp dt)
        [| Dtype.truncate dt v |]
        (List.assoc slot out))
    literals

(* Integer constants *)

(* [held dt v] is the integer [dt] holds for the constant [v]: its integer part,
   rounded toward zero, wrapped to [dt]'s width. *)
let held dt (v : Dtype.value) : Dtype.value =
  let n =
    match v with
    | `Int n -> n
    | `Float x -> Bigint.of_float x
    | `Bool b -> Bigint.of_int (Bool.to_int b)
  in
  let bits = 8 * Dtype.itemsize dt in
  let modulus = Bigint.shift_left Bigint.one bits in
  let r = Bigint.(n - (modulus * fdiv n modulus)) in
  if
    Dtype.is_unsigned dt
    || Bigint.lt r (Bigint.shift_left Bigint.one (bits - 1))
  then `Int r
  else `Int (Bigint.sub r modulus)

(* Each constant stored twice: as the constant of [dt] ([Ops.const]), and as a
   conversion of its own literal to [dt]. *)
let stores_constants dt (cs : Dtype.value list) =
  let n = 2 * List.length cs in
  let at i =
    Ops.index (Shape.param ~shape:[ Int n ] 0 dt) [ Ops.int ~dtype:Int32 i ]
  in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"constants" ())
    (List.concat
       (List.mapi
          (fun i c ->
            let c = (c :> Dtype.const) in
            [
              Ops.store (at (2 * i)) (Ops.const ~dtype:dt c);
              Ops.store (at ((2 * i) + 1)) (Ops.cast (Ops.const c) dt);
            ])
          cs))

(* The extremes of every integer type and their neighbours, integers beyond 64
   bits, and floats past the integer types' ranges. *)
let int_constant =
  let open Gen in
  let ints = List.map (fun n -> `Int n) in
  let extremes =
    List.concat_map
      (fun dt ->
        let lo = Dtype.Value.to_z (Dtype.min dt)
        and hi = Dtype.Value.to_z (Dtype.max dt) in
        ints Bigint.[ pred lo; lo; succ lo; pred hi; hi; succ hi ])
      Dtype.ints
  in
  let power k = Bigint.shift_left Bigint.one k in
  let beyond =
    map
      (fun (k, sign) -> `Int (Bigint.mul (Bigint.of_int sign) (power k)))
      (pair (int_range 60 130) (of_list [ 1; -1 ]))
  in
  (* [1 + k 2^-52] is below 2, so the value is finite even at [e = 1023]: an
     infinity has no integer value. *)
  let large_float =
    map
      (fun ((e, k), sign) ->
        `Float (sign *. Float.ldexp (1. +. Float.ldexp (Float.of_int k) (-52)) e))
      (pair
         (pair (int_range 7 1023) (int_range 0 ((1 lsl 52) - 1)))
         (of_list [ 1.; -1. ]))
  in
  let float_edges =
    of_list
      (List.concat_map
         (fun x -> [ `Float x; `Float (Float.pred x); `Float (Float.succ x) ])
         [ 0x1p31; -0x1p31; 0x1p32; 0x1p63; -0x1p63; 0x1p64; 1e20; -1e20 ])
  in
  frequency
    [
      (3, of_list extremes);
      (2, beyond);
      (3, large_float);
      (2, float_edges);
      (1, map (fun n -> `Int (Bigint.of_int n)) (int_range (-1000) 1000));
    ]

let int_dtypes = Gen.of_list ~pp:Dtype.pp Dtype.ints

let pp_constants ppf (dt, cs) =
  Format.fprintf ppf "%a [%a]" Dtype.pp dt
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Dtype.pp_const)
    (cs :> Dtype.const list)

let constants_of_ints =
  Gen.with_pp pp_constants
    (Gen.pair int_dtypes (Gen.list ~size:(Gen.int_range 1 8) int_constant))

let stores_ints_as_their_type_holds_them (dt, cs) =
  let beyond_64_bits = function
    | `Int n -> (
        match Bigint.to_int64 n with
        | _ -> false
        | exception Bigint.Overflow -> true)
    | `Float x -> Float.abs x >= 0x1p64
    | `Bool _ -> false
  in
  cover "a constant beyond 64 bits" (List.exists beyond_64_bits cs);
  cover "a float" (List.exists (function `Float _ -> true | _ -> false) cs);
  cover "a 64-bit type" (Dtype.itemsize dt = 8);
  cover "an unsigned type" (Dtype.is_unsigned dt);
  let k =
    Run.program (Lazy.force !host)
      (Linearizer.linearize (stores_constants dt cs))
  in
  let expected = List.concat_map (fun c -> [ held dt c; held dt c ]) cs in
  equal values (Array.of_list expected) (slot 0 (Run.on_host k []))

(* An infinity or a NaN has no integer value: its conversion to an integer type
   is the program's to run, and the source compiles. *)
let converts_a_special_to_an_int (dt, x) =
  let at =
    Ops.index (Shape.param ~shape:[ Int 1 ] 0 dt) [ Ops.int ~dtype:Int32 0 ]
  in
  let sink =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"special" ())
      [ Ops.store at (Ops.cast (Ops.float x) dt) ]
  in
  ignore (Run.program (Lazy.force !host) (Linearizer.linearize sink))

(* The kernel reads the four chars 1, 2, 3 and 4 as one little-endian uint,
   clears its low byte and stores it to the first char. *)
let reads_chars_as_a_uint name () =
  let out = Run.on_host (loaded name) [ (0, ints [| 1; 2; 3; 4 |]) ] in
  equal values (ints [| 0; 2; 3; 4 |]) (slot 0 out)

(* The registers hold the uints 1 and 2, read as one little-endian ulong. *)
let reads_registers_as_a_ulong () =
  let out = Run.on_host (loaded "clang_register_cast") [] in
  equal values [| `Int (Bigint.of_string "0x200000001") |] (slot 0 out)

let picks_a_lane_by_a_variable () =
  let k = loaded "clang_dynamic_lane" in
  List.iter
    (fun lane ->
      equal values ~msg:(string_of_int lane)
        (floats [| Float.of_int (lane + 1) |])
        (slot 0 (Run.on_host ~vars:[ ("lane", lane) ] k [])))
    [ 0; 1; 2; 3 ]

let passes_named_parameters () =
  let out =
    Run.on_host
      ~vars:[ ("for", 5) ]
      (loaded "clang_named_params")
      [ (0, floats [| 1.; 2.; 3.; 4. |]) ]
  in
  equal values (floats [| 6.; 7.; 8.; 9. |]) (slot 1 out)

(* The kernel converts x to the narrow type, adds y and multiplies by 3, each
   operation giving a value of the narrow type, then converts to float. *)
let rounds_each_operation_on_halves () =
  let half x =
    match Dtype.truncate Float16 (`Float x) with
    | `Float h -> h
    | _ -> invalid_arg "a half is a float"
  in
  let x = [| 1.; 0.1; 65504.; -3. |] and y = [| 2.; 1.; 1.; 0.5 |] in
  let x = Array.concat [ x; x; x; x ] and y = Array.concat [ y; y; y; y ] in
  let out =
    Run.on_host
      (loaded "clang_dtype_f16")
      [ (1, floats x); (2, floats (Array.map half y)) ]
  in
  let expected i = half (half (half x.(i) +. y.(i)) *. 3.) in
  equal values (floats (Array.init 16 expected)) (slot 0 out)

let wraps_each_operation_on_chars () =
  let out =
    Run.on_host
      (loaded "clang_dtype_u8")
      [ (1, floats (Array.make 16 16.)); (2, ints (Array.make 16 86)) ]
  in
  equal values
    (floats (Array.make 16 (Float.of_int ((16 + 86) * 3 land 0xff))))
    (slot 0 out)

(* Laws *)

(* The interpreter computes the writes of a kernel graph, so it runs every Clang
   case but these, each with its reason. *)
let not_interpreted =
  [
    ("clang_call_out", "calls an external function");
    ("clang_call_ret", "calls an external function");
    ("clang_call_stack", "calls an external function");
    ("clang_unbounded_loop", "has a loop without a trip count");
    ("clang_sum", "accumulates in registers");
    ("clang_matmul", "accumulates in registers");
    ("clang_matmul_upcasted", "accumulates in registers");
    ("clang_add_max_uchar", "accumulates in registers");
    ("clang_padded", "reads outside its buffer in the branch its gate discards");
    ("clang_gated_store_in_loop", "gates its store with an if");
    ("clang_custom", "runs custom code");
    ("clang_table", "reads a constant table");
    ("clang_transcendental_f16", "divides with Ops.FDIV");
    ("clang_transcendental_bf16", "divides with Ops.FDIV");
    ("clang_transcendental_f32", "divides with Ops.FDIV");
    ("clang_transcendental_f64", "divides with Ops.FDIV");
    ("clang_packed_cast", "reads memory at another type");
    ("clang_packed_bitcast", "reads memory at another type");
    ("clang_dynamic_lane", "indexes a vector value");
    ("clang_vector_cast", "stores a vector value");
    ("clang_register_cast", "reads registers at another type");
  ]

let clang_rows =
  List.filter
    (fun row -> row "renderer" = "CLANG" && row "setting" = "-")
    case_rows

let interpreted_rows =
  List.filter
    (fun row -> not (List.mem_assoc (row "case") not_interpreted))
    clang_rows

(* An element on which a kernel's arithmetic is exact and defined: a float is a
   multiple of a half below 16, so that sums and products round nowhere and
   convert to every integer type, and an integer is small and never zero, since
   C leaves a division by zero and a signed overflow undefined. *)
let element dt =
  let open Gen in
  if Dtype.is_float dt then
    map (fun n -> `Float (Float.of_int n /. 2.)) (int_range 0 32)
  else if Dtype.is_bool dt then map (fun b -> `Bool b) bool
  else
    let low = if Dtype.is_unsigned dt then 1 else -100 in
    map
      (fun n -> `Int (Bigint.of_int (if n = 0 then 1 else n)))
      (int_range low 100)

let buffers_of uops =
  List.filter_map
    (fun u ->
      match Ops.arg u with
      | Ops.Param ({ addrspace = Global; _ } as p)
        when Shape.max_shape u <> [] ->
          Some (p.slot, p.dtype, Shape.max_numel u)
      | _ -> None)
    uops

(* The loop a kernel's launch splits into blocks, if any. The launch binds the
   variables of the block's bounds itself, so a run computes the whole loop. *)
let split_of uops =
  match Ops.arg (List.nth uops (List.length uops - 1)) with
  | Ops.Kernel k -> k.split
  | _ -> None

let bounds_a_block uops slot =
  match split_of uops with
  | Some s -> slot = s.lo || slot = s.hi
  | None -> false

(* The variables a run binds: those of the kernel, the bounds of a block left
   out, and those that a split loop's iterations read, which the kernel may no
   longer read. *)
let variables_of uops =
  let iterations =
    match split_of uops with
    | Some { iterations = Sym u; _ } -> Ops.toposort ~calls:Enter u
    | _ -> []
  in
  let variable u =
    match Ops.arg u with
    | Ops.Param
        {
          addrspace = Alu;
          name = Some name;
          vmin_vmax = Some (`Int lo, `Int hi);
          slot;
          _;
        }
      when not (bounds_a_block uops slot) ->
        Some (name, Bigint.to_int lo, Bigint.to_int hi)
    | _ -> None
  in
  List.sort_uniq compare (List.filter_map variable (uops @ iterations))

let pp_inputs ppf (buffers, vars) =
  List.iter (fun (name, v) -> Format.fprintf ppf "%s=%d@ " name v) vars;
  List.iter
    (fun (s, a) ->
      Format.fprintf ppf "@[slot %d:%a@]@ " s
        (fun ppf -> Array.iter (Format.fprintf ppf " %a" Dtype.pp_const))
        a)
    buffers

let draw_inputs uops =
  let cons g rest = Gen.bind g (fun x -> Gen.map (fun r -> x :: r) rest) in
  let buffers =
    List.fold_right
      (fun (slot, dt, n) ->
        cons
          (Gen.map
             (fun a -> (slot, a))
             (Gen.array ~size:(Gen.constant n) (element dt))))
      (buffers_of uops) (Gen.constant [])
  and vars =
    List.fold_right
      (fun (name, lo, hi) ->
        cons (Gen.map (fun v -> (name, v)) (Gen.int_range lo hi)))
      (variables_of uops) (Gen.constant [])
  in
  Gen.with_pp pp_inputs (Gen.pair buffers vars)

(* The bounds of a split kernel's one block, the whole loop, given the values
   [vars] of its variables: block_lo is 0 and block_hi the loop's iterations. *)
let whole_loop uops vars =
  match split_of uops with
  | None -> []
  | Some s ->
      let name slot =
        Option.get
          (List.find_map
             (fun u ->
               match Ops.arg u with
               | Ops.Param p when p.slot = slot -> p.name
               | _ -> None)
             uops)
      in
      [ (name s.lo, 0); (name s.hi, Shape.sym_infer s.iterations vars) ]

let interpreted uops (buffers, vars) =
  let sink = List.nth uops (List.length uops - 1) in
  let vars =
    List.map
      (fun (name, v) -> (name, `Int (Bigint.of_int v)))
      (whole_loop uops vars @ vars)
  in
  let writes = Interpreter.writes ~vars ~buffers sink in
  List.map
    (fun (slot, _, _) ->
      let a = Array.copy (List.assoc slot buffers) in
      List.iter (fun (s, i, v) -> if s = slot then a.(i) <- v) writes;
      (slot, a))
    (buffers_of uops)

(* The symbols object [obj] calls and does not define. *)
let undefined_in obj =
  match Rig_elf.of_string obj with
  | Error why -> failf "%s" why
  | Ok o ->
      List.sort_uniq String.compare
        (List.filter_map
           (fun (r : Rig_elf.relocation) ->
             match r.symbol.place with
             | Undefined -> Some r.symbol.name
             | _ -> None)
           o.relocations)

let undefined p = undefined_in (Uops.binary_of p)

(* [alone] skips a kernel whose object calls a function of the process. *)
let agrees_with_the_interpreter ?(on = !host) ?(alone = false) row =
  let uops = kernel row in
  let k = lazy (Run.program (Lazy.force on) uops) in
  prop ~count:10 (row "case") (draw_inputs uops)
    (fun ((buffers, vars) as inputs) ->
      if alone && undefined (Lazy.force k) <> [] then
        skip ~reason:"the object calls the process" ();
      equal
        (list (pair int values))
        (interpreted uops inputs)
        (Run.on_host ~vars (Lazy.force k) buffers))

(* An element of a kernel over a narrow type, drawn over the whole type so that
   its operations wrap and round: an element of the narrow type is any of its
   values, bounds included, and a float the kernel converts to it is one the
   narrow type holds, since C leaves an integer conversion out of range
   undefined. *)
let narrow_element narrow dt =
  if Dtype.equal dt narrow || not (Dtype.is_float dt) then Dtypes.value_of dt
  else
    Gen.map
      (function `Int z -> `Float (Bigint.to_float z) | #Dtype.value as v -> v)
      (Dtypes.value_of narrow)

let narrow_types = Dtype.[ Int8; Uint8; Int16; Uint16; Float16 ]

(* The narrow type a kernel computes on: that of its first narrow load. *)
let narrow_of uops =
  List.find_map
    (fun u ->
      if Op.equal (Ops.op u) Load && List.mem (Ops.dtype u) narrow_types then
        Some (Ops.dtype u)
      else None)
    uops

let draw_narrow_inputs narrow uops =
  let cons g rest = Gen.bind g (fun x -> Gen.map (fun r -> x :: r) rest) in
  List.fold_right
    (fun (slot, dt, n) ->
      cons
        (Gen.map
           (fun a -> (slot, a))
           (Gen.array ~size:(Gen.constant n) (narrow_element narrow dt))))
    (buffers_of uops) (Gen.constant [])
  |> Gen.map (fun buffers -> (buffers, []))
  |> Gen.with_pp pp_inputs

let narrow_rows =
  List.filter
    (fun row -> Option.is_some (narrow_of (kernel row)))
    interpreted_rows

let wraps_and_rounds_as_the_interpreter row =
  let uops = kernel row in
  let narrow = Option.get (narrow_of uops) in
  let k = lazy (Run.program (Lazy.force !host) uops) in
  prop ~count:20 (row "case") (draw_narrow_inputs narrow uops)
    (fun ((buffers, _) as inputs) ->
      equal
        (list (pair int values))
        (interpreted uops inputs)
        (Run.on_host (Lazy.force k) buffers))

let compiles_and_loads ?(on = !host) row =
  test (row "case") (fun () -> ignore (Run.program (Lazy.force on) (kernel row)))

let execution =
  group "execution on the host"
    [
      test "adds two buffers" adds_two_buffers;
      test "sums a buffer" sums_a_buffer;
      test "takes the maximum of a load and a literal"
        takes_the_maximum_of_a_literal;
      test "stores only where its gate holds" stores_where_its_gate_holds;
      test "loads zero outside its padding" loads_zero_outside_its_padding;
      test "loops until its bottom test fails" loops_until_its_test_fails;
      test "reads a constant table" reads_a_constant_table;
      test "passes variables of 32 and 64 bits"
        passes_variables_of_32_and_64_bits;
      test "computes a chain too deep for one expression" computes_a_long_chain;
      test "runs custom code" runs_custom_code;
      test "accesses volatile buffers" accesses_volatile_buffers;
      test "stores each constant as its type holds it"
        stores_each_constant_as_its_type_holds_it;
      prop ~count:30 "stores each integer constant as its type holds it"
        constants_of_ints stores_ints_as_their_type_holds_them;
      cases "converts an infinity or a NaN to an integer type"
        ~id:(fun (dt, x) -> Format.asprintf "%a %h" Dtype.pp dt x)
        (List.concat_map
           (fun dt ->
             List.map (fun x -> (dt, x)) [ infinity; neg_infinity; nan ])
           Dtype.[ Int8; Uint16; Int32; Uint64 ])
        converts_a_special_to_an_int;
      test "reads four chars as a uint through a cast of their address"
        (reads_chars_as_a_uint "clang_packed_cast");
      test "reads four chars as a uint through a bitcast of their address"
        (reads_chars_as_a_uint "clang_packed_bitcast");
      test "picks a lane of a vector by a variable" picks_a_lane_by_a_variable;
      test "reads two uint registers as a ulong through a cast of their address"
        reads_registers_as_a_ulong;
      test "writes a long kernel in pieces" a_long_kernel_is_written_in_pieces;
      test "keeps two loops whose indices render as one name"
        keeps_two_loops_of_one_index_name;
      test "leaves a GPU's kernels whole" a_gpu_kernel_stays_one_function;
      test "computes a long chain of values read once"
        computes_a_long_chain_of_values_read_once;
      group "cut into pieces, writes what it writes whole"
        (List.map
           (fun (name, f) -> test name (in_pieces f))
           [
             ("reads a value of the outer scope on every trip",
               reads_a_value_of_the_outer_scope);
             ("keeps a value read early in a loop for its next trip",
               keeps_a_value_read_early_in_a_loop);
             ("adds two buffers", adds_two_buffers);
             ("sums a buffer", sums_a_buffer);
             ("takes the maximum of a literal", takes_the_maximum_of_a_literal);
             ("stores where its gate holds", stores_where_its_gate_holds);
             ("loads zero outside its padding", loads_zero_outside_its_padding);
             ("loops until its test fails", loops_until_its_test_fails);
             ("reads a constant table", reads_a_constant_table);
             ("passes variables", passes_variables_of_32_and_64_bits);
             ("computes a long chain", computes_a_long_chain);
             ("runs custom code", runs_custom_code);
             ("accesses volatile buffers", accesses_volatile_buffers);
             ("picks a lane by a variable", picks_a_lane_by_a_variable);
             ("reads registers as a ulong", reads_registers_as_a_ulong);
             ("passes named parameters", passes_named_parameters);
             ("rounds operations on halves", rounds_each_operation_on_halves);
             ("wraps operations on chars", wraps_each_operation_on_chars);
           ]);
      test "passes named parameters" passes_named_parameters;
      test "rounds each operation on halves to a half"
        rounds_each_operation_on_halves;
      test "wraps each operation on unsigned chars"
        wraps_each_operation_on_chars;
      slow
      @@ group "every kernel compiles and loads"
           (List.map compiles_and_loads clang_rows);
      slow
      @@ group "every kernel the interpreter runs writes what it computes"
           (List.map agrees_with_the_interpreter interpreted_rows);
      slow
      @@ group "a kernel over a narrow type wraps and rounds as the interpreter"
           (List.map wraps_and_rounds_as_the_interpreter narrow_rows);
    ]

(* Division

   A division, Ops.FDIV, is the language's [/] on every target, whether or not
   the target lists it among its operations. Metal's is IEEE's division, rounded
   once, which a product by the reciprocal is not for these operands. *)

let dividends = [| 3.; 5.; 7.; 10.; 1.; -6.; 0.; -1. |]
let divisors = [| 7.; 3.; 49.; 0.001; 0.; 3.; 0.; infinity |]
let float32 x = Dtype.Value.to_float (Dtype.truncate Float32 (`Float x))
let quotient i = float32 (float32 dividends.(i) /. float32 divisors.(i))

let product_by_reciprocal i =
  float32 (float32 dividends.(i) *. float32 (1. /. float32 divisors.(i)))

(* [out[i] = a[i] / b[i]] over the eight operands, out, a and b the parameters
   0, 1 and 2. *)
let division =
  let n = Array.length dividends in
  let i = Ops.range ~axis_type:Weak (Int n) [ 0 ] in
  let at slot = Ops.index (Shape.placeholder ~slot [ n ] Float32) [ i ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"fdiv" ())
    [ Ops.end_ (Ops.store (at 0) (Ops.alu (at 1) Fdiv [ at 2 ])) [ i ] ]

let divides_on_metal () =
  match Rig.open_ (Rig_metal.gpu 0) with
  | Error why -> skip ~reason:why ()
  | Ok m ->
      let devices = function "CPU:1" -> m | _ -> Rig.host in
      let n = Array.length dividends in
      let buffer () = Ops.new_buffer "CPU:1" n Float32 in
      let out = buffer () and a = buffer () and b = buffer () in
      let compiled =
        Realize.compile_linear
          ~targets:(fun d -> Tolk_engine.target (devices d))
          (Ops.v Op.Linear ~src:[ Ops.call division [ out; a; b ] ])
      in
      let on_metal xs = Run.buffer m Float32 (floats xs) in
      let quotients = on_metal (Array.make n 0.) in
      let s =
        Tolk_engine.link ~devices
          ~bound:
            [
              (out, quotients);
              (a, on_metal dividends);
              (b, on_metal divisors);
            ]
          compiled
      in
      Tolk_engine.run s [||];
      equal values
        (floats (Array.init n quotient))
        (Run.values Float32 quotients)

let division_group =
  group "division"
    [
      test "the operands tell a quotient from a product by the reciprocal"
        (fun () ->
          is_true
            (List.exists
               (fun i -> quotient i <> product_by_reciprocal i)
               (List.init (Array.length dividends) Fun.id)));
      slow @@ test "Metal divides as IEEE does, rounding once" divides_on_metal;
    ]

(* Tensor cores on Metal

   A warp's bfloat16 product, [D = A B], then [D + 0 C] with a left operand
   of literal zeros: Apple7's compiler service failed on a product of
   bfloat16 matrices with an operand the compiler knows, into float32 and
   into bfloat16 alike. *)

(* Metal's bfloat16 cores, into float32 and into bfloat16. *)
let bf16_cores =
  List.filter (fun (tc : Tc.t) -> Dtype.equal tc.dtype_in Bfloat16) Tc.metal

(* [m.(i).(j)], a small integer: each product's sums are integers below 256,
   exact in bfloat16. *)
let entry salt i j = Float.of_int ((((i * 3) + (j * 5) + salt) mod 7) - 3)

let zero_product (tc : Tc.t) =
  let lanes = Tc.threads tc in
  let lane = Ops.range ~axis_type:Warp (Int lanes) [ 0 ] in
  let buffer slot dt = Shape.placeholder ~slot [ 2 * lanes ] dt in
  let out = buffer 0 tc.dtype_out and a = buffer 1 Bfloat16 in
  let b = buffer 2 Bfloat16 and c = buffer 3 Bfloat16 in
  let at e = Ops.O.((lane * int 2) + int e) in
  let pair buf = Shape.stack (List.init 2 (fun e -> Ops.load (Ops.index buf [ at e ]) [])) in
  let literal dt = Shape.stack (List.init 2 (fun _ -> Ops.float ~dtype:dt 0.)) in
  let product x y acc = Ops.wmma x y ~acc ~dims:(Tc.dims tc) ~threads:lanes in
  let d = product (pair a) (pair b) (literal tc.dtype_out) in
  let d = product (literal Bfloat16) (pair c) d in
  let stores =
    List.init 2 (fun e ->
        Ops.store (Ops.index out [ at e ]) (Ops.index d [ Ops.int ~dtype:Int32 e ]))
  in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"zero_product" ~opts_to_apply:[] ())
    [ Ops.end_ (Ops.group stores) [ lane ] ]

let multiplies_by_zeros_on_metal (tc : Tc.t) =
  match Rig.open_ (Rig_metal.gpu 0) with
  | Error why -> skip ~reason:why ()
  | Ok m ->
      let devices = function "CPU:1" -> m | _ -> Rig.host in
      let fa, fb, fc = Tc.frag_coords tc in
      let lanes = Tc.threads tc in
      (* Each lane's two elements of a fragment, as its coordinates say. *)
      let held coords f =
        Array.init (2 * lanes) (fun i ->
            let r, col = coords.(i / 2).(i mod 2) in
            f r col)
      in
      let _, _, k = Tc.dims tc in
      let d r col =
        List.fold_left ( +. ) 0.
          (List.init k (fun j -> entry 0 r j *. entry 1 j col))
      in
      let buffer dt = Ops.new_buffer "CPU:1" (2 * lanes) dt in
      let out = buffer tc.dtype_out and a = buffer Bfloat16 in
      let b = buffer Bfloat16 and c = buffer Bfloat16 in
      let compiled =
        Realize.compile_linear
          ~targets:(fun d -> Tolk_engine.target (devices d))
          (Ops.v Op.Linear ~src:[ Ops.call (zero_product tc) [ out; a; b; c ] ])
      in
      let on_metal dt xs = Run.buffer m dt (floats xs) in
      let products = on_metal tc.dtype_out (Array.make (2 * lanes) 0.) in
      let s =
        Tolk_engine.link ~devices
          ~bound:
            [
              (out, products);
              (a, on_metal Bfloat16 (held fa (entry 0)));
              (b, on_metal Bfloat16 (held fb (entry 1)));
              (c, on_metal Bfloat16 (held fb (entry 2)));
            ]
          compiled
      in
      Tolk_engine.run s [||];
      equal values (floats (held fc d)) (Run.values tc.dtype_out products)

let metal_cores =
  group "tensor cores on Metal"
    [
      slow
      @@ cases
           "a bfloat16 product with an operand of literal zeros holds the \
            product"
           ~id:(fun (tc : Tc.t) -> Format.asprintf "into %a" Dtype.pp tc.dtype_out)
           bf16_cores multiplies_by_zeros_on_metal;
    ]

(* Negation *)

(* A kernel of one float lane: [data0[0] = f (data1[0], data2[0])]. *)
let lane f =
  let zero = Ops.int ~dtype:Int32 0 in
  let at slot = Ops.index (Shape.param ~shape:[ Int 1 ] slot Float32) [ zero ] in
  Linearizer.linearize
    (Ops.sink
       ~kernel:(Ops.kernel_info ~name:"lane" ())
       [ Ops.store (at 0) (f (Ops.load (at 1) []) (Ops.load (at 2) [])) ])

let minus x y = Ops.alu x Sub [ y ]
let negated x = Ops.alu x Neg []

(* A minus sign before an operand that starts with one would be the decrement
   operator [--]. *)
let negation =
  group "negation"
    [
      cases "a minus before a minus is apart"
        ~id:fst renderers
        (fun (_, r) ->
          let r = r () in
          let neg = List.assoc Op.Neg r.code_for_op
          and sub = List.assoc Op.Sub r.code_for_op in
          equal string "(a- -b)" (sub [ "a"; "-b" ] Float32);
          equal string "- -b" (neg [ "-b" ] Float32);
          equal string "(a-b)" (sub [ "a"; "b" ] Float32);
          equal string "-b" (neg [ "b" ] Float32));
      test "Clang compiles and runs a difference with a negated operand"
        (fun () ->
          let k =
            Run.program (Lazy.force !host)
              (lane (fun x y -> minus x (negated y)))
          in
          let out =
            Run.on_host k [ (1, [| `Float 1. |]); (2, [| `Float 2. |]) ]
          in
          equal values [| `Float 3. |] (List.assoc 0 out));
      test "Clang compiles and runs a difference with a negative constant"
        (fun () ->
          let k =
            Run.program (Lazy.force !host)
              (lane (fun x _ -> minus x (Ops.float ~dtype:Float32 (-1.5))))
          in
          let out = Run.on_host k [ (1, [| `Float 1. |]) ] in
          equal values [| `Float 2.5 |] (List.assoc 0 out));
    ]

(* Grouping *)

(* A product of a product keeps its grouping: at 4 * (2^126 * 0.25) the inner
   product is finite and so is the whole, where (4 * 2^126) * 0.25 overflows. *)
let grouping =
  group "grouping"
    [
      test "Clang computes a product of a product as it is grouped" (fun () ->
          let zero = Ops.int ~dtype:Int32 0 in
          let at slot =
            Ops.index (Shape.param ~shape:[ Int 1 ] slot Float32) [ zero ]
          in
          let load slot = Ops.load (at slot) [] in
          let mul x y = Ops.alu x Mul [ y ] in
          let k =
            Run.program (Lazy.force !host)
              (Linearizer.linearize
                 (Ops.sink
                    ~kernel:(Ops.kernel_info ~name:"grouping" ())
                    [ Ops.store (at 0) (mul (load 1) (mul (load 2) (load 3))) ]))
          in
          let big = Float.ldexp 1. 126 in
          let out =
            Run.on_host k
              [
                (1, [| `Float 4. |]);
                (2, [| `Float big |]);
                (3, [| `Float 0.25 |]);
              ]
          in
          equal values [| `Float big |] (List.assoc 0 out));
    ]

(* Zeros a comparison reads *)

(* [data0[0] = data1[0] < c ? data1[0] : c] at [dt]: a select of a value and a
   constant by their comparison, which a backend may lower to a minimum. *)
let const_select dt c =
  let zero = Ops.int ~dtype:Int32 0 in
  let at slot = Ops.index (Shape.param ~shape:[ Int 1 ] slot dt) [ zero ] in
  let x = Ops.load (at 1) [] in
  let c = Ops.float ~dtype:dt c in
  Linearizer.linearize
    (Ops.sink
       ~kernel:(Ops.kernel_info ~name:"const_select" ())
       [ Ops.store (at 0) (Ops.where (Ops.lt x c) x c) ])

(* A select of a value and a zero by a condition loaded from memory. *)
let zero_masked dt =
  let zero = Ops.int ~dtype:Int32 0 in
  let at slot dt = Ops.index (Shape.param ~shape:[ Int 1 ] slot dt) [ zero ] in
  let x = Ops.load (at 1 dt) [] and c = Ops.load (at 2 Bool) [] in
  Linearizer.linearize
    (Ops.sink
       ~kernel:(Ops.kernel_info ~name:"zero_masked" ())
       [ Ops.store (at 0 dt) (Ops.where c x (Ops.float ~dtype:dt 0.)) ])

let arm64 = Cstyle.clang (target "CPU" "CLANG" "arm64,generic")
let riscv64 = Cstyle.clang (target "CPU" "CLANG" "riscv64,rv64g")
let barrier reg = Printf.sprintf "__asm__(\"\" : \"+%s\"(z))" reg

(* Each Clang target, its register class for a float, the float types it
   compares in, and those whose compared zeros its backend lowers to a minimum
   or a maximum: x86_64 and riscv64 compare a bfloat16 in float32. *)
let clang_targets =
  Dtype.
    [
      ( "arm64",
        arm64,
        "w",
        [ Float16; Bfloat16; Float32; Float64 ],
        [ Float16; Bfloat16; Float32; Float64 ] );
      ("x86_64", clang, "x", [ Float16; Float32; Float64 ], [ Float16 ]);
      ( "riscv64",
        riscv64,
        "f",
        [ Float16; Float32; Float64 ],
        [ Float16; Float32; Float64 ] );
    ]

let per_target =
  List.concat_map
    (fun (name, r, reg, dts, zeros) ->
      List.map (fun dt -> (name, r, reg, dt, List.mem dt zeros)) dts)
    clang_targets

let target_name (name, _, _, dt, _) = name ^ ", " ^ Dtype.name dt

let zeros_compared =
  group "zeros a comparison reads"
    [
      cases
        "a float zero a comparison reads is opaque to a backend that lowers it \
         to an extreme, and a literal elsewhere"
        ~id:target_name per_target (fun (_, r, reg, dt, zeros) ->
          (* a failure prints the source *)
          let says msg want uops =
            let src = render r uops in
            equal ~msg:(msg ^ " in\n" ^ src) bool want
              (contains src (barrier reg))
          in
          says "0" zeros (const_select dt 0.);
          says "-0" zeros (const_select dt (-0.));
          says "1.5" false (const_select dt 1.5);
          says "a mask" false (zero_masked dt));
      cases "a select of a value and a zero picks the zero at -0., on the host"
        ~id:Dtype.name
        Dtype.[ Float16; Float32; Float64 ]
        (fun dt ->
          let k = Run.program (Lazy.force !host) (const_select dt 0.) in
          match List.assoc 0 (Run.on_host k [ (1, [| `Float (-0.) |]) ]) with
          | [| `Float z |] -> equal float_exact 0. z
          | _ -> failf "one float");
    ]

(* Selections of 16-bit floats *)

(* [data0[0] = c ? x : y] and [data0[1] = c ? data4[1] : 0], [x] the bits of
   [data1[0]] as a [dt], [y] [data2[0]] narrowed to [dt] and [c] [data3[0]]:
   a selection and a gated load of [dt], stored as their bits. The golden
   [clang_selected_half] holds x86_64's source of such a kernel. *)
let selected dt =
  let at slot dt i = Ops.index (Shape.param ~shape:[ Int 2 ] slot dt) [ i ] in
  let i n = Ops.int ~dtype:Int32 n in
  let c = Ops.load (at 3 Bool (i 0)) [] in
  let x = Ops.bitcast (Ops.load (at 1 Uint16 (i 0)) []) dt in
  let y = Ops.cast (Ops.load (at 2 Float32 (i 0)) []) dt in
  let gated = Ops.load (at 4 dt (i 1)) [ Ops.float ~dtype:dt 0.; c ] in
  let bits v = Ops.bitcast v Uint16 in
  Linearizer.linearize
    (Ops.sink
       ~kernel:(Ops.kernel_info ~name:"selected" ())
       [
         Ops.store (at 0 Uint16 (i 0)) (bits (Ops.where c x y));
         Ops.store (at 0 Uint16 (i 1)) (bits gated);
       ])

(* Each Clang target and the 16-bit floats it holds as a storage format, a
   [__fp16] or a [__bf16]: x86_64 holds a bfloat16 as its bits. *)
let held =
  Dtype.
    [
      ("arm64", arm64, [ Float16; Bfloat16 ]);
      ("x86_64", clang, [ Float16 ]);
      ("riscv64", riscv64, [ Float16; Bfloat16 ]);
    ]
  |> List.concat_map (fun (name, r, dts) ->
      List.map (fun dt -> (name ^ ", " ^ Dtype.name dt, r, dt)) dts)

(* The occurrences of [sub] in [s]. *)
let occurrences s sub =
  let n = String.length sub in
  let rec go i k =
    if i + n > String.length s then k
    else if String.sub s i n = sub then go (i + n) (k + 1)
    else go (i + 1) k
  in
  go 0 0

let selections =
  group "selections of 16-bit floats"
    [
      cases
        "a selection and a gated load of a 16-bit float held as a storage \
         format choose between bits"
        ~id:(fun (name, _, _) -> name)
        held
        (fun (_, r, dt) ->
          let src = render r (selected dt) in
          let t = if Dtype.equal dt Float16 then "__fp16" else "__bf16" in
          let pick =
            Printf.sprintf "__builtin_bit_cast(%s, (unsigned short)((" t
          in
          equal ~msg:("in\n" ^ src) int 2 (occurrences src pick));
    ]

(* x86_64 Windows

   An object for x86_64 Windows calls by Windows' convention, where a [long]
   has 32 bits, and is called by System V's. On an x86_64 host of another
   system, a kernel that calls none of the process's functions runs as there:
   the object's own calls follow one convention. *)

let windows cpu = Cstyle.clang (target "CPU" "CLANG" ("x86_64-windows," ^ cpu))
let sysv_abi = "__attribute__((sysv_abi)) "

let x86_rows =
  List.filter
    (fun row -> String.starts_with ~prefix:"x86_64," (row "arch"))
    clang_rows

let on_windows row =
  let cpu = List.nth (String.split_on_char ',' (row "arch")) 1 in
  render (windows cpu) (kernel row)

let replace_all ~sub ~by s =
  let n = String.length sub and b = Buffer.create (String.length s) in
  let rec go i =
    if i >= String.length s then ()
    else if i + n <= String.length s && String.sub s i n = sub then (
      Buffer.add_string b by;
      go (i + n))
    else (
      Buffer.add_char b s.[i];
      go (i + 1))
  in
  go 0;
  Buffer.contents b

let is_digit c = c >= '0' && c <= '9'

let is_word c =
  (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || is_digit c || c = '_'

(* [source] written for 64-bit [long]s and no System V attribute: [long long]
   is [long], and a literal [5ll] or [0x7ffull] is [5l] or [0x7fful]. *)
let as_lp64 source =
  let s = replace_all ~sub:sysv_abi ~by:"" source in
  let s = replace_all ~sub:"long long" ~by:"long" s in
  let s = replace_all ~sub:"long_long" ~by:"long" s in
  let b = Buffer.create (String.length s) in
  let rec go i =
    if i >= String.length s then ()
    else if is_word s.[i] then (
      let j = ref i in
      while !j < String.length s && is_word s.[!j] do
        incr j
      done;
      let w = String.sub s i (!j - i) in
      let literal = is_digit w.[0] && String.ends_with ~suffix:"ll" w in
      Buffer.add_string b
        (if literal then String.sub w 0 (String.length w - 1) else w);
      go !j)
    else (
      Buffer.add_char b s.[i];
      go (i + 1))
  in
  go 0;
  Buffer.contents b

(* The words of [source] that spell a 64-bit integer as LP64 does: a [long]
   not in a [long long], and an integer literal that ends in one [l]. *)
let lp64_words source =
  let words =
    String.split_on_char ' '
      (String.map (fun c -> if is_word c then c else ' ') source)
    |> List.filter (( <> ) "")
  in
  let rec go = function
    | "long" :: "long" :: rest -> go rest
    | "long" :: rest -> "long" :: go rest
    | w :: rest ->
        let n = String.length w in
        let lone_l =
          is_digit w.[0] && n >= 2 && w.[n - 1] = 'l' && w.[n - 2] <> 'l'
        in
        if lone_l then w :: go rest else go rest
    | [] -> []
  in
  go words

let windows_host = lazy (windows "native")

(* A source that compiles only for the Windows triple, without a stack probe:
   [_WIN64] is the triple's, and a frame of 16 KiB would call [___chkstk_ms]
   on the way in. *)
let windows_only =
  {|#ifndef _WIN64
#error "not compiled for Windows"
#endif
_Static_assert(sizeof(long) == 4, "LLP64");
__attribute__((sysv_abi)) void k(volatile char *p) {
  volatile char b[16384];
  b[p[0]] = p[1];
  p[2] = b[p[3]];
}
|}

(* On a host of another machine than x86_64 the objects do not run. *)
let on_x86_64 name f =
  if Rig.instruction_set Rig.host = "x86_64" then f ()
  else test name (fun () -> skip ~reason:"the host is not x86_64" ())

let windows_group =
  group "x86_64 Windows"
    [
      cases
        "a kernel's source is its source for 64-bit longs, with its kernel \
         System V's"
        ~id:(fun row -> row "case")
        x86_rows
        (fun row ->
          let src = on_windows row in
          equal (list string) [] (lp64_words src);
          equal text (render (renderer_of_row row) (kernel row)) (as_lp64 src));
      test "the kernel is declared System V's" (fun () ->
          let src = on_windows (find_case "clang_add") in
          equal (list string)
            [
              "__attribute__((sysv_abi)) __attribute__((min_vector_width(512))) \
               void E_16_4(float* restrict data0_64, float* restrict \
               data1_64, float* restrict data2_64) {";
            ]
            (List.filter
               (fun l -> String.starts_with ~prefix:"__attribute__" l)
               (String.split_on_char '\n' src)));
      test "a 64-bit integer is a long long and its literals end in ll"
        (fun () ->
          let src = on_windows (find_case "clang_dtype_i64") in
          Windtrap.contains ~sub:"long long " src;
          Windtrap.contains ~sub:"*3ll)" src);
      test
        "an object is compiled for Windows' triple and calls no stack probe"
        (fun () ->
          let r = windows "x86-64" in
          equal (list string) []
            (undefined_in (Renderer.Compiler.compile r.compiler windows_only)));
      on_x86_64 "every kernel compiles and loads" (fun () ->
          slow
          @@ group "every kernel compiles and loads"
               (List.map (compiles_and_loads ~on:windows_host) clang_rows));
      on_x86_64 "every kernel the interpreter runs writes what it computes"
        (fun () ->
          slow
          @@ group "every kernel the interpreter runs writes what it computes"
               (List.map
                  (agrees_with_the_interpreter ~on:windows_host ~alone:true)
                  interpreted_rows));
    ]

let () =
  exit
    (run "Tolk.Cstyle"
       [
         sources;
         narrowing;
         rewrites;
         declarations;
         written;
         errors;
         parentheses;
         negation;
         grouping;
         rendering;
         compilation;
         cuda_binaries;
         bf16_truncation;
         bfloat16_on_amd;
         execution;
         division_group;
         metal_cores;
         zeros_compared;
         selections;
         windows_group;
       ])
