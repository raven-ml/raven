open Windtrap
open Tolk

(* The host's target, as the engine gives it. *)
let host_target = Tolk_engine.target Nx_device.host
let rejects f = raises_match (Exn.invalid_arg ?substring:None) f
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

let renderer_for (t : Helpers.Target.t) =
  match t.renderer with
  | "CLANG" -> Cstyle.clang t
  | "METAL" -> Cstyle.metal t
  | "CUDA" -> Cstyle.cuda t
  | "HIP" -> Cstyle.hip t
  | r -> invalid_arg ("no renderer " ^ r)

(* The target of a golden row, written in its columns device, renderer and
   arch. *)
let target_of_row row = target (row "device") (row "renderer") (row "arch")
let renderer_of_row row = renderer_for (target_of_row row)
let clang = Cstyle.clang (target "CPU" "CLANG" "x86_64,x86-64")
let metal = Cstyle.metal (target "METAL" "METAL" "Apple9")
let cuda = Cstyle.cuda (target "CUDA" "CUDA" "sm_89")
let hip = Cstyle.hip (target "AMD" "HIP" "gfx1100")
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
        let source () =
          let src = render (renderer_of_row row) (kernel row) in
          if row "renderer" = "CUDA" then Cuda_fp8.tinygrad_of_d16 src else src
        in
        Some (Golden.text (row "case" ^ ".golden") source))
    case_rows

(* A setting is read once per process, so the cases rendered under one run in a
   process of their own, which the stanza starts with the setting's variable set
   and the setting's tag selected. *)
let sources =
  group "sources"
    [
      group "by default" (sources_under "-");
      group ~tags:[ "expand-ssa" ] "with EXPAND_SSA=1"
        (sources_under "EXPAND_SSA=1");
      group ~tags:[ "unaligned" ] "with ALIGNED=0" (sources_under "ALIGNED=0");
    ]

(* C computes an operation on a char, a short or Clang's __fp16 in a wider
   type, so the source casts each such operation back to its type: an operation
   on a scalar of one of these types, inlined into its one user, which does not
   store it. The source of a kernel is then tinygrad's for the kernel with those
   casts, which [with_d17_casts row uops] builds and cases.golden's column
   narrowed names. *)

let narrow_scalars row =
  Dtype.[ Int8; Uint8; Int16; Uint16 ]
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
    && Ops.max_numel u = 1
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

let rewrites =
  group "extra_matcher"
    [
      Golden.cases "rewrites.golden" (fun row ->
          let r = renderer_of_row row in
          equal Uops.uop
            (nth_of rewritten row "output")
            (Ops.graph_rewrite ~ctx:()
               (nth_of rewrite_inputs row "input")
               r.extra_matcher));
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
          equal (option string) (Some cell)
            (Renderer.Compiler.cachekey r.compiler));
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
            (Cstyle.metal (target "METAL" "METAL" "Mac2")).tensor_cores);
      test "CUDA rejects an architecture without a compute capability"
        (fun () -> rejects (fun () -> Cstyle.cuda (target "CUDA" "CUDA" "sm_")));
    ]

(* Parentheses *)

(* Five operations in a row, each on the last's result: an associative operation
   is written without the parentheses of its operands, and a subtraction keeps
   them. *)
let parentheses =
  group "parentheses"
    [
      cases ~name:fst "an associative chain is written flat"
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

(* Float8 infinities on CUDA *)

(* A kernel converts a value to a float8 type when it casts to one anything but
   a constant. *)
let converts_to_fp8 uops =
  List.exists
    (fun u ->
      Op.equal (Ops.op u) Cast
      && List.mem (Ops.dtype u) Dtype.fp8s
      && not (Op.equal (Ops.op (List.hd (Ops.src u))) Const))
    uops

let cuda_rows =
  List.filter
    (fun row -> row "renderer" = "CUDA" && row "setting" = "-")
    case_rows

let guard_only_where_converted () =
  let guarded =
    List.filter (fun row -> converts_to_fp8 (kernel row)) cuda_rows
    |> List.map (fun row -> row "case")
  in
  equal (list string) ~msg:"the kernels that convert"
    [
      "cuda_dtype_float8_e4m3";
      "cuda_dtype_float8_e5m2";
      "cuda_inf_nan_float8_e4m3";
      "cuda_inf_nan_float8_e5m2";
    ]
    guarded;
  List.iter
    (fun row ->
      equal bool ~msg:(row "case")
        (List.mem (row "case") guarded)
        (contains (render cuda (kernel row)) Cuda_fp8.helper))
    cuda_rows

let converts_with_the_infinity_byte (case, t, dt) =
  let call =
    Printf.sprintf "tg_fp8<%s>(val1.x, %s)" t
      (Cuda_fp8.infinity_byte dt infinity)
  in
  satisfies
    ~claim:("a source that converts by " ^ call)
    string
    (fun src -> contains src call)
    (source_of_case case)

let fp8_infinities =
  group "float8 infinities on CUDA"
    [
      test "declares the guard exactly when it converts a value to a float8"
        guard_only_where_converted;
      cases
        ~name:(fun (_, t, _) -> t)
        "converts with the byte of the infinity's image"
        [
          ("cuda_dtype_float8_e4m3", "__nv_fp8_e4m3", Dtype.Fp8e4m3);
          ("cuda_dtype_float8_e5m2", "__nv_fp8_e5m2", Fp8e5m2);
        ]
        converts_with_the_infinity_byte;
      cases ~name:Fun.id "writes an infinite e5m2 constant as its bits"
        [ "0x7c"; "0xfc" ] (fun byte ->
          let bits = "tg_bitcast<__nv_fp8_e5m2>((unsigned char)" ^ byte ^ ")" in
          satisfies
            ~claim:("a source that writes " ^ bits)
            string
            (fun src -> contains src bits)
            (source_of_case "cuda_inf_nan_float8_e5m2"));
      test
        "the image of an infinity is a NaN of its sign in e4m3 and itself in \
         e5m2" (fun () ->
          equal string "0x7f" (Cuda_fp8.infinity_byte Fp8e4m3 infinity);
          equal string "0xff" (Cuda_fp8.infinity_byte Fp8e4m3 neg_infinity);
          equal string "0x7c" (Cuda_fp8.infinity_byte Fp8e5m2 infinity);
          equal string "0xfc" (Cuda_fp8.infinity_byte Fp8e5m2 neg_infinity));
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
    Ops.index (Ops.param ~shape:[ Int 1 ] slot dt) [ Ops.int ~dtype:Int32 0 ]
  in
  let loads =
    List.init (arity op) (fun i -> Ops.load (at (i + 1) (operand i)) [])
  in
  let value = Ops.v op ~src:loads in
  Ops.toposort (Ops.sink [ Ops.store (at 0 (Ops.dtype value)) value ])

let floats_only = Op.[ Exp2; Log2; Sin; Sqrt; Reciprocal; Trunc; Fdiv ]
let ints_only = Op.[ Shl; Shr; And; Or; Xor; Cmod; Cdiv ]

let operations =
  let named =
    [ ("clang", clang); ("metal", metal); ("cuda", cuda); ("hip", hip) ]
  in
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
        (Ops.param ~shape:[ Int 1 ] 7139 Float32)
        [ Ops.int ~dtype:Int32 0 ]
    in
    let sink = Ops.sink [ Ops.store at (Ops.float ~dtype:Float32 7139.) ] in
    Stdlib.Weak.set weak 0 (Some sink);
    ignore (render clang (Ops.toposort sink))
  in
  render_once ();
  Gc.full_major ();
  is_false ~msg:"the kernel is still reachable" (Stdlib.Weak.check weak 0)

let cuda_on_nv () =
  let nv = Cstyle.cuda (target "NV" "CUDA" "sm_89") in
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
      (Ops.param ~shape:[ Int 1 ] slot Float32)
      [ Ops.int ~dtype:Int32 0 ]
  in
  let operands = [ Ops.load (at 1) []; Ops.load (at 2) [] ] in
  let value =
    Ops.v Customi ~src:operands ~arg:(Code { code; dtype = Float32 })
  in
  render clang (Ops.toposort (Ops.sink [ Ops.store (at 0) value ]))

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
      cases ~name:fst "formats custom code as str.format does"
        [
          ("{0}+{1}", "val0+val1");
          ("{1}-{0}", "val1-val0");
          ("{}*{}", "val0*val1");
          ("{{{0}}}", "{val0}");
          ("f({0}) }}", "f(val0) }");
        ]
        formats_custom_code;
      cases ~name:Fun.id "refuses custom code str.format refuses, naming it"
        [ "f({0"; "{0}}"; "{0} } {1}"; "{2}"; "{x}" ] (fun code ->
          raises_match (Exn.invalid_arg ~substring:code) (fun () -> custom code));
      test "writes the same CUDA source for the CUDA and NV devices" cuda_on_nv;
      cases
        ~name:(fun (name, _) -> name)
        "raises Invalid_argument on an operation the target lacks"
        [ ("clang", clang); ("metal", metal); ("cuda", cuda); ("hip", hip) ]
        (fun (_, r) -> rejects (fun () -> render r (one_operation Max Float32)));
      cases ~name:fst
        "raises Invalid_argument on a kernel of another target it cannot write"
        [
          ( "HIP gfx1100 converts to a float8 it lacks",
            (hip, "hip_cdna4_dtype_float8_e4m3") );
          ( "CUDA has no core for chars",
            (cuda, "hip_tc_signed_char_int_16_16_16") );
        ]
        (fun (_, (r, case)) ->
          rejects (fun () -> render r (kernel (find_case case))));
    ]

(* Compilation *)

(* A machine without a target's toolchain refuses every source with the reason
   it cannot load the library. *)
let lacks_toolchain why =
  String.starts_with ~prefix:"failed to load library" why
  || String.starts_with ~prefix:"comgr not available" why

(* The sources the toolchain rejects as tinygrad writes them. *)
let rejected =
  [
    ( "metal_transcendental_bf16",
      "tinygrad's graph truncates a bfloat without the float cast" );
  ]

let compiles_with_its_toolchain row =
  let t =
    test (row "case") (fun () ->
        let r = renderer_of_row row in
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
  group ~tags:[ "slow" ] "every GPU kernel compiles with its target's toolchain"
    (List.map compiles_with_its_toolchain gpu_rows)

(* CUDA's binaries *)

let compiles_to_a_cubin () =
  match Renderer.Compiler.compile cuda.compiler (source_of_case "cuda_add") with
  | binary -> starts_with ~affix:"\x7fELF" binary
  | exception Renderer.Compiler.Compile_error why when lacks_toolchain why ->
      skip ~reason:why ()

let cuda_binaries =
  group "CUDA's binaries"
    [ slow "the CUDA device's kernels compile to a cubin" compiles_to_a_cubin ]

(* bfloat16 truncation on Metal *)

(* Metal has no trunc of a bfloat: it truncates a float, so the extra matcher
   truncates a bfloat16 in float32. *)
let truncated_bf16 =
  let zero = Ops.int ~dtype:Int32 0 in
  let at slot = Ops.index (Ops.param ~shape:[ Int 1 ] slot Bfloat16) [ zero ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"trunc_bf16" ())
    [ Ops.store (at 0) (Ops.trunc (Ops.load (at 1) [])) ]

let through_metal_matcher sink =
  Ops.graph_rewrite ~ctx:() sink metal.extra_matcher

let truncs sink =
  List.filter (fun u -> Op.equal (Ops.op u) Trunc) (Ops.toposort sink)

let truncates_in_float () =
  let rewritten = truncs (through_metal_matcher truncated_bf16) in
  equal (list Dtypes.dtype) [ Dtype.Float32 ] (List.map Ops.dtype rewritten)

(* The graph a kernel reaches the renderer as, after the rewrites that commit
   its constants. *)
let compiles_its_truncation () =
  let sink =
    List.fold_left
      (fun sink m -> Ops.graph_rewrite ~ctx:() sink m)
      (through_metal_matcher truncated_bf16)
      [
        Uop_weak.pm_commit_weak; Uop_weak.pm_lower_weak; Uop_weak.pm_cast_const;
      ]
  in
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
            truncs (Ops.graph_rewrite ~ctx:() truncated_bf16 cuda.extra_matcher)
          in
          equal (list Dtypes.dtype) [ Dtype.Bfloat16 ]
            (List.map Ops.dtype rewritten));
      slow "compiles a kernel that truncates a bfloat16" compiles_its_truncation;
    ]

(* Execution *)

let host = lazy (Cstyle.clang host_target)
let loaded name = Run.program (Lazy.force host) (kernel (find_case name))
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
      (loaded "clang_dtype_half")
      [ (1, floats x); (2, floats (Array.map half y)) ]
  in
  let expected i = half (half (half x.(i) +. y.(i)) *. 3.) in
  equal values (floats (Array.init 16 expected)) (slot 0 out)

let wraps_each_operation_on_chars () =
  let out =
    Run.on_host
      (loaded "clang_dtype_unsigned_char")
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
    ("clang_call_out", "calls a function pointer the test cannot provide");
    ("clang_call_ret", "calls a function pointer the test cannot provide");
    ("clang_call_stack", "calls a function pointer the test cannot provide");
    ("clang_unbounded_loop", "has a loop without a trip count");
    ("clang_sum", "accumulates in registers");
    ("clang_matmul", "accumulates in registers");
    ("clang_matmul_upcasted", "accumulates in registers");
    ("clang_add_max_uchar", "accumulates in registers");
    ("clang_padded", "reads outside its buffer in the branch its gate discards");
    ("clang_gated_store_in_loop", "gates its store with an if");
    ("clang_custom", "runs custom code");
    ("clang_table", "reads a constant table");
    ("clang_transcendental_half", "divides with Ops.FDIV");
    ("clang_transcendental_bf16", "divides with Ops.FDIV");
    ("clang_transcendental_float", "divides with Ops.FDIV");
    ("clang_transcendental_double", "divides with Ops.FDIV");
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
      | Ops.Param ({ addrspace = Some Global; size = Some n; _ } as p) ->
          Some (p.slot, p.dtype, n)
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
    | Some { iterations = Sym u; _ } -> Ops.toposort u
    | _ -> []
  in
  let variable u =
    match Ops.arg u with
    | Ops.Param
        {
          addrspace = Some Alu;
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
      [ (name s.lo, 0); (name s.hi, Ops.sym_infer s.iterations vars) ]

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

let agrees_with_the_interpreter row =
  let uops = kernel row in
  let k = lazy (Run.program (Lazy.force host) uops) in
  prop ~count:10 (row "case") (draw_inputs uops)
    (fun ((buffers, vars) as inputs) ->
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
  let k = lazy (Run.program (Lazy.force host) uops) in
  prop ~count:20 (row "case") (draw_narrow_inputs narrow uops)
    (fun ((buffers, _) as inputs) ->
      equal
        (list (pair int values))
        (interpreted uops inputs)
        (Run.on_host (Lazy.force k) buffers))

let compiles_and_loads row =
  test (row "case") (fun () ->
      ignore (Run.program (Lazy.force host) (kernel row)))

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
      test "runs custom code" runs_custom_code;
      test "accesses volatile buffers" accesses_volatile_buffers;
      test "stores each constant as its type holds it"
        stores_each_constant_as_its_type_holds_it;
      test "reads four chars as a uint through a cast of their address"
        (reads_chars_as_a_uint "clang_packed_cast");
      test "reads four chars as a uint through a bitcast of their address"
        (reads_chars_as_a_uint "clang_packed_bitcast");
      test "picks a lane of a vector by a variable" picks_a_lane_by_a_variable;
      test "reads two uint registers as a ulong through a cast of their address"
        reads_registers_as_a_ulong;
      test "passes named parameters" passes_named_parameters;
      test "rounds each operation on halves to a half"
        rounds_each_operation_on_halves;
      test "wraps each operation on unsigned chars"
        wraps_each_operation_on_chars;
      group ~tags:[ "slow" ] "every kernel compiles and loads"
        (List.map compiles_and_loads clang_rows);
      group ~tags:[ "slow" ]
        "every kernel the interpreter runs writes what it computes"
        (List.map agrees_with_the_interpreter interpreted_rows);
      group ~tags:[ "slow" ]
        "a kernel over a narrow type wraps and rounds as the interpreter"
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
  let i = Ops.range (Int n) [ 0 ] in
  let at slot = Ops.index (Ops.placeholder ~slot [ n ] Float32) [ i ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"fdiv" ())
    [ Ops.end_ (Ops.store (at 0) (Ops.alu (at 1) Fdiv [ at 2 ])) [ i ] ]

let divides_on_metal () =
  match Metal.device with
  | None -> skip ~reason:"no Metal device" ()
  | Some m ->
      let devices =
        Tolk_engine.device [ ("CPU", Nx_device.host); ("CPU:1", m) ]
      in
      let n = Array.length dividends in
      let buffer () = Ops.new_buffer (Single "CPU:1") n Float32 in
      let out = buffer () and a = buffer () and b = buffer () in
      let compiled =
        Hcq2.compile_linear
          ~devices:(fun d -> (devices d).compiler)
          (Ops.v Op.Linear ~src:[ Ops.call division [ out; a; b ] ])
      in
      let on_metal xs = Run.buffer m Float32 (floats xs) in
      let quotients = on_metal (Array.make n 0.) in
      let s =
        Tolk_engine.link ~devices
          ~bound:
            [
              (out, [ quotients ]);
              (a, [ on_metal dividends ]);
              (b, [ on_metal divisors ]);
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
      slow "Metal divides as IEEE does, rounding once" divides_on_metal;
    ]

(* Negation *)

(* A kernel of one float lane: [data0[0] = f (data1[0], data2[0])]. *)
let lane f =
  let zero = Ops.int ~dtype:Int32 0 in
  let at slot = Ops.index (Ops.param ~shape:[ Int 1 ] slot Float32) [ zero ] in
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
      cases
        ~name:(fun (r : Renderer.t) -> r.name)
        "a minus before a minus is apart"
        [ clang; metal; cuda; hip ]
        (fun r ->
          let neg = List.assoc Op.Neg r.code_for_op
          and sub = List.assoc Op.Sub r.code_for_op in
          equal string "(a- -b)" (sub [ "a"; "-b" ] Float32);
          equal string "- -b" (neg [ "-b" ] Float32);
          equal string "(a-b)" (sub [ "a"; "b" ] Float32);
          equal string "-b" (neg [ "b" ] Float32));
      test "Clang compiles and runs a difference with a negated operand"
        (fun () ->
          let k =
            Run.program (Lazy.force host)
              (lane (fun x y -> minus x (negated y)))
          in
          let out =
            Run.on_host k [ (1, [| `Float 1. |]); (2, [| `Float 2. |]) ]
          in
          equal values [| `Float 3. |] (List.assoc 0 out));
      test "Clang compiles and runs a difference with a negative constant"
        (fun () ->
          let k =
            Run.program (Lazy.force host)
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
            Ops.index (Ops.param ~shape:[ Int 1 ] slot Float32) [ zero ]
          in
          let load slot = Ops.load (at slot) [] in
          let mul x y = Ops.alu x Mul [ y ] in
          let k =
            Run.program (Lazy.force host)
              (Linearizer.linearize
                 (Ops.sink
                    ~kernel:(Ops.kernel_info ~name:"grouping" ())
                    [ Ops.store (at 0) (mul (load 1) (mul (load 2) (load 3))) ]))
          in
          let big = Float.ldexp 1. 126 in
          let out =
            Run.on_host k
              [ (1, [| `Float 4. |]); (2, [| `Float big |]); (3, [| `Float 0.25 |]) ]
          in
          equal values [| `Float big |] (List.assoc 0 out));
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
         fp8_infinities;
         rendering;
         compilation;
         cuda_binaries;
         bf16_truncation;
         execution;
         division_group;
       ])
