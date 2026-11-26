open Windtrap
open Tolk

let rejects f = raises_match (Exn.invalid_arg ?sub:None) f

(* The occurrences of [sub] in [s]. *)
let occurrences s sub =
  let n = String.length sub in
  let rec go i k =
    if i + n > String.length s then k
    else if String.sub s i n = sub then go (i + n) (k + 1)
    else go (i + 1) k
  in
  go 0 0

let occurs sub s = occurrences s sub > 0

(* A source without [sub]: a failure prints the source. *)
let lacks ?msg sub src =
  satisfies ?msg
    ~claim:("a source without " ^ sub)
    string
    (fun src -> not (occurs sub src))
    src

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

(* A machine without a target's toolchain refuses every source with the reason
   it cannot load the library: the test skips. *)
let lacks_toolchain why =
  String.starts_with ~prefix:"failed to load library" why
  || String.starts_with ~prefix:"comgr not available" why

let compiles (r : Renderer.t) src =
  match Renderer.Compiler.compile r.compiler src with
  | lib -> lib
  | exception Renderer.Compiler.Compile_error why when lacks_toolchain why ->
      skip ~reason:why ()

(* The target of a golden row, written in its columns device, renderer and
   arch. *)
let target_of_row row = target (row "device") (row "renderer") (row "arch")
let renderer_of_row row = renderer_for (target_of_row row)
let clang_target = target "CPU" "CLANG" "x86_64,x86-64"
let metal_target = target "METAL" "METAL" "Apple9"
let cuda_target = target "CUDA" "CUDA" "sm_89"
let hip_target = target "AMD" "HIP" "gfx1100"
let clang = Cstyle.clang clang_target
let arm64 = Cstyle.clang (target "CPU" "CLANG" "arm64,generic")
let riscv64 = Cstyle.clang (target "CPU" "CLANG" "riscv64,rv64g")
let windows cpu = Cstyle.clang (target "CPU" "CLANG" ("x86_64-windows," ^ cpu))
let metal () = renderer_for metal_target
let cuda () = renderer_for cuda_target
let hip () = renderer_for hip_target

let renderers =
  [ ("clang", fun () -> clang); ("metal", metal); ("cuda", cuda); ("hip", hip) ]

let available =
  List.map renderer_for [ clang_target; metal_target; cuda_target; hip_target ]

let render (r : Renderer.t) uops = r.render uops

(* Recorded kernels *)

(* The cases of kernels.golden are its sink's sources, each a linear program
   named after its case, whose sources are the kernel's nodes in order. *)
let kernels = lazy (Array.of_list (Ops.src (Golden.sink "kernels.golden")))
let kernel row = Ops.src (Lazy.force kernels).(int_of_string (row "kernel"))
let case_rows = Golden.rows "cases.golden"
let find_case name = List.find (fun row -> row "case" = name) case_rows

let source_of_case name =
  render (renderer_of_row (find_case name)) (kernel (find_case name))

(* The cases a renderer writes by default. *)
let rows_of renderer =
  List.filter
    (fun row -> row "renderer" = renderer && row "setting" = "-")
    case_rows

let clang_rows = rows_of "CLANG"

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

let interpreted row = not (List.mem_assoc (row "case") not_interpreted)

(* The host

   A kernel runs on the host as Clang compiles it for the host's processor:
   whole, cut into functions of one statement each ({!Cstyle.clang_in_pieces}),
   and, on an x86_64 host, for Windows' convention. *)

let host_target () = Run.target Rig.host
let host = lazy (Cstyle.clang (host_target ()))
let pieces = lazy (Cstyle.clang_in_pieces ~piece:1 (host_target ()))

let windows_native =
  lazy
    (Clang.here "x86_64-windows,native";
     windows "native")

let on_x86_64 () = Rig.instruction_set Rig.host = "x86_64"

(* Most of a small kernel's compile is the start of Clang's process, so the
   kernels a group runs compile in one run of Clang. [batched r make] is [r]
   with a compiler that answers each source [make] compiles from one
   {!Renderer.Compiler.compile_all} of them, made at its first use: [make] runs
   once with a compiler that records each source and returns no binary, then
   [r]'s compiler compiles the sources it recorded. A source outside them raises
   [Invalid_argument], so that a kernel left out fails its test. Where the run
   fails, each source compiles alone, so that each test reports its own
   error. *)
let batched r make =
  lazy
    (let r : Renderer.t = Lazy.force r in
     let sources = ref [] in
     let record =
       Renderer.Compiler.v (fun src ->
           sources := src :: !sources;
           "")
     in
     make (Renderer.with_compiler record r);
     let sources = List.rev !sources in
     let alone src =
       match Renderer.Compiler.compile_cached r.compiler src with
       | lib -> Ok lib
       | exception Renderer.Compiler.Compile_error why -> Error why
     in
     let binaries =
       match Renderer.Compiler.compile_all r.compiler sources with
       | libs -> List.map Result.ok libs
       | exception Renderer.Compiler.Compile_error _ -> List.map alone sources
     in
     let table = Hashtbl.create 128 in
     List.iter2 (Hashtbl.replace table) sources binaries;
     let answer src =
       match Hashtbl.find_opt table src with
       | Some (Ok lib) -> lib
       | Some (Error why) -> raise (Renderer.Compiler.Compile_error why)
       | None -> invalid_arg "a source outside the batch"
     in
     Renderer.with_compiler (Renderer.Compiler.v answer) r)

let compile_all kernels r =
  List.iter (fun k -> ignore (Run.program r k)) kernels

let program r uops = Run.program (Lazy.force r) uops
let loaded r name = program r (kernel (find_case name))
let floats xs = Array.map (fun x -> `Float x) xs
let ints xs = Array.map (fun n -> `Int (Bigint.of_int n)) xs
let slot s outputs = List.assoc s outputs
let values = array Dtypes.value

(* Kernels of one operation *)

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

(* A kernel of one float lane: [data0[0] = f data1[0] data2[0] data3[0]]. *)
let lane f =
  let zero = Ops.int ~dtype:Int32 0 in
  let at slot =
    Ops.index (Shape.param ~shape:[ Int 1 ] slot Float32) [ zero ]
  in
  let load slot = Ops.load (at slot) [] in
  Linearizer.linearize
    (Ops.sink
       ~kernel:(Ops.kernel_info ~name:"lane" ())
       [ Ops.store (at 0) (f (load 1) (load 2) (load 3)) ])

let minus x y = Ops.alu x Sub [ y ]
let times x y = Ops.alu x Mul [ y ]
let minus_negated = lane (fun x y _ -> minus x (Ops.alu y Neg []))

let minus_constant =
  lane (fun x _ _ -> minus x (Ops.float ~dtype:Float32 (-1.5)))

let grouped = lane (fun x y z -> times x (times y z))

(* Inlining

   [chain n] stores [r n] for [r 0 = data1[0]] and [r (i+1) = (r i << data2[0])
   % data3[0]], over int32. Each step is read once, by the next, so written into
   its reader the chain would nest two brackets a step. *)
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
let pp_renderer ppf (r : Renderer.t) = Format.pp_print_string ppf r.name

let renderers_and_lengths =
  Gen.pair (Gen.of_list ~pp:pp_renderer available) (Gen.int_range 1 300)

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

(* An expression written into its reader is at most 4096 characters long. A
   statement holds at most two of a sum's, its operands, and its own text. *)
let max_statement = (2 * 4096) + 64

(* Each term of a sum is a load of about 19 characters, so a sum of more than
   600 terms written as one expression is longer than [max_statement]. *)
let renderers_and_sums =
  Gen.pair (Gen.of_list ~pp:pp_renderer available) (Gen.int_range 2 700)

let writes_bounded_statements ((r : Renderer.t), n) =
  cover "a sum longer than an inlined text" (n > 600);
  let longest =
    List.fold_left
      (fun m line -> max m (String.length line))
      0
      (String.split_on_char '\n' (render r (sum n)))
  in
  at_most int ~than:max_statement longest

(* [n] float32 values whose float32 sum depends on the order of its additions:
   large values among small ones, whose low bits each addition rounds away. *)
let summands n =
  let f32 x = Int32.float_of_bits (Int32.bits_of_float x) in
  Array.init n (fun i ->
      f32
        (if i mod 3 = 0 then 1e7 +. Float.of_int i else 0.37 *. Float.of_int i))

let long_sum = 1500
let chain_steps = 300

(* NaN constants

   Signalling NaNs of either sign with a payload, as their bits: [data0] holds
   float16s, [data1] float32s and [data2] float64s. *)
let nan_bits =
  [
    (Dtype.Float16, Dtype.Uint16, [ "0x7d01"; "0xfd02" ]);
    (Float32, Uint32, [ "0x7fa00001"; "0xffa00002" ]);
    (Float64, Uint64, [ "0x7ff4000000000001"; "0xfff4000000000002" ]);
  ]

let nan_constants =
  let stores slot (dt, bits, codes) =
    List.mapi
      (fun i code ->
        let at =
          Ops.index
            (Shape.param ~shape:[ Int 2 ] slot dt)
            [ Ops.int ~dtype:Int32 i ]
        in
        let nan = Dtype.bitcast bits dt (`Int (Bigint.of_string code)) in
        Ops.store at (Ops.const ~dtype:dt (nan :> Dtype.const)))
      codes
  in
  Linearizer.linearize
    (Ops.sink
       ~kernel:(Ops.kernel_info ~name:"nan_constants" ())
       (List.concat (List.mapi stores nan_bits)))

(* The preamble's kernels that run on the host. *)
let written_kernels =
  batched host
    (compile_all
       [
         minus_negated;
         minus_constant;
         grouped;
         chain chain_steps;
         sum long_sum;
         nan_constants;
       ])

(* 300 steps of [chain], 600 brackets deep if written as one expression. The
   operands keep every step below 2^27, so int32's [<<] and [%] are OCaml's. *)
let computes_a_long_chain r () =
  let n = chain_steps and x = 123457 and shift = 7 and modulus = 1000003 in
  let step r _ = (r lsl shift) mod modulus in
  let expected = List.fold_left step x (List.init n Fun.id) in
  let out =
    Run.on_host
      (program r (chain n))
      [ (1, ints [| x |]); (2, ints [| shift |]); (3, ints [| modulus |]) ]
  in
  equal values (ints [| expected |]) (slot 0 out)

(* Each addition of the sum rounds to float32, in order, as the fold does in
   double precision and rounds after, which rounds a sum of two float32 values
   correctly. *)
let adds_a_long_sum_in_order () =
  let x = summands long_sum in
  let f32 v = Int32.float_of_bits (Int32.bits_of_float v) in
  let expected = ref x.(0) in
  for i = 1 to long_sum - 1 do
    expected := f32 (!expected +. x.(i))
  done;
  let out =
    Run.on_host (program written_kernels (sum long_sum)) [ (1, floats x) ]
  in
  equal values (floats [| !expected |]) (slot 0 out)

(* The constants' bits are moved, never computed: each element holds its
   constant's bits exactly. *)
let stores_nan_constants_bit_for_bit () =
  let out = Run.on_host (program written_kernels nan_constants) [] in
  List.iteri
    (fun s (dt, bits, codes) ->
      equal values
        ~msg:(Format.asprintf "%a" Dtype.pp dt)
        (Array.of_list (List.map (fun c -> `Int (Bigint.of_string c)) codes))
        (Array.map (Dtype.bitcast dt bits) (slot s out)))
    nan_bits

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

let named_cases = Gen.of_list (List.map (fun row -> row "case") case_rows)

let keeps_no_state (a, b) =
  let first = source_of_case a in
  ignore (source_of_case b);
  equal string first (source_of_case a)

(* Parameters' types

   A kernel names each parameter's type in its signature, read or not, so its
   source declares that type. A store at an index never valid is dropped and its
   parameter stays: over 64 indices, slot 0 takes a value of [dt] where the
   index is negative, and slot 1 a float32 at each. NVRTC reads an undeclared
   type first in a signature as a variable's, and rejects the kernel ([attribute
   "__global__" does not apply here]). *)
let unwritten dt =
  let open Ops.O in
  let i = Ops.range ~axis_type:Global (Int 64) [ 0 ] in
  let at slot dt i = Ops.index (Shape.param ~shape:[ Int 64 ] slot dt) [ i ] in
  let one = Ops.float ~dtype:Float32 1. in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"unwritten" ~opts_to_apply:[] ())
    [
      Ops.end_
        (Ops.group
           [
             Ops.store
               (at 0 dt (Ops.where (i < int 0) i Ops.invalid))
               (Ops.cast one dt);
             Ops.store (at 1 Float32 i) one;
           ])
        [ i ];
    ]

let unwritten_source (r : Renderer.t) dt =
  let p = Codegen.linearize (unwritten dt) r in
  render r (Ops.src (Ops.nth p 1))

let declared_types =
  [
    ( "CUDA bfloat16",
      (cuda_target, Dtype.Bfloat16, "nv_bfloat16*", "#include <cuda_bf16.h>") );
    ( "CUDA float16",
      (cuda_target, Dtype.Float16, "half*", "#include <cuda_fp16.h>") );
    ( "CUDA fp8e4m3",
      (cuda_target, Dtype.Fp8e4m3, "__nv_fp8_e4m3*", "#include <cuda_fp8.h>") );
    ( "HIP float16",
      (hip_target, Dtype.Float16, "half*", "#define half _Float16") );
  ]

(* Division

   Metal's [/] is IEEE's division, rounded once, which a product by the
   reciprocal is not for these operands. *)

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
              (out, quotients); (a, on_metal dividends); (b, on_metal divisors);
            ]
          compiled
      in
      Tolk_engine.run s [||];
      equal values
        (floats (Array.init n quotient))
        (Run.values Float32 quotients)

(* What a renderer writes, on every target *)

(* A setting is read once per process, so the cases rendered under one run in a
   process of their own, which the stanza starts with the setting's variable set
   and the setting's tag selected. *)
let sources_under setting =
  List.filter_map
    (fun row ->
      if row "setting" <> setting then None
      else
        let source () = render (renderer_of_row row) (kernel row) in
        Some (Golden.text (row "case" ^ ".golden") source))
    case_rows

let render_group =
  group "render"
    [
      group "writes each recorded kernel as its golden states"
        (sources_under "-");
      tag "expand-ssa"
      @@ group
           "under EXPAND_SSA=1 writes each recorded kernel as its golden states"
           (sources_under "EXPAND_SSA=1");
      prop "keeps no state from one kernel to the next"
        (Gen.pair named_cases named_cases)
        keeps_no_state;
      prop "a chain of any length nests a bounded number of brackets"
        ~examples:
          (List.concat_map (fun r -> [ (r, 1); (r, 29); (r, 300) ]) available)
        renderers_and_lengths nests_a_bounded_depth;
      prop "a sum of any length writes statements of bounded length"
        ~examples:(List.map (fun r -> (r, 700)) available)
        renderers_and_sums writes_bounded_statements;
      test "the host computes a chain too deep for one expression"
        (computes_a_long_chain written_kernels);
      test "the host adds a sum too long for one expression in order"
        adds_a_long_sum_in_order;
      cases "a minus before an operand's minus stays apart" ~id:fst renderers
        (fun (_, r) ->
          let r = r () in
          let neg = List.assoc Op.Neg r.code_for_op
          and sub = List.assoc Op.Sub r.code_for_op in
          equal string "(a- -b)" (sub [ "a"; "-b" ] Float32);
          equal string "- -b" (neg [ "-b" ] Float32);
          equal string "(a-b)" (sub [ "a"; "b" ] Float32);
          equal string "-b" (neg [ "b" ] Float32));
      test "the host computes a difference with a negated operand" (fun () ->
          let out =
            Run.on_host
              (program written_kernels minus_negated)
              [ (1, floats [| 1. |]); (2, floats [| 2. |]) ]
          in
          equal values (floats [| 3. |]) (slot 0 out));
      test "the host computes a difference with a negative constant" (fun () ->
          let out =
            Run.on_host
              (program written_kernels minus_constant)
              [ (1, floats [| 1. |]) ]
          in
          equal values (floats [| 2.5 |]) (slot 0 out));
      (* At 4 * (2^126 * 0.25) the inner product is finite and so is the whole,
         where (4 * 2^126) * 0.25 overflows. *)
      test "the host computes a product of a product as it is grouped"
        (fun () ->
          let big = Float.ldexp 1. 126 in
          let out =
            Run.on_host
              (program written_kernels grouped)
              [
                (1, floats [| 4. |]);
                (2, floats [| big |]);
                (3, floats [| 0.25 |]);
              ]
          in
          equal values (floats [| big |]) (slot 0 out));
      cases "raises Invalid_argument on an operation the target lacks" ~id:fst
        renderers (fun (_, r) ->
          let r = r () in
          rejects (fun () -> render r (one_operation Max Float32)));
      cases
        "raises Invalid_argument on a kernel of another target it cannot write"
        ~id:fst
        [
          ( "HIP gfx1100 converts to a float8 it lacks",
            (hip, "hip_cdna4_dtype_fp8e4m3") );
          ("CUDA has no core for chars", (cuda, "hip_tc_i8_i32_16_16_16"));
        ]
        (fun (_, (r, case)) ->
          let r = r () in
          rejects (fun () -> render r (kernel (find_case case))));
      test "the host stores a NaN constant with its sign and payload"
        stores_nan_constants_bit_for_bit;
      cases "writes custom code as str.format fills it" ~id:fst
        [
          ("{0}+{1}", "val0+val1");
          ("{1}-{0}", "val1-val0");
          ("{}*{}", "val0*val1");
          ("{{{0}}}", "{val0}");
          ("f({0}) }}", "f(val0) }");
        ]
        (fun (code, expr) ->
          Windtrap.contains ~sub:(" = " ^ expr ^ ";") (custom code));
      cases "raises Invalid_argument naming custom code str.format refuses"
        ~id:Fun.id [ "f({0"; "{0}}"; "{0} } {1}"; "{2}"; "{x}" ] (fun code ->
          raises_match (Exn.invalid_arg ~sub:code) (fun () -> custom code));
      cases "declares the type of a parameter the kernel never reads or writes"
        ~id:fst declared_types (fun (_, (t, dt, param, line)) ->
          let src = unwritten_source (renderer_for t) dt in
          Windtrap.contains ~sub:(param ^ " data0_64") src;
          Windtrap.contains ~sub:line src);
      cases "a CUDA kernel with a parameter it never reads or writes compiles"
        ~id:fst
        (List.filter (fun (_, (t, _, _, _)) -> t == cuda_target) declared_types)
        (fun (_, (t, dt, _, _)) ->
          let r = compiling t in
          ignore (compiles r (unwritten_source r dt)));
      (* test/README.md's D50 check: 3 / 7 is 0x3edb6db7 in float32, where the
         product by the reciprocal is 0x3edb6db8. *)
      test "the operands tell a quotient from a product by the reciprocal"
        (fun () ->
          equal int32 0x3edb6db7l (Int32.bits_of_float (quotient 0));
          equal int32 0x3edb6db8l
            (Int32.bits_of_float (product_by_reciprocal 0)));
      test "Metal divides as IEEE does, rounding once" divides_on_metal;
    ]

(* Rewrites before rendering *)

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
      group "rewrites each recorded input as its golden states"
        [
          (if Golden.updating then
             Golden.graph "rewritten.golden" recorded_rewrites
           else
             Golden.cases "rewrites.golden" (fun row ->
                 equal Uops.uop (nth_of rewritten row "output") (rewrite row)));
        ];
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

(* A compiler's table is named after its toolchain, before a digest of the
   toolchain and its options. The CPU table names the Clang that CC runs, as
   clang-15 for CC=clang-15 (D63). *)
let names_its_table (r : Renderer.t) cell =
  let r = compiling r.target in
  let table = Option.get (Renderer.Compiler.cachekey r.compiler) in
  let named =
    let clang = "compile_clang_" in
    if r.target.renderer <> "CLANG" then cell
    else if not (String.starts_with ~prefix:clang cell) then
      failwith ("a CPU table not named after Clang: " ^ cell)
    else
      let n = String.length clang in
      "compile_" ^ Setting.value Setting.cc ^ "_"
      ^ String.sub cell n (String.length cell - n)
  in
  starts_with ~prefix:(named ^ "_") table

(* renderer.mli states each target's SIMD group and most threads of a workgroup:
   a CUDA warp's 32 and a block's 1024; an AMD wave's 64 on gfx9 and 32 from
   gfx10, and a workgroup's 1024; a Metal SIMD group's 32, and a threadgroup's
   1024, but 512 on Apple2 and Apple3; one thread on the host, which launches no
   workgroups. *)
let threads =
  cases "simd_threads and local_threads_max are each target's"
    ~id:(fun (device, renderer, arch, _, _) ->
      String.concat " " [ device; renderer; arch ])
    [
      ("CPU", "CLANG", "x86_64,x86-64", 1, 1);
      ("CUDA", "CUDA", "sm_89", 32, 1024);
      ("NV", "CUDA", "sm_80", 32, 1024);
      ("AMD", "HIP", "gfx942", 64, 1024);
      ("AMD", "HIP", "gfx1100", 32, 1024);
      ("AMD", "HIP", "gfx1201", 32, 1024);
      ("METAL", "METAL", "Apple9", 32, 1024);
      ("METAL", "METAL", "Apple4", 32, 1024);
      ("METAL", "METAL", "Apple3", 32, 512);
      ("METAL", "METAL", "Apple2", 32, 512);
      ("METAL", "METAL", "Mac2", 32, 1024);
    ]
    (fun (device, renderer, arch, simd, most) ->
      let r = renderer_for (target device renderer arch) in
      equal ~msg:"SIMD group" int simd r.simd_threads;
      equal ~msg:"workgroup" int most r.local_threads_max)

let declarations =
  group "declarations"
    [
      group "each target declares its golden's values"
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
              equal string cell
                (joined " " Dtype.pp (Renderer.supported_dtypes r)));
          declared "cachekey" names_its_table;
        ];
      threads;
    ]

(* Native operations *)

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
  Windtrap.contains ~sub:(" = " ^ expr ^ ";") (render r (one_operation op dt))

let code_for_op =
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
          let write = List.assoc op r.code_for_op in
          let operands =
            List.filteri (fun i _ -> i < arity op) [ "a"; "b"; "c" ]
          in
          equal string (row "written")
            (write operands (Dtypes.dtype_of_cell (row "dtype"))));
      prop "render writes each native operation as code_for_op does" operations
        writes_its_operation;
    ]

(* param_name *)

let param_name =
  group "param_name"
    [
      cases "is the parameter's name, or data and its slot, then its shape"
        ~id:fst
        [
          ("a buffer", (Shape.param ~shape:[ Int 16 ] 0 Float32, "data0_16"));
          ( "a named buffer",
            ( Shape.param ~shape:[ Int 4 ] ~name:"input_tile" 3 Float32,
              "input_tile_4" ) );
          ( "a named scalar",
            (Shape.param ~name:"for" ~addrspace:Alu 2 Int32, "for_") );
        ]
        (fun (_, (p, name)) -> equal string name (Cstyle.param_name p));
      test "raises Invalid_argument on a node that is no parameter" (fun () ->
          rejects (fun () -> Cstyle.param_name (Ops.int ~dtype:Int32 0)));
    ]

(* Clang: kernels on the host *)

(* A loop of [n] operations on each element, an addition of a value loaded
   before the loop and an exclusive or with it in turn, so that a piece of the
   loop reads that value from the frame on every trip. Each operation's value is
   read once: the source writes the chain as one statement per 60 or so
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

(* What [looped_chain n] writes for [x] and [c]. *)
let chain_of n c x =
  List.fold_left
    (fun x i -> if i mod 2 = 0 then x + c else x lxor c)
    x (List.init n Fun.id)

let computes_its_chain n program =
  let c = 3 and xs = [| 10; -2; 0; 7 |] in
  let out = Run.on_host program [ (1, ints xs); (2, ints [| c |]) ] in
  equal values (ints (Array.map (chain_of n c) xs)) (slot 0 out)

(* Past the 20,000 operations from which a host kernel is cut, linearized once
   for the tests that read it. *)
let long_chain = 20_500
let long_program = lazy (looped_chain long_chain)
let outer_chain = 200
let frame = "looped_chain_frame"

(* Two loops one after the other whose indices render as one name, each scoped
   to its loop: the first writes [2 * (3 * i + 100)] through a value it reads
   twice, so that the value is declared, the second [i + 1]. *)
let sibling_loops =
  let trips n = Ops.v Cast ~src:[ Ops.int ~dtype:Int32 n ] ~arg:(Dtype Int32) in
  let loop n slot value =
    let r = Ops.range ~axis_type:Loop ~dtype:Int32 (Sym (trips n)) [ 0 ] in
    let at = Ops.index (Shape.param ~shape:[ Int 4 ] slot Int32) [ r ] in
    Ops.end_ (Ops.store at (value r)) [ r ]
  in
  let first r =
    let v =
      Ops.alu
        (Ops.alu r Mul [ Ops.int ~dtype:Int32 3 ])
        Add
        [ Ops.int ~dtype:Int32 100 ]
    in
    Ops.alu v Add [ v ]
  in
  let second r = Ops.alu r Add [ Ops.int ~dtype:Int32 1 ] in
  Linearizer.linearize
    (Ops.sink
       ~kernel:(Ops.kernel_info ~name:"sibling_loops" ())
       [ loop 4 0 first; loop 3 1 second ])

(* A loop that reads a value loaded before it only at its start, then declares
   values of its type: [2 * (2 * (x + c))^2], each value read twice so that it
   is declared. The value read at the start is read again on the next trip. *)
let early_read =
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

(* The recorded kernels compiled whole, cut into pieces with the kernels of
   [clang_in_pieces]'s own tests, and for Windows. *)
let host_rows = batched host (compile_all (List.map kernel clang_rows))

let pieces_rows =
  batched pieces
    (compile_all
       (List.map kernel clang_rows
       @ [
           looped_chain outer_chain;
           sibling_loops;
           early_read;
           chain chain_steps;
         ]))

let windows_rows =
  batched windows_native (compile_all (List.map kernel clang_rows))

let sums_a_buffer r () =
  let out =
    Run.on_host (loaded r "clang_sum")
      [ (1, floats (Array.init 256 Float.of_int)) ]
  in
  equal values (floats [| 32640. |]) (slot 0 out)

(* The operand of the maximum is the int one above the least, which the source
   writes as a literal. *)
let takes_the_maximum_of_a_literal r () =
  let k = loaded r "clang_inline_const_alu" in
  let max_of x = slot 0 (Run.on_host k [ (1, ints [| x |]) ]) in
  equal values (ints [| 1 |]) (max_of 1);
  equal values (ints [| -0x7fffffff |]) (max_of (-0x80000000))

let stores_where_its_gate_holds r () =
  let out =
    Run.on_host
      (loaded r "clang_gated_store_in_loop")
      [ (0, ints (Array.make 16 (-1))) ]
  in
  equal values
    (ints (Array.init 16 (fun i -> if i < 8 then i else -1)))
    (slot 0 out)

let loads_zero_outside_its_padding r () =
  let x = Array.init 14 (fun i -> Float.of_int (i + 1)) in
  let out = Run.on_host (loaded r "clang_padded") [ (1, floats x) ] in
  let expected =
    Array.init 16 (fun i -> if i = 0 || i = 15 then 1. else x.(i - 1) +. 1.)
  in
  equal values (floats expected) (slot 0 out)

let loops_until_its_test_fails r () =
  let k = loaded r "clang_unbounded_loop" in
  let after start = slot 0 (Run.on_host k [ (0, ints [| start |]) ]) in
  equal values ~msg:"from 0" (ints [| 10 |]) (after 0);
  equal values ~msg:"from 12, the body runs once" (ints [| 13 |]) (after 12)

let reads_a_constant_table r () =
  let out = Run.on_host (loaded r "clang_table") [] in
  equal values (ints [| 0; 127; 128; 255 |]) (slot 0 out)

let passes_variables_of_32_and_64_bits r () =
  let out =
    Run.on_host
      ~vars:[ ("start", 2); ("offset", 1 lsl 40) ]
      (loaded r "clang_scalar_params")
      []
  in
  equal values (ints (Array.init 4 (fun i -> i + 2 + (1 lsl 40)))) (slot 0 out)

let runs_custom_code r () =
  let x = [| -2.5; 0.; 3.; -0. |] in
  let out = Run.on_host (loaded r "clang_custom") [ (1, floats x) ] in
  equal values (floats (Array.map Float.abs x)) (slot 0 out)

(* The kernel reads the four chars 1, 2, 3 and 4 as one little-endian uint,
   clears its low byte and stores it to the first char. *)
let reads_chars_as_a_uint name r () =
  let out = Run.on_host (loaded r name) [ (0, ints [| 1; 2; 3; 4 |]) ] in
  equal values (ints [| 0; 2; 3; 4 |]) (slot 0 out)

(* The registers hold the uints 1 and 2, read as one little-endian ulong. *)
let reads_registers_as_a_ulong r () =
  let out = Run.on_host (loaded r "clang_register_cast") [] in
  equal values [| `Int (Bigint.of_string "0x200000001") |] (slot 0 out)

let picks_a_lane_by_a_variable r () =
  let k = loaded r "clang_dynamic_lane" in
  List.iter
    (fun lane ->
      equal values ~msg:(string_of_int lane)
        (floats [| Float.of_int (lane + 1) |])
        (slot 0 (Run.on_host ~vars:[ ("lane", lane) ] k [])))
    [ 0; 1; 2; 3 ]

(* C computes (uchar)16 + 86, times 3, in int, which gives 306 where the
   kernel's unsigned chars wrap to 50. *)
let wraps_each_operation_on_chars r () =
  let out =
    Run.on_host
      (loaded r "clang_dtype_u8")
      [ (1, floats (Array.make 16 16.)); (2, ints (Array.make 16 86)) ]
  in
  equal values
    (floats (Array.make 16 (Float.of_int ((16 + 86) * 3 land 0xff))))
    (slot 0 out)

(* The claims on known values, each run whole and cut into pieces. *)
let runs =
  [
    ("sums a buffer", sums_a_buffer);
    ("takes the maximum of a load and a literal", takes_the_maximum_of_a_literal);
    ("stores only where its gate holds", stores_where_its_gate_holds);
    ("loads zero outside its padding", loads_zero_outside_its_padding);
    ("loops until its bottom test fails", loops_until_its_test_fails);
    ("reads a constant table", reads_a_constant_table);
    ("passes variables of 32 and 64 bits", passes_variables_of_32_and_64_bits);
    ("runs custom code", runs_custom_code);
    ( "reads four chars as a uint through a cast of their address",
      reads_chars_as_a_uint "clang_packed_cast" );
    ( "reads four chars as a uint through a bitcast of their address",
      reads_chars_as_a_uint "clang_packed_bitcast" );
    ("picks a lane of a vector by a variable", picks_a_lane_by_a_variable);
    ( "reads two uint registers as a ulong through a cast of their address",
      reads_registers_as_a_ulong );
    ("wraps each operation on unsigned chars", wraps_each_operation_on_chars);
  ]

(* Laws over the recorded kernels *)

(* An element on which a kernel's arithmetic is exact and defined: a float is a
   multiple of a half below 16, so that sums and products round nowhere and
   convert to every integer type, and an integer is small and never zero. A
   recorded kernel is linearized without tolk's code generation, whose rewrites
   define a signed overflow, a division by zero and a shift past the width
   ({!Cstyle}'s preamble), so its source leaves them to C, which leaves them
   undefined. Tolk_engine's kernels on the host check those values through code
   generation. *)
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
      | Ops.Param ({ addrspace = Global; _ } as p) when Shape.max_shape u <> []
        ->
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

let cons g rest = Gen.bind g (fun x -> Gen.map (fun r -> x :: r) rest)

let draw_inputs uops =
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

let interpreted_writes uops (buffers, vars) =
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

(* Each draw is interpreted once, and the host's program, the program cut into
   pieces and, on an x86_64 host, the program for Windows each write what the
   interpreter computes. A Windows object that calls a function of the process,
   as Clang calls [fma] or [memcpy], calls it by Windows' convention, which only
   Windows runs: its comparison is left out. *)
let writes_what_it_computes row =
  let uops = kernel row in
  let whole = lazy (program host_rows uops)
  and cut = lazy (program pieces_rows uops)
  and for_windows = lazy (program windows_rows uops) in
  let windows_runs =
    lazy
      (on_x86_64 ()
      && undefined_in (Uops.binary_of (Lazy.force for_windows)) = [])
  in
  let writes = list (pair int values) in
  prop ~count:10 (row "case") (draw_inputs uops)
    (fun ((buffers, vars) as inputs) ->
      let want = interpreted_writes uops inputs in
      let run p = Run.on_host ~vars (Lazy.force p) buffers in
      equal ~msg:"whole" writes want (run whole);
      equal ~msg:"cut into pieces" writes want (run cut);
      if Lazy.force windows_runs then
        equal ~msg:"for Windows" writes want (run for_windows))

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
  List.fold_right
    (fun (slot, dt, n) ->
      cons
        (Gen.map
           (fun a -> (slot, a))
           (Gen.array ~size:(Gen.constant n) (narrow_element narrow dt))))
    (buffers_of uops) (Gen.constant [])
  |> Gen.map (fun buffers -> (buffers, []))
  |> Gen.with_pp pp_inputs

(* The recorded kernels over a narrow type. Naming their tests reads
   kernels.golden as the suite is declared. *)
let narrow_rows =
  List.filter
    (fun row -> interpreted row && Option.is_some (narrow_of (kernel row)))
    clang_rows

(* Each draw is interpreted once, and the host's program and the program cut
   into pieces each write what the interpreter computes: a piece that held a
   narrow value at a wider type across pieces would neither wrap nor round. *)
let wraps_and_rounds_as_the_interpreter row =
  let uops = kernel row in
  let narrow = Option.get (narrow_of uops) in
  let whole = lazy (program host_rows uops)
  and cut = lazy (program pieces_rows uops) in
  let writes = list (pair int values) in
  prop ~count:20 (row "case") (draw_narrow_inputs narrow uops)
    (fun ((buffers, _) as inputs) ->
      let want = interpreted_writes uops inputs in
      let run p = Run.on_host (Lazy.force p) buffers in
      equal ~msg:"whole" writes want (run whole);
      equal ~msg:"cut into pieces" writes want (run cut))

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

(* Each constant of each integer type stored twice, into the buffer of its
   type's slot: as the constant of the type ([Ops.const]), and as a conversion
   of its own literal to the type. *)
let stores_constants (per_type : (Dtype.t * Dtype.value list) list) =
  let stores slot (dt, cs) =
    let n = 2 * List.length cs in
    let at i =
      Ops.index
        (Shape.param ~shape:[ Int n ] slot dt)
        [ Ops.int ~dtype:Int32 i ]
    in
    List.concat
      (List.mapi
         (fun i c ->
           let c = (c :> Dtype.const) in
           [
             Ops.store (at (2 * i)) (Ops.const ~dtype:dt c);
             Ops.store (at ((2 * i) + 1)) (Ops.cast (Ops.const c) dt);
           ])
         cs)
  in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"constants" ())
    (List.concat (List.mapi stores per_type))

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

let pp_constants ppf per_type =
  List.iter
    (fun (dt, cs) ->
      Format.fprintf ppf "%a [%a]@ " Dtype.pp dt
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           Dtype.pp_const)
        (cs :> Dtype.const list))
    per_type

(* A few constants of every integer type. *)
let constants_of_ints =
  List.fold_right
    (fun dt rest ->
      cons
        (Gen.map
           (fun cs -> (dt, cs))
           (Gen.list ~size:(Gen.int_range 1 4) int_constant))
        rest)
    Dtype.ints (Gen.constant [])
  |> Gen.with_pp pp_constants

let stores_ints_as_their_types_hold_them per_type =
  let beyond_64_bits = function
    | `Int n -> (
        match Bigint.to_int64 n with
        | _ -> false
        | exception Bigint.Overflow -> true)
    | `Float x -> Float.abs x >= 0x1p64
    | `Bool _ -> false
  in
  let cs = List.concat_map snd per_type in
  cover "a constant beyond 64 bits" (List.exists beyond_64_bits cs);
  cover "a float" (List.exists (function `Float _ -> true | _ -> false) cs);
  let k = program host (Linearizer.linearize (stores_constants per_type)) in
  let out = Run.on_host k [] in
  List.iteri
    (fun s (dt, cs) ->
      let expected = List.concat_map (fun c -> [ held dt c; held dt c ]) cs in
      equal values
        ~msg:(Format.asprintf "%a" Dtype.pp dt)
        (Array.of_list expected) (slot s out))
    per_type

(* An infinity or a NaN has no integer value: its conversion to an integer type
   is the program's to run, and the source compiles. *)
let special_cases =
  List.concat_map
    (fun dt -> List.map (fun x -> (dt, x)) [ infinity; neg_infinity; nan ])
    Dtype.[ Int8; Uint16; Int32; Uint64 ]

let special (dt, x) =
  let at =
    Ops.index (Shape.param ~shape:[ Int 1 ] 0 dt) [ Ops.int ~dtype:Int32 0 ]
  in
  Linearizer.linearize
    (Ops.sink
       ~kernel:(Ops.kernel_info ~name:"special" ())
       [ Ops.store at (Ops.cast (Ops.float x) dt) ])

let specials = batched host (compile_all (List.map special special_cases))

(* Selections by a comparison

   [data0[0] = data1[0] < c ? data1[0] : c] at [dt]: a select of a value and a
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

let barrier reg = Printf.sprintf "__asm__(\"\" : \"+%s\"(z))" reg

(* Each Clang target, its register class for a float, the float types it
   compares in, those whose compared zeros its backend lowers to a minimum or a
   maximum, and whether it lowers so a select against any constant, whose
   minimum or maximum then quiets a signalling NaN: each compares a bfloat16 in
   float32. *)
let clang_targets =
  Dtype.
    [
      ( "arm64",
        arm64,
        "w",
        [ Float16; Float32; Float64 ],
        [ Float16; Float32; Float64 ],
        true );
      ("x86_64", clang, "x", [ Float16; Float32; Float64 ], [ Float16 ], false);
      ( "riscv64",
        riscv64,
        "f",
        [ Float16; Float32; Float64 ],
        [ Float16; Float32; Float64 ],
        false );
    ]

let per_target =
  List.concat_map
    (fun (name, r, reg, dts, zeros, constants) ->
      List.map (fun dt -> (name, r, reg, dt, List.mem dt zeros, constants)) dts)
    clang_targets

let target_name (name, _, _, dt, _, _) = name ^ ", " ^ Dtype.name dt

(* [data0[0] = c ? x : z], or [c ? z : x], stored at [stored]: [x] the value
   [data1[0]], loaded at [loaded] and converted to [selected], [c] a comparison
   of it at [compared] with the zero [z]. rune compares a float16 in float32 and
   selects between the halves. *)
let converted_select (loaded, compared, selected, stored) z ~zero_first
    ~value_first =
  let zero = Ops.int ~dtype:Int32 0 in
  let at slot dt = Ops.index (Shape.param ~shape:[ Int 1 ] slot dt) [ zero ] in
  let x = Ops.load (at 1 loaded) [] in
  let at_type dt u =
    if Dtype.equal dt (Ops.dtype u) then u else Ops.cast u dt
  in
  let cx = at_type compared x and cz = Ops.float ~dtype:compared z in
  let c = if zero_first then Ops.lt cz cx else Ops.lt cx cz in
  let sx = at_type selected x and sz = Ops.float ~dtype:selected z in
  let s = if value_first then Ops.where c sx sz else Ops.where c sz sx in
  let kernel =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"converted_select" ())
      [ Ops.store (at 0 stored) (at_type stored s) ]
  in
  Linearizer.linearize
    (Codegen.full_rewrite_to_sink ~optimize:false kernel (Lazy.force host))

(* Each conversion, as the types loaded, compared, selected and stored, each
   zero, operand order and arm order. *)
let converted_cases =
  let product xs ys =
    List.concat_map (fun x -> List.map (fun y -> (x, y)) ys) xs
  in
  let bools = [ true; false ] in
  product
    Dtype.
      [
        (Float16, Float32, Float16, Float32);
        (Float16, Float64, Float16, Float64);
        (Float16, Float32, Float64, Float64);
        (Float32, Float64, Float32, Float64);
        (Float32, Float32, Float64, Float64);
        (Float64, Float64, Float32, Float32);
        (Float32, Float32, Float32, Float64);
      ]
    (product [ 0.; -0. ] (product bools bools))

let converted_name
    ((loaded, compared, selected, stored), (z, (zero_first, value_first))) =
  let cmp = if zero_first then "z < x" else "x < z" in
  let arms = if value_first then "x : z" else "z : x" in
  Printf.sprintf
    "%s, compared at %s, selected at %s, stored at %s, z = %h, %s ? %s"
    (Dtype.name loaded) (Dtype.name compared) (Dtype.name selected)
    (Dtype.name stored) z cmp arms

let converted (types, (z, (zero_first, value_first))) =
  converted_select types z ~zero_first ~value_first

(* The unsigned integer type of [dt]'s width, which holds its bits. *)
let bits_of dt =
  match Dtype.itemsize dt with
  | 2 -> Dtype.Uint16
  | 4 -> Dtype.Uint32
  | _ -> Dtype.Uint64

(* [out[i] = x[i] < c ? x[i] : c], or [c < x[i] ? x[i] : c] where [max], over
   rows of four elements of [dt], [i] the row's [r] and the lane's [l], through
   the whole lowering: [c] is the weak literal 80, or [80 + l] where [lanes].
   Where [split], each thread computes a row as one vector, and [80 + l] is a
   vector of four constants. *)
let clamp_kernel dt ~max ~lanes ~split rows =
  let n = 4 * rows in
  let out = Shape.param ~shape:[ Int n ] 0 dt
  and x = Shape.param ~shape:[ Int n ] 1 dt in
  let r = Ops.range ~axis_type:Weak (Int rows) [ 0 ]
  and l = Ops.range ~axis_type:Weak (Int 4) [ 1 ] in
  let i = Ops.O.((r * Ops.int 4) + l) in
  let v = Ops.index x [ i ] in
  let c =
    if lanes then Ops.O.(Ops.cast l dt + Ops.float 80.) else Ops.float 80.
  in
  let s = Ops.where (if max then Ops.lt c v else Ops.lt v c) v c in
  (* a kernel that reads no lane but in [i] has one axis of [n] *)
  let axis = if lanes then 1 else 0 in
  let opts_to_apply =
    if split then
      [ Opt.Split { axis; amount = 4; target = Upcast; top = false } ]
    else []
  in
  Ops.sink
    ~kernel:(Ops.kernel_info ~opts_to_apply ())
    [ Ops.end_ (Ops.store (Ops.index out [ i ]) s) [ r; l ] ]

(* The bits of a signalling and a quiet NaN of [dt], of either sign. *)
let nans = function
  | Dtype.Float16 -> [ 0x7d00L; 0x7e00L; 0xfd00L; 0xfe00L ]
  | Float32 -> [ 0x7fa0_0000L; 0x7fc0_0000L; 0xffa0_0000L; 0xffc0_0000L ]
  | Float64 ->
      [
        0x7ff4_0000_0000_0000L;
        0x7ff8_0000_0000_0000L;
        0xfff4_0000_0000_0000L;
        0xfff8_0000_0000_0000L;
      ]
  | dt -> invalid_arg ("no NaNs of " ^ Dtype.name dt)

let clamp_cases =
  let bools = [ false; true ] in
  List.concat_map
    (fun dt ->
      List.concat_map
        (fun max ->
          List.concat_map
            (fun lanes -> List.map (fun split -> (dt, max, lanes, split)) bools)
            bools)
        bools)
    Dtype.[ Float16; Float32; Float64 ]

let clamp_name (dt, max, lanes, split) =
  Printf.sprintf "%s, %s, c %s%s" (Dtype.name dt)
    (if max then "c < x ? x : c" else "x < c ? x : c")
    (if lanes then "80 + its lane" else "a literal")
    (if split then ", a vector a row" else "")

(* The program of a clamp kernel of two rows, through the whole lowering. *)
let clamp_program (r : Renderer.t) (dt, max, lanes, split) =
  Codegen.compile (Codegen.linearize (clamp_kernel dt ~max ~lanes ~split 2) r) r

(* The selections' kernels, compiled in one run of Clang. *)
let selects =
  batched host (fun r ->
      List.iter
        (fun dt -> ignore (Run.program r (const_select dt 0.)))
        Dtype.[ Float16; Float32; Float64 ];
      ignore (Run.program r (const_select Float64 1.5));
      List.iter (fun c -> ignore (Run.program r (converted c))) converted_cases;
      List.iter (fun c -> ignore (clamp_program r c)) clamp_cases)

(* A comparison with a NaN, signalling or quiet, is false, so the select is [c]
   there; elsewhere it picks by the comparison. Values pass as their bits, which
   {!Dtype.bitcast} keeps, a signalling NaN's included. *)
let clamps_as_selected (dt, max, lanes, split) =
  let bits = bits_of dt in
  let as_bits = function
    | `Float f -> (
        match Dtype.bitcast dt bits (`Float f) with
        | `Int z -> z
        | _ -> invalid_arg "bits are an integer")
    | _ -> invalid_arg "a float"
  in
  let as_float z = Dtype.bitcast bits dt (`Int z) in
  let nans =
    List.map (fun n -> as_float (Bigint.of_int64_unsigned n)) (nans dt)
  in
  let xs =
    Array.of_list (nans @ [ `Float 1.; `Float 100.; `Float 81.; List.hd nans ])
  in
  let selected i x =
    let c = 80. +. if lanes then Float.of_int (i mod 4) else 0. in
    match x with
    | `Float f when if max then c < f else f < c -> x
    | _ -> `Float c
  in
  let hex v =
    Printf.sprintf "0x%Lx"
      (Bigint.to_int64 (Bigint.signed_extract (as_bits v) 0 64))
  in
  let k = clamp_program (Lazy.force selects) (dt, max, lanes, split) in
  let out = Run.on_host k [ (1, xs) ] in
  equal (list string)
    (Array.to_list (Array.mapi (fun i x -> hex (selected i x)) xs))
    (List.map hex (Array.to_list (List.assoc 0 out)))

let opaque_to_extremes (_, r, reg, dt, zeros, constants) =
  let says msg want uops =
    let src = render r uops in
    if want then Windtrap.contains ~msg ~sub:(barrier reg) src
    else lacks ~msg (barrier reg) src
  in
  says "0" zeros (const_select dt 0.);
  says "-0" zeros (const_select dt (-0.));
  says "1.5" constants (const_select dt 1.5);
  says "a mask" false (zero_masked dt);
  let k = clamp_kernel dt ~max:false ~lanes:true ~split:true 2 in
  says "a vector of constants" constants
    (Ops.src (Ops.nth (Codegen.linearize k r) 1))

let one_float out =
  match List.assoc 0 out with [| `Float c |] -> c | _ -> failf "one float"

(* Selections of 16-bit floats

   [data0[0] = c ? x : y] and [data0[1] = c ? data4[1] : 0], [x] the bits of
   [data1[0]] as a [dt], [y] [data2[0]] narrowed to [dt] and [c] [data3[0]]: a
   selection and a gated load of [dt], stored as their bits. The golden
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
   [__fp16] or a [__bf16]: x86_64 and arm64 hold a bfloat16 as its bits. *)
let held_16 =
  Dtype.
    [
      ("arm64", arm64, [ Float16 ]);
      ("x86_64", clang, [ Float16 ]);
      ("riscv64", riscv64, [ Float16; Bfloat16 ]);
    ]
  |> List.concat_map (fun (name, r, dts) ->
      List.map (fun dt -> (name ^ ", " ^ Dtype.name dt, r, dt)) dts)

(* x86_64 Windows

   An object for x86_64 Windows calls by Windows' convention, where a [long] has
   32 bits, and is called by System V's. On an x86_64 host of another system, a
   kernel that calls none of the process's functions runs as there: the object's
   own calls follow one convention. *)

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

(* [source] written for 64-bit [long]s and no System V attribute: [long long] is
   [long], and a literal [5ll] or [0x7ffull] is [5l] or [0x7fful]. *)
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

(* The words of [source] that spell a 64-bit integer as LP64 does: a [long] not
   in a [long long], and an integer literal that ends in one [l]. *)
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

(* A source that compiles only for the Windows triple, without a stack probe:
   [_WIN64] is the triple's, and a frame of 16 KiB would call [___chkstk_ms] on
   the way in. *)
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
          let kernel_line =
            List.find (occurs "void E_16_4(") (String.split_on_char '\n' src)
          in
          starts_with ~prefix:sysv_abi kernel_line);
      test "a 64-bit integer is a long long and its literals end in ll"
        (fun () ->
          let src = on_windows (find_case "clang_dtype_i64") in
          Windtrap.contains ~sub:"long long " src;
          Windtrap.contains ~sub:"*3ll)" src);
      test "an object is compiled for Windows' triple and calls no stack probe"
        (fun () ->
          Clang.here "x86_64-windows,x86-64";
          let r = windows "x86-64" in
          equal (list string) []
            (undefined_in (Renderer.Compiler.compile r.compiler windows_only)));
      group "every kernel the interpreter cannot run compiles"
        (List.filter_map
           (fun row ->
             if interpreted row then None
             else
               Some
                 (test (row "case") (fun () ->
                      if not (on_x86_64 ()) then
                        skip ~reason:"the host is not x86_64" ();
                      ignore (program windows_rows (kernel row)))))
           clang_rows);
    ]

let clang_group =
  group "clang"
    [
      test "an operation computes at most 64 lanes of a vector at once"
        (fun () -> equal int 64 clang.vector_lanes);
      cases
        "a selection and a gated load of a 16-bit float held as a storage \
         format choose between bits"
        ~id:(fun (name, _, _) -> name)
        held_16
        (fun (_, r, dt) ->
          let src = render r (selected dt) in
          let t = if Dtype.equal dt Float16 then "__fp16" else "__bf16" in
          let pick =
            Printf.sprintf "__builtin_bit_cast(%s, (unsigned short)((" t
          in
          equal ~msg:("in\n" ^ src) int 2 (occurrences src pick));
      group "selections by a comparison"
        [
          cases
            "a float constant a comparison reads is opaque to a backend that \
             lowers it to an extreme, and a literal elsewhere"
            ~id:target_name per_target opaque_to_extremes;
          test
            "a select of a value and a constant picks the constant at a \
             signalling NaN, on the host" (fun () ->
              let k = program selects (const_select Float64 1.5) in
              let snan = Int64.float_of_bits 0x7ff4_0000_0000_0000L in
              equal float_exact 1.5
                (one_float (Run.on_host k [ (1, floats [| snan |]) ])));
          cases
            "a select of a value and a constant by their comparison picks the \
             constant at every NaN, lowered whole, on the host"
            ~id:clamp_name clamp_cases clamps_as_selected;
          cases
            "a select of a value and a zero picks the zero at -0., on the host"
            ~id:Dtype.name
            Dtype.[ Float16; Float32; Float64 ]
            (fun dt ->
              let k = program selects (const_select dt 0.) in
              equal float_exact 0.
                (one_float (Run.on_host k [ (1, floats [| -0. |]) ])));
          cases
            "a select of a value and a zero, compared and selected at two \
             types, picks as the select does, on the host"
            ~id:converted_name converted_cases
            (fun ((_, (z, (zero_first, value_first))) as c) ->
              let k = program selects (converted c) in
              List.iter
                (fun x ->
                  let c = if zero_first then z < x else x < z in
                  let want = if c = value_first then x else z in
                  equal ~msg:(Printf.sprintf "at %h" x) float_exact want
                    (one_float (Run.on_host k [ (1, floats [| x |]) ])))
                [ -0.; 0.; Float.nan; -1.; 1. ]);
        ];
      group "long kernels"
        [
          test "a long kernel is written as several functions" (fun () ->
              let source program = render (renderer_for clang_target) program in
              Windtrap.contains ~sub:frame (source (Lazy.force long_program));
              lacks frame (source (looped_chain outer_chain)));
          test "a GPU's kernel stays one function" (fun () ->
              List.iter
                (fun (name, renderer) ->
                  lacks ~msg:name frame
                    (render (renderer ()) (Lazy.force long_program)))
                (List.filter (fun (n, _) -> n <> "clang") renderers));
          test "computes a long chain of values read once" (fun () ->
              computes_its_chain long_chain
                (program host (Lazy.force long_program)));
        ];
      windows_group;
      cases "raises Invalid_argument on an architecture it cannot read"
        ~id:Fun.id [ ""; "x86_64"; "riscv64"; "x86_64-windows"; "mips,generic" ]
        (fun arch ->
          rejects (fun () -> Cstyle.clang (target "CPU" "CLANG" arch)));
      group "on the host"
        (List.map (fun (name, f) -> test name (f host_rows)) runs
        @ [
            prop ~count:30 "stores each integer constant as its type holds it"
              constants_of_ints stores_ints_as_their_types_hold_them;
            cases "converts an infinity or a NaN to an integer type"
              ~id:(fun (dt, x) -> Format.asprintf "%a %h" Dtype.pp dt x)
              special_cases
              (fun c -> ignore (program specials (special c)));
            group
              "every kernel the interpreter runs writes what it computes, \
               whole, cut into pieces and for Windows"
              (List.map writes_what_it_computes
                 (List.filter interpreted clang_rows));
            group
              "a kernel over a narrow type wraps and rounds as the \
               interpreter, whole and cut into pieces"
              (List.map wraps_and_rounds_as_the_interpreter narrow_rows);
            group "every kernel the interpreter cannot run compiles"
              (List.filter_map
                 (fun row ->
                   if interpreted row then None
                   else
                     Some
                       (test (row "case") (fun () ->
                            ignore (program host_rows (kernel row)))))
                 clang_rows);
          ]);
    ]

(* Every recorded kernel of a GPU through its toolchain *)

(* The recorded kernels code generation never hands to CUDA, whose sources NVRTC
   rejects: CUDA's vectors are structs, without casts and with their lanes as
   members, and code generation casts and reads them lane by lane. *)
let rejected =
  [
    ("cuda_dynamic_lane", "reads a lane by a variable, which NVRTC rejects");
    ("cuda_vector_cast", "casts a vector, which NVRTC rejects");
  ]

let compiles_with_its_toolchain row =
  let t =
    test (row "case") (fun () ->
        let r = compiling (target_of_row row) in
        ignore (compiles r (render r (kernel row))))
  in
  match List.assoc_opt (row "case") rejected with
  | Some reason -> xfail ~reason t
  | None -> t

let every_kernel_compiles renderer =
  group "every recorded kernel compiles with the target's toolchain"
    (List.map compiles_with_its_toolchain (rows_of renderer))

let through_matcher (r : Renderer.t) sink =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() sink
    (After_sources r.extra_matcher)

(* Metal *)

(* Metal has no trunc of a bfloat: it truncates a float, so the extra matcher
   truncates a bfloat16 in float32. *)
let truncated_bf16 =
  let zero = Ops.int ~dtype:Int32 0 in
  let at slot =
    Ops.index (Shape.param ~shape:[ Int 1 ] slot Bfloat16) [ zero ]
  in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"trunc_bf16" ())
    [ Ops.store (at 0) (Ops.trunc (Ops.load (at 1) [])) ]

let truncs sink =
  List.filter
    (fun u -> Op.equal (Ops.op u) Trunc)
    (Ops.toposort ~calls:Enter sink)

(* The graph a kernel reaches the renderer as, after the rewrites that commit
   its constants. *)
let compiles_its_truncation () =
  let sink =
    List.fold_left
      (fun sink m ->
        Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() sink
          (After_sources m))
      (through_matcher (metal ()) truncated_bf16)
      [
        Uop_weak.pm_commit_weak; Uop_weak.pm_lower_weak; Uop_weak.pm_cast_const;
      ]
  in
  let metal = compiling metal_target in
  ignore (compiles metal (render metal (Linearizer.linearize sink)))

(* Tensor cores on Metal

   A warp's bfloat16 product, [D = A B], then [D + 0 C] with a left operand of
   literal zeros: Apple7's compiler service failed on a product of bfloat16
   matrices with an operand the compiler knows, into float32 and into bfloat16
   alike. *)

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
  let pair buf =
    Shape.stack (List.init 2 (fun e -> Ops.load (Ops.index buf [ at e ]) []))
  in
  let literal dt =
    Shape.stack (List.init 2 (fun _ -> Ops.float ~dtype:dt 0.))
  in
  let product x y acc = Ops.wmma x y ~acc ~dims:(Tc.dims tc) ~threads:lanes in
  let d = product (pair a) (pair b) (literal tc.dtype_out) in
  let d = product (literal Bfloat16) (pair c) d in
  let stores =
    List.init 2 (fun e ->
        Ops.store
          (Ops.index out [ at e ])
          (Ops.index d [ Ops.int ~dtype:Int32 e ]))
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

let metal_group =
  group "metal"
    [
      cases "raises Invalid_argument on an Apple family that is no integer"
        ~id:Fun.id [ "Apple"; "AppleX" ] (fun arch ->
          rejects (fun () -> Cstyle.metal (target "METAL" "METAL" arch)));
      test "truncates a bfloat16 in float32" (fun () ->
          equal (list Dtypes.dtype) [ Dtype.Float32 ]
            (List.map Ops.dtype
               (truncs (through_matcher (metal ()) truncated_bf16))));
      test "compiles a kernel that truncates a bfloat16" compiles_its_truncation;
      cases
        "a bfloat16 product with an operand of literal zeros holds the product"
        ~id:(fun (tc : Tc.t) -> Format.asprintf "into %a" Dtype.pp tc.dtype_out)
        bf16_cores multiplies_by_zeros_on_metal;
      every_kernel_compiles "METAL";
    ]

(* Conversions to and from bfloat16 *)

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
  around lo
  @ around (Bigint.pred hi)
  @ [ Bigint.zero; Bigint.one; Bigint.neg Bigint.one ]
  @ positive
  @ List.map Bigint.neg positive
  |> List.filter (fun v -> Bigint.compare v lo >= 0 && Bigint.compare v hi < 0)

(* What slot 0 of [k] holds, given slot 1's [inputs]. *)
let written_by k inputs =
  Interpreter.writes ~buffers:[ (1, Array.of_list inputs) ] k
  |> List.filter_map (fun (slot, _, v) -> if slot = 0 then Some v else None)

(* CUDA *)

(* Each integer source: a name, the type loaded, the conversion of a load, and
   the integer it converts, of the integer loaded. Below sm_90 the kernel
   converts an integer within 2^24 through a float32, which holds it exactly: an
   int32 of 24 bits is within it, and one of 25 bits, whose float32 would round
   [2^24 + 2^16 + 1] to bfloat16's tie [2^24 + 2^16], then down, where one
   rounding takes it up, past it. *)
let masked bits x = directly Ops.O.(x land Ops.int ~dtype:Int32 bits)
let masked_value bits v = Bigint.of_int (Bigint.to_int v land bits)

let integer_sources =
  List.map
    (fun dt -> (Format.asprintf "%a" Dtype.pp dt, dt, directly, Fun.id))
    Dtype.[ Bool; Int8; Uint8; Int16; Uint16; Int32; Uint32; Int64; Uint64 ]
  @ [
      ( "an int32 of 24 bits",
        Dtype.Int32,
        masked 0xffffff,
        masked_value 0xffffff );
      ( "an int32 of 25 bits",
        Dtype.Int32,
        masked 0x1ffffff,
        masked_value 0x1ffffff );
    ]

(* The types the conversions to bfloat16 of a kernel converting [src] read,
   after [arch]'s rewrites. *)
let bf16_operands arch src =
  let cuda = renderer_for (target "CUDA" "CUDA" arch) in
  List.filter_map
    (fun u ->
      if Dtype.equal (Ops.dtype u) Bfloat16 then Some (Ops.dtype (Ops.nth u 0))
      else None)
    (bf16_conversions
       (through_matcher cuda (converting src Bfloat16 2 directly)))

let rounds_once_on_cuda (arch, (_, src, f, value)) =
  let bool = Dtype.equal src Bool in
  let inputs = if bool then [ Bigint.zero; Bigint.one ] else bf16_inputs src in
  let elements =
    List.map
      (fun v -> if bool then `Bool (Bigint.sign v <> 0) else `Int v)
      inputs
  in
  let want = List.map (fun v -> `Float (exact_bf16 (value v))) inputs in
  let k = converting src Bfloat16 (List.length inputs) f in
  let cuda = renderer_for (target "CUDA" "CUDA" arch) in
  equal (list Dtypes.value) want (written_by (through_matcher cuda k) elements)

let cuda_group =
  group "cuda"
    [
      cases "raises Invalid_argument on an architecture without a capability"
        ~id:Fun.id [ "sm_"; "sm"; "sm_x" ] (fun arch ->
          rejects (fun () -> Cstyle.cuda (target "CUDA" "CUDA" arch)));
      cases "a device's kernel compiles to a cubin" ~id:fst
        [ ("CUDA", cuda_target); ("NV", target "NV" "CUDA" "sm_89") ]
        (fun (_, t) ->
          let r = compiling t in
          starts_with ~prefix:"\x7fELF" (compiles r (source_of_case "cuda_add")));
      test "writes the same source for the CUDA and NV devices" (fun () ->
          let cuda = cuda ()
          and nv = renderer_for (target "NV" "CUDA" "sm_89") in
          List.iter
            (fun row ->
              equal string ~msg:(row "case")
                (render cuda (kernel row))
                (render nv (kernel row)))
            (rows_of "CUDA"));
      cases "an integer converts to bfloat16 rounding once"
        ~id:(fun (arch, (name, _, _, _)) -> arch ^ " " ^ name)
        (List.concat_map
           (fun arch -> List.map (fun s -> (arch, s)) integer_sources)
           [ "sm_80"; "sm_89"; "sm_90" ])
        rounds_once_on_cuda;
      (* D499: below sm_90 CUDA converts an int to bfloat16 by three conversions
         to float, a compare and a select, where one conversion of a float32,
         which holds the int exactly, rounds the same in fewer instructions. *)
      cases
        "below sm_90 an integer within 2^24 converts to bfloat16 through a \
         float32"
        ~id:Fun.id [ "sm_80"; "sm_89" ] (fun arch ->
          equal (list Dtypes.dtype) [ Dtype.Float32 ] (bf16_operands arch Int16));
      test "leaves a bfloat16 truncation to htrunc" (fun () ->
          equal (list Dtypes.dtype) [ Dtype.Bfloat16 ]
            (List.map Ops.dtype
               (truncs (through_matcher (cuda ()) truncated_bf16))));
      every_kernel_compiles "CUDA";
    ]

(* HIP

   LLVM folds a float32 converted from an integer and narrowed to bfloat16 into
   one conversion from the integer, which AMD's backend cannot select for an
   [i1], and the compiler aborts. These kernels hold no conversion to or from
   bfloat16 for LLVM to fold, and their values are IEEE's. *)

(* The GPUs whose bfloat16 conversions are written on their bits. *)
let hip_archs = [ "gfx1100"; "gfx1201"; "gfx942"; "gfx950" ]

let through_hip arch sink =
  through_matcher (renderer_for (target "AMD" "HIP" arch)) sink

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

let on_archs =
  List.concat_map
    (fun arch -> List.map (fun c -> (arch, c)) conversions)
    hip_archs

let leaves_no_conversion (arch, (_, from, to_, f)) =
  let rewritten = through_hip arch (converting from to_ 2 f) in
  equal ~msg:"conversions of bfloat16" int 0
    (List.length (bf16_conversions rewritten))

let integers_round_once (arch, src) =
  let inputs = bf16_inputs src in
  let elements = List.map (fun v -> `Int v) inputs in
  let want = List.map (fun v -> `Float (exact_bf16 v)) inputs in
  List.iter
    (fun (name, route) ->
      let k = converting src Bfloat16 (List.length inputs) route in
      equal ~msg:name (list Dtypes.value) want
        (written_by (through_hip arch k) elements))
    [ ("directly", directly); ("through narrow", through_narrow) ]

let bools_are_exact arch =
  List.iter
    (fun (name, route) ->
      let k = converting Bool Bfloat16 2 route in
      equal ~msg:name (list Dtypes.value)
        [ `Float 0.; `Float 1. ]
        (written_by (through_hip arch k) [ `Bool false; `Bool true ]))
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
    0;
    0x80000000;
    1;
    0x7fff;
    0x8000;
    0x8001;
    0x18000;
    0x28000;
    0x3f808000;
    0x3f818000;
    0x3f808001;
    0xbf808000;
    0xbf818000;
    0x7f7f7fff;
    0x7f7f8000;
    0x7f7fffff;
    0x7f800000;
    0xff800000;
    0x7f800001;
    0x7fc00000;
    0xffc00001;
  ]

let narrows_float32_codes arch =
  let n = List.length float32_codes in
  let k = through_hip arch (converting Float32 Bfloat16 n directly) in
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
let widens_bf16_codes arch =
  let n = List.length bf16_codes in
  let k = through_hip arch (converting Bfloat16 Float32 n as_float32) in
  let elements = List.map bf16_of_bits bf16_codes in
  List.iter2
    (fun c got -> same_float Float32 (f32_of_bits (c lsl 16)) got)
    bf16_codes (written_by k elements)

let hip_group =
  group "hip"
    [
      cases "a bfloat16 conversion leaves LLVM no conversion to fold"
        ~id:(fun (arch, (name, _, _, _)) -> arch ^ " " ^ name)
        on_archs leaves_no_conversion;
      cases "an integer converts to bfloat16 rounding once"
        ~id:(fun (arch, dt) -> Format.asprintf "%s %a" arch Dtype.pp dt)
        (List.concat_map
           (fun arch -> List.map (fun dt -> (arch, dt)) Dtype.ints)
           hip_archs)
        integers_round_once;
      cases "a bool converts to bfloat16 exactly" ~id:Fun.id hip_archs
        bools_are_exact;
      cases "a float32 converts to its nearest bfloat16" ~id:Fun.id hip_archs
        narrows_float32_codes;
      cases "a bfloat16 converts to float32 exactly" ~id:Fun.id hip_archs
        widens_bf16_codes;
      every_kernel_compiles "HIP";
    ]

let in_pieces_group =
  group "clang_in_pieces"
    [
      group "writes what the whole kernel writes"
        (List.map (fun (name, f) -> test name (f pieces_rows)) runs
        @ [
            test "computes a chain too deep for one expression"
              (computes_a_long_chain pieces_rows);
          ]);
      test "reads a value of the outer scope on every trip" (fun () ->
          computes_its_chain outer_chain
            (program pieces_rows (looped_chain outer_chain)));
      test "keeps a value read early in a loop for its next trip" (fun () ->
          let xs = [| 10; -2; 0; 7 |] and c = 3 in
          let out =
            Run.on_host
              (program pieces_rows early_read)
              [ (1, ints xs); (2, ints [| c |]) ]
          in
          let expected x = 2 * (2 * (x + c)) * (2 * (x + c)) in
          equal values (ints (Array.map expected xs)) (slot 0 out));
      test "keeps two loops whose indices render as one name" (fun () ->
          let out =
            Run.on_host
              (program pieces_rows sibling_loops)
              [ (0, ints [| 0; 0; 0; 0 |]); (1, ints [| 9; 9; 9; 9 |]) ]
          in
          equal values ~msg:"first loop"
            (ints [| 200; 206; 212; 218 |])
            (slot 0 out);
          equal values ~msg:"second loop" (ints [| 1; 2; 3; 9 |]) (slot 1 out));
    ]

let without_compiler =
  group "without_compiler"
    [
      test "raises Invalid_argument on a target of another renderer" (fun () ->
          rejects (fun () ->
              Cstyle.without_compiler (target "CPU" "PTX" "sm_89")));
    ]

let () =
  exit
    (run "Tolk.Cstyle"
       [
         render_group;
         rewrites;
         declarations;
         code_for_op;
         param_name;
         clang_group;
         metal_group;
         cuda_group;
         hip_group;
         in_pieces_group;
         without_compiler;
       ])
