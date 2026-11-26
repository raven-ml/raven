(* Tests of Tolk_engine: the targets it gives devices, and linked schedules,
   whose runs write what their programs compute, on the buffers bound to them,
   one run at a time and without allocating. *)

open Windtrap
open Tolk
module Engine = Tolk_engine
module Buffer = Rig.Buffer

let host = Rig.host
let floats xs = Array.map (fun x -> `Float x) xs

let value =
  let close = Testable.float_rel ~rel:1e-5 ~abs:1e-6 in
  Testable.make ~pp:(Testable.pp Dtypes.value) ~equal:(fun v0 v1 ->
      match (v0, v1) with
      | `Float x, `Float y ->
          (Float.is_nan x && Float.is_nan y) || Testable.equal close x y
      | v0, v1 -> Testable.equal Dtypes.value v0 v1)

let values = array value

(* The recorded programs name the devices of their generator: its CPU is also
   CPU:0, PYTHON is a software device, and its disk is named after a
   directory. *)
let named =
  ("CPU:0", host)
  :: ("PYTHON", List.assoc "CPU:3" (Run.devices ()))
  :: ("DISK:/tmp/tolk-schedule", Rig_disk.device)
  :: Run.devices ()

let device name =
  match List.assoc_opt name named with
  | Some d -> d
  | None -> invalid_arg ("no device is named " ^ name)

let devices = device

let clang = lazy (Cstyle.clang (Run.target host))

(* [linear] compiled for the targets of [devices]. *)
let compile ?(devices = devices) linear =
  Realize.compile_linear ~targets:(fun n -> Run.target (devices n)) linear

(* Devices *)

let target = Testable.make ~pp:Helpers.Target.pp ~equal:( = )

let host_target arch cpu =
  Result.get_ok (Helpers.Target.of_string ("CPU:CLANG:" ^ arch ^ "," ^ cpu))

let targets =
  group "target"
    [
      test
        "the host compiles C for Clang at its arch, tuned for its own \
         processor, naming its system under Windows, whose functions follow \
         a convention of their own" (fun () ->
          if not (Rig.Image.links host) then
            skip ~reason:"rig links no host code here" ();
          let system = if Sys.win32 then "-windows" else "" in
          let isa = Rig.instruction_set host in
          let cpu =
            match Rig.arch host with
            | "arm64" -> "generic"
            | "arm64+fp16" -> "generic,fp16"
            | level when isa = "x86_64" -> level
            | arch -> fail ("an arch this test has no target for: " ^ arch)
          in
          equal target
            (host_target (isa ^ system) (cpu ^ ",tune=native"))
            (Engine.target host));
      test "where rig links no host code, the host runs no program" (fun () ->
          if Rig.Image.links host then
            skip ~reason:"rig links host code here" ();
          raises_match
            (function
              | Device.Uncomputable { device; _ } -> device = Rig.name host
              | _ -> false)
            (fun () -> Engine.target host));
      test "a software device compiles as the host" (fun () ->
          equal target (Run.target host) (Run.target (device "CPU:1")));
      test "a software device is the host's part" (fun () ->
          equal string (Engine.part host) (Engine.part (device "CPU:1")));
      test "the disk runs no program" (fun () ->
          raises_match Exn.invalid_arg (fun () -> Engine.target Rig_disk.device));
      test
        "a target's renderer is made once, and shared by the devices that \
         compile for it" (fun () ->
          ignore (Run.target host);
          let name ppf (r : Renderer.t) = Format.pp_print_string ppf r.name in
          let renderer = Testable.make ~pp:name ~equal:( == ) in
          let made = Engine.renderer host in
          equal renderer made (Engine.renderer host);
          equal renderer made (Engine.renderer (device "CPU:1"));
          raises_match Exn.invalid_arg (fun () ->
              Engine.renderer Rig_disk.device));
      test "Metal compiles for its GPU family" (fun () ->
          let d = Gpus.gpu "Metal" in
          let t = Engine.target d in
          equal (pair string string) ("METAL", Rig.arch d) (t.device, t.arch));
    ]

(* Kernels on the host

   A kernel built from UOps: out[i] = a[i] * n + b[i] for [size] elements
   (default four), where out, a and b are the parameters 0, 1 and 2 and n a
   variable. *)

let kernel ?(name = "axpy") ?(size = 4) ?bound () =
  let open Ops.O in
  let buffer slot = Shape.placeholder ~slot [ size ] Float32 in
  let n =
    Ops.variable ~dtype:Int32 "n" (Dtype.Value.of_int 0)
      (Dtype.Value.of_int 100)
  in
  let n =
    match bound with
    | Some v -> Shape.bind n (`Int (Bigint.of_int v))
    | None -> n
  in
  let i = Ops.range ~axis_type:Weak (Int size) [ 0 ] in
  let at b = Ops.index (buffer b) [ i ] in
  Ops.sink ~kernel:(Ops.kernel_info ~name ())
    [ Ops.end_ (Ops.store (at 0) ((at 1 * Ops.cast n Float32) + at 2)) [ i ] ]

let compiled ?name ?size ?bound () =
  Codegen.to_program (kernel ?name ?size ?bound ()) (Lazy.force clang)

let axpy = lazy (compiled ())

(* Compiles

   [compiles d] agrees with a compile of a kernel for [d], made anew and kept
   nowhere ([Codegen.compile]), so that no binary of an earlier run answers for
   the compiler. *)
(* Whether a kernel compiles for [d], its renderer included: a renderer that
   cannot be made, as where its compiler's library is missing, compiles
   none. *)
let kernel_compiles d =
  match Engine.renderer d with
  | exception (Failure _ | Device.Uncomputable _) -> false
  | r -> (
      match Codegen.compile (Codegen.linearize (kernel ()) r) r with
      | (_ : Ops.t) -> true
      | exception Renderer.Compiler.Compile_error _ -> false)

let agrees d = equal bool (kernel_compiles d) (Result.is_ok (Engine.compiles d))

(* A child's answer for the host: its reason, or [ok]. *)
let parts =
  [
    ( "compiles",
      fun () ->
        match Engine.compiles host with
        | Ok () -> print_string "ok"
        | Error why -> print_string why );
  ]

let compiles =
  group "compiles"
    [
      test "the host is Ok iff a kernel compiles for it" (fun () ->
          if not (Rig.Image.links host) then
            skip ~reason:"rig links no host code here" ();
          agrees host);
      test "the host is an error naming CC where CC names no program" (fun () ->
          if not (Rig.Image.links host) then
            skip ~reason:"rig links no host code here" ();
          let cc = Filename.concat (temp_dir ()) "clang" in
          match
            Disk_cache.child
              ~env:[ ("CC", cc) ]
              ~cachedb:(Disk_cache.fresh ()) "compiles"
          with
          | Ok why -> contains ~sub:cc why
          | Error err -> failf "the child failed: %s" err);
    ]

(* A kernel long enough for the host clock to see it run. *)
let long_axpy () = kernel ~name:"long_axpy" ~size:(1 lsl 18) ()
let a = floats [| 1.; 2.; 3.; 4. |]
let b = floats [| 10.; 20.; 30.; 40. |]

(* [prg] called on parameters of [size] floats on [name], one per slot. *)
let call_on name ~size ~slots prg =
  Ops.call prg
    (List.init slots (fun slot ->
         Shape.param ~shape:[ Ops.Int size ] ~device:name slot Float32))

let buffers () =
  [|
    Some (Run.buffer host Float32 (floats [| 0.; 0.; 0.; 0. |]));
    Some (Run.buffer host Float32 a);
    Some (Run.buffer host Float32 b);
  |]

let axpy_on_host prg =
  Engine.link_call ~devices (call_on "CPU" ~size:4 ~slots:3 prg)

(* A view of [n] bytes of [b] whose first byte lies [phase] bytes past a
   16-byte boundary. *)
let view_at_phase b phase n =
  let rec first j =
    if Buffer.aligned b ~at:(j - phase) 16 then j else first (j + 1)
  in
  Buffer.view b ~first:(first 0) ~length:n

(* [n] bytes of a buffer of [d] whose first byte lies [phase] bytes past a
   16-byte boundary, holding [bytes] where given. *)
let at_phase ?bytes d phase n =
  let v = view_at_phase (Buffer.create d (n + 15)) phase n in
  Option.iter (fun src -> Buffer.copy ~src ~dst:v) bytes;
  v

(* A copy of [b] 4 bytes past a 16-byte boundary. *)
let off_phase b = at_phase ~bytes:b (Buffer.device b) 4 (Buffer.length b)

(* axpy whose parameter 1 lies 4 bytes past a 16-byte boundary, called on the
   host. *)
let phased_kernel () =
  let open Ops.O in
  let buffer slot =
    if slot = 1 then Shape.param ~shape:[ Ops.Int 4 ] ~phase:4 slot Float32
    else Shape.placeholder ~slot [ 4 ] Float32
  in
  let n =
    Ops.variable ~dtype:Int32 "n" (Dtype.Value.of_int 0)
      (Dtype.Value.of_int 100)
  in
  let i = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
  let at b = Ops.index (buffer b) [ i ] in
  let sink =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"phased_axpy" ())
      [ Ops.end_ (Ops.store (at 0) ((at 1 * Ops.cast n Float32) + at 2)) [ i ] ]
  in
  Codegen.to_program sink (Lazy.force clang)

let phased_prg = lazy (phased_kernel ())

let phased_call () =
  Ops.call (Lazy.force phased_prg)
    (List.init 3 (fun slot ->
         let phase = if slot = 1 then 4 else 0 in
         Shape.param ~shape:[ Ops.Int 4 ] ~device:"CPU" ~phase slot
           Float32))

let phased_axpy () = Engine.link_call ~devices (phased_call ())

(* What a refusal of a buffer off its phase says. *)
let off_its_phase = Exn.invalid_arg ~sub:"bytes past a multiple of 16"

(* out[i] = a[i + 3] * n + b[i] for four elements, a eight floats 4 bytes past a
   16-byte boundary: a[3] to a[6] lie on 16 bytes, which the kernel reads as one
   float4. *)
let shifted_axpy =
  lazy
    (let open Ops.O in
     let a = Shape.param ~shape:[ Ops.Int 8 ] ~phase:4 1 Float32 in
     let buffer slot = Shape.placeholder ~slot [ 4 ] Float32 in
     let n =
       Ops.variable ~dtype:Int32 "n" (Dtype.Value.of_int 0)
         (Dtype.Value.of_int 100)
     in
     let i = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
     let at b = Ops.index b [ i ] in
     let a_i = Ops.index a [ i + Ops.int 3 ] in
     let sink =
       Ops.sink
         ~kernel:(Ops.kernel_info ~name:"shifted_axpy" ())
         [
           Ops.end_
             (Ops.store
                (at (buffer 0))
                ((a_i * Ops.cast n Float32) + at (buffer 2)))
             [ i ];
         ]
     in
     Codegen.to_program sink (Lazy.force clang))

(* A buffer the schedule makes 4 bytes past a 16-byte boundary: a copy of slot 1
   into it, then shifted_axpy over it into slot 0. Storage made off its phase
   would put the float4 12 bytes past 16, where x86 faults. *)
let runs_on_made_storage () =
  let param slot n =
    Shape.param ~shape:[ Ops.Int n ] ~device:"CPU" slot Float32
  in
  let made = Ops.new_buffer ~phase:4 "CPU" 8 Float32 in
  let linear =
    Ops.v Op.Linear
      ~src:
        [
          Shape.store_call made (param 1 8);
          Ops.call (Lazy.force shifted_axpy) [ param 0 4; made; param 2 4 ];
        ]
  in
  let s = Engine.link ~devices (compile linear) and bufs = buffers () in
  bufs.(1) <-
    Some (Run.buffer host Float32 (floats [| 0.; 0.; 0.; 1.; 2.; 3.; 4.; 0. |]));
  Engine.run ~vars:[ ("n", 3) ] s bufs;
  equal values
    (floats [| 13.; 26.; 39.; 52. |])
    (Run.values Float32 (Option.get bufs.(0)))

let runs_at_its_phase () =
  let s = phased_axpy () and bufs = buffers () in
  bufs.(1) <- Some (off_phase (Option.get bufs.(1)));
  Engine.run ~vars:[ ("n", 3) ] s bufs;
  equal values
    (floats [| 13.; 26.; 39.; 52. |])
    (Run.values Float32 (Option.get bufs.(0)))

let runs_a_kernel () =
  let s = axpy_on_host (Lazy.force axpy) and bufs = buffers () in
  Engine.run ~vars:[ ("n", 3) ] s bufs;
  equal values
    (floats [| 13.; 26.; 39.; 52. |])
    (Run.values Float32 (Option.get bufs.(0)))

let takes_the_bound_value () =
  let s = axpy_on_host (compiled ~name:"axpy_bound" ~bound:2 ()) in
  let bufs = buffers () in
  Engine.run s bufs;
  equal values
    (floats [| 12.; 24.; 36.; 48. |])
    (Run.values Float32 (Option.get bufs.(0)))

(* The do-while loops of the Linearizer suite, each storing what it counts. *)
let counts_in_a_loop (name, count) =
  test
    (name ^ " runs its loops to " ^ string_of_int count)
    (fun () ->
      let sink =
        Golden.sink ("../../codegen/late/linearizer/" ^ name ^ ".golden")
      in
      let prg = Codegen.to_program sink (Lazy.force clang) in
      equal values
        [| `Int (Bigint.of_int count) |]
        (List.assoc 0 (Run.on_host prg [])))

let with_binary bytes prg =
  let srcs = Ops.src prg in
  Ops.replace prg
    ~src:
      (List.filteri (fun i _ -> i < 3) srcs
      @ [ Ops.v Op.Binary ~arg:(Ops.Bytes bytes) ])

let on_the_host =
  group "kernels on the host"
    [
      test "a kernel runs on its buffers, in the order of its globals"
        runs_a_kernel;
      test "a variable left out of vars takes its bound value"
        takes_the_bound_value;
      group "loops"
        (List.map counts_in_a_loop
           [
             ("wait_loop", 10);
             ("nested_loop", 12);
             ("two_loops", 25);
             ("loop_in_loop", 12);
           ]);
      test "link_call refuses a node that is no call of a compiled program"
        (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Engine.link_call ~devices (Lazy.force axpy)));
      test "link fails with the host's reason for refusing a binary" (fun () ->
          raises_match Exn.failure (fun () ->
              axpy_on_host (with_binary "no object" (Lazy.force axpy))));
      test "run refuses fewer slots than parameters" (fun () ->
          let s = axpy_on_host (Lazy.force axpy) in
          raises_match Exn.invalid_arg (fun () ->
              Engine.run ~vars:[ ("n", 1) ] s (Array.sub (buffers ()) 0 2)));
      test "run refuses a buffer smaller than its parameter" (fun () ->
          let s = axpy_on_host (Lazy.force axpy) in
          let bufs = buffers () in
          bufs.(0) <- Some (Run.buffer host Float32 (floats [| 0.; 0.; 0. |]));
          raises_match Exn.invalid_arg (fun () ->
              Engine.run ~vars:[ ("n", 1) ] s bufs));
      test "run refuses a buffer that starts off its parameter's phase"
        (fun () ->
          let s = axpy_on_host (Lazy.force axpy) in
          let bufs = buffers () in
          bufs.(1) <- Some (off_phase (Option.get bufs.(1)));
          raises_match
            (Exn.invalid_arg ~sub:"slot 1's buffer of CPU starts 4 bytes")
            (fun () -> Engine.run ~vars:[ ("n", 1) ] s bufs);
          equal values
            (floats [| 0.; 0.; 0.; 0. |])
            (Run.values Float32 (Option.get bufs.(0))));
      test "a parameter of phase 4 runs on a buffer 4 bytes past 16"
        runs_at_its_phase;
      test "storage made at link of phase 4 is made 4 bytes past 16"
        runs_on_made_storage;
      test "run refuses a phase-4 parameter's buffer on a 16-byte boundary"
        (fun () ->
          let s = phased_axpy () in
          let bufs = buffers () in
          bufs.(1) <- Some (at_phase ~bytes:(Option.get bufs.(1)) host 0 16);
          raises_match
            (Exn.invalid_arg ~sub:"slot 1's buffer of CPU starts 0 bytes")
            (fun () -> Engine.run ~vars:[ ("n", 1) ] s bufs));
      test "link refuses a parameter that two calls access at two phases"
        (fun () ->
          let axpy = call_on "CPU" ~size:4 ~slots:3 (Lazy.force axpy) in
          raises_match (Exn.invalid_arg ~sub:"slot 1 is accessed") (fun () ->
              Engine.link ~devices
                (compile (Ops.v Op.Linear ~src:[ axpy; phased_call () ]))));
      test "run takes the slots of a phase-4 parameter" (fun () ->
          let s = phased_axpy () in
          Engine.run ~vars:[ ("n", 1) ] s (Engine.slots s));
      test "run refuses an unbound variable" (fun () ->
          let s = axpy_on_host (Lazy.force axpy) in
          raises_match Exn.invalid_arg (fun () -> Engine.run s (buffers ())));
    ]

(* Host programs in blocks

   A kernel of [m] sums, out[i] = sum_j a[i k + j] b[j] for [k] terms, with
   the optimisations [opts] or the heuristic's. Each sum is summed in one
   block, in the same order whatever the blocks. *)

let sums ?(name = "sums") ?opts m k =
  let buffer slot n = Shape.placeholder ~slot [ n ] Float32 in
  let a = buffer 1 (m * k) and b = buffer 2 k in
  let i = Ops.range ~axis_type:Weak (Int m) [ 0 ] in
  let j = Ops.range ~axis_type:Weak (Int k) [ 1 ] in
  let term =
    let open Ops.O in
    Ops.index a [ (i * Ops.int k) + j ] * Ops.index b [ j ]
  in
  Ops.sink ~kernel:(Ops.kernel_info ~name ?opts_to_apply:opts ())
    [
      Ops.end_
        (Ops.store
           (Ops.index (buffer 0 m) [ i ])
           (Ops.reduce term Op.Add [ j ]))
        [ i ];
    ]

(* Values whose sums round differently in another order, from [seed]. *)
let scattered ?(seed = 0) n =
  let at x = Float.of_int (x + (seed * 7919)) in
  floats (Array.init n (fun x -> Float.sin (at x) *. 1e3))

(* The function of [sums m k], unoptimised so that its loop runs [m]
   iterations, compiled and loaded once for each [m] and [k]. *)
let sums_fn =
  let loaded = Hashtbl.create 8 in
  fun m k ->
    match Hashtbl.find_opt loaded (m, k) with
    | Some fn -> fn
    | None ->
        let prg = Codegen.to_program (sums ~opts:[] m k) (Lazy.force clang) in
        let image = Result.get_ok (Rig.Image.load host (Uops.binary_of prg)) in
        let fn = require_some (Rig.Image.resolve image Codegen.symbol) in
        Hashtbl.replace loaded (m, k) fn;
        fn

(* The bits of [sums m k] on scattered values of [seed], launched on the host
   in [blocks] blocks of [(m + blocks - 1) / blocks] threads: its parameters are
   the addresses of out, a and b, then the bounds of its whole loop, [0] and
   [m], two int32s, and a call runs the iterations its threads number. *)
let sums_in ?seed ~blocks m k =
  let s =
    Rig.Submission.(
      make host
        [|
          launch
            ~refs:[| write 0 ~at:0; read 1 ~at:8; read 2 ~at:16 |]
            (sums_fn m k) ~params:32;
        |])
  in
  let params = Bytes.make 32 '\000' in
  Bytes.set_int32_le params 24 0l;
  Bytes.set_int32_le params 28 (Int32.of_int m);
  let args =
    {
      Rig.Submission.groups = (blocks, 1, 1);
      threads = ((m + blocks - 1) / blocks, 1, 1);
      shared = 0;
      params;
    }
  in
  let out = Run.buffer host Float32 (floats (Array.make m 0.)) in
  ignore
    (Rig.submit s
       ~buffers:
         [|
           out;
           Run.buffer host Float32 (scattered ?seed (m * k));
           Run.buffer host Float32 (scattered ?seed k);
         |]
       ~args:[| args |]
      : Rig.Point.t);
  Array.map
    (function `Float x -> Int64.bits_of_float x | _ -> assert false)
    (Run.values Float32 out)

(* A split loop of [m] iterations, in [blocks] blocks, on values of [seed]:
   one block, a block per iteration, a last block that runs fewer iterations
   than the others, or none. *)
let gen_split =
  let open Gen in
  let* m = of_list [ 1; 2; 3; 17; 512; 1031 ] in
  let* blocks =
    frequency [ (1, constant 1); (1, constant m); (4, int_range 1 m) ]
  in
  let+ seed = int_range 0 1000 in
  (m, blocks, seed)

let same_bits_in_any_blocks (m, blocks, seed) =
  let per = (m + blocks - 1) / blocks in
  cover "one block" (blocks = 1);
  cover "a block per iteration" (m > 1 && blocks = m);
  cover "a shorter last block" (m mod per <> 0);
  cover "an empty last block" ((blocks - 1) * per >= m);
  let one = sums_in ~seed ~blocks:1 m 64 in
  not_equal ~msg:"the sums are written" (array int64) (Array.make m 0L) one;
  equal (array int64) one (sums_in ~seed ~blocks m 64)

(* [out[i] = 2 a[i]] over [m] floats: a kernel that moves 8 bytes for each
   operation. *)
let doubled m =
  let buffer slot = Shape.placeholder ~slot [ m ] Float32 in
  let i = Ops.range ~axis_type:Weak (Int m) [ 0 ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:(Printf.sprintf "doubled_%d" m) ())
    [
      Ops.end_
        (Ops.store
           (Ops.index (buffer 0) [ i ])
           Ops.O.(Ops.index (buffer 1) [ i ] * Ops.float 2.))
        [ i ];
    ]

let doubled_program =
  let compiled = Hashtbl.create 8 in
  fun m ->
    match Hashtbl.find_opt compiled m with
    | Some prg -> prg
    | None ->
        let prg = Codegen.to_program (doubled m) (Lazy.force clang) in
        Hashtbl.replace compiled m prg;
        prg

(* The blocks a host launch of the compiled [prg] runs in, by its split and
   estimates. *)
let launch_blocks prg =
  match Ops.arg (Ops.nth prg 0) with
  | Ops.Kernel { split = Some s; estimates = Some e; _ } -> (
      match (s.iterations, e.ops, e.mem) with
      | Int extent, Int ops, Int bytes ->
          fst (Engine.blocks ~extent ~ops ~bytes)
      | _ -> failwith "a split of symbolic estimates")
  | _ -> 1

let doubles_in_blocks (m, seed) =
  let prg = doubled_program m in
  let blocks = launch_blocks prg in
  cover "one block" (blocks = 1);
  cover "several blocks" (blocks > 1);
  let a =
    Array.init m (fun x -> Float.of_int (((x + (seed * 7919)) mod 2001) - 1000))
  in
  let out = List.assoc 0 (Run.on_host prg [ (1, floats a) ]) in
  equal values (floats (Array.map (fun x -> 2. *. x) a)) out

(* The blocks of a launch as {!Tolk_engine.link} states them: a kernel of more
   operations than bytes takes a block for each 2^16 operations, any other a
   block for each 2^18 operations or for each 2^18 bytes up to 8 blocks,
   whichever gives more, at least one and at most one per iteration. Each row
   is its name, the loop's iterations, operations and bytes, and its blocks
   and the iterations of each. *)
let block_rows =
  [
    ("a small kernel", 1024, 1000, 8192, (1, 1024));
    ("operations just below twice 2^16", 4096, (1 lsl 17) - 1, 0, (1, 4096));
    ("operations at twice 2^16", 4096, 1 lsl 17, 0, (2, 2048));
    ("operations alone", 4096, 1 lsl 20, 0, (16, 256));
    ("operations beyond bytes", 512, 1 lsl 24, 1 lsl 16, (256, 2));
    ("an operation more than bytes", 4096, (1 lsl 20) + 1, 1 lsl 20, (16, 256));
    ("as many operations as bytes", 4096, 1 lsl 20, 1 lsl 20, (4, 1024));
    ("bytes just below twice 2^18", 4096, 0, (1 lsl 19) - 1, (1, 4096));
    ("bytes at twice 2^18", 4096, 0, 1 lsl 19, (2, 2048));
    ("a copy, bytes alone", 65536, 0, 1 lsl 20, (4, 16384));
    ("bytes beyond operations", 16384, 1 lsl 18, 3 lsl 18, (3, 5462));
    ("bytes up to 8 blocks", 65536, 1 lsl 20, 1 lsl 30, (8, 8192));
    ("operations past 8 blocks of bytes", 512, 1 lsl 24, 1 lsl 30, (64, 8));
    ("a shorter last block", 10, 4 lsl 16, 0, (4, 3));
    ("at most one per iteration", 3, 1 lsl 30, 1 lsl 30, (3, 1));
    ("no iteration", 0, 1 lsl 20, 1 lsl 20, (1, 1));
  ]

let in_blocks =
  group "host programs in blocks"
    [
      cases
        ~id:(fun (name, _, _, _, _) -> name)
        "a launch of more operations than bytes runs a block per 2^16 \
         operations, another per 2^18 operations or bytes, bytes up to 8"
        block_rows
        (fun (_, extent, ops, bytes, expected) ->
          equal (pair int int) expected (Engine.blocks ~extent ~ops ~bytes));
      prop ~count:60
        ~examples:[ (512, 1, 0); (17, 17, 1); (17, 3, 2); (17, 10, 3) ]
        "a program writes the same bits in any blocks" gen_split
        same_bits_in_any_blocks;
      prop ~count:12
        ~examples:[ (1000, 0); (65536, 1) ]
        "a kernel that moves many bytes for few operations writes its values"
        Gen.(
          pair (of_list [ 1; 1000; 65536; 65537; 262144 ]) (int_range 0 1000))
        doubles_in_blocks;
      test "a run of a large kernel computes its sums" (fun () ->
          let m = 4096 and k = 256 in
          let prg =
            Codegen.to_program (sums ~name:"sums_run" m k) (Lazy.force clang)
          in
          let a = Array.init (m * k) (fun x -> Float.of_int (x mod 7))
          and b = Array.init k (fun x -> Float.of_int (x mod 5)) in
          let out =
            List.assoc 0 (Run.on_host prg [ (1, floats a); (2, floats b) ])
          in
          let expected =
            Array.init m (fun i ->
                let s = ref 0. in
                for j = 0 to k - 1 do
                  s := !s +. (a.((i * k) + j) *. b.(j))
                done;
                !s)
          in
          equal values (floats expected) out);
    ]

(* Linked schedules

   The recorded programs of the Schedule suite, scheduled, compiled for the
   devices they name and linked with their storage bound to buffers of small
   integers, write what their tensors compute (Tensors), with each variable at
   the value it is run with. *)

let program name = Golden.sink ("../../schedule/schedule/" ^ name ^ ".golden")

let element dtype k : Dtype.value =
  if Dtype.is_float dtype then `Float (float_of_int k)
  else if Dtype.equal dtype Bool then `Bool (k > 0)
  else if Dtype.is_unsigned dtype then `Int (Bigint.of_int (k + 3))
  else `Int (Bigint.of_int k)

let placement (p : Ops.param_arg) =
  match p.device with
  | Some d -> d
  | None -> invalid_arg "storage without a device"

(* The storage a program is given, its parameters and buffers, each with its
   contents before the run and its buffer. *)
type storage = {
  node : Ops.t;
  arg : Ops.param_arg;
  before : Dtype.value array;
  buffer : Buffer.t;
}

(* A buffer of [d] holding [values], a file on the disk. *)
let buffer d dt values =
  if not (Rig.equal d Rig_disk.device) then Run.buffer d dt values
  else
    let file =
      Rig_disk.buffer
        (Result.get_ok
           (Rig_disk.create_file
              (Filename.concat (temp_dir ()) "file")
              (Array.length values * Dtype.itemsize dt)))
    in
    Buffer.copy ~src:(Run.buffer host dt values) ~dst:file;
    file

let storage_of ?(devices = devices) ?(seed = 0) big =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | (Param | Buffer), Param p when p.addrspace <> Alu ->
          let before =
            Array.init (Shape.max_numel n) (fun j ->
                element p.dtype ((((j * 7) + (p.slot * 3) + seed) mod 11) - 3))
          in
          let buffer = buffer (devices (placement p)) p.dtype before in
          Some { node = n; arg = p; before; buffer }
      | _ -> None)
    (Ops.toposort ~calls:Skip big)

let bound storage =
  List.filter_map
    (fun s -> if Ops.op s.node = Buffer then Some (s.node, s.buffer) else None)
    storage

let slots storage =
  let n =
    List.fold_left
      (fun n s -> if Ops.op s.node = Param then max n (s.arg.slot + 1) else n)
      0 storage
  in
  let slots = Array.make n None in
  List.iter
    (fun s -> if Ops.op s.node = Param then slots.(s.arg.slot) <- Some s.buffer)
    storage;
  slots

let contents s =
  Array.sub (Run.values s.arg.dtype s.buffer) 0 (Shape.max_numel s.node)

let refill storage =
  List.iter
    (fun s ->
      Buffer.copy ~src:(Run.buffer host s.arg.dtype s.before) ~dst:s.buffer)
    storage

let by_slot storage f = List.map (fun s -> (s.arg.slot, f s)) storage

(* [with_values vars big] is [big] with each variable replaced by its value in
   [vars], or its bound value. *)
let with_values vars big =
  Ops.substitute ~calls:Skip ~pass:Fixed_point big
    (List.filter_map
       (fun n ->
         match (Ops.op n, Ops.arg n) with
         | Param, Param { name = Some v; addrspace = Alu; bound; _ } -> (
             match (List.assoc_opt v vars, bound) with
             | Some x, _ -> Some (n, Ops.int ~dtype:(Ops.dtype n) x)
             | None, Some x ->
                 Some (n, Ops.const ~dtype:(Ops.dtype n) (x :> Dtype.const))
             | None, None -> None)
         | _ -> None)
       (Ops.toposort ~calls:Enter big))

(* What [big] leaves in its storage when run with [vars]. *)
let expected ?(vars = []) big storage =
  let buffers = List.map (fun s -> (s.arg.slot, s.before)) storage in
  let writes = Tensors.writes ~buffers (with_values vars big) in
  by_slot storage (fun s ->
      let after = Array.copy s.before in
      List.iter
        (fun (slot, i, v) -> if slot = s.arg.slot then after.(i) <- v)
        writes;
      after)

let schedule ?(devices = devices) big =
  let linear, vars = Schedule.create_linear_with_vars ~cores:Cores.host big in
  (compile ~devices linear, vars)

let linked ?(devices = devices) big =
  let compiled, vars = schedule ~devices big in
  let storage = storage_of ~devices big in
  (Engine.link ~devices ~bound:(bound storage) compiled, vars, storage)

let slot_values = list (pair int values)

let writes_what_it_computes name =
  test name (fun () ->
      let big = program name in
      let s, vars, storage = linked big in
      Engine.run ~vars s (slots storage);
      equal slot_values (expected ~vars big storage) (by_slot storage contents))

let recorded =
  [
    "add";
    "add_row_from_self";
    "assign";
    "assign_double_diamond";
    "assign_flipped_from_self";
    "assign_permuted_self";
    "attention";
    "cat";
    "chained_functions";
    "clone";
    "contiguous";
    "conv";
    "copy";
    "copy_computed";
    "copy_one";
    "copy_view";
    "disk_store";
    "disk_to";
    "disk_view_to";
    "double_matmul";
    "embedding";
    "full_invalid";
    "inline_function";
    "layernorm";
    "matmul";
    "precompiled_function";
    "precompiled_scalar";
    "read_then_overwrite";
    "reduce_multiple_paths";
    "roll_from_self";
    "select_no_lane";
    "select_no_lane_rows";
    "select_one_lane";
    "setitem";
    "softmax";
    "sort";
    "split_sum";
    "sum";
    "two_outputs";
    "variable_offset";
    "variable_reduce";
    "variable_same";
    "variable_shrink";
    "variable_two";
    "variable_unused";
  ]

(* Tensors runs no custom kernel, nor a store through a bitcast, so these two
   programs state what they write. *)

let slot storage k = List.find (fun s -> s.arg.slot = k) storage

(* custom_kernel: its kernel stores a + 1 into c, for a the double of slot 3 and
   c slot 2, and slot 4 is c + 1. *)
let runs_a_custom_kernel () =
  let s, vars, storage = linked (program "custom_kernel") in
  Engine.run ~vars s [||];
  let x = Array.map Dtype.Value.to_float (slot storage 3).before in
  equal values
    (floats (Array.map (fun x -> (2. *. x) +. 1.) x))
    (contents (slot storage 2));
  equal values
    (floats (Array.map (fun x -> (2. *. x) +. 2.) x))
    (contents (slot storage 4))

(* A custom kernel that takes the variable [n], bound to 5, among its buffers
   stores a + n + [k] into c, for a the double of slot [a] and c slot [c], and
   slot [out] is c + 1. [k] is a variable of the kernel's own, which the run
   binds. *)
let runs_a_custom_kernel_scalar ?(devices = devices) ?(k = []) ?(c = 2)
    ?(a = 3) ?(out = 4) name () =
  let s, vars, storage = linked ~devices (program name) in
  Engine.run ~vars:(k @ vars) s [||];
  let k = List.fold_left (fun _ (_, v) -> v) 0 k in
  let xs = Array.map Dtype.Value.to_int (slot storage a).before in
  let ints f = Array.map (fun x -> `Int (Bigint.of_int (f x))) xs in
  equal values (ints (fun x -> (2 * x) + 5 + k)) (contents (slot storage c));
  equal values (ints (fun x -> (2 * x) + 6 + k)) (contents (slot storage out))

(* custom_kernel_scalar_with_other_variable: slot 4 is slot 3 plus the
   variable m, bound to 70, beside custom_kernel_scalar's kernel on slots 5 to
   7. *)
let runs_beside_another_variable ?(devices = devices) () =
  runs_a_custom_kernel_scalar ~devices ~c:5 ~a:6 ~out:7
    "custom_kernel_scalar_with_other_variable" ();
  let s, vars, storage =
    linked ~devices (program "custom_kernel_scalar_with_other_variable")
  in
  Engine.run ~vars s [||];
  let xs = Array.map Dtype.Value.to_int (slot storage 3).before in
  equal values
    (Array.map (fun x -> `Int (Bigint.of_int (x + 70))) xs)
    (contents (slot storage 4))

(* Two custom kernels on CPU:1 that add a variable [k] to [a], of [n]
   elements, into their own outputs, [k] bound to [k1] in one and [k2] in the
   other. Independent, one just after the other, they run as one launch,
   and each output takes its own [k]. *)
let custom_kernels_of_one_name ?(devices = devices) ~n (k1, k2) () =
  let k = Ops.variable "k" (Dtype.Value.of_int 0) (Dtype.Value.of_int 8) in
  let add = function
    | [ out; v; src ] ->
        (* One element takes no range, as the schedule's own kernels. *)
        let i, ends =
          if n = 1 then (Ops.int 0, Fun.id)
          else
            let i = Ops.range ~axis_type:Weak (Int n) [ 0 ] in
            (i, fun st -> Ops.end_ st [ i ])
        in
        let at b = Ops.index b [ i ] in
        Ops.sink
          ~kernel:(Ops.kernel_info ())
          [ ends (Ops.store (at out) Ops.O.(at src + v)) ]
    | _ -> invalid_arg "add takes three arguments"
  in
  let a = Ops.new_buffer "CPU:1" n Int32 in
  let out v =
    let o = Ops.new_buffer "CPU:1" n Int32 in
    let bound = Shape.bind k (`Int (Bigint.of_int v)) in
    (o, List.hd (Shape.custom_kernel [ o; bound; a ] add))
  in
  let (o1, s1), (o2, s2) = (out k1, out k2) in
  let big = Ops.sink [ s1; s2 ] in
  let calls = Ops.src (fst (Schedule.create_linear_with_vars
                              ~cores:Cores.host big)) in
  equal int ~msg:"launches" 1 (List.length calls);
  let s, vars, storage = linked ~devices big in
  equal
    (list (pair string int))
    ~msg:"the run's variables"
    (if k1 = k2 then [ ("k", k1) ] else [])
    vars;
  Engine.run ~vars s [||];
  let of_node u = List.find (fun s -> s.node == u) storage in
  let xs = Array.map Dtype.Value.to_int (of_node a).before in
  let plus v = Array.map (fun x -> `Int (Bigint.of_int (x + v))) xs in
  equal values ~msg:"k bound first" (plus k1) (contents (of_node o1));
  equal values ~msg:"k bound second" (plus k2) (contents (of_node o2))

(* A custom kernel on CPU:1 that adds [k], bound to 3, to [a], one element,
   beside a kernel of the schedule's own that adds [k] to [a] too: they run as
   one kernel, and both read [k], the run's binding or else the 3 their call
   binds. *)
let custom_kernel_beside_one_by_name ?(devices = devices) () =
  let k = Ops.variable "k" (Dtype.Value.of_int 0) (Dtype.Value.of_int 8) in
  let bound = Shape.bind k (`Int (Bigint.of_int 3)) in
  let add = function
    | [ out; v; src ] ->
        let at b = Ops.index b [ Ops.int 0 ] in
        Ops.sink
          ~kernel:(Ops.kernel_info ())
          [ Ops.store (at out) Ops.O.(at src + v) ]
    | _ -> invalid_arg "add takes three arguments"
  in
  let a = Ops.new_buffer "CPU:1" 1 Int32 in
  let o1 = Ops.new_buffer "CPU:1" 1 Int32
  and o2 = Ops.new_buffer "CPU:1" 1 Int32 in
  let s1 = List.hd (Shape.custom_kernel [ o1; bound; a ] add) in
  let s2 =
    Ops.after o2
      [ Ops.store o2 Ops.O.(a + Shape.reshape (Ops.cast bound Int32) [ Int 1 ]) ]
  in
  let big = Ops.sink [ s1; s2 ] in
  let calls = Ops.src (fst (Schedule.create_linear_with_vars
                              ~cores:Cores.host big)) in
  equal int ~msg:"kernels" 1 (List.length calls);
  let s, vars, storage = linked ~devices big in
  equal (list (pair string int)) ~msg:"the run's variables" [ ("k", 3) ] vars;
  let of_node u = List.find (fun s -> s.node == u) storage in
  let xs = Array.map Dtype.Value.to_int (of_node a).before in
  let plus v = Array.map (fun x -> `Int (Bigint.of_int (x + v))) xs in
  List.iter
    (fun (run, v) ->
      Engine.run ~vars:run s [||];
      let msg what = Printf.sprintf "%s, k = %d" what v in
      equal values ~msg:(msg "the custom kernel's") (plus v)
        (contents (of_node o1));
      equal values ~msg:(msg "the schedule's") (plus v) (contents (of_node o2)))
    [ ([ ("k", 7) ], 7); ([], 3) ]

(* A custom kernel on CPU:1 that adds [k], bound to 3, to [a], of four
   elements: a run that binds [k] adds the run's value, and a run that does not
   adds 3. *)
let custom_kernel_rebound ?(devices = devices) () =
  let k = Ops.variable "k" (Dtype.Value.of_int 0) (Dtype.Value.of_int 8) in
  let add = function
    | [ out; v; src ] ->
        let i = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
        let at b = Ops.index b [ i ] in
        Ops.sink
          ~kernel:(Ops.kernel_info ())
          [ Ops.end_ (Ops.store (at out) Ops.O.(at src + v)) [ i ] ]
    | _ -> invalid_arg "add takes three arguments"
  in
  let a = Ops.new_buffer "CPU:1" 4 Int32
  and o = Ops.new_buffer "CPU:1" 4 Int32 in
  let bound = Shape.bind k (`Int (Bigint.of_int 3)) in
  let big = Ops.sink [ List.hd (Shape.custom_kernel [ o; bound; a ] add) ] in
  let s, vars, storage = linked ~devices big in
  equal (list (pair string int)) ~msg:"the run's variables" [ ("k", 3) ] vars;
  let of_node u = List.find (fun s -> s.node == u) storage in
  let xs = Array.map Dtype.Value.to_int (of_node a).before in
  let plus v = Array.map (fun x -> `Int (Bigint.of_int (x + v))) xs in
  List.iter
    (fun (run, v) ->
      Engine.run ~vars:run s [||];
      equal values ~msg:(Printf.sprintf "k = %d" v) (plus v)
        (contents (of_node o)))
    [ ([ ("k", 1) ], 1); ([ ("k", 7) ], 7); ([], 3) ]

(* A kernel of the schedule's own on CPU:1 that adds [k], bound to 3, to [a],
   of four elements: its call binds no variable, so it reads [k] from the run
   alone, and a run that does not bind [k] is refused. A run binding [k] to 7
   adds 7. *)
let reads_its_variable_from_the_run ?(devices = devices) () =
  let k = Ops.variable "k" (Dtype.Value.of_int 0) (Dtype.Value.of_int 8) in
  let bound = Shape.bind k (`Int (Bigint.of_int 3)) in
  let a = Ops.new_buffer "CPU:1" 4 Int32
  and o = Ops.new_buffer "CPU:1" 4 Int32 in
  let k4 = Shape.reshape (Ops.cast bound Int32) [ Int 1 ] in
  let big = Ops.sink [ Ops.after o [ Ops.store o Ops.O.(a + k4) ] ] in
  let s, vars, storage = linked ~devices big in
  equal (list (pair string int)) ~msg:"the run's variables" [ ("k", 3) ] vars;
  let of_node u = List.find (fun s -> s.node == u) storage in
  raises_match (Exn.invalid_arg ~sub:"variable k is unbound") (fun () ->
      Engine.run ~vars:[] s [||]);
  Engine.run ~vars:[ ("k", 7) ] s [||];
  let xs = Array.map Dtype.Value.to_int (of_node a).before in
  equal values ~msg:"k = 7"
    (Array.map (fun x -> `Int (Bigint.of_int (x + 7))) xs)
    (contents (of_node o))

(* A custom kernel on CPU:1 that adds one to the first [k] of eight elements
   of [a], [k] bound to 3, its range [k] long: a run that binds [k] adds to
   the run's number of elements, and a run that does not to 3, leaving the
   others as [o] held them. *)
let custom_kernel_sized_by_variable ?(devices = devices) () =
  let k = Ops.variable "k" (Dtype.Value.of_int 1) (Dtype.Value.of_int 8) in
  let add = function
    | [ out; v; src ] ->
        let i = Ops.range ~axis_type:Weak (Sym v) [ 0 ] in
        let at b = Ops.index b [ i ] in
        Ops.sink
          ~kernel:(Ops.kernel_info ())
          [
            Ops.end_
              (Ops.store (at out) Ops.O.(at src + Ops.int ~dtype:Int32 1))
              [ i ];
          ]
    | _ -> invalid_arg "add takes three arguments"
  in
  let a = Ops.new_buffer "CPU:1" 8 Int32
  and o = Ops.new_buffer "CPU:1" 8 Int32 in
  let bound = Shape.bind k (`Int (Bigint.of_int 3)) in
  let big = Ops.sink [ List.hd (Shape.custom_kernel [ o; bound; a ] add) ] in
  let s, vars, storage = linked ~devices big in
  equal (list (pair string int)) ~msg:"the run's variables" [ ("k", 3) ] vars;
  let of_node u = List.find (fun s -> s.node == u) storage in
  let xs = Array.map Dtype.Value.to_int (of_node a).before in
  let before = (of_node o).before in
  List.iter
    (fun (run, n) ->
      refill storage;
      Engine.run ~vars:run s [||];
      let want =
        Array.mapi
          (fun j x ->
            if j < n then `Int (Bigint.of_int (x + 1)) else before.(j))
          xs
      in
      equal values ~msg:(Printf.sprintf "k = %d" n) want (contents (of_node o)))
    [ ([ ("k", 5) ], 5); ([], 3) ]

(* A custom kernel on CPU:1 that takes [k], bound to 2, and closes over a
   variable named [closure], bound to 5, storing [a] plus both into [o1],
   beside a kernel of the schedule's own storing [a] plus [k] into [o0]: each
   store with its own values. *)
let custom_kernel_closes ?(devices = devices) closure () =
  let var name x =
    let v = Ops.variable name (`Int Bigint.zero) (`Int (Bigint.of_int 8)) in
    Shape.bind v (`Int (Bigint.of_int x))
  in
  let k = var "k" 2 and m = var closure 5 in
  let add = function
    | [ out; v; src ] ->
        let i = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
        let at b = Ops.index b [ i ] in
        let sum = Ops.O.(at src + Ops.cast v Int32 + Ops.cast m Int32) in
        Ops.sink
          ~kernel:(Ops.kernel_info ())
          [ Ops.end_ (Ops.store (at out) sum) [ i ] ]
    | _ -> invalid_arg "add takes three arguments"
  in
  let a = Ops.new_buffer "CPU:1" 4 Int32 in
  let o0 = Ops.new_buffer "CPU:1" 4 Int32
  and o1 = Ops.new_buffer "CPU:1" 4 Int32 in
  let k4 = Shape.reshape (Ops.cast k Int32) [ Int 1 ] in
  let s0 = Ops.after o0 [ Ops.store o0 Ops.O.(a + k4) ] in
  let big = Ops.sink [ s0; List.hd (Shape.custom_kernel [ o1; k; a ] add) ] in
  let s, vars, storage = linked ~devices big in
  Engine.run ~vars s [||];
  let of_node u = List.find (fun s -> s.node == u) storage in
  let xs = Array.map Dtype.Value.to_int (of_node a).before in
  let plus v = Array.map (fun x -> `Int (Bigint.of_int (x + v))) xs in
  equal values ~msg:"the schedule's" (plus 2) (contents (of_node o0));
  equal values ~msg:"the custom kernel's" (plus 7) (contents (of_node o1))

(* The custom kernels whose variable sits among their buffers. *)
(* [name] leaves in slot [out] the integers [f] computes from what each slot
   held before. *)
let runs_to ?(devices = devices) name out f () =
  let s, vars, storage = linked ~devices (program name) in
  Engine.run ~vars s [||];
  let before k = Array.map Dtype.Value.to_int (slot storage k).before in
  equal values
    (Array.map (fun n -> `Int (Bigint.of_int n)) (f before))
    (contents (slot storage out))

let sum = Array.fold_left ( + ) 0

(* Functions and kernels with stages and sums, each result plus 1:
   custom_kernel_stage and custom_kernel_stage_table stage [r] over [r < 4]
   and read the stage at 2, the element read, or the element 2 of the table
   [10; 20; 30; 40] at that index; reduce_axis_and_range and
   reduce_singleton_axis_and_range sum their input's axes and a range of 3;
   stage_padded_slice stores the rows [r] of the stage of [r] padded by one
   each side, over [r < 2]; stage_free_range_reg reads at 2 a register stage
   of [10 i + j] over [j < 4], and stage_no_range_reg adds 1 to one of
   [10 i], over [i < 3]; after_sum is the sum of slot 2 plus slot 3's element
   ordered after the sum. *)
let stages_and_sums =
  [
    ("custom_kernel_stage", 2, fun _ -> [| 3 |]);
    ("custom_kernel_stage_table", 2, fun _ -> [| 31 |]);
    ("reduce_axis_and_range", 4, fun b -> [| (3 * sum (b 3)) + 1 |]);
    ("reduce_singleton_axis_and_range", 2, fun b -> [| (3 * sum (b 1)) + 1 |]);
    ("stage_padded_slice", 2, fun _ -> [| 1; 1; 1; 1; 1; 2; 2; 1 |]);
    ("stage_free_range_reg", 2, fun _ -> [| 3; 13; 23 |]);
    ("stage_no_range_reg", 2, fun _ -> [| 2; 12; 22 |]);
    ("after_sum", 4, fun b -> [| sum (b 2) + (b 3).(0) |]);
  ]

let custom_kernel_scalars ?(devices = devices) () =
  [
    test "a custom kernel reads the run's binding of its variable"
      (custom_kernel_rebound ~devices);
    test "a custom kernel sized by its variable runs the run's binding"
      (custom_kernel_sized_by_variable ~devices);
    cases "a custom kernel that closes over a bound variable reads its value"
      ~id:Fun.id [ "m"; "k" ]
      (fun closure -> custom_kernel_closes ~devices closure ());
    test
      "custom kernels run as one that read one name at two values read each \
       its own" (custom_kernels_of_one_name ~devices ~n:1 (3, 5));
    test "custom kernels run as one that read one name at one value read it"
      (custom_kernels_of_one_name ~devices ~n:1 (3, 3));
    test "a custom kernel runs as one with a kernel that reads its name"
      (custom_kernel_beside_one_by_name ~devices);
    test
      "custom kernels of many elements run as one that read one name at two \
       values read each its own"
      (custom_kernels_of_one_name ~devices ~n:4 (3, 5));
    test "a custom kernel takes a variable between its buffers"
      (runs_a_custom_kernel_scalar ~devices "custom_kernel_scalar");
    test "a custom kernel takes a variable before its buffers"
      (runs_a_custom_kernel_scalar ~devices "custom_kernel_scalar_first");
    test "a custom kernel takes a variable beside another kernel's"
      (runs_beside_another_variable ~devices);
    test
      "a custom kernel that ignores a buffer takes a variable last and reads \
       its own"
      (runs_a_custom_kernel_scalar ~devices ~k:[ ("k", 10) ] ~c:3 ~a:5 ~out:6
         "custom_kernel_unused_then_variable");
  ]

(* assign_bitcast: slot 2 takes the bits of slot 3 plus one, a float, and slot 4
   that float. *)
let stores_through_a_bitcast () =
  let s, vars, storage = linked (program "assign_bitcast") in
  Engine.run ~vars s [||];
  let sums =
    Array.map (fun x -> Dtype.Value.to_float x +. 1.) (slot storage 3).before
  in
  let bits =
    Array.map (fun x -> `Int (Bigint.of_int32 (Int32.bits_of_float x))) sums
  in
  equal values
    (Array.map (Dtype.truncate Uint32) bits)
    (contents (slot storage 2));
  equal values (floats sums) (contents (slot storage 4))

(* variable_reduce sums the first v rows of its input. *)
let runs_each_binding () =
  let big = program "variable_reduce" in
  let s, _, storage = linked big in
  List.iter
    (fun v ->
      let vars = [ ("v", v) ] in
      refill storage;
      Engine.run ~vars s [||];
      equal slot_values ~msg:(string_of_int v)
        (expected ~vars big storage)
        (by_slot storage contents))
    [ 1; 10; 5 ]

(* Each device [CPU:k] stores twice the window of four of its twelve floats
   that starts at its start [k], plus one; the starts are variables of one
   name. Its kernels differ in their device and start alone, so they are one
   program, which each call launches with its own start. *)
let windows_at starts =
  let window (d, start) =
    let a = Ops.new_buffer d 12 Float32 and c = Ops.new_buffer d 4 Float32 in
    let v =
      Shape.bind
        (Ops.variable "start" (Dtype.Value.of_int 0) (Dtype.Value.of_int 8))
        (`Int (Bigint.of_int start))
    in
    let w = Shape.shrink a [ Some (Sym v, Sym Ops.O.(v + int 4)) ] in
    Ops.after c [ Ops.store c Ops.O.((w * float 2.) + float 1.) ]
  in
  Ops.sink (List.map window starts)

let one_program_per_start () =
  let big = windows_at [ ("CPU:1", 0); ("CPU:2", 5); ("CPU:3", 8) ] in
  let compiled, vars = schedule big in
  equal (list (pair string int)) ~msg:"no run binds the start" [] vars;
  let libs =
    List.filter_map
      (fun u ->
        if Ops.op u <> Program then None
        else Some (Uops.binary_of u))
      (Ops.toposort ~calls:Enter compiled)
  in
  equal int ~msg:"programs" 1 (List.length (List.sort_uniq String.compare libs));
  let s, vars, storage = linked big in
  Engine.run ~vars s [||];
  equal slot_values (expected big storage) (by_slot storage contents)

(* Two kernels on CPU:1 store twice the first [n] of their eight floats, plus
   one, for [n] a variable bound to 3 in one and 6 in the other: their grids
   read the value their calls pass, and at [DEBUG=2] so do their reports'
   throughputs. *)
let sizes_from_calls ?(devices = devices) ?(debug = 0) () =
  let prefix n =
    let a = Ops.new_buffer "CPU:1" 8 Float32
    and c = Ops.new_buffer "CPU:1" 8 Float32 in
    let n =
      Shape.bind
        (Ops.variable "n" (Dtype.Value.of_int 1) (Dtype.Value.of_int 8))
        (`Int (Bigint.of_int n))
    in
    let first b = Shape.shrink b [ Some (Int 0, Sym n) ] in
    Ops.after c [ Ops.store (first c) Ops.O.((first a * float 2.) + float 1.) ]
  in
  let big = Ops.sink (List.map prefix [ 3; 6 ]) in
  let storage =
    Setting.context
      [ B (Setting.debug, debug) ]
      (fun () ->
        let s, vars, storage = linked ~devices big in
        Engine.run ~vars s [||];
        storage)
  in
  equal slot_values (expected big storage) (by_slot storage contents)

(* A kernel on CPU:1 adds two elements of four to two elements of eight,
   each window starting at a variable named [s] bound to 1, of the ranges
   [0, 2] and [0, 6] its buffer allows, the second of type [dtype]. One name
   is one value: a run that binds [s] to 0 moves both windows. *)
let one_name_two_ranges ?(devices = devices) dtype =
  let start ?dtype hi =
    Shape.bind
      (Ops.variable ?dtype "s" (Dtype.Value.of_int 0) (Dtype.Value.of_int hi))
      (`Int Bigint.one)
  in
  let window b s = Shape.shrink b [ Some (Sym s, Sym Ops.O.(s + int 2)) ] in
  let a = Ops.new_buffer "CPU:1" 4 Float32
  and b = Ops.new_buffer "CPU:1" 8 Float32
  and o = Ops.new_buffer "CPU:1" 2 Float32 in
  let x = Ops.O.(window a (start 2) + window b (start ~dtype 6)) in
  let big = Ops.sink [ Ops.after o [ Ops.store o x ] ] in
  let s, vars, storage = linked ~devices big in
  equal (list (pair string int)) ~msg:"the run's variables" [ ("s", 1) ] vars;
  let vars = [ ("s", 0) ] in
  Engine.run ~vars s [||];
  equal slot_values (expected ~vars big storage) (by_slot storage contents)

(* A kernel on CPU:1 stores [s + s * s] for variables named [s] of types
   int32 and int64, bound to 100000, the product read through the int64 one:
   past int32's range, which one value of both types computes in 64 bits,
   whichever variable the sum reads first. *)
let one_name_two_types ?(devices = devices) order =
  let s dtype =
    Shape.bind
      (Ops.variable ~dtype "s" (Dtype.Value.of_int 0)
         (Dtype.Value.of_int 100000))
      (Dtype.Value.of_int 100000)
  in
  let narrow = Ops.cast (s Int32) Int64 and wide = s Int64 in
  let product = Ops.O.(wide * wide) in
  let x =
    if order = "int32 first" then Ops.O.(narrow + product)
    else Ops.O.(product + narrow)
  in
  let o = Ops.new_buffer "CPU:1" 1 Int64 in
  let big =
    Ops.sink [ Ops.after o [ Ops.store o (Shape.reshape x [ Int 1 ]) ] ]
  in
  let s, vars, storage = linked ~devices big in
  Engine.run ~vars s [||];
  equal slot_values (expected ~vars big storage) (by_slot storage contents)

(* The bits of a float64 padded twice, on the outer padding alone: its element
   is invalid in the inner pad, so the kernel's float64 branch folds away and
   the padding's value [v] must keep the float64's width. *)
let bitcast_of_outer_padding v () =
  let x = Shape.reshape (Ops.float ~dtype:Float64 2.) [ Int 1 ] in
  let o = Ops.new_buffer "CPU" 1 Int64 in
  let pad ?value u p = Shape.pad ?value u [ Some (Int p, Int (1 - p)) ] in
  let outer = pad ~value:(`Float v) (pad x 0) 1 in
  let bits = Ops.bitcast (Shape.shrink outer [ Some (Int 0, Int 1) ]) Int64 in
  let big = Ops.sink [ Ops.after o [ Ops.store o bits ] ] in
  let s, vars, storage = linked big in
  Engine.run ~vars s (slots storage);
  equal (list values)
    [ [| `Int (Bigint.of_int64 (Int64.bits_of_float v)) |] ]
    (List.map contents storage)

(* The pad_idiv programs pad integer quotients of their inputs: of the inputs
   themselves, of a stored divisor at least 1, padded twice, and shrunk. Their
   inputs hold no zero: a kernel divides by zero only on the padding, which
   reads zeros and whose quotients the pad drops. *)
let divides_on_padding name () =
  let big = program name in
  let s, vars, storage = linked big in
  let nonzero : Dtype.value -> Dtype.value = function
    | `Int z when Bigint.equal z Bigint.zero -> `Int Bigint.one
    | v -> v
  in
  let storage =
    List.map (fun s -> { s with before = Array.map nonzero s.before }) storage
  in
  refill storage;
  Engine.run ~vars s (slots storage);
  equal slot_values (expected ~vars big storage) (by_slot storage contents)

(* A quotient of bytes by a constant table of 1 to [n], padded by two bytes on
   each side: on the padding the table's load reads 0, which no entry of the
   table holds. [n] is large enough that the padding stays a run-time condition
   of the kernel's lanes. *)
let divides_by_a_table ?(devices = devices) d () =
  let n = 252 and pad = 2 in
  let a = Ops.new_buffer d n Uint8
  and out = Ops.new_buffer d (n + (2 * pad)) Uint8 in
  let bytes = String.init n (fun i -> Char.chr (i + 1)) in
  let table = Ops.v Op.Binary ~arg:(Ops.Bytes bytes) in
  let q = Ops.O.((a // table) + (a % table)) in
  let padded = Shape.pad q [ Some (Ops.Int pad, Ops.Int pad) ] in
  let big = Ops.sink [ Ops.after out [ Ops.store out padded ] ] in
  let s, vars, storage = linked ~devices big in
  Engine.run ~vars s [||];
  let of_node u = List.find (fun s -> s.node == u) storage in
  let xs = Array.map Dtype.Value.to_int (of_node a).before in
  let at i =
    if i < pad || i >= n + pad then 0
    else
      let x = xs.(i - pad) and d = i - pad + 1 in
      (x / d) + (x mod d)
  in
  equal values
    (Array.init (n + (2 * pad)) (fun i -> `Int (Bigint.of_int (at i))))
    (contents (of_node out))

(* [parameterized name] is the recorded program [name] linked with its buffers
   made parameters of their slots, which each run binds, and the program with
   those parameters. *)
let parameterized ?(devices = devices) name =
  let big = program name in
  let parameters =
    List.filter_map
      (fun n ->
        if Ops.op n = Buffer then Some (n, Ops.replace ~op:Param n) else None)
      (Ops.toposort ~calls:Skip big)
  in
  let linear, vars = Schedule.create_linear_with_vars ~cores:Cores.host big in
  let compiled =
    compile ~devices
      (Ops.substitute ~calls:Skip ~pass:Fixed_point linear parameters)
  in
  ( Engine.link ~devices compiled,
    vars,
    Ops.substitute ~calls:Skip ~pass:Fixed_point big parameters )

let runs_on_its_slots ?(devices = devices) name () =
  let s, vars, big = parameterized ~devices name in
  List.iter
    (fun storage ->
      Engine.run ~vars s (slots storage);
      equal slot_values (expected ~vars big storage) (by_slot storage contents))
    [ storage_of ~devices big; storage_of ~devices ~seed:5 big ]

(* A copy reads bytes wherever they start: copy runs with the buffer of slot 1,
   which only its copy reads, 4 bytes past a 16-byte boundary, the parameter's
   phase 0. *)
let copies_off_phase () =
  let s, vars, big = parameterized "copy" in
  let storage =
    List.map
      (fun s ->
        if s.arg.slot <> 1 then s
        else { s with buffer = off_phase s.buffer })
      (storage_of big)
  in
  Engine.run ~vars s (slots storage);
  equal slot_values (expected ~vars big storage) (by_slot storage contents)

(* Floats 1 to 8 of sixteen store one more than twice floats 0 to 7, doubled
   into other storage first: the store's kernel reads that storage, which a
   kernel that runs before it fills, and stores into a view of its floats. *)
let stores_rows_read_elsewhere () =
  let buffer = Ops.new_buffer "CPU" 16 Float32
  and other = Ops.new_buffer "CPU" 8 Float32 in
  let run lo = Shape.shrink buffer [ Some (Int lo, Int (lo + 8)) ] in
  let doubled =
    Ops.after other [ Ops.store other Ops.O.(run 0 * float 2.) ]
  in
  let big =
    Ops.sink
      [ Ops.after buffer [ Ops.store (run 1) Ops.O.(doubled + float 1.) ] ]
  in
  let s, vars, storage = linked big in
  Engine.run ~vars s [||];
  equal slot_values (expected big storage) (by_slot storage contents)

(* A range of three trips around a call of a kernel that adds one to four
   floats, each trip on the next four of twelve. *)
(* A kernel that stores its parameter 1 plus one into its parameter 0, four
   floats. *)
let add_one =
  let open Ops.O in
  let i = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
  let at slot = Ops.index (Shape.placeholder ~slot [ 4 ] Float32) [ i ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"add_one" ())
    [ Ops.end_ (Ops.store (at 0) (at 1 + Ops.O.float 1.)) [ i ] ]

let runs_once_per_trip () =
  let open Ops.O in
  let one = add_one in
  let out = Ops.new_buffer "CPU" 12 Float32
  and src = Ops.new_buffer "CPU" 12 Float32 in
  let r = Ops.range ~axis_type:Weak (Int 3) [ 7 ] in
  let trip b =
    Shape.shrink b
      [ Some (Sym (r * Ops.int 4), Sym ((r * Ops.int 4) + Ops.int 4)) ]
  in
  let linear =
    Ops.v Op.Linear
      ~src:[ Ops.end_ (Ops.call one [ trip out; trip src ]) [ r ] ]
  in
  let compiled = compile ~devices linear in
  let xs = Array.init 12 Float.of_int in
  let out_buffer = Run.buffer host Float32 (floats (Array.make 12 0.)) in
  let s =
    Engine.link ~devices
      ~bound:
        [
          (out, out_buffer); (src, Run.buffer host Float32 (floats xs));
        ]
      compiled
  in
  Engine.run s [||];
  equal values
    (floats (Array.map (fun x -> x +. 1.) xs))
    (Run.values Float32 out_buffer)

(* A scan of three trips, scheduled from its program: a carry [c] of four
   floats, updated in place, and rows of four of [xs] and [ys]. Each trip stores
   [c * 2] into its row of [ys], then adds its row of [xs] to [c]. *)
let scans_with_a_carry () =
  let k = 4 and n = 3 in
  let cpu = "CPU" in
  let p slot = Shape.param ~shape:[ Int k ] ~device:cpu slot Float32 in
  let body =
    Ops.sink
      [
        Ops.store (p 0) Ops.O.(p 0 + p 1);
        Ops.store (p 2) Ops.O.(p 0 * float 2.);
      ]
  in
  let c = Ops.new_buffer cpu k Float32
  and xs = Ops.new_buffer cpu (n * k) Float32
  and ys = Ops.new_buffer cpu (n * k) Float32 in
  let r = Ops.range ~axis_type:Loop (Int n) [ 100 ] in
  let row b =
    Shape.shrink b
      [ Some (Sym Ops.O.(r * int k), Sym Ops.O.((r * int k) + int k)) ]
  in
  let e =
    Ops.end_ (Ops.call ~precompile:true body [ c; row xs; row ys ]) [ r ]
  in
  let linear, _ =
    Schedule.create_linear_with_vars ~cores:Cores.host
      (Ops.sink [ Ops.after c [ e ]; Ops.after ys [ e ] ])
  in
  let compiled = compile ~devices linear in
  let c0 = [| 1.; 2.; 3.; 4. |] and x = Array.init (n * k) Float.of_int in
  let c_buffer = Run.buffer host Float32 (floats c0)
  and ys_buffer = Run.buffer host Float32 (floats (Array.make (n * k) 0.)) in
  let s =
    Engine.link ~devices
      ~bound:
        [
          (c, c_buffer);
          (xs, Run.buffer host Float32 (floats x));
          (ys, ys_buffer);
        ]
      compiled
  in
  Engine.run s [||];
  let carry = Array.copy c0 and y = Array.make (n * k) 0. in
  for t = 0 to n - 1 do
    for j = 0 to k - 1 do
      y.((t * k) + j) <- carry.(j) *. 2.;
      carry.(j) <- carry.(j) +. x.((t * k) + j)
    done
  done;
  equal values ~msg:"the carry" (floats carry) (Run.values Float32 c_buffer);
  equal values ~msg:"the rows" (floats y) (Run.values Float32 ys_buffer)

(* The number of calls in the body of [linear]'s one loop. *)
let calls_in_loop linear =
  match Ops.src linear with
  | [ l ] when Ops.op l = End -> List.length (Ops.src (Ops.nth l 0))
  | entries -> failf "%d entries, not one loop" (List.length entries)

(* Carries of a scan of five trips on [d], each of 64 floats updated in place by
   its function of [updates] from its initial values: the number of calls in the
   scan's body, and each carry's bits after the run. *)
let carried ?(devices = devices) d updates =
  let k = 64 and n = 5 in
  let p slot = Shape.param ~shape:[ Int k ] ~device:d slot Float32 in
  let cs = List.map (fun _ -> Ops.new_buffer d k Float32) updates in
  let body =
    Ops.sink (List.mapi (fun i (f, _) -> Ops.store (p i) (f (p i))) updates)
  in
  let r = Ops.range ~axis_type:Loop (Int n) [ 100 ] in
  let e = Ops.end_ (Ops.call ~precompile:true body cs) [ r ] in
  let linear, vars =
    Schedule.create_linear_with_vars ~cores:Cores.host
      (Ops.sink (List.map (fun c -> Ops.after c [ e ]) cs))
  in
  let calls = calls_in_loop linear in
  let bufs =
    List.map2
      (fun c (_, init) ->
        (c, Run.buffer (devices d) Float32 (floats (Array.init k init))))
      cs updates
  in
  let s = Engine.link ~devices ~bound:bufs (compile ~devices linear) in
  Engine.run ~vars s [||];
  let bits b =
    Array.map
      (function `Float x -> Int64.bits_of_float x | _ -> fail "a float")
      (Run.values Float32 b)
  in
  (calls, List.map (fun (_, b) -> bits b) bufs)

(* Two independent carries of one domain run as one call each trip, and leave
   the bits each leaves run as a scan of its own. *)
let carries_merged ?(devices = devices) d () =
  let scaled =
    ( (fun c -> Ops.O.((c * float 1.5) + float 0.25)),
      fun j -> Float.of_int j /. 7. )
  and decayed =
    ( (fun c -> Ops.O.(Ops.exp2 (c * float (-0.5)) * float 0.75)),
      fun j -> Float.of_int (j - 30) /. 3. )
  in
  let calls, merged = carried ~devices d [ scaled; decayed ] in
  equal int ~msg:"calls a trip" 1 calls;
  let apart =
    List.concat_map
      (fun u -> snd (carried ~devices d [ u ]))
      [ scaled; decayed ]
  in
  equal (list (array int64)) ~msg:"bits" apart merged

(* [plus j u] is [u] plus a variable [k] of [0, 1] bound to [j]. *)
let plus j u =
  let k = Ops.variable "k" (`Int Bigint.zero) (`Int Bigint.one) in
  Ops.O.(u + Ops.cast (Shape.bind k (`Int (Bigint.of_int j))) Float32)

(* Two carries of a scan whose updates read one name bound to 0 in one and 1
   in the other run as one call each trip, and leave the bits each leaves run
   as a scan of its own. *)
let carries_bound_apart ?(devices = devices) d () =
  let update j =
    ( (fun c -> plus j Ops.O.(c * float 0.5)),
      fun i -> Float.of_int (i - (8 * j)) )
  in
  let calls, merged = carried ~devices d [ update 0; update 1 ] in
  equal int ~msg:"calls a trip" 1 calls;
  let apart =
    List.concat_map
      (fun u -> snd (carried ~devices d [ u ]))
      [ update 0; update 1 ]
  in
  equal (list (array int64)) ~msg:"bits" apart merged

(* Two outputs of a scan of three trips on [d] that share the exponential of
   an input, plus one name bound to 0 in one and 1 in the other, are one
   kernel that stores each with its own value. *)
let stores_bound_apart ?(devices = devices) d () =
  let p slot = Shape.param ~shape:[ Int 4 ] ~device:d slot Float32 in
  let shared = Ops.exp2 (p 2) in
  let body =
    Ops.sink
      [ Ops.store (p 0) (plus 0 shared); Ops.store (p 1) (plus 1 shared) ]
  in
  let bufs = List.init 3 (fun _ -> Ops.new_buffer d 4 Float32) in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let e = Ops.end_ (Ops.call ~precompile:true body bufs) [ r ] in
  let outs = [ List.nth bufs 0; List.nth bufs 1 ] in
  let linear, vars =
    Schedule.create_linear_with_vars ~cores:Cores.host
      (Ops.sink (List.map (fun b -> Ops.after b [ e ]) outs))
  in
  equal int ~msg:"calls a trip" 1 (calls_in_loop linear);
  let x = [| -1.; 0.; 2.; 3. |] in
  let given =
    List.map (fun _ -> Run.buffer (devices d) Float32 (floats x)) bufs
  in
  let s =
    Engine.link ~devices ~bound:(List.combine bufs given)
      (compile ~devices linear)
  in
  Engine.run ~vars s [||];
  let stored j = floats (Array.map (fun v -> (2. ** v) +. Float.of_int j) x) in
  equal (list values) [ stored 0; stored 1 ]
    (List.map (Run.values Float32) [ List.nth given 0; List.nth given 1 ])

(* A scan of three trips on [d] whose body stores its input plus [k], bound to
   2, and calls a custom kernel that takes [k] and closes over a variable named
   [closure], bound to 5, storing the input plus both: one kernel a trip, each
   store with its own values, whether [closure] is another name, which a run
   reads by name, or [k] at another value, which the loop's call passes. *)
let custom_kernel_closes_in_loop ?(devices = devices) ~closure d () =
  let p slot = Shape.param ~shape:[ Int 4 ] ~device:d slot Float32 in
  let var name x =
    let v = Ops.variable name (`Int Bigint.zero) (`Int (Bigint.of_int 8)) in
    Shape.bind v (`Int (Bigint.of_int x))
  in
  let k = var "k" 2 and m = var closure 5 in
  let add = function
    | [ out; v; src ] ->
        let i = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
        let at b = Ops.index b [ i ] in
        Ops.sink
          ~kernel:(Ops.kernel_info ())
          [
            Ops.end_
              (Ops.store (at out)
                 Ops.O.(at src + Ops.cast v Float32 + Ops.cast m Float32))
              [ i ];
          ]
    | _ -> invalid_arg "add takes three arguments"
  in
  let body =
    Ops.sink
      [
        Ops.store (p 0) Ops.O.(p 2 + Ops.cast k Float32);
        List.hd (Shape.custom_kernel [ p 1; k; p 2 ] add);
      ]
  in
  let bufs = List.init 3 (fun _ -> Ops.new_buffer d 4 Float32) in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let e = Ops.end_ (Ops.call ~precompile:true body bufs) [ r ] in
  let outs = [ List.nth bufs 0; List.nth bufs 1 ] in
  let linear, vars =
    Schedule.create_linear_with_vars ~cores:Cores.host
      (Ops.sink (List.map (fun b -> Ops.after b [ e ]) outs))
  in
  equal int ~msg:"calls a trip" 1 (calls_in_loop linear);
  let x = [| -1.; 0.; 2.; 3. |] in
  let given =
    List.map (fun _ -> Run.buffer (devices d) Float32 (floats x)) bufs
  in
  let s =
    Engine.link ~devices ~bound:(List.combine bufs given)
      (compile ~devices linear)
  in
  Engine.run ~vars s [||];
  let plus j = floats (Array.map (fun v -> v +. Float.of_int j) x) in
  equal (list values) [ plus 2; plus 7 ]
    (List.map (Run.values Float32) [ List.nth given 0; List.nth given 1 ])

(* A copy of 16 floats of the host stored into rows 2 and 3 of an 8 by 8 grid
   on a software device writes those rows, and leaves the others as they
   were. *)
let copies_into_rows () =
  let dst = Ops.new_buffer "CPU:1" 64 Float32 in
  let src = Ops.new_buffer "CPU" 16 Float32 in
  let rows =
    Shape.shrink (Shape.reshape dst [ Int 8; Int 8 ]) [ Some (Int 2, Int 4); None ]
  in
  let value = Ops.copy_to_device (Shape.reshape src [ Int 2; Int 8 ]) "CPU:1" in
  let linear, _ =
    Schedule.create_linear_with_vars ~cores:Cores.host
      (Ops.sink [ Ops.after dst [ Ops.store rows value ] ])
  in
  let before = Array.init 64 (fun i -> -.Float.of_int i) in
  let x = Array.init 16 (fun i -> Float.of_int (i + 1)) in
  let into = Run.buffer (device "CPU:1") Float32 (floats before) in
  let s =
    Engine.link ~devices
      ~bound:[ (dst, into); (src, Run.buffer host Float32 (floats x)) ]
      (compile linear)
  in
  Engine.run s [||];
  let expected = Array.mapi (fun i v -> if i >= 16 && i < 32 then x.(i - 16) else v) before in
  equal values (floats expected) (Run.values Float32 into)

(* [linear] with the flag [f] of its loop first copied from a new buffer of [d],
   which holds the flag's first value, and that buffer. A loop's flag is storage
   the schedule makes, which no run binds. *)
let flag_from d f linear =
  let init = Ops.new_buffer d 1 Bool in
  (init, Ops.v Op.Linear ~src:(Shape.store_call f init :: Ops.src linear))

(* A loop of at most [max] trips on the device [d] around a call that adds one
   to a carry [c], from zero, stores the carry it read into its row of [ys], 16
   bytes apart, and stores into the flag [f] whether the new carry is below
   [limit], which the call loads, so that every limit runs one program. The
   flag starts as [0 < limit]. Is the schedule linked, with its carry and its
   rows. *)
let flag_loop ?(devices = devices) d (limit, max) =
  let at = d in
  let p slot dt = Shape.param ~shape:[ Int 1 ] ~device:at slot dt in
  let next = Ops.O.(p 0 Float32 + float 1.) in
  let body =
    Ops.sink
      [
        Ops.store (p 0 Float32) next;
        Ops.store (p 1 Bool) Ops.O.(next < p 3 Float32);
        Ops.store (p 2 Float32) (p 0 Float32);
      ]
  in
  let c = Ops.new_buffer at 1 Float32
  and f = Ops.new_buffer at 1 Bool
  and ys = Ops.new_buffer at (4 * max) Float32
  and lim = Ops.new_buffer at 1 Float32 in
  let r = Ops.range ~axis_type:Loop (Int max) [ 100 ] in
  let row =
    Shape.shrink ys
      [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 1)) ]
  in
  let e =
    Ops.backedge
      (Ops.call ~precompile:true body [ c; f; row; lim ])
      ~loop:r ~cond:f
  in
  let linear, _ =
    Schedule.create_linear_with_vars ~cores:Cores.host
      (Ops.sink [ Ops.after c [ e ]; Ops.after f [ e ]; Ops.after ys [ e ] ])
  in
  let init, linear = flag_from d f linear in
  let compiled = compile ~devices linear in
  let on = devices d in
  let c_buffer = Run.buffer on Float32 [| `Float 0. |]
  and f_buffer = Run.buffer on Bool [| `Bool (0 < limit) |]
  and ys_buffer = Run.buffer on Float32 (floats (Array.make (4 * max) (-1.)))
  and lim_buffer = Run.buffer on Float32 [| `Float (Float.of_int limit) |] in
  let s =
    Engine.link ~devices
      ~bound:
        [
          (c, c_buffer); (init, f_buffer); (ys, ys_buffer); (lim, lim_buffer);
        ]
      compiled
  in
  (s, c_buffer, ys_buffer)

(* A run of [flag_loop] makes [min max limit] trips, none for a limit of [0]
   or below. *)
let loops_while_a_flag_holds ?devices d (limit, max) =
  let s, c_buffer, ys_buffer = flag_loop ?devices d (limit, max) in
  Engine.run s [||];
  let trips = Int.min max (Int.max limit 0) in
  equal values ~msg:"the carry"
    (floats [| Float.of_int trips |])
    (Run.values Float32 c_buffer);
  equal values ~msg:"the rows"
    (floats
       (Array.init (4 * max) (fun i ->
            if i mod 4 = 0 && i / 4 < trips then Float.of_int (i / 4) else -1.)))
    (Run.values Float32 ys_buffer)

(* A kernel that stores [f] of the one element of its parameter 1, of dtype
   [src], into the one element of its parameter 0, of dtype [dst]. *)
let scalar_kernel name ~dst ~src f =
  let i = Ops.range ~axis_type:Weak (Int 1) [ 0 ] in
  let at slot dt = Ops.index (Shape.placeholder ~slot [ 1 ] dt) [ i ] in
  Ops.sink ~kernel:(Ops.kernel_info ~name ())
    [ Ops.end_ (Ops.store (at 0 dst) (f (at 1 src))) [ i ] ]

(* A kernel that stores whether twice the one float of its parameter 1 is
   below the one float of its parameter 2, the limit, into its parameter 0:
   every limit runs one program. *)
let below =
  let i = Ops.range ~axis_type:Weak (Int 1) [ 0 ] in
  let at slot dt = Ops.index (Shape.placeholder ~slot [ 1 ] dt) [ i ] in
  Ops.sink ~kernel:(Ops.kernel_info ~name:"below" ())
    [
      Ops.end_
        (Ops.store (at 0 Bool) Ops.O.(at 1 Float32 * float 2. < at 2 Float32))
        [ i ];
    ]

(* A buffer of [d] holding [limit]. *)
let limit_buffer d limit =
  Run.buffer d Float32 [| `Float (Float.of_int limit) |]

(* A loop of at most [max] trips on the device [d] around four calls each trip:
   the first stores [c + 1] into [t], and the others read it to store [2 t] into
   the carry [c], whether [2 t] is below [limit] into the flag [f], and [t] into
   its row of [ys], 16 bytes apart. Each call after the first reads what the
   first wrote in the same trip, and the first reads the carry the second wrote
   in the trip before. The flag starts as [0 < limit]. *)
let loops_around_calls ?(devices = devices) d (limit, max) =
  let at = d in
  let k name dst f = scalar_kernel name ~dst ~src:Float32 f in
  let inc = k "inc" Float32 (fun x -> Ops.O.(x + float 1.))
  and twice = k "twice" Float32 (fun x -> Ops.O.(x * float 2.))
  and copy = k "copy" Float32 Fun.id in
  let c = Ops.new_buffer at 1 Float32
  and t = Ops.new_buffer at 1 Float32
  and f = Ops.new_buffer at 1 Bool
  and ys = Ops.new_buffer at (4 * max) Float32
  and lim = Ops.new_buffer at 1 Float32 in
  let r = Ops.range ~axis_type:Loop (Int max) [ 100 ] in
  let row =
    Shape.shrink ys
      [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 1)) ]
  in
  let linear =
    Ops.v Op.Linear
      ~src:
        [
          Ops.backedge
            (Ops.v Op.Linear
               ~src:
                 [
                   Ops.call inc [ t; c ];
                   Ops.call twice [ c; t ];
                   Ops.call below [ f; t; lim ];
                   Ops.call copy [ row; t ];
                 ])
            ~loop:r ~cond:f;
        ]
  in
  let init, linear = flag_from d f linear in
  let compiled = compile ~devices linear in
  let on = devices d in
  let c_buffer = Run.buffer on Float32 [| `Float 0. |]
  and ys_buffer = Run.buffer on Float32 (floats (Array.make (4 * max) (-1.))) in
  let s =
    Engine.link ~devices
      ~bound:
        [
          (c, c_buffer);
          (t, Run.buffer on Float32 [| `Float (-1.) |]);
          (init, Run.buffer on Bool [| `Bool (0 < limit) |]);
          (ys, ys_buffer);
          (lim, limit_buffer on limit);
        ]
      compiled
  in
  Engine.run s [||];
  (* The carry after [k] trips is [2^(k+1) - 2]; a trip runs while the carry
     before it is below [limit], at most [max] of them. *)
  let carry k = Float.of_int ((1 lsl (k + 1)) - 2) in
  let rec trips k =
    if k < max && carry k < Float.of_int limit then trips (k + 1) else k
  in
  let trips = trips 0 in
  cover "no trip" (trips = 0);
  cover "stops on its flag" (trips > 0 && trips < max);
  cover "stops at its trips" (trips = max);
  equal values ~msg:"the carry"
    (floats [| carry trips |])
    (Run.values Float32 c_buffer);
  equal values ~msg:"the rows"
    (floats
       (Array.init (4 * max) (fun i ->
            if i mod 4 = 0 && i / 4 < trips then carry (i / 4) +. 1. else -1.)))
    (Run.values Float32 ys_buffer)

(* A loop's limit and trips. Limits -1 and 0 run no trip: 2 of 42 values, which
   a uniform draw of 30 misses a quarter of the time, so they are drawn half the
   time. A draw stops on its flag a fifth of the time, which 30 draws miss in
   about one run in 600: an example of each ending holds the covers on every
   run. *)
let loop_bounds =
  Gen.(pair (one_of [ int_range (-1) 0; int_range 1 40 ]) (int_range 1 6))

let loops_around_calls_on_the_host =
  prop ~count:30 ~examples:[ (0, 3); (10, 6); (40, 2) ]
    "a loop runs its calls in order each trip while its flag holds" loop_bounds
    (fun bounds -> loops_around_calls "CPU" bounds)

(* Where a loop of [loops_across_devices] computes its flag. *)
type flag_side = Host_program | Gpu_kernel

(* [loops_around_calls]'s loop with its calls on the GPU [gpu] and on the host:
   a kernel on [gpu] stores [c + 1] into [t], a move copies [t] into [t'] on
   the host, host programs store [2 t'] into [c'] and [t'] into its row of
   [ys], and a move copies [c'] into the carry [c]. The flag [f] holds whether
   [2 t] is below [limit]: a host program stores it from [t'], or a kernel on
   [gpu] from [t]. Each trip's host programs read what its kernel wrote, and
   each trip's kernel what the host programs of the trip before wrote. *)
let loops_across_devices ~devices ~flag gpu (limit, max) =
  let k name dst f = scalar_kernel name ~dst ~src:Float32 f in
  let inc = k "inc" Float32 (fun x -> Ops.O.(x + float 1.))
  and twice = k "twice" Float32 (fun x -> Ops.O.(x * float 2.))
  and copy = k "copy" Float32 Fun.id in
  let flag_device =
    match flag with Host_program -> "CPU" | Gpu_kernel -> gpu
  in
  let on_gpu = gpu and on_host = "CPU" in
  let c = Ops.new_buffer on_gpu 1 Float32
  and t = Ops.new_buffer on_gpu 1 Float32
  and t' = Ops.new_buffer on_host 1 Float32
  and c' = Ops.new_buffer on_host 1 Float32
  and f = Ops.new_buffer flag_device 1 Bool
  and ys = Ops.new_buffer on_host (4 * max) Float32
  and lim = Ops.new_buffer flag_device 1 Float32 in
  let r = Ops.range ~axis_type:Loop (Int max) [ 100 ] in
  let row =
    Shape.shrink ys
      [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 1)) ]
  in
  let flag_call =
    match flag with
    | Host_program -> Ops.call below [ f; t'; lim ]
    | Gpu_kernel -> Ops.call below [ f; t; lim ]
  in
  let linear =
    Ops.v Op.Linear
      ~src:
        [
          Ops.backedge
            (Ops.v Op.Linear
               ~src:
                 [
                   Ops.call inc [ t; c ];
                   Shape.store_call t' t;
                   Ops.call twice [ c'; t' ];
                   flag_call;
                   Ops.call copy [ row; t' ];
                   Shape.store_call c c';
                 ])
            ~loop:r ~cond:f;
        ]
  in
  let init, linear = flag_from flag_device f linear in
  let compiled = compile ~devices linear in
  let on = devices gpu in
  let one d x = Run.buffer d Float32 [| `Float x |] in
  let c_buffer = one on 0.
  and ys_buffer =
    Run.buffer host Float32 (floats (Array.make (4 * max) (-1.)))
  in
  let s =
    Engine.link ~devices
      ~bound:
        [
          (c, c_buffer);
          (t, one on (-1.));
          (t', one host (-1.));
          (c', one host (-1.));
          ( init,
            Run.buffer (devices flag_device) Bool [| `Bool (0 < limit) |] );
          (ys, ys_buffer);
          (lim, limit_buffer (devices flag_device) limit);
        ]
      compiled
  in
  Engine.run s [||];
  (* As in [loops_around_calls]: the carry after [k] trips is [2^(k+1) - 2]. *)
  let carry k = Float.of_int ((1 lsl (k + 1)) - 2) in
  let rec trips k =
    if k < max && carry k < Float.of_int limit then trips (k + 1) else k
  in
  let trips = trips 0 in
  cover "no trip" (trips = 0);
  cover "stops on its flag" (trips > 0 && trips < max);
  cover "stops at its trips" (trips = max);
  equal values ~msg:"the carry"
    (floats [| carry trips |])
    (Run.values Float32 c_buffer);
  equal values ~msg:"the rows"
    (floats
       (Array.init (4 * max) (fun i ->
            if i mod 4 = 0 && i / 4 < trips then carry (i / 4) +. 1. else -1.)))
    (Run.values Float32 ys_buffer)

(* [loops_across_devices] on the GPU [gpu], its flag computed on each side. *)
let loops_across_devices_on ~devices gpu =
  let on (flag, side) =
    prop ~count:30 ~examples:[ (0, 3); (10, 6); (40, 2) ] side loop_bounds
      (loops_across_devices ~devices ~flag gpu)
  in
  group
    "a loop of a kernel and host programs runs them in order each trip while \
     its flag holds"
    (List.map on
       [
         (Host_program, "its flag of a host program");
         (Gpu_kernel, "its flag of a kernel");
       ])

(* A loop's flag is storage of one boolean: a view of one element of wider
   storage is refused, since the engine reads the storage it names. *)
let refuses_a_view_as_flag () =
  let at = "CPU" in
  let p = Shape.param ~shape:[ Int 1 ] ~device:at 0 Float32 in
  let body = Ops.sink [ Ops.store p Ops.O.(p + float 1.) ] in
  let c = Ops.new_buffer at 1 Float32 and f = Ops.new_buffer at 2 Bool in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let flag = Shape.shrink f [ Some (Int 1, Int 2) ] in
  let e =
    Ops.backedge (Ops.call ~precompile:true body [ c ]) ~loop:r ~cond:flag
  in
  raises_match
    (function
      | Invalid_argument m ->
          String.starts_with ~prefix:"UOp verification failed" m
          && List.exists
               (String.equal "Ops.BACKEDGE")
               (String.split_on_char ' ' m)
      | _ -> false)
    (fun () ->
      Schedule.create_linear_with_vars
        ~cores:Cores.host (Ops.sink [ Ops.after c [ e ] ]))

let loops_on_the_host =
  prop "a loop runs its call while its flag holds, at most its trips"
    Gen.(pair (int_range (-1) 6) (int_range 1 5))
    (fun bounds -> loops_while_a_flag_holds "CPU" bounds)

(* The scalar arguments a loop of [n] trips gives its call, as functions of the
   range and of the trip. *)
let trip_arguments n =
  [
    ("r", Fun.id, Fun.id);
    ("2r + 1", (fun r -> Ops.O.((r * int 2) + int 1)), fun t -> (2 * t) + 1);
    ("n - 1 - r", (fun r -> Ops.O.(int (Int.pred n) - r)), fun t -> n - 1 - t);
  ]

(* A loop of [n] trips of the range [axis] on the device [d] around a call that
   stores its scalar argument, an expression of the range, into its row of [ys],
   16 bytes apart, compiled, with [ys] and the buffer the flag [f] starts from
   ([flag_from]). With [limit], the loop is a back edge, at most [n] trips,
   whose call also stores into [f] whether its argument plus one is below
   [limit]. *)
let trip_loop ?(devices = devices) ?limit ?(axis = 100) d n arg =
  let at = d in
  let index = Shape.param ~addrspace:Alu 2 Int32 in
  let p slot dt = Shape.param ~shape:[ Int 1 ] ~device:at slot dt in
  let flag =
    match limit with
    | None -> []
    | Some limit ->
        [
          Ops.store (p 1 Bool)
            (Shape.reshape Ops.O.(index + int 1 < int limit) [ Int 1 ]);
        ]
  in
  let body =
    Ops.sink (Ops.store (p 0 Int32) (Shape.reshape index [ Int 1 ]) :: flag)
  in
  let f = Ops.new_buffer at 1 Bool and ys = Ops.new_buffer at (4 * n) Int32 in
  let r = Ops.range ~axis_type:Loop (Int n) [ axis ] in
  let row =
    Shape.shrink ys
      [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 1)) ]
  in
  let call = Ops.call ~precompile:true body [ row; f; arg r ] in
  let e =
    match limit with
    | None -> Ops.end_ call [ r ]
    | Some _ -> Ops.backedge call ~loop:r ~cond:f
  in
  let linear, _ =
    Schedule.create_linear_with_vars ~cores:Cores.host
      (Ops.sink [ Ops.after ys [ e ]; Ops.after f [ e ] ])
  in
  let init, linear = flag_from d f linear in
  (compile ~devices linear, ys, init)

let reads_its_trip ?(devices = devices) ?limit d n (_, arg, expected) =
  let compiled, ys, f = trip_loop ~devices ?limit d n arg in
  let on = devices d in
  let ys_buffer =
    Run.buffer on Int32 (Array.make (4 * n) (`Int Bigint.minus_one))
  in
  (* A trip runs while the flag the trip before stored holds. *)
  let rec trips t =
    match limit with
    | Some l when t > 0 && expected (t - 1) + 1 >= l -> t
    | _ -> if t < n then trips (t + 1) else t
  in
  let trips = trips 0 in
  let s =
    Engine.link ~devices
      ~bound:
        [ (ys, ys_buffer); (f, Run.buffer on Bool [| `Bool true |]) ]
      compiled
  in
  Engine.run s [||];
  equal values
    (Array.init (4 * n) (fun i ->
         if i mod 4 = 0 && i / 4 < trips then
           `Int (Bigint.of_int (expected (i / 4)))
         else `Int Bigint.minus_one))
    (Run.values Int32 ys_buffer)

(* Loops of 5 and 9 trips, of two ranges, around one call of their trip compile
   one program: it names its trip by its slot, whatever the range and its
   trips. *)
let one_program_for_any_loop () =
  let _, arg, _ = List.nth (trip_arguments 5) 1 in
  let programs axis n =
    let compiled, _, _ = trip_loop ~axis "CPU" n arg in
    List.filter_map
      (fun u ->
        if Ops.op u <> Program then None
        else Some (Uops.binary_of u))
      (Ops.toposort ~calls:Enter compiled)
  in
  equal int 1
    (List.length
       (List.sort_uniq String.compare (programs 100 5 @ programs 101 9)))

(* A back edge whose flag is bound to a buffer reads that buffer: trip 0 stores
   [1 < 2] and trip 1 [2 < 2], so two trips of four run. *)
let reads_a_bound_flag () =
  let compiled, ys, init = trip_loop ~limit:2 "CPU" 4 (fun r -> r) in
  let f =
    List.find
      (fun u ->
        Ops.op u = Buffer && u != ys && u != init && Ops.dtype u = Dtype.Bool)
      (Ops.toposort ~calls:Enter compiled)
  in
  let int i = `Int (Bigint.of_int i) in
  let ys_buffer = Run.buffer host Int32 (Array.make 16 (int (-1))) in
  let flag = Run.buffer host Bool [| `Bool true |] in
  let s =
    Engine.link ~devices
      ~bound:
        [
          (f, flag);
          (ys, ys_buffer);
          (init, Run.buffer host Bool [| `Bool true |]);
        ]
      compiled
  in
  Engine.run s [||];
  equal values
    (Array.init 16 (fun i ->
         if i = 0 then int 0 else if i = 4 then int 1 else int (-1)))
    (Run.values Int32 ys_buffer);
  equal values [| `Bool false |] (Run.values Bool flag)

let reads_trips_on ?devices d =
  cases "a loop's call reads its trip as a scalar argument on each trip"
    ~id:(fun (name, _, _) -> name)
    (trip_arguments 5)
    (reads_its_trip ?devices d 5)

let reads_trips_under_a_back_edge ?devices d =
  prop "a back edge's call reads its trip on each trip, at most its trips"
    Gen.(pair (int_range 0 2) (int_range 1 6))
    (fun (k, limit) ->
      reads_its_trip ?devices ~limit d 4 (List.nth (trip_arguments 4) k))

(* Scalar arguments of each width from one byte to eight, signed and unsigned:
   constants, and the variables [a] and [b] a run binds. A kernel that takes
   them all holds some past padding to their width. With the value each holds
   in a run. *)
let mixed_scalars =
  let int dt i = Ops.const ~dtype:dt (`Int (Bigint.of_int i)) in
  let var dt name hi =
    Ops.variable ~dtype:dt name (Dtype.Value.of_int 0) (Dtype.Value.of_int hi)
  in
  let a = 0x12345678 and b = 0x0123456789abcdef in
  ( [ ("a", a); ("b", b) ],
    [
      (Dtype.Int8, int Int8 (-7), -7);
      (Int16, int Int16 (-1234), -1234);
      (Int32, var Int32 "a" 0x7fffffff, a);
      (Uint8, int Uint8 200, 200);
      (Int64, var Int64 "b" max_int, b);
      (Bool, Ops.const ~dtype:Bool (`Bool true), 1);
      (Uint16, int Uint16 0xbeef, 0xbeef);
      (Int64, int Int64 (-0x0fedcba987654321), -0x0fedcba987654321);
      (Int8, int Int8 99, 99);
    ] )

(* A call on [d] of one kernel whose first parameter is [y], then each of
   [mixed_scalars], then [x]: it stores each scalar, widened to 64 bits, plus
   its element of [x] into its element of [y]. *)
let reads_mixed_scalars ?(devices = devices) d () =
  let vars, scalars = mixed_scalars in
  let n = List.length scalars in
  let param slot = Shape.param ~shape:[ Int n ] ~device:d slot Int64 in
  let widened i (dt, _, _) =
    Ops.cast (Shape.param ~addrspace:Alu (i + 1) dt) Int64
  in
  let added = param (n + 1) in
  let sum = Ops.O.(Shape.stack (List.mapi widened scalars) + added) in
  let body = Ops.sink [ Ops.store (param 0) sum ] in
  let y = Ops.new_buffer d n Int64 and x = Ops.new_buffer d n Int64 in
  let call =
    Ops.call ~precompile:true body
      ((y :: List.map (fun (_, v, _) -> v) scalars) @ [ x ])
  in
  let linear, _ =
    Schedule.create_linear_with_vars
      ~cores:Cores.host (Ops.sink [ Ops.after y [ call ] ])
  in
  let compiled = compile ~devices linear in
  let on = devices d in
  let int i = `Int (Bigint.of_int i) in
  let ys = Run.buffer on Int64 (Array.make n (int 0))
  and xs = Run.buffer on Int64 (Array.init n (fun i -> int (1000 * i))) in
  let s = Engine.link ~devices ~bound:[ (y, ys); (x, xs) ] compiled in
  Engine.run ~vars s [||];
  let expected = List.mapi (fun i (_, _, v) -> int (v + (1000 * i))) scalars in
  equal values (Array.of_list expected) (Run.values Int64 ys)

(* Values of the 64-bit variable [v], each with what a cast of it to [Int32]
   holds: its low 32 bits in two's complement. *)
let narrowed_values =
  [
    (5, 5);
    (0x7fff_ffff, 0x7fff_ffff);
    (0x8000_0000, -0x8000_0000);
    (0x1_2345_6789, 0x2345_6789);
    (0x1_8000_0005, -0x7fff_fffb);
  ]

(* A kernel that stores its scalar of type [dt], slot 1, widened to 64 bits,
   into its one element of slot 0. *)
let widening_of dt =
  let y = Shape.placeholder ~slot:0 [ 1 ] Int64 in
  let s = Shape.param ~addrspace:Alu 1 dt in
  Codegen.to_program
    (Ops.sink
       ~kernel:(Ops.kernel_info ~name:"widening" ())
       [ Ops.store (Ops.index y [ Ops.int 0 ]) (Ops.cast s Int64) ])
    (Lazy.force clang)

let widening = lazy (widening_of Int32)
let widening_u = lazy (widening_of Uint32)

(* The variable [narrowed] of [Int64], from [0] to [max_int]. *)
let narrowed_var =
  Ops.variable ~dtype:Int64 "narrowed" (Dtype.Value.of_int 0)
    (Dtype.Value.of_int max_int)

let int64 v = Ops.const ~dtype:Int64 (`Int (Bigint.of_int v))

(* The schedule of one call on the host of [prg], [widening] by default,
   whose scalar is [arg]. *)
let widened ?(prg = widening) arg =
  let y = Shape.param ~shape:[ Ops.Int 1 ] ~device:"CPU" 0 Int64 in
  Engine.link_call ~devices (Ops.call (Lazy.force prg) [ y; arg ])

(* What a run of [s], a schedule of [widened], stores, with the variable
   [name], [narrowed] by default, bound to [v]. *)
let stored ?(name = "narrowed") ?(v = 0) s =
  let ys = Run.buffer host Int64 [| `Int Bigint.zero |] in
  Engine.run ~vars:[ (name, v) ] s [| Some ys |];
  Run.values Int64 ys

let int_value i = [| `Int (Bigint.of_int i) |]

(* A call whose scalar is the cast of [v] to [Int32], which holds [v]'s low
   32 bits whatever [v] holds. *)
let reads_a_narrowed_scalar (v, low) =
  equal values (int_value low)
    (stored ~v (widened (Ops.cast narrowed_var Int32)))

(* Values of [narrowed], each with what a cast of it to [Uint32] holds: its
   low 32 bits. *)
let narrowed_unsigned =
  [
    (5, 5);
    (0xffff_ffff, 0xffff_ffff);
    (0x1_0000_0000, 0);
    (0x1_8000_0005, 0x8000_0005);
  ]

let reads_an_unsigned_narrowing (v, low) =
  equal values (int_value low)
    (stored ~v (widened ~prg:widening_u (Ops.cast narrowed_var Uint32)))

(* Values of [narrowed], each with the quotient by [2], rounded down, of what
   a cast of it to [Int32] holds: the cast truncates under the division. *)
let halved_values = [ (0x1_0000_0004, 2); (0x1_8000_0006, -0x3fff_fffd) ]

let reads_a_narrowing_under_a_quotient (v, half) =
  let arg = Ops.O.(Ops.cast narrowed_var Int32 // int 2) in
  equal values (int_value half) (stored ~v (widened arg))

(* A cast of a constant keeps its low bits, alone and under a quotient. *)
let reads_a_narrowed_constant () =
  equal values ~msg:"alone" (int_value 5)
    (stored (widened (Ops.cast (int64 0x1_0000_0005) Int32)));
  equal values ~msg:"under a quotient" (int_value 2)
    (stored (widened Ops.O.(Ops.cast (int64 0x1_0000_0004) Int32 // int 2)))

(* The variable [small] of [Int64], from [-10] to [10]. *)
let small_var =
  Ops.variable ~dtype:Int64 "small" (Dtype.Value.of_int (-10))
    (Dtype.Value.of_int 10)

let widening_u64 = lazy (widening_of Uint64)

(* A cast of [small] to [Uint64], which holds [2^64 - 2] of [-2]: alone, as a
   scalar of 8 bytes, it reaches the kernel as those bits, but under a
   quotient by [2^32] its value is past an [int], and the run refuses it. Of
   [6], the quotient is [0]. *)
let refuses_a_wide_unsigned () =
  let wide = Ops.cast small_var Uint64 and two_32 = 1 lsl 32 in
  let alone = widened ~prg:widening_u64 wide
  and divided = widened ~prg:widening_u64 Ops.O.(wide // int two_32) in
  equal values ~msg:"alone" (int_value (-2))
    (stored ~name:"small" ~v:(-2) alone);
  equal values ~msg:"6 divided" (int_value 0)
    (stored ~name:"small" ~v:6 divided);
  raises_match ~msg:"-2 divided"
    (Exn.invalid_arg ~sub:"past an int")
    (fun () -> stored ~name:"small" ~v:(-2) divided)

(* A host kernel, compiled with its loop whole, which the host splits: it
   stores its scalar [n] of type [dt], slot 1, as an [Int32] into the first
   [n] of four 32-bit elements of slot 0. *)
let prefix_of dt =
  let y = Shape.placeholder ~slot:0 [ 4 ] Int32 in
  let n =
    Shape.param
      ~vmin_vmax:(Dtype.Value.of_int 0, Dtype.Value.of_int 4)
      ~addrspace:Alu 1 dt
  in
  let i = Ops.range ~axis_type:Weak (Sym n) [ 0 ] in
  let kernel =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"prefix" ())
      [ Ops.end_ (Ops.store (Ops.index y [ i ]) (Ops.cast n Int32)) [ i ] ]
  in
  Setting.context
    [ B (Setting.noopt, true) ]
    (fun () -> Codegen.to_program kernel (Lazy.force clang))

let prefix_on_the_host = lazy (prefix_of Int32)
let prefix_u64 = lazy (prefix_of Uint64)

(* Four 32-bit elements of the host, each [-1]. *)
let unwritten () =
  Run.buffer host Int32 (Array.make 4 (`Int Bigint.minus_one))

(* A cast of [small] to [Uint64] read as a number, a loop's trips and a split
   loop's count: of [3] it is [3], and of [-2] it holds [2^64 - 2], past an
   [int], which a run refuses where trips of [-2] would run none. The loop
   stores [r + 1] into row [r] of four for each trip [r] of the call of
   [widening]; the split loop is [prefix_u64]'s. *)
let refuses_a_wide_unsigned_count () =
  let wide = Ops.cast small_var Uint64 in
  let int i = `Int (Bigint.of_int i) in
  let looped =
    let r = Ops.range ~axis_type:Loop (Sym wide) [ 100 ] in
    let ys = Shape.param ~shape:[ Ops.Int 4 ] ~device:"CPU" 0 Int64 in
    let row = Shape.shrink ys [ Some (Sym r, Sym Ops.O.(r + int 1)) ] in
    let call = Ops.call (Lazy.force widening) [ row; Ops.O.(r + int 1) ] in
    Engine.link ~devices
      (compile (Ops.v Op.Linear ~src:[ Ops.end_ call [ r ] ]))
  in
  let trips v () =
    let buffer = Run.buffer host Int64 (Array.make 4 (int (-1))) in
    Engine.run ~vars:[ ("small", v) ] looped [| Some buffer |];
    Run.values Int64 buffer
  in
  equal values ~msg:"trips of 3" [| int 1; int 2; int 3; int (-1) |] (trips 3 ());
  raises_match ~msg:"trips of -2"
    (Exn.invalid_arg ~sub:"past an int")
    (trips (-2));
  let ys = Shape.param ~shape:[ Ops.Int 4 ] ~device:"CPU" 0 Int32 in
  let split =
    Engine.link_call ~devices (Ops.call (Lazy.force prefix_u64) [ ys; wide ])
  in
  let count v () =
    let buffer = unwritten () in
    Engine.run ~vars:[ ("small", v) ] split [| Some buffer |];
    Run.values Int32 buffer
  in
  equal values ~msg:"a count of 3" [| int 3; int 3; int 3; int (-1) |]
    (count 3 ());
  raises_match ~msg:"a count of -2"
    (Exn.invalid_arg ~sub:"past an int")
    (count (-2))

(* A cast of [small] to [Uint64] as offsets, numbers of bytes into a buffer,
   which a run refuses past an [int]. A call of [widening] that stores [7]
   into element [wide] of four 64-bit elements [ys], given as a view 32 bytes
   into eight: of [3] it stores into element [4 + 3] of the eight, and of
   [-2], refused, it stores nowhere, where the bits of [-2] would store 16
   bytes before [ys]. A copy of four bytes from byte [wide] of sixteen of the
   host into CPU:1: of [3] it copies bytes 3 to 6, and of [-2] it is
   refused. *)
let refuses_a_wide_unsigned_offset () =
  let wide = Ops.cast small_var Uint64 in
  let int i = `Int (Bigint.of_int i) in
  let ys = Shape.param ~shape:[ Ops.Int 4 ] ~device:"CPU" 0 Int64 in
  let row = Shape.shrink ys [ Some (Sym wide, Sym Ops.O.(wide + int 1)) ] in
  let seven = Ops.const ~dtype:Int32 (int 7) in
  let stores =
    Engine.link_call ~devices (Ops.call (Lazy.force widening) [ row; seven ])
  in
  let store v ys () = Engine.run ~vars:[ ("small", v) ] stores [| Some ys |] in
  let around v =
    let parent = Run.buffer host Int64 (Array.make 8 (int (-1))) in
    let ys = Rig.Buffer.view parent ~first:32 ~length:32 in
    (try store v ys () with Invalid_argument _ -> ());
    Run.values Int64 parent
  in
  equal values ~msg:"a store of 3"
    (Array.init 8 (fun i -> int (if i = 7 then 7 else -1)))
    (around 3);
  equal values ~msg:"the eight elements around a store of -2"
    (Array.make 8 (int (-1)))
    (around (-2));
  raises_match ~msg:"a store of -2"
    (Exn.invalid_arg ~sub:"read as a number")
    (store (-2) (Run.buffer host Int64 (Array.make 4 (int (-1)))));
  let src = Ops.new_buffer "CPU" 16 Uint8
  and dst = Ops.new_buffer "CPU:1" 4 Uint8 in
  let window = Shape.shrink src [ Some (Sym wide, Sym Ops.O.(wide + int 4)) ] in
  let bytes = Array.init 16 int in
  let copied = Run.buffer (device "CPU:1") Uint8 (Array.make 4 (int 0)) in
  let copies =
    Engine.link ~devices
      ~bound:[ (src, Run.buffer host Uint8 bytes); (dst, copied) ]
      (compile (Ops.v Op.Linear ~src:[ Shape.store_call dst window ]))
  in
  let copy v () = Engine.run ~vars:[ ("small", v) ] copies [||] in
  copy 3 ();
  equal values ~msg:"a copy of 3" (Array.sub bytes 3 4)
    (Run.values Uint8 copied);
  raises_match ~msg:"a copy of -2"
    (Exn.invalid_arg ~sub:"read as a number")
    (copy (-2))

(* A cast to [Uint64] of the constant [-2], whose number is read past an
   [int], which the link refuses: divided by [2^32], and as the count of
   [prefix_u64]'s split loop. *)
let refuses_a_wide_unsigned_constant () =
  let two_32 = 1 lsl 32 in
  let wide = Ops.cast (int64 (-2)) Uint64 in
  raises_match ~msg:"a quotient by 2^32"
    (Exn.invalid_arg ~sub:"Tolk_engine.link: a cast to u64 of -2")
    (fun () -> widened ~prg:widening_u64 Ops.O.(wide // int two_32));
  let ys = Shape.param ~shape:[ Ops.Int 4 ] ~device:"CPU" 0 Int32 in
  raises_match ~msg:"a split loop's count"
    (Exn.invalid_arg ~sub:"Tolk_engine.link: a cast to u64 of -2")
    (fun () ->
      Engine.link_call ~devices (Ops.call (Lazy.force prefix_u64) [ ys; wide ]))

(* A cast to [Uint64] as a scalar of 4 bytes, [widening]'s [Int32], whose
   number is read. Of [small]: of [3] it reaches the kernel as [3], and of
   [-2] it holds [2^64 - 2], past an [int], which the run refuses. Of the
   constant [-2], the link refuses it. *)
let refuses_a_wide_unsigned_4_byte_scalar () =
  let wide = widened (Ops.cast small_var Uint64) in
  equal values ~msg:"of 3" (int_value 3) (stored ~name:"small" ~v:3 wide);
  raises_match ~msg:"of -2"
    (Exn.invalid_arg ~sub:"Tolk_engine.run: a cast to u64 of -2")
    (fun () -> stored ~name:"small" ~v:(-2) wide);
  raises_match ~msg:"of the constant -2"
    (Exn.invalid_arg ~sub:"Tolk_engine.link: a cast to u64 of -2")
    (fun () -> widened (Ops.cast (int64 (-2)) Uint64))

(* A split loop of [n] iterations, [n] the cast to [Int32] of [x] of
   [Int64], from [-2^40] to [2^40], bound to [-2^32 + 3]: the cast holds [3],
   so the blocks run 3 iterations. Without its truncation the loop would
   count [-2^32 + 3] iterations, a block of one. *)
let splits_a_narrowed_count () =
  let prg = Lazy.force prefix_on_the_host in
  (match Ops.arg (Ops.nth prg 0) with
  | Ops.Kernel { split = Some _; _ } -> ()
  | _ -> failf "the kernel does not split");
  let x =
    Ops.variable ~dtype:Int64 "x"
      (Dtype.Value.of_int (-(1 lsl 40)))
      (Dtype.Value.of_int (1 lsl 40))
  in
  let ys = Shape.param ~shape:[ Ops.Int 4 ] ~device:"CPU" 0 Int32 in
  let s = Engine.link_call ~devices (Ops.call prg [ ys; Ops.cast x Int32 ]) in
  let int i = `Int (Bigint.of_int i) in
  let buffer = unwritten () in
  Engine.run ~vars:[ ("x", -(1 lsl 32) + 3) ] s [| Some buffer |];
  equal values [| int 3; int 3; int 3; int (-1) |] (Run.values Int32 buffer)

(* The variable [whole] of [Int64], over every [int]. *)
let whole_var =
  Ops.variable ~dtype:Int64 "whole" (Dtype.Value.of_int min_int)
    (Dtype.Value.of_int max_int)

(* Constants and values of [whole] as a 4-byte scalar, uncast: those 32 bits
   hold, signed or unsigned, reach the kernel as their low 32 bits. *)
let held_by_32_bits = [ (-0x8000_0000, -0x8000_0000); (0xffff_ffff, -1) ]
let past_32_bits = [ -0x8000_0001; 0x1_0000_0000 ]

let reads_a_wide_scalar (v, low) =
  equal values ~msg:"a constant" (int_value low) (stored (widened (int64 v)));
  equal values ~msg:"a variable" (int_value low)
    (stored ~name:"whole" ~v (widened whole_var))

(* A link refuses a constant 32 bits do not hold, and a run a value of
   [whole]. *)
let refuses_a_wide_scalar v =
  raises_match ~msg:"a constant" (Exn.invalid_arg ~sub:"4 bytes") (fun () ->
      widened (int64 v));
  let s = widened whole_var in
  raises_match ~msg:"a variable"
    (Exn.invalid_arg ~sub:"which 32 bits hold neither")
    (fun () -> stored ~name:"whole" ~v s)

(* The variable [bounded] of [Int64], from [0] to [2^31 - 1], which a cast to
   [Int32] holds. *)
let bounded_var =
  Ops.variable ~dtype:Int64 "bounded" (Dtype.Value.of_int 0)
    (Dtype.Value.of_int 0x7fff_ffff)

(* Whether a cast narrows is decided by its operand's bounds, so a run
   refuses a binding outside them: [bounded]'s cast to [Int32], halved, is a
   quotient of [bounded] within them, and of [2^32 + 4] the kernel would
   read [2] where the link computes [0x8000_0002]. *)
let refuses_an_unbounded_binding () =
  let s = widened Ops.O.(Ops.cast bounded_var Int32 // int 2) in
  let run v () = stored ~name:"bounded" ~v s in
  equal values ~msg:"within" (int_value 0x3fff_ffff) (run 0x7fff_ffff ());
  raises_match ~msg:"above"
    (Exn.invalid_arg ~sub:"outside its bounds")
    (run ((1 lsl 32) + 4));
  raises_match ~msg:"below" (Exn.invalid_arg ~sub:"outside its bounds") (run (-1))

(* A loop of two trips [r] around a call of [widening] whose scalar is the
   cast to [Int32] of [r * 2^32]: a narrowing of a value that moves with the
   trips, which link refuses. *)
let refuses_a_moving_narrowing () =
  let r = Ops.range ~axis_type:Loop (Int 2) [ 100 ] in
  let y = Shape.param ~shape:[ Ops.Int 1 ] ~device:"CPU" 0 Int64 in
  let two_32 = int64 (1 lsl 32) in
  let wide = Ops.O.(Ops.cast r Int64 * two_32) in
  let call = Ops.call (Lazy.force widening) [ y; Ops.cast wide Int32 ] in
  let linear = compile (Ops.v Op.Linear ~src:[ Ops.end_ call [ r ] ]) in
  raises_match
    (Exn.invalid_arg ~sub:"narrows a value that moves")
    (fun () -> Engine.link ~devices linear)

(* Loops of three trips [o] around [o + 1] trips [i], around a call of
   [widening] on the host that stores [4 o + i] into row [4 o + i] of twelve:
   the inner loop's trips move with the outer trip. *)
let moves_inner_trips () =
  let o = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let i = Ops.range ~axis_type:Loop (Sym Ops.O.(o + int 1)) [ 101 ] in
  let at = Ops.O.((o * int 4) + i) in
  let ys = Shape.param ~shape:[ Ops.Int 12 ] ~device:"CPU" 0 Int64 in
  let row = Shape.shrink ys [ Some (Sym at, Sym Ops.O.(at + int 1)) ] in
  let call = Ops.call (Lazy.force widening) [ row; at ] in
  let loops = Ops.end_ (Ops.end_ call [ i ]) [ o ] in
  let s = Engine.link ~devices (compile (Ops.v Op.Linear ~src:[ loops ])) in
  let buffer = Run.buffer host Int64 (Array.make 12 (`Int Bigint.minus_one)) in
  Engine.run s [| Some buffer |];
  let written k = k mod 4 <= k / 4 in
  equal values
    (Array.init 12 (fun k ->
         `Int (Bigint.of_int (if written k then k else -1))))
    (Run.values Int64 buffer)

(* A loop of three trips [r] on the GPU [d] around a call of a kernel that
   stores its scalar [n] into the first [n] of four 32-bit elements, a grid
   [n] wide, for [n = r + 1]: the grid moves with the trips. *)
let moves_a_grid ?(devices = devices) d () =
  let on = devices d in
  let y = Shape.placeholder ~slot:0 [ 4 ] Int32 in
  let n =
    Shape.param
      ~vmin_vmax:(Dtype.Value.of_int 1, Dtype.Value.of_int 4)
      ~addrspace:Alu 1 Int32
  in
  let i = Ops.range ~axis_type:Weak (Sym n) [ 0 ] in
  let prefix =
    Codegen.to_program
      (Ops.sink
         ~kernel:(Ops.kernel_info ~name:"prefix" ())
         [ Ops.end_ (Ops.store (Ops.index y [ i ]) n) [ i ] ])
      (Run.renderer on)
  in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let ys = Shape.param ~shape:[ Ops.Int 4 ] ~device:d 0 Int32 in
  let call = Ops.call prefix [ ys; Ops.O.(r + int 1) ] in
  let linear = Ops.v Op.Linear ~src:[ Ops.end_ call [ r ] ] in
  let s = Engine.link ~devices (compile ~devices linear) in
  let int i = `Int (Bigint.of_int i) in
  let buffer = Run.buffer on Int32 (Array.make 4 (int (-1))) in
  Engine.run s [| Some buffer |];
  equal values [| int 3; int 3; int 3; int (-1) |] (Run.values Int32 buffer)

(* A call on [d] of one kernel of [y] and 520 buffers of one element, whose
   addresses take 4,168 bytes of arguments, more than a Metal launch copies
   inline: it stores into [y] the sum of each buffer's element times its
   rank, which a buffer read in another's place changes. *)
let reads_many_buffers ?(devices = devices) d () =
  let n = 520 in
  let int i = `Int (Bigint.of_int i) in
  let param slot = Shape.param ~shape:[ Int 1 ] ~device:d slot Int64 in
  let term i =
    let x = param (i + 1)
    and rank = Shape.stack [ Ops.const ~dtype:Int64 (int (i + 1)) ] in
    Ops.O.(x * rank)
  in
  let sum =
    List.fold_left
      (fun acc i -> Ops.O.(acc + term i))
      (term 0)
      (List.init (n - 1) succ)
  in
  let body = Ops.sink [ Ops.store (param 0) sum ] in
  let y = Ops.new_buffer d 1 Int64 in
  let xs = List.init n (fun _ -> Ops.new_buffer d 1 Int64) in
  let call = Ops.call ~precompile:true body (y :: xs) in
  let linear, _ =
    Schedule.create_linear_with_vars
      ~cores:Cores.host (Ops.sink [ Ops.after y [ call ] ])
  in
  let compiled = compile ~devices linear in
  let on = devices d in
  let element i = (i * i) + 1 in
  let ys = Run.buffer on Int64 [| int 0 |] in
  let bound =
    List.mapi (fun i x -> (x, Run.buffer on Int64 [| int (element i) |])) xs
  in
  let s = Engine.link ~devices ~bound:((y, ys) :: bound) compiled in
  Engine.run s [||];
  let expected = List.init n (fun i -> (i + 1) * element i) in
  equal values
    [| int (List.fold_left ( + ) 0 expected) |]
    (Run.values Int64 ys)

(* A scan of three trips, linked once and run on rows of [xs] and [ys], which
   each run binds to its parameters, and of [zs], linked with the carry [c].
   Each trip stores [c * 2] into its row of [ys] and adds its row of [xs] to
   [c]; a second range then stores each row of [ys] plus one into its row of
   [zs]. Each run starts from the carry the previous one left. *)
let replays_a_scan =
  let k = 4 and n = 3 in
  let cpu = "CPU" in
  let p slot = Shape.param ~shape:[ Int k ] ~device:cpu slot Float32 in
  let step =
    Ops.sink
      [
        Ops.store (p 0) Ops.O.(p 0 + p 1);
        Ops.store (p 2) Ops.O.(p 0 * float 2.);
      ]
  and plus_one = Ops.sink [ Ops.store (p 0) Ops.O.(p 1 + float 1.) ] in
  let c = Ops.new_buffer cpu k Float32
  and xs = Ops.new_buffer cpu (n * k) Float32
  and ys = Ops.new_buffer cpu (n * k) Float32
  and zs = Ops.new_buffer cpu (n * k) Float32 in
  let row r b =
    Shape.shrink b
      [ Some (Sym Ops.O.(r * int k), Sym Ops.O.((r * int k) + int k)) ]
  in
  let r = Ops.range ~axis_type:Loop (Int n) [ 100 ]
  and r' = Ops.range ~axis_type:Loop (Int n) [ 101 ] in
  let scan =
    Ops.end_ (Ops.call ~precompile:true step [ c; row r xs; row r ys ]) [ r ]
  in
  let rows =
    Ops.end_
      (Ops.call ~precompile:true plus_one
         [ row r' zs; row r' (Ops.after ys [ scan ]) ])
      [ r' ]
  in
  let linear, _ =
    Schedule.create_linear_with_vars ~cores:Cores.host
      (Ops.sink
         [ Ops.after c [ scan ]; Ops.after ys [ scan ]; Ops.after zs [ rows ] ])
  in
  let slot u =
    match Ops.arg u with Ops.Param p -> p.slot | _ -> assert false
  in
  let parameters =
    [ (xs, Ops.replace ~op:Param xs); (ys, Ops.replace ~op:Param ys) ]
  in
  let compiled =
    lazy
      (compile ~devices
         (Ops.substitute ~calls:Skip ~pass:Fixed_point linear parameters))
  in
  let runs =
    Gen.list ~size:(Gen.int_range 1 4)
      (Gen.array
         ~size:(Gen.constant (n * k))
         (Gen.map Float.of_int (Gen.int_range (-8) 8)))
  in
  prop "a linked scan carries its storage across runs on each run's parameters"
    runs (fun runs ->
      let c0 = [| 1.; 2.; 3.; 4. |] and zeros = Array.make (n * k) 0. in
      let c_buffer = Run.buffer host Float32 (floats c0)
      and zs_buffer = Run.buffer host Float32 (floats zeros) in
      let s =
        Engine.link ~devices
          ~bound:[ (c, c_buffer); (zs, zs_buffer) ]
          (Lazy.force compiled)
      in
      let carry = Array.copy c0 in
      List.iteri
        (fun i x ->
          let ys_buffer = Run.buffer host Float32 (floats zeros) in
          let slots = Array.make (1 + max (slot xs) (slot ys)) None in
          slots.(slot xs) <- Some (Run.buffer host Float32 (floats x));
          slots.(slot ys) <- Some ys_buffer;
          Engine.run s slots;
          let y = Array.make (n * k) 0. in
          for t = 0 to n - 1 do
            for j = 0 to k - 1 do
              y.((t * k) + j) <- carry.(j) *. 2.;
              carry.(j) <- carry.(j) +. x.((t * k) + j)
            done
          done;
          let msg what = Printf.sprintf "run %d: %s" i what in
          equal values ~msg:(msg "the rows of ys") (floats y)
            (Run.values Float32 ys_buffer);
          equal values ~msg:(msg "the rows of zs")
            (floats (Array.map (fun y -> y +. 1.) y))
            (Run.values Float32 zs_buffer);
          equal values ~msg:(msg "the carry") (floats carry)
            (Run.values Float32 c_buffer))
        runs)

(* A memory-planned schedule: z = a + 1; in a range, y = b + 1, into [y] or into
   a view of its first half; then z + 1 and y + 1 into outputs, each kernel on
   the first four elements of its buffers. The range writes y before the call
   that reads it, so the plan does not place y over z, which the range leaves
   for the call after it. The range is an end, or a back edge whose flag holds
   on every trip. *)
let plans_the_buffers_of_a_range ~through_a_view ~loop () =
  let buf n = Ops.new_buffer "CPU" n Float32 in
  let a = buf 4 and b = buf 4 and z = buf 8 and y = buf 8 in
  let out_z = buf 4 and out_y = buf 4 in
  let flag = Ops.new_buffer "CPU" 1 Bool in
  let r = Ops.range ~axis_type:Loop (Int 2) [ 7 ] in
  let around c =
    match loop with
    | `End -> Ops.end_ c [ r ]
    | `Backedge -> Ops.backedge c ~loop:r ~cond:flag
  in
  let linear =
    Ops.v Op.Linear
      ~src:
        [
          Ops.call add_one [ z; a ];
          around
            (Ops.call add_one
               [
                 (if through_a_view then Shape.shrink y [ Some (Int 0, Int 4) ]
                  else y);
                 b;
               ]);
          Ops.call add_one [ out_z; z ];
          Ops.call add_one [ out_y; y ];
        ]
  in
  let init, linear = flag_from "CPU" flag linear in
  let planned =
    Memory.memory_plan_rewrite ~held_bufs:[ a; b; out_z; out_y; init ] linear
  in
  let compiled = compile ~devices planned in
  let zeros () = Run.buffer host Float32 (floats [| 0.; 0.; 0.; 0. |]) in
  let result_z = zeros () and result_y = zeros () in
  let bound =
    [
      (a, Run.buffer host Float32 (floats [| 1.; 2.; 3.; 4. |]));
      (b, Run.buffer host Float32 (floats [| 10.; 20.; 30.; 40. |]));
      (out_z, result_z);
      (out_y, result_y);
      (init, Run.buffer host Bool [| `Bool true |]);
    ]
  in
  Engine.run (Engine.link ~devices ~bound compiled) [||];
  equal values (floats [| 3.; 4.; 5.; 6. |]) (Run.values Float32 result_z);
  equal values (floats [| 12.; 22.; 32.; 42. |]) (Run.values Float32 result_y)

(* A kernel that stores false into its parameter 0, one boolean. *)
let set_false =
  let i = Ops.range ~axis_type:Weak (Int 1) [ 0 ] in
  let flag = Ops.index (Shape.placeholder ~slot:0 [ 1 ] Bool) [ i ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"set_false" ())
    [ Ops.end_ (Ops.store flag (Ops.O.bool false)) [ i ] ]

(* A planned flag set to false, then z = a + 1, then a loop on the flag: the
   loop reads the flag after the last call that takes it, so the plan does not
   place z over it, and the loop runs no trip. *)
let plans_a_loop_flag () =
  let buf n = Ops.new_buffer "CPU" n Float32 in
  let a = buf 4 and b = buf 4 and z = buf 4 and out = buf 4 in
  let flag = Ops.new_buffer "CPU" 1 Bool in
  let r = Ops.range ~axis_type:Loop (Int 2) [ 7 ] in
  let linear =
    Ops.v Op.Linear
      ~src:
        [
          Ops.call set_false [ flag ];
          Ops.call add_one [ z; a ];
          Ops.backedge (Ops.call add_one [ out; b ]) ~loop:r ~cond:flag;
        ]
  in
  let planned = Memory.memory_plan_rewrite ~held_bufs:[ a; b; out ] linear in
  let compiled = compile ~devices planned in
  let result = Run.buffer host Float32 (floats [| 0.; 0.; 0.; 0. |]) in
  let bound =
    [
      (a, Run.buffer host Float32 (floats [| 0.1; 0.2; 0.3; 0.4 |]));
      (b, Run.buffer host Float32 (floats [| 10.; 20.; 30.; 40. |]));
      (out, result);
    ]
  in
  Engine.run (Engine.link ~devices ~bound compiled) [||];
  equal values (floats [| 0.; 0.; 0.; 0. |]) (Run.values Float32 result)

(* A copy out of a device, a copy into it, and the copy out again: the last copy
   reads what the second wrote. *)
let copies_in_order () =
  let bytes k =
    Array.init 16 (fun i -> `Int (Bigint.of_int (i * k land 0xff)))
  in
  let node d = Ops.new_buffer d 16 Uint8 in
  let device_side = node "CPU:1" and out = node "CPU" and fresh = node "CPU" in
  let linear =
    Ops.v Op.Linear
      ~src:
        [
          Shape.store_call out device_side;
          Shape.store_call device_side fresh;
          Shape.store_call out device_side;
        ]
  in
  let compiled = compile ~devices linear in
  let out_buffer = Run.buffer host Uint8 (bytes 0) in
  let s =
    Engine.link ~devices
      ~bound:
        [
          (device_side, Run.buffer (device "CPU:1") Uint8 (bytes 3));
          (out, out_buffer);
          (fresh, Run.buffer host Uint8 (bytes 7));
        ]
      compiled
  in
  Engine.run s [||];
  equal values (bytes 7) (Run.values Uint8 out_buffer)

(* A loop whose trip [r] copies row [r] of a buffer of [src_device] into row
   [r] of one of [dst_device], whose ends move with the trips: between
   software devices a copy part, and between a GPU and the host a move of the
   program. *)
let copies_moving_rows ?(devices = devices) src_device dst_device () =
  let n = 4 in
  let bytes k =
    Array.init (4 * n) (fun i ->
        `Int (Bigint.of_int (((i * k) + 1) land 0xff)))
  in
  let node d = Ops.new_buffer d (4 * n) Uint8 in
  let src = node src_device and dst = node dst_device in
  let r = Ops.range ~axis_type:Loop (Int n) [ 100 ] in
  let row b =
    Shape.shrink b
      [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 4)) ]
  in
  let linear =
    Ops.v Op.Linear
      ~src:[ Ops.end_ (Shape.store_call (row dst) (row src)) [ r ] ]
  in
  let compiled = compile ~devices linear in
  let dst_buffer = Run.buffer (devices dst_device) Uint8 (bytes 0) in
  let s =
    Engine.link ~devices
      ~bound:
        [
          (src, Run.buffer (devices src_device) Uint8 (bytes 3));
          (dst, dst_buffer);
        ]
      compiled
  in
  Engine.run s [||];
  equal values (bytes 3) (Run.values Uint8 dst_buffer)

(* A loop whose trip [r] moves row [r] of a buffer of CPU:1, a GPU, into row
   [r] of host storage made at link, which a move then copies whole into a
   buffer of CPU:1: a move's end in memory made at link moves with the
   trips. *)
let moves_rows_into_storage ?(devices = devices) () =
  let n = 4 in
  let bytes k =
    Array.init (4 * n) (fun i ->
        `Int (Bigint.of_int (((i * k) + 1) land 0xff)))
  in
  let node d = Ops.new_buffer d (4 * n) Uint8 in
  let src = node "CPU:1" and made = node "CPU" and out = node "CPU:1" in
  let r = Ops.range ~axis_type:Loop (Int n) [ 100 ] in
  let row b =
    Shape.shrink b
      [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 4)) ]
  in
  let linear =
    Ops.v Op.Linear
      ~src:
        [
          Ops.end_ (Shape.store_call (row made) (row src)) [ r ];
          Shape.store_call out made;
        ]
  in
  let compiled = compile ~devices linear in
  let out_buffer = Run.buffer (devices "CPU:1") Uint8 (bytes 0) in
  let s =
    Engine.link ~devices
      ~bound:
        [
          (src, Run.buffer (devices "CPU:1") Uint8 (bytes 3));
          (out, out_buffer);
        ]
      compiled
  in
  Engine.run s [||];
  equal values (bytes 3) (Run.values Uint8 out_buffer)

(* A store of [rows], [4; 8], into [pool], [8; 8], at the rows the loaded
   [slots] give, an index that is Invalid where a slot lies outside [pool]: a
   scatter of rows, one kernel ranging over them. *)
let gathered_store device =
  let buffer n dt shape =
    Shape.reshape
      (Ops.new_buffer device n dt)
      (List.map (fun d -> Ops.Int d) shape)
  in
  let pool = buffer 64 Float32 [ 8; 8 ] in
  let slots = buffer 4 Int32 [ 4 ] and rows = buffer 32 Float32 [ 4; 8 ] in
  let inside =
    Ops.bitwise_and (Ops.ge slots (Ops.int 0)) (Ops.lt slots (Ops.int 8))
  in
  let at = Shape.valid (Ops.cast slots Weak_int) inside in
  let dest = Ops.index pool [ at ] in
  Ops.sink [ Ops.after pool [ Ops.store dest (Ops.add rows (Ops.float 0.5)) ] ]

(* [stores_through_a_gather ~devices device targets] runs [gathered_store] on
   [device] with [slots] at [targets]: [pool] holds each row at its slot within
   it, and is otherwise untouched. *)
let stores_through_a_gather ?(devices = devices) device targets () =
  let big = gathered_store device in
  let calls = Ops.src (fst (Schedule.create_linear_with_vars
                              ~cores:Cores.host big)) in
  equal ~msg:"kernels" int 1 (List.length calls);
  let s, vars, storage = linked ~devices big in
  let of_size n = List.find (fun st -> Shape.max_numel st.node = n) storage in
  let pool = of_size 64 and slot = of_size 4 and rows = of_size 32 in
  Buffer.copy
    ~src:
      (Run.buffer host Int32 (Array.map (fun i -> `Int (Bigint.of_int i)) targets))
    ~dst:slot.buffer;
  Engine.run ~vars s (slots storage);
  let num v = match v with `Float f -> f | _ -> fail "a float" in
  let expected =
    Array.mapi
      (fun i v ->
        match Array.find_index (Int.equal (i / 8)) targets with
        | Some r -> `Float (num rows.before.((r * 8) + (i mod 8)) +. 0.5)
        | None -> v)
      pool.before
  in
  equal values expected (contents pool)

(* A store of a row of [x @ w] into [pool], [8; 8], through a shrink whose
   bounds are the loaded [slot] carrying its validity: the shrink's size is 1
   where [slot] lies within [pool], and Invalid elsewhere. *)
let store_at_valid_slot device =
  let buffer n dt shape =
    Shape.reshape
      (Ops.new_buffer device n dt)
      (List.map (fun d -> Ops.Int d) shape)
  in
  let pool = buffer 64 Float32 [ 8; 8 ] in
  let slot = buffer 1 Int32 [] in
  let x = buffer 4 Float32 [ 1; 4; 1 ] and w = buffer 32 Float32 [ 1; 4; 8 ] in
  let inside =
    Ops.bitwise_and (Ops.ge slot (Ops.int 0)) (Ops.lt slot (Ops.int 8))
  in
  let at = Shape.valid slot inside in
  let view =
    Shape.shrink pool
      [ Some (Ops.Sym at, Ops.Sym (Ops.add at (Ops.int 1))); None ]
  in
  let row = Shape.rop (Ops.mul x w) Op.Add [ 1 ] in
  Ops.sink [ Ops.after view [ Ops.store view row ] ]

let valid_slot_store =
  test
    "a store through a shrink whose bounds carry a validity is refused, its \
     size holding Invalid" (fun () ->
      raises_match (Exn.invalid_arg ~sub:"Invalid, which is no number")
        (fun () -> Schedule.create_linear_with_vars
                     ~cores:Cores.host (store_at_valid_slot "CPU")))

(* Aranges past 2^31 elements: [(n, start, step, dtype, first)] reads the
   elements [first] to [first + 2] of an arange of [n] elements from [start] by
   [step], about 2^31, 2^32 or at its end, where an index or a value computed
   in 32 bits would wrap. *)
let long_aranges =
  [
    ((1 lsl 31) + 5, 0, 1, None, (1 lsl 31) - 2);
    ((1 lsl 31) + 5, 0, 1, None, (1 lsl 31) + 2);
    (1 lsl 31, 0, 1, None, (1 lsl 31) - 3);
    (3 lsl 30, -(1 lsl 31), 1, Some Dtype.Int32, (3 lsl 30) - 3);
    ((1 lsl 32) + 3, (1 lsl 32) + 2, -1, None, (1 lsl 32) - 2);
    (1 lsl 32, (1 lsl 32) - 1, -1, Some Dtype.Uint32, (1 lsl 31) - 1);
    ((1 lsl 40) + 1, -7, 3, None, (1 lsl 40) - 2);
    ((1 lsl 31) + 5, 0, 1, Some Dtype.Float32, (1 lsl 31) + 2);
    ((1 lsl 32) + 3, -9, 1, Some Dtype.Float64, (1 lsl 32) - 2);
  ]

let long_arange_id (n, start, step, dtype, first) =
  let dtype = Option.fold ~none:"default" ~some:Dtype.name dtype in
  Printf.sprintf "%d from %d by %d in %s, at %d" n start step dtype first

(* What the one kernel storing three of an arange's elements into storage of
   three writes on [device]: it reads no other element, so the arange takes
   no memory. A schedule that stores more fails before it is linked. *)
let arange_window ?(devices = devices) device (n, start, step, dtype, first) =
  let u = Shape.arange ~start ~step ?dtype (start + (n * step)) in
  let dtype' = Ops.dtype u in
  let renderer = Run.renderer (devices device) in
  if not (List.mem dtype' (Renderer.supported_dtypes renderer)) then
    skip ~reason:(Dtype.name dtype' ^ " is not supported") ();
  let out = Ops.new_buffer device 3 dtype' in
  let view = Shape.shrink u [ Some (Int first, Int (first + 3)) ] in
  let big = Ops.sink [ Ops.after out [ Ops.store out view ] ] in
  let linear, _ = Schedule.create_linear_with_vars ~cores:Cores.host big in
  equal ~msg:"kernels" int 1 (List.length (Ops.src linear));
  List.iter
    (fun b ->
      if Ops.op b = Op.Buffer then
        at_most ~msg:"stored elements" int ~than:3 (Shape.max_numel b))
    (Ops.toposort ~calls:Skip linear);
  let s, vars, storage = linked ~devices big in
  Engine.run ~vars s (slots storage);
  (* The integer, rounded once to a float type. *)
  let at i : Dtype.value =
    let v = start + (i * step) in
    match dtype' with
    | Float32 ->
        `Float (Int32.float_of_bits (Int32.bits_of_float (Float.of_int v)))
    | Float64 -> `Float (Float.of_int v)
    | _ -> `Int (Bigint.of_int v)
  in
  equal
    (list (array Dtypes.value))
    [ Array.init 3 (fun j -> at (first + j)) ]
    (List.map contents storage)

let arange_windows ?devices device =
  cases "an arange past 2^31 elements holds start + i * step"
    ~id:long_arange_id long_aranges
    (arange_window ?devices device)

let gathered_stores =
  group "a store through a gather"
    [
      cases "writes each row at its loaded index and drops an invalid one"
        ~id:(fun t ->
          String.concat ", " (Array.to_list (Array.map string_of_int t)))
        [ [| 0; 3; 7; 5 |]; [| 6; -1; 8; 2 |] ]
        (fun targets -> stores_through_a_gather "CPU" targets ());
    ]

let schedules =
  group "link and run"
    [
      group "each recorded program writes what its tensors compute"
        (List.map writes_what_it_computes recorded);
      test "custom_kernel runs its custom kernel" runs_a_custom_kernel;
      group "custom kernels with variables" (custom_kernel_scalars ());
      test "assign_bitcast stores through a bitcast" stores_through_a_bitcast;
      test "a schedule runs with each binding of its variables"
        runs_each_binding;
      test "a kernel whose call binds no variable reads it from the run alone"
        (reads_its_variable_from_the_run ~devices);
      test
        "one program launched on several devices computes each device's \
         window at its start"
        one_program_per_start;
      test "kernels whose sizes are one name of two values run each its own"
        sizes_from_calls;
      cases "a kernel reads one name of two ranges as one value" ~id:Dtype.name
        [ Dtype.Weak_int; Dtype.Int64 ]
        one_name_two_ranges;
      cases "a kernel reads one name of two types as one value" ~id:Fun.id
        [ "int32 first"; "int64 first" ]
        one_name_two_types;
      test "a copy runs on a buffer off its parameter's phase" copies_off_phase;
      cases "a pad of integer divisions writes what its tensors compute"
        ~id:Fun.id
        [ "pad_idiv"; "pad_idiv_stored"; "pad_idiv_nested"; "pad_idiv_shrunk" ]
        (fun name -> divides_on_padding name ());
      cases "a bitcast of a float64's outer padding reads its bits"
        ~id:string_of_float [ 0.; 1. ]
        (fun v -> bitcast_of_outer_padding v ());
      test "a pad of a quotient by a constant table writes what it computes"
        (divides_by_a_table "CPU");
      cases "a schedule runs on the buffers each run binds to its parameters"
        ~id:Fun.id [ "contiguous"; "copy_view" ] (fun name ->
          runs_on_its_slots name ());
      cases "stages and sums compute what their programs state"
        ~id:(fun (name, _, _) -> name)
        stages_and_sums
        (fun (name, out, f) -> runs_to name out f ());
      test "a range around a call runs it once per trip" runs_once_per_trip;
      test "a scan runs its body once per trip, carrying in place"
        scans_with_a_carry;
      test "a scan's merged calls carry in place as its calls apart do"
        (carries_merged "CPU");
      test "a scan's calls that read one name at two values carry each its own"
        (carries_bound_apart "CPU");
      test "a scan's stores that read one name at two values store each its own"
        (stores_bound_apart "CPU");
      cases
        "a scan's custom kernel that closes over a bound variable runs as one \
         with the scan's own and reads its value"
        ~id:Fun.id [ "m"; "k" ]
        (fun closure -> custom_kernel_closes_in_loop ~closure "CPU" ());
      test "a store into rows of a buffer of a value read through other \
            storage writes what it computes" stores_rows_read_elsewhere;
      test "a copy stored into rows of a buffer writes them alone"
        copies_into_rows;
      replays_a_scan;
      loops_on_the_host;
      loops_around_calls_on_the_host;
      reads_trips_on "CPU";
      reads_trips_under_a_back_edge "CPU";
      test "a call's scalar arguments of mixed widths reach its kernel"
        (reads_mixed_scalars "CPU");
      cases "a narrowing cast's low bits reach the kernel as its scalar"
        ~id:(fun (v, _) -> Printf.sprintf "%#x" v)
        narrowed_values reads_a_narrowed_scalar;
      cases "a cast to an unsigned type keeps its operand's low bits"
        ~id:(fun (v, _) -> Printf.sprintf "%#x" v)
        narrowed_unsigned reads_an_unsigned_narrowing;
      cases "a narrowing cast under a quotient keeps its low bits"
        ~id:(fun (v, _) -> Printf.sprintf "%#x" v)
        halved_values reads_a_narrowing_under_a_quotient;
      test "a narrowing cast of a constant keeps its low bits"
        reads_a_narrowed_constant;
      test
        "a cast to uint64 past an int under another operation is refused at \
         the run"
        refuses_a_wide_unsigned;
      test
        "a cast to uint64 past an int as a loop's trips or a split loop's \
         count is refused at the run"
        refuses_a_wide_unsigned_count;
      test
        "a cast to uint64 past an int as an offset is refused at the run, \
         which writes nothing"
        refuses_a_wide_unsigned_offset;
      test
        "a constant cast to uint64 past an int whose number is read is \
         refused at link"
        refuses_a_wide_unsigned_constant;
      test "a cast to uint64 past an int as a 4-byte scalar is refused"
        refuses_a_wide_unsigned_4_byte_scalar;
      test "a narrowing cast in a split loop's count keeps its low bits"
        splits_a_narrowed_count;
      cases "a 4-byte scalar that 32 bits hold reaches the kernel"
        ~id:(fun (v, _) -> string_of_int v)
        held_by_32_bits reads_a_wide_scalar;
      cases "a 4-byte scalar that 32 bits do not hold is refused"
        ~id:string_of_int past_32_bits refuses_a_wide_scalar;
      test "a binding outside its variable's bounds is refused at the run"
        refuses_an_unbounded_binding;
      test "a narrowing of a value that moves with a loop's trips is refused"
        refuses_a_moving_narrowing;
      test "a loop's trips move with an outer loop's trip" moves_inner_trips;
      test "a call of 520 buffers reads each from its own address"
        (reads_many_buffers "CPU");
      test "a loop of one trip passes its call the trip as a scalar argument"
        (fun () -> reads_its_trip "CPU" 1 (List.nth (trip_arguments 1) 1));
      test "loops of any range and trips around one call compile one program"
        one_program_for_any_loop;
      test "a loop's flag that views wider storage is refused"
        refuses_a_view_as_flag;
      test "a back edge whose flag is bound storage reads that storage"
        reads_a_bound_flag;
      test "a loop's copy part copies the rows its trips name"
        (copies_moving_rows "CPU:1" "CPU:2");
      test "a planned buffer a range writes is not placed over one it leaves"
        (plans_the_buffers_of_a_range ~through_a_view:false ~loop:`End);
      test "a buffer a range writes through a view is not placed over another"
        (plans_the_buffers_of_a_range ~through_a_view:true ~loop:`End);
      test "a planned buffer a loop writes is not placed over one it leaves"
        (plans_the_buffers_of_a_range ~through_a_view:false ~loop:`Backedge);
      test "a loop's planned flag is not placed under a buffer before it"
        plans_a_loop_flag;
      test "copies run in the order of their schedule" copies_in_order;
    ]

(* Refusals *)

let compiled_add = lazy (fst (schedule (program "add")))

let link_refuses what f =
  test ("link refuses " ^ what) (fun () -> raises_match Exn.invalid_arg f)

(* A refusal names the slot. *)
let run_refuses ?(devices = devices) ?(says = Exn.invalid_arg ~sub:"slot") what
    name slots_of =
  test ("run refuses " ^ what) (fun () ->
      let s, vars, big = parameterized ~devices name in
      let storage = storage_of ~devices big in
      raises_match says (fun () -> Engine.run ~vars s (slots_of storage)))

(* [rebound k f storage] is the slots of [storage] with slot [k]'s buffer [b]
   replaced by [f b]. *)
let rebound k f storage =
  let slots = slots storage in
  slots.(k) <- Option.map f slots.(k);
  slots

(* Links the schedule of add with the buffer of its slot [k] replaced by [f] of
   it. *)
let with_bound k f () =
  let storage = storage_of (program "add") in
  let bound =
    List.map
      (fun s -> (s.node, if s.arg.slot = k then f s.buffer else s.buffer))
      storage
  in
  Engine.link ~devices ~bound (Lazy.force compiled_add)

let floats_on d n = Run.buffer (device d) Float32 (Array.make n (`Float 0.))

(* A buffer a take killed. *)
let taken_floats () =
  let b = floats_on "CPU" 16 in
  ignore (require_some ~msg:"a take" (Rig.Buffer.take ~why:"donated" [ b ]));
  b

(* A linked schedule keeps each bound buffer its runs read: a take of one is
   refused. *)
let keeps_bound () =
  let b = floats_on "CPU" 16 in
  let s = with_bound 2 (fun _ -> b) () in
  is_none ~msg:"a take" (Rig.Buffer.take ~why:"taken" [ b ]);
  ignore (Sys.opaque_identity s)

(* A link that raises keeps none of its bound buffers: a take of their memory
   right after it succeeds. *)
let refusal_keeps_nothing () =
  let b = Buffer.create host (64 + 15) in
  raises_match off_its_phase (with_bound 2 (fun _ -> view_at_phase b 4 64));
  is_some ~msg:"a take" (Rig.Buffer.take ~why:"taken" [ b ])

let refusals =
  group "refusals"
    [
      link_refuses "a device the map does not hold" (fun () ->
          Engine.link
            ~devices:(function
              | "CPU" -> host | n -> invalid_arg ("no device is named " ^ n))
            (fst (schedule (program "copy"))));
      link_refuses "a node that is no schedule" (fun () ->
          Engine.link ~devices (Ops.int 1));
      link_refuses "a schedule whose kernels are not compiled" (fun () ->
          Engine.link ~devices
            (fst (Schedule.create_linear_with_vars
                    ~cores:Cores.host (program "add"))));
      link_refuses "storage bound to a buffer of another device"
        (with_bound 4 (fun _ -> floats_on "CPU:1" 16));
      link_refuses "storage bound to fewer bytes than it holds"
        (with_bound 4 (fun _ -> floats_on "CPU" 15));
      link_refuses "storage a kernel reads bound to a dead buffer"
        (with_bound 2 (fun _ -> taken_floats ()));
      test "link keeps the bound buffers its runs read" keeps_bound;
      test "link refuses storage a kernel reads bound off its phase" (fun () ->
          raises_match off_its_phase
            (with_bound 2 (fun _ -> at_phase host 4 64)));
      test "a refused link keeps none of its bound buffers"
        refusal_keeps_nothing;
      link_refuses "a bound node that is no storage" (fun () ->
          let storage = storage_of (program "add") in
          let bound =
            List.map
              (fun s -> (Ops.replace ~op:Param s.node, s.buffer))
              storage
          in
          Engine.link ~devices ~bound (Lazy.force compiled_add));
      run_refuses "a parameter it binds no buffer" "add" (fun _ -> [||]);
      test "run refuses slots that stop before the last parameter" (fun () ->
          let s, vars, big = parameterized "add" in
          let slots = slots (storage_of big) in
          let last = Array.length slots - 1 in
          raises_match
            (Exn.invalid_arg ~sub:(Printf.sprintf "slot %d" last))
            (fun () -> Engine.run ~vars s (Array.sub slots 0 last)));
      run_refuses "a parameter bound to a buffer of another device" "add"
        (fun storage ->
          let k = (List.hd storage).arg.slot in
          rebound k (fun _ -> floats_on "CPU:1" 16) storage);
      run_refuses "a parameter bound to fewer bytes than it holds" "add"
        (fun storage ->
          let k = (List.hd storage).arg.slot in
          rebound k (fun _ -> floats_on "CPU" 15) storage);
      test "run refuses a variable it binds no value" (fun () ->
          let s, _, big = parameterized "variable_shrink" in
          let storage = storage_of big in
          raises_match Exn.invalid_arg (fun () -> Engine.run s (slots storage)));
    ]

(* Runs

   A linked schedule's runs share its storage, and runs of one schedule are
   serialized: from two domains, runs that each add their own parameter times
   their own variable to [x] land as some order of them, each adding its own
   once. A run allocates and loads nothing, and a linked schedule holds its
   storage while reachable. *)

(* A schedule on [d] that adds its parameter 0, one float, times its variable
   [k] to [x], one float from zero, bound to a buffer of [d]. Is it linked,
   and [x]'s buffer. *)
let accumulating ?(devices = devices) d =
  let x = Ops.new_buffer d 1 Float32 and inc = Ops.new_buffer d 1 Float32 in
  let k =
    Shape.bind
      (Ops.variable ~dtype:Int32 "k" (Dtype.Value.of_int 1)
         (Dtype.Value.of_int 3))
      (`Int Bigint.one)
  in
  let k1 = Shape.reshape (Ops.cast k Float32) [ Int 1 ] in
  let added = Ops.O.(x + (inc * k1)) in
  let big = Ops.sink [ Ops.after x [ Ops.store x added ] ] in
  let linear, _ = Schedule.create_linear_with_vars ~cores:Cores.host big in
  let linear =
    Ops.substitute ~calls:Skip ~pass:Fixed_point linear
      [ (inc, Shape.param ~shape:[ Ops.Int 1 ] ~device:d 0 Float32) ]
  in
  let xb = Run.buffer (devices d) Float32 [| `Float 0. |] in
  (Engine.link ~devices ~bound:[ (x, xb) ] (compile ~devices linear), xb)

(* The model of a schedule is the sum its runs added: a run of [a] and [k]
   adds [a * k]. A run lost to another domain's, or one that took another
   run's parameter or variable, leaves a sum no order of the calls explains.
   A read is of one float, which a concurrent run cannot tear. *)
let serialized ?(devices = devices) d =
  let schedule = abstract "s" in
  let a = Gen.int_range 1 4 and k = Gen.int_range 1 3 in
  stateful ~domains:2 ~count:10 ~steps:8
    "runs of one schedule from two domains land as some order of them"
    [
      command "link"
        (Gen.unit @-> makes schedule)
        (fun () -> ref 0)
        (fun () -> accumulating ~devices d);
      command "run"
        (a @-> k @-> schedule ^-> returns unit)
        (fun a k sum -> sum := !sum + (a * k))
        (fun a k (s, _) ->
          let inc =
            Run.buffer (devices d) Float32 [| `Float (Float.of_int a) |]
          in
          Engine.run ~vars:[ ("k", k) ] s [| Some inc |]);
      command "read"
        (schedule ^-> returns values)
        (fun sum -> [| `Float (Float.of_int !sum) |])
        (fun (_, xb) -> Run.values Float32 xb);
    ]

(* The allocations and loads a profile of [f] records. *)
let allocations_and_loads f =
  let r, events = Rig.Profile.take f in
  let count f = List.length (List.filter f events) in
  ( r,
    count (function Rig.Profile.Allocation _ -> true | _ -> false),
    count (function Rig.Profile.Load _ -> true | _ -> false) )

(* A run's allocations and loads are the profile's Allocation and Load events,
   and a run that allocates nothing leaves every device's allocated bytes as
   they were, or fewer, since the collector may return memory meanwhile. *)
let allocates_nothing ?(devices = devices)
    ?(names = [ "CPU"; "CPU:1"; "CPU:2"; "CPU:3" ]) name () =
  let s, vars, big = parameterized ~devices name in
  let slots = slots (storage_of ~devices big) in
  Engine.run ~vars s slots;
  let allocated () = List.map (fun n -> Rig.allocated (devices n)) names in
  Gc.full_major ();
  let before = allocated () in
  let (), allocations, loads =
    allocations_and_loads (fun () -> Engine.run ~vars s slots)
  in
  equal int ~msg:"allocations" 0 allocations;
  equal int ~msg:"loads" 0 loads;
  List.iter2
    (fun name (b, a) -> at_most int ~msg:name ~than:b a)
    names
    (List.combine before (allocated ()))

(* [s] is unreachable after its last use: native code does not keep it longer.
   Its storage, three buffers of a megabyte, is more than the host's cache holds
   of collected buffers. *)
let holds_its_storage () =
  let n = 1 lsl 18 in
  let buffer () = Ops.new_buffer "CPU" n Float32 in
  let compiled =
    compile
      (Ops.v Op.Linear
         ~src:
           [
             Ops.call
               (Codegen.to_program
                  (kernel ~name:"axpy_held" ~size:n ())
                  (Lazy.force clang))
               [ buffer (); buffer (); buffer () ];
           ])
  in
  let allocated () =
    Gc.full_major ();
    Gc.full_major ();
    Rig.allocated host
  in
  let before = allocated () in
  let s = Engine.link ~devices compiled in
  greater int ~msg:"reachable" ~than:before (allocated ());
  ignore (Sys.opaque_identity s);
  at_most int ~msg:"unreachable" ~than:before (allocated ())

(* A schedule linked [Per_run] makes its storage, three buffers of a megabyte,
   at each run and returns it once the run's work is done: reachable, it holds
   less than one of them at link and after a run. *)
let storage_per_run () =
  let n = 1 lsl 18 in
  let buffer () = Ops.new_buffer "CPU" n Float32 in
  let compiled =
    compile
      (Ops.v Op.Linear
         ~src:
           [
             Ops.call
               (Codegen.to_program
                  (kernel ~name:"axpy_per_run" ~size:n ())
                  (Lazy.force clang))
               [ buffer (); buffer (); buffer () ];
           ])
  in
  (* Four rounds of collection, each returning what it collected. *)
  let allocated () =
    for _ = 1 to 4 do
      Gc.full_major ();
      ignore (Rig.Buffer.create host 0);
      Rig.free_cache host
    done;
    Rig.allocated host
  in
  let before = allocated () in
  let s = Engine.link ~devices ~storage:Per_run compiled in
  let one = 4 * n in
  less int ~msg:"linked" ~than:(before + one) (allocated ());
  Engine.run ~vars:[ ("n", 2) ] s [||];
  less int ~msg:"after a run" ~than:(before + one) (allocated ());
  ignore (Sys.opaque_identity s)

(* Storage made per run, against a model of each buffer's first and last use

   A schedule of calls [x <- y + z] over values of [per_run_size] floats on a
   software device of its own, some of them two calls in a loop of two trips:
   value 0 is parameter 1, the others buffers the schedule plans per run
   ({!Memory.memory_plan_rewrite}), and a last call writes parameter 0 from
   the newest value and value 0. A run holds its storage whole: a budget below
   its bytes runs out of memory, and one at them does not. A buffer lives from
   the entry that writes it to the last that reads it, a loop being one entry,
   so no plan holds less than the most buffers live at one entry. *)

let per_run_size = 64
let per_run_bytes = 4 * per_run_size

let per_run_device =
  lazy (Result.get_ok (Rig.software "per-run-model"))

let per_run_add =
  lazy
    (let open Ops.O in
     let buffer slot = Shape.placeholder ~slot [ per_run_size ] Float32 in
     let i = Ops.range ~axis_type:Weak (Int per_run_size) [ 0 ] in
     let at b = Ops.index (buffer b) [ i ] in
     Codegen.to_program
       (Ops.sink ~kernel:(Ops.kernel_info ~name:"per_run_add" ())
          [ Ops.end_ (Ops.store (at 0) (at 1 + at 2)) [ i ] ])
       (Lazy.force clang))

(* The model: each entry's calls, newest entry first, each call's value and
   the two it reads. *)
type uses = { count : int; reads : (int * int * int) list list }

(* The schedule: each value's node, newest first, and its entries. *)
type calls = { nodes : Ops.t list; calls : Ops.t list }

let per_run_param slot =
  Shape.param ~shape:[ Ops.Int per_run_size ] ~device:"PER_RUN" slot Float32

let pick n k = k mod n

let model_add y z (m : uses) =
  let y = pick m.count y and z = pick m.count z in
  { count = m.count + 1; reads = [ (m.count, y, z) ] :: m.reads }

(* Two values, the second of which may read the first. *)
let model_loop y z y' z' (m : uses) =
  let x = m.count in
  let first = (x, pick x y, pick x z)
  and second = (x + 1, pick (x + 1) y', pick (x + 1) z') in
  { count = x + 2; reads = [ first; second ] :: m.reads }

(* A call of a new value, and the nodes with it. *)
let added y z nodes =
  let n = List.length nodes in
  let node k = List.nth nodes (n - 1 - pick n k) in
  let x = Ops.new_buffer "PER_RUN" per_run_size Float32 in
  (Ops.call (Lazy.force per_run_add) [ x; node y; node z ], x :: nodes)

let system_add y z (c : calls) =
  let call, nodes = added y z c.nodes in
  { nodes; calls = call :: c.calls }

let system_loop y z y' z' (c : calls) =
  let first, nodes = added y z c.nodes in
  let second, nodes = added y' z' nodes in
  let r = Ops.range ~axis_type:Loop (Int 2) [ 100 + List.length c.nodes ] in
  let loop = Ops.end_ (Ops.v Op.Linear ~src:[ first; second ]) [ r ] in
  { nodes; calls = loop :: c.calls }

let input_values = Array.init per_run_size (fun i -> Float.of_int (i + 1))

(* The most bytes live at one call, and parameter 0's values. Each call
   takes a step of its own, a loop's in the order of one trip: a value only
   one loop's calls take lives within a trip, from its first use there to
   its last, and a use in a loop that not all of the value's uses lie in
   holds it for the whole loop. *)
let model_peak (m : uses) =
  let entries = List.rev ([ (-1, m.count - 1, 0) ] :: m.reads) in
  let uses = Array.make m.count [] and spans = Hashtbl.create 8 in
  let step = ref 0 in
  List.iteri
    (fun i entry ->
      let loop = List.length entry > 1 and first = !step in
      List.iter
        (fun (x, y, z) ->
          List.iter
            (fun v ->
              if v > 0 then
                uses.(v) <-
                  (!step, if loop then Some i else None) :: uses.(v))
            [ x; y; z ];
          incr step)
        entry;
      if loop then Hashtbl.replace spans i (first, !step - 1))
    entries;
  let life v =
    let within =
      match uses.(v) with
      | (_, Some l) :: rest -> List.for_all (fun (_, l') -> l' = Some l) rest
      | _ -> false
    in
    List.fold_left
      (fun (lo, hi) (s, l) ->
        let a, b =
          match l with
          | Some l when not within -> Hashtbl.find spans l
          | _ -> (s, s)
        in
        (min lo a, max hi b))
      (max_int, -1) uses.(v)
  in
  let lives =
    Array.init m.count (fun v -> if v = 0 then (max_int, -1) else life v)
  in
  let live s =
    Array.fold_left
      (fun n (lo, hi) -> if lo <= s && s <= hi then n + 1 else n)
      0 lives
  in
  let peak = ref 0 in
  for s = 0 to !step - 1 do
    peak := max !peak (live s)
  done;
  let value = Array.make m.count input_values in
  List.iter
    (fun (x, y, z) -> value.(x) <- Array.map2 ( +. ) value.(y) value.(z))
    (List.concat (List.rev m.reads));
  ( !peak * per_run_bytes,
    floats (Array.map2 ( +. ) value.(m.count - 1) value.(0)) )

(* The least budget beyond what the device holds that a run fits in, found by
   bisection, the bytes of the storage the schedule makes, and parameter 0's
   values after a run. *)
let system_run (c : calls) =
  let d = Lazy.force per_run_device in
  let newest = List.hd c.nodes and input = per_run_param 1 in
  let last = Ops.call (Lazy.force per_run_add) [ per_run_param 0; newest; input ] in
  let linear =
    Memory.memory_plan_rewrite
      (Ops.v Op.Linear ~src:(List.rev (last :: c.calls)))
  in
  let storage =
    List.fold_left
      (fun n u ->
        if Ops.op u = Op.Buffer then
          n + (Shape.max_numel u * Dtype.itemsize (Ops.dtype u))
        else n)
      0
      (Ops.toposort ~calls:Skip linear)
  in
  let devices _ = d in
  let s = Engine.link ~devices ~storage:Per_run (compile ~devices linear) in
  let out = Run.buffer d Float32 (floats (Array.make per_run_size 0.)) in
  let slots = [| Some out; Some (Run.buffer d Float32 (floats input_values)) |] in
  (* The memory of earlier cases returns once collected. A run returns what it
     makes, and the device's work is done when it is submitted, so after a run
     its memory is back in the cache. *)
  let settled () =
    Rig.free_cache d;
    Rig.allocated d
  in
  for _ = 1 to 4 do
    Gc.full_major ();
    Rig.free_cache d
  done;
  let held = settled () in
  let fits extra =
    Rig.set_budget d (held + extra);
    let fit =
      match Engine.run s slots with
      | () -> true
      | exception Rig.Out_of_memory _ -> false
    in
    Rig.set_budget d max_int;
    ignore (settled ());
    fit
  in
  let rec least lo hi =
    if lo >= hi then lo
    else
      let mid = (lo + hi) / 2 in
      if fits mid then least lo mid else least (mid + 1) hi
  in
  let peak = least 0 (List.length c.nodes * per_run_bytes) in
  Engine.run s slots;
  (peak, storage, Run.values Float32 out)

(* A run fits its storage's bytes, which hold what is live at once, and
   computes the model's values. *)
let model_run (m : uses) outcome =
  let live, want = model_peak m in
  match outcome with
  | Error e -> raise e
  | Ok (least, storage, got) ->
      equal ~msg:"the least budget, the storage's bytes" int storage least;
      at_least int ~msg:"the most live at once" ~than:live storage;
      equal ~msg:"parameter 0" values want got

let per_run_model =
  let schedule = abstract "s" in
  let k = Gen.int_range 0 63 in
  stateful ~count:40 ~steps:12
    "a run holds its storage whole, no less than its buffers live at once"
    [
      command "start"
        (Gen.unit @-> makes schedule)
        (fun () -> { count = 1; reads = [] })
        (fun () -> { nodes = [ per_run_param 1 ]; calls = [] });
      command "add" (k @-> k @-> schedule ^-> makes schedule) model_add
        system_add;
      command "add" (k @-> k @-> schedule ^-> makes schedule) model_add
        system_add;
      command "loop"
        (k @-> k @-> k @-> k @-> schedule ^-> makes schedule)
        model_loop system_loop;
      command "run"
        (schedule ^-> judges (triple int int values))
        model_run system_run;
    ]

(* Per run and linked

   A schedule of calls [x <- y + z] over two software devices, each value on the
   device its step draws, an operand of the other device copied over first: a
   chain where each call reads the value before it, a diamond where it reads
   older ones. Planned per run, it gives the same work linked either way: the
   same values, the same submissions on each device, and the same images, one
   per device and binary. *)

let per_run_other =
  lazy (Result.get_ok (Rig.software "per-run-other"))

let two_devices name =
  if name = "PER_RUN" then Lazy.force per_run_device
  else Lazy.force per_run_other

let two_names = [| "PER_RUN"; "PER_RUN:1" |]

(* [steps] as a schedule: each step's device and the values it reads, counted
   back from the newest. *)
let two_device_schedule steps =
  let add args = Ops.call (Lazy.force per_run_add) args in
  let calls = ref [] and nodes = ref [ (per_run_param 1, 0) ] in
  (* [u] of device [from] on device [d]: itself, or a copy of it. *)
  let on d (u, from) =
    if from = d then u
    else
      let c = Ops.new_buffer two_names.(d) per_run_size Float32 in
      calls := Shape.store_call c u :: !calls;
      c
  in
  let back k = List.nth !nodes (k mod List.length !nodes) in
  List.iter
    (fun (d, y, z) ->
      let y = on d (back y) and z = on d (back z) in
      let x = Ops.new_buffer two_names.(d) per_run_size Float32 in
      calls := add [ x; y; z ] :: !calls;
      nodes := (x, d) :: !nodes)
    steps;
  let newest = on 0 (List.hd !nodes) in
  calls := add [ per_run_param 0; newest; per_run_param 1 ] :: !calls;
  Memory.memory_plan_rewrite
    (Ops.v Op.Linear ~src:(List.rev !calls))

(* [linear]'s link and one run under a profile: the images loaded, by device
   and binary, each device's submissions in order, and parameter 0's
   values. *)
let linked_and_run storage linear =
  let devices = two_devices in
  let d = Lazy.force per_run_device in
  let compiled = compile ~devices linear in
  let s, linking =
    Rig.Profile.take (fun () -> Engine.link ~devices ~storage compiled)
  in
  let out = Run.buffer d Float32 (floats (Array.make per_run_size 0.)) in
  let slots = [| Some out; Some (Run.buffer d Float32 (floats input_values)) |] in
  let (), running = Rig.Profile.take (fun () -> Engine.run s slots) in
  let loads =
    List.filter_map
      (function
        | Rig.Profile.Load l ->
            Some (Rig.name (Rig.Image.device l.image), Digest.string l.binary)
        | _ -> None)
      linking
  in
  (* The kernels each device ran. *)
  let work =
    List.filter_map
      (function
        | Rig.Profile.Work w -> Some (Rig.name w.device, (w.start, w.name))
        | _ -> None)
      running
  in
  let working = List.sort_uniq compare (List.map fst work) in
  let names n =
    ( n,
      List.map snd
        (List.sort compare
           (List.filter_map
              (fun (n', w) -> if n = n' then Some w else None)
              work)) )
  in
  (List.sort compare loads, List.map names working, Run.values Float32 out)

let pp_steps ppf steps =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
    (fun ppf (d, y, z) -> Format.fprintf ppf "%d: %d + %d" d y z)
    ppf steps

let two_device_steps =
  let step =
    Gen.triple (Gen.int_range 0 1) (Gen.int_range 0 3) (Gen.int_range 0 3)
  in
  Gen.with_pp pp_steps (Gen.list ~size:(Gen.int_range 1 8) step)

let per_run_as_linked steps =
  let linear = two_device_schedule steps in
  let loads, work, got = linked_and_run Per_run linear in
  let loads', work', got' = linked_and_run Linked linear in
  let image = pair string string in
  equal ~msg:"parameter 0" values got' got;
  not_equal ~msg:"the run's submissions" (list (pair string (list string)))
    [] work;
  equal ~msg:"each device's submissions"
    (list (pair string (list string)))
    work' work;
  equal ~msg:"the images" (list image) loads' loads;
  equal ~msg:"an image per device and binary" (list image)
    (List.sort_uniq compare loads) loads;
  cover "a copy between the devices"
    (List.exists (fun (d, _, _) -> d = 1) steps);
  cover "a diamond" (List.exists (fun (_, y, z) -> y > 0 || z > 0) steps)

(* A run keeps none of the buffers it was given: dropped after it, they return
   to their device while the schedule lives. *)
let slots_let_go () =
  let s, vars, big = parameterized "contiguous" in
  let allocated () =
    for _ = 1 to 4 do
      Gc.full_major ();
      ignore (Rig.Buffer.create host 0);
      Rig.free_cache host
    done;
    Rig.allocated host
  in
  let before = allocated () in
  (fun () -> Engine.run ~vars s (slots (storage_of big))) ();
  at_most int ~msg:"after a run" ~than:before (allocated ());
  ignore (Sys.opaque_identity s)

(* The names of the kernels of [compiled], in order. *)
let kernel_names compiled =
  List.concat_map
    (fun u ->
      if Ops.op u <> Program then []
      else
        match Ops.arg (Ops.nth u 0) with
        | Kernel k -> [ Ops.function_name k ]
        | _ -> [])
    (Ops.toposort ~calls:Enter compiled)

(* The names of the kernels in the work of [events] on the host: a submission's
   work names the functions of its launches, joined by [", "]. *)
let host_kernels events =
  List.concat_map
    (function
      | Rig.Profile.Work w when Rig.equal w.device host ->
          String.split_on_char ',' w.name |> List.map String.trim
      | _ -> [])
    events

(* Under a profile, each kernel of a run is in a span of the host, named after
   the kernel's function. *)
let spans_each_kernel () =
  let big = program "contiguous" in
  let compiled, vars = schedule big in
  let s = Engine.link ~devices ~bound:(bound (storage_of big)) compiled in
  let kernels = kernel_names compiled in
  let (), events = Rig.Profile.take (fun () -> Engine.run ~vars s [||]) in
  not_equal (list string) [] kernels;
  equal (slist string String.compare) kernels (host_kernels events)

(* A kernel on [d] that fills four floats with [x]. *)
let fill d x =
  let i = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
  let out = Shape.placeholder ~slot:0 [ 4 ] Float32 in
  let kernel =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"fill" ())
      [ Ops.end_ (Ops.store (Ops.index out [ i ]) (Ops.O.float x)) [ i ] ]
  in
  let y = Ops.new_buffer d 4 Float32 in
  (y, Ops.call kernel [ y ])

(* At [DEBUG=1], a run of a schedule of [n] copies on the host, each a call of
   its own. *)
let run_of_fills n =
  let fills =
    List.init n (fun _ ->
        let y = Ops.new_buffer "CPU" 4 Float32
        and x = Ops.new_buffer "CPU" 4 Float32 in
        ((y, x), Shape.store_call y x))
  in
  let bound =
    List.concat_map
      (fun ((y, x), _) ->
        [
          (y, Run.buffer host Float32 a); (x, Run.buffer host Float32 a);
        ])
      fills
  in
  let compiled = compile (Ops.v Op.Linear ~src:(List.map snd fills)) in
  let s = Engine.link ~devices ~bound compiled in
  Setting.context [ B (Setting.debug, 1) ] (fun () -> Engine.run s [||]);
  List.filter
    (String.starts_with ~prefix:"jit execs")
    (String.split_on_char '\n' (output ()))

(* The time a [DEBUG=2] line gives its kernel, in seconds, from its words ["tm";
   "12.30us"]. *)
let seconds_of words =
  let rec time = function
    | "tm" :: t :: _ -> t
    | _ :: rest -> time rest
    | [] -> failwith "the line gives no time"
  in
  let t = time words in
  let digits, scale =
    if String.ends_with ~suffix:"us" t then (String.length t - 2, 1e-6)
    else if String.ends_with ~suffix:"ms" t then (String.length t - 2, 1e-3)
    else (String.length t - 1, 1.)
  in
  float_of_string (String.sub t 0 digits) *. scale

(* The words of each line [DEBUG=2] printed so far. *)
let reported () =
  List.filter_map
    (fun line ->
      if String.starts_with ~prefix:"*** " line then
        Some (List.filter (( <> ) "") (String.split_on_char ' ' line))
      else None)
    (String.split_on_char '\n' (output ()))

(* Each line counts the kernels run before it: one more than the line before. *)
let counts_up lines =
  let counts = List.map (fun words -> int_of_string (List.nth words 2)) lines in
  match counts with
  | [] -> ()
  | first :: _ ->
      equal (list int) ~msg:"counts"
        (List.init (List.length counts) (fun k -> first + k))
        counts

(* At [DEBUG=2], a run of three kernels on [name] prints a line for each, with a
   time where the device records the span of each. *)
let reports_each_kernel ?(devices = devices) name () =
  let fills = List.init 3 (fun k -> fill name (Float.of_int k)) in
  let bound =
    List.map (fun (y, _) -> (y, Run.buffer (devices name) Float32 a)) fills
  in
  Setting.context
    [ B (Setting.debug, 2) ]
    (fun () ->
      let s =
        Engine.link ~devices ~bound
          (compile ~devices (Ops.v Op.Linear ~src:(List.map snd fills)))
      in
      Engine.run s [||]);
  let lines = reported () in
  equal int ~msg:"one line per kernel" 3 (List.length lines);
  counts_up lines;
  List.iter
    (fun words ->
      if List.mem "tm" words then begin
        let t = seconds_of words in
        at_least float_exact ~msg:"time" ~than:0. t
      end)
    lines

(* At [DEBUG=2], two kernels of one name, a fill of one buffer and a copy
   between two, report each its own name and arguments: the copy's line is
   named apart from the fill's and counts its two buffers. *)
let reports_kernels_of_one_name () =
  let y, fill = fill "CPU" 1. in
  let copy =
    let i = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
    let at slot = Ops.index (Shape.placeholder ~slot [ 4 ] Float32) [ i ] in
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"fill" ())
      [ Ops.end_ (Ops.store (at 0) (at 1)) [ i ] ]
  in
  let z = Ops.new_buffer "CPU" 4 Float32 in
  let bound =
    [ (y, Run.buffer host Float32 a); (z, Run.buffer host Float32 a) ]
  in
  Setting.context
    [ B (Setting.debug, 2) ]
    (fun () ->
      let s =
        Engine.link ~devices ~bound
          (compile (Ops.v Op.Linear ~src:[ fill; Ops.call copy [ z; y ] ]))
      in
      Engine.run s [||]);
  let name_and_args words =
    let rec args = function
      | "arg" :: n :: _ -> n
      | _ :: rest -> args rest
      | [] -> failwith "the line gives no arguments"
    in
    (List.nth words 3, int_of_string (args words))
  in
  equal
    (list (pair string int))
    [ ("fill", 1); ("filln1", 2) ]
    (List.map name_and_args (reported ()))

let runs =
  group "runs"
    [
      serialized "CPU";
      test "a run of kernels allocates and loads nothing"
        (allocates_nothing "contiguous");
      test "a run of copies allocates and loads nothing"
        (allocates_nothing "copy_computed");
      test "a linked schedule holds its storage while it is reachable"
        holds_its_storage;
      test "a schedule linked per run holds no storage between runs"
        storage_per_run;
      per_run_model;
      prop ~count:40
        "a schedule linked per run does the work it does linked"
        two_device_steps per_run_as_linked;
      test "a run keeps none of the buffers it was given" slots_let_go;
      test "each kernel of a run is in a span of the host under a profile"
        spans_each_kernel;
      test "at DEBUG=1, a run of ten calls says how many it runs" (fun () ->
          equal (list string) [ "jit execs 10 calls" ] (run_of_fills 10));
      test "at DEBUG=1, a run of nine calls says nothing" (fun () ->
          equal (list string) [] (run_of_fills 9));
      test "at DEBUG=2, a run prints a timed line per kernel of the host"
        (reports_each_kernel "CPU");
      test
        "at DEBUG=2, a run prints a timed line per kernel of a software device"
        (reports_each_kernel "CPU:1");
      test "at DEBUG=2, kernels of one name report each its own arguments"
        reports_kernels_of_one_name;
    ]

(* Timing *)

(* The devices a time runs on: the host, a software device and each GPU. *)
let timed_on =
  [
    ("the host", "CPU");
    ("a software device", "CPU:1");
    ("Metal", "METAL");
    ("CUDA", "CUDA");
    ("NV", "NV");
    ("AMD", "AMD");
  ]

(* The GPU kind of a device name of [timed_on]. *)
let gpu_kind =
  [ ("METAL", "Metal"); ("CUDA", "CUDA"); ("NV", "NV"); ("AMD", "AMD") ]

(* The devices map and renderer of [name], skipping the test without its
   GPU. *)
let timing_on name =
  match List.assoc_opt name gpu_kind with
  | Some kind ->
      let m = Gpus.gpu kind in
      ((fun _ -> m), Run.renderer m)
  | None -> (devices, Lazy.force clang)

(* The program of [long_axpy ()] linked on [name], and its slots. *)
let timed name =
  let devices, ren = timing_on name in
  let s =
    Engine.link_call ~devices
      (call_on name ~size:(1 lsl 18) ~slots:3
         (Codegen.to_program (long_axpy ()) ren))
  in
  (s, Engine.slots s)

(* The variable of [long_axpy ()], bound. *)
let n_bound = [ ("n", 3) ]

let span_count name events =
  List.length
    (List.filter
       (function
         | Rig.Profile.Work w -> String.starts_with ~prefix:name w.name
         | _ -> false)
       events)

(* [f ()], whether a profile was still taken after it, and the profile's
   events. *)
let profiled f =
  let (r, taken), events =
    Rig.Profile.take (fun () ->
        let r = f () in
        (r, Rig.Profile.enabled ()))
  in
  (r, taken, events)

(* A profile taken around a time records the run's kernels, and the time is
   the sum of their spans. *)
let time_under_profile name =
  let s, slots = timed name in
  let t, _, events = profiled (fun () -> Engine.time ~vars:n_bound s slots) in
  let ns =
    List.fold_left
      (fun ns -> function
        | Rig.Profile.Work w when String.starts_with ~prefix:"long_axpy" w.name
          ->
            ns + (w.stop - w.start)
        | _ -> ns)
      0 events
  in
  equal ~msg:"the time" (float 1e-12) (Float.of_int ns *. 1e-9) t

(* A schedule's consecutive kernels on one device are one submission, whose
   work names them all. The law: a time is the sum of the durations of the
   run's work that runs a kernel, read from a profile of the same run, and the
   submission of several kernels is among that work. The run's work is the
   host's, between the values the host had submitted before and after the
   time. *)
let counts_grouped_kernels () =
  let big = program "contiguous" in
  let compiled, vars = schedule big in
  let s = Engine.link ~devices ~bound:(bound (storage_of big)) compiled in
  let kernels = kernel_names compiled in
  let before = Rig.submitted host in
  let t, _, events = profiled (fun () -> Engine.time ~vars s [||]) in
  let after = Rig.submitted host in
  let works =
    List.filter_map
      (function
        | Rig.Profile.Work w
          when Rig.equal w.device host
               && Rig.Point.value w.point > before
               && Rig.Point.value w.point <= after ->
            let names = List.map String.trim (String.split_on_char ',' w.name) in
            if List.exists (fun n -> List.mem n kernels) names then
              Some (names, w.stop - w.start)
            else None
        | _ -> None)
      events
  in
  satisfies ~claim:"a submission of several kernels among the run's work"
    (list (pair (list string) int))
    (List.exists (fun (names, _) -> List.length names > 1))
    works;
  let ns = List.fold_left (fun ns (_, d) -> ns + d) 0 works in
  equal ~msg:"the time" (float 1e-12) (Float.of_int ns *. 1e-9) t

(* A run may submit a loop's steps once per trip, so a time refuses a
   schedule with a loop, before its run, whatever its trips: none, one or
   more. *)
let refuses_a_loop limit () =
  let s, c, _ = flag_loop "CPU" (limit, 2) in
  raises_match
    (Exn.invalid_arg ~sub:"Tolk_engine.time: a schedule with a loop")
    (fun () -> Engine.time s [||]);
  equal ~msg:"the carry, untouched" values (floats [| 0. |])
    (Run.values Float32 c)

(* A time is of work the call ran: positive, and within the host's times
   around the call. *)
let within_the_call name =
  let s, slots = timed name in
  let t0 = Rig.Profile.now () in
  let t = Engine.time ~vars:n_bound s slots in
  let window = Float.of_int (Rig.Profile.now () - t0) *. 1e-9 in
  greater float_exact ~than:0. t;
  at_most float_exact ~msg:"the call's window" ~than:window t

let leaves_the_profile name =
  let s, slots = timed name in
  let _, taken, _ = profiled (fun () -> Engine.time ~vars:n_bound s slots) in
  satisfies ~claim:"a profile still taken" bool Fun.id taken

let runs_once name =
  let s, slots = timed name in
  let runs = 5 in
  let (), _, events =
    profiled (fun () ->
        for _ = 1 to runs do
          ignore (Engine.time ~vars:n_bound s slots)
        done)
  in
  equal int runs (span_count "long_axpy" events)

(* A time's allocations and loads are the profile's Allocation and Load events,
   and a time that allocates nothing leaves the device's allocated bytes as they
   were, or fewer. *)
let time_allocates_nothing name =
  let s, slots = timed name in
  ignore (Engine.time ~vars:n_bound s slots);
  let devices, _ = timing_on name in
  Gc.full_major ();
  let before = Rig.allocated (devices name) in
  let (), allocations, loads =
    allocations_and_loads (fun () -> ignore (Engine.time ~vars:n_bound s slots))
  in
  equal int ~msg:"allocations" 0 allocations;
  equal int ~msg:"loads" 0 loads;
  at_most int ~than:before (Rig.allocated (devices name))

(* Slots and a linked program, timed and dropped, return what they allocated:
   after a warming round, rounds of them, each collected before the next, leave
   the device's allocated bytes as they were. *)
let returns_its_memory name =
  let devices, _ = timing_on name in
  let allocated () =
    Gc.full_major ();
    Gc.full_major ();
    Rig.allocated (devices name)
  in
  let round () =
    let s, slots = timed name in
    ignore (Sys.opaque_identity (Engine.time ~vars:n_bound s slots))
  in
  round ();
  let before = allocated () in
  for _ = 1 to 4 do
    round ();
    ignore (allocated ())
  done;
  ignore (allocated ());
  ignore (allocated ());
  at_most int ~than:before (allocated ())

let prints_nothing name =
  let s, slots = timed name in
  ignore (output ());
  Setting.context
    [ B (Setting.debug, 2) ]
    (fun () -> ignore (Engine.time ~vars:n_bound s slots));
  expect (output ()) @@ __POS_OF__ {||}

(* [kernel] with its parameter 1 moved to the slot 3. *)
let skipping_a_slot () =
  let k = kernel () in
  let moved u =
    match Ops.arg u with
    | Ops.Param ({ slot = 1; _ } as p) when Ops.op u = Op.Param ->
        [ (u, Ops.replace u ~arg:(Param { p with slot = 3 })) ]
    | _ -> []
  in
  Ops.substitute ~calls:Skip ~pass:Fixed_point k
    (List.concat_map moved (Ops.toposort ~calls:Enter k))

(* The slots of the linked schedule of the program [name], which hold a buffer
   where its storage does, and which a run takes. *)
let slots_fit name =
  let s, vars, storage = linked (program name) in
  let fresh = Engine.slots s in
  let held slots = Array.to_list (Array.map Option.is_some slots) in
  equal (list bool) (held (slots storage)) (held fresh);
  Engine.run ~vars s fresh

(* A time with a scrub reads it before the run: the device's work in a
   profile around the time holds the scrub's read, which ends before the timed
   kernel starts, and a time without it holds none. *)
let reads_scrub name =
  let devices, _ = timing_on name in
  let d = devices name in
  if Option.is_none (Rig.cache_bytes d) then
    skip ~reason:"the driver states no cache bytes" ();
  let s, slots = timed name in
  let scrub = Engine.scrub s in
  let spans ?scrub () =
    let _, _, events =
      profiled (fun () -> Engine.time ~vars:n_bound ?scrub s slots)
    in
    let named prefix =
      List.filter_map
        (function
          | Rig.Profile.Work w
            when String.starts_with ~prefix w.name ->
              Some (w.start, w.stop)
          | _ -> None)
        events
    in
    (named "scrub", named "long_axpy")
  in
  let reads, timed = spans ~scrub () in
  equal int ~msg:"reads of the scrub" 1 (List.length reads);
  equal int ~msg:"timed runs" 1 (List.length timed);
  at_most int ~msg:"the read ends before the run starts"
    ~than:(fst (List.hd timed)) (snd (List.hd reads));
  let reads, _ = spans () in
  equal int ~msg:"reads without a scrub" 0 (List.length reads)

(* Footprint

   What a link of a recorded program makes on each device it names and the
   host, the host and the software device CPU:1, whose images' code, which
   [footprint] leaves out, counts in no budget. A first link, uncounted, makes
   whatever the process makes once. *)

let footprint_of_link (name, named) =
  let compiled, _ = schedule (program name) in
  let counted = Engine.footprint ~devices compiled in
  ignore (Sys.opaque_identity (Engine.link ~devices compiled));
  let on = List.map device named in
  (* Four rounds of collection, each returning what it collected to its
     device: a chain of finalisers frees its memory a round late. *)
  let allocated () =
    for _ = 1 to 4 do
      Gc.full_major ();
      List.iter Rig.free_cache on
    done;
    List.map Rig.allocated on
  in
  let before = allocated () in
  let s = Engine.link ~devices compiled in
  let after = allocated () in
  ignore (Sys.opaque_identity s);
  let by_name l =
    List.sort compare (List.map (fun (d, n) -> (Rig.name d, n)) l)
  in
  let made = by_name (List.combine on (List.map2 ( - ) after before)) in
  satisfies ~claim:"some bytes made" (list (pair string int))
    (List.exists (fun (_, n) -> n > 0))
    made;
  equal (list (pair string int)) made (by_name counted)

let footprint_makes_nothing () =
  let compiled, _ = schedule (program "copy") in
  let _, allocations, loads =
    allocations_and_loads (fun () -> Engine.footprint ~devices compiled)
  in
  equal int ~msg:"allocations" 0 allocations;
  equal int ~msg:"loads" 0 loads

let timing =
  let on f = cases "on" ~id:fst timed_on (fun (_, name) -> f name) in
  group "timing"
    [
      cases "footprint is what link makes on each device and the host"
        ~id:fst
        [
          ("add", [ "CPU" ]);
          ("copy", [ "CPU:1"; "CPU" ]);
          ("assign", [ "CPU" ]);
        ]
        footprint_of_link;
      test "footprint makes no memory and loads no image"
        footprint_makes_nothing;
      cases "run takes the slots of a linked schedule" ~id:Fun.id
        [ "add" ] slots_fit;
      test "link_call refuses a name the map does not hold" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Engine.link_call ~devices
                (call_on "CPU:9" ~size:4 ~slots:3 (Lazy.force axpy))));
      test "a program whose buffers skip a slot links and runs on its slots"
        (fun () ->
          let prg =
            Codegen.to_program (skipping_a_slot ()) (Lazy.force clang)
          in
          let s =
            Engine.link_call ~devices (call_on "CPU" ~size:4 ~slots:4 prg)
          in
          let slots = Engine.slots s in
          equal ~msg:"buffers by slot" (list bool) [ true; true; true; true ]
            (Array.to_list (Array.map Option.is_some slots));
          Engine.run ~vars:n_bound s slots);
      group "a profile taken around a time leaves its value"
        [ on time_under_profile ];
      group "a time is positive, within the call" [ on within_the_call ];
      test "a time counts a submission of several kernels"
        counts_grouped_kernels;
      cases "a time refuses a loop of any trips" ~id:string_of_int [ 0; 1; 2 ]
        (fun limit -> refuses_a_loop limit ());
      group "a time under a profile leaves the profile taken"
        [ on leaves_the_profile ];
      group "each time runs its program once" [ on runs_once ];
      group "a time allocates and loads nothing" [ on time_allocates_nothing ];
      group "slots and a linked program, dropped, return their memory"
        [ on returns_its_memory ];
      group "a time prints nothing at DEBUG=2" [ on prints_nothing ];
      group "a time with a scrub reads it before the run"
        [ on reads_scrub ];
      test "a time refuses an unbound variable" (fun () ->
          let s, slots = timed "CPU" in
          raises_match Exn.invalid_arg (fun () -> Engine.time s slots));
    ]

(* Host programs in blocks: a variable named after a block's bounds *)

(* A kernel that fills four floats with the variable [name], whose bounds are
   those a split of its loop would give its block's bounds. *)
let fill_with name d =
  let i = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
  let out = Shape.placeholder ~slot:0 [ 4 ] Float32 in
  let v =
    Ops.variable ~dtype:Int32 name (Dtype.Value.of_int 0) (Dtype.Value.of_int 4)
  in
  let kernel =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"fill_with" ())
      [ Ops.end_ (Ops.store (Ops.index out [ i ]) (Ops.cast v Float32)) [ i ] ]
  in
  let y = Ops.new_buffer d 4 Float32 in
  (y, Ops.call kernel [ y ])

(* [fill_with name]'s kernel compiled with its loop whole, which the host
   splits. *)
let unoptimized f = Setting.context [ B (Setting.noopt, true) ] f

(* What a run of [fill_with name] on [d] writes with [name] bound to 3. *)
let filled d name =
  let y, call = fill_with name d in
  let compiled =
    unoptimized (fun () -> compile (Ops.v Op.Linear ~src:[ call ]))
  in
  let out = Run.buffer (devices d) Float32 (floats [| 0.; 0.; 0.; 0. |]) in
  let s = Engine.link ~devices ~bound:[ (y, out) ] compiled in
  Engine.run ~vars:[ (name, 3) ] s [||];
  Run.values Float32 out

let named_as_bounds =
  let case d name =
    test (Printf.sprintf "a variable named %s, run on %s" name d) (fun () ->
        equal values (floats [| 3.; 3.; 3.; 3. |]) (filled d name))
  in
  let splits name =
    test (Printf.sprintf "a kernel with a variable named %s splits" name)
      (fun () ->
        let _, call = fill_with name "CPU" in
        let prg =
          unoptimized (fun () ->
              Codegen.to_program (Ops.body call) (Lazy.force clang))
        in
        match Ops.arg (Ops.nth prg 0) with
        | Ops.Kernel { split = Some _; _ } -> ()
        | _ -> failf "%s's kernel does not split" name)
  in
  group "a variable named as a block's bounds is a variable like any other"
    (List.map splits [ "block_lo"; "block_hi" ]
    @ List.concat_map
        (fun d -> List.map (case d) [ "n"; "block_lo"; "block_hi" ])
        [ "CPU"; "CPU:1" ])

(* Host programs in blocks: estimates past an int *)

(* Loops bounded by a loaded value *)

(* How a loop over [j] is bounded at row [i]: by the loaded [x], of int64
   clamped at 0, whose estimated operations count [j] up to 2^63 - 1 times on
   each [i], which fits no int; by the loaded [x] of int32 itself; or by
   [i + 1]. *)
type count = Clamped | Bare | Rows

type loop = {
  count : count;
  own : bool;  (** Whether row [i] loads [x[i]], or [x[0]]. *)
  sum : bool;  (** Whether [out[i]] sums [j^2], or is the last [j^2]. *)
  axis_type : Ops.Axis_type.t;  (** The loop's role. *)
}

let loop = { count = Clamped; own = true; sum = false; axis_type = Loop }
let x_dtype l = if l.count = Bare then Dtype.Int32 else Int64

(* [out[i]] over the trips [j] of [l] at four rows [i], on [d], [opts], if
   given, being the kernel's optimizations. *)
let counts_to_loaded ?(d = "CPU") ?opts l =
  let x = Shape.placeholder ~slot:1 [ 4 ] (x_dtype l)
  and out = Shape.placeholder ~slot:0 [ 4 ] Float32 in
  let i = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
  let count () =
    Ops.load (Ops.index x [ (if l.own then i else Ops.int 0) ]) []
  in
  let range dtype bound = Ops.range ~axis_type:l.axis_type ?dtype bound [ 1 ] in
  let j =
    match l.count with
    | Bare -> range (Some Int32) (Sym (count ()))
    | Rows -> range None (Sym Ops.O.(i + Ops.int 1))
    | Clamped ->
        let clamped = Ops.alu (count ()) Op.Max [ Ops.int ~dtype:Int64 0 ] in
        range None (Sym (Ops.cast clamped Weak_int))
  in
  let jf = Ops.cast j Float32 in
  let at = Ops.index out [ i ] in
  let st =
    if l.sum then Ops.store at (Ops.reduce Ops.O.(jf * jf) Op.Add [ j ])
    else Ops.end_ (Ops.store at Ops.O.(jf * jf)) [ j ]
  in
  let kernel =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"counts_to_loaded" ?opts_to_apply:opts ())
      [ Ops.end_ st [ i ] ]
  in
  let y = Ops.new_buffer d 4 Float32 and xs = Ops.new_buffer d 4 (x_dtype l) in
  (y, xs, Ops.call kernel [ y; xs ])

let loaded_counts = [| 3; 0; 1; 2 |]

let upcast_rows =
  [ Opt.Split { axis = 0; amount = 4; target = Upcast; top = false } ]

(* What a run of [counts_to_loaded] on [d] of [devices] writes. *)
let counted ?(devices = devices) ?(d = "CPU") ?opts l () =
  let y, xs, call = counts_to_loaded ~d ?opts l in
  let compiled = compile ~devices (Ops.v Op.Linear ~src:[ call ]) in
  let out = Run.buffer (devices d) Float32 (floats [| 0.; 0.; 0.; 0. |])
  and counts =
    Run.buffer (devices d) (x_dtype l)
      (Array.map (fun n -> `Int (Bigint.of_int n)) loaded_counts)
  in
  let s = Engine.link ~devices ~bound:[ (y, out); (xs, counts) ] compiled in
  Engine.run s [||];
  Run.values Float32 out

(* What [counted l] is: at each row, the last [j^2] or their sum, over the
   [n] trips [j < n]. *)
let counted_floats l =
  let trips i =
    match l.count with
    | Rows -> i + 1
    | Clamped | Bare -> loaded_counts.(if l.own then i else 0)
  in
  let value i =
    let n = max 0 (trips i) in
    if l.sum then (n - 1) * n * ((2 * n) - 1) / 6
    else if n = 0 then 0
    else (n - 1) * (n - 1)
  in
  floats (Array.init 4 (fun i -> Float.of_int (value i)))

let counted_name l =
  let row = if l.own then "each row's" else "the first row's" in
  let bound =
    match l.count with
    | Clamped -> row ^ " clamped count"
    | Bare -> row ^ " bare count"
    | Rows -> "its row"
  in
  Printf.sprintf "a %s %s bounded by %s"
    (if l.axis_type = Loop then "serial" else "weak")
    (if l.sum then "sum" else "loop")
    bound

(* [out[j] = j^2] for each [j < x[0]], a weak loop of no row, on [d]: the
   only loop of its kernel, its trips are no launch's count of threads or
   blocks. *)
let writes_loaded_trips ?(devices = devices) ?(d = "CPU") ~optimized () =
  let x = Shape.placeholder ~slot:1 [ 4 ] Int32
  and out = Shape.placeholder ~slot:0 [ 4 ] Float32 in
  let count = Ops.load (Ops.index x [ Ops.int 0 ]) [] in
  let j = Ops.range ~axis_type:Weak ~dtype:Int32 (Sym count) [ 0 ] in
  let jf = Ops.cast j Float32 in
  let kernel =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"loaded_trips" ())
      [ Ops.end_ (Ops.store (Ops.index out [ j ]) Ops.O.(jf * jf)) [ j ] ]
  in
  let y = Ops.new_buffer d 4 Float32 and xs = Ops.new_buffer d 4 Int32 in
  let run () =
    compile ~devices (Ops.v Op.Linear ~src:[ Ops.call kernel [ y; xs ] ])
  in
  let compiled = if optimized then run () else unoptimized run in
  let out = Run.buffer (devices d) Float32 (floats [| 0.; 0.; 0.; 0. |])
  and counts =
    Run.buffer (devices d) Int32
      (Array.map (fun n -> `Int (Bigint.of_int n)) loaded_counts)
  in
  let s = Engine.link ~devices ~bound:[ (y, out); (xs, counts) ] compiled in
  Engine.run s [||];
  equal values (floats [| 0.; 1.; 4.; 0. |]) (Run.values Float32 out)

(* At each of four rows [i], a loop writes [i * 10 + k] at [out[i]] for each
   [k < 2], then the sum of [j] over [j < x[i]] is added to it: the sum's loop,
   of a loaded bound, is ordered after the first loop. *)
let follows_a_loop ?(devices = devices) ?(d = "CPU") () =
  let x = Shape.placeholder ~slot:1 [ 4 ] Int32
  and out = Shape.placeholder ~slot:0 [ 4 ] Int32 in
  let i = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
  let k = Ops.range ~axis_type:Loop ~dtype:Int32 (Int 2) [ 1 ] in
  let at = Ops.index out [ i ] in
  let first =
    let row = Ops.O.(Ops.cast i Int32 * Ops.int ~dtype:Int32 10) in
    Ops.end_ (Ops.store at Ops.O.(row + k)) [ k ]
  in
  let count = Ops.load (Ops.index x [ i ]) [] in
  let j = Ops.range ~axis_type:Loop ~dtype:Int32 (Sym count) [ 2 ] in
  let at' = Ops.index (Ops.after out [ first ]) [ i ] in
  let second =
    Ops.store at' Ops.O.(Ops.load at' [] + Ops.reduce j Op.Add [ j ])
  in
  let kernel =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"follows_a_loop" ())
      [ Ops.end_ second [ i ] ]
  in
  let y = Ops.new_buffer d 4 Int32 and xs = Ops.new_buffer d 4 Int32 in
  let compiled =
    compile ~devices (Ops.v Op.Linear ~src:[ Ops.call kernel [ y; xs ] ])
  in
  let ints a = Array.map (fun n -> `Int (Bigint.of_int n)) a in
  let out = Run.buffer (devices d) Int32 (ints [| 0; 0; 0; 0 |])
  and counts = Run.buffer (devices d) Int32 (ints loaded_counts) in
  let s = Engine.link ~devices ~bound:[ (y, out); (xs, counts) ] compiled in
  Engine.run s [||];
  let sum n = n * (n - 1) / 2 in
  equal values
    (ints (Array.mapi (fun i n -> (i * 10) + 1 + sum n) loaded_counts))
    (Run.values Int32 out)

(* [out[i]] the sum of [a[j * 64 + i] * b[j]] over [j < x[i]], for 64 rows
   [i] and at most 4 trips, a product of two operands summed over a loop
   bounded by a load, which the heuristic optimizes on [d]. [a] runs along
   the rows, which a streamed product would upcast. *)
let sums_loaded_products ?(devices = devices) ?(d = "CPU") () =
  let rows = 64 in
  let x = Shape.placeholder ~slot:1 [ rows ] Int32
  and a = Shape.placeholder ~slot:2 [ 4 * rows ] Float32
  and b = Shape.placeholder ~slot:3 [ 4 ] Float32
  and out = Shape.placeholder ~slot:0 [ rows ] Float32 in
  let i = Ops.range ~axis_type:Weak (Int rows) [ 0 ] in
  let count = Ops.load (Ops.index x [ i ]) [] in
  let j = Ops.range ~axis_type:Weak (Sym (Ops.cast count Weak_int)) [ 1 ] in
  let ai = Ops.load (Ops.index a [ Ops.O.((j * Ops.int rows) + i) ]) []
  and bj = Ops.load (Ops.index b [ j ]) [] in
  let sum = Ops.reduce Ops.O.(ai * bj) Op.Add [ j ] in
  let kernel =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"sums_loaded_products" ())
      [ Ops.end_ (Ops.store (Ops.index out [ i ]) sum) [ i ] ]
  in
  let buffer n dt = Ops.new_buffer d n dt in
  let y = buffer rows Float32 and xs = buffer rows Int32 in
  let az = buffer (4 * rows) Float32 and bz = buffer 4 Float32 in
  let compiled =
    compile ~devices
      (Ops.v Op.Linear ~src:[ Ops.call kernel [ y; xs; az; bz ] ])
  in
  let counts = Array.init rows (fun i -> i mod 5)
  and av = Array.init (4 * rows) Float.of_int
  and bv = Array.init 4 (fun j -> Float.of_int (j + 1)) in
  let run_buffer dt vs = Run.buffer (devices d) dt vs in
  let out = run_buffer Float32 (floats (Array.make rows 0.)) in
  let ints = Array.map (fun n -> `Int (Bigint.of_int n)) in
  let bound =
    [
      (y, out);
      (xs, run_buffer Int32 (ints counts));
      (az, run_buffer Float32 (floats av));
      (bz, run_buffer Float32 (floats bv));
    ]
  in
  let s = Engine.link ~devices ~bound compiled in
  Engine.run s [||];
  let expected i =
    let total = ref 0. in
    for j = 0 to counts.(i) - 1 do
      total := !total +. (av.((j * rows) + i) *. bv.(j))
    done;
    !total
  in
  equal values (floats (Array.init rows expected)) (Run.values Float32 out)

(* Loops bounded by a loaded value or by their row, unoptimized, as the
   heuristic optimizes them, and with their rows upcast: a loop has one count
   of trips, so rows whose counts differ cannot be one thread's lanes, and
   their upcast is refused. *)
let loaded_bounds ?(devices = devices) ?(d = "CPU") () =
  let one l how =
    let name = counted_name l ^ ", " ^ how in
    let counted ?opts () = counted ~devices ~d ?opts l () in
    match how with
    | "unoptimized" ->
        test name (fun () ->
            equal values (counted_floats l) (unoptimized counted))
    | "optimized" ->
        test name (fun () -> equal values (counted_floats l) (counted ()))
    | _ when l.own || l.count = Rows ->
        test (name ^ ", refused") (fun () ->
            raises (Invalid_argument "a loop's bound reads axis 0") (fun () ->
                counted ~opts:upcast_rows ()))
    | _ ->
        test name (fun () ->
            equal values (counted_floats l) (counted ~opts:upcast_rows ()))
  in
  (* A loop whose trips store at one row in turn is serial; the trips of a
     sum meet only in it, so its loop may be weak. *)
  let loops =
    List.concat_map
      (fun (count, own, sum) ->
        let l = { count; own; sum; axis_type = Ops.Axis_type.Loop } in
        if sum then [ l; { l with axis_type = Weak } ] else [ l ])
      [
        (Clamped, true, false);
        (Clamped, false, false);
        (Bare, true, false);
        (Bare, false, false);
        (Clamped, true, true);
        (Bare, false, true);
        (Rows, true, false);
        (Rows, true, true);
      ]
  in
  List.concat_map
    (fun l -> List.map (one l) [ "unoptimized"; "optimized"; "upcast" ])
    loops
  @ List.map
      (fun optimized ->
        test
          ("a weak loop of no row bounded by a loaded count, "
          ^ if optimized then "optimized" else "unoptimized")
          (writes_loaded_trips ~devices ~d ~optimized))
      [ false; true ]
  @ [
      test "a loop bounded by a loaded count follows a loop"
        (follows_a_loop ~devices ~d);
      test "a sum of products over a loop bounded by a loaded count"
        (sums_loaded_products ~devices ~d);
    ]

let estimates_past_an_int =
  group "a kernel whose estimates fit no int"
    [
      test "splits and runs" (fun () ->
          equal values (counted_floats loop) (unoptimized (counted loop)));
      test "splits and runs, its loop bounded by a value loaded outside it"
        (fun () ->
          let l = { loop with own = false } in
          equal values (counted_floats l) (unoptimized (counted l)));
      test "prints a timed line at DEBUG=2" (fun () ->
          ignore (output ());
          let written =
            Setting.context
              [ B (Setting.debug, 2) ]
              (fun () -> unoptimized (counted loop))
          in
          equal values (counted_floats loop) written;
          match reported () with
          | [ words ] -> mem string ~msg:"timed" "GFLOPS" words
          | lines -> failf "%d lines, not one" (List.length lines));
    ]

(* The minor words a run of one kernel on [d] of [devices] allocates on the
   calling domain, the run before it having linked and loaded everything. *)
let run_words ?(devices = devices) d () =
  let y, filled = fill d 7. in
  let compiled = compile ~devices (Ops.v Op.Linear ~src:[ filled ]) in
  let s =
    Engine.link ~devices
      ~bound:[ (y, Run.buffer (devices d) Float32 a) ]
      compiled
  in
  Engine.run s [||];
  let before = Gc.minor_words () in
  Engine.run s [||];
  let words = Gc.minor_words () -. before in
  Float.to_int words

let words =
  group "words"
    [
      test "a run of one kernel on the host allocates at most 400 minor words"
        (fun () -> at_most int ~than:400 (run_words "CPU" ()));
    ]

(* GPUs

   The recorded programs copy from the host into CPU:1, which is a GPU here, and
   compute on it. A GPU's tests skip without one. *)

let on_gpu kind name = match name with "CPU:1" -> Gpus.gpu kind | n -> device n
let gpu_names = [ "CPU"; "CPU:1" ]

let computes_on devices name =
  test name (fun () ->
      let big = program name in
      let s, vars, storage = linked ~devices big in
      Engine.run ~vars s (slots storage);
      equal slot_values (expected ~vars big storage) (by_slot storage contents))

let on_a_gpu kind =
  let devices = on_gpu kind in
  group kind
    [
      test "compiles is Ok iff a kernel compiles for the GPU" (fun () ->
          agrees (Gpus.gpu kind));
      group "custom kernels with variables" (custom_kernel_scalars ~devices ());
      test "a pad of a quotient by a constant table writes what it computes"
        (divides_by_a_table ~devices "CPU:1");
      group "loops bounded by a loaded value"
        (loaded_bounds ~devices ~d:"CPU:1" ());
      group "aranges" [ arange_windows ~devices "CPU:1" ];
      test "kernels whose sizes are one name of two values run each its own"
        (sizes_from_calls ~devices);
      cases "a kernel reads one name of two ranges as one value" ~id:Dtype.name
        [ Dtype.Weak_int; Dtype.Int64 ]
        (one_name_two_ranges ~devices);
      cases "a kernel reads one name of two types as one value" ~id:Fun.id
        [ "int32 first"; "int64 first" ]
        (one_name_two_types ~devices);
      test "kernels whose sizes are one name of two values report at DEBUG=2"
        (sizes_from_calls ~devices ~debug:2);
      group "each recorded program writes what its tensors compute"
        (List.map (computes_on devices)
           [
             "add";
             "copy";
             "copy_one";
             "copy_view";
             "copy_computed";
             "variable_shrink";
             "variable_reduce";
           ]);
      cases "a schedule runs on the buffers each run binds to its parameters"
        ~id:Fun.id [ "copy"; "copy_view" ] (fun name ->
          runs_on_its_slots ~devices name ());
      serialized ~devices "CPU:1";
      test "a run allocates and loads nothing"
        (allocates_nothing ~devices ~names:gpu_names "copy");
      test "a run of one kernel allocates at most 400 minor words" (fun () ->
          at_most int ~than:400 (run_words ~devices "CPU:1" ()));
      test
        "a store through a gather writes each row at its loaded index and \
         drops an invalid one"
        (stores_through_a_gather ~devices "CPU:1" [| 6; -1; 8; 2 |]);
      prop "a loop runs its call while its flag holds, at most its trips"
        Gen.(pair (int_range (-1) 6) (int_range 1 5))
        (fun bounds -> loops_while_a_flag_holds ~devices "CPU:1" bounds);
      prop "a loop runs its calls in order each trip while its flag holds"
        loop_bounds (fun bounds -> loops_around_calls ~devices "CPU:1" bounds);
      loops_across_devices_on ~devices "CPU:1";
      reads_trips_on ~devices "CPU:1";
      test "a call's scalar arguments of mixed widths reach its kernel"
        (reads_mixed_scalars ~devices "CPU:1");
      test "a call of 520 buffers reads each from its own address"
        (reads_many_buffers ~devices "CPU:1");
      test "a call's grid moves with its loop's trips"
        (moves_a_grid ~devices "CPU:1");
      reads_trips_under_a_back_edge ~devices "CPU:1";
      test "a scan's merged calls carry in place as its calls apart do"
        (carries_merged ~devices "CPU:1");
      test "a scan's calls that read one name at two values carry each its own"
        (carries_bound_apart ~devices "CPU:1");
      test "a scan's stores that read one name at two values store each its own"
        (stores_bound_apart ~devices "CPU:1");
      cases
        "a scan's custom kernel that closes over a bound variable runs as one \
         with the scan's own and reads its value"
        ~id:Fun.id [ "m"; "k" ]
        (fun closure ->
          custom_kernel_closes_in_loop ~closure ~devices "CPU:1" ());
      test "a loop's move copies the rows its trips name"
        (copies_moving_rows ~devices "CPU:1" "CPU");
      test "a loop's move into storage made at link moves with its trips"
        (moves_rows_into_storage ~devices);
      run_refuses ~devices ~says:off_its_phase
        "a buffer its kernel writes off its phase" "copy"
        (rebound 2 off_phase);
      test "at DEBUG=2, a run prints a line per kernel"
        (reports_each_kernel ~devices "CPU:1");
    ]

let () =
  Disk_cache.play parts;
  exit
    (run "Tolk_engine"
       [
         compiles;
         targets;
         on_the_host;
         in_blocks;
         schedules;
         gathered_stores;
         group "a store through a shrink" [ valid_slot_store ];
         group "aranges" [ arange_windows "CPU" ];
         refusals;
         runs;
         timing;
         named_as_bounds;
         estimates_past_an_int;
         group "loops bounded by a loaded value" (loaded_bounds ());
         words;
         group "GPUs" (List.map on_a_gpu Gpus.kinds);
       ])
