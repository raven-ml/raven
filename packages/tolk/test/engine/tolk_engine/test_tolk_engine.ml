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

(* The recorded programs name tinygrad's devices: its CPU is also CPU:0, PYTHON
   is a memory device, and its disk is named after a directory. *)
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

(* The GPUs the tests run on, each opened once, at its first test, and skipping
   it without one. *)
let gpus =
  let opened opener =
    lazy (match Rig.open_ opener with Ok d -> Ok d | Error why -> Error why)
  in
  [
    ("Metal", opened (Rig_metal.gpu 0));
    ("CUDA", opened (Rig_cuda.gpu 0));
    ("NV", opened (Rig_nv_nvidia.gpu 0));
    ("AMD", opened (Rig_amd_amdgpu.gpu 0));
  ]

let gpu kind =
  match Lazy.force (List.assoc kind gpus) with
  | Ok d -> d
  | Error why -> skip ~reason:why ()

let clang = lazy (Cstyle.clang (Engine.target host))

(* [linear] compiled for the targets of [devices]. *)
let compile ?(devices = devices) linear =
  Realize.compile_linear ~targets:(fun n -> Engine.target (devices n)) linear

(* Devices *)

let target = Testable.make ~pp:Helpers.Target.pp ~equal:( = )

let host_target arch cpu =
  Result.get_ok (Helpers.Target.of_string ("CPU:CLANG:" ^ arch ^ "," ^ cpu))

let targets =
  group "target"
    [
      test "the host compiles C for Clang on its own processor" (fun () ->
          equal target
            (host_target (Rig.arch host) "native")
            (Engine.target host));
      test "a memory device compiles as the host" (fun () ->
          equal target (Engine.target host) (Engine.target (device "CPU:1")));
      test "the disk runs no program" (fun () ->
          raises_match Exn.invalid_arg (fun () -> Engine.target Rig_disk.device));
      test
        "a target's renderer is made once, and shared by the devices that \
         compile for it" (fun () ->
          is_true (Engine.renderer host == Engine.renderer host);
          is_true (Engine.renderer host == Engine.renderer (device "CPU:1"));
          raises_match Exn.invalid_arg (fun () ->
              Engine.renderer Rig_disk.device));
      test "Metal compiles for its GPU family" (fun () ->
          let d = gpu "Metal" in
          let t = Engine.target d in
          equal (pair string string) ("METAL", Rig.arch d) (t.device, t.arch));
    ]

(* Kernels on the host

   A kernel built from UOps: out[i] = a[i] * n + b[i] for [size] elements
   (default four), where out, a and b are the parameters 0, 1 and 2 and n a
   variable. *)

let kernel ?(name = "axpy") ?(size = 4) ?bound () =
  let open Ops.O in
  let buffer slot = Call.placeholder ~slot [ size ] Float32 in
  let n =
    Ops.variable ~dtype:Int32 "n" (Dtype.Value.of_int 0)
      (Dtype.Value.of_int 100)
  in
  let n =
    match bound with
    | Some v -> Call.bind n (`Int (Bigint.of_int v))
    | None -> n
  in
  let i = Ops.range (Int size) [ 0 ] in
  let at b = Ops.index (buffer b) [ i ] in
  Ops.sink ~kernel:(Ops.kernel_info ~name ())
    [ Ops.end_ (Ops.store (at 0) ((at 1 * Ops.cast n Float32) + at 2)) [ i ] ]

let compiled ?name ?size ?bound () =
  Codegen.to_program (kernel ?name ?size ?bound ()) (Lazy.force clang)

let axpy = lazy (compiled ())

(* A kernel long enough for the host clock to see it run. *)
let long_axpy () = kernel ~name:"long_axpy" ~size:(1 lsl 18) ()
let a = floats [| 1.; 2.; 3.; 4. |]
let b = floats [| 10.; 20.; 30.; 40. |]

(* [prg] called on parameters of [size] floats on [name], one per slot. *)
let call_on name ~size ~slots prg =
  Ops.call prg
    (List.init slots (fun slot ->
         Call.param ~shape:[ Ops.Int size ] ~device:(Single name) slot Float32))

let buffers () =
  [|
    [ Run.buffer host Float32 (floats [| 0.; 0.; 0.; 0. |]) ];
    [ Run.buffer host Float32 a ];
    [ Run.buffer host Float32 b ];
  |]

let axpy_on_host prg =
  Engine.link_call ~devices (call_on "CPU" ~size:4 ~slots:3 prg)

(* [n] bytes of a buffer of [d] whose first byte lies [phase] bytes past a
   16-byte boundary, holding [bytes] where given. *)
let at_phase ?bytes d phase n =
  let b = Buffer.create d (n + 15) in
  let rec first j =
    if Buffer.aligned b ~at:(j - phase) 16 then j else first (j + 1)
  in
  let v = Buffer.view b ~first:(first 0) ~length:n in
  Option.iter (fun src -> Buffer.copy ~src ~dst:v) bytes;
  v

(* A copy of [b] 4 bytes past a 16-byte boundary. *)
let off_phase b = at_phase ~bytes:b (Buffer.device b) 4 (Buffer.length b)

(* axpy whose parameter 1 lies 4 bytes past a 16-byte boundary, called on the
   host. *)
let phased_kernel () =
  let open Ops.O in
  let buffer slot =
    if slot = 1 then Call.param ~shape:[ Ops.Int 4 ] ~phase:4 slot Float32
    else Call.placeholder ~slot [ 4 ] Float32
  in
  let n =
    Ops.variable ~dtype:Int32 "n" (Dtype.Value.of_int 0)
      (Dtype.Value.of_int 100)
  in
  let i = Ops.range (Int 4) [ 0 ] in
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
         Call.param ~shape:[ Ops.Int 4 ] ~device:(Single "CPU") ~phase slot
           Float32))

let phased_axpy () = Engine.link_call ~devices (phased_call ())

(* What a refusal of a buffer off its phase says. *)
let off_its_phase = Exn.invalid_arg ~substring:"bytes past a multiple of 16"

(* out[i] = a[i + 3] * n + b[i] for four elements, a eight floats 4 bytes past a
   16-byte boundary: a[3] to a[6] lie on 16 bytes, which the kernel reads as one
   float4. *)
let shifted_axpy =
  lazy
    (let open Ops.O in
     let a = Call.param ~shape:[ Ops.Int 8 ] ~phase:4 1 Float32 in
     let buffer slot = Call.placeholder ~slot [ 4 ] Float32 in
     let n =
       Ops.variable ~dtype:Int32 "n" (Dtype.Value.of_int 0)
         (Dtype.Value.of_int 100)
     in
     let i = Ops.range (Int 4) [ 0 ] in
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
    Call.param ~shape:[ Ops.Int n ] ~device:(Single "CPU") slot Float32
  in
  let made = Ops.new_buffer ~phase:4 (Single "CPU") 8 Float32 in
  let linear =
    Ops.v Op.Linear
      ~src:
        [
          Call.store_call made (param 1 8);
          Ops.call (Lazy.force shifted_axpy) [ param 0 4; made; param 2 4 ];
        ]
  in
  let s = Engine.link ~devices (compile linear) and bufs = buffers () in
  bufs.(1) <-
    [ Run.buffer host Float32 (floats [| 0.; 0.; 0.; 1.; 2.; 3.; 4.; 0. |]) ];
  Engine.run ~vars:[ ("n", 3) ] s bufs;
  equal values
    (floats [| 13.; 26.; 39.; 52. |])
    (Run.values Float32 (List.hd bufs.(0)))

let runs_at_its_phase () =
  let s = phased_axpy () and bufs = buffers () in
  bufs.(1) <- [ off_phase (List.hd bufs.(1)) ];
  Engine.run ~vars:[ ("n", 3) ] s bufs;
  equal values
    (floats [| 13.; 26.; 39.; 52. |])
    (Run.values Float32 (List.hd bufs.(0)))

let runs_a_kernel () =
  let s = axpy_on_host (Lazy.force axpy) and bufs = buffers () in
  Engine.run ~vars:[ ("n", 3) ] s bufs;
  equal values
    (floats [| 13.; 26.; 39.; 52. |])
    (Run.values Float32 (List.hd bufs.(0)))

let takes_the_bound_value () =
  let s = axpy_on_host (compiled ~name:"axpy_bound" ~bound:2 ()) in
  let bufs = buffers () in
  Engine.run s bufs;
  equal values
    (floats [| 12.; 24.; 36.; 48. |])
    (Run.values Float32 (List.hd bufs.(0)))

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
          bufs.(0) <- [ Run.buffer host Float32 (floats [| 0.; 0.; 0. |]) ];
          raises_match Exn.invalid_arg (fun () ->
              Engine.run ~vars:[ ("n", 1) ] s bufs));
      test "run refuses a buffer that starts off its parameter's phase"
        (fun () ->
          let s = axpy_on_host (Lazy.force axpy) in
          let bufs = buffers () in
          bufs.(1) <- [ off_phase (List.hd bufs.(1)) ];
          raises_match
            (Exn.invalid_arg ~substring:"slot 1's buffer of CPU starts 4 bytes")
            (fun () -> Engine.run ~vars:[ ("n", 1) ] s bufs);
          equal values
            (floats [| 0.; 0.; 0.; 0. |])
            (Run.values Float32 (List.hd bufs.(0))));
      test "a parameter of phase 4 runs on a buffer 4 bytes past 16"
        runs_at_its_phase;
      test "storage made at link of phase 4 is made 4 bytes past 16"
        runs_on_made_storage;
      test "run refuses a phase-4 parameter's buffer on a 16-byte boundary"
        (fun () ->
          let s = phased_axpy () in
          let bufs = buffers () in
          bufs.(1) <- [ at_phase ~bytes:(List.hd bufs.(1)) host 0 16 ];
          raises_match
            (Exn.invalid_arg ~substring:"slot 1's buffer of CPU starts 0 bytes")
            (fun () -> Engine.run ~vars:[ ("n", 1) ] s bufs));
      test "link refuses a parameter that two calls access at two phases"
        (fun () ->
          let axpy = call_on "CPU" ~size:4 ~slots:3 (Lazy.force axpy) in
          raises_match (Exn.invalid_arg ~substring:"slot 1 is accessed")
            (fun () ->
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

   A kernel of [m] sums, out[i] = sum_j a[i k + j] b[j] for [k] terms. Each sum
   is summed in one block, in the same order whatever the blocks. *)

let sums ?(name = "sums") m k =
  let buffer slot n = Call.placeholder ~slot [ n ] Float32 in
  let a = buffer 1 (m * k) and b = buffer 2 k in
  let i = Ops.range (Int m) [ 0 ] in
  let j = Ops.range ~axis_type:Reduce (Int k) [ 1 ] in
  let term =
    let open Ops.O in
    Ops.index a [ (i * Ops.int k) + j ] * Ops.index b [ j ]
  in
  Ops.sink ~kernel:(Ops.kernel_info ~name ())
    [
      Ops.end_
        (Ops.store
           (Ops.index (buffer 0 m) [ i ])
           (Ops.reduce term Op.Add [ j ]))
        [ i ];
    ]

(* Values whose sums round differently in another order. *)
let scattered n =
  floats (Array.init n (fun x -> Float.sin (Float.of_int x) *. 1e3))

(* The bits of [sums m k] on scattered values, launched on the host in [blocks]
   blocks: its parameters are the addresses of out, a and b, then its loop's
   iterations, and a call runs the iterations its threads number. *)
let sums_in ~blocks m k =
  let prg = Codegen.to_program (sums m k) (Lazy.force clang) in
  let elf = Device.Tiny_elf.of_program prg in
  let image = Result.get_ok (Rig.Image.load host elf.lib) in
  let fn = require_some (Rig.Image.resolve image elf.name) in
  let s =
    Rig.Submission.(
      make host
        [|
          launch
            ~refs:[| write 0 ~at:0; read 1 ~at:8; read 2 ~at:16 |]
            fn ~params:32;
        |])
  in
  let run = Rig.Submission.Run.make s in
  let block = Rig.Submission.block s 0 in
  Rig.Submission.Run.groups run block blocks 1 1;
  Rig.Submission.Run.threads run block ((m + blocks - 1) / blocks) 1 1;
  Rig.Submission.Run.int64 run block 24 (Int64.of_int m);
  let out = Run.buffer host Float32 (floats (Array.make m 0.)) in
  ignore
    (Rig.submit run
       ~buffers:
         [|
           out;
           Run.buffer host Float32 (scattered (m * k));
           Run.buffer host Float32 (scattered k);
         |]
      : Rig.Point.t);
  Array.map
    (function `Float x -> Int64.bits_of_float x | _ -> assert false)
    (Run.values Float32 out)

let same_bits_in_any_blocks () =
  let one = sums_in ~blocks:1 512 64 in
  List.iter
    (fun blocks ->
      equal
        ~msg:(Printf.sprintf "%d blocks" blocks)
        (array int64) one (sums_in ~blocks 512 64))
    [ 3; 32 ]

let in_blocks =
  group "host programs in blocks"
    [
      test "a program writes the same bits in 1, 3 and 32 blocks"
        same_bits_in_any_blocks;
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
  | Some (Multi ds) -> ds
  | Some (Single d) -> [ d ]
  | None -> invalid_arg "storage without a device"

(* The storage a program is given, its parameters and buffers, each with its
   contents before the run and a buffer per device of its placement. *)
type storage = {
  node : Ops.t;
  arg : Ops.param_arg;
  before : Dtype.value array;
  buffers : Buffer.t list;
}

let per_device (p : Ops.param_arg) = Option.value p.size ~default:1

(* A buffer of [d] holding [values], a file on the disk. *)
let buffer d dt values =
  if not (Rig.equal d Rig_disk.device) then Run.buffer d dt values
  else
    let file =
      Result.get_ok
        (Rig_disk.create_file
           (Filename.concat (temp_dir ()) "file")
           (Array.length values * Dtype.itemsize dt))
    in
    Buffer.copy ~src:(Run.buffer host dt values) ~dst:file;
    file

let storage_of ?(devices = devices) ?(seed = 0) big =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | (Param | Buffer), Param p when p.addrspace <> Some Alu ->
          let names = placement p and m = per_device p in
          let before =
            Array.init
              (List.length names * m)
              (fun j ->
                element p.dtype ((((j * 7) + (p.slot * 3) + seed) mod 11) - 3))
          in
          let buffers =
            List.mapi
              (fun k d ->
                buffer (devices d) p.dtype (Array.sub before (k * m) m))
              names
          in
          Some { node = n; arg = p; before; buffers }
      | _ -> None)
    (Ops.toposort ~calls:Skip big)

let bound storage =
  List.filter_map
    (fun s -> if Ops.op s.node = Buffer then Some (s.node, s.buffers) else None)
    storage

let slots storage =
  let n =
    List.fold_left
      (fun n s -> if Ops.op s.node = Param then max n (s.arg.slot + 1) else n)
      0 storage
  in
  let slots = Array.make n [] in
  List.iter
    (fun s -> if Ops.op s.node = Param then slots.(s.arg.slot) <- s.buffers)
    storage;
  slots

let contents s =
  Array.concat
    (List.map
       (fun b -> Array.sub (Run.values s.arg.dtype b) 0 (per_device s.arg))
       s.buffers)

let refill storage =
  List.iter
    (fun s ->
      let m = per_device s.arg in
      List.iteri
        (fun k dst ->
          Buffer.copy
            ~src:(Run.buffer host s.arg.dtype (Array.sub s.before (k * m) m))
            ~dst)
        s.buffers)
    storage

let by_slot storage f = List.map (fun s -> (s.arg.slot, f s)) storage

(* [with_values vars big] is [big] with each variable replaced by its value in
   [vars], or its bound value. *)
let with_values vars big =
  Ops.substitute ~calls:Skip ~pass:Fixed_point big
    (List.filter_map
       (fun n ->
         match (Ops.op n, Ops.arg n) with
         | Param, Param { name = Some v; addrspace = Some Alu; bound; _ } -> (
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
  let linear, vars = Schedule.create_linear_with_vars big in
  (compile ~devices linear, vars)

let linked ?(devices = devices) big =
  let compiled, vars = schedule ~devices big in
  let storage = storage_of ~devices big in
  (Engine.link ~devices ~bound:(bound storage) compiled, vars, storage)

let slot_values = list (pair int values)

(* One program of each kind of call runs by default: a kernel, a copy between
   devices, lanes of a sharded kernel, a bound scalar and the disk; compiling
   the others takes seconds. Variables run by default in `a schedule runs with
   each binding of its variables`. *)
let by_default =
  [ "add"; "copy"; "shard_add"; "precompiled_scalar"; "disk_store" ]

let writes_what_it_computes name =
  (if List.mem name by_default then test else slow)
    (name ^ " writes what its tensors compute") (fun () ->
      let big = program name in
      let s, vars, storage = linked big in
      Engine.run ~vars s (slots storage);
      equal slot_values (expected ~vars big storage) (by_slot storage contents))

let recorded =
  [
    "add";
    "assign";
    "assign_double_diamond";
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
    "mesh_sum";
    "precompiled_function";
    "precompiled_scalar";
    "read_then_overwrite";
    "reduce_multiple_paths";
    "setitem";
    "shard_add";
    "shard_matmul";
    "shard_sum";
    "shard_to_one";
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

(* The pad_idiv programs pad integer quotients of their inputs: of the inputs
   themselves, of a stored divisor at least 1, padded twice, and shrunk. Their
   inputs hold no zero, and their padding, which reads zeros, divides by none. *)
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
  let linear, vars = Schedule.create_linear_with_vars big in
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
        else { s with buffers = List.map off_phase s.buffers })
      (storage_of big)
  in
  Engine.run ~vars s (slots storage);
  equal slot_values (expected ~vars big storage) (by_slot storage contents)

(* A range of three trips around a call of a kernel that adds one to four
   floats, each trip on the next four of twelve. *)
(* A kernel that stores its parameter 1 plus one into its parameter 0, four
   floats. *)
let add_one =
  let open Ops.O in
  let i = Ops.range (Int 4) [ 0 ] in
  let at slot = Ops.index (Call.placeholder ~slot [ 4 ] Float32) [ i ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"add_one" ())
    [ Ops.end_ (Ops.store (at 0) (at 1 + Ops.O.float 1.)) [ i ] ]

let runs_once_per_trip () =
  let open Ops.O in
  let one = add_one in
  let out = Ops.new_buffer (Single "CPU") 12 Float32
  and src = Ops.new_buffer (Single "CPU") 12 Float32 in
  let r = Ops.range (Int 3) [ 7 ] in
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
          (out, [ out_buffer ]); (src, [ Run.buffer host Float32 (floats xs) ]);
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
  let cpu = Ops.Single "CPU" in
  let p slot = Call.param ~shape:[ Int k ] ~device:cpu slot Float32 in
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
    Schedule.create_linear_with_vars
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
          (c, [ c_buffer ]);
          (xs, [ Run.buffer host Float32 (floats x) ]);
          (ys, [ ys_buffer ]);
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

(* A scan of three trips whose carry [c] has no element: each trip adds one to
   [c], which it cannot reach, and stores its row of [xs] times two into its row
   of [ys]. A call reads and writes no element of an empty argument, which names
   no storage. *)
let scans_with_an_empty_carry () =
  let k = 4 and n = 3 in
  let cpu = Ops.Single "CPU" in
  let p ?(k = k) slot = Call.param ~shape:[ Int k ] ~device:cpu slot Float32 in
  let body =
    Ops.sink
      [
        Ops.store (p ~k:0 0) Ops.O.(p ~k:0 0 + float 1.);
        Ops.store (p 2) Ops.O.(p 1 * float 2.);
      ]
  in
  let c = Ops.new_buffer cpu 0 Float32
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
    Schedule.create_linear_with_vars
      (Ops.sink [ Ops.after c [ e ]; Ops.after ys [ e ] ])
  in
  let compiled = compile ~devices linear in
  let x = Array.init (n * k) Float.of_int in
  let ys_buffer = Run.buffer host Float32 (floats (Array.make (n * k) 0.)) in
  let s =
    Engine.link ~devices
      ~bound:
        [ (xs, [ Run.buffer host Float32 (floats x) ]); (ys, [ ys_buffer ]) ]
      compiled
  in
  Engine.run s [||];
  equal values
    (floats (Array.map (fun x -> x *. 2.) x))
    (Run.values Float32 ys_buffer)

(* [linear] with the flag [f] of its loop first copied from a new buffer of [d],
   which holds the flag's first value, and that buffer. A loop's flag is storage
   the schedule makes, which no run binds. *)
let flag_from d f linear =
  let init = Ops.new_buffer (Ops.Single d) 1 Bool in
  (init, Ops.v Op.Linear ~src:(Call.store_call f init :: Ops.src linear))

(* A loop of at most [max] trips on the device [d] around a call that adds one
   to a carry [c], from zero, stores the carry it read into its row of [ys], 16
   bytes apart, and stores into the flag [f] whether the new carry is below
   [limit]. The flag starts as [0 < limit]. *)
let loops_while_a_flag_holds ?(devices = devices) d (limit, max) =
  let at = Ops.Single d in
  let p slot dt = Call.param ~shape:[ Int 1 ] ~device:at slot dt in
  let next = Ops.O.(p 0 Float32 + float 1.) in
  let body =
    Ops.sink
      [
        Ops.store (p 0 Float32) next;
        Ops.store (p 1 Bool) Ops.O.(next < float (Float.of_int limit));
        Ops.store (p 2 Float32) (p 0 Float32);
      ]
  in
  let c = Ops.new_buffer at 1 Float32
  and f = Ops.new_buffer at 1 Bool
  and ys = Ops.new_buffer at (4 * max) Float32 in
  let r = Ops.range ~axis_type:Loop (Int max) [ 100 ] in
  let row =
    Shape.shrink ys
      [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 1)) ]
  in
  let e =
    Ops.backedge (Ops.call ~precompile:true body [ c; f; row ]) ~loop:r ~cond:f
  in
  let linear, _ =
    Schedule.create_linear_with_vars
      (Ops.sink [ Ops.after c [ e ]; Ops.after f [ e ]; Ops.after ys [ e ] ])
  in
  let init, linear = flag_from d f linear in
  let compiled = compile ~devices linear in
  let on = devices d in
  let c_buffer = Run.buffer on Float32 [| `Float 0. |]
  and f_buffer = Run.buffer on Bool [| `Bool (0 < limit) |]
  and ys_buffer = Run.buffer on Float32 (floats (Array.make (4 * max) (-1.))) in
  let s =
    Engine.link ~devices
      ~bound:[ (c, [ c_buffer ]); (init, [ f_buffer ]); (ys, [ ys_buffer ]) ]
      compiled
  in
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
  let i = Ops.range (Int 1) [ 0 ] in
  let at slot dt = Ops.index (Call.placeholder ~slot [ 1 ] dt) [ i ] in
  Ops.sink ~kernel:(Ops.kernel_info ~name ())
    [ Ops.end_ (Ops.store (at 0 dst) (f (at 1 src))) [ i ] ]

(* A loop of at most [max] trips on the device [d] around four calls each trip:
   the first stores [c + 1] into [t], and the others read it to store [2 t] into
   the carry [c], whether [2 t] is below [limit] into the flag [f], and [t] into
   its row of [ys], 16 bytes apart. Each call after the first reads what the
   first wrote in the same trip, and the first reads the carry the second wrote
   in the trip before. The flag starts as [0 < limit]. *)
let loops_around_calls ?(devices = devices) d (limit, max) =
  let at = Ops.Single d in
  let k name dst f = scalar_kernel name ~dst ~src:Float32 f in
  let inc = k "inc" Float32 (fun x -> Ops.O.(x + float 1.))
  and twice = k "twice" Float32 (fun x -> Ops.O.(x * float 2.))
  and below =
    k "below" Bool (fun x -> Ops.O.(x * float 2. < float (Float.of_int limit)))
  and copy = k "copy" Float32 Fun.id in
  let c = Ops.new_buffer at 1 Float32
  and t = Ops.new_buffer at 1 Float32
  and f = Ops.new_buffer at 1 Bool
  and ys = Ops.new_buffer at (4 * max) Float32 in
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
                   Ops.call below [ f; t ];
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
          (c, [ c_buffer ]);
          (t, [ Run.buffer on Float32 [| `Float (-1.) |] ]);
          (init, [ Run.buffer on Bool [| `Bool (0 < limit) |] ]);
          (ys, [ ys_buffer ]);
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
   time. *)
let loop_bounds =
  Gen.(pair (one_of [ int_range (-1) 0; int_range 1 40 ]) (int_range 1 6))

let loops_around_calls_on_the_host =
  prop ~count:30 "a loop runs its calls in order each trip while its flag holds"
    loop_bounds (fun bounds -> loops_around_calls "CPU" bounds)

(* A loop's flag is storage of one boolean: a view of one element of wider
   storage is refused, since the engine reads the storage it names. *)
let refuses_a_view_as_flag () =
  let at = Ops.Single "CPU" in
  let p = Call.param ~shape:[ Int 1 ] ~device:at 0 Float32 in
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
      Schedule.create_linear_with_vars (Ops.sink [ Ops.after c [ e ] ]))

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
  let at = Ops.Single d in
  let index = Call.param ~addrspace:(Some Alu) 2 Int32 in
  let p slot dt = Call.param ~shape:[ Int 1 ] ~device:at slot dt in
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
    Schedule.create_linear_with_vars
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
        [ (ys, [ ys_buffer ]); (f, [ Run.buffer on Bool [| `Bool true |] ]) ]
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
        else Some (Device.Tiny_elf.of_program u).lib)
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
          (f, [ flag ]);
          (ys, [ ys_buffer ]);
          (init, [ Run.buffer host Bool [| `Bool true |] ]);
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
  cases
    ~name:(fun (name, _, _) -> name)
    "a loop's call reads its trip as a scalar argument on each trip"
    (trip_arguments 5)
    (reads_its_trip ?devices d 5)

let reads_trips_under_a_back_edge ?devices d =
  prop "a back edge's call reads its trip on each trip, at most its trips"
    Gen.(pair (int_range 0 2) (int_range 1 6))
    (fun (k, limit) ->
      reads_its_trip ?devices ~limit d 4 (List.nth (trip_arguments 4) k))

(* A scan of three trips, linked once and run on rows of [xs] and [ys], which
   each run binds to its parameters, and of [zs], linked with the carry [c].
   Each trip stores [c * 2] into its row of [ys] and adds its row of [xs] to
   [c]; a second range then stores each row of [ys] plus one into its row of
   [zs]. Each run starts from the carry the previous one left. *)
let replays_a_scan =
  let k = 4 and n = 3 in
  let cpu = Ops.Single "CPU" in
  let p slot = Call.param ~shape:[ Int k ] ~device:cpu slot Float32 in
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
    Schedule.create_linear_with_vars
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
    compile ~devices
      (Ops.substitute ~calls:Skip ~pass:Fixed_point linear parameters)
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
          ~bound:[ (c, [ c_buffer ]); (zs, [ zs_buffer ]) ]
          compiled
      in
      let carry = Array.copy c0 in
      List.iteri
        (fun i x ->
          let ys_buffer = Run.buffer host Float32 (floats zeros) in
          let slots = Array.make (1 + max (slot xs) (slot ys)) [] in
          slots.(slot xs) <- [ Run.buffer host Float32 (floats x) ];
          slots.(slot ys) <- [ ys_buffer ];
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
  let buf n = Ops.new_buffer (Single "CPU") n Float32 in
  let a = buf 4 and b = buf 4 and z = buf 8 and y = buf 8 in
  let out_z = buf 4 and out_y = buf 4 in
  let flag = Ops.new_buffer (Single "CPU") 1 Bool in
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
      (a, [ Run.buffer host Float32 (floats [| 1.; 2.; 3.; 4. |]) ]);
      (b, [ Run.buffer host Float32 (floats [| 10.; 20.; 30.; 40. |]) ]);
      (out_z, [ result_z ]);
      (out_y, [ result_y ]);
      (init, [ Run.buffer host Bool [| `Bool true |] ]);
    ]
  in
  Engine.run (Engine.link ~devices ~bound compiled) [||];
  equal values (floats [| 3.; 4.; 5.; 6. |]) (Run.values Float32 result_z);
  equal values (floats [| 12.; 22.; 32.; 42. |]) (Run.values Float32 result_y)

(* A kernel that stores false into its parameter 0, one boolean. *)
let set_false =
  let i = Ops.range (Int 1) [ 0 ] in
  let flag = Ops.index (Call.placeholder ~slot:0 [ 1 ] Bool) [ i ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"set_false" ())
    [ Ops.end_ (Ops.store flag (Ops.O.bool false)) [ i ] ]

(* A planned flag set to false, then z = a + 1, then a loop on the flag: the
   loop reads the flag after the last call that takes it, so the plan does not
   place z over it, and the loop runs no trip. *)
let plans_a_loop_flag () =
  let buf n = Ops.new_buffer (Single "CPU") n Float32 in
  let a = buf 4 and b = buf 4 and z = buf 4 and out = buf 4 in
  let flag = Ops.new_buffer (Single "CPU") 1 Bool in
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
      (a, [ Run.buffer host Float32 (floats [| 0.1; 0.2; 0.3; 0.4 |]) ]);
      (b, [ Run.buffer host Float32 (floats [| 10.; 20.; 30.; 40. |]) ]);
      (out, [ result ]);
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
  let node d = Ops.new_buffer (Single d) 16 Uint8 in
  let device_side = node "CPU:1" and out = node "CPU" and fresh = node "CPU" in
  let linear =
    Ops.v Op.Linear
      ~src:
        [
          Call.store_call out device_side;
          Call.store_call device_side fresh;
          Call.store_call out device_side;
        ]
  in
  let compiled = compile ~devices linear in
  let out_buffer = Run.buffer host Uint8 (bytes 0) in
  let s =
    Engine.link ~devices
      ~bound:
        [
          (device_side, [ Run.buffer (device "CPU:1") Uint8 (bytes 3) ]);
          (out, [ out_buffer ]);
          (fresh, [ Run.buffer host Uint8 (bytes 7) ]);
        ]
      compiled
  in
  Engine.run s [||];
  equal values (bytes 7) (Run.values Uint8 out_buffer)

(* A loop whose trip [r] copies row [r] of a buffer of CPU:1 into row [r] of
   one of CPU:2: a copy part whose ends move with the trips. *)
let copies_moving_rows () =
  let n = 4 in
  let bytes k =
    Array.init (4 * n) (fun i ->
        `Int (Bigint.of_int (((i * k) + 1) land 0xff)))
  in
  let node d = Ops.new_buffer (Single d) (4 * n) Uint8 in
  let src = node "CPU:1" and dst = node "CPU:2" in
  let r = Ops.range ~axis_type:Loop (Int n) [ 100 ] in
  let row b =
    Shape.shrink b
      [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 4)) ]
  in
  let linear =
    Ops.v Op.Linear
      ~src:[ Ops.end_ (Call.store_call (row dst) (row src)) [ r ] ]
  in
  let compiled = compile ~devices linear in
  let dst_buffer = Run.buffer (device "CPU:2") Uint8 (bytes 0) in
  let s =
    Engine.link ~devices
      ~bound:
        [
          (src, [ Run.buffer (device "CPU:1") Uint8 (bytes 3) ]);
          (dst, [ dst_buffer ]);
        ]
      compiled
  in
  Engine.run s [||];
  equal values (bytes 3) (Run.values Uint8 dst_buffer)

(* A store of [rows], [4; 8], into [pool], [8; 8], at the rows the loaded
   [slots] give, an index that is Invalid where a slot lies outside [pool]: a
   scatter of rows, one kernel ranging over them. *)
let gathered_store device =
  let buffer n dt shape =
    Shape.reshape
      (Ops.new_buffer (Single device) n dt)
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
  let calls = Ops.src (fst (Schedule.create_linear_with_vars big)) in
  equal ~msg:"kernels" int 1 (List.length calls);
  let s, vars, storage = linked ~devices big in
  let of_size n = List.find (fun st -> per_device st.arg = n) storage in
  let pool = of_size 64 and slot = of_size 4 and rows = of_size 32 in
  List.iter
    (fun dst ->
      Buffer.copy
        ~src:
          (Run.buffer host Int32
             (Array.map (fun i -> `Int (Bigint.of_int i)) targets))
        ~dst)
    slot.buffers;
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
      (Ops.new_buffer (Single device) n dt)
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
      raises_match (Exn.invalid_arg ~substring:"Invalid, which is no number")
        (fun () -> Schedule.create_linear_with_vars (store_at_valid_slot "CPU")))

let gathered_stores =
  group "a store through a gather"
    [
      cases
        ~name:(fun t ->
          String.concat ", " (Array.to_list (Array.map string_of_int t)))
        "writes each row at its loaded index and drops an invalid one"
        [ [| 0; 3; 7; 5 |]; [| 6; -1; 8; 2 |] ]
        (fun targets -> stores_through_a_gather "CPU" targets ());
    ]

let schedules =
  group "link and run"
    [
      group "recorded" (List.map writes_what_it_computes recorded);
      slow "custom_kernel runs its custom kernel" runs_a_custom_kernel;
      slow "assign_bitcast stores through a bitcast" stores_through_a_bitcast;
      test "a schedule runs with each binding of its variables"
        runs_each_binding;
      test "a copy runs on a buffer off its parameter's phase" copies_off_phase;
      cases ~name:Fun.id
        "a pad of integer divisions writes what its tensors compute"
        [ "pad_idiv"; "pad_idiv_stored"; "pad_idiv_nested"; "pad_idiv_shrunk" ]
        (fun name -> divides_on_padding name ());
      cases ~name:Fun.id
        "a schedule runs on the buffers each run binds to its parameters"
        [ "contiguous"; "copy_view"; "shard_add" ] (fun name ->
          runs_on_its_slots name ());
      test "a range around a call runs it once per trip" runs_once_per_trip;
      test "a scan runs its body once per trip, carrying in place"
        scans_with_a_carry;
      test "a scan whose carry has no element runs on its rows"
        scans_with_an_empty_carry;
      replays_a_scan;
      loops_on_the_host;
      loops_around_calls_on_the_host;
      reads_trips_on "CPU";
      reads_trips_under_a_back_edge "CPU";
      test "a loop of one trip passes its call the trip as a scalar argument"
        (fun () -> reads_its_trip "CPU" 1 (List.nth (trip_arguments 1) 1));
      test "loops of any range and trips around one call compile one program"
        one_program_for_any_loop;
      test "a loop's flag that views wider storage is refused"
        refuses_a_view_as_flag;
      test "a back edge whose flag is bound storage reads that storage"
        reads_a_bound_flag;
      test "a loop's copy part copies the rows its trips name"
        copies_moving_rows;
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
let run_refuses ?(devices = devices) ?(says = Exn.invalid_arg ~substring:"slot")
    what name slots_of =
  test ("run refuses " ^ what) (fun () ->
      let s, vars, big = parameterized ~devices name in
      let storage = storage_of ~devices big in
      raises_match says (fun () -> Engine.run ~vars s (slots_of storage)))

(* [rebound storage k f] is the slots of [storage] with slot [k]'s buffers [f
   buffers]. *)
let rebound k f storage =
  let slots = slots storage in
  slots.(k) <- f slots.(k);
  slots

(* Links the schedule of add with the buffers of its slot [k] replaced by [f] of
   them. *)
let with_bound k f () =
  let storage = storage_of (program "add") in
  let bound =
    List.map
      (fun s -> (s.node, if s.arg.slot = k then f s.buffers else s.buffers))
      storage
  in
  Engine.link ~devices ~bound (Lazy.force compiled_add)

let floats_on d n = Run.buffer (device d) Float32 (Array.make n (`Float 0.))

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
            (fst (Schedule.create_linear_with_vars (program "add"))));
      link_refuses "storage bound to buffers of another device"
        (with_bound 4 (fun _ -> [ floats_on "CPU:1" 16 ]));
      link_refuses "storage bound to fewer bytes than it holds"
        (with_bound 4 (fun _ -> [ floats_on "CPU" 15 ]));
      link_refuses "storage bound to a buffer for each of two devices"
        (with_bound 4 (fun bs -> bs @ bs));
      test "link refuses storage a kernel reads bound off its phase" (fun () ->
          raises_match off_its_phase
            (with_bound 2 (fun _ -> [ at_phase host 4 64 ])));
      link_refuses "a bound node that is no storage" (fun () ->
          let storage = storage_of (program "add") in
          let bound =
            List.map
              (fun s -> (Ops.replace ~op:Param s.node, s.buffers))
              storage
          in
          Engine.link ~devices ~bound (Lazy.force compiled_add));
      run_refuses "a parameter it binds no buffers" "add" (fun _ -> [||]);
      test "run refuses slots that stop before the last parameter" (fun () ->
          let s, vars, big = parameterized "add" in
          let slots = slots (storage_of big) in
          let last = Array.length slots - 1 in
          raises_match
            (Exn.invalid_arg ~substring:(Printf.sprintf "slot %d" last))
            (fun () -> Engine.run ~vars s (Array.sub slots 0 last)));
      run_refuses "a parameter bound to buffers of another device" "add"
        (fun storage ->
          let k = (List.hd storage).arg.slot in
          rebound k (fun _ -> [ floats_on "CPU:1" 16 ]) storage);
      run_refuses "a parameter bound to fewer bytes than it holds" "add"
        (fun storage ->
          let k = (List.hd storage).arg.slot in
          rebound k (fun _ -> [ floats_on "CPU" 15 ]) storage);
      run_refuses "a sharded parameter bound to one buffer" "shard_add"
        (fun storage ->
          let k =
            (List.find (fun s -> List.length s.buffers = 2) storage).arg.slot
          in
          rebound k (fun bs -> [ List.hd bs ]) storage);
      run_refuses "a sharded parameter whose second lane is off its phase"
        "shard_add"
        ~says:(fun e ->
          off_its_phase e
          && Exn.invalid_arg ~substring:"buffer of CPU:1 starts 4" e)
        (fun storage ->
          let k =
            (List.find (fun s -> List.length s.buffers = 2) storage).arg.slot
          in
          rebound k
            (function [ a; b ] -> [ a; off_phase b ] | bs -> bs)
            storage);
      test "run refuses a variable it binds no value" (fun () ->
          let s, _, big = parameterized "variable_shrink" in
          let storage = storage_of big in
          raises_match Exn.invalid_arg (fun () -> Engine.run s (slots storage)));
    ]

(* Runs

   A linked schedule's runs share its storage: two domains running one schedule
   on their own parameters each read back what theirs compute. A run allocates
   and loads nothing, and a linked schedule holds its storage while
   reachable. *)

let serialized ?(devices = devices) name () =
  let s, vars, big = parameterized ~devices name in
  let mine = storage_of ~devices big
  and theirs = storage_of ~devices ~seed:4 big in
  let expect storage = expected ~vars big storage in
  let loop storage () =
    let slots = slots storage and want = expect storage in
    let bad = ref None in
    for k = 1 to 200 do
      Engine.run ~vars s slots;
      let got = by_slot storage contents in
      if !bad = None && not (Testable.equal slot_values want got) then
        bad := Some (k, got)
    done;
    !bad
  in
  let other = Domain.spawn (loop theirs) in
  let here = loop mine () in
  let there = Domain.join other in
  let no_mismatch = option (pair int slot_values) in
  equal no_mismatch ~msg:"this domain" None here;
  equal no_mismatch ~msg:"the other domain" None there

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
  let buffer () = Ops.new_buffer (Single "CPU") n Float32 in
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
  let i = Ops.range (Int 4) [ 0 ] in
  let out = Call.placeholder ~slot:0 [ 4 ] Float32 in
  let kernel =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"fill" ())
      [ Ops.end_ (Ops.store (Ops.index out [ i ]) (Ops.O.float x)) [ i ] ]
  in
  let y = Ops.new_buffer (Single d) 4 Float32 in
  (y, Ops.call kernel [ y ])

(* At [DEBUG=1], a run of a schedule of [n] copies on the host, each a call of
   its own. *)
let run_of_fills n =
  let fills =
    List.init n (fun _ ->
        let y = Ops.new_buffer (Single "CPU") 4 Float32
        and x = Ops.new_buffer (Single "CPU") 4 Float32 in
        ((y, x), Call.store_call y x))
  in
  let bound =
    List.concat_map
      (fun ((y, x), _) ->
        [
          (y, [ Run.buffer host Float32 a ]); (x, [ Run.buffer host Float32 a ]);
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
    List.map (fun (y, _) -> (y, [ Run.buffer (devices name) Float32 a ])) fills
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
        at_least float_exact ~msg:"time" ~than:0. t;
        less float_exact ~msg:"time" ~than:1. t
      end)
    lines

let runs =
  group "runs"
    [
      test "runs of one schedule from two domains each compute their own"
        (serialized "contiguous");
      test "a run of kernels allocates and loads nothing"
        (allocates_nothing "contiguous");
      test "a run of copies allocates and loads nothing"
        (allocates_nothing "copy_computed");
      test "a linked schedule holds its storage while it is reachable"
        holds_its_storage;
      test "each kernel of a run is in a span of the host under a profile"
        spans_each_kernel;
      test "at DEBUG=1, a run of ten calls says how many it runs" (fun () ->
          equal (list string) [ "jit execs 10 calls" ] (run_of_fills 10));
      test "at DEBUG=1, a run of nine calls says nothing" (fun () ->
          equal (list string) [] (run_of_fills 9));
      test "at DEBUG=2, a run prints a timed line per kernel of the host"
        (reports_each_kernel "CPU");
      test "at DEBUG=2, a run prints a timed line per kernel of a memory device"
        (reports_each_kernel "CPU:1");
    ]

(* Timing *)

(* The devices a time runs on: the host, a memory device, whose kernels run on
   the host, and each GPU. *)
let timed_on =
  [
    ("the host", "CPU");
    ("a memory device", "CPU:1");
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
      let m = gpu kind in
      ((fun _ -> m), Engine.renderer m)
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

let clock =
  Testable.make
    ~pp:(fun ppf c ->
      Format.pp_print_string ppf
        (match c with Search.Device -> "Device" | Search.Host -> "Host"))
    ~equal:( = )

(* The host runs kernels and records their spans on its clock; a GPU stamps its
   work. *)
let clock_by_device name =
  let s, _ = timed name in
  equal clock
    (if List.mem_assoc name gpu_kind then Search.Device else Search.Host)
    (Engine.clock s)

(* A time under a profile takes its own: its clock is the one without, and its
   value is the sum of the spans of its kernels that the outer profile records
   too. *)
let time_under_profile name =
  let s, slots = timed name in
  let c, _, _ = profiled (fun () -> Engine.clock s) in
  equal ~msg:"the clock" clock (Engine.clock s) c;
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
  equal ~msg:"a submission of several kernels" bool true
    (List.exists (fun (names, _) -> List.length names > 1) works);
  let ns = List.fold_left (fun ns (_, d) -> ns + d) 0 works in
  equal ~msg:"the time" (float 1e-12) (Float.of_int ns *. 1e-9) t

let positive name =
  let s, slots = timed name in
  let t = Engine.time ~vars:n_bound s slots in
  greater float_exact ~than:0. t;
  less float_exact ~than:1. t

let leaves_the_profile name =
  let s, slots = timed name in
  let t, taken, _ = profiled (fun () -> Engine.time ~vars:n_bound s slots) in
  greater float_exact ~than:0. t;
  less float_exact ~than:1. t;
  is_true ~msg:"taken" taken

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

(* The slots of the linked schedule of the program [name], which hold as many
   buffers per slot as its storage's, and which a run takes. *)
let slots_fit name =
  let s, vars, storage = linked (program name) in
  let fresh = Engine.slots s in
  let counts slots = Array.to_list (Array.map List.length slots) in
  equal (list int) (counts (slots storage)) (counts fresh);
  Engine.run ~vars s fresh

let timing =
  let on f = cases ~name:fst "on" timed_on (fun (_, name) -> f name) in
  group "timing"
    [
      cases ~name:Fun.id "run takes the slots of a linked schedule"
        [ "add"; "shard_add" ] slots_fit;
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
          equal ~msg:"buffers by slot" (list int) [ 1; 1; 1; 1 ]
            (Array.to_list (Array.map List.length slots));
          Engine.run ~vars:n_bound s slots);
      group "the clock is a GPU's on a GPU, the host's otherwise"
        [ on clock_by_device ];
      group
        "a profile taken around a time changes neither its clock nor its value"
        [ on time_under_profile ];
      group "a time is positive, under a second" [ on positive ];
      test "a time counts a submission of several kernels"
        counts_grouped_kernels;
      group "a time under a profile leaves the profile taken"
        [ on leaves_the_profile ];
      group "each time runs its program once" [ on runs_once ];
      group "a time allocates and loads nothing" [ on time_allocates_nothing ];
      group "slots and a linked program, dropped, return their memory"
        [ on returns_its_memory ];
      group "a time prints nothing at DEBUG=2" [ on prints_nothing ];
      test "a time refuses an unbound variable" (fun () ->
          let s, slots = timed "CPU" in
          raises_match Exn.invalid_arg (fun () -> Engine.time s slots));
    ]

(* Host programs in blocks: a variable named after a block's bounds *)

(* A kernel that fills four floats with the variable [name], whose bounds are
   those a split of its loop would give its block's bounds. *)
let fill_with name d =
  let i = Ops.range (Int 4) [ 0 ] in
  let out = Call.placeholder ~slot:0 [ 4 ] Float32 in
  let v =
    Ops.variable ~dtype:Int32 name (Dtype.Value.of_int 0) (Dtype.Value.of_int 4)
  in
  let kernel =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"fill_with" ())
      [ Ops.end_ (Ops.store (Ops.index out [ i ]) (Ops.cast v Float32)) [ i ] ]
  in
  let y = Ops.new_buffer (Single d) 4 Float32 in
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
  let s = Engine.link ~devices ~bound:[ (y, [ out ]) ] compiled in
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

(* A kernel on CPU:1 and CPU:2 reads the last four of the eight floats each
   device holds of a sharded parameter, at the shard's address plus the view's
   offset. *)
let reads_a_view_of_each_shard () =
  let lanes = Ops.Multi [ "CPU:1"; "CPU:2" ] in
  let param = Call.param ~shape:[ Int 8 ] ~device:lanes 0 Float32 in
  let out = Ops.new_buffer lanes 4 Float32 in
  let results =
    List.map
      (fun d -> Run.buffer (device d) Float32 (floats [| 0.; 0.; 0.; 0. |]))
      [ "CPU:1"; "CPU:2" ]
  in
  let s =
    Engine.link ~devices
      ~bound:[ (out, results) ]
      (compile
         (Ops.v Op.Linear
            ~src:
              [
                Ops.call add_one
                  [ out; Shape.shrink param [ Some (Int 4, Int 8) ] ];
              ]))
  in
  let shard k = floats (Array.init 8 (fun i -> Float.of_int ((10 * k) + i))) in
  Engine.run s
    [|
      [
        Run.buffer (device "CPU:1") Float32 (shard 1);
        Run.buffer (device "CPU:2") Float32 (shard 2);
      ];
    |];
  equal (list values)
    [ floats [| 15.; 16.; 17.; 18. |]; floats [| 25.; 26.; 27.; 28. |] ]
    (List.map (Run.values Float32) results)

(* The minor words a run of one kernel on [d] of [devices] allocates on the
   calling domain, the run before it having linked and loaded everything. *)
let run_words ?(devices = devices) d () =
  let y, filled = fill d 7. in
  let compiled = compile ~devices (Ops.v Op.Linear ~src:[ filled ]) in
  let s =
    Engine.link ~devices
      ~bound:[ (y, [ Run.buffer (devices d) Float32 a ]) ]
      compiled
  in
  Engine.run s [||];
  let before = Gc.minor_words () in
  Engine.run s [||];
  let words = Gc.minor_words () -. before in
  Float.to_int words

let lanes =
  group "lanes and words"
    [
      test "a kernel reads a view of each shard of a parameter"
        reads_a_view_of_each_shard;
      test "a run of one kernel on the host allocates at most 400 minor words"
        (fun () -> at_most int ~than:400 (run_words "CPU" ()));
    ]

(* GPUs

   The recorded programs copy from the host into CPU:1, which is a GPU here, and
   compute on it. A GPU's tests skip without one. *)

let on_gpu kind name = match name with "CPU:1" -> gpu kind | n -> device n
let gpu_names = [ "CPU"; "CPU:1" ]

let computes_on devices name =
  slow (name ^ " writes what its tensors compute") (fun () ->
      let big = program name in
      let s, vars, storage = linked ~devices big in
      Engine.run ~vars s (slots storage);
      equal slot_values (expected ~vars big storage) (by_slot storage contents))

let on_a_gpu (kind, _) =
  let devices = on_gpu kind in
  group kind
    [
      group "recorded"
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
      cases ~tags:[ "slow" ] ~name:Fun.id
        "a schedule runs on the buffers each run binds to its parameters"
        [ "copy"; "copy_view" ] (fun name -> runs_on_its_slots ~devices name ());
      slow "runs of one schedule from two domains each compute their own"
        (serialized ~devices "copy");
      slow "a run allocates and loads nothing"
        (allocates_nothing ~devices ~names:gpu_names "copy");
      slow "a run of one kernel allocates at most 400 minor words" (fun () ->
          at_most int ~than:400 (run_words ~devices "CPU:1" ()));
      slow
        "a store through a gather writes each row at its loaded index and \
         drops an invalid one"
        (stores_through_a_gather ~devices "CPU:1" [| 6; -1; 8; 2 |]);
      prop ~tags:[ "slow" ]
        "a loop runs its call while its flag holds, at most its trips"
        Gen.(pair (int_range (-1) 6) (int_range 1 5))
        (fun bounds -> loops_while_a_flag_holds ~devices "CPU:1" bounds);
      prop ~tags:[ "slow" ]
        "a loop runs its calls in order each trip while its flag holds"
        loop_bounds (fun bounds -> loops_around_calls ~devices "CPU:1" bounds);
      reads_trips_on ~devices "CPU:1";
      reads_trips_under_a_back_edge ~devices "CPU:1";
      run_refuses ~devices ~says:off_its_phase
        "a buffer its kernel writes off its phase" "copy"
        (rebound 2 (List.map off_phase));
      slow "at DEBUG=2, a run prints a line per kernel"
        (reports_each_kernel ~devices "CPU:1");
    ]

let () =
  exit
    (run "Tolk_engine"
       [
         targets;
         on_the_host;
         in_blocks;
         schedules;
         gathered_stores;
         group "a store through a shrink" [ valid_slot_store ];
         refusals;
         runs;
         timing;
         named_as_bounds;
         lanes;
         group "GPUs" (List.map on_a_gpu gpus);
       ])
