(* Tests of Tolk_next_engine: the targets it gives devices, host programs, and
   linked schedules, whose runs write what their programs compute, on the
   buffers bound to them, one run at a time and without allocating. *)

open Windtrap
open Tolk_next
module Engine = Tolk_next_engine
module Buffer = Nx_device.Buffer

let host = Nx_device.host
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
   is a host device of its own, and its disk is named after a directory. *)
let named =
  ("CPU:0", host)
  :: ("PYTHON", List.assoc "CPU:3" (Run.devices ()))
  :: ("DISK:/tmp/tolk-next-schedule", Nx_device.disk)
  :: Run.devices ()

let device name = List.assoc name named
let devices = Engine.device named
let clang = lazy (Cstyle.clang (Engine.target host))

(* Test devices *)

(* A device of host memory that it does not map. *)
let unmapped =
  lazy
    (Nx_device.Driver.device ~name:"UNMAPPED" ~arch:"test" ~budget:max_int
       (Host_visible { memory = Nx_device.Driver.host_memory; mapping = None }))

let never _ = failwith "the test device runs no work"

(* A device whose own memory the host does not address. *)
let local =
  lazy
    (let memory = Nx_device.Driver.host_memory in
     let queue ~timeline:_ =
       {
         Nx_device.Driver.copy = (fun ~dst:_ ~src:_ _ ~signal:_ -> never ());
         transfer = (fun _ -> None);
         stamp = (fun ~slot:_ ~signal:_ -> never ());
         clock = Host_clock;
       }
     in
     Nx_device.Driver.device ~name:"LOCAL" ~arch:"test" ~budget:max_int
       (Device_local
          {
            memory;
            host_memory = memory;
            mapped = None;
            mapping = Identity;
            queue;
          }))

(* The host of another machine, which no test reaches. *)
let remote =
  lazy
    (Nx_device.Driver.host ~address:"10.0.0.2:6667" ~arch:"x86_64"
       ~memory:Nx_device.Driver.host_memory
       {
         read = (fun ~src:_ ~dst:_ _ -> ());
         write = (fun ~dst:_ ~src:_ _ -> ());
         copy = (fun ~dst:_ ~src:_ _ -> ());
       })

(* Devices *)

let target = Testable.make ~pp:Helpers.Target.pp ~equal:( = )

let host_target arch cpu =
  { (Helpers.target ~arch:(arch ^ "," ^ cpu) "CPU") with renderer = "CLANG" }

let targets =
  group "target"
    [
      test "the host compiles C for Clang on its own processor" (fun () ->
          equal target
            (host_target (Nx_device.arch host) "native")
            (Engine.target host));
      test "a test device of the host's memory compiles as the host" (fun () ->
          equal target (Engine.target host) (Engine.target (device "CPU:1")));
      test
        "a device of host memory that does not map the host's runs no program"
        (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Engine.target (Lazy.force unmapped)));
      test "another machine's host compiles for a generic processor of its arch"
        (fun () ->
          equal target
            (host_target "x86_64" "generic")
            (Engine.target (Lazy.force remote)));
      test "the disk runs no program" (fun () ->
          raises_match Exn.invalid_arg (fun () -> Engine.target Nx_device.disk));
      test "a device whose memory the host does not address runs no program"
        (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Engine.target (Lazy.force local)));
      test
        "a target's renderer is made once, and shared by the devices that \
         compile for it" (fun () ->
          is_true (Engine.renderer host == Engine.renderer host);
          is_true (Engine.renderer host == Engine.renderer (device "CPU:1"));
          raises_match Exn.invalid_arg (fun () ->
              Engine.renderer Nx_device.disk));
      test "Metal compiles for its GPU family" (fun () ->
          match Metal.device with
          | None -> skip ~reason:"no Metal device" ()
          | Some d ->
              let t = Engine.target d in
              equal (pair string string)
                ("METAL", Nx_device.arch d)
                (t.device, t.arch));
    ]

let describing =
  group "device"
    [
      test "a device off Metal, CUDA, AMD and NV runs no queues" (fun () ->
          let d = devices "CPU:1" in
          is_true (Nx_device.equal (device "CPU:1") d.device);
          equal target (Engine.target d.device) d.compiler.target;
          is_true (Option.is_none d.compiler.queues));
      test "the disk is described as a DISK target without queues" (fun () ->
          let d = devices "DISK:/tmp/tolk-next-schedule" in
          is_true (Nx_device.equal Nx_device.disk d.device);
          equal target (Helpers.target "DISK") d.compiler.target;
          is_true (Option.is_none d.compiler.queues));
      test "a name the map does not hold is refused" (fun () ->
          raises_match Exn.invalid_arg (fun () -> devices "CPU:9"));
      test "a map that gives one name twice is refused" (fun () ->
          raises
            (Invalid_argument "Tolk_next_engine.device: CPU names two devices")
            (fun () ->
              Engine.device [ ("CPU", device "CPU:1"); ("CPU", host) ] "CPU"));
      test
        "a device's host that the map does not name is named after itself, and \
         a named one is not" (fun () ->
          let d = Lazy.force unmapped in
          is_true ~msg:"unnamed"
            (Nx_device.equal Nx_device.host
               (Engine.device [ ("X", d) ] "CPU").device);
          raises_match ~msg:"named H" Exn.invalid_arg (fun () ->
              Engine.device [ ("X", d); ("H", host) ] "CPU"));
      test
        "a Metal device whose host's name the map gives to another device is \
         refused" (fun () ->
          match Metal.device with
          | None -> skip ~reason:"no Metal device" ()
          | Some m ->
              raises
                (Invalid_argument
                   "Tolk_next_engine.device: CPU names another device than \
                    METAL's host") (fun () ->
                  Engine.device
                    [ ("METAL", m); ("CPU", device "CPU:1") ]
                    "METAL"));
    ]

(* Host programs

   A kernel built from UOps: out[i] = a[i] * n + b[i] for [size] elements
   (default four), where out, a and b are the parameters 0, 1 and 2 and n a
   variable. *)

let kernel ?(name = "axpy") ?(size = 4) ?bound () =
  let open Ops.O in
  let buffer slot = Ops.placeholder ~slot [ size ] Float32 in
  let n =
    Ops.variable ~dtype:Int32 "n" (Dtype.Value.of_int 0)
      (Dtype.Value.of_int 100)
  in
  let n =
    match bound with Some v -> Ops.bind n (`Int (Z.of_int v)) | None -> n
  in
  let i = Ops.range (Int size) [ 0 ] in
  let at b = Ops.index (buffer b) [ i ] in
  Ops.sink ~kernel:(Ops.kernel_info ~name ())
    [ Ops.end_ (Ops.store (at 0) ((at 1 * Ops.cast n Float32) + at 2)) [ i ] ]

let compiled ?name ?size ?bound () =
  Codegen.to_program (kernel ?name ?size ?bound ()) (Lazy.force clang)

let axpy = lazy (compiled ())

(* A kernel long enough for the host clock to see it run. *)
let long_axpy = lazy (compiled ~name:"long_axpy" ~size:(1 lsl 18) ())
let a = floats [| 1.; 2.; 3.; 4. |]
let b = floats [| 10.; 20.; 30.; 40. |]

let buffers () =
  [
    Run.buffer host Float32 (floats [| 0.; 0.; 0.; 0. |]);
    Run.buffer host Float32 a;
    Run.buffer host Float32 b;
  ]

let runs_a_kernel () =
  let p = Engine.Program.load host (Lazy.force axpy) in
  let bufs = buffers () in
  Engine.Program.run ~vars:[ ("n", 3) ] p bufs;
  equal values
    (floats [| 13.; 26.; 39.; 52. |])
    (Run.values Float32 (List.hd bufs))

let takes_the_bound_value () =
  let p = Engine.Program.load host (compiled ~name:"axpy_bound" ~bound:2 ()) in
  let bufs = buffers () in
  Engine.Program.run p bufs;
  equal values
    (floats [| 12.; 24.; 36.; 48. |])
    (Run.values Float32 (List.hd bufs))

(* The loads of programs a profile of [f] records. *)
let loads f =
  let p = Nx_device.Profile.start () in
  let result = f () in
  let loads =
    List.filter
      (function Nx_device.Profile.Load _ -> true | _ -> false)
      (Nx_device.Profile.stop p)
  in
  (result, List.length loads)

let loads_once () =
  let prg = compiled ~name:"axpy_loaded_once" () in
  let _, n =
    loads (fun () ->
        (Engine.Program.load host prg, Engine.Program.load host prg))
  in
  equal int 1 n

(* The do-while loops of the Linearizer suite, each storing what it counts. *)
let counts_in_a_loop (name, count) =
  test
    (name ^ " runs its loops to " ^ string_of_int count)
    (fun () ->
      let sink =
        Golden.sink ("../../codegen/late/linearizer/" ^ name ^ ".golden")
      in
      let p =
        Engine.Program.load host (Codegen.to_program sink (Lazy.force clang))
      in
      let out = Run.buffer host Int32 [| `Int Z.zero |] in
      Engine.Program.run p [ out ];
      equal values [| `Int (Z.of_int count) |] (Run.values Int32 out))

let with_binary bytes prg =
  let srcs = Ops.src prg in
  Ops.replace prg
    ~src:
      (List.filteri (fun i _ -> i < 3) srcs
      @ [ Ops.v Op.Binary ~arg:(Ops.Bytes bytes) ])

let programs =
  group "Program"
    [
      test "a kernel runs on its buffers, in the order of its globals"
        runs_a_kernel;
      test "a variable left out of vars takes its bound value"
        takes_the_bound_value;
      test "a binary loaded twice is loaded once" loads_once;
      group "loops"
        (List.map counts_in_a_loop
           [
             ("wait_loop", 10);
             ("nested_loop", 12);
             ("two_loops", 25);
             ("loop_in_loop", 12);
           ]);
      test "load refuses a device that runs no program" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Engine.Program.load (Lazy.force local) (Lazy.force axpy)));
      test "load refuses Metal, whose programs its queues run" (fun () ->
          match Metal.device with
          | None -> skip ~reason:"no Metal device" ()
          | Some d ->
              raises_match Exn.invalid_arg (fun () ->
                  Engine.Program.load d (Lazy.force axpy)));
      test "load refuses a node that is no compiled program" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Engine.Program.load host (kernel ())));
      test "load fails with the host's reason for refusing a binary" (fun () ->
          raises_match Exn.failure (fun () ->
              Engine.Program.load host
                (with_binary "no object" (Lazy.force axpy))));
      test "run refuses fewer buffers than parameters" (fun () ->
          let p = Engine.Program.load host (Lazy.force axpy) in
          raises_match Exn.invalid_arg (fun () ->
              Engine.Program.run ~vars:[ ("n", 1) ] p (List.tl (buffers ()))));
      test "run refuses a buffer smaller than its parameter" (fun () ->
          let p = Engine.Program.load host (Lazy.force axpy) in
          let small = Run.buffer host Float32 (floats [| 0.; 0.; 0. |]) in
          raises_match Exn.invalid_arg (fun () ->
              Engine.Program.run
                ~vars:[ ("n", 1) ]
                p
                (small :: List.tl (buffers ()))));
      test "run refuses an unbound variable" (fun () ->
          let p = Engine.Program.load host (Lazy.force axpy) in
          raises_match Exn.invalid_arg (fun () ->
              Engine.Program.run p (buffers ())));
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
  else if Dtype.is_unsigned dtype then `Int (Z.of_int (k + 3))
  else `Int (Z.of_int k)

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
  if not (Nx_device.equal d Nx_device.disk) then Run.buffer d dt values
  else
    let file =
      Result.get_ok
        (Buffer.create_file (temp_file ())
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
                buffer (devices d).device p.dtype (Array.sub before (k * m) m))
              names
          in
          Some { node = n; arg = p; before; buffers }
      | _ -> None)
    (Ops.toposort ~enter_calls:false big)

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
  Ops.substitute big
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
       (Ops.toposort big))

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
  (Hcq2.compile_linear ~devices:(fun n -> (devices n).compiler) linear, vars)

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
    Array.map (fun x -> `Int (Z.of_int32 (Int32.bits_of_float x))) sums
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

(* [parameterized name] is the recorded program [name] linked with its buffers
   made parameters of their slots, which each run binds, and the program with
   those parameters. *)
let parameterized ?(devices = devices) name =
  let big = program name in
  let parameters =
    List.filter_map
      (fun n ->
        if Ops.op n = Buffer then Some (n, Ops.replace ~op:Param n) else None)
      (Ops.toposort ~enter_calls:false big)
  in
  let linear, vars = Schedule.create_linear_with_vars big in
  let compiled =
    Hcq2.compile_linear
      ~devices:(fun n -> (devices n).compiler)
      (Ops.substitute linear parameters)
  in
  (Engine.link ~devices compiled, vars, Ops.substitute big parameters)

let runs_on_its_slots ?(devices = devices) name () =
  let s, vars, big = parameterized ~devices name in
  List.iter
    (fun storage ->
      Engine.run ~vars s (slots storage);
      equal slot_values (expected ~vars big storage) (by_slot storage contents))
    [ storage_of ~devices big; storage_of ~devices ~seed:5 big ]

(* A range of three trips around a call of a kernel that adds one to four
   floats, each trip on the next four of twelve. *)
(* A kernel that stores its parameter 1 plus one into its parameter 0, four
   floats. *)
let add_one =
  let open Ops.O in
  let i = Ops.range (Int 4) [ 0 ] in
  let at slot = Ops.index (Ops.placeholder ~slot [ 4 ] Float32) [ i ] in
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
    Ops.shrink b
      [ Some (Sym (r * Ops.int 4), Sym ((r * Ops.int 4) + Ops.int 4)) ]
  in
  let linear =
    Ops.v Op.Linear
      ~src:[ Ops.end_ (Ops.call one [ trip out; trip src ]) [ r ] ]
  in
  let compiled =
    Hcq2.compile_linear ~devices:(fun n -> (devices n).compiler) linear
  in
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
  let p slot = Ops.param ~shape:[ Int k ] ~device:cpu slot Float32 in
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
    Ops.shrink b
      [ Some (Sym Ops.O.(r * int k), Sym Ops.O.((r * int k) + int k)) ]
  in
  let e =
    Ops.end_ (Ops.call ~precompile:true body [ c; row xs; row ys ]) [ r ]
  in
  let linear, _ =
    Schedule.create_linear_with_vars
      (Ops.sink [ Ops.after c [ e ]; Ops.after ys [ e ] ])
  in
  let compiled =
    Hcq2.compile_linear ~devices:(fun n -> (devices n).compiler) linear
  in
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

(* A memory-planned schedule: z = a + 1; in a range, y = b + 1, into [y] or into
   a view of its first half; then z + 1 and y + 1 into outputs, each kernel on
   the first four elements of its buffers. The range writes y before the call
   that reads it, so the plan does not place y over z, which the range leaves
   for the call after it. *)
let plans_the_buffers_of_a_range ~through_a_view () =
  let buf n = Ops.new_buffer (Single "CPU") n Float32 in
  let a = buf 4 and b = buf 4 and z = buf 8 and y = buf 8 in
  let out_z = buf 4 and out_y = buf 4 in
  let r = Ops.range (Int 2) [ 7 ] in
  let linear =
    Ops.v Op.Linear
      ~src:
        [
          Ops.call add_one [ z; a ];
          Ops.end_
            (Ops.call add_one
               [
                 (if through_a_view then Ops.shrink y [ Some (Int 0, Int 4) ]
                  else y);
                 b;
               ])
            [ r ];
          Ops.call add_one [ out_z; z ];
          Ops.call add_one [ out_y; y ];
        ]
  in
  let planned =
    Memory.memory_plan_rewrite ~held_bufs:[ a; b; out_z; out_y ] linear
  in
  let compiled =
    Hcq2.compile_linear ~devices:(fun n -> (devices n).compiler) planned
  in
  let zeros () = Run.buffer host Float32 (floats [| 0.; 0.; 0.; 0. |]) in
  let result_z = zeros () and result_y = zeros () in
  let bound =
    [
      (a, [ Run.buffer host Float32 (floats [| 1.; 2.; 3.; 4. |]) ]);
      (b, [ Run.buffer host Float32 (floats [| 10.; 20.; 30.; 40. |]) ]);
      (out_z, [ result_z ]);
      (out_y, [ result_y ]);
    ]
  in
  Engine.run (Engine.link ~devices ~bound compiled) [||];
  equal values (floats [| 3.; 4.; 5.; 6. |]) (Run.values Float32 result_z);
  equal values (floats [| 12.; 22.; 32.; 42. |]) (Run.values Float32 result_y)

(* A copy out of a device, a copy into it, and the copy out again: the last copy
   reads what the second wrote. *)
let copies_in_order () =
  let bytes k = Array.init 16 (fun i -> `Int (Z.of_int (i * k land 0xff))) in
  let node d = Ops.new_buffer (Single d) 16 Uint8 in
  let device_side = node "CPU:1" and out = node "CPU" and fresh = node "CPU" in
  let linear =
    Ops.v Op.Linear
      ~src:
        [
          Ops.store_call out device_side;
          Ops.store_call device_side fresh;
          Ops.store_call out device_side;
        ]
  in
  let compiled =
    Hcq2.compile_linear ~devices:(fun n -> (devices n).compiler) linear
  in
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

(* A store through a padded view: a row of 8 of [x @ w] into [pool], [8; 8],
   padded by one row, at the row the loaded [slot] gives, or at the padding row
   when [slot] lies outside [pool]. With [read], the graph also stores the sums
   of the rows of [pool] padded by a row of [fill], [9] of them, a selection of
   [fill] off the very pad node the store goes through, and the sums of [pool]'s
   rows read directly, [8] of them. *)
let padded_store ?(fill = 0.) ?(read = false) device =
  let buffer n dt shape =
    Ops.reshape
      (Ops.new_buffer (Single device) n dt)
      (List.map (fun d -> Ops.Int d) shape)
  in
  let pool = buffer 64 Float32 [ 8; 8 ] in
  let slot = buffer 1 Int32 [] in
  let x = buffer 4 Float32 [ 1; 4; 1 ] and w = buffer 32 Float32 [ 1; 4; 8 ] in
  let inside =
    Ops.bitwise_and (Ops.ge slot (Ops.int 0)) (Ops.lt slot (Ops.int 8))
  in
  let at = Ops.where inside slot (Ops.int 8) in
  let padding = [ Some (Ops.Int 0, Ops.Int 1); None ] in
  let padded = Ops.pad pool padding in
  let view =
    Ops.shrink padded
      [ Some (Ops.Sym at, Ops.Sym (Ops.add at (Ops.int 1))); None ]
  in
  let row = Ops.rop (Ops.mul x w) Op.Add [ 1 ] in
  let stored = Ops.after view [ Ops.store view row ] in
  if not read then Ops.sink [ stored ]
  else
    let filled = Ops.pad ~value:(`Float fill) pool padding in
    if not (List.memq padded (Ops.toposort filled)) then
      fail "the read does not share the stored pad node";
    let sums = buffer 9 Float32 [ 9 ] and direct = buffer 8 Float32 [ 8 ] in
    Ops.sink
      [
        stored;
        Ops.after sums [ Ops.store sums (Ops.rop filled Op.Add [ 1 ]) ];
        Ops.after direct [ Ops.store direct (Ops.rop pool Op.Add [ 1 ]) ];
      ]

(* [stores_through_a_pad ~devices ~fill ~read device at] runs [padded_store]
   with [slot] at [at]: [pool] holds the row at [at] when it lies within it, and
   is otherwise untouched, and with [read] the sums of the padded rows are
   [pool]'s rows before the store and [8 * fill] for the padding, and the direct
   sums [pool]'s rows before the store. *)
let stores_through_a_pad ?(devices = devices) ?fill ?(read = false) device at ()
    =
  let big = padded_store ?fill ~read device in
  let calls = Ops.src (fst (Schedule.create_linear_with_vars big)) in
  if not read then equal ~msg:"kernels" int 1 (List.length calls);
  let s, vars, storage = linked ~devices big in
  let of_size n = List.find (fun st -> per_device st.arg = n) storage in
  let pool = of_size 64 and slot = of_size 1 in
  let x = of_size 4 and w = of_size 32 in
  List.iter
    (fun dst ->
      Buffer.copy ~src:(Run.buffer host Int32 [| `Int (Z.of_int at) |]) ~dst)
    slot.buffers;
  Engine.run ~vars s (slots storage);
  let num v = match v with `Float f -> f | _ -> fail "a float" in
  let expected =
    Array.mapi
      (fun i v ->
        if at >= 0 && at < 8 && i / 8 = at then
          let j = i mod 8 in
          `Float
            (List.fold_left ( +. ) 0.
               (List.init 4 (fun k ->
                    num x.before.(k) *. num w.before.((k * 8) + j))))
        else v)
      pool.before
  in
  equal values expected (contents pool);
  if read then (
    let sum r =
      List.fold_left ( +. ) 0.
        (List.init 8 (fun j -> num pool.before.((r * 8) + j)))
    in
    let fill = Option.value fill ~default:0. in
    equal ~msg:"the padded rows read" values
      (Array.init 9 (fun r -> `Float (if r < 8 then sum r else 8. *. fill)))
      (contents (of_size 9));
    equal ~msg:"the rows read without the pad" values
      (Array.init 8 (fun r -> `Float (sum r)))
      (contents (of_size 8)))


(* A store of a row of [x @ w] into [pool], [8; 8], through a shrink whose
   bounds are the loaded [slot] carrying its validity: the shrink's size is 1
   where [slot] lies within [pool], and Invalid elsewhere. *)
let store_at_valid_slot device =
  let buffer n dt shape =
    Ops.reshape
      (Ops.new_buffer (Single device) n dt)
      (List.map (fun d -> Ops.Int d) shape)
  in
  let pool = buffer 64 Float32 [ 8; 8 ] in
  let slot = buffer 1 Int32 [] in
  let x = buffer 4 Float32 [ 1; 4; 1 ] and w = buffer 32 Float32 [ 1; 4; 8 ] in
  let inside =
    Ops.bitwise_and (Ops.ge slot (Ops.int 0)) (Ops.lt slot (Ops.int 8))
  in
  let at = Ops.valid slot inside in
  let view =
    Ops.shrink pool
      [ Some (Ops.Sym at, Ops.Sym (Ops.add at (Ops.int 1))); None ]
  in
  let row = Ops.rop (Ops.mul x w) Op.Add [ 1 ] in
  Ops.sink [ Ops.after view [ Ops.store view row ] ]

let valid_slot_store =
  test
    "a store through a shrink whose bounds carry a validity is refused, its \
     size holding Invalid" (fun () ->
      raises_match
        (Exn.invalid_arg ~substring:"Invalid, which is no number")
        (fun () ->
          Schedule.create_linear_with_vars (store_at_valid_slot "CPU")))

let padded_stores =
  group "a store through a padded view (D69)"
    [
      cases ~name:string_of_int "writes the row within the source" [ 0; 3; 7 ]
        (fun at -> stores_through_a_pad "CPU" at ());
      cases ~name:string_of_int "writes nothing outside the source" [ -1; 8; 9 ]
        (fun at -> stores_through_a_pad "CPU" at ());
      cases ~name:string_of_int
        "a read of the same padded node reads its fill in the padding"
        [ 3; -1; 8 ] (fun at ->
          stores_through_a_pad ~fill:7. ~read:true "CPU" at ());
    ]

let schedules =
  group "link and run"
    [
      group "recorded" (List.map writes_what_it_computes recorded);
      slow "custom_kernel runs its custom kernel" runs_a_custom_kernel;
      slow "assign_bitcast stores through a bitcast" stores_through_a_bitcast;
      test "a schedule runs with each binding of its variables"
        runs_each_binding;
      cases ~name:Fun.id
        "a schedule runs on the buffers each run binds to its parameters"
        [ "contiguous"; "copy_view"; "shard_add" ] (fun name ->
          runs_on_its_slots name ());
      test "a range around a call runs it once per trip" runs_once_per_trip;
      test "a scan runs its body once per trip, carrying in place (D60)"
        scans_with_a_carry;
      test "a planned buffer a range writes is not placed over one it leaves"
        (plans_the_buffers_of_a_range ~through_a_view:false);
      test "a buffer a range writes through a view is not placed over another"
        (plans_the_buffers_of_a_range ~through_a_view:true);
      test "copies run in the order of their schedule" copies_in_order;
    ]

(* Refusals *)

let compiled_add = lazy (fst (schedule (program "add")))

let link_refuses what f =
  test ("link refuses " ^ what) (fun () -> raises_match Exn.invalid_arg f)

(* A refusal names the slot. *)
let run_refuses ?(devices = devices) what name slots_of =
  test ("run refuses " ^ what) (fun () ->
      let s, vars, big = parameterized ~devices name in
      let storage = storage_of ~devices big in
      raises_match (Exn.invalid_arg ~substring:"slot") (fun () ->
          Engine.run ~vars s (slots_of storage)))

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
            ~devices:(Engine.device [ ("CPU", host) ])
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

(* A run's allocations and loads are the profile's Allocation and Load events,
   and a run that allocates nothing leaves every device's allocated bytes as
   they were, or fewer, since the collector may return memory meanwhile. *)
let allocates_nothing ?(devices = devices)
    ?(names = [ "CPU"; "CPU:1"; "CPU:2"; "CPU:3" ]) name () =
  let s, vars, big = parameterized ~devices name in
  let slots = slots (storage_of ~devices big) in
  Engine.run ~vars s slots;
  let stats () = List.map (fun n -> Nx_device.stats (devices n).device) names in
  Gc.full_major ();
  let before = stats () in
  let p = Nx_device.Profile.start () in
  Engine.run ~vars s slots;
  let events = Nx_device.Profile.stop p in
  let count f = List.length (List.filter f events) in
  equal int ~msg:"allocations" 0
    (count (function Nx_device.Profile.Allocation _ -> true | _ -> false));
  equal int ~msg:"loads" 0
    (count (function Nx_device.Profile.Load _ -> true | _ -> false));
  List.iter2
    (fun name d -> at_most int ~msg:name ~than:0 (Nx_device.Stats.allocated d))
    names
    (List.map2 Nx_device.Stats.diff before (stats ()))

(* [s] is unreachable after its last use: native code does not keep it
   longer. *)
let holds_its_storage () =
  let compiled, _ = schedule (program "contiguous") in
  let allocated () =
    Gc.full_major ();
    Gc.full_major ();
    Nx_device.Stats.allocated (Nx_device.stats host)
  in
  let before = allocated () in
  let s = Engine.link ~devices compiled in
  greater int ~msg:"reachable" ~than:before (allocated ());
  ignore (Sys.opaque_identity s);
  equal int ~msg:"unreachable" before (allocated ())

(* Under a profile, each kernel of a run is a span of the host, named after the
   kernel's function. *)
let spans_each_kernel () =
  let big = program "contiguous" in
  let compiled, vars = schedule big in
  let s = Engine.link ~devices ~bound:(bound (storage_of big)) compiled in
  let kernels =
    List.filter_map
      (fun u ->
        if Ops.op u <> Program then None
        else
          match Ops.arg (Ops.nth u 0) with
          | Kernel k -> Some (Ops.function_name k)
          | _ -> None)
      (Ops.toposort ~enter_calls:true compiled)
  in
  let p = Nx_device.Profile.start () in
  Engine.run ~vars s [||];
  let spans =
    List.filter_map
      (function
        | Nx_device.Profile.Span sp when Nx_device.equal sp.device host ->
            Some sp.name
        | _ -> None)
      (Nx_device.Profile.stop p)
  in
  not_equal (list string) [] kernels;
  equal (slist string String.compare) kernels spans

(* A kernel on [d] that fills four floats with [x]. *)
let fill d x =
  let i = Ops.range (Int 4) [ 0 ] in
  let out = Ops.placeholder ~slot:0 [ 4 ] Float32 in
  let kernel =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"fill" ())
      [ Ops.end_ (Ops.store (Ops.index out [ i ]) (Ops.O.float x)) [ i ] ]
  in
  let y = Ops.new_buffer (Single d) 4 Float32 in
  (y, Ops.call kernel [ y ])

(* At [DEBUG=1], a run of a schedule of [n] fills on the host. *)
let run_of_fills n =
  let fills = List.init n (fun _ -> fill "CPU" 1.) in
  let bound =
    List.map (fun (y, _) -> (y, [ Run.buffer host Float32 a ])) fills
  in
  let compiled =
    Hcq2.compile_linear
      ~devices:(fun n -> (devices n).compiler)
      (Ops.v Op.Linear ~src:(List.map snd fills))
  in
  let s = Engine.link ~devices ~bound compiled in
  Helpers.context [ B (Helpers.debug, 1) ] (fun () -> Engine.run s [||]);
  List.filter
    (String.starts_with ~prefix:"jit execs")
    (String.split_on_char '\n' (output ()))

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
      test "each kernel of a run is a span of the host under a profile"
        spans_each_kernel;
      test "at DEBUG=1, a run of ten calls says how many it runs" (fun () ->
          equal (list string) [ "jit execs 10 calls" ] (run_of_fills 10));
      test "at DEBUG=1, a run of nine calls says nothing" (fun () ->
          equal (list string) [] (run_of_fills 9));
    ]

(* Measuring *)

let measures =
  group "measure"
    [
      test "a program's run on the host takes a positive time, under a second"
        (fun () ->
          let t =
            Engine.measure
              ~vars:[ ("n", 3) ]
              ~devices "CPU" (Lazy.force long_axpy)
          in
          greater float_exact ~than:0. t;
          less float_exact ~than:1. t);
      test "a cold run takes a positive time" (fun () ->
          greater float_exact ~than:0.
            (Engine.measure ~cold:true
               ~vars:[ ("n", 3) ]
               ~devices "CPU:1" (Lazy.force long_axpy)));
      test "a run measured under a profile leaves the profile taken" (fun () ->
          let p = Nx_device.Profile.start () in
          let t, taken =
            Fun.protect
              ~finally:(fun () -> ignore (Nx_device.Profile.stop p))
              (fun () ->
                let t =
                  Engine.measure
                    ~vars:[ ("n", 3) ]
                    ~devices "CPU" (Lazy.force long_axpy)
                in
                (t, Nx_device.Profile.enabled ()))
          in
          greater float_exact ~than:0. t;
          less float_exact ~than:1. t;
          is_true taken);
      test "a kernel longer than 10 us is run once" (fun () ->
          let p = Nx_device.Profile.start () in
          let spans =
            Fun.protect
              ~finally:(fun () ->
                if Nx_device.Profile.enabled () then
                  ignore (Nx_device.Profile.stop p))
              (fun () ->
                ignore
                  (Engine.measure
                     ~vars:[ ("n", 3) ]
                     ~devices "CPU" (Lazy.force long_axpy));
                List.filter
                  (function
                    | Nx_device.Profile.Span sp ->
                        String.starts_with ~prefix:"long_axpy" sp.name
                    | _ -> false)
                  (Nx_device.Profile.stop p))
          in
          equal int 1 (List.length spans));
      test "a four-element kernel takes a positive time, below a clock tick"
        (fun () ->
          for _ = 1 to 20 do
            greater float_exact ~than:0.
              (Engine.measure
                 ~vars:[ ("n", 3) ]
                 ~devices "CPU" (Lazy.force axpy))
          done);
      test "a name the map does not hold is refused" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Engine.measure
                ~vars:[ ("n", 3) ]
                ~devices "CPU:9" (Lazy.force axpy)));
      test "an unbound variable is refused" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Engine.measure ~devices "CPU" (Lazy.force long_axpy)));
    ]

(* Batches

   The NULL devices of test support run their queues on a domain of their own,
   behind the host. The recorded copy puts a copy and a kernel on CPU:1 in one
   batch, and shard_add a kernel on each of CPU and CPU:1. *)

let null = lazy (Null_device.devices ())
let on_null name = (Lazy.force null) name

let computes_on_null name =
  test (name ^ " writes what its tensors compute") (fun () ->
      let big = program name in
      let s, vars, storage = linked ~devices:on_null big in
      Engine.run ~vars s (slots storage);
      equal slot_values (expected ~vars big storage) (by_slot storage contents))

(* Each run of a batch on a device signals the device's next value, once. *)
let signals_once_per_run ?(devices = on_null) () =
  let s, vars, big = parameterized ~devices "copy" in
  let slots = slots (storage_of ~devices big) in
  let d = (devices "CPU:1").device in
  let before = Nx_device.submitted d in
  for _ = 1 to 3 do
    Engine.run ~vars s slots
  done;
  Nx_device.synchronize d;
  equal int ~msg:"submitted" (before + 3) (Nx_device.submitted d);
  equal int ~msg:"signaled" (before + 3) (Nx_device.signaled d)

let link_calls ?profile ~bound calls =
  let devices = on_null in
  let compiled =
    Hcq2.compile_linear ?profile
      ~devices:(fun n -> (devices n).compiler)
      (Ops.v Op.Linear ~src:calls)
  in
  Engine.link ~devices ~bound compiled

(* A copy on CPU:1's queue into CPU:2's memory, which a slow kernel of CPU:2
   filled first: the copy lands last. *)
let waits_for_another_device () =
  let x = Ops.new_buffer (Single "CPU:1") 4 Float32 in
  let y, filled = fill "CPU:2" 7. in
  let src = Run.buffer (Null_device.device "CPU:1") Float32 a in
  let dst =
    Run.buffer
      (Null_device.device "CPU:2")
      Float32
      (floats [| 0.; 0.; 0.; 0. |])
  in
  let fills = link_calls ~bound:[ (y, [ dst ]) ] [ filled ] in
  let copies =
    link_calls ~bound:[ (x, [ src ]); (y, [ dst ]) ] [ Ops.store_call y x ]
  in
  Null_device.with_latency 0.05 (fun () -> Engine.run fills [||]);
  Engine.run copies [||];
  Null_device.synchronize ();
  equal values a (Run.values Float32 dst)

(* The NULL devices, where [b] is the storage of the placeholder tagged
   [tag]. *)
let with_tag tag b n =
  let dev = on_null n in
  let placeholder u =
    match Ops.tag u with
    | Some (String t) when t = tag -> Some b
    | _ -> dev.placeholder u
  in
  { dev with placeholder }

(* A batch of [d] alone whose host program runs [effects], then signals [d]. *)
let host_program ~(devices : string -> Engine.device) d effects =
  let info : Ops.hcq_info =
    {
      device = [ d ];
      kernels = [];
      estimates = { ops = Int 0; lds = Int 0; mem = Int 0 };
      nargs = 0;
      table = -1;
      inputs = [];
      slots = [];
      written_bufs = [];
      writes = [];
    }
  in
  let signal =
    Ops.store (Ops.index (Hcq2.signal_word d) [ Ops.int 0 ]) (Hcq2.value d)
  in
  let body =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"host_program" ())
      (effects @ [ signal ])
  in
  let lowered =
    Hcq2.lower_call
      ~devices:(fun n -> (devices n).compiler)
      (Ops.call ~aux:info body [])
  in
  Realize.lower_and_compile
    ~targets:(fun n -> (devices n).compiler.target)
    (Ops.v Op.Linear ~src:[ lowered ])

(* Work of [d] from a submitter of its own, such as a vendor's kernel launcher,
   that touches [b] alone: 20 ms after its submission, a domain fills [b] with
   [x], then stores the work's value into [d]'s signal word. *)
let fill_later d b x =
  Nx_device.submit [ d ] ~touches:[ b ] (fun s ->
      let v = Nx_device.Submission.value s d in
      let borrowed b = Result.get_ok (Buffer.borrow host b) in
      let elements = borrowed b and word = borrowed (Nx_device.signal_word d) in
      let t0 = Nx_device.Profile.now () in
      Domain.spawn (fun () ->
          while Nx_device.Profile.now () - t0 < 20_000_000 do
            Domain.cpu_relax ()
          done;
          Bigarray.Array1.fill (Buffer.bigarray Bigarray.float32 elements) x;
          (Buffer.bigarray Bigarray.int64 word).{0} <- Int64.of_int v))

(* A copy from CPU:2, a device without queues, into CPU:1's memory, which the
   schedule puts on CPU:1's queue, while other work of CPU:2 fills the source.
   No queue of the batch waits for CPU:2, so the run waits for it on the host.
   The source is an input, whose address the run enters in the address table, or
   storage whose address the link writes, which only the storage the link
   reaches puts among the buffers the run touches. *)
let waits_for_a_device_without_queues ~as_input () =
  let nd = Null_device.device in
  let y = Ops.new_buffer (Single "CPU:2") 4 Float32
  and x = Ops.new_buffer (Single "CPU:1") 4 Float32 in
  let zeros d = Run.buffer (nd d) Float32 (floats [| 0.; 0.; 0.; 0. |]) in
  let filler = zeros "CPU:2" and result = zeros "CPU:1" in
  let src =
    if as_input then
      Ops.param ~shape:[ Int 4 ] ~device:(Single "CPU:2") 0 Float32
    else y
  in
  let devices n = if n = "CPU:2" then devices n else on_null n in
  let compiled =
    Hcq2.compile_linear
      ~devices:(fun n -> (devices n).compiler)
      (Ops.v Op.Linear ~src:[ Ops.store_call x src ])
  in
  let bound =
    (x, [ result ]) :: (if as_input then [] else [ (y, [ filler ]) ])
  in
  let copies = Engine.link ~devices ~bound compiled in
  let filling = fill_later (nd "CPU:2") filler 7. in
  Engine.run copies (if as_input then [| [ filler ] |] else [||]);
  Null_device.synchronize ();
  Domain.join filling;
  equal values (floats [| 7.; 7.; 7.; 7. |]) (Run.values Float32 result)

(* A batch on CPU:4's queue that copies [n] floats of a host buffer into CPU:4's
   memory and CPU:4's [y] back into it, run twice, the host rewriting the buffer
   in between: CPU:4 maps whole pages, so the buffer, of less than 64 KiB, is
   staged. As an input, each run takes a buffer of its own. *)
let stages_host_memory ~as_input n =
  let nd = Null_device.device in
  let from k = floats (Array.init n (fun i -> Float.of_int (k + i))) in
  let h =
    if as_input then Ops.param ~shape:[ Int n ] ~device:(Single "CPU") 0 Float32
    else Ops.new_buffer (Single "CPU") n Float32
  in
  let x = Ops.new_buffer (Single "CPU:4") n Float32
  and y = Ops.new_buffer (Single "CPU:4") n Float32 in
  let hb = Run.buffer host Float32 (from 1)
  and xb = Run.buffer (nd "CPU:4") Float32 (from 0)
  and yb = Run.buffer (nd "CPU:4") Float32 (from 100) in
  let bound =
    (x, [ xb ]) :: (y, [ yb ]) :: (if as_input then [] else [ (h, [ hb ]) ])
  in
  let s = link_calls ~bound [ Ops.store_call x h; Ops.store_call h y ] in
  let run hb = Engine.run s (if as_input then [| [ hb ] |] else [||]) in
  run hb;
  equal values ~msg:"read" (from 1) (Run.values Float32 xb);
  equal values ~msg:"written" (from 100) (Run.values Float32 hb);
  let next =
    if as_input then Run.buffer host Float32 (from 50)
    else begin
      Buffer.copy ~src:(Run.buffer host Float32 (from 50)) ~dst:hb;
      hb
    end
  in
  run next;
  equal values ~msg:"read again" (from 50) (Run.values Float32 xb);
  equal values ~msg:"written again" (from 100) (Run.values Float32 next)

(* A batch that only reads a staged host buffer, on a CPU:4 queue that starts
   late: [run] returns before the work completes, and copies nothing back. A
   batch that writes one returns once it completed. *)
let waits_only_for_written_stages () =
  let d = Null_device.device "CPU:4" in
  let h = Ops.new_buffer (Single "CPU") 4 Float32
  and x = Ops.new_buffer (Single "CPU:4") 4 Float32 in
  let hb = Run.buffer host Float32 a
  and xb = Run.buffer d Float32 (floats [| 0.; 0.; 0.; 0. |]) in
  let bound = [ (h, [ hb ]); (x, [ xb ]) ] in
  let reads = link_calls ~bound [ Ops.store_call x h ]
  and writes = link_calls ~bound [ Ops.store_call h x ] in
  let pending () = Nx_device.signaled d < Nx_device.submitted d in
  Null_device.with_latency 0.05 (fun () -> Engine.run reads [||]);
  is_true ~msg:"a read returns at once" (pending ());
  Null_device.with_latency 0.05 (fun () -> Engine.run writes [||]);
  is_false ~msg:"a write returns once done" (pending ());
  equal values a (Run.values Float32 xb)

(* A batch on CPU:4's late queue copies 64 KiB of a host buffer, which CPU:4
   borrows through a mapping, into CPU:4's memory; the linked schedule is
   dropped and collected before the work runs, and the copy still lands. *)
let borrows_outlive_the_link ~as_input () =
  let n = 16384 in
  let d = Null_device.device "CPU:4" in
  let data = floats (Array.init n Float.of_int) in
  let hb = Run.buffer host Float32 data
  and xb = Run.buffer d Float32 (Array.make n (`Float 0.)) in
  let run () =
    let h =
      if as_input then
        Ops.param ~shape:[ Int n ] ~device:(Single "CPU") 0 Float32
      else Ops.new_buffer (Single "CPU") n Float32
    and x = Ops.new_buffer (Single "CPU:4") n Float32 in
    let bound = (x, [ xb ]) :: (if as_input then [] else [ (h, [ hb ]) ]) in
    let s = link_calls ~bound [ Ops.store_call x h ] in
    Null_device.with_latency 0.05 (fun () ->
        Engine.run s (if as_input then [| [ hb ] |] else [||]))
  in
  run ();
  Gc.full_major ();
  Null_device.synchronize ();
  equal values data (Run.values Float32 xb)

(* A slow copy from CPU:1 into storage of the host that the link allocates,
   enqueued on CPU:1's queue with the storage's address folded in at link, then
   a host kernel that adds one to it. The run records the copy as pending on the
   host, so the host kernel runs once it landed. *)
let leaves_its_work_pending_on_the_host () =
  let x = Ops.new_buffer (Single "CPU:1") 4 Float32
  and t = Ops.new_buffer (Single "CPU") 4 Float32
  and out = Ops.new_buffer (Single "CPU") 4 Float32 in
  let result = Run.buffer host Float32 (floats [| 0.; 0.; 0.; 0. |]) in
  let s =
    link_calls
      ~bound:
        [
          (x, [ Run.buffer (Null_device.device "CPU:1") Float32 a ]);
          (out, [ result ]);
        ]
      [ Ops.store_call t x; Ops.call add_one [ out; t ] ]
  in
  Null_device.with_latency 0.02 (fun () -> Engine.run s [||]);
  equal values (floats [| 2.; 3.; 4.; 5. |]) (Run.values Float32 result)

(* A kernel on CPU:1 and CPU:2 reads the last four of the eight floats each
   device holds of a sharded parameter: its address, entered in the address
   table on each run, is the shard's plus the view's offset. *)
let reads_a_view_of_each_shard () =
  let nd = Null_device.device in
  let lanes = Ops.Multi [ "CPU:1"; "CPU:2" ] in
  let param = Ops.param ~shape:[ Int 8 ] ~device:lanes 0 Float32 in
  let out = Ops.new_buffer lanes 4 Float32 in
  let results =
    List.map
      (fun d -> Run.buffer (nd d) Float32 (floats [| 0.; 0.; 0.; 0. |]))
      [ "CPU:1"; "CPU:2" ]
  in
  let s =
    link_calls
      ~bound:[ (out, results) ]
      [ Ops.call add_one [ out; Ops.shrink param [ Some (Int 4, Int 8) ] ] ]
  in
  let shard k = floats (Array.init 8 (fun i -> Float.of_int ((10 * k) + i))) in
  Engine.run s
    [|
      [
        Run.buffer (nd "CPU:1") Float32 (shard 1);
        Run.buffer (nd "CPU:2") Float32 (shard 2);
      ];
    |];
  equal (list values)
    [ floats [| 15.; 16.; 17.; 18. |]; floats [| 25.; 26.; 27.; 28. |] ]
    (List.map (Run.values Float32) results)

(* Under a profile, a kernel of a batch is a span of its device's compute lane
   and a copy one of its copy lane. *)
let spans_on_lanes () =
  let y, filled = fill "CPU:1" 7. in
  let z = Ops.new_buffer (Single "CPU:2") 4 Float32 in
  let bound =
    [
      (y, [ Run.buffer (Null_device.device "CPU:1") Float32 a ]);
      (z, [ Run.buffer (Null_device.device "CPU:2") Float32 a ]);
    ]
  in
  let s = link_calls ~profile:true ~bound [ filled; Ops.store_call z y ] in
  let p = Nx_device.Profile.start () in
  let events =
    Fun.protect
      ~finally:(fun () ->
        if Nx_device.Profile.enabled () then ignore (Nx_device.Profile.stop p))
      (fun () ->
        Engine.run s [||];
        Null_device.synchronize ();
        Nx_device.Profile.stop p)
  in
  let lanes =
    List.filter_map
      (function
        | Nx_device.Profile.Span sp
          when Nx_device.equal sp.device (Null_device.device "CPU:1") ->
            Some (sp.lane, String.starts_with ~prefix:"fill" sp.name)
        | _ -> None)
      events
  in
  equal
    (slist (pair string bool) compare)
    [ ("compute", true); ("copy", false) ]
    lanes

(* A host program of CPU:1 calls a C function of the host, CPU, which gives no
   word for its address: CPU:1, the batch's device, gives it. *)
let calls_a_function_of_the_host () =
  let d = "CPU:1" in
  let out =
    Ops.placeholder ~device:(Single d) ~volatile:true ~tag:(String "result")
      [ 1 ] Int32
  in
  let ffs =
    Hcq2.ccall ~host:"CPU" ~lib:"libc" ~ret:Int32 "ffs"
      [ Ops.int ~dtype:Int32 0x10 ]
  in
  let result = Nx_device.Buffer.create (Null_device.device d) Int32 1 in
  let devices = with_tag "result" result in
  let compiled =
    host_program ~devices d [ Ops.store (Ops.index out [ Ops.int 0 ]) ffs ]
  in
  Engine.run (Engine.link ~devices compiled) [||];
  Null_device.synchronize ();
  equal values [| `Int (Z.of_int 5) |] (Run.values Int32 result)

(* A vendor's storage of fewer bytes than its placeholder is refused at link,
   before any run could write past it. *)
let refuses_short_storage () =
  let d = "CPU:1" in
  let out =
    Ops.placeholder ~device:(Single d) ~volatile:true ~tag:(String "result")
      [ 2 ] Int32
  in
  let short = Nx_device.Buffer.create (Null_device.device d) Int32 1 in
  let devices = with_tag "result" short in
  let compiled =
    host_program ~devices d
      [ Ops.store (Ops.index out [ Ops.int 1 ]) (Ops.int ~dtype:Int32 7) ]
  in
  raises_match (Exn.invalid_arg ~substring:"holds 4 bytes, not 8") (fun () ->
      ignore (Engine.link ~devices compiled))

(* Each run of a batch on a device runs the device's submitting hook once. *)
let runs_the_submitting_hook () =
  let hooked = ref 0 in
  let devices n =
    if n = "CPU:1" then
      { (on_null n) with submitting = (fun () -> incr hooked) }
    else on_null n
  in
  let s, vars, big = parameterized ~devices "copy" in
  let slots = slots (storage_of ~devices big) in
  for _ = 1 to 3 do
    Engine.run ~vars s slots
  done;
  Null_device.synchronize ();
  equal int 3 !hooked

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

(* At [DEBUG=2], the recorded copy runs its copy into CPU:1 and its kernel on
   the host devices, each a line timed on the host clock. *)
let reports_host_calls () =
  let big = program "copy" in
  Helpers.context
    [ B (Helpers.debug, 2) ]
    (fun () ->
      let s, vars, storage = linked big in
      Engine.run ~vars s (slots storage));
  let lines = reported () in
  equal int ~msg:"a copy and a kernel" 2 (List.length lines);
  counts_up lines;
  List.iter
    (fun words ->
      equal string ~msg:"device" "CPU:1" (List.nth words 1);
      let t = seconds_of words in
      greater float_exact ~msg:"time" ~than:0. t;
      less float_exact ~msg:"time" ~than:1. t)
    lines

(* At [DEBUG=2], a run of three kernels on CPU:1, whose queue starts 20 ms after
   its submission, prints a line for each, timed by the kernel's stamps: the
   time leaves the latency out. *)
let reports_each_kernel () =
  let fills = List.init 3 (fun k -> fill "CPU:1" (Float.of_int k)) in
  let bound =
    List.map
      (fun (y, _) -> (y, [ Run.buffer (Null_device.device "CPU:1") Float32 a ]))
      fills
  in
  Helpers.context
    [ B (Helpers.debug, 2) ]
    (fun () ->
      let s = link_calls ~bound (List.map snd fills) in
      Null_device.with_latency 0.02 (fun () -> Engine.run s [||]));
  let lines = reported () in
  equal int ~msg:"one line per kernel" 3 (List.length lines);
  counts_up lines;
  List.iter
    (fun words ->
      equal string ~msg:"device" "CPU:1" (List.nth words 1);
      let t = seconds_of words in
      at_least float_exact ~msg:"time" ~than:0. t;
      less float_exact ~msg:"time" ~than:0.02 t)
    lines

(* Staging

   NULL devices whose queues address the host's memory alone: a copy between two
   of them stages through the host's staging memory. *)

let apart = lazy (Null_device.devices ~reaches:(fun d -> d = "CPU") ())
let on_apart name = (Lazy.force apart) name

(* A program that copies [xs] from a buffer of [src] into one of [dst], staged,
   and the destination buffer. *)
let staged_copy src dst xs =
  let x = Ops.new_buffer (Single src) 4 Float32
  and y = Ops.new_buffer (Single dst) 4 Float32 in
  let compiled =
    Hcq2.compile_linear
      ~devices:(fun n -> (on_apart n).compiler)
      (Ops.v Op.Linear ~src:[ Ops.store_call y x ])
  in
  let into =
    Run.buffer (Null_device.device dst) Float32 (floats [| 0.; 0.; 0.; 0. |])
  in
  let bound =
    [ (x, [ Run.buffer (Null_device.device src) Float32 xs ]); (y, [ into ]) ]
  in
  (Engine.link ~devices:on_apart ~bound compiled, into)

(* Linking a second program that stages, while the first is reachable, allocates
   no second staging memory on the host. *)
let shares_the_staging_memory () =
  let host_allocated () = Nx_device.Stats.allocated (Nx_device.stats host) in
  let first, _ = staged_copy "CPU:1" "CPU:2" a in
  let before = host_allocated () in
  let second, _ = staged_copy "CPU:2" "CPU:3" a in
  less int ~msg:"bytes the second link allocates on the host" ~than:(1 lsl 20)
    (host_allocated () - before);
  ignore (Sys.opaque_identity (first, second))

(* Program A stages a copy from CPU:1 to CPU:2 on queues that start 50 ms late;
   program B then stages one from CPU:3 to CPU:1 through the same memory. B's
   run waits for A's staged work before it submits, so neither copy reads the
   other's bytes. *)
let staged_runs_take_turns () =
  let xs = floats [| 100.; 101.; 102.; 103. |] in
  let first, into_first = staged_copy "CPU:1" "CPU:2" a
  and second, into_second = staged_copy "CPU:3" "CPU:1" xs in
  Null_device.with_latency 0.05 (fun () -> Engine.run first [||]);
  Engine.run second [||];
  Null_device.synchronize ();
  equal values ~msg:"first" a (Run.values Float32 into_first);
  equal values ~msg:"second" xs (Run.values Float32 into_second)

(* Two staging programs run from two domains, ten times each. *)
let staged_runs_from_two_domains () =
  let xs = floats [| 100.; 101.; 102.; 103. |] in
  let first, into_first = staged_copy "CPU:1" "CPU:2" a
  and second, into_second = staged_copy "CPU:3" "CPU:1" xs in
  let runs s () =
    for _ = 1 to 10 do
      Engine.run s [||]
    done
  in
  let d = Domain.spawn (runs first) in
  runs second ();
  Domain.join d;
  Null_device.synchronize ();
  equal values ~msg:"first" a (Run.values Float32 into_first);
  equal values ~msg:"second" xs (Run.values Float32 into_second)

(* The minor words a run of a batch of one kernel on CPU:1 of [devices]
   allocates on the calling domain, the run before it having linked and loaded
   everything. *)
let batch_run_words ?(devices = on_null) () =
  let y, filled = fill "CPU:1" 7. in
  let compiled =
    Hcq2.compile_linear
      ~devices:(fun n -> (devices n).compiler)
      (Ops.v Op.Linear ~src:[ filled ])
  in
  let d = (devices "CPU:1").device in
  let s =
    Engine.link ~devices ~bound:[ (y, [ Run.buffer d Float32 a ]) ] compiled
  in
  Engine.run s [||];
  Nx_device.synchronize d;
  let before = Gc.minor_words () in
  Engine.run s [||];
  let words = Gc.minor_words () -. before in
  Nx_device.synchronize d;
  Float.to_int words

let refuses_an_unknown_library () =
  let y, filled = fill "CPU:1" 7. in
  let devices n = { (on_null n) with placeholder = (fun _ -> None) } in
  let compiled =
    Hcq2.compile_linear
      ~devices:(fun n -> (devices n).compiler)
      (Ops.v Op.Linear ~src:[ filled ])
  in
  let bound = [ (y, [ Run.buffer (Null_device.device "CPU:1") Float32 a ]) ] in
  raises_match Exn.invalid_arg (fun () -> Engine.link ~devices ~bound compiled)

(* A device of the host's memory whose mapped and pinned memories are told
   apart: each records the sizes it allocates. *)
let mapped_allocs = ref []
and pinned_allocs = ref []

let mapped_device =
  lazy
    (let recording log (a : Nx_device.Driver.allocator) =
       {
         a with
         alloc =
           (fun n ->
             log := n :: !log;
             a.alloc n);
       }
     in
     let memory = Nx_device.Driver.host_memory in
     let queue ~timeline:_ =
       {
         Nx_device.Driver.copy = (fun ~dst:_ ~src:_ _ ~signal:_ -> never ());
         transfer = (fun _ -> None);
         stamp = (fun ~slot:_ ~signal:_ -> never ());
         clock = Host_clock;
       }
     in
     Nx_device.Driver.device ~name:"MAPPED" ~arch:"test" ~budget:max_int
       (Device_local
          {
            memory;
            host_memory = recording pinned_allocs memory;
            mapped = Some (recording mapped_allocs memory);
            mapping = Identity;
            queue;
          }))

(* A batch's kernel arguments are mapped memory, and its command buffer and
   volatile words pinned memory. *)
let places_in_mapped_memory () =
  let d = Lazy.force mapped_device in
  let devices n =
    if n = "CPU:1" then { (on_null n) with device = d } else on_null n
  in
  let y, filled = fill "CPU:1" 7. in
  let compiled =
    Hcq2.compile_linear
      ~devices:(fun n -> (devices n).compiler)
      (Ops.v Op.Linear ~src:[ filled ])
  in
  let placeholders =
    List.filter
      (fun u ->
        Ops.op u = Op.Param
        && Option.is_some (Ops.tag u)
        && (match Ops.device u with
          | Some (Single "CPU:1" | Multi [ "CPU:1" ]) -> true
          | _ -> false)
        (* The engine allocates those the device does not give. *)
        && Option.is_none ((devices "CPU:1").placeholder u))
      (Ops.toposort ~enter_calls:true compiled)
  in
  let bytes u = Ops.max_numel u * Dtype.itemsize (Ops.dtype u) in
  let is_mapped u =
    match (Ops.arg u, Ops.tag u) with
    | Param p, Some (String t) ->
        (not p.volatile)
        && (not (String.starts_with ~prefix:"cmdbuf" t))
        && t <> "timeline"
    | _ -> false
  in
  mapped_allocs := [];
  pinned_allocs := [];
  ignore
    (Engine.link ~devices
       ~bound:[ (y, [ Nx_device.Buffer.create d Float32 4 ]) ]
       compiled);
  let sorted l = List.sort compare l in
  let expected = sorted (List.map bytes (List.filter is_mapped placeholders)) in
  is_true ~msg:"a placeholder is mapped" (expected <> []);
  equal ~msg:"mapped memory" (list int) expected (sorted !mapped_allocs);
  equal ~msg:"pinned memory" (list int)
    (sorted
       (List.map bytes
          (List.filter
             (fun u ->
               (not (is_mapped u)) && Ops.tag u <> Some (String "timeline"))
             placeholders)))
    (sorted !pinned_allocs)

let on_cpu1 storage = List.find (fun s -> placement s.arg = [ "CPU:1" ]) storage

let batches =
  group "batches"
    [
      group "recorded"
        (List.map computes_on_null [ "copy"; "shard_add"; "variable_offset" ]);
      test "each run of a batch signals its device's next value once"
        (signals_once_per_run ~devices:on_null);
      test
        "a batch's kernel arguments are mapped memory, and its command buffers \
         and volatile words pinned memory"
        places_in_mapped_memory;
      test
        "runs of one batched schedule from two domains each compute their own"
        (serialized ~devices:on_null "copy");
      test "a run of a batch allocates and loads nothing"
        (allocates_nothing ~devices:on_null "copy");
      slow "a batch waits for the work of a device outside it"
        waits_for_another_device;
      test "a run waits for a device without queues of an input"
        (waits_for_a_device_without_queues ~as_input:true);
      test "a run waits for a device without queues of storage"
        (waits_for_a_device_without_queues ~as_input:false);
      test "a host kernel runs once the copy that feeds it landed"
        leaves_its_work_pending_on_the_host;
      cases ~name:string_of_int
        "a host buffer the device cannot borrow is staged, as storage"
        [ 3; 64; 320 ]
        (stages_host_memory ~as_input:false);
      cases ~name:string_of_int
        "a host buffer the device cannot borrow is staged, as an input"
        [ 3; 64; 320 ]
        (stages_host_memory ~as_input:true);
      test "a run waits for its batch only when it writes a staged buffer"
        waits_only_for_written_stages;
      test "a batch's borrows outlive its schedule until its work completes"
        (borrows_outlive_the_link ~as_input:false);
      test "a run's borrows outlive its schedule until its work completes"
        (borrows_outlive_the_link ~as_input:true);
      test "a batch reads a view of each shard of a parameter"
        reads_a_view_of_each_shard;
      test "a kernel is a span of its compute lane and a copy of its copy lane"
        spans_on_lanes;
      test "a host program calls a C function through the batch's device"
        calls_a_function_of_the_host;
      test "a vendor's storage shorter than its placeholder is refused at link"
        refuses_short_storage;
      test "each run of a batch runs its device's submitting hook once"
        runs_the_submitting_hook;
      test "at DEBUG=2, a run prints one line per kernel, timed by its stamps"
        reports_each_kernel;
      test "at DEBUG=2, a host copy and a host kernel print a timed line each"
        reports_host_calls;
      test "linked schedules that stage share the host's staging memory"
        shares_the_staging_memory;
      test "staged runs of two programs on other devices take turns"
        staged_runs_take_turns;
      test "staged runs of two programs from two domains each copy their own"
        staged_runs_from_two_domains;
      test "a run of a batch of one kernel allocates at most 400 minor words"
        (fun () -> at_most int ~than:400 (batch_run_words ()));
      test "link refuses a C function of a library it does not know"
        refuses_an_unknown_library;
      run_refuses ~devices:on_null "a batch's parameter it binds no buffers"
        "copy" (fun _ -> [||]);
      run_refuses ~devices:on_null
        "a batch's parameter bound to buffers of another device" "copy"
        (fun storage ->
          let s = on_cpu1 storage in
          rebound s.arg.slot
            (fun _ ->
              [
                Run.buffer
                  (Null_device.device "CPU:2")
                  Float32
                  (Array.make 16 (`Float 0.));
              ])
            storage);
      run_refuses ~devices:on_null
        "a batch's parameter bound to fewer bytes than it holds" "copy"
        (fun storage ->
          let s = on_cpu1 storage in
          rebound s.arg.slot
            (fun _ ->
              [
                Run.buffer
                  (Null_device.device "CPU:1")
                  Float32
                  (Array.make 15 (`Float 0.));
              ])
            storage);
      test "a program's run on a device with queues takes a positive time"
        (fun () ->
          let t =
            Engine.measure
              ~vars:[ ("n", 3) ]
              ~devices:on_null "CPU:1" (Lazy.force long_axpy)
          in
          greater float_exact ~than:0. t;
          less float_exact ~than:1. t);
    ]

(* Metal

   The recorded programs copy from the host into CPU:1, which is the Metal
   device here, and compute on it: a batch of Metal's queues, whose host
   programs the host runs. Metal signals completion in its own way, through a
   shared event. *)

let on_metal name =
  match Metal.device with
  | None -> skip ~reason:"no Metal device" ()
  | Some m -> Engine.device [ ("CPU", host); ("CPU:1", m) ] name

let metal_names = [ "CPU"; "CPU:1" ]

let computes_on_metal name =
  slow (name ^ " writes what its tensors compute") (fun () ->
      let big = program name in
      let s, vars, storage = linked ~devices:on_metal big in
      Engine.run ~vars s (slots storage);
      equal slot_values (expected ~vars big storage) (by_slot storage contents))

(* A kernel of Metal, [out = a * n + b] over [2^18] floats, as [measure] runs it
   on scratch buffers. *)
let metal_axpy () =
  let t = (on_metal "CPU:1").compiler.target in
  match Device.renderer ~arch:t.arch t.device with
  | Error why -> fail why
  | Ok r -> Codegen.to_program (kernel ~name:"metal_axpy" ~size:(1 lsl 18) ()) r

let metal =
  group "Metal"
    [
      group "recorded"
        (List.map computes_on_metal [ "copy"; "copy_one"; "copy_view" ]);
      slow "a batch whose names omit the host runs, the engine naming it"
        (fun () ->
          let on name =
            match Metal.device with
            | None -> skip ~reason:"no Metal device" ()
            | Some m -> Engine.device [ ("CPU:1", m) ] name
          in
          let big = program "copy" in
          let s, vars, storage = linked ~devices:on big in
          Engine.run ~vars s (slots storage);
          equal slot_values
            (expected ~vars big storage)
            (by_slot storage contents));
      cases ~tags:[ "slow" ] ~name:Fun.id
        "a batch runs on the buffers each run binds to its parameters"
        [ "copy"; "copy_view" ] (fun name ->
          runs_on_its_slots ~devices:on_metal name ());
      slow "each run of a batch signals the device's next value once"
        (signals_once_per_run ~devices:on_metal);
      slow
        "runs of one batched schedule from two domains each compute their own"
        (serialized ~devices:on_metal "copy");
      slow "a run of a batch allocates and loads nothing"
        (allocates_nothing ~devices:on_metal ~names:metal_names "copy");
      slow "a run of a batch of one kernel allocates at most 400 minor words"
        (fun () -> at_most int ~than:400 (batch_run_words ~devices:on_metal ()));
      cases ~tags:[ "slow" ] ~name:string_of_int
        "a store through a padded view writes the row within the source, and \
         nothing outside it (D69)"
        [ 0; 7; -1; 8 ] (fun at ->
          stores_through_a_pad ~devices:on_metal "CPU:1" at ());
      cases ~tags:[ "slow" ] ~name:string_of_int
        "a read of a padded node stored through reads its fill in the padding \
         (D69)"
        [ 3; -1 ] (fun at ->
          stores_through_a_pad ~devices:on_metal ~fill:7. ~read:true "CPU:1" at
            ());
      slow "a program's run on Metal takes a positive time, under a second"
        (fun () ->
          let t =
            Engine.measure
              ~vars:[ ("n", 3) ]
              ~devices:on_metal "CPU:1" (metal_axpy ())
          in
          greater float_exact ~than:0. t;
          less float_exact ~than:1. t);
    ]

let () =
  exit
    (run "Tolk_next_engine"
       [
         targets;
         describing;
         programs;
         schedules;
         padded_stores;
         group "a store through a shrink" [ valid_slot_store ];
         refusals;
         runs;
         measures;
         batches;
         metal;
       ])
