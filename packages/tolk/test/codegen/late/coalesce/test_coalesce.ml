open Windtrap
open Tolk

let cpu = Result.get_ok (Helpers.Target.of_string "CPU:CLANG:x86_64,x86-64")

(* A renderer of the CPU device whose vector registers are [count] of [bytes],
   as Clang's are for x86-64's SSE, AVX2 and NEON. *)
let host count bytes =
  Renderer.v ~registers:(fun () -> Some { Renderer.count; bytes }) cpu

let vector = host 16 16

(* The CPU device's renderer, whose vector registers are not known. *)
let cpu_unknown = Renderer.v cpu
let scalar = Renderer.v ~supports_float4:false cpu

(* A GPU's vectors are aligned to their size, so its merges read where storage
   starts; the CPU device's are aligned to their element. *)
let gpu =
  Renderer.v (Result.get_ok (Helpers.Target.of_string "CUDA:CUDA:sm_89"))

(* A GPU that writes a vector of bytes as one integer word ([byte_words]). *)
let gpu_words =
  Renderer.v ~byte_words:true
    (Result.get_ok (Helpers.Target.of_string "CUDA:CUDA:sm_89"))

(* A GPU that writes vectors of eight 16-bit floats ([half8]). *)
let gpu8 =
  Renderer.v ~half8:true
    (Result.get_ok (Helpers.Target.of_string "CUDA:CUDA:sm_89"))

let coalesce ?(renderer = vector) sink =
  Coalesce.memory_coalescing sink renderer

let simplify u =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
    (After_sources Coalesce.indexing_simplify)

(* The cases of an input golden are its sink's sources, in order. *)
let case file cell =
  List.nth (Ops.src (Golden.sink file)) (int_of_string (cell "src"))

(* Accesses *)

let buf = Shape.param ~shape:[ Int 64 ] 0 Float32
let at buf i = Ops.index buf [ i ]
let load_at buf i = Ops.load (at buf (Ops.int i)) []
let loads buf offsets = Ops.sink (List.map (load_at buf) offsets)

let value ?(dtype = Dtype.Float32) k =
  let lo, hi =
    if Dtype.is_float dtype then (`Float (-1.), `Float 1.)
    else (`Int Bigint.minus_one, `Int Bigint.one)
  in
  Ops.variable ~dtype (Printf.sprintf "v%d" k) lo hi

let store_at buf i =
  Ops.store (at buf (Ops.int i)) (value ~dtype:(Ops.dtype buf) i)

let stores buf offsets = Ops.sink (List.map (store_at buf) offsets)
let int_of u = Bigint.to_int (Shape.to_z u)

(* The accesses of [sink], each as the constant offset it starts at and the
   number of elements it reads or writes. *)
let accesses op sink =
  let access u =
    let p = Ops.nth u 0 in
    let offset = int_of (Ops.get_idx (Ops.nth p 1)) in
    match Ops.op p with
    | Op.Shrink -> (offset, int_of (Ops.nth p 2))
    | _ -> (offset, 1)
  in
  List.filter_map
    (fun u -> if Ops.op u = op then Some (access u) else None)
    (Ops.toposort ~calls:Enter sink)

let runs = slist (pair int int) compare

(* The counts of loads and of stores of [width] elements of [dtype]. *)
let vectors ?(dtype = Dtype.Float32) ?(width = 4) sink =
  let wide u = Shape.shape u = [ Ops.Int width ] in
  let count op f =
    List.length
      (List.filter
         (fun u -> Ops.op u = op && f u)
         (Ops.toposort ~calls:Enter sink))
  in
  ( count Op.Load (fun u -> Dtype.equal (Ops.dtype u) dtype && wide u),
    count Op.Store (fun u ->
        let v = Ops.nth u 1 in
        Dtype.equal (Ops.dtype v) dtype && wide v) )

(* Kernels recorded by gen/codegen/late/coalesce.py: the sinks memory_coalescing
   receives when a kernel compiles, for Clang, and for CUDA where the kernel
   needs workgroups. *)

let kernels =
  [
    "basic";
    "multidim";
    "unaligned_load";
    "multidim_unaligned_load";
    "sometimes_unaligned";
    "multidim_sometimes_unaligned";
    "expand";
    "heterogeneous";
    "aligned_variable";
    "unaligned_variable";
    "load_dedup";
    "grouped_store";
    "half";
    "half8";
    "int";
    "uint";
    "fp8e4m3";
    "fp8e5m2fnuz";
    "char";
    "long";
    "double";
    "bfloat16";
    "pad";
    "sum";
    "cuda";
  ]

let kernel name = Golden.sink (name ^ ".golden")

let compiled name =
  Golden.graph (name ^ "_coalesced.golden") (fun () -> coalesce (kernel name))

(* Hand-built accesses *)

let hand_built file renderer =
  Golden.cases "accesses.golden" (fun cell ->
      equal Uops.uop (case file cell)
        (coalesce ~renderer (case "accesses_input.golden" cell)))

(* Loads of [buf] at [x * 4 + k] for each [k] of [ks], or at [k] alone, in
   integers of [dtype]: the weak integer's constants are bare, any other's are
   casts of them. *)
let typed_loads ?dtype ~base ks =
  let int k = Ops.int ?dtype k in
  let x = Ops.variable ?dtype "x" (`Int Bigint.zero) (`Int (Bigint.of_int 8)) in
  let at k = if base then Ops.O.((x * int 4) + int k) else int k in
  Ops.sink (List.map (fun k -> Ops.load (Ops.index buf [ at k ]) []) ks)

let lengths =
  group "lengths"
    [
      cases "loads at typed offsets merge as at weak ones, for"
        ~id:(fun (dt, base) ->
          Format.asprintf "%a, %s" Dtype.pp dt
            (if base then "a variable plus constants" else "constants"))
        (List.concat_map
           (fun dt -> [ (dt, false); (dt, true) ])
           Dtype.[ Int32; Int64; Uint32 ])
        (fun (dtype, base) ->
          let ks = [ 0; 1; 2; 3; 5; 6 ] in
          let weak = vectors (coalesce (typed_loads ~base ks)) in
          equal ~msg:"weak" (pair int int) (1, 0) weak;
          equal (pair int int) weak
            (vectors (coalesce (typed_loads ~dtype ~base ks))));
      test
        "without vectors of eight halves or known registers, eight halves are \
         two loads of four" (fun () ->
          let half = Shape.param ~shape:[ Int 64 ] 2 Float16 in
          equal runs
            [ (0, 4); (4, 4) ]
            (accesses Op.Load
               (coalesce ~renderer:cpu_unknown
                  (loads half (List.init 8 Fun.id)))));
      cases "eight consecutive stores fill registers of"
        ~id:(fun (name, _, dtype, _) ->
          Format.asprintf "%s, %a" name Dtype.pp dtype)
        [
          ("16 bytes", host 16 16, Dtype.Float32, [ (0, 4); (4, 4) ]);
          ("32 bytes", host 16 32, Dtype.Float32, [ (0, 8) ]);
          ( "16 bytes",
            host 32 16,
            Dtype.Float64,
            [ (0, 2); (2, 2); (4, 2); (6, 2) ] );
          ("32 bytes", host 16 32, Dtype.Float64, [ (0, 4); (4, 4) ]);
          ("32 bytes", host 16 32, Dtype.Float16, [ (0, 8) ]);
          ("32 bytes", host 16 32, Dtype.Int32, [ (0, 8) ]);
          ("32 bytes", host 16 32, Dtype.Uint8, [ (0, 8) ]);
        ]
        (fun (_, renderer, dtype, expected) ->
          (* Each store's value is a load of another buffer: a packed lane. *)
          let b = Shape.param ~shape:[ Int 64 ] 0 dtype
          and src = Shape.param ~shape:[ Int 64 ] 1 dtype in
          let copy i =
            Ops.store
              (at b (Ops.int i))
              (Ops.load (at src (Ops.int (2 * i))) [])
          in
          equal runs expected
            (accesses Op.Store
               (coalesce ~renderer (Ops.sink (List.init 8 copy)))));
      cases "on a GPU, eight consecutive loads are two loads of four, of"
        ~id:(Format.asprintf "%a" Dtype.pp) [ Dtype.Float16; Dtype.Bfloat16 ]
        (fun dtype ->
          let b = Shape.param ~shape:[ Int 64 ] 0 dtype in
          equal runs
            [ (0, 4); (4, 4) ]
            (accesses Op.Load
               (coalesce ~renderer:gpu (loads b (List.init 8 Fun.id)))));
      cases "on a GPU that writes bytes as words, eight consecutive"
        ~id:(fun (name, _, _) -> name)
        [
          ("byte loads are two loads of four", Op.Load, loads);
          ("byte stores are two stores of four", Op.Store, stores);
        ]
        (fun (_, op, sink) ->
          let b = Shape.param ~shape:[ Int 64 ] 0 Dtype.Uint8 in
          equal runs
            [ (0, 4); (4, 4) ]
            (accesses op
               (coalesce ~renderer:gpu_words (sink b (List.init 8 Fun.id)))));
      test "on a GPU that writes bytes as words, int8 loads stay apart"
        (fun () ->
          let b = Shape.param ~shape:[ Int 64 ] 0 Dtype.Int8 in
          equal runs
            (List.init 8 (fun k -> (k, 1)))
            (accesses Op.Load
               (coalesce ~renderer:gpu_words (loads b (List.init 8 Fun.id)))));
      cases "thirty-two consecutive loads of bytes fill registers of"
        ~id:(fun (name, _, dtype, _) ->
          Format.asprintf "%s, %a" name Dtype.pp dtype)
        [
          ("16 bytes", host 16 16, Dtype.Uint8, [ (0, 16); (16, 16) ]);
          ("16 bytes", host 16 16, Dtype.Int8, [ (0, 16); (16, 16) ]);
          ("32 bytes", host 16 32, Dtype.Uint8, [ (0, 32) ]);
          ("32 bytes", host 16 32, Dtype.Int8, [ (0, 32) ]);
          ("64 bytes", host 32 64, Dtype.Uint8, [ (0, 32) ]);
        ]
        (fun (_, renderer, dtype, expected) ->
          let b = Shape.param ~shape:[ Int 64 ] 0 dtype in
          equal runs expected
            (accesses Op.Load
               (coalesce ~renderer (loads b (List.init 32 Fun.id)))));
      cases "thirty-two byte loads each read by a computation stay apart, on"
        ~id:(fun (name, _, dtype) ->
          Format.asprintf "%s, %a" name Dtype.pp dtype)
        [
          ("16 bytes", host 16 16, Dtype.Uint8);
          ("32 bytes", host 16 32, Dtype.Int8);
          ("64 bytes", host 32 64, Dtype.Uint8);
        ]
        (fun (_, renderer, dtype) ->
          let b = Shape.param ~shape:[ Int 64 ] 0 dtype in
          let read o = Ops.cast (load_at b o) Dtype.Int32 in
          let sink = Ops.sink (List.map read (List.init 32 Fun.id)) in
          equal runs
            (List.init 32 (fun o -> (o, 1)))
            (accesses Op.Load (coalesce ~renderer sink)));
      cases "without known registers, byte loads stay apart, on"
        ~id:(fun (name, _) -> name)
        [ ("the CPU device", cpu_unknown); ("a GPU", gpu) ]
        (fun (_, renderer) ->
          let b = Shape.param ~shape:[ Int 64 ] 0 Dtype.Uint8 in
          equal runs
            (List.init 8 (fun k -> (k, 1)))
            (accesses Op.Load
               (coalesce ~renderer (loads b (List.init 8 Fun.id)))));
      test "on the host, bfloat16 loads stay apart" (fun () ->
          let b = Shape.param ~shape:[ Int 64 ] 0 Bfloat16 in
          equal runs
            (List.init 8 (fun k -> (k, 1)))
            (accesses Op.Load (coalesce (loads b (List.init 8 Fun.id)))));
      cases "eight consecutive stores of lanes computed apart, on 32 bytes, are"
        ~id:(fun (dtype, _) -> Format.asprintf "%a" Dtype.pp dtype)
        [
          (Dtype.Float64, List.init 8 (fun i -> (i, 1)));
          (Dtype.Uint8, List.init 8 (fun i -> (i, 1)));
          (Dtype.Int8, List.init 8 (fun i -> (i, 1)));
          (Dtype.Float32, [ (0, 8) ]);
        ]
        (fun (dtype, expected) ->
          let b = Shape.param ~shape:[ Int 64 ] 0 dtype in
          equal runs expected
            (accesses Op.Store
               (coalesce ~renderer:(host 16 32) (stores b (List.init 8 Fun.id)))));
    ]

(* A buffer whose first element lies [phase] bytes past a multiple of [align]
   . *)
let phased ?(dtype = Dtype.Float32) ?align phase =
  Shape.param ~shape:[ Int 64 ] ~phase ?align 0 dtype

let phases =
  let coalesce = coalesce ~renderer:gpu in
  group "phase"
    [
      test
        "one float past a boundary, eight loads are of one, two, four and one"
        (fun () ->
          equal runs
            [ (0, 1); (1, 2); (3, 4); (7, 1) ]
            (accesses Op.Load
               (coalesce (loads (phased 4) (List.init 8 Fun.id)))));
      test "three floats past a boundary, a load of four starts at element 1"
        (fun () ->
          equal runs
            [ (0, 1); (1, 4); (5, 2); (7, 1) ]
            (accesses Op.Load
               (coalesce (loads (phased 12) (List.init 8 Fun.id)))));
      test "stores start where loads would" (fun () ->
          equal runs
            [ (0, 1); (1, 2); (3, 4); (7, 1) ]
            (accesses Op.Store
               (coalesce (stores (phased 4) (List.init 8 Fun.id)))));
      test
        "floats read through a bitcast of halves one half past a boundary are \
         not merged" (fun () ->
          let floats = Ops.bitcast (phased ~dtype:Float16 2) Float32 in
          equal runs
            (List.init 8 (fun k -> (k, 1)))
            (accesses Op.Load (coalesce (loads floats (List.init 8 Fun.id)))));
      test "two halves past a boundary, a load of four starts at element 2"
        (fun () ->
          equal runs
            [ (0, 2); (2, 4); (6, 2) ]
            (accesses Op.Load
               (coalesce (loads (phased ~dtype:Float16 4) (List.init 8 Fun.id)))));
      test "floats known to start on 4 bytes alone are not merged" (fun () ->
          equal runs
            (List.init 8 (fun k -> (k, 1)))
            (accesses Op.Load
               (coalesce (loads (phased ~align:4 0) (List.init 8 Fun.id)))));
      test "floats known to start on 8 bytes merge in twos" (fun () ->
          equal runs
            [ (0, 2); (2, 2); (4, 2); (6, 2) ]
            (accesses Op.Store
               (coalesce (stores (phased ~align:8 0) (List.init 8 Fun.id)))));
      test
        "halves three past 8 bytes merge no wider than 8 bytes, from element 1"
        (fun () ->
          equal runs
            [ (0, 1); (1, 4); (5, 2); (7, 1) ]
            (accesses Op.Load
               (coalesce
                  (loads
                     (phased ~dtype:Float16 ~align:8 6)
                     (List.init 8 Fun.id)))));
      test "on the CPU device, floats past a boundary merge as from one"
        (fun () ->
          equal runs
            [ (0, 4); (4, 4) ]
            (accesses Op.Load
               (Coalesce.memory_coalescing
                  (loads (phased 4) (List.init 8 Fun.id))
                  vector)));
    ]

(* Four consecutive loads in the body of a call. *)
let called_loads =
  let body =
    loads (Shape.param ~shape:[ Int 16 ] 0 Float32) (List.init 4 Fun.id)
  in
  Ops.sink [ Ops.call body [ Shape.param ~shape:[ Int 16 ] 5 Float32 ] ]

let left_alone =
  let same name sink =
    test name (fun () -> equal Uops.uop sink (coalesce sink))
  in
  let f4 = List.init 4 Fun.id in
  group "leaves as it is"
    [
      same "register memory where the renderer computes no vectors"
        (loads (Shape.placeholder ~slot:3 ~addrspace:Reg [ 16 ] Float32) f4);
      same "a volatile parameter"
        (loads (Shape.param ~volatile:true ~shape:[ Int 16 ] 4 Float32) f4);
      same "a volatile parameter viewed at another type"
        (loads
           (Ops.bitcast
              (Shape.param ~volatile:true ~shape:[ Int 4 ] 4 Uint32)
              Int32)
           f4);
      same "stores to a volatile parameter"
        (stores (Shape.param ~volatile:true ~shape:[ Int 16 ] 4 Float32) f4);
      test "a store stored twice is one store" (fun () ->
          let s = store_at buf 0 in
          equal runs
            [ (0, 1) ]
            (accesses Op.Store (coalesce (Ops.sink [ s; s ]))));
      same "the body of a call" called_loads;
    ]

let errors =
  let rejects name sink =
    test name (fun () -> raises_match Exn.invalid_arg (fun () -> coalesce sink))
  in
  let first = at buf (Ops.int 0) in
  group "errors"
    [
      rejects "a gated load"
        (Ops.sink
           [ Ops.load first [ Ops.float ~dtype:Float32 0.; Ops.bool true ] ]);
      rejects "a gated store"
        (Ops.sink [ Ops.store ~gate:(Ops.bool true) first (value 0) ]);
      rejects "a load of a parameter" (Ops.sink [ Ops.load buf [] ]);
      rejects "a load through a shrink"
        (Ops.sink
           [ Ops.load (Ops.v ~src:[ buf; Ops.int 0; Ops.int 4 ] Op.Shrink) [] ]);
      rejects "a load through an index of two indices"
        (Ops.sink [ Ops.load (Ops.index buf [ Ops.int 0; Ops.int 1 ]) [] ]);
      rejects "two stores to one element"
        (Ops.sink [ Ops.store first (value 0); Ops.store first (value 1) ]);
      rejects "two stores to one element of a run"
        (Ops.sink
           [
             store_at buf 0;
             store_at buf 1;
             Ops.store (at buf (Ops.int 1)) (value 2);
           ]);
      test "two stores to one element, without vector accesses" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              coalesce ~renderer:scalar
                (Ops.sink
                   [ Ops.store first (value 0); Ops.store first (value 1) ])));
    ]

(* Stores that wait on stores: a merge of accesses one of which waits on a store
   the other comes before would close a cycle. Stores into registers order
   nothing outside the thread; stores into local memory order global accesses,
   and global stores order local ones. *)
let stores_in space sink =
  List.length
    (List.filter
       (fun u ->
         Ops.op u = Op.Store && Ops.addrspace (Ops.nth u 0) = Some space)
       (Ops.toposort ~calls:Skip (coalesce sink)))

let dependencies =
  let i32 n = Ops.int ~dtype:Int32 n in
  let int_param n = Shape.param ~shape:[ Int n ] 0 Int32 in
  let scratch space n =
    Shape.placeholder ~slot:1 ~addrspace:space [ n ] Int32
  in
  let store b i v = Ops.store (at b (Ops.int i)) v in
  group "stores waiting on stores"
    [
      test "global stores after register stores merge" (fun () ->
          let buf = int_param 2 and reg = scratch Reg 2 in
          let lane i =
            store buf i (Ops.after (i32 i) [ store reg i (i32 i) ])
          in
          equal int 1 (stores_in Global (Ops.sink [ lane 0; lane 1 ])));
      test "global stores after local stores merge" (fun () ->
          let buf = int_param 4 and local = scratch Local 4 in
          let lane i =
            store buf i (Ops.after (i32 i) [ store local i (i32 i) ])
          in
          let sink =
            Ops.sink (store buf 0 (i32 0) :: List.map lane [ 1; 2; 3 ])
          in
          equal int 1 (stores_in Global sink));
      test "local stores each waited on by a merged global store stay apart"
        (fun () ->
          let buf = int_param 2 and local = scratch Local 2 in
          let s1 = store buf 1 (i32 5) and t0 = store local 0 (i32 7) in
          let s0 =
            store buf 0 (Ops.load (at (Ops.after local [ t0 ]) (Ops.int 0)) [])
          in
          let t1 =
            store local 1 (Ops.load (at (Ops.after buf [ s1 ]) (Ops.int 1)) [])
          in
          equal int 2 (stores_in Local (Ops.sink [ s0; s1; t0; t1 ])));
    ]

(* Generated kernels copy runs of consecutive elements of [input] into [output]
   inside a loop of four, adding up one or two runs of [input] per run of
   [output]. A run's elements are at constant offsets or at offsets from the
   loop's index, gated by a condition on it or not, and ordered after a store or
   not; its start is any offset, so that runs start aligned or not, and
   overlap. *)
type place = { scale : int; gated : bool; ordered : bool; start : int }
type run = { dst : place; count : int; srcs : place list }

let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ]

(* The store an ordered index waits on. *)
let ordering_store =
  Ops.store
    (Ops.index (Shape.param ~shape:[ Int 1 ] 2 Int32) [ Ops.int 0 ])
    (Ops.int ~dtype:Int32 7)

let index { scale; gated; ordered; start } j =
  let i =
    match (scale, start + j) with
    | 0, k -> Ops.int k
    | s, 0 -> Ops.O.(r * int s)
    | s, k -> Ops.O.((r * int s) + int k)
  in
  let i = if ordered then Ops.after i [ ordering_store ] else i in
  if gated then Shape.valid i Ops.O.(r < int 2) else i

let size = 40

let generated ?(phases = ((0, 16), (0, 16))) ?(local = false) (dtype, runs) =
  let buffer slot (phase, align) =
    Shape.param ~shape:[ Int size ] ~phase ~align slot dtype
  in
  let input = buffer 0 (fst phases) in
  let output =
    if local then Shape.placeholder ~slot:1 ~addrspace:Local [ size ] dtype
    else buffer 1 (snd phases)
  in
  (* A second store of one element under the same key is refused; the first
     store of each element is kept. *)
  let stores, _ =
    List.fold_left
      (fun (stores, seen) { dst; count; srcs } ->
        List.fold_left
          (fun (stores, seen) j ->
            let key = (dst.scale, dst.gated, dst.ordered, dst.start + j) in
            if List.mem key seen then (stores, seen)
            else
              let read p = Ops.load (Ops.index input [ index p j ]) [] in
              let value =
                List.fold_left Ops.add
                  (read (List.hd srcs))
                  (List.map read (List.tl srcs))
              in
              ( Ops.store (Ops.index output [ index dst j ]) value :: stores,
                key :: seen ))
          (stores, seen) (List.init count Fun.id))
      ([], []) runs
  in
  Ops.sink [ Ops.end_ (Ops.group (List.rev stores)) [ r ] ]

let graph_pp ppf u = Format.pp_print_string ppf (Graph.to_string u)

let kernels_of_runs =
  let open Gen in
  let place =
    let+ scale = of_list [ 0; 4; 8 ]
    and+ gated = bool
    and+ ordered = frequency [ (3, constant false); (1, constant true) ]
    and+ start = int_range 0 7 in
    { scale; gated; ordered; start }
  in
  let run =
    let+ dst = place
    and+ count = int_range 1 8
    and+ srcs = list ~size:(int_range 1 2) place in
    { dst; count; srcs }
  in
  let dtype =
    of_list
      Dtype.[ Float32; Float16; Bfloat16; Fp8e4m3; Float64; Int32; Int8; Uint8 ]
  in
  with_pp
    (fun ppf k -> graph_pp ppf (generated k))
    (pair dtype (list ~size:(int_range 1 3) run))

(* Kernels whose buffers start known modulo any alignment, any whole number of
   elements past it, in bytes, or of the alignment where it is less than an
   element. *)
let phased_kernels =
  let open Gen in
  let+ ((dtype, _) as k) = kernels_of_runs
  and+ input = pair (int_range 0 15) (of_list [ 1; 2; 4; 8; 16 ])
  and+ output = pair (int_range 0 15) (of_list [ 1; 2; 4; 8; 16 ]) in
  let known (n, align) =
    let step = min (Dtype.itemsize dtype) align in
    (n mod (align / step) * step, align)
  in
  ((known input, known output), k)

(* The input's elements are distinct, so that a merged access reading the wrong
   elements writes other values. Each type drawn holds them exactly: the floats
   have three bits of mantissa and exponents from -3 to 1, normal in every 8-bit
   float. Sums need no exactness: both sides add the same values in the same
   type. *)
let elements dtype =
  Array.init size (fun i ->
      if Dtype.is_float dtype then
        `Float
          (Float.ldexp (1. +. (Float.of_int (i mod 8) /. 8.)) ((i / 8) - 3))
      else `Int (Bigint.of_int ((3 * i) + 1)))

let write = triple int int Dtypes.value

let writes dtype u =
  Interpreter.writes ~buffers:[ (0, elements dtype); (1, [||]); (2, [||]) ] u

let merged op u =
  List.exists
    (fun u -> Ops.op u = op && Ops.op (Ops.nth u 0) = Op.Shrink)
    (Ops.toposort ~calls:Enter u)

(* The renderers that merge accesses. *)
let merging =
  [
    ("16-byte registers", vector);
    ("32-byte registers", host 16 32);
    ("64-byte registers", host 32 64);
    ("a GPU", gpu);
    ("a GPU writing bytes as words", gpu_words);
    ("a GPU writing eight 16-bit floats", gpu8);
  ]

let preserves_writes ((name, renderer), (phases, (dtype, runs))) =
  let k = generated ~phases (dtype, runs) in
  let coalesced = coalesce ~renderer k in
  List.iter
    (fun (n, _) ->
      cover ("merged on " ^ n)
        (n = name && (merged Op.Load coalesced || merged Op.Store coalesced)))
    merging;
  cover "an ordered index merged"
    (List.exists
       (fun u ->
         Ops.op u = Op.Load
         && Ops.op (Ops.nth u 0) = Op.Shrink
         && Ops.op (Ops.get_idx (Ops.nth (Ops.nth u 0) 1)) = Op.After)
       (Ops.toposort ~calls:Enter coalesced));
  equal (list write) (writes dtype k) (writes dtype coalesced)

let merging_kernels =
  Gen.(
    pair
      (of_list ~pp:(fun ppf (n, _) -> Format.pp_print_string ppf n) merging)
      phased_kernels)

(* The elements of a merged access from the boundary behind its buffer, a
   multiple of its alignment, for each value of the loop's range. *)
let from_boundary u =
  let p = Ops.nth u 0 in
  let lead =
    match Ops.arg (Ops.buf_uop (Ops.nth p 0)) with
    | Param q -> q.phase / Dtype.itemsize (Ops.dtype (Ops.nth p 0))
    | _ -> 0
  in
  List.init 4 (fun k ->
      lead
      + int_of
          (Shape.simplify
             (Ops.substitute ~calls:Skip ~pass:Fixed_point
                (Ops.get_idx (Ops.nth p 1))
                [ (r, Ops.int k) ])))

let aligned (phases, k) =
  let coalesced = coalesce ~renderer:gpu (generated ~phases k) in
  let vector_accesses =
    List.filter
      (fun u ->
        (Ops.op u = Op.Load || Ops.op u = Op.Store)
        && Ops.op (Ops.nth u 0) = Op.Shrink)
      (Ops.toposort ~calls:Enter coalesced)
  in
  cover "a vector access" (vector_accesses <> []);
  List.iter
    (fun u ->
      let width = int_of (Ops.nth (Ops.nth u 0) 2) in
      let buf = Ops.nth (Ops.nth u 0) 0 in
      (match Ops.arg (Ops.buf_uop buf) with
      | Param q ->
          at_most ~msg:"bytes within the alignment" int ~than:q.align
            (width * Dtype.itemsize (Ops.dtype buf))
      | _ -> ());
      List.iter
        (fun e ->
          equal ~msg:(Format.asprintf "%a" Ops.pp u) int 0 (e mod width))
        (from_boundary u))
    vector_accesses

(* Kernels whose every store copies an element of global memory into local
   memory, which a GPU merges by rules of its own: 16-bit floats into eight
   where it writes vectors of eight, bytes into sixteen where it writes them as
   words. Runs start on a 16-byte boundary often. *)
let copies =
  let open Gen in
  let place =
    let+ start =
      frequency [ (1, constant 0); (1, constant 8); (2, int_range 0 15) ]
    in
    { scale = 0; gated = false; ordered = false; start }
  in
  let run =
    let+ dst = place
    and+ count =
      frequency [ (1, constant 8); (1, constant 16); (2, int_range 1 16) ]
    and+ src = place in
    { dst; count; srcs = [ src ] }
  in
  let target =
    of_list
      ~pp:(fun ppf (name, _, _) -> Format.pp_print_string ppf name)
      [
        ("a GPU writing eight 16-bit floats", gpu8, Dtype.[ Float16; Bfloat16 ]);
        ("a GPU writing bytes as words", gpu_words, Dtype.[ Uint8 ]);
      ]
  in
  let* ((_, _, dtypes) as t) = target in
  let+ dtype = of_list dtypes and+ runs = list ~size:(int_range 1 3) run in
  (t, (dtype, runs))

let widest u =
  List.fold_left
    (fun w u ->
      if
        (Ops.op u = Op.Load || Ops.op u = Op.Store)
        && Ops.op (Ops.nth u 0) = Op.Shrink
      then max w (int_of (Ops.nth (Ops.nth u 0) 2))
      else w)
    1
    (Ops.toposort ~calls:Enter u)

let copies_keep_writes ((_, renderer, _), (dtype, runs)) =
  let k = generated ~local:true (dtype, runs) in
  let coalesced = coalesce ~renderer k in
  let width = widest coalesced in
  cover "eight 16-bit floats merged" (Dtype.itemsize dtype = 2 && width = 8);
  cover "sixteen bytes merged" (Dtype.itemsize dtype = 1 && width = 16);
  equal (list write) (writes dtype k) (writes dtype coalesced)

let generated_sinks = Gen.with_pp graph_pp (Gen.map generated kernels_of_runs)

(* The recorded kernels and the hand-built accesses. *)
let recorded_sinks =
  List.map kernel kernels @ Ops.src (Golden.sink "accesses_input.golden")

let laws =
  [
    prop ~count:600
      "keeps the kernel's writes, on every renderer that merges, for every \
       phase and alignment"
      merging_kernels preserves_writes;
    prop ~count:300
      "keeps the kernel's writes, where its stores copy into local memory"
      copies copies_keep_writes;
    prop
      "aligns every vector access to its width, within its buffer's alignment"
      phased_kernels aligned;
  ]

(* The kernels memory_coalescing returns as they are *)

let left_as_it_is =
  [
    prop "without vector accesses, returns the kernel as it is"
      ~examples:recorded_sinks generated_sinks (fun k ->
        equal Uops.uop k (coalesce ~renderer:scalar k));
    prop "under Setting.dmc, returns the kernel as it is"
      ~examples:recorded_sinks generated_sinks (fun k ->
        equal Uops.uop k
          (Setting.context [ B (Setting.dmc, true) ] (fun () -> coalesce k)));
    test "under Setting.dmc, returns a kernel it would refuse as it is"
      (fun () ->
        let gated =
          Ops.sink
            [
              Ops.load
                (at buf (Ops.int 0))
                [ Ops.float ~dtype:Float32 0.; Ops.bool true ];
            ]
        in
        equal Uops.uop gated
          (Setting.context [ B (Setting.dmc, true) ] (fun () -> coalesce gated)));
  ]

(* An index ordered after effects: four loads of [buf] at [index k], each index
   read after the effects [order k] gives, as a staged tile's read waits on the
   next trip's copies. Loads ordered after other effects, or not ordered, must
   not merge: a merged read would move before a copy it waits on. *)
let ordered_loads index order =
  let at k =
    match order k with [] -> index k | effects -> Ops.after (index k) effects
  in
  Ops.sink (List.init 4 (fun k -> Ops.load (Ops.index buf [ at k ]) []))

let ordered_cases =
  let x = Ops.variable "x" (`Int Bigint.zero) (`Int (Bigint.of_int 8)) in
  [
    ( "x * 4 + k after a store",
      (fun k -> Ops.O.((x * int 4) + int k)),
      (fun _ -> [ ordering_store ]),
      [ 4 ] );
  ]

let widths op sink =
  List.filter_map
    (fun u ->
      if Ops.op u = op then
        Some
          (match Ops.op (Ops.nth u 0) with
          | Op.Shrink -> int_of (Ops.nth (Ops.nth u 0) 2)
          | _ -> 1)
      else None)
    (Ops.toposort ~calls:Enter sink)

(* A register of four floats read in full, on a renderer that computes vectors
   of four. *)
let register_loads =
  loads
    (Shape.placeholder ~slot:3 ~addrspace:Reg [ 16 ] Float32)
    (List.init 4 Fun.id)

let merges_as_values =
  [
    cases "loads at indices ordered after a store merge as at their values, at"
      ~id:(fun (name, _, _, _) -> name)
      ordered_cases
      (fun (_, index, order, expected) ->
        equal (list int) expected
          (widths Op.Load (coalesce (ordered_loads index order))));
    test "register memory merges where the renderer computes vectors" (fun () ->
        equal (list int) [ 4 ]
          (widths Op.Load
             (coalesce
                ~renderer:(Renderer.v ~vector_lanes:4 cpu)
                register_loads)));
  ]

(* merges_four *)

let stored_dtypes =
  List.filter
    (fun d -> not (List.mem d Dtype.[ Void; Weak_int; Weak_float ]))
    Dtypes.declared

(* Renderers whose vector registers are not known. *)
let unknown_registers =
  [
    ("the CPU device", cpu_unknown);
    ("a GPU", gpu);
    ("a GPU writing bytes as words", gpu_words);
    ("a renderer without vector accesses", scalar);
  ]

(* The types the interface lists for [r]. *)
let listed (r : Renderer.t) =
  let always =
    Dtype.
      [
        Uint8;
        Uint32;
        Int32;
        Fp8e4m3;
        Fp8e5m2;
        Fp8e4m3fnuz;
        Fp8e5m2fnuz;
        Float16;
        Bfloat16;
        Float32;
      ]
  in
  List.filter
    (fun dt ->
      r.supports_float4 && List.mem dt always
      && (dt <> Bfloat16 || r.target.device <> "CPU")
      && (dt <> Uint8 || r.byte_words))
    stored_dtypes

let four_merge r dt =
  let b = Shape.param ~shape:[ Int 64 ] 0 dt in
  widths Op.Load (coalesce ~renderer:r (loads b (List.init 4 Fun.id))) = [ 4 ]

let merges_four =
  group "merges_four"
    [
      cases "holds for the types the interface lists, on" ~id:fst
        unknown_registers (fun (_, r) ->
          equal (list Dtypes.dtype) (listed r)
            (List.filter (Coalesce.merges_four r) stored_dtypes));
      cases "is whether four consecutive loads merge into one, on" ~id:fst
        unknown_registers (fun (_, r) ->
          equal (list Dtypes.dtype)
            (List.filter (Coalesce.merges_four r) stored_dtypes)
            (List.filter (four_merge r) stored_dtypes));
    ]

(* A gather through a pad: element [r] of slot 2 is [xs.(ids.(r - 4))] from [r =
   4] and [0] before it. Slot 0's index is loaded under the pad's gate. *)
let padded_gather =
  let r = Ops.range ~axis_type:Weak (Int 8) [ 0 ] in
  let ids = Shape.param ~shape:[ Int 4 ] 0 Int32 in
  let xs = Shape.param ~shape:[ Int 4 ] 1 Float32 in
  let out = Shape.param ~shape:[ Int 8 ] 2 Float32 in
  let inside = Ops.O.(int 3 < r) in
  let id =
    Ops.load (Ops.index ids [ Shape.valid Ops.O.(r - int 4) inside ]) []
  in
  let at_id = Shape.valid (Ops.cast id Weak_int) inside in
  let x = Ops.load (Ops.index xs [ at_id ]) [] in
  Ops.sink [ Ops.end_ (Ops.store (Ops.index out [ r ]) x) [ r ] ]

let gathered_writes u =
  let ids = Array.map (fun i -> `Int (Bigint.of_int i)) [| 2; 0; 3; 1 |] in
  let xs = Array.init 4 (fun i -> `Float (Float.of_int i +. 0.5)) in
  Interpreter.writes ~buffers:[ (0, ids); (1, xs); (2, [||]) ] u

let indexing_simplify =
  group "indexing_simplify"
    [
      group "simplifies each index as its golden states"
        [
          Golden.cases "indices.golden" (fun cell ->
              equal Uops.uop
                (case "indices_simplified.golden" cell)
                (simplify (case "indices_input.golden" cell)));
        ];
      (* A load in an index runs whatever the index's gate, so its own index
         keeps its gate. *)
      test "a gather through a pad reads its indices only inside the pad"
        (fun () ->
          equal (list write)
            (gathered_writes padded_gather)
            (gathered_writes (simplify padded_gather)));
    ]

(* Elements of [dt] accessed in each way a kernel accesses memory: each case's
   name, element type, sink, and runs of loads and of stores on a GPU that
   writes vectors of eight 16-bit floats. *)
let sixteen_bits dt =
  let global = Shape.param ~shape:[ Int 64 ] 2 dt
  and other = Shape.param ~shape:[ Int 64 ] 3 dt
  and local = Shape.placeholder ~slot:1 ~addrspace:Local [ 64 ] dt
  and scratch = Shape.placeholder ~slot:4 ~addrspace:Local [ 64 ] dt in
  let eight = List.init 8 Fun.id and fours = [ (0, 4); (4, 4) ] in
  let copy into =
    List.map (fun i -> Ops.store (at into (Ops.int i)) (load_at global i)) eight
  in
  [
    ( "a copy from global into local memory",
      dt,
      Ops.sink (copy local),
      ([ (0, 8) ], [ (0, 8) ]) );
    ("a copy between global buffers", dt, Ops.sink (copy other), (fours, fours));
    ("loads of global memory", dt, loads global eight, (fours, []));
    ("stores into global memory", dt, stores global eight, ([], fours));
    ("loads of local memory", dt, loads local eight, (fours, []));
    ("stores into local memory", dt, stores local eight, ([], fours));
    ( "a copy into local memory whose loads are read elsewhere",
      dt,
      Ops.sink (copy local @ List.map (load_at global) eight),
      (fours, fours) );
    ( "a copy between local buffers",
      dt,
      Ops.sink
        (List.map
           (fun i -> Ops.store (at local (Ops.int i)) (load_at scratch i))
           eight),
      (fours, fours) );
    ( "sixteen copies into local memory, one load read elsewhere",
      dt,
      Ops.sink
        (List.map
           (fun i -> Ops.store (at local (Ops.int i)) (load_at global i))
           (List.init 16 Fun.id)
        @ [ load_at global 15 ]),
      ([ (0, 4); (4, 4); (8, 4); (12, 4) ], [ (0, 4); (4, 4); (8, 4); (12, 4) ])
    );
  ]

(* A GPU that writes vectors of eight 16-bit floats merges 16 bytes of them
   where they are copied from global into local memory. *)
let half8 =
  let on_gpu8 name =
    Golden.graph (name ^ "_cuda.golden") (fun () ->
        coalesce ~renderer:gpu8 (kernel name))
  in
  group "vectors of eight 16-bit floats"
    [
      group "merge each kernel's accesses as its golden states"
        [
          on_gpu8 "half";
          on_gpu8 "half8";
          hand_built "accesses_cuda.golden" gpu8;
        ];
      cases "the runs of eight 16-bit floats' loads and stores"
        ~id:(fun (name, dt, _, _) -> name ^ ", " ^ Dtype.name dt)
        (List.concat_map sixteen_bits [ Dtype.Float16; Dtype.Bfloat16 ])
        (fun (_, _, sink, expected) ->
          let merged = coalesce ~renderer:gpu8 sink in
          equal (pair runs runs) expected
            (accesses Op.Load merged, accesses Op.Store merged));
      test "eight floats are still two loads of four" (fun () ->
          equal runs
            [ (0, 4); (4, 4) ]
            (accesses Op.Load
               (coalesce ~renderer:gpu8 (loads buf (List.init 8 Fun.id)))));
    ]

let memory_coalescing =
  group "memory_coalescing"
    ([
       group "merges each kernel's accesses as its golden states"
         (hand_built "accesses_coalesced.golden" vector
         :: List.map compiled kernels);
       lengths;
       half8;
     ]
    @ merges_as_values @ [ phases; left_alone ] @ left_as_it_is
    @ [ dependencies; errors ] @ laws)

let () =
  exit
    (run "Tolk.Coalesce" [ indexing_simplify; merges_four; memory_coalescing ])
