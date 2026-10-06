open Windtrap
open Tolk

(* The tag of the tests that run with ASSERT_COMPILE set, in a run of their own,
   since a process reads the variable once. *)
let assert_compile = "assert-compile"
let rejects f = raises_match (Exn.invalid_arg ?substring:None) f
let sint = Testable.make ~pp:Shape.Sint.pp ~equal:Shape.Sint.equal

let estimates =
  let equal (e0 : Ops.estimates) (e1 : Ops.estimates) =
    Shape.Sint.equal e0.ops e1.ops
    && Shape.Sint.equal e0.lds e1.lds
    && Shape.Sint.equal e0.mem e1.mem
  in
  Testable.make ~pp:Ops.pp_estimates ~equal

let counts ops lds mem : Ops.estimates =
  { ops = Int ops; lds = Int lds; mem = Int mem }

let cpu = Result.get_ok (Helpers.Target.of_string "CPU")

(* Kernels *)

let f32 x = Ops.float ~dtype:Dtype.Float32 x

let buffer ?(dtype = Dtype.Float32) slot size =
  Shape.param ~shape:[ Int size ] slot dtype

let range ?(axis = 0) size = Ops.range (Int size) [ axis ]
let at buf i = Ops.index buf [ i ]
let load buf i = Ops.load (at buf i) []

(* estimates of hand-written kernels *)

let arithmetic () =
  let a = f32 1. and b = f32 2. in
  let sum = Ops.O.(a + b) and negated = Ops.neg a in
  equal sint (Int 2) (Renderer.Estimates.of_uops [ a; b; sum; negated ]).ops

let multiply_add_is_a_multiply_and_an_add () =
  let globl = buffer ~dtype:Dtype.Int32 0 3 in
  let u1 = at globl (Ops.int 1) and u2 = at globl (Ops.int 2) in
  let u3 = Ops.int ~dtype:Dtype.Int32 3 in
  let separate = Ops.v Add ~src:[ Ops.v Mul ~src:[ u1; u2 ]; u3 ] in
  let fused = Ops.v Mulacc ~src:[ u1; u2; u3 ] in
  let flops_lds u =
    let e = Renderer.Estimates.of_uops (Ops.toposort ~calls:Enter u) in
    (e.ops, e.lds)
  in
  equal (pair sint sint) (flops_lds separate) (flops_lds fused);
  equal sint (Int 2) (fst (flops_lds fused))

let tensor_core ~threads =
  let a = Ops.float ~dtype:Dtype.Float16 1. in
  Ops.wmma a a ~acc:(f32 0.) ~dims:(8, 16, 16) ~threads

let tensor_core_product () =
  let ops threads =
    (Renderer.Estimates.of_uops
       (Ops.toposort ~calls:Enter (tensor_core ~threads)))
      .ops
  in
  equal sint ~msg:"a warp of 32" (Int (2 * 8 * 16 * 16 / 32)) (ops 32);
  equal sint ~msg:"a warp of 64" (Int (2 * 8 * 16 * 16 / 64)) (ops 64)

let not_arithmetic () =
  let buf = buffer 0 4 and out = buffer 1 4 in
  let x = load buf (Ops.int 0) in
  let cast = Ops.cast x Dtype.Float16 and bits = Ops.bitcast x Dtype.Int32 in
  let stored = Ops.store (at out (Ops.int 0)) x in
  equal sint (Int 0)
    (Renderer.Estimates.of_uops
       (Ops.toposort ~calls:Enter (Ops.sink [ stored; cast; bits ])))
      .ops

let lanes () =
  let v = Shape.stack [ f32 1.; f32 2.; f32 3.; f32 4. ] in
  let sum = Ops.O.(v + v) in
  equal sint (Int 4) (Renderer.Estimates.of_uops [ v; sum ]).ops

let a_range_multiplies () =
  let r = range 10 and a = f32 1. in
  let inside = Ops.O.(a + a) in
  let ended = Ops.end_ inside [ r ] in
  let outside = Ops.neg a in
  equal sint (Int 11)
    (Renderer.Estimates.of_uops [ r; a; inside; ended; outside ]).ops

let nested_ranges () =
  let outer = range 10 and inner = range ~axis:1 3 and a = f32 1. in
  let body = Ops.O.(a + a) in
  let ended = Ops.end_ body [ inner ] in
  let after = Ops.neg a in
  let closed = Ops.end_ after [ outer ] in
  equal sint
    (Int ((10 * 3) + 10))
    (Renderer.Estimates.of_uops [ outer; inner; a; body; ended; after; closed ])
      .ops

let a_loop_counts_once () =
  let loop = Ops.loop 0 and a = f32 1. in
  let body = Ops.O.(a + a) in
  let back = Ops.backedge body ~loop ~cond:(Ops.bool false) in
  equal sint (Int 1) (Renderer.Estimates.of_uops [ loop; a; body; back ]).ops

let a_hardware_index_multiplies_what_follows () =
  let gidx = Ops.special (Int 8) "gidx0" and r = range 2 and a = f32 1. in
  let inside = Ops.O.(a + a) in
  let ended = Ops.end_ inside [ r ] in
  let outside = Ops.neg a in
  equal sint
    (Int ((8 * 2) + 8))
    (Renderer.Estimates.of_uops [ gidx; r; a; inside; ended; outside ]).ops

let an_end_without_a_range () =
  let a = f32 1. in
  rejects (fun () ->
      Renderer.Estimates.of_uops [ a; Ops.end_ Ops.O.(a + a) [ range 4 ] ])

let a_tensor_core_without_its_argument () =
  let a = Ops.float ~dtype:Dtype.Float16 1. in
  let product = Ops.v Wmma ~src:[ a; a; f32 0. ] in
  rejects (fun () ->
      Renderer.Estimates.of_uops (Ops.toposort ~calls:Enter product))

let a_backedge_without_a_loop () =
  let a = f32 1. in
  let back = Ops.backedge a ~loop:(Ops.loop 0) ~cond:(Ops.bool false) in
  rejects (fun () -> Renderer.Estimates.of_uops [ a; back ])

let reads_are_capped_at_the_buffer () =
  let buf = buffer 0 1 and zero = Ops.int 0 and r = range 10 in
  let x = load buf zero in
  let ended = Ops.end_ x [ r ] in
  let e = Renderer.Estimates.of_uops [ buf; zero; r; at buf zero; x; ended ] in
  equal sint ~msg:"lds counts every read" (Int 40) e.lds;
  equal sint ~msg:"mem counts the buffer once" (Int 4) e.mem

let a_store_counts_its_value () =
  let buf = buffer ~dtype:Dtype.Int32 0 4 in
  let stored = Ops.store (at buf (Ops.int 1)) (Ops.int ~dtype:Dtype.Int32 5) in
  equal estimates (counts 0 4 4)
    (Renderer.Estimates.of_uops
       (Ops.toposort ~calls:Enter (Ops.sink [ stored ])))

let loads_and_stores_count_apart () =
  let buf = buffer 0 16 and r = range 16 in
  let x = load buf r in
  let stored = Ops.store (at buf r) Ops.O.(x + x) in
  let ended = Ops.end_ stored [ r ] in
  let e =
    Renderer.Estimates.of_uops (Ops.toposort ~calls:Enter (Ops.sink [ ended ]))
  in
  equal sint ~msg:"a read and a write of 64 bytes" (Int 128) e.mem;
  equal sint (Int 128) e.lds

let two_loads_of_one_buffer () =
  let buf = buffer 0 4 and r = range 4 in
  let sum = Ops.O.(load buf r + load buf (Ops.int 3)) in
  let ended = Ops.end_ sum [ r ] in
  let e =
    Renderer.Estimates.of_uops (Ops.toposort ~calls:Enter (Ops.sink [ ended ]))
  in
  equal sint ~msg:"lds counts both" (Int 32) e.lds;
  equal sint ~msg:"mem counts the buffer once" (Int 16) e.mem

let registers_move_no_bytes () =
  let reg = Shape.alloc ~addrspace:Reg [ Int 4 ] Dtype.Float32 in
  let stored = Ops.store (at reg (Ops.int 0)) (f32 1.) in
  let x = load reg (Ops.int 1) in
  equal estimates (counts 0 0 0)
    (Renderer.Estimates.of_uops
       (Ops.toposort ~calls:Enter (Ops.sink [ stored; x ])))

let shared_memory_is_not_a_parameter () =
  let smem = Shape.alloc ~addrspace:Local [ Int 4 ] Dtype.Float32 in
  let stored = Ops.store (at smem (Ops.int 0)) (f32 1.) in
  equal estimates (counts 0 4 0)
    (Renderer.Estimates.of_uops
       (Ops.toposort ~calls:Enter (Ops.sink [ stored ])))

let indexing () =
  let buf = buffer 0 32 and out = buffer 1 32 and r = range 16 in
  let x = load buf Ops.O.((r * int 2) + int 1) in
  let stored = Ops.store (at out r) Ops.O.(x * x) in
  let uops = Ops.toposort ~calls:Enter (Ops.sink [ Ops.end_ stored [ r ] ]) in
  equal sint ~msg:"counting the index"
    (Int (16 * 3))
    (Renderer.Estimates.of_uops uops).ops;
  equal sint ~msg:"ignoring it" (Int 16)
    (Renderer.Estimates.of_uops ~ignore_indexing:true uops).ops

let an_index_shared_with_a_value () =
  let out = buffer 0 32 and r = range 16 in
  let twice = Ops.O.(r * int 2) in
  let stored = Ops.store (at out twice) (Ops.cast twice Dtype.Float32) in
  let uops = Ops.toposort ~calls:Enter (Ops.sink [ Ops.end_ stored [ r ] ]) in
  equal sint (Int 0) (Renderer.Estimates.of_uops ~ignore_indexing:true uops).ops

let shrink_indexing () =
  let storage = buffer 0 32
  and a = Ops.variable "a" (`Int Bigint.zero) (`Int (Bigint.of_int 8)) in
  let shrunk =
    Ops.v Shrink ~src:[ storage; Ops.O.(a + int 1); Ops.O.(a * int 2) ]
  in
  let uops = Ops.toposort ~calls:Enter (Ops.sink [ Ops.load shrunk [] ]) in
  equal sint (Int 0) (Renderer.Estimates.of_uops ~ignore_indexing:true uops).ops

(* An index that reads the result of a loop: the loop computes the index, but
   the End or Backedge that closes it bounds what the index's operations are. *)
let an_index_after_a_loop () =
  let acc = Shape.alloc ~addrspace:Reg [ Int 1 ] Dtype.Int32 in
  let cell = at acc (Ops.int 0) in
  let ops closed =
    let total = Ops.load (Ops.after cell [ closed ]) [] in
    let stored = Ops.store (at (buffer 0 16) total) (f32 1.) in
    (Renderer.Estimates.of_uops ~ignore_indexing:true
       (Ops.toposort ~calls:Enter (Ops.sink [ stored ])))
      .ops
  in
  let r = range 4 and loop = Ops.loop 1 in
  let update i = Ops.store cell Ops.O.(Ops.load cell [] + i) in
  equal sint ~msg:"a range" (Int 4) (ops (Ops.end_ (update r) [ r ]));
  equal sint ~msg:"a loop" (Int 1)
    (ops (Ops.backedge (update (Ops.int 1)) ~loop ~cond:(Ops.bool false)))

let estimates_of_uops =
  group "estimates of hand-written kernels"
    [
      test "an empty kernel is zero" (fun () ->
          equal estimates Renderer.Estimates.zero
            (Renderer.Estimates.of_uops []));
      test "counts each arithmetic operation once" arithmetic;
      test "counts a multiply-add as a multiply and an add"
        multiply_add_is_a_multiply_and_an_add;
      test "counts a tensor core product as 2NMK shared among its threads"
        tensor_core_product;
      test "counts no cast, bitcast, load or store as arithmetic" not_arithmetic;
      test "counts an operation on four lanes four times" lanes;
      test "counts the operations of a range once per iteration"
        a_range_multiplies;
      test "multiplies the trip counts of nested ranges" nested_ranges;
      test "counts a loop without trip count as one iteration"
        a_loop_counts_once;
      test "multiplies everything after a hardware index by its size"
        a_hardware_index_multiplies_what_follows;
      test "rejects an end that closes no range" an_end_without_a_range;
      test "rejects a backedge that closes no loop" a_backedge_without_a_loop;
      test "rejects a tensor core product without its argument"
        a_tensor_core_without_its_argument;
      test "counts every read in lds and a buffer read again once in mem"
        reads_are_capped_at_the_buffer;
      test "counts the bytes a store writes" a_store_counts_its_value;
      test "counts the loads and the stores of a buffer apart in mem"
        loads_and_stores_count_apart;
      test "counts a buffer that two loads read once in mem"
        two_loads_of_one_buffer;
      test "counts no bytes for registers" registers_move_no_bytes;
      test "counts shared memory in lds but not in mem"
        shared_memory_is_not_a_parameter;
      test "ignore_indexing leaves out the operations of indices" indexing;
      test
        "ignore_indexing leaves out an operation an index shares with a value"
        an_index_shared_with_a_value;
      test "ignore_indexing counts a loop whose result an index reads"
        an_index_after_a_loop;
      test "ignore_indexing leaves out the operations of shrinks"
        shrink_indexing;
    ]

(* estimates of recorded kernels *)

(* The cases of a kernel golden are its sink's sources, each a linear program
   whose sources are the kernel's nodes in order. *)
let kernels file = lazy (Array.of_list (Ops.src (Golden.sink file)))
let recorded = kernels "kernels.golden"
let symbolic = kernels "symbolic_kernels.golden"
let kernel file cell = Ops.src (Lazy.force file).(int_of_string (cell "src"))
let n = Ops.variable "n" (`Int Bigint.one) (`Int (Bigint.of_int 8))
let at_n value s = Shape.sym_infer s [ ("n", value) ]

(* A split kernel's counts are those of one block; the goldens count the block
   that runs the whole loop, of the iterations in the cell "split". *)
let whole cell vars =
  match cell "split" with
  | "-" -> vars
  | extent -> ("block_lo", 0) :: ("block_hi", int_of_string extent) :: vars

let recorded_kernels =
  group "estimates of recorded kernels"
    [
      Golden.cases "estimates.golden" (fun cell ->
          let uops = kernel recorded cell in
          let e = Renderer.Estimates.of_uops uops in
          let expect column s =
            equal int ~msg:column
              (int_of_string (cell column))
              (Shape.sym_infer s (whole cell []))
          in
          expect "ops" e.ops;
          expect "lds" e.lds;
          expect "mem" e.mem;
          expect "ops_ignoring_indexing"
            (Renderer.Estimates.of_uops ~ignore_indexing:true uops).ops);
      Golden.cases "symbolic_estimates.golden" ~key:[ "case"; "n" ] (fun cell ->
          let uops = kernel symbolic cell
          and value = int_of_string (cell "n") in
          let e = Renderer.Estimates.of_uops uops in
          let expect column s =
            equal int ~msg:column
              (int_of_string (cell column))
              (Shape.sym_infer s (whole cell [ ("n", value) ]))
          in
          expect "ops" e.ops;
          expect "lds" e.lds;
          expect "mem" e.mem;
          expect "ops_ignoring_indexing"
            (Renderer.Estimates.of_uops ~ignore_indexing:true uops).ops);
    ]

(* estimates of symbolic kernels *)

let symbolic_trip_count () =
  let r = Ops.range (Sym n) [ 0 ] and a = f32 1. in
  let first = Ops.O.(a + a) in
  let second = Ops.neg first in
  let ops =
    (Renderer.Estimates.of_uops
       [ n; r; a; first; second; Ops.end_ second [ r ] ])
      .ops
  in
  equal int ~msg:"n = 3" 6 (at_n 3 ops);
  equal int ~msg:"n = 8" 16 (at_n 8 ops)

let cancelling_trip_count () =
  let size = Ops.O.(n + int 3 - n) in
  let r = Ops.range (Sym size) [ 0 ] and a = f32 1. in
  let body = Ops.O.(a + a) in
  equal sint (Int 3)
    (Renderer.Estimates.of_uops [ r; a; body; Ops.end_ body [ r ] ]).ops

let symbolic_reads_are_capped () =
  let buf = buffer 0 4 and zero = Ops.int 0 in
  let r = Ops.range (Sym n) [ 0 ] in
  let x = load buf zero in
  let e =
    Renderer.Estimates.of_uops
      [ buf; zero; r; at buf zero; x; Ops.end_ x [ r ] ]
  in
  equal int ~msg:"mem at n = 2" 8 (at_n 2 e.mem);
  equal int ~msg:"mem at n = 8" 16 (at_n 8 e.mem);
  equal int ~msg:"lds at n = 8" 32 (at_n 8 e.lds)

let trip_count_of_a_hardware_index () =
  let gidx = Ops.special (Int 4) "gidx0" in
  let r = Ops.range (Sym Ops.O.(gidx + int 3)) [ 0 ] and a = f32 1. in
  let body = Ops.O.(a + a) in
  equal sint
    (Int (4 * 3))
    (Renderer.Estimates.of_uops [ gidx; r; a; body; Ops.end_ body [ r ] ]).ops

let symbolic_kernels =
  group "estimates of symbolic kernels"
    [
      test "a symbolic trip count counts at the value of its variable"
        symbolic_trip_count;
      test "a trip count that simplifies to an integer counts as one"
        cancelling_trip_count;
      test "reads are capped at the buffer for each value of a variable"
        symbolic_reads_are_capped;
      test "a trip count that depends on a hardware index is taken at index 0"
        trip_count_of_a_hardware_index;
    ]

(* add, zero and simplify *)

let integer_estimates =
  let count = Gen.map (fun c -> Ops.Int c) (Gen.int_range 0 1_000_000) in
  Gen.with_pp Ops.pp_estimates
    (let open Gen in
     let+ ops = count and+ lds = count and+ mem = count in
     ({ ops; lds; mem } : Ops.estimates))

let add = Renderer.Estimates.add

let symbolic_sum () =
  let e : Ops.estimates = { ops = Sym n; lds = Sym n; mem = Sym n } in
  let sum = add e e in
  List.iter
    (fun (name, s) -> equal int ~msg:name 6 (at_n 3 s))
    [ ("ops", sum.ops); ("lds", sum.lds); ("mem", sum.mem) ]

let arithmetic_estimates =
  group "add and zero"
    [
      test "zero counts nothing" (fun () ->
          equal estimates (counts 0 0 0) Renderer.Estimates.zero);
      test "add sums each count" (fun () ->
          equal estimates (counts 5 7 9) (add (counts 1 2 3) (counts 4 5 6)));
      prop "add is associative"
        (Gen.triple integer_estimates integer_estimates integer_estimates)
        (Law.associative estimates add);
      prop "add is commutative"
        (Gen.pair integer_estimates integer_estimates)
        (Law.commutative estimates add);
      prop "zero is the neutral element of add" integer_estimates
        (Law.neutral estimates add Renderer.Estimates.zero);
      test "add sums symbolic counts" symbolic_sum;
      test "add past max_int raises rather than wrapping" (fun () ->
          rejects (fun () -> add (counts max_int 0 0) (counts 1 0 0)));
    ]

let symbolic_estimates =
  let open Gen in
  let term =
    let+ k = int_range (-3) 3 in
    Ops.O.(n * int k)
  in
  with_pp Ops.pp_estimates
    (let+ a = term and+ b = term and+ c = int_range 0 5 in
     let count = Ops.Sym Ops.O.(a + b + int c) in
     ({ ops = count; lds = count; mem = Int c } : Ops.estimates))

let at_every_n (e : Ops.estimates) =
  List.init 8 (fun i ->
      let v s =
        match (s : Ops.sint) with Int c -> c | Sym _ -> at_n (i + 1) s
      in
      (v e.ops, v e.lds, v e.mem))

let simplification =
  group "simplify"
    [
      test "leaves integer counts as they are" (fun () ->
          equal estimates (counts 1 2 3)
            (Renderer.Estimates.simplify (counts 1 2 3)));
      test "a count of constants becomes an integer" (fun () ->
          let five = Ops.Sym Ops.O.(Ops.int 2 + Ops.int 3) in
          equal estimates (counts 5 5 5)
            (Renderer.Estimates.simplify { ops = five; lds = five; mem = five }));
      prop "keeps the value of each count" symbolic_estimates (fun e ->
          equal
            (list (triple int int int))
            (at_every_n e)
            (at_every_n (Renderer.Estimates.simplify e)));
      prop "is idempotent" symbolic_estimates
        (Law.idempotent estimates Renderer.Estimates.simplify);
    ]

(* with_storage *)

let accesses = lazy (Array.of_list (Ops.src (Golden.sink "accesses.golden")))
let restatements = Golden.rows "restatements.golden"

let restated () =
  Ops.sink
    (List.map
       (fun cell ->
         let access = (Lazy.force accesses).(int_of_string (cell "src")) in
         Renderer.with_storage access (Dtypes.dtype_of_cell (cell "dtype")))
       restatements)

(* An access: an index of storage of any stored type, possibly ordered after a
   store, possibly loaded, with the storage's type. *)
type access = { access : Ops.t; storage_dtype : Dtype.t }

let pp_access ppf a = Ops.pp ppf a.access

let access =
  let open Gen in
  let storages =
    [
      ("param", fun dtype -> buffer ~dtype 0 16);
      ("buffer", fun dtype -> Ops.new_buffer ~slot:3 (Single "CPU") 16 dtype);
      ("local", fun dtype -> Shape.alloc ~addrspace:Local [ Int 16 ] dtype);
      ("register", fun dtype -> Shape.alloc ~addrspace:Reg [ Int 16 ] dtype);
    ]
  in
  let pp_storage ppf (name, _) = Format.pp_print_string ppf name in
  with_pp pp_access
    (let+ storage_dtype = Dtypes.stored
     and+ _, storage = of_list ~pp:pp_storage storages
     and+ after = bool
     and+ loaded = bool
     and+ gated = bool in
     let r = range 16 and s = storage storage_dtype in
     let s =
       if after then Ops.after s [ Ops.store (at (buffer 1 16) r) (f32 1.) ]
       else s
     in
     let index =
       if gated then Ops.index s [ r; Ops.O.(r < int 8) ] else at s r
     in
     { access = (if loaded then Ops.load index [] else index); storage_dtype })

(* [path u] is the nodes from [u] to its storage, following first sources. *)
let rec path u =
  match Ops.op u with
  | Param | Buffer | Alloc -> [ u ]
  | _ -> u :: path (Ops.nth u 0)

let later_sources u = match Ops.src u with [] -> [] | _ :: rest -> rest

let restating =
  group "with_storage"
    [
      Golden.graph "restated.golden" restated;
      prop "restates the storage at the type and keeps every other source"
        (Gen.pair access Dtypes.stored) (fun (a, dt) ->
          let before = path a.access
          and after = path (Renderer.with_storage a.access dt) in
          equal int ~msg:"the path's length" (List.length before)
            (List.length after);
          equal Dtypes.dtype dt (Ops.dtype (List.hd (List.rev after)));
          List.iter2
            (fun u v ->
              equal (list Uops.uop) (later_sources u) (later_sources v))
            before after);
      prop "restating at the storage's own type is the identity" access
        (fun a ->
          equal Uops.uop a.access
            (Renderer.with_storage a.access a.storage_dtype));
      prop "restating back at the storage's type gives the access back"
        (Gen.pair access Dtypes.stored) (fun (a, dt) ->
          equal Uops.uop a.access
            (Renderer.with_storage
               (Renderer.with_storage a.access dt)
               a.storage_dtype));
      prop "is idempotent" (Gen.pair access Dtypes.stored) (fun (a, dt) ->
          Law.idempotent Uops.uop (fun u -> Renderer.with_storage u dt) a.access);
      test "rejects a node whose first sources reach no storage" (fun () ->
          rejects (fun () -> Renderer.with_storage (Ops.int 3) Dtype.Uint8);
          rejects (fun () -> Renderer.with_storage (range 4) Dtype.Uint8);
          rejects (fun () ->
              Renderer.with_storage Ops.O.(f32 1. + f32 2.) Dtype.Uint8));
    ]

(* Compilers *)

(* A toolchain that counts its runs. *)
type toolchain = { mutable runs : int }

let toolchain () = { runs = 0 }

let build t src =
  t.runs <- t.runs + 1;
  "lib:" ^ src

let tables = ref 0

(* Each test caches into a table of its own, so that no entry of another test or
   run is seen. *)
let fresh_table () =
  incr tables;
  Printf.sprintf "test_renderer_%d_%d" (Unix.getpid ()) !tables

let cached ?(t = toolchain ()) () =
  let table = fresh_table () in
  (Renderer.Compiler.v ~cachekey:(fun () -> table) (build t), table, t)

let get table src = Helpers.Diskcache.get ~table src
let ccache_off f = Setting.context [ B (Setting.ccache, false) ] f
let cache_off f = Setting.context [ B (Setting.cachelevel, 0) ] f

let compiles_once () =
  let c, table, t = cached () in
  equal string "lib:a" (Renderer.Compiler.compile_cached c "a");
  equal (option string) (Some "lib:a") (get table "a");
  equal string "lib:a" (Renderer.Compiler.compile_cached c "a");
  equal int ~msg:"runs of the toolchain" 1 t.runs

let a_cached_binary_wins () =
  let c, table, t = cached () in
  Helpers.Diskcache.put ~table "a" "held";
  equal string "held" (Renderer.Compiler.compile_cached c "a");
  equal int ~msg:"runs of the toolchain" 0 t.runs

let a_damaged_entry_is_compiled_anew () =
  let c, table, t = cached () in
  let before = Disk_cache.entries Helpers.cachedb in
  Helpers.Diskcache.put ~table "a" "held";
  let entry =
    List.find
      (fun e -> not (List.mem e before))
      (Disk_cache.entries Helpers.cachedb)
  in
  Out_channel.with_open_bin entry (fun oc -> output_string oc "damaged");
  equal string "lib:a" (Renderer.Compiler.compile_cached c "a");
  equal int ~msg:"runs of the toolchain" 1 t.runs;
  equal (option string) ~msg:"replaced" (Some "lib:a") (get table "a")

let without_ccache () =
  let c, table, t = cached () in
  ccache_off (fun () ->
      equal string "lib:a" (Renderer.Compiler.compile_cached c "a");
      equal string "lib:a" (Renderer.Compiler.compile_cached c "a"));
  equal (option string) None (get table "a");
  equal int ~msg:"runs of the toolchain" 2 t.runs

(* A compiler made with ccache off keeps its binaries once ccache holds, and
   names its table either way. *)
let ccache_is_read_when_compiling () =
  let table = fresh_table () in
  let c =
    ccache_off (fun () ->
        Renderer.Compiler.v ~cachekey:(fun () -> table) (build (toolchain ())))
  in
  equal (option string) ~msg:"made with ccache off" (Some table)
    (Renderer.Compiler.cachekey c);
  ignore (Renderer.Compiler.compile_cached c "a");
  equal (option string) (Some "lib:a") (get table "a")

let with_the_cache_disabled () =
  let c, table, t = cached () in
  cache_off (fun () ->
      ignore (Renderer.Compiler.compile_cached c "a");
      ignore (Renderer.Compiler.compile_cached c "a"));
  equal int ~msg:"runs of the toolchain" 2 t.runs;
  equal (option string) None (get table "a")

let uncached_compiler () =
  let t = toolchain () in
  let c = Renderer.Compiler.v (build t) in
  equal (option string) None (Renderer.Compiler.cachekey c);
  ignore (Renderer.Compiler.compile_cached c "a");
  ignore (Renderer.Compiler.compile_cached c "a");
  equal int ~msg:"runs of the toolchain" 2 t.runs

let tables_are_apart () =
  let t = toolchain () in
  let c0, _, _ = cached ~t () and c1, _, _ = cached ~t () in
  ignore (Renderer.Compiler.compile_cached c0 "a");
  ignore (Renderer.Compiler.compile_cached c1 "a");
  equal int ~msg:"runs of the toolchain" 2 t.runs

let binaries_are_bytes () =
  let lib = "\x00\xff\n\x7fELF" in
  let table = fresh_table () in
  let c = Renderer.Compiler.v ~cachekey:(fun () -> table) (fun _ -> lib) in
  equal string lib (Renderer.Compiler.compile_cached c "a");
  equal string lib (Renderer.Compiler.compile_cached c "a")

(* The table is asked for when first needed: making the compiler asks nothing,
   and compiles after the first ask nothing more. *)
let asks_for_its_table_once () =
  let asked = ref 0 and table = fresh_table () in
  let cachekey () =
    incr asked;
    table
  in
  let c = Renderer.Compiler.v ~cachekey (build (toolchain ())) in
  equal int ~msg:"made" 0 !asked;
  ignore (Renderer.Compiler.compile_cached c "a");
  ignore (Renderer.Compiler.compile_cached c "b");
  equal (option string) (Some table) (Renderer.Compiler.cachekey c);
  equal int ~msg:"compiled twice" 1 !asked

(* A table that could not be named is asked for again, and nothing is compiled
   meanwhile. *)
let asks_again_after_a_failure () =
  let fails = ref true and table = fresh_table () and t = toolchain () in
  let cachekey () =
    if !fails then raise (Renderer.Compiler.Compile_error "no toolchain")
    else table
  in
  let c = Renderer.Compiler.v ~cachekey (build t) in
  raises (Renderer.Compiler.Compile_error "no toolchain") (fun () ->
      Renderer.Compiler.compile_cached c "a");
  equal int ~msg:"runs of the toolchain" 0 t.runs;
  fails := false;
  equal string "lib:a" (Renderer.Compiler.compile_cached c "a");
  equal (option string) (Some "lib:a") (get table "a")

let rejected src = raise (Renderer.Compiler.Compile_error ("rejected " ^ src))

let errors_are_not_cached () =
  let table = fresh_table () and fails = ref true in
  let compile src = if !fails then rejected src else "lib:" ^ src in
  let c = Renderer.Compiler.v ~cachekey:(fun () -> table) compile in
  raises (Renderer.Compiler.Compile_error "rejected a") (fun () ->
      Renderer.Compiler.compile_cached c "a");
  equal (option string) None (get table "a");
  fails := false;
  equal string "lib:a" (Renderer.Compiler.compile_cached c "a")

let disassembling () =
  let c =
    Renderer.Compiler.v
      ~disassemble:(fun lib -> print_string ("dis " ^ lib))
      Fun.id
  in
  Renderer.Compiler.disassemble c "lib";
  flush stdout;
  equal string "dis lib" (output ())

let disassembling_by_default () =
  Renderer.Compiler.disassemble (Renderer.Compiler.v Fun.id) "lib";
  flush stdout;
  equal string "" (output ())

(* The cache against a model: a table from sources to binaries. *)

let sources = [ ""; "a"; "a\n"; "a\x00"; "b"; "\xc3\xa9" ]
let source = Gen.of_list ~pp:(fun ppf s -> Format.fprintf ppf "%S" s) sources

let foreign =
  Gen.of_list
    ~pp:(fun ppf s -> Format.fprintf ppf "%S" s)
    [ ""; "\x00\xff"; "foreign" ]

type system = { compiler : Renderer.Compiler.t; table : string; t : toolchain }

let cache =
  abstract "c" ~invariant:(fun model s ->
      List.iter
        (fun src ->
          equal (option string)
            ~msg:(Printf.sprintf "entry %S" src)
            (Hashtbl.find_opt model src)
            (get s.table src))
        sources)

(* [compile_cached] returns the binary and whether the toolchain ran. *)
let model_compile_cached model src =
  match Hashtbl.find_opt model src with
  | Some lib -> (lib, false)
  | None ->
      Hashtbl.replace model src ("lib:" ^ src);
      ("lib:" ^ src, true)

let system_compile_cached s src =
  let before = s.t.runs in
  let lib = Renderer.Compiler.compile_cached s.compiler src in
  (lib, s.t.runs > before)

let cache_commands =
  [
    command "v"
      (Gen.unit @-> makes cache)
      (fun () -> Hashtbl.create 8)
      (fun () ->
        let compiler, table, t = cached () in
        { compiler; table; t });
    command "compile_cached"
      (cache ^-> source @-> returns (pair string bool))
      model_compile_cached system_compile_cached;
    command "compile"
      (cache ^-> source @-> returns string)
      (fun _ src -> "lib:" ^ src)
      (fun s src -> Renderer.Compiler.compile s.compiler src);
    command "put"
      (cache ^-> source @-> foreign @-> returns unit)
      (fun model src lib -> Hashtbl.replace model src lib)
      (fun s src lib -> Helpers.Diskcache.put ~table:s.table src lib);
  ]

let compilers =
  group "Compiler"
    [
      test "compile runs the toolchain" (fun () ->
          let t = toolchain () in
          equal string "lib:a"
            (Renderer.Compiler.compile (Renderer.Compiler.v (build t)) "a");
          equal int ~msg:"runs of the toolchain" 1 t.runs);
      test "compile raises what the toolchain rejects" (fun () ->
          raises (Renderer.Compiler.Compile_error "rejected a") (fun () ->
              Renderer.Compiler.compile (Renderer.Compiler.v rejected) "a"));
      test "cachekey is the table" (fun () ->
          let c, table, _ = cached () in
          equal (option string) (Some table) (Renderer.Compiler.cachekey c));
      test "compile_cached compiles a source once and keeps its binary"
        compiles_once;
      test "compile_cached returns the binary the table holds"
        a_cached_binary_wins;
      test "compile_cached compiles a damaged entry anew and replaces it"
        a_damaged_entry_is_compiled_anew;
      test "compile_cached compiles every time with ccache off" without_ccache;
      test "ccache is read when compiling" ccache_is_read_when_compiling;
      test "compile_cached compiles every time with the disk cache disabled"
        with_the_cache_disabled;
      test "compile_cached compiles every time without a cachekey"
        uncached_compiler;
      test "compile_cached keeps two tables apart" tables_are_apart;
      test "the table is asked for once, when first needed"
        asks_for_its_table_once;
      test "a table that raised is asked for again" asks_again_after_a_failure;
      test "compile_cached keeps any bytes" binaries_are_bytes;
      test "compile_cached raises and caches nothing when the toolchain rejects"
        errors_are_not_cached;
      test "compile_cached compiles while ASSERT_COMPILE holds 0" (fun () ->
          let c, _, t = cached () in
          ignore (Renderer.Compiler.compile_cached c "a");
          equal int 1 t.runs);
      stateful "compile_cached behaves as a table of binaries" cache_commands;
      test "disassemble prints with the given function" disassembling;
      test "disassemble prints nothing by default" disassembling_by_default;
    ]

(* ASSERT_COMPILE *)

let under_assert_compile f () =
  if Sys.getenv_opt "ASSERT_COMPILE" <> Some "1" then
    skip ~reason:"runs with ASSERT_COMPILE set" ()
  else f ()

let asserted =
  group ~tags:[ assert_compile ] "ASSERT_COMPILE"
    [
      test "compile_cached refuses to compile, naming the source"
        (under_assert_compile (fun () ->
             let c, table, t = cached () in
             raises_match (Exn.invalid_arg ~substring:"kernel_source")
               (fun () -> Renderer.Compiler.compile_cached c "kernel_source");
             equal int ~msg:"runs of the toolchain" 0 t.runs;
             equal (option string) None (get table "kernel_source")));
      test "compile_cached refuses without a cachekey"
        (under_assert_compile (fun () ->
             rejects (fun () ->
                 Renderer.Compiler.compile_cached
                   (Renderer.Compiler.v Fun.id)
                   "a")));
      test "compile_cached returns a binary the table holds"
        (under_assert_compile a_cached_binary_wins);
      test
        "compile_cached refuses a held binary while the disk cache is disabled"
        (under_assert_compile (fun () ->
             let c, table, _ = cached () in
             Helpers.Diskcache.put ~table "a" "held";
             cache_off (fun () ->
                 rejects (fun () -> Renderer.Compiler.compile_cached c "a"))));
      test "compile still compiles"
        (under_assert_compile (fun () ->
             equal string "a"
               (Renderer.Compiler.compile (Renderer.Compiler.v Fun.id) "a")));
    ]

(* Renderers *)

let defaults () =
  let r = Renderer.v cpu in
  equal string "Renderer" r.name;
  equal string "" r.suffix;
  equal bool true r.supports_float4;
  equal bool true r.has_local;
  equal bool true r.has_shared;
  equal (list int) [ 0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF ] r.global_max;
  equal (list int) [ 0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF ] r.local_max;
  equal (option (list int)) None r.global_prod_max;
  equal int 32768 r.shared_max;
  equal int 0 (List.length r.tensor_cores);
  equal int 0 (List.length r.code_for_op);
  equal (option Uops.uop) None
    (Ops.Pattern_matcher.rewrite r.extra_matcher () (f32 1.))

let given () =
  let target = Result.get_ok (Helpers.Target.of_string "CUDA::sm_80") in
  let render uops = string_of_int (List.length uops) in
  let r =
    Renderer.v ~name:"CUDA" ~suffix:"cu" ~supports_float4:false ~has_local:false
      ~has_shared:false ~global_max:[ 1; 2; 3 ] ~local_max:[ 4; 5; 6 ]
      ~global_prod_max:[ 7; 8; 9 ] ~shared_max:49152 ~tensor_cores:Tc.cuda_sm80
      ~code_for_op:[ (Op.Add, fun xs _ -> String.concat "+" xs) ]
      ~native:(Dtype.equal Dtype.Float32)
      ~render
      ~compiler:(Renderer.Compiler.v String.uppercase_ascii)
      target
  in
  equal string "CUDA" r.name;
  equal string "sm_80" r.target.arch;
  equal string "cu" r.suffix;
  equal (list bool) [ false; false; false ]
    [ r.supports_float4; r.has_local; r.has_shared ];
  equal (list int) [ 1; 2; 3 ] r.global_max;
  equal (list int) [ 4; 5; 6 ] r.local_max;
  equal (option (list int)) (Some [ 7; 8; 9 ]) r.global_prod_max;
  equal int 49152 r.shared_max;
  equal int (List.length Tc.cuda_sm80) (List.length r.tensor_cores);
  equal string "x+y"
    ((List.assoc Op.Add r.code_for_op) [ "x"; "y" ] Dtype.Float32);
  equal bool false (r.native Dtype.Float16);
  equal string "2" (r.render [ f32 1.; f32 2. ]);
  equal string "SRC" (Renderer.Compiler.compile r.compiler "src")

let with_compiler () =
  let r = Renderer.v ~name:"CUDA" ~suffix:"cu" ~has_local:false cpu in
  let r' =
    Renderer.with_compiler (Renderer.Compiler.v String.uppercase_ascii) r
  in
  equal string "SRC" (Renderer.Compiler.compile r'.compiler "src");
  equal string "src" (Renderer.Compiler.compile r.compiler "src");
  equal string "CUDA" r'.name;
  equal string "cu" r'.suffix;
  equal bool false r'.has_local;
  is_true
    (r'.target == r.target && r'.render == r.render && r'.native == r.native)

let renderers =
  group "v"
    [
      test "defaults describe a target that renders nothing" defaults;
      test "keeps the fields it is given" given;
      test "with_compiler replaces the compiler and keeps every other field"
        with_compiler;
      test "native holds for every data type by default" (fun () ->
          let r = Renderer.v cpu in
          List.iter
            (fun dt ->
              is_true ~msg:(Format.asprintf "%a" Dtype.pp dt) (r.native dt))
            Dtype.all);
      test "render raises by default" (fun () ->
          rejects (fun () -> (Renderer.v cpu).render [ f32 1. ]));
      test "the default compiler returns its source and caches nothing"
        (fun () ->
          let c = (Renderer.v cpu).compiler in
          equal string "src" (Renderer.Compiler.compile c "src");
          equal string "src" (Renderer.Compiler.compile_cached c "src");
          equal (option string) None (Renderer.Compiler.cachekey c));
    ]

(* supported_dtypes *)

let names ds = String.concat " " (List.map (Format.asprintf "%a" Dtype.pp) ds)

(* The setting as tinygrad reads it: a [,]-separated list, empty items dropped.
   The cell is the setting's repr. *)
let emulated cell =
  let s = cell "emulated" in
  String.sub s 1 (String.length s - 2)
  |> String.split_on_char ','
  |> List.filter (fun s -> s <> "")

let supported_under names r =
  Setting.context
    [ B (Setting.emulated_dtypes, names) ]
    (fun () -> Renderer.supported_dtypes r)

let without dt = List.filter (fun d -> not (Dtype.equal d dt)) Dtype.all

let supported =
  group "supported_dtypes"
    [
      Golden.cases "supported_dtypes.golden" (fun cell ->
          equal string (cell "supported")
            (names (supported_under (emulated cell) (Renderer.v cpu))));
      test "keeps only native data types, in order" (fun () ->
          let r =
            Renderer.v ~native:(fun d -> not (Dtype.equal d Float16)) cpu
          in
          equal (list Dtypes.dtype) (without Float16) (supported_under [] r));
      test "keeps double when the target lacks long natively" (fun () ->
          let r = Renderer.v ~native:(fun d -> not (Dtype.equal d Int64)) cpu in
          equal (list Dtypes.dtype) (without Int64) (supported_under [] r));
      test "rejects an emulated name that is no data type" (fun () ->
          rejects (fun () -> supported_under [ "nosuchtype" ] (Renderer.v cpu)));
    ]

let () =
  exit
    (run "Tolk.Renderer"
       [
         estimates_of_uops;
         recorded_kernels;
         symbolic_kernels;
         arithmetic_estimates;
         simplification;
         restating;
         compilers;
         asserted;
         renderers;
         supported;
       ])
