open Windtrap
open Tolk

(* The tag of the tests that run with ASSERT_COMPILE set, in a run of their own,
   since a process reads the variable once. *)
let assert_compile = "assert-compile"
let rejects f = raises_match (Exn.invalid_arg ?sub:None) f
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

let range ?(axis = 0) size = Ops.range ~axis_type:Weak (Int size) [ axis ]
let at buf i = Ops.index buf [ i ]
let load buf i = Ops.load (at buf i) []

(* with_storage *)

let accesses = lazy (Array.of_list (Ops.src (Golden.sink "accesses.golden")))
let restatements = lazy (Golden.rows "restatements.golden")

let restated () =
  Ops.sink
    (List.map
       (fun cell ->
         let access = (Lazy.force accesses).(int_of_string (cell "src")) in
         Renderer.with_storage access (Dtypes.dtype_of_cell (cell "dtype")))
       (Lazy.force restatements))

(* An access: an index of storage of any stored type, possibly ordered after a
   store, possibly loaded and the load possibly converted, with the storage's
   type. *)
type access = { access : Ops.t; storage_dtype : Dtype.t }

let pp_access ppf a = Ops.pp ppf a.access

let access =
  let open Gen in
  let storages =
    [
      ("param", fun dtype -> buffer ~dtype 0 16);
      ("buffer", fun dtype -> Ops.new_buffer ~slot:3 "CPU" 16 dtype);
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
     and+ converted = bool
     and+ gated = bool in
     let r = range 16 and s = storage storage_dtype in
     let s =
       if after then Ops.after s [ Ops.store (at (buffer 1 16) r) (f32 1.) ]
       else s
     in
     let index =
       if gated then Ops.index s [ r; Ops.O.(r < int 8) ] else at s r
     in
     let other =
       if Dtype.equal storage_dtype Float32 then Dtype.Int32 else Float32
     in
     let access =
       match (loaded, converted) with
       | false, _ -> index
       | true, false -> Ops.load index []
       | true, true -> Ops.cast (Ops.load index []) other
     in
     { access; storage_dtype })

let is_conversion u = match Ops.op u with Cast | Bitcast -> true | _ -> false

(* [path u] is the nodes from [u] to its storage, following first sources. *)
let rec path u =
  match Ops.op u with
  | Param | Buffer | Alloc -> [ u ]
  | _ -> u :: path (Ops.nth u 0)

let later_sources u = match Ops.src u with [] -> [] | _ :: rest -> rest

let restating =
  group "with_storage"
    [
      group "restates each recorded access as its golden states"
        [ Golden.graph "restated.golden" restated ];
      (* A conversion is on the path 1 time in 4: (3/4)^100 = 3e-13. *)
      prop
        "retypes the storage and the nodes on the way that take their first \
         source's type, keeping every other source and a conversion's type"
        (Gen.pair access Dtypes.stored) (fun (a, dt) ->
          let before = path a.access
          and after = path (Renderer.with_storage a.access dt) in
          cover "a conversion on the path" (List.exists is_conversion before);
          equal int ~msg:"the path's length" (List.length before)
            (List.length after);
          List.iter2
            (fun u v ->
              let expected = if is_conversion u then Ops.dtype u else dt in
              equal Dtypes.dtype
                ~msg:(Format.asprintf "%a" Ops.pp v)
                expected (Ops.dtype v))
            before after;
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
          Law.round_trip Uops.uop Uops.uop
            (fun u -> Renderer.with_storage u dt)
            (fun u -> Renderer.with_storage u a.storage_dtype)
            a.access);
      cases "raises Invalid_argument where the first sources reach no storage"
        ~id:fst
        [
          ("a constant", Ops.int 3);
          ("a range", range 4);
          ("a sum", Ops.O.(f32 1. + f32 2.));
        ]
        (fun (_, u) -> rejects (fun () -> Renderer.with_storage u Dtype.Uint8));
    ]

(* Estimates.zero and Estimates.add *)

let add = Renderer.Estimates.add
let n = Ops.variable "n" (`Int Bigint.one) (`Int (Bigint.of_int 8))
let at_n value s = Shape.sym_infer s [ ("n", value) ]

(* The integer a count holds where [n] is [v], exactly. *)
let value_at v : Ops.sint -> Bigint.t = function
  | Int c -> Bigint.of_int c
  | Sym u ->
      Shape.to_z
        (Ops.substitute ~calls:Skip ~pass:Fixed_point u
           [ (n, Ops.const (`Int (Bigint.of_int v))) ])

let is_int : Ops.sint -> bool = function Int _ -> true | Sym _ -> false

(* Counts drawn small, from the upper half of the ints, where two of them pass
   [max_int], as a node of constants past [max_int], or as a node of [n]. *)
let any_estimates =
  let open Gen in
  let count =
    frequency
      [
        (1, map (fun c -> Ops.Int c) (int_range 0 1000));
        (1, map (fun c -> Ops.Int c) (int_range (max_int / 2) max_int));
        ( 1,
          map
            (fun k ->
              Ops.Sym
                (Ops.const (`Int Bigint.(add (of_int max_int) (of_int k)))))
            (int_range 1 1000) );
        (1, map (fun k -> Ops.Sym Ops.O.(n * int k)) (int_range 0 3));
      ]
  in
  with_pp Ops.pp_estimates
    (let+ ops = count and+ lds = count and+ mem = count in
     ({ ops; lds; mem } : Ops.estimates))

let summed name e0 e1 sum =
  List.iter
    (fun v ->
      equal string
        ~msg:(Printf.sprintf "%s at n = %d" name v)
        (Bigint.to_string (Bigint.add (value_at v e0) (value_at v e1)))
        (Bigint.to_string (value_at v sum)))
    [ 1; 8 ];
  let fits = Bigint.fits_int (Bigint.add (value_at 1 e0) (value_at 1 e1)) in
  satisfies ~msg:name
    ~claim:"an Int iff both counts are Ints and their sum fits one" sint
    (fun s -> is_int s = (is_int e0 && is_int e1 && fits))
    sum

let symbolic_sum () =
  let e : Ops.estimates = { ops = Sym n; lds = Sym n; mem = Sym n } in
  let sum = add e e in
  List.iter
    (fun (name, s) -> equal int ~msg:name 6 (at_n 3 s))
    [ ("ops", sum.ops); ("lds", sum.lds); ("mem", sum.mem) ]

let arithmetic_estimates =
  [
    group "Estimates.zero"
      [
        test "counts nothing" (fun () ->
            equal estimates (counts 0 0 0) Renderer.Estimates.zero);
      ];
    group "Estimates.add"
      [
        (* Two Ints pass max_int in a count 1 time in 16, so in one of a case's
           three counts 1 time in 6: (5/6)^200 = 1e-16. *)
        prop "sums each count exactly" ~count:200
          ~examples:[ (counts max_int 0 0, counts 1 0 0) ]
          (Gen.pair any_estimates any_estimates)
          (fun ((e0 : Ops.estimates), (e1 : Ops.estimates)) ->
            let sum = add e0 e1 in
            let pairs =
              [ (e0.ops, e1.ops); (e0.lds, e1.lds); (e0.mem, e1.mem) ]
            in
            let some p = List.exists p pairs in
            cover "two Ints past max_int"
              (some (fun (c0, c1) ->
                   is_int c0 && is_int c1
                   && not
                        (Bigint.fits_int
                           (Bigint.add (value_at 1 c0) (value_at 1 c1)))));
            cover "an Int and a node"
              (some (fun (c0, c1) -> is_int c0 <> is_int c1));
            cover "two nodes"
              (some (fun (c0, c1) -> not (is_int c0 || is_int c1)));
            summed "ops" e0.ops e1.ops sum.ops;
            summed "lds" e0.lds e1.lds sum.lds;
            summed "mem" e0.mem e1.mem sum.mem);
        test "sums symbolic counts" symbolic_sum;
      ];
  ]

(* Estimates.simplify *)

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
  group "Estimates.simplify"
    [
      cases "gives an integer for a count of constants" ~id:fst
        [
          ("integers", (counts 1 2 3, counts 1 2 3));
          ( "sums of constants",
            let five = Ops.Sym Ops.O.(Ops.int 2 + Ops.int 3) in
            ( ({ ops = five; lds = five; mem = five } : Ops.estimates),
              counts 5 5 5 ) );
        ]
        (fun (_, (e, expected)) ->
          equal estimates expected (Renderer.Estimates.simplify e));
      prop "keeps the value of each count" symbolic_estimates (fun e ->
          equal
            (list (triple int int int))
            (at_every_n e)
            (at_every_n (Renderer.Estimates.simplify e)));
    ]

(* Estimates.of_uops of hand-written kernels *)

let of_uops ?ignore_indexing u =
  Renderer.Estimates.of_uops ?ignore_indexing (Ops.toposort ~calls:Enter u)

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
    let e = of_uops u in
    (e.ops, e.lds)
  in
  equal (pair sint sint) (flops_lds separate) (flops_lds fused);
  equal sint (Int 2) (fst (flops_lds fused))

let tensor_core ~threads =
  let a = Ops.float ~dtype:Dtype.Float16 1. in
  Ops.wmma a a ~acc:(f32 0.) ~dims:(8, 16, 16) ~threads

let tensor_core_product () =
  let ops threads = (of_uops (tensor_core ~threads)).ops in
  equal sint ~msg:"a warp of 32" (Int (2 * 8 * 16 * 16 / 32)) (ops 32);
  equal sint ~msg:"a warp of 64" (Int (2 * 8 * 16 * 16 / 64)) (ops 64)

let not_arithmetic () =
  let buf = buffer 0 4 and out = buffer 1 4 in
  let x = load buf (Ops.int 0) in
  let cast = Ops.cast x Dtype.Float16 and bits = Ops.bitcast x Dtype.Int32 in
  let stored = Ops.store (at out (Ops.int 0)) x in
  equal sint (Int 0) (of_uops (Ops.sink [ stored; cast; bits ])).ops

let lanes () =
  let v = Shape.stack [ f32 1.; f32 2.; f32 3.; f32 4. ] in
  let sum = Ops.O.(v + v) in
  equal sint (Int 4) (Renderer.Estimates.of_uops [ v; sum ]).ops

(* An operation in [inner] runs 10 * 3 times, one after [inner] in [outer] 10
   times, and one after both once. *)
let nested_ranges () =
  let outer = range 10 and inner = range ~axis:1 3 and a = f32 1. in
  let body = Ops.O.(a + a) in
  let ended = Ops.end_ body [ inner ] in
  let after = Ops.neg a in
  let closed = Ops.end_ after [ outer ] in
  let last = Ops.neg body in
  equal sint
    (Int ((10 * 3) + 10 + 1))
    (Renderer.Estimates.of_uops
       [ outer; inner; a; body; ended; after; closed; last ])
      .ops

(* [j < i + 1] inside [i < 3] runs 3 times its greatest count of trips, 3. *)
let a_triangular_loop_counts_its_greatest_trips () =
  let buf = buffer ~dtype:Dtype.Int32 0 6 and i = range 3 in
  let j = Ops.range ~axis_type:Weak (Sym Ops.O.(i + Ops.int 1)) [ 1 ] in
  let idx = Ops.O.((i * (i + Ops.int 1) // Ops.int 2) + j) in
  let st = Ops.store (at buf idx) (Ops.cast j Dtype.Int32) in
  let kernel = Ops.end_ st [ j; i ] in
  equal sint
    (Int (3 * 3 * 4))
    (Renderer.Estimates.of_uops (Ops.toposort ~calls:Skip kernel)).lds

(* [j < x[i]] inside [i < 3], [x] of int64: [j] runs up to 2^63 - 1 times, which
   no [int] count holds. *)
let a_loaded_bound_counts_exactly () =
  let x = buffer ~dtype:Dtype.Int64 1 3 and i = range 3 in
  let j =
    Ops.range ~axis_type:Weak (Sym (Ops.cast (load x i) Dtype.Weak_int)) [ 1 ]
  in
  let body = Ops.O.(f32 1. + Ops.cast j Dtype.Float32) in
  let kernel = Ops.end_ body [ j; i ] in
  let e = Renderer.Estimates.of_uops (Ops.toposort ~calls:Skip kernel) in
  match e.ops with
  | Int n -> failf "%d operations, where 3 * (2^63 - 1) are" n
  | Sym u ->
      equal string
        (Bigint.to_string Bigint.(mul (of_int 3) (pred (shift_left one 63))))
        (Bigint.to_string (Shape.to_z u))

(* [j < x[0]] inside [i < 3], [x] of int64: a bound that reads memory and no
   range counts its greatest value too, 2^63 - 1. [i] opens first, as in the
   kernel's order, though [j]'s bound does not read it. *)
let a_bound_loaded_outside_ranges_counts_its_greatest () =
  let x = buffer ~dtype:Dtype.Int64 1 3 and i = range 3 in
  let bound = Ops.cast (load x (Ops.int 0)) Dtype.Weak_int in
  let j = Ops.range ~axis_type:Weak (Sym bound) [ 1 ] in
  let body = Ops.O.(f32 1. + Ops.cast j Dtype.Float32) in
  let kernel = Ops.end_ body [ j; i ] in
  let inner = List.filter (fun u -> u != i) (Ops.toposort ~calls:Skip kernel) in
  let e = Renderer.Estimates.of_uops (i :: inner) in
  match e.ops with
  | Int n -> failf "%d operations, where 3 * (2^63 - 1) are" n
  | Sym u ->
      equal string
        (Bigint.to_string Bigint.(mul (of_int 3) (pred (shift_left one 63))))
        (Bigint.to_string (Shape.to_z u))

(* [j < x[i]] inside [i < 2^31], [x] of int32: its trips, 2^31 times [x]'s
   greatest value, fit an [int], and the bytes its store writes, four times as
   many, beside the 4 bytes its bound loads on each of [i]'s, do not. *)
let a_count_past_an_int_after_its_ranges () =
  let x = buffer ~dtype:Dtype.Int32 1 4
  and out = buffer ~dtype:Dtype.Int32 0 4 in
  let i = Ops.range ~axis_type:Loop (Int (1 lsl 31)) [ 0 ] in
  let j =
    Ops.range ~axis_type:Loop (Sym (Ops.cast (load x i) Dtype.Weak_int)) [ 1 ]
  in
  let st = Ops.store (at out (Ops.int 0)) (Ops.cast j Dtype.Int32) in
  let kernel = Ops.end_ st [ j; i ] in
  let e = Renderer.Estimates.of_uops (Ops.toposort ~calls:Skip kernel) in
  let greatest = Dtype.Value.to_z (Ops.vmax (Shape.simplify (Ops.nth j 0))) in
  let trips = Bigint.(mul (shift_left one 31) greatest) in
  match e.lds with
  | Int n -> failf "%d bytes, past an int" n
  | Sym u ->
      equal string
        (Bigint.to_string Bigint.(mul (of_int 4) (trips + shift_left one 31)))
        (Bigint.to_string (Shape.to_z u))

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

let malformed =
  let a = f32 1. in
  [
    ( "an end that closes no range",
      fun () ->
        Renderer.Estimates.of_uops [ a; Ops.end_ Ops.O.(a + a) [ range 4 ] ] );
    ( "a backedge that closes no loop",
      fun () ->
        Renderer.Estimates.of_uops
          [ a; Ops.backedge a ~loop:(Ops.loop 0) ~cond:(Ops.bool false) ] );
    ( "a tensor core product without its argument",
      fun () ->
        let h = Ops.float ~dtype:Dtype.Float16 1. in
        of_uops (Ops.v Wmma ~src:[ h; h; f32 0. ]) );
  ]

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
  equal estimates (counts 0 4 4) (of_uops (Ops.sink [ stored ]))

let loads_and_stores_count_apart () =
  let buf = buffer 0 16 and r = range 16 in
  let x = load buf r in
  let stored = Ops.store (at buf r) Ops.O.(x + x) in
  let e = of_uops (Ops.sink [ Ops.end_ stored [ r ] ]) in
  equal sint ~msg:"a read and a write of 64 bytes" (Int 128) e.mem;
  equal sint (Int 128) e.lds

let registers_move_no_bytes () =
  let reg = Shape.alloc ~addrspace:Reg [ Int 4 ] Dtype.Float32 in
  let stored = Ops.store (at reg (Ops.int 0)) (f32 1.) in
  let x = load reg (Ops.int 1) in
  equal estimates (counts 0 0 0) (of_uops (Ops.sink [ stored; x ]))

let shared_memory_is_not_a_parameter () =
  let smem = Shape.alloc ~addrspace:Local [ Int 4 ] Dtype.Float32 in
  let stored = Ops.store (at smem (Ops.int 0)) (f32 1.) in
  equal estimates (counts 0 4 0) (of_uops (Ops.sink [ stored ]))

(* [both u] is [u]'s [ops] without and with [ignore_indexing]. *)
let both u = ((of_uops u).ops, (of_uops ~ignore_indexing:true u).ops)

let indexing () =
  let buf = buffer 0 32 and out = buffer 1 32 and r = range 16 in
  let x = load buf Ops.O.((r * int 2) + int 1) in
  let stored = Ops.store (at out r) Ops.O.(x * x) in
  equal (pair sint sint)
    (Int (16 * 3), Int 16)
    (both (Ops.sink [ Ops.end_ stored [ r ] ]))

let an_index_shared_with_a_value () =
  let out = buffer 0 32 and r = range 16 in
  let twice = Ops.O.(r * int 2) in
  let stored = Ops.store (at out twice) (Ops.cast twice Dtype.Float32) in
  equal (pair sint sint) (Int 16, Int 0)
    (both (Ops.sink [ Ops.end_ stored [ r ] ]))

let shrink_indexing () =
  let storage = buffer 0 32
  and a = Ops.variable "a" (`Int Bigint.zero) (`Int (Bigint.of_int 8)) in
  let shrunk =
    Ops.v Shrink ~src:[ storage; Ops.O.(a + int 1); Ops.O.(a * int 2) ]
  in
  equal (pair sint sint) (Int 2, Int 0) (both (Ops.sink [ Ops.load shrunk [] ]))

(* An index that reads the result of a loop: the loop computes the index, but
   the End or Backedge that closes it bounds what the index's operations are. *)
let an_index_after_a_loop () =
  let acc = Shape.alloc ~addrspace:Reg [ Int 1 ] Dtype.Int32 in
  let cell = at acc (Ops.int 0) in
  let ops closed =
    let total = Ops.load (Ops.after cell [ closed ]) [] in
    let stored = Ops.store (at (buffer 0 16) total) (f32 1.) in
    (of_uops ~ignore_indexing:true (Ops.sink [ stored ])).ops
  in
  let r = Ops.range ~axis_type:Loop (Int 4) [ 0 ] and loop = Ops.loop 1 in
  let update i = Ops.store cell Ops.O.(Ops.load cell [] + i) in
  equal sint ~msg:"a range" (Int 4) (ops (Ops.end_ (update r) [ r ]));
  equal sint ~msg:"a loop" (Int 1)
    (ops (Ops.backedge (update (Ops.int 1)) ~loop ~cond:(Ops.bool false)))

(* Estimates.of_uops of symbolic kernels *)

let symbolic_trip_count () =
  let r = Ops.range ~axis_type:Weak (Sym n) [ 0 ] and a = f32 1. in
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
  let r = Ops.range ~axis_type:Weak (Sym size) [ 0 ] and a = f32 1. in
  let body = Ops.O.(a + a) in
  equal sint (Int 3)
    (Renderer.Estimates.of_uops [ r; a; body; Ops.end_ body [ r ] ]).ops

let symbolic_reads_are_capped () =
  let buf = buffer 0 4 and zero = Ops.int 0 in
  let r = Ops.range ~axis_type:Weak (Sym n) [ 0 ] in
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
  let r = Ops.range ~axis_type:Weak (Sym Ops.O.(gidx + int 3)) [ 0 ]
  and a = f32 1. in
  let body = Ops.O.(a + a) in
  equal sint
    (Int (4 * 3))
    (Renderer.Estimates.of_uops [ gidx; r; a; body; Ops.end_ body [ r ] ]).ops

(* Estimates.of_uops of recorded kernels *)

(* The cases of a kernel golden are its sink's sources, each a linear program
   whose sources are the kernel's nodes in order. kernels.tolk and
   symbolic_kernels.tolk keep their own inputs: the estimates tables' [split]
   column counts the blocks of a host kernel (D80), which upstream's kernels and
   tables do not have. *)
let kernels file = lazy (Array.of_list (Ops.src (Golden.sink file)))
let recorded = kernels "kernels.golden"
let symbolic = kernels "symbolic_kernels.golden"
let kernel file cell = Ops.src (Lazy.force file).(int_of_string (cell "src"))

(* A split kernel's counts are those of one block; the goldens count the block
   that runs the whole loop, of the iterations in the cell "split". *)
let whole cell vars =
  match cell "split" with
  | "-" -> vars
  | extent -> ("block_lo", 0) :: ("block_hi", int_of_string extent) :: vars

let recorded_estimates kernels vars cell =
  let uops = kernel kernels cell in
  let e = Renderer.Estimates.of_uops uops in
  let expect column s =
    equal int ~msg:column
      (int_of_string (cell column))
      (Shape.sym_infer s (whole cell vars))
  in
  expect "ops" e.ops;
  expect "lds" e.lds;
  expect "mem" e.mem;
  expect "ops_ignoring_indexing"
    (Renderer.Estimates.of_uops ~ignore_indexing:true uops).ops

let estimating =
  group "Estimates.of_uops"
    [
      test "an empty kernel is zero" (fun () ->
          equal estimates Renderer.Estimates.zero
            (Renderer.Estimates.of_uops []));
      test "counts each arithmetic operation once" arithmetic;
      test "counts an operation on four lanes four times" lanes;
      test "counts a multiply-add as a multiply and an add"
        multiply_add_is_a_multiply_and_an_add;
      test "counts a tensor core product as 2NMK shared among its threads"
        tensor_core_product;
      test "counts no cast, bitcast, load or store as arithmetic" not_arithmetic;
      test "counts every read in lds and a buffer read again once in mem"
        reads_are_capped_at_the_buffer;
      test "counts the bytes a store writes" a_store_counts_its_value;
      test "counts no bytes for registers" registers_move_no_bytes;
      test "counts shared memory in lds but not in mem"
        shared_memory_is_not_a_parameter;
      test "counts the loads and the stores of a buffer apart in mem"
        loads_and_stores_count_apart;
      test "reads are capped at the buffer for each value of a variable"
        symbolic_reads_are_capped;
      test "counts an operation once per iteration of the ranges around it"
        nested_ranges;
      test "multiplies everything after a hardware index by its size"
        a_hardware_index_multiplies_what_follows;
      test "a symbolic trip count counts at the value of its variable"
        symbolic_trip_count;
      test "counts a loop without trip count as one iteration"
        a_loop_counts_once;
      test "a trip count that depends on a hardware index is taken at index 0"
        trip_count_of_a_hardware_index;
      test "counts a range bounded by another range at its greatest trips"
        a_triangular_loop_counts_its_greatest_trips;
      test "counts a range bounded by a loaded value exactly, past an int"
        a_loaded_bound_counts_exactly;
      test
        "counts a range bounded by a value loaded outside any range at its \
         greatest trips"
        a_bound_loaded_outside_ranges_counts_its_greatest;
      test "ops simplifies a trip count of constants to an Int"
        cancelling_trip_count;
      test "ignore_indexing leaves out the operations of indices" indexing;
      test
        "ignore_indexing leaves out an operation an index shares with a value"
        an_index_shared_with_a_value;
      test "ignore_indexing leaves out the operations of shrinks"
        shrink_indexing;
      test "ignore_indexing counts a loop whose result an index reads"
        an_index_after_a_loop;
      test "counts bytes past an int where the trips fit one"
        a_count_past_an_int_after_its_ranges;
      group "estimates each recorded kernel as its golden states"
        [
          Golden.cases "estimates.golden" (recorded_estimates recorded []);
          Golden.cases "symbolic_estimates.golden" ~key:[ "case"; "n" ]
            (fun cell ->
              recorded_estimates symbolic
                [ ("n", int_of_string (cell "n")) ]
                cell);
        ];
      cases "raises Invalid_argument for a malformed kernel" ~id:fst malformed
        (fun (_, f) -> rejects f);
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
let rejected src = raise (Renderer.Compiler.Compile_error ("rejected " ^ src))

let under_assert_compile f () =
  if Sys.getenv_opt "ASSERT_COMPILE" <> Some "1" then
    skip ~reason:"runs with ASSERT_COMPILE set" ()
  else f ()

let asserted name f = tag assert_compile (test name (under_assert_compile f))

(* A compiler of several sources at once that records each call's sources. *)
let batched () =
  let table = fresh_table () and calls = ref [] in
  let compile srcs =
    calls := srcs :: !calls;
    List.map
      (fun src ->
        if String.starts_with ~prefix:"bad" src then rejected src
        else "lib:" ^ src)
      srcs
  in
  (Renderer.Compiler.batch ~cachekey:(fun () -> table) compile, table, calls)

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
  Helpers.Diskcache.put ~table "a" "held";
  Disk_cache.damage ~table Helpers.cachedb Disk_cache.truncated;
  equal string "lib:a" (Renderer.Compiler.compile_cached c "a");
  equal int ~msg:"runs of the toolchain" 1 t.runs;
  equal (option string) ~msg:"replaced" (Some "lib:a") (get table "a")

(* An entry the process may not read, as in a cache another user shares, is
   compiled anew and replaced. *)
let an_unreadable_entry_is_compiled_anew () =
  if Sys.win32 then skip ~reason:"no file mode forbids reading" ();
  if Unix.geteuid () = 0 then skip ~reason:"root reads any file" ();
  let c, table, t = cached () in
  Helpers.Diskcache.put ~table "a" "held";
  List.iter
    (fun e -> Unix.chmod e 0)
    (Disk_cache.entries ~table Helpers.cachedb);
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

let errors_are_not_cached () =
  let table = fresh_table () and fails = ref true in
  let compile src = if !fails then rejected src else "lib:" ^ src in
  let c = Renderer.Compiler.v ~cachekey:(fun () -> table) compile in
  raises (Renderer.Compiler.Compile_error "rejected a") (fun () ->
      Renderer.Compiler.compile_cached c "a");
  equal (option string) None (get table "a");
  fails := false;
  equal string "lib:a" (Renderer.Compiler.compile_cached c "a")

let compiles_all_at_once () =
  let c, table, calls = batched () in
  Helpers.Diskcache.put ~table "b" "held";
  equal (list string)
    [ "lib:a"; "held"; "lib:c"; "lib:a" ]
    (Renderer.Compiler.compile_all c [ "a"; "b"; "c"; "a" ]);
  equal
    (list (list string))
    [ [ "a"; "c" ] ]
    (List.map (List.sort String.compare) !calls);
  equal (option string) (Some "lib:c") (get table "c")

let compiles_nothing_held () =
  let c, table, calls = batched () in
  Helpers.Diskcache.put ~table "a" "held";
  equal (list string) [ "held"; "held" ]
    (Renderer.Compiler.compile_all c [ "a"; "a" ]);
  equal int ~msg:"calls" 0 (List.length !calls)

(* As [List.map (compile_cached c)]: "a" is compiled and kept before "bad1"
   raises, and "bad0" is not reached. *)
let reports_the_first_rejected () =
  let c, table, _ = batched () in
  raises (Renderer.Compiler.Compile_error "rejected bad1") (fun () ->
      Renderer.Compiler.compile_all c [ "a"; "bad1"; "bad0" ]);
  equal (option string) (Some "lib:a") (get table "a")

(* A compiler whose call of several sources fails for none of them. *)
let fails_together () =
  let c, table, _ = batched () in
  let c =
    Renderer.Compiler.batch
      ~cachekey:(fun () -> table)
      (fun srcs ->
        if List.length srcs > 1 then
          raise (Renderer.Compiler.Compile_error "together")
        else List.map (fun src -> Renderer.Compiler.compile c src) srcs)
  in
  equal (list string) [ "lib:a"; "lib:b" ]
    (Renderer.Compiler.compile_all c [ "a"; "b" ]);
  equal (option string) (Some "lib:b") (get table "b")

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
    command "compile_all"
      (cache
      ^-> Gen.list ~size:(Gen.int_range 0 4) source
      @-> returns (pair (list string) int))
      (fun model srcs ->
        let missing =
          List.sort_uniq String.compare
            (List.filter (fun src -> not (Hashtbl.mem model src)) srcs)
        in
        let libs =
          List.map (fun src -> fst (model_compile_cached model src)) srcs
        in
        (libs, List.length missing))
      (fun s srcs ->
        let before = s.t.runs in
        let libs = Renderer.Compiler.compile_all s.compiler srcs in
        (libs, s.t.runs - before));
    command "compile"
      (cache ^-> source @-> returns string)
      (fun _ src -> "lib:" ^ src)
      (fun s src -> Renderer.Compiler.compile s.compiler src);
    command "put"
      (cache ^-> source @-> foreign @-> returns unit)
      (fun model src lib -> Hashtbl.replace model src lib)
      (fun s src lib -> Helpers.Diskcache.put ~table:s.table src lib);
  ]

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

let kept_held () =
  let c, table, _ = cached () in
  Helpers.Diskcache.put ~table "a" "held";
  equal (option string) (Some "held") (Renderer.Compiler.kept c "a")

let kept_nothing () =
  let c, _, t = cached () in
  equal (option string) None (Renderer.Compiler.kept c "a");
  equal int ~msg:"runs of the toolchain" 0 t.runs

let kept_damaged () =
  let c, table, _ = cached () in
  Helpers.Diskcache.put ~table "a" "held";
  Disk_cache.damage ~table Helpers.cachedb Disk_cache.truncated;
  equal (option string) None (Renderer.Compiler.kept c "a")

let kept_without_ccache () =
  let c, table, _ = cached () in
  Helpers.Diskcache.put ~table "a" "held";
  equal (option string) None
    (ccache_off (fun () -> Renderer.Compiler.kept c "a"))

(* A batch compiler whose function returns one binary fewer or one more than it
   is given sources. *)
let miscounting =
  [
    ("one fewer", fun srcs -> List.tl (List.map (( ^ ) "lib:") srcs));
    ("one more", fun srcs -> "extra" :: List.map (( ^ ) "lib:") srcs);
  ]

let compilers =
  [
    group "Compiler.v"
      [
        test "compiles a source with its function" (fun () ->
            let t = toolchain () in
            equal string "lib:a"
              (Renderer.Compiler.compile (Renderer.Compiler.v (build t)) "a");
            equal int ~msg:"runs of the toolchain" 1 t.runs);
        test "asks for its table once, when first needed"
          asks_for_its_table_once;
        test "asks again for a table that raised" asks_again_after_a_failure;
      ];
    group "Compiler.batch"
      [
        cases
          "makes compile, compile_cached and compile_all raise \
           Invalid_argument where its function returns other than one binary \
           per source"
          ~id:fst miscounting (fun (_, f) ->
            let c = Renderer.Compiler.batch f in
            rejects (fun () -> Renderer.Compiler.compile c "a");
            rejects (fun () -> Renderer.Compiler.compile_cached c "a");
            rejects (fun () ->
                Renderer.Compiler.compile_all c [ "a"; "b"; "c" ]));
      ];
    group "Compiler.cachekey"
      [
        test "is the table's name" (fun () ->
            let c, table, _ = cached () in
            equal (option string) (Some table) (Renderer.Compiler.cachekey c));
        test
          "names the table of a compiler made with ccache off, and keeps in it \
           once ccache holds"
          ccache_is_read_when_compiling;
        test "raises what its function raises" (fun () ->
            let c =
              Renderer.Compiler.v
                ~cachekey:(fun () -> failwith "no toolchain")
                Fun.id
            in
            raises (Failure "no toolchain") (fun () ->
                Renderer.Compiler.cachekey c));
      ];
    group "Compiler.compile"
      [
        test "raises what the toolchain rejects" (fun () ->
            raises (Renderer.Compiler.Compile_error "rejected a") (fun () ->
                Renderer.Compiler.compile (Renderer.Compiler.v rejected) "a"));
        asserted "compiles under ASSERT_COMPILE" (fun () ->
            equal string "a"
              (Renderer.Compiler.compile (Renderer.Compiler.v Fun.id) "a"));
      ];
    group "Compiler.kept"
      [
        test "is the binary the table holds" kept_held;
        test "is none for a source the table lacks, and compiles nothing"
          kept_nothing;
        test "is none for a damaged entry" kept_damaged;
        test "is none while ccache is off" kept_without_ccache;
        test "is none without a cachekey" (fun () ->
            equal (option string) None
              (Renderer.Compiler.kept (Renderer.Compiler.v Fun.id) "a"));
      ];
    group "Compiler.compile_cached"
      [
        test "compiles a source once and keeps its binary" compiles_once;
        test "returns the binary the table holds" a_cached_binary_wins;
        test "compiles a damaged entry anew and replaces it"
          a_damaged_entry_is_compiled_anew;
        test "compiles an unreadable entry anew and replaces it"
          an_unreadable_entry_is_compiled_anew;
        test "compiles every time with ccache off" without_ccache;
        test "compiles every time with the disk cache disabled"
          with_the_cache_disabled;
        test "compiles every time without a cachekey" uncached_compiler;
        test "keeps two tables apart" tables_are_apart;
        test "raises and caches nothing when the toolchain rejects"
          errors_are_not_cached;
        stateful "behaves as a table of binaries" cache_commands;
        asserted "refuses to compile under ASSERT_COMPILE, naming the source"
          (fun () ->
            let c, table, t = cached () in
            raises_match (Exn.invalid_arg ~sub:"kernel_source") (fun () ->
                Renderer.Compiler.compile_cached c "kernel_source");
            equal int ~msg:"runs of the toolchain" 0 t.runs;
            equal (option string) None (get table "kernel_source"));
        asserted "refuses without a cachekey under ASSERT_COMPILE" (fun () ->
            raises_match (Exn.invalid_arg ~sub:"uncached_source") (fun () ->
                Renderer.Compiler.compile_cached
                  (Renderer.Compiler.v Fun.id)
                  "uncached_source"));
        asserted "returns a binary the table holds under ASSERT_COMPILE"
          a_cached_binary_wins;
        asserted
          "refuses a held binary while the disk cache is disabled under \
           ASSERT_COMPILE" (fun () ->
            let c, table, _ = cached () in
            Helpers.Diskcache.put ~table "a" "held";
            cache_off (fun () ->
                rejects (fun () -> Renderer.Compiler.compile_cached c "a")));
      ];
    group "Compiler.compile_all"
      [
        test "compiles the sources the table lacks in one call, each once"
          compiles_all_at_once;
        test "calls nothing when the table holds every source"
          compiles_nothing_held;
        test
          "raises for the first source the toolchain rejects, keeping those \
           before it"
          reports_the_first_rejected;
        test
          "returns each binary where a call of several sources raises though \
           none raises alone"
          fails_together;
        asserted "refuses under ASSERT_COMPILE, naming the first source missing"
          (fun () ->
            let c, table, _ = cached () in
            Helpers.Diskcache.put ~table "held" "lib";
            match
              Renderer.Compiler.compile_all c [ "held"; "z_source"; "y_source" ]
            with
            | _ -> fail "compiled under ASSERT_COMPILE"
            | exception Invalid_argument msg ->
                contains ~sub:"z_source" msg;
                not_contains ~sub:"y_source" msg);
      ];
    group "Compiler.disassemble"
      [
        test "prints with the given function" disassembling;
        test "prints nothing by default" disassembling_by_default;
      ];
  ]

(* Renderers *)

let probe = Ops.int 1

let target_text (t : Helpers.Target.t) =
  Printf.sprintf "%s %s %s" t.device t.renderer t.arch

(* [fields r] is each field of [r] as text: a function as what it gives. *)
let fields (r : Renderer.t) =
  let b = string_of_bool and i = string_of_int in
  let ints l = String.concat " " (List.map i l) in
  let holding p =
    String.concat " "
      (List.filter_map
         (fun dt ->
           if p dt then Some (Format.asprintf "%a" Dtype.pp dt) else None)
         Dtype.all)
  in
  let given f =
    match f () with
    | s -> s
    | exception Invalid_argument _ -> "raises Invalid_argument"
  in
  [
    ("name", r.name);
    ("target", target_text r.target);
    ("suffix", r.suffix);
    ("supports_float4", b r.supports_float4);
    ("half8", b r.half8);
    ("byte_words", b r.byte_words);
    ("full_rate_subnormals", b r.full_rate_subnormals);
    ( "registers",
      match r.registers () with
      | None -> "none"
      | Some { Renderer.count; bytes } ->
          Printf.sprintf "%d of %d bytes" count bytes );
    ("thread_registers", Option.fold ~none:"none" ~some:i r.thread_registers);
    ("vector_lanes", i r.vector_lanes);
    ("has_local", b r.has_local);
    ("has_shared", b r.has_shared);
    ("global_max", ints r.global_max);
    ("local_max", ints r.local_max);
    ("local_threads_max", i r.local_threads_max);
    ("simd_threads", i r.simd_threads);
    ("global_prod_max", Option.fold ~none:"none" ~some:ints r.global_prod_max);
    ("shared_max", i r.shared_max);
    ("async_copy", b r.async_copy);
    ( "tensor_cores",
      String.concat "; " (List.map (Format.asprintf "%a" Tc.pp) r.tensor_cores)
    );
    ("per_byte_first", i r.per_byte_first);
    ("core_threads", i r.core_threads);
    ("stream_threads", i r.stream_threads);
    ("local_workgroups", i r.local_workgroups);
    ("units", i r.units);
    ("in_flight", i r.in_flight);
    ("l2", i r.l2);
    ( "extra_matcher",
      Option.fold ~none:"no rewrite"
        ~some:(Format.asprintf "%a" Ops.pp)
        (Ops.Pattern_matcher.rewrite r.extra_matcher () probe) );
    ("launch_members", i r.launch_members);
    ("launch_args", i r.launch_args);
    ("launch_locals", b r.launch_locals);
    ("packed_args", b r.packed_args);
    ( "code_for_op",
      String.concat "; "
        (List.map
           (fun (op, f) ->
             Format.asprintf "%a: %s" Op.pp op (f [ "x"; "y" ] Dtype.Float32))
           r.code_for_op) );
    ("native", holding r.native);
    ("fma", holding r.fma);
    ("render", given (fun () -> r.render [ probe ]));
    ( "compiler",
      Printf.sprintf "%s, %s"
        (Renderer.Compiler.compile r.compiler "src")
        (Option.value ~default:"no cachekey"
           (Renderer.Compiler.cachekey r.compiler)) );
  ]

let names ds = String.concat " " (List.map (Format.asprintf "%a" Dtype.pp) ds)
let all_dtypes = names Dtype.all

(* The defaults v's documentation lists. *)
let defaults =
  [
    ("name", "Renderer");
    ("target", target_text cpu);
    ("suffix", "");
    ("supports_float4", "true");
    ("half8", "false");
    ("byte_words", "false");
    ("full_rate_subnormals", "false");
    ("registers", "none");
    ("thread_registers", "none");
    ("vector_lanes", "1");
    ("has_local", "true");
    ("has_shared", "true");
    ("global_max", "2415919103 2415919103 2415919103");
    ("local_max", "2415919103 2415919103 2415919103");
    ("local_threads_max", "1024");
    ("simd_threads", "32");
    ("global_prod_max", "none");
    ("shared_max", "32768");
    ("async_copy", "false");
    ("tensor_cores", "");
    ("per_byte_first", "64");
    ("core_threads", "16384");
    ("stream_threads", "32768");
    ("local_workgroups", "256");
    ("units", "0");
    ("in_flight", "0");
    ("l2", "0");
    ("extra_matcher", "no rewrite");
    ("launch_members", string_of_int max_int);
    ("launch_args", string_of_int max_int);
    ("launch_locals", "true");
    ("packed_args", "false");
    ("code_for_op", "");
    ("native", all_dtypes);
    ("fma", names [ Dtype.Float32; Dtype.Float64 ]);
    ("render", "raises Invalid_argument");
    ("compiler", "src, no cachekey");
  ]

let sm_80 = Result.get_ok (Helpers.Target.of_string "CUDA::sm_80")

let doubled =
  Ops.Pattern_matcher.v (fun () ->
      [
        Ops.Pattern_matcher.rule (Ops.Upat.op ~name:"x" Op.Const) (fun m ->
            Some Ops.O.(m "x" + m "x"));
      ])

(* Every field given, each other than its default and than any other field of
   its type. *)
let given () =
  Renderer.v ~name:"CUDA" ~suffix:"cu" ~supports_float4:false ~half8:true
    ~byte_words:true ~full_rate_subnormals:true
    ~registers:(fun () -> Some { Renderer.count = 32; bytes = 64 })
    ~thread_registers:255 ~vector_lanes:4 ~has_local:false ~has_shared:false
    ~global_max:[ 1; 2; 3 ] ~local_max:[ 4; 5; 6 ] ~local_threads_max:512
    ~simd_threads:64 ~global_prod_max:[ 7; 8; 9 ] ~shared_max:49152
    ~async_copy:true ~tensor_cores:Tc.cuda_sm75 ~per_byte_first:65
    ~core_threads:16385 ~stream_threads:32769 ~local_workgroups:257 ~units:100
    ~in_flight:101 ~l2:102 ~extra_matcher:doubled ~launch_members:103
    ~launch_args:104 ~launch_locals:false ~packed_args:true
    ~code_for_op:[ (Op.Add, fun xs _ -> String.concat "+" xs) ]
    ~native:(Dtype.equal Dtype.Float32)
    ~fma:(Dtype.equal Dtype.Float16)
    ~render:(fun uops -> string_of_int (List.length uops))
    ~compiler:(Renderer.Compiler.v String.uppercase_ascii)
    sm_80

let given_fields =
  [
    ("name", "CUDA");
    ("target", target_text sm_80);
    ("suffix", "cu");
    ("supports_float4", "false");
    ("half8", "true");
    ("byte_words", "true");
    ("full_rate_subnormals", "true");
    ("registers", "32 of 64 bytes");
    ("thread_registers", "255");
    ("vector_lanes", "4");
    ("has_local", "false");
    ("has_shared", "false");
    ("global_max", "1 2 3");
    ("local_max", "4 5 6");
    ("local_threads_max", "512");
    ("simd_threads", "64");
    ("global_prod_max", "7 8 9");
    ("shared_max", "49152");
    ("async_copy", "true");
    ( "tensor_cores",
      String.concat "; " (List.map (Format.asprintf "%a" Tc.pp) Tc.cuda_sm75) );
    ("per_byte_first", "65");
    ("core_threads", "16385");
    ("stream_threads", "32769");
    ("local_workgroups", "257");
    ("units", "100");
    ("in_flight", "101");
    ("l2", "102");
    ("extra_matcher", Format.asprintf "%a" Ops.pp Ops.O.(probe + probe));
    ("launch_members", "103");
    ("launch_args", "104");
    ("launch_locals", "false");
    ("packed_args", "true");
    ("code_for_op", Format.asprintf "%a: x+y" Op.pp Op.Add);
    ("native", names [ Dtype.Float32 ]);
    ("fma", names [ Dtype.Float16 ]);
    ("render", "1");
    ("compiler", "SRC, no cachekey");
  ]

let renderers =
  [
    group "v"
      [
        test "defaults describe a target that renders nothing" (fun () ->
            equal (list (pair string string)) defaults (fields (Renderer.v cpu)));
        test "keeps the fields it is given" (fun () ->
            equal (list (pair string string)) given_fields (fields (given ())));
      ];
    group "with_compiler"
      [
        test "replaces the compiler and keeps every other field" (fun () ->
            let r = given () in
            let r' =
              Renderer.with_compiler
                (Renderer.Compiler.v String.lowercase_ascii)
                r
            in
            let without_compiler = List.remove_assoc "compiler" in
            equal
              (list (pair string string))
              (without_compiler (fields r))
              (without_compiler (fields r'));
            equal string "src" (Renderer.Compiler.compile r'.compiler "SRC"));
      ];
  ]

(* takes_core *)

(* A float32 core rounds its inputs to TF32 on CUDA and NV, which a kernel takes
   only under ALLOW_TF32; every other core, and a float32 core of another
   device, it takes. *)
let cores_taken =
  let taken device allow =
    let target =
      {
        Helpers.Target.device;
        renderer = device;
        arch = "";
        interface = "";
        indices = "";
      }
    in
    let r = Renderer.v ~tensor_cores:Tc.cuda_sm89 target in
    Setting.context
      [ Setting.B (Setting.allow_tf32, allow) ]
      (fun () ->
        List.map
          (fun (tc : Tc.t) -> (tc.dtype_in, Renderer.takes_core r tc))
          r.tensor_cores)
  in
  let expected device allow =
    List.map
      (fun (tc : Tc.t) ->
        let tf32 =
          Dtype.equal tc.dtype_in Float32 && List.mem device [ "CUDA"; "NV" ]
        in
        (tc.dtype_in, allow || not tf32))
      Tc.cuda_sm89
  in
  group "takes_core"
    [
      cases
        "takes every core but a float32 one on CUDA or NV, which it takes \
         under ALLOW_TF32"
        ~id:(fun (device, allow) ->
          Printf.sprintf "%s%s" device
            (if allow then " under ALLOW_TF32" else ""))
        (List.concat_map
           (fun d -> [ (d, false); (d, true) ])
           [ "CUDA"; "NV"; "AMD"; "METAL" ])
        (fun (device, allow) ->
          equal
            (list (pair Dtypes.dtype bool))
            (expected device allow) (taken device allow));
    ]

(* supported_dtypes *)

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
      group "is the data types its golden states for each emulated setting"
        [
          Golden.cases "supported_dtypes.golden" (fun cell ->
              equal string (cell "supported")
                (names (supported_under (emulated cell) (Renderer.v cpu))));
        ];
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
       ((restating :: arithmetic_estimates)
       @ [ simplification; estimating ]
       @ compilers @ renderers @ [ cores_taken; supported ]))
