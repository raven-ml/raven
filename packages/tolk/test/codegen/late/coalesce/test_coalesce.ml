open Windtrap
open Tolk

let cpu = Result.get_ok (Helpers.Target.of_string "CPU:CLANG:x86_64,x86-64")
let vector = Renderer.v cpu
let scalar = Renderer.v ~supports_float4:false cpu

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

let value k =
  Ops.variable ~dtype:Float32 (Printf.sprintf "v%d" k) (`Float (-1.))
    (`Float 1.)

let store_at buf i = Ops.store (at buf (Ops.int i)) (value i)
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

(* Loads and stores of [width] elements of [dtype], as test_gen_float4.py counts
   them. *)
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

(* Kernels compiled by tinygrad *)

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
  group name
    [
      Golden.graph (name ^ "_coalesced.golden") (fun () ->
          coalesce (kernel name));
      Golden.graph (name ^ "_scalar.golden") (fun () ->
          coalesce ~renderer:scalar (kernel name));
    ]

let float4 name expected =
  test
    (Printf.sprintf "%s has (%d, %d) loads and stores of four floats" name
       (fst expected) (snd expected))
    (fun () -> equal (pair int int) expected (vectors (coalesce (kernel name))))

let gen_float4 =
  group "test_gen_float4.py"
    [
      float4 "basic" (2, 1);
      float4 "multidim" (4, 2);
      float4 "unaligned_load" (0, 1);
      float4 "multidim_unaligned_load" (0, 2);
      float4 "sometimes_unaligned" (0, 0);
      test
        "multidim_sometimes_unaligned: one vector store, a vector load or not"
        (fun () ->
          mem (pair int int)
            (vectors (coalesce (kernel "multidim_sometimes_unaligned")))
            [ (0, 1); (1, 1) ]);
      float4 "expand" (0, 1);
      float4 "heterogeneous" (1, 1);
      float4 "aligned_variable" (2, 1);
      float4 "unaligned_variable" (1, 1);
    ]

let linearizer =
  let count op sink =
    List.length
      (List.filter (fun u -> Ops.op u = op) (Ops.toposort ~calls:Enter sink))
  in
  group "test_linearizer.py"
    [
      test "load_dedup: one to four loads of three overlapping elements"
        (fun () ->
          let n = count Op.Load (coalesce (kernel "load_dedup")) in
          at_least int ~than:1 n;
          at_most int ~than:4 n);
      test "grouped_store: every store to local or global memory is a vector"
        (fun () ->
          List.iter
            (fun u ->
              if Ops.op u = Op.Store && Ops.addrspace (Ops.nth u 0) <> Some Reg
              then
                at_least ~msg:(Graph.to_string u) int ~than:2
                  (Shape.max_numel (Ops.nth u 1)))
            (Ops.toposort ~calls:Enter (coalesce (kernel "grouped_store"))));
    ]

(* Hand-built accesses *)

let hand_built file renderer =
  group file
    [
      Golden.cases "accesses.golden" (fun cell ->
          equal Uops.uop (case file cell)
            (coalesce ~renderer (case "accesses_input.golden" cell)));
    ]

let lengths =
  group "lengths"
    [
      test "four consecutive loads from offset 0 are one load of four"
        (fun () ->
          equal runs
            [ (0, 4) ]
            (accesses Op.Load (coalesce (loads buf [ 0; 1; 2; 3 ]))));
      test "a run starts where its length divides the offset" (fun () ->
          equal runs
            [ (1, 1); (2, 2); (4, 1) ]
            (accesses Op.Load (coalesce (loads buf [ 1; 2; 3; 4 ]))));
      test "three loads are a load of two and a load of one" (fun () ->
          equal runs
            [ (0, 2); (2, 1) ]
            (accesses Op.Load (coalesce (loads buf [ 0; 1; 2 ]))));
      test "offsets with a gap are separate runs" (fun () ->
          equal runs
            [ (0, 2); (3, 1) ]
            (accesses Op.Load (coalesce (loads buf [ 0; 1; 3 ]))));
      test "the order of the accesses does not matter" (fun () ->
          equal runs
            [ (0, 4) ]
            (accesses Op.Load (coalesce (loads buf [ 3; 1; 0; 2 ]))));
      test "four consecutive stores are one store of four" (fun () ->
          equal runs
            [ (0, 4) ]
            (accesses Op.Store (coalesce (stores buf [ 0; 1; 2; 3 ]))));
      test "a merged store stores the stack of the former values" (fun () ->
          let merged =
            List.hd (Ops.src (coalesce (stores buf [ 0; 1; 2; 3 ])))
          in
          equal Uops.uop (Shape.stack (List.init 4 value)) (Ops.nth merged 1));
      test "a merged load's elements are read back by index" (fun () ->
          let lanes = Ops.src (coalesce (loads buf [ 0; 1; 2; 3 ])) in
          let merged = Ops.nth (List.hd lanes) 0 in
          equal (list Uops.uop)
            (List.init 4 (fun k -> Ops.index merged [ Ops.int k ]))
            lanes);
      test "without ALLOW_HALF8, eight halves are two loads of four" (fun () ->
          let half = Shape.param ~shape:[ Int 64 ] 2 Float16 in
          equal runs
            [ (0, 4); (4, 4) ]
            (accesses Op.Load (coalesce (loads half (List.init 8 Fun.id)))));
    ]

(* A buffer whose first element lies [phase] bytes past a multiple of [align]
  . *)
let phased ?(dtype = Dtype.Float32) ?align phase =
  Shape.param ~shape:[ Int 64 ] ~phase ?align 0 dtype

let phases =
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
    ]

let left_alone =
  let same name sink =
    test name (fun () -> equal Uops.uop sink (coalesce sink))
  in
  let f4 = List.init 4 Fun.id in
  group "left alone"
    [
      same "register memory"
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
      rejects
        "a load through an index of two indices (unstated: tinygrad unpacks \
         one)"
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

let memory_coalescing =
  group "memory_coalescing"
    [
      group "kernels" (List.map compiled kernels);
      gen_float4;
      linearizer;
      group "accesses"
        [
          hand_built "accesses_coalesced.golden" vector;
          hand_built "accesses_scalar.golden" scalar;
        ];
      lengths;
      left_alone;
      errors;
    ]

(* The law *)

(* Generated kernels copy runs of consecutive elements of [input] into [output]
   inside a loop of four, adding up one or two runs of [input] per run of
   [output]. A run's elements are at constant offsets or at offsets from the
   loop's index, gated by a condition on it or not; its start is any offset, so
   that runs start aligned or not, and overlap. *)
type place = { scale : int; gated : bool; start : int }
type run = { dst : place; count : int; srcs : place list }

let r = Ops.range (Int 4) [ 0 ]

let index { scale; gated; start } j =
  let i =
    match (scale, start + j) with
    | 0, k -> Ops.int k
    | s, 0 -> Ops.O.(r * int s)
    | s, k -> Ops.O.((r * int s) + int k)
  in
  if gated then Shape.valid i Ops.O.(r < int 2) else i

let size = 40

let generated ?(phases = ((0, 16), (0, 16))) (dtype, runs) =
  let buffer slot (phase, align) =
    Shape.param ~shape:[ Int size ] ~phase ~align slot dtype
  in
  let input = buffer 0 (fst phases) and output = buffer 1 (snd phases) in
  (* A second store of one element under the same key is refused; the first
     store of each element is kept. *)
  let stores, _ =
    List.fold_left
      (fun (stores, seen) { dst; count; srcs } ->
        List.fold_left
          (fun (stores, seen) j ->
            let key = (dst.scale, dst.gated, dst.start + j) in
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

let kernels_of_runs =
  let open Gen in
  let place =
    let+ scale = of_list [ 0; 4; 8 ]
    and+ gated = bool
    and+ start = int_range 0 7 in
    { scale; gated; start }
  in
  let run =
    let+ dst = place
    and+ count = int_range 1 8
    and+ srcs = list ~size:(int_range 1 2) place in
    { dst; count; srcs }
  in
  let dtype = of_list [ Dtype.Float32; Float16; Int32; Int8 ] in
  with_pp
    (fun ppf k -> Format.pp_print_string ppf (Graph.to_string (generated k)))
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

let elements dtype =
  Array.init size (fun i ->
      if Dtype.is_float dtype then `Float (Float.of_int i +. 0.5)
      else `Int (Bigint.of_int ((3 * i) + 1)))

let write = triple int int Dtypes.value

let preserves_writes ?phases renderer (dtype, runs) =
  let k = generated ?phases (dtype, runs) in
  let coalesced = coalesce ~renderer k in
  let merged op =
    List.exists
      (fun u -> Ops.op u = op && Ops.op (Ops.nth u 0) = Op.Shrink)
      (Ops.toposort ~calls:Enter coalesced)
  in
  if renderer.supports_float4 then begin
    cover "loads were merged" (merged Op.Load);
    cover "stores were merged" (merged Op.Store)
  end;
  let writes u =
    Interpreter.writes ~buffers:[ (0, elements dtype); (1, [||]) ] u
  in
  equal (list write) (writes k) (writes coalesced)

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
  let coalesced = coalesce (generated ~phases k) in
  List.iter
    (fun u ->
      if
        (Ops.op u = Op.Load || Ops.op u = Op.Store)
        && Ops.op (Ops.nth u 0) = Op.Shrink
      then begin
        let width = int_of (Ops.nth (Ops.nth u 0) 2) in
        let buf = Ops.nth (Ops.nth u 0) 0 in
        (match Ops.arg (Ops.buf_uop buf) with
        | Param q ->
            at_most ~msg:"bytes within the alignment" int ~than:q.align
              (width * Dtype.itemsize (Ops.dtype buf))
        | _ -> ());
        cover "a vector access" true;
        List.iter
          (fun e ->
            equal ~msg:(Format.asprintf "%a" Ops.pp u) int 0 (e mod width))
          (from_boundary u)
      end)
    (Ops.toposort ~calls:Enter coalesced)

let laws =
  group "laws"
    [
      prop
        "every vector access is aligned to its width and within the alignment, \
         for every phase and alignment"
        phased_kernels aligned;
      prop
        "coalescing preserves the kernel's writes, for every phase and \
         alignment"
        phased_kernels (fun (phases, k) -> preserves_writes ~phases vector k);
      prop "coalescing preserves the kernel's writes" kernels_of_runs
        (preserves_writes vector);
      prop "without vector accesses, coalescing preserves the kernel's writes"
        kernels_of_runs (preserves_writes scalar);
    ]

(* indexing_simplify *)

(* A gather through a pad: element [r] of slot 2 is [xs.(ids.(r - 4))] from
   [r = 4] and [0] before it. Slot 0's index is loaded under the pad's gate. *)
let padded_gather =
  let r = Ops.range (Int 8) [ 0 ] in
  let ids = Shape.param ~shape:[ Int 4 ] 0 Int32 in
  let xs = Shape.param ~shape:[ Int 4 ] 1 Float32 in
  let out = Shape.param ~shape:[ Int 8 ] 2 Float32 in
  let inside = Ops.O.(int 3 < r) in
  let id = Ops.load (Ops.index ids [ Shape.valid Ops.O.(r - int 4) inside ]) [] in
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
      Golden.cases "indices.golden" (fun cell ->
          equal Uops.uop
            (case "indices_simplified.golden" cell)
            (simplify (case "indices_input.golden" cell)));
      (* A load in an index runs whatever the index's gate, so its own index
         keeps its gate. *)
      test "a gather through a pad reads its indices only inside the pad"
        (fun () ->
          equal (list write)
            (gathered_writes padded_gather)
            (gathered_writes (simplify padded_gather)));
    ]

(* Excluded paths (README) *)

let readme =
  group "README"
    [
      test "an image access, through two indices, is not simplified" (fun () ->
          let image = Shape.param ~shape:[ Int 4; Int 4; Int 4 ] 1 Float32 in
          let r0 = Ops.range (Int 4) [ 0 ] and r1 = Ops.range (Int 4) [ 1 ] in
          let gate = Ops.O.(r0 < int 3) in
          let y = Shape.valid Ops.O.(r0 + int 1) gate
          and x = Shape.valid Ops.O.(r1 * int 2) gate in
          let u = Ops.load (Ops.index image [ y; x ]) [] in
          equal Uops.uop u (simplify u));
      test "a DSP renderer merges as any other: aligned runs of four at most"
        (fun () ->
          let dsp =
            Renderer.v (Result.get_ok (Helpers.Target.of_string "DSP"))
          in
          equal runs
            [ (1, 1); (2, 2); (4, 4); (8, 1) ]
            (accesses Op.Load
               (coalesce ~renderer:dsp
                  (loads buf (List.init 8 (fun k -> k + 1))))));
      test "a DSP renderer leaves 8-bit integers alone" (fun () ->
          let dsp =
            Renderer.v (Result.get_ok (Helpers.Target.of_string "DSP"))
          in
          let char = Shape.param ~shape:[ Int 64 ] 2 Int8 in
          equal runs
            [ (0, 1); (1, 1); (2, 1); (3, 1) ]
            (accesses Op.Load
               (coalesce ~renderer:dsp (loads char [ 0; 1; 2; 3 ]))));
    ]

(* The variables are read once per process, so the stanza runs these groups in
   processes of their own, with the variable set. *)

let allow_half8 =
  let half8 name =
    Golden.graph (name ^ "_allow_half8.golden") (fun () ->
        coalesce (kernel name))
  in
  group ~tags:[ "allow_half8" ] "ALLOW_HALF8"
    [
      test "the suite runs this group with ALLOW_HALF8=1" (fun () ->
          equal (option string) (Some "1") (Sys.getenv_opt "ALLOW_HALF8"));
      half8 "half";
      half8 "half8";
      hand_built "accesses_allow_half8.golden" vector;
      prop "coalescing preserves the kernel's writes" kernels_of_runs
        (preserves_writes vector);
      test "eight halves are one load of eight" (fun () ->
          let half = Shape.param ~shape:[ Int 64 ] 2 Float16 in
          equal runs
            [ (0, 8) ]
            (accesses Op.Load (coalesce (loads half (List.init 8 Fun.id)))));
      test "eight floats are still two loads of four" (fun () ->
          equal runs
            [ (0, 4); (4, 4) ]
            (accesses Op.Load (coalesce (loads buf (List.init 8 Fun.id)))));
    ]

let dmc =
  let same name sink =
    test name (fun () -> equal Uops.uop sink (coalesce sink))
  in
  group ~tags:[ "dmc" ] "DMC"
    ([
       test "the suite runs this group with DMC=1" (fun () ->
           equal (option string) (Some "1") (Sys.getenv_opt "DMC"));
       same "the hand-built accesses are left alone"
         (Golden.sink "accesses_input.golden");
       same "a gated load is left alone, and not rejected"
         (Ops.sink
            [
              Ops.load
                (at buf (Ops.int 0))
                [ Ops.float ~dtype:Float32 0.; Ops.bool true ];
            ]);
     ]
    @ List.map
        (fun name -> same (name ^ " is left alone") (kernel name))
        kernels)

let () =
  exit
    (run "Tolk.Coalesce"
       [
         memory_coalescing;
         phases;
         laws;
         indexing_simplify;
         readme;
         allow_half8;
         dmc;
       ])
