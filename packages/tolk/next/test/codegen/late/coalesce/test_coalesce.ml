open Windtrap
open Tolk_next

let cpu = Result.get_ok (Helpers.Target.parse "CPU:CLANG:x86_64,x86-64")
let vector = Renderer.v cpu
let scalar = Renderer.v ~supports_float4:false cpu

let coalesce ?(renderer = vector) sink =
  Coalesce.memory_coalescing sink renderer

let simplify u = Ops.graph_rewrite ~ctx:() u Coalesce.indexing_simplify

(* The cases of an input golden are its sink's sources, in order. *)
let case file cell =
  List.nth (Ops.src (Golden.sink file)) (int_of_string (cell "src"))

(* Accesses *)

let buf = Ops.param ~shape:[ Int 64 ] 0 Float32
let at buf i = Ops.index buf [ i ]
let load_at buf i = Ops.load (at buf (Ops.int i)) []
let loads buf offsets = Ops.sink (List.map (load_at buf) offsets)

let value k =
  Ops.variable ~dtype:Float32 (Printf.sprintf "v%d" k) (`Float (-1.))
    (`Float 1.)

let store_at buf i = Ops.store (at buf (Ops.int i)) (value i)
let stores buf offsets = Ops.sink (List.map (store_at buf) offsets)
let int_of u = Z.to_int (Ops.to_int u)

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
    (Ops.toposort sink)

let runs = slist (pair int int) compare

(* Loads and stores of [width] elements of [dtype], as test_gen_float4.py counts
   them. *)
let vectors ?(dtype = Dtype.Float32) ?(width = 4) sink =
  let wide u = Ops.shape u = [ Ops.Int width ] in
  let count op f =
    List.length
      (List.filter (fun u -> Ops.op u = op && f u) (Ops.toposort sink))
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
    List.length (List.filter (fun u -> Ops.op u = op) (Ops.toposort sink))
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
                  (Ops.max_numel (Ops.nth u 1)))
            (Ops.toposort (coalesce (kernel "grouped_store"))));
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
          equal Uops.uop (Ops.stack (List.init 4 value)) (Ops.nth merged 1));
      test "a merged load's elements are read back by index" (fun () ->
          let lanes = Ops.src (coalesce (loads buf [ 0; 1; 2; 3 ])) in
          let merged = Ops.nth (List.hd lanes) 0 in
          equal (list Uops.uop)
            (List.init 4 (fun k -> Ops.index merged [ Ops.int k ]))
            lanes);
      test "without ALLOW_HALF8, eight halves are two loads of four" (fun () ->
          let half = Ops.param ~shape:[ Int 64 ] 2 Float16 in
          equal runs
            [ (0, 4); (4, 4) ]
            (accesses Op.Load (coalesce (loads half (List.init 8 Fun.id)))));
    ]

let left_alone =
  let same name sink =
    test name (fun () -> equal Uops.uop sink (coalesce sink))
  in
  let f4 = List.init 4 Fun.id in
  group "left alone"
    [
      same "register memory"
        (loads (Ops.placeholder ~slot:3 ~addrspace:Reg [ 16 ] Float32) f4);
      same "a volatile parameter"
        (loads (Ops.param ~volatile:true ~shape:[ Int 16 ] 4 Float32) f4);
      same "a volatile parameter viewed at another type"
        (loads
           (Ops.bitcast
              (Ops.param ~volatile:true ~shape:[ Int 4 ] 4 Uint32)
              Int32)
           f4);
      same "stores to a volatile parameter"
        (stores (Ops.param ~volatile:true ~shape:[ Int 16 ] 4 Float32) f4);
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

(* indexing_simplify *)

let indexing_simplify =
  group "indexing_simplify"
    [
      Golden.cases "indices.golden" (fun cell ->
          equal Uops.uop
            (case "indices_simplified.golden" cell)
            (simplify (case "indices_input.golden" cell)));
    ]

(* Excluded paths (README) *)

let readme =
  group "README"
    [
      test "an image access, through two indices, is not simplified" (fun () ->
          let image = Ops.param ~shape:[ Int 4; Int 4; Int 4 ] 1 Float32 in
          let r0 = Ops.range (Int 4) [ 0 ] and r1 = Ops.range (Int 4) [ 1 ] in
          let gate = Ops.O.(r0 < int 3) in
          let y = Ops.valid Ops.O.(r0 + int 1) gate
          and x = Ops.valid Ops.O.(r1 * int 2) gate in
          let u = Ops.load (Ops.index image [ y; x ]) [] in
          equal Uops.uop u (simplify u));
      test "a DSP renderer merges as any other: aligned runs of four at most"
        (fun () ->
          let dsp = Renderer.v (Result.get_ok (Helpers.Target.parse "DSP")) in
          equal runs
            [ (1, 1); (2, 2); (4, 4); (8, 1) ]
            (accesses Op.Load
               (coalesce ~renderer:dsp
                  (loads buf (List.init 8 (fun k -> k + 1))))));
      test "a DSP renderer leaves 8-bit integers alone" (fun () ->
          let dsp = Renderer.v (Result.get_ok (Helpers.Target.parse "DSP")) in
          let char = Ops.param ~shape:[ Int 64 ] 2 Int8 in
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
      test "eight halves are one load of eight" (fun () ->
          let half = Ops.param ~shape:[ Int 64 ] 2 Float16 in
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
    (run "Tolk_next.Coalesce"
       [ memory_coalescing; indexing_simplify; readme; allow_half8; dmc ])
