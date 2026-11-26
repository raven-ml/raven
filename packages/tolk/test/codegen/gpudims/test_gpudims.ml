(* Tests of Tolk.Gpudims: the launch dimensions of a kernel's loops fit the
   target's bounds and number every iteration exactly once. Goldens are recorded
   by gen/codegen/gpudims.py. *)

open Windtrap
open Tolk

let i n = `Int (Bigint.of_int n)

let v ?(dtype = Dtype.Weak_int) name lo hi =
  Ops.variable ~dtype name (i lo) (i hi)

let ints = List.map (fun n -> Ops.Int n)
let target = Result.get_ok (Helpers.Target.of_string "")

let renderer ?global_max ?local_max ?global_prod_max () =
  Renderer.v ?global_max ?local_max ?global_prod_max target

let grouped ?reverse ?(prefix = "gidx") dims max_sizes =
  Gpudims.grouped_dims ?reverse prefix dims max_sizes

let rejects f = raises_match (Exn.invalid_arg ?sub:None) f

(* [idxs] with their hardware ranges made hardware indices. *)
let to_special idxs =
  Ops.src
    (Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() (Ops.sink idxs)
       (After_sources Gpudims.pm_range_to_special))

(* The hardware indices under [idxs], once made indices, with their sizes, by
   name. *)
let specials idxs =
  List.concat_map (Ops.toposort ~calls:Enter) (to_special idxs)
  |> List.filter (fun u -> Ops.op u = Special)
  |> List.sort_uniq Ops.compare
  |> List.map (fun u ->
      match (Ops.arg u, Ops.src u) with
      | String name, [ size ] -> (name, Shape.to_z size)
      | _ -> failf "%a is not a hardware index" Ops.pp u)
  |> List.sort (fun (n0, _) (n1, _) -> String.compare n0 n1)

(* Laws *)

(* [numbers_once dims idxs] checks that the loop indices [idxs] of loops of
   sizes [dims], over every value of their hardware indices, give each flat
   iteration of the loops exactly once. *)
let numbers_once dims idxs =
  let total = List.fold_left ( * ) 1 dims in
  let strides =
    List.fold_right (fun d acc -> (d * List.hd acc) :: acc) dims [ 1 ]
    |> List.tl
  in
  let seen = Array.make total false in
  let indices = to_special idxs in
  let rec each bound = function
    | [] ->
        let vars = List.map (fun (name, v) -> (name, i v)) bound in
        let flat =
          List.fold_left2
            (fun acc idx stride ->
              match Interpreter.eval ~vars idx with
              | `Int n -> acc + (Bigint.to_int n * stride)
              | c ->
                  failf "an index is an integer, not %a"
                    (Testable.pp Dtypes.const) c)
            0 indices strides
        in
        if flat < 0 || flat >= total then
          failf "iteration %d is outside the %d iterations" flat total;
        if seen.(flat) then failf "iteration %d is numbered twice" flat;
        seen.(flat) <- true
    | (name, size) :: rest ->
        for v = 0 to Bigint.to_int size - 1 do
          each ((name, v) :: bound) rest
        done
  in
  each [] (specials idxs);
  Array.iteri
    (fun n hit -> if not hit then failf "iteration %d is never numbered" n)
    seen

(* [fits max_sizes idxs] checks that each hardware index under [idxs] has a
   bound and is within it. A size of 1 needs no index. The first axis is exempt
   when there are three: a split of the last axis moves its divisor onto the
   first, which is not checked again, as the golden's case (5, 12, 7) under (8,
   4, 16) shows. *)
let fits max_sizes idxs =
  List.iter
    (fun (name, size) ->
      let axis = int_of_string (String.sub name 4 (String.length name - 4)) in
      match List.nth_opt max_sizes axis with
      | Some _ when axis = 0 && List.length max_sizes = 3 -> ()
      | Some bound -> at_most ~msg:name int ~than:bound (Bigint.to_int size)
      | None -> failf "%s has no bound" name)
    (specials idxs)

(* Goldens *)

let tuple_of_cell s =
  String.sub s 1 (String.length s - 2)
  |> String.split_on_char ',' |> List.map String.trim
  |> List.filter (( <> ) "")
  |> List.map int_of_string

let bounds_of_cell = function "None" -> None | s -> Some (tuple_of_cell s)

let bool_of_cell = function
  | "True" -> true
  | "False" -> false
  | s -> invalid_arg s

let render_specials idxs =
  String.concat " "
    (List.map
       (fun (n, s) -> Printf.sprintf "%s=%s" n (Bigint.to_string s))
       (specials idxs))

let render_idxs idxs =
  "["
  ^ String.concat ", "
      (List.map (Render.render ~simplify:false) (to_special idxs))
  ^ "]"

let grouped_dims_golden =
  Golden.cases ~key:[ "dims"; "max_sizes"; "reverse" ] "grouped_dims.golden"
    (fun cell ->
      let dims = tuple_of_cell (cell "dims") in
      let max_sizes = bounds_of_cell (cell "max_sizes") in
      let reverse = bool_of_cell (cell "reverse") in
      let run () = grouped ~reverse (ints dims) max_sizes in
      match cell "specials" with
      | "RuntimeError" | "IndexError" -> rejects run
      | expected ->
          let idxs = run () in
          equal string expected (render_specials idxs);
          equal string (cell "idxs") (render_idxs idxs))

(* Launches of at most 2^12 iterations, which a law enumerates. *)
let most_iterations = 4096

let small_enough dims =
  Bigint.leq
    (List.fold_left (fun p d -> Bigint.(p * of_int d)) Bigint.one dims)
    (Bigint.of_int most_iterations)

(* The golden's launches that fit, which the law takes as examples. *)
let golden_launches =
  List.filter_map
    (fun cell ->
      let dims = tuple_of_cell (cell "dims") in
      if String.contains (cell "specials") '=' && small_enough dims then
        Some
          ( dims,
            bounds_of_cell (cell "max_sizes"),
            bool_of_cell (cell "reverse") )
      else None)
    (Golden.rows "grouped_dims.golden")

let sizes = [ 1; 2; 3; 4; 5; 6; 7; 8; 9; 12; 16; 23; 32; 64 ]

(* One to five sizes whose product is at most [drawn_iterations]: the golden's
   launches reach the larger ones. *)
let drawn_iterations = 1024

let launch =
  let open Gen in
  let rec dims k product =
    if k = 0 then constant []
    else
      let* d =
        of_list ~pp:Format.pp_print_int
          (List.filter (fun d -> product * d <= drawn_iterations) sizes)
      in
      let+ rest = dims (k - 1) (product * d) in
      d :: rest
  in
  let* n = int_range 1 5 in
  dims n 1

let grouping_case =
  Gen.(
    triple launch
      (map Option.some
         (frequency
            [
              (1, list ~size:(int_range 1 3) (int_range 1 64));
              (2, list ~size:(constant 3) (of_list [ 2; 4; 8; 16; 32 ]));
            ]))
      bool)

let grouped_dims_law =
  prop ~count:300 ~examples:golden_launches
    "numbers each iteration once within its bounds, or refuses" grouping_case
    (fun (dims, max_sizes, reverse) ->
      let laid =
        match grouped ~reverse (ints dims) max_sizes with
        | idxs -> Some idxs
        | exception Invalid_argument _ -> None
      in
      let axes = Option.map (fun idxs -> List.length (specials idxs)) laid in
      let loops = List.length dims in
      cover "refuses" (laid = None);
      cover "merges" (Option.fold ~none:false ~some:(fun n -> n < loops) axes);
      cover "splits" (Option.fold ~none:false ~some:(fun n -> n > loops) axes);
      Option.iter
        (fun idxs ->
          numbers_once dims idxs;
          Option.iter (fun m -> fits m idxs) max_sizes)
        laid)

let hardware axis n = Ops.range ~axis_type:Global (Int n) [ axis ]

let direct_dims =
  test "a loop that keeps its own axis is that hardware index" (fun () ->
      let idxs = grouped (ints [ 2; 3; 4; 5 ]) (Some [ 16; 16; 16 ]) in
      equal (list Uops.uop)
        [ hardware 1 4; hardware 2 5 ]
        [ List.nth idxs 2; List.nth idxs 3 ])

let untouched =
  test "without bounds, every loop is its own hardware index" (fun () ->
      equal (list Uops.uop)
        (List.mapi hardware [ 2; 3; 4; 5 ])
        (grouped (ints [ 2; 3; 4; 5 ]) None))

let graph_of name idxs =
  Golden.graph (name ^ ".golden") (fun () -> Ops.sink (idxs ()))

let symbolic =
  [
    graph_of "grouped_symbolic_crosses_a_limit_by_merging" (fun () ->
        grouped [ Int 1; Sym (v "n" 1 4) ] (Some [ 4; 3 ]));
    graph_of "grouped_symbolic_merges_into_a_symbolic_size" (fun () ->
        grouped [ Sym (v "n" 1 4); Sym (v "m" 1 8) ] (Some [ 64 ]));
    graph_of "grouped_symbolic_keeps_a_fitting_size" (fun () ->
        grouped [ Sym (v "n" 1 16) ] (Some [ 16; 16; 16 ]));
    graph_of "grouped_symbolic_without_bounds" (fun () ->
        grouped [ Sym (v "n" 1 10) ] None);
    graph_of "grouped_symbolic_merges_committed_sizes" (fun () ->
        grouped
          [ Sym (v ~dtype:Int16 "a" 2 4); Sym (v ~dtype:Int32 "b" 2 4) ]
          (Some [ 16 ]));
  ]

let symbolic_failures =
  Golden.cases "grouped_symbolic_failures.golden" (fun cell ->
      let lo = match cell "case" with "split_at_its_maximum" -> 1 | _ -> 17 in
      rejects (fun () -> grouped [ Sym (v "n" lo 32) ] (Some [ 16; 16; 16 ])))

let grouped_dims =
  group "grouped_dims"
    [
      group "lays out each launch as its golden states"
        ([
           grouped_dims_golden;
           graph_of "grouped_thread_indices" (fun () ->
               grouped ~prefix:"lidx"
                 (ints [ 2; 3; 4; 5 ])
                 (Some [ 16; 16; 16 ]));
           graph_of "grouped_reversed_merge" (fun () ->
               grouped ~reverse:true (ints [ 2; 3; 4; 5 ]) (Some [ 32; 16; 16 ]));
         ]
        @ symbolic);
      group "refuses a symbolic size it must split" [ symbolic_failures ];
      grouped_dims_law;
      direct_dims;
      untouched;
    ]

(* group_gpudims *)

let range ?(axis_type = Ops.Axis_type.Global) size axis =
  Ops.range ~axis_type size [ axis ]

let global = range ~axis_type:Global
let local = range ~axis_type:Local
let one = Ops.float ~dtype:Float32 1.
let buffer ?(slot = 0) n = Shape.param ~shape:[ Int n ] slot Float32

let kernel ?(size = 4096) index value ranges =
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (Ops.index (buffer size) [ index ]) value) ranges ]

let group_kernel ?(renderer = renderer ()) sink =
  Ops.graph_rewrite ~calls:Skip ~pass:Once ~ctx:renderer sink
    (After_sources Gpudims.pm_group_gpudims)

let rewritten ?renderer sink = Ops.sink [ sink; group_kernel ?renderer sink ]

let rewrites name ?renderer sink =
  Golden.graph (name ^ ".golden") (fun () -> rewritten ?renderer (sink ()))

let missing_locals sizes =
  let g = global (Int 32) 0 in
  let ls = List.mapi (fun k n -> local (Int n) (k + 1)) sizes in
  let loaded =
    Ops.load
      (Ops.index (buffer ~slot:1 64)
         [ Ops.O.(g + Ops.usum (List.hd ls) (List.tl ls)) ])
      []
  in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.store (Ops.index (buffer 64) [ g ]) loaded) (g :: ls) ]

let group_gpudims_graphs =
  let open Ops.O in
  [
    rewrites "group_gpudims_globals" (fun () ->
        let g0 = global (Int 32) 0 and g1 = global (Int 16) 1 in
        kernel ((g0 * int 16) + g1) one [ g0; g1 ]);
    rewrites "group_gpudims_globals_by_axis_order" (fun () ->
        let g3 = global (Int 32) 3 and g1 = global (Int 16) 1 in
        kernel ((g3 * int 16) + g1) one [ g3; g1 ]);
    rewrites "group_gpudims_globals_and_locals" (fun () ->
        let g = global (Int 32) 0 and l = local (Int 8) 1 in
        kernel ((g * int 8) + l) one [ g; l ]);
    rewrites "group_gpudims_merges_four_globals" (fun () ->
        let gs = List.mapi (fun k n -> global (Int n) k) [ 2; 3; 4; 5 ] in
        let g k = List.nth gs k in
        kernel ~size:120
          ((((((g 0 * int 3) + g 1) * int 4) + g 2) * int 5) + g 3)
          one gs);
    rewrites "group_gpudims_splits_a_global"
      ~renderer:(renderer ~global_max:[ 256; 256; 256 ] ())
      (fun () ->
        let g = global (Int 1024) 0 in
        kernel g one [ g ]);
    rewrites "group_gpudims_keeps_the_warp_apart"
      ~renderer:(renderer ~local_max:[ 1024; 1024; 64 ] ())
      (fun () ->
        let w = range ~axis_type:Warp (Int 32) 0 in
        let l1 = local (Int 2) 1
        and l2 = local (Int 2) 2
        and l3 = local (Int 2) 3 in
        let g = global (Int 4) 4 in
        let index =
          (((((((g * int 32) + w) * int 2) + l1) * int 2) + l2) * int 2) + l3
        in
        kernel index one [ w; l1; l2; l3; g ]);
    rewrites "group_gpudims_bounds_globals_by_threads"
      ~renderer:
        (renderer ~global_max:[ 256; 256; 256 ] ~local_max:[ 128; 128; 128 ]
           ~global_prod_max:[ 128; 128; 128 ] ())
      (fun () ->
        let g = global (Int 256) 0 and l = local (Int 256) 1 in
        Ops.sink ~kernel:(Ops.kernel_info ())
          [
            Ops.end_
              (Ops.store (Ops.index (buffer 512) [ g + l ]) (float 1.))
              [ g; l ];
          ]);
    rewrites "group_gpudims_bounds_globals_by_merged_threads"
      ~renderer:
        (renderer ~global_max:[ 256; 256; 256 ] ~local_max:[ 16; 16; 16 ]
           ~global_prod_max:[ 1024; 1024; 1024 ] ())
      (fun () ->
        let g = global (Int 256) 0 in
        let ls = List.map (fun axis -> local (Int 4) axis) [ 1; 2; 3; 4 ] in
        let index = List.fold_left (fun acc l -> (acc * int 4) + l) g ls in
        kernel ~size:65536 index one (g :: ls));
    (* An empty [global_max] bounds nothing but the product. *)
    rewrites "group_gpudims_bounds_globals_by_threads_alone"
      ~renderer:
        (renderer ~global_max:[] ~local_max:[ 128; 128; 128 ]
           ~global_prod_max:[ 128; 128; 128 ] ())
      (fun () ->
        let g = global (Int 256) 0 and l = local (Int 256) 1 in
        kernel ~size:512 (g + l) one [ g; l ]);
    rewrites "group_gpudims_keeps_an_end_of_a_variable" (fun () ->
        let n = v "n" 0 3 in
        Ops.sink [ Ops.end_ (Ops.store (Ops.index (buffer 4) [ n ]) one) [ n ] ]);
    rewrites "group_gpudims_masks_a_store_by_its_missing_local" (fun () ->
        missing_locals [ 8 ]);
    rewrites "group_gpudims_masks_a_store_by_its_missing_locals" (fun () ->
        missing_locals [ 8; 4 ]);
    rewrites "group_gpudims_masks_a_store_by_its_missing_global" (fun () ->
        let g = global (Int 32) 0 and l = local (Int 8) 1 in
        let loaded =
          Ops.load (Ops.index (buffer ~slot:1 256) [ (g * int 8) + l ]) []
        in
        Ops.sink ~kernel:(Ops.kernel_info ())
          [ Ops.end_ (Ops.store (Ops.index (buffer 8) [ l ]) loaded) [ g; l ] ]);
    rewrites "group_gpudims_keeps_a_loop" (fun () ->
        let g = global (Int 16) 0 and r = range ~axis_type:Loop (Int 8) 1 in
        let loaded =
          Ops.load (Ops.index (buffer ~slot:1 128) [ (g * int 8) + r ]) []
        in
        Ops.sink ~kernel:(Ops.kernel_info ())
          [ Ops.end_ (Ops.store (Ops.index (buffer 16) [ g ]) loaded) [ g; r ] ]);
    (* The mask reads the hardware ranges, which share the loops' identities. *)
    rewrites "group_gpudims_masks_by_hardware_ranges" (fun () ->
        let g = global (Int 3) 0 and l = local (Int 2) 5 in
        let out = Shape.param ~shape:[ Int 3 ] 0 Int32 in
        Ops.sink ~kernel:(Ops.kernel_info ())
          [
            Ops.end_
              (Ops.store (Ops.index out [ g ]) (Ops.cast (g + l) Int32))
              [ g; l ];
          ]);
    rewrites "group_gpudims_keeps_an_upcast_of_a_hardware_identity" (fun () ->
        let g = global (Int 3) 0 and u = range ~axis_type:Upcast (Int 2) 0 in
        let out = Shape.param ~shape:[ Int 6 ] 0 Int32 in
        let idx = (g * int 2) + u in
        Ops.sink ~kernel:(Ops.kernel_info ())
          [
            Ops.end_
              (Ops.store (Ops.index out [ idx ])
                 (Ops.cast ((idx * int 3) + int 7) Int32))
              [ g; u ];
          ]);
    rewrites "group_gpudims_leaves_a_local_store_unmasked" (fun () ->
        let g = global (Int 32) 0 and l = local (Int 8) 1 in
        let shared = Shape.param ~shape:[ Int 32 ] ~addrspace:Local 0 Float32 in
        let loaded =
          Ops.load (Ops.index (buffer ~slot:1 256) [ (g * int 8) + l ]) []
        in
        Ops.sink ~kernel:(Ops.kernel_info ())
          [ Ops.end_ (Ops.store (Ops.index shared [ g ]) loaded) [ g; l ] ]);
    rewrites "group_gpudims_symbolic_global" (fun () ->
        let g = global (Sym (v "n" 1 64)) 0 in
        kernel ~size:64 g one [ g ]);
  ]

let declines =
  let g = global (Int 32) 0 and r = range ~axis_type:Loop (Int 4) 0 in
  let sink = function
    | "no_kernel_info" ->
        Ops.sink
          [ Ops.end_ (Ops.store (Ops.index (buffer 4096) [ g ]) one) [ g ] ]
    | "hardware_indices" ->
        Ops.sink ~kernel:(Ops.kernel_info ())
          [
            Ops.store
              (Ops.index (buffer 32) [ Ops.special (Int 32) "gidx0" ])
              one;
          ]
    | "no_global_or_local" -> kernel r one [ r ]
    | case -> invalid_arg case
  in
  Golden.cases "group_gpudims_declines.golden" (fun cell ->
      equal string (cell "result") "None";
      is_none ~pp:Ops.pp
        (Gpudims.group_gpudims (renderer ()) (sink (cell "case"))))

let failures =
  Golden.cases "group_gpudims_failures.golden" (fun cell ->
      equal string "AssertionError" (cell "raises");
      let g = global (Int 32) 0 and l = local (Int 8) 1 in
      let loaded =
        Ops.load (Ops.index (buffer ~slot:1 64) [ Ops.O.(g + l) ]) []
      in
      let two = Shape.param ~shape:[ Int 32; Int 2 ] 0 Float32 in
      let sink =
        Ops.sink ~kernel:(Ops.kernel_info ())
          [
            Ops.end_
              (Ops.store (Ops.index two [ g; Ops.int 0 ]) loaded)
              [ g; l ];
          ]
      in
      rejects (fun () -> Gpudims.group_gpudims (renderer ()) sink))

(* A symbolic warp is bounded by its size's upper bound, so it keeps its own
   axis. The golden records an error where its size is compared as an
   integer. *)
let symbolic_warp =
  Golden.cases "group_gpudims_symbolic_warp.golden" (fun cell ->
      equal string "ValueError" (cell "raises");
      let w = range ~axis_type:Warp (Sym (v "n" 1 32)) 0 in
      let l = local (Int 4) 1 and g = global (Int 4) 2 in
      let index = Ops.O.((((g * int 32) + w) * int 4) + l) in
      let sink = kernel index one [ w; l; g ] in
      let r =
        require_some
          (Gpudims.group_gpudims
             (renderer ~local_max:[ 1024; 1024; 64 ] ())
             sink)
      in
      let sizes =
        Ops.toposort ~calls:Enter (List.hd (to_special [ r ]))
        |> List.filter (fun u -> Ops.op u = Special)
        |> List.map (fun u ->
            ( Render.render ~simplify:false u,
              Render.render ~simplify:false (Ops.nth u 0) ))
        |> List.sort compare
      in
      equal
        (list (pair string string))
        [ ("gidx0", "4"); ("lidx0", "n"); ("lidx1", "4") ]
        sizes)

(* pm_range_to_special *)

let specializes name ?renderer sink =
  Golden.graph (name ^ ".golden") (fun () ->
      let grouped = group_kernel ?renderer (sink ()) in
      Ops.sink [ grouped; List.hd (to_special [ grouped ]) ])

let range_to_special =
  let open Ops.O in
  group "pm_range_to_special"
    [
      group "makes each kernel's hardware indices as its golden states"
        [
          specializes "range_to_special_globals_and_locals" (fun () ->
              let g = global (Int 32) 0 and l = local (Int 8) 1 in
              kernel ((g * int 8) + l) one [ g; l ]);
          specializes "range_to_special_splits_its_ends"
            ~renderer:(renderer ~global_max:[ 256; 256; 256 ] ())
            (fun () ->
              let g = global (Int 1024) 0 in
              kernel g one [ g ]);
          specializes "range_to_special_keeps_a_loop" (fun () ->
              let g = global (Int 16) 0
              and r = range ~axis_type:Loop (Int 8) 1 in
              let loaded =
                Ops.load (Ops.index (buffer ~slot:1 128) [ (g * int 8) + r ]) []
              in
              Ops.sink ~kernel:(Ops.kernel_info ())
                [
                  Ops.end_
                    (Ops.store (Ops.index (buffer 16) [ g ]) loaded)
                    [ g; r ];
                ]);
        ];
    ]

(* A product on CUDA's 8 x 16 x 16 tensor core of 32 threads, each thread
   storing its four elements of it, by threads of the sizes and types [threads]:
   [lane] is the thread's lane as the kernel computes it, the first range's
   index modulo 32, so that thread [t] of the launch is the lane it computes
   only if a warp of 32 threads takes the first hardware axis. *)
let product_by threads =
  let ranges =
    List.mapi (fun k (axis_type, n) -> range ~axis_type (Int n) k) threads
  in
  let lane = Ops.O.(List.hd ranges % int 32) in
  let elements slot dt n =
    let at e = Ops.O.((lane * int n) + int e) in
    let p = Shape.param ~shape:[ Int (32 * n) ] slot dt in
    Shape.stack (List.init n (fun e -> Ops.load (Ops.index p [ at e ]) []))
  in
  let zero = Ops.float ~dtype:Float32 0. in
  let d =
    Ops.wmma (elements 1 Bfloat16 8) (elements 2 Bfloat16 4)
      ~acc:(Shape.stack [ zero; zero; zero; zero ])
      ~dims:(8, 16, 16) ~threads:32
  in
  let out = Shape.param ~shape:[ Int 128 ] 0 Float32 in
  let store e =
    Ops.store
      (Ops.index out [ Ops.O.((lane * int 4) + int e) ])
      (Ops.index d [ Ops.int ~dtype:Int32 e ])
  in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.group (List.init 4 store)) ranges ]

let threads_name threads =
  String.concat " and "
    (List.map
       (fun (t, n) -> Printf.sprintf "%s%d" (Ops.Axis_type.letter t) n)
       threads)

let tensor_core_lanes =
  group "a tensor core's lanes"
    [
      cases "are a warp of its threads on the first thread axis"
        ~id:threads_name
        [
          [ (Ops.Axis_type.Warp, 32) ];
          [ (Warp, 32); (Local, 4) ];
          [ (Warp, 32); (Global, 8) ];
        ]
        (fun threads ->
          is_some (Gpudims.group_gpudims (renderer ()) (product_by threads)));
      cases "are refused from other ranges" ~id:threads_name
        [
          [ (Ops.Axis_type.Local, 32) ];
          [ (Local, 128) ];
          [ (Global, 32) ];
          [ (Warp, 16); (Local, 2) ];
          [ (Warp, 64) ];
        ]
        (fun threads ->
          raises_match (Exn.invalid_arg ~sub:"tensor core") (fun () ->
              Gpudims.group_gpudims (renderer ()) (product_by threads)));
    ]

(* Generated kernels: global and local loops, each a size from [sizes] and an
   identity of its own, around one to three stores, each into a buffer of its
   own, at an index of the loops it uses, in mixed radix, of the value the index
   plus one. A store that uses not every loop is masked. *)
type store = bool list

type launch_kernel = {
  loops : (Ops.Axis_type.t * int) list;
  stores : store list;
}

let pp_launch_kernel ppf { loops; stores } =
  let pp_list pp =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " ")
      pp
  in
  Format.fprintf ppf "loops [%a], stores using [%a]"
    (pp_list (fun ppf (t, n) ->
         Format.fprintf ppf "%s%d" (Ops.Axis_type.letter t) n))
    loops
    (pp_list (fun ppf used ->
         Format.pp_print_string ppf
           (String.concat "" (List.map (fun b -> if b then "1" else "0") used))))
    stores

let launch_kernels =
  let open Gen in
  let rec loops k product =
    if k = 0 then constant []
    else
      let* n, t =
        pair
          (of_list
             (List.filter
                (fun n -> product * n <= 512)
                [ 1; 2; 3; 4; 6; 8; 16 ]))
          (of_list Ops.Axis_type.[ Global; Local ])
      in
      let+ rest = loops (k - 1) (product * n) in
      (t, n) :: rest
  in
  let* k = int_range 1 4 in
  let* loops = loops k 1 in
  let+ stores =
    list ~size:(int_range 1 3)
      (list ~size:(constant k)
         (frequency [ (3, constant true); (1, constant false) ]))
  in
  { loops; stores }

let bounds =
  Gen.with_pp
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ",")
       Format.pp_print_int)
    Gen.(list ~size:(constant 3) (of_list [ 2; 4; 8; 16; 32 ]))

let kernel_of { loops; stores } =
  let ranges =
    List.mapi (fun axis (t, n) -> range ~axis_type:t (Int n) axis) loops
  in
  let store slot used =
    let picked =
      List.filter_map
        (fun (r, u) -> if u then Some r else None)
        (List.combine ranges used)
    in
    let index =
      List.fold_left
        (fun acc r -> Ops.O.((acc * Ops.nth r 0) + r))
        (Ops.int 0) picked
    in
    let size =
      List.fold_left
        (fun p r -> p * Bigint.to_int (Shape.to_z (Ops.nth r 0)))
        1 picked
    in
    let out = Shape.param ~shape:[ Int size ] slot Int32 in
    Ops.store (Ops.index out [ index ]) (Ops.cast Ops.O.(index + int 1) Int32)
  in
  Ops.sink ~kernel:(Ops.kernel_info ())
    [ Ops.end_ (Ops.group (List.mapi store stores)) ranges ]

let compare_write (s0, i0, v0) (s1, i1, v1) =
  match Int.compare s0 s1 with
  | 0 -> (
      match Int.compare i0 i1 with 0 -> Dtype.Value.compare v0 v1 | c -> c)
  | c -> c

let write = triple int int Dtypes.value

(* The writes of a grouped kernel, once for each value of its hardware indices:
   the writes at each value, with their repeats. *)
let launched_writes grouped =
  let launched = List.hd (to_special [ grouped ]) in
  let indices = specials [ launched ] in
  let rec each vars = function
    | [] -> Interpreter.writes ~vars launched
    | (name, size) :: rest ->
        List.concat_map
          (fun v -> each ((name, i v) :: vars) rest)
          (List.init (Bigint.to_int size) Fun.id)
  in
  List.sort compare_write (each [] indices)

let keeps_writes (k, global_max, local_max) =
  let sink = kernel_of k in
  let renderer = renderer ~global_max ~local_max () in
  let grouped =
    match Gpudims.group_gpudims renderer sink with
    | g -> g
    | exception Invalid_argument _ -> None
  in
  let loops = List.length k.loops in
  let axes =
    Option.map
      (fun g -> List.length (specials [ List.hd (to_special [ g ]) ]))
      grouped
  in
  cover "refuses" (grouped = None);
  cover "a store is masked" (List.exists (List.mem false) k.stores);
  cover "merges" (Option.fold ~none:false ~some:(fun n -> n < loops) axes);
  cover "splits" (Option.fold ~none:false ~some:(fun n -> n > loops) axes);
  Option.iter
    (fun g -> equal (list write) (Interpreter.writes sink) (launched_writes g))
    grouped

let group_gpudims =
  group "group_gpudims"
    [
      group "groups each kernel's loops as its golden states"
        group_gpudims_graphs;
      prop ~count:200
        "writes what the kernel writes, each element once for each point of \
         the indices its store uses"
        Gen.(triple (with_pp pp_launch_kernel launch_kernels) bounds bounds)
        keeps_writes;
      group "is None for each kernel its golden lists" [ declines ];
      group "refuses a masked store of two indices" [ failures ];
      group "keeps a symbolic warp on its own axis" [ symbolic_warp ];
      tensor_core_lanes;
    ]

let () =
  exit (run "Tolk.Gpudims" [ grouped_dims; group_gpudims; range_to_special ])
