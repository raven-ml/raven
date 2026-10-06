open Windtrap
open Tolk

let rejects f = raises_match (Exn.invalid_arg ?substring:None) f

(* The passes, as the codegen pipeline applies them. *)
let split u =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
    (After_sources Linearizer.pm_split_ends)

let chain sink =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point
    ~ctx:(Linearizer.cfg_context sink)
    sink (Before_sources Linearizer.pm_add_control_flow)

(* A linearization is written as the sources of a [LINEAR], tinygrad's node for
   a program in execution order. *)
let linear sink = Ops.v Op.Linear ~src:(Linearizer.linearize sink)
let toposorted f = Setting.context [ B (Setting.tuple_order, false) ] f

(* The cases of an input golden are its sink's sources, in order. *)
let case file cell =
  List.nth (Ops.src (Golden.sink file)) (int_of_string (cell "src"))

(* Kernels compiled by tinygrad, with the passes their goldens record besides
   linearization: [`Split] for pm_split_ends, [`Chain] for pm_add_control_flow.
   The CPU kernels are compiled for Clang; [sum_group] and [matmul_local] for
   CUDA, with workgroup memory and barriers; [dependent_loop_bound] for no
   target in particular. [late_bias_load] and the three after it are the kernels
   of tinygrad's runtime/test_linearizer.py, whose claims are about the order: a
   bias loaded after the reduction's loop ends, a load and its index between two
   ranges, a load hoisted before the range, register stores inside the loop. The
   loops are the do-while kernels of tinygrad's test_wait_loop. *)
let kernels =
  [
    ("matmul", [ `Split; `Chain ]);
    ("matmul_noopt", [ `Split; `Chain ]);
    ("softmax", [ `Split; `Chain ]);
    ("conv", [ `Split; `Chain ]);
    ("two_sums", [ `Split; `Chain ]);
    ("transpose", [ `Split; `Chain ]);
    ("variable", []);
    ("dependent_loop_bound", [ `Chain ]);
    ("sum_group", [ `Split; `Chain ]);
    ("matmul_local", [ `Split ]);
    ("late_bias_load", []);
    ("two_nested_range_alt_indexing", []);
    ("range_outer_op_before_phi", []);
    ("simple_unroll", []);
    ("wait_loop", []);
    ("nested_loop", [ `Chain ]);
    ("two_loops", [ `Chain ]);
    ("loop_in_loop", [ `Chain ]);
  ]

let golden name suffix = name ^ suffix ^ ".golden"
let sink name suffix = Golden.sink (golden name suffix)

(* [recorded pass] is the kernels whose goldens record [pass]. *)
let recorded pass =
  List.filter_map
    (fun (name, passes) -> if List.mem pass passes then Some name else None)
    kernels

let linearized name =
  Golden.graph (golden name "_linear") (fun () -> linear (sink name ""))

let split_kernel name =
  Golden.graph (golden name "_split") (fun () -> split (sink name "_unsplit"))

let chained_kernel name =
  Golden.graph (golden name "_chained") (fun () ->
      chain (sink name "_unchained"))

(* Laws *)

let positions order =
  let at = Ops.Tbl.create 64 in
  List.iteri (fun i u -> Ops.Tbl.replace at u i) order;
  Ops.Tbl.find at

let is_end u = match Ops.op u with Op.End | Op.Backedge -> true | _ -> false

let is_topological sink order =
  let at = positions order in
  equal (slist Uops.uop Ops.compare) (Ops.toposort ~calls:Enter sink) order;
  equal Uops.uop sink (List.hd (List.rev order));
  List.iter
    (fun u ->
      List.iter
        (fun s ->
          less ~msg:"a source is placed after its user" int ~than:(at u) (at s))
        (Ops.src u))
    order

(* Each loop end closes the innermost range still open. *)
let loops_nest order =
  let close opened u =
    match (Ops.op u, opened) with
    | Op.Range, _ -> u :: opened
    | (Op.End | Op.Backedge), innermost :: outer ->
        equal ~msg:"a loop end closes the innermost open range" Uops.uop
          innermost (Ops.nth u 1);
        outer
    | (Op.End | Op.Backedge), [] -> fail "a loop end closes no open range"
    | _ -> opened
  in
  ignore (List.fold_left close [] order)

(* A node that runs inside a range comes after the range opens and before the
   end that closes it. *)
let bodies_inside_loops order =
  let at = positions order in
  let ends = Ops.Tbl.create 16 in
  List.iter
    (fun u -> if is_end u then Ops.Tbl.replace ends (Ops.nth u 1) u)
    order;
  List.iter
    (fun u ->
      List.iter
        (fun r ->
          match Ops.Tbl.find_opt ends r with
          | Some e when not (Ops.equal r u) ->
              less ~msg:"a body node is placed before its range" int
                ~than:(at u) (at r);
              less ~msg:"a body node is placed after its loop end" int
                ~than:(at e) (at u)
          | _ -> ())
        (Ops.Nodes.to_list (Ops.ranges u)))
    order

(* Each end closes one range after pm_split_ends. *)
let ends_are_single sink =
  List.iter
    (fun u ->
      if Ops.op u = Op.End then
        equal ~msg:"an end closes one range" int 2 (List.length (Ops.src u)))
    (Ops.toposort ~calls:Enter sink)

(* Generated kernels: trees of loops storing into parameters. A loop either
   closes its own range, or [fuses] its children's ends into its own, as kernels
   ending several ranges at once do. A store may read the output of the effect
   before it, so that sibling loops depend on each other. *)
type item =
  | Store of { reads : bool }
  | Loop of { size : int; fuses : bool; body : item list }

let rec item depth =
  let open Gen in
  let store = map (fun reads -> Store { reads }) bool in
  if depth = 0 then store
  else
    frequency
      [
        (2, store);
        ( 3,
          let+ size = int_range 0 4
          and+ fuses = frequency [ (3, constant false); (1, constant true) ]
          and+ body = list ~size:(int_range 1 3) (item (depth - 1)) in
          Loop { size; fuses; body } );
      ]

let fbuf slot = Call.param ~shape:[ Int 4096 ] slot Dtype.Float32

(* [kernel items] is the sink of [items]: each store writes the parameter of its
   own slot, from [1], at the sum of its enclosing ranges; a reading store loads
   parameter [0] ordered after the effect before it. *)
let kernel items =
  let slot = ref 0 and axis = ref 0 in
  let fresh r =
    incr r;
    !r
  in
  let rec effects ~defer outer items =
    let step (before, done_) item =
      let u, pending = place ~defer outer before item in
      ((if pending = [] then Some u else None), (u, pending) :: done_)
    in
    List.rev (snd (List.fold_left step (None, []) items))
  and place ~defer outer before = function
    | Store { reads } ->
        let at = List.fold_left Ops.add (Ops.int 0) outer in
        let value =
          match before with
          | Some u when reads ->
              Ops.load (Ops.index (Ops.after (fbuf 0) [ u ]) [ at ]) []
          | _ -> Ops.float ~dtype:Float32 1.0
        in
        (Ops.store (Ops.index (fbuf (fresh slot)) [ at ]) value, [])
    | Loop { size; fuses; body } ->
        let r = Ops.range ~axis_type:Loop (Int size) [ fresh axis ] in
        let children = effects ~defer:fuses (r :: outer) body in
        let u = Ops.group (List.map fst children)
        and pending = r :: List.concat_map snd children in
        if defer then (u, pending) else (Ops.end_ u pending, [])
  in
  Ops.sink (List.map fst (effects ~defer:false [] items))

let kernels_gen =
  Gen.with_pp
    (fun ppf k -> Format.pp_print_string ppf (Graph.to_string k))
    (Gen.map kernel (Gen.list ~size:(Gen.int_range 1 3) (item 3)))

let compiled sink = Linearizer.linearize (chain (split sink))

let unsplit_kernels =
  List.map (fun name -> sink name "_unsplit") (recorded `Split)

let linearize_inputs = List.map (fun (name, _) -> sink name "") kernels

let laws =
  group "laws"
    [
      prop "linearize is a topological order of the sink's nodes, the sink last"
        ~examples:linearize_inputs kernels_gen (fun k ->
          is_topological k (Linearizer.linearize k));
      prop "without the structural tie-break, it is still a topological order"
        ~examples:linearize_inputs kernels_gen (fun k ->
          toposorted (fun () -> is_topological k (Linearizer.linearize k)));
      prop "after splitting and chaining, loops nest" ~examples:unsplit_kernels
        kernels_gen (fun k -> loops_nest (compiled k));
      prop "after splitting and chaining, a range's body runs inside its loop"
        ~examples:unsplit_kernels kernels_gen (fun k ->
          bodies_inside_loops (compiled k));
      prop "pm_split_ends leaves each end closing one range"
        ~examples:unsplit_kernels kernels_gen (fun k ->
          ends_are_single (split k));
      prop "pm_split_ends is idempotent" ~examples:unsplit_kernels kernels_gen
        (Law.idempotent Uops.uop split);
      prop "pm_split_ends closes the ranges it was given" kernels_gen (fun k ->
          let closed u =
            List.concat_map Ops.ended_ranges (Ops.toposort ~calls:Enter u)
            |> List.sort_uniq Ops.compare
          in
          equal (list Uops.uop) (closed k) (closed (split k)));
    ]

(* Tests *)

(* The README's CPython row on tuplize: tinygrad ties [c' + b] and [c + a],
   whose first sources differ only by a tag, and keeps [c' + b] first, as the
   sink lists it; tolk goes on to the second sources, as tinygrad does for [c +
   b] and [c + a]. *)
let tag_blind_order () =
  let c = Ops.int ~dtype:Int32 1 in
  let var name =
    Ops.variable ~dtype:Int32 name (`Int Bigint.zero) (`Int (Bigint.of_int 9))
  in
  let ca = Ops.add c (var "a") and cb = Ops.add c (var "b") in
  let c'b = Ops.add (Ops.rtag ~tag:(Ops.Tag.Int 1) c) (var "b") in
  let before sources earlier later =
    let at = positions (Linearizer.linearize (Ops.sink sources)) in
    less int ~than:(at later) (at earlier)
  in
  before [ cb; ca ] ca cb;
  before [ c'b; ca ] ca c'b

let linearize =
  group "linearize"
    ([
       Golden.cases "orders.golden" (fun cell ->
           equal Uops.uop
             (case "orders_output.golden" cell)
             (linear (case "orders_input.golden" cell)));
       Golden.graph "matmul_linear_toposort.golden" (fun () ->
           toposorted (fun () -> linear (sink "matmul" "")));
       Golden.graph "conv_linear_toposort.golden" (fun () ->
           toposorted (fun () -> linear (sink "conv" "")));
       Golden.graph "orders_output_toposort.golden" (fun () ->
           let cases = Ops.src (Golden.sink "orders_input.golden") in
           toposorted (fun () -> Ops.sink (List.map linear cases)));
       test
         "nodes whose sources differ only by a tag are ordered by their later \
          sources"
         tag_blind_order;
       test "without DEBUG_LINEARIZE, nothing is printed" (fun () ->
           ignore (output ());
           ignore (Linearizer.linearize (sink "dependent_loop_bound" ""));
           flush stdout;
           equal string "" (output ()));
     ]
    @ List.map (fun (name, _) -> linearized name) kernels)

(* DEBUG_LINEARIZE is read once per process, so the stanza runs this group in a
   process of its own, with the variable set. *)
let printed () =
  equal ~msg:"the suite runs this test with DEBUG_LINEARIZE=1" (option string)
    (Some "1")
    (Sys.getenv_opt "DEBUG_LINEARIZE");
  ignore (output ());
  Setting.context
    [ B (Setting.no_color, true) ]
    (fun () -> ignore (Linearizer.linearize (sink "dependent_loop_bound" "")));
  flush stdout;
  output ()

let debug_linearize =
  group ~tags:[ "debug_linearize" ] "DEBUG_LINEARIZE"
    [ Golden.text "debug_linearize.golden" printed ]

(* The README's CPython row on do_split_ends: tinygrad's sort of range arguments
   raises when two identities tie up to an axis type, or when one is a prefix of
   the other. tolk orders identities lexicographically, a prefix first, then
   axis types in declaration order, the greatest innermost. *)
let untied_order () =
  let range axis_type id = Ops.range ~axis_type (Int 4) id in
  let u = Ops.store (Ops.index (fbuf 0) [ Ops.int 0 ]) (Ops.float 1.0) in
  let nests ~outer ~inner =
    equal Uops.uop
      (Ops.end_ (Ops.end_ u [ inner ]) [ outer ])
      (split (Ops.v Op.End ~src:[ u; outer; inner ]));
    equal Uops.uop
      (Ops.end_ (Ops.end_ u [ inner ]) [ outer ])
      (split (Ops.v Op.End ~src:[ u; inner; outer ]))
  in
  nests ~outer:(range Global [ 1 ]) ~inner:(range Loop [ 1 ]);
  nests ~outer:(range Loop [ 1 ]) ~inner:(range Loop [ 1; 0 ])

let pm_split_ends =
  group "pm_split_ends"
    (Golden.cases "splits.golden" (fun cell ->
         equal Uops.uop
           (case "splits_output.golden" cell)
           (split (case "splits_input.golden" cell)))
    :: test "ranges of one identity and different axis types nest by axis type"
         untied_order
    :: List.map split_kernel (recorded `Split))

let pm_add_control_flow =
  group "pm_add_control_flow"
    ([
       Golden.cases "chains.golden" (fun cell ->
           equal Uops.uop
             (case "chains_output.golden" cell)
             (chain (case "chains_input.golden" cell)));
       test "a range that would run after a loop depending on it is rejected"
         (fun () ->
           rejects (fun () ->
               Linearizer.cfg_context (Golden.sink "cyclic.golden")));
     ]
    @ List.map chained_kernel (recorded `Chain))

let () =
  exit
    (run "Tolk.Linearizer"
       [ linearize; pm_split_ends; pm_add_control_flow; laws; debug_linearize ])
