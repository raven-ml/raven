open Windtrap
open Tolk_next

let specs =
  Spec.
    [
      ("shared", shared);
      ("tensor", tensor);
      ("program", program);
      ("hcq", hcq);
      ("full", full);
      ("kernel_graph", kernel_graph);
    ]

let judge spec u = Ops.Pattern_matcher.rewrite spec () u
let verdict = option bool
let uop = Testable.make ~pp:Ops.pp ~equal:Ops.equal

let verdict_of_cell = function
  | "True" -> Some true
  | "False" -> Some false
  | "None" -> None
  | s -> invalid_arg (Printf.sprintf "%S is not a verdict" s)

let with_spec level f = Helpers.context [ B (Helpers.spec, level) ] f
let checking_bounds f = Helpers.context [ B (Helpers.check_oob, true) ] f

(* [failure spec u] is the message of [type_verify spec u], [None] if it
   passes. *)
let failure ?enter_calls spec u =
  match Spec.type_verify ?enter_calls spec u with
  | () -> None
  | exception Invalid_argument msg -> Some msg

let failure_of_cell = function "None" -> None | msg -> Some msg

(* [each_node table check] is the test, per row of [table], of [check cell u] on
   the row's node: the source, at the row's position, of the sink of the graph
   golden [<table>_nodes.golden]. A test is named by its case. *)
let each_node table check =
  let base = Filename.remove_extension table in
  let nodes = Ops.src (Golden.sink (base ^ "_nodes.golden")) in
  cases
    ~name:(fun (cell, _) -> cell "case")
    table
    (List.combine (Golden.rows table) nodes)
    (fun (cell, u) -> check cell u)

(* Nodes, as tests write them *)

let i n = `Int (Z.of_int n)

let var ?(dtype = Dtype.Int32) name lo hi =
  Ops.variable ~dtype name (i lo) (i hi)

let fvar name = Ops.variable ~dtype:Float32 name (`Float (-10.)) (`Float 10.)
let flag name = Ops.variable ~dtype:Bool name (`Bool false) (`Bool true)
let buffer ?(dtype = Dtype.Int32) n = Ops.param ~shape:[ Int n ] 0 dtype
let load buf idx = Ops.load (Ops.index buf [ idx ]) []

let gated_load buf idx =
  Ops.load (Ops.index buf [ idx ]) [ Ops.int ~dtype:Int32 0; flag "g" ]

let store ?gate buf idx =
  Ops.store ?gate (Ops.index buf [ idx ]) (Ops.int ~dtype:Int32 0)

let fresh =
  let n = ref 0 in
  fun () ->
    incr n;
    Ops.Tag.Int !n

(* [fresh_v ~src ~arg op] is a node that construction has never built: its tag
   is new. *)
let fresh_v ?src ?arg op = Ops.v ?src ?arg ~tag:(fresh ()) op

(* Verdicts *)

let verdicts =
  group "verdicts"
    [
      each_node "verdicts.golden" (fun cell u ->
          let expected =
            List.map
              (fun (name, _) -> (name, verdict_of_cell (cell name)))
              specs
          in
          equal
            (list (pair string verdict))
            expected
            (List.map (fun (name, spec) -> (name, judge spec u)) specs));
    ]

(* Bounds *)

(* The bounds of an index into storage of [n] elements, drawn across both
   ends. *)
let index_bounds =
  Gen.(
    with_pp
      (fun ppf (lo, hi, n) -> Format.fprintf ppf "[%d, %d] into %d" lo hi n)
      (let* n = int_range 1 20 in
       let* lo = int_range (-3) (n + 2) in
       let+ hi = int_range lo (n + 3) in
       (lo, hi, n)))

let in_bounds (lo, hi, n) = 0 <= lo && hi < n
let access (lo, hi, n) = load (buffer n) (var "i" lo hi)

let bounds =
  group "bounds"
    [
      each_node "bounds.golden" (fun cell u ->
          equal ~msg:"unchecked" verdict
            (verdict_of_cell (cell "unchecked"))
            (judge Spec.shared u);
          equal ~msg:"checked" verdict
            (verdict_of_cell (cell "checked"))
            (checking_bounds (fun () -> judge Spec.shared u)));
      prop "an access passes iff the bounds of its index lie within the storage"
        index_bounds (fun b ->
          let lo, hi, n = b in
          cover "the last element" (hi = n - 1);
          cover "one past the last element" (hi = n);
          cover "one before the first element" (lo = -1);
          equal verdict
            (Some (in_bounds b))
            (checking_bounds (fun () -> judge Spec.shared (access b))));
      prop "a gate never changes whether an access passes" index_bounds
        (fun (lo, hi, n) ->
          cover "an access in bounds" (in_bounds (lo, hi, n));
          cover "an access out of bounds" (not (in_bounds (lo, hi, n)));
          let buf = buffer n and idx = var "i" lo hi in
          checking_bounds (fun () ->
              equal ~msg:"load" verdict
                (judge Spec.shared (load buf idx))
                (judge Spec.shared (gated_load buf idx));
              equal ~msg:"store" verdict
                (judge Spec.shared (store buf idx))
                (judge Spec.shared (store ~gate:(flag "g") buf idx))));
      test "without CHECK_OOB, every access passes" (fun () ->
          equal verdict (Some true) (judge Spec.shared (access (-5, 100, 4))));
    ]

(* Random graphs *)

(* A graph is built by steps, each an operation on nodes built before it,
   numbered from the leaves. Operations mix types freely, so that some nodes are
   ill-typed; a step whose type cannot be derived is left out. Half the graphs
   are built from casts and stacks alone, the forms of a kernel graph's
   arguments, which the other operations never are. *)

let leaves () =
  Dtype.
    [
      var "a" 0 10;
      fvar "b";
      var ~dtype:Float16 "h" 0 1;
      flag "c";
      Ops.int 3;
      Ops.float 1.5;
      Ops.invalid;
      load (buffer ~dtype:Float32 16) (Ops.int 0);
    ]

let unary = Op.[ Neg; Sin; Sqrt; Trunc ]
let binary = Op.[ Add; Mul; Max; Cmplt; Cmpne; And; Xor; Shl; Cdiv; Floormod ]
let where_step = List.length unary + List.length binary
let cast_step = where_step + 1
let stack_step = where_step + 2

let step pool (k, a, b, c) =
  let nth j = List.nth pool (j mod List.length pool) in
  let src, op, arg =
    if k < List.length unary then ([ nth a ], List.nth unary k, None)
    else if k < where_step then
      ([ nth a; nth b ], List.nth binary (k - List.length unary), None)
    else if k = where_step then ([ nth a; nth b; nth c ], Op.Where, None)
    else if k = cast_step then ([ nth a ], Op.Cast, Some (Ops.Dtype Int32))
    else ([ nth a; nth b ], Op.Stack, None)
  in
  match Ops.v ~src ?arg op with
  | u -> pool @ [ u ]
  | exception Invalid_argument _ -> pool

let build steps =
  let pool = List.fold_left step (leaves ()) steps in
  Ops.sink (List.filteri (fun j _ -> j >= List.length (leaves ())) pool)

let graphs =
  let steps first =
    Gen.(
      list ~size:(int_range 1 6) (quad (int_range first stack_step) nat nat nat))
  in
  Gen.(
    with_pp
      (fun ppf steps -> Ops.pp ppf (build steps))
      (frequency [ (1, steps 0); (1, steps cast_step) ]))

(* Graphs that every seed checks first: the cast of the constant 3, which each
   specification accepts, and the sum of an int32 and a float32, which each
   rejects. *)
let decided_by_every_spec =
  [ [ (cast_step, 4, 0, 0) ]; [ (List.length unary, 0, 1, 0) ] ]

(* type_verify *)

let ill_typed_graph = Ops.sink [ Ops.v ~src:[ fvar "a"; fvar "b" ] Op.And ]
let failed_at msg = Scanf.sscanf_opt msg "UOp verification failed at %d " Fun.id

let type_verify =
  group "type_verify"
    [
      each_node "failures.golden" (fun cell u ->
          equal ~msg:"tensor" (option string)
            (failure_of_cell (cell "tensor"))
            (failure Spec.tensor u);
          equal ~msg:"program" (option string)
            (failure_of_cell (cell "program"))
            (failure Spec.program u);
          equal ~msg:"outside calls" (option string)
            (failure_of_cell (cell "tensor_outside_calls"))
            (failure ~enter_calls:false Spec.tensor u));
      prop
        "fails at the first node, sources first, that the specification does \
         not accept"
        ~examples:decided_by_every_spec graphs (fun steps ->
          let u = build steps in
          List.iter
            (fun (name, spec) ->
              let first =
                List.find_index
                  (fun n -> judge spec n <> Some true)
                  (Ops.toposort u)
              in
              cover (name ^ " accepts a graph") (Option.is_none first);
              cover (name ^ " rejects a graph") (Option.is_some first);
              equal ~msg:name (option int) first
                (Option.bind (failure spec u) failed_at))
            specs);
      test "prints nothing when DEBUG is below 3" (fun () ->
          Helpers.context
            [ B (Helpers.debug, 2) ]
            (fun () -> ignore (failure Spec.shared ill_typed_graph));
          equal string "" (output ()));
      group "prints the graph when DEBUG is 3 or more, before failing"
        [
          Golden.text "debug_listing.golden" (fun () ->
              Helpers.context
                [ B (Helpers.debug, 3) ]
                (fun () -> ignore (failure Spec.shared ill_typed_graph));
              output ());
        ];
    ]

(* Construction *)

let ill_typed () = fresh_v ~src:[ fvar "a"; fvar "b" ] Op.And
let rejects f = raises_match (Exn.invalid_arg ?substring:None) f

let construction =
  group "construction"
    [
      test
        "checks each node it builds against the full specification when SPEC \
         is 2" (fun () -> rejects (fun () -> with_spec 2 ill_typed));
      test "checks nothing when SPEC is 1" (fun () ->
          ignore (with_spec 1 ill_typed));
      test "checks nothing when SPEC is 0" (fun () ->
          ignore (with_spec 0 ill_typed));
      (* tinygrad returns a node it already holds before checking it. *)
      test "checks a node only when it creates it" (fun () ->
          let tag = fresh () in
          let build () = Ops.v ~src:[ fvar "a"; fvar "b" ] ~tag Op.And in
          let u = with_spec 1 build in
          equal uop u (with_spec 2 build));
      test "never checks bounds, whatever CHECK_OOB" (fun () ->
          checking_bounds (fun () ->
              with_spec 2 (fun () ->
                  ignore
                    (fresh_v
                       ~src:[ Ops.index (buffer 4) [ Ops.int 9 ] ]
                       Op.Load))));
      test "computes the shape of each node it builds when SPEC is 3" (fun () ->
          let unbroadcastable () =
            fresh_v
              ~src:[ buffer ~dtype:Float32 4; buffer ~dtype:Float32 5 ]
              Op.Add
          in
          ignore (with_spec 2 unbroadcastable);
          rejects (fun () -> with_spec 3 unbroadcastable));
      test "accepts the forms only the full specification holds" (fun () ->
          with_spec 2 (fun () -> ignore (fresh_v ~src:[ buffer 4 ] Op.Load)));
      prop "rejects a new node iff the full specification does not accept it"
        graphs (fun steps ->
          match List.rev (Ops.src (build steps)) with
          | [] -> reject ()
          | last :: _ ->
              let rebuild () =
                fresh_v ~src:(Ops.src last) ~arg:(Ops.arg last) (Ops.op last)
              in
              let accepted = judge Spec.full (rebuild ()) = Some true in
              cover "an accepted node" accepted;
              cover "a rejected node" (not accepted);
              if accepted then ignore (with_spec 2 rebuild)
              else rejects (fun () -> with_spec 2 rebuild));
    ]

let () =
  exit (run "Tolk_next.Spec" [ verdicts; bounds; type_verify; construction ])
