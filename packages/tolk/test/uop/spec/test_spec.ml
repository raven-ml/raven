open Windtrap
open Tolk

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

let verdict_of_cell = function
  | "True" -> Some true
  | "False" -> Some false
  | "None" -> None
  | s -> invalid_arg (Printf.sprintf "%S is not a verdict" s)

let checking_bounds f = Setting.context [ B (Setting.check_oob, true) ] f

(* [failure ~calls spec u] is the message of [type_verify ~calls spec u], [None]
   if it passes. *)
let failure ~calls spec u =
  match Spec.type_verify ~calls spec u with
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

let i n = `Int (Bigint.of_int n)

let var ?(dtype = Dtype.Int32) name lo hi =
  Ops.variable ~dtype name (i lo) (i hi)

let fvar name = Ops.variable ~dtype:Float32 name (`Float (-10.)) (`Float 10.)
let flag name = Ops.variable ~dtype:Bool name (`Bool false) (`Bool true)
let buffer ?(dtype = Dtype.Int32) n = Call.param ~shape:[ Int n ] 0 dtype
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
            (failure ~calls:Enter Spec.tensor u);
          equal ~msg:"program" (option string)
            (failure_of_cell (cell "program"))
            (failure ~calls:Enter Spec.program u);
          equal ~msg:"outside calls" (option string)
            (failure_of_cell (cell "tensor_outside_calls"))
            (failure ~calls:Skip Spec.tensor u));
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
                  (Ops.toposort ~calls:Enter u)
              in
              cover (name ^ " accepts a graph") (Option.is_none first);
              cover (name ^ " rejects a graph") (Option.is_some first);
              equal ~msg:name (option int) first
                (Option.bind (failure ~calls:Enter spec u) failed_at))
            specs);
      test "prints nothing when DEBUG is below 3" (fun () ->
          Setting.context
            [ B (Setting.debug, 2) ]
            (fun () ->
              ignore (failure ~calls:Enter Spec.shared ill_typed_graph));
          equal string "" (output ()));
      group "prints the graph when DEBUG is 3 or more, before failing"
        [
          Golden.text "debug_listing.golden" (fun () ->
              Setting.context
                [ B (Setting.debug, 3) ]
                (fun () ->
                  ignore (failure ~calls:Enter Spec.shared ill_typed_graph));
              output ());
        ];
    ]

(* Vectors in programs *)

(* An elementwise operation of a program on two lanes, and the same operation on
   one. *)
let lanes op =
  let pair a b = Shape.stack [ fvar a; fvar b ] in
  match op with
  | `Add ->
      (Ops.add (pair "a" "b") (pair "c" "d"), Ops.add (fvar "a") (fvar "c"))
  | `Cast -> (Ops.cast (pair "a" "b") Int32, Ops.cast (fvar "a") Int32)
  | `Where ->
      let cond = Shape.stack [ flag "p"; flag "q" ] in
      ( Ops.where cond (pair "a" "b") (pair "c" "d"),
        Ops.where (flag "p") (fvar "a") (fvar "c") )

let vectors =
  group "vectors in programs"
    [
      cases "a program has no elementwise operation on a vector"
        ~name:(function `Add -> "add" | `Cast -> "cast" | `Where -> "where")
        [ `Add; `Cast; `Where ]
        (fun op ->
          let vector, scalar = lanes op in
          equal verdict (Some false) (judge Spec.program vector);
          is_false (judge Spec.program scalar = Some false));
      test "a program reads a lane of a vector at a constant (D86)" (fun () ->
          let vector = Shape.stack [ fvar "a"; fvar "b" ] in
          let lane i = fresh_v ~src:[ vector; i ] Op.Index in
          let i =
            Ops.variable ~dtype:Int32 "i" (`Int Bigint.zero) (`Int Bigint.one)
          in
          equal verdict (Some false) (judge Spec.program (lane i));
          is_false
            (judge Spec.program (lane (Ops.int ~dtype:Int32 1)) = Some false));
    ]

(* Construction *)

let construction =
  group "construction"
    [
      test
        "builds a node that breaks the full specification, whatever SPEC (D139)"
        (fun () ->
          let ill_typed () = fresh_v ~src:[ fvar "a"; fvar "b" ] Op.And in
          List.iter
            (fun level ->
              let u = Setting.context [ B (Setting.spec, level) ] ill_typed in
              equal verdict (Some false) (judge Spec.full u))
            [ 0; 1; 2; 3 ]);
    ]

(* Loops of calls in the kernel graph *)

(* [row axis_type] is a row of four of twelve floats that moves with a range of
   three trips of [axis_type], with the range. *)
let row axis_type =
  let r = Ops.range ~axis_type (Int 3) [ 100 ] in
  let rows = Call.param ~shape:[ Int 12 ] 1 Float32 in
  ( Shape.shrink rows
      [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 4)) ],
    r )

let looped axis_type =
  let v, r = row axis_type in
  let body = Ops.v Op.Linear ~src:[] in
  (Ops.end_ (Ops.call ~precompile:true body [ v ]) [ r ], v, r)

(* [kernel_graph_judges axis_type] is the kernel graph spec's verdicts on a loop
   of a call over a range of [axis_type], on its view and the view's offset, and
   on the range. *)
let kernel_graph_judges axis_type =
  let e, v, r = looped axis_type in
  List.map (judge Spec.kernel_graph) [ e; v; Ops.nth v 1; r ]

let loops =
  group "kernel_graph › loops of calls"
    [
      test "accepts a loop of a call over a loop range" (fun () ->
          equal (list verdict)
            [ Some true; Some true; Some true; Some true ]
            (kernel_graph_judges Loop));
      test "refuses a loop of a call over a range of any other kind" (fun () ->
          List.iter
            (fun axis_type ->
              equal (list verdict)
                [ Some false; Some false; Some false; Some false ]
                (kernel_graph_judges axis_type))
            Ops.Axis_type.[ Weak; Global; Reduce; Upcast ]);
      test
        "accepts a loop's call whose scalar argument is an expression of its \
         range" (fun () ->
          let v, r = row Loop in
          let trip = Ops.O.(int 2 - r) in
          let call =
            Ops.call ~precompile:true (Ops.v Op.Linear ~src:[]) [ v; trip ]
          in
          equal (list verdict)
            [ Some true; Some true; Some true ]
            (List.map (judge Spec.kernel_graph)
               [ Ops.end_ call [ r ]; call; trip ]));
      test "accepts an open device range" (fun () ->
          equal verdict (Some true)
            (judge Spec.kernel_graph
               (Ops.range ~axis_type:Device (Int 2) [ -1 ])));
      test "refuses a weak sum of a weak integer variable" (fun () ->
          equal verdict (Some false)
            (judge Spec.kernel_graph
               Ops.O.(var ~dtype:Weak_int "n" 0 8 + int 1)));
    ]

(* Arguments against their parameters *)

(* A call, in a loop of three trips, of a body adding one to its parameter of
   four floats, whose start is known to [align] bytes, on rows [stride] floats
   apart. *)
let call_on_rows ~align stride =
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let rows = Call.param ~shape:[ Int ((2 * stride) + 4) ] 1 Float32 in
  let start = Ops.O.(r * int stride) in
  let row = Shape.shrink rows [ Some (Sym start, Sym Ops.O.(start + int 4)) ] in
  let p = Call.param ~shape:[ Int 4 ] ~align 0 Float32 in
  let k = Ops.range (Int 4) [ 0 ] in
  let one = Ops.float ~dtype:Float32 1. in
  let body =
    Ops.sink ~kernel:(Ops.kernel_info ())
      [
        Ops.end_
          (Ops.store (Ops.index p [ k ]) (Ops.add (Ops.index p [ k ]) one))
          [ k ];
      ]
  in
  Ops.call ~precompile:true body [ row ]

let arguments =
  group "kernel_graph › arguments against their parameters"
    [
      test "refuses a row a float apart for a parameter that starts on 16 bytes"
        (fun () ->
          equal verdict (Some false)
            (judge Spec.kernel_graph (call_on_rows ~align:16 5)));
      test
        "accepts a row four floats apart for a parameter that starts on 16 \
         bytes" (fun () ->
          equal verdict (Some true)
            (judge Spec.kernel_graph (call_on_rows ~align:16 4)));
      test "accepts a row a float apart for a parameter that starts on 4 bytes"
        (fun () ->
          equal verdict (Some true)
            (judge Spec.kernel_graph (call_on_rows ~align:4 5)));
    ]

let () =
  exit
    (run "Tolk.Spec"
       [
         verdicts; bounds; type_verify; vectors; construction; loops; arguments;
       ])
