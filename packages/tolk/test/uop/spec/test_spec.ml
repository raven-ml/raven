open Windtrap
open Tolk

let specs =
  Spec.
    [
      ("shared", shared);
      ("tensor", tensor);
      ("program", program);
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

(* Recorded nodes: the node of each row of a table golden is the source, at the
   row's position, of the sink of the graph golden [<table>_nodes.golden]. Both
   are read when a test first asks. *)
let recorded table =
  lazy
    (let base = Filename.remove_extension table in
     let nodes = Ops.src (Golden.sink (base ^ "_nodes.golden")) in
     let cases = List.map (fun cell -> cell "case") (Golden.rows table) in
     if List.compare_lengths cases nodes <> 0 then
       failwith
         (Printf.sprintf "%s has %d rows, its nodes %d" table
            (List.length cases) (List.length nodes));
     List.combine cases nodes)

let node_of recorded cell = List.assoc (cell "case") (Lazy.force recorded)

(* Nodes, as tests write them *)

let i n = `Int (Bigint.of_int n)

let var ?(dtype = Dtype.Int32) name lo hi =
  Ops.variable ~dtype name (i lo) (i hi)

let fvar name = Ops.variable ~dtype:Float32 name (`Float (-10.)) (`Float 10.)
let flag name = Ops.variable ~dtype:Bool name (`Bool false) (`Bool true)
let buffer ?(dtype = Dtype.Int32) n = Shape.param ~shape:[ Int n ] 0 dtype
let load buf idx = Ops.load (Ops.index buf [ idx ]) []
let alt = Ops.int ~dtype:Int32 0
let gated_load buf idx = Ops.load (Ops.index buf [ idx ]) [ alt; flag "g" ]

let store ?gate buf idx =
  Ops.store ?gate (Ops.index buf [ idx ]) (Ops.int ~dtype:Int32 0)

(* [fresh ()] is a float variable that construction has never built: its name is
   new, and so is every node over it. *)
let fresh =
  let n = ref 0 in
  fun () ->
    incr n;
    fvar (Printf.sprintf "fresh%d" !n)

(* Specifications *)

let verdicts = recorded "verdicts.golden"

(* The rows where tolk's verdicts differ from tinygrad's, by the entry of
   DIVERGENCES.md that says why. *)
let verdicts_differ =
  [
    ( "D60",
      [
        "case=a sum of weak integers";
        "case=an end of a store over a range";
        "case=an end of a store over two ranges";
        "case=an end of a store over an int32 range";
        "case=an end of a store over a constant";
        "case=an end of a store over a launch dimension";
        "case=an end of a store over a loop header";
        "case=an end of a value over a range";
        "case=an end of a store alone";
      ] );
    ( "D146",
      [
        "case=a program of a parameter";
        "case=a program of a global buffer after a store";
      ] );
    ( "D170",
      [ "case=a named parameter on a device"; "case=a variable on a device" ] );
    ("D220", [ "case=a load after a store" ]);
    ( "D350",
      [
        "case=a shrink";
        "case=a shrink of a parameter by cast constants";
        "case=a shrink of a bitcast parameter by cast constants";
        "case=a shrink of a parameter by a variable";
        "case=a shrink of a load by cast constants";
      ] );
  ]

(* Each row's five verdicts. [full]'s order shows where [tensor] and [program]
   disagree: [a named parameter on a device] passes [tensor] and fails
   [program], and [full], which asks [tensor]'s rules first, accepts it. *)
let judged_as_its_golden cell =
  let u = node_of verdicts cell in
  equal
    (list (pair string verdict))
    (List.map (fun (name, _) -> (name, verdict_of_cell (cell name))) specs)
    (List.map (fun (name, spec) -> (name, judge spec u)) specs)

let specifications =
  group "Specifications"
    [
      group "each judges each recorded node as its golden states"
        [
          Golden.cases ~differs:verdicts_differ "verdicts.golden"
            judged_as_its_golden;
        ];
    ]

(* shared *)

let accesses = recorded "bounds.golden"

let bounds_differ =
  [
    ( "D506",
      [
        "case=a load of a shrink";
        "case=a store to a shrink past the last element";
      ] );
  ]

let accepted_as_its_golden ~checked column cell =
  let u = node_of accesses cell in
  let judged () = judge Spec.shared u in
  equal verdict
    (verdict_of_cell (cell column))
    (if checked then checking_bounds judged else judged ())

type kind = Load | Store

(* An access through an index of [dtype] whose bounds are [lo] and [hi] into
   storage of [n] elements. *)
type access = { dtype : Dtype.t; lo : int; hi : int; n : int; kind : kind }

let kind_name = function Load -> "a load" | Store -> "a store"

let pp_access ppf a =
  Format.fprintf ppf "%s through an %a index in [%d, %d] into %d"
    (kind_name a.kind) Dtype.pp a.dtype a.lo a.hi a.n

(* Accesses drawn across both ends of storage of up to 20 elements, none
   included. *)
let gen_access =
  Gen.(
    with_pp pp_access
      (let* n = int_range 0 20 in
       let* lo = int_range (-3) (n + 2) in
       let* hi = int_range lo (n + 3) in
       let* dtype = of_list ~pp:Dtype.pp [ Dtype.Int32; Dtype.Int64 ] in
       let+ kind =
         of_list
           ~pp:(fun ppf k -> Format.pp_print_string ppf (kind_name k))
           [ Load; Store ]
       in
       { dtype; lo; hi; n; kind }))

let in_bounds a = 0 <= a.lo && a.hi < a.n

let access a =
  let buf = buffer a.n and idx = var ~dtype:a.dtype "i" a.lo a.hi in
  match a.kind with Load -> load buf idx | Store -> store buf idx

(* The storage's ends, storage of no element, and an int64 index past what an
   int32 holds. *)
let access_examples =
  let a lo hi n = { dtype = Dtype.Int32; lo; hi; n; kind = Load } in
  let big = (1 lsl 31) + 1 in
  [
    a 0 3 4;
    a 0 4 4;
    a (-1) 2 4;
    a 0 0 0;
    { dtype = Int64; lo = big - 2; hi = big - 1; n = big; kind = Store };
    { dtype = Int64; lo = 0; hi = big; n = big; kind = Load };
  ]

let passes_iff_in_bounds a =
  cover "the last element" (a.hi = a.n - 1);
  cover "one past the last element" (a.hi = a.n);
  cover "one before the first element" (a.lo = -1);
  cover "storage of no element" (a.n = 0);
  cover "a store" (a.kind = Store);
  cover "an int64 index" (a.dtype = Int64);
  equal verdict
    (Some (in_bounds a))
    (checking_bounds (fun () -> judge Spec.shared (access a)))

let gate_changes_nothing a =
  cover "an access in bounds" (in_bounds a);
  cover "an access out of bounds" (not (in_bounds a));
  let buf = buffer a.n and idx = var ~dtype:a.dtype "i" a.lo a.hi in
  checking_bounds (fun () ->
      equal ~msg:"load" verdict
        (judge Spec.shared (load buf idx))
        (judge Spec.shared (gated_load buf idx));
      equal ~msg:"store" verdict
        (judge Spec.shared (store buf idx))
        (judge Spec.shared (store ~gate:(flag "g") buf idx)))

(* A shrink of [size] elements of storage of [n], from a start whose bounds are
   [lo] and [hi]: the start is a variable, so the shrink stays. *)
let shrink_of (lo, hi, size, n) =
  let s = var "s" lo hi in
  Shape.shrink (buffer n) [ Some (Sym s, Sym Ops.O.(s + int size)) ]

let gen_shrink =
  Gen.(
    with_pp
      (fun ppf (lo, hi, size, n) ->
        Format.fprintf ppf "%d elements from [%d, %d] of %d" size lo hi n)
      (let* n = int_range 0 20 in
       let* lo = int_range (-2) (n + 1) in
       let* hi = int_range lo (n + 2) in
       let+ size = int_range 0 (n + 1) in
       (lo, hi, size, n)))

let shrink_passes_iff_in_bounds ((lo, hi, size, n) as s) =
  cover "a run ending at the last element" (hi + size = n);
  cover "a run one past the last element" (hi + size = n + 1);
  cover "a start before the first element" (lo = -1);
  equal verdict
    (Some (0 <= lo && hi + size <= n))
    (checking_bounds (fun () -> judge Spec.shared (Ops.load (shrink_of s) [])))

(* A call, in a loop of three trips, of a body adding one to its parameter of
   four floats, whose start is known to [align] bytes, on rows [stride] floats
   apart. *)
let call_on_rows ~align stride =
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let rows = Shape.param ~shape:[ Int ((2 * stride) + 4) ] 1 Float32 in
  let start = Ops.O.(r * int stride) in
  let row = Shape.shrink rows [ Some (Sym start, Sym Ops.O.(start + int 4)) ] in
  let p = Shape.param ~shape:[ Int 4 ] ~align 0 Float32 in
  let k = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
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

(* Calls on rows of floats whose parameter assumes a start: a row a float apart
   starts on 4 bytes only. [program] inherits [shared]'s rule and is tested on
   its own; [tensor] takes any start. *)
let starts spec =
  cases "accepts a call whose arguments start where its parameters assume"
    ~id:(fun (align, stride, _) ->
      Printf.sprintf "rows %d floats apart, a parameter on %d bytes" stride
        align)
    [ (16, 5, false); (16, 4, true); (4, 5, true) ]
    (fun (align, stride, fits) ->
      equal verdict (Some fits) (judge spec (call_on_rows ~align stride)))

let prints_unproved_index () =
  let idx = var "i" (-1) 2 and gate = flag "g" in
  let u = Ops.load (Ops.index (buffer 4) [ idx ]) [ alt; gate ] in
  ignore (output ());
  let judged = checking_bounds (fun () -> judge Spec.shared u) in
  let printed = output () in
  equal verdict (Some false) judged;
  List.iter
    (fun sub -> contains ~sub printed)
    [
      "-1";
      Render.render ~simplify:false idx;
      Render.render ~simplify:false gate;
      "solver";
    ]

(* Two elements from [1, 3] of 4: the run may end at 5. *)
let prints_unproved_shrink () =
  let gate = flag "g" in
  let u = Ops.load (shrink_of (1, 3, 2, 4)) [ alt; gate ] in
  ignore (output ());
  let judged = checking_bounds (fun () -> judge Spec.shared u) in
  let printed = output () in
  equal verdict (Some false) judged;
  List.iter
    (fun sub -> contains ~sub printed)
    [ "[1, 5)"; Render.render ~simplify:false gate; "solver" ]

let shared =
  group "shared"
    [
      group
        "without check_oob, accepts each recorded access as its golden states"
        [
          Golden.cases ~differs:bounds_differ "bounds.golden"
            (accepted_as_its_golden ~checked:false "unchecked");
        ];
      group "with check_oob, accepts each recorded access as its golden states"
        [
          Golden.cases ~differs:bounds_differ "bounds.golden"
            (accepted_as_its_golden ~checked:true "checked");
        ];
      prop "without check_oob, every access through an index passes"
        ~examples:access_examples gen_access (fun a ->
          equal verdict (Some true) (judge Spec.shared (access a)));
      prop
        "with check_oob, an access through an index passes iff the bounds of \
         its index lie within the storage"
        ~examples:access_examples gen_access passes_iff_in_bounds;
      prop "with check_oob, a gate never changes whether an access passes"
        ~examples:access_examples gen_access gate_changes_nothing;
      prop
        "with check_oob, an access through a shrink passes iff its run lies \
         within the storage"
        ~examples:[ (0, 0, 4, 4); (1, 1, 4, 4); (-1, -1, 2, 4); (0, 0, 0, 0) ]
        gen_shrink shrink_passes_iff_in_bounds;
      test
        "with check_oob, an index it cannot prove prints its bounds, the index \
         and the gate"
        prints_unproved_index;
      test
        "with check_oob, a shrink it cannot prove prints its bounds and the \
         gate"
        prints_unproved_shrink;
      starts Spec.shared;
    ]

(* tensor and program *)

let tensor =
  group "tensor"
    [
      test "accepts a call whatever its arguments' starts" (fun () ->
          equal verdict (Some true)
            (judge Spec.tensor (call_on_rows ~align:16 5)));
    ]

let program =
  group "program"
    [
      test
        "refuses a call of rows a float apart for a parameter that starts on \
         16 bytes" (fun () ->
          equal verdict (Some false)
            (judge Spec.program (call_on_rows ~align:16 5)));
    ]

(* kernel_graph *)

(* [row axis_type] is a row of four of twelve floats that moves with a range of
   three trips of [axis_type], with the range. *)
let row axis_type =
  let r = Ops.range ~axis_type (Int 3) [ 100 ] in
  let rows = Shape.param ~shape:[ Int 12 ] 1 Float32 in
  ( Shape.shrink rows
      [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 4)) ],
    r )

(* A loop of a call of the row over the range, the row and the range. *)
let looped axis_type =
  let v, r = row axis_type in
  let body = Ops.v Op.Linear ~src:[] in
  (Ops.end_ (Ops.call ~precompile:true body [ v ]) [ r ], v, r)

let axis_name = function
  | Ops.Axis_type.Global -> "Global"
  | Local -> "Local"
  | Warp -> "Warp"
  | Weak -> "Weak"
  | Loop -> "Loop"
  | Upcast -> "Upcast"
  | Placeholder -> "Placeholder"

let loops =
  group "loops of calls"
    [
      test "accepts an end of a call over loop ranges" (fun () ->
          let e, _, _ = looped Loop in
          equal verdict (Some true) (judge Spec.kernel_graph e));
      test "accepts a view that moves with loop ranges, and its offset"
        (fun () ->
          let _, v, _ = looped Loop in
          equal (pair verdict verdict) (Some true, Some true)
            (judge Spec.kernel_graph v, judge Spec.kernel_graph (Ops.nth v 1)));
      test "accepts a loop range" (fun () ->
          let _, _, r = looped Loop in
          equal verdict (Some true) (judge Spec.kernel_graph r));
      cases "refuses a loop of a call over a range of any other kind"
        ~id:axis_name
        Ops.Axis_type.[ Global; Local; Warp; Weak; Upcast; Placeholder ]
        (fun axis_type ->
          let e, v, r = looped axis_type in
          equal
            (triple verdict verdict verdict)
            (Some false, Some false, Some false)
            ( judge Spec.kernel_graph e,
              judge Spec.kernel_graph v,
              judge Spec.kernel_graph r ));
      test
        "accepts a loop's call whose scalar argument is an expression of its \
         range" (fun () ->
          let v, r = row Loop in
          let trip = Ops.O.(int 2 - r) in
          let call =
            Ops.call ~precompile:true (Ops.v Op.Linear ~src:[]) [ v; trip ]
          in
          equal
            (triple verdict verdict verdict)
            (Some true, Some true, Some true)
            ( judge Spec.kernel_graph (Ops.end_ call [ r ]),
              judge Spec.kernel_graph call,
              judge Spec.kernel_graph trip ));
      test "refuses a weak sum of a weak integer variable" (fun () ->
          equal verdict (Some false)
            (judge Spec.kernel_graph
               Ops.O.(var ~dtype:Weak_int "n" 0 8 + int 1)));
    ]

let kernel_graph = group "kernel_graph" [ loops; starts Spec.kernel_graph ]

(* type_verify *)

let failing = recorded "failures.golden"

let failed_as_its_golden cell =
  let u = node_of failing cell in
  equal ~msg:"tensor" (option string)
    (failure_of_cell (cell "tensor"))
    (failure ~calls:Enter Spec.tensor u);
  equal ~msg:"program" (option string)
    (failure_of_cell (cell "program"))
    (failure ~calls:Enter Spec.program u);
  equal ~msg:"outside calls" (option string)
    (failure_of_cell (cell "tensor_outside_calls"))
    (failure ~calls:Skip Spec.tensor u)

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
let binary = Op.[ Add; Mul; Max; Cmplt; Cmpne; And; Xor; Shl; Truncdiv; Mod ]
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

let failed_at msg = Scanf.sscanf_opt msg "UOp verification failed at %d " Fun.id

let fails_at_first_refused steps =
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
    specs

let ill_typed_graph = Ops.sink [ Ops.v ~src:[ fvar "a"; fvar "b" ] Op.And ]

let type_verify =
  group "type_verify"
    [
      group "fails on each recorded graph with the message its golden states"
        [ Golden.cases "failures.golden" failed_as_its_golden ];
      prop
        "fails at the first node, sources first, that the specification does \
         not accept"
        ~examples:decided_by_every_spec graphs fails_at_first_refused;
      test "prints nothing when DEBUG is below 3" (fun () ->
          Setting.context
            [ B (Setting.debug, 2) ]
            (fun () ->
              ignore (failure ~calls:Enter Spec.shared ill_typed_graph));
          equal string "" (output ()));
      group
        "a failure first prints the graph's nodes when DEBUG is 3 or more, as \
         its golden states"
        [
          Golden.text "debug_listing.golden" (fun () ->
              Setting.context
                [ B (Setting.debug, 3); B (Setting.no_color, true) ]
                (fun () ->
                  ignore
                    (require_some ~msg:"type_verify failed"
                       (failure ~calls:Enter Spec.shared ill_typed_graph)));
              output ());
        ];
    ]

(* Construction *)

let construction =
  group "construction"
    [
      test "builds a node that breaks the full specification, whatever SPEC"
        (fun () ->
          let ill_typed () = Ops.v ~src:[ fresh (); fvar "b" ] Op.And in
          List.iter
            (fun level ->
              let u = Setting.context [ B (Setting.spec, level) ] ill_typed in
              equal verdict (Some false) (judge Spec.full u))
            [ 0; 1; 2; 3 ]);
    ]

let () =
  exit
    (run "Tolk.Spec"
       [
         specifications;
         shared;
         tensor;
         program;
         kernel_graph;
         type_verify;
         construction;
       ])
