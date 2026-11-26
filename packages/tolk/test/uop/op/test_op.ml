open Windtrap
open Tolk

(* The operations, in the order the interface declares them. *)
let declared =
  Op.
    [
      Special;
      Buffer;
      Alloc;
      Noop;
      Param;
      Refparam;
      Call;
      Program;
      Linear;
      Source;
      Binary;
      Sink;
      After;
      Stack;
      Getaddr;
      Index;
      Shrink;
      Load;
      Store;
      Wmma;
      Cast;
      Bitcast;
      Exp2;
      Log2;
      Sin;
      Sqrt;
      Recip;
      Neg;
      Trunc;
      Add;
      Mul;
      Shl;
      Shr;
      Truncdiv;
      Max;
      Truncmod;
      Cmplt;
      Cmpne;
      Cmpeq;
      Xor;
      Or;
      And;
      Threefry;
      Sub;
      Fdiv;
      Pow;
      Div;
      Mod;
      Where;
      Mulacc;
      Barrier;
      Range;
      If;
      End;
      Endif;
      Backedge;
      Const;
      Ins;
      Contiguous_backward;
      Detach;
      Stage;
      Copy;
      Custom_function;
      Reshape;
      Permute;
      Expand;
      Pad;
      Reduce;
    ]

(* Witnesses and generators *)

let op = Testable.make ~pp:Op.pp ~equal:Op.equal
let ops = list op
let set_w = Testable.make ~pp:Op.Set.pp ~equal:Op.Set.equal
let gen_op = Gen.of_list ~pp:Op.pp declared
let gen_ops = Gen.list gen_op

let gen_set =
  Gen.map Op.Set.of_list (Gen.subsequence ~pp:Op.pp declared)
  |> Gen.with_pp Op.Set.pp

let gen_member s = Gen.of_list ~pp:Op.pp (Op.Set.to_list s)
let empty = Op.Set.of_list []
let show o = Format.asprintf "%a" Op.pp o

(* The operations [s] holds, by [mem], in declaration order. *)
let members s = List.filter (fun o -> Op.Set.mem o s) declared

(* [within s s'] asserts that [s'] holds every operation of [s], and prints
   those it does not hold on failure. *)
let within ?msg s s' = equal ?msg set_w empty (Op.Set.diff s s')

let show_set ~margin s =
  Format.asprintf "%a"
    (fun ppf s ->
      Format.pp_set_margin ppf margin;
      Op.Set.pp ppf s)
    s

(* A margin no set reaches. *)
let wide = 10_000

(* The golden has one row per tinygrad operation: its printed form, its integer
   value, and whether each of tinygrad's groups holds it. *)

let golden = lazy (Golden.rows "ops.golden")

let counterpart printed =
  List.find_opt (fun o -> String.equal (show o) printed) declared

(* The operations of tinygrad that tolk leaves out: an error marker, Python
   literals, and the operations of several devices (see the README). *)
let excluded =
  [
    "Ops.REWRITE_ERROR";
    "Ops.MSELECT";
    "Ops.MSTACK";
    "Ops.UNSHARD";
    "Ops.ALLREDUCE";
    "Ops.PYLITERAL";
  ]

(* Operations *)

let value cell = int_of_string (cell "value")

let by_value rows =
  List.stable_sort (fun c c' -> Int.compare (value c) (value c')) rows

let operations =
  group "t"
    [
      test "declares the golden's operations in the order of their values"
        (fun () ->
          equal (list string)
            (List.filter_map
               (fun cell -> Option.map show (counterpart (cell "op")))
               (by_value (Lazy.force golden)))
            (List.map show declared));
      test "lacks only the golden's operations the README leaves out" (fun () ->
          equal (list string) excluded
            (List.filter
               (fun printed -> Option.is_none (counterpart printed))
               (List.map (fun cell -> cell "op") (Lazy.force golden))));
    ]

let equality =
  group "equal"
    [
      cases "holds of an operation and itself alone" ~id:show declared (fun o ->
          equal ops [ o ] (List.filter (Op.equal o) declared));
    ]

(* The operations [compare] puts before [o], equal to it and after it, against
   those [declared] lists before it, [o], and those it lists after. *)
let orders o =
  let i = Option.get (List.find_index (Op.equal o) declared) in
  let by sign = List.filter (fun o' -> sign (Op.compare o' o)) declared in
  equal (triple ops ops ops)
    ( List.filteri (fun j _ -> j < i) declared,
      [ o ],
      List.filteri (fun j _ -> j > i) declared )
    (by (fun c -> c < 0), by (fun c -> c = 0), by (fun c -> c > 0))

let order =
  group "compare"
    [
      cases "orders operations by their declaration order" ~id:show declared
        orders;
    ]

let positions =
  group "to_int"
    [
      test "is the position in declaration order, from 0" (fun () ->
          equal (list int)
            (List.init (List.length declared) Fun.id)
            (List.map Op.to_int declared));
    ]

let reading =
  group "of_string"
    [
      cases "reads back name" ~id:show declared (fun o ->
          equal (result op string) (Ok o) (Op.of_string (Op.name o)));
      cases "of anything but a name is an error that names it"
        ~id:(Printf.sprintf "%S")
        [ "Ops.ADD"; "add"; "Add"; " ADD"; "PYLITERAL"; "REWRITE_ERROR" ]
        (fun s -> contains ~sub:s (require_error (Op.of_string s)));
      test "of the empty string is an error" (fun () ->
          is_error ~pp:Op.pp (Op.of_string ""));
    ]

let printing =
  group "pp"
    [
      cases "is Ops. followed by the name" ~id:show declared (fun o ->
          equal string ("Ops." ^ Op.name o) (show o));
    ]

(* Sets *)

let membership =
  group "Set.mem"
    [
      prop "of of_list's set is membership of the list" gen_ops
        ~examples:[ []; declared ] (fun l ->
          equal ops
            (List.filter (fun o -> List.exists (Op.equal o) l) declared)
            (members (Op.Set.of_list l)));
    ]

let unions =
  group "Set.union"
    [
      prop "holds the operations of either set" (Gen.pair gen_set gen_set)
        (fun (s, s') ->
          equal ops
            (List.filter (fun o -> Op.Set.mem o s || Op.Set.mem o s') declared)
            (members (Op.Set.union s s')));
    ]

let diffs =
  group "Set.diff"
    [
      prop "holds the operations of the first set not in the second"
        (Gen.pair gen_set gen_set)
        ~examples:[ (Op.Set.alu, Op.Set.of_list [ Op.Threefry ]) ]
        (fun (s, s') ->
          equal ops
            (List.filter
               (fun o -> Op.Set.mem o s && not (Op.Set.mem o s'))
               declared)
            (members (Op.Set.diff s s')));
    ]

let listings =
  group "Set.to_list"
    [
      prop "is the set's operations in declaration order" gen_set
        ~examples:[ empty; Op.Set.all ] (fun s ->
          equal ops (members s) (Op.Set.to_list s));
      prop "is inverted by of_list" gen_set
        (Law.round_trip set_w ops Op.Set.to_list Op.Set.of_list);
    ]

(* Two sets, the second half the time the first rebuilt from a shuffled list
   that names each of its operations twice. *)
let gen_set_pair =
  Gen.with_pp
    (fun ppf (s, s') -> Format.fprintf ppf "(%a, %a)" Op.Set.pp s Op.Set.pp s')
    (let open Gen in
     let* s = gen_set in
     one_of
       [
         map (fun s' -> (s, s')) gen_set;
         map
           (fun l -> (s, Op.Set.of_list (l @ l)))
           (permutation (Op.Set.to_list s));
       ])

let equalities =
  group "Set.equal"
    [
      prop "is equality of to_list" gen_set_pair (fun (s, s') ->
          if List.equal Op.equal (Op.Set.to_list s) (Op.Set.to_list s') then (
            cover "the same operations" true;
            equal set_w s s')
          else (
            cover "other operations" true;
            not_equal set_w s s'));
      prop "ignores the order and repetitions of of_list's list" gen_ops
        (Law.ignores ops set_w Op.Set.of_list (fun l -> List.rev_append l l));
    ]

(* The text of [s] on one line, from the operations' own printer. *)
let one_line s =
  "{" ^ String.concat ", " (List.map show (Op.Set.to_list s)) ^ "}"

(* [text] with each line break and the indentation after it read as one
   space. *)
let unbroken text =
  String.split_on_char '\n' text |> List.map String.trim |> String.concat " "

let set_printing =
  group "Set.pp"
    [
      prop "on one line, prints each operation with Op.pp, separated by commas"
        gen_set
        ~examples:[ Op.Set.of_list [ Op.Mul; Op.Add; Op.Mul ]; empty ]
        (fun s -> equal string (one_line s) (show_set ~margin:wide s));
      prop "past the margin, breaks lines only after a comma"
        (Gen.pair gen_set (Gen.int_range 10 200))
        ~examples:[ (Op.Set.all, 78) ]
        (fun (s, margin) ->
          let text = show_set ~margin s in
          cover "a line breaks" (String.contains text '\n');
          equal string (one_line s) (unbroken text));
    ]

(* Named sets *)

let named =
  Op.Set.
    [
      ("Unary", unary);
      ("Binary", binary);
      ("Ternary", ternary);
      ("ALU", alu);
      ("Broadcastable", broadcastable);
      ("Elementwise", elementwise);
      ("Defines", defines);
      ("Irreducible", irreducible);
      ("Movement", movement);
      ("Effect", effectful);
      ("Commutative", commutative);
      ("Associative", associative);
      ("Idempotent", idempotent);
      ("Reduce", reduce);
      ("Comparison", comparison);
      ("All", all);
    ]

(* Each operation of the golden is left out, or held by exactly the sets its row
   marks. *)
let marked_sets cell =
  match counterpart (cell "op") with
  | None -> mem string (cell "op") excluded
  | Some o ->
      equal (list string)
        (List.filter
           (fun name -> String.equal (cell name) "True")
           (List.map fst named))
        (List.filter_map
           (fun (name, s) -> if Op.Set.mem o s then Some name else None)
           named)

(* An operation's meaning, as the library computes it on scalars: a comparison
   gives a Bool whatever its operands' type. *)
let apply o dt args =
  let dt = if Op.Set.mem o Op.Set.comparison then Dtype.Bool else dt in
  Ops.exec_alu o dt args

let gen_types types = Gen.of_list ~pp:(Testable.pp Dtypes.dtype) types

(* A type of operands: an integer, Bool, or, where [floats], a float, each kind
   as likely. *)
let gen_dtype ~floats =
  Gen.one_of
    ([ gen_types (Dtype.Weak_int :: Dtype.ints); gen_types [ Dtype.Bool ] ]
    @ if floats then [ gen_types (Dtype.Weak_float :: Dtype.floats) ] else [])

let kind dt =
  if Dtype.is_float dt then "on floats"
  else if Dtype.is_bool dt then "on booleans"
  else "on integers"

let is_nan = function `Float f -> Float.is_nan f | _ -> false

(* [x] and [y] are floats neither less than the other: a NaN, or two equal
   values such as [-0.] and [0.]. *)
let unordered (x : Dtype.const) (y : Dtype.const) =
  match (x, y) with `Float a, `Float b -> not (a < b || b < a) | _ -> false

(* A maximum of zeros is either zero. *)
let unsigned : Dtype.const -> Dtype.const = function
  | `Float 0. -> `Float 0.
  | c -> c

let pp_case ppf (o, dt, xs) =
  Format.fprintf ppf "(%a, %a, %a)" Op.pp o (Testable.pp Dtypes.dtype) dt
    (Testable.pp (list Dtypes.const))
    xs

(* An operation of [s], a type of operands it reads, and [n] values of that
   type. [floats o] says whether [o] reads floats. *)
let gen_case ~floats s n =
  Gen.with_pp pp_case
    (let open Gen in
     let* o = gen_member s in
     let* dt = gen_dtype ~floats:(floats o) in
     let+ xs = list ~size:(constant n) (Dtypes.value_of dt) in
     (o, dt, (xs :> Dtype.const list)))

(* The bitwise operations read integers and booleans alone. *)
let reads_floats o = not (Op.Set.mem o (Op.Set.of_list Op.[ Xor; And; Or ]))
let f x : Dtype.const = `Float x

let commutes (o, dt, xs) =
  match xs with
  | [ x; y ] ->
      cover (kind dt) true;
      let xy = apply o dt [ x; y ] and yx = apply o dt [ y; x ] in
      if Op.equal o Op.Max && unordered x y then (
        cover "Max of values neither less than the other" true;
        mem Dtypes.const xy [ x; y ];
        mem Dtypes.const yx [ x; y ])
      else equal Dtypes.const xy yx
  | _ -> fail "two operands"

let associates (o, dt, xs) =
  match xs with
  | [ x; y; z ] ->
      cover (kind dt) true;
      let g a b = apply o dt [ a; b ] in
      let left = g (g x y) z and right = g x (g y z) in
      if not (Dtype.is_float dt) then equal Dtypes.const left right
      else if List.exists is_nan xs then (
        cover "Max with a NaN operand" true;
        mem Dtypes.const left xs;
        mem Dtypes.const right xs)
      else equal Dtypes.const (unsigned left) (unsigned right)
  | _ -> fail "three operands"

let gives_back (o, dt, xs) =
  match xs with
  | [ x ] ->
      cover (kind dt) true;
      cover "a NaN" (is_nan x);
      equal Dtypes.const x (apply o dt [ x; x ])
  | _ -> fail "one operand"

let named_sets =
  group "Named sets"
    [
      test "every column of ops.golden is a named set this suite checks"
        (fun () ->
          equal (list string)
            ("op" :: "value" :: List.map fst named)
            (Golden.columns "ops.golden"));
      group "each holds the operations its column of the golden marks"
        [ Golden.cases "ops.golden" marked_sets ];
      test "alu is unary, binary and ternary" (fun () ->
          equal set_w
            (Op.Set.union Op.Set.unary
               (Op.Set.union Op.Set.binary Op.Set.ternary))
            Op.Set.alu);
      test "broadcastable is binary and ternary" (fun () ->
          equal set_w
            (Op.Set.union Op.Set.binary Op.Set.ternary)
            Op.Set.broadcastable);
      test "elementwise is alu, Cast and Bitcast" (fun () ->
          equal set_w
            (Op.Set.union Op.Set.alu (Op.Set.of_list [ Op.Cast; Op.Bitcast ]))
            Op.Set.elementwise);
      cases "commutative, associative and idempotent are binary" ~id:fst
        [
          ("commutative", Op.Set.commutative);
          ("associative", Op.Set.associative);
          ("idempotent", Op.Set.idempotent);
        ]
        (fun (_, s) -> within s Op.Set.binary);
      prop "commutative operations commute, Max up to which operand it gives"
        (gen_case ~floats:reads_floats Op.Set.commutative 2)
        ~examples:
          [
            (Op.Max, Dtype.Float32, [ f Float.nan; f 1. ]);
            (Op.Max, Dtype.Float64, [ f (-0.); f 0. ]);
            (Op.Cmpeq, Dtype.Float32, [ f Float.nan; f Float.nan ]);
            (Op.Add, Dtype.Float64, [ f Float.infinity; f Float.neg_infinity ]);
          ]
        commutes;
      prop
        "associative operations associate on integers and booleans, and Max on \
         floats up to which operand it gives"
        (gen_case ~floats:(Op.equal Op.Max) Op.Set.associative 3)
        ~examples:
          [
            (Op.Max, Dtype.Float32, [ f 1.; f Float.nan; f 2. ]);
            (Op.Max, Dtype.Float64, [ f (-0.); f 0.; f (-1.) ]);
            (Op.Max, Dtype.Weak_float, [ f 3.; f Float.neg_infinity; f 2.5 ]);
          ]
        associates;
      prop "idempotent operations give back their operand"
        (gen_case ~floats:reads_floats Op.Set.idempotent 1)
        ~examples:
          [
            (Op.Max, Dtype.Float32, [ f Float.nan ]);
            (Op.Max, Dtype.Float64, [ f (-0.) ]);
          ]
        gives_back;
      test "reduce operations are commutative and associative" (fun () ->
          within Op.Set.reduce Op.Set.commutative;
          within Op.Set.reduce Op.Set.associative);
      test "all is every operation" (fun () ->
          equal ops declared (Op.Set.to_list Op.Set.all));
    ]

let () =
  exit
    (run "Tolk.Op"
       [
         operations;
         equality;
         order;
         positions;
         reading;
         printing;
         membership;
         unions;
         diffs;
         listings;
         equalities;
         set_printing;
         named_sets;
       ])
