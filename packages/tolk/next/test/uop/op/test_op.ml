open Windtrap
open Tolk_next.Uop

(* The operations, in the order the interface declares them. *)
let declared =
  Op.
    [
      Special;
      Buffer;
      Alloc;
      Noop;
      Param;
      Call;
      Program;
      Linear;
      Source;
      Binary;
      Sink;
      After;
      Group;
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
      Reciprocal;
      Neg;
      Trunc;
      Add;
      Mul;
      Shl;
      Shr;
      Cdiv;
      Max;
      Cmod;
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
      Floordiv;
      Floormod;
      Where;
      Mulacc;
      Barrier;
      Range;
      If;
      End;
      Endif;
      Backedge;
      Const;
      Custom;
      Customi;
      Ins;
      Contiguous_backward;
      Detach;
      Stage;
      Copy;
      Mselect;
      Mstack;
      Custom_function;
      Reshape;
      Permute;
      Expand;
      Pad;
      Flip;
      Unshard;
      Reduce;
      Allreduce;
    ]

(* Witnesses and generators *)

let op =
  Testable.make ~pp:Op.pp ~equal:Op.equal |> Testable.with_compare Op.compare

let ops = list op
let set_w = Testable.make ~pp:Op.Set.pp ~equal:Op.Set.equal
let gen_op = Gen.of_list ~pp:Op.pp declared
let gen_ops = Gen.list gen_op

let gen_set =
  Gen.map Op.Set.of_list (Gen.subsequence ~pp:Op.pp declared)
  |> Gen.with_pp Op.Set.pp

let gen_member g = Gen.of_list ~pp:Op.pp (Op.Set.to_list g)
let empty = Op.Set.of_list []
let show o = Format.asprintf "%a" Op.pp o

(* A set printed on one line: its printer may break lines at the margin. *)
let show_set g =
  Format.asprintf "%a"
    (fun ppf g ->
      Format.pp_set_margin ppf 10_000;
      Op.Set.pp ppf g)
    g

(* [within g g'] asserts that [g'] holds every operation of [g], and prints
   those it does not hold on failure. *)
let within ?msg g g' = equal ?msg set_w empty (Op.Set.diff g g')

(* The golden has one row per tinygrad operation, in declaration order: its
   printed form, its integer value, and whether each of tinygrad's groups holds
   it. *)

let golden = lazy (Golden.rows "ops.golden")
let printed_ops () = List.map (fun cell -> cell "op") (Lazy.force golden)

let counterpart printed =
  List.find_opt (fun o -> String.equal (show o) printed) declared

let value o =
  let row =
    List.find
      (fun cell -> String.equal (cell "op") (show o))
      (Lazy.force golden)
  in
  int_of_string (row "value")

(* The operations of tinygrad that tolk.next leaves out. *)
let excluded = [ "Ops.REWRITE_ERROR"; "Ops.PYLITERAL" ]

(* Operations *)

let operations =
  group "Op"
    [
      test "declares tinygrad's operations in tinygrad's order" (fun () ->
          equal (list string)
            (List.filter
               (fun op -> Option.is_some (counterpart op))
               (printed_ops ()))
            (List.map show declared));
      test "has no counterpart for exactly REWRITE_ERROR and PYLITERAL"
        (fun () ->
          equal (list string) excluded
            (List.filter
               (fun op -> Option.is_none (counterpart op))
               (printed_ops ())));
      test "to_int is the position in declaration order, from 0" (fun () ->
          equal (list int)
            (List.init (List.length declared) Fun.id)
            (List.map Op.to_int declared));
      cases ~name:show "pp is Ops. followed by the name" declared (fun o ->
          equal string ("Ops." ^ Op.name o) (show o));
      test "names words in upper case separated by underscores" (fun () ->
          equal string "ADD" (Op.name Op.Add);
          equal string "CONTIGUOUS_BACKWARD" (Op.name Op.Contiguous_backward));
    ]

(* Equality and order *)

let order =
  group "Op order"
    [
      prop "equal is an equivalence" (Gen.pair gen_op gen_op)
        (Law.equivalence op);
      prop "compare is a total order agreeing with equal"
        (Gen.triple gen_op gen_op gen_op)
        (Law.order op);
      prop "compare agrees with tinygrad's integer order"
        (Gen.pair gen_op gen_op) (fun (a, b) ->
          equal int
            (Int.compare (value a) (value b))
            (Int.compare (Op.compare a b) 0));
      prop "compare agrees with to_int" (Gen.pair gen_op gen_op) (fun (a, b) ->
          equal int
            (Int.compare (Op.to_int a) (Op.to_int b))
            (Int.compare (Op.compare a b) 0));
    ]

(* Sets *)

let membership =
  group "Set.mem"
    [
      prop "of of_list is membership of the list" (Gen.pair gen_op gen_ops)
        (fun (o, l) ->
          equal bool
            (List.exists (Op.equal o) l)
            (Op.Set.mem o (Op.Set.of_list l)));
      prop "of union is membership of either"
        (Gen.triple gen_op gen_set gen_set) (fun (o, g, g') ->
          equal bool
            (Op.Set.mem o g || Op.Set.mem o g')
            (Op.Set.mem o (Op.Set.union g g')));
      prop "of diff is membership of the first and not the second"
        (Gen.triple gen_op gen_set gen_set) (fun (o, g, g') ->
          equal bool
            (Op.Set.mem o g && not (Op.Set.mem o g'))
            (Op.Set.mem o (Op.Set.diff g g')));
      prop "of all always holds" gen_op (fun o ->
          is_true (Op.Set.mem o Op.Set.all));
    ]

let diffs =
  group "Set.diff"
    [
      test "removes Threefry from alu and keeps the rest" (fun () ->
          equal ops
            (List.filter
               (fun o -> not (Op.equal o Op.Threefry))
               (Op.Set.to_list Op.Set.alu))
            (Op.Set.to_list
               (Op.Set.diff Op.Set.alu (Op.Set.of_list [ Op.Threefry ]))));
    ]

let listings =
  group "Set.to_list"
    [
      prop "of of_list is the listed operations in declaration order, once each"
        gen_ops (fun l ->
          equal ops
            (List.filter (fun o -> List.exists (Op.equal o) l) declared)
            (Op.Set.to_list (Op.Set.of_list l)));
      prop "is inverted by of_list" gen_set
        (Law.round_trip set_w ops Op.Set.to_list Op.Set.of_list);
      test "of all is the declared operations" (fun () ->
          equal ops declared (Op.Set.to_list Op.Set.all));
    ]

let equalities =
  group "Set.equal"
    [
      prop "is an equivalence" (Gen.pair gen_set gen_set)
        (Law.equivalence set_w);
      prop "is equality of to_list" (Gen.pair gen_set gen_set) (fun (g, g') ->
          equal bool
            (List.equal Op.equal (Op.Set.to_list g) (Op.Set.to_list g'))
            (Op.Set.equal g g'));
      prop "ignores the order and repetitions of of_list's list" gen_ops
        (Law.ignores ops set_w Op.Set.of_list (fun l -> List.rev_append l l));
    ]

let printing =
  group "Set.pp"
    [
      test "prints operations in declaration order between braces" (fun () ->
          equal string "{Ops.ADD, Ops.MUL}"
            (show_set (Op.Set.of_list [ Op.Mul; Op.Add; Op.Mul ])));
      test "prints the empty set as {}" (fun () ->
          equal string "{}" (show_set empty));
      prop "prints each operation with Op.pp" gen_set (fun g ->
          equal string
            ("{" ^ String.concat ", " (List.map show (Op.Set.to_list g)) ^ "}")
            (show_set g));
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
      ("Commutative", commutative);
      ("Associative", associative);
      ("Idempotent", idempotent);
      ("Reduce", reduce);
      ("Comparison", comparison);
      ("All", all);
    ]

(* The integer meaning of the operations the algebraic sets hold, a comparison
   being 1 when it holds and 0 otherwise. *)
let int_meaning o =
  let bool b = if b then 1 else 0 in
  match (o : Op.t) with
  | Add -> ( + )
  | Mul -> ( * )
  | Max -> Int.max
  | Cmpne -> fun x y -> bool (x <> y)
  | Cmpeq -> fun x y -> bool (x = y)
  | Xor -> ( lxor )
  | Or -> ( lor )
  | And -> ( land )
  | o -> failf "%a has no integer meaning in this suite" Op.pp o

(* Each tinygrad operation is left out or held by the sets its row marks. *)
let in_tinygrads_sets cell =
  match counterpart (cell "op") with
  | None -> mem string (cell "op") excluded
  | Some o ->
      List.iter
        (fun (name, g) ->
          equal ~msg:name bool
            (String.equal (cell name) "True")
            (Op.Set.mem o g))
        named

let named_sets =
  group "Named sets"
    [
      test "are every GroupOp set of tinygrad" (fun () ->
          equal (list string)
            ("op" :: "value" :: List.map fst named)
            (Golden.columns "ops.golden"));
      group "hold tinygrad's members"
        [ Golden.cases "ops.golden" in_tinygrads_sets ];
      test "alu is unary, binary and ternary" (fun () ->
          equal set_w
            (Op.Set.union Op.Set.unary
               (Op.Set.union Op.Set.binary Op.Set.ternary))
            Op.Set.alu);
      test "unary, binary and ternary are disjoint" (fun () ->
          equal set_w Op.Set.unary (Op.Set.diff Op.Set.unary Op.Set.binary);
          equal set_w Op.Set.unary (Op.Set.diff Op.Set.unary Op.Set.ternary);
          equal set_w Op.Set.binary (Op.Set.diff Op.Set.binary Op.Set.ternary));
      test "broadcastable is binary and ternary" (fun () ->
          equal set_w
            (Op.Set.union Op.Set.binary Op.Set.ternary)
            Op.Set.broadcastable);
      test "elementwise is alu, Cast and Bitcast" (fun () ->
          equal set_w
            (Op.Set.union Op.Set.alu (Op.Set.of_list [ Op.Cast; Op.Bitcast ]))
            Op.Set.elementwise);
      cases ~name:fst "is binary"
        [
          ("commutative", Op.Set.commutative);
          ("associative", Op.Set.associative);
          ("idempotent", Op.Set.idempotent);
          ("reduce", Op.Set.reduce);
          ("comparison", Op.Set.comparison);
        ]
        (fun (_, g) -> within g Op.Set.binary);
      test "reduce operations are commutative and associative" (fun () ->
          within Op.Set.reduce Op.Set.commutative;
          within Op.Set.reduce Op.Set.associative);
      prop "commutative operations commute"
        (Gen.triple (gen_member Op.Set.commutative) Gen.small_int Gen.small_int)
        (fun (o, x, y) -> Law.commutative int (int_meaning o) (x, y));
      prop "associative operations associate"
        (Gen.quad
           (gen_member Op.Set.associative)
           Gen.small_int Gen.small_int Gen.small_int)
        (fun (o, x, y, z) -> Law.associative int (int_meaning o) (x, y, z));
      prop "idempotent operations give back their operand"
        (Gen.pair (gen_member Op.Set.idempotent) Gen.small_int)
        (fun (o, x) -> equal int x (int_meaning o x x));
      test "movement leaves out Unshard" (fun () ->
          is_false (Op.Set.mem Op.Unshard Op.Set.movement));
    ]

let () =
  exit
    (run "Tolk_next.Uop.Op"
       [
         operations;
         order;
         membership;
         diffs;
         listings;
         equalities;
         printing;
         named_sets;
       ])
