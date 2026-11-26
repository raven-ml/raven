open Windtrap
open Tolk

let uop = Testable.make ~pp:Ops.pp ~equal:Ops.equal

let var name lo hi =
  Ops.variable name (`Int (Bigint.of_int lo)) (`Int (Bigint.of_int hi))

let a = var "a" 1 5
let b = var "b" 0 9

(* The cases of an input golden are its sink's sources, in order, read once. *)
let cases_of file = lazy (Ops.src (Golden.sink file))
let expressions = cases_of "expressions.golden"
let symbolic = cases_of "symbolic.golden"
let trees = cases_of "trees.golden"
let case cases cell = List.nth (Lazy.force cases) (int_of_string (cell "src"))

(* What [f ()] returns, or the exception it raises. *)
let outcome f = match f () with s -> Ok s | exception e -> Error e

let exn =
  Testable.make
    ~pp:(fun ppf e -> Format.pp_print_string ppf (Printexc.to_string e))
    ~equal:( = )

let written = result string exn

(* Reading an expression back *)

let variables = List.map (fun name -> (name, var name 0 9)) [ "a"; "b"; "c" ]
let comparisons = Op.[ Cmplt; Cmpne; Cmpeq ]
let is_comparison u = List.exists (Op.equal (Ops.op u)) comparisons

(* Two-character symbols come first, so that [<] never reads the start of
   [<<]. *)
let symbols =
  Op.
    [
      ("//", Div);
      ("<<", Shl);
      (">>", Shr);
      ("!=", Cmpne);
      ("==", Cmpeq);
      ("*", Mul);
      ("%", Mod);
      ("+", Add);
      ("-", Sub);
      ("&", And);
      ("^", Xor);
      ("|", Or);
      ("<", Cmplt);
    ]

(* Python's precedence levels, from the loosest. *)
let levels =
  Op.
    [
      comparisons;
      [ Or ];
      [ Xor ];
      [ And ];
      [ Shl; Shr ];
      [ Add; Sub ];
      [ Mul; Div; Mod ];
    ]

(* [read s] is the tree of binary operations over [variables] and integer
   literals that [s] denotes, read with Python's precedence and associating to
   the left, and the positions of the pairs of parentheses in [s] that [render]
   keeps whatever precedence says: the whole text's, a comparison's and a
   comparison's operands'. *)
let read s =
  let pos = ref 0 and kept = ref [ (0, String.length s - 1) ] in
  let at c = !pos < String.length s && s.[!pos] = c in
  let fail () = failwith (Printf.sprintf "cannot read %S at %d" s !pos) in
  let operator () =
    List.find_opt
      (fun (sym, _) ->
        String.length s - !pos >= String.length sym
        && String.sub s !pos (String.length sym) = sym)
      symbols
  in
  let literal () =
    let start = !pos in
    if at '-' then incr pos;
    while !pos < String.length s && s.[!pos] >= '0' && s.[!pos] <= '9' do
      incr pos
    done;
    match int_of_string_opt (String.sub s start (!pos - start)) with
    | Some n -> Ops.int n
    | None -> fail ()
  in
  let rec level = function
    | [] -> primary ()
    | ops :: tighter ->
        let operand () =
          let start = !pos in
          let e = level tighter in
          (e, (start, !pos - 1))
        in
        let rec continue (left, span) =
          match operator () with
          | Some (sym, op) when List.exists (Op.equal op) ops ->
              pos := !pos + String.length sym;
              let right, right_span = operand () in
              if List.exists (Op.equal op) comparisons then
                kept := span :: right_span :: !kept;
              continue
                (Ops.v op ~src:[ left; right ], (fst span, snd right_span))
          | _ -> left
        in
        continue (operand ())
  and primary () =
    if at '(' then (
      let start = !pos in
      incr pos;
      let e = level levels in
      if not (at ')') then fail ();
      if is_comparison e then kept := (start, !pos) :: !kept;
      incr pos;
      e)
    else
      match
        if !pos < String.length s then
          List.assoc_opt (String.make 1 s.[!pos]) variables
        else None
      with
      | Some v ->
          incr pos;
          v
      | None -> literal ()
  in
  let e = level levels in
  if !pos <> String.length s then fail ();
  (e, !kept)

let parse s = fst (read s)

(* The pairs of parentheses of [s], as the positions of each. *)
let pairs s =
  let opened = ref [] and found = ref [] in
  String.iteri
    (fun i c ->
      match (c, !opened) with
      | '(', _ -> opened := i :: !opened
      | ')', j :: rest ->
          found := (j, i) :: !found;
          opened := rest
      | _ -> ())
    s;
  !found

let without s (i, j) =
  String.sub s 0 i
  ^ String.sub s (i + 1) (j - i - 1)
  ^ String.sub s (j + 1) (String.length s - j - 1)

let operations u =
  List.filter (fun n -> Ops.src n <> []) (Ops.toposort ~calls:Enter u)

(* Every pair of parentheses of [render]'s text that [read] does not find kept
   is needed: without it the text reads as another tree, or not at all. *)
let needs_its_parentheses u =
  let s = Render.render ~simplify:false u in
  let kept = snd (read s) in
  let checked = List.filter (fun p -> not (List.mem p kept)) (pairs s) in
  cover "a pair the law checks" (checked <> []);
  cover "a pair precedence drops"
    (List.length (pairs s) < List.length (operations u));
  cover "a comparison under & ^ or |"
    (List.exists
       (fun n ->
         List.exists (Op.equal (Ops.op n)) Op.[ And; Xor; Or ]
         && List.exists is_comparison (Ops.src n))
       (operations u));
  List.iter
    (fun p ->
      let s' = without s p in
      match parse s' with
      | tree -> not_equal ~msg:s' uop u tree
      | exception Failure _ -> ())
    checked

(* Trees of the binary operations *)

let binary ops operand =
  let open Gen in
  let+ op = of_list ~pp:Op.pp ops and+ l = operand and+ r = operand in
  Ops.v op ~src:[ l; r ]

let leaf =
  Gen.one_of
    [
      Gen.of_list ~pp:Ops.pp (List.map snd variables);
      Gen.map Ops.int (Gen.int_range (-3) 9);
    ]

(* A shift count is a leaf or one operation of leaves, at most 18: a count that
   is itself a shift folds to an integer of millions of bits, which takes
   seconds to multiply and gigabytes to shift by. *)
let shift operand =
  let open Gen in
  let+ op = of_list ~pp:Op.pp Op.[ Shl; Shr ]
  and+ l = operand
  and+ r = one_of [ leaf; binary Op.[ Add; Sub; And; Or; Xor ] leaf ] in
  Ops.v op ~src:[ l; r ]

let rec integer depth =
  if depth = 0 then leaf
  else
    let operand = integer (depth - 1) in
    Gen.frequency
      [
        (5, leaf);
        (12, binary Op.[ Add; Sub; Mul; Div; Mod; And; Or; Xor ] operand);
        (3, shift operand);
      ]

let comparison = binary comparisons (integer 2)

let rec boolean depth =
  if depth = 0 then comparison
  else
    Gen.frequency
      [ (1, comparison); (2, binary Op.[ And; Or; Xor ] (boolean (depth - 1))) ]

let expression = Gen.with_pp Ops.pp (Gen.one_of [ integer 4; boolean 2 ])

(* render *)

(* [render u] is [render ~simplify:false] of [Shape.simplify u], or raises as
   simplifying [u] does. *)
let simplifies_then_writes u =
  equal written
    (outcome (fun () -> Render.render ~simplify:false (Shape.simplify u)))
    (outcome (fun () -> Render.render u))

(* The negative-shift row of the README's exclusions: where tinygrad raises
   CPython's ValueError on a shift by a negative count, tolk declines to fold
   the shift, so the golden has no text to compare with, and the tree is held to
   render's definition alone. *)
let simplified_tree cell =
  let u = case trees cell in
  match cell "simplified" with
  | "raises ValueError" -> simplifies_then_writes u
  | "raises ZeroDivisionError" ->
      raises Division_by_zero (fun () -> Render.render u)
  | rendered -> equal string rendered (Render.render u)

(* [(18<<18)*2 << (18<<18)*2] simplifies to [9437184 * 2^9437184], an integer of
   2840883 digits. *)
let huge_shift =
  let x = Ops.O.(Ops.int 3 * Ops.int 6) in
  let y = Ops.shl x x in
  let z = Ops.add y y in
  Ops.shl z z

let movement op src = Ops.v op ~src

let after_simplifying =
  group "after simplifying"
    [
      group "writes each expression as its golden states"
        [
          Golden.cases "expressions_rendered.golden" (fun cell ->
              equal string (cell "simplified")
                (Render.render (case expressions cell)));
          Golden.cases "symbolic_rendered.golden" (fun cell ->
              equal string (cell "simplified")
                (Render.render (case symbolic cell)));
          Golden.cases "trees_rendered.golden" simplified_tree;
        ];
      (* The example's divisor is always 0, which simplifying refuses: a random
         tree has one in about 3% of cases, a rate 100 cases miss once in 20
         runs. *)
      prop "writes the simplified node, or raises as simplifying it does"
        ~examples:[ Ops.O.(a % Ops.int 0) ]
        expression
        (fun u ->
          let raised = Result.is_error (outcome (fun () -> Shape.simplify u)) in
          cover "renders" (not raised);
          cover "raises" raised;
          simplifies_then_writes u);
      test "raises Invalid_argument on an integer Bigint.to_string refuses"
        (fun () ->
          raises_match (Exn.invalid_arg ~sub:"4300 digits") (fun () ->
              Render.render huge_shift));
    ]

let render =
  group "render"
    [
      group "writes each expression as its golden states"
        [
          Golden.cases "expressions_rendered.golden" (fun cell ->
              equal string (cell "render")
                (Render.render ~simplify:false (case expressions cell)));
          Golden.cases "symbolic_rendered.golden" (fun cell ->
              equal string (cell "render")
                (Render.render ~simplify:false (case symbolic cell)));
        ];
      group "writes any other node as Ops.pp prints it"
        [
          Golden.text "unrendered_rendered.golden" (fun () ->
              String.concat "\n"
                (List.map
                   (Render.render ~simplify:false)
                   (Ops.src (Golden.sink "unrendered.golden"))));
        ];
      prop
        "writes a tree of the infix operations over variables and integers \
         that reads back as that tree"
        expression
        (Law.round_trip uop string (Render.render ~simplify:false) parse);
      prop
        "writes a tree of the infix operations over variables and integers \
         with no parentheses but those precedence needs, a comparison's, its \
         operands' and the whole's"
        expression needs_its_parentheses;
      test "writes a movement's sizes simplified, whatever simplify" (fun () ->
          equal string "a.reshape((b,))"
            (Render.render ~simplify:false
               (movement Op.Reshape [ a; Ops.O.(b + Ops.int 0) ])));
      test
        "raises what Shape.simplify raises on a movement's sizes, whatever \
         simplify" (fun () ->
          let size = Ops.O.(b % Ops.int 0) in
          let raised =
            require_error (outcome (fun () -> Shape.simplify size))
          in
          let u = movement Op.Reshape [ a; size ] in
          raises raised (fun () -> Render.render ~simplify:false u));
      test "raises Invalid_argument on a movement whose size is a float"
        (fun () ->
          let u = movement Op.Reshape [ a; Ops.float 1.5 ] in
          raises_match (Exn.invalid_arg ~sub:"is not a size") (fun () ->
              Render.render ~simplify:false u));
      test "raises Invalid_argument on a permutation whose axis is a node"
        (fun () ->
          let u = movement Op.Permute [ a; b ] in
          raises_match (Exn.invalid_arg ~sub:"symbolic axis") (fun () ->
              Render.render ~simplify:false u));
      after_simplifying;
    ]

(* srender *)

let ints = Gen.frequency [ (6, Gen.int_range (-1000) 1000); (1, Gen.int) ]

let srender =
  group "srender"
    [
      prop "writes an integer in decimal"
        ~examples:[ min_int; min_int + 1; -1; 0; 1; max_int - 1; max_int ]
        ints
        (fun n -> equal string (Int.to_string n) (Render.srender (Int n)));
      prop "writes a node as render does" ~examples:[ a ] expression (fun u ->
          equal written
            (outcome (fun () -> Render.render u))
            (outcome (fun () -> Render.srender (Sym u))));
    ]

(* render_uir *)

let uir uops =
  Setting.context
    [ B (Setting.no_color, true) ]
    (fun () -> Render.render_uir uops)

let uir_colored uops =
  Setting.context
    [ B (Setting.no_color, false) ]
    (fun () -> Render.render_uir uops)

(* A graph golden's sink and what it reaches, in order. *)
let listing file = Ops.toposort ~calls:Enter (Golden.sink file)

(* Lines separated by newlines, the last not ended: an empty text has none. *)
let lines = function "" -> [] | s -> String.split_on_char '\n' s

(* A constant, or a stack of constants such as a variable's empty stack of
   bounds, has no line. *)
let has_line u =
  match Ops.op u with
  | Op.Const -> false
  | Op.Stack -> not (List.for_all (fun s -> Ops.op s = Op.Const) (Ops.src u))
  | _ -> true

let line_per_node u =
  let uops = Ops.toposort ~calls:Enter u in
  let nodes = List.filter has_line uops in
  cover "a line" (nodes <> []);
  cover "a constant" (List.exists (fun u -> Ops.op u = Op.Const) uops);
  let written = lines (uir uops) in
  equal int (List.length nodes) (List.length written);
  List.iteri
    (fun i (u, line) ->
      starts_with
        ~prefix:
          (Printf.sprintf "%%%d = %s" i
             (String.lowercase_ascii (Op.name (Ops.op u))))
        line)
    (List.combine nodes written)

(* tinygrad prints the text, ended. *)
let render_uir =
  group "render_uir"
    [
      group "writes each graph as its golden states"
        [
          Golden.text "program_uir.golden" (fun () ->
              uir (listing "program.golden") ^ "\n");
          Golden.text "program_uir_colored.golden" (fun () ->
              uir_colored (listing "program.golden") ^ "\n");
          Golden.text "args_uir.golden" (fun () ->
              uir (listing "args.golden") ^ "\n");
          Golden.text "symbolic_vars_uir.golden" (fun () ->
              uir (listing "symbolic_vars.golden") ^ "\n");
        ];
      test "writes nothing for no node" (fun () -> equal string "" (uir []));
      test "writes a constant in its user" (fun () ->
          let x = Ops.add (Ops.float 1.5) (Ops.float (-0.)) in
          equal string "%0 = add 1.5, -0.0" (uir (Ops.toposort ~calls:Enter x)));
      prop
        "writes a line per node that is no constant nor stack of constants, \
         numbered from 0"
        ~examples:[ Ops.int 3 ]
        expression line_per_node;
      test
        "raises Not_found on a source with a line that is not among the nodes"
        (fun () -> raises Not_found (fun () -> uir [ Ops.add a (Ops.int 1) ]));
    ]

let () = exit (run "Tolk.Render" [ render; srender; render_uir ])
