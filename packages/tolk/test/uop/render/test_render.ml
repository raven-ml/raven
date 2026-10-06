open Windtrap
open Tolk

let uop = Testable.make ~pp:Ops.pp ~equal:Ops.equal

let var name lo hi =
  Ops.variable name (`Int (Bigint.of_int lo)) (`Int (Bigint.of_int hi))

let a = var "a" 1 5

(* The cases of an input golden are its sink's sources, in order. *)
let sources file = Ops.src (Golden.sink file)
let case file cell = List.nth (sources file) (int_of_string (cell "src"))

(* Reading an expression back *)

let variables = List.map (fun name -> (name, var name 0 9)) [ "a"; "b"; "c" ]

(* Two-character symbols come first, so that [<] never reads the start of
   [<<]. *)
let symbols =
  Op.
    [
      ("//", Floordiv);
      ("<<", Shl);
      (">>", Shr);
      ("!=", Cmpne);
      ("*", Mul);
      ("%", Floormod);
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
      [ Cmplt; Cmpne ];
      [ Or ];
      [ Xor ];
      [ And ];
      [ Shl; Shr ];
      [ Add; Sub ];
      [ Mul; Floordiv; Floormod ];
    ]

(* [parse s] is the tree of binary operations over [variables] and integer
   literals that [s] denotes, read with Python's precedence and associating to
   the left. *)
let parse s =
  let pos = ref 0 in
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
        let rec continue left =
          match operator () with
          | Some (sym, op) when List.exists (Op.equal op) ops ->
              pos := !pos + String.length sym;
              continue (Ops.v op ~src:[ left; level tighter ])
          | _ -> left
        in
        continue (level tighter)
  and primary () =
    if at '(' then (
      incr pos;
      let e = level levels in
      if not (at ')') then fail ();
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
  e

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

let rec integer depth =
  if depth = 0 then leaf
  else
    Gen.frequency
      [
        (1, leaf);
        ( 3,
          binary
            Op.[ Add; Sub; Mul; Floordiv; Floormod; Shl; Shr; And; Or; Xor ]
            (integer (depth - 1)) );
      ]

let comparison = binary Op.[ Cmplt; Cmpne ] (integer 2)

let rec boolean depth =
  if depth = 0 then comparison
  else
    Gen.frequency
      [ (1, comparison); (2, binary Op.[ And; Or; Xor ] (boolean (depth - 1))) ]

let expression = Gen.with_pp Ops.pp (Gen.one_of [ integer 4; boolean 2 ])

let rec tag_everywhere u =
  Ops.replace
    ~src:(List.map tag_everywhere (Ops.src u))
    ~tag:(Some (Ops.Tag.Int 1)) u

(* render *)

let render =
  group "render"
    [
      Golden.cases "expressions_rendered.golden" (fun cell ->
          equal string (cell "render")
            (Render.render ~simplify:false (case "expressions.golden" cell)));
      Golden.text "unrendered_rendered.golden" (fun () ->
          String.concat "\n"
            (List.map
               (Render.render ~simplify:false)
               (sources "unrendered.golden")));
      test "writes the operators of O as they read" (fun () ->
          let b = var "b" 0 9 and c = var "c" 0 9 in
          equal string "(a*b+c)"
            (Render.render ~simplify:false Ops.O.((a * b) + c));
          equal string "((a+b)*c)"
            (Render.render ~simplify:false Ops.O.((a + b) * c)));
      prop "an expression reads back as the tree it writes" expression
        (Law.round_trip uop string (Render.render ~simplify:false) parse);
      prop "tags are not written" expression
        (Law.ignores uop string (Render.render ~simplify:false) tag_everywhere);
    ]

(* The [simplified] column follows the README's negative-shift row: where
   tinygrad raises CPython's ValueError on a shift by a negative count, a
   shift's bounds are its type's and folding it declines. A zero divisor still
   raises. *)
let simplified_tree cell =
  if cell "simplified" <> cell "tinygrad" then
    equal string "raises ValueError" (cell "tinygrad");
  let u = case "trees.golden" cell in
  match cell "simplified" with
  | "raises ZeroDivisionError" ->
      raises Division_by_zero (fun () -> Render.render u)
  | rendered -> equal string rendered (Render.render u)

let simplified =
  group "render after simplifying"
    [
      Golden.cases "expressions_rendered.golden" (fun cell ->
          equal string (cell "simplified")
            (Render.render (case "expressions.golden" cell)));
      Golden.cases "symbolic_rendered.golden" (fun cell ->
          let u = case "symbolic.golden" cell in
          equal string (cell "render") (Render.render ~simplify:false u);
          equal string (cell "simplified") (Render.render u));
      Golden.cases "trees_rendered.golden" simplified_tree;
      prop "writes the simplified node, or raises as simplifying does"
        expression (fun u ->
          match Shape.simplify u with
          | s ->
              cover "renders" true;
              equal string (Render.render ~simplify:false s) (Render.render u)
          | exception e ->
              cover "raises" true;
              raises e (fun () -> Render.render u));
    ]

(* srender *)

let ints =
  Gen.frequency
    [
      (6, Gen.int_range (-1000) 1000);
      (1, Gen.int);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ min_int; min_int + 1; -1; 0; 1; max_int - 1; max_int ] );
    ]

let srender =
  group "srender"
    [
      prop "writes an integer in decimal" ints (fun n ->
          equal string (Int.to_string n) (Render.srender (Int n)));
      test "writes a node as render does" (fun () ->
          equal string "a" (Render.srender (Sym a));
          equal string "(a*12)"
            (Render.srender Shape.Sint.(prod [ Sym a; Int 3; Int 4 ]));
          equal string "(a*12)"
            (Render.srender Shape.Sint.(prod [ Int 3; Int 4; Sym a ])));
    ]

(* pp_uops *)

let listing ?(color = false) uops =
  Setting.context
    [ B (Setting.no_color, not color) ]
    (fun () -> Format.asprintf "%a" Render.pp_uops uops)

(* tinygrad prints each line ended; a formatter leaves the last line open. *)
let printed ?color uops =
  match uops with [] -> "" | _ -> listing ?color uops ^ "\n"

let listings =
  [
    ("program_listing.golden", "program.golden", false);
    ("program_listing_colored.golden", "program.golden", true);
    ("partial_listing.golden", "partial.golden", false);
    ("wide_listing.golden", "wide.golden", false);
    ("wide_listing_colored.golden", "wide.golden", true);
  ]

let positions = Gen.list ~size:(Gen.int_range 0 10) (Gen.int_range 0 22)

let one_line_per_node positions =
  let program = sources "program.golden" in
  let lines =
    String.split_on_char '\n' (listing (List.map (List.nth program) positions))
  in
  equal int (max 1 (List.length positions)) (List.length lines);
  List.iteri
    (fun i line ->
      if positions <> [] then
        starts_with ~msg:line ~affix:(Printf.sprintf "%4d " i) line)
    lines

let pp_uops =
  group "pp_uops"
    (List.map
       (fun (golden, input, color) ->
         Golden.text golden (fun () -> printed ~color (sources input)))
       listings
    @ [
        test "an empty list prints nothing" (fun () ->
            equal string "" (listing []));
        prop "prints a line per node, numbered from 0, the last not ended"
          positions one_line_per_node;
      ])

let () = exit (run "Tolk.Render" [ render; simplified; srender; pp_uops ])
