(* Patterns, pattern matchers and graph rewriting. *)

open Windtrap
open Tolk
open Common
module P = Ops.Upat
module Pm = Ops.Pattern_matcher

let str pp x = Format.asprintf "%a" pp x
let matches p u = P.match_ p u <> []
let sunk u = Some (Ops.sink [ u ])

(* [pm_sink p] wraps the node [p] names ["x"] in a sink. *)
let pm_sink p = Pm.v (fun () -> [ Pm.rule p (fun m -> sunk (m "x")) ])
let rewrite m u = Pm.rewrite m () u

let c1 = Ops.float 1.
and c2 = Ops.float 2.
and c3 = Ops.float 3.

let add x y = Ops.v ~src:[ x; y ] Op.Add

(* [binds name u naming] is that [naming] names [u] [name]. *)
let binds name u naming =
  match List.assoc_opt name naming with Some v -> v == u | None -> false

let names_both x y namings =
  List.exists (fun n -> binds "x" x n && binds "y" y n) namings

let upat =
  group "Upat"
    [
      test "matches a constant of a type" (fun () ->
          let m = pm_sink (P.op ~name:"x" ~dtype:[ Weak_float ] Op.Const) in
          equal (option uop) (sunk (Ops.float 1.)) (rewrite m (Ops.float 1.));
          equal (option uop) None (rewrite m (Ops.int 1)));
      test "matches an operation, and no other" (fun () ->
          let m = pm_sink (P.op ~name:"x" Op.Const) in
          equal (option uop) (sunk c1) (rewrite m c1);
          equal (option uop) None (rewrite m (add c1 c1)));
      test "matches any operation of a set" (fun () ->
          let m =
            pm_sink (P.v ~op:(Op.Set.of_list [ Const; Cast ]) ~name:"x" ())
          in
          let f = Ops.bool false in
          let cast = Ops.v ~src:[ f ] ~arg:(Dtype Int32) Op.Cast in
          equal (option uop) (sunk f) (rewrite m f);
          equal (option uop) (sunk cast) (rewrite m cast);
          equal (option uop) None (rewrite m (add c3 c3)));
      test "matches an argument as a number: 0, 0.0 and false are equal"
        (fun () ->
          let m =
            Pm.v (fun () ->
                [
                  Pm.rule
                    (P.op ~arg:(Const (i 0)) ~name:"x" Op.Const)
                    (fun m -> sunk (m "x"));
                  Pm.rule (P.op ~name:"x" Op.Max) (fun m -> sunk (m "x"));
                ])
          in
          let zero = Ops.float 0. in
          equal (option uop) (sunk zero) (rewrite m zero);
          equal (option uop)
            (sunk (Ops.float (-0.)))
            (rewrite m (Ops.float (-0.)));
          equal (option uop)
            (sunk (Ops.bool false))
            (rewrite m (Ops.bool false));
          let max = Ops.v ~src:[ zero; zero ] Op.Max in
          equal (option uop) (sunk max) (rewrite m max);
          equal (option uop) None (rewrite m (Ops.v ~src:[ zero; zero ] Op.Mul));
          equal (option uop) None (rewrite m (Ops.int (-1))));
      test "matches integers and floats exactly" (fun () ->
          let exact = `Int (Bigint.of_string "9007199254740992")
          and next = `Int (Bigint.of_string "9007199254740993") in
          let rounded = Ops.float 9007199254740992. in
          is_true (matches (P.const exact) rounded);
          is_false (matches (P.const next) rounded);
          is_false (matches (P.const (f 9007199254740992.)) (Ops.const next));
          is_true (matches (P.const (i 1)) (Ops.bool true));
          is_false (matches (P.const (i 1)) (Ops.float 1.5));
          is_false (matches (P.const exact) (Ops.float Float.infinity)));
      test "a NaN argument matches no other constant" (fun () ->
          is_false
            (matches (P.op ~arg:(Const (f Float.nan)) Op.Const) (Ops.float 1.));
          is_false
            (matches (P.op ~arg:(Const (f 1.)) Op.Const) (Ops.float Float.nan)));
      test "patterns that differ only in their argument permute" (fun () ->
          let p = P.op ~perm:[ P.const (i 0); P.const (i 1) ] Op.Add in
          is_true (matches p (Ops.v ~src:[ Ops.int 1; Ops.int 0 ] Op.Add)));
      test "a NaN argument matches a NaN constant" (fun () ->
          let m =
            Pm.fold (fun () ->
                [
                  Pm.rule
                    (P.op ~arg:(Const (f Float.nan)) Op.Const)
                    (fun _ -> Some true);
                ])
          in
          equal (option bool) (Some true)
            (Pm.rewrite m () (Ops.float Float.nan)));
      test "the sources of a rule filter it" (fun () ->
          let m =
            Pm.v (fun () ->
                [
                  Pm.rule
                    (P.op ~name:"x"
                       ~perm:
                         [
                           P.op ~name:"c" Op.Const;
                           P.op ~arg:(Const (i 2)) Op.Const;
                         ]
                       Op.Mul)
                    (fun m ->
                      match Ops.value (m "c") with
                      | `Int n when Bigint.equal (Bigint.abs n) Bigint.one ->
                          sunk (m "x")
                      | _ -> None);
                ])
          in
          let mul a b = Ops.v ~src:[ Ops.int a; Ops.int b ] Op.Mul in
          List.iter
            (fun (a, b, fires) ->
              equal
                ~msg:(Printf.sprintf "%d * %d" a b)
                (option uop)
                (if fires then sunk (mul a b) else None)
                (rewrite m (mul a b)))
            [
              (1, 2, true);
              (2, 2, false);
              (-1, 2, true);
              (2, 1, true);
              (2, -1, true);
            ]);
      test "a name used twice matches one node" (fun () ->
          let p =
            P.v ~op:Op.Set.alu ~name:"x"
              ~src:[ P.op ~name:"y" Op.Const; P.op ~name:"y" Op.Const ]
              ()
          in
          is_true (matches p (add c1 c1));
          is_true (matches p (add c1 (Ops.float 1.)));
          is_false (matches p (add c1 c2)));
      test "matches a type among several" (fun () ->
          let m =
            pm_sink (P.op ~name:"x" ~dtype:[ Float32; Float64 ] Op.Cast)
          in
          let cast dt = Ops.cast (Ops.float 1.) dt in
          equal (option uop) (sunk (cast Float32)) (rewrite m (cast Float32));
          equal (option uop) (sunk (cast Float64)) (rewrite m (cast Float64));
          equal (option uop) None (rewrite m (cast Float16));
          equal (option uop) None (rewrite m (Ops.int ~dtype:Int32 1)));
      test "src matches the sources in order, and exactly as many" (fun () ->
          let m =
            pm_sink
              (P.v ~op:Op.Set.alu ~name:"x"
                 ~src:[ P.op Op.Const; P.op Op.Const ]
                 ())
          in
          let s = add c1 c2 in
          equal (option uop) (sunk s) (rewrite m s);
          equal (option uop) None (rewrite m c2);
          equal (option uop) None
            (rewrite m (Ops.v ~src:[ c1; c2; c3 ] Op.Mulacc)));
      test "perm matches the sources in any order" (fun () ->
          let m =
            pm_sink
              (P.v ~op:Op.Set.alu ~name:"x"
                 ~perm:[ P.op Op.Const; P.v ~op:Op.Set.alu () ]
                 ())
          in
          let s3 = add c1 c2 in
          let s4 = add s3 c2 and s5 = add c2 s3 in
          equal (option uop) None (rewrite m s3);
          equal (option uop) (sunk s4) (rewrite m s4);
          equal (option uop) (sunk s5) (rewrite m s5);
          equal (option uop) None (rewrite m (add s3 s4)));
      test "each matches every source" (fun () ->
          let m =
            pm_sink (P.v ~op:Op.Set.alu ~name:"x" ~each:(P.op Op.Const) ())
          in
          let s = add c1 c2 in
          equal (option uop) (sunk s) (rewrite m s);
          equal (option uop) None (rewrite m (add c2 s)));
      test "allow_any_len takes more sources, never fewer" (fun () ->
          let m =
            pm_sink
              (P.op ~name:"x"
                 ~src:[ P.op Op.Const ]
                 ~allow_any_len:true Op.Mulacc)
          in
          let s = Ops.v ~src:[ c1; c2; c3 ] Op.Mulacc in
          equal (option uop) None (rewrite m (Ops.v ~src:[ c1 ] Op.Exp2));
          equal (option uop) None (rewrite m (add c1 c2));
          equal (option uop) (sunk s) (rewrite m s);
          let noop = Ops.v Op.Noop in
          let p =
            Pm.fold (fun () ->
                [
                  Pm.rule
                    (P.op
                       ~src:[ P.op Op.Noop; P.op Op.Noop ]
                       ~allow_any_len:true Op.Noop)
                    (fun _ -> Some true);
                ])
          in
          equal (option bool) (Some true)
            (Pm.rewrite p () (Ops.v ~src:[ noop; noop; noop ] Op.Noop));
          equal (option bool) (Some true)
            (Pm.rewrite p () (Ops.v ~src:[ noop; noop ] Op.Noop));
          equal (option bool) None
            (Pm.rewrite p () (Ops.v ~src:[ noop ] Op.Noop)));
      test "permutations nest" (fun () ->
          let p =
            P.v ~op:Op.Set.alu
              ~perm:
                [
                  P.v ~op:Op.Set.alu ~perm:[ P.var "a"; P.var "b" ] ();
                  P.var "b";
                ]
              ()
          in
          is_true (matches p (add (add c1 c2) c1));
          is_true (matches p (add (add c2 c1) c1)));
      test "any matches through any of its alternatives" (fun () ->
          let v1 = weak_var "a" 0 10 and v2 = weak_var "b" 0 10 in
          let m =
            Pm.v (fun () ->
                [
                  Pm.rule
                    P.(P.var "a" + P.any [ P.var "x"; P.var "y"; P.var "z" ])
                    (fun m ->
                      match m "y" with
                      | y -> sunk Ops.O.(m "a" + y)
                      | exception Invalid_argument _ -> None);
                ])
          in
          equal (option uop)
            (sunk Ops.O.(v1 + v2))
            (rewrite m Ops.O.(v1 + v2)));
      test "match_ names the nodes of each way a pattern matches" (fun () ->
          let a = var "a" 0 4 and b = var "b" 0 4 in
          let namings = P.match_ P.(P.var "x" + P.var "y") Ops.O.(a + b) in
          is_true (names_both a b namings);
          is_true (names_both b a namings);
          equal
            (list (list (pair string uop)))
            []
            (P.match_ (P.op Op.Mul) Ops.O.(a + b)));
      test "v rejects two ways of matching sources" (fun () ->
          rejects (fun () -> P.v ~src:[ P.wild ] ~perm:[ P.wild ] ());
          rejects (fun () -> P.v ~src:[ P.wild ] ~each:P.wild ()));
      test "the constructors match what the node constructors build" (fun () ->
          let a = var "a" 0 4 and x = fvar "x" in
          let p = Shape.param ~shape:(ints [ 4 ]) 0 Float32 in
          let idx = Ops.index p [ a ] in
          let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
          List.iter
            (fun (name, pat, u) -> is_true ~msg:name (matches pat u))
            [
              ("wild", P.wild, a);
              ("var", P.var ~dtype:[ Int32 ] "x", a);
              ("cvar", P.cvar ~arg:(i 1) "c", Ops.int 1);
              ("const", P.const (i 3), Ops.int 3);
              ("named", P.named "n" (P.op Op.Param), a);
              ("or_casted value", P.or_casted (P.op Op.Param), a);
              ("or_casted cast", P.or_casted (P.op Op.Param), Ops.cast a Float32);
              ( "or_bitcasted",
                P.or_bitcasted (P.op Op.Param),
                Ops.bitcast a Uint32 );
              ( "or_after",
                P.or_after (P.op Op.Param),
                Ops.after a [ Ops.sink []; Ops.sink [ a ] ] );
              ("f", P.f (P.op Op.Param) Op.Cast, Ops.cast a Float32);
              ( "or_casted named",
                P.or_casted ~name:"c" (P.op Op.Param),
                Ops.cast a Float32 );
              ("or_bitcasted named", P.or_bitcasted ~name:"c" (P.op Op.Param), a);
              ( "or_after named",
                P.or_after ~name:"c" (P.op Op.Param),
                Ops.after a [ Ops.sink [] ] );
              ("sink", P.sink [ P.wild; P.wild ], Ops.sink [ a; x ]);
              ("index", P.index (P.op Op.Param) [ P.var "i" ], idx);
              ("load", P.load (P.op Op.Index) [], Ops.load idx []);
              ( "store",
                P.store (P.op Op.Index) [ P.wild ],
                Ops.store idx (Ops.float ~dtype:Float32 1.) );
              ( "reduce",
                P.reduce ~op:Op.Add P.wild [ P.op Op.Range ],
                Ops.reduce x Op.Add [ r ] );
              ( "broadcast",
                P.broadcast (P.const (i 1)),
                Ops.broadcast (Ops.int 1) 3 );
              ( "after",
                P.after (P.op Op.Param) [ P.wild ],
                Ops.after a [ Ops.sink [] ] );
              ( "end_",
                P.end_ P.wild [ P.op Op.Range ],
                Ops.end_ (Ops.sink []) [ r ] );
              ( "backedge",
                P.backedge P.wild ~loop:(P.op Op.Range) ~cond:P.wild,
                Ops.backedge (Ops.int 1) ~loop:(Ops.loop 1) ~cond:(flag "p") );
              ("bitcast", P.bitcast ~dtype:Uint32 P.wild, Ops.bitcast a Uint32);
            ]);
      test "a reduce pattern names its operation in the argument" (fun () ->
          let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
          is_false
            (matches
               (P.reduce ~op:Op.Mul P.wild [ P.wild ])
               (Ops.reduce (fvar "x") Op.Add [ r ])));
      test "dtype is the first type a pattern requires" (fun () ->
          equal dtype Float16 (P.dtype (P.var ~dtype:[ Float16; Float32 ] "x"));
          equal dtype Void (P.dtype P.wild));
      test "cast to the type a pattern requires is the pattern" (fun () ->
          let x = fvar "x" in
          is_true (matches (P.cast (P.var ~dtype:[ Float32 ] "x") Float32) x);
          is_false (matches (P.cast (P.var "x") Float32) x));
    ]

(* Elementwise patterns: each builds the pattern of the node its operation
   builds, and names the operands. *)
let alu_pairs =
  Ops.
    [
      ("add", P.add, add, Dtype.Int32);
      ("sub", P.sub, sub, Int32);
      ("mul", P.mul, mul, Int32);
      ("floor division", P.div ~rounding:`Floor, div ~rounding:`Floor, Int32);
      ("truncated division", P.div ~rounding:`Trunc, div ~rounding:`Trunc, Int32);
      ( "true division of floats",
        P.div ?rounding:None,
        div ?rounding:None,
        Float32 );
      ( "floor division of floats",
        P.div ~rounding:`Floor,
        div ~rounding:`Floor,
        Float32 );
      ("mod_", P.mod_, mod_, Int32);
      ("fmod", P.fmod, fmod, Int32);
      ("fmod of floats", P.fmod, fmod, Float32);
      ("pow", P.pow, pow, Float32);
      ("lt", P.lt, lt, Int32);
      ("gt", P.gt, gt, Int32);
      ("le", P.le, le, Int32);
      ("ge", P.ge, ge, Int32);
      ("ne", P.ne, ne, Int32);
      ("eq", P.eq, eq, Int32);
      ("bitwise_and", P.bitwise_and, bitwise_and, Int32);
      ("bitwise_or", P.bitwise_or, bitwise_or, Int32);
      ("bitwise_xor", P.bitwise_xor, bitwise_xor, Int32);
      ("shl", P.shl, shl, Int32);
      ("shr", P.shr, shr, Int32);
      ("maximum", P.maximum, maximum, Int32);
      ("minimum", P.minimum, minimum, Int32);
      ("minimum of floats", P.minimum, minimum, Float32);
    ]

let alu_unary =
  Ops.
    [
      ("neg", P.neg, neg, Dtype.Int32);
      ("neg of a boolean", P.neg, neg, Bool);
      ("logical_not", P.logical_not, logical_not, Int32);
      ("bitwise_not", P.bitwise_not, bitwise_not, Int32);
      ("bitwise_not of an unsigned", P.bitwise_not, bitwise_not, Uint8);
      ("reciprocal", P.reciprocal, reciprocal, Float32);
      ("trunc", P.trunc, trunc, Float32);
      ("floor", P.floor, floor, Float32);
      ("sqrt", P.sqrt, sqrt, Float32);
      ("exp2", P.exp2, exp2, Float32);
      ("log2", P.log2, log2, Float32);
    ]

let operand dt name =
  if Dtype.is_float dt then fvar ~dtype:dt name
  else if Dtype.is_bool dt then flag name
  else var ~dtype:dt name 0 10

let elementwise_patterns =
  group "elementwise patterns"
    [
      cases "a binary pattern matches the node its operation builds"
        ~id:(fun (n, _, _, _) -> n)
        alu_pairs
        (fun (_, pattern, node, dt) ->
          let x = operand dt "x" and y = operand dt "y" in
          is_true
            (names_both x y
               (P.match_
                  (pattern (P.var ~dtype:[ dt ] "x") (P.var ~dtype:[ dt ] "y"))
                  (node x y))));
      cases "a unary pattern matches the node its operation builds"
        ~id:(fun (n, _, _, _) -> n)
        alu_unary
        (fun (_, pattern, node, dt) ->
          let x = operand dt "x" in
          is_true
            (List.exists (binds "x" x)
               (P.match_ (pattern (P.var ~dtype:[ dt ] "x")) (node x))));
      test "a where pattern matches a selection" (fun () ->
          let c = flag "c" and x = var "x" 0 10 and y = var "y" 0 10 in
          is_true
            (names_both x y
               (P.match_
                  (P.where (P.var "c") (P.var "x") (P.var "y"))
                  (Ops.where c x y))));
      test "a commutative operation matches its operands in either order"
        (fun () ->
          let x = var "x" 0 10 in
          is_true (matches P.(P.var "x" + P.cvar "c") Ops.O.(int 2 + x));
          is_true (matches P.(P.var "x" * P.cvar "c") Ops.O.(x * int 2));
          is_false (matches P.(P.var "x" < P.cvar "c") Ops.O.(int 2 < x)));
      test
        "a literal matches a constant of any type, never a typed constant's \
         cast" (fun () ->
          let x = fvar "x" in
          is_true (matches P.(P.var "x" * int 2) Ops.O.(x * float 2.));
          is_true
            (matches
               P.(P.var "x" + float 0.)
               (Ops.add (var "a" 0 1) (Ops.int 0)));
          is_false
            (matches
               P.(P.var "x" * int 2)
               (Ops.v ~src:[ x; Ops.float ~dtype:Float32 2. ] Op.Mul)));
      test "floor division and remainder operators match their operations"
        (fun () ->
          let a = var "a" 0 10 and b = var "b" 1 10 in
          is_true
            (matches P.(P.var "x" // P.var "y") (Ops.alu a Op.Div [ b ]));
          is_true
            (matches P.(P.var "x" % P.var "y") (Ops.alu a Op.Mod [ b ]));
          is_false
            (matches P.(P.var "x" // P.var "y") (Ops.alu a Op.Truncdiv [ b ])));
      test "the pattern operators are the named pattern operations" (fun () ->
          let a = var "a" 0 10 and b = var "b" 1 10 in
          let both p q u = equal bool (matches p u) (matches q u) in
          List.iter
            (fun u ->
              both P.(P.var "x" + P.var "y") (P.add (P.var "x") (P.var "y")) u;
              both P.(P.var "x" - P.var "y") (P.sub (P.var "x") (P.var "y")) u;
              both P.(P.var "x" < P.var "y") (P.lt (P.var "x") (P.var "y")) u;
              both
                P.(P.var "x" land P.var "y")
                (P.bitwise_and (P.var "x") (P.var "y"))
                u)
            Ops.O.[ a + b; a - b; a < b; a land b; a * b ]);
    ]

(* Pattern matchers *)

let matchers =
  let a = var "a" 0 10 in
  group "Pattern_matcher"
    [
      test "v's first rewrite rejects a rule whose pattern has no operation"
        (fun () ->
          let m = Pm.v (fun () -> [ Pm.rule P.wild (fun _ -> Some a) ]) in
          rejects (fun () -> Pm.rewrite m () a));
      test "the first rule that matches and does not decline wins" (fun () ->
          let m =
            Pm.v (fun () ->
                [
                  Pm.rule (P.op Op.Param) (fun _ -> None);
                  Pm.rule (P.op ~name:"x" Op.Param) (fun m -> Some (m "x"));
                  Pm.rule (P.op Op.Param) (fun _ -> Some (Ops.int 1));
                  Pm.rule (P.op Op.Param) (fun _ -> Some (Ops.int 2));
                ])
          in
          equal (option uop) (Some (Ops.int 1)) (rewrite m a));
      test "a rule is given each naming in turn until one gives a result"
        (fun () ->
          let seen = ref [] in
          let m =
            Pm.v (fun () ->
                [
                  Pm.rule
                    P.(P.var "x" + P.cvar "c")
                    (fun m ->
                      seen := (m "x", m "c") :: !seen;
                      None);
                  Pm.rule
                    P.(P.var "x" + P.cvar "c")
                    (fun m -> if m "c" == c2 then Some (m "x") else None);
                ])
          in
          equal (option uop) (Some c3) (rewrite m (add c2 c3));
          equal (list (pair uop uop)) [ (c2, c3); (c3, c2) ] (List.rev !seen));
      test "a part of a pattern that names nothing gives one naming"
        (fun () ->
          let p = P.(var "x" + (const (`Float 1.) * wild)) in
          let u = add a Ops.O.(c1 * c1) in
          equal int 1 (List.length (P.match_ p u));
          let calls = ref 0 in
          let m =
            Pm.v (fun () ->
                [
                  Pm.rule
                    (P.sink
                       [
                         P.broadcast P.(var "x" + var "y");
                         P.(const (`Float 1.) * wild);
                       ])
                    (fun _ ->
                      incr calls;
                      None);
                ])
          in
          let b = var "b" 0 10 in
          equal (option uop) None
            (rewrite m
               (Ops.sink [ Ops.v ~src:[ add a b ] Op.Stack; Ops.O.(c1 * c1) ]));
          equal int 2 !calls);
      test "a fold's rule declines by returning nothing" (fun () ->
          let m =
            Pm.fold (fun () ->
                [
                  Pm.rule (P.op Op.Param) (fun _ -> None);
                  Pm.rule (P.op Op.Param) (fun _ -> Some "second");
                ])
          in
          equal (option string) (Some "second") (Pm.rewrite m () a));
      test "a fold's rule that returns its node is a result" (fun () ->
          let m =
            Pm.fold (fun () ->
                [
                  Pm.rule (P.op ~name:"x" Op.Param) (fun m -> Some (m "x"));
                  Pm.rule (P.op Op.Param) (fun _ -> Some (Ops.int 1));
                ])
          in
          equal (option uop) (Some a) (Pm.rewrite m () a));
      test "concat tries the matchers' rules in order" (fun () ->
          let one =
            Pm.v (fun () ->
                [ Pm.rule (P.op Op.Param) (fun _ -> Some (Ops.int 1)) ])
          in
          let two =
            Pm.v (fun () ->
                [ Pm.rule (P.op Op.Param) (fun _ -> Some (Ops.int 2)) ])
          in
          let none =
            Pm.v (fun () -> [ Pm.rule (P.op Op.Param) (fun _ -> None) ])
          in
          equal (option uop)
            (Some (Ops.int 2))
            (rewrite (Pm.concat [ none; two; one ]) a);
          equal (option uop) None (rewrite (Pm.concat []) a));
      test "a naming the pattern does not bind is rejected" (fun () ->
          let m =
            Pm.v (fun () ->
                [ Pm.rule (P.op ~name:"x" Op.Param) (fun m -> Some (m "y")) ])
          in
          rejects (fun () -> rewrite m a));
      test "rule_ctx reads the context" (fun () ->
          let ctx = ref 0 in
          let m =
            Pm.v (fun () ->
                [
                  Pm.rule_ctx (P.op ~src:[] ~name:"x" Op.Noop) (fun ctx m ->
                      incr ctx;
                      Some (Ops.replace ~src:[ Ops.v Op.Noop ] (m "x")));
                ])
          in
          let once = Option.get (Pm.rewrite m ctx (Ops.v Op.Noop)) in
          equal (option uop) None (Pm.rewrite m ctx once);
          equal int 1 !ctx);
      test "append tries the first matcher's rules first" (fun () ->
          let one =
            Pm.v (fun () ->
                [ Pm.rule (P.op Op.Param) (fun _ -> Some (Ops.int 1)) ])
          in
          let two =
            Pm.v (fun () ->
                [ Pm.rule (P.op Op.Param) (fun _ -> Some (Ops.int 2)) ])
          in
          equal (option uop) (Some (Ops.int 1)) (rewrite (Pm.append one two) a);
          equal (option uop) (Some (Ops.int 2)) (rewrite (Pm.append two one) a));
      test "with_ctx joins a matcher without context to one with" (fun () ->
          let plain =
            Pm.v (fun () ->
                [ Pm.rule (P.op Op.Const) (fun _ -> Some (Ops.int 1)) ])
          in
          let reading =
            Pm.v (fun () ->
                [ Pm.rule_ctx (P.op Op.Param) (fun ctx _ -> Some ctx) ])
          in
          let m = Pm.append reading (Pm.with_ctx plain) in
          equal (option uop) (Some (Ops.int 7)) (Pm.rewrite m (Ops.int 7) a);
          equal (option uop)
            (Some (Ops.int 1))
            (Pm.rewrite m (Ops.int 7) (Ops.int 3)));
      test "early_reject skips a rule unless the sources hold its operations"
        (fun () ->
          let fired = ref 0 in
          let m =
            Pm.v (fun () ->
                [
                  Pm.rule (P.op ~early_reject:[ Op.Mul ] Op.Add) (fun _ ->
                      incr fired;
                      None);
                ])
          in
          equal (option uop) None (rewrite m Ops.O.(a + int 1));
          equal int 0 !fired;
          ignore (rewrite m (Ops.v ~src:[ Ops.O.(a * int 2); a ] Op.Add));
          equal int 1 !fired);
    ]

(* A reference matcher

   Patterns drawn as data, built with Upat's constructors, and matched by a
   matcher that tries every way in turn: each alternative of an [any], each
   ordering of a permutation, each naming of a source before the next. A part
   that names nothing gives one naming. *)

type pat = Node of node_pat | Any of pat list

and node_pat = {
  ops : Op.t list option;
  dtypes : Dtype.t list option;
  arg : int option;
  name : string option;
  srcs : srcs;
  any_len : bool;
}

and srcs = Leaf | Fixed of pat list | Perm of pat list | Each of pat

let const_arg n = Ops.Const (i n)

let rec built = function
  | Any ps -> P.any (List.map built ps)
  | Node n ->
      let src, perm, each =
        match n.srcs with
        | Leaf -> (None, None, None)
        | Fixed ps -> (Some (List.map built ps), None, None)
        | Perm ps -> (None, Some (List.map built ps), None)
        | Each p -> (None, None, Some (built p))
      in
      P.v
        ?op:(Option.map Op.Set.of_list n.ops)
        ?dtype:n.dtypes ?src ?perm ?each ~allow_any_len:n.any_len
        ?arg:(Option.map const_arg n.arg)
        ?name:n.name ()

let rec pp_pat ppf = function
  | Any ps -> Format.fprintf ppf "any [%a]" pp_pats ps
  | Node n -> (
      let pp_opt pp ppf = function
        | None -> Format.pp_print_string ppf "_"
        | Some x -> pp ppf x
      in
      let pp_list pp ppf l =
        Format.pp_print_list
          ~pp_sep:(fun ppf () -> Format.fprintf ppf "|")
          pp ppf l
      in
      Format.fprintf ppf "%a:%a:%a:%a%s"
        (pp_opt (pp_list Op.pp))
        n.ops
        (pp_opt (pp_list Dtype.pp))
        n.dtypes
        (pp_opt Format.pp_print_int)
        n.arg
        (pp_opt Format.pp_print_string)
        n.name
        (if n.any_len then "..." else "");
      match n.srcs with
      | Leaf -> ()
      | Fixed ps -> Format.fprintf ppf " (%a)" pp_pats ps
      | Perm ps -> Format.fprintf ppf " {%a}" pp_pats ps
      | Each p -> Format.fprintf ppf " each (%a)" pp_pat p)

and pp_pats ppf ps =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
    pp_pat ppf ps

let rec names = function
  | Any ps -> List.concat_map names ps
  | Node n -> (
      Option.to_list n.name
      @
      match n.srcs with
      | Leaf -> []
      | Fixed ps | Perm ps -> List.concat_map names ps
      | Each p -> names p)

let distinct_names p = List.sort_uniq String.compare (names p)

let rec permutations = function
  | [] -> [ [] ]
  | l ->
      List.concat
        (List.mapi
           (fun k x ->
             List.map
               (fun rest -> x :: rest)
               (permutations (List.filteri (fun j _ -> j <> k) l)))
           l)

let fits n u =
  (match n.ops with None -> true | Some os -> List.mem (Ops.op u) os)
  && (match n.dtypes with
    | None -> true
    | Some dts -> List.exists (Dtype.equal (Ops.dtype u)) dts)
  && (match n.arg with
    | None -> true
    | Some c -> Ops.equal_arg (const_arg c) (Ops.arg u))
  &&
  let len = List.length (Ops.src u) in
  match n.srcs with
  | Leaf | Each _ -> true
  | Fixed ps | Perm ps ->
      if n.any_len then len >= List.length ps else len = List.length ps

(* [store], in the order of binding, with [u] named [name]. *)
let bind name u store =
  match name with
  | None -> Some store
  | Some n -> (
      match List.assoc_opt n store with
      | Some v -> if v == u then Some store else None
      | None -> Some (store @ [ (n, u) ]))

let rec namings p u store =
  if names p = [] then if ways p u store = [] then [] else [ store ]
  else ways p u store

and ways p u store =
  match p with
  | Any ps -> List.concat_map (fun p -> namings p u store) ps
  | Node n -> (
      if not (fits n u) then []
      else
        match bind n.name u store with
        | None -> []
        | Some store -> (
            let srcs = Ops.src u in
            match n.srcs with
            | Leaf -> [ store ]
            | Fixed ps -> in_order ps srcs [ store ]
            | Perm ps ->
                List.concat_map
                  (fun ps -> in_order ps srcs [ store ])
                  (permutations ps)
            | Each p ->
                List.fold_left
                  (fun stores s ->
                    List.concat_map (fun st -> namings p s st) stores)
                  [ store ] srcs))

and in_order ps srcs stores =
  match (ps, srcs) with
  | p :: ps, s :: srcs ->
      in_order ps srcs (List.concat_map (fun st -> namings p s st) stores)
  | _ -> stores

(* Whether a rule of [p] is tried on [u]: [u] is of an operation of [p], and has
   a source of each operation a source pattern requires alone. *)
let tried p u =
  match p with
  | Any _ -> false
  | Node n ->
      let required = function
        | Node { ops = Some [ o ]; _ } -> [ o ]
        | Node _ | Any _ -> []
      in
      let firsts =
        match n.srcs with
        | Leaf -> []
        | Fixed ps | Perm ps -> ps
        | Each p -> [ p ]
      in
      (match n.ops with Some os -> List.mem (Ops.op u) os | None -> false)
      && List.for_all
           (fun o -> List.exists (fun s -> Ops.op s = o) (Ops.src u))
           (List.concat_map required firsts)

(* Generators *)

let leaves =
  let a = var "a" 0 10 and b = var "b" 0 10 in
  [ a; b; Ops.int ~dtype:Int32 0; Ops.int ~dtype:Int32 1 ]

let rec node_gen depth =
  let open Gen in
  let leaf = of_list ~pp:Ops.pp leaves in
  if depth = 0 then leaf
  else
    let sub = node_gen (depth - 1) in
    let binary o =
      let+ x = sub and+ y = sub in
      Ops.v ~src:[ x; y ] o
    in
    frequency
      [
        (2, leaf);
        (2, binary Op.Add);
        (2, binary Op.Mul);
        ( 1,
          let+ x = sub in
          Ops.v ~src:[ x ] ~arg:(Dtype Float32) Op.Cast );
        ( 1,
          let+ xs = list ~size:(int_range 1 3) sub in
          Ops.v ~src:xs Op.Stack );
      ]

let ops_gen ~root =
  let var_op = Ops.op (List.hd leaves) in
  Gen.of_list
    ((if root then [] else [ None; None ])
    @ [
        Some [ Op.Add ];
        Some [ Op.Mul ];
        Some [ Op.Add; Op.Mul ];
        Some [ Op.Const ];
        Some [ Op.Cast; Op.Stack ];
        Some [ var_op ];
      ])

let rec pat_gen ~root depth =
  let open Gen in
  let node srcs =
    let+ ops = ops_gen ~root
    and+ dtypes =
      of_list
        [ None; None; None; Some [ Dtype.Int32 ]; Some [ Float32; Int32 ] ]
    and+ arg = of_list [ None; None; None; None; Some 0; Some 1 ]
    and+ name = of_list [ None; None; Some "x"; Some "y" ]
    and+ srcs = srcs
    and+ any_len = of_list [ false; false; true ] in
    Node { ops; dtypes; arg; name; srcs; any_len }
  in
  if depth = 0 then node (constant Leaf)
  else
    let sub = pat_gen ~root:false (depth - 1) in
    let srcs =
      frequency
        [
          (1, constant Leaf);
          (2, map (fun ps -> Fixed ps) (list ~size:(int_range 0 3) sub));
          (2, map (fun ps -> Perm ps) (list ~size:(int_range 2 3) sub));
          (1, map (fun p -> Each p) sub);
        ]
    in
    let any =
      let+ ps = list ~size:(int_range 1 3) sub in
      Any ps
    in
    if root then node srcs else frequency [ (4, node srcs); (1, any) ]

let rec all = function
  | [] -> Gen.constant []
  | g :: gs ->
      let open Gen in
      let+ x = g and+ xs = all gs in
      x :: xs

(* A pattern drawn from [u]'s structure, which matches [u] unless one of its
   names is given two nodes, or it ends in a part drawn regardless of [u]. *)
let rec derived ~root depth u =
  let open Gen in
  let srcs = Ops.src u in
  let n = List.length srcs in
  let derived_srcs = all (List.map (derived ~root:false (depth - 1)) srcs) in
  let fixed =
    let+ ps = derived_srcs and+ dropped = int_range 0 (min n 1) in
    (Fixed (List.filteri (fun k _ -> k < n - dropped) ps), dropped > 0)
  in
  let srcs =
    if depth = 0 then constant (Leaf, false)
    else
      frequency
        ([ (1, constant (Leaf, false)); (2, fixed) ]
        @ (if n >= 2 then [ (3, map (fun ps -> (Perm ps, false)) derived_srcs) ]
           else [])
        @
        match srcs with
        | s :: _ ->
            [
              ( 2,
                map
                  (fun p -> (Each p, false))
                  (derived ~root:false (depth - 1) s) );
            ]
        | [] -> [])
  in
  let consts =
    List.filter (fun k -> Ops.equal_arg (const_arg k) (Ops.arg u)) [ 0; 1 ]
  in
  let node =
    let+ ops =
      of_list
        ((if root then [] else [ None; None ])
        @ [ Some [ Ops.op u ]; Some [ Op.Mul; Ops.op u ] ])
    and+ dtypes =
      of_list [ None; Some [ Ops.dtype u ]; Some [ Float32; Ops.dtype u ] ]
    and+ arg = of_list (None :: List.map Option.some consts)
    and+ name = of_list [ None; None; None; Some "x"; Some "y"; Some "z" ]
    and+ srcs, any_len = srcs in
    Node { ops; dtypes; arg; name; srcs; any_len }
  in
  if root then node
  else
    frequency
      [
        (6, node);
        ( 1,
          let+ p = node and+ q = pat_gen ~root:false 1 in
          Any [ q; p ] );
        (1, pat_gen ~root:false 1);
      ]

let case_gen =
  let open Gen in
  let random =
    let* depth = int_range 0 2 in
    pair (pat_gen ~root:true depth) (node_gen 2)
  in
  let matching =
    let* u = node_gen 2 and+ depth = int_range 1 3 in
    let+ p = derived ~root:true depth u in
    (p, u)
  in
  with_pp
    (fun ppf (p, u) -> Format.fprintf ppf "%a@ on@ %a" pp_pat p Ops.pp u)
    (frequency [ (1, random); (3, matching) ])

let rec repeats_names_in_ways = function
  | Any ps -> List.exists repeats_names_in_ways ps
  | Node n -> (
      match n.srcs with
      | Leaf -> false
      | Fixed ps | Perm ps -> List.exists repeats_names_in_ways ps
      | Each q -> (names q <> [] && several_ways q) || repeats_names_in_ways q)

and several_ways = function
  | Any _ -> true
  | Node n -> (
      match n.srcs with
      | Perm (_ :: _ :: _) -> true
      | Leaf | Perm _ -> false
      | Fixed ps -> List.exists several_ways ps
      | Each q -> several_ways q)

let cover_case p expected =
  cover "no naming" (expected = []);
  cover "several namings" (List.length expected > 1);
  cover "a name used twice"
    (expected <> [] && List.length (names p) > List.length (distinct_names p));
  cover "a repeated pattern that names in several ways"
    (expected <> [] && repeats_names_in_ways p)

(* A naming as the node it gives each name of [p], or none. *)
let viewed p m =
  List.map
    (fun n -> (n, try Some (m n) with Invalid_argument _ -> None))
    (distinct_names p)

let view p store =
  List.map (fun n -> (n, List.assoc_opt n store)) (distinct_names p)

let naming = list (pair string (option uop))

(* Cases the generators reach rarely: a name a repeat binds on each source
   that a nested repeat over no sources leaves to a later path, a repeat over
   no sources, a name used inside and outside a repeat, and more orderings
   than a rule enumerates as alternatives. *)
let reference_examples =
  let n ?ops ?name ?(srcs = Leaf) () =
    Node { ops; dtypes = None; arg = None; name; srcs; any_len = false }
  in
  let x = n ~name:"x" () and y = n ~name:"y" () and z = n ~name:"z" () in
  let a = List.nth leaves 0
  and b = List.nth leaves 1
  and c0 = List.nth leaves 2
  and c1 = List.nth leaves 3 in
  let stack srcs = Ops.v ~src:srcs Op.Stack in
  let add u v = Ops.v ~src:[ u; v ] Op.Add in
  (* A variable's one source is a stack of no sources. *)
  let empty = Ops.nth a 0 in
  let var_of_empty_each =
    n ~ops:[ Ops.op a ]
      ~srcs:(Fixed [ n ~ops:[ Op.Stack ] ~srcs:(Each x) () ])
      ()
  in
  let first_later =
    n ~ops:[ Op.Stack ]
      ~srcs:(Each (n ~ops:[ Op.Add ] ~srcs:(Fixed [ var_of_empty_each; x ]) ()))
      ()
  in
  let inside_outside =
    n ~ops:[ Op.Add ]
      ~srcs:(Fixed [ n ~ops:[ Op.Stack ] ~srcs:(Each x) (); x ])
      ()
  in
  let orderings =
    n ~ops:[ Op.Stack ] ~srcs:(Perm [ Any [ x; y ]; Any [ y; z ]; z ]) ()
  in
  [
    (first_later, stack [ add a c0; add b c1 ]);
    (first_later, stack [ add a c0; add b c0 ]);
    (n ~ops:[ Op.Stack ] ~srcs:(Each x) (), empty);
    (inside_outside, add (stack [ a; a ]) a);
    (inside_outside, add (stack [ a; a ]) b);
    (orderings, stack [ a; b; c0 ]);
    (orderings, stack [ a; a; a ]);
  ]

let by_reference =
  group "matching by a reference"
    [
      prop "a rule is given the namings of a node in the reference's order"
        ~count:2000 ~examples:reference_examples case_gen (fun (p, u) ->
          let expected = if tried p u then namings p u [] else [] in
          cover_case p expected;
          let seen = ref [] in
          let m =
            Pm.fold (fun () ->
                [
                  Pm.rule (built p) (fun m ->
                      seen := viewed p m :: !seen;
                      None);
                ])
          in
          equal (option unit) None (Pm.rewrite m () u);
          equal (list naming) (List.map (view p) expected) (List.rev !seen));
      prop "match_ is the namings of a node in the reference's order"
        ~count:2000 ~examples:reference_examples case_gen (fun (p, u) ->
          let expected = namings p u [] in
          cover_case p expected;
          let sorted = List.sort (fun (n, _) (n', _) -> String.compare n n') in
          equal
            (list (list (pair string uop)))
            (List.map sorted expected)
            (List.map sorted (P.match_ (built p) u)));
    ]

(* Rewriting *)

let value_of u =
  match Ops.value u with
  | `Float x -> x
  | `Int n -> Bigint.to_float n
  | _ -> fail "a number"

let fconst x = Ops.float x

(* The rules of tinygrad's graph tests: fold sums and products of constants, and
   a weak integer constant into 1.0 + 2.0. *)
let simple_pm =
  Pm.v (fun () ->
      [
        Pm.rule (P.cvar ~dtype:[ Weak_int ] "x") (fun _ ->
            Some Ops.O.(float 1. + float 2.));
        Pm.rule
          P.(P.cvar "x" + P.cvar "y")
          (fun m -> Some (fconst (value_of (m "x") +. value_of (m "y"))));
        Pm.rule
          P.(P.cvar "x" * P.cvar "y" * P.cvar "z")
          (fun m ->
            Some
              (fconst
                 (value_of (m "x") *. value_of (m "y") *. value_of (m "z"))));
        Pm.rule
          P.(P.var "x" + P.cvar "c1" + P.cvar "c2")
          (fun m ->
            Some Ops.O.(m "x" + float (value_of (m "c1") +. value_of (m "c2"))));
      ])

let rewrite_to value u =
  let out =
    Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
      (After_sources simple_pm)
  in
  equal op Op.Const (Ops.op out);
  equal float_exact value (value_of out)

(* Rewriting with the substitution of [pairs], after the sources unless [order]
   is [before], to a fixed point unless [pass] is [Once], and outside call
   bodies unless [calls] is [Enter]. *)
let table pairs =
  let t = Ops.Tbl.create 4 in
  List.iter (fun (k, v) -> Ops.Tbl.replace t k v) pairs;
  t

let after m = Ops.After_sources m
let before m = Ops.Before_sources m

let rw ?(order = after) ?(pass = Ops.Fixed_point) ?(calls = Ops.Skip) pairs u =
  Ops.graph_rewrite ~calls ~pass ~ctx:(table pairs) u (order Ops.pm_substitute)

let sin u = Ops.alu u Op.Sin []

let three_to_four =
  Pm.v (fun () ->
      [
        Pm.rule (P.op ~arg:(Const (i 3)) Op.Const) (fun _ -> Some (Ops.int 4));
        Pm.rule (P.op ~arg:(Const (i 4)) Op.Const) (fun _ -> Some (Ops.int 3));
      ])

(* A rule that records the node it sees, and declines. *)
let label u =
  match Ops.op u with
  | Op.Const -> str Dtype.pp_const (Ops.value u)
  | o -> str Op.pp o

let recorder tag =
  Pm.v (fun () ->
      [
        Pm.rule_ctx (P.v ~op:Op.Set.all ~name:"x" ()) (fun log m ->
            log := (label (m "x") ^ tag) :: !log;
            None);
      ])

let recorded log = List.rev !log
let gate = Pm.rule (P.op Op.Add) (fun _ -> raise Ops.Bottom_up_gate)

let unreachable o =
  Pm.rule (P.op o) (fun _ -> fail "a gated node's sources were visited")

let rewriting =
  let a = var "a" 0 10 and b = var "b" 0 10 and c = var "c" 0 10 in
  group "graph_rewrite"
    [
      test "folds to a fixed point" (fun () ->
          rewrite_to 3. Ops.O.(c1 + c2);
          rewrite_to 12. Ops.O.(c1 * c2 * (c3 + c3));
          rewrite_to 6. Ops.O.(c1 + c2 + c3);
          rewrite_to 10. Ops.O.(c1 + c2 + c3 + float 4.);
          rewrite_to 7. Ops.O.(c1 + c2 + (c1 + c3));
          rewrite_to 3. (Ops.int 4));
      test "rewrites a node's result in turn" (fun () ->
          let v = fvar "v" in
          let out =
            Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
              Ops.O.(v + c1 + c2)
              (After_sources simple_pm)
          in
          equal uop Ops.O.(v + float 3.) out);
      test "keeps shared nodes shared" (fun () ->
          let v1 = fvar "v" and v2 = fvar "v" in
          let out =
            Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
              Ops.O.(v1 + v2)
              (After_sources (Pm.v (fun () -> [])))
          in
          is_true (Ops.nth out 0 == Ops.nth out 1));
      test "a rule that returns its node declines, whatever the direction"
        (fun () ->
          let m =
            Pm.v (fun () ->
                [ Pm.rule (P.op ~name:"x" Op.Param) (fun m -> Some (m "x")) ])
          in
          equal uop a
            (Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() a
               (After_sources m));
          equal uop a
            (Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() a
               (Before_sources m)));
      test "rejects rules that never settle, whatever the direction" (fun () ->
          rejects (fun () ->
              Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
                (Ops.int 3) (After_sources three_to_four));
          rejects (fun () ->
              Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
                (Ops.int 3) (Before_sources three_to_four)));
      test "rejects a replacement that depends on the node it replaces"
        (fun () ->
          let s = sin (fvar "a") in
          rejects (fun () -> rw [ (s, Ops.sqrt s) ] (Ops.sink [ s ]));
          rejects (fun () ->
              rw ~order:before [ (s, Ops.sqrt s) ] (Ops.sink [ s ]));
          let x = fvar "a" and y = fvar "b" in
          rejects (fun () ->
              rw ~order:before [ (x, Ops.sqrt y); (y, sin x) ] (Ops.sink [ x ])));
      test "rejects a call whose argument depends on the call" (fun () ->
          let staged = Ops.bufferize (fvar "a") [] in
          let m =
            Pm.v (fun () ->
                [
                  Pm.rule (P.op ~name:"x" Op.Stage) (fun m ->
                      Some
                        (Ops.call
                           (Ops.custom_function "f"
                              [ Shape.param_like (m "x") 0 ])
                           [ m "x" ]));
                ])
          in
          rejects (fun () ->
              Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
                (Ops.sink [ staged ]) (Before_sources m)));
      test "a gate keeps a bottom-up node and leaves its sources unvisited"
        (fun () ->
          let m = Pm.v (fun () -> [ gate; unreachable Op.Mul ]) in
          let u = Ops.O.((a * a) + (b * c)) in
          equal uop u
            (Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
               (Before_sources m));
          let m =
            Pm.v (fun () ->
                [
                  Pm.rule
                    P.(P.var "a" + P.var "a")
                    (fun m -> Some Ops.O.(int 2 * m "a"));
                  Pm.rule (P.op Op.Mul) (fun _ -> raise Ops.Bottom_up_gate);
                  unreachable Op.Const;
                ])
          in
          equal uop
            Ops.O.(int 2 * a)
            (Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
               Ops.O.(a + a)
               (Before_sources m)));
      test "a walk lets a gate escape" (fun () ->
          let m = Pm.v (fun () -> [ gate ]) in
          raises Ops.Bottom_up_gate (fun () ->
              Ops.graph_rewrite ~calls:Skip ~pass:Once ~ctx:()
                Ops.O.(a + b)
                (Before_sources m)));
      test "a gate from a rule after the sources escapes" (fun () ->
          let m = Pm.v (fun () -> [ gate ]) in
          raises Ops.Bottom_up_gate (fun () ->
              Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
                Ops.O.(a + b)
                (After_sources m)));
      test "bpm rewrites before the sources and the matcher after them"
        (fun () ->
          let log = ref [] in
          ignore
            (Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:log
               Ops.O.(int 1 + int 2)
               (Around_sources
                  { before = recorder " bpm"; after = recorder " pm" }));
          equal (list string)
            [ "Ops.ADD bpm"; "1 bpm"; "1 pm"; "2 bpm"; "2 pm"; "Ops.ADD pm" ]
            (recorded log));
      test "a rewrite after one an exception ended rewrites as a fresh domain"
        (fun () ->
          let raising =
            Pm.v (fun () ->
                [
                  Pm.rule (P.op ~arg:(Const (`Float 3.)) Op.Const) (fun _ ->
                      raise Exit);
                  Pm.rule (P.op ~arg:(Const (`Float 1.)) Op.Const) (fun _ ->
                      Some (fvar "one"));
                ])
          in
          let wide = Ops.sink (List.init 64 (fun k -> fconst (Float.of_int k))) in
          let u =
            Ops.sink
              ((Ops.int 0 :: List.init 64 (fun k -> fconst (Float.of_int k +. 0.5)))
              @ [ Ops.O.(c1 + c2) ])
          in
          let rewrite () =
            Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
              (Around_sources { before = simple_pm; after = simple_pm })
          in
          let expected = Domain.join (Domain.spawn rewrite) in
          raises Exit (fun () ->
              Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() wide
                (After_sources raising));
          raises Exit (fun () ->
              Ops.graph_rewrite ~calls:Skip ~pass:Once ~ctx:() wide
                (After_sources raising));
          equal uop expected (rewrite ()));
      test "the work list is bounded by REWRITE_STACK_LIMIT" (fun () ->
          equal string "REWRITE_STACK_LIMIT"
            (Setting.key Setting.rewrite_stack_limit);
          equal int 250000 (Setting.value Setting.rewrite_stack_limit);
          let wide = Ops.sink (List.init 64 (fun n -> Ops.int n)) in
          Setting.context
            [ B (Setting.rewrite_stack_limit, 8) ]
            (fun () ->
              rejects (fun () ->
                  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() wide
                    (After_sources (Pm.v (fun () -> []))))));
    ]

let substituting =
  let a = var "a" 0 10
  and b = var "b" 0 10
  and c = var "c" 0 10
  and d = var "d" 0 10 in
  let x = fvar "x" in
  group "substitute"
    [
      test "replaces a node wherever it is" (fun () ->
          equal uop
            Ops.O.(b + int 4)
            (Ops.substitute ~calls:Skip ~pass:Fixed_point
               Ops.O.(a + int 4)
               [ (a, b) ]);
          equal uop
            Ops.O.(c + int 4 + c)
            (Ops.substitute ~calls:Skip ~pass:Fixed_point
               Ops.O.(a + int 4 + b)
               [ (a, c); (b, c) ]);
          equal uop
            Ops.O.(b + int 4 + (b + int 5))
            (Ops.substitute ~calls:Skip ~pass:Fixed_point
               Ops.O.(a + int 4 + (a + int 5))
               [ (a, b) ]));
      test "replaces the node nearest the root first" (fun () ->
          let y = fvar "y" in
          equal uop (sin y)
            (Ops.substitute ~calls:Skip ~pass:Fixed_point
               (sin (sin x))
               [ (sin x, y) ]);
          equal uop
            (sin (Ops.sqrt x))
            (Ops.substitute ~calls:Skip ~pass:Fixed_point
               (sin (sin x))
               [ (sin x, Ops.sqrt x) ]);
          equal uop
            (Ops.sqrt (Ops.sqrt x))
            (Ops.substitute ~calls:Skip ~pass:Fixed_point
               (sin (sin x))
               [ (sin x, Ops.sqrt x); (sin (sin x), Ops.sqrt (sin x)) ]));
      test "ignores nodes paired with themselves" (fun () ->
          let e = Ops.O.(a + int 4) in
          is_true
            (Ops.substitute ~calls:Skip ~pass:Fixed_point e [ (a, a) ] == e);
          is_true (Ops.substitute ~calls:Skip ~pass:Fixed_point e [] == e));
      test "rewrites with extra_pm too, never inside a replacement" (fun () ->
          let three = Ops.int 3 and four = Ops.int 4 in
          let s = Ops.O.(three + four) in
          let visit =
            Pm.v (fun () ->
                [
                  Pm.rule (P.op ~name:"c" Op.Const) (fun m ->
                      let c = m "c" in
                      if c == three || c == four then
                        fail "entered the replaced node"
                      else None);
                ])
          in
          equal uop
            Ops.O.(int 7 + int 2)
            (Ops.substitute ~calls:Skip ~pass:Fixed_point
               ~extra_pm:(Pm.with_ctx visit)
               Ops.O.(s + int 2)
               [ (s, Ops.int 7) ]));
      test "walk does not enter a replacement" (fun () ->
          equal uop
            Ops.O.(b + c + int 4)
            (Ops.substitute ~calls:Skip ~pass:Once
               Ops.O.(a + int 4)
               [ (a, Ops.O.(b + c)); (b, d) ]);
          equal uop
            Ops.O.(d + c + int 4)
            (Ops.substitute ~calls:Skip ~pass:Fixed_point
               Ops.O.(a + int 4)
               [ (a, Ops.O.(b + c)); (b, d) ]));
      prop "is idempotent when no replacement holds a replaced node"
        Nodes.gen_recipe (fun r ->
          let vars = Nodes.leaves () in
          let u = Nodes.build vars r in
          cover "replaces" (List.memq vars.(0) (Ops.toposort ~calls:Enter u));
          let f u =
            Ops.substitute ~calls:Skip ~pass:Fixed_point u
              [ (vars.(0), Ops.O.(vars.(1) + int 1)) ]
          in
          equal uop (f u) (f (f u)));
    ]

let walks =
  let a = var "a" 0 10
  and b = var "b" 0 10
  and c = var "c" 0 10
  and d = var "d" 0 10 in
  let x = fvar "x" in
  group "walk"
    [
      test "top-down, substitutes once" (fun () ->
          equal uop
            Ops.O.(b + int 4)
            (rw ~pass:Once [ (a, b) ] Ops.O.(a + int 4));
          equal uop
            Ops.O.(c + int 4 + (c + int 5))
            (rw ~pass:Once [ (a, c); (b, c) ] Ops.O.(a + int 4 + (b + int 5)));
          equal uop
            Ops.O.(b + int 4 + (b + int 5))
            (rw ~pass:Once [ (a, b) ] Ops.O.(a + int 4 + (a + int 5))));
      test "top-down, does not enter a replacement" (fun () ->
          equal uop
            Ops.O.(b + c + int 4)
            (rw ~pass:Once [ (a, Ops.O.(b + c)); (b, d) ] Ops.O.(a + int 4));
          equal uop
            Ops.O.(d + c + int 4)
            (rw ~order:before [ (a, Ops.O.(b + c)); (b, d) ] Ops.O.(a + int 4)));
      test "top-down, applies a bouncing rule once" (fun () ->
          equal uop (Ops.int 4)
            (Ops.graph_rewrite ~calls:Skip ~pass:Once ~ctx:() (Ops.int 3)
               (After_sources three_to_four)));
      test "top-down, rewrites the sources before the rebuilt node" (fun () ->
          let n1 = sin x in
          equal uop
            (sin (Ops.sqrt x))
            (rw ~pass:Once
               [ (sin x, Ops.sqrt x); (sin n1, Ops.sqrt n1) ]
               (sin n1)));
      test "top-down, accepts a replacement that holds the replaced node"
        (fun () ->
          equal uop
            Ops.O.(Ops.sqrt (sin x) + int 4)
            (rw ~pass:Once [ (sin x, Ops.sqrt (sin x)) ] Ops.O.(sin x + int 4)));
      test "top-down, visits after the sources" (fun () ->
          let log = ref [] in
          ignore
            (Ops.graph_rewrite ~calls:Skip ~pass:Once ~ctx:log
               Ops.O.(int 1 + int 2)
               (After_sources (recorder "")));
          equal (list string) [ "1"; "2"; "Ops.ADD" ] (recorded log));
      test "bottom-up, substitutes once and never enters a replacement"
        (fun () ->
          equal uop
            Ops.O.(b + int 4)
            (rw ~order:before ~pass:Once [ (a, b) ] Ops.O.(a + int 4));
          equal uop
            Ops.O.(b + c + int 4)
            (rw ~order:before ~pass:Once
               [ (a, Ops.O.(b + c)); (b, d) ]
               Ops.O.(a + int 4));
          equal uop
            Ops.O.(c + int 4 + (c + int 5))
            (rw ~order:before ~pass:Once
               [ (a, c); (b, c) ]
               Ops.O.(a + int 4 + (b + int 5))));
      test "bottom-up, a matched node's sources are never visited" (fun () ->
          let n1 = sin x in
          equal uop
            (Ops.sqrt (sin x))
            (rw ~order:before ~pass:Once
               [ (sin x, Ops.sqrt x); (sin n1, Ops.sqrt n1) ]
               (sin n1)));
      test "bottom-up, applies a bouncing rule once" (fun () ->
          equal uop (Ops.int 4)
            (Ops.graph_rewrite ~calls:Skip ~pass:Once ~ctx:() (Ops.int 3)
               (Before_sources three_to_four)));
      test "bottom-up, visits before the sources" (fun () ->
          let log = ref [] in
          ignore
            (Ops.graph_rewrite ~calls:Skip ~pass:Once ~ctx:log
               Ops.O.(int 1 + int 2)
               (Before_sources (recorder "")));
          equal (list string) [ "Ops.ADD"; "1"; "2" ] (recorded log));
      test "both ways, bpm visits before the sources and the matcher after"
        (fun () ->
          let log = ref [] in
          ignore
            (Ops.graph_rewrite ~calls:Skip ~pass:Once ~ctx:log
               Ops.O.(int 1 + int 2)
               (Around_sources
                  { before = recorder " bpm"; after = recorder " pm" }));
          equal (list string)
            [ "Ops.ADD bpm"; "1 bpm"; "1 pm"; "2 bpm"; "2 pm"; "Ops.ADD pm" ]
            (recorded log));
      test "both ways, a bpm match skips the node's sources and its matcher"
        (fun () ->
          let log = ref [] in
          let one = Ops.int 1 and two = Ops.int 2 in
          let bpm =
            Pm.v (fun () ->
                [
                  Pm.rule_ctx (P.v ~op:Op.Set.all ~name:"x" ()) (fun log m ->
                      let x = m "x" in
                      log := (label x ^ " bpm") :: !log;
                      if x == one then Some (Ops.int 10) else None);
                ])
          in
          let out =
            Ops.graph_rewrite ~calls:Skip ~pass:Once ~ctx:log
              Ops.O.(one + two)
              (Around_sources { before = bpm; after = recorder " pm" })
          in
          equal uop Ops.O.(int 10 + two) out;
          is_false (List.mem "1 pm" (recorded log));
          is_true (List.mem "2 pm" (recorded log)));
    ]

let call_bodies =
  let a = Ops.int 3 and b = Ops.int 4 in
  let modes =
    [
      ("fixed point, after", Ops.Fixed_point, after);
      ("fixed point, before", Ops.Fixed_point, before);
      ("once, after", Ops.Once, after);
      ("once, before", Ops.Once, before);
    ]
  in
  group "calls"
    [
      test "a node can become a call that holds it" (fun () ->
          let staged = Ops.bufferize (fvar "a") [] in
          let call = Ops.call (Ops.custom_function "f" [ staged ]) [] in
          List.iter
            (fun (mode, pass, order) ->
              equal ~msg:mode uop (Ops.sink [ call ])
                (rw ~order ~pass [ (staged, call) ] (Ops.sink [ staged ])))
            modes);
      test "a body is rewritten only under Enter, its arguments always"
        (fun () ->
          let call = Ops.call (Ops.custom_function "f" [ a ]) [ a ] in
          List.iter
            (fun (mode, pass, order) ->
              List.iter
                (fun (name, calls, body) ->
                  equal
                    ~msg:(Printf.sprintf "%s, %s" mode name)
                    uop
                    (Ops.call (Ops.custom_function "f" [ body ]) [ b ])
                    (rw ~order ~pass ~calls [ (a, b) ] call))
                [ ("Skip", Ops.Skip, a); ("Enter", Ops.Enter, b) ])
            modes);
      test "a body shared with a sibling is left alone, the sibling rewritten"
        (fun () ->
          let call = Ops.call (Ops.custom_function "f" [ a ]) [] in
          List.iter
            (fun (mode, pass, order) ->
              equal ~msg:mode uop
                (Ops.sink [ call; b ])
                (rw ~order ~pass [ (a, b) ] (Ops.sink [ call; a ]));
              equal ~msg:mode uop
                (Ops.sink [ b; call ])
                (rw ~order ~pass [ (a, b) ] (Ops.sink [ a; call ])))
            modes);
      test "substitute leaves call bodies alone unless under Enter" (fun () ->
          let call = Ops.call (Ops.custom_function "f" [ a ]) [ a ] in
          equal uop
            (Ops.call (Ops.custom_function "f" [ a ]) [ b ])
            (Ops.substitute ~calls:Skip ~pass:Fixed_point call [ (a, b) ]);
          equal uop
            (Ops.call (Ops.custom_function "f" [ b ]) [ b ])
            (Ops.substitute ~calls:Enter ~pass:Fixed_point call [ (a, b) ]));
    ]

(* Fixed points, over expressions of weak integers *)

let fold =
  Pm.v (fun () ->
      [
        Pm.rule
          P.(P.cvar "x" + P.cvar "y")
          (fun m ->
            Some
              (Ops.const
                 (Ops.exec_alu Op.Add Weak_int
                    [ Ops.value (m "x"); Ops.value (m "y") ])));
        Pm.rule
          P.(P.cvar "x" * P.cvar "y")
          (fun m ->
            Some
              (Ops.const
                 (Ops.exec_alu Op.Mul Weak_int
                    [ Ops.value (m "x"); Ops.value (m "y") ])));
        Pm.rule P.(P.var "x" + int 0) (fun m -> Some (m "x"));
      ])

type expr = X | K of int | Plus of expr * expr | Times of expr * expr

let rec pp_expr ppf = function
  | X -> Format.pp_print_string ppf "x"
  | K n -> Format.pp_print_int ppf n
  | Plus (a, b) -> Format.fprintf ppf "(%a + %a)" pp_expr a pp_expr b
  | Times (a, b) -> Format.fprintf ppf "(%a * %a)" pp_expr a pp_expr b

let gen_expr =
  let open Gen in
  let rec go depth =
    if depth = 0 then
      frequency
        [
          (1, constant ~pp:pp_expr X); (2, map (fun n -> K n) (int_range (-3) 3));
        ]
    else
      frequency
        [
          (1, go 0);
          ( 2,
            map
              (fun (a, b) -> Plus (a, b))
              (pair (go (depth - 1)) (go (depth - 1))) );
          ( 2,
            map
              (fun (a, b) -> Times (a, b))
              (pair (go (depth - 1)) (go (depth - 1))) );
        ]
  in
  with_pp pp_expr (bind (int_range 0 5) go)

let rec expr x = function
  | X -> x
  | K n -> Ops.int n
  | Plus (a, b) -> Ops.v ~src:[ expr x a; expr x b ] Op.Add
  | Times (a, b) -> Ops.v ~src:[ expr x a; expr x b ] Op.Mul

let fixed_points =
  let x = weak_var "x" (-5) 5 in
  let rewrite u =
    Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
      (After_sources fold)
  in
  group "fixed points"
    [
      prop "rewriting a rewritten graph changes nothing" gen_expr (fun e ->
          let u = expr x e in
          cover "folds" (rewrite u != u);
          equal uop (rewrite u) (rewrite (rewrite u)));
      prop "a rewrite preserves the value"
        (Gen.pair gen_expr (Gen.int_range (-5) 5))
        (fun (e, at) ->
          let u = expr x e in
          let env = [ ("x", i at) ] in
          equal value (Values.eval env u) (Values.eval env (rewrite u)));
      prop "bottom-up reaches the same fixed point on these rules" gen_expr
        (fun e ->
          let u = expr x e in
          equal uop (rewrite u)
            (Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
               (Before_sources fold)));
    ]

(* The matchers of the module *)

let module_matchers =
  let p = Shape.param ~shape:(ints [ 4 ]) 0 Float32 in
  group "module matchers"
    [
      test "pm_drop_after keeps the first source of each after" (fun () ->
          let st =
            Ops.store (Ops.index p [ Ops.int 0 ]) (Ops.float ~dtype:Float32 1.)
          in
          equal uop
            Ops.O.(p + float 1.)
            (Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
               Ops.O.(Ops.after p [ st ] + float 1.)
               (After_sources Ops.pm_drop_after)));
      test "gate_kernel_sink stops at a linear program and a kernel's sink"
        (fun () ->
          is_false (Ops.gate_kernel_sink (Ops.v Op.Linear));
          is_false
            (Ops.gate_kernel_sink (Ops.sink ~kernel:(Ops.kernel_info ()) []));
          is_true (Ops.gate_kernel_sink (Ops.sink []));
          is_true (Ops.gate_kernel_sink p));
    ]

let groups =
  [
    upat;
    elementwise_patterns;
    matchers;
    by_reference;
    rewriting;
    substituting;
    walks;
    call_bodies;
    fixed_points;
    module_matchers;
  ]
