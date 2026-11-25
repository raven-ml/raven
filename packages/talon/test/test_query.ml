(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let str pp v = Format.asprintf "%a" pp v

(* [report f] is the message of the [Invalid_argument] that [f] raises. *)
let report f =
  match f () with _ -> "no exception" | exception Invalid_argument m -> m

(* [problems f] is [report f] without its last lines, which give the inputs'
   schemas. *)
let problems f =
  let input line =
    List.exists
      (fun prefix -> String.starts_with ~prefix line)
      [ "  input ("; "  left ("; "  right ("; "  rest (" ]
  in
  String.split_on_char '\n' (report f)
  |> List.filter (fun line -> not (input line))
  |> String.concat "\n"

let source ?rows name columns =
  Source.v ~name ~schema:(Schema.v columns) ?rows (fun _ -> Ok [])

let names q = List.map fst (Schema.columns (Query.schema q))
let schema_w = Testable.make ~pp:Schema.pp ~equal:Schema.equal
let query_w = Testable.make ~pp:Query.pp ~equal:Query.equal

(* [words f] is the words [f ()] allocates, and its result. *)
let words f =
  let total () =
    let minor, promoted, major = Gc.counters () in
    minor +. major -. promoted
  in
  let before = total () in
  let x = f () in
  (int_of_float (total () -. before), x)

(* Fixtures *)

let flights_columns =
  Type.
    [
      ("year", Any int16);
      ("month", Any int8);
      ("day", Any int8);
      ("dep_time", Any int32);
      ("sched_dep_time", Any int32);
      ("dep_delay", Any float64);
      ("arr_time", Any int32);
      ("sched_arr_time", Any int32);
      ("arr_delay", Any float64);
      ("carrier", Any string);
      ("flight", Any int32);
      ("tailnum", Any string);
      ("origin", Any string);
      ("dest", Any string);
      ("air_time", Any float64);
      ("distance", Any float64);
      ("hour", Any int8);
      ("minute", Any int8);
      ("time_hour", Any (instant Us));
    ]

let flights = source {|csv "flights.csv"|} flights_columns

let carriers =
  source {|parquet "carriers.parquet"|}
    Type.[ ("carrier", Any string); ("name", Any string) ]

let delay = Col.float "dep_delay"
let epoch_t = Type.ext ~name:"ymir.epoch" Type.float64

let epoch =
  Ext.v ~name:"ymir.epoch" ~ordered:true Type.float64 ~dec:Fun.id ~enc:Fun.id

(* [kinds] has a column of each sort a verb treats apart. *)
let kinds =
  source "kinds"
    Type.
      [
        ("x8", Any int8);
        ("u8", Any uint8);
        ("f", Any float64);
        ("f32", Any float32);
        ("n", Any int64);
        ("g", Any string);
        ("d", Any date);
        ("b", Any bool);
        ("ts", Any (instant Ns));
        ("t", Any epoch_t);
        ("l", Any (list int64));
        ("r", Any (record [ ("e", Any epoch_t) ]));
        ("wait", Any (duration Ms));
      ]

let of_kinds = Query.of_source kinds
let x8 = Col.int "x8"
let u8 = Col.int "u8"
let n = Col.int "n"
let f = Col.float "f"

let late_by_carrier () =
  Query.(
    of_source flights
    |> filter Expr.(delay > float 15.)
    |> aggregate ~by:[ "carrier" ]
         Expr.[ "mean_delay" := mean delay; "flights" := rows ]
    |> join ~on:(Join.keys [ "carrier" ]) ~each_left:One (of_source carriers)
    |> sort [ Order.desc "mean_delay" ])

(* Sel *)

(* [selector] has names that a prefix and a suffix both match, an extension of
   float storage and a list. *)
let selector =
  Query.of_source
    (source "selector"
       Type.
         [
           ("a", Any int8);
           ("ab", Any float64);
           ("b", Any string);
           ("ba", Any epoch_t);
           ("c", Any (list int64));
         ])

let selected ?(q = selector) sel = names (Query.select Expr.[ keep sel ] q)

(* [two_names] draws two lists of distinct names of [selector], each in any
   order. *)
let two_names =
  let one =
    Gen.(
      let* ns = permutation [ "a"; "ab"; "b"; "ba"; "c" ] in
      subsequence ns)
  in
  Gen.with_pp
    (fun ppf (ns0, ns1) ->
      Format.fprintf ppf "[%s], [%s]" (String.concat "; " ns0)
        (String.concat "; " ns1))
    (Gen.pair one one)

let two_names_examples = [ ([], [ "a" ]); ([ "a"; "b" ], [ "b"; "a" ]) ]
let without ns0 ns1 = List.filter (fun n -> not (List.mem n ns1)) ns0

let sel =
  group "Sel"
    [
      cases "selects"
        ~id:(fun (what, _, _) -> what)
        [
          ( "all, every column in schema order",
            Sel.all,
            [ "a"; "ab"; "b"; "ba"; "c" ] );
          ("names, in their order", Sel.names [ "b"; "a" ], [ "b"; "a" ]);
          ( "names, a repeat once at its first position",
            Sel.names [ "b"; "a"; "b" ],
            [ "b"; "a" ] );
          ("prefix, in schema order", Sel.prefix "a", [ "a"; "ab" ]);
          ("suffix, in schema order", Sel.suffix "a", [ "a"; "ba" ]);
          ("of_kind, the columns its kind binds", Sel.of_kind Kind.int, [ "a" ]);
          ("of_kind, no extension column", Sel.of_kind Kind.float, [ "ab" ]);
          ( "where, the columns its predicate holds of",
            Sel.where (fun n _ -> String.length n = 2),
            [ "ab"; "ba" ] );
          ("a prefix that matches nothing, no column", Sel.prefix "z", []);
        ]
        (fun (_, sel, expected) -> equal (list string) expected (selected sel));
      test "names of a missing column is a problem" (fun () ->
          expect
            (problems (fun () ->
                 Query.select Expr.[ keep (Sel.names [ "a"; "nope" ]) ] selector))
          @@ __POS_OF__
               {|
            select: 1 problem
              keep (names ["a"; "nope"])
                no column "nope". The columns are "a", "ab", "b", "ba" and "c".
            |});
      prop "s0 + s1 is s0's columns, then s1's not in s0"
        ~examples:two_names_examples two_names (fun (ns0, ns1) ->
          equal (list string)
            (ns0 @ without ns1 ns0)
            (selected Sel.(names ns0 + names ns1)));
      prop "s0 - s1 is s0's columns not in s1" ~examples:two_names_examples
        two_names (fun (ns0, ns1) ->
          equal (list string) (without ns0 ns1)
            (selected Sel.(names ns0 - names ns1)));
      prop "inter s0 s1 is s0's columns in s1" ~examples:two_names_examples
        two_names (fun (ns0, ns1) ->
          equal (list string)
            (List.filter (fun n -> List.mem n ns1) ns0)
            (selected (Sel.inter (Sel.names ns0) (Sel.names ns1))));
      test "a selector resolves against the input of the verb that applies it"
        (fun () ->
          let s = Sel.(prefix "a" + of_kind Kind.float) in
          equal (list string) [ "a"; "ab" ] (selected s);
          equal (list string) [ "f"; "f32" ] (selected ~q:of_kinds s));
    ]

(* Source.v *)

let source_v =
  let schema = Query.schema of_kinds in
  group "Source.v"
    [
      cases "refuses"
        ~id:(fun (what, _, _, _) -> what)
        [
          ("an empty name", "", None, []);
          ("a line break in the name", "a\nb", None, []);
          ("DEL in the name", "a\x7f", None, []);
          ("a C1 control in the name", "a\xc2\x85", None, []);
          ("a name that is not UTF-8", "\xff", None, []);
          ("a negative row count", "s", Some (-1), []);
          ("a sorted key that names no column", "s", None, [ Order.asc "nope" ]);
          ("a column sorted twice", "s", None, Order.[ asc "f"; desc "f" ]);
          ("a sorted extension column", "s", None, [ Order.asc "t" ]);
          ("a sorted type holding an extension", "s", None, [ Order.asc "r" ]);
        ]
        (fun (_, name, rows, sorted) ->
          raises_match (Exn.invalid_arg ~sub:"Source.v: ") (fun () ->
              Source.v ~name ~schema ?rows ~sorted (fun _ -> Ok [])));
      test "reports every problem of ~sorted" (fun () ->
          expect
            (report (fun () ->
                 Source.v ~name:"s" ~schema
                   ~sorted:Order.[ asc "nope"; desc "f"; desc "f"; asc "t" ]
                   (fun _ -> Ok [])))
          @@ __POS_OF__
               {|
            Source.v: ~sorted:
              asc "nope": no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
              desc "f": the column "f" is already a key.
              asc "t": "t" is ext[ymir.epoch, float64], which orders only through its declaration.
            |});
      test "accepts an order on ordered columns and a row count of zero"
        (fun () ->
          ignore
            (Source.v ~name:"s" ~schema ~rows:0
               ~sorted:Order.[ asc "n"; desc "f" ]
               (fun _ -> Ok [])));
      test "a name with spaces and non-ASCII prints as it reads" (fun () ->
          equal text "query →\nselect []\n└ a\xc2\xa0b é (0 columns)"
            (str Query.pp
               (Query.select [] (Query.of_source (source "a\xc2\xa0b é" [])))));
      test "two sources built from the same arguments are different" (fun () ->
          let s () = source "s" Type.[ ("a", Any int64) ] in
          not_equal query_w (Query.of_source (s ())) (Query.of_source (s ())));
    ]

(* Join conditions *)

(* [semi_on c] is the semi join on [c] of two fixed sources. *)
let semi_on =
  let columns = Type.[ ("a", Any int64); ("b", Any int64) ] in
  let l = Query.of_source (source "l" columns)
  and r = Query.of_source (source "r" columns) in
  fun c -> Query.join ~kind:Semi ~on:c r l

let refused ~by f = raises_match (Exn.invalid_arg ~sub:by) f

let conditions =
  group "Join conditions"
    [
      test "keys refuses no key and a key named twice" (fun () ->
          refused ~by:"Join.keys" (fun () -> Join.keys []);
          refused ~by:"Join.keys" (fun () -> Join.keys [ "a"; "b"; "a" ]));
      test "( && ) refuses position with another atom" (fun () ->
          refused ~by:"Join.( && )" (fun () -> Join.(position && keys [ "a" ]));
          refused ~by:"Join.( && )" (fun () ->
              Join.(keys [ "a"; "b" ] && position));
          refused ~by:"Join.( && )" (fun () -> Join.(position && le "a" "b")));
      test "( && ) refuses an atom twice" (fun () ->
          refused ~by:"Join.( && )" (fun () ->
              Join.(keys [ "a" ] && keys [ "a" ]));
          refused ~by:"Join.( && )" (fun () ->
              Join.(eq "a" "b" && keys [ "c" ] && eq "a" "b"));
          refused ~by:"Join.( && )" (fun () ->
              Join.(lt "a" "b" && keys [ "c" ] && lt "a" "b"));
          refused ~by:"Join.( && )" (fun () ->
              Join.(
                overlaps ("a", "b") ("c", "d") && overlaps ("a", "b") ("c", "d")));
          refused ~by:"Join.( && )" (fun () ->
              Join.(inside "a" ("b", "c") && ge "a" "b")));
      test "( && ) accepts comparisons that bound several columns" (fun () ->
          ignore
            Join.(lt "a" "b" && gt "b" "a" && overlaps ("a", "b") ("a", "b")));
      test "( && ) accepts two equality atoms on one left column" (fun () ->
          ignore Join.(eq "a" "b" && eq "a" "c"));
      test "inside is a lower and an upper comparison" (fun () ->
          equal query_w
            (semi_on Join.(inside "a" ("a", "b")))
            (semi_on Join.(ge "a" "a" && lt "a" "b")));
      test "all is the unit of &&" (fun () ->
          equal query_w
            (semi_on (Join.keys [ "a" ]))
            (semi_on Join.(all && keys [ "a" ]));
          equal query_w
            (semi_on (Join.keys [ "a" ]))
            (semi_on Join.(keys [ "a" ] && all)));
      test "keys of two names is two keys" (fun () ->
          equal query_w
            (semi_on (Join.keys [ "a"; "b" ]))
            (semi_on Join.(keys [ "a" ] && keys [ "b" ])));
      cases "Join.equal tells apart"
        ~id:(fun (what, _, _) -> what)
        Join.
          [
            ("the left columns of eq", eq "a" "b", eq "b" "b");
            ("the right columns of eq", eq "a" "a", eq "a" "b");
            ("position and all", position, all);
            ("position and keys", position, keys [ "a" ]);
            ("one atom and two", keys [ "a" ], keys [ "a"; "b" ]);
            ("the order of atoms", keys [ "a"; "b" ], keys [ "b"; "a" ]);
            ("the comparison of a comparison", lt "a" "b", le "a" "b");
            ("the sides of a comparison", lt "a" "b", lt "b" "a");
            ( "the order of a period",
              overlaps ("a", "b") ("a", "b"),
              overlaps ("b", "a") ("a", "b") );
            ("a comparison and an equality", le "a" "b", eq "a" "b");
          ]
        (fun (_, c0, c1) ->
          equal query_w (semi_on c0) (semi_on c0);
          not_equal query_w (semi_on c0) (semi_on c1));
    ]

(* Query: problems *)

(* [of_columns n] is a query of the first [n] columns of the flights. *)
let of_columns n =
  Query.of_source
    (source "first" (List.filteri (fun i _ -> i < n) flights_columns))

(* [inputs n] is the last line of a verb's report over [of_columns n]. *)
let inputs n =
  let lines =
    String.split_on_char '\n'
      (report (fun () -> Query.slice ~offset:0 ~length:(-1) (of_columns n)))
  in
  List.nth lines (List.length lines - 1)

(* [suggestion key columns] is the problem that [sort] reports for a key [key]
   over the columns [columns]. *)
let suggestion key columns =
  let q =
    Query.of_source
      (source "s" (List.map (fun n -> (n, Type.Any Type.int64)) columns))
  in
  let report = report (fun () -> Query.sort [ Order.asc key ] q) in
  List.nth (String.split_on_char '\n' report) 1

let reports =
  group "Query problems"
    [
      test "a verb reports every problem at once" (fun () ->
          (* Query's "Problems" example. *)
          equal text
            {|aggregate: 3 problems
  ~by: no column "carier". Did you mean "carrier"?
  "mean_delay" := mean dep_dly
    no column "dep_dly". Did you mean "dep_delay"?
  "late" := mean carrier
    Col.float reads float16, float32 or float64, but "carrier" is string.
  input (19 columns): year int16, month int8, day int8, dep_time int32, sched_dep_time int32, dep_delay float64, arr_time int32, sched_arr_time int32, …|}
            (report (fun () ->
                 Query.(
                   of_source flights
                   |> aggregate ~by:[ "carier" ]
                        Expr.
                          [
                            "mean_delay" := mean (Col.float "dep_dly");
                            "late" := mean (Col.float "carrier");
                          ]))));
      test "the last line gives an input of eight columns whole" (fun () ->
          equal text
            "  input (8 columns): year int16, month int8, day int8, dep_time \
             int32, sched_dep_time int32, dep_delay float64, arr_time int32, \
             sched_arr_time int32"
            (inputs 8));
      test "the last line gives an input of one column and of none" (fun () ->
          expect (inputs 1 ^ "\n" ^ inputs 0)
          @@ __POS_OF__
               {|
            input (1 column): year int16
            input (0 columns)
            |});
      test "a join's inputs are left and right" (fun () ->
          expect
            (report (fun () ->
                 Query.join ~on:(Join.keys [ "nope" ])
                   (Query.of_source (source "z" Type.[ ("z", Any int64) ]))
                   (of_columns 2)))
          @@ __POS_OF__
               {|
            join: 2 problems
              left: no column "nope". The columns are "year" and "month".
              right: no column "nope". The columns are "z".
              left (2 columns): year int16, month int8
              right (1 column): z int64
            |});
      test "append's inputs are input and rest" (fun () ->
          expect (report (fun () -> Query.append (of_columns 1) (of_columns 2)))
          @@ __POS_OF__
               {|
            append: 1 problem
              "month" (int8) is not in rest.
              input (2 columns): year int16, month int8
              rest (1 column): year int16
            |});
      cases "a missing name comes with"
        ~id:(fun (what, _, _, _) -> what)
        [
          ( "the column at distance 2",
            "ab",
            [ "abxy"; "zzzzzz" ],
            {|no column "ab". Did you mean "abxy"?|} );
          ( "every column when the nearest is at distance 3",
            "abc",
            [ "xyz" ],
            {|no column "abc". The columns are "xyz".|} );
          ( "every column past distance 2",
            "abcd",
            [ "wxyz"; "abxyz" ],
            {|no column "abcd". The columns are "wxyz" and "abxyz".|} );
          ( "the nearest columns only",
            "abcd",
            [ "abxx"; "abce" ],
            {|no column "abcd". Did you mean "abce"?|} );
          ( "ties in the columns' order",
            "ab",
            [ "xb"; "zzzz"; "ay" ],
            {|no column "ab". Did you mean "xb" or "ay"?|} );
          ("no columns", "a", [], {|no column "a". There are no columns.|});
        ]
        (fun (_, key, columns, expected) ->
          equal string
            (Printf.sprintf "  asc %S: %s" key expected)
            (suggestion key columns));
      test "an exception a user function raises propagates from the verb"
        (fun () ->
          let boom _ _ = raise Exit in
          raises Exit (fun () ->
              Query.select Expr.[ keep (Sel.where boom) ] of_kinds);
          raises Exit (fun () ->
              Query.select Expr.[ across Kind.int Sel.all boom ] of_kinds);
          raises Exit (fun () ->
              Query.select Expr.[ each Sel.all { column = boom } ] of_kinds);
          raises Exit (fun () ->
              Query.select
                Expr.[ "e" := nx { f = (fun _ -> raise Exit) } f ]
                of_kinds));
      test "building, schema, optimize, pp and equal read no data" (fun () ->
          let unread =
            Source.v ~name:"s" ~schema:(Schema.v flights_columns) (fun _ ->
                fail "the source's parts were asked for")
          in
          let q () =
            Query.(
              of_source unread
              |> filter Expr.(delay > float 15.)
              |> aggregate ~by:[ "carrier" ] Expr.[ "n" := rows ])
          in
          ignore (Query.schema (q ()));
          ignore (str Query.pp (Query.optimize (q ())));
          equal query_w (q ()) (q ()));
    ]

(* Query: binding *)

let derive os () = Query.derive os of_kinds
let aggregate os () = Query.aggregate ~by:[] os of_kinds

let binding =
  group "Query binding"
    [
      test "a literal its operand's type does not hold is a problem" (fun () ->
          expect (problems (derive Expr.[ "k" := x8 + int 300 ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := x8 + 300
                int8 does not hold the literal 300.
            |});
      test "a constant its operand's type does not hold is a problem" (fun () ->
          expect (problems (derive Expr.[ "k" := x8 + (int 100 + int 100) ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := x8 + (100 + 100)
                int8 does not hold the constant 200.
            |});
      test "types that do not meet are a problem" (fun () ->
          expect (problems (derive Expr.[ "k" := x8 + u8 ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := x8 + u8
                int8 and uint8 do not meet: cast first.
            |});
      test "a value store's type does not hold is a problem" (fun () ->
          expect
            (problems (derive Expr.[ "k" := store Type.int8 (const 1000) ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := store int8 <const>
                int8 does not hold the value 1000.
            |});
      test "an OCaml value or null that meets no type is a problem" (fun () ->
          expect
            (String.concat "\n"
               (List.map problems
                  Expr.
                    [
                      aggregate [ "k" := sum (const (fun v -> v) $ f) ];
                      derive [ "k" := null ];
                      derive [ "k" := const 1 ];
                      derive [ "k" := if_ (Col.bool "b") (const 1) (const 2) ];
                      derive [ "k" := const (fun a b -> (a, b)) $ n $ option f ];
                    ]))
          @@ __POS_OF__
               {|
            aggregate: 1 problem
              "k" := sum (<const> $ f)
                <const> $ f has no type: give it one with store.
            derive: 1 problem
              "k" := null
                null has no type: give it one with store.
            derive: 1 problem
              "k" := <const>
                <const> has no type: give it one with store.
            derive: 1 problem
              "k" := if_ b <const> <const>
                if_ b <const> <const> has no type: give it one with store.
            derive: 1 problem
              "k" := <const> $ n $ option f
                <const> $ n $ option f has no type: give it one with store.
            |});
      test "across names each selected column its kind does not read" (fun () ->
          expect
            (problems
               (derive
                  Expr.
                    [ across Kind.float Sel.(names [ "f"; "x8"; "t" ]) ( := ) ]))
          @@ __POS_OF__
               {|
            derive: 2 problems
              across float (names ["f"; "x8"; "t"]) <fn>
                across float: "x8" is int8, which float does not read: narrow the selector with Sel.of_kind.
                across float: "t" is ext[ymir.epoch, float64], which float does not read: narrow the selector with Sel.of_kind.
            |});
      test "each refuses to order an extension read without its declaration"
        (fun () ->
          expect
            (problems
               (derive
                  Expr.
                    [
                      each
                        Sel.(names [ "t" ])
                        { column = (fun nm x -> nm := over (min x)) };
                    ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              each (names ["t"]) <fn>
                min orders values, and ext[ymir.epoch, float64] is an extension read without its declaration: read it with an Ext.t declared ~ordered:true.
            |});
      test "a kept name the input lacks is a problem" (fun () ->
          expect (problems (derive Expr.[ keep Sel.(names [ "fx" ]) ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              keep (names ["fx"])
                no column "fx". Did you mean "f"?
            |});
      test "a handle of a kind the column does not have is a problem" (fun () ->
          expect (problems (derive Expr.[ "k" := Col.float "n" +. f ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := n +. f
                Col.float reads float16, float32 or float64, but "n" is int64.
            |});
      test "a missing column read twice is one problem" (fun () ->
          expect
            (problems
               (derive
                  Expr.[ "k" := Col.float "nope" +. f +. Col.float "nope" ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := nope +. f +. nope
                no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
            |});
      test "Ext.col of a column of another type is a problem" (fun () ->
          expect
            (problems (derive Expr.[ "k" := over (min (Ext.col epoch "x8")) ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := over (min x8)
                Ext.col binds ext[ymir.epoch, float64], but "x8" is int8.
            |});
      test "mean of an extension is a problem" (fun () ->
          expect (problems (aggregate Expr.[ "k" := mean (Ext.col epoch "t") ]))
          @@ __POS_OF__
               {|
            aggregate: 1 problem
              "k" := mean t
                mean takes integers or floats, not ext[ymir.epoch, float64].
            |});
      test "min of a record holding an extension is a problem" (fun () ->
          expect
            (problems
               (derive Expr.[ "k" := over (min (Col.v Record.kind "r")) ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := over (min r)
                min orders values, and record[e ext[ymir.epoch, float64]] holds an extension type and has no order.
            |});
      test "over's order on an extension or on a column twice is a problem"
        (fun () ->
          expect
            (problems
               (derive Expr.[ "k" := over ~order:[ Order.asc "t" ] (rank n) ])
            ^ "\n"
            ^ problems
                (derive
                   Expr.
                     [ "k" := over ~order:Order.[ asc "n"; desc "n" ] (rank n) ])
            )
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := over ~order:[asc "t"] (rank n)
                "t" is ext[ymir.epoch, float64], which orders only through its declaration.
            derive: 1 problem
              "k" := over ~order:[asc "n"; desc "n"] (rank n)
                the column "n" is already a key.
            |});
      test "over's keys report a repeat and a missing column" (fun () ->
          expect
            (problems
               (derive
                  Expr.[ "k" := over ~by:[ "g"; "nope"; "g"; "nope" ] (sum n) ]))
          @@ __POS_OF__
               {|
            derive: 3 problems
              "k" := over ~by:["g"; "nope"; "g"; "nope"] (sum n)
                over ~by: "g" is named twice.
                no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
                over ~by: "nope" is named twice.
            |});
      test "nx refuses a function that moves or ignores its argument" (fun () ->
          expect
            (problems (derive Expr.[ "k" := nx { f = (fun a -> Nx.sum a) } f ])
            ^ "\n"
            ^ problems (derive Expr.[ "k" := nx { f = Nx.zeros_like } f ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := nx <fn> f
                the function of nx performs Nx.sum, which is not elementwise.
            derive: 1 problem
              "k" := nx <fn> f
                the function of nx ignores its argument.
            |});
      test "nx refuses an operand with no nx dtype" (fun () ->
          expect
            (problems
               (derive Expr.[ "k" := nx { f = Nx.exp } (Col.span "wait") ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := nx <fn> wait
                nx takes integer, float and boolean operands, not duration[ms].
            |});
      test "one output reports each of its problems" (fun () ->
          expect
            (problems
               (derive Expr.[ "k" := if_ (Col.int "nope" > int 1) x8 u8 ]))
          @@ __POS_OF__
               {|
            derive: 2 problems
              "k" := if_ (nope > 1) x8 u8
                no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
                int8 and uint8 do not meet: cast first.
            |});
      test "a field a calendar value lacks is a problem" (fun () ->
          expect
            (problems
               (derive Expr.[ "k" := Temporal.field `Hour (Col.date "d") ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := Temporal.field `Hour d
                a date has no `Hour.
            |});
      test "a calendar of an instant is a problem" (fun () ->
          expect
            (problems
               (derive Expr.[ "k" := Temporal.field `Hour (Col.instant "ts") ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := Temporal.field `Hour ts
                Temporal.field reads wall clocks, and datetime[ns, UTC] is instants: name a zone with Temporal.wall.
            |});
      test "a cast from a date to text is a problem" (fun () ->
          expect
            (problems (derive Expr.[ "k" := cast Type.string (Col.date "d") ]))
          @@ __POS_OF__
               {|
            derive: 1 problem
              "k" := cast string d
                cast does not convert date to string: use Temporal.format.
            |});
      test "a value outside a dictionary is a problem each time" (fun () ->
          let q =
            Query.of_source
              (source "cats" Type.[ ("c", Any (categorical [| "a"; "b" |])) ])
          in
          let held () =
            Query.filter Expr.(is_in [ String.make 3 'z' ] (Col.string "c")) q
          in
          expect (problems held ^ "\n" ^ problems held)
          @@ __POS_OF__
               {|
            filter: 1 problem
              is_in […] c
                categorical["a", "b"] does not hold the value "zzz".
            filter: 1 problem
              is_in […] c
                categorical["a", "b"] does not hold the value "zzz".
            |});
      test
        "a query built again allocates the same words, however many equal \
         queries are live" (fun () ->
          (* Every query built here stays live to the end of the test, so no
             collection changes which expressions are live while it measures. *)
          let live = ref [ late_by_carrier () ] in
          let build () =
            let w, q = words late_by_carrier in
            live := q :: !live;
            w
          in
          let first = build () in
          let counts = List.init 64 (fun _ -> build ()) in
          equal (list int) (List.init 64 (fun _ -> first)) counts;
          ignore (Sys.opaque_identity !live));
    ]

(* Query: verbs *)

let verbs =
  group "Query verbs"
    [
      test "schema resolves the guide's pipeline" (fun () ->
          equal schema_w
            (Schema.v
               Type.
                 [
                   ("carrier", Any string);
                   ("mean_delay", Any float64);
                   ("flights", Any int64);
                   ("name", Any string);
                 ])
            (Query.schema (late_by_carrier ())));
      test "select of an extension output has its extension type" (fun () ->
          equal schema_w
            (Schema.v Type.[ ("t2", Any epoch_t) ])
            (Query.schema
               (Query.select Expr.[ "t2" := Ext.col epoch "t" ] of_kinds)));
      test "select refuses two outputs of one name" (fun () ->
          expect
            (problems (fun () ->
                 Query.select Expr.[ "a" := f; "a" := f +. float 1. ] of_kinds)
            ^ "\n"
            ^ problems (fun () ->
                Query.select Expr.[ "a" := Col.float "nope"; "a" := f ] of_kinds)
            )
          @@ __POS_OF__
               {|
            select: 1 problem
              the output "a" appears twice.
            select: 2 problems
              "a" := nope
                no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
              the output "a" appears twice.
            |});
      test "an output name that is not UTF-8 raises" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Expr.( := )") (fun () ->
              Expr.("\xff" := f)));
      test "derive refuses two outputs of one name" (fun () ->
          expect
            (problems (fun () ->
                 Query.derive
                   Expr.[ "f" := f; keep Sel.(names [ "f" ]) ]
                   of_kinds))
          @@ __POS_OF__
               {|
            derive: 1 problem
              the output "f" appears twice.
            |});
      test "filter takes a predicate that is an OCaml value" (fun () ->
          let p = Expr.(const (fun x -> Stdlib.( > ) x 1.) $ f) in
          equal schema_w (Query.schema of_kinds)
            (Query.schema (Query.filter p of_kinds)));
      test "filter refuses an extension's values, even of bool" (fun () ->
          let flag =
            Ext.v ~name:"flag" ~ordered:false Type.bool ~dec:Fun.id ~enc:Fun.id
          in
          let q =
            Query.of_source
              (source "flags" Type.[ ("fl", Any (ext ~name:"flag" bool)) ])
          in
          expect (problems (fun () -> Query.filter (Ext.col flag "fl") q))
          @@ __POS_OF__
               {|
            filter: 1 problem
              fl
                fl is ext[flag, bool], where bool is expected.
            |});
      test "filter reports a predicate's problems" (fun () ->
          expect
            (problems (fun () ->
                 Query.filter Expr.(Col.int "f" > int 1) of_kinds)
            ^ "\n"
            ^ problems (fun () ->
                Query.filter Expr.(f > float 1. && Col.bool "nope") of_kinds))
          @@ __POS_OF__
               {|
            filter: 1 problem
              f > 1
                Col.int reads int8 to int64 and uint8 to uint64, but "f" is float64.
            filter: 1 problem
              f > 1. && nope
                no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
            |});
      test
        "sort refuses an extension key, one holding an extension, a missing \
         and a repeated key" (fun () ->
          expect
            (problems (fun () ->
                 Query.sort
                   Order.
                     [
                       asc "t";
                       desc "r";
                       asc "nope";
                       asc "f";
                       nulls_first (desc "f");
                     ]
                   of_kinds))
          @@ __POS_OF__
               {|
            sort: 4 problems
              asc "t": "t" is ext[ymir.epoch, float64], which orders only through its declaration.
              desc "r": "r" is record[e ext[ymir.epoch, float64]], which holds an extension type and has no order.
              asc "nope": no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
              nulls_first (desc "f"): the column "f" is already a key.
            |});
      test "slice refuses a negative length" (fun () ->
          expect
            (problems (fun () -> Query.slice ~offset:0 ~length:(-1) of_kinds))
          @@ __POS_OF__
               {|
            slice: 1 problem
              the length -1 is negative.
            |});
      test
        "aggregate refuses a key twice, a missing key and an output of a key's \
         name" (fun () ->
          let agg by os () = Query.aggregate ~by os of_kinds in
          expect
            (String.concat "\n"
               (List.map problems
                  [
                    agg [ "g"; "n"; "g" ] Expr.[ "n" := sum f ];
                    agg [ "g" ] Expr.[ "g" := sum (Col.float "nope") ];
                    agg [ "gg"; "gg" ] Expr.[ "k" := rows ];
                  ]))
          @@ __POS_OF__
               {|
            aggregate: 2 problems
              ~by: "g" is named twice.
              the output "n" has the name of a key.
            aggregate: 2 problems
              "g" := sum nope
                no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
              the output "g" has the name of a key.
            aggregate: 2 problems
              ~by: no column "gg". Did you mean "g"?
              ~by: "gg" is named twice.
            |});
      test "append meets each column's types" (fun () ->
          (* Query.append: an int32 and an int64 column append as int64, a
             categorical and a string as string; Type.common for the rest. *)
          let appended t0 t1 =
            let side t = Query.of_source (source "s" [ ("x", t) ]) in
            Query.schema (Query.append (side t1) (side t0))
          in
          let cat = Type.Any (Type.categorical [| "a" |]) in
          List.iter
            (fun (t0, t1, t) ->
              equal schema_w (Schema.v [ ("x", t) ]) (appended t0 t1))
            Type.
              [
                (Any int32, Any int64, Any int64);
                (Any uint8, Any int16, Any int16);
                (cat, Any string, Any string);
                (Any string, cat, Any string);
              ]);
      test "append refuses columns only one side has and types that do not meet"
        (fun () ->
          let rest =
            let column = function
              | "x8", _ -> Some ("x8", Type.Any Type.uint8)
              | "wait", _ -> None
              | c -> Some c
            in
            source "rest"
              (List.filter_map column (Schema.columns (Query.schema of_kinds))
              @ [ ("extra", Type.Any Type.bool) ])
          in
          expect
            (problems (fun () -> Query.append (Query.of_source rest) of_kinds))
          @@ __POS_OF__
               {|
            append: 3 problems
              "x8" is int8 in the input and uint8 in rest, which do not meet: cast one first.
              "wait" (duration[ms]) is not in rest.
              "extra" (bool) is only in rest.
            |});
      test "unnest places an output as derive does, holding the element"
        (fun () ->
          let q =
            Query.of_source
              (source "lists"
                 Type.[ ("id", Any int64); ("xs", Any (list int64)) ])
          in
          let xs = Col.v Kind.(list int) "xs" in
          equal schema_w
            (Schema.v
               Type.[ ("id", Any int64); ("xs", Any int64); ("e", Any int64) ])
            (Query.schema (Query.unnest Expr.[ "xs" := xs; "e" := xs ] q)));
    ]

(* Query: schemas drawn *)

let ext_types = Type.[ Any epoch_t; Any (record [ ("e", Any epoch_t) ]) ]

let types =
  Type.
    [
      Any int8;
      Any int16;
      Any int64;
      Any uint8;
      Any float32;
      Any float64;
      Any string;
      Any bool;
      Any date;
      Any (list int64);
      Any (list string);
    ]
  @ ext_types

let pp_any ppf (Type.Any t) = Type.pp ppf t

(* [columns_gen names] draws columns with distinct names of [names], in any
   order, of any type of [types]. *)
let columns_gen names =
  let open Gen in
  let* names = subsequence names in
  let* names = permutation names in
  let+ tys =
    list ~size:(constant (List.length names)) (of_list ~pp:pp_any types)
  in
  List.combine names tys

let schema_gen =
  Gen.with_pp Schema.pp
    (Gen.map Schema.v (columns_gen [ "a"; "b"; "c"; "d"; "e"; "f" ]))

let of_schema s =
  Query.of_source (Source.v ~name:"s" ~schema:s (fun _ -> Ok []))

let subset_gen =
  Gen.(
    let* s = schema_gen in
    let* ns = permutation (List.map fst (Schema.columns s)) in
    let+ ns = subsequence ns in
    (s, ns))

let keeps s =
  let q = of_schema s in
  let ordered (Type.Any t) =
    not (List.exists (fun (Type.Any e) -> Type.equal e t) ext_types)
  in
  let sortable =
    List.filter_map
      (fun (n, t) -> if ordered t then Some (Order.asc n) else None)
      (Schema.columns s)
  in
  List.iter
    (fun q' -> equal schema_w s (Query.schema q'))
    Query.
      [
        select Expr.[ keep Sel.all ] q;
        derive Expr.[ keep Sel.all ] q;
        filter Expr.(bool true) q;
        sort sortable q;
        slice ~offset:(-3) ~length:2 q;
        append q q;
      ]

let derive_one s =
  let replaced = Option.is_some (Schema.find s "c") in
  cover "replaces a column" replaced;
  cover "adds a column" (not replaced);
  let expected =
    if replaced then
      List.map
        (fun (n, t) ->
          if String.equal n "c" then (n, Type.Any Type.int64) else (n, t))
        (Schema.columns s)
    else Schema.columns s @ [ ("c", Type.Any Type.int64) ]
  in
  equal schema_w (Schema.v expected)
    (Query.schema (Query.derive Expr.[ "c" := int 1 ] (of_schema s)))

let aggregate_by (s, by) =
  let key n = (n, Option.get (Schema.find s n)) in
  equal schema_w
    (Schema.v (List.map key by @ [ ("rows_", Type.Any Type.int64) ]))
    (Query.schema (Query.aggregate ~by Expr.[ "rows_" := rows ] (of_schema s)))

let join_gen =
  Gen.(
    let+ left = columns_gen [ "a"; "b"; "c" ]
    and+ right = columns_gen [ "x"; "y"; "z" ] in
    ( Schema.v (left @ [ ("k", Type.Any Type.int8) ]),
      Schema.v (("k", Type.Any Type.int16) :: right) ))
  |> Gen.with_pp (fun ppf (l, r) ->
      Format.fprintf ppf "left: %a; right: %a" Schema.pp l Schema.pp r)

let joined (l, r) =
  let join kind =
    Query.schema
      (Query.join ~kind ~on:(Join.keys [ "k" ]) (of_schema r) (of_schema l))
  in
  let rights = List.tl (Schema.columns r) in
  equal schema_w l (join Join.Semi);
  equal schema_w l (join Join.Anti);
  equal schema_w (Schema.v (Schema.columns l @ rights)) (join Join.Inner);
  equal schema_w (Schema.v (Schema.columns l @ rights)) (join Join.Left);
  let full =
    List.map
      (fun (n, t) ->
        if String.equal n "k" then (n, Type.Any Type.int16) else (n, t))
      (Schema.columns l)
  in
  equal schema_w (Schema.v (full @ rights)) (join Join.Full)

let resolution =
  group "Query schemas"
    [
      prop "keep all, filter, sort, slice and append keep the schema" schema_gen
        keeps;
      prop "derive replaces a column in place or appends it" schema_gen
        derive_one;
      prop "select has its outputs' names, in order" subset_gen (fun (s, ns) ->
          equal (list string) ns
            (names (Query.select Expr.[ keep Sel.(names ns) ] (of_schema s))));
      prop "aggregate has its keys, then its outputs" subset_gen aggregate_by;
      prop "a join has the left columns, then the right ones but the keys"
        join_gen joined;
    ]

(* Query.join *)

let other =
  Query.of_source
    (source "other"
       Type.
         [
           ("y8", Any int8);
           ("g2", Any string);
           ("t2", Any epoch_t);
           ("f2", Any float64);
           ("t3", Any (ext ~name:"ymir.epoch" ~metadata:"v2" float64));
           ("r2", Any (record [ ("e", Any epoch_t) ]));
         ])

let shared =
  Query.of_source
    (source "shared"
       Type.[ ("y8", Any int8); ("f", Any float64); ("n", Any int64) ])

let joined_on ?kind ?(right = other) on () = Query.join ?kind ~on right of_kinds

let joins =
  group "Query.join"
    [
      test "reports a column missing on each side" (fun () ->
          expect (problems (joined_on (Join.keys [ "nope" ])))
          @@ __POS_OF__
               {|
            join: 2 problems
              left: no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
              right: no column "nope". The columns are "y8", "g2", "t2", "f2", "t3" and "r2".
            |});
      test "reports an atom's columns that do not meet" (fun () ->
          expect
            (String.concat "\n"
               (List.map
                  (fun c -> problems (joined_on c))
                  Join.
                    [
                      eq "g" "y8";
                      eq "t" "t3";
                      eq "f" "t2";
                      ge "g" "y8";
                      inside "f" ("y8", "f2");
                      overlaps ("x8", "f") ("y8", "g2");
                    ]))
          @@ __POS_OF__
               {|
            join: 1 problem
              "g" is string and "y8" is int8, which do not meet: cast one first.
            join: 1 problem
              "t" is ext[ymir.epoch, float64] and "t3" is ext[ymir.epoch "v2", float64], which do not meet: cast one first.
            join: 1 problem
              "f" is float64 and "t2" is ext[ymir.epoch, float64], which do not meet: cast one first.
            join: 1 problem
              "g" is string and "y8" is int8, which do not meet: cast one first.
            join: 1 problem
              "f" is float64 and "y8" is int8, which do not meet: cast one first.
            join: 1 problem
              "x8" is int8, "f" is float64, "y8" is int8 and "g2" is string, which do not meet: cast one first.
            |});
      test "reports a comparison on a type without an order" (fun () ->
          expect
            (String.concat "\n"
               (List.map
                  (fun c -> problems (joined_on c))
                  Join.
                    [
                      le "t" "t2";
                      gt "f" "r2";
                      overlaps ("f", "f32") ("f2", "t2");
                    ]))
          @@ __POS_OF__
               {|
            join: 1 problem
              "t" is ext[ymir.epoch, float64], which has no order of its own.
            join: 1 problem
              "r2" is record[e ext[ymir.epoch, float64]], which has no order of its own.
            join: 1 problem
              "t2" is ext[ymir.epoch, float64], which has no order of its own.
            |});
      test "reports a Full join's left column of two equality atoms" (fun () ->
          expect
            (problems
               (joined_on ~kind:Full Join.(eq "x8" "y8" && eq "x8" "f2")))
          @@ __POS_OF__
               {|
            join: 2 problems
              "x8" is int8 and "f2" is float64, which do not meet: cast one first.
              "x8" is the left of two equality atoms, so a Full join cannot coalesce it.
            |});
      test "reports names on both sides" (fun () ->
          expect
            (problems (joined_on ~right:shared Join.(eq "x8" "y8"))
            ^ "\n"
            ^ problems (joined_on ~right:shared Join.(lt "f" "f")))
          @@ __POS_OF__
               {|
            join: 1 problem
              "f", "n" are on both sides: rename one side first.
            join: 1 problem
              "f", "n" are on both sides: rename one side first.
            |});
      test "a Semi join takes right columns of left names" (fun () ->
          equal schema_w (Query.schema of_kinds)
            (Query.schema
               (Query.join ~kind:Semi ~on:Join.(eq "x8" "y8") shared of_kinds)));
    ]

(* Query.equal *)

(* [variants] pairs queries that differ in one argument of one verb. *)
let variants =
  let q = Query.of_source flights and c = Query.of_source carriers in
  let c' = Query.of_source (source "c'" (Schema.columns (Query.schema c))) in
  let semi ?(kind = Join.Semi) ?(each_left = Join.Any) ?(each_right = Join.Any)
      right () =
    Query.join ~kind ~each_left ~each_right ~on:(Join.keys [ "carrier" ]) right
      q
  in
  Query.
    [
      ( "output names",
        (fun () -> select Expr.[ "a" := delay ] q),
        fun () -> select Expr.[ "b" := delay ] q );
      ( "output expressions",
        (fun () -> derive Expr.[ "a" := delay ] q),
        fun () -> derive Expr.[ "a" := delay +. float 1. ] q );
      ( "select and derive",
        (fun () -> select Expr.[ "a" := delay ] q),
        fun () -> derive Expr.[ "a" := delay ] q );
      ( "filter literals",
        (fun () -> filter Expr.(delay > float 15.) q),
        fun () -> filter Expr.(delay > float 16.) q );
      ( "sort keys",
        (fun () -> sort [ Order.asc "year" ] q),
        fun () -> sort [ Order.desc "year" ] q );
      ( "sort inputs",
        (fun () -> sort [ Order.asc "carrier" ] c),
        fun () -> sort [ Order.asc "carrier" ] c' );
      ( "slice offsets",
        (fun () -> slice ~offset:0 ~length:1 q),
        fun () -> slice ~offset:1 ~length:1 q );
      ( "slice lengths",
        (fun () -> slice ~offset:0 ~length:0 q),
        fun () -> slice ~offset:0 ~length:1 q );
      ("verbs", (fun () -> slice ~offset:0 ~length:1 q), fun () -> sort [] q);
      ( "aggregate keys",
        (fun () -> aggregate ~by:[ "carrier" ] Expr.[ "n" := rows ] q),
        fun () -> aggregate ~by:[ "origin" ] Expr.[ "n" := rows ] q );
      ( "aggregate outputs",
        (fun () -> aggregate ~by:[ "carrier" ] Expr.[ "n" := rows ] q),
        fun () -> aggregate ~by:[ "carrier" ] Expr.[ "m" := rows ] q );
      ("sources", (fun () -> q), fun () -> c);
      ("join right inputs", semi c, semi c');
      ("join kinds", semi c, semi ~kind:Anti c);
      ("join left counts", semi c, semi ~each_left:One c);
      ("join right counts", semi c, semi ~each_right:One c);
      ( "join conditions",
        semi c,
        fun () -> join ~kind:Semi ~on:(Join.eq "carrier" "name") c q );
      ("append rests", (fun () -> append c c), fun () -> append c' c);
      ("append inputs", (fun () -> append c c), fun () -> append c c');
    ]

(* [expression_variants] pairs expressions that differ in one attribute: a wrong
   merge prints the same but would let the optimizer share one plan for the
   other. *)
let expression_variants =
  let q =
    Query.of_source
      (source "e"
         Type.
           [
             ("x", Any int64);
             ("y", Any int64);
             ("f", Any float64);
             ("g", Any float64);
             ("b", Any bool);
             ("s", Any string);
             ("d", Any date);
             ("ts", Any (instant Ns));
           ])
  in
  let row what e0 e1 =
    ( what,
      (fun () -> Query.derive Expr.[ "a" := e0 () ] q),
      fun () -> Query.derive Expr.[ "a" := e1 () ] q )
  and agg what e0 e1 =
    ( what,
      (fun () -> Query.aggregate ~by:[] Expr.[ "a" := e0 () ] q),
      fun () -> Query.aggregate ~by:[] Expr.[ "a" := e1 () ] q )
  in
  let x = Col.int "x" and y = Col.int "y" and f = Col.float "f" in
  let g = Col.float "g" and b = Col.bool "b" and s = Col.string "s" in
  Expr.
    [
      row "+ and -" (fun () -> x + y) (fun () -> x - y);
      row "+. and *." (fun () -> f +. g) (fun () -> f *. g);
      row "< and <=" (fun () -> x < y) (fun () -> x <= y);
      row "&& and ||" (fun () -> b && x < y) (fun () -> b || x < y);
      agg "min and max" (fun () -> min x) (fun () -> max x);
      agg "arg_min and arg_max" (fun () -> arg_min x) (fun () -> arg_max x);
      agg "first and last" (fun () -> first x) (fun () -> last x);
      agg "quantiles" (fun () -> quantile 0.25 x) (fun () -> quantile 0.75 x);
      row "shifts" (fun () -> shift 1 x) (fun () -> shift 2 x);
      row "slice offsets"
        (fun () -> Str.slice ~offset:0 ~length:2 s)
        (fun () -> Str.slice ~offset:1 ~length:2 s);
      row "prefix and suffix"
        (fun () -> Str.matches (Str.prefix "a") s)
        (fun () -> Str.matches (Str.suffix "a") s);
      row "cast types"
        (fun () -> cast Type.int32 x)
        (fun () -> cast Type.int16 x);
      row "store types"
        (fun () -> store Type.int64 (int 3))
        (fun () -> store Type.int32 (int 3));
      row "is_in values"
        (fun () -> is_in [ 1; 2 ] x)
        (fun () -> is_in [ 1; 3 ] x);
      row "over keys"
        (fun () -> over ~by:[ "x" ] (sum y))
        (fun () -> over ~by:[ "y" ] (sum y));
      row "over orders"
        (fun () -> over ~order:[ Order.asc "x" ] (sum y))
        (fun () -> over ~order:[ Order.desc "x" ] (sum y));
      row "nx functions"
        (fun () -> nx { f = Nx.exp } f)
        (fun () -> nx { f = Nx.neg } f);
      row "format strings"
        (fun () -> Temporal.format "%Y" (Col.instant "ts"))
        (fun () -> Temporal.format "%m" (Col.instant "ts"));
      row "floor steps"
        (fun () -> Temporal.floor (Time.Days 1) (Col.date "d"))
        (fun () -> Temporal.floor (Time.Days 2) (Col.date "d"));
    ]

(* [built_twice] draws two queries of [variants], each built afresh: the same
   builder half the time, so that the two are equal values that are not one
   value. *)
let built_twice =
  let builders =
    Array.of_list (List.concat_map (fun (_, q0, q1) -> [ q0; q1 ]) variants)
  in
  let last = Array.length builders - 1 in
  Gen.(
    let* i = int_range 0 last in
    let+ j = frequency [ (1, constant i); (1, int_range 0 last) ] in
    (i, j))
  |> Gen.with_pp (fun ppf (i, j) ->
      Format.fprintf ppf "%a@.and %a" Query.pp
        (builders.(i) ())
        Query.pp
        (builders.(j) ()))
  |> Gen.map (fun (i, j) -> (i = j, builders.(i) (), builders.(j) ()))

let floats xs = Query.of_table (Talon.v [ ("x", Column.v Type.float64 xs) ])

let equality =
  group "Query.equal"
    [
      prop "is an equivalence, true of a query built twice" built_twice
        (fun (twice, q0, q1) ->
          cover "built twice" twice;
          Law.equivalence query_w (q0, q1);
          if twice then equal query_w q0 q1);
      prop "equal queries have equal schemas" built_twice (fun (_, q0, q1) ->
          let same = Query.equal q0 q1 in
          cover "equal" same;
          if same then equal schema_w (Query.schema q0) (Query.schema q1));
      cases "tells apart"
        ~id:(fun (what, _, _) -> what)
        variants
        (fun (_, q0, q1) ->
          equal query_w (q0 ()) (q0 ());
          not_equal query_w (q0 ()) (q1 ()));
      cases "tells apart expressions by"
        ~id:(fun (what, _, _) -> what)
        expression_variants
        (fun (_, q0, q1) ->
          equal query_w (q0 ()) (q0 ());
          equal query_w (q1 ()) (q1 ());
          not_equal query_w (q0 ()) (q1 ()));
      test "compares tables by key identity row by row" (fun () ->
          equal query_w (floats [| Float.nan |]) (floats [| Float.nan |]);
          equal query_w (floats [| -0. |]) (floats [| 0. |]);
          not_equal query_w (floats [| 1. |]) (floats [| 2. |]);
          not_equal query_w
            (Query.of_table (Talon.v [ ("x", Column.v Type.int32 [| 1 |]) ]))
            (Query.of_table (Talon.v [ ("x", Column.v Type.int64 [| 1 |]) ])));
      test "over a doubling takes work linear in its depth" (fun () ->
          (* Each level reaches the step below it twice: compared once per path,
             16 levels cost 256 times 8 levels' work; once per pair, twice. *)
          let compared d =
            let s = source "part" Type.[ ("a", Any int64) ] in
            let rec double q d =
              if d = 0 then q else double (Query.append q q) (d - 1)
            in
            let q0 = double (Query.of_source s) d
            and q1 = double (Query.of_source s) d in
            fst (words (fun () -> Query.equal q0 q1))
          in
          less int ~than:(3 * compared 8) (compared 16));
    ]

(* Query.pp *)

let small =
  Query.of_source
    (source "small"
       Type.
         [
           ("f", Any float64);
           ("n", Any int64);
           ("g", Any string);
           ("ts", Any (instant Ns));
           ("l", Any (list int64));
         ])

let one_row =
  Query.of_source
    (source ~rows:1 "other"
       Type.[ ("y", Any float64); ("ys", Any (instant Ns)) ])

let printing =
  group "Query.pp"
    [
      test "prints the guide's pipeline" (fun () ->
          (* Query.pp's example. *)
          equal text
            {|query → carrier string, mean_delay float64, flights int64, name string
sort [desc "mean_delay"]
└ join ~on:(keys ["carrier"]) ~each_left:One
  ├ aggregate ~by:["carrier"] ["mean_delay" := mean dep_delay;
  │                            "flights" := rows]
  │ └ filter (dep_delay > 15.)
  │   └ csv "flights.csv" (19 columns)
  └ parquet "carriers.parquet" (2 columns)|}
            (str Query.pp (late_by_carrier ())));
      test "prints a run of kept columns as one keep" (fun () ->
          expect
            (str Query.pp
               (Query.select
                  Expr.
                    [
                      keep Sel.(names [ "f"; "n" ]);
                      "z" := Col.float "f";
                      keep Sel.(names [ "g" ]);
                    ]
                  small))
          @@ __POS_OF__
               {|
            query → f float64, n int64, z float64, g string
            select [keep (names ["f"; "n"]); "z" := f; keep (names ["g"])]
            └ small (5 columns)
            |});
      test "prints outputs with their selectors resolved" (fun () ->
          expect
            (str Query.pp
               (Query.select
                  Expr.
                    [
                      each
                        Sel.(names [ "f" ])
                        { column = (fun _ x -> "z" := x) };
                    ]
                  small))
          @@ __POS_OF__
               {|
            query → z float64
            select ["z" := f]
            └ small (5 columns)
            |});
      test "prints an nx lift by its function" (fun () ->
          expect
            (str Query.pp
               (Query.derive
                  Expr.[ "k" := nx { f = Nx.exp } (Col.float "f") ]
                  small))
          @@ __POS_OF__
               {|
            query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64], k float64
            derive ["k" := exp f]
            └ small (5 columns)
            |});
      test "prints keys as written inside Order.( … )" (fun () ->
          expect
            (str Query.pp
               (Query.sort Order.[ asc "f"; nulls_first (desc "n") ] small))
          @@ __POS_OF__
               {|
            query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64]
            sort [asc "f"; nulls_first (desc "n")]
            └ small (5 columns)
            |});
      test "prints a slice's offset and length" (fun () ->
          expect (str Query.pp (Query.slice ~offset:(-10) ~length:5 small))
          @@ __POS_OF__
               {|
            query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64]
            slice ~offset:(-10) ~length:5
            └ small (5 columns)
            |});
      test "prints a join's options after ~on, when not their defaults"
        (fun () ->
          expect
            (str Query.pp
               (Query.join ~kind:Left ~each_left:At_most_one
                  ~each_right:At_least_one ~on:Join.all one_row small)
            ^ "\n"
            ^ str Query.pp (Query.join ~on:Join.position one_row small))
          @@ __POS_OF__
               {|
            query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64], y float64, ys datetime[ns, UTC]
            join ~on:all ~kind:Left ~each_left:At_most_one ~each_right:At_least_one
            ├ small (5 columns)
            └ other (2 columns, 1 row)
            query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64], y float64, ys datetime[ns, UTC]
            join ~on:position
            ├ small (5 columns)
            └ other (2 columns, 1 row)
            |});
      test "prints a table by its columns and rows" (fun () ->
          let table rows =
            Query.of_table
              (Talon.v
                 [
                   ("a", Column.v Type.int64 (Array.make rows 1));
                   ("b", Column.v Type.bool (Array.make rows true));
                 ])
          in
          equal text "query → a int64, b bool\ntable (2 columns, 3 rows)"
            (str Query.pp (table 3)));
      test "labels steps reached twice #1, #2 in the order first met" (fun () ->
          let sorted = Query.sort [ Order.asc "f" ] small in
          expect
            (str Query.pp (Query.append sorted small |> Query.append sorted))
          @@ __POS_OF__
               {|
            query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64]
            append
            ├ append
            │ ├ #1 small (5 columns)
            │ └ #2 sort [asc "f"]
            │   └ #1
            └ #2
            |});
      test "wraps a long step under its prefix, and not the schema line"
        (fun () ->
          expect
            (str Query.pp
               Query.(
                 of_source flights
                 |> join ~on:(Join.keys [ "carrier" ]) (of_source carriers)
                 |> derive
                      Expr.
                        [
                          "speed" :=
                            Col.float "distance" /. Col.float "air_time"
                            *. float 60.;
                          "gain" :=
                            Col.float "dep_delay" -. Col.float "arr_delay";
                          "late" := Col.float "arr_delay" > float 15.;
                        ]))
          @@ __POS_OF__
               {|
            query → year int16, month int8, day int8, dep_time int32, sched_dep_time int32, dep_delay float64, arr_time int32, sched_arr_time int32, arr_delay float64, carrier string, flight int32, tailnum string, origin string, dest string, air_time float64, distance float64, hour int8, minute int8, time_hour datetime[us, UTC], name string, speed float64, gain float64, late bool
            derive ["speed" := distance /. air_time *. 60.;
                    "gain" := dep_delay -. arr_delay; "late" := arr_delay > 15.]
            └ join ~on:(keys ["carrier"])
              ├ csv "flights.csv" (19 columns)
              └ parquet "carriers.parquet" (2 columns)
            |});
      test "prints names in UTF-8 as they read" (fun () ->
          let q =
            Query.of_source (source "données" Type.[ ("é", Any int64) ])
          in
          expect
            (str Query.pp
               Query.(
                 q
                 |> derive Expr.[ "délai" := Col.int "é" + int 1 ]
                 |> sort [ Order.asc "délai" ]))
          @@ __POS_OF__
               {|
            query → é int64, délai int64
            sort [asc "délai"]
            └ derive ["délai" := "é" + 1]
              └ données (1 column)
            |});
      test "prints no schema after the arrow of an empty one" (fun () ->
          equal text "query →\nselect []\n└ s (1 column)"
            (str Query.pp
               (Query.select []
                  (Query.of_source (source "s" Type.[ ("a", Any int64) ])))));
    ]

let () =
  exit
    (run "query"
       [
         sel;
         source_v;
         conditions;
         reports;
         binding;
         verbs;
         joins;
         resolution;
         equality;
         printing;
       ])
