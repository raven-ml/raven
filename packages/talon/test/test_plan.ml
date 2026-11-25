(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let str pp v = Format.asprintf "%a" pp v

(* [message f] is the message of the [Invalid_argument] that [f] raises. *)
let message f =
  match f () with _ -> "no exception" | exception Invalid_argument m -> m

let schema_w = Testable.make ~pp:Schema.pp ~equal:Schema.equal
let query_w = Testable.make ~pp:Query.pp ~equal:Query.equal

let source ?rows ?pushdown name columns =
  Source.v ~name ~schema:(Schema.v columns) ?rows ?pushdown (fun _ -> Ok [])

(* [optimized q] is the plan that running [q] runs, printed: a source's request
   shows only there. *)
let optimized q = str Query.pp (Query.optimize q)

(* [same_plan q0 q1] states that [q0] and [q1] optimize to one plan. *)
let same_plan q0 q1 = equal query_w (Query.optimize q0) (Query.optimize q1)

(* [stays p q] states that the filter [p] over [q] stays where it is written,
   over [q]'s own plan. *)
let stays p q =
  equal query_w
    (Query.filter p (Query.optimize q))
    (Query.optimize (Query.filter p q))

(* Answers *)

let exact _ = Source.Exact
let inexact _ = Source.Inexact

let rec pred_column : Source.Pred.t -> string option = function
  | Cmp (c, _, _) | In (c, _) | Null c | Valid c -> Some c
  | Not p -> pred_column p
  | And _ | Or _ -> None

(* [by_column answers p] answers for [p] what [answers] gives its column. *)
let by_column answers p =
  match Option.bind (pred_column p) (fun c -> List.assoc_opt c answers) with
  | Some a -> a
  | None -> Source.Unsupported

(* Fixtures *)

let columns =
  Type.
    [
      ("a", Any int64);
      ("b", Any float64);
      ("c", Any string);
      ("k", Any int8);
      ("l", Any (list int64));
    ]

let s_unsupported = source "u" columns
let s_exact = source ~pushdown:exact "e" columns
let s_inexact = source ~pushdown:inexact "i" columns
let s_rows = source ~rows:100 ~pushdown:exact "r" columns
let other = source "other" Type.[ ("k", Any int8); ("w", Any float64) ]
let a = Col.int "a"
let b = Col.float "b"
let c = Col.string "c"
let k = Col.int "k"
let w = Col.float "w"
let of_source = Query.of_source
let src = of_source s_unsupported
let positive x = x > 0

let flights =
  source {|csv "flights.csv"|}
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

let carriers =
  source {|parquet "carriers.parquet"|}
    Type.[ ("carrier", Any string); ("name", Any string) ]

let late_by_carrier () =
  let delay = Col.float "dep_delay" in
  Query.(
    of_source flights
    |> filter Expr.(delay > float 15.)
    |> aggregate ~by:[ "carrier" ]
         Expr.[ "mean_delay" := mean delay; "flights" := rows ]
    |> join ~on:(Join.keys [ "carrier" ]) ~each_left:One (of_source carriers)
    |> sort [ Order.desc "mean_delay" ])

let epoch =
  Ext.v ~name:"ymir.epoch" ~ordered:true Type.float64 ~dec:Fun.id ~enc:(fun x ->
      x *. 2.)

(* Source pushdown *)

let filtered s =
  Query.(
    of_source s
    |> filter Expr.(a > int 1 && b < float 2. && c = string "x")
    |> Kit.head 10)

let pushdown =
  group "Source pushdown"
    [
      test "the guide's pipeline reads two columns" (fun () ->
          (* Query.optimize's example. *)
          equal text
            {|query → carrier string, mean_delay float64, flights int64, name string
sort [desc "mean_delay"]
└ join ~on:(keys ["carrier"]) ~each_left:One
  ├ aggregate ~by:["carrier"] ["mean_delay" := mean dep_delay;
  │                            "flights" := rows]
  │ └ filter (dep_delay > 15.)
  │   └ csv "flights.csv" (19 columns) ~columns:["dep_delay"; "carrier"]
  └ parquet "carriers.parquet" (2 columns)|}
            (optimized (late_by_carrier ())));
      test "Unsupported keeps every conjunct" (fun () ->
          expect (optimized (filtered s_unsupported))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64]
            slice ~offset:0 ~length:10
            └ filter (a > 1 && b < 2. && c = "x")
              └ u (5 columns)
            |});
      test "Exact hands the source every conjunct, and the limit" (fun () ->
          expect (optimized (filtered s_exact))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64]
            slice ~offset:0 ~length:10
            └ e (5 columns) ~filters:[a > 1; b < 2.; c = "x"] ~limit:10
            |});
      test "Inexact hands the source every conjunct and keeps it" (fun () ->
          expect (optimized (filtered s_inexact))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64]
            slice ~offset:0 ~length:10
            └ filter (a > 1 && b < 2. && c = "x")
              └ i (5 columns) ~filters:[a > 1; b < 2.; c = "x"]
            |});
      test "a source is handed its Exact and Inexact conjuncts, in plan order"
        (fun () ->
          let answering answers =
            source ~pushdown:(by_column answers) "m" columns
          in
          expect
            (optimized (filtered (answering [ ("a", Exact); ("b", Inexact) ]))
            ^ "\n"
            ^ optimized (filtered (answering [ ("a", Exact); ("c", Exact) ])))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64]
            slice ~offset:0 ~length:10
            └ filter (b < 2. && c = "x")
              └ m (5 columns) ~filters:[a > 1; b < 2.]
            query → a int64, b float64, c string, k int8, l list[int64]
            slice ~offset:0 ~length:10
            └ filter (b < 2.)
              └ m (5 columns) ~filters:[a > 1; c = "x"]
            |});
    ]

(* Query.optimize: constants *)

type output = Output : ('a, Expr.row) Expr.t -> output

(* [folded e] is the output [e] of a derive, as optimize folds it. *)
let folded (Output e) =
  let src =
    source "f"
      Type.
        [
          ("a", Any int64);
          ("k", Any int8);
          ("h", Any float16);
          ("x", Any float32);
        ]
  in
  let printed = optimized (Query.derive Expr.[ "y" := e ] (of_source src)) in
  let line = List.nth (String.split_on_char '\n' printed) 1 in
  let prefix = {|derive ["y" := |} in
  let start = String.length prefix in
  String.sub line start (String.length line - start - 1)

(* Expected values: Expr's [/] truncates and is null by zero, [mod] takes the
   dividend's sign, integers wrap as nx's do, floats are IEEE 754's; a value
   past OCaml's int stays unfolded (Query.optimize, "Constants"). *)
let folds =
  let a = Col.int "a" and k = Col.int "k" in
  let p e = Output e in
  Expr.
    [
      ("integers subtract", p (int 7 - int 10), "-3");
      ("integers multiply", p (int 3 * int 4), "12");
      ("a quotient truncates", p (int (-7) / int 2), "-3");
      ("a quotient by zero is null", p (int 1 / int 0), "null");
      ("a remainder takes the dividend's sign", p (int (-7) mod int 2), "-1");
      ("a remainder by zero is null", p (int 7 mod int 0), "null");
      ( "an overflowing sum stays",
        p (int max_int + int 1),
        "4611686018427387903 + 1" );
      ( "an overflowing product stays",
        p (int max_int * int 2),
        "4611686018427387903 * 2" );
      ( "an overflowing difference stays",
        p (int min_int - int 1),
        "-4611686018427387904 - 1" );
      ( "an overflowing quotient stays",
        p (int min_int / int (-1)),
        "-4611686018427387904 / -1" );
      ( "an int8 sum wraps",
        p (store Type.int8 (int 100) + store Type.int8 (int 100)),
        "-56" );
      ("an operand null is null", p (int 1 + store Type.int64 null), "null");
      ("floats subtract", p (float 1.5 -. float 0.25), "1.25");
      ("floats multiply", p (float 2. *. float 3.), "6.");
      ("a power", p (float 2. ** float 3.), "8.");
      ("a quotient by zero is infinite", p (float 1. /. float 0.), "infinity");
      ("zero by zero is NaN", p (float 0. /. float 0.), "nan");
      ( "float16 folds",
        p (Col.float "h" +. (float 0.5 +. float 0.25)),
        "h +. 0.75" );
      ( "float32 rounds",
        p (Col.float "x" +. (float 0.1 +. float 0.2)),
        "x +. 0.30000001192092896" );
      ("not equal", p (int 1 <> int 2), "true");
      ("at most", p (int 2 <= int 2), "true");
      ("NaN comes after every number", p (float 3. > float Float.nan), "false");
      ("at least", p (string "a" >= string "b"), "false");
      ("a comparison with null", p (int 1 = store Type.int64 null), "null");
      ("is_in of a literal", p (is_in [ 1; 2 ] (int 1)), "true");
      ("a text operation of a literal", p (Str.length (string "abc")), "3");
      ( "Str.concat stays",
        p (Str.concat "" [ string "a"; string "b" ]),
        {|Str.concat "" ["a"; "b"]|} );
      ("false and anything", p (bool false && a > int 0), "false");
      ("anything or true", p (a > int 0 || bool true), "true");
      ("null and false", p (store Type.bool null && bool false), "false");
      ("null or null", p (store Type.bool null || store Type.bool null), "null");
      ("not of a literal", p (not (bool true)), "false");
      ("not of null", p (not (store Type.bool null)), "null");
      ("a literal is not null", p (is_null (int 1)), "false");
      ("null is null", p (is_null (store Type.int64 null)), "true");
      ("if_ of true", p (if_ (bool true) a (int 1)), "a");
      ("if_ of null", p (if_ (store Type.bool null) a (int 1)), "1");
      ( "if_ keeps a branch of another type",
        p (if_ (bool true) k (store Type.int16 (int 1))),
        "if_ true k 1" );
      ("coalesce of nulls", p (coalesce [ store Type.int64 null; null ]), "null");
      ("coalesce of one", p (coalesce [ store Type.int64 null; a ]), "a");
      ( "coalesce keeps an operand of another type",
        p (coalesce [ k; store Type.int16 null ]),
        "coalesce [k]" );
      ( "if_ and coalesce simplify",
        p (if_ (bool false) a (coalesce [ null; a; int 3; k ])),
        "coalesce [a; 3]" );
    ]

(* A source of the types that the other operations that can fail take. *)
let s_typed =
  source "v"
    (columns
    @ Type.
        [
          ("t", Any (wall Ns));
          ("dl", Any (list (duration Ns)));
          ("x", Any (tensor Nx.float32 [| 2 |]));
        ])

let t = Col.wall "t"

let constants =
  let src = of_source s_exact in
  group "Query.optimize constants"
    [
      cases "fold to what evaluation gives"
        ~id:(fun (what, _, _) -> what)
        folds
        (fun (_, e, expected) -> equal string expected (folded e));
      test "fold inside frames, calendar operations and lifts" (fun () ->
          let three = Expr.(b +. (float 1. +. float 2.)) in
          let derived o = optimized (Query.derive [ o ] (of_source s_typed)) in
          expect
            (derived Expr.("e" := over (mean three))
            ^ "\n"
            ^ derived
                Expr.("y" := Temporal.format "%Y" (if_ (bool true) t null))
            ^ "\n"
            ^ derived Expr.("z" := nx { f = (fun x -> Nx.add x x) } three))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64], t datetime[ns], dl list[duration[ns]], x tensor[float32, 2], e float64
            derive ["e" := over (mean (b +. 3.))]
            └ v (8 columns)
            query → a int64, b float64, c string, k int8, l list[int64], t datetime[ns], dl list[duration[ns]], x tensor[float32, 2], y string
            derive ["y" := Temporal.format "%Y" t]
            └ v (8 columns)
            query → a int64, b float64, c string, k int8, l list[int64], t datetime[ns], dl list[duration[ns]], x tensor[float32, 2], z float64
            derive ["z" := add (b +. 3.) (b +. 3.)]
            └ v (8 columns)
            |});
      test "a filter of true goes" (fun () ->
          same_plan (Query.filter Expr.(bool true) src) src);
      cases "a filter with a conjunct false or null keeps no row"
        ~id:(fun (what, _) -> what)
        Expr.
          [
            ("false", bool false);
            ("null of bool", store Type.bool null);
            ("a bare null", null);
            ("false beside another", a > int 1 && bool false);
          ]
        (fun (_, p) ->
          same_plan (Query.filter p src) (Query.slice ~offset:0 ~length:0 src));
      test "x && true and false || x are x" (fun () ->
          let x = Expr.(a > int 1) in
          same_plan
            (Query.filter Expr.(x && bool true) src)
            (Query.filter x src);
          same_plan
            (Query.filter Expr.(bool false || x) src)
            (Query.filter x src));
    ]

(* Query.optimize: predicates *)

let typed = of_source s_typed

let can_fail =
  let l = Col.v Kind.(list int) "l" and dl = Col.v Kind.(list span) "dl" in
  Expr.
    [
      ("a narrowing cast", cast Type.int8 a = int 1);
      ("an OCaml function", const positive $ a);
      ("Str.parse", Str.parse Type.int64 c = a);
      ("of_option", of_option (option a) = a);
      ("temporal arithmetic", Temporal.add t (span (Time.Span.days 1)) > t);
      ("temporal parsing", is_null (Temporal.parse "%Y" Type.date c));
      ( "per_list of two lists",
        per_list [ "e" := l; "d" := dl ] (sum (Col.int "e")) > int 0 );
      ("per_list with only", per_list [ "e" := l ] (only (Col.int "e")) > int 0);
      ( "per_list of a duration sum",
        per_list [ "d" := dl ] (sum (Col.span "d")) > span (Time.Span.days 1) );
    ]

let predicates =
  let p = Expr.(a > int 1) in
  let cast_p = Expr.(cast Type.int8 a = int 1) in
  let joined ?kind ?each_left ?(on = Join.keys [ "k" ]) right q =
    Query.join ?kind ?each_left ~on right q
  in
  group "Query.optimize predicates"
    [
      cases "a predicate conjunct goes to an Exact source"
        ~id:(fun (what, _, _) -> what)
        Expr.
          [
            ("a literal on the left", int 3 < a, "3 < a");
            ("is_in", is_in [ "x"; "y" ] c, {|is_in ["x"; "y"] c|});
            ("not is_null", not (is_null b), "not (is_null b)");
            ("is_null", is_null b, "is_null b");
            ( "or and not",
              a = int 1 || not (c = string "x"),
              {|a = 1 || not (c = "x")|} );
            ("not equal, mirrored", float 2. <> b, "2. <> b");
            ("at least, mirrored", int 3 >= a, "3 >= a");
            ( "not of a conjunction",
              not (a = int 1 && b > float 2.),
              "not (a = 1 && b > 2.)" );
          ]
        (fun (_, p, printed) ->
          let plan = optimized (Query.filter p (of_source s_exact)) in
          equal text
            (Printf.sprintf "e (5 columns) ~filters:[%s]" printed)
            (List.nth (String.split_on_char '\n' plan) 1));
      test "an extension's conjunct goes to the source encoded" (fun () ->
          let timed =
            source ~pushdown:exact "t"
              Type.[ ("t", Any (ext ~name:"ymir.epoch" float64)) ]
          in
          let t = Ext.col epoch "t" in
          expect
            (optimized
               (Query.filter
                  Expr.(t < const 1.5 || is_in [ 2. ] t)
                  (of_source timed)))
          @@ __POS_OF__
               {|
            query → t ext[ymir.epoch, float64]
            t (1 column) ~filters:[t < <const> || is_in […] t]
            |});
      cases "a conjunct that is no predicate stays over the source"
        ~id:(fun (what, _) -> what)
        Expr.
          [
            ("two columns", a > k);
            ("an expression of a column", a + int 1 > int 3);
            ("text", Str.length c > int 2);
            ("an OCaml value", const positive $ a);
            ("a null literal", a = null);
          ]
        (fun (_, p) -> stays p (of_source s_exact));
      test "a conjunct passes a sort" (fun () ->
          same_plan
            (Query.filter p (Query.sort [ Order.asc "b" ] src))
            (Query.sort [ Order.asc "b" ] (Query.filter p src)));
      test "a conjunct merges with a row-local filter below it" (fun () ->
          let q = Expr.(b < float 0.) in
          same_plan
            (Query.filter p (Query.filter q src))
            (Query.filter Expr.(q && p) src));
      test "a conjunct passes a select that renames the column it reads"
        (fun () ->
          let renaming =
            Query.select Expr.[ "y" := a; keep Sel.(names [ "b" ]) ]
          in
          same_plan
            (Query.filter Expr.(Col.int "y" > int 1) (renaming src))
            (renaming (Query.filter p src)));
      test "a conjunct passes a derive whose outputs it does not read"
        (fun () ->
          let deriving = Query.derive Expr.[ "y" := a + int 1 ] in
          let q = Expr.(b > float 0.) in
          same_plan
            (Query.filter q (deriving src))
            (deriving (Query.filter q src)));
      test "a conjunct passes a derive that copies the column it reads"
        (fun () ->
          let copying = Query.derive Expr.[ "y" := a ] in
          same_plan
            (Query.filter Expr.(Col.int "y" > int 1) (copying src))
            (copying (Query.filter p src)));
      test "a conjunct on keys without floats passes an aggregate" (fun () ->
          let counting =
            Query.aggregate ~by:[ "a"; "c" ] Expr.[ "n" := rows ]
          in
          let q = Expr.(p && c = string "x") in
          same_plan
            (Query.filter q (counting src))
            (counting (Query.filter q src)));
      test "a conjunct enters both inputs of an append" (fun () ->
          same_plan
            (Query.filter p (Query.append src src))
            (Query.append (Query.filter p src) (Query.filter p src)));
      test "a conjunct enters the side of an inner join it reads" (fun () ->
          let o = of_source other in
          same_plan
            (Query.filter Expr.(p && w > float 0.) (joined o src))
            (joined (Query.filter Expr.(w > float 0.) o) (Query.filter p src)));
      test "a conjunct enters the left input of a semi join" (fun () ->
          let o = of_source other in
          same_plan
            (Query.filter p (joined ~kind:Semi o src))
            (joined ~kind:Semi o (Query.filter p src)));
      test "a left join keeps a conjunct on its right side above it" (fun () ->
          let o = of_source other in
          same_plan
            (Query.filter Expr.(p && w > float 0.) (joined ~kind:Left o src))
            (Query.filter
               Expr.(w > float 0.)
               (joined ~kind:Left o (Query.filter p src))));
      cases "a conjunct stops at"
        ~id:(fun (what, _, _) -> what)
        (let o = of_source other in
         let pw = Expr.(p && w > float 0.) in
         let widened =
           of_source
             (source "n"
                (List.map
                   (function "a", _ -> ("a", Type.Any Type.int32) | c -> c)
                   columns))
         in
         Query.
           [
             ("a slice", p, Kit.head 3 src);
             ("a full join", pw, joined ~kind:Full o src);
             ("a join with an assertion", pw, joined ~each_left:One o src);
             ( "a join by position",
               pw,
               joined ~on:Join.position
                 (of_source (source "w" Type.[ ("w", Any float64) ]))
                 src );
             ("an append that widens the column it reads", p, append widened src);
             ( "a derive over the frame",
               p,
               derive Expr.[ "s" := over (sum a) ] src );
             ("a filter over the frame", p, filter Expr.(b > over (mean b)) src);
             ( "a filter over the frame of an exact source",
               p,
               filter Expr.(b > over (mean b)) (of_source s_exact) );
             ( "a join, reading a right column and a key",
               Expr.(w > float 0. || k = int 1),
               joined o src );
             ( "an aggregate's output",
               Expr.(Col.int "n" > int 1),
               aggregate ~by:[ "c" ] Expr.[ "n" := rows ] src );
             ( "an aggregate's float key",
               Expr.(b > float 0.),
               aggregate ~by:[ "b" ] Expr.[ "n" := rows ] src );
             ( "an aggregate without keys",
               Expr.(Col.int "n" > int 1),
               aggregate ~by:[] Expr.[ "n" := rows ] src );
             ( "a computed column",
               Expr.(Col.int "y" > int 1),
               select Expr.[ "y" := a + int 1 ] src );
             ( "an unnest",
               Expr.(Col.int "id" = int 1),
               unnest
                 Expr.[ "e" := Col.v Kind.(list int) "l" ]
                 (select Expr.[ "id" := a; keep Sel.(names [ "l" ]) ] src) );
           ])
        (fun (_, p, q) -> stays p q);
      test "of an aggregate's keys, a conjunct on one holding floats stays"
        (fun () ->
          expect
            (optimized
               Query.(
                 of_source
                   (source "lf"
                      Type.
                        [
                          ("lf", Any (list float64));
                          ("li", Any (list int64));
                          ("e", Any (ext ~name:"ymir.epoch" float64));
                        ])
                 |> aggregate ~by:[ "lf"; "li"; "e" ] Expr.[ "n" := rows ]
                 |> filter
                      Expr.(
                        is_null (Col.v (Kind.list Kind.float) "lf")
                        && is_null (Col.v (Kind.list Kind.int) "li"))))
          @@ __POS_OF__
               {|
            query → lf list[float64], li list[int64], e ext[ymir.epoch, float64], n int64
            filter (is_null lf)
            └ aggregate ~by:["lf"; "li"; "e"] ["n" := rows]
              └ filter (is_null li)
                └ lf (3 columns)
            |});
      cases "a conjunct that can fail stays above a filter and a join" ~id:fst
        can_fail (fun (_, p) ->
          stays p (Query.filter Expr.(b > float 0.) typed);
          stays p (joined (of_source other) typed));
      test "a conjunct that can fail passes a sort" (fun () ->
          let sorting = Query.sort [ Order.asc "b" ] in
          same_plan
            (Query.filter cast_p (sorting src))
            (sorting (Query.filter cast_p src)));
      test "a conjunct that can fail passes a select" (fun () ->
          let keeping = Query.select Expr.[ keep Sel.(names [ "a"; "b" ]) ] in
          same_plan
            (Query.filter cast_p (keeping src))
            (keeping (Query.filter cast_p src)));
      test "a conjunct that can fail passes a derive" (fun () ->
          let deriving = Query.derive Expr.[ "y" := b ] in
          same_plan
            (Query.filter cast_p (deriving src))
            (deriving (Query.filter cast_p src)));
      test "a conjunct that can fail passes an aggregate's keys and an append"
        (fun () ->
          let keyed = Query.aggregate ~by:[ "a" ] Expr.[ "n" := rows ] in
          same_plan
            (Query.filter cast_p (keyed (Query.append src src)))
            (keyed
               (Query.append (Query.filter cast_p src) (Query.filter cast_p src))));
      test "a conjunct renamed through a select reads the input's name"
        (fun () ->
          let timed =
            source ~pushdown:exact "t"
              Type.
                [
                  ("t", Any (ext ~name:"ymir.epoch" float64));
                  ("l", Any (list int64));
                ]
          in
          expect
            (optimized
               Query.(
                 of_source timed
                 |> select Expr.[ "t2" := Ext.col epoch "t" ]
                 |> filter Expr.(Ext.col epoch "t2" < const 1.5)))
          @@ __POS_OF__
               {|
            query → t2 ext[ymir.epoch, float64]
            select ["t2" := t]
            └ t (2 columns) ~columns:["t"] ~filters:[t < <const>]
            |});
    ]

(* Constant columns *)

let branch ?pushdown name y =
  of_source (source ?pushdown name columns)
  |> Query.derive Expr.[ "year" := int y ]

let fold_of_years =
  branch ~pushdown:inexact "part 2001" 2001
  |> Query.append (branch ~pushdown:inexact "part 2002" 2002)

let year = Col.int "year"

let constant_columns =
  group "Query.optimize constant columns"
    [
      test "a conjunct on a constant column keeps the branches it is true on"
        (fun () ->
          expect
            (optimized
               (Query.filter Expr.(year = int 2001 && a > int 0) fold_of_years))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64], year int64
            append
            ├ derive ["year" := 2001]
            │ └ filter (a > 0)
            │   └ part 2001 (5 columns) ~filters:[a > 0]
            └ derive ["year" := 2002]
              └ slice ~offset:0 ~length:0
                └ part 2002 (5 columns) ~filters:[a > 0] ~limit:0
            |});
      test "a mixed disjunction reaches the source without its constant part"
        (fun () ->
          expect
            (optimized
               (Query.filter Expr.(year = int 2001 || a > int 0) fold_of_years))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64], year int64
            append
            ├ derive ["year" := 2001]
            │ └ part 2001 (5 columns)
            └ derive ["year" := 2002]
              └ filter (a > 0)
                └ part 2002 (5 columns) ~filters:[a > 0]
            |});
      test "a negated constant conjunct empties the branch it is false on"
        (fun () ->
          expect
            (optimized
               (Query.filter Expr.(not (year = int 2001)) fold_of_years))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64], year int64
            append
            ├ derive ["year" := 2001]
            │ └ slice ~offset:0 ~length:0
            │   └ part 2001 (5 columns) ~limit:0
            └ derive ["year" := 2002]
              └ part 2002 (5 columns)
            |});
    ]

(* Query.optimize: slices *)

let slices =
  let head n q = Query.slice ~offset:0 ~length:n q in
  group "Query.optimize slices"
    [
      test "a slice moves below a row-local derive" (fun () ->
          let deriving = Query.derive Expr.[ "y" := a + int 1 ] in
          same_plan (head 5 (deriving src)) (deriving (head 5 src)));
      test "two slices from the start merge" (fun () ->
          same_plan
            (Query.slice ~offset:3 ~length:4
               (Query.slice ~offset:2 ~length:5 src))
            (Query.slice ~offset:5 ~length:2 src));
      test "a slice from the start of a slice from the end stays" (fun () ->
          let q =
            Query.slice ~offset:1 ~length:1
              (Query.slice ~offset:(-5) ~length:3 src)
          in
          equal query_w q (Query.optimize q));
      test "a slice from the start limits both inputs of an append" (fun () ->
          same_plan
            (Query.slice ~offset:2 ~length:3 (Query.append src src))
            (Query.slice ~offset:2 ~length:3
               (Query.append (head 5 src) (head 5 src))));
      test "a slice gives the source below it a limit" (fun () ->
          expect
            (optimized
               (head 5 (Query.filter Expr.(a > int 1) (of_source s_exact))))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64]
            slice ~offset:0 ~length:5
            └ e (5 columns) ~filters:[a > 1] ~limit:5
            |});
      test "a conjunct that stays withholds the limit" (fun () ->
          expect
            (optimized
               (head 5 (Query.filter Expr.(a > int 1) (of_source s_inexact))))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64]
            slice ~offset:0 ~length:5
            └ filter (a > 1)
              └ i (5 columns) ~filters:[a > 1]
            |});
      test
        "a slice from the end of a source that states its rows counts from the \
         start" (fun () ->
          let r = of_source s_rows in
          same_plan (Kit.tail 10 r) (Query.slice ~offset:90 ~length:10 r);
          same_plan
            (Query.slice ~offset:(-150) ~length:70 r)
            (Query.slice ~offset:0 ~length:20 r));
      test "a slice from the end of a source handed a conjunct gets no limit"
        (fun () ->
          expect
            (optimized
               (Kit.tail 10 (Query.filter Expr.(a > int 1) (of_source s_rows))))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64]
            slice ~offset:(-10) ~length:10
            └ r (5 columns, 100 rows) ~filters:[a > 1]
            |});
      test "a slice moves below a select, above a sort" (fun () ->
          expect
            (optimized
               (head 3
                  (Query.select
                     Expr.[ keep Sel.(names [ "a"; "b" ]) ]
                     (Query.sort [ Order.desc "b" ] src))))
          @@ __POS_OF__
               {|
            query → a int64, b float64
            slice ~offset:0 ~length:3
            └ sort [desc "b"]
              └ u (5 columns) ~columns:["a"; "b"]
            |});
      test "a slice stops at a derive over the frame" (fun () ->
          let q = Query.derive Expr.[ "r" := over (rank b) ] src in
          equal query_w (head 3 q) (Query.optimize (head 3 q)));
      test "a slice stays above a conjunct that moves" (fun () ->
          expect
            (optimized
               (head 3
                  (Query.filter
                     Expr.(a > int 1)
                     (Query.select Expr.[ keep Sel.(names [ "a"; "b" ]) ] src))))
          @@ __POS_OF__
               {|
            query → a int64, b float64
            slice ~offset:0 ~length:3
            └ filter (a > 1)
              └ u (5 columns) ~columns:["a"; "b"]
            |});
    ]

(* Query.optimize: projections *)

let projections =
  group "Query.optimize projections"
    [
      test "a derive drops the outputs nothing reads" (fun () ->
          expect
            (optimized
               (Query.select
                  Expr.[ keep Sel.(names [ "y" ]) ]
                  (Query.derive Expr.[ "y" := a + int 1; "z" := b ] src)))
          @@ __POS_OF__
               {|
            query → y int64
            select [keep (names ["y"])]
            └ derive ["y" := a + 1]
              └ u (5 columns) ~columns:["a"]
            |});
      test "an aggregate drops the outputs nothing reads" (fun () ->
          expect
            (optimized
               (Query.select
                  Expr.[ keep Sel.(names [ "c"; "n" ]) ]
                  (Query.aggregate ~by:[ "c" ]
                     Expr.[ "n" := rows; "s" := sum a ]
                     src)))
          @@ __POS_OF__
               {|
            query → c string, n int64
            aggregate ~by:["c"] ["n" := rows]
            └ u (5 columns) ~columns:["c"]
            |});
      test "a select that keeps its input goes" (fun () ->
          same_plan (Query.select Expr.[ keep Sel.all ] src) src);
      test "counting rows reads no column" (fun () ->
          expect (optimized (Query.aggregate ~by:[] Expr.[ "n" := rows ] src))
          @@ __POS_OF__
               {|
            query → n int64
            aggregate ~by:[] ["n" := rows]
            └ u (5 columns) ~columns:[]
            |});
      test "a source reads a filter's column that nothing else reads" (fun () ->
          expect
            (optimized
               (Query.aggregate ~by:[ "c" ]
                  Expr.[ "n" := rows ]
                  (Query.filter Expr.(b > float 0.) src)))
          @@ __POS_OF__
               {|
            query → c string, n int64
            aggregate ~by:["c"] ["n" := rows]
            └ filter (b > 0.)
              └ u (5 columns) ~columns:["b"; "c"]
            |});
      test "an append's inputs keep only the columns it takes" (fun () ->
          expect
            (optimized
               (Query.select
                  Expr.[ keep Sel.(names [ "y" ]) ]
                  (Query.append
                     (Query.derive Expr.[ "y" := int 2 ] src)
                     (Query.derive Expr.[ "y" := a + int 1 ] src))))
          @@ __POS_OF__
               {|
            query → y int64
            append
            ├ select [keep (names ["y"])]
            │ └ derive ["y" := a + 1]
            │   └ u (5 columns) ~columns:["a"]
            └ derive ["y" := 2]
              └ u (5 columns) ~columns:[]
            |});
      test "a semi join reads only the right keys" (fun () ->
          expect
            (optimized
               (Query.join ~kind:Semi ~on:(Join.keys [ "k" ])
                  (of_source
                     (source "kb" Type.[ ("k", Any int8); ("b", Any float64) ]))
                  src))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64]
            join ~on:(keys ["k"]) ~kind:Semi
            ├ u (5 columns)
            └ kb (2 columns) ~columns:["k"]
            |});
      test "over reads its keys and order" (fun () ->
          expect
            (optimized
               (Query.select
                  Expr.[ keep Sel.(names [ "s" ]) ]
                  (Query.derive
                     Expr.
                       [
                         "s" :=
                           over ~by:[ "c" ] ~order:[ Order.asc "b" ] (sum a);
                       ]
                     src)))
          @@ __POS_OF__
               {|
            query → s int64
            select [keep (names ["s"])]
            └ derive ["s" := over ~by:["c"] ~order:[asc "b"] (sum a)]
              └ u (5 columns) ~columns:["a"; "b"; "c"]
            |});
      test "a replaced column keeps its place, and reads none under a select"
        (fun () ->
          expect
            (optimized (Query.derive Expr.[ "a" := int 1 ] src)
            ^ "\n"
            ^ optimized
                (Query.select
                   Expr.[ keep Sel.(names [ "a" ]) ]
                   (Query.derive Expr.[ "a" := k + int 1 ] src)))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64]
            derive ["a" := 1]
            └ u (5 columns)
            query → a int8
            select [keep (names ["a"])]
            └ derive ["a" := k + 1]
              └ u (5 columns) ~columns:["k"]
            |});
      test "a replaced column keeps its place in a step an append reads twice"
        (fun () ->
          let replaced = Query.derive Expr.[ "a" := int 1 ] src in
          let q = Query.append replaced replaced in
          equal schema_w (Query.schema q) (Query.schema (Query.optimize q)));
      test "an unnest repeats only the columns read above it" (fun () ->
          let s =
            source "data"
              Type.
                [
                  ("id", Any int64);
                  ("xs", Any (list int64));
                  ("lc", Any (list float64));
                ]
          in
          expect
            (optimized
               (Query.of_source s
               |> Query.unnest Expr.[ "e" := Col.v Kind.(list int) "xs" ]
               |> Query.select Expr.[ keep (Sel.names [ "e"; "id" ]) ]))
          @@ __POS_OF__
               {|
            query → e int64, id int64
            select [keep (names ["e"; "id"])]
            └ unnest ["e" := xs] ~repeat:["id"]
              └ data (3 columns) ~columns:["id"; "xs"]
            |});
    ]

(* Query.optimize: sharing *)

let sharing =
  let ex = of_source s_exact in
  group "Query.optimize sharing"
    [
      test "a self-join runs its aggregate once" (fun () ->
          let counts = Query.aggregate ~by:[ "k" ] Expr.[ "n" := rows ] src in
          expect
            (optimized
               (Query.join ~on:(Join.keys [ "k" ])
                  (Query.select
                     Expr.[ keep Sel.(names [ "k" ]); "m" := Col.int "n" ]
                     counts)
                  counts))
          @@ __POS_OF__
               {|
            query → k int8, n int64, m int64
            join ~on:(keys ["k"])
            ├ #1 aggregate ~by:["k"] ["n" := rows]
            │ └ u (5 columns) ~columns:["k"]
            └ select [keep (names ["k"]); "m" := n]
              └ #1
            |});
      test "complete reads its source in three places" (fun () ->
          expect
            (optimized
               (Kit.complete [ "c"; "k" ]
                  (Query.select Expr.[ keep Sel.(names [ "c"; "k"; "a" ]) ] src)))
          @@ __POS_OF__
               {|
            query → c string, k int8, a int64
            join ~on:(keys ["c"; "k"]) ~kind:Left
            ├ join ~on:all
            │ ├ aggregate ~by:["c"] []
            │ │ └ u (5 columns) ~columns:["c"]
            │ └ aggregate ~by:["k"] []
            │   └ u (5 columns) ~columns:["k"]
            └ select [keep (names ["c"; "k"; "a"])]
              └ u (5 columns) ~columns:["a"; "c"; "k"]
            |});
      test "one source read with other conjuncts or columns is two reads"
        (fun () ->
          expect
            (optimized
               (Query.append
                  (Query.filter Expr.(a > int 1 && c = string "x") ex)
                  (Query.filter Expr.(a > int 1 && b > float 0.) ex))
            ^ "\n"
            ^ optimized
                (Query.join ~on:(Join.keys [ "k" ])
                   (Query.select Expr.[ keep Sel.(names [ "k"; "b" ]) ] ex)
                   (Query.select Expr.[ keep Sel.(names [ "a"; "k" ]) ] ex)))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64]
            append
            ├ e (5 columns) ~filters:[a > 1; b > 0.]
            └ e (5 columns) ~filters:[a > 1; c = "x"]
            query → a int64, k int8, b float64
            join ~on:(keys ["k"])
            ├ e (5 columns) ~columns:["a"; "k"]
            └ select [keep (names ["k"; "b"])]
              └ e (5 columns) ~columns:["b"; "k"]
            |});
      test "two places with one request are one read" (fun () ->
          expect
            (optimized (Query.append (Kit.head 3 ex) (Kit.head 3 ex))
            ^ "\n"
            ^ optimized (Query.append ex (Kit.head 3 ex)))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64]
            append
            ├ #1 slice ~offset:0 ~length:3
            │ └ e (5 columns) ~limit:3
            └ #1
            query → a int64, b float64, c string, k int8, l list[int64]
            append
            ├ slice ~offset:0 ~length:3
            │ └ e (5 columns) ~limit:3
            └ e (5 columns)
            |});
    ]

(* Query.optimize: laws *)

let other_exact =
  source ~pushdown:exact "other_e" Type.[ ("k", Any int8); ("w", Any float64) ]

let list_l = Col.v Kind.(list int) "l"

(* [steps] is the steps that pipelines are drawn from. A step that does not
   apply to the query it is drawn for is skipped. *)
let steps =
  let joined ?kind ?each_right ?(on = Join.keys [ "k" ]) right q =
    Query.join ?kind ?each_right ~on (of_source right) q
  in
  Query.
    [
      ("filter a > 1", filter Expr.(a > int 1));
      ("filter b < 2. && c = x", filter Expr.(b < float 2. && c = string "x"));
      ("filter is_in k", filter Expr.(is_in [ 1; 2 ] k));
      ("filter b or a", filter Expr.((not (is_null b)) || a = int 0));
      ("filter a + 1 > k", filter Expr.(a + int 1 > k));
      ("filter cast", filter Expr.(cast Type.int8 a = int 1));
      ("filter $", filter Expr.(const positive $ a));
      ("filter over", filter Expr.(b > over (mean b)));
      ("filter true &&", filter Expr.(bool true && a > int 0));
      ("filter w", filter Expr.(w > float 0.));
      ( "filter per_list",
        filter Expr.(per_list [ "e" := list_l ] (sum (Col.int "e")) > int 0) );
      ( "filter per_list naming a renamed column",
        filter
          Expr.(
            per_list [ "a" := list_l ] (sum (Col.int "a" * Col.int "y")) > int 0)
      );
      ("select keep", select Expr.[ keep Sel.(names [ "a"; "b"; "c"; "k" ]) ]);
      ("select rename", select Expr.[ "y" := a; keep Sel.(all - names [ "a" ]) ]);
      ( "select back",
        select Expr.[ "a" := Col.int "y"; keep Sel.(all - names [ "y" ]) ] );
      ( "select computed",
        select Expr.[ "y" := a + int 1; keep Sel.(names [ "k" ]) ] );
      ("derive replace", derive Expr.[ "a" := a * int 2 ]);
      ("derive new", derive Expr.[ "z" := b *. float 2. ]);
      ("derive copy", derive Expr.[ "y" := a ]);
      ("derive over", derive Expr.[ "r" := over (sum a) ]);
      ("derive constant", derive Expr.[ "z" := int 2 * int 3 ]);
      ("sort", sort [ Order.asc "a" ]);
      ("head", Kit.head 4);
      ("slice", slice ~offset:2 ~length:3);
      ("tail", Kit.tail 2);
      ( "aggregate",
        aggregate ~by:[ "c"; "k" ] Expr.[ "a" := sum a; "n" := rows ] );
      ("aggregate all", aggregate ~by:[] Expr.[ "n" := rows; "a" := sum a ]);
      ("join", joined other);
      ("join left", joined ~kind:Left other_exact);
      ("join semi", joined ~kind:Semi other_exact);
      ("join anti", joined ~kind:Anti other);
      ("join full", joined ~kind:Full other);
      ("join asserted", joined ~each_right:At_most_one other_exact);
      ( "join position",
        joined ~on:Join.position (source "v" Type.[ ("v", Any float64) ]) );
      ( "join itself",
        fun q ->
          Query.join ~on:(Join.keys [ "k" ])
            (Query.select Expr.[ keep Sel.(names [ "k" ]); "m" := a ] q)
            q );
      ("append itself", fun q -> append q q);
      ("append a source", append (of_source s_exact));
    ]

let starts =
  [
    ("u", s_unsupported);
    ("e", s_exact);
    ("i", s_inexact);
    ("r", s_rows);
    ( "m",
      source ~pushdown:(by_column [ ("a", Exact); ("b", Inexact) ]) "m" columns
    );
  ]

(* [build (start, ops)] is the pipeline of the steps [ops] over the source
   [start]. *)
let build (start, ops) =
  List.fold_left
    (fun q op ->
      match List.assoc op steps q with
      | q -> q
      | exception Invalid_argument _ -> q)
    (of_source (List.assoc start starts))
    ops

let pipeline_gen =
  Gen.(
    pair
      (of_list ~pp:Format.pp_print_string (List.map fst starts))
      (list ~size:(int_range 0 7) (of_list (List.map fst steps))))
  |> Gen.with_pp (fun ppf (start, ops) ->
      Format.fprintf ppf "%s: %s" start (String.concat " |> " ops))

(* A source gets a limit, is handed conjuncts, and a step is shared on few
   draws: these reach each on every run. *)
let pipeline_examples =
  [ ("e", [ "head" ]); ("u", [ "append itself" ]); ("e", [ "filter a > 1" ]) ]

let idempotent d =
  let o = Query.optimize (build d) in
  let printed = str Query.pp o in
  let shows sub =
    let n = String.length sub in
    let rec at i =
      i + n <= String.length printed
      && (String.equal (String.sub printed i n) sub || at (i + 1))
    in
    at 0
  in
  cover "a source is handed conjuncts" (shows "~filters:");
  cover "a source reads fewer columns" (shows "~columns:");
  cover "a source gets a limit" (shows "~limit:");
  cover "a step is shared" (shows "#1");
  cover "a filter stays" (shows "filter (");
  equal query_w o (Query.optimize o)

(* The pushdown model: a filter of conjuncts over a source that answers for each
   by its column, then maybe a slice. *)

let atoms =
  Expr.
    [
      (a > int 1, "a > 1", Source.Pred.Cmp ("a", `Gt, Value (Type.int64, 1)));
      (b <= float 2., "b <= 2.", Cmp ("b", `Le, Value (Type.float64, 2.)));
      (int 3 < a, "3 < a", Cmp ("a", `Gt, Value (Type.int64, 3)));
      (float 2. <= b, "2. <= b", Cmp ("b", `Ge, Value (Type.float64, 2.)));
      (int 3 > a, "3 > a", Cmp ("a", `Lt, Value (Type.int64, 3)));
      (c = string "x", {|c = "x"|}, Cmp ("c", `Eq, Value (Type.string, "x")));
      ( is_in [ 1; 2 ] k,
        "is_in [1; 2] k",
        In ("k", [ Value (Type.int8, 1); Value (Type.int8, 2) ]) );
      (is_null b, "is_null b", Null "b");
      (not (is_null a), "not (is_null a)", Valid "a");
    ]

let pp_answer ppf a =
  Format.pp_print_string ppf
    (match a with
    | Source.Exact -> "Exact"
    | Inexact -> "Inexact"
    | Unsupported -> "Unsupported")

let model_gen =
  let text (_, t, _) = t in
  Gen.(
    let* chosen = subsequence atoms in
    let* chosen = permutation chosen in
    let* answers =
      list ~size:(constant 4) (of_list [ Source.Exact; Inexact; Unsupported ])
    in
    let+ sliced = bool in
    (chosen, List.combine [ "a"; "b"; "c"; "k" ] answers, sliced))
  |> Gen.with_pp (fun ppf (chosen, answers, sliced) ->
      Format.fprintf ppf "%s; %s%s"
        (String.concat " && " (List.map text chosen))
        (String.concat ", "
           (List.map
              (fun (c, a) -> Format.asprintf "%s:%a" c pp_answer a)
              answers))
        (if sliced then "; sliced" else ""))

(* The source gets a limit on few draws: this reaches it on every run. *)
let model_examples =
  [
    ( [ List.hd atoms ],
      [ ("a", Source.Exact); ("b", Exact); ("c", Exact); ("k", Exact) ],
      true );
  ]

let rec pp_pred ppf : Source.Pred.t -> unit = function
  | Cmp (c, _, Value (ty, _)) -> Format.fprintf ppf "Cmp %S at %a" c Type.pp ty
  | In (c, vs) ->
      let ty ppf (Source.Pred.Value (ty, _)) = Type.pp ppf ty in
      Format.fprintf ppf "In %S at [%a]" c (Format.pp_print_list ty) vs
  | Null c -> Format.fprintf ppf "Null %S" c
  | Valid c -> Format.fprintf ppf "Valid %S" c
  | Not p -> Format.fprintf ppf "Not (%a)" pp_pred p
  | And ps | Or ps ->
      Format.fprintf ppf "[%a]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
           pp_pred)
        ps

let wide pp q =
  let b = Buffer.create 256 in
  let ppf = Format.formatter_of_buffer b in
  Format.pp_set_margin ppf 10_000;
  pp ppf q;
  Format.pp_print_flush ppf ();
  Buffer.contents b

let rec conj = function
  | [] -> Expr.bool true
  | [ e ] -> e
  | e :: es -> Expr.(e && conj es)

let pushdown_model (es, answers, sliced) =
  let asked = ref [] in
  let answer p =
    asked := p :: !asked;
    by_column answers p
  in
  let s = source ~pushdown:answer "m" columns in
  let sliced_by q = if sliced then Kit.head 7 q else q in
  let filtered es s =
    match es with
    | [] -> of_source s
    | _ -> Query.filter (conj (List.map (fun (e, _, _) -> e) es)) (of_source s)
  in
  let optimized = Query.optimize (sliced_by (filtered es s)) in
  let answer_of (_, _, p) = by_column answers p in
  let handed = List.filter (fun e -> answer_of e <> Unsupported) es
  and stays = List.filter (fun e -> answer_of e <> Exact) es in
  cover "a conjunct leaves the plan" (List.length stays < List.length es);
  cover "a conjunct stays" (stays <> []);
  cover "the source gets a limit" (sliced && stays = []);
  (* The model is the plan over a source of the same name, with the request
     Source.request states printed after it. *)
  let model = sliced_by (filtered stays (source "m" columns)) in
  let request =
    (match handed with
      | [] -> ""
      | _ ->
          Format.asprintf " ~filters:[%s]"
            (String.concat "; " (List.map (fun (_, t, _) -> t) handed)))
    ^ if sliced && stays = [] then " ~limit:7" else ""
  in
  equal text (wide Query.pp model ^ request) (wide Query.pp optimized);
  List.iter
    (fun p ->
      mem ~msg:"a predicate the source is asked about is a conjunct's"
        (Testable.make ~pp:pp_pred ~equal:( = ))
        p
        (List.map (fun (_, _, p) -> p) es))
    !asked

(* [words f] is the number of words allocated by [f ()]. *)
let words f =
  let before = Gc.allocated_bytes () in
  ignore (Sys.opaque_identity (f ()));
  (Gc.allocated_bytes () -. before) /. float_of_int (Sys.word_size / 8)

(* Linear work doubles from n to 2n; 2.3 leaves room for a plan's constant part,
   and quadratic work would give 4. *)
let linear at = less (float 0.001) ~than:(2.3 *. at 400) (at 800)

(* [left_fold leaf n] is [n] leaves appended one after the other, as
   [List.fold_left] builds them, under a filter that enters every branch. *)
let left_fold leaf n =
  let rec fold q i =
    if i = n then q else fold (q |> Query.append (leaf i)) (i + 1)
  in
  fold (leaf 0) 1 |> Query.filter Expr.(a > int 0)

let laws =
  group "Query.optimize laws"
    [
      prop "keeps the schema" pipeline_gen (fun d ->
          let q = build d in
          equal schema_w (Query.schema q) (Query.schema (Query.optimize q)));
      prop "is idempotent" ~examples:pipeline_examples pipeline_gen idempotent;
      prop "gives equal plans for a pipeline built twice" pipeline_gen (fun d ->
          equal query_w (Query.optimize (build d)) (Query.optimize (build d)));
      prop "respects the source's answers" ~examples:model_examples model_gen
        pushdown_model;
      cases "plan work over a left fold is linear in its branches" ~id:fst
        [
          ( "sources of their own names",
            fun i ->
              of_source
                (source ~pushdown:inexact (Printf.sprintf "part %d" i) columns)
          );
          ( "sources of one name",
            fun _ -> of_source (source ~pushdown:inexact "part" columns) );
          ( "tables of one shape",
            fun _ ->
              Query.of_table (Talon.v [ ("a", Column.v Type.int64 [| 1 |]) ]) );
        ]
        (fun (_, leaf) ->
          linear (fun n -> words (fun () -> Query.optimize (left_fold leaf n))));
      test "planning union_all is linear in its queries" (fun () ->
          linear (fun n ->
              let leaf _ =
                of_source (source ~pushdown:inexact "part" columns)
              in
              let q = Kit.union_all (List.init n leaf) in
              words (fun () -> Query.optimize (Query.filter Expr.(a > int 0) q))));
      test "plan work over a doubling is linear in its depth" (fun () ->
          (* Each level reaches the step below it twice: rewritten once per
             path, 16 levels cost 256 times 8 levels' work. *)
          let at d =
            let rec double q d =
              if d = 0 then q else double (Query.append q q) (d - 1)
            in
            let q =
              double (of_source (source ~pushdown:inexact "part" columns)) d
            in
            words (fun () -> Query.optimize (Query.filter Expr.(a > int 0) q))
          in
          less (float 0.001) ~than:(2.3 *. at 8) (at 16));
    ]
(* Comparing and formatting *)

let pred_w = Testable.make ~pp:pp_pred ~equal:( = )

(* A source may be asked about a conjunct several times, so the questions are
   compared as a set, in the order first asked. *)
let encoded () =
  let asked = ref [] in
  let timed =
    source
      ~pushdown:(fun p ->
        asked := p :: !asked;
        Exact)
      "t"
      Type.
        [
          ("t", Any (ext ~name:"ymir.epoch" float64));
          ("x", Any int16);
          ("f", Any float32);
          ("h", Any float16);
          ("fs", Any (list float32));
          ("g", Any (categorical [| "AA"; "B6" |]));
        ]
  in
  let t = Ext.col epoch "t" and x = Col.int "x" in
  let f = Col.float "f" and h = Col.float "h" in
  let fs = Col.v Kind.(list float) "fs" in
  ignore
    (Query.optimize
       (Query.filter
          Expr.(
            (t < const 1.5 || is_in [ 2. ] t)
            && store Type.int32 (int 7) = x
            && f > float 0.1
            && is_in [ 0.1 ] f && is_in [ [| 0.1 |] ] fs
            && Col.string "g" = string "AA"
            && Col.string "g" = store Type.string (string "zz")
            && h > float 0.5)
          (of_source timed)));
  equal (list pred_w)
    Source.Pred.
      [
        Or
          [
            Cmp ("t", `Lt, Value (Type.float64, 3.));
            In ("t", [ Value (Type.float64, 4.) ]);
          ];
        Cmp ("x", `Eq, Value (Type.int32, 7));
        Cmp ("f", `Gt, Value (Type.float32, 0.100000001490116119384765625));
        In ("f", [ Value (Type.float32, 0.100000001490116119384765625) ]);
        In
          ( "fs",
            [
              Value (Type.list Type.float32, [| 0.100000001490116119384765625 |]);
            ] );
        Cmp ("g", `Eq, Value (Type.categorical [| "AA"; "B6" |], "AA"));
        Cmp ("g", `Eq, Value (Type.string, "zz"));
      ]
    (List.fold_left
       (fun ps p -> if List.mem p ps then ps else ps @ [ p ])
       [] (List.rev !asked))

let comparing =
  let keep ns = Query.select Expr.[ keep Sel.(names ns) ] (of_source s_exact) in
  group "Query.equal and Query.pp"
    [
      test "optimized plans with different requests differ" (fun () ->
          not_equal query_w
            (Query.optimize (keep [ "a" ]))
            (Query.optimize (keep [ "b" ])));
      test "a source asked for all of it prints without a request" (fun () ->
          equal string
            "query → a int64, b float64, c string, k int8, l list[int64]\n\
             e (5 columns)"
            (str Query.pp (Query.optimize (of_source s_exact))));
      test
        "a source is asked at the type the comparison is made in, with the \
         value it stores there, a categorical as text beside text, and never \
         about a float16 literal"
        encoded;
      test "an unoptimized plan labels a step it reaches twice" (fun () ->
          let q = of_source s_exact in
          expect (str Query.pp (Query.append q q))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64]
            append
            ├ #1 e (5 columns)
            └ #1
            |});
      test "an unoptimized plan prints a predicate as written" (fun () ->
          expect (str Query.pp (Query.filter Expr.null (of_source s_exact)))
          @@ __POS_OF__
               {|
            query → a int64, b float64, c string, k int8, l list[int64]
            filter null
            └ e (5 columns)
            |});
    ]

(* Kit: queries *)

let category = Type.categorical [| "AA"; "B6"; "DL" |]
let epoch_t = Type.ext ~name:"ymir.epoch" Type.float64

let kit_types =
  Type.
    [
      Any int8;
      Any int64;
      Any uint16;
      Any float32;
      Any float64;
      Any string;
      Any bool;
      Any date;
      Any (list int64);
      Any category;
    ]

let pp_any ppf (Type.Any t) = Type.pp ppf t

(* [columns_gen names] draws columns with distinct names of [names], in any
   order, of any type of [kit_types]. *)
let columns_gen names =
  let open Gen in
  let* names = subsequence names in
  let* names = permutation names in
  let+ tys =
    list ~size:(constant (List.length names)) (of_list ~pp:pp_any kit_types)
  in
  List.combine names tys

let kit_schema_gen =
  Gen.with_pp Schema.pp
    (Gen.map Schema.v (columns_gen [ "a"; "b"; "c"; "d"; "e"; "f" ]))

let of_schema s = of_source (Source.v ~name:"s" ~schema:s (fun _ -> Ok []))

let subset_gen =
  Gen.(
    let* s = kit_schema_gen in
    let+ ns = subsequence (List.map fst (Schema.columns s)) in
    (s, ns))
  |> Gen.with_pp (fun ppf (s, ns) ->
      Format.fprintf ppf "%a; %s" Schema.pp s (String.concat ", " ns))

let keeps_schema s =
  let q = of_schema s in
  let keys = List.map (fun (n, _) -> Order.asc n) (Schema.columns s) in
  List.iter
    (fun q' -> equal schema_w s (Query.schema q'))
    [ Kit.head 3 q; Kit.tail 3 q; Kit.top_k 3 keys q; Kit.distinct q ]

let drops (s, ns) =
  equal schema_w
    (Schema.v
       (List.filter (fun (n, _) -> not (List.mem n ns)) (Schema.columns s)))
    (Query.schema (Kit.drop (Sel.names ns) (of_schema s)))

let renames (s, ns) =
  let pairs = List.map (fun n -> (n, n ^ "'")) ns in
  let renamed (n, t) = (Option.value ~default:n (List.assoc_opt n pairs), t) in
  equal schema_w
    (Schema.v (List.map renamed (Schema.columns s)))
    (Query.schema (Kit.rename pairs (of_schema s)))

let union_gen =
  Gen.(
    let* s = kit_schema_gen in
    let* shared = subsequence (Schema.columns s) in
    let+ own = columns_gen [ "g"; "h" ] in
    (s, Schema.v (own @ shared)))
  |> Gen.with_pp (fun ppf (s, r) ->
      Format.fprintf ppf "q: %a; rest: %a" Schema.pp s Schema.pp r)

let unions (s, r) =
  let only_rest =
    List.filter
      (fun (n, _) -> Option.is_none (Schema.find s n))
      (Schema.columns r)
  in
  equal schema_w
    (Schema.v (Schema.columns s @ only_rest))
    (Query.schema (Kit.union (of_schema r) (of_schema s)))

let null_counts s =
  equal schema_w
    (Schema.v
       (List.map (fun (n, _) -> (n, Type.Any Type.int64)) (Schema.columns s)))
    (Query.schema (Kit.null_count (of_schema s)))

let counts (s, ks) =
  let key n = (n, Option.get (Schema.find s n)) in
  equal schema_w
    (Schema.v (List.map key ks @ [ ("count", Type.Any Type.int64) ]))
    (Query.schema (Kit.count_by ks (of_schema s)))

(* [keyed_gen] draws a schema and keys of it, at least one. *)
let keyed_gen =
  Gen.(
    let* t = of_list ~pp:pp_any kit_types in
    let* rest = columns_gen [ "b"; "c"; "d"; "e"; "f" ] in
    let* cs = permutation (("a", t) :: rest) in
    let* ns = permutation (List.map fst cs) in
    let+ ks = subsequence (List.tl ns) in
    (Schema.v cs, List.hd ns :: ks))
  |> Gen.with_pp (fun ppf (s, ns) ->
      Format.fprintf ppf "%a; %s" Schema.pp s (String.concat ", " ns))

let completes (s, ks) =
  equal schema_w s (Query.schema (Kit.complete ks (of_schema s)))

let one_hot_gen =
  Gen.(
    let* s = kit_schema_gen in
    let+ at = int_range 0 (List.length (Schema.columns s)) in
    let cs = Schema.columns s in
    Schema.v
      (List.filteri (fun i _ -> i < at) cs
      @ [ ("cat", Type.Any category) ]
      @ List.filteri (fun i _ -> i >= at) cs))
  |> Gen.with_pp Schema.pp

let one_hots s =
  let indicators =
    List.map
      (fun cat -> ("cat_" ^ cat, Type.Any Type.bool))
      [ "AA"; "B6"; "DL" ]
  in
  equal schema_w
    (Schema.v
       (List.concat_map
          (fun (n, t) ->
            if String.equal n "cat" then indicators else [ (n, t) ])
          (Schema.columns s)))
    (Query.schema (Kit.one_hot "cat" (of_schema s)))

(* Kit's queries are the compositions their documentation shows. *)
let definitions =
  let q = of_source s_unsupported in
  let f =
    of_source
      (source "f"
         Type.[ ("x", Any int8); ("carrier", Any category); ("y", Any bool) ])
  in
  let p =
    of_source (source "p" Type.[ ("c", Any string); ("t", Any epoch_t) ])
  in
  let keep ns q = Query.select Expr.[ keep (Sel.names ns) ] q in
  group "Kit queries are their definitions"
    [
      test "head" (fun () ->
          equal query_w (Query.slice ~offset:0 ~length:3 q) (Kit.head 3 q));
      test "tail" (fun () ->
          equal query_w (Query.slice ~offset:(-3) ~length:3 q) (Kit.tail 3 q));
      test "top_k" (fun () ->
          let keys = [ Order.desc "b" ] in
          equal query_w
            Query.(q |> sort keys |> slice ~offset:0 ~length:2)
            (Kit.top_k 2 keys q));
      test "distinct, with columns and without" (fun () ->
          equal query_w
            (Query.aggregate ~by:[ "a"; "b"; "c"; "k"; "l" ] [] q)
            (Kit.distinct q);
          let none = Query.select [] q in
          equal query_w (Kit.head 1 none) (Kit.distinct none));
      test "count_by" (fun () ->
          equal query_w
            (Query.aggregate ~by:[ "c" ] Expr.[ "count" := rows ] q)
            (Kit.count_by [ "c" ] q));
      test "value_counts" (fun () ->
          equal query_w
            (Kit.count_by [ "c" ] q |> Query.sort [ Order.desc "count" ])
            (Kit.value_counts "c" q));
      test "drop" (fun () ->
          let sel = Sel.(names [ "l" ] + prefix "b") in
          equal query_w
            (Query.select Expr.[ keep Sel.(all - sel) ] q)
            (Kit.drop sel q));
      test "complete" (fun () ->
          let values k = Kit.distinct (keep [ k ] q) in
          equal query_w
            (Query.join ~on:Join.all (values "k") (values "c")
            |> Query.join ~kind:Left ~on:(Join.keys [ "c"; "k" ]) q
            |> keep [ "a"; "b"; "c"; "k"; "l" ])
            (Kit.complete [ "c"; "k" ] q));
      test "one_hot" (fun () ->
          let is cat = Expr.(Col.string "carrier" = string cat) in
          equal query_w
            (Query.select
               Expr.
                 [
                   keep (Sel.names [ "x" ]);
                   "carrier_AA" := is "AA";
                   "carrier_B6" := is "B6";
                   "carrier_DL" := is "DL";
                   keep (Sel.names [ "y" ]);
                 ]
               f)
            (Kit.one_hot "carrier" f));
      test "union" (fun () ->
          let padded =
            Query.derive Expr.[ "t" := store epoch_t null ] q
            |> Query.append
                 (Query.derive
                    Expr.
                      [
                        "a" := store Type.int64 null;
                        "b" := store Type.float64 null;
                        "k" := store Type.int8 null;
                        "l" := store (Type.list Type.int64) null;
                      ]
                    p)
          in
          equal query_w padded (Kit.union p q));
      test "null_count and rename print as their verbs" (fun () ->
          (* Their definitions pass a function to each, which equal compares by
             identity. *)
          expect
            (str Query.pp (Kit.null_count q)
            ^ "\n"
            ^ str Query.pp (Kit.rename [ ("a", "x"); ("k", "key") ] q))
          @@ __POS_OF__
               {|
            query → a int64, b int64, c int64, k int64, l int64
            aggregate ~by:[] ["a" := rows - count a; "b" := rows - count b;
                              "c" := rows - count c; "k" := rows - count k;
                              "l" := rows - count l]
            └ u (5 columns)
            query → x int64, b float64, c string, key int8, l list[int64]
            select ["x" := a; keep (names ["b"; "c"]); "key" := k; keep (names ["l"])]
            └ u (5 columns)
            |});
    ]

let describe () =
  let without =
    of_source (source "t" Type.[ ("s", Any string); ("d", Any date) ])
  in
  equal ~msg:"one schema with and without numeric columns" schema_w
    (Query.schema (Kit.describe (of_source s_unsupported)))
    (Query.schema (Kit.describe without));
  expect
    (str Query.pp (Kit.describe (of_source s_unsupported))
    ^ "\n"
    ^ str Query.pp (Kit.describe without))
  @@ __POS_OF__
       {|
    query → column string, count int64, nulls int64, mean float64, std float64, min float64, q25 float64, median float64, q75 float64, max float64
    append
    ├ append
    │ ├ aggregate ~by:[] ["column" := "a"; "count" := count a;
    │ │                   "nulls" := rows - count a; "mean" := mean a;
    │ │                   "std" := std a; "min" := cast float64 (min a);
    │ │                   "q25" := quantile 0.25 a; "median" := median a;
    │ │                   "q75" := quantile 0.75 a;
    │ │                   "max" := cast float64 (max a)]
    │ │ └ #1 u (5 columns)
    │ └ aggregate ~by:[] ["column" := "b"; "count" := count b;
    │                     "nulls" := rows - count b; "mean" := mean b;
    │                     "std" := std b; "min" := cast float64 (min b);
    │                     "q25" := quantile 0.25 b; "median" := median b;
    │                     "q75" := quantile 0.75 b;
    │                     "max" := cast float64 (max b)]
    │   └ #1
    └ aggregate ~by:[] ["column" := "k"; "count" := count k;
                        "nulls" := rows - count k; "mean" := mean k;
                        "std" := std k; "min" := cast float64 (min k);
                        "q25" := quantile 0.25 k; "median" := median k;
                        "q75" := quantile 0.75 k; "max" := cast float64 (max k)]
      └ #1
    query → column string, count int64, nulls int64, mean float64, std float64, min float64, q25 float64, median float64, q75 float64, max float64
    slice ~offset:0 ~length:0
    └ aggregate ~by:[] ["column" := ""; "count" := count (store float64 null);
                        "nulls" := rows - count (store float64 null);
                        "mean" := mean (store float64 null);
                        "std" := std (store float64 null);
                        "min" := cast float64 (min (store float64 null));
                        "q25" := quantile 0.25 (store float64 null);
                        "median" := median (store float64 null);
                        "q75" := quantile 0.75 (store float64 null);
                        "max" := cast float64 (max (store float64 null))]
      └ t (2 columns)
    |}

let kit_queries =
  group "Kit queries"
    [
      prop "head, tail, top_k and distinct keep the schema" kit_schema_gen
        keeps_schema;
      prop "drop removes exactly what it selects" subset_gen drops;
      prop "rename renames in place" subset_gen renames;
      prop "union has q's columns, then rest's own" union_gen unions;
      prop "null_count counts each column as int64" kit_schema_gen null_counts;
      prop "count_by has the keys, then count" subset_gen counts;
      prop "complete keeps the schema" keyed_gen completes;
      prop "one_hot puts one bool per category in place" one_hot_gen one_hots;
      test "describe has one row per numeric column" describe;
    ]

(* Kit: mistakes *)

let kit_mistakes =
  let q = of_source s_unsupported in
  let refused ~by f = raises_match (Exn.invalid_arg ~sub:by) f in
  group "Kit mistakes"
    [
      test "rename raises on a column renamed twice" (fun () ->
          refused ~by:"Kit.rename" (fun () ->
              Kit.rename [ ("a", "x"); ("a", "y") ] q));
      test "rename's missing and clashing names are its select's problems"
        (fun () ->
          expect
            (message (fun () -> Kit.rename [ ("aa", "x") ] q)
            ^ "\n"
            ^ message (fun () -> Kit.rename [ ("a", "b") ] q))
          @@ __POS_OF__
               {|
            select: 1 problem
              each (all + names ["aa"]) <fn>
                no column "aa". Did you mean "a"?
              input (5 columns): a int64, b float64, c string, k int8, l list[int64]
            select: 1 problem
              the output "b" appears twice.
              input (5 columns): a int64, b float64, c string, k int8, l list[int64]
            |});
      test "complete raises on no key and a key twice" (fun () ->
          refused ~by:"Kit.complete" (fun () -> Kit.complete [] q);
          refused ~by:"Kit.complete" (fun () ->
              Kit.complete [ "c"; "k"; "c" ] q));
      test "one_hot's missing column is its select's problem" (fun () ->
          expect (message (fun () -> Kit.one_hot "cc" q))
          @@ __POS_OF__
               {|
            select: 1 problem
              keep (names ["cc"])
                no column "cc". Did you mean "c"?
              input (5 columns): a int64, b float64, c string, k int8, l list[int64]
            |});
      test "one_hot raises on a column that is not categorical" (fun () ->
          refused ~by:"Kit.one_hot" (fun () -> Kit.one_hot "c" q));
    ]

(* Records and unnest *)

let unnest_printed =
  let s =
    source "data"
      Type.
        [
          ("id", Any int64); ("xs", Any (list int64)); ("lc", Any (list float64));
        ]
  in
  let xs = Col.v Kind.(list int) "xs" in
  group "Query.unnest and records"
    [
      test "print as written" (fun () ->
          let band = Expr.record Expr.[ "x" := Col.int "id"; "l" := xs ] in
          expect
            (str Query.pp
               (Query.of_source s
               |> Query.derive
                    Expr.[ "b" := band; "f" := field Kind.int "x" band ]
               |> Query.unnest ~keep_empty:true Expr.[ "e" := xs ]
               |> Query.select
                    Expr.
                      [
                        within (Col.v Record.kind "b")
                          (keep (Sel.names [ "x" ]));
                      ]))
          @@ __POS_OF__
               {|
            query → x int64
            select ["x" := within b x]
            └ unnest ~keep_empty:true ["e" := xs]
              └ derive ["b" := record ["x" := id; "l" := xs];
                        "f" := field int "x" (record ["x" := id; "l" := xs])]
                └ data (3 columns)
            |});
      test "collect of a list column is a list of lists" (fun () ->
          expect
            (str Query.pp
               (Query.aggregate ~by:[ "id" ]
                  Expr.[ "c" := collect xs ]
                  (Query.of_source s)))
          @@ __POS_OF__
               {|
            query → id int64, c list[list[int64]]
            aggregate ~by:["id"] ["c" := collect xs]
            └ data (3 columns)
            |});
    ]

let () =
  exit
    (run "plan"
       [
         pushdown;
         constants;
         predicates;
         constant_columns;
         slices;
         projections;
         sharing;
         laws;
         comparing;
         unnest_printed;
         kit_queries;
         definitions;
         kit_mistakes;
       ])
