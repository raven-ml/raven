(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next
open Windtrap

let str pp v = Format.asprintf "%a" pp v

(* [message f] is the message of the [Invalid_argument] that [f] raises. *)
let message f =
  match f () with _ -> "no exception" | exception Invalid_argument m -> m

let schema_w = Testable.make ~pp:Schema.pp ~equal:Schema.equal
let query_w = Testable.make ~pp:Query.pp ~equal:Query.equal

let source ?rows ?pushdown name columns =
  Source.v ~name ~schema:(Schema.v columns) ?rows ?pushdown (fun _ -> Ok [])

(* [plans cases] is each titled plan of [cases] optimized and printed. *)
let plans cases =
  String.concat "\n"
    (List.map
       (fun (title, q) -> "# " ^ title ^ "\n" ^ str Query.pp (Query.optimize q))
       cases)

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
let of_source = Query.of_source
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
        ("time_hour", Any (datetime ~zone:"UTC" Us));
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

(* Printed plans *)

let guide () =
  expect (str Query.pp (Query.optimize (late_by_carrier ())))
  @@ __POS_OF__
       {|
    query → carrier string, mean_delay float64, flights int64, name string
    sort [desc "mean_delay"]
    └ join ~on:(keys ["carrier"]) ~each_left:One
      ├ aggregate ~by:["carrier"] ["mean_delay" := mean dep_delay;
      │                            "flights" := rows]
      │ └ filter (dep_delay > 15.)
      │   └ csv "flights.csv" (19 columns) ~columns:["dep_delay"; "carrier"]
      └ parquet "carriers.parquet" (2 columns)
    |}

let answers () =
  let filtered s =
    Query.(
      of_source s
      |> filter Expr.(a > int 1 && b < float 2. && c = string "x")
      |> Kit.head 10)
  in
  expect
    (plans
       [
         ("unsupported", filtered s_unsupported);
         ("exact", filtered s_exact);
         ("inexact", filtered s_inexact);
         ( "exact, inexact, unsupported",
           filtered
             (source
                ~pushdown:(by_column [ ("a", Exact); ("b", Inexact) ])
                "m" columns) );
         ( "exact and unsupported",
           filtered
             (source
                ~pushdown:(by_column [ ("a", Exact); ("c", Exact) ])
                "m" columns) );
       ])
  @@ __POS_OF__
       {|
    # unsupported
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:10
    └ filter (a > 1 && b < 2. && c = "x")
      └ u (5 columns)
    # exact
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:10
    └ e (5 columns) ~filters:[a > 1; b < 2.; c = "x"] ~limit:10
    # inexact
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:10
    └ filter (a > 1 && b < 2. && c = "x")
      └ i (5 columns) ~filters:[a > 1; b < 2.; c = "x"]
    # exact, inexact, unsupported
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:10
    └ filter (b < 2. && c = "x")
      └ m (5 columns) ~filters:[a > 1; b < 2.]
    # exact and unsupported
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:10
    └ filter (b < 2.)
      └ m (5 columns) ~filters:[a > 1; c = "x"]
    |}

let epoch =
  Ext.v ~name:"ymir.epoch" ~ordered:true Type.float64 ~dec:Fun.id ~enc:(fun x ->
      x *. 2.)

let predicates () =
  let filtered p = Query.filter p (of_source s_exact) in
  let timed =
    source ~pushdown:exact "t"
      Type.[ ("t", Any (ext ~name:"ymir.epoch" float64)) ]
  in
  let t = Ext.col epoch "t" in
  expect
    (plans
       Expr.
         [
           ("a literal on the left", filtered (int 3 < a));
           ("is_in", filtered (is_in [ "x"; "y" ] c));
           ("not is_null", filtered (not (is_null b)));
           ("is_null", filtered (is_null b));
           ("or and not", filtered (a = int 1 || not (c = string "x")));
           ("two columns", filtered (a > k));
           ("an expression of a column", filtered (a + int 1 > int 3));
           ("text", filtered (Str.length c > int 2));
           ("an OCaml value", filtered (const positive $ a));
           ("a null literal", filtered (a = null));
           ("not equal, mirrored", filtered (float 2. <> b));
           ("at least, mirrored", filtered (int 3 >= a));
           ("not of a conjunction", filtered (not (a = int 1 && b > float 2.)));
           ( "an extension, encoded",
             Query.filter (t < const 1.5 || is_in [ 2. ] t) (of_source timed) );
         ])
  @@ __POS_OF__
       {|
    # a literal on the left
    query → a int64, b float64, c string, k int8, l list[int64]
    e (5 columns) ~filters:[3 < a]
    # is_in
    query → a int64, b float64, c string, k int8, l list[int64]
    e (5 columns) ~filters:[is_in ["x"; "y"] c]
    # not is_null
    query → a int64, b float64, c string, k int8, l list[int64]
    e (5 columns) ~filters:[not (is_null b)]
    # is_null
    query → a int64, b float64, c string, k int8, l list[int64]
    e (5 columns) ~filters:[is_null b]
    # or and not
    query → a int64, b float64, c string, k int8, l list[int64]
    e (5 columns) ~filters:[a = 1 || not (c = "x")]
    # two columns
    query → a int64, b float64, c string, k int8, l list[int64]
    filter (a > k)
    └ e (5 columns)
    # an expression of a column
    query → a int64, b float64, c string, k int8, l list[int64]
    filter (a + 1 > 3)
    └ e (5 columns)
    # text
    query → a int64, b float64, c string, k int8, l list[int64]
    filter (Str.length c > 2)
    └ e (5 columns)
    # an OCaml value
    query → a int64, b float64, c string, k int8, l list[int64]
    filter (<const> $ a)
    └ e (5 columns)
    # a null literal
    query → a int64, b float64, c string, k int8, l list[int64]
    filter (a = null)
    └ e (5 columns)
    # not equal, mirrored
    query → a int64, b float64, c string, k int8, l list[int64]
    e (5 columns) ~filters:[2. <> b]
    # at least, mirrored
    query → a int64, b float64, c string, k int8, l list[int64]
    e (5 columns) ~filters:[3 >= a]
    # not of a conjunction
    query → a int64, b float64, c string, k int8, l list[int64]
    e (5 columns) ~filters:[not (a = 1 && b > 2.)]
    # an extension, encoded
    query → t ext[ymir.epoch, float64]
    t (1 column) ~filters:[t < <const> || is_in […] t]
    |}

let passes () =
  let p = Expr.(a > int 1) in
  let src = of_source s_unsupported in
  expect
    (plans
       Query.
         [
           ("sort", src |> sort [ Order.asc "b" ] |> filter p);
           ("a filter, merging", src |> filter Expr.(b < float 0.) |> filter p);
           ( "a select renaming",
             src
             |> select Expr.[ "y" := a; keep Sel.(names [ "b" ]) ]
             |> filter Expr.(Col.int "y" > int 1) );
           ( "a derive's other columns",
             src
             |> derive Expr.[ "y" := a + int 1 ]
             |> filter Expr.(b > float 0.) );
           ( "a derive copying",
             src
             |> derive Expr.[ "y" := a ]
             |> filter Expr.(Col.int "y" > int 1) );
           ( "an aggregate's keys",
             src
             |> aggregate ~by:[ "a"; "c" ] Expr.[ "n" := rows ]
             |> filter Expr.(p && c = string "x") );
           ("an unnest", src |> unnest [ "l" ] |> filter p);
           ("an append", src |> append src |> filter p);
           ( "an inner join, both sides",
             src
             |> join ~on:(Join.keys [ "k" ]) (of_source other)
             |> filter Expr.(p && Col.float "w" > float 0.) );
           ( "a semi join, the left side",
             src
             |> join ~kind:Semi ~on:(Join.keys [ "k" ]) (of_source other)
             |> filter p );
           ( "a left join keeps a right conjunct",
             src
             |> join ~kind:Left ~on:(Join.keys [ "k" ]) (of_source other)
             |> filter Expr.(p && Col.float "w" > float 0.) );
         ])
  @@ __POS_OF__
       {|
    # sort
    query → a int64, b float64, c string, k int8, l list[int64]
    sort [asc "b"]
    └ filter (a > 1)
      └ u (5 columns)
    # a filter, merging
    query → a int64, b float64, c string, k int8, l list[int64]
    filter (b < 0. && a > 1)
    └ u (5 columns)
    # a select renaming
    query → y int64, b float64
    select ["y" := a; keep (names ["b"])]
    └ filter (a > 1)
      └ u (5 columns) ~columns:["a"; "b"]
    # a derive's other columns
    query → a int64, b float64, c string, k int8, l list[int64], y int64
    derive ["y" := a + 1]
    └ filter (b > 0.)
      └ u (5 columns)
    # a derive copying
    query → a int64, b float64, c string, k int8, l list[int64], y int64
    derive ["y" := a]
    └ filter (a > 1)
      └ u (5 columns)
    # an aggregate's keys
    query → a int64, c string, n int64
    aggregate ~by:["a"; "c"] ["n" := rows]
    └ filter (a > 1 && c = "x")
      └ u (5 columns) ~columns:["a"; "c"]
    # an unnest
    query → a int64, b float64, c string, k int8, l int64
    unnest ["l"]
    └ filter (a > 1)
      └ u (5 columns)
    # an append
    query → a int64, b float64, c string, k int8, l list[int64]
    append
    ├ #1 filter (a > 1)
    │ └ u (5 columns)
    └ #1
    # an inner join, both sides
    query → a int64, b float64, c string, k int8, l list[int64], w float64
    join ~on:(keys ["k"])
    ├ filter (a > 1)
    │ └ u (5 columns)
    └ filter (w > 0.)
      └ other (2 columns)
    # a semi join, the left side
    query → a int64, b float64, c string, k int8, l list[int64]
    join ~on:(keys ["k"]) ~kind:Semi
    ├ filter (a > 1)
    │ └ u (5 columns)
    └ other (2 columns) ~columns:["k"]
    # a left join keeps a right conjunct
    query → a int64, b float64, c string, k int8, l list[int64], w float64
    filter (w > 0.)
    └ join ~on:(keys ["k"]) ~kind:Left
      ├ filter (a > 1)
      │ └ u (5 columns)
      └ other (2 columns)
    |}

let stops () =
  let p = Expr.(a > int 1) in
  let src = of_source s_unsupported in
  let joined ?kind ?each_left ?(on = Join.keys [ "k" ]) right =
    Query.(
      src
      |> join ?kind ?each_left ~on right
      |> filter Expr.(p && Col.float "w" > float 0.))
  in
  expect
    (plans
       Query.
         [
           ("a slice", src |> Kit.head 3 |> filter p);
           ("a full join", joined ~kind:Full (of_source other));
           ("an assertion", joined ~each_left:One (of_source other));
           ( "position",
             joined ~on:Join.position
               (of_source (source "w" Type.[ ("w", Any float64) ])) );
           ( "closest, right side",
             joined
               ~on:Join.(keys [ "k" ] && closest (ge "a" "t"))
               (of_source
                  (source "t"
                     Type.
                       [ ("k", Any int8); ("t", Any int64); ("w", Any float64) ]))
           );
           ( "a frame-dependent derive",
             src |> derive Expr.[ "s" := over (sum a) ] |> filter p );
           ( "a cumulative ewm",
             src
             |> derive Expr.[ "e" := Kit.cumulative (ewm ~alpha:0.5 b) ]
             |> filter p );
           ( "a frame-dependent filter",
             src |> filter Expr.(b > over (mean b)) |> filter p );
           ( "a frame-dependent filter over an exact source",
             of_source s_exact |> filter Expr.(b > over (mean b)) |> filter p );
           ( "a right column and a key",
             src
             |> join ~on:(Join.keys [ "k" ]) (of_source other)
             |> filter Expr.(Col.float "w" > float 0. || k = int 1) );
           ( "an aggregate's output",
             src
             |> aggregate ~by:[ "c" ] Expr.[ "n" := rows ]
             |> filter Expr.(Col.int "n" > int 1) );
           ( "an aggregate's float key",
             src
             |> aggregate ~by:[ "b" ] Expr.[ "n" := rows ]
             |> filter Expr.(b > float 0.) );
           ( "an aggregate's keys holding floats",
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
                    && is_null (Col.v (Kind.list Kind.int) "li")) );
           ( "an aggregate without keys",
             src
             |> aggregate ~by:[] Expr.[ "n" := rows ]
             |> filter Expr.(Col.int "n" > int 1) );
           ( "a computed column",
             src
             |> select Expr.[ "y" := a + int 1 ]
             |> filter Expr.(Col.int "y" > int 1) );
           ( "an unnested column",
             src |> unnest [ "l" ] |> filter Expr.(Col.int "l" > int 1) );
         ])
  @@ __POS_OF__
       {|
    # a slice
    query → a int64, b float64, c string, k int8, l list[int64]
    filter (a > 1)
    └ slice ~offset:0 ~length:3
      └ u (5 columns) ~limit:3
    # a full join
    query → a int64, b float64, c string, k int8, l list[int64], w float64
    filter (a > 1 && w > 0.)
    └ join ~on:(keys ["k"]) ~kind:Full
      ├ u (5 columns)
      └ other (2 columns)
    # an assertion
    query → a int64, b float64, c string, k int8, l list[int64], w float64
    filter (a > 1 && w > 0.)
    └ join ~on:(keys ["k"]) ~each_left:One
      ├ u (5 columns)
      └ other (2 columns)
    # position
    query → a int64, b float64, c string, k int8, l list[int64], w float64
    filter (a > 1 && w > 0.)
    └ join ~on:position
      ├ u (5 columns)
      └ w (1 column)
    # closest, right side
    query → a int64, b float64, c string, k int8, l list[int64], t int64, w float64
    filter (w > 0.)
    └ join ~on:(keys ["k"] && closest (ge "a" "t"))
      ├ filter (a > 1)
      │ └ u (5 columns)
      └ t (3 columns)
    # a frame-dependent derive
    query → a int64, b float64, c string, k int8, l list[int64], s int64
    filter (a > 1)
    └ derive ["s" := over (sum a)]
      └ u (5 columns)
    # a cumulative ewm
    query → a int64, b float64, c string, k int8, l list[int64], e float64
    filter (a > 1)
    └ derive ["e" := rolling (rows ~before:max_int ~after:0) (ewm ~alpha:0.5 b)]
      └ u (5 columns)
    # a frame-dependent filter
    query → a int64, b float64, c string, k int8, l list[int64]
    filter (a > 1)
    └ filter (b > over (mean b))
      └ u (5 columns)
    # a frame-dependent filter over an exact source
    query → a int64, b float64, c string, k int8, l list[int64]
    filter (a > 1)
    └ filter (b > over (mean b))
      └ e (5 columns)
    # a right column and a key
    query → a int64, b float64, c string, k int8, l list[int64], w float64
    filter (w > 0. || k = 1)
    └ join ~on:(keys ["k"])
      ├ u (5 columns)
      └ other (2 columns)
    # an aggregate's output
    query → c string, n int64
    filter (n > 1)
    └ aggregate ~by:["c"] ["n" := rows]
      └ u (5 columns) ~columns:["c"]
    # an aggregate's float key
    query → b float64, n int64
    filter (b > 0.)
    └ aggregate ~by:["b"] ["n" := rows]
      └ u (5 columns) ~columns:["b"]
    # an aggregate's keys holding floats
    query → lf list[float64], li list[int64], e ext[ymir.epoch, float64], n int64
    filter (is_null lf)
    └ aggregate ~by:["lf"; "li"; "e"] ["n" := rows]
      └ filter (is_null li)
        └ lf (3 columns)
    # an aggregate without keys
    query → n int64
    filter (n > 1)
    └ aggregate ~by:[] ["n" := rows]
      └ u (5 columns) ~columns:[]
    # a computed column
    query → y int64
    filter (y > 1)
    └ select ["y" := a + 1]
      └ u (5 columns) ~columns:["a"]
    # an unnested column
    query → a int64, b float64, c string, k int8, l int64
    filter (l > 1)
    └ unnest ["l"]
      └ u (5 columns)
    |}

let failing () =
  let src = of_source s_unsupported in
  let cast_p = Expr.(cast Type.int8 a = int 1) in
  let user_p = Expr.(const positive $ a) in
  let above f =
    [
      ("above a filter", src |> Query.filter Expr.(b > float 0.) |> f);
      ( "above a join",
        src |> Query.join ~on:(Join.keys [ "k" ]) (of_source other) |> f );
      ("above an unnest", src |> Query.unnest [ "l" ] |> f);
      ( "through a sort and a select",
        src
        |> Query.select Expr.[ keep Sel.(names [ "a"; "b" ]) ]
        |> Query.sort [ Order.asc "b" ]
        |> f );
      ( "through an aggregate's keys and an append",
        src |> Query.append src
        |> Query.aggregate ~by:[ "a" ] Expr.[ "n" := rows ]
        |> f );
    ]
  in
  let titled what = List.map (fun (t, q) -> (what ^ " " ^ t, q)) in
  expect
    (plans
       (titled "a cast" (above (Query.filter cast_p))
       @ titled "a user function" (above (Query.filter user_p))))
  @@ __POS_OF__
       {|
    # a cast above a filter
    query → a int64, b float64, c string, k int8, l list[int64]
    filter (cast int8 a = 1)
    └ filter (b > 0.)
      └ u (5 columns)
    # a cast above a join
    query → a int64, b float64, c string, k int8, l list[int64], w float64
    filter (cast int8 a = 1)
    └ join ~on:(keys ["k"])
      ├ u (5 columns)
      └ other (2 columns)
    # a cast above an unnest
    query → a int64, b float64, c string, k int8, l int64
    filter (cast int8 a = 1)
    └ unnest ["l"]
      └ u (5 columns)
    # a cast through a sort and a select
    query → a int64, b float64
    sort [asc "b"]
    └ filter (cast int8 a = 1)
      └ u (5 columns) ~columns:["a"; "b"]
    # a cast through an aggregate's keys and an append
    query → a int64, n int64
    aggregate ~by:["a"] ["n" := rows]
    └ append
      ├ #1 filter (cast int8 a = 1)
      │ └ u (5 columns) ~columns:["a"]
      └ #1
    # a user function above a filter
    query → a int64, b float64, c string, k int8, l list[int64]
    filter (<const> $ a)
    └ filter (b > 0.)
      └ u (5 columns)
    # a user function above a join
    query → a int64, b float64, c string, k int8, l list[int64], w float64
    filter (<const> $ a)
    └ join ~on:(keys ["k"])
      ├ u (5 columns)
      └ other (2 columns)
    # a user function above an unnest
    query → a int64, b float64, c string, k int8, l int64
    filter (<const> $ a)
    └ unnest ["l"]
      └ u (5 columns)
    # a user function through a sort and a select
    query → a int64, b float64
    sort [asc "b"]
    └ filter (<const> $ a)
      └ u (5 columns) ~columns:["a"; "b"]
    # a user function through an aggregate's keys and an append
    query → a int64, n int64
    aggregate ~by:["a"] ["n" := rows]
    └ append
      ├ #1 filter (<const> $ a)
      │ └ u (5 columns) ~columns:["a"]
      └ #1
    |}

(* A source of the types that the other operations that can fail take. *)
let s_typed =
  source "v"
    (columns
    @ Type.
        [
          ("t", Any (datetime Ns));
          ("dl", Any (list (duration Ns)));
          ("dc", Any (list (decimal ~precision:10 ~scale:2)));
          ("x", Any (tensor Nx.float32 [| 2 |]));
        ])

let t = Col.instant "t"

let can_fail =
  Expr.
    [
      ("Str.parse", Str.parse Type.int64 c = a);
      ("of_option", of_option (option a) = a);
      ("batch", is_null (batch Fun.id (Col.v (Kind.tensor Nx.float32) "x")));
      ("Temporal.add", Temporal.add t (span (Time.Span.days 1)) > t);
    ]

(* [stays_above p] checks that the conjunct [p], which can fail, stays above a
   filter and a join, below which it would meet rows that the plan does not show
   it. *)
let stays_above p =
  let src = of_source s_typed in
  let filtered = Query.filter Expr.(b > float 0.) src
  and joined = Query.join ~on:(Join.keys [ "k" ]) (of_source other) src in
  let check what q =
    equal ~msg:what query_w
      (Query.filter p (Query.optimize q))
      (Query.optimize (Query.filter p q))
  in
  check "above a filter" filtered;
  check "above a join" joined

let renamed () =
  let timed =
    source ~pushdown:exact "t"
      Type.
        [ ("t", Any (ext ~name:"ymir.epoch" float64)); ("l", Any (list int64)) ]
  in
  expect
    (plans
       Query.
         [
           ( "an extension column",
             of_source timed
             |> select Expr.[ "t2" := Ext.col epoch "t" ]
             |> filter Expr.(Ext.col epoch "t2" < const 1.5) );
         ])
  @@ __POS_OF__
       {|
    # an extension column
    query → t2 ext[ymir.epoch, float64]
    select ["t2" := t]
    └ t (2 columns) ~columns:["t"] ~filters:[t < <const>]
    |}

let folds_inside () =
  let src = of_source s_typed in
  let three = Expr.(b +. (float 1. +. float 2.)) in
  expect
    (plans
       Query.
         [
           ( "a window",
             src |> derive Expr.[ "e" := Kit.cumulative (ewm ~alpha:0.5 three) ]
           );
           ( "a calendar operation",
             src
             |> derive
                  Expr.[ "y" := Temporal.format "%Y" (if_ (bool true) t null) ]
           );
           ( "an nx lift",
             src
             |> derive Expr.[ "z" := nx { f = (fun x -> Nx.add x x) } three ] );
         ])
  @@ __POS_OF__
       {|
    # a window
    query → a int64, b float64, c string, k int8, l list[int64], t datetime[ns], dl list[duration[ns]], dc list[decimal[10, 2]], x tensor[float32, 2], e float64
    derive ["e" :=
              rolling (rows ~before:max_int ~after:0) (ewm ~alpha:0.5 (b +. 3.))]
    └ v (9 columns)
    # a calendar operation
    query → a int64, b float64, c string, k int8, l list[int64], t datetime[ns], dl list[duration[ns]], dc list[decimal[10, 2]], x tensor[float32, 2], y string
    derive ["y" := Temporal.format "%Y" t]
    └ v (9 columns)
    # an nx lift
    query → a int64, b float64, c string, k int8, l list[int64], t datetime[ns], dl list[duration[ns]], dc list[decimal[10, 2]], x tensor[float32, 2], z float64
    derive ["z" := add (b +. 3.) (b +. 3.)]
    └ v (9 columns)
    |}

let slices () =
  let src = of_source s_unsupported in
  expect
    (plans
       Query.
         [
           ( "a row-local derive",
             src |> derive Expr.[ "y" := a + int 1 ] |> Kit.head 5 );
           ( "two merged",
             src |> slice ~offset:2 ~length:5 |> slice ~offset:3 ~length:4 );
           ( "from the start of a slice from the end",
             src |> slice ~offset:(-5) ~length:3 |> slice ~offset:1 ~length:1 );
           ("the head of an append", src |> append src |> Kit.head 3);
           ("an append", src |> append src |> slice ~offset:2 ~length:3);
           ( "a source's limit",
             of_source s_exact |> filter Expr.(a > int 1) |> Kit.head 5 );
           ("from the end, with rows", of_source s_rows |> Kit.tail 10);
           ( "from the end, past the start",
             of_source s_rows |> slice ~offset:(-150) ~length:70 );
           ( "from the end, with a conjunct",
             of_source s_rows |> filter Expr.(a > int 1) |> Kit.tail 10 );
           ( "under an inexact conjunct",
             of_source s_inexact |> filter Expr.(a > int 1) |> Kit.head 5 );
           ( "sort, select, slice",
             src
             |> sort [ Order.desc "b" ]
             |> select Expr.[ keep Sel.(names [ "a"; "b" ]) ]
             |> Kit.head 3 );
           ( "a frame-dependent derive",
             src |> derive Expr.[ "r" := over (rank b) ] |> Kit.head 3 );
           ( "above a filter that moves",
             src
             |> select Expr.[ keep Sel.(names [ "a"; "b" ]) ]
             |> filter Expr.(a > int 1)
             |> Kit.head 3 );
         ])
  @@ __POS_OF__
       {|
    # a row-local derive
    query → a int64, b float64, c string, k int8, l list[int64], y int64
    derive ["y" := a + 1]
    └ slice ~offset:0 ~length:5
      └ u (5 columns) ~limit:5
    # two merged
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:5 ~length:2
    └ u (5 columns) ~limit:7
    # from the start of a slice from the end
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:1 ~length:1
    └ slice ~offset:(-5) ~length:3
      └ u (5 columns)
    # the head of an append
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:3
    └ append
      ├ #1 slice ~offset:0 ~length:3
      │ └ u (5 columns) ~limit:3
      └ #1
    # an append
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:2 ~length:3
    └ append
      ├ #1 slice ~offset:0 ~length:5
      │ └ u (5 columns) ~limit:5
      └ #1
    # a source's limit
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:5
    └ e (5 columns) ~filters:[a > 1] ~limit:5
    # from the end, with rows
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:90 ~length:10
    └ r (5 columns, 100 rows) ~limit:100
    # from the end, past the start
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:20
    └ r (5 columns, 100 rows) ~limit:20
    # from the end, with a conjunct
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:(-10) ~length:10
    └ r (5 columns, 100 rows) ~filters:[a > 1]
    # under an inexact conjunct
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:5
    └ filter (a > 1)
      └ i (5 columns) ~filters:[a > 1]
    # sort, select, slice
    query → a int64, b float64
    slice ~offset:0 ~length:3
    └ sort [desc "b"]
      └ u (5 columns) ~columns:["a"; "b"]
    # a frame-dependent derive
    query → a int64, b float64, c string, k int8, l list[int64], r int64
    slice ~offset:0 ~length:3
    └ derive ["r" := over (rank b)]
      └ u (5 columns)
    # above a filter that moves
    query → a int64, b float64
    slice ~offset:0 ~length:3
    └ filter (a > 1)
      └ u (5 columns) ~columns:["a"; "b"]
    |}

let constants () =
  let src = of_source s_exact in
  let f32 = source "f32" Type.[ ("x", Any float32) ] in
  expect
    (plans
       Query.
         [
           ( "integer arithmetic",
             src |> derive Expr.[ "y" := a + (int 2 * int 50) ] );
           ( "a division by zero",
             src |> derive Expr.[ "y" := a + (int 1 / int 0) ] );
           ( "an overflow",
             src |> derive Expr.[ "y" := a + (int max_int + int 1) ] );
           ( "float32 rounding",
             of_source f32
             |> derive Expr.[ "y" := Col.float "x" +. (float 0.1 +. float 0.2) ]
           );
           ("NaN", src |> derive Expr.[ "y" := b +. (float 0. /. float 0.) ]);
           ("a power", src |> derive Expr.[ "y" := b +. (float 2. ** float 3.) ]);
           ( "comparisons",
             src
             |> derive
                  Expr.[ "y" := int 1 < int 2; "z" := string "b" = string "a" ]
           );
           ("a filter of true", src |> filter Expr.(bool true));
           ("a filter of false", src |> filter Expr.(bool false));
           ("a filter of null", src |> filter Expr.(store Type.bool null));
           ("a filter of a bare null", src |> filter Expr.null);
           ("x && true", src |> filter Expr.(a > int 1 && bool true));
           ("x || false", src |> filter Expr.(bool false || a > int 1));
           ( "if_ and coalesce",
             src
             |> derive
                  Expr.
                    [
                      "y" := if_ (bool false) a (coalesce [ null; a; int 3; k ]);
                    ] );
         ])
  @@ __POS_OF__
       {|
    # integer arithmetic
    query → a int64, b float64, c string, k int8, l list[int64], y int64
    derive ["y" := a + 100]
    └ e (5 columns)
    # a division by zero
    query → a int64, b float64, c string, k int8, l list[int64], y int64
    derive ["y" := a + null]
    └ e (5 columns)
    # an overflow
    query → a int64, b float64, c string, k int8, l list[int64], y int64
    derive ["y" := a + (4611686018427387903 + 1)]
    └ e (5 columns)
    # float32 rounding
    query → x float32, y float32
    derive ["y" := x +. 0.30000001192092896]
    └ f32 (1 column)
    # NaN
    query → a int64, b float64, c string, k int8, l list[int64], y float64
    derive ["y" := b +. 0. /. 0.]
    └ e (5 columns)
    # a power
    query → a int64, b float64, c string, k int8, l list[int64], y float64
    derive ["y" := b +. 2. ** 3.]
    └ e (5 columns)
    # comparisons
    query → a int64, b float64, c string, k int8, l list[int64], y bool, z bool
    derive ["y" := true; "z" := false]
    └ e (5 columns)
    # a filter of true
    query → a int64, b float64, c string, k int8, l list[int64]
    e (5 columns)
    # a filter of false
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:0
    └ e (5 columns) ~limit:0
    # a filter of null
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:0
    └ e (5 columns) ~limit:0
    # a filter of a bare null
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:0
    └ e (5 columns) ~limit:0
    # x && true
    query → a int64, b float64, c string, k int8, l list[int64]
    e (5 columns) ~filters:[a > 1]
    # x || false
    query → a int64, b float64, c string, k int8, l list[int64]
    e (5 columns) ~filters:[a > 1]
    # if_ and coalesce
    query → a int64, b float64, c string, k int8, l list[int64], y int64
    derive ["y" := coalesce [a; 3]]
    └ e (5 columns)
    |}

type output = Output : ('a, Expr.row) Expr.t -> output

(* [folded e] is the output [e] of a derive, optimized and printed. *)
let folded (Output e) =
  let src =
    source "f" Type.[ ("a", Any int64); ("k", Any int8); ("h", Any float16) ]
  in
  let printed =
    str Query.pp
      (Query.optimize (Query.derive Expr.[ "y" := e ] (of_source src)))
  in
  let line = List.nth (String.split_on_char '\n' printed) 1 in
  let prefix = {|derive ["y" := |} in
  let start = String.length prefix in
  String.sub line start (String.length line - start - 1)

let folds =
  let a = Col.int "a" and k = Col.int "k" in
  let p e = Output e in
  Expr.
    [
      ("integers subtract", p (int 7 - int 10), "-3");
      ("integers multiply", p (int 3 * int 4), "12");
      ("a quotient truncates", p (int (-7) / int 2), "-3");
      ("a remainder takes the dividend's sign", p (int (-7) mod int 2), "-1");
      ("a remainder by zero is null", p (int 7 mod int 0), "null");
      ( "an overflowing product stays",
        p (int max_int * int 2),
        "4611686018427387903 * 2" );
      ( "an overflowing difference stays",
        p (int min_int - int 1),
        "-4611686018427387904 - 1" );
      ( "an overflowing quotient stays",
        p (int min_int / int (-1)),
        "-4611686018427387904 / -1" );
      ( "a sum its type does not hold stays",
        p (store Type.int8 (int 100) + store Type.int8 (int 100)),
        "100 + 100" );
      ("an operand null is null", p (int 1 + store Type.int64 null), "null");
      ("floats subtract", p (float 1.5 -. float 0.25), "1.25");
      ("floats multiply", p (float 2. *. float 3.), "6.");
      ("a quotient by zero is infinite", p (float 1. /. float 0.), "infinity");
      ( "float16 stays",
        p (Col.float "h" +. (float 0.5 +. float 0.25)),
        "h +. (0.5 +. 0.25)" );
      ("not equal", p (int 1 <> int 2), "true");
      ("at most", p (int 2 <= int 2), "true");
      ("NaN comes after every number", p (float 3. > float Float.nan), "false");
      ("at least", p (string "a" >= string "b"), "false");
      ("a comparison with null", p (int 1 = store Type.int64 null), "null");
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
    ]

let projections () =
  let src = of_source s_unsupported in
  let unpacked =
    src
    |> Query.select Expr.[ "r" := record [ "x" := a; "y" := b; "z" := c ] ]
    |> Query.select Expr.[ unpack (Col.v Record.kind "r") ]
  in
  expect
    (plans
       Query.
         [
           ( "an unused derive output",
             src
             |> derive Expr.[ "y" := a + int 1; "z" := b ]
             |> select Expr.[ keep Sel.(names [ "y" ]) ] );
           ( "an unused aggregate output",
             src
             |> aggregate ~by:[ "c" ] Expr.[ "n" := rows; "s" := sum a ]
             |> select Expr.[ keep Sel.(names [ "c"; "n" ]) ] );
           ("an identity select", src |> select Expr.[ keep Sel.all ]);
           ("counting rows", src |> aggregate ~by:[] Expr.[ "n" := rows ]);
           ( "a filter's column that nothing else reads",
             src
             |> filter Expr.(b > float 0.)
             |> aggregate ~by:[ "c" ] Expr.[ "n" := rows ] );
           ( "an append's inputs",
             src
             |> derive Expr.[ "y" := a + int 1 ]
             |> append (src |> derive Expr.[ "y" := int 2 ])
             |> select Expr.[ keep Sel.(names [ "y" ]) ] );
           ( "a semi join's right column of a left name",
             src
             |> join ~kind:Semi ~on:(Join.keys [ "k" ])
                  (of_source
                     (source "kb" Type.[ ("k", Any int8); ("b", Any float64) ]))
           );
           ( "over's keys",
             src
             |> derive
                  Expr.
                    [ "s" := over ~by:[ "c" ] ~order:[ Order.asc "b" ] (sum a) ]
             |> select Expr.[ keep Sel.(names [ "s" ]) ] );
           ( "a time window's key",
             of_source
               (source "ts"
                  Type.
                    [
                      ("t", Any (datetime ~zone:"UTC" Ns));
                      ("x", Any float64);
                      ("z", Any bool);
                    ])
             |> derive
                  Expr.
                    [
                      "s" :=
                        rolling
                          (Window.time ~before:(Time.Span.days 1) "t")
                          (sum (Col.float "x"));
                    ]
             |> select Expr.[ keep Sel.(names [ "s" ]) ] );
           ( "a replaced column stays in place",
             src |> derive Expr.[ "a" := int 1 ] );
           ( "a replaced column under a select",
             src
             |> derive Expr.[ "a" := k + int 1 ]
             |> select Expr.[ keep Sel.(names [ "a" ]) ] );
           ("an unpacked record", unpacked);
           ( "some fields of an unpacked record",
             unpacked |> select Expr.[ keep Sel.(names [ "x"; "z" ]) ] );
         ])
  @@ __POS_OF__
       {|
    # an unused derive output
    query → y int64
    select [keep (names ["y"])]
    └ derive ["y" := a + 1]
      └ u (5 columns) ~columns:["a"]
    # an unused aggregate output
    query → c string, n int64
    aggregate ~by:["c"] ["n" := rows]
    └ u (5 columns) ~columns:["c"]
    # an identity select
    query → a int64, b float64, c string, k int8, l list[int64]
    u (5 columns)
    # counting rows
    query → n int64
    aggregate ~by:[] ["n" := rows]
    └ u (5 columns) ~columns:[]
    # a filter's column that nothing else reads
    query → c string, n int64
    aggregate ~by:["c"] ["n" := rows]
    └ filter (b > 0.)
      └ u (5 columns) ~columns:["b"; "c"]
    # an append's inputs
    query → y int64
    append
    ├ select [keep (names ["y"])]
    │ └ derive ["y" := a + 1]
    │   └ u (5 columns) ~columns:["a"]
    └ derive ["y" := 2]
      └ u (5 columns) ~columns:[]
    # a semi join's right column of a left name
    query → a int64, b float64, c string, k int8, l list[int64]
    join ~on:(keys ["k"]) ~kind:Semi
    ├ u (5 columns)
    └ kb (2 columns) ~columns:["k"]
    # over's keys
    query → s int64
    select [keep (names ["s"])]
    └ derive ["s" := over ~by:["c"] ~order:[asc "b"] (sum a)]
      └ u (5 columns) ~columns:["a"; "b"; "c"]
    # a time window's key
    query → s float64
    select [keep (names ["s"])]
    └ derive ["s" := rolling (time ~before:24h "t") (sum x)]
      └ ts (3 columns) ~columns:["t"; "x"]
    # a replaced column stays in place
    query → a int64, b float64, c string, k int8, l list[int64]
    derive ["a" := 1]
    └ u (5 columns)
    # a replaced column under a select
    query → a int8
    select [keep (names ["a"])]
    └ derive ["a" := k + 1]
      └ u (5 columns) ~columns:["k"]
    # an unpacked record
    query → x int64, y float64, z string
    select [unpack r]
    └ select ["r" := record ["x" := a; "y" := b; "z" := c]]
      └ u (5 columns) ~columns:["a"; "b"; "c"]
    # some fields of an unpacked record
    query → x int64, z string
    select ["x" := field int "x" r; "z" := field string "z" r]
    └ select ["r" := record ["x" := a; "y" := b; "z" := c]]
      └ u (5 columns) ~columns:["a"; "b"; "c"]
    |}

let sharing () =
  let src = of_source s_unsupported in
  let counts = src |> Query.aggregate ~by:[ "k" ] Expr.[ "n" := rows ] in
  let ex = of_source s_exact in
  expect
    (plans
       Query.
         [
           ( "a self-join",
             counts
             |> join ~on:(Join.keys [ "k" ])
                  (counts
                  |> select
                       Expr.[ keep Sel.(names [ "k" ]); "m" := Col.int "n" ]) );
           ( "complete over one source",
             src
             |> select Expr.[ keep Sel.(names [ "c"; "k"; "a" ]) ]
             |> Kit.complete [ "c"; "k" ] );
           ( "one source, different conjuncts",
             ex
             |> filter Expr.(a > int 1 && b > float 0.)
             |> append (ex |> filter Expr.(a > int 1 && c = string "x")) );
           ( "one source, different columns",
             ex
             |> select Expr.[ keep Sel.(names [ "a"; "k" ]) ]
             |> join ~on:(Join.keys [ "k" ])
                  (ex |> select Expr.[ keep Sel.(names [ "k"; "b" ]) ]) );
           ( "two equal places are one read, with its limit",
             ex |> Kit.head 3 |> append (ex |> Kit.head 3) );
           ("a limit in one of two places", ex |> Kit.head 3 |> append ex);
         ])
  @@ __POS_OF__
       {|
    # a self-join
    query → k int8, n int64, m int64
    join ~on:(keys ["k"])
    ├ #1 aggregate ~by:["k"] ["n" := rows]
    │ └ u (5 columns) ~columns:["k"]
    └ select [keep (names ["k"]); "m" := n]
      └ #1
    # complete over one source
    query → c string, k int8, a int64
    join ~on:(keys ["c"; "k"]) ~kind:Left
    ├ join ~on:all
    │ ├ aggregate ~by:["c"] []
    │ │ └ u (5 columns) ~columns:["c"]
    │ └ aggregate ~by:["k"] []
    │   └ u (5 columns) ~columns:["k"]
    └ select [keep (names ["c"; "k"; "a"])]
      └ u (5 columns) ~columns:["a"; "c"; "k"]
    # one source, different conjuncts
    query → a int64, b float64, c string, k int8, l list[int64]
    append
    ├ e (5 columns) ~filters:[a > 1; b > 0.]
    └ e (5 columns) ~filters:[a > 1; c = "x"]
    # one source, different columns
    query → a int64, k int8, b float64
    join ~on:(keys ["k"])
    ├ e (5 columns) ~columns:["a"; "k"]
    └ select [keep (names ["k"; "b"])]
      └ e (5 columns) ~columns:["b"; "k"]
    # two equal places are one read, with its limit
    query → a int64, b float64, c string, k int8, l list[int64]
    append
    ├ #1 slice ~offset:0 ~length:3
    │ └ e (5 columns) ~limit:3
    └ #1
    # a limit in one of two places
    query → a int64, b float64, c string, k int8, l list[int64]
    append
    ├ slice ~offset:0 ~length:3
    │ └ e (5 columns) ~limit:3
    └ e (5 columns)
    |}

let printed =
  group "printed plans"
    [
      test "the guide's pipeline reads two columns" guide;
      test "a source's answers decide its request" answers;
      test "conjuncts that are predicates go to the source" predicates;
      test "a conjunct passes the steps that keep its rows" passes;
      test "a conjunct stops where its rows would change" stops;
      test "a conjunct that can fail meets only the rows it met" failing;
      cases ~name:fst "a conjunct that can fail stays above a filter and a join"
        can_fail (fun (_, p) -> stays_above p);
      test "a conjunct renamed through a select reads the input's name" renamed;
      test "slices move toward the sources" slices;
      test "constants fold where the literal is exact" constants;
      test "constants fold inside windows, calendar operations and lifts"
        folds_inside;
      test "steps keep the columns read above them" projections;
      cases
        ~name:(fun (what, _, _) -> what)
        "constants fold to what evaluation gives" folds
        (fun (_, e, expected) -> equal string expected (folded e));
      test "equal steps are shared, and each place reads a source of its own"
        sharing;
    ]

(* Laws *)

let w = Col.float "w"

let other_exact =
  source ~pushdown:exact "other_e" Type.[ ("k", Any int8); ("w", Any float64) ]

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
      ("unnest", unnest [ "l" ]);
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
    (fun q i ->
      match snd (List.nth steps i) q with
      | q -> q
      | exception Invalid_argument _ -> q)
    (of_source (snd (List.nth starts start)))
    ops

let pipeline_gen =
  Gen.(
    pair
      (int_range 0 (List.length starts - 1))
      (list ~size:(int_range 0 7) (int_range 0 (List.length steps - 1))))
  |> Gen.with_pp (fun ppf (start, ops) ->
      Format.fprintf ppf "%s: %s"
        (fst (List.nth starts start))
        (String.concat " |> " (List.map (fun i -> fst (List.nth steps i)) ops)))

let same_schema d =
  let q = build d in
  equal schema_w (Query.schema q) (Query.schema (Query.optimize q))

let contains s sub =
  let n = String.length sub in
  let rec at i =
    i + n <= String.length s
    && (String.equal (String.sub s i n) sub || at (i + 1))
  in
  at 0

let idempotent d =
  let o = Query.optimize (build d) in
  let printed = str Query.pp o in
  cover "a source is handed conjuncts" (contains printed "~filters:");
  cover "a source reads fewer columns" (contains printed "~columns:");
  cover "a source gets a limit" (contains printed "~limit:");
  cover "a step is shared" (contains printed "#1");
  cover "a filter stays" (contains printed "filter (");
  equal query_w o (Query.optimize o)

let built_twice d =
  equal query_w (Query.optimize (build d)) (Query.optimize (build d))

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

let answer_gen =
  Gen.of_list
    ~pp:(fun ppf a ->
      Format.pp_print_string ppf
        (match a with
        | Source.Exact -> "Exact"
        | Inexact -> "Inexact"
        | Unsupported -> "Unsupported"))
    [ Source.Exact; Inexact; Unsupported ]

let model_gen =
  Gen.(
    let* chosen = subsequence (List.init (List.length atoms) Fun.id) in
    let* chosen = permutation chosen in
    let* answers = list ~size:(constant 4) answer_gen in
    let+ sliced = bool in
    (chosen, List.combine [ "a"; "b"; "c"; "k" ] answers, sliced))
  |> Gen.with_pp (fun ppf (chosen, answers, sliced) ->
      Format.fprintf ppf "%s; %s%s"
        (String.concat " && "
           (List.map
              (fun i ->
                let _, t, _ = List.nth atoms i in
                t)
              chosen))
        (String.concat ", "
           (List.map
              (fun (c, a) ->
                c ^ ":"
                ^
                match a with
                | Source.Exact -> "E"
                | Inexact -> "I"
                | Unsupported -> "U")
              answers))
        (if sliced then "; sliced" else ""))

let rec pp_pred ppf : Source.Pred.t -> unit = function
  | Cmp (c, _, _) -> Format.fprintf ppf "Cmp %S" c
  | In (c, _) -> Format.fprintf ppf "In %S" c
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

let pushdown_model (chosen, answers, sliced) =
  let asked = ref [] in
  let answer p =
    asked := p :: !asked;
    by_column answers p
  in
  let s = source ~pushdown:answer "m" columns in
  let es = List.map (fun i -> List.nth atoms i) chosen in
  let sliced_by q = if sliced then Kit.head 7 q else q in
  let q =
    sliced_by
      (match es with
      | [] -> of_source s
      | _ ->
          Query.filter (conj (List.map (fun (e, _, _) -> e) es)) (of_source s))
  in
  let optimized = Query.optimize q in
  let answer_of (_, _, p) = by_column answers p in
  let handed = List.filter (fun e -> answer_of e <> Unsupported) es
  and stays = List.filter (fun e -> answer_of e <> Exact) es in
  cover "a conjunct leaves the plan" (List.length stays < List.length es);
  cover "a conjunct stays" (stays <> []);
  cover "the source gets a limit" (sliced && stays = []);
  let model =
    sliced_by
      (match stays with
      | [] -> of_source (source "m" columns)
      | _ ->
          Query.filter
            (conj (List.map (fun (e, _, _) -> e) stays))
            (of_source (source "m" columns)))
  in
  let request =
    (match handed with
      | [] -> ""
      | _ ->
          Format.asprintf " ~filters:[%s]"
            (String.concat "; " (List.map (fun (_, t, _) -> t) handed)))
    ^ if sliced && stays = [] then " ~limit:7" else ""
  in
  let expected =
    let lines = String.split_on_char '\n' (wide Query.pp model) in
    String.concat "\n"
      (List.mapi
         (fun i l -> if i = List.length lines - 1 then l ^ request else l)
         lines)
  in
  equal text expected (wide Query.pp optimized);
  List.iter
    (fun p ->
      mem ~msg:"a predicate the source is asked about is a conjunct's"
        (Testable.make ~pp:pp_pred ~equal:( = ))
        p
        (List.map (fun (_, _, p) -> p) es))
    !asked

let laws =
  group "Query.optimize laws"
    [
      prop "keeps the schema" pipeline_gen same_schema;
      prop "is idempotent" pipeline_gen idempotent;
      prop "gives equal plans for a pipeline built twice" pipeline_gen
        built_twice;
      prop "respects the source's answers" model_gen pushdown_model;
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
         value it stores there, and never about a float16 literal"
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

let completes (s, ks) =
  assume (ks <> []);
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

let kit_plans () =
  let q = of_source s_unsupported in
  let shown =
    [
      ("head", Kit.head 3 q);
      ("tail", Kit.tail 3 q);
      ("top_k", Kit.top_k 2 [ Order.desc "b" ] q);
      ("distinct", Kit.distinct q);
      ("distinct without columns", Kit.distinct (Query.select [] q));
      ("count_by", Kit.count_by [ "c" ] q);
      ("value_counts", Kit.value_counts "c" q);
      ("null_count", Kit.null_count q);
      ("drop", Kit.drop Sel.(names [ "l" ] + prefix "b") q);
      ("rename", Kit.rename [ ("a", "x"); ("k", "key") ] q);
      ("complete", Kit.complete [ "c"; "k" ] q);
      ( "one_hot",
        Kit.one_hot "carrier"
          (of_source
             (source "f"
                Type.
                  [
                    ("x", Any int8); ("carrier", Any category); ("y", Any bool);
                  ])) );
      ( "union",
        Kit.union
          (of_source
             (source "p" Type.[ ("c", Any string); ("t", Any epoch_t) ]))
          q );
    ]
  in
  expect
    (String.concat "\n"
       (List.map (fun (title, q) -> "# " ^ title ^ "\n" ^ str Query.pp q) shown))
  @@ __POS_OF__
       {|
    # head
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:3
    └ u (5 columns)
    # tail
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:(-3) ~length:3
    └ u (5 columns)
    # top_k
    query → a int64, b float64, c string, k int8, l list[int64]
    slice ~offset:0 ~length:2
    └ sort [desc "b"]
      └ u (5 columns)
    # distinct
    query → a int64, b float64, c string, k int8, l list[int64]
    aggregate ~by:["a"; "b"; "c"; "k"; "l"] []
    └ u (5 columns)
    # distinct without columns
    query →
    slice ~offset:0 ~length:1
    └ select []
      └ u (5 columns)
    # count_by
    query → c string, count int64
    aggregate ~by:["c"] ["count" := rows]
    └ u (5 columns)
    # value_counts
    query → c string, count int64
    sort [desc "count"]
    └ aggregate ~by:["c"] ["count" := rows]
      └ u (5 columns)
    # null_count
    query → a int64, b int64, c int64, k int64, l int64
    aggregate ~by:[] ["a" := rows - count a; "b" := rows - count b;
                      "c" := rows - count c; "k" := rows - count k;
                      "l" := rows - count l]
    └ u (5 columns)
    # drop
    query → a int64, c string, k int8
    select [keep (names ["a"; "c"; "k"])]
    └ u (5 columns)
    # rename
    query → x int64, b float64, c string, key int8, l list[int64]
    select ["x" := a; keep (names ["b"; "c"]); "key" := k; keep (names ["l"])]
    └ u (5 columns)
    # complete
    query → a int64, b float64, c string, k int8, l list[int64]
    select [keep (names ["a"; "b"; "c"; "k"; "l"])]
    └ join ~on:(keys ["c"; "k"]) ~kind:Left
      ├ join ~on:all
      │ ├ aggregate ~by:["c"] []
      │ │ └ select [keep (names ["c"])]
      │ │   └ #1 u (5 columns)
      │ └ aggregate ~by:["k"] []
      │   └ select [keep (names ["k"])]
      │     └ #1
      └ #1
    # one_hot
    query → x int8, carrier_AA bool, carrier_B6 bool, carrier_DL bool, y bool
    select [keep (names ["x"]); "carrier_AA" := carrier = "AA";
            "carrier_B6" := carrier = "B6"; "carrier_DL" := carrier = "DL";
            keep (names ["y"])]
    └ f (3 columns)
    # union
    query → a int64, b float64, c string, k int8, l list[int64], t ext[ymir.epoch, float64]
    append
    ├ derive ["t" := store ext[ymir.epoch, float64] null]
    │ └ u (5 columns)
    └ derive ["a" := store int64 null; "b" := store float64 null;
              "k" := store int8 null; "l" := store list[int64] null]
      └ p (2 columns)
    |}

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
      prop "complete keeps the schema" subset_gen completes;
      prop "one_hot puts one bool per category in place" one_hot_gen one_hots;
      test "each query prints as its verbs" kit_plans;
      test "describe has one row per numeric column" describe;
    ]

(* Kit: expressions *)

let kit_expressions () =
  let typed =
    source "t"
      Type.
        [
          ("x8", Any int8);
          ("f32", Any float32);
          ("f", Any float64);
          ("price", Any (decimal ~precision:10 ~scale:2));
          ("d", Any (duration Ms));
          ("g", Any string);
          ("t", Any epoch_t);
          ("score", Any float64);
        ]
  in
  let derived os = Query.schema (Query.derive os (of_source typed)) in
  let column n = Col.v Kind.float n in
  equal schema_w
    (Schema.v
       (Schema.columns (Query.schema (of_source typed))
       @ Type.
           [
             ("i", Any int64);
             ("cs8", Any int64);
             ("cs32", Any float32);
             ("csd", Any (decimal ~precision:18 ~scale:2));
             ("csms", Any (duration Ms));
             ("cm", Any float32);
             ("cc", Any int64);
             ("ff", Any string);
             ("fft", Any epoch_t);
             ("e", Any float64);
           ]))
    (derived
       Expr.
         [
           "i" := Kit.index;
           "cs8" := Kit.cumulative (sum (Col.int "x8"));
           "cs32" := Kit.cumulative (sum (column "f32"));
           "csd" := Kit.cumulative (sum (Col.decimal "price"));
           "csms" := Kit.cumulative (sum (Col.span "d"));
           "cm" := Kit.cumulative (max (column "f32"));
           "cc" := Kit.cumulative (count (Col.string "g"));
           "ff" := Kit.fill_forward (Col.string "g");
           each
             Sel.(names [ "t" ])
             { column = (fun _ x -> "fft" := Kit.fill_forward x) };
           "e" := Kit.cumulative (ewm ~alpha:0.5 (Col.int "x8"));
         ]);
  equal schema_w
    (Schema.v Type.[ ("g", Any string); ("best", Any float32) ])
    (Query.schema
       (Query.aggregate ~by:[ "g" ]
          Expr.
            [ "best" := Kit.arg (arg_max (Col.float "score")) (column "f32") ]
          (of_source typed)))

let kit_printed () =
  let x = Col.float "x" in
  expect
    (String.concat "\n"
       (str Expr.pp Kit.index
        :: List.map (str Expr.pp)
             [ Kit.cumulative (Expr.sum x); Kit.fill_forward x ]
       @ [ str Expr.pp (Kit.arg (Expr.arg_max x) (Col.string "name")) ]))
  @@ __POS_OF__
       {|
    rolling (rows ~before:max_int ~after:0) rows - 1
    rolling (rows ~before:max_int ~after:0) (sum x)
    rolling (rows ~before:max_int ~after:0) (last x)
    first
      (if_ (rolling (rows ~before:max_int ~after:0) rows - 1 = over (arg_max x))
         name null)
    |}

let kit_errors () =
  let q = of_source s_unsupported in
  let ext = of_source (source "x" Type.[ ("t", Any epoch_t) ]) in
  expect
    (String.concat "\n\n"
       (List.map message
          [
            (fun () -> Kit.rename [ ("a", "x"); ("a", "y") ] q);
            (fun () -> Kit.rename [ ("aa", "x") ] q);
            (fun () -> Kit.rename [ ("a", "b") ] q);
            (fun () -> Kit.complete [] q);
            (fun () -> Kit.complete [ "c"; "k"; "c" ] q);
            (fun () -> Kit.one_hot "cc" q);
            (fun () -> Kit.one_hot "c" q);
            (fun () -> Query.derive Expr.[ "s" := Kit.cumulative (sum c) ] q);
            (fun () ->
              Query.derive
                Expr.
                  [
                    each Sel.all
                      { column = (fun n x -> n := Kit.cumulative (max x)) };
                  ]
                ext);
          ]))
  @@ __POS_OF__
       {|
    Kit.rename: "a" is renamed twice

    select: 1 problem
      each (all + names ["aa"]) <fn>
        no column "aa". Did you mean "a"?
      input (5 columns): a int64, b float64, c string, k int8, l list[int64]

    select: 1 problem
      the output "b" appears twice.
      input (5 columns): a int64, b float64, c string, k int8, l list[int64]

    Kit.complete: no column

    Kit.complete: "c" is named twice

    select: 1 problem
      keep (names ["cc"])
        no column "cc". Did you mean "c"?
      input (5 columns): a int64, b float64, c string, k int8, l list[int64]

    Kit.one_hot: "c" is string, not categorical: cast it to a categorical type first

    derive: 1 problem
      "s" := rolling (rows ~before:max_int ~after:0) (sum c)
        sum takes integers, floats, durations or decimals, not string.
      input (5 columns): a int64, b float64, c string, k int8, l list[int64]

    derive: 1 problem
      each all <fn>
        max orders values, and ext[ymir.epoch, float64] is an extension read without its declaration: read it with an Ext.t declared ~ordered:true.
      input (1 column): t ext[ymir.epoch, float64]
    |}

let kit_expressions_group =
  group "Kit expressions"
    [
      test "each takes the type its definition gives" kit_expressions;
      test "each prints as its definition" kit_printed;
      test "mistakes raise or are their verbs' reports" kit_errors;
    ]

let () =
  exit
    (run "plan"
       [ printed; laws; comparing; kit_queries; kit_expressions_group ])
