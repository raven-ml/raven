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

(* [messages fs] is the messages of [fs], separated by blank lines. *)
let messages fs = String.concat "\n\n" (List.map message fs)

let source ?rows name columns =
  Source.v ~name ~schema:(Schema.v columns) ?rows (fun _ -> Ok [])

let schema_w = Testable.make ~pp:Schema.pp ~equal:Schema.equal
let query_w = Testable.make ~pp:Query.pp ~equal:Query.equal

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
      ("time_hour", Any (datetime ~zone:"UTC" Us));
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
        ("ts", Any (datetime ~zone:"UTC" Ns));
        ("t", Any epoch_t);
        ("l", Any (list int64));
        ("r", Any (record [ ("e", Any epoch_t) ]));
        ("wait", Any (duration Ms));
      ]

let late_by_carrier () =
  Query.(
    of_source flights
    |> filter Expr.(delay > float 15.)
    |> aggregate ~by:[ "carrier" ]
         Expr.[ "mean_delay" := mean delay; "flights" := rows ]
    |> join ~on:(Join.keys [ "carrier" ]) ~each_left:One (of_source carriers)
    |> sort [ Order.desc "mean_delay" ])

(* The guide *)

let guide_report () =
  equal text
    {|aggregate: 3 problems
  ~by: no column "carier". Did you mean "carrier"?
  "mean_delay" := mean dep_dly
    no column "dep_dly". Did you mean "dep_delay"?
  "late" := mean carrier
    Col.float reads float16, float32 or float64, but "carrier" is string.
  input (19 columns): year int16, month int8, day int8, dep_time int32, sched_dep_time int32, dep_delay float64, arr_time int32, sched_arr_time int32, …|}
    (message (fun () ->
         Query.(
           of_source flights
           |> aggregate ~by:[ "carier" ]
                Expr.
                  [
                    "mean_delay" := mean (Col.float "dep_dly");
                    "late" := mean (Col.float "carrier");
                  ])))

let guide_plan () =
  expect (str Query.pp (late_by_carrier ()))
  @@ __POS_OF__
       {|
    query → carrier string, mean_delay float64, flights int64, name string
    sort [desc "mean_delay"]
    └ join ~on:(keys ["carrier"]) ~each_left:One
      ├ aggregate ~by:["carrier"] ["mean_delay" := mean dep_delay;
      │                            "flights" := rows]
      │ └ filter (dep_delay > 15.)
      │   └ csv "flights.csv" (19 columns)
      └ parquet "carriers.parquet" (2 columns)
    |}

let guide =
  group "guide"
    [
      test "reports every problem of the aggregate at once" guide_report;
      test "prints the pipeline as its verbs are written" guide_plan;
      test "resolves the pipeline's schema" (fun () ->
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
    ]

(* Reports *)

let of_kinds = Query.of_source kinds

let binding () =
  let derive os () = Query.derive os of_kinds in
  let aggregate os () = Query.aggregate ~by:[] os of_kinds in
  let x8 = Col.int "x8" and u8 = Col.int "u8" and n = Col.int "n" in
  let f = Col.float "f" in
  expect
    (messages
       Expr.
         [
           derive [ "k" := x8 + int 300 ];
           derive [ "k" := x8 + (int 100 + int 100) ];
           derive [ "k" := x8 + u8 ];
           derive [ "k" := store Type.int8 (const 1000) ];
           aggregate [ "k" := sum (const (fun v -> v) $ f) ];
           derive [ "k" := null ];
           derive [ "k" := const 1 ];
           derive [ "k" := if_ (Col.bool "b") (const 1) (const 2) ];
           derive [ "k" := const (fun a b -> (a, b)) $ n $ option f ];
           derive [ across Kind.float Sel.all (fun nm x -> nm := x) ];
           derive
             [
               each
                 Sel.(names [ "t" ])
                 { column = (fun nm x -> nm := over (min x)) };
             ];
           derive [ keep Sel.(names [ "fx" ]) ];
           derive [ "k" := Col.float "n" +. f ];
           derive [ "k" := Col.float "nope" +. f +. Col.float "nope" ];
           derive [ "k" := over (min (Ext.col epoch "x8")) ];
           aggregate [ "k" := ewm ~alpha:0.5 (Ext.col epoch "t") ];
           derive [ "k" := over (min (Col.v Record.kind "r")) ];
           derive [ "k" := over ~order:[ Order.asc "t" ] (rank n) ];
           derive
             [ "k" := over ~order:[ Order.asc "n"; Order.desc "n" ] (rank n) ];
           derive [ "k" := over ~by:[ "g"; "nope"; "g"; "nope" ] (sum n) ];
           derive [ "k" := nx { f = (fun a -> Nx.sum a) } f ];
           derive [ "k" := nx { f = Nx.zeros_like } f ];
           derive [ "k" := nx { f = Nx.exp } (Col.span "wait") ];
           derive [ "k" := if_ (Col.int "nope" > int 1) x8 u8 ];
           derive [ "k" := Temporal.field `Hour (Col.date "d") ];
           derive [ "k" := cast Type.string (Col.date "d") ];
         ])
  @@ __POS_OF__
       {|
    derive: 1 problem
      "k" := x8 + 300
        int8 does not hold the literal 300.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := x8 + (100 + 100)
        int8 does not hold the constant 200.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := x8 + u8
        int8 and uint8 do not meet: cast first.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := store int8 <const>
        int8 does not hold the value 1000.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    aggregate: 1 problem
      "k" := sum (<const> $ f)
        <const> $ f has no type: give it one with store.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := null
        null has no type: give it one with store.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := <const>
        <const> has no type: give it one with store.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := if_ b <const> <const>
        if_ b <const> <const> has no type: give it one with store.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := <const> $ n $ option f
        <const> $ n $ option f has no type: give it one with store.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 11 problems
      across float all <fn>
        across float: "x8" is int8, which float does not read: narrow the selector with Sel.of_kind.
        across float: "u8" is uint8, which float does not read: narrow the selector with Sel.of_kind.
        across float: "n" is int64, which float does not read: narrow the selector with Sel.of_kind.
        across float: "g" is string, which float does not read: narrow the selector with Sel.of_kind.
        across float: "d" is date, which float does not read: narrow the selector with Sel.of_kind.
        across float: "b" is bool, which float does not read: narrow the selector with Sel.of_kind.
        across float: "ts" is datetime[ns, UTC], which float does not read: narrow the selector with Sel.of_kind.
        across float: "t" is ext[ymir.epoch, float64], which float does not read: narrow the selector with Sel.of_kind.
        across float: "l" is list[int64], which float does not read: narrow the selector with Sel.of_kind.
        across float: "r" is record[e ext[ymir.epoch, float64]], which float does not read: narrow the selector with Sel.of_kind.
        across float: "wait" is duration[ms], which float does not read: narrow the selector with Sel.of_kind.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      each (names ["t"]) <fn>
        min orders values, and ext[ymir.epoch, float64] is an extension read without its declaration: read it with an Ext.t declared ~ordered:true.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      keep (names ["fx"])
        no column "fx". Did you mean "f"?
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := n +. f
        Col.float reads float16, float32 or float64, but "n" is int64.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := nope +. f +. nope
        no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := over (min x8)
        Ext.col binds ext[ymir.epoch, float64], but "x8" is int8.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    aggregate: 1 problem
      "k" := ewm ~alpha:0.5 t
        ewm takes integers or floats, not ext[ymir.epoch, float64]: use Ext.storage.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := over (min r)
        min orders values, and record[e ext[ymir.epoch, float64]] holds an extension type and has no order.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := over ~order:[asc "t"] (rank n)
        "t" is ext[ymir.epoch, float64], which orders only through its declaration: order by its storage, derived first.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := over ~order:[asc "n"; desc "n"] (rank n)
        the column "n" is already a key.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 3 problems
      "k" := over ~by:["g"; "nope"; "g"; "nope"] (sum n)
        over ~by: "g" is named twice.
        no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
        over ~by: "nope" is named twice.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := nx <fn> f
        the function of nx performs sum, which is not elementwise.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := nx <fn> f
        the function of nx ignores its argument.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := nx <fn> wait
        nx takes integer, float and boolean operands, not duration[ms].
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 2 problems
      "k" := if_ (nope > 1) x8 u8
        no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
        int8 and uint8 do not meet: cast first.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := Temporal.field `Hour d
        a date has no `Hour.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      "k" := cast string d
        cast does not convert date to string: use Temporal.format.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
    |}

(* [of_columns n] is a query of the first [n] columns of the flights. *)
let of_columns n =
  Query.of_source
    (source "first" (List.filteri (fun i _ -> i < n) flights_columns))

let verbs () =
  let x = Col.float "f" in
  let rest =
    let column = function
      | "x8", _ -> Some ("x8", Type.Any Type.int16)
      | "wait", _ -> None
      | c -> Some c
    in
    source "rest"
      (List.filter_map column (Schema.columns (Query.schema of_kinds))
      @ [ ("extra", Type.Any Type.bool) ])
  in
  expect
    (messages
       Query.
         [
           (fun () -> select Expr.[ "a" := x; "a" := x +. float 1. ] of_kinds);
           (fun () ->
             select Expr.[ "a" := Col.float "nope"; "a" := x ] of_kinds);
           (fun () ->
             derive Expr.[ "a" := x; keep Sel.(names [ "a"; "f" ]) ] of_kinds);
           (fun () ->
             derive Expr.[ "f" := x; keep Sel.(names [ "f" ]) ] of_kinds);
           (fun () ->
             aggregate ~by:[ "g"; "n"; "g" ] Expr.[ "n" := sum x ] of_kinds);
           (fun () ->
             aggregate ~by:[ "g" ]
               Expr.[ "g" := sum (Col.float "nope") ]
               of_kinds);
           (fun () ->
             aggregate ~by:[ "gg"; "gg" ] Expr.[ "k" := rows ] of_kinds);
           (fun () -> filter Expr.(Col.int "f" > int 1) of_kinds);
           (fun () -> filter Expr.(x > float 1. && Col.bool "nope") of_kinds);
           (fun () ->
             sort
               Order.
                 [
                   asc "t";
                   desc "r";
                   asc "nope";
                   asc "f";
                   nulls_first (desc "f");
                 ]
               of_kinds);
           (fun () -> sort Order.[ asc "t"; desc "t" ] of_kinds);
           (fun () -> slice ~offset:0 ~length:(-1) of_kinds);
           (fun () -> slice ~offset:0 ~length:(-1) (of_columns 8));
           (fun () -> slice ~offset:0 ~length:(-1) (of_columns 1));
           (fun () -> slice ~offset:0 ~length:(-1) (of_columns 0));
           (fun () -> of_kinds |> append (of_source rest));
           (fun () -> unnest [] of_kinds);
           (fun () -> unnest [ "l"; "f"; "nope"; "l" ] of_kinds);
           (fun () -> unnest [ "nope"; "nope" ] of_kinds);
         ])
  @@ __POS_OF__
       {|
    select: 1 problem
      the output "a" appears twice.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    select: 2 problems
      "a" := nope
        no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
      the output "a" appears twice.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      keep (names ["a"; "f"])
        no column "a". Did you mean "f", "n", "g", "d", "b", "t", "l" or "r"?
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    derive: 1 problem
      the output "f" appears twice.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    aggregate: 2 problems
      ~by: "g" is named twice.
      the output "n" has the name of a key.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    aggregate: 2 problems
      "g" := sum nope
        no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
      the output "g" has the name of a key.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    aggregate: 2 problems
      ~by: no column "gg". Did you mean "g"?
      ~by: "gg" is named twice.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    filter: 1 problem
      f > 1
        Col.int reads int8 to int64 and uint8 to uint64, but "f" is float64.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    filter: 1 problem
      f > 1. && nope
        no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    sort: 4 problems
      asc "t": "t" is ext[ymir.epoch, float64], which orders only through its declaration: order by its storage, derived first. For example: derive [ "k" := Ext.storage e (Ext.col e "t") ] |> sort [ asc "k" ] |> select [ keep Sel.(all - names [ "k" ]) ].
      desc "r": "r" is record[e ext[ymir.epoch, float64]], which holds an extension type and has no order.
      asc "nope": no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
      nulls_first (desc "f"): the column "f" is already a key.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    sort: 2 problems
      asc "t": "t" is ext[ymir.epoch, float64], which orders only through its declaration: order by its storage, derived first. For example: derive [ "k" := Ext.storage e (Ext.col e "t") ] |> sort [ asc "k" ] |> select [ keep Sel.(all - names [ "k" ]) ].
      desc "t": the column "t" is already a key.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    slice: 1 problem
      the length -1 is negative.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    slice: 1 problem
      the length -1 is negative.
      input (8 columns): year int16, month int8, day int8, dep_time int32, sched_dep_time int32, dep_delay float64, arr_time int32, sched_arr_time int32

    slice: 1 problem
      the length -1 is negative.
      input (1 column): year int16

    slice: 1 problem
      the length -1 is negative.
      input (0 columns)

    append: 3 problems
      "x8" is int8 in the input and int16 in rest.
      "wait" (duration[ms]) is not in rest.
      "extra" (bool) is only in rest.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      rest (13 columns): x8 int16, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    unnest: 1 problem
      no column to unnest.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    unnest: 3 problems
      "f" is float64, not a list.
      no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
      "l" is named twice.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …

    unnest: 2 problems
      no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
      "nope" is named twice.
      input (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
    |}

let joins () =
  let other =
    Query.of_source
      (source "other"
         Type.
           [
             ("y8", Any int8);
             ("g2", Any string);
             ("t2", Any epoch_t);
             ("d2", Any date);
             ("ts2", Any (datetime ~zone:"UTC" Ns));
             ("f2", Any float64);
             ("t3", Any (ext ~name:"ymir.epoch" ~metadata:"v2" float64));
             ("r2", Any (record [ ("e", Any epoch_t) ]));
           ])
  in
  let shared =
    Query.of_source
      (source "shared"
         Type.[ ("y8", Any int8); ("f", Any float64); ("n", Any int64) ])
  in
  let shared_f =
    Query.of_source
      (source "shared_f" Type.[ ("y8", Any int8); ("f", Any float64) ])
  in
  let join ?kind ?(right = other) on () = Query.join ?kind ~on right of_kinds in
  expect
    (messages
       Join.
         [
           join (keys [ "nope" ]);
           join (eq "g" "y8");
           join (eq "t" "t2" && lt "x8" "y8");
           join (lt "t" "t2");
           join (closest ~within:(Col.float "f") (ge "f" "f2"));
           join (closest ~within:Expr.(float (-1.)) (ge "f" "f2"));
           join (closest ~within:Expr.(int 1) (ge "g" "g2"));
           join (closest ~within:Expr.(int 5) (ge "ts" "ts2"));
           join (closest ~within:Expr.(span (Time.Span.hours 1)) (ge "d" "d2"));
           join (nearest "g" "g2");
           join (eq "t" "t3");
           join (eq "f" "t2");
           join (closest ~within:Expr.(float Float.nan) (ge "f" "f2"));
           join
             (closest
                ~within:
                  Expr.(date (Option.get (Time.Date.of_civil (2024, 1, 1))))
                (ge "d" "d2"));
           join (lt "r" "r2");
           join (closest ~within:(Col.float "f") (ge "nope" "f2"));
           join (closest ~within:Expr.(int 300) (ge "x8" "y8"));
           join (closest ~within:Expr.(span (Time.Span.s (-5))) (ge "ts" "ts2"));
           join (nearest ~within:Expr.(float (-1.)) "f" "f2");
           join (nearest ~within:Expr.(float 0.5) "f" "f2");
           join (nearest ~within:Expr.(int 1) "g" "g2");
           join ~right:shared_f (eq "x8" "y8");
           join ~kind:Full (eq "x8" "y8" && eq "x8" "f2");
           join ~right:shared (eq "x8" "y8");
           join ~right:shared ~kind:Semi (eq "x8" "y8");
         ])
  @@ __POS_OF__
       {|
    join: 2 problems
      left: no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
      right: no column "nope". The columns are "y8", "g2", "t2", "d2", "ts2", "f2", "t3" and "r2".
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      "g" is string and "y8" is int8, which do not meet: cast one first.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    no exception

    join: 1 problem
      "t" and "t2" are ext[ymir.epoch, float64], which orders only through its declaration: compare their storage, derived first.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      ~within:f is not a literal.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      ~within:(-1.) is not at least zero.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      ~within:1: "g" and "g2" are string, which has no difference.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      ~within:5 is int, where the difference of "ts" and "ts2" is duration[ns].
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      ~within:1h is not a whole number of days.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      nearest "g" "g2": string has no difference.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      "t" is ext[ymir.epoch, float64] and "t3" is ext[ymir.epoch "v2", float64], which do not meet: cast one first.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      "f" is float64 and "t2" is ext[ymir.epoch, float64], which do not meet: cast one first.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      ~within:nan is not at least zero.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      ~within:2024-01-01 is date, where the difference of "d" and "d2" is duration[s].
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      "r" and "r2" are record[e ext[ymir.epoch, float64]], which holds an extension type and has no order.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 2 problems
      left: no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
      ~within:f is not a literal.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      ~within:300: int8 does not hold it.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      ~within:(-5s) is not at least zero.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      ~within:(-1.) is not at least zero.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    no exception

    join: 1 problem
      nearest "g" "g2": string has no difference.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      "f" is on both sides: rename one side first.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (2 columns): y8 int8, f float64

    join: 2 problems
      "x8" is int8 and "f2" is float64, which do not meet: cast one first.
      "x8" is the left of two equality atoms, so a Full join cannot coalesce it.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (8 columns): y8 int8, g2 string, t2 ext[ymir.epoch, float64], d2 date, ts2 datetime[ns, UTC], f2 float64, t3 ext[ymir.epoch "v2", float64], r2 record[e ext[ymir.epoch, float64]]

    join: 1 problem
      "f", "n" are on both sides: rename one side first.
      left (13 columns): x8 int8, u8 uint8, f float64, f32 float32, n int64, g string, d date, b bool, …
      right (3 columns): y8 int8, f float64, n int64

    no exception
    |}

(* A value outside a dictionary has no order: binding reports it, each time,
   instead of comparing it. Repeated record fields are reported in the order
   they repeat. *)
let values_and_fields () =
  let q =
    Query.of_source
      (source "cats"
         Type.
           [
             ("c", Any (categorical [| "a"; "b" |]));
             ("x", Any int64);
             ("y", Any int64);
           ])
  in
  let c = Col.string "c" and x = Col.int "x" and y = Col.int "y" in
  let zzz () = String.make 3 'z' in
  let held () = Query.filter Expr.(is_in [ zzz () ] c) q in
  expect
    (messages
       Expr.
         [
           held;
           held;
           (fun () -> Query.derive [ "k" := cut [| zzz (); "a" |] c ] q);
           (fun () ->
             Query.derive
               [ "r" := record [ "b" := x; "a" := y; "b" := y; "a" := x ] ]
               q);
         ])
  @@ __POS_OF__
       {|
    filter: 1 problem
      is_in […] c
        categorical["a", "b"] does not hold the value "zzz".
      input (3 columns): c categorical["a", "b"], x int64, y int64

    filter: 1 problem
      is_in […] c
        categorical["a", "b"] does not hold the value "zzz".
      input (3 columns): c categorical["a", "b"], x int64, y int64

    derive: 1 problem
      "k" := cut […] c
        categorical["a", "b"] does not hold the edge "zzz".
      input (3 columns): c categorical["a", "b"], x int64, y int64

    derive: 2 problems
      "r" := record ["b" := x; "a" := y; "b" := y; "a" := x]
        record: the output "b" appears twice.
        record: the output "a" appears twice.
      input (3 columns): c categorical["a", "b"], x int64, y int64
    |}

let reports =
  group "reports"
    [
      test "binding problems, in each verb's report" binding;
      test "problems of each verb's arguments" verbs;
      test "problems of join conditions" joins;
      test "values outside a dictionary and repeated record fields"
        values_and_fields;
    ]

(* Join conditions *)

let rejected_conditions () =
  expect
    (messages
       Join.
         [
           (fun () -> position && keys [ "a" ]);
           (fun () -> closest (ge "a" "b") && nearest "c" "d");
           (fun () -> closest (ge "a" "b") && lt "c" "d");
           (fun () -> ge "ts" "start" && lt "ts" "end");
           (fun () -> closest (keys [ "a" ]));
           (fun () -> keys []);
           (fun () -> keys [ "a"; "b"; "a" ]);
           (fun () -> keys [ "a" ] && keys [ "a" ]);
           (fun () -> keys [ "a"; "b" ] && position);
           (fun () -> eq "a" "b" && keys [ "c" ] && eq "a" "b");
         ])
  @@ __POS_OF__
       {|
    Join.( && ): position && keys ["a"]: position joins row i with row i and takes no other atom

    Join.( && ): closest (ge "a" "b") && nearest "c" "d": a join takes one closest or nearest atom

    Join.( && ): closest (ge "a" "b") && lt "c" "d": a closest or nearest join takes no inequality: filter after it

    Join.( && ): ge "ts" "start" && lt "ts" "end": an inequality join compares to one right column: join on one, then filter

    Join.closest: keys ["a"] is not one inequality atom

    Join.keys: no key; a join on no key is Join.all

    Join.keys: "a" is named twice

    Join.( && ): keys ["a"] && keys ["a"]: keys ["a"] appears twice

    Join.( && ): keys ["a"; "b"] && position: position joins row i with row i and takes no other atom

    Join.( && ): eq "a" "b" && keys ["c"] && eq "a" "b": eq "a" "b" appears twice
    |}

(* [semi_on c] is the semi join on [c] of two fixed sources. *)
let semi_on =
  let columns = Type.[ ("a", Any int64); ("b", Any int64) ] in
  let l = Query.of_source (source "l" columns)
  and r = Query.of_source (source "r" columns) in
  fun c -> Query.join ~kind:Semi ~on:c r l

(* [atom_variants] pairs conditions that differ in one part of one atom. *)
let atom_variants =
  let five = Expr.int 5 and six = Expr.int 6 in
  Join.
    [
      ("the left columns of eq", eq "a" "b", eq "b" "b");
      ("the right columns of eq", eq "a" "a", eq "a" "b");
      ("lt and le", lt "a" "b", le "a" "b");
      ("gt and ge", gt "a" "b", ge "a" "b");
      ("lt and gt", lt "a" "b", gt "a" "b");
      ("the left columns of an inequality", lt "a" "b", lt "b" "b");
      ("the right columns of an inequality", lt "a" "a", lt "a" "b");
      ("the orders of closest", closest (ge "a" "b"), closest (gt "a" "b"));
      ("the left columns of closest", closest (ge "a" "b"), closest (ge "b" "b"));
      ( "the right columns of closest",
        closest (ge "a" "a"),
        closest (ge "a" "b") );
      ( "closest with and without within",
        closest (ge "a" "b"),
        closest ~within:five (ge "a" "b") );
      ( "the withins of closest",
        closest ~within:five (ge "a" "b"),
        closest ~within:six (ge "a" "b") );
      ("closest and nearest", closest (ge "a" "b"), nearest "a" "b");
      ("the left columns of nearest", nearest "a" "b", nearest "b" "b");
      ("the right columns of nearest", nearest "a" "a", nearest "a" "b");
      ( "nearest with and without within",
        nearest "a" "b",
        nearest ~within:five "a" "b" );
      ( "the withins of nearest",
        nearest ~within:five "a" "b",
        nearest ~within:six "a" "b" );
      ("position and all", position, all);
      ("position and keys", position, keys [ "a" ]);
      ("one atom and two", keys [ "a" ], keys [ "a"; "b" ]);
    ]

let conditions =
  group "Join conditions"
    [
      test "refuses each conjunction that no algorithm runs" rejected_conditions;
      test "accepts a range on one right column" (fun () ->
          ignore Join.(ge "ts" "start" && lt "ts2" "start" && keys [ "k" ]));
      test "accepts two equality atoms on one left column" (fun () ->
          ignore Join.(eq "a" "b" && eq "a" "c"));
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
      test "the order of atoms matters" (fun () ->
          not_equal query_w
            (semi_on (Join.keys [ "a"; "b" ]))
            (semi_on (Join.keys [ "b"; "a" ])));
      cases
        ~name:(fun (what, _, _) -> Printf.sprintf "tells apart %s" what)
        "Join.equal" atom_variants
        (fun (_, c0, c1) ->
          equal query_w (semi_on c0) (semi_on c0);
          not_equal query_w (semi_on c0) (semi_on c1));
    ]

(* Schema resolution *)

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

let has_ext (Type.Any t) =
  List.exists (fun (Type.Any e) -> Type.equal e t) ext_types

let keeps s =
  let q = of_schema s in
  let sortable =
    List.filter_map
      (fun (n, t) -> if has_ext t then None else Some (Order.asc n))
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

let subset_gen =
  Gen.(
    let* s = schema_gen in
    let* ns = permutation (List.map fst (Schema.columns s)) in
    let+ ns = subsequence ns in
    (s, ns))

let select_names (s, ns) =
  equal (list string) ns
    (List.map fst
       (Schema.columns
          (Query.schema
             (Query.select Expr.[ keep Sel.(names ns) ] (of_schema s)))))

let aggregate_by (s, by) =
  let key n = (n, Option.get (Schema.find s n)) in
  equal schema_w
    (Schema.v (List.map key by @ [ ("rows_", Type.Any Type.int64) ]))
    (Query.schema (Query.aggregate ~by Expr.[ "rows_" := rows ] (of_schema s)))

let unnest_lists s =
  let lists =
    List.filter_map
      (fun (n, Type.Any t) -> match t with Type.List _ -> Some n | _ -> None)
      (Schema.columns s)
  in
  cover "has a list column" (lists <> []);
  if lists <> [] then
    let element (n, (Type.Any t as a)) =
      match t with Type.List e -> (n, Type.Any e) | _ -> (n, a)
    in
    equal schema_w
      (Schema.v (List.map element (Schema.columns s)))
      (Query.schema (Query.unnest lists (of_schema s)))

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
  group "schema resolution"
    [
      prop "keep all, filter, sort, slice and append keep the schema" schema_gen
        keeps;
      prop "derive replaces a column in place or appends it" schema_gen
        derive_one;
      prop "select has its outputs' names, in order" subset_gen select_names;
      prop "aggregate has its keys, then its outputs" subset_gen aggregate_by;
      prop "unnest replaces lists by their elements in place" schema_gen
        unnest_lists;
      prop "a join has the left columns, then the right ones but the keys"
        join_gen joined;
    ]

(* Equality *)

(* [variants] pairs queries that differ in one argument of one verb. *)
let variants =
  let q = Query.of_source flights and c = Query.of_source carriers in
  let c' = Query.of_source (source "c'" (Schema.columns (Query.schema c))) in
  let x = delay
  and l =
    Query.of_source
      (source "l" Type.[ ("l", Any (list int64)); ("m", Any (list int64)) ])
  in
  let semi ?(kind = Join.Semi) ?(each_left = Join.Any) ?(each_right = Join.Any)
      right () =
    Query.join ~kind ~each_left ~each_right ~on:(Join.keys [ "carrier" ]) right
      q
  in
  Query.
    [
      ( "output names",
        (fun () -> select Expr.[ "a" := x ] q),
        fun () -> select Expr.[ "b" := x ] q );
      ( "output expressions",
        (fun () -> derive Expr.[ "a" := x ] q),
        fun () -> derive Expr.[ "a" := x +. float 1. ] q );
      ( "select and derive",
        (fun () -> select Expr.[ "a" := x ] q),
        fun () -> derive Expr.[ "a" := x ] q );
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
      ( "aggregate outputs",
        (fun () -> aggregate ~by:[ "carrier" ] Expr.[ "n" := rows ] q),
        fun () -> aggregate ~by:[ "carrier" ] Expr.[ "m" := rows ] q );
      ("join right inputs", semi c, semi c');
      ("join kinds", semi c, semi ~kind:Anti c);
      ("join left counts", semi c, semi ~each_left:One c);
      ("join right counts", semi c, semi ~each_right:One c);
      ( "join conditions",
        semi c,
        fun () -> join ~kind:Semi ~on:(Join.eq "carrier" "name") c q );
      ("append rests", (fun () -> append c c), fun () -> append c' c);
      ("append inputs", (fun () -> append c c), fun () -> append c c');
      ( "unnest columns",
        (fun () -> unnest [ "l" ] l),
        fun () -> unnest [ "m" ] l );
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
             ("ts", Any (datetime ~zone:"UTC" Ns));
             ("l", Any (list int64));
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
  let window n = Window.rows ~before:n ~after:0 in
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
      agg "ewm alphas"
        (fun () -> ewm ~alpha:0.25 f)
        (fun () -> ewm ~alpha:0.5 f);
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
      row "cut edges" (fun () -> cut [| 1; 2 |] x) (fun () -> cut [| 1; 3 |] x);
      row "over keys"
        (fun () -> over ~by:[ "x" ] (sum y))
        (fun () -> over ~by:[ "y" ] (sum y));
      row "over orders"
        (fun () -> over ~order:[ Order.asc "x" ] (sum y))
        (fun () -> over ~order:[ Order.desc "x" ] (sum y));
      row "rolling windows"
        (fun () -> rolling (window 3) (sum y))
        (fun () -> rolling (window 4) (sum y));
      row "nx functions"
        (fun () -> nx { f = Nx.exp } f)
        (fun () -> nx { f = Nx.neg } f);
      row "record field names"
        (fun () -> record [ "p" := x; "q" := y ])
        (fun () -> record [ "p" := x; "r" := y ]);
      row "format strings"
        (fun () -> Temporal.format "%Y" (Col.instant "ts"))
        (fun () -> Temporal.format "%m" (Col.instant "ts"));
      row "floor steps"
        (fun () -> Temporal.floor (Time.Days 1) (Col.date "d"))
        (fun () -> Temporal.floor (Time.Days 2) (Col.date "d"));
    ]

(* [query_gen] draws a query of [variants], built afresh, so that two draws of
   one query are equal values that are not one value. *)
let query_gen =
  let builders = List.concat_map (fun (_, q0, q1) -> [ q0; q1 ]) variants in
  Gen.with_pp Query.pp
    (Gen.map
       (fun i -> (List.nth builders i) ())
       (Gen.int_range 0 (List.length builders - 1)))

let equality =
  let filtered threshold =
    Query.(of_source flights |> filter Expr.(delay > float threshold))
  in
  group "Query.equal"
    [
      prop "is an equivalence" (Gen.pair query_gen query_gen) (fun qs ->
          Law.equivalence query_w qs);
      test "a pipeline built twice is equal to itself" (fun () ->
          equal query_w (late_by_carrier ()) (late_by_carrier ()));
      test "equal queries have equal schemas" (fun () ->
          equal schema_w
            (Query.schema (late_by_carrier ()))
            (Query.schema (late_by_carrier ())));
      test "a different literal makes a different query" (fun () ->
          not_equal query_w (filtered 15.) (filtered 16.));
      test "a different key makes a different query" (fun () ->
          let q by =
            Query.(of_source flights |> aggregate ~by Expr.[ "n" := rows ])
          in
          not_equal query_w (q [ "carrier" ]) (q [ "origin" ]));
      test "a different source makes a different query" (fun () ->
          not_equal query_w (Query.of_source flights) (Query.of_source carriers));
      test "sources built from the same arguments are different" (fun () ->
          let s () = source "s" Type.[ ("a", Any int64) ] in
          not_equal query_w (Query.of_source (s ())) (Query.of_source (s ())));
      test "different verbs make different queries" (fun () ->
          let q = Query.of_source flights in
          not_equal query_w
            (Query.slice ~offset:0 ~length:1 q)
            (Query.sort [] q));
      cases
        ~name:(fun (what, _, _) -> Printf.sprintf "tells apart %s" what)
        "each argument" variants
        (fun (_, q0, q1) ->
          equal query_w (q0 ()) (q0 ());
          not_equal query_w (q0 ()) (q1 ()));
      cases
        ~name:(fun (what, _, _) -> Printf.sprintf "tells apart %s" what)
        "each expression attribute" expression_variants
        (fun (_, q0, q1) ->
          equal query_w (q0 ()) (q0 ());
          equal query_w (q1 ()) (q1 ());
          not_equal query_w (q0 ()) (q1 ()));
    ]

(* Printing *)

let steps () =
  let q =
    Query.of_source
      (source "small"
         Type.
           [
             ("f", Any float64);
             ("n", Any int64);
             ("g", Any string);
             ("ts", Any (datetime ~zone:"UTC" Ns));
             ("l", Any (list int64));
           ])
  in
  let other =
    Query.of_source
      (source ~rows:1 "other"
         Type.[ ("y", Any float64); ("ys", Any (datetime ~zone:"UTC" Ns)) ])
  in
  expect
    (String.concat "\n"
       (List.map (str Query.pp)
          Query.
            [
              select
                Expr.
                  [
                    keep Sel.(names [ "f"; "n" ]);
                    "z" := Col.float "f";
                    keep Sel.(names [ "g" ]);
                  ]
                q;
              select
                Expr.
                  [
                    each Sel.(names [ "f" ]) { column = (fun _ x -> "z" := x) };
                  ]
                q;
              derive Expr.[ "k" := nx { f = Nx.exp } (Col.float "f") ] q;
              sort Order.[ asc "f"; nulls_first (desc "n") ] q;
              slice ~offset:(-10) ~length:5 q;
              slice ~offset:0 ~length:0 q;
              unnest [ "l" ] q;
              join ~kind:Left ~each_left:At_most_one ~each_right:At_least_one
                ~on:Join.all other q;
              join ~on:Join.position other q;
              join ~kind:Full ~on:(Join.eq "f" "y") other q;
              join ~kind:Semi ~on:(Join.le "f" "y") other q;
              join ~kind:Anti ~on:(Join.gt "f" "y") other q;
              join
                ~on:
                  Join.(
                    nearest ~within:Expr.(float 0.5) "f" "y" && keys [ "n" ])
                (Query.derive Expr.[ "n" := int 1 ] other)
                q;
              join
                ~on:
                  Join.(
                    closest ~within:Expr.(span (Time.Span.s 5)) (ge "ts" "ys"))
                other q;
              append q q |> append q;
            ]))
  @@ __POS_OF__
       {|
    query → f float64, n int64, z float64, g string
    select [keep (names ["f"; "n"]); "z" := f; keep (names ["g"])]
    └ small (5 columns)
    query → z float64
    select ["z" := f]
    └ small (5 columns)
    query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64], k float64
    derive ["k" := exp f]
    └ small (5 columns)
    query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64]
    sort [asc "f"; nulls_first (desc "n")]
    └ small (5 columns)
    query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64]
    slice ~offset:(-10) ~length:5
    └ small (5 columns)
    query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64]
    slice ~offset:0 ~length:0
    └ small (5 columns)
    query → f float64, n int64, g string, ts datetime[ns, UTC], l int64
    unnest ["l"]
    └ small (5 columns)
    query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64], y float64, ys datetime[ns, UTC]
    join ~on:all ~kind:Left ~each_left:At_most_one ~each_right:At_least_one
    ├ small (5 columns)
    └ other (2 columns, 1 row)
    query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64], y float64, ys datetime[ns, UTC]
    join ~on:position
    ├ small (5 columns)
    └ other (2 columns, 1 row)
    query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64], ys datetime[ns, UTC]
    join ~on:(eq "f" "y") ~kind:Full
    ├ small (5 columns)
    └ other (2 columns, 1 row)
    query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64]
    join ~on:(le "f" "y") ~kind:Semi
    ├ small (5 columns)
    └ other (2 columns, 1 row)
    query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64]
    join ~on:(gt "f" "y") ~kind:Anti
    ├ small (5 columns)
    └ other (2 columns, 1 row)
    query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64], y float64, ys datetime[ns, UTC]
    join ~on:(nearest ~within:0.5 "f" "y" && keys ["n"])
    ├ small (5 columns)
    └ derive ["n" := 1]
      └ other (2 columns, 1 row)
    query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64], y float64, ys datetime[ns, UTC]
    join ~on:(closest ~within:5s (ge "ts" "ys"))
    ├ small (5 columns)
    └ other (2 columns, 1 row)
    query → f float64, n int64, g string, ts datetime[ns, UTC], l list[int64]
    append
    ├ append
    │ ├ #1 small (5 columns)
    │ └ #1
    └ #1
    |}

let wrapping () =
  let q = Query.of_source flights in
  expect
    (str Query.pp
       Query.(
         q
         |> join ~on:(Join.keys [ "carrier" ]) (of_source carriers)
         |> derive
              Expr.
                [
                  "speed" :=
                    Col.float "distance" /. Col.float "air_time" *. float 60.;
                  "gain" := Col.float "dep_delay" -. Col.float "arr_delay";
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
    |}

let unicode () =
  let q = Query.of_source (source "données" Type.[ ("é", Any int64) ]) in
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
    |}

let printing =
  group "Query.pp"
    [
      test "prints each step as its verb is written" steps;
      test "wraps a long step under its prefix" wrapping;
      test "prints names in UTF-8 as they read" unicode;
      test "prints no schema after the arrow of an empty one" (fun () ->
          equal text "query →\nselect []\n└ s (1 column)"
            (str Query.pp
               (Query.select []
                  (Query.of_source (source "s" Type.[ ("a", Any int64) ])))));
    ]

(* Arguments *)

let flag = Ext.v ~name:"flag" ~ordered:false Type.bool ~dec:Fun.id ~enc:Fun.id

let accepted =
  group "accepted arguments"
    [
      test "an extension output has its extension type" (fun () ->
          equal schema_w
            (Schema.v Type.[ ("t2", Any epoch_t); ("w", Any epoch_t) ])
            (Query.schema
               (Query.select
                  Expr.
                    [
                      "t2" := Ext.col epoch "t";
                      "w" := Ext.wrap epoch (Col.float "f");
                    ]
                  of_kinds)));
      test "filter takes a predicate that is an OCaml value" (fun () ->
          let p = Expr.(const (fun x -> Stdlib.( > ) x 1.) $ Col.float "f") in
          equal schema_w (Query.schema of_kinds)
            (Query.schema (Query.filter p of_kinds)));
      test "filter refuses a declaration whose values are bool" (fun () ->
          let q =
            Query.of_source
              (source "flags" Type.[ ("fl", Any (ext ~name:"flag" bool)) ])
          in
          expect (message (fun () -> Query.filter (Ext.col flag "fl") q))
          @@ __POS_OF__
               {|
            filter: 1 problem
              fl
                fl is ext[flag, bool], where bool is expected: use Ext.storage.
              input (1 column): fl ext[flag, bool]
            |});
      test "an output name must be UTF-8" (fun () ->
          raises (Invalid_argument {|Expr.( := ): "\255" is not valid UTF-8|})
            (fun () -> Expr.("\xff" := Col.float "f")));
    ]

(* [suggestion key columns] is the problem that [sort] reports for a key [key]
   over the columns [columns]. *)
let suggestion key columns =
  let q =
    Query.of_source
      (source "s" (List.map (fun n -> (n, Type.Any Type.int64)) columns))
  in
  let report = message (fun () -> Query.sort [ Order.asc key ] q) in
  List.nth (String.split_on_char '\n' report) 1

let suggestions =
  group "suggestions"
    [
      cases
        ~name:(fun (what, _, _, _) -> what)
        "missing names"
        [
          ( "suggests a column at distance 2",
            "ab",
            [ "abxy"; "zzzzzz" ],
            {|no column "ab". Did you mean "abxy"?|} );
          ( "lists the columns when the nearest is at distance 3",
            "abc",
            [ "xyz" ],
            {|no column "abc". The columns are "xyz".|} );
          ( "lists every column past distance 2",
            "abcd",
            [ "wxyz"; "abxyz" ],
            {|no column "abcd". The columns are "wxyz" and "abxyz".|} );
          ( "suggests the nearest columns only",
            "abcd",
            [ "abxx"; "abce" ],
            {|no column "abcd". Did you mean "abce"?|} );
          ( "suggests ties in the columns' order",
            "ab",
            [ "xb"; "zzzz"; "ay" ],
            {|no column "ab". Did you mean "xb" or "ay"?|} );
          ( "says that there are no columns",
            "a",
            [],
            {|no column "a". There are no columns.|} );
        ]
        (fun (_, key, columns, expected) ->
          equal string
            (Printf.sprintf "  asc %S: %s" key expected)
            (suggestion key columns));
    ]

(* Sources *)

let rejected_sources () =
  let v ?rows ?sorted name =
    Source.v ~name ~schema:(Query.schema of_kinds) ?rows ?sorted (fun _ ->
        Ok [])
  in
  expect
    (messages
       [
         (fun () -> v "");
         (fun () -> v "a\nb");
         (fun () -> v "a\x7f");
         (fun () -> v "a\xc2\x85");
         (fun () -> v "\xff");
         (fun () -> v ~rows:(-1) "s");
         (fun () ->
           v
             ~sorted:Order.[ asc "nope"; asc "f"; desc "f"; asc "t"; asc "r" ]
             "s");
       ])
  @@ __POS_OF__
       {|
    Source.v: the name is empty

    Source.v: the name "a\nb" holds a control character

    Source.v: the name "a\127" holds a control character

    Source.v: the name "a\194\133" holds a control character

    Source.v: the name "\255" is not UTF-8

    Source.v: the row count -1 is negative

    Source.v: ~sorted:
      asc "nope": no column "nope". The columns are "x8", "u8", "f", "f32", "n", "g", "d", "b", "ts", "t", "l", "r" and "wait".
      desc "f": the column "f" is already a key.
      asc "t": "t" is ext[ymir.epoch, float64], which orders only through its declaration: order by its storage, derived first.
      asc "r": "r" is record[e ext[ymir.epoch, float64]], which holds an extension type and has no order.
    |}

let sources =
  group "Source.v"
    [
      test "refuses a bad name, row count or order" rejected_sources;
      test "accepts spaces and printable non-ASCII in a name" (fun () ->
          ignore (source "a b" []);
          ignore (source "a\xc2\xa0b \xc3\xa9" []));
      test "accepts an order on ordered columns" (fun () ->
          ignore
            (Source.v ~name:"s" ~schema:(Query.schema of_kinds)
               ~sorted:Order.[ asc "n"; desc "f" ]
               ~rows:0
               (fun _ -> Ok [])));
    ]

let () =
  exit
    (run "query"
       [
         guide;
         reports;
         conditions;
         resolution;
         equality;
         printing;
         accepted;
         suggestions;
         sources;
       ])
