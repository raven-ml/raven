(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let exec q = Error.get_ok (Query.run q)
let date (y, m, d) = Option.get (Time.Date.of_civil (y, m, d))
let civil d = Time.Date.to_civil d
let invalid sub f = raises_match (Exn.invalid_arg ~sub) f

let ymd =
  Testable.make
    ~pp:(fun ppf (y, m, d) -> Format.fprintf ppf "%d-%d-%d" y m d)
    ~equal:( = )

let dates t name =
  Array.to_list (Column.values Kind.date (Talon.column t name))
  |> List.map civil

(* Time ranges *)

let time_range_months () =
  (* The k-th value is k steps from the start, each clamped to its month, so
     no drift: January 31, February 29, March 31, April 30. *)
  let t =
    Kit.time_range Type.date ~start:(date (2024, 1, 31))
      ~stop:(date (2024, 5, 1)) (Months 1) "d"
  in
  equal (list ymd)
    [ (2024, 1, 31); (2024, 2, 29); (2024, 3, 31); (2024, 4, 30) ]
    (dates t "d");
  equal (list string)
    [ "d" ]
    (List.map fst (Schema.columns (Talon.schema t)));
  (* The stop is excluded: a month that reaches it is not taken. *)
  let t =
    Kit.time_range Type.date ~start:(date (2024, 1, 1))
      ~stop:(date (2024, 3, 1)) (Months 1) "d"
  in
  equal (list ymd) [ (2024, 1, 1); (2024, 2, 1) ] (dates t "d")

let time_range_weeks () =
  let t =
    Kit.time_range Type.date ~start:(date (2024, 1, 1))
      ~stop:(date (2024, 1, 20)) (Weeks 1) "w"
  in
  equal (list ymd)
    [ (2024, 1, 1); (2024, 1, 8); (2024, 1, 15) ]
    (dates t "w")

let time_range_exact () =
  (* Hourly datetimes over a day; the stop excluded. *)
  let s = Option.get (Time.of_s 3600L) and e = Option.get (Time.of_s 14400L) in
  let t =
    Kit.time_range Type.(instant Ms) ~start:s ~stop:e
      (Exact (Time.Span.hours 1)) "ts"
  in
  equal (list int)
    [ 1; 2; 3 ]
    (Array.to_list
       (Array.map
          (fun x -> Int64.to_int (Time.to_s x) / 3600)
          (Column.values Kind.instant (Talon.column t "ts"))))

let time_range_empty () =
  let d = date (2024, 1, 1) in
  let n stop =
    Talon.rows (Kit.time_range Type.date ~start:d ~stop (Days 1) "d")
  in
  equal int 0 (n d);
  equal int 0 (n (date (2023, 12, 1)))

let time_range_steps () =
  let d = date (2024, 1, 1) and stop = date (2024, 2, 1) in
  let range step () = Kit.time_range Type.date ~start:d ~stop step "d" in
  invalid "not positive" (range (Days 0));
  invalid "not positive" (range (Months (-1)));
  invalid "not positive" (range (Exact (Time.Span.of_ns 0L)));
  (* An exact step of a date is the binding's problem. *)
  invalid "Temporal.offset" (range (Exact (Time.Span.days 1)))

(* Against a reference: the k-th value is [Temporal.offset (k * step)] applied
   to the start, one row per k through the public expression, kept below the
   stop. *)
let reference ~start ~stop step =
  let t = v [ ("d", Column.v Type.date [| start |]) ] in
  let offset k =
    let step : Time.step =
      match step with
      | `Days n -> Days (k * n)
      | `Weeks n -> Weeks (k * n)
      | `Months n -> Months (k * n)
    in
    (Column.values Kind.date
       (Talon.column
          (exec
             Query.(
               derive
                 Expr.[ "d" := Temporal.offset step (Col.date "d") ]
                 (of_table t)))
          "d")).(0)
  in
  let rec go k acc =
    let d = offset k in
    if Time.Date.compare d stop < 0 then go (k + 1) (d :: acc) else List.rev acc
  in
  go 0 []

let time_range_law (s, days, kind, n) =
  let start = Option.get (Time.Date.of_days s) in
  let stop = Option.get (Time.Date.of_days (s + days)) in
  let step =
    match kind with 0 -> `Days n | 1 -> `Weeks n | _ -> `Months n
  in
  let got =
    Column.values Kind.date
      (Talon.column
         (Kit.time_range Type.date ~start ~stop
            (match step with
            | `Days n -> Days n
            | `Weeks n -> Weeks n
            | `Months n -> Months n)
            "d")
         "d")
  in
  equal (list ymd)
    (List.map civil (reference ~start ~stop step))
    (List.map civil (Array.to_list got))

let time_range_gen =
  Gen.(
    let* s = int_range (-40000) 40000 in
    let* days = int_range 1 900 in
    let* kind = int_range 0 2 in
    let* n = int_range 1 7 in
    constant (s, days, kind, n))

(* A nanosecond range longer than 292 years, whose tick span passes int64: the
   count is exact and the values are in range throughout. *)
let time_range_extreme () =
  let first = Int64.min_int and last = Int64.max_int in
  let t =
    Kit.time_range Type.(instant Ns) ~start:(Time.of_ns first)
      ~stop:(Time.of_ns last)
      (Exact (Time.Span.days 1))
      "ts"
  in
  let day = 86_400_000_000_000L in
  let count =
    Int64.to_int (Int64.unsigned_div (Int64.sub (Int64.pred last) first) day) + 1
  in
  equal int count (Talon.rows t);
  let ns =
    Array.map Time.to_ns (Column.values Kind.instant (Talon.column t "ts"))
  in
  equal int64 first ns.(0);
  equal int64 (Int64.add first (Int64.mul (Int64.of_int (count - 1)) day)) ns.(count - 1);
  less int64 ~than:last ns.(count - 1);
  invalid "2^62 steps" (fun () ->
      Kit.time_range Type.(instant Ns) ~start:(Time.of_ns first)
        ~stop:(Time.of_ns last)
        (Exact (Time.Span.of_ns 1L))
        "ts")

(* Dense *)

let keys xs = Column.v Type.int32 xs

let dense_table i j x = v [ ("i", keys i); ("j", keys j); ("v", x) ]

let dense_zeros () =
  let t =
    dense_table [| 0; 1 |] [| 2; 0 |] (Column.v Type.float64 [| 1.5; -2. |])
  in
  let x, m = to_dense Nx.float64 ~axes:[ ("i", 2); ("j", 3) ] "v" t in
  equal (array float_exact)
    [| 0.; 0.; 1.5; -2.; 0.; 0. |]
    (Nx.to_array (Nx.flatten x));
  let bits = Nx.to_array (Nx.flatten (Nx.cast Nx.bool (Option.get m))) in
  equal (array bool) [| false; false; true; true; false; false |] bits;
  (* A table that fills every cell has no validity. *)
  let full = dense_table [| 0; 1 |] [| 0; 0 |] (Column.v Type.int64 [| 7; 8 |]) in
  let x, m = to_dense Nx.int32 ~axes:[ ("i", 2); ("j", 1) ] "v" full in
  equal (array int) [| 7; 8 |]
    (Array.map Int32.to_int (Nx.to_array (Nx.flatten x)));
  is_none m

let dense_nulls_are_zero () =
  let vs = Column.of_options Type.float64 [| Some 2.; None |] in
  let t = dense_table [| 0; 0 |] [| 0; 1 |] vs in
  let x, m = to_dense Nx.float64 ~axes:[ ("i", 1); ("j", 2) ] "v" t in
  equal (array float_exact) [| 2.; 0. |] (Nx.to_array (Nx.flatten x));
  equal (array bool) [| true; false |]
    (Nx.to_array (Nx.flatten (Nx.cast Nx.bool (Option.get m))))

let dense_categorical () =
  let ty = Type.categorical [| "a"; "b" |] in
  let t =
    v
      [
        ("k", Column.v ty [| "b"; "a" |]);
        ("v", Column.v Type.float64 [| 1.; 2. |]);
      ]
  in
  let x, m = to_dense Nx.float64 ~axes:[ ("k", 2) ] "v" t in
  equal (array float_exact) [| 2.; 1. |] (Nx.to_array x);
  is_none m;
  invalid "dictionary holds 2" (fun () ->
      to_dense Nx.float64 ~axes:[ ("k", 3) ] "v" t)

let dense_errors () =
  let t = dense_table [| 0; 0 |] [| 1; 1 |] (Column.v Type.int64 [| 1; 2 |]) in
  invalid "rows 0 and 1" (fun () ->
      to_dense Nx.int64 ~axes:[ ("i", 2); ("j", 2) ] "v" t);
  let t = dense_table [| 0; 5 |] [| 0; 0 |] (Column.v Type.int64 [| 1; 2 |]) in
  invalid "outside [0, 2)" (fun () ->
      to_dense Nx.int64 ~axes:[ ("i", 2); ("j", 1) ] "v" t);
  let keys = Column.of_options Type.int32 [| Some 0; None |] in
  let t =
    v
      [ ("i", keys); ("j", Column.v Type.int32 [| 0; 0 |]);
        ("v", Column.v Type.int64 [| 1; 2 |]) ]
  in
  invalid "null at row 1" (fun () ->
      to_dense Nx.int64 ~axes:[ ("i", 2); ("j", 1) ] "v" t);
  let t =
    v
      [ ("s", Column.v Type.string [| "x" |]);
        ("v", Column.v Type.int64 [| 1 |]) ]
  in
  invalid "neither categorical" (fun () ->
      to_dense Nx.int64 ~axes:[ ("s", 1) ] "v" t)

let dense_edges () =
  let t = dense_table [||] [||] (Column.v Type.float64 [||]) in
  let x, m = to_dense Nx.float64 ~axes:[ ("i", 2); ("j", 0) ] "v" t in
  equal (array int) [| 2; 0 |] (Nx.shape x);
  is_none m;
  let x, m = to_dense Nx.float64 ~axes:[ ("i", 2); ("j", 1) ] "v" t in
  equal (array float_exact) [| 0.; 0. |] (Nx.to_array (Nx.flatten x));
  equal (array bool) [| false; false |]
    (Nx.to_array (Nx.flatten (Nx.cast Nx.bool (Option.get m))));
  let z = of_dense ~axes:[ "i"; "j" ] "v" (Nx.zeros Nx.float64 [| 3; 0 |]) in
  equal int 0 (Talon.rows z);
  let t = dense_table [| 0 |] [| 0 |] (Column.v Type.int64 [| 1 |]) in
  invalid "no axis" (fun () -> to_dense Nx.int64 ~axes:[] "v" t);
  invalid "named twice" (fun () ->
      to_dense Nx.int64 ~axes:[ ("i", 1); ("i", 1) ] "v" t);
  invalid "below 0" (fun () -> to_dense Nx.int64 ~axes:[ ("i", -1) ] "v" t);
  invalid "too many cells" (fun () ->
      to_dense Nx.int64 ~axes:[ ("i", max_int); ("j", 2) ] "v"
        (dense_table [||] [||] (Column.v Type.int64 [||])));
  invalid "no axis" (fun () ->
      of_dense ~axes:[] "v" (Nx.scalar Nx.int64 1L))

(* Round trips: [to_dense] after [of_dense] keeps values and validity, with
   zeros under clear bits. *)
let dense_law (shape, xs, keep) =
  let cells = Array.fold_left ( * ) 1 shape in
  let x = Nx.create Nx.float64 shape (Array.sub xs 0 cells) in
  let bits = Nx.create Nx.bool shape (Array.sub keep 0 cells) in
  let axes = Array.to_list (Array.mapi (fun i _ -> Printf.sprintf "k%d" i) shape) in
  let t =
    of_dense ~validity:(Nx.cast Nx.bit bits) ~axes "v" x
  in
  let dense, m =
    to_dense Nx.float64 ~axes:(List.mapi (fun i a -> (a, shape.(i))) axes) "v" t
  in
  let expected =
    Nx.where bits x (Nx.zeros_like x) |> Nx.to_array
  in
  equal (array float_exact) expected (Nx.to_array dense);
  let got_bits =
    match m with
    | Some m -> Nx.to_array (Nx.cast Nx.bool m)
    | None -> Array.make cells true
  in
  equal (array bool) (Array.sub keep 0 cells) got_bits

let dense_gen =
  Gen.(
    let* d0 = int_range 1 4 in
    let* d1 = int_range 1 4 in
    let cells = d0 * d1 in
    let* xs = array ~size:(constant cells) (float_range (-9.) 9.) in
    let* keep = array ~size:(constant cells) bool in
    constant ([| d0; d1 |], xs, keep))

(* [Temporal.div] *)

let spans us = Column.v (Type.duration Us) (Array.map Time.Span.us us)

let div_col a b =
  exec
    (Query.derive
       Expr.[ "d" := Temporal.div (Col.span "a") (Col.span "b") ]
       (Query.of_table (v [ ("a", a); ("b", b) ])))

let div_cases () =
  (* Sixty seconds per minute, exactly. *)
  let r =
    div_col
      (spans [| 90_000_000; 0; 86_400_000_000 |])
      (spans [| 60_000_000; 60_000_000; 86_400_000_000 |])
  in
  equal (array float_exact) [| 1.5; 0.; 1. |]
    (Column.values Kind.float (Talon.column r "d"))

let div_literal () =
  (* Days since an event, against a literal span. *)
  let t = v [ ("a", spans [| 172_800_000_000 |]) ] in
  let r =
    exec
      (Query.derive
         Expr.
           [
             "d" :=
               Temporal.div (Col.span "a")
                 (span (Time.Span.days 1));
           ]
         (Query.of_table t))
  in
  equal (array float_exact) [| 2. |]
    (Column.values Kind.float (Talon.column r "d"))

let div_units () =
  (* Two durations of different units, by their values. *)
  let r =
    div_col
      (Column.v (Type.duration S) [| Time.Span.of_ns 90_000_000_000L |])
      (spans [| 60_000_000 |])
  in
  equal (array float_exact) [| 1.5 |]
    (Column.values Kind.float (Talon.column r "d"))

let div_zero_null () =
  let a =
    Column.of_options (Type.duration Us)
      [| Some (Time.Span.us 5); Some (Time.Span.us 0); Some (Time.Span.us (-5));
         None |]
  in
  let o =
    Column.options Kind.float
      (Talon.column (div_col a (spans [| 0; 0; 0; 1 |])) "d")
  in
  equal float_exact infinity (Option.get o.(0));
  equal float_exact nan (Option.get o.(1));
  equal float_exact neg_infinity (Option.get o.(2));
  is_none o.(3)

let div_clock () =
  let t = v [ ("c", Column.v (Type.clock Us) [| Time.Span.us 1 |]) ] in
  invalid "cast a clock" (fun () ->
      Query.derive
        Expr.[ "d" := Temporal.div (Col.span "c") (Col.span "c") ]
        (Query.of_table t))

let () =
  exit
    (run "grids"
       [
         group "time_range"
           [
             test "months from the start" time_range_months;
             test "weeks" time_range_weeks;
             test "an exact step" time_range_exact;
             test "empty" time_range_empty;
             test "steps" time_range_steps;
             prop "against offset by k steps" time_range_gen time_range_law;
             test "across 2^63 ticks" time_range_extreme;
           ];
         group "to_dense"
           [
             test "absent cells are zero" dense_zeros;
             test "a null value is zero" dense_nulls_are_zero;
             test "categorical axes" dense_categorical;
             test "errors" dense_errors;
             test "empty shapes and argument errors" dense_edges;
             prop "round trip through of_dense" dense_gen dense_law;
           ];
         group "div"
           [
             test "by a column" div_cases;
             test "by a literal" div_literal;
             test "across units" div_units;
             test "zero divisor and nulls" div_zero_null;
             test "a clock" div_clock;
           ];
       ])
