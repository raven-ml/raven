(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let exec q = require_ok ~pp:Error.pp (Query.run q)
let table_w = Testable.make ~pp:Talon.pp ~equal:Talon.equal
let cells x = Nx.to_array (Nx.flatten x)
let held m = cells (Nx.cast Nx.bool m)
let keys xs = Column.v Type.int32 xs
let dense_table i j x = v [ ("i", keys i); ("j", keys j); ("v", x) ]

(* Dense *)

(* Two rows of a 2 × 3 grid, at (0, 2) and (1, 0). *)
let sparse =
  dense_table [| 0; 1 |] [| 2; 0 |] (Column.v Type.float64 [| 1.5; -2. |])

let scattering =
  group "to_dense"
    [
      test "scatters value into its cells, row-major" (fun () ->
          let x, _ =
            to_dense Nx.float64 ~axes:[ ("i", 2); ("j", 3) ] "v" sparse
          in
          equal (array float_exact) [| 0.; 0.; 1.5; -2.; 0.; 0. |] (cells x));
      test "a cell no row names holds zero, its bit clear" (fun () ->
          let _, m =
            to_dense Nx.float64 ~axes:[ ("i", 2); ("j", 3) ] "v" sparse
          in
          equal (array bool)
            [| false; false; true; true; false; false |]
            (held (require_some m)));
      test "a null value holds zero, its bit clear" (fun () ->
          let vs = Column.of_options Type.float64 [| Some 2.; None |] in
          let t = dense_table [| 0; 0 |] [| 0; 1 |] vs in
          let x, m = to_dense Nx.float64 ~axes:[ ("i", 1); ("j", 2) ] "v" t in
          equal (array float_exact) [| 2.; 0. |] (cells x);
          equal (array bool) [| true; false |] (held (require_some m)));
      test "a categorical key is its code" (fun () ->
          let t =
            v
              [
                ("k", Column.v (Type.categorical [| "a"; "b" |]) [| "b"; "a" |]);
                ("v", Column.v Type.float64 [| 1.; 2. |]);
              ]
          in
          let x, _ = to_dense Nx.float64 ~axes:[ ("k", 2) ] "v" t in
          equal (array float_exact) [| 2.; 1. |] (cells x));
      test "a boolean value converts as Nx.cast" (fun () ->
          let t =
            dense_table [| 0; 1 |] [| 0; 0 |]
              (Column.v Type.bool [| true; false |])
          in
          let x, _ = to_dense Nx.float64 ~axes:[ ("i", 2); ("j", 1) ] "v" t in
          equal (array float_exact) [| 1.; 0. |] (cells x));
      test "every cell held gives no validity" (fun () ->
          let t =
            dense_table [| 0; 1 |] [| 0; 0 |] (Column.v Type.int64 [| 7; 8 |])
          in
          let _, m = to_dense Nx.int32 ~axes:[ ("i", 2); ("j", 1) ] "v" t in
          is_none m);
      test "a zero extent gives an empty tensor" (fun () ->
          let t = dense_table [||] [||] (Column.v Type.float64 [||]) in
          let x, m = to_dense Nx.float64 ~axes:[ ("i", 2); ("j", 0) ] "v" t in
          equal (array int) [| 2; 0 |] (Nx.shape x);
          is_none m);
      test "no row gives zeros, every bit clear" (fun () ->
          let t = dense_table [||] [||] (Column.v Type.float64 [||]) in
          let x, m = to_dense Nx.float64 ~axes:[ ("i", 2); ("j", 1) ] "v" t in
          equal (array float_exact) [| 0.; 0. |] (cells x);
          equal (array bool) [| false; false |] (held (require_some m)));
      (let one = dense_table [| 0 |] [| 0 |] (Column.v Type.int64 [| 1 |]) in
       let dense axes t () = ignore (to_dense Nx.int64 ~axes "v" t) in
       cases "raises Invalid_argument" ~id:fst
         [
           ("without an axis", dense [] one);
           ("with an axis named twice", dense [ ("i", 1); ("i", 1) ] one);
           ("with a negative extent", dense [ ("i", -1) ] one);
           ( "with an extent other than the dictionary's size",
             dense
               [ ("k", 3) ]
               (v
                  [
                    ("k", Column.v (Type.categorical [| "a"; "b" |]) [| "a" |]);
                    ("v", Column.v Type.int64 [| 1 |]);
                  ]) );
           ( "with a key neither categorical nor integer",
             dense
               [ ("s", 1) ]
               (v
                  [
                    ("s", Column.v Type.string [| "x" |]);
                    ("v", Column.v Type.int64 [| 1 |]);
                  ]) );
           ( "with a null key",
             dense
               [ ("i", 2); ("j", 1) ]
               (v
                  [
                    ("i", Column.of_options Type.int32 [| Some 0; None |]);
                    ("j", keys [| 0; 0 |]);
                    ("v", Column.v Type.int64 [| 1; 2 |]);
                  ]) );
           ( "with a key outside its extent",
             dense
               [ ("i", 2); ("j", 1) ]
               (dense_table [| 0; 5 |] [| 0; 0 |]
                  (Column.v Type.int64 [| 1; 2 |])) );
           ( "with two rows in one cell",
             dense
               [ ("i", 2); ("j", 2) ]
               (dense_table [| 0; 0 |] [| 1; 1 |]
                  (Column.v Type.int64 [| 1; 2 |])) );
           ( "with a value neither numeric nor boolean",
             dense
               [ ("i", 1); ("j", 1) ]
               (dense_table [| 0 |] [| 0 |] (Column.v Type.string [| "x" |])) );
           ( "with more cells than an int counts",
             dense
               [ ("i", max_int); ("j", 2) ]
               (dense_table [||] [||] (Column.v Type.int64 [||])) );
         ]
         (fun (_, f) -> raises_match (Exn.invalid_arg ~sub:"Talon.to_dense") f));
    ]

(* Grids of rank 1 to 3, some extents 0, values with NaN and [-0.]. *)
let grids =
  let open Gen in
  let* shape =
    array ~size:(int_range 1 3)
      (frequency [ (1, constant 0); (4, int_range 1 4) ])
  in
  let n = Array.fold_left ( * ) 1 shape in
  let value =
    frequency
      [ (1, of_list [ nan; -0.; 0.; infinity ]); (4, float_range (-9.) 9.) ]
  in
  let* xs = array ~size:(constant n) value in
  let+ keep =
    frequency
      [ (1, constant (Array.make n true)); (3, array ~size:(constant n) bool) ]
  in
  (shape, xs, keep)

let round_trip (shape, xs, keep) =
  cover "a zero extent" (Array.mem 0 shape);
  cover "every bit set" (Array.for_all Fun.id keep);
  let x = Nx.create Nx.float64 shape xs in
  let bits = Nx.create Nx.bool shape keep in
  let axes = List.init (Array.length shape) (Printf.sprintf "k%d") in
  let t = of_dense ~validity:(Nx.cast Nx.bit bits) ~axes "v" x in
  let extents = List.mapi (fun i a -> (a, shape.(i))) axes in
  let dense, m = to_dense Nx.float64 ~axes:extents "v" t in
  let zeroed = Array.map2 (fun x k -> if k then x else 0.) xs keep in
  equal (array float_exact) zeroed (cells dense);
  let all = Array.make (Array.length keep) true in
  equal (array bool) keep (match m with Some m -> held m | None -> all)

let laying_out =
  group "of_dense"
    [
      test "is one row per element, int64 keys then the values" (fun () ->
          let xs = [| 0.; 1.; 2.; 3.; 4.; 5. |] in
          let bits = [| true; false; true; true; true; false |] in
          let validity = Nx.cast Nx.bit (Nx.create Nx.bool [| 2; 3 |] bits) in
          let opt b x = if b then Some x else None in
          equal table_w
            (v
               [
                 ("r", Column.v Type.int64 [| 0; 0; 0; 1; 1; 1 |]);
                 ("c", Column.v Type.int64 [| 0; 1; 2; 0; 1; 2 |]);
                 ("v", Column.of_options Type.float64 (Array.map2 opt bits xs));
               ])
            (of_dense ~validity ~axes:[ "r"; "c" ] "v"
               (Nx.create Nx.float64 [| 2; 3 |] xs)));
      prop "to_dense returns x, zeros where validity is clear" grids round_trip;
      cases "raises Invalid_argument" ~id:fst
        [
          ( "without an axis",
            fun () -> of_dense ~axes:[] "v" (Nx.scalar Nx.int64 1L) );
          ( "with axes other than x's dimensions",
            fun () -> of_dense ~axes:[ "i" ] "v" (Nx.zeros Nx.int64 [| 2; 2 |])
          );
          ( "with a validity of another shape",
            fun () ->
              of_dense
                ~validity:(Nx.cast Nx.bit (Nx.ones Nx.bool [| 3 |]))
                ~axes:[ "i" ] "v"
                (Nx.zeros Nx.int64 [| 2 |]) );
        ]
        (fun (_, f) -> raises_match (Exn.invalid_arg ~sub:"Talon.of_dense") f);
    ]

(* Temporal.div *)

(* [duration unit_ n] is a one-row column of [n] seconds or microseconds. *)
let duration unit_ n =
  match unit_ with
  | `S -> Column.v (Type.duration S) [| Time.Span.s n |]
  | `Us -> Column.v (Type.duration Us) [| Time.Span.us n |]

let quotient a b =
  let q =
    Query.derive
      Expr.[ "d" := Temporal.div (Col.span "a") (Col.span "b") ]
      (Query.of_table (v [ ("a", a); ("b", b) ]))
  in
  Column.options Kind.float (Talon.column (exec q) "d")

(* Operands below 2^53 once scaled to microseconds, which float64 holds exactly,
   so that their quotient rounds once: seconds within 1e9, microseconds within
   1e15. *)
let divisions =
  let open Gen in
  let unit_ = of_list [ `S; `Us ] in
  let amount = function
    | `S -> int_range (-1_000_000_000) 1_000_000_000
    | `Us -> int_range (-1_000_000_000_000_000) 1_000_000_000_000_000
  in
  let* ua, ub = pair unit_ unit_ in
  let* a = amount ua in
  let+ b = frequency [ (1, constant 0); (4, amount ub) ] in
  ((ua, a), (ub, b))

let pp_division ppf ((ua, a), (ub, b)) =
  let u = function `S -> "s" | `Us -> "us" in
  Format.fprintf ppf "%d%s / %d%s" a (u ua) b (u ub)

let in_us = function `S, n -> Float.of_int n *. 1e6 | `Us, n -> Float.of_int n

let div_law (a, b) =
  cover "a zero divisor" (snd b = 0);
  cover "a coarser dividend" (fst a = `S && fst b = `Us);
  cover "a coarser divisor" (fst a = `Us && fst b = `S);
  equal
    (array (option float_exact))
    [| Some (in_us a /. in_us b) |]
    (quotient (duration (fst a) (snd a)) (duration (fst b) (snd b)))

let dividing =
  group "Temporal.div"
    [
      prop "rounds the exact quotient once"
        ~examples:
          [
            ((`Us, 90_000_000), (`Us, 60_000_000));
            ((`S, 90), (`Us, 60_000_000));
            ((`Us, 86_400_000_000), (`S, 86_400));
          ]
        (Gen.with_pp pp_division divisions)
        div_law;
      test "divides by a literal span" (fun () ->
          let q =
            Query.derive
              Expr.
                [ "d" := Temporal.div (Col.span "a") (span (Time.Span.days 1)) ]
              (Query.of_table (v [ ("a", duration `Us 172_800_000_000) ]))
          in
          equal (array float_exact) [| 2. |]
            (Column.values Kind.float (Talon.column (exec q) "d")));
      cases "a zero divisor divides as IEEE 754" ~id:fst
        [
          ("5 / 0", (5, infinity));
          ("0 / 0", (0, nan));
          ("-5 / 0", (-5, neg_infinity));
        ]
        (fun (_, (a, want)) ->
          equal
            (array (option float_exact))
            [| Some want |]
            (quotient (duration `Us a) (duration `Us 0)));
      test "is null where an operand is" (fun () ->
          let us = Option.map Time.Span.us in
          let a =
            Column.of_options (Type.duration Us) [| None; us (Some 1) |]
          in
          let b =
            Column.of_options (Type.duration Us) [| us (Some 1); None |]
          in
          equal (array (option float_exact)) [| None; None |] (quotient a b));
      test "a clock is a problem" (fun () ->
          let t = v [ ("c", Column.v (Type.clock Us) [| Time.Span.us 1 |]) ] in
          raises_match (Exn.invalid_arg ~sub:"cast a clock") (fun () ->
              Query.derive
                Expr.[ "d" := Temporal.div (Col.span "c") (Col.span "c") ]
                (Query.of_table t)));
    ]

(* Time ranges *)

let date (y, m, d) = Option.get (Time.Date.of_civil (y, m, d))

let ymd =
  Testable.make
    ~pp:(fun ppf (y, m, d) -> Format.fprintf ppf "%d-%02d-%02d" y m d)
    ~equal:( = )

let dates t =
  Column.values Kind.date (Talon.column t "d")
  |> Array.to_list
  |> List.map Time.Date.to_civil

(* [offset step d] is [Temporal.offset step] at [d], through a query. *)
let offset step d =
  let t = v [ ("d", Column.v Type.date [| d |]) ] in
  let q =
    Query.derive
      Expr.[ "d" := Temporal.offset step (Col.date "d") ]
      (Query.of_table t)
  in
  (Column.values Kind.date (Talon.column (exec q) "d")).(0)

let scaled k : Time.step -> Time.step = function
  | Days n -> Days (k * n)
  | Weeks n -> Weeks (k * n)
  | Months n -> Months (k * n)
  | Exact d -> Exact d

(* The values [offset (k × step) start] below [stop], for k from 0. *)
let offsets ~start ~stop step =
  let rec go k acc =
    let d = offset (scaled k step) start in
    if Time.Date.compare d stop < 0 then go (k + 1) (d :: acc) else List.rev acc
  in
  go 0 []

let month_end d =
  let y, m, _ = Time.Date.to_civil d in
  Option.get
    (List.find_map
       (fun day -> Time.Date.of_civil (y, m, day))
       [ 31; 30; 29; 28 ])

(* Ranges start at a month's end half the time, so that a month step reaches a
   shorter month, and stop at or before their start one time in five. *)
let ranges =
  let open Gen in
  let* s = int_range (-40_000) 40_000 in
  let* at_end = bool in
  let* days = frequency [ (1, int_range (-3) 0); (4, int_range 1 900) ] in
  let+ step =
    frequency
      [
        (2, map (fun n -> Time.Months n) (int_range 1 7));
        (1, map (fun n -> Time.Weeks n) (int_range 1 7));
        (1, map (fun n -> Time.Days n) (int_range 1 7));
      ]
  in
  let start = Option.get (Time.Date.of_days s) in
  let start = if at_end then month_end start else start in
  let stop = Option.get (Time.Date.of_days (Time.Date.to_days start + days)) in
  (start, stop, step)

let pp_range ppf (start, stop, step) =
  Format.fprintf ppf "%a to %a by %a" Time.Date.pp start Time.Date.pp stop
    Time.pp_step step

let range_law (start, stop, step) =
  let want = offsets ~start ~stop step in
  let day d =
    let _, _, d = Time.Date.to_civil d in
    d
  in
  cover "a day past a month's end"
    (List.exists (fun d -> day d < day start) want);
  cover "start at or after stop" (Time.Date.compare start stop >= 0);
  equal (list ymd)
    (List.map Time.Date.to_civil want)
    (dates (Kit.time_range Type.date ~start ~stop step "d"))

let ranging =
  let range ty ~start ~stop step () =
    ignore (Kit.time_range ty ~start ~stop step "d")
  in
  let jan1 = date (2024, 1, 1) and feb1 = date (2024, 2, 1) in
  let instant s = Option.get (Time.of_s s) in
  group "Kit.time_range"
    [
      prop "is offset (k × step) start below stop"
        (Gen.with_pp pp_range ranges)
        range_law;
      test "a day past a month's end becomes its last day" (fun () ->
          (* Each step counts from the start, so March has its 31st again. *)
          let t =
            Kit.time_range Type.date
              ~start:(date (2024, 1, 31))
              ~stop:(date (2024, 5, 1))
              (Months 1) "d"
          in
          equal (list ymd)
            [ (2024, 1, 31); (2024, 2, 29); (2024, 3, 31); (2024, 4, 30) ]
            (dates t));
      test "excludes stop" (fun () ->
          let stop = date (2024, 3, 1) in
          let t = Kit.time_range Type.date ~start:jan1 ~stop (Months 1) "d" in
          equal (list ymd) [ (2024, 1, 1); (2024, 2, 1) ] (dates t));
      test "is one column name" (fun () ->
          let t =
            Kit.time_range Type.date ~start:jan1 ~stop:feb1 (Days 1) "d"
          in
          equal (list string) [ "d" ]
            (List.map fst (Schema.columns (Talon.schema t))));
      test "an exact step adds its span" (fun () ->
          let t =
            Kit.time_range
              Type.(instant Ms)
              ~start:(instant 3600L) ~stop:(instant 14400L)
              (Exact (Time.Span.hours 1))
              "ts"
          in
          equal (array int64) [| 3600L; 7200L; 10800L |]
            (Array.map Time.to_s
               (Column.values Kind.instant (Talon.column t "ts"))));
      test "a wall clock moves by calendar months" (fun () ->
          let wall d =
            Option.get (Time.of_s (Int64.of_int (86_400 * Time.Date.to_days d)))
          in
          let t =
            Kit.time_range
              Type.(wall S)
              ~start:(wall (date (2024, 1, 31)))
              ~stop:(wall (date (2024, 4, 1)))
              (Months 1) "w"
          in
          equal (array int64)
            (Array.map
               (fun d -> Time.to_s (wall (date d)))
               [| (2024, 1, 31); (2024, 2, 29); (2024, 3, 31) |])
            (Array.map Time.to_s (Column.values Kind.wall (Talon.column t "w"))));
      cases "is empty when start >= stop" ~id:fst
        [ ("start = stop", jan1); ("start > stop", date (2023, 12, 1)) ]
        (fun (_, stop) ->
          equal int 0
            (Talon.rows
               (Kit.time_range Type.date ~start:jan1 ~stop (Days 1) "d")));
      test "counts past 2^63 ticks" (fun () ->
          (* Nanoseconds over 584 years, a span of ticks past int64. *)
          let first = Int64.min_int and last = Int64.max_int in
          let day = 86_400_000_000_000L in
          let t =
            Kit.time_range
              Type.(instant Ns)
              ~start:(Time.of_ns first) ~stop:(Time.of_ns last)
              (Exact (Time.Span.days 1))
              "ts"
          in
          let count =
            Int64.(to_int (unsigned_div (sub (pred last) first) day)) + 1
          in
          let ns =
            Array.map Time.to_ns
              (Column.values Kind.instant (Talon.column t "ts"))
          in
          equal int count (Array.length ns);
          equal int64 first ns.(0);
          equal int64
            Int64.(add first (mul (of_int (count - 1)) day))
            ns.(count - 1));
      cases "raises Invalid_argument" ~id:fst
        [
          ( "for a step of 0 days",
            ("Kit.time_range", range Type.date ~start:jan1 ~stop:feb1 (Days 0))
          );
          ( "for a step of -1 month",
            ( "Kit.time_range",
              range Type.date ~start:jan1 ~stop:feb1 (Months (-1)) ) );
          ( "for an exact step of 0",
            ( "Kit.time_range",
              range Type.date ~start:jan1 ~stop:feb1
                (Exact (Time.Span.of_ns 0L)) ) );
          ( "for a date by an exact step",
            ( "Temporal.offset",
              range Type.date ~start:jan1 ~stop:feb1 (Exact (Time.Span.days 1))
            ) );
          ( "for an instant by months",
            ( "Temporal.offset",
              range
                Type.(instant S)
                ~start:(instant 0L) ~stop:(instant 86_400L) (Months 1) ) );
          ( "for an instant in seconds by a millisecond",
            ( "Temporal.offset",
              range
                Type.(instant S)
                ~start:(instant 0L) ~stop:(instant 1L)
                (Exact (Time.Span.ms 1)) ) );
          ( "for 2^62 steps",
            ( "Kit.time_range",
              range
                Type.(instant Ns)
                ~start:(Time.of_ns Int64.min_int)
                ~stop:(Time.of_ns Int64.max_int)
                (Exact (Time.Span.of_ns 1L)) ) );
        ]
        (fun (_, (sub, f)) -> raises_match (Exn.invalid_arg ~sub) f);
    ]

let () = exit (run "Grids" [ scattering; laying_out; dividing; ranging ])
