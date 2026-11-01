(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module G = Talon_gen

let one name c = v [ (name, c) ]

(* [run e t] is the column of [e] over [t]'s rows, or the error. *)
let compute e t =
  Result.map
    (fun r -> column r "out")
    (Query.run (Query.select Expr.[ "out" := e ] (Query.of_table t)))

let result e t = require_ok ~pp:Error.pp (compute e t)
let error e t = Format.asprintf "%a" Error.pp (require_error (compute e t))

let rows_are ty expected c =
  equal
    (array (option (G.witness ty)))
    expected
    (Column.options (Type.kind ty) c)

let some vs = Array.map Option.some vs

(* Casts *)

let casts_of name ty vs into expected =
  test name (fun () ->
      let x = Col.v (Type.kind ty) "x" in
      rows_are into expected
        (result (Expr.cast into x) (one "x" (Column.of_options ty vs))))

let casts =
  group "Casts"
    [
      casts_of "an integer in range is kept" Type.int64
        (some [| -128; 127 |])
        Type.int8
        (some [| -128; 127 |]);
      casts_of "a whole float becomes an integer" Type.float64
        (some [| 2.; -0.; 1e9 |])
        Type.int32
        (some [| 2; 0; 1_000_000_000 |]);
      casts_of "bool takes 0 and 1, and gives them" Type.int8
        (some [| 0; 1 |])
        Type.bool
        (some [| false; true |]);
      casts_of "an integer rounds to the nearest float" Type.int64
        (some [| 16_777_217 |]) Type.float32 (some [| 16_777_216. |]);
      test "an unsigned integer takes floats up to 2^64" (fun () ->
          let c =
            result
              (Expr.cast Type.uint64 (Col.float "x"))
              (one "x" (Column.v Type.float64 [| 1e19 |]))
          in
          let bits = Nx.bitcast Nx.int64 (Column.to_tensor Nx.uint64 c) in
          (* 10^19 - 2^64: the bits of the uint64 10^19. *)
          equal (array int64)
            [| -8_446_744_073_709_551_616L |]
            (Nx.to_array bits));
      casts_of "text in the dictionary becomes categorical" Type.string
        (some [| "b"; "a" |])
        (Type.categorical [| "a"; "b" |])
        (some [| "b"; "a" |]);
      casts_of "a categorical becomes its text"
        (Type.categorical [| "a"; "b" |])
        [| Some "b"; None |] Type.string [| Some "b"; None |];
      casts_of "a finer unit multiplies" (Type.duration S)
        (some [| Time.Span.s 3 |])
        (Type.duration Ms)
        (some [| Time.Span.s 3 |]);
      casts_of "a coarser unit divides" (Type.datetime Ms)
        (some [| Time.of_ns (-2_000_000_000L) |])
        (Type.datetime S)
        (some [| Time.of_ns (-2_000_000_000L) |]);
      casts_of "lists cast element by element" (Type.list Type.int64)
        [| Some [| 1; 2 |]; None; Some [||] |]
        (Type.list Type.int8)
        [| Some [| 1; 2 |]; None; Some [||] |];
    ]

let cast_failures =
  let fails name ty vs into msg =
    test name (fun () ->
        let x = Col.v (Type.kind ty) "x" in
        expect
          (error (Expr.cast into x) (one "x" (Column.of_options ty vs)))
          msg)
  in
  group "Cast failures"
    [
      test "a literal fails at the first row, and over no rows not at all"
        (fun () ->
          let e = Expr.cast Type.int8 (Expr.int 300) in
          let t = one "x" (Column.v Type.int8 [| 1; 2 |]) in
          equal int 0
            (Column.length (result e (one "x" (Column.v Type.int8 [||]))));
          expect (error e t)
          @@ __POS_OF__
               {| select ["out" := cast int8 300]: row 0: cannot cast 300 to int8. |});
      fails "a fraction" Type.float64 (some [| 1.; 3.5 |]) Type.int32
      @@ __POS_OF__
           {| select ["out" := cast int32 x]: row 1: cannot cast 3.5 to int32. |};
      fails "a value out of range" Type.int16 (some [| 300 |]) Type.int8
      @@ __POS_OF__
           {| select ["out" := cast int8 x]: row 0: cannot cast 300 to int8. |};
      fails "a negative value to an unsigned type" Type.int8 (some [| -1 |])
        Type.uint64
      @@ __POS_OF__
           {| select ["out" := cast uint64 x]: row 0: cannot cast -1 to uint64. |};
      fails "NaN to an integer" Type.float32 (some [| Float.nan |]) Type.int64
      @@ __POS_OF__
           {| select ["out" := cast int64 x]: row 0: cannot cast nan to int64. |};
      fails "a finite float past the narrower float's range" Type.float64
        (some [| infinity; 1e300 |])
        Type.float32
      @@ __POS_OF__
           {| select ["out" := cast float32 x]: row 1: cannot cast 1e+300 to float32. |};
      fails "an integer past float16's range" Type.int32 (some [| 70000 |])
        Type.float16
      @@ __POS_OF__
           {| select ["out" := cast float16 x]: row 0: cannot cast 70000 to float16. |};
      fails "2 to bool" Type.int8 (some [| 2 |]) Type.bool
      @@ __POS_OF__
           {| select ["out" := cast bool x]: row 0: cannot cast 2 to bool. |};
      fails "text outside the dictionary" Type.string
        (some [| "a"; "z" |])
        (Type.categorical [| "a" |])
      @@ __POS_OF__
           {| select ["out" := cast categorical["a"] x]: row 1: cannot cast "z" to categorical["a"]. |};
      fails "a value that is not whole in the coarser unit" (Type.duration Ms)
        (some [| Time.Span.ms 1500 |])
        (Type.duration S)
      @@ __POS_OF__
           {| select ["out" := cast duration[s] x]: row 0: cannot cast 1s500ms to duration[s]. |};
      fails "a list element, at its row" (Type.list Type.int64)
        [| Some [| 1 |]; None; Some [| 2; 300 |] |]
        (Type.list Type.int8)
      @@ __POS_OF__
           {| select ["out" := cast list[int8] x]: row 2: cannot cast 300 to int8. |};
    ]

(* A cast keeps the values of its operand's kind that its type holds, and fails
   at the first it does not. *)

type pair = Pair : 'a Type.t * 'a Type.t -> pair
type case = Case : 'a Type.t * 'a Type.t * 'a option array -> case

let pairs =
  let all ts =
    List.concat_map (fun a -> List.map (fun b -> Pair (a, b)) ts) ts
  in
  let units = Type.[ S; Ms; Us; Ns ] in
  Type.(
    all [ int8; int16; int32; int64; uint8; uint16; uint32; uint64 ]
    @ all (List.map duration units)
    @ all (List.map clock units)
    @ all (List.map (fun u -> datetime u) units)
    @ all [ datetime ~zone:"UTC" S; datetime ~zone:"Europe/Paris" Ns ]
    @ all [ string; categorical [| "a"; "é"; "" |]; categorical [||] ])

let pp_case ppf (Case (a, b, vs)) =
  Format.fprintf ppf "cast %a of %a" Type.pp b G.pp_sample (G.Sample (a, vs))

let held_casts =
  let gen =
    Gen.bind (Gen.of_list pairs) (fun (Pair (a, b)) ->
        Gen.map (fun vs -> Case (a, b, vs)) (G.options a))
  in
  prop "a cast keeps the values its type holds" (Gen.with_pp pp_case gen)
    (fun (Case (a, b, vs)) ->
      let x = Col.v (Type.kind a) "x" in
      let r = compute (Expr.cast b x) (one "x" (Column.of_options a vs)) in
      let unheld = function Some v -> not (Type.holds b v) | None -> false in
      match Array.find_index unheld vs with
      | None -> rows_are b vs (require_ok ~pp:Error.pp r)
      | Some row ->
          let e = Format.asprintf "%a" Error.pp (require_error r) in
          contains ~sub:(Printf.sprintf ": row %d: cannot cast " row) e)

(* Text *)

let s = Col.string "s"
let texts vs = one "s" (Column.of_options Type.string vs)

(* [index s sub i] is the first byte from [i] at which [sub] lies whole in
   [s]. *)
let rec index s sub i =
  let n = String.length sub in
  if i + n > String.length s then None
  else if String.sub s i n = sub then Some i
  else index s sub (i + 1)

let rec pieces_in s i = function
  | [] -> true
  | p :: ps -> (
      match index s p i with
      | Some j -> pieces_in s (j + String.length p) ps
      | None -> false)

(* Text and needles over few letters, so that partial matches abound. *)
let letters ~min ~max =
  Gen.map (String.concat "")
    (Gen.list ~size:(Gen.int_range min max) (Gen.of_list [ "a"; "b"; "é" ]))

let pieces_law =
  let gen =
    Gen.pair
      (Gen.list ~size:(Gen.int_range 0 8) (letters ~min:0 ~max:40))
      (Gen.list ~size:(Gen.int_range 1 3) (letters ~min:1 ~max:3))
  in
  prop "matches finds pieces in order, without overlap" gen (fun (vs, ps) ->
      let hit v = pieces_in v 0 ps in
      cover "a match" (List.exists hit vs);
      cover "a miss" (List.exists (fun v -> not (hit v)) vs);
      let vs = Array.of_list (List.map Option.some vs) in
      let got =
        Column.options Kind.bool
          (result (Expr.Str.matches (Expr.Str.pieces ps) s) (texts vs))
      in
      equal (array (option bool)) (Array.map (Option.map hit) vs) got)

let literal_law =
  let gen =
    Gen.pair
      (Gen.list ~size:(Gen.int_range 0 8) (letters ~min:0 ~max:6))
      (letters ~min:0 ~max:4)
  in
  let ops =
    [
      (Expr.( = ), Int.equal 0);
      (Expr.( <> ), fun c -> c <> 0);
      (Expr.( < ), fun c -> c < 0);
      (Expr.( <= ), fun c -> c <= 0);
      (Expr.( > ), fun c -> c > 0);
      (Expr.( >= ), fun c -> c >= 0);
    ]
  in
  prop "text compares with a literal as its bytes do" gen (fun (vs, lit) ->
      cover "a row equal to the literal" (List.mem lit vs);
      cover "a row the literal is a prefix of"
        (List.exists (fun v -> v <> lit && String.starts_with ~prefix:lit v) vs);
      let vs = Array.of_list (List.map Option.some vs) in
      List.iter
        (fun (op, holds) ->
          let got =
            Column.options Kind.bool
              (result (op s (Expr.string lit)) (texts vs))
          in
          let expected =
            Array.map (Option.map (fun v -> holds (String.compare v lit))) vs
          in
          equal (array (option bool)) expected got)
        ops)

let text =
  let ints e vs = Column.options Kind.int (result e (texts vs)) in
  let strings e vs = Column.options Kind.string (result e (texts vs)) in
  let bools e vs = Column.options Kind.bool (result e (texts vs)) in
  group "Text"
    [
      test "length counts scalar values" (fun () ->
          equal
            (array (option int))
            [| Some 5; Some 0; None |]
            (ints (Expr.Str.length s) [| Some "héllo"; Some ""; None |]));
      test "slice counts scalar values, from the end when negative" (fun () ->
          let vs = [| Some "héllo"; Some "ab"; None |] in
          equal
            (array (option string))
            [| Some "él"; Some "b"; None |]
            (strings (Expr.Str.slice ~offset:1 ~length:2 s) vs);
          equal
            (array (option string))
            [| Some "ll"; Some "a"; None |]
            (strings (Expr.Str.slice ~offset:(-3) ~length:2 s) vs));
      test "matches" (fun () ->
          let vs =
            [| Some "special requests"; Some "requests special"; None |]
          in
          let m p = bools (Expr.Str.matches p s) vs in
          equal
            (array (option bool))
            [| Some true; Some false; None |]
            (m (Expr.Str.pieces [ "special"; "requests" ]));
          equal
            (array (option bool))
            [| Some true; Some false; None |]
            (m (Expr.Str.prefix "spe"));
          equal
            (array (option bool))
            [| Some false; Some true; None |]
            (m (Expr.Str.suffix "cial"));
          equal
            (array (option bool))
            [| Some true; Some true; None |]
            (m (Expr.Str.literal "q")));
      pieces_law;
      literal_law;
      test "parse fails with the text" (fun () ->
          expect
            (error
               (Expr.Str.parse Type.int32 s)
               (texts [| Some "12"; Some "x1" |]))
          @@ __POS_OF__
               {| select ["out" := Str.parse int32 s]: row 1: "x1": not an integer. |});
    ]

(* [scalars s] is the scalar values of [s], each as its UTF-8 bytes. *)
let scalars s =
  let rec go i acc =
    if i >= String.length s then List.rev acc
    else
      let n = Uchar.utf_decode_length (String.get_utf_8_uchar s i) in
      go (i + n) (String.sub s i n :: acc)
  in
  go 0 []

let slices =
  let text = Option.get (G.value Type.string) in
  let gen =
    Gen.triple
      (Gen.array ~size:(Gen.int_range 0 6) (Gen.option text))
      (Gen.int_range (-8) 8) (Gen.int_range 0 8)
  in
  prop "slice and length count the scalar values of String.get_utf_8_uchar" gen
    (fun (vs, offset, length) ->
      let slice s =
        let us = scalars s in
        let n = List.length us in
        let p = if offset >= 0 then offset else n + offset in
        String.concat "" (List.filteri (fun i _ -> i >= p && i < p + length) us)
      in
      let t = texts vs in
      equal
        (array (option string))
        (Array.map (Option.map slice) vs)
        (Column.options Kind.string
           (result (Expr.Str.slice ~offset ~length s) t));
      equal
        (array (option int))
        (Array.map (Option.map (fun s -> List.length (scalars s))) vs)
        (Column.options Kind.int (result (Expr.Str.length s) t)))

(* Time *)

let date y m d = Option.get (Time.Date.of_civil (y, m, d))
let dates vs = one "d" (Column.of_options Type.date vs)
let d = Col.date "d"

let time =
  let ints e t = Column.options Kind.int (result e t) in
  let days e t = Column.options Kind.date (result e t) in
  group "Time"
    [
      test "fields of dates" (fun () ->
          let t =
            dates
              [|
                Some (date 1970 1 1);
                Some (date 2024 12 31);
                Some (date 2024 3 15);
              |]
          in
          let f field = ints (Expr.Temporal.field field d) t in
          equal (array (option int)) (some [| 1970; 2024; 2024 |]) (f `Year);
          equal (array (option int)) (some [| 4; 2; 5 |]) (f `Weekday);
          equal (array (option int)) (some [| 1; 366; 75 |]) (f `Yearday));
      test "fields of datetimes before 1970" (fun () ->
          let t =
            one "t" (Column.v (Type.datetime Ns) [| Time.of_ns (-1L) |])
          in
          let f field = ints (Expr.Temporal.field field (Col.instant "t")) t in
          equal (array (option int)) (some [| 1969 |]) (f `Year);
          equal (array (option int)) (some [| 23 |]) (f `Hour);
          equal (array (option int)) (some [| 59 |]) (f `Second);
          equal (array (option int)) (some [| 999_999_999 |]) (f `Nanosecond));
      test "fields of datetimes at the ends of their ticks" (fun () ->
          let t =
            one "t"
              (Column.v (Type.datetime Ns)
                 [| Time.of_ns Int64.min_int; Time.of_ns Int64.max_int |])
          in
          let f field = ints (Expr.Temporal.field field (Col.instant "t")) t in
          equal (array (option int)) (some [| 1677; 2262 |]) (f `Year);
          equal (array (option int)) (some [| 9; 4 |]) (f `Month);
          equal (array (option int)) (some [| 21; 11 |]) (f `Day);
          equal (array (option int)) (some [| 0; 23 |]) (f `Hour));
      test
        "a datetime's day whose midnight is below the least tick has no floor"
        (fun () ->
          let t =
            one "t" (Column.v (Type.datetime Ns) [| Time.of_ns Int64.min_int |])
          in
          expect (error (Expr.Temporal.floor (Days 1) (Col.instant "t")) t)
          @@ __POS_OF__
               {| select ["out" := Temporal.floor 1d t]: row 0: the 1d period of 1677-09-21T00:12:43.145224192 is out of range. |});
      test "add and diff" (fun () ->
          let t = dates [| Some (date 2024 2 28) |] in
          equal
            (array (option int))
            [| Some (Time.Date.to_days (date 2024 3 1)) |]
            (Array.map
               (Option.map Time.Date.to_days)
               (days Expr.(Temporal.add d (span (Time.Span.days 2))) t));
          expect
            (error
               Expr.(Temporal.add d (Col.span "x"))
               (v
                  [
                    ("d", Column.v Type.date [| date 2024 2 28 |]);
                    ("x", Column.v (Type.duration S) [| Time.Span.hours 36 |]);
                  ]))
          @@ __POS_OF__
               {| select ["out" := Temporal.add d x]: row 0: 36h is not whole days. |});
      test "add to a time of day" (fun () ->
          let t = one "c" (Column.v (Type.clock Ms) [| Time.Span.hours 9 |]) in
          equal
            (array (option int64))
            [| Some (Time.Span.to_ns (Time.Span.minutes 630)) |]
            (Array.map
               (Option.map Time.Span.to_ns)
               (Column.options Kind.span
                  (result
                     Expr.(
                       Temporal.add (Col.span "c") (span (Time.Span.minutes 90)))
                     t))));
      test "floor and offset" (fun () ->
          let t = dates [| Some (date 2024 1 31); Some (date 2024 3 15) |] in
          let on e = Array.map (Option.map Time.Date.to_days) (days e t) in
          let ds l =
            Array.map
              (fun (y, m, dd) -> Some (Time.Date.to_days (date y m dd)))
              l
          in
          equal
            (array (option int))
            (ds [| (2024, 2, 29); (2024, 4, 15) |])
            (on (Expr.Temporal.offset (Months 1) d));
          equal
            (array (option int))
            (ds [| (2024, 1, 1); (2024, 1, 1) |])
            (on (Expr.Temporal.floor (Months 3) d));
          equal
            (array (option int))
            (ds [| (2024, 1, 29); (2024, 3, 11) |])
            (on (Expr.Temporal.floor (Weeks 1) d)));
      test "an offset reads as UTC, and a zoned datetime writes in UTC"
        (fun () ->
          let fmt = "%Y-%m-%dT%H:%M:%S%z"
          and ty = Type.datetime ~zone:"UTC" S in
          let t = texts [| Some "2024-03-15T10:00:00+02:00" |] in
          let r = Expr.Temporal.(format fmt (parse fmt ty s)) in
          equal
            (array (option string))
            [| Some "2024-03-15T08:00:00Z" |]
            (Column.options Kind.string (result r t)));
      test "an exact step is whole ticks of the datetime's unit" (fun () ->
          let at u ns = one "t" (Column.v (Type.datetime u) [| Time.of_ns ns |])
          and t = Col.instant "t"
          and ms n = Time.Exact (Time.Span.ms n) in
          let problem e u =
            match compute e (at u 0L) with
            | _ -> "no problem"
            | exception Invalid_argument m -> m
          in
          expect
            (String.concat "\n"
               Expr.Temporal.
                 [
                   problem (floor (ms 1500) t) S;
                   problem (offset (ms 500) t) S;
                   problem (floor (ms 1500) t) Ms;
                 ])
          @@ __POS_OF__
               {|
            select: 1 problem
              "out" := Temporal.floor 1s500ms t
                Temporal.floor moves datetime[s] by multiples of 1s, not 1s500ms.
              input (1 column): t datetime[s]
            select: 1 problem
              "out" := Temporal.offset 500ms t
                Temporal.offset moves datetime[s] by multiples of 1s, not 500ms.
              input (1 column): t datetime[s]
            no problem
            |};
          let instants c =
            Array.map (Option.map Time.to_ns) (Column.options Kind.instant c)
          in
          equal
            (array (option int64))
            [| Some 3_000_000_000L |]
            (instants
               (result Expr.Temporal.(floor (ms 1500) t) (at Ms 3_400_000_000L))));
      test "format and parse" (fun () ->
          let t = dates [| Some (date 2024 3 15); None |] in
          equal
            (array (option string))
            [| Some "15/03/2024"; None |]
            (Column.options Kind.string
               (result (Expr.Temporal.format "%d/%m/%Y" d) t));
          expect
            (error
               (Expr.Temporal.parse "%d/%m/%Y" Type.date s)
               (texts [| Some "15/03/2024"; Some "2024-03-15" |]))
          @@ __POS_OF__
               {| select ["out" := Temporal.parse "%d/%m/%Y" date s]: row 1: "2024-03-15": not in the format "%d/%m/%Y". |});
    ]

let fields_agree =
  let days =
    Gen.array ~size:(Gen.int_range 0 20) (Gen.int_range (-1_000_000) 1_000_000)
  in
  prop
    "date fields agree with Time.Date.to_civil over a million days either way"
    days (fun ds ->
      let dates = Array.map (fun n -> Option.get (Time.Date.of_days n)) ds in
      let t = one "d" (Column.v Type.date dates) in
      let field f =
        Column.values Kind.int (result (Expr.Temporal.field f d) t)
      in
      let civil = Array.map Time.Date.to_civil dates in
      let yearday dt (y, _, _) =
        Time.Date.to_days dt - Time.Date.to_days (date y 1 1) + 1
      in
      equal (array int) (Array.map (fun (y, _, _) -> y) civil) (field `Year);
      equal (array int) (Array.map (fun (_, m, _) -> m) civil) (field `Month);
      equal (array int) (Array.map (fun (_, _, d) -> d) civil) (field `Day);
      equal (array int) (Array.map2 yearday dates civil) (field `Yearday))

(* Datetimes of every unit, with and without a zone, drawn at the edges of their
   int64 ticks and of {!Time.Date}'s days as well as anywhere. *)
let extreme_ticks =
  let per : Type.unit_ -> int64 = function
    | S -> 1L
    | Ms -> 1_000L
    | Us -> 1_000_000L
    | Ns -> 1_000_000_000L
  in
  let open Gen in
  let* u = of_list Type.[ S; Ms; Us; Ns ] in
  let* zoned = bool in
  let scaled secs =
    let t = Int64.mul secs (per u) in
    if Int64.div t (per u) = secs then [ t ] else []
  in
  let edges =
    [ Int64.min_int; Int64.succ Int64.min_int; -1L; 0L; 1L ]
    @ [ Int64.pred Int64.max_int; Int64.max_int ]
    @ List.concat_map scaled
        [
          185542587187199L;
          185542587187200L;
          -185542587187200L;
          -185542587187201L;
        ]
  in
  let+ xs =
    array ~size:(int_range 0 20) (frequency [ (1, of_list edges); (1, int64) ])
  in
  (u, zoned, xs)

(* Temporal.add over every operand type and every pairing of its unit with a
   span's unit that it takes, ticks drawn at their bounds as well as anywhere.
   The model computes each row in exact int64 arithmetic: a row whose span does
   not scale to the operand's ticks, whose sum overflows, or whose result leaves
   the operand's range fails the run, at the first such row. *)

type operand = Datetime | Duration | Clock | Date

let per_s : Type.unit_ -> int64 = function
  | S -> 1L
  | Ms -> 1_000L
  | Us -> 1_000_000L
  | Ns -> 1_000_000_000L

let int32_days = (Int64.of_int32 Int32.min_int, Int64.of_int32 Int32.max_int)

let adds =
  let open Gen in
  let units = Type.[ S; Ms; Us; Ns ] in
  let* op = of_list [ Datetime; Duration; Clock; Date ] in
  let* u = of_list units in
  (* A span finer than the operand's unit is a problem of the verb, and a date's
     unit is the second. *)
  let* du =
    let unit_ = if op = Date then Type.S else u in
    of_list (List.filter (fun d -> per_s d <= per_s unit_) units)
  in
  let edges =
    [ Int64.min_int; Int64.succ Int64.min_int; -1L; 0L; 1L ]
    @ [ Int64.pred Int64.max_int; Int64.max_int ]
  in
  let operand =
    match op with
    | Datetime | Duration -> frequency [ (1, of_list edges); (1, int64) ]
    | Clock ->
        let last = Int64.pred (Int64.mul 86_400L (per_s u)) in
        frequency [ (1, of_list [ 0L; 1L; last ]); (1, int64_range 0L last) ]
    | Date ->
        let lo, hi = int32_days in
        frequency
          [
            (1, of_list [ lo; Int64.succ lo; 0L; Int64.pred hi; hi ]);
            (1, int64_range lo hi);
          ]
  in
  let span =
    frequency
      [
        (2, of_list edges);
        (2, int64);
        ( 1,
          map
            (Int64.mul (Int64.mul 86_400L (per_s du)))
            (int64_range (-1000L) 1000L) );
        (1, int64_range (-1000L) 1000L);
      ]
  in
  let+ rows = array ~size:(int_range 0 12) (pair operand span) in
  (op, u, du, rows)

(* [mul_exact k f] is [k * f] for [f > 0], or [None] if it overflows. *)
let mul_exact k f =
  let r = Int64.mul k f in
  if Int64.div r f = k then Some r else None

(* [add_exact x s] is [x + s], or [None] if it overflows. *)
let add_exact x s =
  let r = Int64.add x s in
  if Int64.compare s 0L >= 0 = (Int64.compare r x >= 0) then Some r else None

let model op u du x k =
  let ( let* ) = Option.bind in
  match op with
  | Date ->
      let day = Int64.mul 86_400L (per_s du) in
      let r = Int64.add x (Int64.div k day) in
      let lo, hi = int32_days in
      if Int64.rem k day <> 0L || r < lo || r > hi then None else Some r
  | Datetime | Duration | Clock ->
      let* s = mul_exact k (Int64.div (per_s u) (per_s du)) in
      let* r = add_exact x s in
      let day = Int64.mul 86_400L (per_s u) in
      if op = Clock && (r < 0L || r >= day) then None else Some r

let adds_ticks =
  prop
    "add moves a temporal value by the span's ticks, failing at the first row \
     out of range"
    adds (fun (op, u, du, rows) ->
      let column ty dt xs =
        let values = Nx.P (Nx.create dt [| Array.length xs |] xs) in
        match Column.of_layout ty (Fixed { validity = None; values }) with
        | Ok c -> c
        | Error (row, why) -> failf "row %d: %s" row why
      in
      let x = Array.map fst rows and k = Array.map snd rows in
      let a =
        match op with
        | Datetime -> column (Any (Type.datetime ~zone:"UTC" u)) Nx.int64 x
        | Duration -> column (Any (Type.duration u)) Nx.int64 x
        | Clock -> column (Any (Type.clock u)) Nx.int64 x
        | Date -> column (Any Type.date) Nx.int32 (Array.map Int64.to_int32 x)
      in
      let t =
        v [ ("a", a); ("k", column (Any (Type.duration du)) Nx.int64 k) ]
      in
      let d = Col.span "k" in
      let out =
        match op with
        | Datetime -> compute Expr.(Temporal.add (Col.instant "a") d) t
        | Duration | Clock -> compute Expr.(Temporal.add (Col.span "a") d) t
        | Date -> compute Expr.(Temporal.add (Col.date "a") d) t
      in
      let ticks c =
        match op with
        | Date ->
            Array.map Int64.of_int32 (Nx.to_array (Column.to_tensor Nx.int32 c))
        | _ -> Nx.to_array (Column.to_tensor Nx.int64 c)
      in
      let expected = Array.map2 (model op u du) x k in
      cover "the span's unit is the operand's" (op <> Date && u = du);
      cover "a coarser span" (op <> Date && per_s du < per_s u);
      cover "every row in range"
        (Array.for_all Option.is_some expected && rows <> [||]);
      cover "a row out of range" (Array.exists Option.is_none expected);
      cover "a date" (op = Date && Array.exists Option.is_some expected);
      cover "a clock" (op = Clock && Array.exists Option.is_some expected);
      match Array.find_index Option.is_none expected with
      | None ->
          equal (array int64)
            (Array.map Option.get expected)
            (ticks (require_ok ~pp:Error.pp out))
      | Some row ->
          contains
            ~sub:(Printf.sprintf ": row %d: " row)
            (Format.asprintf "%a" Error.pp (require_error out)))

let formats_round_trip =
  prop "parse reads back what format writes, at every tick" extreme_ticks
    (fun (u, zoned, xs) ->
      let ty, fmt =
        if zoned then (Type.datetime ~zone:"UTC" u, "%Y-%m-%d %H:%M:%S.%f%z")
        else (Type.datetime u, "%Y-%m-%d %H:%M:%S.%f")
      in
      let values = Nx.P (Nx.create Nx.int64 [| Array.length xs |] xs) in
      let c =
        match Column.of_layout (Any ty) (Fixed { validity = None; values }) with
        | Ok c -> c
        | Error (row, why) -> failf "row %d: %s" row why
      in
      let text = Expr.Temporal.format fmt (Col.instant "t") in
      let back = result (Expr.Temporal.parse fmt ty text) (one "t" c) in
      equal (array int64) xs (Nx.to_array (Column.to_tensor Nx.int64 back)))

(* Cuts *)

(* Types that talon's comparisons order: numbers, temporal values, text and
   compound values. Floats are listed twice, so that NaN and the infinities are
   drawn in every run, and tensors, which alone take cut's path of compound
   words, three times. *)
let orderable =
  Type.
    [
      Any int8;
      Any int16;
      Any int32;
      Any int64;
      Any uint8;
      Any uint16;
      Any uint32;
      Any uint64;
      Any float16;
      Any float32;
      Any float64;
      Any float32;
      Any float64;
      Any bool;
      Any date;
      Any (clock Ns);
      Any (duration Ms);
      Any (datetime Us);
      Any (datetime ~zone:"UTC" S);
      Any string;
      Any binary;
      Any (categorical [| "b"; "a"; "é"; "" |]);
      Any (list int8);
      Any (record [ ("a", Any int8); ("b", Any string) ]);
      Any (tensor Nx.float32 [| 2 |]);
      Any (tensor Nx.float32 [| 2 |]);
      Any (tensor Nx.int8 [| 1; 2 |]);
    ]

let is_nan : type a. a Type.t -> a -> bool =
 fun ty v ->
  match Kind.provably_equal (Type.kind ty) Kind.float with
  | Some Equal -> Float.is_nan v
  | None -> false

(* [specials ty] is NaN, the infinities and [-0.] for a float type. *)
let specials : type a. a Type.t -> a list =
 fun ty ->
  match Kind.provably_equal (Type.kind ty) Kind.float with
  | Some Equal -> [ Float.nan; Float.nan; infinity; neg_infinity; -0. ]
  | None -> []

let is_tensor : type a. a Type.t -> bool = function
  | Tensor _ -> true
  | _ -> false

(* [unrounded ty] draws floats that a narrow float type holds and rounds, as
   [0.7], which float32 stores as [0.699999988]. *)
let unrounded : type a. a Type.t -> a Gen.t option =
 fun ty ->
  match Kind.provably_equal (Type.kind ty) Kind.float with
  | Some Equal -> Some (Gen.float_range (-1000.) 1000.)
  | None -> None

(* [ascending ty vs] is the edges [vs] as given and as a column of [ty] stores
   them, without NaN, in talon's stored order and without stored repeats. *)
let ascending ty vs =
  let stored = Column.values (Type.kind ty) (Column.v ty vs) in
  let pairs = List.combine (Array.to_list vs) (Array.to_list stored) in
  let pairs = List.filter (fun (_, s) -> not (is_nan ty s)) pairs in
  let pairs =
    List.sort_uniq (fun (_, a) (_, b) -> Type.compare_value ty a b) pairs
  in
  (Array.of_list (List.map fst pairs), Array.of_list (List.map snd pairs))

type cut_case =
  | Cut_case : 'a Type.t * 'a array * 'a array * 'a option array -> cut_case

(* [pp_edges ty] formats edges as given: floats with every digit, before the
   type rounds them. *)
let pp_edges : type a. a Type.t -> Format.formatter -> a array -> unit =
 fun ty ppf vs ->
  match Kind.provably_equal (Type.kind ty) Kind.float with
  | Some Equal -> Array.iter (fun v -> Format.fprintf ppf "%.17g@ " v) vs
  | None -> G.pp_sample ppf (G.Sample (ty, Array.map Option.some vs))

let pp_cut_case ppf (Cut_case (ty, raw, es, xs)) =
  Format.fprintf ppf "@[<v>edges @[%a@]@,stored %a@,rows %a@]" (pp_edges ty) raw
    (pp_edges ty) es G.pp_sample
    (G.Sample (ty, xs))

(* Edges drawn at a type's bounds as well as anywhere, and rows on the edges,
   anywhere and null. *)
let cut_cases =
  let open Gen in
  let* (Type.Any ty) = of_list orderable in
  let value = Option.get (G.value ty) in
  let edge =
    match unrounded ty with
    | None -> value
    | Some raw -> frequency [ (1, value); (2, raw) ]
  in
  let* raw = array ~size:(int_range 2 6) edge in
  let raw, es = ascending ty raw in
  let n = Array.length es in
  let on_edge = map (fun i -> Some es.(i mod n)) nat in
  let rows =
    [ (3, map Option.some value); (1, constant None) ]
    @ (if n = 0 then [] else [ (3, on_edge) ])
    @
    match specials ty with
    | [] -> []
    | vs -> [ (3, map Option.some (of_list vs)) ]
  in
  let+ xs = array ~size:(int_range 0 12) (frequency rows) in
  Cut_case (ty, raw, es, xs)

let cut_law =
  prop "cut is the bin that a linear scan of the edges finds"
    (Gen.with_pp pp_cut_case cut_cases) (fun (Cut_case (ty, raw, es, xs)) ->
      assume (Array.length es >= 2);
      let c = Column.of_options ty xs in
      let xs = Column.options (Type.kind ty) c in
      let cmp = Type.compare_value ty and last = Array.length es - 1 in
      let rec scan x i =
        if i >= last then None
        else if cmp es.(i) x <= 0 && cmp x es.(i + 1) < 0 then Some i
        else scan x (i + 1)
      in
      let bin x = scan x 0 in
      let on_edge x = Array.exists (fun e -> cmp e x = 0) es in
      let has p = Array.exists (function Some x -> p x | None -> false) xs in
      cover "a value on an edge" (has on_edge);
      cover "a value between two edges"
        (has (fun x -> Option.is_some (bin x) && not (on_edge x)));
      cover "a value below the first edge" (has (fun x -> cmp x es.(0) < 0));
      cover "a value on or above the last edge"
        (has (fun x -> cmp x es.(last) >= 0));
      cover "a null" (Array.exists Option.is_none xs);
      cover "NaN" (has (is_nan ty));
      cover "an edge that its type rounds"
        (Array.exists2 (fun r e -> Type.compare_value ty r e <> 0) raw es);
      cover "a tensor" (is_tensor ty);
      let x = Col.v (Type.kind ty) "x" in
      equal
        (array (option int))
        (Array.map (fun x -> Option.bind x bin) xs)
        (Column.options Kind.int (result (Expr.cut raw x) (one "x" c))))

let epoch ~ordered =
  Ext.v ~name:"epoch" ~ordered Type.float64 ~dec:Fun.id ~enc:Fun.id

let epochs xs =
  let ty = Type.Any (Type.ext ~name:"epoch" Type.float64) in
  let stored = Column.layout (Column.v Type.float64 xs) in
  Result.get_ok (Column.of_layout ty stored)

let cuts =
  let bins ty es xs =
    let x = Col.v (Type.kind ty) "x" in
    Column.options Kind.int
      (result (Expr.cut es x) (one "x" (Column.of_options ty xs)))
  in
  let problem e t =
    match compute e t with
    | _ -> "no problem"
    | exception Invalid_argument m -> m
  in
  group "Cuts"
    [
      cut_law;
      test
        "bins hold their lower edge; NaN, values on the last edge and nulls \
         are in none" (fun () ->
          equal
            (array (option int))
            [|
              Some 0;
              Some 1;
              Some 1;
              Some 2;
              Some 2;
              None;
              None;
              None;
              None;
              None;
            |]
            (bins Type.float64
               [| -1.; 0.; 1.; infinity |]
               [|
                 Some (-1.);
                 Some (-0.);
                 Some 0.5;
                 Some 1.;
                 Some Float.max_float;
                 Some infinity;
                 Some neg_infinity;
                 Some Float.nan;
                 Some (-2.);
                 None;
               |]));
      test "the edge -0. is the edge 0." (fun () ->
          equal
            (array (option int))
            [| Some 0; Some 0; None |]
            (bins Type.float64 [| -0.; 1. |] (some [| 0.; -0.; -1e-300 |])));
      test "edges take the operand's type, as literals do" (fun () ->
          (* float32 stores 0.7 as 0.699999988, which [x >= float 0.7] holds
             for. *)
          equal
            (array (option int))
            [| Some 0; Some 0; None |]
            (bins Type.float32 [| 0.7; 1. |] (some [| 0.7; 0.9; 1. |])));
      test "integer edges at the ends of their types" (fun () ->
          equal
            (array (option int))
            [| Some 0; Some 0; Some 1; Some 1; None |]
            (bins Type.int64 [| min_int; 0; max_int |]
               (some [| min_int; -1; 0; max_int - 1; max_int |]));
          equal
            (array (option int))
            [| Some 0; Some 0; None |]
            (bins Type.uint8 [| 0; 255 |] (some [| 0; 254; 255 |])));
      test "the edges are copied" (fun () ->
          let es = [| 0; 10 |] in
          let e = Expr.cut es (Col.int "x") in
          es.(1) <- 1;
          equal
            (array (option int))
            [| Some 0 |]
            (Column.options Kind.int
               (result e (one "x" (Column.v Type.int64 [| 5 |])))));
      test "an extension declared ordered cuts its values" (fun () ->
          let e = Expr.cut [| 0.; 10. |] (Ext.col (epoch ~ordered:true) "e") in
          equal
            (array (option int))
            [| None; Some 0; Some 0; None |]
            (Column.options Kind.int
               (result e (one "e" (epochs [| -1.; 0.; 5.; 10. |])))));
      test "edges that do not ascend strictly, or a type that has no order"
        (fun () ->
          let t =
            v
              [
                ("x", Column.v Type.float64 [| 1. |]);
                ("f", Column.v Type.float32 [| 1. |]);
                ("i", Column.v Type.int8 [| 1 |]);
                ("n", Column.v Type.int64 [| 1 |]);
                ("c", Column.v (Type.clock S) [| Time.Span.s 1 |]);
                ("e", epochs [| 1. |]);
              ]
          in
          let x = Col.float "x" and n = Col.int "n" in
          expect
            (String.concat "\n"
               [
                 problem (Expr.cut [| 1.; 1. |] x) t;
                 problem (Expr.cut [| 2; 1 |] n) t;
                 problem (Expr.cut [| -0.; 0. |] x) t;
                 problem (Expr.cut [| 0.; Float.nan |] x) t;
                 problem (Expr.cut [| 0.7; 0.70000001 |] (Col.float "f")) t;
                 problem (Expr.cut [| 0; 1000 |] (Col.int "i")) t;
                 problem
                   (Expr.cut
                      [| Time.Span.s 0; Time.Span.days 1 |]
                      (Col.span "c"))
                   t;
                 problem
                   (Expr.cut [| 0.; 1. |] (Ext.col (epoch ~ordered:false) "e"))
                   t;
               ])
          @@ __POS_OF__
               {|
            select: 1 problem
              "out" := cut […] x
                cut takes edges that ascend strictly, and 1. then 1. do not, as float64 stores them.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] n
                cut takes edges that ascend strictly, and 2 then 1 do not, as int64 stores them.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] x
                cut takes edges that ascend strictly, and -0. then 0. do not, as float64 stores them.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] x
                cut takes no NaN edge.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] f
                cut takes edges that ascend strictly, and 0.699999988079071 then 0.699999988079071 do not, as float32 stores them.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] i
                int8 does not hold the edge 1000.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] c
                clock[s] does not hold the edge 24h.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] e
                cut orders values, and the declaration of ext[epoch, float64] is not ~ordered:true.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            |});
    ]

(* Temporal casts *)

let ticks_per_day u = Int64.mul 86_400L (per_s u)

(* [floor_div x k] is [x / k] rounded toward negative infinity, for [k > 0]. *)
let floor_div x k =
  let q = Int64.div x k in
  if Int64.rem x k < 0L then Int64.pred q else q

(* [int64_options c] is the storage of each row of the temporal column [c] as
   int64, [None] under a null. *)
let int64_options c =
  match Column.layout c with
  | Fixed { validity; values = Nx.P x } -> (
      let x = Nx.to_array (Nx.cast Nx.int64 x) in
      match validity with
      | None -> Array.map Option.some x
      | Some valid ->
          let valid = Nx.to_array (Nx.cast Nx.bool valid) in
          Array.mapi (fun i t -> if valid.(i) then Some t else None) x)
  | _ -> fail "a temporal column has a fixed-width layout"

(* [ticks_column ty xs] is the column of the temporal type [ty] that stores
   [xs], int32 days for [date]. *)
let ticks_column (Type.Any ty as t) xs =
  let n = Array.length xs in
  let validity =
    Nx.cast Nx.bit (Nx.create Nx.bool [| n |] (Array.map Option.is_some xs))
  in
  let ticks = Array.map (Option.value ~default:0L) xs in
  let values =
    match ty with
    | Date -> Nx.P (Nx.create Nx.int32 [| n |] (Array.map Int64.to_int32 ticks))
    | _ -> Nx.P (Nx.create Nx.int64 [| n |] ticks)
  in
  match Column.of_layout t (Fixed { validity = Some validity; values }) with
  | Ok c -> c
  | Error (row, why) -> failf "row %d: %s" row why

(* [cast_ticks ty into xs] is [cast into] of the column of [ty] that stores
   [xs]. *)
let cast_ticks (Type.Any ty as t) (Type.Any into) xs =
  let x = Col.v (Type.kind ty) "x" in
  compute (Expr.cast into x) (one "x" (ticks_column t xs))

(* [agrees ~expected xs r] checks the run [r] of a cast of the rows [xs] whose
   model gives [expected], [None] where a row fails: the run fails at the first
   row that fails and is not null, and otherwise gives each row's storage, null
   where [xs] is. *)
let agrees ?(failure = "cannot cast ") ~expected xs r =
  let fails i = Option.is_some xs.(i) && Option.is_none expected.(i) in
  match List.find_opt fails (List.init (Array.length xs) Fun.id) with
  | None ->
      let expected =
        Array.map2 (fun x e -> Option.bind x (fun _ -> e)) xs expected
      in
      equal
        (array (option int64))
        expected
        (int64_options (require_ok ~pp:Error.pp r))
  | Some row ->
      contains
        ~sub:(Printf.sprintf ": row %d: %s" row failure)
        (Format.asprintf "%a" Error.pp (require_error r))

let units = Type.[ S; Ms; Us; Ns ]
let int32_lo, int32_hi = int32_days

(* The days that a datetime[ns] reaches either side of 1970, about 292 years. *)
let ns_days = Int64.div Int64.max_int (ticks_per_day Ns)

let fails xs expected =
  Array.exists2 (fun x e -> x <> None && e = None) xs expected

let before_1970 xs =
  Array.exists (function Some x -> x < 0L | None -> false) xs

let dates_to_datetimes =
  let open Gen in
  let days =
    frequency
      [
        (1, of_list [ int32_lo; Int64.succ int32_lo; Int64.pred int32_hi ]);
        (1, of_list [ int32_hi; Int64.neg (Int64.succ ns_days) ]);
        (1, of_list [ Int64.neg ns_days; -1L; 0L; 1L; ns_days ]);
        (1, of_list [ Int64.succ ns_days ]);
        (2, int64_range (-1_000_000L) 1_000_000L);
        (1, int64_range int32_lo int32_hi);
      ]
  in
  let gen = pair (of_list units) (array ~size:(int_range 0 12) (option days)) in
  prop
    "a date becomes the datetime of its midnight, failing out of the unit's \
     range"
    gen (fun (u, ds) ->
      let expected =
        Array.map
          (fun d -> Option.bind d (fun d -> mul_exact d (ticks_per_day u)))
          ds
      in
      cover "every row in range"
        ((not (fails ds expected)) && Array.exists Option.is_some ds);
      cover "a row out of range" (fails ds expected);
      cover "a date before 1970" (before_1970 ds);
      cover "a null" (Array.exists Option.is_none ds);
      agrees ~expected ds
        (cast_ticks (Any Type.date) (Any (Type.datetime u)) ds))

(* [reach u] is the days either side of 1970 within which a datetime of the unit
   [u] holds every tick of the day, up to a million. *)
let reach u =
  Int64.min 1_000_000L (Int64.pred (Int64.div Int64.max_int (ticks_per_day u)))

let datetimes_to_dates =
  let gen =
    let open Gen in
    let* u = of_list units in
    let day = ticks_per_day u in
    let edges =
      [ Int64.min_int; Int64.succ Int64.min_int; -1L; 0L; 1L ]
      @ [ Int64.pred Int64.max_int; Int64.max_int; Int64.neg day ]
      @ [ Int64.pred (Int64.neg day); Int64.pred day; day ]
    in
    (* The first and last ticks of the date's first and last days, where a unit
       reaches them. *)
    let date_edges =
      List.concat_map
        (fun d ->
          match mul_exact d day with
          | Some t when Int64.compare t (Int64.sub Int64.max_int day) < 0 ->
              [ Int64.pred t; t; Int64.add t (Int64.pred day) ]
          | _ -> [])
        [ int32_lo; int32_hi; Int64.succ int32_hi ]
    in
    let near_midnight =
      let+ d = int64_range (Int64.neg (reach u)) (reach u)
      and+ k = int64_range (-1L) 1L in
      Int64.add (Int64.mul d day) k
    in
    let tick =
      frequency
        [ (2, of_list (edges @ date_edges)); (2, near_midnight); (2, int64) ]
    in
    let+ ts = array ~size:(int_range 0 12) (option tick) in
    (u, ts)
  in
  prop "a datetime becomes the day it falls on, failing out of the date's range"
    gen (fun (u, ts) ->
      let day_of t =
        let d = floor_div t (ticks_per_day u) in
        if d < int32_lo || d > int32_hi then None else Some d
      in
      let expected = Array.map (fun t -> Option.bind t day_of) ts in
      let inside_day t = Int64.rem t (ticks_per_day u) <> 0L in
      cover "a time of day before 1970"
        (Array.exists
           (function Some t -> t < 0L && inside_day t | None -> false)
           ts);
      cover "a day out of the date's range" (fails ts expected);
      cover "a null" (Array.exists Option.is_none ts);
      agrees ~expected ts
        (cast_ticks (Any (Type.datetime u)) (Any Type.date) ts))

let dates_round_trip =
  let gen =
    let open Gen in
    let* u = of_list units in
    let+ ds =
      array ~size:(int_range 0 12)
        (option
           (frequency
              [
                (1, of_list [ Int64.neg (reach u); -1L; 0L; reach u ]);
                (3, int64_range (Int64.neg (reach u)) (reach u));
              ]))
    in
    (u, ds)
  in
  prop "a date cast to a datetime and back is the date" gen (fun (u, ds) ->
      cover "a date before 1970" (before_1970 ds);
      let w = array (option int64) in
      let datetime = Type.Any (Type.datetime u) in
      Law.round_trip w w
        (fun ds ->
          int64_options
            (require_ok ~pp:Error.pp (cast_ticks (Any Type.date) datetime ds)))
        (fun ts ->
          int64_options
            (require_ok ~pp:Error.pp (cast_ticks datetime (Any Type.date) ts)))
        ds)

let days_floor =
  let gen =
    let open Gen in
    let* u = of_list units in
    let day = ticks_per_day u in
    let edges =
      [ Int64.min_int; Int64.succ Int64.min_int; -1L; 0L; 1L ]
      @ [ Int64.pred Int64.max_int; Int64.max_int ]
    in
    let anywhere =
      let+ d = int64_range (Int64.neg (reach u)) (reach u)
      and+ k = int64_range 0L (Int64.pred day) in
      Int64.add (Int64.mul d day) k
    in
    let tick = frequency [ (1, of_list edges); (3, anywhere); (1, int64) ] in
    let+ ts = array ~size:(int_range 0 12) (option tick) in
    (u, ts)
  in
  prop
    "Temporal.floor (Days 1) is the midnight of a datetime's day, and a cast \
     to a date and back is that midnight"
    gen (fun (u, ts) ->
      let day = ticks_per_day u in
      let midnight t = mul_exact (floor_div t day) day in
      let expected = Array.map (fun t -> Option.bind t midnight) ts in
      cover "a time of day before 1970" (before_1970 ts);
      cover "a midnight below the least tick" (fails ts expected);
      let ty = Type.datetime u in
      let x = Col.instant "x" in
      let floor = Expr.Temporal.floor (Days 1) x in
      agrees ~failure:"the 1d period of " ~expected ts
        (compute floor (one "x" (ticks_column (Any ty) ts)));
      (* The cast holds where the midnight and the date both exist. *)
      let defined t =
        let d = floor_div t day in
        Option.is_some (midnight t) && d >= int32_lo && d <= int32_hi
      in
      let kept =
        Array.of_list
          (List.filter
             (function Some t -> defined t | None -> true)
             (Array.to_list ts))
      in
      let t = one "x" (ticks_column (Any ty) kept) in
      equal
        (array (option int64))
        (int64_options (result floor t))
        (int64_options (result Expr.(cast ty (cast Type.date x)) t)))

let clocks_and_durations =
  let gen =
    let open Gen in
    let* u0 = of_list units in
    let* u1 = frequency [ (1, constant u0); (1, of_list units) ] in
    let* to_clock = bool in
    let* in_day = frequency [ (1, constant true); (2, constant false) ] in
    let day = ticks_per_day u0 and second = per_s u0 in
    (* Whole seconds of the day, which every unit holds. *)
    let seconds = map (Int64.mul second) (int64_range 0L 86_399L) in
    let value =
      if in_day then seconds
      else if to_clock then
        frequency
          [
            (2, of_list [ Int64.min_int; -1L; 0L; 1L; Int64.pred day; day ]);
            (1, of_list [ Int64.max_int ]);
            (2, map (Int64.mul second) (int64_range (-86_400L) 172_800L));
            (2, int64_range (Int64.neg day) (Int64.mul 2L day));
            (1, int64);
          ]
      else
        frequency
          [
            (1, of_list [ 0L; 1L; Int64.pred day ]);
            (2, map (Int64.mul second) (int64_range 0L 86_399L));
            (2, int64_range 0L (Int64.pred day));
          ]
    in
    let+ xs = array ~size:(int_range 0 10) (option value) in
    (to_clock, u0, u1, xs)
  in
  let rescale u0 u1 x =
    let n0 = per_s u0 and n1 = per_s u1 in
    if n1 >= n0 then mul_exact x (Int64.div n1 n0)
    else
      let k = Int64.div n0 n1 in
      if Int64.rem x k = 0L then Some (Int64.div x k) else None
  in
  (* Four covers share the cases, the rarest drawn in about one case of sixteen,
     so 300 cases reach each in every run. *)
  prop ~count:300
    "clocks and durations cast both ways, keeping the span and failing outside \
     the day"
    gen (fun (to_clock, u0, u1, xs) ->
      let in_day r = r >= 0L && r < ticks_per_day u1 in
      let model x =
        match rescale u0 u1 x with
        | Some r when (not to_clock) || in_day r -> Some r
        | _ -> None
      in
      let expected = Array.map (fun x -> Option.bind x model) xs in
      let outside x = not (in_day x) in
      cover "a duration outside the day, at the clock's unit"
        (to_clock && u0 = u1
        && Array.exists (function Some x -> outside x | None -> false) xs);
      cover "a clock to a duration of a coarser unit, failing"
        ((not to_clock) && per_s u1 < per_s u0 && fails xs expected);
      cover "a duration to a clock, every row kept"
        (to_clock && (not (fails xs expected)) && Array.exists Option.is_some xs);
      cover "a clock to a duration, every row kept"
        ((not to_clock)
        && (not (fails xs expected))
        && Array.exists Option.is_some xs);
      let from, into =
        if to_clock then (Type.duration u0, Type.clock u1)
        else (Type.clock u0, Type.duration u1)
      in
      agrees ~expected xs (cast_ticks (Any from) (Any into) xs))

let temporal_casts =
  let date y m d = Time.Date.to_days (date y m d) in
  let days_of u ts =
    int64_options
      (require_ok ~pp:Error.pp
         (cast_ticks (Any (Type.datetime u)) (Any Type.date) (some ts)))
  in
  let problem e t =
    match compute e t with
    | _ -> "no problem"
    | exception Invalid_argument m -> m
  in
  let days ds = some (Array.map Int64.of_int ds) in
  group "Temporal casts"
    [
      dates_to_datetimes;
      datetimes_to_dates;
      dates_round_trip;
      days_floor;
      clocks_and_durations;
      test "a datetime's day, before 1970 and at the ends of its ticks"
        (fun () ->
          equal
            (array (option int64))
            (days [| date 1969 12 31; date 1970 1 1; date 1970 1 1 |])
            (days_of S [| -1L; 0L; 86_399L |]);
          equal
            (array (option int64))
            (days [| date 1677 9 21; date 2262 4 11 |])
            (days_of Ns [| Int64.min_int; Int64.max_int |]));
      test "dates at the ends of datetime[ns]" (fun () ->
          equal
            (array (option int64))
            [|
              Some
                (Int64.mul (Int64.of_int (date 2262 4 11)) 86_400_000_000_000L);
            |]
            (int64_options
               (require_ok ~pp:Error.pp
                  (cast_ticks (Any Type.date)
                     (Any (Type.datetime Ns))
                     (days [| date 2262 4 11 |]))));
          expect
            (Format.asprintf "%a" Error.pp
               (require_error
                  (cast_ticks (Any Type.date)
                     (Any (Type.datetime Ns))
                     (days [| date 1677 9 22; date 1677 9 21 |]))))
          @@ __POS_OF__
               {| select ["out" := cast datetime[ns] x]: row 1: cannot cast 1677-09-21 to datetime[ns]. |});
      test "a datetime with a zone holds instants, which do not cast to dates"
        (fun () ->
          let t =
            v
              [
                ("z", Column.v (Type.datetime ~zone:"UTC" S) [| Time.of_ns 0L |]);
                ("d", Column.v Type.date [| Option.get (Time.Date.of_days 0) |]);
              ]
          in
          expect
            (String.concat "\n"
               [
                 problem (Expr.cast Type.date (Col.instant "z")) t;
                 problem
                   (Expr.cast (Type.datetime ~zone:"UTC" S) (Col.date "d"))
                   t;
               ])
          @@ __POS_OF__
               {|
            select: 1 problem
              "out" := cast date z
                cast does not convert datetime[s, UTC] to date.
              input (2 columns): z datetime[s, UTC], d date
            select: 1 problem
              "out" := cast datetime[s, UTC] d
                cast does not convert date to datetime[s, UTC].
              input (2 columns): z datetime[s, UTC], d date
            |});
    ]

(* Trimming *)

(* Unicode's White_Space property, PropList.txt 18.0.0: its 25 scalar values. *)
let white_space =
  [ 0x0009; 0x000A; 0x000B; 0x000C; 0x000D; 0x0020; 0x0085; 0x00A0; 0x1680 ]
  @ List.init 11 (fun i -> 0x2000 + i)
  @ [ 0x2028; 0x2029; 0x202F; 0x205F; 0x3000 ]

(* The scalar values beside each run of White_Space, and the separators and
   zero-width characters outside it. *)
let not_white_space =
  [ 0x0008; 0x000E; 0x001C; 0x001D; 0x001E; 0x001F; 0x0021; 0x0084; 0x0086 ]
  @ [ 0x009F; 0x00A1; 0x167F; 0x1681; 0x180E; 0x1FFF; 0x200B; 0x200C ]
  @ [ 0x200D; 0x2027; 0x202A; 0x202E; 0x2030; 0x205E; 0x2060; 0x2FFF ]
  @ [ 0x3001; 0xFEFF ]

let utf_8 u =
  let b = Buffer.create 4 in
  Buffer.add_utf_8_uchar b (Uchar.of_int u);
  Buffer.contents b

let scalar s =
  Uchar.to_int (Uchar.utf_decode_uchar (String.get_utf_8_uchar s 0))

(* [trimmed s] is [s] without the White_Space scalars at its ends. *)
let trimmed s =
  let rec drop = function
    | u :: us when List.mem (scalar u) white_space -> drop us
    | us -> us
  in
  String.concat "" (List.rev (drop (List.rev (drop (scalars s)))))

let trims vs = Column.options Kind.string (result (Expr.Str.trim s) (texts vs))

(* A scalar value of four bytes of UTF-8, the length no white space has. *)
let four_bytes = utf_8 0x1F600

let trim_law =
  let of_list vs = Gen.of_list ~pp:(fun ppf -> Format.fprintf ppf "%S") vs in
  let token =
    Gen.frequency
      [
        (2, of_list (List.map utf_8 white_space));
        (2, of_list (List.map utf_8 not_white_space));
        (1, of_list [ "a"; "é"; "日" ]);
        (2, of_list [ four_bytes ]);
      ]
  in
  let text =
    Gen.map (String.concat "") (Gen.list ~size:(Gen.int_range 0 8) token)
  in
  prop "trim drops the White_Space scalars at both ends of the text"
    (Gen.array ~size:(Gen.int_range 0 6) (Gen.option text))
    (fun vs ->
      let has p = Array.exists (function Some s -> p s | None -> false) vs in
      let ws s = String.length s > 0 && List.mem (scalar s) white_space in
      let ends s =
        let us = scalars s in
        us <> [] && ws (List.hd us) && ws (List.hd (List.rev us))
      in
      cover "white space at both ends and a scalar outside it"
        (has (fun s -> ends s && trimmed s <> ""));
      cover "white space alone" (has (fun s -> s <> "" && trimmed s = ""));
      let starts s = String.starts_with ~prefix:four_bytes s in
      let ends s = String.ends_with ~suffix:four_bytes s in
      cover "a text that trims to a four-byte scalar at its start"
        (has (fun s -> starts (trimmed s) && not (starts s)));
      cover "a text that trims to a four-byte scalar at its end"
        (has (fun s -> ends (trimmed s) && not (ends s)));
      equal
        (array (option string))
        (Array.map (Option.map trimmed) vs)
        (trims vs))

let trimming =
  group "Str.trim"
    [
      trim_law;
      cases "trims each White_Space scalar at both ends, and keeps it inside"
        ~id:(Printf.sprintf "U+%04X") white_space (fun u ->
          let w = utf_8 u in
          equal
            (array (option string))
            [| Some ("a" ^ w ^ "b"); Some "" |]
            (trims [| Some (w ^ w ^ "a" ^ w ^ "b" ^ w); Some w |]));
      cases "keeps the scalars outside White_Space"
        ~id:(Printf.sprintf "U+%04X") not_white_space (fun u ->
          let n = utf_8 u in
          let kept = n ^ " a " ^ n in
          equal
            (array (option string))
            [| Some kept |]
            (trims [| Some (" " ^ kept ^ "\t") |]));
      test "nulls stay null, and the empty text is kept" (fun () ->
          equal
            (array (option string))
            [| None; Some ""; Some "a" |]
            (trims [| None; Some ""; Some " a\n" |]));
      test "a categorical trims as its text" (fun () ->
          let ty = Type.categorical [| " a"; "a "; "b" |] in
          let t =
            one "s" (Column.of_options ty [| Some " a"; Some "a "; None |])
          in
          equal
            (array (option string))
            [| Some "a"; Some "a"; None |]
            (Column.options Kind.string (result (Expr.Str.trim s) t)));
    ]

(* Joining *)

type part = Text_column of string option array | Literal of string

let pp_part ppf = function
  | Text_column vs -> G.pp_sample ppf (G.Sample (Type.string, vs))
  | Literal s -> Format.fprintf ppf "string %S" s

let concat_law =
  let text = Option.get (G.value Type.string) in
  let gen =
    let open Gen in
    let* n = int_range 0 5 in
    let operand =
      frequency
        [
          ( 3,
            map
              (fun vs -> Text_column vs)
              (array ~size:(constant n) (option text)) );
          (1, map (fun s -> Literal s) text);
        ]
    in
    let+ sep = of_list [ ""; "_"; ", "; "é"; "日本" ]
    and+ ops = list ~size:(int_range 0 4) operand in
    (n, sep, ops)
  in
  let pp ppf (n, sep, ops) =
    Format.fprintf ppf "@[<v>%d rows, separator %S@,%a@]" n sep
      (Format.pp_print_list pp_part)
      ops
  in
  prop "concat joins the texts with the separator, null where one is null"
    (Gen.with_pp pp gen) (fun (n, sep, ops) ->
      let columns =
        List.filter_map
          (function Text_column vs -> Some vs | Literal _ -> None)
          ops
      in
      let name i = Printf.sprintf "c%d" i in
      let t =
        v
          (("rows", Column.v Type.int8 (Array.make n 0))
          :: List.mapi
               (fun i vs -> (name i, Column.of_options Type.string vs))
               columns)
      in
      let k = ref (-1) in
      let es =
        List.map
          (function
            | Text_column _ ->
                incr k;
                Col.string (name !k)
            | Literal s -> Expr.string s)
          ops
      in
      let row i =
        List.map (function Text_column vs -> vs.(i) | Literal s -> Some s) ops
      in
      let expected =
        Array.init n (fun i ->
            let r = row i in
            if List.exists Option.is_none r then None
            else Some (String.concat sep (List.map Option.get r)))
      in
      cover "a row with a null operand" (Array.exists Option.is_none expected);
      cover "a row of two operands or more"
        (List.length ops >= 2 && Array.exists Option.is_some expected);
      cover "a literal and a column"
        (columns <> [] && List.length columns < List.length ops);
      cover "no operand" (ops = [] && n > 0);
      equal
        (array (option string))
        expected
        (Column.options Kind.string (result (Expr.Str.concat sep es) t)))

let joining =
  group "Str.concat"
    [
      concat_law;
      test "a categorical joins as its text" (fun () ->
          let k = Type.categorical [| "x"; "y" |] in
          let t =
            v
              [
                ("k", Column.of_options k [| Some "y"; None; Some "x" |]);
                ("s", Column.v Type.string [| "1"; "2"; "3" |]);
              ]
          in
          equal
            (array (option string))
            [| Some "y-1"; None; Some "x-3" |]
            (Column.options Kind.string
               (result Expr.(Str.concat "-" [ Col.string "k"; s ]) t)));
      test "null joins to null, and no text to the empty text" (fun () ->
          let t = texts [| Some "a"; None |] in
          equal
            (array (option string))
            [| None; None |]
            (Column.options Kind.string
               (result Expr.(Str.concat "_" [ s; null ]) t));
          equal
            (array (option string))
            [| Some ""; Some "" |]
            (Column.options Kind.string (result (Expr.Str.concat "_" []) t)));
    ]

let () =
  exit
    (run "Convert"
       [
         casts;
         cast_failures;
         held_casts;
         text;
         slices;
         time;
         fields_agree;
         adds_ticks;
         formats_round_trip;
         cuts;
         temporal_casts;
         trimming;
         joining;
       ])
