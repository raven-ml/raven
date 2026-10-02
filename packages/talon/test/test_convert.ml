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
       ])
