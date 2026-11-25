(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let str pp v = Format.asprintf "%a" pp v
let rejects f = raises_match (fun e -> Exn.invalid_arg e) f

(* Witnesses *)

let instant_w =
  Testable.make ~pp:Time.pp ~equal:Time.equal
  |> Testable.with_compare Time.compare

let span_w =
  Testable.make ~pp:Time.Span.pp ~equal:Time.Span.equal
  |> Testable.with_compare Time.Span.compare

let date_w =
  Testable.make ~pp:Time.Date.pp ~equal:Time.Date.equal
  |> Testable.with_compare Time.Date.compare

let civil_w = triple int int int
let pp_any ppf (Type.Any t) = Type.pp ppf t

let any_w =
  Testable.make ~pp:pp_any ~equal:(fun (Type.Any a) (Type.Any b) ->
      Type.equal a b)

let binary_w = Testable.make ~pp:Binary.pp ~equal:( = )

(* [by t w] is [w] ordered by talon's order on the values of [t]. *)
let by t w = Testable.with_compare (Type.compare_value t) w

(* [ordered t pp] is the witness of [t]'s values under talon's order, two values
   being equal when that order says so. *)
let ordered t pp =
  Testable.make ~pp ~equal:(fun a b -> Type.compare_value t a b = 0)
  |> Testable.with_compare (Type.compare_value t)

(* [held t vs] is the values of [vs] that [t] holds. *)
let held t vs = List.filter (Type.holds t) vs

(* [mul_exact n u] is [n * u], or [None] if it leaves [int64]. *)
let mul_exact n u =
  let r = Int64.mul n u in
  if Int64.equal (Int64.div r u) n then Some r else None

(* [floor_div x k] is [x / k] rounded toward negative infinity, for [k > 0]. *)
let floor_div x k =
  let q = Int64.div x k in
  if Int64.compare (Int64.rem x k) 0L < 0 then Int64.pred q else q

(* The spec's float identity: [-0.] is [0.] and every NaN is every other. *)
let same_float a b = (Float.is_nan a && Float.is_nan b) || a = b

(* Binary *)

let hex s =
  let byte c = Printf.sprintf "%02x" (Char.code c) in
  "0x" ^ String.concat "" (List.map byte (List.of_seq (String.to_seq s)))

let binary =
  group "Binary"
    [
      test "of_string keeps the bytes it is made of" (fun () ->
          let s = "\x00\xff\n" in
          equal string s (Binary.of_string s :> string));
      cases "pp formats" ~id:snd
        [ ("Hi\n", "0x48690a"); ("", "0x") ]
        (fun (s, expected) ->
          equal string expected (str Binary.pp (Binary.of_string s)));
      prop "pp writes two lowercase hexadecimal digits per byte"
        ~examples:[ "\x00\xff" ] Gen.string (fun s ->
          equal string (hex s) (str Binary.pp (Binary.of_string s)));
    ]

(* Instants *)

let ns_range =
  Gen.frequency
    [
      (6, Gen.int64);
      (2, Gen.map Int64.of_int Gen.small_int);
      ( 1,
        Gen.of_list
          ~pp:(fun ppf -> Format.fprintf ppf "%LdL")
          [
            Int64.min_int;
            Int64.succ Int64.min_int;
            Int64.pred Int64.max_int;
            Int64.max_int;
            -1L;
            0L;
          ] );
    ]

let units =
  [
    ("us", 1_000L, Time.of_us, Time.to_us);
    ("ms", 1_000_000L, Time.of_ms, Time.to_ms);
    ("s", 1_000_000_000L, Time.of_s, Time.to_s);
  ]

(* The last whole unit at each end of the range, and the unit past it. *)
let unit_edges =
  List.concat_map
    (fun (_, u, _, _) ->
      let hi = Int64.div Int64.max_int u and lo = Int64.div Int64.min_int u in
      [ hi; lo; Int64.succ hi; Int64.pred lo ])
    units

(* Seconds in 400 Gregorian years, 146097 days. *)
let cycle = 12_622_780_800L

(* [gmtime_string s] is the UTC time [s] seconds after the epoch, as gmtime
   gives it. Windows' gmtime refuses a time before the epoch, and the calendar
   repeats every 400 years, so such a time is read 400 years later. *)
let gmtime_string s =
  let s, years = if s < 0L then (Int64.add s cycle, 400) else (s, 0) in
  let tm = Unix.gmtime (Int64.to_float s) in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02d"
    (tm.tm_year + 1900 - years)
    (tm.tm_mon + 1) tm.tm_mday tm.tm_hour tm.tm_min tm.tm_sec

(* [fraction ns] is the [ns] nanoseconds of a second as a point and nine digits,
   less each trailing group of three zeros: the fewest of 3, 6 or 9 digits that
   are exact, and nothing for a whole second. *)
let fraction ns =
  let rec trim s =
    if String.ends_with ~suffix:"000" s then
      trim (String.sub s 0 (String.length s - 3))
    else s
  in
  match trim (Printf.sprintf "%09Ld" ns) with "" -> "" | s -> "." ^ s

let instants =
  group "Time"
    [
      prop "to_ns inverts of_ns"
        ~examples:[ Int64.min_int; Int64.max_int ]
        ns_range
        (Law.round_trip int64 instant_w Time.of_ns Time.to_ns);
      prop "to_us, to_ms and to_s round toward negative infinity"
        ~examples:[ -1L; 5_999L ] ns_range (fun n ->
          List.iter
            (fun (name, u, _, to_u) ->
              equal ~msg:name int64 (floor_div n u) (to_u (Time.of_ns n)))
            units);
      prop "of_us, of_ms and of_s are None exactly past the range"
        ~examples:unit_edges ns_range (fun n ->
          List.iter
            (fun (name, u, of_u, _) ->
              equal ~msg:name (option int64) (mul_exact n u)
                (Option.map Time.to_ns (of_u n)))
            units);
      prop "of_us, of_ms and of_s invert to_us, to_ms and to_s on whole units"
        ns_range (fun n ->
          List.iter
            (fun (name, u, of_u, to_u) ->
              let whole = Time.of_ns (Int64.mul (Int64.div n u) u) in
              equal ~msg:name (option instant_w) (Some whole)
                (of_u (to_u whole)))
            units);
      prop "compare orders chronologically"
        (Gen.triple ns_range ns_range ns_range) (fun (a, b, c) ->
          Law.order instant_w (Time.of_ns a, Time.of_ns b, Time.of_ns c);
          equal (list int64)
            (List.sort Int64.compare [ a; b; c ])
            (List.map Time.to_ns
               (List.sort Time.compare (List.map Time.of_ns [ a; b; c ]))));
      cases "pp formats" ~id:snd
        [
          (1710495000_000_000_000L, "2024-03-15T09:30:00");
          (-1L, "1969-12-31T23:59:59.999999999");
        ]
        (fun (ns, expected) ->
          equal string expected (str Time.pp (Time.of_ns ns)));
      prop "pp writes gmtime's time and the fewest exact digits of a fraction"
        ~examples:
          [ 500_000_000L; 1_000L; 1_001_000L; Int64.min_int; Int64.max_int ]
        ns_range (fun ns ->
          let s = floor_div ns 1_000_000_000L in
          let part = Int64.sub ns (Int64.mul s 1_000_000_000L) in
          equal string
            (gmtime_string s ^ fraction part)
            (str Time.pp (Time.of_ns ns)));
    ]

(* Spans *)

let constructors =
  [
    ("ns", Time.Span.ns, 1L);
    ("us", Time.Span.us, 1_000L);
    ("ms", Time.Span.ms, 1_000_000L);
    ("s", Time.Span.s, 1_000_000_000L);
    ("minutes", Time.Span.minutes, 60_000_000_000L);
    ("hours", Time.Span.hours, 3_600_000_000_000L);
    ("days", Time.Span.days, 86_400_000_000_000L);
  ]

(* A constructor and a count: at the ends of the span's range in its unit and
   one past them, at the ends of [int], and anywhere. *)
let constructor_case =
  let open Gen in
  let* name, make, u = of_list constructors in
  let fits v =
    Int64.compare v (Int64.of_int min_int) >= 0
    && Int64.compare v (Int64.of_int max_int) <= 0
  in
  let hi = Int64.div Int64.max_int u and lo = Int64.div Int64.min_int u in
  let edges =
    List.map Int64.to_int
      (List.filter fits [ hi; lo; Int64.succ hi; Int64.pred lo ])
    @ [ max_int; min_int ]
  in
  let+ n = frequency [ (2, of_list edges); (1, int); (1, small_int) ] in
  (name, make, u, n)

let span_units =
  [
    ("us", 1_000L, Time.Span.of_us, Time.Span.to_us);
    ("ms", 1_000_000L, Time.Span.of_ms, Time.Span.to_ms);
    ("s", 1_000_000_000L, Time.Span.of_s, Time.Span.to_s);
  ]

let spans =
  group "Time.Span"
    [
      prop "a constructor is n of its unit, and raises past the range"
        (Gen.with_pp
           (fun ppf (name, _, _, n) -> Format.fprintf ppf "%s %d" name n)
           constructor_case)
        (fun (_, make, u, n) ->
          let expected = mul_exact (Int64.of_int n) u in
          cover "in the range" (Option.is_some expected);
          cover "past the range" (Option.is_none expected);
          match expected with
          | Some ns -> equal int64 ns (Time.Span.to_ns (make n))
          | None -> rejects (fun () -> make n));
      prop "to_us, to_ms and to_s round toward negative infinity"
        ~examples:[ -1L ] ns_range (fun n ->
          List.iter
            (fun (name, u, _, to_u) ->
              equal ~msg:name int64 (floor_div n u) (to_u (Time.Span.of_ns n)))
            span_units);
      prop "of_us, of_ms and of_s are None exactly past the range"
        ~examples:unit_edges ns_range (fun n ->
          List.iter
            (fun (name, u, of_u, _) ->
              equal ~msg:name (option int64) (mul_exact n u)
                (Option.map Time.Span.to_ns (of_u n)))
            span_units);
      prop "of_us, of_ms and of_s invert to_us, to_ms and to_s on whole units"
        ns_range (fun n ->
          List.iter
            (fun (name, u, of_u, to_u) ->
              let whole = Time.Span.of_ns (Int64.mul (Int64.div n u) u) in
              equal ~msg:name (option span_w) (Some whole) (of_u (to_u whole)))
            span_units);
      prop "compare orders from the most negative to the most positive"
        (Gen.triple ns_range ns_range ns_range) (fun (a, b, c) ->
          let d = Time.Span.of_ns in
          Law.order span_w (d a, d b, d c);
          equal (list int64)
            (List.sort Int64.compare [ a; b; c ])
            (List.map Time.Span.to_ns
               (List.sort Time.Span.compare (List.map d [ a; b; c ]))));
      cases "pp formats" ~id:snd
        [
          (Time.Span.days 7, "168h");
          (Time.Span.minutes 90, "1h30m");
          (Time.Span.ms 1500, "1s500ms");
          (Time.Span.us (-250), "-250us");
          (Time.Span.ns 0, "0s");
          (Time.Span.ns 1, "1ns");
          (Time.Span.of_ns 3_723_004_005_006L, "1h2m3s4ms5us6ns");
          (Time.Span.of_ns Int64.min_int, "-2562047h47m16s854ms775us808ns");
          (Time.Span.of_ns Int64.max_int, "2562047h47m16s854ms775us807ns");
        ]
        (fun (d, expected) -> equal string expected (str Time.Span.pp d));
    ]

(* Dates *)

let min_days = Int32.to_int Int32.min_int
let max_days = Int32.to_int Int32.max_int

let days_gen =
  Gen.frequency
    [
      (4, Gen.int_range min_days max_days);
      (4, Gen.int_range (-1_000_000) 1_000_000);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ min_days; min_days + 1; max_days - 1; max_days; -1; 0; 59; 60 ] );
    ]

let is_leap y = (y mod 4 = 0 && y mod 100 <> 0) || y mod 400 = 0

let month_length y m =
  if m = 2 then if is_leap y then 29 else 28
  else if List.mem m [ 4; 6; 9; 11 ] then 30
  else 31

let next_civil (y, m, d) =
  if d < month_length y m then (y, m, d + 1)
  else if m < 12 then (y, m + 1, 1)
  else (y + 1, 1, 1)

let date n = Option.get (Time.Date.of_days n)

(* Civil dates of the years within the range, century years among them, and
   months and days on both sides of their bounds. *)
let civil_gen =
  let open Gen in
  let year =
    frequency
      [
        (2, int_range (-5_877_640) 5_881_579);
        (2, int_range (-500) 2500);
        (1, of_list [ 1900; 2000; 2100; 1600; 0; -100; -400 ]);
      ]
  in
  let month = frequency [ (2, int_range (-1) 14); (1, constant 2) ] in
  let day =
    frequency [ (2, int_range (-1) 32); (1, of_list [ 0; 28; 29; 30; 31; 32 ]) ]
  in
  with_pp
    (fun ppf (y, m, d) -> Format.fprintf ppf "(%d, %d, %d)" y m d)
    (triple year month day)

let dates =
  group "Time.Date"
    [
      cases "of_civil places"
        ~id:(fun ((y, m, d), n) -> Printf.sprintf "%d-%d-%d at %d" y m d n)
        [
          ((1970, 1, 1), 0);
          ((1969, 12, 31), -1);
          ((2000, 1, 1), 10957);
          ((2024, 2, 29), 19782);
          ((2000, 2, 29), 11016);
          ((2000, 3, 1), 11017);
          ((0, 3, 1), -719468);
        ]
        (fun (civil, days) ->
          equal (option int) (Some days)
            (Option.map Time.Date.to_days (Time.Date.of_civil civil)));
      prop "of_civil takes exactly the days of the proleptic Gregorian calendar"
        ~examples:
          [
            (2024, 2, 29);
            (2023, 2, 29);
            (1900, 2, 29);
            (2000, 2, 29);
            (2024, 0, 1);
            (2024, 13, 1);
            (2024, 4, 31);
            (2024, 1, 0);
            (2024, 1, 32);
          ]
        civil_gen
        (fun ((y, m, d) as c) ->
          let valid = m >= 1 && m <= 12 && d >= 1 && d <= month_length y m in
          cover "a date" valid;
          cover "not a date" (not valid);
          match (valid, Time.Date.of_civil c) with
          | true, Some t -> equal civil_w c (Time.Date.to_civil t)
          | false, None -> ()
          | true, None -> fail "of_civil refuses a date"
          | false, Some t -> failf "of_civil takes it as %a" Time.Date.pp t);
      cases "of_civil is None past the range"
        ~id:(fun (y, m, d) -> Printf.sprintf "%d-%d-%d" y m d)
        [
          (5_881_580, 7, 12);
          (-5_877_641, 6, 22);
          (10_000_000, 1, 1);
          (-10_000_000, 1, 1);
          (max_int, 1, 1);
          (min_int, 1, 1);
          (* A year whose day count wraps into the range. *)
          (3_535_382_889_881_440_000, 3, 1);
        ]
        (fun civil -> is_none ~pp:Time.Date.pp (Time.Date.of_civil civil));
      test "dates span -5877641-06-23 to 5881580-07-11" (fun () ->
          equal (pair civil_w civil_w)
            ((-5_877_641, 6, 23), (5_881_580, 7, 11))
            ( Time.Date.to_civil (date min_days),
              Time.Date.to_civil (date max_days) ));
      test "of_days is None past the int32 range" (fun () ->
          is_none ~pp:Time.Date.pp (Time.Date.of_days (max_days + 1));
          is_none ~pp:Time.Date.pp (Time.Date.of_days (min_days - 1)));
      prop "of_civil inverts to_civil" ~examples:[ min_days; max_days; 11016 ]
        days_gen (fun n ->
          Law.round_trip date_w civil_w Time.Date.to_civil
            (fun c -> require_some (Time.Date.of_civil c))
            (date n));
      prop "of_days inverts to_days" days_gen (fun n ->
          equal int n (Time.Date.to_days (date n)));
      prop "the day after is the next civil date"
        ~examples:[ min_days; max_days - 1; -1; 59; 60 ]
        (Gen.int_range min_days (max_days - 1))
        (fun n ->
          equal civil_w
            (next_civil (Time.Date.to_civil (date n)))
            (Time.Date.to_civil (date (n + 1))));
      prop "compare orders chronologically"
        (Gen.triple days_gen days_gen days_gen) (fun (a, b, c) ->
          Law.order date_w (date a, date b, date c);
          equal (list int)
            (List.sort Int.compare [ a; b; c ])
            (List.map Time.Date.to_days
               (List.sort Time.Date.compare (List.map date [ a; b; c ]))));
      cases "pp formats" ~id:snd
        [ ((-44, 3, 15), "-0044-03-15"); ((12345, 1, 1), "+12345-01-01") ]
        (fun (civil, expected) ->
          equal string expected
            (str Time.Date.pp (Option.get (Time.Date.of_civil civil))));
      prop "pp writes YYYY-MM-DD, a year outside 0 to 9999 signed"
        ~examples:[ 0; -719528; 2932896; 2932897; -719529 ] days_gen (fun n ->
          let y, m, d = Time.Date.to_civil (date n) in
          let year =
            if y >= 0 && y <= 9999 then Printf.sprintf "%04d" y
            else Printf.sprintf "%c%04d" (if y < 0 then '-' else '+') (abs y)
          in
          equal string
            (Printf.sprintf "%s-%02d-%02d" year m d)
            (str Time.Date.pp (date n)));
    ]

let step_pp =
  cases "pp_step formats" ~id:snd
    [
      (Time.Months 1, "1mo");
      (Time.Weeks 2, "2w");
      (Time.Days (-1), "-1d");
      (Time.Exact (Time.Span.minutes 15), "15m");
    ]
    (fun (s, expected) -> equal string expected (str Time.pp_step s))

(* Kinds *)

let ext_type = Type.ext ~name:"units.mass" ~metadata:"kg" Type.float64
let ext_kind = Type.kind ext_type

type kind = K : 'a Kind.t -> kind

let all_kinds =
  [
    K Kind.bool;
    K Kind.int;
    K Kind.float;
    K Kind.string;
    K Kind.binary;
    K Kind.date;
    K Kind.instant;
    K Kind.wall;
    K Kind.span;
    K Record.kind;
    K (Kind.list Kind.string);
    K (Kind.list Kind.float);
    K (Kind.list (Kind.list Kind.int));
    K (Kind.tensor Nx.float32);
    K (Kind.tensor Nx.float64);
    K ext_kind;
    K (Kind.list ext_kind);
  ]

let kind_name (K k) = str Kind.pp k

(* [ticks ty xs] is the column of the temporal type [ty] that stores the int64
   ticks [xs], whatever their range. *)
let ticks ty xs =
  let values = Nx.P (Nx.create Nx.int64 [| Array.length xs |] xs) in
  Result.get_ok (Column.of_layout ty (Fixed { validity = None; values }))

let kinds =
  group "Kind"
    [
      test "provably_equal proves a kind equal to itself, unless it holds ext"
        (fun () ->
          let proven (K k0 as a) =
            List.filter_map
              (fun (K k1 as b) ->
                match Kind.provably_equal k0 k1 with
                | Some _ -> Some (kind_name a, kind_name b)
                | None -> None)
              all_kinds
          in
          let free =
            List.filter
              (fun n -> n <> "ext" && n <> "list[ext]")
              (List.map kind_name all_kinds)
          in
          equal
            (list (pair string string))
            (List.map (fun n -> (n, n)) free)
            (List.concat_map proven all_kinds));
      cases "pp formats" ~id:snd
        [
          (str Kind.pp Kind.bool, "bool");
          (str Kind.pp Kind.int, "int");
          (str Kind.pp Kind.float, "float");
          (str Kind.pp Kind.string, "string");
          (str Kind.pp Kind.binary, "binary");
          (str Kind.pp Kind.date, "date");
          (str Kind.pp Kind.instant, "instant");
          (str Kind.pp Kind.wall, "wall");
          (str Kind.pp Kind.span, "span");
          (str Kind.pp Record.kind, "record");
          (str Kind.pp (Kind.list Kind.float), "list[float]");
          (str Kind.pp (Kind.tensor Nx.float32), "tensor[float32]");
          (str Kind.pp ext_kind, "ext");
        ]
        (fun (actual, expected) -> equal string expected actual);
      test "instant, wall and span fail to read a value outside their range"
        (fun () ->
          let past = [| Int64.max_int |] in
          rejects (fun () ->
              Column.values Kind.instant (ticks (Any (Type.instant S)) past));
          rejects (fun () ->
              Column.values Kind.wall (ticks (Any (Type.wall S)) past));
          rejects (fun () ->
              Column.values Kind.span (ticks (Any (Type.duration S)) past)));
    ]

(* Records *)

let ab = Record.(empty |> add Kind.int "a" (Some 1) |> add Kind.string "b" None)

(* The record type of one extension field, and its cells [{m = 1.}] and [{m =
   2.}], read from a column: no public constructor puts a value in an extension
   field. *)
let mass_type = Type.record [ ("m", Any ext_type) ]

let masses () =
  let stored = Column.layout (Column.v Type.float64 [| 1.; 2. |]) in
  let m = Result.get_ok (Column.of_layout (Any ext_type) stored) in
  let fields = [ ("m", m) ] in
  let l = Column.Children { validity = None; length = 2; fields } in
  let c = Result.get_ok (Column.of_layout (Any mass_type) l) in
  match Column.values Record.kind c with
  | [| r1; r2 |] -> (r1, r2)
  | _ -> fail "a column of two rows reads as two records"

let pp_names ppf r =
  Format.fprintf ppf "{%s}" (String.concat ", " (Record.names r))

let records =
  group "Record"
    [
      test "empty has no fields" (fun () ->
          equal (list string) [] (Record.names Record.empty));
      test "add appends a last field" (fun () ->
          equal (list string) [ "a"; "b" ] (Record.names ab));
      prop "names lists the added fields in order"
        (Gen.list ~size:(Gen.int_range 0 6)
           (Gen.string_of ~size:(Gen.int_range 0 3) (Gen.char_range 'a' 'c')))
        (fun names ->
          let names =
            List.rev
              (List.fold_left
                 (fun acc n -> if List.mem n acc then acc else n :: acc)
                 [] names)
          in
          let r =
            List.fold_left
              (fun r n -> Record.add Kind.bool n (Some true) r)
              Record.empty names
          in
          equal (list string) names (Record.names r));
      test "field reads a value and a null" (fun () ->
          equal (option int) (Some 1) (Record.field Kind.int "a" ab);
          equal (option string) None (Record.field Kind.string "b" ab));
      test "field reads a list through the list kind" (fun () ->
          let r =
            Record.(add (Kind.list Kind.int) "l" (Some [| 1; 2 |]) empty)
          in
          equal
            (option (array int))
            (Some [| 1; 2 |])
            (Record.field (Kind.list Kind.int) "l" r);
          rejects (fun () -> Record.field (Kind.list Kind.float) "l" r));
      test "field reads a record through Record.kind" (fun () ->
          let r = Record.(add Record.kind "r" (Some ab) empty) in
          equal
            (option (list string))
            (Some [ "a"; "b" ])
            (Option.map Record.names (Record.field Record.kind "r" r)));
      test "field refuses a missing name" (fun () ->
          rejects (fun () -> Record.field Kind.int "c" ab));
      test "field refuses another kind, even on a null" (fun () ->
          rejects (fun () -> Record.field Kind.float "a" ab);
          rejects (fun () -> Record.field Kind.int "b" ab));
      test "field refuses a field of an extension type" (fun () ->
          let r1, _ = masses () in
          rejects (fun () -> Record.field Kind.float "m" r1));
      test "add refuses a duplicate name" (fun () ->
          rejects (fun () -> Record.add Kind.int "a" None ab));
      test "add refuses a name that is not UTF-8" (fun () ->
          rejects (fun () -> Record.add Kind.int "\xff" None Record.empty));
      test "add refuses the extension kind and lists of it" (fun () ->
          rejects (fun () -> Record.add ext_kind "e" None Record.empty);
          rejects (fun () ->
              Record.add (Kind.list ext_kind) "e" None Record.empty));
    ]

(* Type constructors *)

let type_constructors =
  group "Type constructors"
    [
      test "categorical refuses a duplicate or non-UTF-8 string" (fun () ->
          rejects (fun () -> Type.categorical [| "a"; "b"; "a" |]);
          rejects (fun () -> Type.categorical [| "\xc3" |]));
      test "categorical copies its dictionary" (fun () ->
          let d = [| "a"; "b" |] in
          let t = Type.categorical d in
          d.(0) <- "z";
          equal any_w (Any (Type.categorical [| "a"; "b" |])) (Any t));
      test "categorical accepts an empty dictionary" (fun () ->
          equal string "categorical[]" (str Type.pp (Type.categorical [||])));
      test "record refuses a duplicate or non-UTF-8 name" (fun () ->
          rejects (fun () ->
              Type.record [ ("a", Any Type.bool); ("a", Any Type.int8) ]);
          rejects (fun () -> Type.record [ ("\xff", Any Type.bool) ]));
      test "record accepts no fields and empty names" (fun () ->
          equal string "record[]" (str Type.pp (Type.record []));
          equal string {|record["" bool]|}
            (str Type.pp (Type.record [ ("", Any Type.bool) ])));
      test "tensor refuses an empty shape or a negative dimension" (fun () ->
          rejects (fun () -> Type.tensor Nx.float32 [||]);
          rejects (fun () -> Type.tensor Nx.float32 [| 2; -1 |]));
      test "tensor copies its shape" (fun () ->
          let shape = [| 3; 4 |] in
          let t = Type.tensor Nx.float32 shape in
          shape.(0) <- 9;
          equal string "tensor[float32, 3×4]" (str Type.pp t));
      test "ext refuses an empty or non-UTF-8 name and extension storage"
        (fun () ->
          rejects (fun () -> Type.ext ~name:"" Type.float64);
          rejects (fun () -> Type.ext ~name:"\xff" Type.float64);
          rejects (fun () -> Type.ext ~name:"x" ext_type));
      test "ext's metadata defaults to the empty string" (fun () ->
          equal any_w
            (Any (Type.ext ~name:"x" ~metadata:"" Type.int8))
            (Any (Type.ext ~name:"x" Type.int8)));
    ]

let kind_of (Type.Any t) = str Kind.pp (Type.kind t)

let type_kind =
  cases "Type.kind"
    ~id:(fun (t, k) -> Printf.sprintf "kind of %s is %s" (str pp_any t) k)
    [
      (Any Type.bool, "bool");
      (Any Type.int8, "int");
      (Any Type.int32, "int");
      (Any Type.uint64, "int");
      (Any Type.float16, "float");
      (Any Type.float32, "float");
      (Any Type.float64, "float");
      (Any Type.string, "string");
      (Any (Type.categorical [| "a" |]), "string");
      (Any Type.binary, "binary");
      (Any Type.date, "date");
      (Any (Type.clock Type.Ms), "span");
      (Any (Type.duration Type.Ns), "span");
      (Any (Type.instant Type.Us), "instant");
      (Any (Type.wall Type.S), "wall");
      (Any (Type.list (Type.list Type.int16)), "list[list[int]]");
      (Any (Type.record [ ("a", Any Type.bool) ]), "record");
      (Any (Type.tensor Nx.int32 [| 2 |]), "tensor[int32]");
      (Any ext_type, "ext");
    ]
    (fun (t, expected) -> equal string expected (kind_of t))

(* Values *)

let float32_max = Int32.float_of_bits 0x7f7f_ffffl

(* The midpoint between float32's largest value and 2^128. *)
let float32_mid = ldexp 1. 128 -. ldexp 1. 103

let int_bounds : (string * int Type.t * int * int) list =
  [
    ("int8", Type.int8, -128, 127);
    ("int16", Type.int16, -32768, 32767);
    ("int32", Type.int32, -2147483648, 2147483647);
    ("int64", Type.int64, min_int, max_int);
    ("uint8", Type.uint8, 0, 255);
    ("uint16", Type.uint16, 0, 65535);
    ("uint32", Type.uint32, 0, 4294967295);
    ("uint64", Type.uint64, 0, max_int);
  ]

(* Each integer type's bounds and their neighbours, and [int]'s ends. *)
let int_candidates =
  let around (_, _, lo, hi) =
    [ lo; hi ]
    @ (if lo > min_int then [ lo - 1 ] else [])
    @ if hi < max_int then [ hi + 1 ] else []
  in
  List.sort_uniq Int.compare (List.concat_map around int_bounds)

let float_candidates =
  [
    65504.;
    65519.99;
    65520.;
    -65520.;
    0.1;
    1e-30;
    5e-324;
    -0.;
    Float.nan;
    Float.infinity;
    Float.neg_infinity;
    float32_max;
    Float.pred float32_mid;
    float32_mid;
    -.Float.max_float;
    Float.max_float;
  ]

(* A float rounds to a finite binary16 iff its magnitude is below 65520, the
   midpoint between 65504 and 2^16, which rounds to infinity, ties to even. *)
let finite16 x = Float.abs x < 65520.

(* OCaml rounds a double to binary32 once, to nearest, ties to even. *)
let finite32 x = Float.is_finite (Int32.float_of_bits (Int32.bits_of_float x))

(* Stamps and spans of each unit: multiples of the unit near zero and near a
   day, a day's bounds, and anywhere. *)
let tick_case =
  let open Gen in
  let* u, per =
    of_list
      Type.[ (S, 1_000_000_000L); (Ms, 1_000_000L); (Us, 1_000L); (Ns, 1L) ]
  in
  let day = 86_400_000_000_000L in
  let multiple base lo hi =
    map
      (fun k -> Int64.add base (Int64.mul (Int64.of_int k) per))
      (int_range lo hi)
  in
  let tick =
    frequency
      [
        (2, multiple 0L (-3) 3);
        (2, multiple day (-2) 1);
        (1, of_list [ 0L; -1L; 1L; Int64.pred day; day ]);
        (2, ns_range);
      ]
  in
  let+ ts = list ~size:(int_range 0 8) tick in
  (u, per, ts)

let pp_tick_case ppf (_, per, ts) =
  Format.fprintf ppf "unit of %Ldns, ticks [%s]" per
    (String.concat "; " (List.map Int64.to_string ts))

(* [labels t rs] is the labels of the records of [rs] that [t] holds. *)
let labels t rs = List.map fst (List.filter (fun (_, r) -> Type.holds t r) rs)

let holds =
  let rt = Type.record [ ("a", Any Type.int8); ("b", Any Type.string) ] in
  let r a b = Record.(empty |> add Kind.int "a" a |> add Kind.string "b" b) in
  let r_float a =
    Record.(empty |> add Kind.float "a" a |> add Kind.string "b" None)
  in
  group "Type.holds"
    [
      cases "an integer type holds exactly its range"
        ~id:(fun (n, _, _, _) -> n)
        int_bounds
        (fun (_, t, lo, hi) ->
          equal (list int)
            (List.filter (fun v -> lo <= v && v <= hi) int_candidates)
            (held t int_candidates));
      test
        "a float type holds NaN, the infinities and what rounds to a finite \
         value" (fun () ->
          let check name t finite =
            let expected =
              List.filter
                (fun x -> (not (Float.is_finite x)) || finite x)
                float_candidates
            in
            equal ~msg:name (list float_exact) expected
              (held t float_candidates)
          in
          check "float16" Type.float16 finite16;
          check "float32" Type.float32 finite32;
          check "float64" Type.float64 Float.is_finite);
      test "text holds valid UTF-8, a categorical its dictionary's strings"
        (fun () ->
          let texts =
            [ "é"; "\xe9"; ""; "\xc0\x80"; "\xed\xa0\x80"; "AA"; "B6"; "C" ]
          in
          equal (list string)
            [ "é"; ""; "AA"; "B6"; "C" ]
            (held Type.string texts);
          equal (list string) [ "AA"; "B6" ]
            (held (Type.categorical [| "AA"; "B6" |]) texts));
      test "booleans, byte strings and dates are always held" (fun () ->
          equal (list bool) [ false; true ] (held Type.bool [ false; true ]);
          let bytes = List.map Binary.of_string [ ""; "\xff"; "\xc0\x80" ] in
          equal (list binary_w) bytes (held Type.binary bytes);
          let ds = [ date min_days; date max_days ] in
          equal (list date_w) ds (held Type.date ds));
      prop "stamps and durations hold whole units, clocks whole units in a day"
        (Gen.with_pp pp_tick_case tick_case) (fun (u, per, ts) ->
          let whole t = Int64.equal (Int64.rem t per) 0L in
          let in_day t = t >= 0L && t < 86_400_000_000_000L in
          cover "a whole unit" (List.exists whole ts);
          cover "a tick that is not a whole unit"
            (List.exists (fun t -> not (whole t)) ts);
          cover "a whole unit outside the day"
            (List.exists (fun t -> whole t && not (in_day t)) ts);
          let stamps t =
            List.map Time.to_ns (held t (List.map Time.of_ns ts))
          in
          let spans t =
            List.map Time.Span.to_ns (held t (List.map Time.Span.of_ns ts))
          in
          let wholes = List.filter whole ts in
          equal ~msg:"instant" (list int64) wholes (stamps (Type.instant u));
          equal ~msg:"wall" (list int64) wholes (stamps (Type.wall u));
          equal ~msg:"duration" (list int64) wholes (spans (Type.duration u));
          equal ~msg:"clock" (list int64)
            (List.filter in_day wholes)
            (spans (Type.clock u)));
      test "a list is held when every element is" (fun () ->
          equal
            (list (array int))
            [ [| 0; 255 |]; [||] ]
            (held (Type.list Type.uint8) [ [| 0; 255 |]; [||]; [| 0; 256 |] ]));
      test "a record holds its type's names in order, each non-null field held"
        (fun () ->
          let candidates =
            [
              ({|a=3 b="x"|}, r (Some 3) (Some "x"));
              ("a=null b=null", r None None);
              ("a=300 b=null", r (Some 300) None);
              ({|a=null b="\xff"|}, r None (Some "\xff"));
              ( "b=null a=null",
                Record.(
                  empty |> add Kind.string "b" None |> add Kind.int "a" None) );
              ("a=null", Record.(add Kind.int "a" None empty));
              ("a=1.0 b=null", r_float (Some 1.));
              ("a=null:float b=null", r_float None);
            ]
          in
          equal (list string)
            [ {|a=3 b="x"|}; "a=null b=null"; "a=null:float b=null" ]
            (labels rt candidates));
      test "an extension field is held when null or by its storage value"
        (fun () ->
          let r1, r2 = masses () in
          let candidates =
            [
              ("m=null", Record.(add Kind.float "m" None empty));
              ("m=1.0 plain", Record.(add Kind.float "m" (Some 1.) empty));
              ("m=1.0 stored", r1);
              ("m=2.0 stored", r2);
            ]
          in
          equal (list string)
            [ "m=null"; "m=1.0 stored"; "m=2.0 stored" ]
            (labels mass_type candidates));
      test "a tensor is held with the type's shape" (fun () ->
          let v n = Nx.create Nx.float32 [| n |] (Array.make n 1.) in
          equal
            (list (array int))
            [ [| 2 |] ]
            (List.map Nx.shape
               (held (Type.tensor Nx.float32 [| 2 |]) [ v 2; v 3 ])));
    ]

(* Order *)

let float_w t =
  Testable.make ~pp:(fun ppf -> Format.fprintf ppf "%h") ~equal:same_float
  |> Testable.with_compare (Type.compare_value t)

let special_float =
  Gen.frequency
    [
      (4, Gen.any_float);
      (2, Gen.map float_of_int Gen.small_int);
      ( 2,
        Gen.of_list
          ~pp:(fun ppf -> Format.fprintf ppf "%h")
          [
            Float.nan;
            -.Float.nan;
            0.;
            -0.;
            Float.infinity;
            Float.neg_infinity;
            -.Float.max_float;
            1.;
            -1.;
          ] );
    ]

(* The reference order of floats: NaN last, [-0.] as [0.]. *)
let float_key x = if Float.is_nan x then (1, 0.) else (0, x +. 0.)
let float_order a b = compare (float_key a) (float_key b)

let float_type =
  Gen.of_list ~pp:Type.pp [ Type.float16; Type.float32; Type.float64 ]

let int_array = Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 2)

let optional g =
  Gen.frequency [ (1, Gen.constant None); (3, Gen.map Option.some g) ]

let record_type = Type.record [ ("a", Any Type.int64); ("b", Any Type.string) ]

let ab_record a b =
  Record.(empty |> add Kind.int "a" a |> add Kind.string "b" b)

(* [pp_ab] formats a record cell of [record_type]. *)
let pp_ab ppf r =
  let field pp ppf = function
    | None -> Format.pp_print_string ppf "null"
    | Some v -> pp ppf v
  in
  Format.fprintf ppf "{a=%a; b=%a}"
    (field Format.pp_print_int)
    (Record.field Kind.int "a" r)
    (field (fun ppf -> Format.fprintf ppf "%S"))
    (Record.field Kind.string "b" r)

let record_gen =
  Gen.map
    (fun (a, b) -> ab_record a b)
    (Gen.pair
       (optional (Gen.int_range 0 2))
       (optional (Gen.of_list [ "a"; "b" ])))
  |> Gen.with_pp pp_ab

let record_w =
  let fields r =
    (Record.field Kind.int "a" r, Record.field Kind.string "b" r)
  in
  Testable.make ~pp:pp_ab ~equal:(fun r0 r1 -> fields r0 = fields r1)
  |> Testable.with_compare (Type.compare_value record_type)

let null_last cmp a b =
  match (a, b) with
  | None, None -> 0
  | None, Some _ -> 1
  | Some _, None -> -1
  | Some a, Some b -> cmp a b

(* The reference order of [record_type]'s cells: by [a], then [b], a null after
   every value. *)
let record_order r0 r1 =
  let a = Record.field Kind.int "a" and b = Record.field Kind.string "b" in
  match null_last Int.compare (a r0) (a r1) with
  | 0 -> null_last String.compare (b r0) (b r1)
  | c -> c

let tensor_type = Type.tensor Nx.float32 [| 2 |]
let vector dt a = Nx.create dt [| Array.length a |] a

let tensor_gen =
  Gen.map
    (fun (a, b) -> vector Nx.float32 [| a; b |])
    (Gen.pair
       (Gen.of_list [ 0.; -0.; 1.; Float.nan ])
       (Gen.of_list [ 0.; 1.; Float.nan ]))
  |> Gen.with_pp Nx.pp

let tensor_w =
  Testable.make ~pp:Nx.pp ~equal:(fun t0 t1 ->
      Array.for_all2 same_float (Nx.to_array t0) (Nx.to_array t1))
  |> Testable.with_compare (Type.compare_value tensor_type)

(* [code_points s] is the code points of the UTF-8 text [s]. *)
let code_points s =
  let rec go i acc =
    if i >= String.length s then List.rev acc
    else
      let d = String.get_utf_8_uchar s i in
      let u = Uchar.to_int (Uchar.utf_decode_uchar d) in
      go (i + Uchar.utf_decode_length d) (u :: acc)
  in
  go 0 []

let utf_8_text =
  Gen.map (String.concat "")
    (Gen.list ~size:(Gen.int_range 0 3)
       (Gen.of_list [ "a"; "z"; "\x7f"; "é"; "€"; "\u{FFFF}"; "😀" ]))

let int_values =
  let open Gen in
  let* name, t, lo, hi =
    of_list
      ~pp:(fun ppf (n, _, _, _) -> Format.pp_print_string ppf n)
      int_bounds
  in
  let+ vs =
    list ~size:(int_range 0 6)
      (frequency
         [ (2, int_range lo hi); (1, of_list [ lo; lo + 1; 0; hi - 1; hi ]) ])
  in
  (name, t, vs)

let pp_int_values ppf (name, _, vs) =
  Format.fprintf ppf "%s [%s]" name
    (String.concat "; " (List.map string_of_int vs))

(* A dictionary in any order, and values drawn from it. *)
let dictionary_case =
  let open Gen in
  let* d =
    permutation ~pp:(fun ppf -> Format.fprintf ppf "%S") [ "a"; "b"; "é"; "" ]
  in
  let+ vs = list ~size:(int_range 0 6) (of_list d) in
  (d, vs)

let position d s =
  let rec go i = function
    | [] -> invalid_arg "position"
    | x :: rest -> if x = s then i else go (i + 1) rest
  in
  go 0 d

let compare_values =
  group "Type.compare_value"
    [
      prop "floats order from -inf to inf then NaN, -0. as 0."
        ~examples:
          [
            (Type.float64, (Float.nan, Float.infinity, -.Float.nan));
            (Type.float32, (Float.neg_infinity, -.Float.max_float, -0.));
            (Type.float16, (-0., 0., Float.nan));
          ]
        (Gen.pair float_type
           (Gen.triple special_float special_float special_float))
        (fun (t, (a, b, c)) ->
          Law.order (float_w t) (a, b, c);
          equal
            (list (float_w t))
            (List.sort float_order [ a; b; c ])
            (List.sort (Type.compare_value t) [ a; b; c ]));
      prop "integers order by value" (Gen.with_pp pp_int_values int_values)
        (fun (_, t, vs) ->
          equal (list int) (List.sort Int.compare vs)
            (List.sort (Type.compare_value t) vs));
      test "false comes before true" (fun () ->
          less (by Type.bool bool) ~than:true false);
      prop "text orders by its code points, byte strings by their bytes"
        (Gen.pair
           (Gen.list ~size:(Gen.int_range 0 6) utf_8_text)
           (Gen.list ~size:(Gen.int_range 0 6) Gen.string))
        (fun (texts, bytes) ->
          let by_code_points a b =
            List.compare Int.compare (code_points a) (code_points b)
          in
          equal (list string)
            (List.sort by_code_points texts)
            (List.sort (Type.compare_value Type.string) texts);
          let bytes = List.map Binary.of_string bytes in
          equal (list binary_w) (List.sort compare bytes)
            (List.sort (Type.compare_value Type.binary) bytes));
      prop "categoricals order by their position in the dictionary"
        dictionary_case (fun (d, vs) ->
          let t = Type.categorical (Array.of_list d) in
          let by_position a b = Int.compare (position d a) (position d b) in
          equal (list string)
            (List.stable_sort by_position vs)
            (List.stable_sort (Type.compare_value t) vs));
      prop "dates, spans and stamps order by value"
        (Gen.pair
           (Gen.list ~size:(Gen.int_range 0 6) days_gen)
           (Gen.list ~size:(Gen.int_range 0 6) ns_range))
        (fun (days, ns) ->
          equal ~msg:"date" (list int)
            (List.sort Int.compare days)
            (List.map Time.Date.to_days
               (List.sort (Type.compare_value Type.date) (List.map date days)));
          let sorted t of_ns to_ns =
            List.map to_ns
              (List.sort (Type.compare_value t) (List.map of_ns ns))
          in
          let expected = List.sort Int64.compare ns in
          equal ~msg:"duration" (list int64) expected
            (sorted (Type.duration Ns) Time.Span.of_ns Time.Span.to_ns);
          equal ~msg:"clock" (list int64) expected
            (sorted (Type.clock Ns) Time.Span.of_ns Time.Span.to_ns);
          equal ~msg:"instant" (list int64) expected
            (sorted (Type.instant Ns) Time.of_ns Time.to_ns);
          equal ~msg:"wall" (list int64) expected
            (sorted (Type.wall Ns) Time.of_ns Time.to_ns));
      prop "lists order lexicographically, a proper prefix first"
        ~examples:
          [ ([| 1 |], [| 1; 0 |], [||]); ([| 2 |], [| 1; 9 |], [| 2 |]) ]
        (Gen.triple int_array int_array int_array)
        (fun (a, b, c) ->
          let prefix x y =
            Array.length x < Array.length y
            && Array.sub y 0 (Array.length x) = x
          in
          cover "a proper prefix"
            (List.exists
               (fun (x, y) -> prefix x y || prefix y x)
               [ (a, b); (b, c); (a, c) ]);
          let t = Type.list Type.int8 in
          Law.order (by t (array int)) (a, b, c);
          let lexicographic x y =
            List.compare Int.compare (Array.to_list x) (Array.to_list y)
          in
          equal
            (list (array int))
            (List.sort lexicographic [ a; b; c ])
            (List.sort (Type.compare_value t) [ a; b; c ]));
      prop "records order by their fields in the type's order, a null last"
        ~examples:
          [
            ( ab_record None None,
              ab_record (Some max_int) None,
              ab_record None None );
          ]
        (Gen.triple record_gen record_gen record_gen)
        (fun (a, b, c) ->
          Law.order record_w (a, b, c);
          equal (list record_w)
            (List.sort record_order [ a; b; c ])
            (List.sort (Type.compare_value record_type) [ a; b; c ]));
      test "an extension field orders by its storage value" (fun () ->
          let r1, r2 = masses () in
          less (ordered mass_type pp_names) ~than:r2 r1;
          equal (ordered mass_type pp_names) r1 r1);
      prop "float tensors order by their elements in row-major order"
        (Gen.triple tensor_gen tensor_gen tensor_gen) (fun (a, b, c) ->
          Law.order tensor_w (a, b, c);
          let key t = List.map float_key (Array.to_list (Nx.to_array t)) in
          equal (list tensor_w)
            (List.sort (fun x y -> compare (key x) (key y)) [ a; b; c ])
            (List.sort (Type.compare_value tensor_type) [ a; b; c ]));
      test "tensors of other dtypes order by element, unsigned as unsigned"
        (fun () ->
          let before dt a b =
            let w = ordered (Type.tensor dt [| 1 |]) Nx.pp in
            less w ~than:(vector dt [| b |]) (vector dt [| a |])
          in
          before Nx.int8 (-1) 1;
          before Nx.int64 (-1L) 1L;
          before Nx.uint32 1l (-1l);
          before Nx.uint64 1L (-1L);
          before Nx.bool false true;
          before Nx.float64 Float.infinity Float.nan;
          let c re im = { Complex.re; im } in
          before Nx.complex64 (c 1. 5.) (c 2. 0.);
          before Nx.complex64 (c 1. 0.) (c 1. Float.nan));
      test "a transposed view orders by its elements in row-major order"
        (fun () ->
          let w = ordered (Type.tensor Nx.int32 [| 2; 2 |]) Nx.pp in
          let m a = Nx.create Nx.int32 [| 2; 2 |] a in
          let view = Nx.transpose (m [| 1l; 2l; 3l; 4l |]) in
          equal w (m [| 1l; 3l; 2l; 4l |]) view;
          less w ~than:view (m [| 1l; 2l; 3l; 4l |]));
      test "a null list-of-extension field is held and orders as null"
        (fun () ->
          let t = Type.record [ ("l", Any (Type.list ext_type)) ] in
          let r = Record.(add (Kind.list Kind.float) "l" None empty) in
          equal (list string) [ "l=null" ] (labels t [ ("l=null", r) ]);
          equal (ordered t pp_names) r r);
      test "refuses a category outside the dictionary" (fun () ->
          rejects (fun () ->
              Type.compare_value (Type.categorical [| "a" |]) "a" "b"));
      test "refuses a record without the type's names" (fun () ->
          let r = Record.(add Kind.int "a" None empty) in
          rejects (fun () -> Type.compare_value record_type r r));
      test "refuses a record field of another kind" (fun () ->
          let r =
            Record.(
              empty |> add Kind.float "a" (Some 1.) |> add Kind.string "b" None)
          in
          rejects (fun () -> Type.compare_value record_type r r));
      test "refuses a plain value in an extension field" (fun () ->
          let r = Record.(add Kind.float "m" (Some 1.) empty) in
          rejects (fun () -> Type.compare_value mass_type r r));
      test "refuses a tensor of another shape" (fun () ->
          let v = vector Nx.float32 [| 1. |] in
          rejects (fun () -> Type.compare_value tensor_type v v));
    ]

(* Operands *)

let int_types = List.map (fun (_, t, _, _) -> t) int_bounds

(* Each integer type's range, as floats: every bound is exact, or a distinct
   power of two. *)
let int_range t =
  let open Type in
  match t with
  | Int8 -> (-128., 127.)
  | Int16 -> (-32768., 32767.)
  | Int32 -> (-2147483648., 2147483647.)
  | Int64 -> (-.ldexp 1. 63, ldexp 1. 63)
  | Uint8 -> (0., 255.)
  | Uint16 -> (0., 65535.)
  | Uint32 -> (0., 4294967295.)
  | Uint64 -> (0., ldexp 1. 64)
  | _ -> invalid_arg "int_range"

let reference_common ts =
  let within u t =
    let lu, hu = int_range u and lt, ht = int_range t in
    lt <= lu && hu <= ht
  in
  List.find_opt (fun t -> List.for_all (fun u -> within u t) ts) ts

let cat d = Type.categorical d

(* Types of one kind, a list of them and a permutation of it. *)
type meeting = Meeting : 'a Type.t list * 'a Type.t list -> meeting

let meeting ts =
  let open Gen in
  let* l = list ~size:(int_range 0 5) (of_list ts) in
  let+ p = permutation l in
  Meeting (l, p)

let pp_meeting ppf (Meeting (l, p)) =
  let types =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
      Type.pp
  in
  Format.fprintf ppf "[%a] and [%a]" types l types p

(* Integers; text and dictionaries that begin with each other; clocks of every
   unit and durations. *)
let meetings =
  Gen.with_pp pp_meeting
    (Gen.one_of
       [
         meeting int_types;
         meeting
           [
             Type.string;
             cat [| "a" |];
             cat [| "a"; "b" |];
             cat [| "a"; "b"; "c" |];
             cat [| "b" |];
             cat [| "b"; "a" |];
           ];
         meeting
           Type.
             [ clock S; clock Ms; clock Us; clock Ns; duration S; duration Ns ];
       ])

let dictionary : type a. a Type.t -> string list option = function
  | Type.Categorical d -> Some (Array.to_list (Iarray.to_array d))
  | _ -> None

let common_cases =
  let some t = Some (Type.Any t) and none = None in
  let common ts = Option.map (fun t -> Type.Any t) (Type.common ts) in
  group "Type.common"
    [
      test "nothing meets in an empty list" (fun () ->
          equal (option any_w) none (common ([] : int Type.t list)));
      test "a type meets itself" (fun () ->
          equal (option any_w) (some Type.date)
            (common [ Type.date; Type.date ]));
      prop "integer types meet at the one whose range holds the others"
        ~examples:
          Type.
            [
              [ int8; uint8 ];
              [ int8; uint8; int16 ];
              [ int16; int8; uint8 ];
              [ uint8; int16 ];
              [ uint32; int64 ];
              [ uint64; int64 ];
            ]
        (Gen.list ~size:(Gen.int_range 1 5) (Gen.of_list ~pp:Type.pp int_types))
        (fun ts ->
          equal (option any_w)
            (Option.map (fun t -> Type.Any t) (reference_common ts))
            (common ts));
      prop "does not depend on the order of the types"
        ~examples:
          [
            Meeting
              ( [ cat [| "a"; "b" |]; cat [| "a" |] ],
                [ cat [| "a" |]; cat [| "a"; "b" |] ] );
          ]
        meetings
        (fun (Meeting (ts, p)) ->
          let met = Type.common ts in
          let dictionaries =
            List.sort_uniq compare (List.filter_map dictionary ts)
          in
          cover "meets" (Option.is_some met);
          cover "does not meet" (ts <> [] && Option.is_none met);
          cover "a categorical extending another"
            (List.length dictionaries >= 2
            &&
            match met with
            | Some t -> Option.is_some (dictionary t)
            | None -> false);
          equal (option any_w) (common ts) (common p));
      test "floats meet at the wider" (fun () ->
          equal (option any_w) (some Type.float32)
            (common [ Type.float16; Type.float32 ]));
      test
        "a categorical meets string, and a categorical extending its dictionary"
        (fun () ->
          equal (option any_w) (some Type.string)
            (common [ cat [| "a" |]; Type.string ]);
          equal (option any_w)
            (some (cat [| "a"; "b" |]))
            (common [ cat [| "a"; "b" |]; cat [| "a" |] ]);
          equal (option any_w) none
            (common [ cat [| "b"; "a" |]; cat [| "a" |] ]));
      test "clocks meet at the finer unit, other temporal types only when equal"
        (fun () ->
          equal (option any_w)
            (some (Type.clock Type.Ns))
            (common [ Type.clock Type.S; Type.clock Type.Ns ]);
          equal (option any_w) none
            (common [ Type.clock Type.S; Type.duration Type.S ]);
          equal (option any_w) none
            (common [ Type.duration Type.S; Type.duration Type.Ns ]);
          equal (option any_w) none
            (common [ Type.wall Type.S; Type.wall Type.Ms ]));
      test "lists meet element by element" (fun () ->
          equal (option any_w)
            (some (Type.list Type.int32))
            (common [ Type.list Type.int8; Type.list Type.int32 ]);
          equal (option any_w) none
            (common [ Type.list Type.int8; Type.list Type.uint8 ]));
      test "records with the same names meet field by field" (fun () ->
          let r a b = Type.record [ ("a", Any a); ("b", Any b) ] in
          equal (option any_w)
            (some (r Type.int16 Type.float64))
            (common [ r Type.int8 Type.float64; r Type.int16 Type.float32 ]);
          equal (option any_w) none
            (common
               [
                 r Type.int8 Type.float64;
                 Type.record [ ("b", Any Type.float64); ("a", Any Type.int8) ];
               ]);
          equal (option any_w) none
            (common [ r Type.int8 Type.float64; r Type.float32 Type.float64 ]);
          equal (option any_w)
            (some (Type.record []))
            (common [ Type.record []; Type.record [] ]));
      test "record fields of extension type meet when equal" (fun () ->
          equal (option any_w) (some mass_type)
            (common [ mass_type; mass_type ]));
      test "extensions and tensors meet only when equal" (fun () ->
          equal (option any_w) (some ext_type) (common [ ext_type; ext_type ]);
          equal (option any_w) none
            (common [ ext_type; Type.ext ~name:"units.mass" Type.float64 ]);
          equal (option any_w) none
            (common
               [
                 Type.tensor Nx.float32 [| 2 |]; Type.tensor Nx.float32 [| 3 |];
               ]));
    ]

let type_gen =
  Gen.of_list ~pp:pp_any
    [
      Any Type.bool;
      Any Type.int8;
      Any Type.int16;
      Any Type.int32;
      Any Type.int64;
      Any Type.uint8;
      Any Type.uint16;
      Any Type.uint32;
      Any Type.uint64;
      Any Type.float16;
      Any Type.float32;
      Any Type.float64;
      Any Type.string;
      Any Type.binary;
      Any Type.date;
      Any (Type.clock Type.Ms);
      Any (Type.categorical [| "a"; "b" |]);
      Any (Type.categorical [| "a"; "b" |]);
      Any (Type.categorical [| "b"; "a" |]);
      Any (Type.wall Type.Us);
      Any (Type.instant Type.Us);
      Any (Type.clock Type.Us);
      Any (Type.duration Type.Us);
      Any (Type.list Type.int8);
      Any (Type.record [ ("a", Any Type.int8) ]);
      Any (Type.record [ ("a", Any Type.int8) ]);
      Any (Type.record [ ("a", Any Type.uint8) ]);
      Any (Type.tensor Nx.float32 [| 2 |]);
      Any (Type.tensor Nx.float64 [| 2 |]);
      Any ext_type;
      Any (Type.ext ~name:"units.mass" Type.float64);
    ]

let type_equal =
  group "Type.equal"
    [
      prop "is an equivalence"
        (Gen.pair type_gen type_gen)
        (Law.equivalence any_w);
      test "compares dictionaries, clocks and fields in order" (fun () ->
          not_equal any_w
            (Any (Type.categorical [| "a"; "b" |]))
            (Any (Type.categorical [| "b"; "a" |]));
          not_equal any_w (Any (Type.instant Type.S)) (Any (Type.wall Type.S));
          not_equal any_w
            (Any (Type.record [ ("a", Any Type.int8); ("b", Any Type.int8) ]))
            (Any (Type.record [ ("b", Any Type.int8); ("a", Any Type.int8) ])));
      test "types of different kinds are not equal" (fun () ->
          not_equal any_w (Any Type.int8) (Any Type.float32));
      test "extensions are equal only with equal names, metadata and storage"
        (fun () ->
          let x = Type.Any (Type.ext ~name:"x" ~metadata:"m" Type.float64) in
          not_equal any_w x
            (Any (Type.ext ~name:"y" ~metadata:"m" Type.float64));
          not_equal any_w x
            (Any (Type.ext ~name:"x" ~metadata:"n" Type.float64));
          not_equal any_w x
            (Any (Type.ext ~name:"x" ~metadata:"m" Type.float32)));
    ]

let type_pp =
  let alphabet = Array.init 26 (fun i -> String.make 1 (Char.chr (97 + i))) in
  cases "Type.pp formats" ~id:snd
    [
      (Type.Any Type.bool, "bool");
      (Any Type.int8, "int8");
      (Any Type.int16, "int16");
      (Any Type.int32, "int32");
      (Any Type.int64, "int64");
      (Any Type.uint8, "uint8");
      (Any Type.uint16, "uint16");
      (Any Type.uint32, "uint32");
      (Any Type.uint64, "uint64");
      (Any Type.float16, "float16");
      (Any Type.float32, "float32");
      (Any Type.float64, "float64");
      (Any Type.string, "string");
      (Any Type.binary, "binary");
      (Any Type.date, "date");
      (Any (Type.categorical [| "AA"; "B6" |]), {|categorical["AA", "B6"]|});
      ( Any (Type.categorical alphabet),
        {|categorical["a", "b", "c", "d", "e", "f", "g", "h", … 26]|} );
      ( Any (Type.categorical (Array.sub alphabet 0 8)),
        {|categorical["a", "b", "c", "d", "e", "f", "g", "h"]|} );
      ( Any (Type.categorical [| {|say "hi"|}; "a\\b\n" |]),
        {|categorical["say \"hi\"", "a\\b\x0a"]|} );
      (Any (Type.clock Type.Ns), "clock[ns]");
      (Any (Type.duration Type.Ms), "duration[ms]");
      (Any (Type.wall Type.Us), "datetime[us]");
      (Any (Type.wall Type.S), "datetime[s]");
      (Any (Type.instant Type.Us), "datetime[us, UTC]");
      (Any (Type.list Type.float64), "list[float64]");
      ( Any
          (Type.record
             [ ("carrier", Any Type.string); ("delay", Any Type.float64) ]),
        "record[carrier string, delay float64]" );
      ( Any
          (Type.record
             [
               ("a b", Any Type.int8);
               ("c,d", Any Type.int8);
               ("é", Any Type.int8);
             ]),
        {|record["a b" int8, "c,d" int8, é int8]|} );
      ( Any (Type.record [ ("a\tb", Any Type.int8); ("\x7f", Any Type.int8) ]),
        {|record["a\x09b" int8, "\x7f" int8]|} );
      ( Any (Type.record [ ("a\\b", Any Type.int8); ("\"", Any Type.int8) ]),
        {|record["a\\b" int8, "\"" int8]|} );
      (Any (Type.tensor Nx.float32 [| 3; 4 |]), "tensor[float32, 3×4]");
      (Any (Type.tensor Nx.int64 [| 5 |]), "tensor[int64, 5]");
      ( Any (Type.ext ~name:"ymir.epoch" Type.float64),
        "ext[ymir.epoch, float64]" );
      (Any ext_type, {|ext[units.mass "kg", float64]|});
      ( Any (Type.ext ~name:"a[1]" ~metadata:"\x01" Type.int8),
        {|ext["a[1]" "\x01", int8]|} );
    ]
    (fun (t, expected) -> equal string expected (str pp_any t))

(* Quoted strings *)

(* [by_code_point ppf s] is the quoting written one code point at a time: a
   double quote and a backslash take a backslash, a control byte and an invalid
   byte print as [\xhh], and any other code point prints as itself. *)
let by_code_point ppf s =
  let hex c = Format.fprintf ppf "\\x%02x" (Char.code c) in
  let rec loop i =
    if i < String.length s then (
      let d = String.get_utf_8_uchar s i in
      if not (Uchar.utf_decode_is_valid d) then (
        hex s.[i];
        loop (i + 1))
      else
        let n = Uchar.utf_decode_length d in
        (match s.[i] with
        | ('"' | '\\') as c -> Format.fprintf ppf "\\%c" c
        | c when Char.code c < 0x20 || Char.code c = 0x7f -> hex c
        | _ -> Format.pp_print_string ppf (String.sub s i n));
        loop (i + n))
  in
  Format.pp_print_char ppf '"';
  loop 0;
  Format.pp_print_char ppf '"'

(* [boxed pp s] is [pp] of [s] between break hints, at a margin narrower than
   most strings, as a plan prints a name. *)
let boxed pp s =
  let b = Buffer.create 64 in
  let ppf = Format.formatter_of_buffer b in
  Format.pp_set_margin ppf 12;
  Format.fprintf ppf "@[<hov 2>head@ %a@ tail@]@?" pp s;
  Buffer.contents b

let quotable =
  let byte low high = Gen.map (String.make 1) (Gen.char_range low high) in
  let piece =
    Gen.frequency
      [
        (8, byte 'a' 'z');
        (1, Gen.constant "\"");
        (1, Gen.constant "\\");
        (1, byte '\000' '\031');
        (1, Gen.constant "\127");
        (2, Gen.constant "\xc3\xa9");
        (2, Gen.constant "\xe2\x82\xac");
        (2, Gen.constant "\xf0\x9f\x98\x80");
        (2, byte '\128' '\255');
      ]
  in
  Gen.map (String.concat "") (Gen.list ~size:(Gen.int_range 0 40) piece)

let law_quoted s =
  let special c = c = '"' || c = '\\' || Char.code c < 0x20 || c = '\127' in
  let decoded p =
    let rec go i =
      i < String.length s && (p (String.get_utf_8_uchar s i) || go (i + 1))
    in
    go 0
  in
  let valid = Uchar.utf_decode_is_valid in
  cover "a byte to escape" (String.exists special s);
  cover "an invalid byte" (decoded (fun d -> not (valid d)));
  cover "a valid multi-byte scalar"
    (decoded (fun d -> valid d && Uchar.utf_decode_length d > 1));
  cover "no byte to escape" (s <> "" && not (String.exists special s));
  equal string (boxed by_code_point s) (boxed Type.pp_quoted s)

(* Names of letters, the bytes that make a name quoted, and UTF-8. *)
let name_gen =
  let piece =
    Gen.frequency
      [
        (8, Gen.map (String.make 1) (Gen.char_range 'a' 'z'));
        (1, Gen.of_list [ " "; ","; "["; "]"; "\""; "\\"; "\x01"; "\x7f" ]);
        (1, Gen.constant "é");
      ]
  in
  Gen.map (String.concat "") (Gen.list ~size:(Gen.int_range 0 6) piece)

(* A name byte that leaves a name bare: no ASCII space, comma, bracket, double
   quote, backslash or control byte. *)
let bare c =
  Char.code c > 0x20 && c <> '\x7f' && not (String.contains ",[]\"\\" c)

let quoted =
  group "Type.pp_quoted"
    [
      test "escapes quotes, controls and invalid bytes" (fun () ->
          equal string {|"a\"b\\c\x01é\xff\x7f"|}
            (str Type.pp_quoted "a\"b\\c\001\xc3\xa9\xff\127"));
      prop ~count:500 "prints as its code points do, one at a time" quotable
        law_quoted;
    ]

let pp_name =
  prop "Type.pp_name writes a name as is, or quoted when it must be"
    ~examples:[ ""; "a b"; "é" ] name_gen (fun s ->
      let is_bare = s <> "" && String.for_all bare s in
      cover "as is" is_bare;
      cover "quoted" (not is_bare);
      let expected = if is_bare then s else str Type.pp_quoted s in
      equal string expected (str Type.pp_name s))

(* Schemas *)

let columns = [ ("carrier", Type.Any Type.string); ("delay", Any Type.float64) ]
let schema_w = Testable.make ~pp:Schema.pp ~equal:Schema.equal
let column_names = [ "a"; "b"; "c"; ""; "é" ]

let column_type =
  Gen.of_list ~pp:pp_any Type.[ Any int8; Any int16; Any string; Any float64 ]

(* Columns of distinct names, in any order. *)
let columns_gen =
  let open Gen in
  let* names = subsequence column_names in
  let* names = permutation names in
  let+ ts = list ~size:(constant (List.length names)) column_type in
  List.combine names ts

let retyped (Type.Any t) =
  if Type.equal t Type.int8 then Type.Any Type.int16 else Any Type.int8

(* [changed cs] is [cs] with columns dropped, reordered and retyped, and new
   columns added. *)
let changed cs =
  let open Gen in
  let* kept = subsequence cs in
  let* kept = permutation kept in
  let* flags = list ~size:(constant (List.length kept)) bool in
  let+ added = columns_gen in
  List.map2 (fun (n, t) f -> if f then (n, retyped t) else (n, t)) kept flags
  @ List.filter (fun (n, _) -> not (List.mem_assoc n kept)) added

let pp_columns ppf cs = Schema.pp ppf (Schema.v cs)

(* Two schemas' columns: the second the first changed, equal, or unrelated. *)
let schema_pair =
  let open Gen in
  let pp ppf (a, b) =
    Format.fprintf ppf "[%a] to [%a]" pp_columns a pp_columns b
  in
  with_pp pp
    (let* cs0 = columns_gen in
     let+ cs1 =
       frequency [ (3, changed cs0); (1, columns_gen); (1, constant cs0) ]
     in
     (cs0, cs1))

let same_column (n0, Type.Any t0) (n1, Type.Any t1) =
  n0 = n1 && Type.equal t0 t1

let pp_change ppf = function
  | Schema.Added (n, t) -> Format.fprintf ppf "+%s %a" n pp_any t
  | Removed (n, t) -> Format.fprintf ppf "-%s %a" n pp_any t
  | Retyped (n, t0, t1) ->
      Format.fprintf ppf "~%s %a -> %a" n pp_any t0 pp_any t1

(* The changes from [cs0] to [cs1], as the interface orders them. *)
let reference_diff cs0 cs1 =
  let removed_or_retyped (n, (Type.Any t0 as a)) =
    match List.assoc_opt n cs1 with
    | None -> Some (Schema.Removed (n, a))
    | Some (Type.Any t1 as b) ->
        if Type.equal t0 t1 then None else Some (Retyped (n, a, b))
  in
  let added (n, t) =
    if List.mem_assoc n cs0 then None else Some (Schema.Added (n, t))
  in
  List.filter_map removed_or_retyped cs0 @ List.filter_map added cs1

let schemas =
  group "Schema"
    [
      test "v refuses a duplicate or non-UTF-8 name" (fun () ->
          rejects (fun () ->
              Schema.v [ ("a", Any Type.bool); ("a", Any Type.bool) ]);
          rejects (fun () -> Schema.v [ ("\xff", Any Type.bool) ]));
      prop "columns gives back v's columns, and find looks one up by name"
        (Gen.with_pp pp_columns columns_gen) (fun cs ->
          let s = Schema.v cs in
          equal (list (pair string any_w)) cs (Schema.columns s);
          List.iter
            (fun n ->
              equal ~msg:n (option any_w) (List.assoc_opt n cs)
                (Schema.find s n))
            ("z" :: column_names));
      prop "equal needs the same names with equal types in the same order"
        schema_pair (fun (cs0, cs1) ->
          let same = List.equal same_column cs0 cs1 in
          cover "equal" same;
          cover "not equal" (not same);
          if same then equal schema_w (Schema.v cs0) (Schema.v cs1)
          else not_equal schema_w (Schema.v cs0) (Schema.v cs1));
      prop "diff lists removed and retyped columns, then added ones"
        ~examples:
          [
            ( columns,
              [
                ("origin", Type.Any Type.string);
                ("delay", Any Type.float32);
                ("dest", Any Type.string);
              ] );
            (columns, List.rev columns);
          ]
        schema_pair
        (fun (cs0, cs1) ->
          let expected = reference_diff cs0 cs1 in
          cover "a column removed or retyped"
            (List.exists
               (function Schema.Added _ -> false | _ -> true)
               expected);
          cover "a column added"
            (List.exists
               (function Schema.Added _ -> true | _ -> false)
               expected);
          cover "the same columns in another order"
            (expected = [] && not (List.equal same_column cs0 cs1));
          equal (list string)
            (List.map (str pp_change) expected)
            (List.map (str pp_change)
               (Schema.diff (Schema.v cs0) (Schema.v cs1))));
      cases "pp formats" ~id:snd
        [
          (columns, "carrier string, delay float64");
          ([], "");
          ( [ ("a b", Any Type.int8); ("", Any Type.bool) ],
            {|"a b" int8, "" bool|} );
        ]
        (fun (cs, expected) ->
          equal string expected (str Schema.pp (Schema.v cs)));
    ]

(* Messages *)

let message f =
  match f () with () -> "no exception" | exception Invalid_argument m -> m

let messages =
  test "a refusal names its function and the value at fault" (fun () ->
      let messages =
        List.map message
          [
            (fun () -> ignore (Type.categorical [| "a"; "a" |]));
            (fun () -> ignore (Type.tensor Nx.float32 [| 2; -1 |]));
            (fun () -> ignore (Record.field Kind.float "a" ab));
            (fun () -> ignore (Schema.v [ ("\xff", Any Type.bool) ]));
            (fun () -> ignore (Time.Span.hours max_int));
            (fun () ->
              ignore (Type.compare_value (Type.categorical [| "a" |]) "a" "b"));
          ]
      in
      expect (String.concat "\n" messages)
      @@ __POS_OF__
           {|
            Type.categorical: "a" appears twice
            Type.tensor: dimension -1 is negative
            Record.field: field "a" is int, not float
            Schema.v: column name "\255" is not UTF-8
            Time.Span.hours: 4611686018427387903 is out of range
            Type.compare_value: "b" is not in the dictionary
            |})

let () =
  exit
    (run "talon types"
       [
         binary;
         instants;
         spans;
         dates;
         step_pp;
         kinds;
         records;
         type_constructors;
         type_kind;
         holds;
         compare_values;
         common_cases;
         type_equal;
         type_pp;
         quoted;
         pp_name;
         schemas;
         messages;
       ])
