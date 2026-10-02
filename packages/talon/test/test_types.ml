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

(* The spec's float identity: [-0.] is [0.] and every NaN is every other. *)
let same_float a b = (Float.is_nan a && Float.is_nan b) || a = b

(* Binary *)

let binary =
  group "Binary"
    [
      test "keeps the bytes it is made of" (fun () ->
          let s = "\x00\xff\n" in
          equal string s (Binary.of_string s :> string));
      cases
        ~name:(fun (s, _) -> Printf.sprintf "formats %S in hexadecimal" s)
        "pp"
        [ ("Hi\n", "0x48690a"); ("", "0x"); ("\x00\xff", "0x00ff") ]
        (fun (s, expected) ->
          equal string expected (str Binary.pp (Binary.of_string s)));
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

let instant_conversions =
  group "instant conversions"
    [
      test "of_ns is the identity on nanoseconds" (fun () ->
          equal int64 Int64.min_int (Time.to_ns (Time.of_ns Int64.min_int)));
      cases
        ~name:(fun (n, _, _, _) ->
          Printf.sprintf "to_%s rounds toward negative infinity" n)
        "floor" units
        (fun (_, u, _, to_u) ->
          equal int64 (-1L) (to_u (Time.of_ns (-1L)));
          let t = Time.of_ns (Int64.add (Int64.mul 5L u) (Int64.pred u)) in
          equal int64 5L (to_u t));
      prop "to_* is the floor of the nanoseconds" ns_range (fun n ->
          List.iter
            (fun (name, u, _, to_u) ->
              let r = Int64.sub n (Int64.mul (to_u (Time.of_ns n)) u) in
              at_least ~msg:name int64 ~than:0L r;
              less ~msg:name int64 ~than:u r)
            units);
      prop "of_* inverts to_* on whole units" ns_range (fun n ->
          List.iter
            (fun (name, u, of_u, to_u) ->
              let whole = Int64.mul (Int64.div n u) u in
              let t = Time.of_ns whole in
              equal ~msg:name (option instant_w) (Some t) (of_u (to_u t)))
            units);
      cases
        ~name:(fun (n, _, _, _) ->
          Printf.sprintf "of_%s is None exactly past the range" n)
        "overflow" units
        (fun (_, u, of_u, _) ->
          let hi = Int64.div Int64.max_int u
          and lo = Int64.div Int64.min_int u in
          is_some (of_u hi);
          is_some (of_u lo);
          is_none ~pp:Time.pp (of_u (Int64.succ hi));
          is_none ~pp:Time.pp (of_u (Int64.pred lo)));
      prop "order is chronological" (Gen.triple ns_range ns_range ns_range)
        (fun (a, b, c) ->
          Law.order instant_w (Time.of_ns a, Time.of_ns b, Time.of_ns c);
          equal int (Int64.compare a b)
            (Time.compare (Time.of_ns a) (Time.of_ns b)));
    ]

let gmtime_string s =
  let tm = Unix.gmtime (Int64.to_float s) in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02d" (tm.tm_year + 1900)
    (tm.tm_mon + 1) tm.tm_mday tm.tm_hour tm.tm_min tm.tm_sec

let instant_pp =
  group "Time.pp"
    [
      cases
        ~name:(fun (_, s) -> Printf.sprintf "formats %s" s)
        "spec"
        [
          (Time.of_ns 1710495000_000_000_000L, "2024-03-15T09:30:00");
          (Time.of_ns (-1L), "1969-12-31T23:59:59.999999999");
          (Time.of_ns 500_000_000L, "1970-01-01T00:00:00.500");
          (Time.of_ns 1_000L, "1970-01-01T00:00:00.000001");
          (Time.of_ns 1_001_000L, "1970-01-01T00:00:00.001001");
        ]
        (fun (t, expected) -> equal string expected (str Time.pp t));
      test "formats the ends of the range" (fun () ->
          expect
            (str Time.pp (Time.of_ns Int64.min_int)
            ^ "\n"
            ^ str Time.pp (Time.of_ns Int64.max_int))
          @@ __POS_OF__
               {|
                 1677-09-21T00:12:43.145224192
                 2262-04-11T23:47:16.854775807
               |});
      prop "agrees with gmtime on whole seconds"
        (Gen.int_range (-9_223_372_036) 9_223_372_035) (fun s ->
          let s = Int64.of_int s in
          equal string (gmtime_string s)
            (str Time.pp (Option.get (Time.of_s s))));
    ]

(* Spans *)

let span_constructors =
  let ns_per =
    [
      ("us", Time.Span.us, 1_000L);
      ("ms", Time.Span.ms, 1_000_000L);
      ("s", Time.Span.s, 1_000_000_000L);
      ("minutes", Time.Span.minutes, 60_000_000_000L);
      ("hours", Time.Span.hours, 3_600_000_000_000L);
      ("days", Time.Span.days, 86_400_000_000_000L);
    ]
  in
  group "Span constructors"
    [
      cases
        ~name:(fun (n, _, _) ->
          Printf.sprintf "%s scales to nanoseconds and raises past the range" n)
        "units" ns_per
        (fun (_, make, u) ->
          equal span_w (Time.Span.of_ns (Int64.mul 3L u)) (make 3);
          equal span_w (Time.Span.of_ns (Int64.mul (-3L) u)) (make (-3));
          let hi = Int64.to_int (Int64.div Int64.max_int u) in
          let lo = Int64.to_int (Int64.div Int64.min_int u) in
          equal int64
            (Int64.mul (Int64.of_int hi) u)
            (Time.Span.to_ns (make hi));
          equal int64
            (Int64.mul (Int64.of_int lo) u)
            (Time.Span.to_ns (make lo));
          rejects (fun () -> make (hi + 1));
          rejects (fun () -> make (lo - 1));
          rejects (fun () -> make max_int));
      test "ns takes any int" (fun () ->
          equal int64 (Int64.of_int min_int)
            (Time.Span.to_ns (Time.Span.ns min_int)));
      test "to_us rounds toward negative infinity" (fun () ->
          equal int64 (-1L) (Time.Span.to_us (Time.Span.ns (-1))));
      prop "to_* is the floor and of_* its inverse on whole units" ns_range
        (fun n ->
          let d = Time.Span.of_ns n in
          List.iter
            (fun (name, u, of_u, to_u) ->
              let r = Int64.sub n (Int64.mul (to_u d) u) in
              at_least ~msg:name int64 ~than:0L r;
              less ~msg:name int64 ~than:u r;
              let whole = Time.Span.of_ns (Int64.mul (Int64.div n u) u) in
              equal ~msg:name (option span_w) (Some whole) (of_u (to_u whole)))
            [
              ("us", 1_000L, Time.Span.of_us, Time.Span.to_us);
              ("ms", 1_000_000L, Time.Span.of_ms, Time.Span.to_ms);
              ("s", 1_000_000_000L, Time.Span.of_s, Time.Span.to_s);
            ]);
      test "of_s is None past the range" (fun () ->
          is_none ~pp:Time.Span.pp
            (Time.Span.of_s
               (Int64.succ (Int64.div Int64.max_int 1_000_000_000L)));
          is_none ~pp:Time.Span.pp (Time.Span.of_us Int64.min_int));
      prop "order is numeric" (Gen.triple ns_range ns_range ns_range)
        (fun (a, b, c) ->
          Law.order span_w
            (Time.Span.of_ns a, Time.Span.of_ns b, Time.Span.of_ns c));
    ]

let span_pp =
  cases
    ~name:(fun (_, s) -> Printf.sprintf "Span.pp formats %s" s)
    "Span.pp"
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
    (fun (d, expected) -> equal string expected (str Time.Span.pp d))

let step_pp =
  cases
    ~name:(fun (_, s) -> Printf.sprintf "pp_step formats %s" s)
    "pp_step"
    [
      (Time.Months 1, "1mo");
      (Time.Weeks 2, "2w");
      (Time.Days (-1), "-1d");
      (Time.Exact (Time.Span.minutes 15), "15m");
    ]
    (fun (s, expected) -> equal string expected (str Time.pp_step s))

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

let dates =
  group "Time.Date"
    [
      cases
        ~name:(fun (c, _) ->
          let y, m, d = c in
          Printf.sprintf "places %d-%d-%d" y m d)
        "anchors"
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
            (Option.map Time.Date.to_days (Time.Date.of_civil civil));
          equal civil_w civil (Time.Date.to_civil (date days)));
      cases
        ~name:(fun (y, m, d) -> Printf.sprintf "refuses %d-%d-%d" y m d)
        "invalid"
        [
          (2023, 2, 29);
          (1900, 2, 29);
          (2024, 0, 1);
          (2024, 13, 1);
          (2024, 4, 31);
          (2024, 1, 0);
          (2024, 1, 32);
          (10_000_000, 1, 1);
          (-10_000_000, 1, 1);
          (max_int, 1, 1);
          (min_int, 1, 1);
        ]
        (fun civil -> is_none ~pp:Time.Date.pp (Time.Date.of_civil civil));
      test "refuses a year whose day count wraps into the range" (fun () ->
          is_none ~pp:Time.Date.pp
            (Time.Date.of_civil (3_535_382_889_881_440_000, 3, 1)));
      test "spans the years the interface states" (fun () ->
          let y0, _, _ = Time.Date.to_civil (date min_days) in
          let y1, _, _ = Time.Date.to_civil (date max_days) in
          equal (pair int int) (-5_877_641, 5_881_580) (y0, y1));
      test "of_days refuses days past the int32 range" (fun () ->
          is_none ~pp:Time.Date.pp (Time.Date.of_days (max_days + 1));
          is_none ~pp:Time.Date.pp (Time.Date.of_days (min_days - 1)));
      test "of_civil refuses the day after the last date" (fun () ->
          let last = Time.Date.to_civil (date max_days) in
          let first = Time.Date.to_civil (date min_days) in
          is_none ~pp:Time.Date.pp (Time.Date.of_civil (next_civil last));
          let y, m, d = first in
          let before =
            if d > 1 then (y, m, d - 1) else (y, m - 1, month_length y (m - 1))
          in
          is_none ~pp:Time.Date.pp (Time.Date.of_civil before));
      prop "of_civil inverts to_civil" ~examples:[ min_days; max_days; 11016 ]
        days_gen (fun n ->
          Law.round_trip date_w civil_w Time.Date.to_civil
            (fun c -> require_some (Time.Date.of_civil c))
            (date n));
      prop "of_days inverts to_days" days_gen (fun n ->
          equal int n (Time.Date.to_days (date n)));
      prop "the next day is the next civil date" days_gen (fun n ->
          assume (n < max_days);
          equal civil_w
            (next_civil (Time.Date.to_civil (date n)))
            (Time.Date.to_civil (date (n + 1))));
      prop "order is chronological" (Gen.triple days_gen days_gen days_gen)
        (fun (a, b, c) ->
          Law.order date_w (date a, date b, date c);
          equal int (Int.compare a b) (Time.Date.compare (date a) (date b)));
      cases
        ~name:(fun (_, s) -> Printf.sprintf "pp formats %s" s)
        "pp"
        [
          ((1970, 1, 1), "1970-01-01");
          ((-44, 3, 15), "-0044-03-15");
          ((12345, 1, 1), "+12345-01-01");
          ((0, 1, 1), "0000-01-01");
          ((9999, 12, 31), "9999-12-31");
          ((10000, 1, 1), "+10000-01-01");
          ((-1, 12, 31), "-0001-12-31");
        ]
        (fun (civil, expected) ->
          equal string expected
            (str Time.Date.pp (Option.get (Time.Date.of_civil civil))));
    ]

(* Kinds *)

let ext_type = Type.ext ~name:"units.mass" ~metadata:"kg" Type.float64

let kinds =
  group "Kind"
    [
      test "proves a kind equal to itself" (fun () ->
          is_some (Kind.provably_equal Kind.bool Kind.bool);
          is_some (Kind.provably_equal Kind.int Kind.int);
          is_some (Kind.provably_equal Kind.float Kind.float);
          is_some (Kind.provably_equal Kind.string Kind.string);
          is_some (Kind.provably_equal Kind.binary Kind.binary);
          is_some (Kind.provably_equal Kind.date Kind.date);
          is_some (Kind.provably_equal Kind.instant Kind.instant);
          is_some (Kind.provably_equal Kind.span Kind.span);
          is_some
            (Kind.provably_equal (Kind.list Kind.string) (Kind.list Kind.string));
          is_some
            (Kind.provably_equal (Kind.tensor Nx.float32)
               (Kind.tensor Nx.float32));
          is_some (Kind.provably_equal Record.kind Record.kind));
      test "does not prove different kinds equal" (fun () ->
          is_none (Kind.provably_equal Kind.int Kind.float);
          is_none (Kind.provably_equal Kind.span Kind.instant);
          is_none
            (Kind.provably_equal (Kind.list Kind.int) (Kind.list Kind.float));
          is_none
            (Kind.provably_equal (Kind.tensor Nx.float32)
               (Kind.tensor Nx.float64)));
      test "never proves the extension kind equal, even to itself" (fun () ->
          let k = Type.kind ext_type in
          is_none (Kind.provably_equal k k);
          is_none (Kind.provably_equal (Kind.list k) (Kind.list k)));
      cases
        ~name:(fun (_, s) -> Printf.sprintf "pp formats %s" s)
        "pp"
        [
          (str Kind.pp Kind.bool, "bool");
          (str Kind.pp Kind.int, "int");
          (str Kind.pp Kind.float, "float");
          (str Kind.pp Kind.string, "string");
          (str Kind.pp Kind.binary, "binary");
          (str Kind.pp Kind.date, "date");
          (str Kind.pp Kind.instant, "instant");
          (str Kind.pp Kind.span, "span");
          (str Kind.pp Record.kind, "record");
          (str Kind.pp (Kind.list Kind.float), "list[float]");
          (str Kind.pp (Kind.tensor Nx.float32), "tensor[float32]");
          (str Kind.pp (Type.kind ext_type), "ext");
        ]
        (fun (actual, expected) -> equal string expected actual);
    ]

(* Records *)

let ab = Record.(empty |> add Kind.int "a" (Some 1) |> add Kind.string "b" None)

let records =
  group "Record"
    [
      test "empty has no fields" (fun () ->
          equal (list string) [] (Record.names Record.empty));
      test "add appends a last field" (fun () ->
          equal (list string) [ "a"; "b" ] (Record.names ab));
      test "field reads a value and a null" (fun () ->
          equal (option int) (Some 1) (Record.field Kind.int "a" ab);
          equal (option string) None (Record.field Kind.string "b" ab));
      test "field reads a list field through the list kind" (fun () ->
          let r =
            Record.(add (Kind.list Kind.int) "l" (Some [| 1; 2 |]) empty)
          in
          equal
            (option (array int))
            (Some [| 1; 2 |])
            (Record.field (Kind.list Kind.int) "l" r);
          rejects (fun () -> Record.field (Kind.list Kind.float) "l" r));
      test "field refuses a missing name" (fun () ->
          rejects (fun () -> Record.field Kind.int "c" ab));
      test "field refuses another kind, even on a null" (fun () ->
          rejects (fun () -> Record.field Kind.float "a" ab);
          rejects (fun () -> Record.field Kind.int "b" ab));
      test "add refuses a duplicate name" (fun () ->
          rejects (fun () -> Record.add Kind.int "a" None ab));
      test "add refuses a name that is not UTF-8" (fun () ->
          rejects (fun () -> Record.add Kind.int "\xff" None Record.empty));
      test "add refuses the extension kind and lists of it" (fun () ->
          let k = Type.kind ext_type in
          rejects (fun () -> Record.add k "e" None Record.empty);
          rejects (fun () -> Record.add (Kind.list k) "e" None Record.empty));
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
    ]

(* Types *)

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
      test "datetime refuses an empty or non-UTF-8 zone" (fun () ->
          rejects (fun () -> Type.datetime ~zone:"" Type.S);
          rejects (fun () -> Type.datetime ~zone:"\xff" Type.S));
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
      test "extensions are equal only with equal names, metadata and storage"
        (fun () ->
          let x = Type.ext ~name:"x" ~metadata:"m" Type.float64 in
          is_false
            (Type.equal x (Type.ext ~name:"y" ~metadata:"m" Type.float64));
          is_false
            (Type.equal x (Type.ext ~name:"x" ~metadata:"n" Type.float64));
          is_false
            (Type.equal x (Type.ext ~name:"x" ~metadata:"m" Type.float32)));
      test "ext metadata defaults to empty" (fun () ->
          is_true
            (Type.equal
               (Type.ext ~name:"x" Type.int8)
               (Type.ext ~name:"x" ~metadata:"" Type.int8)));
    ]

let kind_of (Type.Any t) = str Kind.pp (Type.kind t)

let type_kind =
  cases
    ~name:(fun (t, k) -> Printf.sprintf "kind of %s is %s" (str pp_any t) k)
    "Type.kind"
    [
      (Any Type.bool, "bool");
      (Any Type.int8, "int");
      (Any Type.uint64, "int");
      (Any Type.float16, "float");
      (Any Type.float64, "float");
      (Any Type.string, "string");
      (Any (Type.categorical [| "a" |]), "string");
      (Any Type.binary, "binary");
      (Any Type.date, "date");
      (Any (Type.clock Type.Ms), "span");
      (Any (Type.duration Type.Ns), "span");
      (Any (Type.datetime ~zone:"UTC" Type.Us), "instant");
      (Any (Type.list (Type.list Type.int16)), "list[list[int]]");
      (Any (Type.record [ ("a", Any Type.bool) ]), "record");
      (Any (Type.tensor Nx.int32 [| 2 |]), "tensor[int32]");
      (Any ext_type, "ext");
    ]
    (fun (t, expected) -> equal string expected (kind_of t))

let float32_max = Int32.float_of_bits 0x7f7f_ffffl
let float32_mid = ldexp 1. 128 -. ldexp 1. 103

let int_bounds : (string * int Type.t * int * int) list =
  [
    ("int8", Type.int8, -128, 127);
    ("int16", Type.int16, -32768, 32767);
    ("int32", Type.int32, -2147483648, 2147483647);
    ("uint8", Type.uint8, 0, 255);
    ("uint16", Type.uint16, 0, 65535);
    ("uint32", Type.uint32, 0, 4294967295);
  ]

let holds_scalars =
  group "Type.holds scalars"
    [
      cases
        ~name:(fun (n, _, _, _) ->
          Printf.sprintf "%s holds exactly its range" n)
        "integers" int_bounds
        (fun (_, t, lo, hi) ->
          is_true (Type.holds t lo);
          is_true (Type.holds t hi);
          is_false (Type.holds t (lo - 1));
          is_false (Type.holds t (hi + 1)));
      test "int64 holds every int and uint64 every natural" (fun () ->
          is_true (Type.holds Type.int64 min_int);
          is_true (Type.holds Type.int64 max_int);
          is_true (Type.holds Type.uint64 max_int);
          is_true (Type.holds Type.uint64 0);
          is_false (Type.holds Type.uint64 (-1)));
      cases
        ~name:(fun (n, _, v, b) ->
          Printf.sprintf "%s %s %h" n (if b then "holds" else "does not hold") v)
        "floats"
        [
          ("float16", Type.float16, 65504., true);
          ("float16", Type.float16, 65519.99, true);
          ("float16", Type.float16, 65520., false);
          ("float16", Type.float16, -65520., false);
          ("float16", Type.float16, 0.1, true);
          ("float16", Type.float16, Float.nan, true);
          ("float16", Type.float16, Float.neg_infinity, true);
          ("float32", Type.float32, float32_max, true);
          ("float32", Type.float32, Float.pred float32_mid, true);
          ("float32", Type.float32, float32_mid, false);
          ("float32", Type.float32, -.Float.max_float, false);
          ("float32", Type.float32, Float.infinity, true);
          ("float64", Type.float64, Float.max_float, true);
        ]
        (fun (_, t, v, expected) -> equal bool expected (Type.holds t v));
      test "strings must be UTF-8 and categories in the dictionary" (fun () ->
          is_true (Type.holds Type.string "é");
          is_false (Type.holds Type.string "\xe9");
          let c = Type.categorical [| "AA"; "B6" |] in
          is_true (Type.holds c "B6");
          is_false (Type.holds c "C"));
      test "booleans, byte strings and dates are always held" (fun () ->
          is_true (Type.holds Type.bool false);
          is_true (Type.holds Type.binary (Binary.of_string "\xff"));
          is_true (Type.holds Type.date (date min_days)));
      test "datetimes hold whole units" (fun () ->
          let t = Type.datetime ~zone:"UTC" Type.S in
          is_true (Type.holds t (Option.get (Time.of_s (-5L))));
          is_false (Type.holds t (Time.of_ns 5_000_000_001L));
          is_true
            (Type.holds (Type.datetime Type.Ns) (Time.of_ns Int64.max_int)));
      test "durations hold whole units of either sign" (fun () ->
          is_true (Type.holds (Type.duration Type.Us) (Time.Span.us (-3)));
          is_false (Type.holds (Type.duration Type.Ms) (Time.Span.us 1500)));
      test "clocks hold whole units in one day" (fun () ->
          let t = Type.clock Type.Ms in
          is_true (Type.holds t (Time.Span.ns 0));
          is_true (Type.holds t (Time.Span.of_ns 86_399_999_000_000L));
          is_false (Type.holds t (Time.Span.days 1));
          is_false (Type.holds t (Time.Span.ms (-1)));
          is_false (Type.holds t (Time.Span.us 1)));
    ]

let tensor2 dt a = Nx.create dt [| Array.length a |] a

let holds_structures =
  let rt = Type.record [ ("a", Any Type.int8); ("b", Any Type.string) ] in
  group "Type.holds structures"
    [
      test "lists hold when every element is held" (fun () ->
          is_true (Type.holds (Type.list Type.uint8) [| 0; 255 |]);
          is_true (Type.holds (Type.list Type.uint8) [||]);
          is_false (Type.holds (Type.list Type.uint8) [| 0; 256 |]));
      test "records hold their fields' values in order" (fun () ->
          let r a b =
            Record.(empty |> add Kind.int "a" a |> add Kind.string "b" b)
          in
          is_true (Type.holds rt (r (Some 3) (Some "x")));
          is_true (Type.holds rt (r None None));
          is_false (Type.holds rt (r (Some 300) None));
          is_false (Type.holds rt (r None (Some "\xff"))));
      test "records need the type's names in order" (fun () ->
          let ba =
            Record.(empty |> add Kind.string "b" None |> add Kind.int "a" None)
          in
          is_false (Type.holds rt ba);
          is_false (Type.holds rt Record.(add Kind.int "a" None empty)));
      test "a non-null field must have its type's kind" (fun () ->
          let r =
            Record.(
              empty |> add Kind.float "a" (Some 1.) |> add Kind.string "b" None)
          in
          is_false (Type.holds rt r));
      test "a null field is held whatever its kind" (fun () ->
          let r =
            Record.(
              empty |> add Kind.float "a" None |> add Kind.string "b" None)
          in
          is_true (Type.holds rt r));
      test "a null extension field is held" (fun () ->
          let t = Type.record [ ("m", Any ext_type) ] in
          is_true (Type.holds t Record.(add Kind.float "m" None empty)));
      test "a plain value never stands for an extension field" (fun () ->
          let t = Type.record [ ("m", Any ext_type) ] in
          let r = Record.(add Kind.float "m" (Some 1.) empty) in
          is_false (Type.holds t r);
          rejects (fun () -> Type.compare_value t r r));
      test "tensors hold the type's shape" (fun () ->
          let t = Type.tensor Nx.float32 [| 2 |] in
          is_true (Type.holds t (tensor2 Nx.float32 [| 1.; 2. |]));
          is_false (Type.holds t (tensor2 Nx.float32 [| 1.; 2.; 3. |])));
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
            1.;
            -1.;
          ] );
    ]

let int_array = Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 2)

let optional g =
  Gen.frequency [ (1, Gen.constant None); (3, Gen.map Option.some g) ]

let record_type = Type.record [ ("a", Any Type.int64); ("b", Any Type.string) ]

let record_gen =
  Gen.map
    (fun (a, b) ->
      Record.(empty |> add Kind.int "a" a |> add Kind.string "b" b))
    (Gen.pair
       (optional (Gen.int_range 0 2))
       (optional (Gen.of_list [ "a"; "b" ])))
  |> Gen.with_pp (fun ppf r ->
      Format.fprintf ppf "{a=%a; b=%a}"
        (Format.pp_print_option
           ~none:(fun ppf () -> Format.pp_print_string ppf "null")
           Format.pp_print_int)
        (Record.field Kind.int "a" r)
        (Format.pp_print_option
           ~none:(fun ppf () -> Format.pp_print_string ppf "null")
           Format.pp_print_string)
        (Record.field Kind.string "b" r))

let record_w =
  let fields r =
    (Record.field Kind.int "a" r, Record.field Kind.string "b" r)
  in
  Testable.make
    ~pp:(fun ppf r ->
      Format.pp_print_string ppf (String.concat "," (Record.names r)))
    ~equal:(fun r0 r1 -> fields r0 = fields r1)
  |> Testable.with_compare (Type.compare_value record_type)

let tensor_type = Type.tensor Nx.float32 [| 2 |]

let tensor_gen =
  Gen.map
    (fun (a, b) -> tensor2 Nx.float32 [| a; b |])
    (Gen.pair
       (Gen.of_list [ 0.; -0.; 1.; Float.nan ])
       (Gen.of_list [ 0.; 1.; Float.nan ]))
  |> Gen.with_pp (fun ppf t ->
      Format.fprintf ppf "[%h; %h]" (Nx.item [ 0 ] t) (Nx.item [ 1 ] t))

let tensor_w =
  Testable.make
    ~pp:(fun ppf t ->
      Format.fprintf ppf "[%h; %h]" (Nx.item [ 0 ] t) (Nx.item [ 1 ] t))
    ~equal:(fun t0 t1 ->
      Array.for_all2 same_float (Nx.to_array t0) (Nx.to_array t1))
  |> Testable.with_compare (Type.compare_value tensor_type)

let sign c = Int.compare c 0

let null_last cmp a b =
  match (a, b) with
  | None, None -> 0
  | None, Some _ -> 1
  | Some _, None -> -1
  | Some a, Some b -> cmp a b

let compare_values =
  group "Type.compare_value"
    [
      prop "floats are totally ordered, -0 equal to 0 and NaN equal to NaN"
        (Gen.triple special_float special_float special_float)
        (Law.order (float_w Type.float64));
      test "NaN comes after infinity and -0 equals 0" (fun () ->
          let c = Type.compare_value Type.float32 in
          greater int ~than:0 (c Float.nan Float.infinity);
          less int ~than:0 (c Float.neg_infinity (-.Float.max_float));
          equal int 0 (c (-0.) 0.);
          equal int 0 (c Float.nan (-.Float.nan)));
      prop "lists order lexicographically, a prefix first"
        (Gen.triple int_array int_array int_array)
        (Law.order
           (Testable.with_compare
              (Type.compare_value (Type.list Type.int8))
              (array int)));
      prop "lists order as Stdlib orders lists" (Gen.pair int_array int_array)
        (fun (a0, a1) ->
          equal int
            (sign
               (List.compare Int.compare (Array.to_list a0) (Array.to_list a1)))
            (sign (Type.compare_value (Type.list Type.int8) a0 a1)));
      test "a proper prefix comes first" (fun () ->
          less int ~than:0
            (Type.compare_value (Type.list Type.int8) [| 1 |] [| 1; 0 |]);
          greater int ~than:0
            (Type.compare_value (Type.list Type.int8) [| 2 |] [| 1; 9 |]));
      prop "records order by their fields, nulls last"
        (Gen.triple record_gen record_gen record_gen)
        (Law.order record_w);
      prop "records order by a, then b, in the type's order, nulls last"
        (Gen.pair record_gen record_gen) (fun (r0, r1) ->
          let a = Record.field Kind.int "a"
          and b = Record.field Kind.string "b" in
          let expected =
            match null_last Int.compare (a r0) (a r1) with
            | 0 -> null_last String.compare (b r0) (b r1)
            | c -> c
          in
          equal int (sign expected)
            (sign (Type.compare_value record_type r0 r1)));
      test "a null field comes after every value" (fun () ->
          let r a =
            Record.(empty |> add Kind.int "a" a |> add Kind.string "b" None)
          in
          greater int ~than:0
            (Type.compare_value record_type (r None) (r (Some max_int)));
          equal int 0 (Type.compare_value record_type (r None) (r None)));
      prop "tensors order by their elements in row-major order"
        (Gen.triple tensor_gen tensor_gen tensor_gen)
        (Law.order tensor_w);
      prop "tensors order as the lists of their elements, NaN last, -0 as 0"
        (Gen.pair tensor_gen tensor_gen) (fun (t0, t1) ->
          let key x = if Float.is_nan x then (1, 0.) else (0, x +. 0.) in
          let elements t = List.map key (Array.to_list (Nx.to_array t)) in
          equal int
            (sign (compare (elements t0) (elements t1)))
            (sign (Type.compare_value tensor_type t0 t1)));
      test "tensors of unsigned integers order as unsigned" (fun () ->
          let t = Type.tensor Nx.uint32 [| 1 |] in
          let v x = Nx.create Nx.uint32 [| 1 |] [| x |] in
          greater int ~than:0 (Type.compare_value t (v (-1l)) (v 1l)));
      test "complex tensors order by real then imaginary part" (fun () ->
          let t = Type.tensor Nx.complex64 [| 1 |] in
          let v re im =
            Nx.create Nx.complex64 [| 1 |] [| { Complex.re; im } |]
          in
          less int ~than:0 (Type.compare_value t (v 1. 5.) (v 2. 0.));
          less int ~than:0 (Type.compare_value t (v 1. 0.) (v 1. Float.nan)));
      test "categoricals order by the dictionary" (fun () ->
          let t = Type.categorical [| "b"; "a" |] in
          less int ~than:0 (Type.compare_value t "b" "a"));
      test "strings and byte strings order by unsigned bytes" (fun () ->
          less int ~than:0 (Type.compare_value Type.string "z" "é");
          less int ~than:0
            (Type.compare_value Type.binary (Binary.of_string "\x7f")
               (Binary.of_string "\x80")));
      test "false comes before true" (fun () ->
          less int ~than:0 (Type.compare_value Type.bool false true));
      test "integers, dates, spans and instants order by value" (fun () ->
          List.iter
            (fun t -> less int ~than:0 (Type.compare_value t (-1) 1))
            [
              Type.int8;
              Type.int16;
              Type.int32;
              Type.int64;
              Type.uint8;
              Type.uint16;
              Type.uint32;
              Type.uint64;
            ];
          less int ~than:0 (Type.compare_value Type.float16 1. Float.nan);
          less int ~than:0 (Type.compare_value Type.date (date (-1)) (date 0));
          List.iter
            (fun t ->
              less int ~than:0
                (Type.compare_value t (Time.Span.ns (-1)) (Time.Span.ns 0)))
            [ Type.clock Type.Ns; Type.duration Type.Ns ];
          less int ~than:0
            (Type.compare_value (Type.datetime Type.Ns) (Time.of_ns (-1L))
               (Time.of_ns 0L)));
      cases
        ~name:(fun (n, _) -> Printf.sprintf "tensors of %s order by value" n)
        "tensor dtypes"
        [
          ( "int64",
            fun () ->
              let v x = Nx.create Nx.int64 [| 1 |] [| x |] in
              Type.compare_value (Type.tensor Nx.int64 [| 1 |]) (v (-1L)) (v 1L)
          );
          ( "uint64",
            fun () ->
              let v x = Nx.create Nx.uint64 [| 1 |] [| x |] in
              -Type.compare_value
                 (Type.tensor Nx.uint64 [| 1 |])
                 (v (-1L)) (v 1L) );
          ( "int8",
            fun () ->
              let v x = Nx.create Nx.int8 [| 1 |] [| x |] in
              Type.compare_value (Type.tensor Nx.int8 [| 1 |]) (v (-1)) (v 1) );
          ( "float64",
            fun () ->
              let v x = Nx.create Nx.float64 [| 1 |] [| x |] in
              Type.compare_value
                (Type.tensor Nx.float64 [| 1 |])
                (v 1.) (v Float.nan) );
          ( "bool",
            fun () ->
              let v x = Nx.create Nx.bool [| 1 |] [| x |] in
              Type.compare_value
                (Type.tensor Nx.bool [| 1 |])
                (v false) (v true) );
        ]
        (fun (_, c) -> less int ~than:0 (c ()));
      test "a null list-of-extension field is held and orders as null"
        (fun () ->
          let t = Type.record [ ("l", Any (Type.list ext_type)) ] in
          let r = Record.(add (Kind.list Kind.float) "l" None empty) in
          is_true (Type.holds t r);
          equal int 0 (Type.compare_value t r r));
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
      test "refuses a tensor of another shape" (fun () ->
          let v = tensor2 Nx.float32 [| 1. |] in
          rejects (fun () -> Type.compare_value tensor_type v v));
    ]

(* Operands *)

let int_types =
  [
    Type.int8;
    Type.int16;
    Type.int32;
    Type.int64;
    Type.uint8;
    Type.uint16;
    Type.uint32;
    Type.uint64;
  ]

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

let int_type_gen = Gen.of_list ~pp:Type.pp int_types

let common_cases =
  let some t = Some (Type.Any t) and none = None in
  let common ts = Option.map (fun t -> Type.Any t) (Type.common ts) in
  let cat d = Type.categorical d in
  group "Type.common"
    [
      test "nothing meets in an empty list" (fun () ->
          equal (option any_w) none (common ([] : int Type.t list)));
      test "a type meets itself" (fun () ->
          equal (option any_w) (some Type.date)
            (common [ Type.date; Type.date ]));
      test "int8 and uint8 do not meet, but meet with int16" (fun () ->
          equal (option any_w) none (common [ Type.int8; Type.uint8 ]);
          equal (option any_w) (some Type.int16)
            (common [ Type.int8; Type.uint8; Type.int16 ]);
          equal (option any_w) (some Type.int16)
            (common [ Type.int16; Type.int8; Type.uint8 ]));
      test "uint8 meets int16 and uint32 meets int64" (fun () ->
          equal (option any_w) (some Type.int16)
            (common [ Type.uint8; Type.int16 ]);
          equal (option any_w) (some Type.int64)
            (common [ Type.uint32; Type.int64 ]);
          equal (option any_w) none (common [ Type.uint64; Type.int64 ]));
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
            (common [ Type.datetime Type.S; Type.datetime Type.Ms ]);
          equal (option any_w) none
            (common [ Type.datetime ~zone:"UTC" Type.S; Type.datetime Type.S ]));
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
          let r = Type.record [ ("m", Any ext_type) ] in
          equal (option any_w) (some r) (common [ r; r ]));
      test "extensions and tensors meet only when equal" (fun () ->
          equal (option any_w) (some ext_type) (common [ ext_type; ext_type ]);
          equal (option any_w) none
            (common [ ext_type; Type.ext ~name:"units.mass" Type.float64 ]);
          equal (option any_w) none
            (common
               [
                 Type.tensor Nx.float32 [| 2 |]; Type.tensor Nx.float32 [| 3 |];
               ]));
      prop "does not depend on the order of the types"
        (Gen.bind
           (Gen.list ~size:(Gen.int_range 0 5) int_type_gen)
           (fun ts ->
             Gen.map (fun p -> (ts, p)) (Gen.permutation ~pp:Type.pp ts)))
        (fun (ts, p) ->
          cover "meets" (Option.is_some (Type.common ts));
          cover "does not meet" (ts <> [] && Option.is_none (Type.common ts));
          equal (option any_w) (common ts) (common p));
      prop "integer types meet at the one whose range holds the others"
        (Gen.list ~size:(Gen.int_range 1 5) int_type_gen)
        (fun ts ->
          equal (option any_w)
            (Option.map (fun t -> Type.Any t) (reference_common ts))
            (common ts));
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
      Any (Type.datetime Type.Us);
      Any (Type.datetime ~zone:"UTC" Type.Us);
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
      test "compares dictionaries, zones and fields in order" (fun () ->
          is_false
            (Type.equal
               (Type.categorical [| "a"; "b" |])
               (Type.categorical [| "b"; "a" |]));
          is_false
            (Type.equal
               (Type.datetime ~zone:"UTC" Type.S)
               (Type.datetime ~zone:"utc" Type.S));
          is_false
            (Type.equal
               (Type.record [ ("a", Any Type.int8); ("b", Any Type.int8) ])
               (Type.record [ ("b", Any Type.int8); ("a", Any Type.int8) ])));
      test "types of different kinds are not equal" (fun () ->
          is_false (Type.equal Type.int8 Type.float32));
    ]

let type_pp =
  let alphabet = Array.init 26 (fun i -> String.make 1 (Char.chr (97 + i))) in
  cases
    ~name:(fun (_, s) -> Printf.sprintf "pp formats %s" s)
    "Type.pp"
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
      (Any (Type.datetime Type.Us), "datetime[us]");
      (Any (Type.datetime Type.S), "datetime[s]");
      (Any (Type.datetime ~zone:"UTC" Type.Us), "datetime[us, UTC]");
      (Any (Type.datetime ~zone:"a]b" Type.Us), {|datetime[us, "a]b"]|});
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

(* Schemas *)

let columns = [ ("carrier", Type.Any Type.string); ("delay", Any Type.float64) ]
let schema_w = Testable.make ~pp:Schema.pp ~equal:Schema.equal

let pp_change ppf = function
  | Schema.Added (n, t) -> Format.fprintf ppf "+%s %a" n pp_any t
  | Removed (n, t) -> Format.fprintf ppf "-%s %a" n pp_any t
  | Retyped (n, t0, t1) ->
      Format.fprintf ppf "~%s %a -> %a" n pp_any t0 pp_any t1

let diff s0 s1 =
  List.map (str pp_change) (Schema.diff (Schema.v s0) (Schema.v s1))

let schemas =
  group "Schema"
    [
      test "v refuses a duplicate or non-UTF-8 name" (fun () ->
          rejects (fun () ->
              Schema.v [ ("a", Any Type.bool); ("a", Any Type.bool) ]);
          rejects (fun () -> Schema.v [ ("\xff", Any Type.bool) ]));
      test "columns gives back the columns in order" (fun () ->
          equal
            (list (pair string any_w))
            columns
            (Schema.columns (Schema.v columns)));
      test "find looks a column up by name" (fun () ->
          let s = Schema.v columns in
          equal (option any_w) (Some (Any Type.float64)) (Schema.find s "delay");
          equal (option any_w) None (Schema.find s "origin"));
      test "equal needs the same names and types" (fun () ->
          let s cs = Schema.v cs in
          is_false
            (Schema.equal
               (s [ ("a", Any Type.int8) ])
               (s [ ("a", Any Type.int16) ]));
          is_false
            (Schema.equal
               (s [ ("a", Any Type.int8) ])
               (s [ ("b", Any Type.int8) ])));
      test "equal needs the same order" (fun () ->
          equal schema_w (Schema.v columns) (Schema.v columns);
          is_false
            (Schema.equal (Schema.v columns) (Schema.v (List.rev columns))));
      test "diff ignores order" (fun () ->
          equal (list string) [] (diff columns (List.rev columns)));
      test "diff lists removed and retyped columns, then added ones" (fun () ->
          equal (list string)
            [
              "-carrier string";
              "~delay float64 -> float32";
              "+origin string";
              "+dest string";
            ]
            (diff columns
               [
                 ("origin", Any Type.string);
                 ("delay", Any Type.float32);
                 ("dest", Any Type.string);
               ]));
      cases
        ~name:(fun (_, s) -> Printf.sprintf "pp formats %S" s)
        "pp"
        [
          (columns, "carrier string, delay float64");
          ([], "");
          ( [ ("a b", Any Type.int8); ("", Any Type.bool) ],
            {|"a b" int8, "" bool|} );
        ]
        (fun (cs, expected) ->
          equal string expected (str Schema.pp (Schema.v cs)));
    ]

(* Errors *)

let message f =
  match f () with () -> "no exception" | exception Invalid_argument m -> m

let errors =
  let ext_record = Type.record [ ("m", Any ext_type) ] in
  group "errors"
    [
      test "Invalid_argument messages" (fun () ->
          let messages =
            List.map message
              [
                (fun () -> ignore (Type.categorical [| "a"; "a" |]));
                (fun () -> ignore (Type.categorical [| "\xc3" |]));
                (fun () -> ignore (Type.datetime ~zone:"" Type.S));
                (fun () -> ignore (Type.datetime ~zone:"\xff" Type.S));
                (fun () ->
                  ignore
                    (Type.record [ ("a", Any Type.bool); ("a", Any Type.int8) ]));
                (fun () -> ignore (Type.record [ ("\xff", Any Type.bool) ]));
                (fun () -> ignore (Type.tensor Nx.float32 [||]));
                (fun () -> ignore (Type.tensor Nx.float32 [| 2; -1 |]));
                (fun () -> ignore (Type.ext ~name:"" Type.float64));
                (fun () -> ignore (Type.ext ~name:"\xff" Type.float64));
                (fun () -> ignore (Type.ext ~name:"a b" ext_type));
                (fun () -> ignore (Record.add Kind.int "a" None ab));
                (fun () ->
                  ignore (Record.add Kind.int "\xff" None Record.empty));
                (fun () ->
                  ignore
                    (Record.add
                       (Kind.list (Type.kind ext_type))
                       "e" None Record.empty));
                (fun () -> ignore (Record.field Kind.int "c" ab));
                (fun () -> ignore (Record.field Kind.float "a" ab));
                (fun () ->
                  ignore
                    (Schema.v [ ("a", Any Type.bool); ("a", Any Type.bool) ]));
                (fun () -> ignore (Schema.v [ ("\xff", Any Type.bool) ]));
                (fun () -> ignore (Time.Span.hours max_int));
                (fun () ->
                  ignore
                    (Type.compare_value (Type.categorical [| "a" |]) "a" "b"));
                (fun () ->
                  let r = Record.(add Kind.int "a" None empty) in
                  ignore (Type.compare_value record_type r r));
                (fun () ->
                  ignore (Type.compare_value (Type.record []) ab Record.empty));
                (fun () ->
                  let r =
                    Record.(
                      empty
                      |> add Kind.float "a" (Some 1.)
                      |> add Kind.string "b" None)
                  in
                  ignore (Type.compare_value record_type r r));
                (fun () ->
                  let r = Record.(add Kind.float "m" (Some 1.) empty) in
                  ignore (Type.compare_value ext_record r r));
                (fun () ->
                  let v = tensor2 Nx.float32 [| 1. |] in
                  ignore (Type.compare_value tensor_type v v));
              ]
          in
          expect (String.concat "\n" messages)
          @@ __POS_OF__
               {|
            Type.categorical: "a" appears twice
            Type.categorical: "\195" is not UTF-8
            Type.datetime: empty zone
            Type.datetime: zone "\255" is not UTF-8
            Type.record: duplicate field "a"
            Type.record: field name "\255" is not UTF-8
            Type.tensor: empty shape
            Type.tensor: dimension -1 is negative
            Type.ext: empty name
            Type.ext: name "\255" is not UTF-8
            Type.ext: "a b" is stored as an extension type
            Record.add: duplicate field "a"
            Record.add: field name "\255" is not UTF-8
            Record.add: field "e" has the kind list[ext]
            Record.field: no field "c"
            Record.field: field "a" is int, not float
            Schema.v: duplicate column "a"
            Schema.v: column name "\255" is not UTF-8
            Time.Span.hours: 4611686018427387903 is out of range
            Type.compare_value: "b" is not in the dictionary
            Type.compare_value: the record's fields are not [a, b]
            Type.compare_value: the record's fields are not []
            Type.compare_value: field "a" is float, not int
            Type.compare_value: field "m" has an extension type, which no plain value holds
            Type.compare_value: a tensor does not have the type's shape
            |});
    ]

let () =
  exit
    (run "types"
       [
         binary;
         instant_conversions;
         instant_pp;
         span_constructors;
         span_pp;
         step_pp;
         dates;
         kinds;
         records;
         type_constructors;
         type_kind;
         holds_scalars;
         holds_structures;
         compare_values;
         common_cases;
         type_equal;
         type_pp;
         schemas;
         errors;
       ])
