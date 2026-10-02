(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module G = Talon_gen

let show ?(limits = Talon.limits) t = Format.asprintf "%a" (pp_with limits) t
let int8s n = Column.v Type.int8 (Array.init n Fun.id)
let day ymd = Option.get (Time.Date.of_civil ymd)
let bytes xs = Array.map (Option.map Binary.of_string) xs

let message f =
  match f () with _ -> fail "no exception" | exception Invalid_argument m -> m

(* Types *)

let numbers =
  v
    [
      ("bool", Column.of_options Type.bool [| Some true; Some false; None |]);
      ("int8", Column.of_options Type.int8 [| Some (-128); Some 127; None |]);
      ( "int64",
        Column.of_options Type.int64 [| Some min_int; Some max_int; None |] );
      ( "wide",
        Column.of_tensor
          (Nx.create Nx.int64 [| 3 |] [| Int64.min_int; Int64.max_int; 0L |]) );
      ( "uint64",
        Column.of_tensor (Nx.create Nx.uint64 [| 3 |] [| -1L; 0L; 1L |]) );
    ]

let texts =
  v
    [
      ( "string",
        Column.of_options Type.string
          [|
            Some "naïve";
            Some "日本語";
            Some "";
            Some "a\tb\n";
            Some "\"q\" \\";
            None;
          |] );
      ( "binary",
        Column.of_options Type.binary
          (bytes
             [|
               Some "\xff\x00ab"; Some "é"; None; Some ""; Some "x"; Some "\x7f";
             |]) );
      ( "category",
        Column.of_options
          (Type.categorical [| "low"; "high" |])
          [|
            Some "low"; Some "high"; None; Some "low"; Some "low"; Some "high";
          |] );
    ]

let temporal =
  let span ns = Some (Time.Span.of_ns ns) in
  v
    [
      ( "date",
        Column.of_options Type.date
          [|
            Some (day (2024, 3, 15));
            Some (day (-44, 3, 15));
            Some (day (12345, 1, 1));
            None;
          |] );
      ( "clock",
        Column.of_options (Type.clock Ms)
          [| span 34_200_000_000_000L; span 0L; span 1_500_000_000L; None |] );
      ( "duration",
        Column.of_options (Type.duration Ns)
          [| span (-250_000L); span 5_400_000_000_000L; span 0L; None |] );
      ( "datetime",
        Column.of_options (Type.datetime Ns)
          [|
            Some (Time.of_ns 1_500_000_000L);
            Some (Time.of_ns (-1L));
            Some (Time.of_ns 1_710_495_000_000_000_000L);
            None;
          |] );
      ( "zoned",
        Column.of_options
          (Type.datetime ~zone:"UTC" Ms)
          [|
            Some (Time.of_ns 1_710_495_000_500_000_000L);
            Some (Time.of_ns 0L);
            None;
            Some (Time.of_ns (-86_400_000_000_000L));
          |] );
    ]

let compound =
  let record x =
    Some Record.(empty |> add Kind.int "x" x |> add Kind.string "s" (Some "a"))
  in
  let uuid = Type.ext ~name:"uuid" Type.string in
  let storage = Column.layout (Column.v Type.string [| "0f"; "1e" |]) in
  v
    [
      ( "list",
        Column.of_options (Type.list Type.string)
          [| Some [| "a"; "b" |]; Some [||] |] );
      ( "record",
        Column.of_options
          (Type.record [ ("x", Any Type.int8); ("s", Any Type.string) ])
          [| record (Some 1); record None |] );
      ("tensor", Column.of_tensor (Nx.zeros Nx.float32 [| 2; 3 |]));
      ("uuid", Result.get_ok (Column.of_layout (Any uuid) storage));
    ]

let floats =
  let f64 xs = Column.of_options Type.float64 xs in
  v
    [
      ("mean", f64 [| Some 58.; Some 53.4213567; Some 29.; None; Some 0. |]);
      ("fraction", f64 [| Some 0.125; Some 2.; Some 1e5; None; Some (-1.) |]);
      ("tiny", f64 [| Some 1e-9; Some 1.; Some 0.; None; Some 2. |]);
      ("huge", f64 [| Some 1e16; Some (-1.); Some 0.; None; Some 3. |]);
      ( "special",
        f64
          [|
            Some Float.nan;
            Some Float.infinity;
            Some Float.neg_infinity;
            None;
            Some (-0.);
          |] );
      ( "float32",
        Column.of_options Type.float32
          [| Some 0.1; Some 100.; None; None; Some 1. |] );
      ( "float16",
        Column.of_options Type.float16
          [| Some 65504.; Some 0.5; None; None; Some 1. |] );
    ]

let types =
  group "Types"
    [
      test "numbers align right, stored values outside int in full" (fun () ->
          expect (show numbers)
          @@ __POS_OF__
               {|
            table 3 rows × 5 columns
             bool   int8  int64                 wide                  uint64
             bool   int8  int64                 int64                 uint64
             true   -128  -4611686018427387904  -9223372036854775808  18446744073709551615
             false   127   4611686018427387903   9223372036854775807                     0
             ∅         ∅                     ∅                     0                     1
            |});
      test "text shows as it reads, controls and non-UTF-8 bytes escaped"
        (fun () ->
          expect (show texts)
          @@ __POS_OF__
               {|
          table 6 rows × 3 columns
           string      binary      category
           string      binary      categorical["low", "high"]
           naïve       \xff\x00ab  low
           日本語         é           high
                       ∅           ∅
           a\x09b\x0a              low
           "q" \       x           low
           ∅           \x7f        high
          |});
      test "temporal values show in their text form" (fun () ->
          expect (show temporal)
          @@ __POS_OF__
               {|
            table 4 rows × 5 columns
             date          clock      duration      datetime                       zoned
             date          clock[ms]  duration[ns]  datetime[ns]                   datetime[ms, UTC]
             2024-03-15    9h30m      -250us        1970-01-01T00:00:01.5          2024-03-15T09:30:00.5Z
             -0044-03-15   0s         1h30m         1969-12-31T23:59:59.999999999  1970-01-01T00:00:00Z
             +12345-01-01  1s500ms    0s            2024-03-15T09:30:00            ∅
             ∅             ∅          ∅             ∅                              1969-12-31T00:00:00Z
            |});
      test "lists, records, tensors and extensions" (fun () ->
          expect (show compound)
          @@ __POS_OF__
               {|
            table 2 rows × 4 columns
             list          record                    tensor              uuid
             list[string]  record[x int8, s string]  tensor[float32, 3]  ext[uuid, string]
             ["a"; "b"]    <record>                  <tensor>            0f
             []            <record>                  <tensor>            1e
            |});
      test "the floats of a column share their decimals" (fun () ->
          expect (show floats)
          @@ __POS_OF__
               {|
            table 5 rows × 7 columns
             mean     fraction       tiny         huge          special  float32     float16
             float64  float64        float64      float64       float64  float32     float16
             58.0000       0.125000  1.00000e-09   1.00000e+16      nan    0.100000  65504.000000
             53.4214       2.000000  1.00000e+00  -1.00000e+00      inf  100.000000      0.500000
             29.0000  100000.000000  0.00000e+00   0.00000e+00     -inf           ∅             ∅
                   ∅              ∅            ∅             ∅        ∅           ∅             ∅
              0.0000      -1.000000  2.00000e+00   3.00000e+00       -0    1.000000      1.000000
            |});
    ]

(* [float_cell x] is the cell display shows for [x] alone in a column. *)
let float_cell x =
  let t = v [ ("x", Column.v Type.float64 [| x |]) ] in
  String.trim (List.nth (String.split_on_char '\n' (show t)) 3)

(* A value below 0.1 in magnitude, once rounded to six significant digits, needs
   more than six decimals, and one of 10^16 or more has integer digits that are
   not all its own: either shows its column in scientific notation. *)
let float_cuts =
  cases
    ~name:(fun (x, _) -> Printf.sprintf "%.17g" x)
    "Scientific cut"
    [
      (0.1, "0.100000");
      (0.099, "9.90000e-02");
      (-0.099, "-9.90000e-02");
      (0.09999996, "0.100000");
      (99.99996, "100.000");
      (123456., "123456");
      (1234567., "1234567");
      (9999999999999998., "9999999999999998");
      (1e16, "1.00000e+16");
      (-1e16, "-1.00000e+16");
      (1e-300, "1.00000e-300");
      (Float.min_float /. 2., "1.11254e-308");
    ]
    (fun (x, expected) -> equal string expected (float_cell x))

(* Elision *)

let rows_limits = { Talon.limits with head = 2; tail = 1 }

let elision =
  group "Elision"
    [
      test "past head + tail rows, the first and the last" (fun () ->
          expect (show (v [ ("i", int8s 12) ]))
          @@ __POS_OF__
               {|
            table 12 rows × 1 column
             i
             int8
                0
                1
                2
                3
                4
             ⋮
                7
                8
                9
               10
               11
             2 rows not shown
            |});
      test "head + tail rows show in full" (fun () ->
          expect (show ~limits:rows_limits (v [ ("i", int8s 3) ]))
          @@ __POS_OF__
               {|
            table 3 rows × 1 column
             i
             int8
                0
                1
                2
            |});
      test "one row past head + tail" (fun () ->
          expect (show ~limits:rows_limits (v [ ("i", int8s 4) ]))
          @@ __POS_OF__
               {|
            table 4 rows × 1 column
             i
             int8
                0
                1
             ⋮
                3
             1 row not shown
            |});
      test "no head and no tail" (fun () ->
          let limits = { rows_limits with head = 0; tail = 0 } in
          expect (show ~limits (v [ ("i", int8s 2) ]))
          @@ __POS_OF__
               {|
            table 2 rows × 1 column
             i
             int8
             ⋮
             2 rows not shown
            |});
      test "past columns columns, a line naming the others" (fun () ->
          let name i = String.make 1 (Char.chr (Char.code 'a' + i)) in
          let t = v (List.init 14 (fun i -> (name i, int8s 1))) in
          expect (show t)
          @@ __POS_OF__
               {|
            table 1 row × 14 columns
             a     b     c     d     e     f     g     h     i     j     k     l
             int8  int8  int8  int8  int8  int8  int8  int8  int8  int8  int8  int8
                0     0     0     0     0     0     0     0     0     0     0     0
             2 columns not shown: m, n
            |});
      test "no column shown" (fun () ->
          let limits = { Talon.limits with columns = 0 } in
          expect (show ~limits (v [ ("a", int8s 2); ("b c", int8s 2) ]))
          @@ __POS_OF__
               {|
            table 2 rows × 2 columns
             2 columns not shown: a, "b c"
            |});
      test "cells, names and types cut at width, an escape whole" (fun () ->
          let limits = { Talon.limits with width = 4 } in
          let t =
            v
              [
                ( "a_long_name",
                  Column.v Type.string [| "abcdefgh"; "ab\x01cd"; "abcd" |] );
                ( "n",
                  Column.v (Type.list Type.int8) [| [| 1; 2 |]; [||]; [| 3 |] |]
                );
              ]
          in
          expect (show ~limits t)
          @@ __POS_OF__
               {|
            table 3 rows × 2 columns
             a_l…  n
             str…  lis…
             abc…  [1;…
             ab…   []
             abcd  [3]
            |});
      test "a table without columns or rows" (fun () ->
          expect (show (v ~rows:0 []))
          @@ __POS_OF__ {| table 0 rows × 0 columns |});
      test "a column without rows" (fun () ->
          expect (show (v [ ("", int8s 0) ]))
          @@ __POS_OF__
               {|
            table 0 rows × 1 column
             ""
             int8
            |});
      test "one row and one column" (fun () ->
          expect (show (v [ ("x", int8s 1) ]))
          @@ __POS_OF__
               {|
            table 1 row × 1 column
             x
             int8
                0
            |});
    ]

(* Limits *)

let limits_cases =
  let refuse name limits expected =
    test name (fun () ->
        expect (message (fun () -> show ~limits numbers)) expected)
  in
  let l = Talon.limits in
  group "Limits"
    [
      test "limits are five rows each end, twelve columns, 32 wide" (fun () ->
          equal (list int) [ 5; 5; 12; 32 ]
            [ l.head; l.tail; l.columns; l.width ]);
      test "pp is pp_with limits" (fun () ->
          let t = v [ ("i", int8s 40) ] in
          equal text (show t) (Format.asprintf "%a" pp t));
      refuse "a negative head" { l with head = -1 }
      @@ __POS_OF__ {| Talon.pp_with: head is -1, negative |};
      refuse "a negative tail" { l with tail = -1 }
      @@ __POS_OF__ {| Talon.pp_with: tail is -1, negative |};
      refuse "negative columns" { l with columns = -2 }
      @@ __POS_OF__ {| Talon.pp_with: columns is -2, negative |};
      refuse "a width of zero" { l with width = 0 }
      @@ __POS_OF__ {| Talon.pp_with: width is 0, not positive |};
    ]

(* Laws *)

let blind_to_batching =
  let limits = { Talon.limits with head = 2; tail = 2 } in
  let gen =
    Gen.bind G.sample (fun (G.Sample (ty, vs)) ->
        let t = v [ ("a", Column.of_options ty vs) ] in
        Gen.pair
          (Gen.constant
             ~pp:(fun ppf t -> Format.pp_print_string ppf (show t))
             t)
          (G.split t))
  in
  prop "display is blind to batching" gen (fun (t, s) ->
      cover "elided" (rows t > 4);
      cover "several batches" (List.length (batches s) > 1);
      equal text (show ~limits t) (show ~limits s))

(* Types with a text form, drawn as display shows them as they read. *)
type form = Form : 'a Type.t * 'a Gen.t -> form

let readable =
  let piece =
    Gen.of_list [ ""; "a"; "é"; "日本"; "𝄞"; ","; " "; "\""; "\\"; "∅" ]
  in
  Gen.map (String.concat "") (Gen.list ~size:(Gen.int_range 0 3) piece)

let forms =
  let drawn ty = Form (ty, Option.get (G.value ty)) in
  Type.
    [
      drawn bool;
      drawn int8;
      drawn int16;
      drawn int32;
      drawn int64;
      drawn uint8;
      drawn uint16;
      drawn uint32;
      drawn uint64;
      Form (string, readable);
      Form (binary, Gen.map Binary.of_string readable);
      drawn (categorical [| "a"; "é"; ""; " x " |]);
      drawn date;
      drawn (datetime Ns);
      drawn (datetime Us);
      drawn (datetime ~zone:"UTC" S);
      drawn (datetime ~zone:"Europe/Paris" Ms);
    ]

let shown =
  Gen.with_pp G.pp_sample
    (Gen.bind (Gen.of_list forms) (fun (Form (ty, g)) ->
         Gen.map
           (fun vs -> G.Sample (ty, Array.map Option.some vs))
           (Gen.array ~size:(Gen.int_range 1 20) g)))

(* [cells ty vs] is the text of the cells that display shows for [vs]. Text
   aligns left after the line's space, other values are trimmed. *)
let cells (type a) (ty : a Type.t) vs =
  let n = Array.length vs in
  let limits = { head = n; tail = 0; columns = 1; width = max_int } in
  let lines =
    String.split_on_char '\n' (show ~limits (v [ ("a", Column.v ty vs) ]))
  in
  let cell line =
    match ty with
    | String | Binary | Categorical _ ->
        String.sub line 1 (String.length line - 1)
    | _ -> String.trim line
  in
  List.map cell (List.filteri (fun i _ -> i >= 3) lines)

let reads_back =
  prop "Column.parse reads back the cells display shows" shown
    (fun (G.Sample (ty, vs)) ->
      let vs = Array.map Option.get vs in
      let parse texts =
        let c = Column.v Type.string (Array.of_list texts) in
        let pp ppf (row, why) = Format.fprintf ppf "row %d: %s" row why in
        Column.values (Type.kind ty) (require_ok ~pp (Column.parse (Any ty) c))
      in
      Law.round_trip (array (G.witness ty)) (list string) (cells ty) parse vs)

let () =
  exit
    (run "Display"
       [
         types;
         elision;
         float_cuts;
         limits_cases;
         group "Laws" [ blind_to_batching; reads_back ];
       ])
