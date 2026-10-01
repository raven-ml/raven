(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Talon_next
module G = Talon_gen

(* Helpers *)

let pp_any ppf (Type.Any t) = Type.pp ppf t

(* [bytes xs] is the binary column of the texts [xs], [None] a null. *)
let bytes xs =
  Column.of_options Type.binary
    (Array.of_list (List.map (Option.map Binary.of_string) xs))

let parse ty xs = Column.parse (Any ty) (bytes xs)

let parsed ty xs =
  match parse ty xs with
  | Ok c -> c
  | Error (row, why) -> failf "row %d: %s" row why

let refused ty xs =
  match parse ty xs with Ok _ -> fail "parsed" | Error e -> e

let reads ty xs expected =
  equal
    (array (option (G.witness ty)))
    (Array.of_list (List.map Option.some expected))
    (Column.options (Type.kind ty) (parsed ty (List.map Option.some xs)))

let rows ty xs = Column.options (Type.kind ty) (parsed ty xs)
let instant f n = Option.get (f n)
let day n = Option.get (Time.Date.of_days n)

(* Values *)

type case = Case : 'a Type.t * string list * 'a list -> case

let values =
  let dec ~scale us = List.map (fun u -> Decimal.v ~unscaled:u ~scale) us in
  let dict = [| "a"; "b\"c"; "" |] in
  Type.
    [
      Case (bool, [ "true"; "false" ], [ true; false ]);
      Case (int8, [ "-128"; "127"; "+5"; "-0" ], [ -128; 127; 5; 0 ]);
      Case (int16, [ "-32768"; "32767" ], [ -32768; 32767 ]);
      Case (int32, [ "-2147483648"; "2147483647" ], [ -2147483648; 2147483647 ]);
      Case (uint8, [ "0"; "255"; "-0" ], [ 0; 255; 0 ]);
      Case (uint16, [ "65535" ], [ 65535 ]);
      Case (uint32, [ "4294967295" ], [ 4294967295 ]);
      Case (int64, [ "-4611686018427387904"; "0009" ], [ min_int; 9 ]);
      Case (uint64, [ "4611686018427387903"; "+1" ], [ max_int; 1 ]);
      Case
        ( float64,
          [ "1.5"; "-0.25"; ".5"; "5."; "1e3"; "1E-3"; "+2"; "0.1" ],
          [ 1.5; -0.25; 0.5; 5.; 1000.; 0.001; 2.; 0.1 ] );
      Case
        ( float64,
          [ "inf"; "-Infinity"; "NaN"; "1e400"; "-1e-400"; "4.9e-324" ],
          [ infinity; neg_infinity; nan; infinity; -0.; 4.9e-324 ] );
      Case
        ( float64,
          [
            "9007199254740993";
            "0.30000000000000004";
            "2.2250738585072011e-308";
            "123456789012345678901";
          ],
          [
            9007199254740992.;
            0.30000000000000004;
            2.2250738585072011e-308;
            1.2345678901234568e+20;
          ] );
      Case (float32, [ "0.1"; "16777217" ], [ 0.10000000149011612; 16777216. ]);
      Case
        ( float32,
          [
            "1.00000005960464477550461637";
            "1.000000059604644775390625";
            "1.000000059604644775390624";
            "-1.00000005960464477550461637";
          ],
          [ 1.0000001192092896; 1.; 1.; -1.0000001192092896 ] );
      Case
        ( float32,
          [
            "340282356779733661637539395458142568447";
            "340282356779733661637539395458142568448";
            "1e39";
          ],
          [ 3.4028234663852886e+38; infinity; infinity ] );
      Case
        ( float16,
          [ "0.1"; "65504"; "1e5" ],
          [ 0.0999755859375; 65504.; infinity ] );
      Case
        ( float16,
          [ "1.00048828125000000001"; "1.00048828125"; "65519.99999"; "65520" ],
          [ 1.0009765625; 1.; 65504.; infinity ] );
      Case
        ( decimal ~precision:5 ~scale:2,
          [ "123.45"; "-1.5"; "1.500"; "0"; ".5"; "-0.00" ],
          dec ~scale:2 [ 12345L; -150L; 150L; 0L; 50L; 0L ] );
      Case
        ( decimal ~precision:18 ~scale:0,
          [ "999999999999999999" ],
          dec ~scale:0 [ 999999999999999999L ] );
      Case
        ( date,
          [
            "1970-01-01"; "2000-02-29"; "0000-01-01"; "9999-12-31"; "1969-12-31";
          ],
          List.map day [ 0; 11016; -719528; 2932896; -1 ] );
      Case
        ( datetime Us,
          [
            "1970-01-01T00:00:01";
            "1970-01-01 00:00:00.5";
            "1969-12-31T23:59:59.999999";
          ],
          List.map (instant Time.of_us) [ 1_000_000L; 500_000L; -1L ] );
      Case
        ( datetime Ms,
          [ "1970-01-01T00:00:00.001"; "1970-01-01T00:00:00.100000" ],
          List.map (instant Time.of_ms) [ 1L; 100L ] );
      Case
        (datetime S, [ "1970-01-02T00:00:00.000" ], [ instant Time.of_s 86400L ]);
      Case
        ( datetime ~zone:"Europe/Paris" Ns,
          [
            "1970-01-01T01:00:00+01:00";
            "1970-01-01T00:00:00Z";
            "1969-12-31T19:00:00-05:00";
          ],
          List.init 3 (fun _ -> Time.of_ns 0L) );
      Case
        ( datetime ~zone:"UTC" Ns,
          [ "2262-04-11T23:47:16.854775807Z"; "1677-09-21T00:12:43.145224192Z" ],
          [ Time.of_ns Int64.max_int; Time.of_ns Int64.min_int ] );
      Case
        ( string,
          [ "caf\xc3\xa9"; "a\"b"; ""; "\xf0\x9f\x90\xab" ],
          [ "caf\xc3\xa9"; "a\"b"; ""; "\xf0\x9f\x90\xab" ] );
      Case
        ( binary,
          [ "\xff\xfe"; "" ],
          [ Binary.of_string "\xff\xfe"; Binary.of_string "" ] );
      Case (categorical dict, [ "b\"c"; "a"; "" ], [ "b\"c"; "a"; "" ]);
    ]

let value_cases =
  cases
    ~name:(fun (Case (ty, xs, _)) ->
      Format.asprintf "reads %s as %a"
        (String.escaped (String.concat " " xs))
        Type.pp ty)
    "values" values
    (fun (Case (ty, xs, expected)) -> reads ty xs expected)

(* [stored ty xs] is the storage of the texts [xs] read as [ty], as int64
   bits. *)
let stored ty xs =
  let c = parsed ty (List.map Option.some xs) in
  match Column.type_ c with
  | Any Uint64 ->
      Nx.to_array (Nx.bitcast Nx.int64 (Column.to_tensor Nx.uint64 c))
  | _ -> Nx.to_array (Column.to_tensor Nx.int64 c)

let storage =
  group "storage"
    [
      test "int64 reads its extremes" (fun () ->
          equal (array int64)
            [| Int64.min_int; Int64.max_int |]
            (stored Type.int64
               [ "-9223372036854775808"; "9223372036854775807" ]));
      test "uint64 reads values past int64 as their bits" (fun () ->
          equal (array int64)
            [| -1L; Int64.min_int; 1L |]
            (stored Type.uint64
               [ "18446744073709551615"; "9223372036854775808"; "+1" ]));
    ]

(* Refusals *)

let refusals =
  Type.
    [
      (Any bool, "True", "not true or false");
      (Any bool, "1", "not true or false");
      (Any int8, "128", "out of range");
      (Any int8, "-129", "out of range");
      (Any int8, "1.0", "not an integer");
      (Any int8, " 1", "not an integer");
      (Any int8, "+", "not an integer");
      (Any int8, "", "not an integer");
      (Any uint8, "-1", "out of range");
      (Any uint32, "4294967296", "out of range");
      (Any int32, "99999999999999999999999", "out of range");
      (Any int64, "9223372036854775808", "out of range");
      (Any int64, "-9223372036854775809", "out of range");
      (Any uint64, "18446744073709551616", "out of range");
      (Any uint64, "-1", "out of range");
      (Any float64, ".", "not a number");
      (Any float64, "e5", "not a number");
      (Any float64, "1e", "not a number");
      (Any float64, "1.2.3", "not a number");
      (Any float64, "0x10", "not a number");
      (Any float64, "1_000", "not a number");
      (Any float64, "nan1", "not a number");
      (Any float64, "--1", "not a number");
      (Any float32, "", "not a number");
      ( Any (decimal ~precision:5 ~scale:2),
        "1.234",
        "more than 2 digits after the point" );
      (Any (decimal ~precision:5 ~scale:2), "1234.5", "more than 5 digits");
      (Any (decimal ~precision:5 ~scale:2), "1e2", "not a decimal number");
      (Any (decimal ~precision:5 ~scale:2), ".", "not a decimal number");
      (Any date, "2023-02-29", "not a day of the calendar");
      (Any date, "2024-1-01", "not YYYY-MM-DD");
      (Any date, "2024-13-01", "not a day of the calendar");
      (Any date, "2024-01-01T00:00:00", "not YYYY-MM-DD");
      ( Any (datetime Us),
        "2024-01-01T00:00:00.1234567",
        "not a whole number of microseconds" );
      ( Any (datetime Us),
        "2024-01-01T00:00:00Z",
        "not YYYY-MM-DDThh:mm:ss without an offset" );
      (Any (datetime Us), "2024-01-01T24:00:00", "not a time of day");
      (Any (datetime Us), "2024-01-01T00:00:60", "not a time of day");
      ( Any (datetime Us),
        "2024-01-01T00:00:00.",
        "not YYYY-MM-DDThh:mm:ss without an offset" );
      ( Any (datetime Us),
        "2024-01-01T00:00:00.1234567890",
        "not YYYY-MM-DDThh:mm:ss without an offset" );
      ( Any (datetime Us),
        "2024-01-01t00:00:00",
        "not YYYY-MM-DDThh:mm:ss without an offset" );
      ( Any (datetime S),
        "2024-01-01T00:00:00.5",
        "not a whole number of seconds" );
      ( Any (datetime Ms),
        "2024-01-01T00:00:00.0015",
        "not a whole number of milliseconds" );
      ( Any (datetime ~zone:"UTC" Us),
        "2024-01-01T00:00:00",
        "not YYYY-MM-DDThh:mm:ss with an offset" );
      ( Any (datetime ~zone:"UTC" Us),
        "2024-01-01T00:00:00+0100",
        "not YYYY-MM-DDThh:mm:ss with an offset" );
      ( Any (datetime ~zone:"UTC" Us),
        "2024-01-01T00:00:00+24:00",
        "not an offset" );
      ( Any (datetime ~zone:"UTC" Us),
        "2024-01-01T00:00:00+01:60",
        "not an offset" );
      ( Any (datetime ~zone:"UTC" Ns),
        "2262-04-11T23:47:16.854775808Z",
        "out of range" );
      ( Any (datetime ~zone:"UTC" Ns),
        "1677-09-21T00:12:43.145224191Z",
        "out of range" );
      (Any string, "\xff", "invalid UTF-8 at byte 0");
      (Any string, "\xc3", "invalid UTF-8 at byte 0");
      (Any string, "\xc0\x80", "invalid UTF-8 at byte 0");
      (Any string, "\xed\xa0\x80", "invalid UTF-8 at byte 0");
      (Any string, "\xf4\x90\x80\x80", "invalid UTF-8 at byte 0");
      (Any string, "a\xe2\x82", "invalid UTF-8 at byte 1");
      (Any (categorical [| "a" |]), "b", "not in the dictionary");
    ]

let refusal_cases =
  cases
    ~name:(fun (ty, s, _) ->
      Format.asprintf "refuses %s as %a" (String.escaped s) pp_any ty)
    "refusals" refusals
    (fun (Type.Any ty, s, why) ->
      equal (pair int string) (1, why) (refused ty [ None; Some s ]))

let errors =
  group "errors"
    [
      test "names the first row that is not the type's text" (fun () ->
          equal (pair int string) (2, "out of range")
            (refused Type.int8 [ Some "1"; None; Some "300"; Some "x" ]));
      test "names the last row" (fun () ->
          equal (pair int string) (3, "not a number")
            (refused Type.float64 [ Some "1"; Some "2"; Some "3"; Some "x" ]));
      test "reads a column of nulls whatever its type" (fun () ->
          equal
            (array (option (G.witness Type.date)))
            [| None; None |]
            (rows Type.date [ None; None ]));
      test "keeps the nulls of the column it reads" (fun () ->
          let c = parsed Type.int16 [ Some "1"; None; Some "-3" ] in
          equal int 1 (Column.null_count c);
          equal
            (array (option (G.witness Type.int16)))
            [| Some 1; None; Some (-3) |]
            (Column.options Kind.int c));
      test "reads a string column as a binary one" (fun () ->
          let c = Column.of_options Type.string [| Some "42"; None |] in
          equal
            (array (option (G.witness Type.uint8)))
            [| Some 42; None |]
            (Column.options Kind.int
               (Result.get_ok (Column.parse (Any Type.uint8) c))));
      test "raises on a column that is not text" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Column.parse") (fun () ->
              Column.parse (Any Type.int8) (Column.v Type.int8 [| 1 |])));
      test "raises on a type without a text form" (fun () ->
          List.iter
            (fun ty ->
              raises_match (Exn.invalid_arg ~substring:"Column.parse")
                (fun () -> Column.parse ty (bytes [ Some "1" ])))
            Type.
              [
                Any (clock S);
                Any (duration Ms);
                Any (list int8);
                Any (record [ ("a", Any int8) ]);
                Any (tensor Nx.float32 [| 2 |]);
                Any (ext ~name:"x" int8);
              ]);
    ]

(* Laws *)

let float_texts =
  let open Gen in
  let digits = string_of ~size:(int_range 0 22) (char_range '0' '9') in
  let* int = digits in
  let* frac = digits in
  let* exp = option (int_range (-340) 330) in
  let int = if int = "" && frac = "" then "0" else int in
  constant
    (int
    ^ (if frac = "" then "" else "." ^ frac)
    ^ match exp with None -> "" | Some e -> "e" ^ string_of_int e)

let read_one ty s =
  match rows ty [ Some s ] with [| Some x |] -> x | _ -> fail "not one value"

let bits = Int64.bits_of_float

(* [half_of_bits m] is the float16 of bits [m]. *)
let half_of_bits m =
  let e = m lsr 10 and f = m land 0x3ff in
  if e = 0 then Float.ldexp (Float.of_int f) (-24)
  else Float.ldexp (Float.of_int (f lor 0x400)) (e - 25)

(* [halfway ~subnormal ~top of_bits] draws the bits [m] and the consecutive
   values [a] and [b] of bits [m] and [m + 1], both below [top] and of one sign,
   subnormals as often as the others. *)
let halfway ~subnormal ~top of_bits =
  Gen.(
    map
      (fun (m, negative) ->
        let sign x = if negative then -.x else x in
        (m < subnormal, sign (of_bits m), sign (of_bits (m + 1)), m))
      (pair (one_of [ int_range 0 (top - 1); int_range 0 subnormal ]) bool))

(* The texts of [mid] exactly, just above it and just below it, in magnitude. *)
let around mid =
  let s = Printf.sprintf "%.1100e" (Float.abs mid) in
  let e = String.index s 'e' in
  let digits = String.sub s 0 e
  and exp = String.sub s e (String.length s - e) in
  let n = ref (String.length digits) in
  while digits.[!n - 1] = '0' do
    decr n
  done;
  let digits = String.sub digits 0 !n in
  let last = !n - 1 - if digits.[!n - 1] = '.' then 1 else 0 in
  let below =
    String.mapi
      (fun i c -> if i = last then Char.chr (Char.code c - 1) else c)
      digits
  in
  let sign = if mid < 0. then "-" else "" in
  ( sign ^ digits ^ exp,
    sign ^ digits ^ String.make 40 '0' ^ "1" ^ exp,
    sign ^ below ^ String.make 40 '9' ^ exp )

let rounds_halfway ty (subnormal, a, b, m) =
  let mid = (a +. b) /. 2. in
  let exact, above, below = around mid in
  cover "negative" (a < 0.);
  cover "subnormal" subnormal;
  equal int64 (bits (if m land 1 = 0 then a else b)) (bits (read_one ty exact));
  equal int64 (bits b) (bits (read_one ty above));
  equal int64 (bits a) (bits (read_one ty below))

let laws =
  group "laws"
    [
      prop "float64 reads a decimal number as float_of_string does"
        (Gen.with_pp Format.pp_print_string float_texts) (fun s ->
          equal int64
            (bits (float_of_string s))
            (bits (read_one Type.float64 s)));
      prop "float64 reads a float written with 17 digits as itself" Gen.float
        (fun x ->
          Law.round_trip (G.witness Type.float64) string
            (Printf.sprintf "%.17g") (read_one Type.float64) x);
      prop
        "float32 rounds a halfway text to even, and texts beside it to the \
         nearer value"
        (halfway ~subnormal:0x800000 ~top:0x7f7fffff (fun m ->
             Int32.float_of_bits (Int32.of_int m)))
        (rounds_halfway Type.float32);
      prop
        "float16 rounds a halfway text to even, and texts beside it to the \
         nearer value"
        (halfway ~subnormal:0x400 ~top:0x7bff half_of_bits)
        (rounds_halfway Type.float16);
      prop "int64 reads an int64 written in decimal as itself" Gen.int64
        (fun x ->
          equal (array int64) [| x |] (stored Type.int64 [ Int64.to_string x ]));
      prop "date reads a day written by Time.Date.pp as itself"
        Gen.(map day (int_range (-719528) 2932896))
        (Law.round_trip (G.witness Type.date) string
           (Format.asprintf "%a" Time.Date.pp)
           (read_one Type.date));
      prop "datetime reads an instant written by Time.pp as itself"
        Gen.(map Time.of_ns int64)
        (fun t ->
          let utc = Type.datetime ~zone:"UTC" Ns in
          Law.round_trip (G.witness utc) string
            (Format.asprintf "%aZ" Time.pp)
            (read_one utc) t);
      prop "decimal reads a decimal written by Decimal.pp as itself"
        Gen.(
          pair (int_range 0 18)
            (map (fun x -> Int64.rem x 1_000_000_000_000_000_000L) int64))
        (fun (scale, unscaled) ->
          let ty = Type.decimal ~precision:18 ~scale in
          Law.round_trip (G.witness ty) string
            (Format.asprintf "%a" Decimal.pp)
            (read_one ty)
            (Decimal.v ~unscaled ~scale));
    ]

let () =
  exit
    (run "talon.next.parse"
       [ value_cases; storage; refusal_cases; errors; laws ])
