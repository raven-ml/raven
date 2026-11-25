(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
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

let rows ty xs = Column.options (Type.kind ty) (parsed ty xs)
let tick f n = Option.get (f n)
let day n = Option.get (Time.Date.of_days n)

(* [stored ty xs] is the storage of the texts [xs] read as the integer or
   datetime type [ty], as int64 bits. *)
let stored ty xs =
  let c = parsed ty (List.map Option.some xs) in
  match Column.type_ c with
  | Any Uint64 ->
      Nx.to_array (Nx.bitcast Nx.int64 (Column.to_tensor Nx.uint64 c))
  | Any (Int64 | Datetime _) -> Nx.to_array (Column.to_tensor Nx.int64 c)
  | _ -> Array.map Int64.of_int (Column.values Kind.int c)

(* Column.parse *)

type case = Case : 'a Type.t * string list * 'a list -> case

let values =
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
          [
            "1.5";
            "-0.25";
            ".5";
            "5.";
            "1e3";
            "1E-3";
            "+2";
            "0.1";
            "+.5";
            "-.5e-3";
            "1e+3";
          ],
          [ 1.5; -0.25; 0.5; 5.; 1000.; 0.001; 2.; 0.1; 0.5; -0.0005; 1000. ] );
      Case
        ( float64,
          [
            "inf"; "-Infinity"; "NaN"; "INF"; "+inf"; "-nan"; "+NaN"; "iNfInItY";
          ],
          [
            infinity; neg_infinity; nan; infinity; infinity; nan; nan; infinity;
          ] );
      Case
        ( float64,
          [ "1e400"; "-1e-400"; "4.9e-324" ],
          [ infinity; -0.; 4.9e-324 ] );
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
      (* The float below 2^-792, which some C libraries' strtod read as
         2^-792. *)
      Case (float64, [ "3.839223843572815e-239" ], [ 0x1.fffffffffffffp-793 ]);
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
        ( date,
          [
            "1970-01-01"; "2000-02-29"; "0000-01-01"; "9999-12-31"; "1969-12-31";
          ],
          List.map day [ 0; 11016; -719528; 2932896; -1 ] );
      Case
        ( date,
          [
            "-0044-03-15";
            "+12345-01-01";
            "-0001-12-31";
            "+10000-01-01";
            "+012345-01-01";
          ],
          List.map
            (fun ymd -> Option.get (Time.Date.of_civil ymd))
            [
              (-44, 3, 15);
              (12345, 1, 1);
              (-1, 12, 31);
              (10000, 1, 1);
              (12345, 1, 1);
            ] );
      Case
        ( wall Us,
          [
            "1970-01-01T00:00:01";
            "1970-01-01 00:00:00.5";
            "1969-12-31T23:59:59.999999";
          ],
          List.map (tick Time.of_us) [ 1_000_000L; 500_000L; -1L ] );
      Case
        ( wall Ms,
          [ "1970-01-01T00:00:00.001"; "1970-01-01T00:00:00.100000" ],
          List.map (tick Time.of_ms) [ 1L; 100L ] );
      Case (wall S, [ "1970-01-02T00:00:00.000" ], [ tick Time.of_s 86400L ]);
      Case
        ( instant Ns,
          [
            "1970-01-01T01:00:00+01:00";
            "1970-01-01T00:00:00Z";
            "1969-12-31T19:00:00-05:00";
            "1970-01-01T23:59:00+23:59";
            "1970-01-01T00:00:00-00:00";
          ],
          List.init 5 (fun _ -> Time.of_ns 0L) );
      Case
        ( instant Ns,
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

let reads (Case (ty, xs, expected)) =
  equal
    (array (option (G.witness ty)))
    (Array.of_list (List.map Option.some expected))
    (rows ty (List.map Option.some xs))

(* The bounds of each integer type, as texts and as the int64 bits it stores,
   and the integers just outside them. *)
let bounds =
  Type.
    [
      (Any int8, ("-128", -128L), ("127", 127L), "-129", "128");
      (Any int16, ("-32768", -32768L), ("32767", 32767L), "-32769", "32768");
      ( Any int32,
        ("-2147483648", -2147483648L),
        ("2147483647", 2147483647L),
        "-2147483649",
        "2147483648" );
      ( Any int64,
        ("-9223372036854775808", Int64.min_int),
        ("9223372036854775807", Int64.max_int),
        "-9223372036854775809",
        "9223372036854775808" );
      (Any uint8, ("0", 0L), ("255", 255L), "-1", "256");
      (Any uint16, ("0", 0L), ("65535", 65535L), "-1", "65536");
      (Any uint32, ("0", 0L), ("4294967295", 4294967295L), "-1", "4294967296");
      ( Any uint64,
        ("0", 0L),
        ("18446744073709551615", -1L),
        "-1",
        "18446744073709551616" );
    ]

let integer_types = List.map (fun (ty, _, _, _, _) -> ty) bounds

(* [in_range ty x] is [true] iff the int64 [x] is a value of the integer type
   [ty]. *)
let in_range (Type.Any ty) x =
  let signed bits =
    Int64.(x >= neg (shift_left 1L (bits - 1)) && x < shift_left 1L (bits - 1))
  in
  let unsigned bits = x >= 0L && (bits = 64 || x < Int64.shift_left 1L bits) in
  match ty with
  | Type.Int8 -> signed 8
  | Int16 -> signed 16
  | Int32 -> signed 32
  | Int64 -> true
  | Uint8 -> unsigned 8
  | Uint16 -> unsigned 16
  | Uint32 -> unsigned 32
  | Uint64 -> unsigned 64
  | _ -> invalid_arg "in_range"

let reading =
  group "Column.parse"
    [
      cases "reads each type's text"
        ~id:(fun (Case (ty, xs, _)) ->
          Format.asprintf "%a: %s" Type.pp ty
            (String.escaped (String.concat " " xs)))
        values reads;
      cases "reads an integer type's bounds and refuses the integers past them"
        ~id:(fun (ty, _, _, _, _) -> Format.asprintf "%a" pp_any ty)
        bounds
        (fun (Type.Any ty, (lo, lo_bits), (hi, hi_bits), below, above) ->
          equal (array int64) [| lo_bits; hi_bits |] (stored ty [ lo; hi ]);
          equal int 0 (fst (refused ty [ Some below ]));
          equal int 0 (fst (refused ty [ Some above ])));
      prop "reads exactly the int64s in an integer type's range"
        (Gen.pair (Gen.of_list ~pp:pp_any integer_types) Gen.int64)
        (fun (Type.Any ty, x) ->
          let held = in_range (Any ty) x in
          cover "an integer in the range" held;
          cover "an integer past it" (not held);
          match (parse ty [ Some (Int64.to_string x) ], held) with
          | Ok _, true ->
              equal (array int64) [| x |] (stored ty [ Int64.to_string x ])
          | Error _, false -> ()
          | Ok _, false -> failf "read %Ld, past the range" x
          | Error (_, why), true -> failf "refused %Ld, in the range: %s" x why);
      test "reads uint64 values past int64 as their bits" (fun () ->
          equal (array int64)
            [| -1L; Int64.min_int; 1L |]
            (stored Type.uint64
               [ "18446744073709551615"; "9223372036854775808"; "+1" ]));
      test "reads the extremes of a datetime's ticks" (fun () ->
          equal (array int64)
            [| Int64.max_int; Int64.min_int; 185542587187199L |]
            (stored (Type.wall S)
               [
                 "+292277026596-12-04T15:30:07";
                 "-292277022657-01-27T08:29:52";
                 "+5881580-07-11T23:59:59";
               ]));
      test "reads a datetime's signed year, past nanoseconds' range" (fun () ->
          equal (array int64)
            [| 253402300800L; -62167219201L; -62135596800L |]
            (stored (Type.wall S)
               [
                 "+10000-01-01T00:00:00";
                 "-0001-12-31 23:59:59";
                 "0001-01-01T00:00:00";
               ]));
    ]

(* Refusals. The reasons are the phrases that Column.parse writes today. *)

let refusal_rows =
  Type.
    [
      (Any bool, "True", __POS_OF__ {|not true or false|});
      (Any bool, "1", __POS_OF__ {|not true or false|});
      (Any int8, "1.0", __POS_OF__ {|not an integer|});
      (Any int8, " 1", __POS_OF__ {|not an integer|});
      (Any int8, "+", __POS_OF__ {|not an integer|});
      (Any int8, "", __POS_OF__ {|not an integer|});
      (Any int32, "99999999999999999999999", __POS_OF__ {|out of range|});
      (Any float64, ".", __POS_OF__ {|not a number|});
      (Any float64, "e5", __POS_OF__ {|not a number|});
      (Any float64, "1e", __POS_OF__ {|not a number|});
      (Any float64, "1.2.3", __POS_OF__ {|not a number|});
      (Any float64, "0x10", __POS_OF__ {|not a number|});
      (Any float64, "1_000", __POS_OF__ {|not a number|});
      (Any float64, "nan1", __POS_OF__ {|not a number|});
      (Any float64, "--1", __POS_OF__ {|not a number|});
      (Any float64, "+-1", __POS_OF__ {|not a number|});
      (Any float64, "infinit", __POS_OF__ {|not a number|});
      (Any float64, "in", __POS_OF__ {|not a number|});
      (Any float64, " 1.0", __POS_OF__ {|not a number|});
      (Any float32, "", __POS_OF__ {|not a number|});
      (Any date, "2023-02-29", __POS_OF__ {|not a day of the calendar|});
      (Any date, "1900-02-29", __POS_OF__ {|not a day of the calendar|});
      (Any date, "2024-00-10", __POS_OF__ {|not a day of the calendar|});
      (Any date, "2024-01-00", __POS_OF__ {|not a day of the calendar|});
      (Any date, "2024-1-01", __POS_OF__ {|not YYYY-MM-DD|});
      (Any date, "2024-13-01", __POS_OF__ {|not a day of the calendar|});
      (Any date, "2024-01-01T00:00:00", __POS_OF__ {|not YYYY-MM-DD|});
      (Any date, "+123-01-01", __POS_OF__ {|not YYYY-MM-DD|});
      (Any date, "12345-01-01", __POS_OF__ {|not YYYY-MM-DD|});
      (Any date, "+12a45-01-01", __POS_OF__ {|not YYYY-MM-DD|});
      (Any date, "+2024-02-29", __POS_OF__ {|not YYYY-MM-DD|});
      (Any date, "+0000-01-01", __POS_OF__ {|not YYYY-MM-DD|});
      (Any date, "-0000-01-01", __POS_OF__ {|not YYYY-MM-DD|});
      (Any date, "+09999-12-31", __POS_OF__ {|not YYYY-MM-DD|});
      (Any date, "-12345-02-30", __POS_OF__ {|not a day of the calendar|});
      (Any date, "+5881581-01-01", __POS_OF__ {|out of range|});
      (Any date, "-99999999999999999999-01-01", __POS_OF__ {|out of range|});
      ( Any (wall Us),
        "2024-01-01T00:00:00.1234567",
        __POS_OF__ {|not a whole number of microseconds|} );
      ( Any (wall Us),
        "2024-01-01T00:00:00Z",
        __POS_OF__ {|not YYYY-MM-DDThh:mm:ss without an offset|} );
      (Any (wall Us), "2024-01-01T24:00:00", __POS_OF__ {|not a time of day|});
      (Any (wall Us), "2024-01-01T00:00:60", __POS_OF__ {|not a time of day|});
      ( Any (wall Us),
        "2024-01-01T00:00:00.",
        __POS_OF__ {|not YYYY-MM-DDThh:mm:ss without an offset|} );
      ( Any (wall Us),
        "2024-01-01T00:00:00.1234567890",
        __POS_OF__ {|not YYYY-MM-DDThh:mm:ss without an offset|} );
      ( Any (wall Us),
        "2024-01-01t00:00:00",
        __POS_OF__ {|not YYYY-MM-DDThh:mm:ss without an offset|} );
      ( Any (wall S),
        "2024-01-01T00:00:00.5",
        __POS_OF__ {|not a whole number of seconds|} );
      (Any (wall S), "+999-01-01T00:00:00", __POS_OF__ {|not YYYY-MM-DD|});
      (Any (wall S), "+2024-01-01T00:00:00", __POS_OF__ {|not YYYY-MM-DD|});
      (Any (instant S), "-0000-01-01T00:00:00Z", __POS_OF__ {|not YYYY-MM-DD|});
      (Any (wall S), "+292277026596-12-04T15:30:08", __POS_OF__ {|out of range|});
      (Any (wall S), "-292277022657-01-27T08:29:51", __POS_OF__ {|out of range|});
      (Any (wall Ms), "+300000000-01-01T00:00:00", __POS_OF__ {|out of range|});
      ( Any (wall S),
        "+10000-01-01",
        __POS_OF__ {|not YYYY-MM-DDThh:mm:ss without an offset|} );
      ( Any (wall Ms),
        "2024-01-01T00:00:00.0015",
        __POS_OF__ {|not a whole number of milliseconds|} );
      ( Any (instant Us),
        "2024-01-01T00:00:00",
        __POS_OF__ {|not YYYY-MM-DDThh:mm:ss with an offset|} );
      ( Any (instant Us),
        "2024-01-01T00:00:00+0100",
        __POS_OF__ {|not YYYY-MM-DDThh:mm:ss with an offset|} );
      ( Any (instant Us),
        "2024-01-01T00:00:00z",
        __POS_OF__ {|not YYYY-MM-DDThh:mm:ss with an offset|} );
      ( Any (instant Us),
        "2024-01-01T00:00:00+24:00",
        __POS_OF__ {|not an offset|} );
      ( Any (instant Us),
        "2024-01-01T00:00:00+01:60",
        __POS_OF__ {|not an offset|} );
      ( Any (instant Ns),
        "2262-04-11T23:47:16.854775808Z",
        __POS_OF__ {|out of range|} );
      ( Any (instant Ns),
        "1677-09-21T00:12:43.145224191Z",
        __POS_OF__ {|out of range|} );
      (Any string, "\xff", __POS_OF__ {|invalid UTF-8 at byte 0|});
      (Any string, "\xc3", __POS_OF__ {|invalid UTF-8 at byte 0|});
      (Any string, "\xc0\x80", __POS_OF__ {|invalid UTF-8 at byte 0|});
      (Any string, "\xed\xa0\x80", __POS_OF__ {|invalid UTF-8 at byte 0|});
      (Any string, "\xf4\x90\x80\x80", __POS_OF__ {|invalid UTF-8 at byte 0|});
      (Any string, "a\xe2\x82", __POS_OF__ {|invalid UTF-8 at byte 1|});
      (Any (categorical [| "a" |]), "b", __POS_OF__ {|not in the dictionary|});
    ]

let refusing =
  group "Column.parse"
    [
      cases "refuses text that is not its type's, naming the row"
        ~id:(fun (ty, s, _) ->
          Format.asprintf "%s as %a" (String.escaped s) pp_any ty)
        refusal_rows
        (fun (Type.Any ty, s, reason) ->
          let row, why = refused ty [ None; Some s ] in
          equal int 1 row;
          expect why reason);
      cases "names the first row that is not its type's text"
        ~id:(fun (_, xs, _) ->
          String.concat " " (List.map (Option.value ~default:"null") xs))
        [
          (Type.Any Type.int8, [ Some "1"; None; Some "300"; Some "x" ], 2);
          (Any Type.float64, [ Some "1"; Some "2"; Some "3"; Some "x" ], 3);
        ]
        (fun (Type.Any ty, xs, expected) ->
          equal int expected (fst (refused ty xs)));
      test "reads a null row as null" (fun () ->
          equal
            (array (option (G.witness Type.date)))
            [| None; None |]
            (rows Type.date [ None; None ]);
          equal
            (array (option (G.witness Type.int16)))
            [| Some 1; None; Some (-3) |]
            (rows Type.int16 [ Some "1"; None; Some "-3" ]));
      test "reads a string column as it reads a binary one" (fun () ->
          let c = Column.of_options Type.string [| Some "42"; None |] in
          equal
            (array (option (G.witness Type.uint8)))
            [| Some 42; None |]
            (Column.options Kind.int
               (Result.get_ok (Column.parse (Any Type.uint8) c))));
      test "raises on a column that is not text" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Column.parse (Any Type.int8) (Column.v Type.int8 [| 1 |])));
      cases "raises on a type without a text"
        ~id:(Format.asprintf "%a" pp_any)
        Type.
          [
            Any (clock S);
            Any (duration Ms);
            Any (list int8);
            Any (record [ ("a", Any int8) ]);
            Any (tensor Nx.float32 [| 2 |]);
            Any (ext ~name:"x" int8);
          ]
        (fun ty ->
          raises_match Exn.invalid_arg (fun () ->
              Column.parse ty (bytes [ Some "1" ])));
    ]

(* Exact decimals *)

(* The tests write a float's digits from its exact decimal, computed here: the C
   library's printf does not write them on every system, and under mingw-w64
   msvcrt's keeps 17 digits and rounds from those. A decimal [(d, e)] is the
   natural of digits [d], most significant first, times [10^e], without leading
   or trailing zero; zero is [("", 0)]. *)

let normalize d e =
  let n = String.length d in
  let lead = ref 0 and stop = ref n in
  while !lead < n && d.[!lead] = '0' do
    incr lead
  done;
  while !stop > !lead && d.[!stop - 1] = '0' do
    decr stop
  done;
  if !lead = !stop then ("", 0)
  else (String.sub d !lead (!stop - !lead), e + n - !stop)

(* [times ds f] is the little-endian decimal digits [ds] times [f], below
   [2^20], without zero digits on top. *)
let times ds f =
  let n = Array.length ds in
  let r = Array.make (n + 7) 0 and carry = ref 0 in
  for i = 0 to n + 6 do
    let t = (if i < n then ds.(i) * f else 0) + !carry in
    r.(i) <- t mod 10;
    carry := t / 10
  done;
  let top = ref (n + 7) in
  while !top > 0 && r.(!top - 1) = 0 do
    decr top
  done;
  Array.sub r 0 !top

(* [dyadic n k] is the decimal of [n × 2^k], for [n] at least [0]: [n] times
   [2^k], or times [5^-k] and [10^k]. *)
let dyadic n k =
  let rec digits n = if n = 0 then [] else (n mod 10) :: digits (n / 10) in
  let ds = ref (Array.of_list (digits n)) in
  let base, step = if k >= 0 then (2, 19) else (5, 8) in
  let left = ref (Int.abs k) in
  while !left > 0 do
    let j = Int.min step !left in
    let rec pow k = if k = 0 then 1 else base * pow (k - 1) in
    ds := times !ds (pow j);
    left := !left - j
  done;
  let d = !ds in
  let n = Array.length d in
  normalize
    (String.init n (fun i -> Char.chr (48 + d.(n - 1 - i))))
    (Int.min k 0)

(* [decimal x] is the decimal of [|x|], [x] finite. *)
let decimal x =
  let f, e = Float.frexp (Float.abs x) in
  dyadic (Float.to_int (Float.ldexp f 53)) (e - 53)

(* [decimal_of_text s] is the decimal the unsigned text [s] writes, as
   [12.5e-3]. *)
let decimal_of_text s =
  let mantissa, exp =
    match String.index_opt s 'e' with
    | None -> (s, 0)
    | Some i ->
        ( String.sub s 0 i,
          int_of_string (String.sub s (i + 1) (String.length s - i - 1)) )
  in
  let int, frac =
    match String.index_opt mantissa '.' with
    | None -> (mantissa, "")
    | Some i ->
        ( String.sub mantissa 0 i,
          String.sub mantissa (i + 1) (String.length mantissa - i - 1) )
  in
  normalize (int ^ frac) (exp - String.length frac)

let compare_decimal (d0, e0) (d1, e1) =
  match (d0, d1) with
  | "", "" -> 0
  | "", _ -> -1
  | _, "" -> 1
  | _ ->
      let m0 = String.length d0 + e0 and m1 = String.length d1 + e1 in
      if m0 <> m1 then Int.compare m0 m1 else String.compare d0 d1

(* [increment d] is the decimal digits [d] plus one, as long as [d] unless all
   nines. *)
let increment d =
  let b = Bytes.of_string d in
  let rec carry i =
    if i < 0 then "1" ^ Bytes.to_string b
    else if Bytes.get b i = '9' then begin
      Bytes.set b i '0';
      carry (i - 1)
    end
    else begin
      Bytes.set b i (Char.chr (Char.code (Bytes.get b i) + 1));
      Bytes.to_string b
    end
  in
  carry (String.length d - 1)

(* [format_e p negative (d, e)] is C's [%.*e] with [p] digits after the point of
   the decimal [(d, e)], negative if [negative], rounded once, ties to even. *)
let format_e p negative (d, e) =
  let n = p + 1 and len = String.length d in
  let kept, exp =
    if d = "" then (String.make n '0', 0)
    else if len <= n then (d ^ String.make (n - len) '0', len - 1 + e)
    else
      let head = String.sub d 0 n and tail = String.sub d n (len - n) in
      (* [tail] has no trailing zero: it is half a unit exactly iff it is
         [5]. *)
      let odd = (Char.code head.[n - 1] - 48) land 1 = 1 in
      let up = tail.[0] > '5' || (tail.[0] = '5' && (len - n > 1 || odd)) in
      let head = if up then increment head else head in
      if String.length head > n then (String.sub head 0 n, len + e)
      else (head, len - 1 + e)
  in
  Printf.sprintf "%s%c%se%c%02d"
    (if negative then "-" else "")
    kept.[0]
    (if p > 0 then "." ^ String.sub kept 1 p else "")
    (if exp < 0 then '-' else '+')
    (Int.abs exp)

(* [distance a b] is [|a - b|]. *)
let distance (d0, e0) (d1, e1) =
  let e = Int.min e0 e1 in
  let a = d0 ^ String.make (e0 - e) '0' and b = d1 ^ String.make (e1 - e) '0' in
  let big, small =
    if compare_decimal (d0, e0) (d1, e1) >= 0 then (a, b) else (b, a)
  in
  (* [digit s k] is the [k]th digit of [s] from the right, from [1]. *)
  let digit s k =
    let n = String.length s in
    if k > n then 0 else Char.code s.[n - k] - 48
  in
  let n = String.length big in
  let r = Bytes.make n '0' and borrow = ref 0 in
  for k = 1 to n do
    let x = digit big k - digit small k - !borrow in
    Bytes.set r (n - k) (Char.chr (48 + ((x + 10) mod 10)));
    borrow := if x < 0 then 1 else 0
  done;
  normalize (Bytes.to_string r) e

(* [nearest_of t x] is the float64 nearest the decimal [t], ties to even, of the
   float64 [x], at least [0.], and its neighbours. An infinity counts as
   [2^1024], where the next float would be. A reading [x] of [t] is right iff it
   is its own nearest. *)
let nearest_of t x =
  let value y = if y = Float.infinity then dyadic 1 1024 else decimal y in
  let even y = Int64.(logand (bits_of_float y) 1L) = 0L in
  let nearer y z =
    let c = compare_decimal (distance t (value y)) (distance t (value z)) in
    c < 0 || (c = 0 && even y)
  in
  List.fold_left
    (fun best y -> if y >= 0. && nearer y best then y else best)
    x
    [ Float.pred x; Float.succ x ]

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
        (m < subnormal, sign (of_bits m), sign (of_bits (Int64.succ m)), m))
      (pair
         (one_of [ int64_range 0L (Int64.pred top); int64_range 0L subnormal ])
         bool))

(* [midpoint a b] is the decimal of the magnitude halfway between the
   consecutive floats [a] and [b] of one sign: [(2 n + 1) 2^(u - 1)], the
   smaller magnitude being [n 2^u] and their gap [2^u]. *)
let midpoint a b =
  let a = Float.abs a and b = Float.abs b in
  let u = snd (Float.frexp (Float.abs (b -. a))) - 1 in
  dyadic ((2 * Float.to_int (Float.ldexp (Float.min a b) (-u))) + 1) (u - 1)

(* The texts of the decimal [mid], negative if [negative], exactly, just above
   it and just below it, in magnitude. *)
let around negative mid =
  let s = format_e 1100 false mid in
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
  let sign = if negative then "-" else "" in
  ( sign ^ digits ^ exp,
    sign ^ digits ^ String.make 40 '0' ^ "1" ^ exp,
    sign ^ below ^ String.make 40 '9' ^ exp )

let rounds_halfway ty (subnormal, a, b, m) =
  let exact, above, below = around (b < 0.) (midpoint a b) in
  cover "negative" (b < 0.);
  cover "subnormal" subnormal;
  let even = Int64.logand m 1L = 0L in
  equal int64 (bits (if even then a else b)) (bits (read_one ty exact));
  equal int64 (bits b) (bits (read_one ty above));
  equal int64 (bits a) (bits (read_one ty below))

let laws =
  group "Column.parse"
    [
      prop "reads a decimal number as the float64 nearest it"
        ~examples:
          [
            "1e400";
            "123456789e-330";
            "2.4703282292062327e-324";
            "12345678901234567890123";
          ] (Gen.with_pp Format.pp_print_string float_texts) (fun s ->
          let x = read_one Type.float64 s in
          let d = decimal_of_text s in
          cover "overflows to infinity" (x = Float.infinity);
          cover "rounds to a subnormal" (x > 0. && x < Float.min_float);
          cover "rounds to zero from a nonzero text" (x = 0. && d <> ("", 0));
          cover "more than 19 significant digits" (String.length (fst d) > 19);
          equal int64 (bits (nearest_of d x)) (bits x));
      prop "reads a float64 written with 17 digits as itself" Gen.float
        (fun x ->
          Law.round_trip (G.witness Type.float64) string
            (Printf.sprintf "%.17g") (read_one Type.float64) x);
      prop
        "rounds a float64 halfway text to even, and texts beside it to the \
         nearer value"
        (halfway ~subnormal:0x10000000000000L ~top:0x7fefffffffffffffL
           Int64.float_of_bits)
        (rounds_halfway Type.float64);
      prop
        "rounds a float32 halfway text to even, and texts beside it to the \
         nearer value"
        (halfway ~subnormal:0x800000L ~top:0x7f7fffffL (fun m ->
             Int32.float_of_bits (Int64.to_int32 m)))
        (rounds_halfway Type.float32);
      prop
        "rounds a float16 halfway text to even, and texts beside it to the \
         nearer value"
        (halfway ~subnormal:0x400L ~top:0x7bffL (fun m ->
             half_of_bits (Int64.to_int m)))
        (rounds_halfway Type.float16);
      prop "reads a date written by Time.Date.pp as itself"
        Gen.(map day (int_range (-719528) 2932896))
        (Law.round_trip (G.witness Type.date) string
           (Format.asprintf "%a" Time.Date.pp)
           (read_one Type.date));
      prop "reads an instant written by Time.pp as itself"
        Gen.(map Time.of_ns int64)
        (fun t ->
          let utc = Type.instant Ns in
          Law.round_trip (G.witness utc) string
            (Format.asprintf "%aZ" Time.pp)
            (read_one utc) t);
    ]

(* Column.print *)

(* The printer's text, by the rule talon.mli states for Column.print: of the
   fewest significant digits that read back, the nearest, and of two as near the
   one whose last digit is even, written without its exponent from [1e-7] up to
   [1e21]. *)

let positional s =
  let e = String.index s 'e' in
  let exp = int_of_string (String.sub s (e + 1) (String.length s - e - 1)) in
  if exp < -7 || exp >= 21 then s
  else
    let negative = s.[0] = '-' in
    let mantissa =
      String.sub s (Bool.to_int negative) (e - Bool.to_int negative)
    in
    let digits = String.concat "" (String.split_on_char '.' mantissa) in
    let n = String.length digits in
    let text =
      if exp >= n - 1 then digits ^ String.make (exp - n + 1) '0'
      else if exp >= 0 then
        String.sub digits 0 (exp + 1)
        ^ "."
        ^ String.sub digits (exp + 1) (n - exp - 1)
      else "0." ^ String.make (-exp - 1) '0' ^ digits
    in
    if negative then "-" ^ text else text

(* [shortest_nearest ty xs] is the text of each of the finite, positive [xs],
   values of [ty], that the printer's contract states: of the fewest significant
   digits that read back as it, the nearest to it, ties to an even last digit.
   The candidates of [p] digits are the two [p]-digit decimals around the value;
   every value still without a text has both read back at once. *)
let shortest_nearest ty xs =
  let n = Array.length xs in
  let out = Array.make n None in
  let decimals = Array.map decimal xs in
  for p = 1 to 17 do
    let open_ = List.filter (fun i -> out.(i) = None) (List.init n Fun.id) in
    (* Each candidate: its value, whether its last digit is even, its text. *)
    let around i =
      let d, e = decimals.(i) in
      let len = String.length d in
      if len <= p then [ ((d, e), true, format_e (p - 1) false (d, e)) ]
      else
        let lo = String.sub d 0 p and up = increment (String.sub d 0 p) in
        let e' = e + len - p in
        let up' = String.sub up (String.length up - p) p in
        let even s = (Char.code s.[String.length s - 1] - 48) land 1 = 0 in
        [
          (normalize lo e', even lo, format_e (p - 1) false (normalize lo e'));
          (normalize up e', even up', format_e (p - 1) false (normalize up e'));
        ]
    in
    let candidates = List.map (fun i -> (i, around i)) open_ in
    let texts =
      List.concat_map
        (fun (_, cs) -> List.map (fun (_, _, t) -> Some t) cs)
        candidates
    in
    let back = ref (Array.to_list (rows ty texts)) in
    List.iter
      (fun (i, cs) ->
        let fits =
          List.filter
            (fun _ ->
              match !back with
              | y :: rest ->
                  back := rest;
                  Option.get y = xs.(i)
              | [] -> assert false)
            cs
        in
        let nearer (d0, even0, _) (d1, _, _) =
          let c =
            compare_decimal
              (distance d0 decimals.(i))
              (distance d1 decimals.(i))
          in
          c < 0 || (c = 0 && even0)
        in
        match fits with
        | [] -> ()
        | [ (_, _, t) ] -> out.(i) <- Some (positional t)
        | [ a; b ] ->
            let _, _, t = if nearer a b then a else b in
            out.(i) <- Some (positional t)
        | _ -> assert false)
      candidates
  done;
  Array.map Option.get out

(* [canonical ty xs] is the text that talon.mli states for each of the values
   [xs] of the float type [ty]. *)
let canonical ty xs =
  let finite x = Float.is_finite x && x <> 0. in
  let magnitudes =
    Array.of_list
      (List.filter_map
         (fun x -> if finite x then Some (Float.abs x) else None)
         (Array.to_list xs))
  in
  let texts = shortest_nearest ty magnitudes and next = ref 0 in
  Array.map
    (fun x ->
      if Float.is_nan x then "nan"
      else if x = Float.infinity then "inf"
      else if x = Float.neg_infinity then "-inf"
      else if x = 0. then if Float.sign_bit x then "-0" else "0"
      else
        let t = texts.(!next) in
        incr next;
        if x < 0. then "-" ^ t else t)
    xs

let texts c = Column.options Kind.string (Column.print c)
let printer ty xs = Array.map Option.get (texts (Column.v ty xs))

let printed ty vs expected =
  equal
    (array (option string))
    (Array.of_list (List.map Option.some expected))
    (texts (Column.v ty (Array.of_list vs)))

let is_power_of_two x = x <> 0. && Float.abs (fst (Float.frexp x)) = 0.5

let prints_canonical ty ~normal x =
  cover "a subnormal" (x <> 0. && Float.abs x < normal);
  cover "a power of two" (is_power_of_two x);
  equal string (canonical ty [| x |]).(0) (printer ty [| x |]).(0)

(* [floats ~p ~emin of_bits] draws the floats of [p] significant bits and least
   normal exponent [emin], of the bits [of_bits] reads: any bits, subnormals,
   powers of two and ties, [o / 4] for an odd [o] of [p] bits, which lies
   halfway between the two texts of its fewest digits. *)
let floats ~p ~emin of_bits =
  let open Gen in
  let mantissa = Int64.pred (Int64.shift_left 1L (p - 1)) in
  let any = map of_bits int64 in
  let subnormal = map (fun m -> of_bits (Int64.logand m mantissa)) int64 in
  let power = map (Float.ldexp 1.) (int_range (emin - p + 1) (1 - emin)) in
  let tie =
    map
      (fun o -> Float.of_int (o lor 1 lor (1 lsl (p - 1))) /. 4.)
      (int_range 0 ((1 lsl p) - 1))
  in
  let* x = frequency [ (4, any); (1, subnormal); (1, power); (1, tie) ] in
  let+ negative = bool in
  if negative then -.x else x

(* [of_ticks ty xs] is the column of [ty] whose values are the int64 [xs]. *)
let of_ticks ty xs =
  let values = Nx.P (Nx.create Nx.int64 [| Array.length xs |] xs) in
  match Column.of_layout ty (Fixed { validity = None; values }) with
  | Ok c -> c
  | Error (row, why) -> failf "row %d: %s" row why

(* Datetimes drawn at the edges of their int64 ticks and of {!Time.Date}'s days,
   which {!G.value}'s instants do not reach outside nanoseconds. *)
let extreme_ticks =
  let per : Type.unit_ -> int64 = function
    | S -> 1L
    | Ms -> 1_000L
    | Us -> 1_000_000L
    | Ns -> 1_000_000_000L
  in
  let open Gen in
  let* u = of_list Type.[ S; Ms; Us; Ns ] in
  let* utc = bool in
  let scaled secs =
    let t = Int64.mul secs (per u) in
    if Int64.div t (per u) = secs then [ t ] else []
  in
  let edges =
    [ Int64.min_int; Int64.succ Int64.min_int; -1L; 0L; 1L ]
    @ [ Int64.pred Int64.max_int; Int64.max_int ]
    @ scaled 185542587187199L @ scaled 185542587187200L
    @ scaled (-185542587187200L) @ scaled (-185542587187201L)
  in
  let+ xs =
    array ~size:(int_range 1 8) (frequency [ (1, of_list edges); (1, int64) ])
  in
  ((if utc then Type.Any (Type.instant u) else Type.Any (Type.wall u)), xs)

(* The types that Column.parse reads and Column.print writes. *)
let readable =
  Type.
    [
      Any bool;
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
      Any string;
      Any binary;
      Any (categorical [| "a"; "é"; "" |]);
      Any date;
      Any (wall S);
      Any (wall Ms);
      Any (wall Us);
      Any (wall Ns);
      Any (instant S);
      Any (instant Ms);
      Any (instant Us);
      Any (instant Ns);
    ]

(* Rows of each type of [readable], drawn together. *)
let samples =
  let sample (Type.Any ty) =
    Gen.map (fun vs -> G.Sample (ty, vs)) (G.options ty)
  in
  let one_each =
    List.fold_right
      (fun ty rest ->
        Gen.(
          let+ x = sample ty and+ xs = rest in
          x :: xs))
      readable (Gen.constant [])
  in
  Gen.with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space G.pp_sample)
    one_each

let printing =
  group "Column.print"
    [
      cases "writes a float in the fewest digits that read back at its width"
        ~id:(fun (ty, _, text) -> Format.asprintf "%a %s" Type.pp ty text)
        Type.
          [
            (float64, 150., "150");
            (float64, 0.0015, "0.0015");
            (float64, 0.1, "0.1");
            (float64, 123.456, "123.456");
            (float64, 5e-324, "5e-324");
            (float64, Float.max_float, "1.7976931348623157e+308");
            (float32, 0.1, "0.1");
            (float32, 16777216., "16777216");
            (float16, 65504., "65500");
            (float16, 6e-8, "6e-08");
            (float16, 0x1p-6, "0.01563");
            (float32, 0x1p-96, "1.2621775e-29");
            (float64, 0x1p-1017, "7.120236347223045e-307");
          ]
        (fun (ty, x, text) -> printed ty [ x ] [ text ]);
      cases "writes no exponent from 1e-7 up to 1e21, and two digits past it"
        ~id:Fun.id [ "100000000000000000000"; "1e+21"; "0.0000001"; "1e-08" ]
        (fun text -> printed Type.float64 [ float_of_string text ] [ text ]);
      test "writes nan, inf, -inf, 0 and -0" (fun () ->
          printed Type.float64
            [ nan; infinity; neg_infinity; 0.; -0. ]
            [ "nan"; "inf"; "-inf"; "0"; "-0" ]);
      test "writes every float16 as the nearest of its fewest digits" (fun () ->
          let n = 0x7c00 in
          let xs =
            Array.init (2 * n) (fun m ->
                if m < n then half_of_bits m else -.half_of_bits (m - n))
          in
          equal (array string)
            (canonical Type.float16 xs)
            (printer Type.float16 xs));
      test
        "writes every float32 and float64 power of two as the nearest of its \
         fewest digits" (fun () ->
          let powers lo hi =
            Array.init (hi - lo + 1) (fun k -> Float.ldexp 1. (lo + k))
          in
          let f32 = powers (-149) 127 and f64 = powers (-1074) 1023 in
          equal (array string)
            (canonical Type.float32 f32)
            (printer Type.float32 f32);
          equal (array string)
            (canonical Type.float64 f64)
            (printer Type.float64 f64));
      prop "writes a float64 as the nearest of its fewest digits"
        ~examples:[ 5e-324; 0x1p-1017; 1125899906842624.25 ]
        (floats ~p:53 ~emin:(-1022) Int64.float_of_bits)
        (prints_canonical Type.float64 ~normal:Float.min_float);
      prop "writes a float32 as the nearest of its fewest digits"
        ~examples:[ 0x1p-149; 0x1p-96; 2097152.25 ]
        (floats ~p:24 ~emin:(-126) (fun m ->
             Int32.float_of_bits (Int64.to_int32 m)))
        (prints_canonical Type.float32 ~normal:0x1p-126);
      test "writes an integer in full" (fun () ->
          printed Type.int8 [ -128; 127 ] [ "-128"; "127" ];
          equal
            (array (option string))
            [| Some "-9223372036854775808"; Some "9223372036854775807" |]
            (texts
               (of_ticks (Any Type.int64) [| Int64.min_int; Int64.max_int |]));
          equal
            (array (option string))
            [| Some "18446744073709551615"; Some "0" |]
            (texts
               (Column.of_tensor (Nx.create Nx.uint64 [| 2 |] [| -1L; 0L |]))));
      test "writes other types as parse reads them" (fun () ->
          printed Type.bool [ true; false ] [ "true"; "false" ];
          printed (Type.categorical [| "a"; "b" |]) [ "b"; "a" ] [ "b"; "a" ];
          printed Type.date
            [ day 0; day (-719529) ]
            [ "1970-01-01"; "-0001-12-31" ];
          printed (Type.instant Ms)
            [ tick Time.of_ms 1500L; tick Time.of_ms (-1L) ]
            [ "1970-01-01T00:00:01.5Z"; "1969-12-31T23:59:59.999Z" ]);
      test "writes a datetime from its ticks, past Time's range" (fun () ->
          let c =
            of_ticks (Any (Type.wall S)) [| 253402300800L; -62167219201L |]
          in
          equal
            (array (option string))
            [| Some "+10000-01-01T00:00:00"; Some "-0001-12-31T23:59:59" |]
            (texts c));
      test "writes a null row as null" (fun () ->
          equal
            (array (option string))
            [| None; Some "1"; None |]
            (texts (Column.of_options Type.int16 [| None; Some 1; None |])));
      test "is the column itself for text, and a string column otherwise"
        (fun () ->
          let is_itself c =
            satisfies ~claim:"the column printed" pass
              (fun p -> p == c)
              (Column.print c)
          in
          is_itself (Column.v Type.string [| "a,b"; ""; "\"" |]);
          is_itself (Column.v Type.binary [| Binary.of_string "\xff" |]);
          let ty = Column.type_ (Column.print (Column.v Type.int8 [| 1 |])) in
          equal string "string" (Format.asprintf "%a" pp_any ty));
      cases "raises on a type parse does not read"
        ~id:(Format.asprintf "%a" pp_any)
        Type.
          [
            Any (duration S);
            Any (clock S);
            Any (list int8);
            Any (record [ ("a", Any int8) ]);
            Any (tensor Nx.float32 [| 2 |]);
            Any (ext ~name:"x" int8);
          ]
        (fun (Type.Any ty) ->
          raises_match Exn.invalid_arg (fun () ->
              Column.print (Column.of_options ty [| None |])));
      prop "is read back by parse at every tick of a datetime" extreme_ticks
        (fun (ty, xs) ->
          let c = of_ticks ty xs in
          let back =
            match Column.parse (Column.type_ c) (Column.print c) with
            | Ok c -> c
            | Error (row, why) -> failf "row %d: %s" row why
          in
          equal (array int64) xs (Nx.to_array (Column.to_tensor Nx.int64 back)));
      prop "is read back by parse, for every type parse reads" samples
        (List.iter (fun (G.Sample (ty, vs)) ->
             cover "a null row" (Array.exists Option.is_none vs);
             let c = Column.of_options ty vs in
             let back = Column.parse (Column.type_ c) (Column.print c) in
             let pp ppf (row, why) = Format.fprintf ppf "row %d: %s" row why in
             equal
               (array (option (G.witness ty)))
               vs
               (Column.options (Type.kind ty) (require_ok ~pp back))));
    ]

let () = exit (run "talon.parse" [ reading; refusing; laws; printing ])
