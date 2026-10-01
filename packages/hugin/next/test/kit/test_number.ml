(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_kit

let ascii = Locale.v ~minus:"-" ()

let write ?locale ?trim ?group n p x =
  Number.to_string ?locale (Number.v ?trim ?group n p) x

(* Rows [(x, expected)] written in one format; the row names its input with
   every digit. *)
let table name ?locale ?trim ?group n p rows =
  cases
    ~name:(fun (x, e) -> Printf.sprintf "%.17g is %s" x e)
    name rows
    (fun (x, e) -> equal string e (write ?locale ?trim ?group n p x))

(* Expected values are d3-format's tests (d3-format/test/format-type-*-test.js,
   format-trim-test.js), whose digits follow the same rule, written with the
   kit's exponent and prefix forms. *)

let plain =
  group "plain"
    [
      table "one decimal" Plain (Decimals 1)
        [ (0.49, "0.5"); (100., "100.0"); (123456.49, "123456.5") ];
      table "two decimals" Plain (Decimals 2)
        [
          (0.449, "0.45"); (100., "100.00"); (0.429, "0.43"); (-0.429, "−0.43");
        ];
      table "three decimals" Plain (Decimals 3)
        [ (0.4449, "0.445"); (100., "100.000") ];
      table "five decimals" Plain (Decimals 5)
        [ (0.444449, "0.44445"); (100., "100.00000") ];
      table "six decimals" Plain (Decimals 6)
        [ (42., "42.000000"); (-0., "0.000000"); (-1e-12, "0.000000") ];
      table "grouped" ~group:true Plain (Decimals 1)
        [ (123456.49, "123,456.5"); (123456., "123,456.0") ];
      table "grouped, two decimals" ~group:true Plain (Decimals 2)
        [
          (1234567.449, "1,234,567.45");
          (1234567., "1,234,567.00");
          (1e4, "10,000.00");
        ];
      table "grouped, three decimals" ~group:true Plain (Decimals 3)
        [ (12345678.4449, "12,345,678.445"); (12345678., "12,345,678.000") ];
      table "grouped, five decimals" ~group:true Plain (Decimals 5)
        [
          (123456789.444449, "123,456,789.44445");
          (123456789., "123,456,789.00000");
        ];
      table "six significant digits, trimmed" ~trim:true Plain (Significant 6)
        [
          (1., "1");
          (0.1, "0.1");
          (0.01, "0.01");
          (10.0001, "10.0001");
          (123.45, "123.45");
          (123.456, "123.456");
          (123.4567, "123.457");
          (0.000009, "0.000009");
          (0.0000009, "0.0000009");
          (0.00000009, "0.00000009");
          (0.111119, "0.111119");
          (0.1111119, "0.111112");
          (0.11111119, "0.111111");
        ];
      table "three significant digits" Plain (Significant 3)
        [
          (1234.5, "1230"); (0.0012345, "0.00123"); (9.996, "10.0"); (0., "0.00");
        ];
      table "grouped and trimmed" ~group:true ~trim:true Plain (Significant 6)
        [ (10000., "10,000"); (10000.1, "10,000.1") ];
    ]

let exponent =
  group "exponent"
    [
      table "six decimals" Exponent (Decimals 6)
        [
          (0., "0.000000");
          (42., "4.200000×10¹");
          (42000000., "4.200000×10⁷");
          (420000000., "4.200000×10⁸");
          (-4., "−4.000000×10⁰");
          (-42., "−4.200000×10¹");
          (-4200000., "−4.200000×10⁶");
          (-42000000., "−4.200000×10⁷");
          (-0., "0.000000");
          (-1e-12, "−1.000000×10⁻¹²");
        ];
      table "no decimal" Exponent (Decimals 0)
        [ (42., "4×10¹"); (1000., "10³"); (-1000., "−10³") ];
      table "three decimals" Exponent (Decimals 3)
        [ (42., "4.200×10¹"); (1000., "1.000×10³") ];
      table "six decimals, trimmed" ~trim:true Exponent (Decimals 6)
        [
          (0., "0");
          (42., "4.2×10¹");
          (42000000., "4.2×10⁷");
          (0.042, "4.2×10⁻²");
          (-4., "−4×10⁰");
          (-42., "−4.2×10¹");
          (42000000000., "4.2×10¹⁰");
          (0.00000000042, "4.2×10⁻¹⁰");
          (1000., "10³");
          (5e-7, "5×10⁻⁷");
        ];
      table "four decimals, trimmed" ~trim:true Exponent (Decimals 4)
        [
          (0.00000000012345, "1.2345×10⁻¹⁰");
          (0.00000000012340, "1.234×10⁻¹⁰");
          (0.00000000012300, "1.23×10⁻¹⁰");
          (-0.00000000012345, "−1.2345×10⁻¹⁰");
          (12345000000., "1.2345×10¹⁰");
          (12340000000., "1.234×10¹⁰");
          (-12300000000., "−1.23×10¹⁰");
        ];
      table "a mantissa rounding to ten" Exponent (Decimals 1)
        [ (9.96, "1.0×10¹"); (-9.96, "−1.0×10¹"); (99999., "1.0×10⁵") ];
      table "three significant digits" Exponent (Significant 3)
        [ (1234.5, "1.23×10³"); (0., "0.00"); (9.996, "1.00×10¹") ];
    ]

let si =
  group "si"
    [
      table "six significant digits" Si (Significant 6)
        [
          (0., "0.00000");
          (1., "1.00000");
          (10., "10.0000");
          (100., "100.000");
          (999.5, "999.500");
          (999500., "999.500k");
          (1000., "1.00000k");
          (1400., "1.40000k");
          (1500.5, "1.50050k");
          (0.00001, "10.0000µ");
          (0.000001, "1.00000µ");
        ];
      table "three significant digits" Si (Significant 3)
        [
          (0., "0.00");
          (1., "1.00");
          (10., "10.0");
          (100., "100");
          (999.5, "1.00k");
          (999500., "1.00M");
          (1000., "1.00k");
          (1500.5, "1.50k");
          (145500000., "146M");
          (145999999.99999347, "146M");
          (1e26, "100Y");
          (0.000001, "1.00µ");
          (0.009995, "10.0m");
        ];
      table "four significant digits" Si (Significant 4)
        [
          (999.5, "999.5");
          (999500., "999.5k");
          (0.009995, "9.995m");
          (1e33, "1000Q");
        ];
      table "eight significant digits" Si (Significant 8)
        [
          (1.29e-24, "1.2900000y");
          (1.29e-23, "12.900000y");
          (1.29e-22, "129.00000y");
          (1.29e-21, "1.2900000z");
          (-1.29e-24, "−1.2900000y");
          (-1.29e-21, "−1.2900000z");
          (1.23e21, "1.2300000Z");
          (1.23e24, "1.2300000Y");
          (1.23e26, "123.00000Y");
          (1.29e-30, "1.2900000q");
          (1.29e-27, "1.2900000r");
          (1.23e27, "1.2300000R");
          (1.23e30, "1.2300000Q");
          (1.23e31, "12.300000Q");
          (1.29e-33, "0.0012900000q");
        ];
      table "trimmed" ~trim:true Si (Significant 6)
        [
          (0., "0");
          (1., "1");
          (10., "10");
          (100., "100");
          (999.5, "999.5");
          (999500., "999.5k");
          (1000., "1k");
          (1400., "1.4k");
          (1500., "1.5k");
          (1500.5, "1.5005k");
          (1e-15, "1f");
          (1e-12, "1p");
          (1e-9, "1n");
          (1e-6, "1µ");
          (1e-3, "1m");
          (1e6, "1M");
          (1e9, "1G");
          (1e12, "1T");
          (1e15, "1P");
        ];
      table "a prefix chosen after rounding" Si (Decimals 1)
        [ (999.96, "1.0k"); (999.94, "999.9"); (-999.96, "−1.0k") ];
    ]

let percent =
  group "percent"
    [
      table "no decimal" Percent (Decimals 0)
        [
          (0., "0%");
          (0.042, "4%");
          (0.42, "42%");
          (4.2, "420%");
          (-0.042, "−4%");
          (-0.42, "−42%");
          (-4.2, "−420%");
        ];
      table "one decimal" Percent (Decimals 1)
        [ (0.234, "23.4%"); (0.125, "12.5%") ];
      table "two decimals" Percent (Decimals 2) [ (0.234, "23.40%") ];
      table "trimmed" ~trim:true Percent (Decimals 6)
        [
          (0., "0%");
          (0.1, "10%");
          (0.01, "1%");
          (0.001, "0.1%");
          (0.0001, "0.01%");
        ];
      table "grouped" ~group:true Percent (Decimals 0)
        [ (422., "42,200%"); (-422., "−42,200%") ];
    ]

let rounding =
  group "rounding"
    [
      table "halves go away from zero" Plain (Decimals 2)
        [ (0.125, "0.13"); (-0.125, "−0.13"); (0.375, "0.38") ];
      table "halves to no decimal" Plain (Decimals 0)
        [ (2.5, "3"); (-2.5, "−3"); (0.5, "1"); (1.5, "2") ];
      table "digits are the binary value's" Plain (Decimals 20)
        [ (0.1, "0.10000000000000000555"); (0.3, "0.29999999999999998890") ];
      table "zeros are unsigned" Plain (Decimals 2)
        [
          (-0., "0.00"); (-0.001, "0.00"); (-0.00499, "0.00"); (-0.005, "−0.01");
        ];
      table "trimming leaves no separator" ~trim:true Plain (Decimals 2)
        [ (1.5, "1.5"); (2., "2"); (-0.001, "0") ];
      test "the largest float is written in full" (fun () ->
          equal string
            "179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368"
            (write Plain (Decimals 0) Float.max_float));
      test "the least subnormal is written in full" (fun () ->
          let s = write Plain (Decimals 1074) 5e-324 in
          starts_with
            ~affix:
              ("0." ^ String.make 323 '0'
             ^ "494065645841246544176568792868221372365059802")
            s;
          equal int 1076 (String.length s);
          ends_with ~affix:"3447265625" s);
    ]

let non_finite =
  let formats =
    [
      ("plain", Number.v Plain (Decimals 2));
      ("exponent", Number.v Exponent (Significant 3));
      ("si", Number.v ~trim:true Si (Decimals 0));
      ("percent", Number.v ~group:true Percent (Decimals 1));
    ]
  in
  cases ~name:fst "non-finite numbers in" formats (fun (_, f) ->
      equal string "NaN" (Number.to_string f Float.nan);
      equal string "∞" (Number.to_string f Float.infinity);
      equal string "−∞" (Number.to_string f Float.neg_infinity);
      equal string "-∞" (Number.to_string ~locale:ascii f Float.neg_infinity))

let locales =
  let fr = Locale.v ~decimal:"," ~group:"\u{202F}" () in
  let indian = Locale.v ~grouping:[ 3; 2 ] () in
  group "locales"
    [
      test "write separators and minus" (fun () ->
          equal string "−1\u{202F}234\u{202F}567,89"
            (write ~locale:fr ~group:true Plain (Decimals 2) (-1234567.891)));
      test "group in sizes that repeat the last" (fun () ->
          equal string "12,34,56,789"
            (write ~locale:indian ~group:true Plain (Decimals 0) 123456789.));
      test "group with an empty separator" (fun () ->
          equal string "1234567"
            (write ~locale:(Locale.v ~group:"" ()) ~group:true Plain
               (Decimals 0) 1234567.));
      test "write the decimal separator in exponent mantissas" (fun () ->
          equal string "1,5×10³" (write ~locale:fr Exponent (Decimals 1) 1500.));
      test "do not group a short integer part" (fun () ->
          equal string "999.5" (write ~group:true Plain (Decimals 1) 999.5));
    ]

(* Laws *)

let finite = Gen.float
let gen_digits = Gen.int_range 0 20

(* Written zeros have no sign. *)
let unsigned x = if x = 0. then 0. else x

(* A positional number written with an ASCII minus is OCaml float syntax. *)
let value_of = float_of_string

let laws =
  group "laws"
    [
      prop
        "a number written to 1074 decimals reads back as itself, zeros unsigned"
        finite (fun x ->
          equal float_exact (unsigned x)
            (value_of (write ~locale:ascii Plain (Decimals 1074) x)));
      prop "writing is monotone at a fixed precision"
        (Gen.triple gen_digits
           (Gen.float_range (-1e6) 1e6)
           (Gen.float_range (-1e6) 1e6))
        (fun (d, x, y) ->
          let x, y = if x <= y then (x, y) else (y, x) in
          let w v = value_of (write ~locale:ascii Plain (Decimals d) v) in
          at_most float_exact ~than:(w y) (w x));
      prop "a written number is within half a unit of the last digit"
        (Gen.pair gen_digits (Gen.float_range (-1e6) 1e6))
        (fun (d, x) ->
          let w = value_of (write ~locale:ascii Plain (Decimals d) x) in
          let unit = 10. ** Float.of_int (-d) in
          (* Reading [w] back and subtracting each round by an ulp. *)
          let slack = 2. *. (Float.succ (Float.abs x) -. Float.abs x) in
          at_most float_exact ~than:((unit /. 2.) +. slack) (Float.abs (w -. x)));
      prop "trimming only removes zeros"
        (Gen.pair gen_digits (Gen.float_range (-1e6) 1e6))
        (fun (d, x) ->
          let full = write Plain (Decimals d) x
          and trimmed = write ~trim:true Plain (Decimals d) x in
          starts_with ~affix:trimmed full;
          let rest =
            String.sub full (String.length trimmed)
              (String.length full - String.length trimmed)
          in
          is_true ~msg:rest (String.for_all (fun c -> c = '0' || c = '.') rest));
      prop "an SI quotient lies in [1;1000[ between the extreme prefixes"
        (Gen.float_range 1e-29 1e32) (fun x ->
          let s = write ~locale:ascii Si (Significant 4) x in
          let n = String.length s in
          let digits_end = ref n in
          while
            !digits_end > 0
            && not (String.contains "0123456789." s.[!digits_end - 1])
          do
            decr digits_end
          done;
          let q = float_of_string (String.sub s 0 !digits_end) in
          at_least float_exact ~than:1. q;
          less float_exact ~than:1000. q);
    ]

(* Formats *)

let errors =
  group "errors"
    [
      test "v raises on negative decimals" (fun () ->
          raises_match (Exn.invalid_arg ?substring:None) (fun () ->
              Number.v Plain (Decimals (-1))));
      test "v raises below one significant digit" (fun () ->
          raises_match (Exn.invalid_arg ?substring:None) (fun () ->
              Number.v Si (Significant 0)));
    ]

let format = Testable.make ~pp:Number.pp ~equal:Number.equal

let gen_format =
  let open Gen in
  let+ n = of_list [ Number.Plain; Exponent; Si; Percent ]
  and+ significant = bool
  and+ k = int_range 1 3
  and+ trim = bool
  and+ group = bool in
  Number.v ~trim ~group n (if significant then Significant k else Decimals k)

let gen_format = Gen.with_pp Number.pp gen_format

let comparing =
  group "equal"
    [
      test "formats differing in trimming differ" (fun () ->
          not_equal format
            (Number.v Plain (Decimals 2))
            (Number.v ~trim:true Plain (Decimals 2)));
      test "precisions differ by kind" (fun () ->
          not_equal format
            (Number.v Plain (Decimals 2))
            (Number.v Plain (Significant 2)));
      prop "is an equivalence"
        (Gen.pair gen_format gen_format)
        (Law.equivalence format);
      test "equal formats are equal" (fun () ->
          equal format
            (Number.v ~group:true Si (Significant 3))
            (Number.v ~group:true Si (Significant 3)));
    ]

(* Decimals of a source dtype *)

let dtype_name : type a b. (a, b) Nx.dtype -> string = function
  | Float64 -> "float64"
  | Float32 -> "float32"
  | Float16 -> "float16"
  | BFloat16 -> "bfloat16"
  | Float8_e4m3 -> "float8_e4m3"
  | Float8_e5m2 -> "float8_e5m2"
  | _ -> "other"

(* [to_dtype dtype x] is [x] rounded to [dtype] by nx. *)
let to_dtype (type b) (dtype : (float, b) Nx.dtype) x =
  Nx.item [] (Nx.cast Nx.float64 (Nx.cast dtype (Nx.scalar Nx.float64 x)))

type float_dtype = D : (float, 'b) Nx.dtype -> float_dtype

let float_dtypes =
  [
    D Nx.float64;
    D Nx.float32;
    D Nx.float16;
    D Nx.bfloat16;
    D Nx.float8_e4m3;
    D Nx.float8_e5m2;
  ]

let gen_dtype =
  Gen.with_pp
    (fun ppf (D d) -> Format.pp_print_string ppf (dtype_name d))
    (Gen.of_list float_dtypes)

(* [written dtype d x] is [x] written with [d] decimals, read back as a float64
   and rounded to [dtype], or [None] if that float64 is so near a midpoint of
   [dtype] that the second rounding may differ from rounding the decimal
   directly. *)
let written (D dtype) d x =
  let w = float_of_string (write ~locale:ascii Plain (Decimals d) x) in
  match dtype with
  | Nx.Float64 -> Some w
  | _ ->
      let r = to_dtype dtype in
      if Float.equal (r (Float.pred w)) (r (Float.succ w)) then Some (r w)
      else None

let decimals =
  group "decimals"
    [
      test "is 1 for 0.1 in float64" (fun () ->
          equal int 1 (Number.decimals Nx.float64 0.1));
      test "is 4 for the float32 nearest 0.9234" (fun () ->
          equal int 4 (Number.decimals Nx.float32 (to_dtype Nx.float32 0.9234)));
      test "is 17 for 0.1 +. 0.2 in float64" (fun () ->
          equal int 17 (Number.decimals Nx.float64 (0.1 +. 0.2)));
      test "is 1 for the float16 nearest 0.1" (fun () ->
          equal int 1 (Number.decimals Nx.float16 (to_dtype Nx.float16 0.1)));
      test "is 1 for the bfloat16 nearest 0.1" (fun () ->
          equal int 1 (Number.decimals Nx.bfloat16 (to_dtype Nx.bfloat16 0.1)));
      test "reads a float64 value in a narrower dtype" (fun () ->
          equal int 1 (Number.decimals Nx.float32 0.1));
      cases
        ~name:(fun x -> Printf.sprintf "%h" x)
        "is 0 for"
        [ 0.; 3.; -1e300; Float.nan; Float.infinity ]
        (fun x -> equal int 0 (Number.decimals Nx.float64 x));
      test "rounds a float16 tie to even, down" (fun () ->
          equal int 0 (Number.decimals Nx.float16 (1. +. (2. ** -11.))));
      test "rounds a float16 tie to even, up" (fun () ->
          equal int 4 (Number.decimals Nx.float16 (1. +. (3. *. (2. ** -11.)))));
      test "reads the least float16 subnormal" (fun () ->
          equal int 8 (Number.decimals Nx.float16 (2. ** -24.)));
      test "rounds into the float16 subnormals" (fun () ->
          equal int 7 (Number.decimals Nx.float16 1e-7));
      test "saturates float8 e4m3 at 448" (fun () ->
          equal int 0 (Number.decimals Nx.float8_e4m3 470.3));
      test "saturates float8 e5m2 at 57344" (fun () ->
          (* 61440 saturates to 57344; without saturation it would be the
             midpoint above 57344, whose mantissa is odd, and round away. *)
          equal int 0 (Number.decimals Nx.float8_e5m2 61500.3);
          equal int 0 (Number.decimals Nx.float8_e5m2 61439.9));
      test "reads a float16 value whose integer rounds to infinity" (fun () ->
          equal int 1 (Number.decimals Nx.float16 65519.7));
      test "bounds a float8 e4m3 value above" (fun () ->
          equal int 2 (Number.decimals Nx.float8_e4m3 1.18));
      test "is 0 for non-finite values of a saturating dtype" (fun () ->
          equal int 0 (Number.decimals Nx.float8_e4m3 Float.nan);
          equal int 0 (Number.decimals Nx.float8_e4m3 Float.neg_infinity));
      test "is 0 for a fraction that overflows float16" (fun () ->
          equal int 0 (Number.decimals Nx.float16 100000.5));
      test "is 0 for a value that overflows float16" (fun () ->
          equal int 0 (Number.decimals Nx.float16 1e6));
      test "is 0 in every integer dtype" (fun () ->
          equal int 0 (Number.decimals Nx.int32 0.5);
          equal int 0 (Number.decimals Nx.uint8 0.25));
      test "raises on complex and boolean dtypes" (fun () ->
          raises_match (Exn.invalid_arg ?substring:None) (fun () ->
              Number.decimals Nx.complex64 0.5);
          raises_match (Exn.invalid_arg ?substring:None) (fun () ->
              Number.decimals Nx.bool 0.5));
      prop "reproduces a value of its dtype and no fewer decimals do"
        (Gen.pair gen_dtype (Gen.float_range (-1e4) 1e4))
        (fun ((D dtype as dt), x) ->
          let x = to_dtype dtype x in
          assume (Float.is_finite x);
          let d = Number.decimals dtype x in
          (match written dt d x with
          | Some y -> equal float_exact (unsigned x) y
          | None -> reject ());
          if d > 0 then
            match written dt (d - 1) x with
            | Some y -> not_equal float_exact x y
            | None -> reject ());
    ]

let () =
  exit
    (run "Number"
       [
         plain;
         exponent;
         si;
         percent;
         rounding;
         non_finite;
         locales;
         laws;
         errors;
         comparing;
         decimals;
       ])
