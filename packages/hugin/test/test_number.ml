(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_kit

let ascii = Locale.v ~minus:"-" ()
let invalid f = raises_match (Exn.invalid_arg ?substring:None) f

let write ?locale ?trim ?group n p x =
  Number.to_string ?locale (Number.v ?trim ?group n p) x

(* A positional number written with an ASCII minus is OCaml float syntax. *)
let value ?trim d x =
  float_of_string (write ~locale:ascii ?trim Plain (Decimals d) x)

(* Written zeros have no sign. *)
let unsigned x = if x = 0. then 0. else x

(* Rows [(x, expected)] written in one format; each row names its input with
   every digit. *)
let table name ?locale ?trim ?group n p rows =
  cases
    ~name:(fun (x, e) -> Printf.sprintf "%.17g is %s" x e)
    name rows
    (fun (x, e) -> equal string e (write ?locale ?trim ?group n p x))

(* Notations. Most rows are d3-format's tests (format-type-*-test.js,
   format-trim-test.js), whose digits follow the same rule, written with the
   kit's exponent and prefix forms. *)

let plain =
  group "plain"
    [
      table "pads to its decimals" Plain (Decimals 2)
        [ (100., "100.00"); (0.449, "0.45"); (-0.429, "−0.43") ];
      table "rounds to its decimals" Plain (Decimals 1)
        [ (0.49, "0.5"); (123456.49, "123456.5") ];
      table "groups its integer digits" ~group:true Plain (Decimals 2)
        [ (1234567.449, "1,234,567.45"); (1e4, "10,000.00"); (999.5, "999.50") ];
      table "counts significant digits from the first nonzero one" Plain
        (Significant 3)
        [ (1234.5, "1230"); (0.0012345, "0.00123"); (9.996, "10.0") ];
      table "writes zero with one decimal fewer than its significant digits"
        Plain (Significant 3)
        [ (0., "0.00") ];
      table "trims significant digits" ~trim:true Plain (Significant 6)
        [
          (1., "1");
          (10.0001, "10.0001");
          (123.4567, "123.457");
          (0.0000009, "0.0000009");
          (0.1111119, "0.111112");
        ];
      table "groups and trims" ~group:true ~trim:true Plain (Significant 6)
        [ (10000., "10,000"); (10000.1, "10,000.1") ];
    ]

let exponent =
  group "exponent"
    [
      table "writes a mantissa and a superscript exponent" Exponent (Decimals 6)
        [
          (42., "4.200000×10¹");
          (-4., "−4.000000×10⁰");
          (-42000000., "−4.200000×10⁷");
          (-1e-12, "−1.000000×10⁻¹²");
          (0., "0.000000");
          (-0., "0.000000");
        ];
      table "leaves out a mantissa written 1" Exponent (Decimals 0)
        [ (42., "4×10¹"); (1000., "10³"); (-1000., "−10³") ];
      table "trims" ~trim:true Exponent (Decimals 6)
        [
          (0., "0");
          (0.042, "4.2×10⁻²");
          (420000000., "4.2×10⁸");
          (42000000000., "4.2×10¹⁰");
          (0.00000000042, "4.2×10⁻¹⁰");
          (5e-7, "5×10⁻⁷");
          (1e4, "10⁴");
          (1e-9, "10⁻⁹");
          (-0.00000000012345, "−1.2345×10⁻¹⁰");
        ];
      table "carries a mantissa rounding to ten" Exponent (Decimals 1)
        [ (9.96, "1.0×10¹"); (-9.96, "−1.0×10¹"); (99999., "1.0×10⁵") ];
      table "counts significant digits in the mantissa" Exponent (Significant 3)
        [ (1234.5, "1.23×10³"); (9.996, "1.00×10¹"); (0., "0.00") ];
    ]

let prefixes =
  [ "q"; "r"; "y"; "z"; "a"; "f"; "p"; "n"; "µ"; "m"; ""; "k"; "M"; "G"; "T" ]
  @ [ "P"; "E"; "Z"; "Y"; "R"; "Q" ]

let si =
  group "si"
    [
      cases
        ~name:(fun (k, p) -> Printf.sprintf "1e%d is 1%s" (3 * k) p)
        "names each power of 1000"
        (List.mapi (fun i p -> (i - 10, p)) prefixes)
        (fun (k, p) ->
          let x = float_of_string (Printf.sprintf "1e%d" (3 * k)) in
          equal string ("1" ^ p) (write ~trim:true Si (Significant 3) x));
      table "keeps the quotient in [1;1000[" Si (Significant 6)
        [
          (999.5, "999.500");
          (1500.5, "1.50050k");
          (0.00001, "10.0000µ");
          (0., "0.00000");
        ];
      table "chooses the prefix after rounding" Si (Significant 3)
        [
          (999.5, "1.00k");
          (999500., "1.00M");
          (145999999.99999347, "146M");
          (0.009995, "10.0m");
        ];
      table "chooses after rounding to decimals" Si (Decimals 1)
        [ (999.96, "1.0k"); (999.94, "999.9"); (-999.96, "−1.0k") ];
      table "takes the nearer end beyond the prefixes" Si (Significant 8)
        [
          (1.29e-33, "0.0012900000q");
          (1.23e31, "12.300000Q");
          (1e33, "1000.0000Q");
          (-1.29e-24, "−1.2900000y");
        ];
    ]

let percent =
  group "percent"
    [
      table "writes a hundred times the number" Percent (Decimals 0)
        [ (0., "0%"); (0.042, "4%"); (-4.2, "−420%") ];
      table "with its decimals" Percent (Decimals 1) [ (0.125, "12.5%") ];
      table "trims" ~trim:true Percent (Decimals 6)
        [ (0.1, "10%"); (0.001, "0.1%") ];
      table "groups" ~group:true Percent (Decimals 0) [ (-422., "−42,200%") ];
    ]

(* How numbers are written *)

let rounding =
  group "rounding"
    [
      table "halves go away from zero" Plain (Decimals 2)
        [ (0.125, "0.13"); (-0.125, "−0.13"); (0.375, "0.38") ];
      table "halves to no decimal" Plain (Decimals 0)
        [ (2.5, "3"); (-2.5, "−3"); (0.5, "1") ];
      table "digits are the binary value's" Plain (Decimals 20)
        [ (0.1, "0.10000000000000000555"); (0.3, "0.29999999999999998890") ];
      table "zeros are unsigned" Plain (Decimals 2)
        [
          (-0., "0.00"); (-0.001, "0.00"); (-0.00499, "0.00"); (-0.005, "−0.01");
        ];
      table "trimming leaves no separator" ~trim:true Plain (Decimals 2)
        [ (2., "2"); (-0.001, "0") ];
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
      cases ~name:fst "non-finite numbers in"
        [
          ("plain", Number.v Plain (Decimals 2));
          ("exponent", Number.v Exponent (Significant 3));
          ("si", Number.v ~trim:true Si (Decimals 0));
          ("percent", Number.v ~group:true Percent (Decimals 1));
        ]
        (fun (_, f) ->
          equal string "NaN" (Number.to_string f Float.nan);
          equal string "∞" (Number.to_string f Float.infinity);
          equal string "−∞" (Number.to_string f Float.neg_infinity);
          equal string "-∞"
            (Number.to_string ~locale:ascii f Float.neg_infinity));
    ]

let locales =
  let fr = Locale.v ~decimal:"," ~group:"\u{202F}" () in
  group "locales"
    [
      test "write separators and minus" (fun () ->
          equal string "−1\u{202F}234\u{202F}567,89"
            (write ~locale:fr ~group:true Plain (Decimals 2) (-1234567.891)));
      test "group in sizes that repeat the last" (fun () ->
          equal string "12,34,56,789"
            (write
               ~locale:(Locale.v ~grouping:[ 3; 2 ] ())
               ~group:true Plain (Decimals 0) 123456789.));
      test "group with an empty separator" (fun () ->
          equal string "1234567"
            (write ~locale:(Locale.v ~group:"" ()) ~group:true Plain
               (Decimals 0) 1234567.));
      test "write the decimal separator in exponent mantissas" (fun () ->
          equal string "1,5×10³" (write ~locale:fr Exponent (Decimals 1) 1500.));
    ]

(* Laws *)

let gen_decimals = Gen.int_range 0 20
let gen_moderate = Gen.float_range (-1e6) 1e6

(* [si_quotient s] is the number [s] writes before its prefix. *)
let si_quotient s =
  let rec digits_end i =
    if i > 0 && not (String.contains "0123456789." s.[i - 1]) then
      digits_end (i - 1)
    else i
  in
  float_of_string (String.sub s 0 (digits_end (String.length s)))

let laws =
  group "laws"
    [
      prop "a number written to 1074 decimals reads back as itself"
        ~examples:[ Float.max_float; 5e-324; Float.min_float; -0. ] Gen.float
        (fun x -> equal float_exact (unsigned x) (value 1074 x));
      prop "writing is monotone at a fixed precision"
        (Gen.triple gen_decimals gen_moderate gen_moderate) (fun (d, x, y) ->
          let x, y = if x <= y then (x, y) else (y, x) in
          at_most float_exact ~than:(value d y) (value d x));
      prop "a written number is within half a unit of its last digit"
        (Gen.pair gen_decimals gen_moderate) (fun (d, x) ->
          (* Reading back and subtracting each round by an ulp. *)
          let slack = 2. *. (Float.succ (Float.abs x) -. Float.abs x) in
          at_most float_exact
            ~than:((0.5 *. (10. ** Float.of_int (-d))) +. slack)
            (Float.abs (value d x -. x)));
      prop "trimming removes the zeros that end the decimals, and the separator"
        (Gen.pair gen_decimals gen_moderate) (fun (d, x) ->
          let trimmed = write ~trim:true Plain (Decimals d) x in
          starts_with ~affix:trimmed (write Plain (Decimals d) x);
          equal float_exact (value d x) (value ~trim:true d x);
          if String.contains trimmed '.' then
            not_equal char '0' trimmed.[String.length trimmed - 1]);
      prop "an SI quotient lies in [1;1000[ between the extreme prefixes"
        (Gen.float_range 1e-29 1e32) (fun x ->
          let q = si_quotient (write Si (Significant 4) x) in
          at_least float_exact ~than:1. q;
          less float_exact ~than:1000. q);
    ]

(* Formats *)

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

let formats =
  group "formats"
    [
      cases ~name:fst "v raises on"
        [
          ("negative decimals", fun () -> Number.v Plain (Decimals (-1)));
          ("no significant digit", fun () -> Number.v Si (Significant 0));
        ]
        (fun (_, f) -> invalid f);
      prop "equal is an equivalence"
        (Gen.pair gen_format gen_format)
        (Law.equivalence format);
      test "equal tells trimming apart" (fun () ->
          not_equal format
            (Number.v Plain (Decimals 2))
            (Number.v ~trim:true Plain (Decimals 2)));
      test "equal tells the kinds of precision apart" (fun () ->
          not_equal format
            (Number.v Plain (Decimals 2))
            (Number.v Plain (Significant 2)));
    ]

(* Decimals of a source dtype *)

type float_dtype = D : (float, 'b) Nx.dtype -> float_dtype

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

(* [written dtype d x] is [x] written with [d] decimals, read back as a float64
   and rounded to [dtype], or [None] if that float64 is so near a midpoint of
   [dtype] that the second rounding may differ from rounding the decimal
   directly. *)
let written (D dtype) d x =
  let w = value d x in
  match dtype with
  | Nx.Float64 -> Some w
  | _ ->
      let r = to_dtype dtype in
      if Float.equal (r (Float.pred w)) (r (Float.succ w)) then Some (r w)
      else None

let gen_source =
  let open Gen in
  let+ dtype =
    of_list
      [
        D Nx.float64;
        D Nx.float32;
        D Nx.float16;
        D Nx.bfloat16;
        D Nx.float8_e4m3;
        D Nx.float8_e5m2;
      ]
  and+ x = float_range (-1e4) 1e4
  and+ exact = bool in
  let (D d) = dtype in
  (dtype, if exact then to_dtype d x else x)

let gen_source =
  Gen.with_pp
    (fun ppf (D d, x) -> Format.fprintf ppf "%s %h" (dtype_name d) x)
    gen_source

(* Values at the dtypes' ties, subnormals and saturation. *)
let source_edges =
  [
    (D Nx.float16, 1. +. (2. ** -11.));
    (D Nx.float16, 1. +. (3. *. (2. ** -11.)));
    (D Nx.float16, 2. ** -24.);
    (D Nx.float16, 1e-7);
    (D Nx.float16, 65519.7);
    (D Nx.float16, 100000.5);
    (D Nx.float16, 1e6);
    (D Nx.float8_e4m3, 470.3);
    (D Nx.float8_e4m3, 1.18);
    (D Nx.float8_e5m2, 61500.3);
    (D Nx.float8_e5m2, 61439.9);
    (D Nx.float32, 0.1);
  ]

let reproduces ((D dtype as dt), x) =
  let d = Number.decimals dtype x in
  (match written dt d x with
  | Some y -> equal float_exact (unsigned (to_dtype dtype x)) y
  | None -> reject ());
  if d > 0 then
    match written dt (d - 1) x with
    | Some y -> not_equal float_exact (to_dtype dtype x) y
    | None -> reject ()

let decimals =
  group "decimals"
    [
      prop "is the fewest decimals that reproduce a value in its dtype"
        ~examples:source_edges gen_source reproduces;
      test "is 1 for 0.1 in float64" (fun () ->
          equal int 1 (Number.decimals Nx.float64 0.1));
      test "is 4 for the float32 nearest 0.9234" (fun () ->
          equal int 4 (Number.decimals Nx.float32 (to_dtype Nx.float32 0.9234)));
      cases
        ~name:(fun x -> Printf.sprintf "%h" x)
        "is 0 for integers and non-finite values"
        [ 0.; 3.; -1e300; Float.nan; Float.infinity; Float.neg_infinity ]
        (fun x ->
          equal int 0 (Number.decimals Nx.float64 x);
          equal int 0 (Number.decimals Nx.float8_e4m3 x));
      test "is 0 in every integer dtype" (fun () ->
          equal int 0 (Number.decimals Nx.int32 0.5);
          equal int 0 (Number.decimals Nx.uint8 0.25));
      test "raises on complex and boolean dtypes" (fun () ->
          invalid (fun () -> Number.decimals Nx.complex64 0.5);
          invalid (fun () -> Number.decimals Nx.bool 0.5));
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
         locales;
         laws;
         formats;
         decimals;
       ])
