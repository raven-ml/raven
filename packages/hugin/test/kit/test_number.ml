(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_kit

let ascii = Locale.v ~minus:"-" ()
let invalid f = raises_match (Exn.invalid_arg ?sub:None) f

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
  cases name
    ~id:(fun (x, e) -> Printf.sprintf "%.17g is %s" x e)
    rows
    (fun (x, e) -> equal string e (write ?locale ?trim ?group n p x))

let gen_decimals = Gen.int_range 0 20
let gen_moderate = Gen.float_range (-1e6) 1e6

(* How numbers are written. Most rows of this suite are d3-format's tests
   (format-type-*-test.js, format-trim-test.js), whose digits follow the same
   rule, written with the kit's exponent and prefix forms. *)

let digits =
  group "digits"
    [
      cases "round halves away from zero"
        ~id:(fun (d, x, e) -> Printf.sprintf "%.17g to %d decimals is %s" x d e)
        [
          (2, 0.125, "0.13");
          (2, -0.125, "−0.13");
          (2, 0.375, "0.38");
          (0, 2.5, "3");
          (0, -2.5, "−3");
          (0, 0.5, "1");
        ]
        (fun (d, x, e) -> equal string e (write Plain (Decimals d) x));
      table "are the binary value's decimal expansion" Plain (Decimals 20)
        [ (0.1, "0.10000000000000000555"); (0.3, "0.29999999999999998890") ];
      (* Python's ['%.0f' % sys.float_info.max]. *)
      test "reach every digit of the largest float" (fun () ->
          equal string
            "179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368"
            (write Plain (Decimals 0) Float.max_float));
      (* 2^-1074 is 5^1074 / 10^1074: 323 zeros, then the 751 digits of 5^1074
         (Python's [str(5**1074)]). *)
      test "reach every digit of the least subnormal" (fun () ->
          equal string
            ("0." ^ String.make 323 '0'
           ^ "4940656458412465441765687928682213723650598026143247644255856825006755072702087518652998363616359923797965646954457177309266567103559397963987747960107818781263007131903114045278458171678489821036887186360569987307230500063874091535649843873124733972731696151400317153853980741262385655911710266585566867681870395603106249319452715914924553293054565444011274801297099995419319894090804165633245247571478690147267801593552386115501348035264934720193790268107107491703332226844753335720832431936092382893458368060106011506169809753078342277318329247904982524730776375927247874656084778203734469699533647017972677717585125660551199131504891101451037862738167250955837389733598993664809941164205702637090279242767544565229087538682506419718265533447265625"
            )
            (write Plain (Decimals 1074) 5e-324));
      prop "written to 1074 decimals read back as the number"
        ~examples:[ Float.max_float; 5e-324; Float.min_float; -0. ] Gen.float
        (fun x -> equal float_exact (unsigned x) (value 1074 x));
      prop "do not decrease as the number grows"
        (Gen.pair gen_decimals (Gen.pair gen_moderate gen_moderate))
        (fun (d, xy) -> Law.monotone float_exact float_exact (value d) xy);
      prop "lie within half a unit of the last digit"
        (Gen.pair gen_decimals gen_moderate) (fun (d, x) ->
          (* Reading back and subtracting each round by an ulp. *)
          let slack = 2. *. (Float.succ (Float.abs x) -. Float.abs x) in
          at_most float_exact
            ~than:((0.5 *. (10. ** Float.of_int (-d))) +. slack)
            (Float.abs (value d x -. x)));
    ]

let signs =
  group "signs"
    [
      table "a negative number has the minus sign unless every digit is zero"
        Plain (Decimals 2)
        [
          (-0., "0.00");
          (-0.001, "0.00");
          (-0.00499, "0.00");
          (-0.005, "−0.01");
          (-0.429, "−0.43");
        ];
    ]

let trimming =
  group "trimming"
    [
      table "removes the separator when no digit follows it" ~trim:true Plain
        (Decimals 2)
        [ (2., "2"); (-0.001, "0") ];
      prop "removes the zeros that end the decimals and nothing else"
        (Gen.pair gen_decimals gen_moderate) (fun (d, x) ->
          let trimmed = write ~trim:true Plain (Decimals d) x in
          let last = trimmed.[String.length trimmed - 1] in
          let separated = String.contains trimmed '.' in
          cover "a separator is kept" separated;
          starts_with ~prefix:trimmed (write Plain (Decimals d) x);
          equal float_exact (value d x) (value ~trim:true d x);
          not_equal char '.' last;
          if separated then not_equal char '0' last);
    ]

let grouping =
  group "grouping"
    [
      table "separates the integer digits" ~group:true Plain (Decimals 2)
        [ (1234567.449, "1,234,567.45"); (1e4, "10,000.00"); (999.5, "999.50") ];
      table "separates the digits of a trimmed number" ~group:true ~trim:true
        Plain (Significant 6)
        [ (10000., "10,000"); (10000.1, "10,000.1") ];
      table "separates the digits of a percentage" ~group:true Percent
        (Decimals 0)
        [ (-422., "−42,200%") ];
    ]

let every_notation =
  [
    ("plain", Number.v Plain (Decimals 2));
    ("exponent", Number.v Exponent (Significant 3));
    ("si", Number.v ~trim:true Si (Decimals 0));
    ("percent", Number.v ~group:true Percent (Decimals 1));
  ]

let non_finite =
  group "non-finite numbers"
    [
      cases "are NaN, ∞ and the locale's minus sign then ∞ in every notation"
        ~id:(fun (n, x, minus, _) -> Printf.sprintf "%s %h, minus %s" n x minus)
        (List.concat_map
           (fun (n, _) ->
             [
               (n, Float.nan, "−", "NaN");
               (n, Float.infinity, "−", "∞");
               (n, Float.neg_infinity, "−", "−∞");
               (n, Float.neg_infinity, "-", "-∞");
             ])
           every_notation)
        (fun (n, x, minus, e) ->
          let locale = Locale.v ~minus () in
          equal string e
            (Number.to_string ~locale (List.assoc n every_notation) x));
    ]

(* Formats *)

let plain =
  group "Plain"
    [
      table "pads its decimals with zeros" Plain (Decimals 2)
        [ (100., "100.00"); (0.449, "0.45") ];
      table "rounds to its decimals" Plain (Decimals 1)
        [ (0.49, "0.5"); (123456.49, "123456.5") ];
    ]

let exponent =
  group "Exponent"
    [
      table "writes a mantissa and a superscript exponent" Exponent (Decimals 6)
        [
          (42., "4.200000×10¹");
          (-4., "−4.000000×10⁰");
          (-42000000., "−4.200000×10⁷");
          (-1e-12, "−1.000000×10⁻¹²");
        ];
      table "writes zero as Plain does" Exponent (Decimals 6)
        [ (0., "0.000000"); (-0., "0.000000") ];
      table "leaves out a mantissa written 1" Exponent (Decimals 0)
        [ (42., "4×10¹"); (1000., "10³"); (-1000., "−10³") ];
      table "trims its mantissa" ~trim:true Exponent (Decimals 6)
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
    ]

let prefixes =
  [ "q"; "r"; "y"; "z"; "a"; "f"; "p"; "n"; "µ"; "m"; ""; "k"; "M"; "G"; "T" ]
  @ [ "P"; "E"; "Z"; "Y"; "R"; "Q" ]

(* [si_parts s] is the quotient [s] writes and the power [K] of 1000 its prefix
   names. *)
let si_parts s =
  let rec digits_end i =
    if i > 0 && not (String.contains "0123456789." s.[i - 1]) then
      digits_end (i - 1)
    else i
  in
  let n = digits_end (String.length s) in
  let prefix = String.sub s n (String.length s - n) in
  let i =
    require_some ~msg:("a prefix of " ^ s)
      (List.find_index (String.equal prefix) prefixes)
  in
  (float_of_string (String.sub s 0 n), i - 10)

(* Numbers of either sign whose binary exponents spread evenly over the
   prefixes' range, from 2^-99 (above 1e-30) to below 2^109 (below 1e33). *)
let gen_si =
  let open Gen in
  let+ m = float_range 1. 2.
  and+ e = int_range (-99) 107
  and+ negative = bool in
  let x = Float.ldexp m e in
  if negative then -.x else x

let si =
  group "Si"
    [
      cases "names each power of 1000"
        ~id:(fun (k, p) -> Printf.sprintf "1e%d is 1%s" (3 * k) p)
        (List.mapi (fun i p -> (i - 10, p)) prefixes)
        (fun (k, p) ->
          let x = float_of_string (Printf.sprintf "1e%d" (3 * k)) in
          equal string ("1" ^ p) (write ~trim:true Si (Significant 3) x));
      table "keeps the quotient in [1;1000[" Si (Significant 6)
        [ (999.5, "999.500"); (1500.5, "1.50050k"); (0.00001, "10.0000µ") ];
      prop "writes the number as a quotient in [1;1000[ times its prefix"
        (Gen.with_pp (fun ppf x -> Format.fprintf ppf "%h" x) gen_si)
        (fun x ->
          cover "below 1" (Float.abs x < 1.);
          cover "from 1000 up" (Float.abs x >= 1000.);
          cover "negative" (x < 0.);
          let q, k = si_parts (write ~locale:ascii Si (Significant 4) x) in
          let q = Float.abs q in
          let y = Float.abs x /. (10. ** Float.of_int (3 * k)) in
          (* Four significant digits put [q] within half a unit of the fourth
             digit of [y]. Computing [y] (a power within an ulp, a division
             within half) and reading [q] (within half) err by at most 2 ulps of
             [y], 4 across a binade. *)
          let half_unit = 0.5 *. (10. ** (Float.floor (Float.log10 y) -. 3.)) in
          let slack = 4. *. (Float.succ y -. y) in
          at_least float_exact ~than:1. q;
          less float_exact ~than:1000. q;
          at_most float_exact ~than:(half_unit +. slack) (Float.abs (q -. y)));
      table "writes zero as Plain does" Si (Significant 6) [ (0., "0.00000") ];
      table "chooses the prefix after rounding" Si (Significant 3)
        [
          (999.5, "1.00k");
          (999500., "1.00M");
          (145999999.99999347, "146M");
          (0.009995, "10.0m");
        ];
      table "chooses the prefix after rounding to decimals" Si (Decimals 1)
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
  group "Percent"
    [
      table "writes a hundred times the number" Percent (Decimals 0)
        [ (0., "0%"); (0.042, "4%"); (-4.2, "−420%") ];
      table "trims the percentage" ~trim:true Percent (Decimals 6)
        [ (0.1, "10%"); (0.001, "0.1%") ];
    ]

let decimals_precision =
  group "Decimals"
    [
      cases "counts the digits after the separator of what the notation writes"
        ~id:(fun (n, x, e) ->
          Format.asprintf "%a %.17g is %s" Number.pp_notation n x e)
        [
          (Number.Plain, 100., "100.00");
          (Exponent, 42., "4.20×10¹");
          (Si, 1500., "1.50k");
          (Percent, 0.125, "12.50%");
        ]
        (fun (n, x, e) -> equal string e (write n (Decimals 2) x));
    ]

let significant =
  group "Significant"
    [
      cases "counts digits from the first nonzero one"
        ~id:(fun (n, x, e) ->
          Format.asprintf "%a %.17g is %s" Number.pp_notation n x e)
        [
          (Number.Plain, 1234.5, "1230");
          (Plain, 0.0012345, "0.00123");
          (Plain, 9.996, "10.0");
          (Exponent, 1234.5, "1.23×10³");
          (Exponent, 9.996, "1.00×10¹");
          (Si, 1234.5, "1.23k");
        ]
        (fun (n, x, e) -> equal string e (write n (Significant 3) x));
      cases "writes zero with one decimal fewer"
        ~id:(fun (n, _) -> Format.asprintf "%a" Number.pp_notation n)
        [
          (Number.Plain, "0.00");
          (Exponent, "0.00");
          (Si, "0.00");
          (Percent, "0.00%");
        ]
        (fun (n, e) -> equal string e (write n (Significant 3) 0.));
      table "trims significant digits" ~trim:true Plain (Significant 6)
        [
          (1., "1");
          (10.0001, "10.0001");
          (123.4567, "123.457");
          (0.0000009, "0.0000009");
          (0.1111119, "0.111112");
        ];
    ]

(* v *)

let v =
  group "v"
    [
      cases "raises on" ~id:fst
        [
          ("negative decimals", fun () -> Number.v Plain (Decimals (-1)));
          ("no significant digit", fun () -> Number.v Si (Significant 0));
        ]
        (fun (_, f) -> invalid f);
      cases "accepts the least precisions"
        ~id:(fun (_, e) -> e)
        [ (Number.Decimals 0, "1235"); (Significant 1, "1000") ]
        (fun (p, e) -> equal string e (write Plain p 1234.5));
    ]

(* to_string *)

let to_string =
  let fr = Locale.v ~decimal:"," ~group:"\u{202F}" () in
  group "to_string"
    [
      test "writes the locale's separators and minus sign" (fun () ->
          equal string "−1\u{202F}234\u{202F}567,89"
            (write ~locale:fr ~group:true Plain (Decimals 2) (-1234567.891)));
      test "groups in the locale's sizes, repeating the last" (fun () ->
          equal string "12,34,56,789"
            (write
               ~locale:(Locale.v ~grouping:[ 3; 2 ] ())
               ~group:true Plain (Decimals 0) 123456789.));
      test "groups with an empty separator" (fun () ->
          equal string "1234567"
            (write ~locale:(Locale.v ~group:"" ()) ~group:true Plain
               (Decimals 0) 1234567.));
      test "writes the locale's decimal separator in a mantissa" (fun () ->
          equal string "1,5×10³" (write ~locale:fr Exponent (Decimals 1) 1500.));
    ]

(* decimals *)

type float_dtype = D : (float, 'b) Nx.dtype -> float_dtype
type dtype = Any : ('a, 'b) Nx.dtype -> dtype

let float_dtypes =
  [
    D Nx.float64;
    D Nx.float32;
    D Nx.float16;
    D Nx.bfloat16;
    D Nx.float8_e4m3fn;
    D Nx.float8_e5m2;
    D Nx.float4_e2m1fn;
  ]

let name (D d) = Nx.Dtype.name d

(* [to_dtype dtype x] is [x] rounded to [dtype] by nx. *)
let to_dtype (type b) (dtype : (float, b) Nx.dtype) x =
  Nx.item [] (Nx.cast Nx.float64 (Nx.cast dtype (Nx.scalar Nx.float64 x)))

(* [written dtype d x] is [x] written with [d] decimals, read back as a float64
   and rounded to [dtype], or [None] if that float64 is so near a midpoint of
   [dtype] that the second rounding may differ from rounding the decimal
   directly. An integer below 2^53, written with no decimal, reads back exactly,
   so its one rounding is direct. *)
let written (D dtype) d x =
  let w = value d x in
  match dtype with
  | Float64 -> Some w
  | _ when d = 0 && Float.abs w < 0x1p53 -> Some (to_dtype dtype w)
  | _ ->
      let r = to_dtype dtype in
      if Float.equal (r (Float.pred w)) (r (Float.succ w)) then Some (r w)
      else None

let gen_source =
  let open Gen in
  let+ dtype = of_list float_dtypes
  and+ x = float_range (-1e4) 1e4
  and+ exact = bool in
  let (D d) = dtype in
  (dtype, if exact then to_dtype d x else x)

let gen_source =
  Gen.with_pp
    (fun ppf (d, x) -> Format.fprintf ppf "%s %h" (name d) x)
    gen_source

(* Values at the dtypes' ties, subnormals and saturation, and one of each
   dtype. *)
let source_edges =
  [
    (D Nx.float64, 0.3);
    (D Nx.float32, 0.1);
    (D Nx.float16, 1. +. (2. ** -11.));
    (D Nx.float16, 1. +. (3. *. (2. ** -11.)));
    (D Nx.float16, 2. ** -24.);
    (D Nx.float16, 1e-7);
    (D Nx.float16, 65519.7);
    (D Nx.float16, 100000.5);
    (D Nx.float16, 1e6);
    (D Nx.bfloat16, 0.1);
    (D Nx.float8_e4m3fn, 470.3);
    (D Nx.float8_e4m3fn, 1.18);
    (D Nx.float8_e5m2, 61500.3);
    (D Nx.float8_e5m2, 61439.9);
    (D Nx.float4_e2m1fn, 6.5);
    (D Nx.float4_e2m1fn, 0.25);
  ]

let reproduces ((D dtype as dt), x) =
  List.iter (fun d -> cover (name d) (name d = name dt)) float_dtypes;
  let d = Number.decimals dtype x in
  cover "a decimal is needed" (d > 0);
  (match written dt d x with
  | Some y -> equal float_exact (unsigned (to_dtype dtype x)) y
  | None -> reject ());
  if d > 0 then
    match written dt (d - 1) x with
    | Some y -> not_equal float_exact (to_dtype dtype x) y
    | None -> reject ()

let dtype_name (Any d) = Nx.Dtype.name d

let decimals =
  group "decimals"
    [
      prop "is the fewest decimals that reproduce a value in its dtype"
        ~examples:source_edges gen_source reproduces;
      test "is 1 for 0.1 in float64" (fun () ->
          equal int 1 (Number.decimals Nx.float64 0.1));
      test "is 4 for the float32 nearest 0.9234" (fun () ->
          equal int 4 (Number.decimals Nx.float32 (to_dtype Nx.float32 0.9234)));
      cases "is 0 for integers and non-finite values"
        ~id:(fun x -> Printf.sprintf "%h" x)
        [ 0.; 3.; -1e300; Float.nan; Float.infinity; Float.neg_infinity ]
        (fun x ->
          equal int 0 (Number.decimals Nx.float64 x);
          equal int 0 (Number.decimals Nx.float8_e4m3fn x));
      cases "is 0 in every integer dtype" ~id:dtype_name
        Nx.
          [
            Any int4;
            Any uint4;
            Any int8;
            Any uint8;
            Any int16;
            Any uint16;
            Any int32;
            Any uint32;
            Any int64;
            Any uint64;
          ]
        (fun (Any d) -> equal int 0 (Number.decimals d 0.5));
      cases "raises on complex and boolean dtypes" ~id:dtype_name
        Nx.[ Any complex64; Any complex128; Any bool; Any bit ]
        (fun (Any d) -> invalid (fun () -> Number.decimals d 0.5));
    ]

(* equal *)

let format = Testable.make ~pp:Number.pp ~equal:Number.equal

(* What [v] makes a format of: notation, precision, trimming and grouping. *)
let make (n, p, trim, group) = Number.v ~trim ~group n p

let gen_parts =
  let open Gen in
  let+ n = of_list [ Number.Plain; Exponent; Si; Percent ]
  and+ significant = bool
  and+ k = int_range 1 3
  and+ trim = bool
  and+ group = bool in
  (n, (if significant then Number.Significant k else Decimals k), trim, group)

let differences (n, p, t, g) (n', p', t', g') =
  List.length (List.filter not [ n = n'; p = p'; t = t'; g = g' ])

let base = (Number.Plain, Number.Decimals 2, false, false)

let equal_formats =
  group "equal"
    [
      prop "is equality of notation, precision, trimming and grouping"
        ~examples:
          [
            (base, base);
            (base, (Exponent, Decimals 2, false, false));
            (base, (Plain, Decimals 3, false, false));
            (base, (Plain, Significant 2, false, false));
            (base, (Plain, Decimals 2, true, false));
            (base, (Plain, Decimals 2, false, true));
          ]
        (Gen.with_pp
           (fun ppf (a, b) ->
             Format.fprintf ppf "(%a, %a)" Number.pp (make a) Number.pp (make b))
           (Gen.pair gen_parts gen_parts))
        (fun (a, b) ->
          let d = differences a b in
          cover "equal" (d = 0);
          cover "one part differs" (d = 1);
          if d = 0 then equal format (make a) (make b)
          else not_equal format (make a) (make b));
    ]

(* pp_notation *)

let pp_notation =
  group "pp_notation"
    [
      cases "writes the notation's name in lowercase" ~id:snd
        [
          (Number.Plain, "plain");
          (Exponent, "exponent");
          (Si, "si");
          (Percent, "percent");
        ]
        (fun (n, e) ->
          equal string e (Format.asprintf "%a" Number.pp_notation n));
    ]

let () =
  exit
    (run "Number"
       [
         digits;
         signs;
         trimming;
         grouping;
         non_finite;
         plain;
         exponent;
         si;
         percent;
         decimals_precision;
         significant;
         v;
         to_string;
         decimals;
         equal_formats;
         pp_notation;
       ])
