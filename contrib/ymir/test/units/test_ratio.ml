(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Conversion factors: Unit.ratio rounds the exact number of u / w once, ties to
   even, with the dtype's subnormals and an exponent unbounded above, and raises
   where the result is 0, subnormal or past the largest finite value. Checked
   against mpmath at each float format's edges (golden/ratio.golden, written by
   gen/ratio.py), against the C library's correctly rounded decimal reader, and
   by the laws of correct rounding: agreement with rounding the float64 factor
   away from the narrower format's midpoints, scaling by powers of two, and
   monotonicity. *)

open Windtrap
open Ymir_units
open Ymir_units_test

let text = Unit.to_string

(* Float formats, as IEEE 754 and the float8 specifications state them:
   precision in bits, least normal exponent, largest finite value. *)

type dtype = D : (float, 'b) Nx.dtype -> dtype

type format = {
  name : string;
  dtype : dtype;
  prec : int;
  emin : int;
  max : float;
}

let float64 =
  {
    name = "float64";
    dtype = D Nx.float64;
    prec = 53;
    emin = -1022;
    max = Float.max_float;
  }

let formats =
  [
    float64;
    {
      name = "float32";
      dtype = D Nx.float32;
      prec = 24;
      emin = -126;
      max = Float.ldexp 0xffffffp0 104;
    };
    {
      name = "float16";
      dtype = D Nx.float16;
      prec = 11;
      emin = -14;
      max = 65504.;
    };
    {
      name = "bfloat16";
      dtype = D Nx.bfloat16;
      prec = 8;
      emin = -126;
      max = Float.ldexp 0xffp0 120;
    };
    {
      name = "float8_e4m3";
      dtype = D Nx.float8_e4m3;
      prec = 4;
      emin = -6;
      max = 448.;
    };
    {
      name = "float8_e5m2";
      dtype = D Nx.float8_e5m2;
      prec = 3;
      emin = -14;
      max = 57344.;
    };
  ]

let narrow = List.tl formats
let format name = List.find (fun f -> f.name = name) formats

(* Outcomes *)

type outcome = Value of float | Zero | Subnormal | Overflow

let pp_outcome ppf = function
  | Value v -> Format.fprintf ppf "Value %h (%.17g)" v v
  | Zero -> Format.pp_print_string ppf "Zero"
  | Subnormal -> Format.pp_print_string ppf "Subnormal"
  | Overflow -> Format.pp_print_string ppf "Overflow"

let rank = function
  | Zero -> (0, 0.)
  | Subnormal -> (1, 0.)
  | Value v -> (2, v)
  | Overflow -> (3, 0.)

let outcome_w =
  Testable.with_compare
    (fun a b -> compare (rank a) (rank b))
    (Testable.make ~pp:pp_outcome ~equal:(fun a b ->
         match (a, b) with
         | Value a, Value b ->
             Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
         | a, b -> a = b))

let factor_error u w which =
  Printf.sprintf "Unit.ratio: the factor from %s to %s is %s, %s" (text u)
    (text w)
    (text Unit.(u / w))
    which

(* [ratio f u w] is the outcome of [Unit.ratio] in [f]. A raise whose message is
   not the one the interface states for its outcome fails the test. *)
let ratio f u w =
  let (D d) = f.dtype in
  match Unit.ratio d u w with
  | v -> Value v
  | exception Invalid_argument msg ->
      let is which = String.equal msg (factor_error u w which) in
      if is ("which is 0 in " ^ f.name) then Zero
      else if is ("which is subnormal in " ^ f.name) then Subnormal
      else if is ("which overflows " ^ f.name) then Overflow
      else failf "Unit.ratio %s %s %s raised %S" f.name (text u) (text w) msg

(* Goldens *)

let factor_of_token token =
  let base, e =
    match String.index_opt token '^' with
    | None -> (token, "1")
    | Some i ->
        ( String.sub token 0 i,
          String.sub token (i + 1) (String.length token - i - 1) )
  in
  let n, d =
    match String.index_opt e '/' with
    | None -> (int_of_string e, 1)
    | Some j ->
        ( int_of_string (String.sub e 0 j),
          int_of_string (String.sub e (j + 1) (String.length e - j - 1)) )
  in
  ((if base = "pi" then Pi else Int (int_of_string base)), n, d)

let outcome_of_string = function
  | "zero" -> Zero
  | "subnormal" -> Subnormal
  | "overflow" -> Overflow
  | hex -> Value (float_of_string hex)

let goldens =
  In_channel.with_open_text "golden/ratio.golden" In_channel.input_lines
  |> List.map (fun line ->
      match String.split_on_char ' ' line with
      | name :: rest ->
          let rec split acc = function
            | "=" :: [ result ] -> (List.rev acc, outcome_of_string result)
            | t :: rest -> split (t :: acc) rest
            | [] -> failwith ("malformed golden line: " ^ line)
          in
          let tokens, expected = split [] rest in
          (line, format name, List.map factor_of_token tokens, expected)
      | [] -> failwith "empty golden line")

let golden =
  cases
    ~name:(fun (line, _, _, _) -> line)
    "mpmath goldens" goldens
    (fun (_, f, factors, expected) ->
      equal outcome_w expected (ratio f (product factors) Unit.one))

(* Laws *)

(* [round_to f x] is the float64 [x > 0] rounded to [f] with ties to even and
   [f]'s subnormals, and whether [x] is a midpoint between two values of [f]:
   [x] scaled to an integer number of [f]'s units in the last place at its
   exponent, exact since only the exponent changes. *)
let round_to f x =
  let _, e = Float.frexp x in
  let q = Int.max (e - 1) f.emin - (f.prec - 1) in
  let n = Float.ldexp x (-q) in
  let floor = Float.floor n in
  let frac = n -. floor in
  let up = frac > 0.5 || (frac = 0.5 && Float.rem floor 2. = 1.) in
  let v = Float.ldexp (if up then floor +. 1. else floor) q in
  let outcome =
    if v = 0. then Zero
    else if v < Float.ldexp 1. f.emin then Subnormal
    else if v > f.max then Overflow
    else Value v
  in
  (outcome, frac = 0.5)

(* Numbers: products of primes and pi with small rational exponents, scaled by a
   power of two onto an edge of some format or anywhere between. *)

let primes =
  [
    2;
    3;
    5;
    7;
    11;
    13;
    17;
    19;
    23;
    29;
    31;
    37;
    41;
    97;
    65521;
    65537;
    16777213;
    16777259;
    2147483647;
    2305843009213693951;
  ]

let number_factor =
  let open Gen in
  let+ b =
    frequency
      [
        (6, map (fun p -> Int p) (of_list ~pp:pp_int primes)); (2, constant Pi);
      ]
  and+ n = such_that (fun n -> n <> 0) (int_range (-9) 9)
  and+ d = of_list ~pp:pp_int [ 1; 1; 1; 2; 3; 5; 7; 12 ] in
  (b, n, d)

let edges =
  List.concat_map
    (fun f ->
      let emax = snd (Float.frexp f.max) - 1 in
      [
        emax + 1; emax; f.emin; f.emin - 1; f.emin - f.prec + 1; f.emin - f.prec;
      ])
    formats

let log2_factor (b, n, d) =
  let l =
    match b with
    | Int p -> Float.log2 (Float.of_int p)
    | _ -> Float.log2 Float.pi
  in
  Float.of_int n *. l /. Float.of_int d

let numbers =
  let open Gen in
  let target =
    frequency [ (3, of_list ~pp:pp_int edges); (1, int_range (-1100) 1050) ]
  in
  let placed =
    let+ fs = list ~size:(int_range 1 3) number_factor
    and+ t = target
    and+ off = int_range (-1) 1 in
    let l = List.fold_left (fun acc f -> acc +. log2_factor f) 0. fs in
    let j = t + off - Float.to_int (Float.round l) in
    Unit.(product fs * (int 2 ** j))
  in
  with_pp Unit.pp placed

let irrational u =
  List.exists (fun (_, _, d) -> d > 1) (Unit.terms u)
  || List.exists (fun (t, _, _) -> t = Unit.Pi) (Unit.terms u)

(* A decimal [m e k] with a mantissa below 2^61 and an exponent across float64's
   range and past it. *)
let decimals =
  let open Gen in
  let mantissa =
    frequency
      [
        (4, int_range 1 ((1 lsl 61) - 1));
        (2, int_range 1 999);
        ( 1,
          of_list ~pp:pp_int
            [
              1;
              (1 lsl 53) - 1;
              1 lsl 53;
              (1 lsl 53) + 1;
              (1 lsl 61) - 1;
              17976931348623157;
              22250738585072014;
              49406564584124654;
              24703282292062327;
              24703282292062328;
            ] );
      ]
  in
  let exponent =
    frequency
      [
        (3, int_range (-400) 400);
        (2, int_range (-345) (-320));
        (2, int_range (-330) (-300));
        (2, int_range 285 312);
      ]
  in
  let+ m = mantissa and+ k = exponent in
  Printf.sprintf "%de%d" m k

let test_decimal s =
  let x = float_of_string s in
  let expected =
    if x = Float.infinity then Overflow
    else if x = 0. then Zero
    else if x < Float.ldexp 1. (-1022) then Subnormal
    else Value x
  in
  cover "overflow" (expected = Overflow);
  cover "subnormal" (expected = Subnormal);
  cover "zero" (expected = Zero);
  equal outcome_w expected (ratio float64 (Unit.decimal s) Unit.one)

(* Every narrower format rounds as rounding the float64 factor does, unless that
   factor is one of the format's midpoints. The exact number lies within half a
   float64 unit of the factor, and every midpoint of a narrower format is a
   float64, so no other midpoint lies between them. *)
let test_narrow u =
  cover "an irrational number" (irrational u);
  let wide = ratio float64 u Unit.one in
  List.iter
    (fun f ->
      let expected =
        match wide with
        | Value x -> (
            match round_to f x with
            | _, true -> None
            | outcome, false -> Some outcome)
        | Overflow -> Some Overflow
        | Zero | Subnormal -> Some Zero
      in
      match expected with
      | None -> classify "a float64 factor at a midpoint" true
      | Some expected ->
          cover ("a value in " ^ f.name)
            (match expected with Value _ -> true | _ -> false);
          equal ~msg:f.name outcome_w expected (ratio f u Unit.one))
    narrow

(* Scaling by 2^k scales a correctly rounded factor by 2^k while both stay above
   the least normal value and finite. *)
let test_scaling (u, k) =
  List.iter
    (fun f ->
      let least = Float.ldexp 1. f.emin in
      match (ratio f u Unit.one, ratio f Unit.(u * (int 2 ** k)) Unit.one) with
      | Value a, Value b when a > least && b > least ->
          cover "an irrational number scaled" (irrational u);
          equal ~msg:f.name outcome_w (Value (Float.ldexp a k)) (Value b)
      | _ -> ())
    formats

(* A greater number never has a smaller factor: u (n + 1) / n is above u. *)
let test_monotone (u, n) =
  let above = Unit.(u * int (n + 1) / int n) in
  List.iter
    (fun f ->
      at_most ~msg:f.name outcome_w ~than:(ratio f above Unit.one)
        (ratio f u Unit.one))
    formats

(* Symbols cancel: the factor from u to w is the number of u / w. *)
let test_quotient (a, b, s) =
  let s = symbols_of s in
  let u = Unit.(a * s) and w = Unit.(b * s) in
  List.iter
    (fun f ->
      equal ~msg:f.name outcome_w (ratio f Unit.(u / w) Unit.one) (ratio f u w))
    formats

let test_identity u =
  List.iter
    (fun f -> equal ~msg:f.name outcome_w (Value 1.) (ratio f u u))
    formats

let complex_of = function
  | "float32" -> Some (fun u w -> Unit.ratio Nx.complex64 u w)
  | "float64" -> Some (fun u w -> Unit.ratio Nx.complex128 u w)
  | _ -> None

let test_complex u =
  List.iter
    (fun f ->
      match complex_of f.name with
      | None -> ()
      | Some complex -> (
          match ratio f u Unit.one with
          | Value v ->
              let c = complex u Unit.one in
              equal ~msg:f.name
                (pair float_exact float_exact)
                (v, 0.) (c.re, c.im)
          | _ ->
              raises_match ~msg:f.name Exn.invalid_arg (fun () ->
                  complex u Unit.one)))
    formats

let laws =
  group "laws"
    [
      prop ~count:1000 "float64 factors of decimals agree with float_of_string"
        decimals test_decimal;
      prop ~count:500 "narrower formats round as the float64 factor rounds"
        numbers test_narrow;
      prop ~count:300 "scaling by 2^k scales the factor"
        (Gen.pair numbers (Gen.int_range (-60) 60))
        test_scaling;
      prop ~count:300 "factors are monotone"
        (Gen.pair numbers
           (Gen.frequency
              [
                (2, Gen.int_range 1 1000);
                (1, Gen.of_list ~pp:pp_int [ (1 lsl 61) - 2; max_int - 1 ]);
              ]))
        test_monotone;
      prop "the factor from u to w is the number of u / w"
        (Gen.triple numbers numbers units)
        test_quotient;
      prop "the factor from a unit to itself is 1" units test_identity;
      prop "complex dtypes round as their component" numbers test_complex;
    ]

(* Integer dtypes *)

type int_dtype = I : ('a, 'b) Nx.dtype * string * ('a -> string) -> int_dtype

let int_dtypes =
  [
    (I (Nx.int4, "int4", string_of_int), 7);
    (I (Nx.uint4, "uint4", string_of_int), 15);
    (I (Nx.int8, "int8", string_of_int), 127);
    (I (Nx.uint8, "uint8", string_of_int), 255);
    (I (Nx.int16, "int16", string_of_int), 32767);
    (I (Nx.uint16, "uint16", string_of_int), 65535);
    (I (Nx.int32, "int32", Int32.to_string), 0x7fffffff);
    (I (Nx.uint32, "uint32", Int32.to_string), 0xffffffff);
    (I (Nx.int64, "int64", Int64.to_string), max_int);
    (I (Nx.uint64, "uint64", Int64.to_string), max_int);
  ]

let pp_int_dtype ppf (I (_, name, _), _) = Format.pp_print_string ppf name

(* The value as the dtype's OCaml value prints it: uint32 and uint64 as their
   bit patterns. *)
let expected_int name n =
  match name with
  | "int32" | "uint32" -> Int32.to_string (Int32.of_int n)
  | "int64" | "uint64" -> Int64.to_string (Int64.of_int n)
  | _ -> string_of_int n

let test_int_dtype (I (d, name, show), largest) n =
  let u = Unit.int n in
  match Unit.ratio d u Unit.one with
  | v ->
      at_most ~msg:name int ~than:largest n;
      equal ~msg:name string (expected_int name n) (show v)
  | exception Invalid_argument msg ->
      greater ~msg:name int ~than:largest n;
      equal string
        (factor_error u Unit.one ("which " ^ name ^ " does not hold"))
        msg

let int_values =
  let neighbours =
    List.concat_map (fun (_, b) -> [ b - 1; b; b + 1 ]) int_dtypes
    |> List.filter (fun n -> n >= 1 && n <= max_int)
  in
  Gen.frequency
    [ (3, Gen.of_list ~pp:pp_int neighbours); (1, Gen.int_range 1 max_int) ]

(* 2^63 - 1 = 7^2 73 127 337 92737 649657; 2^64 - 1 = 3 5 17 257 641 65537
   6700417. *)
let int64_max =
  Unit.(int 49 * int 73 * int 127 * int 337 * int 92737 * int 649657)

let uint64_max =
  Unit.(int 3 * int 5 * int 17 * int 257 * int 641 * int 65537 * int 6700417)

let wide_ints =
  [
    ( "int64 holds 2^63 - 1",
      fun () ->
        equal int64 Int64.max_int (Unit.ratio Nx.int64 int64_max Unit.one) );
    ( "int64 does not hold 2^63",
      fun () ->
        let u = Unit.(int 2 ** 63) in
        raises
          (Invalid_argument
             (factor_error u Unit.one "which int64 does not hold"))
          (fun () -> Unit.ratio Nx.int64 u Unit.one) );
    ( "uint64 holds 2^63 as its bits",
      fun () ->
        equal int64 Int64.min_int
          (Unit.ratio Nx.uint64 Unit.(int 2 ** 63) Unit.one) );
    ( "uint64 holds 2^64 - 1 as its bits",
      fun () -> equal int64 (-1L) (Unit.ratio Nx.uint64 uint64_max Unit.one) );
    ( "uint64 does not hold 2^64",
      fun () ->
        let u = Unit.(int 2 ** 64) in
        raises
          (Invalid_argument
             (factor_error u Unit.one "which uint64 does not hold"))
          (fun () -> Unit.ratio Nx.uint64 u Unit.one) );
    ( "uint32 holds 2^31 as its bits",
      fun () ->
        equal int32 Int32.min_int
          (Unit.ratio Nx.uint32 Unit.(int 2 ** 31) Unit.one) );
    ( "int8 does not hold 2^200",
      fun () ->
        let u = Unit.(int 2 ** 200) in
        raises
          (Invalid_argument (factor_error u Unit.one "which int8 does not hold"))
          (fun () -> Unit.ratio Nx.int8 u Unit.one) );
    ( "int64 holds the factor from km to mm",
      fun () ->
        equal int64 1_000_000L Unit.(ratio Nx.int64 (kilo metre) (milli metre))
    );
  ]

let non_integers =
  Unit.
    [
      ("1/2", int 1 / int 2);
      ("3/2", int 3 / int 2);
      ("pi", pi);
      ("2^1/2", root 2 (int 2));
      ("1e-30", int 10 ** -30);
      ("1 + 2^-61", int ((1 lsl 61) + 1) / (int 2 ** 61));
    ]

let integers =
  group "integer dtypes"
    [
      prop ~count:500 "an integer dtype holds exactly the integers in its range"
        (Gen.pair (Gen.of_list ~pp:pp_int_dtype int_dtypes) int_values)
        (fun (d, n) -> test_int_dtype d n);
      cases ~name:fst "64-bit bounds" wide_ints (fun (_, f) -> f ());
      cases ~name:fst "not an integer" non_integers (fun (_, u) ->
          List.iter
            (fun (I (d, name, _), _) ->
              raises ~msg:name
                (Invalid_argument
                   (factor_error u Unit.one "which is not an integer"))
                (fun () -> Unit.ratio d u Unit.one))
            int_dtypes);
    ]

(* Errors the interface states *)

let electron = Unit.symbol "electron"

(* Factors whose correct rounding float64 or float32 arithmetic gives: an IEEE
   square root and quotient are correctly rounded. *)

let values =
  cases
    ~name:(fun (n, _, _, _) -> n)
    "correctly rounded values"
    [
      ("pi in float64", Unit.pi, Unit.one, Float.pi);
      ("sqrt 2 in float64", Unit.(root 2 (int 2)), Unit.one, Float.sqrt 2.);
      ("degree in radians", Unit.degree, Unit.radian, Float.pi /. 180.);
      ("1/3 in float64", Unit.(one / int 3), Unit.one, 1. /. 3.);
    ]
    (fun (_, u, w, v) -> equal float_exact v (Unit.ratio Nx.float64 u w))

let pi_float32 =
  test "pi in float32" (fun () ->
      equal float_exact
        (Int32.float_of_bits 0x40490fdbl)
        (Unit.ratio Nx.float32 Unit.pi Unit.one))

let errors =
  group "errors"
    [
      test "units that do not convert" (fun () ->
          raises
            (Invalid_argument
               "Unit.ratio: electron s^-1 does not convert to kg s^-1: their \
                quotient keeps electron kg^-1") (fun () ->
              Unit.(ratio Nx.float32 (electron / second) (kilogram / second))));
      test "scoped symbols of two data sets do not convert" (fun () ->
          raises
            (Invalid_argument
               "Unit.ratio: m{a} does not convert to m{b}: their quotient \
                keeps m{a} m{b}^-1") (fun () ->
              Unit.(
                ratio Nx.float64 (scoped ~scope:"a" "m") (scoped ~scope:"b" "m"))));
      test "a factor that rounds to 0" (fun () ->
          let jy = Unit.((int 10 ** -26) * kilogram / (second ** 2)) in
          raises
            (Invalid_argument
               "Unit.ratio: the factor from 1e-35 kg s^-2 to kg s^-2 is 1e-35, \
                which is 0 in float16") (fun () ->
              Unit.(
                ratio Nx.float16
                  (milli (milli (milli jy)))
                  (kilogram / (second ** 2)))));
      test "a factor that rounds to a subnormal" (fun () ->
          raises
            (Invalid_argument
               "Unit.ratio: the factor from 1e-5 to 1 is 1e-5, which is \
                subnormal in float16") (fun () ->
              Unit.(ratio Nx.float16 (int 10 ** -5) one)));
      test "a factor at float16's overflow tie" (fun () ->
          raises
            (Invalid_argument
               "Unit.ratio: the factor from 6552e1 to 1 is 6552e1, which \
                overflows float16") (fun () ->
              Unit.(ratio Nx.float16 (int 65520) one)));
      test "float8_e4m3 rounds its overflow tie to its largest value" (fun () ->
          equal float_exact 448.
            (Unit.ratio Nx.float8_e4m3 (Unit.int 464) Unit.one));
      test "an evaluation past the budget" (fun () ->
          (* (16777289 / 16777259)^1000000 is about 6, a quotient of two
             naturals of 24 million bits. *)
          let u =
            Unit.((int 16777289 ** 1000000) / (int 16777259 ** 1000000))
          in
          raises
            (Invalid_argument
               "Unit.ratio: the factor from 16777259^-1000000 16777289^1000000 \
                to 1 is 16777259^-1000000 16777289^1000000, whose evaluation \
                needs a natural wider than 65536 bits") (fun () ->
              Unit.ratio Nx.float64 u Unit.one));
      test "an exponent of the quotient that leaves int" (fun () ->
          raises (Invalid_argument "Unit.ratio: the exponent of pi leaves int")
            (fun () -> Unit.(ratio Nx.float64 (pi ** max_int) (pi ** -1))));
      test ~timeout:1.0
        "a root whose radicand is past the budget raises at once" (fun () ->
          (* 3^(10000006/10000007): the radicand alone would be 16 million
             bits. *)
          let u = Unit.(root 10000007 (int 3) ** 10000006) in
          raises_match
            (Exn.invalid_arg ~substring:"needs a natural wider than 65536 bits")
            (fun () -> Unit.ratio Nx.float64 u Unit.one));
      test "bool holds no factor" (fun () ->
          raises (Invalid_argument "Unit.ratio: bool holds no factor")
            (fun () -> Unit.(ratio Nx.bool metre metre)));
      test "bit holds no factor" (fun () ->
          raises (Invalid_argument "Unit.ratio: bit holds no factor") (fun () ->
              Unit.(ratio Nx.bit metre metre)));
    ]

let () =
  exit (run "Unit.ratio" [ golden; values; pi_float32; laws; integers; errors ])
