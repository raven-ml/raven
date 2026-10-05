(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Units as exact data: the group laws with rational exponents wherever both
   sides stay within the algebra's bounds, equality and order as those of the
   canonical text, terms, conversion classes, the bounds themselves and the SI's
   definitions. *)

open Windtrap
open Ymir_units
open Ymir_units_test

(* [bounded sides] discards the case unless every side stays within the
   algebra's bounds. *)
let bounded sides =
  if List.exists (fun side -> Option.is_none (within side)) sides then reject ()

let wide_case us = cover "a wide exponent" (List.exists has_wide_exponent us)

(* Group *)

let test_associative (a, b, c) =
  bounded [ (fun () -> Unit.(a * b * c)); (fun () -> Unit.(a * (b * c))) ];
  wide_case [ a; b; c ];
  Law.associative unit Unit.( * ) (a, b, c)

(* A product leaves the bounds or not whatever the order of its factors. *)
let test_commutative (a, b) =
  let ab = within (fun () -> Unit.(a * b)) in
  cover "a product past the bounds" (Option.is_none ab);
  wide_case [ a; b ];
  equal (option unit) ab (within (fun () -> Unit.(b * a)))

(* The inverse of a unit within the bounds is within them, and multiplying by it
   never raises. *)
let test_inverse u =
  wide_case [ u ];
  Law.invertible unit Unit.( * ) Unit.one (fun u -> Unit.(one / u)) u

let test_quotient (u, w) =
  bounded [ (fun () -> Unit.(u * w)) ];
  wide_case [ u; w ];
  equal unit u Unit.(u * w / w)

let test_division (u, w) =
  bounded [ (fun () -> Unit.(u / w)) ];
  equal unit Unit.(u / w) Unit.(u * (one / w))

let group_laws =
  group "group"
    [
      prop "( * ) is associative"
        (Gen.triple units units units)
        test_associative;
      prop "( * ) is commutative" near_pairs test_commutative;
      prop "one is neutral" units (Law.neutral unit Unit.( * ) Unit.one);
      prop "one / u is the inverse of u" units test_inverse;
      prop "(u * w) / w is u" (Gen.pair units units) test_quotient;
      prop "u / w is u * (one / w)" (Gen.pair units units) test_division;
    ]

(* Powers and roots *)

let exponents =
  Gen.frequency
    [
      (6, Gen.int_range 1 12); (1, Gen.of_list ~pp:pp_int [ max_int; 1 lsl 61 ]);
    ]

let test_root_of_power (u, n) =
  bounded [ (fun () -> Unit.(u ** n)) ];
  wide_case [ u ];
  equal unit u Unit.(root n (u ** n))

let test_power_of_root (u, n) =
  bounded [ (fun () -> Unit.root n u) ];
  equal unit u Unit.(root n u ** n)

let test_power_of_power (u, n, k) =
  let nk = n * k in
  bounded [ (fun () -> Unit.((u ** n) ** k)); (fun () -> Unit.(u ** nk)) ];
  equal unit Unit.(u ** nk) Unit.((u ** n) ** k)

let test_root_of_root (u, n, k) =
  bounded
    [ (fun () -> Unit.(root n (root k u))); (fun () -> Unit.root (n * k) u) ];
  equal unit (Unit.root (n * k) u) Unit.(root n (root k u))

let test_power_of_product n (u, w) =
  bounded
    [ (fun () -> Unit.((u * w) ** n)); (fun () -> Unit.((u ** n) * (w ** n))) ];
  Law.homomorphic unit unit (fun u -> Unit.(u ** n)) Unit.( * ) Unit.( * ) (u, w)

let powers =
  let small = Gen.such_that (fun n -> n <> 0) (Gen.int_range (-5) 5) in
  group "powers"
    [
      prop "root n (u ** n) is u" (Gen.pair units exponents) test_root_of_power;
      prop "root n u ** n is u" (Gen.pair units exponents) test_power_of_root;
      prop "(u ** n) ** k is u ** (n * k)"
        (Gen.triple units small small)
        test_power_of_power;
      prop "root n (root k u) is root (n * k) u"
        (Gen.triple units (Gen.int_range 1 12) (Gen.int_range 1 12))
        test_root_of_root;
      prop "( ** ) distributes over ( * )" (Gen.pair units units)
        (test_power_of_product 3);
      prop "u ** -1 is one / u" units (fun u ->
          equal unit Unit.(one / u) Unit.(u ** -1));
      prop "u ** 0 is one" units (fun u -> equal unit Unit.one Unit.(u ** 0));
      prop "u ** 1 and root 1 u are u" units (fun u ->
          equal unit u Unit.(u ** 1);
          equal unit u (Unit.root 1 u));
    ]

(* Exponents at the int extremes. Each reduced exponent is the exact sum or
   product; a result raises iff its reduced numerator or denominator leaves int,
   min_int excluded. *)

let m = Unit.metre

let leaves_int op base =
  Printf.sprintf "%s: the exponent of %s leaves int" op base

let fits =
  [
    ( "m^max_int m^-1",
      (fun () -> Unit.((m ** max_int) * (m ** -1))),
      "m^4611686018427387902" );
    ( "m^(min_int + 1) m",
      (fun () -> Unit.((m ** (min_int + 1)) * m)),
      "m^-4611686018427387902" );
    ( "m^max_int/2 m^1/2",
      (fun () -> Unit.(root 2 (m ** max_int) * root 2 m)),
      "m^2305843009213693952" );
    ( "m^1/max_int squared",
      (fun () -> Unit.(root max_int m * root max_int m)),
      "m^2/4611686018427387903" );
    ( "m^(2^61 + 1)/2 m^-(3 2^60 + 1)/3, whose cross products leave int",
      (fun () ->
        Unit.(
          root 2 (m ** 2305843009213693953) * root 3 (m ** -3458764513820540929))),
      "m^1/6" );
    ( "(m^1/2)^max_int squared",
      (fun () -> Unit.((root 2 m ** max_int) ** 2)),
      "m^4611686018427387903" );
    ( "(m^1/2)^min_int",
      (fun () -> Unit.(root 2 m ** min_int)),
      "m^-2305843009213693952" );
    ( "(m^1/5)^max_int",
      (fun () -> Unit.(root 5 m ** max_int)),
      "m^4611686018427387903/5" );
    ( "root max_int (m^max_int)",
      (fun () -> Unit.(root max_int (m ** max_int))),
      "m" );
    ( "pi^max_int pi^-1",
      (fun () -> Unit.((pi ** max_int) / pi)),
      "pi^4611686018427387902" );
    ( "16777259^max_int",
      (fun () -> Unit.(int 16777259 ** max_int)),
      "16777259^4611686018427387903" );
  ]

let leaves =
  [
    ( "m^max_int m",
      (fun () -> Unit.((m ** max_int) * m)),
      leaves_int "Unit.( * )" "m" );
    ( "m^(min_int + 1) m^-1",
      (fun () -> Unit.((m ** (min_int + 1)) * (m ** -1))),
      leaves_int "Unit.( * )" "m" );
    ( "m^max_int / m^-1",
      (fun () -> Unit.((m ** max_int) / (m ** -1))),
      leaves_int "Unit.( / )" "m" );
    ("m^min_int", (fun () -> Unit.(m ** min_int)), leaves_int "Unit.( ** )" "m");
    ( "(m^max_int)^2",
      (fun () -> Unit.((m ** max_int) ** 2)),
      leaves_int "Unit.( ** )" "m" );
    ( "m^max_int/3 m^1/3",
      (fun () -> Unit.(root 3 (m ** max_int) * root 3 m)),
      leaves_int "Unit.( * )" "m" );
    ( "m^1/max_int m^1/(max_int - 1)",
      (fun () -> Unit.(root max_int m * root (max_int - 1) m)),
      leaves_int "Unit.( * )" "m" );
    ( "root 2 (m^1/max_int)",
      (fun () -> Unit.(root 2 (root max_int m))),
      leaves_int "Unit.root" "m" );
    ( "pi^max_int pi",
      (fun () -> Unit.((pi ** max_int) * pi)),
      leaves_int "Unit.( * )" "pi" );
    ( "3^max_int/5 3^1/5",
      (fun () -> Unit.((root 5 (int 3) ** max_int) * root 5 (int 3))),
      leaves_int "Unit.( * )" "3" );
    ( "16777259^max_int 16777259",
      (fun () -> Unit.((int 16777259 ** max_int) * int 16777259)),
      leaves_int "Unit.( * )" "16777259" );
  ]

let coefficient op part =
  Printf.sprintf "%s: the coefficient's %s is past 4096 bits" op part

(* The coefficient is the product of the primes below 2^24 with an integer
   exponent; 2^4095, 3^2584, 5^1764 and 10^1233 have 4096 bits, and the next
   power of each has more. *)
let coefficient_fits =
  [
    ("2^4095", fun () -> Unit.(int 2 ** 4095));
    ("2^-4095", fun () -> Unit.(int 2 ** -4095));
    ("3^2584", fun () -> Unit.(int 3 ** 2584));
    ("5^1764", fun () -> Unit.(int 5 ** 1764));
    ("10^1233", fun () -> Unit.(int 10 ** 1233));
    ("10^-1233", fun () -> Unit.(int 10 ** -1233));
    ("3^2584 / 2^4095", fun () -> Unit.((int 3 ** 2584) / (int 2 ** 4095)));
    ("16777259^1000, past 2^24", fun () -> Unit.(int 16777259 ** 1000));
    ("2^(8191/2), not an integer power", fun () -> Unit.(root 2 (int 2) ** 8191));
  ]

let coefficient_past =
  [
    ( "2^4096",
      (fun () -> Unit.(int 2 ** 4096)),
      coefficient "Unit.( ** )" "numerator" );
    ( "2^-4096",
      (fun () -> Unit.(int 2 ** -4096)),
      coefficient "Unit.( ** )" "denominator" );
    ( "3^2585",
      (fun () -> Unit.(int 3 ** 2585)),
      coefficient "Unit.( ** )" "numerator" );
    ( "10^1234",
      (fun () -> Unit.(int 10 ** 1234)),
      coefficient "Unit.( ** )" "numerator" );
    ( "16777213^1000, below 2^24",
      (fun () -> Unit.(int 16777213 ** 1000)),
      coefficient "Unit.( ** )" "numerator" );
    ( "2^4095 2",
      (fun () -> Unit.((int 2 ** 4095) * int 2)),
      coefficient "Unit.( * )" "numerator" );
    ( "2^-4095 / 2",
      (fun () -> Unit.((int 2 ** -4095) / int 2)),
      coefficient "Unit.( / )" "denominator" );
    ( "2^4095 / 2^-1",
      (fun () -> Unit.((int 2 ** 4095) / (int 2 ** -1))),
      coefficient "Unit.( / )" "numerator" );
    ( "2^(8193/2) 2^1/2, an integer power",
      (fun () -> Unit.((root 2 (int 2) ** 8193) * root 2 (int 2))),
      coefficient "Unit.( * )" "numerator" );
  ]

let bounds =
  group "bounds"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "within" fits
        (fun (_, f, text) -> equal string text (Unit.to_string (f ())));
      cases
        ~name:(fun (n, _, _) -> n)
        "exponent leaves int" leaves
        (fun (_, f, msg) -> raises (Invalid_argument msg) f);
      cases ~name:fst "coefficient within 4096 bits" coefficient_fits
        (fun (_, f) -> ignore (f ()));
      cases
        ~name:(fun (n, _, _) -> n)
        "coefficient past 4096 bits" coefficient_past
        (fun (_, f, msg) -> raises (Invalid_argument msg) f);
    ]

(* Equality, order and text *)

let sign n = Int.compare n 0

let test_equal_is_text (u, w) =
  cover "equal" (Unit.equal u w);
  equal bool
    (String.equal (Unit.to_string u) (Unit.to_string w))
    (Unit.equal u w)

let test_compare_is_text (u, w) =
  equal int
    (sign (String.compare (Unit.to_string u) (Unit.to_string w)))
    (sign (Unit.compare u w))

let identity =
  group "identity"
    [
      prop "equal is an equivalence" (Gen.pair units units)
        (Law.equivalence unit);
      prop "compare is a total order"
        (Gen.triple units units units)
        (Law.order unit);
      prop "equal is equality of canonical texts" near_pairs test_equal_is_text;
      prop "compare is the byte order of canonical texts" near_pairs
        test_compare_is_text;
      test "kilo metre is int 1000 * metre" (fun () ->
          equal unit Unit.(int 1000 * metre) Unit.(kilo metre));
    ]

(* Terms *)

let compare_term (a : Unit.term) (b : Unit.term) =
  match (a, b) with
  | Prime p, Prime q -> Int.compare p q
  | Prime _, _ -> -1
  | _, Prime _ -> 1
  | Pi, Pi -> 0
  | Pi, _ -> -1
  | _, Pi -> 1
  | Symbol a, Symbol b -> (
      match String.compare a.name b.name with
      | 0 -> Option.compare String.compare a.scope b.scope
      | c -> c)

let rec gcd a b = if b = 0 then a else gcd b (a mod b)

let pp_term ppf ((t : Unit.term), n, d) =
  match t with
  | Prime p -> Format.fprintf ppf "(Prime %d, %d, %d)" p n d
  | Pi -> Format.fprintf ppf "(Pi, %d, %d)" n d
  | Symbol { name; scope } ->
      Format.fprintf ppf "(Symbol %S %a, %d, %d)" name pp_scope scope n d

let terms_w = Testable.make ~pp:(Format.pp_print_list pp_term) ~equal:( = )

let rec ascending = function
  | (a, _, _) :: ((b, _, _) :: _ as rest) ->
      compare_term a b < 0 && ascending rest
  | _ -> true

let test_terms_reduced u =
  List.iter
    (fun ((_, n, d) as t) ->
      let msg = Format.asprintf "%a" pp_term t in
      at_least ~msg int ~than:1 d;
      not_equal ~msg int 0 n;
      equal ~msg int 1 (gcd (abs n) d))
    (Unit.terms u)

let test_terms_order u =
  satisfies ~claim:"terms in canonical order" terms_w ascending (Unit.terms u)

let terms =
  group "terms"
    [
      prop "terms are reduced with a positive denominator" units
        test_terms_reduced;
      prop "terms are in canonical order" units test_terms_order;
      prop "a unit is the product of its terms" units
        (Law.round_trip unit terms_w Unit.terms of_terms);
      test "terms one is []" (fun () -> equal terms_w [] (Unit.terms Unit.one));
      test "terms order primes, pi, then symbols by name then scope" (fun () ->
          let u =
            Unit.(
              scoped ~scope:"b" "x" * symbol "B" * scoped ~scope:"a" "x"
              * symbol "x" * pi * int 16777259
              * root 2 (int 3))
          in
          equal terms_w
            [
              (Unit.Prime 3, 1, 2);
              (Unit.Prime 16777259, 1, 1);
              (Unit.Pi, 1, 1);
              (Unit.Symbol { name = "B"; scope = None }, 1, 1);
              (Unit.Symbol { name = "x"; scope = None }, 1, 1);
              (Unit.Symbol { name = "x"; scope = Some "a" }, 1, 1);
              (Unit.Symbol { name = "x"; scope = Some "b" }, 1, 1);
            ]
            (Unit.terms u));
      test "scopes order by their bytes, not their encoded text" (fun () ->
          (* "%" encodes as "%25", above "%41" as bytes and below as text. *)
          let u = Unit.(scoped ~scope:"%41" "x" * scoped ~scope:"%" "x") in
          equal terms_w
            [
              (Unit.Symbol { name = "x"; scope = Some "%" }, 1, 1);
              (Unit.Symbol { name = "x"; scope = Some "%41" }, 1, 1);
            ]
            (Unit.terms u));
    ]

(* Conversion classes *)

let symbols u =
  List.filter
    (function Unit.Symbol _, _, _ -> true | _ -> false)
    (Unit.terms u)

let converts expected u w =
  let msg = Unit.to_string u ^ " and " ^ Unit.to_string w in
  equal ~msg bool expected (Unit.convertible u w)

let convertible_w = Testable.make ~pp:Unit.pp ~equal:Unit.convertible

let test_convertible (u, w) =
  let same = symbols u = symbols w in
  cover "convertible" same;
  equal bool same (Unit.convertible u w)

let conversion =
  group "convertible"
    [
      prop "convertible is an equivalence" (Gen.pair units units)
        (Law.equivalence ~respell:(fun u -> Unit.(u * int 2)) convertible_w);
      prop "units convert iff their symbols and exponents agree" near_pairs
        test_convertible;
      prop "a unit converts to itself times any number" (Gen.pair units units)
        (fun (u, w) ->
          let n = Unit.(w / symbols_of w) in
          bounded [ (fun () -> Unit.(u * n)) ];
          converts true Unit.(u * n) u);
      test "a scoped symbol converts only to itself" (fun () ->
          let a = Unit.scoped ~scope:"a" "beam" in
          converts true a (Unit.scoped ~scope:"a" "beam");
          converts false a (Unit.scoped ~scope:"b" "beam");
          converts false a (Unit.symbol "beam"));
      test "hertz does not convert to radians per second" (fun () ->
          converts false Unit.hertz Unit.(radian / second));
    ]

(* Constructors *)

let invalid name f = (name, fun () -> raises_match Exn.invalid_arg f)

let constructors =
  group "constructors"
    [
      test "int 1 is one" (fun () -> equal unit Unit.one (Unit.int 1));
      test "int max_int factors into 3 715827883 2147483647" (fun () ->
          equal string "3 715827883 2147483647"
            (Unit.to_string (Unit.int max_int)));
      test "int factors a square of a prime past 2^24" (fun () ->
          equal string "2147483647^2"
            (Unit.to_string (Unit.int 4611686014132420609)));
      test "int factors a product of primes below 2^24" (fun () ->
          (* 11 17 10099 327289 7461193 *)
          equal string "4611685283988009601"
            (Unit.to_string (Unit.int 4611685283988009601)));
      test "decimal refuses a long mantissa at its first digit past 2^62"
        (fun () ->
          raises_match
            (Exn.invalid_arg ~substring:"has a mantissa of 2^62 or more")
            (fun () -> Unit.decimal (String.make 10_000_000 '9')));
      test "a decimal exponent past int is past the coefficient's bound"
        (fun () ->
          raises
            (Invalid_argument
               "Unit.decimal: the coefficient's numerator is past 4096 bits")
            (fun () -> Unit.decimal "1e99999999999999999999");
          raises
            (Invalid_argument
               "Unit.decimal: the coefficient's denominator is past 4096 bits")
            (fun () -> Unit.decimal "1e-99999999999999999999"));
      test "a mantissa of 61 bits is a decimal" (fun () ->
          equal unit
            (Unit.int 2305843009213693951)
            (Unit.decimal "2305843009213693951");
          equal unit
            Unit.(int 2305843009213693951 / int 10)
            (Unit.decimal "230584300921369395.1"));
      test "a mantissa below 2^62 is a decimal" (fun () ->
          equal unit Unit.(int 2 ** 61) (Unit.decimal "2305843009213693952");
          equal unit
            Unit.((int 2 ** 61) / int 10)
            (Unit.decimal "230584300921369395.2");
          equal unit (Unit.int max_int) (Unit.decimal "4611686018427387903"));
      cases ~name:Fun.id "a mantissa of 2^62 is refused"
        [ "4611686018427387904"; "461168601842738790.4" ] (fun s ->
          raises
            (Invalid_argument
               (Printf.sprintf "Unit.decimal: %S has a mantissa of 2^62 or more"
                  s))
            (fun () -> Unit.decimal s));
      test "decimal reads a fraction and an exponent exactly" (fun () ->
          equal unit Unit.(int 15 / int 1000) (Unit.decimal "1.5e-2");
          equal unit Unit.(int 15 * (int 10 ** 22)) (Unit.decimal "1.5e23"));
      test "symbol names may hold upper case, digits and underscores" (fun () ->
          List.iter
            (fun name -> equal string name (Unit.to_string (Unit.symbol name)))
            [ "_"; "A"; "Pi"; "pi2"; "_1"; "a_B9" ]);
      cases ~name:fst "raise Invalid_argument"
        [
          invalid "int 0" (fun () -> Unit.int 0);
          invalid "int -1" (fun () -> Unit.int (-1));
          invalid "int min_int" (fun () -> Unit.int min_int);
          invalid "root 0" (fun () -> Unit.root 0 Unit.metre);
          invalid "root -1" (fun () -> Unit.root (-1) Unit.metre);
          invalid "root min_int" (fun () -> Unit.root min_int Unit.metre);
          invalid "symbol \"\"" (fun () -> Unit.symbol "");
          invalid "symbol \"pi\"" (fun () -> Unit.symbol "pi");
          invalid "symbol \"1m\"" (fun () -> Unit.symbol "1m");
          invalid "symbol \"m-s\"" (fun () -> Unit.symbol "m-s");
          invalid "symbol \"m s\"" (fun () -> Unit.symbol "m s");
          invalid "symbol with a non-ASCII byte" (fun () ->
              Unit.symbol "\xc3\xa9");
          invalid "symbol \"m{a}\"" (fun () -> Unit.symbol "m{a}");
          invalid "scoped with an empty scope" (fun () ->
              Unit.scoped ~scope:"" "beam");
          invalid "scoped \"pi\"" (fun () -> Unit.scoped ~scope:"a" "pi");
          invalid "scoped \"\"" (fun () -> Unit.scoped ~scope:"a" "");
          invalid "decimal 0" (fun () -> Unit.decimal "0");
          invalid "decimal 0e5" (fun () -> Unit.decimal "0e5");
          invalid "decimal 000.000" (fun () -> Unit.decimal "000.000");
          invalid "decimal \"\"" (fun () -> Unit.decimal "");
          invalid "decimal with a sign" (fun () -> Unit.decimal "+1");
          invalid "decimal with E" (fun () -> Unit.decimal "1E3");
          invalid "decimal with e+" (fun () -> Unit.decimal "1e+3");
          invalid "decimal without integer digits" (fun () -> Unit.decimal ".5");
          invalid "decimal without fraction digits" (fun () ->
              Unit.decimal "1.");
          invalid "decimal with a space" (fun () -> Unit.decimal "1 ");
          invalid "decimal with two points" (fun () -> Unit.decimal "1.2.3");
        ]
        (fun (_, f) -> f ());
    ]

(* The SI *)

let si =
  let open Unit in
  let sr = steradian and w = watt in
  [
    ("metre", metre, symbol "m");
    ("kilogram", kilogram, symbol "kg");
    ("second", second, symbol "s");
    ("ampere", ampere, symbol "A");
    ("kelvin", kelvin, symbol "K");
    ("mole", mole, symbol "mol");
    ("candela", candela, symbol "cd");
    ("radian", radian, symbol "rad");
    ("steradian", steradian, radian ** 2);
    ("hertz", hertz, second ** -1);
    ("newton", newton, kilogram * metre / (second ** 2));
    ("pascal", pascal, kilogram / metre / (second ** 2));
    ("joule", joule, kilogram * (metre ** 2) / (second ** 2));
    ("watt", watt, kilogram * (metre ** 2) / (second ** 3));
    ("coulomb", coulomb, ampere * second);
    ("volt", volt, kilogram * (metre ** 2) / (second ** 3) / ampere);
    ("farad", farad, (ampere ** 2) * (second ** 4) / kilogram / (metre ** 2));
    ("ohm", ohm, kilogram * (metre ** 2) / (second ** 3) / (ampere ** 2));
    ("siemens", siemens, (ampere ** 2) * (second ** 3) / kilogram / (metre ** 2));
    ("weber", weber, kilogram * (metre ** 2) / (second ** 2) / ampere);
    ("tesla", tesla, kilogram / (second ** 2) / ampere);
    ("henry", henry, kilogram * (metre ** 2) / (second ** 2) / (ampere ** 2));
    ("lumen", lumen, candela * sr);
    ("lux", lux, candela * sr / (metre ** 2));
    ("becquerel", becquerel, hertz);
    ("gray", gray, (metre ** 2) / (second ** 2));
    ("sievert", sievert, gray);
    ("katal", katal, mole / second);
    ("gram", gram, (int 10 ** -3) * kilogram);
    ("tonne", tonne, (int 10 ** 3) * kilogram);
    ("minute", minute, int 60 * second);
    ("hour", hour, int 3600 * second);
    ("day", day, int 86400 * second);
    ("litre", litre, (int 10 ** -3) * (metre ** 3));
    ("hectare", hectare, (int 10 ** 4) * (metre ** 2));
    ("astronomical_unit", astronomical_unit, int 149597870700 * metre);
    ("degree", degree, pi / int 180 * radian);
    ("arcminute", arcminute, pi / int 10800 * radian);
    ("arcsecond", arcsecond, pi / int 648000 * radian);
    ("electronvolt", electronvolt, decimal "1.602176634e-19" * joule);
    ("speed_of_light", speed_of_light, int 299792458 * metre / second);
    ("planck", planck, decimal "6.62607015e-34" * joule * second);
    ("hbar", hbar, planck / (int 2 * pi * radian));
    ( "elementary_charge",
      elementary_charge,
      decimal "1.602176634e-19" * ampere * second );
    ("boltzmann", boltzmann, decimal "1.380649e-23" * joule / kelvin);
    ("avogadro", avogadro, decimal "6.02214076e23" / mole);
    ("caesium_frequency", caesium_frequency, int 9192631770 * hertz);
    ("luminous_efficacy", luminous_efficacy, int 683 * candela * sr / w);
    ("gas_constant", gas_constant, avogadro * boltzmann);
    ("faraday", faraday, avogadro * elementary_charge);
    ( "stefan_boltzmann",
      stefan_boltzmann,
      int 2 * (pi ** 5) * (boltzmann ** 4)
      / (int 15 * (planck ** 3) * (speed_of_light ** 2)) );
  ]

let prefixes =
  let open Unit in
  [
    ("quecto", quecto, -30);
    ("ronto", ronto, -27);
    ("yocto", yocto, -24);
    ("zepto", zepto, -21);
    ("atto", atto, -18);
    ("femto", femto, -15);
    ("pico", pico, -12);
    ("nano", nano, -9);
    ("micro", micro, -6);
    ("milli", milli, -3);
    ("centi", centi, -2);
    ("deci", deci, -1);
    ("deca", deca, 1);
    ("hecto", hecto, 2);
    ("kilo", kilo, 3);
    ("mega", mega, 6);
    ("giga", giga, 9);
    ("tera", tera, 12);
    ("peta", peta, 15);
    ("exa", exa, 18);
    ("zetta", zetta, 21);
    ("yotta", yotta, 24);
    ("ronna", ronna, 27);
    ("quetta", quetta, 30);
  ]

let si_units =
  group "SI"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "definitions" si
        (fun (_, u, w) -> equal unit w u);
      cases
        ~name:(fun (n, _, _) -> n)
        "prefixes" prefixes
        (fun (_, p, k) ->
          equal unit Unit.((int 10 ** k) * metre) (p Unit.metre));
    ]

let () =
  exit
    (run "Unit algebra"
       [
         group_laws;
         powers;
         bounds;
         identity;
         terms;
         conversion;
         constructors;
         si_units;
       ])
