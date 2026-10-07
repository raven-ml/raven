(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Vocabularies: reading a symbol with an SI prefix, spelling a unit in as few
   words as the vocabulary allows, and the law that names never change a
   unit. *)

open Windtrap
open Ymir_units
open Ymir_units_test

let strf = Printf.sprintf
let jansky = Unit.(decimal "1e-26" * watt / (metre ** 2) / hertz)
let erg = Unit.(decimal "1e-7" * joule)
let parsec = Unit.(int 648000 / pi * astronomical_unit)
let julian_year = Unit.(int 31557600 * second)
let electron = Unit.symbol "electron"
let text voc u = Format.asprintf "%a" (Vocabulary.pp voc) u

(* The SI prefixes as the SI Brochure writes them. *)
let prefixes =
  [
    (-30, "q");
    (-27, "r");
    (-24, "y");
    (-21, "z");
    (-18, "a");
    (-15, "f");
    (-12, "p");
    (-9, "n");
    (-6, "\xce\xbc");
    (-3, "m");
    (-2, "c");
    (-1, "d");
    (1, "da");
    (2, "h");
    (3, "k");
    (6, "M");
    (9, "G");
    (12, "T");
    (15, "P");
    (18, "E");
    (21, "Z");
    (24, "Y");
    (27, "R");
    (30, "Q");
  ]

let ten k = Unit.decimal (strf "1e%d" k)

let word =
  let pp ppf (w : Vocabulary.word) =
    Format.fprintf ppf "{prefix = %d; symbol = %S; num = %d; den = %d}" w.prefix
      w.symbol w.num w.den
  in
  Testable.make ~pp ~equal:( = )

let spelling =
  let pp ppf (s : Vocabulary.spelling) =
    Format.fprintf ppf "{decade = %d; words = [%a]}" s.decade
      (Format.pp_print_list
         ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
         (Testable.pp word))
      s.words
  in
  Testable.make ~pp ~equal:( = )

let w ?(prefix = 0) ?(den = 1) symbol num : Vocabulary.word =
  { prefix; symbol; num; den }

(* The SI's entries, as [Vocabulary.si] lists them. *)
let si_entries =
  Vocabulary.
    [
      ("kg", Bare, Unit.kilogram);
      ("A", Prefixable, Unit.ampere);
      ("m", Prefixable, Unit.metre);
      ("s", Prefixable, Unit.second);
      ("K", Prefixable, Unit.kelvin);
      ("mol", Prefixable, Unit.mole);
      ("cd", Prefixable, Unit.candela);
      ("rad", Prefixable, Unit.radian);
      ("sr", Prefixable, Unit.steradian);
      ("Hz", Prefixable, Unit.hertz);
      ("N", Prefixable, Unit.newton);
      ("Pa", Prefixable, Unit.pascal);
      ("J", Prefixable, Unit.joule);
      ("W", Prefixable, Unit.watt);
      ("C", Prefixable, Unit.coulomb);
      ("V", Prefixable, Unit.volt);
      ("F", Prefixable, Unit.farad);
      ("\xce\xa9", Prefixable, Unit.ohm);
      ("S", Prefixable, Unit.siemens);
      ("Wb", Prefixable, Unit.weber);
      ("T", Prefixable, Unit.tesla);
      ("H", Prefixable, Unit.henry);
      ("lm", Prefixable, Unit.lumen);
      ("lx", Prefixable, Unit.lux);
      ("g", Prefixable, Unit.gram);
      ("t", Prefixable, Unit.tonne);
      ("L", Prefixable, Unit.litre);
      ("l", Prefixable, Unit.litre);
      ("min", Bare, Unit.minute);
      ("h", Bare, Unit.hour);
      ("d", Bare, Unit.day);
      ("ha", Bare, Unit.hectare);
      ("au", Bare, Unit.astronomical_unit);
      ("\xc2\xb0", Bare, Unit.degree);
      ("\xe2\x80\xb2", Bare, Unit.arcminute);
      ("\xe2\x80\xb3", Bare, Unit.arcsecond);
      ("eV", Prefixable, Unit.electronvolt);
    ]

(* A stand-in for an astronomy vocabulary: the SI, then a few of the units
   astronomers write. *)
let astro_entries =
  Vocabulary.
    [
      ("Jy", Prefixable, jansky);
      ("erg", Bare, erg);
      ("pc", Prefixable, parsec);
      ("yr", Bare, julian_year);
      ("mas", Bare, Unit.(milli arcsecond));
      ("electron", Bare, electron);
    ]

let astro_all = si_entries @ astro_entries
let astro = Vocabulary.(union si (v astro_entries))

(* Constructors *)

let metre_am =
  [
    ("m", Vocabulary.Prefixable, Unit.metre);
    ("am", Prefixable, Unit.symbol "am");
  ]

let constructors =
  group "Vocabulary.v"
    [
      test "an empty symbol raises" (fun () ->
          raises (Invalid_argument "Vocabulary.v: a symbol is empty") (fun () ->
              Vocabulary.v [ ("", Bare, Unit.one) ]));
      test "a symbol given twice raises" (fun () ->
          raises (Invalid_argument {|Vocabulary.v: "m" is given twice|})
            (fun () ->
              Vocabulary.v
                [ ("m", Prefixable, Unit.metre); ("m", Bare, Unit.second) ]));
      test "a string read as two prefixed symbols raises" (fun () ->
          raises
            (Invalid_argument
               {|Vocabulary.v: "dam" reads as "da" on "m" and as "d" on "am"|})
            (fun () -> Vocabulary.v metre_am));
      test "a string read as two prefixed symbols that is a symbol is accepted"
        (fun () ->
          let voc = Vocabulary.v (("dam", Bare, Unit.second) :: metre_am) in
          equal (option unit) (Some Unit.second) (Vocabulary.lookup voc "dam"));
      test "a bare symbol has no prefixed reading" (fun () ->
          let voc =
            Vocabulary.v
              [ ("m", Prefixable, Unit.metre); ("am", Bare, Unit.second) ]
          in
          equal (option unit)
            (Some Unit.(deca metre))
            (Vocabulary.lookup voc "dam"));
      test "union is the first's entries, then the second's" (fun () ->
          let a = Vocabulary.v [ ("Hz", Prefixable, Unit.hertz) ] in
          let b = Vocabulary.v [ ("Bq", Prefixable, Unit.becquerel) ] in
          equal string "Hz" (text (Vocabulary.union a b) Unit.hertz);
          equal string "Bq" (text (Vocabulary.union b a) Unit.hertz));
      test "union raises as v does, naming union" (fun () ->
          let a = Vocabulary.v [ ("m", Prefixable, Unit.metre) ] in
          raises (Invalid_argument {|Vocabulary.union: "m" is given twice|})
            (fun () -> Vocabulary.union a a);
          raises
            (Invalid_argument
               {|Vocabulary.union: "dam" reads as "da" on "m" and as "d" on "am"|})
            (fun () ->
              Vocabulary.union a
                (Vocabulary.v [ ("am", Prefixable, Unit.symbol "am") ])));
    ]

(* Lookup *)

let metres = Vocabulary.v [ ("m", Prefixable, Unit.metre) ]

let lookup =
  group "Vocabulary.lookup"
    [
      cases
        ~name:(fun (k, p) -> strf "%s is 1e%d" p k)
        "an SI prefix on a prefixable symbol" prefixes
        (fun (k, p) ->
          equal (option unit)
            (Some Unit.(ten k * metre))
            (Vocabulary.lookup metres (p ^ "m")));
      cases ~name:Fun.id "micro reads as" [ "u"; "\xc2\xb5"; "\xce\xbc" ]
        (fun p ->
          equal (option unit)
            (Some Unit.(micro metre))
            (Vocabulary.lookup metres (p ^ "m")));
      test "da precedes d" (fun () ->
          equal (option unit)
            (Some Unit.(deca metre))
            (Vocabulary.lookup metres "dam"));
      test "a symbol is read before a prefixed reading" (fun () ->
          let x = Unit.symbol "x" in
          let voc =
            Vocabulary.v [ ("m", Prefixable, Unit.metre); ("mm", Bare, x) ]
          in
          equal (option unit) (Some x) (Vocabulary.lookup voc "mm"));
      cases
        ~name:(fun s -> strf "%S" s)
        "reads as nothing"
        [ ""; "k"; "da"; "x"; "mx"; "kmin"; "kh"; "M m"; "mm "; "Km" ]
        (fun s ->
          let voc =
            Vocabulary.v
              [
                ("m", Prefixable, Unit.metre);
                ("min", Bare, Unit.minute);
                ("h", Bare, Unit.hour);
              ]
          in
          equal (option unit) None (Vocabulary.lookup voc s));
    ]

(* The SI vocabulary *)

let si =
  group "Vocabulary.si"
    [
      cases
        ~name:(fun (s, _, _) -> s)
        "reads each of its symbols" si_entries
        (fun (s, _, u) ->
          equal (option unit) (Some u) (Vocabulary.lookup Vocabulary.si s));
      cases ~name:fst "reads a prefixed symbol"
        Unit.
          [
            ("km", kilo metre);
            ("mrad", milli radian);
            ("k\xce\xa9", kilo ohm);
            ("mL", milli litre);
            ("Mt", mega tonne);
            ("mg", milli gram);
            ("keV", kilo electronvolt);
          ]
        (fun (s, u) ->
          equal (option unit) (Some u) (Vocabulary.lookup Vocabulary.si s));
      cases ~name:Fun.id "omits" [ "Bq"; "Gy"; "Sv"; "kat" ] (fun s ->
          equal (option unit) None (Vocabulary.lookup Vocabulary.si s));
      cases ~name:Fun.id "takes no prefix"
        [
          "kg";
          "min";
          "h";
          "d";
          "ha";
          "au";
          "\xc2\xb0";
          "\xe2\x80\xb2";
          "\xe2\x80\xb3";
        ] (fun s ->
          equal (option unit) None (Vocabulary.lookup Vocabulary.si ("k" ^ s)));
    ]

(* The spellings the rule gives for common units, in the SI and in an
   astronomy vocabulary. A row whose spelling is [None] prints the unit's
   canonical text. *)
let rows =
  let open Unit in
  let cm = centi metre in
  [
    ("kJ", "si", kilo joule, Some "kJ");
    ("kJ/mol", "si", kilo joule / mole, Some "kJ mol^-1");
    ("kg m2", "si", kilogram * (metre ** 2), Some "kg m^2");
    ("N m", "si", newton * metre, Some "J");
    ("W/m2", "si", watt / (metre ** 2), Some "kg s^-3");
    ( "Stefan-Boltzmann",
      "si",
      watt / (metre ** 2) / (kelvin ** 4),
      Some "kg s^-3 K^-4" );
    ("W/sr", "si", watt / steradian, Some "W sr^-1");
    ("J/(mol K)", "si", joule / (mole * kelvin), Some "J K^-1 mol^-1");
    ("ohm m", "si", ohm * metre, Some "\xce\xa9 m");
    ("Pa s", "si", pascal * second, Some "Pa s");
    ("hPa", "si", hecto pascal, Some "hPa");
    ("G", "si", (metre ** 3) / kilogram / (second ** 2), Some "m^3 kg^-1 s^-2");
    ("Jy", "astro", jansky, Some "Jy");
    ("Jy", "si", jansky, Some "1e-26 kg s^-2");
    ("MJy/sr", "astro", mega jansky / steradian, Some "MJy sr^-1");
    ("MJy/sr", "si", mega jansky / steradian, Some "1e-20 kg s^-2 sr^-1");
    ("erg/s/cm2", "astro", erg / second / (cm ** 2), Some "g s^-3");
    ("erg/s", "astro", erg / second, Some "erg s^-1");
    ( "1e-20 W/m2",
      "astro",
      decimal "1e-20" * watt / (metre ** 2),
      Some "MJy s^-1" );
    ("km/s", "si", kilo metre / second, Some "km s^-1");
    ("km2/s2", "si", (kilo metre / second) ** 2, Some "km^2 s^-2");
    ("m/s2", "si", metre / (second ** 2), Some "m s^-2");
    ("Hz", "si", hertz, Some "Hz");
    ("GHz", "si", giga hertz, Some "GHz");
    ("rad/s", "si", radian / second, Some "rad s^-1");
    ("eV", "si", electronvolt, Some "eV");
    ("MeV", "si", mega electronvolt, Some "MeV");
    ("au/day", "si", astronomical_unit / day, Some "au d^-1");
    ("mas/yr", "astro", milli arcsecond / julian_year, Some "mas yr^-1");
    ("electron/s", "astro", electron / second, Some "electron s^-1");
    ("electron/cm3", "astro", electron / (cm ** 3), Some "electron cm^-3");
    ( "electron/(cm2 s)",
      "astro",
      electron / (cm ** 2) / second,
      Some "electron cm^-2 s^-1" );
    ("m3", "si", metre ** 3, Some "m^3");
    ("km3", "si", kilo metre ** 3, Some "km^3");
    ("cm3", "si", cm ** 3, Some "cm^3");
    ("cm-3", "si", cm ** -3, Some "cm^-3");
    ("g/cm3", "si", gram / (cm ** 3), Some "Mg m^-3");
    ("mg", "si", milli gram, Some "mg");
    ("mol/s", "si", mole / second, Some "mol s^-1");
    ("pc/cm3", "astro", parsec / (cm ** 3), Some "Mpc m^-3");
    ( "km/s/Mpc",
      "astro",
      kilo metre / second / mega parsec,
      Some "mm pc^-1 s^-1" );
    ("Jy km/s", "astro", jansky * kilo metre / second, Some "kJy m s^-1");
    ( "1e-17 erg/s/cm2/angstrom",
      "astro",
      decimal "1e-17" * erg / second / (cm ** 2) / (decimal "1e-10" * metre),
      Some "1e-10 Pa s^-1" );
    ("1", "si", one, Some "1");
    ("1e3", "si", int 1000, Some "1e3");
    ("pi/180", "si", pi / int 180, Some "\xc2\xb0 rad^-1");
    ("1/3", "si", int 1 / int 3, None);
  ]

let table =
  let voc = function "si" -> Vocabulary.si | _ -> astro in
  group "Vocabulary.spell on common units"
    [
      cases
        ~name:(fun (name, v, _, _) -> strf "%s in %s" name v)
        "spells" rows
        (fun (_, v, u, expected) ->
          let voc = voc v in
          match expected with
          | Some s -> equal string s (text voc u)
          | None ->
              equal (option spelling) None (Vocabulary.spell voc u);
              equal string (Unit.to_string u) (text voc u));
    ]

(* Spelling *)

let radio =
  Vocabulary.v
    [
      ("Jy", Prefixable, jansky);
      ("sr", Prefixable, Unit.steradian);
      ("W", Prefixable, Unit.watt);
      ("rad", Prefixable, Unit.radian);
      ("m", Prefixable, Unit.metre);
      ("s", Prefixable, Unit.second);
    ]

let spell =
  group "Vocabulary.spell"
    [
      test "words spell a unit whose factors pass the algebra's bound"
        (fun () ->
          (* (deci s)^k is π^k, though (10π)^k's coefficient is past the
             algebra's 4096 bits from k = 1234; the words never form it. *)
          let voc = Vocabulary.v [ ("s", Prefixable, Unit.(int 10 * pi)) ] in
          List.iter
            (fun k ->
              equal (option spelling)
                (Some { decade = 0; words = [ w ~prefix:(-1) "s" k ] })
                (Vocabulary.spell voc Unit.(pi ** k)))
            [ 1234; 2305843009213693951 ]);
      test "a symbol whose unit is the unit is the spelling" (fun () ->
          equal (option spelling)
            (Some { decade = 0; words = [ w "W" 1 ] })
            (Vocabulary.spell radio Unit.watt));
      test "the first symbol whose unit is the unit spells it" (fun () ->
          equal string "L" (text Vocabulary.si Unit.litre);
          let voc =
            Vocabulary.v
              [
                ("m", Prefixable, Unit.metre);
                ("Bq", Prefixable, Unit.becquerel);
                ("Hz", Prefixable, Unit.hertz);
              ]
          in
          equal string "Bq" (text voc Unit.hertz));
      test "one word: a name with a prefix" (fun () ->
          let u =
            Unit.(decimal "1e-20" * kilogram / (radian ** 2) / (second ** 2))
          in
          equal (option spelling)
            (Some { decade = 0; words = [ w ~prefix:6 "Jy" 1; w "sr" (-1) ] })
            (Vocabulary.spell radio u);
          equal (option spelling)
            (Some { decade = 0; words = [ w ~prefix:9 "Hz" 1 ] })
            (Vocabulary.spell Vocabulary.si Unit.(giga hertz)));
      test "one word: no prefix before a prefixed entry" (fun () ->
          equal (option spelling)
            (Some { decade = 0; words = [ w "m" 3 ] })
            (Vocabulary.spell Vocabulary.si Unit.(metre ** 3)));
      test "one word: an entry without a number before one with" (fun () ->
          equal (option spelling)
            (Some { decade = 0; words = [ w ~prefix:3 "m" 3 ] })
            (Vocabulary.spell Vocabulary.si Unit.(kilo metre ** 3)));
      test "one word: a prefix on a positive exponent first" (fun () ->
          let voc =
            Vocabulary.v
              [ ("s", Prefixable, Unit.second); ("Hz", Prefixable, Unit.hertz) ]
          in
          equal string "GHz" (text voc Unit.(giga hertz)));
      test "one word: the least exponent" (fun () ->
          equal (option spelling)
            (Some { decade = 0; words = [ w "sr" (-1) ] })
            (Vocabulary.spell Vocabulary.si Unit.(radian ** -2)));
      test "one word: the earliest entry on a tie" (fun () ->
          let l_first =
            Vocabulary.v
              [ ("l", Prefixable, Unit.litre); ("L", Prefixable, Unit.litre) ]
          in
          let big_l_first =
            Vocabulary.v
              [ ("L", Prefixable, Unit.litre); ("l", Prefixable, Unit.litre) ]
          in
          equal string "kl" (text l_first Unit.(metre ** 3));
          equal string "kL" (text big_l_first Unit.(metre ** 3)));
      test "words: a name only where it shortens the spelling" (fun () ->
          equal string "kJ mol^-1" (text Vocabulary.si Unit.(kilo joule / mole));
          equal string "m s^-2"
            (text Vocabulary.si Unit.(metre / (second ** 2)));
          equal string "kg s^-3" (text Vocabulary.si Unit.(watt / (metre ** 2))));
      test "words: the earliest named word on a tie" (fun () ->
          let voc names =
            Vocabulary.v
              (List.map (fun s -> (s, Vocabulary.Prefixable, Unit.newton)) names
              @ [
                  ("kg", Bare, Unit.kilogram);
                  ("m", Prefixable, Unit.metre);
                  ("s", Prefixable, Unit.second);
                ])
          in
          let u = Unit.(newton * metre) in
          equal string "N m" (text (voc [ "N"; "X" ]) u);
          equal string "X m" (text (voc [ "X"; "N" ]) u));
      test "words: number words give a number that is no power of ten"
        (fun () ->
          equal (option spelling)
            (Some { decade = 0; words = [ w "au" 1; w "d" (-1) ] })
            (Vocabulary.spell Vocabulary.si Unit.(astronomical_unit / day)));
      test "words: a symbol word takes the entry of exponent 1 over a power"
        (fun () -> equal string "s^-2" (text Vocabulary.si Unit.(second ** -2)));
      test "words: positive exponents first, then entry order" (fun () ->
          equal string "rad s^-1" (text Vocabulary.si Unit.(radian / second));
          equal string "A m^2" (text Vocabulary.si Unit.(ampere * (metre ** 2))));
      cases ~name:fst "words: integer exponents give integer words"
        Unit.
          [
            ("N s", kilogram * metre / second);
            ("1e4 m", int 10_000 * metre);
            ("1e-4 m", decimal "1e-4" * metre);
          ]
        (fun (s, u) -> equal string s (text Vocabulary.si u));
      test "a fractional word only for a fractional exponent" (fun () ->
          let voc = Vocabulary.v [ ("sr", Prefixable, Unit.steradian) ] in
          equal (option spelling) None (Vocabulary.spell voc Unit.radian);
          equal string "rad" (text voc Unit.radian);
          equal (option spelling)
            (Some { decade = 0; words = [ w ~den:4 "sr" 1 ] })
            (Vocabulary.spell voc (Unit.root 2 Unit.radian)));
      test "a number no entry gives has no spelling" (fun () ->
          equal (option spelling) None
            (Vocabulary.spell Vocabulary.si Unit.(int 2 * metre));
          equal string "2 m" (text Vocabulary.si Unit.(int 2 * metre)));
      test "a symbol no entry holds has no spelling" (fun () ->
          equal (option spelling) None (Vocabulary.spell radio Unit.kelvin);
          equal (option spelling) None
            (Vocabulary.spell radio Unit.(pi * radian));
          equal string "K" (text radio Unit.kelvin));
      test "an exponent past int has no spelling" (fun () ->
          let x = Unit.symbol "x" in
          let voc = Vocabulary.v [ ("y", Prefixable, Unit.root max_int x) ] in
          equal (option spelling) None (Vocabulary.spell voc Unit.(x ** 2)));
      test "a rest past the coefficient's bound has no spelling" (fun () ->
          let x = Unit.symbol "x" in
          let y = Unit.((int 2 ** 4000) * x) in
          let voc = Vocabulary.v [ ("y", Prefixable, y) ] in
          equal (option spelling) None (Vocabulary.spell voc Unit.(x ** 2));
          equal string "x^2" (text voc Unit.(x ** 2)));
      test "an exponent within int is spelled, whatever its parts" (fun () ->
          (* The word's exponent is 2/3 of an entry whose exponent is 1/2^61;
             the entry's 1/3 power alone has a denominator past int. *)
          let x = Unit.symbol "x" in
          let voc =
            Vocabulary.v [ ("w", Prefixable, Unit.root (1 lsl 61) x) ]
          in
          equal (option spelling)
            (Some { decade = 0; words = [ w ~den:3 "w" 2 ] })
            (Vocabulary.spell voc (Unit.root (3 * (1 lsl 60)) x)));
    ]

(* The prefix and the decade *)

let micros = [ "u"; "\xc2\xb5"; "\xce\xbc" ]

let prefix_rule =
  group "Vocabulary.spell's prefix"
    [
      cases ~name:Fun.id
        "a micro that spells a symbol in any of its spellings is the decade"
        micros (fun p ->
          let voc =
            Vocabulary.v
              [
                ("m", Prefixable, Unit.metre); (p ^ "m", Bare, Unit.symbol "x");
              ]
          in
          equal (option spelling)
            (Some { decade = -6; words = [ w "m" 1 ] })
            (Vocabulary.spell voc Unit.(micro metre)));
      test "micro is written μ (U+03BC)" (fun () ->
          equal string "\xce\xbcm" (text Vocabulary.si Unit.(micro metre)));
      test "the prefix goes to the first word, in written order, that takes it"
        (fun () ->
          equal (option spelling)
            (Some { decade = 0; words = [ w "m" (-2); w ~prefix:3 "s" (-1) ] })
            (Vocabulary.spell Vocabulary.si
               Unit.(decimal "1e-3" / second / (metre ** 2)));
          equal string "electron cm^-3"
            (text astro Unit.(int 1_000_000 * electron / (metre ** 3))));
      test "a mass takes its prefix through g" (fun () ->
          equal (option spelling)
            (Some { decade = 0; words = [ w ~prefix:6 "g" 1; w "m" (-3) ] })
            (Vocabulary.spell Vocabulary.si
               Unit.(int 1000 * kilogram / (metre ** 3)));
          equal string "mg"
            (text Vocabulary.si Unit.(decimal "1e-6" * kilogram)));
      test "a bare word takes no prefix" (fun () ->
          let voc =
            Vocabulary.v
              [ ("h", Bare, Unit.hour); ("m", Prefixable, Unit.metre) ]
          in
          equal (option spelling)
            (Some { decade = 0; words = [ w "h" 1; w ~prefix:3 "m" 1 ] })
            (Vocabulary.spell voc Unit.(kilo (hour * metre))));
      test "a prefix whose prefixed symbol is a symbol is the decade" (fun () ->
          let voc =
            Vocabulary.v
              [ ("m", Prefixable, Unit.metre); ("km", Bare, Unit.symbol "x") ]
          in
          equal (option spelling)
            (Some { decade = 3; words = [ w "m" 1 ] })
            (Vocabulary.spell voc Unit.(kilo metre)));
      test "a power of ten no SI prefix names is the decade" (fun () ->
          equal (option spelling)
            (Some { decade = 4; words = [ w "m" 1 ] })
            (Vocabulary.spell metres Unit.(ten 4 * metre));
          equal (option spelling)
            (Some { decade = -31; words = [ w "m" 1 ] })
            (Vocabulary.spell metres Unit.(ten (-31) * metre)));
      test "a power of ten no word takes is the decade" (fun () ->
          let voc =
            Vocabulary.v
              [ ("s", Prefixable, Unit.second); ("h", Bare, Unit.hour) ]
          in
          equal (option spelling)
            (Some { decade = 3; words = [ w "s" 2 ] })
            (Vocabulary.spell voc Unit.(kilo (second ** 2)));
          let hours = Vocabulary.v [ ("h", Bare, Unit.hour) ] in
          equal string "1e3 h" (text hours Unit.(kilo hour)));
      test "an entry without symbols is read whole and as nothing else"
        (fun () ->
          let voc =
            Vocabulary.v
              [ ("%", Bare, ten (-2)); ("m", Prefixable, Unit.metre) ]
          in
          equal (option spelling)
            (Some { decade = 0; words = [ w "%" 1 ] })
            (Vocabulary.spell voc (ten (-2)));
          equal string "cm" (text voc Unit.(ten (-2) * metre));
          equal string "1e-4" (text voc (ten (-4))));
    ]

(* Names never change a unit *)

(* [pow u num den] is [u] to the power [num/den], term by term, so that no
   intermediate root leaves the algebra's bounds. *)
let pow u num den =
  let rec gcd a b = if b = 0 then abs a else gcd b (a mod b) in
  let times a b =
    let p = a * b in
    if a <> 0 && (p / a <> b || p = min_int) then
      failf "%a to the power %d/%d leaves int" Unit.pp u num den
    else p
  in
  let raise_term (t, n, d) =
    let g = gcd n den and g' = gcd d num in
    let n = times (n / g) (num / g') and d = times (d / g') (den / g) in
    if d < 0 then (t, -n, -d) else (t, n, d)
  in
  of_terms (List.map raise_term (Unit.terms u))

(* [read ~micro voc s] is the product of [s]'s words, each looked up in [voc]
   with its prefix written with [micro] for micro, times 10^decade. *)
let read ~micro voc (s : Vocabulary.spelling) =
  let word (x : Vocabulary.word) =
    let p =
      match x.prefix with 0 -> "" | -6 -> micro | k -> List.assoc k prefixes
    in
    match Vocabulary.lookup voc (p ^ x.symbol) with
    | Some u -> pow u x.num x.den
    | None -> failf "%s%s reads as nothing" p x.symbol
  in
  List.fold_left (fun acc x -> Unit.(acc * word x)) (ten s.decade) s.words

(* A unit near a vocabulary's entries: a product of a few of [units] to small
   rational powers, times a power of ten and sometimes a stray factor. *)
let near units =
  let open Gen in
  let power =
    let+ u = of_list ~pp:Unit.pp units
    and+ n = such_that (fun n -> n <> 0) (int_range (-3) 3)
    and+ d = frequency [ (5, constant 1); (1, int_range 2 3) ] in
    Unit.(root d u ** n)
  in
  let stray =
    frequency
      [
        (6, constant ~pp:Unit.pp Unit.one);
        (1, constant ~pp:Unit.pp Unit.pi);
        (1, constant ~pp:Unit.pp (Unit.int 2));
        (1, constant ~pp:Unit.pp electron);
      ]
  in
  let+ ps = list ~size:(int_range 0 3) power
  and+ k = frequency [ (2, constant 0); (3, int_range (-33) 33) ]
  and+ s = stray in
  List.fold_left Unit.( * ) Unit.(ten k * s) ps

let units_of entries = List.map (fun (_, _, u) -> u) entries
let radio_units = Unit.[ jansky; steradian; watt; radian; metre; second ]

(* Each vocabulary with its entries. *)
let vocabularies =
  Gen.of_list
    ~pp:(fun ppf (name, _, _) -> Format.pp_print_string ppf name)
    [
      ("si", Vocabulary.si, si_entries);
      ("astro", astro, astro_all);
      ( "radio",
        radio,
        List.map2
          (fun s u -> (s, Vocabulary.Prefixable, u))
          [ "Jy"; "sr"; "W"; "rad"; "m"; "s" ]
          radio_units );
    ]

let near_pairs =
  Gen.bind vocabularies (fun (name, voc, es) ->
      Gen.map (fun u -> (name, voc, es, u)) (near (units_of es)))
  |> Gen.with_pp (fun ppf (name, _, _, u) ->
      Format.fprintf ppf "%s, %a" name Unit.pp u)

(* [agrees es voc u] states that [spell voc u] is what the rule gives, as
   the reference computes it, for [voc] the vocabulary of [es]. *)
let agrees es voc u =
  match Spell_reference.spell es u with
  | exception Spell_reference.Overflow -> assume false
  | expected ->
      cover "spelled" (Option.is_some expected);
      cover "no spelling" (Option.is_none expected);
      equal (option spelling) expected (Vocabulary.spell voc u)

(* [integer_words voc u] states that when every exponent of [u] is an
   integer, so is every exponent of its spelling's words. *)
let integer_words voc u =
  let whole = List.for_all (fun (_, _, d) -> d = 1) (Unit.terms u) in
  cover "a unit with integer exponents" whole;
  cover "a unit with a fractional exponent" (not whole);
  match Vocabulary.spell voc u with
  | Some s when whole ->
      cover "spelled with integer exponents" true;
      List.iter
        (fun (x : Vocabulary.word) -> equal ~msg:x.symbol int 1 x.den)
        s.words
  | Some _ | None -> ()

let law =
  group "Names never change a unit"
    [
      prop ~count:500
        "a spelling's words, looked up, times its decade, are the unit"
        near_pairs (fun (_, voc, _, u) ->
          match Vocabulary.spell voc u with
          | None -> cover "no spelling" true
          | Some s ->
              cover "spelled" true;
              cover "with a prefix"
                (List.exists
                   (fun (x : Vocabulary.word) -> x.prefix <> 0)
                   s.words);
              cover "with a decade" (s.decade <> 0);
              cover "with a fractional exponent"
                (List.exists (fun (x : Vocabulary.word) -> x.den > 1) s.words);
              equal unit u (read ~micro:"\xce\xbc" voc s));
      prop ~count:500 "a spelling is the one the rule gives" near_pairs
        (fun (_, voc, es, u) -> agrees es voc u);
      prop ~count:500 "integer exponents in the unit give integer words"
        near_pairs (fun (_, voc, _, u) -> integer_words voc u);
    ]

(* The ranking *)

let items (s : Vocabulary.spelling) =
  List.length s.words + if s.decade = 0 then 0 else 1

let count_symbols u =
  List.length
    (List.filter
       (fun (t, _, _) ->
         match (t : Unit.term) with Symbol _ -> true | Prime _ | Pi -> false)
       (Unit.terms u))

(* An entry over two or more symbols whose number is a power of ten. *)
let is_named (_, _, u) =
  let exponent p =
    List.find_map
      (fun (t, n, d) ->
        match (t : Unit.term) with
        | Prime q when q = p -> Some (n, d)
        | Prime _ | Pi | Symbol _ -> None)
      (Unit.terms u)
  in
  let decimal_number =
    List.for_all
      (fun (t, _, _) ->
        match (t : Unit.term) with
        | Prime (2 | 5) | Symbol _ -> true
        | Prime _ | Pi -> false)
      (Unit.terms u)
    && exponent 2 = exponent 5
  in
  decimal_number && count_symbols u >= 2

(* A unit over some of [si]'s base symbols, each to a small power, times a
   power of ten: one word per symbol and a decade always spell it. *)
let over_base =
  let open Gen in
  let base = [ "kg"; "A"; "m"; "s"; "K"; "mol"; "cd"; "rad" ] in
  let+ symbols = subsequence ~pp:Format.pp_print_string base
  and+ exponents =
    list ~size:(constant 8)
      (frequency
         [
           (6, such_that (fun n -> n <> 0) (int_range (-4) 4)); (1, constant 0);
         ])
  and+ k = frequency [ (2, constant 0); (3, int_range (-33) 33) ] in
  let factor i s =
    let n = List.nth exponents i in
    if n = 0 then Unit.one
    else
      let u = Option.get (Vocabulary.lookup Vocabulary.si s) in
      Unit.(u ** n)
  in
  let u = List.fold_left Unit.( * ) (ten k) (List.mapi factor symbols) in
  u

let ranking =
  group "Vocabulary.spell's ranking"
    [
      prop "a unit an entry equals is spelled by the first such entry"
        (Gen.pair vocabularies (Gen.int_range 0 1000))
        (fun ((_, voc, es), i) ->
          let _, _, u = List.nth es (i mod List.length es) in
          let symbol, _, _ = List.find (fun (_, _, e) -> Unit.equal e u) es in
          equal (option spelling)
            (Some { decade = 0; words = [ w symbol 1 ] })
            (Vocabulary.spell voc u));
      prop ~count:500
        "a spelling has no more items than one word per symbol and a decade"
        (Gen.with_pp Unit.pp over_base) (fun u ->
          let n = count_symbols u in
          List.iter
            (fun (name, voc, es) ->
              let s = require_some ~msg:name (Vocabulary.spell voc u) in
              at_most ~msg:name int ~than:(n + 1) (items s);
              let named =
                List.exists
                  (fun (x : Vocabulary.word) ->
                    List.exists
                      (fun ((sym, _, _) as e) -> sym = x.symbol && is_named e)
                      es)
                  s.words
              in
              cover "a named word" named;
              if named then
                at_most ~msg:(name ^ ", named") int ~than:n (items s))
            [ ("si", Vocabulary.si, si_entries); ("astro", astro, astro_all) ]);
    ]

(* Names never change a unit, on hostile input *)

let pp_prefixing ppf p =
  Format.pp_print_string ppf
    (match p with Vocabulary.Prefixable -> "Prefixable" | Bare -> "Bare")

let pp_entry ppf (s, p, u) =
  Format.fprintf ppf "(%S, %a, %a)" s pp_prefixing p Unit.pp u

let pp_entries =
  Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ") pp_entry

let entry_symbols =
  [
    "m";
    "s";
    "g";
    "x";
    "a";
    "am";
    "d";
    "da";
    "dam";
    "Pa";
    "cd";
    "mm";
    "um";
    "\xc2\xb5s";
    "\xce\xbcK";
    "K";
    "h";
    "kat";
    "Gy";
    "y";
    "%";
    "Jy";
    "e";
    "km";
    "ms";
    "MK";
    "cm";
  ]

let base_units =
  [
    Unit.metre;
    Unit.second;
    Unit.kelvin;
    Unit.radian;
    Unit.symbol "x";
    Unit.scoped ~scope:"field 1" "adu";
  ]

let entry_unit =
  let open Gen in
  let power =
    let+ u = of_list ~pp:Unit.pp base_units
    and+ n = such_that (fun n -> n <> 0) (int_range (-3) 3)
    and+ d = frequency [ (4, constant 1); (1, int_range 2 4) ] in
    Unit.(root d u ** n)
  in
  let+ ps = list ~size:(int_range 0 2) power
  and+ k = frequency [ (3, constant 0); (2, int_range (-12) 12) ]
  and+ stray =
    frequency
      [
        (8, constant ~pp:Unit.pp Unit.one);
        (1, constant ~pp:Unit.pp Unit.pi);
        (1, constant ~pp:Unit.pp (Unit.int 3));
      ]
  in
  List.fold_left Unit.( * ) Unit.(ten k * stray) ps

let prefixing = Gen.of_list ~pp:pp_prefixing [ Vocabulary.Prefixable; Bare ]

(* Entries [Vocabulary.v] accepts. *)
let valid es =
  match Vocabulary.v es with _ -> true | exception Invalid_argument _ -> false

(* A drawn vocabulary: entries with symbols that collide with prefixes and with
   micro's spellings, units over a few symbols with rational exponents, powers
   of ten, π and scoped symbols. *)
let entries =
  let open Gen in
  let entry =
    let+ s = of_list ~pp:Format.pp_print_string entry_symbols
    and+ p = prefixing
    and+ u = entry_unit in
    (s, p, u)
  in
  (* Sometimes one entry's symbol with an SI prefix is itself a symbol. *)
  let collide es =
    let+ i = int_range 0 (List.length es - 1)
    and+ k = of_list ~pp:Format.pp_print_int (List.map fst prefixes)
    and+ p = prefixing
    and+ u = entry_unit
    and+ micro = of_list ~pp:Format.pp_print_string micros in
    let s, _, _ = List.nth es i in
    let prefix = if k = -6 then micro else List.assoc k prefixes in
    es @ [ (prefix ^ s, p, u) ]
  in
  let drawn = list ~size:(int_range 1 6) entry in
  frequency [ (2, drawn); (1, bind drawn collide) ]
  |> such_that valid |> with_pp pp_entries

(* [collisions es] is each SI prefix's power of ten that, on one of [es]'s
   symbols and in any of the prefix's spellings, spells another, with that
   symbol's unit. *)
let collisions es =
  let is_symbol str = List.exists (fun (s, _, _) -> s = str) es in
  let spellings = prefixes @ List.map (fun p -> (-6, p)) micros in
  List.concat_map
    (fun (s, _, u) ->
      List.filter_map
        (fun (k, p) -> if is_symbol (p ^ s) then Some (k, u) else None)
        spellings)
    es

(* A unit to spell: a product of a few entries' units to small rational powers,
   times a power of ten and sometimes a stray factor, or a unit from the
   algebra's own generator, with exponents at the int extremes. *)
let to_spell es =
  let open Gen in
  let entry_units = units_of es in
  let power =
    let+ u = of_list ~pp:Unit.pp entry_units
    and+ n = such_that (fun n -> n <> 0) (int_range (-4) 4)
    and+ d = frequency [ (4, constant 1); (1, int_range 2 3) ] in
    Unit.(root d u ** n)
  in
  let near =
    let+ ps = list ~size:(int_range 0 3) power
    and+ k =
      frequency
        ((match collisions es with
           | [] -> []
           | ks -> [ (3, of_list ~pp:Format.pp_print_int (List.map fst ks)) ])
        @ [
            (1, constant 0);
            (2, of_list ~pp:Format.pp_print_int (List.map fst prefixes));
            (2, int_range (-33) 33);
          ])
    and+ stray =
      frequency
        [
          (6, constant ~pp:Unit.pp Unit.one);
          (1, constant ~pp:Unit.pp Unit.pi);
          (1, constant ~pp:Unit.pp (Unit.int 2));
          (1, constant ~pp:Unit.pp (Unit.scoped ~scope:"field 2" "adu"));
        ]
    in
    within (fun () -> List.fold_left Unit.( * ) Unit.(ten k * stray) ps)
  in
  (* A collided symbol's unit with that prefix's power of ten. *)
  let collided =
    match collisions es with
    | [] -> []
    | ks ->
        let pp ppf (k, u) = Format.fprintf ppf "1e%d %a" k Unit.pp u in
        [
          ( 2,
            map
              (fun (k, u) -> within Unit.(fun () -> ten k * u))
              (of_list ~pp ks) );
        ]
  in
  frequency (collided @ [ (5, near); (1, map Option.some units) ])
  |> such_that Option.is_some |> map Option.get |> with_pp Unit.pp

let pp_drawn ppf (es, u) =
  Format.fprintf ppf "@[<v>entries [%a]@,unit %a@]" pp_entries es Unit.pp u

let drawn =
  Gen.bind entries (fun es -> Gen.map (fun u -> (es, u)) (to_spell es))
  |> Gen.with_pp pp_drawn

(* Entries where one symbol with micro, in one of its spellings, is a symbol,
   and a unit of 10^-6 times one entry's unit: that symbol's, which takes no
   micro, or another's, which can. *)
let micro_drawn =
  let open Gen in
  let drawn =
    let* es = entries in
    let n = List.length es in
    let* i = int_range 0 (n - 1) in
    let+ j = frequency [ (1, constant i); (2, int_range 0 (n - 1)) ]
    and+ micro = of_list ~pp:Format.pp_print_string micros
    and+ p = prefixing
    and+ u = entry_unit in
    let s, _, _ = List.nth es i and _, _, su = List.nth es j in
    (es @ [ (micro ^ s, p, u) ], Unit.(ten (-6) * su))
  in
  such_that (fun (es, _) -> valid es) drawn |> with_pp pp_drawn

(* [well_formed (es, u)] states what every spelling's shape is: reduced,
   non-zero exponents; one word per symbol; at most one prefix, never beside
   a decade; positive exponents before negative ones. *)
let well_formed (es, u) =
  match Vocabulary.spell (Vocabulary.v es) u with
  | None -> ()
  | Some s ->
      let rec gcd a b = if b = 0 then abs a else gcd b (a mod b) in
      List.iter
        (fun (x : Vocabulary.word) ->
          not_equal ~msg:x.symbol int 0 x.num;
          equal ~msg:x.symbol int 1 (gcd x.num x.den);
          less ~msg:x.symbol int ~than:0 (-x.den))
        s.words;
      let symbols = List.map (fun (x : Vocabulary.word) -> x.symbol) s.words in
      equal ~msg:"one word per symbol" int (List.length symbols)
        (List.length (List.sort_uniq String.compare symbols));
      let prefixed =
        List.filter (fun (x : Vocabulary.word) -> x.prefix <> 0) s.words
      in
      cover "a prefix" (prefixed <> []);
      at_most ~msg:"prefixed words" int ~than:1 (List.length prefixed);
      if prefixed <> [] then equal ~msg:"decade beside a prefix" int 0 s.decade;
      let signs = List.map (fun (x : Vocabulary.word) -> x.num < 0) s.words in
      equal ~msg:"positive exponents first" (list bool)
        (List.sort Bool.compare signs)
        signs

let hostile =
  group "Names never change a unit, on hostile input"
    [
      prop ~count:1000 "a spelling read back is the unit" drawn (fun (es, u) ->
          let voc = Vocabulary.v es in
          match Vocabulary.spell voc u with
          | None -> cover "no spelling" true
          | Some s ->
              cover "spelled" true;
              cover "with a prefix"
                (List.exists
                   (fun (x : Vocabulary.word) -> x.prefix <> 0)
                   s.words);
              cover "with a decade" (s.decade <> 0);
              cover "with a fractional exponent"
                (List.exists (fun (x : Vocabulary.word) -> x.den > 1) s.words);
              equal unit u (read ~micro:"\xce\xbc" voc s));
      prop ~count:500 "the SI spells any of the algebra's units faithfully"
        units (fun u ->
          let voc = Vocabulary.si in
          match Vocabulary.spell voc u with
          | None -> cover "no spelling" true
          | Some s ->
              cover "spelled" true;
              equal unit u (read ~micro:"\xce\xbc" voc s));
      prop "a micro word reads back in each of micro's spellings" micro_drawn
        (fun (es, u) ->
          let voc = Vocabulary.v es in
          let s = Vocabulary.spell voc u in
          let micro_word (x : Vocabulary.word) = x.prefix = -6 in
          cover "a micro word"
            (Option.fold ~none:false
               ~some:(fun (s : Vocabulary.spelling) ->
                 List.exists micro_word s.words)
               s);
          Option.iter
            (fun s ->
              List.iter
                (fun micro ->
                  equal ~msg:(strf "%S" micro) unit u (read ~micro voc s))
                micros)
            s);
      prop ~count:1000 "a spelling is the one the rule gives" drawn
        (fun (es, u) -> agrees es (Vocabulary.v es) u);
      prop ~count:1000 "a spelling is well formed" drawn well_formed;
      prop ~count:1000 "integer exponents in the unit give integer words" drawn
        (fun (es, u) -> integer_words (Vocabulary.v es) u);
    ]

let () =
  exit
    (run "Vocabulary"
       [
         constructors;
         lookup;
         si;
         table;
         spell;
         prefix_rule;
         law;
         ranking;
         hostile;
       ])
