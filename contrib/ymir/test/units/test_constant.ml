(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Measured constants: the published notation with its uncertainty, rounding
   once as Unit.ratio rounds with the sign restored, and the CODATA releases,
   checked against NIST's published values, the C library's correctly rounded
   decimal reader, and the exact relations the 2019 SI makes between them. *)

open Windtrap
open Ymir_units
open Ymir_units_test

let strf = Printf.sprintf
let k text = Constant.v ~name:"k" text Unit.metre

(* [first q] is the element of the scalar quantity [q] in its own unit. *)
let first q = (Nx.to_array (Quantity.value (Quantity.unit q) q)).(0)
let f64 k = first (Constant.quantity Nx.float64 k)
let u64 k = first (Constant.uncertainty Nx.float64 k)

(* [refused fn subject why] is the error of [fn] on the constant [k] whose
   [subject] has no value for the reason [why]. *)
let refused fn subject why =
  Invalid_argument (strf "%s: k: %s %s" fn subject why)

(* [reason msg] is the reason a message of [Unit.ratio] gives, after its
   ["which"]: ["is 0 in float16"] of [", which is 0 in float16"]. *)
let reason msg =
  let sep = ", which " in
  let rec find i =
    if i < 0 then failf "%S gives no reason" msg
    else if String.sub msg i (String.length sep) = sep then
      String.sub msg
        (i + String.length sep)
        (String.length msg - i - String.length sep)
    else find (i - 1)
  in
  find (String.length msg - String.length sep)

(* The notation *)

(* Each text with its value and its uncertainty as plain decimals, which the C
   library reads correctly rounded. *)
let texts =
  [
    ("6.67430(15)e-11", "6.67430e-11", "0.00015e-11");
    ("-2.00231930436092(36)", "-2.00231930436092", "0.00000000000036");
    ("10973731.568157(12)", "10973731.568157", "0.000012");
    ("1.5(25)", "1.5", "2.5");
    ("15(3)e2", "15e2", "3e2");
    ("0.0012(3)", "0.0012", "0.0003");
    ("007.50(1)", "7.50", "0.01");
    ("1(1)e-0", "1", "1");
    ("299792458", "299792458", "0");
    ("6.62607015e-34", "6.62607015e-34", "0");
    ("-1e-5", "-1e-5", "0");
  ]

let invalid =
  [
    ("", "is not a constant's value");
    ("-", "is not a constant's value");
    ("+1", "is not a constant's value");
    ("1.", "is not a constant's value");
    (".5", "is not a constant's value");
    ("1(", "is not a constant's value");
    ("1()", "is not a constant's value");
    ("1(2", "is not a constant's value");
    ("1(2)(3)", "is not a constant's value");
    ("1e5(2)", "is not a constant's value");
    ("1e", "is not a constant's value");
    ("1e-", "is not a constant's value");
    ("1e+5", "is not a constant's value");
    ("1E5", "is not a constant's value");
    ("--1", "is not a constant's value");
    (" 1", "is not a constant's value");
    ("1 ", "is not a constant's value");
    ("1.5 (2)", "is not a constant's value");
    ("0", "is zero");
    ("-0.000(1)", "is zero");
    ("1(0)", "has a zero uncertainty; an exact value has no parentheses");
    ("1.5(00)e3", "has a zero uncertainty; an exact value has no parentheses");
    ("4611686018427387904", "has a mantissa of 2^62 or more");
    ("461168601842738790.4", "has a mantissa of 2^62 or more");
    ("1(4611686018427387904)", "has an uncertainty of 2^62 or more");
    (* When the text breaks several rules, the first in the documented order:
       the grammar, the value, the uncertainty, the coefficient's bound. *)
    ("0(0)", "is zero");
    ("0(4611686018427387904)", "is zero");
    ("4611686018427387904(0)", "has a mantissa of 2^62 or more");
    ("1(0)e99999", "has a zero uncertainty; an exact value has no parentheses");
    ("0e99999", "is zero");
  ]

let notation =
  group "Constant.v reads the published notation"
    [
      cases
        ~name:(fun (t, _, _) -> t)
        "value and uncertainty" texts
        (fun (t, value, uncertainty) ->
          equal float_exact (float_of_string value) (f64 (k t));
          equal float_exact (float_of_string uncertainty) (u64 (k t)));
      cases ~name:fst "rejected text" invalid (fun (t, why) ->
          raises
            (Invalid_argument (strf "Constant.v: %S %s" t why))
            (fun () -> k t));
      test "a mantissa just below 2^62 is read" (fun () ->
          equal float_exact 4611686018427387903. (f64 (k "4611686018427387903"));
          equal float_exact 4611686018427387903.
            (u64 (k "1(4611686018427387903)")));
      test "a power of ten past the coefficient's bound raises" (fun () ->
          raises
            (Invalid_argument
               "Constant.v: the coefficient's numerator is past 4096 bits")
            (fun () -> k "1e5000");
          raises
            (Invalid_argument
               "Constant.v: the coefficient's denominator is past 4096 bits")
            (fun () -> k "1(1)e-5000"));
      test "name and unit are the ones given" (fun () ->
          let g = Constant.v ~name:"G" "6.67430(15)e-11" Unit.newton in
          equal string "G" (Constant.name g);
          equal unit Unit.newton (Constant.unit g));
      test "pp writes the name, the text and the unit's canonical text"
        (fun () ->
          let g =
            Constant.v ~name:"Newtonian constant of gravitation"
              "6.67430(15)e-11"
              Unit.((metre ** 3) / kilogram / (second ** 2))
          in
          equal string
            "Newtonian constant of gravitation = 6.67430(15)e-11 kg^-1 m^3 s^-2"
            (Format.asprintf "%a" Constant.pp g));
      test "pp leaves out the unit 1" (fun () ->
          let alpha =
            Constant.v ~name:"fine-structure constant" "7.2973525643(11)e-3"
              Unit.one
          in
          equal string "fine-structure constant = 7.2973525643(11)e-3"
            (Format.asprintf "%a" Constant.pp alpha));
    ]

(* Rounding *)

type dtype = D : string * (float, 'b) Nx.dtype -> dtype

let pp_dtype ppf (D (name, _)) = Format.pp_print_string ppf name

let float_dtypes =
  Gen.of_list ~pp:pp_dtype
    [
      D ("float64", Nx.float64);
      D ("float32", Nx.float32);
      D ("float16", Nx.float16);
      D ("bfloat16", Nx.bfloat16);
      D ("float8_e4m3", Nx.float8_e4m3);
      D ("float8_e5m2", Nx.float8_e5m2);
    ]

(* A drawn text: sign, whole digits, fraction digits, uncertainty digits and
   exponent. *)
type drawn = {
  negative : bool;
  whole : int;
  frac : string;
  digits : int option;
  exponent : int option;
}

let text_of d =
  String.concat ""
    [
      (if d.negative then "-" else "");
      string_of_int d.whole;
      (if d.frac = "" then "" else "." ^ d.frac);
      Option.fold ~none:"" ~some:(strf "(%d)") d.digits;
      Option.fold ~none:"" ~some:(strf "e%d") d.exponent;
    ]

let drawn =
  let open Gen in
  let nonzero d = d.whole <> 0 || String.exists (fun c -> c <> '0') d.frac in
  let digit = map string_of_int (int_range 0 9) in
  let texts =
    let+ negative = bool
    and+ whole = frequency [ (3, int_range 0 9); (1, int_range 0 999999) ]
    and+ frac = map (String.concat "") (list ~size:(int_range 0 8) digit)
    and+ digits = option (int_range 1 999)
    and+ exponent =
      option
        (frequency
           [
             (6, int_range (-40) 40);
             (1, int_range (-330) 330);
             (1, int_range (-12) 12);
           ])
    in
    { negative; whole; frac; digits; exponent }
  in
  texts |> such_that nonzero
  |> with_pp (fun ppf d -> Format.pp_print_string ppf (text_of d))

(* [magnitude d] and [uncertainty d] are the drawn value's magnitude and
   uncertainty as exact units, computed apart from Constant.v's reading. *)
let magnitude d =
  let frac = if d.frac = "" then "" else "." ^ d.frac in
  let e = Option.value ~default:0 d.exponent in
  Unit.decimal (strf "%d%se%d" d.whole frac e)

let uncertainty d u =
  let e = Option.value ~default:0 d.exponent - String.length d.frac in
  Unit.(int u * decimal (strf "1e%d" e))

(* [agrees fn subject d exact sign actual] states that [actual ()] is [sign] of
   [exact] rounded by [Unit.ratio] in [d], or raises for the reason it gives,
   naming [fn] and [subject]. *)
let agrees fn subject d exact sign actual =
  match Unit.ratio d exact Unit.one with
  | f ->
      cover "the value rounds" true;
      equal float_exact (sign f) (actual ())
  | exception Invalid_argument msg ->
      cover "the rounding raises" true;
      raises (refused fn subject (reason msg)) actual

let rounding =
  group "A constant rounds once, as Unit.ratio rounds"
    [
      prop "quantity is Unit.ratio of the magnitude, then the sign"
        Gen.(pair drawn float_dtypes)
        (fun (dr, D (_, d)) ->
          let c = k (text_of dr) in
          let sign = if dr.negative then Float.neg else Fun.id in
          cover "the constant is negative" dr.negative;
          agrees "Constant.quantity" (text_of dr) d (magnitude dr) sign
            (fun () -> first (Constant.quantity d c)));
      prop "uncertainty is Unit.ratio of the uncertainty, or zero when exact"
        Gen.(pair drawn float_dtypes)
        (fun (dr, D (_, d)) ->
          let c = k (text_of dr) in
          match dr.digits with
          | None ->
              cover "the constant is exact" true;
              equal float_exact 0. (first (Constant.uncertainty d c))
          | Some u ->
              agrees "Constant.uncertainty"
                ("the uncertainty of " ^ text_of dr)
                d (uncertainty dr u) Fun.id
                (fun () -> first (Constant.uncertainty d c)));
      test "a quantity is in the constant's unit and converts from it"
        (fun () ->
          let q = Constant.quantity Nx.float64 (k "1.5(2)e3") in
          equal unit Unit.metre (Quantity.unit q);
          equal float_exact 1.5
            (Nx.to_array (Quantity.value Unit.(kilo metre) q)).(0));
      test "a negative complex constant has a zero imaginary part" (fun () ->
          let z = first (Constant.quantity Nx.complex64 (k "-2.5")) in
          equal float_exact (-2.5) z.re;
          equal float_exact 0. z.im;
          equal float_exact 1. (Float.copy_sign 1. z.im));
      test "an integer dtype holds a constant whose magnitude it holds"
        (fun () ->
          equal int 42 (first (Constant.quantity Nx.int8 (k "42")));
          equal int (-127) (first (Constant.quantity Nx.int8 (k "-127")));
          equal int32 (-42l) (first (Constant.quantity Nx.int32 (k "-42.0")));
          equal int64 0L (first (Constant.uncertainty Nx.int64 (k "-42"))));
      test "a magnitude an integer dtype does not hold raises" (fun () ->
          raises
            (Invalid_argument
               "Constant.quantity: k: -128 has a magnitude int8 does not hold")
            (fun () -> Constant.quantity Nx.int8 (k "-128"));
          raises
            (Invalid_argument "Constant.quantity: k: 1.5 is not an integer")
            (fun () -> Constant.quantity Nx.int32 (k "1.5")));
      test "a negative constant in an unsigned dtype raises" (fun () ->
          raises
            (Invalid_argument
               "Constant.quantity: k: -5 is negative, which uint8 does not hold")
            (fun () -> Constant.quantity Nx.uint8 (k "-5"));
          raises
            (Invalid_argument
               "Constant.quantity: k: -5 is negative, which uint64 does not \
                hold") (fun () -> Constant.quantity Nx.uint64 (k "-5")));
      cases ~name:Fun.id
        "a negative constant in an unsigned dtype is refused before rounding"
        [ "-1.5"; "-300"; "-1e-400"; "-1.5(1)" ] (fun t ->
          raises
            (Invalid_argument
               (strf
                  "Constant.quantity: k: %s is negative, which uint8 does not \
                   hold"
                  t))
            (fun () -> Constant.quantity Nx.uint8 (k t)));
      test "a negative constant in bool holds no factor" (fun () ->
          raises (Invalid_argument "Constant.quantity: k: bool holds no factor")
            (fun () -> Constant.quantity Nx.bool (k "-1")));
      test "bool holds no constant and no uncertainty" (fun () ->
          raises (Invalid_argument "Constant.quantity: k: bool holds no factor")
            (fun () -> Constant.quantity Nx.bool (k "1(1)"));
          raises
            (Invalid_argument "Constant.uncertainty: k: bool holds no factor")
            (fun () -> Constant.uncertainty Nx.bool (k "1"));
          raises
            (Invalid_argument "Constant.uncertainty: k: bool holds no factor")
            (fun () -> Constant.uncertainty Nx.bool (k "1(1)")));
      test "bit holds no constant and no uncertainty" (fun () ->
          raises (Invalid_argument "Constant.quantity: k: bit holds no factor")
            (fun () -> Constant.quantity Nx.bit (k "-1"));
          raises (Invalid_argument "Constant.quantity: k: bit holds no factor")
            (fun () -> Constant.quantity Nx.bit (k "1(1)"));
          raises
            (Invalid_argument "Constant.uncertainty: k: bit holds no factor")
            (fun () -> Constant.uncertainty Nx.bit (k "1")));
    ]

(* CODATA *)

let releases = [ ("2018", Codata.v2018); ("2022", Codata.v2022) ]

let accessors =
  Codata.
    [
      ( "newtonian_gravitation",
        newtonian_gravitation,
        Unit.((metre ** 3) / kilogram / (second ** 2)) );
      ("fine_structure", fine_structure, Unit.one);
      ("vacuum_permeability", vacuum_permeability, Unit.(newton / (ampere ** 2)));
      ("vacuum_permittivity", vacuum_permittivity, Unit.(farad / metre));
      ("dalton", dalton, Unit.kilogram);
      ("electron_mass", electron_mass, Unit.kilogram);
      ("muon_mass", muon_mass, Unit.kilogram);
      ("tau_mass", tau_mass, Unit.kilogram);
      ("proton_mass", proton_mass, Unit.kilogram);
      ("neutron_mass", neutron_mass, Unit.kilogram);
      ("deuteron_mass", deuteron_mass, Unit.kilogram);
      ("triton_mass", triton_mass, Unit.kilogram);
      ("helion_mass", helion_mass, Unit.kilogram);
      ("alpha_particle_mass", alpha_particle_mass, Unit.kilogram);
      ("rydberg", rydberg, Unit.(metre ** -1));
      ("bohr_radius", bohr_radius, Unit.metre);
      ("classical_electron_radius", classical_electron_radius, Unit.metre);
      ("compton_wavelength", compton_wavelength, Unit.metre);
      ("thomson_cross_section", thomson_cross_section, Unit.(metre ** 2));
      ("hartree_energy", hartree_energy, Unit.joule);
      ("bohr_magneton", bohr_magneton, Unit.(joule / tesla));
      ("nuclear_magneton", nuclear_magneton, Unit.(joule / tesla));
      ( "electron_magnetic_moment",
        electron_magnetic_moment,
        Unit.(joule / tesla) );
      ("proton_magnetic_moment", proton_magnetic_moment, Unit.(joule / tesla));
      ("electron_g_factor", electron_g_factor, Unit.one);
    ]

let each_release f =
  List.concat_map
    (fun (y, r) -> List.map (fun a -> (y, r, a)) accessors)
    releases
  |> fun rows -> cases ~name:(fun (y, _, (n, _, _)) -> strf "%s %s" y n) f rows

let every_constant f =
  List.concat_map
    (fun (y, r) -> List.map (fun c -> (y, c)) (Codata.constants r))
    releases
  |> fun rows ->
  cases ~name:(fun (y, c) -> strf "%s %s" y (Constant.name c)) f rows

(* [named r name] is the constant NIST names [name] in [r]. *)
let named r name =
  List.find (fun c -> Constant.name c = name) (Codata.constants r)

(* The words of the rows that restate another row in other units. *)
let restatements =
  [
    "relationship";
    "energy equivalent";
    " in MeV";
    " in eV";
    " in u";
    " in Hz";
    " in MHz";
    " in K";
    " in inverse meter";
    "times c in Hz";
    "times hc in";
    "over h-bar c";
  ]

(* Values from NIST's tables of the 2018 and 2022 adjustments. *)
let published =
  [
    ("2018", Codata.v2018, Codata.newtonian_gravitation, "6.67430(15)e-11");
    ("2022", Codata.v2022, Codata.newtonian_gravitation, "6.67430(15)e-11");
    ("2018", Codata.v2018, Codata.electron_mass, "9.1093837015(28)e-31");
    ("2022", Codata.v2022, Codata.electron_mass, "9.1093837139(28)e-31");
    ("2018", Codata.v2018, Codata.fine_structure, "7.2973525693(11)e-3");
    ("2022", Codata.v2022, Codata.fine_structure, "7.2973525643(11)e-3");
    ("2018", Codata.v2018, Codata.dalton, "1.66053906660(50)e-27");
    ("2022", Codata.v2022, Codata.dalton, "1.66053906892(52)e-27");
    ("2018", Codata.v2018, Codata.vacuum_permeability, "1.25663706212(19)e-6");
    ("2022", Codata.v2022, Codata.vacuum_permeability, "1.25663706127(20)e-6");
    ("2018", Codata.v2018, Codata.electron_g_factor, "-2.00231930436256(35)");
    ("2022", Codata.v2022, Codata.electron_g_factor, "-2.00231930436092(36)");
    ("2018", Codata.v2018, Codata.rydberg, "10973731.568160(21)");
    ("2022", Codata.v2022, Codata.rydberg, "10973731.568157(12)");
  ]

(* [plain text] is [text]'s value without its uncertainty. *)
let plain text =
  match String.index_opt text '(' with
  | None -> text
  | Some i ->
      let j = String.index text ')' in
      String.sub text 0 i ^ String.sub text (j + 1) (String.length text - j - 1)

let pp_text c =
  let s = Format.asprintf "%a" Constant.pp c in
  let after = String.length (Constant.name c) + 3 in
  List.hd
    (String.split_on_char ' ' (String.sub s after (String.length s - after)))

let codata =
  group "Codata"
    [
      test "a release's year is its adjustment's" (fun () ->
          equal int 2018 (Codata.year Codata.v2018);
          equal int 2022 (Codata.year Codata.v2022));
      cases
        ~name:(fun (y, _, _, t) -> strf "%s %s" y t)
        "values are NIST's" published
        (fun (_, r, f, t) -> equal string t (pp_text (f r)));
      each_release
        "an accessor's constant is in the release, in the stated unit"
        (fun (_, r, (_, f, u)) ->
          equal unit u (Constant.unit (f r));
          satisfies ~claim:"is one of the release's constants" pass
            (fun c -> List.memq c (Codata.constants r))
            (f r));
      (* NIST's 2018 table has 354 rows, 81 exact and 79 restating another; its
         2022 table 355, 81 exact and 79 restating another, as gen/codata.py
         tallies them. *)
      cases ~name:fst "a release holds every other row of NIST's table"
        [ ("2018", (Codata.v2018, 194)); ("2022", (Codata.v2022, 195)) ]
        (fun (_, (r, n)) -> equal int n (List.length (Codata.constants r)));
      cases ~name:fst "names are distinct" releases (fun (_, r) ->
          let names = List.map Constant.name (Codata.constants r) in
          equal (list string)
            (List.sort_uniq String.compare names)
            (List.sort String.compare names));
      every_constant "no constant is exact or a restatement" (fun (_, c) ->
          let name = Constant.name c in
          greater float_exact ~than:0. (u64 c);
          List.iter
            (fun sub ->
              satisfies
                ~claim:(strf "does not contain %S" sub)
                string
                (fun n -> not (contains ~sub n))
                name)
            restatements);
      every_constant "float64 reads the published decimal correctly rounded"
        (fun (_, c) ->
          equal float_exact (float_of_string (plain (pp_text c))) (f64 c));
      every_constant "a unit holds the radian of an angle per or per angle"
        (fun (_, c) ->
          let name = Constant.name c in
          let has_radian =
            List.exists
              (function
                | Unit.Symbol { name = "rad"; _ }, _, _ -> true | _ -> false)
              (Unit.terms (Constant.unit c))
          in
          if contains ~sub:"gyromag. ratio" name then
            equal unit Unit.(radian / second / tesla) (Constant.unit c)
          else if
            String.starts_with ~prefix:"reduced " name
            && String.ends_with ~suffix:"Compton wavelength" name
          then equal unit Unit.(metre / radian) (Constant.unit c)
          else equal ~msg:name bool false has_radian);
      cases ~name:fst "the Fermi coupling constant is in GeV^-2" releases
        (fun (_, r) ->
          equal unit
            Unit.(giga electronvolt ** -2)
            (Constant.unit (named r "Fermi coupling constant")));
    ]

(* The exact relations of the 2019 SI. [agree c q] states that the quantity [q]
   computed from other constants equals the constant [c] within [c]'s standard
   uncertainty. *)

let q c = Constant.quantity Nx.float64 c
let in_unit u q = (Nx.to_array (Quantity.value u q)).(0)

let agree c rhs =
  let u = Constant.unit c in
  at_most float_exact ~than:(u64 c) (Float.abs (f64 c -. in_unit u rhs))

let relations =
  let open Quantity in
  let relation name f = cases ~name:fst name releases (fun (_, r) -> f r) in
  group "CODATA's constants satisfy the SI's relations"
    [
      relation "mu0 = 2 alpha h / (e^2 c)" (fun r ->
          Codata.fine_structure r |> q
          |> times
               Unit.(
                 int 2 * planck / ((elementary_charge ** 2) * speed_of_light))
          |> agree (Codata.vacuum_permeability r));
      relation "epsilon0 = 1 / (mu0 c^2)" (fun r ->
          Codata.vacuum_permeability r
          |> q |> pow (-1)
          |> per Unit.(speed_of_light ** 2)
          |> agree (Codata.vacuum_permittivity r));
      relation "m_e = 2 R h / (c alpha^2)" (fun r ->
          div (q (Codata.rydberg r)) (pow 2 (q (Codata.fine_structure r)))
          |> times Unit.(int 2 * planck / speed_of_light)
          |> agree (Codata.electron_mass r));
      relation "a0 = alpha / (4 pi R)" (fun r ->
          div (q (Codata.fine_structure r)) (q (Codata.rydberg r))
          |> per Unit.(int 4 * pi)
          |> agree (Codata.bohr_radius r));
      relation "r_e = alpha^2 a0" (fun r ->
          mul (pow 2 (q (Codata.fine_structure r))) (q (Codata.bohr_radius r))
          |> agree (Codata.classical_electron_radius r));
      relation "lambda_C = h / (m_e c)" (fun r ->
          q (Codata.electron_mass r)
          |> pow (-1)
          |> times Unit.(planck / speed_of_light)
          |> agree (Codata.compton_wavelength r));
      relation "sigma_e = 8 pi r_e^2 / 3" (fun r ->
          q (Codata.classical_electron_radius r)
          |> pow 2
          |> times Unit.(int 8 * pi / int 3)
          |> agree (Codata.thomson_cross_section r));
      relation "E_h = 2 R h c" (fun r ->
          q (Codata.rydberg r)
          |> times Unit.(int 2 * planck * speed_of_light)
          |> agree (Codata.hartree_energy r));
      relation "mu_B = e h / (4 pi m_e)" (fun r ->
          q (Codata.electron_mass r)
          |> pow (-1)
          |> times Unit.(elementary_charge * planck / (int 4 * pi))
          |> agree (Codata.bohr_magneton r));
      relation "mu_N = e h / (4 pi m_p)" (fun r ->
          q (Codata.proton_mass r)
          |> pow (-1)
          |> times Unit.(elementary_charge * planck / (int 4 * pi))
          |> agree (Codata.nuclear_magneton r));
      relation "mu_e = g_e mu_B / 2" (fun r ->
          mul (q (Codata.electron_g_factor r)) (q (Codata.bohr_magneton r))
          |> per (Unit.int 2)
          |> agree (Codata.electron_magnetic_moment r));
      relation "gamma_e = 2 |mu_e| / hbar, in rad s^-1 T^-1" (fun r ->
          q (Codata.electron_magnetic_moment r)
          |> map Nx.abs
          |> times Unit.(int 2 / hbar)
          |> agree (named r "electron gyromag. ratio"));
      relation "gamma_p = 2 mu_p / hbar, in rad s^-1 T^-1" (fun r ->
          q (Codata.proton_magnetic_moment r)
          |> times Unit.(int 2 / hbar)
          |> agree (named r "proton gyromag. ratio"));
      relation "a reduced Compton wavelength is lambda_C per 2 pi rad" (fun r ->
          q (Codata.compton_wavelength r)
          |> per Unit.(int 2 * pi * radian)
          |> agree (named r "reduced Compton wavelength"));
      relation "N_A m_u is the molar mass constant" (fun r ->
          q (Codata.dalton r)
          |> times Unit.avogadro
          |> agree (named r "molar mass constant"));
    ]

(* Each float dtype's edges *)

(* The outcome IEEE 754's roundTiesToEven gives a magnitude in a format with
   subnormals and an exponent unbounded above. *)
type outcome = Is of float | Zero | Subnormal | Overflows

(* An edge [(digits, e, outcome)] is the magnitude [digits]·10^[e]: the largest
   finite value, the tie above it and its neighbours; the smallest normal and
   the midpoint below it; the smallest subnormal and half of it; and a value
   just above a tie between two integers, which a detour through float64 rounds
   to the even one. *)
let edges =
  [
    ( D ("float64", Nx.float64),
      [
        ("17976931348623157", 292, Is 0x1.fffffffffffffp1023);
        ("179769313486231580", 291, Is 0x1.fffffffffffffp1023);
        ("179769313486231581", 291, Overflows);
        ("22250738585072014", -324, Is 0x1p-1022);
        ("22250738585072012", -324, Is 0x1p-1022);
        ("22250738585072011", -324, Subnormal);
        ("49406564584124654", -340, Subnormal);
        ("24703282292062328", -340, Subnormal);
        ("24703282292062327", -340, Zero);
      ] );
    ( D ("float32", Nx.float32),
      [
        ("340282346638528859", 21, Is 0x1.fffffep127);
        ("340282356779733661", 21, Is 0x1.fffffep127);
        ("340282356779733662", 21, Overflows);
        ("117549435082228751", -55, Is 0x1p-126);
        ("117549428075736430", -55, Is 0x1p-126);
        ("117549428075736429", -55, Subnormal);
        ("140129846432481707", -62, Subnormal);
        ("700649232162408536", -63, Subnormal);
        ("700649232162408535", -63, Zero);
        ("16777217", 0, Is 16777216.);
        ("16777219", 0, Is 16777220.);
        ("167772170000000001", -10, Is 16777218.);
      ] );
    ( D ("bfloat16", Nx.bfloat16),
      [
        ("338953138925153547", 21, Is 0x1.fep127);
        ("339617752923046005", 21, Is 0x1.fep127);
        ("339617752923046006", 21, Overflows);
        ("117090257601438795", -55, Is 0x1p-126);
        ("117090257601438794", -55, Subnormal);
        ("918354961579912116", -58, Subnormal);
        ("459177480789956058", -58, Subnormal);
        ("459177480789956057", -58, Zero);
        ("257", 0, Is 256.);
        ("259", 0, Is 260.);
        ("2570000000001", -10, Is 258.);
      ] );
    ( D ("float16", Nx.float16),
      [
        ("65504", 0, Is 65504.);
        ("655199999999", -7, Is 65504.);
        ("65520", 0, Overflows);
        ("6103515625", -14, Is 0x1p-14);
        ("61005353927612305", -21, Is 0x1p-14);
        ("61005353927612304", -21, Subnormal);
        ("59604644775390625", -24, Subnormal);
        ("298023223876953126", -25, Subnormal);
        ("298023223876953125", -25, Zero);
        ("2049", 0, Is 2048.);
        ("2051", 0, Is 2052.);
        ("20490000000001", -10, Is 2050.);
      ] );
    ( D ("float8_e4m3", Nx.float8_e4m3),
      [
        ("448", 0, Is 448.);
        ("464", 0, Is 448.);
        ("464000000000001", -12, Overflows);
        ("15625", -6, Is 0x1p-6);
        ("146484375", -10, Is 0x1p-6);
        ("146484374", -10, Subnormal);
        ("1953125", -9, Subnormal);
        ("9765626", -10, Subnormal);
        ("9765625", -10, Zero);
        ("17", 0, Is 16.);
        ("19", 0, Is 20.);
        ("170000000001", -10, Is 18.);
      ] );
    ( D ("float8_e5m2", Nx.float8_e5m2),
      [
        ("57344", 0, Is 57344.);
        ("61439999", -3, Is 57344.);
        ("61440", 0, Overflows);
        ("6103515625", -14, Is 0x1p-14);
        ("5340576171875", -17, Is 0x1p-14);
        ("5340576171874", -17, Subnormal);
        ("152587890625", -16, Subnormal);
        ("762939453126", -17, Subnormal);
        ("762939453125", -17, Zero);
        ("9", 0, Is 8.);
        ("11", 0, Is 12.);
        ("90000000001", -10, Is 10.);
      ] );
  ]

let why name = function
  | Is _ -> assert false
  | Zero -> "is 0 in " ^ name
  | Subnormal -> "is subnormal in " ^ name
  | Overflows -> "overflows " ^ name

let edge_rows =
  List.concat_map
    (fun ((D (name, _) as d), rows) -> List.map (fun e -> (name, d, e)) rows)
    edges

let edge_name (name, _, (digits, e, _)) = strf "%s %se%d" name digits e

let test_edge (_, D (name, d), (digits, e, outcome)) =
  let magnitude = strf "%se%d" digits e in
  let positive = k magnitude in
  let negative = k ("-" ^ magnitude) in
  let uncertain_text = strf "1(%s)e%d" digits e in
  let uncertain = k uncertain_text in
  let value c = first (Constant.quantity d c) in
  let unc c = first (Constant.uncertainty d c) in
  match outcome with
  | Is x ->
      equal ~msg:"quantity" float_exact x (value positive);
      equal ~msg:"negative quantity" float_exact (-.x) (value negative);
      equal ~msg:"uncertainty" float_exact x (unc uncertain)
  | o ->
      let q_fails text = refused "Constant.quantity" text (why name o) in
      raises ~msg:"quantity" (q_fails magnitude) (fun () -> value positive);
      raises ~msg:"negative quantity"
        (q_fails ("-" ^ magnitude))
        (fun () -> value negative);
      raises ~msg:"uncertainty"
        (refused "Constant.uncertainty"
           ("the uncertainty of " ^ uncertain_text)
           (why name o))
        (fun () -> unc uncertain)

(* A complex dtype rounds as its component: complex64 as float32, complex128 as
   float64. *)
let complex_edges =
  let rows name =
    List.assoc name (List.map (fun (D (n, _), r) -> (n, r)) edges)
  in
  let quantity d c = first (Constant.quantity d c) in
  List.map (fun r -> ("complex128", quantity Nx.complex128, r)) (rows "float64")
  @ List.map (fun r -> ("complex64", quantity Nx.complex64, r)) (rows "float32")

let test_complex_edge (name, value, (digits, e, outcome)) =
  let magnitude = strf "%se%d" digits e in
  List.iter
    (fun (sign, text) ->
      match outcome with
      | Is x ->
          let (z : Complex.t) = value (k text) in
          equal ~msg:text
            (pair float_exact float_exact)
            (sign x, 0.)
            (z.re, z.im)
      | o ->
          raises ~msg:text
            (refused "Constant.quantity" text (why name o))
            (fun () -> value (k text)))
    [ (Fun.id, magnitude); (Float.neg, "-" ^ magnitude) ]

let float_edges =
  group "A constant rounds once at each float dtype's edges"
    [
      cases ~name:edge_name "quantity and uncertainty" edge_rows test_edge;
      cases
        ~name:(fun (n, _, (digits, e, _)) -> strf "%s %se%d" n digits e)
        "a complex dtype rounds as its component" complex_edges
        test_complex_edge;
      test "the uncertainty of an exact constant is a positive zero" (fun () ->
          List.iter
            (fun (D (name, d)) ->
              let z = first (Constant.uncertainty d (k "-2.5")) in
              equal ~msg:name float_exact 0. z;
              equal ~msg:name float_exact 1. (Float.copy_sign 1. z))
            (List.map fst edges);
          let z = first (Constant.uncertainty Nx.complex64 (k "-2.5")) in
          equal (pair float_exact float_exact) (0., 0.) (z.re, z.im));
    ]

(* Integer, unsigned and int4 dtypes *)

let not_held fn subject dtype =
  refused fn subject (strf "has a magnitude %s does not hold" dtype)

let quantity_of d text = first (Constant.quantity d (k text))
let uncertainty_of d text = first (Constant.uncertainty d (k text))
let q_fn = "Constant.quantity"
let u_fn = "Constant.uncertainty"

let integer_cases =
  [
    ( "int4 holds 7 and -7",
      fun () ->
        equal int 7 (quantity_of Nx.int4 "7");
        equal int (-7) (quantity_of Nx.int4 "-7") );
    ( "int4 does not hold 8, nor -8, whose magnitude is 8",
      fun () ->
        raises (not_held q_fn "8" "int4") (fun () -> quantity_of Nx.int4 "8");
        raises (not_held q_fn "-8" "int4") (fun () -> quantity_of Nx.int4 "-8")
    );
    ( "uint4 holds 15, not 16",
      fun () ->
        equal int 15 (quantity_of Nx.uint4 "15");
        raises (not_held q_fn "16" "uint4") (fun () ->
            quantity_of Nx.uint4 "16") );
    ( "int8 holds 127, not 128",
      fun () ->
        equal int 127 (quantity_of Nx.int8 "127");
        raises (not_held q_fn "128" "int8") (fun () ->
            quantity_of Nx.int8 "128") );
    ( "uint8 holds 255, not 256",
      fun () ->
        equal int 255 (quantity_of Nx.uint8 "255");
        raises (not_held q_fn "256" "uint8") (fun () ->
            quantity_of Nx.uint8 "256") );
    ( "int16 holds -32767, not -32768",
      fun () ->
        equal int (-32767) (quantity_of Nx.int16 "-32767");
        raises (not_held q_fn "-32768" "int16") (fun () ->
            quantity_of Nx.int16 "-32768") );
    ( "uint16 holds 65535",
      fun () -> equal int 65535 (quantity_of Nx.uint16 "65535") );
    ( "int32 holds -2147483647, not -2147483648",
      fun () ->
        equal int32 (-2147483647l) (quantity_of Nx.int32 "-2147483647");
        raises (not_held q_fn "-2147483648" "int32") (fun () ->
            quantity_of Nx.int32 "-2147483648") );
    ( "uint32 holds 2^31 and 2^32 - 1 as their bits, not 2^32",
      fun () ->
        equal int32 Int32.min_int (quantity_of Nx.uint32 "2147483648");
        equal int32 (-1l) (quantity_of Nx.uint32 "4294967295");
        raises (not_held q_fn "4294967296" "uint32") (fun () ->
            quantity_of Nx.uint32 "4294967296") );
    ( "int64 holds -9223372036854775800, not 9223372036854775810",
      fun () ->
        equal int64 (-9223372036854775800L)
          (quantity_of Nx.int64 "-9.22337203685477580e18");
        raises (not_held q_fn "9.22337203685477581e18" "int64") (fun () ->
            quantity_of Nx.int64 "9.22337203685477581e18") );
    ( "uint64 holds 18446744073709551000 as its bits, not 18446744073709552000",
      fun () ->
        equal int64 (-616L) (quantity_of Nx.uint64 "1.8446744073709551e19");
        raises (not_held q_fn "1.8446744073709552e19" "uint64") (fun () ->
            quantity_of Nx.uint64 "1.8446744073709552e19") );
    ( "an integer constant's uncertainty is the integer of its digits",
      fun () ->
        equal int32 3l (uncertainty_of Nx.int32 "42(3)");
        equal int 3 (uncertainty_of Nx.uint8 "4.2(3)e1") );
    ( "a non-integer uncertainty raises, naming Constant.uncertainty",
      fun () ->
        equal int32 42l (quantity_of Nx.int32 "42.0(5)");
        raises
          (Invalid_argument
             "Constant.uncertainty: k: the uncertainty of 42.0(5) is not an \
              integer") (fun () -> uncertainty_of Nx.int32 "42.0(5)") );
    ( "an uncertainty the dtype does not hold raises, naming \
       Constant.uncertainty",
      fun () ->
        raises (not_held u_fn "the uncertainty of 1(256)" "uint8") (fun () ->
            uncertainty_of Nx.uint8 "1(256)") );
    ( "a negative constant's uncertainty, never negative, is held by an \
       unsigned dtype",
      fun () -> equal int 2 (uncertainty_of Nx.uint8 "-5(2)") );
    ( "an exact constant's uncertainty is 0 in every integer dtype",
      fun () ->
        equal int 0 (uncertainty_of Nx.int4 "-7");
        equal int 0 (uncertainty_of Nx.uint4 "7");
        equal int 0 (uncertainty_of Nx.uint8 "-7");
        equal int32 0l (uncertainty_of Nx.uint32 "7");
        equal int64 0L (uncertainty_of Nx.uint64 "-7") );
  ]

let integers =
  group "A constant in an integer dtype"
    [
      cases ~name:fst "holds its magnitude or raises" integer_cases
        (fun (_, f) -> f ());
    ]

(* Hostile notation *)

let hostile =
  [
    "\000";
    "1\000";
    "1.5(2)\0003";
    "\xef\xbc\x91";
    "1(\xd9\xa3)";
    "1,5";
    "1_000";
    "0x1p3";
    "inf";
    "-inf";
    "nan";
    "1e5.5";
    "1.5.5";
    "1..5";
    "1.e5";
    "-.5";
    "1.(2)";
    "1(2)e";
    "1(2)e-";
    "1(-2)";
    "1(+2)";
    "1(2.5)";
    "1( 2)";
    "1(2 )";
    "(2)";
    "()";
    ")";
    "1)";
    "1(2)3";
    "1(2)(";
    "1(2)e5(3)";
    "1(2)E5";
    "1e5e5";
    "e5";
    "-e5";
    "1e--5";
    "1e-+5";
    "1 e5";
    "1(2) e5";
    "1\n";
    "\t1";
    "-(1)";
    "--";
    "\xe2\x88\x921";
  ]

let notation_hostile =
  group "Constant.v on hostile text"
    [
      cases ~name:(strf "%S") "is not a constant's value" hostile (fun t ->
          raises
            (Invalid_argument
               (strf "Constant.v: %S is not a constant's value" t))
            (fun () -> k t));
      test "the interface's example of a text outside the grammar" (fun () ->
          raises
            (Invalid_argument
               {|Constant.v: "6.67430(15" is not a constant's value|})
            (fun () -> k "6.67430(15"));
      cases ~name:(strf "%S") "a zero value is zero, whatever its uncertainty"
        [ "0.0(5)"; "-0e5"; "000" ] (fun t ->
          raises
            (Invalid_argument (strf "Constant.v: %S is zero" t))
            (fun () -> k t));
      test "leading zeros do not count toward the mantissa's bound" (fun () ->
          equal float_exact 1. (f64 (k "0000000000000000000000001"));
          equal float_exact 1e-25 (f64 (k "0.0000000000000000000000001"));
          equal float_exact 1. (u64 (k "1(0000000000000000000000001)"));
          equal float_exact 1e-5 (f64 (k "1e-0000000000000000000000005")));
      cases ~name:fst "an exponent past the coefficient's bound raises"
        [
          ("1e99999999999999999999", "numerator");
          ("1e-99999999999999999999", "denominator");
          ("1(1)e99999999999999999999", "numerator");
          ("1e4611686018427387904", "numerator");
          ("1e1234", "numerator");
          ("1.05e1233", "numerator");
          ("1e-1234", "denominator");
          ("1(2)e1233", "numerator");
        ]
        (fun (t, part) ->
          raises
            (Invalid_argument
               (strf "Constant.v: the coefficient's %s is past 4096 bits" part))
            (fun () -> k t));
      test "a coefficient just within 4096 bits is read" (fun () ->
          let within t = equal ~msg:t string "k" (Constant.name (k t)) in
          List.iter within [ "1e1233"; "1.04e1233"; "1e-1233"; "1(1)e1233" ]);
    ]

(* What a release holds *)

(* Rows of NIST's tables a release keeps: ratios and dimensionless quantities,
   atomic, natural and Planck units, and other measured rows, with NIST's value
   and its unit. *)
let kept =
  Unit.
    [
      ("2018", "proton-electron mass ratio", "1836.15267343(11)", one);
      ("2022", "proton-electron mass ratio", "1836.152673426(32)", one);
      ( "2018",
        "Sackur-Tetrode constant (1 K, 100 kPa)",
        "-1.15170753706(45)",
        one );
      ( "2022",
        "Sackur-Tetrode constant (1 K, 100 kPa)",
        "-1.15170753496(47)",
        one );
      ("2018", "weak mixing angle", "0.22290(30)", one);
      ("2022", "W to Z mass ratio", "0.88145(13)", one);
      ("2022", "inverse fine-structure constant", "137.035999177(21)", one);
      ( "2022",
        "electron mag. mom. to Bohr magneton ratio",
        "-1.00115965218046(18)",
        one );
      ("2018", "atomic unit of time", "2.4188843265857(47)e-17", second);
      ("2022", "atomic unit of energy", "4.3597447222060(48)e-18", joule);
      ("2022", "natural unit of time", "1.28808866644(40)e-21", second);
      ( "2018",
        "natural unit of momentum",
        "2.73092453075(82)e-22",
        kilogram * metre / second );
      ("2022", "Planck mass", "2.176434(24)e-8", kilogram);
      ("2018", "Planck temperature", "1.416784(16)e32", kelvin);
      ("2018", "characteristic impedance of vacuum", "376.730313668(57)", ohm);
      ("2022", "unified atomic mass unit", "1.66053906892(52)e-27", kilogram);
      ("2018", "neutron-proton mass difference", "2.30557435(82)e-30", kilogram);
      ("2018", "proton rms charge radius", "8.414(19)e-16", metre);
      ("2022", "proton rms charge radius", "8.4075(64)e-16", metre);
      ("2022", "alpha particle rms charge radius", "1.6785(21)e-15", metre);
      ("2022", "Angstrom star", "1.00001495(90)e-10", metre);
      ("2018", "molar mass constant", "0.99999999965(30)e-3", kilogram / mole);
      ("2022", "molar mass constant", "1.00000000105(31)e-3", kilogram / mole);
      ( "2022",
        "shielded proton gyromag. ratio",
        "2.675153194(11)e8",
        radian / second / tesla );
    ]

(* Rows a release leaves out: the exact ones, which are units, and those that
   restate another row in other units. *)
let left_out =
  [
    ("2018", "alpha particle rms charge radius", "not in the 2018 table");
    ("2022", "speed of light in vacuum", "exact");
    ("2022", "Planck constant", "exact");
    ("2022", "reduced Planck constant", "exact");
    ("2022", "elementary charge", "exact");
    ("2022", "Boltzmann constant", "exact");
    ("2022", "Avogadro constant", "exact");
    ("2018", "molar gas constant", "exact");
    ("2022", "Faraday constant", "exact");
    ("2022", "Stefan-Boltzmann constant", "exact");
    ("2018", "Josephson constant", "exact");
    ("2022", "von Klitzing constant", "exact");
    ("2022", "atomic unit of action", "exact");
    ("2018", "atomic unit of charge", "exact");
    ("2022", "natural unit of velocity", "exact");
    ("2022", "molar Planck constant", "exact");
    ("2018", "hyperfine transition frequency of Cs-133", "exact");
    ("2022", "electron volt", "exact");
    ("2022", "Wien wavelength displacement law constant", "exact");
    ("2022", "atomic mass unit-kilogram relationship", "a relationship");
    ("2018", "hartree-kilogram relationship", "a relationship");
    ("2022", "electron mass energy equivalent", "an energy equivalent");
    ("2018", "electron mass energy equivalent in MeV", "in MeV");
    ("2022", "electron mass in u", "in u");
    ("2022", "Bohr magneton in eV/T", "in eV");
    ("2018", "Bohr magneton in Hz/T", "in Hz");
    ("2022", "Bohr magneton in K/T", "in K");
    ("2018", "Bohr magneton in inverse meter per tesla", "in m^-1");
    ("2022", "Rydberg constant times c in Hz", "in Hz");
    ("2018", "Rydberg constant times hc in J", "restates the Hartree energy");
    ( "2022",
      "Newtonian constant of gravitation over h-bar c",
      "restates G in other units" );
    ("2018", "natural unit of momentum in MeV/c", "in MeV");
    ("2022", "Planck mass energy equivalent in GeV", "an energy equivalent");
    ("2022", "electron gyromag. ratio in MHz/T", "in Hz");
    ("2018", "Hartree energy in eV", "in eV");
    ("2022", "neutron-proton mass difference in u", "in u");
  ]

let release_of = function "2018" -> Codata.v2018 | _ -> Codata.v2022

let find r name =
  List.find_opt (fun c -> Constant.name c = name) (Codata.constants r)

let membership =
  group "A release holds NIST's measured rows"
    [
      cases
        ~name:(fun (y, n, _, _) -> strf "%s %s" y n)
        "kept, with NIST's value and unit" kept
        (fun (y, n, text, u) ->
          match find (release_of y) n with
          | None -> failf "CODATA %s has no %S" y n
          | Some c ->
              equal string text (pp_text c);
              equal unit u (Constant.unit c));
      cases
        ~name:(fun (y, n, why) -> strf "%s %s (%s)" y n why)
        "left out" left_out
        (fun (y, n, _) ->
          equal (option string) None
            (Option.map Constant.name (find (release_of y) n)));
    ]

(* The interface's examples *)

let documented =
  group "The interface's error examples"
    [
      test "G is 0 in float8_e4m3" (fun () ->
          raises
            (Invalid_argument
               "Constant.quantity: Newtonian constant of gravitation: \
                6.67430(15)e-11 is 0 in float8_e4m3") (fun () ->
              Constant.quantity Nx.float8_e4m3
                (Codata.newtonian_gravitation Codata.v2022)));
      test "the electron g factor is negative, which uint8 does not hold"
        (fun () ->
          raises
            (Invalid_argument
               "Constant.quantity: electron g factor: -2.00231930436092(36) is \
                negative, which uint8 does not hold") (fun () ->
              Constant.quantity Nx.uint8 (Codata.electron_g_factor Codata.v2022)));
    ]

let () =
  exit
    (run "Constant"
       [
         notation;
         notation_hostile;
         rounding;
         float_edges;
         integers;
         codata;
         membership;
         relations;
         documented;
       ])
