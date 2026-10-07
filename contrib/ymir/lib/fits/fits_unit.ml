(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* FITS unit strings (FITS 4.0 §4.3): symbols of Tables 3 and 4 with the
   prefixes of Table 5, products by a space, '*' or '.', division by '/',
   powers by '**', '^' or trailing digits, and a leading power of ten. *)

open Ymir_units

let strf = Printf.sprintf

(* Vocabulary *)

let photon = Unit.symbol "photon"
let count = Unit.symbol "count"
let bit = Unit.symbol "bit"
let d s = Unit.decimal s

(* The standard's values (Table 4), not the current ones: a parsec is
   3.0857e16 m and an electron volt 1.6021765e-19 J. *)
let ev = Unit.(d "1.6021765e-19" * joule)

(* Table 3, Table 4 with a dagger, and the prefixable archive spellings. *)
let prefixable =
  Unit.
    [
      ("m", metre);
      ("g", gram);
      ("s", second);
      ("rad", radian);
      ("sr", steradian);
      ("K", kelvin);
      ("A", ampere);
      ("mol", mole);
      ("cd", candela);
      ("Hz", hertz);
      ("J", joule);
      ("W", watt);
      ("V", volt);
      ("N", newton);
      ("Pa", pascal);
      ("C", coulomb);
      ("Ohm", ohm);
      ("S", siemens);
      ("F", farad);
      ("Wb", weber);
      ("T", tesla);
      ("H", henry);
      ("lm", lumen);
      ("lx", lux);
      ("yr", int 31557600 * second);
      ("a", int 31557600 * second);
      ("eV", ev);
      ("pc", d "3.0857e16" * metre);
      ("Jy", d "1e-26" * watt / (metre ** 2) / hertz);
      ("mag", symbol "mag");
      ( "R",
        (int 10 ** 10)
        / (int 4 * pi)
        * photon / (metre ** 2) / second / steradian );
      ("G", d "1e-4" * tesla);
      ("barn", d "1e-28" * (metre ** 2));
      ("bit", bit);
      ("byte", int 8 * bit);
    ]

(* Table 4 without a dagger, and the archives' spellings. *)
let bare =
  Unit.
    [
      ("deg", degree);
      ("arcmin", arcminute);
      ("arcsec", arcsecond);
      ("mas", milli arcsecond);
      ("min", minute);
      ("h", hour);
      ("d", day);
      ("erg", d "1e-7" * joule);
      ("Ry", d "13.605692" * ev);
      ("solMass", d "1.9891e30" * kilogram);
      ("u", d "1.6605387e-27" * kilogram);
      ("solLum", d "3.8268e26" * watt);
      ("Angstrom", d "1e-10" * metre);
      ("solRad", d "6.9599e8" * metre);
      ("AU", d "1.49598e11" * metre);
      ("lyr", d "9.460730e15" * metre);
      ("count", count);
      ("ct", count);
      ("photon", photon);
      ("ph", photon);
      ("D", int 1 / int 3 * d "1e-29" * coulomb * metre);
      ("Sun", symbol "Sun");
      ("bin", symbol "bin");
      ("adu", symbol "adu");
      ("electron", symbol "electron");
      (* archives' spellings *)
      ("angstrom", d "1e-10" * metre);
      ("electrons", symbol "electron");
      ("ELECTRON", symbol "electron");
      ("ELECTRONS", symbol "electron");
      ("DN", symbol "DN");
      ("COUNTS", count);
      ("counts", count);
    ]

(* Data-set symbols: each file's own, scoped by the reader. *)
let data_set =
  [
    ("pix", "pix");
    ("pixel", "pix");
    ("voxel", "voxel");
    ("chan", "chan");
    ("beam", "beam");
  ]

let vocabulary =
  Vocabulary.v
    (List.map (fun (s, u) -> (s, Vocabulary.Prefixable, u)) prefixable
    @ List.map (fun (s, u) -> (s, Vocabulary.Bare, u)) bare)

(* Table 5 *)
let prefixes =
  [
    ("da", 1);
    ("y", -24);
    ("z", -21);
    ("a", -18);
    ("f", -15);
    ("p", -12);
    ("n", -9);
    ("u", -6);
    ("m", -3);
    ("c", -2);
    ("d", -1);
    ("h", 2);
    ("k", 3);
    ("M", 6);
    ("G", 9);
    ("T", 12);
    ("P", 15);
    ("E", 18);
    ("Z", 21);
    ("Y", 24);
  ]

let ten k = if k >= 0 then Unit.(int 10 ** k) else Unit.(one / (int 10 ** -k))

(* [word ~scope w] is the unit a symbol, possibly prefixed, names. *)
let word ~scope w =
  match List.assoc_opt w bare with
  | Some u -> Ok u
  | None -> (
      match List.assoc_opt w prefixable with
      | Some u -> Ok u
      | None -> (
          match List.assoc_opt w data_set with
          | Some name -> (
              match scope with
              | Some scope -> Ok (Unit.scoped ~scope name)
              | None ->
                  Error (strf "%s is a data set's own symbol; give ~scope" w))
          | None -> (
              let prefixed (p, k) =
                let n = String.length p in
                if String.length w > n && String.sub w 0 n = p then
                  Option.map
                    (fun u -> Unit.(ten k * u))
                    (List.assoc_opt
                       (String.sub w n (String.length w - n))
                       prefixable)
                else None
              in
              match List.find_map prefixed prefixes with
              | Some u -> Ok u
              | None -> Error (strf "%s is not a FITS unit" w))))

(* Parsing *)

exception Bad of string

let bad fmt = Printf.ksprintf (fun s -> raise (Bad s)) fmt
let is_letter c = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
let is_digit c = c >= '0' && c <= '9'
let rec gcd a b = if b = 0 then abs a else gcd b (a mod b)

(* A power: [num/den] reduced, [den > 0]. *)
let power u (n, d) =
  if n = 0 then Unit.one
  else
    let g = gcd n d in
    let n = n / g and d = d / g in
    Unit.(root d (if n >= 0 then u ** n else one / (u ** -n)))

(* Whole strings archives write, where FITS would read S as siemens. *)
let archive =
  [
    ("ELECTRONS/S", Unit.(symbol "electron" / second));
    ("ELECTRONS/SEC", Unit.(symbol "electron" / second));
    ("COUNTS/S", Unit.(count / second));
    ("COUNTS/SEC", Unit.(count / second));
  ]

let parse ?scope s =
  match List.assoc_opt (String.trim s) archive with
  | Some u -> Ok u
  | None -> (
      let n = String.length s in
      let pos = ref 0 in
      let peek () = if !pos < n then Some s.[!pos] else None in
      let skip_spaces () =
        while !pos < n && s.[!pos] = ' ' do
          incr pos
        done
      in
      let digits () =
        let st = !pos in
        while !pos < n && is_digit s.[!pos] do
          incr pos
        done;
        if !pos = st then bad "a number is missing at byte %d" (st + 1);
        match int_of_string_opt (String.sub s st (!pos - st)) with
        | Some v -> v
        | None -> bad "the number at byte %d is too large" (st + 1)
      in
      let sign () =
        match peek () with
        | Some '+' ->
            incr pos;
            1
        | Some '-' ->
            incr pos;
            -1
        | _ -> 1
      in
      (* inside parentheses: an integer, a decimal or a ratio, with a sign *)
      let rational () =
        let sg = sign () in
        let a = digits () in
        match peek () with
        | Some '/' ->
            incr pos;
            let b = digits () in
            if b = 0 then bad "a power divides by zero";
            (sg * a, b)
        | Some '.' ->
            incr pos;
            let st = !pos in
            let f = digits () in
            let k = !pos - st in
            if k > 9 then bad "the power's decimals are too many";
            let den = int_of_float (10. ** float_of_int k) in
            (sg * ((a * den) + f), den)
        | _ -> (sg * a, 1)
      in
      let paren_power () =
        incr pos;
        let r = rational () in
        if peek () <> Some ')' then
          bad "a power does not close at byte %d" (!pos + 1);
        incr pos;
        r
      in
      let power_of () =
        match peek () with
        | Some '*' when !pos + 1 < n && s.[!pos + 1] = '*' ->
            pos := !pos + 2;
            if peek () = Some '(' then Some (paren_power ())
            else
              let sg = sign () in
              Some (sg * digits (), 1)
        | Some '^' ->
            incr pos;
            if peek () = Some '(' then Some (paren_power ())
            else
              let sg = sign () in
              Some (sg * digits (), 1)
        | Some ('+' | '-') ->
            let sg = sign () in
            Some (sg * digits (), 1)
        | Some c when is_digit c -> Some (digits (), 1)
        | Some '(' -> Some (paren_power ())
        | _ -> None
      in
      let rec product () =
        let u = ref (term ()) in
        let continue = ref true in
        while !continue do
          let st = !pos in
          skip_spaces ();
          match peek () with
          | Some ('*' | '.') ->
              incr pos;
              skip_spaces ();
              u := Unit.(!u * term ())
          | Some c when (is_letter c || c = '(') && !pos > st ->
              u := Unit.(!u * term ())
          | _ ->
              pos := st;
              continue := false
        done;
        !u
      (* '/' divides by the product that follows it *)
      and quotient () =
        skip_spaces ();
        let u = ref (if peek () = Some '/' then Unit.one else product ()) in
        let continue = ref true in
        while !continue do
          skip_spaces ();
          match peek () with
          | Some '/' ->
              incr pos;
              skip_spaces ();
              u := Unit.(!u / product ())
          | _ -> continue := false
        done;
        !u
      and term () =
        match peek () with
        | Some '(' -> (
            incr pos;
            let u = quotient () in
            skip_spaces ();
            if peek () <> Some ')' then
              bad "a parenthesis does not close at byte %d" (!pos + 1);
            incr pos;
            match power_of () with Some p -> power u p | None -> u)
        | Some c when is_letter c -> (
            let st = !pos in
            while !pos < n && is_letter s.[!pos] do
              incr pos
            done;
            let w = String.sub s st (!pos - st) in
            match w with
            | ("log" | "ln" | "exp" | "sqrt") when peek () = Some '(' ->
                incr pos;
                let u = quotient () in
                skip_spaces ();
                if peek () <> Some ')' then bad "%s( does not close" w;
                incr pos;
                if w = "sqrt" && Unit.convertible u Unit.one then Unit.root 2 u
                else bad "%s() of a unit is no monomial" w
            | _ -> (
                let u =
                  match word ~scope w with
                  | Ok u -> u
                  | Error e -> raise (Bad e)
                in
                match power_of () with Some p -> power u p | None -> u))
        | Some c -> bad "%C at byte %d starts no unit" c (!pos + 1)
        | None -> bad "a unit is missing at the end"
      in
      (* a leading power of ten: 10**k, 10^k, 10+k or 10-k *)
      let scale () =
        if n >= 3 && String.sub s 0 2 = "10" && not (is_digit s.[2]) then begin
          pos := 2;
          match power_of () with
          | Some (k, 1) -> ten k
          | Some _ -> bad "the power of ten is not an integer"
          | None -> bad "10 is not a unit"
        end
        else Unit.one
      in
      match
        skip_spaces ();
        if !pos = n then Unit.one
        else
          let k = scale () in
          skip_spaces ();
          let u = if !pos = n then Unit.one else quotient () in
          skip_spaces ();
          if !pos <> n then
            bad "%C at byte %d follows the unit" s.[!pos] (!pos + 1);
          Unit.(k * u)
      with
      | u -> Ok u
      | exception Bad e -> Error (strf "%S: %s" s e)
      | exception Invalid_argument e -> Error (strf "%S: %s" s e))

(* Printing *)

let prefix_char k =
  List.find_map (fun (p, k') -> if k = k' then Some p else None) prefixes

let print u =
  (* Data-set symbols are no vocabulary's: they are written after the rest,
     by name. *)
  let own =
    List.filter_map
      (fun (t, num, den) ->
        match t with
        | Unit.Symbol { name; scope = Some scope } ->
            Some (name, scope, (num, den))
        | _ -> None)
      (Unit.terms u)
  in
  let without =
    List.fold_left
      (fun acc (name, scope, p) ->
        Unit.( / ) acc (power (Unit.scoped ~scope name) p))
      u own
  in
  let exponent (num, den) =
    if den = 1 then if num = 1 then "" else string_of_int num
    else strf "(%d/%d)" num den
  in
  let words =
    if Unit.equal without Unit.one then Ok ([], 0)
    else
      match Vocabulary.spell vocabulary without with
      | None -> Error (strf "%s has no FITS spelling" (Unit.to_string u))
      | Some { decade; words } -> (
          let text (w : Vocabulary.word) =
            match if w.prefix = 0 then Some "" else prefix_char w.prefix with
            | Some p -> Ok (p ^ w.symbol ^ exponent (w.num, w.den))
            | None ->
                Error (strf "%s needs a prefix FITS lacks" (Unit.to_string u))
          in
          match
            List.fold_right
              (fun w acc ->
                Result.bind acc (fun l -> Result.map (fun t -> t :: l) (text w)))
              words (Ok [])
          with
          | Ok l -> Ok (l, decade)
          | Error e -> Error e)
  in
  Result.map
    (fun (ws, decade) ->
      let ten = if decade = 0 then [] else [ strf "10**%d" decade ] in
      let sc = List.map (fun (name, _, p) -> name ^ exponent p) own in
      String.concat " " (ten @ ws @ sc))
    words
