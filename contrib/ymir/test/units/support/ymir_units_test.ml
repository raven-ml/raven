(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Witnesses, builders and generators of the units suites. *)

open Windtrap
open Ymir_units

let unit =
  Testable.with_compare Unit.compare
    (Testable.make ~pp:Unit.pp ~equal:Unit.equal)

let contains ~sub s =
  let n = String.length sub and len = String.length s in
  let rec at i = i + n <= len && (String.sub s i n = sub || at (i + 1)) in
  at 0

(* Building units *)

(* A factor [(b, n, d)] is the base [b] to the power [n/d]. *)
type base = Int of int | Pi | Symbol of string * string option

let base = function
  | Int n -> Unit.int n
  | Pi -> Unit.pi
  | Symbol (name, None) -> Unit.symbol name
  | Symbol (name, Some scope) -> Unit.scoped ~scope name

let power (b, n, d) = Unit.(root d (base b) ** n)
let product fs = List.fold_left (fun u f -> Unit.(u * power f)) Unit.one fs

let pp_factor ppf (b, n, d) =
  let pp_base ppf = function
    | Int n -> Format.fprintf ppf "int %d" n
    | Pi -> Format.pp_print_string ppf "pi"
    | Symbol (name, None) -> Format.fprintf ppf "symbol %S" name
    | Symbol (name, Some scope) ->
        Format.fprintf ppf "scoped ~scope:%S %S" scope name
  in
  Format.fprintf ppf "root %d (%a) ** %d" d pp_base b n

let pp_factors =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " * ")
    pp_factor

(* [of_terms ts] is the product of the terms [ts], last term first. *)
let of_terms ts =
  let factor (t, n, d) =
    let b =
      match (t : Unit.term) with
      | Prime p -> Int p
      | Pi -> Pi
      | Symbol { name; scope } -> Symbol (name, scope)
    in
    (b, n, d)
  in
  product (List.rev_map factor ts)

(* [rebuild u] is [u] built again from its terms. *)
let rebuild u = of_terms (Unit.terms u)

(* [symbols_of u] is the product of [u]'s symbol terms. *)
let symbols_of u =
  List.fold_left
    (fun acc (t, n, d) ->
      match (t : Unit.term) with
      | Symbol { name; scope } ->
          Unit.(acc * power (Symbol (name, scope), n, d))
      | Prime _ | Pi -> acc)
    Unit.one (Unit.terms u)

(* Bounds *)

(* An algebra error states one of the two bounds. *)
let is_bound msg =
  contains ~sub:"leaves int" msg || contains ~sub:"past 4096 bits" msg

(* [within f] is [Some (f ())], or [None] when [f] leaves the algebra's
   bounds. *)
let within f =
  match f () with
  | v -> Some v
  | exception Invalid_argument msg when is_bound msg -> None

(* Generators *)

let pp_int = Format.pp_print_int

(* The int extremes and their neighbours, and powers of two near them. *)
let wide =
  [
    max_int;
    max_int - 1;
    min_int + 1;
    min_int + 2;
    1 lsl 61;
    (1 lsl 61) - 1;
    (1 lsl 61) + 1;
    -(1 lsl 61);
    1 lsl 31;
    (1 lsl 31) - 1;
  ]

let small_exponent = Gen.such_that (fun n -> n <> 0) (Gen.int_range (-6) 6)

let numerator =
  Gen.frequency [ (7, small_exponent); (3, Gen.of_list ~pp:pp_int wide) ]

let denominator =
  Gen.frequency
    [
      (5, Gen.constant ~pp:pp_int 1);
      (3, Gen.int_range 2 12);
      ( 2,
        Gen.of_list ~pp:pp_int
          [ max_int; max_int - 1; 1 lsl 61; (1 lsl 61) - 1; 1 lsl 31; 3 ] );
    ]

(* Primes at the coefficient's 2^24 split and below 2^62, a composite of two
   such primes, and the coefficient's own small factors. *)
let wide_primes =
  [ 16777213; 16777259; 2147483647; 2305843009213693951; 4611686018427387847 ]

let small_ints = [ 2; 3; 5; 6; 7; 10; 12; 1000; 65537; max_int ]
let names = [ "m"; "kg"; "s"; "A"; "a"; "B"; "_"; "_x"; "x1"; "Pi"; "pi2" ]

(* Scopes that order differently as bytes and as their encoded text, and every
   class of byte the text encodes or keeps. *)
let scopes =
  [
    None;
    Some "z";
    Some "A";
    Some "a b";
    Some "%";
    Some "%41";
    Some "{}";
    Some "\xc3\xa9";
    Some "\000";
    Some "~._-:#/@+";
    Some "sha256:9f2c41#SCI";
  ]

let pp_scope ppf = function
  | None -> Format.pp_print_string ppf "None"
  | Some s -> Format.fprintf ppf "Some %S" s

let symbol =
  let open Gen in
  let+ name = of_list ~pp:Format.pp_print_string names
  and+ scope = of_list ~pp:pp_scope scopes in
  Symbol (name, scope)

let factor =
  let open Gen in
  let wide_exponent b =
    let+ n = numerator and+ d = denominator in
    (b, n, d)
  in
  let small b =
    let+ n = small_exponent and+ d = int_range 1 12 in
    (b, n, d)
  in
  frequency
    [
      (3, bind (of_list ~pp:pp_int small_ints) (fun n -> small (Int n)));
      (2, bind (of_list ~pp:pp_int wide_primes) (fun p -> wide_exponent (Int p)));
      (2, wide_exponent Pi);
      (4, bind symbol wide_exponent);
    ]
  |> with_pp pp_factor

(* The generators draw units the algebra builds. Any failure to build one,
   whether a bound or not, is the business of the bounds tests, which state the
   exception. *)
let buildable fs = match product fs with _ -> true | exception _ -> false

(* A unit of up to five factors, each base drawn with exponents at the int
   extremes. *)
let factors =
  Gen.such_that buildable (Gen.list ~size:(Gen.int_range 0 5) factor)
  |> Gen.with_pp pp_factors

let units = Gen.with_pp Unit.pp (Gen.map product factors)

(* Whether [u] holds an exponent of 2^60 or more in numerator or denominator. *)
let has_wide_exponent u =
  List.exists
    (fun (_, n, d) -> abs n >= 1 lsl 60 || d >= 1 lsl 60)
    (Unit.terms u)

(* Pairs that are often equal: a unit with a respelling of itself, or with
   another unit. *)
let near_pairs =
  let open Gen in
  let+ u = units and+ w = units and+ k = int_range 0 2 in
  match k with
  | 0 -> (u, rebuild u)
  | 1 -> (u, Unit.(rebuild u * one))
  | _ -> (u, w)
