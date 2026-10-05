(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

type term =
  | Prime of int
  | Pi
  | Symbol of { name : string; scope : string option }

(* A unit is its terms in canonical order, with reduced exponents and none zero,
   and its canonical text. Every constructor builds this form, so structural
   equality of [terms] is unit equality. *)
type t = { terms : (term * int * int) list; text : string }

let compare_term a b =
  match (a, b) with
  | Prime p, Prime q -> Int.compare p q
  | Prime _, _ -> -1
  | _, Prime _ -> 1
  | Pi, Pi -> 0
  | Pi, _ -> -1
  | _, Pi -> 1
  | Symbol a, Symbol b ->
      let c = String.compare a.name b.name in
      if c <> 0 then c else Option.compare String.compare a.scope b.scope

(* Text items *)

let plain_scope_byte = function
  | 'A' .. 'Z' | 'a' .. 'z' | '0' .. '9' -> true
  | '.' | '_' | '~' | ':' | '#' | '/' | '@' | '+' | '-' -> true
  | _ -> false

let encode_scope s =
  let b = Buffer.create (String.length s) in
  String.iter
    (fun c ->
      if plain_scope_byte c then Buffer.add_char b c
      else Buffer.add_string b (strf "%%%02X" (Char.code c)))
    s;
  Buffer.contents b

let base_text = function
  | Prime p -> string_of_int p
  | Pi -> "pi"
  | Symbol { name; scope = None } -> name
  | Symbol { name; scope = Some s } -> strf "%s{%s}" name (encode_scope s)

let power_text t num den =
  if num = 1 && den = 1 then base_text t
  else if den = 1 then strf "%s^%d" (base_text t) num
  else strf "%s^%d/%d" (base_text t) num den

(* The coefficient *)

(* Primes below [coefficient_limit] with an integer exponent form the
   coefficient. A canonical coefficient thus factors by trial division below the
   limit. *)
let coefficient_limit = 1 lsl 24
let coefficient_bits = 4096

exception Coefficient_past of string
exception Exponent_leaves_int of term

let same_prime p = function Prime q -> q = p | _ -> false

let in_coefficient = function
  | Prime p, _, 1 -> p < coefficient_limit
  | _ -> false

let natural ps =
  List.fold_left
    (fun acc (p, k) -> Nat.mul acc (Nat.pow (Nat.of_int p) k))
    Nat.one ps

(* [coefficient terms] is the coefficient's numerator and denominator. *)
let coefficient terms =
  let num = ref [] and den = ref [] in
  List.iter
    (fun ((t, k, _) as x) ->
      match t with
      | Prime p when in_coefficient x ->
          if k > 0 then num := (p, k) :: !num else den := (p, -k) :: !den
      | _ -> ())
    terms;
  (natural !num, natural !den)

(* [check_coefficient terms] raises if the coefficient's numerator or
   denominator is past [coefficient_bits], estimating its width before computing
   it. *)
let check_coefficient terms =
  let width sign =
    List.fold_left
      (fun acc ((t, k, _) as x) ->
        match t with
        | Prime p when in_coefficient x && k * sign > 0 ->
            acc +. (Float.abs (Float.of_int k) *. Float.log2 (Float.of_int p))
        | _ -> acc)
      0. terms
  in
  let limit = Float.of_int (coefficient_bits + 1) in
  if width 1 > limit then raise (Coefficient_past "numerator");
  if width (-1) > limit then raise (Coefficient_past "denominator");
  let num, den = coefficient terms in
  if Nat.bit_length num > coefficient_bits then
    raise (Coefficient_past "numerator");
  if Nat.bit_length den > coefficient_bits then
    raise (Coefficient_past "denominator");
  (num, den)

(* [coefficient_text terms (num, den)] writes Q = num/den as m 10^k with k = min
   (v2 Q) (v5 Q) when Q 10^-k is an integer m, otherwise as n/d. *)
let coefficient_text terms (num, den) =
  let v p =
    match List.find_opt (fun (t, _, _) -> same_prime p t) terms with
    | Some (_, k, 1) -> k
    | _ -> 0
  in
  let v2 = v 2 and v5 = v 5 in
  let k = min v2 v5 in
  let others_positive =
    List.for_all
      (fun ((t, e, _) as x) ->
        (not (in_coefficient x)) || same_prime 2 t || same_prime 5 t || e > 0)
      terms
  in
  if not others_positive then
    strf "%s/%s" (Nat.to_string num) (Nat.to_string den)
  else
    let m =
      let others =
        List.filter_map
          (fun ((t, e, _) as x) ->
            match t with
            | Prime p when in_coefficient x && p <> 2 && p <> 5 -> Some (p, e)
            | _ -> None)
          terms
      in
      natural ((2, v2 - k) :: (5, v5 - k) :: others)
    in
    if k = 0 then Nat.to_string m else strf "%se%d" (Nat.to_string m) k

let text_of terms coeff =
  let b = Buffer.create 32 in
  let item s =
    if Buffer.length b > 0 then Buffer.add_char b ' ';
    Buffer.add_string b s
  in
  let num, den = coeff in
  if not (Nat.equal num Nat.one && Nat.equal den Nat.one) then
    item (coefficient_text terms coeff);
  List.iter (function Pi, n, d -> item (power_text Pi n d) | _ -> ()) terms;
  List.iter
    (fun ((t, n, d) as x) ->
      match t with
      | Prime _ when not (in_coefficient x) -> item (power_text t n d)
      | _ -> ())
    terms;
  List.iter
    (function (Symbol _ as t), n, d -> item (power_text t n d) | _ -> ())
    terms;
  if Buffer.length b = 0 then "1" else Buffer.contents b

let make terms = { terms; text = text_of terms (check_coefficient terms) }

(* [unchecked_text terms] is the text of [terms] whatever its coefficient's
   width, for error messages about units the algebra has not built. *)
let unchecked_text terms = text_of terms (coefficient terms)

(* Exponents. Each is a reduced fraction of two ints with a positive denominator
   and a numerator other than [min_int], so negation is exact. *)

let rec gcd a b = if b = 0 then a else gcd b (a mod b)

let mul_exn t a b =
  let p = a * b in
  if (a <> 0 && p / a <> b) || p = min_int then raise (Exponent_leaves_int t)
  else p

let add_exn t a b =
  let s = a + b in
  if (a >= 0 = (b >= 0) && s >= 0 <> (a >= 0)) || s = min_int then
    raise (Exponent_leaves_int t)
  else s

(* [wide_sum t a x c y g] is [(s / g', g')] with [s = a x + c y], computed
   exactly, and [g' = gcd s g]. Raises if [s / g'] leaves int. *)
let wide_sum t a x c y g =
  let product a x = (a < 0, Nat.mul (Nat.of_int (abs a)) (Nat.of_int x)) in
  let negative, s =
    match (product a x, product c y) with
    | (n, m), (n', m') when n = n' -> (n, Nat.add m m')
    | (n, m), (_, m') when Nat.compare m m' >= 0 -> (n, Nat.sub m m')
    | (_, m), (n', m') -> (n', Nat.sub m' m)
  in
  let r = snd (Nat.div_rem s (Nat.of_int g)) in
  let g' = gcd g (Option.get (Nat.to_int r)) in
  match Nat.to_int (fst (Nat.div_rem s (Nat.of_int g'))) with
  | Some n -> ((if negative then -n else n), g')
  | None -> raise (Exponent_leaves_int t)

(* [q_add t (a, b) (c, d)] is a/b + c/d reduced, by Knuth's method: with [g =
   gcd b d], [s = a (d/g) + c (b/g)] and [g' = gcd s g], it is [(s/g') / ((b/g)
   (d/g'))], already reduced. [s] can leave int when the reduced sum does not,
   so it is then computed exactly. *)
let q_add t (a, b) (c, d) =
  let g = gcd b d in
  let b' = b / g and d' = d / g in
  let num, g' =
    match add_exn t (mul_exn t a d') (mul_exn t c b') with
    | s ->
        let g' = gcd (abs s) g in
        (s / g', g')
    | exception Exponent_leaves_int _ -> wide_sum t a d' c b' g
  in
  (num, mul_exn t b' (d / g'))

let rec merge xs ys =
  match (xs, ys) with
  | [], r | r, [] -> r
  | ((t, a, b) as x) :: xs', ((t', c, d) as y) :: ys' ->
      let o = compare_term t t' in
      if o < 0 then x :: merge xs' ys
      else if o > 0 then y :: merge xs ys'
      else
        let n, d = q_add t (a, b) (c, d) in
        if n = 0 then merge xs' ys' else (t, n, d) :: merge xs' ys'

(* [normalize ts] is the canonical form of the product of the terms [ts], which
   may repeat a term. *)
let normalize ts =
  let rec combine = function
    | (t, a, b) :: (t', c, d) :: rest when compare_term t t' = 0 ->
        let n, d = q_add t (a, b) (c, d) in
        combine ((t, n, d) :: rest)
    | (_, 0, _) :: rest -> combine rest
    | x :: rest -> x :: combine rest
    | [] -> []
  in
  combine (List.sort (fun (t, _, _) (t', _, _) -> compare_term t t') ts)

let guard fn f x =
  try f x with
  | Exponent_leaves_int t ->
      invalid_arg (strf "%s: the exponent of %s leaves int" fn (base_text t))
  | Coefficient_past part ->
      invalid_arg (strf "%s: the coefficient's %s is past 4096 bits" fn part)

(* Constructors *)

let one = { terms = []; text = "1" }
let of_factors ps = List.map (fun (p, k) -> (Prime p, k, 1)) ps

let int n =
  if n < 1 then invalid_arg (strf "Unit.int: %d is below 1" n);
  guard "Unit.int" make (of_factors (Prime.factor n))

let pi = make [ (Pi, 1, 1) ]

let is_name_start = function
  | 'A' .. 'Z' | 'a' .. 'z' | '_' -> true
  | _ -> false

let is_name_byte c =
  is_name_start c || match c with '0' .. '9' -> true | _ -> false

let valid_name name =
  name <> ""
  && is_name_start name.[0]
  && String.for_all is_name_byte name
  && name <> "pi"

let symbol name =
  if not (valid_name name) then
    invalid_arg (strf "Unit.symbol: %S is not a symbol name" name);
  make [ (Symbol { name; scope = None }, 1, 1) ]

let scoped ~scope name =
  if not (valid_name name) then
    invalid_arg (strf "Unit.scoped: %S is not a symbol name" name);
  if scope = "" then invalid_arg "Unit.scoped: the scope is empty";
  make [ (Symbol { name; scope = Some scope }, 1, 1) ]

(* Algebra *)

let mul u w = guard "Unit.( * )" (fun w -> make (merge u.terms w.terms)) w
let inverse ts = List.map (fun (t, n, d) -> (t, -n, d)) ts

let div u w =
  guard "Unit.( / )" (fun w -> make (merge u.terms (inverse w.terms))) w

let pow u n =
  let scale (t, a, b) =
    let g = gcd (abs n) b in
    (t, mul_exn t a (n / g), b / g)
  in
  if n = 0 then one
  else guard "Unit.( ** )" (fun u -> make (List.map scale u.terms)) u

let root n u =
  if n < 1 then invalid_arg (strf "Unit.root: %d is below 1" n);
  let scale (t, a, b) =
    let g = gcd (abs a) n in
    (t, a / g, mul_exn t b (n / g))
  in
  guard "Unit.root" (fun u -> make (List.map scale u.terms)) u

let decimal_named fn s =
  let decimal_error () =
    invalid_arg (strf "%s: %S is not a positive decimal" fn s)
  in
  let len = String.length s in
  let i = ref 0 in
  let peek c = !i < len && s.[!i] = c in
  (* [digits acc past] reads digits onto [acc] and calls [past] at the first
     digit that takes it to 2^62 or more. *)
  let digits acc past =
    let start = !i and acc = ref acc in
    while !i < len && s.[!i] >= '0' && s.[!i] <= '9' do
      let d = Char.code s.[!i] - Char.code '0' in
      if !acc > (max_int - d) / 10 then past ();
      acc := (!acc * 10) + d;
      incr i
    done;
    if !i = start then decimal_error ();
    !acc
  in
  let mantissa_past () =
    invalid_arg (strf "%s: %S has a mantissa of 2^62 or more" fn s)
  in
  let whole = digits 0 mantissa_past in
  let m, frac =
    if not (peek '.') then (whole, 0)
    else begin
      incr i;
      let start = !i in
      let m = digits whole mantissa_past in
      (m, !i - start)
    end
  in
  (* An exponent past int is past the coefficient's bound. *)
  let exponent () =
    if not (peek 'e') then -frac
    else begin
      incr i;
      let neg = peek '-' in
      if neg then incr i;
      let past () =
        raise (Coefficient_past (if neg then "denominator" else "numerator"))
      in
      let e = digits 0 past in
      match add_exn (Prime 2) (if neg then -e else e) (-frac) with
      | shift -> shift
      | exception Exponent_leaves_int _ -> past ()
    end
  in
  let shift = guard fn exponent () in
  if !i <> len || m = 0 then decimal_error ();
  let ten =
    if shift = 0 then [] else [ (Prime 2, shift, 1); (Prime 5, shift, 1) ]
  in
  let factors = of_factors (Prime.factor m) in
  guard fn (fun ten -> make (merge factors ten)) ten

let decimal s = decimal_named "Unit.decimal" s

(* Conversion *)

let symbols ts = List.filter (function Symbol _, _, _ -> true | _ -> false) ts

let equal_terms ts ts' =
  List.equal
    (fun (t, a, b) (t', c, d) -> compare_term t t' = 0 && a = c && b = d)
    ts ts'

let convertible u w = equal_terms (symbols u.terms) (symbols w.terms)

let exact_factor ts =
  List.filter_map
    (function
      | Prime p, n, d -> Some (Exact.Prime p, n, d)
      | Pi, n, d -> Some (Exact.Pi, n, d)
      | Symbol _, _, _ -> None)
    ts

(* [ratio_error fn d u w q e] is the message of [fn] when the conversion from
   [u] to [w], of exact quotient [q], has no value in [d] for the reason [e]. *)
let ratio_error (type a b) fn (d : (a, b) Nx.dtype) u w q (e : Exact.error) =
  let dt = Nx_dtype.to_string d in
  let factor which =
    strf "%s: the factor from %s to %s is %s, %s" fn u.text w.text
      (unchecked_text q) which
  in
  match e with
  | Zero -> factor (strf "which is 0 in %s" dt)
  | Subnormal -> factor (strf "which is subnormal in %s" dt)
  | Overflow -> factor (strf "which overflows %s" dt)
  | Not_integer -> factor "which is not an integer"
  | Out_of_range -> factor (strf "which %s does not hold" dt)
  | Too_wide ->
      factor
        (strf "whose evaluation needs a natural wider than %d bits" Exact.budget)
  | Boolean -> strf "%s: %s holds no factor" fn dt

let ratio_named (type a b) fn (d : (a, b) Nx.dtype) u w : a =
  let quotient ts ts' = guard fn (merge ts) (inverse ts') in
  if not (convertible u w) then
    invalid_arg
      (strf "%s: %s does not convert to %s: their quotient keeps %s" fn u.text
         w.text
         (unchecked_text (quotient (symbols u.terms) (symbols w.terms))));
  let q = quotient u.terms w.terms in
  match Exact.round d (exact_factor q) with
  | Ok v -> v
  | Error e -> invalid_arg (ratio_error fn d u w q e)

let round d u = Exact.round d (exact_factor u.terms)
let ratio d u w = ratio_named "Unit.ratio" d u w

(* Terms and text *)

let terms u = u.terms
let to_string u = u.text
let pp ppf u = Format.pp_print_string ppf u.text
let equal u w = String.equal u.text w.text
let compare u w = String.compare u.text w.text

(* Parsing canonical text. [parse] reads the grammar and returns the terms the
   items denote, unnormalised; [of_string] then requires the canonical text of
   their product to be [s]. *)

exception Syntax of string

let hex_value = function
  | '0' .. '9' as c -> Char.code c - Char.code '0'
  | 'A' .. 'F' as c -> Char.code c - Char.code 'A' + 10
  | 'a' .. 'f' as c -> Char.code c - Char.code 'a' + 10
  | _ -> -1

let coefficient_digits = 1234 (* 10^1233 < 2^4096 < 10^1234 *)

let parse s =
  let len = String.length s in
  let i = ref 0 in
  let fail what = raise (Syntax (strf "%s at byte %d" what !i)) in
  let peek c = !i < len && s.[!i] = c in
  let eat c =
    peek c
    &&
    (incr i;
     true)
  in
  let digits () =
    let start = !i in
    while !i < len && s.[!i] >= '0' && s.[!i] <= '9' do
      incr i
    done;
    if !i = start then fail "expected digits";
    String.sub s start (!i - start)
  in
  (* An integer below 2^62, checked digit by digit. *)
  let integer () =
    let start = !i in
    let ds = digits () in
    let v =
      String.fold_left
        (fun acc c ->
          let d = Char.code c - Char.code '0' in
          if acc > (max_int - d) / 10 then begin
            i := start;
            fail "an integer of 2^62 or more"
          end;
          (acc * 10) + d)
        0 ds
    in
    v
  in
  let coefficient_nat () =
    let start = !i in
    let ds = digits () in
    if String.length ds > coefficient_digits then (
      i := start;
      fail "a coefficient past 4096 bits");
    let n = Nat.of_digits ds in
    if Nat.bit_length n > coefficient_bits then (
      i := start;
      fail "a coefficient past 4096 bits");
    if Nat.is_zero n then (
      i := start;
      fail "a zero coefficient");
    n
  in
  let factors_of n =
    match Prime.factor_smooth n with
    | Some f -> of_factors f
    | None -> fail "a coefficient with a prime factor of 2^24 or more"
  in
  let exponent () =
    if not (eat '^') then (1, 1)
    else
      let neg = eat '-' in
      let n = integer () in
      let d = if eat '/' then integer () else 1 in
      if d = 0 then fail "a zero denominator";
      let g = gcd n d in
      ((if neg then -n / g else n / g), d / g)
  in
  (* (a/b) (n/d) reduced crosswise, so a product leaves int only when the
     reduced exponent does. *)
  let raise_to (n, d) ts =
    List.map
      (fun (t, a, b) ->
        let g = gcd (abs a) d and g' = gcd (abs n) b in
        (t, mul_exn t (a / g) (n / g'), mul_exn t (b / g') (d / g)))
      ts
  in
  let item () =
    if !i >= len then fail "expected an item";
    match s.[!i] with
    | '0' .. '9' ->
        let start = !i in
        let n = coefficient_nat () in
        if eat 'e' then begin
          let neg = eat '-' in
          let k = integer () in
          if k > coefficient_bits then fail "a coefficient past 4096 bits";
          let k = if neg then -k else k in
          (Prime 2, k, 1) :: (Prime 5, k, 1) :: factors_of n
        end
        else if eat '/' then begin
          let d = coefficient_nat () in
          factors_of n @ inverse (factors_of d)
        end
        else if peek '^' then begin
          if Nat.to_int n = None then begin
            i := start;
            fail "an integer of 2^62 or more"
          end;
          raise_to (exponent ()) (factors_of n)
        end
        else factors_of n
    | c when is_name_start c ->
        let start = !i in
        while !i < len && is_name_byte s.[!i] do
          incr i
        done;
        let name = String.sub s start (!i - start) in
        if name = "pi" then raise_to (exponent ()) [ (Pi, 1, 1) ]
        else begin
          let scope =
            if not (eat '{') then None
            else begin
              let b = Buffer.create 16 in
              while not (peek '}') do
                if !i >= len then fail "expected '}'";
                if eat '%' then begin
                  let digit k =
                    if !i + k < len then hex_value s.[!i + k] else -1
                  in
                  let hi = digit 0 and lo = digit 1 in
                  if hi < 0 || lo < 0 then
                    fail "expected two hexadecimal digits";
                  Buffer.add_char b (Char.chr ((hi * 16) + lo));
                  i := !i + 2
                end
                else begin
                  Buffer.add_char b s.[!i];
                  incr i
                end
              done;
              incr i;
              if Buffer.length b = 0 then fail "an empty scope";
              Some (Buffer.contents b)
            end
          in
          let t = Symbol { name; scope } in
          raise_to (exponent ()) [ (t, 1, 1) ]
        end
    | _ -> fail "expected an item"
  in
  let rec items acc =
    let acc = List.rev_append (item ()) acc in
    if !i = len then acc
    else if eat ' ' then items acc
    else fail "expected a space"
  in
  items []

let of_string s =
  match normalize (parse s) with
  | exception Syntax why ->
      Error (strf "%S is not a unit's canonical text: %s" s why)
  | exception Exponent_leaves_int t ->
      Error
        (strf "%S is not a unit's canonical text: the exponent of %s leaves int"
           s (base_text t))
  | ts -> (
      match make ts with
      | exception Coefficient_past part ->
          Error
            (strf
               "%S is not a unit's canonical text: the coefficient's %s is \
                past 4096 bits"
               s part)
      | u when String.equal u.text s -> Ok u
      | u ->
          Error
            (strf "%S is not canonical: the unit's canonical text is %S" s
               u.text))

(* Structure *)

module Walked = struct
  type nonrec _ t = t

  let walk c u =
    Nx.Ptree.Walk.case c u.text;
    u
end

let ptree : t Nx.Ptree.t = Nx.Ptree.instantiate (module Walked)

(* The SI. The operators shadow integer arithmetic from here on. *)

let ( * ) = mul
let ( / ) = div
let ( ** ) = pow
let ten_to k = make [ (Prime 2, k, 1); (Prime 5, k, 1) ]

(* Base units *)

let metre = symbol "m"
let kilogram = symbol "kg"
let second = symbol "s"
let ampere = symbol "A"
let kelvin = symbol "K"
let mole = symbol "mol"
let candela = symbol "cd"
let radian = symbol "rad"

(* Derived units *)

let steradian = radian ** 2
let hertz = second ** -1
let newton = kilogram * metre / (second ** 2)
let pascal = newton / (metre ** 2)
let joule = newton * metre
let watt = joule / second
let coulomb = ampere * second
let volt = watt / ampere
let farad = coulomb / volt
let ohm = volt / ampere
let siemens = ampere / volt
let weber = volt * second
let tesla = weber / (metre ** 2)
let henry = weber / ampere
let lumen = candela * steradian
let lux = lumen / (metre ** 2)
let becquerel = hertz
let gray = joule / kilogram
let sievert = gray
let katal = mole / second

(* Prefixes *)

let prefix k u = ten_to k * u
let quecto = prefix (-30)
let ronto = prefix (-27)
let yocto = prefix (-24)
let zepto = prefix (-21)
let atto = prefix (-18)
let femto = prefix (-15)
let pico = prefix (-12)
let nano = prefix (-9)
let micro = prefix (-6)
let milli = prefix (-3)
let centi = prefix (-2)
let deci = prefix (-1)
let deca = prefix 1
let hecto = prefix 2
let kilo = prefix 3
let mega = prefix 6
let giga = prefix 9
let tera = prefix 12
let peta = prefix 15
let exa = prefix 18
let zetta = prefix 21
let yotta = prefix 24
let ronna = prefix 27
let quetta = prefix 30

(* Exact constants *)

let speed_of_light = int 299792458 * metre / second
let planck = decimal "6.62607015e-34" * joule * second
let hbar = planck / (int 2 * pi * radian)
let elementary_charge = decimal "1.602176634e-19" * coulomb
let boltzmann = decimal "1.380649e-23" * joule / kelvin
let avogadro = decimal "6.02214076e23" / mole
let caesium_frequency = int 9192631770 * hertz
let luminous_efficacy = int 683 * lumen / watt
let gas_constant = avogadro * boltzmann
let faraday = avogadro * elementary_charge

let stefan_boltzmann =
  int 2 * (pi ** 5) * (boltzmann ** 4)
  / (int 15 * (planck ** 3) * (speed_of_light ** 2))

(* Units accepted for use with the SI *)

let gram = milli kilogram
let minute = int 60 * second
let hour = int 60 * minute
let day = int 24 * hour
let litre = milli (metre ** 3)
let degree = pi / int 180 * radian
let arcminute = degree / int 60
let arcsecond = arcminute / int 60
let electronvolt = elementary_charge * volt
let tonne = int 1000 * kilogram
let hectare = int 10_000 * (metre ** 2)
let astronomical_unit = int 149597870700 * metre
