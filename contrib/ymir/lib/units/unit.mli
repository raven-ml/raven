(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Exact physical units.

    A {e symbol} is a named dimension: [m], [kg], [s], [A], [K], [mol], [cd],
    [rad], or any a program declares ([electron], [adu]). A symbol may carry a
    {e scope}, a string naming the data set it belongs to. A {e term} is a
    prime, π or a symbol. A {e unit} is a product of terms raised to nonzero
    rational exponents. Its terms without symbols form its {e number}, a
    positive real; a unit without symbols is {e dimensionless}. Two units
    {e convert} when their quotient is dimensionless, and the conversion from
    [u] to [w] is the number of [u / w].

    Every function on units is exact, and a unit has one form: [kilo metre] and
    [int 1000 * metre] are one unit, with one canonical text ({!section-text})
    that is its identity in equality, in the keys of compiled programs, in table
    metadata and in files.

    Build units from {!section-si}'s units and constants, {!int}, {!decimal},
    {!pi}, {!symbol} and the {!section-algebra}; leave a unit for a number with
    {!ratio}.

    {[
    let speed = Unit.(kilo metre / second) (* 1e3 m s^-1 *)
    let f = Unit.ratio Nx.float32 speed Unit.(metre / second) (* 1000. *)
    ]} *)

type t
(** The type for units. *)

(** {1:constructors Constructors} *)

val one : t
(** [one] is the dimensionless unit 1. *)

val int : int -> t
(** [int n] is the number [n], factored into primes exactly. Factoring is
    deterministic: the same [n] gives the same unit. Raises [Invalid_argument]
    if [n < 1]. *)

val decimal : string -> t
(** [decimal s] is the exact positive decimal [s], as in ["6.62607015e-34"]:
    [digits ["." digits] ["e" ["-"] digits]], with no sign, no [+] and no [E].
    Raises [Invalid_argument] on a string outside this grammar, a zero value, or
    a mantissa, its digits read as one integer, of 2{^ 62} or more. *)

val pi : t
(** [pi] is the number π. *)

val symbol : string -> t
(** [symbol name] is the symbol [name]. Raises [Invalid_argument] unless [name]
    matches [[A-Za-z_][A-Za-z0-9_]*] and is not ["pi"]. *)

val scoped : scope:string -> string -> t
(** [scoped ~scope name] is the symbol [name] of the data set [scope]. It equals
    only a symbol with the same name and scope, so it converts only to itself.
    Raises [Invalid_argument] as {!symbol} does, or if [scope] is empty. *)

(** {1:algebra Algebra}

    The units form an abelian group with rational exponents, and its identities
    hold exactly: [(u * w) / w] is [u] and [root n (u ** n)] is [u]. Each
    operation raises [Invalid_argument] when the numerator or denominator of a
    reduced exponent leaves [int], as in
    ["Unit.( ** ): the exponent of m leaves int"], or when the numerator or
    denominator of the coefficient ({!section-text}), in lowest terms, is past
    4096 bits, as in
    ["Unit.( * ): the coefficient's numerator is past 4096 bits"]. An exponent
    is a reduced fraction of two [int]s, [min_int] excluded; intermediate values
    never make an operation raise. *)

val ( * ) : t -> t -> t
(** [u * w] is the product of [u] and [w]. *)

val ( / ) : t -> t -> t
(** [u / w] is the quotient of [u] by [w]. *)

val ( ** ) : t -> int -> t
(** [u ** n] is [u] to the power [n]. [u ** 0] is {!one}. *)

val root : int -> t -> t
(** [root n u] is [u] to the power [1/n]. Raises [Invalid_argument] if [n < 1].
*)

(** {1:conversion Conversion} *)

val convertible : t -> t -> bool
(** [convertible u w] is [true] iff [u / w] is dimensionless. *)

val ratio : ('a, 'b) Nx.dtype -> t -> t -> 'a
(** [ratio d u w] is the conversion from [u] to [w], the exact number [v] of
    [u / w], rounded once to [d]:
    - A float dtype rounds [v] as IEEE 754 roundTiesToEven does in [d], with
      [d]'s subnormals and an exponent unbounded above. A complex dtype rounds
      as its component does and has a zero imaginary part. The result is exact
      in float64.
    - An integer dtype holds [v] only if [v] is an integer in its range.
      Unsigned 32-bit and 64-bit values are their bit patterns.

    The result is the same on every platform.

    Raises [Invalid_argument] with:
    - ["Unit.ratio: electron s^-1 does not convert to kg s^-1: their quotient
       keeps electron kg^-1"] if the units do not convert;
    - ["Unit.ratio: the exponent of pi leaves int"] if an exponent of [u / w]
      leaves [int];
    - ["Unit.ratio: the factor from 1e-35 kg s^-2 to kg s^-2 is 1e-35, which is
       0 in float16"] if [v] rounds to 0, and with
      [which is subnormal in float16] if it rounds to a subnormal: a kernel on a
      device that flushes subnormals would compute with 0;
    - [which overflows float16] if [v] is above the largest finite value plus
      half its ulp, or equal to it when that value's last bit is odd: a format
      without infinity (float8_e4m3) raises where it would saturate;
    - [which is not an integer] or [which int8 does not hold] for an integer
      dtype;
    - [whose evaluation needs a natural wider than 65536 bits] when [v]'s exact
      evaluation does;
    - ["Unit.ratio: bool holds no factor"] for [Nx.bool]. *)

(** {1:terms Terms} *)

(** The type for terms. *)
type term =
  | Prime of int  (** A prime number. *)
  | Pi  (** π. *)
  | Symbol of { name : string; scope : string option }
      (** A symbol and its scope, if any. *)

val terms : t -> (term * int * int) list
(** [terms u] is [u]'s terms, each [(t, num, den)] with [t]'s exponent [num/den]
    reduced, [den >= 1] and [num <> 0]. They are in canonical order: primes
    ascending, then π, then symbols in byte order of name, then of scope, a
    symbol without scope first. A scope is ordered by its bytes as given, before
    the canonical text encodes them. [terms one] is [[]]. *)

(** {1:text Canonical text}

    A unit's canonical text follows this grammar, with items separated by single
    spaces:

    {v
    unit   = "1" | item { " " item }
    item   = coeff | power
    coeff  = digits [ "e" [ "-" ] digits ] | digits "/" digits
    power  = base [ "^" [ "-" ] digits [ "/" digits ] ]
    base   = "pi" | digits | name [ "{" scope "}" ]
    v}

    The items are, in order:
    + The coefficient Q: the product of the primes below 2{^ 24} with an integer
      exponent. With [k = min (v2 Q) (v5 Q)], [Q * 10{^-k}] written [m], or
      [mek] when [k <> 0], if it is an integer [m], and [n/d] in lowest terms
      otherwise. It is omitted when Q is 1.
    + [pi^r] when π's exponent [r] is not 0.
    + Each prime whose exponent is not an integer or that is at least 2{^ 24},
      ascending.
    + Each symbol, in the order of {!terms}. A scope is written in braces, with
      every byte outside [[A-Za-z0-9._~:#/@+-]] written [%XX] in upper-case
      hexadecimal.

    A power whose exponent is 1 is its base; otherwise it is [base^n], or
    [base^n/d] with [d > 1], the sign on [n]. The whole unit is ["1"] when no
    item is written.

    {v
    Unit.(kilo metre / second)     1e3 m s^-1
    Unit.degree                    1/180 pi rad
    Unit.hour                      36e2 s
    Unit.(root 2 hertz)            s^-1/2
    Unit.planck                    662607015e-42 kg m^2 s^-1
    v}

    The text is injective: two units are equal iff their texts are. It is a
    stable format, since files and tables store units by their text. *)

val to_string : t -> string
(** [to_string u] is [u]'s canonical text. *)

val of_string : string -> (t, string) result
(** [of_string s] is [Ok u] iff [to_string u = s], and [Error msg] otherwise.
    When [s]'s items denote a unit, [msg] names that unit's canonical text, as
    in [{|"1000 m" is not canonical: the unit's canonical text is "1e3 m"|}];
    otherwise it names the reason, as in
    [{|"m^" is not a unit's canonical text: expected digits at byte 2|}].

    Before any arithmetic it rejects a coefficient whose digits, numerator or
    denominator are past 4096 bits, a power of ten [e<k>] with [|k| > 4096], and
    an exponent or a base of 2{^ 62} or more, so its time is bounded by [s]'s
    length. A coefficient whose prime factors below 2{^ 24} leave a cofactor of
    2{^ 62} or more is not canonical; it is rejected without naming a canonical
    text. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf u] formats [u]'s canonical text. *)

(** {1:predicates Predicates and comparisons} *)

val equal : t -> t -> bool
(** [equal u w] is [true] iff [u] and [w] are the same unit, which is iff
    [to_string u = to_string w]. *)

val compare : t -> t -> int
(** [compare u w] totally orders units, compatibly with {!equal}: it is the byte
    order of their canonical texts. *)

(** {1:structure Structure} *)

val ptree : t Nx.Ptree.t
(** [ptree] is the structure of a unit: one [Walk.case] holding its canonical
    text, and no tensor. A compiled function that takes a unit compiles once per
    unit. *)

(** {1:si The SI}

    Units and constants are named by the SI Brochure's English name, in lower
    case with underscores. The radian is a symbol, so an angle is a dimension:
    [steradian] is [radian ** 2], [degree] is π/180 rad and [hertz] is
    [second ** -1], which is not rad s{^ -1}. *)

(** {2:base Base units} *)

val metre : t
(** [metre] is the symbol [m]. *)

val kilogram : t
(** [kilogram] is the symbol [kg]. *)

val second : t
(** [second] is the symbol [s]. *)

val ampere : t
(** [ampere] is the symbol [A]. *)

val kelvin : t
(** [kelvin] is the symbol [K]. *)

val mole : t
(** [mole] is the symbol [mol]. *)

val candela : t
(** [candela] is the symbol [cd]. *)

val radian : t
(** [radian] is the symbol [rad]. *)

(** {2:derived Derived units} *)

val steradian : t
(** [steradian] is rad{^ 2}. *)

val hertz : t
(** [hertz] is s{^ -1}. *)

val newton : t
(** [newton] is kg m s{^ -2}. *)

val pascal : t
(** [pascal] is kg m{^ -1} s{^ -2}. *)

val joule : t
(** [joule] is kg m{^ 2} s{^ -2}. *)

val watt : t
(** [watt] is kg m{^ 2} s{^ -3}. *)

val coulomb : t
(** [coulomb] is A s. *)

val volt : t
(** [volt] is kg m{^ 2} s{^ -3} A{^ -1}. *)

val farad : t
(** [farad] is A{^ 2} s{^ 4} kg{^ -1} m{^ -2}. *)

val ohm : t
(** [ohm] is kg m{^ 2} s{^ -3} A{^ -2}. *)

val siemens : t
(** [siemens] is A{^ 2} s{^ 3} kg{^ -1} m{^ -2}. *)

val weber : t
(** [weber] is kg m{^ 2} s{^ -2} A{^ -1}. *)

val tesla : t
(** [tesla] is kg s{^ -2} A{^ -1}. *)

val henry : t
(** [henry] is kg m{^ 2} s{^ -2} A{^ -2}. *)

val lumen : t
(** [lumen] is cd sr. *)

val lux : t
(** [lux] is cd sr m{^ -2}. *)

val becquerel : t
(** [becquerel] is s{^ -1}, which is {!hertz}. *)

val gray : t
(** [gray] is m{^ 2} s{^ -2}. *)

val sievert : t
(** [sievert] is m{^ 2} s{^ -2}, which is {!gray}. *)

val katal : t
(** [katal] is mol s{^ -1}. *)

(** {2:accepted Units accepted for use with the SI}

    The units the SI accepts whose value is exact and a multiple of an SI unit.
    The dalton is measured, and the neper, bel and decibel are logarithmic, so
    none of them is a unit here. *)

val gram : t
(** [gram] is 10{^ -3} kg. *)

val tonne : t
(** [tonne] is 10{^ 3} kg. *)

val minute : t
(** [minute] is 60 s. *)

val hour : t
(** [hour] is 3600 s. *)

val day : t
(** [day] is 86400 s. *)

val litre : t
(** [litre] is 10{^ -3} m{^ 3}. *)

val hectare : t
(** [hectare] is 10{^ 4} m{^ 2}. *)

val astronomical_unit : t
(** [astronomical_unit] is 149597870700 m. *)

val degree : t
(** [degree] is π/180 rad. *)

val arcminute : t
(** [arcminute] is π/10800 rad. *)

val arcsecond : t
(** [arcsecond] is π/648000 rad. *)

val electronvolt : t
(** [electronvolt] is {!elementary_charge} times {!volt}: 1.602176634·10{^ -19}
    J. *)

(** {2:prefixes Prefixes}

    A prefix multiplies by a power of ten: [kilo metre] is [int 1000 * metre].
*)

val quecto : t -> t
(** [quecto u] is 10{^ -30} [u]. *)

val ronto : t -> t
(** [ronto u] is 10{^ -27} [u]. *)

val yocto : t -> t
(** [yocto u] is 10{^ -24} [u]. *)

val zepto : t -> t
(** [zepto u] is 10{^ -21} [u]. *)

val atto : t -> t
(** [atto u] is 10{^ -18} [u]. *)

val femto : t -> t
(** [femto u] is 10{^ -15} [u]. *)

val pico : t -> t
(** [pico u] is 10{^ -12} [u]. *)

val nano : t -> t
(** [nano u] is 10{^ -9} [u]. *)

val micro : t -> t
(** [micro u] is 10{^ -6} [u]. *)

val milli : t -> t
(** [milli u] is 10{^ -3} [u]. *)

val centi : t -> t
(** [centi u] is 10{^ -2} [u]. *)

val deci : t -> t
(** [deci u] is 10{^ -1} [u]. *)

val deca : t -> t
(** [deca u] is 10 [u]. *)

val hecto : t -> t
(** [hecto u] is 10{^ 2} [u]. *)

val kilo : t -> t
(** [kilo u] is 10{^ 3} [u]. *)

val mega : t -> t
(** [mega u] is 10{^ 6} [u]. *)

val giga : t -> t
(** [giga u] is 10{^ 9} [u]. *)

val tera : t -> t
(** [tera u] is 10{^ 12} [u]. *)

val peta : t -> t
(** [peta u] is 10{^ 15} [u]. *)

val exa : t -> t
(** [exa u] is 10{^ 18} [u]. *)

val zetta : t -> t
(** [zetta u] is 10{^ 21} [u]. *)

val yotta : t -> t
(** [yotta u] is 10{^ 24} [u]. *)

val ronna : t -> t
(** [ronna u] is 10{^ 27} [u]. *)

val quetta : t -> t
(** [quetta u] is 10{^ 30} [u]. *)

(** {2:constants Exact constants}

    The SI fixes these constants exactly, so they are units and stay exact
    through a formula until a conversion rounds it once. They belong to no
    CODATA release. *)

val speed_of_light : t
(** [speed_of_light] is c, 299792458 m s{^ -1}. *)

val planck : t
(** [planck] is h, 6.62607015·10{^ -34} kg m{^ 2} s{^ -1}. *)

val hbar : t
(** [hbar] is ℏ, [planck / (int 2 * pi * radian)]: the action per radian. ℏω
    with ω in rad s{^ -1} is an energy; ℏν with ν in Hz does not convert to one.
*)

val elementary_charge : t
(** [elementary_charge] is e, 1.602176634·10{^ -19} A s. *)

val boltzmann : t
(** [boltzmann] is k, 1.380649·10{^ -23} kg m{^ 2} s{^ -2} K{^ -1}. *)

val avogadro : t
(** [avogadro] is N{_ A}, 6.02214076·10{^ 23} mol{^ -1}. *)

val caesium_frequency : t
(** [caesium_frequency] is Δν{_ Cs}, the caesium 133 hyperfine transition
    frequency, 9192631770 s{^ -1}. *)

val luminous_efficacy : t
(** [luminous_efficacy] is K{_ cd}, 683 cd sr W{^ -1}. *)

val gas_constant : t
(** [gas_constant] is R, [avogadro * boltzmann]. *)

val faraday : t
(** [faraday] is F, [avogadro * elementary_charge]. *)

val stefan_boltzmann : t
(** [stefan_boltzmann] is σ, 2π{^ 5}k{^ 4}/(15h{^ 3}c{^ 2}). *)
