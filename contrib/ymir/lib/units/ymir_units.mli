(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Exact physical units and quantities.

    {!Unit} is a unit as an exact value: a product of primes, π and named
    symbols with rational exponents, with one canonical text. A conversion
    between two units is one correctly rounded multiply, or it raises.
    {!Quantity} is a value in a unit, a structure that rune's transformations
    carry with no rule of their own.

    The SI fixes some constants exactly, so they are units ({!Unit.planck}). A
    measured constant is a {!Constant} of a {!Codata} release the program names.
*)

(** Units as exact values. *)
module Unit : sig
  (** A {e symbol} is a named dimension: [m], [kg], [s], [A], [K], [mol], [cd],
      [rad], or any a program declares ([electron], [adu]). A symbol may carry a
      {e scope}, a string naming the data set it belongs to. A {e term} is a
      prime, π or a symbol. A {e unit} is a product of terms raised to nonzero
      rational exponents. Its terms without symbols form its {e number}, a
      positive real; a unit without symbols is {e dimensionless}. Two units
      {e convert} when their quotient is dimensionless, and the conversion from
      [u] to [w] is the number of [u / w].

      Every function on units is exact, and a unit has one form: [kilo metre]
      and [int 1000 * metre] are one unit, with one canonical text
      ({!section-text}) that is its identity in equality, in the keys of
      compiled programs, in table metadata and in files.

      Build units from {!section-si}'s units and constants, {!int}, {!decimal},
      {!pi}, {!symbol} and the {!section-algebra}; leave a unit for a number
      with {!ratio}.

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
      Raises [Invalid_argument] on a string outside this grammar, a zero value,
      or a mantissa, its digits read as one integer, of 2{^ 62} or more. *)

  val pi : t
  (** [pi] is the number π. *)

  val symbol : string -> t
  (** [symbol name] is the symbol [name]. Raises [Invalid_argument] unless
      [name] matches [[A-Za-z_][A-Za-z0-9_]*] and is not ["pi"]. *)

  val scoped : scope:string -> string -> t
  (** [scoped ~scope name] is the symbol [name] of the data set [scope]. It
      equals only a symbol with the same name and scope, so it converts only to
      itself. Raises [Invalid_argument] as {!symbol} does, or if [scope] is
      empty. *)

  (** {1:algebra Algebra}

      The units form an abelian group with rational exponents, and its
      identities hold exactly: [(u * w) / w] is [u] and [root n (u ** n)] is
      [u]. Each operation raises [Invalid_argument] when the numerator or
      denominator of a reduced exponent leaves [int], as in
      ["Unit.( ** ): the exponent of m leaves int"], or when the numerator or
      denominator of the coefficient ({!section-text}), in lowest terms, is past
      4096 bits, as in
      ["Unit.( * ): the coefficient's numerator is past 4096 bits"]. An exponent
      is a reduced fraction of two [int]s, [min_int] excluded; intermediate
      values never make an operation raise. *)

  val ( * ) : t -> t -> t
  (** [u * w] is the product of [u] and [w]. *)

  val ( / ) : t -> t -> t
  (** [u / w] is the quotient of [u] by [w]. *)

  val ( ** ) : t -> int -> t
  (** [u ** n] is [u] to the power [n]. [u ** 0] is {!one}. *)

  val root : int -> t -> t
  (** [root n u] is [u] to the power [1/n]. Raises [Invalid_argument] if
      [n < 1]. *)

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
      - ["Unit.ratio: the factor from 1e-35 kg s^-2 to kg s^-2 is 1e-35, which
         is 0 in float16"] if [v] rounds to 0, and with
        [which is subnormal in float16] if it rounds to a subnormal: a kernel on
        a device that flushes subnormals would compute with 0;
      - [which overflows float16] if [v] is above the largest finite value plus
        half its ulp, or equal to it when that value's last bit is odd: a format
        without infinity (float8_e4m3) raises where it would saturate;
      - [which is not an integer] or [which int8 does not hold] for an integer
        dtype;
      - [whose evaluation needs a natural wider than 65536 bits] when [v]'s
        exact evaluation does;
      - ["Unit.ratio: bool holds no factor"] for [Nx.bool], and with [bit] for
        [Nx.bit].

      Each message names [d] as {!Nx_dtype.to_string} writes it. *)

  (** {1:terms Terms} *)

  (** The type for terms. *)
  type term =
    | Prime of int  (** A prime number. *)
    | Pi  (** π. *)
    | Symbol of { name : string; scope : string option }
        (** A symbol and its scope, if any. *)

  val terms : t -> (term * int * int) list
  (** [terms u] is [u]'s terms, each [(t, num, den)] with [t]'s exponent
      [num/den] reduced, [den >= 1] and [num <> 0]. They are in canonical order:
      primes ascending, then π, then symbols in byte order of name, then of
      scope, a symbol without scope first. A scope is ordered by its bytes as
      given, before the canonical text encodes them. [terms one] is [[]]. *)

  (** {1:text Canonical text}

      A unit's canonical text follows this grammar, with items separated by
      single spaces:

      {v
      unit   = "1" | item { " " item }
      item   = coeff | power
      coeff  = digits [ "e" [ "-" ] digits ] | digits "/" digits
      power  = base [ "^" [ "-" ] digits [ "/" digits ] ]
      base   = "pi" | digits | name [ "{" scope "}" ]
      v}

      The items are, in order:
      + The coefficient Q: the product of the primes below 2{^ 24} with an
        integer exponent. With [k = min (v2 Q) (v5 Q)], [Q * 10{^-k}] written
        [m], or [mek] when [k <> 0], if it is an integer [m], and [n/d] in
        lowest terms otherwise. It is omitted when Q is 1.
      + [pi^r] when π's exponent [r] is not 0.
      + Each prime whose exponent is not an integer or that is at least 2{^ 24},
        ascending.
      + Each symbol, in the order of {!terms}. A scope is written in braces,
        with every byte outside [[A-Za-z0-9._~:#/@+-]] written [%XX] in
        upper-case hexadecimal.

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
      denominator are past 4096 bits, a power of ten [e<k>] with [|k| > 4096],
      and an exponent or a base of 2{^ 62} or more, so its time is bounded by
      [s]'s length. A coefficient whose prime factors below 2{^ 24} leave a
      cofactor of 2{^ 62} or more is not canonical; it is rejected without
      naming a canonical text. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf u] formats [u]'s canonical text. *)

  (** {1:predicates Predicates and comparisons} *)

  val equal : t -> t -> bool
  (** [equal u w] is [true] iff [u] and [w] are the same unit, which is iff
      [to_string u = to_string w]. *)

  val compare : t -> t -> int
  (** [compare u w] totally orders units, compatibly with {!equal}: it is the
      byte order of their canonical texts. *)

  (** {1:structure Structure} *)

  val ptree : t Nx.Ptree.t
  (** [ptree] is the structure of a unit: one [Walk.case] holding its canonical
      text, and no tensor. A compiled function that takes a unit compiles once
      per unit. *)

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

      The units the SI accepts whose value is exact and a multiple of an SI
      unit. The dalton is measured, and the neper, bel and decibel are
      logarithmic, so none of them is a unit here. *)

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
  (** [electronvolt] is {!elementary_charge} times {!volt}:
      1.602176634·10{^ -19} J. *)

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
      with ω in rad s{^ -1} is an energy; ℏν with ν in Hz does not convert to
      one. *)

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
end

(** Values in a unit. *)
module Quantity : sig
  (** A {e quantity} is a value, its {e payload}, in a {!Unit.t}. Its operations
      are a quantity's algebra. Within one dimension quantities form a vector
      space: {!add} and {!sub} convert the second argument to the first's unit,
      and scaling is {!map}. Across dimensions they form a group: {!mul},
      {!div}, {!pow} and {!root} compute the unit. {!times} and {!per} change
      the unit and keep the payload. These six raise [Invalid_argument] as the
      unit algebra does ({!Unit.section-algebra}). A number leaves a quantity
      through a named unit, with {!value}:

      {[
      let v = Quantity.v Unit.(kilo metre / second) speeds
      let v_si = Quantity.value Unit.(metre / second) v (* times 1000 *)

      let below =
        Nx.less (Quantity.value Unit.metre a) (Quantity.value Unit.metre b)
      ]}

      [Quantity] is an {!Nx.Ptree.S}: a quantity reports its unit as one
      [Walk.case] holding the unit's canonical text, then walks its payload,
      both at the quantity's path. [jit], [vmap], [scan] and [jvp] carry units
      with no rule of their own, and a compiled function compiles once per unit.
      A [scan] carry keeps the initial carry's unit: a step that computes in
      another unit returns [convert (unit init) q]. [grad] and [vjp] take and
      return bare tensors, with the parameters' units kept as a value of the
      same structure: a gradient is the loss per parameter unit, so a quantity
      given to them would come back labelled with the parameter's unit, and
      {!value} would then convert it by the wrong factor without a word. A
      record of quantities walks each field with {!walk}; {!Nx.Ptree.cast} and
      {!Nx.Ptree.Payload} then act on the payloads and keep the units.

      {b Dtypes.} A float payload converts by one rounded factor. A complex
      payload scales its real and imaginary parts by the real factor, so an
      infinite part never meets a zero imaginary factor. An integer payload
      converts only by an integer factor, and raises rather than wrap, int4 and
      uint4 included. A bool or bit tensor has no unit: {!v} and {!map} refuse
      one, and {!value} raises on one that {!Nx.Ptree.cast} or
      {!Nx.Ptree.Payload} built.

      Arithmetic on payloads ({!mul}, {!div}, {!pow}, {!add}, {!sub}, {!map})
      follows nx: integers wrap and truncate, and float elements overflow and
      underflow as their dtype does. *)

  type 'p t
  (** The type for values ['p] in a unit. *)

  val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'a t -> 'b t
  (** [walk c q] reports [q]'s unit with one [Walk.case] holding its canonical
      text, then walks its payload with [Walk.leaf], both at [c]'s path. A
      quantity's structure is [Nx.Ptree.instantiate (module Quantity)]. *)

  (** {1:constructors Constructors} *)

  val v : Unit.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t t
  (** [v u x] is [x] in [u]. Raises [Invalid_argument]
      ["Quantity.v: a bool tensor has no unit"] if [x] is a bool tensor, and
      with [a bit tensor] for a bit one. *)

  (** {1:queries Queries} *)

  val unit : 'p t -> Unit.t
  (** [unit q] is [q]'s unit. *)

  val value : Unit.t -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t
  (** [value u q] is [q]'s payload in [u]: the payload itself when [unit q] is
      [u], otherwise the payload times [Unit.ratio d (unit q) u], with [d] the
      payload's dtype. The result is within 1.5 ulps of the exact product, and
      eager and compiled calls agree bit for bit on one device.

      Raises [Invalid_argument] as {!Unit.ratio} does, its message naming
      [Quantity.value], as in
      ["Quantity.value: the factor from 1e-35 kg s^-2 to kg s^-2 is 1e-35, which
       is 0 in float16"]. An integer payload raises when an element times the
      factor leaves the dtype, at once on a concrete payload and when the
      compiled call returns on a traced one, as in
      ["Quantity.value: element [41] of an int32 payload overflows converting
       1e3 m to m (factor 1000)"], or ["an int32 payload overflows"] for a
      scalar. The index is the first such element's in C order. *)

  (** {1:conversions Changing the unit} *)

  val convert : Unit.t -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t
  (** [convert u q] is [v u (value u q)]. It raises as {!value} does, its
      message naming [Quantity.convert]. *)

  val times : Unit.t -> 'p t -> 'p t
  (** [times u q] is [q]'s payload in [Unit.(unit q * u)]. *)

  val per : Unit.t -> 'p t -> 'p t
  (** [per u q] is [q]'s payload in [Unit.(unit q / u)]. *)

  (** {1:maps Maps} *)

  val map :
    (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t t -> ('c, 'd) Nx.t t
  (** [map f q] is [f] of [q]'s payload, in [q]'s unit. It is correct when
      [f (c·x) = c·f x] for every positive scalar [c], such as a scaling, a sum
      or a mean; nothing checks [f]. Raises [Invalid_argument]
      ["Quantity.map: the function returns a bool tensor"] if [f] does, and with
      [a bit tensor] for a bit one. *)

  val map2 :
    (('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) ->
    ('a, 'b) Nx.t t ->
    ('a, 'b) Nx.t t ->
    ('a, 'b) Nx.t t
  (** [map2 f a b] is [f] of [a]'s payload and [value (unit a) b], in [a]'s
      unit. It is correct when [f (c·x) (c·y) = c·f x y]. For an integer
      payload, [b] converts only when [unit b / unit a] is an integer: convert
      to the finer unit first. Raises as {!value} does, its message naming
      [Quantity.map2]. *)

  (** {1:algebra Algebra} *)

  val add : ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t
  (** [add a b] is [map2 Nx.add a b]. Its messages name [Quantity.add], as in
      ["Quantity.add: pix{b} does not convert to pix{a}: their quotient keeps
       pix{a}^-1 pix{b}"] for [a] in [pix{a}] and [b] in [pix{b}]. *)

  val sub : ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t
  (** [sub a b] is [map2 Nx.sub a b]. Its messages name [Quantity.sub]. *)

  val mul : ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t
  (** [mul a b] is the product of the payloads in [Unit.(unit a * unit b)]. *)

  val div : ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t
  (** [div a b] is the quotient of the payloads, as {!Nx.div} computes it, in
      [Unit.(unit a / unit b)]. *)

  val pow : int -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t
  (** [pow n q] is [q]'s payload to the power [n] in [Unit.(unit q ** n)],
      computed by repeated products: integers wrap, and the exponent is exact
      whatever the payload's dtype, so [pow 17] of [-1.] is [-1.] in float8. A
      negative [n] is the reciprocal of the power [-n], and [pow 0] is one, NaN
      included. Each product rounds, so a float result is within about [n - 1]
      half-ulps of the exact power: [pow 1000] in float32 can be hundreds of
      ulps off. Raises [Invalid_argument]
      ["Quantity.pow: -1 is negative and the payload is int32"] for a negative
      [n] on an integer payload. *)

  val root : int -> (float, 'b) Nx.t t -> (float, 'b) Nx.t t
  (** [root n q] is the real [n]-th root of [q]'s payload in
      [Unit.root n (unit q)], by [Nx.pow] at the exponent [1/n], then, for
      [n >= 3], one Newton step that corrects the rounding of [1/n]. A payload
      narrower than float32 is rooted in float32 and cast back, so the exponent
      is not rounded to its dtype. For odd [n] the root of an element below zero
      is the negative real root; for even [n] such an element raises
      [Invalid_argument], at once on a concrete payload and when the compiled
      call returns on a traced one, as in
      ["Quantity.root: element [3] of a float32 payload is below 0, whose root
       of order 2 is not real"]. NaN and [-0.] give what [Nx.pow] gives them.
      Raises [Invalid_argument] ["Quantity.root: 0 is below 1"] if [n < 1]. *)

  (** {1:fmt Formatting} *)

  val pp : Format.formatter -> ('a, 'b) Nx.t t -> unit
  (** [pp ppf q] formats [q]'s payload as {!Nx.pp} does, then its unit's
      canonical text. *)
end

(** Measured constants. *)
module Constant : sig
  (** A {e constant} is a measured value: a published decimal, its standard
      uncertainty, a unit and a name. Its value and uncertainty are kept exact,
      so a constant rounds once, from the published decimal, to the dtype a
      program asks for: a float32 program gets G with no detour through float64.

      {[
      let g = Codata.newtonian_gravitation Codata.v2022
      let gm = Quantity.mul (Constant.quantity Nx.float64 g) mass
      ]} *)

  type t
  (** The type for measured constants. *)

  val v : name:string -> string -> Unit.t -> t
  (** [v ~name text u] is the constant [text] in [u], named [name]. [text] is

      {v ["-"] digits ["." digits] ["(" digits ")"] ["e" ["-"] digits] v}

      the parenthesised digits counting units of the mantissa's last digit:
      ["6.67430(15)e-11"] is (6.67430 ± 0.00015)·10{^ -11}. Text without
      parentheses is exact.

      Raises [Invalid_argument] if [text] is outside this grammar, as in
      [{|Constant.v: "6.67430(15" is not a constant's value|}]; if its value is
      zero or its mantissa, its digits read as one integer, is 2{^ 62} or more;
      if its uncertainty is zero, since an exact value has no parentheses, or
      its digits are 2{^ 62} or more; or as {!Unit.decimal} does when the value
      or the uncertainty is past the coefficient's bound, as in
      ["Constant.v: the coefficient's numerator is past 4096 bits"]. When [text]
      breaks several of these, the first in this order is raised: the grammar,
      the value, the uncertainty, the coefficient's bound. *)

  val name : t -> string
  (** [name k] is [k]'s name. *)

  val unit : t -> Unit.t
  (** [unit k] is [k]'s unit. *)

  val quantity : ('a, 'b) Nx.dtype -> t -> ('a, 'b) Nx.t Quantity.t
  (** [quantity d k] is [k]'s value as a scalar of [d] in [unit k]: its
      magnitude rounded once to [d] as {!Unit.ratio} rounds, then its sign. A
      complex dtype has a zero imaginary part. An integer dtype holds [k] only
      if it holds [k]'s magnitude, so [-128] is refused in int8, and an unsigned
      dtype holds no negative constant.

      Raises [Invalid_argument] when [d] holds no rounding of [k]'s magnitude,
      for the reasons {!Unit.ratio} gives, its message naming
      [Constant.quantity], [k] and [k]'s text, as in
      ["Constant.quantity: Newtonian constant of gravitation: 6.67430(15)e-11 is
       0 in float8_e4m3"] and
      ["Constant.quantity: k: -128 has a magnitude int8 does not hold"]; or, for
      a negative [k] and an unsigned dtype, before any rounding, as in
      ["Constant.quantity: electron g factor: -2.00231930436092(36) is negative,
       which uint8 does not hold"]. *)

  val uncertainty : ('a, 'b) Nx.dtype -> t -> ('a, 'b) Nx.t Quantity.t
  (** [uncertainty d k] is [k]'s standard uncertainty, which is never negative,
      as a scalar of [d] in [unit k]: zero for an exact constant, otherwise
      rounded once as {!quantity} rounds. Raises [Invalid_argument] as
      {!quantity} does, its message naming [Constant.uncertainty], [k] and the
      uncertainty of [k]'s text, as in
      ["Constant.uncertainty: k: the uncertainty of 42.0(5) is not an integer"].
  *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf k] formats [k]'s name, its text and its unit's canonical text, as
      in [Newtonian constant of gravitation = 6.67430(15)e-11 kg^-1 m^3 s^-2].
      The unit is left out when it is {!Unit.one}, as in
      [fine-structure constant = 7.2973525643(11)e-3]. *)
end

(** CODATA releases of the measured constants. *)
module Codata : sig
  (** A {e release} is one CODATA adjustment of the measured constants, as NIST
      publishes it. A release is a value, and a function that needs measured
      constants takes one, so a program states which adjustment it computes
      with:

      {[
      let bohr codata = Constant.quantity Nx.float64 (Codata.bohr_radius codata)
      let a0 = bohr Codata.v2022
      ]}

      A release holds every measured quantity of NIST's table for its
      adjustment, named as the table names it, with the value and uncertainty
      the table gives. The table's exact rows are left out: the SI fixes those
      constants, so they are units ({!Unit.section-constants}). So are the rows
      that restate another row in other units: the energy equivalents and
      relationships, and the values given again in MeV, eV, u, Hz, K or m{^ -1}.
      Ratios and dimensionless quantities are kept.

      A quantity has the table's unit, with one change: the table writes the
      radian as 1, and a unit here carries it where the quantity is, by its
      definition, an angle per something or something per angle. A gyromagnetic
      ratio is an angular frequency per tesla, rad s{^ -1} T{^ -1}, and a
      reduced Compton wavelength is a wavelength per radian, m rad{^ -1}. Every
      other quantity, the atomic, natural and Planck units included, has the
      table's unit.

      The releases start at 2018, the first adjustment under the 2019 SI's exact
      constants. The accessors below name the quantities the formulas of physics
      and astronomy most often take as inputs; every other quantity of a release
      is in {!constants}. Both releases hold every accessor's constant. An
      accessor for a constant a release lacks raises [Invalid_argument] naming
      the accessor, the release's year and NIST's name for the constant, as in
      ["Codata.tau_mass: CODATA 2030 has no tau mass"]; accessors let a later
      release add constants without breaking a program. *)

  type t
  (** The type for CODATA releases. *)

  val v2018 : t
  (** [v2018] is the 2018 adjustment. *)

  val v2022 : t
  (** [v2022] is the 2022 adjustment. *)

  val year : t -> int
  (** [year r] is the year of [r]'s adjustment. *)

  val constants : t -> Constant.t list
  (** [constants r] is every constant of [r], in the order of NIST's table. *)

  (** {1:gravitation Gravitation and electromagnetism} *)

  val newtonian_gravitation : t -> Constant.t
  (** [newtonian_gravitation r] is G, the Newtonian constant of gravitation, in
      m{^ 3} kg{^ -1} s{^ -2}. *)

  val fine_structure : t -> Constant.t
  (** [fine_structure r] is α, the fine-structure constant, dimensionless. *)

  val vacuum_permeability : t -> Constant.t
  (** [vacuum_permeability r] is μ{_ 0}, the vacuum magnetic permeability, in N
      A{^ -2}. *)

  val vacuum_permittivity : t -> Constant.t
  (** [vacuum_permittivity r] is ε{_ 0}, the vacuum electric permittivity, in F
      m{^ -1}. *)

  (** {1:masses Masses} *)

  val dalton : t -> Constant.t
  (** [dalton r] is m{_ u}, the atomic mass constant, which is the dalton, in
      kg. *)

  val electron_mass : t -> Constant.t
  (** [electron_mass r] is m{_ e} in kg. *)

  val muon_mass : t -> Constant.t
  (** [muon_mass r] is m{_ μ} in kg. *)

  val tau_mass : t -> Constant.t
  (** [tau_mass r] is m{_ τ} in kg. *)

  val proton_mass : t -> Constant.t
  (** [proton_mass r] is m{_ p} in kg. *)

  val neutron_mass : t -> Constant.t
  (** [neutron_mass r] is m{_ n} in kg. *)

  val deuteron_mass : t -> Constant.t
  (** [deuteron_mass r] is m{_ d} in kg. *)

  val triton_mass : t -> Constant.t
  (** [triton_mass r] is m{_ t} in kg. *)

  val helion_mass : t -> Constant.t
  (** [helion_mass r] is m{_ h}, the mass of the helium-3 nucleus, in kg. *)

  val alpha_particle_mass : t -> Constant.t
  (** [alpha_particle_mass r] is m{_ α}, the mass of the helium-4 nucleus, in
      kg. *)

  (** {1:atomic Atomic physics} *)

  val rydberg : t -> Constant.t
  (** [rydberg r] is R{_ ∞}, the Rydberg constant, in m{^ -1}. *)

  val bohr_radius : t -> Constant.t
  (** [bohr_radius r] is a{_ 0} in m. *)

  val classical_electron_radius : t -> Constant.t
  (** [classical_electron_radius r] is r{_ e} in m. *)

  val compton_wavelength : t -> Constant.t
  (** [compton_wavelength r] is λ{_ C}, the electron's Compton wavelength, in m.
  *)

  val thomson_cross_section : t -> Constant.t
  (** [thomson_cross_section r] is σ{_ e} in m{^ 2}. *)

  val hartree_energy : t -> Constant.t
  (** [hartree_energy r] is E{_ h} in J. *)

  val bohr_magneton : t -> Constant.t
  (** [bohr_magneton r] is μ{_ B} in J T{^ -1}. *)

  val nuclear_magneton : t -> Constant.t
  (** [nuclear_magneton r] is μ{_ N} in J T{^ -1}. *)

  val electron_magnetic_moment : t -> Constant.t
  (** [electron_magnetic_moment r] is μ{_ e}, negative, in J T{^ -1}. *)

  val proton_magnetic_moment : t -> Constant.t
  (** [proton_magnetic_moment r] is μ{_ p} in J T{^ -1}. *)

  val electron_g_factor : t -> Constant.t
  (** [electron_g_factor r] is g{_ e}, negative, dimensionless. *)
end
