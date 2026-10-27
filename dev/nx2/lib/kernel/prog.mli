(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scalar programs: the kinds a node computes, and programs of them.

    A {e program} computes, at one index of a shape, from one element of
    each of its {e operands}, and gives one element for each of its
    {e outputs}. It is a sequence of {e nodes}. A node is a {e leaf}, an
    operand's element at the index, the index along an axis or a constant,
    or a {e kind} applied to earlier nodes. A kind is what one node
    computes from its operands' values, in its dtype's {e compute type}
    ({!section-results}), the type its arithmetic runs in before it rounds
    to the dtype. The dtypes a kind takes are its
    {e domain} ({!section-domains}). Each node has a dtype ({!dtype}), and
    its value is its kind's result in that dtype, exact or within a bound
    as the next paragraph states, a float rounded as
    {!Nx_array.Dtype.of_float} rounds it. A loop of a kernel library
    ({!Nx_kernel.S.section-loops}) runs a program at every index of its
    shape.

    A kind whose result its doc below, IEEE 754 or modular arithmetic
    defines exactly, as [Floor], [Add], a float's [Sqrt] and an integer's
    [Mul] are, gives that result rounded once to its dtype. A float [Recip]
    is [1 / x] by [Fdiv]. In a kernel library, every other kind is within
    the bound {!section-results} states for it, with the special values,
    signs and NaN results stated there. A kernel library may compute a kind
    through the C function of its name in [nx_kinds.h] ([nx_exp_f32] for
    [Exp] at float32), which gives these results, and states whether it
    does. On one device model it computes each kind to the same bits in all
    its paths, so that a kind gives the same bits run alone by its own
    kernel and as a node of any program. On a device that flushes
    subnormal numbers to zero, which its library states, the bounds hold
    where the operands and the result are normal.

    {!v} makes a program from its nodes, {!of_node} from one kind over its
    operands, and {!of_exprs} from {e expressions} ({!E}), nodes written as
    values whose operands are expressions. {!subst} rewrites a program's
    leaves. Programs are plain data, compared and hashed structurally.

    This program draws a normal number from two uniform draws [u1] and
    [u2] of float32 by Box-Muller, written with the operators of a local
    open. Each distinct subexpression becomes one node, in the order the
    walk from [z] finishes them:
    {[
    let f32 = Nx_array.Dtype.Float32 in
    let lit x = Prog.E.const f32 x in
    let u1 = Prog.E.input 0 f32 and u2 = Prog.E.input 1 f32 in
    let z =
      Prog.E.(sqrt (lit (-2.) * log u1) * cos (lit (2. *. Float.pi) * u2))
    in
    Prog.of_exprs ~ins:[| Nx_array.Dtype.Any f32; Nx_array.Dtype.Any f32 |]
      [| Prog.E.any z |]
    ]} *)

(** {1:kinds Kinds} *)

(** The type for kinds of one operand, of its dtype. Trigonometric kinds take
    radians. On complex numbers a kind is its principal value, with the branch
    cuts of C99 7.3 ({!section-results}). On integers [Neg] and [Abs]
    wrap. *)
type unary =
  | Neg
  | Recip  (** On integers, [x] for [1] and [-1], [0] otherwise. *)
  | Abs
  | Sign
      (** [-1], [0] or [1]; NaN for a NaN. On complex numbers [z / |z|], and
          [z] at [0]. *)
  | Sqrt
  | Exp
  | Exp2  (** [2{^x}]. *)
  | Log
  | Log2
  | Log1p  (** [log (1 + x)], accurate near [0]. *)
  | Expm1  (** [exp x - 1], accurate near [0]. *)
  | Sin
  | Cos
  | Tan
  | Asin  (** In \[[-π/2], [π/2]\]. *)
  | Acos  (** In \[[0], [π]\]. *)
  | Atan  (** In \[[-π/2], [π/2]\]. *)
  | Sinh
  | Cosh
  | Tanh
  | Asinh
      (** On complex numbers only, as nx's float [asinh] composes other
          kinds; its branch cuts lie on the imaginary axis beyond [-i] and
          [i]. *)
  | Acosh
      (** On complex numbers only; its real part is not negative, and its
          branch cut lies on the real axis below [1]. *)
  | Atanh
      (** On complex numbers only; its branch cuts lie on the real axis
          beyond [-1] and [1]. *)
  | Erf  (** [2/√π ∫₀ˣ e{^-t²} dt]. *)
  | Floor  (** Toward negative infinity; the identity on integers. *)
  | Ceil  (** Toward positive infinity; the identity on integers. *)
  | Round
      (** To the nearest integer, half away from zero; the identity on
          integers. *)
  | Trunc  (** Toward zero; the identity on integers. *)

(** The type for kinds of two operands of one dtype, of that dtype. Integers
    wrap. *)
type binary =
  | Add
  | Sub
  | Mul
  | Fdiv  (** The IEEE 754 quotient of floats. *)
  | Idiv
      (** The integer quotient truncated toward zero: [0] by zero, and a signed
          dtype's least value by [-1] is that value. *)
  | Mod
      (** The remainder, of the dividend's sign: the dividend by zero on
          integers, [fmod] on floats. *)
  | Pow
      (** The first operand to the power of the second; on complex numbers
          [e{^w log z}] of [z] to the power [w]. On integers a negative
          exponent gives [0], but a base of [1] gives [1], and a base of
          [-1] gives [1] for an even exponent and [-1] for an odd one. *)
  | Atan2  (** The angle of [(x, y)], [y] the first operand, in \[-π, π\]. *)
  | Maximum
      (** The IEEE 754 maximum: NaN propagates and [-0] orders below [+0]. *)
  | Minimum  (** As [Maximum]. *)
  | And  (** Bitwise on integers, logical on booleans. *)
  | Or  (** As [And]. *)
  | Xor  (** As [And]. *)
  | Threefry
      (** Threefry-2x32 with 20 rounds of the counter, the first operand,
          under the key, the second. Each operand and the result is a
          [uint64] holding the two 32-bit words, word [0] in its low
          half. *)

(** The type for comparisons of two operands of one dtype, to booleans.
    Unsigned dtypes order unsigned. *)
type compare = Equal | Not_equal | Less | Less_equal

(** {1:arities Kinds by arity} *)

(** The type for kinds of one operand. *)
type op1 =
  | Copy  (** The operand's bits, NaN payloads included, into its own dtype. *)
  | Unary of unary
  | Cast
      (** The operand's element stored into the result's dtype. A float stores
          as {!Nx_array.Dtype.of_float} says. An integer stores as its exact
          value rounded once to a float format, modulo the width to an integer
          dtype, and as [x <> 0] to a boolean. A boolean stores as [0] or [1].
          Into a complex dtype, these rules give the real part and the imaginary
          part is [0.]. A complex number stores part by part into a complex
          dtype, as [true] into a boolean if either part is non-zero, and by its
          real part into any other dtype. A float that keeps its format, as a
          float32 into a complex64's real part, keeps its bits. Into the
          operand's own dtype it is [Copy]. *)
  | Bitcast  (** The operand's bits read in the result's dtype, of one width. *)

(** The type for kinds of two operands. *)
type op2 = Binary of binary | Compare of compare

(** The type for kinds of three operands. *)
type op3 =
  | Where
      (** The second operand's element where the first's, a boolean, is
          [true], and the third's elsewhere. *)
  | Fma
      (** The product of the first two plus the third, rounded once; wrapping
          on integers. *)

(** {1:domains Domains}

    Each kind's domain, the dtypes it takes, is listed below. Every dtype
    has one order: [false < true]; integers by value; floats by value, [-0]
    below [+0]; complex numbers by real part, then imaginary part.
    [Maximum] and [Minimum] follow it. [Equal], [Not_equal], [Less] and
    [Less_equal] follow it too, but compare floats, and complex numbers'
    parts, as IEEE 754 does, so [-0] equals [+0]. A NaN, and a complex
    number with a NaN part, is a NaN to each kind: every comparison with it
    is [false] except [Not_equal], which is [true], and [Maximum] and
    [Minimum] give it, the first of two. A float NaN's bits are as
    {!section-results} states. On booleans [Maximum] is [Or] and [Minimum]
    is [And]. A unary kind keeps its operand's dtype, so no kind gives a
    complex number's modulus.

    - [Neg], [Recip], [Sign], [Add], [Sub], [Mul], [Pow]: every dtype but
      booleans.
    - [Fdiv]: floats and complex. [Idiv]: integers.
    - [Abs], [Floor], [Ceil], [Round], [Trunc], [Mod], [Fma]: floats and
      integers.
    - [Sqrt], [Exp], [Exp2], [Log], [Log2], and [Sin] to [Tanh]: floats and
      complex. [Asinh], [Acosh], [Atanh]: complex.
    - [Log1p], [Expm1], [Erf], [Atan2]: floats.
    - [And], [Or], [Xor]: integers and booleans. [Threefry]: [uint64].
    - [Maximum], [Minimum] and the comparisons: every dtype.
    - [Copy]: every dtype, into itself. [Cast]: every pair. [Bitcast]: every
      pair of one width.
    - [Where]: a boolean, then two operands of any one dtype. *)

val accepts1 :
  op1 -> ('a, 'b) Nx_array.Dtype.t -> ('v, 's) Nx_array.Dtype.t -> bool
(** [accepts1 k x y] is [true] iff [k] takes an operand of [x] to a result of
    [y]. [y] is [x] for [Copy] and [Unary]. *)

val accepts2 : op2 -> ('a, 'b) Nx_array.Dtype.t -> bool
(** [accepts2 k x] is [true] iff [k] takes two operands of [x]. Its result is
    of [x] for [Binary], [bool] for [Compare]. *)

val accepts3 :
  op3 -> ('c, 'e) Nx_array.Dtype.t -> ('a, 'b) Nx_array.Dtype.t -> bool
(** [accepts3 k c x] is [true] iff [k] takes a first operand of [c] and two of
    [x]. Its result is of [x]. *)

(** {1:results Results}

    A dtype's compute type is the dtype itself for float32, float64, the
    32- and 64-bit integers and the complex dtypes. Floats narrower than
    float32 compute in float32 and round once to their dtype. 8- and 16-bit
    integers compute in the 32-bit integer of their signedness, their
    operands sign- or zero-extended, and wrap to their dtype. Booleans
    compute in uint32. A complex64 computes as a pair of float32, the real
    part first as it is stored, and a complex128 as a pair of float64.

    A float kind's bound holds in its compute type, in ulps: the distance
    in ordered-bit ranks from the correctly rounded result, [-0] and [+0]
    one rank apart.

    {v
    Kind                                        float32   float64
    Exp, Exp2, Sin, Cos, Asin, Acos, Atan,
      Cosh, Tanh, Erf                              2         2
    Expm1, Log, Log2, Log1p                        1         2
    Atan2, Sinh                                    3         2
    Tan, Pow                                       4         2
    v}

    The float32 bounds of the kinds of one operand hold at every operand.
    The other bounds, and the complex ones below, hold at every operand
    measured. The special values of these kinds (C99 Annex F) and the signs
    of their zero results are exact.

    A float kind with a NaN operand gives a NaN, but for [Pow] of a base
    [1] or an exponent [0], which Annex F makes [1]. A kind gives a NaN
    from numbers where IEEE 754 or Annex F does, as [Sin] of an infinity.
    Which NaN, its sign and its payload are unspecified. They differ
    between targets, and between a target's vector and scalar code. [Neg]
    and [Abs] set a NaN's sign bit as they do a number's, its payload kept.

    {2:complex_results Complex numbers}

    [u] is the unit roundoff of a complex dtype's part, [2{^-24}] or
    [2{^-53}]. The error of a result [z'] against the exact [z] is
    normwise, [|z' - z| / |z|]. Each complex bound holds where no part of
    the computation overflows or underflows.

    [Add], [Sub] and [Neg] act on each part, so a NaN part leaves the other
    part's result as it is. [Mul] is the textbook product
    [(ar br - ai bi) + (ar bi + ai br) i], each part one fused multiply-add
    over the other product, within [2u]. [Fdiv] is Smith's algorithm: the
    divisor's larger part divides its smaller, so an intermediate overflows
    or underflows only where a part of the quotient does, and
    [(0 + 2{^-50} i) / (0 - 2{^-1074} i)] is [-inf + 0 i]. Each of its
    products that feeds a sum is a fused multiply-add, and it is within
    [4u]. [Recip] is [(1 + 0 i) / x] by [Fdiv]. A NaN part in an operand of
    [Mul], [Fdiv] or [Recip] makes both parts of the result NaN.

    [Exp], [Exp2], [Log], [Log2], [Sqrt], [Sin], [Cos], [Tan], [Asin],
    [Acos], [Atan], [Sinh], [Cosh], [Tanh], [Asinh], [Acosh], [Atanh] and
    [Pow] are the principal values of C99 7.3 (ISO/IEC 9899), with the
    special values of its Annex G, G.6. Their bounds, in [u], hold where
    the result's modulus is a normal float:

    {v
    Sqrt, Sign                                 3
    Exp, Exp2, Log, Log2                       4
    Asin, Acos, Atan, Asinh, Acosh, Atanh      5
    Sin, Cos, Tan, Sinh, Cosh, Tanh            6
    Pow, z to the power w                      3 + 3 |w log z|
    v}

    On the real axis, at a non-zero real part where the float kind is
    defined, the real part is the float kind's result, and [Pow]'s is at a
    positive base and a real exponent.

    Where Annex G leaves a sign unspecified, each kind keeps
    [f (conj z) = conj (f z)] and its parity bit for bit, a NaN part's sign
    bit counting as its sign. Where a result's sign comes from a NaN part,
    as the infinite imaginary part of [Sqrt (-inf + i NaN)] does, that sign
    is unspecified, as Annex G allows. At operands of positive parts each
    kind gives:

    {v
    Exp (-inf + i inf), Exp (-inf + i NaN)      +0 + i0
    Exp (inf + i inf), Exp (inf + i NaN)        inf + i NaN
    Sqrt (-inf + i NaN)                         NaN + i inf
    Sinh (+0 + i inf), Sinh (+0 + i NaN)        +0 + i NaN
    Sinh (inf + i inf), Sinh (inf + i NaN)      inf + i NaN
    Cosh (+0 + i inf), Cosh (+0 + i NaN),
      Cosh (NaN + i0)                           NaN + i0
    Cosh (inf + i inf)                          inf + i NaN
    Tanh (inf + i inf), Tanh (inf + i NaN)      1 + i0
    Asin (inf + i NaN)                          NaN + i inf
    Acos (inf + i NaN)                          NaN - i inf
    Atan (inf + i NaN)                          pi/2 + i0
    Asinh (NaN + i inf)                         inf + i NaN
    Atanh (NaN + i inf)                         +0 + i pi/2
    v} *)

(** {1:bits Bits} *)

val bits : ('v, 's) Nx_array.Dtype.t -> 'v -> string
(** [bits dt x] is the bits of [x], an element of [dt], in the host's byte
    order: {!Nx_array.Dtype.bytes}[ dt 1] bytes, a sub-byte element in the low
    bits of one byte. It is the payload of [Const]. A float is stored as
    {!Nx_array.Dtype.of_float} says.

    Raises [Invalid_argument] for an [int] outside [dt]'s range. *)

(** {1:programs Programs}

    A program's nodes come in order, and a node refers to earlier nodes by
    index. Every program holds together as {!v} states: {!v} checks it, and
    {!of_node}, {!of_exprs} and {!subst} make programs that hold. *)

(** The type for a program's nodes. *)
type node =
  | In of int  (** [In i] is operand [i] at the iteration index. *)
  | Coord of int
      (** [Coord i] is the iteration index along axis [i], as an [int64]. *)
  | Const of Nx_array.Dtype.any * string
      (** [Const (dt, b)] is the element of [dt] whose bits are [b]
          ({!val-bits}), stored as they are: any bits of [dt]'s width in a
          string of {!Nx_array.Dtype.bytes}[ dt 1] bytes, NaN payloads
          included, for [bool] any byte, [true] iff it is not zero, and for
          [bit] a byte of [0] or [1] alone, as {!v} checks. *)
  | Op1 of op1 * Nx_array.Dtype.any * int
      (** [Op1 (k, dt, i)] is [k] of node [i] into [dt]. *)
  | Op2 of op2 * int * int  (** [Op2 (k, i, j)] is [k] of nodes [i] and [j]. *)
  | Op3 of op3 * int * int * int
      (** [Op3 (k, i, j, l)] is [k] of nodes [i], [j] and [l]. *)

type t = private {
  ins : Nx_array.Dtype.any iarray;  (** The operands' dtypes, in order. *)
  nodes : node iarray;  (** The nodes, in order. *)
  outs : int iarray;  (** The nodes it outputs, in order. *)
}
(** The type for programs. *)

val v :
  ins:Nx_array.Dtype.any iarray -> node iarray -> outs:int iarray -> t
(** [v ~ins nodes ~outs] is the program over operands of dtypes [ins] whose
    results are the nodes [outs].

    Raises [Invalid_argument] unless every reference names an earlier node,
    every [In] an operand and every [Coord] an axis below
    {!Nx_array.Layout.max_rank}; every node's kind takes its operands'
    dtypes ({!dtype}): [Op1 (k, y, i)] where {!accepts1}[ k x y], [x] node
    [i]'s; [Op2 (k, i, j)] where nodes [i] and [j] are of one dtype [x] and
    {!accepts2}[ k x]; [Op3 (k, i, j, l)] where nodes [j] and [l] are of one
    dtype [x] and {!accepts3}[ k c x], [c] node [i]'s; every [Const]'s bits
    are ones [Const] takes; and [outs] is not empty and names nodes. *)

val of_node : ins:Nx_array.Dtype.any iarray -> node -> t
(** [of_node ~ins n] is the program of the one node [n] over operands of dtypes
    [ins]: [v ~ins nodes ~outs:[| k |]], where [nodes] is [In 0] to
    [In (k - 1)] then [n] and [k] the length of [ins]. Its reference [j]
    names operand [j].

    Raises [Invalid_argument] as {!v} does. *)

val dtype : t -> int -> Nx_array.Dtype.any
(** [dtype p i] is the dtype of [p]'s node [i]: its operand's for [In], [int64]
    for [Coord], the one it names for [Const] and [Op1], its operands' for a
    [Binary] kind, [bool] for a [Compare] kind, and its last two operands' for
    [Op3].

    Raises [Invalid_argument] if [i] is not a node of [p]. *)

val is_identity : t -> bool
(** [is_identity p] is [true] iff [p] has one operand and one output, and its
    output is that operand's [In] node, so that [p] gives its operand
    back. *)

val single : t -> node option
(** [single p] is [Some n] iff [p] is {!of_node}[ ~ins n] where [n] reads
    each of [p]'s operands in order: a [Const] or a [Coord] over zero
    operands, or a kind of [m] operands reading nodes [0] to [m - 1] over
    [m] operands. [None] otherwise. A library may run such a program by a
    kernel of [n]'s kind alone. *)

val is_single : t -> bool
(** [is_single p] is [true] iff {!single}[ p] is [Some n], where [n] is then
    [p]'s last node. It allocates nothing. *)

(** {1:exprs Expressions} *)

(** Programs written as values.

    An {e expression} is a node whose operands are expressions, typed by the
    dtype {!Prog.dtype} gives its node. Each constructor's type states its
    kind's domain where a type can: float kinds take floats, comparisons give
    [bool], and two operands of one kind share a dtype. Where it cannot, the
    constructor checks its operands' dtypes as {!Prog.v} does, raising
    [Invalid_argument] where {!Prog.v} refuses the node. An expression used
    twice is one value, which a [let] shares. {!Prog.of_exprs} lowers
    expressions into a program. *)
module E : sig
  type ('v, 's) t
  (** The type for expressions of dtype [('v, 's) Nx_array.Dtype.t]. *)

  type any
  (** The type for expressions of any dtype. *)

  val any : ('v, 's) t -> any
  (** [any e] is [e]. *)

  val dtype : ('v, 's) t -> ('v, 's) Nx_array.Dtype.t
  (** [dtype e] is [e]'s dtype. *)

  (** {1:nodes Nodes} *)

  val input : int -> ('v, 's) Nx_array.Dtype.t -> ('v, 's) t
  (** [input k dt] is [In k], operand [k] of dtype [dt].

      Raises [Invalid_argument] if [k] is negative. *)

  val coord : int -> (int64, Nx_array.Dtype.int64_elt) t
  (** [coord i] is [Coord i].

      Raises [Invalid_argument] unless [i] is an axis below
      {!Nx_array.Layout.max_rank}. *)

  val const : ('v, 's) Nx_array.Dtype.t -> 'v -> ('v, 's) t
  (** [const dt x] is [Const (dt, bits dt x)] ({!Prog.val-bits}).

      Raises [Invalid_argument] as {!Prog.val-bits} does. *)

  val op1 : op1 -> ('w, 'r) Nx_array.Dtype.t -> ('v, 's) t -> ('w, 'r) t
  (** [op1 k dt x] is [Op1 (k, dt, x)]. *)

  val binary : binary -> ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  (** [binary k x y] is [Op2 (Binary k, x, y)]. *)

  val compare :
    compare -> ('v, 's) t -> ('v, 's) t -> (bool, Nx_array.Dtype.bool_elt) t
  (** [compare k x y] is [Op2 (Compare k, x, y)]. *)

  val op3 : op3 -> ('c, 'e) t -> ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  (** [op3 k c x y] is [Op3 (k, c, x, y)]. *)

  (** {1:kinds Kinds}

      Each function makes the node of the kind it names: [sqrt x] is
      [op1 (Unary Sqrt) (dtype x) x], [rem x y] is [binary Mod x y],
      [less x y] is [compare Less x y], [cast dt x] is [op1 Cast dt x] and
      [where c x y] is [op3 Where c x y]. *)

  val neg : ('v, 's) t -> ('v, 's) t
  val recip : ('v, 's) t -> ('v, 's) t
  val abs : ('v, 's) t -> ('v, 's) t
  val sign : ('v, 's) t -> ('v, 's) t
  val sqrt : ('v, 's) t -> ('v, 's) t
  val exp : ('v, 's) t -> ('v, 's) t
  val exp2 : ('v, 's) t -> ('v, 's) t
  val log : ('v, 's) t -> ('v, 's) t
  val log2 : ('v, 's) t -> ('v, 's) t
  val log1p : (float, 's) t -> (float, 's) t
  val expm1 : (float, 's) t -> (float, 's) t
  val sin : ('v, 's) t -> ('v, 's) t
  val cos : ('v, 's) t -> ('v, 's) t
  val tan : ('v, 's) t -> ('v, 's) t
  val asin : ('v, 's) t -> ('v, 's) t
  val acos : ('v, 's) t -> ('v, 's) t
  val atan : ('v, 's) t -> ('v, 's) t
  val sinh : ('v, 's) t -> ('v, 's) t
  val cosh : ('v, 's) t -> ('v, 's) t
  val tanh : ('v, 's) t -> ('v, 's) t
  val asinh : ('v, 's) t -> ('v, 's) t
  val acosh : ('v, 's) t -> ('v, 's) t
  val atanh : ('v, 's) t -> ('v, 's) t
  val erf : (float, 's) t -> (float, 's) t
  val floor : ('v, 's) t -> ('v, 's) t
  val ceil : ('v, 's) t -> ('v, 's) t
  val round : ('v, 's) t -> ('v, 's) t
  val trunc : ('v, 's) t -> ('v, 's) t
  val add : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  val sub : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  val mul : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  val fdiv : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  val idiv : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  val rem : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  val pow : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  val atan2 : (float, 's) t -> (float, 's) t -> (float, 's) t
  val maximum : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  val minimum : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  val logand : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  val logor : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  val logxor : ('v, 's) t -> ('v, 's) t -> ('v, 's) t

  val threefry :
    (int64, Nx_array.Dtype.uint64_elt) t ->
    (int64, Nx_array.Dtype.uint64_elt) t ->
    (int64, Nx_array.Dtype.uint64_elt) t

  val equal : ('v, 's) t -> ('v, 's) t -> (bool, Nx_array.Dtype.bool_elt) t
  val not_equal : ('v, 's) t -> ('v, 's) t -> (bool, Nx_array.Dtype.bool_elt) t
  val less : ('v, 's) t -> ('v, 's) t -> (bool, Nx_array.Dtype.bool_elt) t
  val less_equal : ('v, 's) t -> ('v, 's) t -> (bool, Nx_array.Dtype.bool_elt) t
  val copy : ('v, 's) t -> ('v, 's) t
  val cast : ('w, 'r) Nx_array.Dtype.t -> ('v, 's) t -> ('w, 'r) t
  val bitcast : ('w, 'r) Nx_array.Dtype.t -> ('v, 's) t -> ('w, 'r) t
  val where : (bool, 'c) t -> ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  val fma : ('v, 's) t -> ('v, 's) t -> ('v, 's) t -> ('v, 's) t

  (** {1:operators Operators}

      For a local open, [E.( … )], where they shadow [Stdlib]'s. *)

  val ( ~- ) : ('v, 's) t -> ('v, 's) t
  (** [~- x] is [neg x]. *)

  val ( + ) : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  (** [x + y] is [add x y]. *)

  val ( - ) : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  (** [x - y] is [sub x y]. *)

  val ( * ) : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  (** [x * y] is [mul x y]. *)

  val ( / ) : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  (** [x / y] is [idiv x y] on integers and [fdiv x y] on other dtypes. *)

  val ( mod ) : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  (** [x mod y] is [rem x y]. *)

  val ( ** ) : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  (** [x ** y] is [pow x y]. *)

  val ( land ) : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  (** [x land y] is [logand x y]. *)

  val ( lor ) : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  (** [x lor y] is [logor x y]. *)

  val ( lxor ) : ('v, 's) t -> ('v, 's) t -> ('v, 's) t
  (** [x lxor y] is [logxor x y]. *)

  val ( && ) : (bool, 's) t -> (bool, 's) t -> (bool, 's) t
  (** [x && y] is [logand x y], at the precedence of [Stdlib.( && )], below
      the comparisons. *)

  val ( || ) : (bool, 's) t -> (bool, 's) t -> (bool, 's) t
  (** [x || y] is [logor x y], at the precedence of [Stdlib.( || )]. *)

  val ( = ) : ('v, 's) t -> ('v, 's) t -> (bool, Nx_array.Dtype.bool_elt) t
  (** [x = y] is [equal x y]. *)

  val ( <> ) : ('v, 's) t -> ('v, 's) t -> (bool, Nx_array.Dtype.bool_elt) t
  (** [x <> y] is [not_equal x y]. *)

  val ( < ) : ('v, 's) t -> ('v, 's) t -> (bool, Nx_array.Dtype.bool_elt) t
  (** [x < y] is [less x y]. *)

  val ( <= ) : ('v, 's) t -> ('v, 's) t -> (bool, Nx_array.Dtype.bool_elt) t
  (** [x <= y] is [less_equal x y]. *)

  val ( > ) : ('v, 's) t -> ('v, 's) t -> (bool, Nx_array.Dtype.bool_elt) t
  (** [x > y] is [less y x]. *)

  val ( >= ) : ('v, 's) t -> ('v, 's) t -> (bool, Nx_array.Dtype.bool_elt) t
  (** [x >= y] is [less_equal y x]. *)
end

val of_exprs : ins:Nx_array.Dtype.any iarray -> E.any iarray -> t
(** [of_exprs ~ins outs] is the program over operands of dtypes [ins] whose
    results are the expressions [outs]. Its nodes are [outs]' distinct
    subexpressions, each once, in the order a depth-first walk from the
    outputs, operands left to right, finishes them. Two expressions are
    distinct unless they are leaves of one kind, dtype and operand, axis or
    bits, or nodes of one kind and dtype over the same operands, so the
    program depends on its expressions' values alone, whichever of them
    are one OCaml value.

    Raises [Invalid_argument] if an [In] of [outs] reads no operand of [ins]
    or is not of its dtype, or if [outs] is empty. *)

(** {1:substitution Substitution} *)

val subst :
  ?outs:int iarray ->
  ins:Nx_array.Dtype.any iarray ->
  (node -> E.any option) ->
  t ->
  t
(** [subst ?outs ~ins f p] is the program over operands of dtypes [ins]
    that computes [p]'s outputs, or with [outs] the outputs of [p] at those
    indices, in that order, with each leaf replaced. For each [In], [Coord]
    or [Const] node [n] of [p] these outputs reach, [f n = Some e] replaces
    [n] by the expression [e], over operands of [ins], which must be of
    [n]'s dtype, and [f n = None] keeps [n], where a kept [In i] must name
    operand [i] of [ins], of [n]'s dtype. [f] is called once on each leaf
    these outputs reach, and on no kind node and no other leaf. Kind nodes
    are rebuilt over their substituted operands. Its nodes are lowered as
    {!of_exprs} lowers them, distinct subexpressions once in depth-first
    order, so the result holds as {!v} states.

    Raises [Invalid_argument] if an [e] is not of its node's dtype, as
    {!of_exprs} does for the expressions over [ins], or if [outs] is empty
    or names no output of [p]. *)
