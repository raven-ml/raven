(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scalar programs: the kinds a step computes, and programs of them.

    A kind is what one step of a scalar program computes from one element of
    each operand. Each computes, per compute type, as the function of its name
    in [nx_kinds.h] ([Exp] as [nx_exp_f32] on float32), or as code that gives
    the same bits. That header states the compute types, the transcendental
    kinds' bounds in ulps, their special values and the NaNs each kind gives. *)

(** {1:kinds Kinds} *)

(** The type for kinds of one operand, of its dtype. Trigonometric kinds take
    radians. *)
type unary =
  | Neg
  | Recip  (** On integers, [x] for [1] and [-1], [0] otherwise. *)
  | Abs
  | Sign  (** [-1], [0] or [1]; NaN for a NaN. *)
  | Sqrt
  | Exp
  | Exp2  (** [2{^x}], exact at integers whose power is in the dtype. *)
  | Log
  | Log2  (** Exact at powers of two. *)
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
  | Pow  (** The first operand to the power of the second. *)
  | Atan2  (** The angle of [(x, y)], [y] the first operand, in \[-π, π\]. *)
  | Maximum
      (** The IEEE 754 maximum: NaN propagates and [-0] orders below [+0]. *)
  | Minimum  (** As [Maximum]. *)
  | And  (** Bitwise on integers, logical on booleans. *)
  | Or  (** As [And]. *)
  | Xor  (** As [And]. *)
  | Threefry
      (** Threefry-2x32 with 20 rounds of the counter, the first operand, under
          the key, the second: [uint64] words, the low half first. *)

(** The type for comparisons of two operands of one dtype, to booleans.
    Unsigned dtypes order unsigned. *)
type compare = Equal | Not_equal | Less | Less_equal

(** {1:arities Kinds by arity} *)

(** The type for kinds of no operand. *)
type op0 =
  | Fill of string
      (** [Fill b] is the element whose bits are [b], in the result's dtype and
          the host's byte order. *)
  | Iota of int
      (** [Iota i] is each element's index along the result's axis [i], in the
          result's dtype. *)

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

    Where each kind is defined. Every dtype has one order: [false < true];
    integers by value; floats by value, [-0] below [+0]; complex numbers by
    real part, then imaginary part. [Maximum] and [Minimum] follow it.
    [Equal], [Not_equal], [Less] and [Less_equal] follow it too, but compare
    floats, and complex numbers' parts, as IEEE 754 does: [-0] equals [+0]. A
    NaN, and a complex number with a NaN part, is a NaN to each kind: every
    comparison with it is [false] except [Not_equal], which is [true], and
    [Maximum] and [Minimum] give it. On booleans [Maximum] is [Or] and
    [Minimum] is [And]. A kind keeps its operands' dtype where its type
    says so: the absolute value of a complex number is no kind.

    - [Neg], [Recip], [Add], [Sub], [Mul]: every dtype but booleans.
    - [Fdiv]: floats and complex. [Idiv]: integers.
    - [Abs], [Sign], [Floor], [Ceil], [Round], [Trunc], [Mod], [Pow], [Fma]:
      floats and integers.
    - [Sqrt], [Atan2] and the transcendental kinds, [Exp] to [Erf]: floats.
    - [And], [Or], [Xor]: integers and booleans. [Threefry]: [uint64].
    - [Maximum], [Minimum] and the comparisons: every dtype.
    - [Copy]: every dtype, into itself. [Cast]: every pair. [Bitcast]: every
      pair of one width.
    - [Where]: a boolean, then two operands of any one dtype.
    - [Fill]: every dtype. [Iota]: floats and integers. *)

val accepts0 : op0 -> ('v, 's) Nx_array.Dtype.t -> bool
(** [accepts0 k dt] is [true] iff [k] makes elements of [dt]. *)

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

(** {1:bits Bits} *)

val bits : ('v, 's) Nx_array.Dtype.t -> 'v -> string
(** [bits dt x] is the bits of [x], an element of [dt], in the host's byte
    order: {!Nx_array.Dtype.bytes}[ dt 1] bytes, a sub-byte element in the low
    bits of one byte. It is the payload of [Fill] and [Const]. A float is
    stored as {!Nx_array.Dtype.of_float} says.

    Raises [Invalid_argument] for an [int] outside [dt]'s range. *)

(** {1:programs Programs} *)

(** The type for a program's nodes. A node refers to earlier nodes by index. *)
type node =
  | In of int  (** [In i] is operand [i] at the iteration index. *)
  | Coord of int
      (** [Coord i] is the iteration index along axis [i], as an [int64]. *)
  | Const of Nx_array.Dtype.any * string
      (** [Const (dt, b)] is the element of [dt] whose bits are [b]
          ({!bits}). *)
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
(** The type for programs. A program runs at each index of a shape, reading
    its operands at that index. Each node's value is its exact result rounded
    once to its dtype ({!dtype}). {!v} makes every program, so its fields
    hold together as {!v} states; programs are plain data, compared and hashed
    structurally. *)

val max_operands : int
(** [max_operands] is [16], the most operands plus outputs a program has. *)

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
    are an element of its dtype; [outs] is not empty and names nodes; and
    operands plus outputs are at most {!max_operands}. *)

val of_node : ins:Nx_array.Dtype.any iarray -> node -> t
(** [of_node ~ins n] is the program of the one node [n] over operands of dtypes
    [ins]: [v ~ins nodes ~outs:[| k |]], [nodes] being [In 0] to [In (k - 1)]
    then [n], [k] the length of [ins]. [n]'s references name operands: node
    [j] is operand [j].

    Raises [Invalid_argument] as {!v} does. *)

val dtype : t -> int -> Nx_array.Dtype.any
(** [dtype p i] is the dtype of [p]'s node [i]: its operand's for [In], [int64]
    for [Coord], the one it names for [Const] and [Op1], its operands' for a
    [Binary] kind, [bool] for a [Compare] kind, and its last two operands' for
    [Op3].

    Raises [Invalid_argument] if [i] is not a node of [p]. *)

val is_identity : t -> bool
(** [is_identity p] is [true] iff [p] has one operand and one output, and its
    output is that operand's [In] node: [p] gives its operand back. *)
