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
    once to its dtype ({!dtype}). {!v}, {!of_node} and {!of_exprs} make
    every program, so its fields hold together as {!v} states; programs are
    plain data, compared and hashed structurally. *)

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

(** {1:exprs Expressions} *)

(** Programs written as values.

    An {e expression} is a node whose operands are expressions, typed by the
    dtype {!Prog.dtype} gives its node. Each constructor's type states its
    kind's domain where a type can: float kinds take floats, comparisons give
    [bool], and two operands of one kind share a dtype. Where it cannot, the
    constructor checks its operands' dtypes as {!Prog.v} does, raising
    [Invalid_argument] where {!Prog.v} refuses the node. An expression used
    twice is one value: a [let] shares it. {!Prog.of_exprs} lowers
    expressions into a program.

    This is a Box-Muller normal draw from two uniform draws [u1] and [u2] of
    float32, written with the operators of a local open:
    {[
      let lit x = E.const Nx_array.Dtype.Float32 x in
      E.(sqrt (lit (-2.) * log u1) * cos (lit (2. *. Float.pi) * u2))
    ]} *)
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

      Raises [Invalid_argument] unless [k] is an operand below
      {!max_operands}. *)

  val coord : int -> (int64, Nx_array.Dtype.int64_elt) t
  (** [coord i] is [Coord i].

      Raises [Invalid_argument] unless [i] is an axis below
      {!Nx_array.Layout.max_rank}. *)

  val const : ('v, 's) Nx_array.Dtype.t -> 'v -> ('v, 's) t
  (** [const dt x] is [Const (dt, bits dt x)] ({!Prog.bits}).

      Raises [Invalid_argument] as {!Prog.bits} does. *)

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
  val sqrt : (float, 's) t -> (float, 's) t
  val exp : (float, 's) t -> (float, 's) t
  val exp2 : (float, 's) t -> (float, 's) t
  val log : (float, 's) t -> (float, 's) t
  val log2 : (float, 's) t -> (float, 's) t
  val log1p : (float, 's) t -> (float, 's) t
  val expm1 : (float, 's) t -> (float, 's) t
  val sin : (float, 's) t -> (float, 's) t
  val cos : (float, 's) t -> (float, 's) t
  val tan : (float, 's) t -> (float, 's) t
  val asin : (float, 's) t -> (float, 's) t
  val acos : (float, 's) t -> (float, 's) t
  val atan : (float, 's) t -> (float, 's) t
  val sinh : (float, 's) t -> (float, 's) t
  val cosh : (float, 's) t -> (float, 's) t
  val tanh : (float, 's) t -> (float, 's) t
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
    distinct unless they are leaves of one kind, dtype and field, or nodes of
    one kind and dtype over the same operands: a program depends on its
    expressions' values alone, never on which of them are one OCaml value.

    Raises [Invalid_argument] if an [In] of [outs] reads no operand of [ins]
    or is not of its dtype, or if [outs] is empty or [ins] and [outs] are
    more than {!max_operands}. *)
