(** Builders, and the laws that a rewrite keeps a value, for the suites of
    Shape's rules and of Symbolic.

    The integer law evaluates an expression and its rewrite with
    {!Interpreter.eval} at bindings of their leaves, and compares the two values
    as the machine computes them, wrapping included. The float law does so at
    IEEE's special values, bit for bit. *)

open Windtrap
open Tolk

(** {1:builders Builders} *)

val uop : Ops.t testable
(** [uop] is {!Uops.uop}. *)

val i : int -> [> `Int of Bigint.t ]
(** [i n] is the integer value [n]. *)

val var : ?dtype:Dtype.t -> string -> int -> int -> Ops.t
(** [var ~dtype name lo hi] is the variable [name] of type [dtype] (default
    {!Dtype.Weak_int}) ranging over [lo] to [hi]. *)

val before : ('a, Ops.t) Ops.Pattern_matcher.t -> 'a Ops.rules
(** [before m] is [m] applied before a node's sources. *)

val rewrite :
  ?order:((unit, Ops.t) Ops.Pattern_matcher.t -> unit Ops.rules) ->
  (unit, Ops.t) Ops.Pattern_matcher.t ->
  Ops.t ->
  Ops.t
(** [rewrite ~order m u] is [u] rewritten by [m] to a fixed point, calls
    skipped, [m] applied after a node's sources unless [order] is {!before}. *)

val symbolic : Ops.t -> Ops.t
(** [symbolic u] is [rewrite Shape.symbolic u]. *)

val sym : Ops.t -> Ops.t
(** [sym u] is [rewrite Symbolic.sym u]. *)

(** {1:integers Integer values} *)

val exact : Ops.t -> bool
(** [exact u] is [true] iff every node [u]'s value reads is a constant, a
    variable (a scalar parameter with bounds), a range, a hardware index, a
    cast, a bit reinterpretation or an arithmetic operation other than
    {!Op.Threefry}, none of a float type. A parameter's, buffer's or
    allocation's size is not read. These are the graphs {!keeps_value}
    evaluates. *)

val pp_binding : Format.formatter -> (string * [< Dtype.const ]) list -> unit
(** [pp_binding] formats a binding of leaves by name as [a=1, b=2]. *)

val keeps_value :
  ?count:int -> ?wrapping:bool -> name:string -> Ops.t -> Ops.t -> int
(** [keeps_value ~count ~wrapping ~name before after] checks that [after] has
    [before]'s value, within [after]'s bounds, at [count] (default [16])
    bindings of their leaves: the first binds each leaf at its least value, the
    second at its greatest, the others at values drawn from a state seeded by
    [name] and [before]. It skips a binding under which [before] divides by
    zero, a range is empty or two leaves share a name, and, unless [wrapping]
    (default [false]) or [before] has no leaf, one under which [before] wraps
    ({!Interpreter.overflows}). It is the number of bindings it checked: [0]
    unless [before] and [after] are both {!exact}. *)

(** {1:floats Float values} *)

val float32 : float -> Dtype.value
(** [float32 x] is [x] rounded to {!Dtype.Float32}. *)

val specials : Dtype.value list
(** [specials] is IEEE's special float32 values and a few ordinary ones: the
    zeros, the infinities, NaN, the greatest finite values, the least
    subnormals, the greatest subnormal, [1.], [-1.], [0.5], [3.] and [1e30]. *)

val same_float : Dtype.const testable
(** [same_float] compares constants bit for bit, any NaN equal to any NaN:
    neither nx nor a target fixes a computed NaN's bits. *)

val keeps_float_value : ?count:int -> name:string -> Ops.t -> Ops.t -> unit
(** [keeps_float_value ~count ~name before after] checks that [after] has
    [before]'s value, as {!same_float} compares, at [count] (default [24])
    bindings of the float32 parameters of slots [0] to [2] to {!specials}: the
    first [List.length specials] bindings bind each slot to each special in
    turn, the others to specials drawn from a state seeded by [name] and
    [before]. *)
