(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Symbolic dimensions for shape-polymorphic tensors.

    This module enables shape-polymorphic tensor operations by representing
    tensor dimensions as symbolic expressions that can be resolved at runtime.
    Symbolic shapes support dynamic batch sizes, variable sequence lengths, and
    other runtime-determined dimensions in compiled kernels.

    {1 Overview}

    A symbolic shape consists of dimension expressions that may be static
    constants, symbolic variables, or arithmetic combinations. Variables can be
    bound to concrete values at runtime, enabling a single compiled kernel to
    handle multiple input shapes.

    Create shapes with {!of_ints} for static dimensions or use {!dynamic} to
    introduce symbolic variables. Variables created with {!var} are distinct
    even if they share the same name, preventing accidental aliasing.

    {1 Key Concepts}

    {2 Variables and Identity}

    Each call to {!var} creates a fresh variable with a unique identity.
    Variables with the same name are distinct:
    {[
      let v1 = Symbolic_shape.var "x" ~min:0 ~max:10 in
      let v2 = Symbolic_shape.var "x" ~min:0 ~max:10 in
      assert (v1 != v2)
      (* Distinct variables *)
    ]}

    This prevents unintended variable sharing across independent operations.

    {2 Dimension Expressions}

    Dimensions support arithmetic operations to express relationships:
    {[
      let n = Symbolic_shape.var "n" ~min:1 ~max:100 in
      let dim_n = Symbolic_shape.dim_of_var n in
      let dim_2n = Symbolic_shape.mul dim_n (Symbolic_shape.static 2) in
      let dim_2n_plus_1 = Symbolic_shape.add dim_2n (Symbolic_shape.static 1)
    ]}

    {2 Runtime Binding}

    Variables remain unbound until explicitly assigned values with {!bind}.
    Binding checks that values respect variable bounds. Evaluation functions
    return [None] for shapes containing unbound variables.

    {2 Reshape with Inference}

    The special {!infer} dimension (represented as [-1]) allows NumPy-style
    reshape operations where one dimension is computed from the total element
    count. {!resolve_reshape} computes the inferred dimension:
    {[
      let from_shape = Symbolic_shape.of_ints [| 2; 3; 4 |] in
      let to_shape = [| Symbolic_shape.static 6; Symbolic_shape.infer |] in
      match Symbolic_shape.resolve_reshape ~from_shape ~to_shape with
      | Some resolved -> assert (Symbolic_shape.eval resolved = Some [| 6; 4 |])
      | None -> assert false
    ]} *)

type var
(** A symbolic variable representing a dimension.

    Variables have unique identities (compared by ID, not name or bounds) and
    mutable bindings that persist until changed. Each call to {!var} creates a
    fresh variable even if the name is reused. Variables track minimum and
    maximum bounds to validate runtime values. *)

type expr =
  | Const of int  (** Static dimension with fixed value *)
  | Var of var  (** Symbolic variable *)
  | Add of expr * expr  (** Sum of two expressions *)
  | Mul of expr * expr  (** Product of two expressions *)
  | Neg of expr  (** Negation of an expression *)

type dim = expr
(** A dimension is an expression.

    Dimensions may be static constants, symbolic variables, or arithmetic
    combinations. Use {!static} for constants, {!dim_of_var} for variables, and
    {!add}, {!mul}, {!neg} for arithmetic. *)

type t = dim array
(** A shape is an array of dimensions.

    Shapes are immutable arrays representing the dimensions of multi-dimensional
    tensors. Operations return new shapes without modifying originals. Each
    element is a dimension expression that may contain symbolic variables. Empty
    shapes represent scalars (rank-0 tensors). *)

(** {1 Creation} *)

val static : int -> dim
(** [static n] creates a static dimension with value [n].

    Static dimensions have fixed values known at creation time.

    @raise Invalid_argument if [n < 0]. *)

val dynamic : string -> min:int -> max:int -> dim
(** [dynamic name ~min ~max] creates a dynamic dimension with bounds.

    Creates a fresh symbolic variable and converts it to a dimension expression.
    Equivalent to [dim_of_var (var name ~min ~max)]. The variable can be bound
    to any value in the range [[min, max]] at runtime.

    @raise Invalid_argument if [min < 0] or [min > max]. *)

val var : string -> min:int -> max:int -> var
(** [var name ~min ~max] creates a fresh symbolic variable.

    Each call returns a distinct variable with a unique identity, regardless of
    [name]. Names are used for debugging and display purposes only. Variables
    remain unbound until explicitly assigned with {!bind}.

    The bounds [[min, max]] constrain valid runtime values.

    @raise Invalid_argument if [min < 0] or [min > max]. *)

val dim_of_var : var -> dim
(** [dim_of_var var] wraps a variable as a dimension expression. *)

val of_ints : int array -> t
(** [of_ints arr] creates a shape with all static dimensions.

    Each element of [arr] is converted to a static dimension using {!static}.
    This is the standard way to create concrete shapes.

    @raise Invalid_argument if any element of [arr] is negative. *)

val of_list : int list -> t
(** [of_list lst] creates a shape with all static dimensions.

    Equivalent to [of_ints (Array.of_list lst)].

    @raise Invalid_argument if any element of [lst] is negative. *)

(** {1 Dimension Operations} *)

val add : dim -> dim -> dim
(** [add d1 d2] creates a dimension expression [d1 + d2].

    Constructs an [Add] expression representing the sum of two dimensions.
    Useful for expressing padding or concatenation dimensions. *)

val mul : dim -> dim -> dim
(** [mul d1 d2] creates a dimension expression [d1 * d2].

    Constructs a [Mul] expression representing the product of two dimensions.
    Useful for expressing flattened or tiled dimensions. *)

val neg : dim -> dim
(** [neg d] creates a dimension expression [-d].

    Constructs a [Neg] expression representing the negation of a dimension.
    Rarely used directly; primarily for internal expression manipulation. *)

(** {1 Runtime Binding} *)

val bind : var -> int -> t -> unit
(** [bind var value shape] binds [value] to [var] globally and updates all
    occurrences of [var] in [shape] by identity.

    Performs a global mutation of the variable's mutable state. The shape is
    traversed to find all instances matching [var] by identity, including those
    within compound expressions. The binding persists until changed.

    Variables must be bound before shapes can be evaluated to concrete
    dimensions. Binding is checked against the variable's [min] and [max] bounds
    specified at creation.

    Time complexity: O(n) where n is the total size of all expression trees in
    the shape.

    @raise Invalid_argument if [value] is outside the variable's bounds. *)

val eval : t -> int array option
(** [eval shape] returns concrete shape if all dimensions are bound.

    Evaluates all dimensions in [shape] to produce an integer array. Returns
    [Some arr] if all symbolic variables are bound and all expressions can be
    computed. Returns [None] if any dimension contains an unbound variable.

    This is the primary way to extract concrete shapes for backend operations.
*)

val eval_dim : dim -> int option
(** [eval_dim dim] evaluates a single dimension.

    Returns [Some n] if [dim] is fully bound and can be evaluated to a concrete
    value. Returns [None] if [dim] contains unbound variables.

    Evaluates arithmetic expressions by depth-first recursion without
    memoization, evaluating subexpressions and applying the corresponding
    operations. *)

val partial_eval : t -> int option array
(** [partial_eval shape] returns an array of evaluated dimensions.

    Evaluates each dimension independently, returning [Some n] for bound
    dimensions and [None] for unbound dimensions. Unlike {!eval}, this succeeds
    even when some dimensions remain symbolic.

    Useful for debugging and displaying partially bound shapes. *)

val is_fully_bound : t -> bool
(** [is_fully_bound shape] returns true if all dimensions are bound.

    Checks whether every symbolic variable in [shape] has been assigned a value.
    Static dimensions are always considered bound. Returns [true] if
    [eval shape] would succeed. *)

val numel : t -> int option
(** [numel shape] returns the total number of elements if shape is fully bound.

    Computes the product of all dimensions if the shape can be fully evaluated.
    Returns [Some 1] for empty shapes (scalars). Returns [None] if any dimension
    contains unbound variables. *)

(** {1 Special Values} *)

val infer : dim
(** Special dimension value representing "infer from context".

    Equivalent to [-1] in NumPy reshape operations. Use this in target shapes to
    indicate that a dimension should be computed from the total element count.
    At most one dimension in a shape may be {!infer}; this constraint exists
    because multiple unknowns make element count calculation ambiguous. The
    constraint is enforced at runtime by {!resolve_reshape}.

    The {!resolve_reshape} function computes the concrete value for inferred
    dimensions based on the source shape's element count. *)

val is_infer : dim -> bool
(** [is_infer dim] returns true if dimension should be inferred.

    Checks whether [dim] evaluates to [-1], indicating it should be computed
    from context during reshape operations. Returns [false] for unbound
    variables, which cannot be evaluated to [-1]. *)

(** {1 Shape Resolution} *)

val resolve_reshape : from_shape:t -> to_shape:t -> t option
(** [resolve_reshape ~from_shape ~to_shape] resolves a reshape operation with
    inference.

    Computes concrete dimensions for {!infer} values in [to_shape] based on the
    element count of [from_shape]. At most one dimension in [to_shape] may be
    {!infer}.

    Returns [Some resolved_shape] if:
    - [from_shape] is fully bound (all dimensions have concrete values)
    - [to_shape] contains zero or one {!infer} dimensions
    - The total elements of [from_shape] divides evenly by known dimensions in
      [to_shape]

    Returns [None] if:
    - [from_shape] contains unbound variables
    - Element count doesn't divide evenly (i.e.,
      [total_elements mod known_product != 0])

    {4 Examples}

    Resolving a reshape with one inferred dimension:
    {[
      let from_shape = Symbolic_shape.of_ints [| 2; 3; 4 |] in
      let to_shape = [| Symbolic_shape.static 6; Symbolic_shape.infer |] in
      match Symbolic_shape.resolve_reshape ~from_shape ~to_shape with
      | Some resolved -> assert (Symbolic_shape.eval resolved = Some [| 6; 4 |])
      | None -> ()
    ]}

    @raise Invalid_argument
      if [to_shape] contains multiple {!infer} dimensions or any dimension
      evaluates to an invalid size (zero or negative). *)

val substitute : (var * int) list -> t -> t
(** [substitute bindings shape] substitutes variable bindings into [shape].

    Replaces variables in [shape] with their corresponding values from
    [bindings], creating a new shape with [Const] nodes where variables were
    substituted. Variables not present in [bindings] remain as [Var] nodes.
    Unlike {!bind}, this creates a new shape without mutating variable state.

    Binding list format: [(var, value)] pairs where [var] is matched by its
    unique identity (not by name).

    Useful for creating multiple specialized versions of a parametric shape
    without side effects. *)

(** {1 Analysis} *)

val vars : t -> var list
(** [vars shape] returns all unique symbolic variables in shape.

    Extracts all distinct variables from dimension expressions. Variables are
    compared by identity, so the same variable object appears only once even if
    used in multiple dimensions or compound expressions.

    Returns an empty list for shapes containing only static dimensions. The
    order of variables in the result is unspecified. *)

val var_id : var -> int
(** [var_id v] returns the unique identifier assigned to [v]. *)

val var_name : var -> string
(** [var_name v] returns the user-facing name of [v]. *)

val var_bounds : var -> int * int
(** [var_bounds v] returns the inclusive minimum and maximum bounds for [v]. *)

val is_static : t -> bool
(** [is_static shape] returns true if all dimensions are static.

    Checks whether [shape] contains only [Const] expressions with no symbolic
    variables. Static shapes can be evaluated without binding variables.

    Returns [true] for empty shapes. *)

val rank : t -> int
(** [rank shape] returns number of dimensions. *)

(** {1 Utilities} *)

val to_string : t -> string
(** [to_string shape] returns human-readable representation.

    Formats the shape as a bracketed list of dimension expressions. Static
    dimensions appear as integers. Variables show as [name#id] or
    [name#id=value] if bound. Empty variable names render as [v{id}]. Compound
    expressions use infix notation with parentheses. *)

val equal : t -> t -> bool
(** [equal s1 s2] compares shapes structurally.

    Returns [true] if [s1] and [s2] have the same rank and corresponding
    dimensions are structurally equal. Dimensions are compared by:
    - Constants: Equal if values match
    - Variables: Equal if variable identities match (not names)
    - Expressions: Equal if operators and subexpressions match recursively

    Two shapes with different but equivalent variables are not equal:
    {[
      let v1 = Symbolic_shape.var "x" ~min:0 ~max:10 in
      let v2 = Symbolic_shape.var "x" ~min:0 ~max:10 in
      let s1 = [| Symbolic_shape.dim_of_var v1 |] in
      let s2 = [| Symbolic_shape.dim_of_var v2 |] in
      assert (not (Symbolic_shape.equal s1 s2))
      (* Different identities *)
    ]}

    Performs structural comparison without evaluation; expressions that evaluate
    to the same value may not be equal. For example, [Add (Const 1, Const 2)] is
    not equal to [Const 3].

    Time complexity: O(n*m) where n is shape size and m is average expression
    tree depth. *)
