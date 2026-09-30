(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What computes on arrays.

    A backend is kernels over the arrays of one device: one function per
    operation nx computes, each reading its operands and writing its result into
    [dst], an array nx allocated on the same device, C-contiguous from its first
    element. nx places the operands, allocates the results and calls the backend
    once per device, so a backend never sees nx's values, a placement or another
    device. The kinds of nx's operations are here too: operations of one kind
    share their operands' and results' types, and the kind names the
    mathematical function, after the [Nx] function it implements. *)

(** {1:kinds Kinds} *)

(** The type for elementwise functions of one operand, of its dtype. [Trunc],
    [Ceil], [Floor] and [Round] (half away from zero) are the identity on
    integers. *)
type unary =
  | Neg
  | Recip
  | Abs
  | Sqrt
  | Sign
  | Exp
  | Log
  | Sin
  | Cos
  | Tan
  | Asin
  | Acos
  | Atan
  | Sinh
  | Cosh
  | Tanh
  | Trunc
  | Ceil
  | Floor
  | Round
  | Erf

(** The type for elementwise functions of two operands of one shape and dtype,
    of that dtype. [Fdiv] divides floats and [Idiv] integers, truncating: the
    surface picks one by dtype. [And], [Or] and [Xor] are bitwise. *)
type binary =
  | Add
  | Sub
  | Mul
  | Fdiv
  | Idiv
  | Mod
  | Pow
  | Atan2
  | Maximum
  | Minimum
  | And
  | Or
  | Xor

(** The type for elementwise comparisons of two operands of one shape and dtype,
    to booleans. *)
type compare = Equal | Not_equal | Less | Less_equal

(** The type for associative reductions, over axes ([Nx.sum]) or as running
    values along one axis ([Nx.cumsum]). *)
type reduce = Sum | Prod | Max | Min

(** The type for the position of an extreme along one axis, as [int32]. The
    first of equal extremes is taken. *)
type arg_reduce = Argmax | Argmin

(** The type for dtype conversions: [Cast] converts values, [Bitcast]
    reinterprets the bits of elements of the same width. *)
type conversion = Cast | Bitcast

(** {1:backends Backends} *)

exception Refused of string
(** [Refused reason] is raised by a kernel that does not run its arguments,
    before it writes anything. [reason] names the backend, the operation and
    what it refuses, as in ["counting: matmul: no float64"]. *)

exception
  Linalg_error of {
    op : string;
    kind : [ `Not_positive_definite | `Singular | `No_convergence ];
  }
(** [Linalg_error { op; kind }] is raised by a linear-algebra kernel whose
    computation fails on its values: [op] is the operation, as in ["cholesky"],
    and [kind] the failure: a matrix that is not positive-definite, a singular
    matrix, or an iteration that did not converge. A precondition on shapes or
    dtypes raises [Invalid_argument]. *)

(** The type for arrays, as kernels take them. *)

type int32_array = (int32, Nx_dtype.int32_elt) Nx_array.t
(** The type for arrays of indices. *)

(** The type for backend implementations. Every function but [name] and
    [runs_on] is a kernel: its operands and [dst] are arrays on one device the
    backend runs on, and it writes [dst] whole, or raises {!Refused} before it
    writes. Operands have the shapes and dtypes the operation's [Nx] function
    gives them; [dst] has the result's. *)
module type S = sig
  val name : string
  (** [name] is the backend's name, as placements print it. *)

  val runs_on : Nx_device.t -> bool
  (** [runs_on d] is [true] iff the kernels compute on arrays in [d]'s memory.
  *)

  val unary : unary -> ('a, 'b) Nx_array.t -> dst:('a, 'b) Nx_array.t -> unit

  val binary :
    binary ->
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val compare :
    compare ->
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:(bool, Nx_dtype.bool_elt) Nx_array.t ->
    unit

  val where :
    (bool, Nx_dtype.bool_elt) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val reduce :
    reduce ->
    axes:int array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [reduce k ~axes x ~dst] reduces [x] over [axes], which [dst] drops. *)

  val scan :
    reduce -> axis:int -> ('a, 'b) Nx_array.t -> dst:('a, 'b) Nx_array.t -> unit

  val arg_reduce :
    arg_reduce -> axis:int -> ('a, 'b) Nx_array.t -> dst:int32_array -> unit
  (** [arg_reduce k ~axis x ~dst] is the position of the extreme along [axis],
      which [dst] drops. *)

  val sort :
    descending:bool ->
    axis:int ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val argsort :
    descending:bool ->
    axis:int ->
    ('a, 'b) Nx_array.t ->
    dst:int32_array ->
    unit

  val pad :
    (int * int) array ->
    'a ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [pad padding v x ~dst] is [x] with [padding] elements of value [v] before
      and after each axis. *)

  val cat :
    axis:int -> ('a, 'b) Nx_array.t list -> dst:('a, 'b) Nx_array.t -> unit

  val cast : ('a, 'b) Nx_array.t -> dst:('c, 'd) Nx_array.t -> unit

  val threefry : int32_array -> int32_array -> dst:int32_array -> unit
  (** [threefry key counter ~dst] hashes [counter] under [key]. *)

  val gather :
    axis:int ->
    int32_array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [gather ~axis indices x ~dst] reads [x] along [axis] at [indices], of
      [dst]'s shape. *)

  val scatter :
    mode:[ `Set | `Add ] ->
    unique:bool ->
    axis:int ->
    indices:int32_array ->
    updates:('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [scatter ~mode ~unique ~axis ~indices ~updates x ~dst] is [x] with
      [updates] set or added along [axis] at [indices]. *)

  val update :
    ('a, 'b) Nx_array.t ->
    starts:int32_array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [update x ~starts v ~dst] is [x] with the window at [starts] set to [v].
  *)

  val unfold :
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val fold :
    output_size:int array ->
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val matmul :
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val fft :
    inverse:bool ->
    axes:int array ->
    (Complex.t, 'b) Nx_array.t ->
    dst:(Complex.t, 'b) Nx_array.t ->
    unit
  (** [fft ~inverse ~axes x ~dst] is the unnormalized transform over [axes]. *)

  val rfft :
    axes:int array ->
    (float, 'b) Nx_array.t ->
    dst:(Complex.t, 'c) Nx_array.t ->
    unit

  val irfft :
    axes:int array ->
    s:int array option ->
    (Complex.t, 'b) Nx_array.t ->
    dst:(float, 'c) Nx_array.t ->
    unit

  val contiguous : ('a, 'b) Nx_array.t -> dst:('a, 'b) Nx_array.t -> unit
  (** [contiguous x ~dst] copies [x]'s elements into [dst]. *)

  val cholesky :
    upper:bool -> ('a, 'b) Nx_array.t -> dst:('a, 'b) Nx_array.t -> unit

  val qr :
    reduced:bool ->
    ('a, 'b) Nx_array.t ->
    q:('a, 'b) Nx_array.t ->
    r:('a, 'b) Nx_array.t ->
    unit

  val lu :
    ('a, 'b) Nx_array.t ->
    lu:('a, 'b) Nx_array.t ->
    pivots:int32_array ->
    perm:int32_array ->
    unit

  val svd :
    ('a, 'b) Nx_array.t ->
    u:('a, 'b) Nx_array.t ->
    s:(float, Nx_dtype.float64_elt) Nx_array.t ->
    vt:('a, 'b) Nx_array.t ->
    unit
  (** [svd x ~u ~s ~vt] factors [x]; [u]'s and [vt]'s shapes say whether the
      factors are full or thin. *)

  val eig :
    ('a, 'b) Nx_array.t ->
    values:(Complex.t, Nx_dtype.complex64_elt) Nx_array.t ->
    vectors:(Complex.t, Nx_dtype.complex64_elt) Nx_array.t option ->
    unit

  val eigh :
    ('a, 'b) Nx_array.t ->
    values:(float, Nx_dtype.float64_elt) Nx_array.t ->
    vectors:('a, 'b) Nx_array.t option ->
    unit

  val solve_triangular :
    upper:bool ->
    transpose:bool ->
    unit_diag:bool ->
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
end

type t
(** The type for backends. *)

val make : (module S) -> t
(** [make k] is the backend of kernels [k]. Each [make] is a backend of its own:
    a library makes its backend once and shares the value. *)

val kernels : t -> (module S)
(** [kernels b] is [b]'s kernels, for nx to call. *)

val name : t -> string
(** [name b] is [b]'s name. *)

val runs_on : t -> Nx_device.t -> bool
(** [runs_on b d] is [true] iff [b] computes on arrays in [d]'s memory. *)

val equal : t -> t -> bool
(** [equal b b'] is [true] iff [b] and [b'] are the same {!make}. *)
