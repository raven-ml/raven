(** Core types and low-level operations for nx. *)

(** {2 Core Nx Types} *)

type ('a, 'b) buffer = ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
(** The underlying 1D data storage (Bigarray Array1 with C layout). *)

type float16_elt = Bigarray.float16_elt
type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt
type int8_elt = Bigarray.int8_signed_elt
type uint8_elt = Bigarray.int8_unsigned_elt
type int16_elt = Bigarray.int16_signed_elt
type uint16_elt = Bigarray.int16_unsigned_elt
type int32_elt = Bigarray.int32_elt
type int64_elt = Bigarray.int64_elt
type complex32_elt = Bigarray.complex32_elt
type complex64_elt = Bigarray.complex64_elt

(** GADT representing supported element data types. *)
type ('a, 'b) dtype =
  | Float16 : (float, float16_elt) dtype  (** 16-bit float. *)
  | Float32 : (float, float32_elt) dtype  (** 32-bit float. *)
  | Float64 : (float, float64_elt) dtype  (** 64-bit float. *)
  | Int8 : (int, int8_elt) dtype  (** Signed 8-bit integer. *)
  | Int16 : (int, int16_elt) dtype  (** Signed 16-bit integer. *)
  | Int32 : (int32, int32_elt) dtype  (** Signed 32-bit integer. *)
  | Int64 : (int64, int64_elt) dtype  (** Signed 64-bit integer. *)
  | UInt8 : (int, uint8_elt) dtype  (** Unsigned 8-bit integer. *)
  | UInt16 : (int, uint16_elt) dtype  (** Unsigned 16-bit integer. *)
  | Complex32 : (Complex.t, complex32_elt) dtype  (** 32-bit complex float. *)
  | Complex64 : (Complex.t, complex64_elt) dtype  (** 64-bit complex float. *)

(** Memory layout specification hint. *)
type layout =
  | C_contiguous  (** Row-major contiguous layout hint. *)
  | Strided  (** Non-contiguous or unknown layout hint. *)

type ('a, 'b) descriptor = {
  dtype : ('a, 'b) dtype;  (** Element data type. *)
  shape : int array;  (** Dimensions of the array view. *)
  layout : layout;  (** Memory layout hint. *)
  strides : int array;  (** Step size (in elements) for each dimension. *)
  offset : int;  (** Start index (in elements) within the buffer. *)
}
(** Metadata describing an n-dimensional array view over a buffer. *)

(** {2 Parallel Pool} *)

module Parallel : sig
  type task = { start_idx : int; end_idx : int; compute : int -> int -> unit }
  type pool

  val get_or_setup_pool : unit -> pool
  val get_num_domains : pool -> int
  val parallel_for : pool -> int -> int -> (int -> int -> unit) -> unit

  val parallel_for_reduce :
    pool -> int -> int -> (int -> int -> 'a) -> ('a -> 'a -> 'a) -> 'a -> 'a
end

(** {2 DType Constants and Values} *)

val float16 : (float, float16_elt) dtype
val float32 : (float, float32_elt) dtype
val float64 : (float, float64_elt) dtype
val int8 : (int, int8_elt) dtype
val uint8 : (int, uint8_elt) dtype
val int16 : (int, int16_elt) dtype
val uint16 : (int, uint16_elt) dtype
val int32 : (int32, int32_elt) dtype
val int64 : (int64, int64_elt) dtype
val complex32 : (Complex.t, complex32_elt) dtype
val complex64 : (Complex.t, complex64_elt) dtype

val zero : ('a, 'b) dtype -> 'a
(** Returns the zero value for the given dtype. *)

val one : ('a, 'b) dtype -> 'a
(** Returns the one value for the given dtype. *)

(** {2 Buffer Operations} *)

val kind_of_dtype : ('a, 'b) dtype -> ('a, 'b) Bigarray.kind
(** Get the Bigarray kind associated with a dtype. *)

val dtype_of_kind : ('a, 'b) Bigarray.kind -> ('a, 'b) dtype
(** Get the dtype associated with a Bigarray kind. *)

val create_buffer : ('a, 'b) dtype -> int -> ('a, 'b) buffer
(** Create a new, uninitialized buffer of the specified dtype and size. *)

val fill : 'a -> ('a, 'b) buffer -> unit
(** Fill a buffer entirely with a given value. *)

val blit : ('a, 'b) buffer -> ('a, 'b) buffer -> unit
(** Copy the entire contents of the first buffer to the second buffer. Assumes
    same size. *)

val length : ('a, 'b) buffer -> int
(** Get the total number of elements in the buffer. *)

val size_in_bytes : ('a, 'b) buffer -> int
(** Get the total size of the buffer in bytes. *)

val of_array : ('a, 'b) dtype -> 'a array -> ('a, 'b) buffer
(** Create a buffer by copying data from an OCaml array. *)

(** {2 Properties} *)

val shape : ('a, 'b) descriptor -> int array
(** Get the shape (dimensions) of the array view. *)

val dtype : ('a, 'b) descriptor -> ('a, 'b) dtype
(** Get the data type (dtype) of the array view. *)

val strides : ('a, 'b) descriptor -> int array
(** Get the strides (in elements) for each dimension. *)

val offset : ('a, 'b) descriptor -> int
(** Get the offset (in elements) from the start of the underlying buffer. *)

val layout : ('a, 'b) descriptor -> layout
(** Get the layout hint (C_contiguous or Strided). *)

val size : ('a, 'b) descriptor -> int
(** Get the total number of elements described by the view (product of shape).
*)

val stride : int -> ('a, 'b) descriptor -> int
(** Get the stride (in elements) for a specific dimension index `i`. *)

val dims : ('a, 'b) descriptor -> int array
(** Alias for [shape]. *)

val dim : int -> ('a, 'b) descriptor -> int
(** Get the size of a specific dimension index `i`. *)

val ndim : ('a, 'b) descriptor -> int
(** Get the number of dimensions (rank) of the array view. *)

val itemsize : ('a, 'b) descriptor -> int
(** Get the size (in bytes) of a single element based on the dtype. *)

val nbytes : ('a, 'b) descriptor -> int
(** Get the total size (in bytes) of the array view. This is the product of
    [size] and [itemsize]. *)

(** {2 Shape Manipulation & Views} *)

val broadcast_shapes : int array -> int array -> int array
(** Calculate the resulting shape after broadcasting two shapes. Raises if
    incompatible. *)

val broadcast_to : ('a, 'b) descriptor -> int array -> ('a, 'b) descriptor
(** Create a new descriptor by broadcasting to a target shape (no data copy). *)

val squeeze : ?axes:int array -> ('a, 'b) descriptor -> ('a, 'b) descriptor
(** Create a new descriptor by removing dimensions of size 1 (no data copy). *)

val expand_dims : int -> ('a, 'b) descriptor -> ('a, 'b) descriptor
(** Create a new descriptor by adding a dimension of size 1 at an axis (no data
    copy). *)

val slice :
  ?steps:int array ->
  int array ->
  int array ->
  ('a, 'b) descriptor ->
  ('a, 'b) descriptor
(** Create a new descriptor representing a slice (no data copy). Args: [?steps]
    [starts] [lengths] [descriptor]. *)

val transpose : ?axes:int array -> ('a, 'b) descriptor -> ('a, 'b) descriptor
(** Create a new descriptor by transposing the dimensions (no data copy). *)

(** {2 Conversions} *)

val astype :
  ('c, 'd) dtype -> ('a, 'b) descriptor -> ('a, 'b) buffer -> ('c, 'd) buffer
(** Creates a new buffer by casting elements of the source buffer/descriptor
    view. Copies data. *)

(** {2 Helpers & Checks} *)

val pp_int_array : Format.formatter -> int array -> unit
(** Pretty-printer for integer arrays. *)

val compute_c_strides : int array -> int array
(** Calculate strides (in elements) for a C-contiguous layout given a shape. *)

val check_c_contiguity_from_shape_strides : int array -> int array -> bool
(** Check if a given shape and strides correspond to C-contiguity. *)

val is_c_contiguous : ('a, 'b) descriptor -> bool
(** Check if the descriptor represents a C-contiguous view based on shape and
    strides. *)

val is_scalar : ('a, 'b) descriptor -> bool
(** Check if the descriptor represents a scalar (0 dimensions). *)

val linear_to_md_c_contig : int -> int array -> int array
(** Convert a linear index to multi-dimensional indices for a C-contiguous
    layout. *)

val md_to_linear : int array -> int array -> int
(** Convert multi-dimensional indices to a linear offset using given strides. *)

val compute_broadcast_index : int array -> int array -> int array
(** Compute the broadcasted index for two shapes. *)

val multi_index_from_linear : int -> int array -> int array
(** Convert a linear index to multi-dimensional indices for a given shape. *)

val iter_multi_indices : int array -> (int array -> unit) -> unit
(** Iterate over all multi-dimensional indices for a given shape. *)

(** {2 Dtype Operations} *)

val add_dtype : ('a, 'b) dtype -> 'a -> 'a -> 'a
(** Add two elements of the same dtype. *)

val sub_dtype : ('a, 'b) dtype -> 'a -> 'a -> 'a
(** Subtract two elements of the same dtype. *)

val mul_dtype : ('a, 'b) dtype -> 'a -> 'a -> 'a
(** Multiply two elements of the same dtype. *)

val div_dtype : ('a, 'b) dtype -> 'a -> 'a -> 'a
(** Divide two elements of the same dtype. *)

val fma_dtype : ('a, 'b) dtype -> 'a -> 'a -> 'a -> 'a
(** Fused multiply-add for two elements of the same dtype. *)

(** {2 Backends} *)

(** Backend interface for nx operations. *)
module type Backend_intf = sig
  type ('a, 'b) b_t
  type context

  val create_context : unit -> context
  val descriptor : ('a, 'b) b_t -> ('a, 'b) descriptor
  val buffer : ('a, 'b) b_t -> ('a, 'b) buffer

  val from_buffer :
    context -> ('a, 'b) descriptor -> ('a, 'b) buffer -> ('a, 'b) b_t

  val view : ('a, 'b) descriptor -> ('a, 'b) b_t -> ('a, 'b) b_t
  val empty : context -> ('a, 'b) dtype -> int array -> ('a, 'b) b_t
  val copy : context -> ('a, 'b) b_t -> ('a, 'b) b_t
  val blit : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val fill : context -> 'a -> ('a, 'b) b_t -> unit

  (** Element‑wise arithmetic **)

  val add : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val sub : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val mul : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val div : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val pow : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val rem : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit

  val fma :
    context ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    unit

  (** Element‑wise bitwise operations **)

  val bit_and :
    context -> (int, 'b) b_t -> (int, 'b) b_t -> (int, 'b) b_t -> unit

  val bit_or :
    context -> (int, 'b) b_t -> (int, 'b) b_t -> (int, 'b) b_t -> unit

  val bit_xor :
    context -> (int, 'b) b_t -> (int, 'b) b_t -> (int, 'b) b_t -> unit

  val bit_not : context -> (int, 'b) b_t -> (int, 'b) b_t -> unit

  (** Comparisons (produce a uint8 mask) **)

  val equal :
    context ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    (int, Bigarray.int8_unsigned_elt) b_t ->
    unit

  val greater :
    context ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    (int, Bigarray.int8_unsigned_elt) b_t ->
    unit

  val greater_equal :
    context ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    (int, Bigarray.int8_unsigned_elt) b_t ->
    unit

  val less :
    context ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    (int, Bigarray.int8_unsigned_elt) b_t ->
    unit

  val less_equal :
    context ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    (int, Bigarray.int8_unsigned_elt) b_t ->
    unit

  (** Unary numeric / transcendental **)

  val neg : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val abs : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val sign : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val sqrt : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val exp : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val log : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val sin : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val cos : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val tan : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val asin : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val acos : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val atan : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val sinh : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val cosh : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val tanh : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val asinh : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val acosh : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
  val atanh : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit

  (** Essential non‑trivial numerics **)

  val floor : context -> (float, 'b) b_t -> (float, 'b) b_t -> unit
  val ceil : context -> (float, 'b) b_t -> (float, 'b) b_t -> unit
  val round : context -> (float, 'b) b_t -> (float, 'b) b_t -> unit

  val isnan :
    context -> (float, 'b) b_t -> (int, Bigarray.int8_unsigned_elt) b_t -> unit

  val isinf :
    context -> (float, 'b) b_t -> (int, Bigarray.int8_unsigned_elt) b_t -> unit

  val isfinite :
    context -> (float, 'b) b_t -> (int, Bigarray.int8_unsigned_elt) b_t -> unit

  (** Masked‐select (“where”) & index‐of‐nonzero **)

  val where :
    context ->
    (int, Bigarray.int8_unsigned_elt) b_t ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    unit

  (** Sorting & selection kernels **)

  val sort : context -> axis:int -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit

  val argsort :
    context ->
    axis:int ->
    ('a, 'b) b_t ->
    (int64, Bigarray.int64_elt) b_t ->
    unit

  val argmax :
    context ->
    axis:int ->
    ('a, 'b) b_t ->
    (int64, Bigarray.int64_elt) b_t ->
    unit

  val argmin :
    context ->
    axis:int ->
    ('a, 'b) b_t ->
    (int64, Bigarray.int64_elt) b_t ->
    unit

  (** Reductions along axes **)

  val sum :
    context ->
    axes:int array ->
    keepdims:bool ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    unit

  val prod :
    context ->
    axes:int array ->
    keepdims:bool ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    unit

  val max :
    context ->
    axes:int array ->
    keepdims:bool ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    unit

  val min :
    context ->
    axes:int array ->
    keepdims:bool ->
    ('a, 'b) b_t ->
    ('a, 'b) b_t ->
    unit

  (** Core linear‐algebra primitives **)

  val matmul : context -> ('a, 'b) b_t -> ('a, 'b) b_t -> ('a, 'b) b_t -> unit
end

module Make : functor (B : Backend_intf) -> sig
  type ('a, 'b) t = ('a, 'b) B.b_t
  type context = B.context

  (* Low-level operations *)

  val create_context : unit -> context
  val descriptor : ('a, 'b) t -> ('a, 'b) descriptor
  val buffer : ('a, 'b) t -> ('a, 'b) buffer
  val view : ('a, 'b) descriptor -> ('a, 'b) t -> ('a, 'b) t

  val from_buffer :
    context -> ('a, 'b) descriptor -> ('a, 'b) buffer -> ('a, 'b) t

  (** {1 Creating Nxs} *)

  val create : context -> ('a, 'b) dtype -> int array -> 'a array -> ('a, 'b) t

  val init :
    context -> ('a, 'b) dtype -> int array -> (int array -> 'a) -> ('a, 'b) t

  val scalar : context -> ('a, 'b) dtype -> 'a -> ('a, 'b) t
  val copy : context -> ('a, 'b) t -> ('a, 'b) t
  val fill : context -> 'a -> ('a, 'b) t -> unit
  val blit : context -> ('a, 'b) t -> ('a, 'b) t -> unit
  val full : context -> ('a, 'b) dtype -> int array -> 'a -> ('a, 'b) t
  val full_like : context -> 'a -> ('a, 'b) t -> ('a, 'b) t
  val empty : context -> ('a, 'b) dtype -> int array -> ('a, 'b) t
  val empty_like : context -> ('a, 'b) t -> ('a, 'b) t
  val zeros : context -> ('a, 'b) dtype -> int array -> ('a, 'b) t
  val zeros_like : context -> ('a, 'b) t -> ('a, 'b) t
  val ones : context -> ('a, 'b) dtype -> int array -> ('a, 'b) t
  val ones_like : context -> ('a, 'b) t -> ('a, 'b) t
  val identity : context -> ('a, 'b) dtype -> int -> ('a, 'b) t
  val eye : context -> ?m:int -> ?k:int -> ('a, 'b) dtype -> int -> ('a, 'b) t
  val arange : context -> ('a, 'b) dtype -> int -> int -> int -> ('a, 'b) t

  val arange_f :
    context -> (float, 'b) dtype -> float -> float -> float -> (float, 'b) t

  val linspace :
    context ->
    ('a, 'b) dtype ->
    ?endpoint:bool ->
    float ->
    float ->
    int ->
    ('a, 'b) t

  val logspace :
    context ->
    (float, 'b) dtype ->
    ?endpoint:bool ->
    ?base:float ->
    float ->
    float ->
    int ->
    (float, 'b) t

  val geomspace :
    context ->
    (float, 'b) dtype ->
    ?endpoint:bool ->
    float ->
    float ->
    int ->
    (float, 'b) t

  (** {1 Array Properties} *)

  val data : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
  val ndim : ('a, 'b) t -> int
  val shape : ('a, 'b) t -> int array
  val dim : int -> ('a, 'b) t -> int
  val dims : ('a, 'b) t -> int array
  val dtype : ('a, 'b) t -> ('a, 'b) dtype
  val nbytes : ('a, 'b) t -> int
  val size : ('a, 'b) t -> int
  val stride : int -> ('a, 'b) t -> int
  val strides : ('a, 'b) t -> int array
  val itemsize : ('a, 'b) t -> int
  val offset : ('a, 'b) t -> int
  val layout : ('a, 'b) t -> layout

  (** {1 Element Access and Views} *)

  val get_item : context -> int array -> ('a, 'b) t -> 'a
  val set_item : context -> int array -> 'a -> ('a, 'b) t -> unit
  val get : context -> int array -> ('a, 'b) t -> ('a, 'b) t
  val set : context -> int array -> ('a, 'b) t -> ('a, 'b) t -> unit

  val slice :
    context ->
    ?steps:int array ->
    int array ->
    int array ->
    ('a, 'b) t ->
    ('a, 'b) t

  val set_slice :
    context ->
    ?steps:int array ->
    int array ->
    int array ->
    ('a, 'b) t ->
    ('a, 'b) t ->
    unit

  (** {1 Array Manipulation} *)

  val flatten : context -> ('a, 'b) t -> ('a, 'b) t
  val ravel : context -> ('a, 'b) t -> ('a, 'b) t
  val reshape : context -> int array -> ('a, 'b) t -> ('a, 'b) t
  val transpose : context -> ?axes:int array -> ('a, 'b) t -> ('a, 'b) t
  val squeeze : context -> ?axes:int array -> ('a, 'b) t -> ('a, 'b) t

  (* *)

  val split : context -> ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t list
  val array_split : context -> ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t list

  (* *)

  val concatenate : context -> ?axis:int -> ('a, 'b) t list -> ('a, 'b) t
  val stack : context -> ?axis:int -> ('a, 'b) t list -> ('a, 'b) t
  val vstack : context -> ('a, 'b) t list -> ('a, 'b) t
  val hstack : context -> ('a, 'b) t list -> ('a, 'b) t
  val dstack : context -> ('a, 'b) t list -> ('a, 'b) t

  (* *)

  val pad : context -> (int * int) array -> 'a -> ('a, 'b) t -> ('a, 'b) t
  val expand_dims : context -> int -> ('a, 'b) t -> ('a, 'b) t
  val broadcast_to : context -> int array -> ('a, 'b) t -> ('a, 'b) t
  val broadcast_arrays : context -> ('a, 'b) t list -> ('a, 'b) t list

  (* *)

  val tile : context -> int array -> ('a, 'b) t -> ('a, 'b) t
  val repeat : context -> ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t

  (* *)

  val flip : context -> ?axes:int array -> ('a, 'b) t -> ('a, 'b) t
  val roll : context -> ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t

  (* *)

  val moveaxis : context -> int -> int -> ('a, 'b) t -> ('a, 'b) t
  val swapaxes : context -> int -> int -> ('a, 'b) t -> ('a, 'b) t

  (** {1 Conversion} *)

  val of_bigarray :
    context -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t -> ('a, 'b) t

  val to_bigarray :
    context -> ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t

  val to_array : context -> ('a, 'b) t -> 'a array
  val astype : context -> ('c, 'd) dtype -> ('a, 'b) t -> ('c, 'd) t

  (** {1 Arithmetic and Element-wise Operations} *)

  val add : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val add_inplace : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val add_scalar : context -> ('a, 'b) t -> 'a -> ('a, 'b) t
  val sub : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val sub_inplace : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val sub_scalar : context -> ('a, 'b) t -> 'a -> ('a, 'b) t
  val mul : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val mul_inplace : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val mul_scalar : context -> ('a, 'b) t -> 'a -> ('a, 'b) t
  val div : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val div_inplace : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val div_scalar : context -> ('a, 'b) t -> 'a -> ('a, 'b) t
  val rem : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val rem_inplace : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val rem_scalar : context -> ('a, 'b) t -> 'a -> ('a, 'b) t
  val pow : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val pow_inplace : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val pow_scalar : context -> ('a, 'b) t -> 'a -> ('a, 'b) t
  val maximum : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val maximum_inplace : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val maximum_scalar : context -> ('a, 'b) t -> 'a -> ('a, 'b) t
  val minimum : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val minimum_inplace : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val minimum_scalar : context -> ('a, 'b) t -> 'a -> ('a, 'b) t
  val fma : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t

  val fma_inplace :
    context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t

  (** {2 Unary Mathematical Functions} *)

  val exp : context -> ('a, 'b) t -> ('a, 'b) t
  val log : context -> ('a, 'b) t -> ('a, 'b) t
  val abs : context -> ('a, 'b) t -> ('a, 'b) t
  val neg : context -> ('a, 'b) t -> ('a, 'b) t
  val sign : context -> ('a, 'b) t -> ('a, 'b) t
  val sqrt : context -> (float, 'b) t -> (float, 'b) t
  val square : context -> ('a, 'b) t -> ('a, 'b) t
  val sin : context -> (float, 'b) t -> (float, 'b) t
  val cos : context -> (float, 'b) t -> (float, 'b) t
  val tan : context -> (float, 'b) t -> (float, 'b) t
  val asin : context -> (float, 'b) t -> (float, 'b) t
  val acos : context -> (float, 'b) t -> (float, 'b) t
  val atan : context -> (float, 'b) t -> (float, 'b) t
  val sinh : context -> (float, 'b) t -> (float, 'b) t
  val cosh : context -> (float, 'b) t -> (float, 'b) t
  val tanh : context -> (float, 'b) t -> (float, 'b) t
  val asinh : context -> (float, 'b) t -> (float, 'b) t
  val acosh : context -> (float, 'b) t -> (float, 'b) t
  val atanh : context -> (float, 'b) t -> (float, 'b) t

  (* *)

  val round : context -> (float, 'b) t -> (float, 'b) t
  val floor : context -> (float, 'b) t -> (float, 'b) t
  val ceil : context -> (float, 'b) t -> (float, 'b) t

  (* *)

  val clip : context -> min:'a -> max:'a -> ('a, 'b) t -> ('a, 'b) t

  (** {1 Comparison Operations} *)

  val equal : context -> ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
  val greater : context -> ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
  val greater_equal : context -> ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
  val less : context -> ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
  val less_equal : context -> ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t

  (** {1 Bitwise Operations} *)

  val bitwise_and : context -> (int, 'b) t -> (int, 'b) t -> (int, 'b) t
  val bitwise_or : context -> (int, 'b) t -> (int, 'b) t -> (int, 'b) t
  val bitwise_xor : context -> (int, 'b) t -> (int, 'b) t -> (int, 'b) t
  val invert : context -> (int, 'b) t -> (int, 'b) t

  (** {1 Reductions and Statistical Functions} *)

  val sum :
    context -> ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t

  val prod :
    context -> ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t

  val max :
    context -> ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t

  val min :
    context -> ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t

  (** {1 Statistics} *)

  val mean :
    context ->
    ?axes:int array ->
    ?keepdims:bool ->
    (float, 'b) t ->
    (float, 'b) t

  val var :
    context ->
    ?axes:int array ->
    ?keepdims:bool ->
    (float, 'b) t ->
    (float, 'b) t

  val std :
    context ->
    ?axes:int array ->
    ?keepdims:bool ->
    (float, 'b) t ->
    (float, 'b) t

  (** {1 Linear Algebra and Matrix Operations} *)

  val dot : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val matmul : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t

  (* *)

  val convolve1d :
    context ->
    ?mode:[ `Full | `Valid | `Same ] ->
    ('a, 'b) t ->
    ('a, 'b) t ->
    ('a, 'b) t

  (* *)

  val inv : context -> (float, 'b) t -> (float, 'b) t
  val solve : context -> (float, 'b) t -> (float, 'b) t -> (float, 'b) t

  val svd :
    context -> (float, 'b) t -> (float, 'b) t * (float, 'b) t * (float, 'b) t

  val eig : context -> (float, 'b) t -> (float, 'b) t * (float, 'b) t
  val eigh : context -> (float, 'b) t -> (float, 'b) t * (float, 'b) t

  (** {1 Sorting, Searching, and Unique} *)

  val where :
    context -> (int, uint8_elt) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t

  val sort : context -> ?axis:int -> ('a, 'b) t -> ('a, 'b) t
  val argsort : context -> ?axis:int -> ('a, 'b) t -> (int64, int64_elt) t
  val argmax : context -> ?axis:int -> ('a, 'b) t -> (int64, int64_elt) t
  val argmin : context -> ?axis:int -> ('a, 'b) t -> (int64, int64_elt) t

  (* *)

  val unique : context -> ('a, 'b) t -> ('a, 'b) t
  val nonzero : context -> ('a, 'b) t -> (int64, int64_elt) t

  (** {1 Random Sampling and Distributions} *)

  val rand :
    context -> (float, 'b) dtype -> ?seed:int -> int array -> (float, 'b) t

  val randn :
    context -> (float, 'b) dtype -> ?seed:int -> int array -> (float, 'b) t

  val randint :
    context ->
    ('a, 'b) dtype ->
    ?seed:int ->
    ?high:int ->
    int array ->
    int ->
    ('a, 'b) t

  (** {1 Logical and Masking Operations} *)

  val logical_and :
    context -> (int, uint8_elt) t -> (int, uint8_elt) t -> (int, uint8_elt) t

  val logical_or :
    context -> (int, uint8_elt) t -> (int, uint8_elt) t -> (int, uint8_elt) t

  val logical_not : context -> (int, uint8_elt) t -> (int, uint8_elt) t

  val logical_xor :
    context -> (int, uint8_elt) t -> (int, uint8_elt) t -> (int, uint8_elt) t

  val isnan : context -> (float, 'b) t -> (int, uint8_elt) t
  val isinf : context -> (float, 'b) t -> (int, uint8_elt) t
  val isfinite : context -> (float, 'b) t -> (int, uint8_elt) t
  val array_equal : context -> ('a, 'b) t -> ('a, 'b) t -> bool

  (** {1 Functional and Higher‑order Operations} *)

  val map : context -> ('a -> 'a) -> ('a, 'b) t -> ('a, 'b) t
  val iter : context -> ('a -> unit) -> ('a, 'b) t -> unit
  val fold : context -> ('a -> 'a -> 'a) -> 'a -> ('a, 'b) t -> 'a

  (** {1 Utilities: Printing and Debugging} *)

  val pp_dtype : context -> Format.formatter -> ('a, 'b) dtype -> unit
  val dtype_to_string : context -> ('a, 'b) dtype -> string
  val pp_shape : context -> Format.formatter -> int array -> unit
  val shape_to_string : context -> int array -> string
  val pp : context -> Format.formatter -> ('a, 'b) t -> unit
  val to_string : context -> ('a, 'b) t -> string
  val print : context -> ('a, 'b) t -> unit
  val pp_info : context -> Format.formatter -> ('a, 'b) t -> unit
  val to_string_info : context -> ('a, 'b) t -> string
  val print_info : context -> ('a, 'b) t -> unit
end
