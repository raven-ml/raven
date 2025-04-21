open Descriptor
open Buffer

module type S = sig
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

  val nonzero :
    context -> ('a, 'b) b_t -> (int64, Bigarray.int64_elt) b_t -> unit

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
