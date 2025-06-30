(** Flax-compatible weight initializers for Kaun

    This module provides weight initialization strategies matching the Flax/JAX
    neural network library API. All initializers return functions that take RNG
    seed, shape, device, and dtype parameters. *)

type ('layout, 'dev) t =
  int ->
  int array ->
  'dev Rune.device ->
  (float, 'layout) Rune.dtype ->
  (float, 'layout, 'dev) Rune.t
(** Type for initializer functions *)

(** {1 Basic Initializers} *)

val constant : float -> ('layout, 'dev) t
(** Constant value initializer *)

val zeros : unit -> ('layout, 'dev) t
(** Zero initializer *)

val ones : unit -> ('layout, 'dev) t
(** Ones initializer *)

val zeros_init : unit -> ('layout, 'dev) t
(** Alias for zeros *)

val ones_init : unit -> ('layout, 'dev) t
(** Alias for ones *)

(** {1 Random Initializers} *)

val uniform : ?scale:float -> unit -> ('layout, 'dev) t
(** Uniform random initializer in range [\[0, scale)] *)

val normal : ?stddev:float -> unit -> ('layout, 'dev) t
(** Normal (Gaussian) random initializer *)

val truncated_normal_init :
  ?stddev:float -> ?lower:float -> ?upper:float -> unit -> ('layout, 'dev) t
(** Truncated normal initializer *)

(** {1 Variance Scaling Initializers} *)

val variance_scaling :
  scale:float ->
  mode:[ `Fan_in | `Fan_out | `Fan_avg ] ->
  distribution:[ `Normal | `Truncated_normal | `Uniform ] ->
  in_axis:int ->
  out_axis:int ->
  unit ->
  ('layout, 'dev) t
(** General variance scaling initializer

    @param scale Scaling factor (positive float)
    @param mode One of [`Fan_in], [`Fan_out], [`Fan_avg]
    @param distribution One of [`Normal], [`Truncated_normal], [`Uniform]
    @param in_axis Axis of input dimension (default: -2)
    @param out_axis Axis of output dimension (default: -1) *)

(** {1 Xavier/Glorot Initializers} *)

val glorot_uniform : ?in_axis:int -> ?out_axis:int -> unit -> ('layout, 'dev) t
(** Glorot uniform initializer (aka Xavier uniform)

    Uses variance_scaling with scale=1.0, mode=`Fan_avg, distribution=`Uniform
*)

val glorot_normal : ?in_axis:int -> ?out_axis:int -> unit -> ('layout, 'dev) t
(** Glorot normal initializer (aka Xavier normal)

    Uses variance_scaling with scale=1.0, mode=`Fan_avg,
    distribution=`Truncated_normal *)

val xavier_uniform : ?in_axis:int -> ?out_axis:int -> unit -> ('layout, 'dev) t
(** Alias for glorot_uniform *)

val xavier_normal : ?in_axis:int -> ?out_axis:int -> unit -> ('layout, 'dev) t
(** Alias for glorot_normal *)

(** {1 LeCun Initializers} *)

val lecun_uniform : ?in_axis:int -> ?out_axis:int -> unit -> ('layout, 'dev) t
(** LeCun uniform initializer

    Uses variance_scaling with scale=1.0, mode=`Fan_in, distribution=`Uniform *)

val lecun_normal : ?in_axis:int -> ?out_axis:int -> unit -> ('layout, 'dev) t
(** LeCun normal initializer

    Uses variance_scaling with scale=1.0, mode=`Fan_in,
    distribution=`Truncated_normal *)

(** {1 He/Kaiming Initializers} *)

val he_uniform : ?in_axis:int -> ?out_axis:int -> unit -> ('layout, 'dev) t
(** He uniform initializer (aka Kaiming uniform)

    Uses variance_scaling with scale=2.0, mode=`Fan_in, distribution=`Uniform
    Designed for layers with ReLU activation *)

val he_normal : ?in_axis:int -> ?out_axis:int -> unit -> ('layout, 'dev) t
(** He normal initializer (aka Kaiming normal)

    Uses variance_scaling with scale=2.0, mode=`Fan_in,
    distribution=`Truncated_normal Designed for layers with ReLU activation *)

val kaiming_uniform : ?in_axis:int -> ?out_axis:int -> unit -> ('layout, 'dev) t
(** Alias for he_uniform *)

val kaiming_normal : ?in_axis:int -> ?out_axis:int -> unit -> ('layout, 'dev) t
(** Alias for he_normal *)

(** {1 Orthogonal Initializers} *)

val orthogonal : ?scale:float -> ?column_axis:int -> unit -> ('layout, 'dev) t
(** Orthogonal matrix initializer

    Returns uniformly distributed orthogonal matrices. If the shape is not
    square, the matrices will have orthonormal rows or columns depending on
    which side is smaller.

    @param scale Scaling factor (default: 1.0)
    @param column_axis
      Axis containing columns that should be orthogonal (default: -1) *)

val delta_orthogonal :
  ?scale:float -> ?column_axis:int -> unit -> ('layout, 'dev) t
(** Delta orthogonal initializer for convolutional layers

    Initializer for convolutional layers that preserves identity in the spatial
    dimensions. Requires 3D, 4D, or 5D tensor shape with square spatial
    dimensions.

    @param scale Scaling factor (default: 1.0)
    @param column_axis
      Axis containing columns that should be orthogonal (default: -1) *)

(** {1 Utility Initializers} *)

val uniform_range : low:float -> high:float -> unit -> ('layout, 'dev) t
(** Uniform initializer with explicit range *)

val normal_range : mean:float -> stddev:float -> unit -> ('layout, 'dev) t
(** Normal initializer with explicit mean and stddev *)
