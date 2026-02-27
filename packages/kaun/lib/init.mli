(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Weight initialization strategies.

    Initializers map a shape and float dtype to tensors. Random keys are
    obtained implicitly via {!Nx.Rng.next_key}. Named families (Glorot, He,
    LeCun) are defined in terms of {!variance_scaling}. *)

(** {1:types Types} *)

type t = {
  f : 'layout. int array -> (float, 'layout) Nx.dtype -> (float, 'layout) Nx.t;
}
(** [t] is the type for initializers.

    [i.f shape dtype] is an initialized tensor for [shape] and [dtype]. Random
    keys are drawn from the implicit RNG scope. *)

(** {1:constant Constant} *)

val zeros : t
(** [zeros] is the initializer that fills with [0.0]. *)

val ones : t
(** [ones] is the initializer that fills with [1.0]. *)

val constant : float -> t
(** [constant v] is the initializer that fills with [v]. *)

(** {1:random Random} *)

val uniform : ?scale:float -> unit -> t
(** [uniform ?scale ()] is the initializer that samples from [U(0, scale)].

    [scale] defaults to [0.01].

    Raises [Invalid_argument] if [scale] is negative. *)

val normal : ?stddev:float -> unit -> t
(** [normal ?stddev ()] is the initializer that samples from [N(0, stddev)].

    [stddev] defaults to [0.01].

    Raises [Invalid_argument] if [stddev] is negative. *)

(** {1:variance Variance Scaling} *)

type mode = [ `Fan_in | `Fan_out | `Fan_avg ]
(** The type for variance-scaling fan modes. *)

type distribution = [ `Normal | `Truncated_normal | `Uniform ]
(** The type for variance-scaling distribution families. *)

val variance_scaling :
  scale:float ->
  mode:mode ->
  distribution:distribution ->
  ?in_axis:int ->
  ?out_axis:int ->
  unit ->
  t
(** [variance_scaling ~scale ~mode ~distribution ?in_axis ?out_axis ()] is the
    variance-scaling initializer.

    [in_axis] defaults to [-2] and [out_axis] defaults to [-1]. Negative axes
    are interpreted from the end.

    The target variance is [scale / n], with:
    - [n = fan_in] for [`Fan_in].
    - [n = fan_out] for [`Fan_out].
    - [n = (fan_in + fan_out) / 2] for [`Fan_avg].

    Distributions are:
    - [`Normal]: [N(0, scale / n)].
    - [`Uniform]: [U(-limit, limit)] with [limit = sqrt (3 * scale / n)].
    - [`Truncated_normal]: normal samples truncated to \[[-2];[2]\] and rescaled
      to match [scale / n].

    Raises [Invalid_argument] if:
    - [scale] is negative.
    - [in_axis] or [out_axis] is out of bounds for rank > 1.
    - the computed fan is non-positive. *)

(** {1:glorot Glorot/Xavier} *)

val glorot_uniform : ?in_axis:int -> ?out_axis:int -> unit -> t
(** [glorot_uniform ?in_axis ?out_axis ()] is Glorot/Xavier uniform
    initialization.

    It samples from [U(-limit, limit)] with
    [limit = sqrt (6 / (fan_in + fan_out))].

    This is the Xavier/Glorot scheme of Glorot and Bengio (2010). It is
    implemented via {!variance_scaling} with fan-average mode.

    Raises [Invalid_argument] under the same conditions as {!variance_scaling}.
*)

val glorot_normal : ?in_axis:int -> ?out_axis:int -> unit -> t
(** [glorot_normal ?in_axis ?out_axis ()] is Glorot/Xavier normal
    initialization.

    It uses truncated normal sampling with fan-average target variance
    [2 / (fan_in + fan_out)].

    This is the Xavier/Glorot family of Glorot and Bengio (2010). It is
    implemented via {!variance_scaling}.

    Raises [Invalid_argument] under the same conditions as {!variance_scaling}.
*)

(** {1:he He/Kaiming} *)

val he_uniform : ?in_axis:int -> ?out_axis:int -> unit -> t
(** [he_uniform ?in_axis ?out_axis ()] is He/Kaiming uniform initialization.

    It samples from [U(-limit, limit)] with [limit = sqrt (6 / fan_in)].

    This is the Kaiming/He scheme of He et al. (2015), commonly used for
    ReLU-like activations. It is implemented via {!variance_scaling} in fan-in
    mode.

    Raises [Invalid_argument] under the same conditions as {!variance_scaling}.
*)

val he_normal : ?in_axis:int -> ?out_axis:int -> unit -> t
(** [he_normal ?in_axis ?out_axis ()] is He/Kaiming normal initialization.

    It uses truncated normal sampling with fan-in target variance [2 / fan_in].

    This is the Kaiming/He family of He et al. (2015). It is implemented via
    {!variance_scaling}.

    Raises [Invalid_argument] under the same conditions as {!variance_scaling}.
*)

(** {1:lecun LeCun} *)

val lecun_uniform : ?in_axis:int -> ?out_axis:int -> unit -> t
(** [lecun_uniform ?in_axis ?out_axis ()] is LeCun uniform initialization.

    It samples from [U(-limit, limit)] with [limit = sqrt (3 / fan_in)].

    This is the LeCun fan-in family (Efficient BackProp, LeCun et al., 1998). It
    is implemented via {!variance_scaling}.

    Raises [Invalid_argument] under the same conditions as {!variance_scaling}.
*)

val lecun_normal : ?in_axis:int -> ?out_axis:int -> unit -> t
(** [lecun_normal ?in_axis ?out_axis ()] is LeCun normal initialization.

    It uses truncated normal sampling with fan-in target variance [1 / fan_in].

    This is the LeCun fan-in family (Efficient BackProp, LeCun et al., 1998). It
    is implemented via {!variance_scaling}.

    Raises [Invalid_argument] under the same conditions as {!variance_scaling}.
*)
