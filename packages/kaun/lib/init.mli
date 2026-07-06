(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Weight initializers.

    An initializer is a plain function from a layer's fan geometry, a float
    dtype, and a shape to a freshly drawn tensor. Layers accept one as an
    [?init] argument and supply [fan_in] and [fan_out] from their own geometry:

    {[
    let init ?(w_init = Init.glorot_uniform) ~inputs ~outputs () =
      let w =
        w_init ~fan_in:inputs ~fan_out:outputs Nx.float32 [| inputs; outputs |]
      in
      { w; b = Nx.zeros Nx.float32 [| outputs |] }
    ]}

    Random initializers draw from the implicit RNG scope of {!Nx.Rng}; wrap
    model construction in {!Nx.Rng.run} for reproducibility. The named families
    (Glorot/Xavier, He/Kaiming, LeCun) are instances of {!variance_scaling}. *)

(** {1:types Types} *)

type 'b t =
  fan_in:int ->
  fan_out:int ->
  (float, 'b) Nx.dtype ->
  int array ->
  (float, 'b) Nx.t
(** The type for initializers producing float tensors with layout ['b].

    [init ~fan_in ~fan_out dtype shape] is a fresh tensor of the given [dtype]
    and [shape]. [fan_in] and [fan_out] are the number of input and output
    connections per unit of the parameter being initialized; for a dense weight
    of shape [[| inputs; outputs |]] they are [inputs] and [outputs]. Constant
    initializers ignore the fans; variance-scaling initializers require both to
    be positive and scale by the fans alone, independently of [shape].

    Any function of this type is an initializer, so custom schemes can be passed
    wherever a [t] is expected. Random draws use the implicit RNG scope (see
    {!Nx.Rng}); each application consumes fresh randomness. *)

(** {1:constant Constant} *)

val zeros : 'b t
(** [zeros] fills with [0.0]. The fans are ignored. *)

val ones : 'b t
(** [ones] fills with [1.0]. The fans are ignored. *)

val constant : float -> 'b t
(** [constant v] fills with [v]. The fans are ignored. *)

(** {1:random Random} *)

val uniform : scale:float -> 'b t
(** [uniform ~scale] samples uniformly from \[[0];[scale]). The fans are
    ignored.

    Raises [Invalid_argument] if [scale] is negative. *)

val normal : stddev:float -> 'b t
(** [normal ~stddev] samples from [N(0, stddev²)]. The fans are ignored.

    Raises [Invalid_argument] if [stddev] is negative. *)

(** {1:variance Variance scaling} *)

type mode = [ `Fan_in | `Fan_out | `Fan_avg ]
(** The type for variance-scaling fan modes. *)

type distribution = [ `Normal | `Truncated_normal | `Uniform ]
(** The type for variance-scaling distribution families. *)

val variance_scaling :
  scale:float -> mode:mode -> distribution:distribution -> 'b t
(** [variance_scaling ~scale ~mode ~distribution] samples with target variance
    [scale / n], with:

    - [n = fan_in] for [`Fan_in].
    - [n = fan_out] for [`Fan_out].
    - [n = (fan_in + fan_out) / 2] for [`Fan_avg].

    Distributions are:

    - [`Normal]: [N(0, scale / n)].
    - [`Truncated_normal]: normal samples truncated to two standard deviations
      and rescaled so the result's variance is [scale / n].
    - [`Uniform]: [U(-limit, limit)] with [limit = sqrt (3 * scale / n)].

    Raises [Invalid_argument] if [scale] is negative, or, when the resulting
    initializer is applied, if [fan_in] or [fan_out] is not positive. *)

(** {1:glorot Glorot/Xavier} *)

val glorot_uniform : 'b t
(** [glorot_uniform] samples from [U(-limit, limit)] with
    [limit = sqrt (6 / (fan_in + fan_out))].

    This is the Xavier/Glorot scheme of Glorot and Bengio (2010), suited to
    activations that are roughly linear around zero (tanh, sigmoid). It is
    {!variance_scaling} with [~scale:1.0 ~mode:`Fan_avg ~distribution:`Uniform].

    Raises [Invalid_argument] as {!variance_scaling}. *)

val glorot_normal : 'b t
(** [glorot_normal] samples a truncated normal with target variance
    [2 / (fan_in + fan_out)].

    This is the Xavier/Glorot family of Glorot and Bengio (2010). It is
    {!variance_scaling} with
    [~scale:1.0 ~mode:`Fan_avg ~distribution:`Truncated_normal].

    Raises [Invalid_argument] as {!variance_scaling}. *)

(** {1:he He/Kaiming} *)

val he_uniform : 'b t
(** [he_uniform] samples from [U(-limit, limit)] with
    [limit = sqrt (6 / fan_in)].

    This is the Kaiming/He scheme of He et al. (2015), suited to ReLU-like
    activations. It is {!variance_scaling} with
    [~scale:2.0 ~mode:`Fan_in ~distribution:`Uniform].

    Raises [Invalid_argument] as {!variance_scaling}. *)

val he_normal : 'b t
(** [he_normal] samples a truncated normal with target variance [2 / fan_in].

    This is the Kaiming/He family of He et al. (2015). It is {!variance_scaling}
    with [~scale:2.0 ~mode:`Fan_in ~distribution:`Truncated_normal].

    Raises [Invalid_argument] as {!variance_scaling}. *)

(** {1:lecun LeCun} *)

val lecun_uniform : 'b t
(** [lecun_uniform] samples from [U(-limit, limit)] with
    [limit = sqrt (3 / fan_in)].

    This is the LeCun fan-in family (Efficient BackProp, LeCun et al., 1998),
    suited to SELU and other self-normalizing activations. It is
    {!variance_scaling} with [~scale:1.0 ~mode:`Fan_in ~distribution:`Uniform].

    Raises [Invalid_argument] as {!variance_scaling}. *)

val lecun_normal : 'b t
(** [lecun_normal] samples a truncated normal with target variance [1 / fan_in].

    This is the LeCun fan-in family (Efficient BackProp, LeCun et al., 1998). It
    is {!variance_scaling} with
    [~scale:1.0 ~mode:`Fan_in ~distribution:`Truncated_normal].

    Raises [Invalid_argument] as {!variance_scaling}. *)
