(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Random number generation.

    Random tensors are produced by a counter-based generator: a per-device key
    tensor (derived from the process seed) and a counter tensor track a
    position in a fixed random stream, and each draw builds a graph that mixes
    the counter with the key ({!Elementwise.threefry}) and advances the
    counter in place. Because the state lives in device tensors and the
    advance is part of the graph, a captured computation ({!Jit}) that draws
    random numbers keeps drawing fresh values on every replay.

    Results are deterministic given a seed: after {!manual_seed}, the same
    sequence of calls produces the same values, on any device. Every dimension
    of a requested [shape] must be non-negative; [Invalid_argument] is raised
    otherwise. In this version {!rand} (and the samplers built on it) only
    generates 32-bit floats; {!val-rand} raises [Invalid_argument] for other
    float dtypes. *)

val manual_seed : int -> unit
(** [manual_seed seed] sets the process seed and resets all generator state,
    restarting the random stream. Without a call to [manual_seed] the seed is
    taken from the clock at startup. *)

(** {1 Uniform} *)

val rand :
  ?dtype:Tolk_uop.Dtype.t -> ?contiguous:bool -> int list -> Tensor.t
(** [rand shape] is a tensor of shape [shape] filled with uniform random
    values in [\[0, 1)]. [dtype] must be a 32-bit float type (default: the
    default float). [contiguous] (default [true]) materialises the result
    into its own buffer.

    @raise Invalid_argument if [dtype] is not a 32-bit float dtype. *)

val rand_like :
  ?dtype:Tolk_uop.Dtype.t -> ?contiguous:bool -> Tensor.t -> Tensor.t
(** [rand_like t] is {!val-rand} with the shape of [t] and, unless
    overridden, the dtype of [t]. *)

val uniform :
  ?low:float -> ?high:float -> ?dtype:Tolk_uop.Dtype.t -> int list ->
  Tensor.t
(** [uniform shape] is filled with uniform random values in
    [\[low, high)] (default [\[0, 1)]), cast to [dtype] (default: the default
    float) before the lower bound is added.

    @raise Invalid_argument unless [low < high]. *)

val randint :
  ?low:int -> ?high:int -> ?dtype:Tolk_uop.Dtype.t -> int list -> Tensor.t
(** [randint shape] is filled with uniform random integers in
    [\[low, high)] (default [\[0, 10)]). [dtype] must be an integer type
    (default [int32]).

    @raise Invalid_argument unless [low < high] and [dtype] is integer. *)

val scaled_uniform : ?dtype:Tolk_uop.Dtype.t -> int list -> Tensor.t
(** [scaled_uniform shape] is {!uniform} over [\[-1, 1)] scaled by
    [1/sqrt(numel)]. *)

val glorot_uniform : ?dtype:Tolk_uop.Dtype.t -> int list -> Tensor.t
(** [glorot_uniform shape] is the Glorot (Xavier) uniform initialisation:
    {!uniform} over [\[-b, b)] with
    [b = sqrt (6 / (shape.(0) + prod rest))]. *)

val kaiming_uniform :
  ?a:float -> ?dtype:Tolk_uop.Dtype.t -> int list -> Tensor.t
(** [kaiming_uniform shape] is the Kaiming (He) uniform initialisation for a
    leaky ReLU with negative slope [a] (default [0.01]): {!uniform} over
    [\[-b, b)] with [b = sqrt (6 / (1 + a^2) / prod (tl shape))]. *)

(** {1 Normal} *)

val randn : ?dtype:Tolk_uop.Dtype.t -> int list -> Tensor.t
(** [randn shape] is filled with standard normal random values (mean [0],
    variance [1]), sampled at 32-bit float precision and cast to [dtype]
    (default: the default float). *)

val randn_like : ?dtype:Tolk_uop.Dtype.t -> Tensor.t -> Tensor.t
(** [randn_like t] is {!randn} with the shape of [t] and, unless overridden,
    the dtype of [t]. *)

val normal :
  ?mean:float -> ?std:float -> ?dtype:Tolk_uop.Dtype.t -> int list ->
  Tensor.t
(** [normal shape] is filled with normal random values with the given [mean]
    (default [0]) and standard deviation [std] (default [1]).

    @raise Invalid_argument if [std] is negative. *)

val kaiming_normal :
  ?a:float -> ?dtype:Tolk_uop.Dtype.t -> int list -> Tensor.t
(** [kaiming_normal shape] is the Kaiming (He) normal initialisation for a
    leaky ReLU with negative slope [a] (default [0.01]): {!normal} with
    [std = sqrt (2 / (1 + a^2) / prod (tl shape))]. *)

(** {1 Sampling} *)

val randperm : ?dtype:Tolk_uop.Dtype.t -> int -> Tensor.t
(** [randperm n] is a uniformly random permutation of the integers
    [0 .. n-1], as a tensor of dtype [dtype] (default [int32]). *)

val multinomial : ?num_samples:int -> ?replacement:bool -> Tensor.t -> Tensor.t
(** [multinomial weights] draws [num_samples] (default [1]) indices from the
    categorical distribution proportional to [weights], which must be 1- or
    2-dimensional (a batch of rows). The result is [int32], with one index
    per sample (per row for a 2-D input, shaped [(rows, num_samples)]).
    Without [replacement] (the default) the samples within a row are
    distinct.

    @raise Invalid_argument
      if [weights] is not 1- or 2-dimensional, [num_samples] is not positive,
      or sampling without replacement asks for more samples than there are
      categories. *)

val dropout : ?p:float -> Tensor.t -> Tensor.t
(** [dropout t] zeroes each element of [t] independently with probability
    [p] (default [0.5]) and scales the survivors by [1/(1-p)]. Dropout is
    active only in training mode (the [TRAINING] context variable); otherwise
    [t] is returned unchanged.

    @raise Invalid_argument if [p] is outside [\[0, 1\]]. *)

(**/**)

(* Generator state, exposed for tests: the two-lane uint32 stream counter of
   a device, present once that device has drawn random numbers. *)
val device_rng_counter : string -> Tensor.t option

(**/**)
