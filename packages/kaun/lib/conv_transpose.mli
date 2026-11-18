(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Transposed convolution layers.

    A transposed convolution is the transpose of a convolution with respect to
    its input (Dumoulin and Visin, "A guide to convolution arithmetic for deep
    learning", 2016, §4), as {!Nx.conv_transpose} defines it: where a
    convolution maps [[| n; s…; out_channels |]] to [[| n; t…; in_channels |]],
    its transpose maps [[| n; t…; in_channels |]] to [[| n; s…; out_channels |]]
    by the same filters, each input element spreading over a window of the
    result, so that a stride above one upsamples. Stride, padding and dilation
    are the convolution's. Construct, apply and walk it as {!Conv}. A grouped
    transpose is {!Nx.conv_transpose} with [~groups].

    Weights stored [[| in; out; k… |]] are [Nx.moveaxis 1 (-1)] of the stored
    tensor. A stored output padding [a] beside a padding [p] is the padding
    [(p, p - a)]. *)

(** {1:types Types} *)

type ('a, _) t = { w : 'a; b : 'a option }
(** The type for transposed-convolution parameters over payload ['a]. At
    tensor payloads [w] has shape [[| in_channels; k…; out_channels |]] and
    [b], when present, [[| out_channels |]]. [b] is [None] for a layer built
    without a bias ({!make}[ ~bias:false]). *)

(** {1:constructors Constructors} *)

val make :
  ?w_init:('b, 'd) Init.t ->
  ?bias_init:('b, 'd) Init.t ->
  ?bias:bool ->
  key:'d Nx.Rng.key ->
  in_channels:int ->
  out_channels:int ->
  kernel_size:int array ->
  (float, 'b) Nx.dtype ->
  ((float, 'b, 'd) Nx.t, 'd) t
(** [make ~key ~in_channels ~out_channels ~kernel_size dtype] is
    {!Conv.make}'s construction for this layout, its initializers applied with
    [~fan_in:(in_channels * taps)] and [~fan_out:(out_channels * taps)],
    [taps] the product of [kernel_size]: the inputs one output reads and the
    outputs one input feeds at unit stride. A stride [r] along an axis divides
    the inputs an output reads there by [r] on average, exactly where [r]
    divides the kernel's extent, and leaves the outputs an input feeds as
    they are; an initializer that should count it is wrapped to rescale
    [fan_in].

    Raises [Invalid_argument] if [kernel_size] is empty, or a channel count
    or an extent is not positive. *)

val init :
  key:'d Nx.Rng.key ->
  in_channels:int ->
  out_channels:int ->
  kernel_size:int array ->
  ('d Nx.float32_t, 'd) t
(** [init ~key ~in_channels ~out_channels ~kernel_size] is
    [make ~key ~in_channels ~out_channels ~kernel_size Nx.float32]:
    Glorot-uniform weights, zero bias. *)

(** {1:applying Applying} *)

val apply :
  ?stride:int array ->
  ?padding:(int * int) array ->
  ?dilation:int array ->
  ((float, 'b, 'd) Nx.t, 'd) t ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [apply p x] is {!Nx.conv_transpose} of [x] by [p.w] with [p.b] as its
    [init], broadcast, each output rounded once as in {!Conv.apply}. [x] is
    [[| n; t…; in_channels |]] and the result [[| n; s…; out_channels |]],
    [s = (t - 1) stride + e - lo - hi] as {!Nx.conv_transpose} states, [e] the
    dilated filter's extent. An [hi] below [lo] lengthens the result.
    Differentiable through Rune.

    Raises [Invalid_argument] as {!Nx.conv_transpose} does. *)

(** {1:structure Structure} *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
(** [walk c p] walks [w] at ["w"], then [b] at ["b"], as {!Conv.walk}. *)
