(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** 2-D convolution layers.

    A convolution layer is a plain record of parameters: a stack of filters and
    an optional per-filter bias. It slides its filters over inputs in NCHW
    layout — [[| batch; channels; height; width |]] — computing the
    cross-correlation used by deep-learning frameworks (no kernel flip).
    Construct parameters with {!init} or {!make} and convolve with {!apply};
    {!walk} makes it a structure, exactly as in {!Linear}.

    Pooling has no parameters and lives in {!Pool}. *)

(** {1:types Types} *)

type ('a, _) t = { w : 'a; b : 'a option }
(** The type for convolution parameters over payload ['a].

    At tensor payloads, [w] has shape [[| out_channels; in_channels; kh; kw |]]
    — one [in_channels × kh × kw] filter per output channel — and [b], when
    present, shape [[| out_channels |]]. [b] is [None] for layers built without
    a bias ({!make}[ ~bias:false]); such layers have no bias parameter at all,
    so {!walk} skips it and {!apply} performs no shift. *)

(** {1:constructors Constructors} *)

val make :
  ?w_init:('b, 'd) Init.t ->
  ?bias_init:('b, 'd) Init.t ->
  ?bias:bool ->
  key:'d Nx.Rng.key ->
  in_channels:int ->
  out_channels:int ->
  kernel_size:int * int ->
  (float, 'b) Nx.dtype ->
  ((float, 'b, 'd) Nx.t, 'd) t
(** [make ~key ~in_channels ~out_channels ~kernel_size:(kh, kw) dtype] is a
    fresh layer of [out_channels] filters of shape [in_channels × kh × kw],
    drawn from [key] and lying where it lies, the filters and the bias from two
    keys {!Nx.Rng.split} makes of it, with:

    - [w_init], the weight initializer, applied with
      [~fan_in:(in_channels * kh * kw)] and [~fan_out:(out_channels * kh * kw)],
      the connection counts per output and input unit. Defaults to
      {!Init.glorot_uniform}.
    - [bias_init], the bias initializer, applied with the same fans. Defaults to
      {!Init.zeros}.
    - [bias], whether the layer has a bias parameter. Defaults to [true];
      [false] sets [b] to [None] and ignores [bias_init].

    Raises [Invalid_argument] if [in_channels], [out_channels], [kh] or [kw] is
    not positive. *)

val init :
  key:'d Nx.Rng.key ->
  in_channels:int ->
  out_channels:int ->
  kernel_size:int * int ->
  ('d Nx.float32_t, 'd) t
(** [init ~key ~in_channels ~out_channels ~kernel_size] is
    [make ~key ~in_channels ~out_channels ~kernel_size Nx.float32]:
    Glorot-uniform weights, zero bias. *)

(** {1:applying Applying} *)

val apply :
  ?stride:int * int ->
  ?padding:[ `Same | `Valid ] ->
  ((float, 'b, 'd) Nx.t, 'd) t ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [apply p x] cross-correlates [p]'s filters with [x] and adds the bias (no
    shift when [p.b] is [None]). [x] must have shape
    [[| batch; in_channels; height; width |]] (NCHW); the result has shape
    [[| batch; out_channels; out_height; out_width |]]. Differentiable through
    Rune.

    - [stride], the [(vertical, horizontal)] step between filter applications.
      Defaults to [(1, 1)].
    - [padding], the zero-padding mode. [`Valid] (the default) applies the
      filter only where it fits, so [out_height = (height - kh) / sh + 1] (and
      likewise for the width); [`Same] zero-pads so that
      [out_height = ceil (height / sh)] — with a unit stride the spatial size is
      preserved.

    Raises [Invalid_argument] if [x] is not 4-D, if its channel axis does not
    have size [in_channels], if a stride component is not positive, or if the
    filter does not fit the [`Valid] input ([height < kh] or [width < kw]). *)

(** {1:structure Structure} *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
(** [walk c p] walks [p]'s parameters: [w] at ["w"], then [b] at ["b"],
    reporting whether [b] is present, so a layer with a bias and one without are
    distinct structures. It is the layer's {!Nx.Ptree.S} instance.
    [Nx.Ptree.instantiate (module Conv)] is the layer at one dtype, and
    [Nx.Ptree.cast (module Conv) dtype p] converts its precision. *)
