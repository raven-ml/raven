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
    {!map}, {!map2}, {!iter} and {!names} supply the {!Nx.Ptree.S} and
    checkpoint plumbing, exactly as in {!Linear}.

    Pooling has no parameters and lives in {!Pool}. *)

(** {1:types Types} *)

type 'b params = { w : (float, 'b) Nx.t; b : (float, 'b) Nx.t option }
(** The type for convolution parameters with float dtype layout ['b].

    [w] has shape [[| out_channels; in_channels; kh; kw |]] — one
    [in_channels × kh × kw] filter per output channel — and [b], when present,
    shape [[| out_channels |]]. [b] is [None] for layers built without a bias
    ({!make}[ ~bias:false]); such layers have no bias parameter at all, so
    traversals skip it and {!apply} performs no shift. *)

type t = Nx.float32_elt params
(** The type for single-precision convolution layers, the common case. *)

(** {1:constructors Constructors} *)

val make :
  ?w_init:'b Init.t ->
  ?bias_init:'b Init.t ->
  ?bias:bool ->
  in_channels:int ->
  out_channels:int ->
  kernel_size:int * int ->
  (float, 'b) Nx.dtype ->
  'b params
(** [make ~in_channels ~out_channels ~kernel_size:(kh, kw) dtype] is a fresh
    layer of [out_channels] filters of shape [in_channels × kh × kw], with:

    - [w_init], the weight initializer, applied with
      [~fan_in:(in_channels * kh * kw)] and [~fan_out:(out_channels * kh * kw)],
      the connection counts per output and input unit. Defaults to
      {!Init.glorot_uniform}.
    - [bias_init], the bias initializer, applied with the same fans. Defaults to
      {!Init.zeros}.
    - [bias], whether the layer has a bias parameter. Defaults to [true];
      [false] sets [b] to [None] and ignores [bias_init].

    Random initializers draw from the implicit RNG scope (see {!Nx.Rng}).

    Raises [Invalid_argument] if [in_channels], [out_channels], [kh] or [kw] is
    not positive. *)

val init : in_channels:int -> out_channels:int -> kernel_size:int * int -> t
(** [init ~in_channels ~out_channels ~kernel_size] is
    [make ~in_channels ~out_channels ~kernel_size Nx.float32]: Glorot-uniform
    weights, zero bias. *)

(** {1:applying Applying} *)

val apply :
  ?stride:int * int ->
  ?padding:[ `Same | `Valid ] ->
  'b params ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
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

(** {1:traversals Traversals}

    Plain traversals over the parameter leaves, in the order [w] then [b]. They
    satisfy the {!Nx.Ptree.S} contract at any fixed ['b]. *)

val map : ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) -> 'b params -> 'b params
(** [map f p] is [p] with [f] applied to every parameter leaf. *)

val map2 :
  ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) ->
  'b params ->
  'b params ->
  'b params
(** [map2 f p q] combines [p] and [q] leafwise with [f].

    Raises [Invalid_argument] if one of [p] and [q] has a bias and the other
    does not. *)

val iter : ('a 'c. ('a, 'c) Nx.t -> unit) -> 'b params -> unit
(** [iter f p] applies [f] to every parameter leaf of [p]. *)

val astype : (float, 'c) Nx.dtype -> 'b params -> 'c params
(** [astype dt p] is [p] with every parameter leaf cast to [dt]. Differentiable
    through Rune: gradients flow back at each original leaf's dtype, so an
    astype of float32 parameters inside a loss function yields float32
    gradients. *)

val names : 'b params -> string list
(** [names p] is the checkpoint name of each parameter leaf of [p], in traversal
    order: [["w"; "b"]], or [["w"]] when [p] has no bias. See
    {!Checkpoint.Named}. *)
