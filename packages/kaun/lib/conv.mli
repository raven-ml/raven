(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Convolution layers.

    A convolution layer is a record of parameters: a stack of filters and an
    optional per-filter bias. It slides its filters over inputs laid out
    channels last, [[| batch; s…; channels |]] for [K] spatial axes [s…], as
    {!Nx.conv} defines it: [[| n; frames; c |]] for audio features,
    [[| n; height; width; c |]] for images, [[| n; depth; height; width; c |]]
    for volumes. [K] is the length of the kernel's extents. The kernel is not
    flipped. Construct parameters with {!init} or {!make} and convolve with
    {!apply}; {!walk} makes the layer a structure, as in {!Linear}. Stride,
    padding, dilation and groups are arguments of {!apply}, written where the
    model's forward is.

    Weights stored channels first, [[| out; in / groups; k… |]], are
    [Nx.moveaxis 1 (-1)] of the stored tensor, a view. An importer of such a
    file reads each layer so:

    {[
    let conv dt weights name =
      let float ~shape leaf =
        Nx_io.Archive.float ~shape dt (name ^ leaf) weights
      in
      {
        Conv.w = Nx.moveaxis 1 (-1) (float ~shape:[| 384; 80; 3 |] ".weight");
        b = Some (float ~shape:[| 384 |] ".bias");
      }
    ]}

    Pooling has no parameters and lives in {!Pool}. *)

(** {1:types Types} *)

type ('a, _) t = { w : 'a; b : 'a option }
(** The type for convolution parameters over payload ['a]. At tensor payloads
    [w] has shape [[| out_channels; k…; in_channels / groups |]], one filter
    per output channel, and [b], when present, [[| out_channels |]]. [b] is
    [None] for a layer built without a bias ({!make}[ ~bias:false]); {!walk}
    skips it and {!apply} adds nothing. *)

(** {1:constructors Constructors} *)

val make :
  ?w_init:('b, 'd) Init.t ->
  ?bias_init:('b, 'd) Init.t ->
  ?bias:bool ->
  ?groups:int ->
  key:'d Nx.Rng.key ->
  in_channels:int ->
  out_channels:int ->
  kernel_size:int array ->
  (float, 'b) Nx.dtype ->
  ((float, 'b, 'd) Nx.t, 'd) t
(** [make ~key ~in_channels ~out_channels ~kernel_size dtype] is a fresh layer
    of [out_channels] filters of extents [kernel_size], each reading
    [in_channels / groups] channels, drawn from two keys {!Nx.Rng.split} makes
    of [key] and lying where it lies, with:

    - [w_init], applied with [~fan_in:(in_channels / groups * taps)] and
      [~fan_out:(out_channels / groups * taps)], [taps] the product of
      [kernel_size]: the inputs one output reads and the outputs one input
      feeds, counted at unit stride. Defaults to {!Init.glorot_uniform}.
    - [bias_init], applied with the same fans. Defaults to {!Init.zeros}.
    - [bias], whether the layer has a bias. Defaults to [true].
    - [groups], the number of channel groups, output group [g] reading input
      group [g] alone. Defaults to [1]; [in_channels] makes a depthwise
      layer.

    Raises [Invalid_argument] if [kernel_size] is empty, a channel count or
    an extent is not positive, or [groups] is not positive or does not divide
    both channel counts. *)

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
  ?groups:int ->
  ((float, 'b, 'd) Nx.t, 'd) t ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [apply p x] is {!Nx.conv} of [x] by [p.w] with [p.b] as its [init],
    broadcast. Each output is [round (b + Σ x w)], the products summed in
    the dtype's accumulator, [b] added there, and the result rounded once.
    [x] is [[| n; s…; in_channels |]] and the result
    [[| n; t…; out_channels |]]. [stride], [padding] and [dilation] are
    {!Nx.conv}'s, which states the extents [t…]. [groups] is {!Nx.conv}'s
    too, and must be the one the layer was made with. A filter of odd extent
    [k] keeps an axis's extent at unit stride and dilation under the padding
    [(k / 2, k / 2)]; {!Nx.ceil_padding} gives the padding that keeps
    [⌈s / stride⌉] windows at any stride. Differentiable through Rune.

    Raises [Invalid_argument] as {!Nx.conv} does. *)

(** {1:structure Structure} *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
(** [walk c p] walks [p]'s parameters: [w] at ["w"], then [b] at ["b"],
    reporting whether [b] is present, so a layer with a bias and one without
    are distinct structures. It is the layer's {!Nx.Ptree.S} instance.
    [Nx.Ptree.instantiate (module Conv)] is the layer at one dtype, and
    [Nx.Ptree.cast (module Conv) dtype p] converts its precision. *)
