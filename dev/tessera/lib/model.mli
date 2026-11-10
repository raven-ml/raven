(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** TESSERA v2's encoders: the four distilled students and the 2B teacher.

    A model maps each pixel of a {!Batch.t} to an {e embedding}, a vector of
    {!dims} values. Each sensor has a {e backbone}:
    - its tokens are standardised, [(x − mean) / (std + 1e-9)] per band in
      float32, with the model's statistics, a Sentinel-1 token's those of its
      orbit, and a stand-in to zeros;
    - the bands go through [Linear → ReLU → Linear], and the sinusoid of the day
      of year is added, [sin] on even channels and [cos] on odd ones, at
      frequencies [exp (2i · (−ln 10000 / d))], each product in float32;
    - the encoder layers attend over the tokens of each slot, each key's score
      plus its [log count];
    - one learned query per slot pools the slot's tokens, each scored by the
      query's product plus its bias and [log count].

    There are two {e forms}. The students' post-LN layers compute
    [x ← LN₁ (x + attn x)], then [x ← LN₂ (x + ffn x)], and their two pooled
    vectors are concatenated. The teacher's pre-LN layers compute
    [x ← x + attn (LN₁ x)], then [x ← x + ffn (LN₂ x)], with QK norm, and its
    two pooled vectors, plus a learned vector per sensor, go through fusion
    layers of the same form. Both end in the reducer
    [Linear → LayerNorm → ReLU → Linear] and a layer norm without parameters, so
    each embedding has mean [0] and deviation [1]. Layer norms take eps [1e-5].
    A layer with QK norm divides each head's queries and keys by their root mean
    square, eps float32's machine epsilon, times a learned scale, before the
    scores; the attention's scale is [1 / √head_dim]. [ffn] is
    [Linear → ReLU → Linear].

    The products of the linear layers and the attention of the encoder and
    fusion layers compute in the parameters' dtype. Standardisation, layer and
    QK norms, pooling, sums of residuals and the embedding compute in float32,
    from parameters cast to float32.
    [Nx.Ptree.cast (module Model) Nx.bfloat16 m] rounds every parameter of [m]
    to bfloat16, its layer norms' and QK norms' scales and the fusion's learned
    vectors included, and its pooling query computes in float32. It thus differs
    from PyTorch's bfloat16 autocast, which keeps every parameter in float32 and
    rounds the operands of each product, the pooling query's included.

    A student's embedding is Matryoshka-ordered (Kusupati et al. 2022): its
    first [k] values, for [k] in its {!prefixes}, are an embedding of their own,
    as they are. A model is a structure ({!walk}), placed and cast with
    [Nx.Ptree]'s functions. *)

(** {1:types Types} *)

type ('a, 'd) layer = {
  attn : ('a, 'd) Kaun.Attention.t;
  qk : ('a * 'a) option;  (** QK norm's scales of queries and keys. *)
  norm1 : ('a, 'd) Kaun.Layer_norm.t;
  ffn1 : ('a, 'd) Kaun.Linear.t;
  ffn2 : ('a, 'd) Kaun.Linear.t;
  norm2 : ('a, 'd) Kaun.Layer_norm.t;
}
(** The type for encoder layers. *)

type ('a, 'd) backbone = {
  embed : ('a, 'd) Kaun.Linear.t * ('a, 'd) Kaun.Linear.t;
  layers : ('a, 'd) layer list;
  pool : ('a, 'd) Kaun.Linear.t;  (** [[d; 1]]: the pooling query and bias. *)
}
(** The type for a sensor's backbone. *)

type ('a, 'd) reducer = {
  up : ('a, 'd) Kaun.Linear.t;
  norm : ('a, 'd) Kaun.Layer_norm.t;
  down : ('a, 'd) Kaun.Linear.t;
}
(** The type for reducers. *)

(** The type for the two forms. *)
type ('a, 'd) net =
  | Post_ln of {
      s2 : ('a, 'd) backbone;
      s1 : ('a, 'd) backbone;
      reducer : ('a, 'd) reducer;
    }
  | Pre_ln of {
      s2 : ('a, 'd) backbone;
      s1 : ('a, 'd) backbone;
      modality : 'a;  (** [[2; d]]: added to the S2 and S1 vectors. *)
      fusion : ('a, 'd) layer list;
      reducer : ('a, 'd) reducer;
    }  (** Every layer has QK norm. *)

type 'd norm = {
  s2_mean : 'd Nx.float32_t;  (** [[10]] *)
  s2_std : 'd Nx.float32_t;  (** [[10]] *)
  s1_mean : 'd Nx.float32_t;  (** [[2; 2]]: ascending, then descending. *)
  s1_std : 'd Nx.float32_t;  (** [[2; 2]] *)
}
(** The type for standardisation statistics, fixed arrays of a model. *)

type ('a, 'd) t = {
  net : ('a, 'd) net;
  heads : int;  (** Attention heads of every layer. *)
  norm : 'd norm;
  prefixes : int list;
      (** The Matryoshka prefixes, [[]] where the embedding is not nested. *)
}
(** The type for models. *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
(** [walk c m] walks [m]'s parameters and statistics, and reports its form, its
    [heads] and its lists' lengths. It is the model's {!Nx.Ptree.S} instance,
    and keeps [prefixes]. *)

val dims : ((float, 's, 'd) Nx.t, 'd) t -> int
(** [dims m] is the length of [m]'s embeddings, 128 for the students and 1024
    for the teacher. *)

(** {1:make Making models} *)

val of_pytorch :
  name:string -> Nx_io.Pytorch.value -> (Nx.host Nx.float32_t, Nx.host) t
(** [of_pytorch ~name v] is the model of the released checkpoint [v], named
    [name] in errors. [v] is a student's [{"model"; "args"}], with the students'
    statistics, or the teacher's [{"encoder_state_dict"; "arch"}], with its
    pooled Sentinel-1 statistics. The state dict holds the tensors of the
    upstream modules under their PyTorch names, as
    [s2_backbone.transformer_encoder.layers.0.self_attn.in_proj_weight]. [args]
    and [arch] give [latent_dim] ([d / 4]), [nhead], [num_layers],
    [dim_feedforward] and [repr_dim], the upstream defaults where absent, and a
    student's [matryoshka_dims] (["16,32,64,128"]) and [enable_qk_norm].
    PyTorch's [[outputs; inputs]] weights are read as views of
    [[inputs; outputs]], and a packed [in_proj] as [q], [k] and [v].

    Raises [Failure] naming [name] and the tensor or field if [v] is neither, if
    a tensor is missing, has another shape than [args] or [arch] gives it, or is
    left unused, if a field has another type, or if the teacher's [nhead] is not
    [4] or its [final_layernorm] is false. *)

val load : string -> (Nx.host Nx.float32_t, Nx.host) t
(** [load file] is [of_pytorch ~name:file (Nx_io.load_pytorch file)]. Its arrays
    map the file ([Nx_io]'s loading).

    Raises [Failure] as {!of_pytorch} and [Nx_io.load_pytorch] do. *)

type arch = {
  pre_ln : bool;  (** The teacher's form, else the students'. *)
  width : int;  (** [d], a multiple of [2 · heads]. *)
  heads : int;
  layers : int;
  ffn : int;
  dims : int;
}
(** The type for architectures. A pre-LN model has two fusion layers of
    feed-forward width [2d]. *)

val nano : arch
val small : arch
val medium : arch
val large : arch
val teacher : arch

val init : key:'d Nx.Rng.key -> arch -> ('d Nx.float32_t, 'd) t
(** [init ~key a] is a model of architecture [a] with kaun's initialisers, drawn
    from [key] and lying where it lies, with the students' statistics, and
    Matryoshka prefixes [16; 32; 64; 128] up to [dims] for the students' form.

    Raises [Invalid_argument] if a field of [a] is not positive or [width] is
    not a multiple of [2 · heads]. *)

(** {1:apply Applying} *)

val apply : ((float, 's, 'd) Nx.t, 'd) t -> 'd Batch.t -> 'd Nx.float32_t
(** [apply m b] is [[pixels; dims m]], the embedding of each of [b]'s pixel
    places. An empty place holds a finite vector, for finite parameters: a slot
    or query that sees no key pools zeros, never NaN. *)

(**/**)

val blocks :
  ((float, 's, 'd) Nx.t, 'd) t -> 'd Batch.t -> (string * 'd Nx.float32_t) list
(** [blocks m b] is {!apply}'s intermediate values under the names of the
    upstream layer dumps: per sensor [s], [s_in] (standardised tokens, then the
    day), [s_embedding], [s_pe], [s_encoder_in], [s_layer{i}],
    [s_layer{i}_attn], [s_layer{i}_norm1], [s_layer{i}_linear1], [s_pool_logits]
    and [s_pooled]; then [reducer0_linear0], [reducer1_ln1], [reducer2_relu],
    [reducer4_linear4], [reducer5_output] and [embedding]. Token values are
    [[rows; width; …]], pixel values [[pixels; …]]. *)
