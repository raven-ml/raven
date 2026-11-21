(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** TESSERA v2's encoders: the four distilled students and the 2B teacher.

    A model maps each pixel of a {!Batch.t} to an {e embedding}, a vector of
    {!dims} values. Each sensor has a {e backbone}:
    - its tokens are standardised, [(x − mean) / (std + 1e-9)] per band in
      float32, with the model's statistics, a Sentinel-1 token's those of its
      orbit, and a stand-in to zeros. A Sentinel-2 band below [0] is first
      raised by [1000], TESSERA's encoding of its training data: its
      preprocessing (github.com/ucam-eo/tessera,
      [tessera_preprocessing/s2_fast_processor.py], [harmonize_arr]) subtracts a
      product's offset of [1000] only from digital numbers of at least [1000],
      so its models saw a reflectance of [−r · 10⁻⁴] as [1000 − r]. A tile keeps
      the negative value ({!Tile}); a tile of TESSERA's own files holds none;
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

    The products of the linear layers, the pooling query's included, and the
    attention of the encoder and fusion layers compute in the dtype of the
    products' parameters, each product's sum and bias rounded once. Every other
    value computes in float32: standardisation, layer and QK norms, the
    pooling's softmax and weighted sum, sums of residuals and the embedding. The
    norms' parameters and the fusion's learned vectors are float32 arrays at
    every dtype, so [Nx.Ptree.cast (module Model) Nx.bfloat16 m] rounds the
    products' parameters alone, as PyTorch's bfloat16 autocast does, except that
    a product's sum and bias are rounded once. Train at float32 parameters and
    cast them inside the objective.

    A student's embedding is Matryoshka-ordered (Kusupati et al. 2022): its
    first [k] values, for [k] in its {!prefixes}, are an embedding of their own,
    as they are. A model is a structure ({!walk}), placed and cast with
    [Nx.Ptree]'s functions. *)

(** {1:types Types} *)

type ('a, 'd) layer = {
  attn : ('a, 'd) Kaun.Attention.t;
  qk : ('d Nx.float32_t * 'd Nx.float32_t) option;
      (** QK norm's scales of queries and keys. *)
  norm1 : ('d Nx.float32_t, 'd) Kaun.Layer_norm.t;
  ffn1 : ('a, 'd) Kaun.Linear.t;
  ffn2 : ('a, 'd) Kaun.Linear.t;
  norm2 : ('d Nx.float32_t, 'd) Kaun.Layer_norm.t;
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
  norm : ('d Nx.float32_t, 'd) Kaun.Layer_norm.t;
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
      modality : 'd Nx.float32_t;
          (** [[2; d]]: added to the S2 and S1 vectors. *)
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
}
(** The type for models. *)

module Net : Nx.Ptree.S with type ('a, 'd) t = ('a, 'd) net
(** The trained part of a model: its parameters, the norms' and the fusion's as
    fixed arrays, which {!Nx.Ptree.cast} keeps and a gradient and an optimizer's
    step change as they change the others. {!Nx.Ptree.Payload}'s maps and folds
    skip fixed arrays, so a payload per parameter misses the norms. Its walk
    reports the form and the lists' lengths. *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
(** [walk c m] walks [m]'s net, as {!Net} does, at ["net"], reports its [heads],
    and walks its statistics as fixed arrays. It is the model's {!Nx.Ptree.S}
    instance. Training differentiates and steps a function
    of the net alone ({!Net}), so the statistics stay as they are. *)

val dims : ((float, 's, 'd) Nx.t, 'd) t -> int
(** [dims m] is the length of [m]'s embeddings, 128 for the students and 1024
    for the teacher. *)

val prefixes : ((float, 's, 'd) Nx.t, 'd) t -> int list
(** [prefixes m] is the Matryoshka prefixes of [m]'s embeddings: [16; 32; 64;
    128] up to {!dims} for the students' form, and [[]] for the teacher's,
    whose embedding is not nested. *)

(** {1:make Making models} *)

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
    from [key] and lying where it lies, with the students' statistics.

    Raises [Invalid_argument] if a field of [a] is not positive or [width] is
    not a multiple of [2 · heads]. *)

(** {1:files Files} *)

val save : string -> ((float, 's, 'd) Nx.t, 'd) t -> unit
(** [save file m] writes [m] to the SafeTensors [file]: its arrays under their
    {!walk} paths, and its {!arch} as the file's metadata, each field under its
    name, [pre_ln] as [true] or [false] and the others in decimal. The model a
    distillation trains is written so. *)

val load : string -> (Nx.host Nx.float32_t, Nx.host) t
(** [load file] is the model in [file]: a SafeTensors file {!save} wrote of a
    float32 model, named [*.safetensors]; or else a released checkpoint, a
    PyTorch file ([Nx_io.load_pytorch]) whose arrays other than the linear
    layers' map the file.

    A released checkpoint is a student's [{"model"; "args"}], with the
    students' statistics, or the teacher's [{"encoder_state_dict"; "arch"}],
    with its pooled Sentinel-1 statistics. The state dict holds the tensors of
    the upstream modules under their PyTorch names, as
    [s2_backbone.transformer_encoder.layers.0.self_attn.in_proj_weight].
    [args] and [arch] give [latent_dim] ([d / 4]), [nhead], [num_layers],
    [dim_feedforward] and [repr_dim], the upstream defaults where absent.
    PyTorch's [[outputs; inputs]] weights are read as stored, in C order, the
    layout {!init}'s weights have, and a packed [in_proj] as [q], [k] and [v],
    each weight and bias an array of its own. A compiled forward thus runs the
    kernels it runs over {!init}'s weights.

    Raises [Failure] as [Nx_io.load_pytorch], [Nx_io.load_safetensors] and
    [Nx_io.Archive.to_value] do, and naming [file] and the tensor or field if
    a SafeTensors file's metadata lacks a field of {!arch} or holds it
    malformed, if a checkpoint is neither form, if a tensor is missing, has another shape
    than [args] or [arch] gives it, or is left unused, if the checkpoint,
    [args] or [arch] is not a dict or a field has another type, if a student's
    [enable_qk_norm] is true, or if the teacher's [nhead] is not [4] or its
    [final_layernorm] is false. *)

(** {1:apply Applying} *)

val linear :
  ((float, 's, 'd) Nx.t, 'd) Kaun.Linear.t ->
  (float, 't, 'd) Nx.t ->
  'd Nx.float32_t
(** [linear l x] is [x · w + b] of [l] at its parameters' dtype, [x] cast to it
    first, the result to float32: every product of a model, and of {!Distil}'s
    heads. *)

val apply : ((float, 's, 'd) Nx.t, 'd) t -> 'd Batch.t -> 'd Nx.float32_t
(** [apply m b] is [[pixels; dims m]], the embedding of each of [b]'s pixel
    places. An empty place holds a finite vector, for finite parameters: a slot
    or query that sees no key pools zeros, never NaN. *)

(** {1:train Training} *)

type dropout = {
  layers : float;
      (** At each of an encoder or fusion layer's four sites: its attention
          probabilities after the softmax, its attention's result before the
          residual sum, its feed-forward's hidden values after the ReLU, and its
          feed-forward's result before the residual sum. *)
  reducer : float;  (** Between the reducer's ReLU and its last product. *)
}
(** The type for dropout rates, the sites of PyTorch's
    [nn.TransformerEncoderLayer], which the upstream layers are built as. *)

val v2_dropout : dropout
(** [v2_dropout] is [{ layers = 0.1; reducer = 0.2 }]: v2's distillation rate
    (v2 App. J, Table 9, "Dropout 0.1"), which the paper does not place, and the
    reducer's rate, which the table does not state and the released students'
    code fixes between the reducer's ReLU and its last product
    (github.com/ucam-eo/tessera, commit e368539f,
    [tessera_infer_v2/student/model.py], line 235, [nn.Dropout(0.2)]). *)

val no_dropout : dropout
(** [no_dropout] is the rates [0]. *)

type 'd inputs
(** The type for the values a model computes on: a batch's tokens, each sensor's
    bands standardised by the model's statistics, and their days, with the
    batch's layout. *)

val inputs : ('a, 'd) t -> 'd Batch.t -> 'd inputs
(** [inputs m b] is [b]'s inputs to [m]: its tokens standardised as a
    backbone's first step states, and their days. *)

val mixup : alpha:'d Nx.float32_t -> 'd inputs -> 'd inputs -> 'd inputs
(** [mixup ~alpha i i'] is the {e mixup} (Zhang et al. 2018) of [i] and [i']:
    each token's standardised bands and its day are
    [alpha · x + (1 − alpha) · x'], [x'] the token at the same row and column of
    [i'], each product and the sum rounded once in float32, with [i]'s layout.
    Each orbit's tokens are standardised by their own statistics before they are
    mixed. [i'] must lay its tokens out as [i] does, with equal bounds, counts,
    places and slots.

    Raises [Invalid_argument] if [alpha] is not a scalar, or if an array of
    [i']'s layout differs in shape from [i]'s. *)

val train :
  dropout:dropout ->
  keys:'d Nx.Rng.key ->
  ((float, 's, 'd) Nx.t, 'd) t ->
  'd inputs ->
  'd Nx.float32_t
(** [train ~dropout ~keys m i] is {!apply} with dropout, as a training step
    computes it, on [m]'s inputs [i] ({!inputs}, or their {!mixup}). [keys] is
    [[pixels; 2]], row [k] the key of place [k]. Each site drops a value with
    its rate and scales a kept one by [1 / (1 − rate)]. At a rate above [0] a
    layer's attention draws its mask whole, [[rows; heads; width; width]]
    booleans, and computes as one call of {!Nx.attention} that keeps the pairs
    its mask keeps. A token that keeps no key, one with no key to see included,
    attends to [0] in each head that keeps none. [i] must be inputs by [m]'s
    statistics: inputs by another model's are computed on as they are.

    A mask is a function of the place's key, the site, and the index in its slot
    of each token the value reads, so a pixel's embedding is the same in any
    row, slot, batch or group, on any device, eager or compiled. A token of
    count [m] shares its mask among the [m] copies it stands for: train on
    tokens of count [1] ({!Batch.expand}), whose copies drop independently.

    At rates [0], [train ~keys m (inputs m b)] is [apply m b] bit for bit.

    Raises [Invalid_argument] if a rate is outside \[[0]; [1]), or if [keys] is
    not [[pixels; 2]]. *)

(**/**)

val masks :
  dropout:dropout ->
  keys:'d Nx.Rng.key ->
  ((float, 's, 'd) Nx.t, 'd) t ->
  'd Batch.t ->
  (string * 'd Nx.bool_t) list
(** [masks ~dropout ~keys m b] is the masks of {!train}, [true] where a value is
    kept, in the order the forward draws them, so that a reference
    implementation can be fed them: per sensor [s] and layer [i],
    [s_layer{i}_attn_p] ([[rows; heads; width; width]]), [s_layer{i}_attn],
    [s_layer{i}_ffn_hidden] and [s_layer{i}_ffn] ([[rows; width; …]]);
    [fusion_layer{i}_…] for the fusion's, over [[pixels; 2; …]]; and [reducer]
    ([[pixels; …]]). A site of rate [0] has none. *)
