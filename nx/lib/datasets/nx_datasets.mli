(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dataset loaders and synthetic dataset generators for [Nx].

    Real datasets are downloaded on demand and cached locally. *)

(** {1:cache Cache paths} *)

val get_cache_dir : ?getenv:(string -> string option) -> string -> string
(** [get_cache_dir ?getenv dataset_name] is the cache directory used for
    [dataset_name].

    [getenv] defaults to [Sys.getenv_opt]. *)

(** {1:real Real datasets} *)

(** {2:image Image datasets} *)

val load_mnist : unit -> (Nx.uint8_t * Nx.uint8_t) * (Nx.uint8_t * Nx.uint8_t)
(** [load_mnist ()] is the MNIST dataset as
    [((x_train, y_train), (x_test, y_test))].

    Shapes are:
    - [x_train]: [|60000; 28; 28; 1|]
    - [y_train]: [|60000; 1|]
    - [x_test]: [|10000; 28; 28; 1|]
    - [y_test]: [|10000; 1|]

    Raises [Failure] on download or parsing errors. *)

val load_fashion_mnist :
  unit -> (Nx.uint8_t * Nx.uint8_t) * (Nx.uint8_t * Nx.uint8_t)
(** [load_fashion_mnist ()] is Fashion-MNIST in the same layout as
    {!val-load_mnist}.

    Raises [Failure] on download or parsing errors. *)

val load_cifar10 : unit -> (Nx.uint8_t * Nx.uint8_t) * (Nx.uint8_t * Nx.uint8_t)
(** [load_cifar10 ()] is CIFAR-10 as [((x_train, y_train), (x_test, y_test))].

    Shapes are:
    - [x_train]: [|50000; 32; 32; 3|]
    - [y_train]: [|50000; 1|]
    - [x_test]: [|10000; 32; 32; 3|]
    - [y_test]: [|10000; 1|]

    Raises [Failure] on download or parsing errors. *)

(** {2:tabular Tabular datasets} *)

val load_iris : unit -> Nx.float64_t * Nx.int32_t
(** [load_iris ()] is [(features, labels)] for the Iris classification dataset.

    [features] has shape [|150; 4|]. [labels] has shape [|150; 1|]. *)

val load_breast_cancer : unit -> Nx.float64_t * (int32, Nx.int32_elt) Nx.t
(** [load_breast_cancer ()] is [(features, labels)] for the Breast Cancer
    Wisconsin dataset.

    [features] has shape [|569; 30|]. [labels] has shape [|569; 1|]. *)

val load_diabetes : unit -> Nx.float64_t * Nx.float64_t
(** [load_diabetes ()] is [(features, targets)] for the diabetes regression
    dataset.

    [features] has shape [|442; 10|]. [targets] has shape [|442; 1|]. *)

val load_california_housing : unit -> Nx.float64_t * Nx.float64_t
(** [load_california_housing ()] is [(features, targets)] for the California
    housing regression dataset.

    [features] has shape [|20640; 8|]. [targets] has shape [|20640; 1|]. *)

(** {2:timeseries Time-series dataset} *)

val load_airline_passengers : unit -> Nx.int32_t
(** [load_airline_passengers ()] is a 1D tensor of monthly passenger counts with
    shape [|144|]. *)

(** {1:generated Synthetic datasets} *)

(** {2:classification Classification generators} *)

val make_blobs :
  ?n_samples:int ->
  ?n_features:int ->
  ?centers:[ `N of int | `Array of Nx.float32_t ] ->
  ?cluster_std:float ->
  ?center_box:float * float ->
  ?shuffle:bool ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.int32_t
(** [make_blobs ?n_samples ?n_features ?centers ?cluster_std ?center_box
     ?shuffle ?random_state ()] generates isotropic Gaussian blobs.

    Options are:
    - [n_samples] is the number of samples. Defaults to [100].
    - [n_features] is the number of features per sample. Defaults to [2].
    - [centers] is either [`N k] random centers or an explicit array of centers
      with shape [|k; n_features|]. Defaults to [`N 3].
    - [cluster_std] is the cluster standard deviation. Defaults to [1.0].
    - [center_box] is the random center range. Defaults to [(-10., 10.)].
    - [shuffle] controls sample shuffling. Defaults to [true].
    - [random_state] seeds randomness. Defaults to an unseeded state.

    Returns [(x, y)] with [x] shape [|n_samples; n_features|] and [y] shape
    [|n_samples|]. *)

val make_classification :
  ?n_samples:int ->
  ?n_features:int ->
  ?n_informative:int ->
  ?n_redundant:int ->
  ?n_repeated:int ->
  ?n_classes:int ->
  ?n_clusters_per_class:int ->
  ?weights:float list ->
  ?flip_y:float ->
  ?class_sep:float ->
  ?hypercube:bool ->
  ?shift:float ->
  ?scale:float ->
  ?shuffle:bool ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.int32_t
(** [make_classification ?n_samples ?n_features ?n_informative ?n_redundant
     ?n_repeated ?n_classes ?n_clusters_per_class ?weights ?flip_y ?class_sep
     ?hypercube ?shift ?scale ?shuffle ?random_state ()] generates an
    [n_classes]-class classification problem.

    Options are:
    - [n_samples] is the number of samples. Defaults to [100].
    - [n_features] is the feature count. Defaults to [20].
    - [n_informative] is the informative feature count. Defaults to [2].
    - [n_redundant] is the linear combination feature count. Defaults to [2].
    - [n_repeated] is the duplicated feature count. Defaults to [0].
    - [n_classes] is the class count. Defaults to [2].
    - [n_clusters_per_class] is the cluster count per class. Defaults to [2].
    - [weights] are class proportions. Defaults to uniform proportions.
    - [flip_y] is the fraction of randomly relabeled samples. Defaults to
      [0.01].
    - [class_sep] scales centroid separation. Defaults to [1.0].
    - [hypercube] chooses deterministic hypercube centroids. Defaults to [true].
    - [shift] is an additive feature shift. Defaults to [0.0].
    - [scale] is a multiplicative feature scale. Defaults to [1.0].
    - [shuffle] controls sample shuffling. Defaults to [true].
    - [random_state] seeds randomness. Defaults to an unseeded state.

    Returns [(x, y)] with [x] shape [|n_samples; n_features|] and [y] shape
    [|n_samples|].

    Raises [Failure] if [n_informative + n_redundant + n_repeated > n_features].

    Raises [Invalid_argument] if [weights] has fewer than [n_classes] elements.
*)

val make_gaussian_quantiles :
  ?mean:float array ->
  ?cov:float ->
  ?n_samples:int ->
  ?n_features:int ->
  ?n_classes:int ->
  ?shuffle:bool ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.int32_t
(** [make_gaussian_quantiles ?mean ?cov ?n_samples ?n_features ?n_classes
     ?shuffle ?random_state ()] samples from a Gaussian and assigns class labels
    by radial quantiles.

    Options are:
    - [mean] is the Gaussian mean, with length [n_features]. Defaults to the
      origin.
    - [cov] is the isotropic covariance scalar. Defaults to [1.0].
    - [n_samples] is the number of samples. Defaults to [100].
    - [n_features] is the feature count. Defaults to [2].
    - [n_classes] is the class count. Defaults to [3].
    - [shuffle] controls sample shuffling. Defaults to [true].
    - [random_state] seeds randomness. Defaults to an unseeded state.

    Returns [(x, y)] with [x] shape [|n_samples; n_features|] and [y] shape
    [|n_samples|]. *)

val make_hastie_10_2 :
  ?n_samples:int -> ?random_state:int -> unit -> Nx.float32_t * Nx.int32_t
(** [make_hastie_10_2 ?n_samples ?random_state ()] generates the Hastie 10.2
    binary classification benchmark.

    [n_samples] defaults to [12000]. [random_state] defaults to an unseeded
    state.

    Returns [(x, y)] with [x] shape [|n_samples; 10|] and [y] shape
    [|n_samples|]. *)

val make_circles :
  ?n_samples:int ->
  ?shuffle:bool ->
  ?noise:float ->
  ?random_state:int ->
  ?factor:float ->
  unit ->
  Nx.float32_t * Nx.int32_t
(** [make_circles ?n_samples ?shuffle ?noise ?random_state ?factor ()]
    generates a two-class concentric circles dataset.

    Options are:
    {ul
    {- [n_samples] is the number of samples. Defaults to [100].}
    {- [shuffle] controls sample shuffling. Defaults to [true].}
    {- [noise] is Gaussian feature noise standard deviation.
       Defaults to [0.0].}
    {- [random_state] seeds randomness. Defaults to an unseeded state.}
    {- [factor] is the inner circle scale in ]0;1[. Defaults to [0.8].}}

    Returns [(x, y)] with [x] shape [|n_samples; 2|] and [y]
    shape [|n_samples|].

    Raises [Failure] if [factor <= 0.] or [factor >= 1.]. *)

val make_moons :
  ?n_samples:int ->
  ?shuffle:bool ->
  ?noise:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.int32_t
(** [make_moons ?n_samples ?shuffle ?noise ?random_state ()] generates a
    two-class interleaving half-moons dataset.

    [n_samples] defaults to [100]. [shuffle] defaults to [true]. [noise]
    defaults to [0.0]. [random_state] defaults to an unseeded state.

    Returns [(x, y)] with [x] shape [|n_samples; 2|] and [y] shape
    [|n_samples|]. *)

(** {2:multilabel Multilabel classification} *)

val make_multilabel_classification :
  ?n_samples:int ->
  ?n_features:int ->
  ?n_classes:int ->
  ?n_labels:int ->
  ?length:int ->
  ?allow_unlabeled:bool ->
  ?sparse:bool ->
  ?return_indicator:bool ->
  ?return_distributions:bool ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * [ `Float of Nx.float32_t | `Int of Nx.int32_t ]
(** [make_multilabel_classification ?n_samples ?n_features ?n_classes ?n_labels
     ?length ?allow_unlabeled ?sparse ?return_indicator ?return_distributions
     ?random_state ()] generates a multilabel classification dataset.

    Options are:
    - [n_samples] is the number of samples. Defaults to [100].
    - [n_features] is the feature count. Defaults to [20].
    - [n_classes] is the class count. Defaults to [5].
    - [n_labels] is the target average labels per sample. Defaults to [2].
    - [length] controls average feature occurrences per sample. Defaults to
      [50].
    - [allow_unlabeled] allows samples with no labels. Defaults to [true].
    - [sparse] requests sparse output. Defaults to [false].
    - [return_indicator] selects indicator output. Defaults to [false].
    - [return_distributions] is currently ignored. Defaults to [false].
    - [random_state] seeds randomness. Defaults to an unseeded state.

    Returns [(x, y)] where [x] has shape [|n_samples; n_features|].

    If [return_indicator = true], [y = `Float indicators] with shape
    [|n_samples; n_classes|].

    Otherwise [y = `Int labels] with shape [|n_samples; n_labels|].

    Raises [Failure] if [sparse = true] (not implemented). *)

(** {2:regression Regression generators} *)

val make_regression :
  ?n_samples:int ->
  ?n_features:int ->
  ?n_informative:int ->
  ?n_targets:int ->
  ?bias:float ->
  ?effective_rank:int option ->
  ?tail_strength:float ->
  ?noise:float ->
  ?shuffle:bool ->
  ?coef:bool ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.float32_t * Nx.float32_t option
(** [make_regression ?n_samples ?n_features ?n_informative ?n_targets ?bias
     ?effective_rank ?tail_strength ?noise ?shuffle ?coef ?random_state ()]
    generates a linear regression dataset.

    Options are:
    - [n_samples] is the number of samples. Defaults to [100].
    - [n_features] is the feature count. Defaults to [100].
    - [n_informative] is the informative feature count. Defaults to [10]. Values
      larger than [n_features] are clamped to [n_features].
    - [n_targets] is the target count. Defaults to [1].
    - [bias] is the additive target bias. Defaults to [0.0].
    - [effective_rank] is [Some r] to generate low-rank features, or [None] for
      full-rank Gaussian features. Defaults to [None].
    - [tail_strength] is currently ignored. Defaults to [0.5].
    - [noise] is Gaussian target noise standard deviation. Defaults to [0.0].
    - [shuffle] controls sample shuffling. Defaults to [true].
    - [coef] requests returning the generating coefficients. Defaults to
      [false].
    - [random_state] seeds randomness. Defaults to an unseeded state.

    Returns [(x, y, coef_opt)] where [x] has shape [|n_samples; n_features|].

    [y] has shape [|n_samples|] if [n_targets = 1], otherwise
    [|n_samples; n_targets|].

    [coef_opt] is [Some coef] iff [coef = true]. *)

val make_sparse_uncorrelated :
  ?n_samples:int ->
  ?n_features:int ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.float32_t
(** [make_sparse_uncorrelated ?n_samples ?n_features ?random_state ()] generates
    a regression dataset with sparse informative features.

    [n_samples] defaults to [100]. [n_features] defaults to [10]. [random_state]
    defaults to an unseeded state.

    Returns [(x, y)] with [x] shape [|n_samples; n_features|] and [y] shape
    [|n_samples|].

    If [n_features < 4], [y] is identically zero. *)

val make_friedman1 :
  ?n_samples:int ->
  ?n_features:int ->
  ?noise:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.float32_t
(** [make_friedman1 ?n_samples ?n_features ?noise ?random_state ()] generates
    the Friedman #1 regression benchmark.

    [n_samples] defaults to [100]. [n_features] defaults to [10]. [noise]
    defaults to [0.0]. [random_state] defaults to an unseeded state.

    Returns [(x, y)] with [x] shape [|n_samples; n_features|] and [y] shape
    [|n_samples|].

    Raises [Failure] if [n_features < 5]. *)

val make_friedman2 :
  ?n_samples:int ->
  ?noise:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.float32_t
(** [make_friedman2 ?n_samples ?noise ?random_state ()] generates the Friedman
    #2 regression benchmark.

    [n_samples] defaults to [100]. [noise] defaults to [0.0]. [random_state]
    defaults to an unseeded state.

    Returns [(x, y)] with [x] shape [|n_samples; 4|] and [y] shape
    [|n_samples|]. *)

val make_friedman3 :
  ?n_samples:int ->
  ?noise:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.float32_t
(** [make_friedman3 ?n_samples ?noise ?random_state ()] generates the Friedman
    #3 regression benchmark.

    [n_samples] defaults to [100]. [noise] defaults to [0.0]. [random_state]
    defaults to an unseeded state.

    Returns [(x, y)] with [x] shape [|n_samples; 4|] and [y] shape
    [|n_samples|]. *)

(** {2:manifold Manifold generators} *)

val make_s_curve :
  ?n_samples:int ->
  ?noise:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.float32_t
(** [make_s_curve ?n_samples ?noise ?random_state ()] generates an S-curve
    manifold.

    [n_samples] defaults to [100]. [noise] defaults to [0.0]. [random_state]
    defaults to an unseeded state.

    Returns [(x, t)] where [x] has shape [|n_samples; 3|] and [t] has shape
    [|n_samples|]. *)

val make_swiss_roll :
  ?n_samples:int ->
  ?noise:float ->
  ?random_state:int ->
  ?hole:bool ->
  unit ->
  Nx.float32_t * Nx.float32_t
(** [make_swiss_roll ?n_samples ?noise ?random_state ?hole ()] generates a
    Swiss-roll manifold.

    [n_samples] defaults to [100]. [noise] defaults to [0.0]. [random_state]
    defaults to an unseeded state. [hole] defaults to [false].

    Returns [(x, t)] where [x] has shape [|n; 3|] and [t] has shape [|n|].

    If [hole = true], [n] can be smaller than [n_samples]. *)

(** {2:matrix Matrix generators} *)

val make_low_rank_matrix :
  ?n_samples:int ->
  ?n_features:int ->
  ?effective_rank:int ->
  ?tail_strength:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t
(** [make_low_rank_matrix ?n_samples ?n_features ?effective_rank ?tail_strength
     ?random_state ()] generates a mostly low-rank matrix with shape
    [|n_samples; n_features|].

    [n_samples] defaults to [100]. [n_features] defaults to [100].
    [effective_rank] defaults to [10]. [tail_strength] controls additive
    Gaussian noise amplitude. Defaults to [0.5]. [random_state] defaults to an
    unseeded state. *)

val make_sparse_coded_signal :
  n_samples:int ->
  n_components:int ->
  n_features:int ->
  n_nonzero_coefs:int ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.float32_t * Nx.float32_t
(** [make_sparse_coded_signal ~n_samples ~n_components ~n_features
     ~n_nonzero_coefs ?random_state ()] generates [(y, d, x)] such that
    [y = d @ x].

    Arguments are:
    - [n_samples] is the number of generated signals.
    - [n_components] is the dictionary atom count.
    - [n_features] is the signal dimension.
    - [n_nonzero_coefs] is the non-zero coefficient count per signal. It must
      satisfy [n_nonzero_coefs <= n_components].
    - [random_state] seeds randomness. Defaults to an unseeded state.

    Shapes are:
    - [y]: [|n_features; n_samples|]
    - [d]: [|n_features; n_components|]
    - [x]: [|n_components; n_samples|]

    Raises [Invalid_argument] if [n_nonzero_coefs > n_components]. *)

val make_spd_matrix : ?n_dim:int -> ?random_state:int -> unit -> Nx.float32_t
(** [make_spd_matrix ?n_dim ?random_state ()] generates a symmetric
    positive-definite matrix with shape [|n_dim; n_dim|].

    [n_dim] defaults to [30]. [random_state] defaults to an unseeded state. *)

val make_sparse_spd_matrix :
  ?n_dim:int ->
  ?alpha:float ->
  ?norm_diag:bool ->
  ?smallest_coef:float ->
  ?largest_coef:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t
(** [make_sparse_spd_matrix ?n_dim ?alpha ?norm_diag ?smallest_coef
     ?largest_coef ?random_state ()] generates a sparse symmetric
    positive-definite matrix with shape [|n_dim; n_dim|].

    [n_dim] defaults to [30]. [alpha] is the probability of keeping a zero entry
    and defaults to [0.95]. [norm_diag] is currently ignored and defaults to
    [false]. [smallest_coef] defaults to [0.1]. [largest_coef] defaults to
    [0.9]. [random_state] defaults to an unseeded state. *)

(** {2:bicluster Biclustering generators} *)

val make_biclusters :
  ?shape:int * int ->
  ?n_clusters:int ->
  ?noise:float ->
  ?minval:int ->
  ?maxval:int ->
  ?shuffle:bool ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.int32_t * Nx.int32_t
(** [make_biclusters ?shape ?n_clusters ?noise ?minval ?maxval ?shuffle
     ?random_state ()] generates a block-diagonal bicluster matrix.

    [shape] defaults to [(100, 100)]. [n_clusters] defaults to [5]. [noise]
    defaults to [0.0]. [minval] defaults to [10]. [maxval] defaults to [100].
    [shuffle] defaults to [true]. [random_state] defaults to an unseeded state.

    Returns [(x, row_labels, col_labels)] where [x] has shape
    [|fst shape; snd shape|], [row_labels] has shape [|fst shape|], and
    [col_labels] has shape [|snd shape|].

    Raises [Division_by_zero] if [n_clusters = 0].

    Raises [Invalid_argument] if [maxval <= minval]. *)

val make_checkerboard :
  ?shape:int * int ->
  ?n_clusters:int * int ->
  ?noise:float ->
  ?minval:int ->
  ?maxval:int ->
  ?shuffle:bool ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.int32_t * Nx.int32_t
(** [make_checkerboard ?shape ?n_clusters ?noise ?minval ?maxval ?shuffle
     ?random_state ()] generates a checkerboard bicluster matrix.

    [shape] defaults to [(100, 100)]. [n_clusters] defaults to [(8, 8)] for row
    and column clusters. [noise] defaults to [0.0]. [minval] defaults to [10].
    [maxval] defaults to [100]. [shuffle] defaults to [true]. [random_state]
    defaults to an unseeded state.

    Returns [(x, row_labels, col_labels)] where [x] has shape
    [|fst shape; snd shape|], [row_labels] has shape [|fst shape|], and
    [col_labels] has shape [|snd shape|].

    Raises [Division_by_zero] if [fst n_clusters = 0] or [snd n_clusters = 0].
*)
