(** Dataset loading and generation utilities for Nx.

    This module provides functions to load common machine learning datasets and
    generate synthetic datasets for testing and experimentation. Real datasets
    are downloaded and cached in the platform-specific cache directory. *)

(** {2 Loading Real Datasets}

    Functions to load classic machine learning datasets as Nx tensors. *)

(** {3 Image Datasets} *)

val load_mnist : unit -> (Nx.uint8_t * Nx.uint8_t) * (Nx.uint8_t * Nx.uint8_t)
(** [load_mnist ()] loads MNIST handwritten digits dataset.

    Returns training and test sets with images as uint8 tensors of shape
    [|samples; 28; 28; 1|] and labels as uint8 tensors of shape [|samples; 1|].
    Training set has 60,000 samples, test set has 10,000 samples.

    @raise Failure if download or parsing fails

    Loading MNIST and checking shapes:
    {[
      let (x_train, y_train), (x_test, y_test) = Nx_datasets.load_mnist () in
      Nx.shape x_train = [| 60000; 28; 28; 1 |]
      && Nx.shape y_train = [| 60000; 1 |]
      && Nx.shape x_test = [| 10000; 28; 28; 1 |]
      && Nx.shape y_test = [| 10000; 1 |]
    ]} *)

val load_fashion_mnist :
  unit -> (Nx.uint8_t * Nx.uint8_t) * (Nx.uint8_t * Nx.uint8_t)
(** [load_fashion_mnist ()] loads Fashion-MNIST clothing dataset.

    Returns same format as MNIST: images as uint8 tensors of shape
    [|samples; 28; 28; 1|] and labels as uint8 tensors of shape [|samples; 1|].

    @raise Failure if download or parsing fails *)

val load_cifar10 : unit -> (Nx.uint8_t * Nx.uint8_t) * (Nx.uint8_t * Nx.uint8_t)
(** [load_cifar10 ()] loads CIFAR-10 color image dataset.

    Returns training and test sets with images as uint8 tensors of shape
    [|samples; 32; 32; 3|] and labels as uint8 tensors of shape [|samples; 1|].
    Training set has 50,000 samples, test set has 10,000 samples.

    @raise Failure if download or parsing fails *)

(** {3 Tabular Datasets} *)

val load_iris : unit -> Nx.float64_t * Nx.int32_t
(** [load_iris ()] loads Iris flower classification dataset.

    Returns features as float64 tensor of shape [|150; 4|] and labels as int32
    tensor of shape [|150; 1|]. Features are sepal length/width and petal
    length/width. Labels are 0 (setosa), 1 (versicolor), 2 (virginica). *)

val load_breast_cancer : unit -> Nx.float64_t * (int, Nx.int_elt) Nx.t
(** [load_breast_cancer ()] loads Breast Cancer Wisconsin dataset.

    Returns features as float64 tensor of shape [|569; 30|] and labels as int32
    tensor of shape [|569; 1|]. Labels are 0 (malignant) or 1 (benign). *)

val load_diabetes : unit -> Nx.float64_t * Nx.float64_t
(** [load_diabetes ()] loads diabetes regression dataset.

    Returns features as float64 tensor of shape [|442; 10|] and targets as
    float64 tensor of shape [|442; 1|]. Target is quantitative measure of
    disease progression one year after baseline. *)

val load_california_housing : unit -> Nx.float64_t * Nx.float64_t
(** [load_california_housing ()] loads California housing prices dataset.

    Returns features as float64 tensor of shape [|20640; 8|] and targets as
    float64 tensor of shape [|20640; 1|]. Target is median house value in
    hundreds of thousands of dollars. *)

(** {3 Time Series Datasets} *)

val load_airline_passengers : unit -> Nx.int32_t
(** [load_airline_passengers ()] loads monthly airline passenger counts.

    Returns int32 tensor of shape [|144|] containing monthly passenger totals
    from 1949 to 1960. *)

(** {2 Generating Synthetic Datasets}

    Functions to generate synthetic datasets with controlled properties for
    algorithm development and testing. *)

(** {3 Classification Dataset Generators} *)

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

    Creates clusters of points with each cluster drawn from a normal
    distribution. Returns features and integer labels.

    @param n_samples total number of points (default: 100)
    @param n_features number of features per sample (default: 2)
    @param centers number of centers or fixed center locations (default: 3)
    @param cluster_std standard deviation of clusters (default: 1.0)
    @param center_box bounding box for random centers (default: (-10.0, 10.0))
    @param shuffle whether to shuffle samples (default: true)
    @param random_state seed for reproducibility (default: random)

    Generating 3 well-separated 2D clusters:
    {[
      let x, y = Nx_datasets.make_blobs ~centers:(`N 3) ~cluster_std:0.5 () in
      Nx.shape x = [| 100; 2 |] && Nx.shape y = [| 100 |]
    ]} *)

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
(** [make_classification ?n_samples ?n_features ?n_informative ...] generates
    random n-class classification problem.

    Creates a dataset with controllable characteristics including informative,
    redundant, and useless features. Useful for testing feature selection.

    @param n_samples number of samples (default: 100)
    @param n_features total number of features (default: 20)
    @param n_informative number of informative features (default: 2)
    @param n_redundant number of redundant features (default: 2)
    @param n_repeated number of duplicated features (default: 0)
    @param n_classes number of classes (default: 2)
    @param n_clusters_per_class number of clusters per class (default: 2)
    @param weights class proportions (default: balanced)
    @param flip_y fraction of labels randomly exchanged (default: 0.01)
    @param class_sep factor multiplying hypercube size (default: 1.0)
    @param hypercube place clusters on hypercube vertices (default: true)
    @param shift shift features by specified value (default: 0.0)
    @param scale multiply features by specified value (default: 1.0)
    @param shuffle whether to shuffle samples and features (default: true)
    @param random_state seed for reproducibility (default: random)

    @raise Failure if n_informative + n_redundant + n_repeated > n_features

    Creating a binary classification dataset:
    {[
      let x, y =
        Nx_datasets.make_classification ~n_features:10 ~n_informative:3
          ~n_redundant:1 ()
      in
      Nx.shape x = [| 100; 10 |] && Nx.shape y = [| 100 |]
    ]} *)

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
(** [make_gaussian_quantiles ?mean ?cov ?n_samples ...] generates isotropic
    Gaussian divided by quantiles.

    Divides a single Gaussian cluster into near-equal-size classes separated by
    concentric hyperspheres. Useful for testing algorithms that assume Gaussian
    distributions.

    @param mean center of the distribution (default: origin)
    @param cov scalar covariance for isotropic distribution (default: 1.0)
    @param n_samples number of samples (default: 100)
    @param n_features number of features (default: 2)
    @param n_classes number of classes (default: 3)
    @param shuffle whether to shuffle samples (default: true)
    @param random_state seed for reproducibility (default: random) *)

val make_hastie_10_2 :
  ?n_samples:int -> ?random_state:int -> unit -> Nx.float32_t * Nx.int32_t
(** [make_hastie_10_2 ?n_samples ?random_state ()] generates Hastie et al. 2009
    binary problem.

    Generates 10-dimensional dataset where y = 1 if sum(x_i^2) > 9.34 else 0.
    Standard benchmark for binary classification.

    @param n_samples number of samples (default: 12000)
    @param random_state seed for reproducibility (default: random) *)

val make_circles :
  ?n_samples:int ->
  ?shuffle:bool ->
  ?noise:float ->
  ?random_state:int ->
  ?factor:float ->
  unit ->
  Nx.float32_t * Nx.int32_t
(** [make_circles ?n_samples ?shuffle ?noise ?random_state ?factor ()] generates
    concentric circles.

    Creates a large circle containing a smaller circle in 2D. Tests algorithms'
    ability to learn non-linear boundaries.

    @param n_samples total number of points (default: 100)
    @param shuffle whether to shuffle samples (default: true)
    @param noise standard deviation of Gaussian noise (default: 0.0)
    @param random_state seed for reproducibility (default: random)
    @param factor scale factor between circles, 0 < factor < 1 (default: 0.8)

    @raise Failure if factor not in (0, 1)

    Creating noisy concentric circles:
    {[
      let x, y = Nx_datasets.make_circles ~noise:0.1 ~factor:0.5 () in
      Nx.shape x = [| 100; 2 |]
      && Array.for_all (fun v -> v = 0 || v = 1) (Nx.to_array y)
    ]} *)

val make_moons :
  ?n_samples:int ->
  ?shuffle:bool ->
  ?noise:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.int32_t
(** [make_moons ?n_samples ?shuffle ?noise ?random_state ()] generates two
    interleaving half circles.

    Creates two half-moon shapes. Tests algorithms' ability to handle non-convex
    clusters.

    @param n_samples total number of points (default: 100)
    @param shuffle whether to shuffle samples (default: true)
    @param noise standard deviation of Gaussian noise (default: 0.0)
    @param random_state seed for reproducibility (default: random) *)

(** {3 Multilabel Classification} *)

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
(** [make_multilabel_classification ?n_samples ?n_features ...] generates random
    multilabel problem.

    Creates samples with multiple labels per instance. Models bag-of-words with
    multiple topics per document.

    @param n_samples number of samples (default: 100)
    @param n_features number of features (default: 20)
    @param n_classes number of classes (default: 5)
    @param n_labels average labels per instance (default: 2)
    @param length sum of features per sample (default: 50)
    @param allow_unlabeled allow samples with no labels (default: true)
    @param sparse return sparse matrix (default: false, not implemented)
    @param return_indicator return binary indicators (default: false)
    @param return_distributions ignored (default: false)
    @param random_state seed for reproducibility (default: random)

    @raise Failure if sparse=true (not implemented)

    Returns (X, Y) where Y type depends on return_indicator:
    - false: `Int with shape [n_samples; n_labels] containing label indices
    - true: `Float with shape [n_samples; n_classes] containing binary
      indicators *)

(** {3 Regression Dataset Generators} *)

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
(** [make_regression ?n_samples ?n_features ...] generates random regression
    problem.

    Creates linear combination of random features with optional noise and
    low-rank structure.

    @param n_samples number of samples (default: 100)
    @param n_features number of features (default: 100)
    @param n_informative number of informative features (default: 10)
    @param n_targets number of regression targets (default: 1)
    @param bias bias term in linear model (default: 0.0)
    @param effective_rank approximate rank of input matrix (default: None)
    @param tail_strength ignored (default: 0.5)
    @param noise standard deviation of Gaussian noise (default: 0.0)
    @param shuffle whether to shuffle samples (default: true)
    @param coef whether to return coefficients (default: false)
    @param random_state seed for reproducibility (default: random)

    Creating multi-output regression:
    {[
      let x, y, coef =
        Nx_datasets.make_regression ~n_features:20 ~n_informative:5 ~n_targets:2
          ~coef:true ()
      in
      Nx.shape x = [| 100; 20 |]
      && Nx.shape y = [| 100; 2 |]
      && match coef with Some c -> Nx.shape c = [| 20; 2 |] | None -> false
    ]} *)

val make_sparse_uncorrelated :
  ?n_samples:int ->
  ?n_features:int ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.float32_t
(** [make_sparse_uncorrelated ?n_samples ?n_features ?random_state ()] generates
    sparse uncorrelated design.

    Only first 4 features affect target: y = x0 + 2*x1 - 2*x2 - 1.5*x3

    @param n_samples number of samples (default: 100)
    @param n_features number of features, must be >= 4 (default: 10)
    @param random_state seed for reproducibility (default: random) *)

val make_friedman1 :
  ?n_samples:int ->
  ?n_features:int ->
  ?noise:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.float32_t
(** [make_friedman1 ?n_samples ?n_features ?noise ?random_state ()] generates
    Friedman #1 problem.

    Features uniformly distributed on [0, 1]. Output: y = 10 * sin(pi * x0 * x1)
    \+ 20 * (x2 - 0.5)^2 + 10 * x3 + 5 * x4 + noise

    @param n_samples number of samples (default: 100)
    @param n_features number of features, must be >= 5 (default: 10)
    @param noise standard deviation of Gaussian noise (default: 0.0)
    @param random_state seed for reproducibility (default: random)

    @raise Failure if n_features < 5 *)

val make_friedman2 :
  ?n_samples:int ->
  ?noise:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.float32_t
(** [make_friedman2 ?n_samples ?noise ?random_state ()] generates Friedman #2
    problem.

    Four features with ranges: x0 in [0,100], x1 in [40,560], x2 in [0,1], x3 in
    [1,11]. Output: y = sqrt(x0^2 + (x1 * x2 - 1/(x1 * x3))^2) + noise

    @param n_samples number of samples (default: 100)
    @param noise standard deviation of Gaussian noise (default: 0.0)
    @param random_state seed for reproducibility (default: random) *)

val make_friedman3 :
  ?n_samples:int ->
  ?noise:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.float32_t
(** [make_friedman3 ?n_samples ?noise ?random_state ()] generates Friedman #3
    problem.

    Four features with same ranges as Friedman #2. Output: y = arctan((x1 * x2 -
    1/(x1 * x3)) / x0) + noise

    @param n_samples number of samples (default: 100)
    @param noise standard deviation of Gaussian noise (default: 0.0)
    @param random_state seed for reproducibility (default: random) *)

(** {3 Manifold Learning Generators} *)

val make_s_curve :
  ?n_samples:int ->
  ?noise:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.float32_t
(** [make_s_curve ?n_samples ?noise ?random_state ()] generates S-curve dataset.

    Creates 3D S-shaped manifold. Returns points and their position along curve.

    @param n_samples number of samples (default: 100)
    @param noise standard deviation of Gaussian noise (default: 0.0)
    @param random_state seed for reproducibility (default: random)

    Returns (X, t) where X has shape [n_samples; 3] and t has shape [n_samples]
*)

val make_swiss_roll :
  ?n_samples:int ->
  ?noise:float ->
  ?random_state:int ->
  ?hole:bool ->
  unit ->
  Nx.float32_t * Nx.float32_t
(** [make_swiss_roll ?n_samples ?noise ?random_state ?hole ()] generates swiss
    roll dataset.

    Creates 3D swiss roll manifold. Returns points and their position along
    roll.

    @param n_samples number of samples (default: 100)
    @param noise standard deviation of Gaussian noise (default: 0.0)
    @param random_state seed for reproducibility (default: random)
    @param hole create hole in swiss roll (default: false)

    Returns (X, t) where X has shape [n_samples; 3] and t has shape [n_samples]
*)

(** {3 Matrix Decomposition Generators} *)

val make_low_rank_matrix :
  ?n_samples:int ->
  ?n_features:int ->
  ?effective_rank:int ->
  ?tail_strength:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t
(** [make_low_rank_matrix ?n_samples ?n_features ?effective_rank ...] generates
    mostly low-rank matrix.

    Creates matrix with bell-shaped singular value profile.

    @param n_samples number of samples (default: 100)
    @param n_features number of features (default: 100)
    @param effective_rank approximate number of singular vectors (default: 10)
    @param tail_strength decay of noisy tail (default: 0.5)
    @param random_state seed for reproducibility (default: random) *)

val make_sparse_coded_signal :
  n_samples:int ->
  n_components:int ->
  n_features:int ->
  n_nonzero_coefs:int ->
  ?random_state:int ->
  unit ->
  Nx.float32_t * Nx.float32_t * Nx.float32_t
(** [make_sparse_coded_signal ~n_samples ~n_components ~n_features
     ~n_nonzero_coefs ?random_state ()] generates sparse signal.

    Creates signal Y = D * X where D is dictionary and X is sparse code.

    @param n_samples number of samples
    @param n_components number of dictionary atoms
    @param n_features number of features per sample
    @param n_nonzero_coefs number of active components per sample
    @param random_state seed for reproducibility (default: random)

    Returns (Y, D, X) where:
    - Y has shape [n_features; n_samples] (encoded signal)
    - D has shape [n_features; n_components] (dictionary)
    - X has shape [n_components; n_samples] (sparse codes) *)

val make_spd_matrix : ?n_dim:int -> ?random_state:int -> unit -> Nx.float32_t
(** [make_spd_matrix ?n_dim ?random_state ()] generates symmetric
    positive-definite matrix.

    Creates random SPD matrix using A^T * A + epsilon * I.

    @param n_dim matrix dimension (default: 30)
    @param random_state seed for reproducibility (default: random) *)

val make_sparse_spd_matrix :
  ?n_dim:int ->
  ?alpha:float ->
  ?norm_diag:bool ->
  ?smallest_coef:float ->
  ?largest_coef:float ->
  ?random_state:int ->
  unit ->
  Nx.float32_t
(** [make_sparse_spd_matrix ?n_dim ?alpha ...] generates sparse symmetric
    positive-definite matrix.

    Creates sparse SPD matrix with controllable sparsity.

    @param n_dim matrix dimension (default: 30)
    @param alpha probability of zero coefficient (default: 0.95)
    @param norm_diag ignored, normalization not implemented (default: false)
    @param smallest_coef
      smallest absolute value of non-zero coefficients (default: 0.1)
    @param largest_coef
      largest absolute value of non-zero coefficients (default: 0.9)
    @param random_state seed for reproducibility (default: random) *)

(** {3 Biclustering Generators} *)

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
(** [make_biclusters ?shape ?n_clusters ...] generates constant block diagonal
    structure.

    Creates matrix with block diagonal biclusters.

    @param shape matrix dimensions (default: (100, 100))
    @param n_clusters number of biclusters (default: 5)
    @param noise standard deviation of Gaussian noise (default: 0.0)
    @param minval minimum value in blocks (default: 10)
    @param maxval maximum value in blocks (default: 100)
    @param shuffle whether to shuffle rows and columns (default: true)
    @param random_state seed for reproducibility (default: random)

    Returns (X, row_labels, col_labels) indicating bicluster membership *)

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
(** [make_checkerboard ?shape ?n_clusters ...] generates checkerboard structure.

    Creates matrix with checkerboard pattern of high/low values.

    @param shape matrix dimensions (default: (100, 100))
    @param n_clusters clusters per dimension (default: (8, 8))
    @param noise standard deviation of Gaussian noise (default: 0.0)
    @param minval value for low squares (default: 10)
    @param maxval value for high squares (default: 100)
    @param shuffle whether to shuffle rows and columns (default: true)
    @param random_state seed for reproducibility (default: random)

    Returns (X, row_labels, col_labels) indicating cluster membership *)
