(** ndarray-datasets: Dataset loading utilities for ndarray.

    This module provides functions to load common machine learning datasets as
    [Ndarray]s. Data files are downloaded and cached in the platform-specific
    cache directory (see [Dataset_utils.get_cache_dir]). *)

val load_mnist :
  unit ->
  (Ndarray.uint8_t * Ndarray.uint8_t) * (Ndarray.uint8_t * Ndarray.uint8_t)
(** Load the MNIST handwritten digits dataset.

    This function downloads (if necessary) and returns the MNIST dataset as
    ndarrays for training and test sets. Each image is represented as a uint8
    tensor of shape [|num_samples; height; width; 1|], and labels as a uint8
    tensor of shape [|num_samples; 1|].

    {2 Parameters}
    - [()]: no arguments.

    {2 Returns}
    - [(x_train, y_train), (x_test, y_test)]: * [x_train]: uint8 tensor of shape
      [|60000;28;28;1|] for training images. * [y_train]: uint8 tensor of shape
      [|60000;1|] for training labels. * [x_test]: uint8 tensor of shape
      [|10000;28;28;1|] for test images. * [y_test]: uint8 tensor of shape
      [|10000;1|] for test labels.

    {2 Raises}
    - [Failure] if dataset download or parsing fails. *)

val load_fashion_mnist :
  unit ->
  (Ndarray.uint8_t * Ndarray.uint8_t) * (Ndarray.uint8_t * Ndarray.uint8_t)
(** Load the Fashion-MNIST dataset of Zalando article images.

    This function downloads (if necessary) and returns the Fashion-MNIST dataset
    with the same shape semantics as MNIST: images as uint8 tensors of shape
    [|num_samples; height; width; 1|], and labels as uint8 tensors of shape
    [|num_samples; 1|].

    {2 Parameters}
    - [()]: no arguments.

    {2 Returns}
    - [(x_train, y_train), (x_test, y_test)] similar to [load_mnist].

    {2 Raises}
    - [Failure] if dataset download or parsing fails. *)

val load_cifar10 :
  unit ->
  (Ndarray.uint8_t * Ndarray.uint8_t) * (Ndarray.uint8_t * Ndarray.uint8_t)
(** Load the CIFAR-10 image classification dataset.

    This function downloads (if necessary) and returns the CIFAR-10 dataset as
    ndarrays. Images are uint8 tensors of shape
    [|num_samples; height; width; 3|], and labels are uint8 tensors of shape
    [|num_samples; 1|].

    {2 Parameters}
    - [()]: no arguments.

    {2 Returns}
    - [(x_train, y_train), (x_test, y_test)]: * [x_*]: uint8 tensor of shape
      [|50000| or 10000;32;32;3|]. * [y_*]: uint8 tensor of shape
      [|50000| or 10000;1|] for labels.

    {2 Raises}
    - [Failure] if dataset download or parsing fails. *)

val load_iris : unit -> Ndarray.float64_t * Ndarray.int32_t
(** Load the Iris flower dataset.

    Returns features as a float64 tensor of shape [|num_samples; num_features|],
    and labels as an int32 tensor of shape [|num_samples; 1|].

    {2 Parameters}
    - [()]: no arguments.

    {2 Returns}
    - [features]: float64 tensor of shape [|150;4|].
    - [labels]: int32 tensor of shape [|150;1|]. *)

val load_breast_cancer :
  unit -> Ndarray.float64_t * (int, Bigarray.int_elt) Ndarray.t
(** Load the Breast Cancer Wisconsin dataset.

    Returns features as a float64 tensor of shape [|num_samples; num_features|],
    and labels as an integer tensor of shape [|num_samples; 1|].

    {2 Parameters}
    - [()]: no arguments.

    {2 Returns}
    - [features]: float64 tensor of shape [|num_samples; num_features|].
    - [labels]: int32 tensor of shape [|num_samples;1|]. *)

val load_diabetes : unit -> Ndarray.float64_t * Ndarray.float64_t
(** Load the Diabetes dataset for regression.

    Returns features as a float64 tensor of shape [|num_samples; num_features|],
    and targets as a float64 tensor of shape [|num_samples; 1|].

    {2 Parameters}
    - [()]: no arguments.

    {2 Returns}
    - [features]: float64 tensor of shape [|num_samples; num_features|].
    - [labels]: float64 tensor of shape [|num_samples;1|]. *)

val load_california_housing : unit -> Ndarray.float64_t * Ndarray.float64_t
(** Load the California Housing dataset for regression.

    Returns features as a float64 tensor and targets as a float64 tensor of
    shape [|num_samples;1|].

    {2 Parameters}
    - [()]: no arguments.

    {2 Returns}
    - [features]: float64 tensor of shape [|num_samples; num_features|].
    - [labels]: float64 tensor of shape [|num_samples;1|]. *)

val load_airline_passengers : unit -> Ndarray.int32_t
(** Load the Airline Passengers monthly totals time series dataset.

    Returns a one-dimensional int32 tensor of monthly passenger counts.

    {2 Parameters}
    - [()]: no arguments.

    {2 Returns}
    - 1-D int32 tensor of shape [|num_samples|]. *)
