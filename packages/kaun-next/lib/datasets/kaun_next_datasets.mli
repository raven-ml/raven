(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dataset loaders for kaun-next.

    Each loader returns plain Nx tensors — [(train_x, train_y, test_x, test_y)]
    with axis 0 indexing examples — ready for {!Kaun_next.Data.batches2}:

    {[
    let train_x, train_y, test_x, test_y = Kaun_next_datasets.mnist () in
    let batches = Data.batches2 ~shuffle:true ~batch_size:128 (train_x, train_y)
    ]}

    Datasets are downloaded on demand and cached locally under
    [$RAVEN_CACHE_ROOT/datasets/] (or [$XDG_CACHE_HOME/raven/datasets/]). *)

val mnist :
  ?fashion:bool ->
  ?normalize:bool ->
  ?data_format:[ `NCHW | `NHWC ] ->
  unit ->
  Nx.float32_t * Nx.int32_t * Nx.float32_t * Nx.int32_t
(** [mnist ()] is [(train_x, train_y, test_x, test_y)]: 60,000 training and
    10,000 test examples.

    Images are float32 in \[[0];[1]\] (when [normalize] is [true], the default).
    Labels are int32 class indices.

    [fashion] selects Fashion-MNIST when [true]. Defaults to [false].
    [data_format] defaults to [`NCHW].

    Tensor shapes:
    - [`NCHW]: images [[N; 1; 28; 28]], labels [[N]]
    - [`NHWC]: images [[N; 28; 28; 1]], labels [[N]]

    Raises [Failure] on download or parsing errors. *)

val cifar10 :
  ?normalize:bool ->
  ?data_format:[ `NCHW | `NHWC ] ->
  unit ->
  Nx.float32_t * Nx.int32_t * Nx.float32_t * Nx.int32_t
(** [cifar10 ()] is [(train_x, train_y, test_x, test_y)]: 50,000 training and
    10,000 test examples.

    Images are float32 in \[[0];[1]\] (when [normalize] is [true], the default).
    Labels are int32 class indices (0--9: airplane, automobile, bird, cat, deer,
    dog, frog, horse, ship, truck).

    [data_format] defaults to [`NCHW].

    Tensor shapes:
    - [`NCHW]: images [[N; 3; 32; 32]], labels [[N]]
    - [`NHWC]: images [[N; 32; 32; 3]], labels [[N]]

    Raises [Failure] on download, extraction, or parsing errors. *)
