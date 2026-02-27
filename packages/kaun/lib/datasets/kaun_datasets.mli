(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dataset loaders for kaun.

    Datasets are downloaded on demand and cached locally under
    [$RAVEN_CACHE_ROOT/datasets/] (or [$XDG_CACHE_HOME/raven/datasets/]). *)

val mnist :
  ?fashion:bool ->
  ?normalize:bool ->
  ?data_format:[ `NCHW | `NHWC ] ->
  unit ->
  (Nx.float32_t * Nx.float32_t) Kaun.Data.t
  * (Nx.float32_t * Nx.float32_t) Kaun.Data.t
(** [mnist ()] is [(train, test)] where each is a {!Kaun.Data.t} pipeline
    yielding [(image, label)] pairs.

    Images are float32 in \[0, 1\] (when [normalize] is [true], the default).
    Labels are float32 class indices.

    [fashion] selects Fashion-MNIST when [true]. Defaults to [false].
    [data_format] defaults to [`NCHW].

    Shapes per element:
    - [`NCHW]: image [|1; 28; 28|], label [||]
    - [`NHWC]: image [|28; 28; 1|], label [||]

    Raises [Failure] on download or parsing errors. *)

val cifar10 :
  ?normalize:bool ->
  ?data_format:[ `NCHW | `NHWC ] ->
  unit ->
  (Nx.float32_t * Nx.float32_t) Kaun.Data.t
  * (Nx.float32_t * Nx.float32_t) Kaun.Data.t
(** [cifar10 ()] is [(train, test)] where each is a {!Kaun.Data.t} pipeline
    yielding [(image, label)] pairs.

    Images are float32 in \[0, 1\] (when [normalize] is [true], the default).
    Labels are float32 class indices (0–9: airplane, automobile, bird, cat,
    deer, dog, frog, horse, ship, truck).

    [data_format] defaults to [`NCHW].

    Shapes per element:
    - [`NCHW]: image [|3; 32; 32|], label [||]
    - [`NHWC]: image [|32; 32; 3|], label [||]

    Raises [Failure] on download, extraction, or parsing errors. *)
