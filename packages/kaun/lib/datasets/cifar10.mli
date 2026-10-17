(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The CIFAR-10 loader behind {!Kaun_datasets.cifar10}.

    The binary archive is downloaded on first use and extracted in the
    dataset's cache directory ({!Dataset_utils.get_cache_dir}), [cifar10/].
    The training split is its five data batches and the test split its test
    batch, each of 10,000 examples. *)

val load :
  unit ->
  ((int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Genarray.t
  * (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t)
  * ((int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Genarray.t
    * (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t)
(** [load ()] is [((train_images, train_labels), (test_images,
    test_labels))]. Images have shape [[|n; 3; 32; 32|]] (example, channel,
    row, column) in levels \[[0];[255]\], and labels are class indices. It
    blocks while it downloads, extracts and reads the files.

    Raises [Failure] if the archive cannot be obtained or extracted, or a
    batch does not hold exactly 10,000 entries. *)
