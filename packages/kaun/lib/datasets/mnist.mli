(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The MNIST and Fashion-MNIST loader behind {!Kaun_datasets.mnist}.

    The four IDX files of a dataset are downloaded gzipped on first use and
    kept decompressed in its cache directory ({!Dataset_utils.get_cache_dir}),
    [mnist/] or [fashion-mnist/]. *)

val load :
  fashion_mnist:bool ->
  ((int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array3.t
  * (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t)
  * ((int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array3.t
    * (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t)
(** [load ~fashion_mnist] is [((train_images, train_labels), (test_images,
    test_labels))] of Fashion-MNIST if [fashion_mnist] is [true] and of
    MNIST otherwise. Images are indexed by example, row and column, in
    levels \[[0];[255]\], and labels are class indices. It blocks while it
    downloads and reads the files.

    Raises [Failure] if a file cannot be obtained, or has the wrong magic
    number or a length its header does not give. *)
