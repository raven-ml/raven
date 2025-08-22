(** Ready-to-use datasets for machine learning

    This module provides popular datasets with built-in support for:
    - Streaming and lazy loading (no OOM on large datasets)
    - Automatic caching and prefetching
    - Configurable preprocessing pipelines
    - Efficient batching and shuffling *)

(** {1 Core Types} *)

type ('elt, 'kind, 'dev) tensor_dataset = 
  ('elt, 'kind, 'dev) Rune.t Kaun_dataset.t

(** {1 Vision Datasets} *)

val mnist :
  ?train:bool ->                    (* default: true *)
  ?flatten:bool ->                   (* default: false - keeps 28x28 shape *)
  ?normalize:bool ->                 (* default: true - scales to [0,1] *)
  ?data_format:[ `NCHW | `NHWC ] ->  (* default: `NCHW *)
  ?cache_dir:string ->               (* Optional directory for caching *)
  device:'dev Kaun.device ->
  unit ->
  ((Bigarray.float32_elt, 'dev) Kaun.tensor 
   * (Bigarray.float32_elt, 'dev) Kaun.tensor) Kaun_dataset.t
(** MNIST handwritten digits dataset.
    Returns a dataset of (images, labels) pairs. *)

val cifar10 :
  ?train:bool ->
  ?normalize:bool ->                 (* default: true - ImageNet normalization *)
  ?data_format:[ `NCHW | `NHWC ] ->
  ?augmentation:bool ->              (* default: false - random crops/flips *)
  ?cache_dir:string ->
  device:'dev Kaun.device ->
  unit ->
  ((Bigarray.float32_elt, 'dev) Kaun.tensor 
   * (Bigarray.float32_elt, 'dev) Kaun.tensor) Kaun_dataset.t
(** CIFAR-10 image classification dataset *)

val fashion_mnist :
  ?train:bool ->
  ?flatten:bool ->
  ?normalize:bool ->
  ?data_format:[ `NCHW | `NHWC ] ->
  ?cache_dir:string ->
  device:'dev Kaun.device ->
  unit ->
  ((Bigarray.float32_elt, 'dev) Kaun.tensor 
   * (Bigarray.float32_elt, 'dev) Kaun.tensor) Kaun_dataset.t
(** Fashion-MNIST clothing classification dataset *)

(** {1 Text Datasets} *)

val imdb :
  ?train:bool ->
  ?tokenizer:Kaun_dataset.tokenizer ->  (* default: whitespace_tokenizer *)
  ?max_length:int ->                    (* default: 512 *)
  ?cache_dir:string ->
  device:'dev Kaun.device ->
  unit ->
  (int array * (Bigarray.float32_elt, 'dev) Kaun.tensor) Kaun_dataset.t
(** IMDB movie review sentiment dataset.
    Returns (token_ids, labels) where labels are 0 (negative) or 1 (positive) *)

val wikitext :
  ?dataset_name:[ `Wikitext2 | `Wikitext103 ] ->  (* default: `Wikitext2 *)
  ?tokenizer:Kaun_dataset.tokenizer ->
  ?sequence_length:int ->                         (* default: 1024 *)
  ?cache_dir:string ->
  device:'dev Kaun.device ->
  unit ->
  (int array * int array) Kaun_dataset.t
(** WikiText language modeling dataset.
    Returns (input_ids, target_ids) for next-token prediction *)

(** {1 Structured Data} *)

val iris :
  ?normalize:bool ->
  ?train_split:float ->                (* default: 0.8 *)
  ?shuffle_seed:int ->
  device:'dev Kaun.device ->
  unit ->
  ((Bigarray.float32_elt, 'dev) Kaun.tensor 
   * (Bigarray.float32_elt, 'dev) Kaun.tensor) Kaun_dataset.t
(** Iris flower classification dataset *)

val boston_housing :
  ?normalize:bool ->
  ?train_split:float ->
  device:'dev Kaun.device ->
  unit ->
  ((Bigarray.float32_elt, 'dev) Kaun.tensor 
   * (Bigarray.float32_elt, 'dev) Kaun.tensor) Kaun_dataset.t
(** Boston housing price regression dataset *)

(** {1 Dataset Utilities} *)

val download_and_extract :
  url:string ->
  cache_dir:string ->
  ?extract:bool ->              (* default: true for .tar.gz, .zip files *)
  unit ->
  string
(** Download a dataset file and optionally extract it.
    Returns the path to the downloaded/extracted data. *)

val train_test_split :
  ?test_size:float ->           (* default: 0.2 *)
  ?shuffle:bool ->              (* default: true *)
  ?seed:int ->
  'a Kaun_dataset.t ->
  'a Kaun_dataset.t * 'a Kaun_dataset.t
(** Split a dataset into training and test sets *)

(** {1 Pre-configured Pipelines} *)

val vision_pipeline :
  ?batch_size:int ->           (* default: 32 *)
  ?shuffle_buffer:int ->       (* default: 10000 *)
  ?prefetch_buffer:int ->      (* default: 2 *)
  ?num_workers:int ->          (* default: 4 *)
  ((Bigarray.float32_elt, 'dev) Kaun.tensor 
   * (Bigarray.float32_elt, 'dev) Kaun.tensor) Kaun_dataset.t ->
  ((Bigarray.float32_elt, 'dev) Kaun.tensor 
   * (Bigarray.float32_elt, 'dev) Kaun.tensor) Kaun_dataset.t
(** Standard vision dataset pipeline with batching, shuffling, and prefetching *)

val text_pipeline :
  ?batch_size:int ->
  ?shuffle_buffer:int ->
  ?bucket_boundaries:int list ->   (* For dynamic batching by length *)
  ?prefetch_buffer:int ->
  ?num_workers:int ->
  (int array * 'label) Kaun_dataset.t ->
  (int array array * 'label array) Kaun_dataset.t
(** Standard text dataset pipeline with bucketing for efficient padding *)

(** {1 Examples}

    {[
      (* Simple MNIST training loop *)
      let dataset = 
        Kaun_datasets.mnist ~train:true ~device ()
        |> Kaun_dataset.shuffle ~buffer_size:60000
        |> Kaun_dataset.batch 32
        |> Kaun_dataset.prefetch ~buffer_size:2
      in
      
      Kaun_dataset.iter (fun (x_batch, y_batch) ->
        let loss = train_step model x_batch y_batch in
        Printf.printf "Loss: %f\n" (Kaun.Ops.to_float loss)
      ) dataset
      
      (* Text classification with IMDB *)
      let dataset =
        Kaun_datasets.imdb ~train:true ~max_length:256 ~device ()
        |> Kaun_datasets.text_pipeline ~batch_size:16 
             ~bucket_boundaries:[50; 100; 200]
      in
      
      (* Using train/test split *)
      let all_data = load_custom_dataset () in
      let train_data, test_data = 
        Kaun_datasets.train_test_split ~test_size:0.2 all_data
      in
    ]}
*)