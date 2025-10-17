(** Efficient dataset handling for machine learning pipelines This module
    provides composable dataset transformations with support for:
    - Memory-mapped file reading (no OOM on large datasets)
    - Streaming and lazy evaluation
    - Efficient batching and padding
    - Shuffling with configurable buffer sizes
    - Multi-threaded data loading

    All datasets are unified under the polymorphic ['a t] type, with
    specializations via type aliases where helpful (e.g., for tensors). Text
    handling uses [string t] directly for better composability. *)

(** {1 Core Types} *)

type 'a t
(** A dataset of elements of type ['a]. Datasets are lazy, composable, and
    abstract. Use creation functions to build them and transformations to
    modify. *)

type ('elt, 'kind) tensor_dataset = ('elt, 'kind) Rune.t t
(** Generalized dataset of tensors, parameterized over element, kind, and device
*)

type cardinality =
  | Finite of int
  | Unknown
  | Infinite
      (** Cardinality of a dataset: known finite length, unknown (but finite),
          or infinite *)

type element_spec =
  | Unknown
  | Scalar of string  (** e.g., "string" or "int" *)
  | Tensor of int array * string  (** shape * dtype *)
  | Tuple of element_spec list
  | Array of element_spec
      (** Structured description of dataset element types, similar to TF's
          element_spec. Use for type-safe downstream processing. *)

type tokenizer = string -> int array
(** Function type for pluggable tokenizers *)

val whitespace_tokenizer : tokenizer
(** Built-in whitespace tokenizer *)

(** {1 Dataset Creation} *)

val from_array : 'a array -> 'a t
(** [from_array arr] creates a dataset from an in-memory array *)

val from_list : 'a list -> 'a t
(** [from_list lst] creates a dataset from a list *)

val from_seq : 'a Seq.t -> 'a t
(** [from_seq seq] creates a dataset from a sequence *)

val from_tensor : ('elt, 'kind) Rune.t -> ('elt, 'kind) Rune.t t
(** [from_tensor tensor] creates a dataset where each element is a slice of the
    first dimension *)

val from_tensors :
  ('elt, 'kind) Rune.t * ('elt, 'kind) Rune.t ->
  (('elt, 'kind) Rune.t * ('elt, 'kind) Rune.t) t
(** [from_tensors (x, y)] creates a dataset of (input, target) pairs *)

val from_file : (string -> 'a) -> string -> 'a t
(** [from_file parser path] creates a dataset from a file, parsing each line
    with [parser] *)

(** {2 Text Data Sources} *)

val from_text_file :
  ?encoding:[ `UTF8 | `ASCII ] -> ?chunk_size:int -> string -> string t
(** [from_text_file ?encoding ?chunk_size path] creates a memory-mapped text
    dataset yielding lines as strings.
    - [encoding]: Text encoding (default: UTF8)
    - [chunk_size]: Size of chunks to read at once (default: 64KB) The file is
      memory-mapped and read lazily in chunks. *)

val from_text_files :
  ?encoding:[ `UTF8 | `ASCII ] -> ?chunk_size:int -> string list -> string t
(** [from_text_files paths] creates a dataset from multiple text files. Files
    are processed sequentially without loading all into memory. *)

val from_jsonl : ?field:string -> string -> string t
(** [from_jsonl ?field path] reads a JSONL file where each line is a JSON
    object.
    {ul
     {- [field]: Extract text from this field (default: "text") Example JSONL
        format:
        {v
    {"text": "First document", "label": 0}
    {"text": "Second document", "label": 1}
        v}
     }
    } *)

val sliding_window :
  block_size:int ->
  tokenize:(string -> int list) ->
  string list ->
  ((float, Rune.float32_elt) Rune.t * (float, Rune.float32_elt) Rune.t) t
(** [sliding_window ~block_size ~tokenize texts] creates a dataset of sliding
    window context/target pairs for language modeling.

    @param block_size Size of the context window
    @param tokenize Function to convert text to token indices
    @param texts List of input texts (e.g., names for character-level modeling)
    @return Dataset of (context, target) tensor pairs

    Creates all possible sliding windows of size [block_size] from the input
    texts, where each window predicts the next token. Automatically handles
    padding with a special token.

    Example:
    {[
      let dataset =
        sliding_window ~block_size:3
          ~tokenize:(fun s -> encode_chars ~vocab s)
          [ "hello"; "world" ]
      (* Generates windows like: "...h" -> "e" "..he" -> "l" ".hel" -> "l"
         "hell" -> "o" etc. *)
    ]} *)

val from_csv :
  ?separator:char ->
  ?text_column:int ->
  ?has_header:bool ->
  string ->
  string t
(** [from_csv ?separator ?text_column ?has_header path] reads a CSV file and
    returns the text column as a dataset of strings. Rows that do not contain
    the requested column are skipped. *)

val from_csv_with_labels :
  ?separator:char ->
  ?text_column:int ->
  ?has_header:bool ->
  label_column:int ->
  string ->
  (string * string) t
(** [from_csv_with_labels ?separator ?text_column ?has_header ~label_column path]
    reads a CSV file and returns a dataset of (text, label) tuples. Rows missing
    either the text or label column are skipped. *)

val from_text : tokenizer:tokenizer -> string -> int array t
(** [from_text ~tokenizer path] reads a text file and returns a dataset of token
    ID arrays. The file is read as a single document and tokenized. This is
    useful for language modeling tasks where you want the entire document as a
    sequence of tokens. *)

(** {1 Transformations} *)

val map : ('a -> 'b) -> 'a t -> 'b t
(** [map f dataset] applies function [f] to each element *)

val filter : ('a -> bool) -> 'a t -> 'a t
(** [filter pred dataset] keeps only elements satisfying [pred] *)

val flat_map : ('a -> 'b t) -> 'a t -> 'b t
(** [flat_map f dataset] maps and flattens nested datasets *)

val zip : 'a t -> 'b t -> ('a * 'b) t
(** [zip ds1 ds2] pairs corresponding elements. Stops at shorter dataset. *)

val concatenate : 'a t -> 'a t -> 'a t
(** [concatenate ds1 ds2] appends ds2 after ds1 *)

val interleave : 'a t list -> 'a t
(** [interleave datasets] alternates between datasets in round-robin fashion *)

val enumerate : 'a t -> (int * 'a) t
(** [enumerate dataset] adds indices to elements, starting from 0 *)

(** {2 Text Processing} *)

val tokenize :
  tokenizer ->
  ?max_length:int ->
  ?padding:[ `None | `Max of int | `Dynamic ] ->
  ?truncation:bool ->
  ?add_special_tokens:bool ->
  string t ->
  int array t
(** [tokenize tokenizer ?max_length ?padding ?truncation dataset] tokenizes text
    data using the provided tokenizer.
    - [max_length]: Maximum sequence length
    - [padding]: Padding strategy
    - [truncation]: Whether to truncate long sequences
    - [add_special_tokens]: Add <bos>, <eos> tokens *)

val normalize :
  ?lowercase:bool ->
  ?remove_punctuation:bool ->
  ?collapse_whitespace:bool ->
  string t ->
  string t
(** [normalize ?lowercase ?remove_punctuation ?collapse_whitespace dataset]
    applies text normalization *)

(** {2 Batching} *)

val batch :
  ?drop_remainder:bool ->
  int ->
  ((float, 'layout) Rune.t * (float, 'layout) Rune.t) t ->
  ((float, 'layout) Rune.t * (float, 'layout) Rune.t) t
(** [batch ?drop_remainder size dataset] groups tensor pairs into batches and
    automatically stacks them along the batch dimension.
    - [drop_remainder]: Drop final batch if incomplete (default: false)

    This is the primary batching function for ML workflows where datasets
    contain (input, target) tensor pairs. The tensors are automatically stacked
    using [Rune.stack ~axis:0]. *)

val batch_map : ?drop_remainder:bool -> int -> ('a array -> 'b) -> 'a t -> 'b t
(** [batch_map ?drop_remainder size f dataset] groups elements into batches and
    applies function [f] to each batch.

    This is useful for custom batching logic that can't be handled by [batch] or
    [batch_array]. *)

val bucket_by_length :
  ?boundaries:int list ->
  ?batch_sizes:int list ->
  ('a -> int) ->
  'a t ->
  'a array t
(** [bucket_by_length ?boundaries ?batch_sizes length_fn dataset] groups
    elements into buckets by length for efficient padding. Example:
    {[
      bucket_by_length ~boundaries:[ 10; 20; 30 ] ~batch_sizes:[ 32; 16; 8; 4 ]
        (fun text -> String.length text)
        dataset
    ]}
    Creates 4 buckets: <10, 10-20, 20-30, >30 with different batch sizes *)

(** {2 Shuffling and Sampling} *)

val shuffle : ?rng:Rune.Rng.key -> ?buffer_size:int -> 'a t -> 'a t
(** [shuffle ?rng ?buffer_size dataset] randomly shuffles elements.
    - [rng]: Random state for reproducibility (default: self-init)
    - [buffer_size]: Size of shuffle buffer (default: 10000) Uses a buffer to
      shuffle without loading entire dataset in memory. *)

val sample : ?rng:Rune.Rng.key -> ?replacement:bool -> int -> 'a t -> 'a t
(** [sample ?rng ?replacement n dataset] randomly samples n elements *)

val weighted_sample :
  ?rng:Rune.Rng.key -> weights:float array -> int -> 'a t -> 'a t
(** [weighted_sample ?rng ~weights n dataset] samples with given weights *)

(** {2 Iteration Control} *)

val take : int -> 'a t -> 'a t
(** [take n dataset] takes first n elements *)

val skip : int -> 'a t -> 'a t
(** [skip n dataset] skips first n elements *)

val repeat : ?count:int -> 'a t -> 'a t
(** [repeat ?count dataset] repeats dataset. Infinite if count not specified. *)

val window :
  ?shift:int -> ?stride:int -> ?drop_remainder:bool -> int -> 'a t -> 'a array t
(** [window ?shift ?stride ?drop_remainder size dataset] creates sliding
    windows.
    - [shift]: How many elements to shift window (default: size)
    - [stride]: Stride within window (default: 1) Example:
      [window ~shift:1 3 dataset] creates overlapping windows of size 3 *)

(** {2 Caching and Prefetching} *)

val cache : ?directory:string -> 'a t -> 'a t
(** [cache ?directory dataset] caches dataset elements.
    - [directory]: Directory for file cache, in-memory if not specified *)

val prefetch : ?buffer_size:int -> 'a t -> 'a t
(** [prefetch ?buffer_size dataset] pre-fetches elements in background.
    - [buffer_size]: Number of elements to prefetch (default: 2) Uses a separate
      thread to prepare next elements while current is processed. *)

(** {2 Parallel Processing} *)

val parallel_map : ?num_workers:int -> ('a -> 'b) -> 'a t -> 'b t
(** [parallel_map ?num_workers f dataset] applies f using multiple workers.
    - [num_workers]: Number of parallel workers (default: CPU count) *)

val parallel_interleave :
  ?num_workers:int -> ?block_length:int -> ('a -> 'b t) -> 'a t -> 'b t
(** [parallel_interleave ?num_workers ?block_length f dataset] applies f in
    parallel and interleaves results *)

(** {2 High-level Pipeline} *)

val prepare :
  ?shuffle_buffer:int ->
  ?batch_size:int ->
  ?prefetch:int ->
  ?cache:bool ->
  ?drop_remainder:bool ->
  ((float, 'layout) Rune.t * (float, 'layout) Rune.t) t ->
  ((float, 'layout) Rune.t * (float, 'layout) Rune.t) t
(** [prepare ?shuffle_buffer ?batch_size ?prefetch ?cache ?drop_remainder
     dataset] applies common preprocessing pipeline for tensor datasets: 1.
    Cache (if enabled) 2. Shuffle (if buffer size provided) 3. Batch with
    automatic tensor stacking (if batch size provided) 4. Prefetch (if prefetch
    count provided)

    This is the primary pipeline function for ML training data. *)

(** {1 Iteration} *)

val iter : ('a -> unit) -> 'a t -> unit
(** [iter f dataset] applies f to each element for side effects *)

val fold : ('acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
(** [fold f init dataset] folds over dataset elements *)

val to_seq : 'a t -> 'a Seq.t
(** [to_seq dataset] converts to a sequence for lazy iteration *)

val to_list : 'a t -> 'a list
(** [to_list dataset] materializes dataset as list. Warning: loads all into
    memory. *)

val to_array : 'a t -> 'a array
(** [to_array dataset] materializes dataset as array. Warning: loads all into
    memory. *)

(** {1 Dataset Information} *)

val cardinality : 'a t -> cardinality
(** [cardinality dataset] returns the cardinality (finite length, unknown, or
    infinite) *)

val element_spec : 'a t -> element_spec
(** [element_spec dataset] returns a structured description of element types *)

(** {1 Dataset Control} *)

val reset : 'a t -> unit
(** [reset dataset] resets the dataset to its initial state if supported. This
    makes it possible to iterate a dataset multiple times (e.g., across training
    epochs). If the dataset does not support reset, this is a no-op. *)

(** {1 Common Pipelines} *)

val text_classification_pipeline :
  ?tokenizer:tokenizer ->
  ?max_length:int ->
  ?batch_size:int ->
  ?shuffle_buffer:int ->
  ?num_workers:int ->
  string t ->
  (int32, Rune.int32_elt) Rune.t t
(** Pre-configured pipeline for text classification tasks. Returns batched token
    tensors ready for embedding layers. *)

val language_model_pipeline :
  ?tokenizer:tokenizer ->
  ?sequence_length:int ->
  ?batch_size:int ->
  ?shuffle_buffer:int ->
  ?num_workers:int ->
  string t ->
  ((int32, Rune.int32_elt) Rune.t * (int32, Rune.int32_elt) Rune.t) t
(** Pre-configured pipeline for language modeling. Returns batched (input,
    target) tensor pairs ready for training. *)

(** {1 Examples}
    {[
      (* Load and process text data *)
      let dataset =
        from_text_file "data/corpus.txt"
        |> tokenize whitespace_tokenizer ~max_length:512
        |> shuffle ~buffer_size:10000
        |> batch 32
        |> prefetch ~buffer_size:2
      (* Iterate through batches *)
      dataset
      |> iter (fun batch ->
             let tensor = process_batch batch in
             train_step model tensor)

      (* Multi-file dataset with bucketing *)
      let dataset =
        from_text_files [ "shard1.txt"; "shard2.txt"; "shard3.txt" ]
        |> normalize ~lowercase:true
        |> tokenize whitespace_tokenizer
        |> bucket_by_length ~boundaries:[ 100; 200; 300 ]
             ~batch_sizes:[ 64; 32; 16; 8 ] Array.length
        |> prefetch

      (* Parallel processing *)
      let dataset =
        from_jsonl "data.jsonl"
        |> parallel_map ~num_workers:4 preprocess
        |> cache ~directory:"/tmp/cache"
        |> shuffle ~buffer_size:50000
        |> batch 128

      (* Custom tokenizer and tensor batching *)
      let custom_tok = fun s -> (* ... *) [|1;2;3|] in
      let tensor_ds =
        from_text_file "texts.txt"
        |> tokenize custom_tok
        |> batch_map 32 (Rune.stack ~axis:0)
    ]} *)
