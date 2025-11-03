(** Byte Pair Encoding tokenization algorithm.

    This module implements the BPE subword tokenization algorithm used by GPT-2,
    GPT-3, and related models. BPE iteratively merges the most frequent adjacent
    character or subword pairs to build a vocabulary of subword units.

    {1 Algorithm Overview}

    BPE tokenization proceeds in phases:

    1. Initialize: Split input into individual characters or bytes 2. Apply
    merges: Iteratively merge adjacent pairs according to learned merge rules 3.
    Fallback: Handle unknown characters via unknown token or byte-level fallback
    4. Cache: Store tokenization results for repeated inputs

    The merge rules form a priority queue based on training frequency. Higher
    priority merges (learned earlier) are applied first.

    {1 Key Data Structures}

    The implementation uses several internal structures for efficiency:

    - [merge_map]: Maps character/token pairs to merge priorities (stored as
      [IntPairMap])
    - [word]: Mutable doubly-linked list of symbols for in-place merging
    - [symbol]: Node in the linked list with character ID, length, and prev/next
      pointers
    - [cache]: Optional LRU cache mapping strings to pre-tokenized word
      structures

    {1 Performance Characteristics}

    - Tokenization: O(n * m log m) where n = input length, m = average word
      length
    - Merge application: O(m²) worst case for pathological merge patterns
    - Cache lookup: O(1) with cache enabled
    - Memory: O(vocab_size + merge_count + cache_capacity)

    The cache significantly improves performance for repeated text, but adds
    memory overhead. Set [cache_capacity = 0] to disable.

    {1 Configuration Options}

    - [dropout]: Randomly skip merges during tokenization for data augmentation
      (0.0 to 1.0)
    - [fuse_unk]: Merge consecutive unknown tokens into single token
    - [byte_fallback]: Use byte-level encoding for unknown characters instead of
      unknown token
    - [ignore_merges]: Skip merge rules for words already in vocabulary (faster
      for word-level models)
    - [continuing_subword_prefix]: Mark non-initial subwords (e.g., "##" in
      BERT)
    - [end_of_word_suffix]: Mark word-final tokens (e.g., "</w>" in some
      implementations)

    {1 Compatibility}

    Matches HuggingFace Tokenizers BPE implementation. Token IDs use OCaml
    integers (int) instead of Rust u32. Vocabularies loaded from HuggingFace
    JSON format are compatible.

    {1 Implementation Notes}

    The core merging algorithm uses a priority queue to select the next merge.
    Symbols are stored in a mutable array-backed linked list to avoid allocation
    overhead during merges. After applying a merge, neighboring pairs are
    re-evaluated for potential merges.

    Dropout is applied probabilistically during merge selection. When dropout
    triggers, the merge is skipped and the algorithm moves to the next
    candidate. This creates subword variations useful for data augmentation. *)

(** {1 Core Types} *)

type t
(** BPE tokenization model.

    Contains vocabulary mappings (string ↔ int), merge rules as a priority map,
    optional tokenization cache, and configuration flags. The model is
    internally mutable due to the cache, but vocabulary and merges are immutable
    after creation. *)

type vocab = (string, int) Hashtbl.t
(** Vocabulary mapping tokens to integer IDs.

    Token strings include both base vocabulary (characters or bytes) and merged
    subwords learned during training. IDs are assigned sequentially during
    vocabulary construction. *)

type merges = (string * string) list
(** Merge rules as ordered list of (token_a, token_b) pairs.

    Order matters: earlier merges have higher priority. During tokenization, the
    algorithm applies merges in priority order (based on position in this list).
    Each merge combines two adjacent tokens into a single token. *)

type config = {
  vocab : vocab;
  merges : merges;
  cache_capacity : int;
  dropout : float option;
  unk_token : string option;
  continuing_subword_prefix : string option;
  end_of_word_suffix : string option;
  fuse_unk : bool;
  byte_fallback : bool;
  ignore_merges : bool;
}
(** BPE model configuration.

    @param cache_capacity
      Maximum cache entries (0 disables). LRU eviction when full.
    @param dropout
      Merge skip probability for data augmentation (None or Some 0.0 to 1.0).
    @param unk_token Unknown token string for out-of-vocabulary characters.
    @param continuing_subword_prefix
      Prefix added to non-initial subwords (e.g., "##").
    @param end_of_word_suffix Suffix added to word-final tokens (e.g., "</w>").
    @param fuse_unk Merge consecutive unknown tokens into one.
    @param byte_fallback
      Encode unknown characters as bytes instead of using unknown token.
    @param ignore_merges
      Skip merge rules for words present in vocabulary (optimization for
      word-level models). *)

(** {1 Model Creation} *)

val create : config -> t
(** [create config] constructs a BPE model from configuration.

    Builds internal data structures: reverse vocabulary (int → string), merge
    priority map (pair → rank), and optional cache. The merge priority map
    assigns each merge pair its index in the merges list, used to resolve
    conflicts when multiple merges are possible.

    @raise Invalid_argument
      if vocabulary or merges are empty, or if cache capacity is negative. *)

val from_files : vocab_file:string -> merges_file:string -> t
(** [from_files ~vocab_file ~merges_file] loads a BPE model from files.

    Expects HuggingFace format:
    - [vocab_file]: JSON dictionary mapping tokens to IDs
    - [merges_file]: Text file with one merge per line ("tokenA tokenB")

    Merges are prioritized by line order. Creates model with default
    configuration (no dropout, no unknown token, no byte fallback, cache
    capacity 10000).

    @raise Sys_error if files cannot be read.
    @raise Yojson.Json_error if vocab JSON is malformed. *)

val default : unit -> t
(** [default ()] creates an empty BPE model with default configuration.

    Useful as a starting point for training. Default settings: empty vocabulary,
    empty merges, cache capacity 10000, no dropout, no unknown token, no
    continuing subword prefix, no end-of-word suffix, fuse_unk = false,
    byte_fallback = false, ignore_merges = false. *)

(** {1 Tokenization} *)

type token = { id : int; value : string; offsets : int * int }
(** Token representation with metadata.

    - [id]: Vocabulary index
    - [value]: Token string (may include prefix/suffix markers)
    - [offsets]: Character span (start, end) in original text *)

val tokenize : t -> string -> token list
(** [tokenize model text] encodes text into tokens using BPE algorithm.

    Algorithm: 1. Check cache for previously tokenized text (if cache enabled)
    2. Split text into initial character or byte sequence 3. Build symbol linked
    list for efficient merging 4. Apply merges iteratively by priority until no
    more merges possible 5. Convert symbols to tokens with offsets 6. Cache
    result (if cache enabled)

    Dropout (if configured): Each merge is skipped with probability [dropout],
    creating subword variations.

    Unknown characters: Handled via [unk_token] (if [byte_fallback = false]) or
    individual bytes (if [byte_fallback = true]).

    Performance: O(n * m log m) where n = text length, m = average token count.
    Cache provides O(1) for repeated inputs. *)

val token_to_id : t -> string -> int option
(** [token_to_id model token] looks up token ID in vocabulary.

    Returns [None] if token not found. Exact string match required, including
    any prefix/suffix markers. *)

val id_to_token : t -> int -> string option
(** [id_to_token model id] looks up token string by ID.

    Returns [None] if ID out of bounds [0, vocab_size). *)

(** {1 Vocabulary Management} *)

val get_vocab : t -> (string * int) list
(** [get_vocab model] returns vocabulary as (token, id) pairs.

    Order is unspecified. Includes all base tokens and learned merges. *)

val get_vocab_size : t -> int
(** [get_vocab_size model] returns vocabulary size.

    Equals number of unique tokens, including base characters, merges, and
    special tokens. *)

val get_unk_token : t -> string option
(** [get_unk_token model] retrieves configured unknown token.

    Returns [None] if no unknown token set or if [byte_fallback = true]. *)

val get_continuing_subword_prefix : t -> string option
(** [get_continuing_subword_prefix model] retrieves subword continuation prefix.

    Returns [None] if not configured. *)

val get_end_of_word_suffix : t -> string option
(** [get_end_of_word_suffix model] retrieves word-final token suffix.

    Returns [None] if not configured. *)

val get_merges : t -> (string * string) list
(** [get_merges model] returns the list of merge rules in priority order. Each
    pair [(token_a, token_b)] represents a merge learned during training,
    ordered by the rank assigned when the model was constructed. *)

(** {1 Cache Management} *)

val clear_cache : t -> unit
(** [clear_cache model] removes all cached tokenization results.

    No-op if cache disabled. Use to free memory or ensure fresh tokenization
    after vocabulary changes. *)

val resize_cache : t -> int -> unit
(** [resize_cache model capacity] changes cache capacity.

    Current implementation clears the cache rather than resizing. Set to 0 to
    disable cache (clears existing entries).

    @raise Invalid_argument if capacity is negative. *)

(** {1 Serialization} *)

val save : t -> path:string -> ?name:string -> unit -> unit
(** [save model ~path ?name ()] writes model to files.

    Creates two files in [path] directory:
    - [<name>-vocab.json] or [vocab.json]: Vocabulary as JSON dictionary
    - [<name>-merges.txt] or [merges.txt]: Merges as text file (one per line)

    Overwrites existing files. Format matches HuggingFace tokenizers.

    @raise Sys_error if directory is not writable. *)

val read_files : vocab_file:string -> merges_file:string -> vocab * merges
(** [read_files ~vocab_file ~merges_file] loads vocabulary and merges from
    files.

    Expects HuggingFace format:
    - [vocab_file]: JSON dictionary \{"token": id, ...\}
    - [merges_file]: Text file with "tokenA tokenB" per line

    Returns (vocab hashtable, merges list). Does not construct full model; use
    {!from_files} for that.

    @raise Sys_error if files cannot be read.
    @raise Yojson.Json_error if vocab JSON is malformed. *)

(** {1 Training} *)

val train :
  min_frequency:int ->
  vocab_size:int ->
  show_progress:bool ->
  special_tokens:string list ->
  limit_alphabet:int option ->
  initial_alphabet:char list ->
  continuing_subword_prefix:string option ->
  end_of_word_suffix:string option ->
  max_token_length:int option ->
  string list ->
  t option ->
  t * string list
(** [train ~min_frequency ~vocab_size ~show_progress ~special_tokens
     ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
     ~end_of_word_suffix ~max_token_length texts existing] learns BPE merges
    from training corpus.

    Algorithm: 1. Initialize vocabulary with [initial_alphabet] and
    [special_tokens] 2. Count character pairs in [texts] 3. Iteratively merge
    most frequent pair until [vocab_size] reached 4. Filter merges with
    frequency below [min_frequency] 5. Optionally limit alphabet size to most
    frequent [limit_alphabet] characters 6. Build BPE model with learned
    vocabulary and merges

    @param min_frequency
      Minimum occurrence count for a merge to be learned. Higher values reduce
      vocabulary size and training time.
    @param vocab_size Target vocabulary size (including initial alphabet).
    @param show_progress Print progress during training (not yet implemented).
    @param special_tokens
      Tokens to include in vocabulary without splitting (e.g., "[PAD]",
      "[CLS]").
    @param limit_alphabet
      Restrict base alphabet to this many most-frequent characters. None means
      no limit.
    @param initial_alphabet Starting character set before learning merges.
    @param continuing_subword_prefix
      Prefix for non-initial subwords in trained model.
    @param end_of_word_suffix Suffix for word-final tokens in trained model.
    @param max_token_length
      Maximum characters per token. None means no limit. Limits merge
      combinations.
    @param texts Training corpus as list of strings.
    @param existing
      Optional existing model to extend (vocabulary and merges are added, not
      replaced). Currently unused in implementation.
    @return (trained model, special tokens list).

    @raise Invalid_argument if vocab_size is less than initial alphabet size. *)
