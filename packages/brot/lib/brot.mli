(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tokenization for OCaml.

    Brot tokenizes text into token IDs for language models and reverses the
    process. Tokenization proceeds through configurable stages:

    + {e Normalization}: clean and normalize text (lowercase, accent removal,
      Unicode normalization). See {!Normalizer}.
    + {e Pre-tokenization}: split text into words or sub-words. See
      {!Pre_tokenizer}.
    + {e Tokenization}: apply vocabulary-based encoding (BPE, WordPiece,
      Unigram, word-level, or character-level).
    + {e Post-processing}: add special tokens and set type IDs. See
      {!Post_processor}.
    + {e Padding/Truncation}: adjust sequence lengths for batching.

    Each stage is optional and configurable. Open the module to use it, it
    defines only modules in your scope.

    {1:quick_start Quick start}

    Load a pretrained tokenizer:
    {[
      let tokenizer = Brot.from_file "tokenizer.json" |> Result.get_ok in
      let encoding = Brot.encode tokenizer "Hello world!" in
      let _ids = Encoding.ids encoding
    ]}

    Create a BPE tokenizer from scratch:
    {[
      let tokenizer =
        Brot.bpe
          ~vocab:[("hello", 0); ("world", 1); ("[PAD]", 2)]
          ~merges:[]
          ()
      in
      let encoding = Brot.encode tokenizer "hello world" in
      let _text = Brot.decode tokenizer (Encoding.ids encoding)
    ]}

    Train a new tokenizer:
    {[
      let texts = [ "Hello world"; "How are you?"; "Hello again" ] in
      let tokenizer =
        Brot.train_bpe (`Seq (List.to_seq texts)) ~vocab_size:1000
      in
      Brot.save_pretrained tokenizer ~path:"./my_tokenizer"
    ]}

    {!modules:Encoding Normalizer Pre_tokenizer Post_processor Decoder} *)

module Normalizer = Normalizer
(** Text normalization. *)

module Pre_tokenizer = Pre_tokenizer
(** Pre-tokenization. *)

module Post_processor = Post_processor
(** Post-processing. *)

module Decoder = Decoder
(** Token decoding. *)

module Encoding = Encoding
(** Tokenization encodings. *)

(** {1:types Types} *)

type t
(** The type for tokenizers. Immutable after creation. *)

type direction = [ `Left | `Right ]
(** The type for padding and truncation directions. [`Left] operates at the
    beginning of the sequence, [`Right] at the end. *)

type special = {
  token : string;  (** The token text (e.g., ["<pad>"], ["<unk>"]). *)
  single_word : bool;  (** Whether this token must match whole words only. *)
  lstrip : bool;  (** Whether to strip whitespace on the left. *)
  rstrip : bool;  (** Whether to strip whitespace on the right. *)
  normalized : bool;  (** Whether to apply normalization to this token. *)
}
(** The type for special token configurations.

    Special tokens are never split during tokenization and can be skipped during
    decoding. Token IDs are assigned automatically when added to the vocabulary.
    The semantic role (pad, unk, bos, etc.) is contextual, not encoded in the
    type. *)

type pad_length = [ `Batch_longest | `Fixed of int | `To_multiple of int ]
(** The type for padding length strategies.

    - [`Batch_longest]: pad to the longest sequence in the batch.
    - [`Fixed n]: pad every sequence to exactly [n] tokens.
    - [`To_multiple n]: pad to the smallest multiple of [n] that is at least the
      sequence length. *)

type padding = {
  length : pad_length;
  direction : direction;
  pad_id : int option;
  pad_type_id : int option;
  pad_token : string option;
}
(** The type for padding configurations.

    When [pad_id], [pad_type_id], or [pad_token] are [None], the tokenizer's
    configured padding token is used. Raises [Invalid_argument] at padding time
    if no padding token is configured and these fields are [None]. *)

type truncation = { max_length : int; direction : direction }
(** The type for truncation configurations. Sequences exceeding [max_length]
    tokens are trimmed from the given [direction]. *)

type data = [ `Files of string list | `Seq of string Seq.t ]
(** The type for training data sources.

    - [`Files paths]: read training text from files, one line per example.
    - [`Seq seq]: use a sequence of strings. *)

val special :
  ?single_word:bool ->
  ?lstrip:bool ->
  ?rstrip:bool ->
  ?normalized:bool ->
  string ->
  special
(** [special token] is a special token configuration for [token].

    [single_word] defaults to [false]. [lstrip] and [rstrip] default to [false].
    [normalized] defaults to [false]. *)

val padding :
  ?direction:direction ->
  ?pad_id:int ->
  ?pad_type_id:int ->
  ?pad_token:string ->
  pad_length ->
  padding
(** [padding length] is a padding configuration for the given [length] strategy.

    [direction] defaults to [`Right]. Other fields default to [None] (falls back
    to the tokenizer's configured padding token). *)

val truncation : ?direction:direction -> int -> truncation
(** [truncation max_length] is a truncation configuration limiting sequences to
    [max_length] tokens. [direction] defaults to [`Right]. *)

(** {1:constructors Constructors} *)

val bpe :
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?specials:special list ->
  ?bos_token:string ->
  ?eos_token:string ->
  ?pad_token:string ->
  ?unk_token:string ->
  ?vocab:(string * int) list ->
  ?merges:(string * string) list ->
  ?cache_capacity:int ->
  ?dropout:float ->
  ?continuing_subword_prefix:string ->
  ?end_of_word_suffix:string ->
  ?fuse_unk:bool ->
  ?byte_fallback:bool ->
  ?ignore_merges:bool ->
  unit ->
  t
(** [bpe ()] is a BPE (Byte Pair Encoding) tokenizer. Used by GPT-2, GPT-3,
    RoBERTa.

    - [normalizer]: text normalization. Default: none.
    - [pre]: pre-tokenization strategy. Default: none.
    - [post]: post-processor for special tokens. Default: none.
    - [decoder]: decoding strategy. Default: none.
    - [specials]: special tokens to add to vocabulary. Default: [[]].
    - [bos_token], [eos_token], [pad_token]: role markers; added to vocabulary
      if not already present. Default: none.
    - [unk_token]: token for unknown characters. Configures both the role and
      the BPE model's unknown handling. Default: none.
    - [vocab]: initial vocabulary as [(token, id)] pairs. Default: [[]].
    - [merges]: merge rules as [(left, right)] pairs learned during training.
      Default: [[]].
    - [cache_capacity]: LRU cache size for tokenization results. Default:
      [10000].
    - [dropout]: probability \[[0]; [1]\] of skipping merges (data
      augmentation). Default: none (no dropout).
    - [continuing_subword_prefix]: prefix for non-initial subwords (e.g.,
      ["##"]). Default: none.
    - [end_of_word_suffix]: suffix marking word boundaries (e.g., ["</w>"]).
      Default: none.
    - [fuse_unk]: merge consecutive unknown tokens. Default: [false].
    - [byte_fallback]: use byte-level fallback (["<0x00>"]) instead of unknown
      token. Default: [false].
    - [ignore_merges]: skip merge application (character-level output). Default:
      [false]. *)

val wordpiece :
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?specials:special list ->
  ?bos_token:string ->
  ?eos_token:string ->
  ?pad_token:string ->
  ?unk_token:string ->
  ?vocab:(string * int) list ->
  ?continuing_subword_prefix:string ->
  ?max_input_chars_per_word:int ->
  unit ->
  t
(** [wordpiece ()] is a WordPiece tokenizer. Used by BERT, DistilBERT, Electra.

    WordPiece uses a greedy longest-match-first algorithm to split words into
    subword pieces prefixed with a continuation marker (e.g., ["running"]
    becomes [["run"; "##ning"]]).

    - [vocab]: initial vocabulary as [(token, id)] pairs. Default: [[]].
    - [unk_token]: token for out-of-vocabulary words. Default: ["[UNK]"].
    - [continuing_subword_prefix]: prefix for non-initial subwords. Default:
      ["##"].
    - [max_input_chars_per_word]: words longer than this are replaced with
      [unk_token]. Default: [100].

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [specials],
    [bos_token], [eos_token], [pad_token]) are as in {!bpe}. *)

val word_level :
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?specials:special list ->
  ?bos_token:string ->
  ?eos_token:string ->
  ?pad_token:string ->
  ?unk_token:string ->
  ?vocab:(string * int) list ->
  unit ->
  t
(** [word_level ()] is a word-level tokenizer.

    Maps each word directly to a token ID. No subword splitting is performed.
    Words not in vocabulary map to [unk_token].

    {b Note.} When [pre] is not provided, {!Pre_tokenizer.whitespace} is used by
    default.

    - [vocab]: initial vocabulary as [(word, id)] pairs. Default: [[]].
    - [unk_token]: token for out-of-vocabulary words. Default: ["<unk>"].

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [specials],
    [bos_token], [eos_token], [pad_token]) are as in {!bpe}. *)

val unigram :
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?specials:special list ->
  ?bos_token:string ->
  ?eos_token:string ->
  ?pad_token:string ->
  ?unk_token:string ->
  ?vocab:(string * float) list ->
  unit ->
  t
(** [unigram ()] is a Unigram tokenizer. Used by AlBERT, T5, mBART.

    Unigram uses probabilistic segmentation to find optimal subword splits based
    on token log-probabilities.

    - [vocab]: initial vocabulary as [(token, score)] pairs where scores are
      negative log probabilities. Default: [[]].
    - [unk_token]: token for unknown characters. Default: none.

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [specials],
    [bos_token], [eos_token], [pad_token]) are as in {!bpe}. *)

val chars :
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?specials:special list ->
  ?bos_token:string ->
  ?eos_token:string ->
  ?pad_token:string ->
  ?unk_token:string ->
  unit ->
  t
(** [chars ()] is a character-level tokenizer.

    Each byte in the input becomes a separate token with ID equal to its ordinal
    value. No vocabulary is required.

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [specials],
    [bos_token], [eos_token], [pad_token]) are as in {!bpe}. *)

val from_model_file :
  vocab:string ->
  ?merges:string ->
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?specials:special list ->
  ?bos_token:string ->
  ?eos_token:string ->
  ?pad_token:string ->
  ?unk_token:string ->
  unit ->
  t
(** [from_model_file ~vocab ()] loads a tokenizer from HuggingFace model files.

    The model type is inferred from the arguments: if [merges] is provided, a
    BPE tokenizer is created; otherwise WordPiece.

    - [vocab]: path to vocabulary file ([vocab.json]). Expected format: JSON
      object mapping tokens to IDs ([{"hello": 0, "world": 1}]).
    - [merges]: path to merges file ([merges.txt]). One merge per line as
      space-separated token pairs. Lines starting with ["#version"] are skipped.

    Raises [Sys_error] if a file cannot be read.

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [specials],
    [bos_token], [eos_token], [pad_token], [unk_token]) are as in {!bpe}. *)

val add_tokens : t -> string list -> t
(** [add_tokens t tokens] is [t] with [tokens] added to the vocabulary. Only
    supported for word-level tokenizers.

    Raises [Invalid_argument] if the tokenizer does not support dynamic
    vocabulary extension. *)

(** {1:accessors Accessors} *)

val normalizer : t -> Normalizer.t option
(** [normalizer t] is [t]'s normalizer, if any. *)

val pre_tokenizer : t -> Pre_tokenizer.t option
(** [pre_tokenizer t] is [t]'s pre-tokenizer, if any. *)

val post_processor : t -> Post_processor.t option
(** [post_processor t] is [t]'s post-processor, if any. *)

val decoder : t -> Decoder.t option
(** [decoder t] is [t]'s decoder, if any. *)

val specials : t -> special list
(** [specials t] is [t]'s special tokens. *)

val bos_token : t -> string option
(** [bos_token t] is [t]'s beginning-of-sequence token, if any. *)

val eos_token : t -> string option
(** [eos_token t] is [t]'s end-of-sequence token, if any. *)

val pad_token : t -> string option
(** [pad_token t] is [t]'s padding token, if any. *)

val unk_token : t -> string option
(** [unk_token t] is [t]'s unknown token, if any. *)

(** {1:vocab Vocabulary} *)

val vocab : t -> (string * int) list
(** [vocab t] is [t]'s vocabulary as [(token, id)] pairs. *)

val vocab_size : t -> int
(** [vocab_size t] is the number of tokens in [t]'s vocabulary. *)

val token_to_id : t -> string -> int option
(** [token_to_id t token] is the ID of [token] in [t], if any. *)

val id_to_token : t -> int -> string option
(** [id_to_token t id] is the token string for [id] in [t], if any. *)

(** {1:encoding Encoding and decoding} *)

val encode :
  t ->
  ?pair:string ->
  ?add_special_tokens:bool ->
  ?padding:padding ->
  ?truncation:truncation ->
  string ->
  Encoding.t
(** [encode t text] is the encoding of [text] by [t].

    - [pair]: a second sentence for sentence-pair tasks. The post-processor
      merges both sequences with appropriate type IDs. Default: none.
    - [add_special_tokens]: whether to insert special tokens via the
      post-processor. Default: [true].
    - [padding]: padding configuration. Default: none (no padding).
    - [truncation]: truncation configuration. Default: none (no truncation). *)

val encode_batch :
  t ->
  ?add_special_tokens:bool ->
  ?padding:padding ->
  ?truncation:truncation ->
  string list ->
  Encoding.t list
(** [encode_batch t texts] is the encoding of each text in [texts].

    Optional parameters are as in {!encode}. For sentence-pair tasks, use
    {!encode_pairs_batch}. *)

val encode_pairs_batch :
  t ->
  ?add_special_tokens:bool ->
  ?padding:padding ->
  ?truncation:truncation ->
  (string * string) list ->
  Encoding.t list
(** [encode_pairs_batch t pairs] encodes a batch of sentence pairs. Each element
    is [(primary, secondary)].

    Optional parameters are as in {!encode}. *)

val encode_ids :
  t ->
  ?pair:string ->
  ?add_special_tokens:bool ->
  ?padding:padding ->
  ?truncation:truncation ->
  string ->
  int array
(** [encode_ids t text] is [Encoding.ids (encode t text)].

    Optional parameters are as in {!encode}. *)

val decode : t -> ?skip_special_tokens:bool -> int array -> string
(** [decode t ids] is the text obtained by decoding [ids] through [t]'s
    vocabulary and decoder.

    [skip_special_tokens] defaults to [false]. *)

val decode_batch :
  t -> ?skip_special_tokens:bool -> int array list -> string list
(** [decode_batch t ids_list] decodes each element of [ids_list].

    [skip_special_tokens] defaults to [false]. *)

(** {1:training Training} *)

val train_bpe :
  ?init:t ->
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?specials:special list ->
  ?bos_token:string ->
  ?eos_token:string ->
  ?pad_token:string ->
  ?unk_token:string ->
  ?vocab_size:int ->
  ?min_frequency:int ->
  ?limit_alphabet:int ->
  ?initial_alphabet:string list ->
  ?continuing_subword_prefix:string ->
  ?end_of_word_suffix:string ->
  ?show_progress:bool ->
  ?max_token_length:int ->
  data ->
  t
(** [train_bpe data] trains a BPE tokenizer from [data].

    Learns merge rules by iteratively merging the most frequent adjacent pairs
    until reaching the target vocabulary size.

    - [init]: existing tokenizer to extend. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. Default:
      [30000].
    - [min_frequency]: minimum pair frequency to be merged. Default: [0].
    - [limit_alphabet]: maximum number of initial characters to keep. Default:
      none (keep all).
    - [initial_alphabet]: characters to include regardless of frequency.
      Default: [[]].
    - [continuing_subword_prefix]: prefix for non-initial subwords. Default:
      none.
    - [end_of_word_suffix]: suffix marking word boundaries. Default: none.
    - [show_progress]: display progress bar. Default: [true].
    - [max_token_length]: maximum token length. Default: none.

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [specials],
    [bos_token], [eos_token], [pad_token], [unk_token]) are as in {!bpe}. *)

val train_wordpiece :
  ?init:t ->
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?specials:special list ->
  ?bos_token:string ->
  ?eos_token:string ->
  ?pad_token:string ->
  ?unk_token:string ->
  ?vocab_size:int ->
  ?min_frequency:int ->
  ?limit_alphabet:int ->
  ?initial_alphabet:string list ->
  ?continuing_subword_prefix:string ->
  ?end_of_word_suffix:string ->
  ?show_progress:bool ->
  data ->
  t
(** [train_wordpiece data] trains a WordPiece tokenizer from [data].

    Learns subword vocabulary by maximizing language model likelihood.

    - [init]: existing tokenizer to extend. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. Default:
      [30000].
    - [min_frequency]: minimum frequency for a subword to be included. Default:
      [0].
    - [limit_alphabet]: maximum number of initial characters to keep. Default:
      none (keep all).
    - [initial_alphabet]: characters to include regardless of frequency.
      Default: [[]].
    - [continuing_subword_prefix]: prefix for non-initial subwords. Default:
      ["##"].
    - [end_of_word_suffix]: suffix marking word boundaries. Default: none.
    - [show_progress]: display progress bar. Default: [true].

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [specials],
    [bos_token], [eos_token], [pad_token], [unk_token]) are as in {!bpe}. *)

val train_wordlevel :
  ?init:t ->
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?specials:special list ->
  ?bos_token:string ->
  ?eos_token:string ->
  ?pad_token:string ->
  ?unk_token:string ->
  ?vocab_size:int ->
  ?min_frequency:int ->
  ?show_progress:bool ->
  data ->
  t
(** [train_wordlevel data] trains a word-level tokenizer from [data].

    Builds vocabulary by collecting unique words, optionally filtering by
    frequency. No subword splitting.

    - [init]: existing tokenizer to extend. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. Default:
      [30000].
    - [min_frequency]: minimum frequency for a word to be included. Default:
      [0].
    - [show_progress]: display progress bar. Default: [true].

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [specials],
    [bos_token], [eos_token], [pad_token], [unk_token]) are as in {!bpe}. *)

val train_unigram :
  ?init:t ->
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?specials:special list ->
  ?bos_token:string ->
  ?eos_token:string ->
  ?pad_token:string ->
  ?unk_token:string ->
  ?vocab_size:int ->
  ?show_progress:bool ->
  ?shrinking_factor:float ->
  ?max_piece_length:int ->
  ?n_sub_iterations:int ->
  data ->
  t
(** [train_unigram data] trains a Unigram tokenizer from [data].

    Learns probabilistic subword vocabulary using EM algorithm.

    - [init]: existing tokenizer to extend. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. Default:
      [8000].
    - [show_progress]: display progress bar. Default: [true].
    - [shrinking_factor]: fraction of vocabulary to retain in each pruning
      iteration. Default: [0.75].
    - [max_piece_length]: maximum subword length. Default: [16].
    - [n_sub_iterations]: number of EM sub-iterations per pruning round.
      Default: [2].

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [specials],
    [bos_token], [eos_token], [pad_token], [unk_token]) are as in {!bpe}. *)

(** {1:model_files Model files} *)

val export_tiktoken : t -> merges_path:string -> vocab_path:string -> unit
(** [export_tiktoken t ~merges_path ~vocab_path] exports [t]'s BPE merges and
    vocabulary in tiktoken-compatible format.

    {b Warning.} Only BPE tokenizers are supported. Raises [Failure] for other
    model types. *)

val save_model_files :
  t -> folder:string -> ?prefix:string -> unit -> string list
(** [save_model_files t ~folder ?prefix ()] saves [t]'s underlying model files
    (vocabulary and merges) to [folder] and returns the list of created file
    paths.

    [prefix] defaults to [""]. *)

(** {1:huggingface HuggingFace compatibility} *)

val from_file : string -> (t, string) result
(** [from_file path] is a tokenizer loaded from a HuggingFace [tokenizer.json]
    file. Errors if the file cannot be read or has invalid format. *)

val from_json : Jsont.json -> (t, string) result
(** [from_json json] is a tokenizer deserialized from HuggingFace JSON format.
    Errors if [json] has a missing or unknown model type, or invalid parameters.
*)

val to_json : t -> Jsont.json
(** [to_json t] is [t] serialized to HuggingFace JSON format. *)

val save_pretrained : t -> path:string -> unit
(** [save_pretrained t ~path] saves [t] to [path] in HuggingFace format. Creates
    [path/tokenizer.json].

    Raises [Sys_error] if [path] cannot be written. *)

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a tokenizer for inspection. Shows algorithm type, vocabulary
    size, and configured pipeline stages. *)
