(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tokenization library for Saga.

    This module provides the main tokenization API matching HuggingFace
    Tokenizers design. It supports multiple tokenization algorithms (BPE,
    WordPiece, Unigram, Word-level, Character-level), text normalization,
    pre-tokenization, post-processing, and decoding.

    {1 Quick Start}

    Load a pretrained tokenizer:
    {[
      let tokenizer = Tokenizer.from_file "tokenizer.json" |> Result.get_ok in
      let encoding = Tokenizer.encode tokenizer "Hello world!" in
      let ids = Encoding.get_ids encoding
    ]}

    Create a BPE tokenizer from scratch:
    {[
      let tokenizer =
        Tokenizer.bpe
          ~vocab:[("hello", 0); ("world", 1); ("[PAD]", 2)]
          ~merges:[]
          ()
      in
      let encoding = Tokenizer.encode tokenizer "hello world" in
      let text = Tokenizer.decode tokenizer [0; 1]
    ]}

    Train a new tokenizer:
    {[
      let texts = [ "Hello world"; "How are you?"; "Hello again" ] in
      let tokenizer =
        Tokenizer.train_bpe (`Seq (List.to_seq texts)) ~vocab_size:1000 ()
      in
      Tokenizer.save_pretrained tokenizer ~path:"./my_tokenizer"
    ]}

    {1 Architecture}

    Tokenization proceeds through stages:

    - {b Normalization}: Clean and normalize text (lowercase, accent removal,
      etc.)
    - {b Pre-tokenization}: Split text into words or subwords
    - {b Tokenization}: Apply vocabulary-based encoding (BPE, WordPiece, etc.)
    - {b Post-processing}: Add special tokens, set type IDs
    - {b Padding/Truncation}: Adjust length for batching

    Each stage is optional and configurable via builder methods.

    Post-processing patterns are model-specific:
    - BERT: Adds [CLS] at start, [SEP] at end, type IDs distinguish sequences
    - GPT-2: No special tokens by default, uses BOS/EOS if configured
    - RoBERTa: Uses <s> and </s> tokens similar to BERT but different format *)

module Grapheme : sig
  type ret = [ `Await | `Boundary | `End | `Uchar of Uchar.t ]
  type t

  val create : unit -> t
  val add : t -> [ `Await | `End | `Uchar of Uchar.t ] -> ret
end

module Unicode = Unicode
(** Unicode utilities for normalization. *)

module Normalizers = Normalizers
(** Text normalization (lowercase, NFD/NFC, accent stripping, etc.). *)

module Pre_tokenizers = Pre_tokenizers
(** Pre-tokenization (whitespace splitting, punctuation handling, etc.). *)

module Processors = Processors
(** Post-processing (adding [CLS]/[SEP], setting type IDs, etc.). *)

module Decoders = Decoders
(** Decoding token IDs back to text. *)

module Encoding = Encoding
(** Encoding representation (output of tokenization). *)

type direction = [ `Left | `Right ]
(** Direction for padding or truncation: [`Left] (beginning) or [`Right] (end).
*)

type special = {
  token : string;  (** The token text (e.g., "<pad>", "<unk>"). *)
  single_word : bool;
      (** Whether this token must match whole words only. Default: [false]. *)
  lstrip : bool;
      (** Whether to strip whitespace on the left. Default: [false]. *)
  rstrip : bool;
      (** Whether to strip whitespace on the right. Default: [false]. *)
  normalized : bool;
      (** Whether to apply normalization to this token. Default: [true] for
          regular tokens, [false] for special tokens. *)
}
(** Special token configuration.

    Special tokens are not split during tokenization and can be skipped during
    decoding. Token IDs are assigned automatically when added to the vocabulary.

    All special token types are uniform - the semantic meaning (pad, unk, bos,
    etc.) is contextual, not encoded in the type. *)

type pad_length = [ `Batch_longest | `Fixed of int | `To_multiple of int ]
(** Padding length strategy.

    - [`Batch_longest]: Pad to longest sequence in batch
    - [`Fixed n]: Pad all sequences to fixed length n
    - [`To_multiple n]: Pad to smallest multiple of n >= sequence length *)

type padding = {
  length : pad_length;
  direction : direction;
  pad_id : int option;
  pad_type_id : int option;
  pad_token : string option;
}
(** Padding configuration.

    When optional fields are [None], falls back to tokenizer's configured
    padding token. If the tokenizer has no padding token configured and these
    fields are [None], padding operations will raise [Invalid_argument]. *)

type truncation = { max_length : int; direction : direction }
(** Truncation configuration.

    Limits sequences to [max_length] tokens, removing from specified direction.
*)

type data =
  [ `Files of string list
  | `Seq of string Seq.t
  | `Iterator of unit -> string option ]
(** Training data source.

    - [`Files paths]: Read training text from files
    - [`Seq seq]: Use sequence of strings
    - [`Iterator f]: Pull training data via iterator ([None] signals end) *)

(** {1 Special Token Constructors} *)

module Special : sig
  val make :
    ?single_word:bool ->
    ?lstrip:bool ->
    ?rstrip:bool ->
    ?normalized:bool ->
    string ->
    special
  (** [make ?single_word ?lstrip ?rstrip ?normalized token] creates a special
      token configuration.

      All parameters default to appropriate values for special tokens:

      - [single_word]: [false] - can match partial words
      - [lstrip]: [false] - don't strip left whitespace
      - [rstrip]: [false] - don't strip right whitespace
      - [normalized]: [false] - special tokens not normalized *)

  val pad : string -> special
  (** [pad token] creates a padding token (e.g., ["<pad>"]). *)

  val unk : string -> special
  (** [unk token] creates an unknown token (e.g., ["<unk>"]). *)

  val bos : string -> special
  (** [bos token] creates a beginning-of-sequence token (e.g., ["<s>"]). *)

  val eos : string -> special
  (** [eos token] creates an end-of-sequence token (e.g., ["</s>"]). *)

  val cls : string -> special
  (** [cls token] creates a classification token (e.g., ["[CLS]"]). *)

  val sep : string -> special
  (** [sep token] creates a separator token (e.g., ["[SEP]"]). *)

  val mask : string -> special
  (** [mask token] creates a mask token (e.g., ["[MASK]"]). *)
end

module Tokenizer : sig
  type t

  val normalizer : t -> Normalizers.t option
  (** [normalizer tokenizer] retrieves the configured normalizer.

      Returns [None] if no normalizer is set. The normalizer is applied before
      all other processing stages to clean and normalize text. *)

  val with_normalizer : t -> Normalizers.t option -> t
  (** [with_normalizer tokenizer norm] replaces the tokenizer's normalizer.

      Pass [None] to remove the normalization step entirely. Pass [Some norm] to
      install a new normalizer. Returns updated tokenizer.

      {[
        let tokenizer = Tokenizer.bpe () in
        let tokenizer = Tokenizer.with_normalizer tokenizer
          (Some (Normalizers.sequence [
            Normalizers.nfd ();
            Normalizers.lowercase ();
            Normalizers.strip_accents ();
          ]))
      ]} *)

  val pre_tokenizer : t -> Pre_tokenizers.t option
  (** [pre_tokenizer tokenizer] retrieves the configured pre-tokenizer.

      Returns [None] if no pre-tokenizer is set. The pre-tokenizer splits text
      into pieces before vocabulary-based encoding. *)

  val with_pre_tokenizer : t -> Pre_tokenizers.t option -> t
  (** [with_pre_tokenizer tokenizer pre] replaces the tokenizer's pre-tokenizer.

      Pass [None] to remove pre-tokenization (text processed as-is). Pass
      [Some pre] to install a new pre-tokenizer. Returns updated tokenizer.

      {[
        let tokenizer = Tokenizer.bpe () in
        let tokenizer = Tokenizer.with_pre_tokenizer tokenizer
          (Some (Pre_tokenizers.byte_level ~add_prefix_space:true ()))
      ]} *)

  val post_processor : t -> Processors.t option
  (** [post_processor tokenizer] retrieves the configured post-processor.

      Returns [None] if no post-processor is set. The post-processor adds
      special tokens and sets type IDs after encoding. *)

  val with_post_processor : t -> Processors.t option -> t
  (** [with_post_processor tokenizer post] replaces the tokenizer's
      post-processor.

      Pass [None] to remove post-processing. Pass [Some post] to install a new
      post-processor. Returns updated tokenizer.

      {[
        let tokenizer = Tokenizer.bpe () in
        let tokenizer = Tokenizer.with_post_processor tokenizer
          (Some (Processors.bert_processing
            ~sep:("[SEP]", 102) ~cls:("[CLS]", 101) ()))
      ]} *)

  val decoder : t -> Decoders.t option
  (** [decoder tokenizer] retrieves the configured decoder.

      Returns [None] if no decoder is set. The decoder converts token IDs back
      to text. *)

  val with_decoder : t -> Decoders.t option -> t
  (** [with_decoder tokenizer dec] replaces the tokenizer's decoder.

      Pass [None] to use default decoding (concatenate tokens). Pass [Some dec]
      to install a new decoder. Returns updated tokenizer.

      {[
        let tokenizer = Tokenizer.bpe () in
        let tokenizer = Tokenizer.with_decoder tokenizer
          (Some (Decoders.byte_level ()))
      ]} *)

  val specials : t -> special list
  (** [specials tokenizer] retrieves the configured special tokens. *)

  val with_specials : t -> special list -> t
  (** [with_specials tokenizer specials] replaces the special tokens with the
      provided list. *)

  val add_specials : t -> special list -> t
  (** [add_specials tokenizer specials] extends the set of special tokens. *)

  (** {2 Special Token Roles}

      These functions configure which token strings serve specific roles in the
      tokenizer (BOS, EOS, PAD, UNK). This follows HuggingFace's design where
      roles are separate from token properties. *)

  val bos_token : t -> string option
  (** [bos_token tokenizer] returns the beginning-of-sequence token string, if
      configured. *)

  val set_bos_token : t -> string option -> t
  (** [set_bos_token tokenizer token] sets which token serves as
      beginning-of-sequence marker. Pass [None] to unset. The token should
      already be in the vocabulary. *)

  val eos_token : t -> string option
  (** [eos_token tokenizer] returns the end-of-sequence token string, if
      configured. *)

  val set_eos_token : t -> string option -> t
  (** [set_eos_token tokenizer token] sets which token serves as end-of-sequence
      marker. Pass [None] to unset. *)

  val pad_token : t -> string option
  (** [pad_token tokenizer] returns the padding token string, if configured. *)

  val set_pad_token : t -> string option -> t
  (** [set_pad_token tokenizer token] sets which token serves as padding marker.
      Pass [None] to unset. *)

  val unk_token : t -> string option
  (** [unk_token tokenizer] returns the unknown token string, if configured. *)

  val set_unk_token : t -> string option -> t
  (** [set_unk_token tokenizer token] sets which token serves as unknown token
      marker. Pass [None] to unset. *)

  val vocab : t -> (string * int) list
  (** [vocab tokenizer] returns the vocabulary as (token, id) pairs. *)

  val vocab_size : t -> int
  (** [vocab_size tokenizer] returns the size of the vocabulary. *)

  val token_to_id : t -> string -> int option
  (** [token_to_id tokenizer token] maps a token string to its id. *)

  val id_to_token : t -> int -> string option
  (** [id_to_token tokenizer id] maps an id back to its token string. *)

  val bpe :
    ?normalizer:Normalizers.t ->
    ?pre:Pre_tokenizers.t ->
    ?post:Processors.t ->
    ?decoder:Decoders.t ->
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
  (** [bpe ?normalizer ?pre ?post ?decoder ?specials ?vocab ?merges
       ?cache_capacity ?dropout ?unk_token ?continuing_subword_prefix
       ?end_of_word_suffix ?fuse_unk ?byte_fallback ?ignore_merges ()] creates a
      BPE (Byte Pair Encoding) tokenizer. Used by GPT-2, GPT-3, RoBERTa.

      @param normalizer Text normalization (e.g., lowercase, strip accents)
      @param pre Pre-tokenization strategy (e.g., whitespace splitting)
      @param post Post-processor for special tokens ([CLS], [SEP])
      @param decoder Decoding strategy to reverse tokenization
      @param specials Special tokens to add to vocabulary
      @param bos_token
        Token to use as beginning-of-sequence marker. Configures both the role
        and adds to vocabulary if not present.
      @param eos_token
        Token to use as end-of-sequence marker. Configures both the role and
        adds to vocabulary if not present.
      @param pad_token
        Token to use as padding marker. Configures both the role and adds to
        vocabulary if not present.
      @param unk_token
        Token for unknown characters. Configures both the role and the BPE
        model's unknown handling. Default: None (no unknown handling).
      @param vocab Initial vocabulary mapping tokens to IDs
      @param merges
        Merge rules as [(token1, token2)] pairs learned during training
      @param cache_capacity
        LRU cache size for tokenization results. Default: 10000. Higher = faster
        for repeated inputs but more memory.
      @param dropout
        Probability of skipping merges during tokenization (0.0-1.0). Default:
        None (no dropout). Used for data augmentation. At 1.0, no merges applied
        (character-level).
      @param continuing_subword_prefix
        Prefix for non-initial subwords (e.g., "##" for BERT). Default: None.
      @param end_of_word_suffix
        Suffix marking word boundaries (e.g., "</w>"). Default: None.
      @param fuse_unk
        Whether to merge consecutive unknown tokens. Default: false.
      @param byte_fallback
        Use byte-level fallback for unknown chars (e.g., "<0x00>") instead of
        UNK. Default: false.
      @param ignore_merges
        Skip merge application (character-level output). Default: false.

      See {!Bpe} module for algorithm details. *)

  val wordpiece :
    ?normalizer:Normalizers.t ->
    ?pre:Pre_tokenizers.t ->
    ?post:Processors.t ->
    ?decoder:Decoders.t ->
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
  (** [wordpiece ?normalizer ?pre ?post ?decoder ?specials ?vocab ?unk_token
       ?continuing_subword_prefix ?max_input_chars_per_word ()] creates a
      WordPiece tokenizer. Used by BERT, DistilBERT, Electra.

      WordPiece uses greedy longest-match-first algorithm to split words into
      subword pieces. Subwords are prefixed to indicate they continue a word
      (e.g., "##ing" for "running" → ["run", "##ning"]).

      @param normalizer Text normalization (e.g., lowercase, strip accents)
      @param pre Pre-tokenization strategy (e.g., whitespace splitting)
      @param post Post-processor for special tokens ([CLS], [SEP])
      @param decoder Decoding strategy to reverse tokenization
      @param specials Special tokens to add to vocabulary
      @param vocab Initial vocabulary mapping tokens to IDs
      @param unk_token Token for out-of-vocabulary words. Default: "[UNK]".
      @param continuing_subword_prefix
        Prefix for non-initial subwords (e.g., "##"). Default: "##".
      @param max_input_chars_per_word
        Maximum characters per word. Words longer than this are replaced with
        unk_token. Default: 100.

      See {!Wordpiece} module for algorithm details. *)

  val word_level :
    ?normalizer:Normalizers.t ->
    ?pre:Pre_tokenizers.t ->
    ?post:Processors.t ->
    ?decoder:Decoders.t ->
    ?specials:special list ->
    ?bos_token:string ->
    ?eos_token:string ->
    ?pad_token:string ->
    ?unk_token:string ->
    ?vocab:(string * int) list ->
    unit ->
    t
  (** [word_level ?normalizer ?pre ?post ?decoder ?specials ?vocab ?unk_token
       ()] creates a word-level tokenizer.

      Maps each word directly to a token ID from vocabulary. No subword
      splitting. Words not in vocabulary are mapped to unk_token. Simplest
      tokenization strategy, suitable for smaller vocabularies or
      domain-specific text.

      @param normalizer Text normalization (e.g., lowercase, strip accents)
      @param pre Pre-tokenization strategy (e.g., whitespace splitting)
      @param post Post-processor for special tokens
      @param decoder Decoding strategy to reverse tokenization
      @param specials Special tokens to add to vocabulary
      @param vocab Initial vocabulary mapping words to IDs
      @param unk_token Token for out-of-vocabulary words. Default: "[UNK]".

      See {!Word_level} module for algorithm details. *)

  val unigram :
    ?normalizer:Normalizers.t ->
    ?pre:Pre_tokenizers.t ->
    ?post:Processors.t ->
    ?decoder:Decoders.t ->
    ?specials:special list ->
    ?bos_token:string ->
    ?eos_token:string ->
    ?pad_token:string ->
    ?unk_token:string ->
    ?vocab:(string * float) list ->
    ?byte_fallback:bool ->
    ?max_piece_length:int ->
    ?n_sub_iterations:int ->
    ?shrinking_factor:float ->
    unit ->
    t
  (** [unigram ?normalizer ?pre ?post ?decoder ?specials ?vocab ?unk_token
       ?byte_fallback ?max_piece_length ?n_sub_iterations ?shrinking_factor ()]
      creates a Unigram tokenizer. Used by AlBERT, T5, mBART.

      Unigram uses probabilistic segmentation with Viterbi algorithm to find
      optimal subword splits based on token probabilities. Vocabulary entries
      have associated scores (negative log probabilities).

      @param normalizer Text normalization (e.g., lowercase, strip accents)
      @param pre Pre-tokenization strategy (e.g., whitespace splitting)
      @param post Post-processor for special tokens
      @param decoder Decoding strategy to reverse tokenization
      @param specials Special tokens to add to vocabulary
      @param vocab
        Initial vocabulary mapping tokens to scores (negative log
        probabilities). Higher scores = less likely.
      @param unk_token Token for unknown characters. Default: None.
      @param byte_fallback
        Use byte-level fallback for unknown chars instead of UNK. Default:
        false.
      @param max_piece_length
        Maximum characters per piece during training. Default: 16.
      @param n_sub_iterations
        Number of EM sub-iterations during training. Default: 2.
      @param shrinking_factor
        Fraction of vocabulary to keep in each pruning step during training
        (0.0-1.0). Default: 0.75.

      See {!Unigram} module for algorithm details. *)

  val chars :
    ?normalizer:Normalizers.t ->
    ?pre:Pre_tokenizers.t ->
    ?post:Processors.t ->
    ?decoder:Decoders.t ->
    ?specials:special list ->
    ?bos_token:string ->
    ?eos_token:string ->
    ?pad_token:string ->
    ?unk_token:string ->
    unit ->
    t
  (** [chars ?normalizer ?pre ?post ?decoder ?specials ()] creates a
      character-level tokenizer.

      Splits text into individual characters. Each character in the input
      becomes a separate token. Vocabulary is built from unique characters seen.
      Useful for character-level models or languages with large character sets.

      @param normalizer Text normalization (e.g., lowercase)
      @param pre Pre-tokenization strategy (usually None for char-level)
      @param post Post-processor for special tokens
      @param decoder Decoding strategy to reverse tokenization
      @param specials Special tokens to add to vocabulary

      See {!Chars} module for algorithm details. *)

  val regex :
    string ->
    ?normalizer:Normalizers.t ->
    ?pre:Pre_tokenizers.t ->
    ?post:Processors.t ->
    ?decoder:Decoders.t ->
    ?specials:special list ->
    ?bos_token:string ->
    ?eos_token:string ->
    ?pad_token:string ->
    ?unk_token:string ->
    unit ->
    t
  (** [regex pattern ?normalizer ?pre ?post ?decoder ?specials ()] creates a
      regex-based tokenizer.

      Splits text using a regular expression pattern. Each match of the pattern
      becomes a token. Useful for custom tokenization rules or domain-specific
      formats.

      @param pattern
        Regular expression pattern (Str module syntax) used to match tokens.
        Each match becomes a separate token.

      Pattern examples:
      - ["[a-zA-Z]+"] matches sequences of letters
      - ["[0-9]+"] matches sequences of digits
      - ["[a-zA-Z]+|[0-9]+|[^a-zA-Z0-9 ]"] matches words, numbers, or
        punctuation

      @param normalizer Text normalization (e.g., lowercase, strip accents)
      @param pre Pre-tokenization strategy (applied before pattern matching)
      @param post Post-processor for special tokens
      @param decoder Decoding strategy to reverse tokenization
      @param specials Special tokens to add to vocabulary *)

  val from_model_file :
    vocab:string ->
    ?merges:string ->
    ?normalizer:Normalizers.t ->
    ?pre:Pre_tokenizers.t ->
    ?post:Processors.t ->
    ?decoder:Decoders.t ->
    ?specials:special list ->
    ?bos_token:string ->
    ?eos_token:string ->
    ?pad_token:string ->
    ?unk_token:string ->
    unit ->
    t
  (** [from_model_file ~vocab ?merges ?normalizer ?pre ?post ?decoder ?specials
       ()] loads tokenizer from HuggingFace format model files.

      Loads vocabulary and merge rules from separate files. Model type is
      inferred from files: if merges file provided, creates BPE tokenizer,
      otherwise creates WordPiece tokenizer.

      @param vocab
        Path to vocabulary file (vocab.json). Expected format: JSON object
        mapping tokens to IDs: [{"hello": 0, "world": 1, "[PAD]": 2}].
      @param merges
        Path to merges file (merges.txt). Expected format: one merge per line as
        space-separated token pairs: ["he llo", "wor ld"]. First line may be
        header (ignored if starts with "#version"). Optional for WordPiece,
        required for BPE.
      @param normalizer Text normalization to apply
      @param pre Pre-tokenization strategy
      @param post Post-processor for special tokens
      @param decoder Decoding strategy
      @param specials Special tokens to add to vocabulary

      {[
        let tokenizer =
          Tokenizer.from_model_file ~vocab:"vocab.json" ~merges:"merges.txt"
            ~normalizer:(Normalizers.lowercase ())
            ~pre:(Pre_tokenizers.byte_level ())
            ()
      ]} *)

  val add_tokens : t -> string list -> t
  (** [add_tokens tokenizer tokens] adds regular tokens to the underlying
      vocabulary.

      The underlying model is mutated in-place for performance, but the function
      returns an updated tokenizer value. Not thread-safe: concurrent calls to
      [add_tokens] or other mutating operations on the same tokenizer require
      external synchronization. *)

  val encode :
    t ->
    ?pair:string ->
    ?add_special_tokens:bool ->
    ?padding:padding ->
    ?truncation:truncation ->
    string ->
    Encoding.t
  (** [encode tokenizer ?pair ?add_special_tokens ?padding ?truncation text]
      encodes a single sequence. *)

  val encode_batch :
    t ->
    ?pairs:string option list ->
    ?add_special_tokens:bool ->
    ?padding:padding ->
    ?truncation:truncation ->
    string list ->
    Encoding.t list
  (** [encode_batch tokenizer ?pairs ?add_special_tokens ?padding ?truncation
       texts] encodes a batch of sequences. *)

  val encode_ids :
    t ->
    ?pair:string ->
    ?add_special_tokens:bool ->
    ?padding:padding ->
    ?truncation:truncation ->
    string ->
    int array
  (** [encode_ids tokenizer ?pair ?add_special_tokens ?padding ?truncation text]
      is a convenience helper returning just the token ids. *)

  val decode : t -> ?skip_special_tokens:bool -> int array -> string
  (** [decode tokenizer ?skip_special_tokens ids] decodes ids back into text. *)

  val decode_batch :
    t -> ?skip_special_tokens:bool -> int array list -> string list
  (** [decode_batch tokenizer ?skip_special_tokens ids_list] decodes a batch of
      id sequences. *)

  val train_bpe :
    ?init:t ->
    ?normalizer:Normalizers.t ->
    ?pre:Pre_tokenizers.t ->
    ?post:Processors.t ->
    ?decoder:Decoders.t ->
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
  (** [train_bpe ?init ?normalizer ?pre ?post ?decoder ?specials ?vocab_size
       ?min_frequency ?limit_alphabet ?initial_alphabet
       ?continuing_subword_prefix ?end_of_word_suffix ?show_progress
       ?max_token_length data] trains a BPE tokenizer from training data.

      Learns merge rules by iteratively merging the most frequent adjacent
      character or subword pairs until reaching target vocabulary size.

      @param init
        Existing tokenizer to extend. If provided, training adds to existing
        vocabulary. Default: create new tokenizer.
      @param normalizer Text normalization applied before training
      @param pre Pre-tokenization strategy applied before training
      @param post Post-processor for special tokens
      @param decoder Decoding strategy
      @param specials Special tokens to add to vocabulary
      @param vocab_size
        Target vocabulary size including special tokens. Training continues
        until this size is reached. Default: 30000.
      @param min_frequency
        Minimum occurrences for a token pair to be merged. Higher values create
        smaller vocabularies with more common subwords. Typical: 2-10. Default:
        2.
      @param limit_alphabet
        Maximum initial characters in alphabet. Limits character set to most
        frequent characters. Default: None (unlimited).
      @param initial_alphabet
        Explicit initial character set. If provided, overrides automatic
        alphabet discovery. Default: None.
      @param continuing_subword_prefix
        Prefix for non-initial subwords (e.g., "##"). Default: None.
      @param end_of_word_suffix
        Suffix marking word boundaries (e.g., "</w>"). Default: None.
      @param show_progress
        Display progress bar during training. Requires training data to support
        progress tracking (e.g., [`Files] or [`Seq]). Default: true.
      @param max_token_length
        Maximum characters per token during merge operations. Pairs creating
        tokens longer than this are skipped. Default: None (unlimited).
      @param data Training data source

      {[
        let texts = ["Hello world"; "How are you?"; "Hello again"] in
        let tokenizer = Tokenizer.train_bpe (`Seq (List.to_seq texts))
          ~vocab_size:1000
          ~min_frequency:2
          ~show_progress:false
      ]} *)

  val train_wordpiece :
    ?init:t ->
    ?normalizer:Normalizers.t ->
    ?pre:Pre_tokenizers.t ->
    ?post:Processors.t ->
    ?decoder:Decoders.t ->
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
  (** [train_wordpiece ?init ?normalizer ?pre ?post ?decoder ?specials
       ?vocab_size ?min_frequency ?limit_alphabet ?initial_alphabet
       ?continuing_subword_prefix ?end_of_word_suffix ?unk_token ?show_progress
       data] trains a WordPiece tokenizer from training data.

      Learns subword vocabulary by maximizing language model likelihood,
      selecting subwords that maximize corpus representation efficiency.

      @param init Existing tokenizer to extend. Default: create new tokenizer.
      @param normalizer Text normalization applied before training
      @param pre Pre-tokenization strategy applied before training
      @param post Post-processor for special tokens
      @param decoder Decoding strategy
      @param specials Special tokens to add to vocabulary
      @param vocab_size
        Target vocabulary size including special tokens. Default: 30000.
      @param min_frequency
        Minimum occurrences for a subword to be included. Higher values create
        smaller vocabularies. Typical: 2-10. Default: 2.
      @param limit_alphabet
        Maximum initial characters in alphabet. Default: None (unlimited).
      @param initial_alphabet Explicit initial character set. Default: None.
      @param continuing_subword_prefix
        Prefix for non-initial subwords (e.g., "##"). Default: "##".
      @param end_of_word_suffix Suffix marking word boundaries. Default: None.
      @param unk_token Token for out-of-vocabulary words. Default: "[UNK]".
      @param show_progress Display progress bar during training. Default: true.
      @param data Training data source *)

  val train_wordlevel :
    ?init:t ->
    ?normalizer:Normalizers.t ->
    ?pre:Pre_tokenizers.t ->
    ?post:Processors.t ->
    ?decoder:Decoders.t ->
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
  (** [train_wordlevel ?init ?normalizer ?pre ?post ?decoder ?specials
       ?vocab_size ?min_frequency ?show_progress data] trains a word-level
      tokenizer from training data.

      Builds vocabulary by collecting unique words from training data,
      optionally filtering by frequency. No subword splitting.

      @param init Existing tokenizer to extend. Default: create new tokenizer.
      @param normalizer Text normalization applied before training
      @param pre Pre-tokenization strategy applied before training
      @param post Post-processor for special tokens
      @param decoder Decoding strategy
      @param specials Special tokens to add to vocabulary
      @param vocab_size
        Target vocabulary size including special tokens. Training includes most
        frequent words up to this limit. Default: 30000.
      @param min_frequency
        Minimum occurrences for a word to be included. Higher values create
        smaller vocabularies of common words. Typical: 2-10. Default: 0 (include
        all words).
      @param show_progress Display progress bar during training. Default: true.
      @param data Training data source *)

  val train_unigram :
    ?init:t ->
    ?normalizer:Normalizers.t ->
    ?pre:Pre_tokenizers.t ->
    ?post:Processors.t ->
    ?decoder:Decoders.t ->
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
  (** [train_unigram ?init ?normalizer ?pre ?post ?decoder ?specials ?vocab_size
       ?show_progress ?shrinking_factor ?unk_token ?max_piece_length
       ?n_sub_iterations data] trains a Unigram tokenizer from training data.

      Learns probabilistic subword vocabulary using EM algorithm. Starts with
      large candidate vocabulary and iteratively prunes low-likelihood pieces
      until reaching target size.

      @param init Existing tokenizer to extend. Default: create new tokenizer.
      @param normalizer Text normalization applied before training
      @param pre Pre-tokenization strategy applied before training
      @param post Post-processor for special tokens
      @param decoder Decoding strategy
      @param specials Special tokens to add to vocabulary
      @param vocab_size
        Target vocabulary size including special tokens. Default: 8000.
      @param show_progress Display progress bar during training. Default: true.
      @param shrinking_factor
        Fraction of vocabulary to retain in each pruning iteration (0.0-1.0).
        Larger values slow convergence but may improve quality. Typical:
        0.75-0.95. Default: 0.75.
      @param unk_token Token for unknown characters. Default: None.
      @param max_piece_length
        Maximum characters per subword piece. Longer pieces are not considered.
        Typical: 8-32. Default: 16.
      @param n_sub_iterations
        Number of EM sub-iterations per pruning step. Higher values improve
        accuracy but slow training. Typical: 2-5. Default: 2.
      @param data Training data source *)

  val export_tiktoken : t -> merges_path:string -> vocab_path:string -> unit
  (** [export_tiktoken tokenizer ~merges_path ~vocab_path] exports the BPE
      merges and vocabulary in a tiktoken-compatible format. Currently only
      supported for BPE models. *)

  val save_model_files :
    t -> folder:string -> ?prefix:string -> unit -> string list
  (** [save_model_files tokenizer ~folder ?prefix ()] saves the underlying model
      files (e.g. vocab and merges). *)

  (** {1 Hugging Face Compatibility} *)

  val from_file : string -> (t, exn) result
  (** [from_file path] loads a tokenizer from HuggingFace JSON format. *)

  val from_json : Jsont.json -> (t, exn) result
  (** [from_json json] deserializes a tokenizer from HuggingFace JSON format. *)

  val to_json : t -> Jsont.json
  (** [to_json tokenizer] serializes tokenizer to HuggingFace JSON format. *)

  val save_pretrained : t -> path:string -> unit
  (** [save_pretrained tokenizer ~path] saves tokenizer to directory in
      HuggingFace format. *)
end
