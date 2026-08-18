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
      Brot.train_bpe
        (`Seq (List.to_seq texts))
        ~vocab_size:1000
        ~pre:(Brot.Pre_tokenizer.byte_level ())
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

type added_token = {
  content : string;
      (** The token text: ["[CLS]"] for a special one, ["<name>"] for an
          ordinary vocabulary entry matched atomically. *)
  special : bool;  (** Whether decoding may skip this token. *)
  single_word : bool;  (** Whether this token must match whole words only. *)
  lstrip : bool;
      (** Whether a match extends over the whitespace preceding it. *)
  rstrip : bool;
      (** Whether a match extends over the whitespace following it. *)
  normalized : bool;
      (** Whether this token is matched against normalized text rather than
          against the raw input. *)
}
(** The type for added token configurations.

    Added tokens are matched atomically in the input, ahead of the pre-tokenizer
    and the model, so ["a<pad>b"] encodes as ["a"], ["<pad>"], ["b"] whatever
    the model would make of ["<pad>"]. A token with [normalized] unset is
    matched against the raw input, before normalization; one with it set is
    matched against the normalized text, and emits what it matched there. At a
    given position the longest token wins, and earlier positions win over later
    ones.

    Token IDs are assigned automatically: a token already in the model
    vocabulary keeps its ID, the others are numbered from the end of it. A token
    with [special] set is skipped by {!decode} when [skip_special_tokens] is
    [true]; one without it is an ordinary vocabulary entry that happens to be
    matched atomically. The semantic role (pad, unk, bos, etc.) is contextual,
    not encoded in the type. *)

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

    - [`Files paths]: read training text from files, one line per example. A
      line keeps the newline that ends it, so a byte-level pipeline trained from
      a file learns a token for it.
    - [`Seq seq]: use a sequence of strings. *)

val added_token :
  ?special:bool ->
  ?single_word:bool ->
  ?lstrip:bool ->
  ?rstrip:bool ->
  ?normalized:bool ->
  string ->
  added_token
(** [added_token content] is an added token configuration for [content].

    [special] defaults to [true]; pass [~special:false] for a token that is
    matched atomically but never skipped when decoding. [single_word], [lstrip]
    and [rstrip] default to [false]. [normalized] defaults to [not special].

    The defaults line the two HuggingFace registrations up one for one:
    [added_token c] is [add_special_tokens([c])] ([special] set, [normalized]
    unset) and [added_token ~special:false c] is [add_tokens([c])] ([special]
    unset, [normalized] set). *)

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
    [max_length] tokens. [direction] defaults to [`Right]: the tokens kept are
    the first [max_length], and [`Left] keeps the last.

    Truncation runs before the post-processor, on what is left of [max_length]
    once the special tokens it will add are counted, so a special token never
    pushes content out. A [max_length] at or below that count leaves no room for
    content: the special tokens are still added and the result is longer than
    [max_length]. *)

(** {1:constructors Constructors} *)

val bpe :
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?added_tokens:added_token list ->
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
    - [added_tokens]: added tokens. They are matched atomically in the input on
      {!encode} — against the raw text before normalization when [normalized] is
      unset, against the normalized text when it is set — and are numbered from
      the end of [vocab] without entering it. Default: [[]].
    - [bos_token], [eos_token], [pad_token]: role markers. Each is a special
      token in its own right, matched in the input like the entries of
      [added_tokens] and numbered like them when [vocab] does not hold it.
      Default: none.
    - [unk_token]: token for unknown characters. Configures both the role and
      the BPE model's unknown handling. It is a model parameter, never matched
      atomically in the input. Default: none.
    - [vocab]: initial vocabulary as [(token, id)] pairs. Default: [[]].
    - [merges]: merge rules as [(left, right)] pairs learned during training.
      Default: [[]].
    - [cache_capacity]: number of slots in the pretoken cache, rounded up to a
      power of two. A slot costs 32 bytes and holds one pretoken's tokens; each
      domain encoding with the tokenizer keeps a table of its own, seeded with
      the vocabulary. Default: [262144] (8 MB). [0] disables caching.
    - [dropout]: probability \[[0]; [1]\] of skipping merges (data
      augmentation). Default: none (no dropout).
    - [continuing_subword_prefix]: prefix for non-initial subwords (e.g.,
      ["##"]). Default: none; [""] is the same as none.
    - [end_of_word_suffix]: suffix marking word boundaries (e.g., ["</w>"]).
      Default: none; [""] is the same as none.
    - [fuse_unk]: merge consecutive unknown tokens. Default: [false].
    - [byte_fallback]: use byte-level fallback (["<0x00>"]) instead of unknown
      token. Default: [false].
    - [ignore_merges]: emit a word that is itself in [vocab] as that single
      token, skipping the merges for it. Has no effect under [dropout]. Default:
      [false]. *)

val wordpiece :
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?added_tokens:added_token list ->
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

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token]) are as in {!bpe}. *)

val word_level :
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?added_tokens:added_token list ->
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

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token]) are as in {!bpe}. *)

val unigram :
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?added_tokens:added_token list ->
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

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token]) are as in {!bpe}. *)

val chars :
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?added_tokens:added_token list ->
  ?bos_token:string ->
  ?eos_token:string ->
  ?pad_token:string ->
  ?unk_token:string ->
  unit ->
  t
(** [chars ()] is a character-level tokenizer.

    Each byte in the input becomes a separate token with ID equal to its ordinal
    value. No vocabulary is required.

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token]) are as in {!bpe}. *)

val from_model_file :
  vocab:string ->
  ?merges:string ->
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?added_tokens:added_token list ->
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

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token], [unk_token]) are as in {!bpe}. *)

val add_tokens : t -> added_token list -> t
(** [add_tokens t tokens] is [t] with [tokens] registered as added tokens,
    exactly as if they had been passed as [added_tokens] at construction:
    matched atomically in the input, numbered from the end of the vocabulary,
    and skipped by {!decode} when their [special] is set. Works for every model.

    A token [t] already holds keeps the ID it has; its flags are replaced by the
    new ones. Added tokens never enter the model's own vocabulary: to build one,
    pass [vocab] to the constructor. *)

(** {1:accessors Accessors} *)

val normalizer : t -> Normalizer.t option
(** [normalizer t] is [t]'s normalizer, if any. *)

val pre_tokenizer : t -> Pre_tokenizer.t option
(** [pre_tokenizer t] is [t]'s pre-tokenizer, if any. *)

val post_processor : t -> Post_processor.t option
(** [post_processor t] is [t]'s post-processor, if any. *)

val decoder : t -> Decoder.t option
(** [decoder t] is [t]'s decoder, if any. *)

val added_tokens : t -> added_token list
(** [added_tokens t] is [t]'s added tokens: those given as [added_tokens] at
    construction, plus the [bos_token], [eos_token] and [pad_token] role
    markers. This is what {!to_json} writes. *)

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
(** [vocab t] is [t]'s vocabulary as [(token, id)] pairs, the model's followed
    by the added tokens it does not already hold. *)

val vocab_size : t -> int
(** [vocab_size t] is the number of tokens in [t]'s vocabulary, added tokens
    included. *)

val token_to_id : t -> string -> int option
(** [token_to_id t token] is the ID of [token] in [t], if any. Added tokens take
    precedence over the model. *)

val id_to_token : t -> int -> string option
(** [id_to_token t id] is the token string for [id] in [t], if any. Added tokens
    take precedence over the model, and an added token matched against
    normalized text gives the normalized form of its content, since that is the
    text it stood for: on a SentencePiece model, [id_to_token t 2] is
    ["\u{2581}</s>"] where {!token_to_id} takes ["</s>"]. *)

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

    Added tokens occurring in [text] are matched atomically and emitted with
    their own ID, whatever [add_special_tokens] is.

    {!Encoding.offsets} are byte spans of [text] itself, not of the normalized
    form: a token of ["café"] under a normalizer that strips accents reports the
    bytes of the accented text. {!Encoding.word_ids} number the pretokens of
    [text] from [0], an added token counting as one. Both are worked out when
    they are first read, so a caller that only wants {!Encoding.ids} pays for
    neither.

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
(** [encode_ids t text] is [Encoding.ids (encode t text)], without building the
    encoding when it can be avoided: the alignment metadata is what costs, and
    this asks for none of it.

    Optional parameters are as in {!encode}. *)

val decode : t -> ?skip_special_tokens:bool -> int array -> string
(** [decode t ids] is the text obtained by decoding [ids] through [t]'s
    vocabulary and decoder.

    [skip_special_tokens] defaults to [false]. It drops the added tokens whose
    [special] is set, and only those. *)

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
  ?added_tokens:added_token list ->
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

    The words counted are the pre-tokens the trained tokenizer will itself
    produce: every text is normalized by [normalizer], then cut by [pre], and
    each piece counts as one word — for a byte-level [pre] the words are in
    encoded form, and merges are learned over that form. With no [pre] a whole
    text is one word, so training on space-separated words needs
    [~pre:(Pre_tokenizer.whitespace_split ())].

    The most frequent adjacent pair of characters is then merged over and over
    until [vocab_size] is reached, no pair is left, or the best pair falls below
    [min_frequency].

    - [init]: a tokenizer whose added and special tokens carry over into the
      trained one. Its model does not: training always starts from an empty
      vocabulary. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. Default:
      [30000].
    - [min_frequency]: minimum pair frequency to be merged. Default: [0].
    - [limit_alphabet]: maximum number of distinct characters to keep; the
      rarest go first, and words drop the characters that did not make the cut.
      Default: none (keep all).
    - [initial_alphabet]: characters to keep whatever their frequency. Each
      string stands for the code point it starts with, so ["été"] and ["é"] both
      mean [é]; a string that starts with no code point — empty, or not valid
      UTF-8 — is dropped. Default: [[]].
    - [continuing_subword_prefix]: prefix put on every character of a word but
      the first, before any pair is counted, so merges are learned — and written
      — over the affixed forms. Default: none.
    - [end_of_word_suffix]: suffix put on the last character of a word, before
      any pair is counted. Default: none.
    - [show_progress]: accepted for HuggingFace compatibility; nothing is
      displayed. Default: [true].
    - [max_token_length]: holds a merge back once the run of characters it would
      join reaches that many, so tokens stay shorter than it. The merges of
      single characters that open the training are exempt. Default: none.

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token], [unk_token]) are as in {!bpe}. *)

val train_wordpiece :
  ?init:t ->
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?added_tokens:added_token list ->
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

    Learns the vocabulary with the BPE merge training of {!train_bpe}, over the
    same words, and keeps the merged tokens as WordPiece subwords.

    - [init]: a tokenizer whose added and special tokens carry over into the
      trained one. Its model does not. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. Default:
      [30000].
    - [min_frequency]: minimum pair frequency to be merged. Default: [0].
    - [limit_alphabet]: maximum number of distinct characters to keep. Default:
      none (keep all).
    - [initial_alphabet]: characters to keep whatever their frequency, one code
      point per string as in {!train_bpe}. Default: [[]].
    - [continuing_subword_prefix]: prefix for non-initial subwords. Default:
      ["##"].
    - [end_of_word_suffix]: suffix marking word boundaries. Default: none.
    - [show_progress]: accepted for HuggingFace compatibility; nothing is
      displayed. Default: [true].

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token], [unk_token]) are as in {!bpe}. *)

val train_wordlevel :
  ?init:t ->
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?added_tokens:added_token list ->
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

    The vocabulary is the most frequent of the words counted as in {!train_bpe},
    so it holds whole pre-tokens and nothing is split further.

    - [init]: a tokenizer whose added and special tokens carry over into the
      trained one. Its model does not. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. Default:
      [30000].
    - [min_frequency]: minimum frequency for a word to be included. Default:
      [0].
    - [show_progress]: accepted for HuggingFace compatibility; nothing is
      displayed. Default: [true].

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token], [unk_token]) are as in {!bpe}. *)

val train_unigram :
  ?init:t ->
  ?normalizer:Normalizer.t ->
  ?pre:Pre_tokenizer.t ->
  ?post:Post_processor.t ->
  ?decoder:Decoder.t ->
  ?added_tokens:added_token list ->
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

    {b Warning.} The EM training of the unigram model is not implemented: the
    vocabulary is the most frequent of the words counted as in {!train_bpe},
    each scored by how often it occurs, and no subword piece is ever proposed.
    [shrinking_factor], [max_piece_length] and [n_sub_iterations] are accepted
    for HuggingFace compatibility and have no effect.

    - [init]: a tokenizer whose added and special tokens carry over into the
      trained one. Its model does not. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. Default:
      [8000].
    - [show_progress]: accepted for HuggingFace compatibility; nothing is
      displayed. Default: [true].
    - [shrinking_factor], [max_piece_length], [n_sub_iterations]: inert, as
      above. Defaults: [0.75], [16], [2].

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
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

    The [added_tokens] member gives the tokens matched atomically in the input.
    A missing [special] member reads as [true], and a missing [normalized] as
    [not special], as in {!val-added_token}. Their IDs are reassigned as
    documented in {!type-added_token}, not read from the file. *)

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
