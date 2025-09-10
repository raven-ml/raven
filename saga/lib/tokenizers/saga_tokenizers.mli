(** Tokenizers library - text tokenization for ML.

    This module provides fast and flexible tokenization for machine learning
    applications, supporting multiple algorithms from simple word splitting to
    advanced subword tokenization like BPE and WordPiece.

    {1 Overview}

    The library offers two main API levels:
    - {!section-simple}: High-level functions for common tokenization tasks
    - {!section-advanced}: Advanced tokenizer construction with full
      customization

    All tokenizers work with Unicode text and handle edge cases gracefully. The
    library is designed for both interactive use and high-performance batch
    processing.

    {1 Quick Start}

    {2 Simple tokenization}
    {[
      open Saga_tokenizers

      (* Word tokenization *)
      let words = tokenize "Hello, world! How are you?"
      (* Returns: ["Hello"; ","; "world"; "!"; "How"; "are"; "you"; "?"] *)

      (* Character tokenization *)
      let chars = tokenize ~method_:`Chars "Hello"
      (* Returns: ["H"; "e"; "l"; "l"; "o"] *)

      (* Custom regex pattern *)
      let tokens = tokenize ~method_:(`Regex {|\w+|}) "hello_world 123"
      (* Returns: ["hello_world"; "123"] *)
    ]}

    {2 Encoding and decoding}
    {[
      (* Automatic vocabulary building *)
      let ids = encode "the cat sat on the mat"
      (* Returns: [0; 1; 2; 3; 0; 4] (indices for unique tokens) *)

      let vocab = vocab [ "the"; "cat"; "sat"; "on"; "mat"; "<pad>"; "<unk>" ]
      let text_ids = encode ~vocab "the cat" (* [0; 1] *)
      let recovered = decode vocab text_ids (* "the cat" *)
    ]}

    {2 Batch processing}
    {[
      let texts = [ "hello world"; "hi there"; "goodbye" ]
      let tensor = encode_batch ~max_len:5 ~pad:true texts
      (* Returns: 3x5 tensor with padding *)

      let batch_texts = decode_batch vocab tensor
      (* Returns: ["hello world"; "hi there"; "goodbye"] *)
    ]}

    {1 Key Concepts}

    {2 Tokenization Methods}

    - [`Words]: Splits on whitespace and punctuation, good for natural language
    - [`Chars]: Unicode character-level, good for morphologically rich languages
    - [`Regex pattern]: Custom splitting with regular expressions

    {2 Vocabularies}

    Vocabularies map tokens to numeric indices. They automatically include
    special tokens:
    - [<pad>]: Padding token (index 0)
    - [<unk>]: Unknown token (index 1)
    - [<bos>]: Beginning of sequence (index 2)
    - [<eos>]: End of sequence (index 3)

    {2 Subword Tokenization}

    For handling out-of-vocabulary words and morphologically complex languages:
    - {!Bpe}: Byte-Pair Encoding (GPT-style)
    - {!Wordpiece}: WordPiece algorithm (BERT-style)

    {1 Performance Considerations}

    - Character tokenization is fastest for short texts
    - Word tokenization scales well with vocabulary size
    - BPE/WordPiece are slower but handle OOV words better
    - Use {!encode_batch} for processing many texts efficiently
    - Consider {!vocab} size limits for memory usage *)

module Unicode = Unicode
(** Unicode text processing utilities *)

(** {1 Core Types} *)

type vocab
(** Vocabulary mapping between tokens and indices.

    Vocabularies maintain bidirectional mappings between string tokens and
    integer indices. They automatically include standard special tokens and can
    be saved/loaded for persistence. *)

type tokenizer_method =
  [ `Words
    (** Split on whitespace and punctuation, preserving individual punctuation
        marks *)
  | `Chars
    (** Unicode character-level tokenization, handles all Unicode properly *)
  | `Regex of string
    (** Custom regex pattern for domain-specific tokenization *) ]
(** Built-in tokenization methods for the simple API.

    Each method handles Unicode correctly and provides reasonable defaults for
    common use cases. *)

(** {1:section-simple Simple API - Common Use Cases}

    High-level functions that handle most tokenization needs with minimal setup.
    These functions automatically manage vocabularies and provide sensible
    defaults. *)

val tokenize : ?method_:tokenizer_method -> string -> string list
(** [tokenize ?method_ text] splits text into a list of tokens.

    @param method_ Tokenization method to use (default: [`Words])
    @param text Input text to tokenize
    @return List of token strings

    The default [`Words] method splits on whitespace and treats punctuation as
    separate tokens, which works well for most natural language processing
    tasks.

    Examples:
    {[
      tokenize "Hello world!"
      = [ "Hello"; "world"; "!" ] tokenize ~method_:`Chars "Hi!"
      = [ "H"; "i"; "!" ] tokenize ~method_:(`Regex {|\w+|\d+|})
          "hello123 world456"
      = [ "hello"; "123"; "world"; "456" ]
    ]} *)

val encode : ?vocab:vocab -> string -> int list
(** [encode ?vocab text] tokenizes text and converts tokens to integer indices.

    @param vocab Vocabulary to use for encoding (auto-built if omitted)
    @param text Input text to encode
    @return List of token indices

    When vocab is omitted, creates a vocabulary from the unique tokens in the
    input text. Unknown tokens (not in vocab) are mapped to the UNK token index.

    Examples:
    {[
      (* Automatic vocabulary building *)
      encode "hello world hello" = [0; 1; 0]

      (* With explicit vocabulary *)
      let v = vocab ["hello"; "world"; "goodbye"] in
      encode ~vocab:v "hello world" = [0; 1]
      encode ~vocab:v "hello mars" = [0; 1]  (* "mars" -> UNK index *)
    ]} *)

val encode_batch :
  ?vocab:vocab ->
  ?max_len:int ->
  ?pad:bool ->
  string list ->
  (int32, Bigarray.int32_elt) Nx.t
(** [encode_batch ?vocab ?max_len ?pad texts] encodes multiple texts to a padded
    tensor.

    @param vocab Vocabulary for encoding (auto-built from all texts if omitted)
    @param max_len Maximum sequence length (default: length of longest sequence)
    @param pad Whether to pad shorter sequences (default: false)
    @param texts List of input texts
    @return 2D tensor with shape [num_texts, max_len]

    Efficient batch processing for machine learning pipelines. Auto-builds
    vocabulary from all input texts when not provided, ensuring consistent
    encoding.

    Examples:
    {[
      encode_batch [ "hi"; "hello world" ]
        ~pad:true
          (* Returns: 2x3 tensor [[hi_id; pad_id; pad_id]; [hello_id; world_id;
             pad_id]] *)
        encode_batch
        [ "short"; "much longer text" ]
        ~max_len:2 ~pad:true
      (* Truncates longer sequences to max_len *)
    ]} *)

val decode : vocab -> int list -> string
(** [decode vocab indices] converts token indices back to text.

    @param vocab Vocabulary containing index-to-token mappings
    @param indices List of token indices to decode
    @return Reconstructed text string

    Special tokens (PAD, UNK, BOS, EOS) are handled appropriately during
    decoding. Invalid indices are replaced with UNK tokens.

    Examples:
    {[
      let v = vocab [ "hello"; "world" ] in
      decode v [ 0; 1 ]
      = "hello world" decode v [ 1; 0; 1 ]
      = "world hello world"
    ]} *)

val decode_batch : vocab -> (int32, Bigarray.int32_elt) Nx.t -> string list
(** [decode_batch vocab tensor] decodes a batch of token sequences to texts.

    @param vocab Vocabulary for decoding
    @param tensor 2D tensor of shape [batch_size, seq_len] with token indices
    @return List of decoded text strings

    Efficiently processes multiple sequences in parallel. Automatically handles
    padding tokens and sequence boundaries.

    Example:
    {[
      let tensor = encode_batch [ "hello"; "world" ] ~pad:true in
      decode_batch vocab tensor = [ "hello"; "world" ]
    ]} *)

(** {1 Vocabulary Management}

    Functions for creating, managing, and persisting vocabularies. *)

val vocab : ?max_size:int -> ?min_freq:int -> string list -> vocab
(** [vocab ?max_size ?min_freq tokens] builds a vocabulary from a list of
    tokens.

    @param max_size Maximum vocabulary size (default: unlimited)
    @param min_freq Minimum frequency for token inclusion (default: 1)
    @param tokens List of tokens to build vocabulary from
    @return New vocabulary with special tokens included

    Special tokens are automatically added with reserved indices:
    - [<pad>] (index 0): For sequence padding
    - [<unk>] (index 1): For out-of-vocabulary tokens
    - [<bos>] (index 2): Beginning of sequence marker
    - [<eos>] (index 3): End of sequence marker

    When max_size is specified, keeps the most frequent tokens up to that limit.
    Tokens below min_freq threshold are excluded (except special tokens).

    Examples:
    {[
      let v = vocab ["hello"; "world"; "hello"; "goodbye"] in
      vocab_size v = 7  (* 4 user tokens + 4 special tokens *)

      let v_limited = vocab ~max_size:10 ~min_freq:2
        ["the"; "the"; "cat"; "sat"; "the"] in
      (* Only "the" meets min_freq=2, plus special tokens *)
    ]} *)

val vocab_size : vocab -> int
(** [vocab_size v] returns number of tokens in vocabulary. *)

val vocab_save : vocab -> string -> unit
(** [vocab_save v path] saves vocabulary to file. *)

val vocab_load : string -> vocab
(** [vocab_load path] loads vocabulary from file. *)

(** {2 Text Preprocessing} *)

val normalize :
  ?lowercase:bool ->
  ?strip_accents:bool ->
  ?collapse_whitespace:bool ->
  string ->
  string
(** [normalize ?lowercase ?strip_accents ?collapse_whitespace text] applies
    normalization.

    All options default to false.

    {[
      normalize ~lowercase:true "Hello  WORLD!" = "hello world!"
    ]} *)

(** {2 Advanced API - Custom Tokenizers} *)

module Tokenizer : sig
  type 'a t
  (** Tokenizer type with phantom type parameter for compile-time configuration
      tracking.

      The phantom type parameter 'a tracks the tokenization algorithm used
      (e.g., [`Words], [`BPE], etc.) This enables type-safe configuration and
      helps catch mismatched tokenizer/vocab combinations at compile time. *)

  val words : [ `Words ] t
  (** [words] creates a word-level tokenizer.

      Splits text on whitespace and punctuation boundaries. Simple and
      interpretable, but may struggle with out-of-vocabulary words and
      morphologically rich languages.

      Good for:
      - Simple applications where interpretability matters
      - Languages with clear word boundaries
      - Cases where vocabulary size can be controlled

      {[
        let tokenizer = Saga_tokenizers.Tokenizer.words in
        let tokens = Saga_tokenizers.Tokenizer.run tokenizer "Hello, world!" in
        (* Result: ["Hello"; ","; "world"; "!"] *)
      ]} *)

  val chars : [ `Chars ] t
  (** [chars] creates a character-level tokenizer.

      Splits text into individual Unicode characters. Handles any text robustly
      but loses word-level semantic information and creates very long sequences.

      Good for:
      - Languages without clear word boundaries (Chinese, Japanese)
      - Handling any possible input text
      - When training data is limited
      - Morphologically rich languages

      {[
        let tokenizer = Saga_tokenizers.Tokenizer.chars in
        let tokens = Saga_tokenizers.Tokenizer.run tokenizer "Hello!" in
        (* Result: ["H"; "e"; "l"; "l"; "o"; "!"] *)
      ]} *)

  val regex : string -> [ `Regex ] t
  (** [regex pattern] creates a regex-based tokenizer.
      
      Uses the provided regular expression to split text into tokens.
      Provides maximum flexibility but requires careful pattern design.
      
      @param pattern Regular expression pattern for tokenization.
      
      Pattern design tips:
      - Use capturing groups to keep delimiters: "(\\w+|\\W+)"
      - Consider Unicode: "\\p{L}+" for letters across languages
      - Handle whitespace: "\\s+" for whitespace sequences
      
      {[
        let tokenizer = Saga_tokenizers.Tokenizer.regex "\\w+|\\W+" in
        let tokens = Saga_tokenizers.Tokenizer.run tokenizer "Hello, world!" in
        (* Result: ["Hello"; ","; " "; "world"; "!"] *)
        
        let tokenizer2 = Saga_tokenizers.Tokenizer.regex "\\d+|\\w+|\\W+" in
        let tokens2 = Saga_tokenizers.Tokenizer.run tokenizer2 "Price: $123.45" in
        (* Result: ["Price"; ":"; " "; "$"; "123"; "."; "45"] *)
      ]} *)

  val bpe : vocab:string -> merges:string -> [ `BPE ] t
  (** [bpe ~vocab ~merges] creates a Byte-Pair Encoding tokenizer.

      BPE learns subword units by iteratively merging the most frequent
      character pairs. Balances vocabulary size with sequence length, handling
      out-of-vocabulary words gracefully.

      @param vocab Path to vocabulary file or vocabulary content.
      @param merges Path to merges file or merge rules content.

      BPE merges are applied in order of learning. Files should be formatted as:
      - Vocab: One token per line
      - Merges: "token1 token2" pairs per line, in merge order

      {[
        let tokenizer = Saga_tokenizers.Tokenizer.bpe
          ~vocab:"path/to/vocab.txt"
          ~merges:"path/to/merges.txt" in
        let tokens = Saga_tokenizers.Tokenizer.run tokenizer "unhappiness" in
        (* Result might be: ["un"; "happy"; "ness"] depending on learned merges *)
      ]} *)

  val wordpiece : vocab:string -> unk_token:string -> [ `WordPiece ] t
  (** [wordpiece ~vocab ~unk_token] creates a WordPiece tokenizer.

      WordPiece uses a greedy longest-match-first approach to split words into
      subword pieces. Used by BERT and other transformer models. Handles
      out-of-vocabulary by using unk_token.

      @param vocab
        Path to vocabulary file or vocabulary content. Should contain subword
        pieces.
      @param unk_token Token to use for unknown/out-of-vocabulary subwords.

      WordPiece algorithm: 1. Try to match longest possible subword from start
      of word 2. If no match found, use unk_token 3. Move to next unmatched
      position and repeat 4. Continue until entire word is processed

      {[
        let tokenizer = Saga_tokenizers.Tokenizer.wordpiece
          ~vocab:"path/to/bert-vocab.txt"
          ~unk_token:"[UNK]" in
        let tokens = Saga_tokenizers.Tokenizer.run tokenizer "unhappiness" in
        (* Result might be: ["un"; "##happy"; "##ness"] with ## indicating continuations *)
      ]} *)

  val run : _ t -> string -> string list
  (** [run tokenizer text] tokenizes text into a list of token strings.

      Applies the tokenizer to input text and returns tokens without offset
      information. This is the most common tokenization interface for typical
      use cases.

      @param tokenizer The tokenizer to apply (any type).
      @param text Input text to tokenize.
      @return List of token strings.

      {[
        let tokenizer = Saga_tokenizers.Tokenizer.words in
        let tokens = Saga_tokenizers.Tokenizer.run tokenizer "Hello world!" in
        (* Result: ["Hello"; "world"; "!"] *)

        (* Chain with normalization and pre-tokenization: *)
        let tokenizer_with_preprocessing = tokenizer
          |> Saga_tokenizers.Tokenizer.with_normalizer (fun s -> String.lowercase_ascii s)
          |> Saga_tokenizers.Tokenizer.with_pre_tokenizer Pre_tokenizers.punctuation in
        let tokens = Saga_tokenizers.Tokenizer.run tokenizer_with_preprocessing "Hello, World!" in
        (* Applies lowercase normalization and punctuation pre-tokenization *)
      ]} *)

  val run_with_offsets : _ t -> string -> (string * int * int) list
  (** [run_with_offsets tokenizer text] tokenizes text with character offset
      information.

      Like {!run} but also returns the character positions where each token
      appears in the original text. Essential for tasks that need to map tokens
      back to source text positions.

      @param tokenizer The tokenizer to apply.
      @param text Input text to tokenize.
      @return List of (token, start_offset, end_offset) tuples.

      Use cases:
      - Named entity recognition (highlighting detected entities)
      - Question answering (extracting answer spans)
      - Syntax highlighting in editors
      - Error reporting with precise locations

      {[
        let tokenizer = Saga_tokenizers.Tokenizer.chars in
        let tokens_with_offsets =
          Saga_tokenizers.Tokenizer.run_with_offsets tokenizer "Hi!"
        in
        (* Result: [("H", 0, 1); ("i", 1, 2); ("!", 2, 3)] *)

        (* Verify offset correctness: *)
        List.iter
          (fun (token, start, end_) ->
            let extracted = String.sub text start (end_ - start) in
            assert (extracted = token))
          tokens_with_offsets
      ]} *)

  val with_normalizer : (string -> string) -> 'a t -> 'a t
  (** [with_normalizer normalizer tokenizer] adds text normalization to
      tokenizer pipeline.

      Applies the normalization function to input text before tokenization.
      Common normalizations include lowercasing, accent removal, whitespace
      normalization.

      @param normalizer Function to normalize input text.
      @param tokenizer Base tokenizer to extend.
      @return New tokenizer with normalization step.

      Pipeline order: input text → normalization → pre-tokenization →
      tokenization

      {[
        let lowercase_normalizer text = String.lowercase_ascii text in
        let base_tokenizer = Saga_tokenizers.Tokenizer.words in
        let tokenizer = Saga_tokenizers.Tokenizer.with_normalizer lowercase_normalizer base_tokenizer in

        let tokens = Saga_tokenizers.Tokenizer.run tokenizer "Hello WORLD!" in
        (* Result: ["hello"; "world"; "!"] - notice lowercasing *)

        (* Custom normalization for social media text: *)
        let social_normalizer text =
          text
          |> Str.global_replace (Str.regexp "@[\\w]+") "<USER>"  (* Replace mentions *)
          |> Str.global_replace (Str.regexp "http[s]?://[^\\s]+") "<URL>"  (* Replace URLs *)
          |> String.lowercase_ascii
        in
        let social_tokenizer = Saga_tokenizers.Tokenizer.with_normalizer social_normalizer base_tokenizer in
        let tokens = Saga_tokenizers.Tokenizer.run social_tokenizer "Check out @alice's post: https://example.com!" in
        (* Result: ["check"; "out"; "<user>"; "'s"; "post"; ":"; "<url>"; "!"] *)
      ]} *)

  val with_pre_tokenizer : Pre_tokenizers.t -> 'a t -> 'a t
  (** [with_pre_tokenizer pre_tokenizer tokenizer] adds pre-tokenization step.

      Applies pre-tokenization to split text before the main tokenization
      algorithm. Useful for handling punctuation, whitespace, or format-specific
      splitting.

      @param pre_tokenizer Pre-tokenizer from {!Pre_tokenizers} module.
      @param tokenizer Base tokenizer to extend.
      @return New tokenizer with pre-tokenization step.

      Pipeline order: input text → normalization → pre-tokenization →
      tokenization

      {[
        let base_tokenizer = Saga_tokenizers.Tokenizer.bpe ~vocab:"vocab.txt" ~merges:"merges.txt" in
        let tokenizer = Saga_tokenizers.Tokenizer.with_pre_tokenizer
          (Pre_tokenizers.byte_level ~add_prefix_space:true ())
          base_tokenizer in

        let tokens = Saga_tokenizers.Tokenizer.run tokenizer "Hello world!" in
        (* Applies byte-level pre-tokenization before BPE *)

        (* Chain multiple pre-tokenizers: *)
        let complex_pre_tokenizer = Pre_tokenizers.sequence [
          Pre_tokenizers.punctuation ~behavior:`Isolated ();
          Pre_tokenizers.digits ~individual_digits:false ();
        ] in
        let tokenizer = Saga_tokenizers.Tokenizer.with_pre_tokenizer
          complex_pre_tokenizer base_tokenizer in
        let tokens = Saga_tokenizers.Tokenizer.run tokenizer "Price: $123.45!" in
        (* Handles punctuation and digits before BPE processing *)
      ]} *)
end

module Vocab : sig
  type t = vocab
  (** Vocabulary type for mapping between tokens and indices.

      Vocabularies maintain bidirectional mappings between string tokens and
      integer indices. Essential for converting between human-readable text and
      model-compatible integer sequences. *)

  val create : unit -> t
  (** [create ()] creates a new empty vocabulary.

      The vocabulary starts with no tokens. You'll typically want to add special
      tokens first (PAD, UNK, BOS, EOS) to ensure they get consistent low
      indices.

      {[
        let vocab = Saga_tokenizers.Vocab.create () in
        (* Add special tokens first for consistent indices *)
        Saga_tokenizers.Vocab.add vocab "<PAD>";
        Saga_tokenizers.Vocab.add vocab "<UNK>";
        Saga_tokenizers.Vocab.add vocab "<BOS>";
        Saga_tokenizers.Vocab.add vocab "<EOS>";
        (* Now add regular tokens *)
        Saga_tokenizers.Vocab.add vocab "hello";
        Saga_tokenizers.Vocab.add vocab "world"
      ]} *)

  val add : t -> string -> unit
  (** [add vocab token] adds a token to the vocabulary.

      If the token already exists, this is a no-op. New tokens are assigned the
      next available index (starting from 0). Order of addition determines
      indices.

      @param vocab Vocabulary to modify.
      @param token String token to add.

      {[
        let vocab = Saga_tokenizers.Vocab.create () in
        Saga_tokenizers.Vocab.add vocab "hello";
        Saga_tokenizers.Vocab.add vocab "world";
        Saga_tokenizers.Vocab.add vocab "hello";

        (* No-op: already exists *)
        assert (Saga_tokenizers.Vocab.size vocab = 2);
        assert (Saga_tokenizers.Vocab.get_index vocab "hello" = Some 0);
        assert (Saga_tokenizers.Vocab.get_index vocab "world" = Some 1)
      ]} *)

  val add_batch : t -> string list -> unit
  (** [add_batch vocab tokens] adds multiple tokens to the vocabulary.

      Equivalent to calling {!add} for each token in the list, but more
      efficient. Maintains the same ordering behavior: first unique token gets
      lowest index.

      @param vocab Vocabulary to modify.
      @param tokens List of tokens to add.

      {[
        let vocab = Saga_tokenizers.Vocab.create () in
        Saga_tokenizers.Vocab.add_batch vocab
          [ "<PAD>"; "<UNK>"; "hello"; "world"; "hello" ];

        assert (Saga_tokenizers.Vocab.size vocab = 4);
        (* "hello" counted once *)
        assert (Saga_tokenizers.Vocab.get_index vocab "<PAD>" = Some 0);
        assert (Saga_tokenizers.Vocab.get_index vocab "<UNK>" = Some 1);
        assert (Saga_tokenizers.Vocab.get_index vocab "hello" = Some 2);
        assert (Saga_tokenizers.Vocab.get_index vocab "world" = Some 3)
      ]} *)

  val get_index : t -> string -> int option
  (** [get_index vocab token] retrieves the index of a token.

      Returns [Some index] if the token exists in the vocabulary, [None] if the
      token is not found.

      @param vocab Vocabulary to query.
      @param token String token to look up.
      @return Token index or None if not found.

      {[
        let vocab = Saga_tokenizers.Vocab.create () in
        Saga_tokenizers.Vocab.add vocab "hello";

        match Saga_tokenizers.Vocab.get_index vocab "hello" with
        | Some idx -> Printf.printf "hello has index %d" idx
        | None -> Printf.printf "hello not found"

        match Saga_tokenizers.Vocab.get_index vocab "goodbye" with
        | Some idx -> Printf.printf "goodbye has index %d" idx  (* Won't happen *)
        | None -> Printf.printf "goodbye not in vocabulary"  (* This will print *)
      ]} *)

  val get_token : t -> int -> string option
  (** [get_token vocab index] retrieves the token at a given index.

      Returns [Some token] if the index is valid (0 <= index < size), [None] if
      the index is out of bounds.

      @param vocab Vocabulary to query.
      @param index Index to look up.
      @return Token string or None if index invalid.

      {[
        let vocab = Saga_tokenizers.Vocab.create () in
        Saga_tokenizers.Vocab.add_batch vocab ["hello"; "world"];

        match Saga_tokenizers.Vocab.get_token vocab 0 with
        | Some token -> Printf.printf "Index 0: %s" token  (* "hello" *)
        | None -> Printf.printf "Invalid index 0"

        match Saga_tokenizers.Vocab.get_token vocab 99 with
        | Some token -> Printf.printf "Index 99: %s" token  (* Won't happen *)
        | None -> Printf.printf "Index 99 out of bounds"  (* This will print *)
      ]} *)

  val from_tokens : ?max_size:int -> ?min_freq:int -> string list -> t
  (** [from_tokens ?max_size ?min_freq tokens] builds vocabulary from token
      frequency.

      Analyzes token frequencies and builds vocabulary from most common tokens
      first. Useful for creating vocabularies from large text corpora with
      frequency-based filtering.

      @param max_size
        Maximum vocabulary size. Keeps only the most frequent tokens. Default:
        unlimited.
      @param min_freq
        Minimum frequency threshold. Tokens appearing fewer times are excluded.
        Default: 1.
      @param tokens List of all tokens from training corpus (with repetitions).
      @return New vocabulary built from filtered tokens.

      Algorithm: 1. Count frequency of each unique token 2. Filter tokens by
      min_freq threshold 3. Sort by frequency (descending) 4. Take top max_size
      tokens 5. Build vocabulary with frequency order

      {[
        let corpus_tokens = ["the"; "cat"; "sat"; "on"; "the"; "mat"; "the"; "dog"; "ran"] in

        (* Build full vocabulary *)
        let vocab1 = Saga_tokenizers.Vocab.from_tokens corpus_tokens in
        (* Result order by frequency: "the" (3), "cat" (1), "sat" (1), "on" (1), "mat" (1), "dog" (1), "ran" (1) *)

        (* Limit vocabulary size *)
        let vocab2 = Saga_tokenizers.Vocab.from_tokens ~max_size:3 corpus_tokens in
        (* Keeps only: "the", "cat", "sat" (or other tokens tied at frequency 1) *)

        (* Require minimum frequency *)
        let vocab3 = Saga_tokenizers.Vocab.from_tokens ~min_freq:2 corpus_tokens in
        (* Keeps only: "the" (appears 3 times, >= 2) *)

        (* Combine filters *)
        let vocab4 = Saga_tokenizers.Vocab.from_tokens ~max_size:5 ~min_freq:1 corpus_tokens in
        (* Top 5 tokens with frequency >= 1 *)
      ]} *)

  val size : t -> int
  (** [size vocab] returns the number of tokens in the vocabulary.

      @param vocab Vocabulary to measure.
      @return Total number of unique tokens.

      {[
        let vocab = Saga_tokenizers.Vocab.create () in
        assert (Saga_tokenizers.Vocab.size vocab = 0);

        Saga_tokenizers.Vocab.add vocab "hello";
        assert (Saga_tokenizers.Vocab.size vocab = 1);

        Saga_tokenizers.Vocab.add_batch vocab [ "world"; "!" ];
        assert (Saga_tokenizers.Vocab.size vocab = 3)
      ]} *)

  (** {2 Special Token Indices}

      Standard special tokens used in many NLP models. These functions assume
      you've added the corresponding special tokens to your vocabulary. *)

  val pad_idx : t -> int
  (** [pad_idx vocab] returns the index of the padding token.

      Padding tokens are used to make sequences the same length in batches.
      Conventionally the "<PAD>" or "[PAD]" token, often at index 0.

      @return Index of padding token.
      @raise Not_found if no padding token exists in vocabulary.

      {[
        let vocab = Saga_tokenizers.Vocab.create () in
        Saga_tokenizers.Vocab.add vocab "<PAD>";
        Saga_tokenizers.Vocab.add vocab "hello";

        let pad_id = Saga_tokenizers.Vocab.pad_idx vocab in
        assert (pad_id = 0)
      ]} *)

  val unk_idx : t -> int
  (** [unk_idx vocab] returns the index of the unknown token.

      Unknown tokens represent out-of-vocabulary words during tokenization.
      Conventionally "<UNK>", "[UNK]", or "<unk>".

      @return Index of unknown token.
      @raise Not_found if no unknown token exists in vocabulary. *)

  val bos_idx : t -> int
  (** [bos_idx vocab] returns the index of the beginning-of-sequence token.

      Used to mark the start of sequences in many language models.
      Conventionally "<BOS>", "<s>", or "[CLS]".

      @return Index of beginning-of-sequence token.
      @raise Not_found if no BOS token exists in vocabulary. *)

  val eos_idx : t -> int
  (** [eos_idx vocab] returns the index of the end-of-sequence token.

      Used to mark the end of sequences, important for generation tasks.
      Conventionally "<EOS>", "</s>", or "[SEP]".

      @return Index of end-of-sequence token.
      @raise Not_found if no EOS token exists in vocabulary.

      {[
        let vocab = Saga_tokenizers.Vocab.create () in
        Saga_tokenizers.Vocab.add_batch vocab ["<PAD>"; "<UNK>"; "<BOS>"; "<EOS>"];

        let pad_id = Saga_tokenizers.Vocab.pad_idx vocab in   (* 0 *)
        let unk_id = Saga_tokenizers.Vocab.unk_idx vocab in   (* 1 *)
        let bos_id = Saga_tokenizers.Vocab.bos_idx vocab in   (* 2 *)
        let eos_id = Saga_tokenizers.Vocab.eos_idx vocab in   (* 3 *)

        (* Use in sequence processing *)
        let encode_with_special tokens vocab =
          let bos = [Saga_tokenizers.Vocab.bos_idx vocab] in
          let eos = [Saga_tokenizers.Vocab.eos_idx vocab] in
          let token_ids = List.map (fun token ->
            match Saga_tokenizers.Vocab.get_index vocab token with
            | Some idx -> idx
            | None -> Saga_tokenizers.Vocab.unk_idx vocab
          ) tokens in
          bos @ token_ids @ eos
        in
        let ids = encode_with_special ["hello"; "world"] vocab in
        (* Result: [2; <hello_idx>; <world_idx>; 3] with BOS/EOS wrapping *)
      ]} *)
end

(** {2 Tokenizers}

    Specialized tokenizer modules for different algorithms and approaches. *)

module Bpe = Bpe
(** Byte-Pair Encoding tokenizer implementation.

    BPE learns subword units by iteratively merging the most frequent character
    pairs in a training corpus. Provides a good balance between vocabulary size
    and sequence length, handling out-of-vocabulary words gracefully.

    Key features:
    - Subword-level tokenization
    - Handles OOV words by decomposition
    - Configurable vocabulary size
    - Language-agnostic approach

    Use {!Tokenizer.bpe} for the high-level interface, or this module for direct
    access to BPE-specific functionality. *)

module Wordpiece = Wordpiece
(** WordPiece tokenizer implementation.

    WordPiece uses a greedy longest-match-first approach to split words into
    subword pieces. Originally developed for BERT and other transformer models.
    Maximizes the use of known subwords while falling back to unknown tokens for
    unseen pieces.

    Key features:
    - Greedy longest-first matching
    - Continuation token markers (##)
    - Efficient for transformer models
    - Robust OOV handling

    Use {!Tokenizer.wordpiece} for the high-level interface, or this module for
    direct access to WordPiece-specific functionality. *)

module Pre_tokenizers = Pre_tokenizers
(** Pre-tokenizers for text splitting before vocabulary-based tokenization.

    Pre-tokenization is the first stage in most tokenization pipelines,
    splitting raw text into pieces before applying vocabulary-based algorithms
    like BPE or WordPiece.

    This module provides various splitting strategies:
    - Language-aware splitting (whitespace, punctuation)
    - Format-specific splitting (byte-level, metaspace)
    - Custom pattern-based splitting
    - Composable pipeline building

    See {!Pre_tokenizers} module documentation for detailed usage examples and
    guidance on choosing appropriate pre-tokenizers for different languages and
    use cases.

    {1 Advanced API Usage Examples}

    The following examples demonstrate how to use the advanced tokenizer API to
    build custom tokenization pipelines:

    {2 Building a Complete Tokenization Pipeline}

    {[
      (* Create vocabulary from training data *)
      let training_tokens =
        [ "hello"; "world"; "hello"; "!"; "good"; "morning"; "world" ]
      in
      let vocab =
        Saga_tokenizers.Vocab.from_tokens ~max_size:1000 ~min_freq:1
          training_tokens
      in

      (* Add special tokens *)
      let vocab = Saga_tokenizers.Vocab.create () in
      Saga_tokenizers.Vocab.add_batch vocab
        [ "<PAD>"; "<UNK>"; "<BOS>"; "<EOS>" ];
      List.iter (Saga_tokenizers.Vocab.add vocab) training_tokens;

      (* Create tokenizer with custom pipeline *)
      let tokenizer =
        Saga_tokenizers.Tokenizer.words
        |> Saga_tokenizers.Tokenizer.with_normalizer String.lowercase_ascii
        |> Saga_tokenizers.Tokenizer.with_pre_tokenizer
             (Pre_tokenizers.punctuation ~behavior:`Isolated ())
      in

      (* Tokenize with offset tracking *)
      let text = "Hello, World! Good morning." in
      let tokens_with_offsets =
        Saga_tokenizers.Tokenizer.run_with_offsets tokenizer text
      in
      List.iter
        (fun (token, start, end_) ->
          Printf.printf "'%s' at %d-%d\n" token start end_)
        tokens_with_offsets
    ]}

    {2 Advanced BPE Pipeline}

    {[
      (* Build vocabulary with frequency filtering *)
      let corpus = load_large_corpus () in
      (* Your corpus loading function *)
      let all_tokens = List.concat_map tokenize_simple corpus in
      let vocab =
        Saga_tokenizers.Vocab.from_tokens
          ~max_size:30000 (* Common vocabulary size *)
          ~min_freq:5 (* Filter rare tokens *)
          all_tokens
      in

      (* Create BPE tokenizer with byte-level pre-processing *)
      let tokenizer =
        Saga_tokenizers.Tokenizer.bpe ~vocab:"path/to/vocab.json"
          ~merges:"path/to/merges.txt"
        |> Saga_tokenizers.Tokenizer.with_pre_tokenizer
             (Pre_tokenizers.byte_level ~add_prefix_space:true ~use_regex:true
                ())
        |> Saga_tokenizers.Tokenizer.with_normalizer normalize_unicode
      in

      (* Process batch of texts efficiently *)
      let texts = [ "Hello world!"; "How are you?"; "I'm fine, thanks!" ] in
      let all_tokens =
        List.map (Saga_tokenizers.Tokenizer.run tokenizer) texts
      in
      List.iter
        (fun tokens ->
          Printf.printf "Tokens: [%s]\n" (String.concat "; " tokens))
        all_tokens
    ]}

    {2 Multilingual Text Processing}

    {[
      (* Handle multilingual text with script-aware splitting *)
      let multilingual_tokenizer =
        Saga_tokenizers.Tokenizer.wordpiece ~vocab:"multilingual-vocab.txt"
          ~unk_token:"[UNK]"
        |> Saga_tokenizers.Tokenizer.with_pre_tokenizer
             (Pre_tokenizers.sequence
                [
                  Pre_tokenizers.unicode_scripts;
                  (* Split on script boundaries *)
                  Pre_tokenizers.punctuation ~behavior:`Isolated ();
                  Pre_tokenizers.whitespace;
                ])
        |> Saga_tokenizers.Tokenizer.with_normalizer (fun text ->
               text |> Unicode.normalize_nfc (* Normalize Unicode *)
               |> String.lowercase_ascii (* Lowercase for consistency *))
      in

      let multilingual_text = "Hello 你好 γειά соу! How are you?" in
      let tokens =
        Saga_tokenizers.Tokenizer.run multilingual_tokenizer multilingual_text
      in
      Printf.printf "Multilingual tokens: [%s]\n" (String.concat "; " tokens)
    ]}

    {2 Domain-Specific Tokenization}

    {[
      (* Code tokenization with custom preprocessing *)
      let code_tokenizer =
        let code_normalizer code =
          code
          |> Str.global_replace
               (Str.regexp "/\\*.*\\*/")
               "" (* Remove comments *)
          |> Str.global_replace (Str.regexp "[ \\t]+")
               " " (* Normalize whitespace *)
          |> String.trim
        in
        let code_pre_tokenizer =
          Pre_tokenizers.sequence
            [
              Pre_tokenizers.split ~pattern:"->" ~behavior:`Isolated ();
              (* Arrow operators *)
              Pre_tokenizers.split ~pattern:"::" ~behavior:`Isolated ();
              (* Scope resolution *)
              Pre_tokenizers.punctuation ~behavior:`Merged_with_next ();
              Pre_tokenizers.whitespace_split;
            ]
        in
        Saga_tokenizers.Tokenizer.bpe ~vocab:"code-vocab.json"
          ~merges:"code-merges.txt"
        |> Saga_tokenizers.Tokenizer.with_normalizer code_normalizer
        |> Saga_tokenizers.Tokenizer.with_pre_tokenizer code_pre_tokenizer
      in

      let code_sample = "let result = func(x, y) -> Some(x + y)" in
      let code_tokens =
        Saga_tokenizers.Tokenizer.run code_tokenizer code_sample
      in
      Printf.printf "Code tokens: [%s]\n" (String.concat "; " code_tokens)
    ]}

    {2 Error Handling and Validation}

    {[
      (* Robust tokenization with error handling *)
      let safe_tokenize tokenizer text =
        try
          let tokens = Saga_tokenizers.Tokenizer.run tokenizer text in
          Ok tokens
        with
        | Invalid_argument msg -> Error ("Tokenization failed: " ^ msg)
        | exn -> Error ("Unexpected error: " ^ Printexc.to_string exn)
      in

      let vocab = Saga_tokenizers.Vocab.create () in
      Saga_tokenizers.Vocab.add_batch vocab
        [ "<PAD>"; "<UNK>"; "hello"; "world" ];

      let tokenizer = Saga_tokenizers.Tokenizer.words in

      (* Test with various inputs *)
      let test_texts =
        [ "hello world"; ""; "hello unknown_token world"; "special chars: ☀☁☂" ]
      in
      List.iter
        (fun text ->
          match safe_tokenize tokenizer text with
          | Ok tokens ->
              Printf.printf "'%s' -> [%s]\n" text (String.concat "; " tokens)
          | Error err -> Printf.printf "Error tokenizing '%s': %s\n" text err)
        test_texts
    ]}

    These examples demonstrate the flexibility and power of the advanced
    tokenization API. You can mix and match components to build tokenization
    pipelines suited to your specific domain and requirements. *)
