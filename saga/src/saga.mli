(** Fast tokenization for ML in OCaml *)

(** {2 Core Types} *)

type vocab
(** Vocabulary mapping between tokens and indices *)

type tokenizer_method =
  [ `Words  (** Split on whitespace and punctuation *)
  | `Chars  (** Unicode character-level tokenization *)
  | `Regex of string  (** Custom regex pattern *) ]

(** {2 Simple API - Common Use Cases} *)

val tokenize : ?method_:tokenizer_method -> string -> string list
(** [tokenize ?method_ text] splits text into tokens.

    Default method is [`Words].

    {[
      tokenize "Hello world!"
      = [ "Hello"; "world!" ] tokenize ~method_:`Chars "Hi!"
      = [ "H"; "i"; "!" ]
    ]} *)

val encode : ?vocab:vocab -> string -> int list
(** [encode ?vocab text] tokenizes and encodes text to indices.

    If vocab not provided, builds one automatically from input.

    {[
      encode "hello world hello" = [ 0; 1; 0 ]
    ]} *)

val encode_batch :
  ?vocab:vocab ->
  ?max_len:int ->
  ?pad:bool ->
  string list ->
  (int32, Bigarray.int32_elt) Nx.t
(** [encode_batch ?vocab ?max_len ?pad texts] encodes multiple texts to tensor.

    - [max_len]: Maximum sequence length
    - [pad]: Whether to pad sequences
    - Auto-builds vocab from texts if not provided

    {[
      encode_batch [ "hi"; "hello world" ]
      (* Returns 2x3 tensor with padding *)
    ]} *)

val decode : vocab -> int list -> string
(** [decode vocab indices] converts indices back to text.

    {[
      decode vocab [ 0; 1; 0 ] = "hello world hello"
    ]} *)

val decode_batch : vocab -> (int32, Bigarray.int32_elt) Nx.t -> string list
(** [decode_batch vocab tensor] decodes tensor to texts. *)

(** {2 Vocabulary} *)

val vocab : ?max_size:int -> ?min_freq:int -> string list -> vocab
(** [vocab ?max_size ?min_freq texts] builds vocabulary from texts.

    - [max_size]: Maximum vocabulary size
    - [min_freq]: Minimum token frequency

    Automatically includes special tokens: <pad>, <unk>, <bos>, <eos> *)

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
  (** Tokenizer with tag type for configuration *)

  val words : [ `Words ] t
  (** Whitespace and punctuation tokenizer *)

  val chars : [ `Chars ] t
  (** Unicode character tokenizer *)

  val regex : string -> [ `Regex ] t
  (** Regex-based tokenizer *)

  val bpe : vocab:string -> merges:string -> [ `BPE ] t
  (** Byte-Pair Encoding tokenizer *)

  val wordpiece : vocab:string -> unk_token:string -> [ `WordPiece ] t
  (** WordPiece tokenizer *)

  val run : _ t -> string -> string list
  (** Apply tokenizer to text *)

  val run_with_offsets : _ t -> string -> (string * int * int) list
  (** Tokenize with character offsets *)

  val with_normalizer : (string -> string) -> 'a t -> 'a t
  (** Add text normalizer to tokenizer *)

  val with_pre_tokenizer : (string -> string list) -> 'a t -> 'a t
  (** Add pre-tokenization step *)
end

module Vocab : sig
  type t = vocab

  val create : unit -> t
  (** Create empty vocabulary *)

  val add : t -> string -> unit
  (** Add token to vocabulary *)

  val add_batch : t -> string list -> unit
  (** Add multiple tokens *)

  val get_index : t -> string -> int option
  (** Get token index *)

  val get_token : t -> int -> string option
  (** Get token by index *)

  val from_tokens : ?max_size:int -> ?min_freq:int -> string list -> t
  (** Build from token list *)

  val size : t -> int
  (** Vocabulary size *)

  (** Special token indices *)

  val pad_idx : t -> int
  val unk_idx : t -> int
  val bos_idx : t -> int
  val eos_idx : t -> int
end

(** {2 Unicode Processing} *)

module Bpe = Bpe
module Wordpiece = Wordpiece
module Ngram = Ngram

module Unicode : sig
  (** Unicode text processing utilities *)

  type normalization =
    | NFC  (** Canonical Decomposition, followed by Canonical Composition *)
    | NFD  (** Canonical Decomposition *)
    | NFKC
        (** Compatibility Decomposition, followed by Canonical Composition *)
    | NFKD  (** Compatibility Decomposition *)

  type char_category =
    | Letter
    | Number
    | Punctuation
    | Symbol
    | Whitespace
    | Control
    | Other

  (** {2 Character Classification} *)

  val categorize_char : Uchar.t -> char_category
  (** [categorize_char u] returns Unicode category of character *)

  val is_whitespace : Uchar.t -> bool
  (** [is_whitespace u] checks if character is whitespace *)

  val is_punctuation : Uchar.t -> bool
  (** [is_punctuation u] checks if character is punctuation *)

  val is_word_char : Uchar.t -> bool
  (** [is_word_char u] checks if character is letter or number *)

  val is_cjk : Uchar.t -> bool
  (** [is_cjk u] checks if character is Chinese/Japanese/Korean *)

  (** {2 Text Normalization} *)

  val normalize : normalization -> string -> string
  (** [normalize form text] applies Unicode normalization.

      @raise Invalid_argument on malformed Unicode *)

  val case_fold : string -> string
  (** [case_fold text] performs Unicode case folding .

      @raise Invalid_argument on malformed Unicode *)

  val strip_accents : string -> string
  (** [strip_accents text] removes diacritical marks.

      @raise Invalid_argument on malformed Unicode *)

  val clean_text :
    ?remove_control:bool -> ?normalize_whitespace:bool -> string -> string
  (** [clean_text ?remove_control ?normalize_whitespace text] cleans text.

      - [remove_control]: Remove control characters
      - [normalize_whitespace]: Collapse whitespace

      @raise Invalid_argument on malformed Unicode *)

  (** {2 Text Processing} *)

  val split_words : string -> string list
  (** [split_words text] splits on Unicode word boundaries.

      Handles CJK text where each character is typically a word.

      @raise Invalid_argument on malformed Unicode *)

  val grapheme_count : string -> int
  (** [grapheme_count text] counts user-perceived characters.

      @raise Invalid_argument on malformed Unicode *)

  val is_valid_utf8 : string -> bool
  (** [is_valid_utf8 text] validates UTF-8 encoding *)

  val remove_emoji : string -> string
  (** [remove_emoji text] removes emoji and symbols.

      @raise Invalid_argument on malformed Unicode *)
end
