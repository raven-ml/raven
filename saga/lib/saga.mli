(** Saga - Fast tokenization and text processing for ML in OCaml *)

(** {1 Core Types} *)

type tokenizer
(** Abstract tokenizer type that handles encoding/decoding *)

type vocab
(** Vocabulary mapping between tokens and indices *)

(** {1 Quick Start - Simple API} *)

(** {2 Tokenizers} *)

val tokenizer :
  [ `BPE of string * string  (** vocab_file, merges_file *)
  | `WordPiece of string * string  (** vocab_file, unk_token *)
  | `Words  (** Whitespace tokenization *)
  | `Chars  (** Character-level tokenization *)
  | `Regex of string  (** Custom regex pattern *) ] ->
  tokenizer
(** Create a tokenizer. Examples:
    {[
      let tok = tokenizer (`BPE ("vocab.json", "merges.txt"))
      let tok = tokenizer `Words
    ]} *)

val encode : tokenizer -> string -> int array
(** [encode tokenizer text] converts text to token IDs *)

val decode : tokenizer -> int array -> string
(** [decode tokenizer ids] converts token IDs back to text *)

val encode_batch :
  tokenizer ->
  ?max_length:int ->
  ?padding:bool ->
  ?truncation:bool ->
  string list ->
  (int32, Bigarray.int32_elt) Nx.t
(** [encode_batch tokenizer texts] encodes multiple texts to a tensor. Returns
    shape [batch_size; seq_length] *)

(** {2 Text Processing} *)

val normalize :
  ?lowercase:bool ->
  ?strip_accents:bool ->
  ?clean_whitespace:bool ->
  string ->
  string
(** Simple text normalization. All options default to false. *)

val tokenize : string -> string list
(** [tokenize text] splits text into words (whitespace + punctuation) *)

val split_sentences : string -> string list
(** [split_sentences text] splits text into sentences *)

(** {2 Vocabulary} *)

val build_vocab :
  ?max_size:int -> ?min_freq:int -> tokenizer -> string list -> vocab
(** [build_vocab tokenizer texts] builds vocabulary from texts *)

val vocab_size : vocab -> int
(** Returns vocabulary size *)

val save_vocab : vocab -> string -> unit
(** Save vocabulary to file *)

val load_vocab : string -> vocab
(** Load vocabulary from file *)

(** {1 Language Models - High-level API} *)

module LM : sig
  type t
  (** Abstract language model type *)

  val train_ngram : n:int -> ?smoothing:float -> tokenizer -> string list -> t
  (** [train_ngram ~n tokenizer texts] trains an n-gram model. Example:
      [train_ngram ~n:2 tokenizer texts] for bigram *)

  val generate :
    t ->
    ?max_tokens:int ->
    ?temperature:float ->
    ?top_k:int ->
    ?prompt:string ->
    tokenizer ->
    string
  (** [generate model tokenizer] generates text. Returns decoded string. *)

  val perplexity : t -> tokenizer -> string -> float
  (** [perplexity model tokenizer text] computes perplexity *)

  val save : t -> string -> unit
  (** Save model to file *)

  val load : string -> t
  (** Load model from file *)
end

(** {1 Advanced API - Direct Module Access} *)

(** {2 Tokenizer Implementations} *)

module Tokenizers = Saga_tokenizers
(** Low-level access to BPE, WordPiece, Unicode utilities *)

(** {2 Model Implementations} *)

module Models = Saga_models
(** Low-level access to N-gram models *)

(** {2 Advanced Tokenizer Configuration} *)

module Tokenizer : sig
  type 'a t
  (** Typed tokenizer for advanced configuration *)

  val create :
    [ `BPE of Saga_tokenizers.Bpe.t
    | `WordPiece of Saga_tokenizers.Wordpiece.t
    | `Words
    | `Chars
    | `Regex of string ] ->
    tokenizer
  (** Create from low-level tokenizer *)

  val with_normalizer : (string -> string) -> tokenizer -> tokenizer
  (** Add normalization step *)

  val with_pre_tokenizer : (string -> string list) -> tokenizer -> tokenizer
  (** Add pre-tokenization step *)

  val encode_with_offsets : tokenizer -> string -> (int * int * int) array
  (** Returns (token_id, start_offset, end_offset) for each token *)
end

(** {2 Vocabulary Management} *)

module Vocab : sig
  type t = vocab

  val create : unit -> t
  (** Create empty vocabulary *)

  val add : t -> string -> unit
  (** Add single token *)

  val add_tokens : t -> string list -> unit
  (** Add multiple tokens *)

  val token_to_id : t -> string -> int option
  (** Get token ID *)

  val id_to_token : t -> int -> string option
  (** Get token string *)

  val size : t -> int

  val pad_token : string
  (** Special tokens *)

  val unk_token : string
  val bos_token : string
  val eos_token : string
  val pad_id : t -> int
  val unk_id : t -> int
  val bos_id : t -> int
  val eos_id : t -> int
end

(** {1 Examples}

    {2 Quick tokenization}
    {[
      open Saga

      (* Simple word tokenization *)
      let tok = tokenizer `Words
      let ids = encode tok "Hello world!"
      let text = decode tok ids

      (* BPE tokenization *)
      let tok = tokenizer (`BPE ("vocab.json", "merges.txt"))
      let batch = encode_batch tok [ "Hello"; "World" ] ~padding:true
    ]}

    {2 Training a language model}
    {[
      (* Train a bigram model *)
      let texts = [ "The cat sat"; "The dog ran"; "The cat ran" ]
      let tok = tokenizer `Words
      let model = LM.train_ngram ~n:2 tok texts

      (* Generate text *)
      let generated =
        LM.generate model ~max_tokens:50 ~temperature:0.8 tok print_endline
          generated
    ]}

    {2 Custom tokenizer}
    {[
      (* Tokenizer with normalization *)
      let tok =
        tokenizer `Words
        |> Tokenizer.with_normalizer (normalize ~lowercase:true)

      (* Regex tokenizer for code *)
      let code_tok = tokenizer (`Regex {|[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|.|})]
    ]} *)
