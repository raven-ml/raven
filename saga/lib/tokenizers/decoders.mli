(** Decoding module for converting token IDs back to text. *)

type t
(** Main decoder type *)

(** {1 Constructors} *)

val bpe : ?suffix:string -> unit -> t
(** Create a BPE decoder.
    @param suffix Suffix to remove (default: "") *)

val byte_level : unit -> t
(** Create a byte-level decoder *)

val byte_fallback : unit -> t
(** Create a byte fallback decoder *)

val wordpiece : ?prefix:string -> ?cleanup:bool -> unit -> t
(** Create a WordPiece decoder.
    @param prefix Prefix to remove (default: "##")
    @param cleanup Whether to cleanup tokenization artifacts (default: true) *)

val metaspace : ?replacement:char -> ?add_prefix_space:bool -> unit -> t
(** Create a Metaspace decoder.
    @param replacement Character to replace spaces with (default: '▁')
    @param add_prefix_space Whether prefix space was added (default: true) *)

val ctc :
  ?pad_token:string ->
  ?word_delimiter_token:string ->
  ?cleanup:bool ->
  unit ->
  t
(** Create a CTC decoder.
    @param pad_token Padding token (default: "<pad>")
    @param word_delimiter_token Word delimiter token (default: "|")
    @param cleanup Whether to cleanup artifacts (default: true) *)

val sequence : t list -> t
(** Combine multiple decoders in sequence *)

val replace : pattern:string -> content:string -> unit -> t
(** Create a replace decoder.
    @param pattern Pattern to match
    @param content Replacement string *)

val strip : ?left:bool -> ?right:bool -> ?content:char -> unit -> t
(** Create a strip decoder.
    @param left Strip from left (default: false)
    @param right Strip from right (default: false)
    @param content Character to strip (default: ' ') *)

val fuse : unit -> t
(** Create a fuse decoder that merges tokens *)

(** {1 Operations} *)

val decode : t -> string list -> string
(** Decode a list of tokens back to text *)

(** {1 Serialization} *)

val to_json : t -> Yojson.Basic.t
val of_json : Yojson.Basic.t -> t
