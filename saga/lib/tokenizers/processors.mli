(** Post-processing module for tokenization output.

    Post-processors handle special tokens and formatting after tokenization,
    such as adding [CLS] and [SEP] tokens for BERT, or handling sentence pairs.
*)

type encoding = {
  ids : int array;
  type_ids : int array;
  tokens : string array;
  offsets : (int * int) array;
  special_tokens_mask : int array;
  attention_mask : int array;
  overflowing : encoding list;
  sequence_ranges : (int * int * int) list;
}
(** Type representing an encoding to be processed *)

type t
(** Main post-processor type *)

(** {1 Constructors} *)

val bert : sep:string * int -> cls:string * int -> unit -> t
(** Create a BERT post-processor.
    @param sep Separator token and ID
    @param cls Classification token and ID *)

val roberta :
  sep:string * int ->
  cls:string * int ->
  ?trim_offsets:bool ->
  ?add_prefix_space:bool ->
  unit ->
  t
(** Create a RoBERTa post-processor.
    @param sep Separator token and ID
    @param cls Classification token and ID
    @param trim_offsets Whether to trim offsets (default: true)
    @param add_prefix_space Whether to add prefix space (default: true) *)

val byte_level : ?trim_offsets:bool -> unit -> t
(** Create a byte-level post-processor.
    @param trim_offsets Whether to trim offsets (default: true) *)

val template :
  single:string ->
  ?pair:string ->
  ?special_tokens:(string * int) list ->
  unit ->
  t
(** Create a template post-processor.
    @param single Template for single sequences (e.g., "[CLS] $A [SEP]")
    @param pair Template for sequence pairs (e.g., "[CLS] $A [SEP] $B [SEP]")
    @param special_tokens List of special tokens with their IDs *)

val sequence : t list -> t
(** Combine multiple post-processors in sequence *)

(** {1 Operations} *)

val process : t -> encoding list -> add_special_tokens:bool -> encoding list
(** Process encodings with the post-processor.
    @param t The post-processor
    @param encodings List of encodings to process
    @param add_special_tokens Whether to add special tokens
    @return Processed encodings *)

val added_tokens : t -> is_pair:bool -> int
(** Get the number of tokens added by this post-processor.
    @param t The post-processor
    @param is_pair Whether processing a pair of sequences
    @return Number of added tokens *)

(** {1 Serialization} *)

val to_json : t -> Yojson.Basic.t
(** Convert post-processor to JSON representation *)

val of_json : Yojson.Basic.t -> t
(** Create post-processor from JSON representation *)
