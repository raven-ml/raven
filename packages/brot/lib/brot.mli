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

    Each stage is optional and configurable. Open the module to use it.

    The pre-tokenizer cuts text into {e pretokens} (HuggingFace calls them
    words); {!Encoding.word_ids} numbers them.

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

module Encoding : sig
  (** Tokenization encodings.

      An encoding bundles token IDs for model input with alignment metadata:
      byte offsets, word indices, segment type IDs, attention masks, and
      special-token flags.

      Encodings are produced by {!Brot.encode} and post-processed with
      {!val-truncate} and {!val-pad}. All parallel arrays ({!val-ids},
      {!val-type_ids}, {!val-tokens}, {!val-word_ids}, {!val-offsets},
      {!val-special_tokens_mask}, {!val-attention_mask}) share the same length,
      equal to {!val-length}.

      {!val-tokens}, {!val-word_ids} and {!val-offsets} are derived when they
      are first asked for and kept afterwards: a caller that only reads
      {!val-ids} pays for none of them. Deriving is a mutation: do not race the
      first read of {!val-tokens}, {!val-offsets} or {!val-word_ids} from
      several domains — read them from one domain, or read them before sharing.
  *)

  type t
  (** The type for tokenization encodings. *)

  (** {1:construct Construction} *)

  val empty : t
  (** [empty] is the encoding with no tokens. *)

  val create :
    ids:int array ->
    type_ids:int array ->
    tokens:string array ->
    words:int option array ->
    offsets:(int * int) array ->
    special_tokens_mask:int array ->
    attention_mask:int array ->
    ?overflowing:t list ->
    unit ->
    t
  (** [create ~ids ~type_ids ~tokens ~words ~offsets ~special_tokens_mask
       ~attention_mask ()] is an encoding from the given arrays. [overflowing]
      defaults to [[]].

      Raises [Invalid_argument] if the seven arrays do not all have the same
      length. *)

  val concat : t list -> t
  (** [concat encs] is the encoding with the tokens of each element of [encs] in
      order. {!val-overflowing} is taken from the first element. Allocates once
      rather than creating intermediate arrays per pair. *)

  (** {1:access Accessors} *)

  val ids : t -> int array
  (** [ids enc] is the token ID array. *)

  val type_ids : t -> int array
  (** [type_ids enc] is the segment ID array. Typically [0] for the first
      sequence and [1] for the second in sentence-pair tasks. *)

  val tokens : t -> string array
  (** [tokens enc] is the string representation of each token. *)

  val word_ids : t -> int option array
  (** [word_ids enc] maps each token to the index of the pretoken it came from,
      counting from [0] in each sequence. The tokens a post-processor inserts,
      and padding, are [None]. *)

  val offsets : t -> (int * int) array
  (** [offsets enc] is the [(start, end_)] byte span of each token in the text
      that was encoded, before normalization: a token of ["café"] normalized to
      ["cafe"] reports the bytes of the accented text. Spans are ascending but
      need not tile the text, since a pre-tokenizer drops its delimiters and a
      normalizer may remove characters. The tokens a post-processor inserts, and
      padding, report [(0, 0)]. *)

  val special_tokens_mask : t -> int array
  (** [special_tokens_mask enc] is [1] for special tokens ([CLS], [SEP],
      padding) and [0] for content tokens. An added token found in the input is
      [0], special or not: only what a post-processor inserts is masked. *)

  val attention_mask : t -> int array
  (** [attention_mask enc] is [1] for real tokens and [0] for padding tokens. *)

  val overflowing : t -> t list
  (** [overflowing enc] is the list of overflow encodings produced by
      {!val-truncate} when the input exceeds [max_length]. Each element is a
      sliding window over the excess tokens. *)

  val is_empty : t -> bool
  (** [is_empty enc] is [true] iff [enc] has no tokens. *)

  val length : t -> int
  (** [length enc] is the number of tokens in [enc]. *)

  (** {1:ops Operations} *)

  val with_type_id : t -> int -> t
  (** [with_type_id enc type_id] is [enc] with every {!val-type_ids} entry set
      to [type_id]. Nothing else is read, so an encoding that has not worked out
      its {!val-tokens}, {!val-offsets} or {!val-word_ids} still has not. *)

  val truncate :
    ?stride:int -> ?direction:[ `Left | `Right ] -> t -> max_length:int -> t
  (** [truncate enc ~max_length] limits [enc] to at most [max_length] tokens.

      The tokens kept are the first [max_length] when [direction] is [`Right]
      (the default) and the last [max_length] when it is [`Left]. The excess is
      split into windows of [max_length] tokens overlapping the previous window
      by [stride] (default [0]) and stored in {!val-overflowing}, walking away
      from the kept tokens; the last window stops at the encoding's edge, so it
      may be shorter. If [length enc <= max_length], [enc] is returned
      unchanged. When [max_length] is [0], all tokens move to {!val-overflowing}
      and {!val-empty} is returned.

      Raises [Invalid_argument] if [enc] is truncated and [stride] is not less
      than [max_length]. *)

  val pad :
    t ->
    target_length:int ->
    pad_id:int ->
    pad_type_id:int ->
    pad_token:string ->
    direction:[ `Left | `Right ] ->
    t
  (** [pad enc ~target_length ~pad_id ~pad_type_id ~pad_token ~direction]
      extends [enc] to exactly [target_length] tokens.

      Padding tokens have {!val-attention_mask} [0] and
      {!val-special_tokens_mask} [1]. If [length enc >= target_length], [enc] is
      returned unchanged. Padding is applied recursively to {!val-overflowing}
      encodings. When [direction] is [`Left], {!val-offsets} are shifted
      accordingly. *)

  (** {1:fmt Formatting} *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf enc] formats [enc] as a table for inspection, one row per token:
      index, token, ID, byte offsets, word ID, type ID, attention and
      special-token masks. Reading the table forces the derived {!val-tokens},
      {!val-offsets} and {!val-word_ids}. *)
end

module Normalizer : sig
  (** Text normalization.

      Normalizers transform text before tokenization: lowercasing, accent
      removal, Unicode normalization, whitespace cleanup, and model-specific
      preprocessing. They are the first stage in the tokenization pipeline,
      applied before {!Pre_tokenizer} and vocabulary-based encoding.

      Compose normalizers with {!val-sequence}:
      {[
      let n =
        Normalizer.sequence
          [ Normalizer.nfd; Normalizer.strip_accents; Normalizer.lowercase ]
      in
      Normalizer.apply n "Caf\u{00E9}"
      (* "cafe" *)
      ]} *)

  type t
  (** The type for normalizers. *)

  (** {1:normalizers Normalizers} *)

  (** {2:unicode Unicode normalization} *)

  val nfc : t
  (** [nfc] is Unicode NFC normalization (canonical composition). *)

  val nfd : t
  (** [nfd] is Unicode NFD normalization (canonical decomposition). *)

  val nfkc : t
  (** [nfkc] is Unicode NFKC normalization (compatibility composition). *)

  val nfkd : t
  (** [nfkd] is Unicode NFKD normalization (compatibility decomposition). *)

  (** {2:text Text transforms} *)

  val lowercase : t
  (** [lowercase] is the full Unicode lowercase mapping. It is not case folding:
      ["ß"] and ["ﬁ"] lowercase to themselves. *)

  val strip_accents : t
  (** [strip_accents] removes every mark ([Mn], [Mc], [Me]). It does not
      decompose: compose it after {!val-nfd} to also remove the accents of
      precomposed characters. *)

  val strip : ?left:bool -> ?right:bool -> unit -> t
  (** [strip ?left ?right ()] is a normalizer that strips Unicode whitespace
      from text boundaries. [left] and [right] default to [true]. *)

  val replace : pattern:string -> replacement:string -> t
  (** [replace ~pattern ~replacement] is a normalizer that replaces every
      occurrence of the string [pattern] with [replacement], scanning left to
      right without overlap. An empty [pattern] occurs before every character
      and at the end of the text. Empty text is returned unchanged. *)

  val replace_regex : pattern:string -> replacement:string -> t
  (** [replace_regex ~pattern ~replacement] is a normalizer that replaces every
      match of the regular expression [pattern] with [replacement]. Matches are
      found left to right, leftmost first, without overlap; an empty match right
      after another match is skipped. Empty text is returned unchanged.

      [pattern] is written in the dialect of tokenizer files: Unicode-aware, so
      that [\s], [\d] and [\w] and their negations, [.] and negated classes
      stand for characters rather than bytes, and [\p{..}] selects a general
      category ([L], [Lu], [Nd], ...). [^] and [$] are line anchors, [\A], [\z]
      and [\Z] text anchors. Groups, alternation, greedy and lazy quantifiers
      and bracket classes are supported; case-insensitive matching, lookaround,
      backreferences, word boundaries and possessive quantifiers are not.

      Raises [Invalid_argument] if [pattern] is invalid or uses an unsupported
      construct; the message says which. *)

  val prepend : string -> t
  (** [prepend s] is a normalizer that prepends [s] to non-empty text. Empty
      text is returned unchanged. *)

  (** {2:byte_level Byte-level encoding} *)

  val byte_level : t
  (** [byte_level] is GPT-2 style byte-level encoding: each byte of the text is
      mapped to a printable Unicode character using the GPT-2 byte-to-unicode
      table. A space, for instance, becomes [U+0120]. *)

  (** {2:model Model-specific} *)

  val nmt : t
  (** [nmt] is the control character cleanup of neural machine translation
      models. It removes [U+0001-U+0008], [U+000B], [U+000E-U+001F], [U+007F],
      [U+008F] and [U+009F], and replaces with a space the tab, line feed, form
      feed and carriage return, [U+1680], [U+200B-U+200F], [U+2028], [U+2029],
      [U+2581], [U+FEFF] and [U+FFFD]. *)

  val bert :
    ?clean_text:bool ->
    ?handle_chinese_chars:bool ->
    ?strip_accents:bool option ->
    ?lowercase:bool ->
    unit ->
    t
  (** [bert ()] is a BERT normalizer.

      - [clean_text]: remove control characters and normalize whitespace.
        Default: [true].
      - [handle_chinese_chars]: pad CJK ideographs with spaces. Default: [true].
      - [strip_accents]: decompose to NFD and remove the nonspacing marks.
        Unlike {!val-strip_accents}, spacing and enclosing marks are kept, so
        that the vowel signs of abugidas survive. When [None], accents are
        stripped iff [lowercase] is [true]. Default: [None].
      - [lowercase]: apply the Unicode lowercase mapping. Default: [true]. *)

  (** {2:composition Composition} *)

  val sequence : t list -> t
  (** [sequence ns] is the composition of normalizers [ns], applied left to
      right. *)

  (** {1:applying Applying} *)

  val apply : t -> string -> string
  (** [apply n s] is [s] normalized by [n]. *)

  (** {1:alignment Alignment}

      Normalizing moves text around: {!val-nfd} splits a character in two,
      {!val-strip_accents} drops one, {!val-prepend} adds one. An alignment
      records where the result came from, so that token offsets can be reported
      on the text the user passed in rather than on the normalized text. *)

  type alignment
  (** The type for maps from the bytes of a normalized text back to the bytes of
      the text it was normalized from. *)

  val identity : string -> alignment
  (** [identity s] is the alignment of [s] on itself, for text that was not
      normalized. *)

  val apply_aligned : t -> string -> string * alignment
  (** [apply_aligned n s] is [apply n s] with its alignment on [s]. *)

  val original_span : alignment -> start:int -> stop:int -> int * int
  (** [original_span a ~start ~stop] is the span of the original text that the
      normalized bytes \[[start];[stop]) come from: from the first byte the byte
      at [start] comes from to the last byte the byte at [stop - 1] comes from.

      Every byte of a character has the span of the whole character, so a span
      cutting a character short still reports it whole. A character the
      normalizer inserted has the span of the character it was inserted next to,
      and one it removed has no span at all, so the result can cover original
      bytes that no span reports. An empty [start = stop] gives an empty span.

      Raises [Invalid_argument] unless [0 <= start <= stop <= n], where [n] is
      the length of the normalized text. *)

  (** {1:formatting Formatting} *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf n] formats [n] for inspection. *)

  (** {1:serialization Serialization} *)

  val to_json : t -> Jsont.json
  (** [to_json n] is [n] serialized to HuggingFace-compatible JSON. *)

  val of_json : Jsont.json -> (t, string) result
  (** [of_json json] is a normalizer deserialized from HuggingFace JSON. Errors
      if [json] is not an object, has a missing or unknown ["type"] field, or
      has invalid parameters. *)
end

module Pre_tokenizer : sig
  (** Pre-tokenization.

      Pre-tokenizers split raw text into pretokens before vocabulary-based
      tokenization (BPE, WordPiece, etc.) is applied. Each pretoken carries byte
      offsets into the original text. *)

  type t
  (** The type for pre-tokenizers. *)

  (** {1:constructors Constructors} *)

  val whitespace : t
  (** [whitespace] splits on whitespace using pattern [\w+|[^\w\s]+].

      Groups word characters (letters, digits, underscore) together and groups
      non-word, non-space characters together. Whitespace is used as delimiter
      but not included in output. *)

  val whitespace_split : t
  (** [whitespace_split] splits on any whitespace characters.

      Removes whitespace from output. Simplest and fastest pre-tokenizer. *)

  val bert : t
  (** [bert] applies BERT-style pre-tokenization.

      Splits on whitespace, isolates punctuation, and separates CJK characters
      individually. *)

  val byte_level :
    ?add_prefix_space:bool -> ?use_regex:bool -> ?trim_offsets:bool -> unit -> t
  (** [byte_level ()] is a byte-level pre-tokenizer. Used by GPT-2, GPT-3,
      RoBERTa.

      Converts text to byte representation and applies GPT-2's regex pattern for
      splitting.

      - [add_prefix_space]: add space at beginning if text does not start with
        whitespace. Default: [true].
      - [use_regex]: use GPT-2's regex pattern for splitting. Default: [true].
      - [trim_offsets]: adjust offsets for byte-level encoding. Default: [true].
  *)

  type behavior =
    [ `Isolated  (** Keep each delimiter as a pretoken of its own *)
    | `Removed  (** Drop the delimiters *)
    | `Merged_with_previous  (** Keep a delimiter with what precedes it *)
    | `Merged_with_next  (** Keep a delimiter with what follows it *)
    | `Contiguous
      (** Keep neighbours that are both delimiters, or both not, as one pretoken
      *) ]
  (** Delimiter handling behavior for splitting operations.

      [`Merged_with_previous] and [`Merged_with_next] merge a delimiter into the
      text beside it only: a delimiter that follows another one is a pretoken of
      its own. *)

  val punctuation : ?behavior:behavior -> unit -> t
  (** [punctuation ()] separates punctuation characters from the text around
      them, whitespace included. Every punctuation character is a delimiter.

      [behavior] defaults to [`Isolated]. *)

  val split : pattern:string -> ?behavior:behavior -> ?invert:bool -> unit -> t
  (** [split ~pattern ()] splits on a literal string [pattern]. HuggingFace's
      regular expression patterns have no equivalent here.

      [behavior] defaults to [`Removed]. When [invert] is [true] the delimiters
      are the runs of text between the occurrences of [pattern], and those
      occurrences are what they separate; defaults to [false].

      An empty [pattern] matches at every position, so the pretokens are the
      characters — and none of them when [invert] is [true] and [behavior] is
      [`Removed]. *)

  val char_delimiter : string -> t
  (** [char_delimiter c] splits on the character [c], removing it from the
      output. Equivalent to [split ~pattern:c ~behavior:`Removed ()].

      Raises [Invalid_argument] if [c] is not exactly one character. *)

  val digits : ?individual_digits:bool -> unit -> t
  (** [digits ()] splits on digit boundaries.

      When [individual_digits] is [true], each digit is a separate pretoken;
      when [false] (default), consecutive digits are grouped. *)

  type prepend_scheme =
    [ `First  (** Prepend to the pretoken that opens the document only *)
    | `Never  (** Never prepend *)
    | `Always  (** Prepend to every pretoken not starting with a space *) ]
  (** Controls when metaspace prepends the replacement character. In a
      {!sequence} the pretokens are those of the member before it, so [`First]
      marks the first of them and [`Always] each — and the encode pipeline
      counts a pretoken as opening the document only when nothing, an added
      token included, comes before it. *)

  val metaspace :
    ?replacement:string ->
    ?prepend_scheme:prepend_scheme ->
    ?split:bool ->
    unit ->
    t
  (** [metaspace ()] replaces spaces with a visible marker. Used by
      SentencePiece models.

      - [replacement]: the marker spaces are replaced with. Default: ["▁"]
        (U+2581). It must be exactly one character.
      - [prepend_scheme]: when to prepend the marker, which happens only if the
        marked text does not start with one already. Default: [`Always].
      - [split]: whether to split before each marker. Default: [true].

      Pretokens are those of the marked text; their offsets are the bytes of the
      text they were made from. A marker that replaced a space stands at that
      space, and a prepended one at the character it opens.

      Raises [Invalid_argument] if [replacement] is not exactly one character.
  *)

  val unicode_scripts : t
  (** [unicode_scripts] splits on Unicode script boundaries.

      A pretoken opens where the writing system changes (e.g. Latin to Cyrillic,
      Latin to Han) and runs to the next change. Hiragana and Katakana count as
      Han, as does the prolonged sound mark ["ー"] (U+30FC).

      Spaces and characters of no known script join the pretoken they follow, so
      a leading run of them belongs to no pretoken: unlike the other
      pre-tokenizers, the pretokens need not cover the input. *)

  val fixed_length : int -> t
  (** [fixed_length n] cuts the text into pretokens of [n] characters.

      The last pretoken may be shorter than [n]. *)

  val sequence : t list -> t
  (** [sequence ts] chains multiple pre-tokenizers left-to-right.

      Each pre-tokenizer processes the pretokens from the previous one. Offsets
      are composed correctly through the chain. *)

  (** {1:ops Operations} *)

  val pre_tokenize : t -> string -> (string * (int * int)) list
  (** [pre_tokenize t text] splits [text] into pretokens with byte offsets.

      Returns a list of [(pretoken, (start, end_))] where [start] and [end_] are
      byte positions in [text], ascending and within it. A pretoken is the bytes
      of its span unless [t] rewrote or encoded them, in which case the span is
      where the bytes it was made from lie.

      Two spans can cover the same bytes. A span is widened to whole characters,
      so pretokens of a [text] that is not valid UTF-8 can share one; and the
      pretokens a member of a {!sequence} cuts from one that was rewritten more
      than once, cut by a fixed-length member, or encoded all report that
      pretoken's span, nothing in it being placeable more finely than the whole
      of it. *)

  (** {1:fmt Formatting} *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf t] formats [t] for inspection. *)

  (** {1:byte_level_decode Byte-level decoding} *)

  val byte_level_decode : string -> string
  (** [byte_level_decode s] is the bytes the byte-level alphabet spells [s]
      from: one byte per character. A single character outside the alphabet, an
      invalidly encoded one included, leaves [s] as it is — the fallback is over
      the whole string, not the character.

      The result needs not be valid UTF-8: it is bytes, and turning them into
      text is the caller's step. *)

  (** {1:serialization Serialization} *)

  val to_json : t -> Jsont.json
  (** [to_json t] serializes [t] to HuggingFace JSON format. *)

  val of_json : Jsont.json -> (t, string) result
  (** [of_json json] is a pre-tokenizer from HuggingFace JSON format. Errors if
      [json] is not an object, has a missing or unknown ["type"] field, has
      invalid parameters, or is a ["Split"] whose pattern is a regular
      expression ([{"Regex": ...}]) rather than a literal ([{"String": ...}]).
  *)
end

module Post_processor : sig
  (** Post-processing tokenization output with special tokens.

      Post-processors add special tokens and type IDs to tokenized sequences
      after core tokenization. They handle model-specific requirements like
      [[CLS]] and [[SEP]] for BERT, sentence pair formatting, and byte-level
      offset adjustments. *)

  type t
  (** The type for post-processors. *)

  type token = string * int
  (** A special token as [(text, id)]. *)

  (** {1:constructors Constructors} *)

  val bert : sep:token -> cls:token -> t
  (** [bert ~sep ~cls] is a BERT-style post-processor.

      Single: [[CLS] A [SEP]]. Pair: [[CLS] A [SEP] B [SEP]]. Type IDs: [0] for
      the first sequence, [1] for the second. *)

  val roberta :
    sep:token ->
    cls:token ->
    ?trim_offsets:bool ->
    ?add_prefix_space:bool ->
    unit ->
    t
  (** [roberta ~sep ~cls ()] is a RoBERTa-style post-processor.

      Single: [<s> A </s>]. Pair: [<s> A </s> </s> B </s>]. All type IDs are
      [0].

      [trim_offsets] removes leading and trailing whitespace from the offsets of
      byte-level tokens; it defaults to [true]. [add_prefix_space] tells the
      trimming that a single leading space on a token that starts at offset [0]
      was added by the pre-tokenizer and must be kept; it defaults to [true]. *)

  val byte_level : ?add_prefix_space:bool -> ?trim_offsets:bool -> unit -> t
  (** [byte_level ()] is a byte-level post-processor that adjusts character
      offsets for byte-level encoding.

      [trim_offsets] removes leading and trailing whitespace from offsets.
      Defaults to [true]. [add_prefix_space] tells the trimming that a single
      leading space on a token that starts at offset [0] was added by the
      pre-tokenizer and must be kept; it defaults to [true]. *)

  val template :
    single:string -> ?pair:string -> ?special_tokens:token list -> unit -> t
  (** [template ~single ()] is a template-based post-processor.

      Templates use [$A] and [$B] as sequence placeholders and literal special
      token names (e.g. [[CLS]]). Type IDs can be specified with a colon suffix:
      [$A:0], [[SEP]:1].

      [pair] must reference both [$A] and [$B]; it defaults to ["$A:0 $B:1"].
      [special_tokens] defaults to [[]]. *)

  val sequence : t list -> t
  (** [sequence processors] chains [processors] left-to-right. *)

  (** {1:processing Processing} *)

  val process :
    t -> ?pair:Encoding.t -> Encoding.t -> add_special_tokens:bool -> Encoding.t
  (** [process t enc ~add_special_tokens] adds special tokens and sets type IDs
      on [enc].

      When [~pair] is provided, both sequences are merged into a single
      encoding. [pair] enters as the second segment, with type ID [1], which [t]
      may override. Offsets are not shifted: each sequence keeps the offsets
      into its own text.

      When [~add_special_tokens] is [false], no special token is inserted, but
      everything else still happens: the two sequences are still merged, type
      IDs are still assigned, and byte-level offsets are still trimmed. *)

  val added_tokens : t -> is_pair:bool -> int
  (** [added_tokens t ~is_pair] is the number of special tokens [t] adds. Useful
      for calculating the truncation budget. *)

  (** {1:fmt Formatting} *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a post-processor for inspection. *)

  (** {1:serialization Serialization} *)

  val of_json : Jsont.json -> (t, string) result
  (** [of_json json] is a post-processor from HuggingFace [tokenizer.json]
      format. Errors if [json] is not an object, has a missing or unknown
      ["type"] field, or has invalid parameters. *)

  val to_json : t -> Jsont.json
  (** [to_json t] is [t] serialized to HuggingFace [tokenizer.json] format. *)
end

module Decoder : sig
  (** Decoding tokens back to text.

      Decoders convert token strings back into natural text by reversing
      encoding-specific transformations (prefix/suffix removal, byte-level
      decoding, whitespace normalization, etc.).

      Decoders operate on token {e strings}, not IDs. Convert IDs to strings via
      vocabulary first, then apply {!val-decode}.

      A decoder rewrites a token list into another token list, and {!val-decode}
      is the concatenation of the result. Most decoders rewrite each token on
      its own ({!val-bpe}, {!val-wordpiece}, {!val-metaspace}, {!val-replace},
      {!val-strip}); {!val-byte_fallback} and {!val-ctc} also join or drop
      tokens; {!val-byte_level} and {!val-fuse} collapse the whole list into one
      token. The distinction matters when composing decoders with
      {!val-sequence}: a collapsing decoder hides token boundaries from the
      decoders that follow it. *)

  type t
  (** The type for decoders. *)

  (** {1:constructors Constructors} *)

  val bpe : ?suffix:string -> unit -> t
  (** [bpe ~suffix ()] is a decoder for BPE-encoded tokens. Every occurrence of
      [suffix], which marks the end of a word, becomes the space that follows
      it, except in the last token where it is dropped. [suffix] defaults to
      ["</w>"]. *)

  val byte_level : t
  (** [byte_level] is a collapsing decoder that reads GPT-2 style
      byte-to-Unicode tokens back as the text their bytes spell.

      A token is mapped character by character, and one character outside the
      byte-level alphabet leaves the whole token to stand for its own bytes. The
      bytes of every token are then read as one text, so a character spelled
      across two tokens comes back whole, and every maximal ill-formed byte
      sequence becomes one U+FFFD — four stray bytes cost four, while a
      four-byte character cut short costs one. *)

  val byte_fallback : t
  (** [byte_fallback] is a decoder for byte fallback tokens. Each run of hex
      byte tokens (e.g. ["<0x41>"]) becomes the text those bytes spell; a run
      that is not valid UTF-8 becomes one U+FFFD per byte. Other tokens pass
      through unchanged. *)

  val wordpiece : ?prefix:string -> ?cleanup:bool -> unit -> t
  (** [wordpiece ~prefix ~cleanup ()] is a decoder for WordPiece tokens. Strips
      continuation [prefix] (default ["##"]) from non-initial subwords and gives
      the others a leading space. When [cleanup] is [true] (default), applies
      the detokenization cleanup to every token, once its joining space is
      prepended: the space before [.], [?], [!], [,] and the English
      contractions is taken back, and [" do not"] is rewritten to [" don't"]. So
      ["hello"; ","; "world"] decodes to ["hello, world"], while
      ["3"; "."; "14"] decodes to ["3. 14"] because the space after the full
      stop was never the decoder's to remove. *)

  val metaspace :
    ?replacement:string ->
    ?prepend_scheme:Pre_tokenizer.prepend_scheme ->
    unit ->
    t
  (** [metaspace ~replacement ~prepend_scheme ()] converts metaspace markers
      back to spaces. [replacement] defaults to ["\u{2581}"]. Unless
      [prepend_scheme] is [`Never], the marker was prepended to the text rather
      than standing for a space, so every occurrence of it in the {e first}
      token is dropped instead of becoming a space — on [`First] as on
      [`Always], the two prepending to the same first token. [prepend_scheme]
      defaults to [`Always]. *)

  val ctc :
    ?pad_token:string ->
    ?word_delimiter_token:string ->
    ?cleanup:bool ->
    unit ->
    t
  (** [ctc ~pad_token ~word_delimiter_token ~cleanup ()] is a decoder for
      {{:https://distill.pub/2017/ctc/}CTC (Connectionist Temporal
       Classification)} output. Deduplicates consecutive tokens, then cuts every
      occurrence of [pad_token] (default ["<pad>"]) out of each token, wherever
      in it they fall, and drops the tokens left empty. When [cleanup] is [true]
      (default), applies the same detokenization cleanup as {!val-wordpiece} to
      every token and then replaces [word_delimiter_token] (default ["|"]) with
      spaces. *)

  val sequence : t list -> t
  (** [sequence decoders] chains [decoders] left-to-right. Each decoder's output
      token list feeds into the next. *)

  val replace : pattern:string -> by:string -> unit -> t
  (** [replace ~pattern ~by ()] replaces every literal occurrence of [pattern]
      with [by] in each token. *)

  val strip : ?content:string -> ?start:int -> ?stop:int -> unit -> t
  (** [strip ~content ~start ~stop ()] removes up to [start] leading and [stop]
      trailing occurrences of [content] from each token. [content] defaults to
      [" "], [start] and [stop] to [0]. A token left with nothing between the
      two cuts becomes empty.

      HuggingFace reads [content] as a single character, so {!val-to_json} on a
      decoder whose [content] is longer, or empty, writes a decoder it rejects.
  *)

  val fuse : t
  (** [fuse] is a collapsing decoder that concatenates all tokens into a single
      string with no delimiter. *)

  (** {1:ops Operations} *)

  val decode : t -> string list -> string
  (** [decode decoder tokens] applies [decoder] to [tokens] and returns the
      decoded text. *)

  (** {1:fmt Formatting} *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf decoder] formats [decoder] for debugging. *)

  (** {1:serialization Serialization} *)

  val to_json : t -> Jsont.json
  (** [to_json decoder] serializes [decoder] to HuggingFace JSON format. *)

  val of_json : Jsont.json -> (t, string) result
  (** [of_json json] is a decoder from HuggingFace JSON format. Errors if [json]
      is not an object, has a missing or unknown ["type"] field, or has invalid
      parameters. *)
end

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

type truncation = { max_length : int; stride : int; direction : direction }
(** The type for truncation configurations. Sequences exceeding [max_length]
    tokens are trimmed from the given [direction]; [stride] is the overlap
    between successive overflow windows. *)

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

val truncation : ?stride:int -> ?direction:direction -> int -> truncation
(** [truncation max_length] is a truncation configuration limiting sequences to
    [max_length] tokens. [direction] defaults to [`Right]: the tokens kept are
    the first [max_length], and [`Left] keeps the last.

    Truncation runs before the post-processor, on what is left of [max_length]
    once the special tokens it will add are counted, so a special token never
    pushes content out. A [max_length] at or below that count leaves no room for
    content: the special tokens are still added and the result is longer than
    [max_length].

    [stride] is the overlap between successive overflow windows: each window in
    {!Encoding.overflowing} opens [stride] tokens before the end of the one
    before it, which is how the sliding-window question-answering workflow keeps
    context across windows. Defaults to [0]. It must be less than the tokens
    left for content — on a pair, less than either sequence's share — or
    {!encode} raises [Invalid_argument]. *)

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
    - [cache_capacity]: number of entries in the pretoken cache, rounded up to a
      power of two. An entry costs about 32 bytes, and each domain encoding with
      the tokenizer keeps a table of its own. Default: [262144] (8 MB). [0]
      disables caching.
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
  ?unk_id:int ->
  ?byte_fallback:bool ->
  unit ->
  t
(** [unigram ()] is a Unigram tokenizer. Used by AlBERT, T5, mBART.

    A pretoken is cut into the vocabulary pieces whose scores add up to the
    most, rather than into the longest match at each byte. A character no piece
    covers costs an unknown token, scored ten below the rarest piece of the
    vocabulary.

    - [vocab]: initial vocabulary as [(token, score)] pairs where scores are
      log-probabilities (negative numbers). An entry is identified by its
      position. Default: [[]].
    - [unk_id]: the entry standing for a run of characters the vocabulary does
      not hold, one token for the whole run, as a [tokenizer.json] names it.
      Without one, encoding raises [Failure] as soon as the best path into some
      character would spend an unknown token on it. Default: the position of
      [unk_token] in [vocab], if it is there.
    - [byte_fallback]: spell such a run out as byte tokens (["<0xFF>"]) when the
      vocabulary holds one for every byte of it. Default: [false].
    - [unk_token]: the role marker, and the model's unknown entry when [vocab]
      holds it and [unk_id] is not given. Default: none.

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
(** [chars ()] is a byte-level tokenizer: each byte of the input is one token
    whose ID is the byte's value. A multi-byte character is as many tokens as it
    has bytes. No vocabulary is required.

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
    neither; the {!Encoding} module doc says how that first read behaves across
    domains.

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
  ?domains:int ->
  string list ->
  Encoding.t list
(** [encode_batch t texts] is the encoding of each text in [texts].

    - [domains]: how many domains to encode on. Default:
      [Domain.recommended_domain_count ()]. Work is handed out by bytes, in
      chunks of 64 KB to 256 KB, and a batch holding under 64 KB is encoded on
      the calling domain whatever this says. Raises [Invalid_argument] if under
      one.
    - Other optional parameters are as in {!encode}. For sentence-pair tasks,
      use {!encode_batch_pairs}. *)

val encode_batch_pairs :
  t ->
  ?add_special_tokens:bool ->
  ?padding:padding ->
  ?truncation:truncation ->
  ?domains:int ->
  (string * string) list ->
  Encoding.t list
(** [encode_batch_pairs t pairs] encodes a batch of sentence pairs. Each element
    is [(primary, secondary)].

    There is no ids fast path for pairs: a pair-throughput workload goes through
    this function and reads {!Encoding.ids} off each encoding.

    Optional parameters are as in {!encode_batch}. *)

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

type ids = (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for flat token id buffers: the rows of a batch, one after the
    other. Hand one to [Nx] with
    [Nx.of_bigarray (Bigarray.genarray_of_array1 ids)]. *)

val encode_batch_ids :
  t ->
  ?add_special_tokens:bool ->
  ?padding:padding ->
  ?truncation:truncation ->
  ?domains:int ->
  string list ->
  ids * int array
(** [encode_batch_ids t texts] is [(ids, lengths)]: the token ids of every text
    end to end in one buffer, and the number of ids each of them holds. Row [i]
    starts at the sum of [lengths.(0)] up to [lengths.(i - 1)]; under a padding
    that makes the rows equal, a buffer with at least one row reshapes to
    [Array.length lengths] by [lengths.(0)].

    This is the throughput path. It builds no {!Encoding.t} and allocates
    nothing per token: a row is what {!encode_ids} gives for that text, with
    [`Batch_longest] resolved over the whole batch as {!encode_batch} resolves
    it. Overflowing windows produced by [truncation] are dropped, since an [ids]
    buffer has nowhere to put them; {!encode_batch} keeps them. A long text is
    spread over several domains too, in pieces cut where the pipeline cannot
    tell the difference — a pipeline with a normalizer, or one that does not
    split on spaces, encodes each text on one domain.

    Identifiers are narrowed to 32 bits. No vocabulary reaches that far: an
    identifier is an index into the token table, which holds one string per
    token.

    Optional parameters are as in {!encode_batch}. *)

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
  ?max_token_length:int ->
  data ->
  t
(** [train_bpe data] trains a BPE tokenizer from [data].

    The words counted are the pretokens the trained tokenizer will itself
    produce: every text is normalized by [normalizer], then cut by [pre], and
    each pretoken counts as one word — for a byte-level [pre] the words are in
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

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token], [unk_token]) are as in {!bpe}. *)

val train_word_level :
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
  data ->
  t
(** [train_word_level data] trains a word-level tokenizer from [data].

    The vocabulary is the most frequent of the words counted as in {!train_bpe},
    so it holds whole pretokens and nothing is split further.

    - [init]: a tokenizer whose added and special tokens carry over into the
      trained one. Its model does not. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. Default:
      [30000].
    - [min_frequency]: minimum frequency for a word to be included. Default:
      [0].

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
  data ->
  t
(** [train_unigram data] trains a Unigram tokenizer from [data].

    {b Warning.} The EM training of the unigram model is not implemented: the
    vocabulary is the most frequent of the words counted as in {!train_bpe},
    each scored by how often it occurs, and no subword piece is ever proposed.

    - [init]: a tokenizer whose added and special tokens carry over into the
      trained one. Its model does not. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. Default:
      [8000].
    - [unk_token]: also names the model's unknown entry when [added_tokens] or
      [init] carries it, since a trained vocabulary of whole words covers no
      character on its own. Without one, the trained tokenizer raises [Failure]
      on any word it was not trained on. Default: none.

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token], [unk_token]) are as in {!bpe}. *)

(** {1:model_files Model files} *)

val export_tiktoken : t -> merges_path:string -> vocab_path:string -> unit
(** [export_tiktoken t ~merges_path ~vocab_path] exports [t]'s BPE merges and
    vocabulary in tiktoken-compatible format.

    {b Warning.} Only BPE tokenizers are supported. Raises [Failure] for other
    model types. *)

val save_model_files : ?prefix:string -> t -> folder:string -> string list
(** [save_model_files t ~folder] saves [t]'s underlying model files (vocabulary
    and merges) to [folder] and returns the list of created file paths.

    [prefix] defaults to [""]. *)

(** {1:huggingface HuggingFace compatibility} *)

val from_file : string -> (t, string) result
(** [from_file path] is a tokenizer loaded from a HuggingFace [tokenizer.json]
    file. Errors if the file cannot be read or has invalid format. *)

val of_json : Jsont.json -> (t, string) result
(** [of_json json] is a tokenizer deserialized from HuggingFace JSON format.
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
