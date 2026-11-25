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
      equal to {!val-length}. {!val-create} keeps the arrays it is given and the
      accessors return the encoding's own arrays, so mutating one changes the
      encoding.

      {!val-tokens}, {!val-word_ids} and {!val-offsets} are derived when they
      are first asked for and kept afterwards: a caller that only reads
      {!val-ids} pays for none of them. Deriving is a mutation, so do not race
      the first read of {!val-tokens}, {!val-offsets} or {!val-word_ids} from
      several domains. Read them from one domain, or read them before sharing.
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
      order. Each element keeps its {!val-word_ids}. {!val-overflowing} is taken
      from the first element. [concat []] is {!val-empty}. *)

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
      [0], special or not. Only the tokens a post-processor inserts, and
      padding, are masked. *)

  val attention_mask : t -> int array
  (** [attention_mask enc] is [1] for real tokens and [0] for padding tokens. *)

  val overflowing : t -> t list
  (** [overflowing enc] is the list of windows {!val-truncate} cut off [enc].
      The windows of an encoding from {!Brot.encode} went through its
      post-processor too; {!Brot.truncation} says how they are cut. *)

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
      may be shorter. Each window has no {!val-overflowing} of its own, and
      [enc]'s is dropped. If [length enc <= max_length], [enc] is returned
      unchanged. When [max_length] is [0] and [enc] has tokens, {!val-empty} is
      returned, with [enc] itself, and its {!val-overflowing}, as the one
      window.

      Raises [Invalid_argument] if [max_length] or [stride] is negative, or if
      [max_length > 0], [enc] is truncated and [stride] is not less than
      [max_length]. *)

  val pad :
    t ->
    target_length:int ->
    pad_id:int ->
    pad_type_id:int ->
    pad_token:string ->
    direction:[ `Left | `Right ] ->
    t
  (** [pad enc ~target_length ~pad_id ~pad_type_id ~pad_token ~direction] is
      [enc] with padding tokens added up to [target_length] tokens, after its
      tokens or, when [direction] is [`Left], before them. An [enc] of
      [target_length] tokens or more keeps its tokens as they are.

      Each padding token has ID [pad_id], type ID [pad_type_id], token
      [pad_token], no word, the span [(0, 0)], {!val-attention_mask} [0] and
      {!val-special_tokens_mask} [1]. Each encoding of {!val-overflowing} is
      padded the same way, whatever the length of [enc]. *)

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
      ]}

      The normalizers read the case mappings, general categories and White_Space
      property of Unicode 18.0.0. Decompositions and combining classes are those
      of the Unicode version of the uunf library brot is built with.

      A byte that opens no whole character, a lead byte without the continuation
      bytes it asks for, a continuation byte with no lead or a byte of a
      sequence RFC 3629 forbids, is kept as it is, a character of one byte, by
      every normalizer but {!val-precompiled} and {!val-byte_level}. *)

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
  (** [lowercase] is the full Unicode lowercase mapping. Capital sigma
      lowercases to [σ] wherever it stands, since the final-sigma rule depends
      on context. It is not case folding: ["ß"] and ["ﬁ"] lowercase to
      themselves. *)

  val strip_accents : t
  (** [strip_accents] removes every mark ([Mn], [Mc], [Me]). It does not
      decompose: compose it after {!val-nfd} to also remove the accents of
      precomposed characters. *)

  val strip : ?left:bool -> ?right:bool -> unit -> t
  (** [strip ?left ?right ()] is a normalizer that removes the characters with
      the White_Space property from the start of the text if [left] and from its
      end if [right]. [left] and [right] default to [true]. *)

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
      [replacement] is inserted as it is, so [$1] stands for itself.

      [pattern] is written in the dialect of tokenizer files: Unicode-aware, so
      that [\s], [\d] and [\w] and their negations, [.] and negated classes
      stand for characters rather than bytes, and [\p{..}] selects a general
      category ([L], [Lu], [Nd], ...) or a script ([Han], [Latin], ...) of
      Unicode 18.0.0. Outside a bracket class, [\w] also matches [¹], [²], [³],
      [¼], [½] and [¾]. [^] and [$] are line anchors, [\A], [\z] and [\Z] text
      anchors. A [{] that opens no interval, as in [a{] or [a{,}], stands for
      itself.

      Groups, named groups [(?<name>..)], comments [(?#..)], alternation, greedy
      and lazy quantifiers, [\G] and [\x{..}] escapes and bracket classes are
      supported, nested and intersected with [&&] as in [[\p{L}&&[^\p{Han}]]], a
      leading [^] negating the whole intersection, as are the case-insensitive
      option ([(?i)], [(?i:..)]), under which a character or a bracket class
      stands for every character with the same simple case folding, and a
      lookahead ([(?=..)], or [(?!..)] over one character) that ends the pattern
      or one of its alternatives. A lookahead anywhere else, lookbehind,
      backreferences, atomic groups, word boundaries, possessive quantifiers, a
      [^] that can end a match, group options other than [i], POSIX bracket
      expressions, byte escapes above [\x7F] and repeat counts above 100000 are
      not.

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

  val precompiled : string -> t
  (** [precompiled charsmap] is the normalizer of a SentencePiece model, as T5,
      ALBERT and XLNet ship it. [charsmap] is the model's
      [precompiled_charsmap], as bytes: rules, each replacing a source string by
      another, compiled into a darts-clone double-array trie. Tokenizer files
      store it in base64, which {!of_json} decodes.

      The text is read one extended grapheme cluster at a time, as UAX #29
      defines them at Unicode 18.0.0. A cluster of fewer than 6 bytes that
      starts with a rule's source becomes the replacement of the shortest such
      source, and the rest of the cluster is dropped. In any other cluster each
      character that is a rule's source becomes its replacement and the others
      are kept. A byte that does not start a valid UTF-8 character becomes
      [U+FFFD]. HuggingFace reads clusters at Unicode 17.0.0, so a character new
      in 18.0.0, or a virama and a consonant after no consonant, which 18.0.0
      joins, can be cut otherwise.

      In the alignment, a replacement stands for its source character by
      character, and the characters of a longer one past the source's length
      take the span of the source's last character. The rest of the source of a
      shorter one, like a removed character, belongs to no span. Characters
      removed before anything was written shift the spans of all that follows
      back by as many characters, and a removal right after a longer replacement
      gives its last character the span of the first character removed.

      Raises [Invalid_argument] if [charsmap] is not a well-formed charsmap. *)

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
      - [lowercase]: apply the Unicode lowercase mapping. Default: [true].

      The control characters are those of categories [Cc], [Cf] and [Co] other
      than tab, line feed and carriage return; [clean_text] removes them and
      U+FFFD, and replaces each White_Space character with a space. The CJK
      ideographs are those of U+3400-4DBF, 4E00-9FFF, F900-FAFF, 20000-2A6DF,
      2A700-2B81F, 2B920-2CEAF and 2F800-2FA1F. The steps run in the order of
      the options above. *)

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
      cutting a character short still reports it whole. A character
      {!val-prepend} inserted has the span of the first character. Each
      character of a replacement has the span of the last character it replaced;
      one that replaced nothing has the span of the character before it, or an
      empty span at the start of the text. A character the normalizer removed
      has no span at all, so the result can cover original bytes that no span
      reports. The spans of successive bytes never go backwards. An empty
      [start = stop] gives an empty span, except on a normalized text that is
      empty, where it gives the span of the whole original text.

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
      has invalid parameters. A missing member takes the default of its
      constructor; a member without one, such as [Prepend]'s ["prepend"], is
      required. A member of the wrong type is an error naming it. *)
end

module Pre_tokenizer : sig
  (** Pre-tokenization.

      Pre-tokenizers split raw text into pretokens before vocabulary-based
      tokenization (BPE, WordPiece, etc.) is applied. Each pretoken carries byte
      offsets into the original text.

      A text need not be valid UTF-8. A byte that opens no well-formed character
      is a character of one byte that is not white space, a letter, a digit, a
      mark or punctuation. {!byte_level} and {!split_regex} match it with no
      class, so it lies between their matches. *)

  type t
  (** The type for pre-tokenizers. *)

  (** {1:constructors Constructors} *)

  val whitespace : t
  (** [whitespace] cuts the text into runs of word characters and runs of other
      characters, dropping white space. On a valid UTF-8 text the pretokens are
      the matches of [\w+|[^\w\s]+].

      A word character is alphabetic, a mark, a decimal digit, connector
      punctuation such as [_], or a join control (U+200C, U+200D).
      {!split_regex}'s [\w] differs. It leaves out the join controls and counts
      ², ³, ¹, ¼, ½ and ¾. *)

  val whitespace_split : t
  (** [whitespace_split] splits the text on white space and drops it. White
      space is a character with the Unicode White_Space property. *)

  val bert : t
  (** [bert] splits the text on white space and makes each punctuation character
      a pretoken of its own, as [sequence [whitespace_split; punctuation ()]]
      does. *)

  val byte_level :
    ?add_prefix_space:bool -> ?use_regex:bool -> ?trim_offsets:bool -> unit -> t
  (** [byte_level ()] is a byte-level pre-tokenizer. Used by GPT-2, GPT-3,
      RoBERTa.

      It cuts the text with GPT-2's pattern, whose alternatives are
      ['s|'t|'re|'ve|'m|'ll|'d], [ ?\p{L}+], [ ?\p{N}+], [ ?[^\s\p{L}\p{N}]+],
      [\s+(?!\S)] and [\s+] in that order, and spells each pretoken in the
      byte-level alphabet, one character per byte.

      - [add_prefix_space]: prepend a space (U+0020) to a text that does not
        start with one. The space stands at the text's first character. Default:
        [true].
      - [use_regex]: cut with GPT-2's pattern. Without it the text is one
        pretoken. Default: [true].
      - [trim_offsets]: kept in {!to_json} for {!Post_processor.byte_level}. It
        changes no pretoken. Default: [true]. *)

  type behavior =
    [ `Isolated  (** Keep each delimiter as a pretoken of its own *)
    | `Removed  (** Drop the delimiters *)
    | `Merged_with_previous  (** Keep a delimiter with what precedes it *)
    | `Merged_with_next  (** Keep a delimiter with what follows it *)
    | `Contiguous
      (** Keep neighbours that are both delimiters, or both not, as one pretoken
      *) ]
  (** Delimiter handling behavior for splitting operations.

      [`Merged_with_previous] merges a delimiter into the text before it and
      [`Merged_with_next] into the text after it. A delimiter whose neighbour on
      that side is another delimiter is a pretoken of its own. *)

  val punctuation : ?behavior:behavior -> unit -> t
  (** [punctuation ()] cuts the text at its punctuation characters, the
      delimiters. White space is text like any other here. A punctuation
      character is a printable ASCII character other than a letter or a digit,
      or a character of the Unicode general category P.

      [behavior] defaults to [`Isolated]. *)

  val split : pattern:string -> ?behavior:behavior -> ?invert:bool -> unit -> t
  (** [split ~pattern ()] splits on a literal string [pattern]. See
      {!split_regex} to split on a regular expression.

      [behavior] defaults to [`Removed]. When [invert] is [true] the delimiters
      are the runs of text between the occurrences of [pattern], and those
      occurrences are what they separate; defaults to [false].

      An empty [pattern] matches at every position, so the pretokens are the
      characters. When [invert] is [true] and [behavior] is [`Removed] there are
      none. *)

  val split_regex :
    pattern:string -> ?behavior:behavior -> ?invert:bool -> unit -> t
  (** [split_regex ~pattern ()] is {!split} on the matches of the regular
      expression [pattern] rather than on the occurrences of a string. Matches
      are taken leftmost first without overlap, each the one a backtracking
      engine would find: the first alternative that matches, with greedy
      repetition.

      [pattern] is written in the dialect of tokenizer files, as for
      {!Normalizer.replace_regex}, except that anchors are not supported. A
      lookahead that ends the pattern or one of its alternatives is how
      [\s+(?!\S)] leaves the last space of a run to the word after it.

      An empty match cuts the text where it stands, so a [pattern] that matches
      the empty string everywhere gives the characters.

      Raises [Invalid_argument] if [pattern] is not valid or uses a construct
      that is not supported; the message says which. *)

  val char_delimiter : string -> t
  (** [char_delimiter c] splits on the character [c], removing it from the
      output. Equivalent to [split ~pattern:c ~behavior:`Removed ()].

      Raises [Invalid_argument] if [c] is not exactly one well-formed UTF-8
      character. *)

  val digits : ?individual_digits:bool -> unit -> t
  (** [digits ()] isolates each run of digits from the text around it. A digit
      is a character of the Unicode general category N (Nd, Nl or No).

      When [individual_digits] is [true], each digit is a separate pretoken;
      when [false] (default), consecutive digits are grouped. *)

  type prepend_scheme =
    [ `First  (** Prepend to the pretoken that opens the document only *)
    | `Never  (** Never prepend *)
    | `Always  (** Prepend to every pretoken not starting with a space *) ]
  (** Controls when metaspace prepends the replacement character. In a
      {!sequence} the pretokens are those of the member before it, so [`First]
      marks the first of them and [`Always] each. The encode pipeline counts a
      pretoken as opening the document only when nothing, an added token
      included, comes before it. *)

  val metaspace :
    ?replacement:string ->
    ?prepend_scheme:prepend_scheme ->
    ?split:bool ->
    unit ->
    t
  (** [metaspace ()] replaces each space (U+0020) with a visible marker. Used by
      SentencePiece models.

      - [replacement]: the marker spaces are replaced with. Default: ["▁"]
        (U+2581). It must be exactly one character.
      - [prepend_scheme]: when to prepend the marker, which happens only if the
        marked text does not start with one already. Default: [`Always].
      - [split]: whether to split before each marker. Default: [true].

      Pretokens are those of the marked text; their offsets are the bytes of the
      text they were made from. A marker that replaced a space stands at that
      space, and a prepended one at the character it opens.

      Raises [Invalid_argument] if [replacement] is not exactly one well-formed
      UTF-8 character. *)

  val unicode_scripts : t
  (** [unicode_scripts] splits on Unicode script boundaries.

      A pretoken opens where the writing system changes (e.g. Latin to Cyrillic,
      Latin to Han) and runs to the next change. The writing system is the
      Script property of Unicode 18.0.0. Hiragana and Katakana count as Han, as
      does the prolonged sound mark ["ー"] (U+30FC). HuggingFace's scripts are
      older, so a character it gives no script, such as U+0558, Armenian since
      18.0.0, can cut a pretoken there that it does not.

      Spaces and characters of no known script join the pretoken they follow, so
      a leading run of them belongs to no pretoken: unlike the other
      pre-tokenizers, the pretokens need not cover the input. *)

  val fixed_length : int -> t
  (** [fixed_length n] cuts the text into pretokens of [n] characters.

      The last pretoken may be shorter than [n].

      Raises [Invalid_argument] if [n] is not positive. *)

  val sequence : t list -> t
  (** [sequence ts] cuts the text with the first of [ts], then cuts each
      pretoken with the next, as a text of its own whose spans are placed inside
      that pretoken's. [sequence []] gives the text as one pretoken, an empty
      text included. *)

  (** {1:ops Operations} *)

  val pre_tokenize : t -> string -> (string * (int * int)) list
  (** [pre_tokenize t text] splits [text] into pretokens with byte offsets.

      Returns a list of [(pretoken, (start, end_))] where [start] and [end_] are
      byte positions in [text], ascending and within it. A pretoken is the bytes
      of its span unless [t] rewrote or encoded them, in which case the span is
      where the bytes it was made from lie. An empty text gives no pretoken,
      except under [sequence []].

      Two spans can cover the same bytes. A marker {!metaspace} prepends stands
      at the character it opens, so a member after it that cuts the marker off
      gives both pretokens that span. In a {!sequence} that encodes before its
      last member, rewrites twice or holds a fixed-length member, the pretokens
      a member cuts from one an earlier member rewrote or encoded can report
      that pretoken's whole span, where HuggingFace places each within it. *)

  (** {1:fmt Formatting} *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf t] formats [t] for inspection. *)

  (** {1:byte_level_decode Byte-level decoding} *)

  val byte_level_decode : string -> string
  (** [byte_level_decode s] is the bytes the byte-level alphabet spells [s]
      from, one byte per character. A single character outside the alphabet, an
      invalidly encoded one included, leaves [s] as it is, so one such character
      costs the whole string its mapping.

      The result is bytes and need not be valid UTF-8. *)

  (** {1:serialization Serialization} *)

  val to_json : t -> Jsont.json
  (** [to_json t] serializes [t] to HuggingFace JSON format. *)

  val of_json : Jsont.json -> (t, string) result
  (** [of_json json] is a pre-tokenizer from HuggingFace JSON format. Errors if
      [json] is not an object, has a missing or unknown ["type"] field, has
      invalid parameters, or is a ["Split"] whose regular expression
      ([{"Regex": ...}]) uses a construct {!split_regex} does not support.

      A field it lacks takes its constructor's default, except a ["Metaspace"]'s
      ["replacement"], which it requires, and a ["FixedLength"] without
      ["length"] has length [5]. A field of the wrong type, or a length that is
      not an integer, is an error naming it. *)
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

      [trim_offsets] and [add_prefix_space] trim the offsets of byte-level
      tokens as {!byte_level}'s do. Both default to [true]. *)

  val byte_level : ?add_prefix_space:bool -> ?trim_offsets:bool -> unit -> t
  (** [byte_level ()] trims the byte offsets of byte-level tokens. It inserts no
      token.

      [trim_offsets] moves a token's span past each leading, and before each
      trailing, U+0120 (the encoded space) or White_Space character of the
      token. An encoded tab or newline is a letter and stays. A token of nothing
      else keeps an empty span, at its end, or at its start when its leading
      space is kept. [add_prefix_space] keeps the leading space of a token that
      has exactly one, which the pre-tokenizer added, when the token is the
      first of a sequence or starts at offset [0]. Both default to [true]. *)

  val template :
    single:string -> ?pair:string -> ?special_tokens:token list -> unit -> t
  (** [template ~single ()] is a template-based post-processor.

      Templates use [$A] and [$B] as sequence placeholders and literal special
      token names (e.g. [[CLS]]). A colon suffix sets a piece's type ID, as in
      [$A:0] or [[SEP]:1]; a piece without one has type ID [0]. [$a], [$b] and
      [$] are [$A], [$B] and [$A], and [$n] is [$A:n]. A token is named by its
      text, so one whose text holds [:] cannot appear.

      [pair] must reference both [$A] and [$B]; it defaults to ["$A:0 $B:1"].
      [special_tokens] defaults to [[]]. A later entry of [special_tokens]
      replaces an earlier one of the same text.

      Raises [Invalid_argument] if a template names a token not in
      [special_tokens] or has a suffix that is not an integer in \[0, 2{^ 32}),
      or if [pair] lacks [$A] or [$B]. *)

  val sequence : t list -> t
  (** [sequence processors] chains [processors] left to right. Each hands the
      next the sequences it gives, and {!process} merges them after the last.
      {!bert} and {!roberta} read the first sequence they receive as their
      single one and each later one as a second, {!byte_level} trims each, and a
      {!template} reads the first as [$A] and the second as [$B] and gives each
      of its pieces as a sequence. So after [template ~single:"[X] $A"], [bert]
      receives [[X]] and [A], and the result is [[CLS] [X] [SEP] A [SEP]]. *)

  (** {1:processing Processing} *)

  val process :
    t -> ?pair:Encoding.t -> Encoding.t -> add_special_tokens:bool -> Encoding.t
  (** [process t enc ~add_special_tokens] adds special tokens and sets type IDs
      on [enc]. [enc] enters with type ID [0], and [pair] with [1], whatever
      they arrive with; [t] may override either.

      When [~pair] is provided, both sequences are merged into a single
      encoding. Offsets are not shifted: each sequence keeps the offsets into
      its own text.

      When [~add_special_tokens] is [false], no special token is inserted. The
      two sequences are still merged, type IDs still assigned and byte-level
      offsets still trimmed.

      Raises [Invalid_argument] if [t] holds a template that names a sequence it
      does not receive, as a [single] naming [$B], or that receives more than
      two, as one after a template of three pieces in a {!sequence}. *)

  val added_tokens : t -> is_pair:bool -> int
  (** [added_tokens t ~is_pair] is the number of special tokens {!process}
      inserts into one sequence, or into a pair when [is_pair]. Useful for
      calculating the truncation budget. Raises [Invalid_argument] where
      {!process} does. *)

  (** {1:fmt Formatting} *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a post-processor for inspection. A template prints its
      [single] and [pair] in {!template}'s syntax, each piece with its type ID,
      and each special token with its IDs. *)

  (** {1:serialization Serialization} *)

  val of_json : Jsont.json -> (t, string) result
  (** [of_json json] is a post-processor from HuggingFace [tokenizer.json]
      format. Errors if [json] is not an object, has a missing or unknown
      ["type"] field, or has invalid parameters.

      A missing ["trim_offsets"] or ["add_prefix_space"] is [true], a missing
      ["single"] is ["$A"], and a missing or [null] ["pair"] is ["$A:0 $B:1"]. A
      template may also be a string in {!template}'s syntax, and a later special
      token replaces an earlier one of the same name, as in {!template}. ["sep"]
      and ["cls"] are required, and a member of the wrong type, or an id or type
      id that is not an integer from [0] to 2{^ 31}[ - 1], is an error naming
      it. A ["RobertaProcessing"] without its flags reads as {!roberta} with
      both [true]; HuggingFace reads it as a ["BertProcessing"]. *)

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
      decoders that follow it.

      An empty string to replace, a [bpe] [suffix], a [ctc] token or a [replace]
      [pattern], occurs before every character of a token and at its end. In an
      empty token it occurs once under [bpe] and [ctc], and not at all under
      [replace], as in {!Normalizer.replace}. *)

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
      sequence becomes one U+FFFD: four stray bytes cost four, while a four-byte
      character cut short costs one. *)

  val byte_fallback : t
  (** [byte_fallback] is a decoder for byte fallback tokens. Each run of hex
      byte tokens (e.g. ["<0x41>"]) becomes the text those bytes spell; a run
      that is not valid UTF-8 becomes one U+FFFD per byte. Other tokens pass
      through unchanged. A byte token is [<0x], two hexadecimal digits of either
      case and [>]. *)

  val wordpiece : ?prefix:string -> ?cleanup:bool -> unit -> t
  (** [wordpiece ~prefix ~cleanup ()] is a decoder for WordPiece tokens. Strips
      continuation [prefix] (default ["##"]) from non-initial subwords and gives
      the others a leading space. When [cleanup] is [true] (default), applies
      the detokenization cleanup to every token, once its joining space is
      prepended: the space before [.], [?], [!], [,], [n't], ['m], ['s], ['ve]
      and ['re] is taken back, [" ' "] becomes ["'"] and [" do not"] becomes
      [" don't"]. So ["hello"; ","; "world"] decodes to ["hello, world"], while
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
      token is dropped instead of becoming a space, on [`First] as on [`Always],
      the two prepending to the same first token. [prepend_scheme] defaults to
      [`Always].

      Raises [Invalid_argument] if [replacement] is not exactly one well-formed
      UTF-8 character. *)

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
      with [by] in each token. An empty [pattern] occurs before every character
      and at the end of each token that is not empty. *)

  val strip : ?content:string -> ?start:int -> ?stop:int -> unit -> t
  (** [strip ~content ~start ~stop ()] removes up to [start] leading and [stop]
      trailing occurrences of [content] from each token. [content] defaults to
      [" "], [start] and [stop] to [0]. A token left with nothing between the
      two cuts becomes empty. An empty [content] strips nothing.

      HuggingFace reads [content] as a single character, so {!val-to_json} on a
      decoder whose [content] is longer, or empty, writes a decoder it rejects.

      Raises [Invalid_argument] if [start] or [stop] is negative. *)

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
      parameters. A missing member takes the default of its constructor; a
      [Replace] without ["pattern"] or ["content"] is an error, as is a member
      of the wrong type or a count that is not an integer from [0] to
      2{^ 31}[ - 1]. *)
end

(** {1:types Types} *)

type t
(** The type for tokenizers. What a tokenizer gives is a function of its
    configuration alone, and under {!bpe}'s [dropout] of the state of its [rng]
    too. It holds encoder states, the pretoken caches and buffers that encodings
    reuse, which grow as it encodes and are freed with it. One tokenizer can
    encode from several domains and threads at once. *)

type direction = [ `Left | `Right ]
(** The type for padding and truncation directions. [`Left] operates at the
    beginning of the sequence, [`Right] at the end. *)

type added_token = {
  content : string;
      (** The token text: ["[CLS]"] for a special one, ["<name>"] for an
          ordinary vocabulary entry matched atomically. *)
  special : bool;  (** Whether decoding may skip this token. *)
  single_word : bool;
      (** Whether this token matches only where no word character (an alphabetic
          character, such as a letter or a letter number like Ⅻ, a mark, a
          decimal digit, connector punctuation or a joiner) touches it on either
          side. A match refused this way is passed over: the search resumes
          after it, so no token is tried at its position or inside it. *)
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
    vocabulary keeps its ID, the others take the next free ID, one past the
    largest the model or an earlier added token holds. This departs from Hugging
    Face, which numbers them from the size of the model vocabulary: on a
    vocabulary whose IDs leave gaps, that number can be one the model holds, and
    the model's entry then decodes as the added token. The call that would
    number a token 2{^ 31} or more, past the [int32] values of {!type-ids},
    raises [Invalid_argument]. A token with [special] set is skipped by
    {!decode} when [skip_special_tokens] is [true]; one without it is an
    ordinary vocabulary entry that happens to be matched atomically. The
    semantic role (pad, unk, bos, etc.) is contextual, not encoded in the type.
*)

type pad_length = [ `Batch_longest | `Fixed of int | `To_multiple of int ]
(** The type for padding length strategies.

    - [`Batch_longest]: pad to the longest sequence in the batch. {!encode} and
      {!encode_ids} encode a batch of one, so it adds nothing there.
    - [`Fixed n]: pad every sequence shorter than [n] tokens to [n]; a longer
      one is left as it is.
    - [`To_multiple n]: pad to the smallest multiple of [n] that is at least the
      sequence length. With [n <= 0] nothing is padded. *)

type padding = {
  length : pad_length;
  direction : direction;
  pad_id : int option;
  pad_type_id : int option;
  pad_token : string option;
}
(** The type for padding configurations. An encoding's windows in
    {!Encoding.overflowing} are padded to the length it is padded to, even when
    it has that length already.

    [pad_token] defaults to the tokenizer's padding token and [pad_type_id] to
    [0]. [pad_id] defaults to the identifier of the pad token in use. An encoder
    given a padding raises [Invalid_argument] if [pad_token] is [None] and the
    tokenizer has no padding token, even when [pad_id] is given, or if [pad_id]
    is [None] and the pad token is not in the vocabulary. It raises at the call,
    whatever the lengths of the texts. *)

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
    context across windows. Defaults to [0]. [stride] must be at most the
    tokens left for content of a single text, whatever the text, and less than
    the tokens a sequence keeps when one is truncated to some (on a pair, its
    share); otherwise {!encode} and the other encoders raise
    [Invalid_argument]. The tokens left are those of the call: under
    [~add_special_tokens:false] all of [max_length] is room for content. This
    accepts a stride tokenizers refuses, since it checks the stride when
    truncation is configured, before any call, against the room the special
    tokens leave.

    Before it truncates, {!encode} keeps only the pretokens that hold the first
    [max_length] tokens of a text, or the last under [`Left], so the windows of
    {!Encoding.overflowing} cover those tokens and nothing beyond.

    Raises [Invalid_argument] if [stride] is negative, or if [max_length] is
    positive and [stride] is not less than it. *)

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
  ?rng:Random.State.t ->
  ?continuing_subword_prefix:string ->
  ?end_of_word_suffix:string ->
  ?fuse_unk:bool ->
  ?byte_fallback:bool ->
  ?ignore_merges:bool ->
  unit ->
  t
(** [bpe ()] is a BPE (Byte Pair Encoding) tokenizer. Used by GPT-2, GPT-3,
    RoBERTa. It needs 63-bit integers, and raises [Invalid_argument] where
    OCaml's integers are smaller (js_of_ocaml, wasm_of_ocaml).

    - [normalizer]: text normalization. Default: none.
    - [pre]: pre-tokenization strategy. Default: none.
    - [post]: post-processor for special tokens. Default: none.
    - [decoder]: decoding strategy. Default: none.
    - [added_tokens]: added tokens. They are matched atomically in the input on
      {!encode}: against the raw text before normalization when [normalized] is
      unset, against the normalized text when it is set. They are numbered past
      the largest ID of [vocab] ({!type-added_token}) without entering it. A
      token of empty content is ignored, and a content given twice is one token,
      with the ID of its first and the flags of its last. Raises
      [Invalid_argument] when a normalized token normalizes to the empty text,
      and when two of different content match the same text, as two normalized
      tokens whose contents normalize alike do. Default: [[]].
    - [bos_token], [eos_token], [pad_token]: role markers. Each is a special
      token in its own right, matched in the input like the entries of
      [added_tokens] and numbered like them when [vocab] does not hold it. One
      that [added_tokens] already holds keeps the flags given there. Default:
      none.
    - [unk_token]: the token a character stands as when neither [vocab] nor
      [byte_fallback] covers it. It is a model parameter, never matched
      atomically in the input. Without one, such a character gives no token;
      with one that [vocab] does not hold, encoding raises [Failure] at it.
      Default: none.
    - [vocab]: initial vocabulary as [(token, id)] pairs. A token given twice
      keeps only its last pair. The ids may leave gaps of any size: the
      tokenizer's memory follows the number of tokens. Raises
      [Invalid_argument] if an id is negative or 2{^ 31} or more, past the
      [int32] values of {!type-ids}. Default: [[]].
    - [merges]: merge rules as [(left, right)] pairs learned during training. A
      word starts as its characters; the adjacent pair whose rule comes first in
      [merges] is merged, the leftmost of equal pairs first, until no rule
      applies. Raises [Invalid_argument] if a merge names a token [vocab] does
      not hold, or makes one. Default: [[]].
    - [cache_capacity]: the most entries of a pretoken cache, rounded up to a
      power of two, which keeps the ids of pretokens of up to 15 bytes so that
      one seen before is not merged again. The tokenizer makes a cache for
      each encoding that runs while the others are in use, from any domains or
      threads, and holds at most [Domain.recommended_domain_count ()] of them
      at any time, in use or not, for as long as it lives; an encoding past
      that runs without one. Each starts at 16 KB, grows to hold the
      vocabulary's short entries, and doubles as the pretokens it meets fill
      half of it, up to 32 bytes per entry, and never shrinks: about 32 MB
      over 100 MB of web text. While it doubles, an encoding briefly holds
      the table it leaves too, half the size of the new one, and several
      encodings may double at once. Default: [1048576]: 32 MB each at most, 48
      MB while it doubles, so a tokenizer's caches hold at most
      [Domain.recommended_domain_count ()] times 32 MB between encodings (448
      MB on 14 cores) and times 48 MB while they all double (672 MB). [0]
      disables caching.
    - [dropout]: probability \[[0]; [1]\] of skipping merges (data
      augmentation). Raises [Invalid_argument] if it is outside \[[0]; [1]\],
      NaN included. Default: none (no dropout).
    - [rng]: the state [dropout] draws from, which the tokenizer then owns.
      Encodings draw from states split off it, one for each encoding that runs
      while others do, so two tokenizers made alike from equal [rng] states give
      the same tokens to the same texts encoded one after another. Default: a
      state of the tokenizer's own made from a constant seed, so that its draws
      repeat from run to run; pass [Random.State.make_self_init ()] for draws
      that differ.
    - [continuing_subword_prefix]: prefix for non-initial subwords (e.g.,
      ["##"]). Default: none; [""] is the same as none.
    - [end_of_word_suffix]: suffix marking word boundaries (e.g., ["</w>"]).
      Default: none; [""] is the same as none. [continuing_subword_prefix] goes
      on every character of a word but the first and [end_of_word_suffix] on its
      last, before any merge, so a merge names the affixed forms.
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
  ?cache_capacity:int ->
  unit ->
  t
(** [wordpiece ()] is a WordPiece tokenizer. Used by BERT, DistilBERT, Electra.

    WordPiece uses a greedy longest-match-first algorithm to split words into
    subword pieces prefixed with a continuation marker (e.g., ["running"]
    becomes [["run"; "##ning"]]).

    - [vocab]: initial vocabulary as [(token, id)] pairs. A token given twice
      keeps only its last pair. The ids are as in {!bpe}'s [vocab]. Default:
      [[]].
    - [unk_token]: the token a word stands as when no run of pieces covers it. A
      [vocab] that does not hold it encodes the words it covers, and encoding
      raises [Failure] at a word that needs it. Default: ["[UNK]"].
    - [continuing_subword_prefix]: prefix for non-initial subwords. Default:
      ["##"].
    - [max_input_chars_per_word]: words of more characters than this, a byte
      that opens no whole character counting as one, are replaced with
      [unk_token]. Default: [100].
    - [cache_capacity]: the most entries of a pretoken cache, which keeps the
      ids of pretokens of up to 15 bytes so that one seen before is not
      encoded again. The tokenizer makes a cache for each encoding that runs
      while the others are in use, from any domains or threads, and holds at
      most [Domain.recommended_domain_count ()] of them at any time, in use or
      not, for as long as it lives; an encoding past that runs without one.
      Each starts at 144 KB and doubles as it fills, up to 128 KB plus 32
      bytes per entry. [0] disables them. Default: [262144] (8 MB each at
      most).

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

    - [vocab]: initial vocabulary as [(word, id)] pairs. A word given twice
      keeps only its last pair. The ids are as in {!bpe}'s [vocab]. Default:
      [[]].
    - [unk_token]: the token a word not in [vocab] stands as. A [vocab] that
      does not hold it encodes the words it holds, and encoding raises [Failure]
      at any other. Default: ["<unk>"].

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
  ?cache_capacity:int ->
  unit ->
  t
(** [unigram ()] is a Unigram tokenizer. Used by AlBERT, T5, mBART.

    A pretoken is cut into the vocabulary pieces whose scores add up to the
    most, rather than into the longest match at each byte. The scores are summed
    from the left, and of the cuts whose sums are equal the one whose last piece
    starts first is kept, then the one whose piece before that starts first, and
    so on. A character no piece covers costs an unknown token, scored ten below
    the rarest piece of the vocabulary.

    - [vocab]: initial vocabulary as [(token, score)] pairs where scores are
      log-probabilities (negative numbers). An entry is identified by its
      position. Default: [[]].
    - [unk_id]: the entry standing for a run of characters the vocabulary does
      not hold, one token for the whole run, as a [tokenizer.json] names it.
      Without one, encoding raises [Failure] as soon as the best path into some
      character would spend an unknown token on it. Raises [Invalid_argument] if
      it is not the position of an entry. Default: the position of [unk_token]
      in [vocab], if it is there.
    - [byte_fallback]: spell such a run out as byte tokens (["<0xFF>"]) when the
      vocabulary holds one for every byte of it. Default: [false].
    - [unk_token]: the model's unknown entry when [vocab] holds it and [unk_id]
      is not given. Default: none.
    - [cache_capacity]: the most entries of a pretoken cache, which keeps the
      ids of pretokens of up to 15 bytes so that one seen before is not
      encoded again. The tokenizer makes a cache for each encoding that runs
      while the others are in use, from any domains or threads, and holds at
      most [Domain.recommended_domain_count ()] of them at any time, in use or
      not, for as long as it lives; an encoding past that runs without one.
      Each starts at 144 KB and doubles as it fills, up to 128 KB plus 32
      bytes per entry. [0] disables them. Default: [262144] (8 MB each at
      most).

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
  unit ->
  t
(** [chars ()] is a byte-level tokenizer: each byte of the input is one token
    whose ID is the byte's value. A multi-byte character is as many tokens as it
    has bytes. No vocabulary is required: its vocabulary is the 256 one-byte
    strings, the token of ID [b] being the byte [b].

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

    - [vocab]: path to the vocabulary file. With [merges], a [vocab.json]: a
      JSON object mapping tokens to IDs ([{"hello": 0, "world": 1}]). Without
      [merges], a BERT-style [vocab.txt]: one token per line without the
      White_Space characters that end it, its ID the line's number from 0. A
      blank line holds the empty token; of several blank lines, the empty token
      takes the last one's number.
    - [merges]: path to merges file ([merges.txt]). One merge per line as
      space-separated token pairs. Lines starting with ["#version"] are skipped.

    - [unk_token]: as in {!bpe} with [merges], and as in {!wordpiece} without.

    Raises [Sys_error] if a file cannot be read and [Failure] if one is not in
    its format.

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token]) are as in {!bpe}. *)

val add_tokens : t -> added_token list -> t
(** [add_tokens t tokens] is [t] with [tokens] registered as added tokens,
    exactly as if they had been passed as [added_tokens] at construction:
    matched atomically in the input, numbered past the largest ID in use, and
    skipped by {!decode} when their [special] is set. Works for every model.

    A token [t] already holds keeps the ID it has; its flags are replaced by the
    new ones. Added tokens never enter the model's own vocabulary: to build one,
    pass [vocab] to the constructor.

    Raises [Invalid_argument] if a token's number would be 2{^ 31} or more
    ({!type-added_token}). *)

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
(** [unk_token t] is the token [t]'s model stands unknown text as, if any,
    defaults included, whether or not its vocabulary holds it. A Unigram model's
    is its entry at [unk_id]; a {!chars} model has none. *)

(** {1:vocab Vocabulary} *)

val vocab : t -> (string * int) list
(** [vocab t] is [t]'s vocabulary as [(token, id)] pairs: the model's, in no
    particular order, followed by the added tokens it does not already hold. *)

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
  ?match_special_tokens:bool ->
  ?padding:padding ->
  ?truncation:truncation ->
  string ->
  Encoding.t
(** [encode t text] is the encoding of [text] by [t].

    Added tokens occurring in [text] are matched atomically and emitted with
    their own ID, whatever [add_special_tokens] is. Text that comes from a user
    must be encoded with [~match_special_tokens:false], so that typing
    ["<|endoftext|>"] does not end the text. The model can still give a special
    token's ID when its vocabulary holds that token's text as a piece, as
    T5's holds ["</s>"].

    [text] need not be valid UTF-8: a byte that opens no whole character, a lead
    byte without the continuation bytes it asks for, a continuation byte with no
    lead, or a byte of a sequence RFC 3629 forbids (an overlong form, a
    surrogate, a value above U+10FFFF), is read by every stage as a character of
    one byte.

    {!Encoding.offsets} are byte spans of [text] itself, not of the normalized
    form: a token of ["café"] under a normalizer that strips accents reports the
    bytes of the accented text. {!Encoding.word_ids} number the pretokens of
    [text] from [0], an added token counting as one; a pretoken the model gives
    no token keeps its number, so the numbers can skip. Both are worked out when
    they are first read, so a caller that only wants {!Encoding.ids} pays for
    neither; the {!Encoding} module doc says how that first read behaves across
    domains.

    - [pair]: a second sentence for sentence-pair tasks. The post-processor
      merges both sequences with appropriate type IDs; without one, [pair]'s
      tokens follow [text]'s with type ID [1]. Default: none.
    - [add_special_tokens]: whether to insert special tokens via the
      post-processor. Default: [true].
    - [match_special_tokens]: whether the added tokens whose [special] is set
      are matched in [text] and [pair]. With [false], the text of such a token
      is normalized, pre-tokenized and encoded by the model like the text
      around it. The other added tokens are still matched, except inside such
      a token's text by one matched against the same form of the text, raw or
      normalized. Default: [true].
    - [padding]: padding configuration. Default: none (no padding).
    - [truncation]: truncation configuration. Default: none (no truncation). *)

type exec = int -> (int -> unit) -> unit
(** The type for executors, which run the tasks of a batch on the program's
    domains. [exec n task] calls [task i] once for each [i] from [0] to
    [n - 1], on any domains and in any order, and returns once every call has
    returned. A task never raises: a batch records what its work raised and
    re-raises it once [exec] returns.

    The batch plans the work, since only it knows where a text may be cut and
    how padding spans the rows; each task takes chunks of it until none is left.
    The first chunk that raises ends the handout: each task finishes the chunk
    it holds and takes no other, and a task that runs after the batch has failed
    or [exec] has returned takes none. A batch calls its executor at most once,
    so an executor over a pool the program already runs spreads every batch
    without starting a domain. Over domainslib's [Task] pool, for instance:
    {[
      let exec pool n task =
        Domainslib.Task.run pool (fun () ->
            Domainslib.Task.parallel_for pool ~start:0 ~finish:(n - 1)
              ~body:task)
    ]}
    A batch asks for at most [Domain.recommended_domain_count ()] tasks, so
    an executor may start a domain for each.

    An executor over [d] domains started for the call, the caller's among
    them. A domain runs the program's signal handlers, and a handler that
    raises on a new domain before the runtime has set it up crashes the
    program. So the executor blocks the [signals] the program handles while it
    starts its domains, which keep that mask. Blocking them takes the
    [threads.posix] and [unix] libraries in the caller's program; brot
    depends on neither. On Windows, which has no signal masks, [signals] is
    [[]].
    {[
      let spawn ~signals d n task =
        let next = Atomic.make 0 in
        let rec pull () =
          let i = Atomic.fetch_and_add next 1 in
          if i < n then (task i; pull ())
        in
        let start () = List.init (min d n - 1) (fun _ -> Domain.spawn pull) in
        let others =
          if signals = [] then start ()
          else
            let mask = Thread.sigmask Unix.SIG_BLOCK signals in
            let restore () = ignore (Thread.sigmask Unix.SIG_SETMASK mask) in
            Fun.protect ~finally:restore start
        in
        pull ();
        List.iter Domain.join others
    ]}
    A domain that ends stops every domain for a collection, so a program
    that encodes many batches passes an executor over a pool. *)

val encode_batch :
  t ->
  ?add_special_tokens:bool ->
  ?match_special_tokens:bool ->
  ?padding:padding ->
  ?truncation:truncation ->
  ?exec:exec ->
  string list ->
  Encoding.t list
(** [encode_batch t texts] is the encoding of each text in [texts].

    A batch is encoded on the calling domain unless an executor spreads it over
    others: where a program's work runs is its decision.

    - [exec]: the executor the batch is spread through, as {!type-exec} says.
      Default: none, and the batch is encoded on the calling domain. The work is
      planned in chunks of consecutive texts, each closed at the first text that
      brings it to a target between 64 KB and 256 KB, the last holding what is
      left, and handed out largest first; a batch holding under 64 KB is encoded
      on the calling domain without calling [exec]. The result does not depend
      on [exec]. Raises [Invalid_argument] if [exec] returns before every chunk
      is encoded, and what [exec] raises.
    - Other optional parameters are as in {!encode}. For sentence-pair tasks,
      use {!encode_batch_pairs}. *)

val encode_batch_pairs :
  t ->
  ?add_special_tokens:bool ->
  ?match_special_tokens:bool ->
  ?padding:padding ->
  ?truncation:truncation ->
  ?exec:exec ->
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
  ?match_special_tokens:bool ->
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
    other. [Nx.of_bigarray (Bigarray.genarray_of_array1 ids)] copies one into
    a host value. *)

val encode_batch_ids :
  t ->
  ?add_special_tokens:bool ->
  ?match_special_tokens:bool ->
  ?padding:padding ->
  ?truncation:truncation ->
  ?exec:exec ->
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
    spread through [exec] too, in pieces cut where the pipeline cannot tell the
    difference. A pipeline with a normalizer, or one that does not split on
    spaces, encodes each text in one piece.

    Identifiers are narrowed to 32 bits. Every id is below 2{^ 31} ({!bpe},
    {!type-added_token}), so the narrowing changes none.

    The ids are gathered in buffers, one per domain encoding the batch, before
    they are copied into the result. [t] keeps them for the batches after,
    at most [Domain.recommended_domain_count ()] buffers of at most 8 MB each,
    and they are collected with [t].

    Optional parameters are as in {!encode_batch}. *)

val decode : t -> ?skip_special_tokens:bool -> int array -> string
(** [decode t ids] is the text obtained by decoding [ids] through [t]'s
    vocabulary and decoder. An ID outside the vocabulary gives no text. A
    tokenizer without a decoder joins the tokens with a space between two.

    [skip_special_tokens] defaults to [false]. It drops the added tokens whose
    [special] is set, and only those. *)

(** {1:streaming Streaming decoding}

    A model generates its output one token at a time, and a reader wants each
    piece of text as soon as it is final. Decoding each token on its own gets
    that text wrong. The decoder of a byte-level model may need several tokens
    for one character outside ASCII, and a SentencePiece decoder drops the
    space before the first word only. A {e stream} decodes the tokens as they
    come and gives out the text no later token changes, keeping the rest. A
    stream is a value. A step gives a new stream and leaves the old one as it
    was, so a stream can be kept to decode another continuation from it.

    A stream keeps text back in these cases only, and gives it out once the
    token that settles it arrives or at {!decode_finish}:
    - under a {!Decoder.bpe} decoder, the last token, while it holds the
      suffix, which becomes a space unless the token is the last;
    - under {!Decoder.byte_fallback}, an open run of byte tokens, however
      long, since one byte that is not UTF-8 turns each byte of the run into a
      replacement character;
    - under {!Decoder.byte_level}, up to three bytes that start a character
      not yet complete. A byte no later byte can complete, such as [\xff], is
      given its replacement character at once;
    - after a {!Decoder.byte_level} or {!Decoder.fuse} decoder in a
      {!Decoder.sequence}, the start of a {!Decoder.replace} pattern that may
      yet complete, the occurrences a {!Decoder.strip} may yet cut from either
      end, and everything when a {!Decoder.byte_level} or
      {!Decoder.byte_fallback} decoder follows, since it reads the text whole.

    {[
      let print_tokens tokenizer ids =
        let s = ref (Brot.decode_stream tokenizer) in
        Array.iter
          (fun id ->
            let text, next = Brot.decode_step !s id in
            print_string text;
            s := next)
          ids;
        print_string (Brot.decode_finish !s)
    ]} *)

type stream
(** The type for streaming decodings. *)

val decode_stream : ?skip_special_tokens:bool -> t -> stream
(** [decode_stream t] is a decoding by [t] that has been fed no token.
    [skip_special_tokens] is as in {!decode}. *)

val decode_step : stream -> int -> string * stream
(** [decode_step s id] is [(text, s')], where [s'] is [s] fed [id] and [text]
    is the text that no id fed after [id] changes, less what earlier steps
    gave. [text] is empty when [id] settles nothing, as when it ends in part of
    a character or extends a run of byte-fallback tokens.

    The texts of every step, followed by {!decode_finish}, make up {!decode}
    of the ids fed. Each is valid UTF-8 when the vocabulary is, as no text ends
    inside a character. *)

val decode_finish : stream -> string
(** [decode_finish s] is the text of the ids fed to [s] that no step has given
    out yet, decoded as the end of the ids. *)

(** {1:training Training}

    A trained vocabulary opens with its special tokens, counted in [vocab_size]:
    the added tokens, then the role markers ([bos_token], [eos_token],
    [pad_token]) not among them.

    A trainer raises [Sys_error] if a file of [`Files] cannot be read. *)

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
    each pretoken counts as one word. For a byte-level [pre] the words are in
    encoded form, and merges are learned over that form. With no [pre] a whole
    text is one word, so training on space-separated words needs
    [~pre:Pre_tokenizer.whitespace_split].

    The most frequent adjacent pair of characters is then merged over and over
    until [vocab_size] is reached, no pair is left, or the best pair falls below
    [min_frequency].

    - [init]: a tokenizer whose added tokens, with their flags, role markers
      and {!unk_token} carry over into the trained one; where the call gives the
      same token or role, the call's holds. The trained vocabulary holds the
      unknown token only when the added tokens do, so without it a word the
      model cannot cover raises [Failure]. Its vocabulary does not carry over:
      training always starts from an empty one. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. The
      alphabet is kept whole, so a [vocab_size] below its size gives the
      alphabet and the special tokens. Default: [30000].
    - [min_frequency]: minimum pair frequency to be merged. Default: [0].
    - [limit_alphabet]: maximum number of distinct characters to keep; the
      rarest go first, and words drop the characters that did not make the cut.
      Default: none (keep all).
    - [initial_alphabet]: characters to keep whatever their frequency. Each
      string stands for the code point it starts with, so ["été"] and ["é"] both
      mean [é]; an empty string, or one that does not start with valid UTF-8, is
      dropped. Default: [[]].
    - [continuing_subword_prefix]: prefix put on every character of a word but
      the first, before any pair is counted, so merges are learned and written
      over the affixed forms. Default: none.
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
  data ->
  t
(** [train_wordpiece data] trains a WordPiece tokenizer from [data].

    Learns the vocabulary with the BPE merge training of {!train_bpe}, over the
    same words, and keeps the merged tokens as WordPiece subwords.

    - [init]: as in {!train_bpe}. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. The
      alphabet is kept whole, so a [vocab_size] below its size gives the
      alphabet and the special tokens. Default: [30000].
    - [min_frequency]: minimum pair frequency to be merged. Default: [0].
    - [limit_alphabet]: maximum number of distinct characters to keep. Default:
      none (keep all).
    - [initial_alphabet]: characters to keep whatever their frequency, one code
      point per string as in {!train_bpe}. Default: [[]].
    - [continuing_subword_prefix]: prefix for non-initial subwords. Default:
      ["##"].
    - [unk_token]: as in {!wordpiece}. The trained vocabulary holds it only when
      [added_tokens] or [init] carries it. Default: ["[UNK]"].

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token]) are as in {!bpe}. *)

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
    so it holds whole pretokens and nothing is split further. Of equally
    frequent words the one first in byte order is kept first.

    - [init]: as in {!train_bpe}. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. Default:
      [30000].
    - [min_frequency]: minimum frequency for a word to be included. Default:
      [0].
    - [unk_token]: as in {!word_level}. The trained vocabulary holds it only
      when [added_tokens] or [init] carries it. Default: ["<unk>"].

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token]) are as in {!bpe}. *)

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
    each scored by the log of its share of the words counted, and no subword
    piece is ever proposed. A special token scores [0.].

    - [init]: as in {!train_bpe}. Default: create new.
    - [vocab_size]: target vocabulary size including special tokens. Default:
      [8000].
    - [unk_token]: names the model's unknown entry when [added_tokens] or [init]
      carries it, since a trained vocabulary of whole words covers no character
      on its own. Without one, the trained tokenizer raises [Failure] on any
      word it was not trained on. Default: none.

    Pipeline parameters ([normalizer], [pre], [post], [decoder], [added_tokens],
    [bos_token], [eos_token], [pad_token]) are as in {!bpe}. *)

(** {1:model_files Model files} *)

val save_model_files : ?prefix:string -> t -> folder:string -> string list
(** [save_model_files t ~folder] writes [t]'s model to files in [folder] as
    tokenizers' [Model.save] does and returns their paths. BPE writes
    [vocab.json] and [merges.txt], WordPiece [vocab.txt], WordLevel [vocab.json]
    and Unigram [unigram.json]; a {!chars} tokenizer writes none. With [prefix],
    each name is preceded by [prefix] and [-].

    Raises [Invalid_argument] if a token a JSON file holds is not valid UTF-8. *)

(** {1:huggingface HuggingFace compatibility} *)

val from_file : string -> (t, string) result
(** [from_file path] is a tokenizer loaded from a HuggingFace [tokenizer.json]
    file. Errors if the file cannot be read or has invalid format.

    The tokenizer gives the ids and tokens that Hugging Face's tokenizers gives
    for the same file and text, except on characters whose Unicode data differs
    from the older versions tokenizers reads, and where a value here says
    otherwise. A Unigram model's unknown token is the vocabulary's entry, where
    tokenizers gives the text it stands for.

    A Unigram model's scores are read from their text as HuggingFace's
    tokenizers reads them. It rounds the digits, as a 64-bit integer, to a
    double, and scales that by one multiplication or division by the double
    nearest a power of ten, dividing by 10{^ 308} first for as long as the
    exponent is below -308. The roundings can land one unit in the last place
    from the double the text names: T5's [-12.130167007446289] reads as
    [-12.130167007446287]. That unit can decide a tie between two cuts, and the
    cut follows tokenizers. A score tokenizers refuses as out of range is an
    error. *)

val of_json : Jsont.json -> (t, string) result
(** [of_json json] is a tokenizer deserialized from HuggingFace JSON format.
    Errors if [json] has no [model], a model of unknown type, or invalid
    parameters. A model without a [type] member is the model its members name:
    [merges] makes a BPE, [unk_id] a Unigram, [continuing_subword_prefix] or
    [max_input_chars_per_word] a WordPiece, and [vocab] alone a WordLevel. This
    differs from tokenizers, which reads a WordPiece only when both of those
    members are present, and refuses a WordPiece or WordLevel without
    [unk_token]. A member of the wrong type, or an id or count that is not an
    integer from [0] to 2{^ 31}[ - 1], is an error naming it.

    The [truncation] and [padding] members are not read, by [of_json] or
    {!from_file}. {!encode} takes both as arguments, so a file that sets them
    encodes as tokenizers encodes it when the same settings are passed, with two
    exceptions. Truncation reserves the tokens {!Post_processor.process}
    inserts, so a [Sequence] post-processor whose template is not its last
    member keeps [max_length], where tokenizers reserves what each member counts
    alone and can exceed it or truncate more. And tokenizers loads a [stride]
    past the room for content unchecked, where {!encode} raises.

    The numbers of [json] are taken as given. A tree decoded from text holds
    each number as the double its text names, which can differ in the last place
    from what {!from_file} reads off the same text. Either way a BPE dropout is
    held in single precision, as tokenizers holds it, and draws from a state of
    its own made from a constant seed, as {!bpe}'s default [rng], so that two
    loads of one file draw alike.

    The [added_tokens] member gives the tokens matched atomically in the input.
    A missing [special] member reads as [true], and a missing [normalized] as
    [not special], as in {!val-added_token}. Their IDs are reassigned as
    documented in {!type-added_token}, not read from the file. *)

val to_json : t -> Jsont.json
(** [to_json t] is [t] serialized to HuggingFace JSON format. Its numbers are
    doubles. Encoding the tree with jsont writes them as jsont writes numbers,
    and {!save_pretrained} writes them as tokenizers does.

    [to_json] writes no [bos_token], [eos_token] or [pad_token], since
    [tokenizer.json] has no member for them. The tokenizer {!of_json} reads back
    holds them as added tokens without their roles. A {!chars} tokenizer is
    written with the model type ["Chars"], which only brot reads. *)

val save_pretrained : t -> path:string -> unit
(** [save_pretrained t ~path] saves [t] to [path] in HuggingFace format. Creates
    [path/tokenizer.json].

    Numbers are written as tokenizers writes them, the same on every system. A
    Unigram score is the shortest decimal that reads back as the double, and a
    BPE dropout the shortest that reads back in single precision, as in [0.1],
    [0.0], [0.00001], [1.5e-7] and [1e+16]. Integers are written as their
    digits.

    Raises [Invalid_argument] if a token is not valid UTF-8, since JSON text
    holds UTF-8 alone, and [Sys_error] if [path] cannot be written. *)

(** {1:tiktoken tiktoken encodings}

    A {e tiktoken encoding} is a byte-level BPE given by its {e ranks}, a table
    from byte strings to distinct integers that holds every single byte, a
    pattern that cuts text into pretokens, and its special tokens. A token's
    rank is both its id and its merge priority. A pretoken that is a token is
    that token. Any other starts as its bytes, and the adjacent pair whose
    joined bytes form the token of lowest rank is joined, the leftmost of equal
    pairs first, until no adjacent pair joins into a token.

    A token of one byte or more is {e built} when this rule, run on its own
    bytes, ends with the token. Wherever the rule joins two parts into a token,
    they are the two parts it joins last on that token's bytes, so the rule is
    a {!bpe} with one merge per built token, that pair, in rank order.

    A {e rank file} holds one token per line, read as tiktoken reads a
    well-formed one: a carriage return or a line feed ends a line, an empty line
    is skipped, and a line is two fields between ASCII white space, the base64
    of the token's bytes with its padding and the rank in decimal. A field [=]
    alone is the empty token. Ranks may leave gaps.

    The pattern is in the dialect of {!Pre_tokenizer.split_regex}. Its matches,
    leftmost first without overlap, are the pretokens, and the text no match
    covers is dropped, bytes that are no character included.
    {!Pre_tokenizer.split_regex} takes neither possessive quantifiers nor [$],
    so an encoding whose pattern uses them is passed an equivalent pattern.
    cl100k_base's is
    {[
      String.concat "|"
        [ {|'(?i:[sdmt]|ll|ve|re)|}; {|[^\r\n\p{L}\p{N}]?\p{L}+|};
          {|\p{N}{1,3}|}; {| ?[^\s\p{L}\p{N}]+[\r\n]*|}; {|\s+(?![\s\S])|};
          {|\s*[\r\n]|}; {|\s+(?!\S)|}; {|\s|} ]
    ]}
    and o200k_base's and Whisper's are read as they are written. Characters are
    classified by Unicode 18.0.0 ({!Normalizer.replace_regex}), so text holding
    a character assigned after the Unicode version of tiktoken's regular
    expression engine can get other ids than tiktoken gives. *)

val of_tiktoken :
  ?specials:(string * int) list ->
  pattern:string ->
  string ->
  (t, string) result
(** [of_tiktoken ~pattern ranks] is the tokenizer of the tiktoken encoding of
    the rank file [ranks] and the pattern [pattern]. It has the merges of the
    built tokens, and emits a pretoken found in its vocabulary whole when some
    token is not built.

    [specials] are special tokens with their ids, each in the vocabulary at its
    id and an added token ({!type-added_token}). Default: [[]]. Special tokens
    numbered from the end of the ranks are added with {!add_tokens}.
    {!vocab_size} counts tokens, which is less than the largest id plus one
    when ids leave gaps.

    On valid UTF-8 the tokenizer gives tiktoken's ids for the same ranks,
    pattern and specials. {!encode} with [~match_special_tokens:false] is
    tiktoken's [encode_ordinary], and with the default its [encode] with every
    special allowed, the longer of two specials matched where one is a prefix
    of the other. {!decode} is tiktoken's [decode] with [errors="replace"] on
    ids that name tokens.

    Errors if a line is not a token and a rank, if two lines hold the same
    bytes or the same rank, if a rank is 2{^ 31} or more, if a byte has no
    rank, or if a special token is a token, has a token's id, or is given while
    some token is not built. Raises [Invalid_argument] if [pattern] is invalid,
    unsupported or can match the empty text, or if a special token is empty, is
    changed by {!Pre_tokenizer.val-byte_level_decode}, has an id outside
    \[0, 2{^ 31}), or shares its text or its id with another. *)

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a tokenizer for inspection. Shows algorithm type, vocabulary
    size ({!vocab_size}), and configured pipeline stages. *)

(**/**)

val kernels : bool
(** [kernels] is [true] iff encoding can run the C kernels: native code on a
    little-endian machine. For tests. *)

val hold_cache : t -> (unit -> 'a) -> 'a
(** [hold_cache t f] is [f ()] while one of the encoder states of [t]'s BPE,
    WordPiece or Unigram model is held, so that what is encoded with [t]
    meanwhile does without it. When the model already holds
    [Domain.recommended_domain_count ()] states, [f] runs while a spare state is
    in use, which the model does not keep, as for an encoding. A state that [f]
    raises through is dropped. For tests. *)

val cache_set : string -> sets:int -> int
(** [cache_set key ~sets] is the back set, among [sets], a power of two, that a
    pretoken cache stores the pretoken [key] of 1 to 15 bytes in. Every byte of
    [key] takes part, so keys of 2 bytes or more that share all but their last
    bytes still spread over the sets. For tests. *)

val cache_key : string -> int64 * int64
(** [cache_key key] is the two key words a pretoken cache keys the pretoken
    [key] of 1 to 15 bytes by. For tests. *)

val cache_hashes : int64 -> int64 -> int * int
(** [cache_hashes k0 k1] is the hash a pretoken cache indexes the key words [k0]
    and [k1] by, as OCaml computes it and as the C kernels do. For tests. *)

val batch_document_hook : (unit -> unit) Atomic.t
(** [batch_document_hook] is called by a chunk of {!encode_batch_ids}, when the
    post-processor only wraps a sequence, before each document or piece of a cut
    document it encodes into its ids buffer, once the chunk holds the buffer and
    has read where that text starts in it. For tests. *)

val run_chunks : ?exec:exec -> int array -> (int -> unit) -> unit
(** [run_chunks ?exec sizes work] is [work c] for every chunk [c] of the [sizes]
    in bytes, through [exec] as the batch encoders spread their work, raising
    what a chunk raised. With no [exec] or a single chunk, [work] runs on the
    calling domain and [exec] is not called. For tests. *)

module Float_text : sig
  type format = { precision : int; emin : int }

  val binary64 : format
  val binary32 : format
  val shortest : format -> float -> string * int

  (** [shortest f x] is the decimal [(d, e)], [d] × 10{^ [e]}, of the fewest
      digits that reads as [|x|] in [f] to nearest, ties to even, the nearest to
      [|x|] of those, and the one whose last digit is even at a tie; zero is
      [("0", 0)]. Raises [Invalid_argument] if [f]'s precision is outside 1 to
      53 or [x] is not a value of [f]. *)
end
(** [Float_text] is the decimal digits of floats that {!save_pretrained} writes
    numbers with. For tests. *)
