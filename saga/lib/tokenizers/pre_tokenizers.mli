(** Pre-tokenization for text processing pipelines.

    Pre-tokenizers are the first stage in text tokenization pipelines. They
    split raw text into smaller pieces before vocabulary-based tokenization
    (like BPE or WordPiece) is applied. This splitting is crucial for handling
    different languages, punctuation, and special formatting.

    {1 Overview}

    Pre-tokenization serves several purposes:
    - Language-aware splitting (whitespace varies by language)
    - Punctuation handling (separate or merge with adjacent text)
    - Character encoding normalization (Unicode, byte-level representations)
    - Format preservation (maintaining character offsets for downstream tasks)

    The pre-tokenization process takes raw text and returns a list of (piece,
    (start, end)) tuples, where each piece is a substring and the offsets
    indicate its position in the original text.

    {1 Common Patterns}

    Most tokenizers follow this pattern:
    + {b Normalization}: Convert text to canonical form (lowercase, accent
      removal)
    + {b Pre-tokenization}: Split into words/subwords using this module
    + {b Tokenization}: Apply vocabulary-based encoding (BPE, WordPiece, etc.)
    + {b Post-processing}: Add special tokens, create attention masks

    {1 Usage Examples}

    Basic word-level splitting:
    {[
      let pre_tokenizer = Pre_tokenizers.whitespace in
      let pieces = pre_tokenizer "Hello world! How are you?" in
      (* Result: [("Hello", (0, 5)); ("world!", (6, 12)); ("How", (13, 16));
                   ("are", (17, 20)); ("you?", (21, 25))] *)
    ]}

    Byte-level processing for robust handling:
    {[
      let pre_tokenizer = Pre_tokenizers.byte_level ~add_prefix_space:true ~use_regex:true () in
      let pieces = pre_tokenizer "Hello 🤖 world!" in
      (* Handles Unicode robustly, converts to byte representation *)
    ]}

    Chaining multiple pre-tokenizers:
    {[
      let chain = Pre_tokenizers.sequence [
        Pre_tokenizers.punctuation ~behavior:`Isolated ();
        Pre_tokenizers.whitespace;
        Pre_tokenizers.digits ~individual_digits:false ();
      ] in
      let pieces = chain "Hello, world! The year is 2024." in
      (* Applies punctuation splitting, then whitespace, then digit handling *)
    ]}

    {1 Character Offset Preservation}

    All pre-tokenizers maintain character offsets, crucial for:
    - Highlighting tokens in original text
    - Named entity recognition alignment
    - Question answering span extraction
    - Error reporting and debugging

    {[
      let text = "The quick brown fox jumps" in
      let pieces = Pre_tokenizers.whitespace text in
      List.iter
        (fun (piece, (start, end_)) ->
          Printf.printf "'%s' at positions %d-%d: '%s'\n" piece start end_
            (String.sub text start (end_ - start)))
        pieces
      (* Verifies that substrings match original text positions *)
    ]}

    {1 Language-Specific Considerations}

    Different pre-tokenizers handle various languages:
    - {!whitespace}: Good for space-separated languages (English, Spanish)
    - {!bert}: Handles CJK characters and punctuation (Chinese, Japanese,
      Korean)
    - {!byte_level}: Universal but loses some linguistic structure
    - {!unicode_scripts}: Script-aware splitting for multilingual text

    {1 Performance Notes}

    Pre-tokenization can be a bottleneck in tokenization pipelines:
    - Simple splitters like {!whitespace_split} are fastest
    - Regex-based splitters like {!byte_level} with use_regex are slower but
      more accurate
    - {!sequence} applies all pre-tokenizers, increasing cost linearly
    - Consider caching results for repeated text processing *)

type t = string -> (string * (int * int)) list
(** Pre-tokenizer function type.

    Takes a string and returns a list of (piece, (start_offset, end_offset))
    tuples. Each piece is a substring of the input, and the offsets indicate its
    position in the original text. Offsets are character-based (not byte-based).

    Invariants:
    - Pieces, when concatenated, should reconstruct the original text (possibly
      with some normalization)
    - Offsets must be valid: 0 <= start_offset < end_offset <= String.length
      text
    - Offsets must be non-overlapping and in ascending order *)

val bert : t
(** [bert text] applies BERT-style pre-tokenization.

    Splits on whitespace and separates punctuation. Designed for BERT-family
    models. Handles CJK (Chinese, Japanese, Korean) characters by treating each
    as a separate token.

    Behavior:
    - Whitespace separated into tokens
    - Punctuation characters isolated (each punct char becomes separate token)
    - CJK characters split individually
    - Preserves case and accents

    {[
      let pieces = Pre_tokenizers.bert "Hello, world! 你好" in
      (* Result approximately:
         [("Hello", (0, 5)); (",", (5, 6)); (" ", (6, 7)); ("world", (7, 12));
          ("!", (12, 13)); (" ", (13, 14)); ("你", (14, 15)); ("好", (15, 16))] *)
    ]} *)

val byte_level : ?add_prefix_space:bool -> ?use_regex:bool -> unit -> t
(** [byte_level ?add_prefix_space ?use_regex ()] creates a byte-level
    pre-tokenizer.

    Used by GPT-2 style models. Converts text to byte representation and applies
    regex-based splitting. Handles any Unicode text robustly by treating
    everything as byte sequences.

    @param add_prefix_space
      If true, adds a space at the beginning if text doesn't start with
      whitespace. Default: false.
    @param use_regex
      If true, uses GPT-2's regex pattern for splitting. If false, uses simpler
      splitting. Default: true.

    The GPT-2 regex pattern handles:
    - Apostrophe contractions ("don't", "I'm")
    - Letter sequences
    - Number sequences
    - Whitespace sequences
    - Individual characters as fallback

    {[
      let pre_tokenizer = Pre_tokenizers.byte_level ~add_prefix_space:true ~use_regex:true () in
      let pieces = pre_tokenizer "Hello world!" in
      (* Result handles Unicode robustly, may add prefix space *)

      let pre_tokenizer2 = Pre_tokenizers.byte_level ~add_prefix_space:false ~use_regex:false () in
      let pieces2 = pre_tokenizer2 "café" in
      (* Simpler splitting without regex complexity *)
    ]} *)

val whitespace : t
(** [whitespace text] splits on whitespace using pattern \\w+|[^\\w\\s]+.

    Groups word characters (letters, digits, underscore) together and groups
    non-word, non-space characters together. Whitespace is used as delimiter but
    not included in output pieces.

    Pattern behavior:
    - \\w+: One or more word characters (letters, digits, _)
    - [^\\w\\s]+: One or more characters that are neither word chars nor
      whitespace

    {[
      let pieces = Pre_tokenizers.whitespace "Hello, world! How's it going?" in
      (* Result approximately:
         [("Hello", (0, 5)); (",", (5, 6)); ("world", (7, 12)); ("!", (12, 13));
          ("How", (14, 17)); ("'", (17, 18)); ("s", (18, 19)); ("it", (20, 22));
          ("going", (23, 28)); ("?", (28, 29))] *)
    ]} *)

val whitespace_split : t
(** [whitespace_split text] performs simple whitespace splitting.

    Splits text on any whitespace characters and removes the whitespace. This is
    the simplest and fastest pre-tokenizer, equivalent to String.split_on_char.

    {[
      let pieces = Pre_tokenizers.whitespace_split "Hello   world!\tHow\nare you?" in
      (* Result approximately:
         [("Hello", (0, 5)); ("world!", (8, 14)); ("How", (15, 18));
          ("are", (19, 22)); ("you?", (23, 27))] *)
    ]} *)

type behavior =
  [ `Isolated  (** Keep delimiter as separate token *)
  | `Removed  (** Remove delimiter completely *)
  | `Merged_with_previous  (** Merge delimiter with previous token *)
  | `Merged_with_next  (** Merge delimiter with next token *)
  | `Contiguous  (** Group consecutive delimiters together *) ]
(** Delimiter handling behavior for splitting operations.

    Controls what happens to delimiter characters when splitting text:

    - [`Isolated]: Delimiter becomes its own token (e.g., "hello,world" →
      ["hello"; ","; "world"])
    - [`Removed]: Delimiter is discarded (e.g., "hello,world" →
      ["hello"; "world"])
    - [`Merged_with_previous]: Delimiter attached to preceding token (e.g.,
      "hello,world" → ["hello,"; "world"])
    - [`Merged_with_next]: Delimiter attached to following token (e.g.,
      "hello,world" → ["hello"; ",world"])
    - [`Contiguous]: Multiple consecutive delimiters grouped together (e.g.,
      "hello,,world" → ["hello"; ",,"; "world"]) *)

val punctuation : ?behavior:behavior -> unit -> t
(** [punctuation ?behavior ()] creates a punctuation-aware pre-tokenizer.

    Splits text by separating punctuation characters from alphanumeric content.
    Punctuation includes standard ASCII punctuation and Unicode punctuation
    categories.

    @param behavior How to handle punctuation characters. Default: [`Isolated].

    {[
      let pre_tokenizer = Pre_tokenizers.punctuation ~behavior:`Isolated () in
      let pieces = pre_tokenizer "Hello, world! How are you?" in
      (* Result with `Isolated:
         [("Hello", (0, 5)); (",", (5, 6)); (" world", (6, 12)); ("!", (12, 13));
          (" How are you", (13, 26)); ("?", (26, 27))] *)

      let pre_tokenizer2 = Pre_tokenizers.punctuation ~behavior:`Merged_with_previous () in
      let pieces2 = pre_tokenizer2 "Don't stop" in
      (* Result with `Merged_with_previous:
         [("Don'", (0, 4)); ("t stop", (4, 10))] *)
    ]} *)

val split : pattern:string -> behavior:behavior -> ?invert:bool -> unit -> t
(** [split ~pattern ~behavior ?invert ()] creates a pattern-based splitter.

    Splits text based on a specific string pattern. More flexible than
    punctuation splitting as it allows custom patterns and inversion.

    @param pattern String pattern to split on (literal string, not regex).
    @param behavior How to handle the pattern when found.
    @param invert
      If true, split on everything except the pattern. Default: false.

    {[
      (* Split on commas, keeping them *)
      let pre_tokenizer = Pre_tokenizers.split ~pattern:"," ~behavior:`Isolated () in
      let pieces = pre_tokenizer "apple,banana,cherry" in
      (* Result: [("apple", (0, 5)); (",", (5, 6)); ("banana", (6, 12));
                  (",", (12, 13)); ("cherry", (13, 19))] *)

      (* Split on spaces, removing them *)
      let pre_tokenizer2 = Pre_tokenizers.split ~pattern:" " ~behavior:`Removed () in
      let pieces2 = pre_tokenizer2 "hello world test" in
      (* Result: [("hello", (0, 5)); ("world", (6, 11)); ("test", (12, 16))] *)

      (* Invert: split on everything except letters *)
      let pre_tokenizer3 = Pre_tokenizers.split ~pattern:"abc" ~behavior:`Removed ~invert:true () in
      let pieces3 = pre_tokenizer3 "ab1c2de3f" in
      (* Splits on non-"abc" characters (numbers), removing them *)
    ]} *)

val char_delimiter_split : char -> t
(** [char_delimiter_split delimiter] splits on a specific character delimiter.

    Splits text whenever the specified character is encountered, removing the
    delimiter from the output. Equivalent to String.split_on_char but maintains
    offsets.

    @param delimiter Character to split on (removed from output).

    {[
      let pre_tokenizer = Pre_tokenizers.char_delimiter_split '|' in
      let pieces = pre_tokenizer "apple|banana|cherry" in
      (* Result: [("apple", (0, 5)); ("banana", (6, 12)); ("cherry", (13, 19))] *)

      let pre_tokenizer2 = Pre_tokenizers.char_delimiter_split '\n' in
      let pieces2 = pre_tokenizer2 "line1\nline2\nline3" in
      (* Result: [("line1", (0, 5)); ("line2", (6, 11)); ("line3", (12, 17))] *)
    ]} *)

val digits : ?individual_digits:bool -> unit -> t
(** [digits ?individual_digits ()] creates a digit-aware pre-tokenizer.

    Handles numeric content in text, with configurable granularity. Useful for
    mathematical text, data parsing, or models that need fine-grained number
    handling.

    @param individual_digits
      If true, each digit becomes a separate token. If false, consecutive digits
      are grouped. Default: false.

    {[
      let pre_tokenizer = Pre_tokenizers.digits ~individual_digits:false () in
      let pieces = pre_tokenizer "I have 123 apples and 45 oranges" in
      (* Result with grouped digits:
         [("I have ", (0, 7)); ("123", (7, 10)); (" apples and ", (10, 22));
          ("45", (22, 24)); (" oranges", (24, 32))] *)

      let pre_tokenizer2 = Pre_tokenizers.digits ~individual_digits:true () in
      let pieces2 = pre_tokenizer2 "Price: $42.99" in
      (* Result with individual digits:
         [("Price: $", (0, 8)); ("4", (8, 9)); ("2", (9, 10)); (".", (10, 11));
          ("9", (11, 12)); ("9", (12, 13))] *)
    ]} *)

type prepend_scheme =
  [ `First  (** Only prepend to first piece *)
  | `Never  (** Never prepend *)
  | `Always  (** Always prepend if not starting with space *) ]
(** Prepend scheme controlling when to add replacement character.

    Used by metaspace pre-tokenizer to control prefix behavior:

    - [`First]: Add replacement only to the very first piece of text
    - [`Never]: Never add replacement as prefix
    - [`Always]: Add replacement to any piece that doesn't already start with
      whitespace

    This is important for sentence-level tokenization where you want consistent
    handling of word boundaries across different contexts. *)

val metaspace :
  ?replacement:string ->
  ?prepend_scheme:prepend_scheme ->
  ?split:bool ->
  ?is_first:bool ->
  unit ->
  t
(** [metaspace ?replacement ?prepend_scheme ?split ?is_first ()] creates a
    metaspace pre-tokenizer.

    Used by models like SentencePiece that represent spaces as special
    characters. Replaces whitespace with a visible replacement character
    (typically "▁") to make word boundaries explicit in the token sequence.

    @param replacement Character to replace spaces with. Default: "▁" (U+2581).
    @param prepend_scheme
      When to prepend replacement to pieces. Default: [`Always].
    @param split Whether to split on the replacement character. Default: true.
    @param is_first
      Whether this is the first piece in a sequence (affects [`First] scheme).
      Default: true.

    Behavior: 1. Replace all whitespace with replacement character 2. Apply
    prepend scheme to add replacement prefix where needed 3. Optionally split on
    replacement character boundaries

    {[
      let pre_tokenizer = Pre_tokenizers.metaspace () in
      let pieces = pre_tokenizer "Hello world" in
      (* Result with default settings:
         [("▁Hello", (0, 5)); ("▁world", (6, 11))] *)

      let pre_tokenizer2 = Pre_tokenizers.metaspace
        ~replacement:"_" ~prepend_scheme:`Never ~split:false () in
      let pieces2 = pre_tokenizer2 "Hello world test" in
      (* Result with custom settings:
         [("Hello_world_test", (0, 17))] *)

      let pre_tokenizer3 = Pre_tokenizers.metaspace ~prepend_scheme:`First () in
      let pieces3 = pre_tokenizer3 "First piece" in
      let pieces4 = pre_tokenizer3 "second piece" in
      (* First call: [("▁First", (0, 5)); ("piece", (6, 11))]
         Second call: [("second", (0, 6)); ("piece", (7, 12))] *)
    ]} *)

val sequence : t list -> t
(** [sequence pre_tokenizers] applies multiple pre-tokenizers in sequence.

    Each pre-tokenizer is applied to the output of the previous one. Useful for
    building complex tokenization pipelines by composing simpler parts.

    @param pre_tokenizers List of pre-tokenizers to apply in order.

    The function applies tokenizers left-to-right: 1. Apply first pre-tokenizer
    to input text 2. Apply second pre-tokenizer to each piece from step 1 3.
    Continue until all pre-tokenizers applied 4. Flatten results and maintain
    offset correctness

    {[
      let pipeline = Pre_tokenizers.sequence [
        Pre_tokenizers.punctuation ~behavior:`Isolated ();
        Pre_tokenizers.whitespace_split;
        Pre_tokenizers.digits ~individual_digits:true ();
      ] in
      let pieces = pipeline "Hello, world! Price: $123" in
      (* Step 1: punctuation -> ["Hello"; ","; " world"; "!"; " Price: $123"]
         Step 2: whitespace -> ["Hello"; ","; "world"; "!"; "Price:"; "$123"]
         Step 3: digits -> ["Hello"; ","; "world"; "!"; "Price:"; "$"; "1"; "2"; "3"] *)
    ]} *)

val fixed_length : length:int -> t
(** [fixed_length ~length] splits text into fixed-length character chunks.

    Useful for character-level models or when you need uniform token lengths.
    The last chunk may be shorter if text length is not divisible by chunk
    length.

    @param length Number of characters per chunk. Must be positive.

    {[
      let pre_tokenizer = Pre_tokenizers.fixed_length ~length:3 in
      let pieces = pre_tokenizer "Hello world!" in
      (* Result: [("Hel", (0, 3)); ("lo ", (3, 6)); ("wor", (6, 9));
                  ("ld!", (9, 12))] *)

      let pre_tokenizer2 = Pre_tokenizers.fixed_length ~length:1 in
      let pieces2 = pre_tokenizer2 "Hi!" in
      (* Result: [("H", (0, 1)); ("i", (1, 2)); ("!", (2, 3))] *)
    ]} *)

val unicode_scripts : t
(** [unicode_scripts text] splits text on Unicode script boundaries.

    Separates text when the Unicode script changes (e.g., Latin to Cyrillic,
    Latin to Arabic, etc.). Useful for multilingual text where you want to
    separate different writing systems.

    Unicode scripts include: Latin, Cyrillic, Arabic, Chinese (Han), Japanese
    (Hiragana/Katakana), Korean (Hangul), Thai, Hebrew, Greek, and many others.

    {[
      let pieces = Pre_tokenizers.unicode_scripts "Hello мир world 中国" in
      (* Splits between Latin, Cyrillic, and Chinese scripts:
         [("Hello ", (0, 6)); ("мир", (6, 9)); (" world ", (9, 16)); ("中国", (16, 18))] *)

      let pieces2 = Pre_tokenizers.unicode_scripts "café καφέ" in
      (* Splits between Latin and Greek:
         [("café ", (0, 5)); ("καφέ", (5, 9))] *)
    ]} *)

(** {1 Internal Helpers - exposed for testing} *)

val split_gpt2_pattern : string -> string list
(** [split_gpt2_pattern text] splits text using GPT-2's regex pattern.

    Internal helper function that implements the exact regex pattern used by
    GPT-2 for tokenization. Exposed primarily for testing and debugging
    purposes.

    The pattern handles:
    - Apostrophe contractions (don't, I'm, etc.)
    - Letter sequences (consecutive alphabetic characters)
    - Number sequences (consecutive digits)
    - Whitespace sequences (consecutive whitespace)
    - Individual characters as fallback

    @return List of string pieces (no offset information).

    {[
      let pieces = Pre_tokenizers.split_gpt2_pattern "I'm learning OCaml!" in
      (* Result approximately: ["I"; "'m"; " learning"; " OCaml"; "!"] *)
    ]} *)

val byte_level_decode : string -> string
(** [byte_level_decode encoded] decodes byte-level encoded text.

    Internal helper that reverses the byte-level encoding applied by
    {!byte_level}. Converts the special Unicode characters back to their
    original byte values. Exposed primarily for testing and debugging.

    @param encoded Text that has been byte-level encoded.
    @return Original text before byte-level encoding.

    {[
      let original = "café" in
      let tokenizer = Pre_tokenizers.byte_level () in
      let pieces = tokenizer original in
      let encoded_piece = fst (List.hd pieces) in
      let decoded = Pre_tokenizers.byte_level_decode encoded_piece in
      assert (decoded = original || (* handle encoding differences *))
    ]} *)
