# Unicode Processing

Saga provides comprehensive Unicode support for text processing in machine learning applications.

## Overview

Modern NLP requires proper handling of multilingual text, emoji, special characters, and various writing systems. Saga's Unicode module provides battle-tested utilities for:

- Text normalization and cleaning
- Character classification
- Word boundary detection  
- Grapheme clustering
- CJK text handling

## Text Normalization

### Basic Normalization

Clean and normalize text for consistent tokenization:

```ocaml
open Saga

(* Lowercase and collapse whitespace *)
let normalized = normalize 
  ~lowercase:true 
  ~collapse_whitespace:true 
  "  Hello   WORLD!  "
(* "hello world!" *)

(* Strip accents *)
let normalized = normalize 
  ~strip_accents:true 
  "Café naïve résumé"
(* "Cafe naive resume" *)

(* All options *)
let normalized = normalize
  ~lowercase:true
  ~strip_accents:true  
  ~collapse_whitespace:true
  "  Héllo   WÖRLD!  "
(* "hello world!" *)
```

### Unicode Module

For fine-grained control, use the Unicode module directly:

```ocaml
open Saga.Unicode

(* Case folding *)
let folded = case_fold "Hello WORLD ÄÖÜ"
(* Proper Unicode case folding *)

(* Clean text *)
let cleaned = clean_text 
  ~remove_control:true 
  ~normalize_whitespace:true 
  "Hello\x00\tworld"
(* "Hello world" *)

(* Remove emoji *)
let no_emoji = remove_emoji "Hello 👋 World 🌍!"
(* "Hello  World !" *)
```

## Character Classification

Classify characters for tokenization decisions:

```ocaml
open Saga.Unicode

(* Basic categories *)
let is_letter = categorize_char (Uchar.of_char 'A') = Letter
let is_digit = categorize_char (Uchar.of_char '5') = Number
let is_punct = categorize_char (Uchar.of_char ',') = Punctuation
let is_space = categorize_char (Uchar.of_char ' ') = Whitespace

(* CJK detection *)
let is_chinese = is_cjk (Uchar.of_int 0x4E00)  (* 一 *)
let is_hiragana = is_cjk (Uchar.of_int 0x3042) (* あ *)
let is_katakana = is_cjk (Uchar.of_int 0x30A2) (* ア *)
let is_hangul = is_cjk (Uchar.of_int 0xAC00)   (* 가 *)
```

## Word Splitting

Unicode-aware word boundary detection:

```ocaml
open Saga.Unicode

(* Basic word splitting *)
let words = split_words "Hello, world! How are you?"
(* ["Hello"; "world"; "How"; "are"; "you"] *)

(* Numbers stay with words *)
let words = split_words "test123 456test"
(* ["test123"; "456test"] *)

(* CJK characters split individually *)
let words = split_words "Hello世界"
(* ["Hello"; "世"; "界"] *)

(* Mixed scripts *)
let words = split_words "café résumé 你好"
(* ["café"; "résumé"; "你"; "好"] *)
```

## Grapheme Counting

Count visual characters, not bytes:

```ocaml
open Saga.Unicode

(* ASCII *)
let count = grapheme_count "Hello!"
(* 6 *)

(* Emoji count as single graphemes *)
let count = grapheme_count "Hi 👋"
(* 4 *)

(* Combined characters *)
let count = grapheme_count "é"  (* e + combining acute *)
(* 1 *)
```

## UTF-8 Validation

Ensure text is valid UTF-8:

```ocaml
open Saga.Unicode

(* Valid UTF-8 *)
let valid = is_valid_utf8 "Hello 世界"
(* true *)

(* Invalid sequences *)
let invalid_text = String.make 1 '\xFF' ^ String.make 1 '\xFE'
let valid = is_valid_utf8 invalid_text
(* false *)
```

## Best Practices

### Preprocessing Pipeline

Create a consistent preprocessing pipeline:

```ocaml
let preprocess text =
  text
  |> Saga.normalize 
      ~lowercase:true 
      ~strip_accents:true
      ~collapse_whitespace:true
  |> Saga.Unicode.remove_emoji
  |> Saga.Unicode.clean_text 
      ~remove_control:true
```

### Language-Specific Handling

Different languages need different treatment:

```ocaml
(* English: aggressive normalization *)
let preprocess_english text =
  Saga.normalize 
    ~lowercase:true 
    ~strip_accents:true 
    text

(* Multilingual: preserve more information *)
let preprocess_multilingual text =
  Saga.normalize 
    ~lowercase:false  (* Preserve case *)
    ~strip_accents:false  (* Keep diacritics *)
    ~collapse_whitespace:true 
    text

(* CJK: minimal processing *)
let preprocess_cjk text =
  Saga.Unicode.clean_text 
    ~normalize_whitespace:true 
    text
```

### Performance Considerations

1. **Cache normalized text**: Normalization can be expensive for large texts
2. **Batch processing**: Process multiple texts together when possible
3. **Early validation**: Check UTF-8 validity before processing

```ocaml
(* Validate and cache *)
let process_texts texts =
  texts
  |> List.filter Saga.Unicode.is_valid_utf8
  |> List.map (fun text ->
      let normalized = Saga.normalize text in
      (text, normalized))
```

## Common Issues

### Malformed UTF-8

Handle invalid UTF-8 gracefully:

```ocaml
let safe_tokenize text =
  if Saga.Unicode.is_valid_utf8 text then
    Saga.tokenize text
  else
    [] (* Or handle error appropriately *)
```

### Mixed Scripts

Be aware of script boundaries:

```ocaml
(* Mixed Latin and CJK *)
let tokens = Saga.tokenize "Hello世界world"
(* Might tokenize as ["Hello世界world"] or ["Hello"; "世界"; "world"] 
   depending on tokenizer *)

(* Use Unicode-aware splitting for better results *)
let words = Saga.Unicode.split_words "Hello世界world"
(* ["Hello"; "世"; "界"; "world"] *)
```

### Emoji and Special Characters

Decide how to handle emoji based on your use case:

```ocaml
(* Remove emoji for formal text *)
let formal = Saga.Unicode.remove_emoji text

(* Keep emoji for social media *)
let social = Saga.Unicode.clean_text 
  ~remove_control:true  (* Remove control chars *)
  text  (* Keep emoji *)
```

## Integration with Tokenizers

Unicode processing integrates seamlessly with tokenizers:

```ocaml
(* Normalize before tokenization *)
let tokens = 
  text
  |> Saga.normalize ~lowercase:true
  |> Saga.tokenize

(* Use with BPE *)
let bpe_tokens =
  text
  |> Saga.normalize ~lowercase:true
  |> Saga.Bpe.tokenize bpe

(* Use with WordPiece *)
let wp_tokens =
  text
  |> Saga.normalize ~lowercase:true
  |> Saga.Wordpiece.tokenize wordpiece
```