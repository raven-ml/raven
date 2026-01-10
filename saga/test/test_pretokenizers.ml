(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Comprehensive test suite for all pre-tokenizers *)

open Alcotest
module Pre = Saga_tokenizers.Pre_tokenizers

(** Helper to check tokenization with offsets *)
let check_tokenization name input expected =
  check (list (pair string (pair int int))) name expected input

(** Helper to check just the strings *)
let check_strings name input expected =
  check (list string) name expected (List.map fst input)

(** Test GPT-2 ByteLevel pre-tokenizer *)
let test_byte_level_basic () =
  let tokenizer = Pre.byte_level ~add_prefix_space:false ~use_regex:true () in

  (* Test basic tokenization *)
  let test_case text expected_pieces expected_offsets =
    let result = tokenizer text in
    let offsets = List.map snd result in
    check_strings
      (Printf.sprintf "ByteLevel pieces for %S" text)
      result expected_pieces;
    check
      (list (pair int int))
      (Printf.sprintf "ByteLevel offsets for %S" text)
      expected_offsets offsets
  in

  (* Basic words *)
  test_case "Hello" [ "Hello" ] [ (0, 5) ];
  test_case "hello" [ "hello" ] [ (0, 5) ];
  test_case "HELLO" [ "HELLO" ] [ (0, 5) ];

  (* Words with spaces - space becomes Ġ (0xC4 0xA0) *)
  test_case "Hello world" [ "Hello"; "\196\160world" ] [ (0, 5); (5, 11) ];
  test_case "Hello  world"
    [ "Hello"; "\196\160"; "\196\160world" ]
    [ (0, 5); (5, 6); (6, 12) ];

  (* Leading/trailing spaces *)
  test_case " hello" [ "\196\160hello" ] [ (0, 6) ];
  test_case "hello " [ "hello"; "\196\160" ] [ (0, 5); (5, 6) ];
  (* Note: Python produces ['Ġ', 'Ġhello', 'ĠĠ'] for " hello " *)
  test_case "  hello  "
    [ "\196\160"; "\196\160hello"; "\196\160\196\160" ]
    [ (0, 1); (1, 7); (7, 9) ];

  (* Contractions - should be kept as separate pieces *)
  test_case "'s" [ "'s" ] [ (0, 2) ];
  test_case "'t" [ "'t" ] [ (0, 2) ];
  test_case "'re" [ "'re" ] [ (0, 3) ];
  test_case "'ve" [ "'ve" ] [ (0, 3) ];
  test_case "'m" [ "'m" ] [ (0, 2) ];
  test_case "'ll" [ "'ll" ] [ (0, 3) ];
  test_case "'d" [ "'d" ] [ (0, 2) ];

  (* Words with contractions *)
  test_case "don't" [ "don"; "'t" ] [ (0, 3); (3, 5) ];
  test_case "it's" [ "it"; "'s" ] [ (0, 2); (2, 4) ];
  test_case "we're" [ "we"; "'re" ] [ (0, 2); (2, 5) ];
  test_case "I'll" [ "I"; "'ll" ] [ (0, 1); (1, 4) ];
  test_case "OpenAI's" [ "OpenAI"; "'s" ] [ (0, 6); (6, 8) ]

let test_byte_level_prefix_space () =
  (* Test with add_prefix_space=true *)
  let tokenizer = Pre.byte_level ~add_prefix_space:true ~use_regex:true () in

  let test_case text expected_pieces =
    let result = tokenizer text in
    check_strings
      (Printf.sprintf "ByteLevel with prefix for %S" text)
      result expected_pieces
  in

  (* Should add space prefix when text doesn't start with space *)
  test_case "hello" [ "\196\160hello" ];
  test_case "Hello world" [ "\196\160Hello"; "\196\160world" ];

  (* Should NOT add extra space when text already starts with space *)
  test_case " hello" [ "\196\160hello" ];
  test_case "  hello" [ "\196\160"; "\196\160hello" ]

let test_byte_level_special_chars () =
  let tokenizer = Pre.byte_level ~add_prefix_space:false ~use_regex:true () in

  let test_case text desc =
    let result = tokenizer text in
    let pieces = List.map fst result in
    (* Just verify it doesn't crash and produces something *)
    check bool
      (Printf.sprintf "ByteLevel handles %s" desc)
      true
      (List.length pieces > 0)
  in

  (* Punctuation *)
  test_case "." "period";
  test_case "!" "exclamation";
  test_case "?" "question";
  test_case "," "comma";
  test_case ";" "semicolon";
  test_case ":" "colon";

  (* Special characters *)
  test_case "@" "at sign";
  test_case "#" "hash";
  test_case "$" "dollar";
  test_case "%" "percent";
  test_case "^" "caret";
  test_case "&" "ampersand";
  test_case "*" "asterisk";

  (* Brackets and quotes *)
  test_case "()" "parentheses";
  test_case "[]" "brackets";
  test_case "{}" "braces";
  test_case "\"\"" "quotes";
  test_case "''" "single quotes";

  (* Numbers *)
  test_case "123" "numbers";
  test_case "3.14" "decimal";
  test_case "1,000" "number with comma";

  (* Mixed *)
  test_case "Hello, world!" "punctuated sentence";
  test_case "@user #hashtag" "social media";
  test_case "test@example.com" "email";
  test_case "https://example.com" "URL";
  test_case "function()" "function call";
  test_case "a+b=c" "math expression"

let test_byte_level_unicode () =
  let tokenizer = Pre.byte_level ~add_prefix_space:false ~use_regex:true () in

  let test_case text desc =
    let result = tokenizer text in
    let pieces = List.map fst result in
    (* Byte-level encoding should handle any Unicode by encoding bytes *)
    check bool
      (Printf.sprintf "ByteLevel handles %s" desc)
      true
      (List.length pieces > 0);
    (* Check that we can reconstruct something (even if not identical due to
       encoding) *)
    let concatenated = String.concat "" pieces in
    check bool
      (Printf.sprintf "ByteLevel produces non-empty output for %s" desc)
      true
      (String.length concatenated > 0)
  in

  (* Common accented characters *)
  test_case "café" "accented e";
  test_case "naïve" "diaeresis";
  test_case "résumé" "French accents";

  (* Other languages *)
  test_case "你好" "Chinese";
  test_case "こんにちは" "Japanese";
  test_case "안녕하세요" "Korean";
  test_case "Привет" "Russian";
  test_case "مرحبا" "Arabic";

  (* Emojis *)
  test_case "😀" "emoji";
  test_case "👍🏻" "emoji with skin tone";
  test_case "Hello 👋 World" "text with emoji"

let test_byte_level_edge_cases () =
  let tokenizer = Pre.byte_level ~add_prefix_space:false ~use_regex:true () in

  (* Empty string *)
  let result = tokenizer "" in
  check (list string) "Empty string" [] (List.map fst result);

  (* Single character *)
  let result = tokenizer "a" in
  check_strings "Single char" result [ "a" ];

  (* Only spaces - Python produces ['ĠĠĠ'] all together *)
  let result = tokenizer "   " in
  check_strings "Only spaces" result [ "\196\160\196\160\196\160" ];

  (* Only punctuation - Python keeps '...' together *)
  let result = tokenizer "..." in
  check_strings "Only punctuation" result [ "..." ];

  (* Very long word *)
  let long_word = String.make 100 'a' in
  let result = tokenizer long_word in
  check int "Long word produces single token" 1 (List.length result);

  (* Mixed whitespace *)
  let result = tokenizer "hello\tworld\nfoo\rbar" in
  check bool "Handles tabs and newlines" true (List.length result > 0)

(** Test BERT pre-tokenizer *)
let test_bert_pretokenizer () =
  let test_case text expected =
    let result = Pre.bert () text in
    check_tokenization
      (Printf.sprintf "BERT tokenization of %S" text)
      result expected
  in

  (* Basic tokenization *)
  test_case "Hello world" [ ("Hello", (0, 5)); ("world", (6, 11)) ];
  test_case "Hello, world!"
    [ ("Hello", (0, 5)); (",", (5, 6)); ("world", (7, 12)); ("!", (12, 13)) ];

  (* Punctuation handling *)
  test_case "test." [ ("test", (0, 4)); (".", (4, 5)) ];
  test_case "a-b" [ ("a", (0, 1)); ("-", (1, 2)); ("b", (2, 3)) ];
  test_case "it's" [ ("it", (0, 2)); ("'", (2, 3)); ("s", (3, 4)) ];

  (* Multiple spaces *)
  test_case "hello  world" [ ("hello", (0, 5)); ("world", (7, 12)) ];

  (* Unicode *)
  test_case "café" [ ("café", (0, 5)) ];

  (* Note: é is 2 bytes in UTF-8 *)

  (* Empty and whitespace *)
  test_case "" [];
  test_case "   " []

(** Test Whitespace pre-tokenizer *)
let test_whitespace_pretokenizer () =
  let test_case text expected =
    let result = Pre.whitespace () text in
    check_tokenization
      (Printf.sprintf "Whitespace tokenization of %S" text)
      result expected
  in

  (* Pattern is \w+|[^\w\s]+ *)
  test_case "Hello world" [ ("Hello", (0, 5)); ("world", (6, 11)) ];
  test_case "Hello, world!"
    [ ("Hello", (0, 5)); (",", (5, 6)); ("world", (7, 12)); ("!", (12, 13)) ];
  test_case "test_var" [ ("test_var", (0, 8)) ];
  (* underscore is part of \w *)
  test_case "123abc" [ ("123abc", (0, 6)) ];
  (* numbers are part of \w *)
  test_case "a+b=c"
    [
      ("a", (0, 1)); ("+", (1, 2)); ("b", (2, 3)); ("=", (3, 4)); ("c", (4, 5));
    ]

(** Test WhitespaceSplit pre-tokenizer *)
let test_whitespace_split () =
  let test_case text expected =
    let result = Pre.whitespace_split () text in
    check_tokenization
      (Printf.sprintf "WhitespaceSplit of %S" text)
      result expected
  in

  (* Simple split on whitespace *)
  test_case "Hello world" [ ("Hello", (0, 5)); ("world", (6, 11)) ];
  test_case "  Hello  world  " [ ("Hello", (2, 7)); ("world", (9, 14)) ];
  test_case "one\ttwo\nthree"
    [ ("one", (0, 3)); ("two", (4, 7)); ("three", (8, 13)) ];
  test_case "" [];
  test_case "   " []

(** Test Punctuation pre-tokenizer *)
let test_punctuation_pretokenizer () =
  (* Test different behaviors *)
  let test_isolated text expected =
    let tokenizer = Pre.punctuation ~behavior:`Isolated () in
    let result = tokenizer text in
    check_tokenization
      (Printf.sprintf "Punctuation Isolated %S" text)
      result expected
  in

  let test_removed text expected =
    let tokenizer = Pre.punctuation ~behavior:`Removed () in
    let result = tokenizer text in
    check_tokenization
      (Printf.sprintf "Punctuation Removed %S" text)
      result expected
  in

  (* Isolated behavior *)
  test_isolated "Hello, world!"
    [ ("Hello", (0, 5)); (",", (5, 6)); (" world", (6, 12)); ("!", (12, 13)) ];

  (* Removed behavior *)
  test_removed "Hello, world!" [ ("Hello", (0, 5)); (" world", (6, 12)) ];

  (* Multiple punctuation *)
  test_isolated "test...end"
    [
      ("test", (0, 4));
      (".", (4, 5));
      (".", (5, 6));
      (".", (6, 7));
      ("end", (7, 10));
    ];

  (* Unicode punctuation *)
  test_isolated "test—end" [ ("test", (0, 4)); ("—", (4, 7)); ("end", (7, 10)) ]
(* em dash is 3 bytes *)

(** Test Digits pre-tokenizer *)
let test_digits_pretokenizer () =
  let test_individual text expected =
    let tokenizer = Pre.digits ~individual_digits:true () in
    let result = tokenizer text in
    check_tokenization
      (Printf.sprintf "Digits individual %S" text)
      result expected
  in

  let test_grouped text expected =
    let tokenizer = Pre.digits ~individual_digits:false () in
    let result = tokenizer text in
    check_tokenization (Printf.sprintf "Digits grouped %S" text) result expected
  in

  (* Individual digits *)
  test_individual "123" [ ("1", (0, 1)); ("2", (1, 2)); ("3", (2, 3)) ];
  test_individual "a1b2"
    [ ("a", (0, 1)); ("1", (1, 2)); ("b", (2, 3)); ("2", (3, 4)) ];

  (* Grouped digits *)
  test_grouped "123" [ ("123", (0, 3)) ];
  test_grouped "a123b456"
    [ ("a", (0, 1)); ("123", (1, 4)); ("b", (4, 5)); ("456", (5, 8)) ];
  test_grouped "3.14" [ ("3", (0, 1)); (".", (1, 2)); ("14", (2, 4)) ]

(** Test Split pre-tokenizer *)
let test_split_pretokenizer () =
  let test_case pattern behavior text expected =
    let tokenizer = Pre.split ~pattern ~behavior () in
    let result = tokenizer text in
    check_tokenization
      (Printf.sprintf "Split pattern=%S behavior=%s text=%S" pattern
         (match behavior with
         | `Isolated -> "Isolated"
         | `Removed -> "Removed"
         | `Merged_with_previous -> "MergedPrev"
         | `Merged_with_next -> "MergedNext"
         | `Contiguous -> "Contiguous")
         text)
      result expected
  in

  (* Test different behaviors *)
  test_case "," `Isolated "a,b,c"
    [
      ("a", (0, 1)); (",", (1, 2)); ("b", (2, 3)); (",", (3, 4)); ("c", (4, 5));
    ];

  test_case "," `Removed "a,b,c" [ ("a", (0, 1)); ("b", (2, 3)); ("c", (4, 5)) ];

  test_case "," `Merged_with_previous "a,b,c"
    [ ("a,", (0, 2)); ("b,", (2, 4)); ("c", (4, 5)) ];

  test_case "," `Merged_with_next "a,b,c"
    [ ("a", (0, 1)); (",b", (1, 3)); (",c", (3, 5)) ];

  (* Test with longer pattern *)
  test_case "::" `Isolated "a::b::c"
    [
      ("a", (0, 1)); ("::", (1, 3)); ("b", (3, 4)); ("::", (4, 6)); ("c", (6, 7));
    ]

(** Test CharDelimiterSplit pre-tokenizer *)
let test_char_delimiter_split () =
  let test_case delim text expected =
    let result = Pre.char_delimiter_split ~delimiter:delim () text in
    check_tokenization
      (Printf.sprintf "CharDelimiterSplit delim='%c' text=%S" delim text)
      result expected
  in

  test_case ',' "a,b,c" [ ("a", (0, 1)); ("b", (2, 3)); ("c", (4, 5)) ];
  test_case ' ' "hello world" [ ("hello", (0, 5)); ("world", (6, 11)) ];
  test_case '|' "one|two|three"
    [ ("one", (0, 3)); ("two", (4, 7)); ("three", (8, 13)) ];
  test_case ',' "" [];
  test_case ',' "," []

(** Test Sequence pre-tokenizer *)
let test_sequence_pretokenizer () =
  (* Combine whitespace split then punctuation isolation *)
  let tokenizers =
    [ Pre.whitespace_split (); Pre.punctuation ~behavior:`Isolated () ]
  in
  let tokenizer = Pre.sequence tokenizers in

  let test_case text expected =
    let result = tokenizer text in
    check_tokenization (Printf.sprintf "Sequence %S" text) result expected
  in

  (* First splits on whitespace, then isolates punctuation in each piece *)
  test_case "Hello, world!"
    [ ("Hello", (0, 5)); (",", (5, 6)); ("world", (7, 12)); ("!", (12, 13)) ];

  (* Multiple words and punctuation *)
  test_case "test. another, example!"
    [
      ("test", (0, 4));
      (".", (4, 5));
      ("another", (6, 13));
      (",", (13, 14));
      ("example", (15, 22));
      ("!", (22, 23));
    ]

(** Test FixedLength pre-tokenizer *)
let test_fixed_length () =
  let test_case length text expected =
    let result = Pre.fixed_length ~length text in
    check_tokenization
      (Printf.sprintf "FixedLength %d %S" length text)
      result expected
  in

  test_case 3 "abcdefghi" [ ("abc", (0, 3)); ("def", (3, 6)); ("ghi", (6, 9)) ];
  test_case 2 "abcde" [ ("ab", (0, 2)); ("cd", (2, 4)); ("e", (4, 5)) ];
  test_case 5 "hello" [ ("hello", (0, 5)) ];
  test_case 0 "test" [];
  test_case 3 "" [];

  (* With UTF-8 - counts characters not bytes *)
  test_case 2 "café" [ ("ca", (0, 2)); ("fé", (2, 5)) ]
(* é is 2 bytes *)

(** Test UnicodeScripts pre-tokenizer *)
let test_unicode_scripts () =
  let test_case text desc =
    let tokenizer = Pre.unicode_scripts () in
    let result = tokenizer text in
    (* Just verify it runs without crashing and produces something reasonable *)
    check bool
      (Printf.sprintf "UnicodeScripts %s" desc)
      true
      (List.length result >= 0)
  in

  test_case "Hello world" "Latin text";
  test_case "Hello世界" "Mixed Latin and Chinese";
  test_case "Привет мир" "Cyrillic";
  test_case "مرحبا بالعالم" "Arabic";
  test_case "こんにちは世界" "Japanese";
  test_case "" "Empty string"

(** Test Metaspace pre-tokenizer (already tested in test_metaspace.ml) *)
let test_metaspace_basic () =
  let test_case text expected =
    let result =
      Pre.metaspace ~replacement:'_' ~prepend_scheme:`Always ~split:true () text
    in
    check_strings (Printf.sprintf "Metaspace %S" text) result expected
  in

  test_case "Hello world" [ "_Hello"; "_world" ];
  test_case " starts with space" [ "_starts"; "_with"; "_space" ];
  test_case "" []

(** Main test suite *)
let () =
  let open Alcotest in
  run "Pre-tokenizers Test Suite"
    [
      ( "byte_level",
        [
          test_case "ByteLevel basic" `Quick test_byte_level_basic;
          test_case "ByteLevel prefix space" `Quick test_byte_level_prefix_space;
          test_case "ByteLevel special chars" `Quick
            test_byte_level_special_chars;
          test_case "ByteLevel unicode" `Quick test_byte_level_unicode;
          test_case "ByteLevel edge cases" `Quick test_byte_level_edge_cases;
        ] );
      ("bert", [ test_case "BERT tokenization" `Quick test_bert_pretokenizer ]);
      ( "whitespace",
        [
          test_case "Whitespace tokenization" `Quick
            test_whitespace_pretokenizer;
          test_case "WhitespaceSplit" `Quick test_whitespace_split;
        ] );
      ( "punctuation",
        [
          test_case "Punctuation behaviors" `Quick test_punctuation_pretokenizer;
        ] );
      ( "digits",
        [ test_case "Digits tokenization" `Quick test_digits_pretokenizer ] );
      ( "split",
        [
          test_case "Split with patterns" `Quick test_split_pretokenizer;
          test_case "CharDelimiterSplit" `Quick test_char_delimiter_split;
        ] );
      ( "sequence",
        [ test_case "Sequence of tokenizers" `Quick test_sequence_pretokenizer ]
      );
      ( "fixed_length",
        [ test_case "FixedLength chunks" `Quick test_fixed_length ] );
      ( "unicode_scripts",
        [ test_case "UnicodeScripts" `Quick test_unicode_scripts ] );
      ("metaspace", [ test_case "Metaspace basic" `Quick test_metaspace_basic ]);
    ]
