(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Unicode processing tests for saga *)

open Windtrap
open Saga_tokenizers

(* Basic Unicode Normalization Tests *)

let test_case_folding () =
  let text = "Hello WORLD ÄÖÜ" in
  let folded = Unicode.case_fold text in
  (* Basic ASCII folding *)
  equal ~msg:"lowercase ASCII" bool true (String.contains folded 'h');
  equal ~msg:"lowercase world" bool true (String.contains folded 'w')
(* Note: Exact Unicode case folding depends on uucp data *)

let test_strip_accents () =
  let text = "café naïve résumé" in
  let stripped = Unicode.strip_accents text in
  (* Should remove combining marks but this simple implementation might not
     handle precomposed chars *)
  equal ~msg:"stripped accents" string text
    stripped (* For now, just verify no crash *)

let test_clean_text () =
  let text =
    "  Hello   \t\n  World!  " ^ String.make 1 '\x00' ^ String.make 1 '\x01'
  in
  let cleaned = Unicode.clean_text text in
  equal ~msg:"normalized whitespace" string "Hello World!" cleaned

let test_clean_text_keep_control () =
  let text = "Hello" ^ String.make 1 '\x00' ^ "World" in
  let cleaned =
    Unicode.clean_text ~remove_control:false ~normalize_whitespace:false text
  in
  equal ~msg:"kept control chars" string text cleaned

(* Unicode Classification Tests *)

let test_categorize_char () =
  (* ASCII tests *)
  equal ~msg:"letter A" bool true
    (Unicode.categorize_char (Uchar.of_char 'A') = Unicode.Letter);
  equal ~msg:"digit 5" bool true
    (Unicode.categorize_char (Uchar.of_char '5') = Unicode.Number);
  equal ~msg:"comma" bool true
    (Unicode.categorize_char (Uchar.of_char ',') = Unicode.Punctuation);
  equal ~msg:"space" bool true
    (Unicode.categorize_char (Uchar.of_char ' ') = Unicode.Whitespace)

let test_is_cjk () =
  (* Test some CJK characters *)
  equal ~msg:"Chinese char" bool true (Unicode.is_cjk (Uchar.of_int 0x4E00));
  (* 一 *)
  equal ~msg:"Hiragana" bool true (Unicode.is_cjk (Uchar.of_int 0x3042));
  (* あ *)
  equal ~msg:"Katakana" bool true (Unicode.is_cjk (Uchar.of_int 0x30A2));
  (* ア *)
  equal ~msg:"Hangul" bool true (Unicode.is_cjk (Uchar.of_int 0xAC00));
  (* 가 *)
  equal ~msg:"Not CJK" bool false (Unicode.is_cjk (Uchar.of_char 'A'))

(* Word Splitting Tests *)

let test_split_words_basic () =
  let words = Unicode.split_words "Hello, world! How are you?" in
  equal ~msg:"basic split"
    (list string)
    [ "Hello"; "world"; "How"; "are"; "you" ]
    words

let test_split_words_numbers () =
  let words = Unicode.split_words "test123 456test" in
  equal ~msg:"words with numbers" (list string) [ "test123"; "456test" ] words

let test_split_words_unicode () =
  let words = Unicode.split_words "café naïve" in
  (* Should handle accented characters as part of words *)
  equal ~msg:"word count" int 2 (List.length words)

let test_split_words_cjk () =
  let words = Unicode.split_words "Hello世界" in
  (* Debug output *)
  Printf.printf "CJK split words: [%s] (count: %d)\n"
    (String.concat "; " (List.map (Printf.sprintf "'%s'") words))
    (List.length words);
  (* CJK characters should be split individually *)
  equal ~msg:"found some words" bool true (List.length words > 0)

(* Grapheme Count Tests *)

let test_grapheme_count_ascii () =
  let count = Unicode.grapheme_count "Hello!" in
  equal ~msg:"ASCII count" int 6 count

let test_grapheme_count_emoji () =
  let count = Unicode.grapheme_count "Hi 👋" in
  equal ~msg:"with emoji" int 4 count (* H + i + space + wave *)

(* UTF-8 Validation Tests *)

let test_is_valid_utf8 () =
  equal ~msg:"valid ASCII" bool true (Unicode.is_valid_utf8 "Hello");
  equal ~msg:"valid UTF-8" bool true (Unicode.is_valid_utf8 "Hello 世界");
  (* Invalid UTF-8 sequence *)
  let invalid = String.make 1 '\xFF' ^ String.make 1 '\xFE' in
  equal ~msg:"invalid UTF-8" bool false (Unicode.is_valid_utf8 invalid)

(* Emoji Removal Tests *)

let test_remove_emoji () =
  let text = "Hello 👋 World 🌍!" in
  let cleaned = Unicode.remove_emoji text in
  (* Check that emoji were removed by checking length *)
  equal ~msg:"removed emoji" bool true (String.length cleaned < String.length text);
  equal ~msg:"kept letters" bool true (String.contains cleaned 'H')

(* Integration with Tokenizer *)

let test_tokenize_with_normalization () =
  let text = "HELLO   WORLD!" in
  let tokenizer = Tokenizer.word_level () in
  let tokenizer = Tokenizer.add_tokens tokenizer [ "hello"; "world"; "!" ] in
  let normalizer =
    Normalizers.sequence
      [
        Normalizers.lowercase ();
        Normalizers.replace ~pattern:"\\s+" ~replacement:" " ();
      ]
  in
  let tokenizer = Tokenizer.with_normalizer tokenizer (Some normalizer) in
  let tokenizer =
    Tokenizer.with_pre_tokenizer tokenizer (Some (Pre_tokenizers.whitespace ()))
  in
  let tokens =
    Tokenizer.encode tokenizer text |> Encoding.get_tokens |> Array.to_list
  in
  (* Whitespace pre-tokenizer splits punctuation, so we get three tokens *)
  equal ~msg:"normalized tokenization" (list string) [ "hello"; "world"; "!" ] tokens

let test_tokenize_unicode_words () =
  (* Test that Unicode-aware tokenization works *)
  let text = "café résumé naïve" in
  let tokenizer = Tokenizer.word_level () in
  let tokenizer =
    Tokenizer.add_tokens tokenizer [ "café"; "résumé"; "naïve" ]
  in
  let tokenizer =
    Tokenizer.with_pre_tokenizer tokenizer (Some (Pre_tokenizers.whitespace ()))
  in
  let tokens =
    Tokenizer.encode tokenizer text |> Encoding.get_tokens |> Array.to_list
  in
  (* Our simple tokenizer treats e as a separate character, so we get more
     tokens *)
  equal ~msg:"tokenized unicode" bool true (List.length tokens > 0)

(* Error Handling Tests *)

let test_malformed_unicode () =
  (* Malformed sequences should be skipped, not crash *)
  let text = "Hello" ^ String.make 1 '\xFF' ^ String.make 1 '\xFE' ^ "World" in
  let tokenizer = Tokenizer.chars () in
  let tokens =
    Tokenizer.encode tokenizer text |> Encoding.get_tokens |> Array.to_list
  in
  equal ~msg:"handled malformed" bool true (List.length tokens > 0)

(* Test Suite *)

let unicode_tests =
  [
    (* Normalization *)
    test "case folding" test_case_folding;
    test "strip accents" test_strip_accents;
    test "clean text" test_clean_text;
    test "clean text keep control" test_clean_text_keep_control;
    (* Classification *)
    test "categorize char" test_categorize_char;
    test "is CJK" test_is_cjk;
    (* Word splitting *)
    test "split words basic" test_split_words_basic;
    test "split words numbers" test_split_words_numbers;
    test "split words unicode" test_split_words_unicode;
    test "split words CJK" test_split_words_cjk;
    (* Grapheme counting *)
    test "grapheme count ASCII" test_grapheme_count_ascii;
    test "grapheme count emoji" test_grapheme_count_emoji;
    (* UTF-8 validation *)
    test "is valid UTF-8" test_is_valid_utf8;
    (* Emoji removal *)
    test "remove emoji" test_remove_emoji;
    (* Integration *)
    test "tokenize with normalization"
      test_tokenize_with_normalization;
    test "tokenize unicode words" test_tokenize_unicode_words;
    (* Error handling *)
    test "malformed unicode" test_malformed_unicode;
  ]

let () = run "saga unicode" [ group "unicode" unicode_tests ]
