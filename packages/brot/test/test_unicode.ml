(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Unicode processing tests for brot *)

open Windtrap
open Brot

(* Normalization via public API *)

let test_lowercase_normalization () =
  let text = "HELLO WORLD" in
  let normalizer = Normalizer.lowercase in
  let result = Normalizer.apply normalizer text in
  equal ~msg:"lowercase" string "hello world" result

let test_strip_accents_normalization () =
  let text = "caf\xC3\xA9 na\xC3\xAFve r\xC3\xA9sum\xC3\xA9" in
  let normalizer =
    Normalizer.sequence [ Normalizer.nfd; Normalizer.strip_accents ]
  in
  let result = Normalizer.apply normalizer text in
  equal ~msg:"strip accents" string "cafe naive resume" result

let test_normalization_sequence () =
  let text = "  HELLO  World  " in
  let normalizer =
    Normalizer.sequence
      [
        Normalizer.lowercase;
        Normalizer.strip ();
        Normalizer.replace ~pattern:"\\s+" ~replacement:" ";
      ]
  in
  let result = Normalizer.apply normalizer text in
  equal ~msg:"sequence" string "hello world" result

(* Integration with Tokenizer *)

let test_tokenize_with_normalization () =
  let text = "HELLO   WORLD!" in
  let normalizer =
    Normalizer.sequence
      [
        Normalizer.lowercase;
        Normalizer.replace ~pattern:"\\s+" ~replacement:" ";
      ]
  in
  let tokenizer =
    word_level ~normalizer ~pre:(Pre_tokenizer.whitespace ()) ()
  in
  let tokenizer = add_tokens tokenizer [ "hello"; "world"; "!" ] in
  let tokens = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"normalized tokenization" (list string) [ "hello"; "world"; "!" ]
    tokens

let test_tokenize_unicode_words () =
  let text = "café résumé naïve" in
  let tokenizer = word_level ~pre:(Pre_tokenizer.whitespace ()) () in
  let tokenizer = add_tokens tokenizer [ "café"; "résumé"; "naïve" ] in
  let tokens = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"tokenized unicode" bool true (List.length tokens > 0)

let test_malformed_unicode () =
  let text = "Hello" ^ String.make 1 '\xFF' ^ String.make 1 '\xFE' ^ "World" in
  let tokenizer = chars () in
  let tokens = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"handled malformed" bool true (List.length tokens > 0)

(* Test Suite *)

let unicode_tests =
  [
    (* Normalization *)
    test "lowercase normalization" test_lowercase_normalization;
    test "strip accents normalization" test_strip_accents_normalization;
    test "normalization sequence" test_normalization_sequence;
    (* Integration *)
    test "tokenize with normalization" test_tokenize_with_normalization;
    test "tokenize unicode words" test_tokenize_unicode_words;
    (* Error handling *)
    test "malformed unicode" test_malformed_unicode;
  ]

let () = run "brot unicode" [ group "unicode" unicode_tests ]
