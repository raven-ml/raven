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

(* Case mapping is not case folding: the ligatures and the sharp s lower to
   themselves. Expectations from HuggingFace [normalizers.Lowercase()]. *)
let test_lowercase_is_not_folding () =
  let case text expected =
    equal
      ~msg:(Printf.sprintf "lowercase %S" text)
      string expected
      (Normalizer.apply Normalizer.lowercase text)
  in
  case "\xC3\x9F" "\xC3\x9F";
  case "\xE1\xBA\x9E" "\xC3\x9F";
  case "\xEF\xAC\x81" "\xEF\xAC\x81";
  case "\xEF\xAC\x84" "\xEF\xAC\x84";
  case "\xC4\xB0" "i\xCC\x87";
  case "\xCE\xA3" "\xCF\x83";
  case "\xC7\x85" "\xC7\x86";
  case "\xC3\x80\xC3\x89\xC3\x8E" "\xC3\xA0\xC3\xA9\xC3\xAE"

(* Expectations from HuggingFace [normalizers.StripAccents()], which removes
   every mark and does not decompose. *)
let test_strip_accents_keeps_composition () =
  let case text expected =
    equal
      ~msg:(Printf.sprintf "strip_accents %S" text)
      string expected
      (Normalizer.apply Normalizer.strip_accents text)
  in
  (* Bengali vowel sign O and Devanagari vowel sign I are spacing marks. *)
  case "\xE0\xA7\x8B" "";
  case "\xE0\xA4\xBF" "";
  (* Enclosing and nonspacing marks go too. *)
  case "a\xE0\xA4\x83\xE2\x83\x9D\xCC\x81" "a";
  (* Precomposed characters are left alone without a preceding NFD. *)
  case "\xC3\xA1" "\xC3\xA1";
  case "caf\xC3\xA9" "caf\xC3\xA9";
  case "\xE0\xA4\x95\xE0\xA4\xBC" "\xE0\xA4\x95";
  case "A\xCC\x81" "A"

(* Expectations from HuggingFace [normalizers.BertNormalizer(clean_text=True,
   handle_chinese_chars=True, strip_accents=None, lowercase=True)], the
   bert-base-uncased settings. *)
let test_bert_normalizer () =
  let n = Normalizer.bert () in
  let case text expected =
    equal
      ~msg:(Printf.sprintf "bert normalize %S" text)
      string expected (Normalizer.apply n text)
  in
  (* Only nonspacing marks are stripped, so the vowel signs of an abugida
     survive: ["नमस्ते हिन्दी"] keeps its ि and ी. *)
  case
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87 \
     \xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA5\x8D\xE0\xA4\xA6\xE0\xA5\x80"
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA4\xA4 \
     \xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA4\xA6\xE0\xA5\x80";
  case "a\xE0\xA4\x83\xE2\x83\x9D\xCC\x81" "a\xE0\xA4\x83\xE2\x83\x9D";
  (* Accents proper are stripped, after decomposition. *)
  case "caf\xC3\xA9" "cafe";
  case "A\xCC\x81BC" "abc";
  case "\xC4\xB0" "i";
  case "\xE1\xBE\xBC" "\xCE\xB1";
  (* Lowercasing, not folding. *)
  case "\xC3\x9F" "\xC3\x9F";
  case "\xE1\xBA\x9E" "\xC3\x9F";
  case "\xEF\xAC\x81" "\xEF\xAC\x81";
  case "\xCE\xA3" "\xCF\x83";
  (* Control, format and private use characters are removed, and so are NUL and
     the replacement character. *)
  case "\x00\x07\x7F" "";
  case "a\xC2\xADb" "ab";
  case "a\xE2\x80\x8Bb" "ab";
  case "a\xEE\x80\x80b" "ab";
  case "a\xEF\xBF\xBDb" "ab";
  (* Unassigned codepoints are not controls: they survive to reach the model. *)
  case "\xF4\x8F\xBF\xBF" "\xF4\x8F\xBF\xBF";
  case "\xCD\xB8" "\xCD\xB8";
  (* Whitespace of any kind becomes a plain space. *)
  case "a\xE2\x80\xA8b" "a b";
  case "a\xE3\x80\x80b" "a b";
  (* CJK ideographs are padded with spaces, one at a time. *)
  case "a\xE6\xBC\xA2b" "a \xE6\xBC\xA2 b";
  case "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E"
    " \xE6\x97\xA5  \xE6\x9C\xAC  \xE8\xAA\x9E "

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
    word_level ~normalizer
      ~pre:(Pre_tokenizer.whitespace ())
      ~vocab:[ ("hello", 0); ("world", 1); ("!", 2) ]
      ()
  in
  let tokens = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"normalized tokenization" (list string) [ "hello"; "world"; "!" ]
    tokens

let test_tokenize_unicode_words () =
  let text = "café résumé naïve" in
  let tokenizer =
    word_level
      ~pre:(Pre_tokenizer.whitespace ())
      ~vocab:[ ("café", 0); ("résumé", 1); ("naïve", 2) ]
      ()
  in
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
    test "lowercase is not case folding" test_lowercase_is_not_folding;
    test "strip accents normalization" test_strip_accents_normalization;
    test "strip accents keeps composition" test_strip_accents_keeps_composition;
    test "bert normalizer" test_bert_normalizer;
    test "normalization sequence" test_normalization_sequence;
    (* Integration *)
    test "tokenize with normalization" test_tokenize_with_normalization;
    test "tokenize unicode words" test_tokenize_unicode_words;
    (* Error handling *)
    test "malformed unicode" test_malformed_unicode;
  ]

let () = run "brot unicode" [ group "unicode" unicode_tests ]
