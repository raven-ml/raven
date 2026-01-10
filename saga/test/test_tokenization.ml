(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tokenization tests for saga *)

open Alcotest
open Saga_tokenizers

(* Helper function to tokenize text *)
let tokenize_text text =
  (* Pre-tokenize to get all unique tokens *)
  let pre_tokens = Pre_tokenizers.whitespace () text in
  let unique_tokens =
    List.fold_left
      (fun acc (tok, _) -> if List.mem tok acc then acc else tok :: acc)
      [] pre_tokens
    |> List.rev
  in
  (* Build vocabulary with all tokens from the text plus extras *)
  let all_tokens =
    unique_tokens
    @
    (* Add numbered words for long text test *)
    List.init 1000 (fun i -> Printf.sprintf "word%d" i)
  in
  let vocab = List.mapi (fun i token -> (token, i)) all_tokens in

  (* Create WordLevel tokenizer with the vocabulary *)
  let tokenizer =
    Tokenizer.word_level ~vocab ~unk_token:"<unk>"
      ~pre:(Pre_tokenizers.whitespace ())
      ()
  in
  Tokenizer.encode tokenizer text |> Encoding.get_tokens |> Array.to_list

(* ───── Basic Tokenization Tests ───── *)

let test_tokenize_words_simple () =
  let tokens = tokenize_text "Hello world!" in
  check (list string) "simple words" [ "Hello"; "world"; "!" ] tokens

let test_tokenize_words_punctuation () =
  let tokens = tokenize_text "don't stop, it's fun!" in
  check (list string) "words with punctuation"
    [ "don"; "'"; "t"; "stop"; ","; "it"; "'"; "s"; "fun"; "!" ]
    tokens

let test_tokenize_words_numbers () =
  let tokens = tokenize_text "I have 42 apples and 3.14 pies" in
  check (list string) "words with numbers"
    [ "I"; "have"; "42"; "apples"; "and"; "3"; "."; "14"; "pies" ]
    tokens

let test_tokenize_words_empty () =
  let tokens = tokenize_text "" in
  check (list string) "empty string" [] tokens

let test_tokenize_words_whitespace_only () =
  let tokens = tokenize_text "   \t\n  " in
  check (list string) "whitespace only" [] tokens

let test_tokenize_words_special_chars () =
  let tokens = tokenize_text "hello@world.com #ml $100 C++" in
  check (list string) "special characters"
    [ "hello"; "@"; "world"; "."; "com"; "#"; "ml"; "$"; "100"; "C"; "++" ]
    tokens

(* ───── Character Tokenization Tests ───── *)

let tokenize_chars text =
  let chars = ref [] in
  String.iter (fun c -> chars := String.make 1 c :: !chars) text;
  List.rev !chars

let test_tokenize_chars_ascii () =
  let tokens = tokenize_chars "Hi!" in
  check (list string) "ASCII chars" [ "H"; "i"; "!" ] tokens

let test_tokenize_chars_unicode () =
  let tokens = tokenize_chars "Hello 👋 世界" in
  (* Note: UTF-8 encoding means multi-byte chars may appear differently *)
  check bool "has tokens" true (List.length tokens > 0)

let test_tokenize_chars_empty () =
  let tokens = tokenize_chars "" in
  check (list string) "empty string chars" [] tokens

(* ───── Regex Tokenization Tests ───── *)

let test_tokenize_regex_words () =
  (* Use the helper that sets up vocabulary properly *)
  let tokens = tokenize_text "hello-world test_123" in
  check (list string) "regex words" [ "hello"; "-"; "world"; "test_123" ] tokens

let test_tokenize_regex_custom () =
  (* Test with punctuation pre-tokenizer *)
  let text = "don't stop!" in
  let pre_tokens = Pre_tokenizers.punctuation () text in
  let vocab = List.mapi (fun i (tok, _) -> (tok, i)) pre_tokens in
  let tokenizer =
    Tokenizer.word_level ~vocab ~unk_token:"<unk>"
      ~pre:(Pre_tokenizers.punctuation ())
      ()
  in
  let tokens =
    Tokenizer.encode tokenizer text |> Encoding.get_tokens |> Array.to_list
  in
  check bool "has tokens" true (List.length tokens > 0)

let test_tokenize_regex_no_match () =
  let tokenizer = Tokenizer.word_level () in
  let tokens =
    Tokenizer.encode tokenizer "no numbers here"
    |> Encoding.get_tokens |> Array.to_list
  in
  check (list string) "regex no match" [] tokens

(* ───── Unigram Model Tests ───── *)

(* Round-trip lookups *)
let test_unigram_roundtrip () =
  let tokens = [ "hello"; "world"; "test" ] in
  let vocab = List.map (fun token -> (token, 0.0)) tokens in
  let tokenizer = Tokenizer.unigram ~vocab () in
  List.iteri
    (fun expected_id token ->
      check (option int)
        (Printf.sprintf "token_to_id '%s'" token)
        (Some expected_id)
        (Tokenizer.token_to_id tokenizer token);
      check (option string)
        (Printf.sprintf "id_to_token %d" expected_id)
        (Some token)
        (Tokenizer.id_to_token tokenizer expected_id))
    tokens

(* token_to_id - out of vocab *)
let test_unigram_token_to_id_oov () =
  let tokenizer =
    Tokenizer.unigram ~vocab:[ ("hello", 0.0); ("world", 0.0) ] ()
  in
  check (option int) "token_to_id out-of-vocab" None
    (Tokenizer.token_to_id tokenizer "missing")

(* id_to_token - out of bounds *)
let test_unigram_id_to_token_oob () =
  let tokenizer =
    Tokenizer.unigram ~vocab:[ ("hello", 0.0); ("world", 0.0) ] ()
  in
  check (option string) "id_to_token negative" None
    (Tokenizer.id_to_token tokenizer (-1));
  check (option string) "id_to_token out of bounds" None
    (Tokenizer.id_to_token tokenizer 10)

(* Test empty vocabulary *)
let test_unigram_empty_vocab () =
  let tokenizer = Tokenizer.unigram ~vocab:[] () in
  check (option int) "empty vocab token_to_id" None
    (Tokenizer.token_to_id tokenizer "test");
  check (option string) "empty vocab id_to_token" None
    (Tokenizer.id_to_token tokenizer 0)

(* Test special characters and unicode *)
let test_unigram_special_tokens () =
  let tokenizer =
    Tokenizer.unigram
      ~vocab:
        [
          ("<unk>", 0.0);
          ("<s>", 0.0);
          ("</s>", 0.0);
          ("▁hello", 0.0);
          ("世界", 0.0);
        ]
      ()
  in
  check (option int) "special <unk>" (Some 0)
    (Tokenizer.token_to_id tokenizer "<unk>");
  check (option int) "special <s>" (Some 1)
    (Tokenizer.token_to_id tokenizer "<s>");
  check (option int) "sentencepiece token" (Some 3)
    (Tokenizer.token_to_id tokenizer "▁hello");
  check (option int) "unicode token" (Some 4)
    (Tokenizer.token_to_id tokenizer "世界");
  check (option string) "id to unicode" (Some "世界")
    (Tokenizer.id_to_token tokenizer 4)

let test_unigram_encode_sequence () =
  let tokenizer =
    Tokenizer.unigram ~vocab:[ ("hello", 0.0); ("world", 0.0) ] ()
  in
  let encoding = Tokenizer.encode tokenizer "hello world" in
  let tokens = Encoding.get_tokens encoding |> Array.to_list in
  check (list string) "unigram encode tokens" [ "hello"; "world" ] tokens

let test_pad_token_reassignment_updates_id () =
  let vocab =
    [ ("hello", 0); ("world", 1); ("<unk>", 2); ("<pad>", 3); ("[PAD]", 4) ]
  in
  let tokenizer =
    Tokenizer.word_level ~vocab ~unk_token:"<unk>"
      ~pre:(Pre_tokenizers.whitespace ())
      ~specials:[ Special.pad "<pad>" ]
      ()
  in
  let tokenizer = Tokenizer.set_pad_token tokenizer (Some "[PAD]") in
  check (option string) "pad token updated" (Some "[PAD]")
    (Tokenizer.pad_token tokenizer);
  let pad_id =
    match Tokenizer.token_to_id tokenizer "[PAD]" with
    | Some id -> id
    | None -> failwith "missing pad id"
  in
  let encoding =
    Tokenizer.encode tokenizer "hello"
      ~padding:
        {
          length = `Fixed 3;
          direction = `Right;
          pad_id = None;
          pad_type_id = None;
          pad_token = None;
        }
  in
  let ids = Encoding.get_ids encoding |> Array.to_list in
  let pad_ids = List.tl ids in
  check (list int) "pad id matches reassigned token" [ pad_id; pad_id ] pad_ids

(* ───── Edge Cases ───── *)

let test_tokenize_long_text () =
  let text =
    String.concat " " (List.init 1000 (fun i -> Printf.sprintf "word%d" i))
  in
  let tokens = tokenize_text text in
  check int "long text token count" 1000 (List.length tokens)

let test_tokenize_repeated_punctuation () =
  let tokens = tokenize_text "wow!!! really???" in
  check (list string) "repeated punctuation"
    [ "wow"; "!!!"; "really"; "???" ]
    tokens

let test_tokenize_mixed_whitespace () =
  let tokens = tokenize_text "hello\tworld\nthere\r\nfriend" in
  check (list string) "mixed whitespace"
    [ "hello"; "world"; "there"; "friend" ]
    tokens

(* ───── Test Suite ───── *)

let tokenization_tests =
  [
    (* Words tokenization *)
    test_case "tokenize words simple" `Quick test_tokenize_words_simple;
    test_case "tokenize words punctuation" `Quick
      test_tokenize_words_punctuation;
    test_case "tokenize words numbers" `Quick test_tokenize_words_numbers;
    test_case "tokenize words empty" `Quick test_tokenize_words_empty;
    test_case "tokenize words whitespace only" `Quick
      test_tokenize_words_whitespace_only;
    test_case "tokenize words special chars" `Quick
      test_tokenize_words_special_chars;
    (* Character tokenization *)
    test_case "tokenize chars ASCII" `Quick test_tokenize_chars_ascii;
    test_case "tokenize chars unicode" `Quick test_tokenize_chars_unicode;
    test_case "tokenize chars empty" `Quick test_tokenize_chars_empty;
    (* Regex tokenization *)
    test_case "tokenize regex words" `Quick test_tokenize_regex_words;
    test_case "tokenize regex custom" `Quick test_tokenize_regex_custom;
    test_case "tokenize regex no match" `Quick test_tokenize_regex_no_match;
    (* Edge cases *)
    test_case "tokenize long text" `Quick test_tokenize_long_text;
    test_case "tokenize repeated punctuation" `Quick
      test_tokenize_repeated_punctuation;
    test_case "tokenize mixed whitespace" `Quick test_tokenize_mixed_whitespace;
    (* Unigram model tests *)
    test_case "unigram roundtrip" `Quick test_unigram_roundtrip;
    test_case "unigram token_to_id out-of-vocab" `Quick
      test_unigram_token_to_id_oov;
    test_case "unigram id_to_token out-of-bounds" `Quick
      test_unigram_id_to_token_oob;
    test_case "unigram empty vocab" `Quick test_unigram_empty_vocab;
    test_case "unigram special tokens" `Quick test_unigram_special_tokens;
    test_case "unigram encode sequence" `Quick test_unigram_encode_sequence;
    test_case "pad token reassignment updates id" `Quick
      test_pad_token_reassignment_updates_id;
  ]

let () =
  Alcotest.run "saga tokenization" [ ("tokenization", tokenization_tests) ]
