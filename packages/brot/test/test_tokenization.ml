(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tokenization tests for brot *)

open Windtrap
open Brot

(* Helper function to tokenize text *)
let tokenize_text text =
  (* Pre-tokenize to get all unique tokens *)
  let pre_tokens =
    Pre_tokenizer.pre_tokenize (Pre_tokenizer.whitespace ()) text
  in
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
    word_level ~vocab ~unk_token:"<unk>" ~pre:(Pre_tokenizer.whitespace ()) ()
  in
  encode tokenizer text |> Encoding.tokens |> Array.to_list

(* Basic Tokenization Tests *)

let test_tokenize_words_simple () =
  let tokens = tokenize_text "Hello world!" in
  equal ~msg:"simple words" (list string) [ "Hello"; "world"; "!" ] tokens

let test_tokenize_words_punctuation () =
  let tokens = tokenize_text "don't stop, it's fun!" in
  equal ~msg:"words with punctuation" (list string)
    [ "don"; "'"; "t"; "stop"; ","; "it"; "'"; "s"; "fun"; "!" ]
    tokens

let test_tokenize_words_numbers () =
  let tokens = tokenize_text "I have 42 apples and 3.14 pies" in
  equal ~msg:"words with numbers" (list string)
    [ "I"; "have"; "42"; "apples"; "and"; "3"; "."; "14"; "pies" ]
    tokens

let test_tokenize_words_empty () =
  let tokens = tokenize_text "" in
  equal ~msg:"empty string" (list string) [] tokens

let test_tokenize_words_whitespace_only () =
  let tokens = tokenize_text "   \t\n  " in
  equal ~msg:"whitespace only" (list string) [] tokens

let test_tokenize_words_special_chars () =
  let tokens = tokenize_text "hello@world.com #ml $100 C++" in
  equal ~msg:"special characters" (list string)
    [ "hello"; "@"; "world"; "."; "com"; "#"; "ml"; "$"; "100"; "C"; "++" ]
    tokens

(* Character Tokenization Tests *)

let tokenize_chars text =
  let chars = ref [] in
  String.iter (fun c -> chars := String.make 1 c :: !chars) text;
  List.rev !chars

let test_tokenize_chars_ascii () =
  let tokens = tokenize_chars "Hi!" in
  equal ~msg:"ASCII chars" (list string) [ "H"; "i"; "!" ] tokens

let test_tokenize_chars_unicode () =
  let tokens = tokenize_chars "Hello 👋 世界" in
  (* Note: UTF-8 encoding means multi-byte chars may appear differently *)
  equal ~msg:"has tokens" bool true (List.length tokens > 0)

let test_tokenize_chars_empty () =
  let tokens = tokenize_chars "" in
  equal ~msg:"empty string chars" (list string) [] tokens

(* Pre-tokenizer Pattern Tests *)

let test_tokenize_regex_words () =
  (* Use the helper that sets up vocabulary properly *)
  let tokens = tokenize_text "hello-world test_123" in
  equal ~msg:"regex words" (list string)
    [ "hello"; "-"; "world"; "test_123" ]
    tokens

let test_tokenize_regex_custom () =
  (* Test with punctuation pre-tokenizer *)
  let text = "don't stop!" in
  let pre_tokens =
    Pre_tokenizer.pre_tokenize (Pre_tokenizer.punctuation ()) text
  in
  let vocab = List.mapi (fun i (tok, _) -> (tok, i)) pre_tokens in
  let tokenizer =
    word_level ~vocab ~unk_token:"<unk>" ~pre:(Pre_tokenizer.punctuation ()) ()
  in
  let tokens = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"has tokens" bool true (List.length tokens > 0)

let test_tokenize_regex_no_match () =
  let tokenizer = word_level () in
  let tokens =
    encode tokenizer "no numbers here" |> Encoding.tokens |> Array.to_list
  in
  equal ~msg:"regex no match" (list string) [] tokens

(* Unigram Model Tests *)

(* Round-trip lookups *)
let test_unigram_roundtrip () =
  let tokens = [ "hello"; "world"; "test" ] in
  let vocab = List.map (fun token -> (token, 0.0)) tokens in
  let tokenizer = unigram ~vocab () in
  List.iteri
    (fun expected_id token ->
      equal
        ~msg:(Printf.sprintf "token_to_id '%s'" token)
        (option int) (Some expected_id)
        (token_to_id tokenizer token);
      equal
        ~msg:(Printf.sprintf "id_to_token %d" expected_id)
        (option string) (Some token)
        (id_to_token tokenizer expected_id))
    tokens

(* token_to_id - out of vocab *)
let test_unigram_token_to_id_oov () =
  let tokenizer = unigram ~vocab:[ ("hello", 0.0); ("world", 0.0) ] () in
  equal ~msg:"token_to_id out-of-vocab" (option int) None
    (token_to_id tokenizer "missing")

(* id_to_token - out of bounds *)
let test_unigram_id_to_token_oob () =
  let tokenizer = unigram ~vocab:[ ("hello", 0.0); ("world", 0.0) ] () in
  equal ~msg:"id_to_token negative" (option string) None
    (id_to_token tokenizer (-1));
  equal ~msg:"id_to_token out of bounds" (option string) None
    (id_to_token tokenizer 10)

(* Test empty vocabulary *)
let test_unigram_empty_vocab () =
  let tokenizer = unigram ~vocab:[] () in
  equal ~msg:"empty vocab token_to_id" (option int) None
    (token_to_id tokenizer "test");
  equal ~msg:"empty vocab id_to_token" (option string) None
    (id_to_token tokenizer 0)

(* Test special characters and unicode *)
let test_unigram_special_tokens () =
  let tokenizer =
    unigram
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
  equal ~msg:"special <unk>" (option int) (Some 0)
    (token_to_id tokenizer "<unk>");
  equal ~msg:"special <s>" (option int) (Some 1) (token_to_id tokenizer "<s>");
  equal ~msg:"sentencepiece token" (option int) (Some 3)
    (token_to_id tokenizer "▁hello");
  equal ~msg:"unicode token" (option int) (Some 4) (token_to_id tokenizer "世界");
  equal ~msg:"id to unicode" (option string) (Some "世界")
    (id_to_token tokenizer 4)

let test_unigram_encode_sequence () =
  let tokenizer = unigram ~vocab:[ ("hello", 0.0); ("world", 0.0) ] () in
  let encoding = encode tokenizer "hello world" in
  let tokens = Encoding.tokens encoding |> Array.to_list in
  equal ~msg:"unigram encode tokens" (list string) [ "hello"; "world" ] tokens

let test_pad_token_set_at_construction () =
  let vocab = [ ("hello", 0); ("world", 1); ("<unk>", 2); ("[PAD]", 3) ] in
  let tokenizer =
    word_level ~vocab ~unk_token:"<unk>"
      ~pre:(Pre_tokenizer.whitespace ())
      ~specials:[ special "[PAD]" ]
      ~pad_token:"[PAD]" ()
  in
  equal ~msg:"pad token set" (option string) (Some "[PAD]")
    (pad_token tokenizer);
  let pad_id =
    match token_to_id tokenizer "[PAD]" with
    | Some id -> id
    | None -> failwith "missing pad id"
  in
  let encoding =
    encode tokenizer "hello"
      ~padding:
        {
          length = `Fixed 3;
          direction = `Right;
          pad_id = None;
          pad_type_id = None;
          pad_token = None;
        }
  in
  let ids = Encoding.ids encoding |> Array.to_list in
  let pad_ids = List.tl ids in
  equal ~msg:"pad id matches configured token" (list int) [ pad_id; pad_id ]
    pad_ids

(* Edge Cases *)

let test_tokenize_long_text () =
  let text =
    String.concat " " (List.init 1000 (fun i -> Printf.sprintf "word%d" i))
  in
  let tokens = tokenize_text text in
  equal ~msg:"long text token count" int 1000 (List.length tokens)

let test_tokenize_repeated_punctuation () =
  let tokens = tokenize_text "wow!!! really???" in
  equal ~msg:"repeated punctuation" (list string)
    [ "wow"; "!!!"; "really"; "???" ]
    tokens

let test_tokenize_mixed_whitespace () =
  let tokens = tokenize_text "hello\tworld\nthere\r\nfriend" in
  equal ~msg:"mixed whitespace" (list string)
    [ "hello"; "world"; "there"; "friend" ]
    tokens

(* Test Suite *)

let tokenization_tests =
  [
    (* Words tokenization *)
    test "tokenize words simple" test_tokenize_words_simple;
    test "tokenize words punctuation" test_tokenize_words_punctuation;
    test "tokenize words numbers" test_tokenize_words_numbers;
    test "tokenize words empty" test_tokenize_words_empty;
    test "tokenize words whitespace only" test_tokenize_words_whitespace_only;
    test "tokenize words special chars" test_tokenize_words_special_chars;
    (* Character tokenization *)
    test "tokenize chars ASCII" test_tokenize_chars_ascii;
    test "tokenize chars unicode" test_tokenize_chars_unicode;
    test "tokenize chars empty" test_tokenize_chars_empty;
    (* Regex tokenization *)
    test "tokenize regex words" test_tokenize_regex_words;
    test "tokenize regex custom" test_tokenize_regex_custom;
    test "tokenize regex no match" test_tokenize_regex_no_match;
    (* Edge cases *)
    test "tokenize long text" test_tokenize_long_text;
    test "tokenize repeated punctuation" test_tokenize_repeated_punctuation;
    test "tokenize mixed whitespace" test_tokenize_mixed_whitespace;
    (* Unigram model tests *)
    test "unigram roundtrip" test_unigram_roundtrip;
    test "unigram token_to_id out-of-vocab" test_unigram_token_to_id_oov;
    test "unigram id_to_token out-of-bounds" test_unigram_id_to_token_oob;
    test "unigram empty vocab" test_unigram_empty_vocab;
    test "unigram special tokens" test_unigram_special_tokens;
    test "unigram encode sequence" test_unigram_encode_sequence;
    test "pad token reassignment updates id" test_pad_token_set_at_construction;
  ]

let () = run "brot tokenization" [ group "tokenization" tokenization_tests ]
