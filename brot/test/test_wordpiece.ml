(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Brot

let test_wordpiece_basic () =
  (* Create a simple vocabulary *)
  let vocab =
    [
      ("[UNK]", 0);
      ("hello", 1);
      ("world", 2);
      ("##llo", 3);
      ("##rld", 4);
      ("he", 5);
      ("wo", 6);
    ]
  in

  let tokenizer =
    wordpiece ~vocab ~unk_token:"[UNK]" ~continuing_subword_prefix:"##" ()
  in

  (* Test tokenizing a known word *)
  let encoding = encode tokenizer "hello" in
  let tokens = Encoding.tokens encoding in
  equal ~msg:"single token for 'hello'" int 1 (Array.length tokens);
  equal ~msg:"token value" string "hello" tokens.(0);

  Printf.printf "Tokenized 'hello': ";
  Array.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  equal ~msg:"vocabulary size" int 7 (vocab_size tokenizer)

let test_wordpiece_subwords () =
  (* Create vocabulary with subword pieces *)
  let vocab =
    [
      ("[UNK]", 0);
      ("un", 1);
      ("##able", 2);
      ("##happy", 3);
      ("play", 4);
      ("##ing", 5);
      ("##ed", 6);
    ]
  in

  let tokenizer = wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Test word that can be split into subwords *)
  let encoding = encode tokenizer "playing" in
  let tokens = Encoding.tokens encoding in
  Printf.printf "Tokenized 'playing': ";
  Array.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  equal ~msg:"should split into subwords" int 2 (Array.length tokens);
  equal ~msg:"first token" string "play" tokens.(0);
  equal ~msg:"second token" string "##ing" tokens.(1)

let test_wordpiece_unknown () =
  (* Create minimal vocabulary *)
  let vocab = [ ("[UNK]", 0); ("hello", 1) ] in

  let tokenizer = wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Test unknown word *)
  let encoding = encode tokenizer "goodbye" in
  let tokens = Encoding.tokens encoding in
  equal ~msg:"unknown word becomes single token" int 1 (Array.length tokens);
  equal ~msg:"unknown token" string "[UNK]" tokens.(0)

let test_wordpiece_max_chars () =
  (* Create vocabulary *)
  let vocab = [ ("[UNK]", 0); ("test", 1) ] in

  let tokenizer =
    wordpiece ~vocab ~unk_token:"[UNK]" ~max_input_chars_per_word:5 ()
  in

  (* Test word exceeding max chars *)
  let long_word = String.make 10 'a' in
  let encoding = encode tokenizer long_word in
  let tokens = Encoding.tokens encoding in
  equal ~msg:"long word becomes unknown" int 1 (Array.length tokens);
  equal ~msg:"unknown token" string "[UNK]" tokens.(0)

let test_wordpiece_save_load () =
  (* Create vocabulary *)
  let vocab =
    [
      ("[PAD]", 0);
      ("[UNK]", 1);
      ("[CLS]", 2);
      ("[SEP]", 3);
      ("hello", 4);
      ("world", 5);
    ]
  in

  let tokenizer = wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Save the model *)
  let temp_dir = Filename.temp_dir "wordpiece_test" "" in
  let files = save_model_files tokenizer ~folder:temp_dir () in

  (* Load the model *)
  let vocab_file = List.find (fun f -> Filename.check_suffix f ".txt") files in
  let loaded_tokenizer = from_model_file ~vocab:vocab_file () in

  (* Test that loaded tokenizer works the same *)
  let original_tokens = encode tokenizer "hello" |> Encoding.tokens in
  let loaded_tokens = encode loaded_tokenizer "hello" |> Encoding.tokens in

  equal ~msg:"same number of tokens" int
    (Array.length original_tokens)
    (Array.length loaded_tokens);

  (* Clean up *)
  List.iter Sys.remove files;
  Unix.rmdir temp_dir

let test_tokenizer_integration () =
  (* Create a WordPiece tokenizer using the high-level API *)
  let vocab =
    [
      ("[PAD]", 0);
      ("[UNK]", 1);
      ("[CLS]", 2);
      ("[SEP]", 3);
      ("hello", 4);
      ("world", 5);
      ("##ing", 6);
    ]
  in
  let tokenizer = wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Test encoding *)
  let tokens = encode tokenizer "hello" |> Encoding.tokens |> Array.to_list in

  Printf.printf "wordpiece result: ";
  List.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  equal ~msg:"tokenizer produces output" bool true (List.length tokens > 0)

let test_wordpiece_greedy_matching () =
  (* Test the greedy longest-match-first algorithm *)
  let vocab =
    [
      ("[UNK]", 0);
      ("un", 1);
      ("able", 2);
      ("unable", 3);
      (* Longer match should be preferred *)
      ("##able", 4);
    ]
  in

  let tokenizer = wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Should match "unable" as a single token, not "un" + "##able" *)
  let encoding = encode tokenizer "unable" in
  let tokens = Encoding.tokens encoding in
  equal ~msg:"greedy match finds longest token" int 1 (Array.length tokens);
  equal ~msg:"matched full word" string "unable" tokens.(0)

let () =
  run "WordPiece tests"
    [
      group "basic"
        [
          test "basic tokenization" test_wordpiece_basic;
          test "subword tokenization" test_wordpiece_subwords;
          test "unknown tokens" test_wordpiece_unknown;
          test "max input chars" test_wordpiece_max_chars;
          test "save and load" test_wordpiece_save_load;
          test "tokenizer integration" test_tokenizer_integration;
          test "greedy matching" test_wordpiece_greedy_matching;
        ];
    ]
