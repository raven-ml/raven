(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Brot

let test_bpe_basic () =
  (* Create a simple vocabulary and merges *)
  let vocab =
    [
      ("h", 0);
      ("e", 1);
      ("l", 2);
      ("o", 3);
      ("ll", 4);
      ("he", 5);
      ("llo", 6);
      ("hello", 7);
    ]
  in

  let merges =
    [
      ("l", "l");
      (* rank 0: Merge 'l' + 'l' -> 'll' *)
      ("ll", "o");
      (* rank 1: Merge 'll' + 'o' -> 'llo' *)
      ("he", "llo");
      (* rank 2: Merge 'he' + 'llo' -> 'hello' *)
    ]
  in

  let tokenizer = bpe ~vocab ~merges ~unk_token:"<unk>" () in

  let encoding = encode tokenizer "hello" in
  let tokens = Encoding.tokens encoding |> Array.to_list in

  Printf.printf "Tokenized 'hello': ";
  List.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  equal ~msg:"vocabulary size" int 8 (vocab_size tokenizer)

let test_bpe_builder () =
  let vocab = [ ("a", 0); ("b", 1); ("ab", 2) ] in
  let merges = [ ("a", "b") ] in

  let tokenizer = bpe ~vocab ~merges ~cache_capacity:50 () in

  let encoding = encode tokenizer "ab" in
  let tokens = Encoding.tokens encoding in
  equal ~msg:"single token for 'ab'" int 1 (Array.length tokens)

let test_ignore_merges () =
  (* ["ab"] is in the vocabulary but no merge builds it, so the merges alone
     cannot produce it; [ignore_merges] is what makes the whole word win. *)
  let vocab = [ ("a", 0); ("b", 1); ("c", 2); ("bc", 3); ("ab", 4) ] in
  let merges = [ ("b", "c") ] in
  let tokens tokenizer text =
    encode tokenizer text |> Encoding.tokens |> Array.to_list
  in
  let merging = bpe ~vocab ~merges () in
  let ignoring = bpe ~vocab ~merges ~ignore_merges:true () in
  equal ~msg:"merges decide 'ab'" (list string) [ "a"; "b" ]
    (tokens merging "ab");
  equal ~msg:"ignore_merges keeps 'ab'" (list string) [ "ab" ]
    (tokens ignoring "ab");
  (* A word absent from the vocabulary is merged either way. *)
  equal ~msg:"merges build 'abc'" (list string) [ "a"; "bc" ]
    (tokens merging "abc");
  equal ~msg:"ignore_merges still merges 'abc'" (list string) [ "a"; "bc" ]
    (tokens ignoring "abc")

let test_dropout_overrides_ignore_merges () =
  (* Dropout is drawn per occurrence, so the whole-word shortcut cannot stand in
     for the merges; at probability 1 none of them apply. *)
  let vocab = [ ("a", 0); ("b", 1); ("ab", 2) ] in
  let merges = [ ("a", "b") ] in
  let tokenizer = bpe ~vocab ~merges ~ignore_merges:true ~dropout:1.0 () in
  equal ~msg:"dropout leaves 'ab' unmerged" (list string) [ "a"; "b" ]
    (encode tokenizer "ab" |> Encoding.tokens |> Array.to_list)

let test_empty_affixes () =
  (* Tokenizer files spell "no prefix" and "no suffix" as [""]. *)
  let vocab = [ ("a", 0); ("b", 1); ("ab", 2) ] in
  let merges = [ ("a", "b") ] in
  let tokenizer =
    bpe ~vocab ~merges ~continuing_subword_prefix:"" ~end_of_word_suffix:"" ()
  in
  equal ~msg:"'ab' merges with empty affixes" (list string) [ "ab" ]
    (encode tokenizer "ab" |> Encoding.tokens |> Array.to_list)

let test_bpe_save_load () =
  let vocab = [ ("t", 0); ("e", 1); ("s", 2); ("test", 3) ] in
  let merges = [] in
  (* No merges for simplicity *)

  let tokenizer = bpe ~vocab ~merges () in

  (* Save the model *)
  let temp_dir = Filename.temp_dir "bpe_test" "" in
  let files = save_model_files tokenizer ~folder:temp_dir () in

  (* Load the model *)
  let vocab_file = List.find (fun f -> Filename.check_suffix f ".json") files in
  let merges_file = List.find (fun f -> Filename.check_suffix f ".txt") files in
  let loaded_tokenizer =
    from_model_file ~vocab:vocab_file ~merges:merges_file ()
  in

  (* Test that loaded tokenizer works the same *)
  let original_tokens = encode tokenizer "test" |> Encoding.tokens in
  let loaded_tokens = encode loaded_tokenizer "test" |> Encoding.tokens in

  equal ~msg:"same number of tokens" int
    (Array.length original_tokens)
    (Array.length loaded_tokens);

  (* Clean up *)
  List.iter Sys.remove files;
  Unix.rmdir temp_dir

let test_tokenizer_integration () =
  (* Create a BPE tokenizer using the high-level API *)
  let vocab =
    [
      ("h", 0); ("e", 1); ("l", 2); ("o", 3); ("he", 4); ("llo", 5); ("hello", 6);
    ]
  in
  let merges = [ ("h", "e"); ("he", "llo") ] in
  let tokenizer = bpe ~vocab ~merges () in

  (* Test encoding *)
  let tokens = encode tokenizer "hello" |> Encoding.tokens |> Array.to_list in

  Printf.printf "bpe result: ";
  List.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  equal ~msg:"tokenizer produces output" bool true (List.length tokens > 0)

(* Without an unknown token or byte fallback a character absent from the
   vocabulary contributes no symbol, so a word can reach the merges empty. *)
let test_unknown_character () =
  let vocab = [ ("a", 0); ("b", 1); ("ab", 2) ] in
  let merges = [ ("a", "b") ] in
  let tokenizer = bpe ~vocab ~merges () in
  let tokens text = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"unknown character alone" (list string) [] (tokens "z");
  equal ~msg:"merged word" (list string) [ "ab" ] (tokens "ab");
  (* The merge buffers are reused, so an empty word must not pick up the symbols
     of the word merged before it. *)
  equal ~msg:"unknown character after a merged word" (list string) []
    (tokens "z");
  equal ~msg:"known character before an unknown one" (list string) [ "a" ]
    (tokens "az")

(* The word cache is shared by every domain encoding with the model. These 169
   words share the 16 slots of the smallest cache, so the domains rewrite the
   same slots continuously: a slot whose key and word are published by two
   separate stores then hands out the word of another key. *)
let test_parallel_cache () =
  let letter i = Char.chr (Char.code 'a' + i) in
  let vocab =
    List.init 26 (fun i -> (String.make 1 (letter i), i))
    @ List.init 26 (fun i -> (Printf.sprintf "a%c" (letter i), 26 + i))
  in
  let merges = List.init 26 (fun i -> ("a", String.make 1 (letter i))) in
  let tokenizer = bpe ~vocab ~merges ~cache_capacity:1 () in
  let words =
    Array.init 169 (fun k ->
        Printf.sprintf "a%ca%c" (letter (k / 13)) (letter (k mod 13)))
  in
  let ids text = encode tokenizer text |> Encoding.ids in
  let expected = Array.map ids words in
  let mismatches = Atomic.make 0 in
  let hammer () =
    for _ = 1 to 100 do
      Array.iteri
        (fun i word -> if ids word <> expected.(i) then Atomic.incr mismatches)
        words
    done
  in
  let domains = Array.init 3 (fun _ -> Domain.spawn hammer) in
  hammer ();
  Array.iter Domain.join domains;
  equal ~msg:"parallel tokenization agrees with single-domain" int 0
    (Atomic.get mismatches)

let () =
  run "BPE tests"
    [
      group "basic"
        [
          test "basic tokenization" test_bpe_basic;
          test "builder pattern" test_bpe_builder;
          test "ignore_merges" test_ignore_merges;
          test "dropout overrides ignore_merges"
            test_dropout_overrides_ignore_merges;
          test "empty prefix and suffix" test_empty_affixes;
          test "save and load" test_bpe_save_load;
          test "tokenizer integration" test_tokenizer_integration;
          test "unknown character" test_unknown_character;
        ];
      group "parallel"
        [ test "shared cache across domains" test_parallel_cache ];
    ]
