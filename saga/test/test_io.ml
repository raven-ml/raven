(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Saga_tokenizers

let test_roundtrip_wordpiece () =
  let tok =
    Tokenizer.wordpiece
      ~vocab:
        [
          ("[PAD]", 0);
          ("[UNK]", 1);
          ("[CLS]", 2);
          ("[SEP]", 3);
          ("hello", 4);
          ("world", 5);
          ("##ing", 6);
        ]
      ~unk_token:"[UNK]" ~continuing_subword_prefix:"##" ()
  in
  let temp_dir = Filename.get_temp_dir_name () in
  let save_path = Filename.concat temp_dir "test_wordpiece_tokenizer" in
  Tokenizer.save_pretrained tok ~path:save_path;
  let tokenizer_file = Filename.concat save_path "tokenizer.json" in
  match Tokenizer.from_file tokenizer_file with
  | Ok loaded_tok ->
      let enc1 = Tokenizer.encode tok "hello world" in
      let enc2 = Tokenizer.encode loaded_tok "hello world" in
      let ids1 = Encoding.get_ids enc1 in
      let ids2 = Encoding.get_ids enc2 in
      Alcotest.(check (array int)) "Same encoding" ids1 ids2;
      Sys.remove tokenizer_file;
      Unix.rmdir save_path
  | Error e ->
      Alcotest.failf "Failed to reload tokenizer: %s" (Printexc.to_string e)

let test_roundtrip_bpe () =
  let tok =
    Tokenizer.bpe
      ~vocab:[ ("h", 0); ("e", 1); ("l", 2); ("o", 3); ("he", 4) ]
      ~merges:[ ("h", "e") ]
      ()
  in
  let temp_dir = Filename.get_temp_dir_name () in
  let save_path = Filename.concat temp_dir "test_bpe_tokenizer" in
  Tokenizer.save_pretrained tok ~path:save_path;
  let tokenizer_file = Filename.concat save_path "tokenizer.json" in
  match Tokenizer.from_file tokenizer_file with
  | Ok loaded_tok ->
      let vocab1 = Tokenizer.vocab tok |> List.sort compare in
      let vocab2 = Tokenizer.vocab loaded_tok |> List.sort compare in
      Alcotest.(check (list (pair string int))) "Same vocab" vocab1 vocab2;
      Sys.remove tokenizer_file;
      Unix.rmdir save_path
  | Error e ->
      Alcotest.failf "Failed to reload tokenizer: %s" (Printexc.to_string e)

let test_roundtrip_wordlevel () =
  let tok =
    Tokenizer.word_level
      ~vocab:[ ("hello", 0); ("world", 1); ("[UNK]", 2) ]
      ~unk_token:"[UNK]" ()
  in
  let temp_dir = Filename.get_temp_dir_name () in
  let save_path = Filename.concat temp_dir "test_wordlevel_tokenizer" in
  Tokenizer.save_pretrained tok ~path:save_path;
  let tokenizer_file = Filename.concat save_path "tokenizer.json" in
  match Tokenizer.from_file tokenizer_file with
  | Ok loaded_tok ->
      let vocab1 = Tokenizer.vocab tok |> List.sort compare in
      let vocab2 = Tokenizer.vocab loaded_tok |> List.sort compare in
      Alcotest.(check (list (pair string int))) "Same vocab" vocab1 vocab2;
      Sys.remove tokenizer_file;
      Unix.rmdir save_path
  | Error e ->
      Alcotest.failf "Failed to reload tokenizer: %s" (Printexc.to_string e)

let () =
  Alcotest.run "Tokenizer I/O"
    [
      ( "roundtrip",
        [
          Alcotest.test_case "wordpiece" `Quick test_roundtrip_wordpiece;
          Alcotest.test_case "bpe" `Quick test_roundtrip_bpe;
          Alcotest.test_case "wordlevel" `Quick test_roundtrip_wordlevel;
        ] );
    ]
