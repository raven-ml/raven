(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Alcotest
open Saga_tokenizers

let test_vocab_create_empty () =
  let tokenizer = Tokenizer.word_level () in
  let vocab = Tokenizer.vocab tokenizer in
  check int "empty vocab size" 0 (List.length vocab)

let test_vocab_with_tokenizer () =
  let tokenizer = Tokenizer.word_level () in
  let vocab = Tokenizer.vocab tokenizer in
  check int "initial vocab size" 0 (List.length vocab)

let test_vocab_add_tokens () =
  let tokenizer =
    Tokenizer.add_tokens
      (Tokenizer.add_specials (Tokenizer.word_level ())
         [ Special.pad "<pad>"; Special.unk "<unk>" ])
      [ "hello"; "world" ]
  in
  let vocab_size = Tokenizer.vocab_size tokenizer in
  check bool "vocab size increased" true (vocab_size >= 2)

let test_vocab_encode_decode () =
  let tokenizer =
    Tokenizer.add_tokens
      (Tokenizer.word_level ~pre:(Pre_tokenizers.whitespace ()) ())
      [ "hello"; "world" ]
  in
  let ids = Tokenizer.encode tokenizer "hello world" |> Encoding.get_ids in
  check bool "encoded ids" true (Array.length ids > 0);
  let decoded = Tokenizer.decode tokenizer ids in
  check string "decoded text" "hello world" decoded

let test_vocab_batch_encode () =
  let tokenizer =
    Tokenizer.add_tokens (Tokenizer.word_level ()) [ "hello"; "world" ]
  in
  let encodings = Tokenizer.encode_batch tokenizer [ "hello"; "world" ] in
  check int "batch size" 2 (List.length encodings)

let test_vocab_special_tokens () =
  let tokenizer =
    Tokenizer.add_tokens
      (Tokenizer.add_specials (Tokenizer.word_level ())
         [ Special.cls "[CLS]"; Special.sep "[SEP]" ])
      [ "test" ]
  in
  let tokens =
    Tokenizer.encode ~add_special_tokens:true tokenizer "test"
    |> Encoding.get_tokens
  in
  check bool "tokens emitted" true (Array.length tokens > 0)

let test_vocab_save_load () =
  let tokenizer =
    Tokenizer.add_tokens (Tokenizer.word_level ()) [ "hello"; "world"; "test" ]
  in
  let json = Tokenizer.to_json tokenizer in
  match Tokenizer.from_json json with
  | Error exn ->
      failf "failed to round-trip tokenizer: %s" (Printexc.to_string exn)
  | Ok reloaded ->
      let original_vocab = Tokenizer.vocab tokenizer in
      let loaded_vocab = Tokenizer.vocab reloaded in
      check int "vocab size matches"
        (List.length original_vocab)
        (List.length loaded_vocab);
      List.iter
        (fun (token, _) ->
          check bool
            (Printf.sprintf "token %s preserved" token)
            true
            (Option.is_some (Tokenizer.token_to_id reloaded token)))
        original_vocab

let suite =
  [
    test_case "create empty" `Quick test_vocab_create_empty;
    test_case "with tokenizer" `Quick test_vocab_with_tokenizer;
    test_case "add tokens" `Quick test_vocab_add_tokens;
    test_case "encode decode" `Quick test_vocab_encode_decode;
    test_case "batch encode" `Quick test_vocab_batch_encode;
    test_case "special tokens" `Quick test_vocab_special_tokens;
    test_case "save load" `Quick test_vocab_save_load;
  ]

let () = run "Vocabulary tests" [ ("vocab", suite) ]
