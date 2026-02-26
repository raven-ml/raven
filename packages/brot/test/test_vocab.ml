(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Brot

let test_vocab_create_empty () =
  let tokenizer = word_level () in
  let vocab = vocab tokenizer in
  equal ~msg:"empty vocab size" int 0 (List.length vocab)

let test_vocab_with_tokenizer () =
  let tokenizer = word_level () in
  let vocab = vocab tokenizer in
  equal ~msg:"initial vocab size" int 0 (List.length vocab)

let test_vocab_add_tokens () =
  let tokenizer =
    add_tokens
      (word_level ~specials:[ special "<pad>"; special "<unk>" ] ())
      [ "hello"; "world" ]
  in
  let vocab_size = vocab_size tokenizer in
  equal ~msg:"vocab size increased" bool true (vocab_size >= 2)

let test_vocab_encode_decode () =
  let tokenizer =
    add_tokens
      (word_level ~pre:(Pre_tokenizer.whitespace ()) ())
      [ "hello"; "world" ]
  in
  let ids = encode tokenizer "hello world" |> Encoding.ids in
  equal ~msg:"encoded ids" bool true (Array.length ids > 0);
  let decoded = decode tokenizer ids in
  equal ~msg:"decoded text" string "hello world" decoded

let test_vocab_batch_encode () =
  let tokenizer = add_tokens (Brot.word_level ()) [ "hello"; "world" ] in
  let encodings = encode_batch tokenizer [ "hello"; "world" ] in
  equal ~msg:"batch size" int 2 (List.length encodings)

let test_vocab_special_tokens () =
  let tokenizer =
    add_tokens
      (word_level ~specials:[ special "[CLS]"; special "[SEP]" ] ())
      [ "test" ]
  in
  let tokens =
    encode ~add_special_tokens:true tokenizer "test" |> Encoding.tokens
  in
  equal ~msg:"tokens emitted" bool true (Array.length tokens > 0)

let test_vocab_save_load () =
  let tokenizer =
    add_tokens (Brot.word_level ()) [ "hello"; "world"; "test" ]
  in
  let json = to_json tokenizer in
  match from_json json with
  | Error msg -> failf "failed to round-trip tokenizer: %s" msg
  | Ok reloaded ->
      let original_vocab = vocab tokenizer in
      let loaded_vocab = vocab reloaded in
      equal ~msg:"vocab size matches" int
        (List.length original_vocab)
        (List.length loaded_vocab);
      List.iter
        (fun (token, _) ->
          equal
            ~msg:(Printf.sprintf "token %s preserved" token)
            bool true
            (Option.is_some (token_to_id reloaded token)))
        original_vocab

let suite =
  [
    test "create empty" test_vocab_create_empty;
    test "with tokenizer" test_vocab_with_tokenizer;
    test "add tokens" test_vocab_add_tokens;
    test "encode decode" test_vocab_encode_decode;
    test "batch encode" test_vocab_batch_encode;
    test "special tokens" test_vocab_special_tokens;
    test "save load" test_vocab_save_load;
  ]

let () = run "Vocabulary tests" [ group "vocab" suite ]
