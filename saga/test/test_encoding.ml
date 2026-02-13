(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Saga

let make_word_tokenizer ?(specials = []) () =
  Tokenizer.word_level ~pre:(Pre_tokenizers.whitespace ()) ~specials ()

let test_encode_simple () =
  let tokenizer =
    Tokenizer.add_tokens (make_word_tokenizer ()) [ "hello"; "world" ]
  in
  let ids =
    Tokenizer.encode tokenizer "hello world hello" |> Encoding.get_ids
  in
  equal ~msg:"encoded length" int 3 (Array.length ids);
  equal ~msg:"repeated token same id" bool true (ids.(0) = ids.(2))

let test_encode_with_vocab () =
  let tokenizer =
    Tokenizer.add_tokens (make_word_tokenizer ()) [ "hello"; "world" ]
  in
  let ids =
    Tokenizer.encode tokenizer "hello world"
    |> Encoding.get_ids |> Array.to_list
  in
  equal ~msg:"encoded with vocab" (list int) [ 0; 1 ] ids

let test_encode_unknown_tokens () =
  let tokenizer =
    Tokenizer.add_tokens
      (make_word_tokenizer ~specials:[ Special.unk "<unk>" ] ())
      [ "hello" ]
  in
  let ids =
    Tokenizer.encode tokenizer "hello unknown world"
    |> Encoding.get_ids |> Array.to_list
  in
  equal ~msg:"encoded something" bool true (List.length ids > 0)

let test_encode_empty () =
  let tokenizer = make_word_tokenizer () in
  let ids =
    Tokenizer.encode tokenizer "" |> Encoding.get_ids |> Array.to_list
  in
  equal ~msg:"encode empty" (list int) [] ids

let test_encode_batch_simple () =
  let tokenizer =
    Tokenizer.add_tokens (make_word_tokenizer ())
      [ "hello"; "world"; "hi"; "there" ]
  in
  let encodings =
    Tokenizer.encode_batch tokenizer [ "hello world"; "hi there" ]
  in
  equal ~msg:"batch size" int 2 (List.length encodings);
  let first = List.hd encodings in
  equal ~msg:"first encoding has ids" bool true
    (Array.length (Encoding.get_ids first) > 0)

let test_encode_batch_with_padding () =
  let tokenizer =
    Tokenizer.add_tokens
      (make_word_tokenizer ~specials:[ Special.pad "<pad>" ] ())
      [ "hello"; "world"; "hi"; "there" ]
  in
  let padding =
    {
      length = `Fixed 5;
      direction = `Right;
      pad_id = None;
      pad_type_id = None;
      pad_token = Some "<pad>";
    }
  in
  let encodings =
    Tokenizer.encode_batch tokenizer ~padding [ "hello"; "hi there" ]
  in
  let first = Encoding.get_ids (List.nth encodings 0) in
  let second = Encoding.get_ids (List.nth encodings 1) in
  equal ~msg:"first padded length" int 5 (Array.length first);
  equal ~msg:"second padded length" int 5 (Array.length second)

let test_encode_batch_empty () =
  let tokenizer = make_word_tokenizer () in
  let encodings = Tokenizer.encode_batch tokenizer [] in
  equal ~msg:"empty batch" int 0 (List.length encodings)

let test_decode_simple () =
  let tokenizer =
    Tokenizer.add_tokens (make_word_tokenizer ()) [ "hello"; "world" ]
  in
  let decoded = Tokenizer.decode tokenizer [| 0; 1 |] in
  equal ~msg:"decoded text" string "hello world" decoded

let test_decode_with_special () =
  let tokenizer =
    Tokenizer.add_tokens
      (make_word_tokenizer
         ~specials:[ Special.bos "<bos>"; Special.eos "<eos>" ]
         ())
      [ "hello" ]
  in
  let decoded = Tokenizer.decode tokenizer [| 1; 2; 0 |] in
  equal ~msg:"decoded with special" string "<bos> hello <eos>" decoded

let test_decode_skip_special () =
  let tokenizer =
    Tokenizer.add_tokens
      (make_word_tokenizer
         ~specials:[ Special.bos "<bos>"; Special.eos "<eos>" ]
         ())
      [ "hello" ]
  in
  let decoded =
    Tokenizer.decode ~skip_special_tokens:true tokenizer [| 1; 2; 0 |]
  in
  equal ~msg:"decoded without special" string "hello" decoded

let test_decode_batch () =
  let tokenizer =
    Tokenizer.add_tokens (make_word_tokenizer ())
      [ "hello"; "world"; "hi"; "there" ]
  in
  let decoded = Tokenizer.decode_batch tokenizer [ [| 0; 1 |]; [| 2; 3 |] ] in
  equal ~msg:"decoded count" int 2 (List.length decoded);
  equal ~msg:"first decoded" string "hello world" (List.nth decoded 0);
  equal ~msg:"second decoded" string "hi there" (List.nth decoded 1)

let test_chars_model () =
  let tokenizer = Tokenizer.chars () in
  let ids =
    Tokenizer.encode tokenizer "abc" |> Encoding.get_ids |> Array.to_list
  in
  equal ~msg:"char ids" (list int) [ 97; 98; 99 ] ids

let suite =
  [
    test "encode simple" test_encode_simple;
    test "encode with vocab" test_encode_with_vocab;
    test "encode unknown tokens" test_encode_unknown_tokens;
    test "encode empty" test_encode_empty;
    test "batch simple" test_encode_batch_simple;
    test "batch with padding" test_encode_batch_with_padding;
    test "batch empty request" test_encode_batch_empty;
    test "decode simple" test_decode_simple;
    test "decode with special" test_decode_with_special;
    test "decode skip special" test_decode_skip_special;
    test "decode batch" test_decode_batch;
    test "chars model" test_chars_model;
  ]

let () = run "Encoding tests" [ group "encoding" suite ]
