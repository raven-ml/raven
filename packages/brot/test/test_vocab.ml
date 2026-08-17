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

(* [add_tokens] registers added tokens after the fact, exactly as passing them
   at construction would. *)
let test_vocab_add_tokens () =
  let tokenizer =
    add_tokens
      (word_level ~vocab:[ ("hello", 0); ("world", 1) ] ())
      [ added_token "<pad>"; added_token ~special:false "<x>" ]
  in
  equal ~msg:"vocab size" int 4 (vocab_size tokenizer);
  equal ~msg:"numbered after the vocabulary" (option int) (Some 2)
    (token_to_id tokenizer "<pad>");
  equal ~msg:"<x>" (option int) (Some 3) (token_to_id tokenizer "<x>");
  equal ~msg:"registered" (list string) [ "<pad>"; "<x>" ]
    (List.map (fun (a : added_token) -> a.content) (added_tokens tokenizer));
  equal ~msg:"matched atomically" (array int) [| 0; 2; 1 |]
    (Encoding.ids (encode tokenizer "hello<pad>world"));
  (* A word-level decoder joins with a space. *)
  equal ~msg:"special skipped, plain kept" string "hello <x>"
    (decode tokenizer ~skip_special_tokens:true [| 0; 2; 3 |])

(* Every model takes added tokens, not just the word-level one. *)
let test_add_tokens_any_model () =
  let tokenizer =
    add_tokens
      (bpe ~vocab:[ ("a", 0); ("b", 1) ] ~merges:[] ())
      [ added_token "<s>" ]
  in
  equal ~msg:"vocab size" int 3 (vocab_size tokenizer);
  equal ~msg:"ids" (array int) [| 0; 2; 1 |]
    (Encoding.ids (encode tokenizer "a<s>b"))

(* Added tokens never enter the model's vocabulary, so successive calls keep
   numbering densely from the end of it. *)
let test_add_tokens_twice () =
  let base = word_level ~vocab:[ ("hello", 0); ("world", 1) ] () in
  let once = add_tokens base [ added_token "<pad>" ] in
  let twice = add_tokens once [ added_token "<x>" ] in
  equal ~msg:"<pad>" (option int) (Some 2) (token_to_id twice "<pad>");
  equal ~msg:"<x>" (option int) (Some 3) (token_to_id twice "<x>");
  equal ~msg:"vocab size" int 4 (vocab_size twice);
  equal ~msg:"vocab entries" int 4 (List.length (vocab twice));
  equal ~msg:"registered once each" (list string) [ "<pad>"; "<x>" ]
    (List.map (fun (a : added_token) -> a.content) (added_tokens twice))

(* A model identifier past the end of its vocabulary does not collide with a
   synthesized one. *)
let test_add_tokens_sparse_vocab () =
  let tokenizer =
    add_tokens
      (word_level ~vocab:[ ("a", 0); ("b", 1); ("<s>", 4) ] ())
      [ added_token "<s>"; added_token "<z>" ]
  in
  equal ~msg:"<s> keeps the model id" (option int) (Some 4)
    (token_to_id tokenizer "<s>");
  equal ~msg:"<z> does not collide" (option int) (Some 5)
    (token_to_id tokenizer "<z>")

(* Content that is no text is no token and takes no identifier. *)
let test_add_tokens_empty_content () =
  let tokenizer =
    add_tokens
      (word_level ~vocab:[ ("a", 0) ] ())
      [ added_token ""; added_token "<s>" ]
  in
  equal ~msg:"one token" int 1 (List.length (added_tokens tokenizer));
  equal ~msg:"<s>" (option int) (Some 1) (token_to_id tokenizer "<s>");
  equal ~msg:"vocab size" int 2 (vocab_size tokenizer)

(* A role marker the vocabulary does not hold is still a usable special
   token. *)
let test_pad_token_absent_from_vocab () =
  let tokenizer =
    word_level
      ~vocab:[ ("a", 0) ]
      ~pad_token:"[PAD]"
      ~pre:(Pre_tokenizer.whitespace ())
      ()
  in
  equal ~msg:"[PAD]" (option int) (Some 1) (token_to_id tokenizer "[PAD]");
  equal ~msg:"vocab size" int 2 (vocab_size tokenizer);
  equal ~msg:"matched in the input" (array int) [| 0; 1 |]
    (encode_ids tokenizer "a [PAD]");
  equal ~msg:"skipped when decoding" string "a"
    (decode tokenizer ~skip_special_tokens:true [| 0; 1 |]);
  let padded =
    encode tokenizer ~padding:(padding (`Fixed 3)) "a" |> Encoding.ids
  in
  equal ~msg:"pads with it" (array int) [| 0; 1; 1 |] padded;
  match from_json (to_json tokenizer) with
  | Error msg -> failf "round trip failed: %s" msg
  | Ok reloaded ->
      equal ~msg:"survives the round trip" (option int) (Some 1)
        (token_to_id reloaded "[PAD]");
      equal ~msg:"still special" string "a"
        (decode reloaded ~skip_special_tokens:true [| 0; 1 |])

let test_vocab_encode_decode () =
  let tokenizer =
    word_level
      ~pre:(Pre_tokenizer.whitespace ())
      ~vocab:[ ("hello", 0); ("world", 1) ]
      ()
  in
  let ids = encode tokenizer "hello world" |> Encoding.ids in
  equal ~msg:"encoded ids" bool true (Array.length ids > 0);
  let decoded = decode tokenizer ids in
  equal ~msg:"decoded text" string "hello world" decoded

let test_vocab_batch_encode () =
  let tokenizer = Brot.word_level ~vocab:[ ("hello", 0); ("world", 1) ] () in
  let encodings = encode_batch tokenizer [ "hello"; "world" ] in
  equal ~msg:"batch size" int 2 (List.length encodings)

let test_vocab_special_tokens () =
  let tokenizer =
    word_level
      ~added_tokens:[ added_token "[CLS]"; added_token "[SEP]" ]
      ~vocab:[ ("test", 2) ]
      ()
  in
  let tokens =
    encode ~add_special_tokens:true tokenizer "test" |> Encoding.tokens
  in
  equal ~msg:"tokens emitted" bool true (Array.length tokens > 0)

let test_vocab_save_load () =
  let tokenizer =
    Brot.word_level ~vocab:[ ("hello", 0); ("world", 1); ("test", 2) ] ()
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
    test "add tokens to any model" test_add_tokens_any_model;
    test "add tokens twice" test_add_tokens_twice;
    test "add tokens with a sparse vocabulary" test_add_tokens_sparse_vocab;
    test "add tokens with empty content" test_add_tokens_empty_content;
    test "pad token absent from vocabulary" test_pad_token_absent_from_vocab;
    test "encode decode" test_vocab_encode_decode;
    test "batch encode" test_vocab_batch_encode;
    test "special tokens" test_vocab_special_tokens;
    test "save load" test_vocab_save_load;
  ]

let () = run "Vocabulary tests" [ group "vocab" suite ]
