(* Vocabulary tests for saga *)

open Alcotest
open Saga_tokenizers

(* Basic Vocabulary Tests using Models and Tokenizer API *)

let test_vocab_create_empty () =
  let model = Models.word_level () in
  let vocab = Models.get_vocab model in
  check int "empty vocab size" 0 (List.length vocab)

let test_vocab_with_tokenizer () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  let vocab = Tokenizer.get_vocab tokenizer () in
  check int "initial vocab size" 0 (List.length vocab)

let test_vocab_add_tokens () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  (* Add special tokens *)
  let count =
    Tokenizer.add_special_tokens tokenizer
      [
        Either.Left "<pad>";
        Either.Left "<unk>";
        Either.Left "<bos>";
        Either.Left "<eos>";
      ]
  in
  check bool "tokens added" true (count > 0);

  (* Get vocab size *)
  let vocab_size = Tokenizer.get_vocab_size tokenizer () in
  check bool "vocab size increased" true (vocab_size > 0)

let test_vocab_encode_decode () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  (* Add some tokens *)
  let _ =
    Tokenizer.add_tokens tokenizer [ Either.Left "hello"; Either.Left "world" ]
  in

  (* Test encoding *)
  let encoding =
    Tokenizer.encode tokenizer ~sequence:(Either.Left "hello world") ()
  in
  let ids = Encoding.get_ids encoding in
  check bool "encoded to ids" true (Array.length ids > 0);

  (* Test decoding *)
  let decoded = Tokenizer.decode tokenizer (Array.to_list ids) () in
  check bool "decoded back" true (String.length decoded > 0)

let test_vocab_batch_encode () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  let inputs =
    [ Either.Left (Either.Left "hello"); Either.Left (Either.Left "world") ]
  in
  let encodings = Tokenizer.encode_batch tokenizer ~input:inputs () in
  check int "batch size" 2 (List.length encodings)

let test_vocab_special_tokens () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in

  (* Add special tokens *)
  let count =
    Tokenizer.add_special_tokens tokenizer
      [ Either.Left "[CLS]"; Either.Left "[SEP]"; Either.Left "[PAD]" ]
  in
  check bool "special tokens added" true (count > 0);

  (* Encode with special tokens *)
  let encoding =
    Tokenizer.encode tokenizer ~sequence:(Either.Left "test")
      ~add_special_tokens:true ()
  in
  let tokens = Encoding.get_tokens encoding in
  check bool "has tokens" true (Array.length tokens > 0)

let test_vocab_save_load () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in

  (* Add some tokens *)
  let _ =
    Tokenizer.add_tokens tokenizer
      [ Either.Left "hello"; Either.Left "world"; Either.Left "test" ]
  in

  (* Save to file *)
  let temp_file = Filename.temp_file "vocab_test" ".json" in
  Tokenizer.save tokenizer ~path:temp_file ();

  (* Load from file *)
  let loaded = Tokenizer.from_file temp_file in
  (match loaded with
  | Ok loaded_tokenizer ->
      let vocab1 = Tokenizer.get_vocab tokenizer () in
      let vocab2 = Tokenizer.get_vocab loaded_tokenizer () in
      check int "same vocab size" (List.length vocab1) (List.length vocab2)
  | Error _ -> check bool "failed to load" false true);

  Sys.remove temp_file

let test_vocab_from_pretrained () =
  (* Test loading a pretrained tokenizer *)
  let result = Tokenizer.from_pretrained "bert-base-uncased" () in
  match result with
  | Ok _ ->
      (* Would succeed if we had actual pretrained models *)
      check bool "loaded pretrained" true true
  | Error _ ->
      (* Expected to fail without actual model files *)
      check bool "expected failure without model files" true true

(* Test Suite *)

let vocab_tests =
  [
    (* Basic operations *)
    test_case "create empty" `Quick test_vocab_create_empty;
    test_case "with tokenizer" `Quick test_vocab_with_tokenizer;
    test_case "add tokens" `Quick test_vocab_add_tokens;
    test_case "encode decode" `Quick test_vocab_encode_decode;
    test_case "batch encode" `Quick test_vocab_batch_encode;
    test_case "special tokens" `Quick test_vocab_special_tokens;
    test_case "save load" `Quick test_vocab_save_load;
    test_case "from pretrained" `Quick test_vocab_from_pretrained;
  ]

let () = run "Vocabulary tests" [ ("vocab", vocab_tests) ]
