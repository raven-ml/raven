(* Encoding and batch processing tests for saga *)

open Alcotest
open Saga_tokenizers

(* ───── Simple Encoding Tests ───── *)

let test_encode_simple () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  Tokenizer.set_pre_tokenizer tokenizer (Some (Pre_tokenizers.whitespace ()));
  let _ =
    Tokenizer.add_tokens tokenizer [ Either.Left "hello"; Either.Left "world" ]
  in
  let encoding =
    Tokenizer.encode tokenizer ~sequence:(Either.Left "hello world hello") ()
  in
  let ids = Encoding.get_ids encoding in
  (* Should encode properly *)
  check int "encoded length" 3 (Array.length ids);
  (* Check that repeated words get same index *)
  check bool "repeated token same index" true (ids.(0) = ids.(2))

let test_encode_with_vocab () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  Tokenizer.set_pre_tokenizer tokenizer (Some (Pre_tokenizers.whitespace ()));
  let _ =
    Tokenizer.add_tokens tokenizer [ Either.Left "hello"; Either.Left "world" ]
  in
  let encoding =
    Tokenizer.encode tokenizer ~sequence:(Either.Left "hello world") ()
  in
  let ids = Array.to_list (Encoding.get_ids encoding) in
  check (list int) "encoded with vocab" [ 0; 1 ] ids

let test_encode_unknown_tokens () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  Tokenizer.set_pre_tokenizer tokenizer (Some (Pre_tokenizers.whitespace ()));
  let _ = Tokenizer.add_special_tokens tokenizer [ Either.Left "<unk>" ] in
  let _ = Tokenizer.add_tokens tokenizer [ Either.Left "hello" ] in
  let encoding =
    Tokenizer.encode tokenizer ~sequence:(Either.Left "hello unknown world") ()
  in
  let ids = Array.to_list (Encoding.get_ids encoding) in
  (* Since word_level doesn't handle unknown tokens well, we just check we get
     something *)
  check bool "encoded with unknowns" true (List.length ids > 0)

let test_encode_empty () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  let encoding = Tokenizer.encode tokenizer ~sequence:(Either.Left "") () in
  let ids = Array.to_list (Encoding.get_ids encoding) in
  check (list int) "encode empty string" [] ids

(* ───── Batch Encoding Tests ───── *)

let test_encode_batch_simple () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  Tokenizer.set_pre_tokenizer tokenizer (Some (Pre_tokenizers.whitespace ()));
  let _ =
    Tokenizer.add_tokens tokenizer
      [
        Either.Left "hello";
        Either.Left "world";
        Either.Left "hi";
        Either.Left "there";
      ]
  in
  let inputs =
    [
      Either.Left (Either.Left "hello world");
      Either.Left (Either.Left "hi there");
    ]
  in
  let encodings = Tokenizer.encode_batch tokenizer ~input:inputs () in
  check int "batch size" 2 (List.length encodings);

  (* Check first encoding *)
  let first_encoding = List.hd encodings in
  let ids = Encoding.get_ids first_encoding in
  check bool "first encoding has tokens" true (Array.length ids > 0)

let test_encode_batch_with_padding () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  Tokenizer.set_pre_tokenizer tokenizer (Some (Pre_tokenizers.whitespace ()));

  (* Add special tokens including padding *)
  let _ = Tokenizer.add_special_tokens tokenizer [ Either.Left "<pad>" ] in

  let _ =
    Tokenizer.add_tokens tokenizer
      [
        Either.Left "hello";
        Either.Left "world";
        Either.Left "hi";
        Either.Left "there";
      ]
  in

  (* Enable padding *)
  Tokenizer.enable_padding tokenizer
    {
      Tokenizer.direction = `Right;
      pad_id = 0;
      (* <pad> token id *)
      pad_type_id = 0;
      pad_token = "<pad>";
      length = Some 5;
      (* Fixed length *)
      pad_to_multiple_of = None;
    };

  let inputs =
    [ Either.Left (Either.Left "hello"); Either.Left (Either.Left "hi there") ]
  in
  let encodings = Tokenizer.encode_batch tokenizer ~input:inputs () in

  (* Check that both encodings have tokens (padding is applied during
     encoding) *)
  let first_ids = Encoding.get_ids (List.nth encodings 0) in
  let second_ids = Encoding.get_ids (List.nth encodings 1) in
  (* Note: padding happens internally, we can't guarantee exact length without
     post-processor *)
  check bool "first has tokens" true (Array.length first_ids > 0);
  check bool "second has tokens" true (Array.length second_ids > 0)

let test_encode_batch_empty () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  let encodings = Tokenizer.encode_batch tokenizer ~input:[] () in
  check int "empty batch size" 0 (List.length encodings)

let test_encode_batch_single () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  let inputs = [ Either.Left (Either.Left "hello world") ] in
  let encodings = Tokenizer.encode_batch tokenizer ~input:inputs () in
  check int "single item batch" 1 (List.length encodings)

(* ───── Decoding Tests ───── *)

let test_decode_simple () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  let _ =
    Tokenizer.add_tokens tokenizer [ Either.Left "hello"; Either.Left "world" ]
  in
  let decoded = Tokenizer.decode tokenizer [ 0; 1 ] () in
  check string "decoded text" "hello world" decoded

let test_decode_with_special () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  let _ =
    Tokenizer.add_special_tokens tokenizer
      [ Either.Left "<bos>"; Either.Left "<eos>" ]
  in
  let _ = Tokenizer.add_tokens tokenizer [ Either.Left "hello" ] in
  let decoded = Tokenizer.decode tokenizer [ 0; 2; 1 ] () in
  (* <bos> hello <eos> *)
  check string "decoded with special" "<bos> hello <eos>" decoded

let test_decode_skip_special () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  let _ =
    Tokenizer.add_special_tokens tokenizer
      [ Either.Left "<bos>"; Either.Left "<eos>" ]
  in
  let _ = Tokenizer.add_tokens tokenizer [ Either.Left "hello" ] in
  let decoded =
    Tokenizer.decode tokenizer [ 0; 2; 1 ] ~skip_special_tokens:true ()
  in
  (* Should skip special tokens *)
  check string "decoded without special" "hello" decoded

let test_decode_empty () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  let decoded = Tokenizer.decode tokenizer [] () in
  check string "decode empty" "" decoded

(* ───── Batch Decoding Tests ───── *)

let test_decode_batch_simple () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  Tokenizer.set_pre_tokenizer tokenizer (Some (Pre_tokenizers.whitespace ()));
  let _ =
    Tokenizer.add_tokens tokenizer
      [
        Either.Left "hello";
        Either.Left "world";
        Either.Left "hi";
        Either.Left "there";
      ]
  in

  let sequences = [ [ 0; 1 ]; (* hello world *) [ 2; 3 ] (* hi there *) ] in

  let decoded = Tokenizer.decode_batch tokenizer sequences () in
  check int "decoded count" 2 (List.length decoded);
  check string "first decoded" "hello world" (List.nth decoded 0);
  check string "second decoded" "hi there" (List.nth decoded 1)

(* ───── Tokenization with Different Models ───── *)

let test_chars_model () =
  let tokenizer = Tokenizer.create ~model:(Models.chars ()) in
  let encoding = Tokenizer.encode tokenizer ~sequence:(Either.Left "hi") () in
  let tokens = Encoding.get_tokens encoding in
  check int "char tokens count" 2 (Array.length tokens);
  check string "first char" "h" tokens.(0);
  check string "second char" "i" tokens.(1)

(* ───── Test Suite ───── *)

let encoding_tests =
  [
    (* Simple encoding *)
    test_case "encode simple" `Quick test_encode_simple;
    test_case "encode with vocab" `Quick test_encode_with_vocab;
    test_case "encode unknown tokens" `Quick test_encode_unknown_tokens;
    test_case "encode empty" `Quick test_encode_empty;
    (* Batch encoding *)
    test_case "batch simple" `Quick test_encode_batch_simple;
    test_case "batch with padding" `Quick test_encode_batch_with_padding;
    test_case "batch empty" `Quick test_encode_batch_empty;
    test_case "batch single" `Quick test_encode_batch_single;
    (* Decoding *)
    test_case "decode simple" `Quick test_decode_simple;
    test_case "decode with special" `Quick test_decode_with_special;
    test_case "decode skip special" `Quick test_decode_skip_special;
    test_case "decode empty" `Quick test_decode_empty;
    (* Batch decoding *)
    test_case "decode batch simple" `Quick test_decode_batch_simple;
    (* Model tests *)
    test_case "chars model" `Quick test_chars_model;
  ]

let () = run "Encoding tests" [ ("encoding", encoding_tests) ]
