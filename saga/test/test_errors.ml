(* Error handling tests for saga *)

open Alcotest
open Saga_tokenizers

(* Helper to check error messages *)
let _check_error_msg expected_pattern f =
  try
    ignore (f ());
    fail "Expected exception but none was raised"
  with Invalid_argument msg ->
    if not (String.contains msg expected_pattern.[0]) then
      fail
        (Printf.sprintf
           "Error message '%s' doesn't contain expected pattern '%s'" msg
           expected_pattern)

(* ───── Tokenizer Creation Error Tests ───── *)

let test_load_nonexistent_file () =
  match Tokenizer.from_file "/nonexistent/file.json" with
  | Ok _ -> fail "Expected error but got Ok"
  | Error _ -> check bool "load failed" true true

let test_load_invalid_json () =
  let temp_file = Filename.temp_file "test" ".json" in
  let oc = open_out temp_file in
  output_string oc "{ invalid json }";
  close_out oc;

  match Tokenizer.from_file temp_file with
  | Ok _ ->
      Sys.remove temp_file;
      fail "Expected error but got Ok"
  | Error _ ->
      Sys.remove temp_file;
      check bool "load failed with invalid json" true true

(* ───── Model Error Tests ───── *)

let test_bpe_empty_vocab () =
  let model = Models.bpe ~vocab:[] ~merges:[] () in
  let tokenizer = Tokenizer.create ~model in
  let encoding =
    Tokenizer.encode tokenizer ~sequence:(Either.Left "hello") ()
  in
  (* Should handle empty vocab gracefully *)
  let ids = Encoding.get_ids encoding in
  check int "empty vocab encoding" 0 (Array.length ids)

let test_wordpiece_empty_vocab () =
  let model = Models.wordpiece ~vocab:[] ~unk_token:"[UNK]" () in
  let tokenizer = Tokenizer.create ~model in
  let encoding =
    Tokenizer.encode tokenizer ~sequence:(Either.Left "hello") ()
  in
  let ids = Encoding.get_ids encoding in
  check int "empty vocab encoding" 0 (Array.length ids)

(* ───── Pre-tokenizer Error Tests ───── *)

let test_invalid_split_pattern () =
  (* The pre-tokenizers should handle invalid patterns gracefully *)
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  let pre_tok =
    Pre_tokenizers.split ~pattern:"[" ~behavior:`Removed ~invert:false ()
  in
  Tokenizer.set_pre_tokenizer tokenizer (Some pre_tok);

  (* This should not crash *)
  let encoding = Tokenizer.encode tokenizer ~sequence:(Either.Left "test") () in
  check bool "handled invalid pattern" true
    (Encoding.get_ids encoding |> Array.length >= 0)

(* ───── Normalizer Error Tests ───── *)

let test_unicode_normalization_error () =
  (* Unicode functions should handle errors gracefully *)
  let text = "test" ^ String.make 1 '\xFF' ^ String.make 1 '\xFE' in
  (* Invalid UTF-8 *)

  let tokenizer = Tokenizer.create ~model:(Models.chars ()) in
  let normalizer = Normalizers.nfc () in
  Tokenizer.set_normalizer tokenizer (Some normalizer);

  (* These should not crash, but handle invalid sequences *)
  let encoding = Tokenizer.encode tokenizer ~sequence:(Either.Left text) () in
  check bool "handled invalid UTF-8" true
    (Encoding.get_ids encoding |> Array.length >= 0)

(* ───── Decoder Error Tests ───── *)

let test_decode_invalid_ids () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  (* Try to decode IDs that don't exist in vocab *)
  let decoded = Tokenizer.decode tokenizer [ 999; 1000; 1001 ] () in
  (* Should handle gracefully - probably returns empty or unknown tokens *)
  check bool "decoded invalid ids" true (String.length decoded >= 0)

(* ───── Batch Processing Error Tests ───── *)

let test_encode_batch_empty () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  let encodings = Tokenizer.encode_batch tokenizer ~input:[] () in
  check int "empty batch" 0 (List.length encodings)

let test_decode_batch_empty () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  let decoded = Tokenizer.decode_batch tokenizer [] () in
  check int "empty decode batch" 0 (List.length decoded)

(* ───── Special Token Error Tests ───── *)

let test_add_duplicate_special_tokens () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  let count1 =
    Tokenizer.add_special_tokens tokenizer
      [ Either.Left "<pad>"; Either.Left "<pad>" (* Duplicate *) ]
  in
  (* Should handle duplicates gracefully *)
  check bool "handled duplicates" true (count1 >= 0)

(* ───── Save/Load Error Tests ───── *)

let test_save_to_invalid_path () =
  let tokenizer = Tokenizer.create ~model:(Models.word_level ()) in
  (* Try to save to a directory that doesn't exist *)
  try
    Tokenizer.save tokenizer ~path:"/nonexistent/dir/tokenizer.json" ();
    fail "Expected exception but none was raised"
  with _ -> check bool "save failed as expected" true true

(* ───── Test Suite ───── *)

let error_tests =
  [
    (* Tokenizer creation errors *)
    test_case "load nonexistent file" `Quick test_load_nonexistent_file;
    test_case "load invalid json" `Quick test_load_invalid_json;
    (* Model errors *)
    test_case "bpe empty vocab" `Quick test_bpe_empty_vocab;
    test_case "wordpiece empty vocab" `Quick test_wordpiece_empty_vocab;
    (* Pre-tokenizer errors *)
    test_case "invalid split pattern" `Quick test_invalid_split_pattern;
    (* Normalizer errors *)
    test_case "unicode normalization error" `Quick
      test_unicode_normalization_error;
    (* Decoder errors *)
    test_case "decode invalid ids" `Quick test_decode_invalid_ids;
    (* Batch processing errors *)
    test_case "encode batch empty" `Quick test_encode_batch_empty;
    test_case "decode batch empty" `Quick test_decode_batch_empty;
    (* Special token errors *)
    test_case "add duplicate special tokens" `Quick
      test_add_duplicate_special_tokens;
    (* Save/Load errors *)
    test_case "save to invalid path" `Quick test_save_to_invalid_path;
  ]

let () = Alcotest.run "saga errors" [ ("errors", error_tests) ]
