(* Encoding and batch processing tests for saga *)

open Alcotest
open Saga

(* Helper to check tensor shape *)
let check_shape msg expected tensor =
  check (array int) msg expected (Nx.shape tensor)

(* Helper to check tensor values *)
let _check_tensor_values msg expected tensor =
  let actual = Nx.to_array tensor |> Array.map Int32.to_int in
  check (array int) msg expected actual

(* ───── Simple Encoding Tests ───── *)

let test_encode_simple () =
  let encoded = encode "hello world hello" in
  (* Should build vocab and encode *)
  check int "encoded length" 3 (List.length encoded);
  (* Check that repeated words get same index *)
  check bool "repeated token same index" true
    (List.nth encoded 0 = List.nth encoded 2)

let test_encode_with_vocab () =
  let v = vocab [ "hello"; "world" ] in
  let encoded = encode ~vocab:v "hello world" in
  check (list int) "encoded with vocab" [ 4; 5 ] encoded

let test_encode_unknown_tokens () =
  let v = vocab [ "hello" ] in
  let encoded = encode ~vocab:v "hello unknown world" in
  check (list int) "encoded with unknowns" [ 4; 1; 1 ] encoded (* 1 is <unk> *)

let test_encode_empty () =
  let encoded = encode "" in
  check (list int) "encode empty string" [] encoded

(* ───── Batch Encoding Tests ───── *)

let test_encode_batch_simple () =
  let batch = encode_batch [ "hello world"; "hi there" ] in
  check_shape "batch shape" [| 2; 512 |] batch;

  (* Check that it's padded *)
  let arr = Nx.to_array batch in
  check bool "padded with zeros" true (Int32.to_int arr.((2 * 512) - 1) = 0)

let test_encode_batch_no_padding () =
  let batch = encode_batch ~pad:false [ "hello"; "hi there" ] in
  check_shape "batch shape no padding" [| 2; 2 |] batch (* max length is 2 *)

let test_encode_batch_max_len () =
  let batch = encode_batch ~max_len:3 [ "hello world test"; "hi" ] in
  check_shape "batch shape max_len" [| 2; 3 |] batch

let test_encode_batch_with_vocab () =
  let v = vocab [ "hello"; "world"; "hi"; "there" ] in
  let batch = encode_batch ~vocab:v ~max_len:3 [ "hello world"; "hi there" ] in
  check_shape "batch shape with vocab" [| 2; 3 |] batch;

  (* Check that tokens are encoded (don't assume specific indices) *)
  let arr = Nx.to_array batch |> Array.map Int32.to_int in
  check bool "first token not pad/special" true (arr.(0) >= 4);
  (* not special token *)
  check bool "second token not pad/special" true (arr.(1) >= 4);
  (* not special token *)
  check int "padding" 0 arr.(2)
(* <pad> *)

let test_encode_batch_empty () =
  let batch = encode_batch [] in
  check_shape "empty batch shape" [| 0; 512 |] batch

let test_encode_batch_single () =
  let batch = encode_batch [ "hello world" ] in
  check_shape "single item batch" [| 1; 512 |] batch

(* ───── Decoding Tests ───── *)

let test_decode_simple () =
  let v = vocab [ "hello"; "world" ] in
  let decoded = decode v [ 4; 5 ] in
  check string "decoded text" "hello world" decoded

let test_decode_with_special () =
  let v = vocab [ "hello" ] in
  let decoded = decode v [ 2; 4; 3 ] in
  (* <bos> hello <eos> *)
  check string "decoded with special" "<bos> hello <eos>" decoded

let test_decode_with_unknown () =
  let v = vocab [ "hello" ] in
  let decoded = decode v [ 4; 1; 1 ] in
  (* hello <unk> <unk> *)
  check string "decoded with unknown" "hello <unk> <unk>" decoded

let test_decode_empty () =
  let v = vocab [] in
  let decoded = decode v [] in
  check string "decode empty" "" decoded

(* ───── Batch Decoding Tests ───── *)

let test_decode_batch_simple () =
  let texts = [ "hello world"; "hi there" ] in
  let v = vocab (List.concat_map tokenize texts) in
  let batch = encode_batch ~vocab:v texts in
  let decoded = decode_batch v batch in

  check int "decoded count" 2 (List.length decoded);
  check string "first decoded" "hello world" (List.nth decoded 0);
  check string "second decoded" "hi there" (List.nth decoded 1)

let test_decode_batch_padded () =
  let v = vocab [ "hello"; "world" ] in
  let batch = Nx.zeros Nx.int32 [| 1; 5 |] in
  Nx.set_item [ 0; 0 ] 4l batch;
  (* hello *)
  Nx.set_item [ 0; 1 ] 5l batch;

  (* world *)
  (* Rest are zeros (padding) *)
  let decoded = decode_batch v batch in
  check string "decoded ignoring padding" "hello world" (List.hd decoded)

(* ───── Round-trip Tests ───── *)

let test_round_trip_simple () =
  let text = "hello world test" in
  let v = vocab (tokenize text) in
  let encoded = encode ~vocab:v text in
  let decoded = decode v encoded in
  check string "round trip" text decoded

let test_round_trip_batch () =
  let texts = [ "hello world"; "this is a test"; "one more" ] in
  let v = vocab (List.concat_map tokenize texts) in
  let batch = encode_batch ~vocab:v texts in
  let decoded = decode_batch v batch in

  List.iter2
    (fun orig dec -> check string "batch round trip" orig dec)
    texts decoded

(* ───── Normalization Tests ───── *)

let test_normalize_lowercase () =
  let normalized = normalize ~lowercase:true "Hello WORLD" in
  check string "lowercase" "hello world" normalized

let test_normalize_whitespace () =
  let normalized = normalize ~collapse_whitespace:true "  hello   world  " in
  check string "collapse whitespace" "hello world" normalized

let test_normalize_combined () =
  let normalized =
    normalize ~lowercase:true ~collapse_whitespace:true "  Hello   WORLD!  "
  in
  check string "combined normalization" "hello world!" normalized

(* ───── Advanced Tokenizer Tests ───── *)

let test_advanced_tokenizer_with_normalizer () =
  let tok =
    Tokenizer.words |> Tokenizer.with_normalizer (normalize ~lowercase:true)
  in
  let tokens = Tokenizer.run tok "Hello WORLD" in
  check (list string) "normalized tokens" [ "hello"; "world" ] tokens

let test_advanced_tokenizer_regex () =
  let tok = Tokenizer.regex "\\w+|[^\\w\\s]+" in
  let tokens = Tokenizer.run tok "don't-stop" in
  check (list string) "regex tokens" [ "don"; "'"; "t"; "-"; "stop" ] tokens

(* ───── Edge Cases ───── *)

let test_encode_batch_long_sequences () =
  let long_text = String.concat " " (List.init 99 (fun i -> string_of_int i)) in
  let batch = encode_batch ~max_len:100 [ long_text ] in
  check_shape "fits within max_len" [| 1; 100 |] batch

let test_encode_batch_unicode () =
  let texts = [ "hello 👋"; "世界 🌍" ] in
  let batch = encode_batch texts in
  let v = vocab (List.concat_map tokenize texts) in
  let decoded = decode_batch v batch in
  check int "unicode batch size" 2 (List.length decoded)

(* ───── Test Suite ───── *)

let encoding_tests =
  [
    (* Simple encoding *)
    test_case "encode simple" `Quick test_encode_simple;
    test_case "encode with vocab" `Quick test_encode_with_vocab;
    test_case "encode unknown tokens" `Quick test_encode_unknown_tokens;
    test_case "encode empty" `Quick test_encode_empty;
    (* Batch encoding *)
    test_case "encode batch simple" `Quick test_encode_batch_simple;
    test_case "encode batch no padding" `Quick test_encode_batch_no_padding;
    test_case "encode batch max_len" `Quick test_encode_batch_max_len;
    test_case "encode batch with vocab" `Quick test_encode_batch_with_vocab;
    test_case "encode batch empty" `Quick test_encode_batch_empty;
    test_case "encode batch single" `Quick test_encode_batch_single;
    (* Decoding *)
    test_case "decode simple" `Quick test_decode_simple;
    test_case "decode with special" `Quick test_decode_with_special;
    test_case "decode with unknown" `Quick test_decode_with_unknown;
    test_case "decode empty" `Quick test_decode_empty;
    (* Batch decoding *)
    test_case "decode batch simple" `Quick test_decode_batch_simple;
    test_case "decode batch padded" `Quick test_decode_batch_padded;
    (* Round trips *)
    test_case "round trip simple" `Quick test_round_trip_simple;
    test_case "round trip batch" `Quick test_round_trip_batch;
    (* Normalization *)
    test_case "normalize lowercase" `Quick test_normalize_lowercase;
    test_case "normalize whitespace" `Quick test_normalize_whitespace;
    test_case "normalize combined" `Quick test_normalize_combined;
    (* Advanced tokenizer *)
    test_case "advanced with normalizer" `Quick
      test_advanced_tokenizer_with_normalizer;
    test_case "advanced regex" `Quick test_advanced_tokenizer_regex;
    (* Edge cases *)
    test_case "encode batch long sequences" `Quick
      test_encode_batch_long_sequences;
    test_case "encode batch unicode" `Quick test_encode_batch_unicode;
  ]

let () = Alcotest.run "saga encoding" [ ("encoding", encoding_tests) ]
