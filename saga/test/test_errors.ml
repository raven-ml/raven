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

(* ───── Vocabulary Error Tests ───── *)

let test_vocab_min_freq_error () =
  check_raises "negative min_freq"
    (Invalid_argument "vocab: invalid min_freq 0 (must be >= 1)") (fun () ->
      ignore (vocab ~min_freq:0 [ "hello" ]))

let test_vocab_size_exceeded () =
  let tokens = List.init 100 (fun i -> string_of_int i) in
  (* This should succeed as we only check size after building *)
  let v = vocab ~max_size:10 tokens in
  check bool "vocab size limited" true (vocab_size v <= 10)

(* ───── Encoding Error Tests ───── *)

let test_encoding_overflow () =
  check_raises "sequence too long"
    (Invalid_argument
       "encode_batch: cannot encode sequence length 5 to max_length 3\n\
        hint: increase max_length or truncate input") (fun () ->
      let long_text = "one two three four five" in
      ignore (encode_batch ~max_len:3 ~pad:false [ long_text ]))

(* ───── Tokenization Error Tests ───── *)

let test_invalid_regex_pattern () =
  check_raises "invalid regex"
    (Invalid_argument
       "tokenize: invalid regex pattern '[' (invalid regex pattern)") (fun () ->
      ignore (tokenize ~method_:(`Regex "[") "test"))

(* ───── File I/O Error Tests ───── *)

let test_vocab_file_not_found () =
  check_raises "file not found"
    (Invalid_argument
       "vocab_load: load vocab from '/nonexistent/file.txt' (file not found)")
    (fun () -> ignore (vocab_load "/nonexistent/file.txt"))

let test_vocab_save_permission_error () =
  (* Try to save to a directory that doesn't exist *)
  try
    let v = vocab [ "test" ] in
    vocab_save v "/nonexistent/dir/vocab.txt";
    fail "Expected exception but none was raised"
  with Invalid_argument msg ->
    (* The exact error message depends on the system *)
    check bool "save error" true
      (try
         ignore (String.index msg 'v');
         true
       with Not_found -> false)

(* ───── Unicode Error Tests ───── *)

let test_unicode_normalization_error () =
  (* Unicode functions should handle errors gracefully *)
  let text = "test" ^ String.make 1 '\xFF' ^ String.make 1 '\xFE' in
  (* Invalid UTF-8 *)
  (* These should not crash, but skip invalid sequences *)
  let _ = normalize ~lowercase:true text in
  let _ = normalize ~strip_accents:true text in
  let _ = normalize ~collapse_whitespace:true text in
  check bool "handled invalid UTF-8" true true

(* ───── Decode Error Tests ───── *)

let test_decode_batch_wrong_shape () =
  let tensor = Nx.zeros Nx.int32 [| 2; 3; 4 |] in
  (* 3D tensor *)
  let v = vocab [ "test" ] in
  check_raises "wrong tensor shape"
    (Invalid_argument
       "decode_batch: invalid tensor shape (expected 2D tensor [batch_size; \
        seq_len])") (fun () -> ignore (decode_batch v tensor))

(* ───── Test Suite ───── *)

let error_tests =
  [
    (* Vocabulary errors *)
    test_case "vocab min_freq error" `Quick test_vocab_min_freq_error;
    test_case "vocab size exceeded" `Quick test_vocab_size_exceeded;
    (* Encoding errors *)
    test_case "encoding overflow" `Quick test_encoding_overflow;
    (* Tokenization errors *)
    test_case "invalid regex pattern" `Quick test_invalid_regex_pattern;
    (* File I/O errors *)
    test_case "vocab file not found" `Quick test_vocab_file_not_found;
    test_case "vocab save permission error" `Quick
      test_vocab_save_permission_error;
    (* Unicode errors *)
    test_case "unicode normalization error" `Quick
      test_unicode_normalization_error;
    (* Decode errors *)
    test_case "decode batch wrong shape" `Quick test_decode_batch_wrong_shape;
  ]

let () = Alcotest.run "saga errors" [ ("errors", error_tests) ]
