open Alcotest
open Saga_tokenizers

let make_word_tokenizer ?(specials = []) () =
  Tokenizer.word_level ~pre:(Pre_tokenizers.whitespace ()) ~specials ()

let test_encode_simple () =
  let tokenizer =
    Tokenizer.add_tokens (make_word_tokenizer ()) [ "hello"; "world" ]
  in
  let ids =
    Tokenizer.encode tokenizer "hello world hello" |> Encoding.get_ids
  in
  check int "encoded length" 3 (Array.length ids);
  check bool "repeated token same id" true (ids.(0) = ids.(2))

let test_encode_with_vocab () =
  let tokenizer =
    Tokenizer.add_tokens (make_word_tokenizer ()) [ "hello"; "world" ]
  in
  let ids =
    Tokenizer.encode tokenizer "hello world"
    |> Encoding.get_ids |> Array.to_list
  in
  check (list int) "encoded with vocab" [ 0; 1 ] ids

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
  check bool "encoded something" true (List.length ids > 0)

let test_encode_empty () =
  let tokenizer = make_word_tokenizer () in
  let ids =
    Tokenizer.encode tokenizer "" |> Encoding.get_ids |> Array.to_list
  in
  check (list int) "encode empty" [] ids

let test_encode_batch_simple () =
  let tokenizer =
    Tokenizer.add_tokens (make_word_tokenizer ())
      [ "hello"; "world"; "hi"; "there" ]
  in
  let encodings =
    Tokenizer.encode_batch tokenizer [ "hello world"; "hi there" ]
  in
  check int "batch size" 2 (List.length encodings);
  let first = List.hd encodings in
  check bool "first encoding has ids" true
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
  check int "first padded length" 5 (Array.length first);
  check int "second padded length" 5 (Array.length second)

let test_encode_batch_empty () =
  let tokenizer = make_word_tokenizer () in
  let encodings = Tokenizer.encode_batch tokenizer [] in
  check int "empty batch" 0 (List.length encodings)

let test_decode_simple () =
  let tokenizer =
    Tokenizer.add_tokens (make_word_tokenizer ()) [ "hello"; "world" ]
  in
  let decoded = Tokenizer.decode tokenizer [| 0; 1 |] in
  check string "decoded text" "hello world" decoded

let test_decode_with_special () =
  let tokenizer =
    Tokenizer.add_tokens
      (make_word_tokenizer
         ~specials:[ Special.bos "<bos>"; Special.eos "<eos>" ]
         ())
      [ "hello" ]
  in
  let decoded = Tokenizer.decode tokenizer [| 1; 2; 0 |] in
  check string "decoded with special" "<bos> hello <eos>" decoded

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
  check string "decoded without special" "hello" decoded

let test_decode_batch () =
  let tokenizer =
    Tokenizer.add_tokens (make_word_tokenizer ())
      [ "hello"; "world"; "hi"; "there" ]
  in
  let decoded = Tokenizer.decode_batch tokenizer [ [| 0; 1 |]; [| 2; 3 |] ] in
  check int "decoded count" 2 (List.length decoded);
  check string "first decoded" "hello world" (List.nth decoded 0);
  check string "second decoded" "hi there" (List.nth decoded 1)

let test_chars_model () =
  let tokenizer = Tokenizer.chars () in
  let ids =
    Tokenizer.encode tokenizer "abc" |> Encoding.get_ids |> Array.to_list
  in
  check (list int) "char ids" [ 97; 98; 99 ] ids

let suite =
  [
    test_case "encode simple" `Quick test_encode_simple;
    test_case "encode with vocab" `Quick test_encode_with_vocab;
    test_case "encode unknown tokens" `Quick test_encode_unknown_tokens;
    test_case "encode empty" `Quick test_encode_empty;
    test_case "batch simple" `Quick test_encode_batch_simple;
    test_case "batch with padding" `Quick test_encode_batch_with_padding;
    test_case "batch empty request" `Quick test_encode_batch_empty;
    test_case "decode simple" `Quick test_decode_simple;
    test_case "decode with special" `Quick test_decode_with_special;
    test_case "decode skip special" `Quick test_decode_skip_special;
    test_case "decode batch" `Quick test_decode_batch;
    test_case "chars model" `Quick test_chars_model;
  ]

let () = run "Encoding tests" [ ("encoding", suite) ]
