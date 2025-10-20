open Saga_tokenizers

let test_wordpiece_basic () =
  (* Create a simple vocabulary *)
  let vocab =
    [
      ("[UNK]", 0);
      ("hello", 1);
      ("world", 2);
      ("##llo", 3);
      ("##rld", 4);
      ("he", 5);
      ("wo", 6);
    ]
  in

  let tokenizer =
    Tokenizer.wordpiece ~vocab ~unk_token:"[UNK]"
      ~continuing_subword_prefix:"##" ()
  in

  (* Test tokenizing a known word *)
  let encoding = Tokenizer.encode tokenizer "hello" in
  let tokens = Encoding.get_tokens encoding in
  Alcotest.(check int) "single token for 'hello'" 1 (Array.length tokens);
  Alcotest.(check string) "token value" "hello" tokens.(0);

  Printf.printf "Tokenized 'hello': ";
  Array.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  Alcotest.(check int) "vocabulary size" 7 (Tokenizer.vocab_size tokenizer)

let test_wordpiece_subwords () =
  (* Create vocabulary with subword pieces *)
  let vocab =
    [
      ("[UNK]", 0);
      ("un", 1);
      ("##able", 2);
      ("##happy", 3);
      ("play", 4);
      ("##ing", 5);
      ("##ed", 6);
    ]
  in

  let tokenizer = Tokenizer.wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Test word that can be split into subwords *)
  let encoding = Tokenizer.encode tokenizer "playing" in
  let tokens = Encoding.get_tokens encoding in
  Printf.printf "Tokenized 'playing': ";
  Array.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  Alcotest.(check int) "should split into subwords" 2 (Array.length tokens);
  Alcotest.(check string) "first token" "play" tokens.(0);
  Alcotest.(check string) "second token" "##ing" tokens.(1)

let test_wordpiece_unknown () =
  (* Create minimal vocabulary *)
  let vocab = [ ("[UNK]", 0); ("hello", 1) ] in

  let tokenizer = Tokenizer.wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Test unknown word *)
  let encoding = Tokenizer.encode tokenizer "goodbye" in
  let tokens = Encoding.get_tokens encoding in
  Alcotest.(check int)
    "unknown word becomes single token" 1 (Array.length tokens);
  Alcotest.(check string) "unknown token" "[UNK]" tokens.(0)

let test_wordpiece_max_chars () =
  (* Create vocabulary *)
  let vocab = [ ("[UNK]", 0); ("test", 1) ] in

  let tokenizer =
    Tokenizer.wordpiece ~vocab ~unk_token:"[UNK]" ~max_input_chars_per_word:5 ()
  in

  (* Test word exceeding max chars *)
  let long_word = String.make 10 'a' in
  let encoding = Tokenizer.encode tokenizer long_word in
  let tokens = Encoding.get_tokens encoding in
  Alcotest.(check int) "long word becomes unknown" 1 (Array.length tokens);
  Alcotest.(check string) "unknown token" "[UNK]" tokens.(0)

let test_wordpiece_save_load () =
  (* Create vocabulary *)
  let vocab =
    [
      ("[PAD]", 0);
      ("[UNK]", 1);
      ("[CLS]", 2);
      ("[SEP]", 3);
      ("hello", 4);
      ("world", 5);
    ]
  in

  let tokenizer = Tokenizer.wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Save the model *)
  let temp_dir = Filename.temp_dir "wordpiece_test" "" in
  let files = Tokenizer.save_model_files tokenizer ~folder:temp_dir () in

  (* Load the model *)
  let vocab_file = List.find (fun f -> Filename.check_suffix f ".txt") files in
  let loaded_tokenizer = Tokenizer.from_model_file ~vocab:vocab_file () in

  (* Test that loaded tokenizer works the same *)
  let original_tokens =
    Tokenizer.encode tokenizer "hello" |> Encoding.get_tokens
  in
  let loaded_tokens =
    Tokenizer.encode loaded_tokenizer "hello" |> Encoding.get_tokens
  in

  Alcotest.(check int)
    "same number of tokens"
    (Array.length original_tokens)
    (Array.length loaded_tokens);

  (* Clean up *)
  List.iter Sys.remove files;
  Unix.rmdir temp_dir

let test_tokenizer_integration () =
  (* Create a WordPiece tokenizer using the high-level API *)
  let vocab =
    [
      ("[PAD]", 0);
      ("[UNK]", 1);
      ("[CLS]", 2);
      ("[SEP]", 3);
      ("hello", 4);
      ("world", 5);
      ("##ing", 6);
    ]
  in
  let tokenizer = Tokenizer.wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Test encoding *)
  let tokens =
    Tokenizer.encode tokenizer "hello" |> Encoding.get_tokens |> Array.to_list
  in

  Printf.printf "Tokenizer.wordpiece result: ";
  List.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  Alcotest.(check bool) "tokenizer produces output" true (List.length tokens > 0)

let test_wordpiece_greedy_matching () =
  (* Test the greedy longest-match-first algorithm *)
  let vocab =
    [
      ("[UNK]", 0);
      ("un", 1);
      ("able", 2);
      ("unable", 3);
      (* Longer match should be preferred *)
      ("##able", 4);
    ]
  in

  let tokenizer = Tokenizer.wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Should match "unable" as a single token, not "un" + "##able" *)
  let encoding = Tokenizer.encode tokenizer "unable" in
  let tokens = Encoding.get_tokens encoding in
  Alcotest.(check int)
    "greedy match finds longest token" 1 (Array.length tokens);
  Alcotest.(check string) "matched full word" "unable" tokens.(0)

let () =
  let open Alcotest in
  run "WordPiece tests"
    [
      ( "basic",
        [
          test_case "basic tokenization" `Quick test_wordpiece_basic;
          test_case "subword tokenization" `Quick test_wordpiece_subwords;
          test_case "unknown tokens" `Quick test_wordpiece_unknown;
          test_case "max input chars" `Quick test_wordpiece_max_chars;
          test_case "save and load" `Quick test_wordpiece_save_load;
          test_case "tokenizer integration" `Quick test_tokenizer_integration;
          test_case "greedy matching" `Quick test_wordpiece_greedy_matching;
        ] );
    ]
