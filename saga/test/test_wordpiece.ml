open Saga_tokenizers

let test_wordpiece_basic () =
  (* Create a simple vocabulary *)
  let vocab = Hashtbl.create 10 in
  Hashtbl.add vocab "[UNK]" 0;
  Hashtbl.add vocab "hello" 1;
  Hashtbl.add vocab "world" 2;
  Hashtbl.add vocab "##llo" 3;
  Hashtbl.add vocab "##rld" 4;
  Hashtbl.add vocab "he" 5;
  Hashtbl.add vocab "wo" 6;

  let config : Wordpiece.config =
    {
      vocab;
      unk_token = "[UNK]";
      continuing_subword_prefix = "##";
      max_input_chars_per_word = 100;
    }
  in

  let model = Wordpiece.create config in

  (* Test tokenizing a known word *)
  let tokens = Wordpiece.tokenize model "hello" in
  Alcotest.(check int) "single token for 'hello'" 1 (List.length tokens);
  Alcotest.(check string) "token value" "hello" (List.hd tokens).Wordpiece.value;

  (* Test tokenizing with subwords *)
  let tokens = Wordpiece.tokenize model "hello" in
  Printf.printf "Tokenized 'hello': ";
  List.iter
    (fun t -> Printf.printf "%s (id=%d) " t.Wordpiece.value t.Wordpiece.id)
    tokens;
  Printf.printf "\n";

  Alcotest.(check int) "vocabulary size" 7 (Wordpiece.get_vocab_size model)

let test_wordpiece_subwords () =
  (* Create vocabulary with subword pieces *)
  let vocab = Hashtbl.create 20 in
  Hashtbl.add vocab "[UNK]" 0;
  Hashtbl.add vocab "un" 1;
  Hashtbl.add vocab "##able" 2;
  Hashtbl.add vocab "##happy" 3;
  Hashtbl.add vocab "play" 4;
  Hashtbl.add vocab "##ing" 5;
  Hashtbl.add vocab "##ed" 6;

  let builder = Wordpiece.Builder.create () in
  let builder = Wordpiece.Builder.vocab builder vocab in
  let builder = Wordpiece.Builder.unk_token builder "[UNK]" in
  let model = Wordpiece.Builder.build builder in

  (* Test word that can be split into subwords *)
  let tokens = Wordpiece.tokenize model "playing" in
  Printf.printf "Tokenized 'playing': ";
  List.iter (fun t -> Printf.printf "%s " t.Wordpiece.value) tokens;
  Printf.printf "\n";

  Alcotest.(check int) "should split into subwords" 2 (List.length tokens);
  Alcotest.(check string)
    "first token" "play" (List.nth tokens 0).Wordpiece.value;
  Alcotest.(check string)
    "second token" "##ing" (List.nth tokens 1).Wordpiece.value

let test_wordpiece_unknown () =
  (* Create minimal vocabulary *)
  let vocab = Hashtbl.create 5 in
  Hashtbl.add vocab "[UNK]" 0;
  Hashtbl.add vocab "hello" 1;

  let builder = Wordpiece.Builder.create () in
  let builder = Wordpiece.Builder.vocab builder vocab in
  let builder = Wordpiece.Builder.unk_token builder "[UNK]" in
  let model = Wordpiece.Builder.build builder in

  (* Test unknown word *)
  let tokens = Wordpiece.tokenize model "goodbye" in
  Alcotest.(check int)
    "unknown word becomes single token" 1 (List.length tokens);
  Alcotest.(check string)
    "unknown token" "[UNK]" (List.hd tokens).Wordpiece.value

let test_wordpiece_max_chars () =
  (* Create vocabulary *)
  let vocab = Hashtbl.create 5 in
  Hashtbl.add vocab "[UNK]" 0;
  Hashtbl.add vocab "test" 1;

  let builder = Wordpiece.Builder.create () in
  let builder = Wordpiece.Builder.vocab builder vocab in
  let builder = Wordpiece.Builder.unk_token builder "[UNK]" in
  let builder = Wordpiece.Builder.max_input_chars_per_word builder 5 in
  let model = Wordpiece.Builder.build builder in

  (* Test word exceeding max chars *)
  let long_word = String.make 10 'a' in
  let tokens = Wordpiece.tokenize model long_word in
  Alcotest.(check int) "long word becomes unknown" 1 (List.length tokens);
  Alcotest.(check string)
    "unknown token" "[UNK]" (List.hd tokens).Wordpiece.value

let test_wordpiece_save_load () =
  (* Create vocabulary *)
  let vocab = Hashtbl.create 10 in
  Hashtbl.add vocab "[PAD]" 0;
  Hashtbl.add vocab "[UNK]" 1;
  Hashtbl.add vocab "[CLS]" 2;
  Hashtbl.add vocab "[SEP]" 3;
  Hashtbl.add vocab "hello" 4;
  Hashtbl.add vocab "world" 5;

  let builder = Wordpiece.Builder.create () in
  let builder = Wordpiece.Builder.vocab builder vocab in
  let builder = Wordpiece.Builder.unk_token builder "[UNK]" in
  let model = Wordpiece.Builder.build builder in

  (* Save the model *)
  let temp_dir = Filename.temp_dir "wordpiece_test" "" in
  let _fp = Wordpiece.save model ~path:temp_dir () in

  (* Load the model *)
  let vocab_file = Filename.concat temp_dir "vocab.txt" in
  let loaded_model = Wordpiece.from_file ~vocab_file in

  (* Test that loaded model works the same *)
  let original_tokens = Wordpiece.tokenize model "hello" in
  let loaded_tokens = Wordpiece.tokenize loaded_model "hello" in

  Alcotest.(check int)
    "same number of tokens"
    (List.length original_tokens)
    (List.length loaded_tokens);

  (* Clean up *)
  Sys.remove vocab_file;
  Unix.rmdir temp_dir

let test_tokenizer_integration () =
  (* Create temporary vocab file *)
  let temp_dir = Filename.temp_dir "wordpiece_test" "" in
  let vocab_file = Filename.concat temp_dir "vocab.txt" in

  (* Write a simple vocab file *)
  let oc = open_out vocab_file in
  output_string oc "[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld\n##ing\n";
  close_out oc;

  (* Test the Tokenizer.wordpiece function *)
  let tokenizer = Tokenizer.wordpiece ~vocab:vocab_file ~unk_token:"[UNK]" in
  let tokens = Tokenizer.run tokenizer "hello" in

  Printf.printf "Tokenizer.wordpiece result: ";
  List.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  Alcotest.(check bool) "tokenizer produces output" true (List.length tokens > 0);

  (* Clean up *)
  Sys.remove vocab_file;
  Unix.rmdir temp_dir

let test_wordpiece_greedy_matching () =
  (* Test the greedy longest-match-first algorithm *)
  let vocab = Hashtbl.create 20 in
  Hashtbl.add vocab "[UNK]" 0;
  Hashtbl.add vocab "un" 1;
  Hashtbl.add vocab "able" 2;
  Hashtbl.add vocab "unable" 3;
  (* Longer match should be preferred *)
  Hashtbl.add vocab "##able" 4;

  let builder = Wordpiece.Builder.create () in
  let builder = Wordpiece.Builder.vocab builder vocab in
  let builder = Wordpiece.Builder.unk_token builder "[UNK]" in
  let model = Wordpiece.Builder.build builder in

  (* Should match "unable" as a single token, not "un" + "##able" *)
  let tokens = Wordpiece.tokenize model "unable" in
  Alcotest.(check int) "greedy match finds longest token" 1 (List.length tokens);
  Alcotest.(check string)
    "matched full word" "unable" (List.hd tokens).Wordpiece.value

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
