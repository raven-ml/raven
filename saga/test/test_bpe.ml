open Saga_tokenizers

let test_bpe_basic () =
  (* Create a simple vocabulary and merges *)
  let vocab =
    [
      ("h", 0);
      ("e", 1);
      ("l", 2);
      ("o", 3);
      ("ll", 4);
      ("he", 5);
      ("llo", 6);
      ("hello", 7);
    ]
  in

  let merges =
    [
      ("l", "l");
      (* rank 0: Merge 'l' + 'l' -> 'll' *)
      ("ll", "o");
      (* rank 1: Merge 'll' + 'o' -> 'llo' *)
      ("he", "llo");
      (* rank 2: Merge 'he' + 'llo' -> 'hello' *)
    ]
  in

  let tokenizer = Tokenizer.bpe ~vocab ~merges ~unk_token:"<unk>" () in

  let encoding = Tokenizer.encode tokenizer "hello" in
  let tokens = Encoding.get_tokens encoding |> Array.to_list in

  Printf.printf "Tokenized 'hello': ";
  List.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  Alcotest.(check int) "vocabulary size" 8 (Tokenizer.vocab_size tokenizer)

let test_bpe_builder () =
  let vocab = [ ("a", 0); ("b", 1); ("ab", 2) ] in
  let merges = [ ("a", "b") ] in

  let tokenizer = Tokenizer.bpe ~vocab ~merges ~cache_capacity:50 () in

  let encoding = Tokenizer.encode tokenizer "ab" in
  let tokens = Encoding.get_tokens encoding in
  Alcotest.(check int) "single token for 'ab'" 1 (Array.length tokens)

let test_bpe_save_load () =
  let vocab = [ ("t", 0); ("e", 1); ("s", 2); ("test", 3) ] in
  let merges = [] in
  (* No merges for simplicity *)

  let tokenizer = Tokenizer.bpe ~vocab ~merges () in

  (* Save the model *)
  let temp_dir = Filename.temp_dir "bpe_test" "" in
  let files = Tokenizer.save_model_files tokenizer ~folder:temp_dir () in

  (* Load the model *)
  let vocab_file = List.find (fun f -> Filename.check_suffix f ".json") files in
  let merges_file = List.find (fun f -> Filename.check_suffix f ".txt") files in
  let loaded_tokenizer =
    Tokenizer.from_model_file ~vocab:vocab_file ~merges:merges_file ()
  in

  (* Test that loaded tokenizer works the same *)
  let original_tokens =
    Tokenizer.encode tokenizer "test" |> Encoding.get_tokens
  in
  let loaded_tokens =
    Tokenizer.encode loaded_tokenizer "test" |> Encoding.get_tokens
  in

  Alcotest.(check int)
    "same number of tokens"
    (Array.length original_tokens)
    (Array.length loaded_tokens);

  (* Clean up *)
  List.iter Sys.remove files;
  Unix.rmdir temp_dir

let test_tokenizer_integration () =
  (* Create a BPE tokenizer using the high-level API *)
  let vocab =
    [
      ("h", 0); ("e", 1); ("l", 2); ("o", 3); ("he", 4); ("llo", 5); ("hello", 6);
    ]
  in
  let merges = [ ("h", "e"); ("he", "llo") ] in
  let tokenizer = Tokenizer.bpe ~vocab ~merges () in

  (* Test encoding *)
  let tokens =
    Tokenizer.encode tokenizer "hello" |> Encoding.get_tokens |> Array.to_list
  in

  Printf.printf "Tokenizer.bpe result: ";
  List.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  Alcotest.(check bool) "tokenizer produces output" true (List.length tokens > 0)

let () =
  let open Alcotest in
  run "BPE tests"
    [
      ( "basic",
        [
          test_case "basic tokenization" `Quick test_bpe_basic;
          test_case "builder pattern" `Quick test_bpe_builder;
          test_case "save and load" `Quick test_bpe_save_load;
          test_case "tokenizer integration" `Quick test_tokenizer_integration;
        ] );
    ]
