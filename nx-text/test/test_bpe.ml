open Nx_text

let test_bpe_basic () =
  (* Create a simple vocabulary and merges *)
  let vocab = Hashtbl.create 10 in
  Hashtbl.add vocab "h" 0;
  Hashtbl.add vocab "e" 1;
  Hashtbl.add vocab "l" 2;
  Hashtbl.add vocab "o" 3;
  Hashtbl.add vocab "ll" 4;
  (* Result of merging l + l *)
  Hashtbl.add vocab "he" 5;
  Hashtbl.add vocab "llo" 6;
  (* Result of merging ll + o *)
  Hashtbl.add vocab "hello" 7;

  (* Result of merging he + llo *)
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

  let config : Bpe.config =
    {
      vocab;
      merges;
      cache_capacity = 100;
      dropout = None;
      unk_token = Some "<unk>";
      continuing_subword_prefix = None;
      end_of_word_suffix = None;
      fuse_unk = false;
      byte_fallback = false;
      ignore_merges = false;
    }
  in

  let model = Bpe.create config in
  let tokens = Bpe.tokenize model "hello" in

  Printf.printf "Tokenized 'hello': ";
  List.iter (fun t -> Printf.printf "%s (id=%d) " t.Bpe.value t.Bpe.id) tokens;
  Printf.printf "\n";

  Alcotest.(check int) "vocabulary size" 8 (Bpe.get_vocab_size model)

let test_bpe_builder () =
  let vocab = Hashtbl.create 10 in
  Hashtbl.add vocab "a" 0;
  Hashtbl.add vocab "b" 1;
  Hashtbl.add vocab "ab" 2;

  let merges = [ ("a", "b") ] in

  let builder = Bpe.Builder.create () in
  let builder = Bpe.Builder.vocab_and_merges builder vocab merges in
  let builder = Bpe.Builder.cache_capacity builder 50 in
  let model = Bpe.Builder.build builder in

  let tokens = Bpe.tokenize model "ab" in
  Alcotest.(check int) "single token for 'ab'" 1 (List.length tokens)

let test_bpe_save_load () =
  let vocab = Hashtbl.create 10 in
  Hashtbl.add vocab "t" 0;
  Hashtbl.add vocab "e" 1;
  Hashtbl.add vocab "s" 2;
  Hashtbl.add vocab "test" 3;

  let merges = [] in
  (* No merges for simplicity *)

  let builder = Bpe.Builder.create () in
  let builder = Bpe.Builder.vocab_and_merges builder vocab merges in
  let model = Bpe.Builder.build builder in

  (* Save the model *)
  let temp_dir = Filename.temp_dir "bpe_test" "" in
  Bpe.save model ~path:temp_dir ();

  (* Load the model *)
  let vocab_file = Filename.concat temp_dir "vocab.json" in
  let merges_file = Filename.concat temp_dir "merges.txt" in
  let loaded_model = Bpe.from_files ~vocab_file ~merges_file in

  (* Test that loaded model works the same *)
  let original_tokens = Bpe.tokenize model "test" in
  let loaded_tokens = Bpe.tokenize loaded_model "test" in

  Alcotest.(check int)
    "same number of tokens"
    (List.length original_tokens)
    (List.length loaded_tokens);

  (* Clean up *)
  Sys.remove vocab_file;
  Sys.remove merges_file;
  Unix.rmdir temp_dir

let test_tokenizer_integration () =
  (* Create temporary vocab and merges files *)
  let temp_dir = Filename.temp_dir "bpe_test" "" in
  let vocab_file = Filename.concat temp_dir "vocab.json" in
  let merges_file = Filename.concat temp_dir "merges.txt" in

  (* Write a simple vocab file *)
  let oc = open_out vocab_file in
  output_string oc
    "{\"h\": 0, \"e\": 1, \"l\": 2, \"o\": 3, \"he\": 4, \"llo\": 5, \
     \"hello\": 6}";
  close_out oc;

  (* Write a simple merges file *)
  let oc = open_out merges_file in
  output_string oc "#version: 0.2\nh e\nhe llo\n";
  close_out oc;

  (* Test the Tokenizer.bpe function *)
  let tokenizer = Tokenizer.bpe ~vocab:vocab_file ~merges:merges_file in
  let tokens = Tokenizer.run tokenizer "hello" in

  Printf.printf "Tokenizer.bpe result: ";
  List.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  Alcotest.(check bool) "tokenizer produces output" true (List.length tokens > 0);

  (* Clean up *)
  Sys.remove vocab_file;
  Sys.remove merges_file;
  Unix.rmdir temp_dir

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
