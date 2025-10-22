open Saga_tokenizers

let test_load_bert_tokenizer () =
  let tokenizer_path = "/tmp/bert-tokenizer/tokenizer.json" in
  if Sys.file_exists tokenizer_path then
    match Tokenizer.from_file tokenizer_path with
    | Ok tok ->
        (* Test basic encoding *)
        let enc = Tokenizer.encode tok "Hello world!" in
        let ids = Encoding.get_ids enc in
        Alcotest.(check bool) "Has tokens" true (Array.length ids > 0);
        (* Test vocab size *)
        let vocab = Tokenizer.vocab tok in
        Alcotest.(check bool) "Has vocabulary" true (List.length vocab > 0);
        (* Test that special tokens are present *)
        let vocab_tokens = List.map fst vocab in
        Alcotest.(check bool)
          "Has [PAD] token" true
          (List.mem "[PAD]" vocab_tokens);
        Alcotest.(check bool)
          "Has [CLS] token" true
          (List.mem "[CLS]" vocab_tokens);
        Alcotest.(check bool)
          "Has [SEP] token" true
          (List.mem "[SEP]" vocab_tokens)
    | Error e ->
        Alcotest.failf "Failed to load tokenizer: %s" (Printexc.to_string e)
  else Alcotest.skip ()

let test_load_gpt2_tokenizer () =
  let tokenizer_path = "/tmp/gpt2-tokenizer/tokenizer.json" in
  if Sys.file_exists tokenizer_path then
    match Tokenizer.from_file tokenizer_path with
    | Ok tok ->
        (* Test basic encoding *)
        let enc = Tokenizer.encode tok "Hello world!" in
        let ids = Encoding.get_ids enc in
        Alcotest.(check bool) "Has tokens" true (Array.length ids > 0);
        (* Test vocab size *)
        let vocab = Tokenizer.vocab tok in
        Alcotest.(check bool) "Has vocabulary" true (List.length vocab > 0)
    | Error e ->
        Alcotest.failf "Failed to load tokenizer: %s" (Printexc.to_string e)
  else Alcotest.skip ()

let test_roundtrip_wordpiece () =
  (* Create a simple tokenizer *)
  let tok =
    Tokenizer.wordpiece
      ~vocab:
        [
          ("[PAD]", 0);
          ("[UNK]", 1);
          ("[CLS]", 2);
          ("[SEP]", 3);
          ("hello", 4);
          ("world", 5);
          ("##ing", 6);
        ]
      ~unk_token:"[UNK]" ~continuing_subword_prefix:"##" ()
  in
  (* Save it *)
  let temp_dir = Filename.get_temp_dir_name () in
  let save_path = Filename.concat temp_dir "test_tokenizer" in
  Tokenizer.save_pretrained tok ~path:save_path;
  (* Load it back *)
  let tokenizer_file = Filename.concat save_path "tokenizer.json" in
  match Tokenizer.from_file tokenizer_file with
  | Ok loaded_tok ->
      (* Test that encoding works the same *)
      let enc1 = Tokenizer.encode tok "hello world" in
      let enc2 = Tokenizer.encode loaded_tok "hello world" in
      let ids1 = Encoding.get_ids enc1 in
      let ids2 = Encoding.get_ids enc2 in
      Alcotest.(check (array int)) "Same encoding" ids1 ids2;
      (* Clean up *)
      Sys.remove tokenizer_file;
      Unix.rmdir save_path
  | Error e ->
      Alcotest.failf "Failed to load saved tokenizer: %s" (Printexc.to_string e)

let test_roundtrip_bpe () =
  (* Create a simple BPE tokenizer *)
  let tok =
    Tokenizer.bpe
      ~vocab:[ ("h", 0); ("e", 1); ("l", 2); ("o", 3); ("he", 4) ]
      ~merges:[ ("h", "e") ]
      ()
  in
  (* Save it *)
  let temp_dir = Filename.get_temp_dir_name () in
  let save_path = Filename.concat temp_dir "test_bpe_tokenizer" in
  Tokenizer.save_pretrained tok ~path:save_path;
  (* Load it back *)
  let tokenizer_file = Filename.concat save_path "tokenizer.json" in
  match Tokenizer.from_file tokenizer_file with
  | Ok loaded_tok ->
      (* Test that vocab is preserved *)
      let vocab1 = Tokenizer.vocab tok |> List.sort compare in
      let vocab2 = Tokenizer.vocab loaded_tok |> List.sort compare in
      Alcotest.(check (list (pair string int))) "Same vocab" vocab1 vocab2;
      (* Clean up *)
      Sys.remove tokenizer_file;
      Unix.rmdir save_path
  | Error e ->
      Alcotest.failf "Failed to load saved tokenizer: %s" (Printexc.to_string e)

let test_roundtrip_wordlevel () =
  (* Create a word-level tokenizer *)
  let tok =
    Tokenizer.word_level
      ~vocab:[ ("hello", 0); ("world", 1); ("[UNK]", 2) ]
      ~unk_token:"[UNK]" ()
  in
  (* Save it *)
  let temp_dir = Filename.get_temp_dir_name () in
  let save_path = Filename.concat temp_dir "test_wordlevel_tokenizer" in
  Tokenizer.save_pretrained tok ~path:save_path;
  (* Load it back *)
  let tokenizer_file = Filename.concat save_path "tokenizer.json" in
  match Tokenizer.from_file tokenizer_file with
  | Ok loaded_tok ->
      (* Test that vocab is preserved *)
      let vocab1 = Tokenizer.vocab tok |> List.sort compare in
      let vocab2 = Tokenizer.vocab loaded_tok |> List.sort compare in
      Alcotest.(check (list (pair string int))) "Same vocab" vocab1 vocab2;
      (* Clean up *)
      Sys.remove tokenizer_file;
      Unix.rmdir save_path
  | Error e ->
      Alcotest.failf "Failed to load saved tokenizer: %s" (Printexc.to_string e)

let () =
  let open Alcotest in
  run "Tokenizer I/O"
    [
      ( "load_hf",
        [
          test_case "load bert-base-uncased" `Quick test_load_bert_tokenizer;
          test_case "load gpt2" `Quick test_load_gpt2_tokenizer;
        ] );
      ( "roundtrip",
        [
          test_case "wordpiece" `Quick test_roundtrip_wordpiece;
          test_case "bpe" `Quick test_roundtrip_bpe;
          test_case "wordlevel" `Quick test_roundtrip_wordlevel;
        ] );
    ]
