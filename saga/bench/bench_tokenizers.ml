(** Benchmark suite for Saga tokenizers library *)

open Saga_tokenizers

(** Sample texts for benchmarking *)
let sample_texts = [
  "The quick brown fox jumps over the lazy dog.";
  "Hello, world! How are you doing today?";
  "Machine learning and natural language processing are fascinating fields of study.";
  "This is a longer sentence with multiple clauses, punctuation marks, and various word lengths to test tokenization performance.";
  "1234567890 !@#$%^&*() Testing special characters and numbers in tokenization.";
]

(** Helper to create a simple BPE tokenizer *)
let create_bpe_tokenizer () =
  let vocab = [
    ("h", 0); ("e", 1); ("l", 2); ("o", 3); ("w", 4);
    ("r", 5); ("d", 6); ("t", 7); ("i", 8); ("n", 9);
    ("g", 10); (" ", 11); ("!", 12); (".", 13); (",", 14);
    ("he", 15); ("ll", 16); ("lo", 17); ("wo", 18); ("or", 19);
    ("ld", 20); ("th", 21); ("in", 22); ("ng", 23);
  ] in
  let merges = [
    ("h", "e"); ("l", "l"); ("l", "o"); ("w", "o");
    ("o", "r"); ("l", "d"); ("t", "h"); ("i", "n"); ("n", "g");
  ] in
  Tokenizer.bpe ~vocab ~merges ()

(** Helper to create a WordPiece tokenizer *)
let create_wordpiece_tokenizer () =
  let vocab = [
    ("[PAD]", 0); ("[UNK]", 1); ("[CLS]", 2); ("[SEP]", 3);
    ("the", 4); ("quick", 5); ("brown", 6); ("fox", 7);
    ("jumps", 8); ("over", 9); ("lazy", 10); ("dog", 11);
    ("hello", 12); ("world", 13); ("how", 14); ("are", 15);
    ("you", 16); ("doing", 17); ("today", 18);
    ("##ing", 19); ("##ed", 20); ("##s", 21);
  ] in
  Tokenizer.wordpiece ~vocab ~unk_token:"[UNK]" ~continuing_subword_prefix:"##" ()

(** Helper to create a word-level tokenizer *)
let create_wordlevel_tokenizer () =
  let vocab = [
    ("the", 0); ("quick", 1); ("brown", 2); ("fox", 3);
    ("jumps", 4); ("over", 5); ("lazy", 6); ("dog", 7);
    ("hello", 8); ("world", 9); ("how", 10); ("are", 11);
    ("you", 12); ("doing", 13); ("today", 14); ("[UNK]", 15);
  ] in
  Tokenizer.word_level ~vocab ~unk_token:"[UNK]"
    ~pre:(Pre_tokenizers.whitespace ()) ()

(** Parameterized encoding benchmarks - core performance metric *)
module Encoding = struct
  let text_sizes = [
    ("100 chars", String.make 100 'a');
    ("1K chars", String.make 1000 'a');
    ("10K chars", String.make 10000 'a');
  ]

  let bench_bpe =
    let tok = create_bpe_tokenizer () in
    Ubench.bench_param "BPE encode"
      (fun ~param -> Tokenizer.encode tok param |> ignore)
      ~params:text_sizes

  let bench_wordpiece =
    let tok = create_wordpiece_tokenizer () in
    Ubench.bench_param "WordPiece encode"
      (fun ~param -> Tokenizer.encode tok param |> ignore)
      ~params:text_sizes

  let bench_wordlevel =
    let tok = create_wordlevel_tokenizer () in
    Ubench.bench_param "WordLevel encode"
      (fun ~param -> Tokenizer.encode tok param |> ignore)
      ~params:text_sizes

  let all = bench_bpe @ bench_wordpiece @ bench_wordlevel
end

(** Decoding benchmarks - important for text generation *)
module Decoding = struct
  let token_counts = [
    ("100 tokens", Array.init 100 (fun i -> i mod 10));
    ("1K tokens", Array.init 1000 (fun i -> i mod 10));
  ]

  let bench_bpe =
    let tok = create_bpe_tokenizer () in
    Ubench.bench_param "BPE decode"
      (fun ~param -> Tokenizer.decode tok param |> ignore)
      ~params:token_counts

  let bench_wordpiece =
    let tok = create_wordpiece_tokenizer () in
    Ubench.bench_param "WordPiece decode"
      (fun ~param -> Tokenizer.decode tok param |> ignore)
      ~params:token_counts

  let all = bench_bpe @ bench_wordpiece
end

(** Batch encoding benchmarks - important for real-world usage *)
module Batch = struct
  let batch_sizes = [
    ("10 items", 10);
    ("100 items", 100);
  ]

  let bench_bpe =
    let tok = create_bpe_tokenizer () in
    Ubench.bench_param "BPE batch"
      (fun ~param:size ->
        let batch = List.init size (fun i -> List.nth sample_texts (i mod List.length sample_texts)) in
        Tokenizer.encode_batch tok batch |> ignore)
      ~params:batch_sizes

  let bench_wordpiece =
    let tok = create_wordpiece_tokenizer () in
    Ubench.bench_param "WordPiece batch"
      (fun ~param:size ->
        let batch = List.init size (fun i -> List.nth sample_texts (i mod List.length sample_texts)) in
        Tokenizer.encode_batch tok batch |> ignore)
      ~params:batch_sizes

  let all = bench_bpe @ bench_wordpiece
end

(** Serialization benchmarks - I/O performance *)
module Serialization = struct
  let bench_to_json () =
    let tok = create_wordpiece_tokenizer () in
    Ubench.bench "to_json" (fun () ->
      Tokenizer.to_json tok |> ignore
    )

  let bench_from_file () =
    Ubench.bench_with_setup "from_file"
      ~setup:(fun () ->
        let tok = create_wordpiece_tokenizer () in
        let temp_dir = Filename.get_temp_dir_name () in
        let path = Filename.concat temp_dir "bench_tokenizer_load" in
        Tokenizer.save_pretrained tok ~path;
        let file = Filename.concat path "tokenizer.json" in
        (file, path)
      )
      ~teardown:(fun (file, path) ->
        (try Sys.remove file with _ -> ());
        (try Unix.rmdir path with _ -> ())
      )
      ~f:(fun (file, _path) ->
        match Tokenizer.from_file file with
        | Ok _ -> ()
        | Error _ -> failwith "Failed to load tokenizer"
      )

  let all = [
    bench_to_json ();
    bench_from_file ();
  ]
end

(** Vocabulary operations benchmarks *)
module Vocab = struct
  let bench_add_tokens () =
    let tok = create_wordlevel_tokenizer () in
    let new_tokens = List.init 100 (fun i -> Printf.sprintf "token%d" i) in
    Ubench.bench "add_tokens (100)" (fun () ->
      Tokenizer.add_tokens tok new_tokens |> ignore
    )

  let all = [
    bench_add_tokens ();
  ]
end

(** All benchmarks grouped *)
let all_benchmarks = [
  Ubench.group "Encoding" Encoding.all;
  Ubench.group "Decoding" Decoding.all;
  Ubench.group "Batch" Batch.all;
  Ubench.group "Serialization" Serialization.all;
  Ubench.group "Vocab" Vocab.all;
]

(** Main entry point *)
let () = Ubench.run_cli all_benchmarks
