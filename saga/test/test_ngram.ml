open Alcotest
open Saga_models

(* Test helpers *)
let float_array_testable =
  testable
    (fun fmt arr ->
      Format.fprintf fmt "[%s]"
        (String.concat "; " (Array.to_list (Array.map string_of_float arr))))
    (fun a b ->
      Array.length a = Array.length b
      && Array.for_all2 (fun x y -> abs_float (x -. y) < 1e-6) a b)

let _int_array_testable =
  testable
    (fun fmt arr ->
      Format.fprintf fmt "[%s]"
        (String.concat "; " (Array.to_list (Array.map string_of_int arr))))
    ( = )

let stats_testable =
  testable
    (fun fmt s ->
      Format.fprintf fmt
        "{ vocab_size = %d; total_tokens = %d; unique_ngrams = %d }"
        s.Ngram.vocab_size s.Ngram.total_tokens s.Ngram.unique_ngrams)
    ( = )

(* Test data *)
let simple_tokens = [| 0; 1; 2; 1; 3; 2; 1; 0 |]
let _text_tokens = [| 10; 11; 12; 13; 11; 12; 14; 15; 11; 12 |]
(* "the cat sat on cat sat mat dog cat sat" *)

(* Unigram tests *)
let test_unigram_train () =
  let model = Ngram.Unigram.train simple_tokens in
  let stats = Ngram.Unigram.stats model in

  check int "vocab_size should be correct" 4 stats.vocab_size;
  check int "total_tokens should be correct" 8 stats.total_tokens;
  check int "unique_ngrams should be correct" 4 stats.unique_ngrams

let test_unigram_logits () =
  let model = Ngram.Unigram.train simple_tokens in
  let logits = Ngram.Unigram.logits model 0 in
  (* prev token ignored *)

  check int "logits array should have vocab_size length" 4 (Array.length logits);

  (* Token 1 appears 3 times, should have highest probability *)
  let max_idx = ref 0 in
  let max_val = ref logits.(0) in
  for i = 1 to Array.length logits - 1 do
    if logits.(i) > !max_val then (
      max_val := logits.(i);
      max_idx := i)
  done;
  check int "most frequent token should have highest logit" 1 !max_idx

let test_unigram_sample () =
  let model = Ngram.Unigram.train simple_tokens in

  (* With very low temperature, should be deterministic *)
  let token1 = Ngram.Unigram.sample model ~temperature:0.01 ~seed:42 () in
  let token2 = Ngram.Unigram.sample model ~temperature:0.01 ~seed:42 () in
  check int "same seed should give same result" token1 token2;

  (* Should sample valid tokens *)
  for _ = 1 to 10 do
    let token = Ngram.Unigram.sample model ~seed:(Random.int 1000) () in
    check bool "sampled token should be in range" true (token >= 0 && token < 4)
  done

let test_unigram_from_corpus () =
  let corpus = [ [| 1; 2; 3 |]; [| 2; 3; 4 |]; [| 1; 2; 1 |] ] in
  let model = Ngram.Unigram.train_from_corpus corpus in
  let stats = Ngram.Unigram.stats model in

  check int "total_tokens from corpus" 9 stats.total_tokens;
  check int "vocab_size from corpus" 5 stats.vocab_size

(* Bigram tests *)
let test_bigram_train () =
  let model = Ngram.Bigram.train simple_tokens in
  let stats = Ngram.Bigram.stats model in

  check int "vocab_size should be correct" 4 stats.vocab_size;
  check int "total_tokens should be pairs count" 7 stats.total_tokens;
  check bool "unique_ngrams should be positive" true (stats.unique_ngrams > 0)

let test_bigram_logits () =
  let tokens = [| 0; 1; 0; 1; 0; 2 |] in
  (* 0->1 appears twice, 1->0 appears twice *)
  let model = Ngram.Bigram.train ~smoothing:0.1 tokens in

  (* After 0, we should see 1 more often *)
  let logits_after_0 = Ngram.Bigram.logits model 0 in
  check bool "1 should be more likely after 0" true
    (logits_after_0.(1) > logits_after_0.(2));

  (* After 1, we should see 0 more often *)
  let logits_after_1 = Ngram.Bigram.logits model 1 in
  check bool "0 should be more likely after 1" true
    (logits_after_1.(0) > logits_after_1.(2))

let test_bigram_smoothing () =
  let tokens = [| 0; 1; 2 |] in

  (* Without smoothing, unseen transitions should have very low probability *)
  let model_no_smooth = Ngram.Bigram.train ~smoothing:0.0001 tokens in
  let logits = Ngram.Bigram.logits model_no_smooth 0 in

  (* Token 0->2 never appears, should have very low probability *)
  check bool "unseen transition should have low probability" true
    (logits.(2) < logits.(1));

  (* With smoothing, probabilities should be more uniform *)
  let model_smooth = Ngram.Bigram.train ~smoothing:1.0 tokens in
  let logits_smooth = Ngram.Bigram.logits model_smooth 0 in

  (* The difference should be smaller with smoothing *)
  let diff_no_smooth = abs_float (logits.(1) -. logits.(2)) in
  let diff_smooth = abs_float (logits_smooth.(1) -. logits_smooth.(2)) in
  check bool "smoothing should reduce probability differences" true
    (diff_smooth < diff_no_smooth)

let test_bigram_sample () =
  let tokens = [| 0; 1; 0; 1; 0; 1 |] in
  (* 0 always followed by 1 *)
  let model = Ngram.Bigram.train ~smoothing:0.01 tokens in

  (* With very low temperature after 0, should almost always get 1 *)
  let mut_count = ref 0 in
  for _ = 1 to 20 do
    let next =
      Ngram.Bigram.sample model ~prev:0 ~temperature:0.01
        ~seed:(Random.int 1000) ()
    in
    if next = 1 then incr mut_count
  done;
  check bool "should mostly sample the most likely token" true (!mut_count > 15)

let test_bigram_log_prob () =
  let tokens = [| 0; 1; 0; 1; 0; 2 |] in
  let model = Ngram.Bigram.train ~smoothing:1.0 tokens in

  (* 0->1 appears twice, should have higher probability than 0->2 (once) *)
  let log_prob_01 = Ngram.Bigram.log_prob model ~prev:0 ~next:1 in
  let log_prob_02 = Ngram.Bigram.log_prob model ~prev:0 ~next:2 in
  check bool "frequent transition should have higher log prob" true
    (log_prob_01 > log_prob_02);

  (* Log probabilities should be negative *)
  check bool "log probabilities should be negative" true
    (log_prob_01 < 0.0 && log_prob_02 < 0.0)

(* Trigram tests *)
let test_trigram_train () =
  let tokens = [| 0; 1; 2; 0; 1; 2; 0; 1; 3 |] in
  let model = Ngram.Trigram.train tokens in
  let stats = Ngram.Trigram.stats model in

  check int "vocab_size should be correct" 4 stats.vocab_size;
  check bool "should have trigrams" true (stats.unique_ngrams > 0)

let test_trigram_logits () =
  let tokens = [| 0; 1; 2; 0; 1; 2; 0; 1; 3 |] in
  (* 0,1->2 appears twice, 0,1->3 once *)
  let model = Ngram.Trigram.train ~smoothing:0.1 tokens in

  let logits = Ngram.Trigram.logits model ~prev1:0 ~prev2:1 in

  (* After 0,1 we see 2 twice and 3 once *)
  check bool "2 should be more likely than 3 after 0,1" true
    (logits.(2) > logits.(3))

let test_trigram_sample () =
  let tokens = [| 0; 1; 2; 0; 1; 2; 0; 1; 2 |] in
  (* 0,1 always followed by 2 *)
  let model = Ngram.Trigram.train ~smoothing:0.01 tokens in

  (* Should almost always sample 2 after 0,1 *)
  let mut_count = ref 0 in
  for _ = 1 to 20 do
    let next =
      Ngram.Trigram.sample model ~prev1:0 ~prev2:1 ~temperature:0.01
        ~seed:(Random.int 1000) ()
    in
    if next = 2 then incr mut_count
  done;
  check bool "should mostly sample the most likely token" true (!mut_count > 15)

(* Save/Load tests *)
let test_save_load_unigram () =
  let model = Ngram.Unigram.train simple_tokens in
  let temp_file = Filename.temp_file "test_unigram" ".model" in

  Ngram.Unigram.save model temp_file;
  let loaded = Ngram.Unigram.load temp_file in

  let stats1 = Ngram.Unigram.stats model in
  let stats2 = Ngram.Unigram.stats loaded in

  check stats_testable "loaded model should have same stats" stats1 stats2;

  (* Test that logits are the same *)
  let logits1 = Ngram.Unigram.logits model 0 in
  let logits2 = Ngram.Unigram.logits loaded 0 in
  check float_array_testable "loaded model should produce same logits" logits1
    logits2;

  Sys.remove temp_file

let test_save_load_bigram () =
  let model = Ngram.Bigram.train simple_tokens in
  let temp_file = Filename.temp_file "test_bigram" ".model" in

  Ngram.Bigram.save model temp_file;
  let loaded = Ngram.Bigram.load temp_file in

  let stats1 = Ngram.Bigram.stats model in
  let stats2 = Ngram.Bigram.stats loaded in

  check stats_testable "loaded model should have same stats" stats1 stats2;

  (* Test that logits are the same *)
  let logits1 = Ngram.Bigram.logits model 0 in
  let logits2 = Ngram.Bigram.logits loaded 0 in
  check float_array_testable "loaded model should produce same logits" logits1
    logits2;

  Sys.remove temp_file

(* Generic n-gram tests *)
let test_generic_ngram () =
  let model = Ngram.create ~n:2 ~smoothing:1.0 simple_tokens in
  let logits = Ngram.logits model ~context:[| 0 |] in

  check int "generic logits should return array" 100 (Array.length logits);

  (* Test perplexity *)
  let perplexity = Ngram.perplexity model simple_tokens in
  check bool "perplexity should be positive" true (perplexity > 0.0)

let test_generic_generate () =
  let tokens = [| 0; 1; 2; 0; 1; 2; 0; 1; 2 |] in
  let model = Ngram.create ~n:2 ~smoothing:0.1 tokens in

  let generated =
    Ngram.generate model ~max_tokens:5 ~temperature:1.0 ~seed:42 ()
  in

  check int "should generate requested tokens" 5 (Array.length generated);

  (* All generated tokens should be in vocabulary *)
  Array.iter
    (fun token ->
      check bool "generated token should be in range" true
        (token >= 0 && token < 100))
    generated

(* Edge cases *)
let test_empty_corpus () =
  (* Empty corpus should still create a valid model, just with no
     observations *)
  let model = Ngram.Unigram.train [||] in
  let stats = Ngram.Unigram.stats model in
  check int "empty corpus vocab_size" 1 stats.vocab_size;
  (* min vocab size is 1 *)
  check int "empty corpus total_tokens" 0 stats.total_tokens

let test_single_token () =
  let model = Ngram.Unigram.train [| 5 |] in
  let stats = Ngram.Unigram.stats model in

  check int "single token vocab" 6 stats.vocab_size;
  check int "single token total" 1 stats.total_tokens

let test_large_vocab () =
  (* Test with larger vocabulary *)
  let tokens = Array.init 1000 (fun i -> i mod 100) in
  let model = Ngram.Bigram.train tokens in
  let stats = Ngram.Bigram.stats model in

  check int "large vocab size" 100 stats.vocab_size;
  check bool "should have many bigrams" true (stats.unique_ngrams > 50)

(* Test suite *)
let () =
  run "Ngram"
    [
      ( "unigram",
        [
          test_case "train" `Quick test_unigram_train;
          test_case "logits" `Quick test_unigram_logits;
          test_case "sample" `Quick test_unigram_sample;
          test_case "from_corpus" `Quick test_unigram_from_corpus;
          test_case "save_load" `Quick test_save_load_unigram;
        ] );
      ( "bigram",
        [
          test_case "train" `Quick test_bigram_train;
          test_case "logits" `Quick test_bigram_logits;
          test_case "smoothing" `Quick test_bigram_smoothing;
          test_case "sample" `Quick test_bigram_sample;
          test_case "log_prob" `Quick test_bigram_log_prob;
          test_case "save_load" `Quick test_save_load_bigram;
        ] );
      ( "trigram",
        [
          test_case "train" `Quick test_trigram_train;
          test_case "logits" `Quick test_trigram_logits;
          test_case "sample" `Quick test_trigram_sample;
        ] );
      ( "generic",
        [
          test_case "ngram" `Quick test_generic_ngram;
          test_case "generate" `Quick test_generic_generate;
        ] );
      ( "edge_cases",
        [
          test_case "empty_corpus" `Quick test_empty_corpus;
          test_case "single_token" `Quick test_single_token;
          test_case "large_vocab" `Quick test_large_vocab;
        ] );
    ]
