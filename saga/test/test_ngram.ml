open Alcotest
module Ngram = Saga_models.Ngram
module Sampler = Saga.Sampler

let train ~order ?(smoothing = `Add_k 1.0) sequences =
  Ngram.of_sequences ~order ~smoothing sequences

let test_stats () =
  let seqs = [ [| 0; 1; 2; 1 |]; [| 1; 2; 3 |]; [| 2; 3; 4 |] ] in
  let model = train ~order:2 seqs in
  let stats = Ngram.stats model in
  check int "vocab" 5 stats.vocab_size;
  check int "total" 10 stats.total_tokens;
  check bool "unique" true (stats.unique_ngrams > 0)

let test_logits () =
  let seqs = [ [| 0; 1; 0; 1; 0; 2 |] ] in
  let model = train ~order:2 seqs in
  let context0 = Ngram.logits model ~context:[| 0 |] in
  check bool "1 more likely after 0" true (context0.(1) > context0.(2));
  let context1 = Ngram.logits model ~context:[| 1 |] in
  check bool "0 more likely after 1" true (context1.(0) > context1.(2));
  check bool "negative log probs" true (Array.for_all (( > ) 0.0) context0)

let test_log_prob_perplexity () =
  let seqs = [ [| 0; 1; 2; 1 |]; [| 1; 2; 1; 3 |] ] in
  let model = train ~order:2 seqs in
  let tokens = [| 0; 1; 2; 1 |] in
  let lp = Ngram.log_prob model tokens in
  let ppl = Ngram.perplexity model tokens in
  check bool "log prob negative" true (lp < 0.0);
  check bool "perplexity positive" true (ppl > 0.0)

let drop n lst =
  let rec aux n acc = function
    | [] -> List.rev acc
    | l when n <= 0 -> List.rev_append acc l
    | _ :: tl -> aux (n - 1) acc tl
  in
  aux n [] lst

let test_generation () =
  let seqs = [ [| 0; 1; 0; 1; 0; 1 |] ] in
  let model = train ~order:2 seqs in
  let logits_fn history = Ngram.logits model ~context:(Array.of_list history) in
  let output =
    Sampler.generate ~model:logits_fn ~input_ids:[ 0 ]
      ~generation_config:
        (Sampler.default
        |> Sampler.with_temperature 0.0001
        |> Sampler.with_max_new_tokens 2
        |> Sampler.with_do_sample false)
      ()
  in
  let continuation =
    match output.sequences with seq :: _ -> drop 1 seq | [] -> []
  in
  check (list int) "greedy continuation" [ 1; 0 ] continuation

let tests =
  [
    test_case "stats" `Quick test_stats;
    test_case "logits" `Quick test_logits;
    test_case "log-prob" `Quick test_log_prob_perplexity;
    test_case "generation" `Quick test_generation;
  ]

let () = Alcotest.run "saga ngram" [ ("ngram", tests) ]
