(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Ngram = Saga.Ngram
module Sampler = Saga.Sampler

let train ~order ?(smoothing = `Add_k 1.0) sequences =
  Ngram.of_sequences ~order ~smoothing sequences

let test_stats () =
  let seqs = [ [| 0; 1; 2; 1 |]; [| 1; 2; 3 |]; [| 2; 3; 4 |] ] in
  let model = train ~order:2 seqs in
  let stats = Ngram.stats model in
  equal ~msg:"vocab" int 5 stats.vocab_size;
  equal ~msg:"total" int 10 stats.total_tokens;
  equal ~msg:"unique" bool true (stats.unique_ngrams > 0)

let test_logits () =
  let seqs = [ [| 0; 1; 0; 1; 0; 2 |] ] in
  let model = train ~order:2 seqs in
  let context0 = Ngram.logits model ~context:[| 0 |] in
  equal ~msg:"1 more likely after 0" bool true (context0.(1) > context0.(2));
  let context1 = Ngram.logits model ~context:[| 1 |] in
  equal ~msg:"0 more likely after 1" bool true (context1.(0) > context1.(2));
  equal ~msg:"negative log probs" bool true (Array.for_all (( > ) 0.0) context0)

let test_log_prob_perplexity () =
  let seqs = [ [| 0; 1; 2; 1 |]; [| 1; 2; 1; 3 |] ] in
  let model = train ~order:2 seqs in
  let tokens = [| 0; 1; 2; 1 |] in
  let lp = Ngram.log_prob model tokens in
  let ppl = Ngram.perplexity model tokens in
  equal ~msg:"log prob negative" bool true (lp < 0.0);
  equal ~msg:"perplexity positive" bool true (ppl > 0.0)

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
  equal ~msg:"greedy continuation" (list int) [ 1; 0 ] continuation

let tests =
  [
    test "stats" test_stats;
    test "logits" test_logits;
    test "log-prob" test_log_prob_perplexity;
    test "generation" test_generation;
  ]

let () = run "saga ngram" [ group "ngram" tests ]
