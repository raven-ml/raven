(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Saga.Sampler

(* Test helpers *)
let float_array_testable =
  Testable.make
    ~pp:(fun fmt arr ->
      Format.fprintf fmt "[%s]"
        (String.concat "; " (Array.to_list (Array.map string_of_float arr))))
    ~equal:(fun a b ->
      Array.length a = Array.length b
      && Array.for_all2 (fun x y -> abs_float (x -. y) < 1e-6) a b)
    ()

let int_list_testable =
  Testable.make
    ~pp:(fun fmt lst ->
      Format.fprintf fmt "[%s]"
        (String.concat "; " (List.map string_of_int lst)))
    ~equal:( = )
    ()

(* Mock model for testing *)
let mock_model tokens =
  (* Simple mock that returns different logits based on context *)
  let vocab_size = 10 in
  let logits = Array.make vocab_size (-10.0) in
  (* Favor next sequential token *)
  let last_token = match List.rev tokens with [] -> 0 | h :: _ -> h in
  let next_token = (last_token + 1) mod vocab_size in
  logits.(next_token) <- 5.0;
  logits.((next_token + 1) mod vocab_size) <- 3.0;
  logits.((next_token + 2) mod vocab_size) <- 1.0;
  logits

(* Mock tokenizer and decoder *)
let mock_tokenizer text =
  String.to_seq text |> List.of_seq |> List.map Char.code

let mock_decoder tokens =
  List.map (fun i -> Char.chr (i mod 256)) tokens
  |> List.to_seq |> String.of_seq

(* Test temperature warper *)
let test_temperature_warper () =
  let logits = [| 1.0; 2.0; 3.0; 4.0 |] in

  (* Temperature = 1.0 should not change logits *)
  let warper = S.temperature_warper ~temperature:1.0 in
  let result = warper.process ~prompt_length:0 [] logits in
  equal ~msg:"temperature 1.0 should not change logits" float_array_testable
    logits result;

  (* Temperature = 2.0 should scale down *)
  let warper2 = S.temperature_warper ~temperature:2.0 in
  let result2 = warper2.process ~prompt_length:0 [] logits in
  let expected = [| 0.5; 1.0; 1.5; 2.0 |] in
  equal ~msg:"temperature 2.0 should halve logits" float_array_testable expected
    result2;

  (* Temperature = 0.5 should scale up *)
  let warper3 = S.temperature_warper ~temperature:0.5 in
  let result3 = warper3.process ~prompt_length:0 [] logits in
  let expected3 = [| 2.0; 4.0; 6.0; 8.0 |] in
  equal ~msg:"temperature 0.5 should double logits" float_array_testable
    expected3 result3

(* Test top-k warper *)
let test_top_k_warper () =
  let logits = [| 1.0; 4.0; 2.0; 3.0 |] in

  (* Top-k = 2 should keep only top 2 *)
  let warper = S.top_k_warper ~k:2 in
  let result = warper.process ~prompt_length:0 [] logits in
  equal ~msg:"index 1 should be kept (4.0)" bool true (result.(1) > -1e9);
  equal ~msg:"index 3 should be kept (3.0)" bool true (result.(3) > -1e9);
  equal ~msg:"index 0 should be filtered" bool true (result.(0) < -1e9);
  equal ~msg:"index 2 should be filtered" bool true (result.(2) < -1e9)

(* Test repetition penalty *)
let test_repetition_penalty () =
  let logits = [| 1.0; 2.0; 3.0; 4.0 |] in
  let previous_tokens = [ 1; 3 ] in

  let processor = S.repetition_penalty ~penalty:1.5 in
  let result = processor.process ~prompt_length:0 previous_tokens logits in

  (* Token 1 should be penalized (positive logit divided by penalty) *)
  equal ~msg:"token 1 should be penalized" (float 0.01) (2.0 /. 1.5) result.(1);
  (* Token 3 should be penalized *)
  equal ~msg:"token 3 should be penalized" (float 0.01) (4.0 /. 1.5) result.(3);
  (* Others unchanged *)
  equal ~msg:"token 0 unchanged" (float 0.01) 1.0 result.(0);
  equal ~msg:"token 2 unchanged" (float 0.01) 3.0 result.(2)

(* Test min_new_tokens processor *)
let test_min_new_tokens () =
  let logits = [| 1.0; 2.0; 3.0; 4.0 |] in
  let eos_token_ids = [ 3 ] in

  (* When we haven't generated enough tokens, EOS should be blocked *)
  let processor = S.min_new_tokens ~min_new_tokens:2 ~eos_token_ids in
  let tokens = [ 0 ] in
  (* Only 1 token generated after prompt *)
  let result = processor.process ~prompt_length:0 tokens logits in
  equal ~msg:"EOS should be blocked" bool true (result.(3) < -1e9);
  equal ~msg:"other tokens unchanged" (float 0.01) 1.0 result.(0);

  (* When we have enough tokens, EOS should be allowed *)
  let tokens2 = [ 0; 1; 2 ] in
  (* 3 tokens, enough *)
  let result2 = processor.process ~prompt_length:0 tokens2 logits in
  equal ~msg:"EOS should be allowed" (float 0.01) 4.0 result2.(3)

(* Test generation with config *)
let test_generation_config () =
  Random.init 42;

  (* Build config using builder pattern *)
  let config =
    S.default
    |> S.with_temperature 0.01 (* Almost greedy *)
    |> S.with_max_new_tokens 5 |> S.with_do_sample true
  in

  let output =
    S.generate ~model:mock_model ~input_ids:[ 0 ] ~generation_config:config ()
  in

  match output.sequences with
  | [ seq ] ->
      equal ~msg:"should generate sequence" bool true (List.length seq > 1);
      equal ~msg:"should not exceed max tokens" bool true (List.length seq <= 6)
  | _ -> fail "should generate exactly one sequence"

(* Test preset configurations *)
let test_presets () =
  (* Test creative writing preset *)
  let creative = S.creative_writing in
  equal ~msg:"creative temperature" (float 0.01) 0.8 creative.temperature;
  equal ~msg:"creative top_p" (float 0.01) 0.9 creative.top_p;
  equal ~msg:"creative repetition_penalty" (float 0.01) 1.2
    creative.repetition_penalty;

  (* Test factual preset *)
  let factual = S.factual in
  equal ~msg:"factual temperature" (float 0.01) 0.3 factual.temperature;
  equal ~msg:"factual top_k" int 10 factual.top_k;

  (* Test from_preset *)
  let chat = S.from_preset "chat" in
  equal ~msg:"chat temperature" (float 0.01) 0.7 chat.temperature

(* Test processor composition *)
let test_processor_composition () =
  let logits = [| 1.0; 2.0; 3.0; 4.0 |] in

  (* Compose temperature and top-k *)
  let composed = S.(temperature_warper ~temperature:2.0 @@ top_k_warper ~k:2) in
  let result = composed.process ~prompt_length:0 [] logits in

  (* Should apply both: first temperature scaling, then top-k *)
  equal ~msg:"high values should be kept" bool true
    (result.(3) > -1e9 && result.(3) < 3.0);
  (* Scaled and kept *)
  equal ~msg:"low values should be filtered" bool true (result.(0) < -1e9)

(* Test pipeline operator *)
let test_pipeline_operator () =
  let logits = [| 1.0; 2.0; 3.0; 4.0 |] in

  (* Use pipeline operator *)
  let result =
    logits
    |> S.( |>> ) (S.temperature_warper ~temperature:2.0)
    |> S.( |>> ) (S.top_k_warper ~k:3)
  in

  (* Check that transformations were applied *)
  equal ~msg:"should apply temperature" bool true (result.(3) < 3.0);
  equal ~msg:"should apply top-k filter" bool true (result.(0) < -1e9)

(* Test stopping criteria *)
let test_stopping_criteria () =
  let start_time = Unix.gettimeofday () in

  (* Max length criterion *)
  let max_len = S.max_length_criteria ~max_length:5 in
  equal ~msg:"should not stop before max" bool false
    (max_len.should_stop ~prompt_length:0 ~start_time [ 1; 2; 3 ]);
  equal ~msg:"should stop at max" bool true
    (max_len.should_stop ~prompt_length:0 ~start_time [ 1; 2; 3; 4; 5 ]);

  (* EOS token criterion *)
  let eos = S.eos_token_criteria ~eos_token_ids:[ 9 ] in
  equal ~msg:"should not stop without EOS" bool false
    (eos.should_stop ~prompt_length:0 ~start_time [ 1; 2; 3 ]);
  equal ~msg:"should stop with EOS" bool true
    (eos.should_stop ~prompt_length:0 ~start_time [ 1; 2; 9 ])

(* Test generation with processors *)
let test_generation_with_processors () =
  Random.init 42;

  let config = S.default |> S.with_max_new_tokens 10 |> S.with_do_sample true in

  (* Add custom processors *)
  let processors =
    [ S.temperature_warper ~temperature:0.8; S.repetition_penalty ~penalty:1.2 ]
  in

  let output =
    S.generate ~model:mock_model ~input_ids:[ 0 ] ~generation_config:config
      ~logits_processor:processors ()
  in

  equal ~msg:"should generate with processors" bool true
    (match output.sequences with [ seq ] -> List.length seq > 1 | _ -> false)

let () =
  run "Sampler"
    [
      group "processors"
        [
          test "temperature warper" test_temperature_warper;
          test "top-k warper" test_top_k_warper;
          test "repetition penalty" test_repetition_penalty;
          test "min_new_tokens" test_min_new_tokens;
          test "processor composition" test_processor_composition;
          test "pipeline operator" test_pipeline_operator;
        ];
      group "generation"
        [
          test "generation config" test_generation_config;
          test "presets" test_presets;
          test "stopping criteria" test_stopping_criteria;
          test "generation with processors" test_generation_with_processors;
        ];
    ]
