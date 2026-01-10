(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rune
open Kaun_models.GPT2

let expected_logits =
  [|
    -9.088777005672455e-06;
    -0.14020976424217224;
    -0.20845119655132294;
    -0.028111519291996956;
    -0.09017819911241531;
    -0.19850760698318481;
    4.072483062744141;
    -0.24899107217788696;
    -0.17168566584587097;
    -0.01689625345170498;
  |]

let tolerance = 1e-4

let run_comparison () =
  let model = from_pretrained ~model_id:"gpt2" ~dtype:float32 () in
  let tokenizer = Tokenizer.create () in
  let inputs = Tokenizer.encode tokenizer "Hello world" in
  let outputs = forward model inputs () in
  let hidden = outputs.last_hidden_state in
  let dims = Rune.shape hidden in
  Alcotest.(check int) "batch" 1 dims.(0);
  Alcotest.(check int) "hidden-size" 768 dims.(2);
  let first_position = Rune.slice [ I 0; I 0; A ] hidden in
  let first_values =
    Array.init (Array.length expected_logits) (fun i ->
        Rune.item [ i ] first_position)
  in
  Array.iteri
    (fun idx expected ->
      let actual = first_values.(idx) in
      let diff = Float.abs (actual -. expected) in
      if diff > tolerance then
        Alcotest.failf "logit[%d]: expected %.6f got %.6f (diff %.6g)" idx
          expected actual diff)
    expected_logits

let () =
  Printexc.record_backtrace true;
  Alcotest.run "GPT-2 parity"
    [
      ( "gpt2",
        [ Alcotest.test_case "compare-with-python" `Quick run_comparison ] );
    ]
