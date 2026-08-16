(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Brot
open Windtrap

let with_hf_tokenizer model f =
  let relative =
    Filename.concat "fixtures/hf" (Filename.concat model "tokenizer.json")
  in
  Fixture.with_download relative
    ~from:"packages/brot/test/scripts/download_hf_tokenizers.py" (fun path ->
      match from_file path with
      | Ok tok -> f tok
      | Error msg -> failf "Failed to load tokenizer %s: %s" model msg)

let test_bert_base_uncased () =
  with_hf_tokenizer "bert-base-uncased" (fun tok ->
      let encoding = encode tok "Hello world!" in
      let tokens = Encoding.tokens encoding |> Array.to_list in
      equal ~msg:"token sequence" (list string)
        [ "[CLS]"; "hello"; "world"; "!"; "[SEP]" ]
        tokens;
      let type_ids = Encoding.type_ids encoding |> Array.to_list in
      equal ~msg:"type ids" (list int) [ 0; 0; 0; 0; 0 ] type_ids;
      equal ~msg:"has [MASK]" bool true
        (Option.is_some (token_to_id tok "[MASK]")))

let test_gpt2_small () =
  with_hf_tokenizer "gpt2" (fun tok ->
      let encoding = encode tok "Hello world" in
      let ids = Encoding.ids encoding |> Array.to_list in
      equal ~msg:"ids" (list int) [ 15496; 995 ] ids;
      let roundtrip =
        decode tok (Array.of_list ids) ~skip_special_tokens:true
      in
      equal ~msg:"decode" string "Hello world" roundtrip)

let test_roberta_base () =
  with_hf_tokenizer "roberta-base" (fun tok ->
      let encoding = encode tok "A quick test" in
      let tokens = Encoding.tokens encoding |> Array.to_list in
      equal ~msg:"tokens" (list string)
        [ "<s>"; "A"; "Ġquick"; "Ġtest"; "</s>" ]
        tokens;
      let attention = Encoding.attention_mask encoding |> Array.to_list in
      equal ~msg:"attention mask" (list int) [ 1; 1; 1; 1; 1 ] attention)

let () =
  run "HF tokenizers"
    [
      group "bert-base-uncased" [ test "encode" test_bert_base_uncased ];
      group "gpt2" [ test "encode" test_gpt2_small ];
      group "roberta-base" [ test "encode" test_roberta_base ];
    ]
