(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Saga_tokenizers
open Alcotest

let candidate_roots () =
  match Sys.getenv_opt "DUNE_SOURCEROOT" with
  | Some root -> [ root; Sys.getcwd () ]
  | None -> [ Sys.getcwd () ]

let locate_fixture model =
  let relative =
    Filename.concat "saga/test/fixtures/hf"
      (Filename.concat model "tokenizer.json")
  in
  let rec search = function
    | [] -> None
    | root :: rest ->
        let path = Filename.concat root relative in
        if Sys.file_exists path then Some path else search rest
  in
  search (candidate_roots ())

let with_hf_tokenizer model f =
  match locate_fixture model with
  | None -> skip ()
  | Some path -> (
      match Tokenizer.from_file path with
      | Ok tok -> f tok
      | Error exn ->
          failf "Failed to load tokenizer %s: %s" model (Printexc.to_string exn)
      )

let test_bert_base_uncased () =
  with_hf_tokenizer "bert-base-uncased" (fun tok ->
      let encoding = Tokenizer.encode tok "Hello world!" in
      let tokens = Encoding.get_tokens encoding |> Array.to_list in
      check (list string) "token sequence"
        [ "[CLS]"; "hello"; "world"; "!"; "[SEP]" ]
        tokens;
      let type_ids = Encoding.get_type_ids encoding |> Array.to_list in
      check (list int) "type ids" [ 0; 0; 0; 0; 0 ] type_ids;
      check bool "has [MASK]" true
        (Option.is_some (Tokenizer.token_to_id tok "[MASK]")))

let test_gpt2_small () =
  with_hf_tokenizer "gpt2" (fun tok ->
      let encoding = Tokenizer.encode tok "Hello world" in
      let ids = Encoding.get_ids encoding |> Array.to_list in
      check (list int) "ids" [ 15496; 995 ] ids;
      let roundtrip =
        Tokenizer.decode tok (Array.of_list ids) ~skip_special_tokens:true
      in
      check string "decode" "Hello world" roundtrip)

let test_roberta_base () =
  with_hf_tokenizer "roberta-base" (fun tok ->
      let encoding = Tokenizer.encode tok "A quick test" in
      let tokens = Encoding.get_tokens encoding |> Array.to_list in
      check (list string) "tokens"
        [ "<s>"; "A"; "Ġquick"; "Ġtest"; "</s>" ]
        tokens;
      let attention = Encoding.get_attention_mask encoding |> Array.to_list in
      check (list int) "attention mask" [ 1; 1; 1; 1; 1 ] attention)

let () =
  run "HF tokenizers"
    [
      ("bert-base-uncased", [ test_case "encode" `Quick test_bert_base_uncased ]);
      ("gpt2", [ test_case "encode" `Quick test_gpt2_small ]);
      ("roberta-base", [ test_case "encode" `Quick test_roberta_base ]);
    ]
