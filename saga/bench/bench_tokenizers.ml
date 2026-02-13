(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Benchmark suite for Saga tokenizers using realistic fixtures. *)

open Saga

module Fixtures = struct
  let data_dir = Filename.concat (Sys.getcwd ()) "saga/bench/data"

  let read_file name =
    let path = Filename.concat data_dir name in
    let ic = open_in_bin path in
    let len = in_channel_length ic in
    let content = really_input_string ic len in
    close_in ic;
    content

  let load_tokenizer name =
    let path = Filename.concat data_dir name in
    match Tokenizer.from_file path with
    | Ok tok -> tok
    | Error err ->
        failwith
          (Printf.sprintf "Failed to load tokenizer %s: %s" path
             (Printexc.to_string err))

  let short_text = read_file "news_1k.txt"
  let long_text = read_file "wiki_64k.txt"

  let batch_32 =
    let rec loop acc remaining =
      if remaining = 0 then List.rev acc
      else loop (short_text :: acc) (remaining - 1)
    in
    loop [] 32
end

let encode_single tok text = Tokenizer.encode tok text |> ignore
let encode_batch tok texts = Tokenizer.encode_batch tok texts |> ignore
let decode_ids tok ids = Tokenizer.decode tok ids |> ignore

let make_suite ~label ~tokenizer =
  let open Fixtures in
  let decode_input =
    let encoding = Tokenizer.encode tokenizer long_text in
    Array.copy (Encoding.get_ids encoding)
  in
  let benches =
    [
      Ubench.bench "Encode/single_short" (fun () ->
          encode_single tokenizer short_text);
      Ubench.bench "Encode/single_long" (fun () ->
          encode_single tokenizer long_text);
      Ubench.bench "Encode/batch_32" (fun () -> encode_batch tokenizer batch_32);
      Ubench.bench "Decode/long" (fun () -> decode_ids tokenizer decode_input);
    ]
  in
  Ubench.group label benches

let all_benchmarks =
  let open Fixtures in
  let gpt2 =
    make_suite ~label:"GPT-2" ~tokenizer:(load_tokenizer "gpt2.json")
  in
  let bert =
    make_suite ~label:"BERT-base" ~tokenizer:(load_tokenizer "bert_base.json")
  in
  let llama =
    make_suite ~label:"LLaMA" ~tokenizer:(load_tokenizer "llama.json")
  in
  [ gpt2; bert; llama ]

let () = Ubench.run_cli all_benchmarks
