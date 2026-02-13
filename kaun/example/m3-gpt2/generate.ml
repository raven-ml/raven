(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rune
module GPT2 = Kaun_models.GPT2

let sample_token ~temperature ~top_k logits_array =
  let n = Array.length logits_array in
  let logits = Array.map (fun x -> x /. temperature) logits_array in
  (* Top-k filtering *)
  let logits =
    match top_k with
    | None -> logits
    | Some k when k >= n -> logits
    | Some k ->
        let sorted = Array.copy logits in
        Array.sort (fun a b -> compare b a) sorted;
        let threshold = sorted.(k - 1) in
        Array.map
          (fun x -> if x < threshold then Float.neg_infinity else x)
          logits
  in
  (* Softmax + sample *)
  let max_l = Array.fold_left Stdlib.max Float.neg_infinity logits in
  let exp_l = Array.map (fun x -> Stdlib.exp (x -. max_l)) logits in
  let sum = Array.fold_left ( +. ) 0.0 exp_l in
  let probs = Array.map (fun x -> x /. sum) exp_l in
  let r = Random.float 1.0 in
  let cumsum = ref 0.0 in
  let result = ref 0 in
  (try
     for i = 0 to Array.length probs - 1 do
       cumsum := !cumsum +. probs.(i);
       if !cumsum > r then begin
         result := i;
         raise_notrace Exit
       end
     done
   with Exit -> ());
  !result

let () =
  Printf.printf "Loading GPT-2 model...\n";
  let gpt2 = GPT2.from_pretrained ~dtype:Float32 () in
  let tokenizer = GPT2.Tokenizer.create () in

  let prompts =
    [
      ("The future of artificial intelligence", 1.0, Some 50);
      ("Once upon a time", 0.8, Some 40);
      ("In a groundbreaking discovery", 0.9, Some 30);
    ]
  in

  List.iter
    (fun (prompt, temperature, top_k) ->
      Printf.printf "\n=== Generation ===\n";
      Printf.printf "Prompt: %s\n" prompt;
      Printf.printf "Settings: temp=%.1f, top_k=%s\n" temperature
        (match top_k with Some k -> string_of_int k | None -> "all");

      let tokens = ref (GPT2.Tokenizer.encode_to_array tokenizer prompt) in
      let max_new_tokens = 10 in

      for i = 1 to max_new_tokens do
        let ba =
          Bigarray.Array2.of_array Bigarray.int32 Bigarray.c_layout
            [| Array.map Int32.of_int !tokens |]
        in
        let input_ids = of_bigarray (Bigarray.genarray_of_array2 ba) in

        let logits, _ =
          GPT2.For_causal_lm.forward ~model:gpt2.GPT2.model
            ~params:gpt2.GPT2.params ~compute_dtype:gpt2.GPT2.dtype ~input_ids
            ~training:false ()
        in

        let seq_len = shape input_ids |> fun s -> s.(1) in
        let last_logits = slice [ A; I (seq_len - 1); A ] logits in

        let vocab_size = shape last_logits |> fun s -> s.(1) in
        let logits_array = Array.make vocab_size 0.0 in
        for j = 0 to vocab_size - 1 do
          logits_array.(j) <- item [ 0; j ] last_logits
        done;

        let next_token = sample_token ~temperature ~top_k logits_array in

        tokens := Array.append !tokens [| next_token |];

        if i mod 5 = 0 then Printf.printf "  Generated %d tokens...\n" i
      done;

      let generated_text = GPT2.Tokenizer.decode tokenizer !tokens in
      Printf.printf "Generated: %s\n" generated_text;

      let prompt_tokens = GPT2.Tokenizer.encode_to_array tokenizer prompt in
      let new_tokens_count =
        Array.length !tokens - Array.length prompt_tokens
      in
      Printf.printf "New tokens generated: %d\n" new_tokens_count)
    prompts;

  Printf.printf "\nGeneration complete!\n"
