open Rune
module GPT2 = Kaun_models.GPT2
module Sampler = Saga.Sampler

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

      (* Generate just a few tokens for demonstration *)
      let tokens = ref (GPT2.Tokenizer.encode_to_array tokenizer prompt) in
      let max_new_tokens = 10 in
      (* Only generate 10 new tokens *)

      for i = 1 to max_new_tokens do
        (* Convert current tokens to tensor *)
        let ba =
          Bigarray.Array2.of_array Bigarray.int32 Bigarray.c_layout
            [| Array.map Int32.of_int !tokens |]
        in
        let input_ids = of_bigarray (Bigarray.genarray_of_array2 ba) in

        (* Run forward pass *)
        let logits, _ =
          GPT2.For_causal_lm.forward ~model:gpt2.GPT2.model
            ~params:gpt2.GPT2.params ~compute_dtype:gpt2.GPT2.dtype ~input_ids
            ~training:false ()
        in

        (* Get logits for the last token *)
        let seq_len = shape input_ids |> fun s -> s.(1) in
        let last_logits = slice [ A; I (seq_len - 1); A ] logits in

        (* Convert to float array *)
        let vocab_size = shape last_logits |> fun s -> s.(1) in
        let logits_array = Array.make vocab_size 0.0 in
        for j = 0 to vocab_size - 1 do
          logits_array.(j) <- item [ 0; j ] last_logits
        done;

        (* Sample next token using new API *)
        (* Create a simple model function that returns the logits *)
        let model_fn _tokens = logits_array in

        (* Configure generation for single token *)
        let config =
          Sampler.default
          |> Sampler.with_temperature temperature
          |> Sampler.with_max_new_tokens 1 (* Generate just 1 token *)
          |> Sampler.with_do_sample true
        in
        let config =
          match top_k with
          | Some k -> Sampler.with_top_k k config
          | None -> config
        in

        (* Generate single token *)
        let output =
          Sampler.generate ~model:model_fn ~input_ids:(Array.to_list !tokens)
            ~generation_config:config ()
        in

        (* Extract the generated token *)
        let next_token =
          match output.sequences with
          | [ seq ] ->
              (* Get the last token from the sequence *)
              List.hd (List.rev seq)
          | _ -> failwith "Unexpected generation output"
        in

        (* Add to tokens *)
        tokens := Array.append !tokens [| next_token |];

        (* Optional: print progress *)
        if i mod 5 = 0 then Printf.printf "  Generated %d tokens...\n" i
      done;

      (* Decode and display *)
      let generated_text = GPT2.Tokenizer.decode tokenizer !tokens in
      Printf.printf "Generated: %s\n" generated_text;

      (* Show new tokens count *)
      let prompt_tokens = GPT2.Tokenizer.encode_to_array tokenizer prompt in
      let new_tokens_count =
        Array.length !tokens - Array.length prompt_tokens
      in
      Printf.printf "New tokens generated: %d\n" new_tokens_count)
    prompts;

  Printf.printf "\n✓ Generation complete!\n"
