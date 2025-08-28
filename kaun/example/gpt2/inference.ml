(* GPT-2 Inference Example *)

(* Simple greedy generation without KV cache *)
let generate_greedy model params config prompt_tokens max_length =
  let device = Rune.c in
  let _dtype = Rune.float32 in

  (* Start with prompt *)
  let current_tokens = ref (Array.to_list prompt_tokens) in

  for _ = 1 to max_length do
    (* Convert current tokens to tensor *)
    let input_ids =
      let tokens_array = Array.of_list !current_tokens in
      let batch_size = 1 in
      let seq_len = Array.length tokens_array in

      (* Create float tensor for embeddings - convert token IDs to floats *)
      let ba =
        Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout batch_size
          seq_len
      in
      Array.iteri
        (fun i token_id -> ba.{0, i} <- float_of_int token_id)
        tokens_array;
      Rune.of_bigarray device (Bigarray.genarray_of_array2 ba)
    in

    (* Get model predictions *)
    let logits = Kaun.apply model params ~training:false input_ids in

    (* Get logits for last position *)
    let shape = Rune.shape logits in
    let seq_len = shape.(1) in
    let vocab_size = config.Config.vocab_size in

    (* Extract last token logits *)
    let last_logits =
      Rune.slice_ranges [ 0; seq_len - 1; 0 ] [ 1; seq_len; vocab_size ] logits
    in
    let last_logits = Rune.reshape [| vocab_size |] last_logits in

    (* Get next token (greedy - just argmax) *)
    let next_token_tensor = Rune.argmax last_logits in
    let next_token = Int32.to_int (Rune.unsafe_get [] next_token_tensor) in

    (* Add to sequence *)
    current_tokens := !current_tokens @ [ next_token ];

    (* Stop if we hit EOS token *)
    if next_token = config.Config.eos_token_id then ()
  done;

  Array.of_list !current_tokens

(* Generate with temperature and top-k sampling using Saga *)
let generate_with_sampling model params config prompt_tokens max_length
    ?(temperature = 1.0) ?(top_k = 50) ?(top_p = 0.9) () =
  let device = Rune.c in
  let _dtype = Rune.float32 in

  (* Create logits function for Saga sampler *)
  let current_tokens = ref (Array.to_list prompt_tokens) in

  let logits_fn _prev_token =
    (* Get full sequence logits *)
    let input_ids =
      let tokens_array = Array.of_list !current_tokens in
      let batch_size = 1 in
      let seq_len = Array.length tokens_array in

      (* Create float tensor for embeddings *)
      let ba =
        Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout batch_size
          seq_len
      in
      Array.iteri
        (fun i token_id -> ba.{0, i} <- float_of_int token_id)
        tokens_array;
      Rune.of_bigarray device (Bigarray.genarray_of_array2 ba)
    in

    let logits = Kaun.apply model params ~training:false input_ids in

    (* Get last position logits *)
    let shape = Rune.shape logits in
    let seq_len = shape.(1) in
    let vocab_size = config.Config.vocab_size in

    let last_logits =
      Rune.slice_ranges [ 0; seq_len - 1; 0 ] [ 1; seq_len; vocab_size ] logits
    in
    let last_logits = Rune.reshape [| vocab_size |] last_logits in

    (* Convert to float array for Saga *)
    Array.init vocab_size (fun i -> Rune.unsafe_get [ i ] last_logits)
  in

  (* Generate tokens *)
  let generated_tokens = ref [] in

  for _ = 1 to max_length do
    let logits = logits_fn 0 in

    (* Use Saga's sampler *)
    let next_token =
      Saga.Sampler.sample_token ~temperature ~top_k ~top_p logits
    in

    current_tokens := !current_tokens @ [ next_token ];
    generated_tokens := !generated_tokens @ [ next_token ];

    (* Stop at EOS *)
    if next_token = config.Config.eos_token_id then ()
  done;

  Array.of_list !generated_tokens

(* Main inference function *)
let infer_text text ?(model_name = "gpt2") ?(max_length = 50) () =
  Printf.printf "Loading GPT-2 model: %s\n" model_name;

  (* Get tokenizer *)
  Printf.printf "Loading tokenizer...\n";
  flush stdout;
  let tokenizer = Tokenizer.get_tokenizer () in
  Printf.printf "Tokenizer loaded.\n";
  flush stdout;

  (* For now, just use random weights for testing *)
  let config =
    match model_name with
    | "gpt2" -> Config.gpt2_small
    | "gpt2-medium" -> Config.gpt2_medium
    | "gpt2-large" -> Config.gpt2_large
    | "gpt2-xl" -> Config.gpt2_xl
    | _ -> failwith ("Unknown model: " ^ model_name)
  in
  let model = Gpt2.gpt2_lm_head ~config () in
  let device = Rune.c in
  let dtype = Rune.float32 in
  let key = Rune.Rng.key 42 in
  let params = Kaun.init model ~rngs:key ~device ~dtype in

  Printf.printf "Model loaded! Generating text...\n";
  Printf.printf "Input: %s\n" text;

  (* Tokenize input using proper GPT-2 tokenizer *)
  let tokens = Tokenizer.encode tokenizer text in

  Printf.printf "Input tokens (%d): %s\n" (Array.length tokens)
    (String.concat " " (Array.to_list (Array.map string_of_int tokens)));

  (* Generate continuation *)
  let generated =
    generate_with_sampling model params config tokens max_length
      ~temperature:0.8 ~top_k:40 ()
  in

  Printf.printf "Generated tokens (%d): %s\n" (Array.length generated)
    (String.concat " " (Array.to_list (Array.map string_of_int generated)));

  (* Decode tokens using proper tokenizer *)
  let full_tokens = Array.append tokens generated in
  let decoded = Tokenizer.decode tokenizer full_tokens in

  Printf.printf "Generated text: %s\n" decoded;
  decoded

(* Note: This is a simplified example. For production use: 1. Use proper GPT-2
   tokenizer (BPE) 2. Implement KV caching for efficiency 3. Handle special
   tokens properly 4. Add batch processing support *)
