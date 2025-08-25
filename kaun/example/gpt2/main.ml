open Kaun

(* Example usage of GPT2 model with Kaun *)
let () =
  (* Test inference with pretrained weights *)
  let test_inference () =
    Printf.printf "Testing GPT-2 inference with pretrained weights...\n";
    
    (* Try to load a simple test first *)
    let text = Inference.infer_text "The quick brown fox" ~model_name:"gpt2" ~max_length:10 () in
    Printf.printf "Generated: %s\n" text
  in
  
  (* Test basic model creation *)
  let test_model_creation () =
    Printf.printf "Setting up device and dtype...\n";
    flush stdout;
    (* Set up device and dtype *)
    let device = Rune.c in
    let dtype = Rune.float32 in

    Printf.printf "Initializing RNG...\n";
    flush stdout;
    (* Initialize random number generator *)
    let key = Rune.Rng.key 0 in
    let rngs = key in

    Printf.printf "Getting model configuration...\n";
    flush stdout;
    (* Choose model configuration *)
    let config = Config.gpt2_small in

    Printf.printf "Creating model...\n";
    flush stdout;
    (* Create the model *)
    let model = Gpt2.gpt2_lm_head ~config () in

    Printf.printf "Initializing model parameters...\n";
    flush stdout;
    (* Initialize model parameters *)
    let params = Kaun.init model ~rngs ~device ~dtype in
    
    Printf.printf "Model parameters initialized.\n";
    flush stdout;

  (* Example: Generate text *)
  let _generate_text params prompt_ids max_length =
    (* Start with prompt *)
    let rec generate_loop current_ids length =
      if length >= max_length then current_ids
      else
        (* Get logits for next token *)
        let logits = Kaun.apply model params ~training:false current_ids in

        (* Get logits for last position *)
        let shape = Rune.shape current_ids in
        let batch_size = shape.(0) in
        let seq_len = shape.(1) in
        let last_logits =
          (* TODO: Need slicing - taking last position logits *)
          Rune.slice_ranges
            [ 0; seq_len - 1; 0 ]
            [ batch_size; seq_len; config.vocab_size ]
            logits
        in

        (* Apply temperature (optional) *)
        let temperature = 1.0 in
        let scaled_logits =
          Rune.div last_logits (Rune.scalar device dtype temperature)
        in

        (* Get probabilities *)
        let probs = Rune.softmax ~axes:[| -1 |] scaled_logits in

        (* Sample next token (for now, just take argmax) *)
        let next_token = Rune.argmax ~axis:(-1) ~keepdims:false probs in

        (* Append to sequence *)
        let _next_token = Rune.reshape [| batch_size; 1 |] next_token in
        (* TODO: concat not available, need to implement generation
           differently *)
        let new_ids = current_ids in
        (* Placeholder *)

        generate_loop new_ids (length + 1)
    in
    generate_loop prompt_ids 0
  in

  (* Example: Training step *)
  let _train_step params input_ids labels learning_rate =
    (* Define loss function *)
    let loss_fn params =
      (* Forward pass *)
      let logits = Kaun.apply model params ~training:true input_ids in

      (* Reshape for loss calculation *)
      let shape = Rune.shape logits in
      let batch_size = shape.(0) in
      let seq_len = shape.(1) in
      let vocab_size = config.Config.vocab_size in

      let logits_flat =
        Rune.reshape [| batch_size * seq_len; vocab_size |] logits
      in
      let labels_flat = Rune.reshape [| batch_size * seq_len |] labels in

      (* Cross-entropy loss *)
      (* TODO: Need cross_entropy implementation *)
      let loss =
        Loss.softmax_cross_entropy_with_indices logits_flat labels_flat
      in
      loss
    in

    (* Compute gradients *)
    let loss, grads = Kaun.value_and_grad loss_fn params in

    (* Use AdamW optimizer *)
    let optimizer = Optimizer.adamw ~lr:learning_rate () in
    let opt_state = optimizer.init params in
    let updates, _new_state = optimizer.update opt_state params grads in
    
    (* Apply updates to get new parameters *)
    let new_params = Ptree.map2 Rune.add params updates in
    (new_params, loss)
  in

  (* Example: Fine-tuning loop *)
  let _fine_tune params _dataset num_epochs _batch_size learning_rate =
    (* This is a simplified example - in practice you'd use Kaun_dataset *)
    let optimizer = Optimizer.adamw ~lr:learning_rate () in
    let opt_state = ref (optimizer.init params) in
    let current_params = ref params in
    
    for epoch = 1 to num_epochs do
      Printf.printf "Epoch %d/%d\n" epoch num_epochs;
      
      (* Dummy batch for example - replace with actual dataset iteration *)
      let batch_size = 2 in
      let seq_length = 128 in
      let input_ids =
        let ids =
          Rune.Rng.uniform (Rune.Rng.key (42 + epoch)) device dtype
            [| batch_size; seq_length |]
        in
        Rune.mul ids (Rune.scalar device dtype (float_of_int config.vocab_size))
      in
      let labels = input_ids in (* For language modeling, labels = inputs shifted *)
      
      (* Training step *)
      let loss, grads =
        Kaun.value_and_grad
          (fun params ->
            let logits = Kaun.apply model params ~training:true input_ids in
            let shape = Rune.shape logits in
            let batch_size = shape.(0) in
            let seq_len = shape.(1) in
            let vocab_size = config.Config.vocab_size in
            let logits_flat =
              Rune.reshape [| batch_size * seq_len; vocab_size |] logits
            in
            let labels_flat = Rune.reshape [| batch_size * seq_len |] labels in
            Loss.softmax_cross_entropy_with_indices logits_flat labels_flat)
          !current_params
      in
      
      (* Update weights *)
      let updates, new_state = optimizer.update !opt_state !current_params grads in
      opt_state := new_state;
      current_params := Ptree.map2 Rune.add !current_params updates;
      
      Printf.printf "  Loss: %.4f\n" (Rune.unsafe_get [] loss)
    done;
    
    !current_params
  in

  (* Example: Create dummy data for testing *)
  Printf.printf "Creating dummy input data...\n";
  flush stdout;
  let batch_size = 2 in
  let seq_length = 128 in
  let input_ids =
    (* Random token IDs for testing - using random float and casting *)
    Printf.printf "  Generating random uniform...\n";
    flush stdout;
    let ids =
      Rune.Rng.uniform (Rune.Rng.key 42) device dtype
        [| batch_size; seq_length |]
    in
    Printf.printf "  Scaling to vocab size...\n";
    flush stdout;
    let ids =
      Rune.mul ids (Rune.scalar device dtype (float_of_int config.vocab_size))
    in
    (* Cast to int for indexing - but embedding layer expects float, so keep as
       float *)
    ids
  in

  Printf.printf "Running forward pass...\n";
  flush stdout;
  (* Forward pass *)
  let output = Kaun.apply model params ~training:false input_ids in
  
  Printf.printf "Forward pass completed.\n";
  flush stdout;

  (* Print output shape *)
  let shape = Rune.shape output in
  Printf.printf "Output shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int shape)));

  Printf.printf "\nGPT2 Model initialized successfully!\n";
  Printf.printf "Model config:\n";
  Printf.printf "  - Vocab size: %d\n" config.vocab_size;
  Printf.printf "  - Embedding dim: %d\n" config.n_embd;
  Printf.printf "  - Num layers: %d\n" config.n_layer;
  Printf.printf "  - Num heads: %d\n" config.n_head;
  Printf.printf "  - Max position: %d\n" config.n_positions;

    Printf.printf "\nNote: This is a demonstration of the GPT2 architecture.\n";
    Printf.printf
      "Many operations are placeholders that need implementation in Kaun/Rune.\n"
  in
  
  (* Run tests *)
  Printf.printf "Starting GPT-2 example...\n";
  flush stdout;
  
  Printf.printf "Running model creation test...\n";
  flush stdout;
  test_model_creation ();
  
  Printf.printf "\nAttempting to test inference (this may fail if weights aren't available):\n";
  flush stdout;
  try
    test_inference ()
  with e ->
    Printf.printf "Inference test failed: %s\n" (Printexc.to_string e);
    flush stdout
