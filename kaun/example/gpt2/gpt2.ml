open Kaun
open Ptree

(* GPT2 Model *)
let gpt2_model ~config () =
  let vocab_size = config.Config.vocab_size in
  let max_pos = config.Config.n_positions in
  let embd_dim = config.Config.n_embd in
  let num_layers = config.Config.n_layer in
  let embd_dropout = config.Config.embd_pdrop in
  let eps = config.Config.layer_norm_epsilon in
  
  {
    Module.init = (fun ~rngs ~device ~dtype ->
      let _dev = device in (* Currently unused but kept for future use *)
      
      (* Split RNG for all components *)
      let rngs_split = Rune.Rng.split rngs in
      let num_rngs = 3 + num_layers in (* token_emb, pos_emb, ln_f, and blocks *)
      let rngs_array = Array.init num_rngs (fun i -> 
        if i < Array.length rngs_split then rngs_split.(i)
        else (Rune.Rng.split rngs_split.(0)).(i mod Array.length rngs_split)
      ) in
      
      (* Token embeddings *)
      let token_emb = Layer.embedding ~vocab_size ~embed_dim:embd_dim () in
      let token_emb_params = init token_emb ~rngs:rngs_array.(0) ~device ~dtype in
      
      (* Position embeddings *)
      let pos_emb = Layer.embedding ~vocab_size:max_pos ~embed_dim:embd_dim () in
      let pos_emb_params = init pos_emb ~rngs:rngs_array.(1) ~device ~dtype in
      
      (* Transformer blocks *)
      let blocks = Array.init num_layers (fun i ->
        let block = Block.gpt2_block ~config ~layer_idx:i () in
        let block_params = init block ~rngs:rngs_array.(3 + i) ~device ~dtype in
        block_params
      ) in
      
      (* Final layer norm *)
      let ln_f = Layer.layer_norm ~dim:embd_dim ~eps () in
      let ln_f_params = init ln_f ~rngs:rngs_array.(2) ~device ~dtype in
      
      (* Optional: Dropout layer params if needed *)
      
      Ptree.record_of [
        ("token_emb", token_emb_params);
        ("pos_emb", pos_emb_params);
        ("blocks", List (Array.to_list blocks));
        ("ln_f", ln_f_params);
      ]
    );
    
    apply = (fun params ~training ?rngs input_ids ->
      match params with
      | Record fields ->
        let get_params name =
          match Ptree.Record.find_opt name fields with
          | Some p -> p
          | None -> failwith (Printf.sprintf "Missing field %s" name)
        in
        
        let token_emb_params = get_params "token_emb" in
        let pos_emb_params = get_params "pos_emb" in
        let blocks_params = 
          match get_params "blocks" with
          | List blocks -> blocks
          | _ -> failwith "Expected list of blocks"
        in
        let ln_f_params = get_params "ln_f" in
        
        (* Get input shape *)
        let shape = Rune.shape input_ids in
        let batch_size = shape.(0) in
        let seq_len = shape.(1) in
        
        (* Token embeddings *)
        let token_emb = Layer.embedding ~vocab_size ~embed_dim:embd_dim () in
        let input_embeds = apply token_emb token_emb_params ~training:false input_ids in
        
        (* Position IDs - create range from 0 to seq_len *)
        let position_ids = 
          let pos = Rune.arange (Rune.device input_ids) Rune.int32 0 seq_len 1 in
          (* Expand for batch dimension *)
          let pos = Rune.reshape [|1; seq_len|] pos in
          let pos = Rune.broadcast_to [|batch_size; seq_len|] pos in
          (* Cast to float for embedding layer *)
          Rune.cast (Rune.dtype input_embeds) pos
        in
        
        (* Position embeddings *)
        let pos_emb = Layer.embedding ~vocab_size:max_pos ~embed_dim:embd_dim () in
        let position_embeds = apply pos_emb pos_emb_params ~training:false position_ids in
        
        (* Add token and position embeddings *)
        let x = Rune.add input_embeds position_embeds in
        
        (* Apply embedding dropout *)
        let x = 
          if training && embd_dropout > 0.0 then
            match rngs with
            | Some rng ->
              let dropout_layer = Layer.dropout ~rate:embd_dropout () in
              let dropout_params = Ptree.record_of [] in
              apply dropout_layer dropout_params ~training ~rngs:rng x
            | None -> x
          else
            x
        in
        
        (* Apply transformer blocks *)
        let x = List.fold_left2 (fun acc block_params idx ->
          let block = Block.gpt2_block ~config ~layer_idx:idx () in
          apply block block_params ~training ?rngs acc
        ) x blocks_params (List.init num_layers (fun i -> i)) in
        
        (* Final layer norm *)
        let ln_f = Layer.layer_norm ~dim:embd_dim ~eps () in
        let output = apply ln_f ln_f_params ~training ?rngs x in
        
        output
      | _ -> failwith "gpt2_model: invalid params"
    )
  }

(* GPT2 Language Model Head *)
let gpt2_lm_head ~config () =
  let vocab_size = config.Config.vocab_size in
  
  {
    Module.init = (fun ~rngs ~device ~dtype ->
      let rngs_split = Rune.Rng.split rngs in
      let rng1 = rngs_split.(0) in
      let rng2 = rngs_split.(1) in
      
      (* Base model *)
      let model = gpt2_model ~config () in
      let model_params = init model ~rngs:rng1 ~device ~dtype in
      
      (* Language model head - linear projection to vocab *)
      (* Note: In GPT2, this often shares weights with token embeddings *)
      let lm_head = Layer.linear ~in_features:config.Config.n_embd ~out_features:vocab_size () in
      let lm_head_params = init lm_head ~rngs:rng2 ~device ~dtype in
      
      Ptree.record_of [
        ("transformer", model_params);
        ("lm_head", lm_head_params);
      ]
    );
    
    apply = (fun params ~training ?rngs input_ids ->
      match params with
      | Record fields ->
        let get_params name =
          match Ptree.Record.find_opt name fields with
          | Some p -> p
          | None -> failwith (Printf.sprintf "Missing field %s" name)
        in
        
        let transformer_params = get_params "transformer" in
        let lm_head_params = get_params "lm_head" in
        
        (* Run transformer model *)
        let model = gpt2_model ~config () in
        let hidden_states = apply model transformer_params ~training ?rngs input_ids in
        
        (* Apply language model head *)
        let lm_head = Layer.linear ~in_features:config.Config.n_embd ~out_features:vocab_size () in
        let logits = apply lm_head lm_head_params ~training ?rngs hidden_states in
        
        logits
      | _ -> failwith "gpt2_lm_head: invalid params"
    )
  }