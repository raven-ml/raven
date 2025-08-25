open Kaun
open Ptree

(* GPT2 uses slightly different attention than standard multi-head attention *)
let gpt2_self_attention ~config () =
  let max_pos = config.Config.n_positions in
  let embd_dim = config.Config.n_embd in
  let num_heads = config.Config.n_head in
  let head_dim = embd_dim / num_heads in
  let attn_dropout = config.Config.attn_pdrop in
  let resid_dropout = config.Config.resid_pdrop in
  let scale_attn_weights = config.Config.scale_attn_weights in
  
  {
    Module.init = (fun ~rngs ~device ~dtype ->
      let dev = device in
      let rngs_split = Rune.Rng.split rngs in
      let rng1 = rngs_split.(0) in
      let rng2 = rngs_split.(1) in
      
      let init = Initializer.glorot_uniform () in
      
      (* Combined QKV projection *)
      let c_attn = 
        init.f
          (Rune.Rng.to_int (Rune.Rng.split rng1).(0))
          [| embd_dim; 3 * embd_dim |]
          dev dtype
      in
      
      (* Output projection *)
      let c_proj =
        init.f
          (Rune.Rng.to_int (Rune.Rng.split rng2).(0))
          [| embd_dim; embd_dim |]
          dev dtype
      in
      
      (* Create causal mask - lower triangular matrix *)
      let causal_mask = 
        let mask = Rune.ones dev dtype [| max_pos; max_pos |] in
        Rune.tril mask
      in
      
      Ptree.record_of [
        ("c_attn", Tensor c_attn);
        ("c_proj", Tensor c_proj);
        ("causal_mask", Tensor causal_mask);
      ]
    );
    
    apply = (fun params ~training ?rngs x ->
      match params with
      | Record fields ->
        let get_tensor name =
          match Ptree.Record.find_opt name fields with
          | Some (Tensor t) -> t
          | _ -> failwith (Printf.sprintf "Missing or invalid field %s" name)
        in
        
        let c_attn = get_tensor "c_attn" in
        let c_proj = get_tensor "c_proj" in
        let causal_mask = get_tensor "causal_mask" in
        
        (* x shape: [batch, seq_len, embd_dim] *)
        let shape = Rune.shape x in
        let batch_size = shape.(0) in
        let seq_len = shape.(1) in
        
        
        (* Combined QKV projection *)
        let qkv = Rune.matmul x c_attn in
        
        (* Split into Q, K, V *)
        (* Using slice_ranges to extract Q, K, V *)
        let query = Rune.slice_ranges [0; 0; 0] [batch_size; seq_len; embd_dim] qkv in
        let key = Rune.slice_ranges [0; 0; embd_dim] [batch_size; seq_len; 2 * embd_dim] qkv in
        let value = Rune.slice_ranges [0; 0; 2 * embd_dim] [batch_size; seq_len; 3 * embd_dim] qkv in
        
        (* Make contiguous after slicing to ensure reshape works *)
        let query = Rune.contiguous query in
        let key = Rune.contiguous key in
        let value = Rune.contiguous value in
        
        (* Reshape for multi-head attention *)
        (* [batch, seq_len, embd_dim] -> [batch, seq_len, num_heads, head_dim] *)
        let query = Rune.reshape [|batch_size; seq_len; num_heads; head_dim|] query in
        let key = Rune.reshape [|batch_size; seq_len; num_heads; head_dim|] key in
        let value = Rune.reshape [|batch_size; seq_len; num_heads; head_dim|] value in
        
        (* Transpose to [batch, num_heads, seq_len, head_dim] *)
        let query = Rune.transpose query ~axes:[|0; 2; 1; 3|] in
        let key = Rune.transpose key ~axes:[|0; 2; 1; 3|] in
        let value = Rune.transpose value ~axes:[|0; 2; 1; 3|] in
        
        (* Compute attention scores *)
        (* scores = Q @ K^T / sqrt(d_k) *)
        let key_t = Rune.transpose key ~axes:[|0; 1; 3; 2|] in
        let scores = Rune.matmul query key_t in
        
        let scale = 
          if scale_attn_weights then 
            1.0 /. (float_of_int head_dim ** 0.5)
          else 
            1.0
        in
        let scores = Rune.mul scores (Rune.scalar (Rune.device scores) (Rune.dtype scores) scale) in
        
        (* Apply causal mask *)
        let mask_slice = Rune.slice_ranges [0; 0] [seq_len; seq_len] causal_mask in
        (* Expand mask for batch and heads dimensions *)
        let mask_slice = Rune.reshape [|1; 1; seq_len; seq_len|] mask_slice in
        let mask_slice = Rune.broadcast_to [|batch_size; num_heads; seq_len; seq_len|] mask_slice in
        
        (* Apply mask by setting masked positions to large negative value *)
        let masked_scores = 
          let large_neg = Rune.scalar (Rune.device scores) (Rune.dtype scores) (-1e4) in
          let large_neg_broadcast = Rune.broadcast_to (Rune.shape scores) large_neg in
          (* Convert mask to uint8 for where condition *)
          let mask_bool = Rune.cast Rune.uint8 mask_slice in
          (* where(mask, scores, large_neg) - keep scores where mask is 1, else large_neg *)
          Rune.where mask_bool scores large_neg_broadcast
        in
        
        (* Apply softmax *)
        let attn_weights = Rune.softmax ~axes:[|3|] masked_scores in
        
        (* Apply attention dropout during training *)
        let attn_weights = 
          if training && attn_dropout > 0.0 then
            match rngs with
            | Some rng ->
              let dropout_layer = Layer.dropout ~rate:attn_dropout () in
              let dropout_params = Ptree.record_of [] in (* Dropout has no params *)
              Kaun.apply dropout_layer dropout_params ~training ~rngs:rng attn_weights
            | None -> attn_weights
          else 
            attn_weights
        in
        
        (* Compute attention output *)
        let attn_output = Rune.matmul attn_weights value in
        
        (* Transpose back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim] *)
        let attn_output = Rune.transpose attn_output ~axes:[|0; 2; 1; 3|] in
        
        
        (* Use copy instead of contiguous as a workaround *)
        let attn_output = Rune.copy attn_output in
        
        (* Now reshape should work *)
        let attn_output = Rune.reshape [|batch_size; seq_len; embd_dim|] attn_output in
        
        (* Output projection *)
        let output = Rune.matmul attn_output c_proj in
        
        (* Apply residual dropout *)
        let output = 
          if training && resid_dropout > 0.0 then
            match rngs with
            | Some rng ->
              let dropout_layer = Layer.dropout ~rate:resid_dropout () in
              let dropout_params = Ptree.record_of [] in
              Kaun.apply dropout_layer dropout_params ~training ~rngs:rng output
            | None -> output
          else
            output
        in
        
        output
      | _ -> failwith "gpt2_self_attention: invalid params"
    )
  }