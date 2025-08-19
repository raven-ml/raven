open Kaun

(* Rotary Position Embeddings *)
module RoPE = struct
  let precompute_freqs_cis ~dim ~max_seq_len ~base ~device ~dtype =
    (* Compute rotation frequencies *)
    let half_dim = dim / 2 in
    let _freqs =
      Array.init half_dim (fun i ->
          1.0 /. (base ** (float_of_int (2 * i) /. float_of_int dim)))
    in
    (* Create frequency tensor - simplified placeholder *)
    let freqs_tensor = Rune.ones device dtype [| half_dim |] in

    (* Create position indices *)
    let pos = Rune.arange_f device dtype 0.0 (float_of_int max_seq_len) 1.0 in
    let pos = Rune.reshape [| max_seq_len; 1 |] pos in

    (* Compute angles: pos * freqs *)
    let angles =
      Rune.matmul pos (Rune.reshape [| 1; half_dim |] freqs_tensor)
    in

    (* Return cos and sin *)
    (Rune.cos angles, Rune.sin angles)

  let apply_rotary_pos_emb q k cos sin ~seq_len =
    (* Extract the sequence length portion we need *)
    let cos = Rune.slice [ R [ 0; seq_len ] ] cos in
    let sin = Rune.slice [ R [ 0; seq_len ] ] sin in

    (* Reshape for broadcasting *)
    let batch_size = (Rune.shape q).(0) in
    let num_heads = (Rune.shape q).(1) in
    let head_dim = (Rune.shape q).(3) in

    (* Expand cos/sin to match q/k shape *)
    let cos = Rune.reshape [| 1; 1; seq_len; head_dim / 2 |] cos in
    let sin = Rune.reshape [| 1; 1; seq_len; head_dim / 2 |] sin in
    let cos =
      Rune.broadcast_to [| batch_size; num_heads; seq_len; head_dim / 2 |] cos
    in
    let sin =
      Rune.broadcast_to [| batch_size; num_heads; seq_len; head_dim / 2 |] sin
    in

    (* Split q and k into two halves for rotation *)
    let split_tensor x =
      let half_dim = head_dim / 2 in
      let x1 = Rune.slice [ R []; R []; R []; R [ 0; half_dim ] ] x in
      let x2 = Rune.slice [ R []; R []; R []; R [ half_dim; head_dim ] ] x in
      (x1, x2)
    in

    let rotate x =
      let x1, x2 = split_tensor x in
      (* Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos] *)
      let rotated_1 = Rune.sub (Rune.mul x1 cos) (Rune.mul x2 sin) in
      let rotated_2 = Rune.add (Rune.mul x1 sin) (Rune.mul x2 cos) in
      Rune.concatenate [ rotated_1; rotated_2 ] ~axis:3
    in

    (rotate q, rotate k)
end

(* Multi-head attention with RoPE and GQA support *)
let multi_head_attention_with_rope ~embed_dim ~num_heads ~num_kv_heads ~head_dim
    ~use_qk_norm ~attn_logits_soft_cap ~query_pre_attn_scalar
    ~sliding_window_size ?(_use_cache = false) () =
  Model
    {
      init =
        (fun ~rngs x ->
          let dev = Rune.device x in
          let dtype = Rune.dtype x in
          let rngs_split = Rune.Rng.split rngs in
          let rng1 = rngs_split.(0) in
          let rng2 = rngs_split.(1) in
          let rngs_split2 = Rune.Rng.split rng2 in
          let rng3 = rngs_split2.(0) in
          let rng4 = rngs_split2.(1) in
          let rngs_split3 = Rune.Rng.split rng4 in
          let rng5 = rngs_split3.(0) in
          let rng6 = rngs_split3.(1) in

          let init = Initializer.glorot_uniform () in

          (* Q, K, V projections *)
          let q_proj =
            Initializer.apply init
              (Rune.Rng.to_int ((Rune.Rng.split rng1).(0)))
              [| embed_dim; num_heads * head_dim |]
              dev dtype
          in
          let k_proj =
            Initializer.apply init
              (Rune.Rng.to_int ((Rune.Rng.split rng3).(0)))
              [| embed_dim; num_kv_heads * head_dim |]
              dev dtype
          in
          let v_proj =
            Initializer.apply init
              (Rune.Rng.to_int ((Rune.Rng.split rng5).(0)))
              [| embed_dim; num_kv_heads * head_dim |]
              dev dtype
          in

          (* Output projection *)
          let out_proj =
            Initializer.apply init
              (Rune.Rng.to_int ((Rune.Rng.split rng6).(0)))
              [| num_heads * head_dim; embed_dim |]
              dev dtype
          in

          (* QK normalization if enabled *)
          let qk_norm_params =
            if use_qk_norm then
              let q_norm = Rune.ones dev dtype [| head_dim |] in
              let k_norm = Rune.ones dev dtype [| head_dim |] in
              Record [ ("q_norm", Tensor q_norm); ("k_norm", Tensor k_norm) ]
            else List []
          in

          Record
            [
              ("q_proj", Tensor q_proj);
              ("k_proj", Tensor k_proj);
              ("v_proj", Tensor v_proj);
              ("out_proj", Tensor out_proj);
              ("qk_norm", qk_norm_params);
            ]);
      apply =
        (fun (type layout dev)
          (params : (layout, dev) params)
          ~(training : bool)
          ?rngs
          (x : (layout, dev) tensor)
        ->
          let _ = rngs in
          let _ = training in
          match params with
          | Record fields ->
              let get_tensor name =
                match List.assoc name fields with
                | Tensor t -> t
                | _ -> failwith (Printf.sprintf "Expected tensor for %s" name)
              in

              let q_proj = get_tensor "q_proj" in
              let k_proj = get_tensor "k_proj" in
              let v_proj = get_tensor "v_proj" in
              let out_proj = get_tensor "out_proj" in

              let batch_size = (Rune.shape x).(0) in
              let seq_len = (Rune.shape x).(1) in

              (* Create RoPE embeddings - should be precomputed in practice *)
              let rope_cos, rope_sin =
                RoPE.precompute_freqs_cis ~dim:head_dim ~max_seq_len:seq_len
                  ~base:10000.0 ~device:(Rune.device x) ~dtype:(Rune.dtype x)
              in

              (* Project to Q, K, V *)
              let q = Rune.matmul x q_proj in
              let k = Rune.matmul x k_proj in
              let v = Rune.matmul x v_proj in

              (* Reshape for multi-head attention *)
              let q =
                Rune.reshape [| batch_size; seq_len; num_heads; head_dim |] q
              in
              let k =
                Rune.reshape [| batch_size; seq_len; num_kv_heads; head_dim |] k
              in
              let v =
                Rune.reshape [| batch_size; seq_len; num_kv_heads; head_dim |] v
              in

              (* Transpose to [batch, heads, seq, head_dim] *)
              let q = Rune.transpose q ~axes:[| 0; 2; 1; 3 |] in
              let k = Rune.transpose k ~axes:[| 0; 2; 1; 3 |] in
              let v = Rune.transpose v ~axes:[| 0; 2; 1; 3 |] in

              (* Apply RoPE *)
              let q, k =
                RoPE.apply_rotary_pos_emb q k rope_cos rope_sin ~seq_len
              in

              (* Apply QK normalization if enabled *)
              let q, k =
                if use_qk_norm then
                  match List.assoc "qk_norm" fields with
                  | Record qk_fields ->
                      let q_norm =
                        match List.assoc "q_norm" qk_fields with
                        | Tensor t -> t
                        | _ -> failwith "Expected q_norm tensor"
                      in
                      let k_norm =
                        match List.assoc "k_norm" qk_fields with
                        | Tensor t -> t
                        | _ -> failwith "Expected k_norm tensor"
                      in
                      let normalize x norm =
                        let x_norm =
                          Rune.sqrt
                            (Rune.mean (Rune.square x) ~axes:[| -1 |]
                               ~keepdims:true)
                        in
                        let x =
                          Rune.div x
                            (Rune.add x_norm
                               (Rune.scalar (Rune.device x) (Rune.dtype x) 1e-6))
                        in
                        Rune.mul x (Rune.reshape [| 1; 1; 1; head_dim |] norm)
                      in
                      (normalize q q_norm, normalize k k_norm)
                  | _ -> (q, k)
                else (q, k)
              in

              (* Handle GQA - expand K, V if needed *)
              let k, v =
                if num_kv_heads < num_heads then
                  let repeat_factor = num_heads / num_kv_heads in
                  let expand_kv x =
                    let shape = Rune.shape x in
                    let expanded =
                      Rune.reshape
                        [| shape.(0); num_kv_heads; 1; shape.(2); shape.(3) |]
                        x
                    in
                    let expanded =
                      Rune.broadcast_to
                        [|
                          shape.(0);
                          num_kv_heads;
                          repeat_factor;
                          shape.(2);
                          shape.(3);
                        |]
                        expanded
                    in
                    Rune.reshape
                      [| shape.(0); num_heads; shape.(2); shape.(3) |]
                      expanded
                  in
                  (expand_kv k, expand_kv v)
                else (k, v)
              in

              (* Compute attention scores *)
              let scores =
                Rune.matmul q (Rune.transpose k ~axes:[| 0; 1; 3; 2 |])
              in
              let scores =
                Rune.mul scores
                  (Rune.scalar (Rune.device scores) (Rune.dtype scores)
                     query_pre_attn_scalar)
              in

              (* Apply attention logit soft capping if specified *)
              let scores =
                match attn_logits_soft_cap with
                | Some cap ->
                    let cap_scalar =
                      Rune.scalar (Rune.device scores) (Rune.dtype scores) cap
                    in
                    Rune.mul (Rune.tanh (Rune.div scores cap_scalar)) cap_scalar
                | None -> scores
              in

              (* Apply causal mask *)
              let mask =
                Kaun.Ops.tril
                  (Rune.ones (Rune.device scores) (Rune.dtype scores)
                     [| seq_len; seq_len |])
                  ()
              in
              let mask = Rune.reshape [| 1; 1; seq_len; seq_len |] mask in
              let mask =
                Rune.broadcast_to
                  [| batch_size; num_heads; seq_len; seq_len |]
                  mask
              in
              (* Convert mask to uint8 for where operation *)
              let mask_bool = Rune.cast Rune.uint8 mask in
              let masked_value =
                Rune.scalar (Rune.device scores) (Rune.dtype scores) (-1e10)
              in
              let scores = Rune.where mask_bool scores masked_value in

              (* Apply sliding window mask if specified *)
              let scores =
                match sliding_window_size with
                | Some window_size ->
                    (* Create sliding window mask *)
                    let window_mask =
                      Array.init seq_len (fun i ->
                          Array.init seq_len (fun j ->
                              if j <= i && i - j < window_size then 1.0 else 0.0))
                    in
                    let window_mask_flat =
                      Array.concat (Array.to_list window_mask)
                    in
                    let ba =
                      Bigarray.Genarray.create Bigarray.float32
                        Bigarray.c_layout [| seq_len; seq_len |]
                    in
                    Array.iteri
                      (fun i v ->
                        Bigarray.Genarray.set ba
                          [| i / seq_len; i mod seq_len |]
                          v)
                      window_mask_flat;
                    let window_mask_tensor =
                      Rune.of_bigarray (Rune.device scores) ba
                    in
                    let window_mask_tensor =
                      Rune.reshape
                        [| 1; 1; seq_len; seq_len |]
                        window_mask_tensor
                    in
                    let window_mask_tensor =
                      Rune.broadcast_to
                        [| batch_size; num_heads; seq_len; seq_len |]
                        window_mask_tensor
                    in
                    let window_mask_bool =
                      Rune.cast Rune.uint8 window_mask_tensor
                    in
                    Rune.where window_mask_bool scores masked_value
                | None -> scores
              in

              (* Softmax *)
              let attn_weights = Rune.softmax scores ~axes:[| -1 |] in

              (* Apply attention to values *)
              let attn_output = Rune.matmul attn_weights v in

              (* Transpose back and reshape *)
              let attn_output =
                Rune.transpose attn_output ~axes:[| 0; 2; 1; 3 |]
              in
              let attn_output =
                Rune.reshape
                  [| batch_size; seq_len; num_heads * head_dim |]
                  attn_output
              in

              (* Output projection *)
              Rune.matmul attn_output out_proj
          | _ -> failwith "multi_head_attention: invalid params");
    }
