open Kaun
open Ptree

(* GPT2 Transformer Block: LayerNorm -> Attention -> Residual -> LayerNorm ->
   MLP -> Residual *)
let gpt2_block ~config ~layer_idx:_ () =
  let eps = config.Config.layer_norm_epsilon in
  let embd_dim = config.Config.n_embd in

  {
    init =
      (fun ~rngs ~device ~dtype ->
        let rngs_split = Rune.Rng.split rngs in
        let rng1 = rngs_split.(0) in
        let rng2 = rngs_split.(1) in
        let rngs_split2 = Rune.Rng.split rng2 in
        let rng3 = rngs_split2.(0) in
        let rng4 = rngs_split2.(1) in

        (* Pre-attention layer norm *)
        let ln_1 = Layer.layer_norm ~dim:embd_dim ~eps () in
        let ln_1_params = init ln_1 ~rngs:rng1 ~device ~dtype in

        (* Self-attention *)
        let attn = Attention.gpt2_self_attention ~config () in
        let attn_params = init attn ~rngs:rng3 ~device ~dtype in

        (* Pre-MLP layer norm *)
        let ln_2 = Layer.layer_norm ~dim:embd_dim ~eps () in
        let ln_2_params = init ln_2 ~rngs:rng2 ~device ~dtype in

        (* MLP *)
        let mlp = Mlp.gpt2_mlp ~config () in
        let mlp_params = init mlp ~rngs:rng4 ~device ~dtype in

        Ptree.record_of
          [
            ("ln_1", ln_1_params);
            ("attn", attn_params);
            ("ln_2", ln_2_params);
            ("mlp", mlp_params);
          ]);
    apply =
      (fun params ~training ?rngs x ->
        match params with
        | Record fields ->
            let get_params name =
              match Ptree.Record.find_opt name fields with
              | Some p -> p
              | None -> failwith (Printf.sprintf "Missing field %s" name)
            in

            let ln_1_params = get_params "ln_1" in
            let attn_params = get_params "attn" in
            let ln_2_params = get_params "ln_2" in
            let mlp_params = get_params "mlp" in

            (* Pre-attention layer norm and attention with residual *)
            let residual = x in
            let x_norm =
              let ln_1 = Layer.layer_norm ~dim:embd_dim ~eps () in
              apply ln_1 ln_1_params ~training ?rngs x
            in
            let attn_out =
              let attn = Attention.gpt2_self_attention ~config () in
              apply attn attn_params ~training ?rngs x_norm
            in
            let x = Rune.add residual attn_out in

            (* Pre-MLP layer norm and MLP with residual *)
            let residual = x in
            let x_norm =
              let ln_2 = Layer.layer_norm ~dim:embd_dim ~eps () in
              apply ln_2 ln_2_params ~training ?rngs x
            in
            let mlp_out =
              let mlp = Mlp.gpt2_mlp ~config () in
              apply mlp mlp_params ~training ?rngs x_norm
            in
            let output = Rune.add residual mlp_out in

            output
        | _ -> failwith "gpt2_block: invalid params");
  }
