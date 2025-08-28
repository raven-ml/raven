open Kaun
open Ptree

(* GPT2 MLP: Linear -> GELU -> Linear with dropout *)
let gpt2_mlp ~config () =
  let embd_dim = config.Config.n_embd in
  let inner_dim =
    match config.Config.n_inner with Some d -> d | None -> 4 * embd_dim
  in
  let resid_dropout = config.Config.resid_pdrop in

  {
    init =
      (fun ~rngs ~device ~dtype ->
        let dev = device in
        let rngs_split = Rune.Rng.split rngs in
        let rng1 = rngs_split.(0) in
        let rng2 = rngs_split.(1) in

        let init = Initializer.glorot_uniform () in

        (* First linear layer (expansion) *)
        let c_fc =
          init.f
            (Rune.Rng.to_int (Rune.Rng.split rng1).(0))
            [| embd_dim; inner_dim |] dev dtype
        in

        (* Second linear layer (projection back) *)
        let c_proj =
          init.f
            (Rune.Rng.to_int (Rune.Rng.split rng2).(0))
            [| inner_dim; embd_dim |] dev dtype
        in

        Ptree.record_of [ ("c_fc", Tensor c_fc); ("c_proj", Tensor c_proj) ]);
    apply =
      (fun params ~training ?rngs x ->
        match params with
        | Record fields ->
            let get_tensor name =
              match Ptree.Record.find_opt name fields with
              | Some (Tensor t) -> t
              | _ ->
                  failwith (Printf.sprintf "Missing or invalid field %s" name)
            in

            let c_fc = get_tensor "c_fc" in
            let c_proj = get_tensor "c_proj" in

            (* First linear transformation *)
            let h = Rune.matmul x c_fc in

            (* GELU activation *)
            let h = Activations.gelu h in

            (* Second linear transformation *)
            let output = Rune.matmul h c_proj in

            (* Apply residual dropout *)
            let output =
              if training && resid_dropout > 0.0 then
                match rngs with
                | Some rng ->
                    let dropout_layer = Layer.dropout ~rate:resid_dropout () in
                    let dropout_params = Ptree.record_of [] in
                    Kaun.apply dropout_layer dropout_params ~training ~rngs:rng
                      output
                | None -> output
              else output
            in

            output
        | _ -> failwith "gpt2_mlp: invalid params");
  }
