type 'layout tensor = (float, 'layout) Rune.t

module Dtype = Nx_core.Dtype

type module_ = {
  init :
    'layout. rngs:Rune.Rng.key -> dtype:(float, 'layout) Rune.dtype -> Ptree.t;
  apply :
    'layout.
    Ptree.t ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    'layout tensor ->
    'layout tensor;
}

let list_items_exn context = function
  | Ptree.List items -> items
  | _ -> failwith (context ^ ": invalid params structure")

let relu () =
  {
    init = (fun ~rngs:_ ~dtype:_ -> Ptree.list []);
    apply = (fun _params ~training:_ ?rngs:_ x -> Rune.relu x);
  }

let sigmoid () =
  {
    init = (fun ~rngs:_ ~dtype:_ -> Ptree.list []);
    apply = (fun _params ~training:_ ?rngs:_ x -> Rune.sigmoid x);
  }

let tanh () =
  {
    init = (fun ~rngs:_ ~dtype:_ -> Ptree.list []);
    apply = (fun _params ~training:_ ?rngs:_ x -> Rune.tanh x);
  }

let gelu () =
  {
    init = (fun ~rngs:_ ~dtype:_ -> Ptree.list []);
    apply = (fun _params ~training:_ ?rngs:_ x -> Activations.gelu x);
  }

let swish () =
  {
    init = (fun ~rngs:_ ~dtype:_ -> Ptree.list []);
    apply = (fun _params ~training:_ ?rngs:_ x -> Activations.swish x);
  }

let conv1d ~in_channels ~out_channels ?(kernel_size = 3) ?(stride = 1)
    ?(dilation = 1) ?(padding = `Same) () =
  {
    init =
      (fun (type l) ~rngs ~(dtype : (float, l) Rune.dtype) ->
        Rune.debug_with_context
          (Printf.sprintf "conv1d_%dx%d_%d_init" in_channels out_channels
             kernel_size) (fun () ->
            let rngs_split = Rune.Rng.split rngs in
            let rng1 = rngs_split.(0) in
            let fan_in = in_channels * kernel_size in
            let fan_out = out_channels * kernel_size in
            let limit = sqrt (6.0 /. float_of_int (fan_in + fan_out)) in
            let weight_shape = [| out_channels; in_channels; kernel_size |] in
            let w = Rune.Rng.uniform rng1 dtype weight_shape in
            let w =
              Rune.sub
                (Rune.mul w (Rune.scalar dtype (2.0 *. limit)))
                (Rune.scalar dtype limit)
            in
            let b = Rune.zeros dtype [| out_channels |] in
            Ptree.dict [ ("weight", Ptree.tensor w); ("bias", Ptree.tensor b) ]));
    apply =
      (fun (type l) (params : Ptree.t) ~training:_ ?rngs:_ (x : l tensor) ->
        let fields = Ptree.Dict.fields_exn ~ctx:"conv1d" params in
        let dtype = Rune.dtype x in
        let weight = Ptree.Dict.get_tensor_exn fields ~name:"weight" dtype in
        let bias = Ptree.Dict.get_tensor_exn fields ~name:"bias" dtype in
        Rune.debug_with_context
          (Printf.sprintf "conv1d_%dx%d_%d" in_channels out_channels kernel_size)
          (fun () ->
            let x =
              match padding with
              | `Same -> x
              | `Valid -> x
              | `Causal ->
                  let pad_left = (kernel_size - 1) * dilation in
                  let pad_cfg = [| (0, 0); (0, 0); (pad_left, 0) |] in
                  Rune.pad pad_cfg 0.0 x
            in
            let padding_mode =
              match padding with
              | `Same -> `Same
              | `Valid -> `Valid
              | `Causal -> `Valid
            in
            let conv =
              Rune.convolve1d x weight ~stride ~dilation ~padding_mode
            in
            let b_reshaped = Rune.reshape [| 1; out_channels; 1 |] bias in
            Rune.add conv b_reshaped));
  }

let conv2d ~in_channels ~out_channels ?(kernel_size = (3, 3)) () =
  let kh, kw = kernel_size in
  {
    init =
      (fun (type l) ~rngs ~(dtype : (float, l) Rune.dtype) ->
        Rune.debug_with_context
          (Printf.sprintf "conv2d_%dx%d_%dx%d_init" in_channels out_channels kh
             kw) (fun () ->
            let rngs_split = Rune.Rng.split rngs in
            let rng1 = rngs_split.(0) in
            let fan_in = in_channels * kh * kw in
            let fan_out = out_channels * kh * kw in
            let limit = sqrt (6.0 /. float_of_int (fan_in + fan_out)) in
            let weight_shape = [| out_channels; in_channels; kh; kw |] in
            let w = Rune.Rng.uniform rng1 dtype weight_shape in
            let w =
              Rune.sub
                (Rune.mul w (Rune.scalar dtype (2.0 *. limit)))
                (Rune.scalar dtype limit)
            in
            let b = Rune.zeros dtype [| out_channels |] in
            Ptree.dict [ ("weight", Ptree.tensor w); ("bias", Ptree.tensor b) ]));
    apply =
      (fun (type l) (params : Ptree.t) ~training:_ ?rngs:_ (x : l tensor) ->
        let fields = Ptree.Dict.fields_exn ~ctx:"conv2d" params in
        let dtype = Rune.dtype x in
        let weight = Ptree.Dict.get_tensor_exn fields ~name:"weight" dtype in
        let bias = Ptree.Dict.get_tensor_exn fields ~name:"bias" dtype in
        Rune.debug_with_context
          (Printf.sprintf "conv2d_%dx%d_%dx%d" in_channels out_channels kh kw)
          (fun () ->
            let conv =
              Rune.convolve2d x weight ~stride:(1, 1) ~padding_mode:`Same
            in
            let b_reshaped = Rune.reshape [| 1; out_channels; 1; 1 |] bias in
            Rune.add conv b_reshaped));
  }

let linear ~in_features ~out_features ?weight_init ?bias_init () =
  {
    init =
      (fun ~rngs ~dtype ->
        Rune.debug_with_context
          (Printf.sprintf "linear_%dx%d_init" in_features out_features)
          (fun () ->
            let weight_init_f =
              match weight_init with
              | Some init -> init.Initializers.f
              | None -> (Initializers.glorot_uniform ()).f
            in
            let bias_init_f =
              match bias_init with
              | Some init -> init.Initializers.f
              | None -> (Initializers.zeros ()).f
            in
            let rngs_split = Rune.Rng.split rngs in
            let rng1 = rngs_split.(0) in
            let rng2 = rngs_split.(1) in
            let w =
              weight_init_f (Rune.Rng.to_int rng1)
                [| in_features; out_features |]
                dtype
            in
            let b =
              bias_init_f (Rune.Rng.to_int rng2) [| out_features |] dtype
            in
            Ptree.dict [ ("weight", Ptree.tensor w); ("bias", Ptree.tensor b) ]));
    apply =
      (fun (type l) (params : Ptree.t) ~training:_ ?rngs:_ (x : l tensor) ->
        Rune.debug_with_context
          (Printf.sprintf "linear_%dx%d" in_features out_features) (fun () ->
            let fields = Ptree.Dict.fields_exn ~ctx:"linear" params in
            let dtype = Rune.dtype x in
            let weight =
              Ptree.Dict.get_tensor_exn fields ~name:"weight" dtype
            in
            let bias = Ptree.Dict.get_tensor_exn fields ~name:"bias" dtype in
            let z = Rune.matmul x weight in
            Rune.add z bias));
  }

let dropout ~rate () =
  {
    init = (fun ~rngs:_ ~dtype:_ -> Ptree.list []);
    apply =
      (fun _params ~training ?rngs x ->
        if (not training) || rate = 0.0 then x
        else
          match rngs with
          | Some rng ->
              let seed = Rune.Rng.to_int rng in
              Rune.dropout ~seed ~rate x
          | None -> failwith "dropout requires RNG if rate > 0.0");
  }

(* alias for internal use *)
let dropout_layer = dropout

let batch_norm ~num_features () =
  {
    init =
      (fun ~rngs ~dtype ->
        Rune.debug_with_context
          (Printf.sprintf "batch_norm_%d_init" num_features) (fun () ->
            let _rngs_split = Rune.Rng.split rngs in
            let scale = Rune.ones dtype [| num_features |] in
            let bias = Rune.zeros dtype [| num_features |] in
            Ptree.dict
              [ ("scale", Ptree.tensor scale); ("bias", Ptree.tensor bias) ]));
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        let fields = Ptree.Dict.fields_exn ~ctx:"batch_norm" params in
        let dtype = Rune.dtype x in
        let scale = Ptree.Dict.get_tensor_exn fields ~name:"scale" dtype in
        let bias = Ptree.Dict.get_tensor_exn fields ~name:"bias" dtype in
        Rune.debug_with_context
          (Printf.sprintf "batch_norm_%d_apply" num_features) (fun () ->
            Rune.batch_norm ~scale ~bias x));
  }

let max_pool2d ~kernel_size ?stride () =
  let stride = match stride with Some s -> s | None -> kernel_size in
  {
    init = (fun ~rngs:_ ~dtype:_ -> Ptree.list []);
    apply =
      (fun _params ~training:_ ?rngs:_ x ->
        let pooled, _ = Rune.max_pool2d x ~kernel_size ~stride in
        pooled);
  }

let avg_pool2d ~kernel_size ?stride () =
  let stride = match stride with Some s -> s | None -> kernel_size in
  {
    init = (fun ~rngs:_ ~dtype:_ -> Ptree.list []);
    apply =
      (fun _params ~training:_ ?rngs:_ x ->
        Rune.avg_pool2d x ~kernel_size ~stride);
  }

let flatten () =
  {
    init = (fun ~rngs:_ ~dtype:_ -> Ptree.list []);
    apply =
      (fun _params ~training:_ ?rngs:_ x ->
        let shape = Rune.shape x in
        let batch_size = shape.(0) in
        let flat_size =
          Array.fold_left ( * ) 1 (Array.sub shape 1 (Array.length shape - 1))
        in
        let x = if Rune.is_c_contiguous x then x else Rune.contiguous x in
        Rune.reshape [| batch_size; flat_size |] x);
  }

let sequential models =
  {
    init =
      (fun ~rngs ~dtype ->
        let rec init_layers models acc rngs_current =
          match models with
          | [] -> Ptree.List (List.rev acc)
          | m :: rest ->
              let rngs_split = Rune.Rng.split rngs_current in
              let rngs_layer = rngs_split.(0) in
              let rngs_rest = rngs_split.(1) in
              let params = m.init ~rngs:rngs_layer ~dtype in
              init_layers rest (params :: acc) rngs_rest
        in
        init_layers models [] rngs);
    apply =
      (fun params ~training ?rngs:_ x ->
        let param_list = list_items_exn "sequential" params in
        let rec apply_layers models params x layer_idx =
          match (models, params) with
          | [], [] -> x
          | m :: ms, p :: ps ->
              let x' = m.apply p ~training x in
              apply_layers ms ps x' (layer_idx + 1)
          | _ -> failwith "sequential: mismatched models and params"
        in
        apply_layers models param_list x 1);
  }

let einsum ~einsum_str ~shape ?kernel_init () =
  {
    init =
      (fun ~rngs ~dtype ->
        let kernel_init_f =
          match kernel_init with
          | Some init -> init.Initializers.f
          | None -> (Initializers.glorot_uniform ()).f
        in
        let key = (Rune.Rng.split rngs).(0) in
        let w = kernel_init_f (Rune.Rng.to_int key) shape dtype in
        Ptree.dict [ ("weight", Ptree.tensor w) ]);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        let fields = Ptree.Dict.fields_exn ~ctx:"einsum" params in
        let dtype = Rune.dtype x in
        let w = Ptree.Dict.get_tensor_exn fields ~name:"weight" dtype in
        Rune.einsum einsum_str [| x; w |]);
  }

let rms_norm ~dim ?(eps = 1e-6) ?scale_init () =
  {
    init =
      (fun ~rngs ~dtype ->
        let scale_init_f =
          match scale_init with
          | Some init -> init.Initializers.f
          | None -> (Initializers.ones ()).f
        in
        let key = (Rune.Rng.split rngs).(0) in
        let scale = scale_init_f (Rune.Rng.to_int key) [| dim |] dtype in
        Ptree.dict [ ("scale", Ptree.tensor scale) ]);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        let fields = Ptree.Dict.fields_exn ~ctx:"rms_norm" params in
        let dtype = Rune.dtype x in
        let scale = Ptree.Dict.get_tensor_exn fields ~name:"scale" dtype in
        Rune.rms_norm ~gamma:scale ~epsilon:eps x);
  }

let layer_norm ~dim ?(eps = 1e-5) ?(elementwise_affine = true) () =
  {
    init =
      (fun ~rngs:_ ~dtype ->
        if elementwise_affine then
          let gamma = Rune.ones dtype [| dim |] in
          let beta = Rune.zeros dtype [| dim |] in
          Ptree.dict
            [ ("gamma", Ptree.tensor gamma); ("beta", Ptree.tensor beta) ]
        else Ptree.list []);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        if elementwise_affine then
          let fields = Ptree.Dict.fields_exn ~ctx:"layer_norm" params in
          let dtype = Rune.dtype x in
          let gamma = Ptree.Dict.get_tensor_exn fields ~name:"gamma" dtype in
          let beta = Ptree.Dict.get_tensor_exn fields ~name:"beta" dtype in
          Rune.layer_norm ~gamma ~beta ~epsilon:eps x
        else Rune.layer_norm ~epsilon:eps x);
  }

let embedding ~vocab_size ~embed_dim ?(scale = true) ?embedding_init () =
  {
    init =
      (fun ~rngs ~dtype ->
        let embedding_init_f =
          match embedding_init with
          | Some init -> init.Initializers.f
          | None -> (Initializers.normal_range ~mean:0.0 ~stddev:0.02 ()).f
        in
        let key = (Rune.Rng.split rngs).(0) in
        let embedding =
          embedding_init_f (Rune.Rng.to_int key)
            [| vocab_size; embed_dim |]
            dtype
        in
        Ptree.dict [ ("embedding", Ptree.tensor embedding) ]);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        let fields = Ptree.Dict.fields_exn ~ctx:"embedding" params in
        let dtype = Rune.dtype x in
        let embedding =
          Ptree.Dict.get_tensor_exn fields ~name:"embedding" dtype
        in
        let indices = Rune.cast Rune.int32 x in
        Rune.embedding ~scale ~embedding indices);
  }


let mlp ~in_features ~hidden_features ~out_features ?(activation = `gelu)
    ?(dropout = 0.0) () =
  let act =
    match activation with
    | `relu -> relu ()
    | `gelu -> gelu ()
    | `swish -> swish ()
  in
  sequential
    [
      linear ~in_features ~out_features:hidden_features ();
      act;
      dropout_layer ~rate:dropout ();
      linear ~in_features:hidden_features ~out_features ();
      dropout_layer ~rate:dropout ();
    ]

let rnn ~input_size ~hidden_size ?(return_sequences = false)
    ?(learned_init = false) () =
  {
    init =
      (fun ~rngs ~dtype ->
        let glorot = (Initializers.glorot_uniform ()).f in
        let keys = Rune.Rng.split ~n:2 rngs in
        let w_xh =
          glorot (Rune.Rng.to_int keys.(0)) [| input_size; hidden_size |] dtype
        in
        let w_hh =
          glorot (Rune.Rng.to_int keys.(1)) [| hidden_size; hidden_size |] dtype
        in
        let b = Rune.zeros dtype [| hidden_size |] in
        let base =
          [
            ("w_xh", Ptree.tensor w_xh);
            ("w_hh", Ptree.tensor w_hh);
            ("b", Ptree.tensor b);
          ]
        in
        let base =
          if learned_init then
            let h0 = Rune.zeros dtype [| hidden_size |] in
            base @ [ ("h0", Ptree.tensor h0) ]
          else base
        in
        Ptree.dict base);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Ptree.Dict fields ->
            let dt = Rune.dtype x in
            let param_exn name = Ptree.Dict.get_tensor_exn fields ~name dt in
            let param_opt name = Ptree.Dict.get_tensor fields ~name dt in
            let w_xh = param_exn "w_xh"
            and w_hh = param_exn "w_hh"
            and b = param_exn "b" in
            let batch, seq_len, _ =
              match Rune.shape x with
              | [| b; s; i |] -> (b, s, i)
              | _ -> failwith "rnn: expected [b; s; i]"
            in
            let h_init =
              match param_opt "h0" with
              | Some h0 ->
                  Rune.reshape [| 1; hidden_size |] h0
                  |> Rune.expand [| batch; hidden_size |]
              | None -> Rune.zeros dt [| batch; hidden_size |]
            in
            let h = ref h_init in
            let outputs =
              Array.make seq_len (Rune.zeros dt [| batch; hidden_size |])
            in
            for t = 0 to seq_len - 1 do
              let xt = Rune.slice [ Rune.A; Rune.I t; Rune.A ] x in
              let pre =
                Rune.add (Rune.matmul xt w_xh)
                  (Rune.add (Rune.matmul !h w_hh)
                     (Rune.reshape [| 1; hidden_size |] b))
              in
              h := Rune.tanh pre
            done;
            if return_sequences then (
              (* Fill outputs in second loop to keep simple shape reuse *)
              let h2 = ref h_init in
              for t = 0 to seq_len - 1 do
                let xt = Rune.slice [ Rune.A; Rune.I t; Rune.A ] x in
                let pre =
                  Rune.add (Rune.matmul xt w_xh)
                    (Rune.add (Rune.matmul !h2 w_hh)
                       (Rune.reshape [| 1; hidden_size |] b))
                in
                h2 := Rune.tanh pre;
                outputs.(t) <- !h2
              done;
              Rune.stack ~axis:1 (Array.to_list outputs))
            else !h
        | _ -> failwith "rnn: invalid params");
  }

let gru ~input_size ~hidden_size ?(return_sequences = false)
    ?(learned_init = false) () =
  {
    init =
      (fun ~rngs ~dtype ->
        let glorot = (Initializers.glorot_uniform ()).f in
        let keys = Rune.Rng.split ~n:2 rngs in
        let w_ih =
          glorot
            (Rune.Rng.to_int keys.(0))
            [| input_size; hidden_size * 3 |]
            dtype
        in
        let w_hh =
          glorot
            (Rune.Rng.to_int keys.(1))
            [| hidden_size; hidden_size * 3 |]
            dtype
        in
        let b = Rune.zeros dtype [| hidden_size * 3 |] in
        let base =
          [
            ("w_ih", Ptree.tensor w_ih);
            ("w_hh", Ptree.tensor w_hh);
            ("b", Ptree.tensor b);
          ]
        in
        let base =
          if learned_init then
            let h0 = Rune.zeros dtype [| hidden_size |] in
            base @ [ ("h0", Ptree.tensor h0) ]
          else base
        in
        Ptree.dict base);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Ptree.Dict fields ->
            let dt = Rune.dtype x in
            let param_exn name = Ptree.Dict.get_tensor_exn fields ~name dt in
            let param_opt name = Ptree.Dict.get_tensor fields ~name dt in
            let w_ih = param_exn "w_ih"
            and w_hh = param_exn "w_hh"
            and b = param_exn "b" in
            let batch, seq_len, _ =
              match Rune.shape x with
              | [| b; s; i |] -> (b, s, i)
              | _ -> failwith "gru: expected [b; s; i]"
            in
            let h_init =
              match param_opt "h0" with
              | Some h0 ->
                  Rune.reshape [| 1; hidden_size |] h0
                  |> Rune.expand [| batch; hidden_size |]
              | None -> Rune.zeros dt [| batch; hidden_size |]
            in
            let h = ref h_init in
            let outputs =
              Array.make seq_len (Rune.zeros dt [| batch; hidden_size |])
            in
            for t = 0 to seq_len - 1 do
              let xt = Rune.slice [ Rune.A; Rune.I t; Rune.A ] x in
              let gates =
                Rune.add (Rune.matmul xt w_ih)
                  (Rune.add (Rune.matmul !h w_hh)
                     (Rune.reshape [| 1; hidden_size * 3 |] b))
              in
              let z =
                Rune.sigmoid
                  (Rune.slice [ Rune.A; Rune.R (0, hidden_size) ] gates)
              in
              let r =
                Rune.sigmoid
                  (Rune.slice
                     [ Rune.A; Rune.R (hidden_size, hidden_size * 2) ]
                     gates)
              in
              let n =
                Rune.tanh
                  (Rune.add
                     (Rune.slice
                        [ Rune.A; Rune.R (hidden_size * 2, hidden_size * 3) ]
                        gates)
                     (Rune.matmul (Rune.mul r !h)
                        (Rune.slice [ Rune.A; Rune.R (0, hidden_size) ] w_hh)))
              in
              h :=
                Rune.add
                  (Rune.mul (Rune.sub (Rune.scalar dt 1.0) z) n)
                  (Rune.mul z !h)
            done;
            if return_sequences then (
              let h2 = ref h_init in
              for t = 0 to seq_len - 1 do
                let xt = Rune.slice [ Rune.A; Rune.I t; Rune.A ] x in
                let gates =
                  Rune.add (Rune.matmul xt w_ih)
                    (Rune.add (Rune.matmul !h2 w_hh)
                       (Rune.reshape [| 1; hidden_size * 3 |] b))
                in
                let z =
                  Rune.sigmoid
                    (Rune.slice [ Rune.A; Rune.R (0, hidden_size) ] gates)
                in
                let r =
                  Rune.sigmoid
                    (Rune.slice
                       [ Rune.A; Rune.R (hidden_size, hidden_size * 2) ]
                       gates)
                in
                let n =
                  Rune.tanh
                    (Rune.add
                       (Rune.slice
                          [ Rune.A; Rune.R (hidden_size * 2, hidden_size * 3) ]
                          gates)
                       (Rune.matmul (Rune.mul r !h2)
                          (Rune.slice [ Rune.A; Rune.R (0, hidden_size) ] w_hh)))
                in
                h2 :=
                  Rune.add
                    (Rune.mul (Rune.sub (Rune.scalar dt 1.0) z) n)
                    (Rune.mul z !h2);
                outputs.(t) <- !h2
              done;
              Rune.stack ~axis:1 (Array.to_list outputs))
            else !h
        | _ -> failwith "gru: invalid params");
  }

let lstm ~input_size ~hidden_size ?(return_sequences = false)
    ?(learned_init = false) () =
  {
    init =
      (fun ~rngs ~dtype ->
        let glorot = (Initializers.glorot_uniform ()).f in
        let keys = Rune.Rng.split ~n:2 rngs in
        let w_ih =
          glorot
            (Rune.Rng.to_int keys.(0))
            [| input_size; hidden_size * 4 |]
            dtype
        in
        let w_hh =
          glorot
            (Rune.Rng.to_int keys.(1))
            [| hidden_size; hidden_size * 4 |]
            dtype
        in
        let b = Rune.zeros dtype [| hidden_size * 4 |] in
        let base =
          [
            ("w_ih", Ptree.tensor w_ih);
            ("w_hh", Ptree.tensor w_hh);
            ("b", Ptree.tensor b);
          ]
        in
        let base =
          if learned_init then
            let h0 = Rune.zeros dtype [| hidden_size |] in
            let c0 = Rune.zeros dtype [| hidden_size |] in
            base @ [ ("h0", Ptree.tensor h0); ("c0", Ptree.tensor c0) ]
          else base
        in
        Ptree.dict base);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Ptree.Dict fields ->
            let dt = Rune.dtype x in
            let param_exn name = Ptree.Dict.get_tensor_exn fields ~name dt in
            let param_opt name = Ptree.Dict.get_tensor fields ~name dt in
            let w_ih = param_exn "w_ih"
            and w_hh = param_exn "w_hh"
            and b = param_exn "b" in
            let batch, seq_len, _ =
              match Rune.shape x with
              | [| b; s; i |] -> (b, s, i)
              | _ -> failwith "lstm: expected [b; s; i]"
            in
            let h_init =
              match param_opt "h0" with
              | Some h0 ->
                  Rune.reshape [| 1; hidden_size |] h0
                  |> Rune.expand [| batch; hidden_size |]
              | None -> Rune.zeros dt [| batch; hidden_size |]
            in
            let c_init =
              match param_opt "c0" with
              | Some c0 ->
                  Rune.reshape [| 1; hidden_size |] c0
                  |> Rune.expand [| batch; hidden_size |]
              | None -> Rune.zeros dt [| batch; hidden_size |]
            in
            let h = ref h_init in
            let c = ref c_init in
            let outputs =
              Array.make seq_len (Rune.zeros dt [| batch; hidden_size |])
            in
            for t = 0 to seq_len - 1 do
              let xt = Rune.slice [ Rune.A; Rune.I t; Rune.A ] x in
              let gates =
                Rune.add (Rune.matmul xt w_ih)
                  (Rune.add (Rune.matmul !h w_hh)
                     (Rune.reshape [| 1; hidden_size * 4 |] b))
              in
              let i =
                Rune.sigmoid
                  (Rune.slice [ Rune.A; Rune.R (0, hidden_size) ] gates)
              in
              let f =
                Rune.sigmoid
                  (Rune.slice
                     [ Rune.A; Rune.R (hidden_size, hidden_size * 2) ]
                     gates)
              in
              let g =
                Rune.tanh
                  (Rune.slice
                     [ Rune.A; Rune.R (hidden_size * 2, hidden_size * 3) ]
                     gates)
              in
              let o =
                Rune.sigmoid
                  (Rune.slice
                     [ Rune.A; Rune.R (hidden_size * 3, hidden_size * 4) ]
                     gates)
              in
              c := Rune.add (Rune.mul f !c) (Rune.mul i g);
              h := Rune.mul o (Rune.tanh !c)
            done;
            if return_sequences then (
              let h2 = ref h_init in
              let c2 = ref c_init in
              for t = 0 to seq_len - 1 do
                let xt = Rune.slice [ Rune.A; Rune.I t; Rune.A ] x in
                let gates =
                  Rune.add (Rune.matmul xt w_ih)
                    (Rune.add (Rune.matmul !h2 w_hh)
                       (Rune.reshape [| 1; hidden_size * 4 |] b))
                in
                let i =
                  Rune.sigmoid
                    (Rune.slice [ Rune.A; Rune.R (0, hidden_size) ] gates)
                in
                let f =
                  Rune.sigmoid
                    (Rune.slice
                       [ Rune.A; Rune.R (hidden_size, hidden_size * 2) ]
                       gates)
                in
                let g =
                  Rune.tanh
                    (Rune.slice
                       [ Rune.A; Rune.R (hidden_size * 2, hidden_size * 3) ]
                       gates)
                in
                let o =
                  Rune.sigmoid
                    (Rune.slice
                       [ Rune.A; Rune.R (hidden_size * 3, hidden_size * 4) ]
                       gates)
                in
                c2 := Rune.add (Rune.mul f !c2) (Rune.mul i g);
                h2 := Rune.mul o (Rune.tanh !c2);
                outputs.(t) <- !h2
              done;
              Rune.stack ~axis:1 (Array.to_list outputs))
            else !h
        | _ -> failwith "lstm: invalid params");
  }

let positional_embedding_learned ~max_len ~embed_dim () =
  {
    init =
      (fun ~rngs ~dtype ->
        let initf = (Initializers.normal_range ~mean:0.0 ~stddev:0.02 ()).f in
        let key = (Rune.Rng.split rngs).(0) in
        let table =
          initf (Rune.Rng.to_int key) [| max_len; embed_dim |] dtype
        in
        Ptree.dict [ ("table", Ptree.tensor table) ]);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Ptree.Dict fields ->
            let dtype = Rune.dtype x in
            let table = Ptree.Dict.get_tensor_exn fields ~name:"table" dtype in
            let b, s, _ =
              match Rune.shape x with
              | [| b; s; e |] -> (b, s, e)
              | _ -> failwith "positional_embedding: expected [b; s; e]"
            in
            let pos = Rune.arange Rune.int32 0 s 1 in
            let pos =
              Rune.reshape [| 1; s |] pos
              |> Rune.expand [| b; s |]
              |> Rune.contiguous
            in
            let pos_e = Rune.embedding ~scale:false ~embedding:table pos in
            Rune.add x pos_e
        | _ -> failwith "positional_embedding: invalid params");
  }

let positional_encoding_sinusoidal_table ~max_len ~embed_dim ~dtype =
  let dt = dtype in
  let d2 = embed_dim / 2 in
  let position =
    Rune.arange Rune.int32 0 max_len 1
    |> Rune.cast dt
    |> Rune.reshape [| max_len; 1 |]
  in
  let j =
    Rune.arange Rune.int32 0 d2 1 |> Rune.cast dt |> Rune.reshape [| 1; d2 |]
  in
  let exponent =
    Rune.div
      (Rune.mul (Rune.scalar dt 2.0) j)
      (Rune.scalar dt (float_of_int embed_dim))
  in
  let angle_rate = Rune.pow (Rune.scalar dt 10000.0) exponent in
  let angle = Rune.div position angle_rate in
  let sin_term = Rune.sin angle in
  let cos_term = Rune.cos angle in
  let sin_e = Rune.expand_dims [ 2 ] sin_term in
  let cos_e = Rune.expand_dims [ 2 ] cos_term in
  let stacked = Rune.stack ~axis:2 [ sin_e; cos_e ] in
  (* [L; d2; 2] *)
  Rune.reshape [| max_len; d2 * 2 |] stacked
