open Rune
open Kaun

let load_names path = Saga.read_lines path |> List.map String.lowercase_ascii

let build_vocab (names : string list) =
  let tbl = Hashtbl.create 64 in
  Hashtbl.add tbl '.' 0;
  List.iter
    (fun n ->
      String.iter
        (fun c ->
          if not (Hashtbl.mem tbl c) then Hashtbl.add tbl c (Hashtbl.length tbl))
        n)
    names;
  let vocab_size = Hashtbl.length tbl in
  let idx2ch = Array.make vocab_size '.' in
  Hashtbl.iter (fun c i -> if i < vocab_size then idx2ch.(i) <- c) tbl;
  let ch2idx c = match Hashtbl.find_opt tbl c with Some i -> i | None -> 0 in
  let encode (s : string) =
    s |> String.to_seq |> List.of_seq |> List.map ch2idx
  in
  let decode ids =
    ids |> List.map (fun i -> idx2ch.(i)) |> List.to_seq |> String.of_seq
  in
  (vocab_size, ch2idx, encode, decode)

let make_dataset ~block_size ~batch_size ~vocab_size:_ ~encode names =
  let tokenize s = encode ("." ^ s ^ ".") in
  Dataset.sliding_window ~block_size ~tokenize names
  |> Dataset.batch batch_size
  |> Dataset.shuffle ~buffer_size:200

let split_names_for_val names =
  let n = List.length names in
  let test_set_size = Stdlib.min 1000 (Stdlib.max 1 (n / 10)) in
  (* deterministic split for reproducibility *)
  let indices = Array.init n Fun.id in
  (* simple shuffle with fixed seed *)
  let st = Random.State.make [| 3407 |] in
  for i = n - 1 downto 1 do
    let j = Random.State.int st (i + 1) in
    let tmp = indices.(i) in
    indices.(i) <- indices.(j);
    indices.(j) <- tmp
  done;
  let train_idx = Array.sub indices 0 (n - test_set_size) in
  let test_idx = Array.sub indices (n - test_set_size) test_set_size in
  let arr = Array.of_list names in
  let train = Array.to_list (Array.map (Array.get arr) train_idx) in
  let test = Array.to_list (Array.map (Array.get arr) test_idx) in
  (train, test)

let make_optimizer ~lr ~weight_decay =
  if weight_decay > 0.0 then Optimizer.adamw ~lr ~weight_decay ()
  else Optimizer.adam ~lr ()

let bigram_mle_loss ~vocab_size ~encode names =
  (* Compute exact bigram MLE negative log-likelihood (nats) over the data *)
  let counts = Array.make_matrix vocab_size vocab_size 0. in
  let total_pairs = ref 0 in
  List.iter
    (fun n ->
      let s = "." ^ n ^ "." in
      let ids = Array.of_list (encode s) in
      for i = 0 to Array.length ids - 2 do
        let a = ids.(i) in
        let b = ids.(i + 1) in
        counts.(a).(b) <- counts.(a).(b) +. 1.;
        incr total_pairs
      done)
    names;
  (* Row-normalize with add-k smoothing to avoid zeros *)
  let k = 1e-3 in
  let loglik = ref 0. in
  List.iter
    (fun n ->
      let s = "." ^ n ^ "." in
      let ids = Array.of_list (encode s) in
      for i = 0 to Array.length ids - 2 do
        let a = ids.(i) in
        let b = ids.(i + 1) in
        let row_sum =
          Array.fold_left ( +. ) 0. (Array.map (fun c -> c +. k) counts.(a))
        in
        let p = (counts.(a).(b) +. k) /. row_sum in
        loglik := !loglik +. Stdlib.log p
      done)
    names;
  let n = float_of_int !total_pairs in
  let nll = -. !loglik /. n in
  nll

let train_bigram ~vocab_size ~epochs ~lr ~weight_decay ~val_data train_data =
  let open Layer in
  let model =
    sequential [ embedding ~vocab_size ~embed_dim:vocab_size (); flatten () ]
  in
  let optimizer = make_optimizer ~lr ~weight_decay in
  Printf.printf "[makemore] Training bigram model for %d epoch(s)...\n%!" epochs;
  let state, _ =
    Training.fit ~model ~optimizer
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ~train_data ~val_data
      ~epochs ~progress:true ~rngs:(Rng.key 42) ~dtype:float32 ()
  in
  Printf.printf "[makemore] Training complete.\n%!";
  fun x -> apply model state.Training.State.params ~training:false x

let train_mlp ~vocab_size ~block_size ~n_embd ~n_embd2 ~epochs ~lr ~weight_decay
    ~val_data train_data =
  let open Layer in
  let model =
    sequential
      [
        embedding ~vocab_size ~embed_dim:n_embd ();
        flatten ();
        linear ~in_features:(block_size * n_embd) ~out_features:n_embd2 ();
        tanh ();
        linear ~in_features:n_embd2 ~out_features:vocab_size ();
      ]
  in
  let optimizer = make_optimizer ~lr ~weight_decay in
  Printf.printf "[makemore] Training mlp model for %d epoch(s)...\n%!" epochs;
  let state, _ =
    Training.fit ~model ~optimizer
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ~train_data ~val_data
      ~epochs ~progress:true ~rngs:(Rng.key 42) ~dtype:float32 ()
  in
  Printf.printf "[makemore] Training complete.\n%!";
  fun x -> apply model state.Training.State.params ~training:false x

let train_rnn ~vocab_size ~block_size:_ ~n_embd ~n_embd2 ~epochs ~lr
    ~weight_decay ~val_data train_data =
  let open Layer in
  let model =
    sequential
      [
        embedding ~vocab_size ~embed_dim:n_embd ();
        rnn ~input_size:n_embd ~hidden_size:n_embd2 ~return_sequences:true
          ~learned_init:true ();
        (* take last time step *)
        {
          init = (fun ~rngs:_ ~dtype:_ -> List []);
          apply =
            (fun _ ~training:_ ?rngs:_ x ->
              let s = (Rune.shape x).(1) in
              Rune.slice [ Rune.A; Rune.I (s - 1); Rune.A ] x);
        };
        linear ~in_features:n_embd2 ~out_features:vocab_size ();
      ]
  in
  let optimizer = make_optimizer ~lr ~weight_decay in
  Printf.printf "[makemore] Training rnn model for %d epoch(s)...\n%!" epochs;
  let state, _ =
    Training.fit ~model ~optimizer
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ~train_data ~val_data
      ~epochs ~progress:true ~rngs:(Rng.key 42) ~dtype:float32 ()
  in
  Printf.printf "[makemore] Training complete.\n%!";
  fun x -> apply model state.Training.State.params ~training:false x

let train_lstm ~vocab_size ~block_size:_ ~n_embd ~n_embd2 ~epochs ~lr
    ~weight_decay ~val_data train_data =
  let open Layer in
  let model =
    sequential
      [
        embedding ~vocab_size ~embed_dim:n_embd ();
        lstm ~input_size:n_embd ~hidden_size:n_embd2 ~return_sequences:true
          ~learned_init:true ();
        (* take last time step *)
        {
          init = (fun ~rngs:_ ~dtype:_ -> List []);
          apply =
            (fun _ ~training:_ ?rngs:_ x ->
              let s = (Rune.shape x).(1) in
              Rune.slice [ Rune.A; Rune.I (s - 1); Rune.A ] x);
        };
        linear ~in_features:n_embd2 ~out_features:vocab_size ();
      ]
  in
  let optimizer = make_optimizer ~lr ~weight_decay in
  Printf.printf "[makemore] Training lstm model for %d epoch(s)...\n%!" epochs;
  let state, _ =
    Training.fit ~model ~optimizer
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ~train_data ~val_data
      ~epochs ~progress:true ~rngs:(Rng.key 42) ~dtype:float32 ()
  in
  Printf.printf "[makemore] Training complete.\n%!";
  fun x -> apply model state.Training.State.params ~training:false x

let train_gru ~vocab_size ~block_size:_ ~n_embd ~n_embd2 ~epochs ~lr
    ~weight_decay ~val_data train_data =
  let open Layer in
  let model =
    sequential
      [
        embedding ~vocab_size ~embed_dim:n_embd ();
        gru ~input_size:n_embd ~hidden_size:n_embd2 ~return_sequences:true
          ~learned_init:true ();
        (* take last time step *)
        {
          init = (fun ~rngs:_ ~dtype:_ -> List []);
          apply =
            (fun _ ~training:_ ?rngs:_ x ->
              let s = (Rune.shape x).(1) in
              Rune.slice [ Rune.A; Rune.I (s - 1); Rune.A ] x);
        };
        linear ~in_features:n_embd2 ~out_features:vocab_size ();
      ]
  in
  let optimizer = make_optimizer ~lr ~weight_decay in
  Printf.printf "[makemore] Training gru model for %d epoch(s)...\n%!" epochs;
  let state, _ =
    Training.fit ~model ~optimizer
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ~train_data ~val_data
      ~epochs ~progress:true ~rngs:(Rng.key 42) ~dtype:float32 ()
  in
  Printf.printf "[makemore] Training complete.\n%!";
  fun x -> apply model state.Training.State.params ~training:false x

let train_cnn ~vocab_size ~block_size:_ ~epochs ~lr ~weight_decay ~val_data
    train_data =
  let open Layer in
  let model =
    sequential
      [
        embedding ~vocab_size ~embed_dim:32 ();
        (* to [b; channels; length] for conv1d *)
        {
          init = (fun ~rngs:_ ~dtype:_ -> List []);
          apply =
            (fun _ ~training:_ ?rngs:_ x ->
              Rune.transpose ~axes:[ 0; 2; 1 ] x);
        };
        conv1d ~in_channels:32 ~out_channels:64 ~kernel_size:3 ~padding:`Causal
          ();
        (* back to [b; length; channels] *)
        {
          init = (fun ~rngs:_ ~dtype:_ -> List []);
          apply =
            (fun _ ~training:_ ?rngs:_ x ->
              Rune.transpose ~axes:[ 0; 2; 1 ] x);
        };
        (* take last time step *)
        {
          init = (fun ~rngs:_ ~dtype:_ -> List []);
          apply =
            (fun _ ~training:_ ?rngs:_ x ->
              let s = (Rune.shape x).(1) in
              Rune.slice [ Rune.A; Rune.I (s - 1); Rune.A ] x);
        };
        linear ~in_features:64 ~out_features:vocab_size ();
      ]
  in
  let optimizer = make_optimizer ~lr ~weight_decay in
  Printf.printf "[makemore] Training cnn model for %d epoch(s)...\n%!" epochs;
  let state, _ =
    Training.fit ~model ~optimizer
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ~train_data ~val_data
      ~epochs ~progress:true ~rngs:(Rng.key 42) ~dtype:float32 ()
  in
  Printf.printf "[makemore] Training complete.\n%!";
  fun x -> apply model state.Training.State.params ~training:false x

let train_transformer ~vocab_size ~block_size ~n_layer ~n_head ~n_embd ~lr
    ~weight_decay ~epochs ~val_data train_data =
  let open Layer in
  let embed_dim = n_embd in
  let num_layers = n_layer and n_head = n_head in
  (* Match Python: learned positional embeddings and final LayerNorm *)
  let pos = positional_embedding_learned ~max_len:block_size ~embed_dim () in
  let decoder =
    transformer_decoder ~num_layers ~embed_dim ~num_heads:n_head
      ~mlp_hidden:(4 * embed_dim) ()
  in
  let ln_f = layer_norm ~dim:embed_dim () in
  let model =
    sequential
      [
        embedding ~vocab_size ~embed_dim ();
        pos;
        decoder;
        ln_f;
        (* take last time step *)
        {
          init = (fun ~rngs:_ ~dtype:_ -> List []);
          apply =
            (fun _ ~training:_ ?rngs:_ x ->
              let s = (Rune.shape x).(1) in
              Rune.slice [ Rune.A; Rune.I (s - 1); Rune.A ] x);
        };
        linear ~in_features:embed_dim ~out_features:vocab_size ();
      ]
  in
  let optimizer = make_optimizer ~lr ~weight_decay in
  Printf.printf "[makemore] Training transformer model for %d epoch(s)...\n%!"
    epochs;
  let state, _ =
    Training.fit ~model ~optimizer
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ~train_data ~val_data
      ~epochs ~progress:true ~rngs:(Rng.key 42) ~dtype:float32 ()
  in
  Printf.printf "[makemore] Training complete.\n%!";
  fun x -> apply model state.Training.State.params ~training:false x

(* BoW model *)
let bow_block ~n_embd ~n_embd2 () : Layer.module_ =
  {
    Layer.init =
      (fun ~rngs ~dtype ->
        let glorot = (Initializers.glorot_uniform ()).f in
        let keys = Rune.Rng.split ~n:2 rngs in
        let w1 =
          glorot (Rune.Rng.to_int keys.(0)) [| n_embd; n_embd2 |] dtype
        in
        let b1 = Rune.zeros dtype [| n_embd2 |] in
        let w2 =
          glorot (Rune.Rng.to_int keys.(1)) [| n_embd2; n_embd |] dtype
        in
        let b2 = Rune.zeros dtype [| n_embd |] in
        Ptree.record_of
          [
            ("w1", Ptree.Tensor w1);
            ("b1", Ptree.Tensor b1);
            ("w2", Ptree.Tensor w2);
            ("b2", Ptree.Tensor b2);
          ]);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Ptree.Record fields ->
            let get name =
              match Ptree.Record.find_opt name fields with
              | Some (Ptree.Tensor t) -> t
              | _ -> failwith ("bow_block: missing " ^ name)
            in
            let w1 = get "w1"
            and b1 = get "b1"
            and w2 = get "w2"
            and b2 = get "b2" in
            let b, s, _ =
              match Rune.shape x with
              | [| b; s; c |] -> (b, s, c)
              | _ -> failwith "bow_block: expected [b; s; c]"
            in
            (* causal uniform attention weights *)
            let dt = Rune.dtype x in
            let ones = Rune.ones dt [| s; s |] in
            let lower = Rune.tril ones in
            let counts =
              Rune.arange Rune.int32 1 (s + 1) 1
              |> Rune.cast dt
              |> Rune.reshape [| 1; s; 1 |]
            in
            let weights =
              Rune.reshape [| 1; s; s |] lower
              |> Rune.div counts
              |> Rune.expand [| b; s; s |]
            in
            let ctx = Rune.matmul weights x in
            let y = Rune.add x ctx in
            let h1 = Rune.add (Rune.matmul y w1) b1 |> Rune.tanh in
            let h2 = Rune.add (Rune.matmul h1 w2) b2 in
            Rune.add y h2
        | _ -> failwith "bow_block: invalid params");
  }

let train_bow ~vocab_size ~block_size ~n_embd ~n_embd2 ~epochs ~lr ~weight_decay
    ~val_data train_data =
  let open Layer in
  let pos =
    positional_embedding_learned ~max_len:block_size ~embed_dim:n_embd ()
  in
  let block = bow_block ~n_embd ~n_embd2 () in
  let model =
    sequential
      [
        embedding ~vocab_size ~embed_dim:n_embd ();
        pos;
        block;
        (* take last time step *)
        {
          init = (fun ~rngs:_ ~dtype:_ -> List []);
          apply =
            (fun _ ~training:_ ?rngs:_ x ->
              let s = (Rune.shape x).(1) in
              Rune.slice [ Rune.A; Rune.I (s - 1); Rune.A ] x);
        };
        linear ~in_features:n_embd ~out_features:vocab_size ();
      ]
  in
  let optimizer = make_optimizer ~lr ~weight_decay in
  Printf.printf "[makemore] Training bow model for %d epoch(s)...\n%!" epochs;
  let state, _ =
    Training.fit ~model ~optimizer
      ~loss_fn:Loss.softmax_cross_entropy_with_indices ~train_data ~val_data
      ~epochs ~progress:true ~rngs:(Rng.key 42) ~dtype:float32 ()
  in
  Printf.printf "[makemore] Training complete.\n%!";
  fun x -> apply model state.Training.State.params ~training:false x

let generate ~model_fn ~eos_id ~encode ~decode ~max_new =
  let tokenizer (s : string) = encode s in
  let decoder ids = decode ids in
  let config =
    Saga.Sampler.(
      default |> with_do_sample true |> with_temperature 0.9
      |> with_max_new_tokens max_new)
  in
  let stop_on_eos = Saga.Sampler.eos_token_criteria ~eos_token_ids:[ eos_id ] in
  Saga.Sampler.generate_text ~model:model_fn ~tokenizer ~decoder ~prompt:"."
    ~generation_config:config ~stopping_criteria:[ stop_on_eos ] ()

let main () =
  let model_choice = ref "bigram" in
  let names_path = ref "dev/makemore/names.txt" in
  let epochs = ref 1 in
  let batch_size = ref 256 in
  let n_layer = ref 2 in
  let n_head = ref 4 in
  let n_embd = ref 64 in
  let n_embd2 = ref 64 in
  let learning_rate = ref 2e-3 in
  let weight_decay = ref 0.01 in
  let () =
    Arg.parse
      [
        ( "--model",
          Arg.Set_string model_choice,
          "Model: bigram | mlp | rnn | lstm | gru | cnn | bow | transformer" );
        ("--data", Arg.Set_string names_path, "Path to names.txt");
        ("--epochs", Arg.Set_int epochs, "Number of training epochs (default 1)");
        ("--batch-size", Arg.Set_int batch_size, "Batch size (default 256)");
        ( "--n-layer",
          Arg.Set_int n_layer,
          "Number of transformer layers (default 2)" );
        ("--n-head", Arg.Set_int n_head, "Number of attention heads (default 4)");
        ("--n-embd", Arg.Set_int n_embd, "Embedding dimension (default 64)");
        ( "--n-embd2",
          Arg.Set_int n_embd2,
          "Hidden dimension for MLP/RNN/GRU/LSTM (default 64)" );
        ( "--learning-rate",
          Arg.Set_float learning_rate,
          "Learning rate (default 2e-3)" );
        ( "--weight-decay",
          Arg.Set_float weight_decay,
          "Weight decay (default 0.01, set 0 to disable AdamW)" );
      ]
      (fun _ -> ())
      "makemore options"
  in
  let names = load_names !names_path in
  Printf.printf "[makemore] Loaded %d names from %s\n%!" (List.length names)
    !names_path;
  Printf.printf "[makemore] First 10 names:\n%!";
  List.iteri
    (fun i n -> if i < 10 then Printf.printf "  %2d: %s\n%!" (i + 1) n)
    names;
  let vocab_size, ch2idx, encode, decode = build_vocab names in
  Printf.printf "[makemore] Vocab size: %d\n%!" vocab_size;
  (* Quick estimate of dataset windows for visibility *)
  let window_count block_size =
    List.fold_left
      (fun acc n ->
        let len = String.length n + 2 in
        acc + Stdlib.max 0 (len - block_size))
      0 names
  in
  let train_names, val_names = split_names_for_val names in
  let block_size, fwd =
    match String.lowercase_ascii !model_choice with
    | "bigram" ->
        Printf.printf "[makemore] Model: bigram (block_size=1)\n%!";
        Printf.printf "[makemore] Building dataset (windows ~ %d)...\n%!"
          (window_count 1);
        let train_ds =
          make_dataset ~block_size:1 ~batch_size:!batch_size ~vocab_size ~encode
            train_names
        in
        let val_ds =
          make_dataset ~block_size:1 ~batch_size:!batch_size ~vocab_size ~encode
            val_names
        in
        let mle = bigram_mle_loss ~vocab_size ~encode names in
        Printf.printf "[makemore] Bigram MLE loss (nats): %.4f\n%!" mle;
        Printf.printf "[makemore] Dataset ready.\n%!";
        Printf.printf "[makemore] Training for %d epoch(s)...\n%!" !epochs;
        ( 1,
          train_bigram ~vocab_size ~epochs:!epochs ~lr:!learning_rate
            ~weight_decay:!weight_decay ~val_data:val_ds train_ds )
    | "mlp" ->
        let bs = 3 in
        Printf.printf "[makemore] Model: mlp (block_size=%d)\n%!" bs;
        Printf.printf "[makemore] Building dataset (windows ~ %d)...\n%!"
          (window_count bs);
        let train_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode train_names
        in
        let val_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode val_names
        in
        Printf.printf "[makemore] Dataset ready.\n%!";
        Printf.printf "[makemore] Training for %d epoch(s)...\n%!" !epochs;
        ( bs,
          train_mlp ~vocab_size ~block_size:bs ~n_embd:!n_embd ~n_embd2:!n_embd2
            ~epochs:!epochs ~lr:!learning_rate ~weight_decay:!weight_decay
            ~val_data:val_ds train_ds )
    | "rnn" ->
        let bs = 16 in
        Printf.printf "[makemore] Model: rnn (block_size=%d)\n%!" bs;
        Printf.printf "[makemore] Building dataset (windows ~ %d)...\n%!"
          (window_count bs);
        let train_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode train_names
        in
        let val_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode val_names
        in
        Printf.printf "[makemore] Dataset ready.\n%!";
        ( bs,
          train_rnn ~vocab_size ~block_size:bs ~n_embd:!n_embd ~n_embd2:!n_embd2
            ~epochs:!epochs ~lr:!learning_rate ~weight_decay:!weight_decay
            ~val_data:val_ds train_ds )
    | "lstm" ->
        let bs = 16 in
        Printf.printf "[makemore] Model: lstm (block_size=%d)\n%!" bs;
        Printf.printf "[makemore] Building dataset (windows ~ %d)...\n%!"
          (window_count bs);
        let train_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode train_names
        in
        let val_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode val_names
        in
        Printf.printf "[makemore] Dataset ready.\n%!";
        ( bs,
          train_lstm ~vocab_size ~block_size:bs ~n_embd:!n_embd
            ~n_embd2:!n_embd2 ~epochs:!epochs ~lr:!learning_rate
            ~weight_decay:!weight_decay ~val_data:val_ds train_ds )
    | "gru" ->
        let bs = 16 in
        Printf.printf "[makemore] Model: gru (block_size=%d)\n%!" bs;
        Printf.printf "[makemore] Building dataset (windows ~ %d)...\n%!"
          (window_count bs);
        let train_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode train_names
        in
        let val_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode val_names
        in
        Printf.printf "[makemore] Dataset ready.\n%!";
        ( bs,
          train_gru ~vocab_size ~block_size:bs ~n_embd:!n_embd ~n_embd2:!n_embd2
            ~epochs:!epochs ~lr:!learning_rate ~weight_decay:!weight_decay
            ~val_data:val_ds train_ds )
    | "cnn" ->
        let bs = 16 in
        Printf.printf "[makemore] Model: cnn (block_size=%d)\n%!" bs;
        Printf.printf "[makemore] Building dataset (windows ~ %d)...\n%!"
          (window_count bs);
        let train_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode train_names
        in
        let val_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode val_names
        in
        Printf.printf "[makemore] Dataset ready.\n%!";
        ( bs,
          train_cnn ~vocab_size ~block_size:bs ~epochs:!epochs
            ~lr:!learning_rate ~weight_decay:!weight_decay ~val_data:val_ds
            train_ds )
    | "bow" ->
        let bs = 16 in
        Printf.printf "[makemore] Model: bow (block_size=%d)\n%!" bs;
        Printf.printf "[makemore] Building dataset (windows ~ %d)...\n%!"
          (window_count bs);
        let train_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode train_names
        in
        let val_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode val_names
        in
        Printf.printf "[makemore] Dataset ready.\n%!";
        ( bs,
          train_bow ~vocab_size ~block_size:bs ~n_embd:!n_embd ~n_embd2:!n_embd2
            ~epochs:!epochs ~lr:!learning_rate ~weight_decay:!weight_decay
            ~val_data:val_ds train_ds )
    | "transformer" ->
        let bs = 16 in
        Printf.printf "[makemore] Model: transformer (block_size=%d)\n%!" bs;
        Printf.printf "[makemore] Building dataset (windows ~ %d)...\n%!"
          (window_count bs);
        let train_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode train_names
        in
        let val_ds =
          make_dataset ~block_size:bs ~batch_size:!batch_size ~vocab_size
            ~encode val_names
        in
        Printf.printf "[makemore] Dataset ready.\n%!";
        ( bs,
          train_transformer ~vocab_size ~block_size:bs ~n_layer:!n_layer
            ~n_head:!n_head ~n_embd:!n_embd ~lr:!learning_rate
            ~weight_decay:!weight_decay ~epochs:!epochs ~val_data:val_ds
            train_ds )
    | x ->
        Printf.eprintf "Unknown model: %s\n%!" x;
        exit 2
  in

  (* model_fn for sampler – accepts a history of token ids and returns logits
     for next token *)
  let model_fn (tokens : int list) =
    let open Array in
    let ctx =
      let arr = of_list tokens in
      if block_size = 1 then
        [| (if length arr = 0 then ch2idx '.' else arr.(length arr - 1)) |]
      else
        let pad = ch2idx '.' in
        let ctx = Array.make block_size pad in
        let take = Stdlib.min (length arr) block_size in
        blit arr (Stdlib.max 0 (length arr - take)) ctx (block_size - take) take;
        ctx
    in
    let input =
      create float32 [| 1; block_size |] (Array.map float_of_int ctx)
    in
    fwd input |> to_array
  in

  Printf.printf "\n[makemore] --- Generated names (%s) ---\n%!"
    (String.lowercase_ascii !model_choice);
  for i = 1 to 20 do
    let s =
      generate ~model_fn ~eos_id:(ch2idx '.') ~encode ~decode ~max_new:30
    in
    (* Prefer text between a pair of '.' if present; otherwise strip
       trailing/leading '.' *)
    let cleaned =
      match String.index_opt s '.' with
      | Some i -> (
          match String.index_from_opt s (i + 1) '.' with
          | Some j when j > i + 1 -> String.sub s (i + 1) (j - i - 1)
          | _ -> s)
      | None -> s
    in
    let cleaned =
      let len = String.length cleaned in
      let start = if len > 0 && cleaned.[0] = '.' then 1 else 0 in
      let stop =
        let l = String.length cleaned in
        if l > start && cleaned.[l - 1] = '.' then l - 1 else l
      in
      if stop > start then String.sub cleaned start (stop - start) else ""
    in
    if String.length cleaned > 0 then Printf.printf "%2d: %s\n%!" i cleaned
  done

let () = main ()
