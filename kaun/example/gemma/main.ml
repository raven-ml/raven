open Kaun
open Kaun.Optimizer
open Ptree

(* Training configuration *)
type train_config = {
  (* Model *)
  model_variant : string;
  (* Training hyperparameters *)
  batch_size : int;
  seq_len : int;
  learning_rate : float;
  weight_decay : float;
  warmup_steps : int;
  max_steps : int;
  (* Optimization *)
  gradient_clip : float;
  gradient_accumulation_steps : int;
  (* Logging *)
  log_interval : int;
  eval_interval : int;
  checkpoint_interval : int;
  (* Paths *)
  data_path : string option;
  checkpoint_path : string option;
}

let default_train_config =
  {
    model_variant = "gemma3_1b";
    batch_size = 4;
    seq_len = 512;
    learning_rate = 3e-4;
    weight_decay = 0.01;
    warmup_steps = 1000;
    max_steps = 10000;
    gradient_clip = 1.0;
    gradient_accumulation_steps = 1;
    log_interval = 10;
    eval_interval = 100;
    checkpoint_interval = 1000;
    data_path = None;
    checkpoint_path = None;
  }

(* Learning rate schedule with warmup *)
let get_learning_rate step ~warmup_steps ~max_steps ~base_lr =
  if step < warmup_steps then
    (* Linear warmup *)
    base_lr *. (float_of_int step /. float_of_int warmup_steps)
  else
    (* Cosine decay *)
    let progress =
      float_of_int (step - warmup_steps)
      /. float_of_int (max_steps - warmup_steps)
    in
    let cosine_decay = 0.5 *. (1.0 +. cos (Float.pi *. progress)) in
    base_lr *. cosine_decay

(* Gradient clipping *)
let clip_gradients grads ~max_norm =
  (* Compute global norm *)
  let compute_norm params =
    let rec norm_sq = function
      | Tensor t ->
          let t_sq = Rune.square t in
          Rune.item [] (Rune.sum t_sq)
      | List l -> List.fold_left (fun acc p -> acc +. norm_sq p) 0.0 l
      | Record fields ->
          List.fold_left
            (fun acc (_, p) -> acc +. norm_sq p)
            0.0 (Record.bindings fields)
    in
    sqrt (norm_sq params)
  in

  let grad_norm = compute_norm grads in

  if grad_norm > max_norm then
    let scale = max_norm /. grad_norm in
    let rec scale_params = function
      | Tensor t ->
          Tensor (Rune.mul t (Rune.scalar (Rune.device t) (Rune.dtype t) scale))
      | List l -> List (List.map scale_params l)
      | Record fields ->
          let scaled_bindings =
            List.map
              (fun (k, v) -> (k, scale_params v))
              (Record.bindings fields)
          in
          record_of scaled_bindings
    in
    scale_params grads
  else grads

(* Training step *)
let train_step model params optimizer opt_state batch_inputs batch_targets
    ~config ~step ~rngs =
  (* Compute learning rate for this step *)
  let lr =
    get_learning_rate step ~warmup_steps:config.warmup_steps
      ~max_steps:config.max_steps ~base_lr:config.learning_rate
  in

  (* Apply learning rate schedule by scaling gradients *)
  let lr_scale = lr /. config.learning_rate in

  (* Forward pass and loss computation *)
  let loss_fn params =
    let logits = apply model params ~training:true ~rngs batch_inputs in

    (* Reshape for cross-entropy loss *)
    let batch_size = (Rune.shape logits).(0) in
    let seq_len = (Rune.shape logits).(1) in
    let vocab_size = (Rune.shape logits).(2) in

    let logits_flat =
      Rune.reshape [| batch_size * seq_len; vocab_size |] logits
    in
    let targets_flat = Rune.reshape [| batch_size * seq_len |] batch_targets in

    (* Cross-entropy loss *)
    Loss.softmax_cross_entropy_with_indices logits_flat targets_flat
  in

  (* Compute loss and gradients *)
  let loss, grads = value_and_grad loss_fn params in

  (* Apply gradient accumulation scaling if needed *)
  let grads =
    if config.gradient_accumulation_steps > 1 then
      let accumulation_scale =
        1.0 /. float_of_int config.gradient_accumulation_steps
      in
      let rec scale_params scale = function
        | Tensor t ->
            Tensor
              (Rune.mul t (Rune.scalar (Rune.device t) (Rune.dtype t) scale))
        | List l -> List (List.map (scale_params scale) l)
        | Record fields ->
            let scaled_bindings =
              List.map
                (fun (k, v) -> (k, scale_params scale v))
                (Record.bindings fields)
            in
            record_of scaled_bindings
      in
      scale_params accumulation_scale grads
    else grads
  in

  (* Clip gradients *)
  let grads = clip_gradients grads ~max_norm:config.gradient_clip in

  (* Apply learning rate schedule *)
  let rec scale_by_lr scale = function
    | Tensor t ->
        Tensor (Rune.mul t (Rune.scalar (Rune.device t) (Rune.dtype t) scale))
    | List l -> List (List.map (scale_by_lr scale) l)
    | Record fields ->
        let scaled_bindings =
          List.map
            (fun (k, v) -> (k, scale_by_lr scale v))
            (Record.bindings fields)
        in
        record_of scaled_bindings
  in
  let grads = scale_by_lr lr_scale grads in

  (* Update parameters *)
  let updates, new_state = optimizer.update !opt_state params grads in
  opt_state := new_state;
  apply_updates_inplace params updates;

  loss

(* Evaluation step *)
let eval_step model params batch_inputs batch_targets ~rngs =
  let logits = apply model params ~training:false ~rngs batch_inputs in

  (* Compute loss *)
  let batch_size = (Rune.shape logits).(0) in
  let seq_len = (Rune.shape logits).(1) in
  let vocab_size = (Rune.shape logits).(2) in

  let logits_flat =
    Rune.reshape [| batch_size * seq_len; vocab_size |] logits
  in
  let targets_flat = Rune.reshape [| batch_size * seq_len |] batch_targets in

  let loss = Loss.softmax_cross_entropy_with_indices logits_flat targets_flat in

  (* Compute accuracy *)
  let predictions = Rune.argmax logits_flat ~axis:1 ~keepdims:false in
  let targets_int = Rune.cast Rune.int32 targets_flat in
  let correct = Rune.equal predictions targets_int in
  let accuracy = Rune.mean (Rune.cast Rune.float32 correct) in

  (loss, accuracy)

(* Main training loop *)
let train config_name train_config =
  (* Select model configuration *)
  let model_config =
    match config_name with
    | "gemma3_1b" -> Config.gemma3_1b
    | "gemma3_4b" -> Config.gemma3_4b
    | "gemma3_12b" -> Config.gemma3_12b
    | "gemma3_27b" -> Config.gemma3_27b
    | _ -> failwith (Printf.sprintf "Unknown model config: %s" config_name)
  in

  Printf.printf "Training Gemma %s\n" config_name;
  Printf.printf "=====================================\n";
  Printf.printf "Model Configuration:\n";
  Printf.printf "  Layers: %d\n" model_config.num_layers;
  Printf.printf "  Embed dim: %d\n" model_config.embed_dim;
  Printf.printf "  Hidden dim: %d\n" model_config.hidden_dim;
  Printf.printf "  Num heads: %d\n" model_config.num_heads;
  Printf.printf "  KV heads: %d\n" model_config.num_kv_heads;
  Printf.printf "  Head dim: %d\n" model_config.head_dim;
  Printf.printf "  Vocab size: %d\n" model_config.vocab_size;
  Printf.printf "\n";
  Printf.printf "Training Configuration:\n";
  Printf.printf "  Model variant: %s\n" train_config.model_variant;
  Printf.printf "  Batch size: %d\n" train_config.batch_size;
  Printf.printf "  Sequence length: %d\n" train_config.seq_len;
  Printf.printf "  Learning rate: %.2e\n" train_config.learning_rate;
  Printf.printf "  Gradient accumulation steps: %d\n"
    train_config.gradient_accumulation_steps;
  Printf.printf "  Weight decay: %.4f\n" train_config.weight_decay;
  Printf.printf "  Warmup steps: %d\n" train_config.warmup_steps;
  Printf.printf "  Max steps: %d\n" train_config.max_steps;
  Printf.printf "=====================================\n\n";

  (* Load data if path provided *)
  (match train_config.data_path with
  | Some path -> Printf.printf "Loading data from: %s\n" path
  (* TODO: Implement actual data loading *)
  | None -> Printf.printf "Using synthetic data for training\n");

  (* Create model *)
  let model = Transformer.create_gemma_model model_config in

  (* Initialize RNG *)
  let rngs = Rune.Rng.key 42 in

  (* Create device and dtype *)
  let device = Rune.c in
  let dtype_float = Rune.float32 in

  (* Initialize or load model parameters *)
  let params =
    match train_config.checkpoint_path with
    | Some path when Sys.file_exists path ->
        Printf.printf "Loading checkpoint from: %s\n" path;
        let loaded_params =
          Checkpoint.load_params ~path ~device ~dtype:dtype_float
        in
        Printf.printf "Loaded checkpoint\n";
        loaded_params
    | Some path ->
        Printf.printf "Checkpoint path specified but file not found: %s\n" path;
        Printf.printf "Initializing model parameters from scratch...\n";
        let params = init model ~rngs ~device ~dtype:dtype_float in
        Printf.printf "Model initialized!\n\n";
        params
    | None ->
        Printf.printf "Initializing model parameters from scratch...\n";
        let params = init model ~rngs ~device ~dtype:dtype_float in
        Printf.printf "Model initialized!\n\n";
        params
  in

  (* Create optimizer *)
  let optimizer =
    adamw ~lr:train_config.learning_rate ~weight_decay:train_config.weight_decay
      ()
  in
  let opt_state = ref (optimizer.init params) in

  (* Training metrics - currently unused, kept for future implementation *)
  let _metrics =
    Metrics.Collection.create
      [
        ("train_loss", Metrics.mae ());
        ("eval_loss", Metrics.mae ());
        ("eval_accuracy", Metrics.accuracy ());
      ]
  in

  (* Training loop *)
  Printf.printf "Starting training...\n";
  Printf.printf "-------------------\n";

  for step = 1 to train_config.max_steps do
    (* Generate synthetic batch *)
    let batch_inputs =
      Rune.randint device Rune.int32 ~seed:(step * 1000)
        ~high:model_config.Config.vocab_size
        [| train_config.batch_size; train_config.seq_len |]
        0
    in
    let batch_targets =
      Rune.randint device Rune.int32
        ~seed:((step * 1000) + 1)
        ~high:model_config.Config.vocab_size
        [| train_config.batch_size; train_config.seq_len |]
        0
    in

    (* Training step - cast inputs to float for model interface *)
    let batch_inputs_float = Rune.cast Rune.float32 batch_inputs in
    let batch_targets_float = Rune.cast Rune.float32 batch_targets in
    let train_loss =
      train_step model params optimizer opt_state batch_inputs_float
        batch_targets_float ~config:train_config ~step ~rngs
    in

    (* Update metrics - for now, skip as we don't have proper predictions/targets *)
    (* TODO: Properly update metrics when we have predictions and targets *)
    ();

    (* Logging *)
    (if step mod train_config.log_interval = 0 then
       let train_loss_val = Rune.item [] train_loss in
       let lr =
         get_learning_rate step ~warmup_steps:train_config.warmup_steps
           ~max_steps:train_config.max_steps ~base_lr:train_config.learning_rate
       in
       Printf.printf "Step %5d/%d | Loss: %.4f | LR: %.2e\n" step
         train_config.max_steps train_loss_val lr);

    (* Evaluation *)
    (if step mod train_config.eval_interval = 0 then
       (* Generate eval batch *)
       let eval_inputs =
         Rune.randint device Rune.int32 ~seed:(step * 2000)
           ~high:model_config.Config.vocab_size
           [| train_config.batch_size; train_config.seq_len |]
           0
       in
       let eval_targets =
         Rune.randint device Rune.int32
           ~seed:((step * 2000) + 1)
           ~high:model_config.Config.vocab_size
           [| train_config.batch_size; train_config.seq_len |]
           0
       in

       let eval_inputs_float = Rune.cast Rune.float32 eval_inputs in
       let eval_targets_float = Rune.cast Rune.float32 eval_targets in
       let eval_loss, eval_acc =
         eval_step model params eval_inputs_float eval_targets_float ~rngs
       in
       let eval_loss_val = Rune.item [] eval_loss in
       let eval_acc_val = Rune.item [] eval_acc in

       Printf.printf "  [Eval] Loss: %.4f | Accuracy: %.2f%%\n" eval_loss_val
         (eval_acc_val *. 100.0));

    (* Checkpointing *)
    if step mod train_config.checkpoint_interval = 0 then (
      let checkpoint_path =
        match train_config.checkpoint_path with
        | Some base_path -> Printf.sprintf "%s.step_%d" base_path step
        | None -> Printf.sprintf "checkpoint.step_%d" step
      in
      Printf.printf "  [Checkpoint] Saving model to %s\n" checkpoint_path;
      let metadata =
        [
          ("step", string_of_int step);
          ("model_variant", train_config.model_variant);
          ("batch_size", string_of_int train_config.batch_size);
          ("seq_len", string_of_int train_config.seq_len);
        ]
      in
      Checkpoint.save_params ~path:checkpoint_path ~params ~metadata ())
  done;

  Printf.printf "\nTraining complete!\n"

(* Command-line interface *)
let () =
  let config_name =
    if Array.length Sys.argv > 1 then Sys.argv.(1) else "gemma3_1b"
  in

  let train_config =
    if config_name = "gemma3_1b" then
      {
        default_train_config with
        batch_size = 4;
        seq_len = 256;
        max_steps = 100;
      }
    else default_train_config
  in

  train config_name train_config
