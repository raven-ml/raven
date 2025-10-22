module State = struct
  type 'layout t = {
    step : int;
    params : 'layout Ptree.t;
    opt_state : 'layout Optimizer.opt_state;
    metrics : 'layout Metrics.Collection.t option;
    rngs : Rune.Rng.key;
    model : Layer.module_;
    optimizer : 'layout Optimizer.gradient_transformation;
  }

  let create ~model ~optimizer ?metrics ~rngs ~dtype () =
    (* Always initialize with a dummy input *)
    let params = model.Layer.init ~rngs ~dtype in
    let opt_state = optimizer.Optimizer.init params in
    { step = 0; params; opt_state; metrics; rngs; model; optimizer }

  let apply_gradients state ~grads =
    let updates, opt_state =
      state.optimizer.Optimizer.update state.opt_state state.params grads
    in
    Optimizer.apply_updates_inplace state.params updates;
    { state with opt_state; step = state.step + 1 }

  let reset_metrics state =
    Option.iter Metrics.Collection.reset state.metrics;
    state

  let update_metrics state ~predictions ~targets ?loss () =
    match state.metrics with
    | Some metrics -> (
        match loss with
        | Some l ->
            (* Use the new update_with_loss function *)
            Metrics.Collection.update_with_loss metrics ~loss:l ~predictions
              ~targets ()
        | None -> Metrics.Collection.update metrics ~predictions ~targets ())
    | None -> ()

  let compute_metrics state =
    match state.metrics with
    | Some metrics -> Metrics.Collection.compute metrics
    | None -> []

  let next_rng state =
    let rngs = Rune.Rng.split state.rngs in
    let rng1 = rngs.(0) in
    let rng2 = rngs.(1) in
    (rng1, { state with rngs = rng2 })
end

module History = struct
  type 'layout t = {
    train_loss : float list;
    train_metrics : (string * float list) list;
    val_loss : float list option;
    val_metrics : (string * float list) list option;
  }

  let final_train_loss history =
    match List.rev history.train_loss with [] -> None | hd :: _ -> Some hd

  let final_val_loss history =
    match history.val_loss with
    | None -> None
    | Some losses -> (
        match List.rev losses with [] -> None | hd :: _ -> Some hd)

  let final_train_metrics history =
    List.map
      (fun (name, values) ->
        let final_value =
          match List.rev values with [] -> 0.0 | hd :: _ -> hd
        in
        (name, final_value))
      history.train_metrics

  let final_val_metrics history =
    match history.val_metrics with
    | None -> []
    | Some metrics ->
        List.map
          (fun (name, values) ->
            let final_value =
              match List.rev values with [] -> 0.0 | hd :: _ -> hd
            in
            (name, final_value))
          metrics

  let best_train_loss history =
    match history.train_loss with
    | [] -> None
    | losses -> Some (List.fold_left min Float.max_float losses)

  let best_val_loss history =
    match history.val_loss with
    | None -> None
    | Some [] -> None
    | Some losses -> Some (List.fold_left min Float.max_float losses)

  let best_epoch ?(monitor = "val_loss") history =
    let values =
      if monitor = "val_loss" then Option.value history.val_loss ~default:[]
      else if monitor = "train_loss" then history.train_loss
      else
        (* Try to find in metrics *)
        let find_in_metrics metrics_opt =
          match metrics_opt with
          | None -> []
          | Some metrics ->
              List.find_opt (fun (name, _) -> name = monitor) metrics
              |> Option.map snd |> Option.value ~default:[]
        in
        if String.starts_with ~prefix:"val_" monitor then
          find_in_metrics history.val_metrics
        else find_in_metrics (Some history.train_metrics)
    in
    match values with
    | [] -> None
    | _ ->
        let indexed = List.mapi (fun i v -> (i, v)) values in
        let best_idx, _ =
          List.fold_left
            (fun (best_i, best_v) (i, v) ->
              if v < best_v then (i, v) else (best_i, best_v))
            (0, Float.max_float) indexed
        in
        Some best_idx
end

let train_step ~state ~x ~y ~loss_fn =
  (* Forward and backward pass *)
  let loss, grads =
    Transformations.value_and_grad
      (fun params ->
        let logits = state.State.model.apply params ~training:true x in
        loss_fn logits y)
      state.State.params
  in

  (* Update parameters *)
  let state = State.apply_gradients state ~grads in

  (* Update metrics if available *)
  let logits = state.State.model.apply state.State.params ~training:false x in
  State.update_metrics state ~predictions:logits ~targets:y ~loss ();

  (* Return updated state and loss value *)
  let loss_val = Rune.item [] loss in
  (state, loss_val)

let eval_step ~state ~x ~y ~loss_fn =
  let logits = state.State.model.apply state.State.params ~training:false x in
  let loss = loss_fn logits y in

  (* Update metrics *)
  State.update_metrics state ~predictions:logits ~targets:y ~loss ();

  Rune.item [] loss

let train_epoch ~state ~dataset ~loss_fn ?(progress = false) () =
  let state = State.reset_metrics state in
  let state_ref = ref state in
  let total_loss = ref 0. in
  let batch_count = ref 0 in
  let total_time = ref 0. in

  if progress then Printf.printf "Training: ";

  Dataset.iter
    (fun (x, y) ->
      incr batch_count;
      let step_start = Unix.gettimeofday () in
      let state', loss = train_step ~state:!state_ref ~x ~y ~loss_fn in
      let step_time = Unix.gettimeofday () -. step_start in
      total_time := !total_time +. step_time;
      state_ref := state';
      total_loss := !total_loss +. loss;
      if progress && !batch_count mod 10 = 0 then Printf.printf ".")
    batches;

  (if progress then
     let avg_step_time = !total_time /. float_of_int !batch_count *. 1000. in
     Printf.printf " done (%d steps, avg %.1fms/step)\n%!" !batch_count
       avg_step_time);

  let avg_loss = !total_loss /. float_of_int !batch_count in
  let metrics = State.compute_metrics !state_ref in
  let metric_values =
    List.map (fun (name, tensor) -> (name, Rune.item [] tensor)) metrics
  in

  (!state_ref, avg_loss, metric_values)

module Callbacks = struct
  type 'layout context = {
    epoch : int;
    state : 'layout State.t;
    history : 'layout History.t;
    train_loss : float option;
    val_loss : float option;
    train_metrics : (string * float) list;
    val_metrics : (string * float) list;
  }

  type 'layout t = {
    on_epoch_begin : 'layout context -> bool;
    on_epoch_end : 'layout context -> bool;
    on_train_begin : 'layout context -> unit;
    on_train_end : 'layout context -> unit;
  }

  let early_stopping ?(monitor = "val_loss") ?(patience = 5) ?(mode = `Min)
      ?(min_delta = 0.0) ?(baseline = None) () =
    let best_value = ref None in
    let wait = ref 0 in
    let stopped_epoch = ref 0 in

    let is_better current best =
      match mode with
      | `Min -> current < best -. min_delta
      | `Max -> current > best +. min_delta
    in

    let get_monitored_value ctx =
      if monitor = "val_loss" then ctx.val_loss
      else if monitor = "train_loss" then ctx.train_loss
      else
        (* Look in metrics *)
        let metrics =
          if String.starts_with ~prefix:"val_" monitor then ctx.val_metrics
          else ctx.train_metrics
        in
        List.find_opt (fun (name, _) -> name = monitor) metrics
        |> Option.map snd
    in

    {
      on_epoch_begin = (fun _ -> true);
      on_epoch_end =
        (fun ctx ->
          match get_monitored_value ctx with
          | None -> true (* Continue if metric not available *)
          | Some current ->
              (* Check baseline *)
              let continue =
                match baseline with
                | Some b when not (is_better current b) ->
                    stopped_epoch := ctx.epoch;
                    false
                | _ -> (
                    (* Check improvement *)
                    match !best_value with
                    | None ->
                        best_value := Some current;
                        wait := 0;
                        true
                    | Some best ->
                        if is_better current best then (
                          best_value := Some current;
                          wait := 0;
                          true)
                        else (
                          incr wait;
                          if !wait >= patience then (
                            stopped_epoch := ctx.epoch;
                            Printf.printf "\nEarly stopping at epoch %d\n"
                              ctx.epoch;
                            false)
                          else true))
              in
              continue);
      on_train_begin = (fun _ -> ());
      on_train_end = (fun _ -> ());
    }

  let model_checkpoint ~filepath ?(monitor = "val_loss") ?(mode = `Min)
      ?(save_best_only = true) ?(save_freq = `Best) () =
    let best_value = ref None in

    let is_better current best =
      match mode with `Min -> current < best | `Max -> current > best
    in

    let get_monitored_value ctx =
      if monitor = "val_loss" then ctx.val_loss
      else if monitor = "train_loss" then ctx.train_loss
      else
        let metrics =
          if String.starts_with ~prefix:"val_" monitor then ctx.val_metrics
          else ctx.train_metrics
        in
        List.find_opt (fun (name, _) -> name = monitor) metrics
        |> Option.map snd
    in

    let save_checkpoint ctx =
      let path =
        (* Replace {epoch} placeholder if present *)
        Str.global_replace (Str.regexp "{epoch}") (string_of_int ctx.epoch)
          filepath
      in
      (* Use Kaun_checkpoint module to save *)
      Checkpoint.save_params ~path ~params:ctx.state.params ();
      Printf.printf "Saved checkpoint to %s\n" path
    in

    {
      on_epoch_begin = (fun _ -> true);
      on_epoch_end =
        (fun ctx ->
          let should_save =
            match save_freq with
            | `Epoch n when ctx.epoch mod n = 0 -> true
            | `Best ->
                if save_best_only then
                  match get_monitored_value ctx with
                  | None -> false
                  | Some current -> (
                      match !best_value with
                      | None ->
                          best_value := Some current;
                          true
                      | Some best ->
                          if is_better current best then (
                            best_value := Some current;
                            true)
                          else false)
                else true
            | _ -> false
          in
          if should_save then save_checkpoint ctx;
          true);
      on_train_begin = (fun _ -> ());
      on_train_end = (fun _ -> ());
    }

  let reduce_lr_on_plateau ?(monitor = "val_loss") ?(factor = 0.1)
      ?(patience = 10) ?(mode = `Min) ?(min_delta = 0.0001) ?(cooldown = 0)
      ?(min_lr = 0.0) () =
    let best_value = ref None in
    let wait = ref 0 in
    let cooldown_counter = ref 0 in
    let current_lr = ref None in

    let is_better current best =
      match mode with
      | `Min -> current < best -. min_delta
      | `Max -> current > best +. min_delta
    in

    let get_monitored_value ctx =
      if monitor = "val_loss" then ctx.val_loss
      else if monitor = "train_loss" then ctx.train_loss
      else
        let metrics =
          if String.starts_with ~prefix:"val_" monitor then ctx.val_metrics
          else ctx.train_metrics
        in
        List.find_opt (fun (name, _) -> name = monitor) metrics
        |> Option.map snd
    in

    {
      on_epoch_begin = (fun _ -> true);
      on_epoch_end =
        (fun ctx ->
          if !cooldown_counter > 0 then (
            decr cooldown_counter;
            true)
          else
            match get_monitored_value ctx with
            | None -> true
            | Some current -> (
                match !best_value with
                | None ->
                    best_value := Some current;
                    wait := 0;
                    true
                | Some best ->
                    if is_better current best then (
                      best_value := Some current;
                      wait := 0;
                      true)
                    else (
                      incr wait;
                      if !wait >= patience then (
                        (* Reduce learning rate *)
                        let new_lr =
                          match !current_lr with
                          | None ->
                              (* Initialize with a default if not set *)
                              Printf.printf
                                "\n\
                                 Would reduce learning rate by factor %.2f \
                                 (min_lr: %.6f)\n"
                                factor min_lr;
                              None
                          | Some lr ->
                              let new_lr_value = lr *. factor in
                              if new_lr_value >= min_lr then (
                                Printf.printf
                                  "\nReducing learning rate from %.6f to %.6f\n"
                                  lr new_lr_value;
                                current_lr := Some new_lr_value;
                                Some new_lr_value)
                              else (
                                Printf.printf
                                  "\n\
                                   Learning rate %.6f already at minimum %.6f\n"
                                  lr min_lr;
                                Some lr)
                        in
                        let _ = new_lr in
                        (* Mark as used *)
                        wait := 0;
                        cooldown_counter := cooldown;
                        (* Note: Actually modifying the optimizer state would
                           require access to the optimizer transformation, which
                           we don't have here. A full implementation would need
                           to pass this back to the training loop. *)
                        true)
                      else true)));
      on_train_begin = (fun _ -> ());
      on_train_end = (fun _ -> ());
    }

  let tensorboard ~log_dir ?(update_freq = `Epoch) () =
    (* Ensure log directory exists *)
    let _ = Sys.command (Printf.sprintf "mkdir -p %s" log_dir) in
    let batch_counter = ref 0 in

    {
      on_epoch_begin =
        (fun _ ->
          batch_counter := 0;
          true);
      on_epoch_end =
        (fun ctx ->
          (* Log based on update frequency *)
          let should_log =
            match update_freq with
            | `Epoch -> true
            | `Batch n -> !batch_counter mod n = 0
          in
          if should_log then (
            (* Log metrics to file in simplified format *)
            let log_file = Filename.concat log_dir "metrics.log" in
            let oc = open_out_gen [ Open_append; Open_creat ] 0o644 log_file in
            Printf.fprintf oc "Epoch %d: " ctx.epoch;
            (match ctx.train_loss with
            | Some loss -> Printf.fprintf oc "train_loss=%.4f " loss
            | None -> ());
            List.iter
              (fun (name, value) ->
                Printf.fprintf oc "train_%s=%.4f " name value)
              ctx.train_metrics;
            (match ctx.val_loss with
            | Some loss -> Printf.fprintf oc "val_loss=%.4f " loss
            | None -> ());
            List.iter
              (fun (name, value) -> Printf.fprintf oc "val_%s=%.4f " name value)
              ctx.val_metrics;
            Printf.fprintf oc "\n";
            close_out oc);
          incr batch_counter;
          true);
      on_train_begin = (fun _ -> ());
      on_train_end = (fun _ -> ());
    }

  let custom ?(on_epoch_begin = fun _ -> true) ?(on_epoch_end = fun _ -> true)
      ?(on_train_begin = fun _ -> ()) ?(on_train_end = fun _ -> ()) () =
    { on_epoch_begin; on_epoch_end; on_train_begin; on_train_end }

  let combine callbacks =
    {
      on_epoch_begin =
        (fun ctx -> List.for_all (fun cb -> cb.on_epoch_begin ctx) callbacks);
      on_epoch_end =
        (fun ctx -> List.for_all (fun cb -> cb.on_epoch_end ctx) callbacks);
      on_train_begin =
        (fun ctx -> List.iter (fun cb -> cb.on_train_begin ctx) callbacks);
      on_train_end =
        (fun ctx -> List.iter (fun cb -> cb.on_train_end ctx) callbacks);
    }
end

let evaluate ~state ~dataset ~loss_fn ?(progress = false) () =
  let state = State.reset_metrics state in
  let total_loss = ref 0. in
  let batch_count = ref 0 in

  if progress then Printf.printf "Evaluating: ";

  Dataset.iter
    (fun (x, y) ->
      incr batch_count;
      let loss = eval_step ~state ~x ~y ~loss_fn in
      total_loss := !total_loss +. loss;
      if progress && !batch_count mod 10 = 0 then Printf.printf ".")
    dataset;

  if progress then Printf.printf " done\n%!";

  let avg_loss = !total_loss /. float_of_int !batch_count in
  let metrics = State.compute_metrics state in
  let metric_values =
    List.map (fun (name, tensor) -> (name, Rune.item [] tensor)) metrics
  in

  (avg_loss, metric_values)

let fit ~model ~optimizer ~loss_fn ?metrics ~train_data ?val_data ~epochs
    ?callbacks ?(progress = true) ~rngs ~dtype () =
  (* Create initial training state *)
  let state = State.create ~model ~optimizer ?metrics ~rngs ~dtype () in

  (* Training history *)
  let history =
    History.
      {
        train_loss = [];
        train_metrics = [];
        val_loss = None;
        val_metrics = None;
      }
  in

  let state_ref = ref state in
  let history_ref = ref history in

  (* Combine callbacks if provided *)
  let callback =
    match callbacks with
    | None -> None
    | Some cbs -> Some (Callbacks.combine cbs)
  in

  (* Call on_train_begin *)
  (match callback with
  | Some cb ->
      let ctx =
        Callbacks.
          {
            epoch = 0;
            state = !state_ref;
            history = !history_ref;
            train_loss = None;
            val_loss = None;
            train_metrics = [];
            val_metrics = [];
          }
      in
      cb.on_train_begin ctx
  | None -> ());

  (* Training loop *)
  let continue_training = ref true in
  let epoch_idx = ref 1 in

  while !epoch_idx <= epochs && !continue_training do
    let epoch = !epoch_idx in
    if progress then Printf.printf "\nEpoch %d/%d\n" epoch epochs;

    let epoch_start_time = Unix.gettimeofday () in

    (* Reset datasets at the start of each epoch, if supported *)
    Dataset.reset train_data;
    (match val_data with Some ds -> Dataset.reset ds | None -> ());

    (* Call on_epoch_begin *)
    (match callback with
    | Some cb ->
        let ctx =
          Callbacks.
            {
              epoch;
              state = !state_ref;
              history = !history_ref;
              train_loss = None;
              val_loss = None;
              train_metrics = [];
              val_metrics = [];
            }
        in
        continue_training := cb.on_epoch_begin ctx
    | None -> ());

    (* Train *)
    let state', train_loss, train_metrics =
      train_epoch ~state:!state_ref ~dataset:train_data ~loss_fn ~progress ()
    in
    state_ref := state';

    if progress then Printf.printf "  Train loss: %.4f" train_loss;
    List.iter
      (fun (name, value) ->
        if progress then Printf.printf ", %s: %.4f" name value)
      train_metrics;
    if progress then Printf.printf "\n";

    (* Update history *)
    history_ref :=
      {
        !history_ref with
        train_loss = !history_ref.train_loss @ [ train_loss ];
        train_metrics =
          (if !history_ref.train_metrics = [] then
             List.map (fun (name, value) -> (name, [ value ])) train_metrics
           else
             List.map2
               (fun (name, values) (_, new_value) ->
                 (name, values @ [ new_value ]))
               !history_ref.train_metrics train_metrics);
      };

    (* Validate if data provided *)
    (match val_data with
    | Some val_dataset ->
        let val_loss, val_metrics =
          evaluate ~state:!state_ref ~dataset:val_dataset ~loss_fn ~progress ()
        in

        if progress then Printf.printf "  Val loss: %.4f" val_loss;
        List.iter
          (fun (name, value) ->
            if progress then Printf.printf ", %s: %.4f" name value)
          val_metrics;
        if progress then Printf.printf "\n%!";

        (* Update validation history *)
        let val_loss_list =
          match !history_ref.val_loss with
          | Some l -> l @ [ val_loss ]
          | None -> [ val_loss ]
        in
        let val_metrics_list =
          match !history_ref.val_metrics with
          | Some m when m <> [] ->
              Some
                (List.map2
                   (fun (name, values) (_, new_value) ->
                     (name, values @ [ new_value ]))
                   m val_metrics)
          | _ ->
              Some
                (List.map (fun (name, value) -> (name, [ value ])) val_metrics)
        in
        history_ref :=
          {
            !history_ref with
            val_loss = Some val_loss_list;
            val_metrics = val_metrics_list;
          }
    | None -> ());

    (* Print epoch timing *)
    let epoch_time = Unix.gettimeofday () -. epoch_start_time in
    if progress then Printf.printf "  Time: %.2fs\n%!" epoch_time;

    (* Call on_epoch_end *)
    match callback with
    | Some cb ->
        let ctx =
          Callbacks.
            {
              epoch;
              state = !state_ref;
              history = !history_ref;
              train_loss = Some train_loss;
              val_loss =
                (match val_data with
                | Some _ -> (
                    match !history_ref.val_loss with
                    | Some losses -> (
                        match List.rev losses with
                        | [] -> None
                        | hd :: _ -> Some hd)
                    | None -> None)
                | None -> None);
              train_metrics;
              val_metrics =
                (match val_data with
                | Some _ -> (
                    match !history_ref.val_metrics with
                    | Some metrics ->
                        List.map
                          (fun (name, values) ->
                            match List.rev values with
                            | [] -> (name, 0.0)
                            | hd :: _ -> (name, hd))
                          metrics
                    | None -> [])
                | None -> []);
            }
        in
        if cb.on_epoch_end ctx then incr epoch_idx
        else continue_training := false
    | None -> incr epoch_idx
  done;

  (* Call on_train_end *)
  (match callback with
  | Some cb ->
      let ctx =
        Callbacks.
          {
            epoch = epochs;
            state = !state_ref;
            history = !history_ref;
            train_loss = History.final_train_loss !history_ref;
            val_loss = History.final_val_loss !history_ref;
            train_metrics = History.final_train_metrics !history_ref;
            val_metrics = History.final_val_metrics !history_ref;
          }
      in
      cb.on_train_end ctx
  | None -> ());

  (!state_ref, !history_ref)
