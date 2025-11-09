open Fehu
module Overlay = Overlay
module Video = Wrapper_video
module Sink = Sink

let push = Sink.push

let push_many sink frames =
  Array.iter (fun frame -> Sink.push sink frame) frames

let expect_image_frame source = function
  | Render.Image image -> image
  | Render.None ->
      invalid_arg
        (Printf.sprintf
           "%s: render returned None; ensure render_mode returns frames" source)
  | Render.Text _ ->
      invalid_arg
        (Printf.sprintf "%s: render produced ANSI text; expected image" source)
  | Render.Svg _ ->
      invalid_arg
        (Printf.sprintf "%s: render produced SVG; expected raster image" source)

let default_overlay overlay = Option.value overlay ~default:Overlay.identity

let record_rollout ~env ~policy ~steps ?overlay ~sink () =
  if steps <= 0 then invalid_arg "record_rollout: steps must be positive";
  let overlay = default_overlay overlay in
  let action_space = Env.action_space env in
  let observation, _info = Env.reset env () in
  let current_observation = ref observation in
  let episode_idx = ref 0 in
  let step_idx = ref 0 in
  let finalize () = Sink.close sink in
  Fun.protect ~finally:finalize (fun () ->
      for i = 0 to steps - 1 do
        let action, log_prob, value = policy !current_observation in
        let transition = Env.step env action in
        let frame =
          match Env.render env with
          | Some frame -> frame
          | None ->
              invalid_arg
                "record_rollout: Env.render returned None; choose `Rgb_array` \
                 render mode"
        in
        let image = expect_image_frame "record_rollout" frame in
        let ctx =
          {
            Overlay.step_idx = !step_idx;
            episode_idx = !episode_idx;
            info = transition.info;
            action = Some (Space.pack action_space action);
            value;
            log_prob;
            reward = transition.reward;
            done_ = transition.terminated || transition.truncated;
          }
        in
        let image = overlay image ctx in
        Sink.push sink (Render.Image image);
        incr step_idx;
        if ctx.done_ then (
          incr episode_idx;
          let obs, _info_reset = Env.reset env () in
          current_observation := obs;
          if i < steps - 1 then
            match Env.render env with
            | None -> ()
            | Some frame ->
                let image = expect_image_frame "record_rollout(reset)" frame in
                let reset_ctx =
                  {
                    Overlay.step_idx = !step_idx;
                    episode_idx = !episode_idx;
                    info = Info.empty;
                    action = None;
                    value = None;
                    log_prob = None;
                    reward = 0.;
                    done_ = false;
                  }
                in
                let image = overlay image reset_ctx in
                Sink.push sink (Render.Image image))
        else current_observation := transition.observation
      done)

let take_first n list =
  let rec aux count acc = function
    | _ when count = n -> List.rev acc
    | [] -> List.rev acc
    | x :: xs -> aux (count + 1) (x :: acc) xs
  in
  aux 0 [] list

let mean_and_std floats =
  match floats with
  | [] -> (0., 0.)
  | _ ->
      let n = float_of_int (List.length floats) in
      let sum = List.fold_left ( +. ) 0. floats in
      let mean = sum /. n in
      let variance =
        if List.length floats = 1 then 0.
        else
          List.fold_left
            (fun acc value ->
              let diff = value -. mean in
              acc +. (diff *. diff))
            0. floats
          /. n
      in
      (mean, Float.sqrt variance)

let mean_int ints =
  match ints with
  | [] -> 0.
  | _ ->
      let n = float_of_int (List.length ints) in
      let sum = List.fold_left ( + ) 0 ints in
      float_of_int sum /. n

let record_evaluation ~vec_env ~policy ~n_episodes ?max_steps ~layout ?overlay
    ~sink () =
  if n_episodes <= 0 then invalid_arg "record_evaluation: n_episodes > 0";
  let overlay = default_overlay overlay in
  let observations, _infos = Vector_env.reset vec_env () in
  let observations = ref observations in
  let num_envs = Vector_env.num_envs vec_env in
  let envs = Vector_env.envs vec_env in
  let action_space =
    if Array.length envs = 0 then
      invalid_arg "record_evaluation: empty vector environment"
    else Env.action_space envs.(0)
  in
  let returns = Array.make num_envs 0. in
  let lengths = Array.make num_envs 0 in
  let episode_indices = Array.make num_envs 0 in
  let completed_returns = ref [] in
  let completed_lengths = ref [] in
  let total_episodes = ref 0 in
  let step_idx = ref 0 in
  let finalize () = Sink.close sink in
  Fun.protect ~finally:finalize (fun () ->
      let rec loop () =
        if !total_episodes >= n_episodes then ()
        else
          match max_steps with
          | Some limit when !step_idx >= limit -> ()
          | _ ->
              let actions, log_probs_opt, values_opt = policy !observations in
              if Array.length actions <> num_envs then
                invalid_arg
                  "record_evaluation: policy returned mismatched action array";
              (match log_probs_opt with
              | Some arr when Array.length arr <> num_envs ->
                  invalid_arg
                    "record_evaluation: policy returned mismatched log_probs"
              | _ -> ());
              (match values_opt with
              | Some arr when Array.length arr <> num_envs ->
                  invalid_arg
                    "record_evaluation: policy returned mismatched values"
              | _ -> ());
              let step = Vector_env.step vec_env actions in
              let frames_opt = Vector_env.render vec_env in
              if Array.length frames_opt <> num_envs then
                invalid_arg
                  "record_evaluation: render frame count mismatch with envs";
              let frames =
                Array.map
                  (function
                    | None ->
                        invalid_arg
                          "record_evaluation: environment did not return frames"
                    | Some frame -> expect_image_frame "record_evaluation" frame)
                  frames_opt
              in
              for idx = 0 to num_envs - 1 do
                returns.(idx) <- returns.(idx) +. step.rewards.(idx);
                lengths.(idx) <- lengths.(idx) + 1;
                let done_flag =
                  step.terminations.(idx) || step.truncations.(idx)
                in
                let action_value = Space.pack action_space actions.(idx) in
                let value = Option.map (fun arr -> arr.(idx)) values_opt in
                let log_prob =
                  Option.map (fun arr -> arr.(idx)) log_probs_opt
                in
                let ctx =
                  {
                    Overlay.step_idx = !step_idx;
                    episode_idx = episode_indices.(idx);
                    info = step.infos.(idx);
                    action = Some action_value;
                    value;
                    log_prob;
                    reward = step.rewards.(idx);
                    done_ = done_flag;
                  }
                in
                frames.(idx) <- overlay frames.(idx) ctx;
                if done_flag then (
                  completed_returns := returns.(idx) :: !completed_returns;
                  completed_lengths := lengths.(idx) :: !completed_lengths;
                  returns.(idx) <- 0.;
                  lengths.(idx) <- 0;
                  episode_indices.(idx) <- episode_indices.(idx) + 1;
                  incr total_episodes)
              done;
              incr step_idx;
              let composed =
                match layout with
                | `Single_each ->
                    Utils.compose_grid ~rows:1 ~cols:num_envs frames
                | `NxM_grid (rows, cols) ->
                    if rows * cols <> num_envs then
                      invalid_arg
                        "record_evaluation: grid layout must cover all \
                         environments";
                    Utils.compose_grid ~rows ~cols frames
              in
              Sink.push sink (Render.Image composed);
              observations := step.observations;
              loop ()
      in
      loop ();
      let rewards = take_first n_episodes !completed_returns |> List.rev in
      let lengths = take_first n_episodes !completed_lengths |> List.rev in
      let mean_reward, std_reward = mean_and_std rewards in
      let mean_length = mean_int lengths in
      let open Training in
      { mean_reward; std_reward; mean_length; n_episodes = List.length rewards })
