open Fehu

(* Define the observation space as a 2D grid from (0,0) to (4,4) *)
let obs_space = Space.Box.create ~low:[| 0.0; 0.0 |] ~high:[| 4.0; 4.0 |]

(* Define the action space: 4 possible moves (up, down, left, right) *)
let act_space = Space.Discrete.create 4

let env_rng = Rune.Rng.key 123
let step_count = ref 0

(* Create the environment with reset and step functions *)
let env =
  Env.create ~rng:env_rng ~observation_space:obs_space ~action_space:act_space
    ~reset:(fun _env ?options:_ () ->
      step_count := 0;
      let obs = Rune.create Rune.float32 [| 2 |] [| 0.0; 0.0 |] in
      (obs, Info.empty))
    ~step:(fun _env action ->
      step_count := !step_count + 1;
      let terminated = !step_count >= 10 in
      let obs =
        Rune.create Rune.float32 [| 2 |] [| float_of_int (!step_count mod 5); float_of_int (!step_count / 5) |]
      in
      Env.transition ~observation:obs ~reward:1.0 ~terminated ())
    ()

module Dqn = Dqn

let rng = Rune.Rng.key 42

(* Build a simple neural network for Q-value approximation *)
let q_net =
  Kaun.Layer.sequential
    [
      Kaun.Layer.linear ~in_features:2 ~out_features:8 ();
      Kaun.Layer.relu ();
      Kaun.Layer.linear ~in_features:8 ~out_features:4 ();
    ]

(* Configure the agent: batch size, buffer capacity, etc. *)
let config = Dqn.{ default_config with batch_size = 4; buffer_capacity = 50 }
let agent = Dqn.create ~q_network:q_net ~n_actions:4 ~rng config

(* Data collection for visualization *)
let episode_rewards = ref []
let early_positions = ref []
let late_positions = ref []
let record_positions = ref []

(* State visitation heatmap *)
let grid_size = 5
let visit_counts = Array.make_matrix grid_size grid_size 0.

let _agent =
  Dqn.learn agent ~env ~total_timesteps:100
    ~callback:(fun ~episode ~metrics ->
      (* Reset position tracker at the start of each episode *)
      if metrics.episode_step = 0 then record_positions := [];
      (* Track agent position at each step *)
      record_positions := metrics.observation :: !record_positions;
      (* Update state visitation heatmap *)
      let arr = Rune.to_array metrics.observation in
      let x = int_of_float arr.(0) in
      let y = int_of_float arr.(1) in
      if x >= 0 && x < grid_size && y >= 0 && y < grid_size then
        visit_counts.(x).(y) <- visit_counts.(x).(y) +. 1.;
      (* Save positions for early and late episodes *)
      if metrics.terminated then begin
        if episode = 1 then early_positions := List.rev !record_positions;
        if episode = 100 then late_positions := List.rev !record_positions;
      end;
      episode_rewards := metrics.episode_return :: !episode_rewards;
      Printf.printf "Episode %d: Reward = %.2f\n%!" episode metrics.episode_return;
      true)
    ()

(* Save rewards and positions for plotting *)
let () =
  let out_dir = "/Users/mac/Desktop/Outreachy/raven/fehu/demos/" in

  let oc = open_out (out_dir ^ "episode_rewards.txt") in
  List.iter (fun r -> Printf.fprintf oc "%f\n" r) (List.rev !episode_rewards);
  close_out oc;

  let save_positions positions filename =
    let oc = open_out (out_dir ^ filename) in
    List.iter (fun obs ->
      let arr = Rune.to_array obs in
      Printf.fprintf oc "%f %f\n" arr.(0) arr.(1)
    ) positions;
    close_out oc
  in
  save_positions !early_positions "early_positions.txt";
  save_positions !late_positions "late_positions.txt";

  (* Save state visitation heatmap *)
  let oc_heatmap = open_out (out_dir ^ "visit_counts.txt") in
  Array.iter (fun row ->
    Array.iter (fun v -> Printf.fprintf oc_heatmap "%f " v) row;
    Printf.fprintf oc_heatmap "\n"
  ) visit_counts;
  close_out oc_heatmap