open Fehu
open Dqn

(* Define the observation space as a 2D grid from (0,0) to (4,4) *)
let obs_space = Space.Box.create ~low:[| 0.0; 0.0 |] ~high:[| 4.0; 4.0 |]

(* Define the action space: 4 possible moves (up, down, left, right) *)
let act_space = Space.Discrete.create 4

let grid_size = 5
let visit_counts = Array.make_matrix grid_size grid_size 0.
let episode_rewards = ref []
let early_positions = ref []
let late_positions = ref []
let record_positions = ref []
let step_count = ref 0

let env_rng = Rune.Rng.key 123

(* Create the environment with reset and step functions that depend on agent's actions *)
let env : ((float, Rune.float32_elt) Rune.t, (int32, Rune.int32_elt) Rune.t, unit) Env.t =
  Env.create ~rng:env_rng ~observation_space:obs_space ~action_space:act_space
    ~reset:(fun _env ?options:_ () ->
      step_count := 0;
      record_positions := [];
      let obs = Rune.create Rune.float32 [| 2 |] [| 0.0; 0.0 |] in
      record_positions := [obs];
      (obs, Info.empty))
    ~step:(fun _env action ->
  step_count := !step_count + 1;
  let arr =
  match !record_positions with
  | [] -> [| 0.0; 0.0 |]
  | obs::_ ->
    let a = Rune.to_array obs in
    Printf.eprintf "DEBUG: arr length = %d\n%!" (Array.length a);
    if Array.length a = 2 then a else [| 0.0; 0.0 |]
in
if Array.length arr <> 2 then (
  Printf.eprintf "ERROR: arr length = %d\n%!" (Array.length arr);
  exit 1
);
let x = int_of_float arr.(0) in
let y = int_of_float arr.(1) in
  let (new_x, new_y) =
    match Int32.to_int (Rune.item [] action) with
    | 0 -> (x, min (y + 1) (grid_size - 1)) (* up *)
    | 1 -> (x, max (y - 1) 0)              (* down *)
    | 2 -> (max (x - 1) 0, y)              (* left *)
    | 3 -> (min (x + 1) (grid_size - 1), y) (* right *)
    | _ -> (x, y)
  in
  let obs = Rune.create Rune.float32 [|2|] [| float_of_int new_x; float_of_int new_y |] in
  record_positions := obs :: !record_positions;
  if new_x >= 0 && new_x < grid_size && new_y >= 0 && new_y < grid_size then
    visit_counts.(new_x).(new_y) <- visit_counts.(new_x).(new_y) +. 1.;
  let terminated = !step_count >= 10 || (new_x = 4 && new_y = 4) in
  Env.transition ~observation:obs ~reward:(if new_x = 4 && new_y = 4 then 10. else -0.1) ~terminated ()
)
    ()

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
let config = Dqn.{ default_config with batch_size = 4; buffer_capacity = 500 }
let agent = Dqn.create ~q_network:q_net ~n_actions:4 ~rng config

let episode_counter = ref 0

(* Increase total_timesteps to 100 for 10 episodes *)
let _agent =
  Dqn.learn agent ~env ~total_timesteps:100
    ~callback:(fun ~episode ~metrics ->
      incr episode_counter;
      episode_rewards := metrics.episode_return :: !episode_rewards;
      if metrics.episode_length > 0 then begin
        if !episode_counter = 1 then early_positions := List.rev !record_positions;
        if !episode_counter = 10 then late_positions := List.rev !record_positions;
        record_positions := [];  (* Reset only after saving positions *)
      end;
      Printf.printf "Episode %d: Reward = %.2f\n%!" episode metrics.episode_return;
      true)
    ()

let () =
  (* After training, ensure late_positions is not empty *)
  if !late_positions = [] && !record_positions <> [] then
    late_positions := List.rev !record_positions;

  let out_dir = "/Users/mac/Desktop/Outreachy/raven/fehu/demos/" in

  let oc = open_out (out_dir ^ "episode_rewards.txt") in
  List.iter (fun r -> Printf.fprintf oc "%f\n" r) (List.rev !episode_rewards);
  close_out oc;

  let save_positions positions filename =
  let oc = open_out (out_dir ^ filename) in
  List.iter (fun obs ->
    let arr = Rune.to_array obs in
    Printf.fprintf oc "%f %f\n" (arr.(0) +. 0.5) (arr.(1) +. 0.5)
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