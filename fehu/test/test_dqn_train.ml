open Fehu
open Fehu_algorithms.Dqn

let records = ref []  (* CSV lines collected in reverse order *)

let callback metrics =
  Printf.printf "Callback called: episode_length=%s episode_return=%s\n%!"
    (match metrics.episode_length with Some l -> string_of_int l | None -> "None")
    (match metrics.episode_return with Some r -> string_of_float r | None -> "None");
  match metrics.episode_return with
  | Some r ->
      let len = match metrics.episode_length with Some l -> l | None -> 0 in
      let loss = metrics.loss in
      let eps = metrics.epsilon in
      let avgq = metrics.avg_q_value in
      Printf.printf "Episode finished: reward=%f, length=%d, loss=%f, epsilon=%f, avg_q=%f\n%!" r len loss eps avgq;
      let line = Printf.sprintf "%f,%d,%f,%f,%f\n" r len loss eps avgq in
      records := line :: !records;
      `Continue
  | None -> `Continue

(* Define environment *)
let obs_space = Space.Box.create ~low:[| 0.0; 0.0 |] ~high:[| 4.0; 4.0 |]
let act_space = Space.Discrete.create 4
let env_rng = Rune.Rng.key 123
let step_count = ref 0
let env =
  Env.create ~rng:env_rng ~observation_space:obs_space ~action_space:act_space
    ~reset:(fun _env ?options:_ () ->
      step_count := 0;
      let obs = Rune.create Rune.float32 [| 2 |] [| 0.0; 0.0 |] in
      (obs, Info.empty))
    ~step:(fun _env action ->
      let action_arr = Rune.to_array action in
      let action_int = Int32.to_int action_arr.(0) in
      Printf.printf "Step: action=%d, step_count=%d\n%!" action_int !step_count;
      step_count := !step_count + 1;
      let terminated = !step_count >= 10 in
      let obs =
        Rune.create Rune.float32 [| 2 |]
          [| float_of_int (!step_count mod 5); float_of_int (!step_count / 5) |]
      in
      Env.transition ~observation:obs ~reward:1.0 ~terminated ())
    ()

(* Define Q-network *)
let rng = Rune.Rng.key 42
let q_network =
  Kaun.Layer.sequential
    [
      Kaun.Layer.linear ~in_features:2 ~out_features:8 ();
      Kaun.Layer.relu ();
      Kaun.Layer.linear ~in_features:8 ~out_features:4 ();
    ]

(* Define config *)
let config = { default_config with batch_size = 4; buffer_capacity = 50 }

let () =
  let _params, _state =
    train ~env ~q_network ~rng ~config ~total_timesteps:10000 ~callback ()
  in
  let out = "fehu/demos/metrics.csv" in
  let oc = open_out out in
  output_string oc "episode_return,episode_length,loss,epsilon,avg_q\n";
  List.iter (fun s -> output_string oc s) (List.rev !records);
  close_out oc