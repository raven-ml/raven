open Fehu
open Windtrap

let make_test_env ?(max_steps = 100) () =
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let act_space = Space.Discrete.create 2 in
  let state = ref 5.0 in
  let steps = ref 0 in
  let reset _env ?options:_ () =
    state := 5.0;
    steps := 0;
    (Nx.create Nx.float32 [| 1 |] [| !state |], Info.empty)
  in
  let step _env action =
    let a : Int32.t array = Nx.to_array (Nx.reshape [| 1 |] action) in
    state := !state +. if Int32.to_int a.(0) = 0 then -1.0 else 1.0;
    incr steps;
    let terminated = !state <= 0.0 || !state >= 10.0 in
    let truncated = (not terminated) && !steps >= max_steps in
    Env.step_result
      ~observation:(Nx.create Nx.float32 [| 1 |] [| !state |])
      ~reward:1.0 ~terminated ~truncated ()
  in
  Env.create ~id:"Test-v0" ~observation_space:obs_space ~action_space:act_space
    ~reset ~step ()

let make_envs n = List.init n (fun _ -> make_test_env ())

(* Creation *)

let test_create_num_envs () =
  let venv = Vec_env.create (make_envs 3) in
  equal ~msg:"num_envs" int 3 (Vec_env.num_envs venv)

let test_spaces_match_first_env () =
  let envs = make_envs 3 in
  let venv = Vec_env.create envs in
  let obs_shape = Space.shape (Vec_env.observation_space venv) in
  let act_shape = Space.shape (Vec_env.action_space venv) in
  let first_obs = Space.shape (Env.observation_space (List.hd envs)) in
  let first_act = Space.shape (Env.action_space (List.hd envs)) in
  equal ~msg:"obs space shape" (option (array int)) first_obs obs_shape;
  equal ~msg:"act space shape" (option (array int)) first_act act_shape

let test_empty_list_raises () =
  raises_invalid_arg "Vec_env.create: env list must not be empty" (fun () ->
      ignore (Vec_env.create []))

let test_incompatible_spaces_raises () =
  let obs1 = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let act = Space.Discrete.create 2 in
  let obs2 = Space.Box.create ~low:[| 0.0 |] ~high:[| 5.0 |] in
  let make_env obs =
    let reset _env ?options:_ () =
      (Nx.create Nx.float32 [| 1 |] [| 0.0 |], Info.empty)
    in
    let step _env _action =
      Env.step_result ~observation:(Nx.create Nx.float32 [| 1 |] [| 0.0 |]) ()
    in
    Env.create ~observation_space:obs ~action_space:act ~reset ~step ()
  in
  let e1 = make_env obs1 in
  let e2 = make_env obs2 in
  raises_match ~msg:"incompatible spaces raises"
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Vec_env.create [ e1; e2 ]))

(* Reset *)

let test_reset_obs_length () =
  let venv = Vec_env.create (make_envs 3) in
  let obs, _infos = Vec_env.reset venv () in
  equal ~msg:"obs array length" int 3 (Array.length obs)

let test_reset_infos_length () =
  let venv = Vec_env.create (make_envs 3) in
  let _obs, infos = Vec_env.reset venv () in
  equal ~msg:"infos array length" int 3 (Array.length infos)

(* Step *)

let test_step_result_lengths () =
  let venv = Vec_env.create (make_envs 3) in
  let _obs, _infos = Vec_env.reset venv () in
  let action = Nx.create Nx.int32 [| 1 |] [| 1l |] in
  let actions = Array.make 3 action in
  let s = Vec_env.step venv actions in
  equal ~msg:"observations length" int 3 (Array.length s.observations);
  equal ~msg:"rewards length" int 3 (Array.length s.rewards);
  equal ~msg:"terminated length" int 3 (Array.length s.terminated);
  equal ~msg:"truncated length" int 3 (Array.length s.truncated);
  equal ~msg:"infos length" int 3 (Array.length s.infos)

let test_wrong_action_count_raises () =
  let venv = Vec_env.create (make_envs 3) in
  let _obs, _infos = Vec_env.reset venv () in
  let action = Nx.create Nx.int32 [| 1 |] [| 1l |] in
  let actions = Array.make 2 action in
  raises_invalid_arg "Vec_env.step: expected 3 actions, got 2" (fun () ->
      ignore (Vec_env.step venv actions))

let test_autoreset_final_observation () =
  let env = make_test_env ~max_steps:3 () in
  let venv = Vec_env.create [ env ] in
  let _obs, _infos = Vec_env.reset venv () in
  let right = Nx.create Nx.int32 [| 1 |] [| 1l |] in
  let actions = [| right |] in
  (* Step until truncated at max_steps=3 *)
  let s1 = Vec_env.step venv actions in
  is_false ~msg:"not done after step 1" s1.truncated.(0);
  let s2 = Vec_env.step venv actions in
  is_false ~msg:"not done after step 2" s2.truncated.(0);
  let s3 = Vec_env.step venv actions in
  is_true ~msg:"truncated after step 3" s3.truncated.(0);
  (* After autoreset, info should have final_observation *)
  is_some ~msg:"final_observation key present"
    (Info.find "final_observation" s3.infos.(0));
  (* Observation should be from reset (5.0), not terminal *)
  let arr : float array =
    Nx.to_array (Nx.reshape [| 1 |] s3.observations.(0))
  in
  equal ~msg:"obs is from reset" (float 1e-6) 5.0 arr.(0)

(* Close *)

let test_close_all_envs () =
  let envs = make_envs 3 in
  let venv = Vec_env.create envs in
  Vec_env.close venv;
  List.iter (fun env -> is_true ~msg:"env is closed" (Env.closed env)) envs

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  run "Fehu.Vec_env"
    [
      group "creation"
        [
          test "num_envs" test_create_num_envs;
          test "spaces match first env" test_spaces_match_first_env;
          test "empty list raises" test_empty_list_raises;
          test "incompatible spaces raises" test_incompatible_spaces_raises;
        ];
      group "reset"
        [
          test "observations length" test_reset_obs_length;
          test "infos length" test_reset_infos_length;
        ];
      group "step"
        [
          test "result array lengths" test_step_result_lengths;
          test "wrong action count raises" test_wrong_action_count_raises;
          test "autoreset with final_observation"
            test_autoreset_final_observation;
        ];
      group "close" [ test "closes all inner envs" test_close_all_envs ];
    ]
