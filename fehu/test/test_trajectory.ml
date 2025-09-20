open Fehu

let test_create () =
  let observations =
    [|
      Rune.create Rune.float32 [| 1 |] [| 0.0 |];
      Rune.create Rune.float32 [| 1 |] [| 1.0 |];
      Rune.create Rune.float32 [| 1 |] [| 2.0 |];
    |]
  in
  let actions =
    [|
      Rune.create Rune.int32 [| 1 |] [| 0l |];
      Rune.create Rune.int32 [| 1 |] [| 1l |];
      Rune.create Rune.int32 [| 1 |] [| 0l |];
    |]
  in
  let rewards = [| 1.0; 0.5; 1.0 |] in
  let terminateds = [| false; false; true |] in
  let truncateds = [| false; false; false |] in

  let trajectory =
    Trajectory.create ~observations ~actions ~rewards ~terminateds ~truncateds
      ()
  in

  Alcotest.(check int) "length" 3 (Trajectory.length trajectory)

let test_create_with_log_probs_values () =
  let observations =
    [|
      Rune.create Rune.float32 [| 1 |] [| 0.0 |];
      Rune.create Rune.float32 [| 1 |] [| 1.0 |];
    |]
  in
  let actions =
    [|
      Rune.create Rune.int32 [| 1 |] [| 0l |];
      Rune.create Rune.int32 [| 1 |] [| 1l |];
    |]
  in
  let rewards = [| 1.0; 0.5 |] in
  let terminateds = [| false; true |] in
  let truncateds = [| false; false |] in
  let log_probs = [| -0.5; -0.3 |] in
  let values = [| 0.8; 0.9 |] in

  let trajectory =
    Trajectory.create ~observations ~actions ~rewards ~terminateds ~truncateds
      ~log_probs ~values ()
  in

  Alcotest.(check int) "length" 2 (Trajectory.length trajectory)

let test_concat () =
  let make_trajectory n =
    let observations =
      Array.init n (fun i ->
          Rune.create Rune.float32 [| 1 |] [| float_of_int i |])
    in
    let actions =
      Array.init n (fun i ->
          Rune.create Rune.int32 [| 1 |] [| Int32.of_int i |])
    in
    let rewards = Array.init n (fun _ -> 1.0) in
    let terminateds = Array.init n (fun _ -> false) in
    let truncateds = Array.init n (fun _ -> false) in
    Trajectory.create ~observations ~actions ~rewards ~terminateds ~truncateds
      ()
  in

  let t1 = make_trajectory 3 in
  let t2 = make_trajectory 2 in
  let t3 = make_trajectory 4 in

  let concatenated = Trajectory.concat [ t1; t2; t3 ] in

  Alcotest.(check int) "concatenated length" 9 (Trajectory.length concatenated)

let test_collect () =
  (* Create a simple environment *)
  let rng = Rune.Rng.key 42 in
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let act_space = Space.Discrete.create 2 in

  let step_count = ref 0 in

  let env =
    Env.create ~rng ~observation_space:obs_space ~action_space:act_space
      ~reset:(fun _env ?options:_ () ->
        step_count := 0;
        let obs = Rune.create Rune.float32 [| 1 |] [| 0.0 |] in
        (obs, Info.empty))
      ~step:(fun _env _action ->
        step_count := !step_count + 1;
        let terminated = !step_count >= 5 in
        let obs =
          Rune.create Rune.float32 [| 1 |] [| float_of_int !step_count |]
        in
        Env.transition ~observation:obs ~reward:1.0 ~terminated ())
      ()
  in

  (* Policy that returns action with optional log_prob and value *)
  let policy _obs =
    let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
    let log_prob = Some (-0.5) in
    let value = Some 0.8 in
    (action, log_prob, value)
  in

  let trajectory = Trajectory.collect env ~policy ~n_steps:10 in

  Alcotest.(check int) "collected 10 steps" 10 (Trajectory.length trajectory)

let test_collect_episodes () =
  let rng = Rune.Rng.key 42 in
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let act_space = Space.Discrete.create 2 in

  let step_count = ref 0 in

  let env =
    Env.create ~rng ~observation_space:obs_space ~action_space:act_space
      ~reset:(fun _env ?options:_ () ->
        step_count := 0;
        let obs = Rune.create Rune.float32 [| 1 |] [| 0.0 |] in
        (obs, Info.empty))
      ~step:(fun _env _action ->
        step_count := !step_count + 1;
        let terminated = !step_count >= 3 in
        let obs =
          Rune.create Rune.float32 [| 1 |] [| float_of_int !step_count |]
        in
        Env.transition ~observation:obs ~reward:1.0 ~terminated ())
      ()
  in

  (* Simple policy *)
  let policy _obs =
    let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
    (action, None, None)
  in

  let episodes = Trajectory.collect_episodes env ~policy ~n_episodes:3 () in

  Alcotest.(check int) "collected 3 episodes" 3 (List.length episodes);

  (* Each episode should have 3 steps *)
  List.iter
    (fun episode ->
      Alcotest.(check int) "episode length" 3 (Trajectory.length episode))
    episodes

let test_collect_episodes_with_max_steps () =
  let rng = Rune.Rng.key 42 in
  let obs_space = Space.Box.create ~low:[| -1.0 |] ~high:[| 1.0 |] in
  let act_space = Space.Discrete.create 2 in

  let env =
    Env.create ~rng ~observation_space:obs_space ~action_space:act_space
      ~reset:(fun _env ?options:_ () ->
        let obs = Rune.create Rune.float32 [| 1 |] [| 0.0 |] in
        (obs, Info.empty))
      ~step:(fun _env _action ->
        (* Never terminates *)
        let obs = Rune.create Rune.float32 [| 1 |] [| 0.0 |] in
        Env.transition ~observation:obs ~reward:1.0 ~terminated:false ())
      ()
  in

  let policy _obs =
    let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
    (action, None, None)
  in

  let episodes =
    Trajectory.collect_episodes env ~policy ~n_episodes:2 ~max_steps:5 ()
  in

  Alcotest.(check int) "collected 2 episodes" 2 (List.length episodes);

  (* Each episode should be truncated at 5 steps *)
  List.iter
    (fun episode ->
      Alcotest.(check int)
        "episode truncated at max_steps" 5
        (Trajectory.length episode))
    episodes

let () =
  let open Alcotest in
  run "Trajectory"
    [
      ( "Creation",
        [
          test_case "create trajectory" `Quick test_create;
          test_case "create with log_probs and values" `Quick
            test_create_with_log_probs_values;
          test_case "concatenate trajectories" `Quick test_concat;
        ] );
      ( "Collection",
        [
          test_case "collect steps" `Quick test_collect;
          test_case "collect episodes" `Quick test_collect_episodes;
          test_case "collect episodes with max_steps" `Quick
            test_collect_episodes_with_max_steps;
        ] );
    ]
