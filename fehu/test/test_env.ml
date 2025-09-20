open Fehu

let make_simple_env ~rng () =
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let act_space = Space.Discrete.create 2 in
  let state = ref 0.0 in
  let reset _env ?options:_ () =
    state := 5.0;
    let obs = Rune.create Rune.float32 [| 1 |] [| !state |] in
    (obs, Info.empty)
  in
  let step _env action =
    let action_val =
      let arr : Int32.t array = Rune.to_array (Rune.reshape [| 1 |] action) in
      Int32.to_int arr.(0)
    in
    state := !state +. if action_val = 0 then -1.0 else 1.0;
    let terminated = !state <= 0.0 || !state >= 10.0 in
    let obs = Rune.create Rune.float32 [| 1 |] [| !state |] in
    Env.transition ~observation:obs ~reward:1.0 ~terminated ~truncated:false ()
  in
  Env.create ~id:"Simple-v0" ~rng ~observation_space:obs_space
    ~action_space:act_space ~reset ~step ()

let test_env_creation () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  match Env.id env with
  | Some id -> Alcotest.(check string) "env id" "Simple-v0" id
  | None -> Alcotest.fail "expected env id"

let test_env_reset () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let obs, info = Env.reset env () in
  let shape = Rune.shape obs in
  Alcotest.(check (array int)) "reset obs shape" [| 1 |] shape;
  let arr : float array = Rune.to_array (Rune.reshape [| 1 |] obs) in
  Alcotest.(check (float 0.01)) "reset obs value" 5.0 arr.(0);
  Alcotest.(check bool) "reset info empty" true (Info.is_empty info)

let test_env_step () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let transition = Env.step env action in
  let shape = Rune.shape transition.Env.observation in
  Alcotest.(check (array int)) "step obs shape" [| 1 |] shape;
  Alcotest.(check (float 0.01)) "step reward" 1.0 transition.Env.reward;
  Alcotest.(check bool)
    "step not terminated initially" false transition.Env.terminated

let test_env_episode_termination () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let _, _ = Env.reset env () in
  let action_up = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let rec run_to_termination count =
    if count > 20 then Alcotest.fail "episode did not terminate"
    else
      let transition = Env.step env action_up in
      if transition.Env.terminated then count else run_to_termination (count + 1)
  in
  let steps = run_to_termination 0 in
  Alcotest.(check bool) "episode terminates" true (steps <= 10)

let test_env_metadata () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let metadata = Env.metadata env in
  Alcotest.(check pass) "metadata exists" () ();
  let updated =
    metadata
    |> Metadata.with_description (Some "Test")
    |> Metadata.add_author "Alice"
  in
  Env.set_metadata env updated;
  let new_metadata = Env.metadata env in
  Alcotest.(check (option string))
    "metadata updated" (Some "Test") new_metadata.description

let test_env_rng () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let rng1 = Env.take_rng env in
  let rng2 = Env.rng env in
  Alcotest.(check bool) "rng updated after take" true (rng1 <> rng2)

let test_env_split_rng () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let rngs = Env.split_rng env ~n:5 in
  Alcotest.(check int) "split produces n rngs" 5 (Array.length rngs)

let test_env_spaces () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let obs_space = Env.observation_space env in
  let act_space = Env.action_space env in
  let obs_shape = Space.shape obs_space in
  let act_shape = Space.shape act_space in
  Alcotest.(check (option (array int)))
    "obs space shape" (Some [| 1 |]) obs_shape;
  Alcotest.(check (option (array int))) "act space shape" None act_shape

let test_env_close () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  Alcotest.(check bool) "env initially open" false (Env.closed env);
  Env.close env;
  Alcotest.(check bool) "env closed after close" true (Env.closed env)

let test_env_render () =
  let rng = Rune.Rng.key 42 in
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0 |] in
  let act_space = Space.Discrete.create 2 in
  let env =
    Env.create ~rng ~observation_space:obs_space ~action_space:act_space
      ~reset:(fun _ ?options:_ () ->
        (Rune.create Rune.float32 [| 1 |] [| 0.5 |], Info.empty))
      ~step:(fun _ _action ->
        Env.transition
          ~observation:(Rune.create Rune.float32 [| 1 |] [| 0.5 |])
          ~reward:0.0 ())
      ~render:(fun _ -> Some "test render")
      ()
  in
  match Env.render env with
  | Some str -> Alcotest.(check string) "render output" "test render" str
  | None -> Alcotest.fail "expected render output"

let test_transition_builder () =
  let obs = Rune.create Rune.float32 [| 1 |] [| 1.0 |] in
  let t1 = Env.transition ~observation:obs () in
  Alcotest.(check (float 0.01)) "default reward" 0.0 t1.reward;
  Alcotest.(check bool) "default terminated" false t1.terminated;
  Alcotest.(check bool) "default truncated" false t1.truncated;
  let t2 = Env.transition ~observation:obs ~reward:5.0 ~terminated:true () in
  Alcotest.(check (float 0.01)) "custom reward" 5.0 t2.reward;
  Alcotest.(check bool) "custom terminated" true t2.terminated

let () =
  let open Alcotest in
  run "Env"
    [
      ( "Creation",
        [
          test_case "create env" `Quick test_env_creation;
          test_case "env spaces" `Quick test_env_spaces;
        ] );
      ( "Lifecycle",
        [
          test_case "reset env" `Quick test_env_reset;
          test_case "step env" `Quick test_env_step;
          test_case "episode termination" `Quick test_env_episode_termination;
          test_case "close env" `Quick test_env_close;
        ] );
      ( "Metadata",
        [
          test_case "metadata operations" `Quick test_env_metadata;
          test_case "rng operations" `Quick test_env_rng;
          test_case "split rng" `Quick test_env_split_rng;
        ] );
      ( "Utilities",
        [
          test_case "render" `Quick test_env_render;
          test_case "transition builder" `Quick test_transition_builder;
        ] );
    ]
