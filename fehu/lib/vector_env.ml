open Errors

type autoreset_mode = Next_step | Disabled

type ('obs, 'act, 'render) step = {
  observations : 'obs array;
  rewards : float array;
  terminations : bool array;
  truncations : bool array;
  infos : Info.t array;
}

type ('obs, 'act, 'render) t = {
  envs : ('obs, 'act, 'render) Env.t array;
  autoreset_mode : autoreset_mode;
  observation_space : Space.packed;
  action_space : Space.packed;
  metadata : Metadata.t;
}

let ensure_non_empty envs =
  match envs with
  | [] -> invalid_arg "Vector_env.create_sync: env list cannot be empty"
  | _ -> ()

let ensure_consistent_spaces envs =
  match envs with
  | [] | [ _ ] -> ()
  | first :: rest ->
      let obs_space = Env.observation_space first in
      let act_space = Env.action_space first in
      List.iter
        (fun env ->
          let observation_space = Env.observation_space env in
          let action_space = Env.action_space env in
          if Space.shape obs_space <> Space.shape observation_space then
            raise_error
              (Invalid_metadata
                 "Vector env requires homogeneous observation spaces");
          if Space.shape act_space <> Space.shape action_space then
            raise_error
              (Invalid_metadata "Vector env requires homogeneous action spaces"))
        rest

let create_sync ?(autoreset_mode = Next_step) ~envs () =
  ensure_non_empty envs;
  ensure_consistent_spaces envs;
  let envs = Array.of_list envs in
  let first = envs.(0) in
  {
    envs;
    autoreset_mode;
    observation_space = Space.Pack (Env.observation_space first);
    action_space = Space.Pack (Env.action_space first);
    metadata = Env.metadata first;
  }

let num_envs vector_env = Array.length vector_env.envs
let observation_space vector_env = vector_env.observation_space
let action_space vector_env = vector_env.action_space
let metadata vector_env = vector_env.metadata

let reset vector_env () =
  let num_envs = num_envs vector_env in
  let results =
    Array.init num_envs (fun idx ->
        let env = vector_env.envs.(idx) in
        Env.reset env ())
  in
  let observations = Array.map fst results in
  let infos = Array.map snd results in
  (observations, infos)

let step vector_env actions =
  let num_envs = num_envs vector_env in
  if Array.length actions <> num_envs then
    invalid_arg "Vector_env.step: action array length mismatch";
  let results =
    Array.init num_envs (fun idx ->
        let env = vector_env.envs.(idx) in
        let transition = Env.step env actions.(idx) in
        match vector_env.autoreset_mode with
        | Disabled ->
            ( transition.observation,
              transition.reward,
              transition.terminated,
              transition.truncated,
              transition.info )
        | Next_step ->
            if transition.terminated || transition.truncated then
              let info =
                let packed =
                  Space.pack (Env.observation_space env) transition.observation
                in
                Info.set "vector.final_observation"
                  (Info.string (Space.Value.to_string packed))
                  transition.info
              in
              let obs_reset, info_reset = Env.reset env () in
              ( obs_reset,
                transition.reward,
                transition.terminated,
                transition.truncated,
                Info.merge info info_reset )
            else
              ( transition.observation,
                transition.reward,
                transition.terminated,
                transition.truncated,
                transition.info ))
  in
  let observations = Array.map (fun (obs, _, _, _, _) -> obs) results in
  let rewards = Array.map (fun (_, reward, _, _, _) -> reward) results in
  let terminations =
    Array.map (fun (_, _, terminated, _, _) -> terminated) results
  in
  let truncations =
    Array.map (fun (_, _, _, truncated, _) -> truncated) results
  in
  let infos = Array.map (fun (_, _, _, _, info) -> info) results in
  { observations; rewards; terminations; truncations; infos }

let close vector_env = Array.iter Env.close vector_env.envs
