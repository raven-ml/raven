(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

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

let compatibility_rngs = Array.init 4 (fun i -> Rune.Rng.key (137 + i))

let rec info_of_space_value = function
  | Space.Value.Int i -> Info.int i
  | Space.Value.Float f -> Info.float f
  | Space.Value.Bool b -> Info.bool b
  | Space.Value.Int_array arr -> Info.int_array arr
  | Space.Value.Float_array arr -> Info.float_array arr
  | Space.Value.Bool_array arr -> Info.bool_array arr
  | Space.Value.String s -> Info.string s
  | Space.Value.List values -> Info.list (List.map info_of_space_value values)
  | Space.Value.Tuple values -> Info.list (List.map info_of_space_value values)
  | Space.Value.Dict entries ->
      Info.dict
        (List.map
           (fun (key, value) -> (key, info_of_space_value value))
           entries)

let ensure_space_equivalence kind reference_space candidate_space candidate_env
    =
  let env_id = Env.id candidate_env in
  let fail detail =
    let base = Printf.sprintf "Vector env requires identical %s spaces" kind in
    let message =
      match env_id with
      | None -> Printf.sprintf "%s (%s)" base detail
      | Some id -> Printf.sprintf "%s (env id: %s, %s)" base id detail
    in
    raise_error (Invalid_metadata message)
  in
  let check source target =
    Array.iteri
      (fun sample_idx rng ->
        let sample, _ = Space.sample ~rng source in
        let packed = Space.pack source sample in
        match Space.unpack target packed with
        | Ok value ->
            if not (Space.contains target value) then
              fail
                (Printf.sprintf "sample %d rejected by target space" sample_idx)
        | Error msg ->
            fail (Printf.sprintf "sample %d unpack error: %s" sample_idx msg))
      compatibility_rngs
  in
  let check_boundaries source target =
    List.iter
      (fun boundary ->
        match Space.unpack target boundary with
        | Ok value ->
            if not (Space.contains target value) then
              fail
                (Printf.sprintf "boundary value %s rejected by target space"
                   (Space.Value.to_string boundary))
        | Error msg ->
            fail
              (Printf.sprintf "boundary value %s unpack error: %s"
                 (Space.Value.to_string boundary)
                 msg))
      (Space.boundary_values source)
  in
  check reference_space candidate_space;
  check candidate_space reference_space;
  check_boundaries reference_space candidate_space;
  check_boundaries candidate_space reference_space

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
        rest;
      List.iter
        (fun env ->
          let observation_space = Env.observation_space env in
          let action_space = Env.action_space env in
          ensure_space_equivalence "observation" obs_space observation_space env;
          ensure_space_equivalence "action" act_space action_space env)
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
                Info.set "final_observation"
                  (info_of_space_value packed)
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

let render vector_env = Array.map Env.render vector_env.envs
let envs vector_env = vector_env.envs
let close vector_env = Array.iter Env.close vector_env.envs
