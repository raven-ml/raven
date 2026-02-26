(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Error messages *)

let err_empty = "Vec_env.create: env list must not be empty"
let err_action_len n m = strf "Vec_env.step: expected %d actions, got %d" n m

let err_space kind =
  strf "Vec_env.create: all environments must have the same %s space" kind

(* Types *)

type 'obs step = {
  observations : 'obs array;
  rewards : float array;
  terminated : bool array;
  truncated : bool array;
  infos : Info.t array;
}

type ('obs, 'act, 'render) t = {
  envs : ('obs, 'act, 'render) Env.t array;
  observation_space : 'obs Space.t;
  action_space : 'act Space.t;
}

(* Space compatibility *)

let ensure_compatible envs =
  let first = envs.(0) in
  let obs_spec = Space.spec (Env.observation_space first) in
  let act_spec = Space.spec (Env.action_space first) in
  for i = 1 to Array.length envs - 1 do
    let env = envs.(i) in
    if not (Space.equal_spec obs_spec (Space.spec (Env.observation_space env)))
    then invalid_arg (err_space "observation");
    if not (Space.equal_spec act_spec (Space.spec (Env.action_space env))) then
      invalid_arg (err_space "action")
  done

(* Constructor *)

let create envs =
  match envs with
  | [] -> invalid_arg err_empty
  | first :: _ ->
      let envs = Array.of_list envs in
      ensure_compatible envs;
      {
        envs;
        observation_space = Env.observation_space first;
        action_space = Env.action_space first;
      }

(* Accessors *)

let num_envs t = Array.length t.envs
let observation_space t = t.observation_space
let action_space t = t.action_space

(* Reset *)

let reset t () =
  let n = Array.length t.envs in
  let results = Array.init n (fun i -> Env.reset t.envs.(i) ()) in
  let observations = Array.map fst results in
  let infos = Array.map snd results in
  (observations, infos)

(* Step *)

let step t actions =
  let n = Array.length t.envs in
  if Array.length actions <> n then
    invalid_arg (err_action_len n (Array.length actions));
  let results = Array.init n (fun i -> Env.step t.envs.(i) actions.(i)) in
  let observations = Array.make n results.(0).observation in
  let rewards = Array.make n 0. in
  let terminated = Array.make n false in
  let truncated = Array.make n false in
  let infos = Array.make n Info.empty in
  for i = 0 to n - 1 do
    let result = results.(i) in
    rewards.(i) <- result.reward;
    terminated.(i) <- result.terminated;
    truncated.(i) <- result.truncated;
    if result.terminated || result.truncated then begin
      let final_obs = Space.pack t.observation_space result.observation in
      let info = Info.set "final_observation" final_obs result.info in
      let info = Info.set "final_info" (Info.to_value result.info) info in
      let obs, reset_info = Env.reset t.envs.(i) () in
      observations.(i) <- obs;
      infos.(i) <- Info.merge info reset_info
    end
    else begin
      observations.(i) <- result.observation;
      infos.(i) <- result.info
    end
  done;
  { observations; rewards; terminated; truncated; infos }

(* Close *)

let close t = Array.iter Env.close t.envs
