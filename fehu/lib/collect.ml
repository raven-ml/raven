(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let err_concat_empty = "Collect.concat: empty list"

type ('obs, 'act) t = {
  observations : 'obs array;
  actions : 'act array;
  rewards : float array;
  next_observations : 'obs array;
  terminated : bool array;
  truncated : bool array;
  infos : Info.t array;
  log_probs : float array option;
  values : float array option;
}

let length t = Array.length t.observations

(* Concatenation *)

let concat_opt_field ts get =
  if List.for_all (fun t -> Option.is_some (get t)) ts then
    Some (Array.concat (List.map (fun t -> Option.get (get t)) ts))
  else None

let concat = function
  | [] -> invalid_arg err_concat_empty
  | [ t ] -> t
  | ts ->
      {
        observations = Array.concat (List.map (fun t -> t.observations) ts);
        actions = Array.concat (List.map (fun t -> t.actions) ts);
        rewards = Array.concat (List.map (fun t -> t.rewards) ts);
        next_observations =
          Array.concat (List.map (fun t -> t.next_observations) ts);
        terminated = Array.concat (List.map (fun t -> t.terminated) ts);
        truncated = Array.concat (List.map (fun t -> t.truncated) ts);
        infos = Array.concat (List.map (fun t -> t.infos) ts);
        log_probs = concat_opt_field ts (fun t -> t.log_probs);
        values = concat_opt_field ts (fun t -> t.values);
      }

(* Accumulator for building trajectories *)

type ('obs, 'act) acc = {
  mutable obs : 'obs list;
  mutable acts : 'act list;
  mutable rews : float list;
  mutable next_obs : 'obs list;
  mutable terms : bool list;
  mutable truncs : bool list;
  mutable infos_acc : Info.t list;
  mutable lps : float list;
  mutable vals : float list;
  mutable count : int;
}

let create_acc () =
  {
    obs = [];
    acts = [];
    rews = [];
    next_obs = [];
    terms = [];
    truncs = [];
    infos_acc = [];
    lps = [];
    vals = [];
    count = 0;
  }

let acc_step acc ~current_obs ~action ~lp_opt ~v_opt (s : _ Env.step) =
  acc.obs <- current_obs :: acc.obs;
  acc.acts <- action :: acc.acts;
  acc.rews <- s.reward :: acc.rews;
  acc.next_obs <- s.observation :: acc.next_obs;
  acc.terms <- s.terminated :: acc.terms;
  acc.truncs <- s.truncated :: acc.truncs;
  acc.infos_acc <- s.info :: acc.infos_acc;
  (match lp_opt with Some lp -> acc.lps <- lp :: acc.lps | None -> ());
  (match v_opt with Some v -> acc.vals <- v :: acc.vals | None -> ());
  acc.count <- acc.count + 1

let acc_to_trajectory acc =
  let n = acc.count in
  let log_probs =
    if List.length acc.lps = n then Some (Array.of_list (List.rev acc.lps))
    else None
  in
  let values =
    if List.length acc.vals = n then Some (Array.of_list (List.rev acc.vals))
    else None
  in
  {
    observations = Array.of_list (List.rev acc.obs);
    actions = Array.of_list (List.rev acc.acts);
    rewards = Array.of_list (List.rev acc.rews);
    next_observations = Array.of_list (List.rev acc.next_obs);
    terminated = Array.of_list (List.rev acc.terms);
    truncated = Array.of_list (List.rev acc.truncs);
    infos = Array.of_list (List.rev acc.infos_acc);
    log_probs;
    values;
  }

(* Collecting *)

let rollout env ~policy ~n_steps =
  let acc = create_acc () in
  let obs, _info = Env.reset env () in
  let current_obs = ref obs in
  while acc.count < n_steps do
    let action, lp_opt, v_opt = policy !current_obs in
    let s = Env.step env action in
    acc_step acc ~current_obs:!current_obs ~action ~lp_opt ~v_opt s;
    current_obs := s.observation;
    if s.terminated || s.truncated then begin
      let obs, _info = Env.reset env () in
      current_obs := obs
    end
  done;
  acc_to_trajectory acc

let episodes env ~policy ~n_episodes ?(max_steps = 1000) () =
  let eps = ref [] in
  for _ = 1 to n_episodes do
    let acc = create_acc () in
    let obs, _info = Env.reset env () in
    let current_obs = ref obs in
    let done_flag = ref false in
    while acc.count < max_steps && not !done_flag do
      let action, lp_opt, v_opt = policy !current_obs in
      let s = Env.step env action in
      acc_step acc ~current_obs:!current_obs ~action ~lp_opt ~v_opt s;
      current_obs := s.observation;
      done_flag := s.terminated || s.truncated
    done;
    eps := acc_to_trajectory acc :: !eps
  done;
  List.rev !eps
