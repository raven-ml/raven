(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A random agent on CartPole-v1.

   Demonstrates the Env lifecycle: create, reset, step, render, close. Then uses
   Eval.run for batch evaluation. *)

open Fehu

let () =
  Rune.Rng.run ~seed:42 @@ fun () ->
  Printf.printf "Random Agent on CartPole-v1\n";
  Printf.printf "===========================\n\n";

  let env = Fehu_envs.Cartpole.make ~render_mode:`Ansi () in

  (* -- Manual episode loop ------------------------------------------------ *)
  Printf.printf "Running 5 episodes with random actions...\n\n";

  for episode = 1 to 5 do
    let obs = ref (fst (Env.reset env ())) in
    let total_reward = ref 0.0 in
    let steps = ref 0 in
    let done_ = ref false in
    while not !done_ do
      (* Show the first step of episode 1 *)
      (if episode = 1 && !steps = 0 then
         match Env.render env with
         | Some text -> Printf.printf "%s\n" text
         | None -> ());
      let action = Space.sample (Env.action_space env) in
      let s = Env.step env action in
      total_reward := !total_reward +. s.reward;
      incr steps;
      obs := s.observation;
      done_ := s.terminated || s.truncated
    done;
    Printf.printf "  Episode %d:  reward = %5.1f  length = %3d\n" episode
      !total_reward !steps
  done;

  (* -- Batch evaluation with Eval.run ------------------------------------ *)
  Printf.printf "\nEvaluating over 100 episodes...\n\n";

  let random_policy _obs = Space.sample (Env.action_space env) in
  let stats = Eval.run env ~policy:random_policy ~n_episodes:100 () in
  Printf.printf "  mean reward: %6.2f +/- %.2f\n" stats.mean_reward
    stats.std_reward;
  Printf.printf "  mean length: %6.1f\n" stats.mean_length;

  Env.close env;
  Printf.printf "\nDone.\n"
