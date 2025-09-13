(*
```ocaml
 *)
(* Curriculum learning concepts - standalone module *)

(* Slide 10: Introduction to Curriculum Learning

   The Problem: Learning complex tasks from scratch is hard!
   - Random exploration in complex environments is inefficient
   - Sparse rewards make learning nearly impossible
   - Agent may never discover successful behaviors

   The Solution: Start simple, gradually increase difficulty
   - Like teaching a child: crawl → walk → run
   - Build skills incrementally
   - Transfer knowledge from easy to hard tasks
*)

(* Simple Sokoban levels for curriculum *)
module SimpleLevels = struct
  (* Level 1: Straight corridor - push box to target *)
  let corridor_3x3 () =
    [|
      [|'#'; '#'; '#'; '#'; '#'|];
      [|'#'; '@'; ' '; 'o'; '#'|];  (* @ = player, o = box *)
      [|'#'; ' '; ' '; 'x'; '#'|];  (* x = target *)
      [|'#'; '#'; '#'; '#'; '#'|];
    |]

  (* Level 2: Need to navigate around *)
  let room_5x5 () =
    [|
      [|'#'; '#'; '#'; '#'; '#'; '#'; '#'|];
      [|'#'; '@'; ' '; ' '; ' '; ' '; '#'|];
      [|'#'; ' '; 'o'; ' '; ' '; ' '; '#'|];
      [|'#'; ' '; ' '; ' '; ' '; ' '; '#'|];
      [|'#'; ' '; ' '; ' '; 'x'; ' '; '#'|];
      [|'#'; ' '; ' '; ' '; ' '; ' '; '#'|];
      [|'#'; '#'; '#'; '#'; '#'; '#'; '#'|];
    |]

  (* Level 3: Multiple boxes *)
  let multi_box () =
    [|
      [|'#'; '#'; '#'; '#'; '#'; '#'; '#'|];
      [|'#'; '@'; ' '; ' '; ' '; ' '; '#'|];
      [|'#'; ' '; 'o'; ' '; 'o'; ' '; '#'|];
      [|'#'; ' '; ' '; ' '; ' '; ' '; '#'|];
      [|'#'; ' '; 'x'; ' '; 'x'; ' '; '#'|];
      [|'#'; ' '; ' '; ' '; ' '; ' '; '#'|];
      [|'#'; '#'; '#'; '#'; '#'; '#'; '#'|];
    |]
end

(* Curriculum stages with increasing difficulty *)
type curriculum_stage = {
  name: string;
  difficulty: int;  (* 1-10 scale *)
  grid_size: int;
  n_boxes: int;
  max_steps: int;
  success_threshold: float;  (* Win rate to advance *)
}

let curriculum_stages = [
  { name = "Straight Line"; difficulty = 1;
    grid_size = 3; n_boxes = 1; max_steps = 10;
    success_threshold = 0.9 };
  { name = "Simple Room"; difficulty = 3;
    grid_size = 5; n_boxes = 1; max_steps = 20;
    success_threshold = 0.8 };
  { name = "Two Boxes"; difficulty = 5;
    grid_size = 5; n_boxes = 2; max_steps = 30;
    success_threshold = 0.7 };
  { name = "Complex Room"; difficulty = 7;
    grid_size = 7; n_boxes = 2; max_steps = 50;
    success_threshold = 0.6 };
  { name = "Master Level"; difficulty = 10;
    grid_size = 9; n_boxes = 3; max_steps = 100;
    success_threshold = 0.5 };
]

(* Track curriculum progress *)
type curriculum_state = {
  current_stage: int;
  episodes_in_stage: int;
  recent_wins: bool Queue.t;  (* Track last N episodes *)
  total_episodes: int;
  stage_transitions: (int * int) list;  (* (episode, new_stage) *)
}

let create_curriculum_state () = {
  current_stage = 0;
  episodes_in_stage = 0;
  recent_wins = Queue.create ();
  total_episodes = 0;
  stage_transitions = [];
}

(* Check if ready to advance to next stage *)
let should_advance state window_size =
  let stage = List.nth curriculum_stages state.current_stage in
  if Queue.length state.recent_wins < window_size then
    false
  else
    let wins = Queue.fold (fun acc win -> if win then acc + 1 else acc) 0 state.recent_wins in
    let win_rate = float_of_int wins /. float_of_int window_size in
    win_rate >= stage.success_threshold

(* Update curriculum based on episode result *)
let update_curriculum state won window_size =
  (* Add to recent wins *)
  Queue.add won state.recent_wins;
  if Queue.length state.recent_wins > window_size then
    ignore (Queue.take state.recent_wins);

  let new_state = {
    state with
    episodes_in_stage = state.episodes_in_stage + 1;
    total_episodes = state.total_episodes + 1;
  } in

  (* Check for advancement *)
  if should_advance new_state window_size &&
     new_state.current_stage < List.length curriculum_stages - 1 then
    { new_state with
      current_stage = new_state.current_stage + 1;
      episodes_in_stage = 0;
      recent_wins = Queue.create ();
      stage_transitions = (new_state.total_episodes, new_state.current_stage + 1) :: new_state.stage_transitions;
    }
  else
    new_state

(* Visualize curriculum progress *)
let print_curriculum_progress state =
  let stage = List.nth curriculum_stages state.current_stage in
  let wins = Queue.fold (fun acc w -> if w then acc + 1 else acc) 0 state.recent_wins in
  let n_recent = Queue.length state.recent_wins in
  let win_rate = if n_recent > 0 then float_of_int wins /. float_of_int n_recent else 0.0 in

  Printf.printf "\n=== Curriculum Progress ===\n";
  Printf.printf "Current Stage: %s (Level %d/%d)\n"
    stage.name (state.current_stage + 1) (List.length curriculum_stages);
  Printf.printf "Difficulty: %d/10\n" stage.difficulty;
  Printf.printf "Episodes in stage: %d\n" state.episodes_in_stage;
  Printf.printf "Recent win rate: %.1f%% (%d/%d)\n"
    (win_rate *. 100.0) wins n_recent;
  Printf.printf "Need %.1f%% to advance\n" (stage.success_threshold *. 100.0);

  (* Show progression history *)
  if state.stage_transitions <> [] then begin
    Printf.printf "\nProgression History:\n";
    List.iter (fun (ep, stage) ->
      Printf.printf "  Episode %d: Advanced to stage %d\n" ep stage
    ) (List.rev state.stage_transitions)
  end

(* Main demonstration *)
let main () =
  print_endline "=== Slide 10: Curriculum Learning ===";
  print_endline "Teaching agents to solve complex tasks gradually\n";

  (* Simulate curriculum progression *)
  let state = ref (create_curriculum_state ()) in
  let window_size = 20 in

  Printf.printf "Simulating curriculum with %d-episode window\n\n" window_size;

  (* Simulate episodes with improving performance *)
  for episode = 1 to 100 do
    (* Simulate win probability based on stage difficulty *)
    let stage = List.nth curriculum_stages !state.current_stage in
    (* Win probability decreases with difficulty but improves over time in stage *)
    let base_prob = 1.0 -. (float_of_int stage.difficulty /. 15.0) in
    let experience_bonus = min 0.3 (float_of_int !state.episodes_in_stage /. 50.0) in
    let win_prob = base_prob +. experience_bonus in
    let won = Random.float 1.0 < win_prob in

    state := update_curriculum !state won window_size;

    (* Print progress at key points *)
    if episode = 1 || episode mod 20 = 0 ||
       (List.exists (fun (ep, _) -> ep = episode) !state.stage_transitions) then begin
      Printf.printf "\n--- Episode %d ---\n" episode;
      print_curriculum_progress !state
    end
  done;

  print_endline "\n=== Key Concepts ===";
  print_endline "1. Start with easy tasks (straight corridor)";
  print_endline "2. Track performance with rolling window";
  print_endline "3. Advance when consistently successful";
  print_endline "4. Each stage builds on previous skills";
  print_endline "5. Prevents getting stuck on too-hard tasks"

(*
```
 *)