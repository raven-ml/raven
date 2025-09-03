(** Visualization utilities for RL training *)

type episode_frame = {
  step: int;
  state_repr: string;
  action: string;
  reward: float;
  value: float option;
}

type episode_log = {
  episode_num: int;
  total_reward: float;
  total_steps: int;
  won: bool;
  frames: episode_frame list;
  stage: string option;
}

val clear_screen : unit -> unit
val action_to_string : int -> string
val animate_episode : episode_log -> float -> unit
val save_episode_log : episode_log -> string -> unit
val load_episode_log : string -> episode_log
val print_training_progress : int -> float -> float -> string option -> unit
val visualize_multiple_episodes : episode_log list -> float -> unit
val summary_stats : episode_log list -> unit