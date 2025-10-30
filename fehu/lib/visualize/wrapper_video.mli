(** Video recording wrappers for environments and vectorized environments. *)

type when_to_record =
  | Every_n_episodes of int
  | Steps of (int -> bool)
      (** Recording schedule.

          - [Every_n_episodes n]: record every [n]-th episode (1-indexed)
          - [Steps predicate]: record steps where [predicate global_step] is
            [true] *)

val record_video :
  when_to_record:when_to_record ->
  path:string ->
  ?overlay:Overlay.t ->
  ('obs, 'act, Fehu.Render.t) Fehu.Env.t ->
  ('obs, 'act, Fehu.Render.t) Fehu.Env.t
(** Record an environment to per-episode or per-step video files under [path].

    [overlay] customises rendered frames before they are written. *)

val vec_record_video :
  layout:[ `Single_each | `NxM_grid of int * int ] ->
  when_to_record:when_to_record ->
  path:string ->
  ?overlay:Overlay.t ->
  ('obs, 'act, Fehu.Render.t) Fehu.Vector_env.t ->
  ('obs, 'act, Fehu.Render.t) Fehu.Vector_env.t
(** Record a vectorized environment.

    - [`Single_each]: apply {!record_video} to each inner environment with a
      dedicated sub-directory
    - [`NxM_grid rows_cols]: compose frames into a grid before writing.

    [path] is treated as a directory and created if it does not exist. *)
