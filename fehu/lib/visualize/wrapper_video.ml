(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu

type when_to_record = Every_n_episodes of int | Steps of (int -> bool)

let derive_id env suffix =
  match Env.id env with None -> None | Some id -> Some (id ^ suffix)

let mkdir_p path =
  let rec aux dir =
    if dir = "" || dir = "." || dir = Filename.dir_sep then ()
    else if Sys.file_exists dir then ()
    else (
      aux (Filename.dirname dir);
      try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ())
  in
  aux path

let fps_of_metadata metadata =
  metadata.Metadata.render_fps |> Option.value ~default:30

let default_overlay overlay = Option.value overlay ~default:Overlay.identity

let expect_image_frame source = function
  | Render.Image image -> image
  | Render.None ->
      invalid_arg
        (Printf.sprintf
           "%s: render returned None; choose a render_mode that returns frames"
           source)
  | Render.Text _ ->
      invalid_arg
        (Printf.sprintf "%s: render produced text; expected image" source)
  | Render.Svg _ ->
      invalid_arg
        (Printf.sprintf "%s: render produced SVG; expected raster image" source)

let episode_file path episode_idx =
  Filename.concat path (Printf.sprintf "episode_%06d.mp4" (episode_idx + 1))

let step_file path step_idx =
  Filename.concat path (Printf.sprintf "step_%08d.mp4" step_idx)

let should_record_episode idx = function
  | Every_n_episodes n ->
      if n <= 0 then invalid_arg "record_video: n must be positive";
      (idx + 1) mod n = 0
  | Steps _ -> false

type single_state = {
  episode_idx : int;
  step_in_episode : int;
  global_step : int;
  sink : Sink.t option;
  recording : bool;
}

let record_video ~when_to_record ~path ?overlay env =
  mkdir_p path;
  let overlay = default_overlay overlay in
  let metadata = Env.metadata env in
  let fps = fps_of_metadata metadata in
  let action_space = Env.action_space env in
  let render_mode = Env.render_mode env in
  let state : single_state ref =
    ref
      {
        episode_idx = -1;
        step_in_episode = 0;
        global_step = 0;
        sink = None;
        recording = false;
      }
  in
  let close_sink () =
    match !state.sink with
    | None -> ()
    | Some sink ->
        Sink.close sink;
        state := { !state with sink = None; recording = false }
  in
  let open_episode_sink () =
    let file = episode_file path !state.episode_idx in
    state :=
      {
        !state with
        sink = Some (Sink.ffmpeg ~fps ~path:file ());
        recording = true;
      }
  in
  let open_step_sink step_idx =
    let file = step_file path step_idx in
    state :=
      {
        !state with
        sink = Some (Sink.ffmpeg ~fps ~path:file ());
        recording = true;
      }
  in
  let push_frame ~info ~action_opt ~reward ~is_done image =
    match !state.sink with
    | None -> ()
    | Some sink ->
        let ctx =
          {
            Overlay.step_idx = !state.step_in_episode;
            episode_idx = !state.episode_idx;
            info;
            action = action_opt;
            value = None;
            log_prob = None;
            reward;
            done_ = is_done;
          }
        in
        let image = overlay image ctx in
        Sink.push sink (Render.Image image)
  in
  let reset_handler _ ?options () =
    close_sink ();
    state :=
      { !state with episode_idx = !state.episode_idx + 1; step_in_episode = 0 };
    let observation, info = Env.reset env ?options () in
    if should_record_episode !state.episode_idx when_to_record then (
      open_episode_sink ();
      match Env.render env with
      | None -> ()
      | Some frame ->
          let image =
            expect_image_frame "Wrapper_video.record_video(reset)" frame
          in
          push_frame ~info ~action_opt:None ~reward:0. ~is_done:false image);
    (observation, info)
  in
  let step_handler _ action =
    let step_index = !state.global_step in
    let transition = Env.step env action in
    let done_flag = transition.terminated || transition.truncated in
    let action_value = Space.pack action_space action in
    let capture =
      match when_to_record with
      | Every_n_episodes _ -> !state.recording
      | Steps predicate ->
          let should_capture = predicate step_index in
          if should_capture && not !state.recording then (
            close_sink ();
            open_step_sink step_index)
          else if (not should_capture) && !state.recording then close_sink ();
          should_capture
    in
    (if capture then
       match Env.render env with
       | None ->
           invalid_arg
             "record_video: Env.render returned None while recording. Ensure \
              render_mode returns frames"
       | Some frame ->
           let image = expect_image_frame "Wrapper_video.record_video" frame in
           push_frame ~info:transition.info ~action_opt:(Some action_value)
             ~reward:transition.reward ~is_done:done_flag image);
    state :=
      {
        !state with
        global_step = step_index + 1;
        step_in_episode = (if done_flag then 0 else !state.step_in_episode + 1);
      };
    if done_flag then close_sink ();
    transition
  in
  Env.create
    ?id:(derive_id env "/VideoRecorder")
    ?render_mode ~metadata ~rng:(Env.rng env)
    ~observation_space:(Env.observation_space env)
    ~action_space ~reset:reset_handler ~step:step_handler
    ~render:(fun _ -> Env.render env)
    ~close:(fun _ ->
      close_sink ();
      Env.close env)
    ()

(* The vectorised recorder reuses the single-env wrapper for Single_each layout
   and coordinates frames across environments for NxM grids. *)

type shared = {
  layout_rows : int;
  layout_cols : int;
  when_to_record : when_to_record;
  overlay : Overlay.t;
  base_path : string;
  fps : int;
  num_envs : int;
  frames : Render.image option array;
  step_in_episode : int array;
  episode_counts : int array;
  mutable sink : Sink.t option;
  mutable recording : bool;
  mutable frames_recorded : int;
  mutable global_step : int;
  mutable episode_idx : int;
}

let close_shared_sink shared =
  match shared.sink with
  | None -> ()
  | Some sink ->
      Sink.close sink;
      shared.sink <- None;
      shared.recording <- false

let open_shared_episode_sink shared =
  let file =
    Filename.concat shared.base_path
      (Printf.sprintf "episode_%06d.mp4" (shared.episode_idx + 1))
  in
  shared.sink <- Some (Sink.ffmpeg ~fps:shared.fps ~path:file ());
  shared.recording <- true

let open_shared_step_sink shared =
  let start = shared.global_step in
  let file =
    Filename.concat shared.base_path (Printf.sprintf "step_%08d.mp4" start)
  in
  shared.sink <- Some (Sink.ffmpeg ~fps:shared.fps ~path:file ());
  shared.recording <- true

let clear_frames shared =
  Array.fill shared.frames 0 shared.num_envs None;
  shared.frames_recorded <- 0

let flush_frames shared =
  if shared.frames_recorded = shared.num_envs then (
    (match shared.when_to_record with
    | Steps predicate ->
        let capture = predicate shared.global_step in
        if capture && not shared.recording then (
          close_shared_sink shared;
          open_shared_step_sink shared)
        else if (not capture) && shared.recording then close_shared_sink shared
    | Every_n_episodes _ -> ());
    (match shared.when_to_record with
    | Every_n_episodes _ -> (
        if shared.recording then
          let images =
            Array.map
              (function
                | Some image -> image
                | None -> failwith "wrapper_video: missing frame")
              shared.frames
          in
          let composed =
            Utils.compose_grid ~rows:shared.layout_rows ~cols:shared.layout_cols
              images
          in
          match shared.sink with
          | Some sink -> Sink.push sink (Render.Image composed)
          | None -> ())
    | Steps _ -> (
        if shared.recording then
          let images =
            Array.map
              (function
                | Some image -> image
                | None -> failwith "wrapper_video: missing frame")
              shared.frames
          in
          let composed =
            Utils.compose_grid ~rows:shared.layout_rows ~cols:shared.layout_cols
              images
          in
          match shared.sink with
          | Some sink -> Sink.push sink (Render.Image composed)
          | None -> ()));
    shared.global_step <- shared.global_step + 1;
    clear_frames shared)

let wrap_env_for_grid shared idx env =
  let action_space = Env.action_space env in
  let render_mode = Env.render_mode env in
  let metadata = Env.metadata env in
  let rng = Env.rng env in
  let reset_handler _ ?options () =
    shared.step_in_episode.(idx) <- 0;
    shared.episode_counts.(idx) <- shared.episode_counts.(idx) + 1;
    if idx = 0 then (
      shared.episode_idx <- shared.episode_idx + 1;
      close_shared_sink shared;
      if should_record_episode shared.episode_idx shared.when_to_record then
        open_shared_episode_sink shared);
    let observation, info = Env.reset env ?options () in
    (observation, info)
  in
  let step_handler _ action =
    let transition = Env.step env action in
    let done_flag = transition.terminated || transition.truncated in
    (match Env.render env with
    | None ->
        invalid_arg "vec_record_video: Env.render returned None while recording"
    | Some frame ->
        let image = expect_image_frame "Wrapper_video.vec_record_video" frame in
        let ctx =
          {
            Overlay.step_idx = shared.step_in_episode.(idx);
            episode_idx = shared.episode_counts.(idx);
            info = transition.info;
            action = Some (Space.pack action_space action);
            value = None;
            log_prob = None;
            reward = transition.reward;
            done_ = done_flag;
          }
        in
        let image = shared.overlay image ctx in
        shared.frames.(idx) <- Some image;
        shared.frames_recorded <- shared.frames_recorded + 1;
        flush_frames shared);
    shared.step_in_episode.(idx) <-
      (if done_flag then 0 else shared.step_in_episode.(idx) + 1);
    transition
  in
  Env.create
    ?id:(derive_id env "/VideoRecorder")
    ?render_mode ~metadata ~rng
    ~observation_space:(Env.observation_space env)
    ~action_space ~reset:reset_handler ~step:step_handler
    ~render:(fun _ -> Env.render env)
    ~close:(fun _ ->
      if idx = 0 then close_shared_sink shared;
      Env.close env)
    ()

let vec_record_video ~layout ~when_to_record ~path ?overlay vec_env =
  mkdir_p path;
  let overlay = default_overlay overlay in
  match layout with
  | `Single_each ->
      let envs = Vector_env.envs vec_env in
      Array.iteri
        (fun idx env ->
          let subdir =
            Filename.concat path (Printf.sprintf "env_%02d" (idx + 1))
          in
          envs.(idx) <- record_video ~when_to_record ~path:subdir ~overlay env)
        envs;
      vec_env
  | `NxM_grid (rows, cols) ->
      let num_envs = Vector_env.num_envs vec_env in
      if rows * cols <> num_envs then
        invalid_arg "vec_record_video: grid layout must cover all environments";
      let metadata = Vector_env.metadata vec_env in
      let fps = fps_of_metadata metadata in
      let shared =
        {
          layout_rows = rows;
          layout_cols = cols;
          when_to_record;
          overlay;
          base_path = path;
          fps;
          num_envs;
          frames = Array.make num_envs None;
          step_in_episode = Array.make num_envs 0;
          episode_counts = Array.make num_envs (-1);
          sink = None;
          recording = false;
          frames_recorded = 0;
          global_step = 0;
          episode_idx = -1;
        }
      in
      let envs = Vector_env.envs vec_env in
      Array.iteri
        (fun idx env -> envs.(idx) <- wrap_env_for_grid shared idx env)
        envs;
      vec_env
