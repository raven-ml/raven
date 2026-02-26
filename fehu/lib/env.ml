(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Error messages *)

let err_closed op = strf "Env: operation '%s' on a closed environment" op

let err_needs_reset op =
  strf "Env: operation '%s' requires calling reset first" op

let err_render_mode mode modes =
  strf "Env.create: render mode '%s' not in render_modes [%s]" mode
    (String.concat "; " modes)

let err_obs_reset value =
  strf "Env.reset: observation outside observation_space (value=%s)" value

let err_obs_step value =
  strf "Env.step: observation outside observation_space (value=%s)" value

let err_action value =
  strf "Env.step: action outside action_space (value=%s)" value

let err_split_rng = "Env.split_rng: n must be positive"

(* Step result *)

type 'obs step = {
  observation : 'obs;
  reward : float;
  terminated : bool;
  truncated : bool;
  info : Info.t;
}

let step_result ~observation ?(reward = 0.) ?(terminated = false)
    ?(truncated = false) ?(info = Info.empty) () =
  { observation; reward; terminated; truncated; info }

(* Render mode *)

type render_mode = [ `Human | `Rgb_array | `Ansi | `Svg | `Custom of string ]

let render_mode_to_string = function
  | `Human -> "human"
  | `Rgb_array -> "rgb_array"
  | `Ansi -> "ansi"
  | `Svg -> "svg"
  | `Custom name -> name

(* Shared mutable state *)

type shared = {
  mutable rng : Rune.Rng.key;
  mutable closed : bool;
  mutable needs_reset : bool;
}

(* Environment *)

type ('obs, 'act, 'render) t = {
  id : string option;
  observation_space : 'obs Space.t;
  action_space : 'act Space.t;
  render_mode : render_mode option;
  render_modes : string list;
  shared : shared;
  reset_fn : ?options:Info.t -> unit -> 'obs * Info.t;
  step_fn : 'act -> 'obs step;
  render_fn : unit -> 'render option;
  close_fn : unit -> unit;
}

(* Lifecycle guards *)

let ensure_open shared op = if shared.closed then invalid_arg (err_closed op)

let ensure_reset shared op =
  if shared.needs_reset then invalid_arg (err_needs_reset op)

(* Constructor *)

let create ?id ~rng ~observation_space ~action_space ?render_mode
    ?(render_modes = []) ~reset ~step ?render ?close () =
  (match render_mode with
  | None -> ()
  | Some mode ->
      let mode_s = render_mode_to_string mode in
      if not (List.mem mode_s render_modes) then
        invalid_arg (err_render_mode mode_s render_modes));
  let shared = { rng; closed = false; needs_reset = true } in
  let render_fn = Option.value render ~default:(fun () -> None) in
  let close_fn = Option.value close ~default:(fun () -> ()) in
  let rec env =
    {
      id;
      observation_space;
      action_space;
      render_mode;
      render_modes;
      shared;
      reset_fn = (fun ?options () -> reset env ?options ());
      step_fn = (fun action -> step env action);
      render_fn;
      close_fn;
    }
  in
  env

(* Wrap *)

let wrap ?id ~observation_space ~action_space ?render_mode ~reset ~step ?render
    ?close inner =
  let render_mode =
    match render_mode with Some _ -> render_mode | None -> inner.render_mode
  in
  let render_fn =
    match render with
    | Some f -> fun () -> f inner
    | None -> fun () -> inner.render_fn ()
  in
  let close_fn =
    match close with
    | Some f -> fun () -> f inner
    | None -> fun () -> inner.close_fn ()
  in
  {
    id;
    observation_space;
    action_space;
    render_mode;
    render_modes = inner.render_modes;
    shared = inner.shared;
    reset_fn = (fun ?options () -> reset inner ?options ());
    step_fn = (fun action -> step inner action);
    render_fn;
    close_fn;
  }

(* Accessors *)

let id env = env.id
let observation_space env = env.observation_space
let action_space env = env.action_space
let render_mode env = env.render_mode

(* RNG *)

let rng env = env.shared.rng

let set_rng env key =
  env.shared.rng <- key;
  env.shared.needs_reset <- true

let take_rng env =
  let keys = Rune.Rng.split env.shared.rng in
  env.shared.rng <- keys.(0);
  keys.(1)

let split_rng env ~n =
  if n <= 0 then invalid_arg err_split_rng;
  let keys = Rune.Rng.split ~n:(n + 1) env.shared.rng in
  env.shared.rng <- keys.(0);
  Array.sub keys 1 n

(* Human render helper *)

let maybe_human_render env =
  match env.render_mode with
  | Some `Human -> ignore (env.render_fn ())
  | _ -> ()

(* Lifecycle — all guards live here *)

let closed env = env.shared.closed

let reset env ?options () =
  ensure_open env.shared "reset";
  let observation, info = env.reset_fn ?options () in
  if not (Space.contains env.observation_space observation) then
    invalid_arg
      (err_obs_reset
         (Space.pack env.observation_space observation |> Value.to_string));
  env.shared.needs_reset <- false;
  maybe_human_render env;
  (observation, info)

let step env action =
  ensure_open env.shared "step";
  ensure_reset env.shared "step";
  if not (Space.contains env.action_space action) then
    invalid_arg
      (err_action (Space.pack env.action_space action |> Value.to_string));
  let result = env.step_fn action in
  if not (Space.contains env.observation_space result.observation) then
    invalid_arg
      (err_obs_step
         (Space.pack env.observation_space result.observation |> Value.to_string));
  if result.terminated || result.truncated then env.shared.needs_reset <- true;
  maybe_human_render env;
  result

let render env =
  ensure_open env.shared "render";
  env.render_fn ()

let close env =
  if not env.shared.closed then begin
    env.close_fn ();
    env.shared.closed <- true;
    env.shared.needs_reset <- true
  end

(* Wrapper helpers *)

let err_clip_bounds = "Env.clip_action: mismatched low/high bounds"
let err_clip_obs_bounds = "Env.clip_observation: mismatched low/high bounds"
let err_time_limit = "Env.time_limit: max_episode_steps must be positive"

let derive_id env suffix =
  match env.id with None -> None | Some id -> Some (id ^ suffix)

let clamp_tensor ~low ~high tensor =
  let data = Rune.to_array tensor in
  let clipped = Array.copy data in
  let upper = Array.length clipped - 1 in
  for idx = 0 to upper do
    let lo = low.(idx) in
    let hi = high.(idx) in
    let v = clipped.(idx) in
    if v < lo then clipped.(idx) <- lo else if v > hi then clipped.(idx) <- hi
  done;
  Rune.create Rune.float32 (Rune.shape tensor) clipped

(* Wrappers *)

let map_observation ~observation_space ~f env =
  wrap
    ?id:(derive_id env "/ObservationWrapper")
    ~observation_space ~action_space:env.action_space
    ~reset:(fun inner ?options () ->
      let obs, info = reset inner ?options () in
      f obs info)
    ~step:(fun inner action ->
      let s = step inner action in
      let obs, info = f s.observation s.info in
      { s with observation = obs; info })
    env

let map_action ~action_space ~f env =
  wrap
    ?id:(derive_id env "/ActionWrapper")
    ~observation_space:env.observation_space ~action_space
    ~reset:(fun inner ?options () -> reset inner ?options ())
    ~step:(fun inner action ->
      let s = step inner (f action) in
      {
        observation = s.observation;
        reward = s.reward;
        terminated = s.terminated;
        truncated = s.truncated;
        info = s.info;
      })
    env

let map_reward ~f env =
  wrap
    ?id:(derive_id env "/RewardWrapper")
    ~observation_space:env.observation_space ~action_space:env.action_space
    ~reset:(fun inner ?options () -> reset inner ?options ())
    ~step:(fun inner action ->
      let s = step inner action in
      let reward, info = f ~reward:s.reward ~info:s.info in
      { s with reward; info })
    env

(* Clipping *)

let clip_action env =
  let low, high = Space.Box.bounds env.action_space in
  let element_count = Array.length low in
  if Array.length high <> element_count then invalid_arg err_clip_bounds;
  let relaxed_low =
    Array.init element_count (fun i ->
        if Float.equal low.(i) high.(i) then low.(i) else Float.neg_infinity)
  in
  let relaxed_high =
    Array.init element_count (fun i ->
        if Float.equal low.(i) high.(i) then high.(i) else Float.infinity)
  in
  let relaxed_space = Space.Box.create ~low:relaxed_low ~high:relaxed_high in
  map_action ~action_space:relaxed_space
    ~f:(fun action -> clamp_tensor ~low ~high action)
    env

let clip_observation ~low ~high env =
  let inner_low, inner_high = Space.Box.bounds env.observation_space in
  let n = Array.length low in
  if Array.length high <> n then invalid_arg err_clip_obs_bounds;
  if Array.length inner_low <> n then invalid_arg err_clip_obs_bounds;
  let clamp_low = Array.init n (fun i -> Float.max low.(i) inner_low.(i)) in
  let clamp_high = Array.init n (fun i -> Float.min high.(i) inner_high.(i)) in
  let observation_space = Space.Box.create ~low:clamp_low ~high:clamp_high in
  map_observation ~observation_space
    ~f:(fun obs info ->
      (clamp_tensor ~low:clamp_low ~high:clamp_high obs, info))
    env

(* Limits *)

let time_limit ~max_episode_steps env =
  if max_episode_steps <= 0 then invalid_arg err_time_limit;
  let steps = ref 0 in
  let add_info info elapsed =
    info
    |> Info.set "time_limit.truncated" (Info.bool true)
    |> Info.set "time_limit.elapsed_steps" (Info.int elapsed)
  in
  wrap
    ?id:(derive_id env "/TimeLimit")
    ~observation_space:env.observation_space ~action_space:env.action_space
    ~reset:(fun inner ?options () ->
      steps := 0;
      reset inner ?options ())
    ~step:(fun inner action ->
      incr steps;
      let s = step inner action in
      if s.terminated || s.truncated then begin
        steps := 0;
        s
      end
      else if !steps >= max_episode_steps then begin
        let info = add_info s.info !steps in
        steps := 0;
        { s with truncated = true; info }
      end
      else s)
    env
