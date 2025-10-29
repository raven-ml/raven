open Errors

type ('obs, 'act, 'render) transition = {
  observation : 'obs;
  reward : float;
  terminated : bool;
  truncated : bool;
  info : Info.t;
}

let transition ?(reward = 0.) ?(terminated = false) ?(truncated = false)
    ?(info = Info.empty) ~observation () =
  { observation; reward; terminated; truncated; info }

type ('obs, 'act, 'render) t = {
  id : string option;
  mutable metadata : Metadata.t;
  observation_space : 'obs Space.t;
  action_space : 'act Space.t;
  mutable rng : Rune.Rng.key;
  mutable closed : bool;
  mutable needs_reset : bool;
  reset_impl :
    ('obs, 'act, 'render) t -> ?options:Info.t -> unit -> 'obs * Info.t;
  step_impl :
    ('obs, 'act, 'render) t -> 'act -> ('obs, 'act, 'render) transition;
  render_impl : ('obs, 'act, 'render) t -> 'render option;
  close_impl : ('obs, 'act, 'render) t -> unit;
}

let ensure_open env ~operation =
  if env.closed then
    raise_error
      (Closed_environment
         (Printf.sprintf "Operation '%s' on a closed environment" operation))

let ensure_reset env ~operation =
  if env.needs_reset then
    raise_error
      (Reset_needed
         (Printf.sprintf "Operation '%s' requires calling reset first" operation))

let create ?id ?(metadata = Metadata.default) ?validate_transition ~rng
    ~observation_space ~action_space ~reset:reset_handler ~step:step_handler
    ?render ?close () =
  let render_impl = Option.value render ~default:(fun _ -> None) in
  let close_impl = Option.value close ~default:(fun _ -> ()) in
  let rec env =
    {
      id;
      metadata;
      observation_space;
      action_space;
      rng;
      closed = false;
      needs_reset = true;
      reset_impl;
      step_impl;
      render_impl;
      close_impl;
    }
  and reset_impl env ?options () =
    ensure_open env ~operation:"reset";
    let observation, info = reset_handler env ?options () in
    if not (Space.contains env.observation_space observation) then
      let value =
        Space.pack env.observation_space observation |> Space.Value.to_string
      in
      raise_error
        (Invalid_metadata
           (Printf.sprintf
              "Reset produced an observation outside observation_space \
               (value=%s)"
              value))
    else (
      env.needs_reset <- false;
      (observation, info))
  and step_impl env action =
    ensure_open env ~operation:"step";
    ensure_reset env ~operation:"step";
    (if not (Space.contains env.action_space action) then
       let value =
         Space.pack env.action_space action |> Space.Value.to_string
       in
       ignore
         (raise_error
            (Invalid_action
               (Printf.sprintf "Action outside of action_space (value=%s)" value))));
    let transition = step_handler env action in
    (if not (Space.contains env.observation_space transition.observation) then
       let value =
         Space.pack env.observation_space transition.observation
         |> Space.Value.to_string
       in
       ignore
         (raise_error
            (Invalid_metadata
               (Printf.sprintf
                  "Step produced an observation outside observation_space \
                   (value=%s)"
                  value))));
    Option.iter (fun validate -> validate transition) validate_transition;
    if transition.terminated || transition.truncated then
      env.needs_reset <- true;
    transition
  in
  env

let id env = env.id
let metadata env = env.metadata
let set_metadata env metadata = env.metadata <- metadata
let rng env = env.rng

let set_rng env rng =
  env.rng <- rng;
  env.needs_reset <- true

let take_rng env =
  let keys = Rune.Rng.split env.rng in
  env.rng <- keys.(0);
  keys.(1)

let split_rng env ~n =
  if n <= 0 then invalid_arg "Env.split_rng: n must be positive";
  let keys = Rune.Rng.split ~n:(n + 1) env.rng in
  env.rng <- keys.(0);
  Array.sub keys 1 n

let observation_space env = env.observation_space
let action_space env = env.action_space
let reset env ?options () = env.reset_impl env ?options ()
let step env action = env.step_impl env action

let render env =
  ensure_open env ~operation:"render";
  env.render_impl env

let close env =
  if not env.closed then (
    env.close_impl env;
    env.closed <- true;
    env.needs_reset <- true)

let closed env = env.closed
