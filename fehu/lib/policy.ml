type ('obs, 'act) t = 'obs -> 'act * float option * float option

let deterministic f obs = (f obs, None, None)

let random ?rng env =
  let action_space = Env.action_space env in
  let initial_rng =
    match rng with Some key -> key | None -> Env.take_rng env
  in
  let rng_ref = ref initial_rng in
  fun _ ->
    let action, next = Space.sample ~rng:!rng_ref action_space in
    rng_ref := next;
    (action, None, None)

let greedy_discrete env ~score =
  let action_space = Env.action_space env in
  let start =
    match Space.boundary_values action_space with
    | Space.Value.Int value :: _ -> value
    | _ -> 0
  in
  let to_action index =
    match Space.unpack action_space (Space.Value.Int index) with
    | Ok action -> action
    | Error msg ->
        invalid_arg ("Policy.greedy_discrete: " ^ msg)
  in
  fun obs ->
    let scores = score obs in
    if Array.length scores = 0 then
      invalid_arg "Policy.greedy_discrete: score returned empty array";
    let best_idx = ref 0 in
    let best_val = ref scores.(0) in
    for i = 1 to Array.length scores - 1 do
      let candidate = scores.(i) in
      if candidate > !best_val then (
        best_idx := i;
        best_val := candidate)
    done;
    let action_value = start + !best_idx in
    let action = to_action action_value in
    (action, None, None)
