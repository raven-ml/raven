type init = float array
type transitions = float array array
type emissions = float array array

type t = {
  init : init;
  transitions : transitions;
  emissions : emissions;
  num_states : int;
  num_observations : int;
}

let epsilon = 1e-12
let logf x = log (max x epsilon)

let normalize_row row =
  let sum = Array.fold_left ( +. ) 0.0 row in
  if sum <= 0.0 then
    Array.fill row 0 (Array.length row) (1.0 /. float_of_int (Array.length row))
  else Array.iteri (fun i v -> row.(i) <- v /. sum) row

let create ~init ~transitions ~emissions =
  let num_states = Array.length init in
  if num_states = 0 then invalid_arg "Hmm.create: empty init";
  Array.iter normalize_row transitions;
  Array.iter normalize_row emissions;
  let num_observations =
    if num_states > 0 then Array.length emissions.(0) else 0
  in
  { init; transitions; emissions; num_states; num_observations }

let num_states hmm = hmm.num_states
let num_observations hmm = hmm.num_observations

let forward hmm observations =
  let t_len = Array.length observations in
  if t_len = 0 then [||]
  else
    let alpha = Array.make_matrix t_len hmm.num_states 0.0 in
    let first_obs = observations.(0) in
    for i = 0 to hmm.num_states - 1 do
      alpha.(0).(i) <- hmm.init.(i) *. hmm.emissions.(i).(first_obs)
    done;
    normalize_row alpha.(0);
    for t = 1 to t_len - 1 do
      let obs = observations.(t) in
      for j = 0 to hmm.num_states - 1 do
        let sum = ref 0.0 in
        for i = 0 to hmm.num_states - 1 do
          sum := !sum +. (alpha.(t - 1).(i) *. hmm.transitions.(i).(j))
        done;
        alpha.(t).(j) <- !sum *. hmm.emissions.(j).(obs)
      done;
      normalize_row alpha.(t)
    done;
    alpha

let backward hmm observations =
  let t_len = Array.length observations in
  if t_len = 0 then [||]
  else
    let beta = Array.make_matrix t_len hmm.num_states 1.0 in
    normalize_row beta.(t_len - 1);
    for t = t_len - 2 downto 0 do
      let obs = observations.(t + 1) in
      for i = 0 to hmm.num_states - 1 do
        let sum = ref 0.0 in
        for j = 0 to hmm.num_states - 1 do
          sum :=
            !sum
            +. hmm.transitions.(i).(j)
               *. hmm.emissions.(j).(obs)
               *. beta.(t + 1).(j)
        done;
        beta.(t).(i) <- !sum
      done;
      normalize_row beta.(t)
    done;
    beta

let log_likelihood hmm observations =
  let alpha = forward hmm observations in
  if Array.length alpha = 0 then 0.0
  else
    let last = Array.length alpha - 1 in
    let sum = Array.fold_left ( +. ) 0.0 alpha.(last) in
    logf sum

let viterbi hmm observations =
  let t_len = Array.length observations in
  if t_len = 0 then [||]
  else
    let dp = Array.make_matrix t_len hmm.num_states Float.neg_infinity in
    let backpointer = Array.make_matrix t_len hmm.num_states 0 in
    let first_obs = observations.(0) in
    for i = 0 to hmm.num_states - 1 do
      dp.(0).(i) <- logf hmm.init.(i) +. logf hmm.emissions.(i).(first_obs)
    done;
    for t = 1 to t_len - 1 do
      let obs = observations.(t) in
      for j = 0 to hmm.num_states - 1 do
        let best_score = ref Float.neg_infinity in
        let best_state = ref 0 in
        for i = 0 to hmm.num_states - 1 do
          let score =
            dp.(t - 1).(i)
            +. logf hmm.transitions.(i).(j)
            +. logf hmm.emissions.(j).(obs)
          in
          if score > !best_score then (
            best_score := score;
            best_state := i)
        done;
        dp.(t).(j) <- !best_score;
        backpointer.(t).(j) <- !best_state
      done
    done;
    (* backtrack *)
    let path = Array.make t_len 0 in
    let last_state = ref 0 in
    let last_score = ref Float.neg_infinity in
    for j = 0 to hmm.num_states - 1 do
      if dp.(t_len - 1).(j) > !last_score then (
        last_score := dp.(t_len - 1).(j);
        last_state := j)
    done;
    path.(t_len - 1) <- !last_state;
    for t = t_len - 2 downto 0 do
      path.(t) <- backpointer.(t + 1).(path.(t + 1))
    done;
    path

let baum_welch ?(tol = 1e-4) ?(max_iter = 100) hmm sequences =
  let model =
    create ~init:(Array.copy hmm.init)
      ~transitions:(Array.map Array.copy hmm.transitions)
      ~emissions:(Array.map Array.copy hmm.emissions)
  in
  let log_likelihood seq = log_likelihood model seq in
  let rec iterate iter prev_ll =
    if iter >= max_iter then model
    else
      let numerator_init = Array.make model.num_states 0.0 in
      let numerator_trans =
        Array.init model.num_states (fun _ -> Array.make model.num_states 0.0)
      in
      let numerator_emiss =
        Array.init model.num_states (fun _ ->
            Array.make model.num_observations 0.0)
      in
      let denom_trans = Array.make model.num_states 0.0 in
      let denom_emiss = Array.make model.num_states 0.0 in
      List.iter
        (fun obs ->
          let t_len = Array.length obs in
          if t_len > 0 then
            let alpha = forward model obs in
            let beta = backward model obs in
            let prob = Array.fold_left ( +. ) 0.0 alpha.(t_len - 1) in
            let gamma =
              Array.init t_len (fun t ->
                  Array.init model.num_states (fun i ->
                      let value = alpha.(t).(i) *. beta.(t).(i) in
                      if prob <= 0.0 then 0.0 else value /. prob))
            in
            let xi =
              Array.init (t_len - 1) (fun t ->
                  Array.init model.num_states (fun i ->
                      Array.init model.num_states (fun j ->
                          let value =
                            alpha.(t).(i)
                            *. model.transitions.(i).(j)
                            *. model.emissions.(j).(obs.(t + 1))
                            *. beta.(t + 1).(j)
                          in
                          if prob <= 0.0 then 0.0 else value /. prob)))
            in
            for i = 0 to model.num_states - 1 do
              numerator_init.(i) <- numerator_init.(i) +. gamma.(0).(i);
              for t = 0 to t_len - 1 do
                denom_emiss.(i) <- denom_emiss.(i) +. gamma.(t).(i);
                let obs_t = obs.(t) in
                numerator_emiss.(i).(obs_t) <-
                  numerator_emiss.(i).(obs_t) +. gamma.(t).(i)
              done;
              for t = 0 to t_len - 2 do
                denom_trans.(i) <- denom_trans.(i) +. gamma.(t).(i);
                for j = 0 to model.num_states - 1 do
                  numerator_trans.(i).(j) <-
                    numerator_trans.(i).(j) +. xi.(t).(i).(j)
                done
              done
            done)
        sequences;
      normalize_row numerator_init;
      for i = 0 to model.num_states - 1 do
        if denom_trans.(i) > 0.0 then
          for j = 0 to model.num_states - 1 do
            model.transitions.(i).(j) <-
              numerator_trans.(i).(j) /. denom_trans.(i)
          done;
        normalize_row model.transitions.(i);
        if denom_emiss.(i) > 0.0 then
          for k = 0 to model.num_observations - 1 do
            model.emissions.(i).(k) <-
              numerator_emiss.(i).(k) /. denom_emiss.(i)
          done;
        normalize_row model.emissions.(i)
      done;
      Array.blit numerator_init 0 model.init 0 model.num_states;
      let current_ll =
        List.fold_left (fun acc seq -> acc +. log_likelihood seq) 0.0 sequences
      in
      if Float.abs (current_ll -. prev_ll) < tol then model
      else iterate (iter + 1) current_ll
  in
  let initial_ll =
    List.fold_left (fun acc seq -> acc +. log_likelihood seq) 0.0 sequences
  in
  iterate 0 initial_ll
