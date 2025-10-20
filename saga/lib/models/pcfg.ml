type nonterminal = int
type terminal = int
type production = Binary of nonterminal * nonterminal | Unary of terminal
type rule = { lhs : nonterminal; rhs : production; prob : float }

type t = {
  start : nonterminal;
  num_nonterminals : int;
  num_terminals : int;
  binary_rules :
    (nonterminal * nonterminal, (nonterminal * float) list) Hashtbl.t;
  unary_rules : (terminal, (nonterminal * float) list) Hashtbl.t;
}

let epsilon = 1e-12

let create ~start ~num_nonterminals ~num_terminals rules =
  if num_nonterminals <= 0 then invalid_arg "PCFG.create: nonterminals <= 0";
  if num_terminals <= 0 then invalid_arg "PCFG.create: terminals <= 0";
  let grouped_binary = Hashtbl.create 32 in
  let grouped_unary = Hashtbl.create 32 in
  List.iter
    (fun { lhs; rhs; prob } ->
      match rhs with
      | Binary (b, c) ->
          let key = (b, c) in
          let existing =
            Hashtbl.find_opt grouped_binary key |> Option.value ~default:[]
          in
          Hashtbl.replace grouped_binary key ((lhs, prob) :: existing)
      | Unary term ->
          let existing =
            Hashtbl.find_opt grouped_unary term |> Option.value ~default:[]
          in
          Hashtbl.replace grouped_unary term ((lhs, prob) :: existing))
    rules;
  Hashtbl.iter
    (fun key lst ->
      let sum = List.fold_left (fun acc (_, p) -> acc +. p) 0.0 lst in
      let sum = if sum <= 0.0 then 1.0 else sum in
      Hashtbl.replace grouped_binary key
        (List.map (fun (lhs, p) -> (lhs, p /. sum)) lst))
    grouped_binary;
  Hashtbl.iter
    (fun term lst ->
      let sum = List.fold_left (fun acc (_, p) -> acc +. p) 0.0 lst in
      let sum = if sum <= 0.0 then 1.0 else sum in
      Hashtbl.replace grouped_unary term
        (List.map (fun (lhs, p) -> (lhs, p /. sum)) lst))
    grouped_unary;
  {
    start;
    num_nonterminals;
    num_terminals;
    binary_rules = grouped_binary;
    unary_rules = grouped_unary;
  }

let start_symbol t = t.start
let num_nonterminals t = t.num_nonterminals
let num_terminals t = t.num_terminals

let inside t sentence =
  let n = Array.length sentence in
  let chart = Array.make_matrix (n + 1) (n + 1) [||] in
  for i = 0 to n - 1 do
    let term = sentence.(i) in
    let cell = Array.make t.num_nonterminals 0.0 in
    (match Hashtbl.find_opt t.unary_rules term with
    | Some lst -> List.iter (fun (lhs, p) -> cell.(lhs) <- p) lst
    | None -> ());
    chart.(i).(i + 1) <- cell
  done;
  for span = 2 to n do
    for i = 0 to n - span do
      let j = i + span in
      let cell = Array.make t.num_nonterminals 0.0 in
      for k = i + 1 to j - 1 do
        let left = chart.(i).(k) in
        let right = chart.(k).(j) in
        Hashtbl.iter
          (fun (b, c) lst ->
            let prob = left.(b) *. right.(c) in
            if prob > 0.0 then
              List.iter
                (fun (lhs, p) -> cell.(lhs) <- cell.(lhs) +. (p *. prob))
                lst)
          t.binary_rules
      done;
      chart.(i).(j) <- cell
    done
  done;
  chart

let outside t sentence inside_chart =
  let n = Array.length sentence in
  let outside_chart = Array.make_matrix (n + 1) (n + 1) [||] in
  for i = 0 to n do
    for j = 0 to n do
      outside_chart.(i).(j) <- Array.make t.num_nonterminals 0.0
    done
  done;
  outside_chart.(0).(n).(t.start) <- 1.0;
  for span = n downto 1 do
    for i = 0 to n - span do
      let j = i + span in
      let outside_cell = outside_chart.(i).(j) in
      for k = i + 1 to j - 1 do
        let left_inside = inside_chart.(i).(k) in
        let right_inside = inside_chart.(k).(j) in
        Hashtbl.iter
          (fun (b, c) lst ->
            let left_score = left_inside.(b) in
            let right_score = right_inside.(c) in
            if left_score > 0.0 && right_score > 0.0 then
              List.iter
                (fun (lhs, rule_prob) ->
                  let parent_out = outside_cell.(lhs) *. rule_prob in
                  if parent_out > 0.0 then (
                    outside_chart.(i).(k).(b) <-
                      outside_chart.(i).(k).(b) +. (parent_out *. right_score);
                    outside_chart.(k).(j).(c) <-
                      outside_chart.(k).(j).(c) +. (parent_out *. left_score)))
                lst)
          t.binary_rules
      done
    done
  done;
  outside_chart

let log_probability t sentence =
  let inside_chart = inside t sentence in
  let total = inside_chart.(0).(Array.length sentence).(t.start) in
  log (max total epsilon)

let viterbi t sentence =
  let n = Array.length sentence in
  let score =
    Array.make_matrix (n + 1) (n + 1)
      (Array.make t.num_nonterminals Float.neg_infinity)
  in
  let back =
    Array.make_matrix (n + 1) (n + 1)
      (Array.make t.num_nonterminals (-1, -1, -1))
  in
  for i = 0 to n - 1 do
    let term = sentence.(i) in
    let cell = Array.make t.num_nonterminals Float.neg_infinity in
    (match Hashtbl.find_opt t.unary_rules term with
    | Some lst ->
        List.iter (fun (lhs, p) -> cell.(lhs) <- log (max p epsilon)) lst
    | None -> ());
    score.(i).(i + 1) <- cell
  done;
  for span = 2 to n do
    for i = 0 to n - span do
      let j = i + span in
      let cell = Array.make t.num_nonterminals Float.neg_infinity in
      let back_cell = Array.make t.num_nonterminals (-1, -1, -1) in
      for k = i + 1 to j - 1 do
        let left = score.(i).(k) in
        let right = score.(k).(j) in
        Hashtbl.iter
          (fun (b, c) lst ->
            let left_score = left.(b) in
            let right_score = right.(c) in
            if
              left_score > Float.neg_infinity
              && right_score > Float.neg_infinity
            then
              List.iter
                (fun (lhs, prob) ->
                  let candidate =
                    left_score +. right_score +. log (max prob epsilon)
                  in
                  if candidate > cell.(lhs) then (
                    cell.(lhs) <- candidate;
                    back_cell.(lhs) <- (k, b, c)))
                lst)
          t.binary_rules
      done;
      score.(i).(j) <- cell;
      back.(i).(j) <- back_cell
    done
  done;
  back

let rec sample_tree t inside_chart i j lhs rng acc =
  if j = i + 1 then lhs :: acc
  else
    let rec choose_split splits total r =
      match splits with
      | [] -> failwith "sample_tree: no splits"
      | (k, b, c, weight) :: tl ->
          if r <= weight then (k, b, c)
          else choose_split tl (total -. weight) (r -. weight)
    in
    let splits = ref [] in
    let total = ref 0.0 in
    for k = i + 1 to j - 1 do
      Hashtbl.iter
        (fun (b, c) lst ->
          let left = inside_chart.(i).(k).(b) in
          let right = inside_chart.(k).(j).(c) in
          if left > 0.0 && right > 0.0 then
            List.iter
              (fun (lhs_rule, p) ->
                if lhs_rule = lhs then (
                  let weight = p *. left *. right in
                  splits := (k, b, c, weight) :: !splits;
                  total := !total +. weight))
              lst)
        t.binary_rules
    done;
    if !total <= 0.0 then lhs :: acc
    else
      let r = Random.State.float rng !total in
      let k, b, c = choose_split !splits !total r in
      let acc1 = sample_tree t inside_chart i k b rng (lhs :: acc) in
      sample_tree t inside_chart k j c rng acc1

let sample t sentence =
  let inside_chart = inside t sentence in
  let total = inside_chart.(0).(Array.length sentence).(t.start) in
  if total <= 0.0 then None
  else
    let rng = Random.State.make [| Random.bits () |] in
    Some
      (List.rev
         (sample_tree t inside_chart 0 (Array.length sentence) t.start rng []))

let inside_outside ?(tol = 1e-4) ?(max_iter = 50) grammar sentences =
  let current = ref grammar in
  let previous_ll = ref Float.neg_infinity in
  let finished = ref false in
  let iter = ref 0 in
  while (not !finished) && !iter < max_iter do
    incr iter;
    let accum_binary = Hashtbl.create 64 in
    let accum_unary = Hashtbl.create 64 in
    let z = ref 0.0 in
    List.iter
      (fun sentence ->
        let inside_chart = inside !current sentence in
        let outside_chart = outside !current sentence inside_chart in
        let n = Array.length sentence in
        let logprob = log_probability !current sentence in
        let weight = exp logprob in
        z := !z +. weight;
        for i = 0 to n - 1 do
          let term = sentence.(i) in
          let cell_inside = inside_chart.(i).(i + 1) in
          let cell_outside = outside_chart.(i).(i + 1) in
          Array.iteri
            (fun lhs inside_val ->
              let contrib = cell_outside.(lhs) *. inside_val *. weight in
              if contrib > 0.0 then
                let existing =
                  Hashtbl.find_opt accum_unary term |> Option.value ~default:[]
                in
                Hashtbl.replace accum_unary term ((lhs, contrib) :: existing))
            cell_inside
        done;
        for span = 2 to n do
          for i = 0 to n - span do
            let j = i + span in
            let parent_outside = outside_chart.(i).(j) in
            for k = i + 1 to j - 1 do
              let left_inside = inside_chart.(i).(k) in
              let right_inside = inside_chart.(k).(j) in
              Hashtbl.iter
                (fun (b, c) lst ->
                  let left = left_inside.(b) in
                  let right = right_inside.(c) in
                  if left > 0.0 && right > 0.0 then
                    List.iter
                      (fun (lhs, p) ->
                        let contrib =
                          parent_outside.(lhs) *. p *. left *. right *. weight
                        in
                        if contrib > 0.0 then
                          let key = (b, c) in
                          let existing =
                            Hashtbl.find_opt accum_binary key
                            |> Option.value ~default:[]
                          in
                          Hashtbl.replace accum_binary key
                            ((lhs, contrib) :: existing))
                      lst)
                !current.binary_rules
            done
          done
        done)
      sentences;
    let normalize_acc accum =
      Hashtbl.fold
        (fun key lst acc ->
          let sum = List.fold_left (fun acc (_, v) -> acc +. v) 0.0 lst in
          let sum = if sum <= 0.0 then 1.0 else sum in
          Hashtbl.add acc key (List.map (fun (lhs, v) -> (lhs, v /. sum)) lst);
          acc)
        accum
        (Hashtbl.create (Hashtbl.length accum))
    in
    let new_binary = normalize_acc accum_binary in
    let new_unary = normalize_acc accum_unary in
    current :=
      { !current with binary_rules = new_binary; unary_rules = new_unary };
    let ll =
      List.fold_left
        (fun acc sent -> acc +. log_probability !current sent)
        0.0 sentences
    in
    if Float.abs (ll -. !previous_ll) < tol then (
      finished := true;
      previous_ll := ll)
    else previous_ll := ll
  done;
  !current
