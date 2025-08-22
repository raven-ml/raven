(** Implementation of text generation and sampling utilities *)

type logits_fn = int -> float array

type config = {
  max_tokens : int;
  temperature : float;
  top_k : int option;
  top_p : float option;
  seed : int option;
}

let default_config = {
  max_tokens = 100;
  temperature = 1.0;
  top_k = None;
  top_p = None;
  seed = None;
}

(** Apply temperature scaling to logits *)
let apply_temperature logits temperature =
  if temperature = 1.0 then logits
  else Array.map (fun x -> x /. temperature) logits

(** Convert logits to probabilities using softmax *)
let softmax logits =
  let max_logit = Array.fold_left max neg_infinity logits in
  let exp_logits = Array.map (fun x -> exp (x -. max_logit)) logits in
  let sum = Array.fold_left (+.) 0.0 exp_logits in
  Array.map (fun x -> x /. sum) exp_logits

(** Apply top-k filtering *)
let apply_top_k probs k =
  let n = Array.length probs in
  if k >= n then probs
  else
    (* Get indices sorted by probability (descending) *)
    let indices = Array.init n (fun i -> i) in
    Array.sort (fun i j -> compare probs.(j) probs.(i)) indices;
    
    (* Zero out all but top k *)
    let filtered = Array.copy probs in
    for i = k to n - 1 do
      filtered.(indices.(i)) <- 0.0
    done;
    
    (* Renormalize *)
    let sum = Array.fold_left (+.) 0.0 filtered in
    if sum > 0.0 then
      Array.map (fun x -> x /. sum) filtered
    else
      filtered

(** Apply nucleus (top-p) sampling *)
let apply_top_p probs p =
  (* Get indices sorted by probability (descending) *)
  let n = Array.length probs in
  let indices = Array.init n (fun i -> i) in
  Array.sort (fun i j -> compare probs.(j) probs.(i)) indices;
  
  (* Find smallest set with cumulative probability > p *)
  let cumsum = ref 0.0 in
  let cutoff = ref n in
  for i = 0 to n - 1 do
    cumsum := !cumsum +. probs.(indices.(i));
    if !cumsum > p && !cutoff = n then
      cutoff := i + 1
  done;
  
  (* Zero out tokens not in nucleus *)
  let filtered = Array.copy probs in
  for i = !cutoff to n - 1 do
    filtered.(indices.(i)) <- 0.0
  done;
  
  (* Renormalize *)
  let sum = Array.fold_left (+.) 0.0 filtered in
  if sum > 0.0 then
    Array.map (fun x -> x /. sum) filtered
  else
    filtered

(** Sample from a probability distribution *)
let sample_from_probs probs rng =
  let r = Random.State.float rng 1.0 in
  let cumsum = ref 0.0 in
  let result = ref (Array.length probs - 1) in
  for i = 0 to Array.length probs - 1 do
    cumsum := !cumsum +. probs.(i);
    if !cumsum > r && !result = Array.length probs - 1 then
      result := i
  done;
  !result

let greedy logits =
  let max_idx = ref 0 in
  let max_val = ref logits.(0) in
  for i = 1 to Array.length logits - 1 do
    if logits.(i) > !max_val then (
      max_val := logits.(i);
      max_idx := i
    )
  done;
  !max_idx

let sample_token ?temperature ?top_k ?top_p ?seed logits =
  let temperature = Option.value temperature ~default:1.0 in
  
  (* Initialize RNG *)
  let rng = match seed with
    | Some s -> Random.State.make [| s |]
    | None -> Random.State.make_self_init ()
  in
  
  (* Apply temperature *)
  let logits = apply_temperature logits temperature in
  
  (* Convert to probabilities *)
  let probs = softmax logits in
  
  (* Apply top-k filtering if specified *)
  let probs = match top_k with
    | Some k -> apply_top_k probs k
    | None -> probs
  in
  
  (* Apply top-p filtering if specified *)
  let probs = match top_p with
    | Some p -> apply_top_p probs p
    | None -> probs
  in
  
  (* Sample from distribution *)
  sample_from_probs probs rng

let generate ?max_tokens ?temperature ?top_k ?top_p ?seed ?start ~logits_fn ~tokenizer () =
  let max_tokens = Option.value max_tokens ~default:100 in
  let temperature = Option.value temperature ~default:1.0 in
  
  (* Initialize with start tokens if provided *)
  let tokens = match start with
    | Some text -> Array.to_list (tokenizer text)
    | None -> []
  in
  
  (* Generate tokens *)
  let rec generate_loop tokens remaining =
    if remaining <= 0 then
      List.rev tokens
    else
      let prev_token = match tokens with
        | [] -> 0  (* Start token *)
        | h :: _ -> h
      in
      let logits = logits_fn prev_token in
      let next_token = sample_token ~temperature ?top_k ?top_p ?seed logits in
      generate_loop (next_token :: tokens) (remaining - 1)
  in
  
  let generated = generate_loop (List.rev tokens) max_tokens in
  Array.of_list generated

let generate_text ?max_tokens ?temperature ?top_k ?top_p ?seed ?start 
    ~logits_fn ~tokenizer ~decoder () =
  let tokens = generate ?max_tokens ?temperature ?top_k ?top_p ?seed ?start 
    ~logits_fn ~tokenizer () in
  decoder tokens