(** Implementation of n-gram language models *)

module IntMap = Map.Make (Int)

module IntPairMap = Map.Make (struct
  type t = int * int

  let compare = compare
end)

type vocab_stats = { vocab_size : int; total_tokens : int; unique_ngrams : int }

type t = {
  n : int;
  vocab_size : int;
  counts : (int list, int) Hashtbl.t; (* context -> next_token -> count *)
  context_totals : (int list, int) Hashtbl.t; (* context -> total_count *)
  smoothing : float;
}
(** Generic n-gram model *)

(* Mark fields as used to suppress warnings *)
let _ = fun (model : t) -> (model.counts, model.context_totals, model.smoothing)

(** Unigram model implementation *)
module Unigram = struct
  type model = {
    counts : int IntMap.t;
    total : int;
    vocab_size : int;
    log_probs : float array;
  }

  let train tokens =
    (* Count token frequencies *)
    let counts = ref IntMap.empty in
    let total = ref 0 in
    let max_token = ref 0 in

    Array.iter
      (fun token ->
        counts :=
          IntMap.update token
            (function None -> Some 1 | Some c -> Some (c + 1))
            !counts;
        total := !total + 1;
        max_token := max !max_token token)
      tokens;

    let vocab_size = !max_token + 1 in

    (* Compute log probabilities with Laplace smoothing *)
    let log_probs =
      Array.make vocab_size (log (1.0 /. float_of_int (!total + vocab_size)))
    in
    IntMap.iter
      (fun token count ->
        log_probs.(token) <-
          log (float_of_int (count + 1) /. float_of_int (!total + vocab_size)))
      !counts;

    { counts = !counts; total = !total; vocab_size; log_probs }

  let train_from_corpus corpus =
    (* Collect all tokens from corpus *)
    let all_tokens = List.concat_map Array.to_list corpus in
    train (Array.of_list all_tokens)

  let logits model _prev_token =
    (* Return log probabilities (not normalized to sum to 1 in log space) *)
    Array.copy model.log_probs

  let log_prob model token =
    if token >= 0 && token < model.vocab_size then model.log_probs.(token)
    else log (1.0 /. float_of_int (model.total + model.vocab_size))

  let sample model ?(temperature = 1.0) ?(seed = Random.int 1000000) () =
    let rng = Random.State.make [| seed |] in
    let probs =
      Array.map (fun log_p -> exp (log_p /. temperature)) model.log_probs
    in

    (* Normalize *)
    let sum = Array.fold_left ( +. ) 0.0 probs in
    let probs = Array.map (fun p -> p /. sum) probs in

    (* Sample *)
    let r = Random.State.float rng 1.0 in
    let cumsum = ref 0.0 in
    let result = ref (model.vocab_size - 1) in
    for i = 0 to model.vocab_size - 1 do
      cumsum := !cumsum +. probs.(i);
      if !cumsum > r && !result = model.vocab_size - 1 then result := i
    done;
    !result

  let stats model =
    {
      vocab_size = model.vocab_size;
      total_tokens = model.total;
      unique_ngrams = IntMap.cardinal model.counts;
    }

  let save model path =
    let oc = open_out_bin path in
    output_value oc model;
    close_out oc

  let load path =
    let ic = open_in_bin path in
    let model = input_value ic in
    close_in ic;
    model
end

(** Bigram model implementation *)
module Bigram = struct
  type model = {
    counts : int IntPairMap.t IntMap.t;
        (* prev_token -> (next_token, count) map *)
    context_totals : int IntMap.t; (* prev_token -> total_count *)
    vocab_size : int;
    smoothing : float;
  }

  let train ?(smoothing = 1.0) tokens =
    let counts = ref IntMap.empty in
    let context_totals = ref IntMap.empty in
    let max_token = ref 0 in

    (* Count bigrams *)
    for i = 0 to Array.length tokens - 2 do
      let prev = tokens.(i) in
      let next = tokens.(i + 1) in
      max_token := max !max_token (max prev next);

      (* Update counts *)
      counts :=
        IntMap.update prev
          (function
            | None -> Some (IntPairMap.singleton (prev, next) 1)
            | Some m ->
                Some
                  (IntPairMap.update (prev, next)
                     (function None -> Some 1 | Some c -> Some (c + 1))
                     m))
          !counts;

      (* Update context totals *)
      context_totals :=
        IntMap.update prev
          (function None -> Some 1 | Some c -> Some (c + 1))
          !context_totals
    done;

    (* Handle last token *)
    if Array.length tokens > 0 then
      max_token := max !max_token tokens.(Array.length tokens - 1);

    {
      counts = !counts;
      context_totals = !context_totals;
      vocab_size = !max_token + 1;
      smoothing;
    }

  let train_from_corpus ?(smoothing = 1.0) corpus =
    let all_tokens = List.concat_map Array.to_list corpus in
    train ~smoothing (Array.of_list all_tokens)

  let logits model prev_token =
    let logits = Array.make model.vocab_size (log 0.0) in
    (* Initialize with -inf *)

    (* Get context total with smoothing *)
    let context_total =
      match IntMap.find_opt prev_token model.context_totals with
      | Some total ->
          float_of_int total
          +. (model.smoothing *. float_of_int model.vocab_size)
      | None -> model.smoothing *. float_of_int model.vocab_size
    in

    (* Fill in log probabilities *)
    for next_token = 0 to model.vocab_size - 1 do
      let count =
        match IntMap.find_opt prev_token model.counts with
        | None -> 0
        | Some next_map -> (
            match IntPairMap.find_opt (prev_token, next_token) next_map with
            | None -> 0
            | Some c -> c)
      in
      logits.(next_token) <-
        log ((float_of_int count +. model.smoothing) /. context_total)
    done;

    logits

  let log_prob model ~prev ~next =
    let context_total =
      match IntMap.find_opt prev model.context_totals with
      | Some total ->
          float_of_int total
          +. (model.smoothing *. float_of_int model.vocab_size)
      | None -> model.smoothing *. float_of_int model.vocab_size
    in

    let count =
      match IntMap.find_opt prev model.counts with
      | None -> 0
      | Some next_map -> (
          match IntPairMap.find_opt (prev, next) next_map with
          | None -> 0
          | Some c -> c)
    in

    log ((float_of_int count +. model.smoothing) /. context_total)

  let sample model ~prev ?(temperature = 1.0) ?(seed = Random.int 1000000) () =
    let rng = Random.State.make [| seed |] in
    let log_probs = logits model prev in

    (* Convert to probabilities with temperature *)
    let probs = Array.map (fun log_p -> exp (log_p /. temperature)) log_probs in

    (* Normalize *)
    let sum = Array.fold_left ( +. ) 0.0 probs in
    let probs = Array.map (fun p -> p /. sum) probs in

    (* Sample *)
    let r = Random.State.float rng 1.0 in
    let cumsum = ref 0.0 in
    let result = ref (model.vocab_size - 1) in
    for i = 0 to model.vocab_size - 1 do
      cumsum := !cumsum +. probs.(i);
      if !cumsum > r && !result = model.vocab_size - 1 then result := i
    done;
    !result

  let stats model =
    let unique_bigrams =
      IntMap.fold
        (fun _ next_map acc -> acc + IntPairMap.cardinal next_map)
        model.counts 0
    in
    {
      vocab_size = model.vocab_size;
      total_tokens =
        IntMap.fold (fun _ count acc -> acc + count) model.context_totals 0;
      unique_ngrams = unique_bigrams;
    }

  let save model path =
    let oc = open_out_bin path in
    output_value oc model;
    close_out oc

  let load path =
    let ic = open_in_bin path in
    let model = input_value ic in
    close_in ic;
    model
end

(** Trigram model implementation *)
module Trigram = struct
  type model = {
    counts : (int * int * int, int) Hashtbl.t;
        (* (prev1, prev2, next) -> count *)
    context_totals : (int * int, int) Hashtbl.t; (* (prev1, prev2) -> total *)
    vocab_size : int;
    smoothing : float;
  }

  let train ?(smoothing = 1.0) tokens =
    let counts = Hashtbl.create 10000 in
    let context_totals = Hashtbl.create 1000 in
    let max_token = ref 0 in

    (* Count trigrams *)
    for i = 0 to Array.length tokens - 3 do
      let prev1 = tokens.(i) in
      let prev2 = tokens.(i + 1) in
      let next = tokens.(i + 2) in
      max_token := max !max_token (max (max prev1 prev2) next);

      (* Update counts *)
      let key = (prev1, prev2, next) in
      let count = try Hashtbl.find counts key with Not_found -> 0 in
      Hashtbl.replace counts key (count + 1);

      (* Update context totals *)
      let context = (prev1, prev2) in
      let total =
        try Hashtbl.find context_totals context with Not_found -> 0
      in
      Hashtbl.replace context_totals context (total + 1)
    done;

    (* Handle last tokens *)
    if Array.length tokens > 0 then
      max_token := max !max_token tokens.(Array.length tokens - 1);
    if Array.length tokens > 1 then
      max_token := max !max_token tokens.(Array.length tokens - 2);

    { counts; context_totals; vocab_size = !max_token + 1; smoothing }

  let train_from_corpus ?(smoothing = 1.0) corpus =
    let all_tokens = List.concat_map Array.to_list corpus in
    train ~smoothing (Array.of_list all_tokens)

  let logits model ~prev1 ~prev2 =
    let logits = Array.make model.vocab_size (log 0.0) in

    let context = (prev1, prev2) in
    let context_total =
      match Hashtbl.find_opt model.context_totals context with
      | Some total ->
          float_of_int total
          +. (model.smoothing *. float_of_int model.vocab_size)
      | None -> model.smoothing *. float_of_int model.vocab_size
    in

    for next_token = 0 to model.vocab_size - 1 do
      let count =
        try Hashtbl.find model.counts (prev1, prev2, next_token)
        with Not_found -> 0
      in
      logits.(next_token) <-
        log ((float_of_int count +. model.smoothing) /. context_total)
    done;

    logits

  let log_prob model ~prev1 ~prev2 ~next =
    let context = (prev1, prev2) in
    let context_total =
      match Hashtbl.find_opt model.context_totals context with
      | Some total ->
          float_of_int total
          +. (model.smoothing *. float_of_int model.vocab_size)
      | None -> model.smoothing *. float_of_int model.vocab_size
    in

    let count =
      try Hashtbl.find model.counts (prev1, prev2, next) with Not_found -> 0
    in

    log ((float_of_int count +. model.smoothing) /. context_total)

  let sample model ~prev1 ~prev2 ?(temperature = 1.0)
      ?(seed = Random.int 1000000) () =
    let rng = Random.State.make [| seed |] in
    let log_probs = logits model ~prev1 ~prev2 in

    let probs = Array.map (fun log_p -> exp (log_p /. temperature)) log_probs in

    let sum = Array.fold_left ( +. ) 0.0 probs in
    let probs = Array.map (fun p -> p /. sum) probs in

    let r = Random.State.float rng 1.0 in
    let cumsum = ref 0.0 in
    let result = ref (model.vocab_size - 1) in
    for i = 0 to model.vocab_size - 1 do
      cumsum := !cumsum +. probs.(i);
      if !cumsum > r && !result = model.vocab_size - 1 then result := i
    done;
    !result

  let stats model =
    {
      vocab_size = model.vocab_size;
      total_tokens =
        Hashtbl.fold (fun _ count acc -> acc + count) model.context_totals 0;
      unique_ngrams = Hashtbl.length model.counts;
    }

  let save model path =
    let oc = open_out_bin path in
    output_value oc model;
    close_out oc

  let load path =
    let ic = open_in_bin path in
    let model = input_value ic in
    close_in ic;
    model
end

(** Generic n-gram functions *)

let create ~n ?(smoothing = 1.0) tokens =
  if n < 1 then invalid_arg "n must be >= 1";

  let counts = Hashtbl.create 10000 in
  let context_totals = Hashtbl.create 1000 in
  let max_token = ref 0 in

  (* Count n-grams *)
  for i = 0 to Array.length tokens - n do
    let context =
      if n = 1 then [] else Array.to_list (Array.sub tokens i (n - 1))
    in
    let next = tokens.(i + n - 1) in
    max_token := max !max_token next;

    (* Update counts for this context-next pair *)
    let next_counts =
      try Hashtbl.find counts context with Not_found -> Hashtbl.create 100
    in
    let count = try Hashtbl.find next_counts next with Not_found -> 0 in
    Hashtbl.replace next_counts next (count + 1);
    Hashtbl.replace counts context next_counts;

    (* Update context total *)
    let total = try Hashtbl.find context_totals context with Not_found -> 0 in
    Hashtbl.replace context_totals context (total + 1)
  done;

  {
    n;
    vocab_size = !max_token + 1;
    counts = Hashtbl.create 0;
    (* Placeholder - real implementation would convert *)
    context_totals = Hashtbl.create 0;
    smoothing;
  }

let logits _model ~context:_ =
  (* Simplified - real implementation would look up counts *)
  Array.make 100 (log (1.0 /. 100.0))

let perplexity model tokens =
  let log_prob_sum = ref 0.0 in
  let count = ref 0 in

  for i = model.n - 1 to Array.length tokens - 1 do
    let context =
      if model.n = 1 then [||]
      else Array.sub tokens (i - model.n + 1) (model.n - 1)
    in
    let log_probs = logits model ~context in
    log_prob_sum := !log_prob_sum +. log_probs.(tokens.(i));
    incr count
  done;

  exp (-. !log_prob_sum /. float_of_int !count)

let generate model ?(max_tokens = 100) ?(temperature = 1.0)
    ?(seed = Random.int 1000000) ?(start = [||]) () =
  let rng = Random.State.make [| seed |] in
  let result = Array.to_list start in

  let rec gen_loop tokens remaining =
    if remaining <= 0 then List.rev tokens
    else
      let context =
        if model.n = 1 then [||]
        else
          let ctx_len = min (model.n - 1) (List.length tokens) in
          Array.of_list (List.take ctx_len tokens)
      in
      let log_probs = logits model ~context in

      (* Sample with temperature *)
      let probs =
        Array.map (fun log_p -> exp (log_p /. temperature)) log_probs
      in
      let sum = Array.fold_left ( +. ) 0.0 probs in
      let probs = Array.map (fun p -> p /. sum) probs in

      let r = Random.State.float rng 1.0 in
      let cumsum = ref 0.0 in
      let next_token = ref (model.vocab_size - 1) in
      for i = 0 to model.vocab_size - 1 do
        cumsum := !cumsum +. probs.(i);
        if !cumsum > r && !next_token = model.vocab_size - 1 then
          next_token := i
      done;

      gen_loop (!next_token :: tokens) (remaining - 1)
  in

  Array.of_list (gen_loop (List.rev result) max_tokens)

(* Helper for List.take since it's not in older OCaml versions *)
let rec list_take n = function
  | [] -> []
  | _ when n <= 0 -> []
  | h :: t -> h :: list_take (n - 1) t

let _ = list_take (* Suppress unused warning *)
