(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Modern text generation with composable logits processors *)

(** String.contains_substring implementation *)
module String = struct
  include String

  let contains_substring s sub =
    let len_s = String.length s in
    let len_sub = String.length sub in
    if len_sub = 0 then true
    else if len_sub > len_s then false
    else
      let rec check i =
        if i > len_s - len_sub then false
        else if String.sub s i len_sub = sub then true
        else check (i + 1)
      in
      check 0
end

(** List.take implementation *)
module List = struct
  include List

  let take n lst =
    let rec aux n acc = function
      | [] -> List.rev acc
      | _ when n <= 0 -> List.rev acc
      | h :: t -> aux (n - 1) (h :: acc) t
    in
    aux n [] lst
end

(** {1 Core Types} *)

type logits = float array
type token_ids = int list

(** {2 Logits Processors} *)

type logits_processor = {
  name : string;
  process : prompt_length:int -> token_ids -> logits -> logits;
}

type logits_processor_list = logits_processor list

(** {2 Stopping Criteria} *)

type stopping_criterion = {
  name : string;
  should_stop : prompt_length:int -> start_time:float -> token_ids -> bool;
}

type stopping_criteria_list = stopping_criterion list

(** {2 Generation Configuration} *)

type generation_config = {
  max_length : int;
  max_new_tokens : int option;
  min_length : int;
  min_new_tokens : int;
  do_sample : bool;
  early_stopping : bool;
  num_beams : int;
  temperature : float;
  top_k : int;
  top_p : float;
  repetition_penalty : float;
  length_penalty : float;
  no_repeat_ngram_size : int;
  encoder_repetition_penalty : float;
  bad_words_ids : int list list;
  force_words_ids : int list list;
  num_return_sequences : int;
  output_scores : bool;
  output_attentions : bool;
  output_hidden_states : bool;
  pad_token_id : int option;
  bos_token_id : int option;
  eos_token_id : int option;
  eos_token_ids : int list;
}

(** Default generation configuration *)
let default =
  {
    max_length = 100;
    max_new_tokens = None;
    min_length = 0;
    min_new_tokens = 0;
    do_sample = false;
    early_stopping = false;
    num_beams = 1;
    temperature = 1.0;
    top_k = 0;
    (* 0 means no limit *)
    top_p = 1.0;
    repetition_penalty = 1.0;
    length_penalty = 1.0;
    no_repeat_ngram_size = 0;
    encoder_repetition_penalty = 1.0;
    bad_words_ids = [];
    force_words_ids = [];
    num_return_sequences = 1;
    output_scores = false;
    output_attentions = false;
    output_hidden_states = false;
    pad_token_id = None;
    bos_token_id = None;
    eos_token_id = None;
    eos_token_ids = [];
  }

(** {2 Builder Pattern for Configuration} *)

let with_temperature temperature config = { config with temperature }
let with_top_k top_k config = { config with top_k }
let with_top_p top_p config = { config with top_p }

let with_repetition_penalty repetition_penalty config =
  { config with repetition_penalty }

let with_max_length max_length config = { config with max_length }

let with_max_new_tokens max_new_tokens config =
  { config with max_new_tokens = Some max_new_tokens }

let with_min_length min_length config = { config with min_length }
let with_min_new_tokens min_new_tokens config = { config with min_new_tokens }

let with_no_repeat_ngram no_repeat_ngram_size config =
  { config with no_repeat_ngram_size }

let with_num_beams num_beams config = { config with num_beams }
let with_do_sample do_sample config = { config with do_sample }

(** {2 Preset Configurations} *)

let creative_writing =
  default |> with_do_sample true |> with_temperature 0.8 |> with_top_p 0.9
  |> with_repetition_penalty 1.2
  |> with_no_repeat_ngram 3 |> with_max_new_tokens 512

let chat =
  default |> with_do_sample true |> with_temperature 0.7 |> with_top_p 0.95
  |> with_repetition_penalty 1.1
  |> with_max_new_tokens 512

let code_generation =
  default |> with_do_sample true |> with_temperature 0.2 |> with_top_k 5
  |> with_repetition_penalty 1.0
  |> with_max_new_tokens 1024

let factual =
  default |> with_do_sample true |> with_temperature 0.3 |> with_top_k 10
  |> with_repetition_penalty 1.1
  |> with_max_new_tokens 256

let from_preset = function
  | "creative_writing" -> creative_writing
  | "chat" -> chat
  | "code_generation" -> code_generation
  | "factual" -> factual
  | _ -> default

(** {1 Logits Processors} *)

let neg_infinity = -1e10

(** Temperature scaling processor *)
let temperature_warper ~temperature =
  {
    name = Printf.sprintf "temperature(%.2f)" temperature;
    process =
      (fun ~prompt_length:_ _ logits ->
        if temperature = 1.0 then logits
        else Array.map (fun x -> x /. temperature) logits);
  }

(** Top-k filtering processor *)
let top_k_warper ~k =
  {
    name = Printf.sprintf "top_k(%d)" k;
    process =
      (fun ~prompt_length:_ _ logits ->
        if k <= 0 then logits
        else
          let sorted = Array.mapi (fun i v -> (i, v)) logits in
          Array.sort (fun (_, a) (_, b) -> compare b a) sorted;
          let result = Array.make (Array.length logits) neg_infinity in
          for i = 0 to min k (Array.length sorted) - 1 do
            let idx, value = sorted.(i) in
            result.(idx) <- value
          done;
          result);
  }

(** Top-p (nucleus) filtering processor *)
let top_p_warper ~p =
  {
    name = Printf.sprintf "top_p(%.2f)" p;
    process =
      (fun ~prompt_length:_ _ logits ->
        if p >= 1.0 then logits
        else
          (* Convert to probabilities *)
          let max_logit = Array.fold_left max neg_infinity logits in
          let exp_logits = Array.map (fun x -> exp (x -. max_logit)) logits in
          let sum_exp = Array.fold_left ( +. ) 0.0 exp_logits in
          let probs = Array.map (fun x -> x /. sum_exp) exp_logits in

          (* Sort by probability *)
          let sorted = Array.mapi (fun i v -> (i, v)) probs in
          Array.sort (fun (_, a) (_, b) -> compare b a) sorted;

          (* Find cutoff *)
          let cumsum = ref 0.0 in
          let cutoff_idx = ref (Array.length sorted) in
          (try
             for i = 0 to Array.length sorted - 1 do
               cumsum := !cumsum +. snd sorted.(i);
               if !cumsum > p && i > 0 then (
                 cutoff_idx := i;
                 raise Exit)
             done
           with Exit -> ());

          (* Apply cutoff *)
          let result = Array.make (Array.length logits) neg_infinity in
          for i = 0 to !cutoff_idx - 1 do
            let idx, _ = sorted.(i) in
            result.(idx) <- logits.(idx)
          done;
          result);
  }

(** Repetition penalty processor *)
let repetition_penalty ~penalty =
  {
    name = Printf.sprintf "repetition_penalty(%.2f)" penalty;
    process =
      (fun ~prompt_length:_ previous_tokens logits ->
        if penalty = 1.0 then logits
        else
          let result = Array.copy logits in
          List.iter
            (fun token_id ->
              if token_id < Array.length result then
                let score = result.(token_id) in
                result.(token_id) <-
                  (if score < 0.0 then score *. penalty else score /. penalty))
            previous_tokens;
          result);
  }

(** No repeat n-gram processor *)
let no_repeat_ngram ~ngram_size =
  {
    name = Printf.sprintf "no_repeat_ngram(%d)" ngram_size;
    process =
      (fun ~prompt_length:_ previous_tokens logits ->
        if ngram_size <= 0 || List.length previous_tokens < ngram_size - 1 then
          logits
        else
          let result = Array.copy logits in
          let check_ngram tokens =
            if List.length tokens >= ngram_size then
              let prefix = List.rev (List.tl (List.rev tokens)) in
              let last = List.hd (List.rev tokens) in
              let prev_ngrams =
                let rec find_ngrams acc toks =
                  if List.length toks < ngram_size then acc
                  else
                    let ngram = List.take ngram_size toks in
                    let prefix' = List.rev (List.tl (List.rev ngram)) in
                    if prefix = prefix' then List.hd (List.rev ngram) :: acc
                    else find_ngrams acc (List.tl toks)
                in
                find_ngrams [] previous_tokens
              in
              if List.mem last prev_ngrams then result.(last) <- neg_infinity
          in
          check_ngram
            (List.rev (List.take ngram_size (List.rev previous_tokens)));
          result);
  }

(** Min length processor *)
let min_length ~min_length ~eos_token_ids =
  {
    name = Printf.sprintf "min_length(%d)" min_length;
    process =
      (fun ~prompt_length:_ tokens logits ->
        if List.length tokens >= min_length then logits
        else
          let result = Array.copy logits in
          List.iter
            (fun eos_id ->
              if eos_id < Array.length result then
                result.(eos_id) <- neg_infinity)
            eos_token_ids;
          result);
  }

(** Min new tokens processor *)
let min_new_tokens ~min_new_tokens ~eos_token_ids =
  {
    name = Printf.sprintf "min_new_tokens(%d)" min_new_tokens;
    process =
      (fun ~prompt_length tokens logits ->
        let new_tokens = List.length tokens - prompt_length in
        if new_tokens >= min_new_tokens then logits
        else
          let result = Array.copy logits in
          List.iter
            (fun eos_id ->
              if eos_id < Array.length result then
                result.(eos_id) <- neg_infinity)
            eos_token_ids;
          result);
  }

(** Bad words processor *)
let bad_words ~bad_words_ids =
  {
    name = "bad_words";
    process =
      (fun ~prompt_length:_ tokens logits ->
        let result = Array.copy logits in
        List.iter
          (fun bad_sequence ->
            let seq_len = List.length bad_sequence in
            if seq_len > 0 && List.length tokens >= seq_len - 1 then
              let recent =
                List.rev (List.take (seq_len - 1) (List.rev tokens))
              in
              let prefix = List.rev (List.tl (List.rev bad_sequence)) in
              if recent = prefix then
                let bad_token = List.hd (List.rev bad_sequence) in
                if bad_token < Array.length result then
                  result.(bad_token) <- neg_infinity)
          bad_words_ids;
        result);
  }

(** Force words processor *)
let force_words ~force_words_ids ~iteration =
  {
    name = "force_words";
    process =
      (fun ~prompt_length:_ _ logits ->
        if iteration >= List.length force_words_ids then logits
        else
          let forced_tokens = List.nth force_words_ids iteration in
          let result = Array.make (Array.length logits) neg_infinity in
          List.iter
            (fun token_id ->
              if token_id < Array.length result then
                result.(token_id) <- logits.(token_id))
            forced_tokens;
          result);
  }

(** Custom processor *)
let custom ~name ~process = { name; process }

(** {1 Stopping Criteria} *)

(** Max length stopping criterion *)
let max_length_criteria ~max_length =
  {
    name = Printf.sprintf "max_length(%d)" max_length;
    should_stop =
      (fun ~prompt_length:_ ~start_time:_ tokens ->
        List.length tokens >= max_length);
  }

(** Max new tokens stopping criterion *)
let max_new_tokens_criteria ~max_new_tokens =
  {
    name = Printf.sprintf "max_new_tokens(%d)" max_new_tokens;
    should_stop =
      (fun ~prompt_length ~start_time:_ tokens ->
        List.length tokens - prompt_length >= max_new_tokens);
  }

(** EOS token stopping criterion *)
let eos_token_criteria ~eos_token_ids =
  {
    name = "eos_token";
    should_stop =
      (fun ~prompt_length:_ ~start_time:_ tokens ->
        match List.rev tokens with
        | [] -> false
        | last :: _ -> List.mem last eos_token_ids);
  }

(** Max time stopping criterion *)
let max_time_criteria ~max_time =
  {
    name = Printf.sprintf "max_time(%.1fs)" max_time;
    should_stop =
      (fun ~prompt_length:_ ~start_time tokens ->
        let _ = tokens in
        (* tokens is used for API consistency *)
        Unix.gettimeofday () -. start_time > max_time);
  }

(** Stop strings criterion *)
let stop_strings_criteria ~stop_strings ~decoder =
  {
    name = "stop_strings";
    should_stop =
      (fun ~prompt_length:_ ~start_time:_ tokens ->
        let text = decoder tokens in
        List.exists
          (fun stop_str -> String.contains_substring text stop_str)
          stop_strings);
  }

(** Custom stopping criterion *)
let custom_criteria ~name ~should_stop = { name; should_stop }

(** {1 Utilities} *)

(** Apply all processors in the list *)
let apply_processors ~processors ~prompt_length ~tokens ~logits =
  List.fold_left
    (fun acc processor -> processor.process ~prompt_length tokens acc)
    logits processors

(** Check all stopping criteria *)
let check_stopping ~criteria ~prompt_length ~start_time ~tokens =
  List.exists
    (fun criterion -> criterion.should_stop ~prompt_length ~start_time tokens)
    criteria

(** {1 Main Generation Functions} *)

type generation_output = {
  sequences : int list list;
  scores : float list list option;
  attentions : float array list option;
  hidden_states : float array list option;
}

(** Helper: sample from logits *)
let sample_from_logits logits =
  (* Find max for numerical stability *)
  let max_logit = Array.fold_left max neg_infinity logits in
  let exp_logits = Array.map (fun x -> exp (x -. max_logit)) logits in
  let sum_exp = Array.fold_left ( +. ) 0.0 exp_logits in
  let probs = Array.map (fun x -> x /. sum_exp) exp_logits in

  (* Sample *)
  let r = Random.float 1.0 in
  let cumsum = ref 0.0 in
  let result = ref 0 in
  (try
     for i = 0 to Array.length probs - 1 do
       cumsum := !cumsum +. probs.(i);
       if !cumsum > r then (
         result := i;
         raise Exit)
     done
   with Exit -> ());
  !result

(** Helper: greedy selection *)
let argmax logits =
  let max_idx = ref 0 in
  let max_val = ref logits.(0) in
  for i = 1 to Array.length logits - 1 do
    if logits.(i) > !max_val then (
      max_val := logits.(i);
      max_idx := i)
  done;
  !max_idx

let generate ~model ?(input_ids = []) ?(generation_config = default)
    ?(logits_processor = []) ?(stopping_criteria = []) () =
  let start_time = Unix.gettimeofday () in
  let prompt_length = List.length input_ids in

  (* Build processor list from config *)
  let processors =
    let base_processors = [] in
    let base_processors =
      if generation_config.temperature <> 1.0 then
        temperature_warper ~temperature:generation_config.temperature
        :: base_processors
      else base_processors
    in
    let base_processors =
      if generation_config.top_k > 0 then
        top_k_warper ~k:generation_config.top_k :: base_processors
      else base_processors
    in
    let base_processors =
      if generation_config.top_p < 1.0 then
        top_p_warper ~p:generation_config.top_p :: base_processors
      else base_processors
    in
    let base_processors =
      if generation_config.repetition_penalty <> 1.0 then
        repetition_penalty ~penalty:generation_config.repetition_penalty
        :: base_processors
      else base_processors
    in
    let base_processors =
      if generation_config.no_repeat_ngram_size > 0 then
        no_repeat_ngram ~ngram_size:generation_config.no_repeat_ngram_size
        :: base_processors
      else base_processors
    in
    let base_processors =
      if generation_config.min_length > 0 then
        let eos_ids =
          match generation_config.eos_token_id with
          | Some id -> id :: generation_config.eos_token_ids
          | None -> generation_config.eos_token_ids
        in
        min_length ~min_length:generation_config.min_length
          ~eos_token_ids:eos_ids
        :: base_processors
      else base_processors
    in
    let base_processors =
      if generation_config.min_new_tokens > 0 then
        let eos_ids =
          match generation_config.eos_token_id with
          | Some id -> id :: generation_config.eos_token_ids
          | None -> generation_config.eos_token_ids
        in
        min_new_tokens ~min_new_tokens:generation_config.min_new_tokens
          ~eos_token_ids:eos_ids
        :: base_processors
      else base_processors
    in
    base_processors @ logits_processor
  in

  (* Build stopping criteria from config *)
  let criteria =
    let base_criteria = [] in
    let base_criteria =
      max_length_criteria ~max_length:generation_config.max_length
      :: base_criteria
    in
    let base_criteria =
      match generation_config.max_new_tokens with
      | Some max_new ->
          max_new_tokens_criteria ~max_new_tokens:max_new :: base_criteria
      | None -> base_criteria
    in
    let base_criteria =
      let eos_ids =
        match generation_config.eos_token_id with
        | Some id -> id :: generation_config.eos_token_ids
        | None -> generation_config.eos_token_ids
      in
      if eos_ids <> [] then
        eos_token_criteria ~eos_token_ids:eos_ids :: base_criteria
      else base_criteria
    in
    base_criteria @ stopping_criteria
  in

  (* Generation loop *)
  let rec generate_tokens tokens scores =
    (* Check stopping - but not on first iteration *)
    if
      List.length tokens > prompt_length
      && check_stopping ~criteria ~prompt_length ~start_time ~tokens
    then (List.rev tokens, List.rev scores)
    else
      (* Get and process logits *)
      let raw_logits = model tokens in
      let processed =
        apply_processors ~processors ~prompt_length ~tokens ~logits:raw_logits
      in

      (* Sample or greedy *)
      let next_token =
        if generation_config.do_sample then sample_from_logits processed
        else argmax processed
      in

      (* Track scores if requested *)
      let new_scores =
        if generation_config.output_scores then processed.(next_token) :: scores
        else scores
      in

      generate_tokens (tokens @ [ next_token ]) new_scores
  in

  let sequences, scores = generate_tokens input_ids [] in

  {
    sequences = [ sequences ];
    scores = (if generation_config.output_scores then Some [ scores ] else None);
    attentions = None;
    hidden_states = None;
  }

(** Simplified text generation *)
let generate_text ~model ~tokenizer ~decoder ?(prompt = "")
    ?(generation_config = default) ?(logits_processor = [])
    ?(stopping_criteria = []) () =
  let input_ids = tokenizer prompt in
  let output =
    generate ~model ~input_ids ~generation_config ~logits_processor
      ~stopping_criteria ()
  in
  match output.sequences with seq :: _ -> decoder seq | [] -> ""
