(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Core Types ───── *)

type logits = (float, Bigarray.float32_elt) Nx.t
type token_ids = int array

(* ───── Logits Processors ───── *)

type logits_processor = {
  name : string;
  process : prompt_length:int -> token_ids -> logits -> logits;
}

type logits_processor_list = logits_processor list

(* ───── Stopping Criteria ───── *)

type stopping_criterion = {
  name : string;
  should_stop : prompt_length:int -> start_time:float -> token_ids -> bool;
}

type stopping_criteria_list = stopping_criterion list

(* ───── Generation Configuration ───── *)

type generation_config = {
  max_length : int;
  max_new_tokens : int option;
  min_length : int;
  min_new_tokens : int;
  do_sample : bool;
  temperature : float;
  top_k : int;
  top_p : float;
  repetition_penalty : float;
  no_repeat_ngram_size : int;
  bad_words_ids : int list list;
  force_words_ids : int list list;
  pad_token_id : int option;
  bos_token_id : int option;
  eos_token_id : int option;
  eos_token_ids : int list;
}

let default =
  {
    max_length = 100;
    max_new_tokens = None;
    min_length = 0;
    min_new_tokens = 0;
    do_sample = false;
    temperature = 1.0;
    top_k = 0;
    top_p = 1.0;
    repetition_penalty = 1.0;
    no_repeat_ngram_size = 0;
    bad_words_ids = [];
    force_words_ids = [];
    pad_token_id = None;
    bos_token_id = None;
    eos_token_id = None;
    eos_token_ids = [];
  }

(* ───── Builder Pattern ───── *)

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

let with_do_sample do_sample config = { config with do_sample }

(* ───── Preset Configurations ───── *)

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

(* ───── Logits Processors ───── *)

let neg_infinity = Float.neg_infinity

let temperature_warper ~temperature =
  {
    name = Printf.sprintf "temperature(%.2f)" temperature;
    process =
      (fun ~prompt_length:_ _tokens logits ->
        if temperature = 1.0 then logits else Nx.div_s logits temperature);
  }

let top_k_warper ~k =
  {
    name = Printf.sprintf "top_k(%d)" k;
    process =
      (fun ~prompt_length:_ _tokens logits ->
        if k <= 0 then logits
        else
          let sorted_values, _sorted_indices =
            Nx.sort ~descending:true logits
          in
          let vocab_size = Nx.numel logits in
          let cutoff_k = min k vocab_size in
          let threshold = Nx.item [ cutoff_k - 1 ] sorted_values in
          let mask = Nx.less_s logits threshold in
          Nx.where mask (Nx.full_like logits neg_infinity) logits);
  }

let top_p_warper ~p =
  {
    name = Printf.sprintf "top_p(%.2f)" p;
    process =
      (fun ~prompt_length:_ _tokens logits ->
        if p >= 1.0 then logits
        else
          let probs = Nx.softmax logits in
          let sorted_probs, sorted_indices = Nx.sort ~descending:true probs in
          let cumulative = Nx.cumsum sorted_probs in
          (* Find where cumulative exceeds p, keeping at least 1 token *)
          let cutoff_mask = Nx.greater_s cumulative p in
          (* Shift mask right by 1 so the token that crosses p is kept *)
          let n = Nx.numel logits in
          let shifted_arr = Nx.to_array cutoff_mask in
          let new_mask_arr = Array.make n false in
          for i = 1 to n - 1 do
            new_mask_arr.(i) <- shifted_arr.(i - 1)
          done;
          let shifted_mask = Nx.create Nx.bool [| n |] new_mask_arr in
          (* Map mask back to original token order *)
          let result = Nx.copy logits in
          let sorted_idx_arr = Nx.to_array sorted_indices in
          let shifted_mask_arr = Nx.to_array shifted_mask in
          for i = 0 to n - 1 do
            if shifted_mask_arr.(i) then
              Nx.set_item
                [ Int32.to_int sorted_idx_arr.(i) ]
                neg_infinity result
          done;
          result);
  }

let repetition_penalty ~penalty =
  {
    name = Printf.sprintf "repetition_penalty(%.2f)" penalty;
    process =
      (fun ~prompt_length:_ previous_tokens logits ->
        if penalty = 1.0 then logits
        else
          let result = Nx.copy logits in
          let vocab_size = Nx.numel result in
          Array.iter
            (fun token_id ->
              if token_id < vocab_size then begin
                let score = Nx.item [ token_id ] result in
                let penalized =
                  if score < 0.0 then score *. penalty else score /. penalty
                in
                Nx.set_item [ token_id ] penalized result
              end)
            previous_tokens;
          result);
  }

let no_repeat_ngram ~ngram_size =
  {
    name = Printf.sprintf "no_repeat_ngram(%d)" ngram_size;
    process =
      (fun ~prompt_length:_ previous_tokens logits ->
        let len = Array.length previous_tokens in
        if ngram_size <= 0 || len < ngram_size - 1 then logits
        else
          let result = Nx.copy logits in
          (* Get the last (ngram_size - 1) tokens as the current prefix *)
          let prefix_start = len - (ngram_size - 1) in
          let prefix =
            Array.sub previous_tokens prefix_start (ngram_size - 1)
          in
          (* Scan history for matching prefixes *)
          for i = 0 to len - ngram_size do
            let matches = ref true in
            for j = 0 to ngram_size - 2 do
              if previous_tokens.(i + j) <> prefix.(j) then matches := false
            done;
            if !matches then begin
              let blocked_token = previous_tokens.(i + ngram_size - 1) in
              if blocked_token < Nx.numel result then
                Nx.set_item [ blocked_token ] neg_infinity result
            end
          done;
          result);
  }

let min_length ~min_length ~eos_token_ids =
  {
    name = Printf.sprintf "min_length(%d)" min_length;
    process =
      (fun ~prompt_length:_ tokens logits ->
        if Array.length tokens >= min_length then logits
        else
          let result = Nx.copy logits in
          let vocab_size = Nx.numel result in
          List.iter
            (fun eos_id ->
              if eos_id < vocab_size then
                Nx.set_item [ eos_id ] neg_infinity result)
            eos_token_ids;
          result);
  }

let min_new_tokens ~min_new_tokens ~eos_token_ids =
  {
    name = Printf.sprintf "min_new_tokens(%d)" min_new_tokens;
    process =
      (fun ~prompt_length tokens logits ->
        let new_tokens = Array.length tokens - prompt_length in
        if new_tokens >= min_new_tokens then logits
        else
          let result = Nx.copy logits in
          let vocab_size = Nx.numel result in
          List.iter
            (fun eos_id ->
              if eos_id < vocab_size then
                Nx.set_item [ eos_id ] neg_infinity result)
            eos_token_ids;
          result);
  }

let bad_words ~bad_words_ids =
  {
    name = "bad_words";
    process =
      (fun ~prompt_length:_ tokens logits ->
        let result = Nx.copy logits in
        let len = Array.length tokens in
        let vocab_size = Nx.numel result in
        List.iter
          (fun bad_sequence ->
            let seq_len = List.length bad_sequence in
            if seq_len > 0 && len >= seq_len - 1 then (
              let prefix_len = seq_len - 1 in
              let matches = ref true in
              let prefix = List.rev (List.tl (List.rev bad_sequence)) in
              List.iteri
                (fun i expected ->
                  if tokens.(len - prefix_len + i) <> expected then
                    matches := false)
                prefix;
              if !matches then begin
                let bad_token = List.nth bad_sequence (seq_len - 1) in
                if bad_token < vocab_size then
                  Nx.set_item [ bad_token ] neg_infinity result
              end))
          bad_words_ids;
        result);
  }

let force_words ~force_words_ids ~iteration =
  {
    name = "force_words";
    process =
      (fun ~prompt_length:_ _tokens logits ->
        if iteration >= List.length force_words_ids then logits
        else
          let forced_tokens = List.nth force_words_ids iteration in
          let result = Nx.full_like logits neg_infinity in
          List.iter
            (fun token_id ->
              if token_id < Nx.numel result then
                Nx.set_item [ token_id ] (Nx.item [ token_id ] logits) result)
            forced_tokens;
          result);
  }

let custom ~name ~process = { name; process }

(* ───── Stopping Criteria ───── *)

let max_length_criteria ~max_length =
  {
    name = Printf.sprintf "max_length(%d)" max_length;
    should_stop =
      (fun ~prompt_length:_ ~start_time:_ tokens ->
        Array.length tokens >= max_length);
  }

let max_new_tokens_criteria ~max_new_tokens =
  {
    name = Printf.sprintf "max_new_tokens(%d)" max_new_tokens;
    should_stop =
      (fun ~prompt_length ~start_time:_ tokens ->
        Array.length tokens - prompt_length >= max_new_tokens);
  }

let eos_token_criteria ~eos_token_ids =
  {
    name = "eos_token";
    should_stop =
      (fun ~prompt_length:_ ~start_time:_ tokens ->
        let len = Array.length tokens in
        if len = 0 then false else List.mem tokens.(len - 1) eos_token_ids);
  }

let max_time_criteria ~max_time =
  {
    name = Printf.sprintf "max_time(%.1fs)" max_time;
    should_stop =
      (fun ~prompt_length:_ ~start_time _tokens ->
        Unix.gettimeofday () -. start_time > max_time);
  }

let stop_strings_criteria ~stop_strings ~decoder =
  {
    name = "stop_strings";
    should_stop =
      (fun ~prompt_length:_ ~start_time:_ tokens ->
        let text = decoder tokens in
        List.exists
          (fun stop_str -> String_util.contains_substring text stop_str)
          stop_strings);
  }

let custom_criteria ~name ~should_stop = { name; should_stop }

(* ───── Utilities ───── *)

let apply_processors ~processors ~prompt_length ~tokens ~logits =
  List.fold_left
    (fun acc processor -> processor.process ~prompt_length tokens acc)
    logits processors

let check_stopping ~criteria ~prompt_length ~start_time ~tokens =
  List.exists
    (fun criterion -> criterion.should_stop ~prompt_length ~start_time tokens)
    criteria

(* ───── Main Generation Functions ───── *)

type generation_output = {
  sequences : int array list;
  scores : float list list option;
}

let sample_from_logits logits =
  let probs = Nx.softmax logits in
  let probs_arr = Nx.to_array probs in
  let r = Random.float 1.0 in
  let cumsum = ref 0.0 in
  let result = ref 0 in
  (try
     for i = 0 to Array.length probs_arr - 1 do
       cumsum := !cumsum +. probs_arr.(i);
       if !cumsum > r then begin
         result := i;
         raise_notrace Exit
       end
     done
   with Exit -> ());
  !result

let argmax logits = Int32.to_int (Nx.item [ 0 ] (Nx.argmax logits))

let generate ~model ?(input_ids = [||]) ?(generation_config = default)
    ?(logits_processor = []) ?(stopping_criteria = []) () =
  let start_time = Unix.gettimeofday () in
  let prompt_length = Array.length input_ids in

  let processors =
    let ps = [] in
    let ps =
      if generation_config.temperature <> 1.0 then
        temperature_warper ~temperature:generation_config.temperature :: ps
      else ps
    in
    let ps =
      if generation_config.top_k > 0 then
        top_k_warper ~k:generation_config.top_k :: ps
      else ps
    in
    let ps =
      if generation_config.top_p < 1.0 then
        top_p_warper ~p:generation_config.top_p :: ps
      else ps
    in
    let ps =
      if generation_config.repetition_penalty <> 1.0 then
        repetition_penalty ~penalty:generation_config.repetition_penalty :: ps
      else ps
    in
    let ps =
      if generation_config.no_repeat_ngram_size > 0 then
        no_repeat_ngram ~ngram_size:generation_config.no_repeat_ngram_size :: ps
      else ps
    in
    let eos_ids =
      match generation_config.eos_token_id with
      | Some id -> id :: generation_config.eos_token_ids
      | None -> generation_config.eos_token_ids
    in
    let ps =
      if generation_config.min_length > 0 then
        min_length ~min_length:generation_config.min_length
          ~eos_token_ids:eos_ids
        :: ps
      else ps
    in
    let ps =
      if generation_config.min_new_tokens > 0 then
        min_new_tokens ~min_new_tokens:generation_config.min_new_tokens
          ~eos_token_ids:eos_ids
        :: ps
      else ps
    in
    ps @ logits_processor
  in

  let criteria =
    let cs = [] in
    let cs =
      max_length_criteria ~max_length:generation_config.max_length :: cs
    in
    let cs =
      match generation_config.max_new_tokens with
      | Some max_new -> max_new_tokens_criteria ~max_new_tokens:max_new :: cs
      | None -> cs
    in
    let eos_ids =
      match generation_config.eos_token_id with
      | Some id -> id :: generation_config.eos_token_ids
      | None -> generation_config.eos_token_ids
    in
    let cs =
      if eos_ids <> [] then eos_token_criteria ~eos_token_ids:eos_ids :: cs
      else cs
    in
    cs @ stopping_criteria
  in

  let tokens_ref = ref (Array.copy input_ids) in

  let rec generate_loop () =
    let current_tokens = !tokens_ref in
    if
      Array.length current_tokens > prompt_length
      && check_stopping ~criteria ~prompt_length ~start_time
           ~tokens:current_tokens
    then current_tokens
    else begin
      let raw_logits = model current_tokens in
      let processed =
        apply_processors ~processors ~prompt_length ~tokens:current_tokens
          ~logits:raw_logits
      in
      let next_token =
        if generation_config.do_sample then sample_from_logits processed
        else argmax processed
      in
      tokens_ref := Array.append current_tokens [| next_token |];
      generate_loop ()
    end
  in

  let sequences = generate_loop () in
  { sequences = [ sequences ]; scores = None }

let generate_text ~model ~tokenizer ~decoder ?(prompt = "")
    ?(generation_config = default) ?(logits_processor = [])
    ?(stopping_criteria = []) () =
  let input_ids = tokenizer prompt in
  let output =
    generate ~model ~input_ids ~generation_config ~logits_processor
      ~stopping_criteria ()
  in
  match output.sequences with seq :: _ -> decoder seq | [] -> ""
