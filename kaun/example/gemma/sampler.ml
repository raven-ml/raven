open Kaun

type sampling_config = {
  temperature : float;
  top_k : int option;
  top_p : float option;
  repetition_penalty : float;
  max_new_tokens : int;
}

let default_config =
  {
    temperature = 1.0;
    top_k = Some 50;
    top_p = Some 0.9;
    repetition_penalty = 1.0;
    max_new_tokens = 100;
  }

let sample_from_logits logits ~temperature ~top_k ~top_p ~seed =
  let dev = Rune.device logits in
  let dtype = Rune.dtype logits in

  (* Apply temperature *)
  let logits =
    if temperature <> 1.0 then
      Rune.div logits (Rune.scalar dev dtype temperature)
    else logits
  in

  (* Compute probabilities *)
  let probs = Rune.softmax logits ~axis:(-1) in

  (* Apply top-k filtering if specified *)
  let probs =
    match top_k with
    | Some k when k > 0 ->
        (* This would need proper top-k implementation *)
        probs
    | _ -> probs
  in

  (* Apply top-p (nucleus) filtering if specified *)
  let probs =
    match top_p with
    | Some p when p < 1.0 ->
        (* This would need proper nucleus sampling implementation *)
        probs
    | _ -> probs
  in

  (* Sample from distribution *)
  (* For now, just take argmax (greedy sampling) *)
  Rune.argmax probs ~axis:(-1) ~keepdims:false

let generate model params ~prompt_ids ~config ~rngs =
  let batch_size = 1 in
  let device = Rune.cpu () in

  (* Start with prompt *)
  let input_ids = ref (Array.of_list prompt_ids) in
  let generated = ref [] in

  for _ = 1 to config.max_new_tokens do
    (* Create input tensor *)
    let input_tensor =
      Rune.of_bigarray device Rune.int32
        [| batch_size; Array.length !input_ids |]
        (Bigarray.Array1.of_array Bigarray.int32 Bigarray.c_layout !input_ids)
    in

    (* Get model predictions *)
    let logits = apply model params ~training:false ~rngs input_tensor in

    (* Get logits for last position *)
    let seq_len = Array.length !input_ids in
    let last_logits = Rune.slice [ I 0; I (seq_len - 1) ] logits in

    (* Sample next token *)
    let next_token_tensor =
      sample_from_logits last_logits ~temperature:config.temperature
        ~top_k:config.top_k ~top_p:config.top_p ~seed:(Random.int 1000000)
    in

    let next_token = Rune.to_int next_token_tensor in

    (* Append to sequence *)
    input_ids := Array.append !input_ids [| next_token |];
    generated := next_token :: !generated
  done;

  List.rev !generated
