open Rune
module GPT2 = Kaun_models.GPT2

let () =
  (* Load GPT-2 and tokenizer *)
  let gpt2 = GPT2.from_pretrained ~dtype:Float32 () in
  let tokenizer = GPT2.Tokenizer.create () in
  let prompt = "The future of AI is" in

  (* Encode the prompt *)
  let inputs = GPT2.Tokenizer.encode tokenizer prompt in

  (* Run forward pass - for causal LM we need the full model *)
  let logits, _ =
    GPT2.For_causal_lm.forward ~model:gpt2.model ~params:gpt2.params
      ~input_ids:inputs.input_ids ~training:false ()
  in

  (* Get logits for the last token *)
  let seq_len = shape inputs.input_ids |> fun s -> s.(1) in
  let last_logits = slice [ A; I (seq_len - 1); A ] logits in

  (* Get top prediction *)
  let probs = softmax ~axes:[ 1 ] last_logits in
  let top_prob = max probs |> item [] in
  let predicted_id = argmax ~axis:1 last_logits |> item [] |> Int32.to_int in

  Printf.printf "Prompt: %s\n" prompt;
  Printf.printf "Next token ID: %d (prob: %.4f)\n" predicted_id top_prob;

  (* Decode the predicted token *)
  let next_token = GPT2.Tokenizer.decode tokenizer [| predicted_id |] in
  Printf.printf "Next token: '%s'\n" next_token;
  Printf.printf "Continuation: %s%s\n" prompt next_token
