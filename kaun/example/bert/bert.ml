open Rune
module Bert = Kaun_models.Bert

let () =
  (* Load BERT and tokenizer *)
  let bert = Bert.from_pretrained ~device:c ~dtype:Float32 () in
  let tokenizer = Bert.Tokenizer.create () in
  let sentence = "The bank is by the river" in

  (* Encode and run forward pass *)
  let inputs = Bert.Tokenizer.encode tokenizer sentence ~device:c in
  let output = Bert.forward bert inputs () in

  (* Get CLS token embedding for sentence representation *)
  let cls_embedding = slice [ A; I 0; A ] output.last_hidden_state in
  let cls_mean = mean cls_embedding |> item [] in

  Printf.printf "  CLS mean: %.4f\n" cls_mean
