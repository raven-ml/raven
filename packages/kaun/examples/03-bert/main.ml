(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Fine-tune pretrained BERT for binary sentiment classification.

   Downloads bert-base-uncased from HuggingFace (~440MB on first run), assembles
   a sequence-classification head, and trains on a tiny synthetic dataset to
   show the full pipeline. *)

open Kaun

let print_shape name t =
  let shape = Nx.shape t in
  Printf.printf "%s: [%s]\n" name
    (String.concat "; " (Array.to_list (Array.map string_of_int shape)))

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let dtype = Nx.float32 in
  let num_labels = 2 in

  (* Load pretrained encoder + pooler from HuggingFace *)
  Printf.printf "Loading bert-base-uncased...\n%!";
  let cfg, encoder_params, pooler_params, _mlm_params =
    Bert.from_pretrained ()
  in
  Printf.printf "  hidden=%d  layers=%d  heads=%d  vocab=%d\n\n" cfg.hidden_size
    cfg.num_hidden_layers cfg.num_attention_heads cfg.vocab_size;

  (* Assemble classification model: pretrained encoder + pooler, fresh
     classifier head *)
  let w_init = Init.normal ~stddev:0.02 () in
  let params =
    Ptree.dict
      [
        ("encoder", encoder_params);
        ( "pooler",
          match pooler_params with
          | Some p -> p
          | None ->
              Ptree.dict
                [
                  ( "weight",
                    Ptree.tensor
                      (w_init.f [| cfg.hidden_size; cfg.hidden_size |] dtype) );
                  ("bias", Ptree.tensor (Nx.zeros dtype [| cfg.hidden_size |]));
                ] );
        ( "classifier",
          Ptree.dict
            [
              ( "weight",
                Ptree.tensor (w_init.f [| cfg.hidden_size; num_labels |] dtype)
              );
              ("bias", Ptree.tensor (Nx.zeros dtype [| num_labels |]));
            ] );
      ]
  in

  let model = Bert.for_sequence_classification cfg ~num_labels () in
  let vars = Layer.make_vars ~params ~state:Ptree.empty ~dtype in

  (* Tiny synthetic dataset (token ids from bert-base-uncased tokenizer) *)
  let input_ids =
    Nx.create Nx.int32 [| 4; 6 |]
      [|
        101l;
        1045l;
        2293l;
        2023l;
        102l;
        0l;
        (* "I love this" -> 1 *)
        101l;
        2307l;
        3185l;
        102l;
        0l;
        0l;
        (* "great movie" -> 1 *)
        101l;
        1045l;
        5223l;
        2023l;
        102l;
        0l;
        (* "I hate this" -> 0 *)
        101l;
        6659l;
        2143l;
        102l;
        0l;
        0l;
        (* "terrible film" -> 0 *)
      |]
  in
  let labels = Nx.create Nx.int32 [| 4 |] [| 1l; 1l; 0l; 0l |] in
  let attention_mask =
    Nx.create Nx.int32 [| 4; 6 |]
      [|
        1l;
        1l;
        1l;
        1l;
        1l;
        0l;
        1l;
        1l;
        1l;
        1l;
        0l;
        0l;
        1l;
        1l;
        1l;
        1l;
        1l;
        0l;
        1l;
        1l;
        1l;
        1l;
        0l;
        0l;
      |]
  in
  let ctx =
    Context.empty
    |> Context.set ~name:Attention.attention_mask_key (Ptree.P attention_mask)
  in

  (* --- Inference before training --- *)
  Printf.printf "=== Before training ===\n";
  let logits_before =
    let y, _ = Layer.apply model vars ~training:false ~ctx input_ids in
    y
  in
  print_shape "logits" logits_before;

  (* --- Fine-tune --- *)
  Printf.printf "\n=== Training ===\n%!";
  let trainer =
    Train.make ~model
      ~optimizer:
        (Optim.adamw ~lr:(Optim.Schedule.constant 2e-5) ~weight_decay:0.01 ())
  in
  let st = Train.make_state trainer vars in
  let st =
    Train.fit trainer st ~ctx
      ~report:(fun ~step ~loss _st ->
        Printf.printf "  step %2d  loss %.4f\n%!" step loss)
      (Data.repeat 10
         (input_ids, fun logits -> Loss.cross_entropy_sparse logits labels))
  in

  (* --- Predictions after training --- *)
  Printf.printf "\n=== After training ===\n";
  let logits = Train.predict trainer st ~ctx input_ids in
  let sentences =
    [| "I love this"; "great movie"; "I hate this"; "terrible film" |]
  in
  for i = 0 to 3 do
    let row = Nx.slice [ I i ] logits in
    let v0 = Nx.item [ 0 ] row in
    let v1 = Nx.item [ 1 ] row in
    let pred = if v1 > v0 then "positive" else "negative" in
    let label = Int32.to_int (Nx.item [ i ] labels) in
    let expected = if label = 1 then "positive" else "negative" in
    Printf.printf "  %-20s  pred=%-8s  expected=%-8s  %s\n"
      (Printf.sprintf "\"%s\"" sentences.(i))
      pred expected
      (if String.equal pred expected then "OK" else "WRONG")
  done
