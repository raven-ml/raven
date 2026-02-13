(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Bert = Kaun_models.Bert
open Rune
open Windtrap

let expected_token_ids = [ 101; 7592; 2088; 102 ]
let expected_cls = [ -0.168883; 0.136064; -0.139399; -0.054359; -0.295266 ]
let expected_pooler = [ -0.906153; -0.311154; -0.621656; 0.774093; 0.289867 ]

let list_prefix ~n f =
  let rec aux acc i =
    if i = n then List.rev acc else aux (f i :: acc) (i + 1)
  in
  aux [] 0

let check_close_list label ~epsilon expected actual =
  List.iteri
    (fun i (e, a) ->
      let diff = Float.abs (a -. e) in
      if diff > epsilon then
        failf "%s[%d]: expected %.6f got %.6f (diff %.6f)" label i e a diff)
    (List.combine expected actual)

let test_forward () =
  let bert = Bert.from_pretrained ~dtype:Float32 () in
  let tokenizer = Bert.Tokenizer.create ~model_id:"bert-base-uncased" () in
  let inputs = Bert.Tokenizer.encode tokenizer "Hello world" in

  let token_ids =
    let flat = Rune.reshape [| -1 |] inputs.input_ids in
    List.init (Rune.numel flat) (fun i -> Rune.item [ i ] flat |> Int32.to_int)
  in
  equal ~msg:"token ids" (list int) expected_token_ids token_ids;

  let outputs = Bert.forward bert inputs () in
  let last_hidden_state = outputs.Bert.last_hidden_state in
  let shape = Array.to_list (Rune.shape last_hidden_state) in
  equal ~msg:"hidden-state shape" (list int) [ 1; 4; 768 ] shape;

  let cls_token = Rune.slice [ I 0; I 0 ] last_hidden_state in
  let cls_values =
    list_prefix ~n:(List.length expected_cls) (fun i ->
        Rune.item [ i ] cls_token)
  in
  check_close_list "cls" ~epsilon:1e-2 expected_cls cls_values;

  (match outputs.Bert.pooler_output with
  | None -> fail "expected pooler output"
  | Some pooler ->
      let values =
        list_prefix ~n:(List.length expected_pooler) (fun i ->
            Rune.item [ 0; i ] pooler)
      in
      check_close_list "pooler" ~epsilon:1e-2 expected_pooler values);

  ()

let () =
  Printexc.record_backtrace true;
  run "BERT matches HuggingFace"
    [ group "bert" [ test "forward" test_forward ] ]
