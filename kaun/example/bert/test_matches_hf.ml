(** Test that Kaun BERT output matches HuggingFace transformers *)

module Bert = Kaun_models.Bert
open Rune

let test_bert_matches_hf () =
  Printf.printf "Testing BERT forward pass against HuggingFace\n";
  Printf.printf "=============================================\n\n";

  (* Load pretrained BERT - much simpler! *)
  Printf.printf "Loading pretrained BERT from HuggingFace...\n";
  let bert = Bert.from_pretrained ~device:Rune.c ~dtype:Float32 () in

  (* Test text *)
  let text = "Hello world" in
  Printf.printf "Input text: \"%s\"\n" text;

  (* Initialize tokenizer *)
  let tokenizer = Bert.Tokenizer.create ~model_id:"bert-base-uncased" () in

  (* Encode text - now returns tensors directly! *)
  let inputs = Bert.Tokenizer.encode tokenizer text ~device:Rune.c in

  (* Check tokenization *)
  let token_ids =
    let flat = Rune.reshape [| -1 |] inputs.input_ids in
    let len = Rune.numel flat in
    List.init len (fun i -> Rune.unsafe_get [ i ] flat |> Int32.to_int)
  in
  Printf.printf "Token IDs: [%s]\n"
    (String.concat ", " (List.map string_of_int token_ids));

  (* Expected token IDs from Python: [101, 7592, 2088, 102] *)
  let expected_ids = [ 101; 7592; 2088; 102 ] in
  let tokenizer_match = token_ids = expected_ids in
  Printf.printf "Tokenizer: %s\n"
    (if tokenizer_match then "✓ Matches Python"
     else
       Printf.sprintf "✗ Expected [%s]"
         (String.concat ", " (List.map string_of_int expected_ids)));

  Printf.printf "\nRunning forward pass...\n";

  (* Run forward pass - much cleaner! *)
  let outputs =
    try Bert.forward bert inputs ()
    with e ->
      Printf.printf "Error in forward pass: %s\n" (Printexc.to_string e);
      Printf.printf "Stacktrace:\n%s\n" (Printexc.get_backtrace ());
      exit 1
  in

  let last_hidden_state = outputs.Bert.last_hidden_state in
  let pooler_output = outputs.Bert.pooler_output in

  (* Check shapes *)
  let shape = Rune.shape last_hidden_state |> Array.to_list in
  Printf.printf "\nOutput shape: [%s]\n"
    (String.concat "; " (List.map string_of_int shape));

  let shape_match = shape = [ 1; 4; 768 ] in
  Printf.printf "Shape: %s\n"
    (if shape_match then "✓ Correct [1; 4; 768]" else "✗ Unexpected shape");

  (* Get CLS token (first token) embeddings *)
  let cls_token = Rune.slice [ I 0; I 0 ] last_hidden_state in

  (* Expected values from Python transformers (first 5 values): [0]: -0.168883
     [1]: 0.136064 [2]: -0.139399 [3]: -0.054359 [4]: -0.295266 *)
  let expected_cls_values =
    [ -0.168883; 0.136064; -0.139399; -0.054359; -0.295266 ]
  in

  Printf.printf "\nCLS token embeddings (first 5 values):\n";
  let cls_diffs =
    List.mapi
      (fun i expected ->
        let actual = Rune.unsafe_get [ i ] cls_token in
        let diff = abs_float (actual -. expected) in
        Printf.printf "  [%d] Kaun: %8.6f, HF: %8.6f, Diff: %.9f %s\n" i actual
          expected diff
          (if diff < 1e-3 then "✓" else if diff < 1e-2 then "⚠" else "✗");
        diff)
      expected_cls_values
  in

  let max_cls_diff = List.fold_left Float.max 0.0 cls_diffs in
  Printf.printf "CLS embeddings: %s (max diff: %.6f)\n"
    (if max_cls_diff < 1e-2 then "✓ Match" else "✗ Mismatch")
    max_cls_diff;

  (* Check pooler output if available *)
  (match pooler_output with
  | Some p ->
      (* Expected pooler values from Python (first 5 values): [0]: -0.906153
         [1]: -0.311154 [2]: -0.621656 [3]: 0.774093 [4]: 0.289867 *)
      let expected_pooler =
        [ -0.906153; -0.311154; -0.621656; 0.774093; 0.289867 ]
      in

      Printf.printf "\nPooler output (first 5 values):\n";
      let pooler_diffs =
        List.mapi
          (fun i expected ->
            let actual = Rune.unsafe_get [ 0; i ] p in
            let diff = abs_float (actual -. expected) in
            Printf.printf "  [%d] Kaun: %8.6f, HF: %8.6f, Diff: %.9f %s\n" i
              actual expected diff
              (if diff < 1e-3 then "✓" else if diff < 1e-2 then "⚠" else "✗");
            diff)
          expected_pooler
      in

      let max_pooler_diff = List.fold_left Float.max 0.0 pooler_diffs in
      Printf.printf "Pooler output: %s (max diff: %.6f)\n"
        (if max_pooler_diff < 1e-2 then "✓ Match" else "✗ Mismatch")
        max_pooler_diff
  | None -> Printf.printf "\nNo pooler output available\n");

  (* Summary *)
  Printf.printf "\n";
  Printf.printf "==========================================\n";
  Printf.printf "Summary:\n";
  Printf.printf "  Tokenizer:        %s\n"
    (if tokenizer_match then "✓" else "✗");
  Printf.printf "  Output shape:     %s\n" (if shape_match then "✓" else "✗");
  Printf.printf "  CLS embeddings:   %s\n"
    (if max_cls_diff < 1e-2 then "✓" else "✗");
  (match pooler_output with
  | Some _ ->
      let max_pooler_diff =
        let expected_pooler =
          [ -0.906153; -0.311154; -0.621656; 0.774093; 0.289867 ]
        in
        List.mapi
          (fun i exp ->
            abs_float
              (Rune.unsafe_get [ 0; i ] (Option.get pooler_output) -. exp))
          expected_pooler
        |> List.fold_left Float.max 0.0
      in
      Printf.printf "  Pooler output:    %s\n"
        (if max_pooler_diff < 1e-2 then "✓" else "✗")
  | None -> ());

  let all_pass = tokenizer_match && shape_match && max_cls_diff < 1e-2 in
  Printf.printf "\nResult: %s\n"
    (if all_pass then "✅ PASS - Kaun BERT matches HuggingFace!"
     else "❌ FAIL - Output mismatch")

let () =
  Printexc.record_backtrace true;
  test_bert_matches_hf ()
