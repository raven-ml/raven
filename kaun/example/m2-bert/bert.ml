open Rune
module Bert = Kaun_models.Bert

(** Configuration for BERT tasks *)
module Config = struct
  let model_id = "bert-base-uncased"
  let max_length = 128
  let similarity_threshold = 0.8  (* Cosine similarity threshold *)
end

(** Validate configuration *)
let validate_config () =
  if Config.max_length <= 0 then failwith "max_length must be positive";
  if Config.similarity_threshold <= 0.0 || Config.similarity_threshold > 1.0 then
    failwith "similarity_threshold must be in (0, 1]"

(** Safe model loading with error handling *)
let load_model_safe () =
  try
    Printf.printf "Loading BERT model '%s'...\n%!" Config.model_id;
    let bert = Bert.from_pretrained ~dtype:Float32 () in
    let tokenizer = Bert.Tokenizer.create () in
    Printf.printf "✓ Model loaded successfully\n\n%!";
    (bert, tokenizer)
  with
  | exn ->
      Printf.eprintf "✗ Failed to load model: %s\n" (Printexc.to_string exn);
      exit 1

(** Calculate cosine similarity between two vectors *)
let cosine_similarity vec1 vec2 =
  let dot_product = sum (mul vec1 vec2) |> item [] in
  let norm1 = sqrt (sum (mul vec1 vec1)) |> item [] in
  let norm2 = sqrt (sum (mul vec2 vec2)) |> item [] in
  if norm1 = 0.0 || norm2 = 0.0 then 0.0
  else dot_product /. (norm1 *. norm2)

(** Extract sentence embedding from BERT output *)
let extract_sentence_embedding last_hidden_state method_ =
  match method_ with
  | `CLS -> 
      (* Use CLS token embedding *)
      slice [I 0; I 0; A] last_hidden_state
  | `Mean ->
      (* Mean pooling of all token embeddings *)
      let all_tokens = slice [I 0; A; A] last_hidden_state in
      mean all_tokens ~axes:[0]
  | `Max ->
      (* Max pooling of all token embeddings *)  
      let all_tokens = slice [I 0; A; A] last_hidden_state in
      max all_tokens ~axes:[0]

(** Sentence similarity task *)
let sentence_similarity_task bert tokenizer =
  Printf.printf "=== Sentence Similarity Task ===\n";
  
  let sentence_pairs = [
    ("The cat sits on the mat", "A feline rests on the rug", true);
    ("I love machine learning", "Machine learning is fascinating", true);
    ("The weather is sunny today", "It's raining heavily outside", false);
    ("Programming is fun", "Coding is enjoyable", true);
    ("Dogs are loyal animals", "Cars are fast vehicles", false);
    ("Paris is the capital of France", "The capital of France is Paris", true);
  ] in
  
  let correct_predictions = ref 0 in
  let total_predictions = List.length sentence_pairs in
  
  Printf.printf "Comparing sentence pairs (threshold=%.2f):\n\n" Config.similarity_threshold;
  
  List.iteri (fun i (sent1, sent2, expected_similar) ->
    Printf.printf "%d. Sentence 1: \"%s\"\n" (i+1) sent1;
    Printf.printf "   Sentence 2: \"%s\"\n" sent2;
    
    (* Encode both sentences *)
    let inputs1 = Bert.Tokenizer.encode tokenizer sent1 in
    let inputs2 = Bert.Tokenizer.encode tokenizer sent2 in
    
    (* Get embeddings *)
    let output1 = Bert.forward bert inputs1 () in
    let output2 = Bert.forward bert inputs2 () in
    
    (* Extract sentence embeddings using different methods *)
    let embedding1_cls = extract_sentence_embedding output1.last_hidden_state `CLS in
    let embedding2_cls = extract_sentence_embedding output2.last_hidden_state `CLS in
    let embedding1_mean = extract_sentence_embedding output1.last_hidden_state `Mean in  
    let embedding2_mean = extract_sentence_embedding output2.last_hidden_state `Mean in
    
    (* Calculate similarities *)
    let sim_cls = cosine_similarity embedding1_cls embedding2_cls in
    let sim_mean = cosine_similarity embedding1_mean embedding2_mean in
    
    (* Make prediction based on CLS similarity *)
    let predicted_similar = sim_mean >= Config.similarity_threshold in
    let correct = predicted_similar = expected_similar in
    if correct then incr correct_predictions;
    
    Printf.printf "   Similarity (CLS): %.4f\n" sim_cls;
    Printf.printf "   Similarity (Mean): %.4f\n" sim_mean;
    Printf.printf "   Expected: %s | Predicted: %s | %s\n"
      (if expected_similar then "Similar" else "Different")
      (if predicted_similar then "Similar" else "Different")
      (if correct then "✓" else "✗");
    Printf.printf "\n%!"
  ) sentence_pairs;
  
  let accuracy = Float.of_int !correct_predictions /. Float.of_int total_predictions in
  Printf.printf "Similarity Task Accuracy: %.3f (%d/%d)\n\n" 
    accuracy !correct_predictions total_predictions

(** Word-in-Context Polysemy Verification *)
let word_in_context_task bert tokenizer =
  Printf.printf "=== Word-in-Context (Polysemy Verification) ===\n";

  (* Each pair = same word, two different contexts *)
  let polysemy_pairs = [
    ("bank",
      "I went to the bank to deposit money",
      "The river bank was covered with flowers",
      "financial vs riverside");
    ("plant",
      "The chemical plant employs hundreds",
      "She watered the plant in her garden",
      "factory vs vegetation");
    ("bat",
      "He swung the bat hard",
      "A bat flew into the cave",
      "sports equipment vs animal");
  ] in

  Printf.printf "%-8s | %-45s | %-45s | %-10s | %-8s\n"
    "Word" "Context 1" "Context 2" "CosineSim" "Status";
  Printf.printf "%s\n"
    (String.make 120 '-');

  List.iter (fun (word, sent1, sent2, desc) ->
    (* Encode both sentences *)
    let inputs1 = Bert.Tokenizer.encode tokenizer sent1 in
    let inputs2 = Bert.Tokenizer.encode tokenizer sent2 in
    let output1 = Bert.forward bert inputs1 () in
    let output2 = Bert.forward bert inputs2 () in

    (* Get token embeddings - for simplicity use middle token as proxy *)
    let token_ids1 = Bert.Tokenizer.encode_to_array tokenizer sent1 in
    let token_ids2 = Bert.Tokenizer.encode_to_array tokenizer sent2 in
    let idx1 = Array.length token_ids1 / 2 in
    let idx2 = Array.length token_ids2 / 2 in
    let emb1 = slice [I 0; I idx1; A] output1.last_hidden_state in
    let emb2 = slice [I 0; I idx2; A] output2.last_hidden_state in

    (* Compute cosine similarity between contextual embeddings *)
    let sim = cosine_similarity emb1 emb2 in

    let status =
      if sim < Config.similarity_threshold then "✓" else "✗"
    in

    Printf.printf "%-8s | %-45s | %-45s | %-10.4f | %-8s\n"
      word sent1 sent2 sim status;

    Printf.printf "   Meaning contrast: %s\n\n%!" desc
  ) polysemy_pairs;

  Printf.printf
    "✓  = Distinct senses recognized (low similarity)\n✗  = Senses conflated (high similarity)\n\n%!"



let () =
  try
    (* Validate configuration *)
    validate_config ();
    
    Printf.printf "Configuration:\n";
    Printf.printf "  Model: %s\n" Config.model_id;
    Printf.printf "  Max length: %d\n" Config.max_length;
    Printf.printf "  Similarity threshold: %.2f\n" Config.similarity_threshold;
    Printf.printf "\n";
    
    (* Load model safely *)
    let (bert, tokenizer) = load_model_safe () in
    
    (* Run different tasks *)
    sentence_similarity_task bert tokenizer;
    word_in_context_task bert tokenizer;
    
    
  with
  | Failure msg ->
      Printf.eprintf "Error: %s\n" msg;
      exit 1  
  | exn ->
      Printf.eprintf "Unexpected error: %s\n" (Printexc.to_string exn);
      exit 1
