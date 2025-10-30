open Rune
module Bert = Kaun_models.Bert

(* Lightweight knobs for the demo. Tweak these and re-run to see how the probes
   react. *)
module Config = struct
  let model_id = "bert-base-uncased"
  let max_length = 96
  let similarity_threshold = 0.80
end

(* Guardrails – catching a typo up-front is better than chasing NaNs later. *)
let validate_config () =
  if Config.max_length <= 0 then failwith "Config.max_length must be positive";
  if Config.similarity_threshold <= 0.0 || Config.similarity_threshold > 1.0
  then failwith "Config.similarity_threshold must lie in (0, 1]"

(* Clip 2-D tensors (batch × seq) so we never ask the model to read beyond
   [max_length]. *)
let truncate_sequence tensor =
  let shape = Rune.shape tensor in
  if Array.length shape <> 2 then tensor
  else
    let seq_len = shape.(1) in
    if seq_len <= Config.max_length then tensor
    else slice [ A; R (0, Config.max_length) ] tensor

(* Extend truncation to the structured inputs that `Bert.Tokenizer.encode`
   returns. *)
let truncate_inputs (inputs : Bert.inputs) =
  let open Bert in
  {
    input_ids = truncate_sequence inputs.input_ids;
    attention_mask = truncate_sequence inputs.attention_mask;
    token_type_ids = Option.map truncate_sequence inputs.token_type_ids;
    position_ids = inputs.position_ids;
  }

(* Keep the raw ID arrays in sync with the tensor truncation. *)
let clip_token_ids ids =
  if Array.length ids <= Config.max_length then ids
  else Array.sub ids 0 Config.max_length

(* Helpers for cosine similarity; we cache them because they appear in both
   probes. *)
let scalar tensor = item [] tensor

let norm tensor =
  let squared = scalar (sum (mul tensor tensor)) in
  Float.sqrt (Float.max squared 1e-12)

let cosine_similarity lhs rhs =
  let dot = scalar (sum (mul lhs rhs)) in
  let denom = norm lhs *. norm rhs in
  if denom = 0.0 then 0.0 else dot /. denom

(* Load the pretrained model + tokenizer and report progress so the console
   tells a story. *)
let load_model () =
  Printf.printf "Loading BERT model '%s'...\n%!" Config.model_id;
  let bert = Bert.from_pretrained ~dtype:Float32 () in
  let tokenizer = Bert.Tokenizer.create ~model_id:Config.model_id () in
  Printf.printf "✓ Model ready\n\n%!";
  (bert, tokenizer)

(* One-stop helper: encode → truncate → forward. *)
let forward_sentence bert tokenizer text =
  let inputs = Bert.Tokenizer.encode tokenizer text |> truncate_inputs in
  Bert.forward bert inputs ()

(* Retrieve both CLS and mean-pooled embeddings so we can compare pooling
   strategies. *)
let sentence_embeddings bert tokenizer text =
  let hidden = (forward_sentence bert tokenizer text).Bert.last_hidden_state in
  let cls = slice [ I 0; I 0; A ] hidden in
  let tokens = slice [ I 0; A; A ] hidden in
  let mean_pool = mean tokens ~axes:[ 0 ] in
  (cls, mean_pool)

(* Focus on a specific word: find its token position and pull the contextual
   embedding. *)
let contextual_embedding bert tokenizer ~word sentence =
  let word_lc = String.lowercase_ascii word in
  let hidden =
    (forward_sentence bert tokenizer sentence).Bert.last_hidden_state
  in
  let token_ids =
    Bert.Tokenizer.encode_to_array tokenizer sentence |> clip_token_ids
  in
  let seq_len = Array.length token_ids in
  let find_index () =
    let rec aux i =
      if i >= seq_len - 1 then None
      else
        let token_id = token_ids.(i) in
        if i = 0 then aux (i + 1)
        else
          let token =
            Bert.Tokenizer.decode tokenizer [| token_id |]
            |> String.trim |> String.lowercase_ascii
          in
          if String.equal token word_lc then Some i else aux (i + 1)
    in
    aux 0
  in
  let index = match find_index () with Some idx -> idx | None -> 1 in
  let embedding = slice [ I 0; I index; A ] hidden in
  (embedding, index)

(* Tiny corpora keep the output readable; add your own pairs when
   experimenting. *)
let sentence_pairs =
  [
    ("The cat curls up on the sofa", "A feline naps on the couch", true);
    ("The project deadline is tomorrow", "We have weeks before delivery", false);
    ("Dogs are loyal companions", "Canines make faithful pets", true);
  ]

(* Print a single similarity comparison with both pooling flavours. *)
let report_sentence_pair bert tokenizer index (text_a, text_b, expected) =
  let cls_a, mean_a = sentence_embeddings bert tokenizer text_a in
  let cls_b, mean_b = sentence_embeddings bert tokenizer text_b in
  let sim_cls = cosine_similarity cls_a cls_b in
  let sim_mean = cosine_similarity mean_a mean_b in
  let decide sim =
    if sim >= Config.similarity_threshold then "Similar" else "Different"
  in
  Printf.printf "%d. \"%s\"\n   \"%s\"\n" (index + 1) text_a text_b;
  Printf.printf "   cosine(CLS) : %.4f\n" sim_cls;
  Printf.printf "   cosine(mean): %.4f → %s (expected %s)\n\n" sim_mean
    (decide sim_mean)
    (if expected then "Similar" else "Different")

let sentence_similarity bert tokenizer =
  Printf.printf "=== Sentence Similarity ===\n";
  List.iteri (report_sentence_pair bert tokenizer) sentence_pairs

(* Word-in-context examples that expose common polysemy cases. *)
let polysemy_examples =
  [
    ("bank", "The bank raised interest rates", "We sat on the bank of the river");
    ( "plant",
      "The chemical plant runs three shifts",
      "She watered the plant before sunrise" );
    ("bat", "He hit a home run with the bat", "A bat fluttered across the cave");
  ]

(* Show how far two contextual embeddings drift for the same surface word. *)
let report_polysemy bert tokenizer (word, ctx_a, ctx_b) =
  let emb_a, idx_a = contextual_embedding bert tokenizer ~word ctx_a in
  let emb_b, idx_b = contextual_embedding bert tokenizer ~word ctx_b in
  let cosine = cosine_similarity emb_a emb_b in
  let separated = if cosine < Config.similarity_threshold then "✓" else "✗" in
  Printf.printf "%-6s  %-40s | %-40s | %-8.4f %s\n" word ctx_a ctx_b cosine
    separated;
  Printf.printf "         token positions: %d vs %d (max %d)\n\n" idx_a idx_b
    Config.max_length

let polysemy_probe bert tokenizer =
  Printf.printf "=== Word-in-Context (Polysemy) ===\n";
  Printf.printf "%-6s  %-40s | %-40s | %-8s\n" "Word" "Context A" "Context B"
    "Cosine";
  Printf.printf "%s\n" (String.make 104 '-');
  List.iter (report_polysemy bert tokenizer) polysemy_examples;
  Printf.printf
    "✓  suggests BERT separates the senses (low cosine); ✗ means the contexts \
     stay close.\n\n"

let () =
  validate_config ();
  Printf.printf "Configuration:\n";
  Printf.printf "  model_id              : %s\n" Config.model_id;
  Printf.printf "  max_length            : %d\n" Config.max_length;
  Printf.printf "  similarity_threshold  : %.2f\n\n" Config.similarity_threshold;
  let bert, tokenizer = load_model () in
  sentence_similarity bert tokenizer;
  polysemy_probe bert tokenizer
