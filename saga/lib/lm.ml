(** High-level language model API *)

(** Model type - wraps different backend implementations *)
type model =
  | Ngram : {
      n : int;
      smoothing : float;
      min_freq : int;
      specials : string list;
      tokenizer : 'a Saga_tokenizers.Tokenizer.t;
      backend : Saga_models.Ngram.t option; (* None until trained *)
      vocab : Saga_tokenizers.vocab option; (* None until trained *)
    }
      -> model

(** Create an n-gram model *)
let ngram ~n ?(smoothing = 0.01) ?(min_freq = 1) ?(specials = []) ?tokenizer ()
    =
  (* Determine default special tokens based on tokenizer type *)
  let default_specials =
    match specials with
    | [] -> [ "<bos>"; "<eos>" ] (* Default special tokens *)
    | s -> s
  in
  match tokenizer with
  | Some t ->
      Ngram
        {
          n;
          smoothing;
          min_freq;
          specials = default_specials;
          tokenizer = t;
          backend = None;
          vocab = None;
        }
  | None ->
      (* Default tokenizer is chars *)
      Ngram
        {
          n;
          smoothing;
          min_freq;
          specials = default_specials;
          tokenizer = Saga_tokenizers.Tokenizer.chars;
          backend = None;
          vocab = None;
        }

(** Train a model on texts *)
let train model texts =
  match model with
  | Ngram config ->
      (* Tokenize all texts *)
      let tokenized_texts =
        List.map
          (fun text -> Saga_tokenizers.Tokenizer.run config.tokenizer text)
          texts
      in

      (* Flatten to get all tokens for vocab building *)
      let all_tokens = List.concat tokenized_texts in

      (* Build vocabulary *)
      let vocab = Saga_tokenizers.vocab ~min_freq:config.min_freq all_tokens in

      (* Convert texts to token IDs *)
      let encoded_texts =
        List.map
          (fun tokens ->
            List.filter_map
              (fun token -> Saga_tokenizers.Vocab.get_index vocab token)
              tokens
            |> Array.of_list)
          tokenized_texts
      in

      (* Concatenate all sequences for training *)
      let all_ids =
        encoded_texts |> List.map Array.to_list |> List.concat |> Array.of_list
      in

      (* Create and train the n-gram backend *)
      let backend =
        Saga_models.Ngram.create ~n:config.n
          ~smoothing:(Saga_models.Ngram.Add_k config.smoothing) all_ids
      in

      (* Return updated model with trained backend *)
      Ngram { config with backend = Some backend; vocab = Some vocab }

(** Generate text from model *)
let generate model ?(num_tokens = 20) ?(temperature = 1.0) ?top_k ?top_p ?seed
    ?min_new_tokens ?(prompt = "") () =
  match model with
  | Ngram { backend = None; _ } -> failwith "generate: model not trained"
  | Ngram { vocab = None; _ } -> failwith "generate: model vocab not built"
  | Ngram
      { backend = Some backend; vocab = Some vocab; tokenizer; specials; n; _ }
    ->
      (* Prepare prompt *)
      let prompt_text =
        if prompt = "" then
          (* Start with BOS token if no prompt *)
          List.hd specials (* Use first special token as BOS *)
        else prompt
      in

      (* Get EOS token ID - for character-level with "." as delimiter, we want
         to stop when we see "." AFTER generating some characters *)
      let eos_token =
        if
          List.length specials >= 2
          && List.nth specials 0 <> List.nth specials 1
        then
          List.nth specials
            1 (* Use second special token as EOS if different from first *)
        else if List.length specials >= 2 then
          (* Both specials are the same (e.g., "."), so use it as delimiter *)
          List.nth specials 1
        else "<eos>" (* Default EOS token *)
      in
      let eos_id =
        match Saga_tokenizers.Vocab.get_index vocab eos_token with
        | Some id -> id
        | None -> Saga_tokenizers.Vocab.eos_idx vocab (* Fallback to default *)
      in

      (* Tokenize and encode the prompt to get starting tokens *)
      let prompt_tokens = Saga_tokenizers.Tokenizer.run tokenizer prompt_text in
      let prompt_ids =
        List.filter_map
          (fun token -> Saga_tokenizers.Vocab.get_index vocab token)
          prompt_tokens
      in

      (* Create logits function that works with token history *)
      let logits_fn history =
        (* Get context for n-gram - take last n-1 tokens *)
        let context =
          let hist_array = Array.of_list history in
          let len = Array.length hist_array in
          if len = 0 then [||]
          else
            let context_len = min (n - 1) len in
            Array.sub hist_array (len - context_len) context_len
        in
        Saga_models.Ngram.logits backend ~context
      in

      (* Create generation config using builder pattern *)
      let config =
        let base_config =
          Sampler.default
          |> Sampler.with_max_new_tokens num_tokens
          |> Sampler.with_temperature temperature
          |> Sampler.with_do_sample (temperature > 0.0)
        in
        let base_config =
          match top_k with
          | Some k -> Sampler.with_top_k k base_config
          | None -> base_config
        in
        let base_config =
          match top_p with
          | Some p -> Sampler.with_top_p p base_config
          | None -> base_config
        in
        let base_config =
          match min_new_tokens with
          | Some min_new -> Sampler.with_min_new_tokens min_new base_config
          | None -> base_config
        in
        { base_config with eos_token_id = Some eos_id }
      in

      (* Set random seed if provided *)
      let () = match seed with Some s -> Random.init s | None -> () in

      (* Create model function for new API *)
      let model_fn = logits_fn in

      (* Generate using new sampler API *)
      let output =
        Sampler.generate ~model:model_fn ~input_ids:prompt_ids
          ~generation_config:config ()
      in

      (* Get generated IDs *)
      let generated_ids =
        match output.sequences with seq :: _ -> seq | [] -> []
      in

      (* Decode tokens back to text *)
      let decoded_tokens =
        List.filter_map
          (fun id -> Saga_tokenizers.Vocab.get_token vocab id)
          generated_ids
      in

      (* Join tokens - for chars no space, for words add space *)
      String.concat "" decoded_tokens

(** Calculate log-probability of a sequence *)
let score model text =
  match model with
  | Ngram { backend = None; _ } -> failwith "score: model not trained"
  | Ngram { vocab = None; _ } -> failwith "score: model vocab not built"
  | Ngram { backend = Some backend; vocab = Some vocab; tokenizer; _ } ->
      (* Tokenize and encode text *)
      let tokens = Saga_tokenizers.Tokenizer.run tokenizer text in
      let ids =
        List.filter_map
          (fun token -> Saga_tokenizers.Vocab.get_index vocab token)
          tokens
        |> Array.of_list
      in

      (* Calculate log probability using n-gram model *)
      let log_prob = ref 0.0 in
      let n = match model with Ngram { n; _ } -> n in

      for i = 0 to Array.length ids - 1 do
        (* Get context for position i *)
        let context =
          if i = 0 then [||]
          else
            let context_len = min (n - 1) i in
            Array.sub ids (i - context_len) context_len
        in

        (* Get logits and add log prob of actual token *)
        let logits = Saga_models.Ngram.logits backend ~context in
        log_prob := !log_prob +. logits.(ids.(i))
      done;

      !log_prob

(** Calculate perplexity of text *)
let perplexity model text =
  let log_prob = score model text in
  (* Get number of tokens *)
  match model with
  | Ngram { tokenizer; _ } ->
      let tokens = Saga_tokenizers.Tokenizer.run tokenizer text in
      let n_tokens = List.length tokens in
      if n_tokens = 0 then infinity
      else exp (-.log_prob /. float_of_int n_tokens)

(** Batched perplexities *)
let perplexities model texts = List.map (perplexity model) texts

(** Save model to file *)
let save model filename =
  match model with
  | Ngram { backend = Some backend; vocab = Some vocab; _ } ->
      (* Save backend model *)
      Saga_models.Ngram.save backend (filename ^ ".ngram");
      (* Save vocabulary *)
      Saga_tokenizers.vocab_save vocab (filename ^ ".vocab")
  | Ngram { backend = None; _ } -> failwith "save: model not trained"
  | Ngram { vocab = None; _ } -> failwith "save: model vocab not built"

(** Load model from file *)
let load filename =
  (* This is a simplified version - in practice we'd need to save/load the full
     model configuration including tokenizer settings *)
  let backend = Saga_models.Ngram.load (filename ^ ".ngram") in
  let vocab = Saga_tokenizers.vocab_load (filename ^ ".vocab") in
  let _ = Saga_models.Ngram.stats backend in

  (* Infer n from the model - this is a simplification *)
  (* In a real implementation, we'd save the full config *)
  Ngram
    {
      n = 2;
      (* Default, should be saved/loaded *)
      smoothing = 0.01;
      (* Default, should be saved/loaded *)
      min_freq = 1;
      specials = [ "<bos>"; "<eos>" ];
      tokenizer = Obj.magic Saga_tokenizers.Tokenizer.chars;
      (* Default - force coercion *)
      backend = Some backend;
      vocab = Some vocab;
    }

(** Pipeline helper *)
let pipeline model texts ?(num_samples = 20) ?(temperature = 1.0) ?top_k ?top_p
    ?seed () =
  (* Train the model *)
  let trained_model = train model texts in

  (* Generate samples *)
  List.init num_samples (fun i ->
      (* Use different seed for each sample if seed provided *)
      let sample_seed = Option.map (fun s -> s + i) seed in
      let generated =
        generate trained_model ~num_tokens:50 ~temperature ?top_k ?top_p
          ?seed:sample_seed ()
      in
      let perp = perplexity trained_model generated in
      (generated, perp))
