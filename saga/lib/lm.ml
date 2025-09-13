(** High-level language model API *)

(** Simple vocabulary module for LM *)
module Vocab = struct
  type t = {
    token_to_id : (string, int) Hashtbl.t;
    id_to_token : (int, string) Hashtbl.t;
    eos_idx : int;
  }

  let create tokens ~min_freq =
    (* Count token frequencies *)
    let freq_map = Hashtbl.create 1024 in
    List.iter
      (fun token ->
        let count = try Hashtbl.find freq_map token with Not_found -> 0 in
        Hashtbl.replace freq_map token (count + 1))
      tokens;

    (* Filter by min_freq and create sorted list *)
    let filtered_tokens =
      Hashtbl.fold
        (fun token count acc ->
          if count >= min_freq then (token, count) :: acc else acc)
        freq_map []
      |> List.sort (fun (_, c1) (_, c2) -> compare c2 c1)
      (* Sort by frequency descending *)
    in

    (* Create vocabulary mappings *)
    let token_to_id = Hashtbl.create (List.length filtered_tokens) in
    let id_to_token = Hashtbl.create (List.length filtered_tokens) in
    let _ =
      List.fold_left
        (fun id (token, _) ->
          Hashtbl.add token_to_id token id;
          Hashtbl.add id_to_token id token;
          id + 1)
        0 filtered_tokens
    in

    let eos_idx =
      try Hashtbl.find token_to_id "<eos>"
      with Not_found -> (
        try Hashtbl.find token_to_id "."
        with Not_found ->
          Hashtbl.length token_to_id - 1 (* Use last token as fallback *))
    in
    { token_to_id; id_to_token; eos_idx }

  let get_index vocab token =
    try Some (Hashtbl.find vocab.token_to_id token) with Not_found -> None

  let get_token vocab id =
    try Some (Hashtbl.find vocab.id_to_token id) with Not_found -> None

  let eos_idx vocab = vocab.eos_idx

  let save vocab filename =
    let oc = open_out filename in
    Hashtbl.iter
      (fun token id -> Printf.fprintf oc "%s\t%d\n" token id)
      vocab.token_to_id;
    close_out oc

  let load filename =
    let ic = open_in filename in
    let token_to_id = Hashtbl.create 1024 in
    let id_to_token = Hashtbl.create 1024 in
    let rec read_lines () =
      try
        let line = input_line ic in
        (match String.split_on_char '\t' line with
        | [ token; id_str ] ->
            let id = int_of_string id_str in
            Hashtbl.add token_to_id token id;
            Hashtbl.add id_to_token id token
        | _ -> ());
        read_lines ()
      with End_of_file -> close_in ic
    in
    read_lines ();
    let eos_idx =
      try Hashtbl.find token_to_id "<eos>"
      with Not_found -> (
        try Hashtbl.find token_to_id "."
        with Not_found -> Hashtbl.length token_to_id - 1)
    in
    { token_to_id; id_to_token; eos_idx }
end

(** Model type - wraps different backend implementations *)
type model =
  | Ngram : {
      n : int;
      smoothing : float;
      min_freq : int;
      specials : string list;
      tokenizer : Saga_tokenizers.Tokenizer.t;
      backend : Saga_models.Ngram.t option; (* None until trained *)
      vocab : Vocab.t option; (* None until trained *)
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
          tokenizer =
            Saga_tokenizers.Tokenizer.create
              ~model:(Saga_tokenizers.Models.chars ());
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
          (fun text ->
            let encoding =
              Saga_tokenizers.Tokenizer.encode config.tokenizer
                ~sequence:(Saga_tokenizers.Either.Left text) ()
            in
            Array.to_list (Saga_tokenizers.Encoding.get_tokens encoding))
          texts
      in

      (* Flatten to get all tokens for vocab building *)
      let all_tokens = List.concat tokenized_texts in

      (* Build vocabulary *)
      let vocab = Vocab.create all_tokens ~min_freq:config.min_freq in

      (* Convert texts to token IDs *)
      let encoded_texts =
        List.map
          (fun tokens ->
            List.filter_map (fun token -> Vocab.get_index vocab token) tokens
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
    ?(min_new_tokens = 0) ?(prompt = "") () =
  match model with
  | Ngram { backend = None; _ } -> failwith "generate: model not trained"
  | Ngram { vocab = None; _ } -> failwith "generate: model vocab not built"
  | Ngram
      { backend = Some backend; vocab = Some vocab; tokenizer; specials; n; _ }
    ->
      (* Prepare prompt *)
      let prompt_text = if prompt = "" then List.hd specials else prompt in

      (* EOS handling *)
      let eos_token =
        if List.length specials >= 2 then List.nth specials 1 else "<eos>"
      in
      let eos_id =
        match Vocab.get_index vocab eos_token with
        | Some id -> id
        | None -> Vocab.eos_idx vocab
      in

      (* Tokenize and encode the prompt to get starting tokens *)
      let prompt_encoding =
        Saga_tokenizers.Tokenizer.encode tokenizer
          ~sequence:(Saga_tokenizers.Either.Left prompt_text) ()
      in
      let prompt_tokens =
        Array.to_list (Saga_tokenizers.Encoding.get_tokens prompt_encoding)
      in
      let prompt_ids =
        List.filter_map (fun token -> Vocab.get_index vocab token) prompt_tokens
      in

      (* RNG *)
      let rng =
        match seed with
        | Some s -> Random.State.make [| s |]
        | None -> Random.State.make [| Random.bits () |]
      in

      let apply_temperature logits =
        if temperature = 1.0 then Array.copy logits
        else Array.map (fun x -> x /. temperature) logits
      in

      let apply_top_k logits =
        match top_k with
        | None -> Array.copy logits
        | Some k when k <= 0 || k >= Array.length logits -> Array.copy logits
        | Some k ->
            let idx = Array.init (Array.length logits) (fun i -> i) in
            Array.sort (fun i j -> compare logits.(j) logits.(i)) idx;
            let threshold = logits.(idx.(k - 1)) in
            Array.map
              (fun v -> if v < threshold then neg_infinity else v)
              logits
      in

      let apply_top_p logits =
        match top_p with
        | None -> Array.copy logits
        | Some p when p <= 0.0 || p >= 1.0 -> Array.copy logits
        | Some p ->
            let n = Array.length logits in
            let idx = Array.init n (fun i -> i) in
            Array.sort (fun i j -> compare logits.(j) logits.(i)) idx;
            let probs = Array.map exp logits in
            let sum = Array.fold_left ( +. ) 0.0 probs in
            let probs =
              if sum > 0. then Array.map (fun v -> v /. sum) probs else probs
            in
            let cum = ref 0.0 in
            let cutoff = ref neg_infinity in
            for r = 0 to n - 1 do
              let i = idx.(r) in
              cum := !cum +. probs.(i);
              if !cum >= p && !cutoff = neg_infinity then cutoff := logits.(i)
            done;
            Array.map (fun v -> if v < !cutoff then neg_infinity else v) logits
      in

      let sample_from_logits logits =
        let probs = Array.map exp logits in
        let sum = Array.fold_left ( +. ) 0.0 probs in
        let probs =
          if sum > 0. then Array.map (fun v -> v /. sum) probs else probs
        in
        let r = Random.State.float rng 1.0 in
        let acc = ref 0.0 in
        let choice = ref (Array.length probs - 1) in
        for i = 0 to Array.length probs - 1 do
          acc := !acc +. probs.(i);
          if !acc >= r && !choice = Array.length probs - 1 then choice := i
        done;
        !choice
      in

      let rec loop generated =
        if List.length generated >= num_tokens then List.rev generated
        else
          let ctx_arr =
            let base = List.rev (prompt_ids @ generated) |> Array.of_list in
            if n = 1 then [||]
            else
              let ctx_len = min (n - 1) (Array.length base) in
              Array.sub base (Array.length base - ctx_len) ctx_len
          in
          let logits0 = Saga_models.Ngram.logits backend ~context:ctx_arr in
          let logits1 = apply_temperature logits0 in
          let logits2 = apply_top_k logits1 in
          let logits3 = apply_top_p logits2 in
          let next = sample_from_logits logits3 in
          let new_len = List.length generated + 1 in
          if next = eos_id && new_len >= min_new_tokens then List.rev generated
          else loop (next :: generated)
      in
      let ids = loop [] in
      let tokens = List.filter_map (fun id -> Vocab.get_token vocab id) ids in
      String.concat "" tokens

let generate_ids model ?(num_tokens = 20) ?(temperature = 1.0) ?top_k ?top_p
    ?seed ?(min_new_tokens = 0) ?(prompt = "") () =
  match model with
  | Ngram { backend = None; _ } -> failwith "generate: model not trained"
  | Ngram { vocab = None; _ } -> failwith "generate: model vocab not built"
  | Ngram
      { backend = Some backend; vocab = Some vocab; tokenizer; specials; n; _ }
    ->
      let prompt_text = if prompt = "" then List.hd specials else prompt in
      let eos_token =
        if List.length specials >= 2 then List.nth specials 1 else "<eos>"
      in
      let eos_id =
        match Vocab.get_index vocab eos_token with
        | Some id -> id
        | None -> Vocab.eos_idx vocab
      in
      let prompt_encoding =
        Saga_tokenizers.Tokenizer.encode tokenizer
          ~sequence:(Saga_tokenizers.Either.Left prompt_text) ()
      in
      let prompt_tokens =
        Array.to_list (Saga_tokenizers.Encoding.get_tokens prompt_encoding)
      in
      let prompt_ids =
        List.filter_map (fun token -> Vocab.get_index vocab token) prompt_tokens
      in
      let rng =
        match seed with
        | Some s -> Random.State.make [| s |]
        | None -> Random.State.make [| Random.bits () |]
      in
      let apply_temperature logits =
        if temperature = 1.0 then Array.copy logits
        else Array.map (fun x -> x /. temperature) logits
      in
      let apply_top_k logits =
        match top_k with
        | None -> Array.copy logits
        | Some k when k <= 0 || k >= Array.length logits -> Array.copy logits
        | Some k ->
            let idx = Array.init (Array.length logits) (fun i -> i) in
            Array.sort (fun i j -> compare logits.(j) logits.(i)) idx;
            let threshold = logits.(idx.(k - 1)) in
            Array.map
              (fun v -> if v < threshold then neg_infinity else v)
              logits
      in
      let apply_top_p logits =
        match top_p with
        | None -> Array.copy logits
        | Some p when p <= 0.0 || p >= 1.0 -> Array.copy logits
        | Some p ->
            let n = Array.length logits in
            let idx = Array.init n (fun i -> i) in
            Array.sort (fun i j -> compare logits.(j) logits.(i)) idx;
            let probs = Array.map exp logits in
            let sum = Array.fold_left ( +. ) 0.0 probs in
            let probs =
              if sum > 0. then Array.map (fun v -> v /. sum) probs else probs
            in
            let cum = ref 0.0 in
            let cutoff = ref neg_infinity in
            for r = 0 to n - 1 do
              let i = idx.(r) in
              cum := !cum +. probs.(i);
              if !cum >= p && !cutoff = neg_infinity then cutoff := logits.(i)
            done;
            Array.map (fun v -> if v < !cutoff then neg_infinity else v) logits
      in
      let sample_from_logits logits =
        let probs = Array.map exp logits in
        let sum = Array.fold_left ( +. ) 0.0 probs in
        let probs =
          if sum > 0. then Array.map (fun v -> v /. sum) probs else probs
        in
        let r = Random.State.float rng 1.0 in
        let acc = ref 0.0 in
        let choice = ref (Array.length probs - 1) in
        for i = 0 to Array.length probs - 1 do
          acc := !acc +. probs.(i);
          if !acc >= r && !choice = Array.length probs - 1 then choice := i
        done;
        !choice
      in
      let rec loop generated =
        if List.length generated >= num_tokens then List.rev generated
        else
          let ctx_arr =
            let base = List.rev (prompt_ids @ generated) |> Array.of_list in
            if n = 1 then [||]
            else
              let ctx_len = min (n - 1) (Array.length base) in
              Array.sub base (Array.length base - ctx_len) ctx_len
          in
          let logits0 = Saga_models.Ngram.logits backend ~context:ctx_arr in
          let logits1 = apply_temperature logits0 in
          let logits2 = apply_top_k logits1 in
          let logits3 = apply_top_p logits2 in
          let next = sample_from_logits logits3 in
          let new_len = List.length generated + 1 in
          if next = eos_id && new_len >= min_new_tokens then List.rev generated
          else loop (next :: generated)
      in
      loop []

let decode_ids model ids =
  match model with
  | Ngram { vocab = None; _ } -> failwith "decode_ids: model vocab not built"
  | Ngram { vocab = Some vocab; _ } ->
      List.filter_map (fun id -> Vocab.get_token vocab id) ids

(** Score a text with the model *)
let score model text =
  match model with
  | Ngram { backend = None; _ } -> failwith "score: model not trained"
  | Ngram { vocab = None; _ } -> failwith "score: model vocab not built"
  | Ngram { backend = Some backend; vocab = Some vocab; tokenizer; _ } ->
      (* Tokenize the text *)
      let encoding =
        Saga_tokenizers.Tokenizer.encode tokenizer
          ~sequence:(Saga_tokenizers.Either.Left text) ()
      in
      let tokens =
        Array.to_list (Saga_tokenizers.Encoding.get_tokens encoding)
      in
      let ids =
        List.filter_map (fun token -> Vocab.get_index vocab token) tokens
        |> Array.of_list
      in
      (* Score using the backend *)
      Saga_models.Ngram.log_prob backend ids

(** Calculate perplexity of a text *)
let perplexity model text =
  let log_prob = score model text in
  (* Get token count for normalization *)
  match model with
  | Ngram { tokenizer; _ } ->
      let encoding =
        Saga_tokenizers.Tokenizer.encode tokenizer
          ~sequence:(Saga_tokenizers.Either.Left text) ()
      in
      let tokens =
        Array.to_list (Saga_tokenizers.Encoding.get_tokens encoding)
      in
      let n_tokens = List.length tokens in
      if n_tokens = 0 then infinity
      else exp (-.log_prob /. float_of_int n_tokens)

(** Calculate perplexities for multiple texts *)
let perplexities model texts = List.map (perplexity model) texts

(** Save model to file *)
let save model filename =
  match model with
  | Ngram { backend = Some backend; vocab = Some vocab; _ } ->
      (* Save backend *)
      Saga_models.Ngram.save backend filename;
      (* Save vocabulary *)
      Vocab.save vocab (filename ^ ".vocab")
  | _ -> failwith "save: model not trained"

(** Load model from file *)
let load filename =
  (* Load backend *)
  let backend = Saga_models.Ngram.load filename in
  (* Load vocabulary *)
  let vocab = Vocab.load (filename ^ ".vocab") in
  (* Get n from backend *)
  let n = Saga_models.Ngram.n backend in
  (* Create model with default tokenizer *)
  Ngram
    {
      n;
      smoothing = 0.01;
      (* Default, actual value is in backend *)
      min_freq = 1;
      (* Default *)
      specials = [ "<bos>"; "<eos>" ];
      (* Default *)
      tokenizer =
        Saga_tokenizers.Tokenizer.create
          ~model:(Saga_tokenizers.Models.chars ());
      backend = Some backend;
      vocab = Some vocab;
    }

(** Pipeline function for training and generating samples *)
let pipeline model texts ?(num_samples = 20) ?(temperature = 1.0) ?top_k ?top_p
    ?seed () =
  (* Train the model *)
  let trained_model = train model texts in
  (* Generate samples and compute their perplexities *)
  let samples =
    List.init num_samples (fun _ ->
        let generated =
          generate trained_model ~num_tokens:50 ~temperature ?top_k ?top_p ?seed
            ()
        in
        let perp = perplexity trained_model generated in
        (generated, perp))
  in
  (* Sort by perplexity (lower is better) *)
  List.sort (fun (_, p1) (_, p2) -> compare p1 p2) samples
