open Saga

let () =
  (* Sample text - we'll use a small Shakespeare excerpt for testing *)
  let text =
    "To be or not to be that is the question\n\
     Whether tis nobler in the mind to suffer\n\
     The slings and arrows of outrageous fortune\n\
     Or to take arms against a sea of troubles\n\
     And by opposing end them To die to sleep\n\
     No more and by a sleep to say we end\n\
     The heartache and the thousand natural shocks\n\
     That flesh is heir to Tis a consummation\n\
     Devoutly to be wished To die to sleep\n\
     To sleep perchance to dream ay theres the rub"
  in
  (* Create a tokenizer trained for words on this text *)
  let tok = Tokenizer.create ~model:(Models.word_level ()) in
  Tokenizer.set_pre_tokenizer tok (Some (Pre_tokenizers.whitespace ()));
  (* Train a simple word-level vocab from the text *)
  let seq = Seq.return text in
  Tokenizer.train_from_iterator tok seq ~trainer:(Trainers.word_level ()) ();

  (* Tokenize training data *)
  let sequences =
    [ Tokenizer.encode tok ~sequence:(Either.Left text) () |> Encoding.get_ids ]
  in
  let model = Saga_models.Ngram.of_sequences ~order:2 sequences in

  (* Generate text with different starting prompts *)
  Printf.printf "=== Bigram Language Model Demo ===\n\n";
  Printf.printf "Original text (first 100 chars):\n%s...\n\n"
    (String.sub text 0 (min 100 (String.length text)));

  (* Generate with different temperatures *)
  Printf.printf "Generated text (temperature=0.5, prompt=\"To be\"):\n";
  let generator config prompt =
    let logits_fn history =
      Saga_models.Ngram.logits model ~context:(Array.of_list history)
    in
    let prompt_ids =
      match prompt with
      | Some s ->
          Tokenizer.encode tok ~sequence:(Either.Left s) ()
          |> Encoding.get_ids |> Array.to_list
      | None -> []
    in
    let output =
      Sampler.generate ~model:logits_fn ~input_ids:prompt_ids
        ~generation_config:config ()
    in
    match output.sequences with
    | seq :: _ ->
        seq
        |> List.filter_map (Models.id_to_token (Tokenizer.get_model tok))
        |> String.concat " "
    | [] -> ""
  in
  let config = Sampler.default |> Sampler.with_temperature 0.5 in
  Printf.printf "%s\n\n" (generator config (Some "To be"));

  Printf.printf "Generated text (temperature=1.0, prompt=\"The\"):\n";
  let config =
    Sampler.default |> Sampler.with_temperature 1.0 |> Sampler.with_top_k 10
  in
  Printf.printf "%s\n\n" (generator config (Some "The"));

  Printf.printf "Generated text (temperature=1.5, no prompt):\n";
  let config = Sampler.default |> Sampler.with_temperature 1.5 in
  Printf.printf "%s\n" (generator config None)
