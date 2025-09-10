open Saga

let () =
  (* Create a simple tokenizer *)
  let tok = Tokenizer.words in

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

  (* Train a bigram model *)
  let model = ngram ~n:2 ~tokenizer:tok () in
  let model = train model [ text ] in

  (* Generate text with different starting prompts *)
  Printf.printf "=== Bigram Language Model Demo ===\n\n";
  Printf.printf "Original text (first 100 chars):\n%s...\n\n"
    (String.sub text 0 (min 100 (String.length text)));

  (* Generate with different temperatures *)
  Printf.printf "Generated text (temperature=0.5, prompt=\"To be\"):\n";
  let generated =
    generate model ~num_tokens:50 ~temperature:0.5 ~prompt:"To be" ()
  in
  Printf.printf "%s\n\n" generated;

  Printf.printf "Generated text (temperature=1.0, prompt=\"The\"):\n";
  let generated =
    generate model ~num_tokens:50 ~temperature:1.0 ~top_k:10 ~prompt:"The" ()
  in
  Printf.printf "%s\n\n" generated;

  Printf.printf "Generated text (temperature=1.5, no prompt):\n";
  let generated = generate model ~num_tokens:50 ~temperature:1.5 () in
  Printf.printf "%s\n" generated
