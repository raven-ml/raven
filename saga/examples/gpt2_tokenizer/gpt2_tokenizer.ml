(* GPT-2 Tokenizer Example

   Demonstrates how to use GPT-2's BPE tokenizer with Saga. *)

open Saga

(* Download GPT-2 vocabulary files from HuggingFace *)
let download_gpt2_files () =
  let cache_dir = "/tmp/gpt2_tokenizer" in
  if not (Sys.file_exists cache_dir) then Unix.mkdir cache_dir 0o755;

  let vocab_file = Filename.concat cache_dir "vocab.json" in
  let merges_file = Filename.concat cache_dir "merges.txt" in

  let download url target =
    if not (Sys.file_exists target) then (
      Printf.printf "Downloading %s...\n" (Filename.basename target);
      ignore (Sys.command (Printf.sprintf "curl -sL -o %s '%s'" target url)))
  in

  download "https://huggingface.co/gpt2/raw/main/vocab.json" vocab_file;
  download "https://huggingface.co/gpt2/raw/main/merges.txt" merges_file;

  (vocab_file, merges_file)

let () =
  (* Get GPT-2 tokenizer files *)
  let vocab_file, merges_file = download_gpt2_files () in

  (* Create GPT-2 tokenizer with ByteLevel pre-tokenizer *)
  let pre_tokenizer = Pre_tokenizers.byte_level ~add_prefix_space:false () in
  let tokenizer =
    Tokenizer.from_model_file ~vocab:vocab_file ~merges:merges_file
      ~pre:pre_tokenizer ()
  in

  Printf.printf "\nGPT-2 Tokenizer Example\n";
  Printf.printf "═══════════════════════\n\n";

  (* Example 1: Tokenize some text *)
  let text = "Hello world! GPT-2 is amazing." in
  let encoding = Tokenizer.encode tokenizer text in
  let tokens = Encoding.get_ids encoding in

  Printf.printf "Text: \"%s\"\n" text;
  Printf.printf "Tokens (%d): " (Array.length tokens);
  Array.iter (Printf.printf "%d ") tokens;
  Printf.printf "\n\n";

  (* Example 2: Decode tokens back to text *)
  let decoded = Tokenizer.decode tokenizer tokens in
  Printf.printf "Decoded: \"%s\"\n" decoded;
  Printf.printf "(Note: 'Ġ' = space in GPT-2's encoding)\n\n";

  (* Example 3: See how a word gets tokenized *)
  let word = "tokenization" in
  let word_encoding = Tokenizer.encode tokenizer word in
  let word_tokens = Encoding.get_ids word_encoding in
  let word_token_strings = Encoding.get_tokens word_encoding in

  Printf.printf "The word \"%s\" becomes %d tokens:\n" word
    (Array.length word_tokens);
  Array.iteri
    (fun i id ->
      let decoded_token = word_token_strings.(i) in
      Printf.printf "  %d. \"%s\" (ID: %d)\n" (i + 1) decoded_token id)
    word_tokens
