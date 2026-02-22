(* Loading a real GPT-2 tokenizer.

   Downloads GPT-2's vocabulary and merge files from HuggingFace, builds the
   full byte-level BPE pipeline, and demonstrates encoding, decoding, and
   subword inspection on real-world text. *)

open Brot

let download url dest =
  if not (Sys.file_exists dest) then (
    Printf.printf "Downloading %s...\n%!" (Filename.basename dest);
    Nx_io.Http.download ~url ~dest ())

let () =
  (* Download GPT-2 model files *)
  let cache = "/tmp/brot_gpt2" in
  if not (Sys.file_exists cache) then Sys.mkdir cache 0o755;
  let vocab_file = Filename.concat cache "vocab.json" in
  let merges_file = Filename.concat cache "merges.txt" in
  download "https://huggingface.co/gpt2/raw/main/vocab.json" vocab_file;
  download "https://huggingface.co/gpt2/raw/main/merges.txt" merges_file;

  (* Build the GPT-2 tokenizer: BPE with byte-level pre-tokenizer *)
  let tokenizer =
    from_model_file ~vocab:vocab_file ~merges:merges_file
      ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
      ~decoder:(Decoder.byte_level ()) ()
  in
  Printf.printf "\nVocabulary: %d tokens\n\n" (vocab_size tokenizer);

  (* Encode text *)
  let text = "Hello world! GPT-2 is amazing." in
  let enc = encode tokenizer text in
  Printf.printf "Text:    %S\n" text;
  Printf.printf "Tokens:  [%s]\n"
    (String.concat "; "
       (List.map
          (fun s -> Printf.sprintf "%S" s)
          (Array.to_list (Encoding.tokens enc))));
  Printf.printf "IDs:     [%s]\n"
    (String.concat "; "
       (Array.to_list (Array.map string_of_int (Encoding.ids enc))));

  (* Decode back *)
  let decoded = decode tokenizer (Encoding.ids enc) in
  Printf.printf "Decoded: %S\n" decoded;
  Printf.printf "Round-trip: %b\n\n" (String.equal text decoded);

  (* Subword splitting: see how a long word is broken down *)
  Printf.printf "=== Subword Splitting ===\n\n";
  List.iter
    (fun word ->
      let e = encode tokenizer word in
      let tokens = Encoding.tokens e in
      Printf.printf "  %-20s -> %d tokens: [%s]\n" (Printf.sprintf "%S" word)
        (Array.length tokens)
        (String.concat ", "
           (List.map (fun s -> Printf.sprintf "%S" s) (Array.to_list tokens))))
    [ "tokenization"; "transformer"; "GPT"; "Hello"; "supercalifragilistic" ];

  (* Batch encoding *)
  Printf.printf "\n=== Batch Encoding ===\n\n";
  let texts =
    [
      "The quick brown fox";
      "jumps over the lazy dog";
      "Machine learning is fun";
    ]
  in
  let batch = encode_batch tokenizer texts in
  List.iter2
    (fun text enc ->
      Printf.printf "  %-30s -> %d tokens\n" (Printf.sprintf "%S" text)
        (Encoding.length enc))
    texts batch;

  (* Offsets: map tokens back to source text *)
  Printf.printf "\n=== Token Offsets ===\n\n";
  let text2 = "Hello, world!" in
  let enc2 = encode tokenizer text2 in
  Printf.printf "Text: %S\n" text2;
  let tokens = Encoding.tokens enc2 in
  let offsets = Encoding.offsets enc2 in
  for i = 0 to Encoding.length enc2 - 1 do
    let s, e = offsets.(i) in
    Printf.printf "  %-8s  offsets=(%d, %d)  source=%S\n" tokens.(i) s e
      (String.sub text2 s (e - s))
  done
