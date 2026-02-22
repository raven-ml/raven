(* Training tokenizers.

   Train new tokenizers from a text corpus. Each algorithm learns a different
   vocabulary: BPE learns merge rules, WordPiece learns subword prefixes,
   word-level collects unique words, and Unigram learns token probabilities. *)

open Brot

let corpus =
  [
    "the cat sat on the mat";
    "the dog sat on the log";
    "the cat and the dog are friends";
    "cats and dogs play together";
    "the cat plays with the dog";
    "playing in the park is fun";
    "the park has many cats and dogs";
    "friends play in the park together";
  ]

let show_trained name tokenizer test_texts =
  Printf.printf "--- %s (vocab_size=%d) ---\n" name (vocab_size tokenizer);
  List.iter
    (fun text ->
      let enc = encode tokenizer text in
      Printf.printf "  %S -> [%s]\n" text
        (String.concat ", "
           (List.map
              (fun s -> Printf.sprintf "%S" s)
              (Array.to_list (Encoding.tokens enc)))))
    test_texts;
  print_newline ()

let () =
  let data = `Seq (List.to_seq corpus) in
  let test_texts = [ "the cat plays"; "dogs are friends" ] in

  Printf.printf "Training corpus: %d sentences\n\n" (List.length corpus);

  (* Train BPE: learns merge rules by iteratively combining frequent pairs *)
  let bpe_tok =
    train_bpe data ~vocab_size:100 ~show_progress:false
      ~pre:(Pre_tokenizer.whitespace ())
  in
  show_trained "BPE" bpe_tok test_texts;

  (* Train WordPiece: learns subword prefixes (## for continuation tokens) *)
  let wp_tok =
    train_wordpiece data ~vocab_size:100 ~show_progress:false
      ~pre:(Pre_tokenizer.whitespace ())
  in
  show_trained "WordPiece" wp_tok test_texts;

  (* Train word-level: each unique word is a token *)
  let wl_tok =
    train_wordlevel data ~vocab_size:50 ~show_progress:false
      ~pre:(Pre_tokenizer.whitespace ())
  in
  show_trained "Word-level" wl_tok test_texts;

  (* Train Unigram: probabilistic subword segmentation *)
  let uni_tok = train_unigram data ~vocab_size:100 ~show_progress:false in
  show_trained "Unigram" uni_tok test_texts;

  (* Training with special tokens *)
  Printf.printf "=== Training with Special Tokens ===\n\n";
  let wp_with_specials =
    train_wordpiece data ~vocab_size:100 ~show_progress:false
      ~pre:(Pre_tokenizer.whitespace ())
      ~specials:[ special "[CLS]"; special "[SEP]"; special "[PAD]" ]
      ~pad_token:"[PAD]"
  in
  Printf.printf "WordPiece with specials (vocab=%d):\n"
    (vocab_size wp_with_specials);
  let show_id tok name =
    Printf.printf "  %s id = %s\n" name
      (match token_to_id tok name with
      | Some id -> string_of_int id
      | None -> "N/A")
  in
  show_id wp_with_specials "[CLS]";
  show_id wp_with_specials "[SEP]";
  show_id wp_with_specials "[PAD]";

  (* Add a post-processor to insert special tokens during encoding *)
  Printf.printf "\n  Encoding with post-processor:\n";
  let wp_full =
    train_wordpiece data ~vocab_size:100 ~show_progress:false
      ~pre:(Pre_tokenizer.whitespace ())
      ~post:
        (Post_processor.bert
           ~cls:("[CLS]", Option.get (token_to_id wp_with_specials "[CLS]"))
           ~sep:("[SEP]", Option.get (token_to_id wp_with_specials "[SEP]"))
           ())
      ~specials:[ special "[CLS]"; special "[SEP]"; special "[PAD]" ]
      ~pad_token:"[PAD]"
  in
  let enc = encode wp_full "the cat plays" in
  Printf.printf "  %S -> [%s]\n" "the cat plays"
    (String.concat ", "
       (List.map
          (fun s -> Printf.sprintf "%S" s)
          (Array.to_list (Encoding.tokens enc))))
