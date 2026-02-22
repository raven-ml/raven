(* Tokenization algorithms.

   Five algorithms compared side-by-side: BPE (merge-based), WordPiece (greedy
   longest-match), Unigram (probabilistic), word-level (whole words), and
   character-level (per-byte). Each splits text differently. *)

open Brot

let show name tokenizer text =
  let encoding = encode tokenizer text in
  let tokens = Encoding.tokens encoding in
  let ids = Encoding.ids encoding in
  Printf.printf "  %-12s tokens=[%s]  ids=[%s]\n" name
    (String.concat ", "
       (List.map (fun s -> Printf.sprintf "%S" s) (Array.to_list tokens)))
    (String.concat ", " (Array.to_list (Array.map string_of_int ids)))

let () =
  (* --- BPE: iterative merge-based subwords --- *)
  let bpe_tok =
    bpe
      ~vocab:
        [
          ("p", 0);
          ("l", 1);
          ("a", 2);
          ("y", 3);
          ("i", 4);
          ("n", 5);
          ("g", 6);
          ("pl", 7);
          ("ay", 8);
          ("in", 9);
          ("ng", 10);
          ("play", 11);
          ("ing", 12);
          ("playing", 13);
        ]
      ~merges:
        [
          ("p", "l");
          ("a", "y");
          ("i", "n");
          ("n", "g");
          ("pl", "ay");
          ("in", "g");
          ("play", "ing");
        ]
      ()
  in

  (* --- WordPiece: greedy longest-match with ## prefix --- *)
  let wp_tok =
    wordpiece
      ~vocab:
        [
          ("[UNK]", 0);
          ("play", 1);
          ("##ing", 2);
          ("##ed", 3);
          ("run", 4);
          ("##ning", 5);
          ("un", 6);
          ("##known", 7);
        ]
      ~unk_token:"[UNK]" ()
  in

  (* --- Unigram: probabilistic segmentation --- *)
  let uni_tok =
    unigram
      ~vocab:
        [
          ("playing", -0.5);
          ("play", -1.0);
          ("ing", -1.5);
          ("p", -3.0);
          ("l", -3.0);
          ("a", -3.0);
          ("y", -3.0);
          ("i", -3.0);
          ("n", -3.0);
          ("g", -3.0);
        ]
      ()
  in

  (* --- Word-level: whole words only --- *)
  let wl_tok =
    word_level
      ~vocab:[ ("playing", 0); ("hello", 1); ("<unk>", 2) ]
      ~unk_token:"<unk>"
      ~pre:(Pre_tokenizer.whitespace ())
      ()
  in

  (* --- Character-level: one byte per token --- *)
  let char_tok = chars () in

  Printf.printf "=== Encoding %S ===\n\n" "playing";
  show "BPE" bpe_tok "playing";
  show "WordPiece" wp_tok "playing";
  show "Unigram" uni_tok "playing";
  show "Word-level" wl_tok "playing";
  show "Chars" char_tok "playing";

  Printf.printf "\n=== Encoding %S ===\n\n" "running";
  show "WordPiece" wp_tok "running";
  show "Chars" char_tok "running";

  Printf.printf "\n=== Encoding %S (unknown word) ===\n\n" "unknown";
  show "WordPiece" wp_tok "unknown";
  show "Word-level" wl_tok "unknown";
  show "Chars" char_tok "unknown";

  Printf.printf "\n=== Vocabulary sizes ===\n\n";
  Printf.printf "  BPE:        %d\n" (vocab_size bpe_tok);
  Printf.printf "  WordPiece:  %d\n" (vocab_size wp_tok);
  Printf.printf "  Unigram:    %d\n" (vocab_size uni_tok);
  Printf.printf "  Word-level: %d\n" (vocab_size wl_tok);
  Printf.printf "  Chars:      %d (byte range 0-255)\n" (vocab_size char_tok)
