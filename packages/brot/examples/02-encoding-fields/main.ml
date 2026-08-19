(* Understanding encodings.

   An Encoding bundles token IDs with alignment metadata: byte offsets, word
   indices, segment type IDs, attention masks, and special-token flags. All
   arrays share the same length. [Encoding.pp] prints them side by side, one row
   per token. *)

open Brot

let () =
  (* Word-level tokenizer: each word maps to one token *)
  let vocab =
    [
      ("[UNK]", 0);
      ("hello", 1);
      ("world", 2);
      ("the", 3);
      ("is", 4);
      ("great", 5);
    ]
  in
  let tokenizer =
    word_level ~vocab ~unk_token:"[UNK]" ~pre:Pre_tokenizer.whitespace ()
  in

  let text = "hello world is great" in
  Printf.printf "Text: %S\n" text;
  Printf.printf "Length: %d tokens\n\n"
    (Encoding.length (encode tokenizer text));
  Format.printf "%a@." Encoding.pp (encode tokenizer text);

  (* Show what happens with unknown words *)
  Printf.printf "\n--- Unknown words ---\n\n";
  let text2 = "hello universe" in
  Printf.printf "Text: %S\n" text2;
  Printf.printf "Length: %d tokens\n\n"
    (Encoding.length (encode tokenizer text2));
  Format.printf "%a@." Encoding.pp (encode tokenizer text2);

  (* WordPiece: subword tokens have word_ids linking to the source word *)
  Printf.printf "\n--- Subword tokens (WordPiece) ---\n\n";
  let wp_vocab =
    [
      ("[UNK]", 0);
      ("play", 1);
      ("##ing", 2);
      ("##ed", 3);
      ("un", 4);
      ("##happy", 5);
    ]
  in
  let wp = wordpiece ~vocab:wp_vocab ~unk_token:"[UNK]" () in
  let text3 = "playing" in
  Printf.printf "Text: %S\n" text3;
  Printf.printf "Length: %d tokens\n\n" (Encoding.length (encode wp text3));
  Format.printf "%a@." Encoding.pp (encode wp text3)
