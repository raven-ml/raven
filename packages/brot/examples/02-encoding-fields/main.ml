(* Understanding encodings.

   An Encoding bundles token IDs with alignment metadata: byte offsets, word
   indices, segment type IDs, attention masks, and special-token flags. All
   arrays share the same length. *)

open Brot

let print_encoding enc =
  let ids = Encoding.ids enc in
  let tokens = Encoding.tokens enc in
  let offsets = Encoding.offsets enc in
  let word_ids = Encoding.word_ids enc in
  let type_ids = Encoding.type_ids enc in
  let attn = Encoding.attention_mask enc in
  let special = Encoding.special_tokens_mask enc in

  Printf.printf "%-6s %-10s %-4s %-12s %-8s %-8s %-6s %-8s\n" "Index" "Token"
    "ID" "Offsets" "Word_ID" "Type_ID" "Attn" "Special";
  Printf.printf "%s\n" (String.make 66 '-');

  for i = 0 to Encoding.length enc - 1 do
    let s, e = offsets.(i) in
    let word =
      match word_ids.(i) with Some w -> string_of_int w | None -> "-"
    in
    Printf.printf "%-6d %-10s %-4d (%2d, %2d)     %-8s %-8d %-6d %-8d\n" i
      tokens.(i) ids.(i) s e word type_ids.(i) attn.(i) special.(i)
  done

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
    word_level ~vocab ~unk_token:"[UNK]" ~pre:(Pre_tokenizer.whitespace ()) ()
  in

  let text = "hello world is great" in
  Printf.printf "Text: %S\n" text;
  Printf.printf "Length: %d tokens\n\n"
    (Encoding.length (encode tokenizer text));
  print_encoding (encode tokenizer text);

  (* Show what happens with unknown words *)
  Printf.printf "\n--- Unknown words ---\n\n";
  let text2 = "hello universe" in
  Printf.printf "Text: %S\n" text2;
  Printf.printf "Length: %d tokens\n\n"
    (Encoding.length (encode tokenizer text2));
  print_encoding (encode tokenizer text2);

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
  print_encoding (encode wp text3)
