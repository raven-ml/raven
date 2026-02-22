(* BERT-style pipeline.

   Assembles all pipeline stages into a complete BERT-style tokenizer:
   normalizer, pre-tokenizer, WordPiece algorithm, post-processor, decoder,
   special tokens, padding, and truncation. *)

open Brot

let print_encoding label enc =
  let tokens = Encoding.tokens enc in
  let ids = Encoding.ids enc in
  let type_ids = Encoding.type_ids enc in
  let attn = Encoding.attention_mask enc in
  Printf.printf "%s\n" label;
  Printf.printf "  tokens:    [%s]\n"
    (String.concat ", "
       (List.map (fun s -> Printf.sprintf "%S" s) (Array.to_list tokens)));
  Printf.printf "  ids:       [%s]\n"
    (String.concat ", " (Array.to_list (Array.map string_of_int ids)));
  Printf.printf "  type_ids:  [%s]\n"
    (String.concat ", " (Array.to_list (Array.map string_of_int type_ids)));
  Printf.printf "  attn_mask: [%s]\n"
    (String.concat ", " (Array.to_list (Array.map string_of_int attn)));
  print_newline ()

let () =
  (* Build a BERT-style vocabulary *)
  let vocab =
    [
      ("[PAD]", 0);
      ("[UNK]", 1);
      ("[CLS]", 2);
      ("[SEP]", 3);
      ("the", 4);
      ("cat", 5);
      ("sat", 6);
      ("on", 7);
      ("mat", 8);
      ("dog", 9);
      ("play", 10);
      ("##ing", 11);
      ("##ed", 12);
      ("is", 13);
      ("a", 14);
      ("good", 15);
      ("great", 16);
      ("un", 17);
      ("##happy", 18);
      ("friend", 19);
      ("##s", 20);
      ("how", 21);
      ("are", 22);
      ("you", 23);
    ]
  in
  let specials = List.map special [ "[PAD]"; "[UNK]"; "[CLS]"; "[SEP]" ] in

  (* Assemble the full pipeline *)
  let tokenizer =
    wordpiece ~vocab ~unk_token:"[UNK]"
      ~normalizer:(Normalizer.bert ~lowercase:true ())
      ~pre:(Pre_tokenizer.bert ())
      ~post:(Post_processor.bert ~cls:("[CLS]", 2) ~sep:("[SEP]", 3) ())
      ~decoder:(Decoder.wordpiece ()) ~specials ~pad_token:"[PAD]" ()
  in

  (* Inspect the tokenizer *)
  Printf.printf "=== Tokenizer Configuration ===\n";
  Format.printf "%a@.@." pp tokenizer;

  (* Single sentence *)
  Printf.printf "=== Single Sentence ===\n\n";
  print_encoding "\"The Cat is Playing\""
    (encode tokenizer "The Cat is Playing");

  (* Sentence pair *)
  Printf.printf "=== Sentence Pair ===\n\n";
  print_encoding "A: \"the cat sat\", B: \"how are you\""
    (encode tokenizer ~pair:"how are you" "the cat sat");

  (* Batch with padding *)
  Printf.printf "=== Padded Batch ===\n\n";
  let batch =
    encode_batch tokenizer ~padding:(padding `Batch_longest)
      [ "the cat"; "the cat sat on a mat"; "good" ]
  in
  List.iteri (fun i enc -> print_encoding (Printf.sprintf "[%d]" i) enc) batch;

  (* Sentence pairs batch with padding and truncation *)
  Printf.printf "=== Sentence Pairs (pad=12, trunc=12) ===\n\n";
  let pairs =
    encode_pairs_batch tokenizer
      ~padding:(padding (`Fixed 12))
      ~truncation:(truncation 12)
      [ ("the cat sat", "how are you"); ("good dog", "is a friend") ]
  in
  List.iteri
    (fun i enc -> print_encoding (Printf.sprintf "pair[%d]" i) enc)
    pairs;

  (* Decoding *)
  Printf.printf "=== Decoding ===\n\n";
  let enc = encode tokenizer ~pair:"how are you" "the cat sat" in
  let ids = Encoding.ids enc in
  Printf.printf "  Full decode:   %S\n" (decode tokenizer ids);
  Printf.printf "  Skip specials: %S\n"
    (decode tokenizer ~skip_special_tokens:true ids)
