(* Special tokens and post-processing.

   Special tokens like [CLS] and [SEP] are inserted by post-processors after
   tokenization. They mark sequence boundaries and provide structure for model
   input. Sentence-pair encoding assigns different type IDs to each sequence. *)

open Brot

let print_encoding enc =
  let ids = Encoding.ids enc in
  let tokens = Encoding.tokens enc in
  let type_ids = Encoding.type_ids enc in
  let special = Encoding.special_tokens_mask enc in
  Printf.printf "  %-8s %-4s %-8s %-8s\n" "Token" "ID" "Type_ID" "Special";
  Printf.printf "  %s\n" (String.make 32 '-');
  for i = 0 to Encoding.length enc - 1 do
    Printf.printf "  %-8s %-4d %-8d %-8d\n" tokens.(i) ids.(i) type_ids.(i)
      special.(i)
  done

let () =
  let vocab =
    [
      ("[UNK]", 0);
      ("[CLS]", 1);
      ("[SEP]", 2);
      ("hello", 3);
      ("world", 4);
      ("how", 5);
      ("are", 6);
      ("you", 7);
    ]
  in
  let specials = List.map special [ "[CLS]"; "[SEP]"; "[UNK]" ] in
  let post = Post_processor.bert ~cls:("[CLS]", 1) ~sep:("[SEP]", 2) () in
  let tokenizer =
    word_level ~vocab ~unk_token:"[UNK]" ~specials ~post
      ~pre:(Pre_tokenizer.whitespace ())
      ()
  in

  (* Single sentence: [CLS] A [SEP] *)
  Printf.printf "=== Single Sentence ===\n";
  Printf.printf "Text: \"hello world\"\n\n";
  print_encoding (encode tokenizer "hello world");

  (* Sentence pair: [CLS] A [SEP] B [SEP] *)
  Printf.printf "\n=== Sentence Pair ===\n";
  Printf.printf "A: \"hello world\", B: \"how are you\"\n\n";
  print_encoding (encode tokenizer ~pair:"how are you" "hello world");

  (* Without special tokens *)
  Printf.printf "\n=== Without Special Tokens ===\n";
  Printf.printf "Text: \"hello world\" (add_special_tokens=false)\n\n";
  print_encoding (encode tokenizer ~add_special_tokens:false "hello world");

  (* Template-based post-processor *)
  Printf.printf "\n=== Template Post-processor ===\n";
  let template_post =
    Post_processor.template ~single:"[CLS] $A [SEP]"
      ~pair:"[CLS] $A [SEP] $B:1 [SEP]:1"
      ~special_tokens:[ ("[CLS]", 1); ("[SEP]", 2) ]
      ()
  in
  let tok2 =
    word_level ~vocab ~unk_token:"[UNK]" ~specials ~post:template_post
      ~pre:(Pre_tokenizer.whitespace ())
      ()
  in
  Printf.printf "Template: \"[CLS] $A [SEP] $B:1 [SEP]:1\"\n";
  Printf.printf "A: \"hello\", B: \"world\"\n\n";
  print_encoding (encode tok2 ~pair:"world" "hello")
