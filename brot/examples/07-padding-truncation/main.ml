(* Padding and truncation.

   Batch processing requires uniform sequence lengths. Padding extends short
   sequences with pad tokens; truncation trims long ones. The attention mask
   distinguishes real tokens from padding. *)

open Brot

let print_batch label encodings =
  Printf.printf "%s\n" label;
  List.iteri
    (fun i enc ->
      let ids = Encoding.ids enc in
      let attn = Encoding.attention_mask enc in
      Printf.printf "  [%d] ids=[%s]  attn=[%s]\n" i
        (String.concat ", " (Array.to_list (Array.map string_of_int ids)))
        (String.concat ", " (Array.to_list (Array.map string_of_int attn))))
    encodings;
  print_newline ()

let () =
  let vocab =
    [
      ("[PAD]", 0);
      ("<unk>", 1);
      ("hello", 2);
      ("world", 3);
      ("how", 4);
      ("are", 5);
      ("you", 6);
      ("doing", 7);
      ("today", 8);
    ]
  in
  let tokenizer =
    word_level ~vocab ~unk_token:"<unk>"
      ~specials:[ special "[PAD]" ]
      ~pad_token:"[PAD]"
      ~pre:(Pre_tokenizer.whitespace ())
      ()
  in

  let texts = [ "hello"; "hello world"; "how are you doing today" ] in
  Printf.printf "Texts:\n";
  List.iteri (fun i t -> Printf.printf "  [%d] %S\n" i t) texts;
  print_newline ();

  (* No padding *)
  print_batch "=== No Padding ===" (encode_batch tokenizer texts);

  (* Fixed-length padding *)
  print_batch "=== Fixed Padding (length=6) ==="
    (encode_batch tokenizer ~padding:(padding (`Fixed 6)) texts);

  (* Batch-longest padding *)
  print_batch "=== Batch Longest Padding ==="
    (encode_batch tokenizer ~padding:(padding `Batch_longest) texts);

  (* Left padding *)
  print_batch "=== Left Padding (length=6) ==="
    (encode_batch tokenizer
       ~padding:(padding ~direction:`Left (`Fixed 6))
       texts);

  (* Truncation *)
  print_batch "=== Truncation (max_length=3) ==="
    (encode_batch tokenizer ~truncation:(truncation 3) texts);

  (* Padding + Truncation *)
  print_batch "=== Padding + Truncation (pad=4, trunc=4) ==="
    (encode_batch tokenizer
       ~padding:(padding (`Fixed 4))
       ~truncation:(truncation 4) texts)
