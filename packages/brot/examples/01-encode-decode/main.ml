(* Encode and decode.

   The simplest possible tokenization: convert text to token IDs and back.
   Demonstrates creating a BPE tokenizer from an inline vocabulary and merge
   rules, encoding text, inspecting tokens and IDs, and decoding. *)

open Brot

let () =
  (* Build a small BPE tokenizer. The vocabulary maps token strings to IDs.
     Merge rules define which character pairs to combine, in priority order. *)
  let vocab =
    [
      ("h", 0);
      ("e", 1);
      ("l", 2);
      ("o", 3);
      (" ", 4);
      ("w", 5);
      ("r", 6);
      ("d", 7);
      ("he", 8);
      ("ll", 9);
      ("llo", 10);
      ("hello", 11);
      ("wo", 12);
      ("rl", 13);
      ("rld", 14);
      ("world", 15);
    ]
  in
  let merges =
    [
      ("h", "e");
      ("l", "l");
      ("ll", "o");
      ("he", "llo");
      ("w", "o");
      ("r", "l");
      ("rl", "d");
      ("wo", "rld");
    ]
  in
  let tokenizer = bpe ~vocab ~merges () in

  (* Encode text into an Encoding *)
  let text = "hello world" in
  let encoding = encode tokenizer text in
  let ids = Encoding.ids encoding in
  let tokens = Encoding.tokens encoding in

  Printf.printf "Text:    %S\n" text;
  Printf.printf "Tokens:  [%s]\n"
    (String.concat "; "
       (List.map (fun s -> Printf.sprintf "%S" s) (Array.to_list tokens)));
  Printf.printf "IDs:     [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int ids)));

  (* Decode token IDs back to text *)
  let decoded = decode tokenizer ids in
  Printf.printf "Decoded: %S\n\n" decoded;

  Printf.printf "Round-trip matches: %b\n\n" (String.equal text decoded);

  (* Try another text -- unknown characters become individual tokens *)
  let text2 = "hello" in
  let enc2 = encode tokenizer text2 in
  Printf.printf "Text:    %S\n" text2;
  Printf.printf "Tokens:  [%s]\n"
    (String.concat "; "
       (List.map
          (fun s -> Printf.sprintf "%S" s)
          (Array.to_list (Encoding.tokens enc2))));
  Printf.printf "IDs:     [%s]\n"
    (String.concat "; "
       (Array.to_list (Array.map string_of_int (Encoding.ids enc2))))
