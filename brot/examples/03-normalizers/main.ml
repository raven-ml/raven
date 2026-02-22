(* Text normalization.

   Normalizers transform text before tokenization: lowercasing, accent removal,
   Unicode normalization, whitespace cleanup, and model-specific preprocessing.
   They are the first stage in the tokenization pipeline. *)

open Brot

let show name norm text =
  let result = Normalizer.apply norm text in
  Printf.printf "  %-20s %S -> %S\n" name text result

let () =
  Printf.printf "=== Unicode Normalization ===\n\n";
  show "nfc" Normalizer.nfc "caf\xc3\xa9";
  show "nfkc" Normalizer.nfkc "\xef\xac\x81";

  (* fi ligature -> fi *)
  Printf.printf "\n=== Text Transforms ===\n\n";
  show "lowercase" Normalizer.lowercase "Hello WORLD";
  show "strip_accents" Normalizer.strip_accents
    "caf\xc3\xa9 r\xc3\xa9sum\xc3\xa9";
  show "strip" (Normalizer.strip ()) "  hello  ";
  show "replace"
    (Normalizer.replace ~pattern:"\\d+" ~replacement:"<NUM>")
    "I have 42 apples and 3 oranges";
  show "prepend" (Normalizer.prepend ">> ") "hello";

  Printf.printf "\n=== Model-specific ===\n\n";
  show "bert (default)" (Normalizer.bert ()) "Hello  WORLD!";
  show "bert (no lower)" (Normalizer.bert ~lowercase:false ()) "Hello  WORLD!";

  Printf.printf "\n=== Composition ===\n\n";
  let composed =
    Normalizer.sequence
      [ Normalizer.nfd; Normalizer.strip_accents; Normalizer.lowercase ]
  in
  show "nfd+strip+lower" composed "Caf\xc3\xa9 R\xc3\xa9sum\xc3\xa9";
  show "nfd+strip+lower" composed "HELLO";

  Printf.printf "\n=== Effect on Tokenization ===\n\n";
  let vocab =
    [ ("hello", 0); ("world", 1); ("cafe", 2); ("resume", 3); ("<unk>", 4) ]
  in
  let no_norm =
    word_level ~vocab ~unk_token:"<unk>" ~pre:(Pre_tokenizer.whitespace ()) ()
  in
  let with_norm =
    word_level ~vocab ~unk_token:"<unk>"
      ~pre:(Pre_tokenizer.whitespace ())
      ~normalizer:composed ()
  in

  let text = "HELLO Caf\xc3\xa9" in
  let enc1 = encode no_norm text in
  let enc2 = encode with_norm text in
  Printf.printf "  Text: %S\n" text;
  Printf.printf "  Without normalizer: [%s]\n"
    (String.concat "; "
       (List.map
          (fun s -> Printf.sprintf "%S" s)
          (Array.to_list (Encoding.tokens enc1))));
  Printf.printf "  With normalizer:    [%s]\n"
    (String.concat "; "
       (List.map
          (fun s -> Printf.sprintf "%S" s)
          (Array.to_list (Encoding.tokens enc2))))
