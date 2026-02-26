(* Pre-tokenization.

   Pre-tokenizers split text into fragments before vocabulary-based
   tokenization. Each fragment carries byte offsets into the original text.
   Different strategies produce different splits, affecting how subword
   algorithms see the input. *)

open Brot

let show name pre text =
  let result = Pre_tokenizer.pre_tokenize pre text in
  Printf.printf "  %-24s %S\n" name text;
  List.iter
    (fun (fragment, (s, e)) -> Printf.printf "    %S (%d, %d)\n" fragment s e)
    result;
  print_newline ()

let () =
  let text = "Hello, world! It's 2026." in
  Printf.printf "=== Common Pre-tokenizers ===\n\n";
  Printf.printf "Text: %S\n\n" text;

  show "whitespace" (Pre_tokenizer.whitespace ()) text;
  show "whitespace_split" (Pre_tokenizer.whitespace_split ()) text;
  show "bert" (Pre_tokenizer.bert ()) text;
  show "punctuation" (Pre_tokenizer.punctuation ()) text;
  show "digits (individual)"
    (Pre_tokenizer.digits ~individual_digits:true ())
    text;
  show "digits (grouped)"
    (Pre_tokenizer.digits ~individual_digits:false ())
    text;

  Printf.printf "=== Delimiter-based ===\n\n";

  show "char_delimiter ','" (Pre_tokenizer.char_delimiter ',') "a,b,c";
  show "split on '::'" (Pre_tokenizer.split ~pattern:"::" ()) "mod::func::arg";
  show "fixed_length 3" (Pre_tokenizer.fixed_length 3) "abcdefgh";
  show "metaspace" (Pre_tokenizer.metaspace ()) "Hello world today";

  Printf.printf "=== Composition ===\n\n";

  let composed =
    Pre_tokenizer.sequence
      [
        Pre_tokenizer.whitespace_split ();
        Pre_tokenizer.punctuation ~behavior:`Isolated ();
      ]
  in
  show "whitespace + punctuation" composed text
