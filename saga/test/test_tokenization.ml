(* Tokenization tests for nx-text *)

open Alcotest
open Saga

(* ───── Basic Tokenization Tests ───── *)

let test_tokenize_words_simple () =
  let tokens = tokenize "Hello world!" in
  check (list string) "simple words" [ "Hello"; "world"; "!" ] tokens

let test_tokenize_words_punctuation () =
  let tokens = tokenize "don't stop, it's fun!" in
  check (list string) "words with punctuation"
    [ "don't"; "stop"; ","; "it's"; "fun"; "!" ]
    tokens

let test_tokenize_words_numbers () =
  let tokens = tokenize "I have 42 apples and 3.14 pies" in
  check (list string) "words with numbers"
    [ "I"; "have"; "42"; "apples"; "and"; "3"; "."; "14"; "pies" ]
    tokens

let test_tokenize_words_empty () =
  let tokens = tokenize "" in
  check (list string) "empty string" [] tokens

let test_tokenize_words_whitespace_only () =
  let tokens = tokenize "   \t\n  " in
  check (list string) "whitespace only" [] tokens

let test_tokenize_words_special_chars () =
  let tokens = tokenize "hello@world.com #ml $100 C++" in
  check (list string) "special characters"
    [ "hello"; "@"; "world"; "."; "com"; "#"; "ml"; "$"; "100"; "C"; "+"; "+" ]
    tokens

(* ───── Character Tokenization Tests ───── *)

let test_tokenize_chars_ascii () =
  let tokens = tokenize ~method_:`Chars "Hi!" in
  check (list string) "ASCII chars" [ "H"; "i"; "!" ] tokens

let test_tokenize_chars_unicode () =
  let tokens = tokenize ~method_:`Chars "Hello 👋 世界" in
  check int "unicode char count" 10 (List.length tokens);
  check string "wave emoji" "👋" (List.nth tokens 6);
  check string "first Chinese char" "世" (List.nth tokens 8)

let test_tokenize_chars_empty () =
  let tokens = tokenize ~method_:`Chars "" in
  check (list string) "empty string chars" [] tokens

(* ───── Regex Tokenization Tests ───── *)

let test_tokenize_regex_words () =
  let tokens = tokenize ~method_:(`Regex "\\w+") "hello-world test_123" in
  check (list string) "regex words" [ "hello"; "world"; "test_123" ] tokens

let test_tokenize_regex_custom () =
  let tokens = tokenize ~method_:(`Regex "\\w+|[^\\w\\s]+") "don't stop!" in
  check (list string) "regex custom" [ "don"; "'"; "t"; "stop"; "!" ] tokens

let test_tokenize_regex_no_match () =
  let tokens = tokenize ~method_:(`Regex "\\d+") "no numbers here" in
  check (list string) "regex no match" [] tokens

(* ───── Edge Cases ───── *)

let test_tokenize_long_text () =
  let text =
    String.concat " " (List.init 1000 (fun i -> Printf.sprintf "word%d" i))
  in
  let tokens = tokenize text in
  check int "long text token count" 1000 (List.length tokens)

let test_tokenize_repeated_punctuation () =
  let tokens = tokenize "wow!!! really???" in
  check (list string) "repeated punctuation"
    [ "wow"; "!"; "!"; "!"; "really"; "?"; "?"; "?" ]
    tokens

let test_tokenize_mixed_whitespace () =
  let tokens = tokenize "hello\tworld\nthere\r\nfriend" in
  check (list string) "mixed whitespace"
    [ "hello"; "world"; "there"; "friend" ]
    tokens

(* ───── Test Suite ───── *)

let tokenization_tests =
  [
    (* Words tokenization *)
    test_case "tokenize words simple" `Quick test_tokenize_words_simple;
    test_case "tokenize words punctuation" `Quick
      test_tokenize_words_punctuation;
    test_case "tokenize words numbers" `Quick test_tokenize_words_numbers;
    test_case "tokenize words empty" `Quick test_tokenize_words_empty;
    test_case "tokenize words whitespace only" `Quick
      test_tokenize_words_whitespace_only;
    test_case "tokenize words special chars" `Quick
      test_tokenize_words_special_chars;
    (* Character tokenization *)
    test_case "tokenize chars ASCII" `Quick test_tokenize_chars_ascii;
    test_case "tokenize chars unicode" `Quick test_tokenize_chars_unicode;
    test_case "tokenize chars empty" `Quick test_tokenize_chars_empty;
    (* Regex tokenization *)
    test_case "tokenize regex words" `Quick test_tokenize_regex_words;
    test_case "tokenize regex custom" `Quick test_tokenize_regex_custom;
    test_case "tokenize regex no match" `Quick test_tokenize_regex_no_match;
    (* Edge cases *)
    test_case "tokenize long text" `Quick test_tokenize_long_text;
    test_case "tokenize repeated punctuation" `Quick
      test_tokenize_repeated_punctuation;
    test_case "tokenize mixed whitespace" `Quick test_tokenize_mixed_whitespace;
  ]

let () =
  Alcotest.run "nx-text tokenization" [ ("tokenization", tokenization_tests) ]
