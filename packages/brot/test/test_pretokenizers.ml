(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Pre = Brot.Pre_tokenizer

let check_tokenization name input expected =
  equal ~msg:name (list (pair string (pair int int))) expected input

let check_strings name input expected =
  equal ~msg:name (list string) expected (List.map fst input)

let behavior_name = function
  | `Isolated -> "Isolated"
  | `Removed -> "Removed"
  | `Merged_with_previous -> "MergedWithPrevious"
  | `Merged_with_next -> "MergedWithNext"
  | `Contiguous -> "Contiguous"

let test_byte_level_basic () =
  let tokenizer = Pre.byte_level ~add_prefix_space:false ~use_regex:true () in

  (* Test basic tokenization *)
  let test_case text expected_pieces expected_offsets =
    let result = Pre.pre_tokenize tokenizer text in
    let offsets = List.map snd result in
    check_strings
      (Printf.sprintf "ByteLevel pieces for %S" text)
      result expected_pieces;
    equal
      ~msg:(Printf.sprintf "ByteLevel offsets for %S" text)
      (list (pair int int))
      expected_offsets offsets
  in

  (* Basic words *)
  test_case "Hello" [ "Hello" ] [ (0, 5) ];
  test_case "hello" [ "hello" ] [ (0, 5) ];
  test_case "HELLO" [ "HELLO" ] [ (0, 5) ];

  (* Words with spaces - space becomes Ġ (0xC4 0xA0) *)
  test_case "Hello world" [ "Hello"; "\196\160world" ] [ (0, 5); (5, 11) ];
  test_case "Hello  world"
    [ "Hello"; "\196\160"; "\196\160world" ]
    [ (0, 5); (5, 6); (6, 12) ];

  (* Leading/trailing spaces *)
  test_case " hello" [ "\196\160hello" ] [ (0, 6) ];
  test_case "hello " [ "hello"; "\196\160" ] [ (0, 5); (5, 6) ];
  (* Note: Python produces ['Ġ', 'Ġhello', 'ĠĠ'] for " hello " *)
  test_case "  hello  "
    [ "\196\160"; "\196\160hello"; "\196\160\196\160" ]
    [ (0, 1); (1, 7); (7, 9) ];

  (* Contractions - should be kept as separate pieces *)
  test_case "'s" [ "'s" ] [ (0, 2) ];
  test_case "'t" [ "'t" ] [ (0, 2) ];
  test_case "'re" [ "'re" ] [ (0, 3) ];
  test_case "'ve" [ "'ve" ] [ (0, 3) ];
  test_case "'m" [ "'m" ] [ (0, 2) ];
  test_case "'ll" [ "'ll" ] [ (0, 3) ];
  test_case "'d" [ "'d" ] [ (0, 2) ];

  (* Words with contractions *)
  test_case "don't" [ "don"; "'t" ] [ (0, 3); (3, 5) ];
  test_case "it's" [ "it"; "'s" ] [ (0, 2); (2, 4) ];
  test_case "we're" [ "we"; "'re" ] [ (0, 2); (2, 5) ];
  test_case "I'll" [ "I"; "'ll" ] [ (0, 1); (1, 4) ];
  test_case "OpenAI's" [ "OpenAI"; "'s" ] [ (0, 6); (6, 8) ]

(* The splits as substrings of the input, which is what the GPT-2 pattern
   ['s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+]
   decides; their byte-level re-encoding is checked separately. Every case here
   is the output of HuggingFace [tokenizers]. *)
let test_byte_level_pattern () =
  let tokenizer = Pre.byte_level ~add_prefix_space:false ~use_regex:true () in
  let test_case text expected =
    let spans =
      Pre.pre_tokenize tokenizer text
      |> List.map (fun (_, (start, stop)) ->
          String.sub text start (stop - start))
    in
    equal
      ~msg:(Printf.sprintf "GPT-2 pattern on %S" text)
      (list string) expected spans
  in

  (* A whitespace run gives up its last character to the next split, unless it
     ends the text. *)
  test_case "\n\nNot" [ "\n"; "\n"; "Not" ];
  test_case "  ." [ " "; " ." ];
  test_case "x  y" [ "x"; " "; " y" ];
  test_case "a \n b" [ "a"; " \n"; " b" ];
  test_case "a  " [ "a"; "  " ];
  test_case "  \n  x" [ "  \n "; " x" ];
  test_case " \n " [ " \n " ];
  test_case "a\r\nb" [ "a"; "\r"; "\n"; "b" ];

  (* The optional leading character of the letter, number and other runs is a
     space, not any whitespace. *)
  test_case "\tabc" [ "\t"; "abc" ];
  test_case "\t\tabc" [ "\t"; "\t"; "abc" ];
  test_case "\011\012x" [ "\011"; "\012"; "x" ];

  (* Whitespace is the Unicode White_Space property: no-break space, em space,
     ideographic space, next line, line separator. *)
  test_case "a\xc2\xa0b" [ "a"; "\xc2\xa0"; "b" ];
  test_case "a \xc2\xa0b" [ "a"; " "; "\xc2\xa0"; "b" ];
  test_case "a\xc2\xa0 b" [ "a"; "\xc2\xa0"; " b" ];
  test_case "a\xe2\x80\x83b" [ "a"; "\xe2\x80\x83"; "b" ];
  test_case "a\xe3\x80\x80b" [ "a"; "\xe3\x80\x80"; "b" ];
  test_case "a\xc2\x85b" [ "a"; "\xc2\x85"; "b" ];
  test_case "a\xe2\x80\xa8b" [ "a"; "\xe2\x80\xa8"; "b" ];

  (* Letters are the Letter category, so a combining mark starts a run of its
     own; zero width space is not whitespace. *)
  test_case "e\xcc\x81x" [ "e"; "\xcc\x81"; "x" ];
  test_case "a\xe0\xbd\xb1b" [ "a"; "\xe0\xbd\xb1"; "b" ];
  test_case "a\xe2\x80\x8bb" [ "a"; "\xe2\x80\x8b"; "b" ];

  (* Numbers are the Number category, letter and other numbers included. *)
  test_case "1\xc2\xbd2" [ "1\xc2\xbd2" ];
  test_case "\xe2\x85\xa0\xe2\x85\xa1" [ "\xe2\x85\xa0\xe2\x85\xa1" ];

  (* Contractions are lower case only. *)
  test_case "'ll'LL" [ "'ll"; "'"; "LL" ];
  test_case " 's" [ " '"; "s" ]

let test_byte_level_prefix_space () =
  (* Test with add_prefix_space=true *)
  let tokenizer = Pre.byte_level ~add_prefix_space:true ~use_regex:true () in

  let test_case text expected_pieces =
    let result = Pre.pre_tokenize tokenizer text in
    check_strings
      (Printf.sprintf "ByteLevel with prefix for %S" text)
      result expected_pieces
  in

  (* Should add space prefix when text doesn't start with space *)
  test_case "hello" [ "\196\160hello" ];
  test_case "Hello world" [ "\196\160Hello"; "\196\160world" ];

  (* Should NOT add extra space when text already starts with space *)
  test_case " hello" [ "\196\160hello" ];
  test_case "  hello" [ "\196\160"; "\196\160hello" ];

  (* Only a space is a prefix already there: other whitespace still gets one *)
  test_case "\nx" [ "\196\160"; "\196\138"; "x" ];
  test_case "\tx" [ "\196\160"; "\196\137"; "x" ]

let test_byte_level_special_chars () =
  let tokenizer = Pre.byte_level ~add_prefix_space:false ~use_regex:true () in

  let test_case text desc =
    let result = Pre.pre_tokenize tokenizer text in
    let pieces = List.map fst result in
    (* Just verify it doesn't crash and produces something *)
    equal
      ~msg:(Printf.sprintf "ByteLevel handles %s" desc)
      bool true
      (List.length pieces > 0)
  in

  (* Punctuation *)
  test_case "." "period";
  test_case "!" "exclamation";
  test_case "?" "question";
  test_case "," "comma";
  test_case ";" "semicolon";
  test_case ":" "colon";

  (* Special characters *)
  test_case "@" "at sign";
  test_case "#" "hash";
  test_case "$" "dollar";
  test_case "%" "percent";
  test_case "^" "caret";
  test_case "&" "ampersand";
  test_case "*" "asterisk";

  (* Brackets and quotes *)
  test_case "()" "parentheses";
  test_case "[]" "brackets";
  test_case "{}" "braces";
  test_case "\"\"" "quotes";
  test_case "''" "single quotes";

  (* Numbers *)
  test_case "123" "numbers";
  test_case "3.14" "decimal";
  test_case "1,000" "number with comma";

  (* Mixed *)
  test_case "Hello, world!" "punctuated sentence";
  test_case "@user #hashtag" "social media";
  test_case "test@example.com" "email";
  test_case "https://example.com" "URL";
  test_case "function()" "function call";
  test_case "a+b=c" "math expression"

let test_byte_level_unicode () =
  let tokenizer = Pre.byte_level ~add_prefix_space:false ~use_regex:true () in

  let test_case text desc =
    let result = Pre.pre_tokenize tokenizer text in
    let pieces = List.map fst result in
    (* Byte-level encoding should handle any Unicode by encoding bytes *)
    equal
      ~msg:(Printf.sprintf "ByteLevel handles %s" desc)
      bool true
      (List.length pieces > 0);
    (* Check that we can reconstruct something (even if not identical due to
       encoding) *)
    let concatenated = String.concat "" pieces in
    equal
      ~msg:(Printf.sprintf "ByteLevel produces non-empty output for %s" desc)
      bool true
      (String.length concatenated > 0)
  in

  (* Common accented characters *)
  test_case "café" "accented e";
  test_case "naïve" "diaeresis";
  test_case "résumé" "French accents";

  (* Other languages *)
  test_case "你好" "Chinese";
  test_case "こんにちは" "Japanese";
  test_case "안녕하세요" "Korean";
  test_case "Привет" "Russian";
  test_case "مرحبا" "Arabic";

  (* Emojis *)
  test_case "😀" "emoji";
  test_case "👍🏻" "emoji with skin tone";
  test_case "Hello 👋 World" "text with emoji"

let test_byte_level_edge_cases () =
  let tokenizer = Pre.byte_level ~add_prefix_space:false ~use_regex:true () in

  (* Empty string *)
  let result = Pre.pre_tokenize tokenizer "" in
  equal ~msg:"Empty string" (list string) [] (List.map fst result);

  (* Single character *)
  let result = Pre.pre_tokenize tokenizer "a" in
  check_strings "Single char" result [ "a" ];

  (* Only spaces - Python produces ['ĠĠĠ'] all together *)
  let result = Pre.pre_tokenize tokenizer "   " in
  check_strings "Only spaces" result [ "\196\160\196\160\196\160" ];

  (* Only punctuation - Python keeps '...' together *)
  let result = Pre.pre_tokenize tokenizer "..." in
  check_strings "Only punctuation" result [ "..." ];

  (* Very long word *)
  let long_word = String.make 100 'a' in
  let result = Pre.pre_tokenize tokenizer long_word in
  equal ~msg:"Long word produces single token" int 1 (List.length result);

  (* Mixed whitespace *)
  let result = Pre.pre_tokenize tokenizer "hello\tworld\nfoo\rbar" in
  equal ~msg:"Handles tabs and newlines" bool true (List.length result > 0)

let test_bert_pretokenizer () =
  let test_case text expected =
    let result = Pre.pre_tokenize (Pre.bert ()) text in
    check_tokenization
      (Printf.sprintf "BERT tokenization of %S" text)
      result expected
  in

  (* Basic tokenization *)
  test_case "Hello world" [ ("Hello", (0, 5)); ("world", (6, 11)) ];
  test_case "Hello, world!"
    [ ("Hello", (0, 5)); (",", (5, 6)); ("world", (7, 12)); ("!", (12, 13)) ];

  (* Punctuation handling *)
  test_case "test." [ ("test", (0, 4)); (".", (4, 5)) ];
  test_case "a-b" [ ("a", (0, 1)); ("-", (1, 2)); ("b", (2, 3)) ];
  test_case "it's" [ ("it", (0, 2)); ("'", (2, 3)); ("s", (3, 4)) ];

  (* Multiple spaces *)
  test_case "hello  world" [ ("hello", (0, 5)); ("world", (7, 12)) ];

  (* Unicode *)
  test_case "café" [ ("café", (0, 5)) ];

  (* Note: e is 2 bytes in UTF-8 *)

  (* Empty and whitespace *)
  test_case "" [];
  test_case "   " []

(* The punctuation class of the BERT and Punctuation pre-tokenizers: every
   printable ASCII character that is neither a letter nor a digit, plus the
   Unicode punctuation categories. Symbols are not punctuation. Every case is
   the output of HuggingFace [BertPreTokenizer] and [Punctuation]. *)
let ascii_punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

let test_punctuation_class () =
  let bert_pieces text = List.map fst (Pre.pre_tokenize (Pre.bert ()) text) in
  let punct_pieces text =
    List.map fst
      (Pre.pre_tokenize (Pre.punctuation ~behavior:`Isolated ()) text)
  in
  String.iter
    (fun c ->
      let c = String.make 1 c in
      let text = "a" ^ c ^ "b" in
      equal
        ~msg:(Printf.sprintf "bert splits %S" text)
        (list string) [ "a"; c; "b" ] (bert_pieces text);
      equal
        ~msg:(Printf.sprintf "punctuation splits %S" text)
        (list string) [ "a"; c; "b" ] (punct_pieces text))
    ascii_punctuation;

  (* A run of punctuation becomes one piece per character. *)
  equal ~msg:"bert on \"==\"" (list string) [ "="; "=" ] (bert_pieces "==");
  equal ~msg:"bert on \"a+b=c\"" (list string)
    [ "a"; "+"; "b"; "="; "c" ]
    (bert_pieces "a+b=c");
  equal ~msg:"bert on \"<|endoftext|>\"" (list string)
    [ "<"; "|"; "endoftext"; "|"; ">" ]
    (bert_pieces "<|endoftext|>");
  equal ~msg:"bert on \"#!/bin/sh\"" (list string)
    [ "#"; "!"; "/"; "bin"; "/"; "sh" ]
    (bert_pieces "#!/bin/sh");

  (* Unicode punctuation splits; Unicode symbols do not. *)
  let splits text expected =
    equal
      ~msg:(Printf.sprintf "bert on %S" text)
      (list string) expected (bert_pieces text)
  in
  splits "a\xC2\xABb" [ "a"; "\xC2\xAB"; "b" ];
  splits "a\xE2\x80\x90b" [ "a"; "\xE2\x80\x90"; "b" ];
  splits "a\xE2\x80\xBEb" [ "a"; "\xE2\x80\xBE"; "b" ];
  splits "a\xE2\x80\xBFb" [ "a"; "\xE2\x80\xBF"; "b" ];
  splits "a\xC2\xA1b" [ "a"; "\xC2\xA1"; "b" ];
  splits "a\xC2\xB1b" [ "a\xC2\xB1b" ];
  splits "a\xC2\xA2b" [ "a\xC2\xA2b" ];
  splits "a\xC2\xA9b" [ "a\xC2\xA9b" ];
  splits "a\xC3\xB7b" [ "a\xC3\xB7b" ];
  splits "a\xE2\x86\x92b" [ "a\xE2\x86\x92b" ]

let test_whitespace_pretokenizer () =
  let test_case text expected =
    let result = Pre.pre_tokenize (Pre.whitespace ()) text in
    check_tokenization
      (Printf.sprintf "Whitespace tokenization of %S" text)
      result expected
  in

  (* Pattern is \w+|[^\w\s]+ *)
  test_case "Hello world" [ ("Hello", (0, 5)); ("world", (6, 11)) ];
  test_case "Hello, world!"
    [ ("Hello", (0, 5)); (",", (5, 6)); ("world", (7, 12)); ("!", (12, 13)) ];
  test_case "test_var" [ ("test_var", (0, 8)) ];
  (* underscore is part of \w *)
  test_case "123abc" [ ("123abc", (0, 6)) ];
  (* numbers are part of \w *)
  test_case "a+b=c"
    [
      ("a", (0, 1)); ("+", (1, 2)); ("b", (2, 3)); ("=", (3, 4)); ("c", (4, 5));
    ];

  (* \w is alphabetic, marks, decimal digits and connector punctuation: a
     combining mark, a letter number and an undertie join the word, an other
     number and a circled digit do not, and a no-break space splits it. *)
  let check_words text expected =
    check_strings
      (Printf.sprintf "Whitespace words of %S" text)
      (Pre.pre_tokenize (Pre.whitespace ()) text)
      expected
  in
  check_words "a\xe0\xbd\xb1b" [ "a\xe0\xbd\xb1b" ];
  check_words "a\xe2\x85\xa0b" [ "a\xe2\x85\xa0b" ];
  check_words "a\xe2\x80\xbfb" [ "a\xe2\x80\xbfb" ];
  check_words "a\xc2\xbdb" [ "a"; "\xc2\xbd"; "b" ];
  check_words "a\xe2\x91\xa0b" [ "a"; "\xe2\x91\xa0"; "b" ];
  check_words "a\xc2\xa0b" [ "a"; "b" ]

let test_whitespace_split () =
  let test_case text expected =
    let result = Pre.pre_tokenize (Pre.whitespace_split ()) text in
    check_tokenization
      (Printf.sprintf "WhitespaceSplit of %S" text)
      result expected
  in

  (* Simple split on whitespace *)
  test_case "Hello world" [ ("Hello", (0, 5)); ("world", (6, 11)) ];
  test_case "  Hello  world  " [ ("Hello", (2, 7)); ("world", (9, 14)) ];
  test_case "one\ttwo\nthree"
    [ ("one", (0, 3)); ("two", (4, 7)); ("three", (8, 13)) ];
  test_case "" [];
  test_case "   " []

let test_punctuation_pretokenizer () =
  (* Test different behaviors *)
  let test_isolated text expected =
    let tokenizer = Pre.punctuation ~behavior:`Isolated () in
    let result = Pre.pre_tokenize tokenizer text in
    check_tokenization
      (Printf.sprintf "Punctuation Isolated %S" text)
      result expected
  in

  let test_removed text expected =
    let tokenizer = Pre.punctuation ~behavior:`Removed () in
    let result = Pre.pre_tokenize tokenizer text in
    check_tokenization
      (Printf.sprintf "Punctuation Removed %S" text)
      result expected
  in

  (* Isolated behavior *)
  test_isolated "Hello, world!"
    [ ("Hello", (0, 5)); (",", (5, 6)); (" world", (6, 12)); ("!", (12, 13)) ];

  (* Removed behavior *)
  test_removed "Hello, world!" [ ("Hello", (0, 5)); (" world", (6, 12)) ];

  (* Multiple punctuation *)
  test_isolated "test...end"
    [
      ("test", (0, 4));
      (".", (4, 5));
      (".", (5, 6));
      (".", (6, 7));
      ("end", (7, 10));
    ];

  (* Unicode punctuation *)
  test_isolated "test—end" [ ("test", (0, 4)); ("—", (4, 7)); ("end", (7, 10)) ]
(* em dash is 3 bytes *)

(* The pieces run from one delimiter to the next: a punctuation character for
   Punctuation, an occurrence of the pattern for Split, which [invert] swaps for
   the text between the occurrences. The two merging behaviors are one rule read
   from either side — a piece takes in what follows it when the roles differ —
   so a delimiter that follows another one stands alone. Every expectation below
   is that of HuggingFace tokenizers 0.23.1. *)

let test_punctuation_behaviors () =
  let case behavior text expected =
    check_tokenization
      (Printf.sprintf "Punctuation %s on %S" (behavior_name behavior) text)
      (Pre.pre_tokenize (Pre.punctuation ~behavior ()) text)
      expected
  in
  case `Isolated "a,b,c"
    [
      ("a", (0, 1)); (",", (1, 2)); ("b", (2, 3)); (",", (3, 4)); ("c", (4, 5));
    ];
  case `Isolated "a,,b"
    [ ("a", (0, 1)); (",", (1, 2)); (",", (2, 3)); ("b", (3, 4)) ];
  case `Isolated "«x»" [ ("«", (0, 2)); ("x", (2, 3)); ("»", (3, 5)) ];
  case `Removed "a,b,c" [ ("a", (0, 1)); ("b", (2, 3)); ("c", (4, 5)) ];
  case `Removed "a,,b" [ ("a", (0, 1)); ("b", (3, 4)) ];
  case `Removed "«x»" [ ("x", (2, 3)) ];
  case `Merged_with_previous "a,b,c"
    [ ("a,", (0, 2)); ("b,", (2, 4)); ("c", (4, 5)) ];
  case `Merged_with_previous "a,,b"
    [ ("a,", (0, 2)); (",", (2, 3)); ("b", (3, 4)) ];
  case `Merged_with_previous "«x»" [ ("«", (0, 2)); ("x»", (2, 5)) ];
  case `Merged_with_next "a,b,c"
    [ ("a", (0, 1)); (",b", (1, 3)); (",c", (3, 5)) ];
  case `Merged_with_next "a,,b" [ ("a", (0, 1)); (",", (1, 2)); (",b", (2, 4)) ];
  case `Merged_with_next "«x»" [ ("«x", (0, 3)); ("»", (3, 5)) ];
  case `Contiguous "a,b,c"
    [
      ("a", (0, 1)); (",", (1, 2)); ("b", (2, 3)); (",", (3, 4)); ("c", (4, 5));
    ];
  case `Contiguous "a,,b" [ ("a", (0, 1)); (",,", (1, 3)); ("b", (3, 4)) ];
  case `Contiguous "«x»" [ ("«", (0, 2)); ("x", (2, 3)); ("»", (3, 5)) ]

let test_digits_pretokenizer () =
  let test_individual text expected =
    let tokenizer = Pre.digits ~individual_digits:true () in
    let result = Pre.pre_tokenize tokenizer text in
    check_tokenization
      (Printf.sprintf "Digits individual %S" text)
      result expected
  in

  let test_grouped text expected =
    let tokenizer = Pre.digits ~individual_digits:false () in
    let result = Pre.pre_tokenize tokenizer text in
    check_tokenization (Printf.sprintf "Digits grouped %S" text) result expected
  in

  (* Individual digits *)
  test_individual "123" [ ("1", (0, 1)); ("2", (1, 2)); ("3", (2, 3)) ];
  test_individual "a1b2"
    [ ("a", (0, 1)); ("1", (1, 2)); ("b", (2, 3)); ("2", (3, 4)) ];

  (* Grouped digits *)
  test_grouped "123" [ ("123", (0, 3)) ];
  test_grouped "a123b456"
    [ ("a", (0, 1)); ("123", (1, 4)); ("b", (4, 5)); ("456", (5, 8)) ];
  test_grouped "3.14" [ ("3", (0, 1)); (".", (1, 2)); ("14", (2, 4)) ]

let split_case ?(pattern = ",") behavior ~invert text expected =
  check_tokenization
    (Printf.sprintf "Split %S %s ~invert:%b on %S" pattern
       (behavior_name behavior) invert text)
    (Pre.pre_tokenize (Pre.split ~pattern ~behavior ~invert ()) text)
    expected

let test_split_behaviors () =
  let case = split_case in
  case `Isolated ~invert:false "a,b,c"
    [
      ("a", (0, 1)); (",", (1, 2)); ("b", (2, 3)); (",", (3, 4)); ("c", (4, 5));
    ];
  case `Isolated ~invert:false "a,,b"
    [ ("a", (0, 1)); (",", (1, 2)); (",", (2, 3)); ("b", (3, 4)) ];
  case `Isolated ~invert:false ",," [ (",", (0, 1)); (",", (1, 2)) ];
  case `Isolated ~invert:true "a,b,c"
    [
      ("a", (0, 1)); (",", (1, 2)); ("b", (2, 3)); (",", (3, 4)); ("c", (4, 5));
    ];
  case `Isolated ~invert:true "a,,b"
    [ ("a", (0, 1)); (",", (1, 2)); (",", (2, 3)); ("b", (3, 4)) ];
  case `Isolated ~invert:true ",," [ (",", (0, 1)); (",", (1, 2)) ];
  case `Removed ~invert:false "a,b,c"
    [ ("a", (0, 1)); ("b", (2, 3)); ("c", (4, 5)) ];
  case `Removed ~invert:false "a,,b" [ ("a", (0, 1)); ("b", (3, 4)) ];
  case `Removed ~invert:false ",," [];
  case `Removed ~invert:true "a,b,c" [ (",", (1, 2)); (",", (3, 4)) ];
  case `Removed ~invert:true "a,,b" [ (",", (1, 2)); (",", (2, 3)) ];
  case `Removed ~invert:true ",," [ (",", (0, 1)); (",", (1, 2)) ];
  case `Merged_with_previous ~invert:false "a,b,c"
    [ ("a,", (0, 2)); ("b,", (2, 4)); ("c", (4, 5)) ];
  case `Merged_with_previous ~invert:false "a,,b"
    [ ("a,", (0, 2)); (",", (2, 3)); ("b", (3, 4)) ];
  case `Merged_with_previous ~invert:false ",," [ (",", (0, 1)); (",", (1, 2)) ];
  case `Merged_with_previous ~invert:true "a,b,c"
    [ ("a", (0, 1)); (",b", (1, 3)); (",c", (3, 5)) ];
  case `Merged_with_previous ~invert:true "a,,b"
    [ ("a", (0, 1)); (",", (1, 2)); (",b", (2, 4)) ];
  case `Merged_with_previous ~invert:true ",," [ (",", (0, 1)); (",", (1, 2)) ];
  case `Merged_with_next ~invert:false "a,b,c"
    [ ("a", (0, 1)); (",b", (1, 3)); (",c", (3, 5)) ];
  case `Merged_with_next ~invert:false "a,,b"
    [ ("a", (0, 1)); (",", (1, 2)); (",b", (2, 4)) ];
  case `Merged_with_next ~invert:false ",," [ (",", (0, 1)); (",", (1, 2)) ];
  case `Merged_with_next ~invert:true "a,b,c"
    [ ("a,", (0, 2)); ("b,", (2, 4)); ("c", (4, 5)) ];
  case `Merged_with_next ~invert:true "a,,b"
    [ ("a,", (0, 2)); (",", (2, 3)); ("b", (3, 4)) ];
  case `Merged_with_next ~invert:true ",," [ (",", (0, 1)); (",", (1, 2)) ];
  case `Contiguous ~invert:false "a,b,c"
    [
      ("a", (0, 1)); (",", (1, 2)); ("b", (2, 3)); (",", (3, 4)); ("c", (4, 5));
    ];
  case `Contiguous ~invert:false "a,,b"
    [ ("a", (0, 1)); (",,", (1, 3)); ("b", (3, 4)) ];
  case `Contiguous ~invert:false ",," [ (",,", (0, 2)) ];
  case `Contiguous ~invert:true "a,b,c"
    [
      ("a", (0, 1)); (",", (1, 2)); ("b", (2, 3)); (",", (3, 4)); ("c", (4, 5));
    ];
  case `Contiguous ~invert:true "a,,b"
    [ ("a", (0, 1)); (",,", (1, 3)); ("b", (3, 4)) ];
  case `Contiguous ~invert:true ",," [ (",,", (0, 2)) ]

(* A pattern of several characters, one that is several bytes, and the empty
   pattern, which matches between every pair of characters. *)
let test_split_patterns () =
  let case = split_case in
  case ~pattern:"::" `Isolated ~invert:false "a::b::c"
    [
      ("a", (0, 1)); ("::", (1, 3)); ("b", (3, 4)); ("::", (4, 6)); ("c", (6, 7));
    ];
  case ~pattern:"ab" `Isolated ~invert:false "xababy"
    [ ("x", (0, 1)); ("ab", (1, 3)); ("ab", (3, 5)); ("y", (5, 6)) ];
  case ~pattern:"ab" `Isolated ~invert:true "xababy"
    [ ("x", (0, 1)); ("ab", (1, 3)); ("ab", (3, 5)); ("y", (5, 6)) ];
  case ~pattern:"ab" `Removed ~invert:false "xababy"
    [ ("x", (0, 1)); ("y", (5, 6)) ];
  case ~pattern:"ab" `Removed ~invert:true "xababy"
    [ ("ab", (1, 3)); ("ab", (3, 5)) ];
  case ~pattern:"ab" `Merged_with_previous ~invert:false "xababy"
    [ ("xab", (0, 3)); ("ab", (3, 5)); ("y", (5, 6)) ];
  case ~pattern:"ab" `Merged_with_previous ~invert:true "xababy"
    [ ("x", (0, 1)); ("ab", (1, 3)); ("aby", (3, 6)) ];
  case ~pattern:"ab" `Merged_with_next ~invert:false "xababy"
    [ ("x", (0, 1)); ("ab", (1, 3)); ("aby", (3, 6)) ];
  case ~pattern:"ab" `Merged_with_next ~invert:true "xababy"
    [ ("xab", (0, 3)); ("ab", (3, 5)); ("y", (5, 6)) ];
  case ~pattern:"ab" `Contiguous ~invert:false "xababy"
    [ ("x", (0, 1)); ("abab", (1, 5)); ("y", (5, 6)) ];
  case ~pattern:"ab" `Contiguous ~invert:true "xababy"
    [ ("x", (0, 1)); ("abab", (1, 5)); ("y", (5, 6)) ];

  (* Inverted, a text without an occurrence is one delimiter, not one per
     byte. *)
  case `Isolated ~invert:true "  " [ ("  ", (0, 2)) ];
  case `Removed ~invert:true "  " [];
  case ~pattern:"、" `Isolated ~invert:false "a、、b"
    [ ("a", (0, 1)); ("、", (1, 4)); ("、", (4, 7)); ("b", (7, 8)) ];
  case ~pattern:"、" `Isolated ~invert:true "a、、b"
    [ ("a", (0, 1)); ("、", (1, 4)); ("、", (4, 7)); ("b", (7, 8)) ];
  case ~pattern:"、" `Removed ~invert:true "a、、b"
    [ ("、", (1, 4)); ("、", (4, 7)) ];
  case ~pattern:"、" `Contiguous ~invert:false "a、、b"
    [ ("a", (0, 1)); ("、、", (1, 7)); ("b", (7, 8)) ];

  (* The empty pattern makes every character a piece, and inverted its empty
     occurrences are the text, so [`Removed] leaves nothing. *)
  case ~pattern:"" `Isolated ~invert:false "aé" [ ("a", (0, 1)); ("é", (1, 3)) ];
  case ~pattern:"" `Removed ~invert:false "aé" [ ("a", (0, 1)); ("é", (1, 3)) ];
  case ~pattern:"" `Contiguous ~invert:true "aé"
    [ ("a", (0, 1)); ("é", (1, 3)) ];
  case ~pattern:"" `Removed ~invert:true "aé" []

let test_char_delimiter_split () =
  let test_case delim text expected =
    let result = Pre.pre_tokenize (Pre.char_delimiter delim) text in
    check_tokenization
      (Printf.sprintf "CharDelimiterSplit delim=%S text=%S" delim text)
      result expected
  in

  test_case "," "a,b,c" [ ("a", (0, 1)); ("b", (2, 3)); ("c", (4, 5)) ];
  test_case " " "hello world" [ ("hello", (0, 5)); ("world", (6, 11)) ];
  test_case "|" "one|two|three"
    [ ("one", (0, 3)); ("two", (4, 7)); ("three", (8, 13)) ];
  test_case "\u{2581}" "\u{2581}a\u{2581}b" [ ("a", (3, 4)); ("b", (7, 8)) ];
  test_case "," "" [];
  test_case "," "," []

let test_sequence_pretokenizer () =
  (* Combine whitespace split then punctuation isolation *)
  let tokenizers =
    [ Pre.whitespace_split (); Pre.punctuation ~behavior:`Isolated () ]
  in
  let tokenizer = Pre.sequence tokenizers in

  let test_case text expected =
    let result = Pre.pre_tokenize tokenizer text in
    check_tokenization (Printf.sprintf "Sequence %S" text) result expected
  in

  (* First splits on whitespace, then isolates punctuation in each piece *)
  test_case "Hello, world!"
    [ ("Hello", (0, 5)); (",", (5, 6)); ("world", (7, 12)); ("!", (12, 13)) ];

  (* Multiple words and punctuation *)
  test_case "test. another, example!"
    [
      ("test", (0, 4));
      (".", (4, 5));
      ("another", (6, 13));
      (",", (13, 14));
      ("example", (15, 22));
      ("!", (22, 23));
    ]

let test_fixed_length () =
  let test_case length text expected =
    let result = Pre.pre_tokenize (Pre.fixed_length length) text in
    check_tokenization
      (Printf.sprintf "FixedLength %d %S" length text)
      result expected
  in

  test_case 3 "abcdefghi" [ ("abc", (0, 3)); ("def", (3, 6)); ("ghi", (6, 9)) ];
  test_case 2 "abcde" [ ("ab", (0, 2)); ("cd", (2, 4)); ("e", (4, 5)) ];
  test_case 5 "hello" [ ("hello", (0, 5)) ];
  test_case 0 "test" [];
  test_case 3 "" [];

  (* With UTF-8 - counts characters not bytes *)
  test_case 2 "café" [ ("ca", (0, 2)); ("fé", (2, 5)) ]
(* e is 2 bytes *)

(* Expectations checked against tokenizers.pre_tokenizers.UnicodeScripts(), with
   its character offsets converted to byte offsets. *)
let test_unicode_scripts () =
  let test_case text expected =
    let result = Pre.pre_tokenize (Pre.unicode_scripts ()) text in
    check_tokenization (Printf.sprintf "UnicodeScripts %S" text) result expected
  in

  test_case "Hello world" [ ("Hello world", (0, 11)) ];
  test_case "abc!def" [ ("abc", (0, 3)); ("!", (3, 4)); ("def", (4, 7)) ];
  test_case "a b" [ ("a b", (0, 3)) ];
  test_case "" [];

  (* A leading run of spaces opens no piece and is dropped. *)
  test_case "  !  " [ ("!  ", (2, 5)) ];
  test_case " abc" [ ("abc", (1, 4)) ];
  test_case "  \u{6587}" [ ("\u{6587}", (2, 5)) ];
  test_case "   " [];
  test_case "abc  " [ ("abc  ", (0, 5)) ];

  (* Only U+0020 joins the surrounding run. Every other whitespace character
     keeps its own script: Common, except for the Ogham space mark. *)
  test_case "a\tb" [ ("a", (0, 1)); ("\t", (1, 2)); ("b", (2, 3)) ];
  test_case "a\rb" [ ("a", (0, 1)); ("\r", (1, 2)); ("b", (2, 3)) ];
  test_case "\n\nabc def" [ ("\n\n", (0, 2)); ("abc def", (2, 9)) ];
  test_case "  \n  abc" [ ("\n  ", (2, 5)); ("abc", (5, 8)) ];
  (* U+00A0 no-break space, U+3000 ideographic space. *)
  test_case "a\u{a0}\u{3000}b"
    [ ("a", (0, 1)); ("\u{a0}\u{3000}", (1, 6)); ("b", (6, 7)) ];
  (* U+2028 line separator, U+2029 paragraph separator, U+202F narrow no-break
     space, U+205F medium mathematical space. *)
  test_case "a\u{2028}b" [ ("a", (0, 1)); ("\u{2028}", (1, 4)); ("b", (4, 5)) ];
  test_case "a\u{2029}b" [ ("a", (0, 1)); ("\u{2029}", (1, 4)); ("b", (4, 5)) ];
  test_case "a\u{202f}b" [ ("a", (0, 1)); ("\u{202f}", (1, 4)); ("b", (4, 5)) ];
  test_case "a\u{205f}b" [ ("a", (0, 1)); ("\u{205f}", (1, 4)); ("b", (4, 5)) ];
  (* U+1680 ogham space mark, script Ogham. *)
  test_case "\u{1680}abc" [ ("\u{1680}", (0, 3)); ("abc", (3, 6)) ];

  (* A character of no known script also joins the surrounding run, and is
     dropped when it leads: U+E000 is private use. *)
  test_case "a\u{e000}b" [ ("a\u{e000}b", (0, 5)) ];
  test_case "\u{e000}abc" [ ("abc", (3, 6)) ];
  test_case "  \u{e000}  abc" [ ("abc", (7, 10)) ];

  (* Kana folds into Han; so does the prolonged sound mark U+30FC, which is
     Common. The iteration mark U+3005 is Han already. *)
  test_case "\u{65e5}\u{672c}\u{8a9e}\u{30c6}\u{30ad}\u{30c8}"
    [ ("\u{65e5}\u{672c}\u{8a9e}\u{30c6}\u{30ad}\u{30c8}", (0, 18)) ];
  test_case "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}\u{4e16}\u{754c}"
    [ ("\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}\u{4e16}\u{754c}", (0, 21)) ];
  test_case "\u{4e2d}\u{30fc}\u{3042}" [ ("\u{4e2d}\u{30fc}\u{3042}", (0, 9)) ];
  test_case "\u{4e2d}\u{3005}\u{4e2d} abc"
    [ ("\u{4e2d}\u{3005}\u{4e2d} ", (0, 10)); ("abc", (10, 13)) ];

  (* A script change splits; digits are Common. *)
  test_case "\u{4e2d}\u{6587} abc"
    [ ("\u{4e2d}\u{6587} ", (0, 7)); ("abc", (7, 10)) ];
  test_case "Hello\u{4e16}\u{754c}"
    [ ("Hello", (0, 5)); ("\u{4e16}\u{754c}", (5, 11)) ];
  test_case "a\u{ff11}\u{ff12}\u{ff13}"
    [ ("a", (0, 1)); ("\u{ff11}\u{ff12}\u{ff13}", (1, 10)) ];
  test_case " 123 \u{4e2d}\u{6587} abc "
    [ ("123 ", (1, 5)); ("\u{4e2d}\u{6587} ", (5, 12)); ("abc ", (12, 16)) ];

  (* One script, one piece, however many words. *)
  test_case "\u{395}\u{3bb}\u{3bb}\u{3b7}\u{3bd}\u{3b9}\u{3ba}\u{3ac} abc"
    [
      ("\u{395}\u{3bb}\u{3bb}\u{3b7}\u{3bd}\u{3b9}\u{3ba}\u{3ac} ", (0, 17));
      ("abc", (17, 20));
    ];
  test_case "\u{41f}\u{440}\u{438}\u{432}\u{435}\u{442} \u{43c}\u{438}\u{440}"
    [
      ( "\u{41f}\u{440}\u{438}\u{432}\u{435}\u{442} \u{43c}\u{438}\u{440}",
        (0, 19) );
    ];
  test_case "abc \u{41f}\u{440}\u{438}\u{432}\u{435}\u{442}"
    [
      ("abc ", (0, 4)); ("\u{41f}\u{440}\u{438}\u{432}\u{435}\u{442}", (4, 16));
    ];
  test_case
    "\u{645}\u{631}\u{62d}\u{628}\u{627} \
     \u{628}\u{627}\u{644}\u{639}\u{627}\u{644}\u{645}"
    [
      ( "\u{645}\u{631}\u{62d}\u{628}\u{627} \
         \u{628}\u{627}\u{644}\u{639}\u{627}\u{644}\u{645}",
        (0, 25) );
    ];
  test_case "\u{5e9}\u{5dc}\u{5d5}\u{5dd} abc"
    [ ("\u{5e9}\u{5dc}\u{5d5}\u{5dd} ", (0, 9)); ("abc", (9, 12)) ];
  test_case "\u{c548}\u{b155} abc"
    [ ("\u{c548}\u{b155} ", (0, 7)); ("abc", (7, 10)) ];
  test_case "\u{928}\u{92e}\u{938}\u{94d}\u{924}\u{947} abc"
    [
      ("\u{928}\u{92e}\u{938}\u{94d}\u{924}\u{947} ", (0, 19)); ("abc", (19, 22));
    ];

  (* Inherited is a script of its own, so a combining mark and a variation
     selector each split off what they attach to. *)
  test_case "e\u{301}x" [ ("e", (0, 1)); ("\u{301}", (1, 3)); ("x", (3, 4)) ];
  test_case "a\u{2764}\u{fe0f}b"
    [ ("a", (0, 1)); ("\u{2764}", (1, 4)); ("\u{fe0f}", (4, 7)); ("b", (7, 8)) ]

(* Every case is the output of HuggingFace [Metaspace]. *)
let test_metaspace_huggingface () =
  let pieces ?(replacement = "\u{2581}") ?(prepend_scheme = `Always)
      ?(split = true) text =
    List.map fst
      (Pre.pre_tokenize
         (Pre.metaspace ~replacement ~prepend_scheme ~split ())
         text)
  in
  let check text expected =
    equal
      ~msg:(Printf.sprintf "Metaspace %S" text)
      (list string) expected (pieces text)
  in
  (* The marker is prepended only when the marked text lacks one. *)
  check "a" [ "\u{2581}a" ];
  check " a" [ "\u{2581}a" ];
  check "\u{2581}a" [ "\u{2581}a" ];
  check "\u{2581}hello\u{2581}world" [ "\u{2581}hello"; "\u{2581}world" ];
  check "hello world" [ "\u{2581}hello"; "\u{2581}world" ];
  check "  a" [ "\u{2581}"; "\u{2581}a" ];
  check "\u{2581} a" [ "\u{2581}"; "\u{2581}a" ];
  check " " [ "\u{2581}" ];
  check "" [];
  equal ~msg:"Metaspace `Never" (list string) [ "a"; "\u{2581}b" ]
    (pieces ~prepend_scheme:`Never "a b");
  (* Without splitting, the piece is the marked text and the offsets are those
     of the text as given. *)
  check_tokenization "Metaspace ~split:false"
    (Pre.pre_tokenize
       (Pre.metaspace ~prepend_scheme:`Never ~split:false ())
       " a")
    [ ("\u{2581}a", (0, 2)) ];
  equal ~msg:"Metaspace ~split:false on empty"
    (list (pair string (pair int int)))
    []
    (Pre.pre_tokenize (Pre.metaspace ~split:false ()) "")

(* Both metaspaces walk: without splitting the walk is one span over the marked
   text, which is what places a model's tokens inside it. A sequence carries the
   rewrite only when the metaspace opens it. *)
let test_metaspace_plan () =
  let describe t =
    match Pre.plan t with
    | Pre.Pieces -> "pieces"
    | Pre.Walk { rewrite = Pre.Verbatim; splittable } ->
        Printf.sprintf "verbatim splittable=%b" splittable
    | Pre.Walk { rewrite = Pre.Prefix_space; splittable } ->
        Printf.sprintf "prefix space splittable=%b" splittable
    | Pre.Walk { rewrite = Pre.Space_marker { marker; prepend }; splittable } ->
        Printf.sprintf "marker=%S prepend=%b splittable=%b" marker prepend
          splittable
  in
  let check name t expected = equal ~msg:name string expected (describe t) in
  check "split" (Pre.metaspace ())
    "marker=\"\\226\\150\\129\" prepend=true splittable=false";
  check "no split"
    (Pre.metaspace ~split:false ())
    "marker=\"\\226\\150\\129\" prepend=true splittable=false";
  check "no split, no prepend"
    (Pre.metaspace ~replacement:"_" ~prepend_scheme:`Never ~split:false ())
    "marker=\"_\" prepend=false splittable=false";
  check "no split, then a verbatim walker"
    (Pre.sequence [ Pre.metaspace ~split:false (); Pre.punctuation () ])
    "marker=\"\\226\\150\\129\" prepend=true splittable=false";
  (* A rewrite applies to the whole text, so a metaspace that is not the first
     member has no walk to take part in. *)
  check "after another walker"
    (Pre.sequence [ Pre.whitespace_split (); Pre.metaspace ~split:false () ])
    "pieces"

(* Every expectation is the output of HuggingFace [Metaspace], whose offsets are
   those of the text a piece was made from and never of the marked text: a piece
   spans the union of what its characters came from, and a prepended marker came
   from nothing. *)
let test_metaspace_offsets () =
  let case t text expected =
    check_tokenization (Printf.sprintf "%S" text) (Pre.pre_tokenize t text)
      expected
  in
  let m = "\u{2581}" in
  let split = Pre.metaspace () in
  case split "Hello world" [ (m ^ "Hello", (0, 5)); (m ^ "world", (5, 11)) ];
  case split "trailing " [ (m ^ "trailing", (0, 8)); (m, (8, 9)) ];
  case split "  two  spaces"
    [ (m, (0, 1)); (m ^ "two", (1, 5)); (m, (5, 6)); (m ^ "spaces", (6, 13)) ];
  case split "" [];
  case split " " [ (m, (0, 1)) ];
  case split "\u{65e5}\u{672c} \u{8a9e}"
    [ (m ^ "\u{65e5}\u{672c}", (0, 6)); (m ^ "\u{8a9e}", (6, 10)) ];
  case split (m ^ "already") [ (m ^ "already", (0, 10)) ];
  case split "a\u{a0}b" [ (m ^ "a\u{a0}b", (0, 4)) ];
  let never = Pre.metaspace ~prepend_scheme:`Never () in
  case never "Hello world" [ ("Hello", (0, 5)); (m ^ "world", (5, 11)) ];
  case never "trailing " [ ("trailing", (0, 8)); (m, (8, 9)) ];
  case never "a\u{a0}b" [ ("a\u{a0}b", (0, 4)) ];
  let whole = Pre.metaspace ~split:false () in
  case whole "Hello world" [ (m ^ "Hello" ^ m ^ "world", (0, 11)) ];
  case whole "trailing " [ (m ^ "trailing" ^ m, (0, 9)) ];
  case whole "  two  spaces" [ (m ^ m ^ "two" ^ m ^ m ^ "spaces", (0, 13)) ];
  case whole "" [];
  case whole " " [ (m, (0, 1)) ];
  (* T5 carries this one. The whitespace split drops the run that separates two
     pieces, so a piece starts where its own text does and the trailing space is
     gone before the marker could stand for it. *)
  let t5 = Pre.sequence [ Pre.whitespace_split (); Pre.metaspace () ] in
  case t5 "Hello world" [ (m ^ "Hello", (0, 5)); (m ^ "world", (6, 11)) ];
  case t5 "trailing " [ (m ^ "trailing", (0, 8)) ];
  case t5 "  two  spaces" [ (m ^ "two", (2, 5)); (m ^ "spaces", (7, 13)) ];
  case t5 "" [];
  case t5 " " [];
  case t5 "\u{65e5}\u{672c} \u{8a9e}"
    [ (m ^ "\u{65e5}\u{672c}", (0, 6)); (m ^ "\u{8a9e}", (7, 10)) ];
  (* A marker cut into a piece of its own by a later member takes the span of
     what it opens, rather than none at all: the marker that replaced a space is
     at that space, and a prepended one at the first character. *)
  let marked_then_split replacement =
    Pre.sequence [ Pre.metaspace ~replacement (); Pre.punctuation () ]
  in
  let underscore = marked_then_split "_" in
  case underscore "a" [ ("_", (0, 1)); ("a", (0, 1)) ];
  case underscore "Hello world"
    [ ("_", (0, 1)); ("Hello", (0, 5)); ("_", (5, 6)); ("world", (6, 11)) ];
  case underscore "!!!"
    [ ("_", (0, 1)); ("!", (0, 1)); ("!", (1, 2)); ("!", (2, 3)) ];
  case underscore "_a_b"
    [ ("_", (0, 1)); ("a", (1, 2)); ("_", (2, 3)); ("b", (3, 4)) ];
  let block = marked_then_split m in
  case block "a" [ (m ^ "a", (0, 1)) ];
  case block "!!!" [ (m, (0, 1)); ("!", (0, 1)); ("!", (1, 2)); ("!", (2, 3)) ];
  case block "_a_b"
    [ (m, (0, 1)); ("_", (0, 1)); ("a", (1, 2)); ("_", (2, 3)); ("b", (3, 4)) ];
  (* A member that follows one whose pieces are not bytes of its input places
     nothing more finely than the whole of the piece it was handed: HuggingFace,
     which carries an alignment through the chain, has ["!"] at [(3, 4)]
     here. *)
  let punctuated =
    Pre.sequence
      [ Pre.whitespace_split (); Pre.metaspace (); Pre.punctuation () ]
  in
  case punctuated "a b!" [ (m ^ "a", (0, 1)); (m ^ "b", (2, 4)); ("!", (2, 4)) ]

let test_metaspace_basic () =
  let test_case ?(replacement = "_") text expected =
    let result =
      Pre.pre_tokenize
        (Pre.metaspace ~replacement ~prepend_scheme:`Always ~split:true ())
        text
    in
    check_strings (Printf.sprintf "Metaspace %S" text) result expected
  in

  test_case "Hello world" [ "_Hello"; "_world" ];
  test_case " starts with space" [ "_starts"; "_with"; "_space" ];
  test_case "" [];

  (* The default replacement is U+2581, which no single character could hold.
     The pieces are eight bytes each and the text they were made from five and
     six, which is what the offsets report. *)
  let default = Pre.pre_tokenize (Pre.metaspace ()) "Hello world" in
  check_strings "Metaspace default marker" default
    [ "\u{2581}Hello"; "\u{2581}world" ];
  equal ~msg:"Metaspace default offsets"
    (list (pair int int))
    [ (0, 5); (5, 11) ]
    (List.map snd default);
  test_case ~replacement:"\u{2581}" "a b" [ "\u{2581}a"; "\u{2581}b" ]

(* A corpus of classified code points, and the GPT-2 pattern expressed over
   their classes: at each position, the first alternative of
   ['s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+]
   that matches. The walker must agree with it. Every class below is the one
   HuggingFace gives that code point. *)

type gpt2_class = Space | White | Letter | Number | Other

(* Repeated entries are drawn more often: spaces, and the letters that make a
   contraction. *)
let alphabet =
  [|
    (0x0020, Space);
    (0x0020, Space);
    (0x0020, Space);
    (0x0009, White);
    (0x000A, White);
    (0x000D, White);
    (0x000B, White);
    (0x000C, White);
    (0x00A0, White) (* no-break space *);
    (0x2003, White) (* em space *);
    (0x3000, White) (* ideographic space *);
    (0x0085, White) (* next line *);
    (0x2028, White) (* line separator *);
    (0x1680, White) (* ogham space *);
    (0x0027, Other) (* ' *);
    (0x0027, Other);
    (0x0073, Letter) (* s *);
    (0x0074, Letter) (* t *);
    (0x006D, Letter) (* m *);
    (0x0064, Letter) (* d *);
    (0x0072, Letter) (* r *);
    (0x0065, Letter) (* e *);
    (0x0076, Letter) (* v *);
    (0x006C, Letter) (* l *);
    (0x0053, Letter) (* S *);
    (0x0054, Letter) (* T *);
    (0x0052, Letter) (* R *);
    (0x004C, Letter) (* L *);
    (0x0061, Letter) (* a *);
    (0x005A, Letter) (* Z *);
    (0x00E9, Letter) (* é *);
    (0x00DF, Letter) (* ß *);
    (0x041F, Letter) (* П *);
    (0x05E9, Letter) (* ש *);
    (0x0627, Letter) (* ا *);
    (0x4E2D, Letter) (* 中 *);
    (0x3042, Letter) (* あ *);
    (0xD55C, Letter) (* 한 *);
    (0x0030, Number) (* 0 *);
    (0x0039, Number) (* 9 *);
    (0x0661, Number) (* Arabic-Indic one *);
    (0x00BD, Number) (* ½ *);
    (0x2160, Number) (* Roman numeral one *);
    (0x00B2, Number) (* superscript two *);
    (0x0301, Other) (* combining acute: a mark, not a letter *);
    (0x0F71, Other) (* combining Tibetan vowel *);
    (0x2460, Number) (* circled one: an other number *);
    (0x200B, Other) (* zero width space: not White_Space *);
    (0x00AD, Other) (* soft hyphen *);
    (0x200D, Other) (* zero width joiner *);
    (0xFE0F, Other) (* variation selector *);
    (0x1F600, Other) (* 😀 *);
    (0x1F44D, Other) (* 👍 *);
    (0x1F3FB, Other) (* skin tone *);
    (0xE000, Other) (* private use *);
    (0x10FFFF, Other) (* last code point *);
    (0x002E, Other) (* . *);
    (0x002C, Other);
    (0x0021, Other);
    (0x003F, Other);
    (0x002D, Other);
    (0x005F, Other);
    (0x002B, Other);
    (0x003D, Other);
    (0x007C, Other);
    (0x0040, Other);
    (0x0023, Other);
    (0x002F, Other);
    (0x005C, Other);
    (0x0022, Other);
    (0x0028, Other);
    (0x0029, Other);
  |]

let add_utf8 buf code =
  if code < 0x80 then Buffer.add_char buf (Char.chr code)
  else if code < 0x800 then begin
    Buffer.add_char buf (Char.chr (0xC0 lor (code lsr 6)));
    Buffer.add_char buf (Char.chr (0x80 lor (code land 0x3F)))
  end
  else if code < 0x10000 then begin
    Buffer.add_char buf (Char.chr (0xE0 lor (code lsr 12)));
    Buffer.add_char buf (Char.chr (0x80 lor ((code lsr 6) land 0x3F)));
    Buffer.add_char buf (Char.chr (0x80 lor (code land 0x3F)))
  end
  else begin
    Buffer.add_char buf (Char.chr (0xF0 lor (code lsr 18)));
    Buffer.add_char buf (Char.chr (0x80 lor ((code lsr 12) land 0x3F)));
    Buffer.add_char buf (Char.chr (0x80 lor ((code lsr 6) land 0x3F)));
    Buffer.add_char buf (Char.chr (0x80 lor (code land 0x3F)))
  end

(* Spans of the characters [points], whose byte offsets are [offsets]. *)
let reference_gpt2_spans points offsets stop =
  let n = Array.length points in
  let code k = fst points.(k) in
  let class_of k = snd points.(k) in
  let offset k = if k < n then offsets.(k) else stop in
  let run holds k =
    let j = ref k in
    while !j < n && holds (class_of !j) do
      incr j
    done;
    !j
  in
  let letter c = c = Letter in
  let number c = c = Number in
  let other c = c = Other in
  let space c = c = Space || c = White in
  let contraction k =
    let ascii k = if k < n && code k < 128 then Char.chr (code k) else '\000' in
    if code k <> Char.code '\'' then 0
    else
      let c1 = ascii (k + 1) in
      if c1 = 's' || c1 = 't' || c1 = 'm' || c1 = 'd' then 2
      else
        let c2 = ascii (k + 2) in
        if
          (c1 = 'r' && c2 = 'e')
          || (c1 = 'v' && c2 = 'e')
          || (c1 = 'l' && c2 = 'l')
        then 3
        else 0
  in
  let spans = ref [] in
  let i = ref 0 in
  while !i < n do
    let start = !i in
    let c = contraction !i in
    if c > 0 then i := !i + c
    else begin
      let j = if class_of !i = Space then !i + 1 else !i in
      if j < n && letter (class_of j) then i := run letter j
      else if j < n && number (class_of j) then i := run number j
      else if j < n && other (class_of j) then i := run other j
      else
        (* [\s+(?!\S)] takes the whole run, or all but its last character when a
           non-whitespace character follows. *)
        let e = run space !i in
        i := if e < n && e - 1 > !i then e - 1 else e
    end;
    spans := (offset start, offset !i) :: !spans
  done;
  List.rev !spans

let malformed =
  [|
    "\xc3";
    "\xe2\x80";
    "\xf0\x9f\x98";
    "\x80";
    "\xbf";
    "\xff";
    "\xfe";
    "\xc0";
    "\xc1";
    "\xc0\x80";
    "\xed\xa0\x80";
    "\xf5\x80\x80\x80";
    "\xf8\x88\x80\x80";
    "\xe0\x41\x42";
  |]

(* Deterministic xorshift, so a failure is reproducible. *)
let next_random =
  let state = ref 0x2545F4914F6CDD1D in
  fun bound ->
    let x = !state in
    let x = x lxor (x lsl 13) in
    let x = x lxor (x lsr 7) in
    let x = x lxor (x lsl 17) in
    state := x land max_int;
    !state mod bound

let random_classified length =
  let count = 1 + next_random length in
  let points =
    Array.init count (fun _ -> alphabet.(next_random (Array.length alphabet)))
  in
  let offsets = Array.make count 0 in
  let buf = Buffer.create (4 * count) in
  Array.iteri
    (fun k (code, _) ->
      offsets.(k) <- Buffer.length buf;
      add_utf8 buf code)
    points;
  (Buffer.contents buf, points, offsets)

let random_malformed length =
  let buf = Buffer.create 32 in
  for _ = 1 to 1 + next_random length do
    Buffer.add_string buf malformed.(next_random (Array.length malformed))
  done;
  Buffer.contents buf

let test_byte_level_matches_pattern () =
  let tokenizer = Pre.byte_level ~add_prefix_space:false ~use_regex:true () in
  for _ = 1 to 5000 do
    let text, points, offsets = random_classified 12 in
    equal
      ~msg:(Printf.sprintf "GPT-2 spans of %S" text)
      (list (pair int int))
      (reference_gpt2_spans points offsets (String.length text))
      (List.map snd (Pre.pre_tokenize tokenizer text))
  done

(* Malformed UTF-8 reaches the walker whenever a user hands brot bytes that are
   not text. It must not raise, not read past the string, and still cover it. *)
let test_byte_level_malformed_utf8 () =
  let tokenizer = Pre.byte_level ~add_prefix_space:false ~use_regex:true () in
  let check text =
    let offsets = List.map snd (Pre.pre_tokenize tokenizer text) in
    let position = ref 0 in
    List.iter
      (fun (start, stop) ->
        equal
          ~msg:(Printf.sprintf "span of %S starts where the last stopped" text)
          int !position start;
        equal
          ~msg:(Printf.sprintf "span of %S is not empty" text)
          bool true (stop > start);
        position := stop)
      offsets;
    equal
      ~msg:(Printf.sprintf "spans of %S cover it" text)
      int (String.length text) !position
  in
  Array.iter check malformed;
  for _ = 1 to 5000 do
    check (random_malformed 8)
  done

(* Offsets place a piece in the text the caller gave, whichever pre-tokenizer
   and whatever it rewrote on the way: a piece cut from marked or byte-level
   encoded text can be longer than the text it came from, and it is that text a
   span has to name. *)
let test_offsets_are_source_spans () =
  let tokenizers =
    [
      ("metaspace", Pre.metaspace ());
      ("metaspace ~split:false", Pre.metaspace ~split:false ());
      ("metaspace `Never", Pre.metaspace ~prepend_scheme:`Never ());
      ("metaspace ~replacement:_", Pre.metaspace ~replacement:"_" ());
      ("byte_level", Pre.byte_level ());
      ("byte_level ~use_regex:false", Pre.byte_level ~use_regex:false ());
      ("bert", Pre.bert ());
      ("whitespace", Pre.whitespace ());
      ("unicode_scripts", Pre.unicode_scripts ());
      ("fixed_length 3", Pre.fixed_length 3);
      ( "sequence [whitespace_split; metaspace]",
        Pre.sequence [ Pre.whitespace_split (); Pre.metaspace () ] );
      ( "sequence [metaspace; punctuation]",
        Pre.sequence [ Pre.metaspace (); Pre.punctuation () ] );
      ( "sequence [whitespace_split; metaspace; punctuation]",
        Pre.sequence
          [ Pre.whitespace_split (); Pre.metaspace (); Pre.punctuation () ] );
      ( "sequence [byte_level; punctuation]",
        Pre.sequence [ Pre.byte_level (); Pre.punctuation () ] );
      ( "sequence [metaspace ~split:false]",
        Pre.sequence [ Pre.metaspace ~split:false () ] );
      ( "sequence [fixed_length; metaspace]",
        Pre.sequence [ Pre.fixed_length 3; Pre.metaspace () ] );
    ]
  in
  let check text =
    let len = String.length text in
    List.iter
      (fun (name, t) ->
        let previous = ref 0 in
        List.iter
          (fun (_, (start, stop)) ->
            let msg =
              Printf.sprintf "%s on %S gives %d,%d" name text start stop
            in
            equal ~msg bool true (0 <= start && start <= stop && stop <= len);
            equal ~msg:(msg ^ ", ascending") bool true (start >= !previous);
            previous := start)
          (Pre.pre_tokenize t text))
      tokenizers
  in
  List.iter check
    [
      "";
      " ";
      "  ";
      "Hello world";
      "trailing ";
      "  two  spaces";
      "a b!";
      "\u{65e5}\u{672c} \u{8a9e}";
      "\u{2581}already";
      "a\u{a0}b";
      "tab\tx";
      "don't stop, 3.14!";
    ];
  Array.iter check malformed;
  for _ = 1 to 500 do
    check (random_malformed 6)
  done

(* [byte_level_decode] is the raw bytes of one token: no replacement character
   is written here, and one character outside the alphabet leaves the token as
   it is. Every expectation is what HuggingFace's [CHAR_BYTES] lookup gives
   before its lossy conversion. *)
let test_byte_level_decode () =
  let check token expected =
    equal
      ~msg:(Printf.sprintf "byte_level_decode %S" token)
      string expected
      (Pre.byte_level_decode token)
  in
  check "" "";
  check "\xc4\xa0Hello" " Hello";
  check "\xc4\x80" "\x00";
  check "\xc3\xb0\xc5\x81\xc4\xb3\xc4\xaf" "\xf0\x9f\x91\x8d";
  check "\xc3\xa9" "\xe9";
  check "\xc2\xa1\xc2\xac\xc2\xae\xc3\xbf" "\xa1\xac\xae\xff";
  (* Neither a character above the alphabet, a literal space, nor a byte
     sequence that is not UTF-8 maps, and each takes the whole token with it. *)
  check "\xe6\x97\xa5" "\xe6\x97\xa5";
  check "\xc4\xa0\xe6\x97\xa5" "\xc4\xa0\xe6\x97\xa5";
  check "a b" "a b";
  check "\xc4\xa0a\xc3" "\xc4\xa0a\xc3";
  check "\xc4" "\xc4";
  (* Encoding then decoding is the identity, on any bytes at all. *)
  let roundtrip text =
    equal
      ~msg:(Printf.sprintf "round trip of %S" text)
      string text
      (Pre.byte_level_decode
         (fst
            (List.hd
               (Pre.pre_tokenize
                  (Pre.byte_level ~use_regex:false ~add_prefix_space:false ())
                  text))))
  in
  Array.iter roundtrip malformed;
  for _ = 1 to 2000 do
    roundtrip (random_malformed 8)
  done

(* [fill] hands out spans a bounded chunk at a time, and a sequence whose first
   member covers the whole text in one span cannot make progress until the
   buffer grows. Both paths only show up on inputs bigger than a chunk. *)
let test_chunked_walk () =
  let tokenizer = Pre.byte_level ~add_prefix_space:false ~use_regex:true () in
  let text, points, offsets = random_classified 60_000 in
  equal
    ~msg:(Printf.sprintf "GPT-2 spans of a %d byte text" (String.length text))
    (list (pair int int))
    (reference_gpt2_spans points offsets (String.length text))
    (List.map snd (Pre.pre_tokenize tokenizer text));

  (* No whitespace, so the first member yields a single span and the second
     needs more room than the buffer holds. *)
  let dense =
    String.concat ""
      (List.init 400 (fun i -> if i land 1 = 0 then "a" else "."))
  in
  equal ~msg:"Sequence over a whitespace-free text"
    (list (pair string (pair int int)))
    (Pre.pre_tokenize (Pre.punctuation ()) dense)
    (Pre.pre_tokenize
       (Pre.sequence [ Pre.whitespace_split (); Pre.punctuation () ])
       dense)

(* Serialization. Every expectation is the JSON HuggingFace writes for the same
   pre-tokenizer, and every rejected shape is one it refuses to read. *)

let json_text t =
  match
    Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json (Pre.to_json t)
  with
  | Ok text -> text
  | Error e -> failwith e

let of_text text =
  match Jsont_bytesrw.decode_string Jsont.json text with
  | Ok json -> Pre.of_json json
  | Error e -> failwith e

let test_json_shape () =
  let shape name t expected =
    equal
      ~msg:(Printf.sprintf "%s serializes" name)
      string expected (json_text t)
  in
  shape "split"
    (Pre.split ~pattern:"," ())
    {|{"type":"Split","pattern":{"String":","},"behavior":"Removed","invert":false}|};
  shape "split ~invert"
    (Pre.split ~pattern:"ab" ~behavior:`Isolated ~invert:true ())
    {|{"type":"Split","pattern":{"String":"ab"},"behavior":"Isolated","invert":true}|};
  shape "metaspace" (Pre.metaspace ())
    {|{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}|};
  shape "metaspace ~prepend_scheme:`Never"
    (Pre.metaspace ~replacement:"_" ~prepend_scheme:`Never ~split:false ())
    {|{"type":"Metaspace","replacement":"_","prepend_scheme":"never","split":false}|}

let test_json_round_trip () =
  let round_trip name t =
    match of_text (json_text t) with
    | Ok t' ->
        equal
          ~msg:(Printf.sprintf "%s round-trips" name)
          string (json_text t) (json_text t')
    | Error e -> fail (Printf.sprintf "%s: %s" name e)
  in
  round_trip "byte_level" (Pre.byte_level ());
  round_trip "bert" (Pre.bert ());
  round_trip "whitespace" (Pre.whitespace ());
  round_trip "whitespace_split" (Pre.whitespace_split ());
  round_trip "punctuation" (Pre.punctuation ~behavior:`Merged_with_previous ());
  round_trip "split" (Pre.split ~pattern:"::" ~behavior:`Isolated ());
  round_trip "char_delimiter" (Pre.char_delimiter ",");
  round_trip "char_delimiter ▁" (Pre.char_delimiter "▁");
  round_trip "digits" (Pre.digits ~individual_digits:true ());
  round_trip "metaspace" (Pre.metaspace ());
  round_trip "metaspace ~prepend_scheme:`First"
    (Pre.metaspace ~prepend_scheme:`First ());
  round_trip "unicode_scripts" (Pre.unicode_scripts ());
  round_trip "fixed_length" (Pre.fixed_length 4);
  round_trip "sequence"
    (Pre.sequence [ Pre.whitespace_split (); Pre.punctuation () ])

let test_json_of_hf () =
  let accepts name text expected =
    match of_text text with
    | Ok t ->
        equal
          ~msg:(Printf.sprintf "%s is read" name)
          string expected (json_text t)
    | Error e -> fail (Printf.sprintf "%s: %s" name e)
  in
  let rejects name text =
    match of_text text with
    | Ok _ -> fail (Printf.sprintf "%s was accepted" name)
    | Error _ ->
        equal ~msg:(Printf.sprintf "%s is rejected" name) bool true true
  in
  (* [prepend_scheme] and [split] default as they do in HuggingFace. HuggingFace
     rejects a [Split] without [invert]; brot defaults it. *)
  accepts "metaspace without prepend_scheme"
    {|{"type":"Metaspace","replacement":"▁"}|}
    {|{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}|};
  accepts "split without invert"
    {|{"type":"Split","pattern":{"String":"-"},"behavior":"Isolated"}|}
    {|{"type":"Split","pattern":{"String":"-"},"behavior":"Isolated","invert":false}|};
  rejects "split with a bare pattern"
    {|{"type":"Split","pattern":",","behavior":"Removed","invert":false}|};
  rejects "split with a regex pattern"
    {|{"type":"Split","pattern":{"Regex":"\\d+"},"behavior":"Removed","invert":false}|};
  rejects "metaspace with a capitalised scheme"
    {|{"type":"Metaspace","replacement":"_","prepend_scheme":"Always"}|};
  rejects "metaspace without a replacement" {|{"type":"Metaspace"}|};
  rejects "metaspace with a two-character replacement"
    {|{"type":"Metaspace","replacement":"__"}|};
  rejects "char delimiter of two characters"
    {|{"type":"CharDelimiterSplit","delimiter":"ab"}|};
  accepts "punctuation without behavior" {|{"type":"Punctuation"}|}
    {|{"type":"Punctuation","behavior":"Isolated"}|}

let () =
  run "Pre-tokenizers Test Suite"
    [
      group "byte_level"
        [
          test "ByteLevel basic" test_byte_level_basic;
          test "ByteLevel GPT-2 pattern" test_byte_level_pattern;
          test "ByteLevel prefix space" test_byte_level_prefix_space;
          test "ByteLevel special chars" test_byte_level_special_chars;
          test "ByteLevel unicode" test_byte_level_unicode;
          test "ByteLevel edge cases" test_byte_level_edge_cases;
          test "ByteLevel matches the pattern" test_byte_level_matches_pattern;
          test "ByteLevel on malformed UTF-8" test_byte_level_malformed_utf8;
          test "byte-level decoding" test_byte_level_decode;
        ];
      group "offsets"
        [ test "offsets are source spans" test_offsets_are_source_spans ];
      group "bert"
        [
          test "BERT tokenization" test_bert_pretokenizer;
          test "punctuation class" test_punctuation_class;
        ];
      group "whitespace"
        [
          test "Whitespace tokenization" test_whitespace_pretokenizer;
          test "WhitespaceSplit" test_whitespace_split;
        ];
      group "punctuation"
        [
          test "Punctuation behaviors" test_punctuation_pretokenizer;
          test "every behavior" test_punctuation_behaviors;
        ];
      group "digits" [ test "Digits tokenization" test_digits_pretokenizer ];
      group "split"
        [
          test "every behavior and invert" test_split_behaviors;
          test "patterns of several characters and bytes" test_split_patterns;
          test "CharDelimiterSplit" test_char_delimiter_split;
        ];
      group "sequence"
        [ test "Sequence of tokenizers" test_sequence_pretokenizer ];
      group "fixed_length" [ test "FixedLength chunks" test_fixed_length ];
      group "unicode_scripts" [ test "UnicodeScripts" test_unicode_scripts ];
      group "metaspace"
        [
          test "Metaspace basic" test_metaspace_basic;
          test "Metaspace matches HuggingFace" test_metaspace_huggingface;
          test "Metaspace walks with or without splitting" test_metaspace_plan;
          test "Metaspace offsets are of the source" test_metaspace_offsets;
        ];
      group "chunking" [ test "walking a long text" test_chunked_walk ];
      group "json"
        [
          test "HuggingFace shape" test_json_shape;
          test "round-trip" test_json_round_trip;
          test "reading HuggingFace JSON" test_json_of_hf;
        ];
    ]
