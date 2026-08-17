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

let test_split_pretokenizer () =
  let test_case pattern behavior text expected =
    let tokenizer = Pre.split ~pattern ~behavior () in
    let result = Pre.pre_tokenize tokenizer text in
    check_tokenization
      (Printf.sprintf "Split pattern=%S behavior=%s text=%S" pattern
         (match behavior with
         | `Isolated -> "Isolated"
         | `Removed -> "Removed"
         | `Merged_with_previous -> "MergedPrev"
         | `Merged_with_next -> "MergedNext"
         | `Contiguous -> "Contiguous")
         text)
      result expected
  in

  (* Test different behaviors *)
  test_case "," `Isolated "a,b,c"
    [
      ("a", (0, 1)); (",", (1, 2)); ("b", (2, 3)); (",", (3, 4)); ("c", (4, 5));
    ];

  test_case "," `Removed "a,b,c" [ ("a", (0, 1)); ("b", (2, 3)); ("c", (4, 5)) ];

  test_case "," `Merged_with_previous "a,b,c"
    [ ("a,", (0, 2)); ("b,", (2, 4)); ("c", (4, 5)) ];

  test_case "," `Merged_with_next "a,b,c"
    [ ("a", (0, 1)); (",b", (1, 3)); (",c", (3, 5)) ];

  (* Test with longer pattern *)
  test_case "::" `Isolated "a::b::c"
    [
      ("a", (0, 1)); ("::", (1, 3)); ("b", (3, 4)); ("::", (4, 6)); ("c", (6, 7));
    ]

let test_char_delimiter_split () =
  let test_case delim text expected =
    let result = Pre.pre_tokenize (Pre.char_delimiter delim) text in
    check_tokenization
      (Printf.sprintf "CharDelimiterSplit delim='%c' text=%S" delim text)
      result expected
  in

  test_case ',' "a,b,c" [ ("a", (0, 1)); ("b", (2, 3)); ("c", (4, 5)) ];
  test_case ' ' "hello world" [ ("hello", (0, 5)); ("world", (6, 11)) ];
  test_case '|' "one|two|three"
    [ ("one", (0, 3)); ("two", (4, 7)); ("three", (8, 13)) ];
  test_case ',' "" [];
  test_case ',' "," []

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

let test_metaspace_basic () =
  let test_case text expected =
    let result =
      Pre.pre_tokenize
        (Pre.metaspace ~replacement:'_' ~prepend_scheme:`Always ~split:true ())
        text
    in
    check_strings (Printf.sprintf "Metaspace %S" text) result expected
  in

  test_case "Hello world" [ "_Hello"; "_world" ];
  test_case " starts with space" [ "_starts"; "_with"; "_space" ];
  test_case "" []

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
        ];
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
        [ test "Punctuation behaviors" test_punctuation_pretokenizer ];
      group "digits" [ test "Digits tokenization" test_digits_pretokenizer ];
      group "split"
        [
          test "Split with patterns" test_split_pretokenizer;
          test "CharDelimiterSplit" test_char_delimiter_split;
        ];
      group "sequence"
        [ test "Sequence of tokenizers" test_sequence_pretokenizer ];
      group "fixed_length" [ test "FixedLength chunks" test_fixed_length ];
      group "unicode_scripts" [ test "UnicodeScripts" test_unicode_scripts ];
      group "metaspace" [ test "Metaspace basic" test_metaspace_basic ];
    ]
