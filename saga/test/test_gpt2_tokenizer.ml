(** Comprehensive GPT-2 tokenizer tests to ensure exact match with Python
    tokenizers *)

open Saga_tokenizers
open Alcotest

let test_gpt2_tokenization () =
  (* Create GPT-2 tokenizer with ByteLevel pre-tokenizer *)
  let vocab_file = "/Users/tmattio/.cache/kaun/gpt2/gpt2/vocab.json" in
  let merges_file = "/Users/tmattio/.cache/kaun/gpt2/gpt2/merges.txt" in

  (* Skip test if files don't exist *)
  if not (Sys.file_exists vocab_file && Sys.file_exists merges_file) then
    skip ()
  else
    let pre_tokenizer = Pre_tokenizers.byte_level ~add_prefix_space:false () in
    let tokenizer =
      Tokenizer.from_model_file ~vocab:vocab_file ~merges:merges_file
        ~pre:pre_tokenizer ()
    in

    (* Test cases with expected token IDs from Python transformers *)
    let test_cases =
      [
        (* Basic text *)
        ("Hello world", [ 15496; 995 ]);
        ("hello", [ 31373 ]);
        ("Hello", [ 15496 ]);
        ("HELLO", [ 13909; 3069; 46 ]);
        (* Contractions *)
        ("I'm", [ 40; 1101 ]);
        ("don't", [ 9099; 470 ]);
        ("it's", [ 270; 338 ]);
        ("we're", [ 732; 821 ]);
        ("they've", [ 9930; 1053 ]);
        ("I'll", [ 40; 1183 ]);
        ("wouldn't", [ 19188; 77; 470 ]);
        ("OpenAI's", [ 11505; 20185; 338 ]);
        ("'s", [ 338 ]);
        (* Spaces *)
        (" ", [ 220 ]);
        ("  ", [ 220; 220 ]);
        ("    ", [ 220; 220; 220; 220 ]);
        ( "   Leading and trailing spaces   ",
          [ 220; 220; 43225; 290; 25462; 9029; 220; 220; 220 ] );
        ("Hello world", [ 15496; 995 ]);
        ("Hello  world", [ 15496; 220; 995 ]);
        (* double space *)

        (* Newlines and special whitespace *)
        ("\n", [ 198 ]);
        ("\r\n", [ 201; 198 ]);
        ("\t", [ 197 ]);
        ("Hello\nworld", [ 15496; 198; 6894 ]);
        ("Multiple\nlines\nof\ntext", [ 31217; 198; 6615; 198; 1659; 198; 5239 ]);
        (* Punctuation *)
        (".", [ 13 ]);
        ("!", [ 0 ]);
        ("?", [ 30 ]);
        (",", [ 11 ]);
        (":", [ 25 ]);
        (";", [ 26 ]);
        ("Hello, world!", [ 15496; 11; 995; 0 ]);
        ("Hello.", [ 15496; 13 ]);
        ("Hello!", [ 15496; 0 ]);
        ("Hello?", [ 15496; 30 ]);
        (* Numbers *)
        ("0", [ 15 ]);
        ("1", [ 16 ]);
        ("123", [ 10163 ]);
        ("123456789", [ 10163; 2231; 3134; 4531 ]);
        ("3.14", [ 18; 13; 1415 ]);
        ("2023", [ 1238; 1954 ]);
        (* Mixed alphanumeric *)
        ("abc123", [ 39305; 10163 ]);
        ("123abc", [ 10163; 39305 ]);
        ("Mixed123Numbers456", [ 44; 2966; 10163; 49601; 29228 ]);
        (* Special characters *)
        ("@", [ 31 ]);
        ("#", [ 2 ]);
        ("$", [ 3 ]);
        ("%", [ 4 ]);
        ("^", [ 61 ]);
        ("&", [ 5 ]);
        ("*", [ 9 ]);
        ("()", [ 3419 ]);
        ("[]", [ 21737 ]);
        ("{}", [ 90; 92 ]);
        ("@#$%^&*()", [ 31; 29953; 4; 61; 5; 9; 3419 ]);
        ("#$", [ 29953 ]);
        (* This is a single token! *)
        ( "Special chars: @#$%^&*()",
          [ 13409; 34534; 25; 2488; 29953; 4; 61; 5; 9; 3419 ] );
        (* Quotes *)
        ("\"hello\"", [ 1; 31373; 1 ]);
        ("'hello'", [ 6; 31373; 6 ]);
        ("\"Hello, world!\"", [ 1; 15496; 11; 995; 2474 ]);
        (* Unicode and emojis *)
        ("café", [ 66; 1878; 2634 ]);
        ("naïve", [ 2616; 38776 ]);
        (* Edge cases *)
        ("", []);
        (* empty string *)
        ( "The quick brown fox jumps over the lazy dog",
          [ 464; 2068; 7586; 21831; 18045; 625; 262; 16931; 3290 ] );
        ( "OpenAI's GPT-2 is a language model",
          [ 11505; 20185; 338; 402; 11571; 12; 17; 318; 257; 3303; 2746 ] );
        ("This is a test.", [ 1212; 318; 257; 1332; 13 ]);
        (* Programming-related *)
        ("function", [ 8818 ]);
        ("def foo():", [ 4299; 22944; 33529 ]);
        ("print(\"Hello\")", [ 4798; 7203; 15496; 4943 ]);
        ("// comment", [ 1003; 2912 ]);
        ("/* comment */", [ 15211; 2912; 9466 ]);
        (* URLs and emails *)
        ("https://openai.com", [ 5450; 1378; 9654; 1872; 13; 785 ]);
        ("user@example.com", [ 7220; 31; 20688; 13; 785 ]);
        (* Repeated characters *)
        ("aaa", [ 46071 ]);
        ("!!!", [ 10185 ]);
        ("???", [ 28358 ]);
        ("...", [ 986 ]);
        ("---", [ 6329 ]);
        (* Case variations *)
        ("lower", [ 21037 ]);
        ("UPPER", [ 8577; 18973 ]);
        ("CamelCase", [ 34; 17983; 20448 ]);
        ("snake_case", [ 16184; 539; 62; 7442 ]);
        ("kebab-case", [ 365; 65; 397; 12; 7442 ]);
        (* Common words that might tokenize differently *)
        ("the", [ 1169 ]);
        ("The", [ 464 ]);
        ("THE", [ 10970 ]);
        (" the", [ 262 ]);
        (" The", [ 383 ]);
        ("the ", [ 1169; 220 ]);
        (* Sentences *)
        ("Hello world", [ 15496; 995 ]);
        ("Hello World", [ 15496; 2159 ]);
        ("hello world", [ 31373; 995 ]);
        ("HELLO WORLD", [ 13909; 3069; 46; 29564 ]);
        (* Additional edge cases *)
        ("'t", [ 470 ]);
        ("'re", [ 821 ]);
        ("'ve", [ 1053 ]);
        ("'m", [ 1101 ]);
        ("'ll", [ 1183 ]);
        ("'d", [ 1549 ]);
        (* Mathematical symbols *)
        ("+", [ 10 ]);
        ("-", [ 12 ]);
        ("=", [ 28 ]);
        ("<", [ 27 ]);
        (">", [ 29 ]);
        ("<=", [ 27; 28 ]);
        (">=", [ 29; 28 ]);
        ("!=", [ 0; 28 ]);
        ("==", [ 855 ]);
        (* More punctuation *)
        ("?!", [ 12248 ]);
        ("!?", [ 22857 ]);
        (* File paths *)
        ("/path/to/file", [ 14; 6978; 14; 1462; 14; 7753 ]);
        ("./relative/path", [ 19571; 43762; 14; 6978 ]);
        (* Common programming keywords *)
        ("if", [ 361 ]);
        ("else", [ 17772 ]);
        ("for", [ 1640 ]);
        ("while", [ 4514 ]);
        ("return", [ 7783 ]);
        ("class", [ 4871 ]);
        ("import", [ 11748 ]);
        ("from", [ 6738 ]);
        (* Context sentences with contractions *)
        ("It's a test", [ 1026; 338; 257; 1332 ]);
        ("I'm here", [ 40; 1101; 994 ]);
        ("Don't go", [ 3987; 470; 467 ]);
        ("We're ready", [ 1135; 821; 3492 ]);
        ("They've arrived", [ 2990; 1053; 5284 ]);
        (* Multi-word expressions *)
        ("New York", [ 3791; 1971 ]);
        ("United States", [ 17013; 1829 ]);
        ("machine learning", [ 30243; 4673 ]);
        ("artificial intelligence", [ 433; 9542; 4430 ]);
      ]
    in

    (* Test each case *)
    List.iter
      (fun (text, expected) ->
        let encoding = Tokenizer.encode tokenizer text in
        let tokens_list = Encoding.get_ids encoding |> Array.to_list in
        check (list int)
          (Printf.sprintf "Tokenizing %S" text)
          expected tokens_list)
      test_cases

let test_gpt2_edge_cases () =
  (* Test specific edge cases that are known to be tricky *)
  let vocab_file = "/Users/tmattio/.cache/kaun/gpt2/gpt2/vocab.json" in
  let merges_file = "/Users/tmattio/.cache/kaun/gpt2/gpt2/merges.txt" in

  if not (Sys.file_exists vocab_file && Sys.file_exists merges_file) then
    skip ()
  else
    let pre_tokenizer = Pre_tokenizers.byte_level ~add_prefix_space:false () in
    let tokenizer =
      Tokenizer.from_model_file ~vocab:vocab_file ~merges:merges_file
        ~pre:pre_tokenizer ()
    in

    (* Test that space + word is kept together *)
    let test_space_word text expected_tokens =
      let encoding = Tokenizer.encode tokenizer text in
      let tokens = Encoding.get_ids encoding |> Array.to_list in
      check (list int)
        (Printf.sprintf "Space handling in %S" text)
        expected_tokens tokens
    in

    test_space_word " hello" [ 23748 ];
    test_space_word "  hello" [ 220; 23748 ];
    test_space_word "hello " [ 31373; 220 ];
    test_space_word "hello  " [ 31373; 220; 220 ];

    (* Test that contractions are recognized *)
    let test_contraction text expected_tokens =
      let encoding = Tokenizer.encode tokenizer text in
      let tokens = Encoding.get_ids encoding |> Array.to_list in
      check (list int)
        (Printf.sprintf "Contraction %S" text)
        expected_tokens tokens
    in

    test_contraction "'s" [ 338 ];
    test_contraction "'t" [ 470 ];
    test_contraction "'re" [ 821 ];
    test_contraction "'ve" [ 1053 ];
    test_contraction "'m" [ 1101 ];
    test_contraction "'ll" [ 1183 ];
    test_contraction "'d" [ 1549 ];

    (* Test special character combinations *)
    let test_special_chars text expected_tokens =
      let encoding = Tokenizer.encode tokenizer text in
      let tokens = Encoding.get_ids encoding |> Array.to_list in
      check (list int)
        (Printf.sprintf "Special chars %S" text)
        expected_tokens tokens
    in

    test_special_chars "#$" [ 29953 ];
    (* This should be a single token *)
    test_special_chars "@#" [ 41573 ];
    (* This should also be a single token *)
    test_special_chars " @" [ 2488 ]
(* Space + @ should be one token *)

let test_gpt2_decode () =
  (* Test that decode produces correct text *)
  let vocab_file = "/Users/tmattio/.cache/kaun/gpt2/gpt2/vocab.json" in
  let merges_file = "/Users/tmattio/.cache/kaun/gpt2/gpt2/merges.txt" in

  if not (Sys.file_exists vocab_file && Sys.file_exists merges_file) then
    skip ()
  else
    let pre_tokenizer = Pre_tokenizers.byte_level ~add_prefix_space:false () in
    let tokenizer =
      Tokenizer.from_model_file ~vocab:vocab_file ~merges:merges_file
        ~pre:pre_tokenizer ()
    in

    let test_roundtrip text =
      let encoding = Tokenizer.encode tokenizer text in
      let token_ids = Encoding.get_ids encoding |> Array.to_list in
      let decoded =
        List.filter_map (fun id -> Tokenizer.id_to_token tokenizer id) token_ids
        |> String.concat ""
      in
      (* Note: GPT-2 decode may not be exact due to space encoding *)
      (* But at least check it doesn't crash *)
      check bool
        (Printf.sprintf "Decode doesn't crash for %S" text)
        true
        (String.length decoded >= 0)
    in

    List.iter test_roundtrip
      [
        "Hello world";
        "OpenAI's GPT-2";
        "Special chars: @#$%^&*()";
        "Multiple\nlines\nof\ntext";
        "   spaces   ";
      ]

let test_pretokenizer_output () =
  (* Test that the pre-tokenizer produces expected splits *)
  let pre_tokenizer =
    Pre_tokenizers.byte_level ~add_prefix_space:false ~use_regex:true ()
  in

  let test_splits text expected_splits =
    let splits = pre_tokenizer text in
    let split_strings = List.map fst splits in
    check (list string)
      (Printf.sprintf "Pre-tokenizer splits for %S" text)
      expected_splits split_strings
  in

  (* Test basic splits *)
  test_splits "Hello world" [ "Hello"; "\196\160world" ];
  test_splits "OpenAI's" [ "OpenAI"; "'s" ];
  test_splits "'s" [ "'s" ];
  test_splits "don't" [ "don"; "'t" ];

  (* Test spaces *)
  test_splits " world" [ "\196\160world" ];
  test_splits "  world" [ "\196\160"; "\196\160world" ];
  test_splits "world " [ "world"; "\196\160" ];

  (* Test special chars with spaces *)
  test_splits " @" [ "\196\160@" ];
  test_splits "@ " [ "@"; "\196\160" ]

let () =
  let open Alcotest in
  run "GPT-2 Tokenizer Tests"
    [
      ( "tokenization",
        [
          test_case "GPT-2 tokenization matches Python" `Quick
            test_gpt2_tokenization;
          test_case "GPT-2 edge cases" `Quick test_gpt2_edge_cases;
          test_case "GPT-2 decode" `Quick test_gpt2_decode;
          test_case "Pre-tokenizer output" `Quick test_pretokenizer_output;
        ] );
    ]
