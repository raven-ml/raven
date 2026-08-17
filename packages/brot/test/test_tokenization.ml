(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tokenization tests for brot *)

open Windtrap
open Brot

(* Helper function to tokenize text *)
let tokenize_text text =
  (* Pre-tokenize to get all unique tokens *)
  let pre_tokens =
    Pre_tokenizer.pre_tokenize (Pre_tokenizer.whitespace ()) text
  in
  let unique_tokens =
    List.fold_left
      (fun acc (tok, _) -> if List.mem tok acc then acc else tok :: acc)
      [] pre_tokens
    |> List.rev
  in
  (* Build vocabulary with all tokens from the text plus extras *)
  let all_tokens =
    unique_tokens
    @
    (* Add numbered words for long text test *)
    List.init 1000 (fun i -> Printf.sprintf "word%d" i)
  in
  let vocab = List.mapi (fun i token -> (token, i)) all_tokens in

  (* Create WordLevel tokenizer with the vocabulary *)
  let tokenizer =
    word_level ~vocab ~unk_token:"<unk>" ~pre:(Pre_tokenizer.whitespace ()) ()
  in
  encode tokenizer text |> Encoding.tokens |> Array.to_list

(* Basic Tokenization Tests *)

let test_tokenize_words_simple () =
  let tokens = tokenize_text "Hello world!" in
  equal ~msg:"simple words" (list string) [ "Hello"; "world"; "!" ] tokens

let test_tokenize_words_punctuation () =
  let tokens = tokenize_text "don't stop, it's fun!" in
  equal ~msg:"words with punctuation" (list string)
    [ "don"; "'"; "t"; "stop"; ","; "it"; "'"; "s"; "fun"; "!" ]
    tokens

let test_tokenize_words_numbers () =
  let tokens = tokenize_text "I have 42 apples and 3.14 pies" in
  equal ~msg:"words with numbers" (list string)
    [ "I"; "have"; "42"; "apples"; "and"; "3"; "."; "14"; "pies" ]
    tokens

let test_tokenize_words_empty () =
  let tokens = tokenize_text "" in
  equal ~msg:"empty string" (list string) [] tokens

let test_tokenize_words_whitespace_only () =
  let tokens = tokenize_text "   \t\n  " in
  equal ~msg:"whitespace only" (list string) [] tokens

let test_tokenize_words_special_chars () =
  let tokens = tokenize_text "hello@world.com #ml $100 C++" in
  equal ~msg:"special characters" (list string)
    [ "hello"; "@"; "world"; "."; "com"; "#"; "ml"; "$"; "100"; "C"; "++" ]
    tokens

(* Character Tokenization Tests *)

let tokenize_chars text =
  let chars = ref [] in
  String.iter (fun c -> chars := String.make 1 c :: !chars) text;
  List.rev !chars

let test_tokenize_chars_ascii () =
  let tokens = tokenize_chars "Hi!" in
  equal ~msg:"ASCII chars" (list string) [ "H"; "i"; "!" ] tokens

let test_tokenize_chars_unicode () =
  let tokens = tokenize_chars "Hello 👋 世界" in
  (* Note: UTF-8 encoding means multi-byte chars may appear differently *)
  equal ~msg:"has tokens" bool true (List.length tokens > 0)

let test_tokenize_chars_empty () =
  let tokens = tokenize_chars "" in
  equal ~msg:"empty string chars" (list string) [] tokens

(* Pre-tokenizer Pattern Tests *)

let test_tokenize_regex_words () =
  (* Use the helper that sets up vocabulary properly *)
  let tokens = tokenize_text "hello-world test_123" in
  equal ~msg:"regex words" (list string)
    [ "hello"; "-"; "world"; "test_123" ]
    tokens

let test_tokenize_regex_custom () =
  (* Test with punctuation pre-tokenizer *)
  let text = "don't stop!" in
  let pre_tokens =
    Pre_tokenizer.pre_tokenize (Pre_tokenizer.punctuation ()) text
  in
  let vocab = List.mapi (fun i (tok, _) -> (tok, i)) pre_tokens in
  let tokenizer =
    word_level ~vocab ~unk_token:"<unk>" ~pre:(Pre_tokenizer.punctuation ()) ()
  in
  let tokens = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"has tokens" bool true (List.length tokens > 0)

let test_tokenize_regex_no_match () =
  let tokenizer = word_level () in
  let tokens =
    encode tokenizer "no numbers here" |> Encoding.tokens |> Array.to_list
  in
  equal ~msg:"regex no match" (list string) [] tokens

(* Unigram Model Tests *)

(* Round-trip lookups *)
let test_unigram_roundtrip () =
  let tokens = [ "hello"; "world"; "test" ] in
  let vocab = List.map (fun token -> (token, 0.0)) tokens in
  let tokenizer = unigram ~vocab () in
  List.iteri
    (fun expected_id token ->
      equal
        ~msg:(Printf.sprintf "token_to_id '%s'" token)
        (option int) (Some expected_id)
        (token_to_id tokenizer token);
      equal
        ~msg:(Printf.sprintf "id_to_token %d" expected_id)
        (option string) (Some token)
        (id_to_token tokenizer expected_id))
    tokens

(* token_to_id - out of vocab *)
let test_unigram_token_to_id_oov () =
  let tokenizer = unigram ~vocab:[ ("hello", 0.0); ("world", 0.0) ] () in
  equal ~msg:"token_to_id out-of-vocab" (option int) None
    (token_to_id tokenizer "missing")

(* id_to_token - out of bounds *)
let test_unigram_id_to_token_oob () =
  let tokenizer = unigram ~vocab:[ ("hello", 0.0); ("world", 0.0) ] () in
  equal ~msg:"id_to_token negative" (option string) None
    (id_to_token tokenizer (-1));
  equal ~msg:"id_to_token out of bounds" (option string) None
    (id_to_token tokenizer 10)

(* Test empty vocabulary *)
let test_unigram_empty_vocab () =
  let tokenizer = unigram ~vocab:[] () in
  equal ~msg:"empty vocab token_to_id" (option int) None
    (token_to_id tokenizer "test");
  equal ~msg:"empty vocab id_to_token" (option string) None
    (id_to_token tokenizer 0)

(* Test special characters and unicode *)
let test_unigram_special_tokens () =
  let tokenizer =
    unigram
      ~vocab:
        [
          ("<unk>", 0.0);
          ("<s>", 0.0);
          ("</s>", 0.0);
          ("▁hello", 0.0);
          ("世界", 0.0);
        ]
      ()
  in
  equal ~msg:"special <unk>" (option int) (Some 0)
    (token_to_id tokenizer "<unk>");
  equal ~msg:"special <s>" (option int) (Some 1) (token_to_id tokenizer "<s>");
  equal ~msg:"sentencepiece token" (option int) (Some 3)
    (token_to_id tokenizer "▁hello");
  equal ~msg:"unicode token" (option int) (Some 4) (token_to_id tokenizer "世界");
  equal ~msg:"id to unicode" (option string) (Some "世界")
    (id_to_token tokenizer 4)

let test_unigram_encode_sequence () =
  let tokenizer = unigram ~vocab:[ ("hello", 0.0); ("world", 0.0) ] () in
  let encoding = encode tokenizer "hello world" in
  let tokens = Encoding.tokens encoding |> Array.to_list in
  equal ~msg:"unigram encode tokens" (list string) [ "hello"; "world" ] tokens

let test_pad_token_set_at_construction () =
  let vocab = [ ("hello", 0); ("world", 1); ("<unk>", 2); ("[PAD]", 3) ] in
  let tokenizer =
    word_level ~vocab ~unk_token:"<unk>"
      ~pre:(Pre_tokenizer.whitespace ())
      ~added_tokens:[ added_token "[PAD]" ]
      ~pad_token:"[PAD]" ()
  in
  equal ~msg:"pad token set" (option string) (Some "[PAD]")
    (pad_token tokenizer);
  let pad_id =
    match token_to_id tokenizer "[PAD]" with
    | Some id -> id
    | None -> failwith "missing pad id"
  in
  let encoding =
    encode tokenizer "hello"
      ~padding:
        {
          length = `Fixed 3;
          direction = `Right;
          pad_id = None;
          pad_type_id = None;
          pad_token = None;
        }
  in
  let ids = Encoding.ids encoding |> Array.to_list in
  let pad_ids = List.tl ids in
  equal ~msg:"pad id matches configured token" (list int) [ pad_id; pad_id ]
    pad_ids

(* Edge Cases *)

let test_tokenize_long_text () =
  let text =
    String.concat " " (List.init 1000 (fun i -> Printf.sprintf "word%d" i))
  in
  let tokens = tokenize_text text in
  equal ~msg:"long text token count" int 1000 (List.length tokens)

let test_tokenize_repeated_punctuation () =
  let tokens = tokenize_text "wow!!! really???" in
  equal ~msg:"repeated punctuation" (list string)
    [ "wow"; "!!!"; "really"; "???" ]
    tokens

let test_tokenize_mixed_whitespace () =
  let tokens = tokenize_text "hello\tworld\nthere\r\nfriend" in
  equal ~msg:"mixed whitespace" (list string)
    [ "hello"; "world"; "there"; "friend" ]
    tokens

(* Expectations from HuggingFace [decoders.WordPiece(prefix="##",
   cleanup=...)]. *)
let test_wordpiece_decoder () =
  let case ~cleanup tokens expected =
    equal
      ~msg:
        (Printf.sprintf "decode cleanup=%b %s" cleanup
           (String.concat "|" tokens))
      string expected
      (Decoder.decode (Decoder.wordpiece ~cleanup ()) tokens)
  in
  case ~cleanup:true [ "una"; "##ffa"; "##ble" ] "unaffable";
  case ~cleanup:true [ "hello"; ","; "world"; "!" ] "hello, world!";
  case ~cleanup:true [ "a"; "?"; "b"; "!"; "c"; ","; "d"; "." ] "a? b! c, d.";
  case ~cleanup:true [ "is"; "n't" ] "isn't";
  case ~cleanup:true [ "i"; "'m" ] "i'm";
  case ~cleanup:true [ "it"; "'s" ] "it's";
  (* Only the space the decoder added is taken back, so a full stop that was a
     token of its own does not swallow the space before the next word. *)
  case ~cleanup:true [ "3"; "."; "14" ] "3. 14";
  case ~cleanup:true [ "don"; "'"; "t" ] "don ' t";
  case ~cleanup:true [ "a"; "do"; "not" ] "a do not";
  (* The cleanup runs on the first piece too, and one of its rules rewrites
     content rather than removing a space. *)
  case ~cleanup:true [ "a do not" ] "a don't";
  (* Cleanup neither trims nor collapses the whitespace inside a token. *)
  case ~cleanup:true [ "  x  " ] "  x  ";
  case ~cleanup:true [] "";
  case ~cleanup:false [ "hello"; ","; "world"; "!" ] "hello , world !";
  case ~cleanup:false [ "3"; "."; "14" ] "3 . 14";
  (* The continuing prefix is only stripped after the first token. *)
  case ~cleanup:true [ "##ab"; "cd" ] "##ab cd"

(* Expectations from HuggingFace [decoders.CTC(pad_token="<pad>",
   word_delimiter_token="|", cleanup=...)]. *)
let test_ctc_decoder () =
  let case ~cleanup tokens expected =
    equal
      ~msg:
        (Printf.sprintf "ctc cleanup=%b %s" cleanup (String.concat " " tokens))
      string expected
      (Decoder.decode (Decoder.ctc ~cleanup ()) tokens)
  in
  (* Consecutive repeats collapse, the pad token goes, and the delimiter becomes
     a space. *)
  case ~cleanup:true
    [ "h"; "e"; "l"; "l"; "o"; "|"; "w"; "o"; "r"; "l"; "d" ]
    "helo world";
  case ~cleanup:true [ "a"; "a"; "b"; "<pad>"; "b"; "|"; "c" ] "abb c";
  case ~cleanup:true [ "a"; "|"; "|"; "b" ] "a b";
  case ~cleanup:true [ "<pad>"; "<pad>" ] "";
  case ~cleanup:true [] "";
  (* The detokenization cleanup applies to every token, the [" do not"] rewrite
     included. *)
  case ~cleanup:true [ " ." ] ".";
  case ~cleanup:true [ "a"; " ,"; "b" ] "a,b";
  case ~cleanup:true [ "x"; " n't" ] "xn't";
  case ~cleanup:true [ "a"; " ' "; "b" ] "a'b";
  case ~cleanup:true [ "a do not" ] "a don't";
  (* Cleanup runs before the delimiter is replaced, so a delimiter that becomes
     a space is not itself eligible for the punctuation rules. *)
  case ~cleanup:true [ "|." ] " .";
  case ~cleanup:true [ "  x  " ] "  x  ";
  (* The pad token is cut out wherever it falls in a token, not only when it is
     the whole of one, and cleanup has no say in it. *)
  case ~cleanup:true [ "x<pad>y" ] "xy";
  case ~cleanup:true [ "<pad>x<pad>" ] "x";
  case ~cleanup:true [ "a<pad>b<pad>c" ] "abc";
  case ~cleanup:true [ "<pad>x"; "y<pad>" ] "xy";
  case ~cleanup:false [ "x<pad>y" ] "xy";
  case ~cleanup:false [ "<pad>x<pad>" ] "x";
  (* A token left empty goes, but one left with a space stays. *)
  case ~cleanup:true [ "<pad>" ] "";
  case ~cleanup:true [ ""; "a" ] "a";
  case ~cleanup:true [ "a"; ""; "b" ] "ab";
  case ~cleanup:true [ "|" ] " ";
  case ~cleanup:false [ "a"; " ,"; "b" ] "a ,b";
  case ~cleanup:false [ "h"; "e"; "l"; "l"; "o"; "|"; "w" ] "helo|w";
  (* Dropping the emptied token is what makes the next decoder in a sequence see
     ["▁a"] as its first token. *)
  let sequenced tokens expected =
    equal
      ~msg:(Printf.sprintf "ctc then metaspace %s" (String.concat " " tokens))
      string expected
      (Decoder.decode
         (Decoder.sequence [ Decoder.ctc (); Decoder.metaspace () ])
         tokens)
  in
  sequenced [ "<pad>"; "\xe2\x96\x81a" ] "a";
  sequenced [ "\xe2\x96\x81a"; "\xe2\x96\x81b" ] "a b"

let decodes decoder cases =
  List.iter
    (fun (tokens, expected) ->
      equal
        ~msg:(Printf.sprintf "[%s]" (String.concat "|" tokens))
        string expected
        (Decoder.decode decoder tokens))
    cases

(* Expectations from HuggingFace [decoders.BPEDecoder(suffix="</w>")]. *)
let test_bpe_decoder () =
  (* The suffix stands for the space that follows the word, wherever in the
     token it occurs, and for nothing at all in the last token. *)
  decodes
    (Decoder.bpe ~suffix:"</w>" ())
    [
      ([ "hel"; "lo</w>"; "wor"; "ld</w>" ], "hello world");
      ([ "a</w>"; "b</w>" ], "a b");
      ([ "a</w>b"; "c" ], "a bc");
      ([ "</w>"; "a" ], " a");
      ([ "only</w>" ], "only");
      (* A token without the suffix is not a word end, so no space is added. *)
      ([ "x"; "y" ], "xy");
      ([], "");
    ]

(* Expectations from HuggingFace [decoders.ByteFallback()]. *)
let test_byte_fallback_decoder () =
  decodes (Decoder.byte_fallback ())
    [
      ([ "<0x41>" ], "A");
      ([ "<0x0a>" ], "\n");
      ([ "<0xE2>"; "<0x96>"; "<0x81>" ], "\xe2\x96\x81");
      ([ "a"; "<0xC3>"; "<0xA9>"; "b" ], "a\xc3\xa9b");
      ([ "<0x0A>"; "x"; "<0x0A>" ], "\nx\n");
      (* A run that is not valid UTF-8 is unrecoverable: one replacement
         character per byte, not one for the run. *)
      ([ "<0xFF>" ], "\xef\xbf\xbd");
      ([ "<0xFF>"; "<0xFE>" ], "\xef\xbf\xbd\xef\xbf\xbd");
      (* Only six-byte [<0xNN>] tokens with two hex digits are byte tokens. *)
      ([ "<0xGG>" ], "<0xGG>");
      ([ "<0x1>" ], "<0x1>");
      ([ "<0x_1>" ], "<0x_1>");
      ([ "plain" ], "plain");
      ([], "");
    ]

(* Expectations from HuggingFace [decoders.Metaspace(prepend_scheme=...)]. *)
let test_metaspace_decoder () =
  (* The marker was prepended to the text rather than standing for a space, so
     it is dropped throughout the first token and becomes a space after it. *)
  decodes (Decoder.metaspace ())
    [
      ([ "\xe2\x96\x81Hello"; "\xe2\x96\x81world" ], "Hello world");
      ([ "\xe2\x96\x81"; "\xe2\x96\x81a" ], " a");
      ([ "a\xe2\x96\x81b"; "\xe2\x96\x81c" ], "ab c");
      ([ "\xe2\x96\x81\xe2\x96\x81a"; "b" ], "ab");
      ([ "\xe2\x96\x81" ], "");
      ([], "");
    ];
  decodes
    (Decoder.metaspace ~prepend_scheme:`Never ())
    [
      ([ "\xe2\x96\x81Hello"; "\xe2\x96\x81world" ], " Hello world");
      ([ "\xe2\x96\x81"; "\xe2\x96\x81a" ], "  a");
      ([ "a\xe2\x96\x81b"; "\xe2\x96\x81c" ], "a b c");
      ([ "\xe2\x96\x81" ], " ");
    ]

(* Expectations from HuggingFace [decoders.Replace(pattern, content)] and
   [decoders.Strip(content, left, right)]. *)
let test_replace_and_strip_decoders () =
  decodes
    (Decoder.replace ~pattern:"\xe2\x96\x81" ~by:" " ())
    [ ([ "\xe2\x96\x81a"; "\xe2\x96\x81b" ], " a b"); ([ "a" ], "a"); ([], "") ];
  (* Every token is stripped, not just the ends of the joined text. *)
  decodes
    (Decoder.strip ~content:" " ~start:1 ())
    [ ([ "  a  "; "  b  " ], " a   b  "); ([ "   " ], "  "); ([ "a" ], "a") ];
  decodes
    (Decoder.strip ~content:" " ~stop:1 ())
    [ ([ "  a  "; "  b  " ], "  a   b ") ];
  decodes
    (Decoder.strip ~content:" " ~start:2 ~stop:2 ())
    [ ([ "  a  "; "  b  " ], "ab"); ([ "a" ], "a"); ([], "") ];
  (* [content] is a string, so a multi-byte marker counts as one occurrence. *)
  decodes
    (Decoder.strip ~content:"\xe2\x96\x81" ~start:1 ())
    [ ([ "\xe2\x96\x81\xe2\x96\x81x" ], "\xe2\x96\x81x"); ([ "  a" ], "  a") ]

(* Expectations from HuggingFace for the decoder LLaMA and other SentencePiece
   models carry: [decoders.Sequence([Replace("▁", " "), ByteFallback(), Fuse(),
   Strip(content=" ", left=1)])]. *)
let test_sentencepiece_decoder () =
  let sp =
    Decoder.sequence
      [
        Decoder.replace ~pattern:"\xe2\x96\x81" ~by:" " ();
        Decoder.byte_fallback ();
        Decoder.fuse ();
        Decoder.strip ~content:" " ~start:1 ();
      ]
  in
  decodes sp
    [
      ([ "\xe2\x96\x81a"; "\xe2\x96\x81"; "\xe2\x96\x81b" ], "a  b");
      ([ "\xe2\x96\x81"; "<0x0A>"; "<0x0A>"; "Not" ], "\n\nNot");
      ([ "\xe2\x96\x81Hello"; "\xe2\x96\x81world" ], "Hello world");
      (* [Replace] rewrites each token, so [ByteFallback] still sees the byte
         tokens as tokens of their own. *)
      ([ "\xe2\x96\x81"; "<0xE2>"; "<0x9C>"; "<0x93>" ], "\xe2\x9c\x93");
      ([ "<0xFF>"; "\xe2\x96\x81a" ], "\xef\xbf\xbd a");
      ([ "\xe2\x96\x81" ], "");
      ([], "");
    ]

(* [skip_special_tokens] drops a special added token wherever its identifier
   appears. HuggingFace instead matches the token by the string it decodes to,
   which for a token whose [normalized] is set is the normalized form, and looks
   that string up among unnormalized contents — so on a configuration like this
   one it skips nothing and decodes ["<s> a b"] either way. Brot does not follow
   it there. *)
let test_skip_special_tokens () =
  let marker = "\xe2\x96\x81" in
  let t =
    bpe
      ~normalizer:
        (Normalizer.sequence
           [
             Normalizer.prepend marker;
             Normalizer.replace ~pattern:" " ~replacement:marker;
           ])
      ~decoder:
        (Decoder.sequence
           [
             Decoder.replace ~pattern:marker ~by:" " ();
             Decoder.fuse ();
             Decoder.strip ~content:" " ~start:1 ();
           ])
      ~added_tokens:[ added_token ~normalized:true "<s>" ]
      ~vocab:[ ("<s>", 1); (marker ^ "a", 2); (marker ^ "b", 3) ]
      ~merges:[] ()
  in
  equal ~msg:"the added token stands for the text it matched" (option string)
    (Some (marker ^ "<s>"))
    (id_to_token t 1);
  equal ~msg:"kept" string "<s> a b"
    (decode t ~skip_special_tokens:false [| 1; 2; 3 |]);
  equal ~msg:"skipped" string "a b"
    (decode t ~skip_special_tokens:true [| 1; 2; 3 |]);
  equal ~msg:"nothing but the skipped token" string ""
    (decode t ~skip_special_tokens:true [| 1 |])

(* Test Suite *)

let tokenization_tests =
  [
    (* Words tokenization *)
    test "tokenize words simple" test_tokenize_words_simple;
    test "tokenize words punctuation" test_tokenize_words_punctuation;
    test "tokenize words numbers" test_tokenize_words_numbers;
    test "tokenize words empty" test_tokenize_words_empty;
    test "tokenize words whitespace only" test_tokenize_words_whitespace_only;
    test "tokenize words special chars" test_tokenize_words_special_chars;
    (* Character tokenization *)
    test "tokenize chars ASCII" test_tokenize_chars_ascii;
    test "tokenize chars unicode" test_tokenize_chars_unicode;
    test "tokenize chars empty" test_tokenize_chars_empty;
    (* Regex tokenization *)
    test "tokenize regex words" test_tokenize_regex_words;
    test "tokenize regex custom" test_tokenize_regex_custom;
    test "tokenize regex no match" test_tokenize_regex_no_match;
    (* Edge cases *)
    test "tokenize long text" test_tokenize_long_text;
    test "tokenize repeated punctuation" test_tokenize_repeated_punctuation;
    test "tokenize mixed whitespace" test_tokenize_mixed_whitespace;
    (* Unigram model tests *)
    test "unigram roundtrip" test_unigram_roundtrip;
    test "unigram token_to_id out-of-vocab" test_unigram_token_to_id_oov;
    test "unigram id_to_token out-of-bounds" test_unigram_id_to_token_oob;
    test "unigram empty vocab" test_unigram_empty_vocab;
    test "unigram special tokens" test_unigram_special_tokens;
    test "unigram encode sequence" test_unigram_encode_sequence;
    test "pad token reassignment updates id" test_pad_token_set_at_construction;
    (* Decoding *)
    test "wordpiece decoder" test_wordpiece_decoder;
    test "ctc decoder" test_ctc_decoder;
    test "bpe decoder" test_bpe_decoder;
    test "byte fallback decoder" test_byte_fallback_decoder;
    test "metaspace decoder" test_metaspace_decoder;
    test "replace and strip decoders" test_replace_and_strip_decoders;
    test "sentencepiece decoder" test_sentencepiece_decoder;
    test "skip special tokens" test_skip_special_tokens;
  ]

let () = run "brot tokenization" [ group "tokenization" tokenization_tests ]
