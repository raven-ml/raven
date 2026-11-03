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
  let pre_tokens = Pre_tokenizer.pre_tokenize Pre_tokenizer.whitespace text in
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
    word_level ~vocab ~unk_token:"<unk>" ~pre:Pre_tokenizer.whitespace ()
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

(* A vocabulary of whole words holds no piece for a single character, so the
   only way into a word's second byte is an unknown token: the model needs one
   to record there even though the best path, the whole word, never spends
   it. *)
let test_unigram_encode_sequence () =
  let tokenizer =
    unigram
      ~vocab:[ ("<unk>", 0.0); ("hello", 0.0); ("world", 0.0) ]
      ~unk_id:0 ~unk_token:"<unk>" ~pre:Pre_tokenizer.whitespace ()
  in
  let encoding = encode tokenizer "hello world" in
  let tokens = Encoding.tokens encoding |> Array.to_list in
  equal ~msg:"unigram encode tokens" (list string) [ "hello"; "world" ] tokens

let test_pad_token_set_at_construction () =
  let vocab = [ ("hello", 0); ("world", 1); ("<unk>", 2); ("[PAD]", 3) ] in
  let tokenizer =
    word_level ~vocab ~unk_token:"<unk>" ~pre:Pre_tokenizer.whitespace
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
  decodes Decoder.byte_fallback
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

(* Expectations from HuggingFace [decoders.ByteLevel()]. A token is mapped
   character by character and falls back whole when one of them is outside the
   alphabet; the bytes of every token are then read as one text. *)
let test_byte_level_decoder () =
  decodes Decoder.byte_level
    [
      ([], "");
      ([ "" ], "");
      ([ "\xc4\xa0Hello" ], " Hello");
      ([ "\xc4\x80" ], "\x00");
      (* A character spelled across tokens comes back whole: these are the four
         bytes of U+1F44D, cut in every way. *)
      ([ "\xc3\xb0\xc5\x81\xc4\xb3\xc4\xaf" ], "\xf0\x9f\x91\x8d");
      ([ "\xc3\xb0\xc5\x81"; "\xc4\xb3\xc4\xaf" ], "\xf0\x9f\x91\x8d");
      ([ "\xc3\xb0"; "\xc5\x81"; "\xc4\xb3"; "\xc4\xaf" ], "\xf0\x9f\x91\x8d");
      ([ "\xc3\xa6"; "\xc4\xb9"; "\xc2\xa5" ], "\xe6\x97\xa5");
      (* Bytes that spell nothing become one replacement character per maximal
         subpart: one for a lone [\xE9] and one for a four-byte sequence cut
         short, but one and a ['('] for [\xC3\x28]. *)
      ([ "\xc3\xa9" ], "\xef\xbf\xbd");
      ([ "\xc3\xb0\xc5\x81\xc4\xb3" ], "\xef\xbf\xbd");
      ([ "\xc3\xb0\xc5\x81\xc4\xb3(" ], "\xef\xbf\xbd(");
      ([ "\xc3\x83(" ], "\xef\xbf\xbd(");
      ( [ "\xc2\xa1"; "\xc2\xac"; "\xc2\xae"; "\xc3\xbf" ],
        "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd" );
      (* The lead of a sequence whose second byte is out of range is a subpart
         of its own, so the bytes after it are subparts too: an overlong
         encoding, a surrogate and a value above U+10FFFF all cost one each. *)
      ([ "\xc3\xa0\xc4\xa2\xc4\xa2" ], "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd");
      ([ "\xc3\xad\xc5\x82\xc4\xa2" ], "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd");
      ( [ "\xc3\xb0\xc4\xa2\xc4\xa2\xc4\xa2" ],
        "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd" );
      ( [ "\xc3\xb4\xc4\xb2\xc4\xa2\xc4\xa2" ],
        "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd" );
      ( [ "\xc3\xb5\xc4\xa2\xc4\xa2\xc4\xa2" ],
        "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd" );
      ([ "\xc3\x81\xc4\xa2" ], "\xef\xbf\xbd\xef\xbf\xbd");
      ([ "\xc3\x80\xc4\xa2" ], "\xef\xbf\xbd\xef\xbf\xbd");
      ([ "\xc4\xa2" ], "\xef\xbf\xbd");
      ([ "\xc3\xbf" ], "\xef\xbf\xbd");
      (* A sequence broken after its second byte costs one for the pair. *)
      ([ "\xc3\xa1\xc4\xa2A" ], "\xef\xbf\xbdA");
      ([ "\xc3\xb0\xc5\x81A" ], "\xef\xbf\xbdA");
      (* The edges of what is well formed still decode. *)
      ([ "\xc3\xa2\xc4\xa4\xc2\xac" ], "\xe2\x82\xac");
      ([ "\xc3\xa0\xc5\x82\xc4\xa2" ], "\xe0\xa0\x80");
      ([ "\xc3\xad\xc5\x81\xc2\xbf" ], "\xed\x9f\xbf");
      ([ "\xc3\xb4\xc4\xb1\xc2\xbf\xc2\xbf" ], "\xf4\x8f\xbf\xbf");
      (* One character outside the alphabet costs the whole token its mapping,
         so it stands for its own bytes — and its neighbours still map. The two
         tokens below differ in their last character alone: without it every
         character maps, [\xc4\xa0] and [\xc3\xa9] among them, and with it none
         does. *)
      ([ "\xc4\xa0caf\xc3\xa9X" ], " caf\xef\xbf\xbdX");
      ([ "\xc4\xa0caf\xc3\xa9\xe6\x97\xa5" ], "\xc4\xa0caf\xc3\xa9\xe6\x97\xa5");
      ([ "\xc4\xa0\xe6\x97\xa5" ], "\xc4\xa0\xe6\x97\xa5");
      ([ "a b" ], "a b");
      ([ "\xc4\xa0a"; "\xe6\x97\xa5"; "\xc4\xa0b" ], " a\xe6\x97\xa5 b");
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
    ];
  (* [`First] decodes as [`Always] does: whichever piece the marker was
     prepended to, it is the first token that carries it. *)
  decodes
    (Decoder.metaspace ~prepend_scheme:`First ())
    [
      ([ "\xe2\x96\x81Hello"; "\xe2\x96\x81world" ], "Hello world");
      ( [ "\xe2\x96\x81Hello"; "\xe2\x96\x81world"; "\xe2\x96\x81\xe2\x96\x81x" ],
        "Hello world  x" );
      ([ "\xe2\x96\x81"; "\xe2\x96\x81a" ], " a");
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
        Decoder.byte_fallback;
        Decoder.fuse;
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

(* Expectations from HuggingFace: a Unigram model behind [Metaspace(split=False,
   prepend_scheme=...)]. The pre-tokenizer hands the model the whole marked text
   as one piece, and every token still reports the bytes of the input it stands
   for rather than that whole piece. *)
let test_metaspace_no_split_offsets () =
  let vocab =
    [
      ("<unk>", 0.0);
      ("\xe2\x96\x81", -3.0);
      ("\xe2\x96\x81Hello", -1.0);
      ("\xe2\x96\x81world", -1.0);
      ("\xe2\x96\x81leading", -1.0);
      ("\xe2\x96\x81trailing", -1.0);
      ("\xe2\x96\x81multi", -1.0);
      ("\xe2\x96\x81space", -1.0);
      ("\xe2\x96\x81\xe6\x97\xa5\xe6\x9c\xac", -1.0);
      ("\xe2\x96\x81\xe8\xaa\x9e", -1.0);
      ("\xe2\x96\x81already", -1.0);
      ("\xe2\x96\x81a", -1.0);
      ("\xe2\x96\x81b", -1.0);
      ("\xe2\x96\x81x", -1.0);
      ("Hello", -2.0);
      ("world", -2.0);
      ("trailing", -2.0);
      ("multi", -2.0);
      ("a", -4.0);
      ("b", -4.0);
      ("\xe6\x97\xa5\xe6\x9c\xac", -2.0);
      ("\xe8\xaa\x9e", -2.0);
      ("already", -2.0);
      ("y", -4.0);
      ("x", -4.0);
      ("\xc2\xa0", -4.0);
    ]
  in
  let tokenizer prepend_scheme =
    unigram ~vocab ~unk_id:0 ~unk_token:"<unk>"
      ~pre:(Pre_tokenizer.metaspace ~prepend_scheme ~split:false ())
      ()
  in
  let case t text ids offsets =
    let encoding = encode t text in
    equal
      ~msg:(Printf.sprintf "ids %S" text)
      (list int) ids
      (Array.to_list (Encoding.ids encoding));
    equal
      ~msg:(Printf.sprintf "offsets %S" text)
      (list (pair int int))
      offsets
      (Array.to_list (Encoding.offsets encoding))
  in
  let always = tokenizer `Always in
  case always "Hello world" [ 2; 3 ] [ (0, 5); (5, 11) ];
  case always "  leading" [ 1; 4 ] [ (0, 1); (1, 9) ];
  case always "trailing " [ 5; 1 ] [ (0, 8); (8, 9) ];
  case always "" [] [];
  case always " " [ 1 ] [ (0, 1) ];
  case always "  " [ 1; 1 ] [ (0, 1); (1, 2) ];
  (* A no-break space is not a space, so no marker stands in for it. *)
  case always "a\xc2\xa0b" [ 11; 25; 19 ] [ (0, 1); (1, 3); (3, 4) ];
  case always "multi  space" [ 6; 1; 7 ] [ (0, 5); (5, 6); (6, 12) ];
  case always "\xe2\x96\x81already" [ 10 ] [ (0, 10) ];
  case always "\xe6\x97\xa5\xe6\x9c\xac \xe8\xaa\x9e" [ 8; 9 ]
    [ (0, 6); (6, 10) ];
  case always "x y" [ 13; 1; 23 ] [ (0, 1); (1, 2); (2, 3) ];
  let never = tokenizer `Never in
  case never "Hello world" [ 14; 3 ] [ (0, 5); (5, 11) ];
  case never "  leading" [ 1; 4 ] [ (0, 1); (1, 9) ];
  case never "trailing " [ 16; 1 ] [ (0, 8); (8, 9) ];
  case never "" [] [];
  case never " " [ 1 ] [ (0, 1) ];
  case never "  " [ 1; 1 ] [ (0, 1); (1, 2) ];
  case never "a\xc2\xa0b" [ 18; 25; 19 ] [ (0, 1); (1, 3); (3, 4) ];
  case never "multi  space" [ 17; 1; 7 ] [ (0, 5); (5, 6); (6, 12) ];
  case never "\xe2\x96\x81already" [ 10 ] [ (0, 10) ];
  case never "\xe6\x97\xa5\xe6\x9c\xac \xe8\xaa\x9e" [ 20; 9 ]
    [ (0, 6); (6, 10) ];
  case never "x y" [ 24; 1; 23 ] [ (0, 1); (1, 2); (2, 3) ]

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
             Decoder.fuse;
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

(* Streaming decoding *)

(* The texts a stream gives for [ids], one per step and then the finish. *)
let stream_texts ?skip_special_tokens t ids =
  let texts, s =
    Array.fold_left
      (fun (texts, s) id ->
        let text, s = decode_step s id in
        (text :: texts, s))
      ([], decode_stream ?skip_special_tokens t)
      ids
  in
  List.rev (decode_finish s :: texts)

(* A tokenizer whose ids are the positions of [tokens], decoded by [decoder]. *)
let listed ?decoder tokens =
  word_level ?decoder ~vocab:(List.mapi (fun i token -> (token, i)) tokens) ()

(* Byte-level characters: [â], [Ĥ] and [¬] are the bytes of [€]. *)
let euro = [ "\xc3\xa2"; "\xc4\xa4"; "\xc2\xac" ]

let test_stream_cases () =
  let case msg t ids expected =
    equal ~msg (list string) expected (stream_texts t (Array.of_list ids))
  in
  case "a character waits for its last byte"
    (listed ~decoder:Decoder.byte_level ("a" :: euro))
    [ 0; 1; 2; 3; 0 ] [ "a"; ""; ""; "\xe2\x82\xac"; "a"; "" ];
  case "a character cut short ends the text"
    (listed ~decoder:Decoder.byte_level ("a" :: euro))
    [ 0; 1; 2 ] [ "a"; ""; ""; "\xef\xbf\xbd" ];
  (* The text of a run of byte tokens is known once the run ends: a byte that
     breaks the UTF-8 turns every byte of the run into a replacement
     character. *)
  let fallback =
    listed ~decoder:Decoder.byte_fallback
      [ "<0x20>"; "<0xC3>"; "<0xA9>"; " This"; "<0xFF>" ]
  in
  case "a run of byte tokens waits for its end" fallback [ 0; 1; 2; 3 ]
    [ ""; ""; ""; " \xc3\xa9 This"; "" ];
  case "a run that is not UTF-8" fallback [ 0; 4; 3 ]
    [ ""; ""; "\xef\xbf\xbd\xef\xbf\xbd This"; "" ];
  case "a run at the end" fallback [ 3; 1; 2 ] [ " This"; ""; ""; "\xc3\xa9" ];
  (* The marker that stands for nothing on the first token stands for a space
     on the others. *)
  case "the first token keeps its own text"
    (listed ~decoder:(Decoder.metaspace ()) [ "\xe2\x96\x81This" ])
    [ 0; 0 ] [ "This"; " This"; "" ];
  (* A suffix is a space unless its token is the last. *)
  case "the last token waits for the next"
    (listed ~decoder:(Decoder.bpe ()) [ "a</w>"; "b</w>" ])
    [ 0; 1 ] [ ""; "a "; "b" ];
  case "a token without the suffix is given at once"
    (listed ~decoder:(Decoder.bpe ()) [ "a"; "b</w>" ])
    [ 0; 1; 0; 1 ] [ "a"; ""; "b a"; ""; "b" ];
  (* [ÿ] and [Ģ] are the bytes [\xff] and [\x80], which no later byte makes
     part of a character. *)
  case "a byte no character starts with is replaced at once"
    (listed ~decoder:Decoder.byte_level [ "a"; "\xc3\xbf"; "\xc4\xa2" ])
    [ 0; 1; 2; 0 ]
    [ "a"; "\xef\xbf\xbd"; "\xef\xbf\xbd"; "a"; "" ];
  case "a character's start that a later byte breaks"
    (listed ~decoder:Decoder.byte_level ("a" :: euro))
    [ 1; 0 ] [ ""; "\xef\xbf\xbda"; "" ];
  let long = 10_000 in
  case "a long run of byte tokens"
    (listed ~decoder:Decoder.byte_fallback [ "<0x41>"; "b" ])
    (List.init long (fun _ -> 0) @ [ 1 ])
    (List.init long (fun _ -> "") @ [ String.make long 'A' ^ "b"; "" ]);
  (* After a collapse, a stage keeps what a later piece may change. *)
  let after decoders tokens =
    listed ~decoder:(Decoder.sequence decoders) tokens
  in
  case "the start of a match"
    (after [ Decoder.fuse; Decoder.replace ~pattern:"ab" ~by:"x" () ]
       [ "a"; "b"; "c" ])
    [ 0; 1; 0; 2 ] [ ""; "x"; ""; "ac"; "" ];
  case "the occurrences an end may cut"
    (after [ Decoder.fuse; Decoder.strip ~stop:2 () ] [ "a"; " "; "b" ])
    [ 0; 1; 1; 1; 2; 1 ] [ "a"; ""; ""; " "; "  b"; ""; "" ];
  case "the occurrences the start may cut"
    (after [ Decoder.fuse; Decoder.strip ~start:2 () ] [ "a"; " " ])
    [ 1; 1; 1; 0 ] [ ""; ""; " "; "a"; "" ];
  case "a decoder that reads its text whole"
    (after [ Decoder.fuse; Decoder.byte_level ] [ "a"; "b" ])
    [ 0; 1 ] [ ""; ""; "ab" ];
  (* With no decoder, the tokens are joined with spaces. HuggingFace
     [tokenizers] 0.23.1 decodes [0; 1; 0] of this BPE model to ["a b a"]. *)
  case "no decoder joins the tokens with spaces"
    (bpe ~vocab:[ ("a", 0); ("b", 1) ] ~merges:[] ())
    [ 0; 1; 0 ] [ "a"; " b"; " a"; "" ];
  equal ~msg:"decode with no decoder" string "a b a"
    (decode (bpe ~vocab:[ ("a", 0); ("b", 1) ] ~merges:[] ()) [| 0; 1; 0 |]);
  case "no decoder joins a word-level model's tokens with spaces"
    (listed [ "a"; "b" ]) [ 0; 1; 0 ] [ "a"; " b"; " a"; "" ];
  case "an id outside the vocabulary gives nothing" (listed [ "a" ]) [ 0; 7; 0 ]
    [ "a"; ""; " a"; "" ]

let test_stream_skips_special_tokens () =
  let t =
    word_level
      ~vocab:[ ("a", 0); ("b", 1) ]
      ~added_tokens:[ added_token "<s>" ]
      ()
  in
  let s = Option.get (token_to_id t "<s>") in
  equal ~msg:"skipped" (list string) [ "a"; ""; " b"; "" ]
    (stream_texts ~skip_special_tokens:true t [| 0; s; 1 |]);
  equal ~msg:"kept" (list string) [ "a"; " <s>"; " b"; "" ]
    (stream_texts t [| 0; s; 1 |])

(* Decoders that reach every case a stream keeps text back for: an open run of
   byte tokens, a character not yet complete, a last token, and, after a
   collapse, the start of a match and the ends a strip may cut. *)
let stream_decoders =
  [
    ("bpe", Decoder.bpe ());
    ("bpe a", Decoder.bpe ~suffix:"a" ());
    ("byte_level", Decoder.byte_level);
    ("byte_fallback", Decoder.byte_fallback);
    ("wordpiece", Decoder.wordpiece ());
    ("wordpiece no cleanup", Decoder.wordpiece ~cleanup:false ());
    ("metaspace", Decoder.metaspace ());
    ("metaspace never", Decoder.metaspace ~prepend_scheme:`Never ());
    ("ctc", Decoder.ctc ());
    ("ctc no cleanup", Decoder.ctc ~cleanup:false ());
    ("replace marker", Decoder.replace ~pattern:"\xe2\x96\x81" ~by:" " ());
    ("replace ab", Decoder.replace ~pattern:"ab" ~by:"x" ());
    ("replace aa", Decoder.replace ~pattern:"aa" ~by:"a" ());
    ("strip start", Decoder.strip ~start:1 ());
    ("strip stop", Decoder.strip ~stop:2 ());
    ("strip ab", Decoder.strip ~content:"ab" ~start:1 ~stop:1 ());
    ("strip marker", Decoder.strip ~content:"\xe2\x96\x81" ~stop:1 ());
    ("fuse", Decoder.fuse);
  ]

let stream_tokens =
  [
    "a"; "b"; "ab"; " "; "  "; "\xe2\x96\x81"; "\xe2\x96\x81a";
    "a\xe2\x96\x81"; "##a"; "a</w>"; "</w>"; "<0x41>"; "<0xC3>"; "<0xA9>";
    "<0xE2>"; "<0x82>"; "<0xAC>"; "<0xFF>"; "<pad>"; "|"; "."; " ."; "do";
    " not"; "'"; "n't"; "\xc4\xa0"; "\xc3\x83"; "\xc2\xa9"; "\xc3\xbf";
    "\xc4\xa2"; "é";
  ]
  @ euro

let collapses = [ "byte_level"; "fuse" ]

(* A sequence of decoders, most often with a collapse between two lists, so
   that the decoders after it rewrite text arriving piece by piece. *)
let stream_case =
  let open Gen in
  let leaf =
    of_list ~pp:Format.pp_print_string (List.map fst stream_decoders)
  in
  let+ before = list ~size:(int_range 0 2) leaf
  and+ collapse = of_list [ []; [ "byte_level" ]; [ "fuse" ]; [ "fuse" ] ]
  and+ after = list ~size:(int_range 0 2) leaf
  and+ ids =
    list ~size:(int_range 0 12) (int_range 0 (List.length stream_tokens - 1))
  in
  (before @ collapse @ after, ids)

let stream_case =
  let pp ppf (names, ids) =
    Format.fprintf ppf "[%s], [%s]" (String.concat "; " names)
      (String.concat "; " (List.map string_of_int ids))
  in
  Gen.with_pp pp stream_case

(* Whether after some step the texts given so far differ from [decode] of the
   ids fed so far, which is the stream keeping text back. *)
let kept ?skip_special_tokens t ids texts =
  let texts = Array.of_list texts in
  let rec from i given =
    i < Array.length ids
    &&
    let given = given ^ texts.(i) in
    given <> decode t ?skip_special_tokens (Array.sub ids 0 (i + 1))
    || from (i + 1) given
  in
  from 0 ""

let decoder_stream_law =
  prop ~count:2000 "a stream gives the text of decode" stream_case
    (fun (names, ids) ->
      let decoder =
        Decoder.sequence
          (List.map (fun name -> List.assoc name stream_decoders) names)
      in
      let t = listed ~decoder stream_tokens in
      let ids = Array.of_list ids in
      let texts = stream_texts t ids in
      let rec after_collapse prefix = function
        | name :: rest when List.mem name collapses ->
            List.exists (String.starts_with ~prefix) rest
        | _ :: rest -> after_collapse prefix rest
        | [] -> false
      in
      cover ~share:0.1 "a strip after a collapse"
        (after_collapse "strip" names);
      cover ~share:0.1 "a replace after a collapse"
        (after_collapse "replace" names);
      cover ~share:0.2 "text kept for a later token" (kept t ids texts);
      List.iter
        (fun text ->
          equal ~msg:(Printf.sprintf "%S is UTF-8" text) bool true
            (String.is_valid_utf_8 text))
        texts;
      equal
        ~msg:(String.concat ", " names)
        string (decode t ids) (String.concat "" texts))

(* Training. Every trainer counts the pre-tokens the pipeline hands the model,
   which is what the [tokenizers] wheel counts; the words below were probed with
   it. The unigram vocabulary is brot's own: its trainer keeps the most frequent
   words rather than running the EM training the wheel runs. *)

let train_corpus =
  [
    "low low low low low lower lower";
    "newest newest newest newest newest newest";
    "widest widest widest lowest";
  ]

let sorted_vocab tokenizer =
  vocab tokenizer |> List.map fst |> List.sort String.compare

let test_train_wordpiece_words () =
  let tokenizer =
    train_wordpiece ~pre:Pre_tokenizer.whitespace_split ~vocab_size:30
      (`Seq (List.to_seq train_corpus))
  in
  equal ~msg:"a trained word is one token, an unseen one is split" (list string)
    [ "newest"; "low"; "##est" ]
    (encode tokenizer "newest lowest" |> Encoding.tokens |> Array.to_list)

let test_train_word_level_words () =
  let corpus = `Seq (List.to_seq [ "a a a b b c" ]) in
  let encoded =
    train_word_level
      ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
      ~vocab_size:20 corpus
  in
  equal ~msg:"the words are the byte-level pieces" (list string)
    [ "a"; "Ġa"; "Ġb"; "Ġc" ]
    (sorted_vocab encoded);
  let whole = train_word_level ~vocab_size:20 corpus in
  equal ~msg:"with no pre-tokenizer a text is one word" (list string)
    [ "a a a b b c" ] (sorted_vocab whole)

let test_train_word_level_specials () =
  (* The special tokens take the first ids and count against [vocab_size], so
     the words are numbered after them and one of the three drops out. *)
  let tokenizer =
    train_word_level ~pre:Pre_tokenizer.whitespace_split
      ~added_tokens:(List.map added_token [ "[UNK]"; "[CLS]" ])
      ~vocab_size:4
      (`Seq (List.to_seq [ "a a a b b c" ]))
  in
  equal ~msg:"vocabulary"
    (list (pair string int))
    [ ("[UNK]", 0); ("[CLS]", 1); ("a", 2); ("b", 3) ]
    (vocab tokenizer |> List.sort (fun (_, i) (_, j) -> compare i j))

let test_train_line_separators () =
  (* A line of a corpus file keeps the newline that ends it: a CRLF line keeps
     both bytes, a blank line is a ["\n"] word of its own, and a last line
     without a newline keeps none. *)
  let path = Filename.temp_file "brot_corpus" ".txt" in
  let oc = open_out_bin path in
  output_string oc "a b\nc d\r\n\ne f";
  close_out oc;
  let tokenizer =
    Fun.protect
      ~finally:(fun () -> Sys.remove path)
      (fun () -> train_word_level ~vocab_size:20 (`Files [ path ]))
  in
  equal ~msg:"the words are the lines, separators and all" (list string)
    [ "\n"; "a b\n"; "c d\r\n"; "e f" ]
    (sorted_vocab tokenizer)

let test_train_unigram_words () =
  let tokenizer =
    train_unigram
      ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
      ~vocab_size:20
      (`Seq (List.to_seq [ "low lower low" ]))
  in
  equal ~msg:"the words are the byte-level pieces" (list string)
    [ "low"; "Ġlow"; "Ġlower" ]
    (sorted_vocab tokenizer)

(* A pre-tokenizer that ends in a byte-level one hands the model pieces it has
   already encoded, so the model must match its vocabulary as it is written
   rather than against raw bytes. Encoding them a second time turned [Ġa] into
   [Ä]+[Å]. Read off HuggingFace [tokenizers] 0.23.1 with the same vocabulary
   and merges. *)
let test_byte_level_after_a_split () =
  let tokenizer =
    bpe
      ~pre:
        (Pre_tokenizer.sequence
           [
             Pre_tokenizer.whitespace_split;
             Pre_tokenizer.byte_level ~add_prefix_space:true ();
           ])
      ~vocab:[ ("Ġa", 0); ("Ġb", 1); ("Ġ", 2); ("a", 3); ("b", 4) ]
      ~merges:[ ("Ġ", "a"); ("Ġ", "b") ]
      ()
  in
  equal ~msg:"ids" (array int) [| 0; 1 |]
    (Encoding.ids (encode tokenizer ~add_special_tokens:false "a b"));
  equal ~msg:"tokens" (array string) [| "Ġa"; "Ġb" |]
    (Encoding.tokens (encode tokenizer ~add_special_tokens:false "a b"));
  equal ~msg:"encode_ids agrees" (array int) [| 1; 0 |]
    (encode_ids tokenizer ~add_special_tokens:false "b a")

(* Batch encoding *)

(* The parity corpora, as the documents the parity gate encodes them as. *)
let parity_documents corpus =
  let text =
    Fixture.read (Filename.concat "support/parity" (corpus ^ ".txt"))
  in
  let lines = String.split_on_char '\n' text in
  let join current = String.concat "\n" (List.rev current) in
  let rec split docs current = function
    | [] -> List.rev (join current :: docs)
    | "====" :: rest -> split (join current :: docs) [] rest
    | line :: rest -> split docs (line :: current) rest
  in
  split [] [] lines

let with_pretrained model f =
  Fixture.with_tokenizer_file model (fun path ->
      match from_file path with
      | Ok tokenizer -> f tokenizer
      | Error msg -> failf "failed to load %s: %s" path msg)

(* The pretrained tokenizers name no pad token, so the padding names one. *)
let pad ?direction length =
  padding ?direction ~pad_id:0 ~pad_token:"<pad>" length

(* The rows of a flat batch buffer, read back through the lengths that place
   them. Checks that the buffer holds exactly the rows and nothing else. *)
let rows (buffer, lengths) =
  let total = Array.fold_left ( + ) 0 lengths in
  equal ~msg:"buffer holds the rows and nothing else" int total
    (Bigarray.Array1.dim buffer);
  let at = ref 0 in
  Array.to_list
    (Array.map
       (fun length ->
         let row =
           Array.init length (fun k ->
               Int32.to_int (Bigarray.Array1.get buffer (!at + k)))
         in
         at := !at + length;
         row)
       lengths)

let check_batch ?add_special_tokens ?padding ?truncation ?exec ~msg tokenizer
    documents =
  equal ~msg
    (list (array int))
    (List.map
       (encode_ids tokenizer ?add_special_tokens ?padding ?truncation)
       documents)
    (rows
       (encode_batch_ids tokenizer ?add_special_tokens ?padding ?truncation
          ?exec documents))

let test_batch_ids_matches_encode_ids model () =
  with_pretrained model (fun tokenizer ->
      let documents =
        parity_documents "sample" @ parity_documents "edge_cases"
      in
      check_batch ~msg:"plain" tokenizer documents;
      check_batch ~msg:"padded to a fixed length"
        ~padding:(pad (`Fixed 48))
        tokenizer documents;
      check_batch ~msg:"padded to a multiple, on the left"
        ~padding:(pad ~direction:`Left (`To_multiple 8))
        tokenizer documents;
      check_batch ~msg:"truncated" ~truncation:(truncation 24) tokenizer
        documents;
      check_batch ~msg:"truncated from the left"
        ~truncation:(truncation ~direction:`Left 24)
        tokenizer documents;
      check_batch ~msg:"truncated and padded" ~truncation:(truncation 24)
        ~padding:(pad (`Fixed 24))
        tokenizer documents;
      check_batch ~msg:"without special tokens" ~add_special_tokens:false
        ~padding:(pad (`Fixed 32))
        tokenizer documents)

(* [`Batch_longest] takes the batch as its unit, which is the one thing a row
   does not decide on its own. *)
let test_batch_ids_pads_to_the_longest () =
  with_pretrained "gpt2" (fun tokenizer ->
      let documents = parity_documents "sample" in
      let padding = pad `Batch_longest in
      let expected =
        List.map Encoding.ids (encode_batch tokenizer ~padding documents)
      in
      equal ~msg:"rows"
        (list (array int))
        expected
        (rows (encode_batch_ids tokenizer ~padding documents)))

let test_batch_ids_one_domain () =
  with_pretrained "gpt2" (fun tokenizer ->
      let documents = parity_documents "sample" in
      equal ~msg:"the calling domain and four agree"
        (list (array int))
        (rows (encode_batch_ids tokenizer documents))
        (rows (encode_batch_ids tokenizer ~exec:(Fixture.spawn 4) documents)))

(* A batch encoded while another holds the calling domain's batch buffer — from
   inside the other's executor, between its chunks — encodes into buffers of
   its own: neither overwrites the ids of the other. The outer batch is large
   enough for several chunks, so that its executor runs. *)
let test_batch_ids_nested () =
  with_pretrained "gpt2" (fun tokenizer ->
      let sample = parity_documents "sample" in
      let inner = parity_documents "edge_cases" in
      let bytes docs = List.fold_left (fun n d -> n + String.length d) 0 docs in
      let copies = 1 + (2_000_000 / max 1 (bytes sample)) in
      let outer = List.concat (List.init copies (fun _ -> sample)) in
      let inner_rows = ref [] and tasks = ref 0 in
      let exec n task =
        for i = 0 to n - 1 do
          task i;
          incr tasks;
          inner_rows := rows (encode_batch_ids tokenizer inner) :: !inner_rows
        done
      in
      let outer_rows = rows (encode_batch_ids tokenizer ~exec outer) in
      greater ~msg:"chunks the executor ran" int ~than:1 !tasks;
      equal ~msg:"the outer batch"
        (list (array int))
        (List.map (encode_ids tokenizer) outer)
        outer_rows;
      let expected = List.map (encode_ids tokenizer) inner in
      List.iter
        (equal ~msg:"an inner batch" (list (array int)) expected)
        !inner_rows)

(* An executor that returns while a task it started is still encoding: the
   batch fails, and the buffer that task writes into must stay out of the
   batches after. The straggler is held before its first document; the next
   batch, on the calling domain, lets it write once it has read where its own
   first document starts, and goes on only when the straggler is done. *)
let test_batch_ids_straggler () =
  with_pretrained "gpt2" (fun tokenizer ->
      let caller = Domain.self () in
      let inside = Atomic.make false and go = Atomic.make false in
      let finished = Atomic.make false and armed = Atomic.make false in
      let hook () =
        if Domain.self () <> caller then begin
          if not (Atomic.get go) then begin
            Atomic.set inside true;
            while not (Atomic.get go) do
              Domain.cpu_relax ()
            done
          end
        end
        else if Atomic.compare_and_set armed true false then begin
          Atomic.set go true;
          while not (Atomic.get finished) do
            Domain.cpu_relax ()
          done
        end
      in
      let straggler = ref None in
      let exec _ task =
        straggler :=
          Some
            (Domain.spawn (fun () ->
                 task 0;
                 Atomic.set finished true));
        while not (Atomic.get inside) do
          Domain.cpu_relax ()
        done
      in
      let sample = parity_documents "sample" in
      let big =
        List.concat (List.init 40 (fun _ -> sample))
        |> List.map (fun d -> String.concat " " (List.init 20 (fun _ -> d)))
      in
      let small = parity_documents "edge_cases" in
      Fun.protect
        ~finally:(fun () ->
          Atomic.set Brot.batch_document_hook (fun () -> ());
          Atomic.set go true;
          Option.iter Domain.join !straggler)
        (fun () ->
          Atomic.set Brot.batch_document_hook hook;
          raises_match ~msg:"the batch whose executor returned early"
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> encode_batch_ids tokenizer ~exec big);
          Atomic.set armed true;
          let later = rows (encode_batch_ids tokenizer small) in
          equal ~msg:"the batch after" (list (array int))
            (List.map (encode_ids tokenizer) small)
            later))

let test_batch_ids_empty () =
  with_pretrained "gpt2" (fun tokenizer ->
      let buffer, lengths = encode_batch_ids tokenizer [] in
      equal ~msg:"no rows" (array int) [||] lengths;
      equal ~msg:"no ids" int 0 (Bigarray.Array1.dim buffer);
      check_batch ~msg:"empty documents" tokenizer [ ""; "a"; ""; "" ];
      check_batch ~msg:"empty documents, padded"
        ~padding:(pad (`Fixed 4))
        tokenizer [ ""; "a"; "" ])

(* A post-processor that does more than wrap the sequence sends the batch the
   long way, through encodings. *)
let test_batch_ids_without_affixes () =
  let tokenizer =
    word_level
      ~vocab:[ ("the", 0); ("cat", 1); ("sat", 2); ("[SEP]", 3); ("[UNK]", 4) ]
      ~pre:Pre_tokenizer.whitespace ~unk_token:"[UNK]"
      ~post:
        (Post_processor.template ~single:"$A [SEP] $A"
           ~special_tokens:[ ("[SEP]", 3) ]
           ())
      ()
  in
  equal ~msg:"the sequence is named twice" (array int) [| 0; 1; 3; 0; 1 |]
    (encode_ids tokenizer "the cat");
  let documents = [ "the cat sat"; ""; "sat on the mat"; "the" ] in
  check_batch ~msg:"plain" tokenizer documents;
  check_batch ~msg:"truncated and padded"
    ~truncation:(truncation ~direction:`Left 4)
    ~padding:(pad (`Fixed 6))
    tokenizer documents

(* A document past the cutting threshold is encoded in pieces on several
   domains, which may not change an id of it. *)
let repeat text length =
  let buffer = Buffer.create (length + String.length text) in
  while Buffer.length buffer < length do
    Buffer.add_string buffer text
  done;
  Buffer.contents buffer

let test_batch_ids_cut_document model () =
  with_pretrained model (fun tokenizer ->
      let corpus =
        repeat (String.concat "\n" (parity_documents "sample")) (5 * 1024 * 1024)
      in
      equal ~msg:"the ids of the pieces are the ids of the document"
        (list (array int))
        [ encode_ids tokenizer corpus ]
        (rows (encode_batch_ids tokenizer [ corpus ]));
      let documents = [ "a short one"; corpus; ""; "another short one" ] in
      check_batch ~msg:"among short documents" tokenizer documents;
      check_batch ~msg:"on two domains" ~exec:(Fixture.spawn 2) tokenizer
        documents;
      check_batch ~msg:"among short documents, truncated from the left"
        ~truncation:(truncation ~direction:`Left 64)
        tokenizer documents)

(* The pieces of a cut document pass over the special tokens as the whole
   does when their text is to be encoded as text. *)
let test_batch_ids_cut_unmatched () =
  with_pretrained "gpt2" (fun tokenizer ->
      let corpus =
        repeat
          (String.concat "\n" (parity_documents "edge_cases"))
          (2 * 1024 * 1024)
      in
      let as_text = encode_ids tokenizer ~match_special_tokens:false corpus in
      equal ~msg:"the corpus holds special tokens" bool false
        (as_text = encode_ids tokenizer corpus);
      equal ~msg:"the ids of the pieces are the ids of the document"
        (list (array int))
        [ as_text ]
        (rows
           (encode_batch_ids tokenizer ~match_special_tokens:false [ corpus ])))

(* Added tokens are matched by one scan of the whole document, and a piece
   starting where the whole did not is not the whole: a [single_word] token
   turned down in the whole for the word before it stands alone at the head of a
   piece, and the scan of the whole skips the bytes of a token it turned down,
   which a piece's scan reads. A cut lands only where the pieces cannot tell. *)
let test_batch_ids_cut_avoids_added_tokens () =
  let tokenizer =
    word_level
      ~vocab:
        [
          ("hello", 0);
          ("world", 1);
          ("the", 2);
          ("cat", 3);
          ("sat", 4);
          ("on", 5);
          ("mat", 6);
          ("[UNK]", 7);
        ]
      ~pre:Pre_tokenizer.whitespace ~unk_token:"[UNK]"
      ~added_tokens:
        [
          added_token ~special:false ~single_word:true " world";
          added_token ~special:false ~single_word:true ~normalized:false "n th";
          added_token ~special:false ~normalized:false "th";
        ]
      ()
  in
  equal ~msg:"the whole turns the added tokens down" (array int)
    [| 0; 1; 5; 2 |]
    (encode_ids tokenizer "hello world on the");
  equal ~msg:"a piece would not" (array int) [| 8 |]
    (encode_ids tokenizer " world");
  equal ~msg:"nor read what the whole skipped" (array int) [| 10; 7 |]
    (encode_ids tokenizer " the");
  let mib = 1024 * 1024 in
  let documents =
    [
      repeat "hello world " mib;
      repeat "hello world the cat sat on the mat " (2 * mib);
      "the cat";
      repeat "cat sat on the mat hello world " mib;
    ]
  in
  check_batch ~msg:"cut documents keep their ids" tokenizer documents;
  check_batch ~msg:"on two domains" ~exec:(Fixture.spawn 2) tokenizer
    documents

(* A normalizer may act across a space, so a pipeline with one is never cut,
   however long the document. *)
let test_batch_ids_uncut_under_normalizer () =
  let tokenizer =
    word_level
      ~normalizer:(Normalizer.replace ~pattern:"e c" ~replacement:"ec")
      ~pre:Pre_tokenizer.whitespace
      ~vocab:[ ("the", 0); ("cat", 1); ("thecat", 2); ("[UNK]", 3) ]
      ~unk_token:"[UNK]" ()
  in
  equal ~msg:"the normalizer joins the words" (array int) [| 2; 2 |]
    (encode_ids tokenizer "the cat the cat");
  let mib = 1024 * 1024 in
  let documents =
    List.init 4 (fun j -> String.make j 'x' ^ repeat "the cat the cat " mib)
  in
  check_batch ~msg:"whole documents" tokenizer documents;
  check_batch ~msg:"on two domains" ~exec:(Fixture.spawn 2) tokenizer
    documents

(* A vocabulary of space runs sees a cut inside a run: the guards on the bytes
   either side of the cut keep the run whole. Only a BPE model walks a
   byte-level pre-tokenizer, so only one can be cut. *)
let test_batch_ids_cut_keeps_space_runs () =
  let g = "\u{120}" in
  let tokenizer =
    bpe
      ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
      ~vocab:
        [
          (g, 0);
          (g ^ g, 1);
          (g ^ g ^ g, 2);
          (g ^ "the", 3);
          (g ^ "cat", 4);
          ("the", 5);
          ("x", 6);
          ("t", 7);
          ("h", 8);
          ("e", 9);
          ("c", 10);
          ("a", 11);
          ("th", 12);
          ("ca", 13);
          ("cat", 14);
        ]
      ~merges:
        [
          (g, g);
          (g ^ g, g);
          ("t", "h");
          ("th", "e");
          (g, "the");
          ("c", "a");
          ("ca", "t");
          (g, "cat");
        ]
      ()
  in
  equal ~msg:"space runs are tokens" (array int) [| 5; 1; 4; 0; 3; 0 |]
    (encode_ids tokenizer "the   cat  the ");
  let mib = 1024 * 1024 in
  let documents =
    List.init 5 (fun j -> String.make j 'x' ^ repeat "the   cat  the " mib)
  in
  check_batch ~msg:"cut documents keep their runs" tokenizer documents;
  check_batch ~msg:"on two domains" ~exec:(Fixture.spawn 2) tokenizer
    documents

(* A document a worker cannot encode fails the whole batch, once every domain
   has been joined, and leaves the tokenizer usable. *)
let test_batch_worker_failure () =
  let tokenizer =
    unigram
      ~vocab:
        [
          ("hello", 0.);
          ("world", 0.);
          ("h", -5.);
          ("e", -5.);
          ("l", -5.);
          ("o", -5.);
          ("w", -5.);
          ("r", -5.);
          ("d", -5.);
        ]
      ~pre:Pre_tokenizer.whitespace ()
  in
  let mib = 1024 * 1024 in
  let good = List.init 8 (fun _ -> repeat "hello world " (2 * mib)) in
  let bad = List.concat [ good; [ "hello world zzz" ]; good ] in
  let failure = function Failure _ -> true | _ -> false in
  let exec = Fixture.spawn 6 in
  raises_match ~msg:"encode_batch_ids fails" failure (fun () ->
      encode_batch_ids tokenizer ~exec bad);
  raises_match ~msg:"encode_batch fails" failure (fun () ->
      encode_batch tokenizer ~exec bad);
  check_batch ~msg:"the tokenizer still encodes" ~exec tokenizer good;
  equal ~msg:"encodings too" int (List.length good)
    (List.length (encode_batch tokenizer ~exec good))

(* A chunk raising on a spawned domain stops the batch: the others finish the
   chunk they hold and start no other. The first chunk a spawned domain takes
   raises; every other chunk waits for that, then takes 20 ms, so a domain
   that kept taking chunks would run all 63. Each of the three others holds
   one chunk when it learns of the failure, and a second only if the failure
   took a whole chunk to publish. *)
let test_batch_failure_stops_domains () =
  let caller = Domain.self () in
  let raised = Atomic.make false and done_ = Atomic.make 0 in
  let work _ =
    if Domain.self () <> caller && Atomic.compare_and_set raised false true
    then failwith "chunk"
    else begin
      while not (Atomic.get raised) do
        Domain.cpu_relax ()
      done;
      Unix.sleepf 0.02;
      Atomic.incr done_
    end
  in
  raises ~msg:"the batch fails" (Failure "chunk") (fun () ->
      Brot.run_chunks ~exec:(Fixture.spawn 4) (Array.make 64 1) work);
  at_most ~msg:"chunks run after the failure" int ~than:6 (Atomic.get done_)

(* Executors *)

(* Every task on the calling domain, last first. *)
let sequential n task =
  for i = n - 1 downto 0 do
    task i
  done

(* Every task on one of three domains started for the call, none on the
   caller's. *)
let elsewhere n task =
  let worker j () =
    let i = ref j in
    while !i < n do
      task !i;
      i := !i + 3
    done
  in
  List.iter Domain.join (List.init 3 (fun j -> Domain.spawn (worker j)))

(* A pool in domainslib's shape: domains started once, a [run] that enters the
   pool, and a [parallel_for] that waits for its body over every index. *)
module Pool = struct
  type t = {
    jobs : (unit -> unit) Queue.t;
    lock : Mutex.t;
    changed : Condition.t;
    mutable closing : bool;
    mutable workers : unit Domain.t list;
  }

  let rec work pool =
    Mutex.lock pool.lock;
    while Queue.is_empty pool.jobs && not pool.closing do
      Condition.wait pool.changed pool.lock
    done;
    let job = Queue.take_opt pool.jobs in
    Mutex.unlock pool.lock;
    match job with
    | None -> ()
    | Some job ->
        job ();
        work pool

  let setup ~num_domains =
    let pool =
      {
        jobs = Queue.create ();
        lock = Mutex.create ();
        changed = Condition.create ();
        closing = false;
        workers = [];
      }
    in
    pool.workers <-
      List.init num_domains (fun _ -> Domain.spawn (fun () -> work pool));
    pool

  let teardown pool =
    Mutex.protect pool.lock (fun () ->
        pool.closing <- true;
        Condition.broadcast pool.changed);
    List.iter Domain.join pool.workers

  let run _ f = f ()

  let parallel_for pool ~start ~finish ~body =
    let left = ref (finish - start + 1) in
    let job i () =
      body i;
      Mutex.protect pool.lock (fun () ->
          decr left;
          if !left = 0 then Condition.broadcast pool.changed)
    in
    Mutex.protect pool.lock (fun () ->
        for i = start to finish do
          Queue.push (job i) pool.jobs
        done;
        Condition.broadcast pool.changed;
        while !left > 0 do
          Condition.wait pool.changed pool.lock
        done)
end

(* The adapter brot.mli shows, over the pool above. *)
let pooled pool n task =
  Pool.run pool (fun () ->
      Pool.parallel_for pool ~start:0 ~finish:(n - 1) ~body:task)

(* Long documents that cut into many chunks among short ones, so that every
   task of an executor has chunks to take. *)
let spread_documents () =
  [ repeat "hello world the cat sat on the mat " (256 * 1024); "the cat"; "" ]
  @ List.init 24 (fun j -> repeat "cat sat on the mat hello world " (j * 2048))

let spread_tokenizer () =
  word_level
    ~vocab:
      [
        ("hello", 0);
        ("world", 1);
        ("the", 2);
        ("cat", 3);
        ("sat", 4);
        ("on", 5);
        ("mat", 6);
        ("[UNK]", 7);
        ("[PAD]", 8);
      ]
    ~pre:Pre_tokenizer.whitespace ~unk_token:"[UNK]" ~pad_token:"[PAD]"
    ~post:
      (Post_processor.template ~single:"[UNK] $A"
         ~special_tokens:[ ("[UNK]", 7) ]
         ())
    ()

let test_executors () =
  let tokenizer = spread_tokenizer () and documents = spread_documents () in
  let pool = Pool.setup ~num_domains:3 in
  Fun.protect
    ~finally:(fun () -> Pool.teardown pool)
    (fun () ->
      List.iter
        (fun (name, exec) ->
          check_batch ~msg:name ~exec tokenizer documents;
          check_batch ~msg:(name ^ ", padded and truncated") ~exec
            ~padding:(padding ~direction:`Left (`To_multiple 8))
            ~truncation:(truncation ~direction:`Left 100000)
            tokenizer documents;
          equal ~msg:(name ^ ", encodings")
            (list (array int))
            (List.map Encoding.ids (encode_batch tokenizer documents))
            (List.map Encoding.ids (encode_batch tokenizer ~exec documents)))
        [
          ("sequential, last task first", sequential);
          ("on other domains", elsewhere);
          ("a pool", pooled pool);
          ("the same pool again", pooled pool);
        ])

(* What an executor raises fails the batch, and the tokenizer still encodes. *)
let test_executor_raises () =
  let tokenizer = spread_tokenizer () and documents = spread_documents () in
  let first_then_exit n task =
    if n > 0 then task 0;
    raise Exit
  in
  raises ~msg:"encode_batch_ids" Exit (fun () ->
      encode_batch_ids tokenizer ~exec:first_then_exit documents);
  raises ~msg:"encode_batch" Exit (fun () ->
      encode_batch tokenizer ~exec:first_then_exit documents);
  check_batch ~msg:"afterwards" ~exec:sequential tokenizer documents

(* An executor that returns before its tasks have run leaves chunks unencoded,
   or still being written: the batch fails rather than hand them back. *)
let test_executor_returns_early () =
  let tokenizer = spread_tokenizer () and documents = spread_documents () in
  let refused =
    Invalid_argument "exec returned before every chunk was encoded"
  in
  let nothing _ _ = () in
  raises ~msg:"encode_batch_ids, no task run" refused (fun () ->
      encode_batch_ids tokenizer ~exec:nothing documents);
  raises ~msg:"encode_batch, no task run" refused (fun () ->
      encode_batch tokenizer ~exec:nothing documents);
  check_batch ~msg:"afterwards" tokenizer documents

(* Tasks an executor runs after the batch failed, because it returned early or
   raised, take no chunk. The executor hands its tasks to a domain held until
   the batch has failed. *)
let test_late_tasks_take_nothing () =
  let late outcome =
    let release = Atomic.make false and worked = Atomic.make 0 in
    let held = ref None in
    let exec n task =
      held :=
        Some
          (Domain.spawn (fun () ->
               while not (Atomic.get release) do
                 Domain.cpu_relax ()
               done;
               for i = 0 to n - 1 do
                 task i
               done));
      outcome ()
    in
    let failed =
      match Brot.run_chunks ~exec (Array.make 8 1) (fun _ -> Atomic.incr worked)
      with
      | () -> false
      | exception (Exit | Invalid_argument _) -> true
    in
    Atomic.set release true;
    Option.iter Domain.join !held;
    (failed, Atomic.get worked)
  in
  equal ~msg:"returned early" (pair bool int) (true, 0) (late (fun () -> ()));
  equal ~msg:"raised" (pair bool int) (true, 0) (late (fun () -> raise Exit))

(* A batch asks for no more tasks than domains a program should run at once,
   however many chunks it holds, and the tasks still take every chunk. *)
let test_task_count_is_bounded () =
  let most = Domain.recommended_domain_count () in
  let asked = ref 0 and worked = ref 0 in
  let exec n task =
    asked := n;
    sequential n task
  in
  Brot.run_chunks ~exec (Array.make (most + 50) 1) (fun _ -> incr worked);
  equal ~msg:"tasks" int most !asked;
  equal ~msg:"chunks run" int (most + 50) !worked

(* A chunk that raises stops the handout: the tasks after it take no chunk. *)
let test_failure_stops_the_handout () =
  let calls = ref 0 in
  let work _ =
    incr calls;
    if !calls = 3 then failwith "chunk"
  in
  raises ~msg:"the batch fails" (Failure "chunk") (fun () ->
      Brot.run_chunks ~exec:sequential (Array.make 64 1) work);
  equal ~msg:"chunks run" int 3 !calls

(* A tokenizer's encoder states, caches included, live in it: once it is
   dropped, nothing keeps them or its model. A vocabulary entry built at run
   time is finalised with the model that holds it; a BPE state holds no entry,
   so its 8 MB table is seen in the live words instead. *)
let collected tokenizer_of ~vocab =
  let entry = String.concat "" [ "wor"; "ld" ] in
  let freed = ref false in
  Gc.finalise_last (fun () -> freed := true) entry;
  let encode_and_drop () =
    let tokenizer = tokenizer_of (vocab entry) in
    let text = repeat "hello world " (256 * 1024) in
    ignore (encode_ids tokenizer "hello world" : int array);
    ignore
      (encode_batch_ids tokenizer ~exec:(Fixture.spawn 4) [ text; text ]
        : ids * _)
  in
  (Sys.opaque_identity encode_and_drop) ();
  Gc.full_major ();
  !freed

let test_dropped_tokenizer_is_collected () =
  let words entry = [ "[UNK]"; "hello"; entry; "##o" ] in
  equal ~msg:"wordpiece" bool true
    (collected ~vocab:words (fun words ->
         wordpiece
           ~vocab:(List.mapi (fun i w -> (w, i)) words)
           ~pre:Pre_tokenizer.whitespace ()));
  equal ~msg:"unigram" bool true
    (collected ~vocab:words (fun words ->
         unigram
           ~vocab:(List.map (fun w -> (w, -1.)) words)
           ~unk_id:0 ~pre:Pre_tokenizer.whitespace ()));
  let live () =
    Gc.full_major ();
    (Gc.stat ()).live_words
  in
  let before = live () in
  ignore
    (collected
       ~vocab:(fun entry -> [ ("h", 0); ("e", 1); ("he", 2); (entry, 3) ])
       (fun vocab -> bpe ~vocab ~merges:[ ("h", "e") ] ())
      : bool);
  at_most ~msg:"live words after a BPE tokenizer is dropped" int
    ~than:(before + 65536) (live ())

(* A BPE tokenizer builds at most [Domain.recommended_domain_count ()] states,
   each with a cache its vocabulary seeds, however many encodings hold one at
   once: an encoding that finds them all in use does without a cache. Measured
   on one domain by nesting claims three deeper than that; a state's size is
   measured as one claim's. *)
let live_words () =
  Gc.full_major ();
  (Gc.stat ()).live_words

let small_bpe () =
  bpe ~vocab:[ ("h", 0); ("e", 1); ("he", 2) ] ~merges:[ ("h", "e") ] ()

(* Every word of one to three letters is an entry its letters spell, so the
   18278 entries are seeded into the cache, which doubles to 65536 entries,
   2 MB, to hold them. *)
let seeded_bpe () =
  let letters = List.init 26 (fun i -> String.make 1 (Char.chr (97 + i))) in
  let longer words =
    List.concat_map (fun w -> List.map (( ^ ) w) letters) words
  in
  let two = longer letters in
  let vocab = List.mapi (fun i w -> (w, i)) (letters @ two @ longer two) in
  bpe ~vocab ~merges:[] ()

let test_states_are_bounded () =
  let most = Domain.recommended_domain_count () in
  let t = seeded_bpe () in
  let before = live_words () in
  let one = hold_cache t (fun () -> live_words ()) - before in
  let rec nest depth =
    if depth = 0 then live_words ()
    else hold_cache t (fun () -> nest (depth - 1))
  in
  let peak = nest (most + 3) in
  let after = live_words () in
  ignore (Sys.opaque_identity t : Brot.t);
  greater ~msg:"a state holds its 2 MB cache" int ~than:(1 lsl 17) one;
  at_most ~msg:"live words while every claim is held" int
    ~than:(before + (most * one) + 65536)
    peak;
  at_most ~msg:"live words once every claim is given back" int
    ~than:(before + (most * one) + 65536)
    after

(* A state an exception went through is dropped, and its cache freed. *)
let test_raised_state_is_dropped () =
  let t = small_bpe () in
  let before = live_words () in
  raises ~msg:"the exception passes" Exit (fun () ->
      hold_cache t (fun () -> raise Exit));
  let after = live_words () in
  ignore (Sys.opaque_identity t : Brot.t);
  at_most ~msg:"live words after the raise" int ~than:(before + 65536) after

let batch_tests =
  [
    test "a failing chunk stops the other domains"
      test_batch_failure_stops_domains;
    test "batch ids agree with encode_ids (gpt2)"
      (test_batch_ids_matches_encode_ids "gpt2");
    test "batch ids agree with encode_ids (bert)"
      (test_batch_ids_matches_encode_ids "bert_base");
    test "batch ids agree with encode_ids (llama)"
      (test_batch_ids_matches_encode_ids "llama");
    test "batch ids agree with encode_ids (roberta)"
      (test_batch_ids_matches_encode_ids "roberta_base");
    test "batch ids agree with encode_ids (llama3)"
      (test_batch_ids_matches_encode_ids "llama3");
    test "batch ids agree with encode_ids (gpt-oss)"
      (test_batch_ids_matches_encode_ids "gpt_oss");
    test "batch ids pad to the longest of the batch"
      test_batch_ids_pads_to_the_longest;
    test "batch ids do not depend on the executor" test_batch_ids_one_domain;
    test "batch ids of an empty batch" test_batch_ids_empty;
    test "a batch within another's executor" test_batch_ids_nested;
    test "a batch after one whose executor left a task running"
      test_batch_ids_straggler;
    test "batch ids under a post-processor that does not wrap"
      test_batch_ids_without_affixes;
    test "a cut document keeps its ids (gpt2)"
      (test_batch_ids_cut_document "gpt2");
    test "a cut document keeps its ids (roberta)"
      (test_batch_ids_cut_document "roberta_base");
    slow
    @@ test "a cut document keeps its ids (llama3)"
         (test_batch_ids_cut_document "llama3");
    slow
    @@ test "a cut document keeps its ids (gpt-oss)"
         (test_batch_ids_cut_document "gpt_oss");
    test "a cut avoids the added tokens" test_batch_ids_cut_avoids_added_tokens;
    test "a cut document keeps its special tokens as text"
      test_batch_ids_cut_unmatched;
    test "a pipeline with a normalizer is not cut"
      test_batch_ids_uncut_under_normalizer;
    test "a cut keeps space runs whole" test_batch_ids_cut_keeps_space_runs;
    slow @@ test "a worker failure fails the batch" test_batch_worker_failure;
    test "a dropped tokenizer is collected" test_dropped_tokenizer_is_collected;
    test "a tokenizer's states are bounded" test_states_are_bounded;
    test "a state an exception went through is dropped"
      test_raised_state_is_dropped;
    test "executors give the same ids" test_executors;
    test "an executor that raises fails the batch" test_executor_raises;
    test "an executor that returns early fails the batch"
      test_executor_returns_early;
    test "tasks after a failure take no chunk" test_late_tasks_take_nothing;
    test "a failing chunk stops the handout" test_failure_stops_the_handout;
    test "a batch asks for at most a domain per core"
      test_task_count_is_bounded;
  ]

(* Test Suite *)

(* Truncation with a stride, differential against HuggingFace tokenizers 0.23.1
   on the same tokenizer.json: the primary ids and every overflowing window's
   ids must match the wheel's, single and pair, strides 0, 2 and 7. Expected
   rows generated by test/gen/gen_stride_expected.py. *)

let stride_single =
  "The quick brown fox jumps over the lazy dog while seventeen astronauts"

let stride_pair =
  ( "The quick brown fox jumps over the lazy dog",
    "Seventeen astronauts orbit the small green planet tonight" )

let check_stride_cases model cases () =
  with_pretrained model (fun tokenizer ->
      let case ~max_length ~stride ~direction ~pair expected =
        let truncation = truncation ~stride ~direction max_length in
        let text, pair =
          if pair then (fst stride_pair, Some (snd stride_pair))
          else (stride_single, None)
        in
        let encoding = encode tokenizer ?pair ~truncation text in
        equal
          ~msg:
            (Printf.sprintf "%s max=%d stride=%d %s%s" model max_length stride
               (match direction with `Left -> "left" | `Right -> "right")
               (match pair with Some _ -> " pair" | None -> ""))
          (list (array int))
          expected
          (Encoding.ids encoding
          :: List.map Encoding.ids (Encoding.overflowing encoding))
      in
      cases case)

let gpt2_stride_cases case =
  case ~max_length:8 ~stride:0 ~direction:`Right ~pair:false
    [ [| 464; 2068; 7586; 21831; 18045; 625; 262; 16931 |] ];
  case ~max_length:8 ~stride:2 ~direction:`Right ~pair:false
    [ [| 464; 2068; 7586; 21831; 18045; 625; 262; 16931 |] ];
  case ~max_length:12 ~stride:7 ~direction:`Right ~pair:false
    [
      [|
        464; 2068; 7586; 21831; 18045; 625; 262; 16931; 3290; 981; 38741; 26835;
      |];
    ];
  case ~max_length:8 ~stride:0 ~direction:`Right ~pair:true
    [
      [| 464; 2068; 7586; 21831; 4653; 1151; 6429; 26835 |];
      [| 18045; 625; 262; 16931; 4653; 1151; 6429; 26835 |];
      [| 18045; 625; 262; 16931; 13066; 262; 1402; 4077 |];
      [| 464; 2068; 7586; 21831; 13066; 262; 1402; 4077 |];
    ];
  case ~max_length:10 ~stride:2 ~direction:`Right ~pair:true
    [
      [| 464; 2068; 7586; 21831; 18045; 4653; 1151; 6429; 26835; 13066 |];
      [| 21831; 18045; 625; 262; 16931; 4653; 1151; 6429; 26835; 13066 |];
      [| 21831; 18045; 625; 262; 16931; 26835; 13066; 262; 1402; 4077 |];
      [| 21831; 18045; 625; 262; 16931; 1402; 4077; 5440; 9975 |];
      [| 262; 16931; 3290; 4653; 1151; 6429; 26835; 13066 |];
      [| 262; 16931; 3290; 26835; 13066; 262; 1402; 4077 |];
      [| 262; 16931; 3290; 1402; 4077; 5440; 9975 |];
      [| 464; 2068; 7586; 21831; 18045; 26835; 13066; 262; 1402; 4077 |];
      [| 464; 2068; 7586; 21831; 18045; 1402; 4077; 5440; 9975 |];
    ];
  case ~max_length:20 ~stride:7 ~direction:`Right ~pair:true
    [
      [|
        464;
        2068;
        7586;
        21831;
        18045;
        625;
        262;
        16931;
        3290;
        4653;
        1151;
        6429;
        26835;
        13066;
        262;
        1402;
        4077;
        5440;
        9975;
      |];
    ];
  case ~max_length:10 ~stride:2 ~direction:`Left ~pair:true
    [
      [| 18045; 625; 262; 16931; 3290; 262; 1402; 4077; 5440; 9975 |];
      [| 2068; 7586; 21831; 18045; 625; 262; 1402; 4077; 5440; 9975 |];
      [| 2068; 7586; 21831; 18045; 625; 6429; 26835; 13066; 262; 1402 |];
      [| 2068; 7586; 21831; 18045; 625; 4653; 1151; 6429; 26835 |];
      [| 464; 2068; 7586; 262; 1402; 4077; 5440; 9975 |];
      [| 464; 2068; 7586; 6429; 26835; 13066; 262; 1402 |];
      [| 464; 2068; 7586; 4653; 1151; 6429; 26835 |];
      [| 18045; 625; 262; 16931; 3290; 6429; 26835; 13066; 262; 1402 |];
      [| 18045; 625; 262; 16931; 3290; 4653; 1151; 6429; 26835 |];
    ];
  ()

let bert_stride_cases case =
  case ~max_length:8 ~stride:0 ~direction:`Right ~pair:false
    [
      [| 101; 1996; 4248; 2829; 4419; 14523; 2058; 102 |];
      [| 101; 1996; 13971; 102 |];
    ];
  case ~max_length:8 ~stride:2 ~direction:`Right ~pair:false
    [
      [| 101; 1996; 4248; 2829; 4419; 14523; 2058; 102 |];
      [| 101; 14523; 2058; 1996; 13971; 102 |];
    ];
  case ~max_length:12 ~stride:7 ~direction:`Right ~pair:false
    [
      [|
        101; 1996; 4248; 2829; 4419; 14523; 2058; 1996; 13971; 3899; 2096; 102;
      |];
      [| 101; 4419; 14523; 2058; 1996; 13971; 3899; 2096; 9171; 25881; 102 |];
    ];
  case ~max_length:12 ~stride:7 ~direction:`Left ~pair:false
    [
      [|
        101; 2829; 4419; 14523; 2058; 1996; 13971; 3899; 2096; 9171; 25881; 102;
      |];
      [| 101; 1996; 4248; 2829; 4419; 14523; 2058; 1996; 13971; 3899; 102 |];
    ];
  case ~max_length:8 ~stride:0 ~direction:`Right ~pair:true
    [
      [| 101; 1996; 4248; 102; 9171; 25881; 8753; 102 |];
      [| 101; 2829; 4419; 102; 9171; 25881; 8753; 102 |];
      [| 101; 2829; 4419; 102; 1996; 2235; 2665; 102 |];
      [| 101; 2829; 4419; 102; 4774; 3892; 102 |];
      [| 101; 14523; 2058; 102; 9171; 25881; 8753; 102 |];
      [| 101; 14523; 2058; 102; 1996; 2235; 2665; 102 |];
      [| 101; 14523; 2058; 102; 4774; 3892; 102 |];
      [| 101; 1996; 13971; 102; 9171; 25881; 8753; 102 |];
      [| 101; 1996; 13971; 102; 1996; 2235; 2665; 102 |];
      [| 101; 1996; 13971; 102; 4774; 3892; 102 |];
      [| 101; 1996; 4248; 102; 1996; 2235; 2665; 102 |];
      [| 101; 1996; 4248; 102; 4774; 3892; 102 |];
    ];
  case ~max_length:12 ~stride:2 ~direction:`Right ~pair:true
    [
      [|
        101; 1996; 4248; 2829; 4419; 14523; 102; 9171; 25881; 8753; 1996; 102;
      |];
      [|
        101; 4419; 14523; 2058; 1996; 13971; 102; 9171; 25881; 8753; 1996; 102;
      |];
      [|
        101; 4419; 14523; 2058; 1996; 13971; 102; 8753; 1996; 2235; 2665; 102;
      |];
      [|
        101; 4419; 14523; 2058; 1996; 13971; 102; 2235; 2665; 4774; 3892; 102;
      |];
      [| 101; 1996; 13971; 3899; 102; 9171; 25881; 8753; 1996; 102 |];
      [| 101; 1996; 13971; 3899; 102; 8753; 1996; 2235; 2665; 102 |];
      [| 101; 1996; 13971; 3899; 102; 2235; 2665; 4774; 3892; 102 |];
      [| 101; 1996; 4248; 2829; 4419; 14523; 102; 8753; 1996; 2235; 2665; 102 |];
      [| 101; 1996; 4248; 2829; 4419; 14523; 102; 2235; 2665; 4774; 3892; 102 |];
    ];
  case ~max_length:20 ~stride:7 ~direction:`Right ~pair:true
    [
      [|
        101;
        1996;
        4248;
        2829;
        4419;
        14523;
        2058;
        1996;
        13971;
        3899;
        102;
        9171;
        25881;
        8753;
        1996;
        2235;
        2665;
        4774;
        3892;
        102;
      |];
    ];
  ()

let stride_tests =
  [
    test "gpt2 windows match the wheel"
      (check_stride_cases "gpt2" gpt2_stride_cases);
    test "bert windows match the wheel"
      (check_stride_cases "bert_base" bert_stride_cases);
  ]

(* SentencePiece word-unit sub-cut. A pipeline with no splitting pre-tokenizer
   hands the BPE model whole documents; when no vocabulary piece crosses a
   ▁-opened word boundary the model cuts them into word units and merges each
   alone, which must be invisible: ids, offsets and tokens all match the
   whole-document merge. *)

let sp = "\xe2\x96\x81"

(* The fixture plus one crossing vocabulary piece that no merge builds: its only
   effect is to turn the sub-cut off, so the pair encodes through the two paths
   of the same model. *)
let sp_add_crossing_piece root piece id =
  let open Jsont in
  let in_object name f = function
    | Object (mems, meta) ->
        Object
          ( List.map
              (fun ((k, km), v) -> ((k, km), if k = name then f v else v))
              mems,
            meta )
    | _ -> failf "expected a JSON object around %S" name
  in
  in_object "model"
    (in_object "vocab" (function
      | Object (mems, meta) ->
          Object (mems @ [ (Json.name piece, Json.int id) ], meta)
      | _ -> failf "expected the vocabulary to be a JSON object"))
    root

let sp_llama_pair f =
  Fixture.with_tokenizer_file "llama" (fun path ->
      let text = In_channel.with_open_bin path In_channel.input_all in
      let json =
        match Jsont_bytesrw.decode_string Jsont.json text with
        | Ok json -> json
        | Error msg -> failf "llama.json: %s" msg
      in
      let load json =
        match of_json json with
        | Ok tokenizer -> tokenizer
        | Error msg -> failf "llama tokenizer: %s" msg
      in
      let on = load json in
      let max_id = List.fold_left (fun m (_, id) -> max m id) (-1) (vocab on) in
      let off =
        load (sp_add_crossing_piece json ("zz" ^ sp ^ "zz") (max_id + 1))
      in
      f on off)

let sp_bait_docs =
  [
    "";
    " ";
    "hello world";
    "  leading and   trailing  ";
    "a" ^ sp ^ "b";
    sp ^ sp ^ sp;
    "tabs\tand\nnewlines\r\nCRLF";
    "nbsp\xc2\xa0and\xc2\xa0runs";
    "caf\xc3\xa9 na\xc3\xafve \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e";
    "emoji \xf0\x9f\x98\x80\xf0\x9f\xa7\xa0 fallback";
    "<s> specials </s> inline";
    "punct.,\");:!? runs ... !!! ??? end.";
    "digits 1234567890 mix3d w0rds";
    String.concat " " (List.init 40 (fun i -> Printf.sprintf "w%d" i));
    String.make 5000 'a';
    String.concat "" (List.init 30 (fun _ -> "    indent"));
    String.concat "." (List.init 12 (fun _ -> "ab"));
    "so, it says: \"yes\" (twice); really?! yes... end.";
    ".,\");:!?.,\");:!?" ^ String.concat "," (List.init 8 (fun _ -> "1"));
  ]

let sp_random_docs n =
  let st = Random.State.make [| 0x7352 |] in
  let doc () =
    let len = 1 + Random.State.int st 400 in
    let b = Buffer.create (len * 2) in
    for _ = 1 to len do
      match Random.State.int st 100 with
      | 0 | 1 | 2 -> Buffer.add_string b sp
      | 3 | 4 -> Buffer.add_string b "  "
      | 5 -> Buffer.add_string b "    "
      | 6 -> Buffer.add_char b '\t'
      | 7 -> Buffer.add_char b '\n'
      | 8 -> Buffer.add_string b "\xc3\xa9"
      | 9 -> Buffer.add_string b "\xe6\x97\xa5"
      | 10 -> Buffer.add_string b "\xf0\x9f\x98\x80"
      | 11 -> Buffer.add_string b "\xd0\xb6"
      | 12 -> Buffer.add_char b (Char.chr (1 + Random.State.int st 31))
      | 13 -> Buffer.add_string b "<s>"
      | 14 -> Buffer.add_string b "</s>"
      | k when k < 40 -> Buffer.add_char b ' '
      | k when k < 55 ->
          Buffer.add_char b (Char.chr (Char.code '0' + Random.State.int st 10))
      | k when k < 70 ->
          Buffer.add_char b ".,\"();:!?-'".[Random.State.int st 11]
      | _ ->
          Buffer.add_char b (Char.chr (Char.code 'a' + Random.State.int st 26))
    done;
    Buffer.contents b
  in
  List.init n (fun _ -> doc ())

let test_sp_subcut_differential () =
  sp_llama_pair (fun on off ->
      List.iteri
        (fun i doc ->
          let a = encode on ~add_special_tokens:false doc in
          let b = encode off ~add_special_tokens:false doc in
          let msg what = Printf.sprintf "doc %d %s" i what in
          equal ~msg:(msg "ids") (array int) (Encoding.ids b) (Encoding.ids a);
          equal ~msg:(msg "offsets")
            (array (pair int int))
            (Encoding.offsets b) (Encoding.offsets a);
          equal ~msg:(msg "tokens") (array string) (Encoding.tokens b)
            (Encoding.tokens a))
        (sp_bait_docs @ sp_random_docs 300))

(* ▁-runs must stay with the word they open: a cut at every mark would keep [▁▁]
   from ever forming. The document is long enough for the sub-cut to walk it,
   and the vocabulary has no crossing piece, so it is on. *)
let test_sp_mark_runs () =
  let vocab = [ ("a", 0); ("b", 1); (sp, 2); (sp ^ sp, 3) ] in
  let t = bpe ~vocab ~merges:[ (sp, sp) ] () in
  let doc = String.concat "" (List.init 5 (fun _ -> "a" ^ sp ^ sp ^ "b")) in
  let expected =
    Array.concat (List.init 5 (fun _ -> [| "a"; sp ^ sp; "b" |]))
  in
  equal ~msg:"tokens" (array string) expected (Encoding.tokens (encode t doc))

(* One vocabulary piece with ▁ after a non-▁ character makes the cut unsafe
   (gemma-3's [>▁</] is the live case): the model must fall back to whole
   merges. The merges here build the crossing piece, so a wrongly enabled
   sub-cut could never assemble [a▁b] across its unit boundary. *)
let test_sp_crossing_fallback () =
  let vocab =
    [
      ("z", 0); ("a", 1); ("b", 2); (sp, 3); ("a" ^ sp, 4); ("a" ^ sp ^ "b", 5);
    ]
  in
  let t = bpe ~vocab ~merges:[ ("a", sp); ("a" ^ sp, "b") ] () in
  let doc = String.concat "" (List.init 4 (fun _ -> "za" ^ sp ^ "b")) in
  let expected =
    Array.concat (List.init 4 (fun _ -> [| "z"; "a" ^ sp ^ "b" |]))
  in
  equal ~msg:"tokens" (array string) expected (Encoding.tokens (encode t doc))

(* Under [fuse_unk] with no ▁ piece and no byte fallback, an unknown run can
   contain a ▁: cutting there would split one fused unknown into two, so the
   sub-cut must stay off. *)
let test_sp_fuse_unk_gate () =
  let vocab = [ ("a", 0); ("<u>", 1) ] in
  let t = bpe ~vocab ~merges:[] ~unk_token:"<u>" ~fuse_unk:true () in
  let doc = "aaaaa" ^ "xx" ^ sp ^ "xx" ^ "aaaaa" in
  let expected =
    [| "a"; "a"; "a"; "a"; "a"; "<u>"; "a"; "a"; "a"; "a"; "a" |]
  in
  equal ~msg:"tokens" (array string) expected (Encoding.tokens (encode t doc))

(* A unit may end before a punctuation byte only after a byte that never
   precedes it inside any vocabulary piece: [a.] is a piece here, so cutting
   between [a] and [.] would keep it from ever merging. *)
let test_sp_punct_adjacency () =
  let vocab = [ ("b", 0); ("a", 1); (".", 2); ("a.", 3) ] in
  let t = bpe ~vocab ~merges:[ ("a", ".") ] () in
  let doc = String.concat "" (List.init 6 (fun _ -> "ba.")) in
  let expected = Array.concat (List.init 6 (fun _ -> [| "b"; "a." |])) in
  equal ~msg:"tokens" (array string) expected (Encoding.tokens (encode t doc))

(* Under a reachable unknown token a split byte with no vocabulary piece loses
   its slot: [.] here can sit inside a fused unknown run, which a cut would
   split in two. ▁ keeps its boundary — it is a piece. *)
let test_sp_punct_unk_gate () =
  let vocab = [ ("a", 0); (sp, 1); ("<u>", 2) ] in
  let t = bpe ~vocab ~merges:[] ~unk_token:"<u>" ~fuse_unk:true () in
  let doc = "aaaa" ^ "..." ^ "aaaa" ^ sp ^ "aaaa" in
  let expected =
    [| "a"; "a"; "a"; "a"; "<u>"; "a"; "a"; "a"; "a"; sp; "a"; "a"; "a"; "a" |]
  in
  equal ~msg:"tokens" (array string) expected (Encoding.tokens (encode t doc))

(* A byte-fallback token covers the byte its name only spells, so a merge over
   one can cross a unit boundary neither vocabulary scan can see: any such merge
   disables the sub-cut. Here [a<0xE2>] must swallow the ▁'s lead byte across
   what would be a unit boundary. *)
let test_sp_opaque_merge_fallback () =
  let vocab =
    [ ("a", 0); ("<0xE2>", 1); ("<0x96>", 2); ("<0x81>", 3); ("a<0xE2>", 4) ]
  in
  let t = bpe ~vocab ~merges:[ ("a", "<0xE2>") ] ~byte_fallback:true () in
  let doc = String.make 16 'a' ^ sp ^ "aaaa" in
  let expected =
    Array.concat
      [
        Array.make 15 "a"; [| "a<0xE2>"; "<0x96>"; "<0x81>" |]; Array.make 4 "a";
      ]
  in
  equal ~msg:"tokens" (array string) expected (Encoding.tokens (encode t doc))

(* Affixes give a character the form its place in the word dictates, so a cut
   would hand each unit a word-first character of its own: the sub-cut must stay
   off. The [.] mid-word must come out [##.], never bare [.]. *)
let test_sp_affix_gate () =
  let vocab = [ ("a", 0); ("##a", 1); (".", 2); ("##.", 3) ] in
  let t = bpe ~vocab ~merges:[] ~continuing_subword_prefix:"##" () in
  let doc = String.make 16 'a' ^ "." ^ "aaaa" in
  let expected =
    Array.concat
      [ [| "a" |]; Array.make 15 "##a"; [| "##." |]; Array.make 4 "##a" ]
  in
  equal ~msg:"tokens" (array string) expected (Encoding.tokens (encode t doc))

(* [ignore_merges] emits a word that is itself a vocabulary entry as that one
   token, and a unit is not the word: the sub-cut must stay off. The unit [▁ab]
   is an entry here; the whole span is not. *)
let test_sp_ignore_merges_gate () =
  let vocab = [ ("a", 0); ("b", 1); (sp, 2); (sp ^ "ab", 3) ] in
  let t = bpe ~vocab ~merges:[] ~ignore_merges:true () in
  let doc = String.make 16 'a' ^ sp ^ "ab" in
  let expected = Array.concat [ Array.make 16 "a"; [| sp; "a"; "b" |] ] in
  equal ~msg:"tokens" (array string) expected (Encoding.tokens (encode t doc))

(* The punctuation face of the opaque-merge hole: the adjacency scan reads
   names, and [<0x2E><0x2E>]'s name never puts [.] beside [.], so it would
   certify the split the merge forbids. *)
let test_sp_opaque_merge_punct () =
  let vocab = [ ("a", 0); ("<0x2E>", 1); ("<0x2E><0x2E>", 2) ] in
  let t = bpe ~vocab ~merges:[ ("<0x2E>", "<0x2E>") ] ~byte_fallback:true () in
  let doc = String.make 16 'a' ^ ".." ^ "aaaa" in
  let expected =
    Array.concat [ Array.make 16 "a"; [| "<0x2E><0x2E>" |]; Array.make 4 "a" ]
  in
  equal ~msg:"tokens" (array string) expected (Encoding.tokens (encode t doc))

(* Enablement tripwire: nothing else observes the sub-cut being on — every other
   test passes with it silently disabled, at whole-document heap-merge speed.
   The heap merge allocates ~3.3× the words the sub-cut does on a long fresh
   document, and the count is bit-deterministic, so a bound between the two
   regimes (on ≈ 1.3 M, off ≈ 4.3 M words for this document) fails loudly if the
   cut ever stops engaging. *)
let test_sp_allocation_tripwire () =
  with_pretrained "llama" (fun tok ->
      let doc =
        String.concat " "
          (List.init 20_000 (fun i ->
               Printf.sprintf "word%d rank%d." i (i mod 97)))
      in
      ignore (encode_ids tok "warm");
      let minor, promoted, major = Gc.counters () in
      ignore (Sys.opaque_identity (encode_ids tok doc));
      let minor', promoted', major' = Gc.counters () in
      let words =
        minor' +. major' -. promoted' -. (minor +. major -. promoted)
      in
      is_true
        ~msg:(Printf.sprintf "%.0f words allocated" words)
        (words < 2_500_000.))

let sentencepiece_tests =
  [
    test "sub-cut matches whole merges on llama" test_sp_subcut_differential;
    test "sub-cut keeps mark runs whole" test_sp_mark_runs;
    test "a crossing piece disables the sub-cut" test_sp_crossing_fallback;
    test "fused unknowns need a mark piece" test_sp_fuse_unk_gate;
    test "punctuation splits only where the vocabulary allows"
      test_sp_punct_adjacency;
    test "an unknown punctuation byte keeps no slot" test_sp_punct_unk_gate;
    test "a merge over a fallback token disables the sub-cut"
      test_sp_opaque_merge_fallback;
    test "affixes disable the sub-cut" test_sp_affix_gate;
    test "ignore_merges disables the sub-cut" test_sp_ignore_merges_gate;
    test "a fallback merge blocks a punctuation split"
      test_sp_opaque_merge_punct;
    test "the sub-cut stays engaged (allocation tripwire)"
      test_sp_allocation_tripwire;
  ]

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
    test "byte level decoder" test_byte_level_decoder;
    test "byte level after a split" test_byte_level_after_a_split;
    test "metaspace decoder" test_metaspace_decoder;
    test "metaspace without splitting places tokens"
      test_metaspace_no_split_offsets;
    test "replace and strip decoders" test_replace_and_strip_decoders;
    test "sentencepiece decoder" test_sentencepiece_decoder;
    test "skip special tokens" test_skip_special_tokens;
    test "a stream's texts" test_stream_cases;
    test "a stream skips special tokens" test_stream_skips_special_tokens;
    decoder_stream_law;
    (* Training *)
    test "wordpiece trains on pre-tokens" test_train_wordpiece_words;
    test "wordlevel trains on pre-tokens" test_train_word_level_words;
    test "wordlevel special token ids" test_train_word_level_specials;
    test "a line keeps its separator" test_train_line_separators;
    test "unigram trains on pre-tokens" test_train_unigram_words;
  ]

let () =
  exit
    (run "brot tokenization"
       [
         group "tokenization" tokenization_tests;
         group "batch" batch_tests;
         group "stride" stride_tests;
         group "sentencepiece" sentencepiece_tests;
       ])
