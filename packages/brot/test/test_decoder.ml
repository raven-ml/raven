(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Expected texts are Hugging Face tokenizers 0.23.1's: the decoder of the same
   name in [tokenizers.decoders], given the same tokens, unless a comment names
   another source. *)

open Windtrap
open Brot

let marker = "\u{2581}"
let r = "\u{FFFD}"

(* A row of tokens and the text they decode to. *)
let tokens_id (tokens, _) =
  "[" ^ String.concat "; " (List.map (Printf.sprintf "%S") tokens) ^ "]"

let decodes decoder (tokens, expected) =
  equal string expected (Decoder.decode decoder tokens)

let utf8 u =
  let b = Buffer.create 4 in
  Buffer.add_utf_8_uchar b u;
  Buffer.contents b

(* Valid UTF-8 text, one to four bytes per character in equal measure. *)
let utf8_text =
  Gen.with_pp (fun ppf s -> Format.fprintf ppf "%S" s)
  @@ Gen.map
       (fun us -> String.concat "" (List.map utf8 us))
       (Gen.list ~size:(Gen.int_range 0 12) Gen.uchar)

let json_text json =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json json with
  | Ok text -> text
  | Error msg -> failf "cannot encode: %s" msg

let json_of_text text =
  match Jsont_bytesrw.decode_string Jsont.json text with
  | Ok json -> json
  | Error msg -> failf "cannot decode %S: %s" text msg

let read_decoder text = require_ok (Decoder.of_json (json_of_text text))
let write_decoder d = json_text (Decoder.to_json d)

(* bpe *)

let bpe_tests =
  [
    cases ~id:tokens_id "turns the suffix into the space after its word"
      [
        ([ "hel"; "lo</w>"; "wor"; "ld</w>" ], "hello world");
        ([ "a</w>"; "b</w>" ], "a b");
        ([ "a</w>b"; "c" ], "a bc");
        ([ "</w>"; "a" ], " a");
      ]
      (decodes (Decoder.bpe ()));
    cases ~id:tokens_id "drops the suffix in the last token"
      [ ([ "only</w>" ], "only"); ([ "a"; "b</w>" ], "ab") ]
      (decodes (Decoder.bpe ()));
    cases ~id:tokens_id "adds no space after a token without the suffix"
      [ ([ "x"; "y" ], "xy"); ([], "") ]
      (decodes (Decoder.bpe ()));
    cases ~id:tokens_id "reads the suffix it is given"
      [ ([ "a@"; "b@" ], "a b"); ([ "a</w>"; "b" ], "a</w>b") ]
      (decodes (Decoder.bpe ~suffix:"@" ()));
    cases ~id:tokens_id
      "reads an empty suffix as one before every character and at the end of \
       every token"
      [
        ([ "ab"; "cd" ], " a b cd");
        ([ "ab"; ""; "cd" ], " a b  cd");
        ([ ""; "" ], " ");
      ]
      (decodes (Decoder.bpe ~suffix:"" ()));
  ]

(* byte_level *)

(* A text and where to cut its spelling into tokens: byte [i] of the text is
   character [i] of its spelling by [Normalizer.byte_level], and a [true] at [i]
   ends a token after it, so a cut may fall inside a character of the text. *)
let cut_text =
  let open Gen in
  with_pp (fun ppf (text, cuts) ->
      Format.fprintf ppf "%S cut after bytes [%s]" text
        (String.concat "; "
           (List.concat
              (List.mapi
                 (fun i cut -> if cut then [ string_of_int i ] else [])
                 cuts))))
  @@
  let* text = utf8_text in
  let+ cuts = list ~size:(constant (String.length text)) bool in
  (text, cuts)

let spell (text, cuts) =
  let tokens = ref [] and start = ref 0 in
  List.iteri
    (fun i cut ->
      if cut || i = String.length text - 1 then begin
        let piece = String.sub text !start (i + 1 - !start) in
        tokens := Normalizer.apply Normalizer.byte_level piece :: !tokens;
        start := i + 1
      end)
    cuts;
  List.rev !tokens

let cuts_a_character (text, cuts) =
  List.exists Fun.id
    (List.mapi
       (fun i cut ->
         cut
         && i + 1 < String.length text
         && Char.code text.[i + 1] land 0xC0 = 0x80)
       cuts)

let test_byte_level_spelling case =
  cover "a character cut across tokens" (cuts_a_character case);
  equal string (fst case) (Decoder.decode Decoder.byte_level (spell case))

let byte_level_tests =
  [
    prop ~count:300 "reads back the text Normalizer.byte_level spelled" cut_text
      test_byte_level_spelling;
    cases ~id:tokens_id "joins a character spelled across tokens"
      [
        (* The four bytes of U+1F44D, cut in every way. *)
        ([ "\xc3\xb0\xc5\x81\xc4\xb3\xc4\xaf" ], "\u{1F44D}");
        ([ "\xc3\xb0\xc5\x81"; "\xc4\xb3\xc4\xaf" ], "\u{1F44D}");
        ([ "\xc3\xb0"; "\xc5\x81"; "\xc4\xb3"; "\xc4\xaf" ], "\u{1F44D}");
        ([ "\xc3\xa6"; "\xc4\xb9"; "\xc2\xa5" ], "\u{65E5}");
        ([ "\xc4\xa0Hello" ], " Hello");
        ([ "\xc4\x80" ], "\x00");
        ([ "" ], "");
        ([], "");
      ]
      (decodes Decoder.byte_level);
    cases ~id:tokens_id
      "replaces each maximal ill-formed subsequence with U+FFFD"
      [
        (* A lone [\xE9], and a four-byte sequence cut short, cost one each; so
           does the lead of [\xC3\x28], whose [(] still decodes. *)
        ([ "\xc3\xa9" ], r);
        ([ "\xc3\xb0\xc5\x81\xc4\xb3" ], r);
        ([ "\xc3\xb0\xc5\x81\xc4\xb3(" ], r ^ "(");
        ([ "\xc3\x83(" ], r ^ "(");
        ([ "\xc2\xa1"; "\xc2\xac"; "\xc2\xae"; "\xc3\xbf" ], r ^ r ^ r ^ r);
        (* The lead of a sequence whose second byte is out of range is a subpart
           of its own, so the bytes after it are subparts too: an overlong
           encoding, a surrogate and a value above U+10FFFF cost one each. *)
        ([ "\xc3\xa0\xc4\xa2\xc4\xa2" ], r ^ r ^ r);
        ([ "\xc3\xad\xc5\x82\xc4\xa2" ], r ^ r ^ r);
        ([ "\xc3\xb0\xc4\xa2\xc4\xa2\xc4\xa2" ], r ^ r ^ r ^ r);
        ([ "\xc3\xb4\xc4\xb2\xc4\xa2\xc4\xa2" ], r ^ r ^ r ^ r);
        ([ "\xc3\xb5\xc4\xa2\xc4\xa2\xc4\xa2" ], r ^ r ^ r ^ r);
        ([ "\xc3\x81\xc4\xa2" ], r ^ r);
        ([ "\xc3\x80\xc4\xa2" ], r ^ r);
        ([ "\xc4\xa2" ], r);
        ([ "\xc3\xbf" ], r);
        (* A sequence broken after its second byte costs one for the pair. *)
        ([ "\xc3\xa1\xc4\xa2A" ], r ^ "A");
        ([ "\xc3\xb0\xc5\x81A" ], r ^ "A");
        (* The edges of what is well formed still decode. *)
        ([ "\xc3\xa2\xc4\xa4\xc2\xac" ], "\u{20AC}");
        ([ "\xc3\xa0\xc5\x82\xc4\xa2" ], "\u{0800}");
        ([ "\xc3\xad\xc5\x81\xc2\xbf" ], "\u{D7FF}");
        ([ "\xc3\xb4\xc4\xb1\xc2\xbf\xc2\xbf" ], "\u{10FFFF}");
      ]
      (decodes Decoder.byte_level);
    cases ~id:tokens_id
      "leaves a token with a character outside the alphabet as its own bytes"
      [
        (* The two tokens differ in their last character alone: without it every
           character maps, [\xc4\xa0] and [\xc3\xa9] among them, and with it
           none does. *)
        ([ "\xc4\xa0caf\xc3\xa9X" ], " caf" ^ r ^ "X");
        ([ "\xc4\xa0caf\xc3\xa9\u{65E5}" ], "\xc4\xa0caf\xc3\xa9\u{65E5}");
        ([ "\xc4\xa0\u{65E5}" ], "\xc4\xa0\u{65E5}");
        ([ "a b" ], "a b");
        ([ "\xc4\xa0a"; "\u{65E5}"; "\xc4\xa0b" ], " a\u{65E5} b");
      ]
      (decodes Decoder.byte_level);
  ]

(* byte_fallback *)

(* Tokens are plain text, or a run of byte tokens spelling valid UTF-8 in upper
   or lower case hexadecimal. *)
type segment = Plain of string | Run of string * bool

let segment =
  Gen.one_of
    [
      Gen.map
        (fun s -> Plain s)
        (Gen.of_list [ "a"; marker ^ "b"; "<0x"; "x>"; "<0xGG>"; "" ]);
      Gen.map
        (fun (s, upper) -> Run (s, upper))
        (Gen.pair (Gen.such_that (fun s -> s <> "") utf8_text) Gen.bool);
    ]

let segment_tokens = function
  | Plain s -> [ s ]
  | Run (s, upper) ->
      List.init (String.length s) (fun i ->
          if upper then Printf.sprintf "<0x%02X>" (Char.code s.[i])
          else Printf.sprintf "<0x%02x>" (Char.code s.[i]))

let segment_text = function Plain s | Run (s, _) -> s

let multi_byte = function
  | Run (s, _) -> String.exists (fun c -> c >= '\x80') s
  | Plain _ -> false

let test_byte_fallback_runs segments =
  cover "a multi-byte character" (List.exists multi_byte segments);
  equal string
    (String.concat "" (List.map segment_text segments))
    (Decoder.decode Decoder.byte_fallback
       (List.concat_map segment_tokens segments))

let byte_fallback_tests =
  [
    prop ~count:300 "spells the bytes of a run of byte tokens"
      (Gen.list ~size:(Gen.int_range 0 6) segment)
      test_byte_fallback_runs;
    cases ~id:tokens_id "spells a run of byte tokens as Hugging Face does"
      [
        ([ "<0x41>" ], "A");
        ([ "<0xE2>"; "<0x96>"; "<0x81>" ], marker);
        ([ "a"; "<0xC3>"; "<0xA9>"; "b" ], "a\u{E9}b");
        ([ "<0x0A>"; "x"; "<0x0A>" ], "\nx\n");
        ([], "");
      ]
      (decodes Decoder.byte_fallback);
    cases ~id:tokens_id
      "replaces each byte of a run that is not UTF-8 with U+FFFD"
      [
        ([ "<0xFF>" ], r);
        ([ "<0xFF>"; "<0xFE>" ], r ^ r);
        ([ "<0xC3>"; "a" ], r ^ "a");
      ]
      (decodes Decoder.byte_fallback);
    cases ~id:tokens_id
      "reads a byte token as <0x, two hexadecimal digits of either case, and >"
      [
        ([ "<0x0a>" ], "\n");
        ([ "<0x0A>" ], "\n");
        ([ "<0xGG>" ], "<0xGG>");
        ([ "<0x1>" ], "<0x1>");
        ([ "<0x_1>" ], "<0x_1>");
        ([ "<0X41>" ], "<0X41>");
        ([ "<0x411>" ], "<0x411>");
        ([ "plain" ], "plain");
      ]
      (decodes Decoder.byte_fallback);
  ]

(* wordpiece *)

let wordpiece_tests =
  [
    cases ~id:tokens_id "strips the prefix from every subword but the first"
      [
        ([ "una"; "##ffa"; "##ble" ], "unaffable"); ([ "##ab"; "cd" ], "##ab cd");
      ]
      (decodes (Decoder.wordpiece ()));
    cases ~id:tokens_id "reads the prefix it is given"
      [ ([ "un"; "@@aff"; "@@able" ], "unaffable"); ([ "a"; "##b" ], "a ##b") ]
      (decodes (Decoder.wordpiece ~prefix:"@@" ()));
    cases ~id:tokens_id
      "takes back the space before punctuation and contractions"
      [
        ([ "hello"; ","; "world"; "!" ], "hello, world!");
        ([ "a"; "?"; "b"; "!"; "c"; ","; "d"; "." ], "a? b! c, d.");
        ([ "is"; "n't" ], "isn't");
        ([ "i"; "'m" ], "i'm");
        ([ "it"; "'s" ], "it's");
        ([ "we"; "'ve"; "they"; "'re"; "do"; "n't" ], "we've they're don't");
        ([], "");
      ]
      (decodes (Decoder.wordpiece ()));
    cases ~id:tokens_id
      "turns \" ' \" into \"'\" and \" do not\" into \" don't\""
      [
        (* The cleanup runs on the first token too. *)
        ([ "a do not" ], "a don't");
        ([ "a ' b do not x" ], "a'b don't x");
      ]
      (decodes (Decoder.wordpiece ()));
    cases ~id:tokens_id "cleans each token on its own"
      [
        (* Only the space the decoder added is taken back, so a full stop that
           was a token of its own keeps the space before the next word. *)
        ([ "3"; "."; "14" ], "3. 14");
        ([ "don"; "'"; "t" ], "don ' t");
        ([ "a"; "do"; "not" ], "a do not");
        ([ "  x  " ], "  x  ");
      ]
      (decodes (Decoder.wordpiece ()));
    cases ~id:tokens_id "keeps every space without cleanup"
      [
        ([ "hello"; ","; "world"; "!" ], "hello , world !");
        ([ "3"; "."; "14" ], "3 . 14");
      ]
      (decodes (Decoder.wordpiece ~cleanup:false ()));
  ]

(* metaspace *)

let metaspace_tests =
  [
    cases ~id:tokens_id "drops every marker of the first token"
      [
        ([ marker ^ "Hello"; marker ^ "world" ], "Hello world");
        ([ marker; marker ^ "a" ], " a");
        ([ "a" ^ marker ^ "b"; marker ^ "c" ], "ab c");
        ([ marker ^ marker ^ "a"; "b" ], "ab");
        ([ marker ], "");
        ([], "");
      ]
      (decodes (Decoder.metaspace ()));
    cases ~id:tokens_id "turns every marker into a space with `Never"
      [
        ([ marker ^ "Hello"; marker ^ "world" ], " Hello world");
        ([ marker; marker ^ "a" ], "  a");
        ([ "a" ^ marker ^ "b"; marker ^ "c" ], "a b c");
        ([ marker ], " ");
      ]
      (decodes (Decoder.metaspace ~prepend_scheme:`Never ()));
    cases ~id:tokens_id "decodes `First as `Always"
      [
        ([ marker ^ "Hello"; marker ^ "world" ], "Hello world");
        ( [ marker ^ "Hello"; marker ^ "world"; marker ^ marker ^ "x" ],
          "Hello world  x" );
        ([ marker; marker ^ "a" ], " a");
      ]
      (decodes (Decoder.metaspace ~prepend_scheme:`First ()));
    cases ~id:tokens_id "reads the marker it is given"
      [ ([ "_a"; "_b" ], "a b"); ([ marker ^ "a" ], marker ^ "a") ]
      (decodes (Decoder.metaspace ~replacement:"_" ()));
    cases ~id:(Printf.sprintf "%S")
      "raises Invalid_argument unless the replacement is one character"
      [ ""; "ab"; "__"; "\x81" ] (fun replacement ->
        raises_match Exn.invalid_arg (fun () ->
            Decoder.metaspace ~replacement ()));
  ]

(* ctc *)

let ctc_tests =
  [
    cases ~id:tokens_id "collapses repeats before cutting the pad token"
      [
        ([ "h"; "e"; "l"; "l"; "o"; "|"; "w"; "o"; "r"; "l"; "d" ], "helo world");
        ([ "a"; "a"; "b"; "<pad>"; "b"; "|"; "c" ], "abb c");
        ([ "a"; "|"; "|"; "b" ], "a b");
        ([ "<pad>"; "<pad>" ], "");
        ([], "");
      ]
      (decodes (Decoder.ctc ()));
    cases ~id:tokens_id
      "cuts the pad token wherever it falls and drops the emptied tokens"
      [
        ([ "x<pad>y" ], "xy");
        ([ "<pad>x<pad>" ], "x");
        ([ "a<pad>b<pad>c" ], "abc");
        ([ "<pad>x"; "y<pad>" ], "xy");
        ([ "<pad>" ], "");
        ([ ""; "a" ], "a");
        ([ "a"; ""; "b" ], "ab");
      ]
      (decodes (Decoder.ctc ()));
    cases ~id:tokens_id "cleans each token before the delimiter becomes a space"
      [
        ([ " ." ], ".");
        ([ "a"; " ,"; "b" ], "a,b");
        ([ "x"; " n't" ], "xn't");
        ([ "a"; " ' "; "b" ], "a'b");
        ([ "a do not" ], "a don't");
        (* A delimiter turned into a space is not cleaned. *)
        ([ "|." ], " .");
        ([ "|" ], " ");
        ([ "  x  " ], "  x  ");
      ]
      (decodes (Decoder.ctc ()));
    cases ~id:tokens_id "keeps the delimiter and the spaces without cleanup"
      [
        ([ "x<pad>y" ], "xy");
        ([ "<pad>x<pad>" ], "x");
        ([ "a"; " ,"; "b" ], "a ,b");
        ([ "h"; "e"; "l"; "l"; "o"; "|"; "w" ], "helo|w");
      ]
      (decodes (Decoder.ctc ~cleanup:false ()));
    cases ~id:tokens_id "reads the pad and delimiter tokens it is given"
      [
        ([ "a"; "a"; "[PAD]"; "a"; "_"; "b" ], "aa b"); ([ "<pad>|" ], "<pad>|");
      ]
      (decodes (Decoder.ctc ~pad_token:"[PAD]" ~word_delimiter_token:"_" ()));
    cases ~id:tokens_id
      "reads an empty delimiter as one before every character and at the end \
       of every token"
      [
        ([ "ab"; "cd" ], " a b  c d ");
        ([ "" ], " ");
        ([ "a"; "" ], " a  ");
        ([ "ab"; ""; "cd" ], " a b   c d ");
      ]
      (decodes (Decoder.ctc ~word_delimiter_token:"" ()));
  ]

(* sequence *)

(* The decoder LLaMA and other SentencePiece models carry:
   [Sequence([Replace("▁", " "), ByteFallback(), Fuse(), Strip(" ", 1, 0)])],
   and its JSON, verbatim from LLaMA's tokenizer file. *)
let sentencepiece =
  Decoder.sequence
    [
      Decoder.replace ~pattern:marker ~by:" " ();
      Decoder.byte_fallback;
      Decoder.fuse;
      Decoder.strip ~content:" " ~start:1 ();
    ]

let sentencepiece_json =
  {|{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"|}
  ^ marker
  ^ {|"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]}|}

let sequence_tests =
  [
    cases ~id:tokens_id "feeds each decoder's tokens to the next"
      [
        ([ marker ^ "a"; marker; marker ^ "b" ], "a  b");
        ([ marker; "<0x0A>"; "<0x0A>"; "Not" ], "\n\nNot");
        ([ marker ^ "Hello"; marker ^ "world" ], "Hello world");
        (* [Replace] rewrites each token, so [ByteFallback] still sees the byte
           tokens as tokens of their own. *)
        ([ marker; "<0xE2>"; "<0x9C>"; "<0x93>" ], "\u{2713}");
        ([ "<0xFF>"; marker ^ "a" ], r ^ " a");
        ([ marker ], "");
        ([], "");
      ]
      (decodes sentencepiece);
    cases ~id:tokens_id "gives the next decoder the tokens ctc kept"
      [
        (* The emptied token is gone, so metaspace sees ["▁a"] first. *)
        ([ "<pad>"; marker ^ "a" ], "a");
        ([ marker ^ "a"; marker ^ "b" ], "a b");
      ]
      (decodes (Decoder.sequence [ Decoder.ctc (); Decoder.metaspace () ]));
    cases ~id:tokens_id "hides token boundaries after a collapsing decoder"
      [ ([ " a"; " b" ], "a b") ]
      (decodes
         (Decoder.sequence
            [ Decoder.fuse; Decoder.strip ~content:" " ~start:1 () ]));
    cases ~id:tokens_id "of one decoder decodes as that decoder"
      [ ([ " a"; " b" ], "ab") ]
      (decodes (Decoder.sequence [ Decoder.strip ~content:" " ~start:1 () ]));
    cases ~id:tokens_id "of no decoder concatenates the tokens"
      [ ([ "a"; " b" ], "a b"); ([], "") ]
      (decodes (Decoder.sequence []));
  ]

(* replace *)

let replace_tests =
  [
    cases ~id:tokens_id "replaces every literal occurrence in each token"
      [ ([ marker ^ "a"; marker ^ "b" ], " a b"); ([ "a" ], "a"); ([], "") ]
      (decodes (Decoder.replace ~pattern:marker ~by:" " ()));
    cases ~id:tokens_id "replaces left to right without overlap"
      [ ([ "aaa" ], "ba"); ([ "a.a" ], "a.a") ]
      (decodes (Decoder.replace ~pattern:"aa" ~by:"b" ()));
    cases ~id:tokens_id
      "inserts the replacement before every character and at the end for an \
       empty pattern"
      [ ([ "ab" ], "_a_b_"); ([ "a"; "" ], "_a_") ]
      (decodes (Decoder.replace ~pattern:"" ~by:"_" ()));
  ]

(* strip *)

let strip_tests =
  [
    cases ~id:tokens_id
      "cuts up to start occurrences from the start of each token"
      [ ([ "  a  "; "  b  " ], " a   b  "); ([ "   " ], "  "); ([ "a" ], "a") ]
      (decodes (Decoder.strip ~content:" " ~start:1 ()));
    cases ~id:tokens_id "cuts up to stop occurrences from the end of each token"
      [ ([ "  a  "; "  b  " ], "  a   b ") ]
      (decodes (Decoder.strip ~content:" " ~stop:1 ()));
    cases ~id:tokens_id "empties a token with nothing between the cuts"
      [
        ([ "  a  "; "  b  " ], "ab");
        ([ "a" ], "a");
        ([], "");
        (* brot.mli: the cuts overlap. Hugging Face panics. *)
        ([ "   " ], "");
      ]
      (decodes (Decoder.strip ~content:" " ~start:2 ~stop:2 ()));
    cases ~id:tokens_id "counts a content of several bytes as one occurrence"
      [ ([ marker ^ marker ^ "x" ], marker ^ "x"); ([ "  a" ], "  a") ]
      (decodes (Decoder.strip ~content:marker ~start:1 ()));
    cases ~id:tokens_id "strips a space by default"
      [ ([ "  a" ], " a") ]
      (decodes (Decoder.strip ~start:1 ()));
    (* brot.mli: an empty [content] strips nothing. *)
    cases ~id:tokens_id "strips nothing for an empty content"
      [ ([ "ab" ], "ab") ]
      (decodes (Decoder.strip ~content:"" ~start:1 ~stop:1 ()));
    cases
      ~id:(fun (start, stop) -> Printf.sprintf "~start:%d ~stop:%d" start stop)
      "raises Invalid_argument for a negative start or stop"
      [ (-1, 0); (0, -1); (min_int, 0) ]
      (fun (start, stop) ->
        raises_match Exn.invalid_arg (fun () -> Decoder.strip ~start ~stop ()));
  ]

(* fuse *)

let fuse_tests =
  [
    cases ~id:tokens_id "joins the tokens with nothing between them"
      [ ([ "a"; " b"; "c" ], "a bc"); ([], "") ]
      (decodes Decoder.fuse);
  ]

(* to_json and of_json *)

(* Each decoder with the JSON Hugging Face reads back as the same decoder.
   [ByteLevel]'s members that only a pre-tokenizer reads are written at their
   defaults, which Hugging Face requires; [Metaspace]'s [split] is left out. *)
let decoder_json =
  [
    (Decoder.bpe (), {|{"type":"BPEDecoder","suffix":"</w>"}|});
    ( Decoder.byte_level,
      {|{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true}|} );
    (Decoder.byte_fallback, {|{"type":"ByteFallback"}|});
    (Decoder.wordpiece (), {|{"type":"WordPiece","prefix":"##","cleanup":true}|});
    ( Decoder.metaspace (),
      {|{"type":"Metaspace","replacement":"|} ^ marker
      ^ {|","prepend_scheme":"always"}|} );
    ( Decoder.metaspace ~replacement:"_" ~prepend_scheme:`Never (),
      {|{"type":"Metaspace","replacement":"_","prepend_scheme":"never"}|} );
    ( Decoder.ctc (),
      {|{"type":"CTC","pad_token":"<pad>","word_delimiter_token":"|","cleanup":true}|}
    );
    ( Decoder.replace ~pattern:marker ~by:" " (),
      {|{"type":"Replace","pattern":{"String":"|} ^ marker
      ^ {|"},"content":" "}|} );
    ( Decoder.strip ~content:" " ~start:1 (),
      {|{"type":"Strip","content":" ","start":1,"stop":0}|} );
    (Decoder.fuse, {|{"type":"Fuse"}|});
    (sentencepiece, sentencepiece_json);
  ]

(* Loading a tokenizer file and writing its decoder back gives Hugging Face the
   decoder the file had. GPT-2's carries [add_prefix_space] and [trim_offsets],
   which Hugging Face insists on. *)
let test_file_decoder (model, expected) =
  Fixture.with_tokenizer_file model (fun path ->
      let t = require_ok (from_file path) in
      equal string expected (write_decoder (require_some (decoder t))))

let to_json_tests =
  [
    cases ~id:snd "writes JSON Hugging Face reads back as the same decoder"
      decoder_json (fun (d, expected) ->
        equal string expected (write_decoder d));
    cases ~id:fst "writes back the decoder of a tokenizer file"
      [
        ( "gpt2",
          {|{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true}|}
        );
        ("bert_base", {|{"type":"WordPiece","prefix":"##","cleanup":true}|});
        ("mistral", sentencepiece_json);
      ]
      test_file_decoder;
  ]

let of_json_tests =
  [
    cases ~id:snd "reads back what to_json writes" decoder_json
      (fun (_, text) ->
        Law.round_trip string pass read_decoder write_decoder text);
    test "reads the decoder LLaMA's file carries" (fun () ->
        let tokens = [ marker; "<0x0A>"; "<0x0A>"; "Not" ] in
        equal string "\n\nNot"
          (Decoder.decode (read_decoder sentencepiece_json) tokens));
    test "rejects a Replace whose pattern is a regular expression" (fun () ->
        equal (result pass string)
          (Error
             "Replace decoder: a regular expression pattern is not supported, \
              only a literal one")
          (Decoder.of_json
             (json_of_text
                {|{"type":"Replace","pattern":{"Regex":"a+"},"content":"b"}|})));
    cases ~id:Fun.id
      "errors on a value that is not an object or has no known type"
      [ {|[]|}; {|"Fuse"|}; {|{}|}; {|{"type":3}|}; {|{"type":"Nope"}|} ]
      (fun text ->
        is_error ~pp:Decoder.pp (Decoder.of_json (json_of_text text)));
    cases ~id:fst "reads a missing member as its constructor's default"
      [
        ({|{"type":"BPEDecoder"}|}, Decoder.bpe ());
        ({|{"type":"WordPiece"}|}, Decoder.wordpiece ());
        ({|{"type":"Metaspace"}|}, Decoder.metaspace ());
        ({|{"type":"CTC"}|}, Decoder.ctc ());
        ({|{"type":"Strip"}|}, Decoder.strip ());
      ]
      (fun (text, default) ->
        equal string (write_decoder default) (write_decoder (read_decoder text)));
    test "reads a Replace without content as replacing with nothing" (fun () ->
        equal string
          (write_decoder (Decoder.replace ~pattern:"a" ~by:"" ()))
          (write_decoder
             (read_decoder {|{"type":"Replace","pattern":{"String":"a"}}|})));
    cases ~id:Fun.id
      "errors on a Metaspace whose replacement is not one character"
      [
        {|{"type":"Metaspace","replacement":"ab"}|};
        {|{"type":"Metaspace","replacement":""}|};
      ] (fun text ->
        is_error ~pp:Decoder.pp (Decoder.of_json (json_of_text text)));
    cases ~id:Fun.id "errors on a missing member its constructor requires"
      [ {|{"type":"Replace","content":"b"}|} ] (fun text ->
        is_error ~pp:Decoder.pp (Decoder.of_json (json_of_text text)));
  ]

let () =
  exit
    (run "Decoder"
       [
         group "bpe" bpe_tests;
         group "byte_level" byte_level_tests;
         group "byte_fallback" byte_fallback_tests;
         group "wordpiece" wordpiece_tests;
         group "metaspace" metaspace_tests;
         group "ctc" ctc_tests;
         group "sequence" sequence_tests;
         group "replace" replace_tests;
         group "strip" strip_tests;
         group "fuse" fuse_tests;
         group "to_json" to_json_tests;
         group "of_json" of_json_tests;
       ])
