(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Unicode processing tests for brot *)

open Windtrap
open Brot

(* Normalization via public API *)

let test_lowercase_normalization () =
  let text = "HELLO WORLD" in
  let normalizer = Normalizer.lowercase in
  let result = Normalizer.apply normalizer text in
  equal ~msg:"lowercase" string "hello world" result

let test_strip_accents_normalization () =
  let text = "caf\xC3\xA9 na\xC3\xAFve r\xC3\xA9sum\xC3\xA9" in
  let normalizer =
    Normalizer.sequence [ Normalizer.nfd; Normalizer.strip_accents ]
  in
  let result = Normalizer.apply normalizer text in
  equal ~msg:"strip accents" string "cafe naive resume" result

let test_normalization_sequence () =
  let text = "  HELLO  World  " in
  let normalizer =
    Normalizer.sequence
      [
        Normalizer.lowercase;
        Normalizer.strip ();
        Normalizer.replace ~pattern:"\\s+" ~replacement:" ";
      ]
  in
  let result = Normalizer.apply normalizer text in
  equal ~msg:"sequence" string "hello world" result

(* Case mapping is not case folding: the ligatures and the sharp s lower to
   themselves. Expectations from HuggingFace [normalizers.Lowercase()]. *)
let test_lowercase_is_not_folding () =
  let case text expected =
    equal
      ~msg:(Printf.sprintf "lowercase %S" text)
      string expected
      (Normalizer.apply Normalizer.lowercase text)
  in
  case "\xC3\x9F" "\xC3\x9F";
  case "\xE1\xBA\x9E" "\xC3\x9F";
  case "\xEF\xAC\x81" "\xEF\xAC\x81";
  case "\xEF\xAC\x84" "\xEF\xAC\x84";
  case "\xC4\xB0" "i\xCC\x87";
  case "\xCE\xA3" "\xCF\x83";
  case "\xC7\x85" "\xC7\x86";
  case "\xC3\x80\xC3\x89\xC3\x8E" "\xC3\xA0\xC3\xA9\xC3\xAE"

(* Expectations from HuggingFace [normalizers.StripAccents()], which removes
   every mark and does not decompose. *)
let test_strip_accents_keeps_composition () =
  let case text expected =
    equal
      ~msg:(Printf.sprintf "strip_accents %S" text)
      string expected
      (Normalizer.apply Normalizer.strip_accents text)
  in
  (* Bengali vowel sign O and Devanagari vowel sign I are spacing marks. *)
  case "\xE0\xA7\x8B" "";
  case "\xE0\xA4\xBF" "";
  (* Enclosing and nonspacing marks go too. *)
  case "a\xE0\xA4\x83\xE2\x83\x9D\xCC\x81" "a";
  (* Precomposed characters are left alone without a preceding NFD. *)
  case "\xC3\xA1" "\xC3\xA1";
  case "caf\xC3\xA9" "caf\xC3\xA9";
  case "\xE0\xA4\x95\xE0\xA4\xBC" "\xE0\xA4\x95";
  case "A\xCC\x81" "A"

(* Expectations from HuggingFace [normalizers.BertNormalizer(clean_text=True,
   handle_chinese_chars=True, strip_accents=None, lowercase=True)], the
   bert-base-uncased settings. *)
let test_bert_normalizer () =
  let n = Normalizer.bert () in
  let case text expected =
    equal
      ~msg:(Printf.sprintf "bert normalize %S" text)
      string expected (Normalizer.apply n text)
  in
  (* Only nonspacing marks are stripped, so the vowel signs of an abugida
     survive: ["नमस्ते हिन्दी"] keeps its ि and ी. *)
  case
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87 \
     \xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA5\x8D\xE0\xA4\xA6\xE0\xA5\x80"
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA4\xA4 \
     \xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA4\xA6\xE0\xA5\x80";
  case "a\xE0\xA4\x83\xE2\x83\x9D\xCC\x81" "a\xE0\xA4\x83\xE2\x83\x9D";
  (* Accents proper are stripped, after decomposition. *)
  case "caf\xC3\xA9" "cafe";
  case "A\xCC\x81BC" "abc";
  case "\xC4\xB0" "i";
  case "\xE1\xBE\xBC" "\xCE\xB1";
  (* Lowercasing, not folding. *)
  case "\xC3\x9F" "\xC3\x9F";
  case "\xE1\xBA\x9E" "\xC3\x9F";
  case "\xEF\xAC\x81" "\xEF\xAC\x81";
  case "\xCE\xA3" "\xCF\x83";
  (* Control, format and private use characters are removed, and so are NUL and
     the replacement character. *)
  case "\x00\x07\x7F" "";
  case "a\xC2\xADb" "ab";
  case "a\xE2\x80\x8Bb" "ab";
  case "a\xEE\x80\x80b" "ab";
  case "a\xEF\xBF\xBDb" "ab";
  (* Unassigned codepoints are not controls: they survive to reach the model. *)
  case "\xF4\x8F\xBF\xBF" "\xF4\x8F\xBF\xBF";
  case "\xCD\xB8" "\xCD\xB8";
  (* Whitespace of any kind becomes a plain space. *)
  case "a\xE2\x80\xA8b" "a b";
  case "a\xE3\x80\x80b" "a b";
  (* CJK ideographs are padded with spaces, one at a time. *)
  case "a\xE6\xBC\xA2b" "a \xE6\xBC\xA2 b";
  case "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E"
    " \xE6\x97\xA5  \xE6\x9C\xAC  \xE8\xAA\x9E "

(* Alignment *)

(* The span every character of the normalized text reports on the input, in the
   order the characters come out. Expectations are what HuggingFace reports for
   one-character tokens; regenerate them with [uv run --with tokenizers python3
   test/scripts/hf_alignments.py]. *)
let char_spans normalizer input =
  let normalized, alignment = Normalizer.apply_aligned normalizer input in
  let buffer = Buffer.create 64 in
  let i = ref 0 in
  while !i < String.length normalized do
    let n = Uchar.utf_decode_length (String.get_utf_8_uchar normalized !i) in
    let start, stop =
      Normalizer.original_span alignment ~start:!i ~stop:(!i + n)
    in
    if !i > 0 then Buffer.add_char buffer ' ';
    Buffer.add_string buffer (Printf.sprintf "%d,%d" start stop);
    i := !i + n
  done;
  (normalized, Buffer.contents buffer)

let aligned normalizer label input expected_text expected_spans =
  let text, spans = char_spans normalizer input in
  equal
    ~msg:(Printf.sprintf "%s %S: text" label input)
    string expected_text text;
  equal
    ~msg:(Printf.sprintf "%s %S: spans" label input)
    string expected_spans spans

(* A character standing for several input characters takes the range of the
   first of them, and one standing for none takes the range of the character
   before it, so a decomposition scattered by canonical ordering reports the
   character that ended up in its place rather than the one it came from. *)
let test_align_nfd () =
  let case = aligned Normalizer.nfd "nfd" in
  case "caf\xC3\xA9" "cafe\xCC\x81" "0,1 1,2 2,3 3,5 3,5";
  case "\xE1\xBA\x9B\xCC\xA3xy" "\xC5\xBF\xCC\xA3\xCC\x87xy"
    "0,3 3,5 3,5 5,6 6,7";
  case "\xE1\xB8\x8D\xCC\x81" "d\xCC\xA3\xCC\x81" "0,3 0,3 3,5";
  case "\xED\x95\x9C\xEA\xB8\x80"
    "\xE1\x84\x92\xE1\x85\xA1\xE1\x86\xAB\xE1\x84\x80\xE1\x85\xB3\xE1\x86\xAF"
    "0,3 0,3 0,3 3,6 3,6 3,6";
  case "a\xCC\x81\xCC\x96" "a\xCC\x96\xCC\x81" "0,1 1,3 3,5"

let test_align_nfc () =
  let case = aligned Normalizer.nfc "nfc" in
  case "cafe\xCC\x81" "caf\xC3\xA9" "0,1 1,2 2,3 3,4";
  case "e\xCC\x81\xCC\xA3" "\xE1\xBA\xB9\xCC\x81" "0,1 3,5";
  case "a\xCC\x81\xCC\x96" "\xC3\xA1\xCC\x96" "0,1 3,5";
  case "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8Z" "\xEA\xB0\x81Z" "0,3 9,10";
  case "e\xCC\xA3\xCC\x81\xCC\x96" "\xE1\xBA\xB9\xCC\x96\xCC\x81" "0,1 3,5 5,7";
  case "\xE1\xBA\x9B\xCC\xA3xy" "\xE1\xBA\x9B\xCC\xA3xy" "0,3 3,5 5,6 6,7"

let test_align_nfkc_nfkd () =
  let case = aligned Normalizer.nfkc "nfkc" in
  case "\xEF\xAC\x81x \xE2\x91\xA0 \xEF\xBC\xA1" "fix 1 A"
    "0,3 0,3 3,4 4,5 5,8 8,9 9,12";
  case "\xC2\xBD\xE2\x81\xB5" "1\xE2\x81\x8425" "0,2 0,2 0,2 2,5";
  let case = aligned Normalizer.nfkd "nfkd" in
  case "\xEF\xB7\xBA!"
    "\xD8\xB5\xD9\x84\xD9\x89 \xD8\xA7\xD9\x84\xD9\x84\xD9\x87 \
     \xD8\xB9\xD9\x84\xD9\x8A\xD9\x87 \xD9\x88\xD8\xB3\xD9\x84\xD9\x85!"
    "0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 \
     3,4";
  case "\xEF\xAC\x81x \xE2\x91\xA0 \xC7\x84" "fix 1 DZ\xCC\x8C"
    "0,3 0,3 3,4 4,5 5,8 8,9 9,11 9,11 9,11"

(* A character the normalizer dropped belongs to no span, so the spans of a
   stripped text skip over what it removed. *)
let test_align_text_transforms () =
  aligned Normalizer.lowercase "lowercase" "A\xC4\xB0B\xC3\x9F"
    "ai\xCC\x87b\xC3\x9F" "0,1 1,3 1,3 3,4 4,6";
  aligned Normalizer.strip_accents "strip accents"
    "\xC3\xA1a\xCC\x81b\xCC\x81\xCC\x81c" "\xC3\xA1abc" "0,2 2,3 5,6 10,11";
  aligned (Normalizer.strip ()) "strip" "  a b  " "a b" "2,3 3,4 4,5";
  aligned
    (Normalizer.strip ~right:false ())
    "strip left" "\t\n a b \n" "a b \n" "3,4 4,5 5,6 6,7 7,8";
  aligned
    (Normalizer.strip ~left:false ())
    "strip right" " a b \n" " a b" "0,1 1,2 2,3 3,4"

(* A replacement stands for the last character it replaced, whatever its
   length. *)
let test_align_replace () =
  aligned
    (Normalizer.replace ~pattern:" " ~replacement:"\xE2\x96\x81")
    "replace string" "a  b" "a\xE2\x96\x81\xE2\x96\x81b" "0,1 1,2 2,3 3,4";
  aligned
    (Normalizer.replace ~pattern:"\\s+" ~replacement:" ")
    "replace collapse" "a \t\n b" "a b" "0,1 4,5 5,6";
  aligned
    (Normalizer.replace ~pattern:"a" ~replacement:"xyz")
    "replace grow" "za z" "zxyz z" "0,1 1,2 1,2 1,2 2,3 3,4";
  aligned
    (Normalizer.replace ~pattern:"ab+" ~replacement:"X")
    "replace shrink" "zabbbz" "zXz" "0,1 4,5 5,6";
  aligned
    (Normalizer.replace ~pattern:"\\s+" ~replacement:"")
    "replace delete" "a \t b" "ab" "0,1 4,5"

(* An inserted character stands for the one it was put next to: the prefix and
   the first character of the text share a span. *)
let test_align_prepend () =
  aligned
    (Normalizer.prepend "\xE2\x96\x81")
    "prepend" "Hello" "\xE2\x96\x81Hello" "0,1 0,1 1,2 2,3 3,4 4,5";
  aligned
    (Normalizer.prepend "\xE2\x96\x81")
    "prepend" " x" "\xE2\x96\x81 x" "0,1 0,1 1,2";
  aligned (Normalizer.prepend "<<") "prepend multi" "Hello" "<<Hello"
    "0,1 0,1 0,1 1,2 2,3 3,4 4,5"

let test_align_bert () =
  let case = aligned (Normalizer.bert ()) "bert" in
  case "\xC3\x87a \xE6\xBC\xA2 a\xCC\x81x" "ca  \xE6\xBC\xA2  ax"
    "0,2 2,3 3,4 4,7 4,7 4,7 7,8 8,9 11,12";
  case "a\xE6\xBC\xA2b" "a \xE6\xBC\xA2 b" "0,1 1,4 1,4 1,4 4,5";
  case "a\xC2\xADb\tc" "ab c" "0,1 3,4 4,5 5,6";
  (* The stripped virama leaves a hole, so the spans are not contiguous. *)
  case
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87"
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA4\xA4" "0,3 3,6 6,9 12,15";
  case "Caf\xC3\xA9" "cafe" "0,1 1,2 2,3 3,5"

let test_align_sequence () =
  let llama =
    Normalizer.sequence
      [
        Normalizer.prepend "\xE2\x96\x81";
        Normalizer.replace ~pattern:" " ~replacement:"\xE2\x96\x81";
      ]
  in
  aligned llama "llama" "\n\nNot" "\xE2\x96\x81\n\nNot"
    "0,1 0,1 1,2 2,3 3,4 4,5";
  aligned llama "llama" "a  b" "\xE2\x96\x81a\xE2\x96\x81\xE2\x96\x81b"
    "0,1 0,1 1,2 2,3 3,4";
  aligned
    (Normalizer.sequence
       [ Normalizer.nfd; Normalizer.strip_accents; Normalizer.lowercase ])
    "nfd strip lower" "Caf\xC3\xA9" "cafe" "0,1 1,2 2,3 3,5"

(* Each byte becomes one character, so a character split across bytes reports
   itself whole. [add_prefix_space] has no counterpart in HuggingFace's
   [normalizers.ByteLevel()], which takes no arguments; the rest matches it. *)
let test_align_byte_level () =
  aligned (Normalizer.byte_level ()) "byte level" "a\xC3\xA9"
    "a\xC3\x83\xC2\xA9" "0,1 1,3 1,3";
  aligned (Normalizer.byte_level ()) "byte level" " x" "\xC4\xA0x" "0,1 1,2";
  aligned
    (Normalizer.byte_level ~add_prefix_space:true ())
    "byte level prefix" "ab" "\xC4\xA0ab" "0,1 0,1 1,2"

(* Normalizing the text away leaves one empty span, and it stands for the whole
   of what was normalized rather than for the end of it. *)
let test_align_emptied () =
  let case label normalizer input expected =
    let text, alignment = Normalizer.apply_aligned normalizer input in
    equal ~msg:(Printf.sprintf "%s %S: text" label input) string "" text;
    let start, stop = Normalizer.original_span alignment ~start:0 ~stop:0 in
    equal
      ~msg:(Printf.sprintf "%s %S: span" label input)
      string expected
      (Printf.sprintf "%d,%d" start stop)
  in
  case "strip" (Normalizer.strip ()) "   " "0,3";
  case "bert" (Normalizer.bert ()) "\x00" "0,1";
  case "strip" (Normalizer.strip ()) "" "0,0";
  equal ~msg:"identity of empty text" string "0,0"
    (let start, stop =
       Normalizer.original_span (Normalizer.identity "") ~start:0 ~stop:0
     in
     Printf.sprintf "%d,%d" start stop)

let test_identity_alignment () =
  let a = Normalizer.identity "caf\xC3\xA9" in
  let span ~start ~stop =
    let s, e = Normalizer.original_span a ~start ~stop in
    Printf.sprintf "%d,%d" s e
  in
  equal ~msg:"whole" string "0,5" (span ~start:0 ~stop:5);
  equal ~msg:"ascii" string "0,1" (span ~start:0 ~stop:1);
  (* A span cutting a character short still reports it whole. *)
  equal ~msg:"partial character" string "3,5" (span ~start:3 ~stop:4);
  equal ~msg:"empty" string "2,2" (span ~start:2 ~stop:2);
  equal ~msg:"empty at end" string "5,5" (span ~start:5 ~stop:5);
  raises (Invalid_argument "6,7 is not a span of the 5 normalized bytes")
    (fun () -> Normalizer.original_span a ~start:6 ~stop:7);
  raises (Invalid_argument "0,6 is not a span of the 5 normalized bytes")
    (fun () -> Normalizer.original_span a ~start:0 ~stop:6)

let alignment_corpus =
  [
    "";
    "hello";
    " \t\n ";
    "caf\xC3\xA9 na\xC3\xAFve";
    "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E \xE6\xBC\xA2\xE5\xAD\x97";
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87";
    "\xD7\x91\xD6\xBC\xD6\xB8";
    "\xE1\xBA\x9B\xCC\xA3\xE1\xB8\x8D\xCC\x81a\xCC\x81\xCC\x96";
    "\xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA6";
    "Hello" ^ String.make 1 '\xFF' ^ String.make 1 '\xFE' ^ "World";
    "\x00\x07\x7F a\xC2\xADb";
    Fixture.read "fixtures/parity/edge_cases.txt";
    Fixture.read "fixtures/parity/sample.txt";
  ]

let alignment_normalizers =
  [
    ("nfc", Normalizer.nfc);
    ("nfd", Normalizer.nfd);
    ("nfkc", Normalizer.nfkc);
    ("nfkd", Normalizer.nfkd);
    ("lowercase", Normalizer.lowercase);
    ("strip_accents", Normalizer.strip_accents);
    ("strip", Normalizer.strip ());
    ("replace", Normalizer.replace ~pattern:"\\s+" ~replacement:" ");
    ("prepend", Normalizer.prepend "\xE2\x96\x81");
    ("byte_level", Normalizer.byte_level ~add_prefix_space:true ());
    ("bert", Normalizer.bert ());
    ( "llama",
      Normalizer.sequence
        [
          Normalizer.prepend "\xE2\x96\x81";
          Normalizer.replace ~pattern:" " ~replacement:"\xE2\x96\x81";
        ] );
  ]

(* Reporting where a byte came from goes through a different implementation of
   Unicode normalization than [apply] does, one that can account for every
   character; the two must agree on the text down to the byte. *)
let test_aligned_matches_apply () =
  List.iter
    (fun (name, normalizer) ->
      List.iteri
        (fun i text ->
          equal
            ~msg:(Printf.sprintf "%s on corpus %d" name i)
            string
            (Normalizer.apply normalizer text)
            (fst (Normalizer.apply_aligned normalizer text)))
        alignment_corpus)
    alignment_normalizers

(* Spans stay inside the input and never go backwards, so a token's span is well
   formed wherever it falls. *)
let test_alignment_is_monotonic () =
  List.iter
    (fun (name, normalizer) ->
      List.iteri
        (fun i text ->
          let normalized, alignment =
            Normalizer.apply_aligned normalizer text
          in
          let previous = ref 0 in
          for byte = 0 to String.length normalized - 1 do
            let start, stop =
              Normalizer.original_span alignment ~start:byte ~stop:(byte + 1)
            in
            if start > stop || stop > String.length text || start < !previous
            then
              failf "%s on corpus %d: byte %d reports %d,%d after %d" name i
                byte start stop !previous;
            previous := start
          done)
        alignment_corpus)
    alignment_normalizers

(* Integration with Tokenizer *)

let test_tokenize_with_normalization () =
  let text = "HELLO   WORLD!" in
  let normalizer =
    Normalizer.sequence
      [
        Normalizer.lowercase;
        Normalizer.replace ~pattern:"\\s+" ~replacement:" ";
      ]
  in
  let tokenizer =
    word_level ~normalizer
      ~pre:(Pre_tokenizer.whitespace ())
      ~vocab:[ ("hello", 0); ("world", 1); ("!", 2) ]
      ()
  in
  let tokens = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"normalized tokenization" (list string) [ "hello"; "world"; "!" ]
    tokens

let test_tokenize_unicode_words () =
  let text = "café résumé naïve" in
  let tokenizer =
    word_level
      ~pre:(Pre_tokenizer.whitespace ())
      ~vocab:[ ("café", 0); ("résumé", 1); ("naïve", 2) ]
      ()
  in
  let tokens = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"tokenized unicode" bool true (List.length tokens > 0)

let test_malformed_unicode () =
  let text = "Hello" ^ String.make 1 '\xFF' ^ String.make 1 '\xFE' ^ "World" in
  let tokenizer = chars () in
  let tokens = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"handled malformed" bool true (List.length tokens > 0)

(* Test Suite *)

let unicode_tests =
  [
    (* Normalization *)
    test "lowercase normalization" test_lowercase_normalization;
    test "lowercase is not case folding" test_lowercase_is_not_folding;
    test "strip accents normalization" test_strip_accents_normalization;
    test "strip accents keeps composition" test_strip_accents_keeps_composition;
    test "bert normalizer" test_bert_normalizer;
    test "normalization sequence" test_normalization_sequence;
    (* Alignment *)
    test "nfd alignment" test_align_nfd;
    test "nfc alignment" test_align_nfc;
    test "nfkc and nfkd alignment" test_align_nfkc_nfkd;
    test "text transform alignment" test_align_text_transforms;
    test "replace alignment" test_align_replace;
    test "prepend alignment" test_align_prepend;
    test "bert alignment" test_align_bert;
    test "sequence alignment" test_align_sequence;
    test "byte level alignment" test_align_byte_level;
    test "emptied text alignment" test_align_emptied;
    test "identity alignment" test_identity_alignment;
    test "aligned matches apply" test_aligned_matches_apply;
    test "alignment is monotonic" test_alignment_is_monotonic;
    (* Integration *)
    test "tokenize with normalization" test_tokenize_with_normalization;
    test "tokenize unicode words" test_tokenize_unicode_words;
    (* Error handling *)
    test "malformed unicode" test_malformed_unicode;
  ]

let () = run "brot unicode" [ group "unicode" unicode_tests ]
