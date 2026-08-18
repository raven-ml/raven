(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Brot

let make_word_tokenizer ?(added_tokens = []) ?(vocab = []) () =
  word_level ~pre:(Pre_tokenizer.whitespace ()) ~added_tokens ~vocab ()

let words = List.mapi (fun id word -> (word, id))

let test_encode_simple () =
  let tokenizer = make_word_tokenizer ~vocab:(words [ "hello"; "world" ]) () in
  let ids = encode tokenizer "hello world hello" |> Encoding.ids in
  equal ~msg:"encoded length" int 3 (Array.length ids);
  equal ~msg:"repeated token same id" bool true (ids.(0) = ids.(2))

let test_encode_with_vocab () =
  let tokenizer = make_word_tokenizer ~vocab:(words [ "hello"; "world" ]) () in
  let ids = encode tokenizer "hello world" |> Encoding.ids |> Array.to_list in
  equal ~msg:"encoded with vocab" (list int) [ 0; 1 ] ids

let test_encode_unknown_tokens () =
  let tokenizer =
    make_word_tokenizer
      ~added_tokens:[ added_token "<unk>" ]
      ~vocab:(words [ "hello" ]) ()
  in
  let ids =
    encode tokenizer "hello unknown world" |> Encoding.ids |> Array.to_list
  in
  equal ~msg:"encoded something" bool true (List.length ids > 0)

let test_encode_empty () =
  let tokenizer = make_word_tokenizer () in
  let ids = encode tokenizer "" |> Encoding.ids |> Array.to_list in
  equal ~msg:"encode empty" (list int) [] ids

let test_encode_batch_simple () =
  let tokenizer =
    make_word_tokenizer ~vocab:(words [ "hello"; "world"; "hi"; "there" ]) ()
  in
  let encodings = encode_batch tokenizer [ "hello world"; "hi there" ] in
  equal ~msg:"batch size" int 2 (List.length encodings);
  let first = List.hd encodings in
  equal ~msg:"first encoding has ids" bool true
    (Array.length (Encoding.ids first) > 0)

let test_encode_batch_with_padding () =
  let tokenizer =
    make_word_tokenizer
      ~added_tokens:[ added_token "<pad>" ]
      ~vocab:(words [ "hello"; "world"; "hi"; "there" ])
      ()
  in
  let padding =
    {
      length = `Fixed 5;
      direction = `Right;
      pad_id = None;
      pad_type_id = None;
      pad_token = Some "<pad>";
    }
  in
  let encodings = encode_batch tokenizer ~padding [ "hello"; "hi there" ] in
  let first = Encoding.ids (List.nth encodings 0) in
  let second = Encoding.ids (List.nth encodings 1) in
  equal ~msg:"first padded length" int 5 (Array.length first);
  equal ~msg:"second padded length" int 5 (Array.length second)

let test_encode_batch_empty () =
  let tokenizer = make_word_tokenizer () in
  let encodings = encode_batch tokenizer [] in
  equal ~msg:"empty batch" int 0 (List.length encodings)

let test_decode_simple () =
  let tokenizer = make_word_tokenizer ~vocab:(words [ "hello"; "world" ]) () in
  let decoded = decode tokenizer [| 0; 1 |] in
  equal ~msg:"decoded text" string "hello world" decoded

let test_decode_with_special () =
  let tokenizer =
    make_word_tokenizer
      ~added_tokens:[ added_token "<bos>"; added_token "<eos>" ]
      ~vocab:[ ("<bos>", 0); ("<eos>", 1); ("hello", 2) ]
      ()
  in
  (* <bos>=0, <eos>=1, hello=2 *)
  let decoded = decode tokenizer [| 0; 2; 1 |] in
  equal ~msg:"decoded with special" string "<bos> hello <eos>" decoded

let test_decode_skip_special () =
  let tokenizer =
    make_word_tokenizer
      ~added_tokens:[ added_token "<bos>"; added_token "<eos>" ]
      ~vocab:[ ("<bos>", 0); ("<eos>", 1); ("hello", 2) ]
      ()
  in
  let decoded = decode ~skip_special_tokens:true tokenizer [| 0; 2; 1 |] in
  equal ~msg:"decoded without special" string "hello" decoded

let test_decode_batch () =
  let tokenizer =
    make_word_tokenizer ~vocab:(words [ "hello"; "world"; "hi"; "there" ]) ()
  in
  let decoded = decode_batch tokenizer [ [| 0; 1 |]; [| 2; 3 |] ] in
  equal ~msg:"decoded count" int 2 (List.length decoded);
  equal ~msg:"first decoded" string "hello world" (List.nth decoded 0);
  equal ~msg:"second decoded" string "hi there" (List.nth decoded 1)

let test_chars_model () =
  let tokenizer = chars () in
  let ids = encode tokenizer "abc" |> Encoding.ids |> Array.to_list in
  equal ~msg:"char ids" (list int) [ 97; 98; 99 ] ids

(* Offsets and word ids.

   Every expectation below was read off HuggingFace [tokenizers] first. *)

let offsets_of tokenizer text =
  encode tokenizer ~add_special_tokens:false text
  |> Encoding.offsets |> Array.to_list

let words_of tokenizer text =
  encode tokenizer ~add_special_tokens:false text
  |> Encoding.word_ids |> Array.to_list

(* Normalizing moves the text around, and offsets are reported on the text as it
   was given: ["café"] normalizes to ["cafe"], four bytes standing for five. *)
let accent_free =
  Normalizer.sequence [ Normalizer.nfd; Normalizer.strip_accents ]

let test_offsets_through_normalizer () =
  let tokenizer =
    word_level ~normalizer:accent_free
      ~pre:(Pre_tokenizer.whitespace ())
      ~vocab:(words [ "cafe"; "x" ])
      ()
  in
  equal ~msg:"ids" (array int) [| 0; 1 |]
    (Encoding.ids (encode tokenizer ~add_special_tokens:false "café x"));
  equal ~msg:"offsets"
    (list (pair int int))
    [ (0, 5); (6, 7) ]
    (offsets_of tokenizer "café x")

(* A byte-level pre-tokenizer hands the model the raw bytes of a pretoken, so
   its spans are those of the input and the space it marks as [Ġ] is one byte of
   it. *)
let test_offsets_byte_level () =
  let tokenizer =
    bpe
      ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
      ~vocab:[ ("a", 0); ("Ġb", 1); ("Ġ", 2); ("b", 3) ]
      ~merges:[ ("Ġ", "b") ]
      ()
  in
  equal ~msg:"ids" (array int) [| 0; 1 |]
    (Encoding.ids (encode tokenizer ~add_special_tokens:false "a b"));
  equal ~msg:"offsets"
    (list (pair int int))
    [ (0, 1); (1, 3) ]
    (offsets_of tokenizer "a b")

(* Word ids number the pretokens of a text from zero. An added token is one
   pretoken of its own, and takes the span it matched. *)
let test_word_ids () =
  let tokenizer =
    make_word_tokenizer
      ~added_tokens:[ added_token "<s>" ]
      ~vocab:(words [ "a"; "b" ])
      ()
  in
  equal ~msg:"word ids"
    (list (option int))
    [ Some 0; Some 1; Some 2 ]
    (words_of tokenizer "a <s> b");
  equal ~msg:"offsets"
    (list (pair int int))
    [ (0, 1); (2, 5); (6, 7) ]
    (offsets_of tokenizer "a <s> b")

(* The tokens a post-processor inserts belong to no word and cover nothing. *)
let test_word_ids_of_special_tokens () =
  let tokenizer =
    word_level
      ~pre:(Pre_tokenizer.whitespace ())
      ~post:
        (Post_processor.template ~single:"[CLS] $A"
           ~special_tokens:[ ("[CLS]", 2) ]
           ())
      ~vocab:(words [ "a"; "b" ])
      ()
  in
  let encoding = encode tokenizer "a b" in
  equal ~msg:"word ids"
    (list (option int))
    [ None; Some 0; Some 1 ]
    (Array.to_list (Encoding.word_ids encoding));
  equal ~msg:"offsets"
    (list (pair int int))
    [ (0, 0); (0, 1); (2, 3) ]
    (Array.to_list (Encoding.offsets encoding))

(* A pre-tokenizer that hands back text of its own rather than ranges of what it
   was given may report a range of that text instead, which runs past the end of
   the input. Offsets are reported for every token whatever it says, and always
   land inside the text that was encoded. *)
let test_offsets_of_a_rewriting_pre_tokenizer () =
  let tokenizer =
    unigram
      ~pre:
        (Pre_tokenizer.sequence
           [ Pre_tokenizer.whitespace_split (); Pre_tokenizer.metaspace () ])
      ~vocab:
        [ ("<unk>", 0.0); ("▁", -1.0); ("▁a", 0.0); ("▁b", 0.0); ("a", 0.0) ]
      ()
  in
  List.iter
    (fun text ->
      let encoding = encode tokenizer ~add_special_tokens:false text in
      let offsets = Encoding.offsets encoding in
      equal
        ~msg:(Printf.sprintf "%S: one span per token" text)
        int (Encoding.length encoding) (Array.length offsets);
      Array.iteri
        (fun i (start, stop) ->
          equal
            ~msg:(Printf.sprintf "%S: span %d is inside the text" text i)
            bool true
            (0 <= start && start <= stop && stop <= String.length text))
        offsets)
    [ ""; "a"; "a b"; " a"; "a "; "a  b"; "日本 語" ]

let test_encode_ids_matches_encode () =
  let tokenizer =
    word_level ~normalizer:accent_free
      ~pre:(Pre_tokenizer.whitespace ())
      ~vocab:(words [ "cafe"; "x" ])
      ()
  in
  List.iter
    (fun text ->
      equal ~msg:(Printf.sprintf "%S" text) (array int)
        (Encoding.ids (encode tokenizer ~add_special_tokens:false text))
        (encode_ids tokenizer ~add_special_tokens:false text))
    [ ""; "x"; "café x"; "  x  "; "café café x" ]

(* Truncation.

   HuggingFace truncates before the post-processor runs, so the tokens it adds
   fit inside [max_length] instead of pushing content out. A pair gives up
   tokens from whichever of the two is longer, the first one on a tie. *)

let bounded_tokenizer () =
  word_level
    ~pre:(Pre_tokenizer.whitespace ())
    ~post:(Post_processor.bert ~cls:("[CLS]", 8) ~sep:("[SEP]", 9) ())
    ~vocab:(words [ "a"; "b"; "c"; "d"; "e"; "f"; "[PAD]" ])
    ~pad_token:"[PAD]" ()

let test_truncation_budgets_special_tokens () =
  let tokenizer = bounded_tokenizer () in
  let ids ?(add_special_tokens = true) max_length =
    Encoding.ids
      (encode tokenizer ~add_special_tokens ~truncation:(truncation max_length)
         "a b c d e")
  in
  equal ~msg:"[CLS] and [SEP] come out of the budget" (array int)
    [| 8; 0; 1; 9 |] (ids 4);
  equal ~msg:"without them the budget is all content" (array int)
    [| 0; 1; 2; 3 |]
    (ids ~add_special_tokens:false 4);
  equal ~msg:"a budget the text fits in changes nothing" (array int)
    [| 8; 0; 1; 2; 3; 4; 9 |] (ids 32)

(* Four tokens against three for a budget of five: the shorter sequence cannot
   keep all three, so the budget is halved and the odd token goes to the first.
   Removing one token at a time from the longer would give the odd token to the
   second instead, which is what pins the rule rather than the arithmetic. *)
let test_truncation_of_a_pair () =
  let tokenizer = bounded_tokenizer () in
  let ids max_length =
    Encoding.ids
      (encode tokenizer ~truncation:(truncation max_length) ~pair:"e f a"
         "a b c d")
  in
  equal ~msg:"three against two" (array int)
    [| 8; 0; 1; 2; 9; 4; 5; 9 |]
    (ids 8);
  equal ~msg:"one each" (array int) [| 8; 0; 9; 4; 9 |] (ids 5)

(* An overflowing window is a sequence of its own, so it carries the same
   special tokens as the one it was cut from. *)
let test_truncation_overflowing () =
  let tokenizer = bounded_tokenizer () in
  let encoding = encode tokenizer ~truncation:(truncation 4) "a b c d e f" in
  equal ~msg:"first window" (array int) [| 8; 0; 1; 9 |] (Encoding.ids encoding);
  equal ~msg:"the rest, each wrapped in turn"
    (list (array int))
    [ [| 8; 2; 3; 9 |]; [| 8; 4; 5; 9 |] ]
    (List.map Encoding.ids (Encoding.overflowing encoding));
  equal ~msg:"a window's mask" (array int) [| 1; 0; 0; 1 |]
    (Encoding.special_tokens_mask (List.hd (Encoding.overflowing encoding)))

(* [encode_ids] skips the encoding when the post-processor only wraps the
   sequence, so it has to answer exactly what reading the ids off one does. *)
let test_encode_ids_matches_encode_with_specials () =
  let tokenizer = bounded_tokenizer () in
  List.iter
    (fun (label, truncation, padding) ->
      List.iter
        (fun text ->
          equal
            ~msg:(Printf.sprintf "%s %S" label text)
            (array int)
            (Encoding.ids (encode tokenizer ?truncation ?padding text))
            (encode_ids tokenizer ?truncation ?padding text))
        [ ""; "a"; "a b c"; "a b c d e f"; "a zz b" ])
    [
      ("plain", None, None);
      ("truncated", Some (truncation 4), None);
      ("padded", None, Some (padding (`Fixed 8)));
      ("both", Some (truncation 4), Some (padding (`Fixed 8)));
    ]

(* Truncating from the left keeps the {e last} tokens, which is what HuggingFace
   does; the windows that did not fit run leftwards from there. *)
let test_truncation_from_the_left () =
  let tokenizer =
    make_word_tokenizer ~vocab:(words [ "a"; "b"; "c"; "d" ]) ()
  in
  let truncation = truncation ~direction:`Left 2 in
  let encoding = encode tokenizer ~truncation "a b c d" in
  equal ~msg:"the last two" (array int) [| 2; 3 |] (Encoding.ids encoding);
  equal ~msg:"and the rest, rightmost first"
    (list (array int))
    [ [| 0; 1 |] ]
    (List.map Encoding.ids (Encoding.overflowing encoding));
  equal ~msg:"encode_ids agrees" (array int) [| 2; 3 |]
    (encode_ids tokenizer ~truncation "a b c d")

(* An unknown token stands for whatever the identifiers around it do not
   describe, so one in the middle of a pretoken takes the bytes between its
   neighbours rather than none at all. *)
let test_offsets_of_an_unknown_token () =
  let tokenizer =
    bpe
      ~vocab:[ ("<unk>", 100); ("<0x61>", 0); ("b", 1) ]
      ~merges:[] ~unk_token:"<unk>" ~byte_fallback:true ()
  in
  equal ~msg:"tokens" (array string)
    [| "<0x61>"; "<unk>"; "b" |]
    (Encoding.tokens (encode tokenizer ~add_special_tokens:false "zab"));
  equal ~msg:"offsets"
    (list (pair int int))
    [ (0, 1); (1, 2); (2, 3) ]
    (offsets_of tokenizer "zab")

let test_truncation_without_a_post_processor () =
  let tokenizer = make_word_tokenizer ~vocab:(words [ "a"; "b"; "c" ]) () in
  equal ~msg:"the whole budget is content" (array int) [| 0; 1 |]
    (Encoding.ids (encode tokenizer ~truncation:(truncation 2) "a b c"))

let suite =
  [
    test "encode simple" test_encode_simple;
    test "encode with vocab" test_encode_with_vocab;
    test "encode unknown tokens" test_encode_unknown_tokens;
    test "encode empty" test_encode_empty;
    test "batch simple" test_encode_batch_simple;
    test "batch with padding" test_encode_batch_with_padding;
    test "batch empty request" test_encode_batch_empty;
    test "decode simple" test_decode_simple;
    test "decode with special" test_decode_with_special;
    test "decode skip special" test_decode_skip_special;
    test "decode batch" test_decode_batch;
    test "chars model" test_chars_model;
    test "offsets through a normalizer" test_offsets_through_normalizer;
    test "offsets of a byte-level pipeline" test_offsets_byte_level;
    test "word ids" test_word_ids;
    test "word ids of special tokens" test_word_ids_of_special_tokens;
    test "offsets of a rewriting pre-tokenizer"
      test_offsets_of_a_rewriting_pre_tokenizer;
    test "encode_ids matches encode" test_encode_ids_matches_encode;
    test "truncation budgets the special tokens"
      test_truncation_budgets_special_tokens;
    test "truncation of a pair" test_truncation_of_a_pair;
    test "truncation from the left" test_truncation_from_the_left;
    test "offsets of an unknown token" test_offsets_of_an_unknown_token;
    test "truncation keeps the overflowing windows" test_truncation_overflowing;
    test "encode_ids matches encode with special tokens"
      test_encode_ids_matches_encode_with_specials;
    test "truncation without a post-processor"
      test_truncation_without_a_post_processor;
  ]

let () = run "Encoding tests" [ group "encoding" suite ]
