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

(* A metaspace after a whitespace split marks each word on its own, and every
   token is placed through the alignment of the marking: a marker that was
   prepended stands at the character it opens. Every expectation is the output
   of HuggingFace on the same vocabulary. *)
let test_offsets_of_a_rewriting_pre_tokenizer () =
  let m = "\u{2581}" in
  let vocab =
    [
      ("<unk>", 0.0);
      (m, -3.0);
      (m ^ "a", -1.0);
      (m ^ "b", -1.0);
      ("a", -2.0);
      ("b", -2.0);
      ("!", -2.0);
      (m ^ "!", -1.5);
      ("ab", -2.5);
    ]
  in
  let tokenizer pre = unigram ~pre ~vocab ~unk_id:0 () in
  let case t text tokens offsets =
    let encoding = encode t ~add_special_tokens:false text in
    equal
      ~msg:(Printf.sprintf "%S: tokens" text)
      (list string) tokens
      (Array.to_list (Encoding.tokens encoding));
    equal
      ~msg:(Printf.sprintf "%S: offsets" text)
      (list (pair int int))
      offsets
      (Array.to_list (Encoding.offsets encoding))
  in
  let t5 =
    tokenizer
      (Pre_tokenizer.sequence
         [ Pre_tokenizer.whitespace_split (); Pre_tokenizer.metaspace () ])
  in
  case t5 "" [] [];
  case t5 "a b" [ m ^ "a"; m ^ "b" ] [ (0, 1); (2, 3) ];
  case t5 " a" [ m ^ "a" ] [ (1, 2) ];
  case t5 "a " [ m ^ "a" ] [ (0, 1) ];
  case t5 "a  b" [ m ^ "a"; m ^ "b" ] [ (0, 1); (3, 4) ];
  case t5 "ab" [ m ^ "a"; "b" ] [ (0, 1); (1, 2) ];
  case t5 "a!b" [ m ^ "a"; "!"; "b" ] [ (0, 1); (1, 2); (2, 3) ];
  case t5 (m ^ "a") [ m ^ "a" ] [ (0, 4) ];
  (* The unknown token stands for the whole run of characters no piece
     covers. *)
  case t5 "\u{65e5}\u{672c} \u{8a9e}" [ m; "<unk>"; m; "<unk>" ]
    [ (0, 3); (0, 6); (7, 10); (7, 10) ];
  let punctuated =
    tokenizer
      (Pre_tokenizer.sequence
         [
           Pre_tokenizer.whitespace_split ();
           Pre_tokenizer.metaspace ();
           Pre_tokenizer.punctuation ();
         ])
  in
  case punctuated "a!b" [ m ^ "a"; "!"; "b" ] [ (0, 1); (1, 2); (2, 3) ];
  case punctuated "b!" [ m ^ "b"; "!" ] [ (0, 1); (1, 2) ]

(* On [`First] the marker goes to the piece that opens the document alone: not
   to one that white space or an added token comes before, and not to the first
   piece of a document whose normalizer took away its opening bytes. Every
   expectation is the output of HuggingFace. *)
let test_prepend_first () =
  let m = "\u{2581}" in
  let vocab = words [ "[UNK]"; m ^ "a"; m ^ "b"; "a"; "b"; m ^ "ab"; "ab" ] in
  let first = Pre_tokenizer.metaspace ~prepend_scheme:`First () in
  let split_first =
    Pre_tokenizer.sequence [ Pre_tokenizer.whitespace_split (); first ]
  in
  let tokenizer ?normalizer ?(added_tokens = []) pre =
    word_level ?normalizer ~pre ~added_tokens ~vocab ()
  in
  let case t text tokens offsets =
    let encoding = encode t ~add_special_tokens:false text in
    equal
      ~msg:(Printf.sprintf "%S: tokens" text)
      (list string) tokens
      (Array.to_list (Encoding.tokens encoding));
    equal
      ~msg:(Printf.sprintf "%S: offsets" text)
      (list (pair int int))
      offsets
      (Array.to_list (Encoding.offsets encoding));
    equal
      ~msg:(Printf.sprintf "%S: encode_ids agrees" text)
      (array int) (Encoding.ids encoding)
      (encode_ids t ~add_special_tokens:false text)
  in
  let added = tokenizer ~added_tokens:[ added_token "<s>" ] split_first in
  case added "a b" [ m ^ "a"; "b" ] [ (0, 1); (2, 3) ];
  case added " a b" [ "a"; "b" ] [ (1, 2); (3, 4) ];
  case added "a<s>b" [ m ^ "a"; "<s>"; "b" ] [ (0, 1); (1, 4); (4, 5) ];
  case added "<s>b" [ "<s>"; "b" ] [ (0, 3); (3, 4) ];
  case added " a<s>b" [ "a"; "<s>"; "b" ] [ (1, 2); (2, 5); (5, 6) ];
  case added "a <s> b" [ m ^ "a"; "<s>"; "b" ] [ (0, 1); (2, 5); (6, 7) ];
  let alone = tokenizer ~added_tokens:[ added_token "<s>" ] first in
  case alone "a<s>b" [ m ^ "a"; "<s>"; "b" ] [ (0, 1); (1, 4); (4, 5) ];
  case alone "<s>b" [ "<s>"; "b" ] [ (0, 3); (3, 4) ];
  let stripped = tokenizer ~normalizer:(Normalizer.strip ()) split_first in
  case stripped "  a b" [ "a"; "b" ] [ (2, 3); (4, 5) ];
  case stripped "a b" [ m ^ "a"; "b" ] [ (0, 1); (2, 3) ];
  let lowered = tokenizer ~normalizer:Normalizer.lowercase split_first in
  case lowered "A b" [ m ^ "a"; "b" ] [ (0, 1); (2, 3) ];
  (* An added token matched on the normalized text opens the document just as
     one matched on the raw text does. *)
  let normalized =
    tokenizer
      ~added_tokens:[ added_token ~normalized:true ~special:false "[X]" ]
      first
  in
  case normalized "[X]b" [ "[X]"; "b" ] [ (0, 3); (3, 4) ];
  case normalized "b[X]a" [ m ^ "b"; "[X]"; "a" ] [ (0, 1); (1, 4); (4, 5) ];
  (* The pieces path — here a fixed-length member the walkers cannot take —
     follows the same rule, added tokens and normalizer included. *)
  let fixed =
    Pre_tokenizer.sequence
      [ Pre_tokenizer.whitespace_split (); Pre_tokenizer.fixed_length 2; first ]
  in
  let pieces = tokenizer ~added_tokens:[ added_token "<s>" ] fixed in
  case pieces "ab a" [ m ^ "ab"; "a" ] [ (0, 2); (3, 4) ];
  case pieces "ab<s>ab" [ m ^ "ab"; "<s>"; "ab" ] [ (0, 2); (2, 5); (5, 7) ];
  case pieces "<s>ab" [ "<s>"; "ab" ] [ (0, 3); (3, 5) ];
  case pieces " ab" [ "ab" ] [ (1, 3) ];
  let stripped_pieces = tokenizer ~normalizer:(Normalizer.strip ()) fixed in
  case stripped_pieces "  ab a" [ "ab"; "a" ] [ (2, 4); (5, 6) ];
  case stripped_pieces "ab a" [ m ^ "ab"; "a" ] [ (0, 2); (3, 4) ]

(* A model other than BPE cannot match raw bytes, so behind a byte-level
   pre-tokenizer it is handed the encoded pieces and matches its vocabulary in
   byte-level form, as HuggingFace does. Every expectation is HuggingFace's but
   the one marked. *)
let test_byte_level_behind_a_non_bpe_model () =
  let g = "\u{120}" in
  let case t text tokens offsets =
    let encoding = encode t ~add_special_tokens:false text in
    equal
      ~msg:(Printf.sprintf "%S: tokens" text)
      (list string) tokens
      (Array.to_list (Encoding.tokens encoding));
    equal
      ~msg:(Printf.sprintf "%S: offsets" text)
      (list (pair int int))
      offsets
      (Array.to_list (Encoding.offsets encoding));
    equal
      ~msg:(Printf.sprintf "%S: encode_ids agrees" text)
      (array int) (Encoding.ids encoding)
      (encode_ids t ~add_special_tokens:false text)
  in
  let split_then_bytes =
    Pre_tokenizer.sequence
      [ Pre_tokenizer.whitespace_split (); Pre_tokenizer.byte_level () ]
  in
  let vocab =
    [
      ("<unk>", 0.0);
      (g ^ "a", -1.0);
      (g ^ "b", -1.0);
      ("a", -2.0);
      ("b", -2.0);
      (g ^ "ab", -1.5);
    ]
  in
  let unigram = unigram ~pre:split_then_bytes ~vocab ~unk_id:0 () in
  case unigram "a b" [ g ^ "a"; g ^ "b" ] [ (0, 1); (2, 3) ];
  case unigram "ab b" [ g ^ "ab"; g ^ "b" ] [ (0, 2); (3, 4) ];
  case unigram " a" [ g ^ "a" ] [ (1, 2) ];
  let wordpiece =
    wordpiece ~pre:split_then_bytes
      ~vocab:(words [ "[UNK]"; g ^ "a"; g ^ "b"; "##b"; g ^ "ab" ])
      ~unk_token:"[UNK]" ()
  in
  case wordpiece "a b" [ g ^ "a"; g ^ "b" ] [ (0, 1); (2, 3) ];
  (* The pieces of an encoding pre-tokenizer are placed whole, so the two tokens
     of one share its span; HuggingFace has [(0, 2)] and [(2, 3)] here. *)
  case wordpiece "abb" [ g ^ "ab"; "##b" ] [ (0, 3); (0, 3) ];
  let word_level =
    word_level ~pre:split_then_bytes
      ~vocab:(words [ "[UNK]"; g ^ "a"; g ^ "b"; g ^ "ab" ])
      ()
  in
  case word_level "a b" [ g ^ "a"; g ^ "b" ] [ (0, 1); (2, 3) ];
  case word_level "ab b" [ g ^ "ab"; g ^ "b" ] [ (0, 2); (3, 4) ];
  (* Alone, and without its prefix space, the byte-level pre-tokenizer is walked
     for BPE; the other models are still handed pieces. *)
  let alone =
    Brot.unigram ~pre:(Pre_tokenizer.byte_level ()) ~vocab ~unk_id:0 ()
  in
  case alone "a b" [ g ^ "a"; g ^ "b" ] [ (0, 1); (1, 3) ];
  let plain =
    Brot.unigram
      ~pre:
        (Pre_tokenizer.sequence
           [
             Pre_tokenizer.whitespace_split ();
             Pre_tokenizer.byte_level ~add_prefix_space:false ();
           ])
      ~vocab:[ ("<unk>", 0.0); ("a", -1.0); ("b", -1.0); ("ab", -1.5) ]
      ~unk_id:0 ()
  in
  case plain "a b" [ "a"; "b" ] [ (0, 1); (2, 3) ];
  case plain "ab b" [ "ab"; "b" ] [ (0, 2); (3, 4) ]

(* A run of characters no entry covers is one unknown token — or, with fusing
   off, one per character — that stands for the whole run: the tokens after it
   are placed where they are, not one character on. Every expectation is the
   output of HuggingFace. *)
let test_offsets_of_unknown_runs () =
  let zw = "\u{200b}\u{200c}" in
  let text = "a" ^ zw ^ "b" ^ zw ^ "c" in
  let case t text tokens offsets =
    let encoding = encode t ~add_special_tokens:false text in
    equal
      ~msg:(Printf.sprintf "%S: tokens" text)
      (list string) tokens
      (Array.to_list (Encoding.tokens encoding));
    equal
      ~msg:(Printf.sprintf "%S: offsets" text)
      (list (pair int int))
      offsets
      (Array.to_list (Encoding.offsets encoding))
  in
  let fused = [ (0, 1); (1, 7); (7, 8); (8, 14); (14, 15) ] in
  (* HuggingFace spells a Unigram unknown token as the run itself, where brot
     reports the entry: the token strings are not compared here. *)
  let fusing =
    unigram
      ~vocab:[ ("<unk>", 0.0); ("a", -1.0); ("b", -1.0); ("c", -1.0) ]
      ~unk_id:0 ()
  in
  let encoding = encode fusing ~add_special_tokens:false text in
  equal ~msg:"unigram: ids" (list int) [ 1; 0; 2; 0; 3 ]
    (Array.to_list (Encoding.ids encoding));
  equal ~msg:"unigram: offsets"
    (list (pair int int))
    fused
    (Array.to_list (Encoding.offsets encoding));
  let bpe ~fuse_unk =
    bpe
      ~vocab:[ ("<unk>", 0); ("a", 1); ("b", 2); ("c", 3) ]
      ~merges:[] ~unk_token:"<unk>" ~fuse_unk ()
  in
  case (bpe ~fuse_unk:true) text [ "a"; "<unk>"; "b"; "<unk>"; "c" ] fused;
  case (bpe ~fuse_unk:false) text
    [ "a"; "<unk>"; "<unk>"; "b"; "<unk>"; "<unk>"; "c" ]
    [ (0, 1); (1, 4); (4, 7); (7, 8); (8, 11); (11, 14); (14, 15) ];
  (* The unknown token as a literal match of its own entry is placed as any
     entry, whether alone or in a fused run that spells one. *)
  let literal =
    unigram ~vocab:[ ("<unk>", 0.0); ("a", -1.0); ("b", -1.0) ] ~unk_id:0 ()
  in
  case literal "a<unk>b" [ "a"; "<unk>"; "b" ] [ (0, 1); (1, 6); (6, 7) ];
  case literal
    ("a<unk>" ^ zw ^ "b")
    [ "a"; "<unk>"; "b" ]
    [ (0, 1); (1, 12); (12, 13) ];
  let split = Pre_tokenizer.whitespace_split () in
  case
    (wordpiece ~pre:split
       ~vocab:(words [ "[UNK]"; "a"; "b"; "##b" ])
       ~unk_token:"[UNK]" ())
    "ab xyz ba"
    [ "a"; "##b"; "[UNK]"; "[UNK]" ]
    [ (0, 1); (1, 2); (3, 6); (7, 9) ];
  case
    (word_level ~pre:split ~vocab:(words [ "<unk>"; "a" ]) ())
    "a xyz <unk>" [ "a"; "<unk>"; "<unk>" ]
    [ (0, 1); (2, 5); (6, 11) ]

(* A byte-level member after a whitespace split prepends its space to each word
   and the model is handed each on its own; the space stands at the character it
   opens. Every expectation is the output of HuggingFace. *)
let test_offsets_of_byte_level_after_a_walker () =
  let tokenizer =
    bpe
      ~pre:
        (Pre_tokenizer.sequence
           [ Pre_tokenizer.whitespace_split (); Pre_tokenizer.byte_level () ])
      ~vocab:
        [ ("\u{120}a", 0); ("\u{120}b", 1); ("a", 2); ("b", 3); ("\u{120}", 4) ]
      ~merges:[] ()
  in
  let case text tokens offsets =
    let encoding = encode tokenizer ~add_special_tokens:false text in
    equal
      ~msg:(Printf.sprintf "%S: tokens" text)
      (list string) tokens
      (Array.to_list (Encoding.tokens encoding));
    equal
      ~msg:(Printf.sprintf "%S: offsets" text)
      (list (pair int int))
      offsets
      (Array.to_list (Encoding.offsets encoding))
  in
  let g = "\u{120}" in
  case "a b" [ g; "a"; g; "b" ] [ (0, 1); (0, 1); (2, 3); (2, 3) ];
  case " a  b" [ g; "a"; g; "b" ] [ (1, 2); (1, 2); (4, 5); (4, 5) ];
  case "ab" [ g; "a"; "b" ] [ (0, 1); (0, 1); (1, 2) ]

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
    test "prepend on first" test_prepend_first;
    test "byte level behind a non-BPE model"
      test_byte_level_behind_a_non_bpe_model;
    test "offsets of unknown runs" test_offsets_of_unknown_runs;
    test "offsets of byte level after a walker"
      test_offsets_of_byte_level_after_a_walker;
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
