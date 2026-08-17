(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Brot

let make_encoding ~ids ~tokens ?offsets ~type_id () =
  let len = Array.length ids in
  let offsets =
    match offsets with
    | Some offsets -> Array.copy offsets
    | None -> Array.make len (0, 0)
  in
  Encoding.create ~ids:(Array.copy ids) ~type_ids:(Array.make len type_id)
    ~tokens:(Array.copy tokens) ~words:(Array.make len None) ~offsets
    ~special_tokens_mask:(Array.make len 0) ~attention_mask:(Array.make len 1)
    ()

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let test_template_multi_special () =
  let processor =
    Result.get_ok
      (Post_processor.of_json
         (json_obj
            [
              ("type", Jsont.Json.string "TemplateProcessing");
              ( "single",
                Jsont.Json.list
                  [
                    json_obj
                      [
                        ( "SpecialToken",
                          json_obj
                            [
                              ("id", Jsont.Json.string "<multi>");
                              ("type_id", Jsont.Json.int 2);
                            ] );
                      ];
                    json_obj
                      [
                        ( "Sequence",
                          json_obj
                            [
                              ("id", Jsont.Json.string "A");
                              ("type_id", Jsont.Json.int 0);
                            ] );
                      ];
                  ] );
              ("pair", Jsont.Json.null ());
              ( "special_tokens",
                json_obj
                  [
                    ( "<multi>",
                      json_obj
                        [
                          ("id", Jsont.Json.string "<multi>");
                          ( "ids",
                            Jsont.Json.list
                              [ Jsont.Json.int 100; Jsont.Json.int 101 ] );
                          ( "tokens",
                            Jsont.Json.list
                              [
                                Jsont.Json.string "<m1>";
                                Jsont.Json.string "<m2>";
                              ] );
                        ] );
                  ] );
            ]))
  in
  let base = make_encoding ~ids:[| 10 |] ~tokens:[| "hello" |] ~type_id:0 () in
  let encoding =
    Post_processor.process processor base ~add_special_tokens:true
  in
  equal ~msg:"ids" (array int) [| 100; 101; 10 |] (Encoding.ids encoding);
  equal ~msg:"tokens" (array string)
    [| "<m1>"; "<m2>"; "hello" |]
    (Encoding.tokens encoding);
  equal ~msg:"type ids" (array int) [| 2; 2; 0 |] (Encoding.type_ids encoding);
  equal ~msg:"special mask" (array int) [| 1; 1; 0 |]
    (Encoding.special_tokens_mask encoding);
  equal ~msg:"attention mask" (array int) [| 1; 1; 1 |]
    (Encoding.attention_mask encoding);
  equal ~msg:"added tokens single" int 2
    (Post_processor.added_tokens processor ~is_pair:false)

let test_template_pair_type_ids () =
  let processor =
    Post_processor.template ~single:"$A [SEP]"
      ~pair:"[CLS]:0 $A:0 [SEP]:0 $B:3 [SEP]:3"
      ~special_tokens:[ ("[CLS]", 101); ("[SEP]", 102) ]
      ()
  in
  let seq_a =
    make_encoding ~ids:[| 10; 11 |] ~tokens:[| "hello"; "world" |] ~type_id:0 ()
  in
  let seq_b = make_encoding ~ids:[| 20 |] ~tokens:[| "pair" |] ~type_id:1 () in
  let encoding =
    Post_processor.process processor ~pair:seq_b seq_a ~add_special_tokens:true
  in
  equal ~msg:"pair ids" (array int)
    [| 101; 10; 11; 102; 20; 102 |]
    (Encoding.ids encoding);
  equal ~msg:"pair tokens" (array string)
    [| "[CLS]"; "hello"; "world"; "[SEP]"; "pair"; "[SEP]" |]
    (Encoding.tokens encoding);
  equal ~msg:"pair type ids" (array int) [| 0; 0; 0; 0; 3; 3 |]
    (Encoding.type_ids encoding);
  equal ~msg:"pair special mask" (array int) [| 1; 0; 0; 1; 0; 1 |]
    (Encoding.special_tokens_mask encoding);
  equal ~msg:"added tokens pair" int 3
    (Post_processor.added_tokens processor ~is_pair:true);
  (* Without special tokens the template still orders the sequences and sets
     their type ids; only its special pieces are dropped. *)
  let no_special =
    Post_processor.process processor ~pair:seq_b seq_a ~add_special_tokens:false
  in
  equal ~msg:"no-special ids" (array int) [| 10; 11; 20 |]
    (Encoding.ids no_special);
  equal ~msg:"no-special type ids" (array int) [| 0; 0; 3 |]
    (Encoding.type_ids no_special);
  equal ~msg:"no-special mask" (array int) [| 0; 0; 0 |]
    (Encoding.special_tokens_mask no_special)

(* A template built without a pair still processes a pair, with the template
   HuggingFace fills in: [$A:0 $B:1]. *)
let test_template_default_pair () =
  let processor = Post_processor.template ~single:"$A" () in
  let seq_a = make_encoding ~ids:[| 10 |] ~tokens:[| "hello" |] ~type_id:0 () in
  let seq_b = make_encoding ~ids:[| 20 |] ~tokens:[| "pair" |] ~type_id:0 () in
  let encoding =
    Post_processor.process processor ~pair:seq_b seq_a ~add_special_tokens:true
  in
  equal ~msg:"ids" (array int) [| 10; 20 |] (Encoding.ids encoding);
  equal ~msg:"type ids" (array int) [| 0; 1 |] (Encoding.type_ids encoding);
  equal ~msg:"added tokens" int 0
    (Post_processor.added_tokens processor ~is_pair:true)

(* Without special tokens the template still decides the order of the sequences:
   HuggingFace with pair template ["$B:1 $A:0"] answers [tokens = ['pair';
   'hello'; 'world']] and [type_ids = [1; 0; 0]]. *)
let test_template_pair_order () =
  let processor = Post_processor.template ~single:"$A" ~pair:"$B:1 $A:0" () in
  let seq_a =
    make_encoding ~ids:[| 10; 11 |] ~tokens:[| "hello"; "world" |] ~type_id:0 ()
  in
  let seq_b = make_encoding ~ids:[| 20 |] ~tokens:[| "pair" |] ~type_id:0 () in
  let encoding =
    Post_processor.process processor ~pair:seq_b seq_a ~add_special_tokens:false
  in
  equal ~msg:"ids" (array int) [| 20; 10; 11 |] (Encoding.ids encoding);
  equal ~msg:"type ids" (array int) [| 1; 0; 0 |] (Encoding.type_ids encoding)

(* A template left with a single sequence is that sequence, retyped: its
   offsets, masks and words come through untouched. *)
let test_template_single_sequence () =
  let processor =
    Post_processor.template ~single:"[CLS] $A:2"
      ~special_tokens:[ ("[CLS]", 101) ]
      ()
  in
  let seq_a =
    make_encoding ~ids:[| 10; 11 |] ~tokens:[| "hello"; "world" |]
      ~offsets:[| (0, 5); (6, 11) |]
      ~type_id:0 ()
  in
  let encoding =
    Post_processor.process processor seq_a ~add_special_tokens:false
  in
  equal ~msg:"ids" (array int) [| 10; 11 |] (Encoding.ids encoding);
  equal ~msg:"type ids" (array int) [| 2; 2 |] (Encoding.type_ids encoding);
  equal ~msg:"offsets"
    (array (pair int int))
    [| (0, 5); (6, 11) |]
    (Encoding.offsets encoding);
  equal ~msg:"special mask" (array int) [| 0; 0 |]
    (Encoding.special_tokens_mask encoding)

(* [process] stamps the pair as the second segment whatever type ids it arrives
   with, leaves the first sequence's alone, and shifts no offset: each sequence
   keeps the offsets into its own text. *)
let test_pair_segments_and_offsets () =
  let seq_a =
    make_encoding ~ids:[| 10; 11 |] ~tokens:[| "hello"; "world" |]
      ~offsets:[| (0, 5); (6, 11) |]
      ~type_id:0 ()
  in
  let seq_b =
    make_encoding ~ids:[| 20 |] ~tokens:[| "pair" |]
      ~offsets:[| (0, 4) |]
      ~type_id:7 ()
  in
  let merged =
    Post_processor.process
      (Post_processor.byte_level ~trim_offsets:false ())
      ~pair:seq_b seq_a ~add_special_tokens:false
  in
  equal ~msg:"type ids" (array int) [| 0; 0; 1 |] (Encoding.type_ids merged);
  equal ~msg:"offsets"
    (array (pair int int))
    [| (0, 5); (6, 11); (0, 4) |]
    (Encoding.offsets merged)

let pair_encodings () =
  ( make_encoding ~ids:[| 10; 11 |] ~tokens:[| "hello"; "world" |] ~type_id:0 (),
    make_encoding ~ids:[| 20 |] ~tokens:[| "pair" |] ~type_id:0 () )

(* HuggingFace merges both sequences of a pair even when it adds no special
   token. The second sequence is segment 1 by default — HuggingFace stamps that
   when it tokenizes, before any processor runs — and only [roberta] overrides
   it, since RoBERTa has a single segment. [encode("hello world", "second one",
   add_special_tokens=False)] gives type ids [0; 0; 1; 1] on gpt2 and
   bert-base-uncased, [0; 0; 0; 0] on roberta-base. *)
let test_pair_without_special_tokens () =
  let seq_a, seq_b = pair_encodings () in
  let merged processor =
    Post_processor.process processor ~pair:seq_b seq_a ~add_special_tokens:false
  in
  let bert =
    merged (Post_processor.bert ~sep:("[SEP]", 102) ~cls:("[CLS]", 101) ())
  in
  equal ~msg:"bert ids" (array int) [| 10; 11; 20 |] (Encoding.ids bert);
  equal ~msg:"bert type ids" (array int) [| 0; 0; 1 |] (Encoding.type_ids bert);
  equal ~msg:"bert special mask" (array int) [| 0; 0; 0 |]
    (Encoding.special_tokens_mask bert);
  equal ~msg:"bert attention" (array int) [| 1; 1; 1 |]
    (Encoding.attention_mask bert);
  let roberta =
    merged (Post_processor.roberta ~sep:("</s>", 2) ~cls:("<s>", 0) ())
  in
  equal ~msg:"roberta ids" (array int) [| 10; 11; 20 |] (Encoding.ids roberta);
  equal ~msg:"roberta type ids" (array int) [| 0; 0; 0 |]
    (Encoding.type_ids roberta);
  let byte_level = merged (Post_processor.byte_level ()) in
  equal ~msg:"byte level ids" (array int) [| 10; 11; 20 |]
    (Encoding.ids byte_level);
  equal ~msg:"byte level type ids" (array int) [| 0; 0; 1 |]
    (Encoding.type_ids byte_level)

(* [CLS] A [SEP] B [SEP] with type ids [0 0 0 0 1 1], against <s> A </s> </s> B
   </s> with every type id 0: RoBERTa has a single segment. *)
let test_pair_with_special_tokens () =
  let seq_a, seq_b = pair_encodings () in
  let merged processor =
    Post_processor.process processor ~pair:seq_b seq_a ~add_special_tokens:true
  in
  let bert =
    merged (Post_processor.bert ~sep:("[SEP]", 102) ~cls:("[CLS]", 101) ())
  in
  equal ~msg:"bert ids" (array int)
    [| 101; 10; 11; 102; 20; 102 |]
    (Encoding.ids bert);
  equal ~msg:"bert type ids" (array int) [| 0; 0; 0; 0; 1; 1 |]
    (Encoding.type_ids bert);
  equal ~msg:"bert special mask" (array int) [| 1; 0; 0; 1; 0; 1 |]
    (Encoding.special_tokens_mask bert);
  let roberta =
    merged (Post_processor.roberta ~sep:("</s>", 2) ~cls:("<s>", 0) ())
  in
  equal ~msg:"roberta ids" (array int)
    [| 0; 10; 11; 2; 2; 20; 2 |]
    (Encoding.ids roberta);
  equal ~msg:"roberta type ids" (array int) [| 0; 0; 0; 0; 0; 0; 0 |]
    (Encoding.type_ids roberta);
  equal ~msg:"roberta special mask" (array int) [| 1; 0; 0; 1; 1; 0; 1 |]
    (Encoding.special_tokens_mask roberta)

let byte_level_encoding tokens offsets =
  let len = Array.length tokens in
  Encoding.create ~ids:(Array.make len 0) ~type_ids:(Array.make len 0)
    ~tokens:(Array.copy tokens) ~words:(Array.make len None)
    ~offsets:(Array.copy offsets) ~special_tokens_mask:(Array.make len 0)
    ~attention_mask:(Array.make len 1) ()

let trimmed processor tokens offsets =
  Encoding.offsets
    (Post_processor.process processor
       (byte_level_encoding tokens offsets)
       ~add_special_tokens:false)

(* Trimming drops the whitespace a byte-level token carries, except the space
   the pre-tokenizer prepended to a token starting at offset 0: it is not in the
   input. Every expectation is what HuggingFace produces for the same tokens and
   offsets. *)
let test_byte_level_trim_offsets () =
  let tokens = [| "\xc4\xa0Hello"; "\xc4\xa0world" |] in
  let offsets = [| (0, 6); (6, 12) |] in
  let trimmed processor = trimmed processor tokens offsets in
  equal ~msg:"prefix space kept"
    (array (pair int int))
    [| (0, 6); (7, 12) |]
    (trimmed (Post_processor.byte_level ()));
  equal ~msg:"prefix space trimmed"
    (array (pair int int))
    [| (1, 6); (7, 12) |]
    (trimmed (Post_processor.byte_level ~add_prefix_space:false ()));
  equal ~msg:"trimming off"
    (array (pair int int))
    offsets
    (trimmed (Post_processor.byte_level ~trim_offsets:false ()))

(* A token that is nothing but the prepended space still loses its trailing
   whitespace: the two ends are counted independently. *)
let test_byte_level_trim_whitespace_token () =
  let tokens = [| "\xc4\xa0"; "\xc4\xa0quick" |] in
  let offsets = [| (0, 1); (1, 7) |] in
  let trimmed processor = trimmed processor tokens offsets in
  equal ~msg:"prefix space kept"
    (array (pair int int))
    [| (0, 0); (2, 7) |]
    (trimmed (Post_processor.byte_level ()));
  equal ~msg:"prefix space trimmed"
    (array (pair int int))
    [| (1, 1); (2, 7) |]
    (trimmed (Post_processor.byte_level ~add_prefix_space:false ()))

(* Byte-level encoding maps a newline to U+010A, a letter rather than
   whitespace, so trimming leaves that token's offsets where they are. *)
let test_byte_level_trim_encoded_newline () =
  let tokens = [| "\xc4\xa0a"; "\xc4\xa0"; "\xc4\x8a"; "\xc4\xa0b" |] in
  let offsets = [| (0, 1); (1, 2); (2, 3); (3, 5) |] in
  equal ~msg:"newline kept"
    (array (pair int int))
    [| (0, 1); (2, 2); (2, 3); (4, 5) |]
    (trimmed (Post_processor.byte_level ()) tokens offsets)

let () =
  run "Processors"
    [
      group "template"
        [
          test "multi-id special expansion" test_template_multi_special;
          test "pair template semantics" test_template_pair_type_ids;
          test "default pair template" test_template_default_pair;
          test "pair template order without specials" test_template_pair_order;
          test "single sequence template" test_template_single_sequence;
        ];
      group "pairs"
        [
          test "merged without special tokens" test_pair_without_special_tokens;
          test "merged with special tokens" test_pair_with_special_tokens;
          test "segments and offsets" test_pair_segments_and_offsets;
        ];
      group "byte level"
        [
          test "trim offsets" test_byte_level_trim_offsets;
          test "trim whitespace-only token"
            test_byte_level_trim_whitespace_token;
          test "keep encoded newline" test_byte_level_trim_encoded_newline;
        ];
    ]
