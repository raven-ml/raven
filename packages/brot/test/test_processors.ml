(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Brot

let make_encoding ~ids ~tokens ~type_id =
  let len = Array.length ids in
  Encoding.create ~ids:(Array.copy ids) ~type_ids:(Array.make len type_id)
    ~tokens:(Array.copy tokens) ~words:(Array.make len None)
    ~offsets:(Array.make len (0, 0))
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
  let base = make_encoding ~ids:[| 10 |] ~tokens:[| "hello" |] ~type_id:0 in
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
    make_encoding ~ids:[| 10; 11 |] ~tokens:[| "hello"; "world" |] ~type_id:0
  in
  let seq_b = make_encoding ~ids:[| 20 |] ~tokens:[| "pair" |] ~type_id:1 in
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
  let no_special =
    Post_processor.process processor ~pair:seq_b seq_a ~add_special_tokens:false
  in
  equal ~msg:"no-special ids" (array int) (Encoding.ids seq_a)
    (Encoding.ids no_special)

(* A template built without a pair still processes a pair, with the template
   HuggingFace fills in: [$A:0 $B:1]. *)
let test_template_default_pair () =
  let processor = Post_processor.template ~single:"$A" () in
  let seq_a = make_encoding ~ids:[| 10 |] ~tokens:[| "hello" |] ~type_id:0 in
  let seq_b = make_encoding ~ids:[| 20 |] ~tokens:[| "pair" |] ~type_id:0 in
  let encoding =
    Post_processor.process processor ~pair:seq_b seq_a ~add_special_tokens:true
  in
  equal ~msg:"ids" (array int) [| 10; 20 |] (Encoding.ids encoding);
  equal ~msg:"type ids" (array int) [| 0; 1 |] (Encoding.type_ids encoding);
  equal ~msg:"added tokens" int 0
    (Post_processor.added_tokens processor ~is_pair:true)

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
        ];
      group "byte level"
        [
          test "trim offsets" test_byte_level_trim_offsets;
          test "trim whitespace-only token"
            test_byte_level_trim_whitespace_token;
          test "keep encoded newline" test_byte_level_trim_encoded_newline;
        ];
    ]
