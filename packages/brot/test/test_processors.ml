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

let () =
  run "Processors"
    [
      group "template"
        [
          test "multi-id special expansion" test_template_multi_special;
          test "pair template semantics" test_template_pair_type_ids;
        ];
    ]
