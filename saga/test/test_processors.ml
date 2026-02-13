(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Saga_tokenizers

let make_encoding ~ids ~tokens ~type_id =
  let len = Array.length ids in
  {
    Processors.ids = Array.copy ids;
    type_ids = Array.make len type_id;
    tokens = Array.copy tokens;
    offsets = Array.make len (0, 0);
    special_tokens_mask = Array.make len 0;
    attention_mask = Array.make len 1;
    overflowing = [];
    sequence_ranges = [];
  }

let test_template_multi_special () =
  let processor =
    Processors.of_json
      (`Assoc
         [
           ("type", `String "TemplateProcessing");
           ( "single",
             `List
               [
                 `Assoc
                   [
                     ( "SpecialToken",
                       `Assoc [ ("id", `String "<multi>"); ("type_id", `Int 2) ]
                     );
                   ];
                 `Assoc
                   [
                     ( "Sequence",
                       `Assoc [ ("id", `String "A"); ("type_id", `Int 0) ] );
                   ];
               ] );
           ("pair", `Null);
           ( "special_tokens",
             `Assoc
               [
                 ( "<multi>",
                   `Assoc
                     [
                       ("id", `String "<multi>");
                       ("ids", `List [ `Int 100; `Int 101 ]);
                       ("tokens", `List [ `String "<m1>"; `String "<m2>" ]);
                     ] );
               ] );
         ])
  in
  let base = make_encoding ~ids:[| 10 |] ~tokens:[| "hello" |] ~type_id:0 in
  let result = Processors.process processor [ base ] ~add_special_tokens:true in
  match result with
  | [ encoding ] ->
      equal ~msg:"ids" (array int) [| 100; 101; 10 |] encoding.ids;
      equal ~msg:"tokens"
        (array string)
        [| "<m1>"; "<m2>"; "hello" |]
        encoding.tokens;
      equal ~msg:"type ids" (array int) [| 2; 2; 0 |] encoding.type_ids;
      equal ~msg:"special mask" (array int) [| 1; 1; 0 |]
        encoding.special_tokens_mask;
      equal ~msg:"attention mask" (array int) [| 1; 1; 1 |] encoding.attention_mask;
      equal ~msg:"added tokens single" int 2
        (Processors.added_tokens processor ~is_pair:false)
  | _ -> failwith "expected exactly one encoding"

let test_template_pair_type_ids () =
  let processor =
    Processors.template ~single:"$A [SEP]"
      ~pair:"[CLS]:0 $A:0 [SEP]:0 $B:3 [SEP]:3"
      ~special_tokens:[ ("[CLS]", 101); ("[SEP]", 102) ]
      ()
  in
  let seq_a =
    make_encoding ~ids:[| 10; 11 |] ~tokens:[| "hello"; "world" |] ~type_id:0
  in
  let seq_b = make_encoding ~ids:[| 20 |] ~tokens:[| "pair" |] ~type_id:1 in
  let combined =
    Processors.process processor [ seq_a; seq_b ] ~add_special_tokens:true
  in
  (match combined with
  | [ encoding ] ->
      equal ~msg:"pair ids" (array int) [| 101; 10; 11; 102; 20; 102 |] encoding.ids;
      equal ~msg:"pair tokens"
        (array string)
        [| "[CLS]"; "hello"; "world"; "[SEP]"; "pair"; "[SEP]" |]
        encoding.tokens;
      equal ~msg:"pair type ids" (array int) [| 0; 0; 0; 0; 3; 3 |] encoding.type_ids;
      equal ~msg:"pair special mask" (array int) [| 1; 0; 0; 1; 0; 1 |]
        encoding.special_tokens_mask;
      equal ~msg:"added tokens pair" int 3
        (Processors.added_tokens processor ~is_pair:true)
  | _ -> failwith "expected a single merged encoding");
  let no_special =
    Processors.process processor [ seq_a; seq_b ] ~add_special_tokens:false
  in
  match no_special with
  | [ first; second ] ->
      equal ~msg:"no-special ids first" (array int) seq_a.ids first.ids;
      equal ~msg:"no-special ids second" (array int) seq_b.ids second.ids
  | _ -> failwith "expected original encodings without specials"

let () =
  run "Processors"
    [
      group "template"
        [
          test "multi-id special expansion"
            test_template_multi_special;
          test "pair template semantics" test_template_pair_type_ids;
        ];
    ]
