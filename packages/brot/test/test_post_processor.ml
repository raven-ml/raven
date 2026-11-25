(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Expected values marked HF are what Hugging Face's tokenizers 0.23.1 gives for
   the same processor and input, through [processors.*.process],
   [Tokenizer.encode] or [Tokenizer.from_str]. *)

open Windtrap
open Brot

(* A sequence as a model gives it: content tokens of type [type_id], unmasked
   and attended, token [i] the word [i] spanning the bytes [i, i + 1) of its
   text. Content IDs are 1000 and up, which tells them from the special tokens
   below. *)
let sequence ?(type_id = 0) ids =
  let ids = Array.of_list ids in
  let n = Array.length ids in
  Encoding.create ~ids ~type_ids:(Array.make n type_id)
    ~tokens:(Array.map string_of_int ids)
    ~words:(Array.init n Option.some)
    ~offsets:(Array.init n (fun i -> (i, i + 1)))
    ~special_tokens_mask:(Array.make n 0) ~attention_mask:(Array.make n 1) ()

let first n = sequence (List.init n (fun i -> 1000 + i))
let second n = sequence (List.init n (fun i -> 2000 + i))

(* A sequence of byte-level tokens, whose spaces encode to U+0120 [Ġ]. *)
let byte_level_sequence tokens offsets =
  let n = Array.length tokens in
  Encoding.create
    ~ids:(Array.init n (fun i -> 1000 + i))
    ~type_ids:(Array.make n 0) ~tokens ~words:(Array.init n Option.some)
    ~offsets ~special_tokens_mask:(Array.make n 0)
    ~attention_mask:(Array.make n 1) ()

let space = "\xc4\xa0"
let newline = "\xc4\x8a"
let tab = "\xc4\x89"

let json_text json =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json json with
  | Ok text -> text
  | Error msg -> failf "cannot encode: %s" msg

let json_of_text text =
  match Jsont_bytesrw.decode_string Jsont.json text with
  | Ok json -> json
  | Error msg -> failf "cannot decode %S: %s" text msg

let read text = Post_processor.of_json (json_of_text text)
let written t = json_text (Post_processor.to_json t)
let bert = Post_processor.bert ~sep:("[SEP]", 102) ~cls:("[CLS]", 101)
let roberta = Post_processor.roberta ~sep:("</s>", 2) ~cls:("<s>", 0) ()

let bert_template =
  Post_processor.template ~single:"[CLS] $A [SEP]"
    ~pair:"[CLS] $A [SEP] $B:1 [SEP]:1"
    ~special_tokens:[ ("[CLS]", 101); ("[SEP]", 102) ]
    ()

(* A special token of two IDs, which only a tokenizer.json can declare. *)
let multi_id_text =
  {|{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<multi>","type_id":2}},{"Sequence":{"id":"A","type_id":0}}],"pair":null,"special_tokens":{"<multi>":{"id":"<multi>","ids":[100,101],"tokens":["<m1>","<m2>"]}}}|}

let multi_id () =
  require_ok (Post_processor.of_json (json_of_text multi_id_text))

let x_between =
  Post_processor.template ~single:"$A" ~pair:"$A [X] $B"
    ~special_tokens:[ ("[X]", 7) ]
    ()

let x_before =
  Post_processor.template ~single:"[X] $A" ~special_tokens:[ ("[X]", 7) ] ()

(* The processors that give one sequence for each they receive. *)
let wrappers =
  [
    ("bert", bert);
    ("roberta", roberta);
    ("byte_level", Post_processor.byte_level ());
    ("sequence of two bert", Post_processor.sequence [ bert; bert ]);
  ]

(* The processors the laws range over: [wrappers], a template, the sequence real
   byte-level tokenizers use, and one whose template gives [bert] two
   sequences. *)
let processors =
  wrappers
  @ [
      ("template", bert_template);
      ( "sequence",
        Post_processor.sequence [ Post_processor.byte_level (); bert_template ]
      );
      ("[X] $A, then bert", Post_processor.sequence [ x_before; bert ]);
    ]

let named processors =
  Gen.of_list
    ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name)
    processors

(* [processors] and the special token of two IDs, read in the test that draws
   it. *)
let any_processor =
  named
    (("multi-id template", lazy (multi_id ()))
    :: List.map (fun (name, t) -> (name, Lazy.from_val t)) processors)

(* A processor, the length of the first sequence and, for a pair, of the
   second. *)
let process_domain =
  Gen.(triple any_processor (int_range 0 2) (option (int_range 0 2)))

(* [t], the sequences and what [process] makes of them. *)
let processed ~add_special_tokens ((_, t), a, b) =
  let t = Lazy.force t and a = first a and b = Option.map second b in
  (t, a, b, Post_processor.process t ?pair:b a ~add_special_tokens)

let rows enc =
  List.init (Encoding.length enc) (fun i ->
      ( (Encoding.ids enc).(i),
        (Encoding.word_ids enc).(i),
        (Encoding.offsets enc).(i),
        ( (Encoding.special_tokens_mask enc).(i),
          (Encoding.attention_mask enc).(i) ) ))

let row = quad int (option int) (pair int int) (pair int int)

(* bert *)

let test_bert_single () =
  let single n =
    let enc = Post_processor.process bert (first n) ~add_special_tokens:true in
    (Encoding.ids enc, Encoding.type_ids enc)
  in
  equal ~msg:"empty A"
    (pair (array int) (array int))
    ([| 101; 102 |], [| 0; 0 |])
    (single 0);
  equal ~msg:"two tokens"
    (pair (array int) (array int))
    ([| 101; 1000; 1001; 102 |], [| 0; 0; 0; 0 |])
    (single 2)

(* HF: [0, 0, 0, 0, 1, 1]. *)
let test_bert_pair () =
  let enc =
    Post_processor.process bert ~pair:(second 1) (first 2)
      ~add_special_tokens:true
  in
  equal ~msg:"ids" (array int)
    [| 101; 1000; 1001; 102; 2000; 102 |]
    (Encoding.ids enc);
  equal ~msg:"tokens" (array string)
    [| "[CLS]"; "1000"; "1001"; "[SEP]"; "2000"; "[SEP]" |]
    (Encoding.tokens enc);
  equal ~msg:"type ids" (array int) [| 0; 0; 0; 0; 1; 1 |]
    (Encoding.type_ids enc);
  equal ~msg:"special mask" (array int) [| 1; 0; 0; 1; 0; 1 |]
    (Encoding.special_tokens_mask enc)

(* roberta *)

let test_roberta_single () =
  let enc = Post_processor.process roberta (first 2) ~add_special_tokens:true in
  equal (array int) [| 0; 1000; 1001; 2 |] (Encoding.ids enc)

let test_roberta_pair () =
  let enc =
    Post_processor.process roberta ~pair:(second 1) (first 2)
      ~add_special_tokens:true
  in
  equal ~msg:"ids" (array int)
    [| 0; 1000; 1001; 2; 2; 2000; 2 |]
    (Encoding.ids enc);
  equal ~msg:"tokens" (array string)
    [| "<s>"; "1000"; "1001"; "</s>"; "</s>"; "2000"; "</s>" |]
    (Encoding.tokens enc);
  equal ~msg:"type ids" (array int) [| 0; 0; 0; 0; 0; 0; 0 |]
    (Encoding.type_ids enc);
  equal ~msg:"special mask" (array int) [| 1; 0; 0; 1; 1; 0; 1 |]
    (Encoding.special_tokens_mask enc)

(* Texts as tokenizers' ByteLevel pre-tokenizer cuts them, with the spans of
   each token before trimming. *)
let spaced_texts =
  [
    ( "a  b\n",
      [| "a"; space; space ^ "b"; newline |],
      [| (0, 1); (1, 2); (2, 4); (4, 5) |] );
    (" a", [| space ^ "a" |], [| (0, 2) |]);
    ("  a", [| space; space ^ "a" |], [| (0, 1); (1, 3) |]);
    ("a   ", [| "a"; space ^ space ^ space |], [| (0, 1); (1, 4) |]);
  ]

let spaced_text =
  Gen.of_list
    ~pp:(fun ppf (text, _, _) -> Format.fprintf ppf "%S" text)
    spaced_texts

let trims_as_byte_level ((add_prefix_space, trim_offsets), (_, tokens, offsets))
    =
  let offsets_after t ~add_special_tokens =
    Encoding.offsets
      (Post_processor.process t
         (byte_level_sequence tokens offsets)
         ~add_special_tokens)
  in
  let byte_level =
    offsets_after
      (Post_processor.byte_level ~add_prefix_space ~trim_offsets ())
      ~add_special_tokens:false
  in
  let roberta =
    offsets_after
      (Post_processor.roberta ~sep:("</s>", 2) ~cls:("<s>", 0) ~trim_offsets
         ~add_prefix_space ())
      ~add_special_tokens:true
  in
  equal
    (array (pair int int))
    (Array.concat [ [| (0, 0) |]; byte_level; [| (0, 0) |] ])
    roberta

(* byte_level *)

let trims (_, t, tokens, offsets, expected) =
  equal
    (array (pair int int))
    expected
    (Encoding.offsets
       (Post_processor.process t
          (byte_level_sequence tokens offsets)
          ~add_special_tokens:false))

let prefix_kept = Post_processor.byte_level ()
let prefix_trimmed = Post_processor.byte_level ~add_prefix_space:false ()

(* HF, from [Tokenizer] with a WordLevel model over these tokens,
   [pre_tokenizers.ByteLevel] and [processors.ByteLevel] with the same flags,
   except the rows marked .mli, which [byte_level]'s rule decides. *)
let trim_rows =
  [
    ( "the prepended space is kept",
      prefix_kept,
      [| space ^ "Hello"; space ^ "world" |],
      [| (0, 6); (6, 12) |],
      [| (0, 6); (7, 12) |] );
    ( "without add_prefix_space the prepended space goes",
      prefix_trimmed,
      [| space ^ "Hello"; space ^ "world" |],
      [| (0, 6); (6, 12) |],
      [| (1, 6); (7, 12) |] );
    ( "without trim_offsets every span stays",
      Post_processor.byte_level ~trim_offsets:false (),
      [| space ^ "Hello"; space ^ "world" |],
      [| (0, 6); (6, 12) |],
      [| (0, 6); (6, 12) |] );
    ( "a first token of spaces alone closes at its start",
      prefix_kept,
      [| space; space ^ "quick" |],
      [| (0, 1); (1, 7) |],
      [| (0, 0); (2, 7) |] );
    ( "without add_prefix_space it closes at its end",
      prefix_trimmed,
      [| space; space ^ "quick" |],
      [| (0, 1); (1, 7) |],
      [| (1, 1); (2, 7) |] );
    ( "a later token of spaces alone closes at its end",
      prefix_trimmed,
      [| "a"; space ^ space ^ space |],
      [| (0, 1); (1, 4) |],
      [| (0, 1); (4, 4) |] );
    ( "a first token of two leading spaces loses both",
      prefix_kept,
      [| space ^ space ^ "a" |],
      [| (0, 3) |],
      [| (2, 3) |] );
    ( "without add_prefix_space it loses both too",
      prefix_trimmed,
      [| space ^ space ^ "a" |],
      [| (0, 3) |],
      [| (2, 3) |] );
    ( "an encoded newline is a letter",
      prefix_kept,
      [| space ^ "a"; space; newline; space ^ "b" |],
      [| (0, 1); (1, 2); (2, 3); (3, 5) |],
      [| (0, 1); (2, 2); (2, 3); (4, 5) |] );
    ( "an encoded tab is a letter",
      prefix_kept,
      [| space ^ "a"; tab |],
      [| (0, 1); (1, 2) |],
      [| (0, 1); (1, 2) |] );
    ( "(.mli) the first token keeps its prepended space off offset 0",
      prefix_kept,
      [| space ^ "a"; space ^ "b" |],
      [| (3, 5); (5, 7) |],
      [| (3, 5); (6, 7) |] );
    ( "(.mli) a later token at offset 0 keeps its prepended space",
      prefix_kept,
      [| "a"; space ^ "b" |],
      [| (0, 1); (0, 2) |],
      [| (0, 1); (0, 2) |] );
  ]

(* template *)

let typed_pair (template, ids, type_ids) =
  let t =
    Post_processor.template ~single:"$A" ~pair:template
      ~special_tokens:[ ("[CLS]", 101); ("[SEP]", 102) ]
      ()
  in
  let enc =
    Post_processor.process t ~pair:(second 1) (first 2) ~add_special_tokens:true
  in
  let text = function
    | 101 -> "[CLS]"
    | 102 -> "[SEP]"
    | id -> string_of_int id
  in
  equal
    (triple (array int) (array string) (array int))
    (ids, Array.map text ids, type_ids)
    (Encoding.ids enc, Encoding.tokens enc, Encoding.type_ids enc)

(* HF, every row of both tables. *)
let suffixed =
  [
    ( "[CLS]:0 $A:0 [SEP]:0 $B:3 [SEP]:3",
      [| 101; 1000; 1001; 102; 2000; 102 |],
      [| 0; 0; 0; 0; 3; 3 |] );
    ( "[CLS] $A [SEP] $B [SEP]",
      [| 101; 1000; 1001; 102; 2000; 102 |],
      [| 0; 0; 0; 0; 0; 0 |] );
    ("$a $b:1", [| 1000; 1001; 2000 |], [| 0; 0; 1 |]);
  ]

let typed_single (template, type_ids) =
  let t = Post_processor.template ~single:template () in
  equal (array int) type_ids
    (Encoding.type_ids
       (Post_processor.process t (first 2) ~add_special_tokens:true))

let test_multi_id () =
  let enc =
    Post_processor.process (multi_id ()) (first 1) ~add_special_tokens:true
  in
  equal ~msg:"ids" (array int) [| 100; 101; 1000 |] (Encoding.ids enc);
  equal ~msg:"tokens" (array string)
    [| "<m1>"; "<m2>"; "1000" |]
    (Encoding.tokens enc);
  equal ~msg:"type ids" (array int) [| 2; 2; 0 |] (Encoding.type_ids enc);
  equal ~msg:"special mask" (array int) [| 1; 1; 0 |]
    (Encoding.special_tokens_mask enc)

(* HF: TemplateProcessing(single="[X] $A", special_tokens=[("[X]", 1), ("[X]",
   2)]) processes [5] to [2, 5] and counts 1 special token. *)
let duplicated_x =
  {|{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[X]","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"[X]":{"id":"[X]","ids":[1],"tokens":["[X]"]},"[X]":{"id":"[X]","ids":[2],"tokens":["[X]"]}}}|}

let single_x =
  {|{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[X]","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"[X]":{"id":"[X]","ids":[2],"tokens":["[X]"]}}}|}

let keeps_last (_, t) =
  let t = Lazy.force t in
  equal ~msg:"ids" (array int) [| 2; 1000 |]
    (Encoding.ids (Post_processor.process t (first 1) ~add_special_tokens:true));
  equal ~msg:"added_tokens" int 1 (Post_processor.added_tokens t ~is_pair:false);
  equal ~msg:"to_json" string single_x (written t)

let test_default_pair () =
  let t = Post_processor.template ~single:"$A" () in
  let enc =
    Post_processor.process t ~pair:(second 1) (first 1) ~add_special_tokens:true
  in
  equal
    (pair (array int) (array int))
    ([| 1000; 2000 |], [| 0; 1 |])
    (Encoding.ids enc, Encoding.type_ids enc)

(* HF, pair template ["$B:1 $A:0"]: tokens ['pair', 'hello', 'world'], type_ids
   [1, 0, 0]. *)
let test_pair_order_without_specials () =
  let t = Post_processor.template ~single:"$A" ~pair:"$B:1 $A:0" () in
  let enc =
    Post_processor.process t ~pair:(second 1) (first 2)
      ~add_special_tokens:false
  in
  equal
    (pair (array int) (array int))
    ([| 2000; 1000; 1001 |], [| 1; 0; 0 |])
    (Encoding.ids enc, Encoding.type_ids enc)

let test_single_without_specials () =
  let t =
    Post_processor.template ~single:"[CLS] $A:2"
      ~special_tokens:[ ("[CLS]", 101) ]
      ()
  in
  let enc = first 2 in
  let retyped = Post_processor.process t enc ~add_special_tokens:false in
  equal ~msg:"type ids" (array int) [| 2; 2 |] (Encoding.type_ids retyped);
  equal ~msg:"everything else" (list row) (rows enc) (rows retyped)

let test_colon_token () =
  raises_match (Exn.invalid_arg ~sub:"a:b") (fun () ->
      Post_processor.template ~single:"a:b $A" ~special_tokens:[ ("a:b", 5) ] ())

let raises_on (single, pair, names) =
  raises_match (Exn.invalid_arg ~sub:names) (fun () ->
      Post_processor.template ~single ?pair
        ~special_tokens:[ ("[CLS]", 101) ]
        ())

(* sequence *)

(* A processor that gives one sequence back hands it whole to the next, so a
   chain of them applies each in turn; the last may give more. The first token
   starts past offset 0, so whether [bert] has moved it from index 0 changes
   what [byte_level] trims. *)
let chained (wrappers, last) =
  let e =
    byte_level_sequence
      [| space ^ "a"; space; "b" |]
      [| (1, 3); (3, 4); (4, 5) |]
  in
  let processors = List.map snd (wrappers @ Option.to_list last) in
  cover "no processor" (processors = []);
  cover "two processors" (List.length processors = 2);
  let one_by_one =
    List.fold_left
      (fun enc t -> Post_processor.process t enc ~add_special_tokens:true)
      e processors
  in
  let chained =
    Post_processor.process
      (Post_processor.sequence processors)
      e ~add_special_tokens:true
  in
  equal ~msg:"ids" (array int) (Encoding.ids one_by_one) (Encoding.ids chained);
  equal ~msg:"type ids" (array int)
    (Encoding.type_ids one_by_one)
    (Encoding.type_ids chained);
  equal ~msg:"rows" (list row) (rows one_by_one) (rows chained)

(* HF: processors.Sequence([...]).process(a, b) and num_special_tokens_to_add,
   except the added_tokens of the last row, where HF counts 3: each processor
   keeps the sequences of a pair apart, a template gives each of its pieces as a
   sequence, and [process] merges them once, at the end. *)
let chained_pairs =
  [
    ( "[bert; bert], a pair",
      [ bert; bert ],
      Some 1,
      [| 101; 101; 1000; 1001; 102; 102; 2000; 102; 102 |],
      [| 0; 0; 0; 0; 0; 0; 1; 1; 1 |] );
    ( "[bert; $A [X] $B], a pair",
      [ bert; x_between ],
      Some 1,
      [| 101; 1000; 1001; 102; 7; 2000; 102 |],
      [| 0; 0; 0; 0; 0; 0; 0 |] );
    ( "[roberta; $A [X] $B], a pair",
      [ roberta; x_between ],
      Some 1,
      [| 0; 1000; 1001; 2; 7; 2; 2000; 2 |],
      [| 0; 0; 0; 0; 0; 0; 0; 0 |] );
    ( "[[X] $A; bert], one sequence",
      [ x_before; bert ],
      None,
      [| 101; 7; 102; 1000; 1001; 102 |],
      [| 0; 0; 0; 0; 0; 1 |] );
  ]

let passes_pairs (_, processors, b, ids, type_ids) =
  let t = Post_processor.sequence processors in
  let a = first 2 and b = Option.map second b in
  let enc = Post_processor.process t ?pair:b a ~add_special_tokens:true in
  equal ~msg:"ids" (array int) ids (Encoding.ids enc);
  equal ~msg:"type ids" (array int) type_ids (Encoding.type_ids enc);
  let b_length = match b with Some b -> Encoding.length b | None -> 0 in
  equal ~msg:"added_tokens" int
    (Encoding.length enc - Encoding.length a - b_length)
    (Post_processor.added_tokens t ~is_pair:(b <> None))

(* process *)

(* HF: BertProcessing and ByteLevel .process on an encoding typed 5. *)
let typed_first (_, t, type_ids) =
  let enc =
    Post_processor.process t
      (sequence ~type_id:5 [ 1000; 1001 ])
      ~add_special_tokens:true
  in
  equal (array int) type_ids (Encoding.type_ids enc)

(* HF: the pair of [encode("hello world", "second one",
   add_special_tokens=False)] has type IDs [0, 0, 1, 1] on gpt2 and
   bert-base-uncased, [0, 0, 0, 0] on roberta-base. *)
let typed_without_specials (_, t, type_ids) =
  let enc =
    Post_processor.process t ~pair:(second 1) (first 2)
      ~add_special_tokens:false
  in
  equal (array int) type_ids (Encoding.type_ids enc)

let test_pair_typed_one () =
  let enc =
    Post_processor.process
      (Post_processor.byte_level ~trim_offsets:false ())
      ~pair:(sequence ~type_id:7 [ 2000 ])
      (first 2) ~add_special_tokens:false
  in
  equal ~msg:"type ids" (array int) [| 0; 0; 1 |] (Encoding.type_ids enc);
  equal ~msg:"offsets"
    (array (pair int int))
    [| (0, 1); (1, 2); (0, 1) |]
    (Encoding.offsets enc)

(* An encoding as [encode] gives it for ["a b"], with words and spans. *)
let test_inserted_words_and_spans () =
  let t =
    Post_processor.template ~single:"[CLS] $A"
      ~special_tokens:[ ("[CLS]", 2) ]
      ()
  in
  let enc =
    Encoding.create ~ids:[| 0; 1 |] ~type_ids:[| 0; 0 |] ~tokens:[| "a"; "b" |]
      ~words:[| Some 0; Some 1 |]
      ~offsets:[| (0, 1); (2, 3) |]
      ~special_tokens_mask:[| 0; 0 |] ~attention_mask:[| 1; 1 |] ()
  in
  let processed = Post_processor.process t enc ~add_special_tokens:true in
  equal ~msg:"words"
    (array (option int))
    [| None; Some 0; Some 1 |]
    (Encoding.word_ids processed);
  equal ~msg:"spans"
    (array (pair int int))
    [| (0, 0); (0, 1); (2, 3) |]
    (Encoding.offsets processed)

let inserts_specials case =
  let _, _, _, enc = processed ~add_special_tokens:true case in
  let inserted = List.filter (fun (id, _, _, _) -> id < 1000) (rows enc) in
  cover "inserts" (inserted <> []);
  equal (list row)
    (List.map (fun (id, _, _, _) -> (id, None, (0, 0), (1, 1))) inserted)
    inserted

let keeps_content case =
  let content add_special_tokens =
    let _, a, b, enc = processed ~add_special_tokens case in
    let b = match b with Some b -> rows b | None -> [] in
    (rows a @ b, List.filter (fun (id, _, _, _) -> id >= 1000) (rows enc))
  in
  let expected, with_specials = content true in
  equal ~msg:"with special tokens" (list row) expected with_specials;
  let expected, without = content false in
  equal ~msg:"without special tokens" (list row) expected without

let inserts_nothing case =
  let _, a, b, enc = processed ~add_special_tokens:false case in
  let b = match b with Some b -> Encoding.ids b | None -> [||] in
  equal (array int) (Array.append (Encoding.ids a) b) (Encoding.ids enc)

(* tokenizers panics on the second row. *)
let unplaced =
  [
    ("a single template naming $B", Post_processor.template ~single:"$B" ());
    ( "a template after one of three pieces",
      Post_processor.sequence [ bert_template; bert_template ] );
  ]

let raises_unplaced (_, t) =
  raises_match Exn.invalid_arg (fun () ->
      Post_processor.process t (first 1) ~add_special_tokens:true)

(* added_tokens *)

let counts_inserted case =
  let t, a, b, enc = processed ~add_special_tokens:true case in
  cover "pair" (b <> None);
  let b_length = match b with Some b -> Encoding.length b | None -> 0 in
  equal int
    (Encoding.length a + b_length
    + Post_processor.added_tokens t ~is_pair:(b <> None))
    (Encoding.length enc)

(* pp *)

let test_pp () =
  let t =
    Post_processor.sequence
      [ bert; roberta; Post_processor.byte_level (); bert_template ]
  in
  expect (Format.asprintf "%a" Post_processor.pp t)
  @@ __POS_OF__
       {|
    Sequence[Bert ~cls:"[CLS]" ~sep:"[SEP]",
      Roberta ~cls:"<s>" ~sep:"</s>" ~trim_offsets:true ~add_prefix_space:true,
      ByteLevel ~add_prefix_space:true ~trim_offsets:true, Template]
    |}

(* of_json *)

let not_post_processors =
  [
    ("a list", {|[]|});
    ("no type", {|{}|});
    ("an unknown type", {|{"type":"Nope"}|});
    ("a Sequence without processors", {|{"type":"Sequence"}|});
    ( "a template naming an undeclared token",
      {|{"type":"TemplateProcessing","single":"[X] $A","special_tokens":{}}|} );
    ( "a special token with more ids than tokens",
      {|{"type":"TemplateProcessing","single":"[X] $A","special_tokens":{"[X]":{"id":"[X]","ids":[1,2],"tokens":["a"]}}}|}
    );
    ( "a type ID of 2^32",
      {|{"type":"TemplateProcessing","single":"$A:4294967296","special_tokens":{}}|}
    );
    ( "a sequence that is neither A nor B",
      {|{"type":"TemplateProcessing","single":[{"Sequence":{"id":"C","type_id":0}}],"special_tokens":{}}|}
    );
  ]

(* Members a reader cannot take, each with the member its error names. Hugging
   Face 0.23.1 rejects every row but the RobertaProcessing ones, which its
   untagged enum reads as a BertProcessing; brot names the member instead.
   BertProcessing's and RobertaProcessing's sep and cls have no default. *)
let ill_typed =
  [
    ("BertProcessing without sep or cls", {|{"type":"BertProcessing"}|}, "sep");
    ( "a sep that is a string",
      {|{"type":"BertProcessing","sep":"x","cls":["[CLS]",101]}|},
      "sep" );
    ( "a cls that is a number",
      {|{"type":"BertProcessing","sep":["[SEP]",102],"cls":3}|},
      "cls" );
    ( "a fractional sep id",
      {|{"type":"BertProcessing","sep":["[SEP]",1.5],"cls":["[CLS]",101]}|},
      "sep" );
    ( "a sep id past 2^31",
      {|{"type":"BertProcessing","sep":["[SEP]",2147483648],"cls":["[CLS]",101]}|},
      "sep" );
    ( "RobertaProcessing without cls",
      {|{"type":"RobertaProcessing","sep":["</s>",2]}|},
      "cls" );
    ( "a RobertaProcessing flag that is a string",
      {|{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":"no"}|},
      "trim_offsets" );
    ( "a ByteLevel flag that is a string",
      {|{"type":"ByteLevel","trim_offsets":"no"}|},
      "trim_offsets" );
    ( "a fractional type_id",
      {|{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":1.5}}],"special_tokens":{}}|},
      "type_id" );
    ( "a fractional special token id",
      {|{"type":"TemplateProcessing","single":"[X] $A","special_tokens":{"[X]":{"id":"[X]","ids":[1.5],"tokens":["[X]"]}}}|},
      "ids" );
  ]

let names_the_member (_, text, member) =
  contains ~sub:member (require_error ~pp:Post_processor.pp (read text))

(* A post-processor, compared by the JSON it writes. *)
let by_json =
  Testable.make ~pp:Post_processor.pp ~equal:(fun a b -> written a = written b)

(* Each read back by tokenizers' [Tokenizer.from_str] as the same processor. It
   requires [add_prefix_space] and [trim_offsets] on [ByteLevel], all four
   members on [RobertaProcessing] (with either flag missing it reads the object
   as a [BertProcessing]), and an array for [TemplateProcessing]'s [pair]. *)
let post_processors =
  [
    ( "bert",
      bert,
      {|{"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}|} );
    ( "roberta",
      roberta,
      {|{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":true}|}
    );
    ( "roberta without a prefix space",
      Post_processor.roberta ~sep:("</s>", 2) ~cls:("<s>", 0)
        ~add_prefix_space:false (),
      {|{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":false}|}
    );
    ( "byte_level",
      Post_processor.byte_level (),
      {|{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true}|} );
    ( "byte_level without trimming or a prefix space",
      Post_processor.byte_level ~add_prefix_space:false ~trim_offsets:false (),
      {|{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false}|} );
    ( "template of $A",
      Post_processor.template ~single:"$A" (),
      {|{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}|}
    );
    ( "BERT's template",
      bert_template,
      {|{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}}],"pair":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}},{"Sequence":{"id":"B","type_id":1}},{"SpecialToken":{"id":"[SEP]","type_id":1}}],"special_tokens":{"[CLS]":{"id":"[CLS]","ids":[101],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[102],"tokens":["[SEP]"]}}}|}
    );
    ( "sequence",
      Post_processor.sequence [ Post_processor.byte_level (); bert ],
      {|{"type":"Sequence","processors":[{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true},{"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}]}|}
    );
  ]

let template_of_a =
  {|"single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}|}

let test_use_regex () =
  let text =
    {|{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true}|}
  in
  equal string
    {|{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false}|}
    (written (require_ok (read text)))

(* A [null] pair is what tokenizers writes for a template built without one. *)
let defaults =
  [
    ( "ByteLevel's flags",
      {|{"type":"ByteLevel"}|},
      {|{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true}|} );
    ( "RobertaProcessing's flags",
      {|{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0]}|},
      {|{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":true}|}
    );
    ( "a null pair",
      {|{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],"pair":null,"special_tokens":{}}|},
      {|{"type":"TemplateProcessing",|} ^ template_of_a );
    ( "a missing pair",
      {|{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],"special_tokens":{}}|},
      {|{"type":"TemplateProcessing",|} ^ template_of_a );
    ( "a missing single",
      {|{"type":"TemplateProcessing","special_tokens":{}}|},
      {|{"type":"TemplateProcessing",|} ^ template_of_a );
    ( "templates as strings",
      {|{"type":"TemplateProcessing","single":"$A","pair":"$A $B:1","special_tokens":{}}|},
      {|{"type":"TemplateProcessing",|} ^ template_of_a );
  ]

(* to_json *)

(* Read from the pretrained files and written back: GPT-2 keeps
   [add_prefix_space], without which tokenizers refuses the file. *)
let pretrained =
  [
    ( "gpt2",
      {|{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false}|} );
    ( "bert_base",
      {|{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}}],"pair":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}},{"Sequence":{"id":"B","type_id":1}},{"SpecialToken":{"id":"[SEP]","type_id":1}}],"special_tokens":{"[CLS]":{"id":"[CLS]","ids":[101],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[102],"tokens":["[SEP]"]}}}|}
    );
    ( "mistral",
      {|{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<s>","type_id":1}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"<s>":{"id":"<s>","ids":[1],"tokens":["<s>"]}}}|}
    );
  ]

let writes_pretrained (model, expected) =
  Fixture.with_tokenizer_file model (fun path ->
      let t = require_ok (from_file path) in
      let p = require_some (post_processor t) in
      equal string expected (written p))

let () =
  exit
    (run "Post_processor"
       [
         group "bert"
           [
             test "a single sequence is [CLS] A [SEP], typed 0" test_bert_single;
             test "a pair is [CLS] A [SEP] B [SEP], B typed 1" test_bert_pair;
           ];
         group "roberta"
           [
             test "a single sequence is <s> A </s>" test_roberta_single;
             test "a pair is <s> A </s> </s> B </s>, all typed 0"
               test_roberta_pair;
             prop "trims offsets as byte_level does with the same flags"
               Gen.(pair (pair bool bool) spaced_text)
               trims_as_byte_level;
           ];
         group "byte_level"
           [
             cases "trims each token's span past its white space"
               ~id:(fun (name, _, _, _, _) -> name)
               trim_rows trims;
           ];
         group "template"
           [
             cases "a :n suffix sets a piece's type ID, and none means 0"
               ~id:(fun (template, _, _) -> template)
               suffixed typed_pair;
             cases "$, $0 and $a are $A, and $n is $A:n" ~id:fst
               [
                 ("$", [| 0; 0 |]);
                 ("$0", [| 0; 0 |]);
                 ("$a", [| 0; 0 |]);
                 ("$1", [| 1; 1 |]);
               ]
               typed_single;
             test
               "a special token of several IDs inserts each, with its type ID"
               test_multi_id;
             cases
               "a later special token of the same text replaces an earlier one"
               ~id:fst
               [
                 ( "template",
                   lazy
                     (Post_processor.template ~single:"[X] $A"
                        ~special_tokens:[ ("[X]", 1); ("[X]", 2) ]
                        ()) );
                 ("of_json", lazy (require_ok (read duplicated_x)));
               ]
               keeps_last;
             test "a pair without a pair template is $A:0 $B:1"
               test_default_pair;
             test
               "without special tokens a pair template still orders the \
                sequences"
               test_pair_order_without_specials;
             test "without special tokens a single template only retypes"
               test_single_without_specials;
             test "a token whose text holds : cannot appear" test_colon_token;
             cases
               "raises Invalid_argument naming an unknown token, a bad suffix \
                or a pair without $A and $B"
               ~id:(fun (single, pair, _) ->
                 Printf.sprintf "%S %s" single (Option.value pair ~default:""))
               [
                 ("[X] $A", None, "[X]");
                 ("$A:x", None, "$A:x");
                 ("$A:-1", None, "$A:-1");
                 ("$A:99999999999999999999", None, "$A:99999999999999999999");
                 ("$A:4294967296", None, "$A:4294967296");
                 ("[CLS]:4294967296 $A", None, "[CLS]:4294967296");
                 ("[CLS:0] $A", None, "[CLS:0]");
                 ("$A", Some "$A [CLS]", "pair");
                 ("$A", Some "[CLS] $B", "pair");
               ]
               raises_on;
           ];
         group "sequence"
           [
             prop
               "applies its processors in turn while each but the last gives \
                one sequence for one"
               ~examples:[ ([], None) ]
               Gen.(
                 pair
                   (list ~size:(int_range 0 1) (named wrappers))
                   (option (named processors)))
               chained;
             cases "hands each processor the sequences the one before it gives"
               ~id:(fun (name, _, _, _, _) -> name)
               chained_pairs passes_pairs;
           ];
         group "process"
           [
             cases "types the first sequence 0 whatever it arrives with"
               ~id:(fun (name, _, _) -> name)
               [
                 ("bert", bert, [| 0; 0; 0; 0 |]);
                 ("byte_level", Post_processor.byte_level (), [| 0; 0 |]);
               ]
               typed_first;
             test
               "types the pair 1 whatever it arrives with and shifts no offset"
               test_pair_typed_one;
             cases "without special tokens a pair is still merged and typed"
               ~id:(fun (name, _, _) -> name)
               [
                 ("bert", bert, [| 0; 0; 1 |]);
                 ("byte_level", Post_processor.byte_level (), [| 0; 0; 1 |]);
                 ("roberta", roberta, [| 0; 0; 0 |]);
               ]
               typed_without_specials;
             test
               "keeps an encoded sequence's words and spans beside inserted \
                tokens"
               test_inserted_words_and_spans;
             prop
               "inserted tokens are special, attended, with no word and the \
                span (0, 0)"
               process_domain inserts_specials;
             prop "keeps each sequence's tokens, A's before B's" process_domain
               keeps_content;
             prop "without special tokens inserts nothing" process_domain
               inserts_nothing;
             cases
               "raises Invalid_argument when a template names a sequence it \
                does not receive, or receives more than two"
               ~id:fst unplaced raises_unplaced;
           ];
         group "added_tokens"
           [
             prop "is the number of tokens process inserts" process_domain
               counts_inserted;
           ];
         group "pp" [ test "names each processor and its options" test_pp ];
         group "of_json"
           [
             cases "is an error for what is not a post-processor" ~id:fst
               not_post_processors (fun (_, text) ->
                 is_error ~pp:Post_processor.pp (read text));
             cases "is an error naming a member it cannot read"
               ~id:(fun (name, _, _) -> name)
               ill_typed names_the_member;
             cases "reads back what to_json writes"
               ~id:(fun (name, _, _) -> name)
               post_processors
               (fun (_, _, text) ->
                 Law.round_trip string by_json
                   (fun text -> require_ok (read text))
                   written text);
             test "ignores ByteLevel's use_regex, a pre-tokenizer's member"
               test_use_regex;
             cases "fills in what a file leaves out"
               ~id:(fun (name, _, _) -> name)
               defaults
               (fun (_, text, expected) ->
                 equal string expected (written (require_ok (read text))));
           ];
         group "to_json"
           [
             cases "writes the members tokenizers reads"
               ~id:(fun (name, _, _) -> name)
               post_processors
               (fun (_, t, expected) -> equal string expected (written t));
             cases "writes a pretrained tokenizer's post-processor as read"
               ~id:fst pretrained writes_pretrained;
           ];
       ])
