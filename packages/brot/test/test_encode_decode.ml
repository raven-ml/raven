(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Encoding and decoding: [encode] and its batch forms with the [padding] and
   [truncation] they apply, [exec], [decode] and streaming decoding. A value
   marked HuggingFace is the output of tokenizers 0.23.1 on the same
   pipeline. *)

open Windtrap
open Brot

(* Tokenizers *)

let words = List.mapi (fun id word -> (word, id))

(* The words [a] to [f], ids 0 to 5, then [[PAD]] 6 and [[UNK]] 7. *)
let letters = words [ "a"; "b"; "c"; "d"; "e"; "f"; "[PAD]"; "[UNK]" ]

(* The letters between BERT's [[CLS]] 8 and [[SEP]] 9, padded with [[PAD]]. *)
let bert_words () =
  word_level ~pre:Pre_tokenizer.whitespace
    ~post:(Post_processor.bert ~cls:("[CLS]", 8) ~sep:("[SEP]", 9))
    ~vocab:letters ~unk_token:"[UNK]" ~pad_token:"[PAD]" ()

(* The letters with no post-processor and no pad token. *)
let plain_words () =
  word_level ~pre:Pre_tokenizer.whitespace ~vocab:letters ~unk_token:"[UNK]" ()

(* The text of [k] letters, [a] to [f] over and over, whose ids are [i mod
   6]. *)
let text_of_length k =
  String.concat " " (List.init k (fun i -> String.make 1 "abcdef".[i mod 6]))

let ids_of ?pair ?add_special_tokens ?padding ?truncation t text =
  Encoding.ids (encode t ?pair ?add_special_tokens ?padding ?truncation text)

let windows encoding = List.map Encoding.ids (Encoding.overflowing encoding)

let offsets_of t text =
  Encoding.offsets (encode t ~add_special_tokens:false text)

let tokens_of t text = Encoding.tokens (encode t ~add_special_tokens:false text)

(* The character a byte stands for in the byte-level alphabet. *)
let byte_char =
  let alone =
    Pre_tokenizer.byte_level ~add_prefix_space:false ~use_regex:false ()
  in
  fun b -> fst (List.hd (Pre_tokenizer.pre_tokenize alone (String.make 1 b)))

(* Byte-level BPE over every byte, merging [ab], with an added token, a pad
   token, and a template that ends each sequence with [[END]]. *)
let byte_bpe () =
  bpe
    ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
    ~decoder:Decoder.byte_level
    ~vocab:(("ab", 256) :: List.init 256 (fun b -> (byte_char (Char.chr b), b)))
    ~merges:[ ("a", "b") ]
    ~added_tokens:[ added_token "<s>" ]
    ~pad_token:"<pad>"
    ~post:
      (Post_processor.template ~single:"$A [END]" ~pair:"$A [END] $B:1 [END]:1"
         ~special_tokens:[ ("[END]", 300) ]
         ())
    ()

(* The letters under a template that names the sequence twice, which does more
   than wrap it. *)
let echo_words () =
  word_level ~pre:Pre_tokenizer.whitespace ~vocab:letters ~unk_token:"[UNK]"
    ~pad_token:"[PAD]"
    ~post:
      (Post_processor.template ~single:"$A [SEP] $A"
         ~special_tokens:[ ("[SEP]", 9) ]
         ())
    ()

(* Normalizing moves the text around: ["café"] normalizes to ["cafe"], four
   bytes standing for five. *)
let accent_words () =
  word_level
    ~normalizer:
      (Normalizer.sequence [ Normalizer.nfd; Normalizer.strip_accents ])
    ~pre:Pre_tokenizer.whitespace
    ~vocab:(words [ "cafe"; "a"; "b"; "[UNK]"; "[PAD]" ])
    ~unk_token:"[UNK]" ~pad_token:"[PAD]" ()

let g = "\u{120}"

(* A whitespace split, then a byte-level member that marks each word with
   [Ġ]. *)
let split_then_bytes =
  Pre_tokenizer.sequence
    [ Pre_tokenizer.whitespace_split; Pre_tokenizer.byte_level () ]

let byte_unigram () =
  unigram ~pre:split_then_bytes ~pad_token:"<pad>"
    ~vocab:
      [
        ("<unk>", 0.0);
        (g ^ "a", -1.0);
        (g ^ "b", -1.0);
        ("a", -2.0);
        ("b", -2.0);
        (g ^ "ab", -1.5);
      ]
    ~unk_id:0 ()

(* The tokenizers the laws draw, each built once per worker. *)
let law_tokenizers =
  [
    ("bert words", fixture bert_words);
    ("byte-level BPE", fixture byte_bpe);
    ("echo words", fixture echo_words);
    ("accent-free words", fixture accent_words);
    ("byte-level unigram", fixture byte_unigram);
  ]

let tokenizer name = (List.assoc name law_tokenizers) ()

(* Pretrained tokenizers, each loaded once per worker. A download that is
   missing skips the test. *)
let load model =
  Fixture.with_tokenizer_file model (fun path ->
      match from_file path with
      | Ok t -> t
      | Error msg -> failf "failed to load %s: %s" path msg)

let pretrained_fixtures =
  List.map
    (fun model -> (model, fixture (fun () -> load model)))
    [ "gpt2"; "bert_base"; "llama"; "roberta_base"; "llama3"; "gpt_oss" ]

let pretrained model = (List.assoc model pretrained_fixtures) ()

(* The parity corpora, as the documents the parity suite encodes them as. *)
let parity_documents corpus =
  let text =
    Fixture.read (Filename.concat "support/parity" (corpus ^ ".txt"))
  in
  let join current = String.concat "\n" (List.rev current) in
  let rec split docs current = function
    | [] -> List.rev (join current :: docs)
    | "====" :: rest -> split (join current :: docs) [] rest
    | line :: rest -> split docs (line :: current) rest
  in
  split [] [] (String.split_on_char '\n' text)

(* [text] repeated to at least [length] bytes. *)
let repeat text length =
  let buffer = Buffer.create (length + String.length text) in
  while Buffer.length buffer < length do
    Buffer.add_string buffer text
  done;
  Buffer.contents buffer

let kib = 1024

(* Encodings *)

(* Every field of an encoding, its windows included. *)
type fields = {
  ids : int array;
  type_ids : int array;
  tokens : string array;
  offsets : (int * int) array;
  word_ids : int option array;
  special_tokens_mask : int array;
  attention_mask : int array;
  overflowing : fields list;
}

let rec fields e =
  {
    ids = Encoding.ids e;
    type_ids = Encoding.type_ids e;
    tokens = Encoding.tokens e;
    offsets = Encoding.offsets e;
    word_ids = Encoding.word_ids e;
    special_tokens_mask = Encoding.special_tokens_mask e;
    attention_mask = Encoding.attention_mask e;
    overflowing = List.map fields (Encoding.overflowing e);
  }

let rec pp_fields ppf f =
  let row name w v = Format.fprintf ppf "@,%s: %a" name (Testable.pp w) v in
  Format.fprintf ppf "@[<v 2>{";
  row "ids" (array int) f.ids;
  row "type_ids" (array int) f.type_ids;
  row "tokens" (array string) f.tokens;
  row "offsets" (array (pair int int)) f.offsets;
  row "word_ids" (array (option int)) f.word_ids;
  row "special_tokens_mask" (array int) f.special_tokens_mask;
  row "attention_mask" (array int) f.attention_mask;
  List.iter (Format.fprintf ppf "@,window: %a" pp_fields) f.overflowing;
  Format.fprintf ppf "@]@,}"

let encoding = Testable.contramap fields (Testable.structural ~pp:pp_fields)

(* What [f ()] gives, or the exception it raises: two ways of encoding that must
   agree agree on raising too. *)
let outcome f =
  match f () with v -> Ok v | exception e -> Error (Printexc.to_string e)

(* The rows of a flat batch buffer, read back through the lengths that place
   them. Fails unless the buffer holds the rows and nothing else. *)
let rows (buffer, lengths) =
  equal ~msg:"buffer holds the rows and nothing else" int
    (Array.fold_left ( + ) 0 lengths)
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

(* [encode_batch_ids]'s rows against [encode_ids] of each text. *)
let check_batch ?add_special_tokens ?padding ?truncation ?exec ?msg t texts =
  equal ?msg
    (list (array int))
    (List.map (encode_ids t ?add_special_tokens ?padding ?truncation) texts)
    (rows
       (encode_batch_ids t ?add_special_tokens ?padding ?truncation ?exec texts))

(* The pieces [f] encodes into ids buffers, counted through
   [batch_document_hook]: one per document, more for a document cut. *)
let pieces f =
  let count = Atomic.make 0 in
  Atomic.set batch_document_hook (fun () -> Atomic.incr count);
  let result =
    Fun.protect
      ~finally:(fun () -> Atomic.set batch_document_hook (fun () -> ()))
      f
  in
  (result, Atomic.get count)

(* [check_batch] on a batch that a cut leaves with more pieces than
   documents. *)
let check_cut ?truncation ?exec ~msg t texts =
  let (), count =
    pieces (fun () -> check_batch ?truncation ?exec ~msg t texts)
  in
  greater ~msg:(msg ^ ", pieces") int ~than:(List.length texts) count

(* [check_batch] on a batch of one piece per document. *)
let check_whole ?exec ~msg t texts =
  let (), count = pieces (fun () -> check_batch ?exec ~msg t texts) in
  equal ~msg:(msg ^ ", pieces") int (List.length texts) count

(* Generators *)

(* Pieces that break a pipeline: white space runs, characters of two, three and
   four bytes, bytes that open no character, an added token and a pad token. *)
let text =
  Gen.with_pp
    (fun ppf s -> Format.fprintf ppf "%S" s)
    (Gen.map (String.concat "")
       (Gen.list ~size:(Gen.int_range 0 8)
          (Gen.of_list
             [
               "a";
               "b";
               "ab";
               "c d";
               " ";
               "   ";
               "\t\n";
               "caf\u{e9}";
               "\u{65e5}\u{672c}";
               "\u{1f600}";
               "\xff";
               "\x80";
               "\xc3";
               "<s>";
               "[PAD]";
             ])))

let texts = Gen.list ~size:(Gen.int_range 0 5) text

let tokenizer_name =
  Gen.of_list ~pp:Format.pp_print_string (List.map fst law_tokenizers)

let direction_name = function `Left -> "`Left" | `Right -> "`Right"

let length_name = function
  | `Batch_longest -> "`Batch_longest"
  | `Fixed n -> Printf.sprintf "`Fixed %d" n
  | `To_multiple n -> Printf.sprintf "`To_multiple %d" n

(* The optional arguments of [encode] and the batch encoders. *)
type options = {
  add_special_tokens : bool;
  match_special_tokens : bool;
  padding : padding option;
  truncation : truncation option;
}

let pp_options ppf o =
  let padding =
    match o.padding with
    | None -> "none"
    | Some p -> length_name p.length ^ " " ^ direction_name p.direction
  in
  let truncation =
    match o.truncation with
    | None -> "none"
    | Some t ->
        Printf.sprintf "%d, stride %d, %s" t.max_length t.stride
          (direction_name t.direction)
  in
  Format.fprintf ppf "specials %b, matched %b, padding %s, truncation %s"
    o.add_special_tokens o.match_special_tokens padding truncation

(* Padding to a fixed length or a multiple, and to the batch's longest when
   [longest]; truncation with a stride under [max_length] when [stride], with
   none otherwise. *)
let options ~longest ~stride =
  let open Gen in
  let direction = of_list [ `Left; `Right ] in
  let length =
    one_of
      ([
         map (fun n -> `Fixed n) (int_range 0 8);
         map (fun n -> `To_multiple n) (int_range 1 4);
       ]
      @ if longest then [ constant `Batch_longest ] else [])
  in
  let padding =
    let+ length = length and+ direction = direction in
    padding ~direction length
  in
  let truncation =
    let* max_length = int_range 0 8 in
    let+ stride =
      if stride then int_range 0 (max 0 (max_length - 1)) else constant 0
    and+ direction = direction in
    truncation ~stride ~direction max_length
  in
  with_pp pp_options
    (let+ add_special_tokens = bool
     and+ match_special_tokens = bool
     and+ padding = option padding
     and+ truncation = option truncation in
     { add_special_tokens; match_special_tokens; padding; truncation })

let encode_with o ?pair t text =
  encode t ?pair ~add_special_tokens:o.add_special_tokens
    ~match_special_tokens:o.match_special_tokens ?padding:o.padding
    ?truncation:o.truncation text

let encode_ids_with o ?pair t text =
  encode_ids t ?pair ~add_special_tokens:o.add_special_tokens
    ~match_special_tokens:o.match_special_tokens ?padding:o.padding
    ?truncation:o.truncation text

let has_windows = function
  | Ok encodings ->
      List.exists (fun e -> Encoding.overflowing e <> []) encodings
  | Error _ -> false

(* pad_length *)

let test_fixed_pads () =
  equal encoding
    (Encoding.create ~ids:[| 0; 6; 6 |] ~type_ids:[| 0; 0; 0 |]
       ~tokens:[| "a"; "[PAD]"; "[PAD]" |]
       ~words:[| Some 0; None; None |]
       ~offsets:[| (0, 1); (0, 0); (0, 0) |]
       ~special_tokens_mask:[| 0; 1; 1 |] ~attention_mask:[| 1; 0; 0 |] ())
    (encode (bert_words ()) ~add_special_tokens:false
       ~padding:(padding (`Fixed 3))
       "a")

let test_fixed_keeps_longer =
  cases "`Fixed n leaves a sequence of n tokens or more as it is"
    ~id:string_of_int [ 3; 2; 0; -1 ] (fun n ->
      equal (array int) [| 0; 1; 2 |]
        (ids_of (bert_words ()) ~add_special_tokens:false
           ~padding:(padding (`Fixed n))
           "a b c"))

let test_to_multiple =
  prop ~count:200
    "`To_multiple n pads to the smallest multiple of n at least the length"
    Gen.(pair (int_range 1 8) (int_range 0 12))
    (fun (n, k) ->
      cover "already a multiple" (k mod n = 0);
      cover "no token" (k = 0);
      equal int
        ((k + n - 1) / n * n)
        (Array.length
           (ids_of (bert_words ()) ~add_special_tokens:false
              ~padding:(padding (`To_multiple n))
              (text_of_length k))))

let test_to_multiple_nonpositive =
  cases "`To_multiple n with n <= 0 pads nothing" ~id:string_of_int [ 0; -2 ]
    (fun n ->
      equal (array int) [| 0; 1; 2 |]
        (ids_of (bert_words ()) ~add_special_tokens:false
           ~padding:(padding (`To_multiple n))
           "a b c"))

let test_batch_longest () =
  equal
    (list (array int))
    [ [| 0; 6; 6 |]; [| 0; 1; 2 |]; [| 6; 6; 6 |] ]
    (List.map Encoding.ids
       (encode_batch (bert_words ()) ~add_special_tokens:false
          ~padding:(padding `Batch_longest) [ "a"; "a b c"; "" ]))

let test_batch_longest_of_one () =
  let t = bert_words () and padding = padding `Batch_longest in
  equal ~msg:"encode" (array int) [| 8; 0; 1; 9 |] (ids_of t ~padding "a b");
  equal ~msg:"encode_ids" (array int) [| 8; 0; 1; 9 |]
    (encode_ids t ~padding "a b")

(* padding *)

let direction_w =
  Testable.make
    ~pp:(fun ppf d -> Format.pp_print_string ppf (direction_name d))
    ~equal:( = )

let test_padding_defaults () =
  let p = padding (`Fixed 3) in
  equal ~msg:"direction" direction_w `Right p.direction;
  equal ~msg:"pad_id" (option int) None p.pad_id;
  equal ~msg:"pad_type_id" (option int) None p.pad_type_id;
  equal ~msg:"pad_token" (option string) None p.pad_token

let test_padding_left () =
  equal encoding
    (Encoding.create ~ids:[| 6; 6; 0 |] ~type_ids:[| 0; 0; 0 |]
       ~tokens:[| "[PAD]"; "[PAD]"; "a" |]
       ~words:[| None; None; Some 0 |]
       ~offsets:[| (0, 0); (0, 0); (0, 1) |]
       ~special_tokens_mask:[| 1; 1; 0 |] ~attention_mask:[| 0; 0; 1 |] ())
    (encode (bert_words ()) ~add_special_tokens:false
       ~padding:(padding ~direction:`Left (`Fixed 3))
       "a")

(* A pad token the vocabulary holds, and one it lacks, which the constructor
   then adds. *)
let test_padding_tokenizer_pad =
  cases "takes the tokenizer's pad token when it names none"
    ~id:(fun (name, _, _, _) -> name)
    [
      ( "held by the vocabulary",
        [ ("hello", 0); ("world", 1); ("<unk>", 2); ("[PAD]", 3) ],
        "hello",
        [| 0; 3; 3 |] );
      ("lacked by the vocabulary", [ ("a", 0) ], "a", [| 0; 1; 1 |]);
    ]
    (fun (_, vocab, text, expected) ->
      let t =
        word_level ~vocab ~pre:Pre_tokenizer.whitespace
          ~added_tokens:[ added_token "[PAD]" ]
          ~pad_token:"[PAD]" ()
      in
      equal (array int) expected (ids_of t ~padding:(padding (`Fixed 3)) text))

let test_padding_named () =
  let padded t padding =
    let e = encode t ~add_special_tokens:false ~padding "a" in
    (Encoding.ids e, Encoding.type_ids e, Encoding.tokens e)
  in
  let w = triple (array int) (array int) (array string) in
  equal ~msg:"pad_id and pad_token" w
    ([| 0; 5; 5 |], [| 0; 0; 0 |], [| "a"; "f"; "f" |])
    (padded (bert_words ()) (padding ~pad_id:5 ~pad_token:"f" (`Fixed 3)));
  equal ~msg:"pad_type_id" w
    ([| 0; 6; 6 |], [| 0; 7; 7 |], [| "a"; "[PAD]"; "[PAD]" |])
    (padded (bert_words ()) (padding ~pad_type_id:7 (`Fixed 3)));
  equal ~msg:"pad_token, the tokenizer naming none" w
    ([| 0; 6; 6 |], [| 0; 0; 0 |], [| "a"; "[PAD]"; "[PAD]" |])
    (padded (plain_words ()) (padding ~pad_token:"[PAD]" (`Fixed 3)))

let test_padding_without_token =
  cases
    "raises Invalid_argument when neither it nor the tokenizer names a pad \
     token"
    ~id:fst
    [
      ("no field", padding (`Fixed 3));
      ("pad_id alone", padding ~pad_id:6 (`Fixed 3));
      ("pad_type_id alone", padding ~pad_type_id:1 (`Fixed 3));
    ]
    (fun (_, padding) ->
      raises_match Exn.invalid_arg (fun () ->
          encode (plain_words ()) ~padding "a"))

(* truncation *)

let test_truncation_defaults () =
  let t = truncation 5 in
  equal ~msg:"max_length" int 5 t.max_length;
  equal ~msg:"stride" int 0 t.stride;
  equal ~msg:"direction" direction_w `Right t.direction

let test_truncation_refuses =
  cases
    "raises Invalid_argument exactly for a negative stride or one not under a \
     positive max_length"
    ~id:(fun (stride, max_length, _) ->
      Printf.sprintf "stride %d, max_length %d" stride max_length)
    [
      (4, 4, true);
      (5, 4, true);
      (-1, 4, true);
      (-1, 0, true);
      (3, 4, false);
      (0, 4, false);
      (0, 0, false);
      (5, 0, false);
    ]
    (fun (stride, max_length, refused) ->
      let call () = truncation ~stride max_length in
      if refused then raises_match Exn.invalid_arg call
      else equal int stride (call ()).stride)

(* The kept ids against the rule: the special tokens come out of [max_length],
   and a [max_length] at or below their count keeps them alone. *)
let test_truncation_keeps =
  prop ~count:400
    "keeps the first max_length tokens less the special tokens, or the last \
     under `Left"
    Gen.(quad (int_range 0 8) (int_range 0 10) bool bool)
    (fun (k, max_length, left, specials) ->
      let budget = if specials then max 0 (max_length - 2) else max_length in
      cover "truncated" (k > budget);
      cover "fits" (k <= budget);
      cover "no room for content" (specials && max_length <= 2);
      let content = Array.init k (fun i -> i mod 6) in
      let kept =
        if k <= budget then content
        else if left then Array.sub content (k - budget) budget
        else Array.sub content 0 budget
      in
      let expected =
        if specials then Array.concat [ [| 8 |]; kept; [| 9 |] ] else kept
      in
      let direction = if left then `Left else `Right in
      equal (array int) expected
        (ids_of (bert_words ()) ~add_special_tokens:specials
           ~truncation:(truncation ~direction max_length)
           (text_of_length k)))

(* Four tokens against three for a budget of five: the shorter cannot keep all
   three, so the budget is halved and the odd token goes to the first. Removing
   one token at a time from the longer would give it to the second.
   HuggingFace. *)
let test_truncation_pair () =
  let ids max_length =
    ids_of (bert_words ()) ~truncation:(truncation max_length) ~pair:"e f a"
      "a b c d"
  in
  equal ~msg:"three against two" (array int)
    [| 8; 0; 1; 2; 9; 4; 5; 9 |]
    (ids 8);
  equal ~msg:"one each" (array int) [| 8; 0; 9; 4; 9 |] (ids 5)

(* Each window of the first against the second's kept part and then against each
   of its windows, and last the first's kept part against each window of the
   second. HuggingFace. *)
let test_truncation_pair_windows () =
  let windows_of max_length =
    windows
      (encode (bert_words ()) ~truncation:(truncation max_length) ~pair:"e f a"
         "a b c d")
  in
  equal ~msg:"three against two"
    (list (array int))
    [ [| 8; 3; 9; 4; 5; 9 |]; [| 8; 3; 9; 0; 9 |]; [| 8; 0; 1; 2; 9; 0; 9 |] ]
    (windows_of 8);
  equal ~msg:"one each"
    (list (array int))
    [
      [| 8; 1; 9; 4; 9 |];
      [| 8; 1; 9; 5; 9 |];
      [| 8; 1; 9; 0; 9 |];
      [| 8; 2; 9; 4; 9 |];
      [| 8; 2; 9; 5; 9 |];
      [| 8; 2; 9; 0; 9 |];
      [| 8; 3; 9; 4; 9 |];
      [| 8; 3; 9; 5; 9 |];
      [| 8; 3; 9; 0; 9 |];
      [| 8; 0; 9; 5; 9 |];
      [| 8; 0; 9; 0; 9 |];
    ]
    (windows_of 5)

let test_truncation_wraps_windows () =
  let e = encode (bert_words ()) ~truncation:(truncation 4) "a b c d" in
  equal ~msg:"windows" (list (array int)) [ [| 8; 2; 3; 9 |] ] (windows e);
  equal ~msg:"a window's special tokens mask" (array int) [| 1; 0; 0; 1 |]
    (Encoding.special_tokens_mask (List.hd (Encoding.overflowing e)))

(* HuggingFace. *)
let test_truncation_stride () =
  let windows_of truncation =
    windows (encode (bert_words ()) ~truncation "a b c d e f")
  in
  equal ~msg:"stride 1"
    (list (array int))
    [ [| 8; 1; 2; 9 |]; [| 8; 2; 3; 9 |] ]
    (windows_of (truncation ~stride:1 4));
  equal ~msg:"stride 2"
    (list (array int))
    [ [| 8; 1; 2; 3; 9 |]; [| 8; 2; 3; 4; 9 |] ]
    (windows_of (truncation ~stride:2 5))

(* [e] and [f] lie past the pretoken holding the fourth token; under [`Left], [a
   b] lie before the pretoken holding the fourth from the end, and with no
   post-processor nothing is left to window. HuggingFace. *)
let test_truncation_pretokens () =
  let right = encode (bert_words ()) ~truncation:(truncation 4) "a b c d e f" in
  equal ~msg:"`Right" (list (array int)) [ [| 8; 2; 3; 9 |] ] (windows right);
  let left =
    encode (plain_words ())
      ~truncation:(truncation ~direction:`Left 2)
      "a b c d"
  in
  equal ~msg:"`Left, kept" (array int) [| 2; 3 |] (Encoding.ids left);
  equal ~msg:"`Left, windows" (list (array int)) [] (windows left)

(* support/parity/stride_windows.txt, which gen/gen_stride_expected.py writes
   from HuggingFace: after a comment, the single text and the pair's two texts,
   then a line per case: model, [single] or [pair], max_length, stride,
   direction, then the kept ids and each window's, each after a [|]. *)
let test_truncation_hf_windows =
  cases "gives the windows HuggingFace gives" ~id:Fun.id [ "gpt2"; "bert_base" ]
    (fun model ->
      let t = pretrained model in
      let file = Fixture.read "support/parity/stride_windows.txt" in
      let single, first, second, cases =
        match String.split_on_char '\n' (String.trim file) with
        | _ :: single :: first :: second :: cases ->
            (single, first, second, cases)
        | _ -> failf "stride_windows.txt holds no case"
      in
      let ids row =
        Array.of_list
          (List.map int_of_string
             (List.filter (( <> ) "") (String.split_on_char ' ' row)))
      in
      let check line =
        Scanf.sscanf line "%_s %s %d %d %s | %[^\n]"
          (fun mode max_length stride direction rows ->
            let direction = if direction = "left" then `Left else `Right in
            let truncation = truncation ~stride ~direction max_length in
            let e =
              if mode = "single" then encode t ~truncation single
              else encode t ~truncation ~pair:second first
            in
            equal ~msg:line
              (list (array int))
              (List.map ids (String.split_on_char '|' rows))
              (Encoding.ids e :: windows e))
      in
      let mine = List.filter (String.starts_with ~prefix:(model ^ " ")) cases in
      greater ~msg:"cases" int ~than:0 (List.length mine);
      List.iter check mine)

(* The room for content is [max_length] less the two special tokens, and on the
   pair below, whose budget of six splits three and two, each sequence's
   share. *)
let test_truncation_stride_room =
  cases
    "raises Invalid_argument when it truncates with a stride not under the \
     room for content"
    ~id:(fun (name, _, _, _, _) -> name)
    [
      ("stride at the room", 2, 4, None, true);
      ("stride under the room", 1, 4, None, false);
      ("stride at the second's share", 2, 8, Some "e f a", true);
      ("stride under both shares", 1, 8, Some "e f a", false);
    ]
    (fun (_, stride, max_length, pair, refused) ->
      let call () =
        encode (bert_words ()) ?pair
          ~truncation:(truncation ~stride max_length)
          "a b c d"
      in
      if refused then raises_match Exn.invalid_arg call
      else equal int max_length (Encoding.length (call ())))

(* encode *)

(* HuggingFace on a [BPE] model of the letters with [<s>] added. *)
let test_encode_added () =
  let t =
    bpe
      ~vocab:[ ("a", 0); ("b", 1) ]
      ~merges:[]
      ~added_tokens:[ added_token "<s>" ]
      ()
  in
  List.iter
    (fun add_special_tokens ->
      let msg = Printf.sprintf "add_special_tokens %b" add_special_tokens in
      equal ~msg (array int) [| 0; 2; 1 |]
        (ids_of t ~add_special_tokens "a<s>b");
      equal ~msg:(msg ^ ", encode_ids") (array int) [| 0; 2; 1 |]
        (encode_ids t ~add_special_tokens "a<s>b"))
    [ false; true ]

(* HuggingFace's [test_encode_special_tokens] in [added_vocabulary.rs]: with
   [encode_special_tokens] set, the special [<mask>] and [<pad>] are passed
   over, while [ask>], matched against normalized text, is still found inside
   the text of [<mask>]. *)
let mask_tokenizer () =
  let chars = "hi<mask>terpd \t\u{2000}" in
  let rec scalars i =
    if i >= String.length chars then []
    else
      let n = Uchar.utf_decode_length (String.get_utf_8_uchar chars i) in
      String.sub chars i n :: scalars (i + n)
  in
  bpe ~normalizer:Normalizer.lowercase
    ~vocab:(words (scalars 0))
    ~merges:[]
    ~added_tokens:
      [
        added_token ~lstrip:true ~rstrip:true ~single_word:true "<mask>";
        added_token ~special:false "ask>";
        added_token "<pad>";
      ]
    ()

let mask_text = "Hi <mask> there\t<mask>\t<mask>\u{2000} <pad> <mask><pad><pad>"

let test_encode_unmatched () =
  let t = mask_tokenizer () in
  let pieces =
    [ "h"; "i"; " "; "<"; "m"; "ask>"; " "; "t"; "h"; "e"; "r"; "e"; "\t"; "<";
      "m"; "ask>"; "\t"; "<"; "m"; "ask>"; "\u{2000}"; " "; "<"; "p"; "a"; "d";
      ">"; " "; "<"; "m"; "ask>"; "<"; "p"; "a"; "d"; ">"; "<"; "p"; "a"; "d";
      ">" ]
  [@ocamlformat "disable"]
  in
  let e = encode t ~match_special_tokens:false mask_text in
  equal ~msg:"tokens" (list string) pieces (Array.to_list (Encoding.tokens e));
  equal ~msg:"ids" (list int)
    (List.map (fun piece -> Option.get (token_to_id t piece)) pieces)
    (Array.to_list (Encoding.ids e));
  equal ~msg:"encode_ids" (array int) (Encoding.ids e)
    (encode_ids t ~match_special_tokens:false mask_text)

let test_encode_matched_by_default () =
  equal (list string)
    ([ "h"; "i"; " <mask> "; "t"; "h"; "e"; "r"; "e"; "\t<mask>\t";
       "<mask>\u{2000} "; "<pad>"; " <mask>"; "<pad>"; "<pad>" ]
    [@ocamlformat "disable"])
    (Array.to_list (Encoding.tokens (encode (mask_tokenizer ()) mask_text)))

(* A byte-level vocabulary of bytes of characters cut short. *)
let stray_bytes pre =
  bpe ~pre ~merges:[]
    ~vocab:
      (List.map
         (fun c -> (byte_char c, Char.code c))
         [ 'R'; 'a'; '\x80'; '\xC3'; '\xA9'; '\xE2'; '\x82' ])
    ()

let cl100k =
  {re|(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+|re}

(* A continuation byte with no lead and a lead cut short stand for themselves;
   the bytes of a character all stand for the character. Through the GPT-2
   walker and through a cl100k split. *)
let test_encode_stray_bytes =
  let gpt2 =
    lazy (stray_bytes (Pre_tokenizer.byte_level ~add_prefix_space:false ()))
  in
  let cl100k =
    lazy
      (stray_bytes
         (Pre_tokenizer.sequence
            [
              Pre_tokenizer.split_regex ~pattern:cl100k ~behavior:`Isolated ();
              Pre_tokenizer.byte_level ~add_prefix_space:false ~use_regex:false
                ();
            ]))
  in
  cases "reads a byte that opens no character as a character of one byte"
    ~id:(fun (pipeline, _, text, _) -> Printf.sprintf "%s %S" pipeline text)
    (List.concat_map
       (fun (pipeline, t) ->
         [
           (pipeline, t, "R\x80", [| (0, 1); (1, 2) |]);
           (pipeline, t, "R\x80\x80a", [| (0, 1); (1, 2); (2, 3); (3, 4) |]);
           (pipeline, t, "\xC3\xA9\x80", [| (0, 2); (0, 2); (2, 3) |]);
           (pipeline, t, "a\xE2\x82", [| (0, 1); (1, 2); (2, 3) |]);
           (pipeline, t, "\xE2\x82R", [| (0, 1); (1, 2); (2, 3) |]);
         ])
       [ ("GPT-2", gpt2); ("cl100k", cl100k) ])
    (fun (_, t, text, expected) ->
      equal (array (pair int int)) expected (offsets_of (Lazy.force t) text))

let test_encode_normalized_offsets () =
  let t = accent_words () in
  equal ~msg:"ids" (array int) [| 0; 1 |]
    (ids_of t ~add_special_tokens:false "caf\u{e9} a");
  equal ~msg:"offsets"
    (array (pair int int))
    [| (0, 5); (6, 7) |]
    (offsets_of t "caf\u{e9} a")

(* The space a byte-level pre-tokenizer marks as [Ġ] is one byte of the
   input. *)
let test_encode_byte_level_offsets () =
  let t =
    bpe
      ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
      ~vocab:[ ("a", 0); (g ^ "b", 1); (g, 2); ("b", 3) ]
      ~merges:[ (g, "b") ]
      ()
  in
  equal ~msg:"ids" (array int) [| 0; 1 |]
    (ids_of t ~add_special_tokens:false "a b");
  equal ~msg:"offsets"
    (array (pair int int))
    [| (0, 1); (1, 3) |]
    (offsets_of t "a b")

let test_encode_word_ids () =
  let t =
    word_level ~pre:Pre_tokenizer.whitespace
      ~added_tokens:[ added_token "<s>" ]
      ~vocab:(words [ "a"; "b" ])
      ()
  in
  let e = encode t ~add_special_tokens:false "a <s> b" in
  equal ~msg:"word ids"
    (array (option int))
    [| Some 0; Some 1; Some 2 |]
    (Encoding.word_ids e);
  equal ~msg:"offsets"
    (array (pair int int))
    [| (0, 1); (2, 5); (6, 7) |]
    (Encoding.offsets e)

(* With no pre-tokenizer the text is one pretoken, its tokens at the bytes of
   their pieces. *)
let test_encode_one_pretoken () =
  let e = encode (bpe ~vocab:[ ("a", 0); ("b", 1) ] ~merges:[] ()) "ab" in
  equal ~msg:"word ids"
    (array (option int))
    [| Some 0; Some 0 |] (Encoding.word_ids e);
  equal ~msg:"offsets"
    (array (pair int int))
    [| (0, 1); (1, 2) |]
    (Encoding.offsets e)

(* [z] gives no token, and [b] keeps its number. HuggingFace. *)
let test_encode_word_gap () =
  let t =
    bpe ~pre:Pre_tokenizer.whitespace
      ~vocab:[ ("a", 0); ("b", 1) ]
      ~merges:[] ()
  in
  equal
    (array (option int))
    [| Some 0; Some 2 |]
    (Encoding.word_ids (encode t "a z b"))

(* A metaspace after a whitespace split marks each word on its own, and a marker
   that was prepended stands at the character it opens. HuggingFace. *)
let m = "\u{2581}"

(* Rows of a table over several tokenizers: each tokenizer, built when a row
   first needs it, with its rows. *)
let per_tokenizer groups =
  List.concat_map
    (fun (name, t, rows) ->
      List.map (fun (text, a, b) -> (name, t, text, a, b)) rows)
    groups

let row_id (name, _, text, _, _) = Printf.sprintf "%s %S" name text

let rewriting members =
  lazy
    (unigram
       ~pre:
         (Pre_tokenizer.sequence
            ([ Pre_tokenizer.whitespace_split; Pre_tokenizer.metaspace () ]
            @ members))
       ~vocab:
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
       ~unk_id:0 ())

let check_tokens_and_offsets (_, t, text, tokens, offsets) =
  let t = Lazy.force t in
  equal ~msg:"tokens" (array string) tokens (tokens_of t text);
  equal ~msg:"offsets" (array (pair int int)) offsets (offsets_of t text)

let test_encode_rewriting =
  cases "places the tokens of a rewriting pre-tokenizer as HuggingFace does"
    ~id:row_id
    (per_tokenizer
       [
         ( "split, metaspace",
           rewriting [],
           [
             ("", [||], [||]);
             ("a b", [| m ^ "a"; m ^ "b" |], [| (0, 1); (2, 3) |]);
             (" a", [| m ^ "a" |], [| (1, 2) |]);
             ("a ", [| m ^ "a" |], [| (0, 1) |]);
             ("a  b", [| m ^ "a"; m ^ "b" |], [| (0, 1); (3, 4) |]);
             ("ab", [| m ^ "a"; "b" |], [| (0, 1); (1, 2) |]);
             ("a!b", [| m ^ "a"; "!"; "b" |], [| (0, 1); (1, 2); (2, 3) |]);
             (m ^ "a", [| m ^ "a" |], [| (0, 4) |]);
             (* The unknown token stands for the whole run no piece covers. *)
             ( "\u{65e5}\u{672c} \u{8a9e}",
               [| m; "<unk>"; m; "<unk>" |],
               [| (0, 3); (0, 6); (7, 10); (7, 10) |] );
           ] );
         ( "split, metaspace, punctuation",
           rewriting [ Pre_tokenizer.punctuation () ],
           [
             ("a!b", [| m ^ "a"; "!"; "b" |], [| (0, 1); (1, 2); (2, 3) |]);
             ("b!", [| m ^ "b"; "!" |], [| (0, 1); (1, 2) |]);
           ] );
       ])
    check_tokens_and_offsets

(* A model other than BPE cannot match raw bytes, so behind a byte-level
   pre-tokenizer it matches its vocabulary in byte-level form. HuggingFace. *)
let test_encode_byte_level_models =
  cases "hands a model other than BPE the byte-level form of its pieces"
    ~id:row_id
    (per_tokenizer
       [
         ( "unigram",
           lazy (byte_unigram ()),
           [
             ("a b", [| g ^ "a"; g ^ "b" |], [| (0, 1); (2, 3) |]);
             ("ab b", [| g ^ "ab"; g ^ "b" |], [| (0, 2); (3, 4) |]);
             (" a", [| g ^ "a" |], [| (1, 2) |]);
           ] );
         ( "wordpiece",
           lazy
             (wordpiece ~pre:split_then_bytes
                ~vocab:(words [ "[UNK]"; g ^ "a"; g ^ "b"; "##b"; g ^ "ab" ])
                ~unk_token:"[UNK]" ()),
           [ ("a b", [| g ^ "a"; g ^ "b" |], [| (0, 1); (2, 3) |]) ] );
         ( "word_level",
           lazy
             (word_level ~pre:split_then_bytes
                ~vocab:(words [ "[UNK]"; g ^ "a"; g ^ "b"; g ^ "ab" ])
                ~unk_token:"[UNK]" ()),
           [
             ("a b", [| g ^ "a"; g ^ "b" |], [| (0, 1); (2, 3) |]);
             ("ab b", [| g ^ "ab"; g ^ "b" |], [| (0, 2); (3, 4) |]);
           ] );
         ( "unigram, byte level alone",
           lazy
             (unigram
                ~pre:(Pre_tokenizer.byte_level ())
                ~vocab:[ ("<unk>", 0.0); (g ^ "a", -1.0); (g ^ "b", -1.0) ]
                ~unk_id:0 ()),
           [ ("a b", [| g ^ "a"; g ^ "b" |], [| (0, 1); (1, 3) |]) ] );
         ( "unigram, no prefix space",
           lazy
             (unigram
                ~pre:
                  (Pre_tokenizer.sequence
                     [
                       Pre_tokenizer.whitespace_split;
                       Pre_tokenizer.byte_level ~add_prefix_space:false ();
                     ])
                ~vocab:
                  [ ("<unk>", 0.0); ("a", -1.0); ("b", -1.0); ("ab", -1.5) ]
                ~unk_id:0 ()),
           [
             ("a b", [| "a"; "b" |], [| (0, 1); (2, 3) |]);
             ("ab b", [| "ab"; "b" |], [| (0, 2); (3, 4) |]);
           ] );
       ])
    check_tokens_and_offsets

(* A run of characters no entry covers is one unknown token, or one per
   character with fusing off, standing for the whole run. HuggingFace, which
   spells a Unigram unknown token as the run where brot gives the entry, so ids
   are compared and not tokens. *)
let test_encode_unknown_runs =
  let zw = "\u{200b}\u{200c}" in
  let run = "a" ^ zw ^ "b" ^ zw ^ "c" in
  let fused = [| (0, 1); (1, 7); (7, 8); (8, 14); (14, 15) |] in
  let letters_bpe fuse_unk =
    lazy
      (bpe
         ~vocab:[ ("<unk>", 0); ("a", 1); ("b", 2); ("c", 3) ]
         ~merges:[] ~unk_token:"<unk>" ~fuse_unk ())
  in
  cases "places a run of unknown characters as HuggingFace does" ~id:row_id
    (per_tokenizer
       [
         ( "unigram",
           lazy
             (unigram
                ~vocab:[ ("<unk>", 0.0); ("a", -1.0); ("b", -1.0); ("c", -1.0) ]
                ~unk_id:0 ()),
           [ (run, [| 1; 0; 2; 0; 3 |], fused) ] );
         ("bpe, fused", letters_bpe true, [ (run, [| 1; 0; 2; 0; 3 |], fused) ]);
         ( "bpe, unfused",
           letters_bpe false,
           [
             ( run,
               [| 1; 0; 0; 2; 0; 0; 3 |],
               [| (0, 1); (1, 4); (4, 7); (7, 8); (8, 11); (11, 14); (14, 15) |]
             );
           ] );
         (* The unknown token's own entry, matched literally, is placed as any
            entry, alone or in a fused run that spells one. *)
         ( "unigram, literal <unk>",
           lazy
             (unigram
                ~vocab:[ ("<unk>", 0.0); ("a", -1.0); ("b", -1.0) ]
                ~unk_id:0 ()),
           [
             ("a<unk>b", [| 1; 0; 2 |], [| (0, 1); (1, 6); (6, 7) |]);
             ( "a<unk>" ^ zw ^ "b",
               [| 1; 0; 2 |],
               [| (0, 1); (1, 12); (12, 13) |] );
           ] );
         ( "wordpiece",
           lazy
             (wordpiece ~pre:Pre_tokenizer.whitespace_split
                ~vocab:(words [ "[UNK]"; "a"; "b"; "##b" ])
                ~unk_token:"[UNK]" ()),
           [
             ( "ab xyz ba",
               [| 1; 3; 0; 0 |],
               [| (0, 1); (1, 2); (3, 6); (7, 9) |] );
           ] );
         ( "word_level",
           lazy
             (word_level ~pre:Pre_tokenizer.whitespace_split
                ~vocab:(words [ "<unk>"; "a" ])
                ()),
           [ ("a xyz <unk>", [| 1; 0; 0 |], [| (0, 1); (2, 5); (6, 11) |]) ] );
       ])
    (fun (_, t, text, ids, offsets) ->
      let t = Lazy.force t in
      equal ~msg:"ids" (array int) ids (ids_of t ~add_special_tokens:false text);
      equal ~msg:"offsets" (array (pair int int)) offsets (offsets_of t text))

(* A byte-level member after a whitespace split prepends its space to each word,
   and the space stands at the character it opens. HuggingFace. *)
let test_encode_byte_level_after_split =
  cases "places the space a byte-level member adds at the word it opens"
    ~id:(fun (text, _, _) -> Printf.sprintf "%S" text)
    [
      ("a b", [| g; "a"; g; "b" |], [| (0, 1); (0, 1); (2, 3); (2, 3) |]);
      (" a  b", [| g; "a"; g; "b" |], [| (1, 2); (1, 2); (4, 5); (4, 5) |]);
      ("ab", [| g; "a"; "b" |], [| (0, 1); (0, 1); (1, 2) |]);
    ]
    (fun (text, tokens, offsets) ->
      let t =
        bpe ~pre:split_then_bytes
          ~vocab:[ (g ^ "a", 0); (g ^ "b", 1); ("a", 2); ("b", 3); (g, 4) ]
          ~merges:[] ()
      in
      equal ~msg:"tokens" (array string) tokens (tokens_of t text);
      equal ~msg:"offsets" (array (pair int int)) offsets (offsets_of t text))

(* An unknown token stands for what the ids around it do not describe. *)
let test_encode_unknown_token () =
  let t =
    bpe
      ~vocab:[ ("<unk>", 100); ("<0x61>", 0); ("b", 1) ]
      ~merges:[] ~unk_token:"<unk>" ~byte_fallback:true ()
  in
  equal ~msg:"tokens" (array string)
    [| "<0x61>"; "<unk>"; "b" |]
    (tokens_of t "zab");
  equal ~msg:"offsets"
    (array (pair int int))
    [| (0, 1); (1, 2); (2, 3) |]
    (offsets_of t "zab")

(* Starts and ends each never decrease. *)
let ascending spans =
  let rec from i =
    i + 1 >= Array.length spans
    ||
    let (s, e), (s', e') = (spans.(i), spans.(i + 1)) in
    s <= s' && e <= e' && from (i + 1)
  in
  from 0

(* Every law tokenizer but the one whose template names the sequence twice,
   which repeats its spans. *)
let test_encode_spans =
  prop ~count:300 "gives ascending byte spans of the text"
    Gen.(
      pair
        (of_list ~pp:Format.pp_print_string
           (List.filter (( <> ) "echo words") (List.map fst law_tokenizers)))
        text)
    (fun (name, text) ->
      let e = encode (tokenizer name) ~add_special_tokens:false text in
      cover "a byte that opens no character" (not (String.is_valid_utf_8 text));
      cover "a character outside the BMP" (String.exists (( = ) '\xf0') text);
      cover "no token" (Encoding.length e = 0);
      let n = String.length text in
      satisfies ~claim:"spans within the text, ascending"
        (array (pair int int))
        (fun spans ->
          Array.for_all (fun (s, e) -> 0 <= s && s <= e && e <= n) spans
          && ascending spans)
        (Encoding.offsets e))

(* The same pipeline as [bert_words] without its post-processor. *)
let test_encode_no_specials =
  prop "adds nothing when add_special_tokens is false" text (fun text ->
      let unwrapped =
        word_level ~pre:Pre_tokenizer.whitespace ~vocab:letters
          ~unk_token:"[UNK]" ~pad_token:"[PAD]" ()
      in
      equal (array int) (ids_of unwrapped text)
        (ids_of (bert_words ()) ~add_special_tokens:false text))

(* [[CLS]] and [[SEP]] cover nothing and belong to no word; the letters keep
   theirs. *)
let test_encode_inserted () =
  let e = encode (bert_words ()) "a b" in
  equal ~msg:"word ids"
    (array (option int))
    [| None; Some 0; Some 1; None |]
    (Encoding.word_ids e);
  equal ~msg:"offsets"
    (array (pair int int))
    [| (0, 0); (0, 1); (2, 3); (0, 0) |]
    (Encoding.offsets e)

let test_encode_pair () =
  let e = encode (bert_words ()) ~pair:"c" "a b" in
  equal ~msg:"ids" (array int) [| 8; 0; 1; 9; 2; 9 |] (Encoding.ids e);
  equal ~msg:"type ids" (array int) [| 0; 0; 0; 0; 1; 1 |] (Encoding.type_ids e)

(* exec *)

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

let spread_tokenizer () =
  word_level
    ~vocab:
      (words
         [
           "hello"; "world"; "the"; "cat"; "sat"; "on"; "mat"; "[UNK]"; "[PAD]";
         ])
    ~pre:Pre_tokenizer.whitespace ~unk_token:"[UNK]" ~pad_token:"[PAD]"
    ~post:
      (Post_processor.template ~single:"[UNK] $A"
         ~special_tokens:[ ("[UNK]", 7) ]
         ())
    ()

(* Long documents that cut into many chunks among short ones, so that every task
   of an executor has chunks to take. *)
let spread_documents () =
  [ repeat "hello world the cat sat on the mat " (256 * kib); "the cat"; "" ]
  @ List.init 24 (fun j -> repeat "cat sat on the mat hello world " (j * 2048))

let test_exec_once () =
  let t = spread_tokenizer () and documents = spread_documents () in
  let calls = ref 0 in
  let exec n task =
    incr calls;
    Fixture.spawn 2 n task
  in
  let once msg encode_batch =
    calls := 0;
    encode_batch exec;
    equal ~msg int 1 !calls
  in
  once "encode_batch" (fun exec ->
      ignore (encode_batch t ~exec documents : _ list));
  once "encode_batch_pairs" (fun exec ->
      let pairs = List.map (fun d -> (d, "the cat")) documents in
      ignore (encode_batch_pairs t ~exec pairs : _ list));
  once "encode_batch_ids" (fun exec ->
      ignore (encode_batch_ids t ~exec documents : ids * _))

(* 65535 bytes, in two texts or in one pair. *)
let test_exec_small () =
  let t = spread_tokenizer () in
  let text = String.sub (repeat "the cat " (64 * kib)) 0 ((64 * kib) - 1) in
  let first = String.sub text 0 40_000 in
  let second = String.sub text 40_000 (String.length text - 40_000) in
  let exec _ _ = fail "exec was called" in
  equal ~msg:"encode_batch"
    (list (array int))
    [ encode_ids t first; encode_ids t second ]
    (List.map Encoding.ids (encode_batch t ~exec [ first; second ]));
  equal ~msg:"encode_batch_pairs"
    (list (array int))
    [ encode_ids t ~pair:second first ]
    (List.map Encoding.ids (encode_batch_pairs t ~exec [ (first, second) ]));
  check_batch ~msg:"encode_batch_ids" ~exec t [ first; second ]

let test_exec_task_count () =
  let most = Domain.recommended_domain_count () in
  let asked = ref 0 and worked = ref 0 in
  let exec n task =
    asked := n;
    sequential n task
  in
  run_chunks ~exec (Array.make (most + 50) 1) (fun _ -> incr worked);
  equal ~msg:"tasks" int most !asked;
  equal ~msg:"chunks run" int (most + 50) !worked

(* A document the model has no piece for fails its chunk. *)
let failing_tokenizer () =
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

let good_documents () =
  List.init 16 (fun _ -> repeat "hello world " (128 * kib))

let bad_documents () =
  good_documents () @ [ "hello world zzz" ] @ good_documents ()

let test_exec_tasks_never_raise () =
  let escaped = ref [] in
  let exec n task =
    for i = 0 to n - 1 do
      match task i with
      | () -> ()
      | exception e -> escaped := Printexc.to_string e :: !escaped
    done
  in
  raises_match Exn.failure (fun () ->
      encode_batch_ids (failing_tokenizer ()) ~exec (bad_documents ()));
  equal ~msg:"raised by a task" (list string) [] !escaped

let test_exec_same_ids =
  cases "gives the same ids in any order and on any domain" ~id:fst
    [
      ("sequential, last task first", sequential);
      ("on other domains", elsewhere);
      ("on three domains, the caller's among them", Fixture.spawn 3);
    ]
    (fun (_, exec) ->
      let t = spread_tokenizer () and documents = spread_documents () in
      check_batch ~msg:"ids" ~exec t documents;
      check_batch ~msg:"ids, padded and truncated" ~exec
        ~padding:(padding ~direction:`Left (`To_multiple 8))
        ~truncation:(truncation ~direction:`Left 100000)
        t documents;
      equal ~msg:"encodings"
        (list (array int))
        (List.map Encoding.ids (encode_batch t documents))
        (List.map Encoding.ids (encode_batch t ~exec documents)))

let test_exec_failed_chunk () =
  let t = failing_tokenizer () and exec = Fixture.spawn 6 in
  raises_match ~msg:"encode_batch_ids" Exn.failure (fun () ->
      encode_batch_ids t ~exec (bad_documents ()));
  raises_match ~msg:"encode_batch" Exn.failure (fun () ->
      encode_batch t ~exec (bad_documents ()))

let first_then_exit n task =
  if n > 0 then task 0;
  raise Exit

let returns_at_once _ _ = ()

let test_exec_raised () =
  let t = spread_tokenizer () and documents = spread_documents () in
  raises ~msg:"encode_batch_ids" Exit (fun () ->
      encode_batch_ids t ~exec:first_then_exit documents);
  raises ~msg:"encode_batch" Exit (fun () ->
      encode_batch t ~exec:first_then_exit documents)

let test_exec_returned_early () =
  let t = spread_tokenizer () and documents = spread_documents () in
  let refused =
    Invalid_argument "exec returned before every chunk was encoded"
  in
  raises ~msg:"encode_batch_ids" refused (fun () ->
      encode_batch_ids t ~exec:returns_at_once documents);
  raises ~msg:"encode_batch" refused (fun () ->
      encode_batch t ~exec:returns_at_once documents)

let test_exec_after_failure () =
  let fails msg t documents exec =
    is_error ~msg (outcome (fun () -> encode_batch_ids t ~exec documents))
  in
  let t = failing_tokenizer () in
  fails "a chunk raised" t (bad_documents ()) (Fixture.spawn 6);
  check_batch ~msg:"after a chunk raised" ~exec:(Fixture.spawn 4) t
    (good_documents ());
  let t = spread_tokenizer () and documents = spread_documents () in
  fails "exec raised" t documents first_then_exit;
  check_batch ~msg:"after exec raised" ~exec:(Fixture.spawn 4) t documents;
  fails "exec returned early" t documents returns_at_once;
  check_batch ~msg:"after exec returned early" ~exec:(Fixture.spawn 4) t
    documents

let test_exec_stops_handout () =
  let calls = ref 0 in
  let work _ =
    incr calls;
    if !calls = 3 then failwith "chunk"
  in
  raises ~msg:"the batch fails" (Failure "chunk") (fun () ->
      run_chunks ~exec:sequential (Array.make 64 1) work);
  equal ~msg:"chunks run" int 3 !calls

(* The caller's task holds a chunk while the first chunk a spawned domain takes
   raises. Once that task has returned the failure is out, and the caller
   finishes its chunk and takes no other. A spawned domain's failure once left
   the others to run the whole batch. *)
let test_exec_stops_tasks () =
  let caller = Domain.self () in
  let holding = Atomic.make false and failed = Atomic.make false in
  let worked = Atomic.make 0 in
  let work _ =
    if Domain.self () <> caller then failwith "chunk"
    else begin
      Atomic.set holding true;
      while not (Atomic.get failed) do
        Domain.cpu_relax ()
      done;
      Atomic.incr worked
    end
  in
  let exec n task =
    if n < 2 then skip ~reason:"one domain is recommended" ();
    let failing =
      Domain.spawn (fun () ->
          while not (Atomic.get holding) do
            Domain.cpu_relax ()
          done;
          task 0;
          Atomic.set failed true)
    in
    task 1;
    Domain.join failing;
    for i = 2 to n - 1 do
      task i
    done
  in
  raises ~msg:"the batch fails" (Failure "chunk") (fun () ->
      run_chunks ~exec (Array.make 64 1) work);
  equal ~msg:"chunks the caller ran" int 1 (Atomic.get worked)

(* The executor hands its tasks to a domain held until the batch has failed. *)
let test_exec_late_tasks =
  cases "gives no chunk to a task run after the batch failed" ~id:fst
    [ ("exec returned early", Fun.id); ("exec raised", fun () -> raise Exit) ]
    (fun (_, return) ->
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
        return ()
      in
      let result =
        outcome (fun () ->
            run_chunks ~exec (Array.make 8 1) (fun _ -> Atomic.incr worked))
      in
      Atomic.set release true;
      Option.iter Domain.join !held;
      is_error ~msg:"the batch" result;
      equal ~msg:"chunks run" int 0 (Atomic.get worked))

(* encode_batch *)

let batch_case =
  Gen.(triple tokenizer_name texts (options ~longest:false ~stride:true))

(* The example truncates with a stride at the room for content. *)
let test_batch_is_encode =
  prop ~count:200 "is encode of each text" batch_case
    ~examples:
      [
        ( "bert words",
          [ "a b c" ],
          {
            add_special_tokens = true;
            match_special_tokens = true;
            padding = None;
            truncation = Some (truncation ~stride:1 3);
          } );
      ]
    (fun (name, texts, o) ->
      let t = tokenizer name in
      let each = outcome (fun () -> List.map (encode_with o t) texts) in
      cover "no text" (texts = []);
      cover "windows" (has_windows each);
      cover "padding" (Option.is_some o.padding);
      cover "a raise" (Result.is_error each);
      equal
        (result (list encoding) string)
        each
        (outcome (fun () ->
             encode_batch t ~add_special_tokens:o.add_special_tokens
               ~match_special_tokens:o.match_special_tokens ?padding:o.padding
               ?truncation:o.truncation texts)))

(* encode_batch_pairs *)

let test_batch_pairs_is_encode =
  prop ~count:200 "is encode ~pair of each pair"
    Gen.(
      triple tokenizer_name
        (list ~size:(int_range 0 4) (pair text text))
        (options ~longest:false ~stride:true))
    (fun (name, pairs, o) ->
      let t = tokenizer name in
      let each =
        outcome (fun () ->
            List.map (fun (text, pair) -> encode_with o ~pair t text) pairs)
      in
      cover "no pair" (pairs = []);
      cover "windows" (has_windows each);
      cover "a raise" (Result.is_error each);
      equal
        (result (list encoding) string)
        each
        (outcome (fun () ->
             encode_batch_pairs t ~add_special_tokens:o.add_special_tokens
               ~match_special_tokens:o.match_special_tokens ?padding:o.padding
               ?truncation:o.truncation pairs)))

(* encode_ids *)

let test_ids_is_encode =
  prop ~count:300 "is the ids of encode"
    Gen.(
      quad tokenizer_name text (option text)
        (options ~longest:true ~stride:false))
    (fun (name, text, pair, o) ->
      let t = tokenizer name in
      let expected =
        outcome (fun () -> Encoding.ids (encode_with o ?pair t text))
      in
      cover "a pair" (Option.is_some pair);
      cover "padded" (Result.is_ok expected && Option.is_some o.padding);
      cover "truncated"
        (match (o.truncation, expected) with
        | Some _, Ok _ ->
            not (Encoding.overflowing (encode_with o ?pair t text) = [])
        | _ -> false);
      equal
        (result (array int) string)
        expected
        (outcome (fun () -> encode_ids_with o ?pair t text)))

(* A single template naming [$B], which has no sequence to put there, so
   [encode] raises. *)
let b_alone () =
  word_level ~pre:Pre_tokenizer.whitespace ~vocab:letters ~unk_token:"[UNK]"
    ~post:(Post_processor.template ~single:"$B" ())
    ()

let test_ids_raise () =
  let t = b_alone () in
  raises_match Exn.invalid_arg (fun () -> encode t "a");
  raises_match Exn.invalid_arg (fun () -> encode_ids t "a")

(* encode_batch_ids *)

let test_batch_ids_raise () =
  let t = b_alone () in
  raises_match Exn.invalid_arg (fun () -> encode t "a");
  raises_match Exn.invalid_arg (fun () -> encode_batch_ids t [ "a" ])

let test_batch_ids_rows =
  prop ~count:200 "gives each text's encode_ids as a row placed by lengths"
    Gen.(triple tokenizer_name texts (options ~longest:false ~stride:false))
    (fun (name, texts, o) ->
      let t = tokenizer name in
      cover "no text" (texts = []);
      cover "an empty text" (List.mem "" texts);
      cover "a post-processor that does more than wrap" (name = "echo words");
      cover "padding" (Option.is_some o.padding);
      equal
        (list (array int))
        (List.map (encode_ids_with o t) texts)
        (rows
           (encode_batch_ids t ~add_special_tokens:o.add_special_tokens
              ~match_special_tokens:o.match_special_tokens ?padding:o.padding
              ?truncation:o.truncation texts)))

let test_batch_ids_reshapes () =
  let buffer, lengths =
    encode_batch_ids (bert_words ())
      ~padding:(padding (`Fixed 5))
      [ "a"; "a b c"; "" ]
  in
  equal ~msg:"lengths" (array int) [| 5; 5; 5 |] lengths;
  let matrix = Bigarray.reshape_2 (Bigarray.genarray_of_array1 buffer) 3 5 in
  equal ~msg:"row 1" (array int) [| 8; 0; 1; 2; 9 |]
    (Array.init 5 (fun j -> Int32.to_int matrix.{1, j}))

(* The pretrained tokenizers name no pad token, so the padding names one. *)
let pad ?direction length =
  padding ?direction ~pad_id:0 ~pad_token:"<pad>" length

let test_batch_ids_parity =
  let options =
    [
      ("plain", true, None, None);
      ("padded to 48", true, Some (pad (`Fixed 48)), None);
      ( "padded to a multiple of 8, on the left",
        true,
        Some (pad ~direction:`Left (`To_multiple 8)),
        None );
      ("truncated to 24", true, None, Some (truncation 24));
      ( "truncated to 24 from the left",
        true,
        None,
        Some (truncation ~direction:`Left 24) );
      ( "truncated and padded to 24",
        true,
        Some (pad (`Fixed 24)),
        Some (truncation 24) );
      ( "without special tokens, padded to 32",
        false,
        Some (pad (`Fixed 32)),
        None );
    ]
  in
  cases "agrees with encode_ids on the parity corpora"
    ~id:(fun (model, (option, _, _, _)) -> model ^ ", " ^ option)
    (List.concat_map
       (fun model -> List.map (fun option -> (model, option)) options)
       [ "gpt2"; "bert_base"; "llama"; "roberta_base"; "llama3"; "gpt_oss" ])
    (fun (model, (_, add_special_tokens, padding, truncation)) ->
      check_batch ~add_special_tokens ?padding ?truncation (pretrained model)
        (parity_documents "sample" @ parity_documents "edge_cases"))

let test_batch_ids_longest () =
  let t = pretrained "gpt2" and documents = parity_documents "sample" in
  let padding = pad `Batch_longest in
  equal
    (list (array int))
    (List.map Encoding.ids (encode_batch t ~padding documents))
    (rows (encode_batch_ids t ~padding documents))

let test_batch_ids_windows () =
  let t = bert_words () and truncation = truncation 4 in
  equal ~msg:"encode_batch's windows"
    (list (list (array int)))
    [ [ [| 8; 2; 3; 9 |] ] ]
    (List.map windows (encode_batch t ~truncation [ "a b c d" ]));
  equal ~msg:"rows"
    (list (array int))
    [ [| 8; 0; 1; 9 |] ]
    (rows (encode_batch_ids t ~truncation [ "a b c d" ]))

(* A document past the cutting threshold is encoded in pieces, on one domain or
   several. Among short ones, a document of 512 KiB is twice the largest chunk,
   whatever the machine. *)
let test_batch_ids_cut =
  cases "keeps the ids of a document cut into pieces" ~id:Fun.id
    [ "gpt2"; "roberta_base"; "llama3"; "gpt_oss" ] (fun model ->
      let t = pretrained model in
      let document =
        repeat (String.concat "\n" (parity_documents "sample")) (512 * kib)
      in
      let documents = [ "a short one"; document; ""; "another short one" ] in
      check_cut ~msg:"on the calling domain" t documents;
      check_cut ~msg:"on two domains" ~exec:(Fixture.spawn 2) t documents;
      check_cut ~msg:"truncated from the left"
        ~truncation:(truncation ~direction:`Left 64)
        t documents)

(* A lone document of 256 KiB is twice the chunk its batch plans. *)
let test_batch_ids_cut_unmatched () =
  let t = pretrained "gpt2" in
  let document =
    repeat (String.concat "\n" (parity_documents "edge_cases")) (256 * kib)
  in
  let as_text = encode_ids t ~match_special_tokens:false document in
  not_equal ~msg:"the document holds special tokens" (array int) as_text
    (encode_ids t document);
  let got, count =
    pieces (fun () ->
        rows (encode_batch_ids t ~match_special_tokens:false [ document ]))
  in
  greater ~msg:"pieces" int ~than:1 count;
  equal ~msg:"rows" (list (array int)) [ as_text ] got

(* Added tokens are matched by one scan of the whole document, and a piece
   starting where the whole did not is not the whole: a [single_word] token
   turned down in the whole for the word before it stands alone at the head of a
   piece, and the scan of the whole skips the bytes of a token it turned down,
   which a piece's scan reads. *)
let test_batch_ids_cut_added () =
  let t =
    word_level
      ~vocab:
        (words [ "hello"; "world"; "the"; "cat"; "sat"; "on"; "mat"; "[UNK]" ])
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
    (encode_ids t "hello world on the");
  equal ~msg:"a piece would not" (array int) [| 8 |] (encode_ids t " world");
  equal ~msg:"nor read what the whole skipped" (array int) [| 10; 7 |]
    (encode_ids t " the");
  let documents =
    [
      repeat "hello world " (512 * kib);
      repeat "hello world the cat sat on the mat " (768 * kib);
      "the cat";
      repeat "cat sat on the mat hello world " (512 * kib);
    ]
  in
  check_cut ~msg:"on the calling domain" t documents;
  check_cut ~msg:"on two domains" ~exec:(Fixture.spawn 2) t documents

(* A vocabulary of space runs: the guards on the bytes either side of a cut keep
   each run whole. *)
let test_batch_ids_cut_spaces () =
  let t =
    bpe
      ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
      ~vocab:
        (words
           [
             g;
             g ^ g;
             g ^ g ^ g;
             g ^ "the";
             g ^ "cat";
             "the";
             "x";
             "t";
             "h";
             "e";
             "c";
             "a";
             "th";
             "ca";
             "cat";
           ])
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
    (encode_ids t "the   cat  the ");
  let documents =
    List.init 5 (fun j ->
        String.make j 'x' ^ repeat "the   cat  the " (512 * kib))
  in
  check_cut ~msg:"on the calling domain" t documents;
  check_cut ~msg:"on two domains" ~exec:(Fixture.spawn 2) t documents

(* A normalizer may act across a space. A lone document of 256 KiB is twice the
   chunk its batch plans. *)
let test_batch_ids_whole_under_normalizer () =
  let t =
    word_level
      ~normalizer:(Normalizer.replace ~pattern:"e c" ~replacement:"ec")
      ~pre:Pre_tokenizer.whitespace
      ~vocab:(words [ "the"; "cat"; "thecat"; "[UNK]" ])
      ~unk_token:"[UNK]" ()
  in
  equal ~msg:"the normalizer joins the words" (array int) [| 2; 2 |]
    (encode_ids t "the cat the cat");
  let documents = [ repeat "the cat the cat " (256 * kib) ] in
  check_whole ~msg:"on the calling domain" t documents;
  check_whole ~msg:"on two domains" ~exec:(Fixture.spawn 2) t documents

(* With no pre-tokenizer the merge [a ␣] crosses every space, so a piece opening
   on [ b] would lose it. *)
let test_batch_ids_whole_unsplit () =
  let t =
    bpe ~vocab:(words [ "a"; " "; "b"; "a " ]) ~merges:[ ("a", " ") ] ()
  in
  equal ~msg:"the merge crosses the space" (array int) [| 3; 2; 1; 3; 2 |]
    (encode_ids t "a b a b");
  check_whole ~msg:"one document" t [ repeat "a b " (256 * kib) ]

(* A batch encoded from inside another's executor, while the other holds the
   calling domain's batch buffer, encodes into buffers of its own. *)
let test_batch_ids_nested () =
  let t = pretrained "gpt2" in
  let inner = parity_documents "edge_cases" in
  let outer = List.concat (List.init 24 (fun _ -> parity_documents "sample")) in
  let inner_rows = ref [] and tasks = ref 0 in
  let exec n task =
    for i = 0 to n - 1 do
      task i;
      incr tasks;
      inner_rows := rows (encode_batch_ids t inner) :: !inner_rows
    done
  in
  let outer_rows = rows (encode_batch_ids t ~exec outer) in
  at_least ~msg:"tasks the executor ran" int ~than:1 !tasks;
  equal ~msg:"the outer batch"
    (list (array int))
    (List.map (encode_ids t) outer)
    outer_rows;
  let expected = List.map (encode_ids t) inner in
  List.iter
    (equal ~msg:"an inner batch" (list (array int)) expected)
    !inner_rows

(* An executor that returns while a task it started is still encoding: the batch
   fails, and the buffer that task writes into stays out of the batches after.
   The straggler is held before its first document; the next batch, on the
   calling domain, lets it write once it has read where its own first document
   starts, and goes on only when the straggler is done. *)
let test_batch_ids_straggler () =
  let t = pretrained "gpt2" in
  let caller = Domain.self () in
  let inside = Atomic.make false and go = Atomic.make false in
  let finished = Atomic.make false and armed = Atomic.make false in
  let hook () =
    if Domain.self () <> caller then
      begin if not (Atomic.get go) then begin
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
  let big =
    List.concat (List.init 2 (fun _ -> parity_documents "sample"))
    |> List.map (fun d -> String.concat " " (List.init 20 (fun _ -> d)))
  in
  let small = parity_documents "edge_cases" in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set batch_document_hook (fun () -> ());
      Atomic.set go true;
      Option.iter Domain.join !straggler)
    (fun () ->
      Atomic.set batch_document_hook hook;
      raises_match ~msg:"the batch whose executor returned early"
        Exn.invalid_arg (fun () -> encode_batch_ids t ~exec big);
      Atomic.set armed true;
      let later = rows (encode_batch_ids t small) in
      equal ~msg:"the batch after"
        (list (array int))
        (List.map (encode_ids t) small)
        later)

(* The tokenizer is finalised once dropped, after batches through an executor
   have filled its buffers. *)
let test_batch_ids_collected () =
  let freed = ref 0 in
  let encode_and_drop () =
    let t = spread_tokenizer () in
    Gc.finalise_last (fun () -> incr freed) t;
    let text = repeat "hello world " (256 * kib) in
    ignore (encode_batch_ids t ~exec:(Fixture.spawn 4) [ text; text ] : ids * _)
  in
  (Sys.opaque_identity encode_and_drop) ();
  Gc.full_major ();
  equal ~msg:"tokenizers finalised" int 1 !freed

(* decode *)

(* HuggingFace decodes [0; 1; 0] of this BPE model to ["a b a"]. *)
let test_decode_joins () =
  equal ~msg:"bpe" string "a b a"
    (decode (bpe ~vocab:[ ("a", 0); ("b", 1) ] ~merges:[] ()) [| 0; 1; 0 |]);
  equal ~msg:"word_level" string "a b a" (decode (plain_words ()) [| 0; 1; 0 |])

let test_decode_outside =
  cases "gives no text for an id outside the vocabulary" ~id:string_of_int
    [ -1; 8; 10; max_int; min_int ] (fun id ->
      equal string "a a" (decode (plain_words ()) [| 0; id; 0 |]))

let role_tokenizer () =
  word_level ~pre:Pre_tokenizer.whitespace
    ~added_tokens:
      [
        added_token "<bos>";
        added_token "<eos>";
        added_token ~special:false "<sep>";
      ]
    ~vocab:[ ("<bos>", 0); ("<eos>", 1); ("hello", 2); ("[CLS]", 3) ]
    ()

let test_decode_keeps_specials () =
  equal string "<bos> hello <eos>" (decode (role_tokenizer ()) [| 0; 2; 1 |])

(* A [normalized] special token is dropped by its id. HuggingFace looks the
   decoded, normalized text up among raw contents, so it drops nothing here and
   decodes ["<s> a b"] either way. *)
let test_decode_skips_specials () =
  let t = role_tokenizer () in
  let sep = Option.get (token_to_id t "<sep>") in
  equal ~msg:"special added tokens" string "hello"
    (decode t ~skip_special_tokens:true [| 0; 2; 1 |]);
  equal ~msg:"an added token that is not special" string "hello <sep>"
    (decode t ~skip_special_tokens:true [| 2; sep |]);
  equal ~msg:"a vocabulary entry that is not added" string "[CLS] hello"
    (decode t ~skip_special_tokens:true [| 3; 2 |]);
  let marker = "\u{2581}" in
  let normalized =
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
  equal ~msg:"a normalized special token, kept" string "<s> a b"
    (decode normalized [| 1; 2; 3 |]);
  equal ~msg:"a normalized special token, skipped" string "a b"
    (decode normalized ~skip_special_tokens:true [| 1; 2; 3 |]);
  equal ~msg:"nothing but the skipped token" string ""
    (decode normalized ~skip_special_tokens:true [| 1 |])

(* Streaming decoding *)

(* [s] fed [ids]: the text of each step, and the stream after them. *)
let feed s ids =
  let texts, s =
    List.fold_left
      (fun (texts, s) id ->
        let text, s = decode_step s id in
        (text :: texts, s))
      ([], s) ids
  in
  (List.rev texts, s)

(* The texts a stream gives for [ids], one per step and then the finish. *)
let stream_texts ?skip_special_tokens t ids =
  let texts, s = feed (decode_stream ?skip_special_tokens t) ids in
  texts @ [ decode_finish s ]

(* A tokenizer whose ids are the positions of [tokens], decoded by [decoder]. *)
let listed ?decoder tokens = word_level ?decoder ~vocab:(words tokens) ()

(* Byte-level characters: [â], [Ĥ] and [¬] are the bytes of [€]. *)
let euro = [ "\xc3\xa2"; "\xc4\xa4"; "\xc2\xac" ]
let fallback_tokens = [ "<0x20>"; "<0xC3>"; "<0xA9>"; " This"; "<0xFF>" ]
let long_run = 10_000

(* One row per case a stream keeps text back for, and the cases beside it that
   it does not. *)
let test_stream_cases =
  let byte_level = lazy (listed ~decoder:Decoder.byte_level ("a" :: euro)) in
  let fallback = lazy (listed ~decoder:Decoder.byte_fallback fallback_tokens) in
  let after decoders tokens =
    lazy (listed ~decoder:(Decoder.sequence decoders) tokens)
  in
  cases "keeps text back only while a later token can change it"
    ~id:(fun (name, _, _, _) -> name)
    [
      ( "a character waits for its last byte",
        byte_level,
        [ 0; 1; 2; 3; 0 ],
        [ "a"; ""; ""; "\xe2\x82\xac"; "a"; "" ] );
      ( "a character cut short ends the text",
        byte_level,
        [ 0; 1; 2 ],
        [ "a"; ""; ""; "\xef\xbf\xbd" ] );
      ( "a character's start that a later byte breaks",
        byte_level,
        [ 1; 0 ],
        [ ""; "\xef\xbf\xbda"; "" ] );
      (* A byte that breaks the UTF-8 of a run of byte tokens turns each byte of
         the run into a replacement character. *)
      ( "a run of byte tokens waits for its end",
        fallback,
        [ 0; 1; 2; 3 ],
        [ ""; ""; ""; " \xc3\xa9 This"; "" ] );
      ( "a run that is not UTF-8",
        fallback,
        [ 0; 4; 3 ],
        [ ""; ""; "\xef\xbf\xbd\xef\xbf\xbd This"; "" ] );
      ( "a run at the end",
        fallback,
        [ 3; 1; 2 ],
        [ " This"; ""; ""; "\xc3\xa9" ] );
      ( "a long run of byte tokens",
        lazy (listed ~decoder:Decoder.byte_fallback [ "<0x41>"; "b" ]),
        List.init long_run (fun _ -> 0) @ [ 1 ],
        List.init long_run (fun _ -> "")
        @ [ String.make long_run 'A' ^ "b"; "" ] );
      (* The marker that stands for nothing on the first token stands for a
         space on the others. *)
      ( "the first token keeps its own text",
        lazy (listed ~decoder:(Decoder.metaspace ()) [ m ^ "This" ]),
        [ 0; 0 ],
        [ "This"; " This"; "" ] );
      (* A suffix is a space unless its token is the last. *)
      ( "the last token waits for the next",
        lazy (listed ~decoder:(Decoder.bpe ()) [ "a</w>"; "b</w>" ]),
        [ 0; 1 ],
        [ ""; "a "; "b" ] );
      ( "a token without the suffix is given at once",
        lazy (listed ~decoder:(Decoder.bpe ()) [ "a"; "b</w>" ]),
        [ 0; 1; 0; 1 ],
        [ "a"; ""; "b a"; ""; "b" ] );
      (* [ÿ] and [Ģ] are the bytes [\xff] and [\x80], which no later byte makes
         part of a character. *)
      ( "a byte no character starts with is replaced at once",
        lazy
          (listed ~decoder:Decoder.byte_level [ "a"; "\xc3\xbf"; "\xc4\xa2" ]),
        [ 0; 1; 2; 0 ],
        [ "a"; "\xef\xbf\xbd"; "\xef\xbf\xbd"; "a"; "" ] );
      (* After a collapse, a stage keeps what a later piece may change. *)
      ( "the start of a match",
        after
          [ Decoder.fuse; Decoder.replace ~pattern:"ab" ~by:"x" () ]
          [ "a"; "b"; "c" ],
        [ 0; 1; 0; 2 ],
        [ ""; "x"; ""; "ac"; "" ] );
      ( "the occurrences an end may cut",
        after [ Decoder.fuse; Decoder.strip ~stop:2 () ] [ "a"; " "; "b" ],
        [ 0; 1; 1; 1; 2; 1 ],
        [ "a"; ""; ""; " "; "  b"; ""; "" ] );
      ( "the occurrences the start may cut",
        after [ Decoder.fuse; Decoder.strip ~start:2 () ] [ "a"; " " ],
        [ 1; 1; 1; 0 ],
        [ ""; ""; " "; "a"; "" ] );
      ( "a decoder that reads its text whole",
        after [ Decoder.fuse; Decoder.byte_level ] [ "a"; "b" ],
        [ 0; 1 ],
        [ ""; ""; "ab" ] );
      (* With no decoder the tokens are joined with spaces, by every model.
         HuggingFace decodes [0; 1; 0] of this BPE model to ["a b a"]. *)
      ( "no decoder joins a BPE model's tokens with spaces",
        lazy (bpe ~vocab:[ ("a", 0); ("b", 1) ] ~merges:[] ()),
        [ 0; 1; 0 ],
        [ "a"; " b"; " a"; "" ] );
      ( "no decoder joins a word-level model's tokens with spaces",
        lazy (listed [ "a"; "b" ]),
        [ 0; 1; 0 ],
        [ "a"; " b"; " a"; "" ] );
      ( "an id outside the vocabulary gives nothing",
        lazy (listed [ "a" ]),
        [ 0; 7; 0 ],
        [ "a"; ""; " a"; "" ] );
    ]
    (fun (_, t, ids, expected) ->
      equal (list string) expected (stream_texts (Lazy.force t) ids))

(* Decoders that reach every case a stream keeps text back for: an open run of
   byte tokens, a character not yet complete, a last token, and, after a
   collapse, the start of a match and the ends a strip may cut. *)
let stream_decoders =
  [
    ("bpe", Decoder.bpe ());
    ("bpe a", Decoder.bpe ~suffix:"a" ());
    ("bpe empty", Decoder.bpe ~suffix:"" ());
    ("byte_level", Decoder.byte_level);
    ("byte_fallback", Decoder.byte_fallback);
    ("wordpiece", Decoder.wordpiece ());
    ("wordpiece no cleanup", Decoder.wordpiece ~cleanup:false ());
    ("metaspace", Decoder.metaspace ());
    ("metaspace never", Decoder.metaspace ~prepend_scheme:`Never ());
    ("ctc", Decoder.ctc ());
    ("ctc no cleanup", Decoder.ctc ~cleanup:false ());
    ("ctc empty", Decoder.ctc ~word_delimiter_token:"" ());
    ("replace marker", Decoder.replace ~pattern:m ~by:" " ());
    ("replace ab", Decoder.replace ~pattern:"ab" ~by:"x" ());
    ("replace aa", Decoder.replace ~pattern:"aa" ~by:"a" ());
    ("replace empty", Decoder.replace ~pattern:"" ~by:"_" ());
    ("strip start", Decoder.strip ~start:1 ());
    ("strip stop", Decoder.strip ~stop:2 ());
    ("strip ab", Decoder.strip ~content:"ab" ~start:1 ~stop:1 ());
    ("strip marker", Decoder.strip ~content:m ~stop:1 ());
    ("fuse", Decoder.fuse);
  ]

let collapses = [ "byte_level"; "fuse" ]

(* The decoders that keep nothing back unless a collapse comes before them. *)
let immediate =
  List.filter
    (fun name ->
      not
        (List.mem name
           ([ "bpe"; "bpe a"; "bpe empty"; "byte_fallback" ] @ collapses)))
    (List.map fst stream_decoders)

let stream_tokens =
  [ ""; "a"; "b"; "ab"; " "; "  "; m; m ^ "a"; "a" ^ m; "##a"; "a</w>"; "</w>";
    "<0x41>"; "<0xC3>"; "<0xA9>"; "<0xE2>"; "<0x82>"; "<0xAC>"; "<0xFF>";
    "<pad>"; "|"; "."; " ."; "do"; " not"; "'"; "n't"; "\xc4\xa0"; "\xc3\x83";
    "\xc2\xa9"; "\xc3\xbf"; "\xc4\xa2"; "\u{e9}" ]
  @ euro
[@@ocamlformat "disable"]

let stream_ids =
  Gen.list ~size:(Gen.int_range 0 12)
    (Gen.int_range 0 (List.length stream_tokens - 1))

let pp_ids ppf ids =
  Format.pp_print_string ppf (String.concat "; " (List.map string_of_int ids))

(* A sequence of decoders, most often with a collapse between two lists, so that
   the decoders after it rewrite text arriving piece by piece. *)
let stream_case =
  let open Gen in
  let leaf = of_list (List.map fst stream_decoders) in
  let case =
    let+ before = list ~size:(int_range 0 2) leaf
    and+ collapse = of_list [ []; [ "byte_level" ]; [ "fuse" ]; [ "fuse" ] ]
    and+ after = list ~size:(int_range 0 2) leaf
    and+ ids = stream_ids in
    (before @ collapse @ after, ids)
  in
  with_pp
    (fun ppf (names, ids) ->
      Format.fprintf ppf "[%s], [%a]" (String.concat "; " names) pp_ids ids)
    case

let stream_tokenizer names =
  listed
    ~decoder:
      (Decoder.sequence
         (List.map (fun name -> List.assoc name stream_decoders) names))
    stream_tokens

(* Whether after some step the texts given so far differ from [decode] of the
   ids fed so far, which is the stream keeping text back. *)
let kept t ids texts =
  let ids = Array.of_list ids and texts = Array.of_list texts in
  let rec from i given =
    i < Array.length ids
    &&
    let given = given ^ texts.(i) in
    given <> decode t (Array.sub ids 0 (i + 1)) || from (i + 1) given
  in
  from 0 ""

(* A stream fed [prefix], then each continuation from the stream [prefix]
   left. *)
let test_stream_value =
  prop "leaves the stream a step was given as it was"
    Gen.(triple stream_case stream_ids stream_ids)
    (fun ((names, prefix), first, second) ->
      let t = stream_tokenizer names in
      let prefix_texts, fed = feed (decode_stream t) prefix in
      cover ~share:0.05 "text kept at the fork"
        (String.concat "" prefix_texts <> decode t (Array.of_list prefix));
      List.iter
        (fun continuation ->
          let texts, s = feed fed continuation in
          equal string
            (decode t (Array.of_list (prefix @ continuation)))
            (String.concat "" (prefix_texts @ texts @ [ decode_finish s ])))
        [ first; second ])

(* decode_stream *)

let test_stream_fresh () =
  equal string "" (decode_finish (decode_stream (byte_bpe ())))

let test_stream_skips () =
  let t =
    word_level
      ~vocab:[ ("a", 0); ("b", 1) ]
      ~added_tokens:[ added_token "<s>" ]
      ()
  in
  let s = Option.get (token_to_id t "<s>") in
  equal ~msg:"skipped" (list string) [ "a"; ""; " b"; "" ]
    (stream_texts ~skip_special_tokens:true t [ 0; s; 1 ]);
  equal ~msg:"kept" (list string) [ "a"; " <s>"; " b"; "" ]
    (stream_texts t [ 0; s; 1 ])

(* decode_step *)

let test_step_law =
  prop ~count:2000 "gives texts that with the finish make up decode" stream_case
    (fun (names, ids) ->
      let t = stream_tokenizer names in
      let texts = stream_texts t ids in
      let rec after_collapse prefix = function
        | name :: rest when List.mem name collapses ->
            List.exists (String.starts_with ~prefix) rest
        | _ :: rest -> after_collapse prefix rest
        | [] -> false
      in
      cover ~share:0.05 "a strip after a collapse"
        (after_collapse "strip" names);
      cover ~share:0.05 "a replace after a collapse"
        (after_collapse "replace" names);
      cover ~share:0.2 "text kept for a later token" (kept t ids texts);
      cover ~share:0.05 "an empty token"
        (List.exists (fun id -> id_to_token t id = Some "") ids);
      equal string (decode t (Array.of_list ids)) (String.concat "" texts))

let test_step_utf_8 =
  prop ~count:2000 "gives texts of valid UTF-8" stream_case (fun (names, ids) ->
      cover ~share:0.05 "a byte-level decoder" (List.mem "byte_level" names);
      List.iter
        (satisfies ~claim:"valid UTF-8" string String.is_valid_utf_8)
        (stream_texts (stream_tokenizer names) ids))

let test_step_immediate =
  prop
    "gives decode of the ids fed so far under decoders that keep nothing back"
    Gen.(
      pair
        (list ~size:(int_range 1 3)
           (of_list ~pp:Format.pp_print_string immediate))
        stream_ids)
    (fun (names, ids) ->
      let t = stream_tokenizer names in
      let ids = Array.of_list ids in
      let texts = Array.of_list (stream_texts t (Array.to_list ids)) in
      let given = ref "" in
      for i = 0 to Array.length ids - 1 do
        given := !given ^ texts.(i);
        equal
          ~msg:(Printf.sprintf "after %d ids" (i + 1))
          string
          (decode t (Array.sub ids 0 (i + 1)))
          !given
      done)

(* decode_finish *)

let test_finish () =
  let _, s =
    feed
      (decode_stream (listed ~decoder:(Decoder.bpe ()) [ "a</w>"; "b</w>" ]))
      [ 0; 1 ]
  in
  equal string "b" (decode_finish s)

let () =
  exit
    (run "Encoding and decoding"
       [
         group "pad_length"
           [
             test
               "`Fixed n pads a shorter sequence to n tokens of the pad token"
               test_fixed_pads;
             test_fixed_keeps_longer;
             test_to_multiple;
             test_to_multiple_nonpositive;
             test "`Batch_longest pads every text to the batch's longest"
               test_batch_longest;
             test "`Batch_longest adds nothing to encode's one text"
               test_batch_longest_of_one;
           ];
         group "padding"
           [
             test "defaults to the right and to the tokenizer's pad token"
               test_padding_defaults;
             test "pads at the start under `Left" test_padding_left;
             test_padding_tokenizer_pad;
             test "gives padding the pad_id, pad_type_id and pad_token it names"
               test_padding_named;
             test_padding_without_token;
           ];
         group "truncation"
           [
             test "defaults to the right with no stride"
               test_truncation_defaults;
             test_truncation_refuses;
             test_truncation_keeps;
             test "splits a pair's budget as HuggingFace does"
               test_truncation_pair;
             test "joins a pair's windows as HuggingFace does"
               test_truncation_pair_windows;
             test "wraps each window in the special tokens"
               test_truncation_wraps_windows;
             test
               "opens each window stride tokens before the end of the one \
                before"
               test_truncation_stride;
             test
               "keeps only the pretokens that hold the max_length tokens kept"
               test_truncation_pretokens;
             test_truncation_hf_windows;
             test_truncation_stride_room;
           ];
         group "encode"
           [
             test "matches added tokens whatever add_special_tokens is"
               test_encode_added;
             test
               "encodes a special token's text as text under \
                match_special_tokens false"
               test_encode_unmatched;
             test "matches special tokens by default"
               test_encode_matched_by_default;
             test_encode_stray_bytes;
             test "reports offsets in the text before normalization"
               test_encode_normalized_offsets;
             test "places a byte-level token at the bytes it encodes"
               test_encode_byte_level_offsets;
             test "numbers the pretokens from 0, an added token counting as one"
               test_encode_word_ids;
             test "numbers a text without a pre-tokenizer as one pretoken"
               test_encode_one_pretoken;
             test "keeps the number of a pretoken the model gives no token"
               test_encode_word_gap;
             test_encode_rewriting;
             test_encode_byte_level_models;
             test_encode_unknown_runs;
             test_encode_byte_level_after_split;
             test "gives an unknown token the bytes between its neighbours"
               test_encode_unknown_token;
             test_encode_spans;
             test_encode_no_specials;
             test
               "gives the tokens the post-processor inserts no word and the \
                span (0, 0)"
               test_encode_inserted;
             test "merges a pair through the post-processor with its type ids"
               test_encode_pair;
           ];
         group "exec"
           [
             test "is called at most once by a batch" test_exec_once;
             test "is not called for a batch under 64 KB" test_exec_small;
             test "is asked for at most a task per recommended domain"
               test_exec_task_count;
             test "runs tasks that never raise" test_exec_tasks_never_raise;
             test_exec_same_ids;
             test "fails the batch with what a chunk raised"
               test_exec_failed_chunk;
             test "fails the batch with what it raised" test_exec_raised;
             test
               "fails the batch with Invalid_argument when it returns before \
                every chunk is encoded"
               test_exec_returned_early;
             test "leaves the tokenizer encoding after a failed batch"
               test_exec_after_failure;
             test "stops the handout at the first failing chunk"
               test_exec_stops_handout;
             test "stops the other tasks at the first failing chunk"
               test_exec_stops_tasks;
             test_exec_late_tasks;
           ];
         group "encode_batch" [ test_batch_is_encode ];
         group "encode_batch_pairs" [ test_batch_pairs_is_encode ];
         group "encode_ids"
           [
             test_ids_is_encode;
             test "raises where encode raises on its post-processor"
               test_ids_raise;
           ];
         group "encode_batch_ids"
           [
             test_batch_ids_rows;
             test "raises where encode raises on its post-processor"
               test_batch_ids_raise;
             test "reshapes to rows of one length under `Fixed padding"
               test_batch_ids_reshapes;
             test_batch_ids_parity;
             test "resolves `Batch_longest over the batch as encode_batch does"
               test_batch_ids_longest;
             test "drops the windows encode_batch keeps" test_batch_ids_windows;
             test_batch_ids_cut;
             test "keeps special tokens as text across a cut"
               test_batch_ids_cut_unmatched;
             test "cuts a document only where no added token lies over the cut"
               test_batch_ids_cut_added;
             test "keeps a run of spaces whole across a cut"
               test_batch_ids_cut_spaces;
             test "cuts no document of a pipeline with a normalizer"
               test_batch_ids_whole_under_normalizer;
             test "cuts no document of a pipeline that does not split on spaces"
               test_batch_ids_whole_unsplit;
             test
               "keeps apart the ids of a batch encoded inside another batch's \
                executor"
               test_batch_ids_nested;
             test
               "keeps a later batch's ids from a task its failed batch left \
                running"
               test_batch_ids_straggler;
             test
               "lets a tokenizer whose batches filled its buffers be collected"
               test_batch_ids_collected;
           ];
         group "decode"
           [
             test "joins the tokens with a space between two without a decoder"
               test_decode_joins;
             test_decode_outside;
             test "keeps special tokens by default" test_decode_keeps_specials;
             test
               "drops the special added tokens, and only those, under \
                skip_special_tokens"
               test_decode_skips_specials;
           ];
         group "stream" [ test_stream_cases; test_stream_value ];
         group "decode_stream"
           [
             test "is a stream whose finish gives no text" test_stream_fresh;
             test "skips special tokens as decode does" test_stream_skips;
           ];
         group "decode_step"
           [ test_step_law; test_step_utf_8; test_step_immediate ];
         group "decode_finish"
           [
             test "decodes the text no step gave as the end of the ids"
               test_finish;
           ];
       ])
