(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A tokenizer from [from_file] gives the ids and tokens that HuggingFace's
   tokenizers gives for the same file and text. The expected values are
   tokenizers 0.23.1's: ids, offsets and decoded text written by
   test/gen/gen_parity_expected.py, which documents the corpora and the format,
   and tokens written by test/gen/gen_parity_tokens.py. *)

open Windtrap
open Brot

let corpora = [ "sample"; "edge_cases" ]

(* What the tokenizers that split on a regular expression are also run on. *)
let regex_split_corpora = corpora @ [ "unicode_code" ]

let models =
  [
    ("gpt2", corpora);
    ("llama", corpora);
    ("bert_base", corpora);
    ("roberta_base", corpora);
    ("t5_base", corpora);
    ("mistral", corpora);
    ("llama3", regex_split_corpora);
    ("qwen2_5", regex_split_corpora);
    ("deepseek_v3", regex_split_corpora);
    ("gpt_oss", regex_split_corpora);
    ("phi4", regex_split_corpora);
    ("mistral_nemo", regex_split_corpora);
  ]

(* The models whose tokens gen_parity_tokens.py records. T5 is not among them:
   tokenizers spells a Unigram unknown token as the text it stands for, where
   brot gives the vocabulary's entry. *)
let token_models =
  [
    "gpt2";
    "bert_base";
    "roberta_base";
    "mistral";
    "qwen2_5";
    "deepseek_v3";
    "gpt_oss";
    "phi4";
    "mistral_nemo";
  ]

(* Fixtures *)

let path corpus suffix = Printf.sprintf "support/parity/%s%s" corpus suffix

(* The last byte of a fixture is a newline that belongs to the file, not to the
   last document or the last row. *)
let lines text =
  let n = String.length text in
  let text =
    if n > 0 && text.[n - 1] = '\n' then String.sub text 0 (n - 1) else text
  in
  String.split_on_char '\n' text

(* A corpus as one text, and as its documents, which lines of [====]
   separate. *)
let documents corpus =
  let text = lines (Fixture.read (path corpus ".txt")) in
  let join current = String.concat "\n" (List.rev current) in
  let rec split docs current = function
    | [] -> List.rev (join current :: docs)
    | "====" :: rest -> split (join current :: docs) [] rest
    | line :: rest -> split docs (line :: current) rest
  in
  (String.concat "\n" text, Array.of_list (split [] [] text))

let json file codec text =
  match Jsont_bytesrw.decode_string codec text with
  | Ok value -> value
  | Error msg -> failf "%s: %s" file msg

(* The rows of a [.ids] file: per document the [ids] and [offsets] of
   [add_special_tokens:false], [decode] of those ids, the [special] ids and
   their [mask] with [add_special_tokens:true], and the [unmatched] ids with
   [match_special_tokens:false]; then the [whole] file's ids. *)
type expected = {
  ids : int array array;
  offsets : (int * int) array array;
  decode : string array;
  special : int array array;
  mask : int array array;
  unmatched : int array array;
  whole : int array;
}

let read_expected corpus model =
  let file = path corpus ("." ^ model ^ ".ids") in
  let rows =
    List.filter_map
      (fun line ->
        if String.starts_with ~prefix:"#" line then None
        else
          match String.index_opt line ' ' with
          | None -> Some (line, "")
          | Some i ->
              Some
                ( String.sub line 0 i,
                  String.sub line (i + 1) (String.length line - i - 1) ))
      (lines (Fixture.read file))
  in
  let kind name parse =
    Array.of_list
      (List.filter_map
         (fun (k, rest) -> if k = name then Some (parse rest) else None)
         rows)
  in
  let fields rest = List.filter (( <> ) "") (String.split_on_char ' ' rest) in
  let numbers rest = Array.of_list (List.map int_of_string (fields rest)) in
  let spans rest =
    Array.of_list
      (List.map
         (fun f -> Scanf.sscanf f "%d,%d%!" (fun s e -> (s, e)))
         (fields rest))
  in
  {
    ids = kind "IDS" numbers;
    offsets = kind "OFFSETS" spans;
    decode = kind "DECODE" (json file Jsont.string);
    special = kind "SPECIAL" numbers;
    mask = kind "MASK" numbers;
    unmatched = kind "UNMATCHED" numbers;
    whole =
      (match kind "ALL" numbers with
      | [| whole |] -> whole
      | rows -> failf "%s: %d ALL rows" file (Array.length rows));
  }

(* Tokenizers, each loaded once per worker. A download that is missing skips the
   tests of its model. *)
let tokenizers =
  List.map
    (fun (model, _) ->
      ( model,
        fixture (fun () ->
            Fixture.with_tokenizer_file model (fun file ->
                match from_file file with
                | Ok t -> t
                | Error msg -> failf "failed to load %s: %s" file msg)) ))
    models

let tokenizer model = (List.assoc model tokenizers) ()

(* Each (model, corpus) pair's expected rows, and its documents encoded without
   and with special tokens, made once per worker and read by every test of the
   pair that needs them. *)
type encoded = { plain : Encoding.t array; special : Encoding.t array }

let pairs =
  List.concat_map
    (fun (model, corpora) -> List.map (fun corpus -> (model, corpus)) corpora)
    models

let expected_rows =
  List.map
    (fun (model, corpus) ->
      ((model, corpus), fixture (fun () -> read_expected corpus model)))
    pairs

let expected corpus model = (List.assoc (model, corpus) expected_rows) ()

let encodings =
  List.map
    (fun (model, corpus) ->
      ( (model, corpus),
        fixture (fun () ->
            let t = tokenizer model and _, docs = documents corpus in
            {
              plain = Array.map (encode t ~add_special_tokens:false) docs;
              special = Array.map (encode t) docs;
            }) ))
    pairs

let encoded model corpus = (List.assoc (model, corpus) encodings) ()

(* Checks, each of a model on a corpus *)

(* Each document's row against [actual i text] of document [i], the failure
   naming the document and the entry point. *)
let each_document w ~via corpus rows actual =
  let _, docs = documents corpus in
  equal
    ~msg:(corpus ^ ": one row per document")
    int (Array.length docs) (Array.length rows);
  Array.iteri
    (fun i text ->
      equal
        ~msg:(Printf.sprintf "%s.txt, document %d, %s" corpus i via)
        w rows.(i) (actual i text))
    docs

let check_ids model corpus =
  let t = tokenizer model and rows = expected corpus model in
  let encoded = encoded model corpus in
  each_document (array int) ~via:"encode" corpus rows.ids (fun i _ ->
      Encoding.ids encoded.plain.(i));
  each_document (array int) ~via:"encode_ids" corpus rows.ids (fun _ text ->
      encode_ids t ~add_special_tokens:false text);
  equal ~msg:"the whole file" (array int) rows.whole
    (encode_ids t ~add_special_tokens:false (fst (documents corpus)))

(* Byte spans; the generator converts tokenizers' character spans. *)
let check_offsets model corpus =
  let encoded = encoded model corpus in
  each_document
    (array (pair int int))
    ~via:"encode" corpus (expected corpus model).offsets
    (fun i _ -> Encoding.offsets encoded.plain.(i))

let check_special model corpus =
  let encoded = encoded model corpus in
  each_document (array int) ~via:"encode" corpus (expected corpus model).special
    (fun i _ -> Encoding.ids encoded.special.(i))

let check_mask model corpus =
  let encoded = encoded model corpus in
  each_document (array int) ~via:"encode" corpus (expected corpus model).mask
    (fun i _ -> Encoding.special_tokens_mask encoded.special.(i))

(* One text is one sequence: type id 0 and attention 1 throughout, over the
   whole length of the ids. *)
let check_constant select value model corpus =
  let encoded = encoded model corpus in
  let rows =
    Array.map
      (fun e -> Array.map (fun _ -> value) (Encoding.ids e))
      encoded.special
  in
  each_document (array int) ~via:"encode" corpus rows (fun i _ ->
      select encoded.special.(i))

let check_tokens model corpus =
  let encoded = encoded model corpus in
  let file = path corpus ("." ^ model ^ ".tokens") in
  let rows =
    List.filter_map
      (fun line ->
        if String.starts_with ~prefix:"#" line then None
        else Some (Array.of_list (json file Jsont.(list string) line)))
      (lines (Fixture.read file))
  in
  each_document (array string) ~via:"encode" corpus (Array.of_list rows)
    (fun i _ -> Encoding.tokens encoded.special.(i))

let check_unmatched model corpus =
  let t = tokenizer model and rows = (expected corpus model).unmatched in
  each_document (array int) ~via:"encode" corpus rows (fun _ text ->
      Encoding.ids
        (encode t ~add_special_tokens:false ~match_special_tokens:false text));
  each_document (array int) ~via:"encode_ids" corpus rows (fun _ text ->
      encode_ids t ~add_special_tokens:false ~match_special_tokens:false text)

(* The rows of a flat batch buffer, read back through their lengths. *)
let rows (buffer, lengths) =
  let at = ref 0 in
  Array.map
    (fun length ->
      let row =
        Array.init length (fun k ->
            Int32.to_int (Bigarray.Array1.get buffer (!at + k)))
      in
      at := !at + length;
      row)
    lengths

let check_batches model corpus =
  let t = tokenizer model in
  let docs = Array.to_list (snd (documents corpus)) in
  let unmatched = (expected corpus model).unmatched in
  equal ~msg:"encode_batch"
    (array (array int))
    unmatched
    (Array.of_list
       (List.map Encoding.ids
          (encode_batch t ~add_special_tokens:false ~match_special_tokens:false
             docs)));
  equal ~msg:"encode_batch_ids"
    (array (array int))
    unmatched
    (rows
       (encode_batch_ids t ~add_special_tokens:false ~match_special_tokens:false
          docs))

(* Each document paired with the one as far from the end as it is from the
   start. *)
let check_pairs model corpus =
  let t = tokenizer model in
  if Option.is_none (post_processor t) then
    skip ~reason:"a pair needs a post-processor" ();
  let docs = Array.to_list (snd (documents corpus)) in
  let pairs = List.combine docs (List.rev docs) in
  equal
    (list (array int))
    (List.map
       (fun (text, pair) ->
         Encoding.ids (encode t ~pair ~match_special_tokens:false text))
       pairs)
    (List.map Encoding.ids
       (encode_batch_pairs t ~match_special_tokens:false pairs))

(* The decoder is driven by the reference ids rather than by brot's own
   encoding, so that a decoder divergence shows where the encoder is wrong
   too. *)
let check_decode model corpus =
  let t = tokenizer model and rows = expected corpus model in
  let _, docs = documents corpus in
  equal
    ~msg:(corpus ^ ": one row per document")
    int (Array.length docs) (Array.length rows.decode);
  Array.iteri
    (fun i expected ->
      equal
        ~msg:(Printf.sprintf "%s.txt, document %d, decode" corpus i)
        string expected
        (decode t rows.ids.(i)))
    rows.decode

(* Streaming decoding. A case decodes a stretch of a reference encoding, which
   holds the characters a model spells over several tokens, followed by ids
   drawn from the vocabulary's own, added tokens included. *)

(* The reference ids of a model's corpora, and the ids of its vocabulary, which
   may have gaps. *)
let references =
  List.map
    (fun (model, corpora) ->
      ( model,
        fixture (fun () ->
            let rows =
              Array.concat
                (List.map (fun corpus -> (expected corpus model).ids) corpora)
            in
            let known =
              Array.of_list (List.map snd (vocab (tokenizer model)))
            in
            Array.sort Int.compare known;
            (rows, known)) ))
    models

let stream_case =
  let open Gen in
  let case =
    let+ document = nat
    and+ start = nat
    and+ length = int_range 0 48
    and+ drawn = list ~size:(int_range 0 8) nat
    and+ skip_special_tokens = bool in
    (document, start, length, drawn, skip_special_tokens)
  in
  with_pp
    (fun ppf (document, start, length, drawn, skip) ->
      Format.fprintf ppf "document %d from %d for %d, then [%s], skip %b"
        document start length
        (String.concat "; " (List.map string_of_int drawn))
        skip)
    case

let stream_ids model (document, start, length, drawn, _) =
  let rows, known = (List.assoc model references) () in
  let row = rows.(document mod Array.length rows) in
  let start = start mod (Array.length row + 1) in
  let length = min length (Array.length row - start) in
  let drawn = List.map (fun id -> known.(id mod Array.length known)) drawn in
  Array.append (Array.sub row start length) (Array.of_list drawn)

(* The texts a stream gives for [ids], one per step and then the finish. *)
let stream_texts ~skip_special_tokens t ids =
  let texts, s =
    Array.fold_left
      (fun (texts, s) id ->
        let text, s = decode_step s id in
        (text :: texts, s))
      ([], decode_stream ~skip_special_tokens t)
      ids
  in
  List.rev (decode_finish s :: texts)

(* The two decoders that never wait for a later token. *)
let immediate = [ "bert_base"; "t5_base" ]

(* Whether after some step the texts given so far differ from [decode] of the
   ids fed so far, which is the stream keeping text back. *)
let kept ~skip_special_tokens t ids texts =
  let texts = Array.of_list texts in
  let rec from i given =
    i < Array.length ids
    &&
    let given = given ^ texts.(i) in
    given <> decode t ~skip_special_tokens (Array.sub ids 0 (i + 1))
    || from (i + 1) given
  in
  from 0 ""

let test_stream_law model =
  prop "decode_step gives texts that with the finish make up decode" stream_case
    (fun ((_, _, _, _, skip_special_tokens) as case) ->
      let t = tokenizer model in
      let ids = stream_ids model case in
      let texts = stream_texts ~skip_special_tokens t ids in
      if not (List.mem model immediate) then
        cover ~share:0.02 "text kept for a later token"
          (kept ~skip_special_tokens t ids texts);
      equal string (decode t ~skip_special_tokens ids) (String.concat "" texts))

let test_stream_utf_8 model =
  prop "decode_step gives texts of valid UTF-8" stream_case
    (fun ((_, _, _, _, skip_special_tokens) as case) ->
      List.iter
        (satisfies ~claim:"valid UTF-8" string String.is_valid_utf_8)
        (stream_texts ~skip_special_tokens (tokenizer model)
           (stream_ids model case)))

let () =
  exit
    (run "from_file"
       (List.map
          (fun (model, corpora) ->
            let corpus name check =
              cases name ~id:Fun.id corpora (fun corpus -> check model corpus)
            in
            let tokens =
              if List.mem model token_models then
                [
                  corpus "encode with special tokens gives HuggingFace's tokens"
                    check_tokens;
                ]
              else []
            in
            group model
              ([
                 corpus "encode gives HuggingFace's ids" check_ids;
                 corpus "encode gives HuggingFace's offsets" check_offsets;
                 corpus "encode with special tokens gives HuggingFace's ids"
                   check_special;
                 corpus
                   "encode with special tokens gives HuggingFace's special \
                    tokens mask"
                   check_mask;
                 corpus "encode gives type id 0 to every token of one text"
                   (check_constant Encoding.type_ids 0);
                 corpus "encode gives attention 1 to every token of one text"
                   (check_constant Encoding.attention_mask 1);
               ]
              @ tokens
              @ [
                  corpus
                    "encode with match_special_tokens false gives \
                     HuggingFace's ids"
                    check_unmatched;
                  corpus
                    "encode_batch and encode_batch_ids give HuggingFace's ids \
                     with match_special_tokens false"
                    check_batches;
                  corpus "encode_batch_pairs gives encode ~pair's ids"
                    check_pairs;
                  corpus "decode gives HuggingFace's text" check_decode;
                  test_stream_law model;
                  test_stream_utf_8 model;
                ]))
          models))
