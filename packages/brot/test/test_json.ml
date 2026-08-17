(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* HuggingFace JSON serialization. The expected texts are what the [tokenizers]
   Python package writes for the same object, and what it accepts back. *)

open Windtrap
open Brot

let json_text json =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json json with
  | Ok text -> text
  | Error msg -> failf "cannot encode: %s" msg

let json_of_text text =
  match Jsont_bytesrw.decode_string Jsont.json text with
  | Ok json -> json
  | Error msg -> failf "cannot decode %S: %s" text msg

let json_member name = function
  | Jsont.Object (members, _) -> (
      match Jsont.Json.find_mem name members with
      | Some (_, value) -> value
      | None -> failf "no %S member" name)
  | _ -> failf "looking for %S in something that is not an object" name

let json_without names = function
  | Jsont.Object (members, meta) ->
      Jsont.Object
        ( List.filter (fun ((name, _), _) -> not (List.mem name names)) members,
          meta )
  | json -> json

let marker = "\xe2\x96\x81"

(* Every decoder, with JSON HuggingFace reads back as the same decoder. The
   members that only a pre-tokenizer reads are written at their defaults when
   HuggingFace requires them ([ByteLevel]) and left out when it does not
   ([Metaspace]'s [split], [ByteLevel]'s [use_regex]). *)
let decoders =
  [
    (Decoder.bpe (), {|{"type":"BPEDecoder","suffix":"</w>"}|});
    ( Decoder.byte_level (),
      {|{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true}|} );
    (Decoder.byte_fallback (), {|{"type":"ByteFallback"}|});
    (Decoder.wordpiece (), {|{"type":"WordPiece","prefix":"##","cleanup":true}|});
    ( Decoder.metaspace (),
      {|{"type":"Metaspace","replacement":"|} ^ marker
      ^ {|","prepend_scheme":"always"}|} );
    ( Decoder.metaspace ~replacement:"_" ~prepend_scheme:`Never (),
      {|{"type":"Metaspace","replacement":"_","prepend_scheme":"never"}|} );
    ( Decoder.ctc (),
      {|{"type":"CTC","pad_token":"<pad>","word_delimiter_token":"|","cleanup":true}|}
    );
    ( Decoder.replace ~pattern:marker ~by:" " (),
      {|{"type":"Replace","pattern":{"String":"|} ^ marker
      ^ {|"},"content":" "}|} );
    ( Decoder.strip ~content:" " ~start:1 (),
      {|{"type":"Strip","content":" ","start":1,"stop":0}|} );
    (Decoder.fuse (), {|{"type":"Fuse"}|});
  ]

let test_decoder_json () =
  List.iter
    (fun (decoder, expected) ->
      let text = json_text (Decoder.to_json decoder) in
      equal ~msg:expected string expected text;
      match Decoder.of_json (json_of_text text) with
      | Error msg -> failf "%s does not read back: %s" expected msg
      | Ok reloaded ->
          equal ~msg:("round trip " ^ expected) string
            (Format.asprintf "%a" Decoder.pp decoder)
            (Format.asprintf "%a" Decoder.pp reloaded))
    decoders

(* HuggingFace also writes a regular expression pattern for [Replace], which
   brot does not read; the error must not claim the pattern is missing. *)
let test_replace_regex_json () =
  match
    Decoder.of_json
      (json_of_text
         {|{"type":"Replace","pattern":{"Regex":"a+"},"content":"b"}|})
  with
  | Ok _ -> failf "a regular expression pattern was accepted"
  | Error msg ->
      equal ~msg:"says why" string
        "Replace decoder: a regular expression pattern is not supported, only \
         a literal one"
        msg

(* The decoder LLaMA and other SentencePiece models carry, verbatim from
   [bench/data/llama.json]. *)
let test_sentencepiece_decoder_json () =
  let text =
    {|{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"|}
    ^ marker
    ^ {|"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]}|}
  in
  match Decoder.of_json (json_of_text text) with
  | Error msg -> failf "cannot read the LLaMA decoder: %s" msg
  | Ok decoder ->
      equal ~msg:"decodes" string "\n\nNot"
        (Decoder.decode decoder [ marker; "<0x0A>"; "<0x0A>"; "Not" ]);
      equal ~msg:"serializes back" string text
        (json_text (Decoder.to_json decoder))

(* Loading a pretrained tokenizer and writing it back must give HuggingFace the
   decoder it had. GPT-2's carries [add_prefix_space] and [trim_offsets], which
   the byte-level decoder ignores but HuggingFace insists on. *)
let test_pretrained_decoder_json () =
  let case model expected =
    Fixture.with_download
      ("../bench/data/" ^ model ^ ".json")
      ~from:"bench/download_data.sh"
      (fun path ->
        match from_file path with
        | Error msg -> failf "cannot load %s: %s" model msg
        | Ok t -> (
            match decoder t with
            | None -> failf "%s has no decoder" model
            | Some d ->
                equal ~msg:model string expected (json_text (Decoder.to_json d))
            ))
  in
  case "gpt2"
    {|{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true}|};
  case "bert_base" {|{"type":"WordPiece","prefix":"##","cleanup":true}|};
  case "llama"
    ({|{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"|}
   ^ marker
   ^ {|"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]}|}
    )

(* A BPE model whose flags each change how ["abc"] is tokenized, so that a flag
   dropped in either direction of the JSON shows up in the ids. *)
let bpe_model ~byte_fallback ~fuse_unk =
  Printf.sprintf
    {|{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":%b,"byte_fallback":%b,"ignore_merges":false,"vocab":{"<unk>":0,"a":1,"<0x62>":2,"<0x63>":3},"merges":[]}|}
    fuse_unk byte_fallback

let ignore_merges_model ~ignore_merges =
  Printf.sprintf
    {|{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":%b,"vocab":{"a":0,"b":1,"c":2,"ab":3,"bc":4,"abc":5},"merges":[["b","c"],["a","b"]]}|}
    ignore_merges

let tokenizer_text model =
  Printf.sprintf
    {|{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":%s}|}
    model

let load text =
  match from_json (json_of_text text) with
  | Ok t -> t
  | Error msg -> failf "cannot load %s: %s" text msg

let reload t =
  match from_json (to_json t) with
  | Ok t -> t
  | Error msg -> failf "cannot reload: %s" msg

(* Expectations from HuggingFace [Tokenizer.from_str] on the same models. *)
let test_bpe_flags_json () =
  let ids text t = encode_ids t ~add_special_tokens:false text in
  let case ~msg model text expected =
    let t = load (tokenizer_text model) in
    equal ~msg (array int) expected (ids text t);
    equal
      ~msg:(msg ^ " after a round trip")
      (array int) expected
      (ids text (reload t))
  in
  case ~msg:"neither flag"
    (bpe_model ~byte_fallback:false ~fuse_unk:false)
    "abc" [| 1; 0; 0 |];
  case ~msg:"fuse_unk"
    (bpe_model ~byte_fallback:false ~fuse_unk:true)
    "abc" [| 1; 0 |];
  case ~msg:"byte_fallback"
    (bpe_model ~byte_fallback:true ~fuse_unk:false)
    "abc" [| 1; 2; 3 |];
  case ~msg:"both flags"
    (bpe_model ~byte_fallback:true ~fuse_unk:true)
    "abc" [| 1; 2; 3 |];
  case ~msg:"merges apply"
    (ignore_merges_model ~ignore_merges:false)
    "abc" [| 0; 4 |];
  case ~msg:"ignore_merges"
    (ignore_merges_model ~ignore_merges:true)
    "abc" [| 5 |];
  (* The written members, spelled as HuggingFace reads them back. The vocabulary
     and the merges are left out: they are large, and a JSON object puts no
     order on its members. *)
  let written =
    bpe_model ~byte_fallback:true ~fuse_unk:true
    |> tokenizer_text |> load |> to_json |> json_member "model"
    |> json_without [ "vocab"; "merges" ]
    |> json_text
  in
  equal ~msg:"model json" string
    {|{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":true,"byte_fallback":true,"ignore_merges":false}|}
    written

(* LLaMA, end to end: byte fallback for the characters the vocabulary lacks, and
   a decoder that puts them back. Expectations from HuggingFace
   [Tokenizer.from_file("llama.json")]. *)
let test_llama () =
  Fixture.with_download "../bench/data/llama.json"
    ~from:"bench/download_data.sh" (fun path ->
      match from_file path with
      | Error msg -> failf "cannot load llama.json: %s" msg
      | Ok t ->
          let case tokenizer ~saved text ids =
            let msg text = if saved then text ^ " once saved" else text in
            equal ~msg:(msg text) (array int) ids
              (encode_ids tokenizer ~add_special_tokens:false text);
            equal
              ~msg:(msg ("decode " ^ text))
              string text
              (decode tokenizer ~skip_special_tokens:false ids)
          in
          (* Saving and loading the tokenizer back must keep the byte fallback:
             it is what turns the characters the vocabulary lacks into byte
             tokens rather than one unknown token. *)
          List.iter
            (fun (tokenizer, saved) ->
              let case = case tokenizer ~saved in
              case "Hello world" [| 15043; 3186 |];
              case "\n\nNot" [| 29871; 13; 13; 3664 |];
              case "a  b" [| 263; 29871; 289 |];
              case "  leading" [| 259; 8236 |];
              (* U+0085 is not in the vocabulary, so each of its two UTF-8 bytes
                 becomes a byte token. *)
              case "a\xc2\x85b" [| 263; 197; 136; 29890 |];
              equal ~msg:"BOS" (array int) [| 1; 15043; 3186 |]
                (Encoding.ids
                   (encode tokenizer ~add_special_tokens:true "Hello world")))
            [ (t, false); (reload t, true) ])

let () =
  run "brot json"
    [
      group "serialization"
        [
          test "decoder json" test_decoder_json;
          test "replace regex json" test_replace_regex_json;
          test "sentencepiece decoder json" test_sentencepiece_decoder_json;
          test "pretrained decoder json" test_pretrained_decoder_json;
          test "bpe flags json" test_bpe_flags_json;
          test "llama" test_llama;
        ];
    ]
