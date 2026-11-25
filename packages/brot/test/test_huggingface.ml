(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Model files, tokenizer.json and pp. An expected value whose test or comment
   names tokenizers is what tokenizers 0.23.1 gives, writes or reads back for
   the same tokenizer; the others follow from brot.mli. *)

open Windtrap
open Brot

let json_of_text text =
  match Jsont_bytesrw.decode_string Jsont.json text with
  | Ok json -> json
  | Error msg -> failf "cannot decode %S: %s" text msg

let json_text json =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json json with
  | Ok text -> text
  | Error msg -> failf "cannot encode: %s" msg

let member name = function
  | Jsont.Object (members, _) -> (
      match Jsont.Json.find_mem name members with
      | Some (_, value) -> value
      | None -> failf "no %S member" name)
  | _ -> failf "looking for %S in something that is not an object" name

let without names = function
  | Jsont.Object (members, meta) ->
      Jsont.Object
        ( List.filter (fun ((name, _), _) -> not (List.mem name names)) members,
          meta )
  | json -> json

let model_type t = json_text (member "type" (member "model" (to_json t)))

let load text =
  require_ok ~pp:Format.pp_print_string (of_json (json_of_text text))

let reload t = require_ok ~pp:Format.pp_print_string (of_json (to_json t))

let ids ?(add_special_tokens = false) t text =
  encode_ids t ~add_special_tokens text

let tokens t text =
  Array.to_list (Encoding.tokens (encode t ~add_special_tokens:false text))

let write_file path contents =
  Out_channel.with_open_bin path (fun oc ->
      Out_channel.output_string oc contents)

let read_file path = In_channel.with_open_bin path In_channel.input_all

(* The path of [t] saved by [save_pretrained] into a fresh directory. *)
let saved t =
  let path = Filename.concat (temp_dir ()) "saved" in
  save_pretrained t ~path;
  Filename.concat path "tokenizer.json"

(* The text a JSON value was decoded from. *)
let source text json =
  let loc = Jsont.Meta.textloc (Jsont.Json.meta json) in
  let first = Jsont.Textloc.first_byte loc in
  String.sub text first (Jsont.Textloc.last_byte loc - first + 1)

let decode_with_locs text =
  match Jsont_bytesrw.decode_string ~locs:true Jsont.json text with
  | Ok json -> json
  | Error msg -> failf "cannot decode the file: %s" msg

(* The texts of the numbers of a JSON text, in order. *)
let number_texts text =
  let rec numbers acc = function
    | Jsont.Number _ as n -> source text n :: acc
    | Jsont.Array (items, _) -> List.fold_left numbers acc items
    | Jsont.Object (members, _) ->
        List.fold_left (fun acc (_, v) -> numbers acc v) acc members
    | _ -> acc
  in
  List.rev (numbers [] (decode_with_locs text))

let pretrained name =
  Fixture.with_tokenizer_file name (fun path ->
      require_ok ~pp:Format.pp_print_string (from_file path))

let hex = Printf.sprintf "%h"

(* Unigram scores: support/parity/unigram_scores.txt, written by
   gen/gen_unigram_scores.py, holds score texts and the doubles tokenizers reads
   off them. *)

type score_rows = {
  read : (string * float) list;
  refused : string list;
  saved : (float * string * float) list;
}

let score_rows =
  lazy
    (let rows = { read = []; refused = []; saved = [] } in
     Fixture.read "support/parity/unigram_scores.txt"
     |> String.split_on_char '\n'
     |> List.fold_left
          (fun rows line ->
            match String.split_on_char ' ' line with
            | [ "READ"; text; value ] ->
                { rows with read = (text, float_of_string value) :: rows.read }
            | [ "ERROR"; text ] -> { rows with refused = text :: rows.refused }
            | [ "SAVED"; x; text; value ] ->
                let row = (float_of_string x, text, float_of_string value) in
                { rows with saved = row :: rows.saved }
            | [ "" ] -> rows
            | _ when line.[0] = '#' -> rows
            | _ -> failf "bad row %S" line)
          rows
     |> fun rows ->
     {
       read = List.rev rows.read;
       refused = List.rev rows.refused;
       saved = List.rev rows.saved;
     })

(* A tokenizer.json whose Unigram scores are [texts], written as they are. *)
let unigram_file texts =
  let entry i text = Printf.sprintf {|["t%d",%s]|} i text in
  Printf.sprintf
    {|{"version":"1.0","model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],%s]}}|}
    (String.concat "," (List.mapi entry texts))

(* The scores of a Unigram tokenizer, past its unknown token. *)
let scores t =
  match member "vocab" (member "model" (to_json t)) with
  | Jsont.Array (_ :: entries, _) ->
      List.map
        (function
          | Jsont.Array ([ _; Jsont.Number (score, _) ], _) -> score
          | _ -> failf "bad vocab entry")
        entries
  | _ -> failf "no vocab"

let from_text contents =
  let path = temp_file ~suffix:".json" () in
  write_file path contents;
  from_file path

let dropout t =
  match member "dropout" (member "model" (to_json t)) with
  | Jsont.Number (p, _) -> hex p
  | _ -> failf "no dropout"

(* save_model_files *)

let bpe_model () =
  bpe
    ~vocab:[ ("<unk>", 0); ("a", 1); ("b", 2); ("c", 3); ("ab", 4); ("abc", 5) ]
    ~merges:[ ("a", "b"); ("ab", "c") ]
    ~unk_token:"<unk>" ()

let test_bpe_files () =
  let folder = temp_dir () in
  let files = save_model_files (bpe_model ()) ~folder in
  let ending suffix =
    List.find (fun f -> Filename.check_suffix f suffix) files
  in
  let read =
    from_model_file ~vocab:(ending ".json") ~merges:(ending ".txt")
      ~unk_token:"<unk>" ()
  in
  List.iter
    (fun text ->
      equal ~msg:text (array int) (ids (bpe_model ()) text) (ids read text))
    [ "abc"; "ab"; "cab"; "" ]

let wordpiece_model () =
  wordpiece
    ~vocab:
      [
        ("[PAD]", 0);
        ("[UNK]", 1);
        ("[CLS]", 2);
        ("[SEP]", 3);
        ("hello", 4);
        ("world", 5);
        ("##s", 6);
      ]
    ()

let test_wordpiece_files () =
  let folder = temp_dir () in
  let read =
    match save_model_files (wordpiece_model ()) ~folder with
    | [ vocab ] -> from_model_file ~vocab ()
    | files -> failf "%d files" (List.length files)
  in
  List.iter
    (fun text ->
      equal ~msg:text (array int)
        (ids (wordpiece_model ()) text)
        (ids read text))
    [ "hello"; "worlds"; "hellos world"; "xyz" ]

let test_paths =
  cases "returns the paths of the files it writes in folder" ~id:fst
    [
      ("bpe", bpe_model);
      ("wordpiece", wordpiece_model);
      ("chars", fun () -> chars ());
    ]
    (fun (_, make) ->
      let folder = temp_dir () in
      let files = save_model_files (make ()) ~folder in
      List.iter (fun f -> equal ~msg:f string folder (Filename.dirname f)) files;
      equal (list string)
        (List.sort String.compare (Array.to_list (Sys.readdir folder)))
        (List.sort String.compare (List.map Filename.basename files)))

let test_model_file_scores () =
  let vocab = [ ("<unk>", 0.0); ("a", -0.1); ("b", -1e16); ("c", -1.5e-7) ] in
  let folder = temp_dir () in
  let written =
    save_model_files (unigram ~vocab ~unk_id:0 ()) ~folder
    |> List.concat_map (fun f ->
        number_texts (read_file (Filename.concat folder (Filename.basename f))))
  in
  List.iter
    (fun score -> mem ~msg:score string score written)
    [ "0.0"; "-0.1"; "-1e+16"; "-1.5e-7" ]

(* from_file *)

let test_scores_read () =
  let rows = (Lazy.force score_rows).read in
  let t =
    require_ok ~pp:Format.pp_print_string
      (from_text (unigram_file (List.map fst rows)))
  in
  equal
    (list (pair string string))
    (List.map (fun (text, value) -> (text, hex value)) rows)
    (List.combine (List.map fst rows) (List.map hex (scores t)))

let test_scores_refused () =
  let refused = (Lazy.force score_rows).refused in
  let loaded text = Result.map ignore (from_text (unigram_file [ text ])) in
  equal
    (list (pair string (result unit string)))
    (List.map (fun text -> (text, Error "number out of range")) refused)
    (List.map (fun text -> (text, loaded text)) refused)

(* T5's [---] reads one unit in the last place above its decimal, so a run of
   dashes does not tie. *)
let t5 = fixture (fun () -> pretrained "t5_base")

let d = "-"
and d3 = "---"
and m = "\u{2581}"

let test_t5_dashes =
  cases "cuts T5's runs of dashes as tokenizers does" ~id:fst
    [
      ("-", [ m; d ]);
      ("--", [ m ^ "--" ]);
      ("---", [ m; d3 ]);
      ("----", [ m; d; d3 ]);
      ("-----", [ m ^ "--"; d3 ]);
      ("------", [ m; d3; d3 ]);
      ("-------", [ m; d3; d3; d ]);
      ("--------", [ m ^ "--"; d3; d3 ]);
      ("---------", [ m; d3; d3; d3 ]);
      ("----------", [ m; d3; d3; d; d3 ]);
      ("-----------", [ m ^ "--"; d3; d3; d3 ]);
      ("------------", [ m; d3; d3; d3; d3 ]);
      ("-------------", [ m; d3; d3; d; d3; d3 ]);
      ("\n\n-----------", [ m ^ "--"; d3; d3; d3 ]);
      ("a -------", [ m; "a"; m; d3; d3; d ]);
      ("----a----", [ m; d; d3; "a"; d; d3 ]);
    ]
    (fun (text, expected) -> equal (list string) expected (tokens (t5 ()) text))

let llama_rows =
  [
    ("Hello world", [| 15043; 3186 |]);
    ("\n\nNot", [| 29871; 13; 13; 3664 |]);
    ("a  b", [| 263; 29871; 289 |]);
    ("  leading", [| 259; 8236 |]);
    (* U+0085 is not in the vocabulary: its two bytes are byte tokens. *)
    ("a\xc2\x85b", [| 263; 197; 136; 29890 |]);
  ]

let llama = fixture (fun () -> pretrained "llama")

let test_llama () =
  let t = llama () in
  List.iter
    (fun (text, expected) ->
      equal ~msg:text (array int) expected (ids t text);
      equal ~msg:("decode " ^ text) string text (decode t expected))
    llama_rows;
  equal ~msg:"with BOS" (array int) [| 1; 15043; 3186 |]
    (ids ~add_special_tokens:true t "Hello world")

(* GPT-2's decoder carries [add_prefix_space] and [trim_offsets], which the
   byte-level decoder ignores and tokenizers requires. Mistral's is LLaMA's
   SentencePiece decoder. *)
let test_file_decoder =
  cases "reads a file's decoder as tokenizers wrote it" ~id:fst
    [
      ( "gpt2",
        {|{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true}|} );
      ("bert_base", {|{"type":"WordPiece","prefix":"##","cleanup":true}|});
      ( "mistral",
        {|{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]}|}
      );
    ]
    (fun (name, expected) ->
      let d = require_some ~msg:"a decoder" (decoder (pretrained name)) in
      equal string expected (json_text (Decoder.to_json d)))

(* GPT-2's keeps [add_prefix_space], without which tokenizers refuses the
   file. *)
let test_file_post_processor =
  cases "reads a file's post-processor as tokenizers wrote it" ~id:fst
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
    (fun (name, expected) ->
      let p =
        require_some ~msg:"a post-processor" (post_processor (pretrained name))
      in
      equal string expected (json_text (Post_processor.to_json p)))

(* tokenizers holds a dropout as an f32: [0.1] is 0x1.99999ap-4. *)
let dropout_file =
  {|{"model":{"type":"BPE","dropout":0.1,"vocab":{"a":0},"merges":[]}}|}

let test_file_dropout () =
  let t = require_ok ~pp:Format.pp_print_string (from_text dropout_file) in
  equal string "0x1.99999ap-4" (dropout t)

let test_file_errors =
  cases "errors on a file it cannot read or parse" ~id:fst
    [
      ("a missing file", fun dir -> Filename.concat dir "missing.json");
      ("a directory", Fun.id);
      ( "an empty file",
        fun dir ->
          let path = Filename.concat dir "empty.json" in
          write_file path "";
          path );
      ( "an unclosed object",
        fun dir ->
          let path = Filename.concat dir "brace.json" in
          write_file path "{";
          path );
      ( "invalid UTF-8",
        fun dir ->
          let path = Filename.concat dir "bytes.json" in
          write_file path
            ({|{"model":{"type":"WordLevel","vocab":{"|} ^ "\xff"
           ^ {|":0},"unk_token":"a"}}|});
          path );
    ]
    (fun (_, path) -> is_error ~pp:Brot.pp (from_file (path (temp_dir ()))))

(* of_json *)

let test_no_model =
  cases "errors without a model or with a model of unknown type" ~id:Fun.id
    [
      {|{}|};
      {|[]|};
      {|"model"|};
      {|{"model":null}|};
      {|{"model":{"type":"Nope","vocab":{}}}|};
    ] (fun text -> is_error ~pp:Brot.pp (of_json (json_of_text text)))

let test_untyped_model =
  cases "reads a model without type as the model its members name" ~id:fst
    [
      ({|{"merges":[],"vocab":{"a":0}}|}, {|"BPE"|});
      ({|{"vocab":[["a",-1.0]],"unk_id":0}|}, {|"Unigram"|});
      ( {|{"vocab":{"[UNK]":0,"a":1},"continuing_subword_prefix":"##"}|},
        {|"WordPiece"|} );
      ( {|{"vocab":{"[UNK]":0,"a":1},"max_input_chars_per_word":3}|},
        {|"WordPiece"|} );
      ({|{"vocab":{"a":0}}|}, {|"WordLevel"|});
    ]
    (fun (model, expected) ->
      equal string expected (model_type (load ({|{"model":|} ^ model ^ "}"))))

let test_invalid_parameters =
  cases "errors on invalid parameters" ~id:fst
    [
      ( "a merge of a token not in vocab",
        {|{"model":{"type":"BPE","vocab":{"a":0},"merges":[["a","z"]]}}|} );
      ( "vocab not an object",
        {|{"model":{"type":"BPE","vocab":[["a",0]],"merges":[]}}|} );
      ( "an id that is a string",
        {|{"model":{"type":"BPE","vocab":{"a":"0"},"merges":[]}}|} );
      ("an id of -1", {|{"model":{"type":"BPE","vocab":{"a":-1},"merges":[]}}|});
      ( "an id of 2^31",
        {|{"model":{"type":"BPE","vocab":{"a":2147483648},"merges":[]}}|} );
      ( "an added token without content",
        {|{"added_tokens":[{"id":5}],"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}|}
      );
      ( "an unk_id past the vocabulary",
        {|{"model":{"type":"Unigram","vocab":[["a",-1.0]],"unk_id":3}}|} );
      ( "an id of 1.5",
        {|{"model":{"type":"BPE","vocab":{"a":1.5},"merges":[]}}|} );
      ( "an id of 1e300",
        {|{"model":{"type":"BPE","vocab":{"a":1e300},"merges":[]}}|} );
      ( "an added token id of 5.5",
        {|{"added_tokens":[{"id":5.5,"content":"x"}],"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}|}
      );
      ( "an added token id of -5",
        {|{"added_tokens":[{"id":-5,"content":"x"}],"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}|}
      );
      ( "an added token id that is a string",
        {|{"added_tokens":[{"id":"5","content":"x"}],"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}|}
      );
      ( "a max_input_chars_per_word of 2.5",
        {|{"model":{"type":"WordPiece","vocab":{"[UNK]":0,"a":1},"max_input_chars_per_word":2.5}}|}
      );
    ]
    (fun (_, text) -> is_error ~pp:Brot.pp (of_json (json_of_text text)))

(* Decoded by jsont, the text names the double -12.130167007446289, where
   tokenizers reads one unit in the last place below it. *)
let test_numbers_as_given () =
  let t = load (unigram_file [ "-12.130167007446289" ]) in
  equal (list string) [ hex (-12.130167007446289) ] (List.map hex (scores t))

let test_json_dropout () =
  equal string "0x1.99999ap-4" (dropout (load dropout_file))

(* A BPE model whose flags each change how ["abc"] is cut. *)
let flagged_model ~byte_fallback ~fuse_unk =
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

let test_bpe_flags =
  cases "reads fuse_unk, byte_fallback and ignore_merges as tokenizers does"
    ~id:(fun (name, _, _) -> name)
    [
      ( "neither flag",
        flagged_model ~byte_fallback:false ~fuse_unk:false,
        [| 1; 0; 0 |] );
      ("fuse_unk", flagged_model ~byte_fallback:false ~fuse_unk:true, [| 1; 0 |]);
      ( "byte_fallback",
        flagged_model ~byte_fallback:true ~fuse_unk:false,
        [| 1; 2; 3 |] );
      ( "both flags",
        flagged_model ~byte_fallback:true ~fuse_unk:true,
        [| 1; 2; 3 |] );
      ("merges", ignore_merges_model ~ignore_merges:false, [| 0; 4 |]);
      ("ignore_merges", ignore_merges_model ~ignore_merges:true, [| 5 |]);
    ]
    (fun (_, model, expected) ->
      equal (array int) expected (ids (load (tokenizer_text model)) "abc"))

let flags (a : added_token) =
  Printf.sprintf
    "%s special=%b single_word=%b lstrip=%b rstrip=%b normalized=%b" a.content
    a.special a.single_word a.lstrip a.rstrip a.normalized

(* The defaults of [added_token]. *)
let test_added_token_defaults =
  cases
    "reads a missing special as true and a missing normalized as not special"
    ~id:fst
    [
      ( {|{"id":0,"content":"<x>"}|},
        "<x> special=true single_word=false lstrip=false rstrip=false \
         normalized=false" );
      ( {|{"id":0,"content":"<x>","special":false}|},
        "<x> special=false single_word=false lstrip=false rstrip=false \
         normalized=true" );
      ( {|{"id":0,"content":"<x>","special":true}|},
        "<x> special=true single_word=false lstrip=false rstrip=false \
         normalized=false" );
      ( {|{"id":0,"content":"<x>","normalized":true}|},
        "<x> special=true single_word=false lstrip=false rstrip=false \
         normalized=true" );
    ]
    (fun (entry, expected) ->
      let t =
        load
          (Printf.sprintf
             {|{"added_tokens":[%s],"model":{"type":"WordLevel","vocab":{"a":0},"unk_token":"a"}}|}
             entry)
      in
      equal (list string) [ expected ] (List.map flags (added_tokens t)))

let test_added_token_ids () =
  let t =
    load
      {|{"added_tokens":[{"id":50,"content":"<x>"},{"id":0,"content":"<y>"}],"model":{"type":"WordLevel","vocab":{"a":0,"b":1},"unk_token":"a"}}|}
  in
  equal
    (list (option int))
    [ Some 2; Some 3 ]
    [ token_to_id t "<x>"; token_to_id t "<y>" ]

let test_no_truncation_or_padding () =
  let t =
    load
      {|{"truncation":{"max_length":1,"stride":0,"strategy":"LongestFirst","direction":"Right"},"padding":{"strategy":{"Fixed":4},"direction":"Right","pad_to_multiple_of":null,"pad_id":0,"pad_type_id":0,"pad_token":"a"},"pre_tokenizer":{"type":"WhitespaceSplit"},"model":{"type":"WordLevel","vocab":{"a":0,"b":1},"unk_token":"a"}}|}
  in
  equal (array int) [| 1; 1 |] (ids t "b b")

(* The model's [unk_token] member, and for a Unigram the entry at [unk_id], of
   the file to_json writes. *)
let test_unk_token =
  cases "reads the model's unknown token as unk_token" ~id:fst
    [
      ( "bpe",
        fun () -> bpe ~vocab:[ ("<unk>", 0); ("a", 1) ] ~unk_token:"<unk>" () );
      ( "wordpiece",
        fun () ->
          wordpiece ~vocab:[ ("<unk>", 0); ("a", 1) ] ~unk_token:"<unk>" () );
      ( "word_level",
        fun () ->
          word_level ~vocab:[ ("<unk>", 0); ("a", 1) ] ~unk_token:"<unk>" () );
      ( "unigram",
        fun () -> unigram ~vocab:[ ("<unk>", 0.0); ("a", -1.0) ] ~unk_id:0 () );
    ]
    (fun (_, make) ->
      equal (option string) (Some "<unk>") (unk_token (reload (make ()))))

(* to_json *)

let test_written_bpe_flags () =
  let t =
    load (tokenizer_text (flagged_model ~byte_fallback:true ~fuse_unk:true))
  in
  equal string
    {|{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":true,"byte_fallback":true,"ignore_merges":false}|}
    (json_text (without [ "vocab"; "merges" ] (member "model" (to_json t))))

let byte_tokens = List.init 256 (fun b -> (Printf.sprintf "<0x%02X>" b, -20.0))

let unigram_model () =
  unigram
    ~vocab:([ ("<unk>", 0.0); ("a", -1.0); ("b", -1.0) ] @ byte_tokens)
    ~unk_id:0 ~byte_fallback:true ~unk_token:"<unk>" ()

let test_written_unigram () =
  equal string {|{"type":"Unigram","unk_id":0,"byte_fallback":true}|}
    (json_text
       (without [ "vocab" ] (member "model" (to_json (unigram_model ())))))

let test_written_chars () = equal string {|"Chars"|} (model_type (chars ()))

let test_written_roles () =
  let t =
    word_level
      ~vocab:[ ("[UNK]", 0); ("a", 1) ]
      ~unk_token:"[UNK]" ~bos_token:"<s>" ~eos_token:"</s>" ~pad_token:"<pad>"
      ()
  in
  let read = reload t in
  equal ~msg:"roles"
    (list (option string))
    [ None; None; None ]
    [ bos_token read; eos_token read; pad_token read ];
  equal ~msg:"added tokens" (list string)
    (List.map flags (added_tokens t))
    (List.map flags (added_tokens read))

(* word_level's unk_token defaults to "<unk>". *)
let test_written_word_level_unk () =
  let t = word_level ~vocab:[ ("<unk>", 0); ("a", 1) ] () in
  equal ~msg:"before" (array int) [| 1; 0 |] (ids t "a b");
  equal ~msg:"read back" (array int) [| 1; 0 |] (ids (reload t) "a b")

(* Tokenizers [to_json] can write, each built anew per case. *)
let models =
  [
    ("bpe", bpe_model);
    ( "bpe with fuse_unk",
      fun () ->
        bpe
          ~vocab:[ ("<unk>", 0); ("a", 1); ("b", 2) ]
          ~unk_token:"<unk>" ~fuse_unk:true () );
    (* Byte tokens for the bytes past ASCII alone, so that an ASCII character
       the vocabulary lacks is unknown and any other is spelled in bytes. *)
    ( "bpe with byte_fallback",
      fun () ->
        bpe
          ~vocab:
            (("<unk>", 0) :: ("a", 1)
            :: List.init 128 (fun i ->
                (Printf.sprintf "<0x%02X>" (128 + i), i + 2)))
          ~unk_token:"<unk>" ~byte_fallback:true () );
    ( "bpe with ignore_merges",
      fun () ->
        bpe
          ~vocab:
            [ ("a", 0); ("b", 1); ("c", 2); ("ab", 3); ("bc", 4); ("abc", 5) ]
          ~merges:[ ("b", "c"); ("a", "b") ]
          ~ignore_merges:true () );
    ( "bpe with affixes",
      fun () ->
        bpe
          ~vocab:
            [ ("a", 0); ("##b", 1); ("c</w>", 2); ("##c</w>", 3); ("a##b", 4) ]
          ~continuing_subword_prefix:"##" ~end_of_word_suffix:"</w>" () );
    ( "bpe with a pipeline and added tokens",
      fun () ->
        bpe ~normalizer:Normalizer.lowercase ~pre:Pre_tokenizer.whitespace
          ~decoder:Decoder.fuse
          ~vocab:[ ("a", 0); ("b", 1); ("c", 2); ("ab", 3) ]
          ~merges:[ ("a", "b") ]
          ~added_tokens:
            [
              added_token "<s>";
              added_token ~special:false ~single_word:true ~lstrip:true "<p>";
              added_token ~normalized:true ~rstrip:true "<q>";
            ]
          () );
    ("wordpiece", wordpiece_model);
    ( "wordpiece with @@",
      fun () ->
        wordpiece
          ~vocab:
            [ ("[UNK]", 0); ("hello", 1); ("@@s", 2); ("a", 3); ("@@b", 4) ]
          ~continuing_subword_prefix:"@@" () );
    ( "word_level",
      fun () ->
        word_level
          ~vocab:[ ("[UNK]", 0); ("a", 1); ("hello", 2) ]
          ~unk_token:"[UNK]" () );
    ("unigram", unigram_model);
    ("chars", fun () -> chars ());
  ]

(* Each of [models] with what of_json reads back from its to_json. *)
let read_back =
  fixture (fun () ->
      List.map
        (fun (name, make) ->
          let t = make () in
          (name, t, reload t))
        models)

let text =
  let open Gen in
  let piece =
    of_list ~pp:Format.pp_print_string
      [
        "a";
        "b";
        "c";
        "ab";
        "abc";
        "hello";
        "s";
        "x";
        " ";
        "<s>";
        "<p>";
        " <p>";
        "<q> ";
        "A";
        "😀";
        "\xff";
        "\xc2\x85";
      ]
  in
  map (String.concat "") (list ~size:(int_range 0 8) piece)

let holds subs text =
  List.exists
    (fun sub ->
      let n = String.length sub in
      let rec at i =
        i + n <= String.length text && (String.sub text i n = sub || at (i + 1))
      in
      at 0)
    subs

let test_read_back =
  prop "is read back by of_json into a tokenizer that encodes and decodes alike"
    text (fun text ->
      cover "a merged pair" (holds [ "ab" ] text);
      cover "an added token" (holds [ "<s>"; "<p>"; "<q>" ] text);
      cover "a character no vocabulary holds"
        (holds [ "😀"; "\xc2\x85"; "\xff" ] text);
      List.iter
        (fun (name, t, read) ->
          let expected = ids t text in
          equal ~msg:(name ^ ": ids") (array int) expected (ids read text);
          equal ~msg:(name ^ ": decode") string
            (decode t ~skip_special_tokens:true expected)
            (decode read ~skip_special_tokens:true expected);
          equal ~msg:(name ^ ": added tokens") (list string)
            (List.map flags (added_tokens t))
            (List.map flags (added_tokens read)))
        (read_back ()))

let test_llama_read_back () =
  let t = reload (llama ()) in
  List.iter
    (fun (text, expected) -> equal ~msg:text (array int) expected (ids t text))
    llama_rows

(* save_pretrained *)

(* GPT-2 holds the byte-level stages and the post-processor brot once wrote so
   that tokenizers refused the file (the file saved this way, tokenizers 0.23.1
   reads and encodes alike); BERT holds a normalizer, a template and a WordPiece
   model. Between them every stage is present. Unigram files are read back in
   "writes Unigram scores as tokenizers writes them". *)
let test_saved_file =
  cases
    "creates path/tokenizer.json, which from_file reads back into the same \
     pipeline, encoding alike"
    ~id:Fun.id [ "gpt2"; "bert_base" ] (fun name ->
      let t = pretrained name in
      let read = require_ok ~pp:Format.pp_print_string (from_file (saved t)) in
      let written = to_json t and read_back = to_json read in
      List.iter
        (fun stage ->
          equal ~msg:stage string
            (json_text (member stage written))
            (json_text (member stage read_back)))
        [ "normalizer"; "pre_tokenizer"; "post_processor"; "decoder" ];
      let text = " Hello  world,\n\ncaf\xc3\xa9!" in
      equal ~msg:"ids" (array int)
        (ids ~add_special_tokens:true t text)
        (ids ~add_special_tokens:true read text))

(* What tokenizers 0.23.1's [to_str] writes for [BPE(dropout=p)]. *)
let test_saved_dropout =
  cases
    "writes a dropout as the shortest decimal that reads back in single \
     precision"
    ~id:snd
    [
      (0.1, "0.1");
      (0.3, "0.3");
      (1. /. 3., "0.33333334");
      (0.5, "0.5");
      (1.0, "1.0");
      (0.0, "0.0");
      (2e-5, "0.00002");
      (1e-6, "0.000001");
      (1.5e-6, "0.0000015");
      (1e-7, "1e-7");
    ]
    (fun (p, expected) ->
      let text = read_file (saved (bpe ~vocab:[ ("a", 0) ] ~dropout:p ())) in
      let json = decode_with_locs text in
      equal string expected
        (source text (member "dropout" (member "model" json))))

let test_saved_scores () =
  let rows = (Lazy.force score_rows).saved in
  let vocab =
    ("<unk>", 0.0)
    :: List.mapi (fun i (x, _, _) -> (Printf.sprintf "t%d" i, x)) rows
  in
  let path = saved (unigram ~vocab ~unk_id:0 ()) in
  (* The unk_id, then the unknown token's score, then the rows'. *)
  equal ~msg:"texts written" (list string)
    ("0" :: "0.0" :: List.map (fun (_, text, _) -> text) rows)
    (number_texts (read_file path));
  equal ~msg:"scores read back" (list string)
    (List.map (fun (_, _, value) -> hex value) rows)
    (List.map hex
       (scores (require_ok ~pp:Format.pp_print_string (from_file path))))

let test_unwritable_path () =
  let file = temp_file () in
  raises_match (Exn.sys_error ~sub:"") (fun () ->
      save_pretrained (chars ()) ~path:(Filename.concat file "saved"))

(* pp *)

let test_pp () =
  let shown t = Format.asprintf "%a@." pp t in
  print_string
    (shown (bpe ~vocab:[ ("a", 0); ("b", 1) ] ~pre:Pre_tokenizer.whitespace ()));
  print_string (shown (chars ~decoder:Decoder.fuse ()));
  expect (output ())
  @@ __POS_OF__
       {|
    <brot BPE vocab=2 normalizer=no pre=yes post=no decoder=no>
    <brot Chars vocab=256 normalizer=no pre=no post=no decoder=yes>
    |}

let () =
  exit
    (run "HuggingFace compatibility"
       [
         group "save_model_files"
           [
             test
               "writes vocab.json and merges.txt, which from_model_file reads \
                back"
               test_bpe_files;
             test
               "writes a WordPiece vocab.txt, which from_model_file reads back"
               test_wordpiece_files;
             test_paths;
             test "writes Unigram scores as tokenizers writes them"
               test_model_file_scores;
           ];
         group "from_file"
           [
             test "reads Unigram scores as tokenizers reads their text"
               test_scores_read;
             test "refuses a score tokenizers refuses as out of range"
               test_scores_refused;
             test_t5_dashes;
             test "gives tokenizers' ids and text on LLaMA" test_llama;
             test_file_decoder;
             test_file_post_processor;
             test "holds a BPE dropout in single precision" test_file_dropout;
             test_file_errors;
           ];
         group "of_json"
           [
             test_no_model;
             test_untyped_model;
             test_invalid_parameters;
             test "takes the numbers of a tree as given" test_numbers_as_given;
             test "holds a BPE dropout in single precision" test_json_dropout;
             test_bpe_flags;
             test_added_token_defaults;
             test
               "numbers added tokens from the end of the vocabulary, whatever \
                their id in the file"
               test_added_token_ids;
             test "reads no truncation or padding member"
               test_no_truncation_or_padding;
             test_unk_token;
           ];
         group "to_json"
           [
             test "writes the BPE flags as tokenizers names them"
               test_written_bpe_flags;
             test
               "writes a Unigram's unk_id and byte_fallback as tokenizers \
                names them"
               test_written_unigram;
             test "writes a chars tokenizer with the model type Chars"
               test_written_chars;
             test
               "writes no role marker, so of_json reads them back as added \
                tokens alone"
               test_written_roles;
             test "writes a WordLevel's unknown token"
               test_written_word_level_unk;
             test_read_back;
             test "keeps LLaMA's byte fallback through of_json"
               test_llama_read_back;
           ];
         group "save_pretrained"
           [
             test_saved_file;
             test_saved_dropout;
             test
               "writes Unigram scores as tokenizers writes them, and integers \
                as digits"
               test_saved_scores;
             test "raises Sys_error when path cannot be written"
               test_unwritable_path;
           ];
         group "pp"
           [
             test
               "shows the model type, its vocabulary size and which stages it \
                has"
               test_pp;
           ];
       ])
