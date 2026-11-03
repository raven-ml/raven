(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The fused kernels' resume protocols, observed through the public API. The
   native binary runs the C kernels and the byte_complete one the OCaml
   reference, so every expectation here is checked against both; the dump
   differential separately holds the two identical on adversarial input. Each
   byte-level case drives one hand-back or refusal: a document crossing the span
   chunk, hand-backs at the stage's seams, maximal multi-id spans, caching off,
   [ignore_merges], dropout (kernel not selected), pretokens past the 15-byte
   key, bytes without an id, and a merge whose result stands for other bytes
   than its entry's. The SentencePiece cases drive the unit walker's: units
   past the key, characters without a piece, the resume look-behind at a ▁
   run, punctuation adjacency, byte-fallback caching, a stream outgrowing one
   call's ids reserve, and the len_table refusal. Laws then hold both kernels
   to a reference merge on drawn text. *)

open Windtrap

(* The fixed GPT-2 byte-level alphabet, restated: the library does not expose
   the encoding direction. *)
let byte_alphabet =
  let direct b = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 in
  let next = ref 0 in
  Array.init 256 (fun b ->
      let code =
        if direct b then b
        else begin
          let c = 256 + !next in
          incr next;
          c
        end
      in
      let buf = Buffer.create 2 in
      Buffer.add_utf_8_uchar buf (Uchar.of_int code);
      Buffer.contents buf)

let pre () = Brot.Pre_tokenizer.byte_level ~add_prefix_space:false ()

(* Ids 0..255 are the bytes themselves; " a" merges to [space_a]. *)
let space_a = 256
let base_vocab = List.init 256 (fun b -> (byte_alphabet.(b), b))

let simple ?cache_capacity ?dropout ?ignore_merges () =
  let vocab = base_vocab @ [ ("\196\160a", space_a) ] in
  Brot.bpe ~vocab
    ~merges:[ ("\196\160", "a") ]
    ~pre:(pre ()) ?cache_capacity ?dropout ?ignore_merges ()

let ids tok text = Brot.encode_ids tok ~add_special_tokens:false text

let byte_ids text =
  Array.init (String.length text) (fun i -> Char.code text.[i])

(* 1500 pretokens " a": crosses the stage and the 1024-span buffer, every span
   a cache hit of one id. *)
let test_chunk_boundary () =
  let tok = simple () in
  let text = String.concat "" (List.init 1500 (fun _ -> " a")) in
  equal ~msg:"ids" (array int) (Array.make 1500 space_a) (ids tok text);
  let e = Brot.encode tok ~add_special_tokens:false text in
  equal ~msg:"offsets"
    (array (pair int int))
    (Array.init 1500 (fun i -> (2 * i, (2 * i) + 2)))
    (Brot.Encoding.offsets e);
  equal ~msg:"words"
    (array (option int))
    (Array.init 1500 (fun i -> Some i))
    (Brot.Encoding.word_ids e)

(* A 20-byte pretoken between two chunk-crossing runs: an [Encode] hand-back for
   length, with the marks it appends staying aligned with the spans. *)
let test_long_pretoken_hand_back () =
  let tok = simple () in
  let long = String.make 20 'b' in
  let text =
    String.concat ""
      (List.init 600 (fun _ -> " a")
      @ [ " "; long ]
      @ List.init 600 (fun _ -> " a"))
  in
  let expect =
    Array.concat
      [
        Array.make 600 space_a;
        [| 32 |];
        Array.make 20 (Char.code 'b');
        Array.make 600 space_a;
      ]
  in
  equal ~msg:"ids" (array int) expect (ids tok text);
  let e = Brot.encode tok ~add_special_tokens:false text in
  let offsets = Brot.Encoding.offsets e in
  equal ~msg:"tokens count" int (Array.length expect) (Array.length offsets);
  (* Span 600 is the hand-back: its 21 byte tokens tile [1200;1221). *)
  equal ~msg:"first token of the hand-back" (pair int int) (1200, 1201)
    offsets.(600);
  equal ~msg:"last token of the hand-back" (pair int int) (1220, 1221)
    offsets.(620);
  let words = Brot.Encoding.word_ids e in
  equal ~msg:"tokens of the hand-back share its span" (option int) (Some 600)
    words.(610);
  equal ~msg:"the span after the hand-back" (option int) (Some 601) words.(621)

(* Pretoken counts at and around the seams of the stage, which holds 256 spans
   and up to a batch more, exact ids each. *)
let test_chunk_seam_lengths () =
  let tok = simple () in
  List.iter
    (fun count ->
      let text = String.concat "" (List.init count (fun _ -> " a")) in
      equal
        ~msg:(Printf.sprintf "%d pretokens" count)
        (array int) (Array.make count space_a) (ids tok text))
    [ 1; 63; 64; 65; 127; 128; 129; 255; 256; 257; 258; 511; 512; 513 ]

(* An [Encode] hand-back at the first and at the last span of a stage: the
   long span is staged with the rest, the exit keeps the spans staged after it
   in the cursor, and the re-entry after [encode_into] resolves them before
   walking on. *)
let test_encode_at_chunk_edges () =
  let tok = simple () in
  let long = String.make 20 'b' in
  let long_ids = Array.append [| 32 |] (Array.make 20 (Char.code 'b')) in
  List.iter
    (fun k ->
      let text =
        String.concat ""
          (List.init k (fun _ -> " a")
          @ [ " "; long ]
          @ List.init 300 (fun _ -> " a"))
      in
      let expect =
        Array.concat [ Array.make k space_a; long_ids; Array.make 300 space_a ]
      in
      equal
        ~msg:(Printf.sprintf "hand-back at span %d" k)
        (array int) expect (ids tok text))
    [ 0; 62; 63; 64; 65; 254; 255; 256; 257 ]

(* A class hand-back right after an [Encode] resume: the walk that re-enters
   after the hand-back meets a code point classified on demand. *)
let test_class_after_encode_resume () =
  let tok = simple () in
  let buf = Buffer.create 1024 in
  for _ = 1 to 255 do
    Buffer.add_string buf " a"
  done;
  Buffer.add_string buf (" " ^ String.make 20 'b');
  Buffer.add_utf_8_uchar buf (Uchar.of_int 0x2723);
  Buffer.add_string buf " a";
  let expect =
    Array.concat
      [
        Array.make 255 space_a;
        [| 32 |];
        Array.make 20 (Char.code 'b');
        [| 226; 156; 163 |];
        [| space_a |];
      ]
  in
  equal ~msg:"ids" (array int) expect (ids tok (Buffer.contents buf))

(* 15-byte pretokens with no merges: chunks of maximal multi-id spans across the
   span-chunk boundary. The per-call ids reserve covers this consumption, so
   [Ids_full] itself does not fire here; the dump differential reaches it on its
   generated inputs. *)
let test_many_ids_per_span () =
  let tok = Brot.bpe ~vocab:base_vocab ~merges:[] ~pre:(pre ()) () in
  let piece = "abcdefghijklmno" in
  let text = String.concat "1" (List.init 800 (fun _ -> piece)) in
  equal ~msg:"ids" (array int) (byte_ids text) (ids tok text)

(* A single span far larger than the ids headroom goes back to OCaml whole. *)
let test_huge_span () =
  let tok = Brot.bpe ~vocab:base_vocab ~merges:[] ~pre:(pre ()) () in
  let text = String.make 8000 'c' in
  equal ~msg:"ids" (array int) (byte_ids text) (ids tok text)

(* Caching off still stages and merges in C — the spans are unkeyed and the
   probes skipped; 600 pretokens cross chunk seams in that shape. *)
let test_cache_off () =
  let tok = simple ~cache_capacity:0 () in
  let text = String.concat "" (List.init 600 (fun _ -> " a")) in
  equal ~msg:"ids" (array int) (Array.make 600 space_a) (ids tok text)

(* Under ignore_merges a pretoken that is a vocabulary entry is that entry,
   whether the merges build something else ("abc") or build it ("ab"), and the
   others merge. The pretokens miss the cache, are looked up whole and stored,
   then hit; a cache of one set evicts them, so they miss again; and with no
   cache every one is looked up whole. *)
let test_ignore_merges () =
  let vocab =
    base_vocab
    @ [ ("\196\160a", 256); ("ab", 257); ("abc", 258); ("\196\160abc", 259) ]
  in
  let merges = [ ("\196\160", "a"); ("a", "b") ] in
  let make ?cache_capacity ignore_merges =
    Brot.bpe ~vocab ~merges ~pre:(pre ()) ?cache_capacity ~ignore_merges ()
  in
  let text =
    "abc\nab" ^ String.concat "" (List.init 50 (fun _ -> " abc abcd ab"))
  in
  let expect =
    Array.concat
      ([| 258; 10; 257 |]
      :: List.init 50 (fun _ -> [| 259; 256; 98; 99; 100; 256; 98 |]))
  in
  equal ~msg:"merged without ignore_merges" (array int)
    [| 257; 99; 256; 98; 99 |]
    (ids (make false) "abc abc");
  List.iter
    (fun (name, cache_capacity) ->
      equal ~msg:name (array int) expect (ids (make ?cache_capacity true) text))
    [ ("cached", None); ("evicted", Some 2); ("no cache", Some 0) ]

(* Dropout rules the kernel out at selection; at probability one every merge is
   skipped, deterministically. *)
let test_dropout_not_selected () =
  let tok = simple ~dropout:1.0 () in
  let text = " a a" in
  equal ~msg:"ids" (array int) [| 32; 97; 32; 97 |] (ids tok text)

(* A byte without an id: the span goes back to OCaml, which drops it (no unknown
   token is configured). *)
let test_missing_byte_id () =
  let vocab = List.filter (fun (_, id) -> id <> Char.code 'q') base_vocab in
  let tok = Brot.bpe ~vocab ~merges:[] ~pre:(pre ()) () in
  equal ~msg:"ids" (array int)
    [| Char.code 'a'; Char.code 'b' |]
    (ids tok "aqb")

(* The merge whose result stands for other bytes than its entry's: bytes C4 and
   8A merge into the entry "Ċ" (bytes C4 8A), which decodes to "\n", so
   len_table gives its id 1 byte where the symbol covers 2. The kernel hands the
   span back and emit_word records the opaque run; the offsets must span the
   pretoken. The multi-token texts pin the C exactness check itself: only there
   does dropping it move an offset, since a span's single token takes the whole
   span whatever was recorded. *)
let test_len_table_disagreement () =
  let vocab = [ ("\196", 0); ("\138", 1); ("\196\138", 2); ("a", 3) ] in
  let tok = Brot.bpe ~vocab ~merges:[ ("\196", "\138") ] ~pre:(pre ()) () in
  let offsets text =
    Brot.Encoding.offsets (Brot.encode tok ~add_special_tokens:false text)
  in
  equal ~msg:"ids" (array int) [| 2 |] (ids tok "\196\138");
  equal ~msg:"offsets" (array (pair int int)) [| (0, 2) |] (offsets "\196\138");
  equal ~msg:"ids, token after" (array int) [| 2; 3 |] (ids tok "\196\138a");
  equal ~msg:"offsets, token after"
    (array (pair int int))
    [| (0, 2); (2, 3) |]
    (offsets "\196\138a");
  equal ~msg:"ids, disagreements mid-span" (array int) [| 3; 2; 2; 3 |]
    (ids tok "a\196\138\196\138a");
  equal ~msg:"offsets, disagreements mid-span"
    (array (pair int int))
    [| (0, 1); (1, 3); (3, 5); (5, 6) |]
    (offsets "a\196\138\196\138a")

(* Unicode classes are filled on demand: a code point first met inside the
   kernel is handed back, classified and re-entered. Exercise letters, digits
   and spaces beyond ASCII in one text. *)
let test_class_hand_back () =
  let tok = simple () in
  let text = " a\194\160\206\177\217\160 a" in
  equal ~msg:"ids" (array int)
    [| space_a; 194; 160; 206; 177; 217; 160; space_a |]
    (ids tok text)

(* The spans [tok] encodes [text] in, read back through offsets and word ids. *)
let kernel_spans tok text =
  let e = Brot.encode tok ~add_special_tokens:false text in
  let offsets = Brot.Encoding.offsets e in
  let words = Brot.Encoding.word_ids e in
  let n = Array.length offsets in
  let spans = ref [] in
  let i = ref 0 in
  while !i < n do
    let w = words.(!i) in
    let start = fst offsets.(!i) in
    let j = ref !i in
    while !j + 1 < n && words.(!j + 1) = w do
      incr j
    done;
    spans := (start, snd offsets.(!j)) :: !spans;
    i := !j + 1
  done;
  Array.of_list (List.rev !spans)

let pre_spans pre text =
  Array.of_list (List.map snd (Brot.Pre_tokenizer.pre_tokenize pre text))

(* A long pretoken planted at every span index from 600 to 1400, across the
   1024-span buffer of the recording path and the stage's seams, after three
   shapes of short ones: the spans must be the pre-tokenizer's. A stage that
   starts with no span room left harvests nothing. *)
let test_long_pretoken_at_buffer_seams () =
  let tok = Brot.bpe ~vocab:base_vocab ~merges:[] ~pre:(pre ()) () in
  let long = " " ^ String.make 20 'q' in
  let shapes =
    [
      ("plain", fun k -> String.concat "" (List.init k (fun _ -> " a")));
      ( "every third longer",
        fun k ->
          String.concat ""
            (List.init k (fun i -> if i mod 3 = 0 then " ab" else " a")) );
      ( "every fifth beyond ASCII",
        fun k ->
          String.concat ""
            (List.init k (fun i -> if i mod 5 = 0 then "\u{e9}" else " a")) );
    ]
  in
  List.iter
    (fun (name, shape) ->
      for k = 600 to 1400 do
        let text = shape k ^ long ^ shape 300 in
        equal
          ~msg:(Printf.sprintf "%s, long pretoken after %d" name k)
          (array (pair int int))
          (pre_spans (pre ()) text) (kernel_spans tok text)
      done)
    shapes

(* Each family's shape at every offset 0..80, encoded by [tok] and cut by
   [reference]. *)
let sweep ?(msg = "") tok reference families =
  List.iter
    (fun (name, shape) ->
      for off = 0 to 80 do
        let text = shape off in
        equal
          ~msg:(Printf.sprintf "%s%s at offset %d" msg name off)
          (array (pair int int))
          (pre_spans reference text) (kernel_spans tok text)
      done)
    families

let pad n = String.init n (fun i -> "ab ".[i mod 3])

(* The mask-scanner walker classifies 64-byte batches on a grid anchored at the
   range start. Each shape that can cross a batch edge — letter runs, a span of
   exactly 64 bytes, non-ASCII bad zones, contractions, whitespace give-backs
   (including one decided by a non-ASCII whitespace character), digit runs — is
   planted at every offset 0..80 and the resulting spans are held to the OCaml
   pre-tokenizer's, read back through offsets and word ids. The padding is short
   pretokens and the tails run at least 64 bytes past the shape, so the shape
   crosses streamed batch edges rather than re-anchoring the grid through an
   Encode hand-back or falling to the scalar tail; the 64-byte-span family keeps
   its long-run padding to exercise that re-anchor path. *)
let test_batch_alignments () =
  let tok = Brot.bpe ~vocab:base_vocab ~merges:[] ~pre:(pre ()) () in
  let families =
    [
      ("letters across the edge", fun off -> pad off ^ " " ^ String.make 70 'y');
      ( "a span of exactly 64 bytes",
        fun off -> String.make off 'z' ^ " " ^ String.make 63 'w' ^ " t" );
      ("non-ASCII", fun off -> pad off ^ "\195\169" ^ pad 80);
      ("contractions", fun off -> pad off ^ "'ll they've o'x " ^ pad 80);
      ("whitespace give-back", fun off -> pad off ^ "  \t\r\n  b " ^ pad 80);
      ( "give-back before non-ASCII whitespace",
        fun off -> String.make off 'x' ^ "  \194\160 " ^ String.make 80 'y' );
      ("digits", fun off -> pad off ^ " " ^ String.make 20 '8' ^ "a9 " ^ pad 64);
      ( "letters beyond ASCII across the edge",
        fun off ->
          pad off ^ " \195\169"
          ^ String.concat "" (List.init 30 (fun _ -> "Ж"))
          ^ "a日本" ^ pad 80 );
      ( "white space beyond ASCII at the edge",
        fun off ->
          String.make off 'x'
          ^ " \u{a0}y \u{3000}\u{3000}z\u{a0}\u{a0} 1\u{2028}"
          ^ String.make 80 'y' );
      ( "numbers and punctuation beyond ASCII",
        fun off ->
          pad off
          ^ " \u{663}\u{664}\u{bd} \u{201c}x\u{201d} \u{2014}\u{2014}😀a"
          ^ pad 80 );
      ( "malformed bytes",
        fun off ->
          pad off ^ "a\x80b \xC3 c\xE2\x80d\xF0\x9F\x98 \xC0\xAF\xED\xA0\x80"
          ^ pad 80 );
    ]
  in
  sweep tok (pre ()) families

(* The cl100k pattern, as Llama 3's tokenizer file spells it, and its variants
   with numbers one at a time and with the combining marks as letters. *)
let cl100k =
  {re|(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+|re}

let cl100k_single_digits =
  {re|(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+|re}

let cl100k_marks =
  {re|(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+|re}

let cl100k_patterns =
  [
    ("cl100k", cl100k);
    ("single digits", cl100k_single_digits);
    ("marks", cl100k_marks);
  ]

(* The pipeline a cl100k split serves: the split, then the byte-level encoding
   of its pieces. *)
let pre_split pattern =
  Brot.Pre_tokenizer.sequence
    [
      Brot.Pre_tokenizer.split_regex ~pattern ~behavior:`Isolated ();
      Brot.Pre_tokenizer.byte_level ~add_prefix_space:false ~use_regex:false ();
    ]

(* A pattern in a group is not recognised, so the regular expression engine cuts
   it rather than the walker. *)
let regex_split pattern =
  Brot.Pre_tokenizer.split_regex ~pattern:("(?:" ^ pattern ^ ")")
    ~behavior:`Isolated ()

(* The cl100k walker's batches, as the GPT-2 one's above, held to the regular
   expression: every rule the boundary masks evaluate on ASCII, each white space
   rule (a newline after punctuation, a newline then spaces, the run's last
   newline, a give-back), each prefix a word takes, digit groups, case-blind
   contractions, a run of white space crossing the batch edge, and non-ASCII
   characters that extend a digit or white space run, among them one that
   carries spaces after a newline on to more newlines, or close punctuation
   before newlines. *)
let test_cl100k_alignments () =
  let families =
    [
      ("letters across the edge", fun off -> pad off ^ " " ^ String.make 70 'y');
      ("non-ASCII", fun off -> pad off ^ "\195\169" ^ pad 80);
      ( "contractions",
        fun off -> pad off ^ "'ll THEY'VE o'x 'S's ''d x 'Re" ^ pad 80 );
      ( "newline runs",
        fun off -> pad off ^ " \n \n  x\n\n\ty\r\n  z " ^ pad 80 );
      ( "newlines after punctuation",
        fun off -> pad off ^ "!!\n\n x?\n \n!\r\n\t." ^ pad 80 );
      ("prefixes", fun off -> pad off ^ "\ta ?b  !c\t!d .e ..f\x0Bg" ^ pad 80);
      ( "digits",
        fun off -> pad off ^ " " ^ String.make 20 '8' ^ "a9 1234 12" ^ pad 64 );
      ( "white space across the edge",
        fun off -> pad off ^ "      \n     \n  y \t \n\t z" ^ pad 80 );
      ( "non-ASCII digits and white space",
        fun off ->
          pad off ^ "\217\16312345 \227\128\130\n\n  x\194\160\n y \194\160 !"
          ^ pad 80 );
      ( "give-back before non-ASCII white space",
        fun off -> String.make off 'x' ^ "  \194\160 " ^ String.make 80 'y' );
      ( "spaces after a newline, then non-ASCII white space and newlines",
        fun off -> pad off ^ "  \n  \194\160\n\n\n x" ^ pad 80 );
    ]
  in
  List.iter
    (fun (name, pattern) ->
      let tok =
        Brot.bpe ~vocab:base_vocab ~merges:[] ~pre:(pre_split pattern) ()
      in
      sweep ~msg:(name ^ ": ") tok (regex_split pattern) families)
    cl100k_patterns

(* The SentencePiece kernel: models with no pre-tokenizer hand the whole
   document to the model as one span, and the sub-cut walker owns any span past
   the 15-byte key. The mark is written literally in the texts (no normalizer),
   and every expectation pins ids the OCaml walker and the C walker must both
   produce; word ids pin that the fused path still records one span per
   document. *)

let sp_mark = "\xE2\x96\x81"

(* ▁ is 0, letters a..z are 1..26, "." 27, "," 28, "é" 29, "▁a" 30. *)
let sp_vocab =
  (sp_mark, 0)
  :: List.init 26 (fun i -> (String.make 1 (Char.chr (0x61 + i)), 1 + i))
  @ [ (".", 27); (",", 28); ("\195\169", 29); (sp_mark ^ "a", 30) ]

let sp_simple ?cache_capacity ?dropout () =
  Brot.bpe ~vocab:sp_vocab
    ~merges:[ (sp_mark, "a") ]
    ?cache_capacity ?dropout ()

(* A cache-hit stream: every unit is the seeded "▁a", and the whole document
   stays one span whether the C kernel or the OCaml walker cut it. *)
let test_sp_stream () =
  let tok = sp_simple () in
  let text = String.concat "" (List.init 1500 (fun _ -> sp_mark ^ "a")) in
  equal ~msg:"ids" (array int) (Array.make 1500 30) (ids tok text);
  let e = Brot.encode tok ~add_special_tokens:false text in
  equal ~msg:"offsets"
    (array (pair int int))
    (Array.init 1500 (fun i -> (4 * i, (4 * i) + 4)))
    (Brot.Encoding.offsets e);
  equal ~msg:"one span whatever cut it"
    (array (option int))
    (Array.make 1500 (Some 0)) (Brot.Encoding.word_ids e)

(* 100k units outgrow the per-call ids reserve: Ids_full parks the walk at a
   unit boundary and the next call resumes it. *)
let test_sp_ids_full () =
  let tok = sp_simple () in
  let text = String.concat "" (List.init 100_000 (fun _ -> sp_mark ^ "a")) in
  equal ~msg:"ids" (array int) (Array.make 100_000 30) (ids tok text)

(* A unit past the 15-byte key between short units: an [Encode] hand-back for
   length, encoded whole by OCaml. *)
let test_sp_long_unit () =
  let tok = sp_simple () in
  let text =
    String.concat ""
      (List.init 3 (fun _ -> sp_mark ^ "a")
      @ [ sp_mark; String.make 20 'b' ]
      @ List.init 3 (fun _ -> sp_mark ^ "a"))
  in
  let expect =
    Array.concat [ Array.make 3 30; [| 0 |]; Array.make 20 2; Array.make 3 30 ]
  in
  equal ~msg:"ids" (array int) expect (ids tok text)

(* A character with no piece and no unknown token: the unit hands back and OCaml
   drops the character. *)
let test_sp_missing_char () =
  let tok = sp_simple () in
  let text =
    sp_mark ^ "a" ^ sp_mark ^ "\195\188" ^ sp_mark ^ "a" ^ sp_mark ^ "a"
  in
  equal ~msg:"ids" (array int) [| 30; 0; 30; 30 |] (ids tok text)

(* The resume look-behind: the unit after a hand-back opens with a ▁ run, and
   the re-entered walk must keep the run whole — a wrong look-behind at the
   resume position would cut it and the ("▁", "▁") merge makes that cut visible
   in the ids. *)
let test_sp_resume_mark_run () =
  let vocab = sp_vocab @ [ (sp_mark ^ sp_mark, 31) ] in
  let tok = Brot.bpe ~vocab ~merges:[ (sp_mark, sp_mark); (sp_mark, "a") ] () in
  let text =
    sp_mark ^ "\195\188" ^ sp_mark ^ sp_mark ^ sp_mark ^ "a" ^ sp_mark ^ "a"
  in
  equal ~msg:"ids" (array int) [| 0; 31; 30; 30 |] (ids tok text)

(* Punctuation adjacency decides the split: with no piece holding "a.", the unit
   ends before the dot; a vocabulary that holds "a." (and merges it) keeps the
   dot in the unit. *)
let test_sp_punct_adjacency () =
  let text = sp_mark ^ "a.b" ^ sp_mark ^ "c" ^ sp_mark ^ "a" ^ sp_mark ^ "a" in
  let split = Brot.bpe ~vocab:sp_vocab ~merges:[] () in
  equal ~msg:"safe dot splits" (array int)
    [| 0; 1; 27; 2; 0; 3; 0; 1; 0; 1 |]
    (ids split text);
  let joined =
    Brot.bpe ~vocab:(sp_vocab @ [ ("a.", 31) ]) ~merges:[ ("a", ".") ] ()
  in
  equal ~msg:"a vocabulary piece across the dot keeps it" (array int)
    [| 0; 31; 2; 0; 3; 0; 1; 0; 1 |]
    (ids joined text)

(* Byte fallback: the first ü hands back and OCaml spells it as <0xC3> <0xBC> —
   an exact, cacheable answer — so the second is served by the kernel from the
   cache. Both must come out the same. *)
let test_sp_fallback_cached () =
  let fallback =
    List.init 256 (fun b -> (Printf.sprintf "<0x%02X>" b, 40 + b))
  in
  let tok =
    Brot.bpe ~vocab:(sp_vocab @ fallback)
      ~merges:[ (sp_mark, "a") ]
      ~byte_fallback:true ()
  in
  let unit = sp_mark ^ "\195\188" in
  let text = sp_mark ^ "a" ^ unit ^ sp_mark ^ "a" ^ unit in
  equal ~msg:"ids" (array int)
    [| 30; 0; 40 + 0xC3; 40 + 0xBC; 30; 0; 40 + 0xC3; 40 + 0xBC |]
    (ids tok text)

(* The merge chain spelling out <0x41>, whose len_table entry reads 1 where the
   merged symbol covers 6 bytes: the kernel refuses the unit, OCaml records the
   opaque run, and the offsets must tile the document. *)
let test_sp_len_table_disagreement () =
  let spell =
    [
      ("<", 31);
      ("0", 32);
      ("4", 33);
      ("1", 34);
      (">", 35);
      ("<0", 36);
      ("<0x", 37);
      ("<0x4", 38);
      ("1>", 39);
      ("<0x41>", 40);
    ]
  in
  let tok =
    Brot.bpe ~vocab:(sp_vocab @ spell)
      ~merges:
        [
          (sp_mark, "a");
          ("<", "0");
          ("<0", "x");
          ("<0x", "4");
          ("1", ">");
          ("<0x4", "1>");
        ]
      ~byte_fallback:true ()
  in
  let text = sp_mark ^ "a<0x41>" ^ sp_mark ^ "b" ^ sp_mark ^ "c" in
  equal ~msg:"ids" (array int) [| 30; 40; 0; 2; 0; 3 |] (ids tok text);
  let e = Brot.encode tok ~add_special_tokens:false text in
  equal ~msg:"offsets"
    (array (pair int int))
    [| (0, 4); (4, 10); (10, 13); (13, 14); (14, 17); (17, 18) |]
    (Brot.Encoding.offsets e)

(* Caching off still cuts and merges in C: the units are unkeyed and the probes
   skipped. *)
let test_sp_cache_off () =
  let tok = sp_simple ~cache_capacity:0 () in
  let text = String.concat "" (List.init 600 (fun _ -> sp_mark ^ "a")) in
  equal ~msg:"ids" (array int) (Array.make 600 30) (ids tok text)

(* Dropout leaves the sub-cut off at creation; at probability one every merge is
   skipped, deterministically, through the whole-document path. *)
let test_sp_dropout_not_selected () =
  let tok = sp_simple ~dropout:1.0 () in
  let text = String.concat "" (List.init 8 (fun _ -> sp_mark ^ "a")) in
  equal ~msg:"ids" (array int)
    (Array.concat (List.init 8 (fun _ -> [| 0; 1 |])))
    (ids tok text)

(* The batch sink. [encode_batch_ids] writes the kernels' ids straight into a
   chunk's int32 buffer, while added-token literals and hand-backs go through an
   OCaml spill flushed before every kernel call and at document end. Each
   document interleaves the three sources, so a flush out of order or a lost
   spill shows up as a permuted row; every row must be what [encode_ids] gives
   for that text alone. *)

let batch_rows tok docs exec =
  let buf, lengths =
    Brot.encode_batch_ids tok ~add_special_tokens:false ?exec docs
  in
  let pos = ref 0 in
  Array.to_list lengths
  |> List.map (fun n ->
      let row =
        Array.init n (fun k ->
            Int32.to_int (Bigarray.Array1.get buf (!pos + k)))
      in
      pos := !pos + n;
      row)

let check_batch tok docs =
  List.iter
    (fun (on, exec) ->
      List.iteri
        (fun i (row, doc) ->
          equal
            ~msg:(Printf.sprintf "row %d on %s" i on)
            (array int) (ids tok doc) row)
        (List.combine (batch_rows tok docs exec) docs))
    [ ("the calling domain", None); ("four domains", Some (Fixture.spawn 4)) ]

(* Byte-level: cache-hit runs (kernel, sink), <sep> literals and a 20-byte
   pretoken (OCaml, spill) interleaved; enough ids that the sink outgrows its
   bytes/3 sizing, and a 70k-letter span whose ids-room check grows the sink at
   a stuck resume before handing the span back. *)
let test_batch_sink_spill_order () =
  let vocab = base_vocab @ [ ("\196\160a", space_a) ] in
  let tok =
    Brot.bpe ~vocab
      ~merges:[ ("\196\160", "a") ]
      ~pre:(pre ())
      ~added_tokens:[ Brot.added_token "<sep>" ]
      ()
  in
  let hits k = String.concat "" (List.init k (fun _ -> " a")) in
  let long = " " ^ String.make 20 'b' in
  let docs =
    [
      hits 3000 ^ "<sep>" ^ long ^ "<sep>" ^ hits 3000;
      "<sep>";
      "";
      hits 5 ^ long;
      " " ^ String.make 70_000 'x';
    ]
  in
  check_batch tok docs

(* SentencePiece: "▁a" streams (kernel, sink) around a 21-byte unit and <sep>
   literals (spill), one document ending on a hand-back so the last flush is the
   document-end one. *)
let test_sp_batch_sink_spill_order () =
  let tok =
    Brot.bpe ~vocab:sp_vocab
      ~merges:[ (sp_mark, "a") ]
      ~added_tokens:[ Brot.added_token "<sep>" ]
      ()
  in
  let units k = String.concat "" (List.init k (fun _ -> sp_mark ^ "a")) in
  let long = sp_mark ^ String.make 20 'b' in
  let docs =
    [ units 3000 ^ "<sep>" ^ long ^ units 3000; "<sep>"; units 2 ^ long ]
  in
  check_batch tok docs

(* Laws against a reference BPE: drawn text encodes to what the merge rules
   give when applied the simplest way — repeatedly merge the leftmost
   occurrence of the adjacent pair of lowest rank — over GPT-2's pretokens
   and over Mistral's whole ▁-marked text. The reference reads the vocabulary
   and merges from the tokenizer file and cuts GPT-2's pretokens with its
   regular expression, so it shares neither the walkers, the unit cut, the
   cache nor the merge loops with the library. The single-text and batch
   paths each run against it. *)

let json_member name = function
  | Jsont.Object (members, _) -> (
      match Jsont.Json.find_mem name members with
      | Some (_, v) -> v
      | None -> failf "no member %s" name)
  | _ -> failf "%s: not an object" name

type reference = {
  vocab : (string, int) Hashtbl.t;
  ranks : (string * string, int) Hashtbl.t;
}

let reference_of path =
  let text = Fixture.read path in
  let json =
    match Jsont_bytesrw.decode_string Jsont.json text with
    | Ok json -> json
    | Error e -> failf "%s: %s" path e
  in
  let model = json_member "model" json in
  let vocab = Hashtbl.create 65536 in
  (match json_member "vocab" model with
  | Jsont.Object (members, _) ->
      List.iter
        (fun ((name, _), v) ->
          match v with
          | Jsont.Number (id, _) -> Hashtbl.replace vocab name (int_of_float id)
          | _ -> failf "vocab %s: not a number" name)
        members
  | _ -> failf "vocab: not an object");
  let ranks = Hashtbl.create 65536 in
  (match json_member "merges" model with
  | Jsont.Array (merges, _) ->
      List.iteri
        (fun rank m ->
          let pair =
            match m with
            | Jsont.String (m, _) -> (
                match String.index_opt m ' ' with
                | Some i ->
                    ( String.sub m 0 i,
                      String.sub m (i + 1) (String.length m - i - 1) )
                | None -> failf "merge %S" m)
            | Jsont.Array ([ Jsont.String (a, _); Jsont.String (b, _) ], _) ->
                (a, b)
            | _ -> failf "merge %d" rank
          in
          if not (Hashtbl.mem ranks pair) then Hashtbl.add ranks pair rank)
        merges
  | _ -> failf "merges: not an array");
  { vocab; ranks }

let rec merge_all r symbols =
  let best = ref None in
  let rec scan i = function
    | a :: (b :: _ as rest) ->
        (match (Hashtbl.find_opt r.ranks (a, b), !best) with
        | Some rank, Some (best_rank, _) when rank >= best_rank -> ()
        | Some rank, _ -> best := Some (rank, i)
        | None, _ -> ());
        scan (i + 1) rest
    | _ -> ()
  in
  scan 0 symbols;
  match !best with
  | None -> symbols
  | Some (_, at) ->
      let rec join i = function
        | a :: b :: rest when i = at -> (a ^ b) :: rest
        | a :: rest -> a :: join (i + 1) rest
        | [] -> []
      in
      merge_all r (join 0 symbols)

let chars s =
  let rec go i acc =
    if i >= String.length s then List.rev acc
    else
      let n = Uchar.utf_decode_length (String.get_utf_8_uchar s i) in
      go (i + n) (String.sub s i n :: acc)
  in
  go 0 []

let id_of r symbol =
  match Hashtbl.find_opt r.vocab symbol with
  | Some id -> id
  | None -> failf "no id for %S" symbol

(* GPT-2 merges the byte-level spelling of each pretoken of its pattern. *)
let gpt2_pattern =
  {|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+|}

let gpt2_reference r =
  let pre =
    Brot.Pre_tokenizer.split_regex ~pattern:gpt2_pattern ~behavior:`Isolated ()
  in
  fun text ->
    Brot.Pre_tokenizer.pre_tokenize pre text
    |> List.concat_map (fun (piece, _) ->
        List.init (String.length piece) (fun i ->
            byte_alphabet.(Char.code piece.[i]))
        |> List.filter (Hashtbl.mem r.vocab)
        |> merge_all r |> List.map (id_of r))
    |> Array.of_list

(* Mistral marks spaces with ▁, prepends one to a text that does not open
   with it, and merges the whole text from its characters. A character
   without a piece is spelled by its bytes' fallback tokens when there is
   one for each, and is otherwise held back as the unknown token, which a
   run of such characters shares and which goes out at the next piece or at
   the end: a fallback spelling between overtakes it. *)
let mistral_reference r text =
  let marked = String.concat sp_mark (String.split_on_char ' ' text) in
  let marked =
    if String.starts_with ~prefix:sp_mark marked then marked
    else sp_mark ^ marked
  in
  let rec init pending = function
    | [] -> if pending then [ "<unk>" ] else []
    | c :: rest when Hashtbl.mem r.vocab c ->
        (if pending then [ "<unk>"; c ] else [ c ]) @ init false rest
    | c :: rest ->
        let bytes =
          List.init (String.length c) (fun i ->
              Printf.sprintf "<0x%02X>" (Char.code c.[i]))
        in
        if List.for_all (Hashtbl.mem r.vocab) bytes then
          bytes @ init pending rest
        else init true rest
  in
  if text = "" then [||]
  else
    init false (chars marked)
    |> merge_all r |> List.map (id_of r) |> Array.of_list

(* Words of every length up to past the 15-byte key, repeated characters,
   contractions, digits, the punctuation SentencePiece cuts at, every kind
   of white space, letters beyond ASCII, characters no trimmed vocabulary
   holds, and ▁ itself. *)
let law_alphabet =
  [
    "the"; "and"; "of"; "tokenizer"; "information"; "internationalization";
    "Hello"; "WORLD"; "aaaaaaaaaaaaaaaaaaaaaaaa"; "!!!!!!!!"; "don't";
    "it's"; "we'll"; "they're"; "I'm"; "'"; "1"; "2024"; "3.14159265358979";
    "."; ","; "\""; ")"; ";"; ":"; "!"; "?"; "("; "-"; "..."; " "; " ";
    " "; "  "; "\n"; "\n\n"; "\t"; " \r\n"; "café"; "naïve"; "Ж"; "Привет";
    "日本語"; "😀"; "e\u{0301}"; "ß"; "\u{2581}"; "\u{2581}\u{2581}";
  ]
[@@ocamlformat "disable"]

let text_of ~size alphabet =
  Gen.with_pp
    (fun ppf s -> Format.fprintf ppf "%S" s)
    (Gen.map (String.concat "")
       (Gen.list ~size:(Gen.int_range 0 size) (Gen.of_list alphabet)))

let law_text = text_of ~size:40 law_alphabet

(* The cl100k pattern's own cases on top: contractions in any case, digit
   groups, every white space rule, punctuation before newlines, the prefixes a
   word takes, and characters beyond ASCII that join digits, white space and
   punctuation; texts long enough to cross several 64-byte batches. *)
let cl100k_alphabet =
  law_alphabet
  @ [
      "'S"; "'LL"; "'Re"; "'VE"; "'D"; "''"; "12"; "1234567"; "\r\n"; " \n ";
      "  \n"; "\n\n  "; "\t?"; "?!"; ".\n\n"; "\x0B"; "\u{a0}"; "\u{2019}";
      "\u{663}"; "\u{3002}"; "\u{3000}"; "a\u{301}";
    ]
[@@ocamlformat "disable"]

let cl100k_text = text_of ~size:80 cl100k_alphabet

let has_sub text sub =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length text && (String.sub text i n = sub || go (i + 1))
  in
  go 0

(* The batch walk against the scalar walker it stands for: drawn texts that mix
   letters, numbers, white space and other characters of every length beyond
   ASCII with ASCII runs, contractions, malformed bytes and digits, long enough
   to cross several 64-byte batches, cut by the C kernel as the OCaml
   pre-tokenizer cuts them. *)
let walker_alphabet =
  [
    "a"; "the"; "Ab"; " "; "  "; "\n"; "\t"; "\r\n"; "'s"; "'ll"; "'"; "1";
    "42"; "."; "!!"; "-"; "\195\169"; "Ж"; "ß"; "日本"; "\u{a0}"; "\u{85}";
    "\u{3000}"; "\u{2028}"; "\u{663}"; "\u{bd}"; "\u{201c}"; "\u{201d}";
    "\u{2014}"; "😀"; "e\u{301}"; "\x80"; "\xC3"; "\xE2\x80"; "\xF0\x9F";
    "\xC0\xAF"; "\xED\xA0\x80";
    (* Overlong forms of every ASCII class, of three bytes and of four: a
       letter, a digit, a space, a newline, punctuation, an apostrophe. *)
    "\xC1\x81"; "\xE0\x81\x81"; "\xF0\x80\x81\x81"; "\xE0\x80\xB1";
    "\xF0\x80\x80\xB1"; "\xE0\x80\xA0"; "\xF0\x80\x80\xA0"; "\xE0\x80\x8A";
    "\xF0\x80\x80\x8A"; "\xE0\x80\xAE"; "\xF0\x80\x80\xAE"; "\xE0\x80\xA7";
    (* Letters and numbers of four bytes, code points no table classifies
       before they are met, and bytes that never stand in UTF-8. *)
    "\u{1D400}"; "\u{10400}"; "\u{1D7CE}"; "\u{10107}"; "\u{378}"; "\u{E0001}";
    "\u{10FFFF}"; "\u{FFFE}"; "\xF5"; "\xF8"; "\xFE"; "\xFF";
  ]
[@@ocamlformat "disable"]

let walker_text =
  Gen.with_pp
    (fun ppf s -> Format.fprintf ppf "%S" s)
    (Gen.map (String.concat "")
       (Gen.list ~size:(Gen.int_range 40 160) (Gen.of_list walker_alphabet)))

let has_any text subs = List.exists (has_sub text) subs

(* A white space character in an overlong form of four bytes, which the scalar
   walker decodes as the space it stands for, inside a 64-byte batch. *)
let test_overlong_white_space () =
  let tok = Brot.bpe ~vocab:base_vocab ~merges:[] ~pre:(pre ()) () in
  List.iter
    (fun text ->
      equal ~msg:(Printf.sprintf "%S" text)
        (array (pair int int))
        (pre_spans (pre ()) text) (kernel_spans tok text))
    [
      "abcdefghij\xF0\x80\x80\xA0" ^ String.make 60 'x';
      "abcdefghij \xF0\x80\x80\xA0\xF0\x80\x80\x8A" ^ String.make 60 'x';
      "abcdefghij\xE0\x80\xA0\xF0\x80\x80\xA0y" ^ String.make 60 'x';
    ]

let test_walker_law =
  let tok = lazy (Brot.bpe ~vocab:base_vocab ~merges:[] ~pre:(pre ()) ()) in
  prop ~count:300 "the batch walk cuts as the scalar walker" walker_text
    (fun text ->
      cover "white space beyond ASCII before a letter"
        (has_any text [ "\u{a0}a"; "\u{3000}t"; "\u{2028}A" ]);
      cover "a letter beyond ASCII after a space"
        (has_any text [ " \195\169"; " Ж"; " 日" ]);
      cover "malformed UTF-8" (not (String.is_valid_utf_8 text));
      cover "an overlong white space character"
        (has_any text [ "\xE0\x80\xA0"; "\xF0\x80\x80\xA0" ]);
      cover "a text over two batches" (String.length text > 128);
      equal ~msg:(Printf.sprintf "%S" text)
        (array (pair int int))
        (pre_spans (pre ()) text)
        (kernel_spans (Lazy.force tok) text))

let cover_law text =
  let words = String.split_on_char ' ' text in
  cover "a word over 15 bytes"
    (List.exists (fun w -> String.length w > 15) words);
  cover "a run of one character" (has_sub text "aaaaaaaa");
  cover "a contraction" (has_sub text "'s" || has_sub text "'ll");
  cover "a character beyond ASCII" (String.exists (fun c -> c >= '\x80') text);
  cover "a character no piece holds" (has_sub text "😀");
  cover "a ▁ in the text" (has_sub text sp_mark);
  cover "a split punctuation byte" (has_sub text "." || has_sub text "?");
  cover "white space other than a space"
    (String.exists (fun c -> c = '\n' || c = '\t') text)

(* Llama 3, Qwen 2.5 and Phi-4 merge the byte-level spelling of each piece of
   their cl100k pattern, cut by the regular expression engine; under
   ignore_merges a piece that is itself in the vocabulary is that entry. Qwen
   2.5 takes numbers one at a time and composes its text to NFC first. *)
let cl100k_reference ?(pattern = cl100k) ?(normalize = Fun.id) ~ignore_merges r
    =
  let pre = regex_split pattern in
  fun text ->
    Brot.Pre_tokenizer.pre_tokenize pre (normalize text)
    |> List.concat_map (fun (piece, _) ->
        let symbols =
          List.init (String.length piece) (fun i ->
              byte_alphabet.(Char.code piece.[i]))
        in
        match Hashtbl.find_opt r.vocab (String.concat "" symbols) with
        | Some id when ignore_merges -> [ id ]
        | _ ->
            List.filter (Hashtbl.mem r.vocab) symbols
            |> merge_all r |> List.map (id_of r))
    |> Array.of_list

let cover_cl100k text =
  cover "a contraction in upper case" (has_sub text "'S" || has_sub text "'LL");
  cover "a group of three digits" (has_sub text "1234567");
  cover "a newline then spaces" (has_sub text "\n ");
  cover "punctuation before a newline" (has_sub text ".\n");
  cover "a text over two batches" (String.length text > 128)

(* A SentencePiece pipeline writes ▁ for every space right before the model, in
   its normalizer or in a metaspace that does not split; a BPE model then reads
   the spaces themselves, so the text it is handed keeps them. The same
   pipeline with a stage that changes nothing after the marking (a strip of
   neither side, a sequence of the one metaspace) writes the same text and is
   read marked, and both must give the same tokens, offsets included. The
   vocabulary merges across ▁ runs, spells "ü" with fallback bytes and has no
   piece for "😀"; one added token is matched in the raw text and one in the
   normalized text. *)
let spaced_vocab =
  sp_vocab
  @ [
      (sp_mark ^ sp_mark, 31); (sp_mark ^ "t", 32); ("he", 33);
      (sp_mark ^ "the", 34); ("<0xC3>", 35); ("<0xBC>", 36); ("<unk>", 37);
      ("<s>", 38);
    ]
[@@ocamlformat "disable"]

let spaced_merges =
  [ (sp_mark, "a"); (sp_mark, sp_mark); (sp_mark, "t"); ("h", "e");
    (sp_mark ^ "t", "he") ]
[@@ocamlformat "disable"]

let spaced_alphabet =
  [ "a"; "the"; "q q"; "b"; " "; "  "; "   "; sp_mark; sp_mark ^ sp_mark;
    " " ^ sp_mark; "."; ","; "a.b"; "959"; "9588"; "😀"; "
";
    "<s>"; "aaaaaaaaaaaaaaaaaaaa"; " xxxxxxxxxxxxxxx" ]
[@@ocamlformat "disable"]

let spaced_case =
  let open Gen in
  let+ text =
    map (String.concat "")
      (list ~size:(int_range 0 60) (of_list spaced_alphabet))
  and+ pipeline = int_range 0 3
  and+ byte_fallback = bool in
  (text, pipeline, byte_fallback)

let spaced_case =
  let pp ppf (text, pipeline, byte_fallback) =
    Format.fprintf ppf "%S, pipeline %d, byte_fallback %b" text pipeline
      byte_fallback
  in
  Gen.with_pp pp spaced_case

(* The spaced pipeline and its marked twin: the normalizer Llama spells, then a
   metaspace prepending on [`First], on [`Always] and never. *)
let spaced_pair pipeline ~byte_fallback =
  let make ?normalizer ?pre () =
    Brot.bpe ?normalizer ?pre ~vocab:spaced_vocab ~merges:spaced_merges
      ~unk_token:"<unk>" ~byte_fallback ~fuse_unk:true
      ~added_tokens:
        [
          Brot.added_token ~special:true "<s>";
          Brot.added_token ~special:false ~normalized:true "q q";
        ]
      ()
  in
  let marking =
    [
      Brot.Normalizer.prepend sp_mark;
      Brot.Normalizer.replace ~pattern:" " ~replacement:sp_mark;
    ]
  in
  let noop = Brot.Normalizer.strip ~left:false ~right:false () in
  let metaspace prepend_scheme =
    Brot.Pre_tokenizer.metaspace ~prepend_scheme ~split:false ()
  in
  match pipeline with
  | 0 ->
      ( make ~normalizer:(Brot.Normalizer.sequence marking) (),
        make ~normalizer:(Brot.Normalizer.sequence (marking @ [ noop ])) () )
  | n ->
      let scheme = List.nth [ `First; `Always; `Never ] (n - 1) in
      ( make ~pre:(metaspace scheme) (),
        make ~pre:(Brot.Pre_tokenizer.sequence [ metaspace scheme ]) () )

let spaced_law =
  prop ~count:500 "a spaced pipeline encodes as a marked one" spaced_case
    (fun (text, pipeline, byte_fallback) ->
      let spaced, marked = spaced_pair pipeline ~byte_fallback in
      cover "a space run" (has_sub text "  ");
      cover "a ▁ next to a space" (has_sub text (" " ^ sp_mark));
      cover "a unit past the key" (has_sub text "aaaaaaaaaaaaaaaaaaaa");
      cover "a character with fallback bytes" (has_sub text "9588");
      cover "an added token" (has_sub text "<s>" || has_sub text "q q");
      cover "a text that opens with a space"
        (String.length text > 0 && text.[0] = ' ');
      let encode t = Brot.encode t ~add_special_tokens:false text in
      let a = encode spaced and b = encode marked in
      let msg what = Printf.sprintf "%S: %s" text what in
      equal ~msg:(msg "ids") (array int) (Brot.Encoding.ids b)
        (Brot.Encoding.ids a);
      equal ~msg:(msg "tokens") (array string) (Brot.Encoding.tokens b)
        (Brot.Encoding.tokens a);
      equal ~msg:(msg "offsets")
        (array (pair int int))
        (Brot.Encoding.offsets b) (Brot.Encoding.offsets a);
      match batch_rows spaced [ text; text ] None with
      | [ x; y ] ->
          equal ~msg:(msg "in a batch") (array int) (Brot.Encoding.ids b) x;
          equal ~msg:(msg "again in a batch") (array int) (Brot.Encoding.ids b)
            y
      | _ -> failf "two rows expected")

(* A cache starts at 512 entries and doubles while the kernels fill it, from
   their stores and from OCaml's hand-backs, moving under the kernels' tables
   between calls. Thousands of distinct words of one to six letters, which the
   letter pairs merge to at most four ids and so are kept, encoded three times
   over, must still give the ids an uncached tokenizer gives: the byte-level
   kernel on words with a space before them, the SentencePiece one on ▁ units,
   with some units past the 15-byte key among them. *)
let grown_letters = List.init 6 (fun i -> String.make 1 (Char.chr (97 + i)))

let grown_pairs =
  List.concat_map (fun a -> List.map (fun b -> (a, b)) grown_letters)
    grown_letters

let grown_vocab first extra =
  first
  @ List.mapi
      (fun i (a, b) -> (a ^ b, List.length first + i))
      (extra @ grown_pairs)

let grown_byte_level cache_capacity =
  let space = "\196\160" in
  let extra = List.map (fun l -> (space, l)) grown_letters in
  Brot.bpe
    ~vocab:(grown_vocab base_vocab extra)
    ~merges:(grown_pairs @ extra) ~pre:(pre ()) ?cache_capacity ()

let grown_sp cache_capacity =
  Brot.bpe ~vocab:(grown_vocab sp_vocab []) ~merges:grown_pairs ?cache_capacity
    ()

let grown_case =
  let open Gen in
  let word size = string_of ~size (char_range 'a' 'f') in
  with_pp
    (fun ppf words -> Format.fprintf ppf "%d words" (List.length words))
    (list ~size:(int_range 2000 4000)
       (frequency [ (20, word (int_range 1 6)); (1, word (constant 18)) ]))

let grown_law =
  prop ~count:10 "a cache grown by the kernels finds what it held" grown_case
    (fun words ->
      let check ~msg make text =
        let cached = make None and uncached = make (Some 0) in
        let expect = ids uncached text in
        for round = 1 to 3 do
          equal
            ~msg:(Printf.sprintf "%s, round %d" msg round)
            (array int) expect (ids cached text)
        done
      in
      check ~msg:"byte-level" grown_byte_level (String.concat " " words);
      check ~msg:"sentencepiece" grown_sp
        (String.concat "" (List.map (( ^ ) sp_mark) words)))

(* The tokenizer and the reference are read once, on the first case. *)
let law ?(text = law_text) ?(cover = cover_law) name make =
  let loaded =
    lazy
      (Fixture.with_tokenizer_file name (fun path ->
           match Brot.from_file path with
           | Ok tok -> (tok, make (reference_of path))
           | Error e -> failf "%s: %s" path e))
  in
  prop ~count:300
    (Printf.sprintf "%s encodes as the reference merge" name)
    text
    (fun text ->
      let tok, reference = Lazy.force loaded in
      let expect = reference text in
      cover text;
      equal ~msg:(Printf.sprintf "%S alone" text) (array int) expect
        (ids tok text);
      match batch_rows tok [ text; text ] None with
      | [ a; b ] ->
          equal ~msg:(Printf.sprintf "%S in a batch" text) (array int) expect a;
          equal
            ~msg:(Printf.sprintf "%S again in a batch" text)
            (array int) expect b
      | _ -> failf "two rows expected")

let () =
  exit
    (run "brot kernels"
       [
         group "resume protocol"
           [
             test "a document crossing the span chunk" test_chunk_boundary;
             test "pretoken counts at the chunk seam" test_chunk_seam_lengths;
             test "a long pretoken hands back between chunks"
               test_long_pretoken_hand_back;
             test "hand-backs at the chunk's edges" test_encode_at_chunk_edges;
             test "a long pretoken at every span of the buffer's seams"
               test_long_pretoken_at_buffer_seams;
             test "a class hand-back after an encode resume"
               test_class_after_encode_resume;
             test "maximal multi-id spans across chunks" test_many_ids_per_span;
             test "a span larger than the ids buffer" test_huge_span;
           ];
         group "batch walker"
           [
             test "every shape at every batch alignment" test_batch_alignments;
             test "white space in an overlong form of four bytes"
               test_overlong_white_space;
             test_walker_law;
             test "every cl100k shape at every batch alignment"
               test_cl100k_alignments;
           ];
         group "selection and refusals"
           [
             test "caching off" test_cache_off;
             test "ignore_merges" test_ignore_merges;
             test "dropout leaves the kernel unselected"
               test_dropout_not_selected;
             test "a byte without an id" test_missing_byte_id;
             test "a merge result len_table disagrees with"
               test_len_table_disagreement;
             test "unclassified code points" test_class_hand_back;
           ];
         group "sentencepiece resume protocol"
           [
             test "a cache-hit stream stays one span" test_sp_stream;
             test "a stream outgrowing the ids reserve" test_sp_ids_full;
             test "a unit past the key hands back" test_sp_long_unit;
             test "a character without a piece" test_sp_missing_char;
             test "a mark run right after a resume" test_sp_resume_mark_run;
             test "punctuation adjacency" test_sp_punct_adjacency;
             test "byte fallback is cached and then served"
               test_sp_fallback_cached;
             test "a merge result len_table disagrees with"
               test_sp_len_table_disagreement;
             test "caching off" test_sp_cache_off;
             test "dropout leaves the sub-cut off" test_sp_dropout_not_selected;
           ];
         group "laws"
           [
             spaced_law;
             grown_law;
             law "gpt2" gpt2_reference;
             law "mistral" mistral_reference;
             law ~text:cl100k_text ~cover:cover_cl100k "llama3"
               (cl100k_reference ~ignore_merges:true);
             law ~text:cl100k_text ~cover:cover_cl100k "qwen2_5"
               (cl100k_reference ~pattern:cl100k_single_digits
                  ~normalize:(Brot.Normalizer.apply Brot.Normalizer.nfc)
                  ~ignore_merges:false);
             (* Phi-4 keeps its split's matches rather than isolating them, a
                pipeline the kernel walkers do not take. *)
             law ~text:cl100k_text ~cover:cover_cl100k "phi4"
               (cl100k_reference ~ignore_merges:false);
           ];
         group "batch sink"
           [
             test "literals and hand-backs spill in order"
               test_batch_sink_spill_order;
             test "sentencepiece literals and hand-backs spill in order"
               test_sp_batch_sink_spill_order;
           ];
       ])
