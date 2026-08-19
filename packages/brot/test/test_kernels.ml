(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The fused kernel's resume protocol, observed through the public API. The
   native binary runs the C kernel and the byte_complete one the OCaml
   reference, so every expectation here is checked against both; the dump
   differential separately holds the two identical on adversarial input. Each
   case drives one hand-back or refusal: a document crossing the span chunk,
   maximal multi-id spans, caching off, [ignore_merges], dropout (kernel not
   selected), pretokens past the 15-byte key, bytes without an id, and a merge
   whose result stands for other bytes than its entry's. *)

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

(* 1500 pretokens " a": crosses the 1024-span chunk, every span a cache hit of
   one id. *)
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

let test_cache_off () =
  let tok = simple ~cache_capacity:0 () in
  let text = String.concat "" (List.init 100 (fun _ -> " a")) in
  equal ~msg:"ids" (array int) (Array.make 100 space_a) (ids tok text)

(* Under ignore_merges the kernel may not merge: misses hand back, the
   whole-word lookup answers for vocabulary words and the merges for the
   rest. *)
let test_ignore_merges () =
  let tok = simple ~ignore_merges:true () in
  let text = String.concat "" (List.init 100 (fun _ -> " a")) in
  equal ~msg:"ids" (array int) (Array.make 100 space_a) (ids tok text);
  equal ~msg:"a word not in the vocabulary still merges" (array int)
    [| space_a; Char.code 'b' |]
    (ids tok " ab")

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

let () =
  run "brot kernels"
    [
      group "resume protocol"
        [
          test "a document crossing the span chunk" test_chunk_boundary;
          test "a long pretoken hands back between chunks"
            test_long_pretoken_hand_back;
          test "maximal multi-id spans across chunks" test_many_ids_per_span;
          test "a span larger than the ids buffer" test_huge_span;
        ];
      group "selection and refusals"
        [
          test "caching off" test_cache_off;
          test "ignore_merges" test_ignore_merges;
          test "dropout leaves the kernel unselected" test_dropout_not_selected;
          test "a byte without an id" test_missing_byte_id;
          test "a merge result len_table disagrees with"
            test_len_table_disagreement;
          test "unclassified code points" test_class_hand_back;
        ];
    ]
