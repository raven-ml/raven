(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Native-vs-bytecode differential for the C kernels. The native binary encodes
   through them, the byte_complete one through the OCaml reference that
   [Sys.backend_type] selects; both print this deterministic dump and the build
   diffs the two. Inputs cover every kernel exit: unclassified code points,
   malformed UTF-8 of every shape, pretokens at and past the 15-byte key limit,
   huge spans, more pretokens than a span chunk, every exit at the staged
   chunk's seams, bytes without an id, [ignore_merges], caching off, and a merge
   whose result stands for other bytes than its entry's.

   An optional corpus path argument appends [encode_batch_ids] digests over that
   file split on <|endoftext|>, for dev-time runs over a large corpus. *)

let pf = Printf.printf

(* Deterministic PRNG (splitmix64), identical in both binaries. *)
let rng_state = ref 0x9E3779B97F4A7C15L

let rand64 () =
  let z = Int64.add !rng_state 0x9E3779B97F4A7C15L in
  rng_state := z;
  let z =
    Int64.mul
      (Int64.logxor z (Int64.shift_right_logical z 30))
      0xBF58476D1CE4E5B9L
  in
  let z =
    Int64.mul
      (Int64.logxor z (Int64.shift_right_logical z 27))
      0x94D049BB133111EBL
  in
  Int64.logxor z (Int64.shift_right_logical z 31)

let rand_int bound =
  Int64.to_int
    (Int64.rem
       (Int64.logand (rand64 ()) 0x7FFFFFFFFFFFFFFFL)
       (Int64.of_int bound))

(* Files, resolved as the parity suite resolves them. *)

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let read_opt path = if Sys.file_exists path then Some (read_file path) else None

(* Output: full values for short inputs, a digest above that, so the dump stays
   small while covering megabytes. *)

let ints_line ints =
  String.concat " " (List.map string_of_int (Array.to_list ints))

let show name line =
  if String.length line <= 2048 then pf "%s: %s\n" name line
  else
    pf "%s: len=%d md5=%s\n" name (String.length line)
      (Digest.to_hex (Digest.string line))

let dump_ids tok name text =
  show (name ^ "/ids")
    (ints_line (Brot.encode_ids tok ~add_special_tokens:false text))

let dump_encoding tok name text =
  let e = Brot.encode tok ~add_special_tokens:false text in
  show (name ^ "/e.ids") (ints_line (Brot.Encoding.ids e));
  show (name ^ "/e.offsets")
    (String.concat " "
       (List.map
          (fun (a, b) -> Printf.sprintf "%d-%d" a b)
          (Array.to_list (Brot.Encoding.offsets e))));
  show (name ^ "/e.words")
    (String.concat " "
       (List.map
          (function None -> "_" | Some w -> string_of_int w)
          (Array.to_list (Brot.Encoding.word_ids e))))

let digest_line ints =
  let line = ints_line ints in
  if String.length line <= 2048 then line
  else
    Printf.sprintf "len=%d md5=%s" (String.length line)
      (Digest.to_hex (Digest.string line))

let dump_batch tok name texts =
  let ids, lengths =
    Brot.encode_batch_ids tok ~add_special_tokens:false ~domains:4 texts
  in
  let buf = Buffer.create 65536 in
  for i = 0 to Bigarray.Array1.dim ids - 1 do
    Buffer.add_string buf (Int32.to_string (Bigarray.Array1.get ids i));
    Buffer.add_char buf ' '
  done;
  pf "%s/batch: rows=%d md5=%s lengths=%s\n" name (List.length texts)
    (Digest.to_hex (Digest.string (Buffer.contents buf)))
    (digest_line lengths)

(* The generated adversarial inputs. *)

let adversarial =
  let cases =
    [
      ("empty", "");
      ("nul", "a\000b \000\000 c");
      ("c0_c1", "a\192b\193 \192\128");
      ("f5_ff", "\245\128\128\128 \246x \255 \254\253");
      ("stray_cont", "\128 a\129b \191\191\191");
      ("surrogate", "x\237\160\128y");
      ("above_max", "\244\144\128\128");
      ("cut_2", "ab\206");
      ("cut_3", "ab\226\130");
      ("cut_4", "ab\240\159\152");
      ("contraction_end_s", "it's he'");
      ("contraction_end_re", "we're they'r");
      ("contraction_ll", "she'll we'l");
      ("contraction_other", "a'x 'y '");
      ("spaces_end", "hello   ");
      ("space_only", "     ");
      ("tabs_newlines", "a\tb\nc\r\nd  \n");
      ("space_before_word", "  hello   world");
      ("pretoken_15", " abcdefghijklmn");
      ("pretoken_16", " abcdefghijklmno");
      ("pretoken_14", " abcdefghijklm");
      ("digits", "12345 67890 1a2b3c");
      ("punct", "!!! ...) {[}] ~`@#");
      ("nonascii_letters", "h\195\169llo w\195\182rld \206\177\206\178\206\179");
      ("nonascii_digits", "\217\160\217\161\217\162 42");
      ("nonascii_space", "a\194\160b \227\128\128c");
      ("cjk", "\230\151\165\230\156\172\232\170\158 \228\184\173\230\150\135");
      ("emoji", "a \240\159\152\128\240\159\154\128 b");
      ("mixed_soup", "a\194\160\128\237\160\128\244\144\128\128\192 b\206");
      ("disagree_mid_span", "a\196\138\196\138a \196\138a");
    ]
  in
  let random_ascii n = String.init n (fun _ -> Char.chr (32 + rand_int 95)) in
  let random_bytes n = String.init n (fun _ -> Char.chr (rand_int 256)) in
  let random_utf8 n =
    let buf = Buffer.create (n * 2) in
    while Buffer.length buf < n do
      let cp =
        match rand_int 4 with
        | 0 -> 32 + rand_int 95
        | 1 -> 0xA0 + rand_int 0x500
        | 2 -> 0x3000 + rand_int 0x1000
        | _ -> 0x10000 + rand_int 0x1000
      in
      Buffer.add_utf_8_uchar buf (Uchar.of_int cp)
    done;
    Buffer.contents buf
  in
  (* Batch-alignment adversaries for the mask-scanner walker: the walk
     classifies 64-byte batches on a grid anchored at the range start, so each
     family plants its shape at every offset 0..80. The padding is short
     pretokens ("ab " cycles) and the tails run at least 64 bytes past the
     shape: a padding run over 15 bytes would take the Encode hand-back and
     re-anchor the grid at the shape, and a short tail would route the shape's
     batch to the scalar tail — either way the shape would never cross a
     streamed batch edge, which is the point of the sweep. [span64] keeps the
     long-run padding deliberately: that family exercises the hand-back
     re-anchor path itself. [giveback_nbsp] is the whitespace give-back decided
     by a non-ASCII whitespace character, which only the bad-zone widening
     around non-ASCII bytes sends to the scalar walker. The [cls] family uses a
     distinct code point per offset, so a first-met class hand-back re-enters
     the walk mid-batch everywhere. *)
  let batch_alignments =
    let pad n = String.init n (fun i -> "ab ".[i mod 3]) in
    let fam name shape =
      List.init 81 (fun off -> (Printf.sprintf "%s_%d" name off, shape off))
    in
    List.concat
      [
        fam "bz_hi" (fun off -> pad off ^ "\195\169" ^ pad 80);
        fam "bz_hi_ws" (fun off -> pad off ^ " \194\160 " ^ pad 80);
        fam "bz_giveback_nbsp" (fun off ->
            String.make off 'x' ^ "  \194\160 " ^ String.make 80 'y');
        fam "bz_apo" (fun off -> pad off ^ "'ll they've o'x " ^ pad 80);
        fam "bz_ws" (fun off -> pad off ^ "  \t\r\n  b " ^ pad 80);
        fam "bz_span64" (fun off ->
            String.make off 'z' ^ " " ^ String.make 63 'w' ^ " tail");
        fam "bz_digit" (fun off ->
            pad off ^ " " ^ String.make 20 '8' ^ "a9 " ^ pad 64);
        fam "bz_cls" (fun off ->
            let buf = Buffer.create 192 in
            Buffer.add_string buf (pad off);
            Buffer.add_utf_8_uchar buf (Uchar.of_int (0x2460 + off));
            Buffer.add_char buf ' ';
            Buffer.add_string buf (pad 80);
            Buffer.contents buf);
      ]
  in
  (* Chunk-seam adversaries for the two-phase kernel: the entry loop stages 256
     spans per chunk, so each family plants its event at span indices 250..262 —
     an Encode hand-back, a Class hand-back (a code point per index, first met
     there), identical adjacent pretokens (miss, store, hit across the harvest
     seam) and a plain stream whose length lands on the seam.
     [ids_full_midchunk] outgrows the driver's 65536-id reserve inside one call,
     so Ids_full parks and resumes a chunk mid-stage. *)
  let chunk_seams =
    let spans k = String.concat "" (List.init k (fun _ -> " a")) in
    let fam name shape =
      List.init 13 (fun o ->
          (Printf.sprintf "%s_%d" name (250 + o), shape (250 + o)))
    in
    List.concat
      [
        fam "seam_encode" (fun k ->
            spans k ^ " " ^ String.make 20 'b' ^ spans 300);
        fam "seam_class" (fun k ->
            let buf = Buffer.create 1024 in
            Buffer.add_string buf (spans k);
            Buffer.add_utf_8_uchar buf (Uchar.of_int (0x2500 + k));
            Buffer.add_string buf (spans 300);
            Buffer.contents buf);
        fam "seam_twin" (fun k -> spans k ^ " zq zq" ^ spans 300);
        fam "seam_plain" spans;
      ]
  in
  cases
  @ [
      ("letters_100k", String.make 100_000 'x');
      ("spaces_100k", String.make 100_000 ' ');
      ( "pretokens_3000",
        String.concat " " (List.init 3000 (fun i -> Printf.sprintf "w%d" i)) );
      ( "ids_full_midchunk",
        String.concat "" (List.init 5000 (fun _ -> " abcdefghijklmn")) );
      ("random_ascii_512k", random_ascii 524_288);
      ("random_bytes_512k", random_bytes 524_288);
      ("random_utf8_512k", random_utf8 524_288);
      ("random_bytes_short", random_bytes 300);
    ]
  @ batch_alignments @ chunk_seams

(* The parity corpora, whole and split into their documents. *)

let documents text =
  let lines = String.split_on_char '\n' text in
  let join current = String.concat "\n" (List.rev current) in
  let rec split docs current = function
    | [] -> List.rev (join current :: docs)
    | "====" :: rest -> split (join current :: docs) [] rest
    | line :: rest -> split docs (line :: current) rest
  in
  split [] [] lines

let split_on_marker marker s =
  let n = String.length s and m = String.length marker in
  let rec find i =
    if i + m > n then None
    else
      match String.index_from_opt s i marker.[0] with
      | None -> None
      | Some j ->
          if j + m > n then None
          else if String.sub s j m = marker then Some j
          else find (j + 1)
  in
  let rec go acc start =
    match find start with
    | None -> List.rev (String.sub s start (n - start) :: acc)
    | Some j -> go (String.sub s start (j - start) :: acc) (j + m)
  in
  go [] 0

(* Synthetic byte-level models, always present. The alphabet mapping is the
   fixed GPT-2 one, restated here because the library does not expose the
   encoding direction. *)

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

(* Every byte, plus merges growing common English fragments. *)
let synth_full ?cache_capacity ?ignore_merges () =
  let merges =
    [
      ("t", "h");
      ("th", "e");
      ("i", "n");
      ("e", "r");
      ("a", "n");
      ("\196\160", "t");
      ("\196\160t", "he");
      ("h", "e");
      ("\196\160", "a");
      ("o", "n");
      ("\196\160", "w");
      ("e", "n");
      ("1", "2");
      ("12", "3");
    ]
  in
  let base = List.init 256 (fun b -> byte_alphabet.(b)) in
  let derived =
    List.map (fun (a, b) -> a ^ b) merges
    |> List.filter (fun s -> not (List.mem s base))
    |> List.sort_uniq compare
  in
  let vocab = List.mapi (fun i s -> (s, i)) (base @ derived) in
  Brot.bpe ~vocab ~merges ~pre:(pre ()) ?cache_capacity ?ignore_merges ()

(* Bytes q, z and Q have no entry: spans holding them go back to OCaml. *)
let synth_missing () =
  let keep b =
    let c = Char.chr b in
    c <> 'q' && c <> 'z' && c <> 'Q'
  in
  let base =
    List.init 256 (fun b -> b)
    |> List.filter keep
    |> List.map (fun b -> byte_alphabet.(b))
  in
  let vocab = List.mapi (fun i s -> (s, i)) (base @ [ "th"; "he" ]) in
  Brot.bpe ~vocab ~merges:[ ("t", "h"); ("h", "e") ] ~pre:(pre ()) ()

(* A merge whose result stands for other bytes than its entry: "\xC4" and "\x8A"
   merge into "Ċ" (bytes C4 8A), whose entry decodes to the one byte "\n", so
   len_table reads 1 where the symbol covers 2 bytes and emit_word records the
   run. The kernel must hand such a span back. *)
let synth_disagree () =
  let vocab = [ ("\196", 0); ("\138", 1); ("\196\138", 2); ("a", 3) ] in
  Brot.bpe ~vocab ~merges:[ ("\196", "\138") ] ~pre:(pre ()) ()

let () =
  let corpus_arg =
    if Array.length Sys.argv > 1 then Some Sys.argv.(1) else None
  in
  let fixture name = read_opt (Filename.concat "fixtures/parity" name) in
  let data name = read_opt (Filename.concat "../bench/data" name) in
  let file_tok name =
    let path = Filename.concat "../bench/data" (name ^ ".json") in
    if not (Sys.file_exists path) then None
    else match Brot.from_file path with Ok tok -> Some tok | Error _ -> None
  in
  let texts =
    List.concat
      [
        (match fixture "sample.txt" with
        | None -> []
        | Some text ->
            ("sample", text)
            :: List.mapi
                 (fun i d -> (Printf.sprintf "sample.%d" i, d))
                 (documents text));
        (match fixture "edge_cases.txt" with
        | None -> []
        | Some text ->
            ("edge_cases", text)
            :: List.mapi
                 (fun i d -> (Printf.sprintf "edge_cases.%d" i, d))
                 (documents text));
        (match data "wiki_64k.txt" with
        | None -> []
        | Some text -> [ ("wiki_64k", text) ]);
        adversarial;
      ]
  in
  let run name tok =
    List.iter
      (fun (tname, text) ->
        dump_ids tok (name ^ "/" ^ tname) text;
        dump_encoding tok (name ^ "/" ^ tname) text)
      texts;
    dump_batch tok name (List.map snd texts);
    match corpus_arg with
    | None -> ()
    | Some path ->
        dump_batch tok (name ^ "/corpus")
          (split_on_marker "<|endoftext|>" (read_file path))
  in
  (match file_tok "gpt2" with
  | Some tok -> run "gpt2" tok
  | None -> pf "gpt2: skipped\n");
  (match file_tok "roberta_base" with
  | Some tok -> run "roberta_base" tok
  | None -> pf "roberta_base: skipped\n");
  run "synth_full" (synth_full ());
  run "synth_nocache" (synth_full ~cache_capacity:0 ());
  run "synth_ignore" (synth_full ~ignore_merges:true ());
  run "synth_missing" (synth_missing ());
  run "synth_disagree" (synth_disagree ())
