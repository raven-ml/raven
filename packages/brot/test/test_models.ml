(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The tokenizer type and the WordPiece, word-level, Unigram and byte
   constructors; BPE's are test_bpe's. Expectations that name Hugging Face were
   read off tokenizers 0.23.1 with the same vocabulary. *)

open Windtrap
open Brot

let tokens t text = Array.to_list (Encoding.tokens (encode t text))
let numbered tokens = List.mapi (fun i token -> (token, i)) tokens

(* The length of the character at [i]: a byte that opens no character is one. *)
let char_length text i =
  let d = String.get_utf_8_uchar text i in
  if Uchar.utf_decode_is_valid d then Uchar.utf_decode_length d else 1

let characters text =
  let rec go i acc =
    if i >= String.length text then List.rev acc
    else
      let n = char_length text i in
      go (i + n) (String.sub text i n :: acc)
  in
  go 0 []

let file contents =
  let path = temp_file () in
  Out_channel.with_open_bin path (fun oc ->
      Out_channel.output_string oc contents);
  path

let quoted = Gen.with_pp (fun ppf s -> Format.fprintf ppf "%S" s)
let capacities = Gen.of_list ~pp:Format.pp_print_int [ 1; 4; 16; 4096; 262144 ]

(* t *)

(* WordPiece: a word is cut greedily, the longest entry first; a word no run of
   entries covers, or over [max_input_chars_per_word] characters, is the unknown
   token. Some words are past the 15 bytes a cache key holds. *)

let wordpiece_vocab =
  [ "[UNK]"; "a"; "ab"; "abc"; "b"; "##a"; "##b"; "##c"; "##bc"; "\u{e9}";
    "##\u{e9}"; "x"; "##x"; "aaaa"; "##aaaa"; "."; "," ]
[@@ocamlformat "disable"]

let wordpiece_words =
  [ "a"; "ab"; "abc"; "abcabc"; "b"; "bca"; "\u{e9}"; "a\u{e9}b"; "xx"; "q";
    "aq"; "aaaaaaaaaaaaaaaaaaa"; "abcabcabcabcabc"; "abcabcabcabcabca";
    "\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}" ]
[@@ocamlformat "disable"]

let wordpiece_max_chars = 12

let wordpiece_text =
  let open Gen in
  let piece =
    frequency
      [
        (4, of_list wordpiece_words);
        (2, of_list [ " "; "  "; "\t" ]);
        (1, of_list [ "."; "," ]);
      ]
  in
  quoted (map (String.concat "") (list ~size:(int_range 0 30) piece))

let wordpiece_model ?cache_capacity () =
  wordpiece ~vocab:(numbered wordpiece_vocab)
    ~max_input_chars_per_word:wordpiece_max_chars ~pre:Pre_tokenizer.bert
    ?cache_capacity ()

(* Unigram: pieces that overlap, so the best cut is not the longest match at
   each position, and a model with or without an unknown token and byte
   fallback. Without one, a text holding ["q"] raises on both sides. *)

let overlapping =
  [ ("<unk>", 0.0); ("a", -1.0); ("b", -1.0); ("c", -1.0); ("ab", -2.0);
    ("abc", -5.0); ("bc", -1.5) ]
[@@ocamlformat "disable"]

let unigram_words =
  [ "a"; "abc"; "bca"; "cab"; "\u{e9}"; "a\u{e9}"; "q"; "aq";
    "abcabcabcabcabca"; "\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}";
    "\xff" ]
[@@ocamlformat "disable"]

let unigram_text =
  quoted
    Gen.(
      map (String.concat " ")
        (list ~size:(int_range 0 20) (of_list unigram_words)))

(* The byte tokens are entries only under byte fallback. *)
let unigram_model ~unk ~byte_fallback ?cache_capacity () =
  let bytes =
    if byte_fallback then
      List.init 256 (fun b -> (Printf.sprintf "<0x%02X>" b, -10.0))
    else []
  in
  let vocab = overlapping @ [ ("\u{e9}", -1.0) ] @ bytes in
  unigram ~vocab
    ?unk_id:(if unk then Some 0 else None)
    ~byte_fallback ?cache_capacity ~pre:Pre_tokenizer.whitespace_split ()

let unigram_config =
  Gen.with_pp
    (fun ppf (capacity, unk, byte_fallback) ->
      Format.fprintf ppf "capacity %d, unk %b, byte_fallback %b" capacity unk
        byte_fallback)
    Gen.(triple capacities bool bool)

(* The reference is one uncached tokenizer per configuration, whose ids are a
   function of the text alone. *)
let uncached_wordpiece = lazy (wordpiece_model ~cache_capacity:0 ())

let uncached_unigram =
  let models =
    lazy
      (Array.init 4 (fun i ->
           unigram_model
             ~unk:(i land 1 = 1)
             ~byte_fallback:(i land 2 = 2)
             ~cache_capacity:0 ()))
  in
  fun ~unk ~byte_fallback ->
    (Lazy.force models).(Bool.to_int unk + (2 * Bool.to_int byte_fallback))

let wordpiece_cache =
  let tok = abstract "t" in
  stateful ~steps:30
    "gives the ids an uncached WordPiece tokenizer gives, whatever it encoded \
     before"
    [
      command "wordpiece"
        (capacities @-> makes tok)
        (fun _ -> Lazy.force uncached_wordpiece)
        (fun cache_capacity -> wordpiece_model ~cache_capacity ());
      command "encode_ids"
        (tok ^-> wordpiece_text @-> returns (array int))
        (fun t text -> encode_ids t text)
        (fun t text -> encode_ids t text);
    ]

let unigram_cache =
  let tok = abstract "t" in
  stateful ~steps:30
    "gives the ids an uncached Unigram tokenizer gives, whatever it encoded \
     before"
    [
      command "unigram"
        (unigram_config @-> makes tok)
        (fun (_, unk, byte_fallback) -> uncached_unigram ~unk ~byte_fallback)
        (fun (cache_capacity, unk, byte_fallback) ->
          unigram_model ~unk ~byte_fallback ~cache_capacity ());
      command "encode_ids"
        (tok ^-> unigram_text @-> returns (array int))
        (fun t text -> encode_ids t text)
        (fun t text -> encode_ids t text);
    ]

let wordpiece_domains =
  let tok = abstract "t" in
  stateful ~domains:2 ~count:4 ~steps:6
    "encodes WordPiece text on two domains as on one"
    [
      command "wordpiece"
        (Gen.unit @-> makes tok)
        (fun () -> Lazy.force uncached_wordpiece)
        (fun () -> wordpiece_model ~cache_capacity:1 ());
      command "encode_ids"
        (tok ^-> wordpiece_text @-> returns (array int))
        (fun t text -> encode_ids t text)
        (fun t text -> encode_ids t text);
    ]

let unigram_domains =
  let tok = abstract "t" in
  stateful ~domains:2 ~count:4 ~steps:6
    "encodes Unigram text on two domains as on one"
    [
      command "unigram"
        (Gen.bool @-> makes tok)
        (fun unk -> uncached_unigram ~unk ~byte_fallback:false)
        (fun unk ->
          unigram_model ~unk ~byte_fallback:false ~cache_capacity:1 ());
      command "encode_ids"
        (tok ^-> unigram_text @-> returns (array int))
        (fun t text -> encode_ids t text)
        (fun t text -> encode_ids t text);
    ]

(* Every letter and pair of letters is an entry, with and without the prefix, so
   that a word of up to eight letters is at most four ids and is kept: past 512
   distinct ones, as many as the starting table holds and twice what makes it
   grow, it grows while the text is encoded. *)
let grown_table =
  prop ~count:5 "finds what a WordPiece table held after it grows"
    Gen.(
      list ~size:(int_range 1500 2500)
        (string_of ~size:(int_range 1 8) (char_range 'a' 'd')))
    (fun words ->
      let text = String.concat " " words in
      let letters = [ "a"; "b"; "c"; "d" ] in
      let pieces =
        letters @ List.concat_map (fun a -> List.map (( ^ ) a) letters) letters
      in
      let vocab =
        numbered (("[UNK]" :: pieces) @ List.map (( ^ ) "##") pieces)
      in
      let make cache_capacity =
        wordpiece ~vocab ~pre:Pre_tokenizer.bert ?cache_capacity ()
      in
      let t = make None in
      let expected = encode_ids (make (Some 0)) text in
      cover "more than 512 words kept"
        (List.length (List.sort_uniq compare words) > 512);
      equal ~msg:"cold" (array int) expected (encode_ids t text);
      equal ~msg:"warm" (array int) expected (encode_ids t text))

(* A thread that finds the tokenizer's state held encodes with one of its
   own. *)
let held_state =
  cases "encodes on another thread while a state is held" ~id:fst
    [
      ( "bpe",
        fun cache_capacity ->
          bpe
            ~vocab:(numbered [ "a"; "b"; "c"; "ab"; "abc" ])
            ~merges:[ ("a", "b"); ("ab", "c") ]
            ~cache_capacity () );
      ("wordpiece", fun cache_capacity -> wordpiece_model ~cache_capacity ());
      ( "unigram",
        fun cache_capacity ->
          unigram_model ~unk:true ~byte_fallback:false ~cache_capacity () );
    ]
    (fun (_, make) ->
      let t = make 262144 in
      let text = "abc abcabc a\u{e9}b q abc . aaaa abcdefghabcdefgh" in
      let expected = encode_ids (make 0) text in
      let ids = ref [||] in
      hold_cache t (fun () ->
          Thread.join (Thread.create (fun () -> ids := encode_ids t text) ()));
      equal ~msg:"while held" (array int) expected !ids;
      equal ~msg:"after" (array int) expected (encode_ids t text))

(* A model's states, caches included, live in the tokenizer: once it is dropped,
   nothing keeps them or the vocabulary entry they hold, built at run time to be
   finalised. The batch spreads encodings over four domains. *)
let entries_finalised make =
  let entry = String.concat "" [ "wor"; "ld" ] in
  let finalised = ref 0 in
  Gc.finalise_last (fun () -> incr finalised) entry;
  let encode_and_drop () =
    let t = make entry in
    let text = String.concat "" (List.init 16384 (fun _ -> "hello world ")) in
    ignore (encode_ids t "hello world" : int array);
    ignore (encode_batch_ids t ~exec:(Fixture.spawn 4) [ text; text ] : ids * _)
  in
  (Sys.opaque_identity encode_and_drop) ();
  Gc.full_major ();
  !finalised

let freed_states =
  let words entry = [ "[UNK]"; "hello"; entry; "##o" ] in
  cases "frees its model's states with it" ~id:fst
    [
      ( "wordpiece",
        fun entry ->
          wordpiece
            ~vocab:(numbered (words entry))
            ~pre:Pre_tokenizer.whitespace () );
      ( "unigram",
        fun entry ->
          unigram
            ~vocab:(List.map (fun w -> (w, -1.)) (words entry))
            ~unk_id:0 ~pre:Pre_tokenizer.whitespace () );
    ]
    (fun (_, make) ->
      equal ~msg:"entries finalised" int 1 (entries_finalised make))

(* wordpiece *)

(* The pieces of [word] as (id, start, stop) within it: the longest entry first,
   later pieces prefixed, or the unknown token for the whole word. *)
let greedy_pieces word =
  let id token = Option.get (List.find_index (( = ) token) wordpiece_vocab) in
  let len = String.length word in
  let unk = [ (id "[UNK]", 0, len) ] in
  let rec go pos acc =
    if pos >= len then List.rev acc
    else
      let rec longest stop =
        if stop <= pos then None
        else
          let piece = String.sub word pos (stop - pos) in
          let entry = if pos > 0 then "##" ^ piece else piece in
          if List.mem entry wordpiece_vocab then Some (id entry, stop)
          else longest (stop - 1)
      in
      match longest len with
      | None -> unk
      | Some (id, stop) -> go stop ((id, pos, stop) :: acc)
  in
  if List.length (characters word) > wordpiece_max_chars then unk else go 0 []

(* Bert's words with their start: runs of letters, and each punctuation
   character alone. *)
let bert_words text =
  let len = String.length text in
  let is_space c = c = ' ' || c = '\t' in
  let is_punct c = c = '.' || c = ',' in
  let rec go i acc =
    if i >= len then List.rev acc
    else if is_space text.[i] then go (i + 1) acc
    else if is_punct text.[i] then
      go (i + 1) ((String.make 1 text.[i], i) :: acc)
    else
      let j = ref i in
      while !j < len && not (is_space text.[!j] || is_punct text.[!j]) do
        incr j
      done;
      go !j ((String.sub text i (!j - i), i) :: acc)
  in
  go 0 []

let greedy_law =
  prop ~count:80 "cuts every word greedily, the longest entry first"
    wordpiece_text (fun text ->
      let words = bert_words text in
      cover "a word over the character budget"
        (List.exists
           (fun (w, _) -> List.length (characters w) > wordpiece_max_chars)
           words);
      cover "a word no run of entries covers"
        (List.exists (fun (w, _) -> w = "q" || w = "aq") words);
      let expected =
        List.concat_map
          (fun (word, start) ->
            List.map
              (fun (id, a, b) -> (id, (start + a, start + b)))
              (greedy_pieces word))
          words
      in
      let encoding = encode (wordpiece_model ()) text in
      equal
        (list (pair int (pair int int)))
        expected
        (List.combine
           (Array.to_list (Encoding.ids encoding))
           (Array.to_list (Encoding.offsets encoding))))

(* HF [models.WordPiece] with the same vocabulary. *)
let hf_wordpiece ?max_input_chars_per_word () =
  wordpiece
    ~vocab:
      (numbered
         [ "[UNK]"; "un"; "##able"; "##a"; "##b"; "a"; "\u{e9}"; "##\u{e9}";
           "play"; "##ing" ])
    ?max_input_chars_per_word ()
[@@ocamlformat "disable"]

let unknown_word =
  cases "is one unk_token for a word some piece of which has no entry"
    ~id:(fun (word, _) -> Printf.sprintf "%S" word)
    [
      ("unable", [ "un"; "##able" ]);
      ("unab", [ "un"; "##a"; "##b" ]);
      ("playing", [ "play"; "##ing" ]);
      ("unxyz", [ "[UNK]" ]);
      ("", []);
    ]
    (fun (word, expected) ->
      equal (list string) expected (tokens (hf_wordpiece ()) word))

let e n = String.concat "" (List.init n (fun _ -> "\u{e9}"))
let later_e n = "\u{e9}" :: List.init (n - 1) (fun _ -> "##\u{e9}")

(* The budget counts characters; [None] is the default of 100. *)
let over_budget =
  cases "is unk_token for a word over max_input_chars_per_word characters"
    ~id:(fun (id, _, _, _) -> id)
    [
      ("5 a in 5", Some 5, "aaaaa", [ "a"; "##a"; "##a"; "##a"; "##a" ]);
      ("6 a in 5", Some 5, "aaaaaa", [ "[UNK]" ]);
      ("5 \u{e9} in 5", Some 5, e 5, later_e 5);
      ("6 \u{e9} in 5", Some 5, e 6, [ "[UNK]" ]);
      ("unable in 5", Some 5, "unable", [ "[UNK]" ]);
      ("100 \u{e9} in the default", None, e 100, later_e 100);
      ("101 \u{e9} in the default", None, e 101, [ "[UNK]" ]);
    ]
    (fun (_, max_input_chars_per_word, word, expected) ->
      equal (list string) expected
        (tokens (hf_wordpiece ?max_input_chars_per_word ()) word))

(* Four letters and two stray continuation bytes are over a budget of five,
   though entries cover them. *)
let test_stray_bytes () =
  let t =
    wordpiece
      ~vocab:[ ("[UNK]", 0); ("a", 1); ("##a", 2); ("##\x80", 3) ]
      ~max_input_chars_per_word:5 ()
  in
  equal ~msg:"over" (list string) [ "[UNK]" ] (tokens t "aaaa\x80\x80");
  equal ~msg:"within" (list string)
    [ "a"; "##a"; "##a"; "##\x80" ]
    (tokens t "aaa\x80")

let test_custom_prefix () =
  let t =
    wordpiece
      ~vocab:(numbered [ "[UNK]"; "run"; "@@ning"; "##ning" ])
      ~continuing_subword_prefix:"@@" ()
  in
  equal (list string) [ "run"; "@@ning" ] (tokens t "running")

(* HF: a WordPiece without its unknown token encodes "a aa" as [0, 0, 1] and
   raises "Missing [UNK] token from the vocabulary" at "z". *)
let test_wordpiece_no_unk () =
  let t =
    wordpiece ~pre:Pre_tokenizer.whitespace ~vocab:[ ("a", 0); ("##a", 1) ] ()
  in
  equal ~msg:"known words" (array int) [| 0; 0; 1 |] (encode_ids t "a aa");
  raises_match ~msg:"an unknown word" Exn.failure (fun () -> encode t "a z")

(* A token given twice is its last pair on every path that reads the vocabulary,
   whichever walk the model encodes with. *)
let last_pair t ~pairs =
  equal ~msg:"token_to_id" (option int) (Some 2) (token_to_id t "a");
  equal ~msg:"the earlier id" (option string) None (id_to_token t 0);
  equal ~msg:"the last id" (option string) (Some "a") (id_to_token t 2);
  equal ~msg:"encode" (array int) [| 2 |] (encode_ids t "a");
  equal ~msg:"vocab_size" int (List.length pairs) (vocab_size t);
  equal ~msg:"vocab"
    (list (pair string int))
    pairs
    (List.sort compare (vocab t))

let test_wordpiece_given_twice () =
  let t = wordpiece ~vocab:[ ("a", 0); ("a", 2); ("[UNK]", 1) ] () in
  last_pair t ~pairs:[ ("[UNK]", 1); ("a", 2) ]

(* Tables sized by the largest id took 2^23 words or more for the sparse
   vocabulary below. *)
let sparse_ids make =
  let top = 1 lsl 23 in
  let before = Gc.allocated_bytes () in
  let t = make [ ("[UNK]", 0); ("a", 1); ("b", top) ] in
  let encoding = encode t "a b" in
  let allocated = Gc.allocated_bytes () -. before in
  less ~msg:"bytes allocated" float_exact ~than:16e6 allocated;
  equal ~msg:"ids" (array int) [| 1; top |] (Encoding.ids encoding);
  equal ~msg:"tokens" (array string) [| "a"; "b" |] (Encoding.tokens encoding);
  equal ~msg:"offsets"
    (array (pair int int))
    [| (0, 1); (2, 3) |]
    (Encoding.offsets encoding);
  equal ~msg:"id_to_token" (option string) (Some "b") (id_to_token t top);
  equal ~msg:"decode" string "a b" (decode t [| 1; top |]);
  let top = (1 lsl 31) - 1 in
  let t = make [ ("[UNK]", 0); ("b", top) ] in
  equal ~msg:"2^31 - 1" (option string) (Some "b") (id_to_token t top)

let refused_ids make =
  List.iter
    (fun id ->
      raises_match ~msg:(Printf.sprintf "id %d" id)
        (Exn.invalid_arg ~sub:"2^31") (fun () ->
          make [ ("[UNK]", 0); ("b", id) ]))
    [ -1; 1 lsl 31; max_int; min_int ]

let sparse_wordpiece vocab = wordpiece ~pre:Pre_tokenizer.whitespace ~vocab ()
let sparse_word_level vocab = word_level ~unk_token:"[UNK]" ~vocab ()

(* word_level *)

let test_word_ids () =
  let t = word_level ~vocab:(numbered [ "<unk>"; "a"; "bb" ]) () in
  equal (array int) [| 1; 2; 0; 1 |] (encode_ids t "a bb zz a")

let test_word_level_split () =
  let t = word_level ~vocab:(numbered [ "<unk>"; "a"; ","; "b"; "c" ]) () in
  equal (list string) [ "a"; ","; "b"; "c" ] (tokens t "a,b\t c")

(* HF: a WordLevel without its unknown token raises "Missing [UNK] token from
   the vocabulary" at a word it lacks, and encodes a text of known words. *)
let test_word_level_no_unk () =
  let t = word_level ~vocab:[ ("a", 0) ] () in
  equal ~msg:"known words" (array int) [| 0; 0 |] (encode_ids t "a a");
  raises_match ~msg:"an unknown word" Exn.failure (fun () -> encode t "a zz");
  raises_match ~msg:"no vocabulary" Exn.failure (fun () ->
      encode (word_level ()) "no numbers here")

(* unigram *)

(* The cuts of [text] from [start], each a list of its pieces with their spans:
   an entry of [vocab], or [None] for an unknown character, which a character no
   one-character entry holds may be. *)
let rec cuts vocab text start =
  let len = String.length text in
  if start = len then [ [] ]
  else
    let rest stop piece =
      List.map (fun cut -> (piece, start, stop) :: cut) (cuts vocab text stop)
    in
    let known =
      List.concat_map
        (fun (token, _) ->
          let n = String.length token in
          if start + n <= len && String.sub text start n = token then
            rest (start + n) (Some token)
          else [])
        vocab
    in
    let n = char_length text start in
    if List.mem_assoc (String.sub text start n) vocab then known
    else known @ rest (start + n) None

(* The cut whose scores add up to the most, an unknown character scoring ten
   below the rarest entry; of equal sums, the one whose last piece starts first,
   then the piece before that. Its unknown characters in a row are one unknown
   token, which spans them all. Also the number of cuts of its sum. *)
let best_cut vocab text =
  let rarest = List.fold_left (fun m (_, s) -> Float.min m s) 0. vocab in
  let score cut =
    List.fold_left
      (fun sum (piece, _, _) ->
        match piece with
        | Some t -> sum +. List.assoc t vocab
        | None -> sum +. (rarest -. 10.))
      0. cut
  in
  let better a b =
    score a > score b
    || score a = score b
       && List.rev_map (fun (_, start, _) -> start) a
          < List.rev_map (fun (_, start, _) -> start) b
  in
  let all = cuts vocab text 0 in
  let best =
    List.fold_left (fun b c -> if better c b then c else b) (List.hd all) all
  in
  let rec fuse = function
    | (None, start, _) :: (None, _, stop) :: rest ->
        fuse ((None, start, stop) :: rest)
    | (piece, start, stop) :: rest ->
        (Option.value ~default:"<unk>" piece, (start, stop)) :: fuse rest
    | [] -> []
  in
  (fuse best, List.length (List.filter (fun c -> score c = score best) all))

(* Entries over three letters scored in quarters, so that sums are exact: most
   letters an entry of their own, and longer entries, most of them scored as the
   sum of their letters, which ties the two cuts. Texts hold a letter [z] that
   no entry holds. *)
let unigram_case =
  let open Gen in
  let letters = [ "a"; "b"; "c" ] in
  let quarters = map (fun q -> -.Float.of_int q /. 4.) (int_range 1 4) in
  let single letter =
    frequency [ (3, map (fun s -> [ (letter, s) ]) quarters); (1, constant []) ]
  in
  let longer =
    pair
      (map (String.concat "") (list ~size:(int_range 2 3) (of_list letters)))
      (frequency [ (3, constant None); (1, map Option.some quarters) ])
  in
  let+ a = single "a"
  and+ b = single "b"
  and+ c = single "c"
  and+ longer = list ~size:(int_range 1 5) longer
  and+ text =
    map (String.concat "")
      (list ~size:(int_range 0 8)
         (frequency [ (8, of_list letters); (1, constant "z") ]))
  in
  let singles = a @ b @ c in
  let sum entry =
    List.fold_left
      (fun sum letter ->
        sum +. Option.value ~default:(-1.) (List.assoc_opt letter singles))
      0. (characters entry)
  in
  let entries =
    List.fold_left
      (fun acc (entry, score) ->
        if List.mem_assoc entry acc then acc
        else acc @ [ (entry, Option.value ~default:(sum entry) score) ])
      singles longer
  in
  (entries, text)

let pp_unigram_case ppf (vocab, text) =
  let entry (t, s) = Printf.sprintf "(%S, %g)" t s in
  Format.fprintf ppf "vocab [%s]@ text %S"
    (String.concat "; " (List.map entry vocab))
    text

let best_cut_law =
  let overlap = List.tl overlapping in
  let shade vocab delta =
    List.map (fun (t, s) -> if t = "ab" then (t, s +. delta) else (t, s)) vocab
  in
  prop ~count:150 "cuts a text into the pieces whose scores add up to the most"
    ~examples:
      [
        (overlap, "abc");
        (overlap, "abcabc");
        (overlap, "a");
        (overlap, "cba");
        (overlap, "ab");
        (overlap, "");
        ([ ("ab", -1.); ("bc", -1.) ], "abc");
        (shade overlap 0.1, "ab");
        (shade overlap (-0.1), "ab");
        ([ ("a", -1.); ("aa", -2.) ], "aaaa");
        ([ ("a", -1.) ], "azza");
      ]
    (Gen.with_pp pp_unigram_case unigram_case)
    (fun (vocab, text) ->
      let expected, ties = best_cut vocab text in
      cover "a tie between cuts" (ties > 1);
      cover "an unknown run of two characters"
        (List.exists
           (fun (token, (start, stop)) -> token = "<unk>" && stop - start > 1)
           expected);
      let t = unigram ~vocab:(("<unk>", 0.) :: vocab) ~unk_id:0 () in
      let encoding = encode t text in
      equal ~msg:(Printf.sprintf "%S" text)
        (list (pair string (pair int int)))
        expected
        (List.combine
           (Array.to_list (Encoding.tokens encoding))
           (Array.to_list (Encoding.offsets encoding))))

(* With [a] at [s] and [k] of them as one piece at [k s], every cut of a run of
   [a]s sums to the same total exactly, [s] being a whole number of quarters, so
   the single [a]s left over come first. *)
let ties_law =
  prop ~count:70 "keeps, of equal sums, the cut whose last piece starts first"
    ~examples:[ (2, 3, 4, 0); (7, 3, 4, 1); (8, 4, 1, 2) ]
    Gen.(quad (int_range 0 40) (int_range 2 5) (int_range 1 16) (int_range 0 2))
    (fun (n, k, m, j) ->
      let s = -.Float.ldexp (Float.of_int m) (-j) in
      let t =
        unigram
          ~vocab:
            [
              ("<unk>", 0.0); ("a", s); (String.make k 'a', Float.of_int k *. s);
            ]
          ~unk_id:0 ()
      in
      cover "shorter than the long piece" (n < k);
      cover "single pieces left over" (n >= k && n mod k > 0);
      cover "long pieces only" (n >= k && n mod k = 0);
      equal
        ~msg:(Printf.sprintf "%d a at %g, a*%d" n s k)
        (list int)
        (List.init (n mod k) (fun _ -> 1) @ List.init (n / k) (fun _ -> 2))
        (Array.to_list (encode_ids t (String.make n 'a'))))

let letters = [ ("<unk>", 0.0); ("a", -1.0); ("b", -1.0) ]

let with_bytes ~upto vocab =
  vocab @ List.init upto (fun b -> (Printf.sprintf "<0x%02X>" b, -20.0))

let unigram_ids t text =
  Array.to_list (encode_ids t ~add_special_tokens:false text)

let unigram_offsets t text =
  Array.to_list (Encoding.offsets (encode t ~add_special_tokens:false text))

(* HF reports offsets in characters and brot in bytes; the spans below are the
   byte ones. *)
let test_white_space () =
  let t = unigram ~vocab:letters ~unk_id:0 () in
  equal ~msg:"a space" (list int) [ 1; 0; 2 ] (unigram_ids t "a b");
  equal ~msg:"a tab" (list int) [ 1; 0; 2 ] (unigram_ids t "a\tb");
  equal ~msg:"a newline" (list int) [ 1; 0; 2 ] (unigram_ids t "a\nb");
  equal ~msg:"offsets"
    (list (pair int int))
    [ (0, 1); (1, 2); (2, 3) ]
    (unigram_offsets t "a b");
  let spaced = unigram ~vocab:(letters @ [ (" ", -5.0) ]) ~unk_id:0 () in
  equal ~msg:"an entry" (list int) [ 1; 3; 2 ] (unigram_ids spaced "a b")

let zwsp = "\u{200b}"
let zwnj = "\u{200c}"

let test_unknown_run () =
  let t = unigram ~vocab:letters ~unk_id:0 () in
  equal ~msg:"one character" (list int) [ 1; 0; 2 ]
    (unigram_ids t ("a" ^ zwsp ^ "b"));
  equal ~msg:"two characters" (list int) [ 1; 0; 2 ]
    (unigram_ids t ("a" ^ zwsp ^ zwnj ^ "b"));
  equal ~msg:"a whole text" (list int) [ 0 ] (unigram_ids t "\u{4e2d}\u{6587}");
  equal ~msg:"two runs" (list int) [ 1; 0; 2; 1; 0 ]
    (unigram_ids t ("a" ^ zwsp ^ "ba" ^ zwsp));
  equal ~msg:"offsets of a run"
    (list (pair int int))
    [ (0, 1); (1, 7); (7, 8) ]
    (unigram_offsets t ("a" ^ zwsp ^ zwnj ^ "b"));
  equal ~msg:"offsets of a run between pairs"
    (list (pair int int))
    [ (0, 1); (1, 2); (2, 8); (8, 9); (9, 10) ]
    (unigram_offsets t ("ab" ^ zwsp ^ zwnj ^ "ab"))

(* A lead byte without its continuation bytes takes in none of the bytes after
   it. *)
let test_stray_byte () =
  let t = unigram ~vocab:letters ~unk_id:0 () in
  equal ~msg:"a lead first" (list int) [ 0; 1; 2 ] (unigram_ids t "\xe9ab");
  equal ~msg:"a lead cut short" (list int) [ 1; 0; 2 ]
    (unigram_ids t "a\xe2\x80b");
  equal ~msg:"offsets"
    (list (pair int int))
    [ (0, 1); (1, 3); (3, 4) ]
    (unigram_offsets t "a\xf0\x9fb")

(* Every byte token of a run stands for the whole run, as in HF. *)
let test_byte_fallback () =
  let bytes ?byte_fallback upto =
    unigram ~vocab:(with_bytes ~upto letters) ~unk_id:0 ?byte_fallback ()
  in
  let t = bytes ~byte_fallback:true 256 in
  equal ~msg:"one character" (list int) [ 1; 229; 131; 142; 2 ]
    (unigram_ids t ("a" ^ zwsp ^ "b"));
  equal ~msg:"a run" (list int)
    [ 1; 229; 131; 142; 229; 131; 143; 2 ]
    (unigram_ids t ("a" ^ zwsp ^ zwnj ^ "b"));
  equal ~msg:"a whole text" (list int) [ 229; 131; 142 ] (unigram_ids t zwsp);
  equal ~msg:"offsets"
    (list (pair int int))
    [ (0, 1); (1, 4); (1, 4); (1, 4); (4, 5) ]
    (unigram_offsets t ("a" ^ zwsp ^ "b"));
  equal ~msg:"offsets of a run"
    (list (pair int int))
    [ (0, 1); (1, 7); (1, 7); (1, 7); (1, 7); (1, 7); (1, 7); (7, 8) ]
    (unigram_offsets t ("a" ^ zwsp ^ zwnj ^ "b"));
  equal ~msg:"a byte without an entry" (list int) [ 1; 0; 2 ]
    (unigram_ids (bytes ~byte_fallback:true 0xE0) ("a" ^ zwsp ^ "b"));
  equal ~msg:"byte_fallback unset" (list int) [ 1; 0; 2 ]
    (unigram_ids (bytes 256) ("a" ^ zwsp ^ "b"))

(* HF looks a fused run up as a string. *)
let test_run_that_is_an_entry () =
  let vocab =
    [ ("<unk>", 0.0); ("a", -1.0); ("b", -1.0); ("<unk><unk>", -1.0) ]
  in
  let t = unigram ~vocab ~unk_id:0 () in
  equal ~msg:"alone" (list int) [ 3 ] (unigram_ids t "<unk><unk>");
  equal ~msg:"between letters" (list int) [ 1; 3; 2 ]
    (unigram_ids t "a<unk><unk>b");
  equal ~msg:"offsets"
    (list (pair int int))
    [ (0, 1); (1, 11); (11, 12) ]
    (unigram_offsets t "a<unk><unk>b");
  let bytes =
    unigram ~vocab:(with_bytes ~upto:256 vocab) ~unk_id:0 ~byte_fallback:true ()
  in
  equal ~msg:"before byte fallback" (list int) [ 3 ]
    (unigram_ids bytes "<unk><unk>")

(* [a b] spends one on its space, and [ab] over pairs on its [b]; where a longer
   piece is the best way into every character nothing is spent. *)
let test_without_unk_id () =
  let t = unigram ~vocab:letters () in
  equal ~msg:"nothing spent" (list int) [ 1; 2 ] (unigram_ids t "ab");
  equal ~msg:"a longer piece first" (list int) [ 0 ]
    (unigram_ids (unigram ~vocab:[ ("ab", -1.0); ("a", -2.0) ] ()) "ab");
  equal ~msg:"a longer piece throughout" (list int) [ 2 ]
    (unigram_ids
       (unigram ~vocab:[ ("ab", -1.0); ("a", -2.0); ("abc", -1.0) ] ())
       "abc");
  raises_match ~msg:"on a space" Exn.failure (fun () -> encode t "a b");
  raises_match ~msg:"into a pair's second character" Exn.failure (fun () ->
      encode (unigram ~vocab:[ ("ab", -1.0) ] ()) "ab")

let test_unk_token_position () =
  let t =
    unigram ~vocab:[ ("<unk>", 0.0); ("ab", -1.0) ] ~unk_token:"<unk>" ()
  in
  equal ~msg:"pairs" (list int) [ 1; 1 ] (unigram_ids t "abab");
  equal ~msg:"an unknown character" (list int) [ 1; 0 ] (unigram_ids t "abc")

(* A vocabulary of whole words holds no piece for a single character, so the way
   into a word's second byte is the unknown token, which the best path, the
   whole word, never spends. *)
let test_whole_words () =
  let t =
    unigram
      ~vocab:[ ("<unk>", 0.0); ("hello", 0.0); ("world", 0.0) ]
      ~unk_id:0 ~unk_token:"<unk>" ~pre:Pre_tokenizer.whitespace ()
  in
  equal (list string) [ "hello"; "world" ] (tokens t "hello world")

let test_unk_id_range () =
  raises_match ~msg:"past the entries" Exn.invalid_arg (fun () ->
      unigram ~vocab:letters ~unk_id:3 ());
  raises_match ~msg:"no entries" Exn.invalid_arg (fun () ->
      unigram ~unk_id:0 ());
  raises_match ~msg:"negative" Exn.invalid_arg (fun () ->
      unigram ~vocab:letters ~unk_id:(-1) ())

let test_positions () =
  let vocab = [ "<unk>"; "<s>"; "</s>"; "\u{2581}hello"; "\u{4e16}\u{754c}" ] in
  let t = unigram ~vocab:(List.map (fun token -> (token, 0.0)) vocab) () in
  List.iteri
    (fun id token ->
      equal ~msg:("token_to_id " ^ token) (option int) (Some id)
        (token_to_id t token);
      equal
        ~msg:(Printf.sprintf "id_to_token %d" id)
        (option string) (Some token) (id_to_token t id))
    vocab

let test_long_pretoken () =
  let t =
    unigram ~vocab:[ ("<unk>", 0.0); ("a", -1.0); ("aa", -1.5) ] ~unk_id:0 ()
  in
  equal ~msg:"seven" (list int) [ 1; 2; 2; 2 ] (unigram_ids t "aaaaaaa");
  equal ~msg:"100 KB" (array int) (Array.make 50_000 2)
    (encode_ids t (String.make 100_000 'a'))

(* chars *)

let bytes_law =
  prop ~count:25 "gives each byte of the text as its id"
    ~examples:[ "abc"; "\u{e9}"; "\u{1f600}"; "\xff\xc3"; "" ]
    (quoted Gen.string) (fun text ->
      cover "a byte that opens no character"
        (List.exists
           (fun c -> String.length c = 1 && c.[0] >= '\x80')
           (characters text));
      equal (array int)
        (Array.init (String.length text) (fun i -> Char.code text.[i]))
        (encode_ids (chars ()) text))

let test_chars_vocab () =
  let bytes = List.init 256 (fun b -> (String.make 1 (Char.chr b), b)) in
  let t = chars () in
  equal ~msg:"vocab"
    (list (pair string int))
    bytes
    (List.sort compare (vocab t));
  equal ~msg:"vocab_size" int 256 (vocab_size t);
  let added = chars ~added_tokens:[ added_token "<s>" ] () in
  equal ~msg:"with an added token"
    (list (pair string int))
    (List.sort compare (("<s>", 256) :: bytes))
    (List.sort compare (vocab added));
  equal ~msg:"its vocab_size" int 257 (vocab_size added)

(* from_model_file *)

(* Every line of a vocab.txt takes the next id, a blank one included, as
   tokenizers' WordPiece::read_file and BERT's load_vocab number them. *)
let line_ids =
  cases "numbers each line of a vocab.txt" ~id:fst
    [
      ("no blank line", ("[UNK]\nhello\nworld\n", 2));
      ("a blank line", ("[UNK]\nhello\n\nworld\n", 3));
      ("two blank lines", ("[UNK]\nhello\n\n\nworld\n", 4));
      ("a line of spaces", ("[UNK]\nhello\n  \nworld\n", 3));
      ("CRLF", ("[UNK]\r\nhello\r\n\r\nworld\r\n", 3));
      ("no final newline", ("[UNK]\nhello\n\nworld", 3));
    ]
    (fun (_, (text, id)) ->
      let t = from_model_file ~vocab:(file text) () in
      equal ~msg:"token_to_id" (option int) (Some id) (token_to_id t "world");
      equal ~msg:"id_to_token" (option string) (Some "world") (id_to_token t id);
      equal ~msg:"encode" (array int) [| id |] (encode_ids t "world"))

(* HF numbers them alike. *)
let test_blank_lines () =
  let t = from_model_file ~vocab:(file "[UNK]\n\n\nworld\n") () in
  equal ~msg:"an earlier blank" (option string) None (id_to_token t 1);
  equal ~msg:"the last blank" (option string) (Some "") (id_to_token t 2);
  equal ~msg:"vocab_size" int 3 (vocab_size t)

let test_bpe_files () =
  let vocab = file {|{"a": 0, "b": 1, "ab": 2, "c": 3, "abc": 4}|} in
  let merges = file "#version: 0.2\na b\nab c\n" in
  let t = from_model_file ~vocab ~merges () in
  equal ~msg:"merged" (list string) [ "abc" ] (tokens t "abc");
  equal ~msg:"the first rule alone" (list string) [ "ab"; "b" ] (tokens t "abb");
  equal ~msg:"ids" (array int) [| 4 |] (encode_ids t "abc")

let test_unreadable () =
  let missing = Filename.concat (temp_dir ()) "missing.txt" in
  raises_match ~msg:"vocab.txt" Exn.sys_error (fun () ->
      from_model_file ~vocab:missing ());
  raises_match ~msg:"merges.txt" Exn.sys_error (fun () ->
      from_model_file ~vocab:(file {|{"a": 0}|}) ~merges:missing ())

let test_malformed () =
  raises_match ~msg:"vocab.json" Exn.failure (fun () ->
      from_model_file ~vocab:(file "[1, 2]") ~merges:(file "") ());
  raises_match ~msg:"a merge line" Exn.failure (fun () ->
      from_model_file ~vocab:(file {|{"a": 0}|}) ~merges:(file "a\n") ());
  (* Hugging Face's BPE.from_file refuses both. *)
  raises_match ~msg:"a fractional id" Exn.failure (fun () ->
      from_model_file ~vocab:(file {|{"a": 1.5}|}) ~merges:(file "") ());
  raises_match ~msg:"an id past 2^31" Exn.failure (fun () ->
      from_model_file ~vocab:(file {|{"c": 1e300}|}) ~merges:(file "") ())

(* HF: BPE.from_file(unk_token="<unk>") encodes "az" as [0, 1], and
   WordPiece.from_file(unk_token="<unk>") as ["<unk>"]. *)
let test_files_unk_token () =
  let bpe_files =
    from_model_file
      ~vocab:(file {|{"a": 0, "<unk>": 1}|})
      ~merges:(file "#version: 0.2\n") ~unk_token:"<unk>" ()
  in
  equal ~msg:"vocab.json" (array int) [| 0; 1 |] (encode_ids bpe_files "az");
  let wordpiece_file =
    from_model_file ~vocab:(file "<unk>\na\n") ~unk_token:"<unk>" ()
  in
  equal ~msg:"vocab.txt" (list string) [ "<unk>" ] (tokens wordpiece_file "az")

let () =
  exit
    (run "Constructors"
       [
         group "t"
           [
             wordpiece_cache;
             unigram_cache;
             wordpiece_domains;
             unigram_domains;
             grown_table;
             held_state;
             freed_states;
           ];
         group "wordpiece"
           [
             greedy_law;
             unknown_word;
             over_budget;
             test "counts a byte that opens no character as one"
               test_stray_bytes;
             test "prefixes later pieces with continuing_subword_prefix"
               test_custom_prefix;
             test "raises Failure at a word when vocab lacks unk_token"
               test_wordpiece_no_unk;
             test "keeps a token given twice as its last pair"
               test_wordpiece_given_twice;
             test "allocates for its tokens, not its largest id" (fun () ->
                 sparse_ids sparse_wordpiece);
             test "refuses an id negative or 2^31 and past" (fun () ->
                 refused_ids sparse_wordpiece);
           ];
         group "word_level"
           [
             test "maps each word to its id and an unknown one to unk_token"
               test_word_ids;
             test "splits on white space and punctuation without pre"
               test_word_level_split;
             test "raises Failure at a word neither vocab nor unk_token holds"
               test_word_level_no_unk;
             test "allocates for its tokens, not its largest id" (fun () ->
                 sparse_ids sparse_word_level);
             test "refuses an id negative or 2^31 and past" (fun () ->
                 refused_ids sparse_word_level);
           ];
         group "unigram"
           [
             best_cut_law;
             ties_law;
             test "takes white space as a character" test_white_space;
             test "is one unknown token for a run of uncovered characters"
               test_unknown_run;
             test "reads a byte that opens no character alone" test_stray_byte;
             test "spells an unknown run in bytes when every byte has an entry"
               test_byte_fallback;
             test
               "takes a fused unknown run whose text is an entry as that entry"
               test_run_that_is_an_entry;
             test
               "raises Failure without unk_id when the best path into a \
                character spends an unknown token"
               test_without_unk_id;
             test "takes unk_token's position as unk_id" test_unk_token_position;
             test
               "encodes whole-word pieces with an unknown token it never spends"
               test_whole_words;
             test "refuses an unk_id that is no position of vocab"
               test_unk_id_range;
             test "numbers entries by their position" test_positions;
             test "encodes a pretoken of 100 KB" test_long_pretoken;
           ];
         group "chars"
           [
             bytes_law;
             test "has the 256 bytes as its vocabulary" test_chars_vocab;
           ];
         group "from_model_file"
           [
             line_ids;
             test "gives the empty token the last blank line's number"
               test_blank_lines;
             test "reads a vocab.json and a merges.txt, skipping #version"
               test_bpe_files;
             test "raises Sys_error for a file it cannot read" test_unreadable;
             test "raises Failure for a file not in its format" test_malformed;
             test "gives the model unk_token" test_files_unk_token;
           ];
       ])
