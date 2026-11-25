(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Training. A vocabulary or merge list of train_bpe, train_wordpiece or
   train_word_level written out below is what the tokenizers wheel's trainer for
   the same model learns from the same words. train_unigram's are brot's own: it
   runs no EM. *)

open Windtrap
open Brot

let split = Pre_tokenizer.whitespace_split
let byte_level () = Pre_tokenizer.byte_level ~add_prefix_space:false ()
let texts l = `Seq (List.to_seq l)
let tokens t text = Array.to_list (Encoding.tokens (encode t text))

(* [t]'s vocabulary, ordered by id. *)
let vocabulary t =
  List.map fst (List.sort (fun (_, i) (_, j) -> Int.compare i j) (vocab t))

(* The tokens spelled out in [s], separated by spaces. *)
let spelled s = String.split_on_char ' ' s

(* The merges spelled out in [s] as ["left right, left right"]. *)
let merge_list s = List.map String.trim (String.split_on_char ',' s)
let sorted_vocabulary t = List.sort String.compare (List.map fst (vocab t))
let contents tokens = List.map (fun (a : added_token) -> a.content) tokens

let member name = function
  | Jsont.Object (members, _) -> (
      match Jsont.Json.find_mem name members with
      | Some (_, value) -> value
      | None -> failf "no %S member" name)
  | _ -> failf "looking for %S in something that is not an object" name

let json_text json =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json json with
  | Ok text -> text
  | Error msg -> failf "cannot encode: %s" msg

(* [t]'s merges in the order [to_json] writes them, each as ["left right"]. *)
let merges t =
  match member "merges" (member "model" (to_json t)) with
  | Jsont.Array (pairs, _) ->
      List.map
        (function
          | Jsont.Array ([ Jsont.String (l, _); Jsont.String (r, _) ], _) ->
              l ^ " " ^ r
          | _ -> failf "a merge is not a pair of strings")
        pairs
  | _ -> failf "merges is not an array"

let corpus_file contents =
  let path = temp_file ~suffix:".txt" () in
  Out_channel.with_open_bin path (fun oc ->
      Out_channel.output_string oc contents);
  path

let train_corpus =
  [
    "low low low low low lower lower";
    "newest newest newest newest newest newest";
    "widest widest widest lowest";
  ]

(* The characters of [words], each once. *)
let characters words =
  let rec chars w i acc =
    if i >= String.length w then acc
    else
      let n = Uchar.utf_decode_length (String.get_utf_8_uchar w i) in
      chars w (i + n) (String.sub w i n :: acc)
  in
  List.sort_uniq String.compare (List.concat_map (fun w -> chars w 0 []) words)

(* Each trainer, with the arguments these tests vary. *)
let trainers =
  [
    ("train_bpe", fun ?init data -> train_bpe ?init ~pre:split data);
    ("train_wordpiece", fun ?init data -> train_wordpiece ?init ~pre:split data);
    ( "train_word_level",
      fun ?init data -> train_word_level ?init ~pre:split data );
    ("train_unigram", fun ?init data -> train_unigram ?init ~pre:split data);
  ]

(* data *)

let test_line_keeps_newline () =
  (* A CRLF line keeps both bytes, a blank line is a ["\n"] word of its own, and
     a last line without a newline keeps none. *)
  let path = corpus_file "a b\nc d\r\n\ne f" in
  equal (list string)
    [ "\n"; "a b\n"; "c d\r\n"; "e f" ]
    (sorted_vocabulary (train_word_level (`Files [ path ])))

let test_newline_token () =
  let path = corpus_file (String.concat "\n" train_corpus ^ "\n") in
  let t = train_bpe ~pre:(byte_level ()) ~vocab_size:40 (`Files [ path ]) in
  equal (list string)
    (spelled
       "d e i l n o r s t w Ċ Ġ es est lo low Ġlow ew new newest Ġnewest dest \
        idest widest er Ġwidest Ġlower Ġlowest")
    (vocabulary t)

let test_every_file () =
  let first = corpus_file "a\n" and second = corpus_file "b" in
  equal (list string) [ "a\n"; "b" ]
    (sorted_vocabulary (train_word_level (`Files [ first; second ])))

let test_unreadable_file =
  cases "a file that cannot be read raises Sys_error" ~id:fst trainers
    (fun (_, train) ->
      let missing = Filename.concat (temp_dir ()) "missing.txt" in
      raises_match (Exn.sys_error ~sub:"missing.txt") (fun () ->
          train (`Files [ missing ])))

(* init *)

let init () =
  bpe
    ~vocab:[ ("zz", 0) ]
    ~added_tokens:[ added_token ~special:false "<p>" ]
    ~pad_token:"<pad>" ()

let test_init_added_tokens =
  cases "carries init's added tokens into the trained tokenizer" ~id:fst
    trainers (fun (_, train) ->
      equal (list string) [ "<p>"; "<pad>" ]
        (contents (added_tokens (train ~init:(init ()) (texts [ "ab ab" ])))))

let test_init_model =
  cases "starts from an empty vocabulary whatever init's model holds" ~id:fst
    trainers (fun (_, train) ->
      equal (option int) None
        (token_to_id (train ~init:(init ()) (texts [ "ab ab" ])) "zz"))

(* train_bpe *)

let test_bpe_huggingface () =
  (* The corpus runs out of pairs before vocab_size is reached. *)
  let t = train_bpe ~pre:split ~vocab_size:30 (texts train_corpus) in
  equal ~msg:"vocabulary" (list string)
    (spelled
       "d e i l n o r s t w es est lo low ew new newest dest idest widest er \
        lower lowest")
    (vocabulary t);
  equal ~msg:"merges" (list string)
    (merge_list
       "e s, es t, l o, lo w, e w, n ew, new est, d est, i dest, w idest, e r, \
        low er, low est")
    (merges t)

let test_bpe_normalizer () =
  let t =
    train_bpe ~normalizer:Normalizer.lowercase ~pre:split ~vocab_size:20
      (texts [ "LOW Low low LOWER lower" ])
  in
  equal (list string)
    [ "e"; "l"; "o"; "r"; "w"; "lo"; "low"; "er"; "lower" ]
    (vocabulary t)

let test_bpe_whole_text () =
  (* The merges reach across the spaces of a text, and training stops at
     vocab_size. *)
  let t = train_bpe ~vocab_size:30 (texts train_corpus) in
  equal (list string)
    [
      " ";
      "d";
      "e";
      "i";
      "l";
      "n";
      "o";
      "r";
      "s";
      "t";
      "w";
      "es";
      "est";
      "lo";
      "est ";
      "low";
      " low";
      "ew";
      "new";
      "est new";
      " low low";
      "est newest new";
      "dest ";
      "idest ";
      "widest ";
      "er";
      " lower";
      "widest widest ";
      "lowest";
      "low low low";
    ]
    (vocabulary t)

let byte_level_trained =
  lazy
    (train_bpe ~pre:(byte_level ()) ~decoder:Decoder.byte_level ~vocab_size:40
       (texts train_corpus))

let test_bpe_byte_level () =
  let t = Lazy.force byte_level_trained in
  equal ~msg:"vocabulary" (list string)
    (spelled
       "d e i l n o r s t w Ġ es est lo low Ġlow ew new newest Ġnewest dest \
        idest widest er Ġwidest Ġlower Ġlowest")
    (vocabulary t);
  equal ~msg:"merges" (list string)
    (merge_list
       "e s, es t, l o, lo w, Ġ low, e w, n ew, new est, Ġ newest, d est, i \
        dest, w idest, e r, Ġ widest, Ġlow er, Ġlow est")
    (merges t)

let test_bpe_byte_level_encodes () =
  let t = Lazy.force byte_level_trained in
  equal ~msg:"tokens" (list string) [ "low"; "Ġlower" ] (tokens t "low lower");
  equal ~msg:"decode" string "low lower" (decode t (encode_ids t "low lower"))

let test_bpe_min_frequency () =
  (* Of the three pairs only [a a], seen three times, reaches the floor. *)
  let t =
    train_bpe ~pre:split ~vocab_size:20 ~min_frequency:3
      (texts [ "aa aa aa bb cc cc" ])
  in
  equal ~msg:"vocabulary" (list string) [ "a"; "b"; "c"; "aa" ] (vocabulary t);
  equal ~msg:"merges" (list string) [ "a a" ] (merges t)

let test_bpe_limit_alphabet () =
  (* [z] outranks [b], so a limit of two drops [b]. *)
  let corpus = texts [ "aza aza aza ab ab" ] in
  let whole = train_bpe ~pre:split ~vocab_size:20 corpus in
  let limited = train_bpe ~pre:split ~vocab_size:20 ~limit_alphabet:2 corpus in
  equal ~msg:"whole alphabet" (list string)
    [ "a"; "b"; "z"; "az"; "aza"; "ab" ]
    (vocabulary whole);
  equal ~msg:"two characters" (list string) [ "a"; "z"; "az"; "aza" ]
    (vocabulary limited);
  equal ~msg:"ab without its b" (list string) [ "a" ] (tokens limited "ab")

let test_bpe_initial_alphabet () =
  (* ["ét"] adds [é], not the byte ["\xc3"]; the empty entry and ["\xff"], which
     starts no code point, add nothing. *)
  let t =
    train_bpe ~pre:split ~vocab_size:30
      ~initial_alphabet:[ "é"; "ét"; ""; "\xff" ]
      (texts train_corpus)
  in
  equal (list string)
    (spelled
       "d e i l n o r s t w é es est lo low ew new newest dest idest widest er \
        lower lowest")
    (vocabulary t)

let prefixed =
  lazy
    (train_bpe ~pre:split ~vocab_size:30 ~continuing_subword_prefix:"##"
       (texts train_corpus))

let test_bpe_prefix_merges () =
  (* No word here has an [l] after its first character, so ["##l"] is never
     learned. *)
  let t = Lazy.force prefixed in
  equal ~msg:"vocabulary" (list string)
    (spelled
       "d e i l n o r s t w ##o ##w ##e ##r ##s ##t ##i ##d ##es ##est lo low \
        ne ##west newest wi ##dest widest ##er lower")
    (vocabulary t);
  equal ~msg:"merges" (list string)
    (merge_list
       "##e ##s, ##es ##t, l ##o, lo ##w, n ##e, ##w ##est, ne ##west, w ##i, \
        ##d ##est, wi ##dest, ##e ##r, low ##er")
    (merges t)

let test_bpe_prefix_encodes () =
  (* [slow]'s [l] has no prefixed form, so it drops out. *)
  let t = Lazy.force prefixed in
  equal ~msg:"newest" (list string) [ "newest" ] (tokens t "newest");
  equal ~msg:"lowest" (list string) [ "low"; "##est" ] (tokens t "lowest");
  equal ~msg:"slow" (list string) [ "s"; "##o"; "##w" ] (tokens t "slow")

let suffixed =
  lazy
    (train_bpe ~pre:split ~vocab_size:40 ~end_of_word_suffix:"</w>"
       (texts train_corpus))

let test_bpe_suffix_merges () =
  (* tokenizers puts the three suffixed characters in a random order; the merges
     are the same. *)
  let t = Lazy.force suffixed in
  equal ~msg:"vocabulary" (list string)
    (spelled
       "d e i l n o r s t w w</w> r</w> t</w> es est</w> lo west</w> ewest</w> \
        newest</w> low</w> dest</w> idest</w> widest</w> er</w> wer</w> \
        lower</w> lowest</w>")
    (vocabulary t);
  equal ~msg:"merges" (list string)
    (merge_list
       "e s, es t</w>, l o, w est</w>, e west</w>, n ewest</w>, lo w</w>, d \
        est</w>, i dest</w>, w idest</w>, e r</w>, w er</w>, lo wer</w>, lo \
        west</w>")
    (merges t)

let test_bpe_suffix_encodes () =
  let t = Lazy.force suffixed in
  equal ~msg:"low" (list string) [ "low</w>" ] (tokens t "low");
  equal ~msg:"lowest" (list string) [ "lowest</w>" ] (tokens t "lowest");
  equal ~msg:"slow" (list string) [ "s"; "low</w>" ] (tokens t "slow")

(* The limit counts characters, two bytes each here, and the opening merges of
   single characters are exempt from it. *)
let test_bpe_max_token_length =
  cases "holds a merge back once its run reaches max_token_length"
    ~id:(fun (name, _, _, _) -> name)
    [
      ("no limit", None, "αβγδεζηθ", [ "αβγδεζηθ" ]);
      ("5", Some 5, "αβγδεζηθ", [ "αβγδ"; "εζηθ" ]);
      ("2", Some 2, "αβγδεζηθ", [ "αβ"; "γδ"; "εζ"; "ηθ" ]);
      ("4", Some 4, "αβγδε", [ "αβ"; "γδε" ]);
    ]
    (fun (_, max_token_length, word, expected) ->
      let t =
        train_bpe ~pre:split ~vocab_size:40 ?max_token_length
          (texts [ String.concat " " [ word; word; word ] ])
      in
      equal (list string) expected (tokens t word))

let test_bpe_repeated_merge () =
  (* A pair found again after its merge is merged and recorded again; the model
     keeps the rank given last, so [### ##a] is written once. *)
  let t =
    train_bpe ~pre:split ~vocab_size:24 ~continuing_subword_prefix:"##"
      (texts [ "###a b### a" ])
  in
  equal ~msg:"vocabulary" (list string)
    [ "#"; "a"; "b"; "###"; "##a"; "####"; "b##"; "###a"; "b###" ]
    (vocabulary t);
  equal ~msg:"merges" (list string)
    [ "### ###"; "# ####"; "b ####"; "### ##a"; "b## ###" ]
    (merges t)

let test_bpe_pipeline () =
  let normalizer = Normalizer.lowercase and decoder = Decoder.fuse in
  let post = Post_processor.byte_level () in
  let t =
    train_bpe ~normalizer ~pre:split ~post ~decoder ~bos_token:"<s>"
      ~eos_token:"</s>" ~pad_token:"<pad>" (texts [ "ab" ])
  in
  let given = to_json (bpe ~normalizer ~pre:split ~post ~decoder ()) in
  List.iter
    (fun stage ->
      equal ~msg:stage string
        (json_text (member stage given))
        (json_text (member stage (to_json t))))
    [ "normalizer"; "pre_tokenizer"; "post_processor"; "decoder" ];
  equal ~msg:"roles"
    (list (option string))
    [ Some "<s>"; Some "</s>"; Some "<pad>" ]
    [ bos_token t; eos_token t; pad_token t ]

(* tokenizers 0.23.1: Tokenizer(BPE(unk_token="<unk>")) trained with BpeTrainer
   on ["a"] and the special token "<unk>" encodes "za" as [0, 1]. *)
let test_bpe_unk () =
  let t =
    train_bpe ~unk_token:"<unk>"
      ~added_tokens:[ added_token "<unk>" ]
      (texts [ "a" ])
  in
  equal (array int) [| 0; 1 |] (encode_ids t "za")

(* Words over a few characters, one outside the BMP, so that they share pairs
   and a corpus can run out of them. *)
let words =
  let open Gen in
  let character = of_list ~pp:Format.pp_print_string [ "a"; "b"; "c"; "😀" ] in
  list ~size:(int_range 0 30)
    (map (String.concat "") (list ~size:(int_range 1 6) character))

let specials = Gen.subsequence ~pp:Format.pp_print_string [ "<s>"; "</s>" ]

(* A [vocab_size] below the alphabet and its special tokens, at most eight
   above, or further above, a third of the time each. *)
let bpe_case =
  let open Gen in
  let* words = words in
  let* specials = specials in
  let floor = List.length (characters words) + List.length specials in
  let+ vocab_size =
    frequency
      [
        (1, int_range 0 (max 0 (floor - 1)));
        (1, int_range floor (floor + 8));
        (1, int_range (floor + 9) (floor + 60));
      ]
  in
  (words, specials, vocab_size)

let train_words ?(vocab_size = 100_000) specials words =
  train_bpe ~pre:split
    ~added_tokens:(List.map added_token specials)
    ~vocab_size
    (texts [ String.concat " " words ])

let test_bpe_vocab_size =
  prop "stays within vocab_size, special tokens included, keeping the alphabet"
    ~examples:
      [
        ([], [], 0); ([ "abc"; "ab😀" ], [], 0); ([ "abc"; "ab😀" ], [ "<s>" ], 3);
      ]
    bpe_case
    (fun (words, specials, vocab_size) ->
      let floor = List.length (characters words) + List.length specials in
      let full = Brot.vocab_size (train_words specials words) in
      cover "below the alphabet" (vocab_size < floor);
      cover "vocab_size reached" (floor <= vocab_size && vocab_size < full);
      cover "pairs run out" (full < vocab_size);
      equal int
        (max floor (min vocab_size full))
        (Brot.vocab_size (train_words ~vocab_size specials words)))

let test_bpe_spells_words =
  prop "encodes every word of its corpus into tokens that spell it" bpe_case
    (fun (words, specials, vocab_size) ->
      let t = train_words ~vocab_size specials words in
      List.iter
        (fun w -> equal ~msg:w string w (String.concat "" (tokens t w)))
        words)

(* Source: brot.mli, encode: a byte that opens no whole character is read by
   every stage as a character of one byte, and train_bpe merges adjacent
   characters, so [a] and [b] are never adjacent in ["a\xffb"]. *)
let test_bpe_stray_byte () =
  let t = train_bpe (texts [ "a\xffb" ]) in
  equal ~msg:"ab" (option int) None (token_to_id t "ab");
  equal ~msg:"spelled" string "a\xffb" (String.concat "" (tokens t "a\xffb"))

(* train_wordpiece *)

let test_wordpiece_subwords () =
  let t = train_wordpiece ~pre:split ~vocab_size:30 (texts train_corpus) in
  equal (list string) [ "newest"; "low"; "##est" ] (tokens t "newest lowest")

let test_wordpiece_prefix () =
  let t =
    train_wordpiece ~pre:split ~vocab_size:30 ~continuing_subword_prefix:"@@"
      (texts train_corpus)
  in
  equal (list string) [ "newest"; "low"; "@@est" ] (tokens t "newest lowest")

(* tokenizers 0.23.1's WordPieceTrainer keeps the caller's unknown token and
   adds none of its own: 13 entries below, the alphabet of 11 ([a]-[f] and
   [##b]-[##f]) and the two special tokens. *)
let test_wordpiece_unk () =
  let t =
    train_wordpiece ~pre:split ~unk_token:"<unk>"
      ~added_tokens:[ added_token "<unk>" ]
      (texts [ "ab ab" ])
  in
  equal ~msg:"z ab" (list string) [ "<unk>"; "ab" ] (tokens t "z ab");
  equal ~msg:"[UNK]" (option int) None (token_to_id t "[UNK]")

let test_wordpiece_vocab_size () =
  let corpus = texts [ "abcdef abcdef abc" ] in
  let added_tokens = [ added_token "<s>"; added_token "</s>" ] in
  let t = train_wordpiece ~pre:split ~added_tokens ~vocab_size:8 corpus in
  let bpe =
    train_bpe ~pre:split ~added_tokens ~vocab_size:8
      ~continuing_subword_prefix:"##" corpus
  in
  equal ~msg:"vocabulary" (list string) (vocabulary bpe) (vocabulary t);
  equal ~msg:"vocab_size" int 13 (vocab_size t)

let wordpiece_case =
  let open Gen in
  let+ words = words
  and+ vocab_size = int_range 0 40
  and+ min_frequency = int_range 0 3
  and+ limit_alphabet = option (int_range 1 4)
  and+ initial_alphabet =
    subsequence ~pp:Format.pp_print_string [ "e"; "😀"; "x" ]
  and+ prefix = option (constant ~pp:Format.pp_print_string "@@") in
  (words, vocab_size, min_frequency, limit_alphabet, initial_alphabet, prefix)

let test_wordpiece_is_bpe_training =
  prop "learns the vocabulary train_bpe learns over the same words"
    wordpiece_case
    (fun
      ( words,
        vocab_size,
        min_frequency,
        limit_alphabet,
        initial_alphabet,
        prefix )
    ->
      let corpus = texts [ String.concat " " words ] in
      let wordpiece =
        train_wordpiece ~pre:split ~vocab_size ~min_frequency ?limit_alphabet
          ~initial_alphabet ?continuing_subword_prefix:prefix corpus
      in
      let bpe =
        train_bpe ~pre:split ~vocab_size ~min_frequency ?limit_alphabet
          ~initial_alphabet
          ~continuing_subword_prefix:(Option.value prefix ~default:"##")
          corpus
      in
      cover "a merge learned" (merges bpe <> []);
      let entries t = List.sort compare (vocab t) in
      equal (list (pair string int)) (entries bpe) (entries wordpiece))

(* train_word_level *)

let test_word_level_pieces () =
  let t =
    train_word_level ~pre:(byte_level ()) ~vocab_size:20
      (texts [ "a a a b b c" ])
  in
  equal (list string) [ "a"; "Ġa"; "Ġb"; "Ġc" ] (sorted_vocabulary t)

let test_word_level_whole_text () =
  let t = train_word_level ~vocab_size:20 (texts [ "a a a b b c" ]) in
  equal (list string) [ "a a a b b c" ] (vocabulary t)

let test_word_level_specials () =
  (* Of the three words one drops out to leave room for the two special
     tokens. *)
  let t =
    train_word_level ~pre:split
      ~added_tokens:(List.map added_token [ "[UNK]"; "[CLS]" ])
      ~vocab_size:4 (texts [ "a a a b b c" ])
  in
  equal (list string) [ "[UNK]"; "[CLS]"; "a"; "b" ] (vocabulary t)

(* tokenizers 0.23.1: WordLevel(unk_token="[UNK]") trained with
   WordLevelTrainer(special_tokens=["[UNK]"]) on "a a b" gives "a zz" the tokens
   a and [UNK]. *)
let test_word_level_unk () =
  let t =
    train_word_level ~pre:split ~unk_token:"[UNK]"
      ~added_tokens:[ added_token "[UNK]" ]
      (texts [ "a a b" ])
  in
  equal (list string) [ "a"; "[UNK]" ] (tokens t "a zz")

let test_word_level_min_frequency () =
  let t =
    train_word_level ~pre:split ~min_frequency:2 (texts [ "a a a b b c" ])
  in
  equal (list string) [ "a"; "b" ] (vocabulary t)

(* tokenizers 0.23.1's WordLevelTrainer keeps w00, w01 and w02. *)
let test_word_level_ties () =
  let words = String.concat " " (List.init 30 (Printf.sprintf "w%02d")) in
  let t = train_word_level ~pre:split ~vocab_size:3 (texts [ words ]) in
  equal (list string) [ "w00"; "w01"; "w02" ] (vocabulary t)

let word_level_case =
  let open Gen in
  let+ words =
    list ~size:(int_range 0 30)
      (of_list ~pp:Format.pp_print_string
         [ "a"; "b"; "c"; "d"; "e"; "ab"; "ba" ])
  and+ specials = specials
  and+ vocab_size = int_range 0 10
  and+ min_frequency = int_range 0 4 in
  (words, specials, vocab_size, min_frequency)

let counts words =
  List.fold_left
    (fun acc w ->
      let n = Option.value (List.assoc_opt w acc) ~default:0 in
      (w, n + 1) :: List.remove_assoc w acc)
    [] words

(* The law holds whatever order equally frequent words are kept in. *)
let test_word_level_most_frequent =
  prop "keeps the most frequent words, special tokens counted in vocab_size"
    word_level_case (fun (words, specials, vocab_size, min_frequency) ->
      let t =
        train_word_level ~pre:split
          ~added_tokens:(List.map added_token specials)
          ~vocab_size ~min_frequency
          (texts [ String.concat " " words ])
      in
      let eligible =
        List.filter (fun (_, n) -> n >= min_frequency) (counts words)
      in
      let budget = max 0 (vocab_size - List.length specials) in
      let kept =
        List.filter (fun w -> not (List.mem w specials)) (vocabulary t)
      in
      let dropped =
        List.filter (fun (w, _) -> not (List.mem w kept)) eligible
      in
      cover "a word cut by vocab_size" (List.length eligible > budget);
      cover "a word cut by min_frequency"
        (List.length eligible < List.length (counts words));
      List.iter
        (fun w -> mem ~msg:"a kept word" string w (List.map fst eligible))
        kept;
      equal ~msg:"words kept" int
        (min budget (List.length eligible))
        (List.length kept);
      let count w = Option.value (List.assoc_opt w eligible) ~default:0 in
      let least_kept =
        List.fold_left (fun m w -> min m (count w)) max_int kept
      in
      List.iter
        (fun (w, n) ->
          at_least
            ~msg:("as frequent as the dropped " ^ w)
            int ~than:n least_kept)
        dropped)

(* train_unigram *)

let test_unigram_words () =
  let t =
    train_unigram ~pre:(byte_level ()) ~vocab_size:20
      (texts [ "low lower low" ])
  in
  equal (list string) [ "low"; "Ġlow"; "Ġlower" ] (sorted_vocabulary t)

let unk = added_token "<unk>"

let test_unigram_unk =
  cases
    "spells an unseen word as unk_token when added_tokens or init carries it"
    ~id:fst
    [
      ( "added_tokens",
        fun data ->
          train_unigram ~pre:split ~unk_token:"<unk>" ~added_tokens:[ unk ] data
      );
      ( "init",
        fun data ->
          train_unigram ~pre:split ~unk_token:"<unk>"
            ~init:(bpe ~added_tokens:[ unk ] ())
            data );
    ]
    (fun (_, train) ->
      let t = train (texts [ "ab ab c" ]) in
      equal (list string) [ "<unk>"; "ab" ] (tokens t "zz ab"))

let test_unigram_no_unk =
  cases "raises Failure on an unseen word without an unknown entry" ~id:fst
    [
      ("no unk_token", fun data -> train_unigram ~pre:split data);
      ( "unk_token not carried",
        fun data -> train_unigram ~pre:split ~unk_token:"<unk>" data );
    ]
    (fun (_, train) ->
      let t = train (texts [ "ab ab c" ]) in
      raises_match (Exn.failure ~sub:"") (fun () -> encode t "zz ab"))

let () =
  exit
    (run "Training"
       [
         group "data"
           [
             test "a line of a file keeps the newline that ends it"
               test_line_keeps_newline;
             test "a byte-level pipeline learns a token for that newline"
               test_newline_token;
             test "the lines of every file are counted" test_every_file;
             test_unreadable_file;
           ];
         group "init" [ test_init_added_tokens; test_init_model ];
         group "train_bpe"
           [
             test "learns the vocabulary and merges tokenizers learns"
               test_bpe_huggingface;
             test "counts the words of the normalized text" test_bpe_normalizer;
             test "counts a whole text as one word without a pre-tokenizer"
               test_bpe_whole_text;
             test "learns over the encoded form of a byte-level pre-tokenizer"
               test_bpe_byte_level;
             test "encodes a text into the pieces it learned, which decode back"
               test_bpe_byte_level_encodes;
             test "stops when the best pair falls below min_frequency"
               test_bpe_min_frequency;
             test "drops the rarest characters past limit_alphabet"
               test_bpe_limit_alphabet;
             test "keeps the code point each initial_alphabet entry starts with"
               test_bpe_initial_alphabet;
             test "learns merges over the continuing_subword_prefix forms"
               test_bpe_prefix_merges;
             test "prefixes every piece of a word but the first"
               test_bpe_prefix_encodes;
             test "learns merges over the end_of_word_suffix form"
               test_bpe_suffix_merges;
             test "ends a word's last piece in end_of_word_suffix"
               test_bpe_suffix_encodes;
             test_bpe_max_token_length;
             test "merges and records again a pair found after its merge"
               test_bpe_repeated_merge;
             test "takes the pipeline and role markers as bpe does"
               test_bpe_pipeline;
             test "gives the model unk_token's unknown handling" test_bpe_unk;
             test_bpe_vocab_size;
             test_bpe_spells_words;
             test "reads a byte that opens no character as a character"
               test_bpe_stray_byte;
           ];
         group "train_wordpiece"
           [
             test "keeps the merged tokens as ##-prefixed subwords"
               test_wordpiece_subwords;
             test "prefixes subwords with continuing_subword_prefix"
               test_wordpiece_prefix;
             test "spells an unseen word as unk_token" test_wordpiece_unk;
             test "adds no token past what train_bpe learns"
               test_wordpiece_vocab_size;
             test_wordpiece_is_bpe_training;
           ];
         group "train_word_level"
           [
             test "keeps each byte-level piece whole" test_word_level_pieces;
             test "counts a whole text as one word without a pre-tokenizer"
               test_word_level_whole_text;
             test "numbers the words after the special tokens"
               test_word_level_specials;
             test "spells an unseen word as unk_token" test_word_level_unk;
             test "leaves out a word below min_frequency"
               test_word_level_min_frequency;
             test "keeps the first of equally frequent words in byte order"
               test_word_level_ties;
             test_word_level_most_frequent;
           ];
         group "train_unigram"
           [
             test "keeps the most frequent words whole" test_unigram_words;
             test_unigram_unk;
             test_unigram_no_unk;
           ];
       ])
