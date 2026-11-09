(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Brot

let test_wordpiece_basic () =
  (* Create a simple vocabulary *)
  let vocab =
    [
      ("[UNK]", 0);
      ("hello", 1);
      ("world", 2);
      ("##llo", 3);
      ("##rld", 4);
      ("he", 5);
      ("wo", 6);
    ]
  in

  let tokenizer =
    wordpiece ~vocab ~unk_token:"[UNK]" ~continuing_subword_prefix:"##" ()
  in

  (* Test tokenizing a known word *)
  let encoding = encode tokenizer "hello" in
  let tokens = Encoding.tokens encoding in
  equal ~msg:"single token for 'hello'" int 1 (Array.length tokens);
  equal ~msg:"token value" string "hello" tokens.(0);

  Printf.printf "Tokenized 'hello': ";
  Array.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  equal ~msg:"vocabulary size" int 7 (vocab_size tokenizer)

let test_wordpiece_subwords () =
  (* Create vocabulary with subword pieces *)
  let vocab =
    [
      ("[UNK]", 0);
      ("un", 1);
      ("##able", 2);
      ("##happy", 3);
      ("play", 4);
      ("##ing", 5);
      ("##ed", 6);
    ]
  in

  let tokenizer = wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Test word that can be split into subwords *)
  let encoding = encode tokenizer "playing" in
  let tokens = Encoding.tokens encoding in
  Printf.printf "Tokenized 'playing': ";
  Array.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  equal ~msg:"should split into subwords" int 2 (Array.length tokens);
  equal ~msg:"first token" string "play" tokens.(0);
  equal ~msg:"second token" string "##ing" tokens.(1)

let test_wordpiece_unknown () =
  (* Create minimal vocabulary *)
  let vocab = [ ("[UNK]", 0); ("hello", 1) ] in

  let tokenizer = wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Test unknown word *)
  let encoding = encode tokenizer "goodbye" in
  let tokens = Encoding.tokens encoding in
  equal ~msg:"unknown word becomes single token" int 1 (Array.length tokens);
  equal ~msg:"unknown token" string "[UNK]" tokens.(0)

let test_wordpiece_max_chars () =
  (* Create vocabulary *)
  let vocab = [ ("[UNK]", 0); ("test", 1) ] in

  let tokenizer =
    wordpiece ~vocab ~unk_token:"[UNK]" ~max_input_chars_per_word:5 ()
  in

  (* Test word exceeding max chars *)
  let long_word = String.make 10 'a' in
  let encoding = encode tokenizer long_word in
  let tokens = Encoding.tokens encoding in
  equal ~msg:"long word becomes unknown" int 1 (Array.length tokens);
  equal ~msg:"unknown token" string "[UNK]" tokens.(0)

let test_wordpiece_save_load () =
  (* Create vocabulary *)
  let vocab =
    [
      ("[PAD]", 0);
      ("[UNK]", 1);
      ("[CLS]", 2);
      ("[SEP]", 3);
      ("hello", 4);
      ("world", 5);
    ]
  in

  let tokenizer = wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Save the model *)
  let temp_dir = Filename.temp_dir "wordpiece_test" "" in
  let files = save_model_files tokenizer ~folder:temp_dir in

  (* Load the model *)
  let vocab_file = List.find (fun f -> Filename.check_suffix f ".txt") files in
  let loaded_tokenizer = from_model_file ~vocab:vocab_file () in

  (* Test that loaded tokenizer works the same *)
  let original_tokens = encode tokenizer "hello" |> Encoding.tokens in
  let loaded_tokens = encode loaded_tokenizer "hello" |> Encoding.tokens in

  equal ~msg:"same number of tokens" int
    (Array.length original_tokens)
    (Array.length loaded_tokens);

  (* Clean up *)
  List.iter Sys.remove files;
  Unix.rmdir temp_dir

let test_tokenizer_integration () =
  (* Create a WordPiece tokenizer using the high-level API *)
  let vocab =
    [
      ("[PAD]", 0);
      ("[UNK]", 1);
      ("[CLS]", 2);
      ("[SEP]", 3);
      ("hello", 4);
      ("world", 5);
      ("##ing", 6);
    ]
  in
  let tokenizer = wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Test encoding *)
  let tokens = encode tokenizer "hello" |> Encoding.tokens |> Array.to_list in

  Printf.printf "wordpiece result: ";
  List.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  equal ~msg:"tokenizer produces output" bool true (List.length tokens > 0)

let test_wordpiece_greedy_matching () =
  (* Test the greedy longest-match-first algorithm *)
  let vocab =
    [
      ("[UNK]", 0);
      ("un", 1);
      ("able", 2);
      ("unable", 3);
      (* Longer match should be preferred *)
      ("##able", 4);
    ]
  in

  let tokenizer = wordpiece ~vocab ~unk_token:"[UNK]" () in

  (* Should match "unable" as a single token, not "un" + "##able" *)
  let encoding = encode tokenizer "unable" in
  let tokens = Encoding.tokens encoding in
  equal ~msg:"greedy match finds longest token" int 1 (Array.length tokens);
  equal ~msg:"matched full word" string "unable" tokens.(0)

(* Expectations from HuggingFace [models.WordPiece] with the same vocabulary.
   The whole word falls back to the unknown token as soon as one of its pieces
   has no entry, and the character budget counts characters, not bytes. *)
let test_wordpiece_hf_semantics () =
  let vocab =
    [
      ("[UNK]", 0);
      ("un", 1);
      ("##able", 2);
      ("##a", 3);
      ("##b", 4);
      ("a", 5);
      ("\xC3\xA9", 6);
      ("##\xC3\xA9", 7);
      ("play", 8);
      ("##ing", 9);
    ]
  in
  let tokens ~max text =
    wordpiece ~vocab ~unk_token:"[UNK]" ~max_input_chars_per_word:max ()
    |> fun t -> encode t text |> Encoding.tokens |> Array.to_list
  in
  let case ~max text expected =
    equal
      ~msg:(Printf.sprintf "wordpiece max=%d %S" max text)
      (list string) expected (tokens ~max text)
  in
  (* Longest match first, so ["unable"] is two pieces and not four. *)
  case ~max:100 "unable" [ "un"; "##able" ];
  case ~max:100 "unab" [ "un"; "##a"; "##b" ];
  case ~max:100 "playing" [ "play"; "##ing" ];
  (* A word with one unmatchable piece is unknown as a whole, even though its
     prefix matched. *)
  case ~max:100 "unxyz" [ "[UNK]" ];
  (* The budget is in characters: six two-byte characters exceed a budget of
     five, six one-byte characters do too, and five of either do not. *)
  case ~max:5 "aaaaa" [ "a"; "##a"; "##a"; "##a"; "##a" ];
  case ~max:5 "aaaaaa" [ "[UNK]" ];
  case ~max:5 "\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9"
    [ "\xC3\xA9"; "##\xC3\xA9"; "##\xC3\xA9"; "##\xC3\xA9"; "##\xC3\xA9" ];
  case ~max:5 "\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9" [ "[UNK]" ];
  case ~max:100 "\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9"
    [
      "\xC3\xA9";
      "##\xC3\xA9";
      "##\xC3\xA9";
      "##\xC3\xA9";
      "##\xC3\xA9";
      "##\xC3\xA9";
    ];
  (* A word over budget is unknown even when every piece is in the
     vocabulary. *)
  case ~max:5 "unable" [ "[UNK]" ];
  (* A byte that opens no whole character counts as one character, a stray
     continuation byte too: four letters and two of them are over a budget of
     five, though entries cover them. *)
  let stray =
    wordpiece
      ~vocab:[ ("[UNK]", 0); ("a", 1); ("##a", 2); ("##\x80", 3) ]
      ~unk_token:"[UNK]" ~max_input_chars_per_word:5 ()
  in
  equal ~msg:"stray bytes count" (list string) [ "[UNK]" ]
    (Array.to_list (Encoding.tokens (encode stray "aaaa\x80\x80")));
  equal ~msg:"within the budget" (list string) [ "a"; "##a"; "##a"; "##\x80" ]
    (Array.to_list (Encoding.tokens (encode stray "aaa\x80")));
  case ~max:100 "" []

(* Pretoken cache. A Bert word is cut greedily, the longest entry first; a word
   no run of entries covers, or over [max_input_chars_per_word] characters, is
   the unknown token. The drawn words recur, so most are answered from the
   pretoken cache once seen, and some are past the fifteen bytes a cache key
   holds. *)

let cache_vocab =
  [ "[UNK]"; "a"; "ab"; "abc"; "b"; "##a"; "##b"; "##c"; "##bc"; "\u{e9}";
    "##\u{e9}"; "x"; "##x"; "aaaa"; "##aaaa"; "."; "," ]
[@@ocamlformat "disable"]

let cache_words =
  [ "a"; "ab"; "abc"; "abcabc"; "b"; "bca"; "\u{e9}"; "a\u{e9}b"; "xx"; "q";
    "aq"; "aaaaaaaaaaaaaaaaaaa"; "abcabcabcabcabc"; "abcabcabcabcabca";
    "\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}" ]
[@@ocamlformat "disable"]

let cache_max_chars = 12
let cache_id token = Option.get (List.find_index (( = ) token) cache_vocab)

(* The pieces of [word] as (id, start, stop) within it. *)
let reference_pieces word =
  let len = String.length word in
  let chars =
    String.fold_left
      (fun n c -> if Char.code c land 0xC0 = 0x80 then n else n + 1)
      0 word
  in
  let unk = [ (cache_id "[UNK]", 0, len) ] in
  let rec go pos acc =
    if pos >= len then List.rev acc
    else
      let rec longest stop =
        if stop <= pos then None
        else
          let piece = String.sub word pos (stop - pos) in
          let entry = if pos > 0 then "##" ^ piece else piece in
          if List.mem entry cache_vocab then Some (cache_id entry, stop)
          else longest (stop - 1)
      in
      match longest len with
      | None -> unk
      | Some (id, stop) -> go stop ((id, pos, stop) :: acc)
  in
  if chars > cache_max_chars then unk else go 0 []

let cache_text =
  let open Gen in
  let piece =
    frequency
      [ (4, of_list cache_words); (2, of_list [ " "; "  "; "\t" ]);
        (1, of_list [ "."; "," ]) ]
  in
  map (String.concat "") (list ~size:(int_range 0 30) piece)

let cache_tokenizer ?cache_capacity () =
  wordpiece
    ~vocab:(List.mapi (fun i t -> (t, i)) cache_vocab)
    ~max_input_chars_per_word:cache_max_chars ~pre:Pre_tokenizer.bert
    ?cache_capacity ()

(* Bert's words of [text] with their start: runs of letters, and each
   punctuation character alone. *)
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

let cache_props =
  [
    prop ~count:300 "cache: every word cut greedily, seen or not" cache_text
      (fun text ->
        let t = cache_tokenizer () in
        let expected =
          List.concat_map
            (fun (word, start) ->
              List.map
                (fun (id, a, b) -> (id, (start + a, start + b)))
                (reference_pieces word))
            (bert_words text)
        in
        let words = List.map fst (bert_words text) in
        cover ~share:0.3 "a word seen before"
          (List.length (List.sort_uniq compare words) < List.length words);
        cover ~share:0.2 "a word longer than a cache key"
          (List.exists (fun w -> String.length w > 15) words);
        cover ~share:0.1 "an unknown word"
          (List.exists (fun w -> w = "q" || w = "aq") words);
        (* Cold, then answered from the cache, and with no cache at all. *)
        let uncached = cache_tokenizer ~cache_capacity:0 () in
        List.iteri
          (fun round t ->
            let encoding = encode t text in
            equal
              ~msg:(Printf.sprintf "%S round %d" text round)
              (list (pair int (pair int int)))
              expected
              (List.combine
                 (Array.to_list (Encoding.ids encoding))
                 (Array.to_list (Encoding.offsets encoding)));
            equal
              ~msg:(Printf.sprintf "%S ids round %d" text round)
              (list int) (List.map fst expected)
              (Array.to_list (encode_ids t text)))
          [ t; t; uncached ]);
    (* Every letter and pair of letters is an entry, with and without the
       prefix, so that a word of up to eight letters is at most four ids and is
       kept: past 512 distinct ones, as many as the starting table holds and
       twice what makes it grow, it grows while the text is encoded. Every word
       is still found, and cut as with no cache. *)
    prop ~count:20 "cache: grown, a table finds what it held"
      Gen.(
        list ~size:(int_range 1500 2500)
          (string_of ~size:(int_range 1 8) (char_range 'a' 'd')))
      (fun words ->
        let text = String.concat " " words in
        let letters = [ "a"; "b"; "c"; "d" ] in
        let pieces =
          letters
          @ List.concat_map (fun a -> List.map (( ^ ) a) letters) letters
        in
        let vocab =
          ("[UNK]" :: pieces) @ List.map (( ^ ) "##") pieces
          |> List.mapi (fun i t -> (t, i))
        in
        let make ?cache_capacity () =
          wordpiece ~vocab ~pre:Pre_tokenizer.bert ?cache_capacity ()
        in
        let t = make () and uncached = make ~cache_capacity:0 () in
        let expected = encode_ids uncached text in
        cover ~share:0.95 "more than 512 words kept"
          (List.length (List.sort_uniq compare words) > 512);
        equal ~msg:"cold" (array int) expected (encode_ids t text);
        equal ~msg:"warm" (array int) expected (encode_ids t text));
    (* A thread that finds the model's state held makes one of its own. *)
    test "cache: held by another thread" (fun () ->
        let t = cache_tokenizer () in
        let text = "abc abcabc a\u{e9}b q abc ." in
        let expected = encode_ids (cache_tokenizer ~cache_capacity:0 ()) text in
        let ids = ref [||] in
        hold_cache t (fun () ->
            Thread.join (Thread.create (fun () -> ids := encode_ids t text) ()));
        equal ~msg:"while held" (array int) expected !ids;
        equal ~msg:"after" (array int) expected (encode_ids t text));
  ]

(* Vocabulary files *)

let with_vocab_file text f =
  let path = Filename.temp_file "wordpiece_vocab" ".txt" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      Out_channel.with_open_bin path (fun oc ->
          Out_channel.output_string oc text);
      f (from_model_file ~vocab:path ()))

(* Every line of a vocab.txt takes the next id, a blank one included, as
   tokenizers' WordPiece::read_file and BERT's load_vocab number them. *)
let line_ids =
  cases "each line of a vocab.txt takes an id" ~id:fst
    [
      ("no blank line", ("[UNK]\nhello\nworld\n", 2));
      ("a blank line", ("[UNK]\nhello\n\nworld\n", 3));
      ("two blank lines", ("[UNK]\nhello\n\n\nworld\n", 4));
      ("a line of spaces", ("[UNK]\nhello\n  \nworld\n", 3));
      ("CRLF", ("[UNK]\r\nhello\r\n\r\nworld\r\n", 3));
      ("no final newline", ("[UNK]\nhello\n\nworld", 3));
    ]
    (fun (_, (text, id)) ->
      with_vocab_file text @@ fun t ->
      equal ~msg:"token_to_id" (option int) (Some id) (token_to_id t "world");
      equal ~msg:"id_to_token" (option string) (Some "world")
        (id_to_token t id);
      equal ~msg:"encode" (array int) [| id |]
        (Encoding.ids (encode t "world")))

let blank_ids =
  test "only the last blank line holds the empty token" (fun () ->
      with_vocab_file "[UNK]\n\n\nworld\n" @@ fun t ->
      equal ~msg:"an earlier blank" (option string) None (id_to_token t 1);
      equal ~msg:"the last blank" (option string) (Some "") (id_to_token t 2);
      equal ~msg:"vocab_size counts tokens" int 3 (vocab_size t))

let () =
  exit
    (run "WordPiece tests"
       [
         group "basic"
           [
             test "basic tokenization" test_wordpiece_basic;
             test "subword tokenization" test_wordpiece_subwords;
             test "unknown tokens" test_wordpiece_unknown;
             test "max input chars" test_wordpiece_max_chars;
             test "save and load" test_wordpiece_save_load;
             test "tokenizer integration" test_tokenizer_integration;
             test "greedy matching" test_wordpiece_greedy_matching;
             test "HuggingFace semantics" test_wordpiece_hf_semantics;
           ];
         group "cache" cache_props;
         group "vocabulary files" [ line_ids; blank_ids ];
       ])
