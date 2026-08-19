(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Brot

let tokens tokenizer text =
  encode tokenizer text |> Encoding.tokens |> Array.to_list

let test_bpe_basic () =
  (* Create a simple vocabulary and merges *)
  let vocab =
    [
      ("h", 0);
      ("e", 1);
      ("l", 2);
      ("o", 3);
      ("ll", 4);
      ("he", 5);
      ("llo", 6);
      ("hello", 7);
    ]
  in

  let merges =
    [
      ("l", "l");
      (* rank 0: Merge 'l' + 'l' -> 'll' *)
      ("ll", "o");
      (* rank 1: Merge 'll' + 'o' -> 'llo' *)
      ("he", "llo");
      (* rank 2: Merge 'he' + 'llo' -> 'hello' *)
    ]
  in

  let tokenizer = bpe ~vocab ~merges ~unk_token:"<unk>" () in

  let encoding = encode tokenizer "hello" in
  let tokens = Encoding.tokens encoding |> Array.to_list in

  Printf.printf "Tokenized 'hello': ";
  List.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  equal ~msg:"vocabulary size" int 8 (vocab_size tokenizer)

let test_bpe_builder () =
  let vocab = [ ("a", 0); ("b", 1); ("ab", 2) ] in
  let merges = [ ("a", "b") ] in

  let tokenizer = bpe ~vocab ~merges ~cache_capacity:50 () in

  let encoding = encode tokenizer "ab" in
  let tokens = Encoding.tokens encoding in
  equal ~msg:"single token for 'ab'" int 1 (Array.length tokens)

let test_ignore_merges () =
  (* ["ab"] is in the vocabulary but no merge builds it, so the merges alone
     cannot produce it; [ignore_merges] is what makes the whole word win. *)
  let vocab = [ ("a", 0); ("b", 1); ("c", 2); ("bc", 3); ("ab", 4) ] in
  let merges = [ ("b", "c") ] in
  let merging = bpe ~vocab ~merges () in
  let ignoring = bpe ~vocab ~merges ~ignore_merges:true () in
  equal ~msg:"merges decide 'ab'" (list string) [ "a"; "b" ]
    (tokens merging "ab");
  equal ~msg:"ignore_merges keeps 'ab'" (list string) [ "ab" ]
    (tokens ignoring "ab");
  (* A word absent from the vocabulary is merged either way. *)
  equal ~msg:"merges build 'abc'" (list string) [ "a"; "bc" ]
    (tokens merging "abc");
  equal ~msg:"ignore_merges still merges 'abc'" (list string) [ "a"; "bc" ]
    (tokens ignoring "abc")

let test_dropout_overrides_ignore_merges () =
  (* Dropout is drawn per occurrence, so the whole-word shortcut cannot stand in
     for the merges; at probability 1 none of them apply. *)
  let vocab = [ ("a", 0); ("b", 1); ("ab", 2) ] in
  let merges = [ ("a", "b") ] in
  let tokenizer = bpe ~vocab ~merges ~ignore_merges:true ~dropout:1.0 () in
  equal ~msg:"dropout leaves 'ab' unmerged" (list string) [ "a"; "b" ]
    (encode tokenizer "ab" |> Encoding.tokens |> Array.to_list)

let test_empty_affixes () =
  (* Tokenizer files spell "no prefix" and "no suffix" as [""]. *)
  let vocab = [ ("a", 0); ("b", 1); ("ab", 2) ] in
  let merges = [ ("a", "b") ] in
  let tokenizer =
    bpe ~vocab ~merges ~continuing_subword_prefix:"" ~end_of_word_suffix:"" ()
  in
  equal ~msg:"'ab' merges with empty affixes" (list string) [ "ab" ]
    (encode tokenizer "ab" |> Encoding.tokens |> Array.to_list)

(* The suffix goes on the last character of a word, the prefix on every
   character but the first, so a one-character word takes the suffix alone.
   Expectations probed with the [tokenizers] wheel, e.g.
   [BPE({a,b,a</w>,b</w>,ab</w>}, [(a, b</w>)], end_of_word_suffix="</w>")]
   gives ["a" -> a</w>] and ["ba" -> b, a</w>]. *)
let test_suffix_only () =
  let vocab =
    [ ("a", 0); ("b", 1); ("a</w>", 2); ("b</w>", 3); ("ab</w>", 4) ]
  in
  let merges = [ ("a", "b</w>") ] in
  let tokenizer = bpe ~vocab ~merges ~end_of_word_suffix:"</w>" () in
  equal ~msg:"one character takes the suffix" (list string) [ "a</w>" ]
    (tokens tokenizer "a");
  equal ~msg:"last character takes the suffix" (list string) [ "b"; "a</w>" ]
    (tokens tokenizer "ba");
  equal ~msg:"merge across the suffixed character" (list string) [ "ab</w>" ]
    (tokens tokenizer "ab");
  equal ~msg:"merge inside a longer word" (list string) [ "a"; "ab</w>" ]
    (tokens tokenizer "aab")

let test_suffix_and_merges () =
  (* The suffix is part of the character before any merge runs, so a merge rule
     reaches the last character only if it names the suffixed token. Probed:
     merges [(a, b)] give ["ab" -> a, b</w>] and ["aba" -> ab, a</w>], merges
     [(a, b</w>)] give ["ab" -> ab</w>] and ["aba" -> a, b, a</w>]. *)
  let vocab =
    [ ("a", 0); ("b", 1); ("ab", 2); ("a</w>", 3); ("b</w>", 4); ("ab</w>", 5) ]
  in
  let plain = bpe ~vocab ~merges:[ ("a", "b") ] ~end_of_word_suffix:"</w>" () in
  let suffixed =
    bpe ~vocab ~merges:[ ("a", "b</w>") ] ~end_of_word_suffix:"</w>" ()
  in
  equal ~msg:"unsuffixed merge misses the last character" (list string)
    [ "a"; "b</w>" ] (tokens plain "ab");
  equal ~msg:"unsuffixed merge applies before it" (list string)
    [ "ab"; "a</w>" ] (tokens plain "aba");
  equal ~msg:"suffixed merge reaches the last character" (list string)
    [ "ab</w>" ] (tokens suffixed "ab");
  equal ~msg:"suffixed merge applies nowhere else" (list string)
    [ "a"; "b"; "a</w>" ] (tokens suffixed "aba")

let test_suffix_multibyte () =
  let vocab = [ ("a", 0); ("é", 1); ("a</w>", 2); ("é</w>", 3) ] in
  let tokenizer = bpe ~vocab ~merges:[] ~end_of_word_suffix:"</w>" () in
  equal ~msg:"one multi-byte character" (list string) [ "é</w>" ]
    (tokens tokenizer "é");
  equal ~msg:"multi-byte character last" (list string) [ "a"; "é</w>" ]
    (tokens tokenizer "aé");
  equal ~msg:"multi-byte character first" (list string) [ "é"; "a</w>" ]
    (tokens tokenizer "éa")

let test_suffix_unknown () =
  (* ["a</w>"] is absent, so a word ending in ['a'] falls back to the unknown
     token; the plain ["a"] entry does not stand in for it. *)
  let vocab = [ ("<unk>", 0); ("a", 1); ("b", 2); ("b</w>", 3) ] in
  let tokenizer =
    bpe ~vocab ~merges:[] ~end_of_word_suffix:"</w>" ~unk_token:"<unk>" ()
  in
  equal ~msg:"suffixed form missing" (list string) [ "<unk>" ]
    (tokens tokenizer "a");
  equal ~msg:"unsuffixed form still found" (list string) [ "a"; "b</w>" ]
    (tokens tokenizer "ab");
  equal ~msg:"missing suffixed form at the end" (list string) [ "b"; "<unk>" ]
    (tokens tokenizer "ba");
  (* The reverse: ["a</w>"] is there but the bare ["a"] is not, so ['a'] is
     unknown anywhere but at the end. Probed: ["ab" -> <unk>, b</w>]. *)
  let ending =
    bpe
      ~vocab:[ ("<unk>", 0); ("b", 1); ("a</w>", 2); ("b</w>", 3) ]
      ~merges:[] ~end_of_word_suffix:"</w>" ~unk_token:"<unk>" ()
  in
  equal ~msg:"bare form missing" (list string) [ "<unk>"; "b</w>" ]
    (tokens ending "ab");
  equal ~msg:"suffixed form found at the end" (list string) [ "a</w>" ]
    (tokens ending "a")

let test_suffix_byte_fallback () =
  (* Byte fallback spells out the affixed character, suffix bytes included. *)
  let vocab =
    [ ("a", 0); ("b", 1) ]
    @ List.init 96 (fun i -> (Printf.sprintf "<0x%02X>" (0x20 + i), 2 + i))
  in
  let tokenizer =
    bpe ~vocab ~merges:[] ~end_of_word_suffix:"</w>" ~byte_fallback:true ()
  in
  equal ~msg:"one character falls back with its suffix" (list string)
    [ "<0x61>"; "<0x3C>"; "<0x2F>"; "<0x77>"; "<0x3E>" ]
    (tokens tokenizer "a");
  equal ~msg:"only the last character carries the suffix" (list string)
    [ "a"; "<0x62>"; "<0x3C>"; "<0x2F>"; "<0x77>"; "<0x3E>" ]
    (tokens tokenizer "ab")

let test_prefix_byte_fallback () =
  (* Probed: with only ["a"] in the vocabulary, ["ba" -> <0x62>, <0x23>, <0x23>,
     <0x61>] and ["ab" -> a, <0x23>, <0x23>, <0x62>] — the prefix bytes are part
     of what falls back. *)
  let vocab =
    [ ("a", 0) ]
    @ List.init 96 (fun i -> (Printf.sprintf "<0x%02X>" (0x20 + i), 1 + i))
  in
  let tokenizer =
    bpe ~vocab ~merges:[] ~continuing_subword_prefix:"##" ~byte_fallback:true ()
  in
  equal ~msg:"first character falls back bare" (list string)
    [ "<0x62>"; "<0x23>"; "<0x23>"; "<0x61>" ]
    (tokens tokenizer "ba");
  equal ~msg:"later characters fall back with the prefix" (list string)
    [ "a"; "<0x23>"; "<0x23>"; "<0x62>" ]
    (tokens tokenizer "ab")

(* Byte fallback spells out the affixed character, so a two-byte prefix turns
   one source byte into three symbols and the merge buffers have to be sized for
   the symbols rather than for the bytes. Probed with the [tokenizers] wheel:
   seven characters give 19 tokens and forty give 118. *)
let test_prefix_byte_fallback_long () =
  let vocab =
    ("a", 0)
    :: List.init 96 (fun i -> (Printf.sprintf "<0x%02X>" (0x20 + i), 1 + i))
  in
  let tokenizer =
    bpe ~vocab ~merges:[] ~continuing_subword_prefix:"##" ~byte_fallback:true ()
  in
  (* Only the first character is bare; every one after it carries the prefix,
     and all four of its bytes fall back. *)
  let spelled first rest n =
    first
    :: List.concat_map
         (fun i -> [ "<0x23>"; "<0x23>"; rest i ])
         (List.init (n - 1) Fun.id)
  in
  let b _ = "<0x62>" in
  equal ~msg:"seven characters" (list string) (spelled "<0x62>" b 7)
    (tokens tokenizer (String.make 7 'b'));
  equal ~msg:"forty characters" (list string) (spelled "<0x62>" b 40)
    (tokens tokenizer (String.make 40 'b'));
  (* ["a"] is in the vocabulary, so the first character is a token of its own
     and only the prefixed ones fall back. *)
  equal ~msg:"alternating characters" (list string)
    (spelled "a" (fun i -> if i mod 2 = 0 then "<0x62>" else "<0x61>") 7)
    (tokens tokenizer "abababa")

let test_prefix_and_suffix () =
  let vocab =
    [
      ("a", 0);
      ("b", 1);
      ("c", 2);
      ("##a", 3);
      ("##b", 4);
      ("##c", 5);
      ("a</w>", 6);
      ("b</w>", 7);
      ("c</w>", 8);
      ("##a</w>", 9);
      ("##b</w>", 10);
      ("##c</w>", 11);
    ]
  in
  let both =
    bpe ~vocab ~merges:[] ~continuing_subword_prefix:"##"
      ~end_of_word_suffix:"</w>" ()
  in
  equal ~msg:"one character: suffix, no prefix" (list string) [ "a</w>" ]
    (tokens both "a");
  equal ~msg:"last character: prefix and suffix" (list string)
    [ "a"; "##b</w>" ] (tokens both "ab");
  equal ~msg:"middle character: prefix only" (list string)
    [ "a"; "##b"; "##c</w>" ] (tokens both "abc");
  let prefix_only = bpe ~vocab ~merges:[] ~continuing_subword_prefix:"##" () in
  equal ~msg:"prefix alone leaves one character bare" (list string) [ "a" ]
    (tokens prefix_only "a");
  equal ~msg:"prefix alone on every following character" (list string)
    [ "a"; "##b"; "##c" ] (tokens prefix_only "abc")

let test_bpe_save_load () =
  let vocab = [ ("t", 0); ("e", 1); ("s", 2); ("test", 3) ] in
  let merges = [] in
  (* No merges for simplicity *)

  let tokenizer = bpe ~vocab ~merges () in

  (* Save the model *)
  let temp_dir = Filename.temp_dir "bpe_test" "" in
  let files = save_model_files tokenizer ~folder:temp_dir () in

  (* Load the model *)
  let vocab_file = List.find (fun f -> Filename.check_suffix f ".json") files in
  let merges_file = List.find (fun f -> Filename.check_suffix f ".txt") files in
  let loaded_tokenizer =
    from_model_file ~vocab:vocab_file ~merges:merges_file ()
  in

  (* Test that loaded tokenizer works the same *)
  let original_tokens = encode tokenizer "test" |> Encoding.tokens in
  let loaded_tokens = encode loaded_tokenizer "test" |> Encoding.tokens in

  equal ~msg:"same number of tokens" int
    (Array.length original_tokens)
    (Array.length loaded_tokens);

  (* Clean up *)
  List.iter Sys.remove files;
  Unix.rmdir temp_dir

let test_tokenizer_integration () =
  (* Create a BPE tokenizer using the high-level API *)
  let vocab =
    [
      ("h", 0); ("e", 1); ("l", 2); ("o", 3); ("he", 4); ("llo", 5); ("hello", 6);
    ]
  in
  let merges = [ ("h", "e"); ("he", "llo") ] in
  let tokenizer = bpe ~vocab ~merges () in

  (* Test encoding *)
  let tokens = encode tokenizer "hello" |> Encoding.tokens |> Array.to_list in

  Printf.printf "bpe result: ";
  List.iter (Printf.printf "%s ") tokens;
  Printf.printf "\n";

  equal ~msg:"tokenizer produces output" bool true (List.length tokens > 0)

(* Without an unknown token or byte fallback a character absent from the
   vocabulary contributes no symbol, so a word can reach the merges empty. *)
let test_unknown_character () =
  let vocab = [ ("a", 0); ("b", 1); ("ab", 2) ] in
  let merges = [ ("a", "b") ] in
  let tokenizer = bpe ~vocab ~merges () in
  let tokens text = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"unknown character alone" (list string) [] (tokens "z");
  equal ~msg:"merged word" (list string) [ "ab" ] (tokens "ab");
  (* The merge buffers are reused, so an empty word must not pick up the symbols
     of the word merged before it. *)
  equal ~msg:"unknown character after a merged word" (list string) []
    (tokens "z");
  equal ~msg:"known character before an unknown one" (list string) [ "a" ]
    (tokens "az")

(* Every domain builds a pretoken cache of its own, seeded from the same
   vocabulary. These 169 words share the single two-way set of the smallest
   cache, so each domain rewrites that set continuously while the others read
   theirs. *)
let test_parallel_cache () =
  let letter i = Char.chr (Char.code 'a' + i) in
  let vocab =
    List.init 26 (fun i -> (String.make 1 (letter i), i))
    @ List.init 26 (fun i -> (Printf.sprintf "a%c" (letter i), 26 + i))
  in
  let merges = List.init 26 (fun i -> ("a", String.make 1 (letter i))) in
  let tokenizer = bpe ~vocab ~merges ~cache_capacity:1 () in
  let words =
    Array.init 169 (fun k ->
        Printf.sprintf "a%ca%c" (letter (k / 13)) (letter (k mod 13)))
  in
  let ids text = encode tokenizer text |> Encoding.ids in
  let expected = Array.map ids words in
  let mismatches = Atomic.make 0 in
  let hammer () =
    for _ = 1 to 100 do
      Array.iteri
        (fun i word -> if ids word <> expected.(i) then Atomic.incr mismatches)
        words
    done
  in
  let domains = Array.init 3 (fun _ -> Domain.spawn hammer) in
  hammer ();
  Array.iter Domain.join domains;
  equal ~msg:"parallel tokenization agrees with single-domain" int 0
    (Atomic.get mismatches)

(* The pretoken cache. Every expectation here is that the cache changes nothing:
   a two-way set-associative table answers a pretoken from whichever way of its
   one set holds it, so a collision, an eviction, a seeded entry and a merge run
   from scratch must all agree. *)

(* Eight letters, every two-letter merge, one four-letter merge per letter, and
   a three-letter entry per letter that no merge builds — the case where a
   vocabulary entry is not its own tokenization. *)
let collide_vocab, collide_merges =
  let letters =
    List.init 8 (fun i -> String.make 1 (Char.chr (Char.code 'a' + i)))
  in
  let vocab = ref [] and merges = ref [] and next = ref 0 in
  let add token =
    vocab := (token, !next) :: !vocab;
    incr next
  in
  List.iter add letters;
  List.iter
    (fun a ->
      List.iter
        (fun b ->
          add (a ^ b);
          merges := (a, b) :: !merges)
        letters)
    letters;
  List.iter
    (fun a ->
      let pair = a ^ a in
      add (pair ^ pair);
      merges := (pair, pair) :: !merges)
    letters;
  List.iter (fun a -> add (a ^ a ^ a)) letters;
  (List.rev !vocab, List.rev !merges)

let collide_words =
  let state = Random.State.make [| 20260817 |] in
  Array.init 400 (fun i ->
      let len = 1 + (i mod 40) in
      String.init len (fun _ ->
          Char.chr (Char.code 'a' + Random.State.int state 8)))

let collide ?(ignore_merges = false) cache_capacity =
  bpe ~vocab:collide_vocab ~merges:collide_merges ~cache_capacity ~ignore_merges
    ()

let test_cache_agrees_with_merges () =
  let uncached = collide 0 in
  let expected = Array.map (fun w -> encode_ids uncached w) collide_words in
  (* One set: every pretoken collides with every other, so no answer can come
     from a stale entry and survive. *)
  List.iter
    (fun capacity ->
      let tokenizer = collide capacity in
      (* Twice over: the first pass inserts, the second reads back. *)
      for pass = 1 to 2 do
        Array.iteri
          (fun i word ->
            equal
              ~msg:(Printf.sprintf "%d entries, pass %d, %S" capacity pass word)
              (array int) expected.(i)
              (encode_ids tokenizer word))
          collide_words
      done)
    [ 1; 16; 4096 ]

(* The cache is seeded with the tokenization of every vocabulary entry, run
   through the very function a miss runs. An entry no merge builds — ["aaa"]
   here — must come out as its decomposition, not as itself. *)
let test_seed_agrees_with_merges () =
  let uncached = collide 0 in
  let cached = collide 4096 in
  List.iter
    (fun (token, _) ->
      equal
        ~msg:(Printf.sprintf "seeded %S" token)
        (array int)
        (encode_ids uncached token)
        (encode_ids cached token))
    collide_vocab;
  equal ~msg:"an unreachable entry decomposes" (list string) [ "aa"; "a" ]
    (tokens cached "aaa")

(* Under [ignore_merges] the seed is the entry itself, and a pretoken that
   collides with it in the table must not take its place. *)
let test_seed_ignore_merges () =
  let uncached = collide ~ignore_merges:true 0 in
  let cached = collide ~ignore_merges:true 1 in
  equal ~msg:"an unreachable entry stands" (list string) [ "aaa" ]
    (tokens cached "aaa");
  Array.iter
    (fun word ->
      equal
        ~msg:(Printf.sprintf "ignore_merges %S" word)
        (array int) (encode_ids uncached word) (encode_ids cached word))
    collide_words;
  equal ~msg:"and still stands after the collisions" (list string) [ "aaa" ]
    (tokens cached "aaa")

(* Pretokens above 15 bytes are keyed by their bytes in a table of their own,
   and pretokens of more than four tokens fit in no entry at all. *)
let test_cache_long_and_wide () =
  let uncached = collide 0 in
  let cached = collide 4096 in
  let words =
    [
      "abcdefgh";
      "abcdefghabcdefg";
      "abcdefghabcdefgh";
      "abcdefghabcdefghabcdefghabcdefgh";
      "aaaabbbbccccddddeeeeffffgggghhhh";
      String.make 200 'a';
      String.concat ""
        (List.init 60 (fun i ->
             String.make 1 (Char.chr (Char.code 'a' + (i mod 8)))));
    ]
  in
  List.iter
    (fun word ->
      let want = encode_ids uncached word in
      for _ = 1 to 3 do
        equal
          ~msg:(Printf.sprintf "%d bytes" (String.length word))
          (array int) want (encode_ids cached word)
      done)
    words;
  equal ~msg:"more than four tokens" bool true
    (Array.length (encode_ids cached "aaaabbbbccccddddeeeeffffgggghhhh") > 4)

(* Three pretokens that share the one set of the smallest cache: a miss stores
   into way 0 after shifting way 0 into way 1 and dropping what way 1 held, so
   cycling through three words evicts on every step, and a mixed order re-reads
   entries in either way and just past eviction. Every answer must come out as
   the merges say. *)
let test_set_eviction () =
  let uncached = collide 0 in
  let cached = collide 1 in
  let words = [| "abcd"; "bcda"; "cdab" |] in
  let expected = Array.map (fun w -> encode_ids uncached w) words in
  let check round i =
    equal
      ~msg:(Printf.sprintf "round %d, %S" round words.(i))
      (array int) expected.(i)
      (encode_ids cached words.(i))
  in
  for round = 1 to 4 do
    Array.iteri (fun i _ -> check round i) words
  done;
  List.iter (check 5) [ 0; 1; 0; 2; 1; 0; 2; 2; 0; 1 ]

(* The front table: a small direct-mapped table probed before the two-way one,
   filled by promoting whatever the back table answers. Every word of length up
   to four over eight letters is more distinct pretokens than the front has
   slots, so three passes in three orders exercise a promotion, a front hit, an
   eviction by a colliding promotion and a re-promotion — and every answer must
   still be the merges'. *)
let test_front_promotion_and_eviction () =
  let uncached = collide 0 in
  let cached = collide 262144 in
  let words =
    let acc = ref [] in
    let rec go prefix len =
      if len > 0 then
        for c = 0 to 7 do
          let w = prefix ^ String.make 1 (Char.chr (Char.code 'a' + c)) in
          acc := w :: !acc;
          go w (len - 1)
        done
    in
    go "" 4;
    Array.of_list (List.rev !acc)
  in
  let n = Array.length words in
  let expected = Array.map (fun w -> encode_ids uncached w) words in
  let check pass i =
    equal
      ~msg:(Printf.sprintf "pass %d, %S" pass words.(i))
      (array int) expected.(i)
      (encode_ids cached words.(i))
  in
  for i = 0 to n - 1 do
    check 1 i
  done;
  for i = n - 1 downto 0 do
    check 2 i
  done;
  (* A coprime stride visits every word once more, mixing hot and cold. *)
  let j = ref 0 in
  for _ = 0 to n - 1 do
    check 3 !j;
    j := (!j + 2311) mod n
  done

(* Byte fallback and the unknown token. Expectations probed with the
   [tokenizers] wheel: a fallback never lets a pending unknown token out, so the
   unknown token follows the bytes that came after it and comes out at the next
   vocabulary hit or at the end of the word. *)

let fallback_vocab =
  ("<unk>", 100)
  :: List.mapi
       (fun i b -> (Printf.sprintf "<0x%02X>" b, i))
       [ 0x61; 0x3C; 0x2F; 0x77; 0x3E; 0x62 ]

let test_unk_after_byte_fallback () =
  let plain =
    bpe ~vocab:fallback_vocab ~merges:[] ~unk_token:"<unk>" ~byte_fallback:true
      ~end_of_word_suffix:"</w>" ()
  in
  let fused =
    bpe ~vocab:fallback_vocab ~merges:[] ~unk_token:"<unk>" ~byte_fallback:true
      ~fuse_unk:true ~end_of_word_suffix:"</w>" ()
  in
  let suffixed = [ "<0x61>"; "<0x3C>"; "<0x2F>"; "<0x77>"; "<0x3E>" ] in
  equal ~msg:"a word that falls back whole" (list string) suffixed
    (tokens plain "a");
  equal ~msg:"the unknown token follows the fallback" (list string)
    (suffixed @ [ "<unk>" ]) (tokens plain "za");
  equal ~msg:"and precedes it when it comes last" (list string)
    [ "<0x61>"; "<unk>" ] (tokens plain "az");
  equal ~msg:"one unknown token per character" (list string)
    [ "<0x61>"; "<unk>"; "<unk>" ]
    (tokens plain "zaz");
  equal ~msg:"two of them in a row" (list string) [ "<unk>"; "<unk>" ]
    (tokens plain "zz");
  equal ~msg:"fused across the fallback" (list string) [ "<0x61>"; "<unk>" ]
    (tokens fused "zaz");
  equal ~msg:"fused in a row" (list string) [ "<unk>" ] (tokens fused "zz");
  equal ~msg:"fused before a fallback" (list string) (suffixed @ [ "<unk>" ])
    (tokens fused "zza")

let test_unk_flushed_by_vocab_hit () =
  let vocab = [ ("<unk>", 100); ("<0x61>", 0); ("b", 1) ] in
  let tokenizer =
    bpe ~vocab ~merges:[] ~unk_token:"<unk>" ~byte_fallback:true ()
  in
  equal ~msg:"a vocabulary hit lets the unknown token out" (list string)
    [ "<0x61>"; "<unk>"; "b" ] (tokens tokenizer "zab");
  equal ~msg:"nothing is held back before it" (list string)
    [ "<unk>"; "b"; "<0x61>" ] (tokens tokenizer "zba");
  equal ~msg:"several fallbacks first" (list string)
    [ "<0x61>"; "<0x61>"; "<unk>"; "b" ]
    (tokens tokenizer "zaab");
  (* Offsets run in emission order, as HuggingFace reports them. *)
  equal ~msg:"offsets follow the tokens"
    (list (pair int int))
    [ (0, 1); (1, 2); (2, 3) ]
    (encode tokenizer "zab" |> Encoding.offsets |> Array.to_list)

(* A byte sequence cut short at the end of the input is taken as the bytes that
   are left of it, never as the bytes its lead byte promises. *)
let test_truncated_utf8 () =
  let vocab = [ ("a", 0); ("é", 1); ("€", 2) ] in
  let tokenizer = bpe ~vocab ~merges:[] () in
  equal ~msg:"whole characters" (list string) [ "a"; "é"; "€" ]
    (tokens tokenizer "aé€");
  equal ~msg:"a two-byte character cut to one" (list string) [ "a" ]
    (tokens tokenizer "a\xC3");
  equal ~msg:"a three-byte character cut to two" (list string) [ "a" ]
    (tokens tokenizer "a\xE2\x82");
  equal ~msg:"a four-byte lead byte alone" (list string) [ "a" ]
    (tokens tokenizer "a\xF0");
  (* The affixed path spells the character out to fall back on it, which is
     where a length taken from the lead byte used to read past the end. *)
  let affixed =
    bpe
      ~vocab:
        (("a", 0)
        :: List.init 96 (fun i -> (Printf.sprintf "<0x%02X>" (0x20 + i), 1 + i))
        )
      ~merges:[] ~continuing_subword_prefix:"##" ~byte_fallback:true ()
  in
  equal ~msg:"a truncated character under a prefix" (list string) [ "a" ]
    (tokens affixed "a\xE2\x82");
  let unk =
    bpe
      ~vocab:[ ("<unk>", 0); ("a", 1) ]
      ~merges:[] ~unk_token:"<unk>" ~end_of_word_suffix:"</w>" ()
  in
  equal ~msg:"a truncated character under a suffix" (list string)
    [ "a"; "<unk>" ] (tokens unk "a\xE2\x82")

(* Words of more than [max_linear] symbols leave the rank scan for the heap;
   both must merge in the order HuggingFace merges in — by rank, then by
   position. *)
let test_long_word_merges () =
  let vocab =
    [ ("a", 0); ("b", 1); ("aa", 2); ("aaaa", 3); ("aaaaaaaa", 4); ("ab", 5) ]
  in
  let merges = [ ("a", "a"); ("aa", "aa"); ("aaaa", "aaaa"); ("a", "b") ] in
  let tokenizer = bpe ~vocab ~merges ~cache_capacity:0 () in
  equal ~msg:"eight" (list string) [ "aaaaaaaa" ]
    (tokens tokenizer (String.make 8 'a'));
  equal ~msg:"thirty-two" (list string)
    [ "aaaaaaaa"; "aaaaaaaa"; "aaaaaaaa"; "aaaaaaaa" ]
    (tokens tokenizer (String.make 32 'a'));
  (* Sixty-four characters is past the rank scan's ceiling. *)
  equal ~msg:"sixty-four" (list string)
    (List.init 8 (fun _ -> "aaaaaaaa"))
    (tokens tokenizer (String.make 64 'a'));
  equal ~msg:"a tail that cannot merge" (list string)
    [ "aaaaaaaa"; "aaaaaaaa"; "aaaa"; "aa"; "b" ]
    (tokens tokenizer (String.make 22 'a' ^ "b"));
  let cached = bpe ~vocab ~merges () in
  List.iter
    (fun n ->
      let word = String.make n 'a' in
      equal
        ~msg:(Printf.sprintf "cached and uncached agree at %d" n)
        (array int)
        (encode_ids tokenizer word)
        (encode_ids cached word))
    [ 1; 2; 3; 7; 15; 16; 31; 32; 33; 64; 65; 200 ]

(* Training. Every expectation below was probed with the [tokenizers] wheel,
   running its [BpeTrainer] over the same corpus behind the same pre-tokenizer,
   which is what cuts the words the trainer counts. *)

let split = Pre_tokenizer.whitespace_split ()

let train_corpus =
  [
    "low low low low low lower lower";
    "newest newest newest newest newest newest";
    "widest widest widest lowest";
  ]

let trained_vocab tokenizer =
  let entries = vocab tokenizer in
  let ordered = Array.make (List.length entries) "" in
  List.iter (fun (token, id) -> ordered.(id) <- token) entries;
  Array.to_list ordered

let trained_merges tokenizer =
  let folder = Filename.temp_dir "brot_merges" "" in
  let files = save_model_files tokenizer ~folder () in
  let path = List.find (fun f -> Filename.check_suffix f ".txt") files in
  let ic = open_in path in
  let lines = ref [] in
  (try
     while true do
       lines := input_line ic :: !lines
     done
   with End_of_file -> ());
  close_in ic;
  List.iter Sys.remove files;
  Unix.rmdir folder;
  List.filter
    (fun line -> not (String.starts_with ~prefix:"#version" line))
    (List.rev !lines)

(* Real text against tiny tables: a model trained on one parity corpus encodes
   both corpora to the same ids whatever the cache holds. [0] disables caching
   and is the reference. *)
let test_cache_capacities_on_parity_corpus () =
  let sample = Fixture.read "fixtures/parity/sample.txt" in
  let edge_cases = Fixture.read "fixtures/parity/edge_cases.txt" in
  let trained =
    train_bpe ~pre:split ~vocab_size:300 ~show_progress:false
      (`Seq (List.to_seq [ sample ]))
  in
  let vocab = vocab trained in
  let merges =
    List.map
      (fun line ->
        match String.index_opt line ' ' with
        | Some i ->
            ( String.sub line 0 i,
              String.sub line (i + 1) (String.length line - i - 1) )
        | None -> failf "malformed merge %S" line)
      (trained_merges trained)
  in
  let model capacity =
    bpe ~pre:split ~vocab ~merges ~cache_capacity:capacity ()
  in
  let reference = model 0 in
  let corpora =
    [
      ("sample", sample, encode_ids reference sample);
      ("edge_cases", edge_cases, encode_ids reference edge_cases);
    ]
  in
  List.iter
    (fun capacity ->
      let cached = model capacity in
      for pass = 1 to 2 do
        List.iter
          (fun (name, text, expected) ->
            equal
              ~msg:(Printf.sprintf "%d entries, pass %d, %s" capacity pass name)
              (array int) expected (encode_ids cached text))
          corpora
      done)
    [ 1; 2; 4; 64; 262144 ]

let test_train () =
  let tokenizer =
    train_bpe ~pre:split ~vocab_size:30 ~show_progress:false
      (`Seq (List.to_seq train_corpus))
  in
  (* The corpus runs out of pairs before the target size is reached. *)
  equal ~msg:"vocabulary" (list string)
    [
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
      "low";
      "ew";
      "new";
      "newest";
      "dest";
      "idest";
      "widest";
      "er";
      "lower";
      "lowest";
    ]
    (trained_vocab tokenizer);
  equal ~msg:"merges" (list string)
    [
      "e s";
      "es t";
      "l o";
      "lo w";
      "e w";
      "n ew";
      "new est";
      "d est";
      "i dest";
      "w idest";
      "e r";
      "low er";
      "low est";
    ]
    (trained_merges tokenizer)

let test_train_suffix () =
  let tokenizer =
    train_bpe ~pre:split ~vocab_size:40 ~end_of_word_suffix:"</w>"
      ~show_progress:false
      (`Seq (List.to_seq train_corpus))
  in
  (* The suffix joins the last character before any pair is counted, so the
     vocabulary holds suffixed characters and the merges name them. The three
     suffixed characters come out in the order their words do, which the wheel
     draws at random; the merges it learns are the ones below. *)
  equal ~msg:"vocabulary" (list string)
    [
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
      "w</w>";
      "r</w>";
      "t</w>";
      "es";
      "est</w>";
      "lo";
      "west</w>";
      "ewest</w>";
      "newest</w>";
      "low</w>";
      "dest</w>";
      "idest</w>";
      "widest</w>";
      "er</w>";
      "wer</w>";
      "lower</w>";
      "lowest</w>";
    ]
    (trained_vocab tokenizer);
  equal ~msg:"merges" (list string)
    [
      "e s";
      "es t</w>";
      "l o";
      "w est</w>";
      "e west</w>";
      "n ewest</w>";
      "lo w</w>";
      "d est</w>";
      "i dest</w>";
      "w idest</w>";
      "e r</w>";
      "w er</w>";
      "lo wer</w>";
      "lo west</w>";
    ]
    (trained_merges tokenizer);
  equal ~msg:"a trained word is one suffixed token" (list string) [ "low</w>" ]
    (tokens tokenizer "low");
  equal ~msg:"and so is a longer one" (list string) [ "lowest</w>" ]
    (tokens tokenizer "lowest");
  equal ~msg:"an unseen word still ends in the suffix" (list string)
    [ "s"; "low</w>" ] (tokens tokenizer "slow")

let test_train_prefix () =
  let tokenizer =
    train_bpe ~pre:split ~vocab_size:30 ~continuing_subword_prefix:"##"
      ~show_progress:false
      (`Seq (List.to_seq train_corpus))
  in
  (* Only the characters that turn up after the first one of a word are learned
     in prefixed form: ["##l"] is absent because no word here has an [l] later
     on, which is why ["slow"] loses its [l]. *)
  equal ~msg:"vocabulary" (list string)
    [
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
      "##o";
      "##w";
      "##e";
      "##r";
      "##s";
      "##t";
      "##i";
      "##d";
      "##es";
      "##est";
      "lo";
      "low";
      "ne";
      "##west";
      "newest";
      "wi";
      "##dest";
      "widest";
      "##er";
      "lower";
    ]
    (trained_vocab tokenizer);
  equal ~msg:"merges" (list string)
    [
      "##e ##s";
      "##es ##t";
      "l ##o";
      "lo ##w";
      "n ##e";
      "##w ##est";
      "ne ##west";
      "w ##i";
      "##d ##est";
      "wi ##dest";
      "##e ##r";
      "low ##er";
    ]
    (trained_merges tokenizer);
  equal ~msg:"a trained word is one token" (list string) [ "newest" ]
    (tokens tokenizer "newest");
  equal ~msg:"an unseen word breaks into prefixed pieces" (list string)
    [ "low"; "##est" ]
    (tokens tokenizer "lowest");
  equal ~msg:"a character with no prefixed form drops out" (list string)
    [ "s"; "##o"; "##w" ] (tokens tokenizer "slow")

let test_train_limit_alphabet () =
  (* [z] outranks [b], so a limit of two drops [b], and the word holding it
     loses the character instead of merging it. *)
  let corpus = `Seq (List.to_seq [ "aza aza aza ab ab" ]) in
  let whole = train_bpe ~pre:split ~vocab_size:20 ~show_progress:false corpus in
  let limited =
    train_bpe ~pre:split ~vocab_size:20 ~limit_alphabet:2 ~show_progress:false
      corpus
  in
  equal ~msg:"whole alphabet" (list string)
    [ "a"; "b"; "z"; "az"; "aza"; "ab" ]
    (trained_vocab whole);
  equal ~msg:"two characters" (list string) [ "a"; "z"; "az"; "aza" ]
    (trained_vocab limited);
  equal ~msg:"a dropped character leaves the word" (list string) [ "a" ]
    (tokens limited "ab")

let test_train_max_token_length () =
  (* The limit counts characters, not bytes: these take two bytes each, and a
     four-character token is still learned under a limit of five. It is
     exclusive — under a limit of four a merge stops at three characters — and
     the single-character merges the training opens with are exempt from it. *)
  let trained ?max_token_length corpus =
    train_bpe ~pre:split ~vocab_size:40 ?max_token_length ~show_progress:false
      (`Seq (List.to_seq [ corpus ]))
  in
  let eight = "αβγδεζηθ αβγδεζηθ αβγδεζηθ" in
  equal ~msg:"no limit" (list string) [ "αβγδεζηθ" ]
    (tokens (trained eight) "αβγδεζηθ");
  equal ~msg:"five characters" (list string) [ "αβγδ"; "εζηθ" ]
    (tokens (trained ~max_token_length:5 eight) "αβγδεζηθ");
  equal ~msg:"two characters still reached" (list string)
    [ "αβ"; "γδ"; "εζ"; "ηθ" ]
    (tokens (trained ~max_token_length:2 eight) "αβγδεζηθ");
  equal ~msg:"four characters stops at three" (list string) [ "αβ"; "γδε" ]
    (tokens (trained ~max_token_length:4 "αβγδε αβγδε αβγδε") "αβγδε")

let test_train_repeated_merge () =
  (* A queue entry carries the words its pair was found in when it was queued,
     so a pair found again afterwards is merged a second time and recorded a
     second time. The model keeps the rank it was given last, which is why [###
     ##a] is written once here. *)
  let tokenizer =
    train_bpe ~pre:split ~vocab_size:24 ~continuing_subword_prefix:"##"
      ~show_progress:false
      (`Seq (List.to_seq [ "###a b### a" ]))
  in
  equal ~msg:"vocabulary" (list string)
    [ "#"; "a"; "b"; "###"; "##a"; "####"; "b##"; "###a"; "b###" ]
    (trained_vocab tokenizer);
  equal ~msg:"merges" (list string)
    [ "### ###"; "# ####"; "b ####"; "### ##a"; "b## ###" ]
    (trained_merges tokenizer)

let test_train_initial_alphabet () =
  (* An entry of the initial alphabet stands for the code point it starts with,
     so ["ét"] adds [é] and not the byte ["\xc3"], and an empty entry adds
     nothing. The wheel keeps the first character of each string the same
     way. *)
  let tokenizer =
    train_bpe ~pre:split ~vocab_size:30 ~initial_alphabet:[ "é"; "ét"; "" ]
      ~show_progress:false
      (`Seq (List.to_seq train_corpus))
  in
  equal ~msg:"vocabulary" (list string)
    [
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
      "é";
      "es";
      "est";
      "lo";
      "low";
      "ew";
      "new";
      "newest";
      "dest";
      "idest";
      "widest";
      "er";
      "lower";
      "lowest";
    ]
    (trained_vocab tokenizer)

let test_train_byte_level () =
  (* A byte-level pre-tokenizer hands the trainer encoded pieces, so the merges
     are learned over the encoded form and the vocabulary holds it — which is
     what the trained tokenizer meets again when it encodes. *)
  let tokenizer =
    train_bpe
      ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
      ~decoder:(Decoder.byte_level ()) ~vocab_size:40 ~show_progress:false
      (`Seq (List.to_seq train_corpus))
  in
  equal ~msg:"vocabulary" (list string)
    [
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
      "Ġ";
      "es";
      "est";
      "lo";
      "low";
      "Ġlow";
      "ew";
      "new";
      "newest";
      "Ġnewest";
      "dest";
      "idest";
      "widest";
      "er";
      "Ġwidest";
      "Ġlower";
      "Ġlowest";
    ]
    (trained_vocab tokenizer);
  equal ~msg:"merges" (list string)
    [
      "e s";
      "es t";
      "l o";
      "lo w";
      "Ġ low";
      "e w";
      "n ew";
      "new est";
      "Ġ newest";
      "d est";
      "i dest";
      "w idest";
      "e r";
      "Ġ widest";
      "Ġlow er";
      "Ġlow est";
    ]
    (trained_merges tokenizer);
  equal ~msg:"the space of the second word is part of its token" (list string)
    [ "low"; "Ġlower" ]
    (tokens tokenizer "low lower");
  equal ~msg:"round trip" string "low lower"
    (decode tokenizer (encode tokenizer "low lower" |> Encoding.ids))

let test_train_whole_text () =
  (* With no pre-tokenizer a text is one word, spaces and all, so the merges
     reach across the words. *)
  let tokenizer =
    train_bpe ~vocab_size:30 ~show_progress:false
      (`Seq (List.to_seq train_corpus))
  in
  equal ~msg:"vocabulary" (list string)
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
    (trained_vocab tokenizer)

let test_train_normalizer () =
  (* The normalizer runs before the pre-tokenizer, so the words counted — and
     the vocabulary learned — are the normalized ones. *)
  let tokenizer =
    train_bpe ~normalizer:Normalizer.lowercase ~pre:split ~vocab_size:20
      ~show_progress:false
      (`Seq (List.to_seq [ "LOW Low low LOWER lower" ]))
  in
  equal ~msg:"vocabulary" (list string)
    [ "e"; "l"; "o"; "r"; "w"; "lo"; "low"; "er"; "lower" ]
    (trained_vocab tokenizer)

let with_corpus_file contents f =
  let path = Filename.temp_file "brot_corpus" ".txt" in
  let oc = open_out_bin path in
  output_string oc contents;
  close_out oc;
  Fun.protect ~finally:(fun () -> Sys.remove path) (fun () -> f path)

let test_train_files () =
  (* A line of a corpus file keeps the newline that ends it, so a byte-level
     pipeline meets it and the vocabulary holds ["Ċ"] — the same vocabulary the
     wheel learns from the same file. *)
  let tokenizer =
    with_corpus_file
      (String.concat "\n" train_corpus ^ "\n")
      (fun path ->
        train_bpe
          ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
          ~vocab_size:40 ~show_progress:false (`Files [ path ]))
  in
  equal ~msg:"vocabulary" (list string)
    [
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
      "Ċ";
      "Ġ";
      "es";
      "est";
      "lo";
      "low";
      "Ġlow";
      "ew";
      "new";
      "newest";
      "Ġnewest";
      "dest";
      "idest";
      "widest";
      "er";
      "Ġwidest";
      "Ġlower";
      "Ġlowest";
    ]
    (trained_vocab tokenizer)

let test_train_min_frequency () =
  (* Of the three pairs only [a a], seen three times, reaches the floor. *)
  let tokenizer =
    train_bpe ~pre:split ~vocab_size:20 ~min_frequency:3 ~show_progress:false
      (`Seq (List.to_seq [ "aa aa aa bb cc cc" ]))
  in
  equal ~msg:"vocabulary" (list string) [ "a"; "b"; "c"; "aa" ]
    (trained_vocab tokenizer);
  equal ~msg:"merges" (list string) [ "a a" ] (trained_merges tokenizer)

let () =
  run "BPE tests"
    [
      group "basic"
        [
          test "basic tokenization" test_bpe_basic;
          test "builder pattern" test_bpe_builder;
          test "ignore_merges" test_ignore_merges;
          test "dropout overrides ignore_merges"
            test_dropout_overrides_ignore_merges;
          test "empty prefix and suffix" test_empty_affixes;
          test "save and load" test_bpe_save_load;
          test "end-of-word suffix" test_suffix_only;
          test "end-of-word suffix and merges" test_suffix_and_merges;
          test "end-of-word suffix on a multi-byte character"
            test_suffix_multibyte;
          test "missing suffixed character" test_suffix_unknown;
          test "byte fallback with a suffix" test_suffix_byte_fallback;
          test "byte fallback with a prefix" test_prefix_byte_fallback;
          test "byte fallback with a prefix over a long word"
            test_prefix_byte_fallback_long;
          test "continuing prefix with and without a suffix"
            test_prefix_and_suffix;
          test "tokenizer integration" test_tokenizer_integration;
          test "unknown character" test_unknown_character;
          test "byte fallback before the unknown token"
            test_unk_after_byte_fallback;
          test "a vocabulary hit flushes the unknown token"
            test_unk_flushed_by_vocab_hit;
          test "truncated UTF-8" test_truncated_utf8;
          test "words past the rank scan" test_long_word_merges;
        ];
      group "cache"
        [
          test "collisions agree with the merges" test_cache_agrees_with_merges;
          test "the seed agrees with the merges" test_seed_agrees_with_merges;
          test "the seed under ignore_merges" test_seed_ignore_merges;
          test "long and many-token pretokens" test_cache_long_and_wide;
          test "a set filled beyond two ways" test_set_eviction;
          test "front promotion and eviction" test_front_promotion_and_eviction;
          test "small capacities on the parity corpora"
            test_cache_capacities_on_parity_corpus;
        ];
      group "training"
        [
          test "vocabulary and merges" test_train;
          test "end-of-word suffix" test_train_suffix;
          test "continuing subword prefix" test_train_prefix;
          test "limit_alphabet" test_train_limit_alphabet;
          test "initial_alphabet" test_train_initial_alphabet;
          test "max_token_length" test_train_max_token_length;
          test "min_frequency" test_train_min_frequency;
          test "a byte-level pre-tokenizer" test_train_byte_level;
          test "a normalizer" test_train_normalizer;
          test "no pre-tokenizer" test_train_whole_text;
          test "training from files" test_train_files;
          test "a pair merged twice" test_train_repeated_merge;
        ];
      group "parallel"
        [ test "shared cache across domains" test_parallel_cache ];
    ]
