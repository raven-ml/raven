(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every expectation here was read off HuggingFace [tokenizers] 0.23.1 with the
   same vocabulary, a Unigram model on its own with no pre-tokenizer, so the
   model sees each text whole. HuggingFace reports offsets in characters and
   brot in bytes; the spans below are the byte ones. *)

open Windtrap
open Brot

(* Pieces that overlap, so the best path is not the longest match at each
   position: ["abc"] is [a] + [bc] at [-2.5], not [abc] at [-5.0] nor [ab] + [c]
   at [-3.0]. *)
let overlapping =
  [
    ("<unk>", 0.0);
    ("a", -1.0);
    ("b", -1.0);
    ("c", -1.0);
    ("ab", -2.0);
    ("abc", -5.0);
    ("bc", -1.5);
  ]

(* Two pieces and nothing for anything else. *)
let letters = [ ("<unk>", 0.0); ("a", -1.0); ("b", -1.0) ]

let with_bytes ~upto vocab =
  vocab @ List.init upto (fun b -> (Printf.sprintf "<0x%02X>" b, -20.0))

let case t text ids =
  let encoding = encode t ~add_special_tokens:false text in
  equal
    ~msg:(Printf.sprintf "ids %S" text)
    (list int) ids
    (Array.to_list (Encoding.ids encoding))

let spans t text offsets =
  let encoding = encode t ~add_special_tokens:false text in
  equal
    ~msg:(Printf.sprintf "offsets %S" text)
    (list (pair int int))
    offsets
    (Array.to_list (Encoding.offsets encoding))

(* The pieces whose scores add up to the most, not the longest match at each
   position. *)
let test_best_path () =
  let t = unigram ~vocab:overlapping ~unk_id:0 () in
  case t "abc" [ 1; 6 ];
  case t "abcabc" [ 1; 6; 1; 6 ];
  case t "a" [ 1 ];
  case t "" [];
  case t "cba" [ 3; 2; 1 ];
  spans t "abc" [ (0, 1); (1, 3) ]

(* [ab] scores exactly what [a] and [b] do together, and the piece covering the
   whole span wins. *)
let test_equal_scores_keep_the_longer_piece () =
  let t = unigram ~vocab:overlapping ~unk_id:0 () in
  case t "ab" [ 4 ];
  let shade delta =
    unigram
      ~vocab:
        (List.map
           (fun (token, score) ->
             if token = "ab" then (token, score +. delta) else (token, score))
           overlapping)
      ~unk_id:0 ()
  in
  case (shade 0.1) "ab" [ 4 ];
  case (shade (-0.1)) "ab" [ 1; 2 ]

(* White space reaching the model is a character like any other: a piece of the
   vocabulary when it holds one, and unknown when it does not. *)
let test_white_space_is_a_character () =
  let t = unigram ~vocab:letters ~unk_id:0 () in
  case t "a b" [ 1; 0; 2 ];
  case t "a\tb" [ 1; 0; 2 ];
  case t "a\nb" [ 1; 0; 2 ];
  spans t "a b" [ (0, 1); (1, 2); (2, 3) ];
  let spaced = unigram ~vocab:(letters @ [ (" ", -5.0) ]) ~unk_id:0 () in
  case spaced "a b" [ 1; 3; 2 ]

(* One unknown token per character, and a run of them fused into one. *)
let test_unknown_characters () =
  let t = unigram ~vocab:letters ~unk_id:0 () in
  case t "a\xe2\x80\x8bb" [ 1; 0; 2 ];
  case t "a\xe2\x80\x8b\xe2\x80\x8cb" [ 1; 0; 2 ];
  case t "\xe4\xb8\xad\xe6\x96\x87" [ 0 ];
  case t "\xe2\x80\x8b" [ 0 ];
  case t "a\xe2\x80\x8bba\xe2\x80\x8b" [ 1; 0; 2; 1; 0 ];
  (* The fused run stands for every byte of the characters it covers. *)
  spans t "a\xe2\x80\x8b\xe2\x80\x8cb" [ (0, 1); (1, 7); (7, 8) ];
  spans t "ab\xe2\x80\x8b\xe2\x80\x8cab"
    [ (0, 1); (1, 2); (2, 8); (8, 9); (9, 10) ]

(* Byte fallback spells the whole fused run out, one token per byte, and only
   when the vocabulary holds every one of them. *)
let test_byte_fallback () =
  let t =
    unigram
      ~vocab:(with_bytes ~upto:256 letters)
      ~unk_id:0 ~byte_fallback:true ()
  in
  case t "a\xe2\x80\x8bb" [ 1; 229; 131; 142; 2 ];
  case t "a\xe2\x80\x8b\xe2\x80\x8cb" [ 1; 229; 131; 142; 229; 131; 143; 2 ];
  case t "\xe2\x80\x8b" [ 229; 131; 142 ];
  (* Every byte token of a run stands for the whole run, as it does in
     HuggingFace: (1, 7) throughout on the run of two characters. *)
  spans t "a\xe2\x80\x8bb" [ (0, 1); (1, 4); (1, 4); (1, 4); (4, 5) ];
  spans t "a\xe2\x80\x8b\xe2\x80\x8cb"
    [ (0, 1); (1, 7); (1, 7); (1, 7); (1, 7); (1, 7); (1, 7); (7, 8) ];
  (* [0xE2] has no entry, so the run falls back to the unknown token. *)
  let partial =
    unigram
      ~vocab:(with_bytes ~upto:0xE0 letters)
      ~unk_id:0 ~byte_fallback:true ()
  in
  case partial "a\xe2\x80\x8bb" [ 1; 0; 2 ];
  (* The byte tokens are in the vocabulary but byte fallback is off. *)
  let off = unigram ~vocab:(with_bytes ~upto:256 letters) ~unk_id:0 () in
  case off "a\xe2\x80\x8bb" [ 1; 0; 2 ]

(* A pretoken over any piece of the vocabulary, and one far over any scratch a
   state starts with. *)
let test_long_pretoken () =
  let t =
    unigram ~vocab:[ ("<unk>", 0.0); ("a", -1.0); ("aa", -1.5) ] ~unk_id:0 ()
  in
  case t "aaaaaaa" [ 1; 2; 2; 2 ];
  let encoding = encode t ~add_special_tokens:false (String.make 100_000 'a') in
  equal ~msg:"100 KB of one letter" int 50_000 (Encoding.length encoding);
  equal ~msg:"every one of them the pair" bool true
    (Array.for_all (fun id -> id = 2) (Encoding.ids encoding))

(* Without an unknown token, the error comes as soon as the best path into some
   character would spend one: [a b] on its space, [ab] over pairs on its [b].
   Where a longer piece is the best way into every character, nothing is spent
   and nothing raised. *)
let test_without_an_unknown_token () =
  let t = unigram ~vocab:letters () in
  case t "ab" [ 1; 2 ];
  case (unigram ~vocab:[ ("ab", -1.0); ("a", -2.0) ] ()) "ab" [ 0 ];
  case
    (unigram ~vocab:[ ("ab", -1.0); ("a", -2.0); ("abc", -1.0) ] ())
    "abc" [ 2 ];
  raises
    (Failure
       "Unigram.encode_into: the text holds a character the vocabulary does \
        not have, and the model has no unk_id to stand in for it") (fun () ->
      ignore (encode t "a b"));
  (* [ab] is only ever reached through a piece longer than one character. *)
  let pairs = unigram ~vocab:[ ("ab", -1.0) ] () in
  raises
    (Failure
       "Unigram.encode_into: the text holds a character the vocabulary does \
        not have, and the model has no unk_id to stand in for it") (fun () ->
      ignore (encode pairs "ab"));
  let named = unigram ~vocab:[ ("<unk>", 0.0); ("ab", -1.0) ] ~unk_id:0 () in
  case named "ab" [ 1 ];
  (* [unk_token] names the entry when it is in the vocabulary. *)
  let by_token =
    unigram ~vocab:[ ("<unk>", 0.0); ("ab", -1.0) ] ~unk_token:"<unk>" ()
  in
  case by_token "abab" [ 1; 1 ];
  case by_token "abc" [ 1; 0 ]

(* A fused run whose text is itself an entry is that entry, before byte fallback
   or the unknown token: HuggingFace looks the run up as a string. *)
let test_fused_run_that_is_an_entry () =
  let vocab =
    [ ("<unk>", 0.0); ("a", -1.0); ("b", -1.0); ("<unk><unk>", -1.0) ]
  in
  let t = unigram ~vocab ~unk_id:0 () in
  case t "<unk><unk>" [ 3 ];
  case t "a<unk><unk>b" [ 1; 3; 2 ];
  spans t "a<unk><unk>b" [ (0, 1); (1, 11); (11, 12) ];
  let bytes =
    unigram ~vocab:(with_bytes ~upto:256 vocab) ~unk_id:0 ~byte_fallback:true ()
  in
  case bytes "<unk><unk>" [ 3 ]

(* A character no piece covers on its own is reached by the unknown token even
   where a piece starts there: [ab] does not cover [a], so [a] is unknown, and
   the tie between [ab] + unknown [c] and unknown [a] + [bc] goes to the path
   written first. *)
let test_piece_starting_at_an_uncovered_character () =
  let t =
    unigram ~vocab:[ ("<unk>", 0.0); ("ab", -1.0); ("bc", -1.0) ] ~unk_id:0 ()
  in
  case t "abc" [ 0; 2 ];
  spans t "abc" [ (0, 1); (1, 3) ]

let test_unk_id_outside_the_vocabulary () =
  raises
    (Invalid_argument
       "Unigram.create: unk_id 3 is outside a vocabulary of 3 entries")
    (fun () -> ignore (unigram ~vocab:letters ~unk_id:3 ()));
  raises
    (Invalid_argument
       "Unigram.create: unk_id 0 is outside a vocabulary of 0 entries")
    (fun () -> ignore (unigram ~unk_id:0 ()))

(* The unknown token and byte fallback survive a round trip through
   [tokenizer.json], in the members HuggingFace writes and reads back. *)
let test_json_round_trip () =
  let t =
    unigram
      ~vocab:(with_bytes ~upto:256 letters)
      ~unk_id:0 ~byte_fallback:true ~unk_token:"<unk>" ()
  in
  let json = to_json t in
  (match from_json json with
  | Ok reloaded -> case reloaded "a\xe2\x80\x8bb" [ 1; 229; 131; 142; 2 ]
  | Error msg -> failf "cannot reload: %s" msg);
  let model_text =
    match json with
    | Jsont.Object (members, meta) -> (
        match Jsont.Json.find_mem "model" members with
        | Some (_, Jsont.Object (model, _)) ->
            let written =
              List.filter (fun ((name, _), _) -> name <> "vocab") model
            in
            Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json
              (Jsont.Object (written, meta))
        | _ -> Error "no model object")
    | _ -> Error "not an object"
  in
  equal ~msg:"model json" (result string string)
    (Ok {|{"type":"Unigram","unk_id":0,"byte_fallback":true}|}) model_text

let () =
  run "Unigram tests"
    [
      group "tokenization"
        [
          test "best path over overlapping pieces" test_best_path;
          test "equal scores keep the longer piece"
            test_equal_scores_keep_the_longer_piece;
          test "white space is a character" test_white_space_is_a_character;
          test "unknown characters" test_unknown_characters;
          test "byte fallback" test_byte_fallback;
          test "long pretoken" test_long_pretoken;
          test "without an unknown token" test_without_an_unknown_token;
          test "fused run that is an entry" test_fused_run_that_is_an_entry;
          test "piece starting at an uncovered character"
            test_piece_starting_at_an_uncovered_character;
        ];
      group "model"
        [
          test "unk_id outside the vocabulary"
            test_unk_id_outside_the_vocabulary;
          test "JSON round trip" test_json_round_trip;
        ];
    ]
