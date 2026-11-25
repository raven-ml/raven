(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What a built tokenizer reports: its stages, its role markers and its
   vocabulary. *)

open Windtrap
open Brot

(* The value given, compared by identity. *)
let same pp = option (Testable.make ~pp ~equal:( == ))

(* Each constructor, given or not given each stage. *)
let constructors =
  [
    ( "bpe",
      fun ?normalizer ?pre ?post ?decoder () ->
        bpe ?normalizer ?pre ?post ?decoder () );
    ( "wordpiece",
      fun ?normalizer ?pre ?post ?decoder () ->
        wordpiece ?normalizer ?pre ?post ?decoder () );
    ( "word_level",
      fun ?normalizer ?pre ?post ?decoder () ->
        word_level ?normalizer ?pre ?post ?decoder () );
    ( "unigram",
      fun ?normalizer ?pre ?post ?decoder () ->
        unigram ?normalizer ?pre ?post ?decoder () );
    ( "chars",
      fun ?normalizer ?pre ?post ?decoder () ->
        chars ?normalizer ?pre ?post ?decoder () );
  ]

(* normalizer *)

let given_normalizer =
  cases "is the normalizer given, or None" ~id:fst constructors
    (fun (_, make) ->
      let n = Normalizer.lowercase in
      equal ~msg:"given" (same Normalizer.pp) (Some n)
        (normalizer (make ~normalizer:n ()));
      equal ~msg:"none" (same Normalizer.pp) None (normalizer (make ())))

(* pre_tokenizer *)

let given_pre =
  cases "is the pre-tokenizer given, or None" ~id:fst
    (List.filter (fun (name, _) -> name <> "word_level") constructors)
    (fun (_, make) ->
      let p = Pre_tokenizer.bert in
      equal ~msg:"given" (same Pre_tokenizer.pp) (Some p)
        (pre_tokenizer (make ~pre:p ()));
      equal ~msg:"none" (same Pre_tokenizer.pp) None (pre_tokenizer (make ())))

let test_word_level_pre () =
  equal ~msg:"given" (same Pre_tokenizer.pp) (Some Pre_tokenizer.bert)
    (pre_tokenizer (word_level ~pre:Pre_tokenizer.bert ()));
  equal ~msg:"none" (same Pre_tokenizer.pp) (Some Pre_tokenizer.whitespace)
    (pre_tokenizer (word_level ()))

(* post_processor *)

let given_post =
  cases "is the post-processor given, or None" ~id:fst constructors
    (fun (_, make) ->
      let p = Post_processor.bert ~sep:("[SEP]", 1) ~cls:("[CLS]", 0) in
      equal ~msg:"given" (same Post_processor.pp) (Some p)
        (post_processor (make ~post:p ()));
      equal ~msg:"none" (same Post_processor.pp) None (post_processor (make ())))

(* decoder *)

let given_decoder =
  cases "is the decoder given, or None" ~id:fst constructors (fun (_, make) ->
      let d = Decoder.fuse in
      equal ~msg:"given" (same Decoder.pp) (Some d)
        (decoder (make ~decoder:d ()));
      equal ~msg:"none" (same Decoder.pp) None (decoder (make ())))

(* added_tokens *)

let pp_added_token ppf (a : added_token) =
  Format.fprintf ppf
    "{content %S; special %b; single_word %b; lstrip %b; rstrip %b; normalized \
     %b}"
    a.content a.special a.single_word a.lstrip a.rstrip a.normalized

let test_added_tokens () =
  let x = added_token ~lstrip:true "<x>"
  and y = added_token ~special:false "y" in
  let t =
    bpe ~added_tokens:[ x; y ] ~bos_token:"<s>" ~eos_token:"<x>"
      ~pad_token:"<pad>" ()
  in
  equal
    (list (Testable.make ~pp:pp_added_token ~equal:( = )))
    [ x; y; added_token "<s>"; added_token "<pad>" ]
    (added_tokens t)

(* bos_token, eos_token, pad_token, unk_token *)

let role accessor make =
  equal ~msg:"given" (option string) (Some "<r>") (accessor (make (Some "<r>")));
  equal ~msg:"none" (option string) None (accessor (make None))

(* vocab *)

let pairs = list (pair string int)

let empty_vocab =
  cases "of a tokenizer without vocabulary is empty" ~id:fst
    [
      ("bpe", fun () -> bpe ());
      ("wordpiece", fun () -> wordpiece ());
      ("word_level", fun () -> word_level ());
      ("unigram", fun () -> unigram ());
    ]
    (fun (_, make) -> equal pairs [] (vocab (make ())))

(* vocab_size, token_to_id, id_to_token *)

(* A model drawn with distinct tokens over a few characters, ids a permutation
   of their positions, and added tokens, some of which it holds. A Unigram
   entry's id is its position. *)
let built =
  let open Gen in
  let token =
    map (String.concat "")
      (list ~size:(int_range 1 3) (of_list [ "a"; "b"; "<"; ">"; "\u{e9}" ]))
  in
  let distinct tokens =
    List.fold_left
      (fun acc t -> if List.mem t acc then acc else acc @ [ t ])
      [] tokens
  in
  let* model = of_list [ "bpe"; "wordpiece"; "word_level"; "unigram" ] in
  let* tokens =
    map (fun ts -> distinct ("[UNK]" :: ts)) (list ~size:(int_range 0 12) token)
  in
  let* ids = permutation (List.init (List.length tokens) Fun.id) in
  let+ added = list ~size:(int_range 0 4) (pair token bool) in
  (model, List.combine tokens ids, added)

let make (model, vocab, added) =
  let added_tokens =
    List.map (fun (c, special) -> added_token ~special c) added
  in
  match model with
  | "bpe" -> bpe ~vocab ~merges:[] ~added_tokens ()
  | "wordpiece" -> wordpiece ~vocab ~added_tokens ()
  | "word_level" -> word_level ~vocab ~added_tokens ()
  | _ ->
      let by_id = List.sort (fun (_, a) (_, b) -> compare a b) vocab in
      unigram ~vocab:(List.map (fun (t, _) -> (t, -1.)) by_id) ~added_tokens ()

let pp_built ppf (model, vocab, added) =
  let pair (t, id) = Printf.sprintf "(%S, %d)" t id in
  let token (c, special) = Printf.sprintf "(%S, special %b)" c special in
  Format.fprintf ppf "%s vocab [%s]@ added [%s]" model
    (String.concat "; " (List.map pair vocab))
    (String.concat "; " (List.map token added))

let built = Gen.with_pp pp_built built

let cover_built (_, vocab, added) =
  cover "an added token the model holds"
    (List.exists (fun (c, _) -> List.mem_assoc c vocab) added);
  cover "an added token the model lacks"
    (List.exists (fun (c, _) -> not (List.mem_assoc c vocab)) added)

(* The added contents the drawn model lacks, once each. Their ids are the
   tokenizer's to choose. *)
let lacked (_, vocab, added) =
  List.fold_left
    (fun acc (c, _) ->
      if List.mem_assoc c vocab || List.mem c acc then acc else acc @ [ c ])
    [] added

let vocab_law =
  prop ~count:60 "lists the model's pairs, then the added tokens it lacks" built
    (fun ((_, drawn, _) as case) ->
      cover_built case;
      let listed = vocab (make case) in
      let n = List.length drawn in
      equal ~msg:"the model's" pairs (List.sort compare drawn)
        (List.sort compare (List.filteri (fun i _ -> i < n) listed));
      equal ~msg:"then the added tokens" (list string)
        (List.sort compare (lacked case))
        (List.sort compare
           (List.filteri (fun i _ -> i >= n) listed |> List.map fst)))

let size_law =
  prop ~count:60 "counts the model's tokens and the added ones it lacks" built
    (fun ((_, drawn, _) as case) ->
      cover_built case;
      equal int
        (List.length drawn + List.length (lacked case))
        (vocab_size (make case)))

let token_law =
  prop ~count:60 "gives each token its id" built (fun ((_, drawn, _) as case) ->
      cover_built case;
      let t = make case in
      List.iter
        (fun (token, id) ->
          equal ~msg:token (option int) (Some id) (token_to_id t token))
        drawn;
      List.iter
        (fun c ->
          equal ~msg:c (option string) (Some c)
            (Option.bind (token_to_id t c) (id_to_token t)))
        (lacked case))

let id_law =
  prop ~count:60 "gives each id its token" built (fun ((_, drawn, _) as case) ->
      cover_built case;
      let t = make case in
      List.iter
        (fun (token, id) ->
          equal ~msg:(string_of_int id) (option string) (Some token)
            (id_to_token t id))
        drawn)

let unheld_token =
  cases "is None for a token the tokenizer does not hold"
    ~id:(fun (id, _, _) -> id)
    [
      ("bpe", (fun () -> bpe ~vocab:[ ("a", 0) ] ()), [ "b"; ""; "<unk>" ]);
      ( "wordpiece",
        (fun () -> wordpiece ~vocab:[ ("[UNK]", 0) ] ()),
        [ "a"; "##a" ] );
      ( "word_level",
        (fun () -> word_level ~vocab:[ ("a", 0) ] ()),
        [ "b"; "<unk>" ] );
      ( "unigram",
        (fun () -> unigram ~vocab:[ ("hello", 0.0); ("world", 0.0) ] ()),
        [ "missing"; "" ] );
      ("unigram, empty", (fun () -> unigram ~vocab:[] ()), [ "test" ]);
      ("chars", (fun () -> chars ()), [ "ab"; "" ]);
    ]
    (fun (_, make, tokens) ->
      let t = make () in
      List.iter
        (fun token -> equal ~msg:token (option int) None (token_to_id t token))
        tokens)

let unheld_id =
  let beyond = [ -1; 1 lsl 31; max_int; min_int ] in
  cases "is None for an id no token holds"
    ~id:(fun (id, _, _) -> id)
    [
      ( "bpe",
        (fun () -> bpe ~vocab:[ ("a", 0); ("b", 2) ] ()),
        [ 1; 3 ] @ beyond );
      ( "wordpiece",
        (fun () -> wordpiece ~vocab:[ ("[UNK]", 0); ("a", 2) ] ()),
        [ 1; 3 ] @ beyond );
      ( "word_level",
        (fun () -> word_level ~vocab:[ ("a", 0); ("b", 2) ] ()),
        [ 1; 3 ] @ beyond );
      ( "unigram",
        (fun () -> unigram ~vocab:[ ("hello", 0.0); ("world", 0.0) ] ()),
        [ 2; 10 ] @ beyond );
      ("unigram, empty", (fun () -> unigram ~vocab:[] ()), [ 0 ] @ beyond);
      ("chars", (fun () -> chars ()), [ 256 ] @ beyond);
    ]
    (fun (_, make, ids) ->
      let t = make () in
      List.iter
        (fun id ->
          equal ~msg:(string_of_int id) (option string) None (id_to_token t id))
        ids)

(* A SentencePiece pipeline marks the start of the text with ▁ before an added
   token matched against normalized text can match. *)
let test_normalized_form () =
  let marker = "\u{2581}" in
  let t =
    bpe
      ~normalizer:
        (Normalizer.sequence
           [
             Normalizer.prepend marker;
             Normalizer.replace ~pattern:" " ~replacement:marker;
           ])
      ~added_tokens:[ added_token ~normalized:true "<s>" ]
      ~vocab:[ ("<s>", 1); (marker ^ "a", 2) ]
      ~merges:[] ()
  in
  equal ~msg:"id_to_token" (option string)
    (Some (marker ^ "<s>"))
    (id_to_token t 1);
  equal ~msg:"token_to_id" (option int) (Some 1) (token_to_id t "<s>")

let () =
  exit
    (run "Accessors and vocabulary"
       [
         group "normalizer" [ given_normalizer ];
         group "pre_tokenizer"
           [
             given_pre;
             test "is whitespace for a word_level built without one"
               test_word_level_pre;
           ];
         group "post_processor" [ given_post ];
         group "decoder" [ given_decoder ];
         group "added_tokens"
           [
             test "lists the tokens given, then the role markers not among them"
               test_added_tokens;
           ];
         group "bos_token"
           [
             test "is the token given, or None" (fun () ->
                 role bos_token (fun bos_token -> bpe ?bos_token ()));
           ];
         group "eos_token"
           [
             test "is the token given, or None" (fun () ->
                 role eos_token (fun eos_token -> bpe ?eos_token ()));
           ];
         group "pad_token"
           [
             test "is the token given, or None" (fun () ->
                 role pad_token (fun pad_token -> bpe ?pad_token ()));
           ];
         group "unk_token"
           [
             test "is the token given, or None" (fun () ->
                 role unk_token (fun unk_token -> bpe ?unk_token ()));
           ];
         group "vocab" [ empty_vocab; vocab_law ];
         group "vocab_size" [ size_law ];
         group "token_to_id" [ token_law; unheld_token ];
         group "id_to_token"
           [
             id_law;
             unheld_id;
             test "gives a normalized added token's normalized form"
               test_normalized_form;
           ];
       ])
