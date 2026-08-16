(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Added tokens are matched atomically in the input. Every expectation below was
   read off HuggingFace [tokenizers] 0.23.1 first, from a BPE model with the
   same vocabulary and no merges, so that the model never absorbs an added
   token:

   t = Tokenizer(BPE(vocab={c: i for i, c in enumerate("abcdxz._-1 \t\nSWAB")},
   merges=[])) t.add_tokens([AddedToken("abc", normalized=False), ...])
   t.encode("xabc").ids # [4, 18]

   Note that HuggingFace reports [special_tokens_mask] 0 for an added token
   found in the input, whether it is special or not; only the tokens a
   post-processor inserts are masked. *)

open Windtrap
open Brot

let alphabet = "abcdxz._-1 \t\nSWAB"

let model_vocab =
  List.init (String.length alphabet) (fun i -> (String.make 1 alphabet.[i], i))

let tokenizer ?normalizer specials =
  bpe ?normalizer ~vocab:model_vocab ~merges:[] ~specials ()

let ids_of tokenizer text = Encoding.ids (encode tokenizer text)

let check_encode tokenizer text ~ids ~tokens ~offsets =
  let encoding = encode tokenizer text in
  equal
    ~msg:(Printf.sprintf "%S ids" text)
    (array int) ids (Encoding.ids encoding);
  equal
    ~msg:(Printf.sprintf "%S tokens" text)
    (array string) tokens (Encoding.tokens encoding);
  equal
    ~msg:(Printf.sprintf "%S offsets" text)
    (array (pair int int))
    offsets
    (Encoding.offsets encoding);
  equal
    ~msg:(Printf.sprintf "%S ids via encode_ids" text)
    (array int) ids
    (encode_ids tokenizer text)

(* At a given position the longest content wins; earlier positions win over
   later ones. *)
let test_leftmost_longest () =
  let t = tokenizer [ special "ab"; special "abc"; special "bc" ] in
  equal ~msg:"ab" (option int) (Some 17) (token_to_id t "ab");
  equal ~msg:"abc" (option int) (Some 18) (token_to_id t "abc");
  equal ~msg:"bc" (option int) (Some 19) (token_to_id t "bc");
  equal ~msg:"vocab size" int 20 (vocab_size t);
  equal ~msg:"abc" (array int) [| 18 |] (ids_of t "abc");
  equal ~msg:"xabc" (array int) [| 4; 18 |] (ids_of t "xabc");
  equal ~msg:"abx" (array int) [| 17; 4 |] (ids_of t "abx");
  equal ~msg:"aabc" (array int) [| 0; 18 |] (ids_of t "aabc");
  equal ~msg:"zbcz" (array int) [| 5; 19; 5 |] (ids_of t "zbcz");
  equal ~msg:"abcabc" (array int) [| 18; 18 |] (ids_of t "abcabc")

let test_placement () =
  let t = tokenizer [ special "<s>" ] in
  check_encode t "<s>" ~ids:[| 17 |] ~tokens:[| "<s>" |] ~offsets:[| (0, 3) |];
  check_encode t "<s>a" ~ids:[| 17; 0 |] ~tokens:[| "<s>"; "a" |]
    ~offsets:[| (0, 3); (3, 4) |];
  check_encode t "a<s>" ~ids:[| 0; 17 |] ~tokens:[| "a"; "<s>" |]
    ~offsets:[| (0, 1); (1, 4) |];
  check_encode t "a<s>b" ~ids:[| 0; 17; 1 |] ~tokens:[| "a"; "<s>"; "b" |]
    ~offsets:[| (0, 1); (1, 4); (4, 5) |];
  check_encode t "<s><s>" ~ids:[| 17; 17 |] ~tokens:[| "<s>"; "<s>" |]
    ~offsets:[| (0, 3); (3, 6) |];
  check_encode t "" ~ids:[||] ~tokens:[||] ~offsets:[||]

(* An added token found in the input is never masked, special or not. *)
let test_special_tokens_mask () =
  let t = tokenizer [ special "<s>" ] in
  equal ~msg:"special added token" (array int) [| 0; 0; 0 |]
    (Encoding.special_tokens_mask (encode t "a<s>b"));
  let t = tokenizer [ special ~special:false "<s>" ] in
  equal ~msg:"plain added token" (array int) [| 0; 0; 0 |]
    (Encoding.special_tokens_mask (encode t "a<s>b"))

let test_decode_skips_special_only () =
  let t = tokenizer [ special "<s>" ] in
  equal ~msg:"special kept" string "a<s>b" (decode t (ids_of t "a<s>b"));
  equal ~msg:"special skipped" string "ab"
    (decode t ~skip_special_tokens:true (ids_of t "a<s>b"));
  let t = tokenizer [ special ~special:false "<s>" ] in
  equal ~msg:"plain kept" string "a<s>b" (decode t (ids_of t "a<s>b"));
  equal ~msg:"plain not skipped" string "a<s>b"
    (decode t ~skip_special_tokens:true (ids_of t "a<s>b"))

(* An added token the model does not hold is numbered from the end of the model
   vocabulary and reachable from both sides. *)
let test_id_beyond_vocabulary () =
  let t = tokenizer [ special "<s>"; special "<pad>" ] in
  equal ~msg:"model size" int 17 (List.length model_vocab);
  equal ~msg:"<s>" (option int) (Some 17) (token_to_id t "<s>");
  equal ~msg:"<pad>" (option int) (Some 18) (token_to_id t "<pad>");
  equal ~msg:"id 17" (option string) (Some "<s>") (id_to_token t 17);
  equal ~msg:"id 18" (option string) (Some "<pad>") (id_to_token t 18);
  equal ~msg:"vocab size" int 19 (vocab_size t);
  equal ~msg:"vocab entries" int 19 (List.length (vocab t))

(* Two entries with the same content are one token: it keeps the identifier of
   the first and the flags of the last. *)
let test_duplicate_content () =
  let t = tokenizer [ special "<s>"; special ~lstrip:true "<s>" ] in
  equal ~msg:"one token" int 1 (List.length (specials t));
  equal ~msg:"first identifier" (option int) (Some 17) (token_to_id t "<s>");
  equal ~msg:"one identifier taken" int 18 (vocab_size t);
  equal ~msg:"last flags" (array string) [| "a"; " <s>" |]
    (Encoding.tokens (encode t "a <s>"))

(* Content that is no text is no token, and takes no identifier. *)
let test_empty_content () =
  let t = tokenizer [ special ""; special "<s>" ] in
  equal ~msg:"one token" int 1 (List.length (specials t));
  equal ~msg:"identifier not consumed" (option int) (Some 17)
    (token_to_id t "<s>");
  equal ~msg:"vocab size" int 18 (vocab_size t)

(* The unknown token configures the model; it is never matched atomically.
   HuggingFace agrees: BPE(vocab, merges=[], unk_token="<unk>") encodes
   "a<unk>a" as ['a', '<', 'u', 'n', 'k', '>', 'a'] and has no added token. *)
let test_unk_token_is_not_added () =
  let vocab = [ ("a", 0); ("<", 1); ("u", 2); ("n", 3); ("k", 4); (">", 5) ] in
  let t = bpe ~vocab ~merges:[] ~unk_token:"<unk>" () in
  equal ~msg:"no added token" int 0 (List.length (specials t));
  equal ~msg:"no identifier" (option int) None (token_to_id t "<unk>");
  equal ~msg:"vocabulary unchanged" int 6 (vocab_size t);
  equal ~msg:"tokenized as text" (array int) [| 0; 1; 2; 3; 4; 5; 0 |]
    (ids_of t "a<unk>a")

(* A role marker the model holds is a special token in its own right; one it
   does not hold stays a name. *)
let test_role_tokens () =
  let vocab = [ ("a", 0); ("b", 1); ("[PAD]", 2) ] in
  let held = bpe ~vocab ~merges:[] ~pad_token:"[PAD]" () in
  equal ~msg:"promoted" (list string) [ "[PAD]" ]
    (List.map (fun (s : special) -> s.token) (specials held));
  equal ~msg:"matched atomically" (array int) [| 0; 2; 1 |]
    (ids_of held "a[PAD]b");
  equal ~msg:"skipped when decoding" string "ab"
    (decode held ~skip_special_tokens:true (ids_of held "a[PAD]b"));
  let absent =
    bpe ~vocab:[ ("a", 0); ("b", 1) ] ~merges:[] ~pad_token:"[PAD]" ()
  in
  equal ~msg:"not promoted" int 0 (List.length (specials absent));
  equal ~msg:"vocabulary unchanged" int 2 (vocab_size absent)

(* A token in the model keeps the model's identifier. *)
let test_id_from_model () =
  let t = tokenizer [ special "a"; special "<s>" ] in
  equal ~msg:"a" (option int) (Some 0) (token_to_id t "a");
  equal ~msg:"<s>" (option int) (Some 17) (token_to_id t "<s>");
  equal ~msg:"vocab size" int 18 (vocab_size t)

(* A match counts only when neither neighbour is a word character: the letters,
   the marks, the decimal digits, the connector punctuation and the joiners. *)
let test_single_word () =
  let t = tokenizer [ special ~single_word:true "SW" ] in
  equal ~msg:"SW" (array int) [| 17 |] (ids_of t "SW");
  equal ~msg:".SW." (array int) [| 6; 17; 6 |] (ids_of t ".SW.");
  equal ~msg:"-SW-" (array int) [| 8; 17; 8 |] (ids_of t "-SW-");
  equal ~msg:"x SW x" (array int) [| 4; 10; 17; 10; 4 |] (ids_of t "x SW x");
  equal ~msg:"aSW" (array int) [| 0; 13; 14 |] (ids_of t "aSW");
  equal ~msg:"SWa" (array int) [| 13; 14; 0 |] (ids_of t "SWa");
  equal ~msg:"1SW" (array int) [| 9; 13; 14 |] (ids_of t "1SW");
  equal ~msg:"_SW_" (array int) [| 7; 13; 14; 7 |] (ids_of t "_SW_")

(* A discarded single_word match does not fall back to a shorter token at the
   same position: the search resumes past the match. *)
let test_single_word_no_fallback () =
  let t = tokenizer [ special ~single_word:true "abc"; special "ab" ] in
  equal ~msg:"abc" (option int) (Some 17) (token_to_id t "abc");
  equal ~msg:"ab" (option int) (Some 18) (token_to_id t "ab");
  equal ~msg:"abcd" (array int) [| 0; 1; 2; 3 |] (ids_of t "abcd");
  equal ~msg:"abc." (array int) [| 17; 6 |] (ids_of t "abc.");
  equal ~msg:"ab." (array int) [| 18; 6 |] (ids_of t "ab.")

(* The stripped white space is part of the match, and of the token it emits. *)
let test_lstrip () =
  let t = tokenizer [ special ~lstrip:true "L" ] in
  check_encode t "a  L b" ~ids:[| 0; 17; 10; 1 |]
    ~tokens:[| "a"; "  L"; " "; "b" |]
    ~offsets:[| (0, 1); (1, 4); (4, 5); (5, 6) |];
  check_encode t "L b" ~ids:[| 17; 10; 1 |] ~tokens:[| "L"; " "; "b" |]
    ~offsets:[| (0, 1); (1, 2); (2, 3) |];
  check_encode t "aL" ~ids:[| 0; 17 |] ~tokens:[| "a"; "L" |]
    ~offsets:[| (0, 1); (1, 2) |]

let test_rstrip () =
  let t = tokenizer [ special ~rstrip:true "R" ] in
  check_encode t "a R  b" ~ids:[| 0; 10; 17; 1 |]
    ~tokens:[| "a"; " "; "R  "; "b" |]
    ~offsets:[| (0, 1); (1, 2); (2, 5); (5, 6) |];
  check_encode t "a R" ~ids:[| 0; 10; 17 |] ~tokens:[| "a"; " "; "R" |]
    ~offsets:[| (0, 1); (1, 2); (2, 3) |]

(* White space already taken by a preceding rstrip is not taken again. *)
let test_lstrip_after_rstrip () =
  let t = tokenizer [ special ~lstrip:true "L"; special ~rstrip:true "R" ] in
  check_encode t "R  L" ~ids:[| 18; 17 |] ~tokens:[| "R  "; "L" |]
    ~offsets:[| (0, 3); (3, 4) |]

(* A token with [normalized] set is matched against the normalized text, and
   emits what it matched there; without it, against the raw input. *)
let test_normalized () =
  let normalizer = Normalizer.lowercase in
  let t = tokenizer ~normalizer [ special "AB" ] in
  check_encode t "xABx" ~ids:[| 4; 17; 4 |] ~tokens:[| "x"; "AB"; "x" |]
    ~offsets:[| (0, 1); (1, 3); (3, 4) |];
  equal ~msg:"xabx" (array int) [| 4; 0; 1; 4 |] (ids_of t "xabx");
  let t = tokenizer ~normalizer [ special ~normalized:true "AB" ] in
  equal ~msg:"content is the unnormalized form" (option int) (Some 17)
    (token_to_id t "AB");
  equal ~msg:"normalized form is not a token" (option int) None
    (token_to_id t "ab");
  check_encode t "xABx" ~ids:[| 4; 17; 4 |] ~tokens:[| "x"; "ab"; "x" |]
    ~offsets:[| (0, 1); (1, 3); (3, 4) |];
  check_encode t "xabx" ~ids:[| 4; 17; 4 |] ~tokens:[| "x"; "ab"; "x" |]
    ~offsets:[| (0, 1); (1, 3); (3, 4) |]

(* Tokens matched against the raw input are split out first, so a normalizer
   cannot hide them. *)
let test_normalized_after_raw () =
  let t =
    tokenizer ~normalizer:Normalizer.lowercase
      [ special "AB"; special ~normalized:true "abc" ]
  in
  equal ~msg:"AB" (option int) (Some 17) (token_to_id t "AB");
  equal ~msg:"abc" (option int) (Some 18) (token_to_id t "abc");
  equal ~msg:"ABc" (array int) [| 17; 2 |] (ids_of t "ABc");
  equal ~msg:"abc" (array int) [| 18 |] (ids_of t "abc")

(* Matching does not depend on the post-processor running. *)
let test_add_special_tokens_false () =
  let t = tokenizer [ special "<s>" ] in
  equal ~msg:"encode" (array int) [| 0; 17; 1 |]
    (Encoding.ids (encode t ~add_special_tokens:false "a<s>b"));
  equal ~msg:"encode_ids" (array int) [| 0; 17; 1 |]
    (encode_ids t ~add_special_tokens:false "a<s>b")

let test_json_round_trip () =
  let t =
    tokenizer
      [
        special "<s>";
        special ~special:false ~single_word:true ~lstrip:true "<p>";
        special ~normalized:true ~rstrip:true "<q>";
      ]
  in
  match from_json (to_json t) with
  | Error msg -> failf "round trip failed: %s" msg
  | Ok reloaded ->
      let flags (s : special) =
        Printf.sprintf
          "%s special=%b single_word=%b lstrip=%b rstrip=%b normalized=%b"
          s.token s.special s.single_word s.lstrip s.rstrip s.normalized
      in
      equal ~msg:"specials" (list string)
        (List.map flags (specials t))
        (List.map flags (specials reloaded));
      equal ~msg:"<s>" (option int) (Some 17) (token_to_id reloaded "<s>");
      equal ~msg:"<p>" (option int) (Some 18) (token_to_id reloaded "<p>");
      equal ~msg:"<q>" (option int) (Some 19) (token_to_id reloaded "<q>");
      equal ~msg:"ids" (array int) (ids_of t "a<s>b") (ids_of reloaded "a<s>b");
      equal ~msg:"lstrip survives" (array int) [| 0; 18 |]
        (ids_of reloaded "a <p>");
      equal ~msg:"special survives" string "ab"
        (decode reloaded ~skip_special_tokens:true (ids_of reloaded "a<s>b"));
      equal ~msg:"plain survives" string "a<p>"
        (decode reloaded ~skip_special_tokens:true (ids_of reloaded "a <p>"))

let suite =
  [
    test "leftmost longest" test_leftmost_longest;
    test "placement" test_placement;
    test "special tokens mask" test_special_tokens_mask;
    test "decode skips special only" test_decode_skips_special_only;
    test "id beyond vocabulary" test_id_beyond_vocabulary;
    test "duplicate content" test_duplicate_content;
    test "empty content" test_empty_content;
    test "unk token is not added" test_unk_token_is_not_added;
    test "role tokens" test_role_tokens;
    test "id from model" test_id_from_model;
    test "single word" test_single_word;
    test "single word no fallback" test_single_word_no_fallback;
    test "lstrip" test_lstrip;
    test "rstrip" test_rstrip;
    test "lstrip after rstrip" test_lstrip_after_rstrip;
    test "normalized" test_normalized;
    test "normalized after raw" test_normalized_after_raw;
    test "add_special_tokens false" test_add_special_tokens_false;
    test "json round trip" test_json_round_trip;
  ]

let () = run "Added tokens" [ group "added_tokens" suite ]
