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

let tokenizer ?normalizer tokens =
  bpe ?normalizer ~vocab:model_vocab ~merges:[] ~added_tokens:tokens ()

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
  let t = tokenizer [ added_token "ab"; added_token "abc"; added_token "bc" ] in
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
  let t = tokenizer [ added_token "<s>" ] in
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
  let t = tokenizer [ added_token "<s>" ] in
  equal ~msg:"special added token" (array int) [| 0; 0; 0 |]
    (Encoding.special_tokens_mask (encode t "a<s>b"));
  let t = tokenizer [ added_token ~special:false "<s>" ] in
  equal ~msg:"plain added token" (array int) [| 0; 0; 0 |]
    (Encoding.special_tokens_mask (encode t "a<s>b"))

let test_decode_skips_special_only () =
  let t = tokenizer [ added_token "<s>" ] in
  equal ~msg:"special kept" string "a <s> b" (decode t (ids_of t "a<s>b"));
  equal ~msg:"special skipped" string "a b"
    (decode t ~skip_special_tokens:true (ids_of t "a<s>b"));
  let t = tokenizer [ added_token ~special:false "<s>" ] in
  equal ~msg:"plain kept" string "a <s> b" (decode t (ids_of t "a<s>b"));
  equal ~msg:"plain not skipped" string "a <s> b"
    (decode t ~skip_special_tokens:true (ids_of t "a<s>b"))

(* An added token the model does not hold is numbered from the end of the model
   vocabulary and reachable from both sides. *)
let test_id_beyond_vocabulary () =
  let t = tokenizer [ added_token "<s>"; added_token "<pad>" ] in
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
  let t = tokenizer [ added_token "<s>"; added_token ~lstrip:true "<s>" ] in
  equal ~msg:"one token" int 1 (List.length (added_tokens t));
  equal ~msg:"first identifier" (option int) (Some 17) (token_to_id t "<s>");
  equal ~msg:"one identifier taken" int 18 (vocab_size t);
  equal ~msg:"last flags" (array string) [| "a"; " <s>" |]
    (Encoding.tokens (encode t "a <s>"))

(* Content that is no text is no token, and takes no identifier. *)
let test_empty_content () =
  let t = tokenizer [ added_token ""; added_token "<s>" ] in
  equal ~msg:"one token" int 1 (List.length (added_tokens t));
  equal ~msg:"identifier not consumed" (option int) (Some 17)
    (token_to_id t "<s>");
  equal ~msg:"vocab size" int 18 (vocab_size t)

(* The unknown token configures the model; it is never matched atomically.
   HuggingFace agrees: BPE(vocab, merges=[], unk_token="<unk>") encodes
   "a<unk>a" as ['a', '<', 'u', 'n', 'k', '>', 'a'] and has no added token. *)
let test_unk_token_is_not_added () =
  let vocab = [ ("a", 0); ("<", 1); ("u", 2); ("n", 3); ("k", 4); (">", 5) ] in
  let t = bpe ~vocab ~merges:[] ~unk_token:"<unk>" () in
  equal ~msg:"no added token" int 0 (List.length (added_tokens t));
  equal ~msg:"no identifier" (option int) None (token_to_id t "<unk>");
  equal ~msg:"vocabulary unchanged" int 6 (vocab_size t);
  equal ~msg:"tokenized as text" (array int) [| 0; 1; 2; 3; 4; 5; 0 |]
    (ids_of t "a<unk>a")

(* A role marker is a special token in its own right, whether or not the model
   holds it; it never enters the model's own vocabulary. *)
let test_role_tokens () =
  let held =
    bpe
      ~vocab:[ ("a", 0); ("b", 1); ("[PAD]", 2) ]
      ~merges:[] ~pad_token:"[PAD]" ()
  in
  equal ~msg:"registered" (list string) [ "[PAD]" ]
    (List.map (fun (a : added_token) -> a.content) (added_tokens held));
  equal ~msg:"keeps the model id" (array int) [| 0; 2; 1 |]
    (ids_of held "a[PAD]b");
  equal ~msg:"skipped when decoding" string "a b"
    (decode held ~skip_special_tokens:true (ids_of held "a[PAD]b"));
  let absent =
    bpe ~vocab:[ ("a", 0); ("b", 1) ] ~merges:[] ~pad_token:"[PAD]" ()
  in
  equal ~msg:"registered too" (list string) [ "[PAD]" ]
    (List.map (fun (a : added_token) -> a.content) (added_tokens absent));
  equal ~msg:"numbered after the vocabulary" (option int) (Some 2)
    (token_to_id absent "[PAD]");
  equal ~msg:"vocab size" int 3 (vocab_size absent);
  equal ~msg:"matched atomically" (array int) [| 0; 2; 1 |]
    (ids_of absent "a[PAD]b")

(* A token in the model keeps the model's identifier. *)
let test_id_from_model () =
  let t = tokenizer [ added_token "a"; added_token "<s>" ] in
  equal ~msg:"a" (option int) (Some 0) (token_to_id t "a");
  equal ~msg:"<s>" (option int) (Some 17) (token_to_id t "<s>");
  equal ~msg:"vocab size" int 18 (vocab_size t)

(* A match counts only when neither neighbour is a word character: the letters,
   the marks, the decimal digits, the connector punctuation and the joiners. *)
let test_single_word () =
  let t = tokenizer [ added_token ~single_word:true "SW" ] in
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
  let t = tokenizer [ added_token ~single_word:true "abc"; added_token "ab" ] in
  equal ~msg:"abc" (option int) (Some 17) (token_to_id t "abc");
  equal ~msg:"ab" (option int) (Some 18) (token_to_id t "ab");
  equal ~msg:"abcd" (array int) [| 0; 1; 2; 3 |] (ids_of t "abcd");
  equal ~msg:"abc." (array int) [| 17; 6 |] (ids_of t "abc.");
  equal ~msg:"ab." (array int) [| 18; 6 |] (ids_of t "ab.")

(* The stripped white space is part of the match, and of the token it emits. *)
let test_lstrip () =
  let t = tokenizer [ added_token ~lstrip:true "L" ] in
  check_encode t "a  L b" ~ids:[| 0; 17; 10; 1 |]
    ~tokens:[| "a"; "  L"; " "; "b" |]
    ~offsets:[| (0, 1); (1, 4); (4, 5); (5, 6) |];
  check_encode t "L b" ~ids:[| 17; 10; 1 |] ~tokens:[| "L"; " "; "b" |]
    ~offsets:[| (0, 1); (1, 2); (2, 3) |];
  check_encode t "aL" ~ids:[| 0; 17 |] ~tokens:[| "a"; "L" |]
    ~offsets:[| (0, 1); (1, 2) |]

let test_rstrip () =
  let t = tokenizer [ added_token ~rstrip:true "R" ] in
  check_encode t "a R  b" ~ids:[| 0; 10; 17; 1 |]
    ~tokens:[| "a"; " "; "R  "; "b" |]
    ~offsets:[| (0, 1); (1, 2); (2, 5); (5, 6) |];
  check_encode t "a R" ~ids:[| 0; 10; 17 |] ~tokens:[| "a"; " "; "R" |]
    ~offsets:[| (0, 1); (1, 2); (2, 3) |]

(* White space already taken by a preceding rstrip is not taken again. *)
let test_lstrip_after_rstrip () =
  let t =
    tokenizer [ added_token ~lstrip:true "L"; added_token ~rstrip:true "R" ]
  in
  check_encode t "R  L" ~ids:[| 18; 17 |] ~tokens:[| "R  "; "L" |]
    ~offsets:[| (0, 3); (3, 4) |]

(* A token with [normalized] set is matched against the normalized text, and
   emits what it matched there; without it, against the raw input. *)
let test_normalized () =
  let normalizer = Normalizer.lowercase in
  let t = tokenizer ~normalizer [ added_token "AB" ] in
  check_encode t "xABx" ~ids:[| 4; 17; 4 |] ~tokens:[| "x"; "AB"; "x" |]
    ~offsets:[| (0, 1); (1, 3); (3, 4) |];
  equal ~msg:"xabx" (array int) [| 4; 0; 1; 4 |] (ids_of t "xabx");
  let t = tokenizer ~normalizer [ added_token ~normalized:true "AB" ] in
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
      [ added_token "AB"; added_token ~normalized:true "abc" ]
  in
  equal ~msg:"AB" (option int) (Some 17) (token_to_id t "AB");
  equal ~msg:"abc" (option int) (Some 18) (token_to_id t "abc");
  equal ~msg:"ABc" (array int) [| 17; 2 |] (ids_of t "ABc");
  equal ~msg:"abc" (array int) [| 18 |] (ids_of t "abc")

(* Matching does not depend on the post-processor running. *)
let test_add_special_tokens_false () =
  let t = tokenizer [ added_token "<s>" ] in
  equal ~msg:"encode" (array int) [| 0; 17; 1 |]
    (Encoding.ids (encode t ~add_special_tokens:false "a<s>b"));
  equal ~msg:"encode_ids" (array int) [| 0; 17; 1 |]
    (encode_ids t ~add_special_tokens:false "a<s>b")

let test_json_round_trip () =
  let t =
    tokenizer
      [
        added_token "<s>";
        added_token ~special:false ~single_word:true ~lstrip:true "<p>";
        added_token ~normalized:true ~rstrip:true "<q>";
      ]
  in
  match of_json (to_json t) with
  | Error msg -> failf "round trip failed: %s" msg
  | Ok reloaded ->
      let flags (a : added_token) =
        Printf.sprintf
          "%s special=%b single_word=%b lstrip=%b rstrip=%b normalized=%b"
          a.content a.special a.single_word a.lstrip a.rstrip a.normalized
      in
      equal ~msg:"added tokens" (list string)
        (List.map flags (added_tokens t))
        (List.map flags (added_tokens reloaded));
      equal ~msg:"<s>" (option int) (Some 17) (token_to_id reloaded "<s>");
      equal ~msg:"<p>" (option int) (Some 18) (token_to_id reloaded "<p>");
      equal ~msg:"<q>" (option int) (Some 19) (token_to_id reloaded "<q>");
      equal ~msg:"ids" (array int) (ids_of t "a<s>b") (ids_of reloaded "a<s>b");
      equal ~msg:"lstrip survives" (array int) [| 0; 18 |]
        (ids_of reloaded "a <p>");
      equal ~msg:"special survives" string "a b"
        (decode reloaded ~skip_special_tokens:true (ids_of reloaded "a<s>b"));
      equal ~msg:"plain survives" string "a <p>"
        (decode reloaded ~skip_special_tokens:true (ids_of reloaded "a <p>"))

(* The scan skips to the next byte that can start a token — with a SWAR search
   when the tokens start with one or two distinct bytes, bytewise past that. The
   tests below pin each shape, matches at every alignment of the eight-byte
   loop, and near-matches that trigger the scan without completing a token. *)

let scan_vocab =
  let chars = "abcs<>[]{}d" in
  List.init (String.length chars) (fun i -> (String.make 1 chars.[i], i))

let scan_tokenizer tokens =
  bpe ~vocab:scan_vocab ~merges:[] ~added_tokens:tokens ()

let test_scan_offsets () =
  let t = scan_tokenizer [ added_token "<s>" ] in
  for o = 0 to 20 do
    let text = String.make o 'a' ^ "<s>" ^ String.make (20 - o) 'b' in
    let ids =
      Array.concat [ Array.make o 0; [| 11 |]; Array.make (20 - o) 1 ]
    in
    equal ~msg:(Printf.sprintf "match at %d" o) (array int) ids (ids_of t text);
    let offsets = Encoding.offsets (encode t text) in
    equal
      ~msg:(Printf.sprintf "offsets at %d" o)
      (pair int int)
      (o, o + 3)
      offsets.(o)
  done

let test_scan_two_firsts () =
  let t = scan_tokenizer [ added_token "<s>"; added_token "[c]" ] in
  equal ~msg:"both found" (array int)
    [| 0; 6; 2; 0; 4; 3; 0; 12; 0; 11; 0 |]
    (ids_of t "a[ca<sa[c]a<s>a");
  equal ~msg:"near matches around a match" (array int)
    [| 4; 4; 3; 12; 4; 5; 3 |] (ids_of t "<<s[c]<>s");
  (* The [ sits in a full SWAR window holding no <: a scan seeking only the
     first first byte would skip it. *)
  equal ~msg:"second first byte alone in a full window" (array int)
    (Array.concat [ Array.make 10 0; [| 12 |]; Array.make 10 0 ])
    (ids_of t (String.make 10 'a' ^ "[c]" ^ String.make 10 'a'))

let test_scan_three_firsts () =
  let t =
    scan_tokenizer [ added_token "<s>"; added_token "[c]"; added_token "{d}" ]
  in
  equal ~msg:"all found" (array int)
    [| 0; 13; 1; 11; 2; 12; 10 |]
    (ids_of t "a{d}b<s>c[c]d")

(* HuggingFace's [test_encode_special_tokens] in [added_vocabulary.rs]: with
   [encode_special_tokens] set, the special [<mask>] and [<pad>] are passed
   over, while [ask>], matched against normalized text, is still found inside
   the text of [<mask>]. *)
let test_unmatched_special () =
  let chars = "hi<mask>terpd \t\u{2000}" in
  let rec scalars i =
    if i >= String.length chars then []
    else
      let d = String.get_utf_8_uchar chars i in
      let n = Uchar.utf_decode_length d in
      String.sub chars i n :: scalars (i + n)
  in
  let vocab = List.mapi (fun i c -> (c, i)) (scalars 0) in
  let t =
    bpe ~normalizer:Normalizer.lowercase ~vocab ~merges:[]
      ~added_tokens:
        [
          added_token ~lstrip:true ~rstrip:true ~single_word:true "<mask>";
          added_token ~special:false "ask>";
          added_token "<pad>";
        ]
      ()
  in
  let text = "Hi <mask> there\t<mask>\t<mask>\u{2000} <pad> <mask><pad><pad>" in
  let pieces =
    [ "h"; "i"; " "; "<"; "m"; "ask>"; " "; "t"; "h"; "e"; "r"; "e"; "\t"; "<";
      "m"; "ask>"; "\t"; "<"; "m"; "ask>"; "\u{2000}"; " "; "<"; "p"; "a"; "d";
      ">"; " "; "<"; "m"; "ask>"; "<"; "p"; "a"; "d"; ">"; "<"; "p"; "a"; "d";
      ">" ]
  in
  let ask = Option.get (token_to_id t "ask>") in
  let encoding = encode t ~match_special_tokens:false text in
  equal ~msg:"tokens" (list string) pieces
    (Array.to_list (Encoding.tokens encoding));
  equal ~msg:"ids" (list int)
    (List.map
       (fun piece ->
         if piece = "ask>" then ask else Option.get (token_to_id t piece))
       pieces)
    (Array.to_list (Encoding.ids encoding));
  equal ~msg:"ids via encode_ids" (array int) (Encoding.ids encoding)
    (encode_ids t ~match_special_tokens:false text);
  equal ~msg:"matched by default" (list string)
    [ "h"; "i"; " <mask> "; "t"; "h"; "e"; "r"; "e"; "\t<mask>\t";
      "<mask>\u{2000} "; "<pad>"; " <mask>"; "<pad>"; "<pad>" ]
    (Array.to_list (Encoding.tokens (encode t text)))

let test_scan_dense_near () =
  let t = scan_tokenizer [ added_token "<s>" ] in
  equal ~msg:"near matches all along" (array int) [| 4; 4; 3; 11; 3; 5 |]
    (ids_of t "<<s<s>s>")

let test_scan_long_stretch () =
  let t = scan_tokenizer [ added_token "<s>" ] in
  let run = String.make 100 'a' in
  let ids = ids_of t (run ^ "<s>" ^ run ^ "<s>") in
  equal ~msg:"two matches" (array int)
    (Array.concat [ Array.make 100 0; [| 11 |]; Array.make 100 0; [| 11 |] ])
    ids;
  equal ~msg:"match first" (array int)
    (Array.append [| 11 |] (Array.make 100 0))
    (ids_of t ("<s>" ^ run));
  equal ~msg:"no match" (array int) (Array.make 200 0) (ids_of t (run ^ run))

let test_scan_single_word_resume () =
  let t = tokenizer [ added_token ~single_word:true "SW" ] in
  equal ~msg:"refused then matched" (array int)
    (Array.concat
       [ Array.make 10 0; [| 13; 14 |]; Array.make 10 0; [| 6; 17; 6 |] ])
    (ids_of t "aaaaaaaaaaSWaaaaaaaaaa.SW.")

let test_scan_long_space_run () =
  let t = tokenizer [ added_token ~lstrip:true "L" ] in
  check_encode t
    ("a" ^ String.make 12 ' ' ^ "L")
    ~ids:[| 0; 17 |]
    ~tokens:[| "a"; String.make 12 ' ' ^ "L" |]
    ~offsets:[| (0, 1); (1, 14) |]

(* The scan, against a reference that tries every token at every position.

   The text and the tokens are drawn from characters whose classes the
   reference knows: word characters, white space (Unicode's too), others, and
   bytes of no character, among them the first two bytes of the three of [▁].
   Token contents often share a prefix, as SentencePiece's [▁<s>] and [▁</s>]
   do once normalized, so the scan has to tell them apart past it. *)

let word_chars = [ "a"; "b"; "A"; "x"; "_"; "1"; "\u{e9}"; "\u{65e5}" ]
let space_chars = [ " "; "\t"; "\n"; "\u{a0}"; "\u{3000}" ]
let other_chars = [ "<"; ">"; "/"; "."; "-"; "\u{2581}" ]
let stray_bytes = [ "\xff"; "\xc3"; "\xe2\x96" ]
let law_chars = word_chars @ space_chars @ other_chars

(* No pretoken cache: building one per case is what would cost. *)
let law_tokenizer ?normalizer added_tokens =
  bpe ?normalizer
    ~vocab:(List.mapi (fun i c -> (c, i)) law_chars)
    ~merges:[] ~added_tokens ~cache_capacity:0 ()

let law_model = law_tokenizer []

type spec = {
  content : string;
  special : bool;
  single_word : bool;
  lstrip : bool;
  rstrip : bool;
  normalized : bool;
}

(* The class of the character [s.[start..stop)], if it is one. *)
let class_of s start stop =
  let d = String.get_utf_8_uchar s start in
  if
    Uchar.utf_decode_is_valid d && start + Uchar.utf_decode_length d = stop
  then
    let c = String.sub s start (stop - start) in
    let alphanumeric = function
      | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> true
      | _ -> false
    in
    if List.mem c word_chars || (String.length c = 1 && alphanumeric c.[0])
    then `Word
    else if List.mem c space_chars then `Space
    else `Other
  else `None

let is_continuation s i = Char.code s.[i] land 0xc0 = 0x80

let char_before s i =
  if i <= 0 then None
  else begin
    let start = ref (i - 1) in
    while !start > 0 && is_continuation s !start do
      decr start
    done;
    Some (class_of s !start i, !start)
  end

let char_after s i =
  if i >= String.length s then None
  else
    let d = String.get_utf_8_uchar s i in
    let n =
      if Uchar.utf_decode_is_valid d then Uchar.utf_decode_length d else 1
    in
    Some (class_of s i (i + n), i + n)

let rec space_before s i =
  match char_before s i with Some (`Space, at) -> space_before s at | _ -> i

let rec space_after s i =
  match char_after s i with Some (`Space, at) -> space_after s at | _ -> i

(* What the reference met on the last case, for the law's coverage. *)
let found = ref 0 and refused = ref 0 and stripped = ref 0

let reference_find specs ~match_special s ~pos =
  let len = String.length s in
  let occurs at spec =
    let n = String.length spec.content in
    at + n <= len && String.sub s at n = spec.content
  in
  let longest at =
    List.fold_left
      (fun best spec ->
        match best with
        | Some b
          when String.length b.content >= String.length spec.content ->
            best
        | _ -> if occurs at spec then Some spec else best)
      None specs
  in
  let rec scan at =
    if at >= len then None
    else
      match longest at with
      | None -> scan (at + 1)
      | Some spec ->
          let stop = at + String.length spec.content in
          let word = function Some (`Word, _) -> true | _ -> false in
          if
            (spec.special && not match_special)
            || spec.single_word
               && (word (char_before s at) || word (char_after s stop))
          then begin
            incr refused;
            scan stop
          end
          else
            let start =
              if spec.lstrip then max pos (space_before s at) else at
            in
            let stop' = if spec.rstrip then space_after s stop else stop in
            if start < at || stop' > stop then incr stripped;
            incr found;
            Some (start, stop', spec)
  in
  scan pos

(* Raw tokens cut the text, normalized ones each stretch left between them, and
   the model encodes what is left. *)
(* [norm] is the normalizer: normalized tokens are matched by their content
   normalized, in each stretch the raw ones leave, normalized. *)
let reference_ids t model specs ~norm ~match_special text =
  let raw = List.filter (fun spec -> not spec.normalized) specs in
  let normalized =
    List.filter_map
      (fun spec ->
        if spec.normalized then Some { spec with content = norm spec.content }
        else None)
      specs
  in
  let content_of pattern =
    (List.find
       (fun spec -> spec.normalized && norm spec.content = pattern.content)
       specs)
      .content
  in
  let ids = ref [] in
  let emit spec =
    let content = if spec.normalized then content_of spec else spec.content in
    ids := Option.get (token_to_id t content) :: !ids
  in
  let encode s =
    ids := List.rev_append (Array.to_list (encode_ids model s)) !ids
  in
  let rec split specs s ~gap pos =
    let len = String.length s in
    match reference_find specs ~match_special s ~pos with
    | None -> if pos < len then gap (String.sub s pos (len - pos))
    | Some (start, stop, spec) ->
        if pos < start then gap (String.sub s pos (start - pos));
        emit spec;
        split specs s ~gap stop
  in
  split raw text ~gap:(fun s -> split normalized (norm s) ~gap:encode 0) 0;
  Array.of_list (List.rev !ids)

let prefixes = [ ""; "<"; "\u{2581}<"; "a" ]
let suffixes =
  [ "s>"; "/s>"; "unk>"; "a"; "A"; "b"; "\u{e9}"; " "; "_"; "\u{2581}" ]

let spec_gen =
  let open Gen in
  let+ prefix =
    frequency
      (List.map (fun p -> ((if p = "\u{2581}<" then 3 else 1), constant p))
         prefixes)
  and+ suffix = of_list suffixes
  and+ special = bool
  and+ single_word = bool
  and+ lstrip = bool
  and+ rstrip = bool
  and+ normalized = bool in
  {
    content = prefix ^ suffix;
    special;
    single_word;
    lstrip;
    rstrip;
    normalized;
  }

(* Distinct contents, so that no two entries collapse into one token. *)
let distinct specs =
  List.fold_left
    (fun acc spec ->
      if List.exists (fun s -> s.content = spec.content) acc then acc
      else spec :: acc)
    [] specs
  |> List.rev

let law_case =
  let open Gen in
  let* specs = map distinct (list ~size:(int_range 1 6) spec_gen) in
  (* A twin in capitals collides with its token under a lowercasing normalizer
     when both are normalized. *)
  let* twin = frequency [ (3, constant false); (1, constant true) ] in
  let specs =
    match specs with
    | first :: _ when twin ->
        let content = String.uppercase_ascii first.content in
        distinct (specs @ [ { first with content } ])
    | _ -> specs
  in
  let pieces = List.map (fun spec -> spec.content) specs in
  let fragment =
    frequency
      [
        (6, of_list (law_chars @ stray_bytes));
        (2, of_list prefixes);
        (2, of_list pieces);
      ]
  in
  let+ text = map (String.concat "") (list ~size:(int_range 0 30) fragment)
  and+ match_special = bool
  and+ lowercase = bool in
  (specs, text, match_special, lowercase)

let pp_case ppf (specs, text, match_special, lowercase) =
  let pp_spec ppf s =
    Format.fprintf ppf
      "%S special=%b single_word=%b lstrip=%b rstrip=%b normalized=%b" s.content
      s.special s.single_word s.lstrip s.rstrip s.normalized
  in
  Format.fprintf ppf "@[<v>%a@,text %S match_special=%b lowercase=%b@]"
    (Format.pp_print_list pp_spec)
    specs text match_special lowercase

let law_case = Gen.with_pp pp_case law_case

let added_of spec =
  added_token ~special:spec.special ~single_word:spec.single_word
    ~lstrip:spec.lstrip ~rstrip:spec.rstrip ~normalized:spec.normalized
    spec.content

let law_scan =
  prop ~count:1000 "scan: every token tried at every position" law_case
    (fun (specs, text, match_special, lowercase) ->
      let normalizer = if lowercase then Some Normalizer.lowercase else None in
      let norm s =
        if lowercase then Normalizer.apply Normalizer.lowercase s else s
      in
      let normalized =
        List.filter_map
          (fun spec ->
            if spec.normalized then Some (norm spec.content) else None)
          specs
      in
      let collide =
        List.length (List.sort_uniq compare normalized)
        < List.length normalized
      in
      cover ~share:0.01 "normalized tokens that collide" collide;
      match law_tokenizer ?normalizer (List.map added_of specs) with
      | exception Invalid_argument _ when collide -> ()
      | t ->
          equal ~msg:"no two tokens match the same text" bool false collide;
          found := 0;
          refused := 0;
          stripped := 0;
          let expected =
            reference_ids t law_model specs ~norm ~match_special text
          in
          cover ~share:0.5 "a token found" (!found > 0);
          cover ~share:0.1 "a token refused" (!refused > 0);
          cover ~share:0.05 "white space taken in" (!stripped > 0);
          cover ~share:0.1 "contents sharing a multi-byte prefix"
            (List.length
               (List.filter
                  (fun spec ->
                    String.starts_with ~prefix:"\u{2581}<" spec.content)
                  specs)
            >= 2);
          equal ~msg:(Printf.sprintf "%S" text) (array int) expected
            (encode_ids t ~match_special_tokens:match_special text))

(* Hugging Face refuses two added tokens that match the same text
   (DuplicatePatternError), as two normalized ones whose contents normalize
   alike do. *)
let test_normalized_collision () =
  let make tokens =
    word_level ~normalizer:Normalizer.lowercase
      ~vocab:[ ("[UNK]", 0); ("a", 1) ]
      ~unk_token:"[UNK]" ~added_tokens:tokens ()
  in
  raises
    (Invalid_argument
       "added tokens \"A\" and \"a\" match the same text \"a\" once \
        normalized")
    (fun () ->
      make
        [ added_token ~special:false "A"; added_token ~special:false "a" ]);
  let t = make [ added_token ~special:false "A" ] in
  equal ~msg:"one is found" (array int) [| 2 |] (encode_ids t "a")

(* Hugging Face refuses an added token that normalizes to the empty text. *)
let test_normalized_to_nothing () =
  let make tokens =
    word_level ~normalizer:(Normalizer.strip ())
      ~vocab:[ ("[UNK]", 0); ("a", 1) ]
      ~unk_token:"[UNK]" ~added_tokens:tokens ()
  in
  raises
    (Invalid_argument "added token \"  \" normalizes to the empty text")
    (fun () -> make [ added_token ~special:false "  " ]);
  let t = make [ added_token "  " ] in
  equal ~msg:"matched raw" (array int) [| 1; 2; 1 |] (encode_ids t "a  a")

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
    test "special tokens unmatched" test_unmatched_special;
    test "json round trip" test_json_round_trip;
    test "scan offsets" test_scan_offsets;
    test "scan two firsts" test_scan_two_firsts;
    test "scan three firsts" test_scan_three_firsts;
    test "scan dense near" test_scan_dense_near;
    test "scan long stretch" test_scan_long_stretch;
    test "scan single word resume" test_scan_single_word_resume;
    test "scan long space run" test_scan_long_space_run;
    law_scan;
    test "normalized collision" test_normalized_collision;
    test "normalized to nothing" test_normalized_to_nothing;
  ]

let () = exit (run "Added tokens" [ group "added_tokens" suite ])
