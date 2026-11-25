(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Added tokens are matched atomically in the input. Expectations that name
   Hugging Face were read off tokenizers 0.23.1 first, from a BPE model with the
   same vocabulary and no merges, so that the model never absorbs an added
   token:

   t = Tokenizer(BPE(vocab={c: i for i, c in enumerate("abcdxz._-1 \t\nSWAB")},
   merges=[])) t.add_tokens([AddedToken("abc", normalized=False), ...])
   t.encode("xabc").ids # [4, 18]

   This suite runs twice: natively, where the scan for a token's first byte is
   C, and in bytecode, where it is OCaml. *)

open Windtrap
open Brot

(* One token per character, ids 0 to 16; added tokens are numbered from 17. *)
let alphabet = "abcdxz._-1 \t\nSWAB"

let model_vocab =
  List.init (String.length alphabet) (fun i -> (String.make 1 alphabet.[i], i))

let tokenizer ?normalizer tokens =
  bpe ?normalizer ~vocab:model_vocab ~merges:[] ~added_tokens:tokens ()

let ids_of t text = encode_ids t text

let check_encode t text ~ids ~tokens ~offsets =
  let encoding = encode t text in
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
    ~msg:(Printf.sprintf "%S encode_ids" text)
    (array int) ids (encode_ids t text)

let contents t = List.map (fun (a : added_token) -> a.content) (added_tokens t)

let pp_added_token ppf (a : added_token) =
  Format.fprintf ppf
    "{content %S; special %b; single_word %b; lstrip %b; rstrip %b; normalized \
     %b}"
    a.content a.special a.single_word a.lstrip a.rstrip a.normalized

let added_token_t = Testable.make ~pp:pp_added_token ~equal:( = )

(* added_token *)

let test_special_default () =
  equal added_token_t
    {
      content = "<s>";
      special = true;
      single_word = false;
      lstrip = false;
      rstrip = false;
      normalized = false;
    }
    (added_token "<s>")

let test_plain_default () =
  equal added_token_t
    {
      content = "<x>";
      special = false;
      single_word = false;
      lstrip = false;
      rstrip = false;
      normalized = true;
    }
    (added_token ~special:false "<x>")

(* The matching, against a reference that tries every token at every position.

   The text and the tokens are drawn from characters whose classes the reference
   knows: word characters, white space (Unicode's too), others, and bytes of no
   character, among them the first two bytes of the three of [▁]. Token contents
   often share a prefix, as SentencePiece's [▁<s>] and [▁</s>] do once
   normalized, so the scan has to tell them apart past it. *)

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

(* The class of the character [s.[start..stop)], if it is one. *)
let class_of s start stop =
  let d = String.get_utf_8_uchar s start in
  if Uchar.utf_decode_is_valid d && start + Uchar.utf_decode_length d = stop
  then
    let c = String.sub s start (stop - start) in
    let alphanumeric = function
      | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> true
      | _ -> false
    in
    if List.mem c word_chars || (String.length c = 1 && alphanumeric c.[0]) then
      `Word
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

(* What the reference met on a case. *)
type seen = {
  mutable found : int;
  mutable refused : int;
  mutable stripped : int;
}

(* The leftmost longest token from [pos], with the span its match takes. *)
let reference_find seen tokens ~match_special s ~pos =
  let len = String.length s in
  let occurs at (a : added_token) =
    let n = String.length a.content in
    at + n <= len && String.sub s at n = a.content
  in
  let longest at =
    List.fold_left
      (fun best (a : added_token) ->
        match best with
        | Some (b : added_token)
          when String.length b.content >= String.length a.content ->
            best
        | _ -> if occurs at a then Some a else best)
      None tokens
  in
  let word = function Some (`Word, _) -> true | _ -> false in
  let rec scan at =
    if at >= len then None
    else
      match longest at with
      | None -> scan (at + 1)
      | Some a ->
          let stop = at + String.length a.content in
          if
            (a.special && not match_special)
            || a.single_word
               && (word (char_before s at) || word (char_after s stop))
          then begin
            seen.refused <- seen.refused + 1;
            scan stop
          end
          else
            let start = if a.lstrip then max pos (space_before s at) else at in
            let stop' = if a.rstrip then space_after s stop else stop in
            if start < at || stop' > stop then
              seen.stripped <- seen.stripped + 1;
            seen.found <- seen.found + 1;
            Some (start, stop', a)
  in
  scan pos

let law_model = lazy (law_tokenizer [])

(* Raw tokens cut the text, normalized ones each stretch left between them,
   normalized by [norm], and the model encodes what is left. *)
let reference_ids seen t tokens ~norm ~match_special text =
  let model = Lazy.force law_model in
  let raw = List.filter (fun (a : added_token) -> not a.normalized) tokens in
  let normalized =
    List.filter_map
      (fun (a : added_token) ->
        if a.normalized then Some { a with content = norm a.content } else None)
      tokens
  in
  let content_of (found : added_token) =
    (List.find
       (fun (a : added_token) -> a.normalized && norm a.content = found.content)
       tokens)
      .content
  in
  let ids = ref [] in
  let emit (a : added_token) =
    let content = if a.normalized then content_of a else a.content in
    ids := Option.get (token_to_id t content) :: !ids
  in
  let encode s =
    ids := List.rev_append (Array.to_list (encode_ids model s)) !ids
  in
  let rec split tokens s ~gap pos =
    let len = String.length s in
    match reference_find seen tokens ~match_special s ~pos with
    | None -> if pos < len then gap (String.sub s pos (len - pos))
    | Some (start, stop, a) ->
        if pos < start then gap (String.sub s pos (start - pos));
        emit a;
        split tokens s ~gap stop
  in
  split raw text ~gap:(fun s -> split normalized (norm s) ~gap:encode 0) 0;
  Array.of_list (List.rev !ids)

let prefixes = [ ""; "<"; "\u{2581}<"; "a" ]

let suffixes =
  [ "s>"; "/s>"; "unk>"; "a"; "A"; "b"; "\u{e9}"; " "; "_"; "\u{2581}" ]

let token_gen =
  let open Gen in
  let+ prefix =
    frequency
      (List.map
         (fun p -> ((if p = "\u{2581}<" then 3 else 1), constant p))
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
let distinct tokens =
  List.fold_left
    (fun acc (a : added_token) ->
      if List.exists (fun (b : added_token) -> b.content = a.content) acc then
        acc
      else a :: acc)
    [] tokens
  |> List.rev

let law_case =
  let open Gen in
  let* tokens = map distinct (list ~size:(int_range 1 6) token_gen) in
  (* A twin in capitals collides with its token under a lowercasing normalizer
     when both are normalized, so a drawn twin comes with both. *)
  let* twin = frequency [ (3, constant false); (1, constant true) ] in
  let tokens =
    match tokens with
    | first :: rest when twin ->
        let first = { first with normalized = true } in
        distinct
          ((first :: rest)
          @ [ { first with content = String.uppercase_ascii first.content } ])
    | _ -> tokens
  in
  let fragment =
    frequency
      [
        (6, of_list (law_chars @ stray_bytes));
        (2, of_list prefixes);
        (2, of_list (List.map (fun (a : added_token) -> a.content) tokens));
      ]
  in
  let+ text = map (String.concat "") (list ~size:(int_range 0 30) fragment)
  and+ match_special = bool
  and+ lowercase = bool in
  (tokens, text, match_special, lowercase || twin)

let pp_case ppf (tokens, text, match_special, lowercase) =
  Format.fprintf ppf "@[<v>%a@,text %S match_special=%b lowercase=%b@]"
    (Format.pp_print_list pp_added_token)
    tokens text match_special lowercase

let matching_law =
  prop ~count:200 "is matched as a search trying every token at every position"
    (Gen.with_pp pp_case law_case)
    (fun (tokens, text, match_special, lowercase) ->
      let normalizer = if lowercase then Some Normalizer.lowercase else None in
      let norm s =
        if lowercase then Normalizer.apply Normalizer.lowercase s else s
      in
      let normalized =
        List.filter_map
          (fun (a : added_token) ->
            if a.normalized then Some (norm a.content) else None)
          tokens
      in
      let collide =
        List.length (List.sort_uniq compare normalized) < List.length normalized
      in
      cover "normalized tokens that collide" collide;
      if collide then
        raises_match ~msg:"two tokens match one text" Exn.invalid_arg (fun () ->
            law_tokenizer ?normalizer tokens)
      else begin
        let t = law_tokenizer ?normalizer tokens in
        let seen = { found = 0; refused = 0; stripped = 0 } in
        let expected = reference_ids seen t tokens ~norm ~match_special text in
        cover "a token found" (seen.found > 0);
        cover "a token refused" (seen.refused > 0);
        cover "white space taken in" (seen.stripped > 0);
        cover "contents sharing a multi-byte prefix"
          (List.length
             (List.filter
                (fun (a : added_token) ->
                  String.starts_with ~prefix:"\u{2581}<" a.content)
                tokens)
          >= 2);
        equal ~msg:(Printf.sprintf "%S" text) (array int) expected
          (encode_ids t ~match_special_tokens:match_special text)
      end)

let test_leftmost_longest () =
  let t = tokenizer [ added_token "ab"; added_token "abc"; added_token "bc" ] in
  equal ~msg:"ab" (option int) (Some 17) (token_to_id t "ab");
  equal ~msg:"abc" (option int) (Some 18) (token_to_id t "abc");
  equal ~msg:"bc" (option int) (Some 19) (token_to_id t "bc");
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

(* The scan skips to the next byte that can start a token: a single anchor byte
   is searched by the fastest loop the build has, one or two by a search of
   eight bytes at a time, more bytewise. A match or a lone anchor byte is found
   wherever it lies, up to the last byte, on each side of the widths those loops
   read at a time. *)

let scan_vocab =
  let chars = "abcs<>[]{}d" in
  List.init (String.length chars) (fun i -> (String.make 1 chars.[i], i))

let scan_tokenizer tokens =
  bpe ~vocab:scan_vocab ~merges:[] ~added_tokens:tokens ()

let test_every_offset () =
  let t = scan_tokenizer [ added_token "<s>" ] in
  let a n = Array.make n 0 in
  for o = 0 to 20 do
    let text = String.make o 'a' ^ "<s>" ^ String.make (20 - o) 'b' in
    equal
      ~msg:(Printf.sprintf "a match at %d" o)
      (array int)
      (Array.concat [ a o; [| 11 |]; Array.make (20 - o) 1 ])
      (ids_of t text);
    equal
      ~msg:(Printf.sprintf "its offsets at %d" o)
      (pair int int)
      (o, o + 3)
      (Encoding.offsets (encode t text)).(o)
  done;
  for n = 0 to 70 do
    let run = String.make n 'a' in
    equal
      ~msg:(Printf.sprintf "a match after %d" n)
      (array int)
      (Array.append (a n) [| 11 |])
      (ids_of t (run ^ "<s>"));
    equal
      ~msg:(Printf.sprintf "a lone anchor after %d" n)
      (array int)
      (Array.append (a n) [| 5 |])
      (ids_of t (run ^ ">"))
  done;
  let run = String.make 100 'a' in
  equal ~msg:"two matches far apart" (array int)
    (Array.concat [ a 100; [| 11 |]; a 100; [| 11 |] ])
    (ids_of t (run ^ "<s>" ^ run ^ "<s>"));
  equal ~msg:"a match first" (array int)
    (Array.append [| 11 |] (a 100))
    (ids_of t ("<s>" ^ run));
  equal ~msg:"no match" (array int) (a 200) (ids_of t (run ^ run))

let test_first_bytes () =
  let two = scan_tokenizer [ added_token "<s>"; added_token "[c]" ] in
  equal ~msg:"two first bytes" (array int)
    [| 0; 6; 2; 0; 4; 3; 0; 12; 0; 11; 0 |]
    (ids_of two "a[ca<sa[c]a<s>a");
  equal ~msg:"near matches around a match" (array int)
    [| 4; 4; 3; 12; 4; 5; 3 |] (ids_of two "<<s[c]<>s");
  (* The [ sits in a full eight-byte window holding no <: a scan seeking only
     the first first byte would skip it. *)
  equal ~msg:"the second first byte alone in a window" (array int)
    (Array.concat [ Array.make 10 0; [| 12 |]; Array.make 10 0 ])
    (ids_of two (String.make 10 'a' ^ "[c]" ^ String.make 10 'a'));
  let three =
    scan_tokenizer [ added_token "<s>"; added_token "[c]"; added_token "{d}" ]
  in
  equal ~msg:"three first bytes" (array int)
    [| 0; 13; 1; 11; 2; 12; 10 |]
    (ids_of three "a{d}b<s>c[c]d");
  let one = scan_tokenizer [ added_token "<s>" ] in
  equal ~msg:"near matches all along" (array int) [| 4; 4; 3; 11; 3; 5 |]
    (ids_of one "<<s<s>s>")

let test_decode_skips_special () =
  let special = tokenizer [ added_token "<s>" ] in
  equal ~msg:"special, kept" string "a <s> b"
    (decode special (ids_of special "a<s>b"));
  equal ~msg:"special, skipped" string "a b"
    (decode special ~skip_special_tokens:true (ids_of special "a<s>b"));
  let plain = tokenizer [ added_token ~special:false "<s>" ] in
  equal ~msg:"plain, kept" string "a <s> b"
    (decode plain (ids_of plain "a<s>b"));
  equal ~msg:"plain, not skipped" string "a <s> b"
    (decode plain ~skip_special_tokens:true (ids_of plain "a<s>b"))

(* A word character is a letter, a mark, a decimal digit, connector punctuation
   or a joiner. *)
let test_single_word () =
  let t = tokenizer [ added_token ~single_word:true "SW" ] in
  equal ~msg:"SW" (array int) [| 17 |] (ids_of t "SW");
  equal ~msg:".SW." (array int) [| 6; 17; 6 |] (ids_of t ".SW.");
  equal ~msg:"-SW-" (array int) [| 8; 17; 8 |] (ids_of t "-SW-");
  equal ~msg:"x SW x" (array int) [| 4; 10; 17; 10; 4 |] (ids_of t "x SW x");
  equal ~msg:"aSW" (array int) [| 0; 13; 14 |] (ids_of t "aSW");
  equal ~msg:"SWa" (array int) [| 13; 14; 0 |] (ids_of t "SWa");
  equal ~msg:"1SW" (array int) [| 9; 13; 14 |] (ids_of t "1SW");
  equal ~msg:"_SW_" (array int) [| 7; 13; 14; 7 |] (ids_of t "_SW_");
  equal ~msg:"\u{216B}SW, a letter number before" (array int) [| 13; 14 |]
    (ids_of t "\u{216B}SW");
  equal ~msg:"\u{b2}SW, another number before" (array int) [| 17 |]
    (ids_of t "\u{b2}SW");
  equal ~msg:"refused, then matched further on" (array int)
    (Array.concat
       [ Array.make 10 0; [| 13; 14 |]; Array.make 10 0; [| 6; 17; 6 |] ])
    (ids_of t "aaaaaaaaaaSWaaaaaaaaaa.SW.")

(* The search resumes past the refused match. HF: "abc" single_word and "bc"
   give "abcd" -> a, b, c, d. *)
let test_single_word_refused () =
  let t = tokenizer [ added_token ~single_word:true "abc"; added_token "ab" ] in
  equal ~msg:"abc" (option int) (Some 17) (token_to_id t "abc");
  equal ~msg:"ab" (option int) (Some 18) (token_to_id t "ab");
  equal ~msg:"abcd: none at its position" (array int) [| 0; 1; 2; 3 |]
    (ids_of t "abcd");
  equal ~msg:"abc." (array int) [| 17; 6 |] (ids_of t "abc.");
  equal ~msg:"ab." (array int) [| 18; 6 |] (ids_of t "ab.");
  let inner =
    tokenizer [ added_token ~single_word:true "abc"; added_token "bc" ]
  in
  equal ~msg:"abcd: none inside it" (array int) [| 0; 1; 2; 3 |]
    (ids_of inner "abcd");
  equal ~msg:"xbc." (array int) [| 4; 18; 6 |] (ids_of inner "xbc.")

(* The stripped white space is part of the match, and of the token it emits. *)
let test_lstrip () =
  let t = tokenizer [ added_token ~lstrip:true "L" ] in
  check_encode t "a  L b" ~ids:[| 0; 17; 10; 1 |]
    ~tokens:[| "a"; "  L"; " "; "b" |]
    ~offsets:[| (0, 1); (1, 4); (4, 5); (5, 6) |];
  check_encode t "L b" ~ids:[| 17; 10; 1 |] ~tokens:[| "L"; " "; "b" |]
    ~offsets:[| (0, 1); (1, 2); (2, 3) |];
  check_encode t "aL" ~ids:[| 0; 17 |] ~tokens:[| "a"; "L" |]
    ~offsets:[| (0, 1); (1, 2) |];
  check_encode t
    ("a" ^ String.make 12 ' ' ^ "L")
    ~ids:[| 0; 17 |]
    ~tokens:[| "a"; String.make 12 ' ' ^ "L" |]
    ~offsets:[| (0, 1); (1, 14) |]

let test_rstrip () =
  let t = tokenizer [ added_token ~rstrip:true "R" ] in
  check_encode t "a R  b" ~ids:[| 0; 10; 17; 1 |]
    ~tokens:[| "a"; " "; "R  "; "b" |]
    ~offsets:[| (0, 1); (1, 2); (2, 5); (5, 6) |];
  check_encode t "a R" ~ids:[| 0; 10; 17 |] ~tokens:[| "a"; " "; "R" |]
    ~offsets:[| (0, 1); (1, 2); (2, 3) |]

let test_lstrip_after_rstrip () =
  let t =
    tokenizer [ added_token ~lstrip:true "L"; added_token ~rstrip:true "R" ]
  in
  check_encode t "R  L" ~ids:[| 18; 17 |] ~tokens:[| "R  "; "L" |]
    ~offsets:[| (0, 3); (3, 4) |]

let test_normalized () =
  let normalizer = Normalizer.lowercase in
  let raw = tokenizer ~normalizer [ added_token "AB" ] in
  check_encode raw "xABx" ~ids:[| 4; 17; 4 |] ~tokens:[| "x"; "AB"; "x" |]
    ~offsets:[| (0, 1); (1, 3); (3, 4) |];
  equal ~msg:"raw: xabx" (array int) [| 4; 0; 1; 4 |] (ids_of raw "xabx");
  let t = tokenizer ~normalizer [ added_token ~normalized:true "AB" ] in
  equal ~msg:"its content is the token" (option int) (Some 17)
    (token_to_id t "AB");
  equal ~msg:"its normalized form is not" (option int) None (token_to_id t "ab");
  check_encode t "xABx" ~ids:[| 4; 17; 4 |] ~tokens:[| "x"; "ab"; "x" |]
    ~offsets:[| (0, 1); (1, 3); (3, 4) |];
  check_encode t "xabx" ~ids:[| 4; 17; 4 |] ~tokens:[| "x"; "ab"; "x" |]
    ~offsets:[| (0, 1); (1, 3); (3, 4) |]

(* Tokens matched against the raw input are cut out first, so a normalizer
   cannot hide them. *)
let test_raw_first () =
  let t =
    tokenizer ~normalizer:Normalizer.lowercase
      [ added_token "AB"; added_token ~normalized:true "abc" ]
  in
  equal ~msg:"AB" (option int) (Some 17) (token_to_id t "AB");
  equal ~msg:"abc" (option int) (Some 18) (token_to_id t "abc");
  equal ~msg:"ABc" (array int) [| 17; 2 |] (ids_of t "ABc");
  equal ~msg:"abc" (array int) [| 18 |] (ids_of t "abc")

let test_model_id () =
  let t = tokenizer [ added_token "a"; added_token "<s>" ] in
  equal ~msg:"a" (option int) (Some 0) (token_to_id t "a");
  equal ~msg:"<s>" (option int) (Some 17) (token_to_id t "<s>");
  equal ~msg:"vocab_size" int 18 (vocab_size t);
  let sparse =
    word_level
      ~vocab:[ ("a", 0); ("b", 1); ("<s>", 4) ]
      ~added_tokens:[ added_token "<s>"; added_token "<z>" ]
      ()
  in
  equal ~msg:"a sparse vocabulary" (option int) (Some 4)
    (token_to_id sparse "<s>");
  equal ~msg:"the next one past it" (option int) (Some 5)
    (token_to_id sparse "<z>")

let test_numbered_after () =
  let t = tokenizer [ added_token "<s>"; added_token "<pad>" ] in
  equal ~msg:"<s>" (option int) (Some 17) (token_to_id t "<s>");
  equal ~msg:"<pad>" (option int) (Some 18) (token_to_id t "<pad>");
  equal ~msg:"id 17" (option string) (Some "<s>") (id_to_token t 17);
  equal ~msg:"id 18" (option string) (Some "<pad>") (id_to_token t 18);
  equal ~msg:"vocab_size" int 19 (vocab_size t)

(* [b] keeps its model id, so [<x>] would be numbered 2^31. *)
let test_past_int32 () =
  raises_match (Exn.invalid_arg ~sub:"2^31") (fun () ->
      bpe
        ~vocab:[ ("a", 0); ("b", (1 lsl 31) - 1) ]
        ~added_tokens:[ added_token "b"; added_token "<x>" ]
        ())

(* bpe *)

let test_empty_content () =
  let t = tokenizer [ added_token ""; added_token "<s>" ] in
  equal ~msg:"tokens" (list string) [ "<s>" ] (contents t);
  equal ~msg:"no id taken" (option int) (Some 17) (token_to_id t "<s>");
  equal ~msg:"vocab_size" int 18 (vocab_size t)

let test_given_twice () =
  let t = tokenizer [ added_token "<s>"; added_token ~lstrip:true "<s>" ] in
  equal ~msg:"tokens" (list string) [ "<s>" ] (contents t);
  equal ~msg:"the first id" (option int) (Some 17) (token_to_id t "<s>");
  equal ~msg:"one id taken" int 18 (vocab_size t);
  equal ~msg:"the last flags" (array string) [| "a"; " <s>" |]
    (Encoding.tokens (encode t "a <s>"))

(* HF refuses two added tokens that match the same text (DuplicatePatternError),
   as two normalized ones whose contents normalize alike do. *)
let test_same_text () =
  let make tokens =
    word_level ~normalizer:Normalizer.lowercase
      ~vocab:[ ("[UNK]", 0); ("a", 1) ]
      ~unk_token:"[UNK]" ~added_tokens:tokens ()
  in
  raises_match ~msg:"A and a" Exn.invalid_arg (fun () ->
      make [ added_token ~special:false "A"; added_token ~special:false "a" ]);
  equal ~msg:"A alone" (array int) [| 2 |]
    (encode_ids (make [ added_token ~special:false "A" ]) "a")

(* HF refuses an added token that normalizes to the empty text. *)
let test_normalized_to_nothing () =
  let make tokens =
    word_level ~normalizer:(Normalizer.strip ())
      ~vocab:[ ("[UNK]", 0); ("a", 1) ]
      ~unk_token:"[UNK]" ~added_tokens:tokens ()
  in
  raises_match ~msg:"normalized" Exn.invalid_arg (fun () ->
      make [ added_token ~special:false "  " ]);
  equal ~msg:"matched raw" (array int) [| 1; 2; 1 |]
    (encode_ids (make [ added_token "  " ]) "a  a")

let test_role_markers () =
  let held =
    bpe
      ~vocab:[ ("a", 0); ("b", 1); ("[PAD]", 2) ]
      ~merges:[] ~pad_token:"[PAD]" ()
  in
  equal ~msg:"held: registered" (list string) [ "[PAD]" ] (contents held);
  equal ~msg:"held: its model id" (array int) [| 0; 2; 1 |]
    (ids_of held "a[PAD]b");
  equal ~msg:"held: skipped" string "a b"
    (decode held ~skip_special_tokens:true (ids_of held "a[PAD]b"));
  let absent =
    word_level
      ~vocab:[ ("a", 0) ]
      ~bos_token:"<s>" ~eos_token:"</s>" ~pad_token:"[PAD]" ()
  in
  equal ~msg:"absent: registered" (list string) [ "<s>"; "</s>"; "[PAD]" ]
    (contents absent);
  equal ~msg:"absent: numbered after the vocabulary"
    (list (option int))
    [ Some 1; Some 2; Some 3 ]
    (List.map (token_to_id absent) [ "<s>"; "</s>"; "[PAD]" ]);
  equal ~msg:"absent: matched" (array int) [| 1; 0; 3; 2 |]
    (encode_ids absent "<s>a [PAD]</s>");
  equal ~msg:"absent: skipped" string "a"
    (decode absent ~skip_special_tokens:true [| 1; 0; 3; 2 |])

let test_role_flags () =
  let t =
    bpe
      ~vocab:[ ("a", 0) ]
      ~merges:[]
      ~added_tokens:[ added_token ~special:false "<s>" ]
      ~bos_token:"<s>" ()
  in
  equal ~msg:"registered once" (list string) [ "<s>" ] (contents t);
  equal ~msg:"not skipped" string "<s>"
    (decode t ~skip_special_tokens:true [| 1 |])

(* HF: BPE(vocab, merges=[], unk_token="<unk>") encodes "a<unk>a" as a, <, u, n,
   k, >, a and has no added token. *)
let test_unk_token_unmatched () =
  let t =
    bpe
      ~vocab:(List.mapi (fun i c -> (c, i)) [ "a"; "<"; "u"; "n"; "k"; ">" ])
      ~merges:[] ~unk_token:"<unk>" ()
  in
  equal ~msg:"no added token" (list string) [] (contents t);
  equal ~msg:"no id" (option int) None (token_to_id t "<unk>");
  equal ~msg:"vocab_size" int 6 (vocab_size t);
  equal ~msg:"read as text" (array int) [| 0; 1; 2; 3; 4; 5; 0 |]
    (ids_of t "a<unk>a")

(* add_tokens *)

(* Each model over the law's characters, with its unknown token. *)
let models =
  let vocab = List.mapi (fun i c -> (c, i)) ("[UNK]" :: "<unk>" :: law_chars) in
  [
    ("bpe", fun added_tokens -> bpe ~vocab ~merges:[] ~added_tokens ());
    ("wordpiece", fun added_tokens -> wordpiece ~vocab ~added_tokens ());
    ("word_level", fun added_tokens -> word_level ~vocab ~added_tokens ());
    ( "unigram",
      fun added_tokens ->
        unigram
          ~vocab:(List.map (fun (c, _) -> (c, -1.)) vocab)
          ~unk_id:0 ~added_tokens () );
    ("chars", fun added_tokens -> chars ~added_tokens ());
  ]

let add_case =
  let open Gen in
  let+ model = of_list (List.map fst models)
  and+ first = list ~size:(int_range 0 3) token_gen
  and+ later =
    list ~size:(int_range 1 3)
      (frequency [ (6, token_gen); (1, constant (added_token "")) ])
  and+ text =
    map (String.concat "")
      (list ~size:(int_range 0 20)
         (frequency
            [
              (3, of_list law_chars);
              (1, of_list prefixes);
              (1, of_list suffixes);
            ]))
  in
  (model, first, later, text)

let pp_add_case ppf (model, first, later, text) =
  Format.fprintf ppf "@[<v>%s@,first %a@,later %a@,text %S@]" model
    (Format.pp_print_list pp_added_token)
    first
    (Format.pp_print_list pp_added_token)
    later text

let construction_law =
  prop ~count:80 "gives what passing the tokens at construction gives"
    (Gen.with_pp pp_add_case add_case) (fun (model, first, later, text) ->
      let make = List.assoc model models in
      let added = add_tokens (make first) later in
      let built = make (first @ later) in
      cover "a token of empty content"
        (List.exists (fun (a : added_token) -> a.content = "") later);
      cover "a token given in both"
        (List.exists
           (fun (a : added_token) ->
             List.exists (fun (b : added_token) -> a.content = b.content) first)
           later);
      equal ~msg:"added_tokens" (list added_token_t) (added_tokens built)
        (added_tokens added);
      equal ~msg:"vocab"
        (list (pair string int))
        (List.sort compare (vocab built))
        (List.sort compare (vocab added));
      let ids = encode_ids built text in
      equal ~msg:"encode_ids" (array int) ids (encode_ids added text);
      equal ~msg:"decode" string
        (decode built ~skip_special_tokens:true ids)
        (decode added ~skip_special_tokens:true ids))

let test_two_calls () =
  let base = word_level ~vocab:[ ("hello", 0); ("world", 1) ] () in
  let twice =
    add_tokens (add_tokens base [ added_token "<pad>" ]) [ added_token "<x>" ]
  in
  equal ~msg:"<pad>" (option int) (Some 2) (token_to_id twice "<pad>");
  equal ~msg:"<x>" (option int) (Some 3) (token_to_id twice "<x>");
  equal ~msg:"vocab_size" int 4 (vocab_size twice);
  equal ~msg:"tokens" (list string) [ "<pad>"; "<x>" ] (contents twice)

let test_flags_replaced () =
  let t = tokenizer [ added_token "<s>"; added_token "<p>" ] in
  let t = add_tokens t [ added_token ~special:false ~lstrip:true "<s>" ] in
  equal ~msg:"its id" (option int) (Some 17) (token_to_id t "<s>");
  equal ~msg:"its flags" (list added_token_t)
    [ added_token ~special:false ~lstrip:true "<s>"; added_token "<p>" ]
    (added_tokens t);
  equal ~msg:"not skipped" string "a <s>"
    (decode t ~skip_special_tokens:true (ids_of t "a <s>"))

(* [b] keeps its model id, so [<x>] would be numbered 2^31. *)
let past_int32 =
  cases "raises Invalid_argument past 2^31 - 1" ~id:fst
    [
      ("wordpiece", fun vocab -> wordpiece ~vocab ());
      ("word_level", fun vocab -> word_level ~vocab ());
    ]
    (fun (_, make) ->
      let t = make [ ("[UNK]", 0); ("b", (1 lsl 31) - 1) ] in
      raises_match (Exn.invalid_arg ~sub:"2^31") (fun () ->
          add_tokens t (List.map added_token [ "b"; "<x>" ])))

let () =
  exit
    (run "added_token"
       [
         group "added_token"
           [
             test "is special and matched raw by default" test_special_default;
             test "is matched normalized when not special" test_plain_default;
             matching_law;
             test
               "takes the longest token at a position, then the earliest \
                position"
               test_leftmost_longest;
             test "is cut out of the text with its own offsets" test_placement;
             test "is found at every offset up to the last byte"
               test_every_offset;
             test "is found whichever of its first bytes starts it"
               test_first_bytes;
             test "is skipped by decode only when special"
               test_decode_skips_special;
             test
               "with single_word matches only where no word character touches \
                it"
               test_single_word;
             test "with single_word refused tries no token inside the match"
               test_single_word_refused;
             test "with lstrip takes in the white space before it" test_lstrip;
             test "with rstrip takes in the white space after it" test_rstrip;
             test "with lstrip leaves the white space an rstrip took"
               test_lstrip_after_rstrip;
             test "with normalized matches the normalized text and emits it"
               test_normalized;
             test "without normalized is cut out before normalization"
               test_raw_first;
             test "keeps the id of the model's entry" test_model_id;
             test "is numbered from the end of the vocabulary"
               test_numbered_after;
             test "raises Invalid_argument past 2^31 - 1" test_past_int32;
           ];
         group "bpe"
           [
             test "drops an added token of empty content" test_empty_content;
             test
               "takes a content given twice as one token, the first id and the \
                last flags"
               test_given_twice;
             test "refuses two added tokens that match the same text"
               test_same_text;
             test "refuses a normalized token that normalizes to nothing"
               test_normalized_to_nothing;
             test
               "registers bos_token, eos_token and pad_token as special tokens"
               test_role_markers;
             test "keeps the flags added_tokens gives a role marker"
               test_role_flags;
             test "never matches unk_token in the input"
               test_unk_token_unmatched;
           ];
         group "add_tokens"
           [
             construction_law;
             test "numbers the tokens of two calls densely" test_two_calls;
             test "replaces the flags of a token held, keeping its id"
               test_flags_replaced;
             past_int32;
           ];
       ])
