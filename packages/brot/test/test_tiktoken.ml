(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Brot

(* Rank files *)

let base64 s =
  let alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
  in
  let len = String.length s in
  let out = Bytes.make ((len + 2) / 3 * 4) '=' in
  for g = 0 to ((len + 2) / 3) - 1 do
    let byte k = if (3 * g) + k < len then Char.code s.[(3 * g) + k] else 0 in
    let w = (byte 0 lsl 16) lor (byte 1 lsl 8) lor byte 2 in
    for k = 0 to 3 do
      if k <= 1 || (3 * g) + k - 1 < len then
        Bytes.set out ((4 * g) + k) alphabet.[(w lsr (18 - (6 * k))) land 63]
    done
  done;
  Bytes.to_string out

(* A rank file holding [ranks], one line each, the empty token as [=]. *)
let rank_file ranks =
  String.concat ""
    (List.map
       (fun (b, r) ->
         Printf.sprintf "%s %d\n" (if b = "" then "=" else base64 b) r)
       ranks)

let single_bytes = List.init 256 (fun b -> (String.make 1 (Char.chr b), b))
let whole = {|[\s\S]+|}

let load ?specials ?(pattern = whole) ranks =
  match of_tiktoken ?specials ~pattern (rank_file ranks) with
  | Ok t -> t
  | Error e -> failf "of_tiktoken: %s" e

let ids t text = Array.to_list (encode_ids t text)

(* The reference rule, tiktoken's, on one pretoken: a token is itself, and any
   other pretoken starts as its bytes and joins the adjacent pair of lowest
   joined rank, the leftmost of equal ones, until no pair joins. *)
let merge ranks piece =
  let rank s = List.assoc_opt s ranks in
  let rec go parts =
    let rec best i = function
      | a :: (b :: _ as rest) -> (
          let here = Option.map (fun r -> (r, i)) (rank (a ^ b)) in
          match (here, best (i + 1) rest) with
          | Some (r, i), Some (r', _) when r <= r' -> Some (r, i)
          | Some h, None -> Some h
          | _, later -> later)
      | _ -> None
    in
    match best 0 parts with
    | None -> parts
    | Some (_, i) ->
        go
          (List.concat
             (List.mapi
                (fun j p ->
                  if j = i then [ p ^ List.nth parts (i + 1) ]
                  else if j = i + 1 then []
                  else [ p ])
                parts))
  in
  go (List.init (String.length piece) (fun i -> String.make 1 piece.[i]))

let reference ranks piece =
  if piece = "" then []
  else
    match List.assoc_opt piece ranks with
    | Some r -> [ r ]
    | None -> List.map (fun p -> List.assoc p ranks) (merge ranks piece)

(* Hand tables *)

let abc =
  List.filter (fun (b, _) -> b <> "a" && b <> "b" && b <> "c") single_bytes

(* [a b c bc ab abc]: on [abca] the rule joins [b c], then [a bc], which no
   merge list of [a b] and [ab c] holds. *)
let test_unlisted_join () =
  let t =
    load
      (abc
      @ [
          ("a", 1000);
          ("b", 1001);
          ("c", 1002);
          ("bc", 1003);
          ("ab", 1004);
          ("abc", 1005);
        ])
  in
  equal ~msg:"abca" (list int) [ 1005; 1000 ] (ids t "abca");
  equal ~msg:"abc" (list int) [ 1005 ] (ids t "abc")

(* [abc] is a token no join builds: it is emitted whole, and spelled out inside
   a longer pretoken. *)
let test_unbuilt () =
  let t =
    load (abc @ [ ("a", 1000); ("b", 1001); ("c", 1002); ("abc", 1003) ])
  in
  equal ~msg:"abc" (list int) [ 1003 ] (ids t "abc");
  equal ~msg:"abca" (list int) [ 1000; 1001; 1002; 1000 ] (ids t "abca")

(* A gap in the ranks, the empty token and rank file spellings tiktoken
   reads. *)
let test_file_forms () =
  let ranks = single_bytes @ [ ("ab", 300); ("", 5000) ] in
  let t = load ranks in
  equal ~msg:"ab" (list int) [ 300 ] (ids t "ab");
  equal ~msg:"vocab_size" int 258 (vocab_size t);
  equal ~msg:"the empty token" (option string) (Some "") (id_to_token t 5000);
  equal ~msg:"decode the empty token" string "" (decode t [| 5000 |]);
  let spelled =
    String.concat ""
      (List.map
         (fun (b, r) ->
           Printf.sprintf "\t%s \x0b %d\r\n\n"
             (if b = "" then "=" else base64 b)
             r)
         ranks)
  in
  match of_tiktoken ~pattern:whole spelled with
  | Error e -> failf "CRLF and white space: %s" e
  | Ok t' ->
      equal ~msg:"CRLF and white space" (list int) (ids t "ab xyz ab")
        (ids t' "ab xyz ab")

(* A rank reads as Python's [int] reads it, leading zeros included. *)
let test_leading_zeros () =
  let text =
    rank_file (single_bytes @ [ ("ab", 256) ]) ^ "YWJj 00000000000300\n"
  in
  match of_tiktoken ~pattern:whole text with
  | Error e -> failf "of_tiktoken: %s" e
  | Ok t -> equal ~msg:"abc" (list int) [ 300 ] (ids t "abc")

(* With every token built, a pretoken that is a special's text is spelled out by
   the merges, as tiktoken's [encode_ordinary] spells it, and not looked up
   whole. *)
let test_special_spelled () =
  let t = load ~specials:[ ("<|x|>", 300) ] (single_bytes @ [ ("ab", 256) ]) in
  equal ~msg:"encode_ordinary" (list int) [ 60; 124; 120; 124; 62 ]
    (Array.to_list (encode_ids t ~match_special_tokens:false "<|x|>"));
  equal ~msg:"encode" (list int) [ 300 ] (ids t "<|x|>")

(* Text no match covers is dropped, a byte that is no character included. *)
let test_uncovered () =
  let t = load ~pattern:{|[a-z]+|} (single_bytes @ [ ("ab", 256) ]) in
  equal ~msg:"spaces and digits" (list int) [ 256; 99 ] (ids t " ab 12c ");
  let t = load single_bytes in
  equal ~msg:"a stray byte" (list int) [ 97; 98 ] (ids t "a\xffb")

(* Of two specials where one is a prefix of the other, the longer is matched. *)
let test_longer_special () =
  let t = load ~specials:[ ("<|x", 300); ("<|x|>", 301) ] single_bytes in
  equal ~msg:"<|x|>" (list int) [ 301 ] (ids t "<|x|>");
  equal ~msg:"<|x" (list int) [ 300; 124 ] (ids t "<|x|")

(* Refusals *)

let base = single_bytes @ [ ("ab", 256) ]

let errors =
  let file = rank_file base in
  let drop_ff = rank_file (List.filter (fun (_, r) -> r <> 255) base) in
  [
    ("a line with one field", file ^ "QQ==\n", []);
    ("a line with three fields", file ^ "YWJj 300 1\n", []);
    ("a line of white space", file ^ " \n", []);
    ("base64 without its padding", file ^ "YWJjZA 300\n", []);
    ("a character outside base64", file ^ "YW@= 300\n", []);
    ("bits left over in the last digit", file ^ "YWJ= 300\n", []);
    ("an equals sign after data", file ^ "=YWJ 300\n", []);
    ("a negative rank", file ^ "YWJj -1\n", []);
    ("a signed rank", file ^ "YWJj +300\n", []);
    ("a rank of 2^31", file ^ "YWJj 2147483648\n", []);
    ("a rank of u32 max", file ^ "YWJj 4294967295\n", []);
    ("two lines with the same bytes", file ^ "YWI= 300\n", []);
    ("two lines with the same rank", file ^ "YWJj 256\n", []);
    ("a byte without a rank", drop_ff, []);
    ("a special whose text is a token", file, [ ("ab", 300) ]);
    ("a special on a token's id", file, [ ("<|x|>", 256) ]);
    ( "a special while a token is not built",
      rank_file (single_bytes @ [ ("abc", 256) ]),
      [ ("<|x|>", 300) ] );
  ]

let test_errors =
  cases "content errors"
    ~id:(fun (name, _, _) -> name)
    errors
    (fun (_, text, specials) ->
      is_error ~msg:"of_tiktoken" (of_tiktoken ~specials ~pattern:whole text))

let raising =
  [
    ("an invalid pattern", "(", []);
    ("an unsupported pattern", {|(?<=a)b|}, []);
    ("the empty pattern", "", []);
    ("a pattern matching the empty text", "a*", []);
    ("a lookahead matching empty", "(?=a)", []);
    ("an empty special", whole, [ ("", 300) ]);
    ("a negative id", whole, [ ("<|x|>", -1) ]);
    ("an id of 2^31", whole, [ ("<|x|>", 1 lsl 31) ]);
    ("a special given twice", whole, [ ("<|x|>", 300); ("<|x|>", 301) ]);
    ("two specials on one id", whole, [ ("<|x|>", 300); ("<|y|>", 300) ]);
    ( "a special that byte-level decoding changes",
      whole,
      [ ("<|\xc3\xa9|>", 300) ] );
  ]

let test_raising =
  cases "argument errors"
    ~id:(fun (name, _, _) -> name)
    raising
    (fun (_, pattern, specials) ->
      raises_match ~msg:"of_tiktoken" (Exn.invalid_arg ~sub:"") (fun () ->
          of_tiktoken ~specials ~pattern (rank_file base)))

(* Random tables *)

(* A table of every byte and up to a dozen tokens over [a], [b] and [c], each at
   a distinct rank, the ranks contiguous or with gaps up to 2^22, and handed out
   in a random order, so that a byte can rank above a token built from it. *)
let table =
  let open Gen in
  let* words =
    list ~size:(int_range 0 12)
      (string_of ~size:(int_range 2 5) (char_range 'a' 'c'))
  in
  let words = List.sort_uniq compare words in
  let entries = List.map fst single_bytes @ words in
  let n = List.length entries in
  let* sparse = bool in
  let gap =
    if sparse then frequency [ (6, int_range 1 2); (1, int_range 1 (1 lsl 22)) ]
    else constant 1
  in
  let* gaps = list ~size:(constant n) gap in
  let+ order = permutation entries in
  let ranks =
    List.rev
      (snd
         (List.fold_left
            (fun (r, acc) g -> (r + g, (r + g) :: acc))
            (0, []) gaps))
  in
  List.combine order ranks

let case =
  let open Gen in
  let* ranks = table in
  let tokens = List.filter (fun (b, _) -> String.length b > 1) ranks in
  let pieces = List.map fst tokens @ [ "a"; "b"; "c"; "d" ] in
  let+ text =
    frequency
      [
        (1, of_list ~pp:Format.pp_print_string pieces);
        ( 3,
          map (String.concat "")
            (list ~size:(int_range 0 30)
               (of_list ~pp:Format.pp_print_string pieces)) );
      ]
  in
  (ranks, text)

let built ranks token = merge ranks token = [ token ]

let splits ranks token =
  List.filter
    (fun i ->
      List.mem_assoc (String.sub token 0 i) ranks
      && List.mem_assoc (String.sub token i (String.length token - i)) ranks)
    (List.init (String.length token - 1) (fun i -> i + 1))

let test_reference =
  prop "agrees with the reference rule" ~count:300 case (fun (ranks, text) ->
      let tokens = List.filter (fun (b, _) -> String.length b > 1) ranks in
      cover "a token not built"
        (List.exists (fun (b, _) -> not (built ranks b)) tokens);
      cover "a built token ranked below one of its parts"
        (List.exists
           (fun (b, r) ->
             built ranks b
             && List.exists
                  (fun i ->
                    List.assoc (String.sub b 0 i) ranks > r
                    || List.assoc (String.sub b i (String.length b - i)) ranks
                       > r)
                  (splits ranks b))
           tokens);
      cover "a token with two splits"
        (List.exists (fun (b, _) -> List.length (splits ranks b) >= 2) tokens);
      cover "a rank past 2^22" (List.exists (fun (_, r) -> r > 1 lsl 22) ranks);
      cover "contiguous ranks"
        (List.for_all (fun (_, r) -> r <= List.length ranks) ranks);
      cover "a text that is a token not built"
        (List.mem_assoc text tokens && not (built ranks text));
      cover "a pretoken past the rank scan" (String.length text > 32);
      let t = load ranks in
      let expected = reference ranks text in
      equal ~msg:"ids" (list int) expected (ids t text);
      equal ~msg:"ids again, from the cache" (list int) expected (ids t text);
      equal ~msg:"decode" string text (decode t (encode_ids t text)))

(* Goldens *)

let member name = function
  | Jsont.Object (members, _) -> (
      match Jsont.Json.find_mem name members with
      | Some (_, value) -> value
      | None -> failf "no %S member" name)
  | _ -> failf "looking for %S in something that is not an object" name

let to_string = function Jsont.String (s, _) -> s | _ -> failf "not a string"

let to_int = function
  | Jsont.Number (f, _) -> int_of_float f
  | _ -> failf "not a number"

let to_list = function Jsont.Array (l, _) -> l | _ -> failf "not an array"

type golden = {
  name : string;
  tokenizer : Brot.t Lazy.t;
  specials : (string * int) list;
  texts : (string * int list * int list * string list) list;
}

let goldens =
  let document =
    match
      Jsont_bytesrw.decode_string Jsont.json
        (Fixture.read "support/tiktoken/goldens.json")
    with
    | Ok json -> json
    | Error e -> failf "goldens.json: %s" e
  in
  List.map
    (fun e ->
      let name = to_string (member "name" e) in
      let ranks =
        Fixture.read ("support/tiktoken/" ^ to_string (member "ranks" e))
      in
      let pattern = to_string (member "pattern" e) in
      let specials =
        List.map
          (fun p ->
            match to_list p with
            | [ s; id ] -> (to_string s, to_int id)
            | _ -> failf "a special is not a pair")
          (to_list (member "specials" e))
      in
      let tokenizer =
        lazy
          (let of_specials specials =
             match of_tiktoken ~specials ~pattern ranks with
             | Ok t -> t
             | Error msg -> failf "%s: %s" name msg
           in
           (* Whisper's table is contiguous: its specials are added after its
              ranks, as whisper numbers them. *)
           if name = "whisper" then
             add_tokens (of_specials [])
               (List.map (fun (s, _) -> added_token s) specials)
           else of_specials specials)
      in
      let texts =
        List.map
          (fun c ->
            ( to_string (member "text" c),
              List.map to_int (to_list (member "ordinary" c)),
              List.map to_int (to_list (member "all" c)),
              List.map to_string (to_list (member "decode_each" c)) ))
          (to_list (member "cases" e))
      in
      { name; tokenizer; specials; texts })
    (to_list (member "encodings" document))

let rows =
  List.concat_map (fun g -> List.map (fun text -> (g, text)) g.texts) goldens

let test_goldens =
  cases "tiktoken's ids"
    ~id:(fun (g, (text, _, _, _)) -> Printf.sprintf "%s %S" g.name text)
    rows
    (fun (g, (text, ordinary, all, decode_each)) ->
      let t = Lazy.force g.tokenizer in
      equal ~msg:"encode_ordinary" (list int) ordinary
        (Array.to_list (encode_ids t ~match_special_tokens:false text));
      equal ~msg:"encode, every special allowed" (list int) all (ids t text);
      equal ~msg:"decode" string text (decode t (Array.of_list ordinary));
      equal ~msg:"decode each id" (list string) decode_each
        (List.map (fun id -> decode t [| id |]) ordinary))

let test_special_ids =
  cases "special ids"
    ~id:(fun g -> g.name)
    goldens
    (fun g ->
      let t = Lazy.force g.tokenizer in
      List.iter
        (fun (s, id) ->
          equal ~msg:s (option int) (Some id) (token_to_id t s);
          equal ~msg:s (option string) (Some s) (id_to_token t id))
        g.specials)

(* Whisper's composition: a prompt of specials before the text, timestamps in
   the ids it decodes, which it drops before decoding. *)
let test_whisper () =
  let g = List.find (fun g -> g.name = "whisper") goldens in
  let t = Lazy.force g.tokenizer in
  let id s = List.assoc s g.specials in
  let timestamp_begin = id "<|0.00|>" in
  List.iter
    (fun (text, ordinary, _, _) ->
      let prompt =
        List.map id [ "<|startoftranscript|>"; "<|en|>"; "<|transcribe|>" ]
      in
      let all =
        prompt
        @ (id "<|0.00|>" :: ordinary)
        @ [ id "<|1.08|>"; id "<|endoftext|>" ]
      in
      let kept = List.filter (fun i -> i < timestamp_begin) all in
      equal ~msg:text string
        ("<|startoftranscript|><|en|><|transcribe|>" ^ text ^ "<|endoftext|>")
        (decode t (Array.of_list kept)))
    g.texts

(* A tokenizer saved in HuggingFace's format and read back encodes as it did. *)
let test_round_trip =
  cases "of_json (to_json t)"
    ~id:(fun g -> g.name)
    goldens
    (fun g ->
      let t = Lazy.force g.tokenizer in
      match of_json (to_json t) with
      | Error e -> failf "of_json: %s" e
      | Ok t' ->
          List.iter
            (fun (text, _, all, _) ->
              equal ~msg:text (list int) all (ids t' text))
            g.texts)

let () =
  exit
    (run "tiktoken encodings"
       [
         group "rank files"
           [
             test "a join no merge list holds" test_unlisted_join;
             test "a token no join builds" test_unbuilt;
             test "gaps, the empty token and line forms" test_file_forms;
             test "text no match covers" test_uncovered;
             test "a rank with leading zeros" test_leading_zeros;
             test "a special's text with every token built" test_special_spelled;
             test "a special that is a prefix of another" test_longer_special;
             test_errors;
             test_raising;
           ];
         group "rule" [ test_reference ];
         group "encodings"
           [
             test_goldens;
             test_special_ids;
             test "Whisper's composition" test_whisper;
             test_round_trip;
           ];
       ])
