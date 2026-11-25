(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* tiktoken encodings. The goldens in support/tiktoken are tiktoken 0.14.0's
   output on trimmed rank files, written by gen/gen_tiktoken.py. *)

open Windtrap
open Brot

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
  require_ok ~pp:Format.pp_print_string
    (of_tiktoken ?specials ~pattern (rank_file ranks))

let ids t text = Array.to_list (encode_ids t text)

let ordinary t text =
  Array.to_list (encode_ids t ~match_special_tokens:false text)

(* The rule on one pretoken, as the .mli states it: a token is itself, and any
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

let built ranks token = merge ranks token = [ token ]

let splits ranks token =
  List.filter
    (fun i ->
      List.mem_assoc (String.sub token 0 i) ranks
      && List.mem_assoc (String.sub token i (String.length token - i)) ranks)
    (List.init (String.length token - 1) (fun i -> i + 1))

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

(* A table and a text made of its tokens and single letters, as one pretoken. *)
let table_and_text =
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

(* tiktoken encoding *)

let test_rule =
  prop "joins the adjacent pair whose join ranks lowest, until none joins"
    ~count:300 table_and_text (fun (ranks, text) ->
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
      equal ~msg:"ids of the text seen before" (list int) expected (ids t text))

let test_decodes =
  prop "decodes the ids of a text back to the text" table_and_text
    (fun (ranks, text) ->
      let t = load ranks in
      equal string text (decode t (encode_ids t text)))

let abc =
  List.filter (fun (b, _) -> b <> "a" && b <> "b" && b <> "c") single_bytes

(* On [abca] the rule joins [b c], then [a bc]. *)
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

(* rank file *)

let test_gaps () =
  let t = load (single_bytes @ [ ("ab", 300) ]) in
  equal ~msg:"ab" (list int) [ 300 ] (ids t "ab");
  equal ~msg:"vocab_size" int 257 (vocab_size t)

let test_empty_token () =
  let t = load (single_bytes @ [ ("", 5000) ]) in
  equal ~msg:"id_to_token" (option string) (Some "") (id_to_token t 5000);
  equal ~msg:"decode" string "" (decode t [| 5000 |])

let test_line_forms () =
  let ranks = single_bytes @ [ ("ab", 300); ("", 5000) ] in
  let spelled =
    String.concat ""
      (List.map
         (fun (b, r) ->
           Printf.sprintf "\t%s \x0b\x0c %d\r\n\n\r"
             (if b = "" then "=" else base64 b)
             r)
         ranks)
  in
  let t =
    require_ok ~pp:Format.pp_print_string (of_tiktoken ~pattern:whole spelled)
  in
  equal (list int) (ids (load ranks) "ab xyz ab") (ids t "ab xyz ab")

let test_leading_zeros () =
  let text =
    rank_file (single_bytes @ [ ("ab", 256) ]) ^ "YWJj 00000000000300\n"
  in
  let t =
    require_ok ~pp:Format.pp_print_string (of_tiktoken ~pattern:whole text)
  in
  equal (list int) [ 300 ] (ids t "abc")

(* pattern *)

let test_uncovered () =
  let t = load ~pattern:{|[a-z]+|} (single_bytes @ [ ("ab", 256) ]) in
  equal ~msg:"spaces and digits" (list int) [ 256; 99 ] (ids t " ab 12c ");
  equal ~msg:"a stray byte" (list int) [ 97; 98 ]
    (ids (load single_bytes) "a\xffb")

(* U+A7DD, a Latin letter assigned in Unicode 18.0.0, is a [\p{L}] and so a
   pretoken of its three bytes. *)
let test_unicode_version () =
  let t = load ~pattern:{|\p{L}+|} single_bytes in
  equal (list int) [ 0xea; 0x9f; 0x9d ] (ids t "\u{A7DD}")

(* of_tiktoken *)

type golden = {
  name : string;
  tokenizer : Brot.t;
  specials : (string * int) list;
  texts : (string * int list * int list * string list) list;
}

let to_string = function Jsont.String (s, _) -> s | _ -> failf "not a string"

let to_int = function
  | Jsont.Number (f, _) -> int_of_float f
  | _ -> failf "not a number"

let to_list = function Jsont.Array (l, _) -> l | _ -> failf "not an array"

let member name = function
  | Jsont.Object (members, _) -> (
      match Jsont.Json.find_mem name members with
      | Some (_, value) -> value
      | None -> failf "no %S member" name)
  | _ -> failf "looking for %S in something that is not an object" name

let golden e =
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
  let of_specials specials =
    require_ok ~pp:Format.pp_print_string (of_tiktoken ~specials ~pattern ranks)
  in
  (* Whisper's table is contiguous: its specials are added after its ranks, as
     whisper numbers them. *)
  let tokenizer =
    if name = "whisper" then
      add_tokens (of_specials [])
        (List.map (fun (s, _) -> added_token s) specials)
    else of_specials specials
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
  { name; tokenizer; specials; texts }

let goldens =
  fixture (fun () ->
      match
        Jsont_bytesrw.decode_string Jsont.json
          (Fixture.read "support/tiktoken/goldens.json")
      with
      | Ok json -> List.map golden (to_list (member "encodings" json))
      | Error e -> failf "goldens.json: %s" e)

let encodings =
  [ "gpt2"; "multilingual99"; "multilingual100"; "cl100k"; "o200k"; "whisper" ]

let encoding name = List.find (fun g -> g.name = name) (goldens ())

let test_encode_ordinary =
  cases "gives tiktoken's encode_ordinary under match_special_tokens:false"
    ~id:Fun.id encodings (fun name ->
      let g = encoding name in
      List.iter
        (fun (text, expected, _, _) ->
          equal ~msg:text (list int) expected (ordinary g.tokenizer text))
        g.texts)

let test_encode_all =
  cases "gives tiktoken's encode with every special allowed" ~id:Fun.id
    encodings (fun name ->
      let g = encoding name in
      List.iter
        (fun (text, _, expected, _) ->
          equal ~msg:text (list int) expected (ids g.tokenizer text))
        g.texts)

let test_decode =
  cases "decodes as tiktoken's decode with errors=replace" ~id:Fun.id encodings
    (fun name ->
      let g = encoding name in
      List.iter
        (fun (text, ordinary, _, each) ->
          equal ~msg:text string text
            (decode g.tokenizer (Array.of_list ordinary));
          equal ~msg:(text ^ ", each id") (list string) each
            (List.map (fun id -> decode g.tokenizer [| id |]) ordinary))
        g.texts)

let test_special_ids =
  cases "puts each special at its id" ~id:Fun.id encodings (fun name ->
      let g = encoding name in
      List.iter
        (fun (s, id) ->
          equal ~msg:s (option int) (Some id) (token_to_id g.tokenizer s);
          equal ~msg:s (option string) (Some s) (id_to_token g.tokenizer id))
        g.specials)

(* [abc] is a token no join builds: it is emitted whole, and spelled out inside
   a longer pretoken. *)
let test_unbuilt () =
  let t =
    load (abc @ [ ("a", 1000); ("b", 1001); ("c", 1002); ("abc", 1003) ])
  in
  equal ~msg:"abc" (list int) [ 1003 ] (ids t "abc");
  equal ~msg:"abca" (list int) [ 1000; 1001; 1002; 1000 ] (ids t "abca")

(* With every token built, a pretoken that is a special's text is spelled out by
   the merges, as encode_ordinary spells it. *)
let test_special_spelled () =
  let t = load ~specials:[ ("<|x|>", 300) ] (single_bytes @ [ ("ab", 256) ]) in
  equal ~msg:"encode_ordinary" (list int) [ 60; 124; 120; 124; 62 ]
    (ordinary t "<|x|>");
  equal ~msg:"encode" (list int) [ 300 ] (ids t "<|x|>")

let test_longer_special () =
  let t = load ~specials:[ ("<|x", 300); ("<|x|>", 301) ] single_bytes in
  equal ~msg:"<|x|>" (list int) [ 301 ] (ids t "<|x|>");
  equal ~msg:"<|x|" (list int) [ 300; 124 ] (ids t "<|x|")

(* A prompt of specials, timestamps the caller drops, then the text. *)
let test_whisper () =
  let g = encoding "whisper" in
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
        (decode g.tokenizer (Array.of_list kept)))
    g.texts

let base = single_bytes @ [ ("ab", 256) ]

let test_errors =
  let file = rank_file base in
  let drop_ff = rank_file (List.filter (fun (_, r) -> r <> 255) base) in
  cases "errors on a rank file or special it cannot take"
    ~id:(fun (name, _, _) -> name)
    [
      ("a line with one field", file ^ "QQ==\n", []);
      ("a line with three fields", file ^ "YWJj 300 1\n", []);
      ("a line of white space", file ^ " \n", []);
      ("base64 without its padding", file ^ "YWJjZA 300\n", []);
      ("a character outside base64", file ^ "YW@= 300\n", []);
      (* [YWN=] holds [ac] and two bits left over. *)
      ("bits left over in the last digit", file ^ "YWN= 300\n", []);
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
    (fun (_, text, specials) ->
      is_error ~pp:Brot.pp (of_tiktoken ~specials ~pattern:whole text))

let test_raising =
  cases "raises Invalid_argument on a pattern or special it cannot take"
    ~id:(fun (name, _, _) -> name)
    [
      ("an invalid pattern", "(", []);
      ("an unsupported pattern", {|(?<=a)b|}, []);
      ("a possessive quantifier", {|\p{L}++|}, []);
      ("an end anchor", {|\s+$|}, []);
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
    (fun (_, pattern, specials) ->
      raises_match (Exn.invalid_arg ~sub:"") (fun () ->
          of_tiktoken ~specials ~pattern (rank_file base)))

let test_read_back =
  cases
    "is read back by of_json from to_json into a tokenizer that encodes alike"
    ~id:Fun.id encodings (fun name ->
      let g = encoding name in
      let read =
        require_ok ~pp:Format.pp_print_string (of_json (to_json g.tokenizer))
      in
      List.iter
        (fun (text, _, all, _) ->
          equal ~msg:text (list int) all (ids read text))
        g.texts)

let () =
  exit
    (run "tiktoken encodings"
       [
         group "tiktoken encoding"
           [
             test_rule;
             test_decodes;
             test
               "joins the two parts the rule joins last, which no merge list \
                of [a b] and [ab c] holds"
               test_unlisted_join;
           ];
         group "rank file"
           [
             test "ranks may leave gaps, and vocab_size counts tokens" test_gaps;
             test "a field = alone is the empty token" test_empty_token;
             test
               "a carriage return or a line feed ends a line, and ASCII white \
                space separates fields"
               test_line_forms;
             test "a rank may have leading zeros" test_leading_zeros;
           ];
         group "pattern"
           [
             test
               "drops the text no match covers, bytes that are no character \
                included"
               test_uncovered;
             test "classifies characters by Unicode 18.0.0" test_unicode_version;
           ];
         group "of_tiktoken"
           [
             test_encode_ordinary;
             test_encode_all;
             test_decode;
             test_special_ids;
             test "emits whole a pretoken that is a token no join builds"
               test_unbuilt;
             test "spells out a special's text under match_special_tokens:false"
               test_special_spelled;
             test
               "matches the longer of two specials where one is a prefix of \
                the other"
               test_longer_special;
             test "decodes specials added with add_tokens after the ranks"
               test_whisper;
             test_errors;
             test_raising;
             test_read_back;
           ];
       ])
