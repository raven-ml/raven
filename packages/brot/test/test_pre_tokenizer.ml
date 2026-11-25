(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Pre = Brot.Pre_tokenizer

(* Pretokens *)

let pretokens = list (pair string (pair int int))
let source text (start, stop) = String.sub text start (stop - start)

(* A row of a table: what the pre-tokenizer [t] gives on [text]. *)
let cuts (t, text, expected) =
  equal pretokens expected (Pre.pre_tokenize t text)

let cut_id (t, text, _) = Format.asprintf "%a on %S" Pre.pp t text
let pre_id t = Format.asprintf "%a" Pre.pp t
let text_id (text, _) = Printf.sprintf "%S" text

let rec adjacent = function
  | p :: (q :: _ as rest) -> (p, q) :: adjacent rest
  | _ -> []

(* Texts *)

let quoted ppf s = Format.fprintf ppf "%S" s

(* A text of up to 24 fragments, printed as the fragments it joins. *)
let text_of fragments =
  Gen.map (String.concat "")
    (Gen.list ~size:(Gen.int_range 0 24) (Gen.of_list ~pp:quoted fragments))

(* A character of each class some pre-tokenizer tells apart, the space twice
   more and the apostrophe, which opens a contraction, once more. *)
let characters =
  [ (* Letters *)
    "a"; "Z"; "s"; "t"; "e"; "l"; "r"; "v"; "m"; "d"; "S"; "\u{e9}";
    "\u{df}"; "\u{41f}"; "\u{5e9}"; "\u{627}"; "\u{4e2d}"; "\u{3042}";
    "\u{30a2}"; "\u{d55c}";
    (* Numbers: decimal, other and letter *)
    "1"; "9"; "\u{661}"; "\u{ff11}"; "\u{bd}"; "\u{b2}"; "\u{2460}"; "\u{2167}";
    (* White space *)
    " "; " "; " "; "\t"; "\n"; "\r"; "\x0b"; "\x0c"; "\u{85}"; "\u{a0}";
    "\u{1680}"; "\u{2028}"; "\u{3000}";
    (* Characters that are not white space *)
    "\x00"; "\x1c"; "\u{200b}";
    (* Punctuation *)
    "'"; "'"; "!"; "."; ","; "_"; "-"; "/"; "\u{ab}"; "\u{2014}"; "\u{3001}";
    (* Symbols *)
    "$"; "+"; "\u{b1}"; "\u{a9}";
    (* Marks, joiners, an emoji, private use and the metaspace marker *)
    "\u{301}"; "\u{f71}"; "\u{200d}"; "\u{fe0f}"; "\u{1f600}"; "\u{e000}";
    "\u{2581}" ]
[@@ocamlformat "disable"]

(* Bytes that open no whole character: truncated leads, lone continuation bytes,
   bytes that lead nothing, an overlong NUL, a surrogate, a lead above U+10FFFF
   and a lead before ASCII. *)
let malformed =
  [ "\xc3"; "\xe2\x80"; "\xf0\x9f\x98"; "\x80"; "\xbf"; "\xff"; "\xfe";
    "\xc0"; "\xc1"; "\xc0\x80"; "\xed\xa0\x80"; "\xf5\x80\x80\x80";
    "\xf8\x88\x80\x80"; "\xe0\x41\x42" ]
[@@ocamlformat "disable"]

let text = text_of characters
let bytes = text_of (characters @ malformed)

(* Pre-tokenizers *)

(* GPT-2's pattern, as brot.mli gives it for [byte_level]. *)
let gpt2 =
  {re|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+|re}

(* Pre-tokenizers whose pretokens are bytes of the text. *)
let verbatim =
  [
    Pre.whitespace;
    Pre.whitespace_split;
    Pre.bert;
    Pre.punctuation ();
    Pre.punctuation ~behavior:`Merged_with_next ();
    Pre.split ~pattern:"," ~behavior:`Contiguous ();
    Pre.split_regex ~pattern:gpt2 ~behavior:`Isolated ();
    Pre.char_delimiter "\u{2581}";
    Pre.digits ();
    Pre.digits ~individual_digits:true ();
    Pre.unicode_scripts;
    Pre.fixed_length 2;
    Pre.sequence [ Pre.whitespace_split; Pre.punctuation () ];
  ]

(* Pre-tokenizers that rewrite or encode the text. *)
let rewriting =
  [
    Pre.byte_level ();
    Pre.byte_level ~use_regex:false ();
    Pre.metaspace ();
    Pre.metaspace ~split:false ();
    Pre.metaspace ~replacement:"_" ~prepend_scheme:`Never ();
    Pre.sequence [ Pre.whitespace_split; Pre.metaspace () ];
    Pre.sequence [ Pre.metaspace (); Pre.punctuation () ];
    Pre.sequence [ Pre.whitespace_split; Pre.metaspace (); Pre.punctuation () ];
    Pre.sequence [ Pre.byte_level (); Pre.punctuation () ];
    Pre.sequence [ Pre.fixed_length 3; Pre.metaspace () ];
  ]

let member list = Gen.of_list ~pp:Pre.pp list

let behavior_name = function
  | `Isolated -> "`Isolated"
  | `Removed -> "`Removed"
  | `Merged_with_previous -> "`Merged_with_previous"
  | `Merged_with_next -> "`Merged_with_next"
  | `Contiguous -> "`Contiguous"

let behaviors =
  [ `Isolated; `Removed; `Merged_with_previous; `Merged_with_next; `Contiguous ]

let behavior =
  Gen.of_list
    ~pp:(fun ppf b -> Format.pp_print_string ppf (behavior_name b))
    behaviors

(* Goldens

   What HuggingFace tokenizers 0.23.1 gives, which no test can compute, as
   gen/gen_split_regex_expected.py writes it. *)

let field name = function
  | Jsont.Object (members, _) -> (
      match Jsont.Json.find_mem name members with
      | Some (_, value) -> value
      | None -> failf "the golden has no member %S" name)
  | _ -> failf "the golden has no object holding %S" name

let as_string = function
  | Jsont.String (s, _) -> s
  | _ -> fail "the golden has no string here"

let as_int = function
  | Jsont.Number (n, _) -> int_of_float n
  | _ -> fail "the golden has no number here"

let as_list = function
  | Jsont.Array (items, _) -> items
  | _ -> fail "the golden has no array here"

let read_json file =
  match Jsont_bytesrw.decode_string Jsont.json (Fixture.read file) with
  | Ok json -> json
  | Error e -> failf "%s: %s" file e

(* A text and its pretokens, each written [[piece, start, stop]]. *)
let golden_case json =
  let pretoken json =
    match as_list json with
    | [ piece; start; stop ] -> (as_string piece, (as_int start, as_int stop))
    | _ -> fail "the golden has a pretoken that is not [piece, start, stop]"
  in
  ( as_string (field "text" json),
    List.map pretoken (as_list (field "pretokens" json)) )

let behavior_of_string = function
  | "isolated" -> `Isolated
  | "removed" -> `Removed
  | "merged_with_previous" -> `Merged_with_previous
  | "merged_with_next" -> `Merged_with_next
  | "contiguous" -> `Contiguous
  | name -> failf "the golden names no behavior %S" name

type split_set = {
  group : string;
  pattern : string;
  behavior : Pre.behavior;
  invert : bool;
  cases : (string * (string * (int * int)) list) list;
}

let split_golden =
  fixture (fun () ->
      let json = read_json "support/pre_tokenizer/split_regex.json" in
      let patterns =
        match field "patterns" json with
        | Jsont.Object (members, _) ->
            List.map (fun ((name, _), value) -> (name, as_string value)) members
        | _ -> fail "the golden's patterns are not an object"
      in
      let set json =
        {
          group = as_string (field "group" json);
          pattern = as_string (field "pattern" json);
          behavior = behavior_of_string (as_string (field "behavior" json));
          invert = field "invert" json = Jsont.Json.bool true;
          cases = List.map golden_case (as_list (field "cases" json));
        }
      in
      (patterns, List.map set (as_list (field "sets" json))))

let scripts_golden =
  fixture (fun () ->
      List.map golden_case
        (as_list (read_json "support/pre_tokenizer/unicode_scripts.json")))

(* A pattern of tokenizer files by its name in the golden: cl100k, cl100k_hf,
   qwen2, qwen35, o200k, tekken or kimi. *)
let pattern name = List.assoc name (fst (split_golden ()))

(* A pattern brot knows by name gives the same pretokens written in a group,
   which brot does not recognise. *)
let in_a_group pattern = "(?:" ^ pattern ^ ")"

let matches_huggingface group () =
  List.iter
    (fun { group = g; pattern; behavior; invert; cases } ->
      if g = group then begin
        let written = Pre.split_regex ~pattern ~behavior ~invert () in
        let grouped =
          Pre.split_regex ~pattern:(in_a_group pattern) ~behavior ~invert ()
        in
        List.iter
          (fun (text, expected) ->
            subtest
              (Printf.sprintf "%S %s ~invert:%b on %S" pattern
                 (behavior_name behavior) invert text)
              (fun () ->
                equal ~msg:"as written" pretokens expected
                  (Pre.pre_tokenize written text);
                equal ~msg:"in a group" pretokens expected
                  (Pre.pre_tokenize grouped text)))
          cases
      end)
    (snd (split_golden ()))

(* whitespace *)

(* Every expectation is the output of HuggingFace's [Whitespace]. *)
let whitespace_rows =
  [
    ("", []);
    ("   \t\n  ", []);
    ("Hello world", [ ("Hello", (0, 5)); ("world", (6, 11)) ]);
    ( "Hello, world!",
      [ ("Hello", (0, 5)); (",", (5, 6)); ("world", (7, 12)); ("!", (12, 13)) ]
    );
    ("test_var", [ ("test_var", (0, 8)) ]);
    ("123abc", [ ("123abc", (0, 6)) ]);
    ( "a+b=c",
      [
        ("a", (0, 1)); ("+", (1, 2)); ("b", (2, 3)); ("=", (3, 4)); ("c", (4, 5));
      ] );
    ( "don't stop, it's fun!",
      [
        ("don", (0, 3));
        ("'", (3, 4));
        ("t", (4, 5));
        ("stop", (6, 10));
        (",", (10, 11));
        ("it", (12, 14));
        ("'", (14, 15));
        ("s", (15, 16));
        ("fun", (17, 20));
        ("!", (20, 21));
      ] );
    ( "I have 42 apples and 3.14 pies",
      [
        ("I", (0, 1));
        ("have", (2, 6));
        ("42", (7, 9));
        ("apples", (10, 16));
        ("and", (17, 20));
        ("3", (21, 22));
        (".", (22, 23));
        ("14", (23, 25));
        ("pies", (26, 30));
      ] );
    ( "hello@world.com #ml $100 C++",
      [
        ("hello", (0, 5));
        ("@", (5, 6));
        ("world", (6, 11));
        (".", (11, 12));
        ("com", (12, 15));
        ("#", (16, 17));
        ("ml", (17, 19));
        ("$", (20, 21));
        ("100", (21, 24));
        ("C", (25, 26));
        ("++", (26, 28));
      ] );
    ( "hello-world test_123",
      [
        ("hello", (0, 5));
        ("-", (5, 6));
        ("world", (6, 11));
        ("test_123", (12, 20));
      ] );
    ( "wow!!! really???",
      [
        ("wow", (0, 3)); ("!!!", (3, 6)); ("really", (7, 13)); ("???", (13, 16));
      ] );
    ( "hello\tworld\nthere\r\nfriend",
      [
        ("hello", (0, 5));
        ("world", (6, 11));
        ("there", (12, 17));
        ("friend", (19, 25));
      ] );
  ]

(* A combining mark, a letter number, an undertie and the join controls are word
   characters; an other number and a circled digit are not. Every expectation is
   the output of HuggingFace's [Whitespace]. *)
let word_character_rows =
  [
    ("a\u{f71}b", [ ("a\u{f71}b", (0, 5)) ]);
    ("a\u{2160}b", [ ("a\u{2160}b", (0, 5)) ]);
    ("a\u{203f}b", [ ("a\u{203f}b", (0, 5)) ]);
    ("a\u{200d}b", [ ("a\u{200d}b", (0, 5)) ]);
    ("a\u{200c}b", [ ("a\u{200c}b", (0, 5)) ]);
    ("a\u{bd}b", [ ("a", (0, 1)); ("\u{bd}", (1, 3)); ("b", (3, 4)) ]);
    ("a\u{2460}b", [ ("a", (0, 1)); ("\u{2460}", (1, 4)); ("b", (4, 5)) ]);
    ("a\u{a0}b", [ ("a", (0, 1)); ("b", (3, 4)) ]);
  ]

let whitespace_cuts (text, expected) =
  equal pretokens expected (Pre.pre_tokenize Pre.whitespace text)

let whitespace_pattern =
  lazy
    (Pre.split_regex ~pattern:{|\w+|[^\w\s]+|} ~behavior:`Removed ~invert:true
       ())

(* Valid UTF-8 without the join controls and \u{b2} and \u{bd}, where
   split_regex's [\w] differs from whitespace's. *)
let word_text =
  text_of
    (List.filter
       (fun c -> not (List.mem c [ "\u{200d}"; "\u{b2}"; "\u{bd}" ]))
       characters)

let is_its_pattern text =
  equal pretokens
    (Pre.pre_tokenize (Lazy.force whitespace_pattern) text)
    (Pre.pre_tokenize Pre.whitespace text)

(* whitespace_split *)

let white_space = lazy (Pre.split_regex ~pattern:{|\s+|} ())

let splits_on_white_space text =
  equal pretokens
    (Pre.pre_tokenize (Lazy.force white_space) text)
    (Pre.pre_tokenize Pre.whitespace_split text)

(* Next line, no-break space, Ogham space mark, line separator, ideographic
   space and the ASCII controls have the White_Space property; the information
   separator U+001C and zero width space do not. Every expectation is the output
   of HuggingFace's [WhitespaceSplit]. *)
let white_space_rows =
  [
    ("a\u{85}b", [ ("a", (0, 1)); ("b", (3, 4)) ]);
    ("a\u{a0}b", [ ("a", (0, 1)); ("b", (3, 4)) ]);
    ("a\u{1680}b", [ ("a", (0, 1)); ("b", (4, 5)) ]);
    ("a\u{2028}b", [ ("a", (0, 1)); ("b", (4, 5)) ]);
    ("a\u{3000}b", [ ("a", (0, 1)); ("b", (4, 5)) ]);
    ( "a\tb\nc\rd\x0be\x0cf",
      [
        ("a", (0, 1));
        ("b", (2, 3));
        ("c", (4, 5));
        ("d", (6, 7));
        ("e", (8, 9));
        ("f", (10, 11));
      ] );
    ("a\x1cb", [ ("a\x1cb", (0, 3)) ]);
    ("a\u{200b}b", [ ("a\u{200b}b", (0, 5)) ]);
  ]

let white_space_split (text, expected) =
  equal pretokens expected (Pre.pre_tokenize Pre.whitespace_split text)

(* bert *)

let is_whitespace_split_then_punctuation text =
  equal pretokens
    (Pre.pre_tokenize
       (Pre.sequence [ Pre.whitespace_split; Pre.punctuation () ])
       text)
    (Pre.pre_tokenize Pre.bert text)

(* byte_level *)

(* The matches as bytes of the text. Every expectation is the output of
   HuggingFace's [ByteLevel(add_prefix_space=False)]. *)
let gpt2_rows =
  [
    (* A run of white space gives its last character to what follows, unless it
       ends the text. *)
    ("\n\nNot", [ "\n"; "\n"; "Not" ]);
    ("  .", [ " "; " ." ]);
    ("x  y", [ "x"; " "; " y" ]);
    ("a \n b", [ "a"; " \n"; " b" ]);
    ("a  ", [ "a"; "  " ]);
    ("  \n  x", [ "  \n "; " x" ]);
    (" \n ", [ " \n " ]);
    ("a\r\nb", [ "a"; "\r"; "\n"; "b" ]);
    ("   ", [ "   " ]);
    (* The optional character before a letter, number or other run is a space,
       and no other white space. *)
    ("\tabc", [ "\t"; "abc" ]);
    ("\t\tabc", [ "\t"; "\t"; "abc" ]);
    ("\x0b\x0cx", [ "\x0b"; "\x0c"; "x" ]);
    (* White space is the White_Space property. *)
    ("a\u{a0}b", [ "a"; "\u{a0}"; "b" ]);
    ("a \u{a0}b", [ "a"; " "; "\u{a0}"; "b" ]);
    ("a\u{a0} b", [ "a"; "\u{a0}"; " b" ]);
    ("a\u{2003}b", [ "a"; "\u{2003}"; "b" ]);
    ("a\u{3000}b", [ "a"; "\u{3000}"; "b" ]);
    ("a\u{85}b", [ "a"; "\u{85}"; "b" ]);
    ("a\u{2028}b", [ "a"; "\u{2028}"; "b" ]);
    (* Letters are the Letter category, so a combining mark opens a run of its
       own; zero width space is not white space. *)
    ("e\u{301}x", [ "e"; "\u{301}"; "x" ]);
    ("a\u{f71}b", [ "a"; "\u{f71}"; "b" ]);
    ("a\u{200b}b", [ "a"; "\u{200b}"; "b" ]);
    (* Numbers are the Number category, letter and other numbers included. *)
    ("1\u{bd}2", [ "1\u{bd}2" ]);
    ("\u{2160}\u{2161}", [ "\u{2160}\u{2161}" ]);
    ("...", [ "..." ]);
    (* Contractions are lower case only. *)
    ("don't", [ "don"; "'t" ]);
    ("'ll'LL", [ "'ll"; "'"; "LL" ]);
    (" 's", [ " '"; "s" ]);
  ]

let gpt2_cuts (text, expected) =
  equal (list string) expected
    (List.map
       (fun (_, span) -> source text span)
       (Pre.pre_tokenize (Pre.byte_level ~add_prefix_space:false ()) text))

(* What the walked patterns tell characters apart by. *)
type kind =
  | Upper  (** [Lu] and [Lt]: in o200k's [U] only. *)
  | Lower  (** [Ll]: in o200k's [W] only. *)
  | Both  (** [Lm] and [Lo]: letters in [U] and [W]. *)
  | Mark  (** In [U] and [W], and not a letter. *)
  | Number
  | Space  (** Whitespace other than the newlines. *)
  | Newline
  | Apostrophe
  | Other
  | Stray  (** A byte that belongs to no character. *)

(* One representative of everything the patterns tell apart. *)
let kinds =
  [
    ("a", Lower); ("Z", Upper); ("s", Lower); ("S", Upper); ("t", Lower);
    ("l", Lower); ("L", Upper); ("r", Lower); ("v", Lower); ("e", Lower);
    ("E", Upper); ("m", Lower); ("d", Lower); ("\xC5\xBF", Lower);
    ("\xC3\xA9", Lower); ("\xC3\x81", Upper); ("\xC7\x85", Upper);
    ("\xCA\xB0", Both); ("\xE6\x97\xA5", Both); ("\xCC\x81", Mark);
    ("\xE0\xA4\xBF", Mark);
    ("1", Number); ("2", Number); ("\xD9\xA3", Number); ("\xC2\xB2", Number);
    ("\xE2\x85\xA7", Number);
    (" ", Space); ("\t", Space); ("\x0C", Space); ("\xC2\xA0", Space);
    ("\xE3\x80\x80", Space); ("\xE2\x80\xA8", Space); ("\xC2\x85", Space);
    ("\n", Newline); ("\r", Newline);
    ("'", Apostrophe);
    ("!", Other); (".", Other); ("/", Other); ("_", Other); ("\x00", Other);
    ("\xE2\x80\x9C", Other); ("\xF0\x9F\x98\x80", Other);
    ("\xE2\x80\x8D", Other); ("\xEF\xB8\x8F", Other);
  ]
[@@ocamlformat "disable"]

(* Bytes that belong to no character, an overlong form, a surrogate and a value
   above U+10FFFF among them. *)
let stray =
  [
    "\xFF"; "\x80"; "\xC3"; "\xE2\x82"; "\xE0\x81\x81"; "\xED\xA0\x80";
    "\xF4\x90\x80\x80";
  ]
[@@ocamlformat "disable"]

(* The spaces and the apostrophe twice as often, and runs that reach the
   alternatives single characters seldom do: number groups past three,
   contractions, camel case and newlines after spaces. *)
let runs =
  [ " "; " "; "'"; "1212"; "'s"; "'re"; "'LL"; "XMLHttp"; "  "; " \r\n" ]

let split_text = text_of (List.map fst kinds @ stray @ runs)

let by_gpt2 =
  lazy
    (Pre.sequence
       [
         Pre.split_regex ~pattern:gpt2 ~behavior:`Isolated ();
         Pre.byte_level ~add_prefix_space:false ~use_regex:false ();
       ])

let contractions = [ "'s"; "'t"; "'re"; "'ve"; "'m"; "'ll"; "'d" ]
let is_ascii_space c = c = ' ' || c = '\t' || c = '\n' || c = '\r'

let cuts_as_gpt2 text =
  let expected = Pre.pre_tokenize (Lazy.force by_gpt2) text in
  let words = List.map (fun (_, span) -> source text span) expected in
  let opens_with_space w =
    String.length w > 1 && w.[0] = ' ' && not (is_ascii_space w.[1])
  in
  cover "a byte of no character" (not (String.is_valid_utf_8 text));
  cover "a contraction" (List.exists (fun w -> List.mem w contractions) words);
  cover "a space opening a run" (List.exists opens_with_space words);
  cover "white space giving its last space to the run after it"
    (List.exists
       (fun (p, q) -> String.for_all is_ascii_space p && opens_with_space q)
       (adjacent words));
  equal pretokens expected
    (Pre.pre_tokenize (Pre.byte_level ~add_prefix_space:false ()) text)

(* A text of about 2000 pretokens, more than [pre_tokenize] hands out at
   once. *)
let long_text =
  let kinds = Array.of_list (List.map fst kinds) in
  String.concat ""
    (List.init 3000 (fun i -> kinds.(i * 7 mod Array.length kinds)))

let spells_its_spans (use_regex, text) =
  let t = Pre.byte_level ~add_prefix_space:false ~use_regex () in
  let stop =
    List.fold_left
      (fun at (piece, ((start, stop) as span)) ->
        equal ~msg:"a span starts where the one before stopped" int at start;
        less ~msg:"a span is not empty" int ~than:stop start;
        equal ~msg:"a pretoken spells its span" string (source text span)
          (Pre.byte_level_decode piece);
        stop)
      0 (Pre.pre_tokenize t text)
  in
  equal ~msg:"the spans reach the end of the text" int (String.length text) stop

(* Every expectation is the output of HuggingFace's [ByteLevel()]. The space
   stands at the character it opens. *)
let prefix_rows =
  [
    ("", []);
    ("hello", [ ("\u{120}hello", (0, 5)) ]);
    ("Hello world", [ ("\u{120}Hello", (0, 5)); ("\u{120}world", (5, 11)) ]);
    (" hello", [ ("\u{120}hello", (0, 6)) ]);
    ("  hello", [ ("\u{120}", (0, 1)); ("\u{120}hello", (1, 7)) ]);
    ("\tx", [ ("\u{120}", (0, 1)); ("\u{109}", (0, 1)); ("x", (1, 2)) ]);
    ("\nx", [ ("\u{120}", (0, 1)); ("\u{10a}", (0, 1)); ("x", (1, 2)) ]);
    ( "\u{a0}x",
      [ ("\u{120}", (0, 2)); ("\u{c2}\u{142}", (0, 2)); ("x", (2, 3)) ] );
  ]

let prefixed (text, expected) =
  equal pretokens expected (Pre.pre_tokenize (Pre.byte_level ()) text)

let ignores_trim_offsets (add_prefix_space, use_regex, text) =
  let t trim_offsets =
    Pre.byte_level ~add_prefix_space ~use_regex ~trim_offsets ()
  in
  equal pretokens
    (Pre.pre_tokenize (t true) text)
    (Pre.pre_tokenize (t false) text)

(* A byte-level member after a walker prepends its space to each pretoken the
   walker hands it and encodes each on its own; the space stands at the
   character it opens. Every expectation is the output of HuggingFace. *)
let byte_level_in_a_sequence_rows =
  let g = "\u{120}" in
  let whole =
    Pre.sequence [ Pre.whitespace_split; Pre.byte_level ~use_regex:false () ]
  in
  let regex = Pre.sequence [ Pre.whitespace_split; Pre.byte_level () ] in
  let punctuated = Pre.sequence [ Pre.punctuation (); Pre.byte_level () ] in
  let merged =
    Pre.sequence
      [
        Pre.split ~pattern:" " ~behavior:`Merged_with_next ();
        Pre.byte_level ~add_prefix_space:false ();
      ]
  in
  let removed = Pre.sequence [ Pre.split ~pattern:" " (); Pre.byte_level () ] in
  [
    (whole, "Hello world", [ (g ^ "Hello", (0, 5)); (g ^ "world", (6, 11)) ]);
    ( whole,
      " Hello  w\u{f6}rld ",
      [ (g ^ "Hello", (1, 6)); (g ^ "w\u{c3}\u{b6}rld", (8, 14)) ] );
    (whole, "a", [ (g ^ "a", (0, 1)) ]);
    (regex, "Hello world", [ (g ^ "Hello", (0, 5)); (g ^ "world", (6, 11)) ]);
    (regex, "it's  x", [ (g ^ "it", (0, 2)); ("'s", (2, 4)); (g ^ "x", (6, 7)) ]);
    (regex, "ab12 c", [ (g ^ "ab", (0, 2)); ("12", (2, 4)); (g ^ "c", (5, 6)) ]);
    ( punctuated,
      "Hello, world!",
      [
        (g ^ "Hello", (0, 5));
        (g ^ ",", (5, 6));
        (g ^ "world", (6, 12));
        (g ^ "!", (12, 13));
      ] );
    (punctuated, "a b", [ (g ^ "a", (0, 1)); (g ^ "b", (1, 3)) ]);
    (merged, "Hello world", [ ("Hello", (0, 5)); (g ^ "world", (5, 11)) ]);
    (removed, "Hello  world", [ (g ^ "Hello", (0, 5)); (g ^ "world", (7, 12)) ]);
  ]

(* The pretokens of a sequence reach the model encoded once. *)
let test_byte_level_encodes_once () =
  let tokenizer =
    Brot.bpe
      ~pre:(Pre.sequence [ Pre.whitespace_split; Pre.byte_level () ])
      ~vocab:[ ("Ġa", 0); ("Ġb", 1); ("Ġ", 2); ("a", 3); ("b", 4) ]
      ~merges:[ ("Ġ", "a"); ("Ġ", "b") ]
      ()
  in
  let encoding = Brot.encode tokenizer ~add_special_tokens:false "a b" in
  equal ~msg:"tokens" (array string) [| "Ġa"; "Ġb" |]
    (Brot.Encoding.tokens encoding);
  equal ~msg:"ids" (array int) [| 0; 1 |] (Brot.Encoding.ids encoding);
  equal ~msg:"encode_ids" (array int) [| 1; 0 |]
    (Brot.encode_ids tokenizer ~add_special_tokens:false "b a")

(* behavior *)

(* Every expectation is the output of HuggingFace tokenizers 0.23.1. *)
let behavior_rows =
  let punctuation behavior = Pre.punctuation ~behavior () in
  let split behavior ~invert = Pre.split ~pattern:"," ~behavior ~invert () in
  [
    ( punctuation `Isolated,
      "a,,b",
      [ ("a", (0, 1)); (",", (1, 2)); (",", (2, 3)); ("b", (3, 4)) ] );
    ( punctuation `Isolated,
      "\u{ab}x\u{bb}",
      [ ("\u{ab}", (0, 2)); ("x", (2, 3)); ("\u{bb}", (3, 5)) ] );
    (punctuation `Removed, "a,,b", [ ("a", (0, 1)); ("b", (3, 4)) ]);
    (punctuation `Removed, "\u{ab}x\u{bb}", [ ("x", (2, 3)) ]);
    ( punctuation `Merged_with_previous,
      "a,b,c",
      [ ("a,", (0, 2)); ("b,", (2, 4)); ("c", (4, 5)) ] );
    ( punctuation `Merged_with_previous,
      "a,,b",
      [ ("a,", (0, 2)); (",", (2, 3)); ("b", (3, 4)) ] );
    ( punctuation `Merged_with_previous,
      "\u{ab}x\u{bb}",
      [ ("\u{ab}", (0, 2)); ("x\u{bb}", (2, 5)) ] );
    ( punctuation `Merged_with_next,
      "a,b,c",
      [ ("a", (0, 1)); (",b", (1, 3)); (",c", (3, 5)) ] );
    ( punctuation `Merged_with_next,
      "a,,b",
      [ ("a", (0, 1)); (",", (1, 2)); (",b", (2, 4)) ] );
    ( punctuation `Merged_with_next,
      "\u{ab}x\u{bb}",
      [ ("\u{ab}x", (0, 3)); ("\u{bb}", (3, 5)) ] );
    ( punctuation `Contiguous,
      "a,b,c",
      [
        ("a", (0, 1)); (",", (1, 2)); ("b", (2, 3)); (",", (3, 4)); ("c", (4, 5));
      ] );
    ( punctuation `Contiguous,
      "a,,b",
      [ ("a", (0, 1)); (",,", (1, 3)); ("b", (3, 4)) ] );
    ( split `Merged_with_previous ~invert:false,
      "a,,,b",
      [ ("a,", (0, 2)); (",", (2, 3)); (",", (3, 4)); ("b", (4, 5)) ] );
    ( split `Merged_with_next ~invert:false,
      "a,,,b",
      [ ("a", (0, 1)); (",", (1, 2)); (",", (2, 3)); (",b", (3, 5)) ] );
    (split `Isolated ~invert:false, ",,", [ (",", (0, 1)); (",", (1, 2)) ]);
    (split `Isolated ~invert:true, ",,", [ (",", (0, 1)); (",", (1, 2)) ]);
    (split `Removed ~invert:false, ",,", []);
    (split `Removed ~invert:true, ",,", [ (",", (0, 1)); (",", (1, 2)) ]);
    (split `Removed ~invert:true, "a,,b", [ (",", (1, 2)); (",", (2, 3)) ]);
    ( split `Merged_with_previous ~invert:false,
      ",,",
      [ (",", (0, 1)); (",", (1, 2)) ] );
    ( split `Merged_with_previous ~invert:true,
      "a,,b",
      [ ("a", (0, 1)); (",", (1, 2)); (",b", (2, 4)) ] );
    ( split `Merged_with_next ~invert:false,
      ",,",
      [ (",", (0, 1)); (",", (1, 2)) ] );
    ( split `Merged_with_next ~invert:true,
      "a,,b",
      [ ("a,", (0, 2)); (",", (2, 3)); ("b", (3, 4)) ] );
    (split `Contiguous ~invert:false, ",,", [ (",,", (0, 2)) ]);
    ( split `Contiguous ~invert:true,
      "a,,b",
      [ ("a", (0, 1)); (",,", (1, 3)); ("b", (3, 4)) ] );
  ]

(* punctuation *)

(* The ASCII rows are every printable character other than a letter or a digit.
   The others: the left guillemet, the hyphen, the overline, the undertie and
   the inverted exclamation mark are of category P; plus-minus, cent, copyright,
   division and an arrow are symbols. Every (is, the, output) of HuggingFace's
   [Punctuation]. *)
let punctuation_rows =
  List.map
    (fun c -> (String.make 1 c, true))
    (List.of_seq (String.to_seq "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
  @ [
      ("\u{ab}", true);
      ("\u{2010}", true);
      ("\u{203e}", true);
      ("\u{203f}", true);
      ("\u{a1}", true);
      ("\u{b1}", false);
      ("\u{a2}", false);
      ("\u{a9}", false);
      ("\u{f7}", false);
      ("\u{2192}", false);
    ]

let delimits (c, is_punctuation) =
  let n = String.length c in
  let text = "a" ^ c ^ "b" in
  let expected =
    if is_punctuation then
      [ ("a", (0, 1)); (c, (1, 1 + n)); ("b", (1 + n, 2 + n)) ]
    else [ (text, (0, 2 + n)) ]
  in
  equal pretokens expected (Pre.pre_tokenize (Pre.punctuation ()) text)

(* The punctuation characters as a class of [split_regex]: the category P and
   the printable ASCII that is not a letter or a digit. *)
let punctuation_class = {|[\p{P}\x21-\x2F\x3A-\x40\x5B-\x60\x7B-\x7E]|}

let by_class =
  lazy
    (List.map
       (fun behavior ->
         (behavior, Pre.split_regex ~pattern:punctuation_class ~behavior ()))
       behaviors)

let splits_on_its_class (behavior, text) =
  equal pretokens
    (Pre.pre_tokenize (List.assoc behavior (Lazy.force by_class)) text)
    (Pre.pre_tokenize (Pre.punctuation ~behavior ()) text)

(* split *)

(* The text cut at the occurrences of [pattern], left to right without overlap,
   each part marked as a delimiter or not. [pattern] is not empty. *)
let occurrences ~pattern text =
  let n = String.length pattern and len = String.length text in
  let rec from i start parts =
    if i + n > len then
      List.rev (if start < len then (start, len, false) :: parts else parts)
    else if String.sub text i n <> pattern then from (i + 1) start parts
    else
      let parts = if start < i then (start, i, false) :: parts else parts in
      from (i + n) (i + n) ((i, i + n, true) :: parts)
  in
  from 0 0 []

(* The spans [behavior] makes of [parts], as brot.mli states each. *)
let behave behavior parts =
  let span (start, stop, _) = (start, stop) in
  let rec contiguous = function
    | (s, _, d) :: (_, e, d') :: rest when d = d' ->
        contiguous ((s, e, d) :: rest)
    | part :: rest -> span part :: contiguous rest
    | [] -> []
  in
  let rec with_previous = function
    | (s, _, false) :: (_, e, true) :: rest -> (s, e) :: with_previous rest
    | part :: rest -> span part :: with_previous rest
    | [] -> []
  in
  let rec with_next = function
    | (s, _, true) :: (_, e, false) :: rest -> (s, e) :: with_next rest
    | part :: rest -> span part :: with_next rest
    | [] -> []
  in
  match behavior with
  | `Isolated -> List.map span parts
  | `Removed ->
      List.filter_map (fun (s, e, d) -> if d then None else Some (s, e)) parts
  | `Contiguous -> contiguous parts
  | `Merged_with_previous -> with_previous parts
  | `Merged_with_next -> with_next parts

let split_patterns = [ ","; "::"; "ab"; "\u{3001}"; "\u{2581}" ]

let split_case =
  Gen.quad
    (Gen.of_list ~pp:quoted split_patterns)
    behavior Gen.bool
    (text_of ([ "a"; "b"; "x"; " "; ":"; "\u{e9}" ] @ split_patterns))

let follows_the_occurrences (pattern, behavior, invert, text) =
  let parts =
    List.map (fun (s, e, d) -> (s, e, d <> invert)) (occurrences ~pattern text)
  in
  cover "two delimiters side by side"
    (List.exists (fun ((_, _, d), (_, _, d')) -> d && d') (adjacent parts));
  let expected =
    List.map (fun span -> (source text span, span)) (behave behavior parts)
  in
  equal pretokens expected
    (Pre.pre_tokenize (Pre.split ~pattern ~behavior ~invert ()) text)

let empty_pattern_rows =
  let split behavior ~invert = Pre.split ~pattern:"" ~behavior ~invert () in
  let characters = [ ("a", (0, 1)); ("\u{e9}", (1, 3)) ] in
  [
    (split `Isolated ~invert:false, "a\u{e9}", characters);
    (split `Removed ~invert:false, "a\u{e9}", characters);
    (split `Contiguous ~invert:true, "a\u{e9}", characters);
    (split `Removed ~invert:true, "a\u{e9}", []);
  ]

(* split_regex *)

(* A byte that belongs to no character matches no class, so it is text between
   the matches, and [(?!\S)] does not hold before it, as before a character that
   is not white space. *)
let stray_rows =
  [
    ("R\x80", [ ("R", (0, 1)); ("\x80", (1, 2)) ]);
    ("ab\x80\x81 c", [ ("ab", (0, 2)); ("\x80\x81", (2, 4)); (" c", (4, 6)) ]);
    ( "1\x80!\xBF",
      [ ("1", (0, 1)); ("\x80", (1, 2)); ("!", (2, 3)); ("\xBF", (3, 4)) ] );
    ( " \x80\n\xC3",
      [ (" ", (0, 1)); ("\x80", (1, 2)); ("\n", (2, 3)); ("\xC3", (3, 4)) ] );
    ("a\xFFb", [ ("a", (0, 1)); ("\xFF", (1, 2)); ("b", (2, 3)) ]);
    ( "a \xFF\xFE!",
      [ ("a", (0, 1)); (" ", (1, 2)); ("\xFF\xFE", (2, 4)); ("!", (4, 5)) ] );
    ("a\xE2\x82", [ ("a", (0, 1)); ("\xE2\x82", (1, 3)) ]);
    ("  \xC3", [ (" ", (0, 1)); (" ", (1, 2)); ("\xC3", (2, 3)) ]);
    (* An overlong form, a surrogate and a value above U+10FFFF. *)
    ( "a\xE0\x81\x81b",
      [ ("a", (0, 1)); ("\xE0\x81\x81", (1, 4)); ("b", (4, 5)) ] );
    ( "!\xED\xA0\x80!",
      [ ("!", (0, 1)); ("\xED\xA0\x80", (1, 4)); ("!", (4, 5)) ] );
    ( " \xF4\x90\x80\x80 ",
      [ (" ", (0, 1)); ("\xF4\x90\x80\x80", (1, 5)); (" ", (5, 6)) ] );
  ]

let isolated pattern = Pre.split_regex ~pattern ~behavior:`Isolated ()

(* Every expectation is the output of HuggingFace's [Split(Regex("\\w+"))]. *)
let word_rows =
  [
    ("a\u{bd}b", [ ("a\u{bd}b", (0, 4)) ]);
    ( "a\u{b2}\u{b3}\u{b9}\u{bc}\u{be}bc",
      [ ("a\u{b2}\u{b3}\u{b9}\u{bc}\u{be}bc", (0, 13)) ] );
    ("a\u{200d}b", [ ("a", (0, 1)); ("\u{200d}", (1, 4)); ("b", (4, 5)) ]);
    ("a\u{200c}b", [ ("a", (0, 1)); ("\u{200c}", (1, 4)); ("b", (4, 5)) ]);
  ]

let word_runs = lazy (isolated {|\w+|})

let word_cuts (text, expected) =
  equal pretokens expected (Pre.pre_tokenize (Lazy.force word_runs) text)

(* OLMo 2 and Phi-4 ask for the same pretokens this way round: the matches are
   the text, and what lies between them, which is nothing in valid UTF-8, is
   removed. *)
let kept pattern = Pre.split_regex ~pattern ~behavior:`Removed ~invert:true ()

(* A pattern brot knows by name, as written and in a group. *)
type spelling = { written : Pre.t; grouped : Pre.t }

(* Each walked pattern's spellings, its matches isolated and kept. *)
type walked = { isolated : spelling; kept : spelling }

let walked =
  fixture (fun () ->
      List.map
        (fun name ->
          let p = pattern name and g = in_a_group (pattern name) in
          ( name,
            {
              isolated = { written = isolated p; grouped = isolated g };
              kept = { written = kept p; grouped = kept g };
            } ))
        [ "cl100k"; "cl100k_hf"; "qwen2"; "qwen35"; "o200k"; "tekken" ])

(* The kinds of the characters of [s], a stray byte for anything outside
   [kinds]. *)
let kinds_of s =
  let rec go i acc =
    if i >= String.length s then List.rev acc
    else
      let d = String.get_utf_8_uchar s i in
      let n = Uchar.utf_decode_length d in
      let k =
        if not (Uchar.utf_decode_is_valid d) then Stray
        else
          Option.value (List.assoc_opt (String.sub s i n) kinds) ~default:Stray
      in
      go (i + n) (k :: acc)
  in
  go 0 []

let is_letter = function Upper | Lower | Both | Mark -> true | _ -> false
let is_space = function Space | Newline -> true | _ -> false

(* Marks the cases that reach each alternative of o200k's pattern, from the
   kinds of the characters of each pretoken and of the one after it. *)
let cover_o200k ~digits pretokens =
  let kinds = List.map (fun (p, _) -> kinds_of p) pretokens in
  let rec pairs = function
    | [] -> []
    | [ p ] -> [ (p, None) ]
    | p :: (q :: _ as rest) -> (p, Some q) :: pairs rest
  in
  let pairs = pairs kinds in
  let any f = List.exists f pairs in
  let starts_with k = function Some (k' :: _) -> k' = k | _ -> false in
  let last l = List.nth l (List.length l - 1) in
  let word p = List.filter is_letter p in
  cover "a word after a prefix"
    (any (function
      | (Space | Other | Mark) :: k :: _, _ -> is_letter k
      | _ -> false));
  cover "a word of capitals then lower case"
    (any (fun (p, _) ->
         let w = word p in
         List.mem Upper w && List.mem Lower w));
  cover "capitals giving back to a letter of both classes"
    (any (fun (p, q) ->
         let w = word p in
         w <> []
         && last w = Both
         && (not (List.mem Lower w))
         && starts_with Upper q));
  cover "a word of capitals alone"
    (any (fun (p, _) ->
         let w = word p in
         w <> [] && List.for_all (( = ) Upper) w));
  cover "a word opened by a mark"
    (any (function Mark :: _, _ -> true | _ -> false));
  cover "a group of numbers cut short"
    (any (fun (p, q) ->
         List.length p = digits
         && List.for_all (( = ) Number) p
         && starts_with Number q));
  cover "punctuation after a space"
    (any (function Space :: (Other | Apostrophe) :: _, _ -> true | _ -> false));
  cover "punctuation with newlines after it"
    (any (fun (p, _) ->
         match p with
         | (Other | Apostrophe) :: _ -> last p = Newline
         | _ -> false));
  cover "whitespace through its newlines"
    (any (fun (p, _) -> List.for_all is_space p && List.mem Newline p));
  cover "whitespace leaving its last character"
    (any (fun (p, q) -> List.for_all (( = ) Space) p && starts_with Space q));
  cover "whitespace at the end"
    (any (fun (p, q) -> q = None && p <> [] && List.for_all is_space p));
  cover "bytes of no character" (any (fun (p, _) -> List.mem Stray p))

(* [name]'s pattern as written gives the pretokens it gives in a group, its
   matches isolated and kept. *)
let as_in_a_group name text =
  let { isolated; kept } = List.assoc name (walked ()) in
  List.iter
    (fun { written; grouped } ->
      equal pretokens
        (Pre.pre_tokenize grouped text)
        (Pre.pre_tokenize written text))
    [ isolated; kept ]

let isolated_pretokens name text =
  Pre.pre_tokenize (List.assoc name (walked ())).isolated.written text

(* [as_in_a_group] for Tekken, marking the alternatives its isolated pretokens
   reach. *)
let tekken_law text =
  cover_o200k ~digits:1 (isolated_pretokens "tekken" text);
  as_in_a_group "tekken" text

(* The same for o200k, whose words take a contraction. *)
let o200k_law text =
  let pretokens = isolated_pretokens "o200k" text in
  cover_o200k ~digits:3 pretokens;
  cover "a contraction"
    (List.exists
       (fun (p, _) ->
         let k = kinds_of p in
         List.exists is_letter k && List.mem Apostrophe k
         && is_letter (List.hd k))
       pretokens);
  as_in_a_group "o200k" text

let cl100k_cuts (text, expected) =
  let { written; grouped } = (List.assoc "cl100k" (walked ())).isolated in
  equal ~msg:"as written" pretokens expected (Pre.pre_tokenize written text);
  equal ~msg:"in a group" pretokens expected (Pre.pre_tokenize grouped text)

let tiles text pretokens =
  let stop =
    List.fold_left
      (fun at (piece, ((start, stop) as span)) ->
        equal ~msg:"a pretoken starts where the one before stopped" int at start;
        equal ~msg:"a pretoken is the bytes of its span" string
          (source text span) piece;
        stop)
      0 pretokens
  in
  equal ~msg:"the pretokens reach the end of the text" int (String.length text)
    stop

let named_patterns =
  [ "cl100k"; "cl100k_hf"; "qwen2"; "qwen35"; "o200k"; "tekken"; "kimi" ]

let isolated_named =
  fixture (fun () ->
      List.map (fun name -> (name, isolated (pattern name))) named_patterns)

let isolated_tiles (name, text) =
  tiles text (Pre.pre_tokenize (List.assoc name (isolated_named ())) text)

let test_parity_corpora () =
  List.iter
    (fun corpus ->
      let text = Fixture.read ("support/parity/" ^ corpus ^ ".txt") in
      List.iter
        (fun (name, { isolated; kept }) ->
          List.iter
            (fun { written; grouped } ->
              equal
                ~msg:(Printf.sprintf "%s on %s.txt" name corpus)
                pretokens
                (Pre.pre_tokenize grouped text)
                (Pre.pre_tokenize written text))
            [ isolated; kept ])
        (walked ()))
    [ "sample"; "edge_cases"; "unicode_code" ]

let rejected_rows =
  [
    ("^a", "anchors are not supported in this pattern");
    ("a$", "anchors are not supported in this pattern");
    ("\\Aa", "anchors are not supported in this pattern");
    ("a\\z", "anchors are not supported in this pattern");
    ("\\Ga", "anchors are not supported in this pattern");
    ("\\Z", "anchors are not supported in this pattern");
    ("\\b", "word boundaries are not supported");
    ("\\s+(?!\\S)a", "lookahead is supported only where it ends the pattern");
    ("a(?!bc)", "a negative lookahead is supported over one character only");
    ("(?<=a)b", "lookbehind is not supported");
    ("a++", "possessive quantifiers are not supported");
    ("(?>a)", "atomic groups are not supported");
    ("\\1", "backreferences are not supported");
    ("\\p{Foo}", "unsupported property \\p{Foo}");
    ("(", "unmatched (");
    ("a)", "unmatched )");
    ("a{2,1}", "invalid interval {2,1}");
    ("[z-a]", "empty range in character class");
    ("[a", "unterminated character class");
  ]

let is_rejected (pattern, reason) =
  raises
    (Invalid_argument
       (Printf.sprintf "invalid regular expression %S: %s" pattern reason))
    (fun () -> Pre.split_regex ~pattern ())

(* char_delimiter *)

let delimiters = [ ","; " "; "\u{2581}"; "\u{e9}" ]

let is_split_on_it (c, text) =
  equal pretokens
    (Pre.pre_tokenize (Pre.split ~pattern:c ~behavior:`Removed ()) text)
    (Pre.pre_tokenize (Pre.char_delimiter c) text)

(* digits *)

let numbers = lazy (isolated {|\p{N}+|})
let number = lazy (isolated {|\p{N}|})

let isolates_numbers (individual_digits, text) =
  let reference = if individual_digits then number else numbers in
  equal pretokens
    (Pre.pre_tokenize (Lazy.force reference) text)
    (Pre.pre_tokenize (Pre.digits ~individual_digits ()) text)

(* An other number, a letter number, an Arabic-Indic and a fullwidth digit.
   Every expectation is the output of HuggingFace's [Digits]. *)
let number_rows =
  [
    ("a\u{bd}b", [ ("a", (0, 1)); ("\u{bd}", (1, 3)); ("b", (3, 4)) ]);
    ("a\u{2167}b", [ ("a", (0, 1)); ("\u{2167}", (1, 4)); ("b", (4, 5)) ]);
    ("a\u{661}b", [ ("a", (0, 1)); ("\u{661}", (1, 3)); ("b", (3, 4)) ]);
    ("a\u{ff11}b", [ ("a", (0, 1)); ("\u{ff11}", (1, 4)); ("b", (4, 5)) ]);
  ]

let digit_cuts (text, expected) =
  equal pretokens expected (Pre.pre_tokenize (Pre.digits ()) text)

(* prepend_scheme *)

(* Every expectation is the output of HuggingFace's
   [Metaspace(prepend_scheme="first")]. *)
let first_rows =
  let m = "\u{2581}" and g = "\u{120}" in
  let first = Pre.metaspace ~prepend_scheme:`First () in
  let alone = first in
  let unsplit = Pre.metaspace ~prepend_scheme:`First ~split:false () in
  let after_split = Pre.sequence [ Pre.whitespace_split; first ] in
  let fixed =
    Pre.sequence [ Pre.whitespace_split; Pre.fixed_length 2; first ]
  in
  let then_bytes =
    Pre.sequence
      [ Pre.whitespace_split; first; Pre.byte_level ~use_regex:false () ]
  in
  let twice = Pre.sequence [ first; first ] in
  [
    (alone, "Hello world", [ (m ^ "Hello", (0, 5)); (m ^ "world", (5, 11)) ]);
    (alone, " Hello world", [ (m ^ "Hello", (0, 6)); (m ^ "world", (6, 12)) ]);
    (alone, m ^ "x y", [ (m ^ "x", (0, 4)); (m ^ "y", (4, 6)) ]);
    (alone, "  a", [ (m, (0, 1)); (m ^ "a", (1, 3)) ]);
    (unsplit, "Hello world", [ (m ^ "Hello" ^ m ^ "world", (0, 11)) ]);
    (unsplit, " Hello world", [ (m ^ "Hello" ^ m ^ "world", (0, 12)) ]);
    (after_split, "Hello world", [ (m ^ "Hello", (0, 5)); ("world", (6, 11)) ]);
    (after_split, " Hello world", [ ("Hello", (1, 6)); ("world", (7, 12)) ]);
    (after_split, m ^ "x y", [ (m ^ "x", (0, 4)); ("y", (5, 6)) ]);
    (after_split, "  a", [ ("a", (2, 3)) ]);
    ( fixed,
      "trailing ",
      [ (m ^ "tr", (0, 2)); ("ai", (2, 4)); ("li", (4, 6)); ("ng", (6, 8)) ] );
    (fixed, " ab cd", [ ("ab", (1, 3)); ("cd", (4, 6)) ]);
    (fixed, "abc", [ (m ^ "ab", (0, 2)); ("c", (2, 3)) ]);
    ( then_bytes,
      "  two  spaces",
      [ (g ^ "two", (2, 5)); (g ^ "spaces", (7, 13)) ] );
    ( then_bytes,
      "two spaces",
      [ (g ^ "\u{e2}\u{138}\u{123}two", (0, 3)); (g ^ "spaces", (4, 10)) ] );
    (twice, "a b", [ (m ^ "a", (0, 1)); (m ^ "b", (1, 3)) ]);
    (twice, " a b", [ (m ^ "a", (0, 2)); (m ^ "b", (2, 4)) ]);
  ]

(* Every expectation is the output of HuggingFace. *)
let test_first_in_a_document () =
  let m = "\u{2581}" in
  let vocab =
    List.mapi
      (fun id word -> (word, id))
      [ "[UNK]"; m ^ "a"; m ^ "b"; "a"; "b"; m ^ "ab"; "ab" ]
  in
  let first = Pre.metaspace ~prepend_scheme:`First () in
  let split_first = Pre.sequence [ Pre.whitespace_split; first ] in
  let fixed =
    Pre.sequence [ Pre.whitespace_split; Pre.fixed_length 2; first ]
  in
  let tokenizer ?normalizer ?(added_tokens = []) pre =
    Brot.word_level ?normalizer ~pre ~added_tokens ~vocab ()
  in
  let case t text tokens offsets =
    subtest (Printf.sprintf "%S" text) (fun () ->
        let encoding = Brot.encode t ~add_special_tokens:false text in
        equal ~msg:"tokens" (list string) tokens
          (Array.to_list (Brot.Encoding.tokens encoding));
        equal ~msg:"offsets"
          (list (pair int int))
          offsets
          (Array.to_list (Brot.Encoding.offsets encoding));
        equal ~msg:"encode_ids" (array int)
          (Brot.Encoding.ids encoding)
          (Brot.encode_ids t ~add_special_tokens:false text))
  in
  let s = Brot.added_token "<s>" in
  let added = tokenizer ~added_tokens:[ s ] split_first in
  case added "a b" [ m ^ "a"; "b" ] [ (0, 1); (2, 3) ];
  case added " a b" [ "a"; "b" ] [ (1, 2); (3, 4) ];
  case added "a<s>b" [ m ^ "a"; "<s>"; "b" ] [ (0, 1); (1, 4); (4, 5) ];
  case added "<s>b" [ "<s>"; "b" ] [ (0, 3); (3, 4) ];
  case added " a<s>b" [ "a"; "<s>"; "b" ] [ (1, 2); (2, 5); (5, 6) ];
  case added "a <s> b" [ m ^ "a"; "<s>"; "b" ] [ (0, 1); (2, 5); (6, 7) ];
  let alone = tokenizer ~added_tokens:[ s ] first in
  case alone "a<s>b" [ m ^ "a"; "<s>"; "b" ] [ (0, 1); (1, 4); (4, 5) ];
  case alone "<s>b" [ "<s>"; "b" ] [ (0, 3); (3, 4) ];
  let stripped = tokenizer ~normalizer:(Brot.Normalizer.strip ()) split_first in
  case stripped "  a b" [ "a"; "b" ] [ (2, 3); (4, 5) ];
  case stripped "a b" [ m ^ "a"; "b" ] [ (0, 1); (2, 3) ];
  let lowered = tokenizer ~normalizer:Brot.Normalizer.lowercase split_first in
  case lowered "A b" [ m ^ "a"; "b" ] [ (0, 1); (2, 3) ];
  (* An added token matched on the normalized text opens the document just as
     one matched on the raw text does. *)
  let normalized =
    tokenizer
      ~added_tokens:[ Brot.added_token ~normalized:true ~special:false "[X]" ]
      first
  in
  case normalized "[X]b" [ "[X]"; "b" ] [ (0, 3); (3, 4) ];
  case normalized "b[X]a" [ m ^ "b"; "[X]"; "a" ] [ (0, 1); (1, 4); (4, 5) ];
  let pieces = tokenizer ~added_tokens:[ s ] fixed in
  case pieces "ab a" [ m ^ "ab"; "a" ] [ (0, 2); (3, 4) ];
  case pieces "ab<s>ab" [ m ^ "ab"; "<s>"; "ab" ] [ (0, 2); (2, 5); (5, 7) ];
  case pieces "<s>ab" [ "<s>"; "ab" ] [ (0, 3); (3, 5) ];
  case pieces " ab" [ "ab" ] [ (1, 3) ];
  let stripped_pieces =
    tokenizer ~normalizer:(Brot.Normalizer.strip ()) fixed
  in
  case stripped_pieces "  ab a" [ "ab"; "a" ] [ (2, 4); (5, 6) ];
  case stripped_pieces "ab a" [ m ^ "ab"; "a" ] [ (0, 2); (3, 4) ]

(* metaspace *)

let scheme_name = function
  | `Always -> "`Always"
  | `First -> "`First"
  | `Never -> "`Never"

let scheme =
  Gen.of_list
    ~pp:(fun ppf s -> Format.pp_print_string ppf (scheme_name s))
    [ `Always; `First; `Never ]

let markers = [ "\u{2581}"; "_" ]

(* The text with each space replaced by [marker], and [marker] prepended unless
   [prepend_scheme] is [`Never] or the result opens with one. *)
let marked ~marker ~prepend_scheme text =
  let replaced = String.concat marker (String.split_on_char ' ' text) in
  if prepend_scheme = `Never || String.starts_with ~prefix:marker replaced then
    replaced
  else marker ^ replaced

let marks_and_prepends (marker, prepend_scheme, text) =
  let expected =
    if text = "" then []
    else [ (marked ~marker ~prepend_scheme text, (0, String.length text)) ]
  in
  equal pretokens expected
    (Pre.pre_tokenize
       (Pre.metaspace ~replacement:marker ~prepend_scheme ~split:false ())
       text)

let cuts_before_markers (marker, prepend_scheme, text) =
  let whole = marked ~marker ~prepend_scheme text in
  let expected =
    if text = "" then []
    else
      List.map (source whole)
        (behave `Merged_with_next (occurrences ~pattern:marker whole))
  in
  equal (list string) expected
    (List.map fst
       (Pre.pre_tokenize
          (Pre.metaspace ~replacement:marker ~prepend_scheme ())
          text))

(* [piece] with each [marker] read back as a space. *)
let unmarked ~marker piece =
  String.concat ""
    (List.map
       (fun (start, stop, is_marker) ->
         if is_marker then " " else source piece (start, stop))
       (occurrences ~pattern:marker piece))

(* Texts without the marker, so that each marker in a pretoken stands for a
   space or is the one prepended. *)
let unmarked_text = text_of (List.filter (fun c -> c <> "\u{2581}") characters)

let spans_its_source (prepend_scheme, text) =
  let marker = "\u{2581}" in
  let prepended = prepend_scheme <> `Never && text <> "" && text.[0] <> ' ' in
  List.iteri
    (fun i (piece, span) ->
      let expected =
        if prepended && i = 0 then " " ^ source text span else source text span
      in
      equal string expected (unmarked ~marker piece))
    (Pre.pre_tokenize (Pre.metaspace ~prepend_scheme ()) text)

(* Every expectation is the output of HuggingFace's [Metaspace]. *)
let metaspace_rows =
  let m = "\u{2581}" in
  let split = Pre.metaspace () in
  let never = Pre.metaspace ~prepend_scheme:`Never () in
  let whole = Pre.metaspace ~split:false () in
  let underscore = Pre.metaspace ~replacement:"_" () in
  [
    (split, "", []);
    (split, " ", [ (m, (0, 1)) ]);
    (split, "a", [ (m ^ "a", (0, 1)) ]);
    (split, "  a", [ (m, (0, 1)); (m ^ "a", (1, 3)) ]);
    (split, m ^ "a", [ (m ^ "a", (0, 4)) ]);
    (split, m ^ " a", [ (m, (0, 3)); (m ^ "a", (3, 5)) ]);
    ( split,
      m ^ "hello" ^ m ^ "world",
      [ (m ^ "hello", (0, 8)); (m ^ "world", (8, 16)) ] );
    (split, "Hello world", [ (m ^ "Hello", (0, 5)); (m ^ "world", (5, 11)) ]);
    (split, "trailing ", [ (m ^ "trailing", (0, 8)); (m, (8, 9)) ]);
    ( split,
      "  two  spaces",
      [ (m, (0, 1)); (m ^ "two", (1, 5)); (m, (5, 6)); (m ^ "spaces", (6, 13)) ]
    );
    ( split,
      "\u{65e5}\u{672c} \u{8a9e}",
      [ (m ^ "\u{65e5}\u{672c}", (0, 6)); (m ^ "\u{8a9e}", (6, 10)) ] );
    (* A no-break space is not a space. *)
    (split, "a\u{a0}b", [ (m ^ "a\u{a0}b", (0, 4)) ]);
    (never, "Hello world", [ ("Hello", (0, 5)); (m ^ "world", (5, 11)) ]);
    (never, "trailing ", [ ("trailing", (0, 8)); (m, (8, 9)) ]);
    (never, "a\u{a0}b", [ ("a\u{a0}b", (0, 4)) ]);
    (whole, "Hello world", [ (m ^ "Hello" ^ m ^ "world", (0, 11)) ]);
    (whole, "trailing ", [ (m ^ "trailing" ^ m, (0, 9)) ]);
    (whole, "  two  spaces", [ (m ^ m ^ "two" ^ m ^ m ^ "spaces", (0, 13)) ]);
    (whole, " ", [ (m, (0, 1)) ]);
    (underscore, "Hello world", [ ("_Hello", (0, 5)); ("_world", (5, 11)) ]);
    (underscore, "_a b", [ ("_a", (0, 2)); ("_b", (2, 4)) ]);
  ]

(* A member after the metaspace cuts its pretokens and places its own through
   the marking; members before it hand it the segments it marks. Every
   expectation is the output of HuggingFace. *)
let later_member_rows =
  let m = "\u{2581}" in
  let t5 = Pre.sequence [ Pre.whitespace_split; Pre.metaspace () ] in
  let punctuated =
    Pre.sequence [ Pre.whitespace_split; Pre.metaspace (); Pre.punctuation () ]
  in
  let punctuation_first =
    Pre.sequence [ Pre.punctuation (); Pre.metaspace () ]
  in
  let unsplit =
    Pre.sequence [ Pre.whitespace_split; Pre.metaspace ~split:false () ]
  in
  let digits =
    Pre.sequence [ Pre.digits ~individual_digits:true (); Pre.metaspace () ]
  in
  let scripts = Pre.sequence [ Pre.unicode_scripts; Pre.metaspace () ] in
  let nested =
    Pre.sequence
      [
        Pre.sequence [ Pre.whitespace_split; Pre.punctuation () ];
        Pre.metaspace ();
      ]
  in
  let nested_inner =
    Pre.sequence
      [
        Pre.whitespace_split;
        Pre.sequence [ Pre.metaspace (); Pre.punctuation () ];
      ]
  in
  [
    (* The whitespace split drops the run between two pretokens, so a marker
       never stands for it. T5 carries this one. *)
    (t5, "Hello world", [ (m ^ "Hello", (0, 5)); (m ^ "world", (6, 11)) ]);
    (t5, "trailing ", [ (m ^ "trailing", (0, 8)) ]);
    (t5, "  two  spaces", [ (m ^ "two", (2, 5)); (m ^ "spaces", (7, 13)) ]);
    (t5, " ", []);
    ( t5,
      "\u{65e5}\u{672c} \u{8a9e}",
      [ (m ^ "\u{65e5}\u{672c}", (0, 6)); (m ^ "\u{8a9e}", (7, 10)) ] );
    (punctuated, "a b!", [ (m ^ "a", (0, 1)); (m ^ "b", (2, 3)); ("!", (3, 4)) ]);
    ( punctuated,
      "a,b c",
      [ (m ^ "a", (0, 1)); (",", (1, 2)); ("b", (2, 3)); (m ^ "c", (4, 5)) ] );
    (punctuated, "!!", [ (m, (0, 1)); ("!", (0, 1)); ("!", (1, 2)) ]);
    (* The segments a punctuation split hands to the metaspace can hold spaces,
       which the marker then stands at, and a marker prepended to a lone
       punctuation mark stands at that mark. *)
    ( punctuation_first,
      "a b!",
      [ (m ^ "a", (0, 1)); (m ^ "b", (1, 3)); (m ^ "!", (3, 4)) ] );
    ( punctuation_first,
      "a, b",
      [ (m ^ "a", (0, 1)); (m ^ ",", (1, 2)); (m ^ "b", (2, 4)) ] );
    (unsplit, "a b!c", [ (m ^ "a", (0, 1)); (m ^ "b!c", (2, 5)) ]);
    (unsplit, "  x", [ (m ^ "x", (2, 3)) ]);
    (digits, "a1 b", [ (m ^ "a", (0, 1)); (m ^ "1", (1, 2)); (m ^ "b", (2, 4)) ]);
    (* A leading run that belongs to no pretoken belongs to no segment. *)
    (scripts, " ab", [ (m ^ "ab", (1, 3)) ]);
    ( scripts,
      "ab\u{433}\u{434}",
      [ (m ^ "ab", (0, 2)); (m ^ "\u{433}\u{434}", (2, 6)) ] );
    ( nested,
      "a b!c",
      [
        (m ^ "a", (0, 1));
        (m ^ "b", (2, 3));
        (m ^ "!", (3, 4));
        (m ^ "c", (4, 5));
      ] );
    ( nested_inner,
      "a b!c",
      [ (m ^ "a", (0, 1)); (m ^ "b", (2, 3)); ("!", (3, 4)); ("c", (4, 5)) ] );
  ]

(* A marker a later member cuts into a pretoken of its own stands at the
   character it opens: the marker that replaced a space at that space, a
   prepended one at the first character. Every expectation is the output of
   HuggingFace. *)
let lone_marker_rows =
  let m = "\u{2581}" in
  let marked_then_punctuation replacement =
    Pre.sequence [ Pre.metaspace ~replacement (); Pre.punctuation () ]
  in
  let underscore = marked_then_punctuation "_" in
  let block = marked_then_punctuation m in
  [
    (underscore, "a", [ ("_", (0, 1)); ("a", (0, 1)) ]);
    ( underscore,
      "Hello world",
      [ ("_", (0, 1)); ("Hello", (0, 5)); ("_", (5, 6)); ("world", (6, 11)) ] );
    ( underscore,
      "!!!",
      [ ("_", (0, 1)); ("!", (0, 1)); ("!", (1, 2)); ("!", (2, 3)) ] );
    ( underscore,
      "_a_b",
      [ ("_", (0, 1)); ("a", (1, 2)); ("_", (2, 3)); ("b", (3, 4)) ] );
    (block, "a", [ (m ^ "a", (0, 1)) ]);
    (block, "!!!", [ (m, (0, 1)); ("!", (0, 1)); ("!", (1, 2)); ("!", (2, 3)) ]);
    ( block,
      "_a_b",
      [
        (m, (0, 1)); ("_", (0, 1)); ("a", (1, 2)); ("_", (2, 3)); ("b", (3, 4));
      ] );
  ]

(* Every expectation is the output of HuggingFace. *)
let test_unsplit_token_offsets () =
  let m = "\u{2581}" in
  let vocab =
    [
      ("<unk>", 0.0); (m, -3.0); (m ^ "Hello", -1.0); (m ^ "world", -1.0);
      (m ^ "leading", -1.0); (m ^ "trailing", -1.0); (m ^ "multi", -1.0);
      (m ^ "space", -1.0); (m ^ "\u{65e5}\u{672c}", -1.0);
      (m ^ "\u{8a9e}", -1.0); (m ^ "already", -1.0); (m ^ "a", -1.0);
      (m ^ "b", -1.0); (m ^ "x", -1.0); ("Hello", -2.0); ("world", -2.0);
      ("trailing", -2.0); ("multi", -2.0); ("a", -4.0); ("b", -4.0);
      ("\u{65e5}\u{672c}", -2.0); ("\u{8a9e}", -2.0); ("already", -2.0);
      ("y", -4.0); ("x", -4.0); ("\u{a0}", -4.0);
    ]
    [@@ocamlformat "disable"]
  in
  let tokenizer prepend_scheme =
    Brot.unigram ~vocab ~unk_id:0 ~unk_token:"<unk>"
      ~pre:(Pre.metaspace ~prepend_scheme ~split:false ())
      ()
  in
  let case t text ids offsets =
    subtest (Printf.sprintf "%S" text) (fun () ->
        let encoding = Brot.encode t text in
        equal ~msg:"ids" (list int) ids
          (Array.to_list (Brot.Encoding.ids encoding));
        equal ~msg:"offsets"
          (list (pair int int))
          offsets
          (Array.to_list (Brot.Encoding.offsets encoding)))
  in
  let always = tokenizer `Always in
  case always "Hello world" [ 2; 3 ] [ (0, 5); (5, 11) ];
  case always "  leading" [ 1; 4 ] [ (0, 1); (1, 9) ];
  case always "trailing " [ 5; 1 ] [ (0, 8); (8, 9) ];
  case always "" [] [];
  case always " " [ 1 ] [ (0, 1) ];
  case always "  " [ 1; 1 ] [ (0, 1); (1, 2) ];
  case always "a\u{a0}b" [ 11; 25; 19 ] [ (0, 1); (1, 3); (3, 4) ];
  case always "multi  space" [ 6; 1; 7 ] [ (0, 5); (5, 6); (6, 12) ];
  case always (m ^ "already") [ 10 ] [ (0, 10) ];
  case always "\u{65e5}\u{672c} \u{8a9e}" [ 8; 9 ] [ (0, 6); (6, 10) ];
  case always "x y" [ 13; 1; 23 ] [ (0, 1); (1, 2); (2, 3) ];
  let never = tokenizer `Never in
  case never "Hello world" [ 14; 3 ] [ (0, 5); (5, 11) ];
  case never "  leading" [ 1; 4 ] [ (0, 1); (1, 9) ];
  case never "trailing " [ 16; 1 ] [ (0, 8); (8, 9) ];
  case never "" [] [];
  case never " " [ 1 ] [ (0, 1) ];
  case never "  " [ 1; 1 ] [ (0, 1); (1, 2) ];
  case never "a\u{a0}b" [ 18; 25; 19 ] [ (0, 1); (1, 3); (3, 4) ];
  case never "multi  space" [ 17; 1; 7 ] [ (0, 5); (5, 6); (6, 12) ];
  case never (m ^ "already") [ 10 ] [ (0, 10) ];
  case never "\u{65e5}\u{672c} \u{8a9e}" [ 20; 9 ] [ (0, 6); (6, 10) ];
  case never "x y" [ 24; 1; 23 ] [ (0, 1); (1, 2); (2, 3) ]

(* unicode_scripts *)

let matches_huggingface_scripts () =
  List.iter
    (fun (text, expected) ->
      subtest (Printf.sprintf "%S" text) (fun () ->
          equal pretokens expected (Pre.pre_tokenize Pre.unicode_scripts text)))
    (scripts_golden ())

(* The text, then Hiragana and Katakana after Han, then the prolonged sound mark
   between Han and Hiragana: one pretoken each, as brot.mli states. *)
let han_rows =
  [
    "\u{65e5}\u{672c}\u{8a9e}\u{30c6}\u{30ad}\u{30b9}\u{30c8}";
    "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}\u{4e16}\u{754c}";
    "\u{4e2d}\u{30fc}\u{3042}";
  ]

let is_one_pretoken text =
  equal pretokens
    [ (text, (0, String.length text)) ]
    (Pre.pre_tokenize Pre.unicode_scripts text)

(* Spaces, a private use character, which has no known script, and characters of
   several scripts. *)
let script_text =
  text_of
    [ " "; " "; "\u{e000}"; "a"; "Z"; "\u{41f}"; "\u{4e2d}"; "\u{3042}"; "1";
      "!"; "\t"; "\u{301}" ]
[@@ocamlformat "disable"]

let drops_the_leading_run text =
  let rec run i =
    if i < String.length text && text.[i] = ' ' then run (i + 1)
    else if
      String.starts_with ~prefix:"\u{e000}"
        (source text (i, String.length text))
    then run (i + 3)
    else i
  in
  let start = run 0 in
  cover "a leading run before a character"
    (start > 0 && start < String.length text);
  equal string
    (source text (start, String.length text))
    (String.concat ""
       (List.map fst (Pre.pre_tokenize Pre.unicode_scripts text)))

(* fixed_length *)

let character_spans text =
  let rec from i spans =
    if i >= String.length text then List.rev spans
    else
      let n = Uchar.utf_decode_length (String.get_utf_8_uchar text i) in
      from (i + n) ((i, i + n) :: spans)
  in
  from 0 []

(* The spans of runs of [n] characters, the last one shorter. *)
let chunks n spans =
  let rec go k start stop = function
    | [] -> if k = 0 then [] else [ (start, stop) ]
    | (s, e) :: rest ->
        let start = if k = 0 then s else start in
        if k + 1 = n then (start, e) :: go 0 e e rest
        else go (k + 1) start e rest
  in
  go 0 0 0 spans

let cuts_n_characters (n, text) =
  let expected =
    List.map
      (fun span -> (source text span, span))
      (chunks n (character_spans text))
  in
  equal pretokens expected (Pre.pre_tokenize (Pre.fixed_length n) text)

(* sequence *)

(* The pretokens of [second] in each pretoken of [first], placed in it. *)
let composed first second text =
  List.concat_map
    (fun (piece, (start, _)) ->
      List.map
        (fun (p, (s, e)) -> (p, (start + s, start + e)))
        (Pre.pre_tokenize second piece))
    (Pre.pre_tokenize first text)

let cuts_each_pretoken (first, second, text) =
  equal pretokens
    (composed first second text)
    (Pre.pre_tokenize (Pre.sequence [ first; second ]) text)

(* A text whose one whitespace-split pretoken a punctuation member cuts into
   400. *)
let dense =
  String.concat "" (List.init 400 (fun i -> if i land 1 = 0 then "a" else "."))

(* The pretokens alone: where a member cuts a pretoken an earlier one rewrote,
   brot.mli lets the spans be coarser in one form than in another. *)
let nests (a, b, c, text) =
  let pieces t = List.map fst (Pre.pre_tokenize t text) in
  let flat = pieces (Pre.sequence [ a; b; c ]) in
  equal ~msg:"sequence [a; sequence [b; c]]" (list string) flat
    (pieces (Pre.sequence [ a; Pre.sequence [ b; c ] ]));
  equal ~msg:"sequence [sequence [a; b]; c]" (list string) flat
    (pieces (Pre.sequence [ Pre.sequence [ a; b ]; c ]))

(* The lookahead holds at the end of a pretoken whatever follows it in the
   text. *)
let test_member_sees_a_whole_text () =
  let t =
    Pre.sequence
      [
        Pre.split ~pattern:"|" ~behavior:`Removed ();
        Pre.split_regex ~pattern:{re|\s+(?!\S)|\s+|\S+|re} ~behavior:`Isolated
          ();
      ]
  in
  equal pretokens
    [
      ("a", (0, 1));
      ("  ", (1, 3));
      ("b", (4, 5));
      (" ", (5, 6));
      (" ", (6, 7));
      ("c", (7, 8));
    ]
    (Pre.pre_tokenize t "a  |b  c")

(* Every expectation is the output of HuggingFace's [Sequence([])]. *)
let empty_sequence_rows =
  [
    (Pre.sequence [], "a b", [ ("a b", (0, 3)) ]);
    (Pre.sequence [], "", [ ("", (0, 0)) ]);
  ]

(* pre_tokenize *)

let every = verbatim @ rewriting

let ascends_within (t, text) =
  let len = String.length text in
  cover "a text that is not valid UTF-8" (not (String.is_valid_utf_8 text));
  ignore
    (List.fold_left
       (fun previous (_, (start, stop)) ->
         at_least ~msg:"a span starts where or after the one before" int
           ~than:previous start;
         at_most ~msg:"a span starts before it stops" int ~than:stop start;
         at_most ~msg:"a span stops within the text" int ~than:len stop;
         start)
       0 (Pre.pre_tokenize t text)
      : int)

(* Sequences RFC 3629 forbids: overlong forms of a space (three and four bytes),
   of "!" and of "A", a surrogate, a value above U+10FFFF and an F5 lead. *)
let forbidden =
  [ "\xe0\x80\xa0"; "\xf0\x80\x80\xa0"; "\xe0\x80\xa1"; "\xe0\x81\x81";
    "\xed\xa0\x80"; "\xf4\x90\x80\x80"; "\xf5\x80\x80\x80" ]
[@@ocamlformat "disable"]

(* The text with each forbidden sequence replaced by as many lone continuation
   bytes. *)
let as_strays fragments =
  String.concat ""
    (List.map
       (fun f ->
         if List.mem f forbidden then String.make (String.length f) '\x80'
         else f)
       fragments)

(* The pre-tokenizers that cut the text's characters. A member after
   [byte_level] cuts the bytes' alphabet instead, where a byte is a character of
   its own value. *)
let reading_characters =
  verbatim
  @ [
      Pre.byte_level ();
      Pre.metaspace ();
      Pre.sequence [ Pre.whitespace_split; Pre.metaspace () ];
      Pre.sequence [ Pre.fixed_length 3; Pre.metaspace () ];
    ]

let reads_forbidden_as_strays ((t, fragments) : Pre.t * string list) =
  cover "a forbidden sequence"
    (List.exists (fun f -> List.mem f forbidden) fragments);
  equal
    (list (pair int int))
    (List.map snd (Pre.pre_tokenize t (as_strays fragments)))
    (List.map snd (Pre.pre_tokenize t (String.concat "" fragments)))

let is_its_span (t, text) =
  List.iter
    (fun (piece, span) -> equal string (source text span) piece)
    (Pre.pre_tokenize t text)

let gives_none t = equal pretokens [] (Pre.pre_tokenize t "")

(* A member's spans lie within those of the member before it, whatever that one
   rewrote. *)
let lies_within (first, second, text) =
  let outer = List.map snd (Pre.pre_tokenize first text) in
  let within (start, stop) =
    List.exists (fun (s, e) -> s <= start && stop <= e) outer
  in
  List.iter
    (fun (_, span) ->
      satisfies ~claim:"a span within one of the first member's" (pair int int)
        within span)
    (Pre.pre_tokenize (Pre.sequence [ first; second ]) text)

(* Baseline. HuggingFace places "," at (3, 4) and "c" at (4, 5) in each. *)
let test_whole_spans () =
  List.iter
    (fun t ->
      Format.printf "%a:@." Pre.pp t;
      List.iter
        (fun (piece, (start, stop)) ->
          Format.printf "  %S (%d, %d)@." piece start stop)
        (Pre.pre_tokenize t "a b,c"))
    [
      Pre.sequence [ Pre.byte_level ~use_regex:false (); Pre.punctuation () ];
      Pre.sequence
        [
          Pre.metaspace ~split:false ();
          Pre.metaspace ~split:false ();
          Pre.punctuation ();
        ];
      Pre.sequence [ Pre.fixed_length 3; Pre.metaspace (); Pre.punctuation () ];
    ];
  expect (output ())
  @@ __POS_OF__
       {|
    Sequence[ByteLevel(add_prefix_space=true, use_regex=false, trim_offsets=true),
     Punctuation(Isolated)]:
      "\196\160a\196\160b" (0, 5)
      "," (0, 5)
      "c" (0, 5)
    Sequence[Metaspace("\226\150\129", always, split=false),
     Metaspace("\226\150\129", always, split=false), Punctuation(Isolated)]:
      "\226\150\129a\226\150\129b" (0, 5)
      "," (0, 5)
      "c" (0, 5)
    Sequence[FixedLength(3), Metaspace("\226\150\129", always, split=true),
     Punctuation(Isolated)]:
      "\226\150\129a" (0, 1)
      "\226\150\129b" (1, 3)
      "\226\150\129" (3, 5)
      "," (3, 5)
      "c" (3, 5)
  |}

(* byte_level_decode *)

(* Every expectation is what HuggingFace's [CHAR_BYTES] lookup gives. *)
let alphabet_rows =
  [
    ("", "");
    ("\u{120}Hello", " Hello");
    ("\u{100}", "\x00");
    ("\u{f0}\u{141}\u{133}\u{12f}", "\xf0\x9f\x91\x8d");
    ("\u{e9}", "\xe9");
    ("\u{a1}\u{ac}\u{ae}\u{ff}", "\xa1\xac\xae\xff");
  ]

(* A character above the alphabet, a space and bytes that are no character each
   leave the whole string as it is. *)
let outside_rows =
  [ "\u{65e5}"; "\u{120}\u{65e5}"; "a b"; "\u{120}a\xc3"; "\xc4" ]

let encoded text =
  String.concat ""
    (List.map fst
       (Pre.pre_tokenize
          (Pre.byte_level ~add_prefix_space:false ~use_regex:false ())
          text))

(* to_json and of_json *)

(* JSON text with the members of every object sorted, so that two documents
   compare as values. *)
let json_text json =
  let rec sorted = function
    | Jsont.Object (members, meta) ->
        let members =
          List.map (fun (name, value) -> (name, sorted value)) members
        in
        Jsont.Object
          ( List.sort (fun ((a, _), _) ((b, _), _) -> String.compare a b) members,
            meta )
    | Jsont.Array (items, meta) -> Jsont.Array (List.map sorted items, meta)
    | json -> json
  in
  match
    Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json (sorted json)
  with
  | Ok text -> text
  | Error e -> fail e

let parse text =
  match Jsont_bytesrw.decode_string Jsont.json text with
  | Ok json -> json
  | Error e -> fail e

(* Every JSON text is what HuggingFace tokenizers 0.23.1 writes for the same
   pre-tokenizer. *)
let huggingface_rows =
  [
    ( Pre.byte_level (),
      {|{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}|}
    );
    ( Pre.byte_level ~add_prefix_space:false ~use_regex:false
        ~trim_offsets:false (),
      {|{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}|}
    );
    (Pre.whitespace, {|{"type":"Whitespace"}|});
    (Pre.whitespace_split, {|{"type":"WhitespaceSplit"}|});
    (Pre.bert, {|{"type":"BertPreTokenizer"}|});
    ( Pre.punctuation ~behavior:`Merged_with_previous (),
      {|{"type":"Punctuation","behavior":"MergedWithPrevious"}|} );
    ( Pre.split ~pattern:"," (),
      {|{"type":"Split","pattern":{"String":","},"behavior":"Removed","invert":false}|}
    );
    ( Pre.split ~pattern:"ab" ~behavior:`Isolated ~invert:true (),
      {|{"type":"Split","pattern":{"String":"ab"},"behavior":"Isolated","invert":true}|}
    );
    ( Pre.split_regex ~pattern:{|\d+|} ~behavior:`Merged_with_next (),
      {|{"type":"Split","pattern":{"Regex":"\\d+"},"behavior":"MergedWithNext","invert":false}|}
    );
    (Pre.char_delimiter ",", {|{"type":"CharDelimiterSplit","delimiter":","}|});
    ( Pre.digits ~individual_digits:true (),
      {|{"type":"Digits","individual_digits":true}|} );
    ( Pre.metaspace (),
      {|{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}|}
    );
    ( Pre.metaspace ~replacement:"_" ~prepend_scheme:`Never ~split:false (),
      {|{"type":"Metaspace","replacement":"_","prepend_scheme":"never","split":false}|}
    );
    ( Pre.metaspace ~prepend_scheme:`First (),
      {|{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true}|}
    );
    (Pre.unicode_scripts, {|{"type":"UnicodeScripts"}|});
    (Pre.fixed_length 4, {|{"type":"FixedLength","length":4}|});
    ( Pre.sequence [ Pre.whitespace_split; Pre.punctuation () ],
      {|{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Punctuation","behavior":"Isolated"}]}|}
    );
  ]

let json_id (t, _) = pre_id t

let writes_huggingface (t, expected) =
  equal string (json_text (parse expected)) (json_text (Pre.to_json t))

(* A text every pre-tokenizer above cuts. *)
let sample = "Hello, world! 123 \u{4e2d}\u{6587}  x\u{2581}y"

let reads_back (t, _) =
  let read = require_ok (Pre.of_json (Pre.to_json t)) in
  equal ~msg:"JSON" string
    (json_text (Pre.to_json t))
    (json_text (Pre.to_json read));
  equal ~msg:"pretokens" pretokens
    (Pre.pre_tokenize t sample)
    (Pre.pre_tokenize read sample)

(* A Split on Kimi K2's pattern, which intersects classes and names a script. *)
let test_reads_kimi () =
  let t = Pre.split_regex ~pattern:(pattern "kimi") ~behavior:`Isolated () in
  let read = require_ok (Pre.of_json (Pre.to_json t)) in
  equal pretokens (Pre.pre_tokenize t sample) (Pre.pre_tokenize read sample)

let default_rows =
  [
    ({|{"type":"Metaspace","replacement":"▁"}|}, Pre.metaspace ());
    ({|{"type":"Punctuation"}|}, Pre.punctuation ());
    ( {|{"type":"Split","pattern":{"String":"-"},"behavior":"Isolated"}|},
      Pre.split ~pattern:"-" ~behavior:`Isolated () );
    ({|{"type":"Digits"}|}, Pre.digits ());
    ({|{"type":"ByteLevel"}|}, Pre.byte_level ());
  ]

let takes_defaults (text, expected) =
  let read = require_ok (Pre.of_json (parse text)) in
  equal string (json_text (Pre.to_json expected)) (json_text (Pre.to_json read))

let error_rows =
  [
    ("not an object", {|["Whitespace"]|});
    ("no type", {|{}|});
    ("an unknown type", {|{"type":"Bogus"}|});
    ( "a Split whose pattern is a bare string",
      {|{"type":"Split","pattern":",","behavior":"Removed","invert":false}|} );
    ( "a Split whose regex uses lookbehind",
      {|{"type":"Split","pattern":{"Regex":"(?<=a)b"},"behavior":"Removed","invert":false}|}
    );
    ( "a Split of an unknown behavior",
      {|{"type":"Split","pattern":{"String":","},"behavior":"Bogus","invert":false}|}
    );
    ( "a Metaspace with a capitalised scheme",
      {|{"type":"Metaspace","replacement":"_","prepend_scheme":"Always"}|} );
    ("a Metaspace without a replacement", {|{"type":"Metaspace"}|});
    ( "a Metaspace of two characters",
      {|{"type":"Metaspace","replacement":"__"}|} );
    ( "a CharDelimiterSplit of two characters",
      {|{"type":"CharDelimiterSplit","delimiter":"ab"}|} );
    ("a FixedLength of length 0", {|{"type":"FixedLength","length":0}|});
    ("a Sequence without members", {|{"type":"Sequence"}|});
    ( "a Sequence with an unknown member",
      {|{"type":"Sequence","pretokenizers":[{"type":"Bogus"}]}|} );
  ]

let () =
  exit
    (run "Pre_tokenizer"
       [
         group "whitespace"
           [
             prop ~count:500 "keeps split_regex's matches of \\w+|[^\\w\\s]+"
               word_text is_its_pattern;
             cases ~id:text_id
               "cuts runs of word characters and runs of other characters, \
                dropping white space"
               whitespace_rows whitespace_cuts;
             cases ~id:text_id
               "takes marks, letter numbers, connector punctuation and join \
                controls into words"
               word_character_rows whitespace_cuts;
           ];
         group "whitespace_split"
           [
             prop ~count:500 "is split_regex removing \\s+" bytes
               splits_on_white_space;
             cases ~id:text_id "splits on the White_Space property"
               white_space_rows white_space_split;
           ];
         group "bert"
           [
             prop ~count:1000
               ~examples:
                 [
                   "";
                   "   ";
                   "==";
                   "<|endoftext|>";
                   "#!/bin/sh";
                   "\u{4e2d}\u{6587}ab";
                 ]
               "is whitespace_split then punctuation" bytes
               is_whitespace_split_then_punctuation;
           ];
         group "byte_level"
           [
             cases ~id:text_id "cuts the text with GPT-2's pattern" gpt2_rows
               gpt2_cuts;
             test "gives a byte of no character a pretoken of its own"
               (fun () ->
                 equal pretokens
                   [
                     ("\u{109}", (0, 1));
                     ("$", (1, 2));
                     ("\u{c3}", (2, 3));
                     ("\u{10a}", (3, 4));
                   ]
                   (Pre.pre_tokenize
                      (Pre.byte_level ~add_prefix_space:false ())
                      "\t$\xc3\n"));
             prop ~count:2000
               ~examples:[ long_text; "\t$\xc3\n"; "$\x80$"; " \x80" ]
               "is split_regex on GPT-2's pattern then byte_level without it"
               split_text cuts_as_gpt2;
             prop ~count:1000
               ~examples:(List.map (fun s -> (true, s)) malformed)
               "spells the text in pretokens whose spans tile it"
               (Gen.pair Gen.bool Gen.string)
               spells_its_spans;
             cases ~id:text_id
               "prepends a space to a text that does not start with one, at \
                its first character"
               prefix_rows prefixed;
             test "without the regex, gives the text as one pretoken" (fun () ->
                 equal pretokens
                   [ ("a\u{120}b", (0, 3)) ]
                   (Pre.pre_tokenize
                      (Pre.byte_level ~add_prefix_space:false ~use_regex:false
                         ())
                      "a b"));
             prop ~count:200 "changes no pretoken with trim_offsets"
               (Gen.triple Gen.bool Gen.bool bytes)
               ignores_trim_offsets;
             cases ~id:cut_id
               "in a sequence, prepends its space to each pretoken it is handed"
               byte_level_in_a_sequence_rows cuts;
             test "in a sequence, hands the model each pretoken encoded once"
               test_byte_level_encodes_once;
           ];
         group "behavior"
           [
             cases ~id:cut_id
               "keeps, drops or merges each delimiter as HuggingFace does"
               behavior_rows cuts;
           ];
         group "punctuation"
           [
             cases ~id:fst
               "delimits printable ASCII other than letters and digits, and \
                the category P"
               punctuation_rows delimits;
             prop ~count:500 "is split_regex on its class, under each behavior"
               (Gen.pair behavior bytes) splits_on_its_class;
             test "isolates each delimiter by default" (fun () ->
                 equal pretokens
                   [ ("a", (0, 1)); (",", (1, 2)); ("b", (2, 3)) ]
                   (Pre.pre_tokenize (Pre.punctuation ()) "a,b"));
           ];
         group "split"
           [
             prop ~count:1000
               ~examples:
                 [
                   (",", `Merged_with_next, false, "a,,,b");
                   ("ab", `Contiguous, true, "xababy");
                   ("\u{3001}", `Removed, true, "a\u{3001}\u{3001}b");
                   (* Inverted, a text without an occurrence is one
                      delimiter. *)
                   (",", `Isolated, true, "  ");
                   (",", `Removed, true, "  ");
                 ]
               "cuts at each occurrence of the pattern, as each behavior and \
                invert say"
               split_case follows_the_occurrences;
             test "removes the delimiters and does not invert by default"
               (fun () ->
                 equal pretokens
                   [ ("a", (0, 1)); ("b", (2, 3)) ]
                   (Pre.pre_tokenize (Pre.split ~pattern:"," ()) "a,b"));
             cases ~id:cut_id "with an empty pattern, cuts each character"
               empty_pattern_rows cuts;
           ];
         group "split_regex"
           [
             test "matches cl100k's pattern as HuggingFace does"
               (matches_huggingface "cl100k");
             test "matches o200k's pattern as HuggingFace does"
               (matches_huggingface "o200k");
             test "matches Tekken's pattern as HuggingFace does"
               (matches_huggingface "tekken");
             test "matches Kimi K2's pattern as HuggingFace does"
               (matches_huggingface "kimi");
             test
               "matches nested and intersected classes and scripts as \
                HuggingFace does"
               (matches_huggingface "classes");
             test
               "cuts at empty matches under each behavior as HuggingFace does"
               (matches_huggingface "behaviors");
             cases ~id:text_id
               "counts ², ³, ¹, ¼, ½ and ¾ as \\w and leaves out the join \
                controls"
               word_rows word_cuts;
             cases ~id:text_id "matches no class with a byte of no character"
               stray_rows cl100k_cuts;
             prop ~count:2000 "matches cl100k's pattern as it does in a group"
               split_text (as_in_a_group "cl100k");
             prop ~count:2000
               "matches cl100k's pattern as files spell it as it does in a \
                group"
               split_text
               (as_in_a_group "cl100k_hf");
             prop ~count:2000 "matches Qwen2's pattern as it does in a group"
               split_text (as_in_a_group "qwen2");
             prop ~count:2000 "matches Qwen3.5's pattern as it does in a group"
               split_text (as_in_a_group "qwen35");
             prop ~count:2000 "matches o200k's pattern as it does in a group"
               split_text o200k_law;
             prop ~count:2000 "matches Tekken's pattern as it does in a group"
               split_text tekken_law;
             test
               "matches each named pattern as in a group on the parity corpora"
               test_parity_corpora;
             prop ~count:3000 "under `Isolated, cuts the text into pretokens"
               (Gen.pair (Gen.of_list ~pp:quoted named_patterns) split_text)
               isolated_tiles;
             cases ~id:fst
               "raises Invalid_argument naming the construct it does not \
                support"
               rejected_rows is_rejected;
           ];
         group "char_delimiter"
           [
             prop ~count:500
               ~examples:
                 [
                   (",", "a,b,c");
                   (" ", "hello world");
                   ("\u{2581}", "\u{2581}a\u{2581}b");
                   (",", ",");
                 ]
               "is split removing its character"
               (Gen.pair
                  (Gen.of_list ~pp:quoted delimiters)
                  (text_of (delimiters @ [ "a"; "b"; "\u{e2}" ])))
               is_split_on_it;
             cases ~id:(Printf.sprintf "%S")
               "raises Invalid_argument unless given one character"
               [ ""; "ab"; "\u{2581}\u{2581}"; "\x81"; "\xc3"; "\xe0\x80\xa0" ]
               (fun c ->
                 raises_match Exn.invalid_arg (fun () -> Pre.char_delimiter c));
           ];
         group "digits"
           [
             prop ~count:500 "isolates runs of numbers, or each number"
               (Gen.pair Gen.bool bytes) isolates_numbers;
             cases ~id:text_id "takes a digit to be any character of category N"
               number_rows digit_cuts;
           ];
         group "prepend_scheme"
           [
             cases ~id:cut_id
               "`First marks the first pretoken of the member before it"
               first_rows cuts;
             test
               "`First marks a pretoken of encode only when nothing comes \
                before it in the document"
               test_first_in_a_document;
           ];
         group "metaspace"
           [
             prop ~count:500
               "without splitting, marks each space and prepends a marker the \
                text lacks"
               (Gen.triple (Gen.of_list ~pp:quoted markers) scheme text)
               marks_and_prepends;
             prop ~count:500 "cuts the marked text before each marker"
               (Gen.triple (Gen.of_list ~pp:quoted markers) scheme text)
               cuts_before_markers;
             prop ~count:500 "spans the text each pretoken was made from"
               (Gen.pair scheme unmarked_text)
               spans_its_source;
             cases ~id:cut_id
               "marks texts that open with spaces or markers as HuggingFace \
                does"
               metaspace_rows cuts;
             cases ~id:cut_id
               "in a sequence, places the pretokens of the members around it \
                through the marking"
               later_member_rows cuts;
             cases ~id:cut_id
               "places a marker cut into a pretoken of its own at the \
                character it opens"
               lone_marker_rows cuts;
             test
               "without splitting, places a model's tokens at the bytes they \
                were made from"
               test_unsplit_token_offsets;
             cases ~id:(Printf.sprintf "%S")
               "raises Invalid_argument unless the replacement is one character"
               [ ""; "ab"; "__"; "\x81"; "\xc3"; "\xe0\x80\xa0" ]
               (fun replacement ->
                 raises_match Exn.invalid_arg (fun () ->
                     Pre.metaspace ~replacement ()));
           ];
         group "unicode_scripts"
           [
             test "cuts where the script changes as HuggingFace does"
               matches_huggingface_scripts;
             cases ~id:(Printf.sprintf "%S")
               "counts Hiragana, Katakana and U+30FC as Han" han_rows
               is_one_pretoken;
             (* Scripts.txt 18.0.0: U+0558 is Armenian. It is new in 18.0.0, so
                HuggingFace, whose table is older, gives it no script and joins
                it to the run around it. *)
             cases ~id:text_id "reads the scripts of Unicode 18.0.0"
               [
                 ( "a\u{558}b",
                   [ ("a", (0, 1)); ("\u{558}", (1, 3)); ("b", (3, 4)) ] );
                 ("\u{558}abc", [ ("\u{558}", (0, 2)); ("abc", (2, 5)) ]);
                 ( "\u{65e5}\u{558}\u{672c}",
                   [
                     ("\u{65e5}", (0, 3));
                     ("\u{558}", (3, 5));
                     ("\u{672c}", (5, 8));
                   ] );
               ]
               (fun (text, expected) ->
                 equal pretokens expected
                   (Pre.pre_tokenize Pre.unicode_scripts text));
             prop ~count:500
               "gives the text without its leading run of spaces and \
                characters of no script"
               script_text drops_the_leading_run;
           ];
         group "fixed_length"
           [
             prop ~count:500
               "cuts pretokens of n characters, the last one shorter"
               (Gen.pair (Gen.int_range 1 4) text)
               cuts_n_characters;
             test "counts a byte that opens no whole character as a character"
               (fun () ->
                 equal pretokens
                   [ ("\xe2\x82", (0, 2)); ("a\xff", (2, 4)) ]
                   (Pre.pre_tokenize (Pre.fixed_length 2) "\xe2\x82a\xff"));
             cases ~id:string_of_int
               "raises Invalid_argument unless n is positive" [ 0; -1 ]
               (fun n ->
                 raises_match Exn.invalid_arg (fun () -> Pre.fixed_length n));
           ];
         group "sequence"
           [
             prop ~count:1000
               ~examples:[ (Pre.whitespace_split, Pre.punctuation (), dense) ]
               "cuts each pretoken of a member with the next, placing its \
                spans inside"
               (Gen.triple (member verbatim) (member verbatim) bytes)
               cuts_each_pretoken;
             prop ~count:1000
               "takes a nested sequence's members as written in place"
               (Gen.quad (member every) (member every) (member every) text)
               nests;
             prop ~count:1000
               "places a member's spans within the member before it"
               (Gen.triple (member every) (member every) bytes)
               lies_within;
             test "hands a member each pretoken as a whole text"
               test_member_sees_a_whole_text;
             cases ~id:cut_id "with no member, gives the text as one pretoken"
               empty_sequence_rows cuts;
           ];
         group "pre_tokenize"
           [
             prop ~count:1000 "gives ascending spans within the text"
               (Gen.pair (member every) bytes)
               ascends_within;
             test "reads an overlong space as bytes of no character" (fun () ->
                 equal pretokens
                   [ ("a\xe0\x80\xa0b", (0, 5)) ]
                   (Pre.pre_tokenize Pre.whitespace_split "a\xe0\x80\xa0b"));
             prop ~count:1000
               "reads a sequence RFC 3629 forbids as bytes of no character"
               (Gen.pair
                  (member reading_characters)
                  (Gen.list ~size:(Gen.int_range 0 24)
                     (Gen.of_list ~pp:quoted (characters @ forbidden))))
               reads_forbidden_as_strays;
             prop ~count:1000
               "gives the bytes of its span as a pretoken nothing rewrote"
               (Gen.pair (member verbatim) bytes)
               is_its_span;
             cases ~id:pre_id "gives no pretoken of an empty text" every
               gives_none;
             test
               "can give the pretokens cut from a rewritten one that \
                pretoken's span, on \"a b,c\""
               test_whole_spans;
           ];
         group "byte_level_decode"
           [
             cases ~id:text_id "gives the bytes the alphabet spells"
               alphabet_rows (fun (s, expected) ->
                 equal string expected (Pre.byte_level_decode s));
             cases ~id:(Printf.sprintf "%S")
               "leaves a string with a character outside the alphabet as it is"
               outside_rows (fun s -> equal string s (Pre.byte_level_decode s));
             prop ~count:1000 "inverts byte_level's encoding of any bytes"
               Gen.string
               (Law.round_trip string string encoded Pre.byte_level_decode);
           ];
         group "to_json"
           [
             cases ~id:json_id "writes HuggingFace's JSON" huggingface_rows
               writes_huggingface;
           ];
         group "of_json"
           [
             cases ~id:json_id "reads what to_json writes" huggingface_rows
               reads_back;
             test "reads a Split on Kimi K2's pattern" test_reads_kimi;
             cases ~id:fst "takes the constructor's default for a missing field"
               default_rows takes_defaults;
             cases ~id:fst
               "errors on what is not an object, lacks or names no known type, \
                or has invalid parameters"
               error_rows (fun (_, text) -> is_error (Pre.of_json (parse text)));
           ];
       ])
