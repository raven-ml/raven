(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let s = Col.string "s"
let texts vs = v [ ("s", Column.of_options Type.string vs) ]

let result e t =
  let run = Query.run (Query.select Expr.[ "out" := e ] (Query.of_table t)) in
  column (require_ok ~pp:Error.pp run) "out"

let matches p vs =
  Column.options Kind.bool (result (Expr.Str.matches p s) (texts vs))

let extract ?group p vs =
  Column.options Kind.string (result (Expr.Str.extract ?group p s) (texts vs))

let find_all p vs =
  Column.options Kind.(list string) (result (Expr.Str.find_all p s) (texts vs))

let split p vs =
  Column.options Kind.(list string) (result (Expr.Str.split p s) (texts vs))

let replace p ~by vs =
  Column.options Kind.string (result (Expr.Str.replace p ~by s) (texts vs))

let regex = Expr.Str.regex
let literal = Expr.Str.literal
let same_rows = array (option string)
let bools = array (option bool)
let lists = array (option (array string))
let one t = [| Some t |]

let utf_8 u =
  let b = Buffer.create 4 in
  Buffer.add_utf_8_uchar b (Uchar.of_int u);
  Buffer.contents b

let valid_utf_8 = satisfies ~claim:"valid UTF-8" string String.is_valid_utf_8

(* [occurs ~sub t] is [true] iff [sub] is a substring of [t]. *)
let occurs ~sub t =
  let n = String.length sub in
  let rec at i =
    i + n <= String.length t && (String.sub t i n = sub || at (i + 1))
  in
  at 0

(* [answers_alike p q rows] checks that [p] and [q] give the same answer to
   every operation on [rows]. *)
let answers_alike p q rows =
  equal bools (matches q rows) (matches p rows);
  equal lists (find_all q rows) (find_all p rows);
  equal lists (split q rows) (split p rows);
  equal same_rows (extract q rows) (extract p rows);
  equal same_rows (replace q ~by:"<x>" rows) (replace p ~by:"<x>" rows)

(* [refused_at r byte] checks that [regex r] raises, naming [byte]. *)
let refused_at r byte =
  raises_match
    (Exn.invalid_arg ~sub:(Printf.sprintf "byte %d" byte))
    (fun () -> regex r)

(* Generators *)

let text_gen =
  let scalar =
    Gen.of_list
      ~pp:(fun ppf -> Format.fprintf ppf "%S")
      [ "a"; "b"; "é"; "日"; "😀"; " "; "\n"; "k"; "ſ"; "K"; "ς" ]
  in
  Gen.map (String.concat "") (Gen.list ~size:(Gen.int_range 0 10) scalar)

let plain_gen = Gen.such_that (fun s -> s <> "") text_gen

let pattern_gen =
  let atom =
    Gen.of_list ~pp:Format.pp_print_string
      [
        "a";
        "b";
        "é";
        "日";
        "😀";
        ".";
        "\\d";
        "\\w";
        "\\s";
        "\\W";
        "[ab]";
        "[^a]";
        "[a-z]";
        "^";
        "$";
        "\\b";
        "\\B";
        "\\A";
        "\\z";
        "(?i)k";
        "(?m)^";
        "(?s).";
        "[\\w]";
      ]
  in
  let open Gen in
  let rec go depth =
    if depth = 0 then atom
    else
      let sub = go (depth - 1) in
      frequency
        [
          (3, atom);
          (2, map (fun (a, b) -> a ^ b) (pair sub sub));
          (1, map (fun (a, b) -> a ^ "|" ^ b) (pair sub sub));
          (2, map (fun a -> "(?:" ^ a ^ ")*") sub);
          (2, map (fun a -> "(?:" ^ a ^ ")+?") sub);
          (1, map (fun a -> "(" ^ a ^ ")?") sub);
          (1, map (fun a -> "(?:" ^ a ^ "){1,2}") sub);
        ]
  in
  with_pp (fun ppf -> Format.fprintf ppf "%S") (bind (int_range 0 3) go)

let pattern_and_text = Gen.pair pattern_gen text_gen

(* Str's patterns *)

let patterns =
  group "Str.pattern"
    [
      prop "a literal is the regex of its quoted text, for every operation"
        (Gen.pair plain_gen text_gen) (fun (p, t) ->
          answers_alike (literal p) (regex (Expr.Str.quote p)) (one t));
      prop
        "prefix, suffix and pieces are the regexes they name, for every \
         operation"
        ~examples:[ ("ab", "b", "abcab"); ("ab", "b", "") ]
        (Gen.triple plain_gen plain_gen text_gen)
        (fun (p, q, t) ->
          let quote = Expr.Str.quote in
          cover "the text starts with the prefix"
            (String.starts_with ~prefix:p t);
          cover "the text does not" (not (String.starts_with ~prefix:p t));
          answers_alike (Expr.Str.prefix p) (regex ("\\A" ^ quote p)) (one t);
          answers_alike (Expr.Str.suffix p) (regex (quote p ^ "\\z")) (one t);
          answers_alike
            (Expr.Str.pieces [ p; q ])
            (regex (quote p ^ "(?s:.*?)" ^ quote q))
            (one t));
      test "a null row is null in every operation" (fun () ->
          let r = regex "a" in
          let vs = [| None; Some "a" |] in
          equal same_rows [| None; Some "a" |] (extract r vs);
          equal lists [| None; Some [| "a" |] |] (find_all r vs);
          equal lists [| None; Some [| ""; "" |] |] (split r vs);
          equal same_rows [| None; Some "b" |] (replace r ~by:"b" vs);
          equal bools [| None; Some true |] (matches r vs));
    ]

let refusals =
  [
    group "Str.literal"
      [
        cases "raises on empty or non-UTF-8 text" ~id:String.escaped
          [ ""; "\xff" ] (fun t ->
            raises_match Exn.invalid_arg (fun () -> literal t));
      ];
    group "Str.prefix"
      [
        cases "raises on empty or non-UTF-8 text" ~id:String.escaped
          [ ""; "\xc3" ] (fun t ->
            raises_match Exn.invalid_arg (fun () -> Expr.Str.prefix t));
      ];
    group "Str.suffix"
      [
        cases "raises on empty or non-UTF-8 text" ~id:String.escaped
          [ ""; "\xff" ] (fun t ->
            raises_match Exn.invalid_arg (fun () -> Expr.Str.suffix t));
      ];
    group "Str.pieces"
      [
        cases "raises on no strings, or an empty or non-UTF-8 one"
          ~id:(fun ss -> String.concat "; " (List.map (Printf.sprintf "%S") ss))
          [ []; [ "" ]; [ "a"; "" ]; [ "a"; "\xff" ] ]
          (fun ss ->
            raises_match Exn.invalid_arg (fun () -> Expr.Str.pieces ss));
      ];
  ]

(* Str.regex and Str.quote *)

let compiling =
  group "Str.regex"
    [
      cases "refuses a pattern outside the language, naming the byte" ~id:fst
        [
          ("(?=a)", 0);
          ("(?!a)", 0);
          ("(?<=a)", 0);
          ("(?<!a)", 0);
          ("(a)\\1", 3);
          ("(?P=n)", 0);
          ("\\k<n>", 0);
          ("[a&&b]", 2);
          ("[a--b]", 2);
          ("[[ab]]", 1);
          ("[[:alpha:]]", 1);
          ("a{2}{3}", 4);
          ("a*+", 2);
          ("(?>a)", 0);
          ("(?x)a", 2);
          ("(?#c)", 0);
          ("\\X", 0);
          ("\\R", 0);
          ("\\C", 0);
          ("\\Qa", 0);
          ("*", 0);
          ("a{", 2);
          ("a{2,1}", 1);
          ("(a", 2);
          ("a)", 1);
          ("[a", 0);
          ("[b-a]", 2);
          ("\\p{NotAProperty}", 0);
          ("\\x{D800}", 2);
          ("\\x{110000}", 2);
          ("(?<a>b)(?<a>c)", 7);
          ("\\x{7fffffffffffffff}", 2);
          ("[\\x{7fffffffffffffff}-a]", 3);
        ]
        (fun (r, byte) -> refused_at r byte);
      test "refuses a pattern that is not UTF-8, naming the byte" (fun () ->
          refused_at "a\xff" 1);
      cases
        "reads \\ before punctuation as that punctuation, and repeats an \
         assertion"
        ~id:(fun (r, _, _) -> r)
        [
          ("\\_", "a_b", Some "_");
          ("\\.", "a.b", Some ".");
          ("\\b*", "ab", Some "");
        ]
        (fun (r, t, expected) ->
          equal (option string) expected (extract (regex r) (one t)).(0));
    ]

let quoting =
  let metachar =
    Gen.of_list
      ~pp:(fun ppf -> Format.fprintf ppf "%S")
      [
        ".";
        "*";
        "+";
        "?";
        "(";
        ")";
        "|";
        "[";
        "]";
        "{";
        "}";
        "^";
        "$";
        "\\";
        "-";
        "&";
        "#";
        "~";
        "a";
      ]
  in
  let text =
    Gen.map (String.concat "") (Gen.list ~size:(Gen.int_range 1 6) metachar)
  in
  group "Str.quote"
    [
      prop "matches exactly its text, metacharacters included"
        ~examples:[ ("a.", "xa.y"); ("a.", "xaby") ]
        (Gen.pair text text)
        (fun (q, t) ->
          let holds = occurs ~sub:q t in
          cover "the text holds it" holds;
          cover "the text does not" (not holds);
          let r = regex (Expr.Str.quote q) in
          equal bools (one holds) (matches r (one t));
          equal same_rows
            [| (if holds then Some q else None) |]
            (extract r (one t)));
    ]

(* Str.matches, extract, find_all, split, replace *)

(* A pattern of a thousand-byte window has states of about a thousand
   instructions each, so a row of a few thousand random bytes fills a kernel's
   word budget for matching and the states are dropped within the row. *)
let long_row () =
  let window = 1000 and n = 4000 in
  let r = regex (Printf.sprintf "[ab]*a[ab]{%d}c" window) in
  let rng = Random.State.make [| 7 |] in
  let row n =
    String.init n (fun _ -> if Random.State.bool rng then 'a' else 'b')
  in
  let hit =
    let b = Bytes.of_string (row n) in
    Bytes.set b (n - window - 1) 'a';
    Bytes.to_string b ^ "c"
  in
  let rows = [| Some (row n ^ "d"); Some hit; Some (row n) |] in
  equal bools [| Some false; Some true; Some false |] (matches r rows);
  equal bools
    (Array.map
       (Option.map Option.is_some)
       (Array.map Option.some (extract r rows)))
    (matches r rows)

let matching =
  group "Str.matches"
    [
      prop "is true exactly where extract finds a match"
        ~examples:[ ("z", "ab") ]
        pattern_and_text
        (fun (p, t) ->
          let r = regex p in
          let found = Option.is_some (extract r (one t)).(0) in
          cover "a text without a match" (not found);
          cover "a text with a match" found;
          equal bools (one found) (matches r (one t)));
      test
        "answers on a row whose states exceed a kernel's memory as extract does"
        long_row;
    ]

let extracting =
  group "Str.extract"
    [
      test "is null without a match, and where the group did not take part"
        (fun () ->
          let r = regex "(a)|(b)" in
          equal same_rows [| Some "a"; None; None |]
            (extract ~group:1 r [| Some "a"; Some "b"; Some "c" |]);
          equal same_rows (one "") (extract (regex "x*") (one "ab")));
      cases "leaves unset a group that only an abandoned attempt set"
        ~id:(fun (r, t, g, _) -> Printf.sprintf "%s on %s, group %d" r t g)
        [
          ("(x)y|z", "xz", 1, None);
          ("(a)b|c", "ac", 1, None);
          ("(a)(b)c|(a)d", "abad", 2, None);
          ("(a)(b)c|(a)d", "abad", 3, Some "a");
        ]
        (fun (r, t, group, expected) ->
          equal (option string) expected (extract ~group (regex r) (one t)).(0));
      cases "raises on a group the pattern lacks"
        ~id:(fun (r, g) -> Printf.sprintf "group %d of %s" g r)
        [ ("(a)", 2); ("(a)", -1) ]
        (fun (r, group) ->
          raises_match Exn.invalid_arg (fun () ->
              Expr.Str.extract ~group (regex r) s));
      test "raises on a group of a literal" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Expr.Str.extract ~group:1 (literal "a") s));
      prop "finds a group inside the whole match" pattern_and_text
        (fun (p, t) ->
          let r = regex p in
          let whole = (extract r (one t)).(0) in
          let has g =
            match Expr.Str.extract ~group:g r s with
            | _ -> true
            | exception Invalid_argument _ -> false
          in
          for g = 1 to 2 do
            if has g then
              match (extract ~group:g r (one t)).(0) with
              | None -> ()
              | Some text ->
                  valid_utf_8 text;
                  contains ~sub:text (Option.get whole)
          done);
    ]

let splitting =
  group "Str.split"
    [
      cases "and find_all follow the same matches"
        ~id:(fun (r, t, _, _) -> r ^ " on " ^ t)
        [
          ("x*", "ab", [| ""; "a"; "b"; "" |], [| ""; ""; "" |]);
          ("", "é.x", [| ""; "é"; "."; "x"; "" |], [| ""; ""; ""; "" |]);
          ("a||b", "ab", [| ""; "b"; "" |], [| "a"; "" |]);
          (",", "a,,b", [| "a"; ""; "b" |], [| ","; "," |]);
          ("z", "ab", [| "ab" |], [||]);
          (",", "", [| "" |], [||]);
          ("_", "sub-01_ses-02", [| "sub-01"; "ses-02" |], [| "_" |]);
        ]
        (fun (r, t, pieces, found) ->
          equal lists [| Some pieces |] (split (regex r) (one t));
          equal lists [| Some found |] (find_all (regex r) (one t)));
      prop
        "pieces interleaved with find_all's matches are the row, and replace \
         writes by in the matches' place"
        pattern_and_text (fun (p, t) ->
          let r = regex p in
          let pieces = Option.get (split r (one t)).(0) in
          let found = Option.get (find_all r (one t)).(0) in
          cover "an empty match" (Array.mem "" found);
          cover "two matches or more" (Array.length found >= 2);
          equal int (Array.length found + 1) (Array.length pieces);
          let rebuilt around =
            String.concat ""
              (List.concat
                 (List.mapi
                    (fun i piece ->
                      if i < Array.length found then [ piece; around found.(i) ]
                      else [ piece ])
                    (Array.to_list pieces)))
          in
          equal string t (rebuilt Fun.id);
          equal same_rows
            (one (rebuilt (fun m -> "<" ^ m ^ ">")))
            (replace r ~by:"<$0>" (one t));
          Array.iter valid_utf_8 pieces);
    ]

let replacing =
  group "Str.replace"
    [
      cases "expands the template"
        ~id:(fun (r, by, _, _) -> r ^ " by " ^ by)
        [
          ("(\\d+)/(\\d+)", "$2-$1", "12/34", "34-12");
          ("(?<y>\\d+)", "<${y}>", "1", "<1>");
          ("(?P<y>\\d+)", "<$y>", "1", "<1>");
          ("(a)(b)?", "$1$2", "a", "a");
          ("(a)", "$0$0", "a", "aa");
          ("(a)", "$$1", "a", "$1");
          ("(a)", "$1_x", "a", "a_x");
          ("a", "$0", "a", "a");
        ]
        (fun (r, by, t, expected) ->
          equal same_rows (one expected) (replace (regex r) ~by (one t)));
      cases "raises on a template that names nothing the pattern has" ~id:Fun.id
        [ "$"; "$x"; "$2"; "${"; "${n}"; "$ " ] (fun by ->
          raises_match Exn.invalid_arg (fun () ->
              Expr.Str.replace (regex "(a)") ~by s));
      test "raises on by that is not UTF-8" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Expr.Str.replace (literal "a") ~by:"\xc3" s));
      test "writes by as text unless the pattern is a regex" (fun () ->
          equal same_rows (one "$0") (replace (literal "a") ~by:"$0" (one "a")));
      test "leaves a row without a match unchanged" (fun () ->
          equal same_rows (one "cat") (replace (regex "z+") ~by:"x" (one "cat")));
      test "writes nothing for a group that did not take part" (fun () ->
          equal same_rows (one "a[]")
            (replace (regex "(a)b|c") ~by:"[$1]" (one "ac")));
    ]

(* Regular expressions *)

(* golden/regex.txt, from gen/regex.py: for each pattern and text, what Rust's
   regex crate answers through Polars. Each field is hex of UTF-8, [=] the empty
   text, [-] a null, [.] no groups or matches. *)
type answer = {
  text : string;
  contains : bool;
  whole : string option;
  groups : string option array;
  all : string array;
  replaced : string;
}

let reference =
  let hex h =
    String.init
      (String.length h / 2)
      (fun i -> Char.chr (int_of_string ("0x" ^ String.sub h (2 * i) 2)))
  in
  let field = function "-" -> None | "=" -> Some "" | h -> Some (hex h) in
  let fields sep = function
    | "." -> [||]
    | f -> Array.of_list (List.map field (String.split_on_char sep f))
  in
  let answers = Hashtbl.create 1000 and order = ref [] in
  In_channel.with_open_text "golden/regex.txt" In_channel.input_lines
  |> List.filter (fun l -> l <> "" && l.[0] <> '#')
  |> List.iter (fun l ->
      match String.split_on_char ' ' l with
      | [ p; t; c; w; g; a; r ] ->
          let p = hex p in
          if not (Hashtbl.mem answers p) then order := p :: !order;
          Hashtbl.add answers p
            {
              text = Option.get (field t);
              contains = c = "1";
              whole = field w;
              groups = fields ';' g;
              all = Array.map Option.get (fields ',' a);
              replaced = Option.get (field r);
            }
      | _ -> failwith l);
  List.rev_map (fun p -> (p, List.rev (Hashtbl.find_all answers p))) !order

let agrees (p, answers) =
  let r = regex p in
  let rows = Array.of_list (List.map (fun a -> Some a.text) answers) in
  let column f = Array.of_list (List.map f answers) in
  equal ~msg:"matches" bools
    (column (fun a -> Some a.contains))
    (matches r rows);
  equal ~msg:"extract" same_rows (column (fun a -> a.whole)) (extract r rows);
  let groups =
    List.fold_left (fun n a -> max n (Array.length a.groups)) 0 answers
  in
  for g = 1 to groups do
    let group a =
      if Array.length a.groups = 0 then None else a.groups.(g - 1)
    in
    equal
      ~msg:(Printf.sprintf "extract ~group:%d" g)
      same_rows (column group) (extract ~group:g r rows)
  done;
  equal ~msg:"find_all" lists (column (fun a -> Some a.all)) (find_all r rows);
  equal ~msg:"replace" same_rows
    (column (fun a -> Some a.replaced))
    (replace r ~by:"<$0>" rows)

let expressions =
  group "regular expressions"
    [
      cases "agree with Rust's regex crate on every operation" ~id:fst reference
        agrees;
      cases
        "discard an empty match abutting the previous one, and resume one \
         scalar on"
        ~id:(fun (r, t, _) -> r ^ " on " ^ String.escaped t)
        [
          ("x*", "abxd", "-a-b-d-");
          ("", "é.x", "-é-.-x-");
          ("", "", "-");
          ("a||b", "ab", "-b-");
          ("||b", "b", "-b-");
          ("a*?", "baaa", "-b-a-a-a-");
          ("(?:|a)*", "aaa", "-a-a-a-");
          ("\\b", "ab cd", "-ab- -cd-");
          ("\\b", " \xcc\x81", " -\xcc\x81-");
          ("\\B", "ab", "a-b");
          ("$", "ab", "ab-");
          ("^", "ab", "-ab");
          ("\\b|", "ab", "-a-b-");
        ]
        (fun (r, t, expected) ->
          equal same_rows (one expected) (replace (regex r) ~by:"-" (one t)));
      (let a = String.make 50_000 'a' and x = String.make 50_000 'x' in
       (* Each row holds the pattern's required literal, so an automaton runs; a
          backtracking engine would not finish these rows. *)
       cases "match a pathological pattern in time linear in the row"
         ~id:(fun (r, _, _) -> r)
         [
           ("(a+)+b", "b" ^ a, None);
           ("(a|a)*c", a ^ "c", Some (a ^ "c"));
           ("(a*)*b", "b" ^ a, Some "b");
           ("(x+x+)+y", "y" ^ x, None);
           ("^(([a-z])+.)+[A-Z]([a-z])+$", a, None);
         ]
         (fun (r, row, expected) ->
           let r = regex r in
           equal bools (one (Option.is_some expected)) (matches r (one row));
           equal same_rows [| expected |] (extract r (one row))));
      cases "match each construct of the table as it states"
        ~id:(fun (r, t, _) -> r ^ " on " ^ String.escaped t)
        [
          ("a", "cat", Some "a");
          (".", "é", Some "é");
          ("[a-c]+", "xabcab", Some "abcab");
          ("[^a]", "é", Some "é");
          ("\\d+", "x१२३ 456", Some "१२३");
          ("\\w+", "café au", Some "café");
          ("\\s", "a\xe2\x80\x83b", Some "\xe2\x80\x83");
          ("\\p{Lu}\\p{Ll}+", "Hello", Some "Hello");
          ("\\p{Script=Greek}+", "x αβγ", Some "αβγ");
          ("\\p{greek}+", "αβ", Some "αβ");
          ("\\p{IsGreek}+", "x αβ", Some "αβ");
          ("\\p{gc=isLu}", "aB", Some "B");
          ("\\p{isc}", "a\x01", Some "\x01");
          ("\\p{gc=Lu}", "aB", Some "B");
          ("\\p{General_Category=Decimal_Number}", "a5", Some "5");
          ("\\p{Decimal Number}", "a5", Some "5");
          ("\\p{decimal-NUMBER}", "a5", Some "5");
          ("\\pL", "1a", Some "a");
          ("\\x41", "A", Some "A");
          ("\\x{1F600}", "😀", Some "😀");
          ("\\n", "\n", Some "\n");
          ("\\t", "a\tb", Some "\t");
          ("\\r", "a\rb", Some "\r");
          ("\\f", "a\x0cb", Some "\x0c");
          ("\\v", "a\x0bb", Some "\x0b");
          ("\\a", "a\x07b", Some "\x07");
          ("\\e", "a\x1bb", Some "\x1b");
          ("^abc$", "abc", Some "abc");
          ("(?m)^b$", "a\nb", Some "b");
          ("(?m)a$", "a\nb", Some "a");
          ("(?m)^b", "a\rb", None);
          ("(?m)^b", "a\xe2\x80\xa8b", None);
          ("\\Aab\\z", "ab", Some "ab");
          ("\\bfoo\\b", "a foo b", Some "foo");
          ("\\Bb", "ab", Some "b");
          ("(a)|(b)", "b", Some "b");
          ("a|ab", "ab", Some "a");
          ("a*?", "aaa", Some "");
          ("a+?", "aaa", Some "a");
          ("a{2}", "aaa", Some "aa");
          ("a{2,}", "aaaa", Some "aaaa");
          ("a{1,2}?", "aa", Some "a");
          ("(?:ab)+", "ababx", Some "abab");
          ("(?<y>\\d+)-(?P<m>\\d+)", "2024-11", Some "2024-11");
          ("(?i)HeLLo", "hello", Some "hello");
          ("(?i:a)b", "Ab", Some "Ab");
          ("(?s).", "\n", Some "\n");
          ("(?-i)a", "A", None);
          ("(?i)a(?-i)b", "AB", None);
          ("(?:(?i)a|b)c", "Bc", Some "Bc");
          ("(?:a(?i)|b)c", "Bc", Some "Bc");
          ("[]a]", "]", Some "]");
          ("[-a]", "-", Some "-");
          ("[a-]", "-", Some "-");
          ("[\\d_]+", "1_2", Some "1_2");
          ("(a|ab)(c|bcd)", "abcd", Some "abcd");
          ("[^\\P{L}\\p{Greek}]", "1αa", Some "a");
        ]
        (fun (r, t, expected) ->
          equal (option string) expected (extract (regex r) (one t)).(0));
      cases "match nothing with an empty class" ~id:fst
        [
          ("[^\\x{0}-\\x{10FFFF}]", "");
          ("\\P{any}", "x");
          ("[^\\P{Lu}\\p{Lu}]", "A");
          ("(?i)[^\\P{Lu}\\p{Ll}\\p{Lu}]", "x");
        ]
        (fun (r, t) ->
          let r = regex r in
          equal bools (one false) (matches r (one t));
          equal lists (one [||]) (find_all r (one t));
          equal same_rows [| None |] (extract r (one t));
          equal same_rows (one t) (replace r ~by:"x" (one t)));
      test "compile \\p{Cs}, which no valid text holds" (fun () ->
          ignore (regex "\\p{Cs}"));
    ]

(* Unicode 18.0.0 *)

(* golden/unicode.txt, from gen/case.py: [C name ranges] the members of a class,
   [A class name] another name of the class, [F a b ...] an orbit of simple case
   folding. *)
let unicode_lines =
  In_channel.with_open_text "golden/unicode.txt" In_channel.input_lines
  |> List.filter (fun l -> l <> "" && l.[0] <> '#')
  |> List.map (fun l -> List.filter (( <> ) "") (String.split_on_char ' ' l))

let hex h = int_of_string ("0x" ^ h)

let range r =
  match String.split_on_char '.' r with
  | [ a; ""; b ] -> (hex a, hex b)
  | _ -> failwith r

let classes =
  List.filter_map
    (function
      | "C" :: name :: ranges -> Some (name, List.map range ranges) | _ -> None)
    unicode_lines

let aliases =
  List.filter_map
    (function [ "A"; name; alias ] -> Some (name, alias) | _ -> None)
    unicode_lines

let orbits =
  List.filter_map
    (function "F" :: us -> Some (List.map hex us) | _ -> None)
    unicode_lines

(* [member ranges] is membership in the ascending [ranges], by bisection. *)
let member ranges =
  let a = Array.of_list ranges in
  let rec go u lo hi =
    if lo >= hi then false
    else
      let mid = (lo + hi) / 2 in
      let l, h = a.(mid) in
      if u < l then go u lo mid else if u > h then go u (mid + 1) hi else true
  in
  fun u -> go u 0 (Array.length a)

let is_scalar u = u >= 0 && u <= 0x10FFFF && (u < 0xD800 || u > 0xDFFF)

(* The scalars on each side of each range's ends, where an off-by-one in a
   compiled class shows. *)
let boundaries ranges =
  List.concat_map (fun (a, b) -> [ a - 1; a; b; b + 1 ]) ranges
  |> List.filter is_scalar |> List.sort_uniq compare |> Array.of_list

let every_scalar =
  lazy
    (let surrogates = 0xE000 - 0xD800 in
     Array.init (0x110000 - surrogates) (fun i ->
         if i < 0xD800 then i else i + surrogates))

let alone us = Array.map (fun u -> Some (utf_8 u)) us

(* [wrong us expected got] names the scalars of [us] whose row of [got] is not
   [expected u]. *)
let wrong us expected got =
  List.filteri (fun i _ -> got.(i) <> Some (expected us.(i))) (Array.to_list us)
  |> List.map (Printf.sprintf "U+%04X")

let every_scalar_of r name =
  test (Printf.sprintf "%s matches exactly Annex C's scalars" r) (fun () ->
      let us = Lazy.force every_scalar in
      let ranges = List.assoc name classes in
      equal (list string) []
        (wrong us (member ranges) (matches (regex ("^" ^ r ^ "$")) (alone us))))

let unicode =
  group "Unicode 18.0.0"
    [
      cases "each \\p class holds exactly its ranges, at every boundary" ~id:fst
        classes (fun (name, ranges) ->
          let us = boundaries ranges in
          let holds = member ranges in
          equal (list string) []
            (wrong us holds (matches (regex ("\\p{" ^ name ^ "}")) (alone us)));
          equal (list string) []
            (wrong us
               (fun u -> not (holds u))
               (matches (regex ("\\P{" ^ name ^ "}")) (alone us))));
      cases "a class answers to each of its names" ~id:snd aliases
        (fun (name, alias) ->
          let us = boundaries (List.assoc name classes) in
          let us = Array.sub us 0 (min 12 (Array.length us)) in
          equal bools
            (matches (regex ("\\p{" ^ name ^ "}")) (alone us))
            (matches (regex ("\\p{" ^ alias ^ "}")) (alone us)));
      every_scalar_of "\\w" "word";
      every_scalar_of "\\d" "digit";
      every_scalar_of "\\s" "space";
      every_scalar_of "\\p{alpha}" "alpha";
      test "\\W, \\D and \\S are the complements" (fun () ->
          let rows = alone (Array.sub (Lazy.force every_scalar) 0 0x3000) in
          let flip = Array.map (Option.map not) in
          equal bools
            (flip (matches (regex "^\\w$") rows))
            (matches (regex "^\\W$") rows);
          equal bools
            (flip (matches (regex "^\\d$") rows))
            (matches (regex "^\\D$") rows);
          equal bools
            (flip (matches (regex "^\\s$") rows))
            (matches (regex "^\\S$") rows));
      test ". matches every scalar but U+000A, and (?s). every scalar"
        (fun () ->
          let us = Lazy.force every_scalar in
          equal (list string) []
            (wrong us (fun u -> u <> 0x0A) (matches (regex "^.$") (alone us)));
          equal (list string) []
            (wrong us (Fun.const true) (matches (regex "(?s)^.$") (alone us))));
      cases "(?i) closes each simple case-folding orbit"
        ~id:(fun us ->
          String.concat " " (List.map (Printf.sprintf "U+%04X") us))
        orbits
        (fun orbit ->
          let us = Array.of_list orbit in
          List.iter
            (fun u ->
              equal
                ~msg:(Printf.sprintf "(?i)U+%04X" u)
                bools
                (Array.map (Fun.const (Some true)) us)
                (matches (regex ("(?i)^" ^ utf_8 u ^ "$")) (alone us)))
            orbit);
      cases "(?i) closes a class before it negates"
        ~id:(fun (r, t, _) -> r ^ " on " ^ t)
        [
          ("(?i)[^a]", "a", false);
          ("(?i)[^a]", "A", false);
          ("(?i)[^a]", "b", true);
          ("(?i)[^K]", "\xe2\x84\xaa" (* U+212A KELVIN SIGN *), false);
          ("(?i)\\P{Lu}", "a", false);
          ("(?i)ß", "ẞ" (* U+1E9E, its simple folding orbit *), true);
          ("(?i)ß", "SS", false);
          ("(?i)σ", "ς", true);
          ("(?i)σ", "Σ", true);
        ]
        (fun (r, t, expected) ->
          equal bools (one expected) (matches (regex r) (one t)));
    ]

let () =
  exit
  @@ run "Regex"
       ([ patterns ] @ refusals
       @ [
           compiling;
           quoting;
           matching;
           extracting;
           splitting;
           replacing;
           expressions;
           unicode;
         ])
