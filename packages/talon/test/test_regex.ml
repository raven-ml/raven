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

let matches p vs = Column.options Kind.bool (result (Expr.Str.matches p s) (texts vs))

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

let utf_8 u =
  let b = Buffer.create 4 in
  Buffer.add_utf_8_uchar b (Uchar.of_int u);
  Buffer.contents b

let is_scalar u = u >= 0 && u <= 0x10FFFF && not (u >= 0xD800 && u <= 0xDFFF)

(* Reference data *)

let hex_decode h =
  let n = String.length h / 2 in
  String.init n (fun i -> Char.chr (int_of_string ("0x" ^ String.sub h (2 * i) 2)))

let field = function "-" -> None | "=" -> Some "" | h -> Some (hex_decode h)

let list_field = function
  | "." -> [||]
  | f -> Array.of_list (List.map field (String.split_on_char ',' f))

type golden = {
  pattern : string;
  text : string;
  contains : bool;
  whole : string option;
  captures : string option array;
  all : string option array;
  replaced : string;
}

let golden_lines =
  In_channel.with_open_text "golden/regex.txt" In_channel.input_lines
  |> List.filter (fun l -> l <> "" && l.[0] <> '#')
  |> List.map (fun l ->
         match String.split_on_char ' ' l with
         | [ p; t; c; w; g; a; r ] ->
             let split_groups = function
               | "." -> [||]
               | f -> Array.of_list (List.map field (String.split_on_char ';' f))
             in
             {
               pattern = hex_decode p;
               text = Option.get (field t);
               contains = c = "1";
               whole = field w;
               captures = split_groups g;
               all = list_field a;
               replaced = Option.get (field r);
             }
         | _ -> failwith l)

let golden_agreement =
  test "agrees with Rust's regex crate through Polars, on every column" (fun () ->
      let wrong = ref [] in
      let report c pat text what expected actual =
        wrong :=
          Printf.sprintf "%S on %S %s: expected %s, got %s" pat text what expected
            actual
          :: !wrong;
        ignore c
      in
      let pp_opt = function None -> "null" | Some s -> Printf.sprintf "%S" s in
      List.iter
        (fun g ->
          let r = regex g.pattern in
          let rows = [| Some g.text |] in
          let c = (matches r rows).(0) in
          if c <> Some g.contains then
            report () g.pattern g.text "contains"
              (string_of_bool g.contains)
              (Option.fold ~none:"null" ~some:string_of_bool c);
          let e = (extract r rows).(0) in
          if e <> g.whole then
            report () g.pattern g.text "whole" (pp_opt g.whole) (pp_opt e);
          Array.iteri
            (fun i cap ->
              let e = (extract ~group:(i + 1) r rows).(0) in
              if e <> cap then
                report () g.pattern g.text
                  (Printf.sprintf "group %d" (i + 1))
                  (pp_opt cap) (pp_opt e))
            g.captures;
          let a = Option.get (find_all r rows).(0) in
          let expect_all = Array.map Option.get g.all in
          if a <> expect_all then
            report () g.pattern g.text "find_all"
              (String.concat "|" (Array.to_list expect_all))
              (String.concat "|" (Array.to_list a));
          let rep = Option.get (replace r ~by:"<$0>" rows).(0) in
          if rep <> g.replaced then
            report () g.pattern g.text "replace" (pp_opt (Some g.replaced))
              (pp_opt (Some rep)))
        golden_lines;
      equal ~msg:(Printf.sprintf "%d lines" (List.length golden_lines))
        (list string) [] (List.rev !wrong))

(* Unicode *)

type cls = { name : string; ranges : (int * int) list }

let unicode_lines =
  In_channel.with_open_text "golden/unicode.txt" In_channel.input_lines
  |> List.filter (fun l -> l <> "" && l.[0] <> '#')

let parse_ranges body =
  if body = "" then []
  else
    List.map
      (fun r ->
        match String.index_opt r '.' with
        | None -> failwith r
        | Some d ->
            let first = int_of_string ("0x" ^ String.sub r 0 d) in
            let last =
              int_of_string
                ("0x" ^ String.sub r (d + 2) (String.length r - d - 2))
            in
            (first, last))
      (String.split_on_char ' ' body)

let classes =
  List.filter_map
    (fun l ->
      if String.length l > 2 && l.[0] = 'C' && l.[1] = ' ' then
        let i = String.index_from l 2 ' ' in
        let name = String.sub l 2 (i - 2) in
        Some { name; ranges = parse_ranges (String.sub l (i + 1) (String.length l - i - 1)) }
      else None)
    unicode_lines

let aliases =
  List.filter_map
    (fun l ->
      if String.length l > 2 && l.[0] = 'A' then
        match String.split_on_char ' ' l with
        | [ _; name; alias ] -> Some (name, alias)
        | _ -> failwith l
      else None)
    unicode_lines

let orbits =
  List.filter_map
    (fun l ->
      if String.length l > 2 && l.[0] = 'F' then
        Some
          (List.map
             (fun h -> int_of_string ("0x" ^ h))
             (List.tl (String.split_on_char ' ' l)))
      else None)
    unicode_lines

(* [member ranges u] is [true] iff [u] lies in the ascending [ranges]. *)
let member ranges =
  let a = Array.of_list ranges in
  fun u ->
    let rec go lo hi =
      if lo >= hi then false
      else
        let mid = (lo + hi) / 2 in
        let l, h = a.(mid) in
        if u < l then go lo mid else if u > h then go (mid + 1) hi else true
    in
    go 0 (Array.length a)

let class_named n = List.find (fun c -> c.name = n) classes

(* A class's name in the pattern syntax: gc=X, sc=X or scx=X read through
   \p{...} directly. Boundary rows are where an off-by-one in a compiled class
   shows. *)
let class_pattern c = Printf.sprintf "\\p{%s}" c.name

let boundary_rows ranges =
  List.concat_map (fun (a, b) -> [ a - 1; a; b; b + 1 ]) ranges
  |> List.filter is_scalar
  |> List.sort_uniq compare

let each_scalar =
  let surrogates = 0xE000 - 0xD800 in
  let scalar i = if i < 0xD800 then i else i + surrogates in
  Array.init (0x110000 - surrogates) scalar

let classes_boundaries =
  test "each class holds exactly its Unicode ranges, at every boundary" (fun () ->
      let wrong = ref [] in
      List.iter
        (fun c ->
          let rows = Array.of_list (boundary_rows c.ranges) in
          let got =
            matches (regex (class_pattern c))
              (Array.map (fun u -> Some (utf_8 u)) rows)
          in
          let holds = member c.ranges in
          Array.iteri
            (fun i u ->
              if got.(i) <> Some (holds u) then
                wrong :=
                  Printf.sprintf "%s of U+%04X" c.name u :: !wrong)
            rows;
          let negated =
            matches (regex (Printf.sprintf "\\P{%s}" c.name))
              (Array.map (fun u -> Some (utf_8 u)) rows)
          in
          Array.iteri
            (fun i u ->
              if negated.(i) <> Some (not (holds u)) then
                wrong :=
                  Printf.sprintf "\\P{%s} of U+%04X" c.name u :: !wrong)
            rows)
        classes;
      equal (list string) [] (List.rev !wrong))

let alias_names =
  test "a class answers to every name of the Unicode files" (fun () ->
      List.iter
        (fun (name, alias) ->
          let c = class_named name in
          let rows =
            let r = boundary_rows c.ranges in
            Array.of_list
              (List.map
                 (fun u -> Some (utf_8 u))
                 (List.filteri (fun i _ -> i < 12) r))
          in
          equal ~msg:alias bools
            (matches (regex (class_pattern c)) rows)
            (matches (regex (Printf.sprintf "\\p{%s}" alias)) rows))
        aliases)

let scalar_rows = Array.map (fun u -> Some (utf_8 u)) each_scalar

let every_scalar_of pattern name =
  test (Printf.sprintf "%s: every scalar" pattern) (fun () ->
      let c = class_named name in
      let holds = member c.ranges in
      let got = matches (regex ("^" ^ pattern ^ "$")) scalar_rows in
      let wrong =
        List.filteri
          (fun i _ -> got.(i) <> Some (holds each_scalar.(i)))
          (Array.to_list each_scalar)
      in
      equal (list string) [] (List.map (Printf.sprintf "U+%04X") wrong))

let annex_c =
  group "Annex C"
    [
      every_scalar_of "\\w" "word";
      every_scalar_of "\\d" "digit";
      every_scalar_of "\\s" "space";
      every_scalar_of "\\p{alpha}" "alpha";
      test "\\W, \\D and \\S are the complements" (fun () ->
          let rows = Array.sub scalar_rows 0 0x3000 in
          let flip = Array.map (Option.map not) in
          equal bools (matches (regex "^\\W$") rows)
            (flip (matches (regex "^\\w$") rows));
          equal bools (matches (regex "^\\D$") rows)
            (flip (matches (regex "^\\d$") rows));
          equal bools (matches (regex "^\\S$") rows)
            (flip (matches (regex "^\\s$") rows)));
    ]

let dot_is_every_scalar =
  test ". matches any scalar but newline, and (?s) every scalar" (fun () ->
      let dot = matches (regex "^.$") scalar_rows in
      let dot_all = matches (regex "(?s)^.$") scalar_rows in
      let wrong =
        List.filteri
          (fun i _ ->
            dot.(i) <> Some (each_scalar.(i) <> 0x0A)
            || dot_all.(i) <> Some true)
          (Array.to_list each_scalar)
      in
      equal (list string) [] (List.map (Printf.sprintf "U+%04X") wrong))

let case_folding =
  test "(?i) closes each simple case-folding orbit" (fun () ->
      let wrong = ref [] in
      List.iter
        (fun orbit ->
          let rows = Array.of_list (List.map (fun u -> Some (utf_8 u)) orbit) in
          List.iter
            (fun u ->
              let got = matches (regex (Printf.sprintf "(?i)^%s$" (utf_8 u))) rows in
              List.iteri
                (fun i v ->
                  if got.(i) <> Some true then
                    wrong :=
                      Printf.sprintf "U+%04X folds with U+%04X" u v :: !wrong)
                orbit)
            orbit)
        orbits;
      equal (list string) [] (List.rev !wrong))

let case_negation =
  cases "(?i) closes a class before it negates"
    ~id:(fun (p, t, _) -> p ^ " on " ^ t)
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
    (fun (p, t, expected) ->
      equal bools [| Some expected |] (matches (regex p) [| Some t |]))

(* The language *)

let language =
  group "the language table"
    [
      cases "accepted, with the first match of each text"
        ~id:(fun (p, _, _) -> p)
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
          ("\\p{gc=Lu}", "aB", Some "B");
          ("\\p{General_Category=Decimal_Number}", "a5", Some "5");
          ("\\pL", "1a", Some "a");
          ("\\x41", "A", Some "A");
          ("\\x{1F600}", "😀", Some "😀");
          ("\\n", "\n", Some "\n");
          ("\\.", "a.b", Some ".");
          ("^abc$", "abc", Some "abc");
          ("(?m)^b$", "a\nb", Some "b");
          ("\\Aab\\z", "ab", Some "ab");
          ("\\bfoo\\b", "a foo b", Some "foo");
          ("\\Bb", "ab", Some "b");
          ("(a)|(b)", "b", Some "b");
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
          ("[]a]", "]", Some "]");
          ("[-a]", "-", Some "-");
          ("[a-]", "-", Some "-");
          ("[\\d_]+", "1_2", Some "1_2");
          ("(a|ab)(c|bcd)", "abcd", Some "abcd");
        ]
        (fun (p, t, expected) ->
          equal (option string) expected (extract (regex p) [| Some t |]).(0));
      cases "(?i) reaches the end of its enclosing group, across alternatives"
        ~id:(fun (p, _, _) -> p)
        [ ("(?:(?i)a|b)c", "Bc", true); ("(?:a(?i)|b)c", "Bc", true) ]
        (fun (p, t, expected) ->
          equal bools [| Some expected |] (matches (regex p) [| Some t |]));
      cases "refused, naming the byte"
        ~id:(fun (p, _, _) -> p)
        [
          ("(?=a)", "lookahead", 0);
          ("(?!a)", "lookahead", 0);
          ("(?<=a)", "lookbehind", 0);
          ("(?<!a)", "lookbehind", 0);
          ("(a)\\1", "backreference", 3);
          ("(?P=n)", "backreference", 0);
          ("\\k<n>", "backreference", 0);
          ("[a&&b]", "intersection", 2);
          ("[a--b]", "difference", 2);
          ("[[ab]]", "nested", 1);
          ("[[:alpha:]]", "POSIX", 1);
          ("a{2}{3}", "quantifier cannot follow", 4);
          ("a*+", "possessive", 2);
          ("(?>a)", "atomic", 0);
          ("(?x)a", "not supported", 2);
          ("(?#c)", "comments", 0);
          ("\\X", "\\X", 0);
          ("\\R", "\\R", 0);
          ("\\C", "\\C", 0);
          ("\\Qa", "Str.quote", 0);
          ("*", "nothing to repeat", 0);
          ("a{", "expected a number", 2);
          ("a{2,1}", "min above max", 1);
          ("(a", "unclosed group", 2);
          ("a)", "unmatched", 1);
          ("[a", "unclosed character class", 0);
          ("[b-a]", "reversed", 2);
          ("\\p{NotAProperty}", "unknown Unicode property", 0);
          ("\\x{D800}", "scalar", 2);
          ("\\x{110000}", "scalar", 2);
          ("(?<a>b)(?<a>c)", "duplicate", 7);
          ("\\b*", "", -1);
        ]
        (fun (p, sub, at) ->
          let raised =
            try
              ignore (regex p);
              None
            with Invalid_argument m -> Some m
          in
          if at < 0 then equal (option string) None raised
          else (
            let m = require_some raised in
            contains ~sub m;
            contains ~sub:(Printf.sprintf "byte %d" at) m));
      test "a pattern that is not valid UTF-8 is refused" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"not valid UTF-8")
            (fun () -> regex "a\xff"));
    ]

(* Empty matches and the rest of the semantics *)

let rule_3 =
  cases "the empty-match rule, from Polars, DuckDB and Go"
    ~id:(fun (p, t, _) -> p ^ " on " ^ t)
    [
      ("x*", "abxd", "-a-b-d-");
      ("", "é.x", "-é-.-x-");
      ("", "", "-");
      ("a||b", "ab", "-b-");
      ("||b", "b", "-b-");
      ("a*?", "baaa", "-b-a-a-a-");
      ("(?:|a)*", "aaa", "-a-a-a-");
      ("\\b", "ab cd", "-ab- -cd-");
      ("\\B", "ab", "a-b");
      ("$", "ab", "ab-");
      ("^", "ab", "-ab");
      ("\\b|", "ab", "-a-b-");
    ]
    (fun (p, t, expected) ->
      equal same_rows [| Some expected |] (replace (regex p) ~by:"-" [| Some t |]))

let split_cases =
  cases "split and find_all follow the same matches"
    ~id:(fun (p, t, _, _) -> p ^ " on " ^ t)
    [
      ("x*", "ab", [| ""; "a"; "b"; "" |], [| ""; ""; "" |]);
      ("", "é.x", [| ""; "é"; "."; "x"; "" |], [| ""; ""; ""; "" |]);
      ("a||b", "ab", [| ""; "b"; "" |], [| "a"; "" |]);
      (",", "a,,b", [| "a"; ""; "b" |], [| ","; "," |]);
      ("z", "ab", [| "ab" |], [||]);
      (",", "", [| "" |], [||]);
    ]
    (fun (p, t, pieces, found) ->
      let r = regex p in
      equal lists [| Some pieces |] (split r [| Some t |]);
      equal lists [| Some found |] (find_all r [| Some t |]))

let empty_class =
  cases "an empty class matches nothing"
    ~id:(fun (p, _) -> p)
    [
      ("[^\\x{0}-\\x{10FFFF}]", "");
      ("\\p{Cs}", "\xed\xa0\x80" (* no valid UTF-8: only its mention *));
      ("\\P{any}", "x");
      ("[^\\P{Lu}\\p{Lu}]", "A");
      ("(?i)[^\\P{Lu}\\p{Ll}\\p{Lu}]", "x");
    ]
    (fun (p, t) ->
      if not (String.is_valid_utf_8 t) then
        (* Compiling is what is checked: [t] states the data the class meant. *)
        ignore (regex p)
      else begin
        let r = regex p in
        equal bools [| Some false |] (matches r [| Some t |]);
        equal lists [| Some [||] |] (find_all r [| Some t |]);
        equal same_rows [| None |] (extract r [| Some t |]);
        equal same_rows [| Some t |] (replace r ~by:"x" [| Some t |])
      end)

let nulls =
  group "nulls"
    [
      test "extract is null without a match or without the group" (fun () ->
          let r = regex "(a)|(b)" in
          equal same_rows [| Some "a"; None; None |]
            (extract ~group:1 r [| Some "a"; Some "b"; Some "c" |]);
          equal same_rows [| Some "" |] (extract (regex "x*") [| Some "ab" |]));
      test "a null row is null in every operation" (fun () ->
          let r = regex "a" in
          let vs = [| None; Some "a" |] in
          equal same_rows [| None; Some "a" |] (extract r vs);
          equal lists [| None; Some [| "a" |] |] (find_all r vs);
          equal lists [| None; Some [| ""; "" |] |] (split r vs);
          equal same_rows [| None; Some "b" |] (replace r ~by:"b" vs);
          equal bools [| None; Some true |] (matches r vs));
      test "an out-of-range group raises at the call" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"no group")
            (fun () -> Expr.Str.extract ~group:2 (regex "(a)") s);
          raises_match
            (Exn.invalid_arg ~sub:"no group")
            (fun () -> Expr.Str.extract ~group:1 (literal "a") s));
    ]

let templates =
  group "replace"
    [
      cases "templates"
        ~id:(fun (p, by, _, _) -> p ^ " by " ^ by)
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
        (fun (p, by, t, expected) ->
          equal same_rows [| Some expected |] (replace (regex p) ~by [| Some t |]));
      cases "bad templates raise at the call"
        ~id:Fun.id
        [ "$"; "$x"; "$2"; "${"; "${n}"; "$ " ]
        (fun by ->
          raises_match
            (Exn.invalid_arg ~sub:"Expr.Str.replace")
            (fun () -> Expr.Str.replace (regex "(a)") ~by s));
      test "by is literal text unless the pattern is a regex" (fun () ->
          equal same_rows [| Some "$0" |]
            (replace (literal "a") ~by:"$0" [| Some "a" |]));
      test "a row with no match is unchanged" (fun () ->
          equal same_rows [| Some "cat" |]
            (replace (regex "z+") ~by:"x" [| Some "cat" |]));
    ]

(* A pathological pattern completes, in a time bounded by the row. *)
let pathological =
  cases "a pathological pattern stays linear"
    ~id:Fun.id
    [ "(a+)+b"; "(a|a)*c"; "(a*)*b"; "(x+x+)+y"; "^(([a-z])+.)+[A-Z]([a-z])+$" ]
    (fun p ->
      let row = String.make 50_000 'a' in
      let r = regex p in
      equal bools [| Some false |] (matches r [| Some row |]);
      equal same_rows [| None |] (extract r [| Some row |]))

(* Laws *)

let scalar_token =
  let strs = [ "a"; "b"; "é"; "日"; "😀"; " "; "\n"; "k"; "ſ"; "K"; "ς" ] in
  Gen.of_list ~pp:(fun ppf -> Format.fprintf ppf "%S") strs

let text_gen =
  Gen.map (String.concat "") (Gen.list ~size:(Gen.int_range 0 10) scalar_token)

let atom_gen =
  Gen.of_list ~pp:Format.pp_print_string
    [ "a"; "b"; "é"; "日"; "😀"; "."; "\\d"; "\\w"; "\\s"; "\\W"; "[ab]";
      "[^a]"; "[a-z]"; "^"; "$"; "\\b"; "\\B"; "\\A"; "\\z"; "(?i)k"; "(?m)^";
      "(?s)."; "[\\w]" ]

let pattern_gen =
  let open Gen in
  let rec go depth =
    if depth = 0 then atom_gen
    else
      frequency
        [
          (3, atom_gen);
          (2, map2 ( ^ ) (go (depth - 1)) (go (depth - 1)));
          (1, map2 (fun a b -> a ^ "|" ^ b) (go (depth - 1)) (go (depth - 1)));
          (2, map (fun a -> "(?:" ^ a ^ ")*") (go (depth - 1)));
          (2, map (fun a -> "(?:" ^ a ^ ")+?") (go (depth - 1)));
          (1, map (fun a -> "(" ^ a ^ ")?") (go (depth - 1)));
          (1, map (fun a -> "(?:" ^ a ^ "){1,2}") (go (depth - 1)));
        ]
  and map2 f a b =
    bind a (fun x -> map (fun y -> f x y) b)
  in
  bind (int_range 0 3) go |> with_pp (fun ppf -> Format.fprintf ppf "%S")

let boundaries t =
  let rec go i acc =
    if i >= String.length t then List.rev (i :: acc)
    else
      go (i + Uchar.utf_decode_length (String.get_utf_8_uchar t i)) (i :: acc)
  in
  go 0 []

let pattern_and_text =
  Gen.pair
    (Gen.such_that
       (fun p ->
         try
           ignore (regex p);
           true
         with Invalid_argument _ -> false)
       pattern_gen)
    text_gen

let pike_agrees r t =
  let suffixes =
    List.map (fun i -> String.sub t i (String.length t - i)) (boundaries t)
    |> List.map Option.some |> Array.of_list
  in
  equal bools
    (Array.map (Option.map Option.is_some) (Array.map Option.some (extract r suffixes)))
    (matches r suffixes)

let dfa_equals_pike =
  prop "the kernel of Str.matches agrees with the kernel of Str.extract"
    pattern_and_text (fun (p, t) ->
      cover "a non-matching text" (not (Option.get (matches (regex p) [| Some t |]).(0)));
      pike_agrees (regex p) t)

(* The DFA's cache of [limit] states forced small, so it clears inside a row. *)
let dfa_cache_agrees limit =
  prop
    (Printf.sprintf "the DFA with a cache of %d states agrees with the Pike VM"
       limit)
    pattern_and_text
    (fun (p, t) ->
      let r = regex p in
      match Expr.Str.dfa_test ~limit r [| t |] with
      | None -> ()
      | Some got ->
          equal (option string)
            (if got.(0) then (extract r [| Some t |]).(0) else None)
            (extract r [| Some t |]).(0))

let dfa_clears =
  test "a pattern of many states clears the cache with the right answers"
    (fun () ->
      (* A window of two letters on a random text of [a] and [b] makes about
         2^14 distinct DFA states: more than the cache holds. *)
      let window = 14 in
      let p =
        Printf.sprintf "(?:a|b)*a(?:a|b){%d}c" window
      in
      let rng = Random.State.make [| 7 |] in
      let row n =
        String.init n (fun _ -> if Random.State.bool rng then 'a' else 'b')
      in
      let rows = Array.init 40 (fun i -> row (200 + (i * 37))) in
      rows.(11) <- rows.(11) ^ "c";
      let r = regex p in
      let dfa = Option.get (Expr.Str.dfa_test ~limit:16 r rows) in
      let pike =
        Array.map (function Some _ -> true | None -> false)
          (extract r (Array.map Option.some rows))
      in
      equal ~msg:"some row matches" bool true (Array.exists Fun.id pike);
      equal (array bool) pike dfa)

let round_trip =
  prop "split's pieces interleaved with find_all's matches are the row"
    pattern_and_text (fun (p, t) ->
      let r = regex p in
      let pieces = Option.get (split r [| Some t |]).(0) in
      let matches = Option.get (find_all r [| Some t |]).(0) in
      equal int (Array.length matches + 1) (Array.length pieces);
      let b = Buffer.create 16 in
      Array.iteri
        (fun i piece ->
          Buffer.add_string b piece;
          if i < Array.length matches then Buffer.add_string b matches.(i))
        pieces;
      equal string t (Buffer.contents b);
      Array.iter (fun piece -> equal bool true (String.is_valid_utf_8 piece)) pieces;
      Array.iter (fun m -> equal bool true (String.is_valid_utf_8 m)) matches)

let quote_equals_literal =
  let plain = Gen.such_that (fun s -> s <> "") text_gen in
  prop "a literal is its quoted text, for every operation"
    (Gen.pair plain text_gen)
    (fun (p, t) ->
      let by = "<x>" in
      let rows = [| Some t |] in
      let q = regex (Expr.Str.quote p) and l = literal p in
      equal bools (matches l rows) (matches q rows);
      equal lists (find_all l rows) (find_all q rows);
      equal lists (split l rows) (split q rows);
      equal same_rows (extract l rows) (extract q rows);
      equal same_rows (replace l ~by rows) (replace q ~by rows))

let quote_escapes =
  prop "a quoted text matches only itself"
    (Gen.pair text_gen text_gen)
    (fun (p, t) ->
      assume (p <> "");
      let q = regex (Expr.Str.quote p) in
      equal bools
        [| Some (Option.is_some (Expr.Str.pieces [ p ] |> fun pat ->
              (extract pat [| Some t |]).(0))) |]
        (matches q [| Some t |]))

let prefix_suffix_pieces =
  test "prefix, suffix and pieces are the regexes they name" (fun () ->
      let rows = [| Some "abcab"; Some "b"; Some "" |] in
      let check pat r = equal same_rows (extract pat rows) (extract r rows) in
      check (Expr.Str.prefix "ab") (regex "\\Aab");
      check (Expr.Str.suffix "ab") (regex "ab\\z");
      check (Expr.Str.pieces [ "a"; "c" ]) (regex "a(?s:.*?)c"))

let () =
  exit
  @@ run "Regex"
       [
         group "against the reference"
           [ golden_agreement ];
         group "Unicode 18.0.0"
           [
             classes_boundaries;
             alias_names;
             annex_c;
             dot_is_every_scalar;
             case_folding;
             case_negation;
           ];
         language;
         rule_3;
         split_cases;
         empty_class;
         nulls;
         templates;
         pathological;
         group "laws"
           [
             dfa_equals_pike;
             dfa_cache_agrees 2;
             dfa_cache_agrees 16;
             dfa_clears;
             round_trip;
             quote_equals_literal;
             quote_escapes;
             prefix_suffix_pieces;
           ];
       ]
