(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_text
open Text_support

let text_w =
  Testable.make ~pp:Text.pp ~equal:Text.equal
  |> Testable.with_compare Text.compare

let concat2 a b = Text.concat [ a; b ]
let trees = gen_tree ()
let tree2 = Gen.pair trees trees
let tree3 = Gen.triple trees trees trees

(* Valid UTF-8 strings of any scalar values. *)
let utf_8 =
  let encode us =
    let b = Buffer.create 16 in
    List.iter (Buffer.add_utf_8_uchar b) us;
    Buffer.contents b
  in
  Gen.with_pp
    (fun ppf s -> Format.fprintf ppf "%S" s)
    (Gen.map encode (Gen.list ~size:(Gen.int_range 0 8) Gen.uchar))

(* Strings *)

(* Maximal subparts, from the Unicode Standard, section 3.9, "U+FFFD
   Substitution of Maximal Subparts". *)
let repairs =
  [
    ("a truncated sequence before a character", "\xE2\x82a", "\u{FFFD}a");
    ("a byte that never starts a sequence", "\xFF", "\u{FFFD}");
    ("an overlong NUL", "\xC0\x80", "\u{FFFD}\u{FFFD}");
    ("an encoded surrogate", "\xED\xA0\x80", "\u{FFFD}\u{FFFD}\u{FFFD}");
    ("a truncated four-byte sequence", "\xF0\x9F\x98", "\u{FFFD}");
    ( "a sequence past U+10FFFF",
      "\xF4\x90\x80\x80",
      "\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}" );
    ( "the standard's example",
      "a\xF1\x80\x80\xE1\x80\xC2b\x80c\x80\xBFd",
      "a\u{FFFD}\u{FFFD}\u{FFFD}b\u{FFFD}c\u{FFFD}\u{FFFD}d" );
  ]

(* The decoding the contract names, as a reference. *)
let decoded s =
  let b = Buffer.create (String.length s) in
  let rec loop i =
    if i < String.length s then begin
      let d = String.get_utf_8_uchar s i in
      Buffer.add_utf_8_uchar b (Uchar.utf_decode_uchar d);
      loop (i + Uchar.utf_decode_length d)
    end
  in
  loop 0;
  Buffer.contents b

let strings =
  group "v"
    [
      cases
        ~name:(fun (name, _, _) -> name)
        "replaces each maximal invalid subpart with U+FFFD" repairs
        (fun (_, s, repaired) -> equal text_w (Text.v repaired) (Text.v s));
      prop "agrees with String.get_utf_8_uchar on any bytes" Gen.string
        (fun s ->
          classify "invalid" (not (String.is_valid_utf_8 s));
          equal text_w (Text.v (decoded s)) (Text.v s));
      prop "of a concatenation is the concatenation, split at a character"
        (Gen.pair utf_8 utf_8) (fun ab ->
          Law.homomorphic string text_w Text.v ( ^ ) concat2 ab);
      test "keeps newlines and control characters" (fun () ->
          not_equal text_w (Text.v "ab") (Text.v "a\nb");
          not_equal text_w (Text.v "") (Text.v "\x00");
          not_equal text_w (Text.v "ab") (Text.v "a\tb"));
    ]

(* Concatenation *)

let concatenation =
  group "concat"
    [
      test "of nothing is the empty text" (fun () ->
          equal text_w (Text.v "") (Text.concat []));
      test "keeps the order of its texts" (fun () ->
          not_equal text_w
            (Text.concat [ Text.v "a"; Text.v "b" ])
            (Text.concat [ Text.v "b"; Text.v "a" ]));
      prop "is associative" tree3 (fun (a, b, c) ->
          Law.associative text_w concat2 (text a, text b, text c));
      prop "has the empty text as its unit" trees (fun t ->
          Law.neutral text_w concat2 (Text.v "") (text t));
    ]

(* Styles *)

let powers_of_two = Gen.map (fun e -> Float.ldexp 1. e) (Gen.int_range (-8) 8)

let styles =
  group "styles"
    [
      prop "bold is idempotent" trees (fun t ->
          Law.idempotent text_w Text.bold (text t));
      prop "italic is idempotent" trees (fun t ->
          Law.idempotent text_w Text.italic (text t));
      prop "color is idempotent" trees (fun t ->
          Law.idempotent text_w (Text.color Color.red) (text t));
      prop "bold and italic commute" trees (fun t ->
          Law.commutes text_w Text.bold Text.italic (text t));
      prop "bold and a script commute" trees (fun t ->
          Law.commutes text_w Text.bold Text.sup (text t));
      prop "bold distributes over concat" tree2 (fun (a, b) ->
          Law.homomorphic text_w text_w Text.bold concat2 concat2
            (text a, text b));
      prop "color distributes over concat" tree2 (fun (a, b) ->
          Law.homomorphic text_w text_w (Text.color Color.red) concat2 concat2
            (text a, text b));
      prop "scale distributes over concat" tree2 (fun (a, b) ->
          Law.homomorphic text_w text_w (Text.scale 2.) concat2 concat2
            (text a, text b));
      prop "sup distributes over concat" tree2 (fun (a, b) ->
          Law.homomorphic text_w text_w Text.sup concat2 concat2 (text a, text b));
      prop "an inner colour wins" trees (fun t ->
          let t = text t in
          equal text_w (Text.color Color.blue t)
            (Text.color Color.red (Text.color Color.blue t)));
      prop "colours only the characters without one" tree2 (fun (a, b) ->
          let a = text a and b = text b in
          equal text_w
            (Text.concat [ Text.color Color.red a; Text.color Color.blue b ])
            (Text.color Color.red (Text.concat [ a; Text.color Color.blue b ])));
      prop "scales compose by their product"
        (Gen.triple powers_of_two powers_of_two trees) (fun (s, s', t) ->
          let t = text t in
          equal text_w (Text.scale (s *. s') t) (Text.scale s (Text.scale s' t)));
      prop "scale 1. changes nothing" trees (fun t ->
          equal text_w (text t) (Text.scale 1. (text t)));
      test "a script differs from its text" (fun () ->
          not_equal text_w (Text.v "2") (Text.sup (Text.v "2"));
          not_equal text_w (Text.v "2") (Text.sub (Text.v "2"));
          not_equal text_w (Text.sub (Text.v "2")) (Text.sup (Text.v "2")));
      test "a script of a script is not commutative" (fun () ->
          not_equal text_w
            (Text.sub (Text.sup (Text.v "i")))
            (Text.sup (Text.sub (Text.v "i"))));
      test "a superscript is not a reduced text" (fun () ->
          not_equal text_w (Text.scale 0.7 (Text.v "2")) (Text.sup (Text.v "2")));
      cases ~name:(Printf.sprintf "scale %h raises") "scale rejects"
        [ 0.; -0.; -1.; Float.nan; Float.infinity; Float.neg_infinity ]
        (fun s ->
          raises_match Exn.invalid_arg (fun () -> Text.scale s (Text.v "a")));
      test "scale accepts the extremes of the positive floats" (fun () ->
          ignore (Text.scale Float.min_float (Text.v "a"));
          ignore (Text.scale 5e-324 (Text.v "a"));
          ignore (Text.scale Float.max_float (Text.v "a")));
    ]

(* Comparing and formatting *)

(* Formatting *)

let pp_at margin t =
  let b = Buffer.create 64 in
  let ppf = Format.formatter_of_buffer b in
  Format.pp_set_margin ppf margin;
  Format.fprintf ppf "%a@?" Text.pp t;
  Buffer.contents b

(* Each end of the two control ranges and its neighbour outside it. *)
let literal_ends =
  [
    ("U+0000", "\u{0}", {|"\000"|});
    ("U+001F", "\u{1F}", {|"\031"|});
    ("U+0020", " ", {|" "|});
    ("U+007E", "~", {|"~"|});
    ("U+009F", "\u{9F}", {|"\194\159"|});
    ("U+00A0", "\u{A0}", "\"\u{A0}\"");
    ("U+00AD", "\u{AD}", "\"\u{AD}\"");
  ]

(* The literal of [wide] is 8 columns, two quotes, two characters and two
   escapes, and 12 bytes, so the one-line form is 26 columns and fits a margin
   of 27 but not of 26. *)
let column_count () =
  let wide = Text.(concat [ v "中中\"\""; bold (v "x") ]) in
  equal string {|(text "中中\"\"" ("x" bold))|} (pp_at 27 wide);
  equal string "(text \"中中\\\"\\\"\"\n (\"x\" bold))" (pp_at 26 wide)

let literal_chars =
  Gen.of_list
    ~pp:(fun ppf s -> Format.fprintf ppf "%S" s)
    [
      "\u{0}";
      "\n";
      "\u{1F}";
      " ";
      "a";
      "\"";
      "\\";
      "~";
      "\u{7F}";
      "\u{9F}";
      "\u{A0}";
      "\u{AD}";
      "é";
      "中";
      "\u{2028}";
      "\u{1F600}";
    ]

let non_empty =
  Gen.with_pp
    (fun ppf s -> Format.fprintf ppf "%S" s)
    (Gen.map (String.concat "")
       (Gen.list ~size:(Gen.int_range 1 8) literal_chars))

let comparing =
  group "equal and compare"
    [
      prop "equal is an equivalence" tree2 (fun (a, b) ->
          Law.equivalence text_w (text a, text b));
      prop "compare is a total order agreeing with equal" tree3
        (fun (a, b, c) -> Law.order text_w (text a, text b, text c));
      test "a text split in two equals the text" (fun () ->
          equal text_w (Text.v "ab") (Text.concat [ Text.v "a"; Text.v "b" ]));
      test "styled pieces equal the styled whole" (fun () ->
          equal text_w
            (Text.bold (Text.v "ab"))
            (Text.concat [ Text.bold (Text.v "a"); Text.bold (Text.v "b") ]);
          equal text_w
            (Text.bold (Text.v "ab"))
            (Text.bold (Text.concat [ Text.v "a"; Text.bold (Text.v "b") ])));
      test "colours compare by value" (fun () ->
          equal text_w
            (Text.color Color.red (Text.v "a"))
            (Text.color (Color.v 1. 0. 0.) (Text.v "a")));
      cases
        ~name:(fun (name, _, _) -> name)
        "texts differing in one style are unequal"
        [
          ("bold", Text.v "a", Text.bold (Text.v "a"));
          ("italic", Text.v "a", Text.italic (Text.v "a"));
          ("bold and italic", Text.bold (Text.v "a"), Text.italic (Text.v "a"));
          ("size", Text.v "a", Text.scale 2. (Text.v "a"));
          ( "colour",
            Text.color Color.red (Text.v "a"),
            Text.color Color.blue (Text.v "a") );
          ("no colour", Text.v "a", Text.color Color.black (Text.v "a"));
          ("character", Text.v "a", Text.v "b");
          ("emptiness", Text.v "", Text.v " ");
        ]
        (fun (_, a, b) ->
          not_equal text_w a b;
          not_equal int 0 (Text.compare a b));
      test "formats spans with their styles" (fun () ->
          let t =
            Text.(
              scale 2.
                (concat
                   [
                     v "x";
                     sup (v "2");
                     sub (sup (bold (v "i")));
                     color Color.red (italic (v " ok"));
                   ]))
          in
          expect (Format.asprintf "%a" Text.pp t)
          @@ __POS_OF__
               {|
            (text ("x" size 2) ("2" size 1.4 shift -0.8) ("i" bold size 0.98 shift -0.16)
             (" ok" italic size 2 #ff0000))
            |});
      test "formats a plain text as its string" (fun () ->
          expect (Format.asprintf "%a" Text.pp (Text.v "step size \u{03B7}"))
          @@ __POS_OF__ {| (text "step size η") |});
      test "formats strings escaping only quotes, backslashes and controls"
        (fun () ->
          equal string {|(text "a\"b\\c\td\194\133é中" ("\"" bold))|}
            (Format.asprintf "%a" Text.pp
               Text.(concat [ v "a\"b\\c\td\u{85}é中"; bold (v "\"") ])));
      cases
        ~name:(fun (n, _, _) -> n)
        "formats at the ends of the controls" literal_ends
        (fun (_, s, lit) ->
          equal string
            ("(text " ^ lit ^ ")")
            (Format.asprintf "%a" Text.pp (Text.v s)));
      test "breaks lines counting a character as one column" column_count;
      prop "formats a literal that reads back as its string" non_empty (fun s ->
          equal string s
            (Scanf.sscanf
               (Format.asprintf "%a" Text.pp (Text.v s))
               "(text %S)" Fun.id));
    ]

let () =
  exit
    (run "hugin.text: Text" [ strings; concatenation; styles; comparing ])
