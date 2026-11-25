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

(* Texts *)

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

let texts =
  group "v"
    [
      cases "replaces each maximal invalid subpart with U+FFFD"
        ~id:(fun (name, _, _) -> name)
        repairs
        (fun (_, s, repaired) -> equal text_w (Text.v repaired) (Text.v s));
      prop "agrees with String.get_utf_8_uchar on any bytes" Gen.string
        (fun s ->
          cover "invalid" (not (String.is_valid_utf_8 s));
          equal text_w (Text.v (decoded s)) (Text.v s));
      prop "of a concatenation is the concatenation, split at a character"
        (Gen.pair utf_8 utf_8) (fun ab ->
          Law.homomorphic string text_w Text.v ( ^ ) concat2 ab);
      test "keeps newlines and control characters" (fun () ->
          not_equal text_w (Text.v "ab") (Text.v "a\nb");
          not_equal text_w (Text.v "") (Text.v "\x00");
          not_equal text_w (Text.v "ab") (Text.v "a\tb"));
    ]

let concatenation =
  group "concat"
    [
      test "of nothing is the empty text" (fun () ->
          equal text_w (Text.v "") (Text.concat []));
      prop "is associative" tree3 (fun (a, b, c) ->
          Law.associative text_w concat2 (text a, text b, text c));
      prop "has the empty text as its unit" trees (fun t ->
          Law.neutral text_w concat2 (Text.v "") (text t));
    ]

(* Styles *)

let one_of styles = Gen.of_list ~pp:pp_style styles
let name style = Format.asprintf "%a" pp_style style

(* The styles that a second application leaves as the first set them. *)
let idempotent = [ Bold; Italic; Color Color.red; Color Color.blue ]

let styling =
  group "styles"
    [
      prop ~count:200 "each combinator distributes over concat"
        (Gen.pair (one_of styles) tree2)
        (fun (style, (a, b)) ->
          List.iter (fun s -> cover (name s) (s == style)) styles;
          Law.homomorphic text_w text_w (apply style) concat2 concat2
            (text a, text b));
      prop "bold, italic and color are idempotent"
        (Gen.pair (one_of idempotent) trees)
        (fun (style, t) ->
          List.iter (fun s -> cover (name s) (s == style)) idempotent;
          Law.idempotent text_w (apply style) (text t));
      prop "bold and italic commute" trees (fun t ->
          Law.commutes text_w Text.bold Text.italic (text t));
      prop "bold and a script commute" trees (fun t ->
          Law.commutes text_w Text.bold Text.sup (text t));
      prop "an inner colour wins over an outer one" tree2 (fun (a, b) ->
          let a = text a and b = text b in
          equal text_w
            (Text.concat [ Text.color Color.red a; Text.color Color.blue b ])
            (Text.color Color.red (Text.concat [ a; Text.color Color.blue b ])));
    ]

let powers_of_two = Gen.map (fun e -> Float.ldexp 1. e) (Gen.int_range (-8) 8)

let scaling =
  group "scale"
    [
      prop "composes by the product of its factors"
        (Gen.triple powers_of_two powers_of_two trees) (fun (s, s', t) ->
          let t = text t in
          equal text_w (Text.scale (s *. s') t) (Text.scale s (Text.scale s' t)));
      prop "by 1. changes nothing" trees (fun t ->
          equal text_w (text t) (Text.scale 1. (text t)));
      cases "raises on a factor that is not finite and positive"
        ~id:(Printf.sprintf "%h")
        [ 0.; -0.; -1.; Float.nan; Float.infinity; Float.neg_infinity ]
        (fun s ->
          raises_match Exn.invalid_arg (fun () -> Text.scale s (Text.v "a")));
      cases "accepts the extremes of the positive floats"
        ~id:(Printf.sprintf "%h") [ Float.min_float; 5e-324; Float.max_float ]
        (fun s -> ignore (Text.scale s (Text.v "a")));
    ]

(* Comparing and formatting *)

(* Pairs of texts that differ in one style or character. *)
let one_style =
  [
    ("bold", Str "a", Style (Bold, Str "a"));
    ("italic", Str "a", Style (Italic, Str "a"));
    ("bold and italic", Style (Bold, Str "a"), Style (Italic, Str "a"));
    ("size", Str "a", Style (Scale 2., Str "a"));
    ( "colour",
      Style (Color Color.red, Str "a"),
      Style (Color Color.blue, Str "a") );
    ("no colour", Str "a", Style (Color Color.black, Str "a"));
    ("character", Str "a", Str "b");
    ("emptiness", Str "", Str " ");
  ]

let equality =
  group "equal"
    [
      prop "is an equivalence" tree2 (fun (a, b) ->
          Law.equivalence text_w (text a, text b));
      test "compares colours by value" (fun () ->
          equal text_w
            (Text.color Color.red (Text.v "a"))
            (Text.color (Color.v 1. 0. 0.) (Text.v "a")));
      cases "is false for texts differing in one style"
        ~id:(fun (name, _, _) -> name)
        one_style
        (fun (_, a, b) -> not_equal text_w (text a) (text b));
    ]

let ordering =
  group "compare"
    [
      prop "is a total order agreeing with equal"
        ~examples:(List.map (fun (_, a, b) -> (a, b, a)) one_style)
        tree3
        (fun (a, b, c) -> Law.order text_w (text a, text b, text c));
    ]

let show t = Format.asprintf "%a" Text.pp t

let show_at margin t =
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
    ("U+007F", "\u{7F}", {|"\127"|});
    ("U+009F", "\u{9F}", {|"\194\159"|});
    ("U+00A0", "\u{A0}", "\"\u{A0}\"");
    ("U+00AD", "\u{AD}", "\"\u{AD}\"");
  ]

(* The literal of [wide] is 8 columns, two quotes, two characters and two
   escapes, and 12 bytes, so the one-line form is 26 columns and fits a margin
   of 27 but not of 26. *)
let column_count () =
  let wide = Text.(concat [ v "中中\"\""; bold (v "x") ]) in
  equal string {|(text "中中\"\"" ("x" bold))|} (show_at 27 wide);
  equal string "(text \"中中\\\"\\\"\"\n (\"x\" bold))" (show_at 26 wide)

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

let formatting =
  group "pp"
    [
      test "formats spans of one style" (fun () ->
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
          expect (show t)
          @@ __POS_OF__
               {|
            (text ("x" size 2) ("2" size 1.4 shift -0.8) ("i" bold size 0.98 shift -0.16)
             (" ok" italic size 2 #ff0000))
            |});
      test "formats a plain text as its string" (fun () ->
          expect (show (Text.v "step size \u{03B7}"))
          @@ __POS_OF__ {| (text "step size η") |});
      test "escapes only double quotes, backslashes and control characters"
        (fun () ->
          equal string {|(text "a\"b\\c\td\194\133é中" ("\"" bold))|}
            (show Text.(concat [ v "a\"b\\c\td\u{85}é中"; bold (v "\"") ])));
      cases "escapes the ends of the control ranges and not their neighbours"
        ~id:(fun (n, _, _) -> n)
        literal_ends
        (fun (_, s, lit) ->
          equal string ("(text " ^ lit ^ ")") (show (Text.v s)));
      test "breaks lines counting a character as one column" column_count;
      prop "writes a literal that reads back as its string" non_empty (fun s ->
          equal string s (Scanf.sscanf (show (Text.v s)) "(text %S)" Fun.id));
    ]

let () =
  exit
    (run "hugin.text: Text"
       [
         texts; concatenation; styling; scaling; equality; ordering; formatting;
       ])
