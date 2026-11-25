(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Expected texts marked "Hugging Face" are tokenizers 0.23.1's: the normalizer
   of the same name in [tokenizers.normalizers], given the same text. *)

open Windtrap
open Brot

let marker = "\u{2581}"

let utf8 u =
  let b = Buffer.create 4 in
  Buffer.add_utf_8_uchar b u;
  Buffer.contents b

(* A text made of pieces drawn from [alphabet], so that each piece lands at
   every offset of the eight-byte words some normalizers scan by. *)
let quoted ppf s = Format.fprintf ppf "%S" s
let piece_of alphabet = Gen.of_list ~pp:quoted alphabet

let text_of ?(size = Gen.int_range 0 12) alphabet =
  Gen.with_pp quoted
    (Gen.map (String.concat "") (Gen.list ~size (piece_of alphabet)))

let text_id = Printf.sprintf "%S"
let row_id (text, _) = Printf.sprintf "%S" text

let normalizes n (text, expected) =
  equal ~msg:(Printf.sprintf "%S" text) string expected
    (Normalizer.apply n text)

(* The span every character of the normalized text reports on the input, in the
   order the characters come out, as ["start,stop"] separated by spaces. *)
let char_spans n input =
  let normalized, alignment = Normalizer.apply_aligned n input in
  let spans = ref [] and i = ref 0 in
  while !i < String.length normalized do
    let len = Uchar.utf_decode_length (String.get_utf_8_uchar normalized !i) in
    let start, stop =
      Normalizer.original_span alignment ~start:!i ~stop:(!i + len)
    in
    spans := Printf.sprintf "%d,%d" start stop :: !spans;
    i := !i + len
  done;
  (normalized, String.concat " " (List.rev !spans))

(* Every Unicode scalar value [u], with ["a" ^ u ^ "b"] normalized by [n], gives
   ["a" ^ expected u ^ "b"] ([before] and [after] stand for ["a"] and ["b"] when
   given). The reference reads uucp, at Unicode 18.0.0, or the sets brot.mli
   lists. *)
let maps_every_scalar ?(before = "a") ?(after = "b") n expected =
  for code = 0 to 0x10FFFF do
    if Uchar.is_valid code then begin
      let u = Uchar.of_int code in
      let want = before ^ expected u ^ after in
      let got = Normalizer.apply n (before ^ utf8 u ^ after) in
      if got <> want then
        equal ~msg:(Printf.sprintf "U+%04X" code) string want got
    end
  done

let to_lower u =
  match Uucp.Case.Map.to_lower u with
  | `Self -> utf8 u
  | `Uchars us -> String.concat "" (List.map utf8 us)

let is_mark u =
  match Uucp.Gc.general_category u with `Mn | `Mc | `Me -> true | _ -> false

(* Unicode normalization forms *)

(* Characters whose normalization does something: marks of several combining
   classes, Hangul jamo and syllables (U+11C3 is no trailing jamo), characters
   with a canonical or compatibility decomposition, singletons, composition
   exclusions, decompositions that start with a mark, the parts of compositions
   in Indic and Tibetan scripts, and the Unicode 16 starters of Tulu-Tigalari,
   Gurung Khema and Kirat Rai whose decompositions compose backward. *)
let nf_alphabet =
  [ "a"; "e"; "A"; "d"; "s"; " "; "\u{E9}"; "\u{301}"; "\u{300}"; "\u{302}";
    "\u{316}"; "\u{323}"; "\u{327}"; "\u{307}"; "\u{345}"; "\u{344}";
    "\u{1100}"; "\u{1161}"; "\u{11A8}"; "\u{11C2}"; "\u{11C3}"; "\u{AC00}";
    "\u{AC01}"; "\u{D7A3}"; "\u{1E9B}"; "\u{1E0D}"; "\u{17F}"; "\u{212B}";
    "\u{2126}"; "\u{FB01}"; "\u{2460}"; "\u{FF21}"; "\u{BD}"; "\u{1C5}";
    "\u{FDFA}"; "\u{F73}"; "\u{F71}"; "\u{F72}"; "\u{1D15E}"; "\u{1D165}";
    "\u{1D16D}"; "\u{CCB}"; "\u{CC6}"; "\u{CC2}"; "\u{CD5}"; "\u{9CB}";
    "\u{9C7}"; "\u{9BE}"; "\u{2ADC}"; "\u{958}"; "\u{915}"; "\u{93C}";
    "\u{3099}"; "\u{304B}"; "\u{FF76}"; "\u{FF9E}"; "\u{1138B}"; "\u{113C2}";
    "\u{113C5}"; "\u{113C7}"; "\u{113C8}"; "\u{1611E}"; "\u{16121}";
    "\u{16122}"; "\u{16123}"; "\u{16124}"; "\u{16125}"; "\u{16126}";
    "\u{16127}"; "\u{16128}"; "\u{16D63}"; "\u{16D67}"; "\u{16D68}";
    "\u{16D69}" ]
[@@ocamlformat "disable"]

(* Valid UTF-8 after a run of ASCII: pieces of [nf_alphabet] and any scalar
   value, one to four bytes in equal measure. *)
let nf_text =
  let open Gen in
  with_pp quoted
  @@
  let piece = one_of [ piece_of nf_alphabet; map utf8 uchar ] in
  let+ ascii = string_of ~size:(int_range 0 15) (char_range 'a' 'z')
  and+ pieces = list ~size:(int_range 0 12) piece in
  ascii ^ String.concat "" pieces

(* An accent composed at every offset of a 25-byte text. *)
let composed_at_every_offset =
  List.init 25 (fun k ->
      String.make k 'a' ^ "e\u{301}" ^ String.make (24 - k) 'b')

(* Pairs whose second character decomposes to one that composes with the first:
   U+16123 is U+1611E U+1611F, so U+1611E U+16123 is U+1611E U+1611E U+1611F,
   which composes to U+16126. Every such pair below U+30000, found against uunf
   18.0.0; CPython's unicodedata agrees. *)
let composed_across_a_decomposition =
  [
    "\u{1138B}\u{113C5}"; "\u{1138B}\u{113C7}"; "\u{1138B}\u{113C8}";
    "\u{113C2}\u{113C5}"; "\u{113C2}\u{113C7}"; "\u{113C2}\u{113C8}";
    "\u{1611E}\u{16121}"; "\u{1611E}\u{16122}"; "\u{1611E}\u{16123}";
    "\u{1611E}\u{16124}"; "\u{1611E}\u{16125}"; "\u{1611E}\u{16126}";
    "\u{1611E}\u{16127}"; "\u{1611E}\u{16128}"; "\u{16D63}\u{16D68}";
    "\u{16D67}\u{16D68}"; "\u{16D69}\u{16D68}";
  ]
[@@ocamlformat "disable"]

let agrees_with_uunf form n text =
  equal string (Uunf_string.normalize_utf_8 form text) (Normalizer.apply n text)

let nfc_tests =
  [
    prop ~count:500
      ~examples:(composed_at_every_offset @ composed_across_a_decomposition)
      "is Unicode's NFC" nf_text
      (agrees_with_uunf `NFC Normalizer.nfc);
    (* UAX #15: an LV syllable composes with a trailing jamo of U+11A8 to U+11C2
       by arithmetic, an LVT syllable with none; a mark blocked by one of the
       same combining class stays. Hugging Face agrees. *)
    cases ~id:row_id "composes Hangul by arithmetic and stops at a blocked mark"
      [
        ("\u{1100}\u{1161}\u{11A8}", "\u{AC01}");
        ("\u{AC00}\u{11A8}", "\u{AC01}");
        ("\u{AC01}\u{11A8}", "\u{AC01}\u{11A8}");
        ("\u{AC00}\u{11C3}", "\u{AC00}\u{11C3}");
        ("\u{1100}\u{1161}\u{11C3}", "\u{AC00}\u{11C3}");
        ("a\u{316}\u{323}", "a\u{316}\u{323}");
      ]
      (normalizes Normalizer.nfc);
  ]

let nfd_tests =
  [
    prop ~count:500 "is Unicode's NFD" nf_text
      (agrees_with_uunf `NFD Normalizer.nfd);
  ]

let nfkc_tests =
  [
    prop ~count:500 ~examples:composed_across_a_decomposition
      "is Unicode's NFKC" nf_text
      (agrees_with_uunf `NFKC Normalizer.nfkc);
  ]

let nfkd_tests =
  [
    prop ~count:500 "is Unicode's NFKD" nf_text
      (agrees_with_uunf `NFKD Normalizer.nfkd);
  ]

(* lowercase *)

(* Capitals of every kind, ASCII and not, and characters that lower to
   themselves. *)
let case_alphabet =
  [ "a"; "Z"; "M"; " "; "1"; "\u{C0}"; "\u{C9}"; "\u{130}"; "\u{3A3}";
    "\u{1C5}"; "\u{1E9E}"; "\u{DF}"; "\u{FB01}"; "\u{410}"; "\u{10400}";
    "\u{301}"; "\u{6F22}" ]
[@@ocamlformat "disable"]

(* An ASCII capital at every offset of a 25-byte text. *)
let capital_at_every_offset =
  List.init 25 (fun k -> String.make k 'a' ^ "X" ^ String.make (24 - k) 'b')

let lowercase = Normalizer.apply Normalizer.lowercase

let lowercase_tests =
  [
    test "maps every character to its full lowercase mapping" (fun () ->
        maps_every_scalar Normalizer.lowercase to_lower;
        (* Hugging Face's rows. *)
        List.iter
          (normalizes Normalizer.lowercase)
          [
            ("\u{C0}\u{C9}\u{CE}", "\u{E0}\u{E9}\u{EE}");
            ("\u{130}", "i\u{307}");
            ("\u{1C5}", "\u{1C6}");
            ("\u{1E9E}", "\u{DF}");
            ("\u{3A3}", "\u{3C3}");
          ]);
    cases ~id:row_id
      "lowercases capital sigma to small sigma wherever it stands"
      [
        ( "\u{39F}\u{394}\u{39F}\u{3A3} \u{3A3}\u{391}",
          "\u{3BF}\u{3B4}\u{3BF}\u{3C3} \u{3C3}\u{3B1}" );
      ]
      (normalizes Normalizer.lowercase);
    cases ~id:row_id "leaves \u{DF} and \u{FB01} as they are"
      [
        ("\u{DF}", "\u{DF}"); ("\u{FB01}", "\u{FB01}"); ("\u{FB04}", "\u{FB04}");
      ]
      (normalizes Normalizer.lowercase);
    prop ~count:300 ~examples:capital_at_every_offset
      "lowercases ASCII as String.lowercase_ascii does"
      (Gen.string_of ~size:(Gen.int_range 0 40) (Gen.char_range '\x00' '\x7F'))
      (fun text -> equal string (String.lowercase_ascii text) (lowercase text));
    prop ~count:300 "lowercases each character on its own"
      (Gen.pair (text_of case_alphabet) (text_of case_alphabet))
      (Law.homomorphic string string lowercase ( ^ ) ( ^ ));
  ]

(* strip_accents *)

let strip_accents = Normalizer.apply Normalizer.strip_accents

let mark_alphabet =
  [ "a"; "e"; " "; "\u{E9}"; "\u{301}"; "\u{323}"; "\u{903}"; "\u{93F}";
    "\u{9CB}"; "\u{20DD}"; "\u{93C}"; "\u{915}"; "\u{1D165}"; "\u{6F22}" ]
[@@ocamlformat "disable"]

let strip_accents_tests =
  [
    test "removes every nonspacing, spacing and enclosing mark" (fun () ->
        maps_every_scalar Normalizer.strip_accents (fun u ->
            if is_mark u then "" else utf8 u);
        (* Hugging Face's rows: Bengali vowel sign O and Devanagari vowel sign I
           are spacing marks. *)
        List.iter
          (normalizes Normalizer.strip_accents)
          [
            ("\u{9CB}", "");
            ("\u{93F}", "");
            ("a\u{903}\u{20DD}\u{301}", "a");
            ("\u{915}\u{93C}", "\u{915}");
            ("A\u{301}", "A");
          ]);
    cases ~id:row_id "keeps a precomposed character whole"
      [ ("\u{E1}", "\u{E1}"); ("caf\u{E9}", "caf\u{E9}") ]
      (normalizes Normalizer.strip_accents);
    prop ~count:300 "removes each mark on its own"
      (Gen.pair (text_of mark_alphabet) (text_of mark_alphabet))
      (Law.homomorphic string string strip_accents ( ^ ) ( ^ ));
  ]

(* strip *)

let is_white_space piece =
  piece <> ""
  && Uchar.utf_decode_is_valid (String.get_utf_8_uchar piece 0)
  && Uucp.White.is_white_space
       (Uchar.utf_decode_uchar (String.get_utf_8_uchar piece 0))

(* Characters with the White_Space property, and characters next to them that
   lack it: a zero width space, the Mongolian vowel separator, a byte order
   mark, U+0084 and U+2060. *)
let white_space =
  [ "\t"; "\n"; "\x0B"; "\r"; " "; "\u{85}"; "\u{A0}"; "\u{1680}";
    "\u{2000}"; "\u{200A}"; "\u{2028}"; "\u{202F}"; "\u{3000}" ]
[@@ocamlformat "disable"]

let not_white_space =
  [ "a"; "\u{E9}"; "\u{200B}"; "\u{180E}"; "\u{FEFF}"; "\u{84}"; "\u{2060}";
    "\xFF" ]
[@@ocamlformat "disable"]

let strip_case =
  let open Gen in
  let+ left = option bool
  and+ right = option bool
  and+ pieces =
    list ~size:(int_range 0 12)
      (frequency [ (2, piece_of white_space); (1, piece_of not_white_space) ])
  in
  (left, right, pieces)

let rec drop_white_space = function
  | p :: rest when is_white_space p -> drop_white_space rest
  | pieces -> pieces

let test_strip (left, right, pieces) =
  let text = String.concat "" pieces in
  cover "white space at both ends"
    (match pieces with
    | first :: _ :: _ ->
        is_white_space first
        && is_white_space (List.nth pieces (List.length pieces - 1))
    | _ -> false);
  cover "the left end kept" (left = Some false);
  cover "the right end kept" (right = Some false);
  let pieces =
    if Option.value left ~default:true then drop_white_space pieces else pieces
  in
  let pieces =
    if Option.value right ~default:true then
      List.rev (drop_white_space (List.rev pieces))
    else pieces
  in
  equal ~msg:(Printf.sprintf "%S" text) string (String.concat "" pieces)
    (Normalizer.apply (Normalizer.strip ?left ?right ()) text)

(* Trailing white space from every offset of a 24-byte text. *)
let white_space_at_every_offset =
  List.init 25 (fun k ->
      ( None,
        None,
        List.init k (fun _ -> "a") @ List.init (24 - k) (fun _ -> " ") ))

let strip_tests =
  [
    prop ~count:400 ~examples:white_space_at_every_offset
      "removes White_Space from the start if left and the end if right"
      strip_case test_strip;
    test "removes every character with the White_Space property and no other"
      (fun () ->
        maps_every_scalar ~before:"" ~after:"" (Normalizer.strip ()) (fun u ->
            if Uucp.White.is_white_space u then "" else utf8 u));
  ]

(* replace *)

(* A literal pattern replaced leftmost first, without overlap, byte for byte. *)
let reference_replace ~pattern ~replacement s =
  let n = String.length pattern and len = String.length s in
  let b = Buffer.create len in
  let rec go i =
    if i >= len then ()
    else if i + n <= len && String.sub s i n = pattern then begin
      Buffer.add_string b replacement;
      go (i + n)
    end
    else begin
      Buffer.add_char b s.[i];
      go (i + 1)
    end
  in
  go 0;
  Buffer.contents b

let replace_alphabet =
  [ "a"; " "; "  "; "\t"; "\u{E9}"; marker; "\u{3000}"; "\xFF"; "\x81";
    "\xE2" ]
[@@ocamlformat "disable"]

let replace_case =
  let open Gen in
  let+ pattern = of_list [ " "; "a"; "\x81"; "\xE2"; "a "; marker; "aa" ]
  and+ replacement = of_list [ ""; marker; "_"; "  "; "\xFF"; "bb" ]
  and+ text = text_of ~size:(int_range 0 40) replace_alphabet in
  (pattern, replacement, text)

(* Hugging Face's rows, with the replacement and the text they were run on. *)
let replace_examples =
  [
    (" ", marker, "a  b");
    ("a", "bb", "aaa");
    ("aa", "b", "aaaaa");
    ("a.b", "X", "a.baxb");
    ("\\s", "X", "a\\sb b");
    ("\u{E9}", "e", "a\u{E9}\u{E9}b");
    ("``", "\"", "``a''");
    ("ab", "_", "a");
    ("abc", "_", "ab");
    (" ", marker, "aaaaaaa aaaaaaa a a");
    ("z", "y", "aaaaaaa aaaaaaa a a");
    ("a", ".", String.make 20 'a');
  ]
  @ List.init 25 (fun k ->
      (" ", marker, String.make k 'a' ^ " " ^ String.make (24 - k) 'b'))

let test_replace (pattern, replacement, text) =
  let n = Normalizer.replace ~pattern ~replacement in
  let found_late =
    String.length text > 8
    && reference_replace ~pattern ~replacement
         (String.sub text 8 (String.length text - 8))
       <> String.sub text 8 (String.length text - 8)
  in
  cover "a match past the eighth byte" found_late;
  equal ~msg:(Printf.sprintf "%S" text) string
    (reference_replace ~pattern ~replacement text)
    (Normalizer.apply n text)

let replace_tests =
  [
    prop ~count:500 ~examples:replace_examples
      "replaces every occurrence left to right without overlap" replace_case
      test_replace;
    (* brot.mli; Hugging Face agrees. *)
    cases ~id:row_id
      "inserts the replacement before every character and at the end for an \
       empty pattern"
      [ ("ab", "_a_b_"); ("\u{E9}", "_\u{E9}_") ]
      (normalizes (Normalizer.replace ~pattern:"" ~replacement:"_"));
    cases ~id:Fun.id "returns empty text unchanged" [ ""; "a" ] (fun pattern ->
        equal string ""
          (Normalizer.apply (Normalizer.replace ~pattern ~replacement:"_") ""));
  ]

(* replace_regex *)

let regex_id (pattern, replacement, text, _) =
  Printf.sprintf "%S by %S in %S" pattern replacement text

let replaces (pattern, replacement, text, expected) =
  equal string expected
    (Normalizer.apply (Normalizer.replace_regex ~pattern ~replacement) text)

(* The pattern CLIP's tokenizer file splits on. *)
let clip_split =
  "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"

let rejects (pattern, reason) =
  raises
    (Invalid_argument
       (Printf.sprintf "invalid regular expression %S: %s" pattern reason))
    (fun () -> Normalizer.replace_regex ~pattern ~replacement:"")

let class_rows =
  [
    ("\\s+", " ", "a  b\x09\x0Ac\xC2\xA0d\xE3\x80\x80e\xE2\x80\x8Bf\xEF\xBB\xBFg\xE2\x80\xA8h", "a b c d e\xE2\x80\x8Bf\xEF\xBB\xBFg h");
    ("\\s+", "_", "a\xE1\x9A\x80b\xE2\x80\x80\xE2\x80\x8Ac\xE2\x80\xAFd\xE2\x81\x9Fe\xC2\x85f\x0Bg\x0Ch", "a_b_c_d_e_f_g_h");
    (" {2,}", " ", "a  b c   d    e", "a b c d e");
    ("\\d", "#", "a1\xD9\xA3\xC2\xB2\xE0\xA5\xA7\xEF\xBC\x91b", "a##\xC2\xB2##b");
    ("\\d+", "#", "12ab\xD9\xA3\xD9\xA4c", "#ab#c");
    ("\\D+", "#", "12ab\xD9\xA3\xD9\xA4c", "12#\xD9\xA3\xD9\xA4#");
    ( "\\w+", "_",
      "h\xC3\xA9llo w\xC3\xB6rld_1 \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E! \xC2\xB2\xC2\xBD \xE2\x80\x8C\xE2\x80\x8D x",
      "_ _ _! _ \xE2\x80\x8C\xE2\x80\x8D _" );
    ( "\\W+", "_",
      "h\xC3\xA9llo w\xC3\xB6rld_1 \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E! \xC2\xB2\xC2\xBD x",
      "h\xC3\xA9llo_w\xC3\xB6rld_1_\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E_\xC2\xB2\xC2\xBD_x" );
    ("\\w", "_", "a\xCC\x81\xE2\x85\xA0\xE2\x93\x90\xE2\x91\xA0\xE2\x88\x80", "____\xE2\x91\xA0\xE2\x88\x80");
    ("\\S+", "_", " a\xC2\xA0b ", " _\xC2\xA0_ ");
    (".", "_", "a\x0A\xF0\x9F\x98\x80b\x0D", "_\x0A___");
    (".+", "_", "ab\x0Acd", "_\x0A_");
    ("[^a]", "_", "a\x0A\xF0\x9F\x98\x80b", "a___");
    ("[^\\w\\s]+", "_", "a-b_c d.e\xE2\x80\x9C", "a_b_c d_e_");
    ("[\\r\\n]+", " ", "a\x0D\x0Ab\x0A\x0Ac\x0Dd", "a b c d");
    ("\\s", "_", "\xF0\x9F\x98\x80 \xF0\x9F\x98\x80", "\xF0\x9F\x98\x80_\xF0\x9F\x98\x80");
  ]
[@@ocamlformat "disable"]

let latin1_rows =
  [
    ("[\\w]", "_", "a\xC2\xB2", "_\xC2\xB2");
    ("[^\\W]+", "_", "a\xC2\xB2b", "_\xC2\xB2_");
    ("\\w+", "_", "a\xC2\xB2b", "_");
    ("\\w", "_", "\xC2\xB9\xC2\xB2\xC2\xB3\xC2\xBC\xC2\xBD\xC2\xBE", "______");
  ]
[@@ocamlformat "disable"]

let category_rows =
  [
    ("\\p{L}+", "_", "ab1cd \xC3\xA9\xE6\x97\xA5 \xCC\x81", "_1_ _ \xCC\x81");
    ("\\P{L}+", "_", "ab1cd \xC3\xA9\xE6\x97\xA5 \xCC\x81", "ab_cd_\xC3\xA9\xE6\x97\xA5_");
    ("\\p{^L}+", "_", "ab1cd \xC3\xA9\xE6\x97\xA5", "ab_cd_\xC3\xA9\xE6\x97\xA5");
    ("\\P{^L}+", "_", "ab1cd \xC3\xA9\xE6\x97\xA5", "_1_ _");
    ("\\p{Lu}", "_", "aA\xC3\x89\xC3\xA9\xC7\x85", "a__\xC3\xA9\xC7\x85");
    ("\\p{ lu }", "_", "aA\xC3\x89\xC3\xA9", "a__\xC3\xA9");
    ("\\p{L_u}", "_", "aA", "a_");
    ("\\p{Letter}+", "_", "ab1cd", "_1_");
    ("\\p{Uppercase_Letter}", "_", "aAb", "a_b");
    ("\\p{ Decimal-Number }", "_", "a1b", "a_b");
    ("\\p{Nd}+", "_", "a12\xD9\xA3\xC2\xB2b", "a_\xC2\xB2b");
    ("\\p{N}+", "_", "a12\xD9\xA3\xC2\xB2\xE2\x85\xA0b", "a_b");
    ("\\p{P}+", "_", "a-b_c d.e!\xE2\x80\x9C", "a_b_c d_e_");
    ("\\p{Z}+", "_", "a  b\xC2\xA0\xE3\x80\x80c\xE2\x80\xA8d", "a_b_c_d");
    ("\\p{Zs}+", "_", "a  b\xC2\xA0\xE3\x80\x80c\xE2\x80\xA8d", "a_b_c\xE2\x80\xA8d");
    ("\\p{C}+", "_", "a\x00\x1F\xE2\x80\x8Bb\xF3\xA0\x82\x80c", "a_b_c");
    ("\\p{Cn}", "_", "a\xF3\xA0\x82\x80b", "a_b");
    ("\\p{Co}", "_", "a\xEE\x80\x80b", "a_b");
    ("\\p{M}+", "_", "e\xCC\x81\xCC\xA3\xE0\xA4\xBF\xE2\x83\x9Dx", "e_x");
    ("\\p{S}+", "_", "a$+^\xC2\xA9b", "a_b");
    ("\\p{Any}", "_", "a\x0Ab", "___");
    ("\\p{LC}", "_", "aA\xC7\x85\xCA\xB0", "___\xCA\xB0");
    ("[\\p{Ll}A]", "_", "aAbB", "___B");
    ("[^\\p{Ll}A]", "_", "aAbB", "aAb_");
    ("[\\P{Ll}]", "_", "aAbB", "a_b_");
    ("a[^\\x00-\\x{10FFFF}]|b", "_", "axyb", "axy_");
    ("\\p{Cs}", "_", "axyb", "axyb");
    ("\\P{Any}", "_", "axyb", "axyb");
  ]
[@@ocamlformat "disable"]

let script_rows =
  [
    ("\\p{Latin}+", "_", "a\u{E9}\u{6F22}b", "_\u{6F22}_");
    ("\\p{Han}", "_", "a\u{6F22}b", "a_b");
  ]
[@@ocamlformat "disable"]

let intersection_rows =
  [
    ("[a-z&&[^aeiou]]", "_", "abcde", "a___e");
    ("[^a-z&&[^aeiou]]", "_", "abcde1", "_bcd__");
  ]
[@@ocamlformat "disable"]

let bracket_rows =
  [
    ("[^\\W\\d]+", "_", "a-b1c", "_-_1_");
    ("[\\d-]", "_", "a-1", "a__");
    ("[a-]", "_", "a-b", "__b");
    ("[-a]", "_", "a-b", "__b");
    ("[]a]", "_", "a]b", "__b");
    ("[]-a]", "_", "a]^b", "___b");
    ("[^]]", "_", "a]b", "_]_");
    ("[a^]", "_", "a^b", "__b");
    ("[\\-\\]\\[\\^]", "_", "a-][^b", "a____b");
    ("[a-z]+", "_", "abc\xC3\xA9d", "_\xC3\xA9_");
    ("[\xC3\xA9-\xC3\xBC]+", "_", "a\xC3\xA9\xC3\xB2\xC3\xBCb", "a_b");
    ("[\\x{1F600}-\\x{1F64F}]", "_", "a\xF0\x9F\x98\x80\xF0\x9F\x99\x8F\xF0\x9F\x99\x90b", "a__\xF0\x9F\x99\x90b");
    ("\\x{2581}", " ", "a\xE2\x96\x81b", "a b");
    ("\\x41\\x{42}", "_", "aABb", "a_b");
    ("\\x4", "_", "a\x04b", "a_b");
    ("\\t\\n\\r\\f\\v\\e\\a", "_", "a\x09\x0A\x0D\x0C\x0B\x1B\x07b", "a_b");
    ("\\0\\07\\077\\0777", "_", "a\x00\x07??7b", "a_b");
    ("\\/\\-\\ \\.\\*\\+\\?\\(\\)\\{\\}\\|\\\\", "_", "a/- .*+?(){}|\\b", "a_b");
    ("\\\xC3\xA9", "_", "a\xC3\xA9b", "a_b");
  ]
[@@ocamlformat "disable"]

let brace_rows =
  [
    ("a{", "_", "a{b", "_b");
    ("a{x}", "_", "a{x}b", "_b");
    ("a{2", "_", "a{2b", "_b");
    ("a{,}", "_", "a{,}b", "_b");
    ("a{2,,3}", "_", "a{2,,3}b", "_b");
    ("a{ 2}", "_", "a{ 2}b", "_b");
  ]
[@@ocamlformat "disable"]

let repeat_rows =
  [
    ("a{,2}", "_", "aaab", "__b_");
    ("a{2,}", "_", "aaab", "_b");
    ("a{2,3}", "_", "aaaab", "_ab");
    ("a{02}", "_", "aaab", "_ab");
    ("a{0}", "_", "aab", "_a_a_b_");
    ("a{2}?", "_", "aaaab", "__b_");
    ("a{2,3}?", "_", "aaaab", "__b");
    ("a{1,2}{2}", "_", "aaaaa", "_a");
    ("(a*){1,3}?", "_", "a", "_");
    ("(a+?)+", "_", "aaa", "_");
    ("(a*)+?b", "_", "aab", "_");
    ("a*?", "_", "aab", "_a_a_b_");
    ("a??", "_", "aab", "_a_a_b_");
    ("a+?", "_", "aab", "__b");
    ("a**", "_", "aab", "_b_");
    ("a?*", "_", "aab", "_b_");
    ("a*{2}", "_", "aab", "_b_");
    ("(a*)(a*)", "_", "aab", "_b_");
    ("a??b", "_", "aab", "a_");
    ("(a|b)*c", "_", "ababc", "_");
    ("(a|b)*?", "_", "aab", "_a_a_b_");
    ("(?:ab)+", "_", "ababc", "_c");
    ("(?<n>ab)+", "_", "ababc", "_c");
    ("(?<_1>ab)+", "_", "ababc", "_c");
    ("a(?#comment)b", "_", "ab", "_");
    ("(?#)a", "_", "aab", "__b");
    ("ab*c", "_", "abbc ac abc", "_ _ _");
  ]
[@@ocamlformat "disable"]

let leftmost_rows =
  [
    ("\\r\\n|\\n", " ", "a\x0D\x0Ab\x0A\x0Ac", "a b  c");
    ("\\n|\\r\\n", " ", "a\x0D\x0Ab\x0A\x0Ac", "a b  c");
    ("a|ab", "_", "abab", "_b_b");
    ("ab|a", "_", "abab", "__");
    ("(a|ab)(c|bcd)", "_", "abcd", "_");
    ("(a|ab)c", "_", "abc", "_");
    ("a*b|a", "_", "aab", "_");
    (clip_split, "|", "<|startoftext|>a photo of 2 cats' toys, don't!  \xC3\xA9t\xC3\xA9 \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E 42<|endoftext|>", "|| | | | || || |||  | | |||");
    (clip_split, "|", "it's 3.5% -- ok\xE2\x80\xA6", "|| |||| | ||");
  ]
[@@ocamlformat "disable"]

let empty_match_rows =
  [
    ("x*", "_", "a\xC3\xA9 b", "_a_\xC3\xA9_ _b_");
    ("x*", "_", "xa", "_a_");
    ("x*", "_", "x\xC3\xA9x", "_\xC3\xA9_");
    ("x*", "_", "\xF0\x9F\x98\x80a", "_\xF0\x9F\x98\x80_a_");
    ("x*", "_", "x", "_");
    ("", "_", "ab", "_a_b_");
    ("a|", "_", "bab", "_b_b_");
    ("|a", "_", "bab", "_b_a_b_");
    ("(a|)", "_", "ab", "_b_");
    ("(|)", "_", "ab", "_a_b_");
  ]
[@@ocamlformat "disable"]

let empty_text_rows =
  [
    ("x*", "_", "", "");
    ("", "_", "", "");
    ("$", "_", "", "");
  ]
[@@ocamlformat "disable"]

let anchor_rows =
  [
    ("a$", "_", "a\x0Ab", "_\x0Ab");
    ("a$", "_", "a\x0A", "_\x0A");
    ("a$", "_", "a\x0A\x0A", "_\x0A\x0A");
    ("$", "_", "a\x0Ab", "a_\x0Ab_");
    ("$", "_", "\x0A\x0A", "_\x0A_\x0A_");
    ("a\\z", "_", "a\x0A", "a\x0A");
    ("\\z", "_", "a\x0Ab\x0A", "a\x0Ab\x0A_");
    ("\\Z", "_", "a\x0Ab\x0A", "a\x0Ab_\x0A_");
    ("\\Z", "_", "a\x0A\x0A", "a\x0A_\x0A_");
    ("a\\Z", "_", "a\x0A\x0A", "a\x0A\x0A");
    ("\\A", "_", "a\x0Ab\x0A", "_a\x0Ab\x0A");
    ("\\Aa", "_", "aa", "_a");
    ("\\Ga", "_", "aab", "__b");
    ("^a", "_", "\x0Aa", "\x0A_");
    ("^\\s+", "_", "  a\x0A  b", "_a\x0A_b");
    ("\\s+$", "_", "a  \x0Ab  ", "a_\x0Ab_");
    ("a^b", "_", "a\x0Ab", "a\x0Ab");
    ("a\\n^b", "_", "a\x0Ab", "_");
    ("a$\\nb", "_", "a\x0Ab", "_");
    ("\\n(^b)", "_", "a\x0Ab\x0A", "a_\x0A");
  ]
[@@ocamlformat "disable"]

let case_folding_rows =
  [
    ("(?i)a", "_", "aAbB", "__bB");
    ("(?i:a)b", "_", "abAbaB", "__aB");
    ("(?-i:a)", "_", "aA", "_A");
    ("(?i)a(?-i)b", "_", "abAbaBAB", "__aBAB");
    ("a(?i)b|c", "_", "aBAbCc", "_AbCc");
    ("(?i:a|b)c", "_", "acBcbC", "__bC");
    ("(?i:'s)", "_", "'s'S'\xC5\xBF", "___");
    ("(?i)k", "_", "kK\xE2\x84\xAA", "___");
    ("(?i)[k]", "_", "\xE2\x84\xAA", "_");
    ("(?i)\\x{17F}", "_", "sS\xC5\xBF", "___");
    ("(?i)[a-c]+", "_", "aBcD", "_D");
    ("(?i)[A-Z]", "_", "a\xC3\xA9\xC5\xBF\xE2\x84\xAA", "_\xC3\xA9__");
    ("(?i)[^a]", "_", "aAbB", "aA__");
    ("(?i)\xC3\xA9", "_", "\xC3\xA9\xC3\x89e", "__e");
    ("(?i)\xCF\x83", "_", "\xCF\x83\xCE\xA3\xCF\x82", "___");
    ("(?i)\xC7\x85", "_", "\xC7\x84\xC7\x85\xC7\x86", "___");
    ("(?i)\\p{Lu}", "_", "aA1", "a_1");
    ("(?i)\\P{Lu}", "_", "aA1", "_A_");
    ("(?i)[\\p{Lu}]", "_", "aA1", "__1");
    ("(?i)[\\p{Lu}x]", "_", "aA1X", "__1_");
    ("(?i)[^\\p{Lu}]", "_", "aA1", "aA_");
    ("(?i)[^\\P{Lu}]", "_", "aA1", "aA1");
    ("(?i)\\d", "_", "1a", "_a");
  ]
[@@ocamlformat "disable"]

let lookahead_rows =
  [
    ("a(?!b)", "_", "abaca", "ab_c_");
    ("a(?=b)", "_", "abaca", "_baca");
    ("\\s+(?!\\S)", "_", "a  b   c ", "a_ b_ c_");
    ("\\s+(?!\\S)|\\s+", "_", "a  b   c d", "a__b__c_d");
    ("a(?=bc|d)", "_", "abcadab", "_bc_dab");
    ("(a(?=b)|c)", "_", "abcac", "_b_a_");
    ("(?:x|a(?!b))", "_", "xabac", "_ab_c");
    ("a(?![bc])", "_", "abacada", "abac_d_");
    ("a(?!.)", "_", "ab\x0Aa\x0A", "ab\x0A_\x0A");
    ("a(?!\\n)", "_", "a\x0Aab", "a\x0A_b");
    ("a(?!\xC3\xA9)", "_", "a\xC3\xA9aea", "a\xC3\xA9_e_");
    ("(?i)a(?!b)", "_", "aBAbac", "aBAb_c");
    ("(?=b)", "_", "abab", "a_ba_b");
    ("a*(?=b)", "_", "aabab", "_b_b");
    ("a+?(?=b)", "_", "aaab", "_b");
    ("\\d{1,3}(?=(?:\\d{3})+)", ",", "1234567", ",,567");
  ]
[@@ocamlformat "disable"]

let rejected_patterns =
  [
    ("(?m).", "group options other than i are not supported");
    ("(?im).", "group options other than i are not supported");
    ("(?<=a)b", "lookbehind is not supported");
    ("(?<!a)b", "lookbehind is not supported");
    ("a(?=b)c", "lookahead is supported only where it ends the pattern");
    ("(?!a)b", "lookahead is supported only where it ends the pattern");
    ("(a(?=b))+", "lookahead is supported only where it ends the pattern");
    ("(a(?!b)|c)d", "lookahead is supported only where it ends the pattern");
    ("a(?=b(?=c))", "lookahead is supported only where it ends the pattern");
    ("a(?!bc)", "a negative lookahead is supported over one character only");
    ("a(?!b|c)", "a negative lookahead is supported over one character only");
    ("a(?!(b))", "a negative lookahead is supported over one character only");
    ("(?>a)", "atomic groups are not supported");
    ("(a)\\1", "backreferences are not supported");
    ("\\ba", "word boundaries are not supported");
    ("a++", "possessive quantifiers are not supported");
    ("[[:alpha:]]", "POSIX bracket expressions are not supported");
    ("[a[b]", "unterminated character class");
    ("[a-[b]]", "invalid range in character class");
    ("\\p{Foo}", "unsupported property \\p{Foo}");
    ("\\p{Hrkt}", "unsupported property \\p{Hrkt}");
    ("\\p{Katakana_Or_Hiragana}", "unsupported property \\p{Katakana_Or_Hiragana}");
    ("\\h", "unsupported escape \\h");
    ("\\R", "unsupported escape \\R");
    ("\\xc3\\xa9", "byte escapes above \\x7F are not supported, use \\x{..}");
    ("\\x{110000}", "invalid code point escape");
    ("\\x{D800}", "invalid code point escape");
    ("\\xg", "invalid hexadecimal escape");
    ("^", "a ^ that can end a match is not supported");
    ("^\\s*", "a ^ that can end a match is not supported");
    ("\\n^", "a ^ that can end a match is not supported");
    ("(^|x)a*", "a ^ that can end a match is not supported");
    ("*a", "nothing to repeat");
    ("{2}", "nothing to repeat");
    ("a|*", "nothing to repeat");
    ("(", "unmatched (");
    (")", "unmatched )");
    ("a\\", "pattern ends with a backslash");
    ("[a", "unterminated character class");
    ("[z-a]", "empty range in character class");
    ("[\\d-z]", "invalid range in character class");
    ("[a-\\d]", "invalid range in character class");
    ("a{3,2}", "invalid interval {3,2}");
    ("a{100001}", "repeat count above 100000");
    ("\\p{L", "unterminated property escape");
    ("\\pL", "invalid property escape");
    ("(?<1a>x)", "invalid group name");
    ("(?#a", "unterminated comment");
    ("\xC3", "pattern is not valid UTF-8");
  ]
[@@ocamlformat "disable"]

(* The two-letter general categories, each with a character of it. *)
let general_categories =
  [ ("Lu", `Lu, 0x41); ("Ll", `Ll, 0x61); ("Lt", `Lt, 0x1C5);
    ("Lm", `Lm, 0x2B0); ("Lo", `Lo, 0x6F22); ("Mn", `Mn, 0x301);
    ("Mc", `Mc, 0x903); ("Me", `Me, 0x20DD); ("Nd", `Nd, 0x31);
    ("Nl", `Nl, 0x2160); ("No", `No, 0xB2); ("Pc", `Pc, 0x5F);
    ("Pd", `Pd, 0x2D); ("Ps", `Ps, 0x28); ("Pe", `Pe, 0x29);
    ("Pi", `Pi, 0xAB); ("Pf", `Pf, 0xBB); ("Po", `Po, 0x21);
    ("Sm", `Sm, 0x2B); ("Sc", `Sc, 0x24); ("Sk", `Sk, 0x5E);
    ("So", `So, 0xA9); ("Zs", `Zs, 0x20); ("Zl", `Zl, 0x2028);
    ("Zp", `Zp, 0x2029); ("Cc", `Cc, 0x00); ("Cf", `Cf, 0xAD);
    ("Co", `Co, 0xE000); ("Cn", `Cn, 0x378) ]
[@@ocamlformat "disable"]

let category_patterns =
  lazy
    (List.map
       (fun (name, gc, _) ->
         ( name,
           gc,
           Normalizer.replace_regex
             ~pattern:("\\p{" ^ name ^ "}")
             ~replacement:"_" ))
       general_categories)

let test_category_membership u =
  List.iter
    (fun (name, gc, n) ->
      let text = utf8 u in
      equal
        ~msg:(Printf.sprintf "\\p{%s} on U+%04X" name (Uchar.to_int u))
        string
        (if Uucp.Gc.general_category u = gc then "_" else text)
        (Normalizer.apply n text))
    (Lazy.force category_patterns)

(* Every row is Hugging Face's [Replace(Regex(pattern), replacement)]. *)
let replace_regex_tests =
  [
    cases ~id:regex_id
      "reads \\s, \\d, \\w, . and their negations as characters" class_rows
      replaces;
    cases ~id:regex_id
      "matches \u{B9}, \u{B2}, \u{B3}, \u{BC}, \u{BD} and \u{BE} with \\w \
       outside a bracket class"
      latin1_rows replaces;
    prop ~count:1000
      ~examples:
        (List.map (fun (_, _, code) -> Uchar.of_int code) general_categories)
      "matches with \\p{X} the characters of general category X" Gen.uchar
      test_category_membership;
    cases ~id:regex_id "selects a general category with \\p" category_rows
      replaces;
    cases ~id:regex_id "selects a script with \\p" script_rows replaces;
    cases ~id:regex_id "intersects bracket classes with &&" intersection_rows
      replaces;
    cases ~id:regex_id "reads bracket classes, ranges and escapes" bracket_rows
      replaces;
    cases ~id:regex_id "reads a { that opens no interval as itself" brace_rows
      replaces;
    cases ~id:regex_id
      "repeats greedily or lazily as a backtracking engine does" repeat_rows
      replaces;
    cases ~id:regex_id
      "takes the leftmost match and the first alternative that matches there"
      leftmost_rows replaces;
    cases ~id:regex_id "skips an empty match right after a match"
      empty_match_rows replaces;
    cases ~id:regex_id "returns empty text unchanged" empty_text_rows replaces;
    cases ~id:regex_id
      "reads ^ and $ as line anchors and \\A, \\z and \\Z as text anchors"
      anchor_rows replaces;
    cases ~id:regex_id
      "matches every character of the same simple case folding under (?i)"
      case_folding_rows replaces;
    cases ~id:regex_id
      "matches a lookahead that ends the pattern or an alternative"
      lookahead_rows replaces;
    cases ~id:regex_id "inserts the replacement as it is"
      [ ("(a)", "$1", "xa", "x$1") ]
      replaces;
    cases ~id:fst "raises Invalid_argument naming what it cannot read"
      rejected_patterns rejects;
  ]

(* prepend *)

let prepend_tests =
  [
    prop ~count:300 "prepends to text that is not empty"
      (Gen.pair
         (Gen.of_list [ marker; "<<"; ""; " " ])
         (Gen.string_of ~size:(Gen.int_range 1 20) Gen.char))
      (fun (prefix, text) ->
        equal string (prefix ^ text)
          (Normalizer.apply (Normalizer.prepend prefix) text));
    cases ~id:Fun.id "returns empty text unchanged" [ marker; "<<" ]
      (fun prefix ->
        equal string "" (Normalizer.apply (Normalizer.prepend prefix) ""));
  ]

(* byte_level *)

(* GPT-2's [bytes_to_unicode] (encoder.py): a printable byte of Latin-1 stands
   for itself, and the others, in order, for U+0100 onwards. *)
let gpt2_char byte =
  let printable b =
    (b >= 0x21 && b <= 0x7E)
    || (b >= 0xA1 && b <= 0xAC)
    || (b >= 0xAE && b <= 0xFF)
  in
  if printable byte then utf8 (Uchar.of_int byte)
  else
    let below =
      List.length
        (List.filter (fun b -> not (printable b)) (List.init byte Fun.id))
    in
    utf8 (Uchar.of_int (0x100 + below))

let byte_level_tests =
  [
    test "maps each byte to its character in GPT-2's table" (fun () ->
        equal (list string) (List.init 256 gpt2_char)
          (List.init 256 (fun b ->
               Normalizer.apply Normalizer.byte_level
                 (String.make 1 (Char.chr b)))));
    cases ~id:row_id "maps a text byte by byte"
      [ (" x", "\u{120}x"); ("a\u{E9}", "a\u{C3}\u{A9}"); ("", "") ]
      (normalizes Normalizer.byte_level);
  ]

(* nmt *)

(* The code points brot.mli lists. *)
let nmt_removed code =
  (code >= 0x01 && code <= 0x08)
  || code = 0x0B
  || (code >= 0x0E && code <= 0x1F)
  || code = 0x7F || code = 0x8F || code = 0x9F

let nmt_spaced code =
  List.mem code
    [ 0x09; 0x0A; 0x0C; 0x0D; 0x1680; 0x2028; 0x2029; 0x2581; 0xFEFF; 0xFFFD ]
  || (code >= 0x200B && code <= 0x200F)

let nmt_tests =
  [
    test "removes and spaces exactly the characters it lists" (fun () ->
        maps_every_scalar Normalizer.nmt (fun u ->
            let code = Uchar.to_int u in
            if nmt_removed code then ""
            else if nmt_spaced code then " "
            else utf8 u));
    (* brot.mli, [encode]: a byte that starts no character is a character. *)
    cases ~id:row_id "keeps a byte that starts no character"
      [ ("a\xFFb", "a\xFFb") ]
      (normalizes Normalizer.nmt);
  ]

(* precompiled *)

(* The charsmap of test/gen/gen_precompiled.py: a few of T5's rules, compiled by
   SentencePiece, and the rules themselves. *)
let charsmap_document =
  lazy
    (match
       Jsont_bytesrw.decode_string Jsont.json
         (Fixture.read "support/precompiled/charsmap.json")
     with
    | Ok json -> json
    | Error msg -> failf "cannot read the charsmap: %s" msg)

let member name = function
  | Jsont.Object (members, _) -> (
      match Jsont.Json.find_mem name members with
      | Some (_, value) -> value
      | None -> failf "no %S member" name)
  | _ -> failf "looking for %S in something that is not an object" name

let charsmap_json () = member "normalizer" (Lazy.force charsmap_document)

let charsmap_rules () =
  match member "rules" (Lazy.force charsmap_document) with
  | Jsont.Array (rules, _) ->
      List.map
        (function
          | Jsont.Array ([ Jsont.String (s, _); Jsont.String (r, _) ], _) ->
              (s, r)
          | _ -> failf "a rule is not a pair of strings")
        rules
  | _ -> failf "the rules are not an array"

let precompiled = lazy (require_ok (Normalizer.of_json (charsmap_json ())))

(* The reference reads the rules themselves rather than the trie compiled from
   them. Its alphabet holds characters of four kinds only, whose clusters are
   simple: CR LF, a control alone, or any other character and the extends after
   it. *)
let precompiled_alphabet =
  [
    "a"; "e"; "C"; "x"; " "; "\xEF\xBD\x85"; "\xEF\xBD\xB6"; "\xEF\xBC\xA1";
    "\xEF\xAC\x81"; "\xE2\x84\xA2"; "\xC2\xBD"; "\xEF\xBF\xBD"; "\xCC\x81";
    "\xCC\x80"; "\xCC\x82"; "\xCC\xA7"; "\xEF\xBE\x9E"; "\x01"; "\t"; "\r";
    "\n"; "\r\n"; "\xE2\x80\x8B"; "\xFF"; "\x80"; "\xC3";
  ]
[@@ocamlformat "disable"]

let extends = [ 0x300; 0x301; 0x302; 0x327; 0xFF9E ]
let controls = [ 0x01; 0x09; 0x0A; 0x0D; 0x200B ]

(* The characters of [s]: [Some code] and its bytes, or [None] for a byte that
   does not start one. *)
let characters s =
  let rec go i acc =
    if i >= String.length s then List.rev acc
    else
      let d = String.get_utf_8_uchar s i in
      if Uchar.utf_decode_is_valid d then
        let n = Uchar.utf_decode_length d in
        go (i + n)
          ((Some (Uchar.to_int (Uchar.utf_decode_uchar d)), String.sub s i n)
          :: acc)
      else go (i + 1) ((None, String.sub s i 1) :: acc)
  in
  go 0 []

let clusters s =
  let joins prev cur =
    match (prev, cur) with
    | Some 0x0D, Some 0x0A -> true
    | None, _ | _, None -> false
    | Some p, Some c -> List.mem c extends && not (List.mem p controls)
  in
  let rec go prev current acc = function
    | [] -> List.rev (List.rev current :: acc)
    | ((code, _) as c) :: rest ->
        if current <> [] && joins prev code then go code (c :: current) acc rest
        else
          go code [ c ]
            (if current = [] then acc else List.rev current :: acc)
            rest
  in
  if s = "" then [] else go None [] [] (characters s)

let reference rules s =
  let starts chunk (source, _) = String.starts_with ~prefix:source chunk in
  let shortest chunk =
    List.fold_left
      (fun best ((source, _) as rule) ->
        if not (starts chunk rule) then best
        else
          match best with
          | Some (s, _) when String.length s <= String.length source -> best
          | _ -> Some rule)
      None rules
  in
  let character (code, bytes) =
    if code = None then "\xEF\xBF\xBD"
    else match shortest bytes with Some (_, r) -> r | None -> bytes
  in
  let cluster chars =
    let bytes = String.concat "" (List.map snd chars) in
    match (String.length bytes < 6, shortest bytes) with
    | true, Some (_, r) when List.for_all (fun (c, _) -> c <> None) chars -> r
    | _ -> String.concat "" (List.map character chars)
  in
  String.concat "" (List.map cluster (clusters s))

let test_precompiled_rules text =
  let rules = charsmap_rules () in
  let groups = clusters text in
  let text_of chars = String.concat "" (List.map snd chars) in
  let bytes chars = String.length (text_of chars) in
  cover "a cluster cut short"
    (List.exists
       (fun chars ->
         List.length chars > 1
         && bytes chars < 6
         && List.exists
              (fun (s, _) ->
                String.length s < bytes chars
                && String.starts_with ~prefix:s (text_of chars))
              rules)
       groups);
  cover "a cluster of six bytes or more"
    (List.exists (fun chars -> bytes chars >= 6) groups);
  cover "CR LF"
    (List.exists
       (fun chars -> List.map fst chars = [ Some 0x0D; Some 0x0A ])
       groups);
  cover "a malformed byte"
    (List.exists (fun chars -> List.mem_assoc None chars) groups);
  equal string (reference rules text)
    (Normalizer.apply (Lazy.force precompiled) text)

(* Grapheme clusters through Unicode's conformance test for UAX #29 at Unicode
   18.0.0 (support/unicode/GraphemeBreakTest-18.0.0.txt). The charsmap of
   support/precompiled/graphemes.json names every code point of the test but
   NUL, U+0308 as [<0308>], so a cluster under six bytes comes out as its first
   character's name and a longer one as the names of all its characters. A test
   line is [÷ 0020 × 0308 ÷ ... # comment]: code points in hex, [÷] where a
   boundary falls and [×] where none does. *)
let test_precompiled_clusters () =
  let n =
    match
      Jsont_bytesrw.decode_string Jsont.json
        (Fixture.read "support/precompiled/graphemes.json")
    with
    | Error msg -> failf "cannot read the charsmap: %s" msg
    | Ok json -> require_ok (Normalizer.of_json json)
  in
  let name cp = if cp = 0 then "\x00" else Printf.sprintf "<%04X>" cp in
  let code_points line =
    let fields =
      String.split_on_char ' ' (List.hd (String.split_on_char '#' line))
      |> List.map String.trim
      |> List.filter (fun f -> f <> "")
    in
    let rec go current acc = function
      | [] -> List.rev acc
      | "\xC3\xB7" :: rest ->
          go [] (if current = [] then acc else List.rev current :: acc) rest
      | "\xC3\x97" :: rest -> go current acc rest
      | cp :: rest -> go (int_of_string ("0x" ^ cp) :: current) acc rest
    in
    go [] [] fields
  in
  let encode cluster =
    String.concat "" (List.map (fun cp -> utf8 (Uchar.of_int cp)) cluster)
  in
  let expected cluster =
    match cluster with
    | first :: _ when first <> 0 && String.length (encode cluster) < 6 ->
        name first
    | _ -> String.concat "" (List.map name cluster)
  in
  let lines =
    Fixture.read "support/unicode/GraphemeBreakTest-18.0.0.txt"
    |> String.split_on_char '\n'
    |> List.filter (fun l -> String.length l > 0 && l.[0] <> '#')
  in
  equal ~msg:"test lines" int 853 (List.length lines);
  List.iter
    (fun line ->
      let cs = code_points line in
      equal ~msg:line string
        (String.concat "" (List.map expected cs))
        (Normalizer.apply n (String.concat "" (List.map encode cs))))
    lines

(* A charsmap of one rule, [a] to [b], laid out by hand: unit 0 with children at
   offset 1, the unit for byte [a] (label 0x61, a leaf below, children at offset
   2), and its leaf, of value 0. [blob] lays out any units in one block of
   256. *)
let u32 n =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.of_int n);
  Bytes.to_string b

let blob ?(size = 1024) units replacements =
  let trie =
    String.concat ""
      (List.init 256 (fun i ->
           u32 (match List.assoc_opt i units with Some u -> u | None -> 0)))
  in
  u32 size ^ trie ^ replacements

let offset o = o lsl 10
let root = (0, offset 1)
let unit_a = (1 lxor 0x61, 0x61 lor 0x100 lor offset 2)
let leaf v = (1 lxor 0x61 lxor 2, 0x8000_0000 lor v)
let a_to_b () = blob [ root; unit_a; leaf 0 ] "b\x00"

(* The checks SentencePiece makes before it reads a charsmap: a trie of whole
   blocks of 256 units that ends before the replacements, which are UTF-8, and
   whose every child and value is in range. *)
let malformed_charsmaps =
  [
    ("too short", "\x00\x04", "the charsmap is truncated");
    ("no replacements", blob [ root; unit_a; leaf 0 ] "",
     "the trie overruns the charsmap");
    ("a part of a block", blob ~size:1020 [ root; unit_a; leaf 0 ] "b\x00",
     "the trie size is not a multiple of 1024");
    ("replacements not UTF-8", blob [ root; unit_a; leaf 0 ] "\xFF\x00",
     "the replacements are not valid UTF-8");
    ("root without children", blob [] "b\x00", "the trie is malformed");
    ("value past the replacements", blob [ root; unit_a; leaf 2 ] "b\x00",
     "the trie is malformed");
    ("value inside a character", blob [ root; unit_a; leaf 1 ] "\xC3\xA9\x00",
     "the trie is malformed");
    ("children past the trie",
     blob [ root; (1 lxor 0x61, 0x61 lor offset 300) ] "b\x00",
     "the trie is malformed");
  ]
[@@ocamlformat "disable"]

let r = "\xEF\xBF\xBD"

let precompiled_tests =
  [
    prop ~count:1000 "agrees with its rules, cluster by cluster"
      (text_of precompiled_alphabet)
      test_precompiled_rules;
    test "reads text by extended grapheme cluster" test_precompiled_clusters;
    (* UnicodeData.txt 18.0.0: U+05C8 is a mark, an extend that joins the "½"
       before it in one cluster of four bytes, which becomes the replacement of
       "½". HuggingFace, at Unicode 17.0.0, keeps U+05C8 as a cluster of its
       own. *)
    cases ~id:row_id "reads the clusters of Unicode 18.0.0"
      [ ("\u{bd}\u{5c8}", "1\u{2044}2") ]
      (fun row -> normalizes (Lazy.force precompiled) row);
    (* SentencePiece. *)
    cases ~id:row_id
      "turns each byte that starts no valid character into U+FFFD"
      [
        ("a\xFFb", "a" ^ r ^ "b");
        ("\xC3", r);
        ("\xE2\x82", r ^ r);
        ("\xC0\xAF", r ^ r);
        ("\xED\xA0\x80", r ^ r ^ r);
        ("\xF4\x90\x80\x80", r ^ r ^ r ^ r);
        (* Such a U+FFFD is not looked up: the rule for U+FFFD itself does not
           apply to it. *)
        ("\xFF\xEF\xBC\xA1", r ^ "A");
        ("\x80\xE2\x84\xA2", r ^ "TM");
        ("e\x80\xCC\x81", "e" ^ r ^ "\xCC\x81");
      ]
      (fun row -> normalizes (Lazy.force precompiled) row);
    cases
      ~id:(fun (input, _, _) -> text_id input)
      "gives each U+FFFD the span of its own byte"
      [
        ("a\xFF\xEF\xBC\xA1", "a" ^ r ^ "A", "0,1 1,2 2,5");
        ("e\x80\xCC\x81", "e" ^ r ^ "\xCC\x81", "0,1 1,2 2,4");
        ("\xFF\x80", r ^ r, "0,1 1,2");
        ("a\xE2\x82b", "a" ^ r ^ r ^ "b", "0,1 1,2 2,3 3,4");
      ]
      (fun (input, text, spans) ->
        equal (pair string string) (text, spans)
          (char_spans (Lazy.force precompiled) input));
    test "reads a charsmap of one rule" (fun () ->
        equal string "bbc"
          (Normalizer.apply (Normalizer.precompiled (a_to_b ())) "abc"));
    cases
      ~id:(fun (label, _, _) -> label)
      "raises Invalid_argument on a malformed charsmap" malformed_charsmaps
      (fun (_, charsmap, reason) ->
        raises
          (Invalid_argument ("invalid precompiled charsmap: " ^ reason))
          (fun () -> Normalizer.precompiled charsmap));
  ]

(* bert *)

(* Text without spacing or enclosing marks, which [strip_accents] removes and
   [bert] keeps, so that dropping the nonspacing marks is dropping the marks. *)
let bert_alphabet =
  [ "a"; "Z"; "M"; "0"; " "; "\t"; "\x01"; "\u{E9}"; "\u{C9}"; "\u{301}";
    "\u{323}"; "\u{1E9B}"; "\u{130}"; "\u{3A3}"; "\u{DF}"; "\u{1E9E}";
    "\u{FB01}"; "\u{FF21}"; "\u{AC01}"; "\u{1F600}"; "\u{6F22}"; "\xFF";
    "\xC3"; "HELLO"; "world" ]
[@@ocamlformat "disable"]

(* Texts the one-pass BERT normalizer was checked on against its stages. *)
let bert_corpus =
  [
    "";
    "Hello, World!";
    "Caf\xC3\xA9 NA\xC3\x8FVE r\xC3\xA9sum\xC3\xA9";
    "A\xCC\x81\xCC\xA3 a\xCC\xA3\xCC\x81 \xE1\xBA\x9B\xCC\xA3";
    "\xC3\x87a \xE6\xBC\xA2 \xC3\xA1x \xE6\x97\xA5\xE6\x9C\xAC";
    "\xEA\xB0\x81Z \xED\x95\x9C\xEA\xB8\x80 \
     \xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8";
    "\xCE\x9F\xCE\x94\xCE\x9F\xCE\xA3 \
     \xCE\xA3\xCE\xAF\xCF\x83\xCF\x85\xCF\x86\xCE\xBF\xCF\x82";
    "\xC4\xB0stanbul I\xCC\x87 \xC3\x9F \xE1\xBA\x9E \xEF\xAC\x81";
    "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82, \
     \xD0\xBC\xD0\xB8\xD1\x80!";
    "Ti\xE1\xBA\xBFng Vi\xE1\xBB\x87t c\xC3\xB3 d\xE1\xBA\xA5u";
    "a\xF0\x9F\x98\x80b \xEF\xBC\xA1\xEF\xBC\xA2 \xE2\x91\xA0";
    "Hello\xFF\xFEWorld";
    "\xE6\xBC\xA2\x80\xE6\xBC \xE6\xBC\xA2";
    String.concat ""
      (List.init 40 (fun i -> if i mod 7 = 3 then "\xC3\x89" else "aB"));
  ]

let same_alignment ~msg n reference text =
  equal ~msg (pair string string)
    (char_spans reference text)
    (char_spans n text)

(* With [clean_text] and [handle_chinese_chars] off, [bert] runs the stages its
   other options name, text and alignment alike. *)
let test_bert_stages text =
  let bert = Normalizer.bert ~clean_text:false ~handle_chinese_chars:false in
  same_alignment ~msg:"lowercase alone"
    (bert ~strip_accents:(Some false) ())
    Normalizer.lowercase text;
  same_alignment ~msg:"strip_accents alone"
    (bert ~lowercase:false ~strip_accents:(Some true) ())
    (Normalizer.sequence [ Normalizer.nfd; Normalizer.strip_accents ])
    text;
  same_alignment ~msg:"both" (bert ())
    (Normalizer.sequence
       [ Normalizer.nfd; Normalizer.strip_accents; Normalizer.lowercase ])
    text;
  equal ~msg:"neither" string text
    (Normalizer.apply
       (bert ~lowercase:false ~strip_accents:(Some false) ())
       text)

(* brot.mli's sets. *)
let is_bert_control u =
  (match Uucp.Gc.general_category u with `Cc | `Cf | `Co -> true | _ -> false)
  && not (List.mem (Uchar.to_int u) [ 0x09; 0x0A; 0x0D ])

let is_cjk_ideograph u =
  let code = Uchar.to_int u in
  List.exists
    (fun (first, last) -> code >= first && code <= last)
    [
      (0x3400, 0x4DBF);
      (0x4E00, 0x9FFF);
      (0xF900, 0xFAFF);
      (0x20000, 0x2A6DF);
      (0x2A700, 0x2B81F);
      (0x2B920, 0x2CEAF);
      (0x2F800, 0x2FA1F);
    ]

(* On ASCII [bert ()] is a byte map: controls go, white space becomes a space,
   letters lower, and every byte kept stands for itself. *)
let bert_ascii c =
  match c with
  | '\t' | '\n' | '\r' | ' ' -> " "
  | '!' .. '~' -> String.make 1 (Char.lowercase_ascii c)
  | _ -> ""

(* [apply] lowers eight printable bytes at a time, by arithmetic whose bounds
   are the bytes around the capitals, '@' and '['; a word holding a control goes
   a byte at a time. *)
let ascii_text =
  let open Gen in
  string_of ~size:(int_range 0 40)
    (frequency
       [
         (8, char_range ' ' '~');
         (3, of_list [ '@'; '['; 'A'; 'Z'; '`'; '{' ]);
         (1, char_range '\x00' '\x1F');
         (1, constant '\x7F');
       ])

(* An eight-byte word of [text] from its start, printable and holding a byte
   next to the capitals. *)
let has_capital_bound_word text =
  let printable c = c >= ' ' && c <= '~' in
  List.exists
    (fun k ->
      let word = String.sub text (8 * k) 8 in
      String.for_all printable word
      && (String.contains word '@' || String.contains word '['))
    (List.init (String.length text / 8) Fun.id)

let test_bert_ascii text =
  cover "a printable word holding '@' or '['" (has_capital_bound_word text);
  let kept =
    List.filter
      (fun i -> bert_ascii text.[i] <> "")
      (List.init (String.length text) Fun.id)
  in
  let expected =
    String.concat "" (List.map (fun i -> bert_ascii text.[i]) kept)
  in
  equal ~msg:"apply" string expected
    (Normalizer.apply (Normalizer.bert ()) text);
  equal ~msg:"apply_aligned" (pair string string)
    ( expected,
      String.concat " "
        (List.map (fun i -> Printf.sprintf "%d,%d" i (i + 1)) kept) )
    (char_spans (Normalizer.bert ()) text)

let bert_ascii_examples =
  [
    "\x00\x07\x7F";
    "  Hello\tWorld\r\n";
    String.make 100 'A';
    "@[`{~!AZ@[`{~!AZ";
  ]

let bert_tests =
  [
    (* Hugging Face, the bert-base-uncased settings. *)
    cases ~id:row_id "strips only nonspacing marks, after decomposing"
      [
        (* ["नमस्ते हिन्दी"] keeps its vowel signs ि and ी. *)
        ( "\u{928}\u{92E}\u{938}\u{94D}\u{924}\u{947} \
           \u{939}\u{93F}\u{928}\u{94D}\u{926}\u{940}",
          "\u{928}\u{92E}\u{938}\u{924} \u{939}\u{93F}\u{928}\u{926}\u{940}" );
        ("a\u{903}\u{20DD}\u{301}", "a\u{903}\u{20DD}");
        ("caf\u{E9}", "cafe");
        ("A\u{301}BC", "abc");
        ("\u{130}", "i");
        ("\u{1FBC}", "\u{3B1}");
      ]
      (normalizes (Normalizer.bert ()));
    test "removes controls and U+FFFD and turns White_Space into a space"
      (fun () ->
        maps_every_scalar
          (Normalizer.bert ~handle_chinese_chars:false
             ~strip_accents:(Some false) ~lowercase:false ()) (fun u ->
            let code = Uchar.to_int u in
            if is_bert_control u || code = 0xFFFD then ""
            else if Uucp.White.is_white_space u then " "
            else utf8 u);
        (* Hugging Face's rows: unassigned code points are kept. *)
        List.iter
          (normalizes (Normalizer.bert ()))
          [
            ("\x00\x07\x7F", "");
            ("a\u{AD}b", "ab");
            ("a\u{200B}b", "ab");
            ("a\u{E000}b", "ab");
            ("a\u{FFFD}b", "ab");
            ("\u{10FFFF}", "\u{10FFFF}");
            ("\u{378}", "\u{378}");
            ("a\u{2028}b", "a b");
            ("a\u{3000}b", "a b");
            ("a\tb\r\nc", "a b  c");
          ]);
    test "pads each CJK ideograph with spaces" (fun () ->
        maps_every_scalar
          (Normalizer.bert ~clean_text:false ~strip_accents:(Some false)
             ~lowercase:false ()) (fun u ->
            if is_cjk_ideograph u then " " ^ utf8 u ^ " " else utf8 u);
        (* Hugging Face's rows: a compatibility ideograph is decomposed, as
           accents are stripped. *)
        List.iter
          (normalizes (Normalizer.bert ()))
          [
            ("a\u{6F22}b", "a \u{6F22} b");
            ("\u{65E5}\u{672C}\u{8A9E}", " \u{65E5}  \u{672C}  \u{8A9E} ");
            ("a\u{F900}b", "a \u{8C48} b");
            ("a\u{2F800}b", "a \u{4E3D} b");
          ]);
    cases ~id:row_id "lowercases without folding"
      [
        ("\u{DF}", "\u{DF}");
        ("\u{1E9E}", "\u{DF}");
        ("\u{FB01}", "\u{FB01}");
        ("\u{3A3}", "\u{3C3}");
      ]
      (normalizes (Normalizer.bert ()));
    test "strips accents iff it lowercases when strip_accents is None"
      (fun () ->
        equal string "a" (Normalizer.apply (Normalizer.bert ()) "\u{C1}");
        equal string "\u{C1}"
          (Normalizer.apply (Normalizer.bert ~lowercase:false ()) "\u{C1}"));
    prop ~count:300 ~examples:bert_corpus
      "agrees with the single stages it combines" (text_of bert_alphabet)
      test_bert_stages;
    prop ~count:300 ~examples:bert_ascii_examples "maps ASCII byte for byte"
      ascii_text test_bert_ascii;
  ]

(* sequence *)

(* The normalizers [sequence] composes in its law. *)
let stages =
  [
    ("nfc", Normalizer.nfc);
    ("nfd", Normalizer.nfd);
    ("nfkd", Normalizer.nfkd);
    ("lowercase", Normalizer.lowercase);
    ("strip_accents", Normalizer.strip_accents);
    ("strip ()", Normalizer.strip ());
    ("replace", Normalizer.replace ~pattern:" " ~replacement:marker);
    ("replace_regex", Normalizer.replace_regex ~pattern:"\\s+" ~replacement:" ");
    ("prepend", Normalizer.prepend marker);
    ("byte_level", Normalizer.byte_level);
    ("nmt", Normalizer.nmt);
    ("bert ()", Normalizer.bert ());
  ]

let stage =
  Gen.of_list ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name) stages

let sequence_alphabet =
  [ "a"; "A"; " "; "  "; "\t"; "\u{E9}"; "e\u{301}"; "\u{130}"; "\u{6F22}";
    "\x01"; "\u{200B}"; "\u{FB01}" ]
[@@ocamlformat "disable"]

let test_sequence (stages, text) =
  cover "no normalizer" (List.length stages = 0);
  cover "two normalizers or more" (List.length stages >= 2);
  equal string
    (List.fold_left (fun s (_, n) -> Normalizer.apply n s) text stages)
    (Normalizer.apply (Normalizer.sequence (List.map snd stages)) text)

(* CLIP's normalizer, as its tokenizer file writes it. *)
let clip =
  Normalizer.sequence
    [
      Normalizer.nfc;
      Normalizer.replace_regex ~pattern:"\\s+" ~replacement:" ";
      Normalizer.lowercase;
    ]

let sequence_tests =
  [
    prop ~count:300 "applies its normalizers left to right"
      (Gen.pair
         (Gen.list ~size:(Gen.int_range 0 3) stage)
         (text_of sequence_alphabet))
      test_sequence;
    (* The example of the module's documentation. *)
    cases ~id:row_id "removes accents and lowercases as the module example does"
      [
        ("Caf\u{E9}", "cafe");
        ("caf\u{E9} NA\u{CF}VE r\u{E9}sum\u{E9}", "cafe naive resume");
      ]
      (normalizes
         (Normalizer.sequence
            [ Normalizer.nfd; Normalizer.strip_accents; Normalizer.lowercase ]));
    (* Hugging Face. *)
    cases ~id:row_id "normalizes as CLIP's sequence"
      [
        ("A\u{301}  Photo\t\nof\u{A0}a\u{3000}CAT ", "\u{E1} photo of a cat ");
        ("x", "x");
      ]
      (normalizes clip);
  ]

(* Alignment *)

(* identity *)

let boundaries s =
  let rec go i acc =
    if i >= String.length s then List.rev (i :: acc)
    else go (i + Uchar.utf_decode_length (String.get_utf_8_uchar s i)) (i :: acc)
  in
  go 0 []

let test_identity_spans text =
  let a = Normalizer.identity text and bounds = boundaries text in
  List.iter
    (fun start ->
      List.iter
        (fun stop ->
          if start <= stop then
            equal
              ~msg:(Printf.sprintf "%d,%d" start stop)
              (pair int int) (start, stop)
              (Normalizer.original_span a ~start ~stop))
        bounds)
    bounds

let span_id (_, (start, stop), _) = Printf.sprintf "%d,%d" start stop

let spans_as a (_, (start, stop), expected) =
  equal (pair int int) expected (Normalizer.original_span a ~start ~stop)

let identity_tests =
  [
    prop ~count:200 ~examples:[ "" ]
      "reports a span of whole characters as itself" nf_text test_identity_spans;
    cases ~id:span_id "reports a character cut short whole"
      [
        ("caf\u{E9}", (3, 4), (3, 5));
        ("caf\u{E9}", (4, 5), (3, 5));
        ("caf\u{E9}", (2, 4), (2, 5));
      ]
      (fun row -> spans_as (Normalizer.identity "caf\u{E9}") row);
  ]

(* apply_aligned *)

let bert_variants =
  List.concat_map
    (fun clean_text ->
      List.concat_map
        (fun handle_chinese_chars ->
          List.concat_map
            (fun strip_accents ->
              List.map
                (fun lowercase ->
                  ( Printf.sprintf
                      "bert ~clean_text:%b ~handle_chinese_chars:%b \
                       ~strip_accents:%s ~lowercase:%b"
                      clean_text handle_chinese_chars
                      (match strip_accents with
                      | None -> "None"
                      | Some b -> Printf.sprintf "(Some %b)" b)
                      lowercase,
                    Normalizer.bert ~clean_text ~handle_chinese_chars
                      ~strip_accents ~lowercase () ))
                [ true; false ])
            [ None; Some true; Some false ])
        [ true; false ])
    [ true; false ]

(* Every normalizer, each option of [bert] set both ways, and sequences of the
   kinds tokenizer files hold. *)
let every_normalizer () =
  [
    ("nfc", Normalizer.nfc);
    ("nfd", Normalizer.nfd);
    ("nfkc", Normalizer.nfkc);
    ("nfkd", Normalizer.nfkd);
    ("lowercase", Normalizer.lowercase);
    ("strip_accents", Normalizer.strip_accents);
    ("strip", Normalizer.strip ());
    ("strip ~left:false", Normalizer.strip ~left:false ());
    ("replace", Normalizer.replace ~pattern:" " ~replacement:marker);
    ("replace empty", Normalizer.replace ~pattern:"" ~replacement:"_");
    ("replace_regex", Normalizer.replace_regex ~pattern:"\\s+" ~replacement:" ");
    ( "replace_regex empty",
      Normalizer.replace_regex ~pattern:"\\p{L}*" ~replacement:"_" );
    ("prepend", Normalizer.prepend marker);
    ("byte_level", Normalizer.byte_level);
    ("nmt", Normalizer.nmt);
    ("precompiled", Lazy.force precompiled);
    ( "strip prepend replace",
      Normalizer.sequence
        [
          Normalizer.strip ();
          Normalizer.prepend marker;
          Normalizer.replace ~pattern:" " ~replacement:marker;
        ] );
    ( "prepend strip",
      Normalizer.sequence [ Normalizer.prepend "  x"; Normalizer.strip () ] );
    ( "nfkc strip_accents lowercase",
      Normalizer.sequence
        [ Normalizer.nfkc; Normalizer.strip_accents; Normalizer.lowercase ] );
    ("clip", clip);
  ]
  @ bert_variants

(* Characters every normalizer does something to, and bytes that start no
   character. *)
let aligned_alphabet =
  [ "a"; "Z"; " "; "  "; "\t"; "\n"; "\r\n"; "\x00"; "\x01"; "\x7F"; "\u{E9}";
    "e\u{301}"; "\u{301}"; "\u{316}"; "\u{1E9B}\u{323}"; "\u{AC01}";
    "\u{1100}\u{1161}\u{11A8}"; "\u{6F22}"; "\u{FB01}"; "\u{FDFA}"; "\u{130}";
    "\u{AD}"; "\u{200B}"; "\u{3000}"; "\u{FFFD}"; "\u{1D165}"; "\u{1D16D}";
    "\u{928}\u{94D}"; "\u{1F468}\u{200D}\u{1F469}"; "\u{FF45}\u{301}"; "HELLO@[`{~!";
    "\u{2122}"; "\xFF"; "\xC3"; "\xE2\x82" ]
[@@ocamlformat "disable"]

let has_stray_byte text =
  List.exists (fun (code, _) -> code = None) (characters text)

let aligned_text_is_text text =
  List.iter
    (fun (label, n) ->
      equal ~msg:label string (Normalizer.apply n text)
        (fst (Normalizer.apply_aligned n text)))
    (every_normalizer ())

let test_aligned_text text =
  cover "a byte that starts no character" (has_stray_byte text);
  aligned_text_is_text text

let parity_corpora () =
  [
    Fixture.read "support/parity/edge_cases.txt";
    Fixture.read "support/parity/sample.txt";
  ]

(* apply *)

(* Bytes that open no whole character: a stray lead, a stray continuation, a
   sequence cut short, an overlong encoding, a surrogate and a value above
   U+10FFFF (RFC 3629). *)
let stray_bytes =
  [ "\xFF"; "\x80"; "\xC3"; "\xE2\x82"; "\xC0\xAF"; "\xED\xA0\x80";
    "\xF4\x90\x80\x80" ]
[@@ocamlformat "disable"]

(* The normalizers of [every_normalizer] that leave ["a"] and ["b"] alone and
   insert nothing. *)
let leaves_ascii_letters (label, _) =
  List.mem label
    [ "nfc"; "nfd"; "nfkc"; "nfkd"; "lowercase"; "strip_accents"; "strip";
      "strip ~left:false"; "replace"; "replace_regex"; "nmt";
      "nfkc strip_accents lowercase"; "clip" ]
  || String.starts_with ~prefix:"bert" label
[@@ocamlformat "disable"]

(* Source: brot.mli, encode: a byte that opens no whole character is read by
   every stage as a character of one byte. *)
let test_stray_bytes_kept stray =
  let text = "a" ^ stray ^ "b" in
  List.iter
    (fun (label, n) ->
      let normalized, a = Normalizer.apply_aligned n text in
      equal ~msg:label string text normalized;
      String.iteri
        (fun i _ ->
          equal
            ~msg:(Printf.sprintf "%s, byte %d" label (i + 1))
            (pair int int)
            (i + 1, i + 2)
            (Normalizer.original_span a ~start:(i + 1) ~stop:(i + 2)))
        stray)
    (List.filter leaves_ascii_letters (every_normalizer ()))

let apply_tests =
  [
    cases ~id:text_id
      "keeps a byte that opens no whole character as a character of its own"
      stray_bytes test_stray_bytes_kept;
    cases
      ~id:(fun (label, _, text, _) -> Printf.sprintf "%s %S" label text)
      "normalizes the characters around a byte that opens none"
      [
        ("lowercase", Normalizer.lowercase, "A\xFFB", "a\xFFb");
        ("bert", Normalizer.bert (), "A\xFFB", "a\xFFb");
        ("strip_accents", Normalizer.strip_accents, "A\xFFB", "A\xFFB");
        ("nfc", Normalizer.nfc, "e\u{301}\xE2\x82", "\u{E9}\xE2\x82");
        ( "replace empty",
          Normalizer.replace ~pattern:"" ~replacement:"_",
          "a\xE2\x82b",
          "_a_\xE2_\x82_b_" );
      ]
      (fun (_, n, text, expected) ->
        equal string expected (Normalizer.apply n text));
  ]

let apply_aligned_tests =
  [
    prop ~count:200 "gives the text apply gives" (text_of aligned_alphabet)
      test_aligned_text;
    test "gives the text apply gives on the parity corpora" (fun () ->
        List.iter aligned_text_is_text (parity_corpora ()));
  ]

(* original_span *)

(* support/parity/normalizer_alignments.txt, from test/gen/hf_alignments.py: per
   line, a label naming a normalizer, an input, the text Hugging Face normalizes
   it to and the span of each of its characters. *)
let hf_alignments =
  lazy
    (Fixture.read "support/parity/normalizer_alignments.txt"
    |> String.split_on_char '\n'
    |> List.filter (fun line -> line <> "")
    |> List.map (fun line ->
        Scanf.sscanf line "%S %S %S %S%!" (fun label input text spans ->
            (label, input, text, spans))))

let aligns_as_hf label n =
  let rows =
    List.filter (fun (l, _, _, _) -> l = label) (Lazy.force hf_alignments)
  in
  if rows = [] then failf "normalizer_alignments.txt has no %S row" label;
  List.iter
    (fun (_, input, text, spans) ->
      equal
        ~msg:(Printf.sprintf "%s %S" label input)
        (pair string string) (text, spans) (char_spans n input))
    rows

let test_spans_in_order text =
  List.iter
    (fun (label, n) ->
      let normalized, a = Normalizer.apply_aligned n text in
      let previous = ref 0 in
      for byte = 0 to String.length normalized - 1 do
        let msg = Printf.sprintf "%s, byte %d" label byte in
        let start, stop =
          Normalizer.original_span a ~start:byte ~stop:(byte + 1)
        in
        at_least ~msg int ~than:!previous start;
        at_least ~msg int ~than:start stop;
        at_most ~msg int ~than:(String.length text) stop;
        previous := start
      done)
    (every_normalizer ())

let strip = Normalizer.strip ()
let prepend = Normalizer.prepend marker
let replace = Normalizer.replace ~pattern:" " ~replacement:marker

let original_span_tests =
  [
    test "gives the spans Hugging Face gives under nfd" (fun () ->
        aligns_as_hf "nfd" Normalizer.nfd);
    test "gives the spans Hugging Face gives under nfc" (fun () ->
        aligns_as_hf "nfc" Normalizer.nfc);
    test "gives the spans Hugging Face gives under nfkc and nfkd" (fun () ->
        aligns_as_hf "nfkc" Normalizer.nfkc;
        aligns_as_hf "nfkd" Normalizer.nfkd);
    test "gives the spans Hugging Face gives under lowercase" (fun () ->
        aligns_as_hf "lowercase" Normalizer.lowercase);
    test "gives a removed character no span" (fun () ->
        aligns_as_hf "strip accents" Normalizer.strip_accents;
        aligns_as_hf "strip" strip;
        aligns_as_hf "strip left" (Normalizer.strip ~right:false ());
        aligns_as_hf "strip right" (Normalizer.strip ~left:false ()));
    test "gives a replacement the span of the last character it replaced"
      (fun () ->
        aligns_as_hf "replace string" replace;
        aligns_as_hf "replace collapse"
          (Normalizer.replace_regex ~pattern:"\\s+" ~replacement:" ");
        aligns_as_hf "replace grow"
          (Normalizer.replace ~pattern:"a" ~replacement:"xyz");
        aligns_as_hf "replace shrink"
          (Normalizer.replace_regex ~pattern:"ab+" ~replacement:"X");
        aligns_as_hf "replace delete"
          (Normalizer.replace_regex ~pattern:"\\s+" ~replacement:"");
        aligns_as_hf "replace class"
          (Normalizer.replace_regex ~pattern:"\\p{L}+" ~replacement:"L"));
    test
      "gives a replacement of nothing the span of the character before it, or \
       an empty span at the start" (fun () ->
        aligns_as_hf "replace empty"
          (Normalizer.replace_regex ~pattern:"x*" ~replacement:"_");
        aligns_as_hf "replace empty string"
          (Normalizer.replace ~pattern:"" ~replacement:"_");
        aligns_as_hf "replace anchor"
          (Normalizer.replace_regex ~pattern:"$" ~replacement:"!"));
    test "gives a prepended text the span of the first character" (fun () ->
        aligns_as_hf "prepend" prepend;
        aligns_as_hf "prepend multi" (Normalizer.prepend "<<"));
    test "gives each byte of a sequence RFC 3629 forbids a span of its own"
      (fun () ->
        (* A surrogate's three bytes are three characters of one byte (brot.mli,
           encode), so the prepended text stands at the first of them. *)
        let _, a =
          Normalizer.apply_aligned (Normalizer.prepend "_") "\xed\xa0\x80b"
        in
        equal
          (list (pair int int))
          [ (0, 1); (0, 1); (1, 2); (2, 3); (3, 4) ]
          (List.init 5 (fun i ->
               Normalizer.original_span a ~start:i ~stop:(i + 1))));
    test "composes the spans of a sequence" (fun () ->
        let seq = Normalizer.sequence in
        aligns_as_hf "llama" (seq [ prepend; replace ]);
        aligns_as_hf "nfd strip lower"
          (seq
             [ Normalizer.nfd; Normalizer.strip_accents; Normalizer.lowercase ]);
        aligns_as_hf "strip prepend" (seq [ strip; prepend ]);
        aligns_as_hf "prepend strip" (seq [ Normalizer.prepend "<<"; strip ]);
        aligns_as_hf "prepend strip left"
          (seq [ Normalizer.prepend "  x"; Normalizer.strip ~right:false () ]);
        aligns_as_hf "prepend prepend" (seq [ prepend; Normalizer.prepend "x" ]);
        aligns_as_hf "strip strip"
          (seq
             [
               Normalizer.strip ~right:false (); Normalizer.strip ~left:false ();
             ]);
        aligns_as_hf "prepend strip replace" (seq [ prepend; strip; replace ]);
        aligns_as_hf "strip prepend replace" (seq [ strip; prepend; replace ]);
        aligns_as_hf "strip lowercase" (seq [ strip; Normalizer.lowercase ]);
        aligns_as_hf "prepend nfd" (seq [ prepend; Normalizer.nfd ]);
        aligns_as_hf "clip" clip);
    test
      "gives each character byte_level writes the span of its byte's character"
      (fun () -> aligns_as_hf "byte level" Normalizer.byte_level);
    test "gives the spans Hugging Face gives under nmt" (fun () ->
        aligns_as_hf "nmt" Normalizer.nmt);
    test "gives the spans Hugging Face gives under bert" (fun () ->
        aligns_as_hf "bert" (Normalizer.bert ());
        aligns_as_hf "bert nolower" (Normalizer.bert ~lowercase:false ());
        aligns_as_hf "bert noclean" (Normalizer.bert ~clean_text:false ());
        aligns_as_hf "bert nostrip"
          (Normalizer.bert ~strip_accents:(Some false) ()));
    test "gives the spans Hugging Face gives under precompiled" (fun () ->
        aligns_as_hf "precompiled" (Lazy.force precompiled));
    cases ~id:string_of_int
      "gives a replacement the span of the character it replaced at every \
       offset"
      (List.init 25 Fun.id) (fun k ->
        let text = String.make k 'a' ^ " " ^ String.make (24 - k) 'b' in
        let _, a = Normalizer.apply_aligned replace text in
        equal (pair int int)
          (k, k + 1)
          (Normalizer.original_span a ~start:k ~stop:(k + 3)));
    cases ~id:span_id "gives an empty span for an empty range"
      [
        ("caf\u{E9}", (0, 0), (0, 0));
        ("caf\u{E9}", (2, 2), (2, 2));
        ("caf\u{E9}", (5, 5), (5, 5));
      ]
      (fun row -> spans_as (Normalizer.identity "caf\u{E9}") row);
    cases
      ~id:(fun (label, _, input, _) -> Printf.sprintf "%s %S" label input)
      "gives the whole original text for an empty range of an emptied text"
      [
        ("strip", Normalizer.strip (), "   ", (0, 3));
        ("bert", Normalizer.bert (), "\x00", (0, 1));
        ("strip", Normalizer.strip (), "", (0, 0));
      ]
      (fun (_, n, input, expected) ->
        let text, a = Normalizer.apply_aligned n input in
        equal
          (pair string (pair int int))
          ("", expected)
          (text, Normalizer.original_span a ~start:0 ~stop:0));
    prop ~count:200 "never goes backwards and stays inside the input"
      (text_of aligned_alphabet) test_spans_in_order;
    test "never goes backwards and stays inside the parity corpora" (fun () ->
        List.iter test_spans_in_order (parity_corpora ()));
    cases ~id:span_id "raises Invalid_argument unless 0 <= start <= stop <= n"
      [
        ("caf\u{E9}", (-1, 0), (0, 0));
        ("caf\u{E9}", (2, 1), (0, 0));
        ("caf\u{E9}", (0, 6), (0, 0));
        ("caf\u{E9}", (6, 6), (0, 0));
      ]
      (fun (text, (start, stop), _) ->
        raises_match Exn.invalid_arg (fun () ->
            Normalizer.original_span (Normalizer.identity text) ~start ~stop));
  ]

(* to_json and of_json *)

let json_text json =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json json with
  | Ok text -> text
  | Error msg -> failf "cannot encode: %s" msg

let json_of_text text =
  match Jsont_bytesrw.decode_string Jsont.json text with
  | Ok json -> json
  | Error msg -> failf "cannot decode %S: %s" text msg

let read_normalizer text = require_ok (Normalizer.of_json (json_of_text text))
let write_normalizer n = json_text (Normalizer.to_json n)

(* Each normalizer with the JSON Hugging Face reads back as the same normalizer.
   [Replace] writes its pattern as a [String] or a [Regex]. *)
let normalizer_json =
  [
    (Normalizer.nfc, {|{"type":"NFC"}|});
    (Normalizer.nfd, {|{"type":"NFD"}|});
    (Normalizer.nfkc, {|{"type":"NFKC"}|});
    (Normalizer.nfkd, {|{"type":"NFKD"}|});
    (Normalizer.lowercase, {|{"type":"Lowercase"}|});
    (Normalizer.strip_accents, {|{"type":"StripAccents"}|});
    ( Normalizer.strip (),
      {|{"type":"Strip","strip_left":true,"strip_right":true}|} );
    ( Normalizer.strip ~left:false (),
      {|{"type":"Strip","strip_left":false,"strip_right":true}|} );
    ( Normalizer.replace ~pattern:" " ~replacement:marker,
      {|{"type":"Replace","pattern":{"String":" "},"content":"|} ^ marker
      ^ {|"}|} );
    ( Normalizer.replace_regex ~pattern:"\\s+" ~replacement:" ",
      {|{"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "}|} );
    ( Normalizer.prepend marker,
      {|{"type":"Prepend","prepend":"|} ^ marker ^ {|"}|} );
    (Normalizer.byte_level, {|{"type":"ByteLevel"}|});
    (Normalizer.nmt, {|{"type":"Nmt"}|});
    ( Normalizer.bert (),
      {|{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true}|}
    );
    ( Normalizer.bert ~clean_text:false ~strip_accents:(Some false)
        ~lowercase:false (),
      {|{"type":"BertNormalizer","clean_text":false,"handle_chinese_chars":true,"strip_accents":false,"lowercase":false}|}
    );
    ( clip,
      {|{"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "},{"type":"Lowercase"}]}|}
    );
  ]

let charsmap_member json =
  match member "precompiled_charsmap" json with
  | Jsont.String (s, _) -> s
  | _ -> failf "the charsmap is not a string"

let precompiled_of_base64 b64 =
  Normalizer.of_json
    (Jsont.Json.object'
       [
         (Jsont.Json.name "type", Jsont.Json.string "Precompiled");
         (Jsont.Json.name "precompiled_charsmap", Jsont.Json.string b64);
       ])

let to_json_tests =
  [
    cases ~id:snd "writes JSON Hugging Face reads back as the same normalizer"
      normalizer_json (fun (n, expected) ->
        equal string expected (write_normalizer n));
    test "writes back the charsmap it read" (fun () ->
        equal string
          (charsmap_member (charsmap_json ()))
          (charsmap_member (Normalizer.to_json (Lazy.force precompiled))));
  ]

let of_json_tests =
  [
    cases ~id:snd "reads back what to_json writes" normalizer_json
      (fun (_, text) ->
        Law.round_trip string pass read_normalizer write_normalizer text);
    test "reads a ByteLevel normalizer written with its pre-tokenizer's members"
      (fun () ->
        equal string {|{"type":"ByteLevel"}|}
          (write_normalizer
             (read_normalizer
                {|{"type":"ByteLevel","add_prefix_space":true,"use_regex":true}|})));
    (* Hugging Face reads the padding of the last group complete, partial or
       absent. *)
    test "reads a base64 charsmap with all, part or none of its padding"
      (fun () ->
        let padded =
          charsmap_member
            (Normalizer.to_json (Normalizer.precompiled (a_to_b ())))
        in
        let bare = String.sub padded 0 (String.index padded '=') in
        List.iter
          (fun b64 ->
            let n = require_ok (precompiled_of_base64 b64) in
            equal ~msg:b64 string "bbc" (Normalizer.apply n "abc"))
          [ padded; bare; bare ^ "=" ]);
    cases ~id:Fun.id "rejects base64 that is not canonical"
      [ "AAA*"; "AB=="; "AB"; "AA==="; "AAAA="; "AAAAA"; "AA==AAAA" ]
      (fun b64 ->
        equal (result pass string) (Error "precompiled_charsmap is not base64")
          (precompiled_of_base64 b64));
    test "rejects a pattern it cannot translate, with the reason" (fun () ->
        equal (result pass string)
          (Error
             {|invalid regular expression "(?>a)": atomic groups are not supported|})
          (Normalizer.of_json
             (json_of_text
                {|{"type":"Replace","pattern":{"Regex":"(?>a)"},"content":"b"}|})));
    cases ~id:Fun.id
      "errors on a value that is not an object or has no known type"
      [ {|[]|}; {|"NFC"|}; {|{}|}; {|{"type":3}|}; {|{"type":"Nope"}|} ]
      (fun text ->
        is_error ~pp:Normalizer.pp (Normalizer.of_json (json_of_text text)));
    cases ~id:fst "reads a missing member as its constructor's default"
      [
        ({|{"type":"Strip"}|}, Normalizer.strip ());
        ({|{"type":"BertNormalizer"}|}, Normalizer.bert ());
      ]
      (fun (text, default) ->
        equal string (write_normalizer default)
          (write_normalizer (read_normalizer text)));
    cases ~id:Fun.id "errors on a missing member its constructor requires"
      [
        {|{"type":"Prepend"}|};
        {|{"type":"Replace","pattern":{"String":"a"}}|};
        {|{"type":"Replace","content":"a"}|};
        {|{"type":"Precompiled"}|};
        {|{"type":"Precompiled","precompiled_charsmap":""}|};
      ] (fun text ->
        is_error ~pp:Normalizer.pp (Normalizer.of_json (json_of_text text)));
    (* Hugging Face 0.23.1 rejects each row, and reads it with the member under
       test given a value of its type. *)
    cases ~id:Fun.id "errors on a member of the wrong type"
      [
        {|{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":"false"}|};
        {|{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":3}|};
        {|{"type":"BertNormalizer","clean_text":null,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true}|};
        {|{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":"no","lowercase":true}|};
        {|{"type":"Strip","strip_left":"x","strip_right":true}|};
        {|{"type":"Strip","strip_left":true,"strip_right":1}|};
      ] (fun text ->
        is_error ~pp:Normalizer.pp (Normalizer.of_json (json_of_text text)));
  ]

let () =
  exit
    (run "Normalizer"
       [
         group "nfc" nfc_tests;
         group "nfd" nfd_tests;
         group "nfkc" nfkc_tests;
         group "nfkd" nfkd_tests;
         group "lowercase" lowercase_tests;
         group "strip_accents" strip_accents_tests;
         group "strip" strip_tests;
         group "replace" replace_tests;
         group "replace_regex" replace_regex_tests;
         group "prepend" prepend_tests;
         group "byte_level" byte_level_tests;
         group "nmt" nmt_tests;
         group "precompiled" precompiled_tests;
         group "bert" bert_tests;
         group "sequence" sequence_tests;
         group "apply" apply_tests;
         group "identity" identity_tests;
         group "apply_aligned" apply_aligned_tests;
         group "original_span" original_span_tests;
         group "to_json" to_json_tests;
         group "of_json" of_json_tests;
       ])
