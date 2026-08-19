(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Unicode processing tests for brot *)

open Windtrap
open Brot

(* Normalization via public API *)

let test_lowercase_normalization () =
  let text = "HELLO WORLD" in
  let normalizer = Normalizer.lowercase in
  let result = Normalizer.apply normalizer text in
  equal ~msg:"lowercase" string "hello world" result

let test_strip_accents_normalization () =
  let text = "caf\xC3\xA9 na\xC3\xAFve r\xC3\xA9sum\xC3\xA9" in
  let normalizer =
    Normalizer.sequence [ Normalizer.nfd; Normalizer.strip_accents ]
  in
  let result = Normalizer.apply normalizer text in
  equal ~msg:"strip accents" string "cafe naive resume" result

let test_normalization_sequence () =
  let text = "  HELLO  World  " in
  let normalizer =
    Normalizer.sequence
      [
        Normalizer.lowercase;
        Normalizer.strip ();
        Normalizer.replace_regex ~pattern:"\\s+" ~replacement:" ";
      ]
  in
  let result = Normalizer.apply normalizer text in
  equal ~msg:"sequence" string "hello world" result

(* Expectations from HuggingFace [normalizers.Replace(pattern, content)] with a
   string pattern. *)
let test_replace () =
  let case pattern replacement text expected =
    equal
      ~msg:(Printf.sprintf "replace %S by %S in %S" pattern replacement text)
      string expected
      (Normalizer.apply (Normalizer.replace ~pattern ~replacement) text)
  in
  case " " "\xE2\x96\x81" "a  b" "a\xE2\x96\x81\xE2\x96\x81b";
  case "" "_" "ab" "_a_b_";
  case "" "_" "" "";
  case "" "_" "\xC3\xA9" "_\xC3\xA9_";
  case "a" "bb" "aaa" "bbbbbb";
  case "aa" "b" "aaaaa" "bba";
  case "a.b" "X" "a.baxb" "Xaxb";
  case "\\s" "X" "a\\sb b" "aXb b";
  case "\xC3\xA9" "e" "a\xC3\xA9\xC3\xA9b" "aeeb";
  case "``" "\"" "``a''" "\"a''";
  case "ab" "_" "a" "a";
  case "abc" "_" "ab" "ab"

(* Expectations from HuggingFace [normalizers.Replace(Regex(pattern), content)],
   whose engine matches Unicode text: [\s], [\d], [\w], [.], [\p{..}] and
   negated classes stand for characters, [^] and [$] are line anchors, [{n}?] is
   an optional repetition, an empty match right after a match is skipped and the
   search moves on by whole characters. Regenerate with [uv run --with
   tokenizers python3] on the same calls. *)
let test_replace_regex () =
  let case pattern replacement text expected =
    equal
      ~msg:
        (Printf.sprintf "replace regex %S by %S in %S" pattern replacement text)
      string expected
      (Normalizer.apply (Normalizer.replace_regex ~pattern ~replacement) text)
  in
  case "\\s+" " "
    "a  b\x09\x0Ac\xC2\xA0d\xE3\x80\x80e\xE2\x80\x8Bf\xEF\xBB\xBFg\xE2\x80\xA8h"
    "a b c d e\xE2\x80\x8Bf\xEF\xBB\xBFg h";
  case "\\s+" "_"
    "a\xE1\x9A\x80b\xE2\x80\x80\xE2\x80\x8Ac\xE2\x80\xAFd\xE2\x81\x9Fe\xC2\x85f\x0Bg\x0Ch"
    "a_b_c_d_e_f_g_h";
  case " {2,}" " " "a  b c   d    e" "a b c d e";
  case "\\d" "#" "a1\xD9\xA3\xC2\xB2\xE0\xA5\xA7\xEF\xBC\x91b" "a##\xC2\xB2##b";
  case "\\d+" "#" "12ab\xD9\xA3\xD9\xA4c" "#ab#c";
  case "\\D+" "#" "12ab\xD9\xA3\xD9\xA4c" "12#\xD9\xA3\xD9\xA4#";
  case "\\w+" "_"
    "h\xC3\xA9llo w\xC3\xB6rld_1 \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E! \
     \xC2\xB2\xC2\xBD \xE2\x80\x8C\xE2\x80\x8D x"
    "_ _ _! _ \xE2\x80\x8C\xE2\x80\x8D _";
  case "\\W+" "_"
    "h\xC3\xA9llo w\xC3\xB6rld_1 \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E! \
     \xC2\xB2\xC2\xBD x"
    "h\xC3\xA9llo_w\xC3\xB6rld_1_\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E_\xC2\xB2\xC2\xBD_x";
  case "\\w" "_" "a\xCC\x81\xE2\x85\xA0\xE2\x93\x90\xE2\x91\xA0\xE2\x88\x80"
    "____\xE2\x91\xA0\xE2\x88\x80";
  (* The Latin-1 quirk applies to a standalone [\w] only, not in a class. *)
  case "[\\w]" "_" "a\xC2\xB2" "_\xC2\xB2";
  case "[^\\W]+" "_" "a\xC2\xB2b" "_\xC2\xB2_";
  case "\\w+" "_" "a\xC2\xB2b" "_";
  case "\\S+" "_" " a\xC2\xA0b " " _\xC2\xA0_ ";
  case "." "_" "a\x0A\xF0\x9F\x98\x80b\x0D" "_\x0A___";
  case ".+" "_" "ab\x0Acd" "_\x0A_";
  case "[^a]" "_" "a\x0A\xF0\x9F\x98\x80b" "a___";
  case "[^\\w\\s]+" "_" "a-b_c d.e\xE2\x80\x9C" "a_b_c d_e_";
  case "[\\r\\n]+" " " "a\x0D\x0Ab\x0A\x0Ac\x0Dd" "a b c d";
  case "\\r\\n|\\n" " " "a\x0D\x0Ab\x0A\x0Ac" "a b  c";
  case "\\n|\\r\\n" " " "a\x0D\x0Ab\x0A\x0Ac" "a b  c";
  case "\\p{L}+" "_" "ab1cd \xC3\xA9\xE6\x97\xA5 \xCC\x81" "_1_ _ \xCC\x81";
  case "\\P{L}+" "_" "ab1cd \xC3\xA9\xE6\x97\xA5 \xCC\x81"
    "ab_cd_\xC3\xA9\xE6\x97\xA5_";
  case "\\p{^L}+" "_" "ab1cd \xC3\xA9\xE6\x97\xA5" "ab_cd_\xC3\xA9\xE6\x97\xA5";
  case "\\P{^L}+" "_" "ab1cd \xC3\xA9\xE6\x97\xA5" "_1_ _";
  case "\\p{Lu}" "_" "aA\xC3\x89\xC3\xA9\xC7\x85" "a__\xC3\xA9\xC7\x85";
  case "\\p{ lu }" "_" "aA\xC3\x89\xC3\xA9" "a__\xC3\xA9";
  case "\\p{L_u}" "_" "aA" "a_";
  case "\\p{Letter}+" "_" "ab1cd" "_1_";
  case "\\p{Uppercase_Letter}" "_" "aAb" "a_b";
  case "\\p{ Decimal-Number }" "_" "a1b" "a_b";
  case "\\p{Nd}+" "_" "a12\xD9\xA3\xC2\xB2b" "a_\xC2\xB2b";
  case "\\p{N}+" "_" "a12\xD9\xA3\xC2\xB2\xE2\x85\xA0b" "a_b";
  case "\\p{P}+" "_" "a-b_c d.e!\xE2\x80\x9C" "a_b_c d_e_";
  case "\\p{Z}+" "_" "a  b\xC2\xA0\xE3\x80\x80c\xE2\x80\xA8d" "a_b_c_d";
  case "\\p{Zs}+" "_" "a  b\xC2\xA0\xE3\x80\x80c\xE2\x80\xA8d"
    "a_b_c\xE2\x80\xA8d";
  case "\\p{C}+" "_" "a\x00\x1F\xE2\x80\x8Bb\xF3\xA0\x82\x80c" "a_b_c";
  case "\\p{Cn}" "_" "a\xF3\xA0\x82\x80b" "a_b";
  case "\\p{Co}" "_" "a\xEE\x80\x80b" "a_b";
  case "\\p{M}+" "_" "e\xCC\x81\xCC\xA3\xE0\xA4\xBF\xE2\x83\x9Dx" "e_x";
  case "\\p{S}+" "_" "a$+^\xC2\xA9b" "a_b";
  case "\\p{Any}" "_" "a\x0Ab" "___";
  case "\\p{LC}" "_" "aA\xC7\x85\xCA\xB0" "___\xCA\xB0";
  case "[\\p{Ll}A]" "_" "aAbB" "___B";
  case "[^\\p{Ll}A]" "_" "aAbB" "aAb_";
  case "[\\P{Ll}]" "_" "aAbB" "a_b_";
  case "[^\\W\\d]+" "_" "a-b1c" "_-_1_";
  case "[\\d-]" "_" "a-1" "a__";
  case "[a-]" "_" "a-b" "__b";
  case "[-a]" "_" "a-b" "__b";
  case "[]a]" "_" "a]b" "__b";
  case "[]-a]" "_" "a]^b" "___b";
  case "[^]]" "_" "a]b" "_]_";
  case "[a^]" "_" "a^b" "__b";
  case "[\\-\\]\\[\\^]" "_" "a-][^b" "a____b";
  case "[a-z]+" "_" "abc\xC3\xA9d" "_\xC3\xA9_";
  case "[\xC3\xA9-\xC3\xBC]+" "_" "a\xC3\xA9\xC3\xB2\xC3\xBCb" "a_b";
  case "[\\x{1F600}-\\x{1F64F}]" "_"
    "a\xF0\x9F\x98\x80\xF0\x9F\x99\x8F\xF0\x9F\x99\x90b" "a__\xF0\x9F\x99\x90b";
  case "\\x{2581}" " " "a\xE2\x96\x81b" "a b";
  case "\\x41\\x{42}" "_" "aABb" "a_b";
  case "\\x4" "_" "a\x04b" "a_b";
  case "\\t\\n\\r\\f\\v\\e\\a" "_" "a\x09\x0A\x0D\x0C\x0B\x1B\x07b" "a_b";
  case "\\0\\07\\077\\0777" "_" "a\x00\x07??7b" "a_b";
  case "\\/\\-\\ \\.\\*\\+\\?\\(\\)\\{\\}\\|\\\\" "_" "a/- .*+?(){}|\\b" "a_b";
  case "\\\xC3\xA9" "_" "a\xC3\xA9b" "a_b";
  case "a{" "_" "a{b" "_b";
  case "a{x}" "_" "a{x}b" "_b";
  case "a{2" "_" "a{2b" "_b";
  case "a{,}" "_" "a{,}b" "_b";
  case "a{2,,3}" "_" "a{2,,3}b" "_b";
  case "a{ 2}" "_" "a{ 2}b" "_b";
  case "a{,2}" "_" "aaab" "__b_";
  case "a{2,}" "_" "aaab" "_b";
  case "a{2,3}" "_" "aaaab" "_ab";
  case "a{02}" "_" "aaab" "_ab";
  case "a{0}" "_" "aab" "_a_a_b_";
  case "a{2}?" "_" "aaaab" "__b_";
  case "a{2,3}?" "_" "aaaab" "__b";
  case "a{1,2}{2}" "_" "aaaaa" "_a";
  (* A lazy quantifier leaves the quantifiers under it greedy. *)
  case "(a*){1,3}?" "_" "a" "_";
  case "(a+?)+" "_" "aaa" "_";
  case "(a*)+?b" "_" "aab" "_";
  case "a*?" "_" "aab" "_a_a_b_";
  case "a??" "_" "aab" "_a_a_b_";
  case "a+?" "_" "aab" "__b";
  case "a**" "_" "aab" "_b_";
  case "a?*" "_" "aab" "_b_";
  case "a*{2}" "_" "aab" "_b_";
  case "x*" "_" "a\xC3\xA9 b" "_a_\xC3\xA9_ _b_";
  case "x*" "_" "xa" "_a_";
  case "x*" "_" "x\xC3\xA9x" "_\xC3\xA9_";
  case "x*" "_" "\xF0\x9F\x98\x80a" "_\xF0\x9F\x98\x80_a_";
  case "x*" "_" "x" "_";
  case "x*" "_" "" "";
  case "" "_" "ab" "_a_b_";
  case "" "_" "" "";
  case "a|" "_" "bab" "_b_b_";
  case "|a" "_" "bab" "_b_a_b_";
  case "(a|)" "_" "ab" "_b_";
  case "(|)" "_" "ab" "_a_b_";
  case "a|ab" "_" "abab" "_b_b";
  case "ab|a" "_" "abab" "__";
  case "(a|ab)(c|bcd)" "_" "abcd" "_";
  case "(a|ab)c" "_" "abc" "_";
  case "a*b|a" "_" "aab" "_";
  case "(a*)(a*)" "_" "aab" "_b_";
  case "a??b" "_" "aab" "a_";
  case "(a|b)*c" "_" "ababc" "_";
  case "(a|b)*?" "_" "aab" "_a_a_b_";
  case "(?:ab)+" "_" "ababc" "_c";
  case "(?<n>ab)+" "_" "ababc" "_c";
  case "(?<_1>ab)+" "_" "ababc" "_c";
  case "a(?#comment)b" "_" "ab" "_";
  case "(?#)a" "_" "aab" "__b";
  case "a$" "_" "a\x0Ab" "_\x0Ab";
  case "a$" "_" "a\x0A" "_\x0A";
  case "a$" "_" "a\x0A\x0A" "_\x0A\x0A";
  case "$" "_" "a\x0Ab" "a_\x0Ab_";
  case "$" "_" "\x0A\x0A" "_\x0A_\x0A_";
  case "$" "_" "" "";
  case "a\\z" "_" "a\x0A" "a\x0A";
  case "\\z" "_" "a\x0Ab\x0A" "a\x0Ab\x0A_";
  case "\\Z" "_" "a\x0Ab\x0A" "a\x0Ab_\x0A_";
  case "\\Z" "_" "a\x0A\x0A" "a\x0A_\x0A_";
  case "a\\Z" "_" "a\x0A\x0A" "a\x0A\x0A";
  case "\\A" "_" "a\x0Ab\x0A" "_a\x0Ab\x0A";
  case "\\Aa" "_" "aa" "_a";
  case "\\Ga" "_" "aab" "__b";
  case "^a" "_" "\x0Aa" "\x0A_";
  case "^\\s+" "_" "  a\x0A  b" "_a\x0A_b";
  case "\\s+$" "_" "a  \x0Ab  " "a_\x0Ab_";
  case "a^b" "_" "a\x0Ab" "a\x0Ab";
  case "a\\n^b" "_" "a\x0Ab" "_";
  case "a$\\nb" "_" "a\x0Ab" "_";
  case "\\n(^b)" "_" "a\x0Ab\x0A" "a_\x0A";
  case "a[^\\x00-\\x{10FFFF}]|b" "_" "axyb" "axy_";
  case "\\p{Cs}" "_" "axyb" "axyb";
  case "\\P{Any}" "_" "axyb" "axyb";
  case "ab*c" "_" "abbc ac abc" "_ _ _";
  case "\\s" "_" "\xF0\x9F\x98\x80 \xF0\x9F\x98\x80"
    "\xF0\x9F\x98\x80_\xF0\x9F\x98\x80";
  (* The pattern CLIP's tokenizer file splits on. *)
  let clip_split =
    "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"
  in
  case clip_split "|"
    "<|startoftext|>a photo of 2 cats' toys, don't!  \xC3\xA9t\xC3\xA9 \
     \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E 42<|endoftext|>"
    "|| | | | || || |||  | | |||";
  case clip_split "|" "it's 3.5% -- ok\xE2\x80\xA6" "|| |||| | ||"

(* What the translation refuses, with a message that says why. *)
let test_replace_regex_rejected () =
  let case pattern reason =
    match Normalizer.replace_regex ~pattern ~replacement:"" with
    | exception Invalid_argument msg ->
        equal
          ~msg:(Printf.sprintf "rejecting %S" pattern)
          string
          (Printf.sprintf "invalid regular expression %S: %s" pattern reason)
          msg
    | _ -> failf "%S was accepted" pattern
  in
  case "(?i)a" "group options are not supported";
  case "(?i:a)" "group options are not supported";
  case "(?m)." "group options are not supported";
  case "a(?=b)" "lookaround is not supported";
  case "(?<=a)b" "lookaround is not supported";
  case "(?>a)" "atomic groups are not supported";
  case "(a)\\1" "backreferences are not supported";
  case "\\ba" "word boundaries are not supported";
  case "a++" "possessive quantifiers are not supported";
  case "[[:alpha:]]" "POSIX bracket expressions are not supported";
  case "[a[b]]" "nested character classes are not supported";
  case "[a-z&&[^aeiou]]" "character class intersection is not supported";
  case "\\p{Han}" "unsupported property \\p{Han}";
  case "\\p{Foo}" "unsupported property \\p{Foo}";
  case "\\h" "unsupported escape \\h";
  case "\\R" "unsupported escape \\R";
  case "\\xc3\\xa9" "byte escapes above \\x7F are not supported, use \\x{..}";
  case "\\x{110000}" "invalid code point escape";
  case "\\x{D800}" "invalid code point escape";
  case "\\xg" "invalid hexadecimal escape";
  case "^" "a ^ that can end a match is not supported";
  case "^\\s*" "a ^ that can end a match is not supported";
  case "\\n^" "a ^ that can end a match is not supported";
  case "(^|x)a*" "a ^ that can end a match is not supported";
  case "*a" "nothing to repeat";
  case "{2}" "nothing to repeat";
  case "a|*" "nothing to repeat";
  case "(" "unmatched (";
  case ")" "unmatched )";
  case "a\\" "pattern ends with a backslash";
  case "[a" "unterminated character class";
  case "[z-a]" "empty range in character class";
  case "[\\d-z]" "invalid range in character class";
  case "[a-\\d]" "invalid range in character class";
  case "a{3,2}" "invalid interval {3,2}";
  case "a{100001}" "repeat count above 100000";
  case "\\p{L" "unterminated property escape";
  case "\\pL" "invalid property escape";
  case "(?<1a>x)" "invalid group name";
  case "(?#a" "unterminated comment";
  case "\xC3" "pattern is not valid UTF-8"

(* Expectations from HuggingFace [normalizers.Nmt()]. *)
let test_nmt () =
  let case text expected =
    equal
      ~msg:(Printf.sprintf "nmt %S" text)
      string expected
      (Normalizer.apply Normalizer.nmt text)
  in
  case
    "a\x01b\x08c\x0E\x1F\x7F\xC2\x80\xC2\x8F\xC2\x9F\xEF\xBB\xBF\xE2\x80\x8B\xC2\xADx\x00\x09\x0A\x0D\x0B\x0C\xC2\x85 "
    "abc\xC2\x80  \xC2\xADx\x00    \xC2\x85 ";
  case
    "a\xE1\x9A\x80b\xE2\x80\x8Fc\xE2\x80\xA8d\xE2\x80\xA9e\xE2\x96\x81f\xEF\xBF\xBDg\xE2\x80\x8A"
    "a b c d e f g\xE2\x80\x8A";
  case "" "";
  (* Bytes that are not UTF-8 pass through. *)
  case "a\xFFb" "a\xFFb"

(* Expectations from HuggingFace [normalizers.ByteLevel()], which takes no
   options: no prefix space is added. *)
let test_byte_level () =
  let case text expected =
    equal
      ~msg:(Printf.sprintf "byte level %S" text)
      string expected
      (Normalizer.apply Normalizer.byte_level text)
  in
  case " x" "\xC4\xA0x";
  case "a\xC3\xA9" "a\xC3\x83\xC2\xA9";
  case "ab" "ab";
  case "" ""

(* The CLIP normalizer, as its tokenizer file writes it. Expectations from
   HuggingFace [normalizers.Sequence([NFC(), Replace(Regex(r"\s+"), " "),
   Lowercase()])]. *)
let test_clip_normalizer () =
  let n =
    Normalizer.sequence
      [
        Normalizer.nfc;
        Normalizer.replace_regex ~pattern:"\\s+" ~replacement:" ";
        Normalizer.lowercase;
      ]
  in
  let case text expected =
    equal
      ~msg:(Printf.sprintf "clip %S" text)
      string expected (Normalizer.apply n text)
  in
  case "A\xCC\x81  Photo\x09\x0Aof\xC2\xA0a\xE3\x80\x80CAT "
    "\xC3\xA1 photo of a cat ";
  case "x" "x"

(* Case mapping is not case folding: the ligatures and the sharp s lower to
   themselves. Expectations from HuggingFace [normalizers.Lowercase()]. *)
let test_lowercase_is_not_folding () =
  let case text expected =
    equal
      ~msg:(Printf.sprintf "lowercase %S" text)
      string expected
      (Normalizer.apply Normalizer.lowercase text)
  in
  case "\xC3\x9F" "\xC3\x9F";
  case "\xE1\xBA\x9E" "\xC3\x9F";
  case "\xEF\xAC\x81" "\xEF\xAC\x81";
  case "\xEF\xAC\x84" "\xEF\xAC\x84";
  case "\xC4\xB0" "i\xCC\x87";
  case "\xCE\xA3" "\xCF\x83";
  case "\xC7\x85" "\xC7\x86";
  case "\xC3\x80\xC3\x89\xC3\x8E" "\xC3\xA0\xC3\xA9\xC3\xAE"

(* Composition, at its edges: Hangul is composed by arithmetic, an LVT syllable
   takes no further trailing jamo and U+11C3 is not one; a mark blocked by one
   of the same combining class stays; a truncated sequence is the replacement
   character. Expectations from HuggingFace [normalizers.NFC()]. *)
let test_nfc_edges () =
  let case text expected =
    equal
      ~msg:(Printf.sprintf "nfc %S" text)
      string expected
      (Normalizer.apply Normalizer.nfc text);
    equal
      ~msg:(Printf.sprintf "nfc %S aligned" text)
      string expected
      (fst (Normalizer.apply_aligned Normalizer.nfc text))
  in
  case "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8" "\xEA\xB0\x81";
  case "\xEA\xB0\x80\xE1\x86\xA8" "\xEA\xB0\x81";
  case "\xEA\xB0\x81\xE1\x86\xA8" "\xEA\xB0\x81\xE1\x86\xA8";
  case "\xEA\xB0\x80\xE1\x87\x83" "\xEA\xB0\x80\xE1\x87\x83";
  case "\xE1\x84\x80\xE1\x85\xA1\xE1\x87\x83" "\xEA\xB0\x80\xE1\x87\x83";
  case "a\xCC\x96\xCC\xA3" "a\xCC\x96\xCC\xA3";
  case "abc\xC3" "abc\xEF\xBF\xBD"

(* Expectations from HuggingFace [normalizers.StripAccents()], which removes
   every mark and does not decompose. *)
let test_strip_accents_keeps_composition () =
  let case text expected =
    equal
      ~msg:(Printf.sprintf "strip_accents %S" text)
      string expected
      (Normalizer.apply Normalizer.strip_accents text)
  in
  (* Bengali vowel sign O and Devanagari vowel sign I are spacing marks. *)
  case "\xE0\xA7\x8B" "";
  case "\xE0\xA4\xBF" "";
  (* Enclosing and nonspacing marks go too. *)
  case "a\xE0\xA4\x83\xE2\x83\x9D\xCC\x81" "a";
  (* Precomposed characters are left alone without a preceding NFD. *)
  case "\xC3\xA1" "\xC3\xA1";
  case "caf\xC3\xA9" "caf\xC3\xA9";
  case "\xE0\xA4\x95\xE0\xA4\xBC" "\xE0\xA4\x95";
  case "A\xCC\x81" "A"

(* Expectations from HuggingFace [normalizers.BertNormalizer(clean_text=True,
   handle_chinese_chars=True, strip_accents=None, lowercase=True)], the
   bert-base-uncased settings. *)
let test_bert_normalizer () =
  let n = Normalizer.bert () in
  let case text expected =
    equal
      ~msg:(Printf.sprintf "bert normalize %S" text)
      string expected (Normalizer.apply n text)
  in
  (* Only nonspacing marks are stripped, so the vowel signs of an abugida
     survive: ["नमस्ते हिन्दी"] keeps its ि and ी. *)
  case
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87 \
     \xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA5\x8D\xE0\xA4\xA6\xE0\xA5\x80"
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA4\xA4 \
     \xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA4\xA6\xE0\xA5\x80";
  case "a\xE0\xA4\x83\xE2\x83\x9D\xCC\x81" "a\xE0\xA4\x83\xE2\x83\x9D";
  (* Accents proper are stripped, after decomposition. *)
  case "caf\xC3\xA9" "cafe";
  case "A\xCC\x81BC" "abc";
  case "\xC4\xB0" "i";
  case "\xE1\xBE\xBC" "\xCE\xB1";
  (* Lowercasing, not folding. *)
  case "\xC3\x9F" "\xC3\x9F";
  case "\xE1\xBA\x9E" "\xC3\x9F";
  case "\xEF\xAC\x81" "\xEF\xAC\x81";
  case "\xCE\xA3" "\xCF\x83";
  (* Control, format and private use characters are removed, and so are NUL and
     the replacement character. *)
  case "\x00\x07\x7F" "";
  case "a\xC2\xADb" "ab";
  case "a\xE2\x80\x8Bb" "ab";
  case "a\xEE\x80\x80b" "ab";
  case "a\xEF\xBF\xBDb" "ab";
  (* Unassigned codepoints are not controls: they survive to reach the model. *)
  case "\xF4\x8F\xBF\xBF" "\xF4\x8F\xBF\xBF";
  case "\xCD\xB8" "\xCD\xB8";
  (* Whitespace of any kind becomes a plain space. *)
  case "a\xE2\x80\xA8b" "a b";
  case "a\xE3\x80\x80b" "a b";
  (* CJK ideographs are padded with spaces, one at a time. *)
  case "a\xE6\xBC\xA2b" "a \xE6\xBC\xA2 b";
  case "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E"
    " \xE6\x97\xA5  \xE6\x9C\xAC  \xE8\xAA\x9E "

(* Text is scanned eight bytes at a time for what a stage has to do; a byte of
   interest must be found wherever it falls with respect to those words. *)
let test_word_scans () =
  for k = 0 to 24 do
    let before = String.make k 'a' and after = String.make (24 - k) 'b' in
    equal
      ~msg:(Printf.sprintf "nfc at %d" k)
      string
      (before ^ "\xC3\xA9" ^ after)
      (Normalizer.apply Normalizer.nfc (before ^ "e\xCC\x81" ^ after));
    equal
      ~msg:(Printf.sprintf "lowercase at %d" k)
      string
      (before ^ "x" ^ after)
      (Normalizer.apply Normalizer.lowercase (before ^ "X" ^ after));
    let text = before ^ " " ^ after in
    let normalized, alignment =
      Normalizer.apply_aligned
        (Normalizer.replace ~pattern:" " ~replacement:"\xE2\x96\x81")
        text
    in
    equal
      ~msg:(Printf.sprintf "replace at %d" k)
      string
      (before ^ "\xE2\x96\x81" ^ after)
      normalized;
    equal
      ~msg:(Printf.sprintf "replace at %d: span" k)
      (pair int int)
      (k, k + 1)
      (Normalizer.original_span alignment ~start:k ~stop:(k + 3));
    equal
      ~msg:(Printf.sprintf "strip at %d" k)
      string (String.make k 'a')
      (Normalizer.apply (Normalizer.strip ())
         (String.make k 'a' ^ String.make (24 - k) ' '))
  done;
  let spaces = "aaaaaaa aaaaaaa a a" in
  equal ~msg:"replace many" string
    "aaaaaaa\xE2\x96\x81aaaaaaa\xE2\x96\x81a\xE2\x96\x81a"
    (Normalizer.apply
       (Normalizer.replace ~pattern:" " ~replacement:"\xE2\x96\x81")
       spaces);
  equal ~msg:"replace none" string spaces
    (Normalizer.apply (Normalizer.replace ~pattern:"z" ~replacement:"y") spaces);
  equal ~msg:"replace all" string (String.make 20 '.')
    (Normalizer.apply
       (Normalizer.replace ~pattern:"a" ~replacement:".")
       (String.make 20 'a'))

(* Alignment *)

(* The span every character of the normalized text reports on the input, in the
   order the characters come out. Expectations are what HuggingFace reports for
   one-character tokens; regenerate them with [uv run --with tokenizers python3
   test/scripts/hf_alignments.py]. *)
let char_spans normalizer input =
  let normalized, alignment = Normalizer.apply_aligned normalizer input in
  let buffer = Buffer.create 64 in
  let i = ref 0 in
  while !i < String.length normalized do
    let n = Uchar.utf_decode_length (String.get_utf_8_uchar normalized !i) in
    let start, stop =
      Normalizer.original_span alignment ~start:!i ~stop:(!i + n)
    in
    if !i > 0 then Buffer.add_char buffer ' ';
    Buffer.add_string buffer (Printf.sprintf "%d,%d" start stop);
    i := !i + n
  done;
  (normalized, Buffer.contents buffer)

let aligned normalizer label input expected_text expected_spans =
  let text, spans = char_spans normalizer input in
  equal
    ~msg:(Printf.sprintf "%s %S: text" label input)
    string expected_text text;
  equal
    ~msg:(Printf.sprintf "%s %S: spans" label input)
    string expected_spans spans

(* A character standing for several input characters takes the range of the
   first of them, and one standing for none takes the range of the character
   before it, so a decomposition scattered by canonical ordering reports the
   character that ended up in its place rather than the one it came from. *)
let test_align_nfd () =
  let case = aligned Normalizer.nfd "nfd" in
  case "caf\xC3\xA9" "cafe\xCC\x81" "0,1 1,2 2,3 3,5 3,5";
  case "\xE1\xBA\x9B\xCC\xA3xy" "\xC5\xBF\xCC\xA3\xCC\x87xy"
    "0,3 3,5 3,5 5,6 6,7";
  case "\xE1\xB8\x8D\xCC\x81" "d\xCC\xA3\xCC\x81" "0,3 0,3 3,5";
  case "\xED\x95\x9C\xEA\xB8\x80"
    "\xE1\x84\x92\xE1\x85\xA1\xE1\x86\xAB\xE1\x84\x80\xE1\x85\xB3\xE1\x86\xAF"
    "0,3 0,3 0,3 3,6 3,6 3,6";
  case "a\xCC\x81\xCC\x96" "a\xCC\x96\xCC\x81" "0,1 1,3 3,5"

let test_align_nfc () =
  let case = aligned Normalizer.nfc "nfc" in
  case "cafe\xCC\x81" "caf\xC3\xA9" "0,1 1,2 2,3 3,4";
  case "e\xCC\x81\xCC\xA3" "\xE1\xBA\xB9\xCC\x81" "0,1 3,5";
  case "a\xCC\x81\xCC\x96" "\xC3\xA1\xCC\x96" "0,1 3,5";
  case "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8Z" "\xEA\xB0\x81Z" "0,3 9,10";
  case "e\xCC\xA3\xCC\x81\xCC\x96" "\xE1\xBA\xB9\xCC\x96\xCC\x81" "0,1 3,5 5,7";
  case "\xE1\xBA\x9B\xCC\xA3xy" "\xE1\xBA\x9B\xCC\xA3xy" "0,3 3,5 5,6 6,7";
  case "\xEA\xB0\x80\xE1\x87\x83" "\xEA\xB0\x80\xE1\x87\x83" "0,3 3,6"

let test_align_nfkc_nfkd () =
  let case = aligned Normalizer.nfkc "nfkc" in
  case "\xEF\xAC\x81x \xE2\x91\xA0 \xEF\xBC\xA1" "fix 1 A"
    "0,3 0,3 3,4 4,5 5,8 8,9 9,12";
  case "\xC2\xBD\xE2\x81\xB5" "1\xE2\x81\x8425" "0,2 0,2 0,2 2,5";
  let case = aligned Normalizer.nfkd "nfkd" in
  case "\xEF\xB7\xBA!"
    "\xD8\xB5\xD9\x84\xD9\x89 \xD8\xA7\xD9\x84\xD9\x84\xD9\x87 \
     \xD8\xB9\xD9\x84\xD9\x8A\xD9\x87 \xD9\x88\xD8\xB3\xD9\x84\xD9\x85!"
    "0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 0,3 \
     3,4";
  case "\xEF\xAC\x81x \xE2\x91\xA0 \xC7\x84" "fix 1 DZ\xCC\x8C"
    "0,3 0,3 3,4 4,5 5,8 8,9 9,11 9,11 9,11"

(* A character the normalizer dropped belongs to no span, so the spans of a
   stripped text skip over what it removed. *)
let test_align_text_transforms () =
  aligned Normalizer.lowercase "lowercase" "A\xC4\xB0B\xC3\x9F"
    "ai\xCC\x87b\xC3\x9F" "0,1 1,3 1,3 3,4 4,6";
  aligned Normalizer.strip_accents "strip accents"
    "\xC3\xA1a\xCC\x81b\xCC\x81\xCC\x81c" "\xC3\xA1abc" "0,2 2,3 5,6 10,11";
  aligned (Normalizer.strip ()) "strip" "  a b  " "a b" "2,3 3,4 4,5";
  aligned
    (Normalizer.strip ~right:false ())
    "strip left" "\t\n a b \n" "a b \n" "3,4 4,5 5,6 6,7 7,8";
  aligned
    (Normalizer.strip ~left:false ())
    "strip right" " a b \n" " a b" "0,1 1,2 2,3 3,4"

(* A replacement stands for the last character it replaced, whatever its length,
   and one that replaced nothing for the character before it: an empty span at
   the very start. *)
let test_align_replace () =
  aligned
    (Normalizer.replace ~pattern:" " ~replacement:"\xE2\x96\x81")
    "replace string" "a  b" "a\xE2\x96\x81\xE2\x96\x81b" "0,1 1,2 2,3 3,4";
  aligned
    (Normalizer.replace_regex ~pattern:"\\s+" ~replacement:" ")
    "replace collapse" "a \t\n b" "a b" "0,1 4,5 5,6";
  aligned
    (Normalizer.replace ~pattern:"a" ~replacement:"xyz")
    "replace grow" "za z" "zxyz z" "0,1 1,2 1,2 1,2 2,3 3,4";
  aligned
    (Normalizer.replace_regex ~pattern:"ab+" ~replacement:"X")
    "replace shrink" "zabbbz" "zXz" "0,1 4,5 5,6";
  aligned
    (Normalizer.replace_regex ~pattern:"\\s+" ~replacement:"")
    "replace delete" "a \t b" "ab" "0,1 4,5";
  aligned
    (Normalizer.replace_regex ~pattern:"x*" ~replacement:"_")
    "replace empty" "a\xC3\xA9 b" "_a_\xC3\xA9_ _b_"
    "0,0 0,1 0,1 1,3 1,3 3,4 3,4 4,5 4,5";
  aligned
    (Normalizer.replace_regex ~pattern:"x*" ~replacement:"_")
    "replace empty" "xa" "_a_" "0,1 1,2 1,2";
  aligned
    (Normalizer.replace ~pattern:"" ~replacement:"_")
    "replace empty string" "a\xC3\xA9" "_a_\xC3\xA9_" "0,0 0,1 0,1 1,3 1,3";
  aligned
    (Normalizer.replace_regex ~pattern:"\\p{L}+" ~replacement:"L")
    "replace class" "ab1\xC3\xA9\xE6\x97\xA5" "L1L" "1,2 2,3 5,8";
  aligned
    (Normalizer.replace_regex ~pattern:"$" ~replacement:"!")
    "replace anchor" "a\nb" "a!\nb!" "0,1 0,1 1,2 2,3 2,3"

(* An inserted character stands for the one it was put next to: the prefix and
   the first character of the text share a span. *)
let test_align_prepend () =
  aligned
    (Normalizer.prepend "\xE2\x96\x81")
    "prepend" "Hello" "\xE2\x96\x81Hello" "0,1 0,1 1,2 2,3 3,4 4,5";
  aligned
    (Normalizer.prepend "\xE2\x96\x81")
    "prepend" " x" "\xE2\x96\x81 x" "0,1 0,1 1,2";
  aligned (Normalizer.prepend "<<") "prepend multi" "Hello" "<<Hello"
    "0,1 0,1 0,1 1,2 2,3 3,4 4,5"

let test_align_bert () =
  let case = aligned (Normalizer.bert ()) "bert" in
  case "\xC3\x87a \xE6\xBC\xA2 a\xCC\x81x" "ca  \xE6\xBC\xA2  ax"
    "0,2 2,3 3,4 4,7 4,7 4,7 7,8 8,9 11,12";
  case "a\xE6\xBC\xA2b" "a \xE6\xBC\xA2 b" "0,1 1,4 1,4 1,4 4,5";
  case "a\xC2\xADb\tc" "ab c" "0,1 3,4 4,5 5,6";
  (* The stripped virama leaves a hole, so the spans are not contiguous. *)
  case
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87"
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA4\xA4" "0,3 3,6 6,9 12,15";
  case "Caf\xC3\xA9" "cafe" "0,1 1,2 2,3 3,5";
  case "\xE1\xBA\x9B\xCC\xA3 x" "\xC5\xBF x" "0,3 5,6 6,7";
  case "\xEA\xB0\x81Z" "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8z" "0,3 0,3 0,3 3,4";
  case "\xC2\xA0\xC4\xB0\xE2\x80\xA8" " i " "0,2 2,4 4,7";
  (* A dropped control does not separate the marks around it, so a spacing mark
     of a lower combining class still moves before the accent and stands for
     it. *)
  case "a\xCC\x81\x01\xF0\x9D\x85\xA5" "a\xF0\x9D\x85\xA5" "0,1 1,3";
  case "a\xF0\x9D\x85\xAD\x7F\xCC\x81\xF0\x9D\x85\xA5"
    "a\xF0\x9D\x85\xA5\xF0\x9D\x85\xAD" "0,1 1,5 6,8";
  case "a\xCC\x81\x00\xF0\x9D\x85\xA5b" "a\xF0\x9D\x85\xA5b" "0,1 1,3 8,9";
  aligned
    (Normalizer.bert ~lowercase:false ())
    "bert nolower" "\xC3\x87a \xE6\xBC\xA2 \xC3\xA1x"
    "\xC3\x87a  \xE6\xBC\xA2  \xC3\xA1x"
    "0,2 2,3 3,4 4,7 4,7 4,7 7,8 8,10 10,11";
  aligned
    (Normalizer.bert ~clean_text:false ())
    "bert noclean" "a\x01\xC3\x89\tb" "a\x01e\tb" "0,1 1,2 2,4 4,5 5,6";
  aligned
    (Normalizer.bert ~strip_accents:(Some false) ())
    "bert nostrip" "\xC3\x89a" "\xC3\xA9a" "0,2 2,3"

(* Prefixing and slicing compose without copying the alignment: the prefix
   stands for the first character of what it was put before, whatever came
   before that. *)
let test_align_views () =
  let strip = Normalizer.strip ()
  and prepend = Normalizer.prepend "\xE2\x96\x81" in
  let replace = Normalizer.replace ~pattern:" " ~replacement:"\xE2\x96\x81" in
  aligned
    (Normalizer.sequence [ strip; prepend ])
    "strip prepend" "  ab " "\xE2\x96\x81ab" "2,3 2,3 3,4";
  aligned
    (Normalizer.sequence [ strip; prepend ])
    "strip prepend" " x" "\xE2\x96\x81x" "1,2 1,2";
  aligned
    (Normalizer.sequence [ Normalizer.prepend "<<"; strip ])
    "prepend strip" " a " "<< a" "0,1 0,1 0,1 1,2";
  aligned
    (Normalizer.sequence [ Normalizer.prepend "<<"; strip ])
    "prepend strip" "a " "<<a" "0,1 0,1 0,1";
  aligned
    (Normalizer.sequence
       [ Normalizer.prepend "  x"; Normalizer.strip ~right:false () ])
    "prepend strip left" " a " "x a " "0,1 0,1 1,2 2,3";
  aligned
    (Normalizer.sequence [ prepend; Normalizer.prepend "x" ])
    "prepend prepend" "ab" "x\xE2\x96\x81ab" "0,1 0,1 0,1 1,2";
  aligned
    (Normalizer.sequence
       [ Normalizer.strip ~right:false (); Normalizer.strip ~left:false () ])
    "strip strip" "  a b  " "a b" "2,3 3,4 4,5";
  aligned
    (Normalizer.sequence [ prepend; strip; replace ])
    "prepend strip replace" "  a b "
    "\xE2\x96\x81\xE2\x96\x81\xE2\x96\x81a\xE2\x96\x81b"
    "0,1 0,1 1,2 2,3 3,4 4,5";
  aligned
    (Normalizer.sequence [ strip; prepend; replace ])
    "strip prepend replace" "  a b " "\xE2\x96\x81a\xE2\x96\x81b"
    "2,3 2,3 3,4 4,5";
  aligned
    (Normalizer.sequence [ strip; Normalizer.lowercase ])
    "strip lowercase" " A\xC3\x89 " "a\xC3\xA9" "1,2 2,4";
  aligned
    (Normalizer.sequence [ prepend; Normalizer.nfd ])
    "prepend nfd" "\xC3\xA9" "\xE2\x96\x81e\xCC\x81" "0,2 0,2 0,2"

(* The BERT normalizer runs its four passes as one; on text where the extra
   passes have nothing to do, that one pass must agree with the single stages,
   text and alignment alike. *)
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
    "Hello" ^ String.make 1 '\xFF' ^ String.make 1 '\xFE' ^ "World";
    "\xE6\xBC\xA2\x80\xE6\xBC \xE6\xBC\xA2";
    String.concat ""
      (List.init 40 (fun i -> if i mod 7 = 3 then "\xC3\x89" else "aB"));
  ]

let same_alignment label a b =
  List.iteri
    (fun i text ->
      let ta, sa = char_spans a text and tb, sb = char_spans b text in
      equal ~msg:(Printf.sprintf "%s on %d: text" label i) string tb ta;
      equal ~msg:(Printf.sprintf "%s on %d: spans" label i) string sb sa)
    bert_corpus

let test_bert_single_stages () =
  same_alignment "bert lowercase only"
    (Normalizer.bert ~clean_text:false ~handle_chinese_chars:false
       ~strip_accents:(Some false) ())
    Normalizer.lowercase;
  (* The corpus has no spacing or enclosing mark, so dropping the nonspacing
     marks is dropping the marks. *)
  same_alignment "bert strip only"
    (Normalizer.bert ~clean_text:false ~handle_chinese_chars:false
       ~lowercase:false ~strip_accents:(Some true) ())
    (Normalizer.sequence [ Normalizer.nfd; Normalizer.strip_accents ]);
  same_alignment "bert strip and lowercase"
    (Normalizer.bert ~clean_text:false ~handle_chinese_chars:false ())
    (Normalizer.sequence
       [ Normalizer.nfd; Normalizer.strip_accents; Normalizer.lowercase ]);
  List.iter
    (fun text ->
      equal ~msg:"bert with nothing to do" string text
        (Normalizer.apply
           (Normalizer.bert ~clean_text:false ~handle_chinese_chars:false
              ~lowercase:false ~strip_accents:(Some false) ())
           text))
    bert_corpus

(* On ASCII the BERT normalizer is a byte map: controls go, white space becomes
   a space, letters lower, and every byte kept stands for itself. *)
let test_bert_ascii () =
  let reference text =
    let b = Buffer.create (String.length text) in
    String.iter
      (fun c ->
        let code = Char.code c in
        if code = 9 || code = 10 || code = 13 || code = 32 then
          Buffer.add_char b ' '
        else if code >= 33 && code < 127 then
          Buffer.add_char b (Char.lowercase_ascii c))
      text;
    Buffer.contents b
  in
  let n = Normalizer.bert () in
  let texts =
    List.init 200 (fun i ->
        String.init i (fun j -> Char.chr (((j * 37) + (i * 11)) mod 128)))
    @ [ "\x00\x07\x7F"; "  Hello\tWorld\r\n"; String.make 100 'A' ]
  in
  List.iter
    (fun text ->
      let expected = reference text in
      equal
        ~msg:(Printf.sprintf "bert ascii %S" text)
        string expected (Normalizer.apply n text);
      let normalized, alignment = Normalizer.apply_aligned n text in
      equal ~msg:"bert ascii aligned text" string expected normalized;
      let src = ref 0 in
      for i = 0 to String.length normalized - 1 do
        while reference (String.make 1 text.[!src]) = "" do
          incr src
        done;
        equal
          ~msg:(Printf.sprintf "bert ascii %S byte %d" text i)
          (pair int int)
          (!src, !src + 1)
          (Normalizer.original_span alignment ~start:i ~stop:(i + 1));
        incr src
      done)
    texts

let test_align_sequence () =
  let llama =
    Normalizer.sequence
      [
        Normalizer.prepend "\xE2\x96\x81";
        Normalizer.replace ~pattern:" " ~replacement:"\xE2\x96\x81";
      ]
  in
  aligned llama "llama" "\n\nNot" "\xE2\x96\x81\n\nNot"
    "0,1 0,1 1,2 2,3 3,4 4,5";
  aligned llama "llama" "a  b" "\xE2\x96\x81a\xE2\x96\x81\xE2\x96\x81b"
    "0,1 0,1 1,2 2,3 3,4";
  aligned
    (Normalizer.sequence
       [ Normalizer.nfd; Normalizer.strip_accents; Normalizer.lowercase ])
    "nfd strip lower" "Caf\xC3\xA9" "cafe" "0,1 1,2 2,3 3,5"

(* Each byte becomes one character, so a character split across bytes reports
   itself whole. *)
let test_align_byte_level () =
  aligned Normalizer.byte_level "byte level" "a\xC3\xA9" "a\xC3\x83\xC2\xA9"
    "0,1 1,3 1,3";
  aligned Normalizer.byte_level "byte level" " x" "\xC4\xA0x" "0,1 1,2"

(* A removed character leaves a hole, a replaced one stands for itself. *)
let test_align_nmt () =
  aligned Normalizer.nmt "nmt" "a\x01b\tc\xE2\x80\x8Bd" "ab c d"
    "0,1 2,3 3,4 4,5 5,8 8,9"

let test_align_clip () =
  aligned
    (Normalizer.sequence
       [
         Normalizer.nfc;
         Normalizer.replace_regex ~pattern:"\\s+" ~replacement:" ";
         Normalizer.lowercase;
       ])
    "clip" "\xC3\x81  B\tc" "\xC3\xA1 b c" "0,2 3,4 4,5 5,6 6,7"

(* Normalizing the text away leaves one empty span, and it stands for the whole
   of what was normalized rather than for the end of it. *)
let test_align_emptied () =
  let case label normalizer input expected =
    let text, alignment = Normalizer.apply_aligned normalizer input in
    equal ~msg:(Printf.sprintf "%s %S: text" label input) string "" text;
    let start, stop = Normalizer.original_span alignment ~start:0 ~stop:0 in
    equal
      ~msg:(Printf.sprintf "%s %S: span" label input)
      string expected
      (Printf.sprintf "%d,%d" start stop)
  in
  case "strip" (Normalizer.strip ()) "   " "0,3";
  case "bert" (Normalizer.bert ()) "\x00" "0,1";
  case "strip" (Normalizer.strip ()) "" "0,0";
  equal ~msg:"identity of empty text" string "0,0"
    (let start, stop =
       Normalizer.original_span (Normalizer.identity "") ~start:0 ~stop:0
     in
     Printf.sprintf "%d,%d" start stop)

let test_identity_alignment () =
  let a = Normalizer.identity "caf\xC3\xA9" in
  let span ~start ~stop =
    let s, e = Normalizer.original_span a ~start ~stop in
    Printf.sprintf "%d,%d" s e
  in
  equal ~msg:"whole" string "0,5" (span ~start:0 ~stop:5);
  equal ~msg:"ascii" string "0,1" (span ~start:0 ~stop:1);
  (* A span cutting a character short still reports it whole. *)
  equal ~msg:"partial character" string "3,5" (span ~start:3 ~stop:4);
  equal ~msg:"empty" string "2,2" (span ~start:2 ~stop:2);
  equal ~msg:"empty at end" string "5,5" (span ~start:5 ~stop:5);
  raises (Invalid_argument "6,7 is not a span of the 5 normalized bytes")
    (fun () -> Normalizer.original_span a ~start:6 ~stop:7);
  raises (Invalid_argument "0,6 is not a span of the 5 normalized bytes")
    (fun () -> Normalizer.original_span a ~start:0 ~stop:6)

let alignment_corpus =
  [
    "";
    "hello";
    " \t\n ";
    "caf\xC3\xA9 na\xC3\xAFve";
    "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E \xE6\xBC\xA2\xE5\xAD\x97";
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87";
    "\xD7\x91\xD6\xBC\xD6\xB8";
    "\xE1\xBA\x9B\xCC\xA3\xE1\xB8\x8D\xCC\x81a\xCC\x81\xCC\x96";
    "\xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA6";
    "Hello" ^ String.make 1 '\xFF' ^ String.make 1 '\xFE' ^ "World";
    "\x00\x07\x7F a\xC2\xADb";
    Fixture.read "fixtures/parity/edge_cases.txt";
    Fixture.read "fixtures/parity/sample.txt";
  ]

let alignment_normalizers =
  [
    ("nfc", Normalizer.nfc);
    ("nfd", Normalizer.nfd);
    ("nfkc", Normalizer.nfkc);
    ("nfkd", Normalizer.nfkd);
    ("lowercase", Normalizer.lowercase);
    ("strip_accents", Normalizer.strip_accents);
    ("strip", Normalizer.strip ());
    ("replace", Normalizer.replace ~pattern:" " ~replacement:"\xE2\x96\x81");
    ("replace regex", Normalizer.replace_regex ~pattern:"\\s+" ~replacement:" ");
    ( "replace empty",
      Normalizer.replace_regex ~pattern:"\\p{L}*" ~replacement:"_" );
    ("prepend", Normalizer.prepend "\xE2\x96\x81");
    ("byte_level", Normalizer.byte_level);
    ("nmt", Normalizer.nmt);
    ("bert", Normalizer.bert ());
    ("bert nolower", Normalizer.bert ~lowercase:false ());
    ("bert noclean", Normalizer.bert ~clean_text:false ());
    ("bert nostrip", Normalizer.bert ~strip_accents:(Some false) ());
    ( "bert nocjk noclean",
      Normalizer.bert ~clean_text:false ~handle_chinese_chars:false () );
    ( "strip prepend replace",
      Normalizer.sequence
        [
          Normalizer.strip ();
          Normalizer.prepend "\xE2\x96\x81";
          Normalizer.replace ~pattern:" " ~replacement:"\xE2\x96\x81";
        ] );
    ( "prepend strip",
      Normalizer.sequence [ Normalizer.prepend "  x"; Normalizer.strip () ] );
    ( "nfkc strip lower",
      Normalizer.sequence
        [ Normalizer.nfkc; Normalizer.strip_accents; Normalizer.lowercase ] );
    ( "llama",
      Normalizer.sequence
        [
          Normalizer.prepend "\xE2\x96\x81";
          Normalizer.replace ~pattern:" " ~replacement:"\xE2\x96\x81";
        ] );
  ]

(* Reporting where a byte came from goes through a different implementation of
   Unicode normalization than [apply] does, one that can account for every
   character; the two must agree on the text down to the byte. *)
let test_aligned_matches_apply () =
  List.iter
    (fun (name, normalizer) ->
      List.iteri
        (fun i text ->
          equal
            ~msg:(Printf.sprintf "%s on corpus %d" name i)
            string
            (Normalizer.apply normalizer text)
            (fst (Normalizer.apply_aligned normalizer text)))
        alignment_corpus)
    alignment_normalizers

(* Spans stay inside the input and never go backwards, so a token's span is well
   formed wherever it falls. *)
let test_alignment_is_monotonic () =
  List.iter
    (fun (name, normalizer) ->
      List.iteri
        (fun i text ->
          let normalized, alignment =
            Normalizer.apply_aligned normalizer text
          in
          let previous = ref 0 in
          for byte = 0 to String.length normalized - 1 do
            let start, stop =
              Normalizer.original_span alignment ~start:byte ~stop:(byte + 1)
            in
            if start > stop || stop > String.length text || start < !previous
            then
              failf "%s on corpus %d: byte %d reports %d,%d after %d" name i
                byte start stop !previous;
            previous := start
          done)
        alignment_corpus)
    alignment_normalizers

(* Integration with Tokenizer *)

let test_tokenize_with_normalization () =
  let text = "HELLO   WORLD!" in
  let normalizer =
    Normalizer.sequence
      [
        Normalizer.lowercase;
        Normalizer.replace_regex ~pattern:"\\s+" ~replacement:" ";
      ]
  in
  let tokenizer =
    word_level ~normalizer ~pre:Pre_tokenizer.whitespace
      ~vocab:[ ("hello", 0); ("world", 1); ("!", 2) ]
      ()
  in
  let tokens = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"normalized tokenization" (list string) [ "hello"; "world"; "!" ]
    tokens

let test_tokenize_unicode_words () =
  let text = "café résumé naïve" in
  let tokenizer =
    word_level ~pre:Pre_tokenizer.whitespace
      ~vocab:[ ("café", 0); ("résumé", 1); ("naïve", 2) ]
      ()
  in
  let tokens = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"tokenized unicode" bool true (List.length tokens > 0)

let test_malformed_unicode () =
  let text = "Hello" ^ String.make 1 '\xFF' ^ String.make 1 '\xFE' ^ "World" in
  let tokenizer = chars () in
  let tokens = encode tokenizer text |> Encoding.tokens |> Array.to_list in
  equal ~msg:"handled malformed" bool true (List.length tokens > 0)

(* Test Suite *)

let unicode_tests =
  [
    (* Normalization *)
    test "lowercase normalization" test_lowercase_normalization;
    test "lowercase is not case folding" test_lowercase_is_not_folding;
    test "strip accents normalization" test_strip_accents_normalization;
    test "strip accents keeps composition" test_strip_accents_keeps_composition;
    test "bert normalizer" test_bert_normalizer;
    test "word scans" test_word_scans;
    test "nfc edges" test_nfc_edges;
    test "normalization sequence" test_normalization_sequence;
    test "replace" test_replace;
    test "replace regex" test_replace_regex;
    test "replace regex rejected" test_replace_regex_rejected;
    test "nmt" test_nmt;
    test "byte level" test_byte_level;
    test "clip normalizer" test_clip_normalizer;
    (* Alignment *)
    test "nfd alignment" test_align_nfd;
    test "nfc alignment" test_align_nfc;
    test "nfkc and nfkd alignment" test_align_nfkc_nfkd;
    test "text transform alignment" test_align_text_transforms;
    test "replace alignment" test_align_replace;
    test "prepend alignment" test_align_prepend;
    test "bert alignment" test_align_bert;
    test "view alignment" test_align_views;
    test "bert single stages" test_bert_single_stages;
    test "bert ascii" test_bert_ascii;
    test "sequence alignment" test_align_sequence;
    test "byte level alignment" test_align_byte_level;
    test "nmt alignment" test_align_nmt;
    test "clip alignment" test_align_clip;
    test "emptied text alignment" test_align_emptied;
    test "identity alignment" test_identity_alignment;
    test "aligned matches apply" test_aligned_matches_apply;
    test "alignment is monotonic" test_alignment_is_monotonic;
    (* Integration *)
    test "tokenize with normalization" test_tokenize_with_normalization;
    test "tokenize unicode words" test_tokenize_unicode_words;
    (* Error handling *)
    test "malformed unicode" test_malformed_unicode;
  ]

let () = run "brot unicode" [ group "unicode" unicode_tests ]
