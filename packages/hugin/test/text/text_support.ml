(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Texts as expressions, to generate and print them, and faces built for the
   layout tests. *)

open Windtrap
open Hugin_gg
open Hugin_font
open Hugin_text

(* Expressions *)

type style = Bold | Italic | Sup | Sub | Scale of float | Color of Color.t
type tree = Str of string | Cat of tree list | Style of style * tree

let apply = function
  | Bold -> Text.bold
  | Italic -> Text.italic
  | Sup -> Text.sup
  | Sub -> Text.sub
  | Scale s -> Text.scale s
  | Color c -> Text.color c

let rec text = function
  | Str s -> Text.v s
  | Cat ts -> Text.concat (List.map text ts)
  | Style (s, t) -> apply s (text t)

(* The characters of [t]'s text: its strings in order, which the generators keep
   valid UTF-8. *)
let rec chars = function
  | Str s -> s
  | Cat ts -> String.concat "" (List.map chars ts)
  | Style (_, t) -> chars t

let pp_style ppf = function
  | Bold -> Format.pp_print_string ppf "bold"
  | Italic -> Format.pp_print_string ppf "italic"
  | Sup -> Format.pp_print_string ppf "sup"
  | Sub -> Format.pp_print_string ppf "sub"
  | Scale s -> Format.fprintf ppf "scale %g" s
  | Color c -> Format.fprintf ppf "color %a" Color.pp c

let rec pp_tree ppf = function
  | Str s -> Format.fprintf ppf "v %S" s
  | Cat ts ->
      Format.fprintf ppf "@[<1>concat [%a]@]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ")
           pp_tree)
        ts
  | Style (s, t) -> Format.fprintf ppf "@[<2>%a@ (%a)@]" pp_style s pp_tree t

(* Generators *)

(* Characters that exercise the layout: kerning pairs, spaces, every kind of
   newline, scripts, default ignorables, and characters the bundled faces lack
   (U+4E2D, a tab). *)
let pieces =
  [
    "a";
    "b";
    "A";
    "V";
    "T";
    "o";
    "y";
    ".";
    "2";
    " ";
    " ";
    "  ";
    "\n";
    "\r\n";
    "\r";
    "\u{2028}";
    "\u{03B7}";
    "\u{200D}";
    "\u{00AD}";
    "\u{FE0F}";
    "\u{4E2D}";
    "\t";
  ]

let gen_string ?(pieces = pieces) () =
  Gen.map (String.concat "")
    (Gen.list ~size:(Gen.int_range 0 6) (Gen.of_list pieces))

let styles =
  [
    Bold;
    Italic;
    Sup;
    Sub;
    Scale 2.;
    Scale 0.5;
    Color Color.red;
    Color Color.blue;
  ]

let gen_tree ?pieces () =
  let str = Gen.map (fun s -> Str s) (gen_string ?pieces ()) in
  let rec tree depth =
    if depth = 0 then str
    else
      let sub = tree (depth - 1) in
      Gen.frequency
        [
          (3, str);
          ( 2,
            Gen.map (fun ts -> Cat ts) (Gen.list ~size:(Gen.int_range 0 3) sub)
          );
          ( 2,
            Gen.map
              (fun (s, t) -> Style (s, t))
              (Gen.pair (Gen.of_list styles) sub) );
        ]
  in
  Gen.with_pp pp_tree (tree 3)

(* Faces *)

let decoded data =
  match Font.of_string data with
  | Ok f -> f
  | Error e -> Format.kasprintf failwith "%a" Font.pp_error e

let os2 ~weight ~slant =
  let b = Bytes.make 78 '\000' in
  Bytes.set_uint16_be b 4 weight;
  Bytes.set_uint16_be b 62
    (match slant with `Normal -> 0x40 | `Italic -> 0x1 | `Oblique -> 0x200);
  Bytes.to_string b

let hhea ~ascent ~descent ~gap =
  Build.cat
    [
      Build.be32 0x00010000;
      Build.be16 ascent;
      Build.be16 (-descent);
      Build.be16 gap;
      String.make 24 '\000';
      Build.be16 1;
    ]

(* [face chars] is a face of 1000 units per em whose glyph 1, a square of ink
   0.1 em wide, renders each of the code points [chars]. Every glyph advances
   0.5 em, and the face reserves [ascent] above the baseline, [descent] below it
   and [gap] between lines, in units, and has no cap height of its own, so its
   cap height is its ascent. *)
let face ?(weight = 400) ?(slant = `Normal) ?(ascent = 800) ?(descent = 200)
    ?(gap = 100) chars =
  let groups =
    List.map (fun u -> (u, u, 1)) (List.sort_uniq Int.compare chars)
  in
  let tables =
    [ ("OS/2", os2 ~weight ~slant); ("hhea", hhea ~ascent ~descent ~gap) ]
  in
  decoded
    (Build.tiny_file ~tables ~cmap:(Build.cmap_12 groups)
       [ ""; Build.simple [ Build.square ] ])
