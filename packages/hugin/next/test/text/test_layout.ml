(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_font
open Hugin_next_text
open Text_support

let regular = Font.regular
let inter = [ Font.regular; Font.bold ]

(* The bundled faces' metrics, in em, as their OS/2 tables record them in units
   of 2048. *)
let ascent = 1984. /. 2048.
let descent = 494. /. 2048.
let cap = 1490. /. 2048.

(* The pitch of the lines of the bundled faces at size 10, whose line gap is
   0. *)
let pitch = 10. *. (ascent +. descent)

let lay ?width ?halign ?valign ?(fonts = inter) ?(size = 10.) t =
  Text.Layout.v ?width ?halign ?valign ~fonts ~size t

let box l = Text.Layout.box l
let close = float 1e-9
let layout_w = Testable.make ~pp:Text.Layout.pp ~equal:Text.Layout.equal
let font_w = Testable.make ~pp:Font.pp ~equal:Font.equal
let color_w = Testable.make ~pp:Color.pp ~equal:Color.equal

let box_w =
  let near a b = Float.abs (a -. b) <= 1e-9 in
  Testable.make ~pp:Box2.pp ~equal:(fun a b ->
      near (Box2.minx a) (Box2.minx b)
      && near (Box2.miny a) (Box2.miny b)
      && near (Box2.maxx a) (Box2.maxx b)
      && near (Box2.maxy a) (Box2.maxy b))

let u = Uchar.of_int
let tree_w = Testable.make ~pp:pp_tree ~equal:( = )

let enc us =
  let b = Buffer.create 8 in
  List.iter (fun u -> Buffer.add_utf_8_uchar b (Uchar.of_int u)) us;
  Buffer.contents b

(* Observing layouts *)

let runs l =
  List.rev (Text.Layout.fold (fun acc c at r -> (c, at, r) :: acc) [] l)

let texts l = List.map (fun (_, _, r) -> Run.text r) (runs l)

type glyph = { font : Font.t; id : Font.glyph; x : float; y : float; s : float }

let glyphs l =
  List.concat_map
    (fun (_, at, r) ->
      List.init (Run.length r) (fun i ->
          {
            font = Run.font r;
            id = Run.glyph r i;
            x = P2.x at +. Run.x r i;
            y = P2.y at +. Run.y r i;
            s = Run.size r;
          }))
    (runs l)

(* Where the advance of the last glyph of a run ends. *)
let run_end (_, at, r) =
  let n = Run.length r - 1 in
  P2.x at +. Run.x r n
  +. (Font.advance (Run.font r) (Run.glyph r n) *. Run.size r)

(* The advance of [c] in [f], in em. *)
let adv f c = Font.advance f (Font.glyph f (Uchar.of_char c))

let kern f a b =
  Font.kerning f
    (Font.glyph f (Uchar.of_char a))
    (Font.glyph f (Uchar.of_char b))

let width ?fonts ?size s = Box2.w (box (lay ?fonts ?size (Text.v s)))

(* Faces *)

(* [ranked ~style specs] checks that the faces [(weight, slant, rank)] of
   [specs], listed in that order, rank for [style] as their [rank]s say. The
   face of rank [r] maps the letters [a] to the [r + 1]th, so the [i]th letter
   of a text is set in the face of rank [i] exactly when the ranking is
   right. *)
let ranked ~style specs =
  let letter r = 0x61 + r in
  let faces =
    List.map
      (fun (weight, slant, rank) ->
        (rank, face ~weight ~slant (List.init (rank + 1) letter)))
      specs
  in
  let text = String.init (List.length specs) (fun i -> Char.chr (letter i)) in
  let l = lay ~fonts:(List.map snd faces) (style (Text.v text)) in
  equal (list font_w)
    (List.map snd (List.sort (fun (r, _) (r', _) -> Int.compare r r') faces))
    (List.map (fun g -> g.font) (glyphs l))

let rankings =
  [
    ( "weights from 450 to 500 up, then below down, then above up",
      Text.italic,
      [
        (450, `Normal, 6);
        (600, `Italic, 4);
        (300, `Italic, 3);
        (450, `Oblique, 5);
        (500, `Italic, 1);
        (420, `Italic, 2);
        (480, `Italic, 0);
      ] );
    ( "weights from 400 to 500 up, then below down, then above up",
      Text.italic,
      [
        (400, `Normal, 6);
        (600, `Italic, 5);
        (350, `Italic, 3);
        (500, `Italic, 2);
        (100, `Italic, 4);
        (450, `Italic, 1);
        (400, `Italic, 0);
      ] );
    ( "weight 500, then below down, then above up",
      Text.italic,
      [
        (500, `Normal, 5);
        (900, `Italic, 4);
        (400, `Italic, 2);
        (600, `Italic, 3);
        (480, `Italic, 1);
        (500, `Italic, 0);
      ] );
    ( "weights up to 300 down, then the others up",
      Text.italic,
      [
        (300, `Normal, 6);
        (700, `Italic, 5);
        (100, `Italic, 2);
        (400, `Italic, 4);
        (300, `Italic, 0);
        (350, `Italic, 3);
        (200, `Italic, 1);
      ] );
    ( "weights from 600 up, then the others down",
      Text.italic,
      [
        (600, `Normal, 6);
        (100, `Italic, 5);
        (550, `Italic, 3);
        (900, `Italic, 2);
        (500, `Italic, 4);
        (650, `Italic, 1);
        (600, `Italic, 0);
      ] );
    ( "bold asks for weight 700",
      Text.bold,
      [
        (400, `Normal, 3);
        (300, `Normal, 4);
        (700, `Italic, 5);
        (600, `Normal, 2);
        (900, `Normal, 1);
        (700, `Normal, 0);
      ] );
    ( "bold italic asks for weight 700 and the italic slant",
      (fun t -> Text.bold (Text.italic t)),
      [
        (400, `Normal, 5);
        (700, `Normal, 4);
        (400, `Italic, 2);
        (700, `Oblique, 3);
        (900, `Italic, 1);
        (700, `Italic, 0);
      ] );
    ( "italic, then oblique, then normal, then position",
      Text.italic,
      [
        (400, `Normal, 2);
        (400, `Oblique, 1);
        (400, `Normal, 3);
        (400, `Italic, 0);
      ] );
    ( "oblique, then italic, then normal, for an oblique plain face",
      Fun.id,
      [
        (400, `Oblique, 0);
        (400, `Normal, 3);
        (400, `Italic, 2);
        (400, `Oblique, 1);
      ] );
    ( "normal, then oblique, then italic, for a normal plain face",
      Fun.id,
      [
        (400, `Normal, 0);
        (400, `Italic, 3);
        (400, `Oblique, 2);
        (400, `Normal, 1);
      ] );
  ]

let fonts_of text ~fonts = List.map (fun g -> g.font) (glyphs (lay ~fonts text))

(* The model of fallback among faces of one style: the first face of the list
   that maps the character, else the first face's [.notdef]. *)
let fallback_agrees (sets, s) =
  let faces =
    List.mapi
      (fun i set -> face ((Char.code '0' + i) :: List.map Char.code set))
      sets
  in
  let l = lay ~fonts:faces (Text.v s) in
  let expected =
    List.init (String.length s) (fun i ->
        let c = s.[i] in
        let rec find k = function
          | [] -> (List.hd faces, 0)
          | set :: sets ->
              if List.mem c set then (List.nth faces k, 1)
              else find (k + 1) sets
        in
        find 0 sets)
  in
  let missing =
    List.sort_uniq Char.compare
      (List.filter
         (fun c -> not (List.exists (List.mem c) sets))
         (List.init (String.length s) (String.get s)))
  in
  classify "a fallback"
    (List.exists (fun (f, g) -> g = 1 && f != List.hd faces) expected);
  equal
    (list (pair font_w int))
    expected
    (List.map (fun g -> (g.font, g.id)) (glyphs l));
  equal
    (slist uchar Uchar.compare)
    (List.map Uchar.of_char missing)
    (Text.Layout.missing l)

let fallback_cases =
  let letters = [ 'a'; 'b'; 'c'; 'd' ] in
  let set = Gen.subsequence ~pp:Format.pp_print_char letters in
  Gen.pair
    (Gen.list ~size:(Gen.int_range 1 3) set)
    (Gen.string_of ~size:(Gen.int_range 0 8)
       (Gen.of_list [ 'a'; 'b'; 'c'; 'd'; 'e' ]))

let cjk = face [ 0x4E2D ]

let faces =
  group "faces"
    [
      test "plain text is set in the first face" (fun () ->
          equal (list font_w) [ regular; regular ]
            (fonts_of (Text.v "ax") ~fonts:[ regular; Font.bold ]);
          equal (list font_w) [ Font.bold; Font.bold ]
            (fonts_of (Text.v "ax") ~fonts:[ Font.bold; regular ]));
      test "bold text is set in the bundled bold face" (fun () ->
          equal (list font_w) [ Font.bold ]
            (fonts_of (Text.bold (Text.v "x")) ~fonts:inter));
      test "bold text without a bold face is set in the plain face" (fun () ->
          equal (list font_w) [ regular ]
            (fonts_of (Text.bold (Text.v "x")) ~fonts:[ regular ]));
      test "italic text without an italic face is set upright" (fun () ->
          equal (list font_w) [ regular ]
            (fonts_of (Text.italic (Text.v "x")) ~fonts:inter));
      cases
        ~name:(fun (name, _, _) -> name)
        "faces rank by slant, weight and position" rankings
        (fun (_, style, specs) -> ranked ~style specs);
      test "each style of one text asks for its own face" (fun () ->
          let plain = face [ 0x61 ] and bold = face ~weight:700 [ 0x61 ] in
          let italic = face ~slant:`Italic [ 0x61 ] in
          let bold_italic = face ~weight:700 ~slant:`Italic [ 0x61 ] in
          let fonts = [ plain; bold; italic; bold_italic ] in
          equal (list font_w)
            [ plain; bold; italic; bold_italic ]
            (fonts_of ~fonts
               Text.(
                 concat
                   [
                     v "a"; bold (v "a"); italic (v "a"); bold (italic (v "a"));
                   ]));
          equal (list font_w) [ italic; bold ]
            (fonts_of ~fonts Text.(concat [ italic (v "a"); bold (v "a") ])));
      test "a character the first face lacks comes from the next that has it"
        (fun () ->
          let late = face [ 0x4E2D; 0x61 ] in
          equal (list font_w) [ regular; cjk; regular ]
            (fonts_of (Text.v "a\u{4E2D}a") ~fonts:[ regular; cjk; late ]);
          equal (list font_w) [ cjk ]
            (fonts_of
               (Text.bold (Text.v "\u{4E2D}"))
               ~fonts:[ regular; Font.bold; cjk ]));
      test "a character no face has is the .notdef of the best-ranked face"
        (fun () ->
          let g t = List.map (fun g -> (g.font, g.id)) (glyphs (lay t)) in
          equal
            (list (pair font_w int))
            [ (regular, 0) ]
            (g (Text.v "\u{4E2D}"));
          equal
            (list (pair font_w int))
            [ (Font.bold, 0) ]
            (g (Text.bold (Text.v "\u{4E2D}"))));
      prop "fallback takes the first face of the list that has the character"
        fallback_cases fallback_agrees;
      test "missing lists each absent character once, in order" (fun () ->
          equal (list uchar)
            [ u 0x4E2D; u 0x6587; u 0x09 ]
            (Text.Layout.missing
               (lay (Text.v "\u{4E2D}x\u{6587}\u{4E2D}\t\u{6587}"))));
      test "newlines and default ignorables are never missing" (fun () ->
          let l =
            lay
              ~fonts:[ face [ 0x61 ] ]
              (Text.v "a\n\r\n\u{2029}\u{200D}\u{E0001}\u{00AD}a")
          in
          equal (list uchar) [] (Text.Layout.missing l));
      test "a missing character is in the text of its run" (fun () ->
          equal (list string) [ "x\u{4E2D}y" ]
            (texts (lay (Text.v "x\u{4E2D}y"))));
    ]

(* Default ignorables *)

(* The ranges of Default_Ignorable_Code_Point in Unicode 16.0. *)
let ignorable_ranges =
  [
    (0x00AD, 0x00AD);
    (0x034F, 0x034F);
    (0x061C, 0x061C);
    (0x115F, 0x1160);
    (0x17B4, 0x17B5);
    (0x180B, 0x180F);
    (0x200B, 0x200F);
    (0x202A, 0x202E);
    (0x2060, 0x206F);
    (0x3164, 0x3164);
    (0xFE00, 0xFE0F);
    (0xFEFF, 0xFEFF);
    (0xFFA0, 0xFFA0);
    (0xFFF0, 0xFFF8);
    (0x1BCA0, 0x1BCA3);
    (0x1D173, 0x1D17A);
    (0xE0000, 0xE0FFF);
  ]

let newlines = [ 0x0A; 0x0B; 0x0C; 0x0D; 0x85; 0x2028; 0x2029 ]
let is_newline c = List.mem c newlines

let ignorable_edges =
  List.concat_map
    (fun (lo, hi) ->
      let outside =
        List.filter
          (fun c ->
            (not (is_newline c))
            && not
                 (List.exists
                    (fun (lo, hi) -> lo <= c && c <= hi)
                    ignorable_ranges))
          [ lo - 1; hi + 1 ]
      in
      List.map (fun c -> (c, true)) (List.sort_uniq Int.compare [ lo; hi ])
      @ List.map (fun c -> (c, false)) outside)
    ignorable_ranges

(* A face that draws every edge with ink. *)
let edges_face = lazy (face (0x61 :: List.map fst ignorable_edges))

(* [t] with default ignorable characters inserted around every string. *)
let rec hide = function
  | Str s ->
      let rec chars i acc =
        if i >= String.length s then List.rev acc
        else
          let n = Uchar.utf_decode_length (String.get_utf_8_uchar s i) in
          chars (i + n) (String.sub s i n :: acc)
      in
      Cat
        [
          Str "\u{200B}";
          Str (String.concat "\u{2060}" (chars 0 []));
          Str "\u{FE0F}";
        ]
  | Cat ts -> Cat (Str "\u{E0001}" :: List.map hide ts)
  | Style (st, t) -> Style (st, hide t)

let ignorables =
  group "default ignorables"
    [
      cases
        ~name:(fun (c, hidden) ->
          Printf.sprintf "U+%04X is %s" c (if hidden then "hidden" else "drawn"))
        "the property's edges, mapped by the face" ignorable_edges
        (fun (c, hidden) ->
          let l =
            lay
              ~fonts:[ Lazy.force edges_face ]
              (Text.v (enc [ 0x61; c; 0x61 ]))
          in
          equal (list string)
            [ (if hidden then "aa" else enc [ 0x61; c; 0x61 ]) ]
            (texts l);
          equal (list uchar) [] (Text.Layout.missing l));
      test "a soft hyphen a face draws is hidden" (fun () ->
          let soft = face [ 0x61; 0x62; 0xAD ] in
          List.iter
            (fun fonts ->
              let l = lay ~fonts (Text.v "a\u{00AD}b") in
              equal (list string) [ "ab" ] (texts l);
              equal int 2 (List.length (glyphs l));
              equal (list uchar) [] (Text.Layout.missing l))
            [ [ soft ]; [ regular; soft ] ]);
      test "kerning crosses a hidden joiner" (fun () ->
          equal layout_w (lay (Text.v "AV")) (lay (Text.v "A\u{200D}V")));
      test "a hidden character after a space lets the space hang" (fun () ->
          equal layout_w
            (lay ~halign:`Right (Text.v "x"))
            (lay ~halign:`Right (Text.v "x \u{200D}")));
      test "a hidden character among leading spaces offers no break" (fun () ->
          equal layout_w
            (lay ~width:0. (Text.v "  x y"))
            (lay ~width:0. (Text.v " \u{FEFF} x y")));
      test "a hidden character between CR and LF keeps them one newline"
        (fun () ->
          equal layout_w (lay (Text.v "a\r\nb")) (lay (Text.v "a\r\u{200B}\nb")));
      test "a text of default ignorables is the empty text" (fun () ->
          equal layout_w
            (lay (Text.v ""))
            (lay (Text.v "\u{200B}\u{FEFF}\u{E0001}")));
      prop "a text sets as it would without its default ignorables"
        (Gen.pair (gen_tree ()) (Gen.float_range 0. 60.))
        (fun (t, width) ->
          Law.ignores tree_w layout_w (fun t -> lay ~width (text t)) hide t);
    ]

(* Lines *)

(* The number of lines of [l] when every line has the bundled faces' pitch. *)
let line_count l = Float.to_int (Float.round (Box2.h (box l) /. pitch))
let ys l = List.map (fun g -> g.y) (glyphs l)
let xs l = List.map (fun g -> g.x) (glyphs l)

let decode s =
  let rec loop i acc =
    if i >= String.length s then List.rev acc
    else
      let d = String.get_utf_8_uchar s i in
      loop
        (i + Uchar.utf_decode_length d)
        (Uchar.to_int (Uchar.utf_decode_uchar d) :: acc)
  in
  loop 0 []

let words s = List.filter (( <> ) "") (String.split_on_char ' ' s)

let word =
  Gen.map (String.concat "")
    (Gen.list ~size:(Gen.int_range 1 5)
       (Gen.of_list [ "a"; "A"; "V"; "o"; "T"; "y"; "." ]))

let widths =
  Gen.frequency
    [
      (1, Gen.constant ~pp:Format.pp_print_float 0.);
      (6, Gen.float_range 0. 120.);
      (1, Gen.constant ~pp:Format.pp_print_float infinity);
    ]

(* Paragraphs of words, one space apart, and a width to break them at. *)
let breaking_cases =
  Gen.pair
    (Gen.list ~size:(Gen.int_range 1 3)
       (Gen.list ~size:(Gen.int_range 0 6) word))
    widths

let breaking_law (paragraphs, w) =
  let source = String.concat "\n" (List.map (String.concat " ") paragraphs) in
  let l = lay ~fonts:[ regular ] ~width:w (Text.v source) in
  let rs = runs l in
  List.iter (fun (_, at, _) -> equal float_exact 0. (P2.x at)) rs;
  let rec take n rs acc =
    if n <= 0 then (List.rev acc, rs)
    else
      match rs with
      | [] -> failf "the lines lack %d words" n
      | ((_, _, r) as run) :: rs ->
          take (n - List.length (words (Run.text r))) rs (run :: acc)
  in
  let rec first_fit = function
    | a :: (b :: _ as rest) ->
        let joined = String.concat " " (a @ [ List.hd b ]) in
        greater float_exact ~than:w (width ~fonts:[ regular ] joined);
        cover "a line that cannot take the next word" true;
        first_fit rest
    | _ -> ()
  in
  let rec check paragraphs rs =
    match paragraphs with
    | [] -> equal int 0 (List.length rs)
    | ws :: paragraphs ->
        let lines, rs = take (List.length ws) rs [] in
        let line_words = List.map (fun (_, _, r) -> words (Run.text r)) lines in
        equal (list string) ws (List.concat line_words);
        List.iter2
          (fun ((_, _, r) as run) lw ->
            equal string (String.concat " " lw) (Run.text r);
            if List.length lw > 1 then begin
              cover "a line of several words" true;
              at_most float_exact ~than:w (run_end run)
            end
            else classify "a line of one word" true)
          lines line_words;
        first_fit line_words;
        check paragraphs rs
  in
  check paragraphs rs;
  let baselines = List.map (fun (_, at, _) -> P2.y at) rs in
  ignore
    (List.fold_left
       (fun prev y ->
         greater float_exact ~than:prev y;
         y)
       neg_infinity baselines)

(* The default ignorables among [pieces]. *)
let hidden = [ 0x200D; 0xAD; 0xFE0F ]

(* The lines that newlines end in [us]. *)
let paragraphs us =
  let rec loop line acc = function
    | [] -> List.rev (List.rev line :: acc)
    | 0x0D :: 0x0A :: us -> loop [] (List.rev line :: acc) us
    | c :: us when is_newline c -> loop [] (List.rev line :: acc) us
    | c :: us -> loop (c :: line) acc us
  in
  loop [] [] us

let rec drop_spaces = function 0x20 :: us -> drop_spaces us | us -> us

let carry_law (t, w) =
  let us = List.filter (fun c -> not (List.mem c hidden)) (decode (chars t)) in
  let drawn = String.concat "" (texts (lay ~width:w (text t))) in
  let visible us = List.filter (fun c -> c <> 0x20 && not (is_newline c)) us in
  equal string (enc (visible us)) (enc (visible (decode drawn)));
  if w = infinity then begin
    cover "unbroken" true;
    let trimmed l = List.rev (drop_spaces (List.rev l)) in
    equal string (enc (List.concat_map trimmed (paragraphs us))) drawn
  end

let lines =
  group "lines"
    [
      cases ~name:(Printf.sprintf "U+%04X ends a line") "newlines" newlines
        (fun c ->
          equal layout_w
            (lay (Text.v "a\nb"))
            (lay (Text.v (enc [ 0x61; c; 0x62 ]))));
      test "CR LF is one newline and LF CR two" (fun () ->
          equal int 2 (line_count (lay (Text.v "a\r\nb")));
          equal int 3 (line_count (lay (Text.v "a\n\rb")));
          equal int 3 (line_count (lay (Text.v "a\r\r\nb"))));
      cases
        ~name:(fun n -> Printf.sprintf "%d newlines make %d lines" n (n + 1))
        "line counts" [ 0; 1; 2; 5 ]
        (fun n ->
          equal int (n + 1) (line_count (lay (Text.v (String.make n '\n')))));
      test "the empty text is one empty line of the plain face's height"
        (fun () ->
          let l = lay (Text.v "") in
          equal box_w
            (Box2.of_pts (P2.v 0. (-10. *. ascent)) (P2.v 0. (10. *. descent)))
            (box l);
          equal int 0 (List.length (runs l));
          is_none (Text.Layout.ink l);
          equal (list uchar) [] (Text.Layout.missing l));
      test "spaces alone make empty lines" (fun () ->
          equal layout_w (lay (Text.v "")) (lay (Text.v "   "));
          equal layout_w (lay (Text.v "\n")) (lay (Text.v "  \n ")));
      test "leading spaces count" (fun () ->
          let l = lay ~fonts:[ regular ] (Text.v "  x") in
          equal (list string) [ "  x" ] (texts l);
          equal close
            (10.
            *. (adv regular ' ' +. kern regular ' ' ' ' +. adv regular ' '
              +. kern regular ' ' 'x' +. adv regular 'x'))
            (Box2.w (box l)));
      test "the spaces that end a line hang" (fun () ->
          equal layout_w (lay (Text.v "x")) (lay (Text.v "x   "));
          equal layout_w
            (lay ~halign:`Right (Text.v "x\ny"))
            (lay ~halign:`Right (Text.v "x  \ny"));
          equal layout_w
            (lay ~width:0. ~halign:`Center (Text.v "x\ny"))
            (lay ~width:0. ~halign:`Center (Text.v "x   y")));
      test "only a space offers a break" (fun () ->
          equal (list string) [ "a-b\u{00A0}c/d"; "e" ]
            (texts (lay ~width:0. (Text.v "a-b\u{00A0}c/d e"))));
      test "no break follows leading spaces" (fun () ->
          equal (list string) [ "  a"; "b" ]
            (texts (lay ~width:0. (Text.v "  a b"))));
      test "width 0 puts each word on its own line" (fun () ->
          equal (list string) [ "ab"; "cd"; "ef" ]
            (texts (lay ~width:0. (Text.v "ab cd  ef"))));
      test "a word wider than the width overflows a line of its own" (fun () ->
          equal (list string) [ "a"; "bbbbbb"; "a b" ]
            (texts (lay ~width:(width "a b") (Text.v "a bbbbbb a b"))));
      test "a line takes words while it is at most the width" (fun () ->
          let w = width "aa aa" in
          equal (list string) [ "aa aa"; "aa" ]
            (texts (lay ~width:w (Text.v "aa aa aa")));
          equal (list string) [ "aa"; "aa"; "aa" ]
            (texts (lay ~width:(Float.pred w) (Text.v "aa aa aa"))));
      test "an infinite width breaks only at newlines" (fun () ->
          equal (list string) [ "aa aa aa"; "b" ]
            (texts (lay (Text.v "aa aa aa\nb"))));
      test "lines from breaks and newlines follow one another" (fun () ->
          let l = lay ~width:0. (Text.v "a b\nc") in
          equal (list string) [ "a"; "b"; "c" ] (texts l);
          equal (list close) [ -2. *. pitch; -.pitch; 0. ] (ys l));
      prop "lines break where the width says" breaking_cases breaking_law;
      prop "runs carry the characters of the text"
        (Gen.pair (gen_tree ()) widths)
        carry_law;
    ]

(* Placing glyphs *)

let kerning_letters = [ 'A'; 'V'; 'T'; 'o'; 'y'; '.'; 'W'; 'L'; 'a'; 'v'; 'F' ]

let seam_cases =
  Gen.pair
    (Gen.string_of ~size:(Gen.int_range 1 6) (Gen.of_list kerning_letters))
    (Gen.string_of ~size:(Gen.int_range 1 6) (Gen.of_list kerning_letters))

let seam_law (a, b) =
  let k = kern regular a.[String.length a - 1] b.[0] in
  cover "a kerned seam" (k <> 0.);
  let width = width ~fonts:[ regular ] in
  equal close (width a +. width b +. (10. *. k)) (width (a ^ b))

let wide t = Box2.w (box (lay t))
let sized t = List.map (fun g -> (g.s, g.y)) (glyphs (lay t))

let no_newlines =
  List.filter (fun p -> not (List.exists is_newline (decode p))) pieces

let contiguous t =
  let rec loop = function
    | ((c, at, r) as a) :: ((c', at', r') :: _ as rest) ->
        let same =
          Font.equal (Run.font r) (Run.font r')
          && Float.equal (Run.size r) (Run.size r')
          && Float.equal (P2.y at) (P2.y at')
        in
        is_false ~msg:"a run continues the previous one"
          (same && Option.equal Color.equal c c');
        let n = Run.length r - 1 in
        let k =
          if same then
            Font.kerning (Run.font r) (Run.glyph r n) (Run.glyph r' 0)
            *. Run.size r
          else 0.
        in
        classify "kerned across runs" (k <> 0.);
        equal close (run_end a +. k) (P2.x at');
        loop rest
    | _ -> ()
  in
  loop (runs (lay (text t)))

let placing =
  group "placing"
    [
      test "the bundled faces kern A and V" (fun () ->
          less float_exact ~than:0. (kern regular 'A' 'V'));
      prop "a concatenation advances by its parts and their kerning" seam_cases
        seam_law;
      test "kerning crosses a colour change" (fun () ->
          equal close (width "AV")
            (wide Text.(concat [ v "A"; color Color.red (v "V") ])));
      test "kerning never crosses a face change" (fun () ->
          equal close
            (10. *. (adv regular 'A' +. adv Font.bold 'V'))
            (wide Text.(concat [ v "A"; bold (v "V") ])));
      test "kerning crosses a style set in one face" (fun () ->
          equal close
            (width ~fonts:[ regular ] "AV")
            (Box2.w
               (box
                  (lay ~fonts:[ regular ] Text.(concat [ v "A"; bold (v "V") ])))));
      test "kerning never crosses a size change" (fun () ->
          equal close
            ((10. *. adv regular 'A') +. (20. *. adv regular 'V'))
            (wide Text.(concat [ v "A"; scale 2. (v "V") ])));
      test "kerning never crosses a shift change" (fun () ->
          equal close
            (7. *. (adv regular 'A' +. adv regular 'V'))
            (wide Text.(concat [ sup (v "A"); sub (v "V") ]));
          equal close
            (7. *. (adv regular 'A' +. kern regular 'A' 'V' +. adv regular 'V'))
            (wide Text.(concat [ sup (v "A"); sup (v "V") ])));
      test "a superscript is set at 0.7 of the size, 0.4 of it up" (fun () ->
          equal
            (list (pair close close))
            [ (10., 0.); (7., -4.) ]
            (sized Text.(concat [ v "x"; sup (v "2") ])));
      test "a subscript is set at 0.7 of the size, 0.2 of it down" (fun () ->
          equal
            (list (pair close close))
            [ (10., 0.); (7., 2.) ]
            (sized Text.(concat [ v "x"; sub (v "2") ])));
      test "scales and scripts compose inside out" (fun () ->
          equal
            (list (pair close close))
            [ (20., 0.); (14., -8.) ]
            (sized Text.(scale 2. (concat [ v "x"; sup (v "2") ])));
          equal
            (list (pair close close))
            [ (4.9, -2.6) ]
            (sized Text.(sup (sub (v "i"))));
          equal
            (list (pair close close))
            [ (4.9, -0.8) ]
            (sized Text.(sub (sup (v "i"))));
          equal
            (list (pair close close))
            [ (5., 0.) ]
            (sized (Text.scale 0.5 (Text.v "x"))));
      cases
        ~name:(fun (name, _, _, _) -> name)
        "runs are maximal sequences of one face, size, shift and colour"
        [
          ("one style", Text.v "ab", inter, [ "ab" ]);
          ( "a colour change",
            Text.(concat [ v "a"; color Color.red (v "b") ]),
            inter,
            [ "a"; "b" ] );
          ( "an equal colour",
            Text.(
              concat
                [ color Color.red (v "a"); color (Color.v 1. 0. 0.) (v "b") ]),
            inter,
            [ "ab" ] );
          ( "a face change",
            Text.(concat [ v "a"; bold (v "b") ]),
            inter,
            [ "a"; "b" ] );
          ( "a style set in one face",
            Text.(concat [ v "a"; bold (v "b") ]),
            [ regular ],
            [ "ab" ] );
          ( "a size change",
            Text.(concat [ v "a"; scale 2. (v "b") ]),
            inter,
            [ "a"; "b" ] );
          ( "a shift change",
            Text.(concat [ sup (v "a"); sub (v "b") ]),
            inter,
            [ "a"; "b" ] );
          ( "a fallback",
            Text.v "a\u{4E2D}b",
            [ regular; cjk ],
            [ "a"; "\u{4E2D}"; "b" ] );
          ("spaces between words", Text.v "a  b", inter, [ "a  b" ]);
        ]
        (fun (_, t, fonts, expected) ->
          equal (list string) expected (texts (lay ~fonts t)));
      prop "a run's glyphs start at its origin, on its baseline" (gen_tree ())
        (fun t ->
          List.iter
            (fun (_, _, r) ->
              equal float_exact 0. (Run.x r 0);
              for i = 0 to Run.length r - 1 do
                equal float_exact 0. (Run.y r i)
              done)
            (runs (lay ~width:30. (text t))));
      prop "a run has one glyph per character" (gen_tree ()) (fun t ->
          List.iter
            (fun (_, _, r) ->
              equal int (List.length (decode (Run.text r))) (Run.length r))
            (runs (lay ~width:30. (text t))));
      prop "the runs of a line are contiguous and maximal"
        (gen_tree ~pieces:no_newlines ())
        contiguous;
    ]

(* Line metrics *)

(* The cap height of the first line of [t], from where [`Cap] puts its first
   glyph against where [`Baseline] does on one line. *)
let cap_of ?(fonts = inter) t =
  let y valign = (List.hd (glyphs (lay ~valign ~fonts t))).y in
  y `Cap -. y `Baseline

let metrics =
  group "line metrics"
    [
      test "a line reserves the plain face's ascent and descent" (fun () ->
          equal box_w
            (Box2.of_pts
               (P2.v 0. (-10. *. ascent))
               (P2.v (width "x") (10. *. descent)))
            (box (lay (Text.v "x"))));
      test "a taller face makes its line taller" (fun () ->
          let tall = face ~ascent:1500 ~descent:500 [ 0x4E2D ] in
          let b = box (lay ~fonts:[ regular; tall ] (Text.v "x\u{4E2D}")) in
          equal close (-15.) (Box2.miny b);
          equal close 5. (Box2.maxy b));
      test "a superscript makes its line taller" (fun () ->
          let b = box (lay Text.(concat [ v "x"; sup (v "2") ])) in
          equal close (-.((7. *. ascent) +. 4.)) (Box2.miny b);
          equal close (10. *. descent) (Box2.maxy b));
      test "a subscript makes its line deeper" (fun () ->
          let b = box (lay Text.(concat [ v "x"; sub (v "2") ])) in
          equal close (-10. *. ascent) (Box2.miny b);
          equal close ((7. *. descent) +. 2.) (Box2.maxy b));
      test "the plain face keeps the pitch of small text" (fun () ->
          equal (list close) [ -.pitch; 0. ]
            (ys (lay (Text.scale 0.5 (Text.v "x\nx")))));
      test "baselines are apart by descent, line gap and ascent" (fun () ->
          let p = face [ 0x61 ] in
          equal (list close) [ -11.; 0. ]
            (ys (lay ~fonts:[ p ] (Text.v "a\na")));
          equal (list close) [ 8.; 19. ]
            (ys (lay ~valign:`Top ~fonts:[ p ] (Text.v "a\na"))));
      test "the cap height is the plain face's at the baseline's size"
        (fun () ->
          equal close (10. *. cap) (cap_of (Text.v "x"));
          equal close (20. *. cap) (cap_of (Text.scale 2. (Text.v "X")));
          equal close (5. *. cap) (cap_of (Text.scale 0.5 (Text.v "X")));
          equal close (20. *. cap)
            (cap_of Text.(concat [ v "x"; scale 2. (v "X") ]));
          equal close (20. *. cap)
            (cap_of Text.(concat [ scale 2. (v "X"); v "x" ])));
      test "a fallback face's cap height does not count" (fun () ->
          equal close (10. *. cap)
            (cap_of ~fonts:[ regular; cjk ] (Text.v "\u{4E2D}")));
      test "scripts do not count for the cap height" (fun () ->
          equal close (10. *. cap)
            (cap_of Text.(concat [ v "x"; sup (scale 2. (v "2")) ]));
          let only_script = Text.sup (Text.v "2") in
          let y valign = (List.hd (glyphs (lay ~valign only_script))).y in
          equal close (10. *. cap) (y `Cap -. y `Baseline));
    ]

(* Alignment *)

let valign_name = function
  | `Top -> "top"
  | `Cap -> "cap"
  | `Middle -> "middle"
  | `Baseline -> "baseline"
  | `Bottom -> "bottom"

let moved_by d b =
  Box2.of_pts
    (P2.v (Box2.minx b) (Box2.miny b +. d))
    (P2.v (Box2.maxx b) (Box2.maxy b +. d))

let valign_law t =
  let t = text t in
  let base = lay ~width:40. t in
  let shift valign =
    let l = lay ~width:40. ~valign t in
    let d = Box2.miny (box l) -. Box2.miny (box base) in
    equal box_w (moved_by d (box base)) (box l);
    List.iter2
      (fun g g' ->
        equal float_exact g.x g'.x;
        equal close (g.y +. d) g'.y)
      (glyphs base) (glyphs l);
    d
  in
  let top = shift `Top and cap = shift `Cap and middle = shift `Middle in
  let bottom = shift `Bottom in
  equal close (-.Box2.miny (box base)) top;
  equal close (-.Box2.maxy (box base)) bottom;
  equal close (cap /. 2.) middle

let halign_law t =
  let t = text t in
  let b halign = box (lay ~width:40. ~halign t) in
  let w = Box2.w (b `Left) in
  equal close 0. (Box2.minx (b `Left));
  equal close (-.w) (Box2.minx (b `Right));
  equal close 0. (Box2.maxx (b `Right));
  equal close (-.w /. 2.) (Box2.minx (b `Center));
  equal close (w /. 2.) (Box2.maxx (b `Center))

let alignment =
  group "alignment"
    [
      cases
        ~name:(fun (v, _) -> valign_name v ^ " on one line")
        "vertical alignments put their line at y = 0"
        [
          (`Top, 10. *. ascent);
          (`Cap, 10. *. cap);
          (`Middle, 5. *. cap);
          (`Baseline, 0.);
          (`Bottom, -10. *. descent);
        ]
        (fun (valign, y) ->
          equal (list close) [ y ] (ys (lay ~valign (Text.v "x"))));
      cases
        ~name:(fun (v, _) -> valign_name v ^ " on two lines")
        "vertical alignments of two lines"
        [
          (`Top, [ 10. *. ascent; (10. *. ascent) +. pitch ]);
          (`Cap, [ 10. *. cap; (10. *. cap) +. pitch ]);
          ( `Middle,
            [ ((10. *. cap) -. pitch) /. 2.; ((10. *. cap) +. pitch) /. 2. ] );
          (`Baseline, [ -.pitch; 0. ]);
          (`Bottom, [ -.pitch -. (10. *. descent); -10. *. descent ]);
        ]
        (fun (valign, y) ->
          equal (list close) y (ys (lay ~valign (Text.v "x\nx"))));
      test "bottom takes the last line's descent" (fun () ->
          let l = lay ~valign:`Bottom Text.(concat [ v "x\nx"; sub (v "2") ]) in
          equal close (-.((7. *. descent) +. 2.)) (List.nth (ys l) 1));
      test "cap takes the first line's cap height" (fun () ->
          let l =
            lay ~valign:`Cap Text.(concat [ scale 2. (v "X"); v "\nx" ])
          in
          equal close (20. *. cap) (List.hd (ys l)));
      test "horizontal alignments place each line" (fun () ->
          let ends halign =
            List.map
              (fun r -> ((fun (_, at, _) -> P2.x at) r, run_end r))
              (runs (lay ~halign (Text.v "xx\nx")))
          in
          let wxx = width "xx" and wx = width "x" in
          equal (list (pair close close)) [ (0., wxx); (0., wx) ] (ends `Left);
          equal
            (list (pair close close))
            [ (-.wxx, 0.); (-.wx, 0.) ]
            (ends `Right);
          equal
            (list (pair close close))
            [ (-.wxx /. 2., wxx /. 2.); (-.wx /. 2., wx /. 2.) ]
            (ends `Center));
      test "the default alignment is left and baseline" (fun () ->
          equal layout_w
            (lay ~halign:`Left ~valign:`Baseline (Text.v "a\nb"))
            (lay (Text.v "a\nb")));
      prop "vertical alignments move a layout as their anchors say"
        (gen_tree ()) valign_law;
      prop "horizontal alignments place the widest line" (gen_tree ())
        halign_law;
    ]

(* Measuring and comparing *)

let moved (_, at, _) b =
  Box2.of_pts
    (P2.v (Box2.minx b +. P2.x at) (Box2.miny b +. P2.y at))
    (P2.v (Box2.maxx b +. P2.x at) (Box2.maxy b +. P2.y at))

let ink_law t =
  let l = lay ~width:30. (text t) in
  let expected =
    List.fold_left
      (fun acc ((_, _, r) as run) ->
        match Run.bounds r with
        | None -> acc
        | Some b ->
            let b = moved run b in
            Some (match acc with None -> b | Some u -> Box2.union u b))
      None (runs l)
  in
  equal (option box_w) expected (Text.Layout.ink l)

(* [t] written differently: its strings split in two, its styles pushed into its
   concatenations. *)
let rec respell = function
  | Str s ->
      let rec middle i =
        if i >= String.length s / 2 then i
        else middle (i + Uchar.utf_decode_length (String.get_utf_8_uchar s i))
      in
      let i = middle 0 in
      Cat [ Str (String.sub s 0 i); Str (String.sub s i (String.length s - i)) ]
  | Cat ts -> Cat (List.map respell ts)
  | Style (st, Cat ts) -> Cat (List.map (fun t -> Style (st, respell t)) ts)
  | Style (st, t) -> Style (st, respell t)

let measuring =
  group "measuring and comparing"
    [
      test "the box spans every line" (fun () ->
          let wxx = width "xx" in
          equal box_w
            (Box2.of_pts
               (P2.v (-.wxx /. 2.) (-.pitch -. (10. *. ascent)))
               (P2.v (wxx /. 2.) (10. *. descent)))
            (box (lay ~halign:`Center (Text.v "xx\nx"))));
      prop "ink is the union of the ink of the runs" (gen_tree ()) ink_law;
      test "the ink of a square glyph" (fun () ->
          equal (option box_w)
            (Some (Box2.of_pts (P2.v 0. (-1.)) (P2.v 1. 0.)))
            (Text.Layout.ink (lay ~fonts:[ face [ 0x61 ] ] (Text.v "a"))));
      test "a text without ink has none" (fun () ->
          List.iter
            (fun s -> is_none (Text.Layout.ink (lay (Text.v s))))
            [ ""; " "; "  \n "; "\u{200B}"; " \u{00A0}" ]);
      test "fold visits the runs in the order of the text" (fun () ->
          let l =
            lay
              Text.(
                concat
                  [ v "a"; color Color.red (v "b"); v "\nc "; bold (v "d") ])
          in
          equal (list string) [ "a"; "b"; "c "; "d" ] (texts l);
          equal
            (list (option color_w))
            [ None; Some Color.red; None; None ]
            (List.map (fun (c, _, _) -> c) (runs l)));
      prop "equal is an equivalence"
        (Gen.pair (gen_tree ()) (gen_tree ()))
        (fun (a, b) -> Law.equivalence layout_w (lay (text a), lay (text b)));
      prop "equal texts make equal layouts" (gen_tree ()) (fun t ->
          Law.ignores tree_w layout_w
            (fun t -> lay ~width:40. (text t))
            respell t);
      test "layouts with equal runs and different boxes are unequal" (fun () ->
          not_equal layout_w
            (lay ~valign:`Top (Text.v "x"))
            (lay ~valign:`Top (Text.v "x\n")));
      cases
        ~name:(fun (name, _, _) -> name)
        "layouts with different runs are unequal"
        [
          ("characters", lay (Text.v "a"), lay (Text.v "b"));
          ("colours", lay (Text.v "a"), lay (Text.color Color.red (Text.v "a")));
          ("origins", lay (Text.v "a\nb"), lay ~halign:`Center (Text.v "a\nb"));
          ("faces", lay (Text.v "a"), lay (Text.bold (Text.v "a")));
        ]
        (fun (_, a, b) -> not_equal layout_w a b);
    ]

(* Characters *)

let characters =
  group "characters"
    [
      test "characters beyond the bundled faces are .notdef, deterministically"
        (fun () ->
          let l = lay (Text.v "\u{4E2D}\u{6587}") in
          equal
            (list (pair font_w int))
            [ (regular, 0); (regular, 0) ]
            (List.map (fun g -> (g.font, g.id)) (glyphs l));
          equal close (20. *. Font.advance regular 0) (Box2.w (box l));
          equal (list uchar) [ u 0x4E2D; u 0x6587 ] (Text.Layout.missing l);
          equal layout_w l (lay (Text.v "\u{4E2D}\u{6587}")));
      test "a tab is missing from the bundled faces" (fun () ->
          let l = lay (Text.v "x\ty") in
          equal (list uchar) [ u 0x09 ] (Text.Layout.missing l);
          equal (list int) [ 0 ]
            (List.filter (( = ) 0) (List.map (fun g -> g.id) (glyphs l))));
      test "NUL is a blank glyph of the bundled faces" (fun () ->
          let l = lay (Text.v "\x00") in
          equal (list uchar) [] (Text.Layout.missing l);
          not_equal (list int) [ 0 ] (List.map (fun g -> g.id) (glyphs l));
          equal close (10. *. 1200. /. 2048.) (Box2.w (box l));
          is_none (Text.Layout.ink l));
      test "a combining mark is set on its own" (fun () ->
          let l = lay (Text.v "e\u{0301}") in
          equal (list uchar) [ u 0x301 ] (Text.Layout.missing l);
          equal int 2 (List.length (glyphs l)));
      test "right-to-left characters are placed left to right" (fun () ->
          let l =
            lay ~fonts:[ face [ 0x5D0; 0x5D1 ] ] (Text.v "\u{05D0}\u{05D1}")
          in
          equal (list string) [ "\u{05D0}\u{05D1}" ] (texts l);
          equal (list close) [ 0.; 5. ] (xs l));
      test "invalid UTF-8 is drawn as U+FFFD" (fun () ->
          let l = lay (Text.v "a\xFF") in
          equal (list string) [ "a\u{FFFD}" ] (texts l);
          equal (list uchar) [ Uchar.rep ] (Text.Layout.missing l));
    ]

(* Errors *)

let rejects f = raises_match Exn.invalid_arg (fun () -> ignore (f ()))

let errors =
  group "errors"
    [
      test "an empty face list" (fun () ->
          rejects (fun () -> Text.Layout.v ~fonts:[] ~size:10. (Text.v "a")));
      cases ~name:(Printf.sprintf "size %h")
        "sizes that are not finite and positive"
        [ 0.; -0.; -1.; Float.nan; Float.infinity; Float.neg_infinity ]
        (fun size -> rejects (fun () -> lay ~size (Text.v "a")));
      cases ~name:(Printf.sprintf "width %h") "widths that are negative or nan"
        [ -1.; Float.nan; Float.neg_infinity; -.Float.min_float ] (fun width ->
          rejects (fun () -> lay ~width (Text.v "a")));
      test "widths 0 and -0 are accepted" (fun () ->
          ignore (lay ~width:0. (Text.v "a b"));
          ignore (lay ~width:(-0.) (Text.v "a b")));
      cases
        ~name:(fun (name, _) -> name)
        "characters whose size is not finite and positive"
        [
          ("a size that underflows", Text.(scale 1e-200 (scale 1e-200 (v "x"))));
          ("a size that overflows", Text.(scale 1e200 (scale 1e200 (v "x"))));
          ("a newline's size", Text.(scale 1e-200 (scale 1e-200 (v "\n"))));
        ]
        (fun (_, t) -> rejects (fun () -> lay t));
      test "the size of a default ignorable character is not checked" (fun () ->
          equal layout_w
            (lay (Text.v ""))
            (lay Text.(scale 1e-200 (scale 1e-200 (v "\u{200B}")))));
      cases
        ~name:(fun (name, _, _) -> name)
        "coordinates that are not finite"
        [
          ("advances past max_float", 1e308, Text.v "xxxx");
          ( "a shift past max_float",
            1e10,
            Text.(scale 1e300 (sup (scale 1e-300 (v "x")))) );
        ]
        (fun (_, size, t) -> rejects (fun () -> lay ~size t));
      test "a size near max_float with finite coordinates is accepted"
        (fun () -> ignore (lay ~size:1e300 (Text.v "x")));
    ]

(* Layouts of figures *)

let show l = Format.asprintf "%a" Text.Layout.pp l

let goldens =
  group "figure layouts"
    [
      test "a panel title sitting on its anchor" (fun () ->
          expect
            (show (lay ~valign:`Bottom ~size:9.6 (Text.bold (Text.v "(a)"))))
          @@ __POS_OF__
               {|
            (layout [(0, -11.6156) (12.8109, 0)]
             ((0, -2.31562) -
              (run (font "Inter" 700 normal) 9.6 "(a)" 375@0,0#0 147@3.61875,0#1
               376@9.19219,0#2)))
            |});
      test "an axis title with a subscript, hanging below the axis" (fun () ->
          expect
            (show
               (lay ~halign:`Center ~valign:`Cap
                  Text.(concat [ v "step size \u{03B7}"; sub (v "0") ])))
          @@ __POS_OF__
               {|
            (layout [(-27.6943, -2.41211) (27.6943, 10.9639)]
             ((-27.6943, 7.27539) -
              (run (font "Inter" 400 normal) 10 "step size η" 241@0,0#0 247@5.27832,0#1
               171@8.45215,0#2 235@14.2822,0#3 560@20.4053,0#4 241@23.2178,0#5
               192@28.4961,0#6 273@30.918,0#7 171@36.2939,0#8 560@42.124,0#9
               300@44.9365,0#10))
             ((23.1553, 9.27539) - (run (font "Inter" 400 normal) 7 "0" 720@0,0#0)))
            |});
      test "a tick label centred beside its tick" (fun () ->
          expect
            (show (lay ~halign:`Right ~valign:`Middle ~size:8. (Text.v "0.25")))
          @@ __POS_OF__
               {|
            (layout [(-17.8672, -4.83984) (0, 4.83984)]
             ((-17.8672, 2.91016) -
              (run (font "Inter" 400 normal) 8 "0.25" 720@0,0#0 421@5.1875,0#1
               722@7.49219,0#2 725@12.6797,0#3)))
            |});
      test "a legend entry wrapped to its column" (fun () ->
          expect
            (show
               (lay ~width:96. ~valign:`Top ~size:8.
                  (Text.v "validation loss of the larger model, smoothed")))
          @@ __POS_OF__
               {|
            (layout [(0, 0) (89.7344, 19.3594)]
             ((0, 7.75) -
              (run (font "Inter" 400 normal) 8 "validation loss of the" 264@0,0#0
               147@4.45703,0#1 209@8.94922,0#2 192@10.8867,0#3 167@12.8242,0#4
               147@17.7227,0#5 247@22.2148,0#6 192@24.832,0#7 222@26.7695,0#8
               216@31.5664,0#9 560@36.293,0#10 209@38.543,0#11 222@40.4805,0#12
               241@45.2773,0#13 241@49.5,0#14 560@53.7227,0#15 222@55.9727,0#16
               182@60.7695,0#17 560@63.7305,0#18 247@65.9805,0#19 189@68.5977,0#20
               171@73.3281,0#21))
             ((0, 17.4297) -
              (run (font "Inter" 400 normal) 8 "larger model, smoothed" 209@0,0#0
               147@1.9375,0#1 237@6.42969,0#2 184@9.30859,0#3 171@14.2148,0#4
               237@18.8789,0#5 560@21.8906,0#6 215@24.1406,0#7 222@31.1484,0#8
               167@35.9453,0#9 171@40.8438,0#10 209@45.5078,0#11 420@47.4453,0#12
               560@49.75,0#13 241@52,0#14 215@56.2227,0#15 222@63.2305,0#16
               222@68.0273,0#17 247@72.8242,0#18 189@75.4414,0#19 171@80.1719,0#20
               167@84.8359,0#21)))
            |});
      test "a fallback face, bold and a colour" (fun () ->
          expect
            (show
               (lay
                  ~fonts:[ regular; Font.bold; cjk ]
                  Text.(
                    concat
                      [
                        v "loss ";
                        bold (v "\u{4E2D}x");
                        color Color.red (v " diverged");
                      ])))
          @@ __POS_OF__
               {|
            (layout [(0, -9.6875) (76.8848, 2.41211)]
             ((0, 0) -
              (run (font "Inter" 400 normal) 10 "loss " 209@0,0#0 222@2.42188,0#1
               241@8.41797,0#2 241@13.6963,0#3 560@18.9746,0#4))
             ((21.7871, 0) - (run (font "" 400 normal) 10 "中" 1@0,0#0))
             ((26.7871, 0) - (run (font "Inter" 700 normal) 10 "x" 267@0,0#0))
             ((32.5879, 0) #ff0000
              (run (font "Inter" 400 normal) 10 " diverged" 560@0,0#0 167@2.8125,0#1
               192@8.93555,0#2 264@11.3574,0#3 171@16.7822,0#4 237@22.6123,0#5
               184@26.2109,0#6 171@32.3438,0#7 167@38.1738,0#8)))
            |});
    ]

let () =
  exit
    (run "hugin.next.text: Layout"
       [
         faces;
         ignorables;
         lines;
         placing;
         metrics;
         alignment;
         measuring;
         characters;
         errors;
         goldens;
       ])
