(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_font
open Hugin_text
open Text_support

let regular = Font.regular
let inter = [ Font.regular; Font.bold ]
let cjk = face [ 0x4E2D ]

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

(* A length here is a sum of at most a few hundred terms, each rounded once, so
   two orders of evaluation differ by a few hundred half-ulps of the length:
   below 1e-12 of it, or 1e-12 for a length below 1. *)
let close = float_rel ~rel:1e-12 ~abs:1e-12

let box_w =
  let near a b = Testable.equal close a b in
  Testable.make ~pp:Box2.pp ~equal:(fun a b ->
      near (Box2.minx a) (Box2.minx b)
      && near (Box2.miny a) (Box2.miny b)
      && near (Box2.maxx a) (Box2.maxx b)
      && near (Box2.maxy a) (Box2.maxy b))

let layout_w = Testable.make ~pp:Text.Layout.pp ~equal:Text.Layout.equal
let font_w = Testable.make ~pp:Font.pp ~equal:Font.equal
let color_w = Testable.make ~pp:Color.pp ~equal:Color.equal
let tree_w = Testable.make ~pp:pp_tree ~equal:( = )
let u = Uchar.of_int

let enc us =
  let b = Buffer.create 8 in
  List.iter (fun u -> Buffer.add_utf_8_uchar b (Uchar.of_int u)) us;
  Buffer.contents b

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

let fonts_of ~fonts t = List.map (fun g -> g.font) (glyphs (lay ~fonts t))
let xs l = List.map (fun g -> g.x) (glyphs l)
let ys l = List.map (fun g -> g.y) (glyphs l)

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

(* The ranking of the faces section, as a reference. A request is a pair of
   booleans, bold and italic, and a face spec a pair of a weight and a slant,
   the plain face's first. *)
let asked (bold, italic) specs =
  let weight, slant = List.hd specs in
  ((if bold then 700 else weight), if italic then `Italic else slant)

let slants = function
  | `Italic -> [ `Italic; `Oblique; `Normal ]
  | `Oblique -> [ `Oblique; `Italic; `Normal ]
  | `Normal -> [ `Normal; `Oblique; `Italic ]

let index x l = Option.get (List.find_index (( = ) x) l)

(* The place of weight [x] in the order for the weight [w] asked for: its group,
   then [x] in a group of increasing weights and [-x] in one of decreasing
   weights. *)
let weight_key w x =
  if 400 <= w && w <= 500 then
    if w <= x && x <= 500 then (0, x) else if x < w then (1, -x) else (2, x)
  else if w < 400 then if x <= w then (0, -x) else (1, x)
  else if x >= w then (0, x)
  else (1, -x)

(* [ranking request specs] is the positions of [specs] from best to worst. *)
let ranking request specs =
  let w, s = asked request specs in
  let key i (weight, slant) =
    (index slant (slants s), weight_key w weight, i)
  in
  List.map snd
    (List.sort compare (List.mapi (fun i spec -> (key i spec, i)) specs))

let letter r = 0x61 + r

(* The face of rank [r] maps the letters [a] to the [r + 1]th, so the [i]th
   letter of a text is set in the face of rank [i] exactly when the layout ranks
   the faces as [ranking] does. *)
let ranks_law ((bold, italic) as request) specs =
  let order = ranking request specs in
  let faces =
    List.mapi
      (fun i (weight, slant) ->
        face ~weight ~slant (List.init (index i order + 1) letter))
      specs
  in
  let w, s = asked request specs in
  cover "a weight below 400 asked" (w < 400);
  cover "a weight from 400 to 500 asked" (400 <= w && w <= 500);
  cover "a weight above 500 asked" (w > 500);
  cover "the italic slant asked" (s = `Italic);
  cover "the oblique slant asked" (s = `Oblique);
  cover "the normal slant asked" (s = `Normal);
  cover "two faces of one slant and weight"
    (List.length (List.sort_uniq compare specs) < List.length specs);
  let text = String.init (List.length specs) (fun i -> Char.chr (letter i)) in
  let t = Text.v text in
  let t = if italic then Text.italic t else t in
  let t = if bold then Text.bold t else t in
  equal (list font_w)
    (List.map (List.nth faces) order)
    (fonts_of ~fonts:faces t)

let pp_slant ppf = function
  | `Normal -> Format.pp_print_string ppf "`Normal"
  | `Italic -> Format.pp_print_string ppf "`Italic"
  | `Oblique -> Format.pp_print_string ppf "`Oblique"

let pp_ranks ppf ((bold, italic), specs) =
  let pp_spec ppf (w, s) = Format.fprintf ppf "(%d, %a)" w pp_slant s in
  Format.fprintf ppf "@[<1>((%b, %b),@ [%a])@]" bold italic
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ")
       pp_spec)
    specs

let rank_cases =
  let slant = Gen.of_list [ `Normal; `Italic; `Oblique ] in
  let weight lo hi = Gen.map (fun k -> 50 * k) (Gen.int_range lo hi) in
  (* The plain face's weight is below 400, from 400 to 500 or above 500 equally
     often, so that a plain request reaches each group of weights. *)
  let plain =
    Gen.pair (Gen.one_of [ weight 1 7; weight 8 10; weight 11 20 ]) slant
  in
  let others =
    Gen.list ~size:(Gen.int_range 0 5) (Gen.pair (weight 1 20) slant)
  in
  Gen.with_pp pp_ranks
    (Gen.pair
       (Gen.pair Gen.bool Gen.bool)
       (Gen.map (fun (p, ps) -> p :: ps) (Gen.pair plain others)))

let plain = (false, false)
let bold = (true, false)
let italic = (false, true)

(* Requests that reach each rule of the faces section. *)
let rank_rows =
  [
    (* Italic weights from 450 to 500 up, then below down, then above up. *)
    ( italic,
      [
        (450, `Normal);
        (600, `Italic);
        (300, `Italic);
        (450, `Oblique);
        (500, `Italic);
        (420, `Italic);
        (480, `Italic);
      ] );
    (* Weights from 400 to 500 up, then below down, then above up. *)
    ( italic,
      [
        (400, `Normal);
        (600, `Italic);
        (350, `Italic);
        (500, `Italic);
        (100, `Italic);
        (450, `Italic);
        (400, `Italic);
      ] );
    (* Weight 500, then below down, then above up. *)
    ( italic,
      [
        (500, `Normal);
        (900, `Italic);
        (400, `Italic);
        (600, `Italic);
        (480, `Italic);
        (500, `Italic);
      ] );
    (* Weights up to 300 down, then the others up. *)
    ( italic,
      [
        (300, `Normal);
        (700, `Italic);
        (100, `Italic);
        (400, `Italic);
        (300, `Italic);
        (350, `Italic);
        (200, `Italic);
      ] );
    (* Weights from 600 up, then the others down. *)
    ( italic,
      [
        (600, `Normal);
        (100, `Italic);
        (550, `Italic);
        (900, `Italic);
        (500, `Italic);
        (650, `Italic);
        (600, `Italic);
      ] );
    (* Bold asks for weight 700. *)
    ( bold,
      [
        (400, `Normal);
        (300, `Normal);
        (700, `Italic);
        (600, `Normal);
        (900, `Normal);
        (700, `Normal);
      ] );
    (* Bold italic asks for weight 700 and the italic slant. *)
    ( (true, true),
      [
        (400, `Normal);
        (700, `Normal);
        (400, `Italic);
        (700, `Oblique);
        (900, `Italic);
        (700, `Italic);
      ] );
    (* Italic, then oblique, then normal, then position. *)
    (italic, [ (400, `Normal); (400, `Oblique); (400, `Normal); (400, `Italic) ]);
    (* Oblique, then italic, then normal, for an oblique plain face. *)
    (plain, [ (400, `Oblique); (400, `Normal); (400, `Italic); (400, `Oblique) ]);
    (* Normal, then oblique, then italic, for a normal plain face. *)
    (plain, [ (400, `Normal); (400, `Italic); (400, `Oblique); (400, `Normal) ]);
    (* Plain text in the first face, whichever it is. *)
    (plain, [ (400, `Normal); (700, `Normal) ]);
    (plain, [ (700, `Normal); (400, `Normal) ]);
    (* Bold in the bold face, or in the plain face without one. *)
    (bold, [ (400, `Normal); (700, `Normal) ]);
    (bold, [ (400, `Normal) ]);
    (* Italic upright without an italic face. *)
    (italic, [ (400, `Normal); (700, `Normal) ]);
  ]

(* [fallback_faces sets] is a face for each set of letters, mapping them and a
   digit of its own. *)
let fallback_faces sets =
  List.mapi
    (fun i set -> face ((Char.code '0' + i) :: List.map Char.code set))
    sets

(* The model of fallback among faces of one style: the first face of the list
   that maps the character, else the first face's [.notdef]. *)
let fallback_agrees (sets, s) =
  let faces = fallback_faces sets in
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
  cover "a fallback"
    (List.exists (fun (f, g) -> g = 1 && f != List.hd faces) expected);
  equal
    (list (pair font_w int))
    expected
    (List.map (fun g -> (g.font, g.id)) (glyphs (lay ~fonts:faces (Text.v s))))

let fallback_cases =
  let letters = [ 'a'; 'b'; 'c'; 'd' ] in
  let set = Gen.subsequence ~pp:Format.pp_print_char letters in
  Gen.pair
    (Gen.list ~size:(Gen.int_range 1 3) set)
    (Gen.string_of ~size:(Gen.int_range 0 8)
       (Gen.of_list [ 'a'; 'b'; 'c'; 'd'; 'e' ]))

let faces =
  group "faces"
    [
      prop ~count:200 ~examples:rank_rows
        "rank by slant, then weight, then position" rank_cases
        (fun (request, specs) -> ranks_law request specs);
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
      test "a missing character is in the text of its run" (fun () ->
          equal (list string) [ "x\u{4E2D}y" ]
            (texts (lay (Text.v "x\u{4E2D}y"))));
      test "a combining mark is set as a character of its own" (fun () ->
          let l = lay ~fonts:[ face [ 0x65; 0x301 ] ] (Text.v "e\u{0301}") in
          equal (list close) [ 0.; 5. ] (xs l));
      test "right-to-left characters are placed left to right" (fun () ->
          let l =
            lay ~fonts:[ face [ 0x5D0; 0x5D1 ] ] (Text.v "\u{05D0}\u{05D1}")
          in
          equal (list string) [ "\u{05D0}\u{05D1}" ] (texts l);
          equal (list close) [ 0.; 5. ] (xs l));
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

(* [t] with default ignorable characters inserted around every string and
   between its characters. *)
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

(* Texts where a hidden character sits where it could change the layout: in a
   kerned pair, after a space that hangs, among leading spaces, between CR and
   LF, and alone. *)
let hidden_examples =
  [
    (Str "AV", 60.);
    (Str "x ", 60.);
    (Str "  x y", 0.);
    (Str "a\r\nb", 60.);
    (Str "", 60.);
  ]

let ignorables =
  group "default ignorables"
    [
      cases "a default ignorable is set as absent, its neighbours are drawn"
        ~id:(fun (c, hidden) ->
          Printf.sprintf "U+%04X is %s" c (if hidden then "hidden" else "drawn"))
        ignorable_edges
        (fun (c, hidden) ->
          let l =
            lay
              ~fonts:[ Lazy.force edges_face ]
              (Text.v (enc [ 0x61; c; 0x61 ]))
          in
          equal (list string)
            [ (if hidden then "aa" else enc [ 0x61; c; 0x61 ]) ]
            (texts l));
      test "a soft hyphen a face draws is set as absent" (fun () ->
          let soft = face [ 0x61; 0x62; 0xAD ] in
          equal layout_w
            (lay ~fonts:[ soft ] (Text.v "ab"))
            (lay ~fonts:[ soft ] (Text.v "a\u{00AD}b"));
          equal layout_w
            (lay ~fonts:[ regular; soft ] (Text.v "ab"))
            (lay ~fonts:[ regular; soft ] (Text.v "a\u{00AD}b")));
      prop ~examples:hidden_examples
        "a text sets as it would without its default ignorables"
        (Gen.pair (gen_tree ()) (Gen.float_range 0. 60.))
        (fun (t, width) ->
          Law.ignores tree_w layout_w (fun t -> lay ~width (text t)) hide t);
    ]

(* Lines *)

(* The number of lines of [l] when every line has the bundled faces' pitch. *)
let line_count l = Float.to_int (Float.round (Box2.h (box l) /. pitch))
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

(* [lines_of paragraphs rs] is the runs [rs] of a text in the bundled regular
   face cut into the lines of each paragraph, one run a line. *)
let rec lines_of paragraphs rs =
  let rec take n rs acc =
    if n <= 0 then (List.rev acc, rs)
    else
      match rs with
      | [] -> failf "the lines lack %d words" n
      | ((_, _, r) as run) :: rs ->
          take (n - List.length (words (Run.text r))) rs (run :: acc)
  in
  match paragraphs with
  | [] ->
      equal int 0 (List.length rs);
      []
  | ws :: paragraphs ->
      let lines, rs = take (List.length ws) rs [] in
      lines :: lines_of paragraphs rs

let breaking_law (paragraphs, w) =
  let source = String.concat "\n" (List.map (String.concat " ") paragraphs) in
  let rs = runs (lay ~fonts:[ regular ] ~width:w (Text.v source)) in
  let line_words =
    List.map
      (List.map (fun (_, _, r) -> words (Run.text r)))
      (lines_of paragraphs rs)
  in
  let several lw = List.length lw > 1 in
  cover "a line of several words" (List.exists (List.exists several) line_words);
  cover "a line that cannot take the next word" (List.exists several line_words);
  cover "a word wider than the width"
    (List.exists (fun run -> run_end run > w) rs);
  List.iter2
    (fun ws lws -> equal (list string) ws (List.concat lws))
    paragraphs line_words;
  List.iter
    (fun ((_, at, r) as run) ->
      let lw = words (Run.text r) in
      equal float_exact 0. (P2.x at);
      equal string (String.concat " " lw) (Run.text r);
      if several lw then at_most float_exact ~than:w (run_end run))
    rs;
  let rec first_fit = function
    | a :: (b :: _ as rest) ->
        let joined = String.concat " " (a @ [ List.hd b ]) in
        greater float_exact ~than:w (width ~fonts:[ regular ] joined);
        first_fit rest
    | _ -> ()
  in
  List.iter first_fit line_words;
  ignore
    (List.fold_left
       (fun prev (_, at, _) ->
         greater float_exact ~than:prev (P2.y at);
         P2.y at)
       neg_infinity rs)

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
  cover "unbroken" (w = infinity);
  let us = List.filter (fun c -> not (List.mem c hidden)) (decode (chars t)) in
  let drawn = String.concat "" (texts (lay ~width:w (text t))) in
  let visible us = List.filter (fun c -> c <> 0x20 && not (is_newline c)) us in
  equal string (enc (visible us)) (enc (visible (decode drawn)));
  if w = infinity then
    let trimmed l = List.rev (drop_spaces (List.rev l)) in
    equal string (enc (List.concat_map trimmed (paragraphs us))) drawn

let lines =
  group "lines"
    [
      cases "a newline ends a line" ~id:(Printf.sprintf "U+%04X") newlines
        (fun c ->
          equal layout_w
            (lay (Text.v "a\nb"))
            (lay (Text.v (enc [ 0x61; c; 0x62 ]))));
      cases "a neighbour of the newlines ends no line"
        ~id:(Printf.sprintf "U+%04X") [ 0x09; 0x0E; 0x84; 0x86; 0x2027; 0x202A ]
        (fun c ->
          equal int 1 (line_count (lay (Text.v (enc [ 0x61; c; 0x62 ])))));
      test "U+000D U+000A counts as one newline" (fun () ->
          equal int 2 (line_count (lay (Text.v "a\r\nb")));
          equal int 3 (line_count (lay (Text.v "a\r\r\nb"))));
      test "U+000A U+000D counts as two newlines" (fun () ->
          equal int 3 (line_count (lay (Text.v "a\n\rb"))));
      cases "n newlines make n + 1 lines" ~id:(Printf.sprintf "%d newlines")
        [ 0; 1; 2; 5 ] (fun n ->
          equal int (n + 1) (line_count (lay (Text.v (String.make n '\n')))));
      test "the empty text is one empty line" (fun () ->
          let l = lay (Text.v "") in
          equal (list string) [] (texts l);
          equal box_w
            (Box2.of_pts (P2.v 0. (-10. *. ascent)) (P2.v 0. (10. *. descent)))
            (box l));
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
      prop ~count:200 "runs carry the characters of the text"
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

(* What a run shares with its glyphs: a face, a size, a baseline and a
   colour. *)
let run_style (c, at, r) = (Run.font r, Run.size r, P2.y at, c)
let style_w = quad font_w float_exact float_exact (option color_w)

let contiguous t =
  let rs = runs (lay (text t)) in
  cover "two runs on one line" (List.length rs > 1);
  let rec loop = function
    | ((_, _, r) as a) :: ((_, at', r') as b) :: rest ->
        not_equal style_w (run_style a) (run_style b);
        let f, s, y, _ = run_style a and f', s', y', _ = run_style b in
        let k =
          if Font.equal f f' && Float.equal s s' && Float.equal y y' then
            Font.kerning f (Run.glyph r (Run.length r - 1)) (Run.glyph r' 0)
            *. s
          else 0.
        in
        classify "kerned across runs" (k <> 0.);
        equal close (run_end a +. k) (P2.x at');
        loop (b :: rest)
    | _ -> ()
  in
  loop rs

(* A line of four runs, the first two kerned across a colour change. *)
let several_runs =
  Cat
    [
      Str "A";
      Style (Color Color.red, Str "V");
      Style (Bold, Str "o");
      Style (Sup, Str "T");
    ]

(* Line metrics *)

(* The cap height of the first line of [t], from where [`Cap] puts its first
   glyph against where [`Baseline] does on one line. *)
let cap_of ?(fonts = inter) t =
  let y valign = (List.hd (glyphs (lay ~valign ~fonts t))).y in
  y `Cap -. y `Baseline

let line_metrics =
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
        equal (list close) [ -11.; 0. ] (ys (lay ~fonts:[ p ] (Text.v "a\na")));
        equal (list close) [ 8.; 19. ]
          (ys (lay ~valign:`Top ~fonts:[ p ] (Text.v "a\na"))));
    test "the cap height is the plain face's at the baseline's size" (fun () ->
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
        equal close (10. *. cap) (cap_of (Text.sup (Text.v "2"))));
  ]

let placing =
  group "placing"
    ([
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
                   (lay ~fonts:[ regular ]
                      Text.(concat [ v "A"; bold (v "V") ])))));
       test "kerning never crosses a size change" (fun () ->
           equal close
             ((10. *. adv regular 'A') +. (20. *. adv regular 'V'))
             (wide Text.(concat [ v "A"; scale 2. (v "V") ])));
       test "kerning never crosses a shift change" (fun () ->
           equal close
             (7. *. (adv regular 'A' +. adv regular 'V'))
             (wide Text.(concat [ sup (v "A"); sub (v "V") ])));
       test "kerning crosses an equal shift" (fun () ->
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
       cases "scales and scripts compose inside out"
         ~id:(fun (name, _, _) -> name)
         [
           ( "scale 2 of a superscript",
             Text.(scale 2. (concat [ v "x"; sup (v "2") ])),
             [ (20., 0.); (14., -8.) ] );
           ( "a superscript of a subscript",
             Text.(sup (sub (v "i"))),
             [ (4.9, -2.6) ] );
           ( "a subscript of a superscript",
             Text.(sub (sup (v "i"))),
             [ (4.9, -0.8) ] );
           ("scale 0.5", Text.(scale 0.5 (v "x")), [ (5., 0.) ]);
         ]
         (fun (_, t, expected) ->
           equal (list (pair close close)) expected (sized t));
       cases "runs are maximal sequences of one face, size, shift and colour"
         ~id:(fun (name, _, _, _) -> name)
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
         ~examples:[ several_runs ]
         (gen_tree ~pieces:no_newlines ())
         contiguous;
     ]
    @ line_metrics)

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
      cases "vertical alignments put their line at y = 0"
        ~id:(fun (v, _) -> valign_name v ^ " on one line")
        [
          (`Top, 10. *. ascent);
          (`Cap, 10. *. cap);
          (`Middle, 5. *. cap);
          (`Baseline, 0.);
          (`Bottom, -10. *. descent);
        ]
        (fun (valign, y) ->
          equal (list close) [ y ] (ys (lay ~valign (Text.v "x"))));
      cases "vertical alignments of two lines"
        ~id:(fun (v, _) -> valign_name v ^ " on two lines")
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
              (fun ((_, at, _) as r) -> (P2.x at, run_end r))
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
      prop "vertical alignments move a layout as their anchors say"
        (gen_tree ()) valign_law;
      prop "horizontal alignments place the widest line" (gen_tree ())
        halign_law;
    ]

(* Layouts *)

let rejects f = raises_match Exn.invalid_arg (fun () -> ignore (f ()))

let halign_name = function
  | `Left -> "left"
  | `Center -> "center"
  | `Right -> "right"

(* [scaled_law] checks that a layout at size [10. *. c] has the lengths of the
   layout at size [10.] times [c]. A power of two [c] scales every rounded
   product and sum exactly, so the two layouts break at the same points. *)
let scaled_law ((t, e), (w, halign, valign)) =
  let c = Float.ldexp 1. e in
  let at size width = lay ~size ~width ~halign ~valign (text t) in
  let l = at 10. w and l' = at (10. *. c) (w *. c) in
  let times b =
    Box2.of_pts
      (P2.v (c *. Box2.minx b) (c *. Box2.miny b))
      (P2.v (c *. Box2.maxx b) (c *. Box2.maxy b))
  in
  let placed k l =
    List.map
      (fun (_, at, r) -> ((k *. P2.x at, k *. P2.y at), k *. Run.size r))
      (runs l)
  in
  equal box_w (times (box l)) (box l');
  equal (option box_w)
    (Option.map times (Text.Layout.ink l))
    (Text.Layout.ink l');
  equal (list (pair (pair close close) close)) (placed c l) (placed 1. l')

let scaled_cases =
  Gen.pair
    (Gen.pair (gen_tree ()) (Gen.int_range (-3) 3))
    (Gen.triple widths
       (Gen.of_list
          ~pp:(fun ppf h -> Format.pp_print_string ppf (halign_name h))
          [ `Left; `Center; `Right ])
       (Gen.of_list
          ~pp:(fun ppf v -> Format.pp_print_string ppf (valign_name v))
          [ `Top; `Cap; `Middle; `Baseline; `Bottom ]))

(* A panel title, a tick label and a legend entry wrapped to its column, set at
   sizes 5, 2.5 and 5 against 10. *)
let figure_texts =
  [
    ((Style (Bold, Str "(a)"), -1), (infinity, `Left, `Bottom));
    ((Str "0.25", -2), (infinity, `Right, `Middle));
    ( (Str "validation loss of the larger model, smoothed", -1),
      (120., `Left, `Top) );
  ]

let v =
  group "v"
    [
      prop ~examples:figure_texts "sets lengths in the unit of its size"
        scaled_cases scaled_law;
      test "aligns left and on the baseline by default" (fun () ->
          equal layout_w
            (lay ~halign:`Left ~valign:`Baseline (Text.v "a\nb"))
            (lay (Text.v "a\nb")));
      test "raises on an empty face list" (fun () ->
          rejects (fun () -> Text.Layout.v ~fonts:[] ~size:10. (Text.v "a")));
      cases "raises on a size that is not finite and positive"
        ~id:(Printf.sprintf "%h")
        [ 0.; -0.; -1.; Float.nan; Float.infinity; Float.neg_infinity ]
        (fun size -> rejects (fun () -> lay ~size (Text.v "a")));
      cases "raises on a width that is negative or nan"
        ~id:(Printf.sprintf "%h")
        [ -1.; Float.nan; Float.neg_infinity; -.Float.min_float ] (fun width ->
          rejects (fun () -> lay ~width (Text.v "a")));
      cases "raises on a character whose size is not finite and positive"
        ~id:(fun (name, _) -> name)
        [
          ("a size that underflows", Text.(scale 1e-200 (scale 1e-200 (v "x"))));
          ("a size that overflows", Text.(scale 1e200 (scale 1e200 (v "x"))));
          ("a newline's size", Text.(scale 1e-200 (scale 1e-200 (v "\n"))));
        ]
        (fun (_, t) -> rejects (fun () -> lay t));
      test "ignores the size of a default ignorable character" (fun () ->
          equal layout_w
            (lay (Text.v ""))
            (lay Text.(scale 1e-200 (scale 1e-200 (v "\u{200B}")))));
      cases "raises on a coordinate that is not finite"
        ~id:(fun (name, _, _) -> name)
        [
          ("advances past max_float", 1e308, Text.v "xxxx");
          ( "a shift past max_float",
            1e10,
            Text.(scale 1e300 (sup (scale 1e-300 (v "x")))) );
        ]
        (fun (_, size, t) -> rejects (fun () -> lay ~size t));
      test "accepts a size near max_float with finite coordinates" (fun () ->
          ignore (lay ~size:1e300 (Text.v "x")));
    ]

let boxes =
  group "box"
    [
      test "spans every line" (fun () ->
          let wxx = width "xx" in
          equal box_w
            (Box2.of_pts
               (P2.v (-.wxx /. 2.) (-.pitch -. (10. *. ascent)))
               (P2.v (wxx /. 2.) (10. *. descent)))
            (box (lay ~halign:`Center (Text.v "xx\nx"))));
    ]

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

let ink =
  group "ink"
    [
      prop "is the union of the ink of the runs" (gen_tree ()) ink_law;
      test "of a square glyph is the square at its origin" (fun () ->
          equal (option box_w)
            (Some (Box2.of_pts (P2.v 0. (-1.)) (P2.v 1. 0.)))
            (Text.Layout.ink (lay ~fonts:[ face [ 0x61 ] ] (Text.v "a"))));
      cases "is None for a text without ink" ~id:(Printf.sprintf "%S")
        [ ""; " "; "  \n "; "\u{200B}"; " \u{00A0}" ] (fun s ->
          equal (option box_w) None (Text.Layout.ink (lay (Text.v s))));
    ]

(* The model of [missing] for the faces of [fallback_faces sets]: the letters of
   [s] that no set has, each once, in order of first occurrence. *)
let missing_agrees (sets, s) =
  let letters = List.init (String.length s) (String.get s) in
  let mapped c = List.exists (List.mem c) sets in
  let first_lacks c = not (List.mem c (List.hd sets)) in
  cover "a character only a later face has"
    (List.exists (fun c -> first_lacks c && mapped c) letters);
  let once =
    List.fold_left
      (fun acc c -> if mapped c || List.mem c acc then acc else acc @ [ c ])
      [] letters
  in
  equal (list uchar)
    (List.map Uchar.of_char once)
    (Text.Layout.missing (lay ~fonts:(fallback_faces sets) (Text.v s)))

let missing =
  group "missing"
    [
      prop "lists the characters no face has, each once, in order"
        fallback_cases missing_agrees;
      test "lists each absent character once, in order" (fun () ->
          equal (list uchar)
            [ u 0x4E2D; u 0x6587; u 0x09 ]
            (Text.Layout.missing
               (lay (Text.v "\u{4E2D}x\u{6587}\u{4E2D}\t\u{6587}"))));
      test "never lists newlines or default ignorables" (fun () ->
          let l =
            lay
              ~fonts:[ face [ 0x61 ] ]
              (Text.v "a\n\r\n\u{2029}\u{200D}\u{E0001}\u{00AD}a")
          in
          equal (list uchar) [] (Text.Layout.missing l));
    ]

(* [paint l] paints the runs of [l] as {!Text.Layout.fold} says, in black where
   they have no colour. *)
let paint l =
  Hugin_vg.Picture.group
    (List.rev
       (Text.Layout.fold
          (fun acc c at r ->
            Hugin_vg.Picture.glyphs (Option.value c ~default:Color.black) at r
            :: acc)
          [] l))

let fold =
  group "fold"
    [
      test "visits the runs in the order of the text, with their colours"
        (fun () ->
          let l =
            lay
              Text.(
                concat
                  [ v "a"; color Color.red (v "b"); v "\nc "; bold (v "d") ])
          in
          equal
            (list (pair (option color_w) string))
            [ (None, "a"); (Some Color.red, "b"); (None, "c "); (None, "d") ]
            (List.map (fun (c, _, r) -> (c, Run.text r)) (runs l)));
      prop "its runs painted by Picture.glyphs have the layout's ink"
        (gen_tree ()) (fun t ->
          let l = lay ~width:30. (text t) in
          equal (option box_w) (Text.Layout.ink l)
            (Hugin_vg.Picture.bounds (paint l)));
    ]

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

let equality =
  group "equal"
    [
      prop "is an equivalence"
        (Gen.pair (gen_tree ()) (gen_tree ()))
        (fun (a, b) -> Law.equivalence layout_w (lay (text a), lay (text b)));
      prop "holds for the layouts of equal texts" (gen_tree ()) (fun t ->
          Law.ignores tree_w layout_w
            (fun t -> lay ~width:40. (text t))
            respell t);
      test "is false for equal runs in different boxes" (fun () ->
          not_equal layout_w
            (lay ~valign:`Top (Text.v "x"))
            (lay ~valign:`Top (Text.v "x\n")));
      cases "is false for different runs"
        ~id:(fun (name, _, _) -> name)
        [
          ("characters", lay (Text.v "a"), lay (Text.v "b"));
          ("colours", lay (Text.v "a"), lay (Text.color Color.red (Text.v "a")));
          ("origins", lay (Text.v "a\nb"), lay ~halign:`Center (Text.v "a\nb"));
          ("faces", lay (Text.v "a"), lay (Text.bold (Text.v "a")));
        ]
        (fun (_, a, b) -> not_equal layout_w a b);
    ]

let show l = Format.asprintf "%a" Text.Layout.pp l

let pp =
  group "pp"
    [
      test "formats the box and each run at its origin" (fun () ->
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
      test "formats each run's face and colour" (fun () ->
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
    (run "hugin.text: Layout"
       [
         faces;
         ignorables;
         lines;
         placing;
         alignment;
         v;
         boxes;
         ink;
         missing;
         fold;
         equality;
         pp;
       ])
