(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_font

let run_t = Testable.make ~pp:Run.pp ~equal:Run.equal
let font = Font.regular

(* One glyph per character of [text], set at [size] from [x0]. *)
let set ?(size = 10.) ?(x0 = 0.) text =
  let rec chars i acc =
    if i >= String.length text then List.rev acc
    else
      let d = String.get_utf_8_uchar text i in
      chars (i + Uchar.utf_decode_length d) (Uchar.utf_decode_uchar d :: acc)
  in
  let glyphs = Array.of_list (List.map (Font.glyph font) (chars 0 [])) in
  let xs = Array.make (Array.length glyphs) x0 in
  for i = 1 to Array.length glyphs - 1 do
    xs.(i) <- xs.(i - 1) +. (size *. Font.advance font glyphs.(i - 1))
  done;
  (glyphs, xs)

let fields () =
  let glyphs, xs = set "Hé→" in
  let r = Run.v ~font ~size:10. ~text:"Hé→" ~glyphs ~xs () in
  equal int 3 (Run.length r);
  equal string "Hé→" (Run.text r);
  equal float_exact 10. (Run.size r);
  is_true (Font.equal font (Run.font r));
  equal (list int) (Array.to_list glyphs) (List.init 3 (Run.glyph r));
  equal (list float_exact) (Array.to_list xs) (List.init 3 (Run.x r));
  equal ~msg:"ys default to the baseline" (list float_exact) [ 0.; 0.; 0. ]
    (List.init 3 (Run.y r));
  equal ~msg:"one glyph per character" (list int) [ 0; 1; 3 ]
    (List.init 3 (Run.cluster r))

let copies () =
  let glyphs, xs = set "Ho" in
  let ys = [| 1.; 2. |] and clusters = [| 0; 1 |] in
  let r = Run.v ~ys ~clusters ~font ~size:10. ~text:"Ho" ~glyphs ~xs () in
  let expected =
    Run.v ~ys:(Array.copy ys) ~clusters:[| 0; 1 |] ~font ~size:10. ~text:"Ho"
      ~glyphs:(Array.copy glyphs) ~xs:(Array.copy xs) ()
  in
  glyphs.(0) <- 0;
  xs.(0) <- 99.;
  ys.(0) <- 99.;
  clusters.(1) <- 0;
  equal run_t expected r

let clusters_accepted =
  [
    ("an empty run", "", [||]);
    ("a ligature", "fi", [| 0 |]);
    ("a decomposition", "é", [| 0; 0 |]);
    ("one glyph per byte of ASCII", "ab", [| 0; 1 |]);
    ("glyphs for empty text", "", [| 0 |]);
  ]

let accepts (_, text, clusters) =
  let n = Array.length clusters in
  let r =
    Run.v ~clusters ~font ~size:10. ~text ~glyphs:(Array.make n 0)
      ~xs:(Array.make n 0.) ()
  in
  equal (list int) (Array.to_list clusters) (List.init n (Run.cluster r))

(* Each invalid call and a part of its message. *)
let rejected =
  let g2 = [| 1; 1 |] and x2 = [| 0.; 0. |] in
  let v ?ys ?clusters ?(size = 10.) ?(glyphs = g2) ?(xs = x2) text () =
    ignore (Run.v ?ys ?clusters ~font ~size ~text ~glyphs ~xs ())
  in
  [
    ("xs of another length", v ~xs:[| 0. |] "ab", "2 glyphs but 1 xs");
    ("ys of another length", v ~ys:[| 0. |] "ab", "2 glyphs but 1 ys");
    ( "clusters of another length",
      v ~clusters:[| 0 |] "ab",
      "2 glyphs but 1 clusters" );
    ("a negative glyph", v ~glyphs:[| -1; 1 |] "ab", "glyph -1 not in [0, 719]");
    ( "a glyph past the font",
      v ~glyphs:[| 1; 720 |] "ab",
      "glyph 720 not in [0, 719]" );
    ("a negative size", v ~size:(-1.) "ab", "invalid size");
    ("a NaN size", v ~size:Float.nan "ab", "invalid size");
    ("an infinite size", v ~size:infinity "ab", "invalid size");
    ("a NaN x", v ~xs:[| 0.; Float.nan |] "ab", "xs holds nan");
    ("an infinite y", v ~ys:[| 0.; infinity |] "ab", "ys holds inf");
    ("text that is not UTF-8", v ~clusters:[| 0; 1 |] "a\xFF", "not valid UTF-8");
    ( "no glyph for some text",
      v ~glyphs:[||] ~xs:[||] ~clusters:[||] "a",
      "no glyph renders" );
    ( "clusters not starting at 0",
      v ~clusters:[| 1; 1 |] "ab",
      "clusters start at 1" );
    ( "decreasing clusters",
      v ~clusters:[| 0; 1; 0 |] ~glyphs:[| 1; 1; 1 |] ~xs:[| 0.; 0.; 0. |] "ab",
      "clusters decrease at glyph 2" );
    ( "a cluster inside a character",
      v ~clusters:[| 0; 1 |] "é",
      "cluster 1 does not start" );
    ( "a cluster at the end of the text",
      v ~clusters:[| 0; 2 |] "ab",
      "cluster 2 does not start" );
    ( "a cluster in empty text",
      v ~glyphs:[| 1 |] ~xs:[| 0. |] ~clusters:[| 1 |] "",
      "clusters start at 1" );
    ("fewer characters than glyphs", v "a", "2 glyphs for 1 characters");
    ("more characters than glyphs", v "abc", "2 glyphs for 3 characters");
  ]

let pp_box ppf b =
  Format.fprintf ppf "[%.17g, %.17g; %.17g, %.17g]" (Box2.minx b) (Box2.miny b)
    (Box2.maxx b) (Box2.maxy b)

let box_t = Testable.make ~pp:pp_box ~equal:Box2.equal

let box_near =
  Testable.make ~pp:pp_box ~equal:(fun a b ->
      let close x y = Float.abs (x -. y) <= 1e-12 in
      close (Box2.minx a) (Box2.minx b)
      && close (Box2.miny a) (Box2.miny b)
      && close (Box2.maxx a) (Box2.maxx b)
      && close (Box2.maxy a) (Box2.maxy b))

let em units = Float.of_int units /. 2048.

(* Ink boxes of H and o read with fontTools: (180, 0, 1342, 1490) and (104, -24,
   1124, 1132) in font units, y up. *)
let bounds_of_ink () =
  let glyphs, xs = set ~size:10. ~x0:5. "Ho" in
  let r = Run.v ~ys:[| 2.; 2. |] ~font ~size:10. ~text:"Ho" ~glyphs ~xs () in
  let o = xs.(1) in
  equal (option box_near)
    (Some
       (Box2.of_pts
          (P2.v (5. +. (10. *. em 180)) (2. -. (10. *. em 1490)))
          (P2.v (o +. (10. *. em 1124)) (2. +. (10. *. em 24)))))
    (Run.bounds r)

(* Runs of inked and uninked glyphs (H, o, i, space and .notdef) at positions in
   any order, one character of text per glyph. *)
let gen_placed =
  let open Gen in
  let glyph =
    of_list
      (0
      :: List.map
           (fun c -> Font.glyph font (Uchar.of_char c))
           [ 'H'; 'o'; 'i'; ' ' ])
  in
  let coord = float_range (-1000.) 1000. in
  let+ size = of_list [ 0.; 0.5; 10.; 1000. ]
  and+ placed = list ~size:(int_range 0 6) (triple glyph coord coord) in
  let glyphs = Array.of_list (List.map (fun (g, _, _) -> g) placed) in
  let xs = Array.of_list (List.map (fun (_, x, _) -> x) placed) in
  let ys = Array.of_list (List.map (fun (_, _, y) -> y) placed) in
  let text = String.make (Array.length glyphs) 'a' in
  Run.v ~ys ~font ~size ~text ~glyphs ~xs ()

(* The definition of [Run.bounds]: the union of each glyph's ink box moved by
   the transform that sets the glyph. *)
let union_of_inks r =
  let s = Run.size r in
  List.fold_left
    (fun acc i ->
      let m = Affine.(translate (Run.x r i) (Run.y r i) * scale s s) in
      match (acc, Font.ink font (Run.glyph r i)) with
      | acc, None -> acc
      | None, Some b -> Some (Box2.transform m b)
      | Some u, Some b -> Some (Box2.union u (Box2.transform m b)))
    None
    (List.init (Run.length r) Fun.id)

let gen_run =
  let open Gen in
  let+ text = of_list [ ""; "a"; "Ho"; "fi"; "Hé→" ]
  and+ size = of_list [ 0.; 10. ]
  and+ x0 = of_list [ 0.; 1.5 ] in
  let glyphs, xs = set ~size ~x0 text in
  Run.v ~font ~size ~text ~glyphs ~xs ()

let gen_run = Gen.with_pp Run.pp gen_run

let variants () =
  let glyphs, xs = set "Ho" in
  let r = Run.v ~font ~size:10. ~text:"Ho" ~glyphs ~xs () in
  let differs msg r' = not_equal ~msg run_t r r' in
  differs "font" (Run.v ~font:Font.bold ~size:10. ~text:"Ho" ~glyphs ~xs ());
  differs "size" (Run.v ~font ~size:11. ~text:"Ho" ~glyphs ~xs ());
  differs "text" (Run.v ~font ~size:10. ~text:"Hp" ~glyphs ~xs ());
  differs "glyphs" (Run.v ~font ~size:10. ~text:"Ho" ~glyphs:[| 1; 1 |] ~xs ());
  differs "xs" (Run.v ~font ~size:10. ~text:"Ho" ~glyphs ~xs:[| 0.; 1. |] ());
  differs "ys"
    (Run.v ~ys:[| 0.; 1. |] ~font ~size:10. ~text:"Ho" ~glyphs ~xs ());
  differs "clusters"
    (Run.v ~clusters:[| 0; 0 |] ~font ~size:10. ~text:"Ho" ~glyphs ~xs ());
  differs "length"
    (Run.v ~clusters:[| 0 |] ~font ~size:10. ~text:"Ho" ~glyphs:[| 1 |]
       ~xs:[| 0. |] ());
  let reloaded =
    match Font.of_string (Font.bytes font) with
    | Ok f -> f
    | Error _ -> fail "reload"
  in
  equal ~msg:"a reloaded font" run_t r
    (Run.v ~font:reloaded ~size:10. ~text:"Ho" ~glyphs ~xs ())

let index_cases =
  [
    ("glyph", fun r i -> ignore (Run.glyph r i));
    ("x", fun r i -> ignore (Run.x r i));
    ("y", fun r i -> ignore (Run.y r i));
    ("cluster", fun r i -> ignore (Run.cluster r i));
  ]

let runs =
  group "Run"
    [
      test "v keeps what it is given" fields;
      test "v copies its arrays" copies;
      cases ~name:(fun (n, _, _) -> n) "v accepts" clusters_accepted accepts;
      cases
        ~name:(fun (n, _, _) -> n)
        "v raises on" rejected
        (fun (_, f, sub) -> raises_match (Exn.invalid_arg ~substring:sub) f);
      cases ~name:fst "raises on an index out of range in" index_cases
        (fun (_, f) ->
          let glyphs, xs = set "Ho" in
          let r = Run.v ~font ~size:10. ~text:"Ho" ~glyphs ~xs () in
          raises_match (Exn.invalid_arg ~substring:"index -1 not in [0, 1]")
            (fun () -> f r (-1));
          raises_match (Exn.invalid_arg ~substring:"index 2 not in [0, 1]")
            (fun () -> f r 2));
      test "bounds is the union of the glyphs' ink, placed and scaled"
        bounds_of_ink;
      prop "bounds is the union of the placed and scaled ink boxes"
        (Gen.with_pp Run.pp gen_placed) (fun r ->
          equal (option box_t) (union_of_inks r) (Run.bounds r));
      test "bounds is None without ink" (fun () ->
          let glyphs, xs = set "  " in
          is_none ~pp:pp_box
            (Run.bounds (Run.v ~font ~size:10. ~text:"  " ~glyphs ~xs ()));
          is_none ~pp:pp_box
            (Run.bounds
               (Run.v ~font ~size:10. ~text:"" ~glyphs:[||] ~xs:[||] ())));
      test "bounds raises on a corner past max_float" (fun () ->
          let glyphs, _ = set "H" in
          let r =
            Run.v ~font ~size:max_float ~text:"H" ~glyphs ~xs:[| max_float |] ()
          in
          raises_match (Exn.invalid_arg ~substring:"not finite") (fun () ->
              ignore (Run.bounds r)));
      test "equal compares every field, fonts by Font.equal" variants;
      prop "equal is an equivalence" (Gen.pair gen_run gen_run)
        (Law.equivalence run_t);
    ]

let () = exit (run "hugin.next.font run" [ runs ])
