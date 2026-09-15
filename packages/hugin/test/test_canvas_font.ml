(*---------------------------------------------------------------------------
  Tests for the canvas fonts: the bundled Inter faces, their metrics, kerning
  and outlines.
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_canvas

let approx = float 1e-6
let size = 100.

let test_identity () =
  equal string "Inter" (Font.family Font.regular);
  equal int 400 (Font.weight Font.regular);
  equal string "Inter" (Font.family Font.bold);
  equal int 700 (Font.weight Font.bold)

let test_em_metrics () =
  (* Inter's hhea ascender and descender are 1984 and -494 for 2048 units. *)
  equal approx (1984. /. 2048. *. size) (Font.ascent Font.regular ~size);
  equal approx (494. /. 2048. *. size) (Font.descent Font.regular ~size)

let test_advance_scales_with_size () =
  let a = Font.advance Font.regular ~size "Hello" in
  equal approx (2. *. a) (Font.advance Font.regular ~size:(2. *. size) "Hello");
  equal ~msg:"empty text has no advance" approx 0.
    (Font.advance Font.regular ~size "")

let test_advance_is_additive_without_kerning () =
  let adv s = Font.advance Font.regular ~size s in
  (* Two glyphs Inter does not kern against each other. *)
  equal approx (adv "H" +. adv "H") (adv "HH")

let test_kerning_tightens_pairs () =
  let adv s = Font.advance Font.regular ~size s in
  satisfies ~msg:"AV is kerned" approx
    (fun w -> w < adv "A" +. adv "V")
    (adv "AV");
  satisfies ~msg:"To is kerned" approx
    (fun w -> w < adv "T" +. adv "o")
    (adv "To")

let test_bold_is_wider () =
  satisfies approx
    (fun w -> w > Font.advance Font.regular ~size "Hello")
    (Font.advance Font.bold ~size "Hello")

let test_bounds () =
  let x0, y0, x1, y1 = Font.bounds Font.regular ~size "H" in
  (* Ink sits on the baseline and rises to the cap height, about 0.73 em. *)
  equal ~msg:"bottom at the baseline" approx 0. y1;
  satisfies ~msg:"top near the cap height" approx
    (fun y -> y < -70. && y > -76.)
    y0;
  satisfies ~msg:"starts near the origin" approx
    (fun x -> x >= 0. && x < 15.)
    x0;
  satisfies ~msg:"narrower than its advance" approx
    (fun x -> x < Font.advance Font.regular ~size "H")
    x1;
  equal ~msg:"whitespace has no ink"
    (pair (pair approx approx) (pair approx approx))
    ((0., 0.), (0., 0.))
    (let a, b, c, d = Font.bounds Font.regular ~size "  " in
     ((a, b), (c, d)))

let test_bounds_descender () =
  let _, _, _, y1 = Font.bounds Font.regular ~size "g" in
  satisfies ~msg:"g descends below the baseline" approx (fun y -> y > 0.) y1

let test_glyphs_positions () =
  let gs = Font.glyphs Font.regular ~size "ab" in
  equal int 2 (List.length gs);
  let (ga, xa), (gb, xb) = (List.nth gs 0, List.nth gs 1) in
  not_equal int ga gb;
  equal approx 0. xa;
  equal approx (Font.advance Font.regular ~size "a") xb

let test_unmapped_uses_fallback () =
  (* U+1F600 is outside the bundled subset. *)
  let gs = Font.glyphs Font.regular ~size "\u{1F600}" in
  equal (list int) [ 0 ] (List.map fst gs);
  satisfies ~msg:"the fallback glyph has an advance" approx
    (fun w -> w > 0.)
    (Font.advance Font.regular ~size "\u{1F600}")

let count_segments p =
  Path.fold
    ~move:(fun n _ _ -> n + 1)
    ~line:(fun n _ _ -> n + 1)
    ~curve:(fun n _ _ _ _ _ _ -> n + 1)
    ~close:(fun n -> n + 1)
    0 p

let test_outline () =
  is_true (Path.is_empty (Font.outline Font.regular ~size " "));
  let o = Font.outline Font.regular ~size "O" in
  satisfies ~msg:"O has two contours worth of curves" int
    (fun n -> n > 8)
    (count_segments o);
  (* The outline is y-down: its ink lies above the baseline. *)
  let top =
    Path.fold
      ~move:(fun m _ y -> Float.min m y)
      ~line:(fun m _ y -> Float.min m y)
      ~curve:(fun m _ _ _ _ _ y -> Float.min m y)
      ~close:Fun.id infinity o
  in
  satisfies approx (fun y -> y < 0.) top;
  (* Accented letters are composites of a base and a mark. *)
  let e = count_segments (Font.outline Font.regular ~size "e") in
  satisfies ~msg:"é has more segments than e" int
    (fun n -> n > e)
    (count_segments (Font.outline Font.regular ~size "é"))

let test_of_string_errors () =
  equal ~msg:"garbage is malformed" bool true
    (match Font.of_string "not a font" with
    | Error (Font.Malformed _) -> true
    | _ -> false);
  equal ~msg:"CFF is unsupported" bool true
    (match Font.of_string "OTTO\000\000\000\000\000\000\000\000" with
    | Error (Font.Unsupported _) -> true
    | _ -> false)

let () =
  run "Canvas font"
    [
      test "identity" test_identity;
      test "em metrics" test_em_metrics;
      test "advance scales with size" test_advance_scales_with_size;
      test "advance additive without kerning"
        test_advance_is_additive_without_kerning;
      test "kerning tightens pairs" test_kerning_tightens_pairs;
      test "bold is wider" test_bold_is_wider;
      test "bounds" test_bounds;
      test "bounds descender" test_bounds_descender;
      test "glyph positions" test_glyphs_positions;
      test "unmapped uses fallback" test_unmapped_uses_fallback;
      test "outline" test_outline;
      test "of_string errors" test_of_string_errors;
    ]
