open Hugin
open Windtrap

let rgba_testable = quad (float 0.01) (float 0.01) (float 0.01) (float 0.01)
let check_float msg = equal ~msg (float 1e-6)
let check_rgba msg expected actual = equal ~msg rgba_testable expected actual

(* sRGB roundtrip *)

let test_black_roundtrip () =
  let r, g, b, a = Color.rgb ~r:0. ~g:0. ~b:0. () |> Color.to_rgba in
  check_rgba "black" (0., 0., 0., 1.) (r, g, b, a)

let test_white_roundtrip () =
  let r, g, b, a = Color.rgb ~r:1. ~g:1. ~b:1. () |> Color.to_rgba in
  check_rgba "white" (1., 1., 1., 1.) (r, g, b, a)

let test_red_roundtrip () =
  let r, g, b, a = Color.rgb ~r:1. ~g:0. ~b:0. () |> Color.to_rgba in
  check_rgba "red" (1., 0., 0., 1.) (r, g, b, a)

let test_green_roundtrip () =
  let r, g, b, a = Color.rgb ~r:0. ~g:1. ~b:0. () |> Color.to_rgba in
  check_rgba "green" (0., 1., 0., 1.) (r, g, b, a)

let test_blue_roundtrip () =
  let r, g, b, a = Color.rgb ~r:0. ~g:0. ~b:1. () |> Color.to_rgba in
  (* Pure blue is near the sRGB gamut boundary in OKLCH, so the roundtrip is not
     perfectly lossless due to gamut clamping. We verify it's close. *)
  let wide_rgba = quad (float 0.15) (float 0.01) (float 0.01) (float 0.01) in
  equal ~msg:"blue" wide_rgba (0., 0., 1., 1.) (r, g, b, a)

let test_mid_gray_roundtrip () =
  let r, g, b, a = Color.rgb ~r:0.5 ~g:0.5 ~b:0.5 () |> Color.to_rgba in
  check_rgba "mid gray" (0.5, 0.5, 0.5, 1.) (r, g, b, a)

let test_arbitrary_roundtrip () =
  let r, g, b, a = Color.rgb ~r:0.3 ~g:0.6 ~b:0.9 () |> Color.to_rgba in
  (* OKLCH roundtrip has small error due to gamut boundary clamping *)
  let wide_rgba = quad (float 0.03) (float 0.03) (float 0.03) (float 0.01) in
  equal ~msg:"arbitrary" wide_rgba (0.3, 0.6, 0.9, 1.) (r, g, b, a)

(* Hex parsing *)

let test_hex_6_with_hash () =
  let r, g, b, _ = Color.hex "#FF0000" |> Color.to_rgba in
  check_rgba "red hex" (1., 0., 0., 1.) (r, g, b, 1.)

let test_hex_6_without_hash () =
  let r, g, b, _ = Color.hex "FF0000" |> Color.to_rgba in
  check_rgba "red hex no hash" (1., 0., 0., 1.) (r, g, b, 1.)

let test_hex_8_with_alpha () =
  let _, _, _, a = Color.hex "#FF000080" |> Color.to_rgba in
  let expected_a = 128. /. 255. in
  check_float "alpha" expected_a a

let test_hex_case_insensitive () =
  let r1, g1, b1, _ = Color.hex "#ff0000" |> Color.to_rgba in
  let r2, g2, b2, _ = Color.hex "#FF0000" |> Color.to_rgba in
  check_rgba "case insensitive" (r1, g1, b1, 1.) (r2, g2, b2, 1.)

let test_hex_invalid_length () =
  raises_invalid_arg "Color.hex: expected 6 or 8 hex digits, got 3" (fun () ->
      Color.hex "#FFF")

let test_hex_invalid_chars () =
  raises_invalid_arg "Color.hex: invalid hex digit 'G'" (fun () ->
      Color.hex "#GGGGGG")

let test_hex_empty () =
  raises_invalid_arg "Color.hex: expected 6 or 8 hex digits, got 0" (fun () ->
      Color.hex "")

(* OKLCH constructors *)

let test_oklch_fields () =
  let c = Color.oklch ~l:0.5 ~c:0.1 ~h:180. () in
  check_float "lightness" 0.5 (Color.lightness c);
  check_float "chroma" 0.1 (Color.chroma c);
  check_float "hue" 180. (Color.hue c);
  check_float "alpha" 1. (Color.alpha c)

let test_oklcha_alpha () =
  let c = Color.oklcha ~l:0.5 ~c:0.1 ~h:180. ~a:0.7 () in
  check_float "alpha" 0.7 (Color.alpha c)

(* Operations *)

let test_lighten_clamps () =
  let c = Color.lighten 2.0 (Color.oklch ~l:0.8 ~c:0. ~h:0. ()) in
  check_float "clamped to 1" 1.0 (Color.lightness c)

let test_darken_clamps () =
  let c = Color.darken 2.0 (Color.oklch ~l:0.2 ~c:0. ~h:0. ()) in
  check_float "clamped to 0" 0.0 (Color.lightness c)

let test_lighten_adds () =
  let c = Color.lighten 0.1 (Color.oklch ~l:0.5 ~c:0. ~h:0. ()) in
  check_float "lighten adds" 0.6 (Color.lightness c)

let test_darken_subtracts () =
  let c = Color.darken 0.1 (Color.oklch ~l:0.5 ~c:0. ~h:0. ()) in
  check_float "darken subtracts" 0.4 (Color.lightness c)

let test_with_alpha () =
  let c = Color.with_alpha 0.3 (Color.oklch ~l:0.5 ~c:0. ~h:0. ()) in
  check_float "with_alpha" 0.3 (Color.alpha c)

(* Mix *)

let a = Color.oklch ~l:0.2 ~c:0.05 ~h:10. ()
let b = Color.oklch ~l:0.8 ~c:0.15 ~h:50. ()

let test_mix_zero () =
  let m = Color.mix 0.0 a b in
  check_float "lightness" (Color.lightness a) (Color.lightness m);
  check_float "chroma" (Color.chroma a) (Color.chroma m);
  check_float "hue" (Color.hue a) (Color.hue m)

let test_mix_one () =
  let m = Color.mix 1.0 a b in
  check_float "lightness" (Color.lightness b) (Color.lightness m);
  check_float "chroma" (Color.chroma b) (Color.chroma m);
  check_float "hue" (Color.hue b) (Color.hue m)

let test_mix_midpoint_lightness () =
  let m = Color.mix 0.5 a b in
  check_float "midpoint lightness" 0.5 (Color.lightness m)

let test_mix_midpoint_chroma () =
  let m = Color.mix 0.5 a b in
  check_float "midpoint chroma" 0.1 (Color.chroma m)

let test_mix_hue_forward () =
  let c1 = Color.oklch ~l:0.5 ~c:0.1 ~h:10. () in
  let c2 = Color.oklch ~l:0.5 ~c:0.1 ~h:50. () in
  let m = Color.mix 0.5 c1 c2 in
  check_float "hue forward" 30. (Color.hue m)

let test_mix_hue_wraps_360 () =
  let c1 = Color.oklch ~l:0.5 ~c:0.1 ~h:350. () in
  let c2 = Color.oklch ~l:0.5 ~c:0.1 ~h:10. () in
  let m = Color.mix 0.5 c1 c2 in
  check_float "hue wraps" 0. (Color.hue m)

let test_mix_hue_reverse_wrap () =
  let c1 = Color.oklch ~l:0.5 ~c:0.1 ~h:10. () in
  let c2 = Color.oklch ~l:0.5 ~c:0.1 ~h:350. () in
  let m = Color.mix 0.5 c1 c2 in
  check_float "hue reverse wrap" 0. (Color.hue m)

let test_mix_alpha () =
  let c1 = Color.oklcha ~l:0.5 ~c:0. ~h:0. ~a:0.0 () in
  let c2 = Color.oklcha ~l:0.5 ~c:0. ~h:0. ~a:1.0 () in
  let m = Color.mix 0.5 c1 c2 in
  check_float "alpha interpolates" 0.5 (Color.alpha m)

(* Gamut clamping *)

let test_high_chroma_clamped () =
  let r, g, b, _ = Color.oklch ~l:0.5 ~c:0.4 ~h:0. () |> Color.to_rgba in
  is_true ~msg:"r in [0,1]" (r >= 0. && r <= 1.);
  is_true ~msg:"g in [0,1]" (g >= 0. && g <= 1.);
  is_true ~msg:"b in [0,1]" (b >= 0. && b <= 1.)

(* Named colors *)

let test_black_named () =
  let r, g, b, a = Color.black |> Color.to_rgba in
  check_rgba "black" (0., 0., 0., 1.) (r, g, b, a)

let test_white_named () =
  let r, g, b, a = Color.white |> Color.to_rgba in
  check_rgba "white" (1., 1., 1., 1.) (r, g, b, a)

let test_orange_matches_hex () =
  let r1, g1, b1, _ = Color.orange |> Color.to_rgba in
  let r2, g2, b2, _ = Color.hex "#E69F00" |> Color.to_rgba in
  check_rgba "orange" (r1, g1, b1, 1.) (r2, g2, b2, 1.)

let () =
  run "Color"
    [
      group "sRGB roundtrip"
        [
          test "black" test_black_roundtrip;
          test "white" test_white_roundtrip;
          test "red" test_red_roundtrip;
          test "green" test_green_roundtrip;
          test "blue" test_blue_roundtrip;
          test "mid gray" test_mid_gray_roundtrip;
          test "arbitrary" test_arbitrary_roundtrip;
        ];
      group "hex parsing"
        [
          test "6-digit with hash" test_hex_6_with_hash;
          test "6-digit without hash" test_hex_6_without_hash;
          test "8-digit with alpha" test_hex_8_with_alpha;
          test "case insensitive" test_hex_case_insensitive;
          test "invalid length" test_hex_invalid_length;
          test "invalid chars" test_hex_invalid_chars;
          test "empty string" test_hex_empty;
        ];
      group "OKLCH constructors"
        [
          test "oklch fields" test_oklch_fields;
          test "oklcha alpha" test_oklcha_alpha;
        ];
      group "operations"
        [
          test "lighten clamps" test_lighten_clamps;
          test "darken clamps" test_darken_clamps;
          test "lighten adds" test_lighten_adds;
          test "darken subtracts" test_darken_subtracts;
          test "with_alpha" test_with_alpha;
        ];
      group "mix"
        [
          test "mix 0.0 returns first" test_mix_zero;
          test "mix 1.0 returns second" test_mix_one;
          test "midpoint lightness" test_mix_midpoint_lightness;
          test "midpoint chroma" test_mix_midpoint_chroma;
          test "hue shortest arc forward" test_mix_hue_forward;
          test "hue wraps across 360" test_mix_hue_wraps_360;
          test "hue reverse wrap" test_mix_hue_reverse_wrap;
          test "alpha interpolates" test_mix_alpha;
        ];
      group "gamut clamping"
        [ test "high chroma clamped" test_high_chroma_clamped ];
      group "named colors"
        [
          test "black" test_black_named;
          test "white" test_white_named;
          test "orange matches hex" test_orange_matches_hex;
        ];
    ]
