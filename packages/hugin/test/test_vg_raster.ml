(*---------------------------------------------------------------------------
  Tests for the vg rasterizer: coverage, fill rules, strokes, clipping,
  transforms, stamps, text and images.
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_vg

let red = Color.v 1. 0. 0.
let blue = Color.v 0. 0. 1.

let render ?background ~w ~h p =
  Hugin_vg_raster.render ?background ~width:w ~height:h p

let px img y x c = Nx.item [ y; x; c ] img

(* Alpha of pixel [(x, y)] as a coverage in [0;1] on a transparent vg. *)
let cov img y x = float_of_int (px img y x 3) /. 255.

(* Total coverage of an image drawn on a transparent vg. *)
let total img =
  let s = ref 0. in
  let shape = Nx.shape img in
  for y = 0 to shape.(0) - 1 do
    for x = 0 to shape.(1) - 1 do
      s := !s +. cov img y x
    done
  done;
  !s

let rgba img y x = (px img y x 0, px img y x 1, px img y x 2, px img y x 3)
let color = quad int int int int

let test_dimensions () =
  let img = render ~w:7 ~h:3 Picture.empty in
  equal (array int) [| 3; 7; 4 |] (Nx.shape img);
  equal ~msg:"transparent by default" color (0, 0, 0, 0) (rgba img 1 1);
  let img = render ~background:blue ~w:2 ~h:2 Picture.empty in
  equal color (0, 0, 255, 255) (rgba img 0 0);
  raises
    (Invalid_argument
       "Hugin_vg_raster.render: width and height must be positive") (fun () ->
      render ~w:0 ~h:2 Picture.empty)

let test_full_rect () =
  let img = render ~w:4 ~h:3 (Picture.fill red (Path.rect 0. 0. 4. 3.)) in
  for y = 0 to 2 do
    for x = 0 to 3 do
      equal color (255, 0, 0, 255) (rgba img y x)
    done
  done

let test_half_pixel_edges () =
  (* A rectangle from x = 0.5 to 2.5 half covers its edge columns. *)
  let img = render ~w:4 ~h:1 (Picture.fill red (Path.rect 0.5 0. 2. 1.)) in
  equal (float 0.01) 0.5 (cov img 0 0);
  equal (float 0.01) 1. (cov img 0 1);
  equal (float 0.01) 0.5 (cov img 0 2);
  equal (float 0.01) 0. (cov img 0 3);
  (* A fractional rectangle's coverage sums to its area. *)
  let img = render ~w:10 ~h:10 (Picture.fill red (Path.rect 1.3 2.7 4.2 3.9)) in
  equal (float 0.02) (4.2 *. 3.9) (total img)

let test_circle_area () =
  let img = render ~w:40 ~h:40 (Picture.fill red (Path.circle 20. 20. 15.)) in
  (* Flattening inscribes the curve, so the area falls a little short. *)
  equal (float 8.) (Float.pi *. 15. *. 15.) (total img);
  equal ~msg:"centre is solid" color (255, 0, 0, 255) (rgba img 20 20);
  equal ~msg:"corner is empty" color (0, 0, 0, 0) (rgba img 0 0)

let test_orientation_independent () =
  let cw = Path.polygon [| 1.; 5.; 5.; 1. |] [| 1.; 1.; 5.; 5. |] in
  let ccw = Path.polygon [| 1.; 1.; 5.; 5. |] [| 1.; 5.; 5.; 1. |] in
  let a = render ~w:6 ~h:6 (Picture.fill red cw) in
  let b = render ~w:6 ~h:6 (Picture.fill red ccw) in
  equal (float 0.01) 16. (total a);
  equal (float 0.01) 16. (total b)

let test_fill_rules () =
  let outer = Path.rect 0. 0. 10. 10. and inner = Path.rect 3. 3. 4. 4. in
  let both = Path.append outer inner in
  let nz = render ~w:10 ~h:10 (Picture.fill ~rule:`Nonzero red both) in
  let eo = render ~w:10 ~h:10 (Picture.fill ~rule:`Evenodd red both) in
  equal ~msg:"nonzero fills the inner square" (float 0.01) 1. (cov nz 5 5);
  equal ~msg:"evenodd leaves a hole" (float 0.01) 0. (cov eo 5 5);
  equal ~msg:"both fill the ring" (float 0.01) 1. (cov eo 1 1)

let test_open_subpath_is_closed_for_fill () =
  let tri =
    Path.empty |> Path.move_to 0. 0. |> Path.line_to 10. 0.
    |> Path.line_to 0. 10.
  in
  let img = render ~w:10 ~h:10 (Picture.fill red tri) in
  equal (float 0.1) 50. (total img)

let test_alpha_compositing () =
  let half = Color.v ~a:0.5 1. 0. 0. in
  let img =
    render ~background:blue ~w:1 ~h:1
      (Picture.fill half (Path.rect 0. 0. 1. 1.))
  in
  let r, _, b, a = rgba img 0 0 in
  satisfies ~msg:"red halfway" int (fun v -> abs (v - 128) <= 1) r;
  satisfies ~msg:"blue halfway" int (fun v -> abs (v - 128) <= 1) b;
  equal ~msg:"opaque background stays opaque" int 255 a;
  (* On a transparent vg the output alpha is the paint's and the color is not
     premultiplied. *)
  let img = render ~w:1 ~h:1 (Picture.fill half (Path.rect 0. 0. 1. 1.)) in
  let r, _, _, a = rgba img 0 0 in
  satisfies int (fun v -> abs (v - 128) <= 1) a;
  equal ~msg:"straight alpha keeps full red" int 255 r

let test_stroke_butt () =
  let line = Path.polyline [| 2.; 8. |] [| 5.; 5. |] in
  let s = Stroke.v ~cap:`Butt 2. in
  let img = render ~w:10 ~h:10 (Picture.stroke s red line) in
  equal (float 0.01) 12. (total img);
  equal (float 0.01) 1. (cov img 4 5);
  equal (float 0.01) 1. (cov img 5 5);
  equal ~msg:"nothing beyond the ends" (float 0.01) 0. (cov img 5 1);
  equal ~msg:"nothing above" (float 0.01) 0. (cov img 3 5)

let test_stroke_caps () =
  let line = Path.polyline [| 5.; 15. |] [| 10.; 10. |] in
  let area cap =
    total (render ~w:20 ~h:20 (Picture.stroke (Stroke.v ~cap 4.) red line))
  in
  equal (float 0.05) 40. (area `Butt);
  equal (float 0.05) 56. (area `Square);
  equal (float 0.2) (40. +. (Float.pi *. 4.)) (area `Round)

let test_stroke_joins () =
  (* A right-angle corner at (10, 10), width 4. *)
  let corner = Path.polyline [| 2.; 10.; 10. |] [| 10.; 10.; 2. |] in
  let area join =
    total
      (render ~w:20 ~h:20
         (Picture.stroke (Stroke.v ~cap:`Butt ~join 4.) red corner))
  in
  (* Two 8 by 4 bodies overlapping in a 2 by 2 square plus the join. *)
  let bodies = 32. +. 32. -. 4. in
  equal ~msg:"bevel adds half the outer square" (float 0.05) (bodies +. 2.)
    (area `Bevel);
  equal ~msg:"miter adds the full outer square" (float 0.05) (bodies +. 4.)
    (area `Miter);
  equal ~msg:"round adds a quarter disc" (float 0.2) (bodies +. Float.pi)
    (area `Round);
  let sharp = Path.polyline [| 2.; 10.; 2. |] [| 12.; 10.; 8. |] in
  let miter =
    total
      (render ~w:20 ~h:20
         (Picture.stroke
            (Stroke.v ~cap:`Butt ~join:`Miter ~miter_limit:1.5 4.)
            red sharp))
  in
  let bevel =
    total
      (render ~w:20 ~h:20
         (Picture.stroke (Stroke.v ~cap:`Butt ~join:`Bevel 4.) red sharp))
  in
  equal ~msg:"a sharp miter falls back to bevel under the limit" (float 0.05)
    bevel miter

let test_stroke_dash () =
  let line = Path.polyline [| 0.; 10. |] [| 1.; 1. |] in
  let s = Stroke.v ~cap:`Butt ~dash:[| 2.; 3. |] 2. in
  let img = render ~w:10 ~h:2 (Picture.stroke s red line) in
  (* Dashes cover [0;2] and [5;7]. *)
  equal (float 0.02) 8. (total img);
  equal (float 0.01) 1. (cov img 0 1);
  equal (float 0.01) 0. (cov img 0 3);
  equal (float 0.01) 1. (cov img 0 6);
  equal (float 0.01) 0. (cov img 0 8)

let test_dots () =
  let line = Path.polyline [| 0.; 20. |] [| 5.; 5. |] in
  let s = Stroke.v ~cap:`Round ~dash:[| 0.; 10. |] 4. in
  let img = render ~w:20 ~h:10 (Picture.stroke s red line) in
  (* Dots at x = 0, 10 and 20; the first and last are half visible. *)
  equal (float 0.2) (2. *. Float.pi *. 4.) (total img);
  equal (float 0.01) 1. (cov img 5 10)

let test_zero_width_stroke_draws_nothing () =
  let line = Path.polyline [| 0.; 10. |] [| 5.; 5. |] in
  let img =
    render ~w:10 ~h:10 (Picture.stroke (Stroke.v ~cap:`Butt 0.) red line)
  in
  equal (float 0.01) 0. (total img)

let test_clip_rect () =
  let fill = Picture.fill red (Path.rect 0. 0. 10. 10.) in
  let img = render ~w:10 ~h:10 (Picture.clip (Path.rect 2. 2. 4. 4.) fill) in
  equal (float 0.01) 16. (total img);
  equal (float 0.01) 1. (cov img 3 3);
  equal (float 0.01) 0. (cov img 1 1);
  (* A fractional clip edge antialiases. *)
  let img = render ~w:10 ~h:10 (Picture.clip (Path.rect 2.5 2. 4. 4.) fill) in
  equal (float 0.05) 16. (total img);
  equal (float 0.01) 0.5 (cov img 3 2);
  equal (float 0.01) 0.5 (cov img 3 6)

let test_clip_nested_and_shaped () =
  let fill = Picture.fill red (Path.rect 0. 0. 20. 20.) in
  let disc = Path.circle 10. 10. 8. in
  let clipped = total (render ~w:20 ~h:20 (Picture.clip disc fill)) in
  equal ~msg:"a shaped clip covers what the shape fills" (float 0.05)
    (total (render ~w:20 ~h:20 (Picture.fill red disc)))
    clipped;
  let img =
    render ~w:20 ~h:20
      (Picture.clip (Path.rect 0. 0. 10. 20.) (Picture.clip disc fill))
  in
  equal ~msg:"half disc" (float 0.05) (clipped /. 2.) (total img);
  equal (float 0.01) 0. (cov img 10 15);
  equal (float 0.01) 1. (cov img 10 5)

let test_clip_masks_images_and_text () =
  let data = Nx.create Nx.uint8 [| 1; 1; 3 |] [| 255; 0; 0 |] in
  let img =
    render ~w:10 ~h:10
      (Picture.clip (Path.rect 0. 0. 5. 10.)
         (Picture.image ~x:0. ~y:0. ~w:10. ~h:10. data))
  in
  equal (float 0.01) 50. (total img)

let test_transform () =
  let sq = Picture.fill red (Path.rect 0. 0. 2. 2.) in
  let img =
    render ~w:10 ~h:10 (Picture.transform (Affine.translate 4. 6.) sq)
  in
  equal (float 0.01) 4. (total img);
  equal (float 0.01) 1. (cov img 7 5);
  let img = render ~w:10 ~h:10 (Picture.transform (Affine.scale 3. 2.) sq) in
  equal (float 0.01) 24. (total img);
  (* Stroke width follows the transform's scale. *)
  let line =
    Picture.stroke (Stroke.v ~cap:`Butt 1.) red
      (Path.polyline [| 0.; 5. |] [| 2.; 2. |])
  in
  let img = render ~w:20 ~h:20 (Picture.transform (Affine.scale 2. 2.) line) in
  equal (float 0.05) 20. (total img);
  (* Nested transforms compose. *)
  let img =
    render ~w:10 ~h:10
      (Picture.transform (Affine.translate 4. 0.)
         (Picture.transform (Affine.translate 0. 6.) sq))
  in
  equal (float 0.01) 1. (cov img 7 5)

let test_rotation () =
  let sq = Picture.fill red (Path.rect (-2.) (-2.) 4. 4.) in
  let img =
    render ~w:10 ~h:10
      (Picture.transform Affine.(translate 5. 5. * rotate (Float.pi /. 4.)) sq)
  in
  equal ~msg:"area is preserved" (float 0.05) 16. (total img);
  equal ~msg:"centre is solid" (float 0.01) 1. (cov img 5 5);
  equal ~msg:"corners of the axis box are outside" (float 0.01) 0. (cov img 2 2)

let test_stamp_matches_group () =
  let marker = Picture.fill red (Path.circle 0. 0. 2.) in
  let xs = [| 3.; 10.; 15. |] and ys = [| 4.; 10.; 3. |] in
  let stamped = render ~w:20 ~h:20 (Picture.stamp marker xs ys) in
  let grouped =
    render ~w:20 ~h:20
      (Picture.group
         (Array.to_list
            (Array.mapi
               (fun i x -> Picture.transform (Affine.translate x ys.(i)) marker)
               xs)))
  in
  for y = 0 to 19 do
    for x = 0 to 19 do
      equal color (rgba grouped y x) (rgba stamped y x)
    done
  done

let test_stamp_clips_and_transforms () =
  let marker = Picture.fill red (Path.rect (-1.) (-1.) 2. 2.) in
  let pic =
    Picture.clip (Path.rect 0. 0. 10. 10.)
      (Picture.transform (Affine.scale 2. 2.)
         (Picture.stamp marker [| 2.; 6. |] [| 2.; 2. |]))
  in
  let img = render ~w:20 ~h:20 pic in
  (* Each stamp is 4 by 4 after scaling; the second one at x = 12 is clipped. *)
  equal (float 0.01) 16. (total img);
  equal (float 0.01) 1. (cov img 4 4)

let test_text () =
  let pic = Picture.text Font.regular ~size:20. red ~x:5. ~y:25. "Hg" in
  let img = render ~w:40 ~h:40 pic in
  satisfies ~msg:"draws some ink" float_exact (fun t -> t > 20.) (total img);
  (* Ink stays within the font's bounds, which are y-down around the origin. *)
  let bx0, by0, bx1, by1 = Font.bounds Font.regular ~size:20. "Hg" in
  for y = 0 to 39 do
    for x = 0 to 39 do
      if cov img y x > 0.01 then begin
        let fx = float_of_int x and fy = float_of_int y in
        is_true ~msg:"inside horizontally"
          (fx +. 1. >= 5. +. bx0 && fx <= 5. +. bx1);
        is_true ~msg:"inside vertically"
          (fy +. 1. >= 25. +. by0 && fy <= 25. +. by1)
      end
    done
  done;
  (* Bold ink is heavier. *)
  let bold =
    render ~w:40 ~h:40 (Picture.text Font.bold ~size:20. red ~x:5. ~y:25. "Hg")
  in
  satisfies float_exact (fun t -> t > total img) (total bold)

let test_image_nearest () =
  let data =
    Nx.create Nx.uint8 [| 2; 2; 3 |]
      [| 255; 0; 0; 0; 255; 0; 0; 0; 255; 255; 255; 255 |]
  in
  let img = render ~w:4 ~h:4 (Picture.image ~x:0. ~y:0. ~w:4. ~h:4. data) in
  equal color (255, 0, 0, 255) (rgba img 0 0);
  equal color (255, 0, 0, 255) (rgba img 1 1);
  equal color (0, 255, 0, 255) (rgba img 0 3);
  equal color (0, 0, 255, 255) (rgba img 3 0);
  equal color (255, 255, 255, 255) (rgba img 3 3)

let test_image_gray_and_alpha () =
  let gray = Nx.create Nx.uint8 [| 1; 2 |] [| 0; 200 |] in
  let img = render ~w:2 ~h:1 (Picture.image ~x:0. ~y:0. ~w:2. ~h:1. gray) in
  equal color (0, 0, 0, 255) (rgba img 0 0);
  equal color (200, 200, 200, 255) (rgba img 0 1);
  let rgba_data = Nx.create Nx.uint8 [| 1; 1; 4 |] [| 255; 0; 0; 128 |] in
  let img =
    render ~background:blue ~w:1 ~h:1
      (Picture.image ~x:0. ~y:0. ~w:1. ~h:1. rgba_data)
  in
  let r, _, b, _ = rgba img 0 0 in
  satisfies int (fun v -> abs (v - 128) <= 1) r;
  satisfies int (fun v -> abs (v - 127) <= 1) b

let test_image_downscale_averages () =
  (* A checkerboard shown at a quarter of its size averages to grey. *)
  let data =
    Nx.init Nx.uint8 [| 8; 8 |] (fun i ->
        if (i.(0) + i.(1)) mod 2 = 0 then 0 else 255)
  in
  let img = render ~w:2 ~h:2 (Picture.image ~x:0. ~y:0. ~w:2. ~h:2. data) in
  let v, _, _, _ = rgba img 0 0 in
  satisfies int (fun v -> abs (v - 128) <= 2) v

let test_image_partially_off_vg () =
  let data = Nx.create Nx.uint8 [| 1; 1; 3 |] [| 255; 0; 0 |] in
  let img = render ~w:4 ~h:4 (Picture.image ~x:2. ~y:2. ~w:10. ~h:10. data) in
  equal (float 0.01) 4. (total img)

let test_image_shape_validation () =
  let bad = Nx.create Nx.uint8 [| 1; 1; 2 |] [| 0; 0 |] in
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Picture.image ~x:0. ~y:0. ~w:1. ~h:1. bad)

let test_non_finite_points_break_lines () =
  let line =
    Path.polyline [| 0.; 4.; nan; 6.; 10. |] [| 1.; 1.; 1.; 1.; 1. |]
  in
  let img =
    render ~w:10 ~h:2 (Picture.stroke (Stroke.v ~cap:`Butt 2.) red line)
  in
  equal (float 0.02) 16. (total img);
  equal (float 0.01) 0. (cov img 0 5)

let () =
  run "Vg raster"
    [
      group "vg"
        [
          test "dimensions and background" test_dimensions;
          test "alpha compositing" test_alpha_compositing;
        ];
      group "fill"
        [
          test "full rect" test_full_rect;
          test "half pixel edges" test_half_pixel_edges;
          test "circle area" test_circle_area;
          test "orientation independent" test_orientation_independent;
          test "fill rules" test_fill_rules;
          test "open subpath closed" test_open_subpath_is_closed_for_fill;
          test "non-finite points" test_non_finite_points_break_lines;
        ];
      group "stroke"
        [
          test "butt" test_stroke_butt;
          test "caps" test_stroke_caps;
          test "joins" test_stroke_joins;
          test "dash" test_stroke_dash;
          test "dots" test_dots;
          test "zero width" test_zero_width_stroke_draws_nothing;
        ];
      group "clip"
        [
          test "rect" test_clip_rect;
          test "nested and shaped" test_clip_nested_and_shaped;
          test "images" test_clip_masks_images_and_text;
        ];
      group "transform"
        [
          test "translate and scale" test_transform;
          test "rotation" test_rotation;
        ];
      group "stamp"
        [
          test "matches group" test_stamp_matches_group;
          test "clips and transforms" test_stamp_clips_and_transforms;
        ];
      group "text" [ test "ink within bounds" test_text ];
      group "image"
        [
          test "nearest" test_image_nearest;
          test "gray and alpha" test_image_gray_and_alpha;
          test "downscale averages" test_image_downscale_averages;
          test "partially off vg" test_image_partially_off_vg;
          test "shape validation" test_image_shape_validation;
        ];
    ]
