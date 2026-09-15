(*---------------------------------------------------------------------------
  Tests for the vg geometry: affine transforms, paths and stroke styles.
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_vg

let point = pair (float 1e-9) (float 1e-9)

(* Affine *)

let test_apply_composition () =
  let m = Affine.(translate 10. 20. * scale 2. 3.) in
  equal ~msg:"scale then translate" point (12., 26.) (Affine.apply m 1. 2.);
  let m = Affine.(scale 2. 3. * translate 10. 20.) in
  equal ~msg:"translate then scale" point (22., 66.) (Affine.apply m 1. 2.)

let test_rotate_quarter_turn () =
  let m = Affine.rotate (Float.pi /. 2.) in
  equal point (0., 1.) (Affine.apply m 1. 0.)

let test_invert_roundtrip () =
  let m = Affine.(translate 5. (-3.) * rotate 0.7 * scale 2. 0.5) in
  let x, y = Affine.apply m 4. 9. in
  equal point (4., 9.) (Affine.apply (Affine.invert m) x y)

let test_invert_singular () =
  raises (Invalid_argument "Affine.invert: singular transform") (fun () ->
      Affine.invert (Affine.scale 0. 1.))

let test_is_translation () =
  is_true (Affine.is_translation (Affine.translate 1. 2.));
  is_false (Affine.is_translation (Affine.scale 2. 2.))

(* Paths *)

type seg =
  | M of float * float
  | L of float * float
  | C of float * float * float * float * float * float
  | Z

let segments p =
  Path.fold
    ~move:(fun acc x y -> M (x, y) :: acc)
    ~line:(fun acc x y -> L (x, y) :: acc)
    ~curve:(fun acc a b c d e f -> C (a, b, c, d, e, f) :: acc)
    ~close:(fun acc -> Z :: acc)
    [] p
  |> List.rev

let seg_testable =
  Testable.make
    ~pp:(fun fmt -> function
      | M (x, y) -> Format.fprintf fmt "M %g %g" x y
      | L (x, y) -> Format.fprintf fmt "L %g %g" x y
      | C (a, b, c, d, e, f) ->
          Format.fprintf fmt "C %g %g %g %g %g %g" a b c d e f
      | Z -> Format.fprintf fmt "Z")
    ~equal:( = )

let segs = list seg_testable

let test_building () =
  let p =
    Path.empty |> Path.move_to 0. 0. |> Path.line_to 1. 0.
    |> Path.curve_to 1. 1. 0. 1. 0. 0.
    |> Path.close
  in
  equal segs
    [ M (0., 0.); L (1., 0.); C (1., 1., 0., 1., 0., 0.); Z ]
    (segments p)

let test_segment_without_current_point_moves () =
  equal segs [ M (3., 4.) ] (segments (Path.line_to 3. 4. Path.empty));
  let p =
    Path.empty |> Path.move_to 0. 0. |> Path.line_to 1. 1. |> Path.close
  in
  equal segs
    [ M (0., 0.); L (1., 1.); Z; M (5., 5.) ]
    (segments (Path.line_to 5. 5. p))

let test_close_is_idempotent () =
  let p = Path.empty |> Path.move_to 0. 0. |> Path.line_to 1. 1. in
  equal segs
    [ M (0., 0.); L (1., 1.); Z ]
    (segments (p |> Path.close |> Path.close));
  equal ~msg:"closing a lone move does nothing" segs
    [ M (0., 0.) ]
    (segments (Path.close (Path.move_to 0. 0. Path.empty)));
  is_true (Path.is_empty (Path.close Path.empty))

let test_polyline () =
  let p = Path.polyline [| 0.; 1.; 2. |] [| 0.; 1.; 0. |] in
  equal segs [ M (0., 0.); L (1., 1.); L (2., 0.) ] (segments p);
  equal ~msg:"polygon closes" segs
    [ M (0., 0.); L (1., 1.); L (2., 0.); Z ]
    (segments (Path.polygon [| 0.; 1.; 2. |] [| 0.; 1.; 0. |]));
  equal ~msg:"close after polyline" segs
    [ M (0., 0.); L (1., 1.); L (2., 0.); Z ]
    (segments (Path.close p));
  is_true (Path.is_empty (Path.polyline [| 1. |] [| 1. |]));
  raises (Invalid_argument "Path.polyline: xs and ys differ in length")
    (fun () -> Path.polyline [| 0.; 1. |] [| 0. |])

let test_append_order () =
  let p = Path.rect 0. 0. 1. 1. and q = Path.move_to 5. 5. Path.empty in
  equal segs
    [ M (0., 0.); L (1., 0.); L (1., 1.); L (0., 1.); Z; M (5., 5.) ]
    (segments (Path.append p q))

let test_transform () =
  let p = Path.polyline [| 0.; 1. |] [| 0.; 1. |] in
  equal segs
    [ M (10., 20.); L (12., 23.) ]
    (segments (Path.transform Affine.(translate 10. 20. * scale 2. 3.) p));
  let c = Path.transform (Affine.translate 1. 1.) (Path.circle 0. 0. 1.) in
  equal ~msg:"circle starts at the rightmost point" segs
    [ M (2., 1.) ]
    (List.filteri (fun i _ -> i = 0) (segments c))

let test_circle_endpoints () =
  (* Each arc lands on an axis point of the circle. *)
  let ends =
    List.filter_map
      (function C (_, _, _, _, x, y) -> Some (x, y) | _ -> None)
      (segments (Path.circle 3. 4. 2.))
  in
  equal (list point) [ (3., 6.); (1., 4.); (3., 2.); (5., 4.) ] ends

(* Stroke *)

let test_stroke_defaults () =
  let s = Stroke.v 2. in
  equal float_exact 2. s.width;
  is_true (s.cap = `Round);
  is_true (s.join = `Round);
  equal (array float_exact) [||] s.dash;
  equal float_exact 4. s.miter_limit

let test_stroke_invariants () =
  raises (Invalid_argument "Stroke.v: negative width") (fun () ->
      Stroke.v (-1.));
  raises (Invalid_argument "Stroke.v: miter limit below 1") (fun () ->
      Stroke.v ~miter_limit:0.5 1.);
  raises (Invalid_argument "Stroke.v: negative dash length") (fun () ->
      Stroke.v ~dash:[| 1.; -1. |] 1.);
  raises (Invalid_argument "Stroke.v: dash pattern sums to zero") (fun () ->
      Stroke.v ~dash:[| 0.; 0. |] 1.);
  let dotted = Stroke.v ~dash:[| 0.; 4. |] 2. in
  equal ~msg:"zero-length dashes are dots" (array float_exact) [| 0.; 4. |]
    dotted.dash

let () =
  run "Vg geometry"
    [
      group "affine"
        [
          test "composition order" test_apply_composition;
          test "quarter turn" test_rotate_quarter_turn;
          test "invert round trip" test_invert_roundtrip;
          test "invert singular" test_invert_singular;
          test "is_translation" test_is_translation;
        ];
      group "path"
        [
          test "building" test_building;
          test "segment without current point"
            test_segment_without_current_point_moves;
          test "close is idempotent" test_close_is_idempotent;
          test "polyline" test_polyline;
          test "append order" test_append_order;
          test "transform" test_transform;
          test "circle endpoints" test_circle_endpoints;
        ];
      group "stroke"
        [
          test "defaults" test_stroke_defaults;
          test "invariants" test_stroke_invariants;
        ];
    ]
