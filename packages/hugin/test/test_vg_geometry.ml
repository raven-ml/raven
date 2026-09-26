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

let box =
  Testable.make
    ~pp:(fun fmt (b : Box.t) ->
      Format.fprintf fmt "[%g %g %g %g]" b.x0 b.y0 b.x1 b.y1)
    ~equal:(fun (a : Box.t) b ->
      let close u v = Float.abs (u -. v) < 1e-6 in
      close a.x0 b.x0 && close a.y0 b.y0 && close a.x1 b.x1 && close a.y1 b.y1)

let test_box () =
  equal ~msg:"corners in any order" box (Box.v 0. 0. 2. 3.) (Box.v 2. 3. 0. 0.);
  equal box (Box.v 0. 0. 5. 3.)
    (Box.union (Box.v 0. 0. 2. 3.) (Box.v 4. 1. 5. 2.));
  equal ~msg:"transform encloses the rotated corners" box
    (Box.v (-1.) (-1.) 1. 1.)
    (Box.transform (Affine.rotate (Float.pi /. 2.)) (Box.v (-1.) (-1.) 1. 1.));
  equal box (Box.v 10. 20. 12. 23.)
    (Box.transform Affine.(translate 10. 20. * scale 2. 3.) (Box.v 0. 0. 1. 1.))

let test_flatten () =
  let p = Path.circle 0. 0. 10. in
  let n, closes =
    Path.flatten Affine.id
      ~move:(fun (n, c) _ _ -> (n + 1, c))
      ~line:(fun (n, c) _ _ -> (n + 1, c))
      ~close:(fun (n, c) -> (n, c + 1))
      (0, 0) p
  in
  satisfies ~msg:"a circle becomes many chords" int (fun n -> n > 16) n;
  equal int 1 closes;
  (* Every chord endpoint lies on the circle within the tolerance. *)
  Path.flatten Affine.id
    ~move:(fun () _ _ -> ())
    ~line:(fun () x y ->
      equal ~msg:"on the circle" (float 0.15) 10. (Float.hypot x y))
    ~close:Fun.id () p;
  let coarse =
    Path.flatten ~tolerance:5. Affine.id
      ~move:(fun n _ _ -> n + 1)
      ~line:(fun n _ _ -> n + 1)
      ~close:Fun.id 0 p
  in
  satisfies ~msg:"a loose tolerance gives fewer chords" int
    (fun c -> c < n)
    coarse;
  (* A non-finite point ends the subpath. *)
  let broken = Path.polyline [| 0.; 1.; nan; 3. |] [| 0.; 0.; 0.; 0. |] in
  equal ~msg:"two subpaths" int 2
    (Path.flatten Affine.id
       ~move:(fun n _ _ -> n + 1)
       ~line:(fun n _ _ -> n)
       ~close:Fun.id 0 broken)

let test_path_bounds () =
  is_none (Path.bounds Path.empty);
  equal (option box)
    (Some (Box.v 1. 2. 4. 6.))
    (Path.bounds (Path.rect 1. 2. 3. 4.));
  let b = Option.get (Path.bounds (Path.circle 5. 5. 2.)) in
  equal ~msg:"circle bounds" (float 0.02) 3. b.x0;
  equal (float 0.02) 7. b.x1;
  equal (float 0.02) 3. b.y0;
  equal (float 0.02) 7. b.y1

let test_pp () =
  let show p =
    let b = Buffer.create 256 in
    let fmt = Format.formatter_of_buffer b in
    Format.pp_set_margin fmt 200;
    Format.fprintf fmt "%a@?" Picture.pp p;
    Buffer.contents b
  in
  equal string "empty" (show Picture.empty);
  equal string "(fill rgb(1 0 0) \"M 0 0 L 2 0 L 2 1 L 0 1 Z\")"
    (show (Picture.fill (Color.v 1. 0. 0.) (Path.rect 0. 0. 2. 1.)));
  equal string
    "(stroke width:2 cap:butt join:miter miter-limit:4 dash:[3 1] rgba(0 0 1 \
     0.5) \"M 0 0 L 5 5\")"
    (show
       (Picture.stroke
          (Stroke.v ~cap:`Butt ~join:`Miter ~dash:[| 3.; 1. |] 2.)
          (Color.v ~a:0.5 0. 0. 1.)
          (Path.polyline [| 0.; 5. |] [| 0.; 5. |])));
  equal string "(text \"Inter\" 700 12 rgb(0 0 0) 1 2 \"a\\\"b\")"
    (show (Picture.text Font.bold ~size:12. Color.black ~x:1. ~y:2. "a\"b"));
  equal string "(image 0 0 4 4 [2 2 3])"
    (show
       (Picture.image ~x:0. ~y:0. ~w:4. ~h:4. (Nx.zeros Nx.uint8 [| 2; 2; 3 |])));
  let sq = Picture.fill Color.black (Path.rect 0. 0. 1. 1.) in
  equal string
    "(clip \"M 0 0 L 9 0 L 9 9 L 0 9 Z\" (transform [2 0 0 2 0 0] (group (fill \
     rgb(0 0 0) \"M 0 0 L 1 0 L 1 1 L 0 1 Z\") (stamp 2 (fill rgb(0 0 0) \"M 0 \
     0 L 1 0 L 1 1 L 0 1 Z\") 3 4 5 6))))"
    (show
       (Picture.clip (Path.rect 0. 0. 9. 9.)
          (Picture.transform (Affine.scale 2. 2.)
             (Picture.group [ sq; Picture.stamp sq [| 3.; 5. |] [| 4.; 6. |] ]))))

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
  exit
    (run "Vg geometry"
       [
         group "affine"
           [
             test "composition order" test_apply_composition;
             test "quarter turn" test_rotate_quarter_turn;
             test "invert round trip" test_invert_roundtrip;
             test "invert singular" test_invert_singular;
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
         group "box" [ test "operations" test_box ];
         group "flatten"
           [ test "chords" test_flatten; test "bounds" test_path_bounds ];
         group "printing" [ test "pictures" test_pp ];
         group "stroke"
           [
             test "defaults" test_stroke_defaults;
             test "invariants" test_stroke_invariants;
           ];
       ])
