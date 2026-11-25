(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_kit

let symbol = Testable.make ~pp:Symbol.pp ~equal:Symbol.equal
let name s = Format.asprintf "%a" Symbol.pp s
let path = Testable.make ~pp:Path.pp ~equal:Path.equal

let all =
  Symbol.
    [
      circle; square; diamond; triangle; cross; star; wye; plus; times; asterisk;
    ]

let open_ = Symbol.[ plus; times; asterisk ]
let closed = List.filter (fun s -> not (List.memq s open_)) all
let paints = [ `Fill; `Stroke ]
let paint_name = function `Fill -> "`Fill" | `Stroke -> "`Stroke"
let label s paint = name s ^ " " ^ paint_name paint

(* Tolerances. A vertex of size 1 comes from a few rounded operations on its
   constants (square roots, sines, a division) and scaling it adds one or two
   more, so each coordinate is within [16 ε R] of the stated point, [R] the
   symbol's largest radius. Angles reach [2π], whose relative error [2 ε] moves
   a point by [4π ε R] more; [64 ε R] holds both, on each side of a
   comparison. *)
let eps = epsilon_float
let rounding = 64. *. eps
let rel e x = float (e *. Float.abs x)

(* The circle's arcs depart from it by at most 0.03% of its radius, so its area
   is within [2 × 0.03%] and its length within [0.03%] of the true circle's;
   chords within [1e-7 √a] of the arcs add less than [1e-6]. *)
let circle_eps = 1e-3

(* Points and subpaths *)

let pt_pp ppf (x, y) = Format.fprintf ppf "(%h, %h)" x y
let near_pt tol = pair (float tol) (float tol)

(* A point at the origin: both coordinates zero, of either sign. *)
let origin = Testable.make ~pp:pt_pp ~equal:( = )

(* The points [Path.fold] visits, control points included. *)
let points p =
  Path.fold
    ~move:(fun acc x y -> (x, y) :: acc)
    ~line:(fun acc x y -> (x, y) :: acc)
    ~cubic:(fun acc a b c d e f -> (e, f) :: (c, d) :: (a, b) :: acc)
    ~close:Fun.id [] p
  |> List.rev

let radius pts =
  List.fold_left (fun r (x, y) -> Float.max r (Float.hypot x y)) 0. pts

(* Subpaths as flattened by [Path.flatten]: their points and whether they are
   closed. *)
type subpath = { pts : (float * float) list; closed : bool }

let subpath =
  Testable.make
    ~pp:(fun ppf sp ->
      Format.fprintf ppf "%s [%a]"
        (if sp.closed then "closed" else "open")
        (Format.pp_print_list ~pp_sep:Format.pp_print_space pt_pp)
        sp.pts)
    ~equal:( = )

let subpaths ?(tolerance = 1e-12) p =
  let finish acc = function
    | Some pts -> { pts = List.rev pts; closed = false } :: acc
    | None -> acc
  in
  let acc, cur =
    Path.flatten ~tolerance Affine.id
      ~move:(fun (acc, cur) x y -> (finish acc cur, Some [ (x, y) ]))
      ~line:(fun (acc, cur) x y ->
        match cur with
        | Some pts -> (acc, Some ((x, y) :: pts))
        | None -> (acc, Some [ (x, y) ]))
      ~close:(fun (acc, cur) ->
        match cur with
        | Some pts -> ({ pts = List.rev pts; closed = true } :: acc, None)
        | None -> (acc, None))
      ([], None) p
  in
  List.rev (finish acc cur)

(* [flat paint a s] is [Symbol.path paint a s] flattened, the circle's arcs cut
   into chords within [1e-7] of the root of its size. *)
let flat paint a s =
  let tolerance = 1e-7 *. Float.max (Float.sqrt a) Float.min_float in
  subpaths ~tolerance (Symbol.path paint a s)

let vertices sps = List.concat_map (fun sp -> sp.pts) sps

(* The edges of a subpath, the closing one included. *)
let edges { pts; closed } =
  let rec go acc = function
    | a :: (b :: _ as rest) -> go ((a, b) :: acc) rest
    | [ last ] when closed -> List.rev ((last, List.hd pts) :: acc)
    | _ -> List.rev acc
  in
  go [] pts

let all_edges sps = List.concat_map edges sps
let edge_length ((x0, y0), (x1, y1)) = Float.hypot (x1 -. x0) (y1 -. y0)

let length sps =
  List.fold_left (fun l e -> l +. edge_length e) 0. (all_edges sps)

(* Shoelace sums with y pointing down: an outline clockwise on screen has a
   positive area. *)
let cross_ ((x0, y0), (x1, y1)) = (x0 *. y1) -. (x1 *. y0)

let area sps =
  List.fold_left (fun a e -> a +. (cross_ e /. 2.)) 0. (all_edges sps)

let region_centroid sps =
  let a = area sps in
  let cx, cy =
    List.fold_left
      (fun (cx, cy) (((x0, y0), (x1, y1)) as e) ->
        let c = cross_ e in
        (cx +. ((x0 +. x1) *. c), cy +. ((y0 +. y1) *. c)))
      (0., 0.) (all_edges sps)
  in
  (cx /. (6. *. a), cy /. (6. *. a))

let outline_centroid sps =
  let l = length sps in
  let cx, cy =
    List.fold_left
      (fun (cx, cy) (((x0, y0), (x1, y1)) as e) ->
        let d = edge_length e in
        (cx +. (d *. (x0 +. x1) /. 2.), cy +. (d *. (y0 +. y1) /. 2.)))
      (0., 0.) (all_edges sps)
  in
  (cx /. l, cy /. l)

(* The outline of a closed symbol of size 1. *)
let outline s =
  match subpaths (Symbol.path `Fill 1. s) with
  | [ sp ] -> sp
  | sps -> failf "%d subpaths" (List.length sps)

(* The interior angles of a clockwise outline in degrees, the [i]th at its [i]th
   vertex. *)
let angles sp =
  let v = Array.of_list sp.pts in
  let n = Array.length v in
  List.init n (fun i ->
      let ax, ay = v.((i + n - 1) mod n)
      and bx, by = v.(i)
      and cx, cy = v.((i + 1) mod n) in
      let ux = bx -. ax and uy = by -. ay and wx = cx -. bx and wy = cy -. by in
      let turn =
        Float.atan2 ((ux *. wy) -. (uy *. wx)) ((ux *. wx) +. (uy *. wy))
      in
      180. -. (turn *. 180. /. Float.pi))

(* [from a l] is the cycle [l] turned to start at its first angle near [a], so
   that no vertex of the outline is taken as its first. *)
let from a l =
  let rec go before = function
    | x :: rest when Float.abs (x -. a) < 1. -> (x :: rest) @ List.rev before
    | x :: rest -> go (x :: before) rest
    | [] -> failf "no angle of %g degrees" a
  in
  go [] l

(* Vertices within [16 ε R] move the direction of an edge of at least [R / 4] by
   [128 ε] radians, so an angle by twice that, here in degrees. *)
let angle = list (float (256. *. eps *. 180. /. Float.pi))

let equal_edges sp =
  let ls = List.map edge_length (edges sp) in
  let side = List.hd ls in
  equal (list (rel (128. *. eps) side)) (List.map (fun _ -> side) ls) ls

let edge = pair (pair float_exact float_exact) (pair float_exact float_exact)

let parallel_to_an_axis =
  satisfies ~claim:"an edge parallel to an axis" edge
    (fun ((x0, y0), (x1, y1)) -> x0 = x1 || y0 = y1)

(* The vertices sorted from the top of the screen down. *)
let top_down sp = List.sort (fun (_, y) (_, y') -> Float.compare y y') sp.pts

let rotate t (x, y) =
  ( (x *. Float.cos t) -. (y *. Float.sin t),
    (x *. Float.sin t) +. (y *. Float.cos t) )

(* Generators *)

(* Sizes whose outlines' coordinates multiply without underflow; the examples
   reach the ends of the float range. *)
let gen_size = Gen.float_range 1e-6 1e3
let size_examples = [ 1e-200; 1e-12; 1.; 1e12; 1e200 ]

(* Sizes *)

let ink a = 2. *. Float.sqrt (Float.pi *. a)
let tolerance s = if Symbol.equal s Symbol.circle then circle_eps else rounding

let filled_area a =
  List.iter
    (fun s ->
      equal ~msg:(name s) (rel (tolerance s) a) a (area (flat `Fill a s)))
    closed

let stroked_length a =
  List.iter
    (fun s ->
      equal ~msg:(name s)
        (rel (tolerance s) (ink a))
        (ink a)
        (length (flat `Stroke a s)))
    closed

let open_length a =
  List.iter
    (fun s ->
      List.iter
        (fun paint ->
          equal ~msg:(label s paint)
            (rel rounding (ink a))
            (ink a)
            (length (flat paint a s)))
        paints)
    open_

(* [Symbol.path paint a s] multiplies its constants by [√a] where the reference
   multiplies them by [1.] then [√a]: a few roundings of each number apart. A
   number near zero, as a control point at a quarter turn, is a difference of
   terms as large as [√a] times the size-1 radius, which is below [2]. *)
let scaling a =
  let k = Float.sqrt a in
  let near u v =
    Float.abs (u -. v) <= 16. *. eps *. ((2. *. k) +. Float.abs u)
  in
  let pt =
    Testable.make ~pp:pt_pp ~equal:(fun (u, v) (u', v') ->
        near u u' && near v v')
  in
  List.iter
    (fun s ->
      List.iter
        (fun paint ->
          equal ~msg:(label s paint) (list pt)
            (points
               (Path.transform (Affine.scale k k) (Symbol.path paint 1. s)))
            (points (Symbol.path paint a s)))
        paints)
    all

let sizes =
  group "sizes"
    [
      prop "filled, a closed symbol encloses its size" ~examples:size_examples
        gen_size filled_area;
      prop "stroked, a closed symbol's outline is as long as the circle's"
        ~examples:size_examples gen_size stroked_length;
      prop "open symbols are as long as the circle's outline however painted"
        ~examples:size_examples gen_size open_length;
      prop "a symbol of size a is the symbol of size 1 scaled by its root"
        ~examples:size_examples gen_size scaling;
    ]

(* Symmetry *)

(* The order of each symbol's rotational symmetry, read from its shape. The
   circle is [Path.circle], checked under symbols. *)
let order s =
  Symbol.(
    List.assq s
      [
        (square, 4);
        (diamond, 2);
        (triangle, 3);
        (cross, 4);
        (star, 5);
        (wye, 3);
        (plus, 4);
        (times, 4);
        (asterisk, 6);
      ])

let symmetric a =
  List.iter
    (fun s ->
      List.iter
        (fun paint ->
          let pts = points (Symbol.path paint a s) in
          let turn = 2. *. Float.pi /. Float.of_int (order s) in
          let near = near_pt (rounding *. radius pts) in
          List.iter
            (fun p -> mem ~msg:(label s paint) near (rotate turn p) pts)
            pts)
        paints)
    (List.filter (fun s -> not (Symbol.equal s Symbol.circle)) all)

(* A centroid is a sum over [n] edges, each term rounded a few times and at most
   [R] times its weight, divided by the total weight; with vertices within [16 ε
   R], it lies within [(n + 64) ε R] of the exact one. *)
let centroids a =
  List.iter
    (fun s ->
      List.iter
        (fun paint ->
          let sps = flat paint a s in
          let n = Float.of_int (List.length (all_edges sps)) in
          let near = near_pt ((n +. 64.) *. eps *. radius (vertices sps)) in
          equal
            ~msg:("outline of " ^ label s paint)
            near (0., 0.) (outline_centroid sps);
          if not (List.memq s open_) then
            equal
              ~msg:("region of " ^ label s paint)
              near (0., 0.) (region_centroid sps))
        paints)
    all

let symmetry =
  group "symmetry"
    [
      prop "each symbol is symmetric under its rotation about the origin"
        ~examples:size_examples gen_size symmetric;
      prop "centroids of region and outline lie at the origin"
        ~examples:size_examples gen_size centroids;
    ]

(* Symbols *)

let square_sides () =
  let sp = outline Symbol.square in
  equal angle [ 90.; 90.; 90.; 90. ] (angles sp);
  equal_edges sp;
  List.iter parallel_to_an_axis (edges sp)

(* Two equilateral triangles: corners of 60 degrees at the ends, 120 where they
   join. Standing on a corner and taller than wide, the 120-degree corners are
   the two of middle y. *)
let diamond_triangles () =
  let sp = outline Symbol.diamond in
  equal angle [ 60.; 120.; 60.; 120. ] (from 60. (angles sp));
  equal_edges sp;
  let corners =
    List.combine sp.pts (angles sp)
    |> List.sort (fun ((_, y), _) ((_, y'), _) -> Float.compare y y')
  in
  equal ~msg:"angles from the top down" angle [ 60.; 120.; 120.; 60. ]
    (List.map snd corners);
  match corners with
  | [ _; ((_, y), _); ((_, y'), _); _ ] ->
      equal ~msg:"joining side" float_exact y y'
  | cs -> failf "%d vertices" (List.length cs)

let triangle_up () =
  let sp = outline Symbol.triangle in
  let near = float (rounding *. radius sp.pts) in
  equal angle [ 60.; 60.; 60. ] (angles sp);
  equal_edges sp;
  match top_down sp with
  | [ (x, y); (x0, y0); (x1, y1) ] ->
      equal ~msg:"base" near y0 y1;
      less ~msg:"apex" float_exact ~than:y0 y;
      equal ~msg:"apex over the base's middle" near ((x0 +. x1) /. 2.) x
  | pts -> failf "%d vertices" (List.length pts)

(* A Greek cross of equal sides turns 270 degrees into each arm and 90 at each
   arm's two outer corners. *)
let cross_squares () =
  let sp = outline Symbol.cross in
  equal angle
    (List.concat (List.init 4 (fun _ -> [ 270.; 90.; 90. ])))
    (from 270. (angles sp));
  equal_edges sp

(* A star of equal edges with points of 36 degrees and inner corners of 252 is
   regular, and each inner corner lies on the lines that join the points. *)
let star_regular () =
  let sp = outline Symbol.star in
  equal angle
    (List.concat (List.init 5 (fun _ -> [ 36.; 252. ])))
    (from 36. (angles sp));
  equal_edges sp

let star_up () =
  let sp = outline Symbol.star in
  match top_down sp with
  | (x, _) :: _ -> equal (float (rounding *. radius sp.pts)) 0. x
  | [] -> fail "no vertex"

(* Squares on the sides of a triangle: 90 degrees at each square's outer
   corners, [90 + 60 + 90] at each corner of the triangle. *)
let wye_squares () =
  let sp = outline Symbol.wye in
  equal angle
    (List.concat (List.init 3 (fun _ -> [ 240.; 90.; 90. ])))
    (from 240. (angles sp));
  equal_edges sp

(* The arm below ends in the side of greatest y, which is horizontal. *)
let wye_down () =
  let sp = outline Symbol.wye in
  match List.rev (top_down sp) with
  | (_, y) :: (_, y') :: (_, y'') :: _ ->
      equal ~msg:"lowest side" (float (rounding *. radius sp.pts)) y y';
      less ~msg:"next vertex" float_exact ~than:y' y''
  | _ -> fail "fewer than three vertices"

let strokes s = subpaths (Symbol.path `Stroke 1. s)

(* The directions of the strokes of an open symbol, in degrees in [0;180[, off
   by a few roundings of [atan2] and of the strokes' ends. *)
let directions s =
  List.map
    (fun sp ->
      match sp.pts with
      | [ (x0, y0); (x1, y1) ] ->
          let d = Float.atan2 (y1 -. y0) (x1 -. x0) *. 180. /. Float.pi in
          let d = Float.rem (d +. 360.) 180. in
          if d > 180. -. 1e-12 then 0. else d
      | _ -> fail "not a stroke")
    (strokes s)
  |> List.sort Float.compare

let direction = list (float 1e-12)

let stroke_lengths s =
  let ls = List.map (fun sp -> length [ sp ]) (strokes s) in
  let l = List.hd ls in
  equal (list (rel rounding l)) (List.map (fun _ -> l) ls) ls

let stroke_middles s =
  let middle sp =
    match sp.pts with
    | [ (x0, y0); (x1, y1) ] -> ((x0 +. x1) /. 2., (y0 +. y1) /. 2.)
    | _ -> fail "not a stroke"
  in
  let sps = strokes s in
  let ms = List.map middle sps in
  List.iter
    (equal (near_pt (rounding *. radius (vertices sps))) (List.hd ms))
    ms

let symbols =
  group "symbols"
    [
      cases "the circle is Path.circle of the circle's area" ~id:paint_name
        paints (fun paint ->
          equal path
            (Path.circle (P2.v 0. 0.) (Float.sqrt (2. /. Float.pi)))
            (Symbol.path paint 2. Symbol.circle));
      test "square is four equal sides parallel to the axes" square_sides;
      test
        "diamond is two equilateral triangles joined by a horizontal side, on \
         a corner"
        diamond_triangles;
      test "triangle is equilateral, pointing up" triangle_up;
      test "cross is five equal squares" cross_squares;
      test "star is regular, its inner corners on the lines joining its points"
        star_regular;
      test "star points up" star_up;
      test "wye is a triangle with a square on each side" wye_squares;
      test "wye has one arm down" wye_down;
      cases "plus, times and asterisk have strokes of equal length" ~id:name
        open_ stroke_lengths;
      cases "plus, times and asterisk cross at their strokes' middles" ~id:name
        open_ stroke_middles;
      test "plus is parallel to the axes" (fun () ->
          equal direction [ 0.; 90. ] (directions Symbol.plus));
      test "times is plus turned by an eighth of a turn" (fun () ->
          equal direction [ 45.; 135. ] (directions Symbol.times));
      test "asterisk's strokes are a sixth of a turn apart, one vertical"
        (fun () ->
          equal direction [ 30.; 90.; 150. ] (directions Symbol.asterisk));
    ]

(* Sets *)

let sets =
  group "sets"
    [
      test "filled is circle, cross, diamond, square, star, triangle and wye"
        (fun () ->
          equal (list symbol)
            Symbol.[ circle; cross; diamond; square; star; triangle; wye ]
            Symbol.filled);
      test
        "stroked is circle, plus, times, triangle, asterisk, square and diamond"
        (fun () ->
          equal (list symbol)
            Symbol.[ circle; plus; times; triangle; asterisk; square; diamond ]
            Symbol.stroked);
    ]

(* Path *)

let clockwise s =
  List.iter
    (fun paint ->
      List.iter
        (fun sp ->
          satisfies ~msg:(paint_name paint) ~claim:"a closed subpath" subpath
            (fun sp -> sp.closed)
            sp;
          greater ~msg:(paint_name paint) float_exact ~than:0. (area [ sp ]))
        (flat paint 3. s))
    paints

let one_segment_strokes s =
  List.iter
    (fun paint ->
      List.iter
        (satisfies ~msg:(paint_name paint)
           ~claim:"an open subpath of one segment" subpath (fun sp ->
             (not sp.closed) && List.length sp.pts = 2))
        (subpaths (Symbol.path paint 2. s)))
    paints

let at_origin s =
  List.iter
    (fun paint ->
      List.iter
        (equal ~msg:(paint_name paint) origin (0., 0.))
        (points (Symbol.path paint 0. s)))
    paints

let raises_on a =
  List.iter
    (fun s ->
      List.iter
        (fun paint ->
          raises_match ~msg:(label s paint) (Exn.invalid_arg ?sub:None)
            (fun () -> Symbol.path paint a s))
        paints)
    all

let path_ =
  group "path"
    [
      cases "closed outlines turn clockwise on screen" ~id:name closed clockwise;
      cases "open symbols are one segment per stroke" ~id:name open_
        one_segment_strokes;
      cases "size 0. puts every point at the origin" ~id:name all at_origin;
      cases "raises on a negative or non-finite size" ~id:(Printf.sprintf "%h")
        [ -1.; -.Float.min_float; -.Float.succ 0.; nan; infinity; neg_infinity ]
        raises_on;
    ]

(* Comparing and formatting *)

let comparing =
  group "comparing and formatting"
    [
      test "equal holds of a symbol and itself alone" (fun () ->
          List.iter
            (fun s ->
              List.iter
                (fun s' ->
                  if s == s' then equal symbol s s' else not_equal symbol s s')
                all)
            all);
      test "pp formats a symbol's name" (fun () ->
          equal (list string)
            [
              "circle";
              "square";
              "diamond";
              "triangle";
              "cross";
              "star";
              "wye";
              "plus";
              "times";
              "asterisk";
            ]
            (List.map name all));
    ]

let () =
  exit (run "Symbol" [ sizes; symmetry; symbols; sets; path_; comparing ])
