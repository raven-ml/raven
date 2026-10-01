(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_kit

let invalid f = raises_match (Exn.invalid_arg ?substring:None) f
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

(* Subpaths as flattened by [Path.flatten]: their points and whether they are
   closed. *)
type subpath = { pts : (float * float) list; closed : bool }

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

(* The edges of a subpath, the closing one included. *)
let edges { pts; closed } =
  let rec go acc = function
    | a :: (b :: _ as rest) -> go ((a, b) :: acc) rest
    | [ last ] when closed -> List.rev ((last, List.hd pts) :: acc)
    | _ -> List.rev acc
  in
  go [] pts

let length sps =
  List.fold_left
    (fun l sp ->
      List.fold_left
        (fun l ((x0, y0), (x1, y1)) -> l +. Float.hypot (x1 -. x0) (y1 -. y0))
        l (edges sp))
    0. sps

(* Shoelace sums with y pointing down: an outline clockwise on screen has a
   positive area. *)
let cross_ ((x0, y0), (x1, y1)) = (x0 *. y1) -. (x1 *. y0)

let area sps =
  List.fold_left
    (fun a sp -> List.fold_left (fun a e -> a +. (cross_ e /. 2.)) a (edges sp))
    0. sps

let region_centroid sps =
  let a = area sps in
  let cx, cy =
    List.fold_left
      (fun (cx, cy) sp ->
        List.fold_left
          (fun (cx, cy) (((x0, y0), (x1, y1)) as e) ->
            let c = cross_ e in
            (cx +. ((x0 +. x1) *. c), cy +. ((y0 +. y1) *. c)))
          (cx, cy) (edges sp))
      (0., 0.) sps
  in
  (cx /. (6. *. a), cy /. (6. *. a))

let outline_centroid sps =
  let l = length sps in
  let cx, cy =
    List.fold_left
      (fun (cx, cy) sp ->
        List.fold_left
          (fun (cx, cy) ((x0, y0), (x1, y1)) ->
            let d = Float.hypot (x1 -. x0) (y1 -. y0) in
            (cx +. (d *. (x0 +. x1) /. 2.), cy +. (d *. (y0 +. y1) /. 2.)))
          (cx, cy) (edges sp))
      (0., 0.) sps
  in
  (cx /. l, cy /. l)

let vertices s =
  List.concat_map (fun sp -> sp.pts) (subpaths (Symbol.path `Fill 1. s))

(* Generators *)

let gen_size =
  Gen.frequency
    [
      (4, Gen.float_range 0. 1000.);
      ( 1,
        Gen.of_list
          ~pp:(fun ppf a -> Format.fprintf ppf "%h" a)
          [ 1.; 1e-12; 1e12; 1e-200; 1e200 ] );
    ]

(* Sizes whose outlines' coordinates multiply without underflow. *)
let gen_positive = Gen.such_that (fun a -> a >= 1e-200) gen_size
let gen_of l = Gen.of_list ~pp:Symbol.pp l

let gen_paint =
  Gen.of_list
    ~pp:(fun ppf p ->
      Format.pp_print_string ppf
        (match p with `Fill -> "`Fill" | `Stroke -> "`Stroke"))
    [ `Fill; `Stroke ]

let rel eps x = float (eps *. Float.abs x)

(* Sizing by ink *)

(* Polygons are exact up to rounding; the circle's arcs depart from it by less
   than 0.03% of its radius. *)
let eps s = if Symbol.equal s Symbol.circle then 1e-3 else 1e-12
let filled_area (s, a) = equal (rel (eps s) a) a (area (flat `Fill a s))

let stroked_length (s, a) =
  let ink = 2. *. Float.sqrt (Float.pi *. a) in
  equal (rel (eps s) ink) ink (length (flat `Stroke a s))

let open_length (s, paint, a) =
  let ink = 2. *. Float.sqrt (Float.pi *. a) in
  equal (rel 1e-12 ink) ink (length (flat paint a s))

let centroids (s, paint, a) =
  let sps = flat paint a s in
  let near = float (1e-9 *. Float.sqrt a) in
  let x, y = outline_centroid sps in
  equal ~msg:"outline x" near 0. x;
  equal ~msg:"outline y" near 0. y;
  if not (List.memq s open_) then begin
    let x, y = region_centroid sps in
    equal ~msg:"region x" near 0. x;
    equal ~msg:"region y" near 0. y
  end

(* The numbers [Path.fold] visits, a [nan] standing for a close. *)
let numbers p =
  Path.fold
    ~move:(fun acc x y -> y :: x :: acc)
    ~line:(fun acc x y -> y :: x :: acc)
    ~cubic:(fun acc a b c d e f -> f :: e :: d :: c :: b :: a :: acc)
    ~close:(fun acc -> nan :: acc)
    [] p
  |> List.rev

let scaling (s, paint, a) =
  let k = Float.sqrt a in
  let near u v =
    (Float.is_nan u && Float.is_nan v)
    || Float.abs (u -. v) <= 1e-12 *. (k +. Float.abs u)
  in
  equal
    (list (Testable.make ~pp:Format.pp_print_float ~equal:near))
    (numbers (Path.transform (Affine.scale k k) (Symbol.path paint 1. s)))
    (numbers (Symbol.path paint a s))

let sizing =
  group "sizing"
    [
      prop "filled, a symbol encloses its size"
        (Gen.pair (gen_of closed) gen_positive)
        filled_area;
      prop "stroked, a symbol's outline is as long as the circle's"
        (Gen.pair (gen_of closed) gen_positive)
        stroked_length;
      prop "open symbols are as long as the circle's outline however painted"
        (Gen.triple (gen_of open_) gen_paint gen_positive)
        open_length;
      prop "centroids lie at the origin"
        (Gen.triple (gen_of all) gen_paint gen_positive)
        centroids;
      prop "a symbol of size a is the symbol of size 1 scaled by its root"
        (Gen.triple (gen_of all) gen_paint gen_positive)
        scaling;
      cases "a circle is the same filled and stroked" ~name [ Symbol.circle ]
        (fun s ->
          equal path (Symbol.path `Fill 7. s) (Symbol.path `Stroke 7. s));
      cases "size 0 puts every point at the origin" ~name all (fun s ->
          List.iter
            (fun paint ->
              List.iter
                (fun sp ->
                  List.iter
                    (fun (x, y) ->
                      equal float_exact 0. x;
                      equal float_exact 0. y)
                    sp.pts)
                (subpaths (Symbol.path paint 0. s)))
            [ `Fill; `Stroke ]);
      cases "a size of -0. is a size of 0" ~name all (fun s ->
          List.iter
            (fun paint ->
              equal path (Symbol.path paint 0. s) (Symbol.path paint (-0.) s))
            [ `Fill; `Stroke ]);
      cases "path raises on a negative or non-finite size"
        ~name:(fun a -> Printf.sprintf "%h" a)
        [ -1.; -.Float.min_float; nan; infinity; neg_infinity ]
        (fun a ->
          List.iter (fun s -> invalid (fun () -> Symbol.path `Fill a s)) all);
    ]

(* Shapes *)

let pt = pair (float 1e-12) (float 1e-12)

let clockwise s =
  List.iter
    (fun paint ->
      List.iter
        (fun sp ->
          is_true ~msg:"closed" sp.closed;
          greater float_exact ~than:0. (area [ sp ]))
        (flat paint 3. s))
    [ `Fill; `Stroke ]

let one_segment_strokes s =
  let sps = subpaths (Symbol.path `Fill 2. s) in
  List.iter
    (fun sp ->
      is_false ~msg:"closed" sp.closed;
      equal int 2 (List.length sp.pts))
    sps

(* The directions of the strokes of an open symbol, in degrees in [0;180[. *)
let directions s =
  List.map
    (fun sp ->
      match sp.pts with
      | [ (x0, y0); (x1, y1) ] ->
          let d = Float.atan2 (y1 -. y0) (x1 -. x0) *. 180. /. Float.pi in
          let d = Float.rem (d +. 360.) 180. in
          if d > 180. -. 1e-9 then 0. else d
      | _ -> fail "not a stroke")
    (subpaths (Symbol.path `Stroke 1. s))
  |> List.sort Float.compare

let stroke_middles s =
  List.iter
    (fun sp ->
      match sp.pts with
      | [ (x0, y0); (x1, y1) ] ->
          equal pt (0., 0.) ((x0 +. x1) /. 2., (y0 +. y1) /. 2.)
      | _ -> fail "not a stroke")
    (subpaths (Symbol.path `Stroke 1. s))

let triangle_shape () =
  match vertices Symbol.triangle with
  | [ (ax, ay); (bx, by); (cx, cy) ] ->
      let d (x, y) (x', y') = Float.hypot (x -. x') (y -. y') in
      let side = d (ax, ay) (bx, by) in
      equal (rel 1e-12 side) side (d (bx, by) (cx, cy));
      equal (rel 1e-12 side) side (d (cx, cy) (ax, ay));
      equal (float 1e-12) 0. ax;
      less float_exact ~than:0. ay
  | v -> failf "%d vertices" (List.length v)

let diamond_shape () =
  match vertices Symbol.diamond with
  | [ (0., top); (right, 0.); (0., bottom); (left, 0.) ] ->
      equal (rel 1e-12 top) (-.bottom) top;
      equal (rel 1e-12 left) (-.right) left;
      equal (float 1e-12) (Float.sqrt 3.) (bottom /. right)
  | v -> failf "%d vertices" (List.length v)

let square_shape () =
  match vertices Symbol.square with
  | [ (x0, y0); (x1, y1); (x2, y2); (x3, y3) ] ->
      equal float_exact y0 y1;
      equal float_exact x1 x2;
      equal float_exact y2 y3;
      equal float_exact x3 x0;
      equal (float 1e-12) (x1 -. x0) (y2 -. y1)
  | v -> failf "%d vertices" (List.length v)

let cross_shape () =
  let v = Array.of_list (vertices Symbol.cross) in
  equal int 12 (Array.length v);
  let side = Float.hypot (fst v.(1) -. fst v.(0)) (snd v.(1) -. snd v.(0)) in
  Array.iteri
    (fun i (x, y) ->
      let x', y' = v.((i + 1) mod 12) in
      equal
        ~msg:(Printf.sprintf "edge %d" i)
        (rel 1e-12 side) side
        (Float.hypot (x' -. x) (y' -. y)))
    v

(* Each inner corner lies on the line joining the points on either side of the
   point it neighbours. *)
let star_shape () =
  let v = Array.of_list (vertices Symbol.star) in
  equal int 10 (Array.length v);
  equal pt (0., snd v.(0)) v.(0);
  less float_exact ~than:0. (snd v.(0));
  for i = 0 to 4 do
    let x, y = v.((2 * i) + 1) in
    let ax, ay = v.(2 * i) and bx, by = v.(((2 * i) + 4) mod 10) in
    equal
      ~msg:(Printf.sprintf "inner corner %d" i)
      (float 1e-12) 0.
      (((bx -. ax) *. (y -. ay)) -. ((by -. ay) *. (x -. ax)))
  done

let wye_shape () =
  let v = Array.of_list (vertices Symbol.wye) in
  equal int 9 (Array.length v);
  (* One arm points down: its two outer corners have the largest y. *)
  equal (float 1e-12) (snd v.(1)) (snd v.(2));
  equal (float 1e-12) (-.fst v.(1)) (fst v.(2));
  Array.iter (fun (_, y) -> at_most float_exact ~than:(snd v.(1) +. 1e-12) y) v

let shapes =
  group "shapes"
    [
      cases "closed outlines turn clockwise on screen" ~name closed clockwise;
      cases "open symbols are one segment per stroke" ~name open_
        one_segment_strokes;
      cases "strokes cross at their middles" ~name open_ stroke_middles;
      test "plus is parallel to the axes" (fun () ->
          equal (list (float 1e-9)) [ 0.; 90. ] (directions Symbol.plus));
      test "times is plus turned by an eighth of a turn" (fun () ->
          equal (list (float 1e-9)) [ 45.; 135. ] (directions Symbol.times));
      test "asterisk's strokes are a sixth of a turn apart, one vertical"
        (fun () ->
          equal
            (list (float 1e-9))
            [ 30.; 90.; 150. ]
            (directions Symbol.asterisk));
      test "the circle is Path.circle" (fun () ->
          equal path
            (Path.circle (P2.v 0. 0.) (Float.sqrt (2. /. Float.pi)))
            (Symbol.path `Fill 2. Symbol.circle));
      test "triangle is equilateral, pointing up" triangle_shape;
      test "diamond is two equilateral triangles, taller than wide"
        diamond_shape;
      test "square has sides parallel to the axes" square_shape;
      test "cross is five equal squares" cross_shape;
      test "star points up, its inner corners on the lines joining its points"
        star_shape;
      test "wye has one arm down" wye_shape;
    ]

(* Sets, comparing and formatting *)

let sets =
  group "sets, comparing and formatting"
    [
      test "filled" (fun () ->
          equal (list symbol)
            Symbol.[ circle; cross; diamond; square; star; triangle; wye ]
            Symbol.filled);
      test "stroked" (fun () ->
          equal (list symbol)
            Symbol.[ circle; plus; times; triangle; asterisk; square; diamond ]
            Symbol.stroked);
      test "distinct symbols are unequal" (fun () ->
          List.iteri
            (fun i s ->
              List.iteri
                (fun j s' ->
                  if i = j then equal symbol s s'
                  else not_equal ~msg:(name s ^ " " ^ name s') symbol s s')
                all)
            all);
      test "pp names a symbol" (fun () ->
          expect (String.concat " " (List.map name all))
          @@ __POS_OF__
               {| circle square diamond triangle cross star wye plus times asterisk |});
    ]

let () = exit (run "Symbol" [ sizing; shapes; sets ])
