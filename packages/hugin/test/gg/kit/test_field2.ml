(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_gg_kit

let nan = Float.nan

(* Fields from rows of samples. *)
let tensor rows =
  let ny = List.length rows in
  let nx = match rows with [] -> 0 | r :: _ -> Array.length r in
  Nx.create Nx.float64 [| ny; nx |] (Array.concat rows)

let field ?xs ?ys rows = Field2.v ?xs ?ys (tensor rows)
let domain f = Field2.isoband ~lo:neg_infinity ~hi:infinity f

(* [R l], the region of level [l]. *)
let region l f = Field2.isoband ~lo:l ~hi:infinity f

(* Witnesses print every digit, so that a failure shows the difference. *)

let pp_pts ppf pts =
  Format.fprintf ppf "@[<1>[";
  List.iteri
    (fun i (x, y) ->
      if i > 0 then Format.fprintf ppf ";@ ";
      Format.fprintf ppf "(%.17g, %.17g)" x y)
    pts;
  Format.fprintf ppf "]@]"

let ring_pts r =
  List.init (Ring2.length r) (fun i -> (Ring2.x r i, Ring2.y r i))

let pp_pgon ppf p =
  Format.fprintf ppf "@[<v>%a@]"
    (Format.pp_print_list (fun ppf r -> pp_pts ppf (ring_pts r)))
    (Pgon2.rings p)

let pgon = Testable.make ~pp:pp_pgon ~equal:Pgon2.equal
let path = Testable.make ~pp:Path.pp ~equal:Path.equal

let pp_box ppf b =
  Format.fprintf ppf "[%.17g, %.17g; %.17g, %.17g]" (Box2.minx b) (Box2.miny b)
    (Box2.maxx b) (Box2.maxy b)

let box2 = Testable.make ~pp:pp_box ~equal:Box2.equal

(* Areas compare within 5e-9. Coordinates here stay below 1e6, a ring's width
   and height below 10, and a band's rings hold fewer than 100 points in all.

   ring2.mli bounds a ring's rounding error by n^2 eps w h. Its sum of the [n -
   2] terms [ax by - bx ay], over differences to the first point, meets it, with
   u = eps / 2: each difference rounds by u of itself, so each product, at most
   w h, rounds by 3 u w h and each term, below 2 w h, by 8 u w h; the running
   sum adds (n - 3) u times the sum of the terms' magnitudes, below (n - 3) (n -
   2) 2 u w h; halving is exact. In all, (n - 2) (n + 1) u w h, below n^2 eps w
   h with room for second-order terms.

   So a band's area, the sum of its rings', rounds by less than 100^2 eps 10^2 =
   2.2e-10, and a sum of five bands by five times that. A crossing lies off its
   exact line by its coordinates' rounding, below 1e6 eps = 2.2e-10, which moves
   an area by at most that times the ring's extent for each of at most three
   crossings on a cut cell's diagonal: 6.7e-10 on a unit diagonal. *)
let area = float 5e-9

(* The curves of a path: points, and whether the curve is closed. *)
let curves p =
  let flush cur acc =
    match cur with
    | None -> acc
    | Some (pts, closed) -> (List.rev pts, closed) :: acc
  in
  let cur, acc =
    Path.fold
      ~move:(fun (cur, acc) x y -> (Some ([ (x, y) ], false), flush cur acc))
      ~line:(fun (cur, acc) x y ->
        match cur with
        | Some (pts, c) -> (Some ((x, y) :: pts, c), acc)
        | None -> failf "a line without a start")
      ~cubic:(fun _ _ _ _ _ _ _ -> failf "a curve in an isoline")
      ~close:(fun (cur, acc) ->
        match cur with
        | Some (pts, _) -> (Some (pts, true), acc)
        | None -> (cur, acc))
      (None, []) p
  in
  List.rev (flush cur acc)

let pp_curves ppf cs =
  Format.fprintf ppf "@[<v>%a@]"
    (Format.pp_print_list (fun ppf (pts, closed) ->
         Format.fprintf ppf "%s %a"
           (if closed then "closed" else "open")
           pp_pts pts))
    cs

(* A ring or a closed curve compared up to its first point. *)
let rotations pts =
  let n = List.length pts in
  let a = Array.of_list pts in
  List.init n (fun k -> List.init n (fun i -> a.((i + k) mod n)))

let same_cycle a b = List.mem b (rotations a)

(* Rings compared up to their order and first points. *)
let same_rings expected p =
  let actual = List.map ring_pts (Pgon2.rings p) in
  let pp = Format.pp_print_list pp_pts in
  let ok =
    List.length expected = List.length actual
    && List.for_all (fun e -> List.exists (same_cycle e) actual) expected
  in
  if not ok then
    failf
      "@[<v>rings differ up to order and first points@,\
       expected@,\
      \  %a@,\
       actual@,\
      \  %a@]"
      pp expected pp actual

(* Curves compared up to their order and the first point of closed ones. *)
let same_curves expected p =
  let actual = curves p in
  let same (e, c) (a, d) = c = d && if c then same_cycle e a else e = a in
  let ok =
    List.length expected = List.length actual
    && List.for_all (fun e -> List.exists (same e) actual) expected
  in
  if not ok then
    failf "@[<v>curves differ up to order@,expected@,  %a@,actual@,  %a@]"
      pp_curves expected pp_curves actual

let signed_area pts =
  let a = Array.of_list pts in
  let n = Array.length a in
  let s = ref 0. in
  for i = 0 to n - 1 do
    let x0, y0 = a.(i) and x1, y1 = a.((i + 1) mod n) in
    s := !s +. ((x0 *. y1) -. (x1 *. y0))
  done;
  0.5 *. !s

(* The winding number of the rings of [p] around [pt], by the reference's rule
   for points on rings. *)
let winding p pt =
  List.fold_left (fun w r -> w + Winding.ring pt r) 0 (Pgon2.rings p)

(* Fields of the hand cases *)

(* The L1 cone [4 - |x - 2| - |y - 2|] is linear on every cell, so its regions
   are exact diamonds. *)
let cone =
  lazy
    (field
       (List.init 5 (fun i ->
            Array.init 5 (fun j ->
                4.
                -. Float.abs (Float.of_int (i - 2))
                -. Float.abs (Float.of_int (j - 2))))))

(* Peaks and ridges at level 1 with only lower samples around. *)
let ridge = [ [| 0.; 0.; 0. |]; [| 1.; 1.; 1. |]; [| 0.; 0.; 0. |] ]
let border_row = [ [| 1.; 1.; 1. |]; [| 0.; 0.; 0. |] ]
let peak = [ [| 0.; 0.; 0. |]; [| 0.; 1.; 0. |]; [| 0.; 0.; 0. |] ]

(* Laws over small random fields *)

type sample = {
  rows : float array list;
  xs : float array option;
  ys : float array option;
}

let pp_sample ppf s =
  let pp_arr ppf a =
    Format.fprintf ppf "[|%a|]"
      (Format.pp_print_array
         ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
         (fun ppf x -> Format.fprintf ppf "%.17g" x))
      a
  in
  let pp_opt name ppf = function
    | None -> ()
    | Some a -> Format.fprintf ppf "~%s:%a " name pp_arr a
  in
  Format.fprintf ppf "@[<v>%a%a@,%a@]" (pp_opt "xs") s.xs (pp_opt "ys") s.ys
    (Format.pp_print_list pp_arr)
    s.rows

let sample_field s = field ?xs:s.xs ?ys:s.ys s.rows
let coord a k = match a with None -> Float.of_int k | Some a -> a.(k)

(* The cells of a sample: their corners' points and samples, in order around the
   cell. *)
let cells s =
  let rows = Array.of_list s.rows in
  let ny = Array.length rows in
  let nx = if ny = 0 then 0 else Array.length rows.(0) in
  let corner (i, j) = ((coord s.xs j, coord s.ys i), rows.(i).(j)) in
  List.concat_map
    (fun i ->
      List.map
        (fun j ->
          List.map corner [ (i, j); (i, j + 1); (i + 1, j + 1); (i + 1, j) ])
        (List.init (Int.max 0 (nx - 1)) Fun.id))
    (List.init (Int.max 0 (ny - 1)) Fun.id)

let missing cell =
  List.length (List.filter (fun (_, v) -> not (Float.is_finite v)) cell)

let cell_area cell =
  match cell with
  | [ ((x0, y0), _); _; ((x1, y1), _); _ ] ->
      Float.abs ((x1 -. x0) *. (y1 -. y0))
  | _ -> assert false

(* The domain's area by its definition: whole cells, and half of each cell with
   one missing sample. *)
let domain_area s =
  List.fold_left
    (fun a cell ->
      match missing cell with
      | 0 -> a +. cell_area cell
      | 1 -> a +. (cell_area cell /. 2.)
      | _ -> a)
    0. (cells s)

let grid_box s =
  let rows = Array.of_list s.rows in
  let ny = Array.length rows and nx = Array.length rows.(0) in
  Box2.of_pts
    (P2.v (coord s.xs 0) (coord s.ys 0))
    (P2.v (coord s.xs (nx - 1)) (coord s.ys (ny - 1)))

(* The diagonals of the cells with exactly one missing sample: the segments
   between the two corners next to the missing one. *)
let cut_diagonals s =
  List.filter_map
    (fun cell ->
      if missing cell <> 1 then None
      else
        let c = Array.of_list cell in
        let k = ref 0 in
        Array.iteri (fun i (_, v) -> if not (Float.is_finite v) then k := i) c;
        Some (fst c.((!k + 1) mod 4), fst c.((!k + 3) mod 4)))
    (cells s)

(* Whether [(x, y)] lies within [tol] of segment [a, b], [tol] a billionth of
   the coordinates' magnitude: far wider than the rounding error by which
   crossings lie off a diagonal. *)
let near_segment (x, y) ((ax, ay), (bx, by)) =
  let tol =
    1e-9 *. (1. +. Float.abs ax +. Float.abs ay +. Float.abs bx +. Float.abs by)
  in
  Float.min ax bx -. tol <= x
  && x <= Float.max ax bx +. tol
  && Float.min ay by -. tol <= y
  && y <= Float.max ay by +. tol
  && Float.abs (((bx -. ax) *. (y -. ay)) -. ((by -. ay) *. (x -. ax)))
     <= tol *. Float.hypot (bx -. ax) (by -. ay)

(* Whether two of the finite levels lie within rounding error of each other, so
   that their crossings on a side do. *)
let close_levels levels =
  let rec loop = function
    | a :: (b :: _ as rest) ->
        (Float.is_finite b && b -. a <= 4. *. epsilon_float *. Float.abs b)
        || loop rest
    | _ -> false
  in
  loop levels

(* Values cluster on the levels the laws use, with their neighbours, so that
   ties, crossings at corners and nearly equal levels are common. Other values
   lie on a grid of 1/64: below it, coordinates of crossings and their
   differences become subnormal, where products underflow and the conventions'
   rule for points on rings is no longer exact. *)
let level_values = [ 0.; 1.; 2.; 3.; 1.5; Float.succ 1.; Float.pred 2.; 2.5 ]

let on_grid lo hi =
  Gen.map (fun x -> Float.round (x *. 64.) /. 64.) (Gen.float_range lo hi)

let gen_value =
  Gen.frequency
    [
      (8, Gen.of_list level_values);
      (3, on_grid (-0.5) 3.5);
      (1, Gen.of_list [ nan; infinity; neg_infinity ]);
    ]

let gen_coords n =
  let open Gen in
  frequency
    [
      (2, constant None);
      ( 1,
        let+ start = float_range (-3.) 3.
        and+ steps =
          array ~size:(constant (max 0 (n - 1))) (float_range 0.25 2.)
        and+ down = bool in
        let a = Array.make n start in
        for i = 1 to n - 1 do
          a.(i) <-
            (if down then a.(i - 1) -. steps.(i - 1)
             else a.(i - 1) +. steps.(i - 1))
        done;
        Some a );
    ]

let gen_sample =
  let open Gen in
  with_pp pp_sample
    (let* ny, nx = pair (int_range 1 5) (int_range 1 5) in
     let+ rows = list ~size:(constant ny) (array ~size:(constant nx) gen_value)
     and+ xs = gen_coords nx
     and+ ys = gen_coords ny in
     { rows; xs; ys })

let gen_levels =
  Gen.map
    (fun ls -> List.sort_uniq Float.compare ls)
    (Gen.list ~size:(Gen.int_range 1 4)
       (Gen.frequency [ (3, Gen.of_list level_values); (1, on_grid 0. 3.) ]))

let gen_case =
  Gen.with_pp
    (fun ppf (s, ls) ->
      Format.fprintf ppf "@[<v>%a@,levels %a@]" pp_sample s
        (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf l ->
             Format.fprintf ppf "%.17g" l))
        ls)
    (Gen.pair gen_sample gen_levels)

let segments_of_rings p =
  List.concat_map
    (fun r ->
      let pts = Array.of_list (ring_pts r) in
      let n = Array.length pts in
      List.init n (fun i -> (pts.(i), pts.((i + 1) mod n))))
    (Pgon2.rings p)

let segments_of_curves cs =
  List.concat_map
    (fun (pts, closed) ->
      let a = Array.of_list pts in
      let n = Array.length a in
      let m = if closed then n else n - 1 in
      List.init m (fun i -> (a.(i), a.((i + 1) mod n))))
    cs

(* The sign of the orientation of [c] from [a] to [b], [0] within rounding
   error. *)
let orient (ax, ay) (bx, by) (cx, cy) =
  let p = (bx -. ax) *. (cy -. ay) and q = (by -. ay) *. (cx -. ax) in
  let tol = 1e-9 *. (Float.abs p +. Float.abs q) in
  if p -. q > tol then 1 else if q -. p > tol then -1 else 0

(* Whether segments share more than a point. Collinear overlaps arise only along
   the horizontal and vertical lines of the grid, where they are exact; a
   crossing is reported only when every orientation is clear. *)
let cross_or_overlap ((a, b) as s) ((c, d) as t) =
  let overlap coord =
    let lo1 = Float.min (coord a) (coord b)
    and hi1 = Float.max (coord a) (coord b) in
    let lo2 = Float.min (coord c) (coord d)
    and hi2 = Float.max (coord c) (coord d) in
    Float.min hi1 hi2 > Float.max lo1 lo2
  in
  let horizontal (p, q) = snd p = snd q and vertical (p, q) = fst p = fst q in
  (a = c && b = d)
  || (a = d && b = c)
  || (horizontal s && horizontal t && snd a = snd c && overlap fst)
  || (vertical s && vertical t && fst a = fst c && overlap snd)
  || (orient a b c * orient a b d < 0 && orient c d a * orient c d b < 0)

(* Whether segment [p, q] lies on segment [a, b], running the same way. A point
   computed by another route lies off the line of [a, b] by a few roundings of
   its coordinates; the cross product it gives is that distance times the
   segment's length, far below 1e-12 times the squared magnitude of the
   coordinates. *)
let lies_on (p, q) (a, b) =
  (p = a && q = b)
  ||
  let (px, py), (qx, qy), (ax, ay), (bx, by) = (p, q, a, b) in
  let within u v w = Float.min v w <= u && u <= Float.max v w in
  let same_way =
    ((qx -. px) *. (bx -. ax)) +. ((qy -. py) *. (by -. ay)) > 0.
  in
  let tol =
    1e-12
    *. (1. +. Float.abs ax +. Float.abs ay +. Float.abs bx +. Float.abs by)
       ** 2.
  in
  let on (ux, uy) =
    within ux ax bx && within uy ay by
    && Float.abs (((bx -. ax) *. (uy -. ay)) -. ((by -. ay) *. (ux -. ax)))
       <= tol
  in
  same_way && on p && on q

let fail_segments what (a, b) (c, d) =
  failf "%s: segments %a and %a cross or overlap" what pp_pts [ a; b ] pp_pts
    [ c; d ]

(* Whether two segments of [segs] cross or overlap, [exempt] aside. *)
let check_simple ?(exempt = fun _ _ -> false) what segs =
  let segs = Array.of_list segs in
  let n = Array.length segs in
  for i = 0 to n - 1 do
    for j = i + 1 to n - 1 do
      if cross_or_overlap segs.(i) segs.(j) && not (exempt segs.(i) segs.(j))
      then fail_segments what segs.(i) segs.(j)
    done
  done

(* The points of each cell a quarter, a half and three quarters across it along
   each axis, and the domain's vertices. *)
let probes s f =
  let quarters = [ 0.25; 0.5; 0.75 ] in
  let in_cell cell =
    match cell with
    | [ ((x0, y0), _); _; ((x1, y1), _); _ ] ->
        List.concat_map
          (fun u ->
            List.map
              (fun v -> (x0 +. (u *. (x1 -. x0)), y0 +. (v *. (y1 -. y0))))
              quarters)
          quarters
    | _ -> assert false
  in
  List.concat_map in_cell (cells s)
  @ List.concat_map ring_pts (Pgon2.rings (domain f))

(* The bands between [levels] and infinite ends. *)
let bands levels f =
  let rec pairs = function
    | a :: (b :: _ as rest) -> (a, b) :: pairs rest
    | _ -> []
  in
  List.map
    (fun (lo, hi) -> Field2.isoband ~lo ~hi f)
    (pairs ((neg_infinity :: levels) @ [ infinity ]))

(* The band between the least and greatest of [levels], or above the level if
   there is one, with its bounds. *)
let band_of (s, levels) =
  let f = sample_field s in
  let lo = List.hd levels and hi = List.nth levels (List.length levels - 1) in
  let hi = if hi = lo then infinity else hi in
  (f, lo, hi, Field2.isoband ~lo ~hi f)

(* Whether points lie near a cut cell's diagonal where levels within rounding
   error of each other let rings cross. *)
let near_close_diagonal s lo hi pts =
  close_levels [ lo; hi ]
  && List.exists
       (fun dg -> List.for_all (fun pt -> near_segment pt dg) pts)
       (cut_diagonals s)

(* Fields on which the laws once failed: crossings on a diagonal rounding onto
   the line of another side, crossings of near levels on a diagonal that
   inverted a band, a saddle whose region collapses onto its diagonal, a point
   within rounding error of edges of three bands. Then cut cells far from the
   origin against their height, whose diagonals bend most at their crossings and
   whose step is set by y: one with bands overlapping on its diagonal, one whose
   crossing near an end would otherwise round onto the top side. *)

(* A cut cell 5.6e-6 high at y = 792240, where a unit in the last place of y is
   2e-5 of the cell's height. *)
let far_ys = [| 792240.; 792240.0000056 |]

let regressions =
  let p2 = Float.pred 2. and s1 = Float.succ 1. in
  let case ?xs ?ys rows levels = ({ rows; xs; ys }, levels) in
  [
    case
      ~xs:[| 0.2502392362475277; 0.50023923624752764 |]
      ~ys:[| 0.; 0.25; 0.5 |]
      [ [| 0.; 0. |]; [| nan; 1. |]; [| 2.; 2. |] ]
      [ s1; 2. ];
    case
      [ [| 0.; 0.; 2.; 1.; nan |]; [| 0.; 0.; 0.; 2.; 2.0000660703615822 |] ]
      [ s1 ];
    case
      ~ys:[| 0.; 1.0000359577848634; 1.2500359577848634 |]
      [ [| 0.; 0. |]; [| 3.; p2 |]; [| 2.5; nan |] ]
      [ 0.; 2. ];
    case
      [
        [| 0.; 0.; 0.; 0. |];
        [| 0.; nan; 2.5; nan |];
        [| 0.; 0.; 2.; 1. |];
        [| 0.; nan; 2.5; 0. |];
      ]
      [ p2; 2.; 3. ];
    case
      ~ys:[| 0.; 0.25; 0.5; 0.75068742307833647 |]
      [
        [| 0.; 0.; 0.; 0. |];
        [| 0.; 0.; 0.; 0. |];
        [| 0.; 0.; 9.9920962214652108e-16; nan |];
        [| 0.; 0.; 0.; 2.5935403645858823 |];
      ]
      [ 0.; p2; 2. ];
    case ~xs:[| 0.; 0.25; 0.5 |]
      [ [| 0.; 2.; p2 |]; [| 0.; p2; 2. |] ]
      [ 0.; 2. ];
    case
      ~xs:
        [|
          -0.989882094496219;
          -1.5000075269943542;
          -1.7500075269943542;
          -2.000007526994354;
        |]
      ~ys:[| -0.00039530343356600488; 1.50019762959069 |]
      [ [| 0.; nan; 3.; 0. |]; [| 0.; 0.; 0.; 0. |] ]
      [ 1.; s1 ];
    case ~xs:[| 2.; 3. |] ~ys:far_ys [ [| 3.; 0. |]; [| 3.; nan |] ] [ 1. ];
    case
      ~xs:[| -16.814359448746295; -16.607636386910656 |]
      ~ys:[| -5.1714227594632955; -3.5619193239620968 |]
      [
        [| 2.000000000000001; nan |];
        [| 0.6999999999999996; 1.9999999999999998 |];
      ]
      [ 1.9999999999999996; 2.0000000000000004 ];
    case ~xs:[| 2.; 3. |] ~ys:far_ys [ [| 3.; 0. |]; [| 1.5; nan |] ] [ 3e-6 ];
  ]

(* Domain and regions *)

(* Only the sample of row 0 and column 2 reaches 0.5: [R 0.5] is the triangle it
   cuts off at [(xs.(2), ys.(0))], to the midpoints of its sides. *)
let placement () =
  let f =
    field ~xs:[| 10.; 11.; 12. |] ~ys:[| 20.; 21. |]
      [ [| 0.; 0.; 1. |]; [| 0.; 0.; 0. |] ]
  in
  equal (option box2)
    (Some (Box2.v 11.5 20. 0.5 0.5))
    (Pgon2.bounds (region 0.5 f))

let missing_alike v =
  let with_ v = field [ [| 0.; 1.; 2. |]; [| 1.; v; 1. |]; [| 2.; 1.; 0. |] ] in
  equal pgon
    (Field2.isoband ~lo:0.5 ~hi:1.5 (with_ nan))
    (Field2.isoband ~lo:0.5 ~hi:1.5 (with_ v))

let domain_is_pieces s =
  let ms = List.map missing (cells s) in
  cover "a whole cell" (List.mem 0 ms);
  cover "a cell with one missing sample" (List.mem 1 ms);
  cover "a cell with two missing samples or more" (List.exists (( <= ) 2) ms);
  equal area (domain_area s) (Pgon2.area (domain (sample_field s)))

(* A NaN at the centre of a 5 x 5 grid, a NaN at a corner of a 3 x 3 grid, and
   two NaNs side by side in a 4 x 4 grid. *)
let domain_examples =
  let sample rows = { rows; xs = None; ys = None } in
  [
    sample
      (List.init 5 (fun i ->
           Array.init 5 (fun j ->
               if i = 2 && j = 2 then nan else Float.of_int j)));
    sample [ [| nan; 0.; 0. |]; [| 0.; 0.; 0. |]; [| 0.; 0.; 0. |] ];
    sample
      (List.init 4 (fun i ->
           Array.init 4 (fun j -> if i = 1 && (j = 1 || j = 2) then nan else 0.)));
  ]

(* Coordinates ordered with [0.] equal to [-0.]. *)
let coordinate = Testable.with_compare Float.compare float_exact

let in_grid_box (s, levels) =
  let f = sample_field s in
  let g = grid_box s in
  let within what b =
    at_least ~msg:(what ^ ": minx") coordinate ~than:(Box2.minx g) (Box2.minx b);
    at_least ~msg:(what ^ ": miny") coordinate ~than:(Box2.miny g) (Box2.miny b);
    at_most ~msg:(what ^ ": maxx") coordinate ~than:(Box2.maxx g) (Box2.maxx b);
    at_most ~msg:(what ^ ": maxy") coordinate ~than:(Box2.maxy g) (Box2.maxy b)
  in
  List.iter
    (fun l ->
      Option.iter (within "isoline") (Path.bounds (Field2.isoline l f));
      Option.iter (within "region") (Pgon2.bounds (region l f)))
    levels

(* The diamond of radius [r = 4 - l] around the cone's peak has area [2 r^2];
   past [r = 2] its four tips of area [(r - 2)^2] each lie beyond the grid. *)
let cone_regions = [ (3.5, 0.5); (2.5, 4.5); (1.5, 12.5 -. 1.) ]

(* On a unit saddle of samples 0 and 1, whose mean is 0.5, the crossings at [l]
   lie [l] from the out corners. Joined, the region is the cell less two
   triangles of legs [l], area [1 - l^2]; split, it is two triangles of legs [1
   - l], area [(1 - l)^2]. *)
let saddles =
  let saddle = [ [| 1.; 0. |]; [| 0.; 1. |] ] in
  let other = [ [| 0.; 1. |]; [| 1.; 0. |] ] in
  [
    ("below the mean", saddle, 0.4, 1, 1. -. (0.4 *. 0.4));
    ("at the mean", saddle, 0.5, 1, 1. -. (0.5 *. 0.5));
    ("above the mean", saddle, 0.6, 2, 0.4 *. 0.4);
    ("the other diagonal, at the mean", other, 0.5, 1, 1. -. (0.5 *. 0.5));
    ("the other diagonal, above the mean", other, 0.6, 2, 0.4 *. 0.4);
  ]

let peaks =
  [
    ("a ridge", None, ridge);
    ("a border row", None, border_row);
    ("an isolated peak", None, peak);
    ("an isolated peak at uneven coordinates", Some [| -0.5; 0.3; 1.1 |], peak);
  ]

(* The pit's neighbours equal 2, so the crossings of level 2 next to the pit are
   those neighbours: [R 1] less [R 2] is their diamond. *)
let pit_crossings () =
  let f = field [ [| 2.; 2.; 2. |]; [| 2.; 1.; 2. |]; [| 2.; 2.; 2. |] ] in
  same_rings
    [ [ (1., 0.); (2., 1.); (1., 2.); (0., 1.) ] ]
    (Field2.isoband ~lo:1. ~hi:2. f)

(* The field [1.5e308 (1 - 2x)] reaches 1e308 at [x = 1/6]. *)
let overflowing_samples () =
  let f = field [ [| 1.5e308; -1.5e308 |]; [| 1.5e308; -1.5e308 |] ] in
  equal area (1. /. 6.) (Pgon2.area (region 1e308 f))

(* The centre of a whole cell whose samples are all at least [l] is in [R l],
   and that of one whose samples are all below [l] is not. *)
let whole_cells (s, levels) =
  let f = sample_field s in
  let l = List.hd levels in
  let r = region l f in
  let all p cell = List.for_all (fun (_, v) -> Float.is_finite v && p v) cell in
  let above = all (fun v -> v >= l) and below = all (fun v -> v < l) in
  let cs = cells s in
  cover "a cell at or above l" (List.exists above cs);
  cover "a cell below l" (List.exists below cs);
  List.iter
    (fun cell ->
      let (x0, y0), _ = List.nth cell 0 and (x1, y1), _ = List.nth cell 2 in
      let c = P2.v ((x0 +. x1) /. 2.) ((y0 +. y1) /. 2.) in
      let msg = Format.asprintf "the centre %a" P2.pp c in
      if above cell then equal ~msg int 1 (winding r c);
      if below cell then equal ~msg int 0 (winding r c))
    cs

(* The diagonal of the [far_ys] cell runs one unit left from its top right
   corner, sample 0, to its bottom left one, sample 3. The crossing of a level
   [l] lies at x = 3 - l / 3 before it moves, and moves by at most the fraction
   256 ε (c + d) / d of the diagonal, which the cell's height sets: about 0.008
   in x. A crossing farther than that from both ends stays inside the cell. *)
let far_crossings_move_within_bound () =
  let f = field ~xs:[| 2.; 3. |] ~ys:far_ys [ [| 3.; 0. |]; [| 3.; nan |] ] in
  let y0 = far_ys.(0) and y1 = far_ys.(1) in
  let fraction c0 c1 =
    let d = Float.abs (c1 -. c0) in
    256. *. epsilon_float *. (Float.max (Float.abs c0) (Float.abs c1) +. d) /. d
  in
  let bound = Float.max (fraction 2. 3.) (fraction y0 y1) in
  for k = 1 to 999 do
    let l = 3. *. Float.of_int k /. 1000. in
    let exact = 3. -. (l /. 3.) in
    let inside =
      List.concat_map
        (fun r ->
          List.filter
            (fun (x, y) -> 2. < x && x < 3. && y0 < y && y < y1)
            (ring_pts r))
        (Pgon2.rings (region l f))
    in
    let msg = Printf.sprintf "level %.17g" l in
    if exact -. 2. > bound && 3. -. exact > bound then
      equal ~msg int 1 (List.length inside);
    List.iter
      (fun (x, _) ->
        at_most ~msg float_exact ~than:bound (Float.abs (x -. exact)))
      inside
  done

let mirrored () =
  let rows = [ [| 0.; 1.; 3. |]; [| 1.; 2.; 1. |]; [| 3.; 1.; 0. |] ] in
  let flip a =
    Array.init (Array.length a) (fun i -> a.(Array.length a - 1 - i))
  in
  let up = field rows in
  let left = field ~xs:[| 2.; 1.; 0. |] (List.map flip rows) in
  let down = field ~ys:[| 2.; 1.; 0. |] (List.rev rows) in
  let both =
    field ~xs:[| 2.; 1.; 0. |] ~ys:[| 2.; 1.; 0. |] (List.rev_map flip rows)
  in
  let areas f =
    List.map
      (fun (lo, hi) -> Pgon2.area (Field2.isoband ~lo ~hi f))
      [ (0.5, 1.5); (1.5, infinity) ]
  in
  List.iter
    (fun (name, f) -> equal ~msg:name (list area) (areas up) (areas f))
    [
      ("xs decreasing", left); ("ys decreasing", down); ("both decreasing", both);
    ]

(* The cell's ys are 2^52 and 2^52 + 1, one unit in the last place apart, so the
   crossings of levels 1 and 2 on its right side, 0.1 and 0.2 below its top
   right corner, round onto that corner: the band between them collapses onto
   the top side and encloses no area. *)
let collapsed_band () =
  let f =
    field ~ys:[| 0x1p52; 0x1p52 +. 1. |] [ [| 3.; 0. |]; [| 3.; 10. |] ]
  in
  same_rings [] (Field2.isoband ~lo:1. ~hi:2. f)

let domain_and_regions =
  group "domain and regions"
    [
      test "the sample of row i and column j lies at (xs.(j), ys.(i))" placement;
      cases "infinite samples are missing as NaN is"
        ~id:(fun v -> Printf.sprintf "%g" v)
        [ infinity; neg_infinity ] missing_alike;
      prop
        "the domain is the whole cells and the triangles of cells with one \
         missing sample"
        ~count:300 ~examples:domain_examples gen_sample domain_is_pieces;
      prop "isolines and regions lie in the grid's box" ~count:300
        ~examples:regressions gen_case in_grid_box;
      cases "R l of a field linear on each cell is where it is at least l"
        ~id:(fun (l, _) -> Printf.sprintf "l = %g" l)
        cone_regions
        (fun (l, a) -> equal area a (Pgon2.area (region l (Lazy.force cone))));
      cases
        "a saddle cell joins its in corners iff the mean of its samples is at \
         least l"
        ~id:(fun (n, _, _, _, _) -> n)
        saddles
        (fun (_, rows, l, n, a) ->
          let r = region l (field rows) in
          equal (pair int area) (n, a)
            (List.length (Pgon2.rings r), Pgon2.area r));
      test "a crossing is the in corner when its sample is l" pit_crossings;
      test "a polygon that encloses no area is dropped" collapsed_band;
      cases "samples at l with only lower ones around are not in R l"
        ~id:(fun (n, _, _) -> n)
        peaks
        (fun (_, c, rows) -> same_rings [] (region 1. (field ?xs:c ?ys:c rows)));
      test
        "a crossing lies where the samples reach l when their difference \
         overflows"
        overflowing_samples;
      prop
        "R l holds the centres of whole cells at or above l, not of those below"
        ~count:300 ~examples:regressions gen_case whole_cells;
      test "a crossing on a diagonal moves by at most its stated fraction"
        far_crossings_move_within_bound;
      test "decreasing coordinates mirror the regions" mirrored;
    ]

(* v *)

let invalid =
  [
    ("a vector", fun () -> Field2.v (Nx.zeros Nx.float64 [| 4 |]));
    ("three dimensions", fun () -> Field2.v (Nx.zeros Nx.float64 [| 2; 2; 2 |]));
    ("a complex tensor", fun () -> Field2.v (Nx.zeros Nx.complex64 [| 2; 2 |]));
    ("a boolean tensor", fun () -> Field2.v (Nx.zeros Nx.bool [| 2; 2 |]));
    ( "xs of another length",
      fun () -> field ~xs:[| 0.; 1. |] [ [| 0.; 0.; 0. |] ] );
    ("ys of another length", fun () -> field ~ys:[| 0. |] [ [| 0. |]; [| 0. |] ]);
    ("a NaN coordinate", fun () -> field ~xs:[| 0.; nan |] [ [| 0.; 0. |] ]);
    ( "an infinite coordinate",
      fun () -> field ~ys:[| 0.; infinity |] [ [| 0. |]; [| 0. |] ] );
    ( "repeated coordinates",
      fun () -> field ~xs:[| 0.; 1.; 1. |] [ [| 0.; 0.; 0. |] ] );
    ( "coordinates that turn back",
      fun () -> field ~xs:[| 0.; 2.; 1. |] [ [| 0.; 0.; 0. |] ] );
    ( "coordinates spanning more than the largest float",
      fun () -> field ~xs:[| -1e308; 1e308 |] [ [| 0.; 0. |] ] );
  ]

(* Each field's lower row is 2 and its upper 0: [R 1] is half the cell. *)
let real_dtypes =
  [
    ( "int32",
      fun () -> Field2.v (Nx.create Nx.int32 [| 2; 2 |] [| 0l; 0l; 2l; 2l |]) );
    ( "float16",
      fun () ->
        Field2.v (Nx.cast Nx.float16 (tensor [ [| 0.; 0. |]; [| 2.; 2. |] ])) );
  ]

let any_set () =
  let memory n = Result.get_ok (Rig.software n) in
  let module M = (val Nx.devices [ memory "m0" ]) in
  let module Two = (val Nx.devices [ memory "m1"; memory "m2" ]) in
  let z = tensor [ [| 0.; 0. |]; [| 2.; 2. |] ] in
  let half z = Pgon2.area (region 1. (Field2.v z)) in
  equal ~msg:"one device" area 0.5 (half (Nx.place M.on z));
  equal ~msg:"cut over two" area 0.5 (half (Nx.place (Two.split ~axis:0) z))

let own_elements () =
  let b = Bigarray.(Array1.of_array float64 c_layout [| 0.; 0.; 2.; 2. |]) in
  let f = Field2.v (Nx.Host.wrap [| 2; 2 |] b) in
  Bigarray.Array1.fill b 5.;
  equal area 0.5 (Pgon2.area (region 1. f))

let own_coordinates () =
  let xs = [| 0.; 1. |] in
  let f = field ~xs [ [| 0.; 0. |]; [| 0.; 0. |] ] in
  xs.(1) <- 10.;
  equal area 1. (Pgon2.area (domain f))

let no_cell =
  [
    ("one row", [ [| 0.; 1.; 2. |] ]);
    ("one column", [ [| 0. |]; [| 1. |]; [| 2. |] ]);
    ("no sample", []);
  ]

let constructing =
  group "v"
    [
      cases "raises on" ~id:fst invalid (fun (_, make) ->
          raises_match Exn.invalid_arg make);
      cases "converts the elements of a real dtype to floats" ~id:fst
        real_dtypes (fun (_, make) ->
          equal area 0.5 (Pgon2.area (region 1. (make ()))));
      test "reads a tensor of any set, cut over its devices" any_set;
      test "defaults xs and ys to the column and row indices" (fun () ->
          let f = field [ [| 0.; 0.; 0. |]; [| 0.; 0.; 0. |] ] in
          equal (option box2)
            (Some (Box2.v 0. 0. 2. 1.))
            (Pgon2.bounds (domain f)));
      test "keeps its own copy of the elements" own_elements;
      test "keeps its own copy of the coordinates" own_coordinates;
      cases "a field of fewer than two rows or columns has an empty domain"
        ~id:fst no_cell (fun (_, rows) -> same_rings [] (domain (field rows)));
    ]

(* Isolines and isobands *)

let deterministic (s, levels) =
  let f = sample_field s in
  let l = List.hd levels in
  equal pgon (region l f) (region l (sample_field s));
  equal path (Field2.isoline l f) (Field2.isoline l (sample_field s))

let golden_field =
  lazy
    (field
       [
         [| 0.; 1.; 2.; 1.; 0. |];
         [| 1.; 3.; 1.; 3.; 1. |];
         [| 2.; 1.; nan; 1.; 2. |];
         [| 1.; 0.; 1.; 2.; 3. |];
         [| 0.; 1.; 2.; 3.; 3. |];
       ])

let golden () =
  let f = Lazy.force golden_field in
  let pp_band (lo, hi) =
    Format.printf "@[<v 2>band [%g, %g):@,%a@]@." lo hi Pgon2.pp
      (Field2.isoband ~lo ~hi f)
  in
  List.iter pp_band [ (neg_infinity, 1.); (1., 2.); (2., infinity) ];
  List.iter
    (fun l ->
      Format.printf "@[<v 2>isoline %g:@,%a@]@." l Path.pp (Field2.isoline l f))
    [ 1.; 2.; 2.5 ];
  expect (output ())
  @@ __POS_OF__
       {|
    band [-inf, 1):
      M 0 1 L 0 0 L 1 0 Z M 3 0 L 4 0 L 4 1 Z M 0 3 L 1 2 L 2 3 L 1 4 L 0 4 Z
    band [1, 2):
      M 1 0.5 L 0.5 1 L 1 1.5 L 1.5 1 Z
      M 0 1 L 1 0 L 2 0 L 3 0 L 4 1 L 4 2 L 3 3 L 2 4 L 1 4 L 2 3 L 3 2 L 2 1
      L 1 2 L 0 3 L 0 2 Z M 3 0.5 L 2.5 1 L 3 1.5 L 3.5 1 Z
    band [2, inf):
      M 0.5 1 L 1 0.5 L 1.5 1 L 1 1.5 Z M 2.5 1 L 3 0.5 L 3.5 1 L 3 1.5 Z
      M 4 2 L 4 3 L 4 4 L 3 4 L 2 4 L 3 3 Z
    isoline 1:
      M 3 0 L 4 1 M 1 4 L 2 3 M 1 2 L 0 3 M 0 1 L 1 0
    isoline 2:
      M 0.5 1 L 1 0.5 L 1.5 1 L 1 1.5 Z M 2.5 1 L 3 0.5 L 3.5 1 L 3 1.5 Z M 2 4
      L 3 3 L 4 2
    isoline 2.5:
      M 0.75 1 L 1 0.75 L 1.25 1 L 1 1.25 Z M 2.75 1 L 3 0.75 L 3.25 1 L 3 1.25 Z
      M 2.5 4 L 3 3.5 L 3.5 3 L 4 2.5
    |}

let contours =
  group "isolines and isobands"
    [
      prop "order, first points and splits depend only on the arguments"
        ~count:300 ~examples:regressions gen_case deterministic;
      test "a field with a hole keeps its baseline bands and isolines" golden;
    ]

(* isoline *)

(* The isoline at [l] is the boundary of [R l] off the domain's boundary: its
   segments are segments of [R l], the same way, and reversed of the band below
   [l]; the other segments of [R l] lie on the domain's boundary. *)
let bounds_region (s, levels) =
  let f = sample_field s in
  let l = List.hd levels in
  let segs = segments_of_curves (curves (Field2.isoline l f)) in
  let above = segments_of_rings (region l f) in
  let below = segments_of_rings (Field2.isoband ~lo:neg_infinity ~hi:l f) in
  let dom = segments_of_rings (domain f) in
  List.iter
    (fun ((a, b) as seg) ->
      if not (List.mem seg above) then
        failf "isoline segment %a is not on R l" pp_pts [ a; b ];
      if not (List.exists (lies_on (b, a)) below) then
        failf "isoline segment %a reversed is not on the band below" pp_pts
          [ a; b ])
    segs;
  List.iter
    (fun seg ->
      if not (List.mem seg segs || List.exists (lies_on seg) dom) then
        failf
          "segment %a of R l is neither the isoline nor the domain's boundary"
          pp_pts
          [ fst seg; snd seg ])
    above

let curve_points (s, levels) =
  let f = sample_field s in
  List.iter
    (fun (pts, closed) ->
      let n = List.length pts in
      at_least ~msg:"points of a curve" int ~than:(if closed then 3 else 2) n;
      let a = Array.of_list pts in
      let m = if closed then n else n - 1 in
      for i = 0 to m - 1 do
        if a.(i) = a.((i + 1) mod n) then
          failf "a curve repeats a point %a" pp_pts pts
      done)
    (curves (Field2.isoline (List.hd levels) f))

let curves_simple (s, levels) =
  let f = sample_field s in
  check_simple "isoline"
    (segments_of_curves (curves (Field2.isoline (List.hd levels) f)))

(* The end points of an open curve lie on the domain's boundary. *)
let open_ends (s, levels) =
  let f = sample_field s in
  let cs = curves (Field2.isoline (List.hd levels) f) in
  cover "an open curve" (List.exists (fun (_, closed) -> not closed) cs);
  let boundary = segments_of_rings (domain f) in
  List.iter
    (fun (pts, closed) ->
      if not closed then
        List.iter
          (fun pt ->
            if not (List.exists (near_segment pt) boundary) then
              failf "the open curve %a ends off the domain's boundary" pp_pts
                pts)
          [ List.hd pts; List.nth pts (List.length pts - 1) ])
    cs

(* Field values are the column index, cut by a NaN at the centre: the line x =
   2.5 runs upwards, [R 2.5] on its right, and stops at the hole the NaN cuts,
   the diamond through (2, 1), (3, 2), (2, 3) and (1, 2). *)
let open_at_a_hole () =
  let f =
    field
      (List.init 5 (fun i ->
           Array.init 5 (fun j ->
               if i = 2 && j = 2 then nan else Float.of_int j)))
  in
  same_curves
    [
      ([ (2.5, 4.); (2.5, 3.); (2.5, 2.5) ], false);
      ([ (2.5, 1.5); (2.5, 1.); (2.5, 0.) ], false);
    ]
    (Field2.isoline 2.5 f)

(* Agreement with d3-contour on its test fields, at 0.5, none of which has a
   saddle. d3 puts samples at half-integers and winds outer rings the other way;
   its rings repeat their first point. *)

let d3_ring pts =
  (* Reversed, without the repeated point. *)
  match List.rev pts with
  | _ :: rest -> List.map (fun (x, y) -> (x -. 0.5, y -. 0.5)) rest
  | [] -> []

let grid10 rows =
  field (List.map (fun r -> Array.of_list (List.map Float.of_int r)) rows)

let d3_cases =
  let z = [ 0; 0; 0; 0; 0; 0; 0; 0; 0; 0 ] in
  [
    ( "simple polygon",
      [
        z;
        z;
        z;
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        z;
        z;
      ],
      [
        [
          (6., 7.5);
          (6., 6.5);
          (6., 5.5);
          (6., 4.5);
          (6., 3.5);
          (5.5, 3.);
          (4.5, 3.);
          (3.5, 3.);
          (3., 3.5);
          (3., 4.5);
          (3., 5.5);
          (3., 6.5);
          (3., 7.5);
          (3.5, 8.);
          (4.5, 8.);
          (5.5, 8.);
          (6., 7.5);
        ];
      ] );
    ( "polygon with a hole",
      [
        z;
        z;
        z;
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 0; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 0; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 0; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        z;
        z;
      ],
      [
        [
          (6., 7.5);
          (6., 6.5);
          (6., 5.5);
          (6., 4.5);
          (6., 3.5);
          (5.5, 3.);
          (4.5, 3.);
          (3.5, 3.);
          (3., 3.5);
          (3., 4.5);
          (3., 5.5);
          (3., 6.5);
          (3., 7.5);
          (3.5, 8.);
          (4.5, 8.);
          (5.5, 8.);
          (6., 7.5);
        ];
        [
          (4.5, 7.);
          (4., 6.5);
          (4., 5.5);
          (4., 4.5);
          (4.5, 4.);
          (5., 4.5);
          (5., 5.5);
          (5., 6.5);
          (4.5, 7.);
        ];
      ] );
    ( "multipolygon",
      [
        z;
        z;
        z;
        [ 0; 0; 0; 1; 1; 0; 1; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 0; 1; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 0; 1; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 0; 1; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 0; 1; 0; 0; 0 ];
        z;
        z;
      ],
      [
        [
          (5., 7.5);
          (5., 6.5);
          (5., 5.5);
          (5., 4.5);
          (5., 3.5);
          (4.5, 3.);
          (3.5, 3.);
          (3., 3.5);
          (3., 4.5);
          (3., 5.5);
          (3., 6.5);
          (3., 7.5);
          (3.5, 8.);
          (4.5, 8.);
          (5., 7.5);
        ];
        [
          (7., 7.5);
          (7., 6.5);
          (7., 5.5);
          (7., 4.5);
          (7., 3.5);
          (6.5, 3.);
          (6., 3.5);
          (6., 4.5);
          (6., 5.5);
          (6., 6.5);
          (6., 7.5);
          (6.5, 8.);
          (7., 7.5);
        ];
      ] );
    ( "multipolygon with holes",
      [
        z;
        z;
        z;
        [ 0; 1; 1; 1; 0; 1; 1; 1; 0; 0 ];
        [ 0; 1; 0; 1; 0; 1; 0; 1; 0; 0 ];
        [ 0; 1; 1; 1; 0; 1; 1; 1; 0; 0 ];
        z;
        z;
        z;
        z;
      ],
      [
        [
          (4., 5.5);
          (4., 4.5);
          (4., 3.5);
          (3.5, 3.);
          (2.5, 3.);
          (1.5, 3.);
          (1., 3.5);
          (1., 4.5);
          (1., 5.5);
          (1.5, 6.);
          (2.5, 6.);
          (3.5, 6.);
          (4., 5.5);
        ];
        [ (2.5, 5.); (2., 4.5); (2.5, 4.); (3., 4.5); (2.5, 5.) ];
        [
          (8., 5.5);
          (8., 4.5);
          (8., 3.5);
          (7.5, 3.);
          (6.5, 3.);
          (5.5, 3.);
          (5., 3.5);
          (5., 4.5);
          (5., 5.5);
          (5.5, 6.);
          (6.5, 6.);
          (7.5, 6.);
          (8., 5.5);
        ];
        [ (6.5, 5.); (6., 4.5); (6.5, 4.); (7., 4.5); (6.5, 5.) ];
      ] );
  ]

(* The cone's isoline at 2.5 is the diamond of radius 1.5 around its peak, area
   2 * 1.5^2 = 4.5, positive since higher values lie inside. *)
let around_higher () =
  let cs = curves (Field2.isoline 2.5 (Lazy.force cone)) in
  equal
    (list (pair bool area))
    [ (true, 4.5) ]
    (List.map (fun (pts, closed) -> (closed, signed_area pts)) cs)

(* Isolines through samples equal to the level, walked with [R l] on their
   right. *)
let through_corners =
  [
    ( "a plateau's edge",
      [ [| 2.; 2.; 2. |]; [| 1.; 1.; 1. |]; [| 0.; 0.; 0. |] ],
      [ ([ (2., 1.); (1., 1.); (0., 1.) ], false) ] );
    ( "a cell's diagonal",
      [ [| 1.; 0. |]; [| 2.; 1. |] ],
      [ ([ (0., 0.); (1., 1.) ], false) ] );
  ]

(* The saddles of [saddles]: joined at 0.5, the isolines cut off the out
   corners; split at 0.6, the in corners. *)
let saddle_curves =
  [
    ( "joined",
      0.5,
      [ ([ (0.5, 0.); (1., 0.5) ], false); ([ (0.5, 1.); (0., 0.5) ], false) ]
    );
    ( "split",
      0.6,
      [ ([ (0.4, 0.); (0., 0.4) ], false); ([ (0.6, 1.); (1., 0.6) ], false) ]
    );
  ]

let constant = [ [| 1.; 1.; 1. |]; [| 1.; 1.; 1. |] ]

let no_isoline =
  [
    ("a ridge at l", ridge, 1.);
    ("a border row at l", border_row, 1.);
    ("an isolated peak at l", peak, 1.);
    ("a constant field at l", constant, 1.);
    ("a constant field above l", constant, 0.);
  ]

let isoline =
  group "isoline"
    [
      prop "is the boundary of R l off the domain's boundary" ~count:300
        ~examples:regressions gen_case bounds_region;
      prop "has open curves that end on the domain's boundary" ~count:300
        ~examples:regressions gen_case open_ends;
      test "has an open curve that ends at a hole" open_at_a_hole;
      cases "has closed curves around regions off the domain's boundary"
        ~id:(fun (n, _, _) -> n)
        d3_cases
        (fun (_, rows, _) ->
          let cs = curves (Field2.isoline 0.5 (grid10 rows)) in
          equal (list bool) (List.map (fun _ -> true) cs) (List.map snd cs));
      test "has closed curves around higher values positively oriented"
        around_higher;
      cases "runs with R l on its right through samples equal to l"
        ~id:(fun (n, _, _) -> n)
        through_corners
        (fun (_, rows, expected) ->
          same_curves expected (Field2.isoline 1. (field rows)));
      cases
        "cuts off a saddle's out corners when joined, its in corners when split"
        ~id:(fun (n, _, _) -> n)
        saddle_curves
        (fun (_, l, expected) ->
          same_curves expected
            (Field2.isoline l (field [ [| 1.; 0. |]; [| 0.; 1. |] ])));
      prop
        "has curves of two points or more, three if closed, consecutive ones \
         distinct"
        ~count:300 ~examples:regressions gen_case curve_points;
      prop "has curves that neither cross nor overlap" ~count:300
        ~examples:regressions gen_case curves_simple;
      cases "is empty where R l has no area or is the domain"
        ~id:(fun (n, _, _) -> n)
        no_isoline
        (fun (_, rows, l) ->
          equal path Path.empty (Field2.isoline l (field rows)));
      cases "is empty at an infinite level"
        ~id:(fun l -> Printf.sprintf "%g" l)
        [ infinity; neg_infinity ]
        (fun l -> equal path Path.empty (Field2.isoline l (Lazy.force cone)));
      test "raises on a NaN level" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Field2.isoline nan (Lazy.force cone)));
    ]

(* isoband *)

(* The bands of [levels] with open ends tile the domain: their areas add up to
   the domain's, and a point, decided by the conventions' rule for points on
   rings, is in exactly one band if it is in the domain and in none otherwise.
   Near a cut cell's diagonal, where bands bend at their crossings and the
   domain does not, a point is in at most one band, unless two levels lie within
   rounding error of each other. *)
let bands_tile (s, levels) =
  let f = sample_field s in
  cover "a missing sample"
    (List.exists (Array.exists (fun v -> not (Float.is_finite v))) s.rows);
  cover "a sample equal to a level"
    (List.exists (Array.exists (fun v -> List.mem v levels)) s.rows);
  cover "two adjacent levels"
    (List.exists (fun l -> List.mem (Float.succ l) levels) levels);
  let bands = bands levels f in
  let d = domain f in
  let total = List.fold_left (fun a b -> a +. Pgon2.area b) 0. bands in
  equal ~msg:"areas add up to the domain's" area (Pgon2.area d) total;
  let vertices =
    List.concat_map (fun b -> List.concat_map ring_pts (Pgon2.rings b)) bands
  in
  let diagonals = cut_diagonals s and close = close_levels levels in
  List.iter
    (fun (x, y) ->
      let p = P2.v x y in
      let n = List.length (List.filter (Winding.mem p) bands) in
      if not (List.exists (near_segment (x, y)) diagonals) then begin
        let expected = if Winding.mem p d then 1 else 0 in
        if n <> expected then
          failf "(%.17g, %.17g) is in %d bands, %d expected" x y n expected
      end
      else if n > 1 && not close then
        failf "(%.17g, %.17g) is in %d bands" x y n)
    (probes s f @ vertices)

let ring_points case =
  let _, _, _, band = band_of case in
  List.iter
    (fun r ->
      let pts = ring_pts r in
      let n = List.length pts in
      at_least ~msg:"points of a ring" int ~than:3 n;
      let a = Array.of_list pts in
      for i = 0 to n - 1 do
        if a.(i) = a.((i + 1) mod n) then
          failf "a ring repeats a point %a" pp_pts pts
      done)
    (Pgon2.rings band)

(* Near a cut cell's diagonal, rings of levels within rounding error of each
   other may cross. *)
let rings_simple ((s, _) as case) =
  let _, lo, hi, band = band_of case in
  check_simple
    ~exempt:(fun (a, b) (c, d) -> near_close_diagonal s lo hi [ a; b; c; d ])
    "band" (segments_of_rings band)

let rings_wind ((s, _) as case) =
  let f, lo, hi, band = band_of case in
  List.iter
    (fun pt ->
      if not (near_close_diagonal s lo hi [ pt ]) then
        let w = winding band (P2.v (fst pt) (snd pt)) in
        if w <> 0 && w <> 1 then
          failf "the rings wind %d times around (%.17g, %.17g)" w (fst pt)
            (snd pt))
    (probes s f)

(* Every segment of a band lies on the domain's boundary, on the isoline of its
   lower level the same way, or on that of its upper level reversed. *)
let rings_follow case =
  let f, lo, hi, band = band_of case in
  let dom = segments_of_rings (domain f) in
  let lo_line = segments_of_curves (curves (Field2.isoline lo f)) in
  let hi_line =
    List.map
      (fun (a, b) -> (b, a))
      (segments_of_curves (curves (Field2.isoline hi f)))
  in
  List.iter
    (fun seg ->
      let on l = List.exists (lies_on seg) l in
      if not (on dom || on lo_line || on hi_line) then
        failf "segment %a is on neither the domain's boundary nor an isoline"
          pp_pts
          [ fst seg; snd seg ])
    (segments_of_rings band)

(* Levels a few units in the last place apart whose crossings on the diagonal
   round to neighbouring multiples of the step, so that they lie a step apart.
   Their bands overlap with a step of 8 or 32 rounding errors and stay apart
   with 256. *)
let step_cases =
  [
    ( [| 0.797302913532759; 1.0679835797987471 |],
      [| 1.3616971934929403; 3.155449259613466 |],
      [| 2.18277012134777; nan |],
      [| 0.19155382558576306; 2.1502660026096754 |],
      [ 2.1667778990365343; 2.1667778990365347; 2.1667778990366466 ] );
    ( [| -2.14711828494428; -1.0354536283897178 |],
      [| 0.9115560817856547; 1.834905994438033 |],
      [| nan; 1.3185791774140974 |],
      [| 1.277457855137686; 2.6177877563401766 |],
      [ 1.2880633573784808; 1.288063357378481 ] );
    ( [| 2.494447018285971; 3.109902385233981 |],
      [| -0.13578220799741825; 0.5133614194118594 |],
      [| nan; 0.9213107305341494 |],
      [| 0.9101629531291233; 2.887993084758688 |],
      [ 0.9117739595635503; 0.9117739595635516 ] );
  ]

let step_keeps_bands_apart (xs, ys, top, bottom, levels) =
  let bs = bands levels (field ~xs ~ys [ top; bottom ]) in
  let count x y = List.length (List.filter (Winding.mem (P2.v x y)) bs) in
  List.iter
    (fun b ->
      List.iter
        (fun r ->
          let n = Ring2.length r in
          for i = 0 to n - 1 do
            let x = Ring2.x r i and y = Ring2.y r i in
            let j = (i + 1) mod n in
            let mx = (x +. Ring2.x r j) /. 2. in
            let my = (y +. Ring2.y r j) /. 2. in
            at_most ~msg:"a vertex" int ~than:1 (count x y);
            at_most ~msg:"a midpoint" int ~than:1 (count mx my)
          done)
        (Pgon2.rings b))
    bs

let invalid_bounds =
  [ ("lo NaN", nan, 1.); ("hi NaN", 0., nan); ("lo > hi", 2., 1.) ]

let isoband =
  group "isoband"
    [
      cases "agrees with d3-contour's rings"
        ~id:(fun (n, _, _) -> n)
        d3_cases
        (fun (_, rows, rings) ->
          same_rings (List.map d3_ring rings) (region 0.5 (grid10 rows)));
      prop
        "bands of increasing levels tile the domain away from cut cells' \
         diagonals"
        ~count:300 ~examples:regressions gen_case bands_tile;
      prop "has rings that wind 0 or 1 times around every point off them"
        ~count:300 ~examples:regressions gen_case rings_wind;
      prop "has rings of three points or more, consecutive ones distinct"
        ~count:300 ~examples:regressions gen_case ring_points;
      prop "has rings that neither cross nor overlap" ~count:300
        ~examples:regressions gen_case rings_simple;
      prop
        "has segments on the domain's boundary or on the isolines of its levels"
        ~count:300 ~examples:regressions gen_case rings_follow;
      cases
        "bands of levels whose diagonal crossings lie apart do not overlap \
         there"
        ~id:(fun (xs, _, _, _, _) -> Printf.sprintf "xs.(0) = %g" xs.(0))
        step_cases step_keeps_bands_apart;
      cases "is empty when lo = hi"
        ~id:(fun l -> Printf.sprintf "%g" l)
        [ 2.; infinity; neg_infinity ]
        (fun l -> same_rings [] (Field2.isoband ~lo:l ~hi:l (Lazy.force cone)));
      cases "raises on a NaN bound or lo > hi"
        ~id:(fun (n, _, _) -> n)
        invalid_bounds
        (fun (_, lo, hi) ->
          raises_match Exn.invalid_arg (fun () ->
              Field2.isoband ~lo ~hi (Lazy.force cone)));
    ]

(* The reference

   The laws decide points on rings with [Winding], in test support: these
   winding numbers follow from its rule, a point on a ring decided as the point
   moved from it towards positive x, then infinitely less towards positive y.
   The slanted rows put the point within rounding error of an edge, on the side
   its exact orientation gives and the rounded cross product does not; the
   triangles there are positively oriented. *)

let unit_square = [ (0., 0.); (1., 0.); (1., 1.); (0., 1.) ]
let twice ring = ring @ ring
let left_of (l, u) = [ l; u; (-40., snd u) ]
let right_of (l, u) = [ u; l; (40., snd l) ]

let edge_a =
  ( (-21.938145353255926, 20.846024216233957),
    (15.826477138596843, 31.793716220071403) )

let edge_b =
  ( (-0.0005240707458162172, 8.845845059190371e-05),
    (21.098654996442377, 28.117601157885833) )

let windings =
  [
    ("square, centre", unit_square, (0.5, 0.5), 1);
    ("reversed square, centre", List.rev unit_square, (0.5, 0.5), -1);
    ("square, a point right of it", unit_square, (2., 0.5), 0);
    ("square, a point above it", unit_square, (0.5, -1.), 0);
    ("square, top left vertex", unit_square, (0., 0.), 1);
    ("reversed square, top left vertex", List.rev unit_square, (0., 0.), -1);
    ("square, top right vertex", unit_square, (1., 0.), 0);
    ("square, bottom right vertex", unit_square, (1., 1.), 0);
    ("square, bottom left vertex", unit_square, (0., 1.), 0);
    ("square, top edge", unit_square, (0.5, 0.), 1);
    ("square, left edge", unit_square, (0., 0.5), 1);
    ("square, bottom edge", unit_square, (0.5, 1.), 0);
    ("square, right edge", unit_square, (1., 0.5), 0);
    ("two points, a point between them", [ (0., 0.); (1., 1.) ], (0.5, 0.5), 0);
    ("square traversed twice, centre", twice unit_square, (0.5, 0.5), 2);
    ( "slanted edge, triangle on the left",
      [ (0.1, 0.2); (0.7, 1.3); (-1., 1.3) ],
      (0.49056068382391227, 0.9160279203438393),
      1 );
    ( "slanted edge, triangle on the right",
      [ (0.7, 1.3); (0.1, 0.2); (1.5, 0.2) ],
      (0.49056068382391227, 0.9160279203438393),
      0 );
    ( "long edge, triangle on the left",
      left_of edge_a,
      (-3.228226320010573, 26.26989495761091),
      0 );
    ( "long edge, triangle on the right",
      right_of edge_a,
      (-3.228226320010573, 26.26989495761091),
      1 );
    ( "edge from near the origin, triangle on the left",
      left_of edge_b,
      (13.201660671605467, 17.59378705550328),
      1 );
    ( "edge from near the origin, triangle on the right",
      right_of edge_b,
      (13.201660671605467, 17.59378705550328),
      0 );
  ]

let reference =
  cases "Winding.ring counts windings by its rule for points on rings"
    ~id:(fun (n, _, _, _) -> n)
    windings
    (fun (_, pts, (x, y), w) ->
      let r =
        Ring2.v
          (Array.of_list (List.map fst pts))
          (Array.of_list (List.map snd pts))
      in
      equal int w (Winding.ring (P2.v x y) r))

(* Seven triangles around a common centre share their slanted edges: a point on
   an edge two of them share lies in at most one, whatever the rounding of that
   edge, as [bands_tile] needs of adjacent bands. *)
let fan_shares_edges (k, t) =
  let n = 7 in
  let px i =
    let a = 2. *. Float.pi *. Float.of_int i /. Float.of_int n in
    (3. *. Float.cos a, 3. *. Float.sin a)
  in
  let tri i =
    let (x0, y0), (x1, y1) = (px i, px ((i + 1) mod n)) in
    Pgon2.v [ Ring2.v [| 0.1; x0; x1 |] [| 0.2; y0; y1 |] ]
  in
  let x1, y1 = px k in
  let p = P2.v (0.1 +. (t *. (x1 -. 0.1))) (0.2 +. (t *. (y1 -. 0.2))) in
  at_most int ~than:1
    (List.length (List.filter (Winding.mem p) (List.init n tri)))

let fan =
  prop "Winding.mem holds a point on an edge two triangles share in at most one"
    (Gen.pair (Gen.int_range 0 6) (Gen.float_range 0. 1.))
    fan_shares_edges

let () =
  exit
    (run "Field2"
       [
         domain_and_regions;
         constructing;
         contours;
         isoline;
         isoband;
         reference;
         fan;
       ])
