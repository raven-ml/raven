(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord

let utm zone north = Crs.utm ~zone ~north
let host = Nx.Host.on

(* The pixel points of [src] at [dst]'s centres, as a caller composes them. *)
let points ~src ~dst = Warp.pixel src (Warp.centres ~at:host src.Grid.crs dst)

let north_up crs ~rows ~cols ~x ~y d =
  Grid.v crs ~rows ~cols ~origin:(x, y) ~row_step:(0., -.d) ~col_step:(d, 0.)

(* The grid of [x]'s first two axes, of unit cells in UTM 33N, whose pixel
   points are their own points. *)
let unit_grid x =
  north_up (utm 33 true) ~rows:(Nx.dim 0 x) ~cols:(Nx.dim 1 x) ~x:0. ~y:0. 1.

let pixels l =
  Nx.create Nx.float64
    [| List.length l; 2 |]
    (Array.of_list (List.concat_map (fun (a, b) -> [ a; b ]) l))

let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let read (v, ok) = (floats v, Nx.to_array ok)
let bits a = Array.map Int64.bits_of_float (floats a)
let result = pair (array float_exact) (array bool)

(* The sources of warp_gdal.dat: value (i, j) of a grid, in [1, m], and the
   cells that are holes. *)
let value m i j =
  float_of_int ((((i * 7919) + (j * 104729) + (i * j * 31)) mod m) + 1)

let hole i j = ((i * 13) + (j * 7)) mod 11 = 0

let source m (g : Grid.t) =
  let shape = [| g.rows; g.cols |] in
  ( Nx.init Nx.float64 shape (fun a -> value m a.(0) a.(1)),
    Nx.init Nx.bool shape (fun a -> not (hole a.(0) a.(1))) )

(* GDAL's warps *)

type fixture = {
  name : string;
  src : Grid.t;
  dst : Grid.t;
  outputs : (string * float array) list;
}

(* warp_gdal.dat and warp_area.dat: for each case, a line [case name], the
   source's and the destination's grid as [src|dst epsg rows cols ox oy rx ry
   cx cy], then each output's name on a line and its rows, NaN where GDAL
   wrote no data. A [source] block has the source's rows. *)
let read_fixtures file =
  let lines =
    In_channel.with_open_text file In_channel.input_lines
    |> List.filter (fun l -> l <> "" && l.[0] <> '#')
  in
  let words l = String.split_on_char ' ' l in
  let grid l =
    match words l with
    | [ _; epsg; rows; cols; ox; oy; rx; ry; cx; cy ] ->
        let f = float_of_string in
        Grid.v
          (Result.get_ok (Crs.of_epsg (int_of_string epsg)))
          ~rows:(int_of_string rows) ~cols:(int_of_string cols)
          ~origin:(f ox, f oy) ~row_step:(f rx, f ry) ~col_step:(f cx, f cy)
    | _ -> failwith (file ^ ": not a grid: " ^ l)
  in
  let rec outputs (src : Grid.t) (dst : Grid.t) acc = function
    | name :: rest when not (String.contains name ' ') ->
        let n = if name = "source" then src.rows else dst.rows in
        let rows = List.filteri (fun i _ -> i < n) rest in
        let rest = List.filteri (fun i _ -> i >= n) rest in
        let v = Array.of_list (List.concat_map words rows) in
        outputs src dst ((name, Array.map float_of_string v) :: acc) rest
    | rest -> (List.rev acc, rest)
  in
  let rec cases acc = function
    | [] -> List.rev acc
    | c :: s :: d :: rest when String.starts_with ~prefix:"case " c ->
        let src = grid s and dst = grid d in
        let outputs, rest = outputs src dst [] rest in
        let name = String.sub c 5 (String.length c - 5) in
        cases ({ name; src; dst; outputs } :: acc) rest
    | l :: _ -> failwith (file ^ ": unexpected line: " ^ l)
  in
  cases [] lines

let fixtures = read_fixtures "warp_gdal.dat"
let fixture name = List.find (fun f -> f.name = name) fixtures

(* jord's result for each of a fixture's outputs, NaN where the mask is
   false: nearest on [x], bilinear on [x] as float64, and [rounded],
   bilinear on [x] in its own dtype. *)
let jord ~valid src p x =
  let nan_off (v, ok) =
    let v, ok = read (v, ok) in
    Array.mapi (fun k v -> if ok.(k) then v else Float.nan) v
  in
  [
    ("nearest", nan_off (Warp.nearest ~valid src p x));
    ("bilinear", nan_off (Warp.bilinear ~valid src p (Nx.cast Nx.float64 x)));
    ("rounded", nan_off (Warp.bilinear ~valid src p x));
  ]

(* jord's points and PROJ's under GDAL agree to within 1.25e-9 source pixel
   (nanometres on cells of 10 m or more), and a bilinear value moves by at most
   twice its source's largest difference between valid neighbouring cells
   times that. *)
let parity_bound x valid =
  let rows = Nx.dim 0 x and cols = Nx.dim 1 x in
  let at i j =
    if i >= 0 && j >= 0 && i < rows && j < cols && Nx.item [ i; j ] valid then
      Some (Nx.item [ i; j ] x)
    else None
  in
  let largest = ref 0. in
  for i = 0 to rows - 1 do
    for j = 0 to cols - 1 do
      List.iter
        (fun (di, dj) ->
          match (at i j, at (i + di) (j + dj)) with
          | Some a, Some b -> largest := Float.max !largest (Float.abs (a -. b))
          | _ -> ())
        [ (0, 1); (1, 0); (1, 1); (1, -1) ]
    done
  done;
  2. *. 1.25e-9 *. !largest

(* Each case of warp_gdal.dat and the dtype of its source. *)
let parity_cases =
  [
    ("zones", `Float);
    ("tm35fin", `Float);
    ("to_tm35fin", `Float);
    ("to_lonlat", `Float);
    ("antimeridian", `Float);
    ("lattice", `Uint16);
    ("int16lattice", `Int16);
  ]

let gdal_parity =
  cases "GDAL -et 0" ~id:fst parity_cases (fun (name, kind) ->
      let f = fixture name in
      let p = points ~src:f.src ~dst:f.dst in
      let x, valid = source (if kind = `Float then 1009 else 12000) f.src in
      let bound = parity_bound x valid in
      let outputs =
        match kind with
        | `Float -> jord ~valid f.src p x
        | `Uint16 -> jord ~valid f.src p (Nx.cast Nx.uint16 x)
        | `Int16 -> jord ~valid f.src p (Nx.cast Nx.int16 (Nx.sub_s x 6000.))
      in
      List.iter
        (fun (output, gdal) ->
          let mine = List.assoc output outputs in
          let has = Array.map (fun v -> not (Float.is_nan v)) in
          equal ~msg:(output ^ "'s mask") (array bool) (has gdal) (has mine);
          let close =
            if output = "nearest" then float_exact else float bound
          in
          Array.iteri
            (fun k g ->
              if not (Float.is_nan g) then equal ~msg:output close g mine.(k))
            gdal)
        f.outputs;
      if kind = `Int16 then
        let ties =
          List.filter
            (fun v -> v < 0. && v -. Float.floor v = 0.5)
            (Array.to_list (List.assoc "bilinear" outputs))
        in
        greater ~msg:"negative ties" int ~than:0 (List.length ties))

(* The lattice *)

(* The four cells around pixel point (r, c), weighted and summed as written,
   over the valid ones of positive weight: the reference for a lattice, where
   the weights and sums of integers are exact in any order. *)
let stencil x valid (r, c) =
  let u = r -. 0.5 and v = c -. 0.5 in
  let i0 = Float.floor u and j0 = Float.floor v in
  let a = u -. i0 and b = v -. j0 in
  let sum = ref 0. and weight = ref 0. in
  List.iter
    (fun (di, wi) ->
      List.iter
        (fun (dj, wj) ->
          let i = int_of_float i0 + di and j = int_of_float j0 + dj in
          let w = wi *. wj in
          let inside = i >= 0 && j >= 0 && i < Nx.dim 0 x && j < Nx.dim 1 x in
          if w > 0. && inside && Nx.item [ i; j ] valid then (
            sum := !sum +. (w *. Nx.item [ i; j ] x);
            weight := !weight +. w))
        [ (0, 1. -. b); (1, b) ])
    [ (0, 1. -. a); (1, a) ];
  !sum /. !weight

(* A 20 m grid of int16 values of either sign with holes, and a 10 m grid
   whose origin lies on its lattice, overlapping it. *)
let lattice_pair =
  Gen.(
    let+ kx = int_range 10000 30000
    and+ ky = int_range 250000 300000
    and+ rows, cols = pair (int_range 1 8) (int_range 1 8)
    and+ dx, dy = pair (int_range (-3) 10) (int_range (-3) 10)
    and+ rows', cols' = pair (int_range 1 10) (int_range 1 10)
    and+ seed = int_range 0 10000 in
    let x = float_of_int (20 * kx) and y = float_of_int (20 * ky) in
    let src = north_up (utm 31 true) ~rows ~cols ~x ~y 20. in
    let dst =
      north_up (utm 31 true) ~rows:rows' ~cols:cols'
        ~x:(x +. float_of_int (10 * dx))
        ~y:(y -. float_of_int (10 * dy))
        10.
    in
    (src, dst, seed))

let lattice =
  prop "a lattice warp is its stencil exactly, and integers round as floor"
    lattice_pair (fun ((src : Grid.t), dst, seed) ->
      let hash i j = (((i * 7919) + (j * 104729) + seed) * 31) land 0xffff in
      let shape = [| src.rows; src.cols |] in
      let value a = float_of_int ((hash a.(0) a.(1) mod 601) - 300) in
      let x = Nx.init Nx.float64 shape value in
      let hole a = hash a.(1) a.(0) mod 5 = 0 in
      let valid = Nx.init Nx.bool shape (fun a -> not (hole a)) in
      let p = points ~src ~dst in
      let pts = Nx.to_array p in
      Array.iter
        (fun r ->
          let frac = r -. 0.5 -. Float.floor (r -. 0.5) in
          equal ~msg:"fraction" bool true (frac = 0.25 || frac = 0.75))
        pts;
      let mean, ok = read (Warp.bilinear ~valid src p x) in
      let rounded, _ =
        read (Warp.bilinear ~valid src p (Nx.cast Nx.int16 x))
      in
      Array.iteri
        (fun k ok ->
          if ok then (
            let want = stencil x valid (pts.(2 * k), pts.((2 * k) + 1)) in
            cover "a tie" (want -. Float.floor want = 0.5);
            equal ~msg:"mean" float_exact want mean.(k);
            let half_up = Float.floor (want +. 0.5) in
            equal ~msg:"int16" float_exact half_up rounded.(k)))
        ok)

let rounding =
  cases "integers round as floor (v + 1/2), on every sign"
    ~id:(fun (a, b, c, _) -> Printf.sprintf "[%d; %d] at %g" a b c)
    [
      (-3, -2, 1., -2.);
      (2, 3, 1., 3.);
      (-1, 0, 1., 0.);
      (* The mean is 100.499998, which float32 would round to 100.5. *)
      (100, 101, 1. -. 2e-6, 100.);
    ]
    (fun (a, b, c, expected) ->
      let x = Nx.create Nx.int16 [| 1; 2 |] [| a; b |] in
      let v, _ = read (Warp.bilinear (unit_grid x) (pixels [ (0.5, c) ]) x) in
      equal float_exact expected v.(0))

let float32_once =
  test "a float32 mean rounds once, from float64" (fun () ->
      let x = Nx.create Nx.float32 [| 1; 2 |] [| 100.; 101. |] in
      let v, _ =
        read (Warp.bilinear (unit_grid x) (pixels [ (0.5, 1. -. 2e-6) ]) x)
      in
      equal float_exact 100.5 v.(0))

(* Laws of the samplers *)

(* An array on a small grid, values with NaN, -0. and infinities, a mask, and
   pixel points within a cell of the grid's edge, NaN, borders and centres
   among them. *)
let field =
  Gen.(
    let* rows, cols = pair (int_range 1 5) (int_range 1 5) in
    let n = rows * cols in
    let num =
      frequency
        [
          (12, float_range (-100.) 100.);
          (1, of_list [ Float.nan; -0.; Float.infinity; Float.neg_infinity ]);
        ]
    in
    let coord m =
      frequency
        [
          (8, float_range (-1.) (float_of_int m +. 1.));
          (2, map float_of_int (int_range (-1) (m + 1)));
          (1, map (fun k -> float_of_int k +. 0.5) (int_range (-1) m));
          (1, constant Float.nan);
        ]
    in
    let mostly = frequency [ (4, constant true); (1, constant false) ] in
    let+ x = array ~size:(constant n) num
    and+ valid = array ~size:(constant n) mostly
    and+ p = list ~size:(int_range 1 12) (pair (coord rows) (coord cols)) in
    ( Nx.create Nx.float64 [| rows; cols |] x,
      Nx.create Nx.bool [| rows; cols |] valid,
      pixels p ))

let masked_never_matter =
  prop "a cell that is not valid never changes a value" field
    (fun (x, valid, p) ->
      let nan = Nx.where valid x (Nx.full_like x Float.nan) in
      let zero = Nx.where valid x (Nx.zeros_like x) in
      equal ~msg:"nearest" result
        (read (Warp.nearest ~valid (unit_grid x) p nan))
        (read (Warp.nearest ~valid (unit_grid x) p zero));
      equal ~msg:"bilinear" result
        (read (Warp.bilinear ~valid (unit_grid x) p nan))
        (read (Warp.bilinear ~valid (unit_grid x) p zero)))

(* The cell containing each point, if it is inside and valid. *)
let containing x valid p =
  let rows = float_of_int (Nx.dim 0 x) and cols = float_of_int (Nx.dim 1 x) in
  let p = Nx.to_array p in
  List.init (Array.length p / 2) (fun k ->
      let i = Float.floor p.(2 * k) and j = Float.floor p.((2 * k) + 1) in
      if not (i >= 0. && j >= 0. && i < rows && j < cols) then None
      else
        let i = int_of_float i and j = int_of_float j in
        if Nx.item [ i; j ] valid then Some (i, j) else None)

let nearest_reads =
  prop "nearest reads the valid cell containing each point" field
    (fun (x, valid, p) ->
      let cells = containing x valid p in
      let at = function Some (i, j) -> Nx.item [ i; j ] x | None -> 0. in
      equal result
        ( Array.of_list (List.map at cells),
          Array.of_list (List.map Option.is_some cells) )
        (read (Warp.nearest ~valid (unit_grid x) p x)))

let same_mask =
  prop "bilinear's mask is nearest's" field (fun (x, valid, p) ->
      equal (array bool)
        (snd (read (Warp.nearest ~valid (unit_grid x) p x)))
        (snd (read (Warp.bilinear ~valid (unit_grid x) p x))))

(* The valid values of the 3 × 3 cells around cell (i, j). *)
let around x valid (i, j) =
  List.concat_map
    (fun i ->
      List.filter_map
        (fun j ->
          let inside = i >= 0 && j >= 0 && i < Nx.dim 0 x && j < Nx.dim 1 x in
          if inside && Nx.item [ i; j ] valid then Some (Nx.item [ i; j ] x)
          else None)
        [ j - 1; j; j + 1 ])
    [ i - 1; i; i + 1 ]

let mean_of_neighbours =
  prop "bilinear lies within its valid neighbours' range" field
    (fun (x, valid, p) ->
      let x = Nx.where (Nx.isfinite x) x (Nx.zeros_like x) in
      let v = floats (fst (Warp.bilinear ~valid (unit_grid x) p x)) in
      List.iteri
        (fun k cell ->
          match cell with
          | None -> equal ~msg:"no data" float_exact 0. v.(k)
          | Some c ->
              let near = around x valid c in
              let lo = List.fold_left Float.min Float.infinity near
              and hi = List.fold_left Float.max Float.neg_infinity near in
              (* The mean rounds once, within an ulp of its terms. *)
              let ulp = 1e-12 *. Float.max 1. (Float.abs lo +. Float.abs hi) in
              equal ~msg:"a number" bool false (Float.is_nan v.(k));
              less ~msg:"low" float_exact ~than:ulp (lo -. v.(k));
              less ~msg:"high" float_exact ~than:ulp (v.(k) -. hi))
        (containing x valid p))

(* A grid of exact binary numbers in UTM 33N, a block of its cells that may
   reach past it, and a band count. *)
let shift =
  Gen.(
    let+ rows = int_range 0 6
    and+ cols = int_range 0 6
    and+ block =
      quad (int_range (-3) 3) (int_range (-3) 3) (int_range 0 6)
        (int_range 0 6)
    and+ bands = int_range 0 2
    and+ seed = int_range 0 1000 in
    let g = north_up (utm 33 true) ~rows ~cols ~x:300000. ~y:5800000. 10. in
    (g, block, bands, seed))

let shifted =
  prop "a shift by whole cells reads the same cells" shift
    (fun ((g : Grid.t), (row, col, rows, cols), bands, seed) ->
      let shape =
        if bands = 0 then [| g.rows; g.cols |] else [| g.rows; g.cols; bands |]
      in
      let hash a = Array.fold_left (fun h v -> (h * 31) + v) seed a in
      let x = Nx.init Nx.float32 shape (fun a -> float_of_int (hash a)) in
      let valid =
        Nx.init Nx.bool [| g.rows; g.cols |] (fun a -> hash a mod 4 <> 0)
      in
      let p = points ~src:g ~dst:(Grid.window ~row ~col ~rows ~cols g) in
      let per = max 1 bands in
      (* Cell k of the block is cell (i, j) of [g]. *)
      let cell k = (row + (k / cols), col + (k mod cols)) in
      let has (i, j) =
        i >= 0 && j >= 0 && i < g.rows && j < g.cols && Nx.item [ i; j ] valid
      in
      let mask =
        Array.init (rows * cols * per) (fun kb -> has (cell (kb / per)))
      in
      let band (i, j) b = if bands = 0 then [ i; j ] else [ i; j; b ] in
      let values =
        Array.init (rows * cols * per) (fun kb ->
            let c = cell (kb / per) in
            if has c then Nx.item (band c (kb mod per)) x else 0.)
      in
      equal ~msg:"nearest" result (values, mask)
        (read (Warp.nearest ~valid g p x));
      equal ~msg:"bilinear" result (values, mask)
        (read (Warp.bilinear ~valid g p x)))

let edges =
  test "bilinear at the edge and by holes uses the valid cells only" (fun () ->
      let x = Nx.create Nx.float64 [| 2; 2 |] [| 1.; 3.; 5.; Float.nan |] in
      let valid = Nx.create Nx.bool [| 2; 2 |] [| true; true; true; false |] in
      let p =
        pixels
          [
            (0.25, 0.25);
            (0.25, 1.);
            (0.5, 1.5);
            (1., 1.);
            (1.5, 1.5);
            (2., 0.5);
          ]
      in
      equal result
        ( [| 1.; 2.; 3.; 0.; 0.; 0. |],
          [| true; true; true; false; false; false |] )
        (read (Warp.bilinear ~valid (unit_grid x) p x));
      (* Next to the hole, the three valid cells' weights are renormalised. *)
      let v, _ =
        read (Warp.bilinear ~valid (unit_grid x) (pixels [ (0.75, 0.75) ]) x)
      in
      equal (array float_exact) [| (9. +. 9. +. 15.) /. 15. |] v)

let nan_data =
  test "a valid NaN is data where its weight is not 0" (fun () ->
      let x = Nx.create Nx.float64 [| 1; 2 |] [| 2.; Float.nan |] in
      let p = pixels [ (0.5, 0.5); (0.5, 0.75); (0.5, 1.5) ] in
      equal result
        ([| 2.; Float.nan; Float.nan |], [| true; true; true |])
        (read (Warp.bilinear (unit_grid x) p x)))

let borders =
  test "a point on a border reads the cell after it" (fun () ->
      let x = Nx.create Nx.int32 [| 2; 3 |] [| 0l; 1l; 2l; 3l; 4l; 5l |] in
      let p =
        pixels
          [ (1., 2.); (-0., -0.); (0.999, 1.); (2., 0.); (0., 3.); (0., -0.5) ]
      in
      equal result
        ( [| 5.; 0.; 1.; 0.; 0.; 0. |],
          [| true; true; true; false; false; false |] )
        (read (Warp.nearest (unit_grid x) p x)))

let dtypes =
  test "nearest keeps the dtype, bilinear rounds once to it" (fun () ->
      let p = pixels [ (0.5, 0.75); (0.6, 1.1) ] in
      let x8 = Nx.create Nx.uint8 [| 1; 2 |] [| 7; 255 |] in
      let g = unit_grid x8 in
      equal (array int) [| 7; 255 |] (Nx.to_array (fst (Warp.nearest g p x8)));
      let b = Nx.create Nx.bool [| 1; 2 |] [| false; true |] in
      equal (array bool) [| false; true |]
        (Nx.to_array (fst (Warp.nearest g p b)));
      let x32 = Nx.create Nx.float32 [| 1; 2 |] [| 0.1; 0.7 |] in
      let x64 = Nx.cast Nx.float64 x32 in
      equal (array float_exact)
        (floats (Nx.cast Nx.float32 (fst (Warp.bilinear g p x64))))
        (floats (fst (Warp.bilinear g p x32))))

let shapes =
  test "results have the points' batch and the array's other axes" (fun () ->
      let x = Nx.zeros Nx.float32 [| 3; 4; 2; 5 |] in
      let g = unit_grid x in
      let v, ok = Warp.bilinear g (Nx.full Nx.float64 [| 6; 1; 2 |] 1.5) x in
      equal (array int) [| 6; 1; 2; 5 |] (Nx.shape v);
      equal (array int) [| 6; 1; 2; 5 |] (Nx.shape ok);
      let v, ok = Warp.nearest g (Nx.zeros Nx.float64 [| 0; 2 |]) x in
      equal (array int) [| 0; 2; 5 |] (Nx.shape v);
      equal (array int) [| 0; 2; 5 |] (Nx.shape ok);
      let empty = Nx.zeros Nx.float64 [| 0; 3 |] in
      equal result
        ([| 0. |], [| false |])
        (read (Warp.bilinear (unit_grid empty) (pixels [ (0., 0.) ]) empty)))

let refusals =
  test "samplers refuse points, arrays and masks of other shapes" (fun () ->
      let x = Nx.zeros Nx.float64 [| 3; 4 |] and p = pixels [ (0., 0.) ] in
      let g = unit_grid x in
      raises_match (Exn.invalid_arg ~sub:"last axis") (fun () ->
          Warp.nearest g (Nx.zeros Nx.float64 [| 2; 3 |]) x);
      raises_match (Exn.invalid_arg ~sub:"rank") (fun () ->
          Warp.bilinear g p (Nx.zeros Nx.float64 [| 3 |]));
      raises_match (Exn.invalid_arg ~sub:"mask") (fun () ->
          Warp.nearest ~valid:(Nx.ones Nx.bool [| 4; 3 |]) g p x);
      raises_match (Exn.invalid_arg ~sub:"not a real dtype") (fun () ->
          Warp.bilinear g p (Nx.zeros Nx.bool [| 3; 4 |])))

(* Gradients *)

(* A smooth source of [n × n] cells. *)
let sines n =
  Nx.init Nx.float64 [| n; n |] (fun a ->
      let i = float_of_int a.(0) and j = float_of_int a.(1) in
      sin (0.45 *. i) +. cos (0.3 *. j) +. (0.5 *. sin (0.2 *. (i +. j))))

let truth = (0.3, -0.2)
let shift (r, c) = Nx.create Nx.float64 [| 2 |] [| r; c |]

(* A pair of images: [x] at the centres of its cells 4 in from its edges, and
   [x] at those centres moved by [truth]. [loss s] is their squared distance
   with the first's points moved by [s] instead. *)
let loss x =
  let margin = 4 in
  let m = Nx.dim 0 x - (2 * margin) in
  let centres =
    Nx.init Nx.float64 [| m; m; 2 |] (fun a ->
        float_of_int (a.(a.(2)) + margin) +. 0.5)
  in
  let sample s =
    fst
      (Warp.bilinear (unit_grid x)
         (Nx.add centres (Nx.reshape [| 1; 1; 2 |] s))
         x)
  in
  let target = sample (shift truth) in
  fun s ->
    let d = Nx.sub (sample s) target in
    Nx.sum (Nx.mul d d)

let centre_gradient =
  test "the gradient on the centres is the right-hand derivative" (fun () ->
      let x = sines 24 in
      List.iter
        (fun (name, x) ->
          let loss = loss x in
          let l s = Nx.item [] (loss (shift s)) and h = 1e-7 in
          let steps = [| (h, 0.); (0., h) |] in
          Array.iteri
            (fun k g ->
              let msg = Printf.sprintf "%s, axis %d" name k in
              let right = (l steps.(k) -. l (0., 0.)) /. h in
              greater ~msg float_exact ~than:1. (Float.abs g);
              equal ~msg (float (1e-5 *. Float.abs g)) right g)
            (Nx.to_array (Rune.grad' loss (shift (0., 0.)))))
        [
          ("one band", x);
          ("three bands", Nx.stack ~axis:2 [ x; Nx.mul_s x 2.; Nx.neg x ]);
        ])

(* Channel 0 of a 4 × 4 source, 10 i + j² with NaN at [nans], as the first of
   two channels whose second has NaN at [other] too; [holes] are not valid.
   [at p] is channel 0's value at the point [p] moved by [s]. *)
let nan_source ~nans ~other ~holes =
  let x =
    Nx.init Nx.float64 [| 4; 4; 2 |] (fun a ->
        let c = (a.(0), a.(1)) in
        if List.mem c nans || (a.(2) = 1 && List.mem c other) then Float.nan
        else float_of_int ((10 * a.(0)) + (a.(1) * a.(1))))
  in
  let valid =
    Nx.init Nx.bool [| 4; 4 |] (fun a -> not (List.mem (a.(0), a.(1)) holes))
  in
  fun (r, c) s ->
    let p = Nx.add (pixels [ (r, c) ]) (Nx.reshape [| 1; 2 |] s) in
    Nx.sum (Nx.slice [ A; I 0 ] (fst (Warp.bilinear ~valid (unit_grid x) p x)))

let nan_neighbours =
  cases ~id:(fun (name, _, _, _, _, _) -> name)
    "a NaN left out leaves the gradient of the cells kept"
    [
      (* Point (1.5, 1.5) is cell (1, 1)'s centre: its far cells, row 2 and
         column 2, have weight 0. *)
      ("NaN on the diagonal", [ (2, 2) ], [], [], (1.5, 1.5), [ None; None ]);
      ("NaN in a hole", [ (2, 2) ], [], [ (2, 2) ], (1.7, 1.7), [ None; None ]);
      ( "NaN in another channel",
        [],
        [ (2, 1) ],
        [],
        (1.5, 1.5),
        [ None; None ] );
      (* Along its own axis a NaN neighbour gives 0, as a hole does. *)
      ("NaN along an axis", [ (2, 1) ], [], [], (1.5, 1.5), [ Some 0.; None ]);
    ]
    (fun (_, nans, other, holes, p, want) ->
      let at = nan_source ~nans ~other ~holes p in
      let l s = Nx.item [] (at (shift s)) and h = 1e-7 in
      let steps = [| (h, 0.); (0., h) |] in
      let g = Nx.to_array (Rune.grad' at (shift (0., 0.))) in
      List.iteri
        (fun k want ->
          let right = (l steps.(k) -. l (0., 0.)) /. h in
          let want = Option.value want ~default:right in
          equal ~msg:(Printf.sprintf "axis %d" k) (float 1e-5) want g.(k))
        want)

let descent =
  test "descent from the centres finds a sub-pixel shift" (fun () ->
      let loss = loss (sines 24) in
      let s = ref (shift (0., 0.)) in
      for _ = 1 to 300 do
        s := Nx.sub !s (Nx.mul_s (Rune.grad' loss !s) 0.002)
      done;
      let r, c = truth in
      equal (array (float 1e-6)) [| r; c |] (Nx.to_array !s))

(* Systems *)

let utm33 = north_up (utm 33 true) ~rows:4 ~cols:5 ~x:500000. ~y:5300000. 10.

(* A grid 60° from its zone's meridian: no transform takes its points. *)
let far = north_up (utm 33 true) ~rows:2 ~cols:3 ~x:7.5e6 ~y:1e6 10.

let equal_systems =
  test "equal systems take the centres with no transform" (fun () ->
      List.iter
        (fun (g : Grid.t) ->
          equal (array float_exact)
            (Nx.to_array (Grid.centres ~at:host g))
            (Nx.to_array (Warp.centres ~at:host g.crs g)))
        [ utm33; far ])

let nan_points t = Array.map Float.is_nan (Nx.to_array t)

(* The area rules move a footprint's corners to within 180° of its first,
   so a geographic cell with a corner 180° or more from its first would be
   read as its complement, the other side of the circle: corners refuses
   it, on every system but Albers', and says to split it. *)
let half_turn =
  test "corners refuse a cell of 180° of longitude or more" (fun () ->
      let cells ?(rx = 0.) cx =
        Grid.v Crs.lonlat ~rows:1 ~cols:2 ~origin:(-180., 10.)
          ~row_step:(rx, -10.) ~col_step:(cx, 0.)
      in
      let take g = ignore (Warp.corners ~at:host Crs.lonlat g) in
      List.iter
        (fun g ->
          raises_match
            (Exn.invalid_arg
               ~sub:"180° or more of longitude from its first; split it")
            (fun () -> take g))
        [ cells 200.; cells 180.; cells (-180.); cells ~rx:100. 90. ];
      take (cells 179.9);
      take (cells ~rx:100. (-90.));
      (* Pseudo-Mercator's longitude is linear in x, λ - λ₀ = x / a: cells of
         180° and of 170° from 180° W. *)
      let mercator = Result.get_ok (Crs.of_epsg 3857) in
      let a = 6378137. in
      let merc d =
        north_up mercator ~rows:1 ~cols:2 ~x:(-.Float.pi *. a) ~y:1e6 d
      in
      raises_match (Exn.invalid_arg ~sub:"180° or more") (fun () ->
          take (merc (Float.pi *. a)));
      take (merc (Float.pi *. a *. 17. /. 18.));
      take (north_up (utm 33 true) ~rows:1 ~cols:1 ~x:0. ~y:0. 1e6))

(* The apex of an Albers cone [crs] and the radius of the arc that is the
   pole [lat]'s image, by the circle through three of its points. *)
let apex crs lat =
  let image lon =
    let p =
      Crs.transform ~src:Crs.lonlat ~dst:crs
        (Nx.create Nx.float64 [| 2 |] [| lon; lat |])
    in
    (Nx.item [ 0 ] p, Nx.item [ 1 ] p)
  in
  let (ax, ay), (bx, by), (cx, cy) = (image 0., image 60., image (-60.)) in
  (* The circumcentre, by the determinants of the three points. *)
  let along a b c = (a *. (by -. cy)) +. (b *. (cy -. ay)) +. (c *. (ay -. by))
  and across a b c = (a *. (cx -. bx)) +. (b *. (ax -. cx)) +. (c *. (bx -. ax))
  and s x y = (x *. x) +. (y *. y) in
  let d = 2. *. along ax bx cx in
  let ux = along (s ax ay) (s bx by) (s cx cy) /. d
  and uy = across (s ax ay) (s bx by) (s cx cy) /. d in
  (ux, uy, Float.hypot (ax -. ux) (ay -. uy))

let polar_albers =
  {
    Crs.lonlat with
    projection =
      Some
        (Crs.albers ~lat0:88. ~lon0:0. ~lat1:89.5 ~lat2:85. ~false_easting:0.
           ~false_northing:0.);
  }

(* On a projection, corners compares a cell's corners by the longitudes its
   system gives them. A sinusoid's cell across λ₀ 120 m from the pole has
   corners 286° apart, and the rule read its 74° complement, 158.74 where
   the mean of |λ| is 68.83. *)
let projected_half_turn =
  test "corners refuse a projected cell of 180° or more, by its longitudes"
    (fun () ->
      let take crs g = ignore (Warp.corners ~at:host crs g) in
      let top = 6371007.181 *. Float.pi /. 2. in
      let sinusoid ~x ~y w h =
        Grid.v Crs.modis ~rows:1 ~cols:1 ~origin:(x, y) ~row_step:(0., -.h)
          ~col_step:(w, 0.)
      in
      let across = sinusoid ~x:(-300.) ~y:(top -. 120.) 600. 10. in
      raises_match
        (Exn.invalid_arg ~sub:"cell [0, 0] has a corner 180° or more")
        (fun () -> take Crs.lonlat across);
      (* On the sinusoid itself the footprint is the cell. *)
      take Crs.modis across;
      (* Valid cells near the pole: MODIS h18v00's rows next to it, and a
         cell across λ₀ 5 km from it. *)
      let m = 926.625433055833 in
      take Crs.lonlat
        (north_up Crs.modis ~rows:2 ~cols:3 ~x:0. ~y:(10007554.677 -. m) m);
      take Crs.lonlat (sinusoid ~x:(-500.) ~y:(top -. 5000.) 1000. 1000.);
      (* A corner on the sinusoid's cut, at 180° on the equator, lies on the
         side of its easting. *)
      let e = 6371007.181 *. Float.pi in
      take Crs.lonlat
        (Grid.v Crs.modis ~rows:1 ~cols:1 ~origin:(e, 0.)
           ~row_step:(-1000., 1000.) ~col_step:(-1000., -1000.));
      (* Albers' cut, λ₀ + 180° above its apex, lies inside its plane: a cell
         across it clear of the pole's arc is valid. *)
      let ux, uy, rho = apex polar_albers 90. in
      let h = 3. *. rho in
      take Crs.lonlat
        (Grid.v polar_albers ~rows:1 ~cols:1
           ~origin:(ux -. h, uy +. (1.2 *. rho) +. h)
           ~row_step:(0., -.h) ~col_step:(2. *. h, 0.)))

(* The exemption has two sides. A projected grid in the corners' own system
   is not checked, since its cells are their footprints. A geographic grid
   in its own system is: a cap of one cell of 360° from the pole, or two of
   180°, would be read as cells of 0° and of 180° the other way. *)
let own_system =
  test "corners check a geographic grid in its own system, not a projected one"
    (fun () ->
      let cap cols =
        Grid.v Crs.lonlat ~rows:1 ~cols ~origin:(-180., 90.)
          ~row_step:(0., -1.)
          ~col_step:(360. /. float_of_int cols, 0.)
      in
      List.iter
        (fun cols ->
          raises_match
            (Exn.invalid_arg ~sub:"cell [0, 0] has a corner 180° or more")
            (fun () -> Warp.corners ~at:host Crs.lonlat (cap cols)))
        [ 1; 2 ];
      ignore (Warp.corners ~at:host Crs.lonlat (cap 3));
      (* A sinusoid's cell across λ₀ 120 m from the pole, whose corners are
         286° apart: refused into Lonlat, its own cell in the sinusoid. *)
      let top = 6371007.181 *. Float.pi /. 2. in
      let across =
        Grid.v Crs.modis ~rows:1 ~cols:1 ~origin:(-300., top -. 120.)
          ~row_step:(0., -10.) ~col_step:(600., 0.)
      in
      raises_match (Exn.invalid_arg ~sub:"180° or more") (fun () ->
          Warp.corners ~at:host Crs.lonlat across);
      ignore (Warp.corners ~at:host Crs.modis across))

(* A cell's corners are its grid's pixel points at integers, by Grid.world's
   map: exact in their own system, so they lie on the lattice of a source
   grid in it, and NaN outside the domain of another as the centres are. *)
let corners_law =
  test "the corners are the grid's pixel points at integers" (fun () ->
      let g =
        Grid.v (utm 33 true) ~rows:2 ~cols:3 ~origin:(500000., 5300000.)
          ~row_step:(0., -10.) ~col_step:(10., 0.)
      in
      let at =
        Nx.init Nx.float64 [| 3; 4; 2 |] (fun a ->
            float_of_int (if a.(2) = 0 then a.(0) else a.(1)))
      in
      let c = Warp.corners ~at:host g.crs g in
      equal (array float_exact)
        (Nx.to_array (Grid.world g at))
        (Nx.to_array c);
      (* Adding 0 turns the row's -0, 0 over a negative step, into 0. *)
      equal (array float_exact) (Nx.to_array at)
        (Nx.to_array (Nx.add_s (Warp.pixel g c) 0.));
      equal (array bool) (Array.make 24 true)
        (nan_points (Warp.corners ~at:host (utm 60 true) g)))

let no_source =
  test "a centre outside the source's domain has no data" (fun () ->
      let c = Warp.centres ~at:host (utm 60 true) utm33 in
      equal (array bool) (Array.make 40 true) (nan_points c);
      let src = north_up (utm 60 true) ~rows:2 ~cols:2 ~x:5e5 ~y:5.3e6 10. in
      let x = Nx.zeros Nx.int32 [| 2; 2 |] in
      let _, ok = Warp.nearest src (Warp.pixel src c) x in
      equal (array bool) (Array.make 20 false) (Nx.to_array ok);
      (* A Lonlat grid whose first row lies past the pole. *)
      let polar = north_up Crs.lonlat ~rows:3 ~cols:1 ~x:15. ~y:90.5 0.5 in
      equal (array bool)
        [| true; true; false; false; false; false |]
        (nan_points (Warp.centres ~at:host (utm 33 true) polar)))

let own_domain =
  test "a centre outside the destination's own domain raises" (fun () ->
      raises_match (Exn.invalid_arg ~sub:"domain") (fun () ->
          Warp.centres ~at:host Crs.lonlat far);
      (* Corners name themselves and the corner, which the program made:
         a zone's grid whose first corner lies past the pole's image. *)
      let past =
        north_up (utm 33 true) ~rows:2 ~cols:2 ~x:490000. ~y:1.0008e7 1e4
      in
      raises_match
        (Exn.invalid_arg
           ~sub:
             "Jord.Warp.corners: corner [0, 0] (490000, 10008000) is outside \
              WGS 84 / UTM zone 33N's domain; dst's corners must lie in its \
              system's domain (Crs.in_domain), and around a pole dst is a \
              geographic cap of cells narrower than 180°")
        (fun () -> Warp.corners ~at:host Crs.lonlat past))

let nad83_utm zone = Result.get_ok (Crs.of_epsg (26900 + zone))

(* Grids of 3 cells of 100 km on zone 17's meridian, whose centres lie near
   24.8°N, 23.9°N and 23.0°N: the last south of NAD83's box (23.81°N). *)
let across_box crs = north_up crs ~rows:3 ~cols:1 ~x:4.5e5 ~y:2.8e6 1e5

(* A UTM 35N tile of 1 km at Finland's northernmost point, Nuorgam,
   (27.9556, 70.0922), north of the box of EPSG's area for 10701. *)
let nuorgam =
  let x, y =
    match
      Nx.to_array
        (Crs.transform ~src:Crs.lonlat ~dst:(utm 35 true)
           (Nx.create Nx.float64 [| 1; 2 |] [| 27.9556; 70.0922 |]))
    with
    | [| x; y |] -> (x, y)
    | _ -> assert false
  in
  north_up (utm 35 true) ~rows:10 ~cols:10 ~x:(x -. 500.) ~y:(y +. 500.) 100.

let across_datums =
  test "centres across datums have data beyond a step's area" (fun () ->
      let tm35fin = Result.get_ok (Crs.of_epsg 3067) in
      equal (array bool) (Array.make 200 false)
        (nan_points (Warp.centres ~at:host tm35fin nuorgam));
      (* NAD83 / UTM 17N onto WGS 84's, across 1325's south edge. *)
      equal (array bool) (Array.make 6 false)
        (nan_points
           (Warp.centres ~at:host (utm 17 true) (across_box (nad83_utm 17)))))

(* The antimeridian *)

(* The antimeridian fixture's source grid, its west edge at 179.998 or at
   -180.002. *)
let lonlat_at x = north_up Crs.lonlat ~rows:30 ~cols:40 ~x ~y:(-16.998) 0.0001

let any_360 =
  test "a Lonlat source reads the same cells in either 360°" (fun () ->
      let f = fixture "antimeridian" in
      (* Each cell's value is its column; column 20 is the antimeridian's. *)
      let x = Nx.init Nx.float64 [| 30; 40 |] (fun a -> float_of_int a.(1)) in
      let at src = read (Warp.nearest src (points ~src ~dst:f.dst) x) in
      let cols, ok = at (lonlat_at 179.998) in
      equal result (cols, ok) (at (lonlat_at (-180.002)));
      let cols = List.filteri (fun k _ -> ok.(k)) (Array.to_list cols) in
      equal ~msg:"numbers" bool false (List.exists Float.is_nan cols);
      less ~msg:"west of it" float_exact ~than:20.
        (List.fold_left Float.min 40. cols);
      greater ~msg:"east of it" float_exact ~than:19.
        (List.fold_left Float.max 0. cols))

let in_range =
  prop "pixel keeps a longitude in its 360° exactly"
    Gen.(pair (float_range (-400.) 400.) (float_range 0. 359.999))
    (fun (w, off) ->
      let g = north_up Crs.lonlat ~rows:10 ~cols:10 ~x:w ~y:10. 1. in
      let p = pixels [ (w +. off, 5.) ] in
      equal (array float_exact)
        (Nx.to_array (Grid.pixel g p))
        (Nx.to_array (Warp.pixel g p)))

(* Windows *)

let window_cases =
  test "a window holds the cells each rule reads" (fun () ->
      let g = north_up (utm 33 true) ~rows:10 ~cols:10 ~x:0. ~y:0. 1. in
      let w rule l = Warp.window rule g (pixels l) in
      let quad = quad int int int int in
      equal quad (2, 4, 1, 1) (w Nearest [ (2.3, 4.7) ]);
      equal quad (1, 4, 2, 2) (w Bilinear [ (2.3, 4.7) ]);
      equal quad (0, 0, 1, 1) (w Bilinear [ (0.2, 0.2) ]);
      equal quad (0, 0, 0, 0) (w Bilinear [ (-0.2, 3.); (Float.nan, 1.) ]);
      (* Only the points in a cell count, not their bounding box. *)
      equal quad (2, 1, 6, 9)
        (w Nearest [ (2.5, 1.); (7.9, 9.9); (5., 12.); (-1., 4.) ]))

let windowed =
  prop "a window's block gives the whole array's bits" field
    (fun (x, valid, p) ->
      let g = unit_grid x in
      let check rule sample =
        let row, col, rows, cols = Warp.window rule g p in
        let block a = Nx.slice [ R (row, row + rows); R (col, col + cols) ] a in
        equal result
          (read (sample ?row:None ?col:None ?valid:(Some valid) g p x))
          (read
             (sample ?row:(Some row) ?col:(Some col)
                ?valid:(Some (block valid)) g p (block x)))
      in
      check Nearest Warp.nearest;
      check Bilinear Warp.bilinear)

(* Through ~row ~col, a sampler's gradients in the points and in the
   block's values are the whole array's, bit for bit, eagerly and compiled:
   nearest's in the points is 0, and the block's values take the whole
   gradient's cells of the block. *)
let windowed_gradients =
  test "a sampler's gradients through its block are the whole array's"
    (fun () ->
      let g = north_up (utm 33 true) ~rows:12 ~cols:14 ~x:0. ~y:0. 1. in
      let x, valid = source 1009 g in
      let p =
        Nx.init Nx.float64 [| 4; 5; 2 |] (fun a ->
            let k = a.(2) in
            (float_of_int a.(k) *. 1.7) +. 2.3 +. (0.35 *. float_of_int k))
      in
      let check rule sample =
        let row, col, rows, cols = Warp.window rule g p in
        let cut a = Nx.slice [ R (row, row + rows); R (col, col + cols) ] a in
        let whole p x = Nx.sum (fst (sample ?row:None ?col:None ~valid g p x))
        and part p x =
          Nx.sum
            (fst
               (sample ?row:(Some row) ?col:(Some col) ~valid:(cut valid) g p
                  x))
        in
        is_true ~msg:"a block" (rows < 12 && cols < 14);
        equal (array int64)
          (bits (Rune.grad' (fun p -> whole p x) p))
          (bits (Rune.grad' (fun p -> part p (cut x)) p));
        equal (array int64)
          (bits (Rune.jit' (Rune.grad' (fun p -> whole p x)) p))
          (bits (Rune.jit' (Rune.grad' (fun p -> part p (cut x))) p));
        equal (array int64)
          (bits (cut (Rune.grad' (fun x -> whole p x) x)))
          (bits (Rune.grad' (fun x -> part p x) (cut x)))
      in
      check Nearest (fun ?row ?col ~valid g p x ->
          Warp.nearest ?row ?col ~valid g p x);
      check Bilinear (fun ?row ?col ~valid g p x ->
          Warp.bilinear ?row ?col ~valid g p x))

(* On a 360° source bilinear's window is a run of columns modulo their
   number, as area_window's: across the seam, the last columns and the
   first, which a program reads in two parts. The sampler on it gives the
   whole array's bits, and a block a column short raises, eagerly and
   compiled. *)
let seam_window =
  test "bilinear's window across a 360° source's seam" (fun () ->
      let src = north_up Crs.lonlat ~rows:10 ~cols:360 ~x:(-180.) ~y:10. 1. in
      let x, valid = source 1009 src in
      let p =
        Warp.pixel src
          (pixels [ (179.9, 5.3); (-179.7, 4.2); (179.2, 6.6); (-179.95, 5.) ])
      in
      let ((row, col, rows, cols) as b) = Warp.window Bilinear src p in
      equal (quad int int int int) (2, 358, 5, 3) b;
      let read (row, col, rows, cols) a =
        let part col cols =
          Nx.slice [ R (row, row + rows); R (col, col + cols) ] a
        in
        if col + cols <= 360 then part col cols
        else
          Nx.concatenate ~axis:1
            [ part col (360 - col); part 0 (col + cols - 360) ]
      in
      let rule b p =
        let row, col, _, _ = b in
        fst (Warp.bilinear ~row ~col ~valid:(read b valid) src p (read b x))
      in
      equal (array int64)
        (bits (fst (Warp.bilinear ~valid src p x)))
        (bits (rule b p));
      equal (array int64)
        (bits (Rune.jit' (rule (0, 0, 10, 360)) p))
        (bits (Rune.jit' (rule b) p));
      let reads = Exn.invalid_arg ~sub:"reads a cell outside the block" in
      raises_match reads (fun () ->
          Nx.to_array (rule (row, col, rows, cols - 1) p));
      raises_match reads (fun () ->
          Nx.to_array (Rune.jit' (rule (row, col + 1, rows, cols - 1)) p));
      (* A point whose one missing tap lies across the seam, east or west. *)
      List.iter
        (fun (lon, b) ->
          raises_match reads (fun () ->
              Nx.to_array (rule b (Warp.pixel src (pixels [ (lon, 5.3) ])))))
        [ (179.9, (4, 359, 2, 1)); (-179.7, (4, 0, 2, 1)) ];
      (* Nearest's block does not wrap. *)
      equal (quad int int int int) (3, 0, 3, 360) (Warp.window Nearest src p))

(* A block a cell short of the one window gives raises at the first point
   that reads the missing cell, eagerly and compiled: nearest a column short,
   bilinear on nearest's block, and bilinear a row short at the top. Without
   the grid, a sampler read such a cell as one outside the raster, and gave
   a false mask or left out a tap. *)
let short_blocks =
  test "a sampler on a block short of its window raises" (fun () ->
      let g = north_up (utm 33 true) ~rows:6 ~cols:7 ~x:0. ~y:0. 1. in
      let x, valid = source 1009 g in
      let p =
        Nx.init Nx.float64 [| 5; 5; 2 |] (fun a ->
            let k = a.(2) in
            (float_of_int a.(k) *. 0.9) +. 1.3 +. (0.2 *. float_of_int k))
      in
      let cut (row, col, rows, cols) a =
        Nx.slice [ R (row, row + rows); R (col, col + cols) ] a
      in
      let run rule (row, col, rows, cols) p =
        let b = (row, col, rows, cols) in
        let sample =
          match rule with
          | Warp.Nearest -> Warp.nearest
          | Bilinear -> Warp.bilinear
        in
        sample ~row ~col ~valid:(cut b valid) g p (cut b x)
      in
      let short rule b =
        let reads = Exn.invalid_arg ~sub:"reads a cell outside the block" in
        raises_match reads (fun () -> Nx.to_array (fst (run rule b p)));
        raises_match reads (fun () ->
            Nx.to_array (Rune.jit' (fun p -> fst (run rule b p)) p))
      in
      let row, col, rows, cols = Warp.window Nearest g p in
      short Nearest (row, col, rows, cols - 1);
      short Bilinear (row, col, rows, cols);
      let row, col, rows, cols = Warp.window Bilinear g p in
      short Bilinear (row + 1, col, rows - 1, cols);
      short Nearest (0, 0, 6, 5);
      (* A point outside the grid has no data in any block, and a tap outside
         it is left out as on the whole array. *)
      let outside = pixels [ (-0.25, 2.); (3., 7.5) ] in
      List.iter
        (fun rule ->
          equal result
            ([| 0.; 0. |], [| false; false |])
            (read (run rule (1, 1, 2, 2) outside)))
        [ Warp.Nearest; Bilinear ];
      let edge = pixels [ (0.2, 2.5) ] in
      equal result
        (read (Warp.bilinear ~valid g edge x))
        (read (run Bilinear (0, 1, 2, 3) edge));
      (* An array that is not a block of the grid. *)
      List.iter
        (fun (row, col, rows, cols) ->
          let x = Nx.zeros Nx.float64 [| rows; cols |] in
          raises_match (Exn.invalid_arg ~sub:"not in the grid") (fun () ->
              Warp.bilinear ~row ~col g p x))
        [
          (-1, 0, 2, 2); (0, -1, 2, 2); (5, 0, 2, 2); (0, 6, 2, 2); (0, 0, 7, 7);
        ])

(* Area rules *)

(* [dst]'s corners as pixel points of [src], as a caller composes them. *)
let corners ~src ~dst =
  Warp.pixel src (Warp.corners ~at:host src.Grid.crs dst)

(* A grid of [d]-metre cells turned by [t] radians from north-up. *)
let turned crs ~rows ~cols ~x ~y d t =
  Grid.v crs ~rows ~cols ~origin:(x, y)
    ~row_step:(d *. sin t, -.d *. cos t)
    ~col_step:(d *. cos t, d *. sin t)

(* A source of unit cells for footprints given as pixel points. *)
(* Corners [(r + i·a + j·b)], [[rows + 1; cols + 1; 2]], pixel points. *)
let quads ~rows ~cols (r0, c0) (ar, ac) (br, bc) =
  Nx.init Nx.float64
    [| rows + 1; cols + 1; 2 |]
    (fun a ->
      let i = float_of_int a.(0) and j = float_of_int a.(1) in
      if a.(2) = 0 then r0 +. (i *. ar) +. (j *. br)
      else c0 +. (i *. ac) +. (j *. bc))

(* A rule's values, NaN where the fraction is 0. *)
let with_nan (v, fraction) =
  let v = floats v and f = Nx.to_array fraction in
  Array.mapi (fun k v -> if f.(k) > 0. then v else Float.nan) v

let masks_agree ~msg want mine =
  let has = Array.map (fun v -> not (Float.is_nan v)) in
  equal ~msg:(msg ^ "'s mask") (array bool) (has want) (has mine)

(* [want] and [mine] where [want] has data and [keep] holds, to [tol] of each
   value, exactly where [tol] gives 0. *)
let agree ?(keep = fun _ -> true) ~msg tol want mine =
  masks_agree ~msg want mine;
  Array.iteri
    (fun k w ->
      if keep k && not (Float.is_nan w) then
        let t = tol w in
        equal ~msg (if t > 0. then float t else float_exact) w mine.(k))
    want

(* Whether footprint [k] of [q] lies in the [rows × cols] cells. *)
let within ~rows ~cols q k =
  let c = Nx.dim 1 q - 1 in
  let i = k / c and j = k mod c in
  List.for_all
    (fun (a, b) ->
      let r = Nx.item [ i + a; j + b; 0 ] q
      and c = Nx.item [ i + a; j + b; 1 ] q in
      r >= 0. && r <= float_of_int rows && c >= 0. && c <= float_of_int cols)
    [ (0, 0); (0, 1); (1, 0); (1, 1) ]

let areas = read_fixtures "warp_area.dat"

(* Each case's k by warp.mli's law, ⌊s (|cos θ| + |sin θ|)⌋ + 2 for cells s
   source cells wide turned by θ: 2 to 3 cells on the source's own axes,
   and 1.00 turned by 4.6° across UTM zones. *)
let law = function
  | "aligned20" | "shifted25" | "mode20" | "mode25" -> 4
  | "holes30" | "ties30" -> 5
  | "xzone" -> 3
  | name -> failwith ("no k for " ^ name)
let area_fixture name = List.find (fun f -> f.name = name) areas

(* GDAL and jord agree where a footprint lies in the source. GDAL weighs the
   edge cell of a footprint reaching past the source by 1 plus the overhang,
   since it clamps the cells it reads to the source and weighs the last by its
   distance to the footprint's edge (gdalwarpkernel.cpp:6690-6701 and :6839-6856
   at 3.12.2), where jord keeps the cell's area in the footprint: a named
   divergence, which the fixtures' generator checks against that rule. Both
   take the corners from the affine map at each pixel point, jord's through
   Warp.corners, so they agree to rounding: a mean here agrees to 4e-11 times
   the range of the values, within the 4e-10 allowed. *)
let gdal_case name ~tol output rule =
  let f = area_fixture name in
  let q = corners ~src:f.src ~dst:f.dst in
  let keep = within ~rows:f.src.rows ~cols:f.src.cols q in
  agree ~keep
    ~msg:(name ^ " " ^ output)
    tol
    (List.assoc output f.outputs)
    (with_nan (rule f q))

let gdal_average =
  cases "GDAL -r average, one system" ~id:Fun.id [ "aligned20"; "shifted25" ]
    (fun name ->
      gdal_case name
        ~tol:(fun _ -> 4e-10 *. 1008.)
        "average"
        (fun f q ->
          let x, valid = source 1009 f.src in
          Warp.average ~k:(law f.name) ~valid f.src q x))

(* On integers both store a mean as ⌊v + ½⌋ of its float64 value, so they agree
   but where v lies within rounding of a tie, where the two sums, taken in
   different orders, may fall on its two sides: none here. [ties30]'s means of 6
   and 8 cells are exact ties, which both round up. *)
let gdal_rounded =
  cases "GDAL -r average on integers" ~id:Fun.id [ "holes30"; "ties30" ]
    (fun name ->
      let f = area_fixture name in
      let x, valid = source 12000 f.src in
      gdal_case name
        ~tol:(fun _ -> 4e-10 *. 12000.)
        "average"
        (fun f q -> Warp.average ~k:(law name) ~valid f.src q x);
      gdal_case name
        ~tol:(fun _ -> 0.)
        "rounded"
        (fun f q ->
          Warp.average ~k:(law name) ~valid f.src q (Nx.cast Nx.uint16 x));
      if name = "ties30" then
        let mean = List.assoc "average" f.outputs in
        is_true ~msg:"exact ties"
          (Array.exists (fun v -> v -. Float.floor v = 0.5) mean))

let gdal_mode =
  cases "GDAL -r mode, MODE_TIES=MIN, one system" ~id:Fun.id
    [ "mode20"; "mode25" ] (fun name ->
      gdal_case name
        ~tol:(fun _ -> 0.)
        "mode"
        (fun f q ->
          let x, valid = source 4 f.src in
          Warp.mode ~k:(law f.name) ~valid f.src q (Nx.cast Nx.uint8 x)))

(* jord's corners and pyproj's agree to within 1.25e-9 source pixel, as for the
   samplers, and moving a footprint's corners by e moves its mean by at most 4 e
   times the range of the values it covers. *)
let exact_xzone =
  test "the exact area mean across UTM zones 31 and 32" (fun () ->
      let f = area_fixture "xzone" in
      let x =
        Nx.create Nx.float64
          [| f.src.rows; f.src.cols |]
          (List.assoc "source" f.outputs)
      in
      let valid = Nx.not_equal_s x 0. in
      let q = corners ~src:f.src ~dst:f.dst in
      let mine = with_nan (Warp.average ~k:3 ~valid f.src q x) in
      let range = Nx.item [] (Nx.max x) -. Nx.item [] (Nx.min x) in
      agree ~msg:"xzone"
        (fun _ -> 4. *. 1.25e-9 *. range)
        (List.assoc "average" f.outputs)
        mine)

let utm_src = north_up (utm 33 true) ~rows:9 ~cols:11 ~x:500000. ~y:5300000. 10.

let own_grid =
  test "on its own grid, a rule gives the array" (fun () ->
      let x, valid = source 1009 utm_src in
      let q = corners ~src:utm_src ~dst:utm_src in
      let ok = Nx.to_array valid in
      let want = Array.mapi (fun k v -> if ok.(k) then v else 0.) (floats x) in
      let fraction = Array.map (fun ok -> if ok then 1. else 0.) ok in
      let v, f = Warp.average ~k:3 ~valid utm_src q x in
      equal ~msg:"average" (array float_exact) want (floats v);
      equal ~msg:"fraction" (array float_exact) fraction (Nx.to_array f);
      let v, f = Warp.mode ~k:3 ~valid utm_src q (Nx.cast Nx.int32 x) in
      equal ~msg:"mode" (array float_exact) want (floats v);
      equal ~msg:"mode's fraction" (array float_exact) fraction (Nx.to_array f))

(* A grid turned and placed anywhere has corners a few ulps off the integers:
   slivers of area near 1e-10 at most. *)
let own_turned_grid =
  prop "on its own turned grid, average is the array to slivers"
    Gen.(
      triple (float_range 0. 1000.) (float_range 0. 1000.)
        (float_range (-1.5) 1.5))
    (fun (dx, dy, t) ->
      let g =
        turned (utm 33 true) ~rows:6 ~cols:7 ~x:(500000. +. dx)
          ~y:(5300000. +. dy) 10. t
      in
      let x, valid = source 1009 g in
      let ok = Nx.to_array valid in
      let v, f = Warp.average ~k:3 ~valid g (corners ~src:g ~dst:g) x in
      let v = floats v and f = Nx.to_array f in
      Array.iteri
        (fun k x ->
          if ok.(k) then begin
            equal ~msg:"value" (float 1e-6) x v.(k);
            equal ~msg:"fraction" (float 1e-9) 1. f.(k)
          end
          else equal ~msg:"no data" (float 1e-9) 0. f.(k))
        (floats x))

let constant =
  prop "a constant field stays constant"
    Gen.(
      triple (float_range (-1.5) 1.5) (float_range 0.3 3.)
        (pair bool (float_range (-6.) 6.)))
    (fun (t, d, (negative, e)) ->
      let c = (if negative then -1. else 1.) *. (10. ** e) in
      let src =
        north_up (utm 33 true) ~rows:16 ~cols:16 ~x:500000. ~y:5300000. 1.
      in
      let dst =
        turned (utm 33 true) ~rows:8 ~cols:8 ~x:500003.7 ~y:5299991.2 d t
      in
      let _, valid = source 1 src in
      let x = Nx.full Nx.float64 [| 16; 16 |] c in
      let v, f = Warp.average ~k:6 ~valid src (corners ~src ~dst) x in
      let f = Nx.to_array f in
      Array.iteri
        (fun k v ->
          is_true ~msg:"fraction in [0, 1]" (f.(k) >= 0. && f.(k) <= 1.);
          if f.(k) > 0. then
            equal ~msg:"value" (float_rel ~rel:1e-12 ~abs:0.) c v
          else equal ~msg:"no data" float_exact 0. v)
        (floats v))

(* The area of each footprint of [q], by the shoelace formula. *)
let footprint_areas q =
  let rows = Nx.dim 0 q - 1 and cols = Nx.dim 1 q - 1 in
  Array.init (rows * cols) (fun k ->
      let i = k / cols and j = k mod cols in
      let at (a, b) =
        (Nx.item [ i + a; j + b; 0 ] q, Nx.item [ i + a; j + b; 1 ] q)
      in
      let c = List.map at [ (0, 0); (0, 1); (1, 1); (1, 0) ] in
      let next = List.tl c @ [ List.hd c ] in
      let twice =
        List.fold_left2
          (fun s (y, x) (y', x') -> s +. (x *. y') -. (x' *. y))
          0. c next
      in
      Float.abs twice /. 2.)

let conservation =
  prop ~count:30 "the footprints that cover the source conserve its integral"
    Gen.(pair (float_range (-1.5) 1.5) (float_range 14.5 19.))
    (fun (t, d) ->
      let src =
        north_up (utm 33 true) ~rows:12 ~cols:12 ~x:500000. ~y:5300000. 10.
      in
      (* 12 cells of [d] ≥ 14.5 m around the source's centre reach past its
         corners, 60√2 m from it, at any angle. *)
      let half = 6. *. d in
      let x0 = 500060. -. (half *. (cos t +. sin t))
      and y0 = 5299940. -. (half *. (sin t -. cos t)) in
      let dst = turned (utm 33 true) ~rows:12 ~cols:12 ~x:x0 ~y:y0 d t in
      let x, valid = source 1009 src in
      let q = corners ~src ~dst in
      let v, f = Warp.average ~k:4 ~valid src q x in
      let v = floats v and f = Nx.to_array f in
      let a = footprint_areas q in
      let mine = ref 0. in
      Array.iteri (fun k v -> mine := !mine +. (v *. f.(k) *. a.(k))) v;
      let ok = Nx.to_array valid in
      let want = ref 0. in
      Array.iteri (fun k x -> if ok.(k) then want := !want +. x) (floats x);
      equal (float (1e-9 *. !want)) !want !mine)

(* A destination of 2 × 2 source cells, one cell up and left of the source, so
   its edges cover half cells and its corners quarters. *)
let blocks =
  test "aligned blocks: the valid cells' mean, their share, the least mode"
    (fun () ->
      let src =
        north_up (utm 33 true) ~rows:8 ~cols:10 ~x:500000. ~y:5300000. 10.
      in
      let dst =
        north_up (utm 33 true) ~rows:5 ~cols:6 ~x:499990. ~y:5300010. 20.
      in
      let x, valid = source 1009 src and classes, _ = source 3 src in
      let q = corners ~src ~dst in
      let x = floats x
      and classes = floats classes
      and ok = Nx.to_array valid in
      let cells i j =
        List.filter_map
          (fun (a, b) ->
            let r = (2 * i) - 1 + a and c = (2 * j) - 1 + b in
            if r >= 0 && r < 8 && c >= 0 && c < 10 && ok.((r * 10) + c) then
              Some ((r * 10) + c)
            else None)
          [ (0, 0); (0, 1); (1, 0); (1, 1) ]
      in
      let mean = Array.make 30 0.
      and share = Array.make 30 0.
      and least = Array.make 30 0. in
      for i = 0 to 4 do
        for j = 0 to 5 do
          let ks = cells i j and k = (i * 6) + j in
          let n = float_of_int (List.length ks) in
          share.(k) <- n /. 4.;
          if ks <> [] then begin
            mean.(k) <- List.fold_left (fun s k -> s +. x.(k)) 0. ks /. n;
            let count v =
              List.length (List.filter (fun k -> classes.(k) = v) ks)
            in
            let best =
              List.fold_left
                (fun b k ->
                  let v = classes.(k) in
                  if count v > count b || (count v = count b && v < b) then v
                  else b)
                classes.(List.hd ks)
                ks
            in
            least.(k) <- best
          end
        done
      done;
      let v, f =
        Warp.average ~k:4 ~valid src q (Nx.create Nx.float64 [| 8; 10 |] x)
      in
      equal ~msg:"mean" (array float_exact) mean (floats v);
      equal ~msg:"fraction" (array float_exact) share (Nx.to_array f);
      let v, _ =
        Warp.mode ~k:4 ~valid src q
          (Nx.cast Nx.uint8 (Nx.create Nx.float64 [| 8; 10 |] classes))
      in
      equal ~msg:"mode" (array float_exact) least (floats v);
      is_true ~msg:"some cell has no data"
        (Array.exists (fun s -> s = 0.) share);
      is_true ~msg:"some ties" (Array.exists (fun s -> s = 0.5) share))

let unit_squares =
  prop "over one source cell's square at a point, average is bilinear"
    Gen.(pair (float_range (-2.) 2.) (float_range (-2.) 2.))
    (fun (r, c) ->
      let g = north_up (utm 33 true) ~rows:8 ~cols:8 ~x:0. ~y:0. 1. in
      let x, valid = source 1009 g in
      let q = quads ~rows:10 ~cols:10 (r, c) (1., 0.) (0., 1.) in
      let p = Nx.add_s (Nx.slice [ R (0, 10); R (0, 10) ] q) 0.5 in
      let b, ok = read (Warp.bilinear ~valid g p x) in
      let v = floats (fst (Warp.average ~k:3 ~valid g q x)) in
      Array.iteri (fun k ok -> if ok then equal (float 1e-9) b.(k) v.(k)) ok)

(* On a source that spans 360°, bilinear reads its columns modulo their
   number, as the area rules do, so the law holds at its seam. Reading the
   seam as an edge, bilinear gave 0.8727 at 179.9° where the average of the
   cell centred there is 0.1745. The same points on the source rolled to put
   its seam at 0° give the same values, on sources running east and west. *)
let bilinear_seam =
  test "bilinear reads a 360° source across its seam" (fun () ->
      let geo ~x cx =
        Grid.v Crs.lonlat ~rows:10 ~cols:360 ~origin:(x, 10.)
          ~row_step:(0., -1.) ~col_step:(cx, 0.)
      in
      let src = geo ~x:(-180.) 1. in
      let field (g : Grid.t) =
        let lon = Nx.slice [ A; A; I 0 ] (Grid.centres ~at:host g) in
        Nx.mul_s (Nx.sin (Nx.mul_s lon (Float.pi /. 180.))) 100.
      in
      let lons = [ 0.3; 179.4; 179.9; -179.9; -179.6 ] in
      let at lon lat = Nx.create Nx.float64 [| 1; 2 |] [| lon; lat |] in
      List.iter
        (fun lon ->
          let p = Warp.pixel src (at lon 5.3) in
          let b = Nx.item [ 0 ] (fst (Warp.bilinear src p (field src))) in
          let q =
            Warp.pixel src
              (Nx.create Nx.float64 [| 2; 2; 2 |]
                 [| lon -. 0.5; 5.8; lon +. 0.5; 5.8; lon -. 0.5; 4.8;
                    lon +. 0.5; 4.8 |])
          in
          let v, _ = Warp.average ~k:3 src q (field src) in
          equal ~msg:(Printf.sprintf "at %g°" lon) (float 1e-9)
            (Nx.item [ 0; 0 ] v) b)
        lons;
      (* A point past the last column has no cell, as on any source. *)
      equal result
        ([| 0. |], [| false |])
        (read (Warp.bilinear src (pixels [ (4.7, 360.2) ]) (field src)));
      let p = Nx.concatenate ~axis:0 (List.map (fun l -> at l 5.3) lons) in
      List.iter
        (fun (g : Grid.t) ->
          let v, ok = Warp.bilinear g (Warp.pixel g p) (field g) in
          let v', ok' = Warp.bilinear src (Warp.pixel src p) (field src) in
          equal (array (float 1e-9)) (Nx.to_array v') (Nx.to_array v);
          equal (array bool) (Nx.to_array ok') (Nx.to_array ok))
        [ geo ~x:0. 1.; geo ~x:180. (-1.) ])


let inside_one_cell =
  test "footprints inside one source cell read it, as nearest" (fun () ->
      let g = north_up (utm 33 true) ~rows:5 ~cols:5 ~x:0. ~y:0. 1. in
      let x, valid = source 1009 g in
      let q = quads ~rows:20 ~cols:20 (0., 0.) (0.25, 0.) (0., 0.25) in
      let p = Nx.add_s (Nx.slice [ R (0, 20); R (0, 20) ] q) 0.125 in
      let v, ok = read (Warp.nearest ~valid g p x) in
      let near = Array.mapi (fun k v -> if ok.(k) then v else Float.nan) v in
      agree ~msg:"average"
        (fun _ -> 0.)
        near
        (with_nan (Warp.average ~k:2 ~valid g q x));
      agree ~msg:"mode"
        (fun _ -> 0.)
        near
        (with_nan (Warp.mode ~k:2 ~valid g q (Nx.cast Nx.int16 x))))

let mode_by_area =
  test "mode weighs each class by its area, and ties go to the least" (fun () ->
      let one x q =
        let v = fst (Warp.mode ~k:3 (unit_grid x) q x) in
        Nx.to_array (Nx.cast Nx.float64 v)
      in
      let x l = Nx.create Nx.int32 [| 2; 2 |] l in
      let square a b = quads ~rows:1 ~cols:1 (a, a) (b, 0.) (0., b) in
      (* One cell of class 2 covers 1, three of class 1 cover 0.69. *)
      equal ~msg:"area" (array float_exact) [| 2. |]
        (one (x [| 2l; 1l; 1l; 1l |]) (square 0. 1.3));
      equal ~msg:"tie" (array float_exact) [| 1. |]
        (one (x [| 1l; 2l; 2l; 1l |]) (square 0. 2.));
      equal ~msg:"tie" (array float_exact) [| 2. |]
        (one (x [| 3l; 2l; 3l; 2l |]) (square 0. 2.));
      equal ~msg:"negative" (array float_exact) [| -3. |]
        (one (x [| -3l; 5l; 5l; -3l |]) (square 0. 2.));
      equal ~msg:"inside one cell" (array float_exact) [| -3. |]
        (one (x [| -3l; 5l; 5l; -3l |]) (square 0.2 0.5)))

let partial =
  test "a footprint's fraction is its share over cells with data" (fun () ->
      let x = Nx.ones Nx.float64 [| 4; 4 |] in
      let valid =
        Nx.init Nx.bool [| 4; 4 |] (fun a -> a.(0) <> 1 || a.(1) <> 1)
      in
      let fraction q =
        Nx.to_array (snd (Warp.average ~k:4 ~valid (unit_grid x) q x))
      in
      let square r c s = quads ~rows:1 ~cols:1 (r, c) (s, 0.) (0., s) in
      equal
        (array (float 1e-15))
        [| 0.25 |]
        (fraction (square (-0.5) (-0.5) 1.));
      equal (array (float 1e-15)) [| 0.5 |] (fraction (square 3. 0. 2.));
      equal (array (float 1e-15)) [| 0.75 |] (fraction (square 2.5 0. 2.));
      equal (array (float 1e-15)) [| 0.75 |] (fraction (square 0.5 0.5 1.));
      equal
        (array (float 1e-15))
        [| 1. -. (0.3 *. 0.3 /. 4.) |]
        (fraction (square 1.7 1.7 2.));
      equal (array float_exact) [| 0. |] (fraction (square 5. 5. 1.));
      let nan =
        Nx.init Nx.float64 [| 2; 2; 2 |] (fun a ->
            if a.(0) = 1 && a.(1) = 1 then Float.nan
            else float_of_int a.(a.(2)))
      in
      let v, f = Warp.average ~k:3 ~valid (unit_grid x) nan x in
      equal ~msg:"a NaN corner" (array float_exact) [| 0.; 0. |]
        (Array.append (floats v) (Nx.to_array f)))

(* A diamond whose candidate cells hold cell (0, 0), which it does not
   meet: its edge crosses column 1 at row 1.1. *)
let nan_unreached =
  test "a NaN the footprint does not reach stays out" (fun () ->
      let diamond =
        Nx.create Nx.float64 [| 2; 2; 2 |]
          [| 0.6; 1.5; 2.; 2.9; 2.; 0.1; 3.4; 1.5 |]
      in
      let at nan =
        let x =
          Nx.init Nx.float64 [| 4; 4 |] (fun a ->
              if (a.(0), a.(1)) = nan then Float.nan else 1.)
        in
        let v, f = Warp.average ~k:4 (unit_grid x) diamond x in
        (Nx.item [ 0; 0 ] v, Nx.item [ 0; 0 ] f)
      in
      equal ~msg:"unreached"
        (pair (float 1e-12) (float 1e-12))
        (1., 1.) (at (0, 0));
      is_true ~msg:"reached" (Float.is_nan (fst (at (1, 1)))))

(* The same turned cells listed from the south: their footprints turn the
   other way. *)
let mirrored =
  test "a mirrored destination reads its cells in the reverse order" (fun () ->
      let src =
        north_up (utm 33 true) ~rows:12 ~cols:12 ~x:500000. ~y:5300000. 10.
      in
      let dst =
        turned (utm 33 true) ~rows:5 ~cols:6 ~x:500013.3 ~y:5299987.1 17. 0.31
      in
      let (rx, ry), o = (dst.row_step, Grid.world dst (pixels [ (5., 0.) ])) in
      let flipped =
        Grid.v dst.crs ~rows:5 ~cols:6
          ~origin:(Nx.item [ 0; 0 ] o, Nx.item [ 0; 1 ] o)
          ~row_step:(-.rx, -.ry) ~col_step:dst.col_step
      in
      let x, valid = source 1009 src in
      let at dst = Warp.average ~k:4 ~valid src (corners ~src ~dst) x in
      let v, f = at dst and v', f' = at flipped in
      (* The flipped grid's origin is a rounded point of the first's, so their
         corners agree to 1e-10 cell. *)
      let rev a = Nx.to_array (Nx.flip ~axes:[ 0 ] a) in
      equal ~msg:"value" (array (float 1e-7)) (floats v) (rev v');
      equal ~msg:"fraction" (array (float 1e-9)) (Nx.to_array f) (rev f');
      is_true ~msg:"data" (Array.exists (fun f -> f > 0.5) (Nx.to_array f)))

(* A lattice of 2 × 2 footprints whose last corner is NaN: the four
   footprints around it have no data, and the others read as without it. *)
let nan_corner =
  test "a NaN corner leaves its footprints no data, the others' values and \
        gradient"
    (fun () ->
      let x = sines 12 in
      let q = quads ~rows:3 ~cols:3 (0.7, 0.9) (2.3, 0.4) (-0.3, 2.1) in
      let holed =
        Nx.where
          (Nx.init Nx.bool [| 4; 4; 2 |] (fun a -> a.(0) = 1 && a.(1) = 1))
          (Nx.full Nx.float64 [| 4; 4; 2 |] Float.nan)
          q
      in
      let v, f = Warp.average ~k:4 (unit_grid x) q x
      and v', f' = Warp.average ~k:4 (unit_grid x) holed x in
      let touched k = k / 3 <= 1 && k mod 3 <= 1 in
      let v = Nx.to_array v and f = Nx.to_array f
      and v' = Nx.to_array v' and f' = Nx.to_array f' in
      Array.iteri
        (fun k _ ->
          if touched k then
            equal ~msg:"no data" (pair float_exact float_exact) (0., 0.)
              (v'.(k), f'.(k))
          else
            equal ~msg:"the others" (pair float_exact float_exact)
              (v.(k), f.(k)) (v'.(k), f'.(k)))
        v;
      let loss q = Nx.sum (fst (Warp.average ~k:4 (unit_grid x) q x)) in
      let g = Nx.to_array (Rune.grad' loss holed) in
      is_true ~msg:"finite gradient" (Array.for_all Float.is_finite g))

(* A rectangle of [h] by [w] cells inside cell (0, 0), from [at]. *)
let sliver ?(at = (0.3, 0.2)) h w = quads ~rows:1 ~cols:1 at (h, 0.) (0., w)

(* Rounding's bound in a cell is 24 u M W. For a sliver inside a cell, M is 1
   and W twice its width, so the bound is an area of 48 u w: the sliver is
   reached where it is thicker than 48 u. Its W, the edges' widths, makes it
   scale-free, and M is at least 1 near the cell's corner too. *)
let tiny_footprints =
  test "a footprint inside a valid cell reads it past rounding's bound"
    (fun () ->
      let x = Nx.full Nx.float64 [| 2; 2 |] 7. in
      let read q =
        let v, f = Warp.average ~k:2 (unit_grid x) q x in
        (Nx.item [ 0; 0 ] v, Nx.item [ 0; 0 ] f)
      in
      let side = sqrt 9e-13 in
      (* The footprint's area comes from corners near 0.3, so a footprint
         1e-6 wide has it to 3e-11, and one 1e-10 wide to 3e-7. *)
      let both = pair float_exact (float 1e-6) in
      equal ~msg:"9e-13 cells" both (7., 1.) (read (sliver side side));
      equal ~msg:"1e-20 cells" both (7., 1.) (read (sliver 1e-10 1e-10));
      let u = epsilon_float /. 2. in
      equal ~msg:"a quarter over the bound" both (7., 1.)
        (read (sliver (60. *. u) 0.5));
      equal ~msg:"a quarter under the bound" both (0., 0.)
        (read (sliver (36. *. u) 0.5));
      equal ~msg:"half the bound, by the cell's corner" both (0., 0.)
        (read (sliver ~at:(0.01, 0.01) (24. *. u) 0.1)))

(* A footprint 1e7 cells long, beside the source: its candidates are clipped
   to the source, so it costs one cell. *)
let long_outside =
  test "a long footprint outside the source costs one candidate" (fun () ->
      let x = Nx.full Nx.float64 [| 2; 2 |] 7. in
      let q = quads ~rows:1 ~cols:1 (3., 0.) (1e7, 0.) (0., 1.) in
      let v, f = Warp.average ~k:2 (unit_grid x) q x in
      equal (pair float_exact float_exact) (0., 0.)
        (Nx.item [ 0; 0 ] v, Nx.item [ 0; 0 ] f))

(* Whether the rectangle [q], [[2; 2; 2]], and cell (i, j) lie apart by more
   than 1e-9 along one of their axes. *)
let apart q (i, j) =
  let corner a b = (Nx.item [ a; b; 0 ] q, Nx.item [ a; b; 1 ] q) in
  let quad = [ corner 0 0; corner 0 1; corner 1 1; corner 1 0 ] in
  let fi = float_of_int i and fj = float_of_int j in
  let cell =
    [ (fi, fj); (fi, fj +. 1.); (fi +. 1., fj +. 1.); (fi +. 1., fj) ]
  in
  let axis (y0, x0) (y1, x1) = (x1 -. x0, -.(y1 -. y0)) in
  let along (ay, ax) l = List.map (fun (y, x) -> (y *. ay) +. (x *. ax)) l in
  let lo l = List.fold_left Float.min Float.infinity l
  and hi l = List.fold_left Float.max Float.neg_infinity l in
  List.exists
    (fun a ->
      let p = along a quad and c = along a cell in
      lo p > hi c +. 1e-9 || lo c > hi p +. 1e-9)
    [ (1., 0.); (0., 1.); axis (List.nth quad 0) (List.nth quad 1);
      axis (List.nth quad 1) (List.nth quad 2) ]

(* NaN in every cell a turned rectangle stays apart from: a cell must pass
   rounding's bound to give its value, and those cells hold areas of
   rounding. *)
let rounding_stays_out =
  prop "NaN in the cells a footprint does not meet stays out"
    Gen.(
      pair
        (pair (float_range 1.5 5.5) (float_range 1.5 5.5))
        (triple (float_range 0.2 2.) (float_range 0.2 2.) (float_range 0. 3.2)))
    (fun ((r, c), (h, w, t)) ->
      let q =
        quads ~rows:1 ~cols:1
          (r -. (h *. cos t /. 2.) +. (w *. sin t /. 2.),
           c -. (h *. sin t /. 2.) -. (w *. cos t /. 2.))
          (h *. cos t, h *. sin t) (-.w *. sin t, w *. cos t)
      in
      let x =
        Nx.init Nx.float64 [| 7; 7 |] (fun a ->
            if apart q (a.(0), a.(1)) then Float.nan else 1.)
      in
      let v = Nx.item [ 0; 0 ] (fst (Warp.average ~k:6 (unit_grid x) q x)) in
      equal (float 1e-12) 1. v)

(* The loss of [rule]'s result on the 4 × 4 footprints [q] over [x], each
   footprint weighed by a fixed number. *)
let weighed rule x q =
  let w =
    Nx.init Nx.float64 [| 4; 4 |] (fun a ->
        1. +. (0.37 *. float_of_int (((a.(0) * 5) + (a.(1) * 3)) mod 7)))
  in
  Nx.sum (Nx.mul w (rule x q))

(* [h] at the [k]th element of an array of [shape], 0 elsewhere. *)
let basis shape k h =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.reshape shape
    (Nx.init Nx.float64 [| n |] (fun a -> if a.(0) = k then h else 0.))

let central f q h =
  Array.init (Nx.numel q) (fun k ->
      let e = basis (Nx.shape q) k h in
      (Nx.item [] (f (Nx.add q e)) -. Nx.item [] (f (Nx.sub q e))) /. (2. *. h))

let value_rule ~k x q = fst (Warp.average ~k (unit_grid x) q x)
let fraction_rule ~k x q = snd (Warp.average ~k (unit_grid x) q x)

let area_gradient =
  test "the gradient in the corners and the values is the derivative" (fun () ->
      let x = sines 14 in
      (* Footprints turned and stretched, some past the source's edge. *)
      let q = quads ~rows:4 ~cols:4 (-0.73, 2.27) (2.9, 0.83) (-0.61, 3.1) in
      List.iter
        (fun (name, rule) ->
          let loss = weighed rule x in
          let g = Nx.to_array (Rune.grad' loss q)
          and fd = central loss q 1e-6 in
          Array.iteri
            (fun k g ->
              equal ~msg:name
                (float (1e-5 *. Float.max 1. (Float.abs g)))
                fd.(k) g)
            g)
        [ ("value", value_rule ~k:5); ("fraction", fraction_rule ~k:5) ];
      let loss x = weighed (value_rule ~k:5) x q in
      let g = Nx.to_array (Rune.grad' loss x) and fd = central loss x 1e-6 in
      Array.iteri (fun k g -> equal ~msg:"in x" (float 1e-6) fd.(k) g) g)

let right_hand =
  test "on the lattice the gradient is the right-hand derivative" (fun () ->
      let x = sines 12 in
      let q = quads ~rows:4 ~cols:4 (1., 2.) (2., 0.) (0., 2.) in
      let loss = weighed (value_rule ~k:4) x in
      let g = Nx.to_array (Rune.grad' loss q) in
      let h = 1e-7 and l0 = Nx.item [] (loss q) in
      Array.iteri
        (fun k g ->
          let e = basis (Nx.shape q) k h in
          let right = (Nx.item [] (loss (Nx.add q e)) -. l0) /. h in
          let left = Nx.item [] (loss (Nx.sub q e)) -. l0 in
          equal ~msg:"right-hand"
            (float (1e-5 *. Float.max 1. (Float.abs g)))
            right g;
          equal ~msg:"continuous" (float 1e-6) 0. left)
        g)

(* A [Lonlat] source across the antimeridian, written from its west edge at
   179.998 or at -180.002, and a 30 m grid of UTM 1S over that edge and the
   antimeridian. *)
let lonlat_area =
  test "a Lonlat source's footprints across its west edge and the antimeridian"
    (fun () ->
      let x, valid = source 1009 (lonlat_at 179.998) in
      let o =
        Crs.transform ~src:Crs.lonlat ~dst:(utm 1 false)
          (pixels [ (179.9979, -16.9984) ])
      in
      let dst =
        north_up (utm 1 false) ~rows:7 ~cols:11
          ~x:(Nx.item [ 0; 0 ] o)
          ~y:(Nx.item [ 0; 1 ] o)
          30.
      in
      let at src = Warp.average ~k:4 ~valid src (corners ~src ~dst) x in
      let v, f = at (lonlat_at 179.998)
      and v', f' = at (lonlat_at (-180.002)) in
      (* A corner [pixel] takes to the far end of the turn and back keeps the
         rounding of a column near 3.6e6, 5e-10 of a cell. *)
      equal ~msg:"value" (array (float 1e-6)) (floats v) (floats v');
      equal ~msg:"fraction"
        (array (float 1e-9))
        (Nx.to_array f) (Nx.to_array f');
      let f = Nx.to_array f in
      is_true ~msg:"west edge" (f.(0) > 0. && f.(0) < 1.);
      is_true ~msg:"antimeridian" (Array.exists (fun f -> f = 1.) f))

let seam =
  test "a footprint across a 360° source's west edge reads both sides"
    (fun () ->
      let src = north_up Crs.lonlat ~rows:3 ~cols:360 ~x:(-180.) ~y:1.5 1. in
      let x =
        Nx.init Nx.float64 [| 3; 360 |] (fun a ->
            float_of_int ((a.(0) * 1000) + a.(1)))
      in
      let dst = north_up Crs.lonlat ~rows:1 ~cols:1 ~x:179.5 ~y:0.5 1. in
      let v, f = Warp.average ~k:3 src (corners ~src ~dst) x in
      equal ~msg:"value" (array float_exact) [| 1179.5 |] (floats v);
      equal ~msg:"fraction" (array float_exact) [| 1. |] (Nx.to_array f))

(* A footprint across a 360° source's west meridian whose east corner lies on
   it, 360° on: the 360° east of the meridian are closed on the east, so the
   footprint stays at the source's east end. *)
let closed_east =
  test "a footprint ending on the 360° from the west meridian stays there"
    (fun () ->
      let src = north_up Crs.lonlat ~rows:3 ~cols:360 ~x:(-180.) ~y:1.5 1. in
      let x =
        Nx.init Nx.float64 [| 3; 360 |] (fun a ->
            float_of_int ((a.(0) * 1000) + a.(1)))
      in
      let dst = north_up Crs.lonlat ~rows:1 ~cols:1 ~x:179. ~y:0.5 1. in
      let v, f = Warp.average ~k:3 src (corners ~src ~dst) x in
      equal ~msg:"value" (array float_exact) [| 1359. |] (floats v);
      equal ~msg:"fraction" (array float_exact) [| 1. |] (Nx.to_array f))

(* The same cells of a Lonlat source written east from its west edge and west
   from its east edge, with a grid over that west edge. *)
let westward =
  test "a Lonlat source whose columns run west reads as one running east"
    (fun () ->
      let east =
        Grid.v Crs.lonlat ~rows:4 ~cols:8 ~origin:(179.96, -10.)
          ~row_step:(0., -0.01) ~col_step:(0.01, 0.)
      and west =
        Grid.v Crs.lonlat ~rows:4 ~cols:8 ~origin:(180.04, -10.)
          ~row_step:(0., -0.01) ~col_step:(-0.01, 0.)
      in
      let x =
        Nx.init Nx.float64 [| 4; 8 |] (fun a ->
            float_of_int ((a.(0) * 100) + a.(1) + 1))
      in
      let dst =
        Grid.v Crs.lonlat ~rows:3 ~cols:6 ~origin:(179.955, -10.004)
          ~row_step:(0., -0.0093) ~col_step:(0.0131, 0.)
      in
      let at src x = Warp.average ~k:3 src (corners ~src ~dst) x in
      let v, f = at east x and v', f' = at west (Nx.flip ~axes:[ 1 ] x) in
      (* Its pixel points come from two origins, so they agree to rounding. *)
      equal ~msg:"value" (array (float 1e-9)) (floats v) (floats v');
      equal ~msg:"fraction" (array (float 1e-9)) (Nx.to_array f)
        (Nx.to_array f');
      let f = Nx.to_array f in
      is_true ~msg:"across the west edge" (f.(0) > 0. && f.(0) < 1.))

(* rune's bound on a float64 sum of [n] terms against its exact value,
   γ(β(n)) Σ|x| (rune.mli, Numerics, "Reductions"; β as nx_kernel.mli's
   error bounds define it), as a factor of Σ|x|. *)
let sum_bound n =
  let u = epsilon_float /. 2. in
  let beta = ((5 * min n 256) + 3) / 4 + ((n + 127) / 128) in
  let b = float_of_int beta *. u in
  b /. (1. -. b)

(* Each fixture's rules compiled by rune, against the same rules run
   eagerly. A compiled sum combines its terms in an order the compiler
   chooses, so the two agree bit for bit where the sums are exact, as on the
   source's lattice ([exact]), and else within the sums' bound. The rules
   have two sums of [k²] terms per footprint, both flat over its candidates,
   and the rest of their arithmetic is exact compiled: the mean S / W, with
   S = Σ aⱼ xⱼ and W = Σ aⱼ over the cells with data, and the fraction,
   W / A for the footprint's area A. Eager and compiled each lie within
   g = γ(β(k²)) Σ|·| of each exact sum, so they differ by at most 2 g Σ|·|,
   and to first order the mean by 2 g (Σ aⱼ|xⱼ| + |S / W| W) / W, that is
   2 g (the mean of |x| + |the mean|), and the fraction by 2 g times
   itself; each division rounds once more, u of its result. The mean of |x|
   is [average] of |x|, eagerly. *)
let compiled =
  cases "the area rules compile, to the sums' bound" ~id:fst
    [ ("aligned20", true); ("shifted25", false); ("holes30", true);
      ("ties30", true); ("mode20", true); ("mode25", false);
      ("xzone", false) ]
    (fun (name, exact) ->
      let f = area_fixture name in
      let q = corners ~src:f.src ~dst:f.dst and k = law name in
      let x, valid =
        match name with
        | "xzone" ->
            let x =
              Nx.create Nx.float64 [| f.src.rows; f.src.cols |]
                (List.assoc "source" f.outputs)
            in
            (x, Nx.not_equal_s x 0.)
        | "mode20" | "mode25" -> source 4 f.src
        | _ -> source 1009 f.src
      in
      let g = 2. *. sum_bound (k * k) and u = epsilon_float /. 2. in
      let magnitude =
        floats (fst (Warp.average ~k ~valid f.src q (Nx.abs x)))
      in
      (* [bound k v], for the eager result [v] at element [k]. *)
      let same ?(exact = exact) msg bound rule =
        let eager = floats (rule q) and jit = floats (Rune.jit' rule q) in
        Array.iteri
          (fun i v ->
            let t = bound i v in
            let close = if exact || t = 0. then float_exact else float t in
            equal ~msg close v jit.(i))
          eager
      in
      let mean i v =
        (g *. (magnitude.(i) +. Float.abs v)) +. (u *. Float.abs v)
      and fraction _ v = (g +. u) *. Float.abs v in
      if String.starts_with ~prefix:"mode" name then begin
        let x = Nx.cast Nx.uint8 x in
        same ~exact:true "mode" fraction (fun q ->
            fst (Warp.mode ~k ~valid f.src q x));
        same "mode's fraction" fraction (fun q ->
            snd (Warp.mode ~k ~valid f.src q x))
      end
      else begin
        same "average" mean (fun q -> fst (Warp.average ~k ~valid f.src q x));
        same "fraction" fraction (fun q ->
            snd (Warp.average ~k ~valid f.src q x));
        if exact then
          let x = Nx.cast Nx.uint16 x in
          same "rounded" fraction (fun q ->
              fst (Warp.average ~k ~valid f.src q x))
      end)

(* Sources and destinations that place footprints every way the rules do:
   across 180° and across a 360° source's west edge at steps whose turn is
   not a whole number of columns, on sources running west, from 0°, narrower
   than 360°, a gridline-registered 361 columns cut to their first 360,
   rotated, projected,
   and destinations in other systems, past the source's edges and off it. *)
let area_cases =
  let lonlat ~rows ~cols ~x ~y ?(rx = 0.) ?(cy = 0.) cx ry =
    Grid.v Crs.lonlat ~rows ~cols ~origin:(x, y) ~row_step:(rx, ry)
      ~col_step:(cx, cy)
  in
  let deg ?(rows = 2) ?(cols = 4) x y d =
    north_up Crs.lonlat ~rows ~cols ~x ~y d
  in
  let in_utm zone x =
    north_up (utm zone true) ~rows:2 ~cols:3 ~x ~y:200000. 100000.
  in
  let global step =
    let cols = int_of_float (Float.round (360. /. step)) in
    lonlat ~rows:40 ~cols ~x:(-180.) ~y:2. (360. /. float_of_int cols) (-0.1)
  in
  let seams step =
    let src = global step and at s = Printf.sprintf "%g° global, %s" step s in
    [
      (at "1° across 180°", src, deg 178.5 0.5 1.);
      (at "1° from -181.5°", src, deg (-181.5) 0.5 1.);
      (at "1° from 0°", src, deg 0. 0.5 1.);
      (at "UTM 60N across 180°", src, in_utm 60 683000.);
      (at "UTM 1N across 180°", src, in_utm 1 16000.);
    ]
  in
  let westward = lonlat ~rows:40 ~cols:360 ~x:180. ~y:2. (-1.) (-0.1)
  and from0 = lonlat ~rows:40 ~cols:3600 ~x:0. ~y:2. 0.1 (-0.1)
  and narrow = lonlat ~rows:20 ~cols:200 ~x:170. ~y:1. 0.1 (-0.1)
  and gridline = lonlat ~rows:21 ~cols:361 ~x:(-180.5) ~y:10.5 1. (-1.)
  and rotated t =
    lonlat ~rows:60 ~cols:60 ~x:10. ~y:9. ~rx:(0.1 *. sin t) ~cy:(0.1 *. sin t)
      (0.1 *. cos t) (-0.1 *. cos t)
  in
  let z33 =
    north_up (utm 33 true) ~rows:80 ~cols:80 ~x:700000. ~y:5300000. 10.
  in
  let z34 ~rows d dx =
    let o =
      Crs.transform ~src:(utm 33 true) ~dst:(utm 34 true)
        (Nx.create Nx.float64 [| 2 |] [| 700100.; 5299900. |])
    in
    north_up (utm 34 true) ~rows ~cols:rows ~x:(Nx.item [ 0 ] o -. dx)
      ~y:(Nx.item [ 1 ] o +. dx) d
  in
  List.concat_map seams [ 0.1; 0.2; 0.05; 1.; 360. /. 7. ]
  @ [
      ("1° running west, across 180°", westward, deg ~cols:6 177.5 0.5 1.);
      ("1° running west, inside", westward, deg ~cols:6 10.25 0.5 1.);
      ("0.1° global from 0°, across 0°", from0, deg (-1.5) 0.5 1.);
      ("170°E to 170°W, across its west edge", narrow,
        deg ~rows:1 ~cols:2 169.5 0.5 1.);
      ("170°E to 170°W, across 180°", narrow,
        deg ~rows:1 ~cols:4 178.3 0.5 1.);
      ( "361 columns of 1° windowed to a turn, across 180°",
        Grid.window ~row:0 ~col:0 ~rows:21 ~cols:360 gridline,
        deg 178. 5. 1. );
      ("rotated by 0.3 rad", rotated 0.3, deg 11. 7.5 0.7);
      ("rotated by -0.3 rad, across its west corner", rotated (-0.3),
        deg 8. 3.6 0.37);
      ("UTM 33 onto UTM 34", z33, z34 ~rows:5 60. 0.);
      ("UTM 34 past the source's edges", z33, z34 ~rows:12 90. 300.);
      ("lonlat off the source", narrow, deg 175. 30. 10.);
      ( "lonlat past a UTM source's domain, NaN corners",
        north_up (utm 33 true) ~rows:40 ~cols:40 ~x:4.6e6 ~y:100000. 10000.,
        deg ~rows:3 ~cols:9 51. 0.4 0.5 );
    ]

(* The least k by search: the first the rule takes. *)
let least src q x =
  let rec go k =
    match Nx.to_array (fst (Warp.average ~k src q x)) with
    | _ -> k
    | exception Invalid_argument _ -> go (k + 1)
  in
  go 1

let span_least =
  cases "span is the least k" ~id:(fun (n, _, _) -> n) area_cases
    (fun (_, src, dst) ->
      let q = corners ~src ~dst and x, _ = source 1009 src in
      equal ~msg:"span" int (least src q x) (Warp.span src q))

(* Reader 6's case, 1° cells across 180° on a 0.1° global source: a k from
   the corners' differences was 3592, which exhausts memory. *)
let span_cases =
  test "span across 180°, and off the source" (fun () ->
      let src =
        north_up Crs.lonlat ~rows:1800 ~cols:3600 ~x:(-180.) ~y:90. 0.1
      in
      let dst = north_up Crs.lonlat ~rows:2 ~cols:4 ~x:178.5 ~y:0.5 1. in
      equal ~msg:"across 180°" int 11 (Warp.span src (corners ~src ~dst));
      let src = north_up Crs.lonlat ~rows:20 ~cols:3600 ~x:(-180.) ~y:1. 0.1 in
      let dst = north_up Crs.lonlat ~rows:2 ~cols:2 ~x:0. ~y:40. 10. in
      let q = corners ~src ~dst in
      equal ~msg:"off the source" int 1 (Warp.span src q);
      let v, f = Warp.average ~k:1 src q (Nx.ones Nx.float64 [| 20; 3600 |]) in
      equal ~msg:"no data" (array float_exact) (Array.make 8 0.)
        (Array.append (Nx.to_array v) (Nx.to_array f));
      raises_match (Exn.invalid_arg ~sub:"traced") (fun () ->
          let k q = Nx.scalar_like q (float_of_int (Warp.span src q)) in
          Rune.jit' k q))

(* A footprint over rows 0.5 to 3.5 spans 4 cells: k = 3 is too small. *)
let too_small =
  test "a k too small raises, eagerly and compiled" (fun () ->
      let x = Nx.ones Nx.float64 [| 5; 5 |] in
      let q = quads ~rows:1 ~cols:1 (0.5, 1.2) (3., 0.) (0., 1.) in
      let rule q = fst (Warp.average ~k:3 (unit_grid x) q x) in
      let spans =
        Exn.invalid_arg ~sub:"footprint [0, 0] spans 4 cells, more than k = 3"
      in
      raises_match spans (fun () -> Nx.to_array (rule q));
      raises_match spans (fun () -> Nx.to_array (Rune.jit' rule q));
      equal ~msg:"k = 4 suffices" (array float_exact) [| 1. |]
        (Nx.to_array (fst (Warp.average ~k:4 (unit_grid x) q x)));
      raises_match
        (Exn.invalid_arg ~sub:"k = 0 is below 1; use Warp.span for k")
        (fun () -> Warp.average ~k:0 (unit_grid x) q x);
      raises_match (Exn.invalid_arg ~sub:"k = 0 is below 1") (fun () ->
          Warp.mode ~k:0 (unit_grid x) q (Nx.cast Nx.int32 x)))

(* A mask has its values' shape: the cells, then the source's other axes,
   the same value along them, so a value and its mask join on any grid. *)
let mask_shapes =
  test "each rule's mask has its values' shape" (fun () ->
      let x =
        Nx.init Nx.float64 [| 3; 4; 2 |] (fun a ->
            float_of_int ((a.(0) * 8) + (a.(1) * 2) + a.(2)))
      in
      let valid = Nx.init Nx.bool [| 3; 4 |] (fun a -> a.(0) + a.(1) <> 3) in
      let p = pixels [ (0.5, 0.5); (1.2, 2.7); (2.9, 3.9); (5., 1.) ] in
      let same name (v, m) flat =
        equal ~msg:name (array int) (Nx.shape v) (Nx.shape m);
        let first = Nx.slice [ A; I 0 ] m and second = Nx.slice [ A; I 1 ] m in
        equal ~msg:name (array bool) (Nx.to_array first) (Nx.to_array second);
        equal ~msg:name (array bool) (Nx.to_array flat) (Nx.to_array first)
      in
      let g = unit_grid x in
      let n, nm = Warp.nearest ~valid g p x
      and _, nm0 = Warp.nearest ~valid g p (Nx.slice [ A; A; I 0 ] x) in
      same "nearest" (n, nm) nm0;
      let b, bm = Warp.bilinear ~valid g p x
      and _, bm0 = Warp.bilinear ~valid g p (Nx.slice [ A; A; I 0 ] x) in
      same "bilinear" (b, bm) bm0;
      let q = quads ~rows:2 ~cols:3 (0., 0.) (1., 0.) (0., 1.) in
      let a, af = Warp.average ~k:3 ~valid g q x in
      equal ~msg:"average" (array int) (Nx.shape a) (Nx.shape af);
      let m, mf = Warp.mode ~k:3 ~valid g q (Nx.cast Nx.int32 x) in
      equal ~msg:"mode" (array int) (Nx.shape m) (Nx.shape mf))

(* jord.mli's example on images of one channel, [[h; w; 1]], and a grid
   whose rows and columns differ, where a mask of another shape than its
   values would not join. *)
let example_join =
  test "jord.mli's example joins images of one channel" (fun () ->
      let utm33 = utm 33 true in
      let src = north_up utm33 ~rows:4 ~cols:6 ~x:0. ~y:40. 10.
      and src' = north_up utm33 ~rows:2 ~cols:3 ~x:0. ~y:40. 20.
      and dst = north_up utm33 ~rows:3 ~cols:5 ~x:2. ~y:38. 11. in
      let dn = Nx.init Nx.uint16 [| 4; 6; 1 |] (fun a -> (a.(0) * 6) + a.(1))
      and scl =
        Nx.init Nx.uint8 [| 2; 3; 1 |] (fun a -> if a.(1) = 2 then 9 else 4)
      in
      let band =
        {
          Band.name = None;
          common_name = None;
          center_wavelength = None;
          full_width_half_max = None;
          unit = None;
          kind = Band.Measurement;
          scale = 1.;
          offset = 0.;
          nodata = Some 0.;
          saturated = None;
        }
      in
      let join dn scl =
        let c = Warp.centres ~at:Nx.Host.on src.crs dst in
        let red, valid = Band.decode band Nx.float32 dn in
        let red, ok = Warp.bilinear ~valid src (Warp.pixel src c) red in
        let scl, seen = Warp.nearest src' (Warp.pixel src' c) scl in
        let masked =
          Sentinel2.Scl.(
            mem
              [ No_data; Saturated_or_defective; Cloud_medium_probability;
                Cloud_high_probability ]
              scl)
        in
        (red, Nx.(logical_and (logical_and ok seen) (logical_not masked)))
      in
      let red, mask = join dn scl in
      equal (array int) [| 3; 5; 1 |] (Nx.shape red);
      equal (array int) [| 3; 5; 1 |] (Nx.shape mask);
      (* The arrays of one value per cell give the same. *)
      let flat x = Nx.reshape [| Nx.dim 0 x; Nx.dim 1 x |] x in
      equal result
        (read (join (flat dn) (flat scl)))
        (read (red, mask)))

(* [src]'s block (row, col, rows, cols) of [a], its columns modulo
   [src.cols] where they pass the last, read in two parts as a program reads
   a file's. *)
let read_block (src : Grid.t) (row, col, rows, cols) a =
  let part col cols =
    Nx.slice [ R (row, row + rows); R (col, col + cols) ] a
  in
  if col + cols <= src.cols then part col cols
  else
    Nx.concatenate ~axis:1
      [ part col (src.cols - col); part 0 (col + cols - src.cols) ]

(* On area_window's block, at ~row ~col, the area rules give the whole
   array's values, fractions and classes, at the least k and past it. *)
let windowed_area =
  cases "an area rule on its block gives the whole array's bits"
    ~id:(fun (n, _, _) -> n)
    area_cases
    (fun (name, src, dst) ->
      let q = corners ~src ~dst and x, valid = source 1009 src in
      let classes = Nx.cast Nx.int32 (Nx.div_s x 100.) in
      let ((row, col, _, _) as b) = Warp.area_window src q in
      let cut a = read_block src b a in
      List.iter
        (fun k ->
          let msg s = Printf.sprintf "k %d: %s" k s in
          let v, f = Warp.average ~k ~valid src q x
          and v', f' =
            Warp.average ~k ~row ~col ~valid:(cut valid) src q (cut x)
          in
          equal ~msg:(msg "values") (array int64) (bits v) (bits v');
          equal ~msg:(msg "fractions") (array int64) (bits f) (bits f');
          let m, g = Warp.mode ~k ~valid src q classes
          and m', g' =
            Warp.mode ~k ~row ~col ~valid:(cut valid) src q (cut classes)
          in
          equal ~msg:(msg "classes") (array int64) (bits m) (bits m');
          equal ~msg:(msg "mode's fractions") (array int64) (bits g) (bits g');
          if name <> "lonlat off the source" then
            is_true ~msg:(msg "data")
              (Array.exists (fun a -> a > 0.) (floats f)))
        [ Warp.span src q; Warp.span src q + 2 ];
      (* The gradients in the corners, through the block, are the whole
         array's. *)
      let k = Warp.span src q in
      let sum ?row ?col x valid q =
        Nx.sum (fst (Warp.average ~k ?row ?col ~valid src q x))
      in
      equal ~msg:"gradient" (array int64)
        (bits (Rune.grad' (sum x valid) q))
        (bits (Rune.grad' (sum ~row ~col (cut x) (cut valid)) q)))

(* On a 360° source, strips of 1° columns [c, c + w), NaN between them: the
   block is the circle less the widest of several gaps, and an arc inside a
   longer one leaves no gap. *)
let block_gaps =
  test "area_window takes the widest gap" (fun () ->
      let src = north_up Crs.lonlat ~rows:20 ~cols:360 ~x:(-180.) ~y:10. 1. in
      let strip (c, w) =
        let a = float_of_int c -. 179.5 and b = float_of_int (c + w) -. 180.5 in
        let lonlat = [| a; 5.; b; 5.; a; 4.5; b; 4.5 |] in
        Warp.pixel src (Nx.create Nx.float64 [| 2; 2; 2 |] lonlat)
      in
      let nan = Nx.full Nx.float64 [| 2; 1; 2 |] Float.nan in
      let x, _ = source 1009 src in
      List.iter
        (fun (l, want) ->
          let q =
            Nx.concatenate ~axis:1
              (List.tl (List.concat_map (fun s -> [ nan; strip s ]) l))
          in
          let ((row, col, _, cols) as b) = Warp.area_window src q in
          equal (pair int int) want (col, cols);
          let k = Warp.span src q in
          equal (array int64)
            (bits (fst (Warp.average ~k src q x)))
            (bits (fst (Warp.average ~k ~row ~col src q (read_block src b x)))))
        [
          ([ (10, 11); (180, 11); (280, 6) ], (180, 201));
          ([ (0, 150); (10, 10); (180, 10); (220, 130) ], (180, 330));
        ])

(* The audit's sources (design notes, audit-probe): global grids at 0.1°,
   1° and 0.5°, from 180° W, from 0° and running west, grids at the
   dateline either way, a sheared one, projected ones straight and turned,
   360° grids whose turn is not a whole number of columns, and rotated
   geographic grids whose west corner is not their origin; each with random
   destinations at its west corner, its west or east edge or inside it,
   north-up or turned, in its system or in UTM, on a field with holes and a
   field of -0 with a lattice of -1, which shows a zero's sign. *)
let audit_windows =
  test "an area rule on its block gives the whole array's bits, at random"
    (fun () ->
      let geo ~rows ~cols ~x ~y ?(across = 1.) d =
        Grid.v Crs.lonlat ~rows ~cols ~origin:(x, y) ~row_step:(0., -.d)
          ~col_step:(across *. d, 0.)
      and rotated ~rows ~cols ~x ~y s t =
        Grid.v Crs.lonlat ~rows ~cols ~origin:(x, y)
          ~row_step:(s *. sin t, -.s *. cos t)
          ~col_step:(s *. cos t, s *. sin t)
      and west ~rows ~cols ~x ~y d =
        Grid.v Crs.lonlat ~rows ~cols ~origin:(x, y) ~row_step:(0., -.d)
          ~col_step:(-.d, 0.)
      and z33 t =
        Grid.v (utm 33 true) ~rows:200 ~cols:300 ~origin:(500000., 5300000.)
          ~row_step:(10. *. sin t, -10. *. cos t)
          ~col_step:(10. *. cos t, 10. *. sin t)
      in
      let global d =
        let cols = int_of_float (Float.round (360. /. d)) in
        Grid.v Crs.lonlat ~rows:40 ~cols ~origin:(-180., 2.)
          ~row_step:(0., -0.1) ~col_step:(360. /. float_of_int cols, 0.)
      in
      let sources =
        [
          geo ~rows:60 ~cols:3600 ~x:(-180.) ~y:3. 0.1;
          rotated ~rows:100 ~cols:100 ~x:179.6 ~y:10.5 0.01 0.2;
          geo ~rows:180 ~cols:360 ~x:(-180.) ~y:90. 1.;
          geo ~rows:180 ~cols:360 ~x:0. ~y:90. 1.;
          geo ~rows:360 ~cols:720 ~x:(-180.) ~y:90. 0.5;
          west ~rows:180 ~cols:360 ~x:180. ~y:90. 1.;
          geo ~rows:100 ~cols:100 ~x:179.5 ~y:10.5 0.01;
          west ~rows:100 ~cols:100 ~x:180.5 ~y:10.5 0.01;
          geo ~rows:200 ~cols:300 ~x:13.0 ~y:52.6 ~across:1.5 0.001;
          z33 0.;
          z33 0.3;
          rotated ~rows:80 ~cols:30 ~x:10. ~y:40. 1. (-0.05);
          rotated ~rows:120 ~cols:40 ~x:(-30.3) ~y:50. 0.5 (-0.1);
          rotated ~rows:200 ~cols:50 ~x:100.7 ~y:30. 0.25 (-0.02);
          rotated ~rows:60 ~cols:60 ~x:10. ~y:3. 0.1 0.3;
        ]
        @ List.map global [ 0.1; 0.2; 0.05; 360. /. 7. ]
      in
      let st = Random.State.make [| 7 |] in
      let destination (src : Grid.t) =
        let at r c =
          let p = Grid.world src (Nx.create Nx.float64 [| 2 |] [| r; c |]) in
          (Nx.item [ 0 ] p, Nx.item [ 1 ] p)
        in
        let rows = float_of_int src.rows and cols = float_of_int src.cols in
        let west_corner =
          List.fold_left
            (fun ((bx, _) as b) (r, c) ->
              let ((x, _) as p) = at r c in
              if x < bx then p else b)
            (at 0. 0.)
            [ (rows, 0.); (0., cols); (rows, cols) ]
        in
        let x, y =
          match Random.State.int st 4 with
          | 0 -> at (rows /. 2.) 0.
          | 1 -> at (rows /. 2.) cols
          | 2 -> west_corner
          | _ -> at (Random.State.float st rows) (Random.State.float st cols)
        in
        let (rx, ry), (cx, cy) = (src.row_step, src.col_step) in
        let res = Float.sqrt (Float.abs ((rx *. cy) -. (cx *. ry))) in
        let s = 0.3 +. Random.State.float st 5. in
        let t =
          if Random.State.bool st then 0. else Random.State.float st 6.283
        in
        let rows = 1 + Random.State.int st 6
        and cols = 1 + Random.State.int st 6 in
        let crs, x, y, d =
          if
            src.crs.projection = None
            && Random.State.int st 3 = 0
            && Float.abs y < 80.
          then
            let lon = Float.rem (x +. 540.) 360. -. 180. in
            let zone = 1 + (int_of_float ((lon +. 180.) /. 6.) mod 60) in
            let crs = Crs.utm ~zone ~north:(y >= 0.) in
            let p =
              Crs.transform ~src:Crs.lonlat ~dst:crs
                (Nx.create Nx.float64 [| 2 |] [| lon; y |])
            in
            ( crs,
              Nx.item [ 0 ] p,
              Nx.item [ 1 ] p,
              s *. res *. 111000. *. Float.cos (y *. Float.pi /. 180.) )
          else (src.crs, x, y, s *. res)
        in
        let dx = (Random.State.float st 1. -. 0.5) *. d *. float_of_int cols
        and dy = (Random.State.float st 1. -. 0.5) *. d *. float_of_int rows in
        Grid.v crs ~rows ~cols ~origin:(x -. dx, y +. dy)
          ~row_step:(d *. sin t, -.d *. cos t)
          ~col_step:(d *. cos t, d *. sin t)
      in
      let hash (g : Grid.t) =
        let x =
          Nx.init Nx.float64 [| g.rows; g.cols |] (fun a ->
              let h = (a.(0) * 7919) + (a.(1) * 104729) in
              float_of_int ((h * 2654435761) land 0xffff) /. 37.)
        in
        (x, Nx.greater_s x 300.)
      and zeros (g : Grid.t) =
        ( Nx.init Nx.float64 [| g.rows; g.cols |] (fun a ->
              if a.(0) mod 4 = 3 && a.(1) mod 4 = 3 then -1. else -0.),
          Nx.ones Nx.bool [| g.rows; g.cols |] )
      in
      let checked = ref 0 in
      List.iter
        (fun (src : Grid.t) ->
          for _ = 1 to 8 do
            let dst = destination src in
            (* A destination off a transform's domain, or of cells half a
               turn wide, has no corners. *)
            match corners ~src ~dst with
            | exception Invalid_argument _ -> ()
            | q ->
                incr checked;
                let ((row, col, _, _) as b) = Warp.area_window src q in
                let cut a = read_block src b a in
                let k = Warp.span src q in
                List.iter
                  (fun (x, valid) ->
                    let v, f = Warp.average ~k:(k + 1) ~valid src q x
                    and v', f' =
                      Warp.average ~k:(k + 1) ~row ~col ~valid:(cut valid) src
                        q (cut x)
                    in
                    equal (array int64) (bits v) (bits v');
                    equal (array int64) (bits f) (bits f');
                    let x = Nx.cast Nx.int32 (Nx.div_s x 400.) in
                    let m, _ = Warp.mode ~k ~valid src q x
                    and m', _ =
                      Warp.mode ~k ~row ~col ~valid:(cut valid) src q (cut x)
                    in
                    equal (array int64) (bits m) (bits m'))
                  [ hash src; zeros src ]
          done)
        sources;
      is_true ~msg:"most destinations checked" (!checked > 120))

(* Each edge of area_window's block is needed: a block one row or column
   short at any edge raises, so the block is the least. *)
let least_block =
  cases "a block short at any edge raises" ~id:(fun (n, _, _) -> n) area_cases
    (fun (_, (src : Grid.t), dst) ->
      let q = corners ~src ~dst and x, _ = source 1009 src in
      let k = Warp.span src q in
      let row, col, rows, cols = Warp.area_window src q in
      let short (row, col, rows, cols) =
        raises_match (Exn.invalid_arg ~sub:"outside the block") (fun () ->
            let x = read_block src (row, col, rows, cols) x in
            Nx.to_array (fst (Warp.average ~k ~row ~col src q x)))
      in
      if rows > 0 then begin
        short (row + 1, col, rows - 1, cols);
        short (row, col, rows - 1, cols);
        short (row, (col + 1) mod src.cols, rows, cols - 1);
        short (row, col, rows, cols - 1)
      end)

(* Reader 6's case: across 180° on a 0.1° global source the block is 41
   columns across the seam, where the hull of the corners' cells was the
   whole width; a short one raises compiled as well, and the compiled rule
   on the block gives the compiled whole's bits. *)
let seam_block =
  test "a block across a 360° source's west edge, eagerly and compiled"
    (fun () ->
      let src =
        north_up Crs.lonlat ~rows:40 ~cols:3600 ~x:(-180.) ~y:2. 0.1
      in
      let dst = north_up Crs.lonlat ~rows:2 ~cols:4 ~x:178.5 ~y:0.5 1. in
      let q = corners ~src ~dst and x, valid = source 1009 src in
      let k = Warp.span src q in
      let ((row, col, rows, cols) as b) = Warp.area_window src q in
      equal (quad int int int int) (15, 3585, 21, 41) b;
      let rule ~row ~col b q =
        let cut a = read_block src b a in
        fst (Warp.average ~k ~row ~col ~valid:(cut valid) src q (cut x))
      in
      equal (array int64)
        (bits (Rune.jit' (rule ~row:0 ~col:0 (0, 0, 40, 3600)) q))
        (bits (Rune.jit' (rule ~row ~col b) q));
      raises_match (Exn.invalid_arg ~sub:"outside the block") (fun () ->
          let b = (row, col, rows, cols - 1) in
          Nx.to_array (Rune.jit' (rule ~row ~col b) q)))

(* A geographic source that spans 360° is read modulo its columns, so a
   footprint across its west edge, the meridian of its east edge, reads both
   sides: every footprint near that seam gives the bits that the source
   rolled to put its edge at 0° gives, holes in [valid] included. *)
let wrapped_seam =
  test "a 360° source's footprints across its edge read both sides" (fun () ->
      let src = north_up Crs.lonlat ~rows:9 ~cols:360 ~x:(-180.) ~y:4.5 1. in
      let rolled = north_up Crs.lonlat ~rows:9 ~cols:360 ~x:0. ~y:4.5 1. in
      let x =
        Nx.init Nx.float64 [| 9; 360 |] (fun a ->
            float_of_int ((a.(0) * 1000) + a.(1)))
      in
      let valid =
        Nx.init Nx.bool [| 9; 360 |] (fun a -> (a.(0) + a.(1)) mod 5 <> 0)
      in
      let roll a =
        Nx.concatenate ~axis:1
          [ Nx.slice [ A; R (180, 360) ] a; Nx.slice [ A; R (0, 180) ] a ]
      in
      (* Cells of 1.25° from 175.25° to 186.5°, across the seam at 180°,
         their corners off the lattice. *)
      let dst =
        Grid.v Crs.lonlat ~rows:3 ~cols:9 ~origin:(175.25, 3.25)
          ~row_step:(0., -1.25) ~col_step:(1.25, 0.)
      in
      let bits a = Array.map Int64.bits_of_float (Nx.to_array a) in
      let v, f = Warp.average ~k:4 ~valid src (corners ~src ~dst) x
      and v', f' =
        Warp.average ~k:4 ~valid:(roll valid) rolled
          (corners ~src:rolled ~dst) (roll x)
      in
      equal ~msg:"values" (array int64) (bits v') (bits v);
      equal ~msg:"fractions" (array int64) (bits f') (bits f);
      let classes = Nx.cast Nx.int32 (Nx.div_s x 37.) in
      let m, g = Warp.mode ~k:4 ~valid src (corners ~src ~dst) classes
      and m', g' =
        Warp.mode ~k:4 ~valid:(roll valid) rolled (corners ~src:rolled ~dst)
          (roll classes)
      in
      equal ~msg:"classes" (array int32) (Nx.to_array m') (Nx.to_array m);
      equal ~msg:"mode's fractions" (array int64) (bits g') (bits g))

(* Steps whose turn is not their columns in floating point, each the double
   nearest 360° / cols, and columns that run west: a footprint across each
   source's west edge reads both sides, as on the source rolled to put its
   edge inside. *)
let wrapped_steps =
  cases "a 360° source is read modulo its columns to rounding" ~id:fst
    [
      ("0.2°", (0.2, 1800));
      ("0.1°", (0.1, 3600));
      ("0.05°", (0.05, 7200));
      ("360° / 161", (360. /. 161., 161));
      ("1° running west", (-1., 360));
    ]
    (fun (_, (s, cols)) ->
      let a = Float.abs s and h = cols / 2 in
      let edge = if s > 0. then -180. else 180. in
      let geo x =
        Grid.v Crs.lonlat ~rows:3 ~cols ~origin:(x, 1.5 *. a)
          ~row_step:(0., -.a) ~col_step:(s, 0.)
      in
      let src = geo edge and rolled = geo (edge +. (float_of_int h *. s)) in
      let x =
        Nx.init Nx.float64 [| 3; cols |] (fun i ->
            float_of_int ((i.(0) * 100000) + i.(1)))
      in
      let roll a =
        Nx.concatenate ~axis:1
          [ Nx.slice [ A; R (h, cols) ] a; Nx.slice [ A; R (0, h) ] a ]
      in
      let dst =
        Grid.v Crs.lonlat ~rows:1 ~cols:2
          ~origin:(-180. -. (1.1 *. a), 0.6 *. a)
          ~row_step:(0., -1.2 *. a) ~col_step:(1.5 *. a, 0.)
      in
      let v, f = Warp.average ~k:4 src (corners ~src ~dst) x
      and v', f' =
        Warp.average ~k:4 rolled (corners ~src:rolled ~dst) (roll x)
      in
      equal ~msg:"fraction" (array (float 1e-12)) [| 1.; 1. |] (Nx.to_array f);
      equal ~msg:"rolled's fraction" (array (float 1e-12)) (Nx.to_array f')
        (Nx.to_array f);
      equal ~msg:"values" (array (float 1e-6)) (floats v') (floats v))

(* A source whose column step has a latitude has a turn whose row is not
   0, so its columns are not read modulo their number, though they span
   360° of longitude. A footprint across its west edge read the east side
   alone, half its fraction, where the source holds the other side a turn
   east, so it raises. *)
let sloped_steps =
  test "a source whose columns climb raises across its west edge" (fun () ->
      let src =
        Grid.v Crs.lonlat ~rows:3 ~cols:360 ~origin:(-180., 1.5)
          ~row_step:(0., -1.) ~col_step:(1., 1e-6)
      in
      let x = Nx.ones Nx.float64 [| 3; 360 |] in
      let dst = north_up Crs.lonlat ~rows:1 ~cols:1 ~x:179.5 ~y:0.5 1. in
      raises_match (Exn.invalid_arg ~sub:"0.00036 rows, against its 360")
        (fun () -> Warp.average ~k:3 src (corners ~src ~dst) x))

(* Footprints around a pole, made by hand, wind once around the circle, so
   placed they have an edge of 180° of longitude or more, and the rule read a
   self-crossing quadrilateral: fraction 1 and a value that changed with the
   corners' order, or fraction 0. They raise on global sources of every
   orientation, eagerly and compiled, and so do cells of jord's own corners
   that hold a pole. On a source whose rows they miss they read nothing. *)
let pole_cells =
  test "a footprint around a pole raises where its rows meet the source"
    (fun () ->
      let lonlat ~rows ~cols ~x ~y ?(rx = 0.) ?(cy = 0.) cx ry =
        Grid.v Crs.lonlat ~rows ~cols ~origin:(x, y) ~row_step:(rx, ry)
          ~col_step:(cx, cy)
      in
      let global = lonlat ~rows:180 ~cols:360 ~x:(-180.) ~y:90. 1. (-1.) in
      let globals =
        [
          global;
          lonlat ~rows:180 ~cols:360 ~x:(-180.) ~y:(-90.) 1. 1.;
          lonlat ~rows:180 ~cols:360 ~x:180. ~y:90. (-1.) (-1.);
          lonlat ~rows:180 ~cols:360 ~x:0. ~y:90. 1. (-1.);
          lonlat ~rows:180 ~cols:360 ~x:(-180.) ~y:90. ~rx:0.001 ~cy:(-0.001)
            1. (-1.);
        ]
      and regional = lonlat ~rows:30 ~cols:50 ~x:(-10.) ~y:60. 1. (-1.) in
      (* Corners (λ, φ) in the rules' order (0, 0), (0, 1), (1, 1), (1, 0). *)
      let quad ((x0, y0), (x1, y1), (x2, y2), (x3, y3)) =
        Nx.create Nx.float64 [| 2; 2; 2 |]
          [| x0; y0; x1; y1; x3; y3; x2; y2 |]
      in
      let poles =
        [
          ((45., 89.5), (135., 89.5), (-135., 89.5), (-45., 89.5));
          ((10., 89.2), (100., 89.6), (190., 89.4), (280., 89.7));
          ((280., 89.7), (190., 89.4), (100., 89.6), (10., 89.2));
          ((-20., 88.), (60., 89.9), (170., 88.5), (-120., 89.0));
          ((45., -89.5), (-45., -89.5), (-135., -89.5), (135., -89.5));
          ((-170., 89.), (-80., 89.), (10., 89.), (100., 89.));
          (* An edge from 0° to 180°, through the pole. *)
          ((0., 89.), (180., 89.), (180., 88.), (0., 88.));
        ]
      and valid =
        [
          ((0., 89.), (179.9, 89.), (179.9, 88.), (0., 88.));
          ((0., 90.), (45., 89.), (0., 88.), (-45., 89.));
          ((-45., 90.), (45., 90.), (45., 89.5), (-45., 89.5));
          ((10., 89.5), (100., 89.5), (100., 89.), (10., 89.));
          ((10., 46.), (11., 46.), (11., 45.), (10., 45.));
        ]
      in
      let winds = Exn.invalid_arg ~sub:"footprint [0, 0] has an edge" in
      List.iter
        (fun (src : Grid.t) ->
          let x = Nx.ones Nx.float64 [| src.rows; src.cols |] in
          List.iter
            (fun c ->
              let q = Warp.pixel src (quad c) in
              raises_match winds (fun () -> Warp.span src q);
              raises_match winds (fun () -> Warp.average ~k:400 src q x);
              raises_match winds (fun () ->
                  Warp.mode ~k:400 src q (Nx.cast Nx.int32 x)))
            poles;
          List.iter
            (fun c ->
              let q = Warp.pixel src (quad c) in
              let _, f = Warp.average ~k:(Warp.span src q) src q x in
              greater ~msg:"fraction" float_exact ~than:0.5
                (Nx.item [ 0; 0 ] f))
            valid)
        globals;
      let q = Warp.pixel global (quad (List.hd poles)) in
      let x = Nx.ones Nx.float64 [| 180; 360 |] in
      let rule q = fst (Warp.average ~k:400 global q x) in
      raises_match winds (fun () -> Nx.to_array (Rune.jit' rule q));
      let x = Nx.ones Nx.float64 [| 30; 50 |] in
      List.iter
        (fun c ->
          let q = Warp.pixel regional (quad c) in
          equal ~msg:"no data" (array float_exact) [| 0. |]
            (Nx.to_array (snd (Warp.average ~k:1 regional q x))))
        poles;
      (* Albers cells of 6 arcs about the apex, and a grid of 21 × 21 cells
         of 2 arcs whose middle cell holds it. *)
      let ux, uy, rho = apex polar_albers 90. in
      List.iter
        (fun (sub, rows, d) ->
          let h = float_of_int rows *. d /. 2. in
          let dst =
            north_up polar_albers ~rows ~cols:rows ~x:(ux -. h) ~y:(uy +. h) d
          in
          raises_match (Exn.invalid_arg ~sub) (fun () ->
              Warp.span global (corners ~src:global ~dst)))
        [
          ("footprint [0, 0] has an edge", 1, 6. *. rho);
          ("footprint [10, 10] has an edge", 21, 2. *. rho);
        ])

let lonlat ~rows ~cols ~x ~y ?(rx = 0.) ?(cy = 0.) cx ry =
  Grid.v Crs.lonlat ~rows ~cols ~origin:(x, y) ~row_step:(rx, ry)
    ~col_step:(cx, cy)

(* A footprint across the west edge of a source that does not span 360° is
   placed east of the edge and reads that side alone. Where the source holds
   the other side a turn east, as 361 or 3601 gridline columns do, or a step
   printed short of 360° / cols, its fraction was a half or a quarter, so it
   raises. A footprint at 0° or inside column 0 reads all it covers, and a
   source that spans 360° or holds the other side nowhere reads as before.
   Each footprint is 2 steps square, on a field of 1. *)
let seam_cases =
  (* The remedy each message gives: a cut or a pad where the turn is whole
     columns, else a step restated or a cut that holds the footprint. *)
  let cut n =
    Printf.sprintf
      "Grid.window it to its first turn, %d columns, and slice x to them" n
  and other = "no whole number of its columns makes a turn: where its column"
  in
  let holds =
    [
      ( "361 × 1° from -180.5°",
        lonlat ~rows:20 ~cols:361 ~x:(-180.5) ~y:10. 1. (-1.),
        cut 360 );
      ( "3601 × 0.1°",
        lonlat ~rows:20 ~cols:3601 ~x:(-180.05) ~y:1. 0.1 (-0.1),
        cut 3600 );
      ( "520 × 0.7°",
        lonlat ~rows:20 ~cols:520 ~x:(-180.) ~y:10. 0.7 (-0.7),
        other );
      ( "1201 × 0.3°",
        lonlat ~rows:20 ~cols:1201 ~x:(-180.) ~y:3. 0.3 (-0.3),
        cut 1200 );
      ( "43200 × 0.0083333333°",
        lonlat ~rows:4 ~cols:43200 ~x:(-180.) ~y:0.02 0.0083333333
          (-0.0083333333),
        other );
      ( "43200 × 0.0083333334°",
        lonlat ~rows:4 ~cols:43200 ~x:(-180.) ~y:0.02 0.0083333334
          (-0.0083333334),
        other );
      ( "361 running west",
        lonlat ~rows:20 ~cols:361 ~x:180.5 ~y:10. (-1.) (-1.),
        cut 360 );
      ( "361 south-up",
        lonlat ~rows:20 ~cols:361 ~x:(-180.5) ~y:(-10.) 1. 1.,
        cut 360 );
      ( "361 rotated by 1e-4",
        lonlat ~rows:20 ~cols:361 ~x:(-180.5) ~y:10. ~rx:1e-4 ~cy:(-1e-4) 1.
          (-1.),
        other );
      ( "359 × 1°, across by more than a column",
        lonlat ~rows:20 ~cols:359 ~x:(-180.) ~y:10. 1. (-1.),
        "pad it to one turn, 360 columns" );
    ]
  and reads =
    [
      ("360 × 1°", lonlat ~rows:20 ~cols:360 ~x:(-180.) ~y:10. 1. (-1.), 1.);
      ( "43200 × 1/120°",
        lonlat ~rows:4 ~cols:43200 ~x:(-180.) ~y:0.02 (1. /. 120.)
          (-1. /. 120.),
        1. );
      ("359 × 1°", lonlat ~rows:20 ~cols:359 ~x:(-180.) ~y:10. 1. (-1.), 0.5);
      ("170° to 190°", lonlat ~rows:20 ~cols:20 ~x:170. ~y:10. 1. (-1.), 0.5);
      ( "150° W to 150° E",
        lonlat ~rows:20 ~cols:300 ~x:(-150.) ~y:10. 1. (-1.),
        0.5 );
    ]
  in
  (* The 2-step cell from [lon], two steps inside the source's first rows. *)
  let fraction (src : Grid.t) lon =
    let d = Float.abs (fst src.col_step) and ry = snd src.row_step in
    let dst =
      lonlat ~rows:1 ~cols:1 ~x:lon ~y:(snd src.origin +. (2. *. ry))
        (2. *. d) (Float.abs ry *. Float.of_int (compare ry 0.))
    in
    let q = corners ~src ~dst
    and x = Nx.ones Nx.float64 [| src.rows; src.cols |] in
    Nx.item [ 0; 0 ] (snd (Warp.average ~k:(Warp.span src q) src q x))
  in
  let west (src : Grid.t) =
    let w, _ = src.origin and cx, _ = src.col_step in
    Float.min w (w +. (float_of_int src.cols *. cx))
  in
  cases "a footprint across a source's west edge raises where it holds the \
         other side"
    ~id:fst
    (List.map (fun (n, s, r) -> (n, `Holds (s, r))) holds
    @ List.map (fun (n, s, f) -> (n, `Reads (s, f))) reads)
    (fun (_, c) ->
      match c with
      | `Holds (src, remedy) ->
          let d = Float.abs (fst src.col_step) in
          List.iter
            (fun lon ->
              raises_match (Exn.invalid_arg ~sub:remedy) (fun () ->
                  fraction src lon))
            [ west src -. (1.5 *. d); 180. -. (1.5 *. d) ];
          List.iter
            (fun lon ->
              equal ~msg:"fraction" (float 1e-12) 1. (fraction src lon))
            [ 0.; west src +. (0.1 *. d) ]
      | `Reads (src, f) ->
          let d = Float.abs (fst src.col_step) in
          equal ~msg:"across the west edge" (float 1e-12) f
            (fraction src (west src -. d)))

(* The message names the turn, so that a printed step shows as its cause;
   a 361-column source's footprint across 180° raises in every rule. *)
let seam_message =
  test "the seam's refusal names the source's turn" (fun () ->
      let src = lonlat ~rows:21 ~cols:361 ~x:(-180.5) ~y:10.5 1. (-1.) in
      let dst = north_up Crs.lonlat ~rows:2 ~cols:4 ~x:178. ~y:5. 1. in
      let q = corners ~src ~dst and x = Nx.ones Nx.float64 [| 21; 361 |] in
      let turn =
        Exn.invalid_arg
          ~sub:"its turn is 360 columns and 0 rows, against its 361 columns"
      in
      raises_match turn (fun () -> Warp.span src q);
      raises_match turn (fun () -> Warp.area_window src q);
      raises_match turn (fun () -> Warp.average ~k:20 src q x);
      raises_match turn (fun () -> Warp.mode ~k:20 src q (Nx.cast Nx.int32 x));
      let rule q = fst (Warp.average ~k:20 src q x) in
      raises_match turn (fun () -> Nx.to_array (Rune.jit' rule q));
      (* Off its rows, the footprint reads nothing. *)
      let dst = north_up Crs.lonlat ~rows:2 ~cols:4 ~x:178. ~y:50. 1. in
      let q = corners ~src ~dst in
      equal ~msg:"no data" (array float_exact) (Array.make 8 0.)
        (Nx.to_array (snd (Warp.average ~k:(Warp.span src q) src q x)));
      let src =
        lonlat ~rows:4 ~cols:43200 ~x:(-180.) ~y:0.02 0.0083333333
          (-0.0083333333)
      in
      let dst = north_up Crs.lonlat ~rows:1 ~cols:1 ~x:179.99 ~y:0.01 0.02 in
      raises_match (Exn.invalid_arg ~sub:"its turn is 43200.00017 columns")
        (fun () -> Warp.span src (corners ~src ~dst)))

(* Each refusal of a block, a k or a seam ends in the fix that fits it. A
   block across a 360° source's west edge read as one part, to the last
   column, is told apart from a block too short, and a source whose step is
   printed off 360° / cols is told to state it so, where no whole number of
   cells cuts or pads it to a turn. *)
let remedies =
  test "the refusals name their remedies" (fun () ->
      let src = north_up Crs.lonlat ~rows:20 ~cols:360 ~x:(-180.) ~y:10. 1. in
      let dst = north_up Crs.lonlat ~rows:1 ~cols:1 ~x:179. ~y:5. 2. in
      let q = corners ~src ~dst and x = Nx.ones Nx.float64 [| 20; 360 |] in
      let k = Warp.span src q in
      let ((row, col, rows, cols) as b) = Warp.area_window src q in
      equal (quad int int int int) (5, 359, 3, 3) b;
      let average ?(k = k) (row, col, rows, cols) =
        let x = read_block src (row, col, rows, cols) x in
        Nx.to_array (fst (Warp.average ~k ~row ~col src q x))
      in
      raises_match (Exn.invalid_arg ~sub:"more than k = 2; use Warp.span")
        (fun () -> average ~k:(k - 1) b);
      raises_match
        (Exn.invalid_arg
           ~sub:"reads cells outside the block; read the block Warp.area_window"
        )
        (fun () -> average (row, col, rows - 1, cols));
      raises_match
        (Exn.invalid_arg
           ~sub:
             "across the source's west edge past the block's last column; \
              read the block Warp.area_window gives in two parts")
        (fun () ->
          let x = Nx.slice [ R (row, row + rows); R (col, 360) ] x in
          Nx.to_array (fst (Warp.average ~k ~row ~col src q x)));
      let p = points ~src ~dst in
      let row, col, rows, _ = Warp.window Bilinear src p in
      raises_match
        (Exn.invalid_arg
           ~sub:
             "outside the block; read the block Warp.window Bilinear gives, \
              in two parts")
        (fun () ->
          let x = Nx.slice [ R (row, row + rows); R (col, 360) ] x in
          Nx.to_array (fst (Warp.bilinear ~row ~col src p x)));
      let g = north_up (utm 33 true) ~rows:6 ~cols:7 ~x:0. ~y:0. 1. in
      let x = Nx.ones Nx.float64 [| 3; 7 |] and p = pixels [ (2.5, 2.5) ] in
      raises_match
        (Exn.invalid_arg ~sub:"read the block Warp.window Nearest gives")
        (fun () -> Nx.to_array (fst (Warp.nearest ~row:3 g p x)));
      (* A step 1e-7 off 1°: no whole number of cells makes a turn, and a
         restated step is one remedy. *)
      let dst = north_up Crs.lonlat ~rows:1 ~cols:1 ~x:(-181.) ~y:5. 2. in
      List.iter
        (fun d ->
          let src =
            north_up Crs.lonlat ~rows:20 ~cols:360 ~x:(-180.) ~y:10. d
          in
          raises_match
            (Exn.invalid_arg
               ~sub:
                 "against its 360 columns; no whole number of its columns \
                  makes a turn: where its column step rounds (±360° / n, 0)")
            (fun () -> Warp.span src (corners ~src ~dst)))
        [ 0.9999999; 1.0000001 ])

(* Each remedy the seam's refusal gives reads the footprint it refused: a
   cut or a pad where the turn is whole columns, a step restated where it
   is a rounding, and, where the step is the source's own, a cut narrower
   than a turn that holds the footprint whole, or cells off the edge. The
   first 514 columns of 520 of 0.7° keep the edge, and raise. A 0.7° step
   restated as 360° / 514 moves the last cells by 0.2°, so it is no
   remedy. *)
let seam_remedies =
  test "each remedy of the seam's refusal reads the footprint" (fun () ->
      let fraction ?valid (src : Grid.t) ~lon ~w =
        let dst = north_up Crs.lonlat ~rows:1 ~cols:1 ~x:lon ~y:5. w in
        let q = corners ~src ~dst
        and x = Nx.ones Nx.float64 [| src.rows; src.cols |] in
        Nx.item [ 0; 0 ]
          (snd (Warp.average ~k:(Warp.span src q) ?valid src q x))
      in
      let cut col cols (g : Grid.t) =
        Grid.window ~row:0 ~col ~rows:g.rows ~cols g
      in
      let refused sub src ~lon ~w =
        raises_match (Exn.invalid_arg ~sub) (fun () -> fraction src ~lon ~w)
      in
      let gridline = lonlat ~rows:20 ~cols:361 ~x:(-180.5) ~y:10. 1. (-1.) in
      refused "Grid.window it" gridline ~lon:(-181.) ~w:1.;
      equal ~msg:"cut" (float 1e-12) 1.
        (fraction (cut 0 360 gridline) ~lon:(-181.) ~w:1.);
      let short = lonlat ~rows:20 ~cols:359 ~x:(-180.) ~y:10. 1. (-1.) in
      refused "pad it" short ~lon:(-181.5) ~w:2.;
      let whole = lonlat ~rows:20 ~cols:360 ~x:(-180.) ~y:10. 1. (-1.) in
      let valid = Nx.init Nx.bool [| 20; 360 |] (fun a -> a.(1) < 359) in
      equal ~msg:"pad" (float 1e-12) 0.5
        (fraction ~valid whole ~lon:(-181.5) ~w:2.);
      let other = "no whole number of its columns makes a turn" in
      let printed =
        lonlat ~rows:20 ~cols:360 ~x:(-180.) ~y:10. 0.9999999 (-1.)
      in
      refused other printed ~lon:(-180.5) ~w:1.;
      equal ~msg:"restated" (float 1e-12) 1.
        (fraction whole ~lon:(-180.5) ~w:1.);
      let own = lonlat ~rows:20 ~cols:520 ~x:(-180.) ~y:10. 0.7 (-0.7) in
      refused other own ~lon:(-180.5) ~w:1.;
      refused other (cut 0 514 own) ~lon:(-180.5) ~w:1.;
      equal ~msg:"a cut that holds it" (float 1e-9) 1.
        (fraction (cut 6 514 own) ~lon:(-180.5) ~w:1.);
      List.iter
        (fun lon ->
          equal ~msg:"off the edge" (float 1e-9) 1. (fraction own ~lon ~w:1.))
        [ -181.; -180. ];
      let narrow = lonlat ~rows:20 ~cols:514 ~x:(-180.) ~y:10. 0.7 (-0.7) in
      refused other narrow ~lon:(-180.5) ~w:1.;
      equal ~msg:"off the edge, short of a turn" (float 1e-9) 0.8
        (fraction narrow ~lon:(-181.) ~w:1.))

let area_refusals =
  test "area rules refuse what they cannot read" (fun () ->
      let x = Nx.zeros Nx.float64 [| 3; 3 |]
      and g = unit_grid (Nx.zeros Nx.float64 [| 3; 3 |]) in
      let q = quads ~rows:1 ~cols:1 (0., 0.) (1., 0.) (0., 1.) in
      raises_match (Exn.invalid_arg ~sub:"corners") (fun () ->
          Warp.average ~k:3 g (Nx.zeros Nx.float64 [| 2; 2 |]) x);
      (* A block that is not in the grid. *)
      List.iter
        (fun (row, col, rows, cols) ->
          raises_match (Exn.invalid_arg ~sub:"not in the grid") (fun () ->
              Warp.average ~k:3 ~row ~col g q
                (Nx.zeros Nx.float64 [| rows; cols |])))
        [ (-1, 0, 2, 2); (0, -1, 2, 2); (2, 0, 2, 3); (0, 1, 3, 3) ];
      let world = north_up Crs.lonlat ~rows:2 ~cols:4 ~x:(-180.) ~y:1. 90. in
      raises_match (Exn.invalid_arg ~sub:"not in the grid") (fun () ->
          Warp.average ~k:3 world q (Nx.zeros Nx.float64 [| 2; 5 |]));
      raises_match (Exn.invalid_arg ~sub:"not in the grid") (fun () ->
          Warp.average ~k:3 ~col:4 world q (Nx.zeros Nx.float64 [| 2; 1 |]));
      raises_match (Exn.invalid_arg ~sub:"real") (fun () ->
          Warp.average ~k:3 g q (Nx.zeros Nx.bool [| 3; 3 |]));
      raises_match (Exn.invalid_arg ~sub:"integer") (fun () ->
          Warp.mode ~k:3 g q x))

let () =
  exit
    (run "jord.warp"
       [
         group "gdal" [ gdal_parity ];
         group "lattice" [ lattice; rounding; float32_once ];
         group "samplers"
           [
             masked_never_matter;
             nearest_reads;
             same_mask;
             mean_of_neighbours;
             shifted;
             edges;
             nan_data;
             borders;
             dtypes;
             shapes;
             refusals;
           ];
         group "gradient" [ centre_gradient; nan_neighbours; descent ];
         group "systems"
           [
             equal_systems;
             corners_law;
             half_turn;
             projected_half_turn;
             own_system;
             no_source;
             own_domain;
             across_datums;
           ];
         group "antimeridian" [ any_360; in_range ];
         group "windows"
           [
             window_cases;
             windowed;
             windowed_gradients;
             short_blocks;
             seam_window;
           ];
         group "area gdal"
           [ gdal_average; gdal_rounded; gdal_mode; exact_xzone ];
         group "area laws"
           [
             own_grid;
             own_turned_grid;
             constant;
             conservation;
             blocks;
             unit_squares;
             bilinear_seam;
             inside_one_cell;
             mode_by_area;
             partial;
             nan_unreached;
             mirrored;
             nan_corner;
             tiny_footprints;
             long_outside;
             rounding_stays_out;
           ];
         group "area gradient" [ area_gradient; right_hand ];
         group "area antimeridian"
           [
             lonlat_area;
             seam;
             closed_east;
             westward;
             wrapped_seam;
             wrapped_steps;
             sloped_steps;
           ];
         group "area refusals"
           [
             area_refusals;
             too_small;
             pole_cells;
             seam_cases;
             seam_message;
             remedies;
             seam_remedies;
           ];
         group "area span" [ span_least; span_cases ];
         group "masks" [ mask_shapes; example_join ];
         group "area windows"
           [
             windowed_area; audit_windows; block_gaps; least_block; seam_block;
           ];
         group "area compiled" [ compiled ];
       ])
