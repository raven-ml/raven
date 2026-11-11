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

let pixels l =
  Nx.create Nx.float64
    [| List.length l; 2 |]
    (Array.of_list (List.concat_map (fun (a, b) -> [ a; b ]) l))

let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let read (v, ok) = (floats v, Nx.to_array ok)
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

(* warp_gdal.dat: for each case, a line [case name], the source's and the
   destination's grid as [src|dst epsg rows cols ox oy rx ry cx cy], then each
   output's name on a line and its rows, NaN where GDAL wrote no data. *)
let fixtures =
  let lines =
    In_channel.with_open_text "warp_gdal.dat" In_channel.input_lines
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
    | _ -> failwith ("warp_gdal.dat: not a grid: " ^ l)
  in
  let rec outputs (dst : Grid.t) acc = function
    | name :: rest when not (String.contains name ' ') ->
        let rows = List.filteri (fun i _ -> i < dst.rows) rest in
        let rest = List.filteri (fun i _ -> i >= dst.rows) rest in
        let v = Array.of_list (List.concat_map words rows) in
        outputs dst ((name, Array.map float_of_string v) :: acc) rest
    | rest -> (List.rev acc, rest)
  in
  let rec cases acc = function
    | [] -> List.rev acc
    | c :: s :: d :: rest when String.starts_with ~prefix:"case " c ->
        let src = grid s and dst = grid d in
        let outputs, rest = outputs dst [] rest in
        let name = String.sub c 5 (String.length c - 5) in
        cases ({ name; src; dst; outputs } :: acc) rest
    | l :: _ -> failwith ("warp_gdal.dat: unexpected line: " ^ l)
  in
  cases [] lines

let fixture name = List.find (fun f -> f.name = name) fixtures

(* jord's result for each of a fixture's outputs, NaN where the mask is
   false: nearest on [x], bilinear on [x] as float64, and [rounded],
   bilinear on [x] in its own dtype. *)
let jord ~valid p x =
  let nan_off (v, ok) =
    let v, ok = read (v, ok) in
    Array.mapi (fun k v -> if ok.(k) then v else Float.nan) v
  in
  [
    ("nearest", nan_off (Warp.nearest ~valid p x));
    ("bilinear", nan_off (Warp.bilinear ~valid p (Nx.cast Nx.float64 x)));
    ("rounded", nan_off (Warp.bilinear ~valid p x));
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
        | `Float -> jord ~valid p x
        | `Uint16 -> jord ~valid p (Nx.cast Nx.uint16 x)
        | `Int16 -> jord ~valid p (Nx.cast Nx.int16 (Nx.sub_s x 6000.))
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
      let mean, ok = read (Warp.bilinear ~valid p x) in
      let rounded, _ = read (Warp.bilinear ~valid p (Nx.cast Nx.int16 x)) in
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
      let v, _ = read (Warp.bilinear (pixels [ (0.5, c) ]) x) in
      equal float_exact expected v.(0))

let float32_once =
  test "a float32 mean rounds once, from float64" (fun () ->
      let x = Nx.create Nx.float32 [| 1; 2 |] [| 100.; 101. |] in
      let v, _ = read (Warp.bilinear (pixels [ (0.5, 1. -. 2e-6) ]) x) in
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
        (read (Warp.nearest ~valid p nan))
        (read (Warp.nearest ~valid p zero));
      equal ~msg:"bilinear" result
        (read (Warp.bilinear ~valid p nan))
        (read (Warp.bilinear ~valid p zero)))

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
        (read (Warp.nearest ~valid p x)))

let same_mask =
  prop "bilinear's mask is nearest's" field (fun (x, valid, p) ->
      equal (array bool)
        (snd (read (Warp.nearest ~valid p x)))
        (snd (read (Warp.bilinear ~valid p x))))

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
      let v = floats (fst (Warp.bilinear ~valid p x)) in
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
      let mask = Array.init (rows * cols) (fun k -> has (cell k)) in
      let band (i, j) b = if bands = 0 then [ i; j ] else [ i; j; b ] in
      let values =
        Array.init (rows * cols * per) (fun kb ->
            let c = cell (kb / per) in
            if has c then Nx.item (band c (kb mod per)) x else 0.)
      in
      equal ~msg:"nearest" result (values, mask)
        (read (Warp.nearest ~valid p x));
      equal ~msg:"bilinear" result (values, mask)
        (read (Warp.bilinear ~valid p x)))

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
        (read (Warp.bilinear ~valid p x));
      (* Next to the hole, the three valid cells' weights are renormalised. *)
      let v, _ = read (Warp.bilinear ~valid (pixels [ (0.75, 0.75) ]) x) in
      equal (array float_exact) [| (9. +. 9. +. 15.) /. 15. |] v)

let nan_data =
  test "a valid NaN is data where its weight is not 0" (fun () ->
      let x = Nx.create Nx.float64 [| 1; 2 |] [| 2.; Float.nan |] in
      let p = pixels [ (0.5, 0.5); (0.5, 0.75); (0.5, 1.5) ] in
      equal result
        ([| 2.; Float.nan; Float.nan |], [| true; true; true |])
        (read (Warp.bilinear p x)))

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
        (read (Warp.nearest p x)))

let dtypes =
  test "nearest keeps the dtype, bilinear rounds once to it" (fun () ->
      let p = pixels [ (0.5, 0.75); (0.6, 1.1) ] in
      let x8 = Nx.create Nx.uint8 [| 1; 2 |] [| 7; 255 |] in
      equal (array int) [| 7; 255 |] (Nx.to_array (fst (Warp.nearest p x8)));
      let b = Nx.create Nx.bool [| 1; 2 |] [| false; true |] in
      equal (array bool) [| false; true |]
        (Nx.to_array (fst (Warp.nearest p b)));
      let x32 = Nx.create Nx.float32 [| 1; 2 |] [| 0.1; 0.7 |] in
      let x64 = Nx.cast Nx.float64 x32 in
      equal (array float_exact)
        (floats (Nx.cast Nx.float32 (fst (Warp.bilinear p x64))))
        (floats (fst (Warp.bilinear p x32))))

let shapes =
  test "results have the points' batch and the array's other axes" (fun () ->
      let x = Nx.zeros Nx.float32 [| 3; 4; 2; 5 |] in
      let v, ok = Warp.bilinear (Nx.full Nx.float64 [| 6; 1; 2 |] 1.5) x in
      equal (array int) [| 6; 1; 2; 5 |] (Nx.shape v);
      equal (array int) [| 6; 1 |] (Nx.shape ok);
      let v, ok = Warp.nearest (Nx.zeros Nx.float64 [| 0; 2 |]) x in
      equal (array int) [| 0; 2; 5 |] (Nx.shape v);
      equal (array int) [| 0 |] (Nx.shape ok);
      let empty = Nx.zeros Nx.float64 [| 0; 3 |] in
      equal result
        ([| 0. |], [| false |])
        (read (Warp.bilinear (pixels [ (0., 0.) ]) empty)))

let refusals =
  test "samplers refuse points, arrays and masks of other shapes" (fun () ->
      let x = Nx.zeros Nx.float64 [| 3; 4 |] and p = pixels [ (0., 0.) ] in
      raises_match (Exn.invalid_arg ~sub:"last axis") (fun () ->
          Warp.nearest (Nx.zeros Nx.float64 [| 2; 3 |]) x);
      raises_match (Exn.invalid_arg ~sub:"rank") (fun () ->
          Warp.bilinear p (Nx.zeros Nx.float64 [| 3 |]));
      raises_match (Exn.invalid_arg ~sub:"mask") (fun () ->
          Warp.nearest ~valid:(Nx.ones Nx.bool [| 4; 3 |]) p x);
      raises_match (Exn.invalid_arg ~sub:"not a real dtype") (fun () ->
          Warp.bilinear p (Nx.zeros Nx.bool [| 3; 4 |])))

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
    fst (Warp.bilinear (Nx.add centres (Nx.reshape [| 1; 1; 2 |] s)) x)
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
    Nx.sum (Nx.slice [ A; I 0 ] (fst (Warp.bilinear ~valid p x)))

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

let no_source =
  test "a centre outside the source's domain has no data" (fun () ->
      let c = Warp.centres ~at:host (utm 60 true) utm33 in
      equal (array bool) (Array.make 40 true) (nan_points c);
      let src = north_up (utm 60 true) ~rows:2 ~cols:2 ~x:5e5 ~y:5.3e6 10. in
      let x = Nx.zeros Nx.int32 [| 2; 2 |] in
      let _, ok = Warp.nearest (Warp.pixel src c) x in
      equal (array bool) (Array.make 20 false) (Nx.to_array ok);
      (* A Lonlat grid whose first row lies past the pole. *)
      let polar = north_up Crs.lonlat ~rows:3 ~cols:1 ~x:15. ~y:90.5 0.5 in
      equal (array bool)
        [| true; true; false; false; false; false |]
        (nan_points (Warp.centres ~at:host (utm 33 true) polar)))

let own_domain =
  test "a centre outside the destination's own domain raises" (fun () ->
      raises_match (Exn.invalid_arg ~sub:"domain") (fun () ->
          Warp.centres ~at:host Crs.lonlat far))

(* The antimeridian *)

(* The antimeridian fixture's source grid, its west edge at 179.998 or at
   -180.002. *)
let lonlat_at x = north_up Crs.lonlat ~rows:30 ~cols:40 ~x ~y:(-16.998) 0.0001

let any_360 =
  test "a Lonlat source reads the same cells in either 360°" (fun () ->
      let f = fixture "antimeridian" in
      (* Each cell's value is its column; column 20 is the antimeridian's. *)
      let x = Nx.init Nx.float64 [| 30; 40 |] (fun a -> float_of_int a.(1)) in
      let at src = read (Warp.nearest (points ~src ~dst:f.dst) x) in
      let cols, ok = at (lonlat_at 179.998) in
      equal result (cols, ok) (at (lonlat_at (-180.002)));
      let cols = List.filteri (fun k _ -> ok.(k)) (Array.to_list cols) in
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
      let rows = Nx.dim 0 x and cols = Nx.dim 1 x in
      let g = north_up (utm 33 true) ~rows ~cols ~x:0. ~y:0. 1. in
      let check rule sample =
        let row, col, rows, cols = Warp.window rule g p in
        let block a = Nx.slice [ R (row, row + rows); R (col, col + cols) ] a in
        let corner = [| float_of_int row; float_of_int col |] in
        let q = Nx.sub p (Nx.create Nx.float64 [| 2 |] corner) in
        equal result
          (read (sample ?valid:(Some valid) p x))
          (read (sample ?valid:(Some (block valid)) q (block x)))
      in
      check Nearest Warp.nearest;
      check Bilinear Warp.bilinear)

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
         group "systems" [ equal_systems; no_source; own_domain ];
         group "antimeridian" [ any_360; in_range ];
         group "windows" [ window_cases; windowed ];
       ])
