(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Pixel grids: centres and corners where the convention puts them, windows that
   keep their base's cells bit for bit, measures that are each cell's solid
   angle or area, and windows placed around points. *)

open Windtrap
open Ymir
open Ymir_test.Frames

let f64 = Nx.float64
let tensor shape xs = Nx.create f64 shape xs
let one x = Quantity.v Unit.one x
let deg x = Quantity.v Unit.degree x

let starts xs =
  Nx.create Nx.int64 [| Array.length xs / 2; 2 |] (Array.map Int64.of_int xs)

let start i j = Nx.create Nx.int64 [| 2 |] [| Int64.of_int i; Int64.of_int j |]
let plane shape = Grid.pixels ~shape f64 Transform.id
let values u q = Nx.to_array (Quantity.value u q)

(* A TAN image whose pixels are [scale] degrees on a side, axes along north and
   west, CRVAL [(lon, lat)] at pixel [crpix]. *)
let tan ?(crpix = [| 0.; 0. |]) ~scale (lon, lat) =
  let cd = deg (tensor [| 2; 2 |] [| -.scale; 0.; 0.; scale |]) in
  Transform.(
    axes [| 1; 0 |] ~origin:1
    >> shift (one (tensor [| 2 |] (Array.map (fun x -> x +. 1.) crpix)))
    >> linear cd
    >> celestial Tan Frame.icrs ~pv:(Nx.zeros f64 [| 0 |])
         ~native:(deg (tensor [| 2 |] [| 0.; 90. |]))
         ~crval:(deg (tensor [| 2 |] [| lon; lat |]))
         ~lonpole:(deg (scalar 180.))
         ~latpole:(deg (scalar 90.)))

(* The solid angle of the tangent-plane rectangle [x1, x2] × [y1, y2] in
   radians: [∫∫ dx dy / (1 + x² + y²)^(3/2)]. *)
let rectangle x1 x2 y1 y2 =
  let f x y = Float.atan (x *. y /. Float.sqrt (1. +. (x *. x) +. (y *. y))) in
  f x2 y2 -. f x1 y2 -. f x2 y1 +. f x1 y1

let cells =
  group "Cells"
    [
      test "centres are the cells' indices" (fun () ->
          let g = plane [| 2; 3 |] in
          equal (array float_exact)
            [| 0.; 0.; 0.; 1.; 0.; 2.; 1.; 0.; 1.; 1.; 1.; 2. |]
            (values Unit.one (Grid.centres g)));
      test "corners are half a cell away, counter-clockwise" (fun () ->
          let g =
            Grid.window ~start:(start 4 7) ~shape:[| 1; 1 |]
              (plane [| 10; 10 |])
          in
          equal (array float_exact)
            [| 3.5; 6.5; 4.5; 6.5; 4.5; 7.5; 3.5; 7.5 |]
            (values Unit.one (Grid.corners g)));
      test "shapes" (fun () ->
          let g =
            Grid.window
              ~start:(starts [| 0; 0; 3; 4; 5; 6 |])
              ~shape:[| 4; 2 |]
              (plane [| 10; 10 |])
          in
          equal (array int) [| 4; 2 |] (Grid.shape g);
          equal (array int) [| 3; 4; 2; 2 |]
            (Nx.shape (Quantity.value Unit.one (Grid.centres g)));
          equal (array int) [| 3; 4; 2; 4; 2 |]
            (Nx.shape (Quantity.value Unit.one (Grid.corners g)));
          equal (array int) [| 3; 4; 2 |]
            (Nx.shape (Quantity.value Unit.one (Grid.measure g))));
      test "pixels refuses a shape of another rank" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Grid.pixels") (fun () ->
              Grid.pixels ~shape:[| 3 |] f64 Transform.id));
      prop "a window's centres are its base's at the same cells, bit for bit"
        Gen.(pair (int_range (-20) 120) (int_range (-20) 120))
        (fun (i, j) ->
          let g =
            Grid.pixels ~shape:[| 100; 100 |] f64
              (tan ~crpix:[| 50.; 50. |] ~scale:1e-3 (30., -20.))
          in
          let w = Grid.window ~start:(start i j) ~shape:[| 3; 4 |] g in
          let big =
            Grid.window ~start:(start (-20) (-20)) ~shape:[| 150; 150 |] g
          in
          let x = Direction.xyz (Grid.centres w)
          and y = Direction.xyz (Grid.centres big) in
          cover "beyond the base" (i < 0 || j < 0 || i > 96 || j > 95);
          let expected =
            Nx.slice [ Nx.R (i + 20, i + 23); Nx.R (j + 20, j + 24) ] y
          in
          equal (array float_exact) (Nx.to_array expected) (Nx.to_array x));
    ]

let measures =
  group "Measures"
    [
      prop "a TAN pixel's measure is its solid angle"
        Gen.(
          triple (int_range (-3000) 3000) (int_range (-3000) 3000)
            (of_list [ 0.031 /. 3600.; 0.5 ]))
        (fun (i, j, scale) ->
          let i, j = if scale > 0.1 then (i / 30, j / 30) else (i, j) in
          let g =
            Grid.pixels ~shape:[| 1; 1 |] f64 (tan ~scale (110.8, -73.4))
          in
          let g = Grid.window ~start:(start i j) ~shape:[| 1; 1 |] g in
          let s = scale *. Float.pi /. 180. in
          (* Column j runs west, -x; row i north, +y. *)
          let x1 = -.(float_of_int j +. 0.5) *. s
          and x2 = -.(float_of_int j -. 0.5) *. s in
          let y1 = (float_of_int i -. 0.5) *. s
          and y2 = (float_of_int i +. 0.5) *. s in
          let rel = if scale > 0.1 then 1e-10 else 5e-9 in
          equal (float_rel ~rel ~abs:0.) (rectangle x1 x2 y1 y2)
            (values Unit.steradian (Grid.measure g)).(0));
      test "a TAN image's measures differ from the reference pixel's by cos³θ"
        (fun () ->
          let g =
            Grid.pixels ~shape:[| 1; 1 |] f64 (tan ~scale:(1. /. 60.) (0., 0.))
          in
          let at i =
            (values Unit.steradian
               (Grid.measure
                  (Grid.window ~start:(start i 0) ~shape:[| 1; 1 |] g))).(0)
          in
          (* One degree north: cos³ of 1° is 1 - 4.6e-4. *)
          equal (float 1e-6)
            (Float.pow (Float.cos (Float.pi /. 180.)) 3.)
            (at 60 /. at 0));
      test "a plane grid's measure is the matrix's determinant" (fun () ->
          let t =
            Transform.(linear (deg (tensor [| 2; 2 |] [| 2.; 0.5; -1.; 3. |])))
          in
          let g = Grid.pixels ~shape:[| 2; 2 |] f64 t in
          equal
            (array (float 1e-14))
            [| 6.5; 6.5; 6.5; 6.5 |]
            (values Unit.(degree ** 2) (Grid.measure g)));
      test "a float32 grid's geometry is its float64 grid's" (fun () ->
          (* CRPIX from a NIRCam header, 2e-4 pixel from its nearest float32. *)
          let t =
            tan
              ~crpix:[| 5098.44382803652; 2372.7753424908956 |]
              ~scale:8.67445987394292e-06
              (110.75544256521349, -73.46776600616062)
          in
          let g64 = Grid.pixels ~shape:[| 3; 4 |] f64 t
          and g32 = Grid.pixels ~shape:[| 3; 4 |] Nx.float32 t in
          let xyz g = Nx.to_array (Direction.xyz (Grid.centres g)) in
          equal (array float_exact) (xyz g64) (xyz g32);
          equal (array float_exact)
            (Nx.to_array
               (Nx.cast Nx.float32
                  (Quantity.value Unit.steradian (Grid.measure g64))))
            (Nx.to_array (Quantity.value Unit.steradian (Grid.measure g32))));
      test "a float32 grid's measures are float32" (fun () ->
          let g = Grid.pixels ~shape:[| 2; 2 |] Nx.float32 Transform.id in
          equal int 4
            (Array.length
               (Nx.to_array (Quantity.value Unit.one (Grid.measure g)))));
    ]

let around =
  group "around"
    [
      test "an odd window is centred on the cell holding the point" (fun () ->
          let g = plane [| 100; 100 |] in
          let w =
            Grid.around
              (one (tensor [| 2 |] [| 40.4; 60.6 |]))
              ~shape:[| 5; 5 |] g
          in
          let (Grid.Pixels { start; _ }) = Grid.kind w in
          equal (array int64) [| 38L; 59L |] (Nx.to_array start));
      test "an even window puts its extra cell on the high side" (fun () ->
          let g = plane [| 100; 100 |] in
          let w =
            Grid.around
              (one (tensor [| 2 |] [| 40.4; 60.6 |]))
              ~shape:[| 4; 4 |] g
          in
          let (Grid.Pixels { start; _ }) = Grid.kind w in
          equal (array int64) [| 39L; 60L |] (Nx.to_array start));
      test "starts clamp to [-shape, base], and NaN goes beyond the base"
        (fun () ->
          let g = plane [| 100; 80 |] in
          let x =
            one (tensor [| 3; 2 |] [| -1e30; 1e30; 1e300; -5.; Float.nan; 3. |])
          in
          let (Grid.Pixels { start; _ }) =
            Grid.kind (Grid.around x ~shape:[| 5; 6 |] g)
          in
          equal (array int64)
            [| -5L; 80L; 100L; -6L; 100L; 80L |]
            (Nx.to_array start));
      test "a direction outside TAN's domain gets a window beyond the base"
        (fun () ->
          let g =
            Grid.pixels ~shape:[| 100; 100 |] f64
              (tan ~crpix:[| 50.; 50. |] ~scale:1e-3 (0., 0.))
          in
          let x =
            Direction.lonlat Frame.icrs
              ~lon:(deg (tensor [| 2 |] [| 0.01; 180. |]))
              ~lat:(deg (tensor [| 2 |] [| 0.; 0. |]))
          in
          let (Grid.Pixels { start; _ }) =
            Grid.kind (Grid.around x ~shape:[| 4; 4 |] g)
          in
          equal (array int64) [| 49L; 39L; 100L; 100L |] (Nx.to_array start));
      test "a direction's window holds its cell" (fun () ->
          let g =
            Grid.pixels ~shape:[| 100; 100 |] f64
              (tan ~crpix:[| 50.; 50. |] ~scale:1e-3 (30., -20.))
          in
          let x =
            Grid.centres (Grid.window ~start:(start 17 81) ~shape:[| 1; 1 |] g)
          in
          let x =
            Nx.Ptree.map (Direction.ptree ())
              (fun _ v -> Nx.reshape [| 3 |] v)
              x
          in
          let (Grid.Pixels { start; _ }) =
            Grid.kind (Grid.around x ~shape:[| 7; 7 |] g)
          in
          equal (array int64) [| 14L; 78L |] (Nx.to_array start));
    ]

let structure =
  group "Structure"
    [
      test "a grid's visits" (fun () ->
          let g =
            Grid.window ~start:(start 1 2) ~shape:[| 3; 3 |]
              (plane [| 10; 10 |])
          in
          expect
            (String.concat "\n"
               (List.map
                  (Format.asprintf "%a" Nx.Ptree.pp_visit)
                  (Nx.Ptree.visits (Grid.ptree ()) g)))
          @@ __POS_OF__
               {|
            the root: case "pixels"
            base: int 2
            base: int 10
            base: int 10
            shape: int 2
            shape: int 3
            shape: int 3
            the root: case "float64"
            start: a leaf
            transform: int 0
            |});
    ]

let () = exit (run "Grid" [ cells; measures; around; structure ])
