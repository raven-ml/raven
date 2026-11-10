(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord

let utm33 = Crs.utm ~zone:33 ~north:true

let points l =
  Nx.create Nx.float64
    [| List.length l; 2 |]
    (Array.of_list (List.concat_map (fun (a, b) -> [ a; b ]) l))

let pairs t =
  let a = Nx.to_array t in
  List.init (Array.length a / 2) (fun k -> (a.(2 * k), a.((2 * k) + 1)))

let pt = pair float_exact float_exact

(* A north-up 10 m grid whose numbers are exact in binary. *)
let north_up =
  Grid.v utm33 ~rows:4 ~cols:3 ~origin:(300000., 5800000.) ~row_step:(0., -10.)
    ~col_step:(10., 0.)

let corner =
  test "pixel point (0, 0) is the first cell's upper-left corner" (fun () ->
      equal (list pt)
        [ (300000., 5800000.); (300015., 5799975.) ]
        (pairs (Grid.world north_up (points [ (0., 0.); (2.5, 1.5) ]))))

let exact_pixel =
  test "an axis-aligned grid maps a border to an integer" (fun () ->
      equal (list pt)
        [ (3., 2.); (3.5, 2.5) ]
        (pairs
           (Grid.pixel north_up
              (points [ (300020., 5799970.); (300025., 5799965.) ]))))

let centres =
  test "centres are each cell's middle, rows then columns" (fun () ->
      let c = Grid.centres ~at:Nx.Host.on north_up in
      equal (array int) [| 4; 3; 2 |] (Nx.shape c);
      equal pt (300005., 5799995.) (Nx.item [ 0; 0; 0 ] c, Nx.item [ 0; 0; 1 ] c);
      equal pt (300025., 5799965.) (Nx.item [ 3; 2; 0 ] c, Nx.item [ 3; 2; 1 ] c))

(* Grids of any invertible map. *)
let affine =
  Gen.(
    let+ ox = float_range (-1e6) 1e6
    and+ oy = float_range (-1e6) 1e7
    and+ a = float_range 0.5 30.
    and+ b = float_range 0.5 30.
    and+ angle = float_range (-.Float.pi) Float.pi
    and+ shear = float_range (-0.5) 0.5 in
    let c = Float.cos angle and s = Float.sin angle in
    let row_step = (s *. a, -.c *. a)
    and col_step = ((c +. (shear *. s)) *. b, (s -. (shear *. c)) *. b) in
    Grid.v utm33 ~rows:100 ~cols:100 ~origin:(ox, oy) ~row_step ~col_step)

let round_trip =
  prop "pixel inverts world"
    (Gen.pair affine
       (Gen.pair (Gen.float_range (-10.) 110.) (Gen.float_range (-10.) 110.)))
    (fun (g, (r, c)) ->
      let r', c' =
        List.hd (pairs (Grid.pixel g (Grid.world g (points [ (r, c) ]))))
      in
      (* The world point is 1e7 m at most, a 2e-9 m ulp over steps of 0.5 m. *)
      less float_exact ~than:1e-7 (Float.hypot (r' -. r) (c' -. c)))

let window =
  prop "a window's cells lie where the grid's do"
    (Gen.pair affine (Gen.pair (Gen.int_range (-5) 50) (Gen.int_range (-5) 50)))
    (fun (g, (row, col)) ->
      let w = Grid.window ~row ~col ~rows:7 ~cols:9 g in
      equal int 7 w.rows;
      equal pt
        (List.hd
           (pairs
              (Grid.world g (points [ (float_of_int row, float_of_int col) ]))))
        w.origin;
      let a = List.hd (pairs (Grid.world w (points [ (2.5, 3.5) ]))) in
      let b =
        List.hd
          (pairs
             (Grid.world g
                (points [ (float_of_int row +. 2.5, float_of_int col +. 3.5) ])))
      in
      less float_exact ~than:1e-6
        (Float.hypot (fst a -. fst b) (snd a -. snd b)))

let refusals =
  test "v refuses a negative size, a non-finite number and parallel steps"
    (fun () ->
      let v ?(rows = 1) ?(origin = (0., 0.)) ?(row_step = (0., -1.)) () =
        Grid.v utm33 ~rows ~cols:1 ~origin ~row_step ~col_step:(1., 0.)
      in
      raises_match (Exn.invalid_arg ~sub:"negative") (fun () -> v ~rows:(-1) ());
      raises_match (Exn.invalid_arg ~sub:"finite") (fun () ->
          v ~origin:(Float.nan, 0.) ());
      raises_match (Exn.invalid_arg ~sub:"finite") (fun () ->
          v ~row_step:(Float.infinity, 0.) ());
      raises_match (Exn.invalid_arg ~sub:"determinant") (fun () ->
          v ~row_step:(2., 0.) ());
      equal int 0
        (Grid.v utm33 ~rows:0 ~cols:0 ~origin:(0., 0.) ~row_step:(0., -1.)
           ~col_step:(1., 0.))
          .rows)

let equality =
  test "equal compares every field exactly" (fun () ->
      let g = north_up in
      equal bool true
        (Grid.equal g (Grid.window ~row:0 ~col:0 ~rows:4 ~cols:3 g));
      equal bool false
        (Grid.equal g (Grid.window ~row:0 ~col:0 ~rows:4 ~cols:2 g));
      equal bool false
        (Grid.equal g (Grid.window ~row:1 ~col:0 ~rows:4 ~cols:3 g));
      let shifted =
        Grid.v utm33 ~rows:4 ~cols:3
          ~origin:(Float.succ 300000., 5800000.)
          ~row_step:(0., -10.) ~col_step:(10., 0.)
      in
      equal bool false (Grid.equal g shifted);
      let other =
        Grid.v
          (Crs.utm ~zone:33 ~north:false)
          ~rows:4 ~cols:3 ~origin:(300000., 5800000.) ~row_step:(0., -10.)
          ~col_step:(10., 0.)
      in
      equal bool false (Grid.equal g other))

let () =
  exit
    (run "jord.grid"
       [
         group "pixels" [ corner; exact_pixel; centres; round_trip; window ];
         group "values" [ refusals; equality ];
       ])
