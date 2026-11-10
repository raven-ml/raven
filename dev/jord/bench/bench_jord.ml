(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Jord

(* The grid of GeoTessera's landmask grid_16.35_48.05.tiff, 10 m cells in UTM
   33N, and fields tiling it as parcels do: the cells between the corners of a
   lattice every 25 by 15 cells, each corner moved by up to 4 cells, so
   neighbours share their edges. Half the parcels are fields, about 1150 of them
   with four to five edges each. *)
let grid =
  Grid.v
    (Crs.utm ~zone:33 ~north:true)
    ~rows:1124 ~cols:765
    ~origin:(596785.7115403633, 5328362.815087351)
    ~row_step:(0., -10.) ~col_step:(10., 0.)

let fields =
  let st = Random.State.make [| 7 |] in
  let step_r = 25 and step_c = 15 and jitter = 4. in
  let nr = (grid.rows / step_r) + 1 and nc = (grid.cols / step_c) + 1 in
  let corner =
    Array.init (nr + 1) (fun i ->
        Array.init (nc + 1) (fun j ->
            let move () = Random.State.float st (2. *. jitter) -. jitter in
            ( float_of_int (i * step_r) +. move (),
              float_of_int (j * step_c) +. move () )))
  in
  let quads = ref [] in
  for i = 0 to nr - 1 do
    for j = 0 to nc - 1 do
      if Random.State.bool st then
        quads :=
          [
            corner.(i).(j);
            corner.(i).(j + 1);
            corner.(i + 1).(j + 1);
            corner.(i + 1).(j);
          ]
          :: !quads
    done
  done;
  let quads = List.rev !quads in
  let n = List.length quads in
  let pts =
    Array.of_list
      (List.concat_map (List.concat_map (fun (r, c) -> [ r; c ])) quads)
  in
  let px = Nx.create Nx.float64 [| 4 * n; 2 |] pts in
  Polygons.v grid.crs (Grid.world grid px)
    ~rings:(Array.init (n + 1) (fun k -> 4 * k))
    ~features:(Array.init (n + 1) Fun.id)

let () =
  Thumper.run "jord"
    [
      Thumper.group "rasterize"
        [
          Thumper.bench "centres/tile" (fun () -> Rasterize.centres grid fields);
          Thumper.bench "touched/tile" (fun () -> Rasterize.touched grid fields);
        ];
    ]
  |> exit
