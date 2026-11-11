(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Jord
open Tessera

(* Cambridge, England: a small area both collections cover. *)
let lon = 0.12
let lat = 52.2
let bbox = (lon -. 0.01, lat -. 0.01, lon +. 0.01, lat +. 0.01)

(* July 2025: Sentinel-2 bands written since March 2025 store 15 bits. *)
let days = ((2025, 7, 1), (2025, 7, 31))

(* MPC's bands store tiles of 512². *)
let tile = 512.

(* A grid of [n] × [n] cells of 10 m in [src]'s system, centred on the tile
   corner nearest the area's centre and 3.3 m off [src]'s lattice, so that
   its window reads four tiles and its centres fall between [src]'s. *)
let grid (src : Grid.t) n =
  let p = Nx.create Nx.float64 [| 1; 2 |] [| lon; lat |] in
  let p = Grid.pixel src (Crs.transform ~src:Crs.lonlat ~dst:src.crs p) in
  let corner v = (Float.round (v /. tile) *. tile) -. Float.of_int (n / 2) in
  let r = corner (Nx.item [ 0; 0 ] p) and c = corner (Nx.item [ 0; 1 ] p) in
  let w = Grid.world src (Nx.create Nx.float64 [| 1; 2 |] [| r; c |]) in
  let x = Nx.item [ 0; 0 ] w +. 3.3 and y = Nx.item [ 0; 1 ] w -. 3.3 in
  Grid.v src.crs ~rows:n ~cols:n ~origin:(x, y) ~row_step:(0., -10.)
    ~col_step:(10., 0.)

let session = Stac.session ()

let read ~dir ~name ~collection ~key ~save n =
  let items = Stac.search ~collection ~bbox days in
  Printf.printf "%s: %d items\n%!" collection (List.length items);
  let item = List.hd items in
  let asset = List.assoc key item.assets in
  let src =
    match asset.grid with
    | Some (Ok g) -> g
    | _ -> failwith (item.id ^ ": no grid")
  in
  let c = Warp.centres ~at:Nx.Host.on src.crs (grid src n) in
  let w = Stac.window session Bilinear asset src.crs c in
  let shape = save (Filename.concat dir (name ^ ".npy")) w.samples in
  Printf.printf "%s %s %d %d %d %d\n%!" name asset.href w.row w.col shape.(0)
    shape.(1)

let () =
  let dir = Sys.argv.(1) in
  read ~dir ~name:"s2" ~collection:"sentinel-2-l2a" ~key:"B04" 100
    ~save:(fun f x ->
      let x = Nx.unpack Nx.uint16 x in
      Nx_io.save_npy f x;
      Nx.shape x);
  read ~dir ~name:"s1" ~collection:"sentinel-1-rtc" ~key:"vv" 64
    ~save:(fun f x ->
      let x = Nx.unpack Nx.float32 x in
      Nx_io.save_npy f x;
      Nx.shape x)
