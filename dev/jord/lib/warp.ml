(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Points *)

let centres ~at crs (dst : Grid.t) =
  let c = Grid.centres ~at dst in
  if Crs.equal crs dst.crs then c
  else
    let ll = Crs.transform ~src:dst.crs ~dst:Crs.lonlat c in
    let inside = Crs.in_domain crs ll in
    let inside = Nx.reshape [| dst.rows; dst.cols; 1 |] inside in
    let ll = Nx.where inside ll (Nx.full_like ll Float.nan) in
    Crs.transform ~src:Crs.lonlat ~dst:crs ll

(* A product that OCaml's arm64 backend cannot fuse into the addition that uses
   it, as [Grid]'s. *)
let mul a b = Sys.opaque_identity (a *. b)

(* The least longitude of [g]'s four corners, as [Grid.world] computes it. *)
let west (g : Grid.t) =
  let ox, _ = g.origin and rx, _ = g.row_step and cx, _ = g.col_step in
  let r = float_of_int g.rows and c = float_of_int g.cols in
  let lon r c = mul r rx +. mul c cx +. ox in
  Float.min
    (Float.min (lon 0. 0.) (lon r 0.))
    (Float.min (lon 0. c) (lon r c))

let pixel (g : Grid.t) x =
  match g.crs with
  | Crs.Lonlat ->
      let batch, lon, lat = Points.split "Jord.Warp.pixel" x in
      let w = west g in
      let turns = Nx.floor (Nx.div_s (Nx.sub_s lon w) 360.) in
      let lon = Nx.sub lon (Nx.mul_s turns 360.) in
      Grid.pixel g (Points.join batch lon lat)
  | Crs.Utm _ | Crs.Tm35fin -> Grid.pixel g x

(* Samplers *)

(* [x]'s rows and columns and its other axes, after checking [valid]. *)
let axes fn ?valid x =
  let shape = Nx.shape x in
  let rank = Array.length shape in
  if rank < 2 then
    invalid_arg (strf "%s: the array's rank %d is below 2" fn rank);
  let rows = shape.(0) and cols = shape.(1) in
  (match valid with
  | Some v when Nx.shape v <> [| rows; cols |] ->
      invalid_arg
        (strf "%s: the mask's shape is not the array's first two axes" fn)
  | _ -> ());
  (rows, cols, Array.sub shape 2 (rank - 2))

(* Whether cell (i, j), of float indices, lies in the [rows × cols] grid, and
   its flat index there, -1 elsewhere. Each axis is checked alone, since a
   flat index of column -1 or [cols] lands in the grid. NaN is in no cell, and
   is never cast. *)
let cell ~rows ~cols i j =
  let within v n = Nx.logical_and (Nx.greater_equal_s v 0.) (Nx.less_s v n) in
  let inside =
    Nx.logical_and (within i (float_of_int rows)) (within j (float_of_int cols))
  in
  let k = Nx.add (Nx.mul_s i (float_of_int cols)) j in
  (inside, Nx.cast Nx.int64 (Nx.where inside k (Nx.full_like k (-1.))))

(* [x] as one row per cell, [[rows · cols; ...]], and whether the cells of
   flat indices [k], [inside] the grid, have data. *)
let cells ~rows ~cols rest ?valid x =
  let data =
    match valid with
    | None -> fun _ inside -> inside
    | Some v ->
        let v = Nx.reshape [| rows * cols |] v in
        fun k _ -> Nx.take k v
  in
  (Nx.reshape (Array.append [| rows * cols |] rest) x, data)

(* [m], a mask of [n] points, broadcast over [rank] trailing axes. *)
let lift n rank m = Nx.reshape (Array.append [| n |] (Array.make rank 1)) m

let nearest ?valid p x =
  let rows, cols, rest = axes "Jord.Warp.nearest" ?valid x in
  let batch, r, c = Points.split "Jord.Warp.nearest" p in
  let n = Nx.numel r in
  let x, data = cells ~rows ~cols rest ?valid x in
  let inside, k = cell ~rows ~cols (Nx.floor r) (Nx.floor c) in
  let ok = data k inside in
  let v = Nx.take ~axis:0 k x in
  let v = Nx.where (lift n (Array.length rest) ok) v (Nx.zeros_like v) in
  (Nx.reshape (Array.append batch rest) v, Nx.reshape batch ok)

(* How a float64 mean is stored in [dtype]: as it is for floats, which the
   cast rounds once; as ⌊v + 1/2⌋ for integers, GDAL's rule on every sign,
   whose integral result the cast keeps. *)
let rounding (type v s) (dtype : (v, s) Nx.dtype) =
  match Nx.Dtype.kind dtype with
  | Float -> Fun.id
  | Signed | Unsigned -> fun v -> Nx.floor (Nx.add_s v 0.5)
  | Complex | Boolean ->
      let name = Nx.Dtype.name dtype in
      invalid_arg (strf "Jord.Warp.bilinear: %s is not a real dtype" name)

let bilinear ?valid p x =
  let round = rounding (Nx.dtype x) in
  let rows, cols, rest = axes "Jord.Warp.bilinear" ?valid x in
  let batch, r, c = Points.split "Jord.Warp.bilinear" p in
  let n = Nx.numel r and rank = Array.length rest in
  let dtype = Nx.dtype x in
  let x, data = cells ~rows ~cols rest ?valid x in
  (* The four cells around (r, c) are those whose centres bound it: rows i0
     and i0 + 1 from u = r - 1/2, weighted 1 - a and a, and likewise for
     columns. *)
  let u = Nx.sub_s r 0.5 and v = Nx.sub_s c 0.5 in
  let i0 = Nx.floor u and j0 = Nx.floor v in
  let a = Nx.sub u i0 and b = Nx.sub v j0 in
  let along i0 a = [ (i0, Nx.rsub_s 1. a); (Nx.add_s i0 1., a) ] in
  (* A tap left out is selected away, never multiplied by 0: NaN · 0 is NaN. *)
  let tap (i, wi) (j, wj) =
    let w = Nx.mul wi wj in
    let inside, k = cell ~rows ~cols i j in
    let use = Nx.logical_and (data k inside) (Nx.greater_s w 0.) in
    let xk = Nx.cast Nx.float64 (Nx.take ~axis:0 k x) in
    let wx = Nx.mul (lift n rank w) xk in
    ( Nx.where (lift n rank use) wx (Nx.zeros_like wx),
      Nx.where use w (Nx.zeros_like w) )
  in
  let taps =
    List.concat_map (fun i -> List.map (tap i) (along j0 b)) (along i0 a)
  in
  let total l = List.fold_left Nx.add (List.hd l) (List.tl l) in
  let sum = total (List.map fst taps) and weight = total (List.map snd taps) in
  let inside, k = cell ~rows ~cols (Nx.floor r) (Nx.floor c) in
  let ok = data k inside in
  let mean = Nx.div sum (lift n rank weight) in
  let mean = Nx.where (lift n rank ok) mean (Nx.zeros_like mean) in
  ( Nx.reshape (Array.append batch rest) (Nx.cast dtype (round mean)),
    Nx.reshape batch ok )

(* Windows *)

type rule = Nearest | Bilinear

let window rule (g : Grid.t) p =
  let _, r, c = Points.split "Jord.Warp.window" p in
  let inside, _ = cell ~rows:g.rows ~cols:g.cols (Nx.floor r) (Nx.floor c) in
  if not (Nx.item [] (Nx.any inside)) then (0, 0, 0, 0)
  else
    (* The least and the greatest index a point of a cell reads along one
       axis, of [n] cells. *)
    let span x n =
      let lo, hi =
        match rule with
        | Nearest -> (Nx.floor x, Nx.floor x)
        | Bilinear ->
            ( Nx.maximum_s (Nx.floor (Nx.sub_s x 0.5)) 0.,
              Nx.minimum_s (Nx.floor (Nx.add_s x 0.5)) (float_of_int (n - 1)) )
      in
      let pick v fill = Nx.where inside v (Nx.full_like v fill) in
      let first = Nx.item [] (Nx.min (pick lo Float.infinity)) in
      let last = Nx.item [] (Nx.max (pick hi Float.neg_infinity)) in
      (int_of_float first, int_of_float last - int_of_float first + 1)
    in
    let row, rows = span r g.rows and col, cols = span c g.cols in
    (row, col, rows, cols)
