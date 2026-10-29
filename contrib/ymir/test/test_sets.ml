(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Ymir off the host: values on a set of two memory devices, which compute on
   the host, whole on each device or split between them. A function's results
   and the constants it makes lie where its operands do. *)

open Windtrap
open Ymir

module Pair =
  (val Nx.devices
         (List.init 2 (fun k ->
              Result.get_ok (Rig.memory_device (Printf.sprintf "ymir-s%d" k)))))

let f64 = Nx.float64
let pp p = Format.asprintf "%a" Nx.Placement.pp p
let lies p name x = equal ~msg:name string (pp p) (pp (Nx.placement x))
let deg x = Quantity.v Unit.degree x
let placements = [ ("whole", Pair.on); ("split", Pair.split ~axis:0) ]

(* Two directions on the host. *)
let host_sky () =
  Direction.lonlat Frame.icrs
    ~lon:(deg (Nx.create f64 [| 2 |] [| 10.; 20. |]))
    ~lat:(deg (Nx.create f64 [| 2 |] [| -5.; 30. |]))

module Sky = struct
  type (_, 'd) t = (Frame.icrs, 'd) Direction.t

  let walk = Direction.walk
end

let directions =
  List.concat_map
    (fun (where, p) ->
      [
        test ("a placed direction lies " ^ where) (fun () ->
            let d = Nx.Ptree.place (module Sky) p (host_sky ()) in
            lies p "xyz" (Direction.xyz d);
            lies p "lon" (Quantity.value Unit.degree (Direction.lon d)));
        test ("a gnomonic projection lies with its points " ^ where)
          (fun () ->
            let d = Nx.Ptree.place (module Sky) p (host_sky ()) in
            let at =
              Nx.Ptree.place (module Sky) Pair.on
                (Direction.lonlat Frame.icrs ~lon:(deg (Nx.scalar f64 15.))
                   ~lat:(deg (Nx.scalar f64 10.)))
            in
            let t = Transform.gnomonic at in
            lies p "plane" (Quantity.value Unit.degree (Transform.apply t d));
            lies p "covers" (Option.get (Transform.covers t d)));
      ])
    placements

let quantities =
  [
    test "a placed quantity lies where it is placed" (fun () ->
        let q = Quantity.v Unit.metre (Nx.create f64 [| 2 |] [| 1.; 2. |]) in
        let q = Nx.Ptree.place (module Quantity.Structure) Pair.on q in
        lies Pair.on "payload" (Quantity.value Unit.metre q));
    test "a placed cosmology computes where it lies" (fun () ->
        let c = Cosmology.planck2018 ~codata:Codata.v2022 f64 in
        let c = Nx.Ptree.place (module Cosmology.Structure) Pair.on c in
        let z = Nx.place Pair.on (Nx.create f64 [| 2 |] [| 0.5; 1. |]) in
        lies Pair.on "comoving distance"
          (Quantity.value Unit.metre (Cosmology.comoving_distance c z));
        lies Pair.on "hubble"
          (Quantity.value
             Unit.(kilo metre / second / mega Units.parsec)
             (Cosmology.hubble c z));
        let z = Nx.place (Pair.split ~axis:0) z in
        lies (Pair.split ~axis:0) "comoving distance of split redshifts"
          (Quantity.value Unit.metre (Cosmology.comoving_distance c z)));
  ]

(* A gnomonic WCS of a small image at [at], about (10°, 20°). *)
let grids =
  let wcs at =
    Transform.(
      axes [| 1; 0 |] ~origin:1
      >> shift (Quantity.v Unit.one (Nx.place at (Nx.create f64 [| 2 |] [| 2.; 3. |])))
      >> linear
           (deg (Nx.place at (Nx.create f64 [| 2; 2 |] [| -1e-3; 0.; 0.; 1e-3 |])))
      >> celestial Tan Frame.icrs
           ~pv:(Nx.zeros_at at f64 [| 0 |])
           ~native:(deg (Nx.place at (Nx.create f64 [| 2 |] [| 0.; 90. |])))
           ~crval:(deg (Nx.place at (Nx.create f64 [| 2 |] [| 10.; 20. |])))
           ~lonpole:(deg (Nx.scalar_at at f64 180.))
           ~latpole:(deg (Nx.scalar_at at f64 90.)))
  in
  [
    test "a grid on the set lies there" (fun () ->
        let g = Grid.pixels ~at:Pair.on ~shape:[| 4; 6 |] f64 (wcs Pair.on) in
        lies Pair.on "centres" (Direction.xyz (Grid.centres g));
        lies Pair.on "measure" (Quantity.value Unit.steradian (Grid.measure g));
        lies Pair.on "agree" (Grid.agree g g));
    test "an integral lies with its observation" (fun () ->
        let g = Grid.pixels ~at:Pair.on ~shape:[| 4; 6 |] f64 (wcs Pair.on) in
        let data = Quantity.v Unit.one (Nx.ones_at Pair.on f64 [| 4; 6 |]) in
        let o = Observation.v g data in
        let about =
          Nx.Ptree.place (module Sky) Pair.on
            (Direction.lonlat Frame.icrs ~lon:(deg (Nx.scalar f64 10.))
               ~lat:(deg (Nx.scalar f64 20.)))
        in
        let r =
          Region.circle (Transform.gnomonic about)
            ~radius:(deg (Nx.scalar_at Pair.on f64 1e-3))
        in
        let v = (Observation.integrate r o).value in
        lies Pair.on "value" (Quantity.value (Quantity.unit v) v));
  ]

(* A TAN header about (10°, 20°), read at a placement of the set. *)
let header =
  let module H = Ymir_fits.Fits.Header in
  let module V = Ymir_fits.Fits.Value in
  H.(
    empty
    |> set V.string "CTYPE1" "RA---TAN"
    |> set V.string "CTYPE2" "DEC--TAN"
    |> set V.float "CRPIX1" 2. |> set V.float "CRPIX2" 3.
    |> set V.float "CRVAL1" 10. |> set V.float "CRVAL2" 20.
    |> set V.float "CDELT1" (-1e-3) |> set V.float "CDELT2" 1e-3)

let keywords =
  let module H = Ymir_fits.Fits.Header in
  let module V = Ymir_fits.Fits.Value in
  {
    Wcs.float = (fun k -> H.find V.float k header);
    int = (fun k -> H.find V.int k header);
    text = (fun k -> H.find V.string k header);
  }

let wcs =
  [
    test "a WCS read at the set maps pixels there" (fun () ->
        let t = Result.get_ok (Wcs.read ~at:Pair.on Frame.icrs keywords) in
        let px =
          Quantity.v Unit.one
            (Nx.place (Pair.split ~axis:0)
               (Nx.create f64 [| 2; 2 |] [| 0.; 0.; 1.; 2. |]))
        in
        let sky = Transform.apply t px in
        lies (Pair.split ~axis:0) "directions" (Direction.xyz sky);
        lies (Pair.split ~axis:0) "covers" (Option.get (Transform.covers t px)));
  ]

let () =
  exit
    (run "Ymir on a device set"
       [
         group "directions" directions;
         group "quantities" quantities;
         group "grids" grids;
         group "wcs" wcs;
       ])
