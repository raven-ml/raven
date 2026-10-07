(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ymir's workloads, eager and compiled.

   A compiled row builds its function and calls it once in its setup, so the
   timed region replays the program. *)

open Ymir

let f64 = Nx.float64
let n = 1_000_000
let sync () = Nx_device.synchronize Nx_device.host
let radians x = Quantity.v Unit.radian x
let in_radians q = Quantity.value Unit.radian q

(* [n] directions spread over the sphere, as ICRS vectors [[n; 3]]. *)
let directions seed () =
  let u = Nx.linspace f64 0. 1. n in
  let lon = Nx.mul_s (Nx.add_s (Nx.mul_s u 7919.) seed) (2. *. Float.pi) in
  let lat = Nx.asin (Nx.sub_s (Nx.mul_s u 2.) 1.) in
  Direction.xyz
    (Direction.lonlat Frame.icrs ~lon:(radians lon) ~lat:(radians lat))

let icrs v =
  Nx.Ptree.map (Direction.ptree ())
    (fun _ x -> Nx.cast (Nx.dtype x) v)
    (Direction.of_xyz Frame.icrs (Nx.create f64 [| 3 |] [| 1.; 0.; 0. |]))

(* The Galactic latitude of each ICRS direction. *)
let galactic_lat v =
  in_radians (Direction.lat (Direction.rotate Frame.galactic (icrs v)))

let separation a b = in_radians (Direction.separation (icrs a) (icrs b))

let timed f =
  ignore (Sys.opaque_identity (f ()));
  sync ()

let galactic () =
  Thumper.group "galactic-lat-1m"
    [
      Thumper.bench_with_setup ~setup:(directions 0.) "eager" (fun v ->
          timed (fun () -> galactic_lat v));
      Thumper.bench_with_setup
        ~setup:(fun () ->
          let f = Rune.jit' galactic_lat and v = directions 0. () in
          ignore (Sys.opaque_identity (f v));
          (f, v))
        "compiled"
        (fun (f, v) -> timed (fun () -> f v));
    ]

let separations () =
  let inputs () = (directions 0. (), directions 0.25 ()) in
  Thumper.group "separation-1m"
    [
      Thumper.bench_with_setup ~setup:inputs "eager" (fun (a, b) ->
          timed (fun () -> separation a b));
      Thumper.bench_with_setup
        ~setup:(fun () ->
          let f =
            Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) separation
          in
          let a, b = inputs () in
          ignore (Sys.opaque_identity (f a b));
          (f, a, b))
        "compiled"
        (fun (f, a, b) -> timed (fun () -> f a b));
    ]

(* FITS: what a photometry program and a catalogue reader call. Each file
   is written once in the setup, so the timed region reads it. *)

let fits_side = 4096

let fits_path name =
  Filename.concat
    (Filename.get_temp_dir_name ())
    ("ymir-bench-" ^ name ^ ".fits")

let fits_file name hdus () =
  let path = fits_path name in
  (match Fits.write path hdus with Ok () -> () | Error e -> failwith e);
  match Fits.read path with Ok h -> h | Error e -> failwith e

let fits_image () =
  Nx.init Nx.float32 [| fits_side; fits_side |] (fun i ->
      100.
      +. (Float.of_int (((i.(0) * 7919) + (i.(1) * 104729)) mod 1000) *. 0.01))

let ok = function Ok x -> x | Error e -> failwith e

let fits_reads () =
  let plain () =
    List.hd
      (fits_file "plain"
         [ Fits.Image.hdu Fits.Header.empty (fits_image ()) ]
         ())
  in
  let rice () =
    let t =
      Nx.init Nx.int16 [| fits_side; fits_side |] (fun i ->
          ((i.(0) * 7) + i.(1)) mod 3000)
    in
    List.nth
      (fits_file "rice"
         [ Fits.Image.hdu ~tiles:[| 64; 64 |] Fits.Header.empty t ]
         ())
      1
  in
  Thumper.group "fits-image-4096"
    [
      Thumper.bench_with_setup ~setup:plain "values-float32" (fun h ->
          timed (fun () -> ok (Fits.Image.values Nx.float32 h)));
      Thumper.bench_with_setup ~setup:rice "raw-int16-rice" (fun h ->
          timed (fun () -> ok (Fits.Image.raw Nx.int16 h)));
    ]

(* A 4096² image and its error under a JWST-like TAN header, read as an
   observation. *)
let fits_observation () =
  let set k v h = Fits.Header.set Fits.Value.string k v h in
  let setf k v h = Fits.Header.set Fits.Value.float k v h in
  let pixel = 0.031 /. 3600. in
  let sci =
    Fits.Header.empty |> set "EXTNAME" "SCI" |> set "BUNIT" "MJy/sr"
    |> set "CTYPE1" "RA---TAN" |> set "CTYPE2" "DEC--TAN"
    |> setf "CRPIX1" 2048.5 |> setf "CRPIX2" 2048.5 |> setf "CRVAL1" 110.8375
    |> setf "CRVAL2" (-73.4537) |> setf "CDELT1" (-.pixel)
    |> setf "CDELT2" pixel |> setf "PIXAR_SR" 2.26e-14
  in
  let err = Fits.Header.empty |> set "EXTNAME" "ERR" |> set "BUNIT" "MJy/sr" in
  let hdus () =
    fits_file "observation"
      [
        Fits.Image.hdu sci (fits_image ());
        Fits.Image.hdu err (Nx.mul_s (fits_image ()) 0.01);
      ]
      ()
  in
  Thumper.group "fits-observation-4096"
    [
      Thumper.bench_with_setup ~setup:hdus "sci-err-float32" (fun hdus ->
          timed (fun () ->
              ok
                (Fits.observation ~dtype:Nx.float32 ~frame:Frame.icrs
                   ~data:"SCI" ~error:"ERR" hdus)));
    ]

let fits_quantize () =
  let side = 1024 in
  let image () =
    Nx.init Nx.float32 [| side; side |] (fun i ->
        100. +. sin (Float.of_int ((i.(0) * side) + i.(1))))
  in
  Thumper.group "fits-quantized-1024"
    [
      Thumper.bench_with_setup ~setup:image "write" (fun t ->
          let path = fits_path "quantized" in
          timed (fun () ->
              ok
                (Fits.write path
                   [ Fits.Image.quantized 16. Fits.Header.empty t ])));
    ]

let fits_table () =
  let columns = 50 and rows = 100_000 in
  let table () =
    let col i =
      ( Printf.sprintf "c%d" i,
        Fits.Header.empty,
        Fits.Table.Array
          {
            values =
              Nx.P (Nx.init f64 [| rows |] (fun r -> Float.of_int (r.(0) + i)));
            validity = None;
          } )
    in
    let hdu = ok (Fits.Table.hdu Fits.Header.empty (List.init columns col)) in
    List.nth (fits_file "table" [ hdu ] ()) 1
  in
  Thumper.group "fits-table-50x100k"
    [
      Thumper.bench_with_setup ~setup:table "raw-one-column" (fun h ->
          timed (fun () -> ok (Fits.Table.raw f64 "c7" h)));
      Thumper.bench_with_setup ~setup:table "read" (fun h ->
          timed (fun () -> ok (Fits.Table.read h)));
    ]

(* Grids: a JWST-like TAN header with 0.031 arcsecond pixels. *)

let degrees x = Quantity.v Unit.degree x
let arcsec x = Quantity.v Unit.arcsecond x
let pixel = 0.031 /. 3600.

let wcs () =
  Transform.(
    axes [| 1; 0 |] ~origin:1
    >> shift (Quantity.v Unit.one (Nx.create f64 [| 2 |] [| 2048.5; 2048.5 |]))
    >> linear (degrees (Nx.create f64 [| 2; 2 |] [| -.pixel; 0.; 0.; pixel |]))
    >> celestial Tan Frame.icrs ~pv:(Nx.zeros f64 [| 0 |])
         ~native:(degrees (Nx.create f64 [| 2 |] [| 0.; 90. |]))
         ~crval:(degrees (Nx.create f64 [| 2 |] [| 110.8375; -73.4537 |]))
         ~lonpole:(degrees (Nx.scalar f64 180.))
         ~latpole:(degrees (Nx.scalar f64 90.)))

(* [n] pixel coordinates spread over a 4096² image, [[n; 2]]. *)
let pixels () =
  let u = Nx.linspace f64 0. 1. n in
  let v = Nx.mul_s u 7919. in
  let v = Nx.sub v (Nx.floor v) in
  Nx.stack ~axis:(-1) [ Nx.mul_s u 4095.; Nx.mul_s v 4095. ]

let to_sky p = Direction.xyz (Transform.apply (wcs ()) (Quantity.v Unit.one p))

let wcs_rows () =
  Thumper.group "wcs-1m"
    [
      Thumper.bench_with_setup ~setup:pixels "eager" (fun p ->
          timed (fun () -> to_sky p));
      Thumper.bench_with_setup
        ~setup:(fun () ->
          let f = Rune.jit' to_sky and p = pixels () in
          ignore (Sys.opaque_identity (f p));
          (f, p))
        "compiled"
        (fun (f, p) -> timed (fun () -> f p));
    ]

(* Distortions and solved projections, on [m] points: a SIP header of order 3
   over the same image, its inverse solving each point, and ZPN's
   deprojection solving each radius. *)

let m = 100_000

let sip_wcs () =
  (* [coefficients ts] is a 4 × 4 SIP matrix from [((p, q), v)] terms. *)
  let coefficients ts =
    let c = Array.make 16 0. in
    List.iter (fun ((p, q), v) -> c.((4 * p) + q) <- v) ts;
    Nx.create f64 [| 4; 4 |] c
  in
  let a =
    coefficients
      [ ((2, 0), 2.1e-6); ((1, 1), -1.4e-6); ((0, 2), 8.2e-7); ((3, 0), 3.1e-10) ]
  and b =
    coefficients
      [ ((2, 0), -9.5e-7); ((1, 1), 1.9e-6); ((0, 2), -2.6e-6); ((0, 3), 1.4e-10) ]
  in
  Transform.(
    axes [| 1; 0 |] ~origin:1
    >> shift (Quantity.v Unit.one (Nx.create f64 [| 2 |] [| 2048.5; 2048.5 |]))
    >> sip (a, b)
    >> linear (degrees (Nx.create f64 [| 2; 2 |] [| -.pixel; 0.; 0.; pixel |]))
    >> celestial Tan Frame.icrs ~pv:(Nx.zeros f64 [| 0 |])
         ~native:(degrees (Nx.create f64 [| 2 |] [| 0.; 90. |]))
         ~crval:(degrees (Nx.create f64 [| 2 |] [| 110.8375; -73.4537 |]))
         ~lonpole:(degrees (Nx.scalar f64 180.))
         ~latpole:(degrees (Nx.scalar f64 90.)))

let sip_pixels () = Nx.slice [ Nx.R (0, m) ] (pixels ())

let sip_sky () =
  let t = sip_wcs () in
  (t, Transform.apply t (Quantity.v Unit.one (sip_pixels ())))

let to_pixels t d = Quantity.value Unit.one (Transform.apply (Transform.inverse t) d)

let zpn () =
  let pv = Nx.init f64 [| 30 |] (function [| 1 |] -> 1. | [| 3 |] -> -0.25 | _ -> 0.) in
  let t =
    Transform.celestial Zpn Frame.icrs ~pv
      ~native:(degrees (Nx.create f64 [| 2 |] [| 0.; 90. |]))
      ~crval:(degrees (Nx.create f64 [| 2 |] [| 120.; -45. |]))
      ~lonpole:(degrees (Nx.scalar f64 180.))
      ~latpole:(degrees (Nx.scalar f64 90.))
  in
  let u = Nx.linspace f64 0. 1. m in
  let r = Nx.mul_s u 40. and phi = Nx.mul_s u (7919. *. 2. *. Float.pi) in
  (t, degrees (Nx.stack ~axis:(-1) [ Nx.mul r (Nx.sin phi); Nx.mul r (Nx.cos phi) ]))

let solved_rows () =
  Thumper.group "solved-100k"
    [
      Thumper.bench_with_setup ~setup:sip_pixels "sip-forward-eager" (fun p ->
          timed (fun () ->
              Direction.xyz (Transform.apply (sip_wcs ()) (Quantity.v Unit.one p))));
      Thumper.bench_with_setup ~setup:sip_sky "sip-inverse-eager" (fun (t, d) ->
          timed (fun () -> to_pixels t d));
      Thumper.bench_with_setup ~setup:zpn "zpn-deproject-eager" (fun (t, p) ->
          timed (fun () -> Direction.xyz (Transform.apply t p)));
    ]

let mosaic () =
  let shape = [| 4096; 4096 |] in
  let g = Grid.pixels ~shape f64 (wcs ()) in
  let data =
    Nx.init f64 shape (fun i ->
        float_of_int (((i.(0) * 31) + (i.(1) * 17)) mod 101))
  in
  Observation.v
    ~area:(Quantity.v Unit.steradian (Nx.scalar f64 2.26e-14))
    g
    (Quantity.v Unit.(symbol "Jy" / steradian) data)

let target =
  lazy
    (Direction.lonlat Frame.icrs
       ~lon:(degrees (Nx.scalar f64 110.8375))
       ~lat:(degrees (Nx.scalar f64 (-73.4537))))

(* The Guide's aperture: a 0.5 arcsecond circle less the mean of a 1-1.5
   arcsecond annulus, on a 104 × 104 stamp, as a function of [east; north;
   radius] in arcseconds. *)
let flux stamp p =
  let target = Lazy.force target in
  let at =
    Transform.(about target >> shift (arcsec (Nx.slice [ Nx.R (0, 2) ] p)))
  in
  let sum =
    Observation.integrate
      (Region.circle at ~radius:(arcsec (Nx.slice [ Nx.I 2 ] p)))
      stamp
  in
  let sky =
    Observation.integrate
      (Region.annulus at
         ~inner:(arcsec (Nx.scalar f64 1.))
         ~outer:(arcsec (Nx.scalar f64 1.5)))
      stamp
  in
  Quantity.(
    value (Unit.symbol "Jy")
      (sub sum.value (mul (div sky.value sky.area) sum.area)))

let aperture_inputs () =
  let stamp =
    Observation.around (Lazy.force target) ~shape:[| 104; 104 |] (mosaic ())
  in
  (stamp, Nx.create f64 [| 3 |] [| 0.; 0.; 0.5 |])

let aperture_rows () =
  let grad stamp p = Rune.value_and_grad' (flux stamp) p in
  Thumper.group "aperture-104"
    [
      Thumper.bench_with_setup ~setup:aperture_inputs "eager" (fun (s, p) ->
          timed (fun () -> flux s p));
      Thumper.bench_with_setup ~setup:aperture_inputs "eager-grad"
        (fun (s, p) -> timed (fun () -> grad s p));
      Thumper.bench_with_setup
        ~setup:(fun () ->
          let f =
            Rune.jit
              Nx.Ptree.(
                Observation.ptree () @-> tensor @-> returns (pair tensor tensor))
              grad
          in
          let s, p = aperture_inputs () in
          ignore (Sys.opaque_identity (f s p));
          (f, s, p))
        "compiled-grad"
        (fun (f, s, p) -> timed (fun () -> f s p));
    ]

(* The difference of two exposures of the mosaic, with their variances. *)
let difference_rows () =
  let exposures () =
    let o = mosaic () in
    let variance =
      Quantity.v
        Unit.((symbol "Jy" / steradian) ** 2)
        (Nx.full f64 [| 4096; 4096 |] 0.25)
    in
    let o = Observation.v ~variance ?area:(Observation.area o)
        (Observation.grid o) (Observation.data o) in
    (o, Observation.scale (Quantity.v Unit.one (Nx.scalar f64 0.5)) o)
  in
  Thumper.group "observation-4096"
    [
      Thumper.bench_with_setup ~setup:exposures "sub-eager" (fun (a, b) ->
          timed (fun () -> Observation.data (Observation.sub a b)));
    ]

(* An ellipse and a hexagon about the aperture's centre on its stamp. *)
let shape_rows () =
  let stamp () = Observation.grid (fst (aperture_inputs ())) in
  let at () = Transform.about (Lazy.force target) in
  let ellipse g =
    Region.weights
      (Region.ellipse (at ())
         ~a:(arcsec (Nx.scalar f64 1.2))
         ~b:(arcsec (Nx.scalar f64 0.7))
         ~angle:(degrees (Nx.scalar f64 33.)))
      g
  in
  let hexagon g =
    let v =
      Nx.init f64 [| 6; 2 |] (fun i ->
          let a = Float.pi *. float_of_int i.(0) /. 3. in
          (if i.(1) = 0 then Float.sin a else Float.cos a) *. 1.2 /. 3600.)
    in
    Region.weights
      (Region.polygon f64 (at ())
         (Transform.apply
            (Transform.inverse (at ()))
            (degrees v)))
      g
  in
  Thumper.group "shapes-104"
    [
      Thumper.bench_with_setup ~setup:stamp "ellipse-eager" (fun g ->
          timed (fun () -> ellipse g));
      Thumper.bench_with_setup ~setup:stamp "polygon-eager" (fun g ->
          timed (fun () -> hexagon g));
    ]

(* Cosmology *)

let planck = Cosmology.planck2018 ~codata:Codata.v2022 f64

let compile f x =
  let f = Rune.jit' f in
  ignore (Sys.opaque_identity (f x));
  (f, x)

(* A supernova fit's log density: the distance moduli of 1590 redshifts in
   each of 4 chains, Omega_cb of shape [4; 1]. *)
let supernovae () =
  let z = Nx.linspace f64 0.01 2.3 1590 in
  let modulus omega_cb =
    Cosmology.distance_modulus { planck with omega_cb } ~observed:z z
  in
  let chains () = Nx.create f64 [| 4; 1 |] [| 0.28; 0.3; 0.31; 0.33 |] in
  Thumper.group "distance-modulus-4x1590"
    [
      Thumper.bench_with_setup ~setup:chains "eager" (fun x ->
          timed (fun () -> modulus x));
      Thumper.bench_with_setup
        ~setup:(fun () -> compile modulus (chains ()))
        "compiled"
        (fun (f, x) -> timed (fun () -> f x));
      Thumper.bench_with_setup
        ~setup:(fun () ->
          compile (Rune.grad' (fun x -> Nx.sum (modulus x))) (chains ()))
        "grad"
        (fun (f, x) -> timed (fun () -> f x));
    ]

(* Ages at 10^3 redshifts, with one massive species. *)
let ages () =
  let age z =
    Quantity.value Unit.(giga Units.julian_year) (Cosmology.age planck z)
  in
  let redshifts () = Nx.logspace f64 0. 6. 1_000 in
  Thumper.group "age-1k"
    [
      Thumper.bench_with_setup ~setup:redshifts "eager" (fun z ->
          timed (fun () -> age z));
      Thumper.bench_with_setup
        ~setup:(fun () -> compile age (redshifts ()))
        "compiled"
        (fun (f, z) -> timed (fun () -> f z));
    ]

let suite () =
  [
    galactic ();
    separations ();
    wcs_rows ();
    solved_rows ();
    aperture_rows ();
    shape_rows ();
    difference_rows ();
    fits_reads ();
    fits_observation ();
    fits_quantize ();
    fits_table ();
    supernovae ();
    ages ();
  ]

let config = Thumper.Config.(default |> deadline 60.)

let () =
  match Array.to_list Sys.argv with
  | [ _; "--warm" ] ->
      (* Each case once, in as few calls as a trial takes: what the setups
         compile lands in tolk's disk cache, which a measurement then reads. *)
      ignore
        (Thumper.measure
           ~config:Thumper.Config.(config |> samples 3 |> warmup 0.)
           (suite ()))
  | _ ->
      Thumper.run "ymir" ~config
        ~budgets:
          [
            Thumper.Budget.no_slower_than 0.05;
            Thumper.Budget.no_more_alloc_than 0.01;
          ]
        (suite ())
      |> exit
