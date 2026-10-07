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

module Fits = Ymir_fits.Fits

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

let suite () =
  [
    galactic (); separations (); fits_reads (); fits_quantize (); fits_table ();
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
