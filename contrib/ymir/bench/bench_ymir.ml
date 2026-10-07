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

let suite () = [ galactic (); separations () ]
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
