(* Observations and aperture sums.

   An observation is data on a grid, with an optional variance and a mask of
   valid samples. [integrate] sums it over a region with each cell weighed by
   its exact overlap, and propagates the variance. *)

open Ymir

let f64 = Nx.float64
let pixels x = Quantity.v Unit.one x
let point r c = pixels (Nx.create f64 [| 2 |] [| r; c |])
let size x = pixels (Nx.scalar f64 x)

(* Counts per cell: [Grid.cell] in the unit makes each cell count once. *)
let rate = Unit.(symbol "electron" / second / Grid.cell)
let e_s = Unit.(symbol "electron" / second)

(* A 64x64 image: a Gaussian source of 5000 e/s total at (30.3, 33.6) on a flat
   background of 2 e/s per cell. *)
let source_total = 5000.
let background = 2.

let image =
  let sigma = 2. in
  Nx.init f64 [| 64; 64 |] (fun i ->
      let r = float_of_int i.(0) -. 30.3 and c = float_of_int i.(1) -. 33.6 in
      background
      +. source_total
         *. exp (-.((r *. r) +. (c *. c)) /. (2. *. sigma *. sigma))
         /. (2. *. Float.pi *. sigma *. sigma))

let print name (i : Nx.float64_elt Observation.integral) =
  Printf.printf "%-8s sum = %9.3f e/s  area = %7.3f cells  coverage = %.3f\n"
    name
    (Nx.item [] (Quantity.value e_s i.value))
    (Nx.item [] (Quantity.value Grid.cell i.area))
    (Nx.item [] i.coverage)

let () =
  (* Poisson-like variance, and one dead pixel near the source. *)
  let variance = Quantity.v Unit.(rate ** 2) (Nx.abs image) in
  let valid =
    Nx.init Nx.bit [| 64; 64 |] (fun i -> not (i.(0) = 31 && i.(1) = 36))
  in
  let grid = Grid.pixels ~shape:[| 64; 64 |] f64 Transform.id in
  let obs = Observation.v ~variance ~valid grid (Quantity.v rate image) in

  (* A circular aperture and a background annulus about the source. *)
  let at = Transform.shift (point 30.3 33.6) in
  let aperture = Region.circle at ~radius:(size 8.) in
  let sky = Region.annulus at ~inner:(size 12.) ~outer:(size 18.) in
  let a = Observation.integrate aperture obs in
  let s = Observation.integrate sky obs in
  print "aperture" a;
  print "annulus" s;

  (* Subtract the annulus's mean per cell times the aperture's area. *)
  let flux = Quantity.(sub a.value (mul (div s.value s.area) a.area)) in
  Printf.printf "source flux = %.3f e/s (true %.0f, less the dead pixel)\n"
    (Nx.item [] (Quantity.value e_s flux))
    source_total;
  Option.iter
    (fun v ->
      Printf.printf "aperture sum error = %.3f e/s\n"
        (sqrt (Nx.item [] (Quantity.value Unit.(e_s ** 2) v))))
    a.variance;

  (* A window of static shape about a point reads the same cells, so the sum
     does not change; one that clips the aperture raises. *)
  let stamp = Observation.around (point 30.3 33.6) ~shape:[| 24; 24 |] obs in
  print "window" (Observation.integrate aperture stamp);
  (try
     ignore
       (Observation.integrate aperture
          (Observation.around (point 30.3 33.6) ~shape:[| 8; 8 |] obs))
   with Invalid_argument msg -> print_endline msg);

  (* An aperture over the image's edge reports the share it covers. *)
  print "edge"
    (Observation.integrate
       (Region.circle (Transform.shift (point 0. 20.)) ~radius:(size 4.))
       obs)
