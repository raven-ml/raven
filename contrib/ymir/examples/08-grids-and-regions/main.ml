(* Grids and regions.

   A grid is an image's cells seen through a transform to its world: a plane, or
   the sky. A region is a shape placed on that world, and weighs each cell by
   the exact fraction of its area inside the shape. *)

open Ymir

let f64 = Nx.float64
let sum x = Nx.item [] (Nx.sum x)

let () =
  (* A 9x9 grid whose world is its own pixel plane. *)
  let plane = Grid.pixels ~shape:[| 9; 9 |] f64 Transform.id in
  Format.printf "cell measures: %a@." Quantity.pp
    (Quantity.map (Nx.slice [ Nx.I 0; Nx.R (0, 3) ]) (Grid.measure plane));

  (* A disc of radius 2.5 cells about pixel (4.2, 3.9). The weights sum to its
     area, pi r^2, because the overlap is exact. *)
  let at = Quantity.v Unit.one (Nx.create f64 [| 2 |] [| 4.2; 3.9 |]) in
  let disc =
    Region.circle (Transform.shift at)
      ~radius:(Quantity.v Unit.one (Nx.scalar f64 2.5))
  in
  let w = Region.weights disc plane in
  print_endline "weights, in percent:";
  for i = 0 to 8 do
    for j = 0 to 8 do
      Printf.printf "%4.0f" (100. *. Nx.item [ i; j ] w)
    done;
    print_newline ()
  done;
  Printf.printf "sum of weights = %.12f, pi r^2 = %.12f\n" (sum w)
    (Float.pi *. 2.5 *. 2.5);

  (* An annulus weighs the outer disc less the inner one. *)
  let ring =
    Region.annulus (Transform.shift at)
      ~inner:(Quantity.v Unit.one (Nx.scalar f64 1.))
      ~outer:(Quantity.v Unit.one (Nx.scalar f64 2.))
  in
  Printf.printf "annulus area = %.12f, expected %.12f\n"
    (sum (Region.weights ring plane))
    (Float.pi *. ((2. *. 2.) -. (1. *. 1.)));

  (* The same on the sky: 0.1 arcsecond pixels about a tangent point. Cell
     measures are solid angles, and a circle placed with [Transform.about] is
     the cap of that angular radius. *)
  let centre =
    Direction.lonlat Frame.icrs
      ~lon:(Quantity.v Unit.degree (Nx.scalar f64 30.))
      ~lat:(Quantity.v Unit.degree (Nx.scalar f64 60.))
  in
  let cd =
    Quantity.v Unit.arcsecond (Nx.create f64 [| 2; 2 |] [| -0.1; 0.; 0.; 0.1 |])
  in
  let to_sky =
    Transform.(
      axes [| 1; 0 |] ~origin:0
      >> shift (Quantity.v Unit.one (Nx.create f64 [| 2 |] [| 15.; 15. |]))
      >> linear cd
      >> inverse (gnomonic centre))
  in
  let sky = Grid.pixels ~shape:[| 31; 31 |] f64 to_sky in
  let arcsec2 = Unit.(arcsecond ** 2) in
  Printf.printf "centre cell = %.12f arcsec^2\n"
    (Nx.item [ 15; 15 ] (Quantity.value arcsec2 (Grid.measure sky)));
  let cap =
    Region.circle (Transform.about centre)
      ~radius:(Quantity.v Unit.arcsecond (Nx.scalar f64 1.))
  in
  let w = Region.weights cap sky in
  let area = Nx.mul w (Quantity.value arcsec2 (Grid.measure sky)) in
  Printf.printf "cap area = %.9f arcsec^2, pi (1\")^2 = %.9f\n" (sum area)
    Float.pi
