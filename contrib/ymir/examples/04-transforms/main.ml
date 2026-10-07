(* Transforms.

   A transform is a list of stages from one kind of point to another: pixel
   coordinates to a plane, a plane to directions on the sky. Stages compose with
   [>>], and [inverse] runs them backwards. *)

open Ymir

let f64 = Nx.float64
let one x = Quantity.v Unit.one (Nx.create f64 [| Array.length x |] x)

let () =
  (* The tangent point of the projection. *)
  let centre =
    Direction.lonlat Frame.icrs
      ~lon:(Quantity.v Unit.degree (Nx.scalar f64 150.))
      ~lat:(Quantity.v Unit.degree (Nx.scalar f64 2.))
  in

  (* A 100x100 image with 1 arcsecond pixels and north up, east left: pixel
     (row, column) -> (x, y) about pixel (49.5, 49.5) -> arcseconds east and
     north -> the sky through a gnomonic (TAN) projection. *)
  let arcsec = Unit.arcsecond in
  let cd = Quantity.v arcsec (Nx.create f64 [| 2; 2 |] [| -1.; 0.; 0.; 1. |]) in
  let pixel_to_sky =
    Transform.(
      axes [| 1; 0 |] ~origin:0
      >> shift (one [| 49.5; 49.5 |])
      >> linear cd
      >> inverse (gnomonic centre))
  in
  Format.printf "%a@." Transform.pp pixel_to_sky;

  (* Apply it to a batch of pixels, [n; 2] in (row, column) order. *)
  let pixels =
    Quantity.v Unit.one
      (Nx.create f64 [| 3; 2 |] [| 49.5; 49.5; 0.; 0.; 49.5; 99. |])
  in
  let sky = Transform.apply pixel_to_sky pixels in
  let lon = Nx.to_array (Quantity.value Unit.degree (Direction.lon sky)) in
  let lat = Nx.to_array (Quantity.value Unit.degree (Direction.lat sky)) in
  Array.iteri
    (fun i l -> Printf.printf "pixel %d -> (%.6f, %+.6f) deg\n" i l lat.(i))
    lon;

  (* The inverse maps the sky back to pixels. *)
  let back = Transform.apply (Transform.inverse pixel_to_sky) sky in
  let back = Nx.to_array (Quantity.value Unit.one back) in
  for i = 0 to 2 do
    Printf.printf "back to pixel %d: (%.9f, %.9f)\n" i
      back.(2 * i)
      back.((2 * i) + 1)
  done;

  (* A gnomonic projection covers only the hemisphere about its centre. [covers]
     says where [apply] is defined; [apply] raises elsewhere. *)
  let far =
    Direction.lonlat Frame.icrs
      ~lon:(Quantity.v Unit.degree (Nx.create f64 [| 2 |] [| 151.; 330. |]))
      ~lat:(Quantity.v Unit.degree (Nx.create f64 [| 2 |] [| 2.; -2. |]))
  in
  let to_pixels = Transform.inverse pixel_to_sky in
  Format.printf "covered: %a@." Nx.pp (Transform.covers to_pixels far);
  try ignore (Transform.apply to_pixels far)
  with Invalid_argument msg -> print_endline msg
