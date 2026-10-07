(* Frames and directions.

   A direction is a batch of unit vectors in a celestial frame. The frame is
   part of the type: comparing an ICRS direction with a Galactic one is a type
   error until one is rotated into the other's frame. *)

open Ymir

let deg x = Quantity.v Unit.degree (Nx.create Nx.float64 [| Array.length x |] x)
let in_deg q = Nx.to_array (Quantity.value Unit.degree q)

let print_lonlat name d =
  let lon = in_deg (Direction.lon d) and lat = in_deg (Direction.lat d) in
  Array.iteri
    (fun i l -> Printf.printf "  %s %d: (%9.5f, %+9.5f) deg\n" name i l lat.(i))
    lon

let () =
  (* Three directions in ICRS, built from longitudes and latitudes. *)
  let stars =
    Direction.lonlat Frame.icrs
      ~lon:(deg [| 10.68470; 83.82208; 266.41683 |])
      ~lat:(deg [| 41.26875; -5.39111; -29.00781 |])
  in
  Printf.printf "In %s:\n" (Frame.name Frame.icrs);
  print_lonlat "star" stars;

  (* Rotate them into the Galactic frame. *)
  let galactic = Direction.rotate Frame.galactic stars in
  Printf.printf "In %s:\n" (Frame.name Frame.galactic);
  print_lonlat "star" galactic;

  (* Separations and bearings broadcast over batch axes: one reference direction
     against the three stars. *)
  let pole =
    Direction.lonlat Frame.icrs ~lon:(deg [| 0. |]) ~lat:(deg [| 90. |])
  in
  Printf.printf "Separation from the north pole (deg): %s\n"
    (String.concat ", "
       (Array.to_list
          (Array.map (Printf.sprintf "%.5f")
             (in_deg (Direction.separation pole stars)))));

  let a =
    Direction.lonlat Frame.icrs ~lon:(deg [| 10. |]) ~lat:(deg [| 0. |])
  in
  let b =
    Direction.lonlat Frame.icrs ~lon:(deg [| 10. |]) ~lat:(deg [| 1. |])
  in
  let c =
    Direction.lonlat Frame.icrs ~lon:(deg [| 11. |]) ~lat:(deg [| 0. |])
  in
  Printf.printf "Position angle of a point due north: %.3f deg\n"
    (in_deg (Direction.position_angle a b)).(0);
  Printf.printf "Position angle of a point due east:  %.3f deg\n"
    (in_deg (Direction.position_angle a c)).(0);

  (* [Frame.matrix] gives the rotation itself, for vectors that are not
     directions, such as velocities. *)
  Format.printf "ICRS -> Galactic:@.%a@." Nx.pp
    (Frame.matrix Frame.icrs Frame.galactic)
