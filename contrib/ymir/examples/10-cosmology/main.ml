(* Background cosmology.

   A cosmology is a record of tensors. Distances, volumes and times are
   functions of it and of redshifts, and its leaves broadcast with the
   redshifts, so one call evaluates many models at once. *)

open Ymir

let f64 = Nx.float64
let mpc = Unit.mega Units.parsec
let gyr = Unit.giga Units.julian_year

let row x =
  String.concat " " (List.map (Printf.sprintf "%10.4f") (Array.to_list x))

let () =
  let planck = Cosmology.planck2018 ~codata:Codata.v2022 f64 in
  Format.printf "%a@." Cosmology.pp planck;

  let z = Nx.create f64 [| 4 |] [| 0.1; 0.5; 1.; 3. |] in
  Printf.printf "z                     %s\n" (row (Nx.to_array z));
  let show name unit q =
    Printf.printf "%-21s %s\n" name (row (Nx.to_array (Quantity.value unit q)))
  in
  show "comoving (Mpc)" mpc (Cosmology.comoving_distance planck z);
  show "luminosity (Mpc)" mpc (Cosmology.luminosity_distance planck z);
  show "ang. diameter (Mpc)" mpc (Cosmology.angular_diameter_distance planck z);
  show "lookback (Gyr)" gyr (Cosmology.lookback_time planck z);
  show "age (Gyr)" gyr (Cosmology.age planck z);
  Printf.printf "%-21s %s\n" "distance modulus"
    (row (Nx.to_array (Cosmology.distance_modulus planck z)));

  (* Density parameters of each component sum to 1 at every redshift. *)
  let omega c = Cosmology.density_parameter planck c z in
  Printf.printf "%-21s %s\n" "Omega matter"
    (row (Nx.to_array (omega Cold_matter)));
  Printf.printf "%-21s %s\n" "Omega dark energy"
    (row (Nx.to_array (omega Dark_energy)));

  (* Other models are record updates: here dark energy with w0 = -0.9, and a
     curved universe. *)
  let wcdm = { planck with w0 = Nx.scalar f64 (-0.9) } in
  let curved = { planck with omega_k = Nx.scalar f64 0.05 } in
  show "wCDM comoving (Mpc)" mpc (Cosmology.comoving_distance wcdm z);
  show "curved comoving (Mpc)" mpc (Cosmology.comoving_distance curved z);

  (* Batching: three values of w0 as leaves of shape [3; 1] against four
     redshifts give a [3; 4] result in one call. *)
  let w0s = Nx.create f64 [| 3; 1 |] [| -1.1; -1.; -0.9 |] in
  let lanes = { planck with w0 = w0s } in
  let d = Quantity.value mpc (Cosmology.comoving_distance lanes z) in
  Printf.printf "comoving distance by w0, shape [%s]:\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Nx.shape d))));
  Array.iteri
    (fun i w ->
      Printf.printf "  w0 = %4.1f  %s\n" w
        (row (Nx.to_array (Nx.slice [ Nx.I i ] d))))
    (Nx.to_array w0s)
