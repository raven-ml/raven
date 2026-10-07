(* Units and quantities.

   A unit is an exact value: kilometres per second is 1000 m s^-1 with no
   rounding, and its canonical text is its identity. A quantity is a tensor in a
   unit, and a number leaves it only through a named unit. *)

open Ymir

let () =
  (* Build units from the SI's units, prefixes and the algebra. *)
  let km_s = Unit.(kilo metre / second) in
  let parsec_per_year = Unit.(Units.parsec / Units.julian_year) in
  Printf.printf "km/s       = %s\n" (Unit.to_string km_s);
  Printf.printf "degree     = %s\n" (Unit.to_string Unit.degree);
  Printf.printf "pc/yr      = %s\n" (Unit.to_string parsec_per_year);

  (* Two spellings of one unit are one value. *)
  Printf.printf "km = 1000 m: %b\n"
    (Unit.equal (Unit.kilo Unit.metre) Unit.(int 1000 * metre));

  (* [ratio] rounds the exact conversion factor once, to the dtype asked for. *)
  Printf.printf "pc/yr in km/s: %.10g\n"
    (Unit.ratio Nx.float64 parsec_per_year km_s);

  (* Units that don't convert raise, naming what their quotient keeps. *)
  (try ignore (Unit.ratio Nx.float64 km_s Unit.second)
   with Invalid_argument msg -> Printf.printf "%s\n" msg);

  (* A quantity is a tensor in a unit. [value] reads it in another unit with one
     rounded multiply. *)
  let speeds =
    Quantity.v km_s (Nx.create Nx.float32 [| 3 |] [| 1.; 2.5; 30. |])
  in
  Format.printf "speeds     = %a@." Quantity.pp speeds;
  Format.printf "in m/s     = %a@." Nx.pp
    (Quantity.value Unit.(metre / second) speeds);

  (* Arithmetic computes the unit, and [add] converts its second argument to the
     first's unit. *)
  let time = Quantity.v Unit.hour (Nx.scalar Nx.float32 2.) in
  let distance = Quantity.mul speeds time in
  Format.printf "distance   = %a@." Quantity.pp distance;
  Format.printf "in km      = %a@." Nx.pp
    (Quantity.value (Unit.kilo Unit.metre) distance);
  let total =
    Quantity.add
      (Quantity.v (Unit.kilo Unit.metre) (Nx.scalar Nx.float64 1.))
      (Quantity.v Unit.metre (Nx.scalar Nx.float64 250.))
  in
  Format.printf "1 km + 250 m = %a@." Quantity.pp total
