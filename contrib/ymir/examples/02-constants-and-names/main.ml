(* Constants and names.

   The SI fixes some constants exactly, so they are units. A measured constant
   belongs to a CODATA release the program names, and rounds once to the dtype
   it is asked for. A unit has no name of its own: a vocabulary spells it. *)

open Ymir

let () =
  (* Exact constants are units: c, h, k, e and their products. *)
  let hc = Unit.(planck * speed_of_light) in
  Printf.printf "h c        = %s\n" (Unit.to_string hc);
  Printf.printf "h c in eV m: %.10g\n"
    (Unit.ratio Nx.float64 hc Unit.(electronvolt * metre));

  (* Measured constants come from a release. *)
  let g = Codata.newtonian_gravitation Codata.v2022 in
  Format.printf "%a@." Constant.pp g;
  Format.printf "G (float32) = %a@." Quantity.pp
    (Constant.quantity Nx.float32 g);
  Format.printf "G (float64) = %a +/- %a@." Quantity.pp
    (Constant.quantity Nx.float64 g)
    Quantity.pp
    (Constant.uncertainty Nx.float64 g);

  (* A constant of your own, in the published notation. *)
  let msun =
    Constant.v ~name:"nominal solar mass parameter" "1.3271244e20"
      Unit.((metre ** 3) / (second ** 2))
  in
  Format.printf "%a@." Constant.pp msun;

  (* Vocabularies spell units with symbols. The SI's knows the derived units and
     prefixes. *)
  let pp_si = Vocabulary.pp Vocabulary.si in
  Format.printf "kg m^2 s^-2      -> %a@." pp_si
    Unit.(kilogram * (metre ** 2) / (second ** 2));
  Format.printf "1e-3 kg          -> %a@." pp_si Unit.gram;
  Format.printf "1e3 J mol^-1     -> %a@." pp_si Unit.(kilo joule / mole);

  (* Add symbols of your own, and read prefixed symbols back. *)
  let jansky = Unit.(decimal "1e-26" * watt / (metre ** 2) / hertz) in
  let voc = Vocabulary.(union (v [ ("Jy", Prefixable, jansky) ]) si) in
  let brightness = Unit.(mega jansky / steradian) in
  Format.printf "MJy/sr in SI     -> %s@." (Unit.to_string brightness);
  Format.printf "MJy/sr spelled   -> %a@." (Vocabulary.pp voc) brightness;
  match Vocabulary.lookup voc "mJy" with
  | Some u ->
      Printf.printf "mJy in Jy        -> %g\n" (Unit.ratio Nx.float64 u jansky)
  | None -> print_endline "mJy is not in the vocabulary"
