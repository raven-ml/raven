(* Derivatives and compilation.

   Every ymir function is a formula of nx operations, so rune differentiates and
   compiles it with no rule of its own: a distance in every field of a
   cosmology, an aperture sum in the aperture's centre and radius. *)

open Ymir

let f64 = Nx.float64
let planck = Cosmology.planck2018 ~codata:Codata.v2022 f64
let z = Nx.create f64 [| 3 |] [| 0.1; 1.; 3. |]
let cosmology = Nx.Ptree.instantiate (module Cosmology)
let kms_mpc = Unit.(kilo metre / second / mega Units.parsec)
let mu c = Cosmology.distance_modulus c z

let print name x =
  Printf.printf "%-28s %s\n" name
    (String.concat " "
       (List.map (Printf.sprintf "%9.5f") (Array.to_list (Nx.to_array x))))

let derivatives () =
  print "mu(z)" (mu planck);
  (* One gradient per redshift: a cosmology whose fields are d mu / d field,
     each per the unit planck holds it in. *)
  for i = 0 to 2 do
    let g = Rune.grad cosmology (fun c -> Nx.slice [ Nx.I i ] (mu c)) planck in
    Printf.printf "z = %g\n" (Nx.item [ i ] z);
    print "  d mu / d Om_cb" g.omega_cb;
    print "  d mu / d w0" g.w0;
    print "  d mu / d H0, per km/s/Mpc" (Quantity.value kms_mpc g.h0);
    print "  d mu / d m_nu, per eV" (Quantity.value Unit.electronvolt g.m_nu)
  done;
  (* Forward mode: mu's change per km/s/Mpc of H0, at every z at once. *)
  let zero = Nx.Ptree.map cosmology (fun _ x -> Nx.zeros_like x) planck in
  let dh0 = { zero with h0 = Quantity.v kms_mpc (Nx.scalar f64 1.) } in
  print "d mu / d H0, forward"
    (snd (Rune.jvp cosmology Nx.Ptree.tensor mu planck dh0));
  print "mu(z), compiled"
    (Rune.jit Nx.Ptree.(cosmology @-> returns tensor) mu planck)

(* A Gaussian source of total 100 at (15.2, 14.7) on a 32x32 grid. *)
let image =
  Nx.init f64 [| 32; 32 |] (fun i ->
      let r = float_of_int i.(0) -. 15.2 and c = float_of_int i.(1) -. 14.7 in
      100. *. exp (-.((r *. r) +. (c *. c)) /. 8.) /. (8. *. Float.pi))

let obs =
  let grid = Grid.pixels ~shape:[| 32; 32 |] f64 Transform.id in
  Observation.v grid (Quantity.v Unit.(one / Grid.cell) image)

(* The sum in a circle as a function of p = (row, column, radius). *)
let aperture p =
  let centre = Quantity.v Unit.one (Nx.slice [ Nx.R (0, 2) ] p) in
  let radius = Quantity.v Unit.one (Nx.slice [ Nx.I 2 ] p) in
  let region = Region.circle (Transform.shift centre) ~radius in
  Quantity.value Unit.one (Observation.integrate region obs).value

let photometry () =
  (* Off the source's centre, the sum grows toward it; the radius derivative is
     the image integrated along the circle's edge. *)
  let p = Nx.create f64 [| 3 |] [| 14.; 16.; 4. |] in
  print "aperture sum" (aperture p);
  print "d sum / d(row, col, r)" (Rune.grad' aperture p)

let () =
  derivatives ();
  photometry ()
