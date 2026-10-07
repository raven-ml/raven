(* Derivatives and compilation.

   Every ymir function is a formula of nx operations, so rune differentiates and
   compiles it with no rule of its own: a distance in a cosmological parameter,
   an aperture sum in the aperture's centre and radius. *)

open Ymir

let f64 = Nx.float64
let planck = Cosmology.planck2018 ~codata:Codata.v2022 f64
let z = Nx.create f64 [| 3 |] [| 0.1; 1.; 3. |]

(* The distance modulus at [z] as a function of theta = (omega_cb, w0). *)
let modulus theta =
  let c =
    {
      planck with
      omega_cb = Nx.slice [ Nx.I 0 ] theta;
      w0 = Nx.slice [ Nx.I 1 ] theta;
    }
  in
  Cosmology.distance_modulus c z

let print name x =
  Printf.printf "%-24s %s\n" name
    (String.concat " "
       (List.map (Printf.sprintf "%9.5f") (Array.to_list (Nx.to_array x))))

let cosmology () =
  let theta = Nx.create f64 [| 2 |] [| 0.31; -1. |] in
  print "mu(z)" (modulus theta);
  (* One gradient per redshift, each a [2] vector of d mu / d theta. *)
  for i = 0 to 2 do
    let g = Rune.grad' (fun t -> Nx.slice [ Nx.I i ] (modulus t)) theta in
    print (Printf.sprintf "d mu(%g) / d(Om, w0)" (Nx.item [ i ] z)) g
  done;
  (* The same function compiled. *)
  let compiled = Rune.jit' modulus in
  print "mu(z), compiled" (compiled theta)

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
  cosmology ();
  photometry ()
