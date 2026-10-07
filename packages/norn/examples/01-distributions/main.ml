(* Distributions.

   A distribution describes one whole tensor value: its log density is a scalar.
   Parameters are tensors and broadcast, so one normal with a vector of
   locations is independent normals. Each family knows its support and the
   bijector that maps unconstrained coordinates onto it. *)

module D = Norn.Dist

let f64 = Nx.float64
let s x = Nx.scalar f64 x
let vec xs = Nx.create f64 [| Array.length xs |] xs
let show name x = Format.printf "%-28s %a@." name Nx.pp x

let () =
  let d = D.normal ~loc:(s 0.) ~scale:(s 2.) in
  Format.printf "%a@." D.pp d;
  show "log density at 1" (D.log_density d (s 1.));
  show "quantiles 5%, 50%, 95%" (D.quantile d (vec [| 0.05; 0.5; 0.95 |]));

  (* Broadcasting parameters: three normals, one log density per element with
     [factors], their sum with [log_density]. *)
  let three = D.normal ~loc:(vec [| -1.; 0.; 1. |]) ~scale:(s 1.) in
  show "factors at 0" (D.factors three (Nx.zeros f64 [| 3 |]));
  show "log density at 0" (D.log_density three (Nx.zeros f64 [| 3 |]));

  (* [iid] adds independent copies in front of the value's shape. *)
  let rows = D.iid [| 4 |] three in
  Printf.printf "%-28s [%s]\n" "shape of iid [4] three"
    (String.concat "; "
       (Array.to_list (Array.map string_of_int (D.shape rows))));

  (* Draws from a key: the same key gives the same draw. *)
  let key = Nx.Rng.key 7 in
  let x =
    D.sample key
      (D.iid [| 10000 |] (D.gamma ~concentration:(s 3.) ~rate:(s 2.)))
  in
  Printf.printf "%-28s mean %.3f (exact 1.5), var %.3f (exact 0.75)\n"
    "gamma(3, 2), 10000 draws"
    (Nx.item [] (Nx.mean x))
    (Nx.item [] (Nx.var x));

  (* Discrete and vector families. *)
  let counts = D.sample key (D.iid [| 5 |] (D.poisson ~rate:(s 3.))) in
  Printf.printf "%-28s %s\n" "poisson(3) draws"
    (String.concat " "
       (Array.to_list (Array.map Int32.to_string (Nx.to_array counts))));
  let p = D.sample key (D.dirichlet ~concentration:(vec [| 1.; 2.; 3. |])) in
  show "dirichlet draw" p;
  show "  sums to" (Nx.sum p);
  show "categorical log p of 2"
    (D.log_density
       (D.categorical ~logits:(vec [| 0.; 1.; 2. |]))
       (Nx.scalar Nx.int64 2L));

  (* A value outside the support has density -inf; a parameter outside its
     domain raises when the distribution is used. *)
  show "exponential at -1"
    (D.log_density (D.exponential ~rate:(s 1.)) (s (-1.)));
  (try ignore (D.log_density (D.normal ~loc:(s 0.) ~scale:(s (-1.))) (s 0.))
   with Invalid_argument msg -> print_endline msg);

  (* Supports and the bijectors samplers move through. *)
  List.iter
    (fun (name, d) ->
      Format.printf "%-28s support %a, coordinates %a@." name Norn.Support.pp
        (D.support d) Norn.Bij.pp (D.coords d))
    [
      ("normal", D.normal ~loc:(s 0.) ~scale:(s 1.));
      ("gamma", D.gamma ~concentration:(s 2.) ~rate:(s 1.));
      ("beta", D.beta ~a:(s 2.) ~b:(s 2.));
      ("dirichlet", D.dirichlet ~concentration:(vec [| 1.; 1.; 1. |]));
    ]
