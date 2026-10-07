(* Bijectors.

   A bijector maps unconstrained coordinates, any real numbers, onto a support,
   and reports the log-determinant of its Jacobian. Samplers move in
   coordinates; a density over values is pulled back to them by adding the
   log-determinant. *)

module B = Norn.Bij
module D = Norn.Dist

let f64 = Nx.float64
let vec xs = Nx.create f64 [| Array.length xs |] xs
let show name x = Format.printf "%-30s %a@." name Nx.pp x

let () =
  let u = vec [| -3.; 0.; 2. |] in
  show "coordinates" u;

  (* Elementwise maps: one log-determinant per element. *)
  let x, ld = B.forward B.exp u in
  show "exp" x;
  show "  log |det J|" ld;
  let x, _ =
    B.forward (B.interval ~low:(Nx.scalar f64 0.) ~high:(Nx.scalar f64 10.)) u
  in
  show "interval (0, 10)" x;
  show "  back to coordinates"
    (B.inverse (B.interval ~low:(Nx.scalar f64 0.) ~high:(Nx.scalar f64 10.)) x);

  (* Vector maps: two coordinates make three positive components summing to one;
     one log-determinant per vector. *)
  let x, ld = B.forward B.simplex (vec [| 0.5; -1. |]) in
  show "simplex" x;
  show "  log |det J|" ld;
  let x, _ = B.forward B.ordered u in
  show "ordered" x;
  Printf.printf "%-30s %d\n" "cholesky_corr coordinates, 3x3"
    (B.shape B.cholesky_corr [| 3; 3 |]).(0);

  (* Pulling a density back: in coordinates u = log x, the density of a gamma
     becomes log p(exp u) + u, a density over the whole line. *)
  let gamma =
    D.gamma ~concentration:(Nx.scalar f64 2.) ~rate:(Nx.scalar f64 1.)
  in
  let pulled u =
    let x, ld = B.forward (D.coords gamma) u in
    Nx.add (D.factors gamma x) ld
  in
  show "gamma pulled back at u" (pulled u);

  (* A distribution of a bijector's image: lognormal as exp of a normal. *)
  let normal = D.normal ~loc:(Nx.scalar f64 0.) ~scale:(Nx.scalar f64 1.) in
  let at = Nx.scalar f64 2. in
  show "transform exp normal at 2" (D.log_density (D.transform B.exp normal) at);
  show "lognormal at 2"
    (D.log_density
       (D.lognormal ~loc:(Nx.scalar f64 0.) ~scale:(Nx.scalar f64 1.))
       at)
